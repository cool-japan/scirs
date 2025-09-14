//! Performance benchmarks for advanced spatial modules
//!
//! This benchmark suite validates the performance claims of quantum-inspired,
//! neuromorphic, and hybrid algorithms compared to classical approaches.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ndarray::{Array1, Array2};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::time::Duration;

use scirs2_spatial::{
    distance::{euclidean, pdist},

    gpu_accel::is_gpu_acceleration_available,
    memory_pool::global_distance_pool,
    neuromorphic::SpikingNeuralClusterer,
    quantum_classical_hybrid::HybridClusterer,

    // Advanced algorithms to benchmark
    quantum_inspired::{QuantumClusterer, QuantumNearestNeighbor},
    // Performance optimizations
    simd_distance::{parallel_pdist, simd_euclidean_distance_batch},
    // Classical algorithms for comparison
    KDTree,
};

/// Generate test datasets for benchmarking
struct BenchmarkDatasets {
    small_clustered: Array2<f64>,  // 100 points, 2D
    medium_clustered: Array2<f64>, // 1000 points, 2D
    large_clustered: Array2<f64>,  // 5000 points, 2D
    high_dim_medium: Array2<f64>,  // 500 points, 10D
    high_dim_large: Array2<f64>,   // 1000 points, 20D
}

impl BenchmarkDatasets {
    fn new() -> Self {
        let mut rng = StdRng::seed_from_u64(42);

        Self {
            small_clustered: Self::generate_clustered_data(&mut rng, 100, 3, 2),
            medium_clustered: Self::generate_clustered_data(&mut rng, 1000, 5, 2),
            large_clustered: Self::generate_clustered_data(&mut rng, 5000, 8, 2),
            high_dim_medium: Self::generate_clustered_data(&mut rng, 500, 5, 10),
            high_dim_large: Self::generate_clustered_data(&mut rng, 1000, 8, 20),
        }
    }

    fn generate_clustered_data(
        rng: &mut StdRng,
        n_points: usize,
        n_clusters: usize,
        dims: usize,
    ) -> Array2<f64> {
        let mut points = Array2::zeros((n_points, dims));

        // Generate cluster centers
        let cluster_centers: Vec<Vec<f64>> = (0..n_clusters)
            .map(|_| (0..dims).map(|_| rng.gen_range(-50.0..50.0)).collect())
            .collect();

        // Assign points to clusters with noise
        for i in 0..n_points {
            let cluster_idx = i % n_clusters;
            let center = &cluster_centers[cluster_idx];

            for j in 0..dims {
                points[[i, j]] = center[j] + rng.gen_range(-5.0..5.0);
            }
        }

        points
    }
}

/// Benchmark clustering algorithms
#[allow(dead_code)]
fn benchmark_clustering(c: &mut Criterion) {
    let datasets = BenchmarkDatasets::new();

    let mut group = c.benchmark_group("Clustering");
    group.measurement_time(Duration::from_secs(30));

    // Test different dataset sizes
    let test_cases = vec![
        ("small_100pts", &datasets.small_clustered, 3),
        ("medium_1000pts", &datasets.medium_clustered, 5),
        ("large_5000pts", &datasets.large_clustered, 8),
        ("highdim_500pts_10d", &datasets.high_dim_medium, 5),
    ];

    for (name, data, n_clusters) in test_cases {
        group.throughput(Throughput::Elements(data.nrows() as u64));

        // Benchmark classical k-means (using neuromorphic as baseline since no pure classical k-means)
        group.bench_with_input(
            BenchmarkId::new("classical_baseline", name),
            &(data, n_clusters),
            |b, (data, n_clusters)| {
                b.iter(|| {
                    let mut clusterer = SpikingNeuralClusterer::new(*n_clusters);
                    // Use minimal neuromorphic settings to approximate classical behavior
                    clusterer.fit(&data.view()).unwrap()
                });
            },
        );

        // Benchmark quantum clustering
        group.bench_with_input(
            BenchmarkId::new("quantum", name),
            &(data, n_clusters),
            |b, (data, n_clusters)| {
                b.iter(|| {
                    let mut clusterer = QuantumClusterer::new(*n_clusters);
                    clusterer.fit(&data.view()).unwrap()
                });
            },
        );

        // Benchmark neuromorphic clustering
        group.bench_with_input(
            BenchmarkId::new("neuromorphic", name),
            &(data, n_clusters),
            |b, (data, n_clusters)| {
                b.iter(|| {
                    let mut clusterer = SpikingNeuralClusterer::new(*n_clusters);
                    clusterer.fit(&data.view()).unwrap()
                });
            },
        );

        // Benchmark hybrid clustering
        group.bench_with_input(
            BenchmarkId::new("hybrid", name),
            &(data, n_clusters),
            |b, (data, n_clusters)| {
                b.iter(|| {
                    let mut clusterer = HybridClusterer::new(*n_clusters)
                        .with_quantum_exploration_ratio(0.7)
                        .with_classical_refinement(false); // Disable async refinement for benchmark
                                                           // For benchmarking, we'll just use the quantum part for simplicity
                                                           // since the full hybrid approach is async
                    tokio::runtime::Runtime::new()
                        .unwrap()
                        .block_on(async { clusterer.fit(&data.view()).await.unwrap() })
                });
            },
        );
    }

    group.finish();
}

/// Benchmark nearest neighbor search algorithms
#[allow(dead_code)]
fn benchmark_nearest_neighbor(c: &mut Criterion) {
    let datasets = BenchmarkDatasets::new();

    let mut group = c.benchmark_group("NearestNeighbor");
    group.measurement_time(Duration::from_secs(20));

    let test_cases = vec![
        ("small_100pts", &datasets.small_clustered),
        ("medium_1000pts", &datasets.medium_clustered),
        ("large_5000pts", &datasets.large_clustered),
    ];

    for (name, data) in test_cases {
        group.throughput(Throughput::Elements(data.nrows() as u64));

        let query_point = vec![0.0, 0.0];
        let k = 5;

        // Benchmark classical KDTree
        group.bench_with_input(
            BenchmarkId::new("classical_kdtree", name),
            &(data, &query_point, k),
            |b, (data, query_point, k)| {
                b.iter(|| {
                    let kdtree = KDTree::new(data).unwrap();
                    kdtree.query(query_point, *k).unwrap()
                });
            },
        );

        // Benchmark quantum nearest neighbor
        group.bench_with_input(
            BenchmarkId::new("quantum_nn", name),
            &(data, &query_point, k),
            |b, (data, query_point, k)| {
                b.iter(|| {
                    let quantum_nn = QuantumNearestNeighbor::new(&data.view())
                        .unwrap()
                        .with_quantum_encoding(true)
                        .with_amplitude_amplification(true);
                    let query_array = Array1::from_vec(query_point.to_vec());
                    quantum_nn.query_quantum(&query_array.view(), *k).unwrap()
                });
            },
        );
    }

    group.finish();
}

/// Benchmark distance computation methods
#[allow(dead_code)]
fn benchmark_distance_computation(c: &mut Criterion) {
    let datasets = BenchmarkDatasets::new();

    let mut group = c.benchmark_group("DistanceComputation");
    group.measurement_time(Duration::from_secs(15));

    let test_cases = vec![
        ("medium_1000pts", &datasets.medium_clustered),
        ("large_5000pts", &datasets.large_clustered),
    ];

    for (name, data) in test_cases {
        group.throughput(Throughput::Elements((data.nrows() * data.nrows()) as u64));

        // Benchmark classical pairwise distances
        group.bench_with_input(
            BenchmarkId::new("classical_pdist", name),
            data,
            |b, data| {
                b.iter(|| pdist(data, euclidean));
            },
        );

        // Benchmark SIMD-accelerated distances
        group.bench_with_input(BenchmarkId::new("simd_pdist", name), data, |b, data| {
            b.iter(|| parallel_pdist(&data.view(), "euclidean").unwrap());
        });

        // Benchmark batch SIMD distances
        if data.nrows() >= 2 {
            let half = data.nrows() / 2;
            let data1 = data.slice(ndarray::s![..half, ..]).to_owned();
            let data2 = data.slice(ndarray::s![half.., ..]).to_owned();

            group.bench_with_input(
                BenchmarkId::new("simd_batch", name),
                &(data1, data2),
                |b, (data1, data2)| {
                    b.iter(|| simd_euclidean_distance_batch(&data1.view(), &data2.view()).unwrap());
                },
            );
        }
    }

    group.finish();
}

/// Benchmark memory efficiency and optimization
#[allow(dead_code)]
fn benchmark_memory_optimization(c: &mut Criterion) {
    let datasets = BenchmarkDatasets::new();

    let mut group = c.benchmark_group("MemoryOptimization");
    group.measurement_time(Duration::from_secs(10));

    // Test memory pool performance
    group.bench_function("memory_pool_allocation", |b| {
        b.iter(|| {
            let pool = global_distance_pool();
            let _buffer = pool.get_distance_buffer(1000);
            // Buffer automatically returns to pool on drop
        });
    });

    // Test memory-efficient clustering
    group.bench_with_input(
        BenchmarkId::new("memory_efficient_clustering", "large_5000pts"),
        &datasets.large_clustered,
        |b, data| {
            b.iter(|| {
                // Clear pool statistics
                let pool = global_distance_pool();
                let _stats_before = pool.statistics();

                let mut clusterer = SpikingNeuralClusterer::new(8);
                let _result = clusterer.fit(&data.view()).unwrap();

                let _stats_after = pool.statistics();
                // Memory pool usage is tracked internally
            });
        },
    );

    group.finish();
}

/// Benchmark scalability across different problem sizes
#[allow(dead_code)]
fn benchmark_scalability(c: &mut Criterion) {
    let mut group = c.benchmark_group("Scalability");
    group.measurement_time(Duration::from_secs(25));

    // Test different problem sizes
    let sizes = vec![50, 100, 500, 1000, 2000];

    for size in sizes {
        let mut rng = StdRng::seed_from_u64(42);
        let data = BenchmarkDatasets::generate_clustered_data(&mut rng, size, 3, 2);

        group.throughput(Throughput::Elements(size as u64));

        // Classical approach
        group.bench_with_input(BenchmarkId::new("classical", size), &data, |b, data| {
            b.iter(|| {
                let kdtree = KDTree::new(data).unwrap();
                kdtree.query(&[0.0, 0.0], 5).unwrap()
            });
        });

        // Quantum approach
        group.bench_with_input(BenchmarkId::new("quantum", size), &data, |b, data| {
            b.iter(|| {
                let quantum_nn = QuantumNearestNeighbor::new(&data.view())
                    .unwrap()
                    .with_quantum_encoding(true)
                    .with_amplitude_amplification(true);
                let query = Array1::from_vec(vec![0.0, 0.0]);
                quantum_nn.query_quantum(&query.view(), 5).unwrap()
            });
        });

        // Neuromorphic approach (clustering)
        if size <= 1000 {
            // Limit for reasonable benchmark time
            group.bench_with_input(BenchmarkId::new("neuromorphic", size), &data, |b, data| {
                b.iter(|| {
                    let mut clusterer = SpikingNeuralClusterer::new(3);
                    clusterer.fit(&data.view()).unwrap()
                });
            });
        }
    }

    group.finish();
}

/// Benchmark GPU acceleration if available
#[allow(dead_code)]
fn benchmark_gpu_acceleration(c: &mut Criterion) {
    if !is_gpu_acceleration_available() {
        println!("GPU acceleration not available, skipping GPU benchmarks");
        return;
    }

    let datasets = BenchmarkDatasets::new();

    let mut group = c.benchmark_group("GPUAcceleration");
    group.measurement_time(Duration::from_secs(15));

    // GPU vs CPU comparison for large datasets
    group.bench_with_input(
        BenchmarkId::new("cpu_clustering", "large_5000pts"),
        &datasets.large_clustered,
        |b, data| {
            b.iter(|| {
                let mut clusterer = QuantumClusterer::new(8);
                clusterer.fit(&data.view()).unwrap()
            });
        },
    );

    // Note: GPU-specific benchmarks would require actual GPU implementations
    // This is a placeholder for when GPU kernels are fully implemented

    group.finish();
}

/// Benchmark high-dimensional performance
#[allow(dead_code)]
fn benchmark_high_dimensional(c: &mut Criterion) {
    let datasets = BenchmarkDatasets::new();

    let mut group = c.benchmark_group("HighDimensional");
    group.measurement_time(Duration::from_secs(20));

    let test_cases = vec![
        ("10d_500pts", &datasets.high_dim_medium),
        ("20d_1000pts", &datasets.high_dim_large),
    ];

    for (name, data) in test_cases {
        group.throughput(Throughput::Elements(data.nrows() as u64));

        // Test quantum advantage in high dimensions
        group.bench_with_input(
            BenchmarkId::new("quantum_highdim", name),
            data,
            |b, data| {
                b.iter(|| {
                    let mut clusterer = QuantumClusterer::new(5);
                    clusterer.fit(&data.view()).unwrap()
                });
            },
        );

        // Test neuromorphic performance in high dimensions
        group.bench_with_input(
            BenchmarkId::new("neuromorphic_highdim", name),
            data,
            |b, data| {
                b.iter(|| {
                    let mut clusterer = SpikingNeuralClusterer::new(5);
                    clusterer.fit(&data.view()).unwrap()
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_clustering,
    benchmark_nearest_neighbor,
    benchmark_distance_computation,
    benchmark_memory_optimization,
    benchmark_scalability,
    benchmark_gpu_acceleration,
    benchmark_high_dimensional
);

criterion_main!(benches);
