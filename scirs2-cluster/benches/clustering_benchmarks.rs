//! Benchmarks for clustering algorithms
//!
//! This module provides comprehensive benchmarks for all clustering algorithms
//! to track performance regression and optimize implementations.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ndarray::{Array1, Array2};
use rand::prelude::*;
use std::time::Duration;

use scirs2_cluster::density::dbscan;
use scirs2_cluster::hierarchy::{linkage, LinkageMethod, Metric};
use scirs2_cluster::spectral::{spectral_clustering, AffinityMode, SpectralClusteringOptions};
use scirs2_cluster::vq::{kmeans2, MinitMethod, MissingMethod};

/// Generate synthetic clustered data for benchmarks
fn generate_clustered_data(
    n_samples: usize,
    n_features: usize,
    n_clusters: usize,
    noise: f64,
) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(42); // Fixed seed for reproducibility

    let cluster_size = n_samples / n_clusters;
    let mut data = Array2::zeros((n_samples, n_features));

    for cluster_id in 0..n_clusters {
        let start_idx = cluster_id * cluster_size;
        let end_idx = if cluster_id == n_clusters - 1 {
            n_samples // Handle remainder samples in last cluster
        } else {
            start_idx + cluster_size
        };

        // Create cluster center
        let mut center = Array1::zeros(n_features);
        for j in 0..n_features {
            center[j] = rng.random_range(-10.0..10.0);
        }

        // Generate points around cluster center
        for i in start_idx..end_idx {
            for j in 0..n_features {
                data[[i, j]] = center[j] + rng.random_range(-noise..noise);
            }
        }
    }

    data
}

/// Generate random data for worst-case benchmarks
fn generate_random_data(n_samples: usize, n_features: usize) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(42);
    let mut data = Array2::zeros((n_samples, n_features));

    for i in 0..n_samples {
        for j in 0..n_features {
            data[[i, j]] = rng.random_range(-1.0..1.0);
        }
    }

    data
}

/// Benchmark K-means clustering
fn bench_kmeans(c: &mut Criterion) {
    let mut group = c.benchmark_group("kmeans");

    // Test different data sizes
    let sizes = vec![100, 500, 1000, 2000];
    let n_features = 10;
    let k = 5;

    for &n_samples in &sizes {
        let data = generate_clustered_data(n_samples, n_features, k, 1.0);

        group.throughput(Throughput::Elements(n_samples as u64));
        group.bench_with_input(
            BenchmarkId::new("clustered_data", n_samples),
            &data,
            |b, data| {
                b.iter(|| {
                    kmeans2(
                        data.view(),
                        k,
                        Some(10), // iterations
                        None,     // threshold
                        Some(MinitMethod::Random),
                        Some(MissingMethod::Warn),
                        Some(false), // check_finite
                        Some(42),    // seed
                    )
                })
            },
        );

        // Also benchmark with different initialization methods
        group.bench_with_input(
            BenchmarkId::new("kmeans_plus_plus", n_samples),
            &data,
            |b, data| {
                b.iter(|| {
                    kmeans2(
                        data.view(),
                        k,
                        Some(10), // iterations
                        None,     // threshold
                        Some(MinitMethod::PlusPlus),
                        Some(MissingMethod::Warn),
                        Some(false), // check_finite
                        Some(42),    // seed
                    )
                })
            },
        );
    }

    group.finish();
}

/// Benchmark hierarchical clustering
fn bench_hierarchical(c: &mut Criterion) {
    let mut group = c.benchmark_group("hierarchical_clustering");
    group.measurement_time(Duration::from_secs(10)); // Longer measurement time for slower algorithms

    // Test smaller sizes for hierarchical clustering (O(n^3) complexity)
    let sizes = vec![50, 100, 200, 300];
    let n_features = 5;

    for &n_samples in &sizes {
        let data = generate_clustered_data(n_samples, n_features, 3, 1.0);

        group.throughput(Throughput::Elements(n_samples as u64));

        // Benchmark different linkage methods
        let methods = vec![
            LinkageMethod::Single,
            LinkageMethod::Complete,
            LinkageMethod::Average,
            LinkageMethod::Ward,
        ];

        for method in methods {
            group.bench_with_input(
                BenchmarkId::new(format!("{:?}", method), n_samples),
                &data,
                |b, data| b.iter(|| linkage(data.view(), method, Metric::Euclidean)),
            );
        }
    }

    group.finish();
}

/// Benchmark DBSCAN clustering
fn bench_dbscan(c: &mut Criterion) {
    let mut group = c.benchmark_group("dbscan");

    let sizes = vec![100, 500, 1000, 2000];
    let n_features = 10;

    for &n_samples in &sizes {
        let data = generate_clustered_data(n_samples, n_features, 5, 1.0);

        group.throughput(Throughput::Elements(n_samples as u64));
        group.bench_with_input(
            BenchmarkId::new("eps_0.5_min_5", n_samples),
            &data,
            |b, data| b.iter(|| dbscan(data.view(), 0.5, 5, None)),
        );

        // Test different parameter combinations
        group.bench_with_input(
            BenchmarkId::new("eps_1.0_min_3", n_samples),
            &data,
            |b, data| b.iter(|| dbscan(data.view(), 1.0, 3, None)),
        );
    }

    group.finish();
}

/// Benchmark spectral clustering
fn bench_spectral(c: &mut Criterion) {
    let mut group = c.benchmark_group("spectral_clustering");
    group.measurement_time(Duration::from_secs(15)); // Longer time for eigenvalue computations

    // Smaller sizes for spectral clustering (expensive eigenvalue computation)
    let sizes = vec![50, 100, 200];
    let n_features = 5;
    let k = 3;

    for &n_samples in &sizes {
        let data = generate_clustered_data(n_samples, n_features, k, 1.0);

        group.throughput(Throughput::Elements(n_samples as u64));

        let options = SpectralClusteringOptions {
            affinity: AffinityMode::RBF,
            n_neighbors: 10,
            gamma: 1.0,
            normalized_laplacian: true,
            max_iter: 20, // Reduced iterations for benchmarking
            n_init: 1,    // Single initialization for benchmarking
            tol: 1e-4,
            random_seed: Some(42),
            eigen_solver: "arpack".to_string(),
            auto_n_clusters: false,
        };

        group.bench_with_input(
            BenchmarkId::new("rbf_affinity", n_samples),
            &data,
            |b, data| b.iter(|| spectral_clustering(data.view(), k, Some(options.clone()))),
        );
    }

    group.finish();
}

/// Benchmark distance computation metrics
fn bench_distance_metrics(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance_metrics");

    let sizes = vec![100, 500, 1000];
    let n_features = 20;

    for &n_samples in &sizes {
        let data = generate_random_data(n_samples, n_features);

        group.throughput(Throughput::Elements(n_samples as u64));

        // Benchmark different distance metrics through hierarchical clustering
        let metrics = vec![
            Metric::Euclidean,
            Metric::Manhattan,
            Metric::Chebyshev,
            Metric::Correlation,
        ];

        for metric in metrics {
            group.bench_with_input(
                BenchmarkId::new(format!("{:?}", metric), n_samples),
                &data,
                |b, data| b.iter(|| linkage(data.view(), LinkageMethod::Single, metric)),
            );
        }
    }

    group.finish();
}

/// Benchmark cluster validation metrics
fn bench_validation_metrics(c: &mut Criterion) {
    use scirs2_cluster::hierarchy::{cophenet, validate_linkage_matrix};

    let mut group = c.benchmark_group("validation_metrics");

    let sizes = vec![50, 100, 200];
    let n_features = 5;

    for &n_samples in &sizes {
        let data = generate_clustered_data(n_samples, n_features, 3, 1.0);
        let linkage_matrix = linkage(data.view(), LinkageMethod::Ward, Metric::Euclidean).unwrap();

        // Create distance matrix for cophenetic correlation
        let mut distances = Array1::zeros(n_samples * (n_samples - 1) / 2);
        let mut idx = 0;
        for i in 0..n_samples {
            for j in (i + 1)..n_samples {
                let mut dist = 0.0;
                for k in 0..n_features {
                    let diff = data[[i, k]] - data[[j, k]];
                    dist += diff * diff;
                }
                distances[idx] = dist.sqrt();
                idx += 1;
            }
        }

        group.throughput(Throughput::Elements(n_samples as u64));

        // Benchmark cophenetic correlation
        group.bench_with_input(
            BenchmarkId::new("cophenetic_correlation", n_samples),
            &(&linkage_matrix, &distances),
            |b, (linkage, dist)| b.iter(|| cophenet(linkage, dist)),
        );

        // Benchmark linkage validation
        group.bench_with_input(
            BenchmarkId::new("linkage_validation", n_samples),
            &linkage_matrix,
            |b, linkage| b.iter(|| validate_linkage_matrix(linkage.view(), n_samples)),
        );
    }

    group.finish();
}

/// Benchmark data structure operations
fn bench_data_structures(c: &mut Criterion) {
    use scirs2_cluster::hierarchy::{condensed_to_square, square_to_condensed, DisjointSet};

    let mut group = c.benchmark_group("data_structures");

    // Benchmark DisjointSet operations
    let sizes = vec![100, 500, 1000, 5000];

    for &n_elements in &sizes {
        group.throughput(Throughput::Elements(n_elements as u64));

        group.bench_with_input(
            BenchmarkId::new("disjoint_set_union_find", n_elements),
            &n_elements,
            |b, &n| {
                b.iter(|| {
                    let mut ds = DisjointSet::new();

                    // Initialize all elements
                    for i in 0..n {
                        ds.make_set(i);
                    }

                    // Perform random unions
                    let mut rng = StdRng::seed_from_u64(42);
                    for _ in 0..(n / 2) {
                        let i = rng.random_range(0..n);
                        let j = rng.random_range(0..n);
                        ds.union(i, j);
                    }

                    // Perform random finds
                    for _ in 0..n {
                        let i = rng.random_range(0..n);
                        ds.find(&i);
                    }
                })
            },
        );
    }

    // Benchmark condensed matrix operations
    for &n_points in &[10, 20, 50, 100] {
        let condensed_size = n_points * (n_points - 1) / 2;
        let condensed = Array1::from_iter((0..condensed_size).map(|i| i as f64));

        group.throughput(Throughput::Elements(n_points as u64));

        group.bench_with_input(
            BenchmarkId::new("condensed_to_square", n_points),
            &condensed,
            |b, condensed| b.iter(|| condensed_to_square(condensed.view())),
        );

        let square = condensed_to_square(condensed.view()).unwrap();
        group.bench_with_input(
            BenchmarkId::new("square_to_condensed", n_points),
            &square,
            |b, square| b.iter(|| square_to_condensed(square.view())),
        );
    }

    group.finish();
}

/// Benchmark memory usage and scalability
fn bench_scalability(c: &mut Criterion) {
    let mut group = c.benchmark_group("scalability");
    group.measurement_time(Duration::from_secs(20));
    group.sample_size(10); // Fewer samples for long-running benchmarks

    // Test scalability with increasing data sizes
    let sizes = vec![100, 200, 500, 1000];
    let n_features = 10;

    for &n_samples in &sizes {
        let data = generate_clustered_data(n_samples, n_features, 5, 1.0);

        group.throughput(Throughput::Elements(n_samples as u64));

        // K-means scales well
        group.bench_with_input(
            BenchmarkId::new("kmeans_scalability", n_samples),
            &data,
            |b, data| {
                b.iter(|| {
                    kmeans2(
                        data.view(),
                        5,
                        Some(10),
                        None,
                        Some(MinitMethod::Random),
                        Some(MissingMethod::Warn),
                        Some(false),
                        Some(42),
                    )
                })
            },
        );

        // DBSCAN scalability
        group.bench_with_input(
            BenchmarkId::new("dbscan_scalability", n_samples),
            &data,
            |b, data| b.iter(|| dbscan(data.view(), 1.0, 5, None)),
        );
    }

    group.finish();
}

/// Benchmark worst-case scenarios
fn bench_worst_case(c: &mut Criterion) {
    let mut group = c.benchmark_group("worst_case");
    group.measurement_time(Duration::from_secs(15));

    let n_samples = 500;
    let n_features = 10;

    // Completely random data (no clusters)
    let random_data = generate_random_data(n_samples, n_features);

    group.bench_function("kmeans_random_data", |b| {
        b.iter(|| {
            kmeans2(
                random_data.view(),
                5,
                Some(10),
                None,
                Some(MinitMethod::Random),
                Some(MissingMethod::Warn),
                Some(false),
                Some(42),
            )
        })
    });

    group.bench_function("dbscan_random_data", |b| {
        b.iter(|| dbscan(random_data.view(), 0.5, 5, None))
    });

    // High-dimensional data
    let high_dim_data = generate_clustered_data(200, 50, 3, 1.0);

    group.bench_function("kmeans_high_dimensional", |b| {
        b.iter(|| {
            kmeans2(
                high_dim_data.view(),
                3,
                Some(10),
                None,
                Some(MinitMethod::Random),
                Some(MissingMethod::Warn),
                Some(false),
                Some(42),
            )
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_kmeans,
    bench_hierarchical,
    bench_dbscan,
    bench_spectral,
    bench_distance_metrics,
    bench_validation_metrics,
    bench_data_structures,
    bench_scalability,
    bench_worst_case
);

criterion_main!(benches);
