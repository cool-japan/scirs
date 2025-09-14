//! Comprehensive Performance Benchmarking Suite for scirs2-spatial
//!
//! This benchmark suite validates SIMD and parallel processing performance claims
//! across different data sizes, metrics, and architectures. It provides actionable
//! insights for optimization decisions and performance scaling analysis.

use criterion::{
    criterion_group, criterion_main, BenchmarkId, Criterion, PlotConfiguration, Throughput,
};
use ndarray::Array2;
use rand::{rngs::StdRng, Rng, SeedableRng};
use scirs2_spatial::{
    distance::{euclidean, pdist},
    simd_distance::{
        parallel_cdist, parallel_pdist, simd_euclidean_distance, simd_euclidean_distance_batch,
        simd_knn_search, simd_manhattan_distance,
    },
    BallTree, KDTree,
};
use std::hint::black_box;
use std::time::Duration;

// Benchmark configuration constants
const SMALL_SIZES: &[usize] = &[100, 500, 1_000];
const MEDIUM_SIZES: &[usize] = &[1_000, 5_000, 10_000];
const LARGE_SIZES: &[usize] = &[10_000, 50_000, 100_000];
#[allow(dead_code)]
const DIMENSIONS: &[usize] = &[2, 3, 5, 10, 20, 50, 100];
const DISTANCE_METRICS: &[&str] = &["euclidean", "manhattan", "chebyshev"];
const KNN_K_VALUES: &[usize] = &[1, 5, 10, 20, 50];

// Seed for reproducible benchmarks
const BENCHMARK_SEED: u64 = 12345;

/// Generate reproducible random points for benchmarking
#[allow(dead_code)]
fn generate_points(_npoints: usize, dimensions: usize, seed: u64) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    Array2::from_shape_fn((_npoints, dimensions), |_| rng.gen_range(-10.0..10.0))
}

/// Generate two sets of random points for cross-distance benchmarks
#[allow(dead_code)]
fn generate_point_pairs(
    n1: usize,
    n2: usize,
    dimensions: usize,
    seed: u64,
) -> (Array2<f64>, Array2<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let points1 = Array2::from_shape_fn((n1, dimensions), |_| rng.gen_range(-10.0..10.0));
    let points2 = Array2::from_shape_fn((n2, dimensions), |_| rng.gen_range(-10.0..10.0));
    (points1, points2)
}

/// Benchmark SIMD vs scalar distance calculations for different data sizes
#[allow(dead_code)]
fn bench_simd_vs_scalar_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_vs_scalar_distance");
    group
        .plot_config(PlotConfiguration::default().summary_scale(criterion::AxisScale::Logarithmic));

    for &size in SMALL_SIZES {
        for &dim in &[3, 10, 50, 100] {
            let points1 = generate_points(size, dim, BENCHMARK_SEED);
            let points2 = generate_points(size, dim, BENCHMARK_SEED + 1);

            group.throughput(Throughput::Elements(size as u64));

            // Scalar benchmark
            group.bench_with_input(
                BenchmarkId::new("scalar_euclidean", format!("{size}x{dim}")),
                &(size, dim),
                |b, _| {
                    b.iter(|| {
                        for (row1, row2) in points1.outer_iter().zip(points2.outer_iter()) {
                            black_box(euclidean(
                                row1.as_slice().unwrap(),
                                row2.as_slice().unwrap(),
                            ));
                        }
                    })
                },
            );

            // SIMD batch benchmark
            group.bench_with_input(
                BenchmarkId::new("simd_euclidean_batch", format!("{size}x{dim}")),
                &(size, dim),
                |b, _| {
                    b.iter(|| {
                        black_box(
                            simd_euclidean_distance_batch(&points1.view(), &points2.view())
                                .unwrap(),
                        )
                    })
                },
            );

            // Individual SIMD calls for comparison
            group.bench_with_input(
                BenchmarkId::new("simd_euclidean_individual", format!("{size}x{dim}")),
                &(size, dim),
                |b, _| {
                    b.iter(|| {
                        for (row1, row2) in points1.outer_iter().zip(points2.outer_iter()) {
                            black_box(
                                simd_euclidean_distance(
                                    row1.as_slice().unwrap(),
                                    row2.as_slice().unwrap(),
                                )
                                .unwrap(),
                            );
                        }
                    })
                },
            );
        }
    }

    group.finish();
}

/// Benchmark parallel vs sequential spatial operations
#[allow(dead_code)]
fn bench_parallel_vs_sequential(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_vs_sequential");
    group
        .plot_config(PlotConfiguration::default().summary_scale(criterion::AxisScale::Logarithmic));

    for &size in MEDIUM_SIZES {
        let points = generate_points(size, 5, BENCHMARK_SEED);

        group.throughput(Throughput::Elements((size * (size - 1) / 2) as u64));

        // Sequential pdist
        group.bench_with_input(
            BenchmarkId::new("sequential_pdist", size),
            &size,
            |b_, _| b_.iter(|| black_box(pdist(&points, euclidean))),
        );

        // Parallel pdist
        group.bench_with_input(BenchmarkId::new("parallel_pdist", size), &size, |b_, _| {
            b_.iter(|| black_box(parallel_pdist(&points.view(), "euclidean").unwrap()))
        });
    }

    // KDTree construction benchmarks
    for &size in MEDIUM_SIZES {
        let points = generate_points(size, 3, BENCHMARK_SEED);

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(
            BenchmarkId::new("kdtree_construction", size),
            &size,
            |b_, _| b_.iter(|| black_box(KDTree::new(&points).unwrap())),
        );
    }

    group.finish();
}

/// Benchmark memory efficiency for large datasets
#[allow(dead_code)]
fn bench_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_efficiency");
    group.measurement_time(Duration::from_secs(20));

    for &size in LARGE_SIZES {
        let points = generate_points(size, 10, BENCHMARK_SEED);
        let data_size_mb = (size * 10 * std::mem::size_of::<f64>()) as f64 / (1024.0 * 1024.0);

        group.throughput(Throughput::Bytes(
            (size * 10 * std::mem::size_of::<f64>()) as u64,
        ));

        // Memory-efficient distance matrix computation
        group.bench_with_input(
            BenchmarkId::new("memory_efficient_pdist", format!("{data_size_mb:.1}MB")),
            &size,
            |b, _| {
                b.iter(|| {
                    // Only compute a subset to avoid memory explosion
                    let subset_size = (size / 10).max(100);
                    let subset = points.slice(ndarray::s![..subset_size, ..]);
                    black_box(parallel_pdist(&subset, "euclidean").unwrap())
                })
            },
        );

        // Chunked processing simulation
        group.bench_with_input(
            BenchmarkId::new("chunked_processing", format!("{data_size_mb:.1}MB")),
            &size,
            |b, _| {
                b.iter(|| {
                    let chunk_size = 1000;
                    let mut total_distance = 0.0;
                    for chunk_start in (0..size).step_by(chunk_size) {
                        let chunk_end = (chunk_start + chunk_size).min(size);
                        let chunk = points.slice(ndarray::s![chunk_start..chunk_end, ..]);
                        if chunk.nrows() > 1 {
                            let distances = parallel_pdist(&chunk, "euclidean").unwrap();
                            total_distance += distances.sum();
                        }
                    }
                    black_box(total_distance)
                })
            },
        );
    }

    group.finish();
}

/// Benchmark different distance metrics performance comparison
#[allow(dead_code)]
fn bench_distance_metrics_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance_metrics_comparison");

    let size = 5000;
    let points = generate_points(size, 10, BENCHMARK_SEED);

    group.throughput(Throughput::Elements((size * (size - 1) / 2) as u64));

    for &metric in DISTANCE_METRICS {
        group.bench_with_input(
            BenchmarkId::new("parallel_pdist", metric),
            metric,
            |b, metric| b.iter(|| black_box(parallel_pdist(&points.view(), metric).unwrap())),
        );
    }

    // Compare individual distance calculations
    let p1: Vec<f64> = (0..100).map(|i| i as f64).collect();
    let p2: Vec<f64> = (0..100).map(|i| (i + 1) as f64).collect();

    for &metric in &["euclidean", "manhattan"] {
        group.bench_with_input(
            BenchmarkId::new("simd_single_distance", metric),
            metric,
            |b, metric| match metric {
                "euclidean" => b.iter(|| black_box(simd_euclidean_distance(&p1, &p2).unwrap())),
                "manhattan" => b.iter(|| black_box(simd_manhattan_distance(&p1, &p2).unwrap())),
                _ => unreachable!(),
            },
        );
    }

    group.finish();
}

/// Benchmark cross-architecture performance
#[allow(dead_code)]
fn bench_cross_architecture_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("cross_architecture_performance");

    // Report available SIMD features
    eprintln!("=== SIMD Architecture Report ===");

    #[cfg(target_arch = "x86_64")]
    {
        eprintln!("Architecture: x86_64");
        eprintln!("  SSE2: {}", is_x86_feature_detected!("sse2"));
        eprintln!("  AVX: {}", is_x86_feature_detected!("avx"));
        eprintln!("  AVX2: {}", is_x86_feature_detected!("avx2"));
        eprintln!("  AVX-512F: {}", is_x86_feature_detected!("avx512f"));
    }

    #[cfg(target_arch = "aarch64")]
    {
        eprintln!("Architecture: aarch64");
        eprintln!(
            "  NEON: {}",
            std::arch::is_aarch64_feature_detected!("neon")
        );
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        eprintln!("Architecture: Other (using scalar fallbacks)");
    }

    eprintln!("================================");

    for &dim in &[8, 16, 32, 64, 128] {
        let p1: Vec<f64> = (0..dim).map(|i| i as f64).collect();
        let p2: Vec<f64> = (0..dim).map(|i| (i + 1) as f64).collect();

        group.throughput(Throughput::Elements(dim as u64));

        // Benchmark SIMD implementation
        group.bench_with_input(BenchmarkId::new("simd_euclidean", dim), &dim, |b_, _| {
            b_.iter(|| black_box(simd_euclidean_distance(&p1, &p2).unwrap()))
        });

        // Benchmark scalar fallback
        group.bench_with_input(BenchmarkId::new("scalar_euclidean", dim), &dim, |b_, _| {
            b_.iter(|| black_box(euclidean(&p1, &p2)))
        });
    }

    group.finish();
}

/// Benchmark spatial data structure performance
#[allow(dead_code)]
fn bench_spatial_data_structures(c: &mut Criterion) {
    let mut group = c.benchmark_group("spatial_data_structures");
    group.measurement_time(Duration::from_secs(15));

    for &size in &[1_000, 5_000, 10_000] {
        let points = generate_points(size, 3, BENCHMARK_SEED);
        let query_points = generate_points(100, 3, BENCHMARK_SEED + 1);

        // KDTree construction
        group.bench_with_input(
            BenchmarkId::new("kdtree_construction", size),
            &size,
            |b_, _| b_.iter(|| black_box(KDTree::new(&points).unwrap())),
        );

        // BallTree construction
        group.bench_with_input(
            BenchmarkId::new("balltree_construction", size),
            &size,
            |b, _| {
                b.iter(|| black_box(BallTree::with_euclidean_distance(&points.view(), 10).unwrap()))
            },
        );

        // Query performance
        let kdtree = KDTree::new(&points).unwrap();
        let balltree = BallTree::with_euclidean_distance(&points.view(), 10).unwrap();

        for &k in &[1, 5, 10] {
            group.bench_with_input(
                BenchmarkId::new("kdtree_query", format!("{size}pts_k{k}")),
                &(size, k),
                |b, _| {
                    b.iter(|| {
                        for query in query_points.outer_iter() {
                            black_box(kdtree.query(query.as_slice().unwrap(), k).unwrap());
                        }
                    })
                },
            );

            group.bench_with_input(
                BenchmarkId::new("balltree_query", format!("{size}pts_k{k}")),
                &(size, k),
                |b, _| {
                    b.iter(|| {
                        for query in query_points.outer_iter() {
                            black_box(balltree.query(query.as_slice().unwrap(), k, true).unwrap());
                        }
                    })
                },
            );
        }
    }

    group.finish();
}

/// Benchmark KNN search performance and scaling
#[allow(dead_code)]
fn bench_knn_performance_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("knn_performance_scaling");

    let data_size = 10_000;
    let query_size = 1_000;
    let dim = 5;

    let data_points = generate_points(data_size, dim, BENCHMARK_SEED);
    let query_points = generate_points(query_size, dim, BENCHMARK_SEED + 1);

    for &k in KNN_K_VALUES {
        group.throughput(Throughput::Elements((query_size * k) as u64));

        group.bench_with_input(BenchmarkId::new("simd_knn_search", k), &k, |b, &k| {
            b.iter(|| {
                black_box(
                    simd_knn_search(&query_points.view(), &data_points.view(), k, "euclidean")
                        .unwrap(),
                )
            })
        });
    }

    // Compare different metrics for KNN
    let k = 5;
    for &metric in DISTANCE_METRICS {
        group.bench_with_input(
            BenchmarkId::new("knn_by_metric", metric),
            metric,
            |b, metric| {
                b.iter(|| {
                    black_box(
                        simd_knn_search(&query_points.view(), &data_points.view(), k, metric)
                            .unwrap(),
                    )
                })
            },
        );
    }

    group.finish();
}

/// Benchmark scaling behavior with problem size
#[allow(dead_code)]
fn bench_scaling_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("scaling_analysis");
    group
        .plot_config(PlotConfiguration::default().summary_scale(criterion::AxisScale::Logarithmic));

    // Distance matrix scaling (O(n²))
    for &size in &[500, 1_000, 2_000, 4_000] {
        let points = generate_points(size, 5, BENCHMARK_SEED);
        let expected_operations = size * (size - 1) / 2;

        group.throughput(Throughput::Elements(expected_operations as u64));

        group.bench_with_input(BenchmarkId::new("pdist_scaling", size), &size, |b_, _| {
            b_.iter(|| {
                // Limit computation to avoid excessive runtime
                let subset_size = if size > 2000 { 1000 } else { size };
                let subset = points.slice(ndarray::s![..subset_size, ..]);
                black_box(parallel_pdist(&subset, "euclidean").unwrap())
            })
        });
    }

    // Cross-distance scaling (O(nm))
    let n_values = [500, 1_000, 2_000];
    let m_values = [500, 1_000, 2_000];

    for &n in &n_values {
        for &m in &m_values {
            let (points1, points2) = generate_point_pairs(n, m, 5, BENCHMARK_SEED);

            group.throughput(Throughput::Elements((n * m) as u64));

            group.bench_with_input(
                BenchmarkId::new("cdist_scaling", format!("{n}x{m}")),
                &(n, m),
                |b, _| {
                    b.iter(|| {
                        black_box(
                            parallel_cdist(&points1.view(), &points2.view(), "euclidean").unwrap(),
                        )
                    })
                },
            );
        }
    }

    group.finish();
}

/// Benchmark memory allocation patterns
#[allow(dead_code)]
fn bench_memory_allocation_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_allocation_patterns");

    let size = 2_000;
    let dim = 10;

    // Pre-allocated vs dynamic allocation
    group.bench_function("preallocated_computation", |b| {
        let points = generate_points(size, dim, BENCHMARK_SEED);
        b.iter(|| {
            // Simulate pre-allocated buffer reuse
            let _distances = parallel_pdist(&points.view(), "euclidean").unwrap();
            black_box(_distances)
        })
    });

    group.bench_function("dynamic_allocation", |b| {
        b.iter(|| {
            // Generate fresh data each iteration (worst case)
            let points = generate_points(size, dim, BENCHMARK_SEED);
            let _distances = parallel_pdist(&points.view(), "euclidean").unwrap();
            black_box(_distances)
        })
    });

    // Different memory access patterns
    let points = generate_points(size, dim, BENCHMARK_SEED);

    group.bench_function("sequential_access", |b| {
        b.iter(|| {
            let mut sum = 0.0;
            for row in points.outer_iter() {
                sum += row.sum();
            }
            black_box(sum)
        })
    });

    group.bench_function("strided_access", |b| {
        b.iter(|| {
            let mut sum = 0.0;
            for col in points.axis_iter(ndarray::Axis(1)) {
                sum += col.sum();
            }
            black_box(sum)
        })
    });

    group.finish();
}

/// Generate performance report
#[allow(dead_code)]
fn bench_performance_report(c: &mut Criterion) {
    let mut group = c.benchmark_group("performance_report");

    // Quick system characterization
    let test_size = 1_000;
    let test_dim = 10;
    let points = generate_points(test_size, test_dim, BENCHMARK_SEED);

    group.bench_function("system_characterization", |b| {
        b.iter(|| {
            // Run a representative workload
            let _distances = parallel_pdist(&points.view(), "euclidean").unwrap();
            let _sum = _distances.sum();
            black_box(_sum)
        })
    });

    group.finish();

    // Output performance characteristics to stderr for capture
    eprintln!("\n=== Performance Characteristics Report ===");
    eprintln!("Test configuration: {test_size} points, {test_dim} dimensions");
    eprintln!("Expected operations: {}", test_size * (test_size - 1) / 2);
    eprintln!(
        "Data size: {:.2} MB",
        (test_size * test_dim * 8) as f64 / (1024.0 * 1024.0)
    );
    eprintln!("==========================================");
}

// Define benchmark groups
criterion_group!(
    benches,
    bench_simd_vs_scalar_distance,
    bench_parallel_vs_sequential,
    bench_memory_efficiency,
    bench_distance_metrics_comparison,
    bench_cross_architecture_performance,
    bench_spatial_data_structures,
    bench_knn_performance_scaling,
    bench_scaling_analysis,
    bench_memory_allocation_patterns,
    bench_performance_report,
);

criterion_main!(benches);
