//! Comprehensive benchmarks comparing scirs2-spatial with SciPy spatial module
//!
//! This benchmark suite validates performance claims and provides detailed
//! comparisons with scipy.spatial for v0.2.0 release.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_spatial::distance;
use scirs2_spatial::projections::{
    geographic_to_utm, geographic_to_web_mercator, utm_to_geographic,
};
use scirs2_spatial::variogram::VariogramModel as VariogramModelType;
use scirs2_spatial::*;
use std::hint::black_box;
use std::time::Duration;

/// Generate random points for benchmarking using scirs2-core random
fn generate_random_points(n: usize, dim: usize) -> Array2<f64> {
    let mut rng = scirs2_core::random::Random::seed(42);
    let dist = scirs2_core::random::Uniform::new(0.0_f64, 100.0)
        .expect("Failed to create uniform distribution");
    let values: Vec<f64> = rng.sample_vec(dist, n * dim);
    Array2::from_shape_vec((n, dim), values).expect("Failed to create Array2")
}

/// Generate a random Array1 for benchmarking
fn generate_random_array1(n: usize, low: f64, high: f64) -> Array1<f64> {
    let mut rng = scirs2_core::random::Random::seed(42);
    let dist = scirs2_core::random::Uniform::new(low, high)
        .expect("Failed to create uniform distribution");
    let values: Vec<f64> = rng.sample_vec(dist, n);
    Array1::from_vec(values)
}

/// Generate a random Array2 for benchmarking
fn generate_random_array2(rows: usize, cols: usize, low: f64, high: f64) -> Array2<f64> {
    let mut rng = scirs2_core::random::Random::seed(42);
    let dist = scirs2_core::random::Uniform::new(low, high)
        .expect("Failed to create uniform distribution");
    let values: Vec<f64> = rng.sample_vec(dist, rows * cols);
    Array2::from_shape_vec((rows, cols), values).expect("Failed to create Array2")
}

/// Benchmark distance computations
fn bench_distance_metrics(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance_metrics");

    for size in [100, 1000, 10000].iter() {
        let points1 = generate_random_array1(*size, 0.0, 100.0);
        let points2 = generate_random_array1(*size, 0.0, 100.0);

        let p1_slice = points1.as_slice().expect("slice conversion failed");
        let p2_slice = points2.as_slice().expect("slice conversion failed");

        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::new("euclidean", size), size, |b, _| {
            b.iter(|| black_box(distance::euclidean(p1_slice, p2_slice)))
        });

        group.bench_with_input(BenchmarkId::new("manhattan", size), size, |b, _| {
            b.iter(|| black_box(distance::manhattan(p1_slice, p2_slice)))
        });

        group.bench_with_input(BenchmarkId::new("cosine", size), size, |b, _| {
            b.iter(|| black_box(distance::cosine(p1_slice, p2_slice)))
        });
    }

    group.finish();
}

/// Benchmark KD-Tree construction and queries
fn bench_kdtree_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("kdtree");
    group.measurement_time(Duration::from_secs(10));

    for n_points in [100, 1000, 10000].iter() {
        let points = generate_random_points(*n_points, 3);

        // Construction benchmark
        group.throughput(Throughput::Elements(*n_points as u64));
        group.bench_with_input(
            BenchmarkId::new("construction", n_points),
            &points,
            |b, pts| b.iter(|| black_box(KDTree::new(pts).expect("KDTree construction failed"))),
        );

        // Query benchmark
        let tree = KDTree::new(&points).expect("KDTree construction failed");
        let query_point = vec![50.0, 50.0, 50.0];

        group.bench_with_input(BenchmarkId::new("query_k10", n_points), n_points, |b, _| {
            b.iter(|| black_box(tree.query(&query_point, 10).expect("Query failed")))
        });

        // Radius query
        group.bench_with_input(
            BenchmarkId::new("radius_query", n_points),
            n_points,
            |b, _| {
                b.iter(|| {
                    black_box(
                        tree.query_radius(&query_point, 10.0)
                            .expect("Radius query failed"),
                    )
                })
            },
        );
    }

    group.finish();
}

/// Benchmark distance matrix computation (cdist/pdist)
fn bench_distance_matrices(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance_matrices");
    group.measurement_time(Duration::from_secs(15));

    for n_points in [50, 100, 500].iter() {
        let points = generate_random_points(*n_points, 10);

        group.throughput(Throughput::Elements((*n_points * *n_points) as u64));

        // pdist benchmark
        group.bench_with_input(BenchmarkId::new("pdist", n_points), &points, |b, pts| {
            b.iter(|| black_box(distance::pdist(pts, distance::euclidean)))
        });

        // Parallel pdist benchmark
        group.bench_with_input(
            BenchmarkId::new("parallel_pdist", n_points),
            &points,
            |b, pts| {
                b.iter(|| {
                    black_box(
                        parallel_pdist(&pts.view(), "euclidean").expect("parallel_pdist failed"),
                    )
                })
            },
        );

        // cdist benchmark
        let points2 = generate_random_points(*n_points / 2, 10);
        group.bench_with_input(BenchmarkId::new("cdist", n_points), n_points, |b, _| {
            b.iter(|| black_box(distance::cdist(&points, &points2, distance::euclidean)))
        });
    }

    group.finish();
}

/// Benchmark SIMD-accelerated distance computations
fn bench_simd_distances(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_distances");

    for size in [100, 1000, 10000].iter() {
        let points1 = generate_random_array1(*size, 0.0, 100.0);
        let points2 = generate_random_array1(*size, 0.0, 100.0);

        let p1_slice = points1.as_slice().expect("slice conversion failed");
        let p2_slice = points2.as_slice().expect("slice conversion failed");

        group.throughput(Throughput::Elements(*size as u64));

        // SIMD Euclidean
        group.bench_with_input(BenchmarkId::new("simd_euclidean", size), size, |b, _| {
            b.iter(|| {
                black_box(
                    simd_euclidean_distance(p1_slice, p2_slice).expect("SIMD computation failed"),
                )
            })
        });

        // SIMD Manhattan
        group.bench_with_input(BenchmarkId::new("simd_manhattan", size), size, |b, _| {
            b.iter(|| {
                black_box(
                    simd_manhattan_distance(p1_slice, p2_slice).expect("SIMD computation failed"),
                )
            })
        });
    }

    group.finish();
}

/// Benchmark spatial statistics computations
fn bench_spatial_statistics(c: &mut Criterion) {
    let mut group = c.benchmark_group("spatial_statistics");

    for n_points in [50, 100, 500].iter() {
        let values = generate_random_array1(*n_points, 0.0, 10.0);
        let weights = generate_random_array2(*n_points, *n_points, 0.0, 1.0);

        group.throughput(Throughput::Elements(*n_points as u64));

        // Moran's I
        group.bench_with_input(BenchmarkId::new("morans_i", n_points), n_points, |b, _| {
            b.iter(|| {
                black_box(morans_i(&values.view(), &weights.view()).expect("Moran's I failed"))
            })
        });

        // Geary's C
        group.bench_with_input(BenchmarkId::new("gearys_c", n_points), n_points, |b, _| {
            b.iter(|| {
                black_box(gearys_c(&values.view(), &weights.view()).expect("Geary's C failed"))
            })
        });

        // Local Moran's I
        group.bench_with_input(
            BenchmarkId::new("local_morans_i", n_points),
            n_points,
            |b, _| {
                b.iter(|| {
                    black_box(
                        local_morans_i(&values.view(), &weights.view())
                            .expect("Local Moran's I failed"),
                    )
                })
            },
        );
    }

    group.finish();
}

/// Benchmark variogram computations
fn bench_variograms(c: &mut Criterion) {
    let mut group = c.benchmark_group("variograms");

    for n_points in [50, 100, 500].iter() {
        let coords = generate_random_points(*n_points, 2);
        let values = generate_random_array1(*n_points, 0.0, 10.0);

        group.throughput(Throughput::Elements(*n_points as u64));

        // Experimental variogram
        group.bench_with_input(
            BenchmarkId::new("experimental", n_points),
            n_points,
            |b, _| {
                b.iter(|| {
                    black_box(
                        experimental_variogram(&coords.view(), &values.view(), 10, None)
                            .expect("Experimental variogram failed"),
                    )
                })
            },
        );

        // Variogram fitting
        let (lags, gamma) = experimental_variogram(&coords.view(), &values.view(), 10, None)
            .expect("Experimental variogram failed");

        group.bench_with_input(
            BenchmarkId::new("fit_spherical", n_points),
            n_points,
            |b, _| {
                b.iter(|| {
                    black_box(
                        fit_variogram(&lags, &gamma, VariogramModelType::Spherical)
                            .expect("Variogram fitting failed"),
                    )
                })
            },
        );
    }

    group.finish();
}

/// Benchmark distance transforms
fn bench_distance_transforms(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance_transforms");

    for size in [32, 64, 128].iter() {
        // Create binary image with some features
        let mut binary = Array2::<i32>::zeros((*size, *size));
        for i in 0..*size {
            for j in 0..*size {
                if (i % 10 == 0) || (j % 10 == 0) {
                    binary[[i, j]] = 1;
                }
            }
        }

        group.throughput(Throughput::Elements((*size * *size) as u64));

        // Euclidean distance transform
        group.bench_with_input(BenchmarkId::new("euclidean_dt", size), &binary, |b, img| {
            b.iter(|| {
                black_box(
                    euclidean_distance_transform::<f64>(
                        &img.view(),
                        DistanceTransformMetric::Euclidean,
                    )
                    .expect("Distance transform failed"),
                )
            })
        });

        // Manhattan distance transform
        group.bench_with_input(BenchmarkId::new("manhattan_dt", size), &binary, |b, img| {
            b.iter(|| {
                black_box(
                    euclidean_distance_transform::<f64>(
                        &img.view(),
                        DistanceTransformMetric::Manhattan,
                    )
                    .expect("Distance transform failed"),
                )
            })
        });
    }

    group.finish();
}

/// Benchmark coordinate projections
fn bench_projections(c: &mut Criterion) {
    let mut group = c.benchmark_group("projections");

    let test_coords = vec![
        (40.7128, -74.0060),  // New York
        (51.5074, -0.1278),   // London
        (35.6762, 139.6503),  // Tokyo
        (-33.8688, 151.2093), // Sydney
    ];

    group.bench_function("geographic_to_utm", |b| {
        b.iter(|| {
            for &(lat, lon) in &test_coords {
                black_box(geographic_to_utm(lat, lon).expect("UTM conversion failed"));
            }
        })
    });

    group.bench_function("geographic_to_web_mercator", |b| {
        b.iter(|| {
            for &(lat, lon) in &test_coords {
                black_box(
                    geographic_to_web_mercator(lat, lon).expect("Web Mercator conversion failed"),
                );
            }
        })
    });

    // Roundtrip benchmark
    group.bench_function("utm_roundtrip", |b| {
        b.iter(|| {
            for &(lat, lon) in &test_coords {
                let (zone, e, n) = geographic_to_utm(lat, lon).expect("UTM conversion failed");
                black_box(utm_to_geographic(e, n, zone).expect("UTM inverse failed"));
            }
        })
    });

    group.finish();
}

/// Benchmark convex hull computation
fn bench_convex_hull(c: &mut Criterion) {
    let mut group = c.benchmark_group("convex_hull");

    for n_points in [10, 50, 100, 500].iter() {
        let points = generate_random_points(*n_points, 2);

        group.throughput(Throughput::Elements(*n_points as u64));

        group.bench_with_input(BenchmarkId::new("2d_hull", n_points), &points, |b, pts| {
            b.iter(|| black_box(ConvexHull::new(&pts.view()).expect("Convex hull failed")))
        });
    }

    group.finish();
}

/// Benchmark Delaunay triangulation
fn bench_delaunay(c: &mut Criterion) {
    let mut group = c.benchmark_group("delaunay");
    group.measurement_time(Duration::from_secs(10));

    for n_points in [10, 50, 100].iter() {
        let points = generate_random_points(*n_points, 2);

        group.throughput(Throughput::Elements(*n_points as u64));

        group.bench_with_input(
            BenchmarkId::new("triangulation", n_points),
            &points,
            |b, pts| b.iter(|| black_box(Delaunay::new(pts).expect("Delaunay failed"))),
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_distance_metrics,
    bench_kdtree_operations,
    bench_distance_matrices,
    bench_simd_distances,
    bench_spatial_statistics,
    bench_variograms,
    bench_distance_transforms,
    bench_projections,
    bench_convex_hull,
    bench_delaunay,
);

criterion_main!(benches);
