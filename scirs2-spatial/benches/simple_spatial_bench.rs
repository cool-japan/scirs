//! Simple Spatial Performance Benchmarks
//!
//! A lightweight benchmark suite that focuses on core spatial operations
//! without complex dependencies.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scirs2_core::ndarray::Array2;
use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_spatial::distance::{euclidean, manhattan, pdist};
use std::hint::black_box;
use std::time::Duration;

// Simple data generator
#[allow(dead_code)]
fn generate_test_data(_npoints: usize, dimensions: usize) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(42);
    Array2::from_shape_fn((_npoints, dimensions), |_| rng.random_range(-10.0..10.0))
}

// Basic distance calculation benchmark
#[allow(dead_code)]
fn bench_distance_calculations(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance_calculations");

    for &size in &[100, 500, 1000] {
        for &dim in &[3, 10, 50] {
            let points = generate_test_data(size, dim);

            group.throughput(Throughput::Elements(size as u64));

            group.bench_with_input(
                BenchmarkId::new("euclidean_pairwise", format!("{size}x{dim}")),
                &(size, dim),
                |b, _| {
                    b.iter(|| {
                        let mut total = 0.0;
                        for i in 0..points.nrows() {
                            for j in (i + 1)..points.nrows() {
                                let p1 = points.row(i).to_vec();
                                let p2 = points.row(j).to_vec();
                                total += black_box(euclidean(&p1, &p2));
                            }
                        }
                        total
                    })
                },
            );

            group.bench_with_input(
                BenchmarkId::new("manhattan_pairwise", format!("{size}x{dim}")),
                &(size, dim),
                |b, _| {
                    b.iter(|| {
                        let mut total = 0.0;
                        for i in 0..points.nrows() {
                            for j in (i + 1)..points.nrows() {
                                let p1 = points.row(i).to_vec();
                                let p2 = points.row(j).to_vec();
                                total += black_box(manhattan(&p1, &p2));
                            }
                        }
                        total
                    })
                },
            );
        }
    }

    group.finish();
}

// Distance matrix benchmark
#[allow(dead_code)]
fn bench_distance_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance_matrix");

    for &size in &[100, 200, 500] {
        let points = generate_test_data(size, 5);
        let expected_ops = size * (size - 1) / 2;

        group.throughput(Throughput::Elements(expected_ops as u64));

        group.bench_with_input(BenchmarkId::new("pdist_euclidean", size), &size, |b, _| {
            b.iter(|| black_box(pdist(&points, euclidean)))
        });

        group.bench_with_input(BenchmarkId::new("pdist_manhattan", size), &size, |b, _| {
            b.iter(|| black_box(pdist(&points, manhattan)))
        });
    }

    group.finish();
}

// Memory scaling benchmark
#[allow(dead_code)]
fn bench_memory_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_scaling");
    group.measurement_time(Duration::from_secs(10));

    for &size in &[100, 300, 500, 800] {
        let points = generate_test_data(size, 8);
        let memory_mb = (size * 8 * std::mem::size_of::<f64>()) as f64 / (1024.0 * 1024.0);

        group.throughput(Throughput::Bytes(
            (size * 8 * std::mem::size_of::<f64>()) as u64,
        ));

        group.bench_with_input(
            BenchmarkId::new("memory_usage", format!("{memory_mb:.1}MB")),
            &size,
            |b, _| {
                b.iter(|| {
                    let distances = pdist(&points, euclidean);
                    black_box(distances.sum())
                })
            },
        );
    }

    group.finish();
}

// Simple performance validation
#[allow(dead_code)]
fn bench_performance_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("performance_validation");

    let points = generate_test_data(1000, 10);

    group.bench_function("baseline_performance", |b| {
        b.iter(|| {
            // Representative workload
            let distances = pdist(&points, euclidean);
            let sum = distances.sum();
            black_box(sum)
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_distance_calculations,
    bench_distance_matrix,
    bench_memory_scaling,
    bench_performance_validation,
);

criterion_main!(benches);
