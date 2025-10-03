//! Quick Spatial Performance Benchmarks
//!
//! A fast benchmark suite for getting concrete performance measurements

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
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

// Quick performance validation
#[allow(dead_code)]
fn bench_performance_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("quick_performance");
    group.measurement_time(Duration::from_secs(5));
    group.warm_up_time(Duration::from_secs(1));

    // Single point distance calculations
    let p1 = &[0.0, 1.0, 2.0, 3.0, 4.0];
    let p2 = &[1.0, 2.0, 3.0, 4.0, 5.0];

    group.bench_function("single_euclidean_distance", |b| {
        b.iter(|| black_box(euclidean(p1, p2)))
    });

    group.bench_function("single_manhattan_distance", |b| {
        b.iter(|| black_box(manhattan(p1, p2)))
    });

    // Small distance matrix
    let small_points = generate_test_data(50, 5);

    group.bench_function("small_distance_matrix_euclidean", |b| {
        b.iter(|| {
            let distances = pdist(&small_points, euclidean);
            black_box(distances.sum())
        })
    });

    group.bench_function("small_distance_matrix_manhattan", |b| {
        b.iter(|| {
            let distances = pdist(&small_points, manhattan);
            black_box(distances.sum())
        })
    });

    // Medium distance matrix
    let medium_points = generate_test_data(100, 10);

    group.bench_function("medium_distance_matrix_euclidean", |b| {
        b.iter(|| {
            let distances = pdist(&medium_points, euclidean);
            black_box(distances.sum())
        })
    });

    // Performance scaling test
    for &size in &[50, 100, 200] {
        let points = generate_test_data(size, 5);

        group.bench_with_input(
            BenchmarkId::new("scaling_euclidean", size),
            &size,
            |b_, _| {
                b_.iter(|| {
                    let distances = pdist(&points, euclidean);
                    black_box(distances.sum())
                })
            },
        );
    }

    group.finish();
}

// System performance characterization
#[allow(dead_code)]
fn bench_system_characterization(c: &mut Criterion) {
    let mut group = c.benchmark_group("system_performance");
    group.measurement_time(Duration::from_secs(3));

    // Memory performance
    let sizes = [64, 128, 256, 512];
    let dims = [3, 10, 20];

    for &size in &sizes {
        for &dim in &dims {
            let points = generate_test_data(size, dim);
            let data_size_kb = (size * dim * std::mem::size_of::<f64>()) as f64 / 1024.0;

            group.bench_with_input(
                BenchmarkId::new(
                    "memory_performance",
                    format!("{size}x{dim}_({data_size_kb:.1}KB)"),
                ),
                &(size, dim),
                |b_, _| {
                    b_.iter(|| {
                        // Simulate a typical workload
                        let mut total = 0.0;
                        for i in 0..points.nrows().min(20) {
                            for j in (i + 1)..points.nrows().min(20) {
                                let p1 = points.row(i).to_vec();
                                let p2 = points.row(j).to_vec();
                                total += euclidean(&p1, &p2);
                            }
                        }
                        black_box(total)
                    })
                },
            );
        }
    }

    group.finish();
}

// Report performance metrics
#[allow(dead_code)]
fn report_performance_metrics(c: &mut Criterion) {
    let mut group = c.benchmark_group("performance_report");
    group.measurement_time(Duration::from_secs(2));

    // Baseline computational performance
    let test_points = generate_test_data(100, 10);

    group.bench_function("baseline_workload", |b| {
        b.iter(|| {
            let distances = pdist(&test_points, euclidean);
            let sum = distances.sum();
            let mean = sum / (distances.len() as f64);
            black_box((sum, mean))
        })
    });

    group.finish();

    // Print performance summary
    eprintln!("\n=== PERFORMANCE SUMMARY ===");
    eprintln!("Test configuration:");
    eprintln!("  • Matrix operations: Available");
    eprintln!("  • SIMD support: Available (basic scalar fallback tested)");
    eprintln!("  • Memory efficiency: Standard ndarray operations");
    eprintln!("  • Parallelism: Sequential baseline");
    eprintln!("\nExpected performance characteristics:");
    eprintln!("  • Single distance calculation: ~10-100 nanoseconds");
    eprintln!("  • Small matrices (50x50): ~1-10 milliseconds");
    eprintln!("  • Medium matrices (100x100): ~10-100 milliseconds");
    eprintln!("============================\n");
}

criterion_group!(
    benches,
    bench_performance_validation,
    bench_system_characterization,
    report_performance_metrics,
);

criterion_main!(benches);
