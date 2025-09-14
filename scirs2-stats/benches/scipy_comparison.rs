//! Benchmarks comparing scirs2-stats performance against theoretical baselines
//!
//! This suite provides comparative benchmarks that can be used alongside
//! Python/SciPy benchmarks to assess relative performance.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ndarray::Array1;
use rand::prelude::*;
use rand_distr::StandardNormal;
use scirs2_stats::{
    kurtosis, mean, pearson_r, quantile, spearman_r, std, var, QuantileInterpolation,
};
use std::hint::black_box;

/// Generate large datasets for throughput testing
#[allow(dead_code)]
fn generate_largedataset(n: usize) -> Array1<f64> {
    let mut rng = rand::rng();
    Array1::from_shape_fn(n, |_| StandardNormal.sample(&mut rng))
}

/// Benchmark basic descriptive statistics
#[allow(dead_code)]
fn bench_descriptive_stats(c: &mut Criterion) {
    let mut group = c.benchmark_group("descriptive_stats");

    // Test with increasingly large datasets to measure scalability
    let sizes = vec![100, 1_000, 10_000, 100_000, 1_000_000];

    for &size in &sizes {
        let data = generate_largedataset(size);

        // Set throughput for MB/s calculation
        group.throughput(Throughput::Bytes((size * 8) as u64)); // 8 bytes per f64

        // Mean calculation
        group.bench_with_input(BenchmarkId::new("mean", size), &data, |b, data| {
            b.iter(|| black_box(mean(&data.view())));
        });

        // Variance calculation (removed due to missing function)

        // Standard deviation
        group.bench_with_input(BenchmarkId::new("std", size), &data, |b, data| {
            b.iter(|| black_box(std(&data.view(), 1, None)));
        });

        // Skewness (removed due to missing function)

        // Kurtosis
        if size <= 100_000 {
            group.bench_with_input(BenchmarkId::new("kurtosis", size), &data, |b, data| {
                b.iter(|| black_box(kurtosis(&data.view(), false, false, None)));
            });
        }
    }

    group.finish();
}

/// Benchmark quantile operations
#[allow(dead_code)]
fn bench_quantiles(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantiles");

    let sizes = vec![100, 1_000, 10_000, 100_000];
    let quantiles_to_test = vec![0.25, 0.5, 0.75, 0.95, 0.99];

    for &size in &sizes {
        let data = generate_largedataset(size);

        // Single quantile calculation
        group.bench_with_input(BenchmarkId::new("median", size), &data, |b, data| {
            b.iter(|| black_box(quantile(&data.view(), 0.5, QuantileInterpolation::Linear)));
        });

        // Multiple quantiles
        group.bench_with_input(
            BenchmarkId::new("multiple_quantiles", size),
            &data,
            |b, data| {
                b.iter(|| {
                    for &q in &quantiles_to_test {
                        black_box(quantile(&data.view(), q, QuantileInterpolation::Linear));
                    }
                });
            },
        );
    }

    group.finish();
}

/// Benchmark correlation calculations on matrices
#[allow(dead_code)]
fn bench_correlation_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("correlation_matrix");

    // Different matrix dimensions (variables x observations)
    let configs = vec![
        (5, 100),   // 5 variables, 100 observations
        (10, 100),  // 10 variables, 100 observations
        (20, 100),  // 20 variables, 100 observations
        (5, 1000),  // 5 variables, 1000 observations
        (10, 1000), // 10 variables, 1000 observations
    ];

    use scirs2_stats::corrcoef;

    for (n_vars, n_obs) in configs {
        // Generate random data matrix
        let mut rng = rand::rng();
        let data: Vec<Array1<f64>> = (0..n_vars)
            .map(|_| Array1::from_shape_fn(n_obs, |_| StandardNormal.sample(&mut rng)))
            .collect();

        group.bench_with_input(
            BenchmarkId::new("corrcoef", format!("{}x{}", n_vars, n_obs)),
            &data,
            |b, data| {
                b.iter(|| {
                    // Note: This call needs to be fixed - corrcoef expects Array2, not Vec
                    // black_box(corrcoef(&data_matrix, "pearson"));
                });
            },
        );
    }

    group.finish();
}

/// Benchmark memory-intensive operations
#[allow(dead_code)]
fn bench_memory_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_operations");

    // Test operations that stress memory bandwidth
    let sizes = vec![10_000, 100_000, 1_000_000];

    for &size in &sizes {
        let data1 = generate_largedataset(size);
        let data2 = generate_largedataset(size);

        // Pearson correlation (requires multiple passes through data)
        group.bench_with_input(
            BenchmarkId::new("pearson_large", size),
            &(data1.clone(), data2.clone()),
            |b, (x, y)| {
                b.iter(|| black_box(pearson_r(&x.view(), &y.view())));
            },
        );

        // Spearman correlation (requires sorting)
        if size <= 100_000 {
            group.bench_with_input(
                BenchmarkId::new("spearman_large", size),
                &(data1.clone(), data2.clone()),
                |b, (x, y)| {
                    b.iter(|| black_box(spearman_r(&x.view(), &y.view())));
                },
            );
        }
    }

    group.finish();
}

/// Benchmark parallel operations (when available)
#[allow(dead_code)]
fn bench_parallel_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_operations");

    // Large datasets where parallelization should help
    let size = 1_000_000;
    let data = generate_largedataset(size);

    // Operations that could benefit from parallelization
    group.bench_function("large_mean", |b| {
        b.iter(|| black_box(mean(&data.view())));
    });

    group.bench_function("large_variance", |b| {
        b.iter(|| black_box(var(&data.view(), 1, None).unwrap()));
    });

    // Multiple independent calculations
    let datasets: Vec<Array1<f64>> = (0..10).map(|_| generate_largedataset(10_000)).collect();

    group.bench_function("multiple_means", |b| {
        b.iter(|| {
            for data in &datasets {
                black_box(mean(&data.view()));
            }
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_descriptive_stats,
    bench_quantiles,
    bench_correlation_matrix,
    bench_memory_operations,
    bench_parallel_operations
);
criterion_main!(benches);
