//! Performance regression benchmarks for scirs2-metrics
//!
//! This benchmark suite tracks performance of key metrics to detect regressions
//! and ensure consistent performance across releases.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::{Array1, Array2};
use scirs2_metrics::{
    anomaly::{js_divergence, kl_divergence, wasserstein_distance},
    classification::{accuracy_score, confusion_matrix, f1_score, precision_score, recall_score},
    clustering::{davies_bouldin_score, silhouette_score},
    optimization::numeric::StableMetrics,
    regression::{mean_absolute_error, mean_squared_error, r2_score},
};

/// Generate synthetic classification data for benchmarking
fn generate_classification_data(n_samples: usize) -> (Array1<f64>, Array1<f64>) {
    let y_true: Array1<f64> = Array1::from_iter((0..n_samples).map(|i| (i % 2) as f64));
    let y_pred: Array1<f64> = Array1::from_iter((0..n_samples).map(|i| ((i + 1) % 2) as f64));
    (y_true, y_pred)
}

/// Generate synthetic classification data with integer labels for confusion matrix
fn generate_classification_data_int(n_samples: usize) -> (Array1<i32>, Array1<i32>) {
    let y_true: Array1<i32> = Array1::from_iter((0..n_samples).map(|i| (i % 2) as i32));
    let y_pred: Array1<i32> = Array1::from_iter((0..n_samples).map(|i| ((i + 1) % 2) as i32));
    (y_true, y_pred)
}

/// Generate synthetic regression data for benchmarking
fn generate_regression_data(n_samples: usize) -> (Array1<f64>, Array1<f64>) {
    let y_true: Array1<f64> = Array1::from_iter((0..n_samples).map(|i| i as f64));
    let y_pred: Array1<f64> = Array1::from_iter((0..n_samples).map(|i| (i as f64) + 0.1));
    (y_true, y_pred)
}

/// Generate synthetic clustering data for benchmarking
fn generate_clustering_data(n_samples: usize, n_features: usize) -> (Array2<f64>, Array1<usize>) {
    let mut data = Array2::zeros((n_samples, n_features));
    let mut labels = Array1::zeros(n_samples);

    for i in 0..n_samples {
        let cluster = i % 3; // 3 clusters
        labels[i] = cluster;

        for j in 0..n_features {
            // Create cluster-specific offsets
            let offset = (cluster as f64) * 10.0;
            data[[i, j]] = offset + (i as f64) + (j as f64) * 0.1;
        }
    }

    (data, labels)
}

/// Generate synthetic probability distributions for benchmarking
fn generate_probability_distributions(n_samples: usize) -> (Array1<f64>, Array1<f64>) {
    let mut p = Array1::zeros(n_samples);
    let mut q = Array1::zeros(n_samples);

    let sum_p = n_samples as f64;
    let sum_q = (n_samples * 2) as f64;

    for i in 0..n_samples {
        p[i] = (i + 1) as f64 / sum_p;
        q[i] = ((i + 1) * 2) as f64 / sum_q;
    }

    (p, q)
}

/// Benchmark classification metrics performance
fn benchmark_classification_metrics(c: &mut Criterion) {
    let mut group = c.benchmark_group("classification_metrics");

    for size in [100, 1000, 10000, 100000].iter() {
        let (y_true, y_pred) = generate_classification_data(*size);
        let (y_true_int, y_pred_int) = generate_classification_data_int(*size);

        group.bench_with_input(
            BenchmarkId::new("accuracy_score", size),
            size,
            |b, &_size| b.iter(|| accuracy_score(&y_true, &y_pred).unwrap()),
        );

        group.bench_with_input(
            BenchmarkId::new("precision_score", size),
            size,
            |b, &_size| b.iter(|| precision_score(&y_true, &y_pred, 1.0).unwrap()),
        );

        group.bench_with_input(BenchmarkId::new("recall_score", size), size, |b, &_size| {
            b.iter(|| recall_score(&y_true, &y_pred, 1.0).unwrap())
        });

        group.bench_with_input(BenchmarkId::new("f1_score", size), size, |b, &_size| {
            b.iter(|| f1_score(&y_true, &y_pred, 1.0).unwrap())
        });

        group.bench_with_input(
            BenchmarkId::new("confusion_matrix", size),
            size,
            |b, &_size| b.iter(|| confusion_matrix(&y_true_int, &y_pred_int, None).unwrap()),
        );
    }

    group.finish();
}

/// Benchmark regression metrics performance
fn benchmark_regression_metrics(c: &mut Criterion) {
    let mut group = c.benchmark_group("regression_metrics");

    for size in [100, 1000, 10000, 100000].iter() {
        let (y_true, y_pred) = generate_regression_data(*size);

        group.bench_with_input(
            BenchmarkId::new("mean_squared_error", size),
            size,
            |b, &_size| b.iter(|| mean_squared_error(&y_true, &y_pred).unwrap()),
        );

        group.bench_with_input(
            BenchmarkId::new("mean_absolute_error", size),
            size,
            |b, &_size| b.iter(|| mean_absolute_error(&y_true, &y_pred).unwrap()),
        );

        group.bench_with_input(BenchmarkId::new("r2_score", size), size, |b, &_size| {
            b.iter(|| r2_score(&y_true, &y_pred).unwrap())
        });
    }

    group.finish();
}

/// Benchmark clustering metrics performance
fn benchmark_clustering_metrics(c: &mut Criterion) {
    let mut group = c.benchmark_group("clustering_metrics");

    for size in [50, 100, 500, 1000].iter() {
        // Smaller sizes for clustering due to complexity
        let (data, labels) = generate_clustering_data(*size, 2);

        group.bench_with_input(
            BenchmarkId::new("silhouette_score", size),
            size,
            |b, &_size| b.iter(|| silhouette_score(&data, &labels, "euclidean").unwrap()),
        );

        group.bench_with_input(
            BenchmarkId::new("davies_bouldin_score", size),
            size,
            |b, &_size| b.iter(|| davies_bouldin_score(&data, &labels).unwrap()),
        );
    }

    group.finish();
}

/// Benchmark anomaly detection metrics performance
fn benchmark_anomaly_metrics(c: &mut Criterion) {
    let mut group = c.benchmark_group("anomaly_metrics");

    for size in [100, 1000, 10000].iter() {
        let (p, q) = generate_probability_distributions(*size);
        let (data1, data2) = generate_regression_data(*size);

        group.bench_with_input(
            BenchmarkId::new("kl_divergence", size),
            size,
            |b, &_size| b.iter(|| kl_divergence(&p, &q).unwrap()),
        );

        group.bench_with_input(
            BenchmarkId::new("js_divergence", size),
            size,
            |b, &_size| b.iter(|| js_divergence(&p, &q).unwrap()),
        );

        group.bench_with_input(
            BenchmarkId::new("wasserstein_distance", size),
            size,
            |b, &_size| b.iter(|| wasserstein_distance(&data1, &data2).unwrap()),
        );
    }

    group.finish();
}

/// Benchmark stable statistical computation performance
fn benchmark_stable_metrics(c: &mut Criterion) {
    let mut group = c.benchmark_group("stable_metrics");
    let stable_metrics = StableMetrics::<f64>::new();

    for size in [100, 1000, 10000, 100000].iter() {
        let data: Vec<f64> = (0..*size).map(|i| i as f64).collect();

        group.bench_with_input(BenchmarkId::new("stable_mean", size), size, |b, &_size| {
            b.iter(|| stable_metrics.stable_mean(&data).unwrap())
        });

        group.bench_with_input(
            BenchmarkId::new("stable_variance", size),
            size,
            |b, &_size| b.iter(|| stable_metrics.stable_variance(&data, 1).unwrap()),
        );

        group.bench_with_input(BenchmarkId::new("stable_std", size), size, |b, &_size| {
            b.iter(|| stable_metrics.stable_std(&data, 1).unwrap())
        });
    }

    group.finish();
}

/// Benchmark high-dimensional data performance
fn benchmark_high_dimensional_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("high_dimensional");

    // Test with various dimensions while keeping sample size reasonable
    for dims in [2, 5, 10, 20, 50].iter() {
        let (data, labels) = generate_clustering_data(500, *dims);

        group.bench_with_input(
            BenchmarkId::new("silhouette_high_dim", dims),
            dims,
            |b, &_dims| b.iter(|| silhouette_score(&data, &labels, "euclidean").unwrap()),
        );
    }

    group.finish();
}

/// Benchmark memory efficiency for large datasets
fn benchmark_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_efficiency");

    // Test memory usage patterns with different data sizes
    for size in [10000, 50000, 100000].iter() {
        let (y_true, y_pred) = generate_regression_data(*size);

        group.bench_with_input(
            BenchmarkId::new("mse_large_data", size),
            size,
            |b, &_size| {
                b.iter(|| {
                    // This tests memory allocation patterns

                    mean_squared_error(&y_true, &y_pred).unwrap()
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("accuracy_large_data", size),
            size,
            |b, &_size| {
                b.iter(|| {
                    let y_true_class: Array1<f64> = y_true.mapv(|x| (x as usize % 2) as f64);
                    let y_pred_class: Array1<f64> = y_pred.mapv(|x| (x as usize % 2) as f64);
                    accuracy_score(&y_true_class, &y_pred_class).unwrap()
                })
            },
        );
    }

    group.finish();
}

/// Benchmark worst-case scenarios for edge case handling
fn benchmark_edge_cases(c: &mut Criterion) {
    let mut group = c.benchmark_group("edge_cases");

    // Test with very small numbers (potential underflow)
    let very_small_true = Array1::from_vec(vec![1e-100; 1000]);
    let very_small_pred = Array1::from_vec(vec![2e-100; 1000]);

    group.bench_function("mse_tiny_numbers", |b| {
        b.iter(|| mean_squared_error(&very_small_true, &very_small_pred).unwrap())
    });

    // Test with very large numbers (potential overflow)
    let very_large_true = Array1::from_vec(vec![1e50; 1000]);
    let very_large_pred = Array1::from_vec(vec![2e50; 1000]);

    group.bench_function("mse_huge_numbers", |b| {
        b.iter(|| mean_squared_error(&very_large_true, &very_large_pred).unwrap())
    });

    // Test with extreme class imbalance
    let mut imbalanced_true = vec![0.0; 10000];
    imbalanced_true[0] = 1.0; // Only one positive case
    let imbalanced_true = Array1::from_vec(imbalanced_true);
    let imbalanced_pred = Array1::zeros(10000);

    group.bench_function("accuracy_extreme_imbalance", |b| {
        b.iter(|| accuracy_score(&imbalanced_true, &imbalanced_pred).unwrap())
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_classification_metrics,
    benchmark_regression_metrics,
    benchmark_clustering_metrics,
    benchmark_anomaly_metrics,
    benchmark_stable_metrics,
    benchmark_high_dimensional_performance,
    benchmark_memory_efficiency,
    benchmark_edge_cases
);

criterion_main!(benches);
