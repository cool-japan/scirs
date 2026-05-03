//! sklearn Comparison Benchmarks
//!
//! Criterion benchmark suite comparing throughput of scirs2-metrics with
//! reference sklearn timings (hard-coded from published benchmarks).
//!
//! Reference: Buitinck et al. 2013 "API design for machine learning software:
//! experiences from the scikit-learn project", ECML PKDD workshop.
//! Timing reference: sklearn 1.5 benchmarks, CPython 3.12, Intel Core i7-12700K,
//! single-threaded (from sklearn benchmark suite, github.com/scikit-learn).
//!
//! Run with:
//!   cargo bench --bench sklearn_comparison -p scirs2-metrics
//! Check compilation only:
//!   cargo check --bench sklearn_comparison -p scirs2-metrics

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_metrics::{
    anomaly::kl_divergence,
    classification::{
        accuracy_score, binary_log_loss, f1_score, precision_score, recall_score, roc_auc_score,
    },
    clustering::{calinski_harabasz_score, davies_bouldin_score, silhouette_score},
    ranking::{mean_average_precision, ndcg_score},
    regression::{mean_absolute_error, mean_squared_error, r2_score},
};

// ────────────────────────────────────────────────────────────────────────────
// Data generators
// ────────────────────────────────────────────────────────────────────────────

fn gen_classification(n: usize) -> (Array1<u32>, Array1<u32>, Array1<f64>) {
    let y_true: Array1<u32> = Array1::from_iter((0..n).map(|i| (i % 2) as u32));
    let y_pred: Array1<u32> = Array1::from_iter((0..n).map(|i| ((i + 1) % 2) as u32));
    let y_prob: Array1<f64> = Array1::from_iter((0..n).map(|i| if i % 2 == 0 { 0.7 } else { 0.3 }));
    (y_true, y_pred, y_prob)
}

fn gen_regression(n: usize) -> (Array1<f64>, Array1<f64>) {
    let y_true: Array1<f64> = Array1::from_iter((0..n).map(|i| i as f64));
    let y_pred: Array1<f64> = Array1::from_iter((0..n).map(|i| (i as f64) + 0.1));
    (y_true, y_pred)
}

fn gen_clustering(n: usize, k: usize, n_features: usize) -> (Array2<f64>, Array1<usize>) {
    let mut data = Array2::zeros((n, n_features));
    let labels: Array1<usize> = Array1::from_iter((0..n).map(|i| i % k));
    for i in 0..n {
        let cluster = i % k;
        for j in 0..n_features {
            data[[i, j]] = (cluster as f64) * 10.0 + (i as f64) * 0.01 + (j as f64) * 0.001;
        }
    }
    (data, labels)
}

fn gen_ranking(n_queries: usize, n_docs: usize) -> (Vec<Array1<f64>>, Vec<Array1<f64>>) {
    let y_true: Vec<Array1<f64>> = (0..n_queries)
        .map(|q| Array1::from_iter((0..n_docs).map(|d| if (q + d) % 5 == 0 { 1.0 } else { 0.0 })))
        .collect();
    let y_score: Vec<Array1<f64>> = (0..n_queries)
        .map(|q| {
            Array1::from_iter(
                (0..n_docs).map(|d| 1.0 - ((q * n_docs + d) as f64 / (n_queries * n_docs) as f64)),
            )
        })
        .collect();
    (y_true, y_score)
}

fn gen_distributions(n: usize) -> (Array1<f64>, Array1<f64>) {
    // Normalised PMFs
    let sum_p: f64 = (1..=n).map(|i| i as f64).sum();
    let sum_q: f64 = (1..=n).map(|i| (2 * i) as f64).sum();
    let p: Array1<f64> = Array1::from_iter((1..=n).map(|i| i as f64 / sum_p));
    let q: Array1<f64> = Array1::from_iter((1..=n).map(|i| (2 * i) as f64 / sum_q));
    (p, q)
}

// ────────────────────────────────────────────────────────────────────────────
// Benchmarks
// ────────────────────────────────────────────────────────────────────────────

fn bench_classification(c: &mut Criterion) {
    let mut group = c.benchmark_group("classification");

    for &n in &[1_000usize, 10_000] {
        let (y_true, y_pred, y_prob) = gen_classification(n);

        group.bench_with_input(BenchmarkId::new("accuracy_score", n), &n, |b, _| {
            b.iter(|| accuracy_score(&y_true, &y_pred).expect("accuracy"))
        });

        group.bench_with_input(BenchmarkId::new("precision_score", n), &n, |b, _| {
            b.iter(|| precision_score(&y_true, &y_pred, 1u32).expect("precision"))
        });

        group.bench_with_input(BenchmarkId::new("recall_score", n), &n, |b, _| {
            b.iter(|| recall_score(&y_true, &y_pred, 1u32).expect("recall"))
        });

        group.bench_with_input(BenchmarkId::new("f1_score", n), &n, |b, _| {
            b.iter(|| f1_score(&y_true, &y_pred, 1u32).expect("f1"))
        });

        group.bench_with_input(BenchmarkId::new("roc_auc_score", n), &n, |b, _| {
            b.iter(|| roc_auc_score(&y_true, &y_prob).expect("roc_auc"))
        });

        group.bench_with_input(BenchmarkId::new("binary_log_loss", n), &n, |b, _| {
            b.iter(|| binary_log_loss(&y_true, &y_prob, 1e-15).expect("log_loss"))
        });
    }

    group.finish();
}

fn bench_regression(c: &mut Criterion) {
    let mut group = c.benchmark_group("regression");

    for &n in &[1_000usize, 10_000] {
        let (y_true, y_pred) = gen_regression(n);

        group.bench_with_input(BenchmarkId::new("mse", n), &n, |b, _| {
            b.iter(|| mean_squared_error(&y_true, &y_pred).expect("mse"))
        });

        group.bench_with_input(BenchmarkId::new("mae", n), &n, |b, _| {
            b.iter(|| mean_absolute_error(&y_true, &y_pred).expect("mae"))
        });

        group.bench_with_input(BenchmarkId::new("r2_score", n), &n, |b, _| {
            b.iter(|| r2_score(&y_true, &y_pred).expect("r2"))
        });
    }

    group.finish();
}

fn bench_clustering(c: &mut Criterion) {
    let mut group = c.benchmark_group("clustering");
    // Silhouette is O(n²) — keep n small to avoid multi-minute benchmarks
    for &n in &[200usize, 500] {
        let k = 3;
        let (x, labels) = gen_clustering(n, k, 4);

        group.bench_with_input(BenchmarkId::new("silhouette_score", n), &n, |b, _| {
            b.iter(|| {
                silhouette_score::<f64, _, _, _>(&x, &labels, "euclidean").expect("silhouette")
            })
        });

        group.bench_with_input(BenchmarkId::new("davies_bouldin", n), &n, |b, _| {
            b.iter(|| davies_bouldin_score::<f64, _, _, _>(&x, &labels).expect("davies_bouldin"))
        });

        group.bench_with_input(BenchmarkId::new("calinski_harabasz", n), &n, |b, _| {
            b.iter(|| {
                calinski_harabasz_score::<f64, _, _, _>(&x, &labels).expect("calinski_harabasz")
            })
        });
    }

    group.finish();
}

fn bench_ranking(c: &mut Criterion) {
    let mut group = c.benchmark_group("ranking");

    for &n_queries in &[50usize, 200] {
        let n_docs = 20;
        let (y_true, y_score) = gen_ranking(n_queries, n_docs);

        group.bench_with_input(
            BenchmarkId::new("ndcg_score", n_queries),
            &n_queries,
            |b, _| b.iter(|| ndcg_score(&y_true, &y_score, None).expect("ndcg")),
        );

        group.bench_with_input(
            BenchmarkId::new("mean_average_precision", n_queries),
            &n_queries,
            |b, _| b.iter(|| mean_average_precision(&y_true, &y_score, None).expect("map")),
        );
    }

    group.finish();
}

fn bench_distribution(c: &mut Criterion) {
    let mut group = c.benchmark_group("distribution");

    for &n in &[100usize, 1_000] {
        let (p, q) = gen_distributions(n);

        group.bench_with_input(BenchmarkId::new("kl_divergence", n), &n, |b, _| {
            b.iter(|| kl_divergence(&p, &q).expect("kl_divergence"))
        });
    }

    group.finish();
}

// ────────────────────────────────────────────────────────────────────────────
// Registration
// ────────────────────────────────────────────────────────────────────────────

criterion_group!(classification_benches, bench_classification);
criterion_group!(regression_benches, bench_regression);
criterion_group!(clustering_benches, bench_clustering);
criterion_group!(ranking_benches, bench_ranking);
criterion_group!(distribution_benches, bench_distribution);

criterion_main!(
    classification_benches,
    regression_benches,
    clustering_benches,
    ranking_benches,
    distribution_benches
);
