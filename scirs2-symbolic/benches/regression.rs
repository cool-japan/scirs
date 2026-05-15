//! Symbolic regression performance benchmarks.

use criterion::{criterion_group, criterion_main, Criterion};
use ndarray::{Array1, Array2};
use scirs2_symbolic::regression::{discover, SrConfig};
use std::hint::black_box;

fn build_data_x_squared(n: usize) -> (Array2<f64>, Array1<f64>) {
    let xs: Vec<f64> = (0..n)
        .map(|i| (i as f64 - (n as f64) / 2.0) * 0.1)
        .collect();
    let ys: Vec<f64> = xs.iter().map(|x| x * x).collect();
    let features = Array2::from_shape_vec((n, 1), xs).expect("features shape");
    let targets = Array1::from_vec(ys);
    (features, targets)
}

fn bench_discover_x_squared_50(c: &mut Criterion) {
    let (features, targets) = build_data_x_squared(50);
    let config = SrConfig::default().with_max_iter(20).with_top_n(3);
    c.bench_function("discover x^2 from 50 samples (max_iter=20)", |b| {
        b.iter(|| discover(features.view(), targets.view(), black_box(&config)))
    });
}

fn bench_discover_x_squared_500(c: &mut Criterion) {
    let (features, targets) = build_data_x_squared(500);
    let config = SrConfig::default().with_max_iter(15).with_top_n(3);
    c.bench_function("discover x^2 from 500 samples (max_iter=15)", |b| {
        b.iter(|| discover(features.view(), targets.view(), black_box(&config)))
    });
}

criterion_group!(
    regression_benches,
    bench_discover_x_squared_50,
    bench_discover_x_squared_500,
);
criterion_main!(regression_benches);
