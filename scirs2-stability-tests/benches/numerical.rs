//! Numerical benchmark comparisons.
//!
//! Benchmarks key operations for performance regression detection.
//! These cover the same algorithmic surface as SciPy / NumPy / LAPACK so that
//! relative performance can be tracked across releases.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::Array2;

// ---------------------------------------------------------------------------
// Matrix operations
// ---------------------------------------------------------------------------

fn bench_matrix_multiply(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul");
    for &size in &[32usize, 64, 128, 256] {
        let a = Array2::<f64>::ones((size, size));
        let b = Array2::<f64>::ones((size, size));
        group.bench_with_input(BenchmarkId::new("ndarray_dot", size), &size, |bench, _| {
            bench.iter(|| a.dot(&b))
        });
    }
    group.finish();
}

fn bench_eye_alloc(c: &mut Criterion) {
    let mut group = c.benchmark_group("eye_alloc");
    for &size in &[64usize, 128, 256, 512] {
        group.bench_with_input(BenchmarkId::new("Array2_eye", size), &size, |bench, _| {
            bench.iter(|| Array2::<f64>::eye(size))
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Linear algebra: SVD, QR, LU
// ---------------------------------------------------------------------------

fn bench_svd(c: &mut Criterion) {
    use scirs2_linalg::svd;
    let mut group = c.benchmark_group("svd");
    for &size in &[16usize, 32, 64] {
        let a = Array2::<f64>::ones((size, size));
        group.bench_with_input(BenchmarkId::new("full", size), &size, |bench, _| {
            bench.iter(|| svd(&a.view(), false, None).expect("svd"))
        });
    }
    group.finish();
}

fn bench_qr(c: &mut Criterion) {
    use scirs2_linalg::qr;
    let mut group = c.benchmark_group("qr");
    for &size in &[32usize, 64, 128] {
        let a = Array2::<f64>::ones((size, size));
        group.bench_with_input(BenchmarkId::new("qr", size), &size, |bench, _| {
            bench.iter(|| qr(&a.view(), None).expect("qr"))
        });
    }
    group.finish();
}

fn bench_lu(c: &mut Criterion) {
    use scirs2_linalg::lu;
    let mut group = c.benchmark_group("lu");
    for &size in &[32usize, 64, 128] {
        let a = Array2::<f64>::ones((size, size));
        group.bench_with_input(BenchmarkId::new("lu", size), &size, |bench, _| {
            bench.iter(|| lu(&a.view(), None).expect("lu"))
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Statistical distributions: Normal PDF / CDF / PPF
// ---------------------------------------------------------------------------

fn bench_normal_pdf(c: &mut Criterion) {
    use scirs2_stats::distributions::Normal;
    let dist = Normal::new(0.0_f64, 1.0_f64).expect("valid");
    c.bench_function("normal_pdf_1000", |b| {
        b.iter(|| {
            let mut acc = 0.0_f64;
            for i in 0..1000_i64 {
                acc += dist.pdf(i as f64 * 0.01 - 5.0);
            }
            acc
        })
    });
}

fn bench_normal_cdf(c: &mut Criterion) {
    use scirs2_stats::distributions::Normal;
    let dist = Normal::new(0.0_f64, 1.0_f64).expect("valid");
    c.bench_function("normal_cdf_1000", |b| {
        b.iter(|| {
            let mut acc = 0.0_f64;
            for i in 0..1000_i64 {
                acc += dist.cdf(i as f64 * 0.01 - 5.0);
            }
            acc
        })
    });
}

fn bench_normal_ppf(c: &mut Criterion) {
    use scirs2_stats::distributions::Normal;
    let dist = Normal::new(0.0_f64, 1.0_f64).expect("valid");
    c.bench_function("normal_ppf_100", |b| {
        b.iter(|| {
            let mut acc = 0.0_f64;
            for i in 1..=100_i64 {
                acc += dist
                    .ppf(i as f64 * 0.009_f64 + 0.005_f64)
                    .unwrap_or(f64::NAN);
            }
            acc
        })
    });
}

// ---------------------------------------------------------------------------
// Criterion harness wiring
// ---------------------------------------------------------------------------

criterion_group!(benches_matmul, bench_matrix_multiply, bench_eye_alloc,);

criterion_group!(benches_linalg, bench_svd, bench_qr, bench_lu,);

criterion_group!(
    benches_stats,
    bench_normal_pdf,
    bench_normal_cdf,
    bench_normal_ppf,
);

criterion_main!(benches_matmul, benches_linalg, benches_stats);
