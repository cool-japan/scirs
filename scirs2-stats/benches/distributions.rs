//! Benchmarks for statistical distributions
//!
//! This benchmark suite compares the performance of scirs2-stats distributions
//! against reference implementations and theoretical expectations.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::Array1;
use scirs2_stats::distributions::{
    beta, binom, chi2, expon, f, gamma, norm, poisson, t, uniform, Beta, Binomial, ChiSquare,
    Exponential, Gamma, Normal, Poisson, StudentT, Uniform, F,
};
use scirs2_stats::Distribution;
use statrs::statistics::Statistics;
use std::hint::black_box;

/// Benchmark PDF calculations for continuous distributions
#[allow(dead_code)]
fn bench_continuous_pdf(c: &mut Criterion) {
    let mut group = c.benchmark_group("continuous_pdf");

    // Test different sample sizes
    let samplesizes = vec![10, 100, 1000, 10000];

    for &n in &samplesizes {
        // Generate test points
        let x: Array1<f64> = Array1::linspace(-3.0, 3.0, n);

        // Normal distribution
        group.bench_with_input(BenchmarkId::new("normal", n), &x, |b, x| {
            let dist = norm(0.0, 1.0).unwrap();
            b.iter(|| {
                for &xi in x.iter() {
                    black_box(dist.pdf(xi));
                }
            });
        });

        // Student's t distribution
        group.bench_with_input(BenchmarkId::new("student_t", n), &x, |b, x| {
            let dist = t(5.0, 0.0, 1.0).unwrap();
            b.iter(|| {
                for &xi in x.iter() {
                    black_box(dist.pdf(xi));
                }
            });
        });

        // Uniform distribution
        let x_uniform: Array1<f64> = Array1::linspace(0.1, 0.9, n);
        group.bench_with_input(BenchmarkId::new("uniform", n), &x_uniform, |b, x| {
            let dist = uniform(0.0, 1.0).unwrap();
            b.iter(|| {
                for &xi in x.iter() {
                    black_box(dist.pdf(xi));
                }
            });
        });
    }

    group.finish();
}

/// Benchmark CDF calculations for continuous distributions
#[allow(dead_code)]
fn bench_continuous_cdf(c: &mut Criterion) {
    let mut group = c.benchmark_group("continuous_cdf");

    let samplesizes = vec![10, 100, 1000];

    for &n in &samplesizes {
        let x: Array1<f64> = Array1::linspace(-3.0, 3.0, n);

        // Normal distribution CDF
        group.bench_with_input(BenchmarkId::new("normal", n), &x, |b, x| {
            let dist = norm(0.0, 1.0).unwrap();
            b.iter(|| {
                for &xi in x.iter() {
                    black_box(dist.cdf(xi));
                }
            });
        });

        // Chi-square distribution CDF
        let x_positive: Array1<f64> = Array1::linspace(0.1, 10.0, n);
        group.bench_with_input(BenchmarkId::new("chi_square", n), &x_positive, |b, x| {
            let dist = chi2(5.0, 0.0, 1.0).unwrap();
            b.iter(|| {
                for &xi in x.iter() {
                    black_box(dist.cdf(xi));
                }
            });
        });
    }

    group.finish();
}

/// Benchmark random number generation
#[allow(dead_code)]
fn bench_random_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("random_generation");

    let samplesizes = vec![100, 1000, 10000, 100000];

    for &n in &samplesizes {
        // Normal distribution
        group.bench_with_input(BenchmarkId::new("normal", n), &n, |b, &n| {
            let dist = norm(0.0, 1.0).unwrap();
            b.iter(|| {
                black_box(dist.rvs(n));
            });
        });

        // Uniform distribution
        group.bench_with_input(BenchmarkId::new("uniform", n), &n, |b, &n| {
            let dist = uniform(0.0, 1.0).unwrap();
            b.iter(|| {
                black_box(dist.rvs(n));
            });
        });

        // Exponential distribution
        group.bench_with_input(BenchmarkId::new("exponential", n), &n, |b, &n| {
            let dist = expon(1.0, 0.0).unwrap();
            b.iter(|| {
                black_box(dist.rvs(n));
            });
        });

        // Poisson distribution
        group.bench_with_input(BenchmarkId::new("poisson", n), &n, |b, &n| {
            let dist = poisson(3.0, 0.0).unwrap();
            b.iter(|| {
                black_box(dist.rvs(n));
            });
        });
    }

    group.finish();
}

/// Benchmark statistical moments calculation
#[allow(dead_code)]
fn bench_moments(c: &mut Criterion) {
    let mut group = c.benchmark_group("moments");

    // Normal distribution moments
    group.bench_function("normal_mean", |b| {
        let dist = norm(5.0, 2.0).unwrap();
        b.iter(|| black_box(dist.mean()));
    });

    group.bench_function("normal_variance", |b| {
        let dist = norm(5.0, 2.0).unwrap();
        b.iter(|| black_box(dist.var()));
    });

    // Gamma distribution moments
    group.bench_function("gamma_mean", |b| {
        let dist = gamma(2.0, 3.0, 0.0).unwrap();
        b.iter(|| black_box(dist.mean()));
    });

    group.bench_function("gamma_variance", |b| {
        let dist = gamma(2.0, 3.0, 0.0).unwrap();
        b.iter(|| black_box(dist.var()));
    });

    // Beta distribution moments
    group.bench_function("beta_mean", |b| {
        let dist = beta(2.0, 5.0, 0.0, 1.0).unwrap();
        b.iter(|| black_box(dist.mean()));
    });

    group.finish();
}

/// Benchmark discrete distributions
#[allow(dead_code)]
fn bench_discrete_distributions(c: &mut Criterion) {
    let mut group = c.benchmark_group("discrete_distributions");

    let test_values = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];

    // Poisson PMF
    group.bench_function("poisson_pmf", |b| {
        let dist = poisson(3.0, 0.0).unwrap();
        b.iter(|| {
            for &k in &test_values {
                black_box(dist.pmf(k));
            }
        });
    });

    // Binomial PMF
    group.bench_function("binomial_pmf", |b| {
        let dist = binom(10, 0.3).unwrap();
        b.iter(|| {
            for &k in &test_values {
                black_box(dist.pmf(k));
            }
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_continuous_pdf,
    bench_continuous_cdf,
    bench_random_generation,
    bench_moments,
    bench_discrete_distributions
);
criterion_main!(benches);
