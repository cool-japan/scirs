//! Comprehensive benchmark suite comparing scirs2-stats with SciPy
//!
//! This benchmark suite provides performance comparisons for all major
//! statistical functions between scirs2-stats and SciPy.

#![allow(unused_must_use)]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, PlotConfiguration};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use scirs2_core::random::prelude::*;
use scirs2_core::random::{Distribution, Exponential, StandardNormal};
use scirs2_stats::tests::ttest::Alternative;
use scirs2_stats::*;
use scirs2_stats::{coefficient_of_variation_simd, mad_simd, quantiles_simd};
use std::hint::black_box;
use std::time::Duration;

/// Configuration for benchmark runs
struct BenchConfig {
    samplesizes: Vec<usize>,
    warmup_time: Duration,
    measurement_time: Duration,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            samplesizes: vec![10, 50, 100, 500, 1000, 5000, 10000],
            warmup_time: Duration::from_secs(1),
            measurement_time: Duration::from_secs(3),
        }
    }
}

/// Generate various types of test data
mod data_generators {
    use super::*;

    pub fn normal(n: usize, mean: f64, std: f64) -> Array1<f64> {
        let mut rng = thread_rng();
        let normal = Normal::new(mean, std).unwrap();
        Array1::from_shape_fn(n, |_| rng.sample(normal))
    }

    pub fn uniform(n: usize, low: f64, high: f64) -> Array1<f64> {
        let mut rng = thread_rng();
        let uniform = Uniform::new(low, high).unwrap();
        Array1::from_shape_fn(n, |_| rng.sample(uniform))
    }

    pub fn exponential(n: usize, lambda: f64) -> Array1<f64> {
        let mut rng = thread_rng();
        let exp = Exponential::new(lambda).unwrap();
        Array1::from_shape_fn(n, |_| rng.sample(exp))
    }

    pub fn multivariate_normal(n: usize, dim: usize) -> Array2<f64> {
        let mut rng = thread_rng();
        let normal = StandardNormal;
        Array2::from_shape_fn((n, dim), |_| rng.sample(normal))
    }

    pub fn correlateddata(n: usize, correlation: f64) -> (Array1<f64>, Array1<f64>) {
        let mut rng = thread_rng();
        let normal = StandardNormal;
        let x = Array1::from_shape_fn(n, |_| rng.sample(normal));
        let noise = Array1::from_shape_fn(n, |_| rng.sample(normal));
        let y = correlation * &x + (1.0 - correlation.powi(2)).sqrt() * &noise;
        (x, y)
    }
}

/// Benchmark descriptive statistics
#[allow(dead_code)]
fn bench_descriptive_stats(c: &mut Criterion) {
    let config = BenchConfig::default();
    let mut group = c.benchmark_group("descriptive_statistics");
    group.warm_up_time(config.warmup_time);
    group.measurement_time(config.measurement_time);

    for &n in &config.samplesizes {
        let data = data_generators::normal(n, 0.0, 1.0);

        // Mean
        group.bench_with_input(BenchmarkId::new("mean", n), &data, |b, data| {
            b.iter(|| black_box(mean(&data.view())))
        });

        // Variance
        group.bench_with_input(BenchmarkId::new("variance", n), &data, |b, data| {
            b.iter(|| black_box(var(&data.view(), 1, None)))
        });

        // Standard deviation
        group.bench_with_input(BenchmarkId::new("std", n), &data, |b, data| {
            b.iter(|| black_box(std(&data.view(), 1, None)))
        });

        // Skewness
        group.bench_with_input(BenchmarkId::new("skewness", n), &data, |b, data| {
            b.iter(|| black_box(skew(&data.view(), false, None)))
        });

        // Kurtosis
        group.bench_with_input(BenchmarkId::new("kurtosis", n), &data, |b, data| {
            b.iter(|| black_box(kurtosis(&data.view(), false, false, None)))
        });

        // Median
        group.bench_with_input(BenchmarkId::new("median", n), &data, |b, data| {
            b.iter(|| {
                let mut data_copy = data.clone();
                black_box(median(&data_copy.view()))
            })
        });
    }

    group.finish();
}

/// Benchmark correlation functions
#[allow(dead_code)]
fn bench_correlation(c: &mut Criterion) {
    let config = BenchConfig::default();
    let mut group = c.benchmark_group("correlation");
    group.warm_up_time(config.warmup_time);
    group.measurement_time(config.measurement_time);

    for &n in &config.samplesizes {
        let (x, y) = data_generators::correlateddata(n, 0.7);

        // Pearson correlation
        group.bench_with_input(BenchmarkId::new("pearson", n), &(&x, &y), |b, (x, y)| {
            b.iter(|| black_box(pearsonr(&x.view(), &y.view(), "propagate")))
        });

        // Spearman correlation
        group.bench_with_input(BenchmarkId::new("spearman", n), &(&x, &y), |b, (x, y)| {
            b.iter(|| black_box(spearmanr(&x.view(), &y.view(), "propagate")))
        });

        // Kendall tau
        if n <= 1000 {
            // Kendall tau has O(n²) complexity
            group.bench_with_input(BenchmarkId::new("kendalltau", n), &(&x, &y), |b, (x, y)| {
                b.iter(|| black_box(kendalltau(&x.view(), &y.view(), "nan_policy", "method")))
            });
        }
    }

    group.finish();
}

/// Benchmark statistical tests
#[allow(dead_code)]
fn bench_statistical_tests(c: &mut Criterion) {
    let config = BenchConfig::default();
    let mut group = c.benchmark_group("statistical_tests");
    group.warm_up_time(config.warmup_time);
    group.measurement_time(config.measurement_time);

    for &n in &config.samplesizes {
        let data = data_generators::normal(n, 0.0, 1.0);
        let data2 = data_generators::normal(n, 0.5, 1.0);

        // T-tests
        group.bench_with_input(BenchmarkId::new("ttest_1samp", n), &data, |b, data| {
            b.iter(|| {
                black_box(ttest_1samp(
                    &data.view(),
                    0.0,
                    Alternative::TwoSided,
                    "propagate",
                ))
            })
        });

        group.bench_with_input(
            BenchmarkId::new("ttest_ind", n),
            &(&data, &data2),
            |b, (d1, d2)| {
                b.iter(|| {
                    black_box(ttest_ind(
                        &d1.view(),
                        &d2.view(),
                        true,
                        Alternative::TwoSided,
                        "propagate",
                    ))
                })
            },
        );

        // Non-parametric tests
        group.bench_with_input(
            BenchmarkId::new("mann_whitney", n),
            &(&data, &data2),
            |b, (d1, d2)| b.iter(|| black_box(mann_whitney(&d1.view(), &d2.view(), "auto", true))),
        );

        // Normality tests
        if n <= 5000 {
            // Shapiro-Wilk is expensive for large samples
            group.bench_with_input(BenchmarkId::new("shapiro_wilk", n), &data, |b, data| {
                b.iter(|| black_box(shapiro_wilk(&data.view())))
            });
        }

        group.bench_with_input(BenchmarkId::new("anderson_darling", n), &data, |b, data| {
            b.iter(|| black_box(anderson_darling(&data.view())))
        });
    }

    group.finish();
}

/// Benchmark distribution functions
#[allow(dead_code)]
fn bench_distributions(c: &mut Criterion) {
    let config = BenchConfig::default();
    let mut group = c.benchmark_group("distributions");
    group.warm_up_time(config.warmup_time);
    group.measurement_time(config.measurement_time);

    let eval_points = vec![10, 100, 1000, 10000];

    for &n in &eval_points {
        let x = data_generators::uniform(n, -3.0, 3.0);

        // Normal distribution
        let normal = distributions::Normal::new(0.0, 1.0).unwrap();
        group.bench_with_input(BenchmarkId::new("normal_pdf", n), &x, |b, x| {
            b.iter(|| {
                for &xi in x.iter() {
                    black_box(normal.pdf(xi));
                }
            })
        });

        group.bench_with_input(BenchmarkId::new("normal_cdf", n), &x, |b, x| {
            b.iter(|| {
                for &xi in x.iter() {
                    black_box(normal.cdf(xi));
                }
            })
        });

        // Student's t distribution
        let t_dist = distributions::StudentT::new(10.0, 0.0, 1.0).unwrap();
        group.bench_with_input(BenchmarkId::new("t_pdf", n), &x, |b, x| {
            b.iter(|| {
                for &xi in x.iter() {
                    black_box(t_dist.pdf(xi));
                }
            })
        });

        // Chi-square distribution
        let chi2 = distributions::ChiSquare::new(5.0, 0.0, 1.0).unwrap();
        let x_positive = data_generators::uniform(n, 0.1, 10.0);
        group.bench_with_input(BenchmarkId::new("chi2_pdf", n), &x_positive, |b, x| {
            b.iter(|| {
                for &xi in x.iter() {
                    black_box(chi2.pdf(xi));
                }
            })
        });
    }

    group.finish();
}

/// Benchmark regression functions
#[allow(dead_code)]
fn bench_regression(c: &mut Criterion) {
    let config = BenchConfig::default();
    let mut group = c.benchmark_group("regression");
    group.warm_up_time(config.warmup_time);
    group.measurement_time(config.measurement_time);

    for &n in &config.samplesizes {
        // Simple linear regression
        let x = data_generators::uniform(n, 0.0, 10.0);
        let y = &x * 2.0 + data_generators::normal(n, 0.0, 0.1);

        group.bench_with_input(
            BenchmarkId::new("linear_regression", n),
            &(&x, &y),
            |b, (x, y)| b.iter(|| black_box(regression::linregress(&x.view(), &y.view()))),
        );

        // Multiple linear regression
        if n >= 50 {
            let x_multi = data_generators::multivariate_normal(n, 5);
            let y = data_generators::normal(n, 0.0, 1.0);

            group.bench_with_input(
                BenchmarkId::new("multiple_regression", n),
                &(&x_multi, &y),
                |b, (x, y)| {
                    b.iter(|| black_box(regression::multilinear_regression(&x.view(), &y.view())))
                },
            );
        }

        // Polynomial regression
        group.bench_with_input(
            BenchmarkId::new("polynomial_regression_deg3", n),
            &(&x, &y),
            |b, (x, y)| b.iter(|| black_box(regression::polyfit(&x.view(), &y.view(), 3))),
        );
    }

    group.finish();
}

/// Benchmark quantile functions
#[allow(dead_code)]
fn bench_quantiles(c: &mut Criterion) {
    let config = BenchConfig::default();
    let mut group = c.benchmark_group("quantiles");
    group.warm_up_time(config.warmup_time);
    group.measurement_time(config.measurement_time);

    for &n in &config.samplesizes {
        let data = data_generators::normal(n, 0.0, 1.0);

        // Single quantile
        group.bench_with_input(BenchmarkId::new("quantile_50", n), &data, |b, data| {
            b.iter(|| {
                let mut data_copy = data.clone();
                black_box(quantile(
                    &data_copy.view(),
                    0.5,
                    QuantileInterpolation::Linear,
                ))
            })
        });

        // Multiple quantiles
        let quantiles = Array1::from(vec![0.25, 0.5, 0.75]);
        group.bench_with_input(
            BenchmarkId::new("quantiles_multiple", n),
            &data,
            |b, data| {
                b.iter(|| {
                    let mut data_copy = data.clone();
                    black_box(quantiles_simd(
                        &mut data_copy.view_mut(),
                        &quantiles.view(),
                        "linear",
                    ))
                })
            },
        );
    }

    group.finish();
}

/// Benchmark dispersion measures
#[allow(dead_code)]
fn bench_dispersion(c: &mut Criterion) {
    let config = BenchConfig::default();
    let mut group = c.benchmark_group("dispersion");
    group.warm_up_time(config.warmup_time);
    group.measurement_time(config.measurement_time);

    for &n in &config.samplesizes {
        let data = data_generators::normal(n, 0.0, 1.0);

        // MAD (Median Absolute Deviation)
        group.bench_with_input(BenchmarkId::new("mad", n), &data, |b, data| {
            b.iter(|| {
                let mut data_copy = data.clone();
                black_box(mad_simd(&mut data_copy.view_mut(), 1.0, "propagate"))
            })
        });

        // IQR (Interquartile Range)
        group.bench_with_input(BenchmarkId::new("iqr", n), &data, |b, data| {
            b.iter(|| {
                let mut data_copy = data.clone();
                black_box(iqr(&data_copy.view(), Some("linear")))
            })
        });

        // Coefficient of Variation
        group.bench_with_input(BenchmarkId::new("cv", n), &data, |b, data| {
            b.iter(|| black_box(coefficient_of_variation_simd(&data.view(), "propagate")))
        });

        // TODO: Gini coefficient - function not yet implemented
        // group.bench_with_input(BenchmarkId::new("gini", n), &data, |b, data| {
        //     b.iter(|| black_box(gini(&data.view())))
        // });
    }

    group.finish();
}

/// Benchmark random sampling functions
#[allow(dead_code)]
fn bench_random_sampling(c: &mut Criterion) {
    let config = BenchConfig::default();
    let mut group = c.benchmark_group("random_sampling");
    group.warm_up_time(config.warmup_time);
    group.measurement_time(config.measurement_time);

    for &n in &config.samplesizes {
        let data = data_generators::normal(n, 0.0, 1.0);

        // Choice with replacement
        group.bench_with_input(
            BenchmarkId::new("choice_with_replacement", n),
            &data,
            |b, data| {
                b.iter(|| {
                    let mut rng = scirs2_core::random::rng();
                    black_box(random::choice(&data.view(), 100.min(n), true, None, None).unwrap())
                })
            },
        );

        // Choice without replacement
        if n >= 100 {
            group.bench_with_input(
                BenchmarkId::new("choice_without_replacement", n),
                &data,
                |b, data| {
                    b.iter(|| {
                        let mut rng = scirs2_core::random::rng();
                        black_box(
                            random::choice(&data.view(), 100.min(n), false, None, None).unwrap(),
                        )
                    })
                },
            );
        }

        // Permutation
        group.bench_with_input(BenchmarkId::new("permutation", n), &n, |b, &n| {
            b.iter(|| {
                let arr = Array1::from_iter(0..n);
                black_box(random::permutation(&arr.view(), Some(42)))
            })
        });
    }

    group.finish();
}

// Main benchmark groups
criterion_group! {
    name = benches;
    config = Criterion::default()
        .with_plots()
        .sample_size(100);
    targets =
        bench_descriptive_stats,
        bench_correlation,
        bench_statistical_tests,
        bench_distributions,
        bench_regression,
        bench_quantiles,
        bench_dispersion,
        bench_random_sampling
}

criterion_main!(benches);
