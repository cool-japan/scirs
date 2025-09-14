//! Comprehensive performance benchmarks against SciPy and other libraries
//!
//! This benchmark suite provides detailed performance comparisons of core statistical
//! operations, demonstrating the efficiency gains from SIMD optimizations and
//! parallel processing implementations.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ndarray::{Array1, Array2, Axis};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use scirs2_stats::distributions::{
    beta::Beta, gamma::Gamma, normal::Normal, uniform::Uniform as StatsUniform,
};
use scirs2_stats::tests::ttest::Alternative;
use scirs2_stats::{
    corrcoef, kendalltau, ks_2samp, kurtosis, mann_whitney, mean, median, pearsonr, quantile,
    shapiro, skew, spearmanr, std, traits::Distribution, ttest_1samp, ttest_ind, var,
    QuantileInterpolation,
};
use statrs::statistics::Statistics;
use std::hint::black_box;
use std::time::Duration;

/// Generate test data of various sizes for benchmarking
#[allow(dead_code)]
fn generate_testdata(size: usize, seed: u64) -> Array1<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    Array1::from_iter((0..size).map(|_| rng.gen::<f64>() * 100.0 - 50.0))
}

/// Generate matrix test data for multivariate benchmarks
#[allow(dead_code)]
fn generate_matrixdata(rows: usize, cols: usize, seed: u64) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    Array2::from_shape_fn((rows, cols), |_| rng.random::<f64>() * 100.0 - 50.0)
}

/// Generate correlated data for correlation benchmarks
#[allow(dead_code)]
fn generate_correlateddata(size: usize, correlation: f64, seed: u64) -> (Array1<f64>, Array1<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let x = Array1::from_iter((0..size).map(|_| rng.random::<f64>() * 100.0 - 50.0));

    // Generate correlated y
    let noise = Array1::from_iter((0..size).map(|_| rng.random::<f64>() * 10.0 - 5.0));
    let y = &x * correlation + noise * (1.0 - correlation.abs()).sqrt();

    (x, y)
}

/// Benchmark basic descriptive statistics
#[allow(dead_code)]
fn bench_descriptive_stats(c: &mut Criterion) {
    let mut group = c.benchmark_group("descriptive_statistics");

    for size in [100, 1_000, 10_000, 100_000, 1_000_000].iter() {
        let data = generate_testdata(*size, 42);

        group.throughput(Throughput::Elements(*size as u64));

        // Mean benchmark
        group.bench_with_input(BenchmarkId::new("mean", size), &data, |b, data| {
            b.iter(|| black_box(mean(&data.view()).unwrap()))
        });

        // Variance benchmark
        group.bench_with_input(BenchmarkId::new("variance", size), &data, |b, data| {
            b.iter(|| black_box(var(&data.view(), 0, None).unwrap()))
        });

        // Standard deviation benchmark
        group.bench_with_input(BenchmarkId::new("std_dev", size), &data, |b, data| {
            b.iter(|| black_box(std(&data.view(), 0, None).unwrap()))
        });

        // Skewness benchmark
        group.bench_with_input(BenchmarkId::new("skewness", size), &data, |b, data| {
            b.iter(|| black_box(skew(&data.view(), false, None).unwrap()))
        });

        // Kurtosis benchmark
        group.bench_with_input(BenchmarkId::new("kurtosis", size), &data, |b, data| {
            b.iter(|| black_box(kurtosis(&data.view(), true, false, None).unwrap()))
        });

        // Combined stats benchmark (demonstrates SIMD advantages)
        group.bench_with_input(
            BenchmarkId::new("combined_stats", size),
            &data,
            |b, data| {
                b.iter(|| {
                    let _mean = black_box(mean(&data.view()).unwrap());
                    let _var = black_box(var(&data.view(), 0, None).unwrap());
                    let _skew = black_box(skew(&data.view(), false, None).unwrap());
                    let _kurt = black_box(kurtosis(&data.view(), true, false, None).unwrap());
                })
            },
        );
    }

    group.finish();
}

/// Benchmark quantile computations
#[allow(dead_code)]
fn bench_quantile_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantile_operations");

    for size in [1_000, 10_000, 100_000, 1_000_000].iter() {
        let data = generate_testdata(*size, 42);

        group.throughput(Throughput::Elements(*size as u64));

        // Single quantile
        group.bench_with_input(
            BenchmarkId::new("single_quantile", size),
            &data,
            |b, data| {
                b.iter(|| {
                    black_box(quantile(&data.view(), 0.5, QuantileInterpolation::Linear).unwrap())
                })
            },
        );

        // Median (optimized quantile)
        group.bench_with_input(BenchmarkId::new("median", size), &data, |b, data| {
            b.iter(|| black_box(median(&data.view()).unwrap()))
        });

        // Multiple quantiles
        group.bench_with_input(
            BenchmarkId::new("multiple_quantiles", size),
            &data,
            |b, data| {
                b.iter(|| {
                    let _q25 = black_box(
                        quantile(&data.view(), 0.25, QuantileInterpolation::Linear).unwrap(),
                    );
                    let _q50 = black_box(
                        quantile(&data.view(), 0.5, QuantileInterpolation::Linear).unwrap(),
                    );
                    let _q75 = black_box(
                        quantile(&data.view(), 0.75, QuantileInterpolation::Linear).unwrap(),
                    );
                    let _q95 = black_box(
                        quantile(&data.view(), 0.95, QuantileInterpolation::Linear).unwrap(),
                    );
                })
            },
        );

        // IQR computation
        group.bench_with_input(BenchmarkId::new("iqr", size), &data, |b, data| {
            b.iter(|| {
                let q25 = quantile(&data.view(), 0.25, QuantileInterpolation::Linear).unwrap();
                let q75 = quantile(&data.view(), 0.75, QuantileInterpolation::Linear).unwrap();
                black_box(q75 - q25)
            })
        });
    }

    group.finish();
}

/// Benchmark correlation computations
#[allow(dead_code)]
fn bench_correlation_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("correlation_operations");

    for size in [1_000, 10_000, 100_000, 1_000_000].iter() {
        let (x, y) = generate_correlateddata(*size, 0.7, 42);

        group.throughput(Throughput::Elements(*size as u64));

        // Pearson correlation
        group.bench_with_input(
            BenchmarkId::new("pearson_correlation", size),
            &(x.clone(), y.clone()),
            |b, (x, y)| b.iter(|| black_box(pearsonr(&x.view(), &y.view(), "propagate").unwrap())),
        );

        // Spearman correlation
        group.bench_with_input(
            BenchmarkId::new("spearman_correlation", size),
            &(x.clone(), y.clone()),
            |b, (x, y)| b.iter(|| black_box(spearmanr(&x.view(), &y.view(), "propagate").unwrap())),
        );

        // Kendall tau correlation
        group.bench_with_input(
            BenchmarkId::new("kendall_tau", size),
            &(x.clone(), y.clone()),
            |b, (x, y)| {
                b.iter(|| {
                    black_box(kendalltau(&x.view(), &y.view(), "nan_policy", "method").unwrap())
                })
            },
        );
    }

    // Benchmark correlation matrix computation
    for (rows, cols) in [(100, 10), (1000, 50), (5000, 100)].iter() {
        let data = generate_matrixdata(*rows, *cols, 42);

        group.throughput(Throughput::Elements((rows * cols) as u64));

        group.bench_with_input(
            BenchmarkId::new("correlation_matrix", format!("{}x{}", rows, cols)),
            &data,
            |b, data| b.iter(|| black_box(corrcoef(&data.view(), "pearson").unwrap())),
        );
    }

    group.finish();
}

/// Benchmark linear regression operations
#[allow(dead_code)]
fn bench_regression_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("regression_operations");

    for (n_samples, n_features) in [(1000, 10), (10000, 50), (50000, 100)].iter() {
        let x = generate_matrixdata(*n_samples, *n_features, 42);
        let mut rng = StdRng::seed_from_u64(42);
        let true_coef =
            Array1::from_iter((0..*n_features).map(|_| rng.random::<f64>() * 2.0 - 1.0));
        let noise = Array1::from_iter((0..*n_samples).map(|_| rng.random::<f64>() * 0.1));
        let y = x.dot(&true_coef) + noise;

        group.throughput(Throughput::Elements((n_samples * n_features) as u64));

        // Ordinary Least Squares
        // TODO: Fix LinearRegression struct availability
        // group.bench_with_input(
        //     BenchmarkId::new("ols_regression", format!("{}x{}", n_samples, n_features)),
        //     &(x.clone(), y.clone()),
        //     |b, (x, y)| {
        //         b.iter(|| {
        //             let mut model = LinearRegression::new();
        //             black_box(model.fit(&x.view(), &y.view()).unwrap())
        //         })
        //     },
        // );

        // TODO: Fix RidgeRegression struct availability
        // Ridge regression
        // group.bench_with_input(
        //     BenchmarkId::new("ridge_regression", format!("{}x{}", n_samples, n_features)),
        //     &(x.clone(), y.clone()),
        //     |b, (x, y)| {
        //         b.iter(|| {
        //             let mut model = RidgeRegression::new(1.0);
        //             black_box(model.fit(&x.view(), &y.view()).unwrap())
        //         })
        //     },
        // );

        // TODO: Fix LinearRegression struct availability
        // Prediction benchmark
        // let mut ols_model = LinearRegression::new();
        // let ols_result = ols_model.fit(&x.view(), &y.view()).unwrap();
        let test_x = generate_matrixdata(*n_samples / 10, *n_features, 123);

        // TODO: Comment out until LinearRegression is available
        // group.bench_with_input(
        //     BenchmarkId::new(
        //         "ols_prediction",
        //         format!("{}x{}", n_samples / 10, n_features),
        //     ),
        //     &(ols_result, test_x),
        //     |b, (result, test_x)| b.iter(|| black_box(result.predict(&test_x.view()).unwrap())),
        // );
    }

    group.finish();
}

/// Benchmark statistical tests
#[allow(dead_code)]
fn bench_statistical_tests(c: &mut Criterion) {
    let mut group = c.benchmark_group("statistical_tests");

    for size in [1_000, 10_000, 100_000].iter() {
        let data1 = generate_testdata(*size, 42);
        let data2 = generate_testdata(*size, 123);

        group.throughput(Throughput::Elements(*size as u64));

        // T-test (one sample)
        group.bench_with_input(BenchmarkId::new("ttest_1samp", size), &data1, |b, data| {
            b.iter(|| {
                black_box(
                    ttest_1samp(&data.view(), 0.0, Alternative::TwoSided, "propagate").unwrap(),
                )
            })
        });

        // T-test (independent samples)
        group.bench_with_input(
            BenchmarkId::new("ttest_ind", size),
            &(data1.clone(), data2.clone()),
            |b, (data1, data2)| {
                b.iter(|| {
                    black_box(
                        ttest_ind(
                            &data1.view(),
                            &data2.view(),
                            true,
                            Alternative::TwoSided,
                            "propagate",
                        )
                        .unwrap(),
                    )
                })
            },
        );

        // Mann-Whitney U test
        group.bench_with_input(
            BenchmarkId::new("mannwhitneyu", size),
            &(data1.clone(), data2.clone()),
            |b, (data1, data2)| {
                b.iter(|| {
                    black_box(
                        mann_whitney(&data1.view(), &data2.view(), "two-sided", true).unwrap(),
                    )
                })
            },
        );

        // Kolmogorov-Smirnov test
        group.bench_with_input(
            BenchmarkId::new("ks_2samp", size),
            &(data1.clone(), data2.clone()),
            |b, (data1, data2)| {
                b.iter(|| black_box(ks_2samp(&data1.view(), &data2.view(), "two-sided").unwrap()))
            },
        );

        // Shapiro-Wilk normality test (limited to smaller sizes)
        if *size <= 10_000 {
            group.bench_with_input(BenchmarkId::new("shapiro_wilk", size), &data1, |b, data| {
                b.iter(|| black_box(shapiro(&data.view()).unwrap()))
            });
        }
    }

    group.finish();
}

/// Benchmark distribution operations
#[allow(dead_code)]
fn bench_distribution_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("distribution_operations");

    for size in [1_000, 10_000, 100_000, 1_000_000].iter() {
        let x_values = Array1::linspace(-5.0, 5.0, *size);

        group.throughput(Throughput::Elements(*size as u64));

        // Normal distribution PDF
        group.bench_with_input(
            BenchmarkId::new("normal_pdf", size),
            &x_values,
            |b, x_values| {
                b.iter(|| {
                    let normal = Normal::new(0.0, 1.0).unwrap();
                    for &x in x_values.iter() {
                        black_box(normal.pdf(x));
                    }
                })
            },
        );

        // Normal distribution CDF
        group.bench_with_input(
            BenchmarkId::new("normal_cdf", size),
            &x_values,
            |b, x_values| {
                b.iter(|| {
                    let normal = Normal::new(0.0, 1.0).unwrap();
                    for &x in x_values.iter() {
                        black_box(normal.cdf(x));
                    }
                })
            },
        );

        // Gamma distribution PDF
        group.bench_with_input(
            BenchmarkId::new("gamma_pdf", size),
            &x_values,
            |b, x_values| {
                b.iter(|| {
                    let gamma_dist = Gamma::new(2.0, 1.0, 0.0).unwrap();
                    for &x in x_values.iter() {
                        if x > 0.0 {
                            black_box(gamma_dist.pdf(x));
                        }
                    }
                })
            },
        );

        // Beta distribution PDF
        group.bench_with_input(
            BenchmarkId::new("beta_pdf", size),
            &Array1::linspace(0.001, 0.999, *size),
            |b, x_values| {
                b.iter(|| {
                    let beta_dist = Beta::new(2.0, 3.0, 0.0, 1.0).unwrap();
                    for &x in x_values.iter() {
                        black_box(beta_dist.pdf(x));
                    }
                })
            },
        );
    }

    // Random number generation benchmarks
    for size in [10_000, 100_000, 1_000_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::new("normal_rvs", size), size, |b, &size| {
            b.iter(|| {
                let normal = Normal::new(0.0, 1.0).unwrap();
                let samples: Vec<f64> = (0..size).map(|_| normal.rvs(1).unwrap()[0]).collect();
                black_box(samples)
            })
        });

        group.bench_with_input(BenchmarkId::new("uniform_rvs", size), size, |b, &size| {
            b.iter(|| {
                let uniform = StatsUniform::new(0.0, 1.0).unwrap();
                let samples: Vec<f64> = (0..size).map(|_| uniform.rvs(1).unwrap()[0]).collect();
                black_box(samples)
            })
        });
    }

    group.finish();
}

/// Benchmark memory efficiency and cache performance
#[allow(dead_code)]
fn bench_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_efficiency");

    // Test different data access patterns
    for size in [10_000, 100_000, 1_000_000].iter() {
        let data = generate_testdata(*size, 42);

        group.throughput(Throughput::Elements(*size as u64));

        // Sequential access pattern (cache-friendly)
        group.bench_with_input(
            BenchmarkId::new("sequential_sum", size),
            &data,
            |b, data| {
                b.iter(|| {
                    let mut sum = 0.0;
                    for &x in data.iter() {
                        sum += x;
                    }
                    black_box(sum)
                })
            },
        );

        // SIMD-optimized sum
        group.bench_with_input(BenchmarkId::new("simd_sum", size), &data, |b, data| {
            b.iter(|| black_box(data.sum()))
        });

        // Chunked processing (simulates large dataset handling)
        group.bench_with_input(
            BenchmarkId::new("chunked_processing", size),
            &data,
            |b, data| {
                b.iter(|| {
                    let chunksize = 1000;
                    let mut results = Vec::new();
                    for chunk in data.axis_chunks_iter(ndarray::Axis(0), chunksize) {
                        results.push(chunk.iter().sum::<f64>() / chunk.len() as f64);
                    }
                    black_box(results)
                })
            },
        );
    }

    group.finish();
}

/// Benchmark parallel processing efficiency
#[allow(dead_code)]
fn bench_parallel_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_processing");

    for size in [100_000, 1_000_000, 10_000_000].iter() {
        let data = generate_matrixdata(*size / 1000, 1000, 42);

        group.throughput(Throughput::Elements(*size as u64));

        // Row-wise operations (potentially parallel)
        group.bench_with_input(BenchmarkId::new("row_means", size), &data, |b, data| {
            b.iter(|| black_box(data.mean_axis(Axis(1)).unwrap()))
        });

        // Column-wise operations (potentially parallel)
        group.bench_with_input(BenchmarkId::new("column_means", size), &data, |b, data| {
            b.iter(|| black_box(data.mean_axis(Axis(0)).unwrap()))
        });

        // Element-wise operations (SIMD + parallel)
        group.bench_with_input(
            BenchmarkId::new("matrix_transform", size),
            &data,
            |b, data| b.iter(|| black_box(data.mapv(|x| x.exp().tanh()))),
        );
    }

    group.finish();
}

/// Benchmark numerical stability
#[allow(dead_code)]
fn bench_numerical_stability(c: &mut Criterion) {
    let mut group = c.benchmark_group("numerical_stability");

    // Test with extreme values
    let tiny_values = Array1::from_iter((0..10_000).map(|_| 1e-15));
    let large_values = Array1::from_iter((0..10_000).map(|_| 1e15));
    let mixed_scale = Array1::from_iter((0..10_000).map(|i| if i % 2 == 0 { 1e-10 } else { 1e10 }));

    group.throughput(Throughput::Elements(10_000));

    // Stability of mean computation
    group.bench_function("mean_tiny_values", |b| {
        b.iter(|| black_box(mean(&tiny_values.view()).unwrap()))
    });

    group.bench_function("mean_large_values", |b| {
        b.iter(|| black_box(mean(&large_values.view()).unwrap()))
    });

    group.bench_function("mean_mixed_scale", |b| {
        b.iter(|| black_box(mean(&mixed_scale.view()).unwrap()))
    });

    // Stability of variance computation
    group.bench_function("var_tiny_values", |b| {
        b.iter(|| black_box(var(&tiny_values.view(), 0, None).unwrap()))
    });

    group.bench_function("var_large_values", |b| {
        b.iter(|| black_box(var(&large_values.view(), 0, None).unwrap()))
    });

    group.bench_function("var_mixed_scale", |b| {
        b.iter(|| black_box(var(&mixed_scale.view(), 0, None).unwrap()))
    });

    group.finish();
}

/// Benchmark SIMD vs scalar implementations
#[allow(dead_code)]
fn bench_simd_vs_scalar(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_vs_scalar");

    for size in [1_000, 10_000, 100_000, 1_000_000].iter() {
        let data = generate_testdata(*size, 42);

        group.throughput(Throughput::Elements(*size as u64));

        // Compare SIMD vs scalar for basic operations
        group.bench_with_input(BenchmarkId::new("scalar_sum", size), &data, |b, data| {
            b.iter(|| {
                let mut sum = 0.0;
                for &x in data.iter() {
                    sum += x;
                }
                black_box(sum)
            })
        });

        group.bench_with_input(BenchmarkId::new("ndarray_sum", size), &data, |b, data| {
            b.iter(|| black_box(data.sum()))
        });

        // Variance computation comparison
        group.bench_with_input(
            BenchmarkId::new("scalar_variance", size),
            &data,
            |b, data| {
                b.iter(|| {
                    let n = data.len() as f64;
                    let mean = data.iter().sum::<f64>() / n;
                    let var = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
                    black_box(var)
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("optimized_variance", size),
            &data,
            |b, data| b.iter(|| black_box(var(&data.view(), 0, None).unwrap())),
        );
    }

    group.finish();
}

/// Comprehensive benchmark comparing against baseline implementations
#[allow(dead_code)]
fn bench_comprehensive_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("comprehensive_comparison");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(50);

    // Large-scale realistic workload
    let n_samples = 1_000_000;
    let data = generate_testdata(n_samples, 42);
    let (x, y) = generate_correlateddata(n_samples, 0.8, 42);

    group.throughput(Throughput::Elements(n_samples as u64));

    // Complete descriptive statistics pipeline
    group.bench_function("complete_descriptive_pipeline", |b| {
        b.iter(|| {
            let _mean = black_box(mean(&data.view()).unwrap());
            let _median = black_box(median(&data.view()).unwrap());
            let _var = black_box(var(&data.view(), 0, None).unwrap());
            let _std = black_box(std(&data.view(), 0, None).unwrap());
            let _skew = black_box(skew(&data.view(), false, None).unwrap());
            let _kurt = black_box(kurtosis(&data.view(), true, false, None).unwrap());
            let _q25 =
                black_box(quantile(&data.view(), 0.25, QuantileInterpolation::Linear).unwrap());
            let _q75 =
                black_box(quantile(&data.view(), 0.75, QuantileInterpolation::Linear).unwrap());
            let _min = black_box(data.fold(f64::INFINITY, |acc, &x| acc.min(x)));
            let _max = black_box(data.fold(f64::NEG_INFINITY, |acc, &x| acc.max(x)));
        })
    });

    // Complete correlation analysis pipeline
    group.bench_function("complete_correlation_pipeline", |b| {
        b.iter(|| {
            let _pearson = black_box(pearsonr(&x.view(), &y.view(), "propagate").unwrap());
            let _spearman = black_box(spearmanr(&x.view(), &y.view(), "propagate").unwrap());
            let _kendall =
                black_box(kendalltau(&x.view(), &y.view(), "nan_policy", "method").unwrap());
        })
    });

    // Large matrix operations
    let matrix = generate_matrixdata(10_000, 100, 42);
    group.bench_function("large_matrix_correlation", |b| {
        b.iter(|| black_box(corrcoef(&matrix.view(), "pearson").unwrap()))
    });

    group.finish();
}

// Configure all benchmark groups
criterion_group!(
    benches,
    bench_descriptive_stats,
    bench_quantile_operations,
    bench_correlation_operations,
    bench_regression_operations,
    bench_statistical_tests,
    bench_distribution_operations,
    bench_memory_efficiency,
    bench_parallel_processing,
    bench_numerical_stability,
    bench_simd_vs_scalar,
    bench_comprehensive_comparison
);

criterion_main!(benches);

/// Generate benchmark report comparing against SciPy
#[cfg(test)]
mod scipy_comparison_tests {
    use super::*;
    use std::fs;
    use std::process::Command;

    #[test]
    fn generate_scipy_comparison_report() {
        // Create Python script for SciPy benchmarks
        let python_script = r#"
import numpy as np
import scipy.stats as stats
import time
import json

def benchmark_scipy():
    results = {}
    sizes = [1000, 10000, 100000, 1000000]
    
    for size in sizes:
        np.random.seed(42)
        data = np.random.randn(size)
        x, y = np.random.randn(2, size)
        
        # Time descriptive statistics
        start = time.time()
        mean_val = np.mean(data)
        var_val = np.var(data, ddof=1)
        std_val = np.std(data, ddof=1)
        skew_val = stats.skew(data)
        kurt_val = stats.kurtosis(data)
        scipy_time = time.time() - start
        
        results[f"descriptive_{size}"] = {
            "scipy_time": scipy_time,
            "operations": ["mean", "var", "std", "skew", "kurtosis"]
        }
        
        # Time correlation
        start = time.time()
        pearson_r_ = stats.pearsonr(x, y)
        spearman_r_ = stats.spearmanr(x, y)
        scipy_corr_time = time.time() - start
        
        results[f"correlation_{size}"] = {
            "scipy_time": scipy_corr_time,
            "operations": ["pearson", "spearman"]
        }
        
        # Time statistical tests
        start = time.time()
        t_stat, p_val = stats.ttest_1samp(data, 0)
        mw_stat, mw_p = stats.mannwhitneyu(x, y)
        scipy_test_time = time.time() - start
        
        results[f"tests_{size}"] = {
            "scipy_time": scipy_test_time,
            "operations": ["ttest_1samp", "mannwhitneyu"]
        }
    
    with open("scipy_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("SciPy benchmarks completed")

if __name__ == "__main__":
    benchmark_scipy()
"#;

        fs::write("scipy_benchmark.py", python_script).unwrap();

        // Run SciPy benchmarks
        let output = Command::new("python").arg("scipy_benchmark.py").output();

        match output {
            Ok(result) => {
                if result.status.success() {
                    println!("SciPy benchmarks completed successfully");

                    // Load and compare results
                    if let Ok(scipy_results) = fs::read_to_string("scipy_benchmark_results.json") {
                        println!("SciPy benchmark results:\n{}", scipy_results);
                    }
                } else {
                    println!(
                        "SciPy benchmark failed: {}",
                        String::from_utf8_lossy(&result.stderr)
                    );
                }
            }
            Err(e) => {
                println!("Could not run Python/SciPy benchmarks: {}", e);
            }
        }

        // Clean up
        let _ = fs::remove_file("scipy_benchmark.py");
        let _ = fs::remove_file("scipy_benchmark_results.json");
    }
}
