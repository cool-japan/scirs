//! Comprehensive benchmarks for scirs2-special functions
//!
//! This module provides extensive benchmarks covering all major function families
//! and includes utilities for comparing with SciPy performance data.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use scirs2_special::{
    // Airy functions
    ai,
    // Gamma functions
    beta,
    bi,
    dawsn,
    digamma,
    // Error functions
    erf,
    erfc,
    erfcx,
    erfi,
    gamma,
    gammaln,
    // Bessel functions
    i0,
    j0,
    j1,
    jn,
    jv,
    // Lambert W
    lambert_w_real,
    polygamma,
    shichi,
    sici,
    spence,
    spherical_jn,
    wofz,
};
use std::hint::black_box;
use std::path::Path;

/// Check if SciPy benchmark results exist
#[allow(dead_code)]
fn scipy_results_exist() -> bool {
    Path::new("benches/scipy_benchmark_results.json").exists()
}

#[allow(dead_code)]
fn bench_bessel_comprehensive(c: &mut Criterion) {
    let mut group = c.benchmark_group("bessel_comprehensive");

    // j0 benchmarks with SciPy comparison
    group.bench_function("j0_small", |b| {
        b.iter(|| {
            for i in 0..100 {
                let x = i as f64 * 0.01;
                black_box(j0(black_box(x)));
            }
        })
    });

    group.bench_function("j0_medium", |b| {
        b.iter(|| {
            for i in 0..100 {
                let x = i as f64 * 0.1 + 10.0;
                black_box(j0(black_box(x)));
            }
        })
    });

    group.bench_function("j0_large", |b| {
        b.iter(|| {
            for i in 0..100 {
                let x = i as f64 * 10.0 + 100.0;
                black_box(j0(black_box(x)));
            }
        })
    });

    // j1 benchmarks
    group.bench_function("j1_small", |b| {
        b.iter(|| {
            for i in 0..100 {
                let x = i as f64 * 0.01;
                black_box(j1(black_box(x)));
            }
        })
    });

    // jn with different orders
    for n in [2, 5, 10, 20] {
        group.bench_with_input(BenchmarkId::new("jn_order", n), &n, |b, &n| {
            b.iter(|| {
                for i in 0..50 {
                    let x = i as f64 * 0.2 + 5.0;
                    black_box(jn(black_box(n), black_box(x)));
                }
            })
        });
    }

    // jv with different orders
    for v in [0.0, 1.0, 2.0, 0.5, 1.5, 2.5] {
        group.bench_with_input(
            BenchmarkId::new("jv_order", format!("{v:.1}")),
            &v,
            |b, &v| {
                b.iter(|| {
                    for i in 0..50 {
                        let x = i as f64 * 0.2 + 5.0;
                        black_box(jv(black_box(v), black_box(x)));
                    }
                })
            },
        );
    }

    // Modified Bessel
    group.bench_function("i0_small", |b| {
        b.iter(|| {
            for i in 0..100 {
                let x = i as f64 * 0.01;
                black_box(i0(black_box(x)));
            }
        })
    });

    group.bench_function("i0_medium", |b| {
        b.iter(|| {
            for i in 0..100 {
                let x = i as f64 * 0.1 + 1.0;
                black_box(i0(black_box(x)));
            }
        })
    });

    // Spherical Bessel
    group.bench_function("spherical_j0", |b| {
        b.iter(|| {
            for i in 0..100 {
                let x = i as f64 * 0.1 + 1.0;
                black_box(spherical_jn(black_box(0), black_box(x)));
            }
        })
    });

    group.finish();
}

#[allow(dead_code)]
fn bench_gamma_comprehensive(c: &mut Criterion) {
    let mut group = c.benchmark_group("gamma_comprehensive");

    // Gamma function
    group.bench_function("gamma", |b| {
        b.iter(|| {
            for i in 0..100 {
                let x = i as f64 * 0.1 + 1.0;
                black_box(gamma(black_box(x)));
            }
        })
    });

    // Log gamma
    group.bench_function("gammaln", |b| {
        b.iter(|| {
            for i in 0..100 {
                let x = i as f64 * 0.1 + 1.0;
                black_box(gammaln(black_box(x)));
            }
        })
    });

    // Digamma
    group.bench_function("digamma", |b| {
        b.iter(|| {
            for i in 0..100 {
                let x = i as f64 * 0.1 + 1.0;
                black_box(digamma(black_box(x)));
            }
        })
    });

    // Beta function
    group.bench_function("beta", |b| {
        b.iter(|| {
            for i in 0..10 {
                for j in 0..10 {
                    let a = i as f64 * 0.1 + 1.0;
                    let b = j as f64 * 0.1 + 1.0;
                    black_box(beta(black_box(a), black_box(b)));
                }
            }
        })
    });

    group.finish();
}

#[allow(dead_code)]
fn bench_error_functions(c: &mut Criterion) {
    let mut group = c.benchmark_group("error_functions");

    // Error function
    group.bench_function("erf", |b| {
        b.iter(|| {
            for i in -50..=50 {
                let x = i as f64 * 0.1;
                black_box(erf(black_box(x)));
            }
        })
    });

    // Complementary error function
    group.bench_function("erfc", |b| {
        b.iter(|| {
            for i in -50..=50 {
                let x = i as f64 * 0.1;
                black_box(erfc(black_box(x)));
            }
        })
    });

    group.finish();
}

#[allow(dead_code)]
fn bench_airy_functions(c: &mut Criterion) {
    let mut group = c.benchmark_group("airy_functions");

    // Airy Ai
    group.bench_function("airy_ai", |b| {
        b.iter(|| {
            for i in -100..=100 {
                let x = i as f64 * 0.1;
                black_box(ai(black_box(x)));
            }
        })
    });

    // Airy Bi
    group.bench_function("airy_bi", |b| {
        b.iter(|| {
            for i in -100..=100 {
                let x = i as f64 * 0.1;
                black_box(bi(black_box(x)));
            }
        })
    });

    group.finish();
}

#[allow(dead_code)]
fn bench_lambert_w(c: &mut Criterion) {
    let mut group = c.benchmark_group("lambert_w");

    group.bench_function("lambertw", |b| {
        b.iter(|| {
            for i in 0..100 {
                let x = i as f64 * 0.1 + 0.1;
                let _ = black_box(lambert_w_real(black_box(x), 1e-8));
            }
        })
    });

    group.finish();
}

#[allow(dead_code)]
fn bench_array_like_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("array_like_operations");

    // Simulate array operations - small arrays
    let small_values: Vec<f64> = (0..100).map(|i| i as f64 * 0.1 + 1.0).collect();
    group.bench_function("gamma_small_array", |b| {
        b.iter(|| {
            let results: Vec<f64> = small_values.iter().map(|&x| gamma(black_box(x))).collect();
            black_box(results);
        })
    });

    // Simulate array operations - large arrays
    let large_values: Vec<f64> = (0..10000).map(|i| i as f64 * 0.001 + 1.0).collect();
    group.bench_function("gamma_large_array", |b| {
        b.iter(|| {
            let results: Vec<f64> = large_values.iter().map(|&x| gamma(black_box(x))).collect();
            black_box(results);
        })
    });

    group.finish();
}

#[allow(dead_code)]
fn bench_array_vs_scalar(c: &mut Criterion) {
    let mut group = c.benchmark_group("array_vs_scalar");

    let values: Vec<f64> = (0..1000).map(|i| i as f64 * 0.01 + 1.0).collect();

    // Scalar operations
    group.bench_function("scalar_gamma_1000", |b| {
        b.iter(|| {
            for &x in &values {
                black_box(gamma(black_box(x)));
            }
        })
    });

    // Array-like operations with collect
    group.bench_function("array_like_gamma_1000", |b| {
        b.iter(|| {
            let results: Vec<f64> = values.iter().map(|&x| gamma(black_box(x))).collect();
            black_box(results);
        })
    });

    group.finish();
}

#[allow(dead_code)]
fn bench_memory_usage(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage");

    // Test different array sizes to understand memory scaling
    for size in [100, 1000, 10000, 100000] {
        let values: Vec<f64> = (0..size).map(|i| i as f64 * 0.001 + 1.0).collect();

        group.bench_with_input(
            BenchmarkId::new("gamma_array_like", size),
            &values,
            |b, vals| {
                b.iter(|| {
                    let results: Vec<f64> = vals.iter().map(|&x| gamma(black_box(x))).collect();
                    black_box(results);
                })
            },
        );
    }

    group.finish();
}

#[allow(dead_code)]
fn bench_advanced_functions(c: &mut Criterion) {
    let mut group = c.benchmark_group("advanced_functions");

    // Dawson's integral benchmarks
    group.bench_function("dawsn", |b| {
        b.iter(|| {
            for i in -50..51 {
                let x = i as f64 * 0.1;
                black_box(dawsn(black_box(x)));
            }
        })
    });

    // Polygamma function (trigamma)
    group.bench_function("polygamma", |b| {
        b.iter(|| {
            for i in 0..100 {
                let x = i as f64 * 0.1 + 1.0;
                black_box(polygamma(black_box(1), black_box(x)));
            }
        })
    });

    // Scaled error functions
    group.bench_function("erfcx", |b| {
        b.iter(|| {
            for i in 0..100 {
                let x = i as f64 * 0.1;
                black_box(erfcx(black_box(x)));
            }
        })
    });

    group.bench_function("erfi", |b| {
        b.iter(|| {
            for i in 0..100 {
                let x = i as f64 * 0.1;
                black_box(erfi(black_box(x)));
            }
        })
    });

    // Faddeeva function
    group.bench_function("wofz", |b| {
        b.iter(|| {
            for i in 0..100 {
                let x = i as f64 * 0.1;
                black_box(wofz(black_box(x)));
            }
        })
    });

    // Sine and cosine integrals
    group.bench_function("sici", |b| {
        b.iter(|| {
            for i in 1..101 {
                let x = i as f64 * 0.1;
                let _ = black_box(sici(black_box(x)));
            }
        })
    });

    group.bench_function("shichi", |b| {
        b.iter(|| {
            for i in 1..101 {
                let x = i as f64 * 0.1;
                let _ = black_box(shichi(black_box(x)));
            }
        })
    });

    // Spence function (dilogarithm)
    group.bench_function("spence", |b| {
        b.iter(|| {
            for i in 1..51 {
                let x = i as f64 * 0.1 + 0.1;
                let _ = black_box(spence(black_box(x)));
            }
        })
    });

    group.finish();
}

criterion_group!(
    comprehensive_benches,
    bench_bessel_comprehensive,
    bench_gamma_comprehensive,
    bench_error_functions,
    bench_airy_functions,
    bench_lambert_w,
    bench_array_like_operations,
    bench_array_vs_scalar,
    bench_memory_usage
);
criterion_main!(comprehensive_benches);
