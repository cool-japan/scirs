use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ndarray::Array1;
use scirs2_interpolate::bspline::{
    generate_knots, make_interp_bspline, make_lsq_bspline, BSpline, ExtrapolateMode,
};
use scirs2_interpolate::cache::{BSplineCache, CacheConfig, CachedBSpline};
use scirs2_interpolate::fast_bspline::make_fast_bspline_evaluator;
use std::hint::black_box;

#[allow(dead_code)]
fn generate_test_data(n: usize) -> (Array1<f64>, Array1<f64>) {
    let x = Array1::linspace(0.0, 10.0, n);
    let y = x.mapv(|xi| (xi * 0.5_f64).sin() + 0.1 * xi + 0.05 * (3.0 * xi).cos());
    (x, y)
}

#[allow(dead_code)]
fn generate_bspline(degree: usize, ncoeffs: usize) -> BSpline<f64> {
    let knots = Array1::linspace(0.0, 10.0, ncoeffs + degree + 1);
    let _coeffs = Array1::linspace(-1.0, 1.0, ncoeffs);
    BSpline::new(
        &knots.view(),
        &_coeffs.view(),
        degree,
        ExtrapolateMode::Extrapolate,
    )
    .unwrap()
}

#[allow(dead_code)]
fn bench_bspline_evaluation(c: &mut Criterion) {
    let mut group = c.benchmark_group("bspline_evaluation");

    for degree in [1, 2, 3, 5].iter() {
        for n_coeffs in [50, 100, 500, 1000].iter() {
            let spline = generate_bspline(*degree, *n_coeffs);
            let queries = Array1::linspace(0.5, 9.5, 1000);

            group.throughput(Throughput::Elements(queries.len() as u64));
            group.bench_with_input(
                BenchmarkId::new(format!("degree_{}_coeffs_{}", degree, n_coeffs), n_coeffs),
                n_coeffs,
                |b, _| {
                    b.iter(|| {
                        for &query in queries.iter() {
                            let _ = black_box(spline.evaluate(black_box(query)));
                        }
                    });
                },
            );
        }
    }

    group.finish();
}

#[allow(dead_code)]
fn bench_fast_bspline_vs_standard(c: &mut Criterion) {
    let mut group = c.benchmark_group("fast_vs_standard_bspline");

    let spline = generate_bspline(3, 500);
    let fast_evaluator = make_fast_bspline_evaluator(&spline);
    let queries = Array1::linspace(0.5, 9.5, 1000);

    group.throughput(Throughput::Elements(queries.len() as u64));

    group.bench_function("standard_evaluation", |b| {
        b.iter(|| {
            for &query in queries.iter() {
                let _ = black_box(spline.evaluate(black_box(query)));
            }
        });
    });

    group.bench_function("fast_evaluation", |b| {
        b.iter(|| {
            for &query in queries.iter() {
                let _ = black_box(fast_evaluator.evaluate_fast(black_box(query)));
            }
        });
    });

    group.bench_function("fast_array_evaluation", |b| {
        b.iter(|| {
            let _ = black_box(fast_evaluator.evaluate_array_fast(black_box(&queries.view())));
        });
    });

    group.finish();
}

#[allow(dead_code)]
fn bench_cached_bspline(c: &mut Criterion) {
    let mut group = c.benchmark_group("cached_bspline");

    let knots = Array1::linspace(0.0, 10.0, 100);
    let coeffs = Array1::linspace(-1.0, 1.0, 95);

    // Test different cache configurations
    let configs = [
        (
            "no_cache",
            CacheConfig {
                track_stats: false,
                ..Default::default()
            },
        ),
        (
            "small_cache",
            CacheConfig {
                max_basis_cache_size: 256,
                max_matrix_cache_size: 16,
                track_stats: true,
                ..Default::default()
            },
        ),
        (
            "large_cache",
            CacheConfig {
                max_basis_cache_size: 2048,
                max_matrix_cache_size: 128,
                track_stats: true,
                ..Default::default()
            },
        ),
    ];

    for (config_name, cache_config) in configs.iter() {
        let cache = BSplineCache::new(cache_config.clone());
        let mut cached_spline = CachedBSpline::new(
            &knots.view(),
            &coeffs.view(),
            3,
            ExtrapolateMode::Extrapolate,
            cache,
        )
        .unwrap();

        let queries = Array1::linspace(0.5, 9.5, 1000);

        group.throughput(Throughput::Elements(queries.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("config", config_name),
            config_name,
            |b, _| {
                b.iter(|| {
                    for &query in queries.iter() {
                        let _ = black_box(cached_spline.evaluate_cached(black_box(query)));
                    }
                });
            },
        );
    }

    group.finish();
}

#[allow(dead_code)]
fn bench_bspline_derivatives(c: &mut Criterion) {
    let mut group = c.benchmark_group("bspline_derivatives");

    let spline = generate_bspline(3, 500);
    let queries = Array1::linspace(0.5, 9.5, 500);

    for order in [1, 2, 3].iter() {
        group.throughput(Throughput::Elements(queries.len() as u64));
        group.bench_with_input(BenchmarkId::new("order", order), order, |b, _| {
            b.iter(|| {
                for &query in queries.iter() {
                    let _ = black_box(spline.derivative(black_box(query), black_box(*order)));
                }
            });
        });
    }

    group.finish();
}

#[allow(dead_code)]
fn bench_bspline_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("bspline_construction");

    // Test interpolating spline construction
    for data_size in [50, 100, 500, 1000].iter() {
        let (x, y) = generate_test_data(*data_size);

        group.bench_with_input(
            BenchmarkId::new("interpolating", data_size),
            data_size,
            |b, _| {
                b.iter(|| {
                    let _ = black_box(make_interp_bspline(
                        black_box(&x.view()),
                        black_box(&y.view()),
                        black_box(3), // cubic
                        black_box(ExtrapolateMode::Extrapolate),
                    ));
                });
            },
        );
    }

    // Test least-squares spline construction
    for data_size in [100, 500, 1000, 2000].iter() {
        let (x, y) = generate_test_data(*data_size);
        let knots = generate_knots(&x.view(), 3, "uniform").unwrap();

        group.bench_with_input(
            BenchmarkId::new("least_squares", data_size),
            data_size,
            |b, _| {
                b.iter(|| {
                    let _ = black_box(make_lsq_bspline(
                        black_box(&x.view()),
                        black_box(&y.view()),
                        black_box(&knots.view()),
                        black_box(3),    // cubic
                        black_box(None), // no weights
                        black_box(ExtrapolateMode::Extrapolate),
                    ));
                });
            },
        );
    }

    group.finish();
}

#[allow(dead_code)]
fn bench_knot_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("knot_generation");

    let styles = ["uniform", "average", "clamped"];

    for style in styles.iter() {
        for data_size in [100, 500, 1000, 5000].iter() {
            let (x_, _) = generate_test_data(*data_size);

            group.bench_with_input(
                BenchmarkId::new(format!("{}_style", style), data_size),
                data_size,
                |b, _| {
                    b.iter(|| {
                        let _ = black_box(generate_knots(
                            black_box(&x_.view()),
                            black_box(3),
                            black_box(*style),
                        ));
                    });
                },
            );
        }
    }

    group.finish();
}

#[allow(dead_code)]
fn bench_bspline_integration(c: &mut Criterion) {
    let mut group = c.benchmark_group("bspline_integration");

    for n_coeffs in [50, 100, 500, 1000].iter() {
        let spline = generate_bspline(3, *n_coeffs);

        group.bench_with_input(BenchmarkId::new("coeffs", n_coeffs), n_coeffs, |b, _| {
            b.iter(|| {
                let _ = black_box(spline.integrate(black_box(1.0), black_box(9.0)));
            });
        });
    }

    group.finish();
}

#[allow(dead_code)]
fn bench_basis_element_evaluation(c: &mut Criterion) {
    let mut group = c.benchmark_group("basis_element_evaluation");

    for degree in [1, 2, 3, 5].iter() {
        let knots = Array1::linspace(0.0, 10.0, 100);
        let queries = Array1::linspace(0.5, 9.5, 500);

        // Create basis element
        let basis =
            BSpline::basis_element(*degree, 10, &knots.view(), ExtrapolateMode::Extrapolate)
                .unwrap();

        group.throughput(Throughput::Elements(queries.len() as u64));
        group.bench_with_input(BenchmarkId::new("degree", degree), degree, |b, _| {
            b.iter(|| {
                for &query in queries.iter() {
                    let _ = black_box(basis.evaluate(black_box(query)));
                }
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_bspline_evaluation,
    bench_fast_bspline_vs_standard,
    bench_cached_bspline,
    bench_bspline_derivatives,
    bench_bspline_construction,
    bench_knot_generation,
    bench_bspline_integration,
    bench_basis_element_evaluation
);
criterion_main!(benches);
