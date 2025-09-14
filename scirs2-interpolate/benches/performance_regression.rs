//! Performance regression detection benchmarks
//!
//! This benchmark suite is designed for continuous integration to detect
//! performance regressions in critical interpolation operations. It focuses
//! on stable, reproducible measurements across different operations and scales.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ndarray::{Array1, Array2};
use scirs2_interpolate::{
    advanced::{
        enhanced_rbf::{EnhancedRBFInterpolator, KernelWidthStrategy},
        kriging::{CovarianceFunction, KrigingInterpolator},
        rbf::{RBFInterpolator, RBFKernel},
    },
    bspline::{BSpline, ExtrapolateMode},
    cubic_interpolate, linear_interpolate, pchip_interpolate,
    spline::{CubicSpline, SplineBoundaryCondition},
    traits::{Interpolator, SplineInterpolator},
};
use std::hint::black_box;
use std::time::Duration;

// Fixed seeds for reproducible benchmarks
const BENCHMARK_SEED: u64 = 42;

/// Generate reproducible test data for regression testing
#[allow(dead_code)]
fn generate_regression_data_1d(n: usize) -> (Array1<f64>, Array1<f64>) {
    let x = Array1::linspace(0.0, 10.0, n);
    let y = x.mapv(|xi: f64| {
        // Deterministic function for reproducible results
        (xi * 0.5).sin() + 0.1 * xi + 0.05 * (3.0 * xi).cos() + 0.02 * (xi * xi)
    });
    (x, y)
}

/// Generate reproducible 2D test data
#[allow(dead_code)]
fn generate_regression_data_2d(n: usize) -> (Array2<f64>, Array1<f64>) {
    let mut points = Array2::zeros((n, 2));
    let mut values = Array1::zeros(n);

    let sqrt_n = (n as f64).sqrt() as usize;
    for i in 0..sqrt_n {
        for j in 0..sqrt_n {
            let idx = i * sqrt_n + j;
            if idx < n {
                let x = i as f64 / (sqrt_n - 1) as f64 * 10.0;
                let y = j as f64 / (sqrt_n - 1) as f64 * 10.0;

                points[[idx, 0]] = x;
                points[[idx, 1]] = y;
                values[idx] = (x * 0.1).sin() * (y * 0.1).cos() + 0.05 * x + 0.02 * y;
            }
        }
    }

    (points, values)
}

/// Core 1D interpolation methods regression test
#[allow(dead_code)]
fn bench_core_1d_regression(c: &mut Criterion) {
    let mut group = c.benchmark_group("core_1d_regression");
    group.measurement_time(Duration::from_secs(5));

    // Standard test case for regression detection
    let n_data = 1000;
    let n_queries = 500;
    let (x_data, y_data) = generate_regression_data_1d(n_data);
    let x_queries = Array1::linspace(0.5, 9.5, n_queries);

    group.throughput(Throughput::Elements(n_queries as u64));

    group.bench_function("linear_interpolation", |b| {
        b.iter(|| {
            black_box(linear_interpolate(
                black_box(&x_data.view()),
                black_box(&y_data.view()),
                black_box(&x_queries.view()),
            ))
        })
    });

    group.bench_function("cubic_interpolation", |b| {
        b.iter(|| {
            black_box(cubic_interpolate(
                black_box(&x_data.view()),
                black_box(&y_data.view()),
                black_box(&x_queries.view()),
            ))
        })
    });

    group.bench_function("pchip_interpolation", |b| {
        b.iter(|| {
            black_box(pchip_interpolate(
                black_box(&x_data.view()),
                black_box(&y_data.view()),
                black_box(&x_queries.view()),
                black_box(false), // monotonic parameter
            ))
        })
    });

    group.finish();
}

/// Spline methods regression test
#[allow(dead_code)]
fn bench_spline_regression(c: &mut Criterion) {
    let mut group = c.benchmark_group("spline_regression");
    group.measurement_time(Duration::from_secs(5));

    let n_data = 500;
    let n_queries = 200;
    let (x_data, y_data) = generate_regression_data_1d(n_data);
    let x_queries = Array1::linspace(0.5, 9.5, n_queries);

    // Test cubic spline construction and evaluation
    group.throughput(Throughput::Elements(n_queries as u64));

    group.bench_function("cubic_spline_construction", |b| {
        b.iter(|| {
            black_box(CubicSpline::new(
                black_box(&x_data.view()),
                black_box(&y_data.view()),
            ))
        })
    });

    // Pre-construct spline for evaluation tests
    let spline = CubicSpline::new(&x_data.view(), &y_data.view()).unwrap();

    group.bench_function("cubic_spline_evaluation", |b| {
        b.iter(|| {
            for &x in x_queries.iter() {
                black_box(spline.evaluate(black_box(x)));
            }
        })
    });

    group.bench_function("cubic_spline_derivatives", |b| {
        b.iter(|| {
            for &x in x_queries.iter() {
                black_box(spline.derivative_n(black_box(x), black_box(1)));
            }
        })
    });

    group.bench_function("cubic_spline_integration", |b| {
        b.iter(|| {
            for i in 0..x_queries.len() - 1 {
                black_box(spline.integrate(black_box(x_queries[i]), black_box(x_queries[i + 1])));
            }
        })
    });

    group.finish();
}

/// B-spline regression test
#[allow(dead_code)]
fn bench_bspline_regression(c: &mut Criterion) {
    let mut group = c.benchmark_group("bspline_regression");
    group.measurement_time(Duration::from_secs(5));

    // Standard B-spline test case
    let degree = 3;
    let n_coeffs = 100;
    let n_queries = 300;

    let knots = Array1::linspace(0.0, 10.0, n_coeffs + degree + 1);
    let coeffs = Array1::linspace(-1.0, 1.0, n_coeffs);
    let queries = Array1::linspace(0.5, 9.5, n_queries);

    let spline = BSpline::new(
        &knots.view(),
        &coeffs.view(),
        degree,
        ExtrapolateMode::Extrapolate,
    )
    .unwrap();

    group.throughput(Throughput::Elements(n_queries as u64));

    group.bench_function("bspline_evaluation", |b| {
        b.iter(|| {
            for &x in queries.iter() {
                black_box(spline.evaluate(black_box(x)));
            }
        })
    });

    group.bench_function("bspline_derivative", |b| {
        b.iter(|| {
            for &x in queries.iter() {
                black_box(spline.derivative(black_box(x), black_box(1)));
            }
        })
    });

    group.bench_function("bspline_batch_evaluation", |b| {
        b.iter(|| black_box(spline.evaluate_batch_fast(black_box(&queries.view()))))
    });

    group.finish();
}

/// RBF methods regression test
#[allow(dead_code)]
fn bench_rbf_regression(c: &mut Criterion) {
    let mut group = c.benchmark_group("rbf_regression");
    group.measurement_time(Duration::from_secs(8));

    let n_data = 200;
    let n_queries = 100;
    let (points, values) = generate_regression_data_2d(n_data);
    let (query_points_, _) = generate_regression_data_2d(n_queries);

    group.throughput(Throughput::Elements(n_queries as u64));

    // Test basic RBF interpolation
    group.bench_function("rbf_gaussian_construction", |b| {
        b.iter(|| {
            black_box(RBFInterpolator::new(
                black_box(&points.view()),
                black_box(&values.view()),
                black_box(RBFKernel::Gaussian),
                black_box(1.0),
            ))
        })
    });

    let rbf =
        RBFInterpolator::new(&points.view(), &values.view(), RBFKernel::Gaussian, 1.0).unwrap();

    group.bench_function("rbf_gaussian_evaluation", |b| {
        b.iter(|| black_box(rbf.evaluate(black_box(&query_points_.view()))))
    });

    // Test enhanced RBF (commented out due to API incompatibility)
    // group.bench_function("enhanced_rbf_construction", |b| {
    //     b.iter(|| {
    //         black_box(EnhancedRBFInterpolator::new(
    //             black_box(&points.view()),
    //             black_box(&values.view()),
    //             black_box(scirs2_interpolate::advanced::enhanced_rbf::KernelType::Gaussian),
    //             black_box(KernelWidthStrategy::MeanDistance),
    //         ))
    //     })
    // });

    group.finish();
}

/// Kriging regression test
#[allow(dead_code)]
fn bench_kriging_regression(c: &mut Criterion) {
    let mut group = c.benchmark_group("kriging_regression");
    group.measurement_time(Duration::from_secs(10));

    let n_data = 100; // Smaller for Kriging due to O(n³) complexity
    let n_queries = 50;
    let (points, values) = generate_regression_data_2d(n_data);
    let (query_points_, _) = generate_regression_data_2d(n_queries);

    group.throughput(Throughput::Elements(n_queries as u64));

    // Kriging construction commented out due to API incompatibility
    // group.bench_function("kriging_construction", |b| {
    //     b.iter(|| {
    //         black_box(KrigingInterpolator::new(
    //             black_box(&points.view()),
    //             black_box(&values.view()),
    //             black_box(CovarianceFunction::Exponential),
    //             black_box(1.0),
    //             black_box(0.1),
    //             black_box(1.0),
    //             black_box(0.1),
    //         ))
    //     })
    // });

    // Kriging interpolator commented out due to API incompatibility
    // let kriging = KrigingInterpolator::new(
    //     &points.view(),
    //     &values.view(),
    //     CovarianceFunction::Exponential,
    //     1.0,
    //     0.1,
    //     1.0,
    //     0.1,
    // )
    // .unwrap();

    // group.bench_function("kriging_evaluation", |b| {
    //     b.iter(|| black_box(kriging.evaluate(black_box(&query_points_.view()))))
    // });

    group.finish();
}

/// Memory allocation regression test
#[allow(dead_code)]
fn bench_memory_regression(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_regression");
    group.measurement_time(Duration::from_secs(5));

    // Test memory allocation patterns for different scales
    let scales = [100, 500, 1000];

    for &scale in &scales {
        let (x_data, y_data) = generate_regression_data_1d(scale);
        let x_queries = Array1::linspace(0.5, 9.5, scale / 2);

        group.throughput(Throughput::Elements(scale as u64));
        group.bench_with_input(
            BenchmarkId::new("data_generation", scale),
            &scale,
            |b, &s| b.iter(|| black_box(generate_regression_data_1d(s))),
        );

        group.bench_with_input(
            BenchmarkId::new("spline_memory_usage", scale),
            &scale,
            |b, _| {
                b.iter(|| {
                    let spline = black_box(CubicSpline::new(&x_data.view(), &y_data.view()));
                    black_box(spline)
                })
            },
        );
    }

    group.finish();
}

/// Performance consistency test
#[allow(dead_code)]
fn bench_performance_consistency(c: &mut Criterion) {
    let mut group = c.benchmark_group("performance_consistency");
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(50); // More samples for consistency measurement

    // Test the same operation multiple times to detect inconsistencies
    let (x_data, y_data) = generate_regression_data_1d(500);
    let x_queries = Array1::linspace(0.5, 9.5, 200);

    group.bench_function("consistency_linear", |b| {
        b.iter(|| {
            black_box(linear_interpolate(
                black_box(&x_data.view()),
                black_box(&y_data.view()),
                black_box(&x_queries.view()),
            ))
        })
    });

    group.bench_function("consistency_cubic", |b| {
        b.iter(|| {
            black_box(cubic_interpolate(
                black_box(&x_data.view()),
                black_box(&y_data.view()),
                black_box(&x_queries.view()),
            ))
        })
    });

    group.finish();
}

criterion_group!(
    performance_regression,
    bench_core_1d_regression,
    bench_spline_regression,
    bench_bspline_regression,
    bench_rbf_regression,
    bench_kriging_regression,
    bench_memory_regression,
    bench_performance_consistency
);

criterion_main!(performance_regression);
