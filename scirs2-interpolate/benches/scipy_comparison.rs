use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ndarray::{Array1, Array2};
use std::hint::black_box;
use std::time::Duration;

// Import all scirs2 interpolation methods
use scirs2_interpolate::{
    advanced::{
        enhanced_kriging::EnhancedKrigingBuilder,
        enhanced_rbf::{EnhancedRBFInterpolator, KernelWidthStrategy},
        fast_kriging::{FastKrigingBuilder, FastKrigingMethod},
        kriging::{CovarianceFunction, KrigingInterpolator},
        rbf::{RBFInterpolator, RBFKernel},
        thinplate::ThinPlateSpline,
    },
    bspline::{BSpline, ExtrapolateMode},
    cubic_interpolate,
    interp1d::monotonic::{MonotonicInterpolator, MonotonicMethod},
    linear_interpolate,
    local::mls::{MovingLeastSquares, PolynomialBasis, WeightFunction},
    pchip_interpolate,
    spline::{CubicSpline, SplineBoundaryCondition},
};

#[cfg(feature = "scipy-comparison")]
use pyo3::prelude::*;

/// Generate test data for 1D interpolation
#[allow(dead_code)]
fn generate_1d_data(n: usize, noise: bool) -> (Array1<f64>, Array1<f64>) {
    let x = Array1::linspace(0.0, 10.0, n);
    let y = if noise {
        x.mapv(|xi: f64| {
            (xi * 0.5).sin() + 0.1 * xi + 0.05 * (3.0 * xi).cos() + 0.01 * rand::random::<f64>()
        })
    } else {
        x.mapv(|xi: f64| (xi * 0.5).sin() + 0.1 * xi + 0.05 * (3.0 * xi).cos())
    };
    (x, y)
}

/// Generate test data for 2D interpolation
#[allow(dead_code)]
fn generate_2d_data(n: usize, noise: bool) -> (Array2<f64>, Array1<f64>) {
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
                if noise {
                    values[idx] += 0.01 * rand::random::<f64>();
                }
            }
        }
    }

    (points, values)
}

/// Generate query points for evaluation
#[allow(dead_code)]
fn generate_query_points_1d(n: usize) -> Array1<f64> {
    Array1::linspace(0.1, 9.9, n)
}

#[allow(dead_code)]
fn generate_query_points_2d(n: usize) -> Array2<f64> {
    let sqrt_n = (n as f64).sqrt() as usize;
    let mut queries = Array2::zeros((n, 2));

    for i in 0..sqrt_n {
        for j in 0..sqrt_n {
            let idx = i * sqrt_n + j;
            if idx < n {
                queries[[idx, 0]] = i as f64 / (sqrt_n - 1) as f64 * 9.8 + 0.1;
                queries[[idx, 1]] = j as f64 / (sqrt_n - 1) as f64 * 9.8 + 0.1;
            }
        }
    }

    queries
}

/// Benchmark 1D linear interpolation
#[allow(dead_code)]
fn bench_linear_1d_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("linear_1d_comparison");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));

    for &n_points in &[100, 1000, 10_000, 100_000] {
        let (x, y) = generate_1d_data(n_points, false);
        let queries = generate_query_points_1d(1000);

        // Benchmark scirs2
        group.throughput(Throughput::Elements(queries.len() as u64));
        group.bench_with_input(BenchmarkId::new("scirs2", n_points), &n_points, |b, _| {
            b.iter(|| black_box(linear_interpolate(&x.view(), &y.view(), &queries.view())));
        });

        // Benchmark SciPy equivalent (if available)
        #[cfg(feature = "scipy-comparison")]
        {
            group.bench_with_input(BenchmarkId::new("scipy", n_points), &n_points, |b, _| {
                Python::with_gil(|py| {
                    let scipy_interp = py.import("scipy.interpolate").unwrap();
                    let numpy = py.import("numpy").unwrap();

                    // Convert to numpy arrays
                    let x_py = numpy.call_method1("array", (x.to_vec(),)).unwrap();
                    let y_py = numpy.call_method1("array", (y.to_vec(),)).unwrap();
                    let queries_py = numpy.call_method1("array", (queries.to_vec(),)).unwrap();

                    // Create interpolator
                    let interp1d = scipy_interp.getattr("interp1d").unwrap();
                    let interpolator = interp1d.call1((x_py, y_py)).unwrap();

                    b.iter(|| {
                        black_box(interpolator.call1((queries_py,)).unwrap());
                    });
                });
            });
        }
    }

    group.finish();
}

/// Benchmark cubic spline interpolation
#[allow(dead_code)]
fn bench_cubic_spline_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("cubic_spline_comparison");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));

    for &n_points in &[100, 1000, 10_000, 50_000] {
        let (x, y) = generate_1d_data(n_points, false);
        let queries = generate_query_points_1d(1000);

        // Benchmark scirs2 construction
        group.bench_with_input(
            BenchmarkId::new("scirs2_construction", n_points),
            &n_points,
            |b, _| {
                b.iter(|| black_box(CubicSpline::new(&x.view(), &y.view())));
            },
        );

        // Benchmark scirs2 evaluation
        let spline = CubicSpline::new(&x.view(), &y.view()).unwrap();
        group.throughput(Throughput::Elements(queries.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("scirs2_evaluation", n_points),
            &n_points,
            |b, _| {
                b.iter(|| black_box(spline.evaluate_array(&queries.view())));
            },
        );

        // Benchmark SciPy equivalent
        #[cfg(feature = "scipy-comparison")]
        {
            Python::with_gil(|py| {
                let scipy_interp = py.import("scipy.interpolate").unwrap();
                let numpy = py.import("numpy").unwrap();

                // Convert to numpy arrays
                let x_py = numpy.call_method1("array", (x.to_vec(),)).unwrap();
                let y_py = numpy.call_method1("array", (y.to_vec(),)).unwrap();
                let queries_py = numpy.call_method1("array", (queries.to_vec(),)).unwrap();

                // Benchmark construction
                group.bench_with_input(
                    BenchmarkId::new("scipy_construction", n_points),
                    &n_points,
                    |b, _| {
                        b.iter(|| {
                            let cubic_spline = scipy_interp.getattr("CubicSpline").unwrap();
                            black_box(cubic_spline.call1((x_py, y_py)).unwrap());
                        });
                    },
                );

                // Benchmark evaluation
                let cubic_spline = scipy_interp.getattr("CubicSpline").unwrap();
                let spline_scipy = cubic_spline.call1((x_py, y_py)).unwrap();

                group.bench_with_input(
                    BenchmarkId::new("scipy_evaluation", n_points),
                    &n_points,
                    |b, _| {
                        b.iter(|| {
                            black_box(spline_scipy.call1((queries_py,)).unwrap());
                        });
                    },
                );
            });
        }
    }

    group.finish();
}

/// Benchmark RBF interpolation for 2D data
#[allow(dead_code)]
fn bench_rbf_2d_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("rbf_2d_comparison");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(15));

    for &n_points in &[100, 400, 900, 2500] {
        let (points, values) = generate_2d_data(n_points, false);
        let queries = generate_query_points_2d(100);

        // Test different RBF kernels
        let kernels = [
            ("gaussian", RBFKernel::Gaussian),
            ("multiquadric", RBFKernel::Multiquadric),
            ("thin_plate", RBFKernel::ThinPlateSpline),
        ];

        for (kernel_name, kernel) in &kernels {
            // Benchmark scirs2
            group.bench_with_input(
                BenchmarkId::new(format!("scirs2_{}", kernel_name), n_points),
                &n_points,
                |b, _| {
                    let interpolator =
                        RBFInterpolator::new(&points.view(), &values.view(), *kernel, 1.0).unwrap();

                    b.iter(|| black_box(interpolator.interpolate(&queries.view())));
                },
            );

            // Benchmark SciPy equivalent
            #[cfg(feature = "scipy-comparison")]
            {
                Python::with_gil(|py| {
                    let scipy_interp = py.import("scipy.interpolate").unwrap();
                    let numpy = py.import("numpy").unwrap();

                    // Convert to numpy arrays
                    let points_py = numpy
                        .call_method1("array", (points.clone().into_raw_vec(),))
                        .unwrap();
                    let points_py = points_py.call_method1("reshape", (n_points, 2)).unwrap();
                    let values_py = numpy.call_method1("array", (values.to_vec(),)).unwrap();
                    let queries_py = numpy
                        .call_method1("array", (queries.clone().into_raw_vec(),))
                        .unwrap();
                    let queries_py = queries_py.call_method1("reshape", (100, 2)).unwrap();

                    // Map kernel names
                    let scipy_kernel = match kernel_name {
                        &"gaussian" => "gaussian",
                        &"multiquadric" => "multiquadric",
                        &"thin_plate" => "thin_plate_spline",
                        _ => "gaussian",
                    };

                    group.bench_with_input(
                        BenchmarkId::new(format!("scipy_{}", kernel_name), n_points),
                        &n_points,
                        |b, _| {
                            b.iter(|| {
                                let rbf = scipy_interp.getattr("RBFInterpolator").unwrap();
                                let interpolator = rbf
                                    .call1((points_py, values_py))
                                    .unwrap()
                                    .call_method1("__call__", (queries_py,))
                                    .unwrap();
                                black_box(interpolator);
                            });
                        },
                    );
                });
            }
        }
    }

    group.finish();
}

/// Benchmark scalability to large datasets
#[allow(dead_code)]
fn bench_large_scale_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("large_scale_performance");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));

    // Test with increasingly large datasets
    for &n_points in &[10_000, 100_000, 500_000, 1_000_000] {
        let (x, y) = generate_1d_data(n_points, true);
        let queries = generate_query_points_1d(10_000);

        // Linear interpolation (should scale well)
        group.throughput(Throughput::Elements(n_points as u64));
        group.bench_with_input(BenchmarkId::new("linear", n_points), &n_points, |b, _| {
            b.iter(|| black_box(linear_interpolate(&x.view(), &y.view(), &queries.view())));
        });

        // B-spline interpolation
        if n_points <= 100_000 {
            group.bench_with_input(BenchmarkId::new("bspline", n_points), &n_points, |b, _| {
                let bspline =
                    BSpline::new(&x.view(), &y.view(), 3, ExtrapolateMode::Extrapolate).unwrap();
                b.iter(|| black_box(bspline.evaluate_array(&queries.view())));
            });
        }
    }

    group.finish();
}

/// Benchmark SIMD optimization effectiveness
#[allow(dead_code)]
fn bench_simd_effectiveness(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_effectiveness");

    // Test different batch sizes to see SIMD effectiveness
    let n_points = 10_000;
    let (x, y) = generate_1d_data(n_points, false);

    for &query_size in &[16, 32, 64, 128, 256, 512, 1024, 4096] {
        let queries = generate_query_points_1d(query_size);

        group.throughput(Throughput::Elements(query_size as u64));
        group.bench_with_input(
            BenchmarkId::new("linear", query_size),
            &query_size,
            |b, _| {
                b.iter(|| black_box(linear_interpolate(&x.view(), &y.view(), &queries.view())));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("cubic", query_size),
            &query_size,
            |b, _| {
                b.iter(|| black_box(cubic_interpolate(&x.view(), &y.view(), &queries.view())));
            },
        );
    }

    group.finish();
}

/// Benchmark parallel processing effectiveness
#[allow(dead_code)]
fn bench_parallel_effectiveness(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_effectiveness");
    group.sample_size(10);

    // Test with different numbers of points to process in parallel
    for &n_queries in &[1000, 10_000, 100_000, 1_000_000] {
        let (points, values) = generate_2d_data(1000, false);
        let queries = generate_query_points_2d(n_queries);

        // RBF interpolation (computationally intensive, good for parallelization)
        let interpolator =
            RBFInterpolator::new(&points.view(), &values.view(), RBFKernel::Gaussian, 1.0).unwrap();

        group.throughput(Throughput::Elements(n_queries as u64));
        group.bench_with_input(
            BenchmarkId::new("rbf_gaussian", n_queries),
            &n_queries,
            |b, _| {
                b.iter(|| black_box(interpolator.interpolate(&queries.view())));
            },
        );
    }

    group.finish();
}

/// Benchmark memory efficiency
#[allow(dead_code)]
fn bench_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_efficiency");

    // Test methods that should be memory efficient
    let sizes = [1000, 10_000, 50_000];

    for &n_points in &sizes {
        let (x, y) = generate_1d_data(n_points, false);
        let queries = generate_query_points_1d(1000);

        // Streaming evaluation (memory efficient)
        group.bench_with_input(
            BenchmarkId::new("streaming_linear", n_points),
            &n_points,
            |b, _| {
                b.iter(|| {
                    for &q in queries.iter() {
                        black_box(linear_interpolate(
                            &x.view(),
                            &y.view(),
                            &Array1::from_vec(vec![q]).view(),
                        ));
                    }
                });
            },
        );

        // Batch evaluation (less memory efficient but faster)
        group.bench_with_input(
            BenchmarkId::new("batch_linear", n_points),
            &n_points,
            |b, _| {
                b.iter(|| black_box(linear_interpolate(&x.view(), &y.view(), &queries.view())));
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_linear_1d_comparison,
    bench_cubic_spline_comparison,
    bench_rbf_2d_comparison,
    bench_large_scale_performance,
    bench_simd_effectiveness,
    bench_parallel_effectiveness,
    bench_memory_efficiency
);
criterion_main!(benches);
