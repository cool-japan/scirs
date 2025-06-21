use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ndarray::{Array1, Array2};
use scirs2_interpolate::advanced::enhanced_kriging::EnhancedKrigingBuilder;
use scirs2_interpolate::advanced::enhanced_rbf::{EnhancedRBFInterpolator, KernelWidthStrategy};
use scirs2_interpolate::advanced::fast_kriging::{FastKrigingBuilder, FastKrigingMethod};
use scirs2_interpolate::advanced::kriging::{CovarianceFunction, KrigingInterpolator};
use scirs2_interpolate::advanced::rbf::{RBFInterpolator, RBFKernel};
use scirs2_interpolate::advanced::thinplate::ThinPlateSpline;
use scirs2_interpolate::local::mls::{MovingLeastSquares, PolynomialBasis, WeightFunction};
use scirs2_interpolate::sparse_grid::make_sparse_grid_interpolator;

fn generate_2d_test_data(n: usize) -> (Array2<f64>, Array1<f64>) {
    let mut points = Array2::zeros((n, 2));
    let mut values = Array1::zeros(n);

    for i in 0..n {
        let x = (i as f64 / (n - 1) as f64) * 10.0;
        let y = ((i * 7) % n) as f64 / (n - 1) as f64 * 10.0;

        points[[i, 0]] = x;
        points[[i, 1]] = y;
        values[i] = (x * 0.1).sin() * (y * 0.1).cos() + 0.05 * x + 0.02 * y;
    }

    (points, values)
}

#[allow(dead_code)]
fn generate_3d_test_data(n: usize) -> (Array2<f64>, Array1<f64>) {
    let mut points = Array2::zeros((n, 3));
    let mut values = Array1::zeros(n);

    for i in 0..n {
        let x = (i as f64 / (n - 1) as f64) * 5.0;
        let y = ((i * 7) % n) as f64 / (n - 1) as f64 * 5.0;
        let z = ((i * 11) % n) as f64 / (n - 1) as f64 * 5.0;

        points[[i, 0]] = x;
        points[[i, 1]] = y;
        points[[i, 2]] = z;
        values[i] = (x * 0.2).sin() * (y * 0.2).cos() * (z * 0.2).sin() + 0.1 * (x + y + z);
    }

    (points, values)
}

fn generate_query_points_2d(n: usize) -> Array2<f64> {
    let mut queries = Array2::zeros((n, 2));
    for i in 0..n {
        queries[[i, 0]] = (i as f64 / (n - 1) as f64) * 9.0 + 0.5;
        queries[[i, 1]] = ((i * 3) % n) as f64 / (n - 1) as f64 * 9.0 + 0.5;
    }
    queries
}

fn bench_rbf_interpolation(c: &mut Criterion) {
    let mut group = c.benchmark_group("rbf_interpolation");

    let kernels = [
        ("gaussian", RBFKernel::Gaussian),
        ("multiquadric", RBFKernel::Multiquadric),
        ("inverse_multiquadric", RBFKernel::InverseMultiquadric),
        ("thin_plate", RBFKernel::ThinPlateSpline),
    ];

    for (kernel_name, kernel) in kernels.iter() {
        for data_size in [50, 100, 250, 500].iter() {
            let (points, values) = generate_2d_test_data(*data_size);
            let interpolator = RBFInterpolator::new(
                &points.view(),
                &values.view(),
                *kernel,
                1.0, // epsilon
            )
            .unwrap();

            let queries = generate_query_points_2d(100);

            group.throughput(Throughput::Elements(queries.nrows() as u64));
            group.bench_with_input(
                BenchmarkId::new(format!("{}_data", kernel_name), data_size),
                data_size,
                |b, _| {
                    b.iter(|| {
                        let _ = black_box(interpolator.interpolate(black_box(&queries.view())));
                    });
                },
            );
        }
    }

    group.finish();
}

fn bench_enhanced_rbf(c: &mut Criterion) {
    let mut group = c.benchmark_group("enhanced_rbf");

    let strategies = [
        ("mean_distance", KernelWidthStrategy::MeanDistance),
        ("cross_validation", KernelWidthStrategy::CrossValidation(5)),
        ("fixed", KernelWidthStrategy::Fixed),
    ];

    for (strategy_name, strategy) in strategies.iter() {
        let (points, values) = generate_2d_test_data(200);

        group.bench_with_input(
            BenchmarkId::new("construction", strategy_name),
            strategy_name,
            |b, _| {
                b.iter(|| {
                    let _ = black_box(
                        EnhancedRBFInterpolator::builder()
                            .with_standard_kernel(RBFKernel::Gaussian)
                            .with_width_strategy(*strategy)
                            .build(&points.view(), &values.view()),
                    );
                });
            },
        );
    }

    group.finish();
}

fn bench_kriging_interpolation(c: &mut Criterion) {
    let mut group = c.benchmark_group("kriging_interpolation");

    let covariance_functions = [
        ("exponential", CovarianceFunction::Exponential),
        (
            "squared_exponential",
            CovarianceFunction::SquaredExponential,
        ),
        ("matern32", CovarianceFunction::Matern32),
        ("matern52", CovarianceFunction::Matern52),
    ];

    for (cov_name, cov_func) in covariance_functions.iter() {
        for data_size in [50, 100, 200].iter() {
            // Smaller sizes for Kriging due to computational cost
            let (points, values) = generate_2d_test_data(*data_size);
            let interpolator = KrigingInterpolator::new(
                &points.view(),
                &values.view(),
                *cov_func,
                1.0,  // sigma_sq (variance)
                1.0,  // length_scale
                0.01, // nugget
                1.0,  // alpha
            )
            .unwrap();

            let queries = generate_query_points_2d(50);

            group.throughput(Throughput::Elements(queries.nrows() as u64));
            group.bench_with_input(
                BenchmarkId::new(format!("{}_data", cov_name), data_size),
                data_size,
                |b, _| {
                    b.iter(|| {
                        for i in 0..queries.nrows() {
                            let query = queries.slice(ndarray::s![i..i + 1, ..]);
                            let _ = black_box(interpolator.predict(black_box(&query)));
                        }
                    });
                },
            );
        }
    }

    group.finish();
}

fn bench_enhanced_kriging(c: &mut Criterion) {
    let mut group = c.benchmark_group("enhanced_kriging");

    for data_size in [50, 100, 150].iter() {
        let (points, values) = generate_2d_test_data(*data_size);

        group.bench_with_input(
            BenchmarkId::new("construction", data_size),
            data_size,
            |b, _| {
                b.iter(|| {
                    let _ = black_box(
                        EnhancedKrigingBuilder::new()
                            .points(black_box(points.clone()))
                            .values(black_box(values.clone()))
                            .optimize_parameters(true)
                            .build(),
                    );
                });
            },
        );
    }

    group.finish();
}

fn bench_fast_kriging(c: &mut Criterion) {
    let mut group = c.benchmark_group("fast_kriging");

    let methods = [
        ("fixed_rank", FastKrigingMethod::FixedRank(50)),
        ("local", FastKrigingMethod::Local),
        ("tapering", FastKrigingMethod::Tapering(2.0)),
    ];

    for (method_name, method) in methods.iter() {
        let (points, values) = generate_2d_test_data(300);

        group.bench_with_input(
            BenchmarkId::new("construction", method_name),
            method_name,
            |b, _| {
                b.iter(|| {
                    let _ = black_box(
                        FastKrigingBuilder::new()
                            .points(black_box(points.clone()))
                            .values(black_box(values.clone()))
                            .approximation_method(black_box(*method))
                            .build(),
                    );
                });
            },
        );
    }

    group.finish();
}

fn bench_thin_plate_splines(c: &mut Criterion) {
    let mut group = c.benchmark_group("thin_plate_splines");

    for data_size in [50, 100, 250, 500].iter() {
        let (points, values) = generate_2d_test_data(*data_size);
        let interpolator = ThinPlateSpline::new(&points.view(), &values.view(), 0.0).unwrap();

        let queries = generate_query_points_2d(100);

        group.throughput(Throughput::Elements(queries.nrows() as u64));
        group.bench_with_input(
            BenchmarkId::new("data_size", data_size),
            data_size,
            |b, _| {
                b.iter(|| {
                    let _ = black_box(interpolator.evaluate(black_box(&queries.view())));
                });
            },
        );
    }

    group.finish();
}

fn bench_sparse_grid_interpolation(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_grid_interpolation");

    for dim in [2, 3, 4, 5].iter() {
        for max_level in [2, 3, 4].iter() {
            let bounds = vec![(0.0, 1.0); *dim];

            group.bench_with_input(
                BenchmarkId::new(
                    format!("dim_{}_level_{}", dim, max_level),
                    format!("{}_{}", dim, max_level),
                ),
                &(dim, max_level),
                |b, _| {
                    b.iter(|| {
                        let _ = black_box(make_sparse_grid_interpolator(
                            black_box(bounds.clone()),
                            black_box(*max_level),
                            black_box(|x: &[f64]| x.iter().sum::<f64>()),
                        ));
                    });
                },
            );
        }
    }

    group.finish();
}

fn bench_moving_least_squares(c: &mut Criterion) {
    let mut group = c.benchmark_group("moving_least_squares");

    let weight_functions = [
        ("gaussian", WeightFunction::Gaussian),
        ("wendlandc2", WeightFunction::WendlandC2),
        ("inverse_distance", WeightFunction::InverseDistance),
    ];

    let polynomial_bases = [
        ("constant", PolynomialBasis::Constant),
        ("linear", PolynomialBasis::Linear),
        ("quadratic", PolynomialBasis::Quadratic),
    ];

    for (weight_name, weight_func) in weight_functions.iter() {
        for (basis_name, poly_basis) in polynomial_bases.iter() {
            let (points, values) = generate_2d_test_data(200);
            let interpolator = MovingLeastSquares::new(
                points.clone(),
                values.clone(),
                *weight_func,
                *poly_basis,
                0.5, // bandwidth
            )
            .unwrap();

            let queries = generate_query_points_2d(50);

            group.throughput(Throughput::Elements(queries.nrows() as u64));
            group.bench_with_input(
                BenchmarkId::new(format!("{}_{}", weight_name, basis_name), basis_name),
                basis_name,
                |b, _| {
                    b.iter(|| {
                        for i in 0..queries.nrows() {
                            let query = queries.slice(ndarray::s![i, ..]);
                            let _ = black_box(interpolator.evaluate(black_box(&query)));
                        }
                    });
                },
            );
        }
    }

    group.finish();
}

fn bench_dimensionality_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("dimensionality_scaling");

    // Test how different methods scale with dimensionality
    for dim in [2, 3, 4, 5].iter() {
        let n_points = 100;

        // Generate test data for arbitrary dimensions
        let mut points = Array2::zeros((n_points, *dim));
        let mut values = Array1::zeros(n_points);

        for i in 0..n_points {
            let mut sum = 0.0;
            for j in 0..*dim {
                let coord = (i as f64 / (n_points - 1) as f64) * 5.0 + (j as f64) * 0.1;
                points[[i, j]] = coord;
                sum += coord * (j + 1) as f64 * 0.1;
            }
            values[i] = sum.sin();
        }

        // Test RBF scaling
        group.bench_with_input(BenchmarkId::new("rbf_gaussian", dim), dim, |b, _| {
            b.iter(|| {
                let _ = black_box(RBFInterpolator::new(
                    black_box(&points.view()),
                    black_box(&values.view()),
                    black_box(RBFKernel::Gaussian),
                    black_box(1.0),
                ));
            });
        });

        // Test MLS scaling (if dimension is reasonable)
        if *dim <= 4 {
            group.bench_with_input(BenchmarkId::new("mls_linear", dim), dim, |b, _| {
                b.iter(|| {
                    let _ = black_box(MovingLeastSquares::new(
                        black_box(points.clone()),
                        black_box(values.clone()),
                        black_box(WeightFunction::Gaussian),
                        black_box(PolynomialBasis::Linear),
                        black_box(0.5),
                    ));
                });
            });
        }
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_rbf_interpolation,
    bench_enhanced_rbf,
    bench_kriging_interpolation,
    bench_enhanced_kriging,
    bench_fast_kriging,
    bench_thin_plate_splines,
    bench_sparse_grid_interpolation,
    bench_moving_least_squares,
    bench_dimensionality_scaling
);
criterion_main!(benches);
