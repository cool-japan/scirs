use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ndarray::Array1;
use scirs2_interpolate::interp1d::monotonic::{MonotonicInterpolator, MonotonicMethod};
use scirs2_interpolate::spline::CubicSpline;
use scirs2_interpolate::{cubic_interpolate, linear_interpolate, pchip_interpolate};
use std::hint::black_box;

#[allow(dead_code)]
fn generate_test_data(n: usize) -> (Array1<f64>, Array1<f64>) {
    let x = Array1::linspace(0.0, 10.0, n);
    let y = x.mapv(|xi| (xi * 0.5_f64).sin() + 0.1 * xi + 0.05 * (3.0 * xi).cos());
    (x, y)
}

#[allow(dead_code)]
fn generate_query_points(n: usize, x_min: f64, xmax: f64) -> Array1<f64> {
    Array1::linspace(x_min + 0.1, xmax - 0.1, n)
}

#[allow(dead_code)]
fn bench_linear_interpolation(c: &mut Criterion) {
    let mut group = c.benchmark_group("linear_interpolation");

    for data_size in [100, 500, 1000, 5000].iter() {
        let (x, y) = generate_test_data(*data_size);
        let queries = generate_query_points(100, 0.0, 10.0);

        group.throughput(Throughput::Elements(queries.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("data_points", data_size),
            data_size,
            |b, _| {
                b.iter(|| {
                    let _ = black_box(linear_interpolate(
                        black_box(&x.view()),
                        black_box(&y.view()),
                        black_box(&queries.view()),
                    ));
                });
            },
        );
    }

    group.finish();
}

#[allow(dead_code)]
fn bench_cubic_interpolation(c: &mut Criterion) {
    let mut group = c.benchmark_group("cubic_interpolation");

    for data_size in [100, 500, 1000, 5000].iter() {
        let (x, y) = generate_test_data(*data_size);
        let queries = generate_query_points(100, 0.0, 10.0);

        group.throughput(Throughput::Elements(queries.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("data_points", data_size),
            data_size,
            |b, _| {
                b.iter(|| {
                    let _ = black_box(cubic_interpolate(
                        black_box(&x.view()),
                        black_box(&y.view()),
                        black_box(&queries.view()),
                    ));
                });
            },
        );
    }

    group.finish();
}

#[allow(dead_code)]
fn bench_pchip_interpolation(c: &mut Criterion) {
    let mut group = c.benchmark_group("pchip_interpolation");

    for data_size in [100, 500, 1000, 5000].iter() {
        let (x, y) = generate_test_data(*data_size);
        let queries = generate_query_points(100, 0.0, 10.0);

        group.throughput(Throughput::Elements(queries.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("data_points", data_size),
            data_size,
            |b, _| {
                b.iter(|| {
                    let _ = black_box(pchip_interpolate(
                        black_box(&x.view()),
                        black_box(&y.view()),
                        black_box(&queries.view()),
                        black_box(true), // extrapolate parameter
                    ));
                });
            },
        );
    }

    group.finish();
}

#[allow(dead_code)]
fn bench_monotonic_interpolation(c: &mut Criterion) {
    let mut group = c.benchmark_group("monotonic_interpolation");

    for data_size in [100, 500, 1000, 2000].iter() {
        let (x, y) = generate_test_data(*data_size);
        let queries = generate_query_points(100, 0.0, 10.0);

        // Pre-build the interpolator
        let interpolator =
            MonotonicInterpolator::new(&x.view(), &y.view(), MonotonicMethod::Pchip, true).unwrap();

        group.throughput(Throughput::Elements(queries.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("data_points", data_size),
            data_size,
            |b, _| {
                b.iter(|| {
                    for &query in queries.iter() {
                        let _ = black_box(interpolator.evaluate(black_box(query)));
                    }
                });
            },
        );
    }

    group.finish();
}

#[allow(dead_code)]
fn bench_cubic_spline(c: &mut Criterion) {
    let mut group = c.benchmark_group("cubic_spline");

    // Test spline construction
    for data_size in [100, 500, 1000, 5000].iter() {
        let (x, y) = generate_test_data(*data_size);

        group.bench_with_input(
            BenchmarkId::new("construction", data_size),
            data_size,
            |b, _| {
                b.iter(|| {
                    let _ = black_box(CubicSpline::new(black_box(&x.view()), black_box(&y.view())));
                });
            },
        );
    }

    // Test spline evaluation
    for data_size in [100, 500, 1000, 5000].iter() {
        let (x, y) = generate_test_data(*data_size);
        let queries = generate_query_points(100, 0.0, 10.0);
        let spline = CubicSpline::new(&x.view(), &y.view()).unwrap();

        group.throughput(Throughput::Elements(queries.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("evaluation", data_size),
            data_size,
            |b, _| {
                b.iter(|| {
                    for &query in queries.iter() {
                        let _ = black_box(spline.evaluate(black_box(query)));
                    }
                });
            },
        );
    }

    // Test derivative evaluation
    for data_size in [100, 500, 1000, 5000].iter() {
        let (x, y) = generate_test_data(*data_size);
        let queries = generate_query_points(100, 0.0, 10.0);
        let spline = CubicSpline::new(&x.view(), &y.view()).unwrap();

        group.throughput(Throughput::Elements(queries.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("derivative", data_size),
            data_size,
            |b, _| {
                b.iter(|| {
                    for &query in queries.iter() {
                        let _ = black_box(spline.derivative(black_box(query)));
                    }
                });
            },
        );
    }

    group.finish();
}

#[allow(dead_code)]
fn bench_array_evaluation(c: &mut Criterion) {
    let mut group = c.benchmark_group("array_evaluation");

    let (x, y) = generate_test_data(1000);
    let spline = CubicSpline::new(&x.view(), &y.view()).unwrap();

    for query_size in [100, 500, 1000, 5000].iter() {
        let queries = generate_query_points(*query_size, 0.0, 10.0);

        group.throughput(Throughput::Elements(*query_size as u64));
        group.bench_with_input(
            BenchmarkId::new("cubic_spline_array", query_size),
            query_size,
            |b, _| {
                b.iter(|| {
                    let _ = black_box(spline.evaluate_array(black_box(&queries.view())));
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_linear_interpolation,
    bench_cubic_interpolation,
    bench_pchip_interpolation,
    bench_monotonic_interpolation,
    bench_cubic_spline,
    bench_array_evaluation
);
criterion_main!(benches);
