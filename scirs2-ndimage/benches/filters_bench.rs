use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::{Array1, Array2, Array3};
use scirs2_ndimage::filters::{
    bilateral_filter, filter_functions, gaussian_filter, generic_filter, maximum_filter,
    median_filter, minimum_filter, uniform_filter, BorderMode,
};
use std::time::Duration;

#[cfg(feature = "simd")]
use scirs2_ndimage::filters::{bilateral_filter_simd_f32, bilateral_filter_simd_f64};

/// Benchmark generic filter with different functions
fn bench_generic_filter(c: &mut Criterion) {
    let mut group = c.benchmark_group("generic_filter");
    group.measurement_time(Duration::from_secs(10));

    // Test different array sizes
    let sizes = vec![(10, 10), (50, 50), (100, 100), (200, 200)];

    for (rows, cols) in sizes {
        let input = Array2::from_shape_fn((rows, cols), |(i, j)| (i * j) as f64);
        let kernel_size = [3, 3];

        // Benchmark mean filter
        group.bench_with_input(
            BenchmarkId::new("mean", format!("{}x{}", rows, cols)),
            &input,
            |b, input| {
                b.iter(|| {
                    generic_filter(
                        black_box(input),
                        filter_functions::mean,
                        black_box(&kernel_size),
                        Some(BorderMode::Reflect),
                        None,
                    )
                    .unwrap()
                })
            },
        );

        // Benchmark range filter
        group.bench_with_input(
            BenchmarkId::new("range", format!("{}x{}", rows, cols)),
            &input,
            |b, input| {
                b.iter(|| {
                    generic_filter(
                        black_box(input),
                        filter_functions::range,
                        black_box(&kernel_size),
                        Some(BorderMode::Reflect),
                        None,
                    )
                    .unwrap()
                })
            },
        );

        // Benchmark variance filter
        group.bench_with_input(
            BenchmarkId::new("variance", format!("{}x{}", rows, cols)),
            &input,
            |b, input| {
                b.iter(|| {
                    generic_filter(
                        black_box(input),
                        filter_functions::variance,
                        black_box(&kernel_size),
                        Some(BorderMode::Reflect),
                        None,
                    )
                    .unwrap()
                })
            },
        );
    }

    group.finish();
}

/// Benchmark standard filters for comparison
fn bench_standard_filters(c: &mut Criterion) {
    let mut group = c.benchmark_group("standard_filters");
    group.measurement_time(Duration::from_secs(10));

    let input = Array2::from_shape_fn((100, 100), |(i, j)| (i * j) as f64);
    let kernel_size = [3, 3];

    group.bench_function("uniform_filter", |b| {
        b.iter(|| uniform_filter(black_box(&input), black_box(&kernel_size), None, None).unwrap())
    });

    group.bench_function("minimum_filter", |b| {
        b.iter(|| minimum_filter(black_box(&input), black_box(&kernel_size), None, None).unwrap())
    });

    group.bench_function("maximum_filter", |b| {
        b.iter(|| maximum_filter(black_box(&input), black_box(&kernel_size), None, None).unwrap())
    });

    group.bench_function("median_filter", |b| {
        b.iter(|| median_filter(black_box(&input), black_box(&kernel_size), None).unwrap())
    });

    group.bench_function("gaussian_filter", |b| {
        b.iter(|| gaussian_filter(black_box(&input), 1.0, None, None).unwrap())
    });

    group.finish();
}

/// Benchmark bilateral filter with and without SIMD
fn bench_bilateral_filter(c: &mut Criterion) {
    let mut group = c.benchmark_group("bilateral_filter");
    group.measurement_time(Duration::from_secs(10));

    let input = Array2::from_shape_fn((50, 50), |(i, j)| (i * j) as f64);

    group.bench_function("bilateral_regular", |b| {
        b.iter(|| bilateral_filter(black_box(&input), 1.0, 1.0, None).unwrap())
    });

    #[cfg(feature = "simd")]
    {
        let input_f32 = input.map(|&x| x as f32);
        let input_f64 = input.clone();

        group.bench_function("bilateral_simd_f32", |b| {
            b.iter(|| bilateral_filter_simd_f32(black_box(&input_f32), 1.0, 1.0, None).unwrap())
        });

        group.bench_function("bilateral_simd_f64", |b| {
            b.iter(|| bilateral_filter_simd_f64(black_box(&input_f64), 1.0, 1.0, None).unwrap())
        });
    }

    group.finish();
}

/// Benchmark different border modes
fn bench_border_modes(c: &mut Criterion) {
    let mut group = c.benchmark_group("border_modes");
    group.measurement_time(Duration::from_secs(10));

    let input = Array2::from_shape_fn((100, 100), |(i, j)| (i * j) as f64);
    let kernel_size = [5, 5];

    let modes = vec![
        BorderMode::Constant,
        BorderMode::Reflect,
        BorderMode::Nearest,
        BorderMode::Wrap,
        BorderMode::Mirror,
    ];

    for mode in modes {
        group.bench_with_input(
            BenchmarkId::new("uniform_filter", format!("{:?}", mode)),
            &mode,
            |b, mode| {
                b.iter(|| {
                    uniform_filter(
                        black_box(&input),
                        black_box(&kernel_size),
                        Some(*mode),
                        None,
                    )
                    .unwrap()
                })
            },
        );
    }

    group.finish();
}

/// Benchmark different dimensionalities
fn bench_dimensionalities(c: &mut Criterion) {
    let mut group = c.benchmark_group("dimensionalities");
    group.measurement_time(Duration::from_secs(10));

    // 1D benchmark
    let input_1d = Array1::from_shape_fn(1000, |i| i as f64);
    group.bench_function("generic_filter_1d", |b| {
        b.iter(|| {
            generic_filter(
                black_box(&input_1d),
                filter_functions::mean,
                black_box(&[5]),
                Some(BorderMode::Reflect),
                None,
            )
            .unwrap()
        })
    });

    // 2D benchmark
    let input_2d = Array2::from_shape_fn((100, 100), |(i, j)| (i * j) as f64);
    group.bench_function("generic_filter_2d", |b| {
        b.iter(|| {
            generic_filter(
                black_box(&input_2d),
                filter_functions::mean,
                black_box(&[5, 5]),
                Some(BorderMode::Reflect),
                None,
            )
            .unwrap()
        })
    });

    // 3D benchmark (smaller size due to memory constraints)
    let input_3d = Array3::from_shape_fn((20, 20, 20), |(i, j, k)| (i * j * k) as f64);
    group.bench_function("generic_filter_3d", |b| {
        b.iter(|| {
            generic_filter(
                black_box(&input_3d),
                filter_functions::mean,
                black_box(&[3, 3, 3]),
                Some(BorderMode::Reflect),
                None,
            )
            .unwrap()
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_generic_filter,
    bench_standard_filters,
    bench_bilateral_filter,
    bench_border_modes,
    bench_dimensionalities
);
criterion_main!(benches);
