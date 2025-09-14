use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::{Array1, Array2, Array3};
use scirs2_ndimage::filters::{
    bilateral_filter,
    filter_functions,
    gaussian_filter,
    gaussian_filter_f32,
    generic_filter,
    gradient_magnitude,
    gradient_magnitude_optimized,
    laplace,
    laplace_2d_optimized,
    maximum_filter,
    median_filter,
    minimum_filter,
    // Edge detection filters
    sobel,
    sobel_2d_optimized,
    uniform_filter,
    BorderMode,
};
use std::hint::black_box;
use std::time::Duration;

#[cfg(feature = "simd")]
use scirs2_ndimage::filters::{bilateral_filter_simd_f32, bilateral_filter_simd_f64};

/// Benchmark generic filter with different functions
#[allow(dead_code)]
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
#[allow(dead_code)]
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
#[allow(dead_code)]
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
#[allow(dead_code)]
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
#[allow(dead_code)]
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

/// Benchmark edge detection filters: optimized vs standard
#[allow(dead_code)]
fn bench_edge_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("edge_detection");
    group.measurement_time(Duration::from_secs(10));

    // Test different sizes to see scaling behavior
    let sizes = vec![100, 500, 1000];

    for size in sizes {
        let input = Array2::from_shape_fn((size, size), |(i, j)| {
            ((i as f64 / 10.0).sin() + (j as f64 / 10.0).cos()) * 255.0
        });

        // Benchmark standard Sobel
        group.bench_with_input(
            BenchmarkId::new("sobel_standard", format!("{}x{}", size, size)),
            &input,
            |b, input| b.iter(|| sobel(black_box(input), 0, Some(BorderMode::Reflect)).unwrap()),
        );

        // Benchmark optimized Sobel
        group.bench_with_input(
            BenchmarkId::new("sobel_optimized", format!("{}x{}", size, size)),
            &input,
            |b, input| {
                b.iter(|| {
                    sobel_2d_optimized(black_box(&input.view()), 0, Some(BorderMode::Reflect))
                        .unwrap()
                })
            },
        );

        // Benchmark standard Laplacian
        group.bench_with_input(
            BenchmarkId::new("laplace_standard", format!("{}x{}", size, size)),
            &input,
            |b, input| {
                b.iter(|| {
                    laplace(black_box(input), Some(BorderMode::Reflect), Some(false)).unwrap()
                })
            },
        );

        // Benchmark optimized Laplacian
        group.bench_with_input(
            BenchmarkId::new("laplace_optimized", format!("{}x{}", size, size)),
            &input,
            |b, input| {
                b.iter(|| {
                    laplace_2d_optimized(black_box(&input.view()), false, Some(BorderMode::Reflect))
                        .unwrap()
                })
            },
        );

        // Benchmark gradient magnitude computation
        let grad_x = sobel(&input, 1, Some(BorderMode::Reflect)).unwrap();
        let grad_y = sobel(&input, 0, Some(BorderMode::Reflect)).unwrap();

        group.bench_with_input(
            BenchmarkId::new("gradient_magnitude_standard", format!("{}x{}", size, size)),
            &(&grad_x, &grad_y),
            |b, (gx, gy)| {
                b.iter(|| {
                    let result = (&(**gx) * &(**gx) + &(**gy) * &(**gy)).mapv(|x| x.sqrt());
                    black_box(result)
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("gradient_magnitude_optimized", format!("{}x{}", size, size)),
            &(&grad_x, &grad_y),
            |b, (gx, gy)| {
                b.iter(|| {
                    gradient_magnitude_optimized(black_box(&gx.view()), black_box(&gy.view()))
                        .unwrap()
                })
            },
        );
    }

    group.finish();
}

/// Benchmark performance characteristics and scaling behavior
#[allow(dead_code)]
fn bench_performance_characteristics(c: &mut Criterion) {
    let mut group = c.benchmark_group("performance_characteristics");
    group.measurement_time(Duration::from_secs(10));

    // Test scaling behavior with different array sizes
    let scaling_sizes = vec![(50, 50), (100, 100), (200, 200), (400, 400)];

    for (rows, cols) in scaling_sizes {
        let input = Array2::from_shape_fn((rows, cols), |(i, j)| {
            ((i as f64 / 10.0).sin() * (j as f64 / 10.0).cos() * 255.0)
        });

        // Test linear scaling of uniform filter
        group.bench_with_input(
            BenchmarkId::new("uniform_scaling", format!("{}x{}", rows, cols)),
            &input,
            |b, input| {
                b.iter(|| uniform_filter(black_box(input), black_box(&[5, 5]), None, None).unwrap())
            },
        );

        // Test Gaussian filter scaling
        group.bench_with_input(
            BenchmarkId::new("gaussian_scaling", format!("{}x{}", rows, cols)),
            &input,
            |b, input| {
                b.iter(|| gaussian_filter(black_box(input), black_box(2.0), None, None).unwrap())
            },
        );
    }

    group.finish();
}

/// Benchmark memory vs computation trade-offs
#[allow(dead_code)]
fn bench_memory_computation_tradeoffs(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_computation_tradeoffs");
    group.measurement_time(Duration::from_secs(8));

    let input = Array2::from_shape_fn((200, 200), |(i, j)| (i + j) as f64);

    // Compare different kernel sizes (memory vs computation)
    let kernel_sizes = vec![3, 5, 7, 9, 11, 15];

    for size in kernel_sizes {
        group.bench_with_input(
            BenchmarkId::new("uniform_kernel_size", size),
            &input,
            |b, input| {
                b.iter(|| {
                    uniform_filter(black_box(input), black_box(&[size, size]), None, None).unwrap()
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("median_kernel_size", size),
            &input,
            |b, input| {
                b.iter(|| median_filter(black_box(input), black_box(&[size, size]), None).unwrap())
            },
        );
    }

    group.finish();
}

/// Benchmark different data type performance
#[allow(dead_code)]
fn bench_data_type_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("data_type_performance");
    group.measurement_time(Duration::from_secs(8));

    let size = (100, 100);

    // f32 arrays
    let input_f32 = Array2::from_shape_fn(size, |(i, j)| (i + j) as f32);
    group.bench_with_input(
        BenchmarkId::new("gaussian_f32", format!("{}x{}", size.0, size.1)),
        &input_f32,
        |b, input| {
            b.iter(|| gaussian_filter_f32(black_box(input), black_box(1.0f32), None, None).unwrap())
        },
    );

    // f64 arrays
    let input_f64 = Array2::from_shape_fn(size, |(i, j)| (i + j) as f64);
    group.bench_with_input(
        BenchmarkId::new("gaussian_f64", format!("{}x{}", size.0, size.1)),
        &input_f64,
        |b, input| {
            b.iter(|| gaussian_filter(black_box(input), black_box(1.0f64), None, None).unwrap())
        },
    );

    group.finish();
}

/// Benchmark cache efficiency with different access patterns
#[allow(dead_code)]
fn bench_cache_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_efficiency");
    group.measurement_time(Duration::from_secs(8));

    // Test different array shapes with same total size to test cache behavior
    let shapes = vec![
        (400, 400),  // Square
        (200, 800),  // Wide rectangle
        (800, 200),  // Tall rectangle
        (100, 1600), // Very wide
        (1600, 100), // Very tall
    ];

    for (rows, cols) in shapes {
        let input = Array2::from_shape_fn((rows, cols), |(i, j)| (i + j) as f64);

        group.bench_with_input(
            BenchmarkId::new("uniform_cache", format!("{}x{}", rows, cols)),
            &input,
            |b, input| {
                b.iter(|| uniform_filter(black_box(input), black_box(&[3, 3]), None, None).unwrap())
            },
        );
    }

    group.finish();
}

/// Benchmark parallel vs sequential performance
#[cfg(feature = "parallel")]
#[allow(dead_code)]
fn bench_parallel_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_performance");
    group.measurement_time(Duration::from_secs(10));

    let sizes = vec![(100, 100), (300, 300), (500, 500)];

    for (rows, cols) in sizes {
        let input = Array2::from_shape_fn((rows, cols), |(i, j)| (i + j) as f64);

        group.bench_with_input(
            BenchmarkId::new("gaussian_parallel", format!("{}x{}", rows, cols)),
            &input,
            |b, input| {
                b.iter(|| gaussian_filter(black_box(input), black_box(2.0), None, None).unwrap())
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_generic_filter,
    bench_standard_filters,
    bench_bilateral_filter,
    bench_border_modes,
    bench_dimensionalities,
    bench_edge_detection,
    bench_performance_characteristics,
    bench_memory_computation_tradeoffs,
    bench_data_type_performance,
    bench_cache_efficiency /* bench_parallel_performance - conditionally compiled */
);
criterion_main!(benches);
