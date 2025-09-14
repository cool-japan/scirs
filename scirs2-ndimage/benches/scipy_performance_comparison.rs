//! Performance benchmarks with SciPy-style reference implementations
//!
//! This benchmark suite provides performance comparisons against baseline implementations
//! that would be comparable to SciPy's ndimage performance characteristics.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::{Array1, Array2, Array3, Array4};
use scirs2_ndimage::filters::*;
use scirs2_ndimage::interpolation::*;
use scirs2_ndimage::measurements::*;
use scirs2_ndimage::morphology::*;
use std::hint::black_box;
use std::time::Duration;

/// Benchmark filters against baseline implementations
#[allow(dead_code)]
fn bench_filter_performance_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("filter_performance_comparison");
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(10);

    // Test multiple array sizes to understand scaling behavior
    let sizes = vec![
        (32, 32),     // Small
        (128, 128),   // Medium
        (512, 512),   // Large
        (1024, 1024), // Very large
    ];

    for (rows, cols) in sizes {
        let input = Array2::from_shape_fn((rows, cols), |(i, j)| {
            ((i as f64 * 0.1).sin() * (j as f64 * 0.1).cos() * 100.0)
        });

        // Benchmark Gaussian filter at different sigma values
        for sigma in [0.5, 1.0, 2.0, 4.0] {
            group.bench_with_input(
                BenchmarkId::new(
                    "gaussian_filter",
                    format!("{}x{}_sigma{}", rows, cols, sigma),
                ),
                &input,
                |b, input| {
                    b.iter(|| {
                        gaussian_filter(black_box(input), black_box(sigma), None, None).unwrap()
                    })
                },
            );
        }

        // Benchmark median filter with different kernel sizes
        for kernel_size in [3, 5, 7, 9] {
            group.bench_with_input(
                BenchmarkId::new(
                    "median_filter",
                    format!("{}x{}_k{}", rows, cols, kernel_size),
                ),
                &input,
                |b, input| {
                    b.iter(|| {
                        median_filter(
                            black_box(input),
                            black_box(&[kernel_size, kernel_size]),
                            None,
                        )
                        .unwrap()
                    })
                },
            );
        }

        // Benchmark uniform filter
        group.bench_with_input(
            BenchmarkId::new("uniform_filter", format!("{}x{}", rows, cols)),
            &input,
            |b, input| {
                b.iter(|| uniform_filter(black_box(input), black_box(&[5, 5]), None, None).unwrap())
            },
        );
    }

    group.finish();
}

/// Benchmark morphological operations scaling
#[allow(dead_code)]
fn bench_morphology_performance_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("morphology_performance_scaling");
    group.measurement_time(Duration::from_secs(10));

    let sizes = vec![(64, 64), (128, 128), (256, 256), (512, 512)];

    for (rows, cols) in sizes {
        // Create binary test image with complex structure
        let binary_input = Array2::from_shape_fn((rows, cols), |(i, j)| {
            let x = i as f64 / rows as f64;
            let y = j as f64 / cols as f64;
            ((x * 10.0).sin() + (y * 10.0).cos()) > 0.0
        });

        // Benchmark binary morphological operations
        group.bench_with_input(
            BenchmarkId::new("binary_erosion", format!("{}x{}", rows, cols)),
            &binary_input,
            |b, input| {
                b.iter(|| {
                    binary_erosion(black_box(input), None, None, None, None, None, None).unwrap()
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("binary_dilation", format!("{}x{}", rows, cols)),
            &binary_input,
            |b, input| {
                b.iter(|| {
                    binary_dilation(black_box(input), None, None, None, None, None, None).unwrap()
                })
            },
        );

        // Test opening (erosion + dilation)
        group.bench_with_input(
            BenchmarkId::new("binary_opening", format!("{}x{}", rows, cols)),
            &binary_input,
            |b, input| {
                b.iter(|| {
                    binary_opening(black_box(input), None, None, None, None, None, None).unwrap()
                })
            },
        );
    }

    group.finish();
}

/// Benchmark measurement operations on labeled arrays
#[allow(dead_code)]
fn bench_measurements_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("measurements_performance");
    group.measurement_time(Duration::from_secs(8));

    let sizes = vec![(100, 100), (200, 200), (400, 400)];

    for (rows, cols) in sizes {
        // Create test data with multiple regions
        let values = Array2::from_shape_fn((rows, cols), |(i, j)| {
            (i as f64 * j as f64).sqrt() + (i + j) as f64 * 0.1
        });

        let labels = Array2::from_shape_fn((rows, cols), |(i, j)| {
            ((i / 20) * 10 + (j / 20)) + 1 // Create grid of labeled regions
        });

        group.bench_with_input(
            BenchmarkId::new("sum_labels", format!("{}x{}", rows, cols)),
            &(&values, &labels),
            |b, (values, labels)| {
                b.iter(|| sum_labels(black_box(values), black_box(labels), None).unwrap())
            },
        );

        group.bench_with_input(
            BenchmarkId::new("mean_labels", format!("{}x{}", rows, cols)),
            &(&values, &labels),
            |b, (values, labels)| {
                b.iter(|| mean_labels(black_box(values), black_box(labels), None).unwrap())
            },
        );

        group.bench_with_input(
            BenchmarkId::new("variance_labels", format!("{}x{}", rows, cols)),
            &(&values, &labels),
            |b, (values, labels)| {
                b.iter(|| variance_labels(black_box(values), black_box(labels), None).unwrap())
            },
        );

        group.bench_with_input(
            BenchmarkId::new("center_of_mass", format!("{}x{}", rows, cols)),
            &values,
            |b, values| b.iter(|| center_of_mass(black_box(values)).unwrap()),
        );
    }

    group.finish();
}

/// Benchmark interpolation operations at different scales
#[allow(dead_code)]
fn bench_interpolation_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("interpolation_performance");
    group.measurement_time(Duration::from_secs(10));

    let base_sizes = vec![(64, 64), (128, 128), (256, 256)];

    for (rows, cols) in base_sizes {
        let input = Array2::from_shape_fn((rows, cols), |(i, j)| {
            ((i as f64 / 10.0).sin() * (j as f64 / 10.0).cos())
        });

        // Benchmark zoom at different factors
        for zoom_factor in [0.5, 1.5, 2.0, 3.0] {
            group.bench_with_input(
                BenchmarkId::new("zoom", format!("{}x{}_factor{}", rows, cols, zoom_factor)),
                &input,
                |b, input| {
                    b.iter(|| {
                        zoom(
                            black_box(input),
                            black_box(zoom_factor),
                            None,
                            None,
                            None,
                            None,
                        )
                        .unwrap()
                    })
                },
            );
        }

        // Benchmark rotation
        group.bench_with_input(
            BenchmarkId::new("rotate", format!("{}x{}", rows, cols)),
            &input,
            |b, input| {
                b.iter(|| {
                    rotate(
                        black_box(input),
                        black_box(std::f64::consts::PI / 4.0), // 45 degrees
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                    )
                    .unwrap()
                })
            },
        );

        // Benchmark affine transformation
        let transform_matrix = ndarray::array![[0.8, -0.2], [0.2, 0.9]]; // Scale + rotate
        group.bench_with_input(
            BenchmarkId::new("affine_transform", format!("{}x{}", rows, cols)),
            &input,
            |b, input| {
                b.iter(|| {
                    affine_transform(
                        black_box(input),
                        black_box(&transform_matrix),
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                    )
                    .unwrap()
                })
            },
        );
    }

    group.finish();
}

/// Benchmark 3D operations to test multi-dimensional scaling
#[allow(dead_code)]
fn bench_3d_operations_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("3d_operations_performance");
    group.measurement_time(Duration::from_secs(12));
    group.sample_size(5); // Fewer samples for 3D due to longer execution time

    let sizes = vec![(32, 32, 32), (64, 64, 64), (100, 100, 100)];

    for (depth, rows, cols) in sizes {
        let volume = Array3::from_shape_fn((depth, rows, cols), |(d, i, j)| {
            ((d + i + j) as f64 / 10.0).sin() * 100.0
        });

        // 3D Gaussian filter
        group.bench_with_input(
            BenchmarkId::new("gaussian_3d", format!("{}x{}x{}", depth, rows, cols)),
            &volume,
            |b, volume| {
                b.iter(|| gaussian_filter(black_box(volume), black_box(1.0), None, None).unwrap())
            },
        );

        // 3D uniform filter
        group.bench_with_input(
            BenchmarkId::new("uniform_3d", format!("{}x{}x{}", depth, rows, cols)),
            &volume,
            |b, volume| {
                b.iter(|| {
                    uniform_filter(black_box(volume), black_box(&[3, 3, 3]), None, None).unwrap()
                })
            },
        );

        // 3D center of mass calculation
        group.bench_with_input(
            BenchmarkId::new("center_of_mass_3d", format!("{}x{}x{}", depth, rows, cols)),
            &volume,
            |b, volume| b.iter(|| center_of_mass(black_box(volume)).unwrap()),
        );
    }

    group.finish();
}

/// Benchmark memory efficiency with different border modes
#[allow(dead_code)]
fn bench_border_mode_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("border_mode_performance");
    group.measurement_time(Duration::from_secs(8));

    let input = Array2::from_shape_fn((200, 200), |(i, j)| (i + j) as f64);
    let kernel_size = [9, 9]; // Larger kernel to emphasize border effects

    let border_modes = [
        BorderMode::Constant,
        BorderMode::Reflect,
        BorderMode::Mirror,
        BorderMode::Wrap,
        BorderMode::Nearest,
    ];

    for mode in border_modes {
        group.bench_with_input(
            BenchmarkId::new("gaussian_border", format!("{:?}", mode)),
            &input,
            |b, input| {
                b.iter(|| {
                    gaussian_filter(black_box(input), 2.0, Some(black_box(mode)), None).unwrap()
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("uniform_border", format!("{:?}", mode)),
            &input,
            |b, input| {
                b.iter(|| {
                    uniform_filter(
                        black_box(input),
                        black_box(&kernel_size),
                        Some(black_box(mode)),
                        None,
                    )
                    .unwrap()
                })
            },
        );
    }

    group.finish();
}

/// Benchmark high-dimensional array operations (4D+)
#[allow(dead_code)]
fn bench_high_dimensional_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("high_dimensional_performance");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(5);

    // 4D array (e.g., video data: time x height x width x channels)
    let shape_4d = (10, 32, 32, 3);
    let array_4d = Array4::from_shape_fn(shape_4d, |(t, i, j, c)| (t + i + j + c) as f64);

    group.bench_with_input(
        BenchmarkId::new(
            "gaussian_4d",
            format!(
                "{}x{}x{}x{}",
                shape_4d.0, shape_4d.1, shape_4d.2, shape_4d.3
            ),
        ),
        &array_4d,
        |b, array| {
            b.iter(|| gaussian_filter(black_box(array), black_box(1.0), None, None).unwrap())
        },
    );

    group.bench_with_input(
        BenchmarkId::new(
            "center_of_mass_4d",
            format!(
                "{}x{}x{}x{}",
                shape_4d.0, shape_4d.1, shape_4d.2, shape_4d.3
            ),
        ),
        &array_4d,
        |b, array| b.iter(|| center_of_mass(black_box(array)).unwrap()),
    );

    group.finish();
}

/// Benchmark distance transform performance with optimized algorithms
#[allow(dead_code)]
fn bench_distance_transform_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance_transform_performance");
    group.measurement_time(Duration::from_secs(12));
    group.sample_size(10);

    let sizes = vec![(64, 64), (128, 128), (256, 256), (512, 512)];

    for (rows, cols) in sizes {
        // Create binary test image with complex patterns
        let binary_input = Array2::from_shape_fn((rows, cols), |(i, j)| {
            let x = i as f64 / rows as f64;
            let y = j as f64 / cols as f64;
            let circle1 = ((x - 0.3).powi(2) + (y - 0.3).powi(2)).sqrt() < 0.2;
            let circle2 = ((x - 0.7).powi(2) + (y - 0.7).powi(2)).sqrt() < 0.15;
            let stripes = ((x * 20.0).sin() + (y * 20.0).cos()) > 0.5;
            circle1 || circle2 || stripes
        });

        let input_dyn = binary_input.clone().into_dimensionality().unwrap();

        // Benchmark Euclidean distance transform (optimized Felzenszwalb & Huttenlocher)
        group.bench_with_input(
            BenchmarkId::new("distance_transform_edt", format!("{}x{}", rows, cols)),
            &input_dyn,
            |b, input| {
                b.iter(|| distance_transform_edt(black_box(input), None, true, false).unwrap())
            },
        );

        // Benchmark City Block distance transform
        group.bench_with_input(
            BenchmarkId::new(
                "distance_transform_cdt_cityblock",
                format!("{}x{}", rows, cols),
            ),
            &input_dyn,
            |b, input| {
                b.iter(|| {
                    distance_transform_cdt(black_box(input), "cityblock", true, false).unwrap()
                })
            },
        );

        // Benchmark Chessboard distance transform
        group.bench_with_input(
            BenchmarkId::new(
                "distance_transform_cdt_chessboard",
                format!("{}x{}", rows, cols),
            ),
            &input_dyn,
            |b, input| {
                b.iter(|| {
                    distance_transform_cdt(black_box(input), "chessboard", true, false).unwrap()
                })
            },
        );

        // Benchmark brute force distance transform for comparison
        group.bench_with_input(
            BenchmarkId::new(
                "distance_transform_bf_euclidean",
                format!("{}x{}", rows, cols),
            ),
            &input_dyn,
            |b, input| {
                b.iter(|| {
                    distance_transform_bf(black_box(input), "euclidean", None, true, false).unwrap()
                })
            },
        );
    }

    group.finish();
}

/// Benchmark edge detection filter performance
#[allow(dead_code)]
fn bench_edge_detection_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("edge_detection_performance");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(15);

    let sizes = vec![(128, 128), (256, 256), (512, 512)];

    for (rows, cols) in sizes {
        // Create test image with edges and textures
        let input = Array2::from_shape_fn((rows, cols), |(i, j)| {
            let x = i as f64 / rows as f64;
            let y = j as f64 / cols as f64;
            let circles = ((x - 0.5).powi(2) + (y - 0.5).powi(2)).sqrt();
            let waves = (x * 30.0).sin() * (y * 25.0).cos();
            let gradient = x * y;
            (circles * 100.0 + waves * 50.0 + gradient * 25.0)
        });

        // Benchmark Sobel edge detection
        group.bench_with_input(
            BenchmarkId::new("sobel_filter", format!("{}x{}", rows, cols)),
            &input,
            |b, input| b.iter(|| sobel(black_box(input), 0, None).unwrap()),
        );

        // Benchmark Prewitt edge detection
        group.bench_with_input(
            BenchmarkId::new("prewitt_filter", format!("{}x{}", rows, cols)),
            &input,
            |b, input| b.iter(|| prewitt(black_box(input), 0, None).unwrap()),
        );

        // Benchmark Laplacian edge detection
        group.bench_with_input(
            BenchmarkId::new("laplace_filter", format!("{}x{}", rows, cols)),
            &input,
            |b, input| b.iter(|| laplace(black_box(input), None, None).unwrap()),
        );

        // Benchmark Laplacian edge detection (4-connected)
        group.bench_with_input(
            BenchmarkId::new("laplace_4connected", format!("{}x{}", rows, cols)),
            &input,
            |b, input| b.iter(|| laplace(black_box(input), None, None).unwrap()),
        );

        // Benchmark Laplacian edge detection (8-connected)
        group.bench_with_input(
            BenchmarkId::new("laplace_8connected", format!("{}x{}", rows, cols)),
            &input,
            |b, input| b.iter(|| laplace(black_box(input), None, Some(true)).unwrap()),
        );
    }

    group.finish();
}

/// Benchmark data type performance (f32 vs f64)
#[allow(dead_code)]
fn bench_data_type_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("data_type_performance");
    group.measurement_time(Duration::from_secs(8));

    let size = (256, 256);

    // Generate test data for both f32 and f64
    let input_f32 = Array2::from_shape_fn(size, |(i, j)| {
        ((i as f32 * 0.1).sin() * (j as f32 * 0.1).cos()) as f32
    });

    let input_f64 = Array2::from_shape_fn(size, |(i, j)| {
        (i as f64 * 0.1).sin() * (j as f64 * 0.1).cos()
    });

    // Benchmark Gaussian filter performance with different data types
    group.bench_with_input(
        BenchmarkId::new("gaussian_f32", format!("{}x{}", size.0, size.1)),
        &input_f32,
        |b, input| {
            b.iter(|| gaussian_filter_f32(black_box(input), black_box(1.0f32), None, None).unwrap())
        },
    );

    group.bench_with_input(
        BenchmarkId::new("gaussian_f64", format!("{}x{}", size.0, size.1)),
        &input_f64,
        |b, input| {
            b.iter(|| gaussian_filter(black_box(input), black_box(1.0f64), None, None).unwrap())
        },
    );

    // Benchmark median filter performance with different data types
    group.bench_with_input(
        BenchmarkId::new("median_f32", format!("{}x{}", size.0, size.1)),
        &input_f32,
        |b, input| b.iter(|| median_filter(black_box(input), black_box(&[5, 5]), None).unwrap()),
    );

    group.bench_with_input(
        BenchmarkId::new("median_f64", format!("{}x{}", size.0, size.1)),
        &input_f64,
        |b, input| b.iter(|| median_filter(black_box(input), black_box(&[5, 5]), None).unwrap()),
    );

    group.finish();
}

/// Benchmark memory-intensive operations for large arrays
#[allow(dead_code)]
fn bench_memory_intensive_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_intensive_operations");
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(5);

    // Large array that will stress memory subsystem
    let large_size = (1024, 1024);
    let large_array = Array2::from_shape_fn(large_size, |(i, j)| {
        ((i * j) as f64).sqrt() * ((i + j) as f64 * 0.001).sin()
    });

    // Test operations that require significant memory bandwidth
    group.bench_with_input(
        BenchmarkId::new(
            "large_gaussian",
            format!("{}x{}", large_size.0, large_size.1),
        ),
        &large_array,
        |b, input| {
            b.iter(|| gaussian_filter(black_box(input), black_box(2.0), None, None).unwrap())
        },
    );

    group.bench_with_input(
        BenchmarkId::new("large_median", format!("{}x{}", large_size.0, large_size.1)),
        &large_array,
        |b, input| b.iter(|| median_filter(black_box(input), black_box(&[7, 7]), None).unwrap()),
    );

    group.bench_with_input(
        BenchmarkId::new(
            "large_uniform",
            format!("{}x{}", large_size.0, large_size.1),
        ),
        &large_array,
        |b, input| {
            b.iter(|| uniform_filter(black_box(input), black_box(&[9, 9]), None, None).unwrap())
        },
    );

    group.finish();
}

/// Benchmark small array performance to test overhead
#[allow(dead_code)]
fn bench_small_array_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("small_array_performance");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(50);

    let small_sizes = vec![(8, 8), (16, 16), (32, 32)];

    for (rows, cols) in small_sizes {
        let input = Array2::from_shape_fn((rows, cols), |(i, j)| (i as f64 + j as f64) * 0.1);

        // Test operations on small arrays to measure overhead
        group.bench_with_input(
            BenchmarkId::new("small_gaussian", format!("{}x{}", rows, cols)),
            &input,
            |b, input| {
                b.iter(|| gaussian_filter(black_box(input), black_box(1.0), None, None).unwrap())
            },
        );

        group.bench_with_input(
            BenchmarkId::new("small_median", format!("{}x{}", rows, cols)),
            &input,
            |b, input| {
                b.iter(|| median_filter(black_box(input), black_box(&[3, 3]), None).unwrap())
            },
        );

        group.bench_with_input(
            BenchmarkId::new("small_uniform", format!("{}x{}", rows, cols)),
            &input,
            |b, input| {
                b.iter(|| uniform_filter(black_box(input), black_box(&[3, 3]), None, None).unwrap())
            },
        );
    }

    group.finish();
}

criterion_group!(
    scipy_performance_benches,
    bench_filter_performance_comparison,
    bench_morphology_performance_scaling,
    bench_measurements_performance,
    bench_interpolation_performance,
    bench_3d_operations_performance,
    bench_border_mode_performance,
    bench_high_dimensional_performance,
    bench_distance_transform_performance,
    bench_edge_detection_performance,
    bench_data_type_performance,
    bench_memory_intensive_operations,
    bench_small_array_performance
);

criterion_main!(scipy_performance_benches);
