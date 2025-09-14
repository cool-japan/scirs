use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::{Array2, Array3};
use scirs2_ndimage::morphology::{
    binary_dilation,
    binary_dilation_2d_optimized,
    binary_erosion,
    binary_erosion_2d_optimized,
    binary_hit_or_miss,
    black_tophat,
    grey_dilation,
    grey_dilation_2d_optimized,
    grey_erosion,
    // Import optimized versions
    grey_erosion_2d_optimized,
    morphological_gradient,
    // Import simple morphology for comparison
    simple_morph::{binary_dilation_2d, binary_erosion_2d, grey_dilation_2d, grey_erosion_2d},
    white_tophat,
};
use std::hint::black_box;
use std::time::Duration;

/// Benchmark binary morphological operations
#[allow(dead_code)]
fn bench_binary_morphology(c: &mut Criterion) {
    let mut group = c.benchmark_group("binary_morphology");
    group.measurement_time(Duration::from_secs(10));

    // Test different array sizes
    let sizes = vec![(50, 50), (100, 100), (200, 200)];

    for (rows, cols) in sizes {
        // Create binary test image with some structure
        let input = Array2::from_shape_fn((rows, cols), |(i, j)| (i % 10 < 5) && (j % 10 < 5));

        // 3x3 structuring element
        let structure = Array2::from_elem((3, 3), true);

        group.bench_with_input(
            BenchmarkId::new("binary_erosion", format!("{}x{}", rows, cols)),
            &input,
            |b, input| {
                b.iter(|| {
                    binary_erosion(
                        black_box(input),
                        Some(&structure),
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

        group.bench_with_input(
            BenchmarkId::new("binary_dilation", format!("{}x{}", rows, cols)),
            &input,
            |b, input| {
                b.iter(|| {
                    binary_dilation(
                        black_box(input),
                        Some(&structure),
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

/// Benchmark grayscale morphological operations
#[allow(dead_code)]
fn bench_grayscale_morphology(c: &mut Criterion) {
    let mut group = c.benchmark_group("grayscale_morphology");
    group.measurement_time(Duration::from_secs(10));

    let input = Array2::from_shape_fn((100, 100), |(i, j)| (i * j) as f64);
    let structure = Array2::from_elem((3, 3), true);

    group.bench_function("grey_erosion", |b| {
        b.iter(|| {
            grey_erosion(black_box(&input), None, Some(&structure), None, None, None).unwrap()
        })
    });

    group.bench_function("grey_dilation", |b| {
        b.iter(|| {
            grey_dilation(black_box(&input), None, Some(&structure), None, None, None).unwrap()
        })
    });

    group.bench_function("morphological_gradient", |b| {
        b.iter(|| {
            morphological_gradient(black_box(&input), None, Some(&structure), None, None, None)
                .unwrap()
        })
    });

    group.bench_function("white_tophat", |b| {
        b.iter(|| {
            white_tophat(black_box(&input), None, Some(&structure), None, None, None).unwrap()
        })
    });

    group.bench_function("black_tophat", |b| {
        b.iter(|| {
            black_tophat(black_box(&input), None, Some(&structure), None, None, None).unwrap()
        })
    });

    group.finish();
}

/// Benchmark hit-or-miss transform
#[allow(dead_code)]
fn bench_hit_or_miss(c: &mut Criterion) {
    let mut group = c.benchmark_group("hit_or_miss");
    group.measurement_time(Duration::from_secs(10));

    // Create test pattern
    let input = Array2::from_shape_fn((100, 100), |(i, j)| {
        ((i / 10) % 2 == 0) && ((j / 10) % 2 == 0)
    });

    // Simple 3x3 structure for pattern detection
    let structure1 = Array2::from_shape_fn((3, 3), |(i, j)| i == 1 && j == 1);

    group.bench_function("binary_hit_or_miss", |b| {
        b.iter(|| {
            binary_hit_or_miss(
                black_box(&input),
                Some(&structure1),
                None,
                None,
                None,
                None,
                None,
            )
            .unwrap()
        })
    });

    group.finish();
}

/// Benchmark different structuring element sizes
#[allow(dead_code)]
fn bench_structuring_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("structuring_sizes");
    group.measurement_time(Duration::from_secs(10));

    let input = Array2::from_shape_fn((100, 100), |(i, j)| (i * j) as f64);

    let sizes = vec![3, 5, 7, 9, 11];

    for size in sizes {
        let structure = Array2::from_elem((size, size), 1.0);

        group.bench_with_input(
            BenchmarkId::new("grey_erosion", format!("{}x{}", size, size)),
            &structure,
            |b, _structure| {
                b.iter(|| grey_erosion(black_box(&input), None, None, None, None, None).unwrap())
            },
        );
    }

    group.finish();
}

/// Benchmark 3D morphological operations
#[allow(dead_code)]
fn bench_3d_morphology(c: &mut Criterion) {
    let mut group = c.benchmark_group("3d_morphology");
    group.measurement_time(Duration::from_secs(15));

    // Smaller 3D arrays for feasible benchmarking
    let input = Array3::from_shape_fn((30, 30, 30), |(i, j, k)| {
        (i % 5 < 3) && (j % 5 < 3) && (k % 5 < 3)
    });

    let structure = Array3::from_elem((3, 3, 3), true);

    group.bench_function("binary_erosion_3d", |b| {
        b.iter(|| {
            binary_erosion(
                black_box(&input),
                Some(&structure),
                None,
                None,
                None,
                None,
                None,
            )
            .unwrap()
        })
    });

    group.bench_function("binary_dilation_3d", |b| {
        b.iter(|| {
            binary_dilation(
                black_box(&input),
                Some(&structure),
                None,
                None,
                None,
                None,
                None,
            )
            .unwrap()
        })
    });

    group.finish();
}

/// Benchmark optimized vs simple morphological operations
#[allow(dead_code)]
fn bench_optimized_vs_simple(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimized_vs_simple");
    group.measurement_time(Duration::from_secs(10));

    // Test with different sizes to see scaling behavior
    let sizes = vec![100, 500, 1000];

    for size in sizes {
        // Create test data
        let grayscale_input = Array2::from_shape_fn((size, size), |(i, j)| {
            ((i as f64) * (j as f64)).sin() * 255.0
        });
        let binary_input =
            Array2::from_shape_fn((size, size), |(i, j)| (i % 10 < 5) && (j % 10 < 5));

        // Compare grayscale erosion
        group.bench_with_input(
            BenchmarkId::new("grey_erosion_simple", format!("{}x{}", size, size)),
            &grayscale_input,
            |b, input| {
                b.iter(|| grey_erosion_2d(black_box(input), None, None, None, None).unwrap())
            },
        );

        group.bench_with_input(
            BenchmarkId::new("grey_erosion_optimized", format!("{}x{}", size, size)),
            &grayscale_input,
            |b, input| {
                b.iter(|| {
                    grey_erosion_2d_optimized(black_box(input), None, None, None, None).unwrap()
                })
            },
        );

        // Compare grayscale dilation
        group.bench_with_input(
            BenchmarkId::new("grey_dilation_simple", format!("{}x{}", size, size)),
            &grayscale_input,
            |b, input| {
                b.iter(|| grey_dilation_2d(black_box(input), None, None, None, None).unwrap())
            },
        );

        group.bench_with_input(
            BenchmarkId::new("grey_dilation_optimized", format!("{}x{}", size, size)),
            &grayscale_input,
            |b, input| {
                b.iter(|| {
                    grey_dilation_2d_optimized(black_box(input), None, None, None, None).unwrap()
                })
            },
        );

        // Compare binary erosion
        group.bench_with_input(
            BenchmarkId::new("binary_erosion_simple", format!("{}x{}", size, size)),
            &binary_input,
            |b, input| {
                b.iter(|| binary_erosion_2d(black_box(input), None, None, None, None).unwrap())
            },
        );

        group.bench_with_input(
            BenchmarkId::new("binary_erosion_optimized", format!("{}x{}", size, size)),
            &binary_input,
            |b, input| {
                b.iter(|| {
                    binary_erosion_2d_optimized(black_box(input), None, None, None, None).unwrap()
                })
            },
        );

        // Compare binary dilation
        group.bench_with_input(
            BenchmarkId::new("binary_dilation_simple", format!("{}x{}", size, size)),
            &binary_input,
            |b, input| {
                b.iter(|| binary_dilation_2d(black_box(input), None, None, None, None).unwrap())
            },
        );

        group.bench_with_input(
            BenchmarkId::new("binary_dilation_optimized", format!("{}x{}", size, size)),
            &binary_input,
            |b, input| {
                b.iter(|| {
                    binary_dilation_2d_optimized(black_box(input), None, None, None, None).unwrap()
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_binary_morphology,
    bench_grayscale_morphology,
    bench_hit_or_miss,
    bench_structuring_sizes,
    bench_3d_morphology,
    bench_optimized_vs_simple
);
criterion_main!(benches);
