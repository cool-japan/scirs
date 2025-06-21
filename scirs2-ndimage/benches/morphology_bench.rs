use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::{Array2, Array3};
use scirs2_ndimage::morphology::{
    binary_dilation, binary_erosion, binary_hit_or_miss, black_tophat, grey_dilation, grey_erosion,
    morphological_gradient, white_tophat,
};
use std::time::Duration;

/// Benchmark binary morphological operations
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

criterion_group!(
    benches,
    bench_binary_morphology,
    bench_grayscale_morphology,
    bench_hit_or_miss,
    bench_structuring_sizes,
    bench_3d_morphology
);
criterion_main!(benches);
