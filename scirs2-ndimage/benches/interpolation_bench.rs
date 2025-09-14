use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::{Array2, Array3};
use scirs2_ndimage::interpolation::{
    affine_transform, map_coordinates, rotate, shift, zoom, InterpolationOrder,
};
use std::hint::black_box;
use std::time::Duration;

/// Benchmark basic interpolation operations
#[allow(dead_code)]
fn bench_interpolation_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("interpolation_operations");
    group.measurement_time(Duration::from_secs(10));

    let input = Array2::from_shape_fn((100, 100), |(i, j)| (i * j) as f64);

    group.bench_function("shift", |b| {
        b.iter(|| shift(black_box(&input), &[2.5, 3.5], None, None, None, None).unwrap())
    });

    group.bench_function("zoom_2x", |b| {
        b.iter(|| zoom(black_box(&input), 2.0, None, None, None, None).unwrap())
    });

    group.bench_function("zoom_0.5x", |b| {
        b.iter(|| zoom(black_box(&input), 0.5, None, None, None, None).unwrap())
    });

    group.bench_function("rotate_45deg", |b| {
        b.iter(|| rotate(black_box(&input), 45.0, None, None, None, None, None, None).unwrap())
    });

    group.finish();
}

/// Benchmark affine transformations
#[allow(dead_code)]
fn bench_affine_transform(c: &mut Criterion) {
    let mut group = c.benchmark_group("affine_transform");
    group.measurement_time(Duration::from_secs(10));

    let input = Array2::from_shape_fn((100, 100), |(i, j)| (i * j) as f64);

    // Identity matrix (no transformation)
    let identity_matrix =
        Array2::from_shape_vec((2, 3), vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).unwrap();

    // Rotation matrix (45 degrees)
    let angle = std::f64::consts::PI / 4.0;
    let cos_a = angle.cos();
    let sin_a = angle.sin();
    let rotation_matrix =
        Array2::from_shape_vec((2, 3), vec![cos_a, -sin_a, 0.0, sin_a, cos_a, 0.0]).unwrap();

    // Scale matrix (2x zoom)
    let scale_matrix = Array2::from_shape_vec((2, 3), vec![2.0, 0.0, 0.0, 0.0, 2.0, 0.0]).unwrap();

    group.bench_function("identity", |b| {
        b.iter(|| {
            affine_transform(
                black_box(&input),
                &identity_matrix,
                None,
                None,
                None,
                None,
                None,
                None,
            )
            .unwrap()
        })
    });

    group.bench_function("rotation", |b| {
        b.iter(|| {
            affine_transform(
                black_box(&input),
                &rotation_matrix,
                None,
                None,
                None,
                None,
                None,
                None,
            )
            .unwrap()
        })
    });

    group.bench_function("scale", |b| {
        b.iter(|| {
            affine_transform(
                black_box(&input),
                &scale_matrix,
                None,
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

/// Benchmark different interpolation orders
#[allow(dead_code)]
fn bench_interpolation_orders(c: &mut Criterion) {
    let mut group = c.benchmark_group("interpolation_orders");
    group.measurement_time(Duration::from_secs(10));

    let input = Array2::from_shape_fn((50, 50), |(i, j)| (i * j) as f64);

    let orders = vec![
        InterpolationOrder::Nearest,
        InterpolationOrder::Linear,
        InterpolationOrder::Cubic,
    ];

    for order in orders {
        group.bench_with_input(
            BenchmarkId::new("zoom", format!("{:?}", order)),
            &order,
            |b, order| {
                b.iter(|| zoom(black_box(&input), 1.5, Some(*order), None, None, None).unwrap())
            },
        );
    }

    group.finish();
}

/// Benchmark map_coordinates for direct coordinate mapping
#[allow(dead_code)]
fn bench_map_coordinates(c: &mut Criterion) {
    let mut group = c.benchmark_group("map_coordinates");
    group.measurement_time(Duration::from_secs(10));

    let input = Array2::from_shape_fn((50, 50), |(i, j)| (i * j) as f64);

    // Create simple coordinate transformation (identity with slight offset)
    let (rows, cols) = input.dim();
    let coordinates = Array2::from_shape_fn((rows, cols), |(i, j)| (i * j) as f64 + 0.5)
        .into_dimensionality::<ndarray::IxDyn>()
        .unwrap();

    group.bench_function("map_coordinates_linear", |b| {
        b.iter(|| {
            map_coordinates(
                black_box(&input),
                black_box(&coordinates),
                Some(1), // Linear interpolation
                None,
                None,
                None,
            )
            .unwrap()
        })
    });

    group.bench_function("map_coordinates_cubic", |b| {
        b.iter(|| {
            map_coordinates(
                black_box(&input),
                black_box(&coordinates),
                Some(3), // Cubic interpolation
                None,
                None,
                None,
            )
            .unwrap()
        })
    });

    group.finish();
}

/// Benchmark 3D interpolation operations
#[allow(dead_code)]
fn bench_3d_interpolation(c: &mut Criterion) {
    let mut group = c.benchmark_group("3d_interpolation");
    group.measurement_time(Duration::from_secs(15));

    // Smaller 3D array for feasible benchmarking
    let input = Array3::from_shape_fn((20, 20, 20), |(i, j, k)| (i * j * k) as f64);

    group.bench_function("shift_3d", |b| {
        b.iter(|| shift(black_box(&input), &[1.5, 2.5, 1.0], None, None, None, None).unwrap())
    });

    group.bench_function("zoom_3d", |b| {
        b.iter(|| zoom(black_box(&input), 1.5, None, None, None, None).unwrap())
    });

    group.finish();
}

/// Benchmark different array sizes for scaling behavior
#[allow(dead_code)]
fn bench_scaling_behavior(c: &mut Criterion) {
    let mut group = c.benchmark_group("scaling_behavior");
    group.measurement_time(Duration::from_secs(10));

    let sizes = vec![(25, 25), (50, 50), (100, 100), (150, 150)];

    for (rows, cols) in sizes {
        let input = Array2::from_shape_fn((rows, cols), |(i, j)| (i * j) as f64);

        group.bench_with_input(
            BenchmarkId::new("shift", format!("{}x{}", rows, cols)),
            &input,
            |b, input| {
                b.iter(|| shift(black_box(input), &[2.5, 3.5], None, None, None, None).unwrap())
            },
        );

        group.bench_with_input(
            BenchmarkId::new("zoom", format!("{}x{}", rows, cols)),
            &input,
            |b, input| b.iter(|| zoom(black_box(input), 1.5, None, None, None, None).unwrap()),
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_interpolation_operations,
    bench_affine_transform,
    bench_interpolation_orders,
    bench_map_coordinates,
    bench_3d_interpolation,
    bench_scaling_behavior
);
criterion_main!(benches);
