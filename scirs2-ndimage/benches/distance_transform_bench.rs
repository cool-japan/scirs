//! Distance transform benchmarks
//!
//! This module contains benchmarks for distance transform algorithms,
//! comparing the optimized separable algorithm against the brute force approach.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::{Array2, Array3, IxDyn};
use scirs2_ndimage::morphology::{distance_transform_bf, distance_transform_edt};

fn create_test_pattern_2d(rows: usize, cols: usize) -> Array2<bool> {
    Array2::from_shape_fn((rows, cols), |(i, j)| {
        // Create a pattern with some foreground regions
        let center_i = rows / 2;
        let center_j = cols / 2;
        let dist_sq = (i as f64 - center_i as f64).powi(2) + (j as f64 - center_j as f64).powi(2);
        let radius_sq = (std::cmp::min(rows, cols) / 4) as f64;
        radius_sq.powi(2) < dist_sq && dist_sq < (radius_sq * 2.0).powi(2)
    })
}

fn create_test_pattern_3d(size_x: usize, size_y: usize, size_z: usize) -> Array3<bool> {
    Array3::from_shape_fn((size_x, size_y, size_z), |(i, j, k)| {
        // Create a 3D pattern with some complexity
        let center_i = size_x / 2;
        let center_j = size_y / 2;
        let center_k = size_z / 2;
        let dist_sq = (i as f64 - center_i as f64).powi(2)
            + (j as f64 - center_j as f64).powi(2)
            + (k as f64 - center_k as f64).powi(2);
        let radius_sq = (std::cmp::min(std::cmp::min(size_x, size_y), size_z) / 4) as f64;
        radius_sq.powi(2) < dist_sq && dist_sq < (radius_sq * 2.0).powi(2)
    })
}

fn bench_distance_transform_2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance_transform_2d");

    let sizes = vec![
        (50, 50, "Small"),
        (100, 100, "Medium"),
        (200, 200, "Large"),
        (400, 400, "Very Large"),
    ];

    for (rows, cols, _label) in sizes {
        let input = create_test_pattern_2d(rows, cols);
        let input_dyn = input.clone().into_dimensionality::<IxDyn>().unwrap();

        // Benchmark optimized algorithm
        group.bench_with_input(
            BenchmarkId::new("Optimized", format!("{}x{}", rows, cols)),
            &input_dyn,
            |b, input| {
                b.iter(|| {
                    let (distances, _) =
                        distance_transform_edt(black_box(input), None, true, false);
                    black_box(distances)
                })
            },
        );

        // Only benchmark brute force for smaller sizes due to performance
        if rows <= 100 {
            group.bench_with_input(
                BenchmarkId::new("Brute Force", format!("{}x{}", rows, cols)),
                &input_dyn,
                |b, input| {
                    b.iter(|| {
                        let (distances, _) =
                            distance_transform_bf(black_box(input), "euclidean", None, true, false);
                        black_box(distances)
                    })
                },
            );
        }
    }

    group.finish();
}

fn bench_distance_transform_3d(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance_transform_3d");

    let sizes = vec![
        (20, 20, 20, "Small 3D"),
        (30, 30, 30, "Medium 3D"),
        (50, 50, 50, "Large 3D"),
    ];

    for (size_x, size_y, size_z, label) in sizes {
        let input = create_test_pattern_3d(size_x, size_y, size_z);
        let input_dyn = input.clone().into_dimensionality::<IxDyn>().unwrap();

        // Only benchmark optimized algorithm for 3D due to performance
        group.bench_with_input(
            BenchmarkId::new("Optimized 3D", label),
            &input_dyn,
            |b, input| {
                b.iter(|| {
                    let (distances, _) =
                        distance_transform_edt(black_box(input), None, true, false);
                    black_box(distances)
                })
            },
        );

        // Brute force only for very small sizes
        if size_x <= 20 {
            group.bench_with_input(
                BenchmarkId::new("Brute Force 3D", label),
                &input_dyn,
                |b, input| {
                    b.iter(|| {
                        let (distances, _) =
                            distance_transform_bf(black_box(input), "euclidean", None, true, false);
                        black_box(distances)
                    })
                },
            );
        }
    }

    group.finish();
}

fn bench_distance_metrics(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance_metrics");

    let input = create_test_pattern_2d(100, 100);
    let input_dyn = input.clone().into_dimensionality::<IxDyn>().unwrap();

    let metrics = vec![
        ("euclidean", "Euclidean"),
        ("cityblock", "Manhattan"),
        ("chessboard", "Chessboard"),
    ];

    for (metric, label) in metrics {
        group.bench_with_input(BenchmarkId::new("Metric", label), &input_dyn, |b, input| {
            b.iter(|| {
                let (distances, _) =
                    distance_transform_bf(black_box(input), metric, None, true, false);
                black_box(distances)
            })
        });
    }

    group.finish();
}

fn bench_sampling_effects(c: &mut Criterion) {
    let mut group = c.benchmark_group("sampling_effects");

    let input = create_test_pattern_2d(100, 100);
    let input_dyn = input.clone().into_dimensionality::<IxDyn>().unwrap();

    let sampling_options = vec![
        (None, "Unit Sampling"),
        (Some(vec![1.0, 1.0]), "Equal Sampling"),
        (Some(vec![2.0, 1.0]), "Anisotropic X"),
        (Some(vec![1.0, 2.0]), "Anisotropic Y"),
        (Some(vec![0.5, 0.5]), "High Resolution"),
    ];

    for (sampling, label) in sampling_options {
        group.bench_with_input(
            BenchmarkId::new("Sampling", label),
            &input_dyn,
            |b, input| {
                b.iter(|| {
                    let (distances, _) = distance_transform_edt(
                        black_box(input),
                        sampling.as_ref().map(|s| s.as_slice()),
                        true,
                        false,
                    );
                    black_box(distances)
                })
            },
        );
    }

    group.finish();
}

fn bench_return_modes(c: &mut Criterion) {
    let mut group = c.benchmark_group("return_modes");

    let input = create_test_pattern_2d(100, 100);
    let input_dyn = input.clone().into_dimensionality::<IxDyn>().unwrap();

    let return_modes = vec![
        (true, false, "Distances Only"),
        (false, true, "Indices Only"),
        (true, true, "Both"),
    ];

    for (return_distances, return_indices, label) in return_modes {
        group.bench_with_input(
            BenchmarkId::new("Return Mode", label),
            &input_dyn,
            |b, input| {
                b.iter(|| {
                    let (distances, indices) = distance_transform_edt(
                        black_box(input),
                        None,
                        return_distances,
                        return_indices,
                    );
                    black_box((distances, indices))
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    distance_transform_benches,
    bench_distance_transform_2d,
    bench_distance_transform_3d,
    bench_distance_metrics,
    bench_sampling_effects,
    bench_return_modes
);

criterion_main!(distance_transform_benches);
