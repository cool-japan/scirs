use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_linalg::{
    blas_accelerated,
    simd_ops::{simd_dot_f32, simd_matmul_f32, simd_matvec_f32},
};

fn regular_matmul_f32(a: &ArrayView2<f32>, b: &ArrayView2<f32>) -> Array2<f32> {
    let (m, k) = a.dim();
    let (_, n) = b.dim();
    let mut result = Array2::zeros((m, n));

    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for l in 0..k {
                sum += a[[i, l]] * b[[l, j]];
            }
            result[[i, j]] = sum;
        }
    }

    result
}

fn regular_matvec_f32(a: &ArrayView2<f32>, x: &ArrayView1<f32>) -> Array1<f32> {
    let n = a.dim().0;
    let mut result = Array1::zeros(n);

    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..a.ncols() {
            sum += a[[i, j]] * x[j];
        }
        result[i] = sum;
    }

    result
}

fn regular_dot_f32(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> f32 {
    let mut sum = 0.0;

    for i in 0..a.len() {
        sum += a[i] * b[i];
    }

    sum
}

fn create_random_array1_f32(size: usize) -> Array1<f32> {
    Array1::from_iter((0..size).map(|i| (i % 100) as f32 / 100.0))
}

fn create_random_array2_f32(rows: usize, cols: usize) -> Array2<f32> {
    Array2::from_shape_fn((rows, cols), |(i, j)| ((i * cols + j) % 100) as f32 / 100.0)
}

fn bench_matvec(c: &mut Criterion) {
    let mut group = c.benchmark_group("MatVec");

    for size in [100, 500, 1000, 5000].iter() {
        let matrix = create_random_array2_f32(*size, *size);
        let vector = create_random_array1_f32(*size);

        group.bench_with_input(BenchmarkId::new("Regular", size), &size, |b, _| {
            b.iter(|| {
                black_box(regular_matvec_f32(
                    &black_box(matrix.view()),
                    &black_box(vector.view()),
                ))
            })
        });

        group.bench_with_input(BenchmarkId::new("SIMD", size), &size, |b, _| {
            b.iter(|| {
                black_box(
                    simd_matvec_f32(&black_box(matrix.view()), &black_box(vector.view())).unwrap(),
                )
            })
        });

        group.bench_with_input(BenchmarkId::new("BLAS", size), &size, |b, _| {
            b.iter(|| {
                black_box(
                    blas_accelerated::gemv(
                        1.0,
                        &matrix.view(),
                        &vector.view(),
                        0.0,
                        &Array1::<f32>::zeros(matrix.nrows()).view(),
                    )
                    .unwrap(),
                )
            })
        });
    }

    group.finish();
}

fn bench_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("MatMul");

    for size in [50, 100, 200, 500].iter() {
        let matrix_a = create_random_array2_f32(*size, *size);
        let matrix_b = create_random_array2_f32(*size, *size);

        group.bench_with_input(BenchmarkId::new("Regular", size), &size, |b, _| {
            b.iter(|| {
                black_box(regular_matmul_f32(
                    &black_box(matrix_a.view()),
                    &black_box(matrix_b.view()),
                ))
            })
        });

        group.bench_with_input(BenchmarkId::new("SIMD", size), &size, |b, _| {
            b.iter(|| {
                black_box(
                    simd_matmul_f32(&black_box(matrix_a.view()), &black_box(matrix_b.view()))
                        .unwrap(),
                )
            })
        });

        group.bench_with_input(BenchmarkId::new("BLAS", size), &size, |b, _| {
            b.iter(|| {
                black_box(
                    blas_accelerated::matmul(
                        &black_box(matrix_a.view()),
                        &black_box(matrix_b.view()),
                    )
                    .unwrap(),
                )
            })
        });
    }

    group.finish();
}

fn bench_dot(c: &mut Criterion) {
    let mut group = c.benchmark_group("Dot");

    for size in [100, 1_000, 10_000, 100_000, 1_000_000].iter() {
        let vec_a = create_random_array1_f32(*size);
        let vec_b = create_random_array1_f32(*size);

        group.bench_with_input(BenchmarkId::new("Regular", size), &size, |b, _| {
            b.iter(|| {
                black_box(regular_dot_f32(
                    &black_box(vec_a.view()),
                    &black_box(vec_b.view()),
                ))
            })
        });

        group.bench_with_input(BenchmarkId::new("SIMD", size), &size, |b, _| {
            b.iter(|| {
                black_box(simd_dot_f32(&black_box(vec_a.view()), &black_box(vec_b.view())).unwrap())
            })
        });

        group.bench_with_input(BenchmarkId::new("BLAS", size), &size, |b, _| {
            b.iter(|| {
                black_box(
                    blas_accelerated::dot(&black_box(vec_a.view()), &black_box(vec_b.view()))
                        .unwrap(),
                )
            })
        });
    }

    group.finish();
}

criterion_group!(benches, bench_matvec, bench_matmul, bench_dot);
criterion_main!(benches);
