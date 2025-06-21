//! Comprehensive benchmarks for linear algebra functions
//!
//! This benchmark suite covers all major operation categories in scirs2-linalg
//! to ensure comprehensive performance monitoring.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ndarray::{s, Array1, Array2};
use scirs2_linalg::blas::{dot, nrm2};
use scirs2_linalg::mixed_precision::{
    mixed_precision_dot, mixed_precision_matmul, mixed_precision_solve,
};
use scirs2_linalg::prelude::*;
use scirs2_linalg::structured::{CirculantMatrix, ToeplitzMatrix};
use scirs2_linalg::*;
use std::time::Duration;

/// Create a well-conditioned test matrix
fn create_test_matrix(n: usize) -> Array2<f64> {
    let mut matrix = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            if i == j {
                matrix[[i, j]] = (i + 1) as f64; // Diagonal dominance
            } else {
                matrix[[i, j]] = 0.1 * ((i * n + j) as f64 * 0.01).sin();
            }
        }
    }
    matrix
}

/// Create a symmetric positive definite matrix
fn create_spd_matrix(n: usize) -> Array2<f64> {
    let a = Array2::from_shape_fn((n, n), |(i, j)| ((i + j + 1) as f64 * 0.1).sin());
    a.t().dot(&a) + Array2::<f64>::eye(n) * (n as f64)
}

/// Create a test vector
fn create_test_vector(n: usize) -> Array1<f64> {
    Array1::from_shape_fn(n, |i| ((i + 1) as f64 * 0.1).sin())
}

/// Benchmark basic BLAS operations
fn bench_blas_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("blas_operations");

    for &size in &[100, 1000, 10000] {
        let x = create_test_vector(size);
        let y = create_test_vector(size);
        let matrix = create_test_matrix(size.min(500)); // Limit matrix size for efficiency

        group.throughput(Throughput::Elements(size as u64));

        // Vector dot product
        group.bench_with_input(BenchmarkId::new("dot", size), &(&x, &y), |b, (x, y)| {
            b.iter(|| dot(black_box(&x.view()), black_box(&y.view())))
        });

        // Vector norm
        group.bench_with_input(BenchmarkId::new("nrm2", size), &x, |b, x| {
            b.iter(|| nrm2(black_box(&x.view())))
        });

        // Matrix-vector multiplication (if matrix size allows)
        if size <= 500 {
            let mv_size = size.min(matrix.nrows());
            let v = x.slice(s![..mv_size]).to_owned();
            group.bench_with_input(
                BenchmarkId::new("matvec", mv_size),
                &(&matrix, &v),
                |b, (m, v)| b.iter(|| m.dot(v)), // Remove black_box from inside dot
            );
        }
    }

    group.finish();
}

/// Benchmark iterative solvers
fn bench_iterative_solvers(c: &mut Criterion) {
    let mut group = c.benchmark_group("iterative_solvers");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));

    for &size in &[50, 100, 200] {
        let matrix = create_spd_matrix(size);
        let rhs = create_test_vector(size);

        group.throughput(Throughput::Elements(size as u64 * size as u64));

        // Conjugate Gradient
        group.bench_with_input(
            BenchmarkId::new("conjugate_gradient", size),
            &(&matrix, &rhs),
            |b, (m, r)| {
                b.iter(|| {
                    conjugate_gradient(black_box(&m.view()), black_box(&r.view()), 100, 1e-10, None)
                })
            },
        );

        // Jacobi method
        group.bench_with_input(
            BenchmarkId::new("jacobi_method", size),
            &(&matrix, &rhs),
            |b, (m, r)| {
                b.iter(|| {
                    jacobi_method(black_box(&m.view()), black_box(&r.view()), 50, 1e-10, None)
                })
            },
        );

        // Gauss-Seidel
        group.bench_with_input(
            BenchmarkId::new("gauss_seidel", size),
            &(&matrix, &rhs),
            |b, (m, r)| {
                b.iter(|| gauss_seidel(black_box(&m.view()), black_box(&r.view()), 50, 1e-10, None))
            },
        );
    }

    group.finish();
}

/// Benchmark mixed precision operations
fn bench_mixed_precision(c: &mut Criterion) {
    let mut group = c.benchmark_group("mixed_precision");

    for &size in &[50, 100, 200] {
        let matrix_f64 = create_test_matrix(size);
        let matrix_f32 = matrix_f64.mapv(|x| x as f32);
        let vector_f64 = create_test_vector(size);
        let vector_f32 = vector_f64.mapv(|x| x as f32);

        group.throughput(Throughput::Elements(size as u64 * size as u64));

        // Mixed precision matrix multiplication
        group.bench_with_input(
            BenchmarkId::new("mixed_matmul", size),
            &(&matrix_f32, &matrix_f32),
            |bencher, (a, mat_b)| {
                bencher.iter(|| {
                    mixed_precision_matmul::<f32, f32, f64, f32>(
                        black_box(&a.view()),
                        black_box(&mat_b.view()),
                    )
                })
            },
        );

        // Mixed precision dot product
        group.bench_with_input(
            BenchmarkId::new("mixed_dot", size),
            &(&vector_f32, &vector_f32),
            |b, (x, y)| {
                b.iter(|| {
                    mixed_precision_dot::<f32, f32, f64, f64>(
                        black_box(&x.view()),
                        black_box(&y.view()),
                    )
                })
            },
        );

        // Mixed precision solve
        group.bench_with_input(
            BenchmarkId::new("mixed_solve", size),
            &(&matrix_f32, &vector_f32),
            |b, (m, v)| {
                b.iter(|| mixed_precision_solve(black_box(&m.view()), black_box(&v.view())))
            },
        );
    }

    group.finish();
}

/// Benchmark structured matrix operations
fn bench_structured_matrices(c: &mut Criterion) {
    let mut group = c.benchmark_group("structured_matrices");

    for &size in &[50, 100, 200] {
        let first_row = create_test_vector(size);
        let first_col = create_test_vector(size);
        let rhs = create_test_vector(size);

        group.throughput(Throughput::Elements(size as u64 * size as u64));

        // Toeplitz matrix solve
        group.bench_with_input(
            BenchmarkId::new("toeplitz_solve", size),
            &(&first_row, &rhs),
            |b, (r, rhs)| {
                b.iter(|| {
                    let toeplitz = ToeplitzMatrix::new(r.clone(), r.clone()).unwrap();
                    toeplitz.solve(black_box(&rhs.view()))
                })
            },
        );

        // Circulant matrix solve
        group.bench_with_input(
            BenchmarkId::new("circulant_solve", size),
            &(&first_row, &rhs),
            |b, (r, rhs)| {
                b.iter(|| {
                    let mut circulant = CirculantMatrix::new(r.clone()).unwrap();
                    circulant.solve(black_box(&rhs.view()))
                })
            },
        );
    }

    group.finish();
}

/// Benchmark matrix factorizations
fn bench_matrix_factorizations(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_factorizations");
    group.sample_size(10);

    for &size in &[20, 50, 100] {
        let matrix = create_test_matrix(size);

        group.throughput(Throughput::Elements(size as u64 * size as u64));

        // Non-negative Matrix Factorization
        group.bench_with_input(BenchmarkId::new("nmf", size), &matrix, |b, m| {
            let non_negative = m.mapv(|x| x.abs());
            b.iter(|| nmf(black_box(&non_negative.view()), 10, 50, 1e-6))
        });

        // CUR decomposition
        group.bench_with_input(
            BenchmarkId::new("cur_decomposition", size),
            &matrix,
            |b, m| {
                b.iter(|| cur_decomposition(black_box(&m.view()), 10, None, None, "deterministic"))
            },
        );

        // Rank-revealing QR
        group.bench_with_input(
            BenchmarkId::new("rank_revealing_qr", size),
            &matrix,
            |b, m| b.iter(|| rank_revealing_qr(black_box(&m.view()), 1e-12)),
        );
    }

    group.finish();
}

/// Benchmark complex number operations
fn bench_complex_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("complex_operations");

    for &size in &[50, 100, 200] {
        use num_complex::Complex64;
        let matrix = Array2::from_shape_fn((size, size), |(i, j)| {
            Complex64::new(((i + j) as f64 * 0.1).sin(), ((i - j) as f64 * 0.1).cos())
        });

        group.throughput(Throughput::Elements(size as u64 * size as u64));

        // Complex matrix multiplication
        group.bench_with_input(
            BenchmarkId::new("complex_matmul", size),
            &(&matrix, &matrix),
            |bencher, (a, mat_b)| {
                bencher.iter(|| complex_matmul(black_box(&a.view()), black_box(&mat_b.view())))
            },
        );

        // Complex determinant
        group.bench_with_input(BenchmarkId::new("complex_det", size), &matrix, |b, m| {
            b.iter(|| complex_det(black_box(&m.view())))
        });

        // Complex inverse
        group.bench_with_input(
            BenchmarkId::new("complex_inverse", size),
            &matrix,
            |b, m| b.iter(|| complex_inverse(black_box(&m.view()))),
        );
    }

    group.finish();
}

/// Benchmark random matrix generation
fn bench_random_matrices(c: &mut Criterion) {
    let mut group = c.benchmark_group("random_matrices");

    for &size in &[100, 200, 500] {
        group.throughput(Throughput::Elements(size as u64 * size as u64));

        // Random uniform matrix
        group.bench_with_input(BenchmarkId::new("uniform", size), &size, |b, &s| {
            b.iter(|| {
                uniform(
                    black_box(s),
                    black_box(s),
                    black_box(-1.0),
                    black_box(1.0),
                    black_box(None),
                )
            })
        });

        // Random normal matrix
        group.bench_with_input(BenchmarkId::new("normal", size), &size, |b, &s| {
            b.iter(|| {
                normal(
                    black_box(s),
                    black_box(s),
                    black_box(0.0),
                    black_box(1.0),
                    black_box(None),
                )
            })
        });

        // Random orthogonal matrix
        group.bench_with_input(BenchmarkId::new("orthogonal", size), &size, |b, &s| {
            b.iter(|| orthogonal(black_box(s), black_box(None)))
        });

        // Random SPD matrix
        group.bench_with_input(BenchmarkId::new("spd", size), &size, |b, &s| {
            b.iter(|| {
                spd(
                    black_box(s),
                    black_box(1.0),
                    black_box(10.0),
                    black_box(None),
                )
            })
        });
    }

    group.finish();
}

/// Benchmark Kronecker operations
fn bench_kronecker_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("kronecker_operations");

    for &size in &[10, 20, 30] {
        // Keep sizes small for Kronecker products
        let matrix_a = create_test_matrix(size);
        let matrix_b = create_test_matrix(size);
        let vector = create_test_vector(size * size);

        group.throughput(Throughput::Elements(
            size as u64 * size as u64 * size as u64 * size as u64,
        ));

        // Kronecker product
        group.bench_with_input(
            BenchmarkId::new("kron", size),
            &(&matrix_a, &matrix_b),
            |bencher, (a, mat_b)| {
                bencher.iter(|| kron(black_box(&a.view()), black_box(&mat_b.view())))
            },
        );

        // Kronecker matrix-vector multiplication
        group.bench_with_input(
            BenchmarkId::new("kron_matvec", size),
            &(&matrix_a, &matrix_b, &vector),
            |bencher, (a, mat_b, v)| {
                bencher.iter(|| {
                    kron_matvec(
                        black_box(&a.view()),
                        black_box(&mat_b.view()),
                        black_box(&v.view()),
                    )
                })
            },
        );
    }

    group.finish();
}

/// Benchmark projection operations
fn bench_projection_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("projection_operations");

    for &size in &[100, 500, 1000] {
        let matrix = create_test_matrix(size.min(200)); // Limit for efficiency
        let target_dim = size / 4;

        group.throughput(Throughput::Elements(size as u64 * size as u64));

        // Gaussian random projection
        group.bench_with_input(
            BenchmarkId::new("gaussian_projection", size),
            &(&matrix, target_dim),
            |bencher, (m, d)| {
                let proj_matrix =
                    gaussian_random_matrix(black_box(*d), black_box(m.ncols())).unwrap();
                bencher.iter(|| project(black_box(&m.view()), black_box(&proj_matrix.view())))
            },
        );

        // Johnson-Lindenstrauss transform
        if size <= 500 {
            group.bench_with_input(
                BenchmarkId::new("jl_transform", size),
                &matrix,
                |bencher, m| {
                    bencher.iter(|| {
                        johnson_lindenstrauss_transform(black_box(&m.view()), black_box(0.1))
                    })
                },
            );
        }
    }

    group.finish();
}

// Group all benchmarks
criterion_group!(
    benches,
    bench_blas_operations,
    bench_iterative_solvers,
    bench_mixed_precision,
    bench_structured_matrices,
    bench_matrix_factorizations,
    bench_complex_operations,
    bench_random_matrices,
    bench_kronecker_operations,
    bench_projection_operations
);

criterion_main!(benches);
