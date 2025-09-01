//! Benchmarks for quantized matrix-free solvers
//!
//! This benchmark compares the performance of regular iterative solvers with
//! specialized solvers for quantized matrices.

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::{Array1, Array2, ArrayView1};
use scirs2__linalg::matrixfree::{conjugate_gradient, gmres, LinearOperator};
use scirs2__linalg::quantization::{
    quantized_matrixfree::quantized_to_linear_operator,
    quantized_matrixfree::QuantizedMatrixFreeOp,
    solvers::{
        quantized_conjugate_gradient, quantized_gmres, quantized_jacobi_preconditioner,
        quantized_preconditioned_conjugate_gradient,
    },
    QuantizationMethod,
};

/// Create a random matrix with specified dimensions
#[allow(dead_code)]
fn create_randomarray2_f32(rows: usize, cols: usize) -> Array2<f32> {
    let mut rng = rand::rng();
    let mut matrix = Array2::zeros((_rows, cols));

    for i in 0.._rows {
        for j in 0..cols {
            matrix[[i, j]] = rng.random_range(-1.0..1.0);
        }
    }

    // For SPD matrices (used in CG)..we need to ensure positive definiteness
    if _rows == cols {
        // Make the matrix symmetric
        for i in 0.._rows {
            for j in i + 1..cols {
                matrix[[j, i]] = matrix[[i, j]];
            }
        }

        // Add a diagonal dominance to ensure positive definiteness
        for i in 0.._rows {
            matrix[[i, i]] += _rows as f32;
        }
    }

    matrix
}

/// Benchmark conjugate gradient solver with different matrix representations
#[allow(dead_code)]
fn bench_conjugate_gradient(c: &mut Criterion) {
    let mut group = c.benchmark_group("Conjugate Gradient Solver");
    let sizes = [10, 50, 100];
    let bit_widths = [4, 8];

    for &size in &sizes {
        // Benchmark standard conjugate gradient
        group.bench_with_input(BenchmarkId::new("Standard", size), &size, |bench, &size| {
            // Create a symmetric positive definite matrix
            let matrix = create_randomarray2_f32(size, size);

            // Create a right-hand side vector
            let b = Array1::from_vec(vec![1.0; size]);

            // Create a standard matrix-free operator
            let matrix_clone = matrix.clone();
            let standard_op =
                LinearOperator::new(size, move |v: &ArrayView1<f32>| matrix_clone.dot(v))
                    .symmetric()
                    .positive_definite();

            bench.iter(|| black_box(conjugate_gradient(&standard_op, &b, 100, 1e-6).unwrap()))
        });

        // Benchmark quantized matrix with standard solver (through LinearOperator)
        for &bits in &bit_widths {
            group.bench_with_input(
                BenchmarkId::new(
                    format!("Quantized({} bits) with Standard Solver", bits),
                    size,
                ),
                &size,
                |bench, &size| {
                    // Create a symmetric positive definite matrix
                    let matrix = create_randomarray2_f32(size, size);

                    // Create a right-hand side vector
                    let b = Array1::from_vec(vec![1.0; size]);

                    // Create a quantized matrix-free operator
                    let quantized_op = QuantizedMatrixFreeOp::frommatrix(
                        &matrix.view(),
                        bits,
                        QuantizationMethod::Symmetric,
                    )
                    .unwrap()
                    .symmetric()
                    .positive_definite();

                    // Convert to standard LinearOperator
                    let quantized_linear_op = quantized_to_linear_operator(&quantizedop);

                    bench.iter(|| {
                        black_box(conjugate_gradient(&quantized_linear_op, &b, 100, 1e-6).unwrap())
                    })
                },
            );

            // Benchmark specialized solver for quantized matrix
            group.bench_with_input(
                BenchmarkId::new(
                    format!("Quantized({} bits) with Specialized Solver", bits),
                    size,
                ),
                &size,
                |bench, &size| {
                    // Create a symmetric positive definite matrix
                    let matrix = create_randomarray2_f32(size, size);

                    // Create a right-hand side vector
                    let b = Array1::from_vec(vec![1.0; size]);

                    // Create a quantized matrix-free operator
                    let quantized_op = QuantizedMatrixFreeOp::frommatrix(
                        &matrix.view(),
                        bits,
                        QuantizationMethod::Symmetric,
                    )
                    .unwrap()
                    .symmetric()
                    .positive_definite();

                    bench.iter(|| {
                        black_box(
                            quantized_conjugate_gradient(&quantized_op, &b, 100, 1e-6, false)
                                .unwrap(),
                        )
                    })
                },
            );

            // Benchmark specialized solver with adaptive precision
            group.bench_with_input(
                BenchmarkId::new(
                    format!("Quantized({} bits) with Adaptive Precision", bits),
                    size,
                ),
                &size,
                |bench, &size| {
                    // Create a symmetric positive definite matrix
                    let matrix = create_randomarray2_f32(size, size);

                    // Create a right-hand side vector
                    let b = Array1::from_vec(vec![1.0; size]);

                    // Create a quantized matrix-free operator
                    let quantized_op = QuantizedMatrixFreeOp::frommatrix(
                        &matrix.view(),
                        bits,
                        QuantizationMethod::Symmetric,
                    )
                    .unwrap()
                    .symmetric()
                    .positive_definite();

                    bench.iter(|| {
                        black_box(
                            quantized_conjugate_gradient(&quantized_op, &b, 100, 1e-6, true)
                                .unwrap(),
                        )
                    })
                },
            );

            // Benchmark preconditioned conjugate gradient
            group.bench_with_input(
                BenchmarkId::new(format!("Preconditioned Quantized({} bits)", bits), size),
                &size,
                |bench, &size| {
                    // Create a symmetric positive definite matrix
                    let matrix = create_randomarray2_f32(size, size);

                    // Create a right-hand side vector
                    let b = Array1::from_vec(vec![1.0; size]);

                    // Create a quantized matrix-free operator
                    let quantized_op = QuantizedMatrixFreeOp::frommatrix(
                        &matrix.view(),
                        bits,
                        QuantizationMethod::Symmetric,
                    )
                    .unwrap()
                    .symmetric()
                    .positive_definite();

                    // Create a preconditioner
                    let precond = quantized_jacobi_preconditioner(&quantizedop).unwrap();

                    bench.iter(|| {
                        black_box(
                            quantized_preconditioned_conjugate_gradient(
                                &quantized_op,
                                &precond,
                                &b,
                                100,
                                1e-6,
                                false,
                            )
                            .unwrap(),
                        )
                    })
                },
            );
        }
    }

    group.finish();
}

/// Benchmark GMRES solver with different matrix representations
#[allow(dead_code)]
fn bench_gmres(c: &mut Criterion) {
    let mut group = c.benchmark_group("GMRES Solver");
    let sizes = [10, 50, 100];
    let bit_widths = [4, 8];

    for &size in &sizes {
        // Benchmark standard GMRES
        group.bench_with_input(BenchmarkId::new("Standard", size), &size, |bench, &size| {
            // Create a general matrix (not necessarily symmetric)
            let matrix = create_randomarray2_f32(size, size);

            // Create a right-hand side vector
            let b = Array1::from_vec(vec![1.0; size]);

            // Create a standard matrix-free operator
            let matrix_clone = matrix.clone();
            let standard_op =
                LinearOperator::new(size, move |v: &ArrayView1<f32>| matrix_clone.dot(v));

            bench.iter(|| black_box(gmres(&standard_op, &b, 100, 1e-6, Some(20)).unwrap()))
        });

        // Benchmark quantized matrix with standard solver (through LinearOperator)
        for &bits in &bit_widths {
            group.bench_with_input(
                BenchmarkId::new(
                    format!("Quantized({} bits) with Standard Solver", bits),
                    size,
                ),
                &size,
                |bench, &size| {
                    // Create a general matrix (not necessarily symmetric)
                    let matrix = create_randomarray2_f32(size, size);

                    // Create a right-hand side vector
                    let b = Array1::from_vec(vec![1.0; size]);

                    // Create a quantized matrix-free operator
                    let quantized_op = QuantizedMatrixFreeOp::frommatrix(
                        &matrix.view(),
                        bits,
                        QuantizationMethod::Symmetric,
                    )
                    .unwrap();

                    // Convert to standard LinearOperator
                    let quantized_linear_op = quantized_to_linear_operator(&quantizedop);

                    bench.iter(|| {
                        black_box(gmres(&quantized_linear_op, &b, 100, 1e-6, Some(20)).unwrap())
                    })
                },
            );

            // Benchmark specialized solver for quantized matrix
            group.bench_with_input(
                BenchmarkId::new(
                    format!("Quantized({} bits) with Specialized Solver", bits),
                    size,
                ),
                &size,
                |bench, &size| {
                    // Create a general matrix (not necessarily symmetric)
                    let matrix = create_randomarray2_f32(size, size);

                    // Create a right-hand side vector
                    let b = Array1::from_vec(vec![1.0; size]);

                    // Create a quantized matrix-free operator
                    let quantized_op = QuantizedMatrixFreeOp::frommatrix(
                        &matrix.view(),
                        bits,
                        QuantizationMethod::Symmetric,
                    )
                    .unwrap();

                    bench.iter(|| {
                        black_box(
                            quantized_gmres(&quantized_op, &b, 100, 1e-6, Some(20), false).unwrap(),
                        )
                    })
                },
            );

            // Benchmark specialized solver with adaptive precision
            group.bench_with_input(
                BenchmarkId::new(
                    format!("Quantized({} bits) with Adaptive Precision", bits),
                    size,
                ),
                &size,
                |bench, &size| {
                    // Create a general matrix (not necessarily symmetric)
                    let matrix = create_randomarray2_f32(size, size);

                    // Create a right-hand side vector
                    let b = Array1::from_vec(vec![1.0; size]);

                    // Create a quantized matrix-free operator
                    let quantized_op = QuantizedMatrixFreeOp::frommatrix(
                        &matrix.view(),
                        bits,
                        QuantizationMethod::Symmetric,
                    )
                    .unwrap();

                    bench.iter(|| {
                        black_box(
                            quantized_gmres(&quantized_op, &b, 100, 1e-6, Some(20), true).unwrap(),
                        )
                    })
                },
            );
        }
    }

    group.finish();
}

/// Benchmark special case: large sparse banded matrix
#[allow(dead_code)]
fn bench_large_bandedmatrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("Large Banded Matrix");

    group.bench_with_input(
        BenchmarkId::new("Standard CG", 500),
        &500,
        |bench, &size| {
            // Create a large sparse tridiagonal matrix operator
            let standard_op = LinearOperator::new(size, move |v: &ArrayView1<f32>| {
                let mut result = Array1::zeros(size);

                // Main diagonal (2.0)
                for i in 0..size {
                    result[i] += 2.0 * v[i];
                }

                // Super-diagonal (1.0)
                for i in 0..size - 1 {
                    result[i] += 1.0 * v[i + 1];
                }

                // Sub-diagonal (1.0)
                for i in 1..size {
                    result[i] += 1.0 * v[i - 1];
                }

                result
            })
            .symmetric()
            .positive_definite();

            // Create a right-hand side vector
            let b = Array1::from_vec(vec![1.0; size]);

            bench.iter(|| black_box(conjugate_gradient(&standard_op, &b, 100, 1e-6).unwrap()))
        },
    );

    group.bench_with_input(
        BenchmarkId::new("Quantized Banded CG", 500),
        &500,
        |bench, &size| {
            // Create a right-hand side vector
            let b = Array1::from_vec(vec![1.0; size]);

            // Create the bands for the tridiagonal matrix
            let main_diag = Array1::from_vec(vec![2.0; size]);
            let off_diag = Array1::from_vec(vec![1.0; size - 1]);

            // Create a quantized banded matrix operator
            let bands = vec![
                (0, main_diag.view()),
                (1, off_diag.view()),
                (-1, off_diag.view()),
            ];

            let quantized_op =
                QuantizedMatrixFreeOp::banded(size, bands, 8, QuantizationMethod::Symmetric)
                    .unwrap()
                    .symmetric()
                    .positive_definite();

            bench.iter(|| {
                black_box(
                    quantized_conjugate_gradient(&quantized_op, &b, 100, 1e-6, false).unwrap(),
                )
            })
        },
    );

    group.bench_with_input(
        BenchmarkId::new("Preconditioned Quantized Banded CG", 500),
        &500,
        |bench, &size| {
            // Create a right-hand side vector
            let b = Array1::from_vec(vec![1.0; size]);

            // Create the bands for the tridiagonal matrix
            let main_diag = Array1::from_vec(vec![2.0; size]);
            let off_diag = Array1::from_vec(vec![1.0; size - 1]);

            // Create a quantized banded matrix operator
            let bands = vec![
                (0, main_diag.view()),
                (1, off_diag.view()),
                (-1, off_diag.view()),
            ];

            let quantized_op =
                QuantizedMatrixFreeOp::banded(size, bands, 8, QuantizationMethod::Symmetric)
                    .unwrap()
                    .symmetric()
                    .positive_definite();

            // Create a preconditioner
            let precond = quantized_jacobi_preconditioner(&quantizedop).unwrap();

            bench.iter(|| {
                black_box(
                    quantized_preconditioned_conjugate_gradient(
                        &quantized_op,
                        &precond,
                        &b,
                        100,
                        1e-6,
                        false,
                    )
                    .unwrap(),
                )
            })
        },
    );

    group.finish();
}

/// Benchmark memory usage by solving progressively larger problems
#[allow(dead_code)]
fn bench_memory_usage(c: &mut Criterion) {
    let mut group = c.benchmark_group("Memory Usage");
    let sizes = [100, 500, 1000];

    for &size in &sizes {
        group.bench_with_input(
            BenchmarkId::new("Quantized CG", size),
            &size,
            |bench, &size| {
                // Create a symmetric positive definite matrix
                let matrix = create_randomarray2_f32(size, size);

                // Create a right-hand side vector
                let b = Array1::from_vec(vec![1.0; size]);

                // Use 4-bit quantization for maximum memory savings
                let bits = 4;

                // Create a quantized matrix-free operator
                let quantized_op = QuantizedMatrixFreeOp::frommatrix(
                    &matrix.view(),
                    bits,
                    QuantizationMethod::Symmetric,
                )
                .unwrap()
                .symmetric()
                .positive_definite();

                bench.iter(|| {
                    black_box(
                        quantized_conjugate_gradient(&quantized_op, &b, 50, 1e-5, false).unwrap(),
                    )
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_conjugate_gradient,
    bench_gmres,
    bench_large_bandedmatrix,
    bench_memory_usage
);
criterion_main!(benches);
