use criterion::{
    black_box, criterion_group, criterion_main, AxisScale, BenchmarkId, Criterion,
    PlotConfiguration,
};
use ndarray::{Array1, Array2};
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use scirs2_linalg::{
    cholesky, det, eigen, inv, lstsq, lu, matrix_norm, qr, solve, solve_triangular, svd,
};
use std::time::Instant;

// Benchmark configuration
const MATRIX_SIZES: &[usize] = &[10, 50, 100, 200, 500, 1000];
const SEED: u64 = 42;

/// Generate a random matrix with controlled properties
fn generate_matrix(n: usize, condition_number: Option<f64>) -> Array2<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(SEED);

    match condition_number {
        Some(cond) => {
            // Generate matrix with specific condition number for numerical stability tests
            let u = Array2::random_using((n, n), Uniform::new(-1.0, 1.0), &mut rng);
            let mut s = Array1::linspace(1.0, 1.0 / cond, n);
            s.mapv_inplace(|x| x.abs()); // Ensure positive singular values

            // Simple construction: U * diag(s) * V^T (with V = U for simplicity)
            let mut result = Array2::zeros((n, n));
            for i in 0..n {
                for j in 0..n {
                    for k in 0..n {
                        result[[i, j]] += u[[i, k]] * s[k] * u[[j, k]];
                    }
                }
            }
            result
        }
        None => {
            // Generate well-conditioned random matrix
            Array2::random_using((n, n), Uniform::new(-1.0, 1.0), &mut rng)
        }
    }
}

/// Generate a symmetric positive definite matrix
fn generate_spd_matrix(n: usize) -> Array2<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(SEED);
    let a = Array2::random_using((n, n), Uniform::new(-1.0, 1.0), &mut rng);

    // A^T * A is always positive definite
    let at = a.t();
    at.dot(&a) + Array2::<f64>::eye(n) * 0.1 // Add small diagonal term for numerical stability
}

/// Benchmark basic matrix operations
fn bench_basic_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("basic_operations");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for &size in MATRIX_SIZES {
        let matrix = generate_matrix(size, None);

        // Matrix determinant
        group.bench_with_input(BenchmarkId::new("determinant", size), &size, |b, _| {
            b.iter(|| {
                let result = det(&matrix.view(), None);
                black_box(result)
            })
        });

        // Matrix inverse
        if size <= 500 {
            // Limit inverse to smaller matrices
            group.bench_with_input(BenchmarkId::new("inverse", size), &size, |b, _| {
                b.iter(|| {
                    let result = inv(&matrix.view(), None);
                    black_box(result)
                })
            });
        }

        // Matrix norms
        group.bench_with_input(BenchmarkId::new("frobenius_norm", size), &size, |b, _| {
            b.iter(|| {
                let result = matrix_norm(&matrix.view(), "frobenius", None);
                black_box(result)
            })
        });

        group.bench_with_input(BenchmarkId::new("spectral_norm", size), &size, |b, _| {
            b.iter(|| {
                let result = matrix_norm(&matrix.view(), "2", None);
                black_box(result)
            })
        });
    }

    group.finish();
}

/// Benchmark matrix decompositions
fn bench_decompositions(c: &mut Criterion) {
    let mut group = c.benchmark_group("decompositions");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for &size in MATRIX_SIZES {
        let matrix = generate_matrix(size, None);
        let spd_matrix = generate_spd_matrix(size);

        // LU decomposition
        group.bench_with_input(BenchmarkId::new("lu_decomposition", size), &size, |b, _| {
            b.iter(|| {
                let result = lu(&matrix.view(), None);
                black_box(result)
            })
        });

        // QR decomposition
        group.bench_with_input(BenchmarkId::new("qr_decomposition", size), &size, |b, _| {
            b.iter(|| {
                let result = qr(&matrix.view(), None);
                black_box(result)
            })
        });

        // SVD decomposition
        if size <= 200 {
            // SVD is expensive, limit to smaller matrices
            group.bench_with_input(
                BenchmarkId::new("svd_decomposition", size),
                &size,
                |b, _| {
                    b.iter(|| {
                        let result = svd(&matrix.view(), false, None);
                        black_box(result)
                    })
                },
            );
        }

        // Cholesky decomposition (for SPD matrices)
        group.bench_with_input(
            BenchmarkId::new("cholesky_decomposition", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let result = cholesky(&spd_matrix.view(), None);
                    black_box(result)
                })
            },
        );
    }

    group.finish();
}

/// Benchmark linear system solvers
fn bench_linear_solvers(c: &mut Criterion) {
    let mut group = c.benchmark_group("linear_solvers");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for &size in MATRIX_SIZES {
        let matrix = generate_matrix(size, None);
        let spd_matrix = generate_spd_matrix(size);
        let rhs = Array1::from_vec((0..size).map(|i| i as f64).collect());

        // General linear solver
        group.bench_with_input(BenchmarkId::new("general_solve", size), &size, |b, _| {
            b.iter(|| {
                let result = solve(&matrix.view(), &rhs.view(), None);
                black_box(result)
            })
        });

        // Least squares solver
        group.bench_with_input(BenchmarkId::new("least_squares", size), &size, |b, _| {
            b.iter(|| {
                let result = lstsq(&matrix.view(), &rhs.view(), None);
                black_box(result)
            })
        });

        // Triangular solver (using Cholesky factor)
        if let Ok(l) = cholesky(&spd_matrix.view(), None) {
            group.bench_with_input(BenchmarkId::new("triangular_solve", size), &size, |b, _| {
                b.iter(|| {
                    let result = solve_triangular(&l.view(), &rhs.view(), true, false);
                    black_box(result)
                })
            });
        }
    }

    group.finish();
}

/// Benchmark eigenvalue computations
fn bench_eigenvalues(c: &mut Criterion) {
    let mut group = c.benchmark_group("eigenvalues");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    // Limit eigenvalue computations to smaller matrices due to computational cost
    let eigen_sizes = &[10, 25, 50, 100];

    for &size in eigen_sizes {
        let spd_matrix = generate_spd_matrix(size);

        // Symmetric eigenvalue problem
        group.bench_with_input(
            BenchmarkId::new("symmetric_eigenvalues", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let result = eigen::eigvalsh(&spd_matrix.view(), None);
                    black_box(result)
                })
            },
        );

        // Symmetric eigenvectors
        group.bench_with_input(
            BenchmarkId::new("symmetric_eigenvectors", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let result = eigen::eigh(&spd_matrix.view(), None);
                    black_box(result)
                })
            },
        );
    }

    group.finish();
}

/// Benchmark numerical stability with ill-conditioned matrices
fn bench_numerical_stability(c: &mut Criterion) {
    let mut group = c.benchmark_group("numerical_stability");

    let test_sizes = &[50, 100, 200];
    let condition_numbers = &[1e3, 1e6, 1e12];

    for &size in test_sizes {
        for &cond in condition_numbers {
            let matrix = generate_matrix(size, Some(cond));

            group.bench_with_input(
                BenchmarkId::new(format!("solve_cond_{:.0e}", cond), size),
                &size,
                |b, _| {
                    let rhs = Array1::ones(size);
                    b.iter(|| {
                        let result = solve(&matrix.view(), &rhs.view(), None);
                        black_box(result)
                    })
                },
            );

            group.bench_with_input(
                BenchmarkId::new(format!("lstsq_cond_{:.0e}", cond), size),
                &size,
                |b, _| {
                    let rhs = Array1::ones(size);
                    b.iter(|| {
                        let result = lstsq(&matrix.view(), &rhs.view(), None);
                        black_box(result)
                    })
                },
            );
        }
    }

    group.finish();
}

/// Benchmark memory efficiency by measuring peak memory usage
fn bench_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_efficiency");

    for &size in &[100, 500, 1000] {
        let matrix = generate_matrix(size, None);

        group.bench_with_input(
            BenchmarkId::new("memory_efficient_solve", size),
            &size,
            |b, _| {
                b.iter_custom(|iters| {
                    let start = Instant::now();

                    for _ in 0..iters {
                        let rhs = Array1::ones(size);
                        let _result = solve(&matrix.view(), &rhs.view(), None);
                        // Force deallocation
                        drop(_result);
                    }

                    start.elapsed()
                })
            },
        );
    }

    group.finish();
}

/// Performance analysis and reporting
fn performance_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("performance_analysis");

    // Test different algorithm complexity patterns
    let sizes = vec![10, 20, 50, 100, 200, 500];

    for size in sizes {
        let matrix = generate_matrix(size, None);

        // O(n^3) operations
        group.bench_with_input(
            BenchmarkId::new("matrix_multiply_complexity", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let result = matrix.dot(&matrix);
                    black_box(result)
                })
            },
        );

        // O(n^2) operations
        group.bench_with_input(
            BenchmarkId::new("matrix_add_complexity", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let result = &matrix + &matrix;
                    black_box(result)
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_basic_operations,
    bench_decompositions,
    bench_linear_solvers,
    bench_eigenvalues,
    bench_numerical_stability,
    bench_memory_efficiency,
    performance_analysis
);

criterion_main!(benches);
