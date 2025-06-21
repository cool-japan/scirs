//! Performance benchmarks for SciPy compatibility layer
//!
//! This benchmark suite measures the performance of SciPy-compatible functions
//! and compares them against pure Rust implementations where available.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ndarray::{Array1, Array2, Axis};
use scirs2_linalg::matrix_functions::{expm, logm, sqrtm};
use scirs2_linalg::norm::{matrix_norm, vector_norm};
use scirs2_linalg::{
    cholesky, compat, cond, det, eig, eigh, eigvals, eigvalsh, inv, lstsq, lu, matrix_rank, qr,
    solve, svd,
};
use std::time::Duration;

/// Generate a well-conditioned test matrix of given size
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

/// Generate a random-like vector of given size
fn create_test_vector(n: usize) -> Array1<f64> {
    Array1::from_shape_fn(n, |i| ((i + 1) as f64 * 0.1).sin())
}

/// Generate a symmetric positive definite matrix
fn create_spd_matrix(n: usize) -> Array2<f64> {
    let a = Array2::from_shape_fn((n, n), |(i, j)| ((i + j + 1) as f64 * 0.1).sin());
    a.t().dot(&a) + Array2::eye(n) * (n as f64) // A^T * A + n*I
}

/// Benchmark basic matrix operations
fn bench_basic_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("basic_operations");

    for &size in &[10, 50, 100] {
        let matrix = create_test_matrix(size);

        // Benchmark determinant calculation
        group.throughput(Throughput::Elements(size as u64 * size as u64));
        group.bench_with_input(BenchmarkId::new("det_compat", size), &matrix, |b, m| {
            b.iter(|| compat::det(&m.view(), false, true).unwrap())
        });

        group.bench_with_input(BenchmarkId::new("det_basic", size), &matrix, |b, m| {
            b.iter(|| det(&m.view(), None).unwrap())
        });

        // Benchmark matrix inverse
        group.bench_with_input(BenchmarkId::new("inv_compat", size), &matrix, |b, m| {
            b.iter(|| compat::inv(&m.view(), false, true).unwrap())
        });

        group.bench_with_input(BenchmarkId::new("inv_basic", size), &matrix, |b, m| {
            b.iter(|| inv(&m.view(), None).unwrap())
        });
    }

    group.finish();
}

/// Benchmark matrix norms
fn bench_matrix_norms(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_norms");

    for &size in &[20, 100, 200] {
        let matrix = create_test_matrix(size);

        group.throughput(Throughput::Elements(size as u64 * size as u64));

        // Frobenius norm
        group.bench_with_input(
            BenchmarkId::new("frobenius_compat", size),
            &matrix,
            |b, m| b.iter(|| compat::norm(&m.view(), Some("fro"), None, false, true).unwrap()),
        );

        group.bench_with_input(
            BenchmarkId::new("frobenius_basic", size),
            &matrix,
            |b, m| b.iter(|| matrix_norm(&m.view(), "frobenius").unwrap()),
        );

        // 1-norm
        group.bench_with_input(BenchmarkId::new("norm_1_compat", size), &matrix, |b, m| {
            b.iter(|| compat::norm(&m.view(), Some("1"), None, false, true).unwrap())
        });

        group.bench_with_input(BenchmarkId::new("norm_1_basic", size), &matrix, |b, m| {
            b.iter(|| matrix_norm(&m.view(), "1").unwrap())
        });

        // Infinity norm
        group.bench_with_input(
            BenchmarkId::new("norm_inf_compat", size),
            &matrix,
            |b, m| b.iter(|| compat::norm(&m.view(), Some("inf"), None, false, true).unwrap()),
        );

        group.bench_with_input(BenchmarkId::new("norm_inf_basic", size), &matrix, |b, m| {
            b.iter(|| norm_mod::matrix_norm(&m.view(), "inf").unwrap())
        });
    }

    group.finish();
}

/// Benchmark vector norms
fn bench_vector_norms(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_norms");

    for &size in &[100, 1000, 10000] {
        let vector = create_test_vector(size);

        group.throughput(Throughput::Elements(size as u64));

        // 2-norm
        group.bench_with_input(
            BenchmarkId::new("vector_2norm_compat", size),
            &vector,
            |b, v| b.iter(|| compat::vector_norm(&v.view(), Some(2.0), true).unwrap()),
        );

        group.bench_with_input(
            BenchmarkId::new("vector_2norm_basic", size),
            &vector,
            |b, v| b.iter(|| norm_mod::vector_norm(&v.view(), "2").unwrap()),
        );

        // 1-norm
        group.bench_with_input(
            BenchmarkId::new("vector_1norm_compat", size),
            &vector,
            |b, v| b.iter(|| compat::vector_norm(&v.view(), Some(1.0), true).unwrap()),
        );

        group.bench_with_input(
            BenchmarkId::new("vector_1norm_basic", size),
            &vector,
            |b, v| b.iter(|| norm_mod::vector_norm(&v.view(), "1").unwrap()),
        );

        // Infinity norm
        group.bench_with_input(
            BenchmarkId::new("vector_infnorm_compat", size),
            &vector,
            |b, v| b.iter(|| compat::vector_norm(&v.view(), Some(f64::INFINITY), true).unwrap()),
        );

        group.bench_with_input(
            BenchmarkId::new("vector_infnorm_basic", size),
            &vector,
            |b, v| b.iter(|| norm_mod::vector_norm(&v.view(), "inf").unwrap()),
        );
    }

    group.finish();
}

/// Benchmark matrix decompositions
fn bench_decompositions(c: &mut Criterion) {
    let mut group = c.benchmark_group("decompositions");
    group.sample_size(10); // Reduce sample size for expensive operations
    group.measurement_time(Duration::from_secs(30));

    for &size in &[20, 50, 100] {
        let matrix = create_test_matrix(size);
        let spd_matrix = create_spd_matrix(size);

        group.throughput(Throughput::Elements(size as u64 * size as u64));

        // LU decomposition
        group.bench_with_input(BenchmarkId::new("lu_compat", size), &matrix, |b, m| {
            b.iter(|| compat::lu(&m.view(), false, false, true, false).unwrap())
        });

        group.bench_with_input(BenchmarkId::new("lu_basic", size), &matrix, |b, m| {
            b.iter(|| decomposition::lu(&m.view()).unwrap())
        });

        // QR decomposition
        group.bench_with_input(BenchmarkId::new("qr_compat", size), &matrix, |b, m| {
            b.iter(|| compat::qr(&m.view(), false, None, "full", false, true).unwrap())
        });

        group.bench_with_input(BenchmarkId::new("qr_basic", size), &matrix, |b, m| {
            b.iter(|| decomposition::qr(&m.view()).unwrap())
        });

        // SVD (smaller sizes due to computational cost)
        if size <= 50 {
            group.bench_with_input(BenchmarkId::new("svd_compat", size), &matrix, |b, m| {
                b.iter(|| compat::svd(&m.view(), true, true, false, true, "gesdd").unwrap())
            });

            group.bench_with_input(BenchmarkId::new("svd_basic", size), &matrix, |b, m| {
                b.iter(|| decomposition::svd(&m.view(), false).unwrap())
            });
        }

        // Cholesky decomposition (SPD matrices only)
        group.bench_with_input(
            BenchmarkId::new("cholesky_compat", size),
            &spd_matrix,
            |b, m| b.iter(|| compat::cholesky(&m.view(), true, false, true).unwrap()),
        );

        group.bench_with_input(
            BenchmarkId::new("cholesky_basic", size),
            &spd_matrix,
            |b, m| b.iter(|| decomposition::cholesky(&m.view()).unwrap()),
        );
    }

    group.finish();
}

/// Benchmark eigenvalue computations
fn bench_eigenvalues(c: &mut Criterion) {
    let mut group = c.benchmark_group("eigenvalues");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(20));

    for &size in &[20, 50, 100] {
        let spd_matrix = create_spd_matrix(size);

        group.throughput(Throughput::Elements(size as u64 * size as u64));

        // Eigenvalues only
        group.bench_with_input(
            BenchmarkId::new("eigvals_compat", size),
            &spd_matrix,
            |b, m| {
                b.iter(|| {
                    compat::eigh(
                        &m.view(),
                        None,
                        false,
                        true,
                        false,
                        false,
                        true,
                        None,
                        None,
                        None,
                        1,
                    )
                    .unwrap()
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("eigvals_basic", size),
            &spd_matrix,
            |b, m| b.iter(|| eigen::eigvalsh(&m.view()).unwrap()),
        );

        // Eigenvalues and eigenvectors (smaller sizes)
        if size <= 50 {
            group.bench_with_input(
                BenchmarkId::new("eigh_compat", size),
                &spd_matrix,
                |b, m| {
                    b.iter(|| {
                        compat::eigh(
                            &m.view(),
                            None,
                            false,
                            false,
                            false,
                            false,
                            true,
                            None,
                            None,
                            None,
                            1,
                        )
                        .unwrap()
                    })
                },
            );

            group.bench_with_input(BenchmarkId::new("eigh_basic", size), &spd_matrix, |b, m| {
                b.iter(|| eigen::eigh(&m.view()).unwrap())
            });
        }
    }

    group.finish();
}

/// Benchmark linear system solvers
fn bench_linear_solvers(c: &mut Criterion) {
    let mut group = c.benchmark_group("linear_solvers");
    group.sample_size(20);

    for &size in &[20, 50, 100] {
        let matrix = create_test_matrix(size);
        let rhs_2d = create_test_vector(size).insert_axis(Axis(1));
        let rhs_1d = create_test_vector(size);

        group.throughput(Throughput::Elements(size as u64 * size as u64));

        // General solve
        group.bench_with_input(
            BenchmarkId::new("solve_compat", size),
            &(&matrix, &rhs_2d),
            |b, (m, r)| {
                b.iter(|| {
                    compat::compat_solve(
                        &m.view(),
                        &r.view(),
                        false,
                        false,
                        false,
                        true,
                        None,
                        false,
                    )
                    .unwrap()
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("solve_basic", size),
            &(&matrix, &rhs_1d),
            |b, (m, r)| b.iter(|| solve::solve(&m.view(), &r.view()).unwrap()),
        );

        // Least squares (overdetermined system)
        if size >= 20 {
            let overdetermined = matrix.slice(ndarray::s![..size + 10, ..]).to_owned();
            let overdetermined_rhs = create_test_vector(size + 10).insert_axis(Axis(1));

            group.bench_with_input(
                BenchmarkId::new("lstsq_compat", size),
                &(&overdetermined, &overdetermined_rhs),
                |b, (m, r)| {
                    b.iter(|| {
                        compat::lstsq(&m.view(), &r.view(), None, false, false, true, None).unwrap()
                    })
                },
            );

            group.bench_with_input(
                BenchmarkId::new("lstsq_basic", size),
                &(&overdetermined, &overdetermined_rhs.column(0).to_owned()),
                |b, (m, r)| b.iter(|| solve::lstsq(&m.view(), &r.view()).unwrap()),
            );
        }
    }

    group.finish();
}

/// Benchmark matrix functions
fn bench_matrix_functions(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_functions");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));

    for &size in &[10, 20, 30] {
        // Smaller sizes for expensive matrix functions
        let matrix = create_spd_matrix(size) * 0.1; // Scale down for numerical stability

        group.throughput(Throughput::Elements(size as u64 * size as u64));

        // Matrix exponential
        group.bench_with_input(BenchmarkId::new("expm_compat", size), &matrix, |b, m| {
            b.iter(|| compat::expm(&m.view()).unwrap())
        });

        group.bench_with_input(BenchmarkId::new("expm_basic", size), &matrix, |b, m| {
            b.iter(|| matrix_functions::expm(&m.view()).unwrap())
        });

        // Matrix square root
        group.bench_with_input(BenchmarkId::new("sqrtm_compat", size), &matrix, |b, m| {
            b.iter(|| compat::sqrtm(&m.view()).unwrap())
        });

        group.bench_with_input(BenchmarkId::new("sqrtm_basic", size), &matrix, |b, m| {
            b.iter(|| matrix_functions::sqrtm(&m.view(), 100, 1e-12).unwrap())
        });

        // Matrix logarithm
        if size <= 20 {
            // Even smaller for log
            group.bench_with_input(BenchmarkId::new("logm_compat", size), &matrix, |b, m| {
                b.iter(|| compat::logm(&m.view()).unwrap())
            });

            group.bench_with_input(BenchmarkId::new("logm_basic", size), &matrix, |b, m| {
                b.iter(|| matrix_functions::logm(&m.view()).unwrap())
            });
        }
    }

    group.finish();
}

/// Benchmark condition numbers and matrix properties
fn bench_matrix_properties(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_properties");

    for &size in &[20, 50, 100] {
        let matrix = create_test_matrix(size);

        group.throughput(Throughput::Elements(size as u64 * size as u64));

        // Condition number
        group.bench_with_input(BenchmarkId::new("cond_compat", size), &matrix, |b, m| {
            b.iter(|| compat::cond(&m.view(), Some("2")).unwrap())
        });

        group.bench_with_input(BenchmarkId::new("cond_basic", size), &matrix, |b, m| {
            b.iter(|| norm_mod::cond(&m.view()).unwrap())
        });

        // Matrix rank
        group.bench_with_input(BenchmarkId::new("rank_compat", size), &matrix, |b, m| {
            b.iter(|| compat::matrix_rank(&m.view(), None, false, true).unwrap())
        });

        group.bench_with_input(BenchmarkId::new("rank_basic", size), &matrix, |b, m| {
            b.iter(|| norm_mod::matrix_rank(&m.view(), None).unwrap())
        });

        // Pseudoinverse (smaller sizes)
        if size <= 50 {
            group.bench_with_input(BenchmarkId::new("pinv_compat", size), &matrix, |b, m| {
                b.iter(|| compat::pinv(&m.view(), None, false, true).unwrap())
            });
        }
    }

    group.finish();
}

/// Benchmark advanced decompositions
fn bench_advanced_decompositions(c: &mut Criterion) {
    let mut group = c.benchmark_group("advanced_decompositions");
    group.sample_size(10);

    for &size in &[10, 20, 30] {
        let matrix = create_spd_matrix(size);

        group.throughput(Throughput::Elements(size as u64 * size as u64));

        // RQ decomposition
        group.bench_with_input(BenchmarkId::new("rq_compat", size), &matrix, |b, m| {
            b.iter(|| compat::rq(&m.view(), false, None, "full", true).unwrap())
        });

        // Polar decomposition
        group.bench_with_input(BenchmarkId::new("polar_right", size), &matrix, |b, m| {
            b.iter(|| compat::polar(&m.view(), "right").unwrap())
        });

        group.bench_with_input(BenchmarkId::new("polar_left", size), &matrix, |b, m| {
            b.iter(|| compat::polar(&m.view(), "left").unwrap())
        });
    }

    group.finish();
}

/// Benchmark utility functions
fn bench_utility_functions(c: &mut Criterion) {
    let mut group = c.benchmark_group("utility_functions");

    for &num_blocks in &[2, 5, 10] {
        let block_size = 10;
        let blocks: Vec<Array2<f64>> = (0..num_blocks)
            .map(|i| create_test_matrix(block_size + i % 3)) // Varying block sizes
            .collect();
        let block_views: Vec<_> = blocks.iter().map(|b| b.view()).collect();

        let total_elements = blocks.iter().map(|b| b.len()).sum::<usize>();
        group.throughput(Throughput::Elements(total_elements as u64));

        group.bench_with_input(
            BenchmarkId::new("block_diag", num_blocks),
            &block_views,
            |b, blocks| b.iter(|| compat::block_diag(&blocks).unwrap()),
        );
    }

    group.finish();
}

/// Memory allocation benchmark
fn bench_memory_allocation(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_allocation");

    for &size in &[50, 100, 200] {
        group.throughput(Throughput::Elements(size as u64 * size as u64));

        // Benchmark the overhead of creating result matrices
        group.bench_function(BenchmarkId::new("matrix_creation", size), |b| {
            b.iter(|| Array2::<f64>::zeros((size, size)))
        });

        // Benchmark with computation
        let matrix = create_test_matrix(size);
        group.bench_with_input(
            BenchmarkId::new("det_with_allocation", size),
            &matrix,
            |b, m| {
                b.iter(|| {
                    let _temp = m.clone(); // Simulate allocation
                    compat::det(&m.view(), false, true).unwrap()
                })
            },
        );
    }

    group.finish();
}

/// Scalability benchmark across different sizes
fn bench_scalability(c: &mut Criterion) {
    let mut group = c.benchmark_group("scalability");
    group.sample_size(15);

    let sizes = [10, 20, 50, 100, 150];

    for &size in &sizes {
        let matrix = create_test_matrix(size);

        group.throughput(Throughput::Elements(size as u64 * size as u64));

        // Test determinant scaling
        group.bench_with_input(BenchmarkId::new("det_scaling", size), &matrix, |b, m| {
            b.iter(|| compat::det(&m.view(), false, true).unwrap())
        });

        // Test norm scaling
        group.bench_with_input(BenchmarkId::new("norm_scaling", size), &matrix, |b, m| {
            b.iter(|| compat::norm(&m.view(), Some("fro"), None, false, true).unwrap())
        });

        // Test LU scaling (up to size 100)
        if size <= 100 {
            group.bench_with_input(BenchmarkId::new("lu_scaling", size), &matrix, |b, m| {
                b.iter(|| compat::lu(&m.view(), false, false, true, false).unwrap())
            });
        }
    }

    group.finish();
}

// Group all benchmarks
criterion_group!(
    benches,
    bench_basic_operations,
    bench_matrix_norms,
    bench_vector_norms,
    bench_decompositions,
    bench_eigenvalues,
    bench_linear_solvers,
    bench_matrix_functions,
    bench_matrix_properties,
    bench_advanced_decompositions,
    bench_utility_functions,
    bench_memory_allocation,
    bench_scalability
);

criterion_main!(benches);
