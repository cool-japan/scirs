//! Comprehensive benchmarks for structured matrix operations
//!
//! This benchmark suite covers operations on structured matrices including
//! Toeplitz, Circulant, Hankel, banded, tridiagonal, and block matrices,
//! leveraging their special structure for efficient algorithms.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ndarray::{Array1, Array2};
use scirs2_linalg::prelude::*;

/// Create a test vector
fn create_test_vector(n: usize) -> Array1<f64> {
    Array1::from_shape_fn(n, |i| ((i + 1) as f64 * 0.1).sin())
}

/// Create a general test matrix for comparison
fn create_general_matrix(n: usize) -> Array2<f64> {
    Array2::from_shape_fn((n, n), |(i, j)| ((i + j + 1) as f64 * 0.1).sin())
}

/// Create block matrices for testing
fn create_block_matrices(block_size: usize, num_blocks: usize) -> Vec<Array2<f64>> {
    (0..num_blocks)
        .map(|k| {
            Array2::from_shape_fn((block_size, block_size), |(i, j)| {
                ((i + j + k + 1) as f64 * 0.1).sin()
            })
        })
        .collect()
}

/// Benchmark Toeplitz matrix operations
fn bench_toeplitz_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("toeplitz_operations");
    group.sample_size(20);

    for &size in &[100, 500, 1000, 2000] {
        let first_row = create_test_vector(size);
        let first_col = create_test_vector(size);
        let vector = create_test_vector(size);
        let rhs = create_test_vector(size);

        group.throughput(Throughput::Elements(size as u64 * size as u64));

        // Toeplitz matrix creation
        group.bench_with_input(
            BenchmarkId::new("toeplitz_create", size),
            &(&first_row, &first_col),
            |b, (r, c)| b.iter(|| ToeplitzMatrix::new(black_box(r.view()), black_box(c.view()))),
        );

        // Toeplitz matrix-vector multiplication
        group.bench_with_input(
            BenchmarkId::new("toeplitz_matvec", size),
            &(&first_row, &first_col, &vector),
            |b, (r, c, v)| {
                b.iter(|| {
                    let toeplitz = ToeplitzMatrix::new(black_box(r.view()), black_box(c.view()));
                    toeplitz_matvec(&toeplitz, black_box(&v.view()))
                })
            },
        );

        // Toeplitz matrix-vector multiplication (FFT-based)
        group.bench_with_input(
            BenchmarkId::new("toeplitz_matvec_fft", size),
            &(&first_row, &first_col, &vector),
            |b, (r, c, v)| {
                b.iter(|| {
                    let toeplitz = ToeplitzMatrix::new(black_box(r.view()), black_box(c.view()));
                    toeplitz_matvec_fft(&toeplitz, black_box(&v.view()))
                })
            },
        );

        // Toeplitz linear system solver
        group.bench_with_input(
            BenchmarkId::new("toeplitz_solve", size),
            &(&first_row, &rhs),
            |b, (r, rhs)| {
                b.iter(|| {
                    let toeplitz = ToeplitzMatrix::new(black_box(r.view()), black_box(r.view()));
                    solve_toeplitz(&toeplitz, black_box(&rhs.view()))
                })
            },
        );

        // Toeplitz linear system solver (Levinson algorithm)
        group.bench_with_input(
            BenchmarkId::new("toeplitz_solve_levinson", size),
            &(&first_row, &rhs),
            |b, (r, rhs)| {
                b.iter(|| {
                    let toeplitz = ToeplitzMatrix::new(black_box(r.view()), black_box(r.view()));
                    solve_toeplitz_levinson(&toeplitz, black_box(&rhs.view()))
                })
            },
        );

        // Toeplitz determinant
        if size <= 1000 {
            group.bench_with_input(
                BenchmarkId::new("toeplitz_determinant", size),
                &first_row,
                |b, r| {
                    b.iter(|| {
                        let toeplitz =
                            ToeplitzMatrix::new(black_box(r.view()), black_box(r.view()));
                        toeplitz_determinant(&toeplitz)
                    })
                },
            );
        }

        // Toeplitz inverse
        if size <= 500 {
            group.bench_with_input(
                BenchmarkId::new("toeplitz_inverse", size),
                &first_row,
                |b, r| {
                    b.iter(|| {
                        let toeplitz =
                            ToeplitzMatrix::new(black_box(r.view()), black_box(r.view()));
                        toeplitz_inverse(&toeplitz)
                    })
                },
            );
        }
    }

    group.finish();
}

/// Benchmark Circulant matrix operations
fn bench_circulant_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("circulant_operations");
    group.sample_size(25);

    for &size in &[100, 500, 1000, 2000] {
        let first_row = create_test_vector(size);
        let vector = create_test_vector(size);
        let rhs = create_test_vector(size);

        group.throughput(Throughput::Elements(size as u64 * size as u64));

        // Circulant matrix creation
        group.bench_with_input(
            BenchmarkId::new("circulant_create", size),
            &first_row,
            |b, r| b.iter(|| CirculantMatrix::new(black_box(r.clone()))),
        );

        // Circulant matrix-vector multiplication (FFT-based)
        group.bench_with_input(
            BenchmarkId::new("circulant_matvec_fft", size),
            &(&first_row, &vector),
            |b, (r, v)| {
                b.iter(|| {
                    let circulant = CirculantMatrix::new(black_box(r.clone()));
                    circulant_matvec_fft(&circulant, black_box(&v.view()))
                })
            },
        );

        // Circulant matrix-vector multiplication (direct)
        if size <= 1000 {
            group.bench_with_input(
                BenchmarkId::new("circulant_matvec_direct", size),
                &(&first_row, &vector),
                |b, (r, v)| {
                    b.iter(|| {
                        let circulant = CirculantMatrix::new(black_box(r.clone()));
                        circulant_matvec_direct(&circulant, black_box(&v.view()))
                    })
                },
            );
        }

        // Circulant linear system solver (FFT-based)
        group.bench_with_input(
            BenchmarkId::new("circulant_solve_fft", size),
            &(&first_row, &rhs),
            |b, (r, rhs)| {
                b.iter(|| {
                    let circulant = CirculantMatrix::new(black_box(r.clone()));
                    solve_circulant_fft(&circulant, black_box(&rhs.view()))
                })
            },
        );

        // Circulant eigenvalues (FFT-based)
        group.bench_with_input(
            BenchmarkId::new("circulant_eigenvalues", size),
            &first_row,
            |b, r| {
                b.iter(|| {
                    let circulant = CirculantMatrix::new(black_box(r.clone()));
                    circulant_eigenvalues(&circulant)
                })
            },
        );

        // Circulant determinant
        group.bench_with_input(
            BenchmarkId::new("circulant_determinant", size),
            &first_row,
            |b, r| {
                b.iter(|| {
                    let circulant = CirculantMatrix::new(black_box(r.clone()));
                    circulant_determinant(&circulant)
                })
            },
        );

        // Circulant inverse (FFT-based)
        group.bench_with_input(
            BenchmarkId::new("circulant_inverse_fft", size),
            &first_row,
            |b, r| {
                b.iter(|| {
                    let circulant = CirculantMatrix::new(black_box(r.clone()));
                    circulant_inverse_fft(&circulant)
                })
            },
        );
    }

    group.finish();
}

/// Benchmark Hankel matrix operations
fn bench_hankel_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("hankel_operations");
    group.sample_size(20);

    for &size in &[100, 500, 1000] {
        let first_row = create_test_vector(size);
        let last_col = create_test_vector(size);
        let vector = create_test_vector(size);

        group.throughput(Throughput::Elements(size as u64 * size as u64));

        // Hankel matrix creation
        group.bench_with_input(
            BenchmarkId::new("hankel_create", size),
            &(&first_row, &last_col),
            |b, (r, c)| b.iter(|| HankelMatrix::new(black_box(r.clone()), black_box(c.clone()))),
        );

        // Hankel matrix-vector multiplication
        group.bench_with_input(
            BenchmarkId::new("hankel_matvec", size),
            &(&first_row, &last_col, &vector),
            |b, (r, c, v)| {
                b.iter(|| {
                    let hankel = HankelMatrix::new(black_box(r.clone()), black_box(c.clone()));
                    hankel_matvec(&hankel, black_box(&v.view()))
                })
            },
        );

        // Hankel matrix-vector multiplication (FFT-based)
        group.bench_with_input(
            BenchmarkId::new("hankel_matvec_fft", size),
            &(&first_row, &last_col, &vector),
            |b, (r, c, v)| {
                b.iter(|| {
                    let hankel = HankelMatrix::new(black_box(r.clone()), black_box(c.clone()));
                    hankel_matvec_fft(&hankel, black_box(&v.view()))
                })
            },
        );

        // Hankel determinant
        if size <= 500 {
            group.bench_with_input(
                BenchmarkId::new("hankel_determinant", size),
                &(&first_row, &last_col),
                |b, (r, c)| {
                    b.iter(|| {
                        let hankel = HankelMatrix::new(black_box(r.clone()), black_box(c.clone()));
                        hankel_determinant(&hankel)
                    })
                },
            );
        }

        // Hankel SVD (structured approach)
        if size <= 200 {
            group.bench_with_input(
                BenchmarkId::new("hankel_svd", size),
                &(&first_row, &last_col),
                |b, (r, c)| {
                    b.iter(|| {
                        let hankel = HankelMatrix::new(black_box(r.clone()), black_box(c.clone()));
                        hankel_svd(&hankel)
                    })
                },
            );
        }
    }

    group.finish();
}

/// Benchmark tridiagonal matrix operations
fn bench_tridiagonal_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("tridiagonal_operations");
    group.sample_size(30);

    for &size in &[1000, 5000, 10000, 20000] {
        let diagonal = create_test_vector(size);
        let sub_diagonal = create_test_vector(size - 1);
        let super_diagonal = create_test_vector(size - 1);
        let vector = create_test_vector(size);
        let rhs = create_test_vector(size);

        group.throughput(Throughput::Elements(size as u64));

        // Tridiagonal matrix creation
        group.bench_with_input(
            BenchmarkId::new("tridiagonal_create", size),
            &(&sub_diagonal, &diagonal, &super_diagonal),
            |b, (sub, diag, sup)| {
                b.iter(|| {
                    TridiagonalMatrix::new(
                        black_box(sub.clone()),
                        black_box(diag.clone()),
                        black_box(sup.clone()),
                    )
                })
            },
        );

        // Tridiagonal matrix-vector multiplication
        group.bench_with_input(
            BenchmarkId::new("tridiagonal_matvec", size),
            &(&sub_diagonal, &diagonal, &super_diagonal, &vector),
            |b, (sub, diag, sup, v)| {
                b.iter(|| {
                    let tridiag = TridiagonalMatrix::new(
                        black_box(sub.clone()),
                        black_box(diag.clone()),
                        black_box(sup.clone()),
                    );
                    tridiagonal_matvec(&tridiag, black_box(&v.view()))
                })
            },
        );

        // Tridiagonal linear system solver (Thomas algorithm)
        group.bench_with_input(
            BenchmarkId::new("tridiagonal_solve_thomas", size),
            &(&sub_diagonal, &diagonal, &super_diagonal, &rhs),
            |b, (sub, diag, sup, rhs)| {
                b.iter(|| {
                    let tridiag = TridiagonalMatrix::new(
                        black_box(sub.clone()),
                        black_box(diag.clone()),
                        black_box(sup.clone()),
                    );
                    solve_tridiagonal_thomas(&tridiag, black_box(&rhs.view()))
                })
            },
        );

        // Tridiagonal linear system solver (LU decomposition)
        group.bench_with_input(
            BenchmarkId::new("tridiagonal_solve_lu", size),
            &(&sub_diagonal, &diagonal, &super_diagonal, &rhs),
            |b, (sub, diag, sup, rhs)| {
                b.iter(|| {
                    let tridiag = TridiagonalMatrix::new(
                        black_box(sub.clone()),
                        black_box(diag.clone()),
                        black_box(sup.clone()),
                    );
                    solve_tridiagonal_lu(&tridiag, black_box(&rhs.view()))
                })
            },
        );

        // Tridiagonal determinant
        group.bench_with_input(
            BenchmarkId::new("tridiagonal_determinant", size),
            &(&sub_diagonal, &diagonal, &super_diagonal),
            |b, (sub, diag, sup)| {
                b.iter(|| {
                    let tridiag = TridiagonalMatrix::new(
                        black_box(sub.clone()),
                        black_box(diag.clone()),
                        black_box(sup.clone()),
                    );
                    tridiagonal_determinant(&tridiag)
                })
            },
        );

        // Tridiagonal eigenvalues (Sturm sequence)
        if size <= 5000 {
            group.bench_with_input(
                BenchmarkId::new("tridiagonal_eigenvalues", size),
                &(&sub_diagonal, &diagonal, &super_diagonal),
                |b, (sub, diag, sup)| {
                    b.iter(|| {
                        let tridiag = TridiagonalMatrix::new(
                            black_box(sub.clone()),
                            black_box(diag.clone()),
                            black_box(sup.clone()),
                        );
                        tridiagonal_eigenvalues(&tridiag)
                    })
                },
            );
        }

        // Tridiagonal eigenvectors (QR algorithm)
        if size <= 1000 {
            group.bench_with_input(
                BenchmarkId::new("tridiagonal_eigenvectors", size),
                &(&sub_diagonal, &diagonal, &super_diagonal),
                |b, (sub, diag, sup)| {
                    b.iter(|| {
                        let tridiag = TridiagonalMatrix::new(
                            black_box(sub.clone()),
                            black_box(diag.clone()),
                            black_box(sup.clone()),
                        );
                        tridiagonal_eigenvectors(&tridiag)
                    })
                },
            );
        }
    }

    group.finish();
}

/// Benchmark banded matrix operations
fn bench_banded_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("banded_operations");
    group.sample_size(25);

    for &size in &[500, 1000, 2000] {
        for &bandwidth in &[5, 10, 20, 50] {
            let matrix_data = Array2::from_shape_fn((2 * bandwidth + 1, size), |(i, j)| {
                ((i + j + 1) as f64 * 0.1).sin()
            });
            let vector = create_test_vector(size);
            let rhs = create_test_vector(size);

            group.throughput(Throughput::Elements(size as u64 * bandwidth as u64));

            // Banded matrix creation
            group.bench_with_input(
                BenchmarkId::new(format!("banded_create_bw_{}", bandwidth), size),
                &(&matrix_data, bandwidth),
                |b, (data, bw)| {
                    b.iter(|| BandedMatrix::new(black_box(data.clone()), black_box(*bw)))
                },
            );

            // Banded matrix-vector multiplication
            group.bench_with_input(
                BenchmarkId::new(format!("banded_matvec_bw_{}", bandwidth), size),
                &(&matrix_data, bandwidth, &vector),
                |b, (data, bw, v)| {
                    b.iter(|| {
                        let banded = BandedMatrix::new(black_box(data.clone()), black_box(*bw));
                        banded_matvec(&banded, black_box(&v.view()))
                    })
                },
            );

            // Banded linear system solver (LU decomposition)
            group.bench_with_input(
                BenchmarkId::new(format!("banded_solve_lu_bw_{}", bandwidth), size),
                &(&matrix_data, bandwidth, &rhs),
                |b, (data, bw, rhs)| {
                    b.iter(|| {
                        let banded = BandedMatrix::new(black_box(data.clone()), black_box(*bw));
                        solve_banded_lu(&banded, black_box(&rhs.view()))
                    })
                },
            );

            // Banded Cholesky decomposition (for SPD banded matrices)
            if bandwidth <= 20 {
                group.bench_with_input(
                    BenchmarkId::new(format!("banded_cholesky_bw_{}", bandwidth), size),
                    &(&matrix_data, bandwidth),
                    |b, (data, bw)| {
                        b.iter(|| {
                            let banded = BandedMatrix::new(black_box(data.clone()), black_box(*bw));
                            banded_cholesky(&banded)
                        })
                    },
                );
            }
        }
    }

    group.finish();
}

/// Benchmark block diagonal matrix operations
fn bench_block_diagonal_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("block_diagonal_operations");
    group.sample_size(20);

    for &block_size in &[10, 20, 50] {
        for &num_blocks in &[5, 10, 20, 50] {
            let blocks = create_block_matrices(block_size, num_blocks);
            let vector = create_test_vector(block_size * num_blocks);
            let rhs = create_test_vector(block_size * num_blocks);

            let total_size = block_size * num_blocks;
            group.throughput(Throughput::Elements(total_size as u64 * total_size as u64));

            // Block diagonal matrix creation
            group.bench_with_input(
                BenchmarkId::new(
                    format!("block_diag_create_{}x{}", num_blocks, block_size),
                    total_size,
                ),
                &blocks,
                |b, blocks| {
                    b.iter(|| {
                        let block_views: Vec<_> = blocks.iter().map(|b| b.view()).collect();
                        BlockDiagonalMatrix::new(black_box(block_views))
                    })
                },
            );

            // Block diagonal matrix-vector multiplication
            group.bench_with_input(
                BenchmarkId::new(
                    format!("block_diag_matvec_{}x{}", num_blocks, block_size),
                    total_size,
                ),
                &(&blocks, &vector),
                |b, (blocks, v)| {
                    b.iter(|| {
                        let block_views: Vec<_> = blocks.iter().map(|b| b.view()).collect();
                        let block_diag = BlockDiagonalMatrix::new(black_box(block_views));
                        block_diagonal_matvec(&block_diag, black_box(&v.view()))
                    })
                },
            );

            // Block diagonal linear system solver
            group.bench_with_input(
                BenchmarkId::new(
                    format!("block_diag_solve_{}x{}", num_blocks, block_size),
                    total_size,
                ),
                &(&blocks, &rhs),
                |b, (blocks, rhs)| {
                    b.iter(|| {
                        let block_views: Vec<_> = blocks.iter().map(|b| b.view()).collect();
                        let block_diag = BlockDiagonalMatrix::new(black_box(block_views));
                        solve_block_diagonal(&block_diag, black_box(&rhs.view()))
                    })
                },
            );

            // Block diagonal determinant
            if block_size <= 20 && num_blocks <= 20 {
                group.bench_with_input(
                    BenchmarkId::new(
                        format!("block_diag_determinant_{}x{}", num_blocks, block_size),
                        total_size,
                    ),
                    &blocks,
                    |b, blocks| {
                        b.iter(|| {
                            let block_views: Vec<_> = blocks.iter().map(|b| b.view()).collect();
                            let block_diag = BlockDiagonalMatrix::new(black_box(block_views));
                            block_diagonal_determinant(&block_diag)
                        })
                    },
                );
            }

            // Block diagonal inverse
            if block_size <= 20 && num_blocks <= 10 {
                group.bench_with_input(
                    BenchmarkId::new(
                        format!("block_diag_inverse_{}x{}", num_blocks, block_size),
                        total_size,
                    ),
                    &blocks,
                    |b, blocks| {
                        b.iter(|| {
                            let block_views: Vec<_> = blocks.iter().map(|b| b.view()).collect();
                            let block_diag = BlockDiagonalMatrix::new(black_box(block_views));
                            block_diagonal_inverse(&block_diag)
                        })
                    },
                );
            }
        }
    }

    group.finish();
}

/// Benchmark block tridiagonal matrix operations
fn bench_block_tridiagonal_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("block_tridiagonal_operations");
    group.sample_size(20);

    for &block_size in &[10, 20, 50] {
        for &num_blocks in &[5, 10, 20] {
            let diagonal_blocks = create_block_matrices(block_size, num_blocks);
            let off_diagonal_blocks = create_block_matrices(block_size, num_blocks - 1);
            let vector = create_test_vector(block_size * num_blocks);
            let rhs = create_test_vector(block_size * num_blocks);

            let total_size = block_size * num_blocks;
            group.throughput(Throughput::Elements(total_size as u64 * total_size as u64));

            // Block tridiagonal matrix creation
            group.bench_with_input(
                BenchmarkId::new(
                    format!("block_tridiag_create_{}x{}", num_blocks, block_size),
                    total_size,
                ),
                &(&diagonal_blocks, &off_diagonal_blocks),
                |b, (diag, off_diag)| {
                    b.iter(|| {
                        let diag_views: Vec<_> = diag.iter().map(|b| b.view()).collect();
                        let off_diag_views: Vec<_> = off_diag.iter().map(|b| b.view()).collect();
                        BlockTridiagonalMatrix::new(
                            black_box(off_diag_views.clone()),
                            black_box(diag_views),
                            black_box(off_diag_views),
                        )
                    })
                },
            );

            // Block tridiagonal matrix-vector multiplication
            group.bench_with_input(
                BenchmarkId::new(
                    format!("block_tridiag_matvec_{}x{}", num_blocks, block_size),
                    total_size,
                ),
                &(&diagonal_blocks, &off_diagonal_blocks, &vector),
                |b, (diag, off_diag, v)| {
                    b.iter(|| {
                        let diag_views: Vec<_> = diag.iter().map(|b| b.view()).collect();
                        let off_diag_views: Vec<_> = off_diag.iter().map(|b| b.view()).collect();
                        let block_tridiag = BlockTridiagonalMatrix::new(
                            black_box(off_diag_views.clone()),
                            black_box(diag_views),
                            black_box(off_diag_views),
                        );
                        block_tridiagonal_matvec(&block_tridiag, black_box(&v.view()))
                    })
                },
            );

            // Block tridiagonal linear system solver
            group.bench_with_input(
                BenchmarkId::new(
                    format!("block_tridiag_solve_{}x{}", num_blocks, block_size),
                    total_size,
                ),
                &(&diagonal_blocks, &off_diagonal_blocks, &rhs),
                |b, (diag, off_diag, rhs)| {
                    b.iter(|| {
                        let diag_views: Vec<_> = diag.iter().map(|b| b.view()).collect();
                        let off_diag_views: Vec<_> = off_diag.iter().map(|b| b.view()).collect();
                        let block_tridiag = BlockTridiagonalMatrix::new(
                            black_box(off_diag_views.clone()),
                            black_box(diag_views),
                            black_box(off_diag_views),
                        );
                        solve_block_tridiagonal(&block_tridiag, black_box(&rhs.view()))
                    })
                },
            );

            // Block tridiagonal LU decomposition
            if block_size <= 20 && num_blocks <= 10 {
                group.bench_with_input(
                    BenchmarkId::new(
                        format!("block_tridiag_lu_{}x{}", num_blocks, block_size),
                        total_size,
                    ),
                    &(&diagonal_blocks, &off_diagonal_blocks),
                    |b, (diag, off_diag)| {
                        b.iter(|| {
                            let diag_views: Vec<_> = diag.iter().map(|b| b.view()).collect();
                            let off_diag_views: Vec<_> =
                                off_diag.iter().map(|b| b.view()).collect();
                            let block_tridiag = BlockTridiagonalMatrix::new(
                                black_box(off_diag_views.clone()),
                                black_box(diag_views),
                                black_box(off_diag_views),
                            );
                            block_tridiagonal_lu(&block_tridiag)
                        })
                    },
                );
            }
        }
    }

    group.finish();
}

/// Benchmark structured vs general matrix operations comparison
fn bench_structured_vs_general_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("structured_vs_general_comparison");
    group.sample_size(20);

    for &size in &[500, 1000, 2000] {
        let first_row = create_test_vector(size);
        let vector = create_test_vector(size);
        let general_matrix = create_general_matrix(size);

        group.throughput(Throughput::Elements(size as u64 * size as u64));

        // Toeplitz vs general matrix-vector multiplication
        group.bench_with_input(
            BenchmarkId::new("toeplitz_vs_general_matvec_toeplitz", size),
            &(&first_row, &vector),
            |b, (r, v)| {
                b.iter(|| {
                    let toeplitz = ToeplitzMatrix::new(black_box(r.view()), black_box(r.view()));
                    toeplitz_matvec(&toeplitz, black_box(&v.view()))
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("toeplitz_vs_general_matvec_general", size),
            &(&general_matrix, &vector),
            |b, (m, v)| b.iter(|| m.dot(black_box(&v.view()))),
        );

        // Circulant vs general matrix-vector multiplication
        group.bench_with_input(
            BenchmarkId::new("circulant_vs_general_matvec_circulant", size),
            &(&first_row, &vector),
            |b, (r, v)| {
                b.iter(|| {
                    let circulant = CirculantMatrix::new(black_box(r.clone()));
                    circulant_matvec_fft(&circulant, black_box(&v.view()))
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("circulant_vs_general_matvec_general", size),
            &(&general_matrix, &vector),
            |b, (m, v)| b.iter(|| m.dot(black_box(&v.view()))),
        );

        // Memory usage comparison
        group.bench_with_input(
            BenchmarkId::new("memory_usage_toeplitz", size),
            &first_row,
            |b, r| {
                b.iter(|| {
                    let _toeplitz = ToeplitzMatrix::new(black_box(r.view()), black_box(r.view()));
                    // Just creation to measure memory allocation
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("memory_usage_general", size),
            &size,
            |b, &s| {
                b.iter(|| {
                    let _matrix = Array2::<f64>::zeros((black_box(s), black_box(s)));
                    // Just creation to measure memory allocation
                })
            },
        );
    }

    group.finish();
}

/// Benchmark specialized algorithms for structured matrices
fn bench_specialized_algorithms(c: &mut Criterion) {
    let mut group = c.benchmark_group("specialized_algorithms");
    group.sample_size(15);

    for &size in &[100, 500, 1000] {
        let first_row = create_test_vector(size);
        let autocorr_data = create_test_vector(size);

        group.throughput(Throughput::Elements(size as u64 * size as u64));

        // Levinson-Durbin algorithm for Toeplitz systems
        group.bench_with_input(
            BenchmarkId::new("levinson_durbin", size),
            &autocorr_data,
            |b, data| b.iter(|| levinson_durbin(black_box(&data.view()))),
        );

        // Yule-Walker equations solving
        group.bench_with_input(
            BenchmarkId::new("yule_walker", size),
            &autocorr_data,
            |b, data| b.iter(|| yule_walker(black_box(&data.view()))),
        );

        // Fast Toeplitz matrix inversion
        // TODO: Implement fast_toeplitz_inverse function
        // if size <= 500 {
        //     group.bench_with_input(
        //         BenchmarkId::new("fast_toeplitz_inverse", size),
        //         &first_row,
        //         |b, r| {
        //             b.iter(|| {
        //                 let toeplitz =
        //                     ToeplitzMatrix::new(black_box(r.view()), black_box(r.view()));
        //                 fast_toeplitz_inverse(&toeplitz)
        //             })
        //         },
        //     );
        // }

        // Gohberg-Semencul formula for Toeplitz inverse
        // TODO: Implement gohberg_semencul_inverse function
        // if size <= 500 {
        //     group.bench_with_input(
        //         BenchmarkId::new("gohberg_semencul_inverse", size),
        //         &first_row,
        //         |b, r| {
        //             b.iter(|| {
        //                 let toeplitz =
        //                     ToeplitzMatrix::new(black_box(r.view()), black_box(r.view()));
        //                 gohberg_semencul_inverse(&toeplitz)
        //             })
        //         },
        //     );
        // }

        // Discrete Fourier Transform matrix operations
        // TODO: Implement dft_matrix_multiply function
        // group.bench_with_input(
        //     BenchmarkId::new("dft_matrix_multiply", size),
        //     &first_row,
        //     |b, data| b.iter(|| dft_matrix_multiply(black_box(&data.view()))),
        // );

        // Hadamard matrix operations (sizes must be powers of 2)
        // TODO: Implement hadamard_transform function
        // let hadamard_size = (size as f64).log2().floor() as usize;
        // let hadamard_size = 2_usize.pow(hadamard_size as u32);
        // if hadamard_size >= 4 {
        //     let hadamard_vector = create_test_vector(hadamard_size);
        //     group.bench_with_input(
        //         BenchmarkId::new(
        //             format!("hadamard_transform_{}", hadamard_size),
        //             hadamard_size,
        //         ),
        //         &hadamard_vector,
        //         |b, v| b.iter(|| hadamard_transform(black_box(&v.view()))),
        //     );
        // }
    }

    group.finish();
}

// Group all benchmarks
criterion_group!(
    benches,
    bench_toeplitz_operations,
    bench_circulant_operations,
    bench_hankel_operations,
    bench_tridiagonal_operations,
    bench_banded_operations,
    bench_block_diagonal_operations,
    bench_block_tridiagonal_operations,
    bench_structured_vs_general_comparison,
    bench_specialized_algorithms
);

criterion_main!(benches);
