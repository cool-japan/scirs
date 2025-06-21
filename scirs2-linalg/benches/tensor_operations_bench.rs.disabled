//! Comprehensive benchmarks for tensor operations
//!
//! This benchmark suite covers tensor contraction, decompositions, Einstein summation,
//! and higher-order tensor operations used in machine learning and scientific computing.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ndarray::{Array1, Array2, Array3, Array4, Axis};
use scirs2_linalg::prelude::*;
use std::time::Duration;

/// Create a test tensor of given dimensions with deterministic values
fn create_test_tensor_3d(d1: usize, d2: usize, d3: usize) -> Array3<f64> {
    Array3::from_shape_fn((d1, d2, d3), |(i, j, k)| {
        ((i + j + k + 1) as f64 * 0.1).sin()
    })
}

/// Create a test tensor of given dimensions with deterministic values
fn create_test_tensor_4d(d1: usize, d2: usize, d3: usize, d4: usize) -> Array4<f64> {
    Array4::from_shape_fn((d1, d2, d3, d4), |(i, j, k, l)| {
        ((i + j + k + l + 1) as f64 * 0.1).sin()
    })
}

/// Create a random-like matrix for tensor operations
fn create_test_matrix(m: usize, n: usize) -> Array2<f64> {
    Array2::from_shape_fn((m, n), |(i, j)| ((i + j + 1) as f64 * 0.1).sin())
}

/// Create a test vector
fn create_test_vector(n: usize) -> Array1<f64> {
    Array1::from_shape_fn(n, |i| ((i + 1) as f64 * 0.1).sin())
}

/// Benchmark basic tensor contraction operations
fn bench_tensor_contraction(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_contraction");
    group.sample_size(20);

    for &size in &[10, 20, 30, 50] {
        let tensor_3d = create_test_tensor_3d(size, size, size);
        let tensor_4d = create_test_tensor_4d(size, size, size, size);
        let matrix = create_test_matrix(size, size);
        let vector = create_test_vector(size);

        let total_elements = size * size * size;
        group.throughput(Throughput::Elements(total_elements as u64));

        // Mode-n product (tensor-matrix multiplication)
        group.bench_with_input(
            BenchmarkId::new("mode_n_product_1", size),
            &(&tensor_3d, &matrix),
            |b, (t, m)| {
                b.iter(|| mode_n_product(black_box(&t.view()), black_box(&m.view()), 0).unwrap())
            },
        );

        group.bench_with_input(
            BenchmarkId::new("mode_n_product_2", size),
            &(&tensor_3d, &matrix),
            |b, (t, m)| {
                b.iter(|| mode_n_product(black_box(&t.view()), black_box(&m.view()), 1).unwrap())
            },
        );

        group.bench_with_input(
            BenchmarkId::new("mode_n_product_3", size),
            &(&tensor_3d, &matrix),
            |b, (t, m)| {
                b.iter(|| mode_n_product(black_box(&t.view()), black_box(&m.view()), 2).unwrap())
            },
        );

        // Tensor-vector contraction
        group.bench_with_input(
            BenchmarkId::new("tensor_vector_contract", size),
            &(&tensor_3d, &vector),
            |b, (t, v)| {
                b.iter(|| {
                    tensor_vector_contract(black_box(&t.view()), black_box(&v.view()), 0).unwrap()
                })
            },
        );

        // Tensor-tensor contraction (4D to 2D)
        if size <= 30 {
            // Limit size for 4D tensors
            group.bench_with_input(
                BenchmarkId::new("tensor_tensor_contract", size),
                &tensor_4d,
                |b, t| {
                    b.iter(|| tensor_contract_axes(black_box(&t.view()), &[0, 2], &[1, 3]).unwrap())
                },
            );
        }

        // Tensor inner product
        group.bench_with_input(
            BenchmarkId::new("tensor_inner_product", size),
            &(&tensor_3d, &tensor_3d),
            |b, (t1, t2)| {
                b.iter(|| {
                    tensor_inner_product(black_box(&t1.view()), black_box(&t2.view())).unwrap()
                })
            },
        );

        // Tensor outer product
        if size <= 20 {
            // Smaller sizes for outer products
            group.bench_with_input(
                BenchmarkId::new("tensor_outer_product", size),
                &(&tensor_3d, &tensor_3d),
                |b, (t1, t2)| {
                    b.iter(|| {
                        tensor_outer_product(black_box(&t1.view()), black_box(&t2.view())).unwrap()
                    })
                },
            );
        }
    }

    group.finish();
}

/// Benchmark Einstein summation operations
fn bench_einsum_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("einsum_operations");
    group.sample_size(20);

    for &size in &[10, 20, 30, 50] {
        let matrix_a = create_test_matrix(size, size);
        let matrix_b = create_test_matrix(size, size);
        let vector = create_test_vector(size);
        let tensor_3d = create_test_tensor_3d(size, size, size);

        group.throughput(Throughput::Elements(size as u64 * size as u64));

        // Matrix multiplication: ij,jk->ik
        group.bench_with_input(
            BenchmarkId::new("einsum_matmul", size),
            &(&matrix_a, &matrix_b),
            |b, (a, matrix_b)| {
                b.iter(|| {
                    einsum(
                        "ij,jk->ik",
                        &[black_box(&a.view()), black_box(&matrix_b.view())],
                    )
                    .unwrap()
                })
            },
        );

        // Matrix-vector multiplication: ij,j->i
        group.bench_with_input(
            BenchmarkId::new("einsum_matvec", size),
            &(&matrix_a, &vector),
            |b, (m, v)| {
                b.iter(|| einsum("ij,j->i", &[black_box(&m.view()), black_box(&v.view())]).unwrap())
            },
        );

        // Element-wise product: ij,ij->ij
        group.bench_with_input(
            BenchmarkId::new("einsum_hadamard", size),
            &(&matrix_a, &matrix_b),
            |b, (a, matrix_b)| {
                b.iter(|| {
                    einsum(
                        "ij,ij->ij",
                        &[black_box(&a.view()), black_box(&matrix_b.view())],
                    )
                    .unwrap()
                })
            },
        );

        // Trace: ii->
        group.bench_with_input(BenchmarkId::new("einsum_trace", size), &matrix_a, |b, m| {
            b.iter(|| einsum("ii->", &[black_box(&m.view())]).unwrap())
        });

        // Transpose: ij->ji
        group.bench_with_input(
            BenchmarkId::new("einsum_transpose", size),
            &matrix_a,
            |b, m| b.iter(|| einsum("ij->ji", &[black_box(&m.view())]).unwrap()),
        );

        // Tensor contraction: ijk,jkl->il
        if size <= 30 {
            let tensor_b = create_test_tensor_3d(size, size, size);
            group.bench_with_input(
                BenchmarkId::new("einsum_tensor_contract", size),
                &(&tensor_3d, &tensor_b),
                |b, (t1, t2)| {
                    b.iter(|| {
                        einsum(
                            "ijk,jkl->il",
                            &[black_box(&t1.view()), black_box(&t2.view())],
                        )
                        .unwrap()
                    })
                },
            );
        }

        // Batch matrix multiplication: ijk,ikl->ijl
        if size <= 30 {
            group.bench_with_input(
                BenchmarkId::new("einsum_batch_matmul", size),
                &(&tensor_3d, &tensor_3d),
                |b, (t1, t2)| {
                    b.iter(|| {
                        einsum(
                            "ijk,ikl->ijl",
                            &[black_box(&t1.view()), black_box(&t2.view())],
                        )
                        .unwrap()
                    })
                },
            );
        }

        // Bilinear form: i,ij,j->
        group.bench_with_input(
            BenchmarkId::new("einsum_bilinear", size),
            &(&vector, &matrix_a, &vector),
            |b, (v1, m, v2)| {
                b.iter(|| {
                    einsum(
                        "i,ij,j->",
                        &[
                            black_box(&v1.view()),
                            black_box(&m.view()),
                            black_box(&v2.view()),
                        ],
                    )
                    .unwrap()
                })
            },
        );
    }

    group.finish();
}

/// Benchmark Higher-Order SVD (HOSVD) operations
fn bench_hosvd_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("hosvd_operations");
    group.sample_size(10); // HOSVD is expensive
    group.measurement_time(Duration::from_secs(45));

    for &size in &[10, 20, 30] {
        let tensor_3d = create_test_tensor_3d(size, size, size);
        let tensor_4d = create_test_tensor_4d(size, size, size, size);

        group.throughput(Throughput::Elements(
            size as u64 * size as u64 * size as u64,
        ));

        // Higher-Order SVD (3D tensor)
        group.bench_with_input(BenchmarkId::new("hosvd_3d", size), &tensor_3d, |b, t| {
            b.iter(|| hosvd(black_box(&t.view())).unwrap())
        });

        // Higher-Order SVD with truncation
        let truncation_ranks = vec![size / 2, size / 2, size / 2];
        group.bench_with_input(
            BenchmarkId::new("hosvd_3d_truncated", size),
            &(&tensor_3d, &truncation_ranks),
            |b, (t, ranks)| {
                b.iter(|| hosvd_truncated(black_box(&t.view()), black_box(ranks)).unwrap())
            },
        );

        // Sequentially Truncated Higher-Order SVD (ST-HOSVD)
        group.bench_with_input(
            BenchmarkId::new("st_hosvd_3d", size),
            &(&tensor_3d, &truncation_ranks),
            |b, (t, ranks)| b.iter(|| st_hosvd(black_box(&t.view()), black_box(ranks)).unwrap()),
        );

        // 4D HOSVD (smaller sizes only)
        if size <= 20 {
            group.bench_with_input(BenchmarkId::new("hosvd_4d", size), &tensor_4d, |b, t| {
                b.iter(|| hosvd(black_box(&t.view())).unwrap())
            });
        }

        // HOSVD reconstruction
        group.bench_with_input(
            BenchmarkId::new("hosvd_reconstruct", size),
            &tensor_3d,
            |b, t| {
                b.iter(|| {
                    let decomp = hosvd(black_box(&t.view())).unwrap();
                    hosvd_reconstruct(&decomp).unwrap()
                })
            },
        );
    }

    group.finish();
}

/// Benchmark Tucker decomposition operations
fn bench_tucker_decomposition(c: &mut Criterion) {
    let mut group = c.benchmark_group("tucker_decomposition");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(40));

    for &size in &[10, 20, 30] {
        let tensor_3d = create_test_tensor_3d(size, size, size);

        group.throughput(Throughput::Elements(
            size as u64 * size as u64 * size as u64,
        ));

        // Tucker decomposition (ALS algorithm)
        let tucker_ranks = vec![size / 2, size / 2, size / 2];
        group.bench_with_input(
            BenchmarkId::new("tucker_als", size),
            &(&tensor_3d, &tucker_ranks),
            |b, (t, ranks)| {
                b.iter(|| tucker_als(black_box(&t.view()), black_box(ranks), 50, 1e-8).unwrap())
            },
        );

        // Tucker decomposition (HOOI algorithm)
        group.bench_with_input(
            BenchmarkId::new("tucker_hooi", size),
            &(&tensor_3d, &tucker_ranks),
            |b, (t, ranks)| {
                b.iter(|| tucker_hooi(black_box(&t.view()), black_box(ranks), 50, 1e-8).unwrap())
            },
        );

        // Tucker decomposition (initialization via HOSVD)
        group.bench_with_input(
            BenchmarkId::new("tucker_hosvd_init", size),
            &(&tensor_3d, &tucker_ranks),
            |b, (t, ranks)| {
                b.iter(|| {
                    tucker_with_hosvd_init(black_box(&t.view()), black_box(ranks), 50, 1e-8)
                        .unwrap()
                })
            },
        );

        // Tucker decomposition reconstruction
        group.bench_with_input(
            BenchmarkId::new("tucker_reconstruct", size),
            &(&tensor_3d, &tucker_ranks),
            |b, (t, ranks)| {
                b.iter(|| {
                    let decomp =
                        tucker_als(black_box(&t.view()), black_box(ranks), 20, 1e-6).unwrap();
                    tucker_reconstruct(&decomp).unwrap()
                })
            },
        );

        // Tucker tensor-times-matrix (TTM)
        let matrix = create_test_matrix(size / 2, size);
        group.bench_with_input(
            BenchmarkId::new("tucker_ttm", size),
            &(&tensor_3d, &tucker_ranks, &matrix),
            |b, (t, ranks, m)| {
                b.iter(|| {
                    let decomp =
                        tucker_als(black_box(&t.view()), black_box(ranks), 10, 1e-6).unwrap();
                    tucker_ttm(&decomp, black_box(&m.view()), 0).unwrap()
                })
            },
        );
    }

    group.finish();
}

/// Benchmark Tensor Train (TT) decomposition operations
fn bench_tensor_train_decomposition(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_train_decomposition");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(40));

    for &size in &[10, 20, 30] {
        let tensor_3d = create_test_tensor_3d(size, size, size);
        let tensor_4d = create_test_tensor_4d(size, size, size, size);

        group.throughput(Throughput::Elements(
            size as u64 * size as u64 * size as u64,
        ));

        // Tensor Train decomposition via SVD
        let tt_ranks = vec![1, size / 2, size / 2, 1];
        group.bench_with_input(
            BenchmarkId::new("tt_svd_3d", size),
            &(&tensor_3d, &tt_ranks),
            |b, (t, ranks)| b.iter(|| tt_svd(black_box(&t.view()), black_box(ranks)).unwrap()),
        );

        // Tensor Train decomposition via ALS
        group.bench_with_input(
            BenchmarkId::new("tt_als_3d", size),
            &(&tensor_3d, &tt_ranks),
            |b, (t, ranks)| {
                b.iter(|| tt_als(black_box(&t.view()), black_box(ranks), 50, 1e-8).unwrap())
            },
        );

        // Tensor Train rounding
        group.bench_with_input(
            BenchmarkId::new("tt_round", size),
            &(&tensor_3d, &tt_ranks),
            |b, (t, ranks)| {
                b.iter(|| {
                    let tt_decomp = tt_svd(black_box(&t.view()), black_box(ranks)).unwrap();
                    tt_round(&tt_decomp, 1e-8).unwrap()
                })
            },
        );

        // 4D Tensor Train (smaller sizes)
        if size <= 20 {
            let tt_ranks_4d = vec![1, size / 3, size / 3, size / 3, 1];
            group.bench_with_input(
                BenchmarkId::new("tt_svd_4d", size),
                &(&tensor_4d, &tt_ranks_4d),
                |b, (t, ranks)| b.iter(|| tt_svd(black_box(&t.view()), black_box(ranks)).unwrap()),
            );
        }

        // Tensor Train reconstruction
        group.bench_with_input(
            BenchmarkId::new("tt_reconstruct", size),
            &(&tensor_3d, &tt_ranks),
            |b, (t, ranks)| {
                b.iter(|| {
                    let tt_decomp = tt_svd(black_box(&t.view()), black_box(ranks)).unwrap();
                    tt_reconstruct(&tt_decomp).unwrap()
                })
            },
        );

        // Tensor Train addition
        group.bench_with_input(
            BenchmarkId::new("tt_add", size),
            &(&tensor_3d, &tt_ranks),
            |b, (t, ranks)| {
                b.iter(|| {
                    let tt1 = tt_svd(black_box(&t.view()), black_box(ranks)).unwrap();
                    let tt2 = tt_svd(black_box(&t.view()), black_box(ranks)).unwrap();
                    tt_add(&tt1, &tt2).unwrap()
                })
            },
        );
    }

    group.finish();
}

/// Benchmark Canonical Polyadic (CP) decomposition operations
fn bench_cp_decomposition(c: &mut Criterion) {
    let mut group = c.benchmark_group("cp_decomposition");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(40));

    for &size in &[10, 20, 30] {
        let tensor_3d = create_test_tensor_3d(size, size, size);

        group.throughput(Throughput::Elements(
            size as u64 * size as u64 * size as u64,
        ));

        // CP decomposition via ALS
        let cp_rank = size / 2;
        group.bench_with_input(
            BenchmarkId::new("cp_als", size),
            &(&tensor_3d, cp_rank),
            |b, (t, &rank)| b.iter(|| cp_als(black_box(&t.view()), rank, 100, 1e-8).unwrap()),
        );

        // CP decomposition with random initialization
        group.bench_with_input(
            BenchmarkId::new("cp_als_random_init", size),
            &(&tensor_3d, cp_rank),
            |b, (t, &rank)| {
                b.iter(|| cp_als_random_init(black_box(&t.view()), rank, 100, 1e-8, None).unwrap())
            },
        );

        // CP decomposition with HOSVD initialization
        group.bench_with_input(
            BenchmarkId::new("cp_als_hosvd_init", size),
            &(&tensor_3d, cp_rank),
            |b, (t, &rank)| {
                b.iter(|| cp_als_hosvd_init(black_box(&t.view()), rank, 100, 1e-8).unwrap())
            },
        );

        // CP reconstruction
        group.bench_with_input(
            BenchmarkId::new("cp_reconstruct", size),
            &(&tensor_3d, cp_rank),
            |b, (t, &rank)| {
                b.iter(|| {
                    let cp_decomp = cp_als(black_box(&t.view()), rank, 50, 1e-6).unwrap();
                    cp_reconstruct(&cp_decomp).unwrap()
                })
            },
        );

        // CP tensor-times-matrix
        let matrix = create_test_matrix(size, size / 2);
        group.bench_with_input(
            BenchmarkId::new("cp_ttm", size),
            &(&tensor_3d, cp_rank, &matrix),
            |b, (t, &rank, m)| {
                b.iter(|| {
                    let cp_decomp = cp_als(black_box(&t.view()), rank, 20, 1e-6).unwrap();
                    cp_ttm(&cp_decomp, black_box(&m.view()), 0).unwrap()
                })
            },
        );

        // CP normalization
        group.bench_with_input(
            BenchmarkId::new("cp_normalize", size),
            &(&tensor_3d, cp_rank),
            |b, (t, &rank)| {
                b.iter(|| {
                    let mut cp_decomp = cp_als(black_box(&t.view()), rank, 20, 1e-6).unwrap();
                    cp_normalize(&mut cp_decomp).unwrap()
                })
            },
        );
    }

    group.finish();
}

/// Benchmark tensor network operations
fn bench_tensor_networks(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_networks");
    group.sample_size(15);

    for &size in &[10, 20, 30] {
        let tensor_3d = create_test_tensor_3d(size, size, size);
        let matrix = create_test_matrix(size, size);

        group.throughput(Throughput::Elements(
            size as u64 * size as u64 * size as u64,
        ));

        // Matrix Product State (MPS) decomposition
        let bond_dims = vec![1, size / 2, size / 2, 1];
        group.bench_with_input(
            BenchmarkId::new("mps_decomposition", size),
            &(&tensor_3d, &bond_dims),
            |b, (t, dims)| {
                b.iter(|| mps_decomposition(black_box(&t.view()), black_box(dims)).unwrap())
            },
        );

        // Matrix Product Operator (MPO) application
        group.bench_with_input(
            BenchmarkId::new("mpo_application", size),
            &(&tensor_3d, &matrix),
            |b, (t, m)| b.iter(|| mpo_apply(black_box(&t.view()), black_box(&m.view())).unwrap()),
        );

        // Tree Tensor Network contraction
        group.bench_with_input(
            BenchmarkId::new("ttn_contraction", size),
            &tensor_3d,
            |b, t| b.iter(|| tree_tensor_network_contract(black_box(&t.view())).unwrap()),
        );

        // PEPS (Projected Entangled Pair States) operations (2D only)
        if size <= 20 {
            let matrix_2d = create_test_matrix(size, size);
            group.bench_with_input(
                BenchmarkId::new("peps_contraction", size),
                &matrix_2d,
                |b, m| b.iter(|| peps_contraction(black_box(&m.view())).unwrap()),
            );
        }

        // Tensor network simplification
        group.bench_with_input(BenchmarkId::new("tn_simplify", size), &tensor_3d, |b, t| {
            b.iter(|| tensor_network_simplify(black_box(&t.view())).unwrap())
        });
    }

    group.finish();
}

/// Benchmark batch tensor operations for ML applications
fn bench_batch_tensor_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_tensor_operations");
    group.sample_size(20);

    for &batch_size in &[10, 50, 100] {
        for &size in &[10, 20, 30] {
            let batch_tensor = create_test_tensor_4d(batch_size, size, size, size);
            let weight_matrix = create_test_matrix(size, size);

            group.throughput(Throughput::Elements(
                batch_size as u64 * size as u64 * size as u64 * size as u64,
            ));

            // Batch tensor-matrix multiplication
            group.bench_with_input(
                BenchmarkId::new(format!("batch_tmm_{}x{}", batch_size, size), size),
                &(&batch_tensor, &weight_matrix),
                |b, (t, w)| {
                    b.iter(|| {
                        batch_tensor_matrix_multiply(black_box(&t.view()), black_box(&w.view()))
                            .unwrap()
                    })
                },
            );

            // Batch tensor contraction
            group.bench_with_input(
                BenchmarkId::new(format!("batch_contract_{}x{}", batch_size, size), size),
                &batch_tensor,
                |b, t| b.iter(|| batch_tensor_contract(black_box(&t.view())).unwrap()),
            );

            // Batch mode-n product
            group.bench_with_input(
                BenchmarkId::new(format!("batch_mode_n_{}x{}", batch_size, size), size),
                &(&batch_tensor, &weight_matrix),
                |b, (t, w)| {
                    b.iter(|| {
                        batch_mode_n_product(black_box(&t.view()), black_box(&w.view()), 1).unwrap()
                    })
                },
            );

            // Batch tensor normalization
            group.bench_with_input(
                BenchmarkId::new(format!("batch_normalize_{}x{}", batch_size, size), size),
                &batch_tensor,
                |b, t| b.iter(|| batch_tensor_normalize(black_box(&t.view())).unwrap()),
            );
        }
    }

    group.finish();
}

/// Benchmark memory efficiency in tensor operations
fn bench_tensor_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_memory_efficiency");
    group.sample_size(15);

    for &size in &[20, 30, 50] {
        let tensor_3d = create_test_tensor_3d(size, size, size);

        group.throughput(Throughput::Elements(
            size as u64 * size as u64 * size as u64,
        ));

        // In-place vs out-of-place tensor operations
        group.bench_with_input(
            BenchmarkId::new("tensor_copy_overhead", size),
            &tensor_3d,
            |b, t| {
                b.iter(|| {
                    let _copy = t.clone();
                    mode_n_product(black_box(&t.view()), &Array2::eye(size), 0).unwrap()
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("tensor_no_copy", size),
            &tensor_3d,
            |b, t| b.iter(|| mode_n_product(black_box(&t.view()), &Array2::eye(size), 0).unwrap()),
        );

        // Memory-efficient Tucker decomposition
        let tucker_ranks = vec![size / 2, size / 2, size / 2];
        group.bench_with_input(
            BenchmarkId::new("tucker_memory_efficient", size),
            &(&tensor_3d, &tucker_ranks),
            |b, (t, ranks)| {
                b.iter(|| {
                    tucker_memory_efficient(black_box(&t.view()), black_box(ranks), 20, 1e-6)
                        .unwrap()
                })
            },
        );

        // Streaming tensor operations (chunk-wise)
        group.bench_with_input(
            BenchmarkId::new("tensor_streaming", size),
            &tensor_3d,
            |b, t| b.iter(|| tensor_streaming_operation(black_box(&t.view()), size / 2).unwrap()),
        );
    }

    group.finish();
}

// Group all benchmarks
criterion_group!(
    benches,
    bench_tensor_contraction,
    bench_einsum_operations,
    bench_hosvd_operations,
    bench_tucker_decomposition,
    bench_tensor_train_decomposition,
    bench_cp_decomposition,
    bench_tensor_networks,
    bench_batch_tensor_operations,
    bench_tensor_memory_efficiency
);

criterion_main!(benches);
