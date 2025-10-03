//! Example demonstrating sparse-dense matrix operations
//!
//! This example demonstrates how to use the sparse-dense operations module
//! for efficient operations between sparse and dense matrices.

use scirs2_core::ndarray::{array, Array1, Array2};
use scirs2_linalg::sparse_dense::{
    sparse_dense_add, sparse_dense_elementwise_mul, sparse_dense_matmul, sparse_dense_matvec,
    sparse_from_ndarray, sparse_transpose,
};

#[allow(dead_code)]
fn main() {
    println!("Sparse-Dense Matrix Operations Example");
    println!("======================================\n");

    // Create a dense matrix with some zero elements
    let dense_a = array![[1.0, 0.0, 2.0], [0.0, 0.0, 3.0], [4.0, 5.0, 0.0]];

    // Create another dense matrix
    let dense_b = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

    // Create a dense vector
    let vec_c = array![1.0, 2.0, 3.0];

    println!("Dense Matrix A:");
    println!("{:?}\n", dense_a);

    println!("Dense Matrix B:");
    println!("{:?}\n", dense_b);

    println!("Dense Vector C:");
    println!("{:?}\n", vec_c);

    // Convert the first matrix to sparse format
    println!("Converting Matrix A to sparse format...");
    let sparse_a = sparse_from_ndarray(&dense_a.view(), 1e-10).unwrap();
    println!("Number of non-zeros: {}\n", sparse_a.nnz());

    // Basic Information
    println!("Sparse Matrix Properties:");
    println!("Dimensions: {:?}", sparse_a.shape());
    println!("Non-zero elements: {}", sparse_a.nnz());
    println!(
        "Density: {:.2}%\n",
        (sparse_a.nnz() as f64 / (sparse_a.rows * sparse_a.cols) as f64) * 100.0
    );

    // Matrix-Matrix Multiplication
    println!("Sparse-Dense Matrix Multiplication (A * B):");
    let result_ab = sparse_dense_matmul(&sparse_a, &dense_b.view()).unwrap();
    println!("{:?}\n", result_ab);

    // Verify with standard matrix multiplication
    println!("Verifying with standard dense matrix multiplication:");
    let expected_ab = dense_a.dot(&dense_b);
    println!("{:?}\n", expected_ab);

    let error_ab = (&result_ab - &expected_ab).mapv(|x: f64| x.abs()).sum() / expected_ab.sum();
    println!("Relative error: {:.2e}\n", error_ab);

    // Matrix-Vector Multiplication
    println!("Sparse-Dense Matrix-Vector Multiplication (A * C):");
    let result_ac = sparse_dense_matvec(&sparse_a, &vec_c.view()).unwrap();
    println!("{:?}\n", result_ac);

    // Verify with standard matrix-vector multiplication
    println!("Verifying with standard dense matrix-vector multiplication:");
    let expected_ac = dense_a.dot(&vec_c);
    println!("{:?}\n", expected_ac);

    let error_ac = (&result_ac - &expected_ac).mapv(|x: f64| x.abs()).sum() / expected_ac.sum();
    println!("Relative error: {:.2e}\n", error_ac);

    // Create another dense matrix for element-wise operations
    let dense_d = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];

    println!("Dense Matrix D:");
    println!("{:?}\n", dense_d);

    // Element-wise Addition
    println!("Sparse-Dense Element-wise Addition (A + D):");
    let result_ad_add = sparse_dense_add(&sparse_a, &dense_d.view()).unwrap();
    println!("{:?}\n", result_ad_add);

    // Verify with standard element-wise addition
    println!("Verifying with standard dense element-wise addition:");
    let expected_ad_add = &dense_a + &dense_d;
    println!("{:?}\n", expected_ad_add);

    let error_ad_add = (&result_ad_add - &expected_ad_add)
        .mapv(|x: f64| x.abs())
        .sum()
        / expected_ad_add.sum();
    println!("Relative error: {:.2e}\n", error_ad_add);

    // Element-wise Multiplication
    println!("Sparse-Dense Element-wise Multiplication (A .* D):");
    let result_ad_mul = sparse_dense_elementwise_mul(&sparse_a, &dense_d.view()).unwrap();
    let result_ad_mul_dense = result_ad_mul.to_dense();
    println!("{:?}\n", result_ad_mul_dense);

    // Verify with standard element-wise multiplication
    println!("Verifying with standard dense element-wise multiplication:");
    let expected_ad_mul = &dense_a * &dense_d;
    println!("{:?}\n", expected_ad_mul);

    let error_ad_mul = (&result_ad_mul_dense - &expected_ad_mul)
        .mapv(|x: f64| x.abs())
        .sum()
        / expected_ad_mul.sum();
    println!("Relative error: {:.2e}\n", error_ad_mul);

    // Transpose
    println!("Sparse Matrix Transpose:");
    let sparse_a_t = sparse_transpose(&sparse_a).unwrap();
    let sparse_a_t_dense = sparse_a_t.to_dense();
    println!("{:?}\n", sparse_a_t_dense);

    // Verify with standard transpose
    println!("Verifying with standard dense transpose:");
    let expected_a_t = dense_a.t().to_owned();
    println!("{:?}\n", expected_a_t);

    let error_a_t = (&sparse_a_t_dense - &expected_a_t)
        .mapv(|x: f64| x.abs())
        .sum()
        / expected_a_t.sum();
    println!("Relative error: {:.2e}\n", error_a_t);

    // Performance Benchmark (Simple)
    println!("Simple Performance Benchmark:");
    println!("----------------------------");

    // Create larger matrices for benchmark
    let n = 500;
    let m = 500;
    let k = 100; // Sparsity factor (1 in k elements is non-zero)

    println!(
        "Creating a {}x{} sparse matrix with ~{}% non-zeros...",
        n,
        n,
        100.0 / (k as f64)
    );

    // Create a random sparse matrix
    let mut dense_large = Array2::<f64>::zeros((n, m));
    let mut nnz = 0;

    // Use a simpler approach with a fixed pattern for the sparse matrix
    for i in 0..n {
        for j in 0..m {
            if (i + j) % k == 0 {
                dense_large[[i, j]] = (i * j) as f64 / (n * m) as f64;
                nnz += 1;
            }
        }
    }

    println!(
        "Created sparse matrix with {} non-zeros ({:.2}% density)",
        nnz,
        (nnz as f64 / (n * m) as f64) * 100.0
    );

    // Convert to sparse format
    let sparse_large = sparse_from_ndarray(&dense_large.view(), 1e-10).unwrap();

    // Create a dense vector for testing
    let dense_vec = Array1::<f64>::ones(m);

    // Measure time for sparse-dense matvec
    use std::time::Instant;

    println!("\nMatrix-Vector Multiplication Benchmark:");
    let start = Instant::now();
    for _ in 0..10 {
        let _ = sparse_dense_matvec(&sparse_large, &dense_vec.view()).unwrap();
    }
    let sparse_time = start.elapsed();

    let start = Instant::now();
    for _ in 0..10 {
        let _ = dense_large.dot(&dense_vec);
    }
    let dense_time = start.elapsed();

    println!("Time for 10 sparse-dense matvecs: {:?}", sparse_time);
    println!("Time for 10 dense-dense matvecs: {:?}", dense_time);

    if dense_time > sparse_time {
        println!(
            "Sparse-dense is {:.2}x faster",
            dense_time.as_secs_f64() / sparse_time.as_secs_f64()
        );
    } else {
        println!(
            "Dense-dense is {:.2}x faster",
            sparse_time.as_secs_f64() / dense_time.as_secs_f64()
        );
    }
}
