//! Example demonstrating parallel algorithm dispatch in scirs2-linalg
//!
//! This example shows how to use the parallel dispatch functionality
//! to automatically select between serial and parallel implementations
//! based on matrix size and worker configuration.

use scirs2_core::ndarray::{array, Array2};
use scirs2_linalg::parallel::set_global_workers;
use scirs2_linalg::parallel_dispatch::{ParallelDecomposition, ParallelOperations, ParallelSolver};

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Set global worker count for all operations
    set_global_workers(Some(4));

    // Example 1: Parallel Cholesky decomposition
    println!("=== Parallel Cholesky Decomposition ===");

    // Small matrix - will use serial implementation
    let smallmatrix = array![[4.0, 2.0], [2.0, 5.0]];
    let l_small = ParallelDecomposition::cholesky(&smallmatrix.view(), Some(4))?;
    println!("Small matrix Cholesky result shape: {:?}", l_small.shape());

    // Large matrix - will use parallel implementation if workers > 0
    let n = 100;
    let largematrix = create_spdmatrix(n);
    let l_large = ParallelDecomposition::cholesky(&largematrix.view(), Some(4))?;
    println!("Large matrix Cholesky result shape: {:?}", l_large.shape());

    // Example 2: Parallel LU decomposition
    println!("\n=== Parallel LU Decomposition ===");
    let matrix = Array2::from_shape_fn((50, 50), |(i, j)| {
        if i == j {
            10.0
        } else {
            0.1 * ((i + j) as f64).sin()
        }
    });
    let (p, l, u) = ParallelDecomposition::lu(&matrix.view(), Some(4))?;
    println!(
        "LU decomposition complete - P: {:?}, L: {:?}, U: {:?}",
        p.shape(),
        l.shape(),
        u.shape()
    );

    // Example 3: Parallel iterative solvers
    println!("\n=== Parallel Iterative Solvers ===");
    let a = create_spdmatrix(50);
    let b = Array2::from_shape_fn((50, 1), |(i, _j)| (i as f64).sin())
        .column(0)
        .to_owned();

    // Conjugate Gradient
    let x_cg = ParallelSolver::conjugate_gradient(&a.view(), &b.view(), 100, 1e-10, Some(4))?;
    println!("CG solution norm: {}", x_cg.mapv(|x| x * x).sum().sqrt());

    // Jacobi method
    let x_jacobi = ParallelSolver::jacobi(&a.view(), &b.view(), 100, 1e-6, Some(4))?;
    println!(
        "Jacobi solution norm: {}",
        x_jacobi.mapv(|x| x * x).sum().sqrt()
    );

    // Example 4: Parallel matrix operations
    println!("\n=== Parallel Matrix Operations ===");
    let mat_a = Array2::from_shape_fn((100, 80), |(i, j)| ((i + j) as f64).sin());
    let mat_b = Array2::from_shape_fn((80, 60), |(i, j)| ((i * j) as f64).cos());

    // Matrix multiplication
    let c = ParallelOperations::matmul(&mat_a.view(), &mat_b.view(), Some(4))?;
    println!("Matrix multiplication result shape: {:?}", c.shape());

    // Matrix-vector multiplication
    let vec = Array2::from_shape_fn((80, 1), |(i, _j)| (i as f64).sin())
        .column(0)
        .to_owned();
    let result = ParallelOperations::matvec(&mat_a.view(), &vec.view(), Some(4))?;
    println!(
        "Matrix-vector multiplication result shape: {:?}",
        result.shape()
    );

    // Example 5: Comparing serial vs parallel performance
    println!("\n=== Performance Comparison ===");
    use std::time::Instant;

    let large_mat = create_spdmatrix(200);

    // Serial execution (no workers specified)
    let start = Instant::now();
    let _ = ParallelDecomposition::cholesky(&large_mat.view(), None)?;
    let serial_time = start.elapsed();
    println!("Serial Cholesky time: {:?}", serial_time);

    // Parallel execution with 4 workers
    let start = Instant::now();
    let _ = ParallelDecomposition::cholesky(&large_mat.view(), Some(4))?;
    let parallel_time = start.elapsed();
    println!("Parallel Cholesky time (4 workers): {:?}", parallel_time);

    let speedup = serial_time.as_secs_f64() / parallel_time.as_secs_f64();
    println!("Speedup: {:.2}x", speedup);

    Ok(())
}

/// Create a symmetric positive definite matrix
#[allow(dead_code)]
fn create_spdmatrix(n: usize) -> Array2<f64> {
    let a = Array2::from_shape_fn((n, n), |(i, j)| ((i + j + 1) as f64 * 0.1).sin());
    // Make it symmetric positive definite
    let sym = &a + &a.t();
    &sym.dot(&sym.t()) + Array2::eye(n) * (n as f64)
}
