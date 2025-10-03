//! Comprehensive Linear Algebra Example
//!
//! This example demonstrates the core linear algebra functionality provided by scirs2-linalg,
//! including basic operations, decompositions, solvers, and parallel processing.

use scirs2_core::ndarray::{array, Array1, Array2};
use scirs2_linalg::{
    cholesky, det, eig, eigh,
    error::LinalgResult,
    inv, lu, matrix_norm,
    parallel::{algorithms, WorkerConfig},
    qr, solve, svd, vector_norm,
};

#[allow(dead_code)]
fn main() -> LinalgResult<()> {
    println!("=== Comprehensive Linear Algebra Example ===\n");

    // Basic matrix operations
    basicmatrix_operations()?;

    // Matrix decompositions
    matrix_decompositions()?;

    // Linear system solving
    linear_system_solving()?;

    // Eigenvalue computations
    eigenvalue_computations()?;

    // Parallel processing examples
    parallel_processing_examples()?;

    // Error handling demonstration
    error_handling_demonstration();

    println!("✅ All examples completed successfully!");
    Ok(())
}

/// Demonstrates basic matrix operations
#[allow(dead_code)]
fn basicmatrix_operations() -> LinalgResult<()> {
    println!("📊 Basic Matrix Operations");
    println!("{}", "=".repeat(50));

    // Create test matrices
    let a = array![[1.0, 2.0], [3.0, 4.0]];
    let b = array![[5.0, 6.0], [7.0, 8.0]];

    println!("Matrix A:");
    println!("{:?}", a);
    println!("\nMatrix B:");
    println!("{:?}", b);

    // Matrix addition and multiplication
    let sum = &a + &b;
    let product = a.dot(&b);

    println!("\nA + B:");
    println!("{:?}", sum);
    println!("\nA * B:");
    println!("{:?}", product);

    // Determinant calculation
    let det_a = det(&a.view(), None)?;
    println!("\nDeterminant of A: {:.6}", det_a);

    // Matrix inverse
    if let Ok(inv_a) = inv(&a.view(), None) {
        println!("\nInverse of A:");
        println!("{:?}", inv_a);

        // Verify: A * A^-1 = I
        let identity_check = a.dot(&inv_a);
        println!("\nA * A^(-1) (should be identity):");
        println!("{:.6?}", identity_check);
    }

    // Matrix norms
    let frobenius_norm = matrix_norm(&a.view(), "fro", None)?;
    let one_norm = matrix_norm(&a.view(), "1", None)?;
    let inf_norm = matrix_norm(&a.view(), "inf", None)?;

    println!("\nMatrix norms of A:");
    println!("  Frobenius: {:.6}", frobenius_norm);
    println!("  1-norm:    {:.6}", one_norm);
    println!("  ∞-norm:    {:.6}", inf_norm);

    // Vector operations
    let x = array![1.0, 2.0, 3.0];
    let l1_norm = vector_norm(&x.view(), 1)?;
    let l2_norm = vector_norm(&x.view(), 2)?;
    let linf_norm = vector_norm(&x.view(), usize::MAX)?;

    println!("\nVector x = {:?}", x);
    println!("Vector norms:");
    println!("  L1:  {:.6}", l1_norm);
    println!("  L2:  {:.6}", l2_norm);
    println!("  L∞:  {:.6}", linf_norm);

    println!("\n");
    Ok(())
}

/// Demonstrates matrix decompositions
#[allow(dead_code)]
fn matrix_decompositions() -> LinalgResult<()> {
    println!("🔬 Matrix Decompositions");
    println!("{}", "=".repeat(50));

    // Test matrix
    let a = array![[4.0, 2.0, 1.0], [2.0, 5.0, 3.0], [1.0, 3.0, 6.0]];

    println!("Matrix A (3x3):");
    println!("{:?}", a);

    // LU decomposition
    println!("\n🔹 LU Decomposition");
    let (l, u, p) = lu(&a.view(), None)?;
    println!("L matrix:");
    println!("{:.4?}", l);
    println!("U matrix:");
    println!("{:.4?}", u);
    println!("Permutation vector: {:?}", p);

    // QR decomposition
    println!("\n🔹 QR Decomposition");
    let (q, r) = qr(&a.view(), None)?;
    println!("Q matrix:");
    println!("{:.4?}", q);
    println!("R matrix:");
    println!("{:.4?}", r);

    // Verify: Q * R = A
    let qr_product = q.dot(&r);
    println!("Q * R (should equal A):");
    println!("{:.4?}", qr_product);

    // SVD
    println!("\n🔹 Singular Value Decomposition");
    let (u_svd, s, vt) = svd(&a.view(), false, None)?;
    println!("U matrix:");
    println!("{:.4?}", u_svd);
    println!("Singular values: {:?}", s);
    println!("V^T matrix:");
    println!("{:.4?}", vt);

    // Cholesky decomposition (for positive definite matrices)
    println!("\n🔹 Cholesky Decomposition");
    if let Ok(l_chol) = cholesky(&a.view(), None) {
        println!("L matrix:");
        println!("{:.4?}", l_chol);

        // Verify: L * L^T = A
        let lt = l_chol.t();
        let chol_product = l_chol.dot(&lt.view());
        println!("L * L^T (should equal A):");
        println!("{:.4?}", chol_product);
    } else {
        println!("Matrix is not positive definite, Cholesky decomposition unavailable");
    }

    println!("\n");
    Ok(())
}

/// Demonstrates linear system solving
#[allow(dead_code)]
fn linear_system_solving() -> LinalgResult<()> {
    println!("🎯 Linear System Solving");
    println!("{}", "=".repeat(50));

    // System: A * x = b
    let a = array![[3.0, 2.0, -1.0], [2.0, -2.0, 4.0], [-1.0, 0.5, -1.0]];
    let b = array![1.0, -2.0, 0.0];

    println!("Solving system A * x = b");
    println!("Matrix A:");
    println!("{:?}", a);
    println!("Vector b: {:?}", b);

    // Direct solve
    let x = solve(&a.view(), &b.view(), None)?;
    println!("\nSolution x: {:?}", x);

    // Verify solution
    let ax = a.dot(&x);
    println!("A * x = {:?}", ax);
    println!("b     = {:?}", b);

    let residual = &ax - &b;
    let residual_norm = vector_norm(&residual.view(), 2)?;
    println!("Residual norm: {:.2e}", residual_norm);

    // Demonstrate solving multiple right-hand sides
    let b_multi = array![[1.0, 2.0], [-2.0, 1.0], [0.0, -1.0]];

    println!("\n🔹 Multiple Right-Hand Sides");
    println!("B matrix:");
    println!("{:?}", b_multi);

    // Solve for each column
    for i in 0..b_multi.ncols() {
        let bi = b_multi.column(i);
        let xi = solve(&a.view(), &bi, None)?;
        println!("Solution for column {}: {:?}", i, xi);
    }

    println!("\n");
    Ok(())
}

/// Demonstrates eigenvalue and eigenvector computations
#[allow(dead_code)]
fn eigenvalue_computations() -> LinalgResult<()> {
    println!("🌀 Eigenvalue Computations");
    println!("{}", "=".repeat(50));

    // Symmetric matrix for real eigenvalues
    let a_sym = array![[4.0, -2.0, 1.0], [-2.0, 2.0, -1.0], [1.0, -1.0, 3.0]];

    println!("Symmetric matrix A:");
    println!("{:?}", a_sym);

    // Symmetric eigenvalue decomposition
    println!("\n🔹 Symmetric Eigenvalue Decomposition");
    let (eigenvals, eigenvecs) = eigh(&a_sym.view(), None)?;
    println!("Eigenvalues: {:?}", eigenvals);
    println!("Eigenvectors:");
    println!("{:.4?}", eigenvecs);

    // Verify: A * v = λ * v for first eigenpair
    if !eigenvals.is_empty() && eigenvecs.ncols() > 0 {
        let lambda0 = eigenvals[0];
        let v0 = eigenvecs.column(0);
        let av0 = a_sym.dot(&v0);
        let lambda_v0 = &v0 * lambda0;

        println!("\nVerification for first eigenpair:");
        println!("A * v₀ = {:?}", av0);
        println!("λ₀ * v₀ = {:?}", lambda_v0);

        let diff = &av0 - &lambda_v0;
        let error = vector_norm(&diff.view(), 2)?;
        println!("||A*v₀ - λ₀*v₀|| = {:.2e}", error);
    }

    // General eigenvalue decomposition
    let a_gen = array![[1.0, 2.0], [3.0, 4.0]];

    println!("\n🔹 General Eigenvalue Decomposition");
    println!("Matrix A:");
    println!("{:?}", a_gen);

    if let Ok((eigenvals_gen, eigenvecs_gen)) = eig(&a_gen.view(), None) {
        println!("Eigenvalues: {:?}", eigenvals_gen);
        println!("Eigenvectors:");
        println!("{:.4?}", eigenvecs_gen);
    } else {
        println!(
            "General eigenvalue decomposition failed (complex eigenvalues or numerical issues)"
        );
    }

    println!("\n");
    Ok(())
}

/// Demonstrates parallel processing capabilities
#[allow(dead_code)]
fn parallel_processing_examples() -> LinalgResult<()> {
    println!("⚡ Parallel Processing Examples");
    println!("{}", "=".repeat(50));

    // Configure parallel processing
    let config = WorkerConfig::new()
        .with_workers(4)
        .with_threshold(100) // Use parallel for matrices with 100+ elements
        .with_chunksize(32);

    println!("Worker configuration: {:?}", config);

    // Large test matrices
    let size = 50;
    let mut a = Array2::zeros((size, size));
    let mut b = Array2::zeros((size, size));
    let x = Array1::ones(size);

    // Initialize with random-like values
    for i in 0..size {
        for j in 0..size {
            a[[i, j]] = ((i + j) as f64).sin();
            b[[i, j]] = ((i * j + 1) as f64).cos();
        }
    }

    println!("\nTesting with {}x{} matrices", size, size);

    // Parallel matrix-vector multiplication
    println!("\n🔹 Parallel Matrix-Vector Multiplication");
    let result_matvec = algorithms::parallel_matvec(&a.view(), &x.view(), &config)?;
    let reference_matvec = a.dot(&x);

    let diff_matvec = &result_matvec - &reference_matvec;
    let error_matvec = vector_norm(&diff_matvec.view(), 2)?;
    println!("Matrix-vector multiplication error: {:.2e}", error_matvec);

    // Parallel matrix multiplication
    println!("\n🔹 Parallel Matrix Multiplication");
    let result_gemm = algorithms::parallel_gemm(&a.view(), &b.view(), &config)?;
    let reference_gemm = a.dot(&b);

    let diff_gemm = &result_gemm - &reference_gemm;
    let error_gemm = matrix_norm(&diff_gemm.view(), "fro", None)?;
    println!("Matrix multiplication Frobenius error: {:.2e}", error_gemm);

    // Parallel vector operations
    println!("\n🔹 Parallel Vector Operations");
    let y = Array1::from_shape_fn(size, |i| (i as f64 + 1.0).sqrt());

    let dot_result = algorithms::vector_ops::parallel_dot(&x.view(), &y.view(), &config)?;
    let dot_reference = x
        .iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| xi * yi)
        .sum::<f64>();
    println!("Parallel dot product: {:.6}", dot_result);
    println!("Reference dot product: {:.6}", dot_reference);
    println!(
        "Dot product error: {:.2e}",
        (dot_result - dot_reference).abs()
    );

    let norm_result = algorithms::vector_ops::parallel_norm(&x.view(), &config)?;
    let norm_reference = vector_norm(&x.view(), 2)?;
    println!("Parallel norm: {:.6}", norm_result);
    println!("Reference norm: {:.6}", norm_reference);
    println!("Norm error: {:.2e}", (norm_result - norm_reference).abs());

    println!("\n");
    Ok(())
}

/// Demonstrates error handling and validation
#[allow(dead_code)]
fn error_handling_demonstration() {
    println!("🚨 Error Handling Demonstration");
    println!("{}", "=".repeat(50));

    // Dimension mismatch errors
    println!("🔹 Dimension Mismatch Errors");
    let a = array![[1.0, 2.0], [3.0, 4.0]]; // 2x2
    let b = array![1.0, 2.0, 3.0]; // 3x1 (wrong size)

    match solve(&a.view(), &b.view(), None) {
        Ok(_) => println!("Unexpected success!"),
        Err(e) => println!("Expected error: {}", e),
    }

    // Singular matrix errors
    println!("\n🔹 Singular Matrix Errors");
    let singular = array![[1.0, 2.0], [2.0, 4.0]]; // Rank deficient

    match inv(&singular.view(), None) {
        Ok(_) => println!("Unexpected success!"),
        Err(e) => println!("Expected error: {}", e),
    }

    // Invalid norm specification
    println!("\n🔹 Invalid Parameter Errors");
    match matrix_norm(&a.view(), "invalid_norm", None) {
        Ok(_) => println!("Unexpected success!"),
        Err(e) => println!("Expected error: {}", e),
    }

    // Non-finite values
    println!("\n🔹 Non-finite Value Errors");
    let invalidmatrix = array![[1.0, f64::NAN], [f64::INFINITY, 4.0]];

    match det(&invalidmatrix.view(), None) {
        Ok(_) => println!("Unexpected success!"),
        Err(e) => println!("Expected error: {}", e),
    }

    println!("\n✅ Error handling demonstration completed");
    println!("\n");
}
