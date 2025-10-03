//! Example demonstrating the use of specialized solvers for quantized matrices
//!
//! This example shows how to use the specialized iterative solvers for quantized
//! matrices, including conjugate gradient, GMRES, and preconditioned variants.

use scirs2_core::ndarray::{array, Array1, Array2, ArrayView1};
use scirs2_linalg::matrixfree::{conjugate_gradient, gmres, LinearOperator};
use scirs2_linalg::quantization::{
    quantized_matrixfree::QuantizedMatrixFreeOp,
    solvers::{
        quantized_conjugate_gradient, quantized_gmres, quantized_jacobi_preconditioner,
        quantized_preconditioned_conjugate_gradient,
    },
    QuantizationMethod,
};

#[allow(dead_code)]
fn main() {
    println!("=== Quantized Matrix Solvers Example ===");

    // Example 1: Solving a small symmetric positive definite system with CG
    println!("\n--- Example 1: Conjugate Gradient with Quantized Matrix ---");
    example_conjugate_gradient();

    // Example 2: Solving a non-symmetric system with GMRES
    println!("\n--- Example 2: GMRES with Quantized Matrix ---");
    example_gmres();

    // Example 3: Preconditioned conjugate gradient
    println!("\n--- Example 3: Preconditioned Conjugate Gradient ---");
    example_preconditioned_cg();

    // Example 4: Solving a tridiagonal system (banded matrix)
    println!("\n--- Example 4: Banded Matrix Example ---");
    example_bandedmatrix();

    // Example 5: Adaptive precision for ill-conditioned matrices
    println!("\n--- Example 5: Adaptive Precision for Ill-conditioned Matrix ---");
    example_adaptive_precision();
}

/// Example of using conjugate gradient with a quantized matrix
#[allow(dead_code)]
fn example_conjugate_gradient() {
    // Create a small symmetric positive definite matrix
    let matrix = array![[4.0f32, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 5.0]];
    println!("Matrix A:\n{:?}", matrix);

    // Create a right-hand side vector
    let b = array![1.0f32, 2.0, 3.0];
    println!("Right-hand side b: {:?}", b);

    // Create a standard LinearOperator for comparison
    let matrix_clone = matrix.clone();
    let standard_op = LinearOperator::new(3, move |v: &ArrayView1<f32>| matrix_clone.dot(v))
        .symmetric()
        .positive_definite();

    // Solve with standard conjugate gradient
    let x_standard = conjugate_gradient(&standard_op, &b, 10, 1e-6).unwrap();
    println!("Solution using standard CG: {:?}", x_standard);

    // Create a quantized matrix operator with 8-bit precision
    let quantized_op =
        QuantizedMatrixFreeOp::frommatrix(&matrix.view(), 8, QuantizationMethod::Symmetric)
            .unwrap()
            .symmetric()
            .positive_definite();

    // Solve with specialized quantized conjugate gradient
    let x_quantized = quantized_conjugate_gradient(&quantized_op, &b, 10, 1e-6, false).unwrap();
    println!("Solution using quantized CG (8-bit): {:?}", x_quantized);

    // Create a quantized matrix operator with 4-bit precision
    let quantized_op_4bit =
        QuantizedMatrixFreeOp::frommatrix(&matrix.view(), 4, QuantizationMethod::Symmetric)
            .unwrap()
            .symmetric()
            .positive_definite();

    // Solve with specialized quantized conjugate gradient
    let x_quantized_4bit =
        quantized_conjugate_gradient(&quantized_op_4bit, &b, 10, 1e-6, false).unwrap();
    println!(
        "Solution using quantized CG (4-bit): {:?}",
        x_quantized_4bit
    );

    // Verify the solutions
    let residual_standard = &matrix.dot(&x_standard) - &b;
    let residual_quantized = &matrix.dot(&x_quantized) - &b;
    let residual_quantized_4bit = &matrix.dot(&x_quantized_4bit) - &b;

    println!("Residual norm (standard): {}", l2_norm(&residual_standard));
    println!("Residual norm (8-bit): {}", l2_norm(&residual_quantized));
    println!(
        "Residual norm (4-bit): {}",
        l2_norm(&residual_quantized_4bit)
    );
}

/// Example of using GMRES with a quantized matrix
#[allow(dead_code)]
fn example_gmres() {
    // Create a small non-symmetric matrix
    let matrix = array![[3.0f32, 1.0, 0.5], [1.0, 4.0, 2.0], [0.5, 1.0, 3.0]];
    println!("Matrix A:\n{:?}", matrix);

    // Create a right-hand side vector
    let b = array![1.0f32, 2.0, 3.0];
    println!("Right-hand side b: {:?}", b);

    // Create a standard LinearOperator for comparison
    let matrix_clone = matrix.clone();
    let standard_op = LinearOperator::new(3, move |v: &ArrayView1<f32>| matrix_clone.dot(v));

    // Solve with standard GMRES
    let x_standard = gmres(&standard_op, &b, 10, 1e-6, None).unwrap();
    println!("Solution using standard GMRES: {:?}", x_standard);

    // Create a quantized matrix operator with 8-bit precision
    let quantized_op =
        QuantizedMatrixFreeOp::frommatrix(&matrix.view(), 8, QuantizationMethod::Symmetric)
            .unwrap();

    // Solve with specialized quantized GMRES
    let x_quantized = quantized_gmres(&quantized_op, &b, 10, 1e-6, None, false).unwrap();
    println!("Solution using quantized GMRES (8-bit): {:?}", x_quantized);

    // Create a quantized matrix operator with 4-bit precision
    let quantized_op_4bit =
        QuantizedMatrixFreeOp::frommatrix(&matrix.view(), 4, QuantizationMethod::Symmetric)
            .unwrap();

    // Solve with specialized quantized GMRES
    let x_quantized_4bit = quantized_gmres(&quantized_op_4bit, &b, 10, 1e-6, None, false).unwrap();
    println!(
        "Solution using quantized GMRES (4-bit): {:?}",
        x_quantized_4bit
    );

    // Verify the solutions
    let residual_standard = &matrix.dot(&x_standard) - &b;
    let residual_quantized = &matrix.dot(&x_quantized) - &b;
    let residual_quantized_4bit = &matrix.dot(&x_quantized_4bit) - &b;

    println!("Residual norm (standard): {}", l2_norm(&residual_standard));
    println!("Residual norm (8-bit): {}", l2_norm(&residual_quantized));
    println!(
        "Residual norm (4-bit): {}",
        l2_norm(&residual_quantized_4bit)
    );
}

/// Example of using preconditioned conjugate gradient with a quantized matrix
#[allow(dead_code)]
fn example_preconditioned_cg() {
    // Create a small symmetric positive definite matrix
    let matrix = array![[10.0f32, 1.0, 0.0], [1.0, 8.0, 3.0], [0.0, 3.0, 15.0]];
    println!("Matrix A:\n{:?}", matrix);

    // Create a right-hand side vector
    let b = array![1.0f32, 2.0, 3.0];
    println!("Right-hand side b: {:?}", b);

    // Create a quantized matrix operator with 8-bit precision
    let quantized_op =
        QuantizedMatrixFreeOp::frommatrix(&matrix.view(), 8, QuantizationMethod::Symmetric)
            .unwrap()
            .symmetric()
            .positive_definite();

    // Create a Jacobi preconditioner
    let precond = quantized_jacobi_preconditioner(&quantized_op).unwrap();

    // Solve with standard quantized conjugate gradient (without preconditioner)
    let x_unpreconditioned =
        quantized_conjugate_gradient(&quantized_op, &b, 10, 1e-6, false).unwrap();
    println!(
        "Solution using unpreconditioned CG: {:?}",
        x_unpreconditioned
    );

    // Solve with preconditioned conjugate gradient
    let x_preconditioned =
        quantized_preconditioned_conjugate_gradient(&quantized_op, &precond, &b, 10, 1e-6, false)
            .unwrap();
    println!("Solution using preconditioned CG: {:?}", x_preconditioned);

    // Verify the solutions
    let residual_unpreconditioned = &matrix.dot(&x_unpreconditioned) - &b;
    let residual_preconditioned = &matrix.dot(&x_preconditioned) - &b;

    println!(
        "Residual norm (unpreconditioned): {}",
        l2_norm(&residual_unpreconditioned)
    );
    println!(
        "Residual norm (preconditioned): {}",
        l2_norm(&residual_preconditioned)
    );
}

/// Example of solving a system with a tridiagonal matrix (banded structure)
#[allow(dead_code)]
fn example_bandedmatrix() {
    // Define a tridiagonal matrix using its bands
    let n = 10;
    let main_diag = Array1::from_vec(vec![2.0f32; n]);
    let upper_diag = Array1::from_vec(vec![-1.0f32; n - 1]);
    let lower_diag = Array1::from_vec(vec![-1.0f32; n - 1]);

    // Create a right-hand side vector
    let b = Array1::from_vec(vec![1.0f32; n]);
    println!("Right-hand side b: {:?}", b);

    // Create a banded matrix operator directly
    let bands = vec![
        (0, main_diag.view()),
        (1, upper_diag.view()),
        (-1, lower_diag.view()),
    ];

    let banded_op = QuantizedMatrixFreeOp::banded(n, bands, 8, QuantizationMethod::Symmetric)
        .unwrap()
        .symmetric()
        .positive_definite();

    // Solve with specialized quantized conjugate gradient
    let x_banded = quantized_conjugate_gradient(&banded_op, &b, 20, 1e-6, false).unwrap();
    println!("Solution using banded matrix CG: {:?}", x_banded);

    // For verification, we'll create a dense matrix representation
    let mut densematrix = Array2::zeros((n, n));
    for i in 0..n {
        densematrix[[i, i]] = 2.0;
    }
    for i in 0..n - 1 {
        densematrix[[i, i + 1]] = -1.0;
        densematrix[[i + 1, i]] = -1.0;
    }

    // Verify the solution
    let residual = &densematrix.dot(&x_banded) - &b;
    println!("Residual norm: {}", l2_norm(&residual));
}

/// Example demonstrating adaptive precision for ill-conditioned matrices
#[allow(dead_code)]
fn example_adaptive_precision() {
    // Create an ill-conditioned matrix
    let matrix = array![[100.0f32, 99.0, 0.0], [99.0, 100.0, 0.01], [0.0, 0.01, 1.0]];
    println!("Ill-conditioned matrix A:\n{:?}", matrix);

    // Create a right-hand side vector
    let b = array![1.0f32, 1.0, 1.0];
    println!("Right-hand side b: {:?}", b);

    // Create a quantized matrix operator with 4-bit precision
    // This low precision will exacerbate the ill-conditioning
    let quantized_op =
        QuantizedMatrixFreeOp::frommatrix(&matrix.view(), 4, QuantizationMethod::Symmetric)
            .unwrap()
            .symmetric()
            .positive_definite();

    // Solve with standard quantized conjugate gradient
    let x_standard = quantized_conjugate_gradient(&quantized_op, &b, 50, 1e-5, false).unwrap();
    println!("Solution using standard quantized CG: {:?}", x_standard);

    // Solve with adaptive precision
    let x_adaptive = quantized_conjugate_gradient(&quantized_op, &b, 50, 1e-5, true).unwrap();
    println!("Solution using adaptive precision CG: {:?}", x_adaptive);

    // Verify the solutions
    let residual_standard = &matrix.dot(&x_standard) - &b;
    let residual_adaptive = &matrix.dot(&x_adaptive) - &b;

    println!("Residual norm (standard): {}", l2_norm(&residual_standard));
    println!("Residual norm (adaptive): {}", l2_norm(&residual_adaptive));
}

/// Helper function to compute L2 norm of a vector
#[allow(dead_code)]
fn l2_norm(v: &Array1<f32>) -> f32 {
    v.dot(v).sqrt()
}
