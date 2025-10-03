//! Example demonstrating matrix trigonometric functions
//!
//! This example shows how to use the matrix trigonometric functions
//! (cosm, sinm, tanm) in both the direct API and SciPy-compatible interface.

use scirs2_core::ndarray::array;
use scirs2_linalg::compat;
use scirs2_linalg::matrix_functions::{cosm, sinm, tanm};

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Matrix Trigonometric Functions Example");
    println!("=====================================\n");

    // Test with zero matrix
    println!("1. Zero Matrix Tests:");
    let zeromatrix = array![[0.0, 0.0], [0.0, 0.0]];

    let cos_zero = cosm(&zeromatrix.view())?;
    let sin_zero = sinm(&zeromatrix.view())?;
    let tan_zero = tanm(&zeromatrix.view())?;

    println!("cos(0) = \n{:8.6}", cos_zero);
    println!("sin(0) = \n{:8.6}", sin_zero);
    println!("tan(0) = \n{:8.6}", tan_zero);
    println!();

    // Test with diagonal matrix
    println!("2. Diagonal Matrix Tests:");
    let diagmatrix = array![
        [std::f64::consts::FRAC_PI_4, 0.0],
        [0.0, std::f64::consts::FRAC_PI_6]
    ];

    let cos_diag = cosm(&diagmatrix.view())?;
    let sin_diag = sinm(&diagmatrix.view())?;
    let tan_diag = tanm(&diagmatrix.view())?;

    println!("Matrix A = \n{:8.6}", diagmatrix);
    println!("cos(A) = \n{:8.6}", cos_diag);
    println!("sin(A) = \n{:8.6}", sin_diag);
    println!("tan(A) = \n{:8.6}", tan_diag);
    println!();

    // Test fundamental trigonometric identity: sin²(A) + cos²(A) = I
    println!("3. Trigonometric Identity Test:");
    let testmatrix = array![[0.1, 0.02], [0.02, 0.1]];

    let sin_test = sinm(&testmatrix.view())?;
    let cos_test = cosm(&testmatrix.view())?;

    // Compute sin²(A) and cos²(A)
    let sin_squared = sin_test.dot(&sin_test);
    let cos_squared = cos_test.dot(&cos_test);

    // sin²(A) + cos²(A) should equal I
    let identity_test = &sin_squared + &cos_squared;
    let identity = array![[1.0, 0.0], [0.0, 1.0]];

    println!("Test matrix A = \n{:8.6}", testmatrix);
    println!("sin²(A) + cos²(A) = \n{:8.6}", identity_test);
    println!("Expected identity = \n{:8.6}", identity);

    let diff = &identity_test - &identity;
    let max_error = diff.iter().map(|&x: &f64| x.abs()).fold(0.0, f64::max);
    println!("Maximum error: {:.2e}", max_error);
    println!();

    // Test using SciPy-compatible interface
    println!("4. SciPy-Compatible Interface:");
    let scipymatrix = array![[0.0, 0.5], [-0.5, 0.0]]; // Antisymmetric matrix

    let scipy_cos = compat::cosm(&scipymatrix.view())?;
    let scipy_sin = compat::sinm(&scipymatrix.view())?;
    let scipy_tan = compat::tanm(&scipymatrix.view())?;

    println!("Antisymmetric matrix A = \n{:8.6}", scipymatrix);
    println!("cos(A) via SciPy interface = \n{:8.6}", scipy_cos);
    println!("sin(A) via SciPy interface = \n{:8.6}", scipy_sin);
    println!("tan(A) via SciPy interface = \n{:8.6}", scipy_tan);
    println!();

    // Verify that exp(A) ≈ cos(A) + sin(A) for antisymmetric matrix
    println!("5. Relationship with Matrix Exponential:");
    let exp_antisym = scirs2_linalg::matrix_functions::expm(&scipymatrix.view(), None)?;
    let cos_plus_sin = &scipy_cos + &scipy_sin;

    println!("exp(A) = \n{:8.6}", exp_antisym);
    println!("cos(A) + sin(A) = \n{:8.6}", cos_plus_sin);

    let exp_diff = &exp_antisym - &cos_plus_sin;
    let exp_max_error = exp_diff.iter().map(|&x: &f64| x.abs()).fold(0.0, f64::max);
    println!("Max difference: {:.2e}", exp_max_error);
    println!();

    // Show that for antisymmetric matrices, exp(A) is orthogonal
    println!("6. Orthogonality Test for exp(Antisymmetric):");
    let exp_transpose = exp_antisym.t();
    let should_be_identity = exp_antisym.dot(&exp_transpose);

    println!("exp(A) * exp(A)^T = \n{:8.6}", should_be_identity);

    let ortho_diff = &should_be_identity - &identity;
    let ortho_error = ortho_diff
        .iter()
        .map(|&x: &f64| x.abs())
        .fold(0.0, f64::max);
    println!("Orthogonality error: {:.2e}", ortho_error);

    println!("\nMatrix trigonometric functions implemented successfully!");

    Ok(())
}
