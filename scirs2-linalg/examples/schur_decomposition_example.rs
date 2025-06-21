//! Example demonstrating Schur decomposition
//!
//! This example shows how to use the Schur decomposition in both the direct API
//! and SciPy-compatible interface. The Schur decomposition factors a matrix A
//! as A = Z * T * Z^T, where Z is orthogonal and T is upper triangular
//! (or quasi-triangular for real matrices).

use ndarray::{array, Array2};
use scirs2_linalg::compat;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Schur Decomposition Example");
    println!("==========================\n");

    // Test with a simple 2x2 matrix
    println!("1. Simple 2x2 Matrix:");
    let a = array![[1.0, 2.0], [3.0, 4.0]];
    println!("Matrix A = \n{:8.6}", a);

    let (z, t) = compat::schur(&a.view(), "real", None, false, None, true)?;
    println!("Orthogonal matrix Z = \n{:8.6}", z);
    println!("Upper triangular T = \n{:8.6}", t);

    // Verify the decomposition: A = Z * T * Z^T
    let reconstructed = z.dot(&t).dot(&z.t());
    println!("Reconstructed A = Z * T * Z^T = \n{:8.6}", reconstructed);

    let diff = &a - &reconstructed;
    let max_error = diff.iter().map(|&x: &f64| x.abs()).fold(0.0, f64::max);
    println!("Reconstruction error: {:.2e}\n", max_error);

    // Verify Z is orthogonal: Z * Z^T = I
    let ztzi = z.dot(&z.t());
    let identity = ndarray::Array2::<f64>::eye(2);
    println!("Z * Z^T = \n{:8.6}", ztzi);
    let ortho_error = (&ztzi - &identity)
        .iter()
        .map(|&x: &f64| x.abs())
        .fold(0.0, f64::max);
    println!("Orthogonality error: {:.2e}\n", ortho_error);

    // Test with a symmetric matrix
    println!("2. Symmetric Matrix:");
    let symmetric = array![[2.0, 1.0], [1.0, 3.0]];
    println!("Symmetric matrix A = \n{:8.6}", symmetric);

    let (z_sym, t_sym): (Array2<f64>, Array2<f64>) =
        compat::schur(&symmetric.view(), "real", None, false, None, true)?;
    println!("Z = \n{:8.6}", z_sym);
    println!("T = \n{:8.6}", t_sym);

    // For symmetric matrices, T should be diagonal (eigenvalues on diagonal)
    println!(
        "Diagonal elements of T (eigenvalues): [{:.6}, {:.6}]",
        t_sym[[0, 0]],
        t_sym[[1, 1]]
    );

    // Verify off-diagonal elements are small for symmetric case
    let off_diag_error: f64 = t_sym[[0, 1]].abs() + t_sym[[1, 0]].abs();
    println!("Off-diagonal error in T: {:.2e}\n", off_diag_error);

    // Test with a 3x3 matrix
    println!("3. Larger 3x3 Matrix:");
    let large = array![[1.0, 2.0, 3.0], [0.0, 4.0, 5.0], [0.0, 0.0, 6.0]];
    println!("Upper triangular matrix A = \n{:8.6}", large);

    let (z_large, t_large) = compat::schur(&large.view(), "real", None, false, None, true)?;
    println!("Z = \n{:8.6}", z_large);
    println!("T = \n{:8.6}", t_large);

    // Verify reconstruction
    let reconstructed_large = z_large.dot(&t_large).dot(&z_large.t());
    let large_diff = &large - &reconstructed_large;
    let large_error = large_diff
        .iter()
        .map(|&x: &f64| x.abs())
        .fold(0.0, f64::max);
    println!("3x3 reconstruction error: {:.2e}\n", large_error);

    // Test using SciPy-compatible interface
    println!("4. SciPy-Compatible Interface:");
    let scipy_matrix = array![[2.0, -1.0], [1.0, 0.0]];
    println!("Matrix A = \n{:8.6}", scipy_matrix);

    let (z_scipy, t_scipy) = compat::schur(&scipy_matrix.view(), "real", None, false, None, true)?;
    println!("Z (via SciPy interface) = \n{:8.6}", z_scipy);
    println!("T (via SciPy interface) = \n{:8.6}", t_scipy);

    // Verify the decomposition
    let scipy_reconstructed = z_scipy.dot(&t_scipy).dot(&z_scipy.t());
    let scipy_diff = &scipy_matrix - &scipy_reconstructed;
    let scipy_error = scipy_diff
        .iter()
        .map(|&x: &f64| x.abs())
        .fold(0.0, f64::max);
    println!(
        "SciPy interface reconstruction error: {:.2e}\n",
        scipy_error
    );

    // Demonstrate eigenvalue extraction from Schur form
    println!("5. Eigenvalue Extraction:");
    let eig_matrix = array![[5.0, 1.0], [0.0, 3.0]];
    println!("Matrix A = \n{:8.6}", eig_matrix);

    let (_, t_eig) = compat::schur(&eig_matrix.view(), "real", None, false, None, true)?;
    println!("Schur form T = \n{:8.6}", t_eig);

    // For upper triangular matrices, eigenvalues are the diagonal elements
    println!(
        "Eigenvalues from diagonal of T: [{:.6}, {:.6}]",
        t_eig[[0, 0]],
        t_eig[[1, 1]]
    );

    // Verify these are indeed eigenvalues by checking det(A - λI) = 0
    let lambda1 = t_eig[[0, 0]];
    let lambda2 = t_eig[[1, 1]];

    let a_minus_lambda1_i = &eig_matrix - &(ndarray::Array2::<f64>::eye(2) * lambda1);
    let a_minus_lambda2_i = &eig_matrix - &(ndarray::Array2::<f64>::eye(2) * lambda2);

    // Calculate determinants (should be close to zero)
    let det1 = a_minus_lambda1_i[[0, 0]] * a_minus_lambda1_i[[1, 1]]
        - a_minus_lambda1_i[[0, 1]] * a_minus_lambda1_i[[1, 0]];
    let det2 = a_minus_lambda2_i[[0, 0]] * a_minus_lambda2_i[[1, 1]]
        - a_minus_lambda2_i[[0, 1]] * a_minus_lambda2_i[[1, 0]];

    println!("det(A - λ₁I) = {:.2e}", det1);
    println!("det(A - λ₂I) = {:.2e}", det2);

    println!("\nSchur decomposition completed successfully!");
    println!("The Schur decomposition is useful for:");
    println!("- Computing eigenvalues efficiently");
    println!("- Matrix function computations");
    println!("- Solving matrix equations");
    println!("- Numerical analysis of dynamical systems");

    Ok(())
}
