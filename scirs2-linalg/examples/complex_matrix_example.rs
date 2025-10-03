//! Complex matrix operations example
//!
//! This example demonstrates the use of complex matrix operations.

use scirs2_core::ndarray::array;
use scirs2_core::Complex;
use scirs2_linalg::complex::{complex_matmul, complex_norm_frobenius, hermitian_transpose};
use scirs2_linalg::error::LinalgResult;

#[allow(dead_code)]
fn main() -> LinalgResult<()> {
    println!("Complex Matrix Operations Examples");
    println!("==================================\n");

    // Create a complex matrix
    let a = array![
        [Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)],
        [Complex::new(5.0, 6.0), Complex::new(7.0, 8.0)]
    ];

    println!("Matrix A:\n{:?}\n", a);

    // Compute Hermitian transpose
    let a_h = hermitian_transpose(&a.view());
    println!("Hermitian transpose of A:\n{:?}\n", a_h);

    // Matrix multiplication
    let a_a_h = complex_matmul(&a.view(), &a_h.view())?;
    println!("A * A^H:\n{:?}\n", a_a_h);

    // Frobenius norm
    let norm = complex_norm_frobenius(&a.view());
    println!("Frobenius norm of A: {:.6}\n", norm);

    // Matrix inverse
    let a_inv = scirs2_linalg::complex::complex_inverse(&a.view())?;
    println!("Inverse of A:\n{:?}\n", a_inv);

    // Verify inverse: A * A^(-1) should be identity
    let identity = complex_matmul(&a.view(), &a_inv.view())?;
    println!("A * A^(-1) (should be close to identity):\n{:?}", identity);

    Ok(())
}
