//! Example demonstrating complex matrix decompositions
//!
//! This example shows how to use various complex matrix decompositions
//! including LU, QR, SVD, Cholesky, and eigenvalue decompositions.

use scirs2_core::ndarray::{array, Array2};
use scirs2_core::Complex;
use scirs2_linalg::complex::*;
use scirs2_linalg::LinalgResult;

type C64 = Complex<f64>;

#[allow(dead_code)]
fn main() -> LinalgResult<()> {
    println!("Complex Matrix Decompositions Example");
    println!("====================================\n");

    // Complex LU decomposition
    demo_complex_lu()?;

    // Complex QR decomposition
    demo_complex_qr()?;

    // Complex SVD
    demo_complex_svd()?;

    // Complex Cholesky decomposition
    demo_complex_cholesky()?;

    // Complex eigenvalue decomposition
    demo_complex_eig()?;

    // Hermitian eigenvalue decomposition
    demo_complex_eigh()?;

    Ok(())
}

#[allow(dead_code)]
fn demo_complex_lu() -> LinalgResult<()> {
    println!("Complex LU Decomposition");
    println!("----------------------");

    let a = array![
        [C64::new(4.0, 2.0), C64::new(1.0, 1.0), C64::new(2.0, -1.0)],
        [C64::new(1.0, -1.0), C64::new(3.0, 1.0), C64::new(1.0, 0.0)],
        [C64::new(2.0, 1.0), C64::new(0.0, 2.0), C64::new(5.0, -1.0)]
    ];

    println!("Matrix A:");
    print_complexmatrix(&a);

    let lu_result = complex_lu(&a.view())?;

    // Extract L and U
    let n = a.nrows();
    let mut l = Array2::eye(n);
    let mut u = Array2::zeros((n, n));

    for i in 0..n {
        for j in 0..n {
            if i > j {
                l[[i, j]] = lu_result.lu[[i, j]];
            } else {
                u[[i, j]] = lu_result.lu[[i, j]];
            }
        }
    }

    println!("\nLower triangular L:");
    print_complexmatrix(&l);

    println!("\nUpper triangular U:");
    print_complexmatrix(&u);

    // Verify: L * U = P * A
    let lu_product = complex_matmul(&l.view(), &u.view())?;
    println!("\nL * U:");
    print_complexmatrix(&lu_product);

    println!();
    Ok(())
}

#[allow(dead_code)]
fn demo_complex_qr() -> LinalgResult<()> {
    println!("Complex QR Decomposition");
    println!("----------------------");

    let a = array![
        [C64::new(1.0, 0.0), C64::new(1.0, 1.0), C64::new(0.0, 1.0)],
        [C64::new(0.0, 1.0), C64::new(1.0, 0.0), C64::new(1.0, -1.0)],
        [C64::new(1.0, 1.0), C64::new(0.0, -1.0), C64::new(2.0, 0.0)]
    ];

    println!("Matrix A:");
    print_complexmatrix(&a);

    let qr_result = complex_qr(&a.view())?;

    println!("\nUnitary matrix Q:");
    print_complexmatrix(&qr_result.q);

    println!("\nUpper triangular R:");
    print_complexmatrix(&qr_result.r);

    // Verify: Q * R = A
    let qr_product = complex_matmul(&qr_result.q.view(), &qr_result.r.view())?;
    println!("\nQ * R:");
    print_complexmatrix(&qr_product);

    // Verify Q is unitary: Q^H * Q = I
    let qh = hermitian_transpose(&qr_result.q.view());
    let qhq = complex_matmul(&qh.view(), &qr_result.q.view())?;
    println!("\nQ^H * Q (should be identity):");
    print_complexmatrix(&qhq);

    println!();
    Ok(())
}

#[allow(dead_code)]
fn demo_complex_svd() -> LinalgResult<()> {
    println!("Complex SVD");
    println!("----------");

    let a = array![
        [C64::new(1.0, 0.0), C64::new(0.0, 1.0)],
        [C64::new(0.0, -1.0), C64::new(2.0, 0.0)],
        [C64::new(1.0, 1.0), C64::new(0.0, 0.0)]
    ];

    println!("Matrix A (3x2):");
    print_complexmatrix(&a);

    let svd_result = complex_svd(&a.view(), false)?;

    println!("\nLeft singular vectors U:");
    print_complexmatrix(&svd_result.u);

    println!("\nSingular values S:");
    println!("{:?}", svd_result.s);

    println!("\nRight singular vectors V^H:");
    print_complexmatrix(&svd_result.vh);

    // Reconstruct: U * S * V^H
    // The correct shape for S should be (U.ncols, Vh.nrows) for multiplication
    let u_cols = svd_result.u.ncols();
    let vh_rows = svd_result.vh.nrows();
    let mut smatrix = Array2::zeros((u_cols, vh_rows));

    // Fill diagonal with singular values
    for i in 0..svd_result.s.len().min(u_cols).min(vh_rows) {
        smatrix[[i, i]] = C64::new(svd_result.s[i], 0.0);
    }

    let us = complex_matmul(&svd_result.u.view(), &smatrix.view())?;
    let reconstructed = complex_matmul(&us.view(), &svd_result.vh.view())?;

    println!("\nReconstructed U * S * V^H:");
    print_complexmatrix(&reconstructed);

    println!();
    Ok(())
}

#[allow(dead_code)]
fn demo_complex_cholesky() -> LinalgResult<()> {
    println!("Complex Cholesky Decomposition");
    println!("----------------------------");

    // Create a positive-definite Hermitian matrix
    // A = B * B^H + I for some B
    let b = array![
        [C64::new(2.0, 0.0), C64::new(1.0, 1.0)],
        [C64::new(0.0, 1.0), C64::new(2.0, -1.0)]
    ];

    let bh = hermitian_transpose(&b.view());
    let mut a = complex_matmul(&b.view(), &bh.view())?;

    // Add identity to ensure positive definiteness
    for i in 0..a.nrows() {
        a[[i, i]] += C64::new(1.0, 0.0);
    }

    println!("Positive-definite Hermitian matrix A:");
    print_complexmatrix(&a);

    let l = complex_cholesky(&a.view())?;

    println!("\nLower triangular L:");
    print_complexmatrix(&l);

    // Verify: L * L^H = A
    let lh = hermitian_transpose(&l.view());
    let llh = complex_matmul(&l.view(), &lh.view())?;

    println!("\nL * L^H:");
    print_complexmatrix(&llh);

    println!();
    Ok(())
}

#[allow(dead_code)]
fn demo_complex_eig() -> LinalgResult<()> {
    println!("Complex Eigenvalue Decomposition");
    println!("-------------------------------");

    // Note: Full complex eigendecomposition is not yet fully implemented
    // This is a demonstration of the interface

    let a = array![
        [C64::new(3.0, 0.0), C64::new(1.0, 1.0)],
        [C64::new(1.0, -1.0), C64::new(2.0, 0.0)]
    ];

    println!("Matrix A:");
    print_complexmatrix(&a);

    match complex_eig(&a.view()) {
        Ok(eig_result) => {
            println!("\nEigenvalues:");
            for (i, eigenvalue) in eig_result.eigenvalues.iter().enumerate() {
                println!("λ{} = {}", i + 1, eigenvalue);
            }

            println!("\nEigenvectors:");
            print_complexmatrix(&eig_result.eigenvectors);
        }
        Err(e) => {
            println!("\nComplex eigendecomposition not fully implemented: {}", e);
        }
    }

    println!();
    Ok(())
}

#[allow(dead_code)]
fn demo_complex_eigh() -> LinalgResult<()> {
    println!("Hermitian Eigenvalue Decomposition");
    println!("---------------------------------");

    // Create a Hermitian matrix
    let a = array![
        [C64::new(3.0, 0.0), C64::new(1.0, 1.0), C64::new(0.0, -2.0)],
        [C64::new(1.0, -1.0), C64::new(2.0, 0.0), C64::new(1.0, 1.0)],
        [C64::new(0.0, 2.0), C64::new(1.0, -1.0), C64::new(4.0, 0.0)]
    ];

    println!("Hermitian matrix A:");
    print_complexmatrix(&a);

    // Verify it's Hermitian
    let ah = hermitian_transpose(&a.view());
    let is_hermitian = a
        .iter()
        .zip(ah.iter())
        .all(|(a, ah)| (a - ah).norm() < 1e-10);
    println!("\nMatrix is Hermitian: {}", is_hermitian);

    match complex_eigh(&a.view()) {
        Ok(eig_result) => {
            println!("\nEigenvalues (should all be real):");
            for (i, eigenvalue) in eig_result.eigenvalues.iter().enumerate() {
                println!("λ{} = {} + {}i", i + 1, eigenvalue.re, eigenvalue.im);
            }

            println!("\nEigenvectors:");
            print_complexmatrix(&eig_result.eigenvectors);

            // Verify eigenvectors are orthogonal
            let vh = hermitian_transpose(&eig_result.eigenvectors.view());
            let vhv = complex_matmul(&vh.view(), &eig_result.eigenvectors.view())?;
            println!("\nV^H * V (should be identity):");
            print_complexmatrix(&vhv);

            // --- Hermitian eigenvalue decomposition check: A ≈ QΛQ^H ---
            use scirs2_core::ndarray::Array2;
            let recon = eig_result
                .eigenvectors
                .dot(&Array2::from_diag(&eig_result.eigenvalues))
                .dot(&hermitian_transpose(&eig_result.eigenvectors.view()));
            let mut max_diff = 0.0;
            for i in 0..a.nrows() {
                for j in 0..a.ncols() {
                    let diff = (a[[i, j]] - recon[[i, j]]).norm();
                    if diff > max_diff {
                        max_diff = diff;
                    }
                }
            }
            println!("\nA ≈ QΛQ^H max abs diff: {:.3e}", max_diff);
            if max_diff > 1e-8 {
                println!("WARNING: A and QΛQ^H differ by more than 1e-8");
                println!("A:");
                print_complexmatrix(&a);
                println!("QΛQ^H:");
                print_complexmatrix(&recon);
            }
        }
        Err(e) => {
            println!("\nHermitian eigendecomposition error: {}", e);
        }
    }

    println!();
    Ok(())
}

#[allow(dead_code)]
fn print_complexmatrix(a: &Array2<C64>) {
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            let c = a[[i, j]];
            print!("{:6.2} + {:5.2}i  ", c.re, c.im);
        }
        println!();
    }
}
