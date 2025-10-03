//! Basic Linear Algebra Tutorial
//!
//! This tutorial covers fundamental operations in scirs2-linalg.

use scirs2_core::ndarray::array;
use scirs2_linalg::prelude::*;
use scirs2_linalg::LinalgError;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Basic Linear Algebra Tutorial ===\n");

    // 1. Matrix Creation
    println!("1. Matrix Creation");
    println!("-----------------");

    let a = array![[3.0, 1.0], [1.0, 2.0]];
    println!("Matrix A:");
    println!("{}\n", a);

    let b = array![[1.0, 0.0], [0.0, 1.0]];
    println!("Matrix B (identity):");
    println!("{}\n", b);

    // 2. Basic Operations
    println!("2. Basic Operations");
    println!("------------------");

    // Determinant
    let det_a = det(&a.view(), None)?;
    println!("det(A) = {}", det_a);

    // Matrix inverse (only for 2x2 currently)
    let inv_a = inv(&a.view(), None)?;
    println!("inv(A) =");
    println!("{}", inv_a);

    // Verify A * A^(-1) = I
    let should_be_identity = a.dot(&inv_a);
    println!("A * inv(A) =");
    println!("{}\n", should_be_identity);

    // 3. Solving Linear Systems
    println!("3. Solving Linear Systems");
    println!("------------------------");

    // Solve Ax = b
    let b_vec = array![9.0, 8.0];
    println!("Solving Ax = b where b = {:?}", b_vec);

    let x = solve(&a.view(), &b_vec.view(), None)?;
    println!("Solution x = {:?}", x);

    // Verify solution
    let ax = a.dot(&x);
    println!("Verification: Ax = {:?}\n", ax);

    // 4. Matrix Decompositions
    println!("4. Matrix Decompositions");
    println!("-----------------------");

    // LU Decomposition
    let (p, l, u) = lu(&a.view(), None)?;
    println!("LU Decomposition:");
    println!("P =\n{}", p);
    println!("L =\n{}", l);
    println!("U =\n{}", u);

    // Verify P*A = L*U
    let pa = p.dot(&a);
    let lu_product = l.dot(&u);
    println!(
        "Verification: P*A == L*U? {}\n",
        pa.abs_diff_eq(&lu_product, 1e-10)
    );

    // QR Decomposition
    let (q, r) = qr(&a.view(), None)?;
    println!("QR Decomposition:");
    println!("Q =\n{}", q);
    println!("R =\n{}", r);

    // Verify A = Q*R
    let qr_product = q.dot(&r);
    println!(
        "Verification: A == Q*R? {}\n",
        a.abs_diff_eq(&qr_product, 1e-10)
    );

    // 5. Eigenvalues and Eigenvectors
    println!("5. Eigenvalues and Eigenvectors");
    println!("------------------------------");

    // For symmetric matrices
    let symmetric = array![[4.0, 1.0], [1.0, 3.0]];
    println!("Symmetric matrix:");
    println!("{}", symmetric);

    let (eigenvalues, eigenvectors) = eigh(&symmetric.view(), None)?;
    println!("Eigenvalues: {:?}", eigenvalues);
    println!("Eigenvectors:");
    println!("{}\n", eigenvectors);

    // 6. Matrix Norms
    println!("6. Matrix Norms");
    println!("--------------");

    let norm_fro = matrix_norm(&a.view(), "fro", None)?;
    println!("Frobenius norm of A: {}", norm_fro);

    let norm_1 = matrix_norm(&a.view(), "1", None)?;
    println!("1-norm of A: {}", norm_1);

    let norm_inf = matrix_norm(&a.view(), "inf", None)?;
    println!("Infinity norm of A: {}\n", norm_inf);

    // 7. Error Handling
    println!("7. Error Handling");
    println!("----------------");

    // Trying to invert a singular matrix
    let singular = array![[1.0, 2.0], [2.0, 4.0]];
    println!("Attempting to invert singular matrix:");
    println!("{}", singular);

    match inv(&singular.view(), None) {
        Ok(_) => println!("Unexpected success!"),
        Err(LinalgError::SingularMatrixError(msg)) => {
            println!("Expected error: {}", msg);
        }
        Err(e) => println!("Unexpected error: {}", e),
    }

    println!("\n=== Tutorial Complete ===");
    Ok(())
}
