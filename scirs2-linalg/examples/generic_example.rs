//! Example demonstrating type-generic linear algebra operations

use ndarray::array;
use scirs2_linalg::generic::PrecisionSelector;
use scirs2_linalg::prelude::*;

fn main() {
    println!("Type-generic linear algebra operations\n");

    // Example with f64
    demo_f64();

    // Example with f32
    demo_f32();

    // Example of automatic precision selection
    demo_precision_selection();
}

fn demo_f64() {
    println!("=== f64 operations ===");

    let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
    let b = array![[5.0, 6.0], [7.0, 8.0]];

    // Matrix multiplication
    let c = gemm(&a.view(), &b.view()).unwrap();
    println!("A * B = \n{}", c);

    // Determinant
    let det = gdet(&a.view()).unwrap();
    println!("\ndet(A) = {}", det);

    // Inverse
    let a_inv = ginv(&a.view()).unwrap();
    println!("\nA^(-1) = \n{}", a_inv);

    // SVD
    let svd = gsvd(&a.view(), false).unwrap();
    println!("\nSVD singular values: {}", svd.s);

    // QR decomposition
    let qr = gqr(&a.view()).unwrap();
    println!("\nQR decomposition Q = \n{}", qr.q);

    // Eigendecomposition
    let eigen = geig(&a.view()).unwrap();
    println!("\nEigenvalues: {:?}", eigen.eigenvalues);

    // Linear solve
    let b_vec = array![[5.0], [11.0]];
    let x = gsolve(&a.view(), &b_vec.view()).unwrap();
    println!("\nSolution to Ax = b: \n{}", x);
}

fn demo_f32() {
    println!("\n=== f32 operations ===");

    let a = array![[1.0_f32, 2.0], [3.0, 4.0]];
    let b = array![[5.0, 6.0], [7.0, 8.0]];

    // Matrix multiplication
    let c = gemm(&a.view(), &b.view()).unwrap();
    println!("A * B = \n{}", c);

    // Determinant
    let det = gdet(&a.view()).unwrap();
    println!("\ndet(A) = {}", det);
}

fn demo_precision_selection() {
    println!("\n=== Automatic Precision Selection ===");

    // For well-conditioned matrices
    let good_condition = 100.0;
    let use_high_precision = f32::should_use_high_precision(good_condition);
    println!(
        "Condition number {}: Use high precision? {}",
        good_condition, use_high_precision
    );

    // For ill-conditioned matrices
    let bad_condition = 1e7;
    let use_high_precision = f32::should_use_high_precision(bad_condition);
    println!(
        "Condition number {}: Use high precision? {}",
        bad_condition, use_high_precision
    );
}
