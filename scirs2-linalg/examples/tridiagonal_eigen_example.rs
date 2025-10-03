//! Tridiagonal eigenvalue solver example
//!
//! This example demonstrates the specialized eigenvalue solver for tridiagonal matrices.

use scirs2_core::ndarray::array;
use scirs2_linalg::error::LinalgResult;
use scirs2_linalg::{tridiagonal_eigh, tridiagonal_eigvalsh};

#[allow(dead_code)]
fn main() -> LinalgResult<()> {
    println!("Tridiagonal Eigenvalue Solver Example");
    println!("====================================\n");

    // Create diagonal and off-diagonal elements for a tridiagonal matrix
    // This represents the matrix:
    // [2 1 0 0]
    // [1 3 1 0]
    // [0 1 4 1]
    // [0 0 1 5]
    let diagonal = array![2.0, 3.0, 4.0, 5.0];
    let off_diagonal = array![1.0, 1.0, 1.0];

    println!("Tridiagonal Matrix:");
    println!("Main diagonal: {:?}", diagonal);
    println!("Off diagonal: {:?}\n", off_diagonal);

    // Compute eigenvalues only
    let eigenvalues = tridiagonal_eigvalsh(&diagonal.view(), &off_diagonal.view())?;
    println!("Eigenvalues: {:?}\n", eigenvalues);

    // Compute both eigenvalues and eigenvectors
    let (evals, evecs) = tridiagonal_eigh(&diagonal.view(), &off_diagonal.view())?;
    println!("Eigenvalues from full decomposition: {:?}", evals);
    println!("Eigenvectors:");
    for i in 0..evecs.ncols() {
        println!("v{}: {:?}", i, evecs.column(i));
    }

    // Verify orthogonality of eigenvectors
    println!("\nVerifying orthogonality of eigenvectors:");
    for i in 0..evecs.ncols() {
        for j in i..evecs.ncols() {
            let dot_product = evecs.column(i).dot(&evecs.column(j));
            println!("v{} · v{} = {:.10}", i, j, dot_product);
        }
    }

    Ok(())
}
