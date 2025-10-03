//! Example demonstrating the use of specialized matrix implementations
//!
//! This example shows how to create and use the specialized matrix implementations:
//! - Tridiagonal matrices
//! - Banded matrices
//! - Symmetric matrices
//!
//! These implementations provide efficient storage and operations for special matrix structures.

use scirs2_core::ndarray::{array, Array2};
use scirs2_linalg::{
    specialized::{BandedMatrix, SpecializedMatrix, SymmetricMatrix, TridiagonalMatrix},
    LinalgResult,
};

#[allow(dead_code)]
fn main() -> LinalgResult<()> {
    println!("=== Specialized Matrices Example ===");

    // Example 1: Tridiagonal Matrix
    println!("\n== Tridiagonal Matrix ==\n");

    // Create a 5x5 tridiagonal matrix
    let diag = array![2.0, 2.0, 2.0, 2.0, 2.0]; // Main diagonal
    let superdiag = array![-1.0, -1.0, -1.0, -1.0]; // Superdiagonal
    let subdiag = array![-1.0, -1.0, -1.0, -1.0]; // Subdiagonal

    let tri = TridiagonalMatrix::new(diag.view(), superdiag.view(), subdiag.view())?;

    // Print the matrix
    println!("Tridiagonal matrix:");
    let dense = tri.to_dense()?;
    printmatrix(&dense);

    // Perform matrix-vector multiplication
    let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
    println!("\nVector x: {:?}", x);

    let y = tri.matvec(&x.view())?;
    println!("Tridiagonal * x = {:?}", y);

    // Solve a tridiagonal system
    let b = array![5.0, 10.0, 15.0, 20.0, 25.0];
    println!("\nSolving Tridiagonal * x = b where b = {:?}", b);

    let solution = tri.solve(&b.view())?;
    println!("Solution x = {:?}", solution);

    // Verify solution
    let verify = tri.matvec(&solution.view())?;
    println!("Verification: Tridiagonal * x = {:?}", verify);

    // Example 2: Banded Matrix
    println!("\n== Banded Matrix ==\n");

    // Create a 5x5 banded matrix with lower bandwidth 1 and upper bandwidth 2
    let mut band_data = Array2::zeros((4, 5)); // 4 diagonals (1+1+2), 5 columns

    // Set the values for each diagonal
    // Lower diagonal (the 1 subdiagonal)
    band_data[[0, 0]] = 1.0;
    band_data[[0, 1]] = 2.0;
    band_data[[0, 2]] = 3.0;
    band_data[[0, 3]] = 4.0;

    // Main diagonal
    band_data[[1, 0]] = 5.0;
    band_data[[1, 1]] = 6.0;
    band_data[[1, 2]] = 7.0;
    band_data[[1, 3]] = 8.0;
    band_data[[1, 4]] = 9.0;

    // First superdiagonal
    band_data[[2, 0]] = 10.0;
    band_data[[2, 1]] = 11.0;
    band_data[[2, 2]] = 12.0;
    band_data[[2, 3]] = 13.0;

    // Second superdiagonal
    band_data[[3, 0]] = 14.0;
    band_data[[3, 1]] = 15.0;
    band_data[[3, 2]] = 16.0;

    let band = BandedMatrix::new(band_data.view(), 1, 2, 5, 5)?;

    // Print the matrix
    println!("Banded matrix:");
    let dense = band.to_dense()?;
    printmatrix(&dense);

    // Perform matrix-vector multiplication
    let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
    println!("\nVector x: {:?}", x);

    let y = band.matvec(&x.view())?;
    println!("Banded * x = {:?}", y);

    // Example 3: Symmetric Matrix
    println!("\n== Symmetric Matrix ==\n");

    // Create a 4x4 symmetric matrix
    let sym_data = array![
        [2.0, -1.0, 0.0, 0.0],
        [-1.0, 2.0, -1.0, 0.0],
        [0.0, -1.0, 2.0, -1.0],
        [0.0, 0.0, -1.0, 2.0]
    ];

    let sym = SymmetricMatrix::frommatrix(&sym_data.view())?;

    // Print the matrix
    println!("Symmetric matrix:");
    let dense = sym.to_dense()?;
    printmatrix(&dense);

    // Perform matrix-vector multiplication
    let x = array![1.0, 2.0, 3.0, 4.0];
    println!("\nVector x: {:?}", x);

    let y = sym.matvec(&x.view())?;
    println!("Symmetric * x = {:?}", y);

    // Cholesky decomposition
    println!("\nCholesky decomposition:");
    let chol = sym.cholesky()?;
    printmatrix(&chol);

    // Solve a symmetric system
    let b = array![4.0, 0.0, 0.0, 4.0];
    println!("\nSolving Symmetric * x = b where b = {:?}", b);

    let solution = sym.solve(&b.view())?;
    println!("Solution x = {:?}", solution);

    // Verify solution
    let verify = sym.matvec(&solution.view())?;
    println!("Verification: Symmetric * x = {:?}", verify);

    // We'll skip the matrix-free operator example for now, due to issues with
    // the tensor_contraction module that we want to avoid

    Ok(())
}

// Helper function to print a matrix
#[allow(dead_code)]
fn printmatrix<T: std::fmt::Display>(matrix: &Array2<T>) {
    for i in 0..matrix.nrows() {
        print!("[");
        for j in 0..matrix.ncols() {
            print!("{:6.2}", matrix[[i, j]]);
            if j < matrix.ncols() - 1 {
                print!(", ");
            }
        }
        println!(" ]");
    }
}
