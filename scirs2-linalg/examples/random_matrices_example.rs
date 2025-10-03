//! Example demonstrating random matrix generation utilities

use scirs2_core::random::prelude::*;
use scirs2_core::ChaCha8Rng;
use scirs2_linalg::error::LinalgResult;
use scirs2_linalg::random_matrices::{
    random_complexmatrix, random_hermitian, randommatrix, Distribution1D, MatrixType,
};

#[allow(dead_code)]
fn main() -> LinalgResult<()> {
    println!("SciRS2 Random Matrix Generation Examples");
    println!("======================================\n");

    // Use a deterministic RNG for reproducible examples
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    // Example 1: General random matrix
    demo_generalmatrix(&mut rng)?;

    // Example 2: Symmetric matrix
    demo_symmetricmatrix(&mut rng)?;

    // Example 3: Positive definite matrix
    demo_positive_definite(&mut rng)?;

    // Example 4: Orthogonal matrix
    demo_orthogonalmatrix(&mut rng)?;

    // Example 5: Correlation matrix
    demo_correlationmatrix(&mut rng)?;

    // Example 6: Sparse matrix
    demo_sparsematrix(&mut rng)?;

    // Example 7: Complex matrices
    demo_complex_matrices(&mut rng)?;

    Ok(())
}

#[allow(dead_code)]
fn demo_generalmatrix<R: scirs2_core::random::Rng>(rng: &mut R) -> LinalgResult<()> {
    println!("1. General Random Matrix");
    println!("----------------------");

    // Uniform distribution
    let uniformmatrix = randommatrix(
        3,
        4,
        MatrixType::General(Distribution1D::Uniform { a: -1.0, b: 1.0 }),
        rng,
    )?;
    println!("Uniform[-1, 1] (3x4):");
    printmatrix(&uniformmatrix);

    // Standard normal distribution
    let normalmatrix = randommatrix(
        4,
        3,
        MatrixType::General(Distribution1D::StandardNormal),
        rng,
    )?;
    println!("\nStandard Normal (4x3):");
    printmatrix(&normalmatrix);

    println!();
    Ok(())
}

#[allow(dead_code)]
fn demo_symmetricmatrix<R: scirs2_core::random::Rng>(rng: &mut R) -> LinalgResult<()> {
    println!("2. Symmetric Random Matrix");
    println!("------------------------");

    let symmatrix = randommatrix(
        4,
        4,
        MatrixType::Symmetric(Distribution1D::Normal {
            mean: 0.0,
            std_dev: 2.0,
        }),
        rng,
    )?;

    println!("Symmetric matrix (4x4):");
    printmatrix(&symmatrix);

    // Verify symmetry
    let mut max_diff: f64 = 0.0;
    for i in 0..4 {
        for j in i + 1..4 {
            let diff = (symmatrix[[i, j]] - symmatrix[[j, i]]).abs();
            max_diff = max_diff.max(diff);
        }
    }
    println!("Maximum asymmetry: {:.2e}", max_diff);

    println!();
    Ok(())
}

#[allow(dead_code)]
fn demo_positive_definite<R: scirs2_core::random::Rng>(rng: &mut R) -> LinalgResult<()> {
    println!("3. Positive Definite Matrix");
    println!("--------------------------");

    let pdmatrix = randommatrix(
        3,
        3,
        MatrixType::PositiveDefinite {
            eigenvalue_min: 0.5,
            eigenvalue_max: 5.0,
        },
        rng,
    )?;

    println!("Positive definite matrix (3x3):");
    printmatrix(&pdmatrix);

    // Test Cholesky decomposition (only works for positive definite)
    use scirs2_linalg::cholesky;
    match cholesky(&pdmatrix.view(), None) {
        Ok(_) => println!("✓ Cholesky decomposition successful (matrix is positive definite)"),
        Err(_) => println!("✗ Cholesky decomposition failed"),
    }

    println!();
    Ok(())
}

#[allow(dead_code)]
fn demo_orthogonalmatrix<R: scirs2_core::random::Rng>(rng: &mut R) -> LinalgResult<()> {
    println!("4. Orthogonal Matrix");
    println!("-------------------");

    let orthomatrix = randommatrix(3, 3, MatrixType::Orthogonal, rng)?;

    println!("Orthogonal matrix Q (3x3):");
    printmatrix(&orthomatrix);

    // Verify Q^T * Q = I
    let qt = orthomatrix.t();
    let qtq = qt.dot(&orthomatrix);

    println!("\nQ^T * Q (should be identity):");
    printmatrix(&qtq);

    // Check orthogonality error
    let mut max_error: f64 = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { 1.0 } else { 0.0 };
            let error = (qtq[[i, j]] - expected).abs();
            max_error = max_error.max(error);
        }
    }
    println!("Maximum deviation from identity: {:.2e}", max_error);

    println!();
    Ok(())
}

#[allow(dead_code)]
fn demo_correlationmatrix<R: scirs2_core::random::Rng>(rng: &mut R) -> LinalgResult<()> {
    println!("5. Correlation Matrix");
    println!("--------------------");

    let corrmatrix = randommatrix(5, 5, MatrixType::Correlation, rng)?;

    println!("Correlation matrix (5x5):");
    printmatrix(&corrmatrix);

    // Verify properties
    println!("\nVerifying correlation matrix properties:");

    // Check diagonal = 1
    let mut diag_ok = true;
    for i in 0..5 {
        if (corrmatrix[[i, i]] - 1.0).abs() > 1e-10 {
            diag_ok = false;
            break;
        }
    }
    println!("✓ Diagonal elements = 1: {}", diag_ok);

    // Check symmetry
    let mut sym_ok = true;
    for i in 0..5 {
        for j in i + 1..5 {
            if (corrmatrix[[i, j]] - corrmatrix[[j, i]]).abs() > 1e-10 {
                sym_ok = false;
                break;
            }
        }
    }
    println!("✓ Symmetric: {}", sym_ok);

    // Check values in [-1, 1]
    let mut range_ok = true;
    for &val in corrmatrix.iter() {
        if !(-1.0..=1.0).contains(&val) {
            range_ok = false;
            break;
        }
    }
    println!("✓ All values in [-1, 1]: {}", range_ok);

    println!();
    Ok(())
}

#[allow(dead_code)]
fn demo_sparsematrix<R: scirs2_core::random::Rng>(rng: &mut R) -> LinalgResult<()> {
    println!("6. Sparse Matrix");
    println!("---------------");

    let sparsematrix = randommatrix(
        6,
        8,
        MatrixType::Sparse {
            density: 0.2,
            distribution: Distribution1D::Normal {
                mean: 0.0,
                std_dev: 1.0,
            },
        },
        rng,
    )?;

    println!("Sparse matrix (6x8, density=0.2):");
    print_sparsematrix(&sparsematrix);

    // Count non-zero elements
    let nnz = sparsematrix.iter().filter(|&&x| x.abs() > 1e-10).count();
    let total = sparsematrix.nrows() * sparsematrix.ncols();
    let actual_density = nnz as f64 / total as f64;

    println!(
        "\nNon-zero elements: {} / {} (density: {:.3})",
        nnz, total, actual_density
    );

    println!();
    Ok(())
}

#[allow(dead_code)]
fn demo_complex_matrices<R: scirs2_core::random::Rng>(rng: &mut R) -> LinalgResult<()> {
    println!("7. Complex Matrices");
    println!("------------------");

    // General complex matrix
    let complexmatrix = random_complexmatrix(
        3,
        3,
        Distribution1D::StandardNormal,
        Distribution1D::StandardNormal,
        rng,
    )?;

    println!("Complex matrix (3x3):");
    print_complexmatrix(&complexmatrix);

    // Hermitian matrix
    let hermitian = random_hermitian(
        4,
        Distribution1D::StandardNormal,
        Distribution1D::Uniform { a: -0.5, b: 0.5 },
        rng,
    )?;

    println!("\nHermitian matrix (4x4):");
    print_complexmatrix(&hermitian);

    // Verify Hermitian property
    let mut max_error: f64 = 0.0;
    for i in 0..4 {
        for j in 0..4 {
            let diff = (hermitian[[i, j]] - hermitian[[j, i]].conj()).norm();
            max_error = max_error.max(diff);
        }
    }
    println!("Maximum deviation from Hermitian: {:.2e}", max_error);

    println!();
    Ok(())
}

// Helper functions for pretty printing

#[allow(dead_code)]
fn printmatrix(matrix: &scirs2_core::ndarray::Array2<f64>) {
    for row in matrix.rows() {
        print!("[");
        for (i, &val) in row.iter().enumerate() {
            if i > 0 {
                print!(", ");
            }
            print!("{:7.3}", val);
        }
        println!("]");
    }
}

#[allow(dead_code)]
fn print_sparsematrix(matrix: &scirs2_core::ndarray::Array2<f64>) {
    for row in matrix.rows() {
        print!("[");
        for (i, &val) in row.iter().enumerate() {
            if i > 0 {
                print!(", ");
            }
            if val.abs() < 1e-10 {
                print!("      .");
            } else {
                print!("{:7.3}", val);
            }
        }
        println!("]");
    }
}

#[allow(dead_code)]
fn print_complexmatrix(matrix: &scirs2_core::ndarray::Array2<scirs2_core::Complex<f64>>) {
    for row in matrix.rows() {
        print!("[");
        for (i, &val) in row.iter().enumerate() {
            if i > 0 {
                print!(", ");
            }
            print!("{:6.2}{:+.2}i", val.re, val.im);
        }
        println!("]");
    }
}
