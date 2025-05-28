//! Example demonstrating random matrix generation utilities

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use scirs2_linalg::error::LinalgResult;
use scirs2_linalg::random_matrices::{
    random_complex_matrix, random_hermitian, random_matrix, Distribution1D, MatrixType,
};

fn main() -> LinalgResult<()> {
    println!("SciRS2 Random Matrix Generation Examples");
    println!("======================================\n");

    // Use a deterministic RNG for reproducible examples
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    // Example 1: General random matrix
    demo_general_matrix(&mut rng)?;

    // Example 2: Symmetric matrix
    demo_symmetric_matrix(&mut rng)?;

    // Example 3: Positive definite matrix
    demo_positive_definite(&mut rng)?;

    // Example 4: Orthogonal matrix
    demo_orthogonal_matrix(&mut rng)?;

    // Example 5: Correlation matrix
    demo_correlation_matrix(&mut rng)?;

    // Example 6: Sparse matrix
    demo_sparse_matrix(&mut rng)?;

    // Example 7: Complex matrices
    demo_complex_matrices(&mut rng)?;

    Ok(())
}

fn demo_general_matrix<R: rand::Rng>(rng: &mut R) -> LinalgResult<()> {
    println!("1. General Random Matrix");
    println!("----------------------");

    // Uniform distribution
    let uniform_matrix = random_matrix::<f64, _>(
        3,
        4,
        MatrixType::General(Distribution1D::Uniform { a: -1.0, b: 1.0 }),
        rng,
    )?;
    println!("Uniform[-1, 1] (3x4):");
    print_matrix(&uniform_matrix);

    // Standard normal distribution
    let normal_matrix = random_matrix::<f64, _>(
        4,
        3,
        MatrixType::General(Distribution1D::StandardNormal),
        rng,
    )?;
    println!("\nStandard Normal (4x3):");
    print_matrix(&normal_matrix);

    println!();
    Ok(())
}

fn demo_symmetric_matrix<R: rand::Rng>(rng: &mut R) -> LinalgResult<()> {
    println!("2. Symmetric Random Matrix");
    println!("------------------------");

    let sym_matrix = random_matrix::<f64, _>(
        4,
        4,
        MatrixType::Symmetric(Distribution1D::Normal {
            mean: 0.0,
            std_dev: 2.0,
        }),
        rng,
    )?;

    println!("Symmetric matrix (4x4):");
    print_matrix(&sym_matrix);

    // Verify symmetry
    let mut max_diff: f64 = 0.0;
    for i in 0..4 {
        for j in i + 1..4 {
            let diff = (sym_matrix[[i, j]] - sym_matrix[[j, i]]).abs();
            max_diff = max_diff.max(diff);
        }
    }
    println!("Maximum asymmetry: {:.2e}", max_diff);

    println!();
    Ok(())
}

fn demo_positive_definite<R: rand::Rng>(rng: &mut R) -> LinalgResult<()> {
    println!("3. Positive Definite Matrix");
    println!("--------------------------");

    let pd_matrix = random_matrix::<f64, _>(
        3,
        3,
        MatrixType::PositiveDefinite {
            eigenvalue_min: 0.5,
            eigenvalue_max: 5.0,
        },
        rng,
    )?;

    println!("Positive definite matrix (3x3):");
    print_matrix(&pd_matrix);

    // Test Cholesky decomposition (only works for positive definite)
    use scirs2_linalg::cholesky;
    match cholesky(&pd_matrix.view()) {
        Ok(_) => println!("✓ Cholesky decomposition successful (matrix is positive definite)"),
        Err(_) => println!("✗ Cholesky decomposition failed"),
    }

    println!();
    Ok(())
}

fn demo_orthogonal_matrix<R: rand::Rng>(rng: &mut R) -> LinalgResult<()> {
    println!("4. Orthogonal Matrix");
    println!("-------------------");

    let ortho_matrix = random_matrix::<f64, _>(3, 3, MatrixType::Orthogonal, rng)?;

    println!("Orthogonal matrix Q (3x3):");
    print_matrix(&ortho_matrix);

    // Verify Q^T * Q = I
    let qt = ortho_matrix.t();
    let qtq = qt.dot(&ortho_matrix);

    println!("\nQ^T * Q (should be identity):");
    print_matrix(&qtq);

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

fn demo_correlation_matrix<R: rand::Rng>(rng: &mut R) -> LinalgResult<()> {
    println!("5. Correlation Matrix");
    println!("--------------------");

    let corr_matrix = random_matrix::<f64, _>(5, 5, MatrixType::Correlation, rng)?;

    println!("Correlation matrix (5x5):");
    print_matrix(&corr_matrix);

    // Verify properties
    println!("\nVerifying correlation matrix properties:");

    // Check diagonal = 1
    let mut diag_ok = true;
    for i in 0..5 {
        if (corr_matrix[[i, i]] - 1.0).abs() > 1e-10 {
            diag_ok = false;
            break;
        }
    }
    println!("✓ Diagonal elements = 1: {}", diag_ok);

    // Check symmetry
    let mut sym_ok = true;
    for i in 0..5 {
        for j in i + 1..5 {
            if (corr_matrix[[i, j]] - corr_matrix[[j, i]]).abs() > 1e-10 {
                sym_ok = false;
                break;
            }
        }
    }
    println!("✓ Symmetric: {}", sym_ok);

    // Check values in [-1, 1]
    let mut range_ok = true;
    for &val in corr_matrix.iter() {
        if !(-1.0..=1.0).contains(&val) {
            range_ok = false;
            break;
        }
    }
    println!("✓ All values in [-1, 1]: {}", range_ok);

    println!();
    Ok(())
}

fn demo_sparse_matrix<R: rand::Rng>(rng: &mut R) -> LinalgResult<()> {
    println!("6. Sparse Matrix");
    println!("---------------");

    let sparse_matrix = random_matrix::<f64, _>(
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
    print_sparse_matrix(&sparse_matrix);

    // Count non-zero elements
    let nnz = sparse_matrix.iter().filter(|&&x| x.abs() > 1e-10).count();
    let total = sparse_matrix.nrows() * sparse_matrix.ncols();
    let actual_density = nnz as f64 / total as f64;

    println!(
        "\nNon-zero elements: {} / {} (density: {:.3})",
        nnz, total, actual_density
    );

    println!();
    Ok(())
}

fn demo_complex_matrices<R: rand::Rng>(rng: &mut R) -> LinalgResult<()> {
    println!("7. Complex Matrices");
    println!("------------------");

    // General complex matrix
    let complex_matrix = random_complex_matrix::<f64, _>(
        3,
        3,
        Distribution1D::StandardNormal,
        Distribution1D::StandardNormal,
        rng,
    )?;

    println!("Complex matrix (3x3):");
    print_complex_matrix(&complex_matrix);

    // Hermitian matrix
    let hermitian = random_hermitian::<f64, _>(
        4,
        Distribution1D::StandardNormal,
        Distribution1D::Uniform { a: -0.5, b: 0.5 },
        rng,
    )?;

    println!("\nHermitian matrix (4x4):");
    print_complex_matrix(&hermitian);

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

fn print_matrix(matrix: &ndarray::Array2<f64>) {
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

fn print_sparse_matrix(matrix: &ndarray::Array2<f64>) {
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

fn print_complex_matrix(matrix: &ndarray::Array2<num_complex::Complex<f64>>) {
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
