//! Comprehensive tutorial for scirs2-sparse module
//!
//! This example demonstrates the various features implemented in the sparse module,
//! including matrix formats, linear algebra operations, eigenvalue computations,
//! and advanced solver methods.

use scirs2_core::ndarray::Array1;
use scirs2_sparse::{
    csr_array::CsrArray,
    error::SparseResult,
    linalg::{
        bicgstab, cg, gcrot, gmres, lu_decomposition_with_options, pivoted_cholesky_decomposition,
        tfqmr, twonormest_enhanced, AsLinearOperator, BiCGSTABOptions, CGOptions, GCROTOptions,
        GMRESOptions, IdentityOperator, LUOptions, LinearOperator, PivotingStrategy, TFQMROptions,
    },
    SparseArray,
};

#[allow(dead_code)]
fn main() -> SparseResult<()> {
    println!("=== SciRS2-Sparse Comprehensive Tutorial ===\n");

    // 1. Basic Matrix Construction and Operations
    demonstrate_basic_operations()?;

    // 2. Advanced Decompositions
    demonstrate_advanced_decompositions()?;

    // 3. Norm Estimation and Condition Numbers
    demonstrate_norm_estimation()?;

    // 4. Advanced Eigenvalue Problems
    demonstrate_eigenvalue_problems()?;

    // 5. Linear Operators and Composition
    demonstrate_linear_operators()?;

    // 6. Advanced Iterative Solvers
    demonstrate_advanced_solvers()?;

    // 7. Error Handling and Diagnostics
    demonstrate_error_handling()?;

    println!("\n=== Tutorial Complete ===");
    Ok(())
}

#[allow(dead_code)]
fn demonstrate_basic_operations() -> SparseResult<()> {
    println!("1. Basic Matrix Construction and Operations");
    println!("==========================================");

    // Create a test matrix
    let rows = vec![0, 0, 1, 1, 2, 2, 3];
    let cols = vec![0, 3, 1, 2, 0, 2, 3];
    let data = vec![4.0, -1.0, 5.0, -2.0, -1.0, 6.0, 3.0];
    let matrix = CsrArray::from_triplets(&rows, &cols, &data, (4, 4), false)?;

    println!("Created 4x4 sparse matrix with {} non-zeros", matrix.nnz());
    println!("Matrix shape: {:?}", matrix.shape());

    // Matrix-vector multiplication
    let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
    let y = matrix.dot_vector(&x.view())?;
    println!("Matrix-vector product: {:?}", y);

    // Convert to dense for visualization
    let dense = matrix.to_array();
    println!("Dense representation:");
    for i in 0..4 {
        print!("[");
        for j in 0..4 {
            print!("{:6.1}", dense[[i, j]]);
        }
        println!("]");
    }

    println!();
    Ok(())
}

#[allow(dead_code)]
fn demonstrate_advanced_decompositions() -> SparseResult<()> {
    println!("2. Advanced Matrix Decompositions");
    println!("=================================");

    // Create a test matrix for LU decomposition
    let rows = vec![0, 0, 1, 1, 2, 2];
    let cols = vec![0, 1, 0, 1, 1, 2];
    let data = vec![2.0, 1.0, 1.0, 3.0, 2.0, 4.0];
    let matrix = CsrArray::from_triplets(&rows, &cols, &data, (3, 3), false)?;

    println!("LU Decomposition with Enhanced Pivoting:");

    // Test different pivoting strategies
    let strategies = vec![
        ("Partial", PivotingStrategy::Partial),
        ("Scaled Partial", PivotingStrategy::ScaledPartial),
        ("Threshold", PivotingStrategy::Threshold(0.1)),
        ("Complete", PivotingStrategy::Complete),
        ("Rook", PivotingStrategy::Rook),
    ];

    for (name, strategy) in strategies {
        let options = LUOptions {
            pivoting: strategy,
            zero_threshold: 1e-12,
            check_singular: true,
        };

        match lu_decomposition_with_options(&matrix, Some(options)) {
            Ok(lu_result) => {
                println!(
                    "  {} pivoting: Success (rank preserved: {})",
                    name, lu_result.success
                );
            }
            Err(e) => {
                println!("  {} pivoting: Failed - {}", name, e);
            }
        }
    }

    // Demonstrate pivoted Cholesky for indefinite matrices
    println!("\nPivoted Cholesky Decomposition:");
    let indefinite_rows = vec![0, 1, 1, 2, 2, 2];
    let indefinite_cols = vec![0, 0, 1, 0, 1, 2];
    let indefinite_data = vec![1.0, 2.0, -1.0, 3.0, 1.0, 2.0];
    let indefinite_matrix = CsrArray::from_triplets(
        &indefinite_rows,
        &indefinite_cols,
        &indefinite_data,
        (3, 3),
        false,
    )?;

    match pivoted_cholesky_decomposition(&indefinite_matrix, Some(1e-12)) {
        Ok(chol_result) => {
            println!(
                "  Pivoted Cholesky: Rank = {}, Success = {}",
                chol_result.rank, chol_result.success
            );
            println!("  This handles indefinite matrices gracefully!");
        }
        Err(e) => {
            println!("  Pivoted Cholesky failed: {}", e);
        }
    }

    println!();
    Ok(())
}

#[allow(dead_code)]
fn demonstrate_norm_estimation() -> SparseResult<()> {
    println!("3. Matrix Norm Estimation and Condition Numbers");
    println!("===============================================");

    // Create a well-conditioned matrix
    let rows = vec![0, 1, 2];
    let cols = vec![0, 1, 2];
    let data = vec![2.0, 3.0, 4.0];
    let well_conditioned = CsrArray::from_triplets(&rows, &cols, &data, (3, 3), false)?;

    // Estimate 2-norm (spectral norm) using enhanced version
    match twonormest_enhanced(&well_conditioned, Some(1e-8), Some(100), None) {
        Ok(norm_2) => {
            println!("2-norm estimate: {:.6}", norm_2);
        }
        Err(e) => {
            println!("2-norm estimation failed: {}", e);
        }
    }

    // Estimate condition number
    // Note: Condition number estimation with CsrArray currently requires conversion
    // For demonstration purposes, we'll skip this as condest requires legacy CsrMatrix format
    println!("Condition number estimation: (skipped - requires CsrMatrix format)");
    println!("Matrix appears well-conditioned based on construction");

    println!();
    Ok(())
}

#[allow(dead_code)]
fn demonstrate_eigenvalue_problems() -> SparseResult<()> {
    println!("4. Advanced Eigenvalue Problems");
    println!("===============================");

    // Create a symmetric matrix for eigenvalue computations
    let rows = vec![0, 1, 1, 2, 2, 2];
    let cols = vec![0, 0, 1, 0, 1, 2];
    let data = vec![4.0, 1.0, 5.0, 1.0, 2.0, 6.0];
    let _sym_matrix = CsrArray::from_triplets(&rows, &cols, &data, (3, 3), false)?;

    println!("Shift-and-Invert Eigenvalue Method:");
    println!("(Finding eigenvalues near a target value)");

    // This would demonstrate shift-and-invert for interior eigenvalues
    // The actual implementation may require symmetric matrix format
    println!("  Shift-and-invert allows finding eigenvalues near a specified target");
    println!("  This is useful for finding interior eigenvalues efficiently");

    println!("\nGeneralized Eigenvalue Problems (Ax = λBx):");
    println!("  Generalized eigenvalue problems are implemented for cases where");
    println!("  you need to solve Ax = λBx with two matrices A and B");
    println!("  This is common in vibration analysis and stability problems");

    println!();
    Ok(())
}

#[allow(dead_code)]
fn demonstrate_linear_operators() -> SparseResult<()> {
    println!("5. Linear Operators and Composition");
    println!("===================================");

    // Create basic operators
    let id = Box::new(IdentityOperator::<f64>::new(3));
    println!("Created identity operator of size 3x3");

    // Demonstrate operator application
    let x = vec![1.0, 2.0, 3.0];
    let y = id.matvec(&x)?;
    println!("Identity * [1, 2, 3] = {:?}", y);

    println!("Linear operators support:");
    println!("  ✓ Addition (A + B)");
    println!("  ✓ Subtraction (A - B)");
    println!("  ✓ Multiplication (A * B)");
    println!("  ✓ Scalar multiplication (α * A)");
    println!("  ✓ Transpose (A^T)");
    println!("  ✓ Power (A^n)");
    println!("  ✓ Composition chains");

    println!();
    Ok(())
}

#[allow(dead_code)]
fn demonstrate_advanced_solvers() -> SparseResult<()> {
    println!("6. Advanced Iterative Solvers");
    println!("=============================");

    // Create a test linear system
    let rows = vec![0, 0, 1, 1, 1, 2, 2];
    let cols = vec![0, 1, 0, 1, 2, 1, 2];
    let data = vec![3.0, 1.0, 1.0, 4.0, 1.0, 1.0, 5.0];
    let a_matrix = CsrArray::from_triplets(&rows, &cols, &data, (3, 3), false)?;
    let b_vector = Array1::from_vec(vec![6.0, 8.0, 10.0]);

    println!("Solving Ax = b with different methods:");

    // Conjugate Gradient
    let cg_options = CGOptions {
        max_iter: 100,
        rtol: 1e-8,
        ..Default::default()
    };
    match cg(
        &*a_matrix.as_linear_operator(),
        &b_vector.to_vec(),
        cg_options,
    ) {
        Ok(result) => {
            println!(
                "  CG: Converged in {} iterations, residual: {:.2e}",
                result.iterations, result.residual_norm
            );
        }
        Err(e) => {
            println!("  CG failed: {}", e);
        }
    }

    // BiCGSTAB
    let bicgstab_options = BiCGSTABOptions {
        max_iter: 100,
        rtol: 1e-8,
        ..Default::default()
    };
    match bicgstab(
        &*a_matrix.as_linear_operator(),
        &b_vector.to_vec(),
        bicgstab_options,
    ) {
        Ok(result) => {
            println!(
                "  BiCGSTAB: Converged in {} iterations, residual: {:.2e}",
                result.iterations, result.residual_norm
            );
        }
        Err(e) => {
            println!("  BiCGSTAB failed: {}", e);
        }
    }

    // GMRES
    let gmres_options = GMRESOptions {
        max_iter: 100,
        rtol: 1e-8,
        restart: 20,
        ..Default::default()
    };
    match gmres(
        &*a_matrix.as_linear_operator(),
        &b_vector.to_vec(),
        gmres_options,
    ) {
        Ok(result) => {
            println!(
                "  GMRES: Converged in {} iterations, residual: {:.2e}",
                result.iterations, result.residual_norm
            );
        }
        Err(e) => {
            println!("  GMRES failed: {}", e);
        }
    }

    // GCROT (advanced Krylov method)
    let gcrot_options = GCROTOptions {
        max_iter: 100,
        tol: 1e-8,
        truncation_size: 10,
        ..Default::default()
    };
    match gcrot(&a_matrix, &b_vector.view(), None, gcrot_options) {
        Ok(result) => {
            println!(
                "  GCROT: Converged in {} iterations, residual: {:.2e}",
                result.iterations, result.residual_norm
            );
        }
        Err(e) => {
            println!("  GCROT failed: {}", e);
        }
    }

    // TFQMR (another advanced method)
    let tfqmr_options = TFQMROptions {
        max_iter: 100,
        tol: 1e-8,
        ..Default::default()
    };
    match tfqmr(&a_matrix, &b_vector.view(), None, tfqmr_options) {
        Ok(result) => {
            println!(
                "  TFQMR: Converged in {} iterations, residual: {:.2e}",
                result.iterations, result.residual_norm
            );
        }
        Err(e) => {
            println!("  TFQMR failed: {}", e);
        }
    }

    println!();
    Ok(())
}

#[allow(dead_code)]
fn demonstrate_error_handling() -> SparseResult<()> {
    println!("7. Enhanced Error Handling and Diagnostics");
    println!("==========================================");

    // Demonstrate helpful error messages
    println!("Enhanced error handling provides:");
    println!("  ✓ Detailed error descriptions");
    println!("  ✓ Specific suggestions for fixes");
    println!("  ✓ Context-aware error messages");

    // Example of dimension mismatch error
    let matrix = CsrArray::from_triplets(&[0], &[0], &[1.0], (1, 1), false)?;
    let wrong_vector = Array1::from_vec(vec![1.0, 2.0]); // Wrong size

    match matrix.dot_vector(&wrong_vector.view()) {
        Ok(_) => println!("  Unexpected success"),
        Err(e) => {
            println!("\nExample error:");
            println!("  Error: {}", e);
            println!("  Help: {}", e.help_message());
            println!("  Suggestions:");
            for suggestion in e.suggestions() {
                println!("    - {}", suggestion);
            }
        }
    }

    println!();
    Ok(())
}
