//! Extended precision matrix operations example
//!
//! This example demonstrates the use of extended precision operations for improved accuracy.
//! This includes matrix operations, determinant calculation, factorizations and eigendecompositions.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_linalg::error::LinalgResult;
use scirs2_linalg::prelude::*;

#[allow(dead_code)]
fn main() -> LinalgResult<()> {
    println!("Extended Precision Matrix Operations Example");
    println!("==========================================\n");

    // Demo with a Hilbert matrix, which is notoriously ill-conditioned
    println!("Example with Hilbert Matrix (ill-conditioned)");
    println!("-------------------------------------------\n");

    // Create a small Hilbert matrix
    let n = 5;
    let mut hilbert = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            hilbert[[i, j]] = 1.0 / ((i + j + 1) as f64);
        }
    }

    println!("Hilbert matrix of size {}x{}:", n, n);
    for i in 0..n {
        for j in 0..n {
            print!("{:.6} ", hilbert[[i, j]]);
        }
        println!();
    }
    println!();

    // Create a known solution vector
    let x_true: Array1<f64> = Array1::from_vec((0..n).map(|i| (i + 1) as f64).collect());
    println!("True solution x: {:?}", x_true);

    // Compute right-hand side b = A*x
    let b = hilbert.dot(&x_true);
    println!("Right-hand side b: {:?}\n", b);

    // Solve using standard precision
    let x_std = match scirs2_linalg::solve(&hilbert.view(), &b.view(), None) {
        Ok(result) => result,
        Err(e) => {
            println!("Error in standard precision solve: {}", e);
            Array1::zeros(n)
        }
    };

    // Solve using extended precision (f64 calculation internally for f32 data)
    // For this example, we'll manually convert to f32 and back
    let hilbert_f32: Array2<f32> = Array2::from_shape_fn((n, n), |(i, j)| hilbert[[i, j]] as f32);
    let b_f32: Array1<f32> = Array1::from_shape_fn(n, |i| b[i] as f32);

    let x_ext = extended_solve::<f32, f64>(&hilbert_f32.view(), &b_f32.view())?;

    // Compute errors
    let error_std = x_std
        .iter()
        .zip(x_true.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, |max, val| if val > max { val } else { max });

    let error_ext = x_ext
        .iter()
        .zip((0..n).map(|i| (i + 1) as f32))
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, |max, val| if val > max { val } else { max });

    println!("Solution using standard precision:");
    println!("{:?}", x_std);
    println!("Maximum absolute error: {:.6e}\n", error_std);

    println!("Solution using extended precision:");
    println!("{:?}", x_ext);
    println!("Maximum absolute error: {:.6e}\n", error_ext);

    // Matrix-matrix multiplication example
    println!("Extended Precision Matrix Multiplication");
    println!("-------------------------------------\n");

    let a_f32 = Array2::from_shape_fn((3, 3), |(i, j)| 1.0 / ((i + j + 1) as f32));

    let b_f32 = Array2::from_shape_fn((3, 2), |(i, j)| ((i + 1) * (j + 1)) as f32);

    println!("Matrix A (f32):");
    for i in 0..a_f32.nrows() {
        for j in 0..a_f32.ncols() {
            print!("{:.6} ", a_f32[[i, j]]);
        }
        println!();
    }

    println!("\nMatrix B (f32):");
    for i in 0..b_f32.nrows() {
        for j in 0..b_f32.ncols() {
            print!("{:.6} ", b_f32[[i, j]]);
        }
        println!();
    }

    // Standard precision multiplication
    let c_std = a_f32.dot(&b_f32);

    // Extended precision multiplication
    let c_ext = extended_matmul::<f32, f64>(&a_f32.view(), &b_f32.view())?;

    println!("\nResult with standard precision:");
    for i in 0..c_std.nrows() {
        for j in 0..c_std.ncols() {
            print!("{:.10} ", c_std[[i, j]]);
        }
        println!();
    }

    println!("\nResult with extended precision:");
    for i in 0..c_ext.nrows() {
        for j in 0..c_ext.ncols() {
            print!("{:.10} ", c_ext[[i, j]]);
        }
        println!();
    }

    // Calculate the difference
    let mut max_diff = 0.0f32;
    for i in 0..c_std.nrows() {
        for j in 0..c_std.ncols() {
            let diff = (c_std[[i, j]] - c_ext[[i, j]]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
    }

    println!("\nMaximum difference between methods: {:.6e}", max_diff);

    // Extended precision determinant calculation
    println!("\nExtended Precision Determinant Calculation");
    println!("---------------------------------------\n");

    // Create a Hilbert matrix of order 6 (extremely ill-conditioned)
    let n = 6;
    let mut hilbert_det = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            hilbert_det[[i, j]] = 1.0 / ((i + j + 1) as f64);
        }
    }

    // Convert to f32 for comparison
    let hilbert_det_f32: Array2<f32> =
        Array2::from_shape_fn((n, n), |(i, j)| hilbert_det[[i, j]] as f32);

    println!("Hilbert matrix of order {}:", n);
    for i in 0..n {
        for j in 0..n {
            print!("{:.4} ", hilbert_det[[i, j]]);
        }
        println!();
    }

    // Calculate determinant with standard precision
    let det_std = match det(&hilbert_det_f32.view(), None) {
        Ok(d) => d,
        Err(e) => {
            println!("Error calculating standard precision determinant: {}", e);
            0.0
        }
    };

    // Calculate determinant with extended precision
    let det_ext = extended_det::<_, f64>(&hilbert_det_f32.view())?;

    // Calculate f64 reference
    let det_f64 = det(&hilbert_det.view(), None)?;
    let det_f64_as_f32 = det_f64 as f32;

    println!(
        "\nDeterminant with standard precision (f32): {:.10e}",
        det_std
    );
    println!(
        "Determinant with extended precision (f32->f64->f32): {:.10e}",
        det_ext
    );
    println!("Reference determinant (f64): {:.10e}", det_f64);
    println!(
        "Reference determinant (f64 as f32): {:.10e}",
        det_f64_as_f32
    );

    // Compute errors with respect to f64 reference
    let error_std = (det_std - det_f64_as_f32).abs();
    let error_ext = (det_ext - det_f64_as_f32).abs();

    println!(
        "\nAbsolute error from reference (standard precision): {:.10e}",
        error_std
    );
    println!(
        "Absolute error from reference (extended precision): {:.10e}",
        error_ext
    );

    if error_ext < error_std {
        println!("Improvement factor: {:.2}x", error_std / error_ext);
    } else {
        println!("No improvement observed in this case.");
    }

    // Extended precision factorization example
    println!("\nExtended Precision Matrix Factorizations");
    println!("-------------------------------------\n");

    // Create an ill-conditioned test matrix
    let testmatrix_f32 = Array2::from_shape_fn((4, 4), |(i, j)| {
        if i == j {
            1.0f32
        } else {
            0.9f32 // Close to identity but ill-conditioned
        }
    });

    println!("Test matrix (close to singular):");
    for i in 0..testmatrix_f32.nrows() {
        for j in 0..testmatrix_f32.ncols() {
            print!("{:.4} ", testmatrix_f32[[i, j]]);
        }
        println!();
    }

    // LU factorization with extended precision
    println!("\nLU factorization with extended precision:");
    let (p, l, u) = extended_lu::<_, f64>(&testmatrix_f32.view())?;

    println!("L matrix:");
    for i in 0..l.nrows() {
        for j in 0..l.ncols() {
            print!("{:.6} ", l[[i, j]]);
        }
        println!();
    }

    println!("\nU matrix:");
    for i in 0..u.nrows() {
        for j in 0..u.ncols() {
            print!("{:.6} ", u[[i, j]]);
        }
        println!();
    }

    // Verify A ≈ P^T * L * U
    let p_t = p.t();
    let lu = l.dot(&u);
    let reconstructed = p_t.dot(&lu);

    println!("\nOriginal matrix:");
    for i in 0..testmatrix_f32.nrows() {
        for j in 0..testmatrix_f32.ncols() {
            print!("{:.6} ", testmatrix_f32[[i, j]]);
        }
        println!();
    }

    println!("\nReconstructed matrix (P^T * L * U):");
    for i in 0..reconstructed.nrows() {
        for j in 0..reconstructed.ncols() {
            print!("{:.6} ", reconstructed[[i, j]]);
        }
        println!();
    }

    // Calculate reconstruction error
    let mut max_error = 0.0f32;
    for i in 0..testmatrix_f32.nrows() {
        for j in 0..testmatrix_f32.ncols() {
            let error = (testmatrix_f32[[i, j]] - reconstructed[[i, j]]).abs();
            if error > max_error {
                max_error = error;
            }
        }
    }

    println!("\nMaximum reconstruction error: {:.10e}", max_error);

    // Extended precision eigenvalue decomposition
    println!("\nExtended Precision Eigenvalue Decomposition");
    println!("----------------------------------------\n");

    // Create a symmetric matrix that's moderately ill-conditioned
    let n = 5;
    let mut symmatrix = Array2::<f32>::zeros((n, n));

    // Fill with Hilbert-like entries but ensure symmetry
    for i in 0..n {
        for j in 0..=i {
            symmatrix[[i, j]] = 1.0 / ((i + j + 1) as f32);
            symmatrix[[j, i]] = symmatrix[[i, j]]; // Ensure symmetry
        }
    }

    println!("Symmetric matrix:");
    for i in 0..n {
        for j in 0..n {
            print!("{:.4} ", symmatrix[[i, j]]);
        }
        println!();
    }

    // Compute eigenvalues and eigenvectors with standard precision
    let (eigvals_std, eigvecs_std) = match scirs2_linalg::eigh(&symmatrix.view(), None) {
        Ok(result) => result,
        Err(e) => {
            println!(
                "Error computing standard precision eigendecomposition: {}",
                e
            );
            (Array1::zeros(n), Array2::zeros((n, n)))
        }
    };

    // Compute eigenvalues and eigenvectors with extended precision
    let (eigvals_ext, eigvecs_ext) = extended_eigh::<_, f64>(&symmatrix.view(), None, None)?;

    println!("\nEigenvalues with standard precision:");
    for (i, &val) in eigvals_std.iter().enumerate() {
        println!("λ{} = {:.10e}", i + 1, val);
    }

    println!("\nEigenvalues with extended precision:");
    for (i, &val) in eigvals_ext.iter().enumerate() {
        println!("λ{} = {:.10e}", i + 1, val);
    }

    // Check orthogonality of eigenvectors
    let mut max_nonortho_std = 0.0f32;
    let mut max_nonortho_ext = 0.0f32;

    for i in 0..n {
        for j in i + 1..n {
            let dot_std = eigvecs_std.column(i).dot(&eigvecs_std.column(j));
            let dot_ext = eigvecs_ext.column(i).dot(&eigvecs_ext.column(j));

            max_nonortho_std = max_nonortho_std.max(dot_std.abs());
            max_nonortho_ext = max_nonortho_ext.max(dot_ext.abs());
        }
    }

    println!(
        "\nMaximum non-orthogonality of eigenvectors (standard precision): {:.10e}",
        max_nonortho_std
    );
    println!(
        "Maximum non-orthogonality of eigenvectors (extended precision): {:.10e}",
        max_nonortho_ext
    );

    // Check eigenvector quality: A*v = lambda*v
    let mut max_residual_std = 0.0f32;
    let mut max_residual_ext = 0.0f32;

    for j in 0..n {
        // Standard precision
        let v_std = eigvecs_std.column(j).to_owned();
        let av_std = symmatrix.dot(&v_std);
        let lambda_v_std = &v_std * eigvals_std[j];

        let residual_std = (&av_std - &lambda_v_std)
            .iter()
            .map(|&x| x.abs())
            .fold(0.0f32, |a, b| a.max(b));

        max_residual_std = max_residual_std.max(residual_std);

        // Extended precision
        let v_ext = eigvecs_ext.column(j).to_owned();
        let av_ext = symmatrix.dot(&v_ext);
        let lambda_v_ext = &v_ext * eigvals_ext[j];

        let residual_ext = (&av_ext - &lambda_v_ext)
            .iter()
            .map(|&x| x.abs())
            .fold(0.0f32, |a, b| a.max(b));

        max_residual_ext = max_residual_ext.max(residual_ext);
    }

    println!(
        "\nMaximum residual |A*v - lambda*v| (standard precision): {:.10e}",
        max_residual_std
    );
    println!(
        "Maximum residual |A*v - lambda*v| (extended precision): {:.10e}",
        max_residual_ext
    );
    println!(
        "Improvement factor: {:.2}x",
        max_residual_std / max_residual_ext
    );

    // Summary of benefits
    println!("\nSummary of Extended Precision Benefits");
    println!("=====================================");
    println!("1. Improved accuracy for ill-conditioned matrices");
    println!("2. Better numerical stability for matrix decompositions");
    println!("3. More accurate determinant calculation for nearly singular matrices");
    println!("4. Better solutions for linear systems with challenging numerical properties");
    println!("5. Reduced accumulation of rounding errors in matrix operations");
    println!("6. Higher quality eigendecompositions for sensitive applications");
    println!("7. More orthogonal eigenvectors for symmetric matrices");

    Ok(())
}
