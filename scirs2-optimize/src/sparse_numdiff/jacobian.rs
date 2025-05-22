//! Sparse Jacobian computation using finite differences
//!
//! This module provides functions for computing sparse Jacobian matrices
//! using various finite difference methods.

use ndarray::{Array1, ArrayView1};
use rayon::prelude::*;
use scirs2_sparse::{csr_array::CsrArray, sparray::SparseArray};

use super::coloring::determine_column_groups;
use super::finite_diff::{compute_step_sizes, SparseFiniteDiffOptions};
use crate::error::OptimizeError;

// Helper function to replace get_index and set_value_by_index which are not available in CsrArray
fn update_sparse_value(matrix: &mut CsrArray<f64>, row: usize, col: usize, value: f64) {
    // Only update if the position is non-zero in the sparsity pattern and set operation succeeds
    if matrix.get(row, col) != 0.0 && matrix.set(row, col, value).is_err() {
        // If this fails, just silently continue
    }
}

/// Computes a sparse Jacobian matrix using finite differences
///
/// # Arguments
///
/// * `func` - Function to differentiate, takes ArrayView1<f64> and returns Array1<f64>
/// * `x` - Point at which to compute the Jacobian
/// * `f0` - Function value at `x` (if None, computed internally)
/// * `sparsity_pattern` - Sparse matrix indicating the known sparsity pattern (if None, dense Jacobian)
/// * `options` - Options for finite differences computation
///
/// # Returns
///
/// * `CsrArray<f64>` - Sparse Jacobian matrix in CSR format
///
pub fn sparse_jacobian<F>(
    func: F,
    x: &ArrayView1<f64>,
    f0: Option<&Array1<f64>>,
    sparsity_pattern: Option<&CsrArray<f64>>,
    options: Option<SparseFiniteDiffOptions>,
) -> Result<CsrArray<f64>, OptimizeError>
where
    F: Fn(&ArrayView1<f64>) -> Array1<f64> + Sync,
{
    let options = options.unwrap_or_default();

    // Compute f0 if not provided
    let f0_owned: Array1<f64>;
    let f0_ref = match f0 {
        Some(f) => f,
        None => {
            f0_owned = func(x);
            &f0_owned
        }
    };

    let n = x.len(); // Input dimension
    let m = f0_ref.len(); // Output dimension

    // If no sparsity pattern provided, create a dense one
    let sparsity_owned: CsrArray<f64>;
    let sparsity = match sparsity_pattern {
        Some(p) => p,
        None => {
            // Create dense sparsity pattern
            let mut data = Vec::with_capacity(m * n);
            let mut rows = Vec::with_capacity(m * n);
            let mut cols = Vec::with_capacity(m * n);

            for i in 0..m {
                for j in 0..n {
                    data.push(1.0);
                    rows.push(i);
                    cols.push(j);
                }
            }

            sparsity_owned = CsrArray::from_triplets(&rows, &cols, &data, (m, n), false)?;
            &sparsity_owned
        }
    };

    // Choose implementation based on specified method
    match options.method.as_str() {
        "2-point" => compute_jacobian_2point(func, x, f0_ref, sparsity, &options),
        "3-point" => compute_jacobian_3point(func, x, sparsity, &options),
        "cs" => compute_jacobian_complex_step(func, x, sparsity, &options),
        _ => Err(OptimizeError::ValueError(format!(
            "Unknown method: {}. Valid options are '2-point', '3-point', and 'cs'",
            options.method
        ))),
    }
}

/// Computes Jacobian using 2-point finite differences
fn compute_jacobian_2point<F>(
    func: F,
    x: &ArrayView1<f64>,
    f0: &Array1<f64>,
    sparsity: &CsrArray<f64>,
    options: &SparseFiniteDiffOptions,
) -> Result<CsrArray<f64>, OptimizeError>
where
    F: Fn(&ArrayView1<f64>) -> Array1<f64> + Sync,
{
    let _n = x.len();
    let _m = f0.len();

    // Determine column groups for parallel evaluation using a greedy coloring algorithm
    // First get the transpose and convert it to a CsrArray
    let transposed = sparsity.transpose()?;
    let transposed_csr = match transposed.as_any().downcast_ref::<CsrArray<f64>>() {
        Some(csr) => csr,
        None => {
            return Err(OptimizeError::ValueError(
                "Failed to downcast to CsrArray".to_string(),
            ))
        }
    };
    let groups = determine_column_groups(transposed_csr, None, None)?;

    // Compute step sizes
    let h = compute_step_sizes(x, options);

    // Create result matrix with the same sparsity pattern
    let (rows, cols, _) = sparsity.find();
    let m = sparsity.shape().0;
    let n = sparsity.shape().1;
    let zeros = vec![0.0; rows.len()];
    let mut jac = CsrArray::from_triplets(&rows.to_vec(), &cols.to_vec(), &zeros, (m, n), false)?;

    // Create a mutable copy of x for perturbing
    let mut x_perturbed = x.to_owned();

    // Choose between parallel and serial execution based on options
    let parallel = options
        .parallel
        .as_ref()
        .map(|p| p.num_workers.unwrap_or(1) > 1)
        .unwrap_or(false);

    if parallel {
        // For parallel evaluation, we need to collect the derivatives first and apply them later
        let derivatives: Vec<(usize, usize, f64)> = groups
            .par_iter()
            .flat_map(|group| {
                let mut derivatives = Vec::new();
                let mut x_local = x.to_owned();

                for &col in group {
                    // Apply perturbation for this column
                    x_local[col] += h[col];

                    // Evaluate function at perturbed point
                    let f_plus = func(&x_local.view());

                    // Reset perturbation
                    x_local[col] = x[col];

                    // Compute finite difference approximation for all affected rows
                    for (row, &f0_val) in f0.iter().enumerate() {
                        // Calculate derivative and collect it
                        let derivative = (f_plus[row] - f0_val) / h[col];

                        // Only collect if position is in sparsity pattern
                        if jac.get(row, col) != 0.0 {
                            derivatives.push((row, col, derivative));
                        }
                    }
                }

                derivatives
            })
            .collect();

        // Now apply all derivatives
        for (row, col, derivative) in derivatives {
            if jac.set(row, col, derivative).is_err() {
                // If this fails, just silently continue
            }
        }
    } else {
        // Serial version processes each group sequentially
        for group in &groups {
            for &col in group {
                // Apply perturbation
                x_perturbed[col] += h[col];

                // Evaluate function at perturbed point
                let f_plus = func(&x_perturbed.view());

                // Reset perturbation
                x_perturbed[col] = x[col];

                // Compute derivatives for all affected rows
                for (row, &f0_val) in f0.iter().enumerate() {
                    // Calculate derivative and update if in sparsity pattern
                    let derivative = (f_plus[row] - f0_val) / h[col];
                    update_sparse_value(&mut jac, row, col, derivative);
                }
            }
        }
    }

    Ok(jac)
}

/// Computes Jacobian using 3-point finite differences (more accurate but twice as expensive)
fn compute_jacobian_3point<F>(
    func: F,
    x: &ArrayView1<f64>,
    sparsity: &CsrArray<f64>,
    options: &SparseFiniteDiffOptions,
) -> Result<CsrArray<f64>, OptimizeError>
where
    F: Fn(&ArrayView1<f64>) -> Array1<f64> + Sync,
{
    let _n = x.len();
    let _m = sparsity.shape().0;

    // Determine column groups for parallel evaluation
    // First get the transpose and convert it to a CsrArray
    let transposed = sparsity.transpose()?;
    let transposed_csr = match transposed.as_any().downcast_ref::<CsrArray<f64>>() {
        Some(csr) => csr,
        None => {
            return Err(OptimizeError::ValueError(
                "Failed to downcast to CsrArray".to_string(),
            ))
        }
    };
    let groups = determine_column_groups(transposed_csr, None, None)?;

    // Compute step sizes
    let h = compute_step_sizes(x, options);

    // Create result matrix with the same sparsity pattern
    let (rows, cols, _) = sparsity.find();
    let m = sparsity.shape().0;
    let n = sparsity.shape().1;
    let zeros = vec![0.0; rows.len()];
    let mut jac = CsrArray::from_triplets(&rows.to_vec(), &cols.to_vec(), &zeros, (m, n), false)?;

    // Create a mutable copy of x for perturbing
    let mut x_perturbed = x.to_owned();

    // Choose between parallel and serial execution
    let parallel = options
        .parallel
        .as_ref()
        .map(|p| p.num_workers.unwrap_or(1) > 1)
        .unwrap_or(false);

    if parallel {
        // For parallel evaluation, we need to collect the derivatives first and apply them later
        let derivatives: Vec<(usize, usize, f64)> = groups
            .par_iter()
            .flat_map(|group| {
                let mut derivatives = Vec::new();
                let mut x_local = x.to_owned();

                for &col in group {
                    // Forward perturbation
                    x_local[col] += h[col];
                    let f_plus = func(&x_local.view());

                    // Backward perturbation
                    x_local[col] = x[col] - h[col];
                    let f_minus = func(&x_local.view());

                    // Reset
                    x_local[col] = x[col];

                    // Compute central difference approximation for all affected rows
                    for row in 0..m {
                        // Calculate derivative and collect it
                        let derivative = (f_plus[row] - f_minus[row]) / (2.0 * h[col]);

                        // Only collect if position is in sparsity pattern
                        if jac.get(row, col) != 0.0 {
                            derivatives.push((row, col, derivative));
                        }
                    }
                }

                derivatives
            })
            .collect();

        // Now apply all derivatives
        for (row, col, derivative) in derivatives {
            if jac.set(row, col, derivative).is_err() {
                // If this fails, just silently continue
            }
        }
    } else {
        for group in &groups {
            for &col in group {
                // Forward perturbation
                x_perturbed[col] += h[col];
                let f_plus = func(&x_perturbed.view());

                // Backward perturbation
                x_perturbed[col] = x[col] - h[col];
                let f_minus = func(&x_perturbed.view());

                // Reset
                x_perturbed[col] = x[col];

                // Compute central difference approximation
                for row in 0..m {
                    // Calculate derivative and update if in sparsity pattern
                    let derivative = (f_plus[row] - f_minus[row]) / (2.0 * h[col]);
                    update_sparse_value(&mut jac, row, col, derivative);
                }
            }
        }
    }

    Ok(jac)
}

/// Computes Jacobian using the complex step method (highly accurate)
///
/// Note: This requires the function to support complex inputs, which isn't
/// checked at compile time. Use with caution.
fn compute_jacobian_complex_step<F>(
    _func: F,
    _x: &ArrayView1<f64>,
    _sparsity: &CsrArray<f64>,
    _options: &SparseFiniteDiffOptions,
) -> Result<CsrArray<f64>, OptimizeError>
where
    F: Fn(&ArrayView1<f64>) -> Array1<f64> + Sync,
{
    // Create error - this is just a stub implementation since complex step
    // requires much more infrastructure than we can provide here
    Err(OptimizeError::NotImplementedError(
        "Complex step method for Jacobian computation is not yet implemented".to_string(),
    ))
}
