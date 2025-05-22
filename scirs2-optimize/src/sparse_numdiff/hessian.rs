//! Sparse Hessian computation using finite differences
//!
//! This module provides functions for computing sparse Hessian matrices
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

// Helper function to check if a position exists in the sparsity pattern
fn exists_in_sparsity(matrix: &CsrArray<f64>, row: usize, col: usize) -> bool {
    matrix.get(row, col) != 0.0
}

/// Computes a sparse Hessian matrix using finite differences
///
/// # Arguments
///
/// * `func` - Function to differentiate, takes ArrayView1<f64> and returns f64
/// * `grad` - Optional gradient function, takes ArrayView1<f64> and returns Array1<f64>
/// * `x` - Point at which to compute the Hessian
/// * `f0` - Function value at `x` (if None, computed internally)
/// * `g0` - Gradient value at `x` (if None, computed internally)
/// * `sparsity_pattern` - Sparse matrix indicating the known sparsity pattern (if None, dense Hessian)
/// * `options` - Options for finite differences computation
///
/// # Returns
///
/// * `CsrArray<f64>` - Sparse Hessian matrix in CSR format
///
pub fn sparse_hessian<F, G>(
    func: F,
    grad: Option<G>,
    x: &ArrayView1<f64>,
    f0: Option<f64>,
    g0: Option<&Array1<f64>>,
    sparsity_pattern: Option<&CsrArray<f64>>,
    options: Option<SparseFiniteDiffOptions>,
) -> Result<CsrArray<f64>, OptimizeError>
where
    F: Fn(&ArrayView1<f64>) -> f64 + Sync,
    G: Fn(&ArrayView1<f64>) -> Array1<f64> + Sync + 'static,
{
    let options = options.unwrap_or_default();
    let n = x.len();

    // If gradient function is provided, use it to compute Hessian via forward differences
    // on the gradient
    if let Some(gradient_fn) = grad {
        return compute_hessian_from_gradient(gradient_fn, x, g0, sparsity_pattern, &options);
    }

    // If no sparsity pattern provided, create a dense one
    let sparsity_owned: CsrArray<f64>;
    let sparsity = match sparsity_pattern {
        Some(p) => {
            // Validate sparsity pattern
            if p.shape().0 != n || p.shape().1 != n {
                return Err(OptimizeError::ValueError(format!(
                    "Sparsity pattern shape {:?} does not match input dimension {}",
                    p.shape(),
                    n
                )));
            }
            p
        }
        None => {
            // Create dense sparsity pattern
            let mut data = Vec::with_capacity(n * n);
            let mut rows = Vec::with_capacity(n * n);
            let mut cols = Vec::with_capacity(n * n);

            for i in 0..n {
                for j in 0..n {
                    data.push(1.0);
                    rows.push(i);
                    cols.push(j);
                }
            }

            sparsity_owned = CsrArray::from_triplets(&rows, &cols, &data, (n, n), false)?;
            &sparsity_owned
        }
    };

    // Ensure sparsity pattern is symmetric (Hessian is symmetric)
    // In practice, we only need to compute the upper triangle and then
    // fill in the lower triangle at the end
    let symmetric_sparsity = make_symmetric_sparsity(sparsity)?;

    // Choose implementation based on specified method
    let result = match options.method.as_str() {
        "2-point" => {
            let f0_val = f0.unwrap_or_else(|| func(x));
            compute_hessian_2point(func, x, f0_val, &symmetric_sparsity, &options)
        }
        "3-point" => compute_hessian_3point(func, x, &symmetric_sparsity, &options),
        "cs" => compute_hessian_complex_step(func, x, &symmetric_sparsity, &options),
        _ => Err(OptimizeError::ValueError(format!(
            "Unknown method: {}. Valid options are '2-point', '3-point', and 'cs'",
            options.method
        ))),
    }?;

    // Fill in the lower triangle to ensure symmetry
    fill_symmetric_hessian(&result)
}

/// Computes Hessian from a gradient function using forward differences
fn compute_hessian_from_gradient<G>(
    grad_fn: G,
    x: &ArrayView1<f64>,
    g0: Option<&Array1<f64>>,
    sparsity_pattern: Option<&CsrArray<f64>>,
    options: &SparseFiniteDiffOptions,
) -> Result<CsrArray<f64>, OptimizeError>
where
    G: Fn(&ArrayView1<f64>) -> Array1<f64> + Sync + 'static,
{
    let _n = x.len();

    // Compute g0 if not provided
    let g0_owned: Array1<f64>;
    let g0_ref = match g0 {
        Some(g) => g,
        None => {
            g0_owned = grad_fn(x);
            &g0_owned
        }
    };

    // The gradient function can be treated as a vector-valued function,
    // so we can use sparse_jacobian to compute the Hessian (which is the Jacobian of the gradient)
    let jac_options = SparseFiniteDiffOptions {
        method: options.method.clone(),
        rel_step: options.rel_step,
        abs_step: options.abs_step,
        bounds: options.bounds.clone(),
        parallel: options.parallel.clone(),
        seed: options.seed,
        max_group_size: options.max_group_size,
    };

    // Use sparse_jacobian to compute the Hessian
    let hessian = super::jacobian::sparse_jacobian(
        grad_fn,
        x,
        Some(g0_ref),
        sparsity_pattern,
        Some(jac_options),
    )?;

    // Ensure the Hessian is symmetric
    fill_symmetric_hessian(&hessian)
}

/// Computes Hessian using 2-point finite differences
fn compute_hessian_2point<F>(
    func: F,
    x: &ArrayView1<f64>,
    f0: f64,
    sparsity: &CsrArray<f64>,
    options: &SparseFiniteDiffOptions,
) -> Result<CsrArray<f64>, OptimizeError>
where
    F: Fn(&ArrayView1<f64>) -> f64 + Sync,
{
    let _n = x.len();

    // Determine column groups using a graph coloring algorithm
    let groups = determine_column_groups(sparsity, None, None)?;

    // Compute step sizes
    let h = compute_step_sizes(x, options);

    // Create result matrix with the same sparsity pattern as the upper triangle
    let (rows, cols, _) = sparsity.find();
    let (m, n) = sparsity.shape();
    let zeros = vec![0.0; rows.len()];
    let mut hess = CsrArray::from_triplets(&rows.to_vec(), &cols.to_vec(), &zeros, (m, n), false)?;

    // Create a mutable copy of x for perturbing
    let mut x_perturbed = x.to_owned();

    // Choose between parallel and serial execution
    let parallel = options
        .parallel
        .as_ref()
        .map(|p| p.num_workers.unwrap_or(1) > 1)
        .unwrap_or(false);

    // First set of function evaluations for the diagonal elements
    let diag_evals: Vec<f64> = if parallel {
        (0..n)
            .into_par_iter()
            .map(|i| {
                let mut x_local = x.to_owned();
                x_local[i] += h[i];
                func(&x_local.view())
            })
            .collect()
    } else {
        let mut diag_vals = vec![0.0; n];
        for i in 0..n {
            x_perturbed[i] += h[i];
            diag_vals[i] = func(&x_perturbed.view());
            x_perturbed[i] = x[i];
        }
        diag_vals
    };

    // Set diagonal elements of the Hessian
    for i in 0..n {
        // Calculate second derivative
        let d2f_dxi2 = (diag_evals[i] - 2.0 * f0 + diag_evals[i]) / (h[i] * h[i]);

        // Update value if in sparsity pattern
        update_sparse_value(&mut hess, i, i, d2f_dxi2);
    }

    // Now compute off-diagonal elements
    if parallel {
        // For parallel evaluation, we need to collect the derivatives first and apply them later
        let derivatives: Vec<(usize, usize, f64)> = groups
            .par_iter()
            .flat_map(|group| {
                let mut derivatives = Vec::new();
                let mut x_local = x.to_owned();

                for &j in group {
                    // Only compute upper triangle
                    for i in 0..j {
                        if exists_in_sparsity(&hess, i, j) {
                            // Apply perturbation for both indices
                            x_local[i] += h[i];
                            x_local[j] += h[j];

                            // f(x + h_i*e_i + h_j*e_j)
                            let f_ij = func(&x_local.view());

                            // f(x + h_i*e_i)
                            x_local[j] = x[j];
                            let f_i = diag_evals[i];

                            // f(x + h_j*e_j)
                            x_local[i] = x[i];
                            x_local[j] += h[j];
                            let f_j = diag_evals[j];

                            // Mixed partial derivative
                            let d2f_dxidxj = (f_ij - f_i - f_j + f0) / (h[i] * h[j]);

                            // Collect derivative
                            derivatives.push((i, j, d2f_dxidxj));

                            // Reset
                            x_local[j] = x[j];
                        }
                    }
                }

                derivatives
            })
            .collect();

        // Now apply all derivatives
        for (i, j, d2f_dxidxj) in derivatives {
            if hess.set(i, j, d2f_dxidxj).is_err() {
                // If this fails, just silently continue
            }
        }
    } else {
        for group in &groups {
            for &j in group {
                // Only compute upper triangle
                for i in 0..j {
                    if exists_in_sparsity(&hess, i, j) {
                        // Apply perturbation for both indices
                        x_perturbed[i] += h[i];
                        x_perturbed[j] += h[j];

                        // f(x + h_i*e_i + h_j*e_j)
                        let f_ij = func(&x_perturbed.view());

                        // Mixed partial derivative
                        let d2f_dxidxj =
                            (f_ij - diag_evals[i] - diag_evals[j] + f0) / (h[i] * h[j]);

                        // Update value
                        update_sparse_value(&mut hess, i, j, d2f_dxidxj);

                        // Reset
                        x_perturbed[i] = x[i];
                        x_perturbed[j] = x[j];
                    }
                }
            }
        }
    }

    Ok(hess)
}

/// Computes Hessian using 3-point finite differences (more accurate but more expensive)
fn compute_hessian_3point<F>(
    _func: F,
    _x: &ArrayView1<f64>,
    _sparsity: &CsrArray<f64>,
    _options: &SparseFiniteDiffOptions,
) -> Result<CsrArray<f64>, OptimizeError>
where
    F: Fn(&ArrayView1<f64>) -> f64 + Sync,
{
    // This is a placeholder implementation that would need to be expanded
    // with the full 3-point algorithm. For now, we just return an error.
    Err(OptimizeError::NotImplementedError(
        "3-point method for Hessian computation is not yet fully implemented".to_string(),
    ))
}

/// Computes Hessian using the complex step method (highly accurate)
fn compute_hessian_complex_step<F>(
    _func: F,
    _x: &ArrayView1<f64>,
    _sparsity: &CsrArray<f64>,
    _options: &SparseFiniteDiffOptions,
) -> Result<CsrArray<f64>, OptimizeError>
where
    F: Fn(&ArrayView1<f64>) -> f64 + Sync,
{
    // This is a placeholder implementation that would need to be expanded
    // with the complex step algorithm. For now, we just return an error.
    Err(OptimizeError::NotImplementedError(
        "Complex step method for Hessian computation is not yet implemented".to_string(),
    ))
}

/// Ensures a sparsity pattern is symmetric
fn make_symmetric_sparsity(sparsity: &CsrArray<f64>) -> Result<CsrArray<f64>, OptimizeError> {
    let (m, n) = sparsity.shape();
    if m != n {
        return Err(OptimizeError::ValueError(
            "Sparsity pattern must be square for Hessian computation".to_string(),
        ));
    }

    // Convert to dense for simplicity
    let dense = sparsity.to_array();
    let dense_transposed = dense.t().to_owned();

    // Create arrays for the triplets
    let mut data = Vec::new();
    let mut rows = Vec::new();
    let mut cols = Vec::new();

    // Combine the original and its transpose
    for i in 0..n {
        for j in 0..n {
            if dense[[i, j]] > 0.0 || dense_transposed[[i, j]] > 0.0 {
                rows.push(i);
                cols.push(j);
                data.push(1.0); // Binary sparsity pattern
            }
        }
    }

    // Create symmetric sparsity pattern
    Ok(CsrArray::from_triplets(&rows, &cols, &data, (n, n), false)?)
}

/// Fills the lower triangle of a Hessian matrix based on the upper triangle
fn fill_symmetric_hessian(upper: &CsrArray<f64>) -> Result<CsrArray<f64>, OptimizeError> {
    let (n, _) = upper.shape();
    if n != upper.shape().1 {
        return Err(OptimizeError::ValueError(
            "Hessian matrix must be square".to_string(),
        ));
    }

    // We need to create a new symmetric matrix from the upper triangular matrix

    // Convert the upper triangle matrix to dense temporarily
    let upper_dense = upper.to_array();

    // Create arrays for the triplets
    let mut data = Vec::new();
    let mut rows = Vec::new();
    let mut cols = Vec::new();

    // Collect all non-zero entries including the symmetric counterparts
    for i in 0..n {
        for j in 0..n {
            let value = upper_dense[[i, j]];
            if value != 0.0 {
                // Add the original element
                rows.push(i);
                cols.push(j);
                data.push(value);

                // If not on diagonal, add the symmetric element
                if i != j {
                    rows.push(j);
                    cols.push(i);
                    data.push(value);
                }
            }
        }
    }

    // Create new symmetric matrix
    let full = CsrArray::from_triplets(&rows, &cols, &data, (n, n), false)?;

    Ok(full)
}
