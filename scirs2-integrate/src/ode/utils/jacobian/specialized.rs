//! Specialized Jacobian approximation techniques
//!
//! This module provides specialized techniques for Jacobian approximation
//! for various types of ODE systems. These include methods for banded Jacobians,
//! sparse Jacobians, and systems with special structure.

use crate::common::IntegrateFloat;
use ndarray::{Array1, Array2, ArrayView1};

/// Computes Jacobian for a banded system with specified number of lower and upper diagonals
pub fn compute_banded_jacobian<F, Func>(
    f: &Func,
    t: F,
    y: &Array1<F>,
    f_current: &Array1<F>,
    lower: usize,
    upper: usize,
) -> Array2<F>
where
    F: IntegrateFloat,
    Func: Fn(F, ArrayView1<F>) -> Array1<F>,
{
    let n = y.len();
    let mut jac = Array2::<F>::zeros((n, n));
    let eps = F::from_f64(1e-8).unwrap();

    // Compute only the diagonals within the band
    for j in 0..n {
        // Define the range of rows affected by column j
        let row_start = j.saturating_sub(lower);
        let row_end = (j + upper + 1).min(n);

        // Only compute if there are rows in range
        if row_start < row_end {
            // Perturb the j-th component
            let mut y_perturbed = y.clone();
            let perturbation = eps * (F::one() + y[j].abs()).max(F::one());
            y_perturbed[j] += perturbation;

            // Evaluate function at perturbed point
            let f_perturbed = f(t, y_perturbed.view());

            // Compute entries in the band for this column
            for i in row_start..row_end {
                jac[[i, j]] = (f_perturbed[i] - f_current[i]) / perturbation;
            }
        }
    }

    jac
}

/// Computes Jacobian for a system with diagonal or block-diagonal structure
pub fn compute_diagonal_jacobian<F, Func>(
    f: &Func,
    t: F,
    y: &Array1<F>,
    f_current: &Array1<F>,
    block_size: usize,
) -> Array2<F>
where
    F: IntegrateFloat,
    Func: Fn(F, ArrayView1<F>) -> Array1<F>,
{
    let n = y.len();
    let mut jac = Array2::<F>::zeros((n, n));
    let eps = F::from_f64(1e-8).unwrap();

    // For each column, only compute the elements in the block-diagonal
    for j in 0..n {
        // Determine which block this column belongs to
        let block_idx = j / block_size;
        let block_start = block_idx * block_size;
        let block_end = (block_start + block_size).min(n);

        // Perturb the j-th component
        let mut y_perturbed = y.clone();
        let perturbation = eps * (F::one() + y[j].abs()).max(F::one());
        y_perturbed[j] += perturbation;

        // Evaluate function at perturbed point
        let f_perturbed = f(t, y_perturbed.view());

        // Compute only the elements in the same block
        for i in block_start..block_end {
            jac[[i, j]] = (f_perturbed[i] - f_current[i]) / perturbation;
        }
    }

    jac
}

/// Group variables based on their interactions to minimize function evaluations
pub fn compute_colored_jacobian<F, Func>(
    f: &Func,
    t: F,
    y: &Array1<F>,
    f_current: &Array1<F>,
    coloring: &[usize], // Each entry contains the color of the corresponding variable
) -> Array2<F>
where
    F: IntegrateFloat,
    Func: Fn(F, ArrayView1<F>) -> Array1<F>,
{
    let n = y.len();
    let mut jac = Array2::<F>::zeros((n, n));
    let eps = F::from_f64(1e-8).unwrap();

    // Determine the number of colors
    let max_color = coloring.iter().max().map_or(0, |&x| x) + 1;

    // For each color, perturb all variables of that color simultaneously
    for color in 0..max_color {
        // Create a perturbation vector that perturbs all variables of this color
        let mut y_perturbed = y.clone();
        let mut perturbations = vec![F::zero(); n];

        // Set perturbations for variables of this color
        for j in 0..n {
            if coloring[j] == color {
                let perturbation = eps * (F::one() + y[j].abs()).max(F::one());
                y_perturbed[j] += perturbation;
                perturbations[j] = perturbation;
            }
        }

        // Evaluate function with all variables of this color perturbed
        let f_perturbed = f(t, y_perturbed.view());

        // Compute Jacobian columns for variables of this color
        for j in 0..n {
            if coloring[j] == color && perturbations[j] > F::zero() {
                for i in 0..n {
                    jac[[i, j]] = (f_perturbed[i] - f_current[i]) / perturbations[j];
                }
            }
        }
    }

    jac
}

/// Generate a simple coloring for a banded matrix
pub fn generate_banded_coloring(n: usize, lower: usize, upper: usize) -> Vec<usize> {
    let bandwidth = lower + upper + 1;
    let mut coloring = vec![0; n];

    for (i, color) in coloring.iter_mut().enumerate().take(n) {
        // Assign a color (modulo bandwidth) to each variable
        // This ensures no two variables that could interact have the same color
        *color = i % bandwidth;
    }

    coloring
}

/// Update the Jacobian using Broyden's method (rank-1 update)
/// J_{k+1} = J_k + (df - J_k * dy) * dy^T / (dy^T * dy)
pub fn broyden_update<F>(jac: &mut Array2<F>, delta_y: &Array1<F>, delta_f: &Array1<F>)
where
    F: IntegrateFloat,
{
    let n = delta_y.len();

    // Compute J_k * dy
    let mut jac_dy = Array1::zeros(n);
    for i in 0..n {
        for j in 0..n {
            jac_dy[i] += jac[[i, j]] * delta_y[j];
        }
    }

    // Compute correction vector: df - J_k * dy
    let correction = delta_f - &jac_dy;

    // Compute denominator: dy^T * dy
    let dy_norm_squared = delta_y.iter().map(|&x| x * x).sum::<F>();

    // Apply update if denominator is not too small
    if dy_norm_squared > F::from_f64(1e-14).unwrap() {
        for i in 0..n {
            for j in 0..n {
                jac[[i, j]] += correction[i] * delta_y[j] / dy_norm_squared;
            }
        }
    }
}

/// Performs a block-update of the Jacobian using block structure
pub fn block_update<F>(
    jac: &mut Array2<F>,
    delta_y: &Array1<F>,
    delta_f: &Array1<F>,
    block_size: usize,
) where
    F: IntegrateFloat,
{
    let n = delta_y.len();
    let n_blocks = n.div_ceil(block_size);

    // Process each block separately
    for block in 0..n_blocks {
        let start = block * block_size;
        let end = (start + block_size).min(n);

        // Extract block components
        let mut block_dy = Array1::zeros(end - start);
        let mut block_df = Array1::zeros(end - start);

        for i in start..end {
            block_dy[i - start] = delta_y[i];
            block_df[i - start] = delta_f[i];
        }

        // Compute block_jac * block_dy
        let mut block_jac_dy = Array1::zeros(end - start);
        for i in 0..(end - start) {
            for j in 0..(end - start) {
                block_jac_dy[i] += jac[[i + start, j + start]] * block_dy[j];
            }
        }

        // Update the block using Broyden's formula
        let correction = &block_df - &block_jac_dy;
        let dy_norm_squared = block_dy.iter().map(|&x| x * x).sum::<F>();

        if dy_norm_squared > F::from_f64(1e-14).unwrap() {
            for i in 0..(end - start) {
                for j in 0..(end - start) {
                    jac[[i + start, j + start]] += correction[i] * block_dy[j] / dy_norm_squared;
                }
            }
        }
    }
}
