//! Parallel Jacobian computation for ODE systems
//!
//! This module provides functions for computing Jacobian matrices in parallel,
//! which can significantly speed up derivative calculations for large systems.
//!
//! To enable parallel Jacobian computation, activate the "parallel_jacobian" feature:
//! ```toml
//! [dependencies]
//! scirs2-integrate = { version = "0.1.0-alpha.3", features = ["parallel_jacobian"] }
//! ```
//!
//! Parallel computation is especially beneficial for:
//! - Large ODE systems (dimension > 20)
//! - Computationally expensive right-hand side functions
//! - Multi-core systems where serial computation would be a bottleneck

use crate::common::IntegrateFloat;
use crate::error::IntegrateResult;
use ndarray::{Array1, Array2, ArrayView1};

#[cfg(feature = "parallel_jacobian")]
use rayon::prelude::*;

/// Compute Jacobian matrix using parallel finite differences
///
/// This function divides the work of computing each column of the Jacobian
/// across multiple threads, which can provide significant speedup for large systems.
///
/// # Arguments
///
/// * `f` - Function to differentiate
/// * `t` - Time value
/// * `y` - State vector
/// * `f_current` - Current function evaluation at (t, y)
/// * `perturbation_scale` - Scaling factor for perturbation size
///
/// # Returns
///
/// Jacobian matrix (∂f/∂y)
///
/// # Features
///
/// Requires the "parallel_jacobian" feature to be enabled for actual parallel execution.
/// Falls back to serial execution if the feature is not enabled.
pub fn parallel_finite_difference_jacobian<F, Func>(
    f: &Func,
    t: F,
    y: &Array1<F>,
    f_current: &Array1<F>,
    perturbation_scale: F,
) -> IntegrateResult<Array2<F>>
where
    F: IntegrateFloat + Send + Sync,
    Func: Fn(F, ArrayView1<F>) -> Array1<F> + Sync,
{
    let n_dim = y.len();
    let mut jacobian = Array2::<F>::zeros((n_dim, n_dim));

    // Calculate base perturbation size
    let eps_base = F::from_f64(1e-8).unwrap() * perturbation_scale;

    #[cfg(feature = "parallel_jacobian")]
    {
        // Compute columns in parallel using rayon
        let columns: Vec<(usize, Array1<F>)> = (0..n_dim)
            .into_par_iter()
            .map(|j| {
                // Scale perturbation by variable magnitude
                let eps = eps_base * (F::one() + y[j].abs()).max(F::one());

                // Perturb the j-th component
                let mut y_perturbed = y.clone();
                y_perturbed[j] = y_perturbed[j] + eps;

                // Evaluate function at perturbed point
                let f_perturbed = f(t, y_perturbed.view());

                // Calculate the j-th column using finite differences
                let mut column = Array1::<F>::zeros(n_dim);
                for i in 0..n_dim {
                    column[i] = (f_perturbed[i] - f_current[i]) / eps;
                }

                (j, column)
            })
            .collect();

        // Assemble the Jacobian from columns
        for (j, column) in columns {
            for i in 0..n_dim {
                jacobian[[i, j]] = column[i];
            }
        }
    }

    #[cfg(not(feature = "parallel_jacobian"))]
    {
        // Fallback to serial implementation when feature is not enabled
        for j in 0..n_dim {
            // Scale perturbation by variable magnitude
            let eps = eps_base * (F::one() + y[j].abs()).max(F::one());

            // Perturb the j-th component
            let mut y_perturbed = y.clone();
            y_perturbed[j] += eps;

            // Evaluate function at perturbed point
            let f_perturbed = f(t, y_perturbed.view());

            // Calculate the j-th column using finite differences
            for i in 0..n_dim {
                jacobian[[i, j]] = (f_perturbed[i] - f_current[i]) / eps;
            }
        }
    }

    Ok(jacobian)
}

/// Compute sparse Jacobian matrix in parallel using coloring
///
/// This function uses graph coloring to identify independent columns of the
/// Jacobian that can be computed simultaneously, reducing the number of
/// function evaluations needed for sparse Jacobians.
///
/// # Arguments
///
/// * `f` - Function to differentiate
/// * `t` - Time value
/// * `y` - State vector
/// * `f_current` - Current function evaluation at (t, y)
/// * `sparsity_pattern` - Optional sparsity pattern (true for non-zero entries)
/// * `perturbation_scale` - Scaling factor for perturbation size
///
/// # Returns
///
/// Jacobian matrix (∂f/∂y)
///
/// # Features
///
/// Requires the "parallel_jacobian" feature to be enabled for actual parallel execution.
/// Falls back to a serial implementation if the feature is not enabled.
pub fn parallel_sparse_jacobian<F, Func>(
    f: &Func,
    t: F,
    y: &Array1<F>,
    f_current: &Array1<F>,
    sparsity_pattern: Option<&Array2<bool>>,
    perturbation_scale: F,
) -> IntegrateResult<Array2<F>>
where
    F: IntegrateFloat + Send + Sync,
    Func: Fn(F, ArrayView1<F>) -> Array1<F> + Sync,
{
    let n_dim = y.len();
    let mut jacobian = Array2::<F>::zeros((n_dim, n_dim));

    // Determine sparsity pattern and coloring
    let (pattern, colors) = if let Some(pattern) = sparsity_pattern {
        // Use provided sparsity pattern
        (pattern.clone(), greedy_coloring(pattern))
    } else {
        // Assume dense Jacobian
        let dense_pattern = Array2::<bool>::from_elem((n_dim, n_dim), true);
        // For dense matrices, each column needs its own color
        let dense_colors = (0..n_dim).collect::<Vec<_>>();
        (dense_pattern, dense_colors)
    };

    // Find maximum color
    let max_color = colors.iter().max().cloned().unwrap_or(0);

    // Base perturbation size
    let eps_base = F::from_f64(1e-8).unwrap() * perturbation_scale;

    #[cfg(feature = "parallel_jacobian")]
    {
        // Process each color group in parallel
        let color_results: Vec<_> = (0..=max_color)
            .into_par_iter()
            .map(|color| {
                // Find all columns with this color
                let columns_with_color: Vec<usize> = colors
                    .iter()
                    .enumerate()
                    .filter_map(|(j, &c)| if c == color { Some(j) } else { None })
                    .collect();

                if columns_with_color.is_empty() {
                    return Vec::new();
                }

                // Create a perturbation vector that perturbs all columns of this color
                let mut y_perturbed = y.clone();
                let mut perturbation_sizes = Vec::with_capacity(columns_with_color.len());

                for &j in &columns_with_color {
                    let eps = eps_base * (F::one() + y[j].abs()).max(F::one());
                    y_perturbed[j] = y_perturbed[j] + eps;
                    perturbation_sizes.push(eps);
                }

                // Evaluate function with perturbed values
                let f_perturbed = f(t, y_perturbed.view());

                // Extract columns for this color
                let mut color_columns = Vec::with_capacity(columns_with_color.len());

                for (idx, &j) in columns_with_color.iter().enumerate() {
                    let eps = perturbation_sizes[idx];

                    // Extract non-zero elements for this column based on sparsity pattern
                    let mut col_indices = Vec::new();
                    for i in 0..n_dim {
                        if pattern[[i, j]] {
                            col_indices.push(i);
                        }
                    }

                    // For each non-zero element, compute its value
                    let mut column_values = Vec::with_capacity(col_indices.len());
                    for &i in &col_indices {
                        let df = f_perturbed[i] - f_current[i];
                        column_values.push((i, j, df / eps));
                    }

                    color_columns.push(column_values);
                }

                // Flatten all columns for this color
                color_columns.into_iter().flatten().collect::<Vec<_>>()
            })
            .collect();

        // Combine all results into the Jacobian matrix
        for color_column in color_results {
            for (i, j, value) in color_column {
                jacobian[[i, j]] = value;
            }
        }
    }

    #[cfg(not(feature = "parallel_jacobian"))]
    {
        // Serial implementation of the coloring algorithm
        for color in 0..=max_color {
            // Find all columns with this color
            let columns_with_color: Vec<usize> = colors
                .iter()
                .enumerate()
                .filter_map(|(j, &c)| if c == color { Some(j) } else { None })
                .collect();

            if columns_with_color.is_empty() {
                continue;
            }

            // Create a perturbation vector that perturbs all columns of this color
            let mut y_perturbed = y.clone();
            let mut perturbation_sizes = Vec::with_capacity(columns_with_color.len());

            for &j in &columns_with_color {
                let eps = eps_base * (F::one() + y[j].abs()).max(F::one());
                y_perturbed[j] += eps;
                perturbation_sizes.push(eps);
            }

            // Evaluate function with perturbed values
            let f_perturbed = f(t, y_perturbed.view());

            // Process columns for this color
            for (idx, &j) in columns_with_color.iter().enumerate() {
                let eps = perturbation_sizes[idx];

                // Compute values for non-zero elements
                for i in 0..n_dim {
                    if pattern[[i, j]] {
                        let df = f_perturbed[i] - f_current[i];
                        jacobian[[i, j]] = df / eps;
                    }
                }
            }
        }
    }

    Ok(jacobian)
}

/// Graph coloring algorithm for parallel Jacobian computation
///
/// This function implements a greedy coloring algorithm to color the
/// columns of the Jacobian matrix such that no two columns with non-zero
/// elements in the same row share the same color.
///
/// # Arguments
///
/// * `sparsity_pattern` - Sparsity pattern of the Jacobian
///
/// # Returns
///
/// Vector of colors for each column
fn greedy_coloring(sparsity_pattern: &Array2<bool>) -> Vec<usize> {
    let (n_rows, n_cols) = sparsity_pattern.dim();

    // Initialize colors for all columns
    let mut colors = vec![usize::MAX; n_cols];

    // Process each column
    for j in 0..n_cols {
        // Find non-zero entries in this column
        let mut non_zero_rows = Vec::new();
        for i in 0..n_rows {
            if sparsity_pattern[[i, j]] {
                non_zero_rows.push(i);
            }
        }

        // Find colors used by conflicting columns
        let mut used_colors = Vec::new();
        for &row in &non_zero_rows {
            for k in 0..j {
                if sparsity_pattern[[row, k]] && colors[k] != usize::MAX {
                    used_colors.push(colors[k]);
                }
            }
        }

        // Find lowest unused color
        let mut color = 0;
        while used_colors.contains(&color) {
            color += 1;
        }

        // Assign color to this column
        colors[j] = color;
    }

    colors
}

/// Determine if parallel Jacobian computation should be used
///
/// This function uses heuristics to decide whether it's worth using
/// parallel computation for the Jacobian based on system size, sparsity, etc.
///
/// # Arguments
///
/// * `n_dim` - Dimension of the system
/// * `is_sparse` - Whether the Jacobian is known to be sparse
/// * `num_threads` - Number of available threads
///
/// # Returns
///
/// True if parallel computation is likely beneficial
pub fn should_use_parallel_jacobian(n_dim: usize, is_sparse: bool, num_threads: usize) -> bool {
    // Check if parallel_jacobian feature is enabled
    #[cfg(not(feature = "parallel_jacobian"))]
    {
        let _ = n_dim;
        let _ = is_sparse;
        let _ = num_threads;
        false // Parallel computation not available without the feature
    }

    #[cfg(feature = "parallel_jacobian")]
    {
        // Don't parallelize if only 1 thread available
        if num_threads <= 1 {
            return false;
        }

        // For small systems, overhead of parallelization likely exceeds benefits
        if n_dim < 20 {
            return false;
        }

        // For sparse systems, need to look at the specific sparsity pattern
        // to determine if parallelization will help
        if is_sparse {
            // If very large, parallelize even if sparse
            if n_dim > 100 {
                return true;
            }
            // For medium sized sparse systems, need more information
            return false;
        }

        // For dense systems, parallelize if large enough
        n_dim >= 20
    }
}

/// Struct to manage parallel Jacobian computation strategy
pub struct ParallelJacobianStrategy {
    /// Whether to use parallel computation
    pub use_parallel: bool,
    /// Whether the Jacobian is sparse
    pub is_sparse: bool,
    /// Sparsity pattern if known
    pub sparsity_pattern: Option<Array2<bool>>,
    /// Last computed Jacobian
    pub jacobian: Option<Array2<f64>>,
    /// Number of threads to use
    pub num_threads: usize,
}

impl ParallelJacobianStrategy {
    /// Create a new strategy object
    pub fn new(n_dim: usize, is_sparse: bool) -> Self {
        // Determine if parallel computation is available and beneficial
        #[cfg(feature = "parallel_jacobian")]
        let (use_parallel, num_threads) = {
            // Get number of available threads from rayon
            let threads = rayon::current_num_threads();
            // Check if parallel computation is worthwhile
            (
                should_use_parallel_jacobian(n_dim, is_sparse, threads),
                threads,
            )
        };

        #[cfg(not(feature = "parallel_jacobian"))]
        let (use_parallel, num_threads) = {
            let _ = n_dim;
            (false, 1)
        };

        ParallelJacobianStrategy {
            use_parallel,
            is_sparse,
            sparsity_pattern: None,
            jacobian: None,
            num_threads,
        }
    }

    /// Set the sparsity pattern
    pub fn set_sparsity_pattern(&mut self, pattern: Array2<bool>) {
        self.sparsity_pattern = Some(pattern);
        self.is_sparse = true;
    }

    /// Compute the Jacobian matrix
    pub fn compute_jacobian<F, Func>(
        &mut self,
        f: &Func,
        t: F,
        y: &Array1<F>,
        f_current: &Array1<F>,
        perturbation_scale: F,
    ) -> IntegrateResult<Array2<F>>
    where
        F: IntegrateFloat + Send + Sync,
        Func: Fn(F, ArrayView1<F>) -> Array1<F> + Sync,
    {
        // Decide which algorithm to use
        let jacobian = if !self.use_parallel {
            // Sequential computation
            crate::ode::utils::common::finite_difference_jacobian(
                f,
                t,
                y,
                f_current,
                perturbation_scale,
            )
        } else {
            // We know the parallel_jacobian feature is enabled if we get here
            if self.is_sparse && self.sparsity_pattern.is_some() {
                // Parallel sparse computation
                parallel_sparse_jacobian(
                    f,
                    t,
                    y,
                    f_current,
                    self.sparsity_pattern.as_ref(),
                    perturbation_scale,
                )?
            } else {
                // Parallel dense computation
                parallel_finite_difference_jacobian(f, t, y, f_current, perturbation_scale)?
            }
        };

        // Store result
        let jacobian_f64 = jacobian.mapv(|x| x.to_f64().unwrap_or(0.0));
        self.jacobian = Some(jacobian_f64);

        // Return the jacobian
        Ok(jacobian)
    }

    /// Check if parallel computation is available and enabled
    pub fn is_parallel_available() -> bool {
        #[cfg(feature = "parallel_jacobian")]
        {
            true
        }

        #[cfg(not(feature = "parallel_jacobian"))]
        {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array1, ArrayView1};

    // Test function for Jacobian computation
    fn test_func(_t: f64, y: ArrayView1<f64>) -> Array1<f64> {
        array![y[0] * y[0] + y[1], y[0] * y[1] + y[2], y[2] * y[2] - y[0]]
    }

    // Analytic Jacobian for test function
    fn analytic_jacobian(y: &[f64]) -> Array2<f64> {
        array![
            [2.0 * y[0], 1.0, 0.0],
            [y[1], y[0], 1.0],
            [-1.0, 0.0, 2.0 * y[2]]
        ]
    }

    #[test]
    fn test_parallel_jacobian() {
        let y = array![1.0, 2.0, 3.0];
        let t = 0.0;
        let f_current = test_func(t, y.view());

        // Compute parallel Jacobian
        let numerical_jac =
            parallel_finite_difference_jacobian(&test_func, t, &y, &f_current, 1.0).unwrap();

        // Compute analytic Jacobian
        let analytic_jac = analytic_jacobian(&[1.0, 2.0, 3.0]);

        // Check that they match within tolerance
        for i in 0..3 {
            for j in 0..3 {
                assert!((numerical_jac[[i, j]] - analytic_jac[[i, j]]).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn test_sparse_jacobian() {
        let y = array![1.0, 2.0, 3.0];
        let t = 0.0;
        let f_current = test_func(t, y.view());

        // Create sparsity pattern
        let sparsity_pattern = array![[true, true, false], [true, true, true], [true, false, true]];

        // Compute sparse Jacobian
        let numerical_jac =
            parallel_sparse_jacobian(&test_func, t, &y, &f_current, Some(&sparsity_pattern), 1.0)
                .unwrap();

        // Compute analytic Jacobian
        let analytic_jac = analytic_jacobian(&[1.0, 2.0, 3.0]);

        // Check that non-zero entries match within tolerance
        for i in 0..3 {
            for j in 0..3 {
                if sparsity_pattern[[i, j]] {
                    assert!((numerical_jac[[i, j]] - analytic_jac[[i, j]]).abs() < 1e-5);
                } else {
                    assert_eq!(numerical_jac[[i, j]], 0.0);
                }
            }
        }
    }

    #[test]
    fn test_greedy_coloring() {
        // Simple test case
        let sparsity_pattern = array![
            [true, true, false, false],
            [true, false, true, false],
            [false, true, true, true]
        ];

        let colors = greedy_coloring(&sparsity_pattern);

        // Verify that no adjacent columns have the same color
        for i in 0..sparsity_pattern.nrows() {
            for j1 in 0..sparsity_pattern.ncols() {
                if !sparsity_pattern[[i, j1]] {
                    continue;
                }

                for j2 in (j1 + 1)..sparsity_pattern.ncols() {
                    if sparsity_pattern[[i, j2]] {
                        assert_ne!(colors[j1], colors[j2]);
                    }
                }
            }
        }
    }
}
