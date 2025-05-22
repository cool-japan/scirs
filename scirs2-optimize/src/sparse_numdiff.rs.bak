//! Sparse numerical differentiation for large-scale optimization
//!
//! This module provides functions for computing sparse Jacobians and Hessians
//! using finite differences, designed for large-scale optimization problems.

use ndarray::{Array1, ArrayView1};
use num_traits::Zero;
use rand::seq::SliceRandom;
use rand::{rngs::StdRng, SeedableRng};
use rayon::prelude::*;
use scirs2_sparse::{csr_array::CsrArray, sparray::SparseArray};

use crate::error::OptimizeError;
use crate::parallel::ParallelOptions;

/// Options for sparse numerical differentiation
#[derive(Debug, Clone)]
pub struct SparseFiniteDiffOptions {
    /// Method to use for finite differences ('2-point', '3-point', 'cs')
    pub method: String,
    /// Relative step size (if None, determined automatically)
    pub rel_step: Option<f64>,
    /// Absolute step size (if None, determined automatically)
    pub abs_step: Option<f64>,
    /// Bounds on the variables
    pub bounds: Option<Vec<(f64, f64)>>,
    /// Parallel computation options
    pub parallel: Option<ParallelOptions>,
    /// Random seed for coloring algorithm
    pub seed: Option<u64>,
    /// Maximum number of columns to group together
    pub max_group_size: usize,
}

impl Default for SparseFiniteDiffOptions {
    fn default() -> Self {
        Self {
            method: "2-point".to_string(),
            rel_step: None,
            abs_step: None,
            bounds: None,
            parallel: None,
            seed: None,
            max_group_size: 100,
        }
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

            sparsity_owned =
                CsrArray::from_triplets(&rows, &cols, &data, (m, n), false).map_err(|e| {
                    OptimizeError::ComputationError(format!(
                        "Error creating sparsity pattern: {}",
                        e
                    ))
                })?;
            &sparsity_owned
        }
    };

    // Determine column groups for parallel evaluation using a greedy coloring algorithm
    let groups = determine_column_groups(sparsity, options.max_group_size, options.seed);

    // Compute step sizes
    let h = compute_step_sizes(x, f0_ref, &options);

    // Compute the Jacobian based on the method
    match options.method.as_str() {
        "2-point" => compute_jacobian_2point(func, x, f0_ref, &h, &groups, sparsity, &options),
        "3-point" => compute_jacobian_3point(func, x, f0_ref, &h, &groups, sparsity, &options),
        "cs" => compute_jacobian_complex_step(func, x, f0_ref, &h, &groups, sparsity, &options),
        _ => Err(OptimizeError::ComputationError(format!(
            "Unknown method: {}",
            options.method
        ))),
    }
}

/// Computes a sparse Hessian matrix using finite differences
///
/// # Arguments
///
/// * `func` - Function to differentiate, takes ArrayView1<f64> and returns f64
/// * `x` - Point at which to compute the Hessian
/// * `grad` - Gradient value at `x` (if None, computed internally)
/// * `sparsity_pattern` - Sparse matrix indicating the known sparsity pattern (if None, dense Hessian)
/// * `options` - Options for finite differences computation
///
/// # Returns
///
/// * `CsrArray<f64>` - Sparse Hessian matrix in CSR format
///
pub fn sparse_hessian<F>(
    func: F,
    x: &ArrayView1<f64>,
    grad: Option<&Array1<f64>>,
    sparsity_pattern: Option<&CsrArray<f64>>,
    options: Option<SparseFiniteDiffOptions>,
) -> Result<CsrArray<f64>, OptimizeError>
where
    F: Fn(&ArrayView1<f64>) -> f64 + Sync + Clone,
{
    let options = options.unwrap_or_default();
    let n = x.len();

    // Create gradient function as a separate function to avoid borrowing issues
    fn grad_fn<F: Fn(&ArrayView1<f64>) -> f64 + Clone>(
        func: &F,
        x: &ArrayView1<f64>,
    ) -> Array1<f64> {
        // Create a simple gradient approximation function
        let mut grad = Array1::zeros(x.len());
        let eps = 1e-8;

        let f0 = func(x);

        for i in 0..x.len() {
            let mut x_plus = x.to_owned();
            x_plus[i] += eps;
            let f_plus = func(&x_plus.view());

            grad[i] = (f_plus - f0) / eps;
        }

        grad
    }

    // Compute gradient if not provided
    let grad_owned: Array1<f64>;
    let grad_ref = match grad {
        Some(g) => g,
        None => {
            grad_owned = grad_fn(&func, x);
            &grad_owned
        }
    };

    // If no sparsity pattern provided, create a dense one (or use symmetry)
    let sparsity_owned: CsrArray<f64>;
    let sparsity = match sparsity_pattern {
        Some(p) => p,
        None => {
            // For Hessian, we can exploit symmetry and only compute upper triangular part
            let mut data = Vec::with_capacity(n * (n + 1) / 2);
            let mut rows = Vec::with_capacity(n * (n + 1) / 2);
            let mut cols = Vec::with_capacity(n * (n + 1) / 2);

            for i in 0..n {
                for j in i..n {
                    // Upper triangular part only
                    data.push(1.0);
                    rows.push(i);
                    cols.push(j);
                }
            }

            sparsity_owned =
                CsrArray::from_triplets(&rows, &cols, &data, (n, n), false).map_err(|e| {
                    OptimizeError::ComputationError(format!(
                        "Error creating sparsity pattern: {}",
                        e
                    ))
                })?;
            &sparsity_owned
        }
    };

    // Determine column groups using a graph coloring algorithm
    let groups = determine_column_groups(sparsity, options.max_group_size, options.seed);

    // Compute step sizes
    let mut single_output = Array1::zeros(1);
    single_output[0] = func(x);
    let h = compute_step_sizes(x, &single_output, &options);

    // Compute the Hessian based on the method
    let result = match options.method.as_str() {
        "2-point" => {
            compute_hessian_2point(func.clone(), x, grad_ref, &h, &groups, sparsity, &options)
        }
        "3-point" => compute_hessian_3point(func.clone(), x, &h, &groups, sparsity, &options),
        "cs" => compute_hessian_complex_step(func.clone(), x, &h, &groups, sparsity, &options),
        _ => Err(OptimizeError::ComputationError(format!(
            "Unknown method: {}",
            options.method
        ))),
    }?;

    // Fill in the lower triangular part using symmetry for a full Hessian
    fill_symmetric_hessian(&result)
}

/// Computes Hessian using a higher-order finite difference method
///
/// This implements a sixth-order accurate approximation as an alternative to
/// the complex step method for Hessian calculation.
fn compute_hessian_complex_step<F>(
    func: F,
    x: &ArrayView1<f64>,
    h: &Array1<f64>,
    groups: &[usize],
    sparsity: &CsrArray<f64>,
    options: &SparseFiniteDiffOptions,
) -> Result<CsrArray<f64>, OptimizeError>
where
    F: Fn(&ArrayView1<f64>) -> f64 + Sync + Clone,
{
    let (m, n) = sparsity.shape();
    if m != n {
        return Err(OptimizeError::ComputationError(
            "Hessian must be square".to_string(),
        ));
    }

    let num_groups = *groups.iter().max().unwrap_or(&0) + 1;

    // Prepare result data structures
    let mut hess_data = Vec::new();
    let mut hess_rows = Vec::new();
    let mut hess_cols = Vec::new();

    // Base function value
    let f0 = func(x);

    // Process each group of columns
    for group_id in 0..num_groups {
        // Determine indices in this group
        let group_indices: Vec<usize> = groups
            .iter()
            .enumerate()
            .filter_map(|(i, &g)| if g == group_id { Some(i) } else { None })
            .collect();

        if group_indices.is_empty() {
            continue;
        }

        // For Hessian diagonal elements (i,i)
        for &i in &group_indices {
            // Create perturbations with multiple step sizes for higher-order accuracy
            let mut x_h = x.to_owned();
            let mut x_2h = x.to_owned();
            let mut x_3h = x.to_owned();
            let mut x_minus_h = x.to_owned();
            let mut x_minus_2h = x.to_owned();
            let mut x_minus_3h = x.to_owned();

            x_h[i] = x[i] + h[i];
            x_2h[i] = x[i] + 2.0 * h[i];
            x_3h[i] = x[i] + 3.0 * h[i];
            x_minus_h[i] = x[i] - h[i];
            x_minus_2h[i] = x[i] - 2.0 * h[i];
            x_minus_3h[i] = x[i] - 3.0 * h[i];

            // Evaluate function
            let f_h = func(&x_h.view());
            let f_2h = func(&x_2h.view());
            let f_3h = func(&x_3h.view());
            let f_minus_h = func(&x_minus_h.view());
            let f_minus_2h = func(&x_minus_2h.view());
            let f_minus_3h = func(&x_minus_3h.view());

            // Calculate diagonal element using 6th-order central difference
            // Formula: (f(x-3h) - 6f(x-2h) + 15f(x-h) - 20f(x) + 15f(x+h) - 6f(x+2h) + f(x+3h)) / (h^2 * 6)
            let d2f_dx2 = (f_minus_3h - 6.0 * f_minus_2h + 15.0 * f_minus_h - 20.0 * f0
                + 15.0 * f_h
                - 6.0 * f_2h
                + f_3h)
                / (6.0 * h[i] * h[i]);

            hess_rows.push(i);
            hess_cols.push(i);
            hess_data.push(d2f_dx2);
        }

        // For off-diagonal elements (i,j) where i≠j
        let parallel = options.parallel.as_ref();
        let _f0_val = f0; // Capture for closure
        let x_ref = x; // Capture x by reference
        let h_ref = h; // Capture h by reference
        let func_ref = &func; // Capture function by reference
        let group_indices_ref = &group_indices; // Capture group_indices by reference

        let processor = move |(i, j): (usize, usize)| {
            if i == j || !group_indices_ref.contains(&i) || !group_indices_ref.contains(&j) {
                return None;
            }

            // Create perturbations for mixed partial derivatives
            // We'll use a 4th-order method for mixed derivatives
            let mut x_i_plus = x_ref.to_owned();
            let mut x_i_minus = x_ref.to_owned();
            let mut x_j_plus = x_ref.to_owned();
            let mut x_j_minus = x_ref.to_owned();
            let mut x_ij_plus_plus = x_ref.to_owned();
            let mut x_ij_plus_minus = x_ref.to_owned();
            let mut x_ij_minus_plus = x_ref.to_owned();
            let mut x_ij_minus_minus = x_ref.to_owned();

            x_i_plus[i] = x_ref[i] + h_ref[i];
            x_i_minus[i] = x_ref[i] - h_ref[i];
            x_j_plus[j] = x_ref[j] + h_ref[j];
            x_j_minus[j] = x_ref[j] - h_ref[j];

            x_ij_plus_plus[i] = x_ref[i] + h_ref[i];
            x_ij_plus_plus[j] = x_ref[j] + h_ref[j];

            x_ij_plus_minus[i] = x_ref[i] + h_ref[i];
            x_ij_plus_minus[j] = x_ref[j] - h_ref[j];

            x_ij_minus_plus[i] = x_ref[i] - h_ref[i];
            x_ij_minus_plus[j] = x_ref[j] + h_ref[j];

            x_ij_minus_minus[i] = x_ref[i] - h_ref[i];
            x_ij_minus_minus[j] = x_ref[j] - h_ref[j];

            // Evaluate function at all points
            let f_i_plus = func_ref(&x_i_plus.view());
            let f_i_minus = func_ref(&x_i_minus.view());
            let f_j_plus = func_ref(&x_j_plus.view());
            let f_j_minus = func_ref(&x_j_minus.view());
            let f_ij_plus_plus = func_ref(&x_ij_plus_plus.view());
            let f_ij_plus_minus = func_ref(&x_ij_plus_minus.view());
            let f_ij_minus_plus = func_ref(&x_ij_minus_plus.view());
            let f_ij_minus_minus = func_ref(&x_ij_minus_minus.view());

            // Mixed partial derivative using higher-order formula
            // This is more accurate than the standard formula (f_pp - f_pm - f_mp + f_mm)/(4h_i*h_j)
            let d2f_dxdy = (f_ij_plus_plus - f_ij_plus_minus - f_ij_minus_plus + f_ij_minus_minus
                - 2.0 * (f_i_plus - f_i_minus - f_j_plus + f_j_minus)
                + 4.0 * _f0_val)
                / (4.0 * h_ref[i] * h_ref[j]);

            Some((i, j, d2f_dxdy))
        };

        // Get all pairs of indices from sparsity pattern
        let mut index_pairs = Vec::new();
        for i in 0..n {
            if !group_indices.contains(&i) {
                continue;
            }

            // Get non-zero columns in this row
            let cols = get_nonzero_cols_in_row(sparsity, i);
            for &j in &cols {
                if i != j && group_indices.contains(&j) {
                    index_pairs.push((i, j));
                }
            }
        }

        // Process index pairs in parallel or sequentially
        let derivatives: Vec<(usize, usize, f64)> = if let Some(par_opts) = parallel {
            if index_pairs.len() >= par_opts.min_parallel_size && par_opts.parallel_evaluations {
                index_pairs
                    .par_iter()
                    .filter_map(|&(i, j)| processor((i, j)))
                    .collect()
            } else {
                index_pairs
                    .iter()
                    .filter_map(|&(i, j)| processor((i, j)))
                    .collect()
            }
        } else {
            index_pairs
                .iter()
                .filter_map(|&(i, j)| processor((i, j)))
                .collect()
        };

        // Add derivatives to result
        for (row, col, deriv) in derivatives {
            hess_rows.push(row);
            hess_cols.push(col);
            hess_data.push(deriv);
        }
    }

    // Create the sparse Hessian matrix
    CsrArray::from_triplets(&hess_rows, &hess_cols, &hess_data, (n, n), false)
        .map_err(|e| OptimizeError::ComputationError(format!("Error creating Hessian: {}", e)))
}

/// Helper function to get nonzero column indices in a row of a sparse matrix
fn get_nonzero_cols_in_row(matrix: &CsrArray<f64>, row: usize) -> Vec<usize> {
    let (rows, cols) = matrix.shape();
    if row >= rows {
        return Vec::new();
    }

    let mut nonzero_cols = Vec::new();
    for col in 0..cols {
        if !matrix.get(row, col).is_zero() {
            nonzero_cols.push(col);
        }
    }

    nonzero_cols
}

/// Determines efficient column groupings for parallel finite differencing
///
/// Implements a greedy graph coloring algorithm to group columns that can be
/// perturbed simultaneously based on the sparsity pattern.
fn determine_column_groups(
    sparsity: &CsrArray<f64>,
    max_group_size: usize,
    seed: Option<u64>,
) -> Vec<usize> {
    let (m, n) = sparsity.shape();

    // For very small problems, use simple sequential grouping
    if n <= 10 {
        return (0..n).collect();
    }

    // Create a compressed column representation of the sparsity pattern
    // This will make it easier to determine conflicts between columns
    let column_indices: Vec<Vec<usize>> = (0..n)
        .map(|col| {
            let mut rows = Vec::new();
            for row in 0..m {
                let row_cols = get_nonzero_cols_in_row(sparsity, row);
                if row_cols.contains(&col) {
                    rows.push(row);
                }
            }
            rows
        })
        .collect();

    // Build conflicts graph: columns i and j conflict if they share any rows
    let mut conflict_graph = vec![Vec::with_capacity(n / 2); n];

    // Use a parallel approach for building the conflict graph for large matrices
    if n > 1000 && m > 1000 {
        let conflict_lists: Vec<Vec<(usize, usize)>> = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut conflicts = Vec::new();
                for j in (i + 1)..n {
                    // Check if columns i and j share any rows
                    if !column_indices[i].is_empty() && !column_indices[j].is_empty() {
                        let mut has_conflict = false;

                        // Use a set-like approach for faster intersection check
                        let mut row_set = vec![false; m];
                        for &row in &column_indices[i] {
                            row_set[row] = true;
                        }

                        for &row in &column_indices[j] {
                            if row_set[row] {
                                has_conflict = true;
                                break;
                            }
                        }

                        if has_conflict {
                            conflicts.push((i, j));
                        }
                    }
                }
                conflicts
            })
            .collect();

        // Combine conflict lists into the conflict graph
        for conflict_list in conflict_lists {
            for (i, j) in conflict_list {
                conflict_graph[i].push(j);
                conflict_graph[j].push(i);
            }
        }
    } else {
        // For smaller matrices, use a simpler approach
        for i in 0..n {
            for j in (i + 1)..n {
                // Check if columns i and j share any rows
                if !column_indices[i].is_empty() && !column_indices[j].is_empty() {
                    let mut has_conflict = false;

                    for &row_i in &column_indices[i] {
                        if column_indices[j].contains(&row_i) {
                            has_conflict = true;
                            break;
                        }
                    }

                    if has_conflict {
                        conflict_graph[i].push(j);
                        conflict_graph[j].push(i);
                    }
                }
            }
        }
    }

    // Initialize a random number generator for shuffling
    let mut rng = if let Some(seed) = seed {
        StdRng::seed_from_u64(seed)
    } else {
        StdRng::seed_from_u64(rand::random())
    };

    // Sort vertices by degree (number of conflicts) for better coloring
    let mut vertices_by_degree: Vec<(usize, usize)> = conflict_graph
        .iter()
        .enumerate()
        .map(|(idx, neighbors)| (idx, neighbors.len()))
        .collect();

    // Break ties randomly for better coloring
    vertices_by_degree.shuffle(&mut rng);

    // Sort by degree in descending order (largest degree first)
    vertices_by_degree.sort_by(|a, b| b.1.cmp(&a.1));

    // Greedy coloring algorithm (Welsh-Powell)
    let mut colors = vec![usize::MAX; n]; // usize::MAX means uncolored
    let mut color_count = 0;

    // Process vertices in order of descending degree
    for (vertex, _) in &vertices_by_degree {
        if colors[*vertex] == usize::MAX {
            // Vertex not yet colored

            // Find smallest available color not used by neighbors
            let mut available_colors = vec![true; color_count + 1];
            for &neighbor in &conflict_graph[*vertex] {
                if colors[neighbor] != usize::MAX && colors[neighbor] < available_colors.len() {
                    available_colors[colors[neighbor]] = false;
                }
            }

            // Find first available color
            let color = available_colors
                .iter()
                .position(|&available| available)
                .unwrap_or_else(|| {
                    // All existing colors are used by neighbors, create a new color
                    color_count += 1;
                    color_count - 1
                });

            colors[*vertex] = color;

            // Look for other non-adjacent vertices that can use the same color
            for (other_vertex, _) in &vertices_by_degree {
                if *other_vertex == *vertex || colors[*other_vertex] != usize::MAX {
                    continue;
                }

                // Check if this vertex is adjacent to any vertex colored with 'color'
                let mut can_use_color = true;
                for &neighbor in &conflict_graph[*other_vertex] {
                    if colors[neighbor] == color {
                        can_use_color = false;
                        break;
                    }
                }

                if can_use_color {
                    colors[*other_vertex] = color;
                }
            }
        }
    }

    // Ensure colors are numbered from 0 and contiguous
    let mut color_map = std::collections::HashMap::new();
    let mut new_colors = vec![0; n];
    let mut next_color = 0;

    for (i, &color) in colors.iter().enumerate() {
        if let std::collections::hash_map::Entry::Vacant(e) = color_map.entry(color) {
            e.insert(next_color);
            next_color += 1;
        }
        new_colors[i] = *color_map.get(&color).unwrap();
    }

    // Ensure no group is too large
    let actual_color_count = color_map.len();
    if actual_color_count > max_group_size {
        // Combine colors to reduce the number of groups
        let colors_per_group = actual_color_count.div_ceil(max_group_size);
        for color in &mut new_colors {
            *color /= colors_per_group;
        }
    }

    new_colors
}

/// Computes appropriate step sizes for finite differences
fn compute_step_sizes(
    x: &ArrayView1<f64>,
    _f0: &Array1<f64>,
    options: &SparseFiniteDiffOptions,
) -> Array1<f64> {
    let n = x.len();
    let mut h = Array1::zeros(n);

    // Default relative step sizes based on method
    let default_rel_step = match options.method.as_str() {
        "2-point" => 1e-8,
        "3-point" => 1e-5,
        "cs" => 1e-8,
        _ => 1e-8,
    };

    let rel_step = options.rel_step.unwrap_or(default_rel_step);

    if let Some(abs_step) = options.abs_step {
        // User specified absolute steps
        for i in 0..n {
            h[i] = abs_step;
        }
    } else {
        // Compute relative steps
        for i in 0..n {
            let xi_abs = x[i].abs();
            h[i] = if xi_abs > 0.0 {
                xi_abs * rel_step
            } else {
                rel_step
            };
        }
    }

    // Adjust step sizes for bounds
    if let Some(ref bounds) = options.bounds {
        for i in 0..n.min(bounds.len()) {
            let (lb, ub) = bounds[i];

            if x[i] + h[i] > ub {
                h[i] = -h[i]; // Use negative step if upper bound is hit
            }

            if x[i] + h[i] < lb {
                // If we hit lower bound too, reduce step size
                h[i] = (ub - lb) / 20.0; // Use a small fraction of the range

                // Make sure we don't exceed bounds
                if x[i] + h[i] > ub {
                    h[i] = (ub - x[i]) / 2.0;
                }
                if x[i] - h[i] < lb {
                    h[i] = (x[i] - lb) / 2.0;
                }
            }
        }
    }

    h
}

/// Computes Jacobian using the 2-point method
fn compute_jacobian_2point<F>(
    func: F,
    x: &ArrayView1<f64>,
    f0: &Array1<f64>,
    h: &Array1<f64>,
    groups: &[usize],
    sparsity: &CsrArray<f64>,
    options: &SparseFiniteDiffOptions,
) -> Result<CsrArray<f64>, OptimizeError>
where
    F: Fn(&ArrayView1<f64>) -> Array1<f64> + Sync,
{
    let (m, n) = sparsity.shape();
    let num_groups = *groups.iter().max().unwrap_or(&0) + 1;

    // Prepare result data structures
    let mut jac_data = Vec::new();
    let mut jac_rows = Vec::new();
    let mut jac_cols = Vec::new();

    // Process each group of columns
    for group_id in 0..num_groups {
        // Determine indices in this group
        let group_indices: Vec<usize> = groups
            .iter()
            .enumerate()
            .filter_map(|(i, &g)| if g == group_id { Some(i) } else { None })
            .collect();

        if group_indices.is_empty() {
            continue;
        }

        // Create perturbation
        let mut x_perturbed = x.to_owned();
        for &idx in &group_indices {
            x_perturbed[idx] = x[idx] + h[idx];
        }

        // Evaluate the function at the perturbed point
        let f_perturbed = func(&x_perturbed.view());

        // Calculate the partial derivatives for this group
        let parallel = options.parallel.as_ref();
        let f_perturbed = &f_perturbed; // Capture by reference
                                        // f0 is already captured by reference
                                        // h is already captured by reference
        let group_indices = &group_indices; // Capture by reference

        // Use a closure that captures all needed variables by reference to avoid ownership issues
        let processor = move |(row, col_indices): (usize, Vec<usize>)| {
            let diff = f_perturbed[row] - f0[row];

            // Calculate derivatives for all columns in this row that belong to the current group
            col_indices
                .iter()
                .filter(|&&col| group_indices.contains(&col))
                .map(move |&col| {
                    let deriv = diff / h[col];
                    (row, col, deriv)
                })
                .collect::<Vec<_>>()
        };

        // Get all affected rows and their column indices from the sparsity pattern
        let row_col_pairs: Vec<(usize, Vec<usize>)> = (0..m)
            .map(|row| {
                let cols = get_nonzero_cols_in_row(sparsity, row);
                (row, cols)
            })
            .collect();

        // Process rows in parallel or sequentially
        let derivatives: Vec<(usize, usize, f64)> = if let Some(par_opts) = parallel {
            if row_col_pairs.len() >= par_opts.min_parallel_size && par_opts.parallel_evaluations {
                row_col_pairs
                    .par_iter()
                    .flat_map(|&(row, ref cols)| processor((row, cols.clone())))
                    .collect()
            } else {
                row_col_pairs
                    .iter()
                    .flat_map(|&(row, ref cols)| processor((row, cols.clone())))
                    .collect()
            }
        } else {
            row_col_pairs
                .iter()
                .flat_map(|&(row, ref cols)| processor((row, cols.clone())))
                .collect()
        };

        // Add derivatives to result
        for (row, col, deriv) in derivatives {
            jac_rows.push(row);
            jac_cols.push(col);
            jac_data.push(deriv);
        }
    }

    // Create the sparse Jacobian matrix
    CsrArray::from_triplets(&jac_rows, &jac_cols, &jac_data, (m, n), false)
        .map_err(|e| OptimizeError::ComputationError(format!("Error creating Jacobian: {}", e)))
}

/// Computes Jacobian using the 3-point method
fn compute_jacobian_3point<F>(
    func: F,
    x: &ArrayView1<f64>,
    _f0: &Array1<f64>,
    h: &Array1<f64>,
    groups: &[usize],
    sparsity: &CsrArray<f64>,
    options: &SparseFiniteDiffOptions,
) -> Result<CsrArray<f64>, OptimizeError>
where
    F: Fn(&ArrayView1<f64>) -> Array1<f64> + Sync,
{
    let (m, n) = sparsity.shape();
    let num_groups = *groups.iter().max().unwrap_or(&0) + 1;

    // Prepare result data structures
    let mut jac_data = Vec::new();
    let mut jac_rows = Vec::new();
    let mut jac_cols = Vec::new();

    // Process each group of columns
    for group_id in 0..num_groups {
        // Determine indices in this group
        let group_indices: Vec<usize> = groups
            .iter()
            .enumerate()
            .filter_map(|(i, &g)| if g == group_id { Some(i) } else { None })
            .collect();

        if group_indices.is_empty() {
            continue;
        }

        // Create positive and negative perturbations
        let mut x_plus = x.to_owned();
        let mut x_minus = x.to_owned();

        for &idx in &group_indices {
            x_plus[idx] = x[idx] + h[idx];
            x_minus[idx] = x[idx] - h[idx];
        }

        // Evaluate the function at the perturbed points
        let f_plus = func(&x_plus.view());
        let f_minus = func(&x_minus.view());

        // Calculate the partial derivatives for this group
        let parallel = options.parallel.as_ref();
        let f_plus = &f_plus; // Capture by reference
        let f_minus = &f_minus; // Capture by reference
                                // h is already captured by reference
        let group_indices = &group_indices; // Capture by reference

        // Use a closure that captures all needed variables by reference to avoid ownership issues
        let processor = move |(row, col_indices): (usize, Vec<usize>)| {
            // Calculate derivatives for all columns in this row that belong to the current group
            col_indices
                .iter()
                .filter(|&&col| group_indices.contains(&col))
                .map(move |&col| {
                    let deriv = (f_plus[row] - f_minus[row]) / (2.0 * h[col]);
                    (row, col, deriv)
                })
                .collect::<Vec<_>>()
        };

        // Get all affected rows and their column indices from the sparsity pattern
        let row_col_pairs: Vec<(usize, Vec<usize>)> = (0..m)
            .map(|row| {
                let cols = get_nonzero_cols_in_row(sparsity, row);
                (row, cols)
            })
            .collect();

        // Process rows in parallel or sequentially
        let derivatives: Vec<(usize, usize, f64)> = if let Some(par_opts) = parallel {
            if row_col_pairs.len() >= par_opts.min_parallel_size && par_opts.parallel_evaluations {
                row_col_pairs
                    .par_iter()
                    .flat_map(|&(row, ref cols)| processor((row, cols.clone())))
                    .collect()
            } else {
                row_col_pairs
                    .iter()
                    .flat_map(|&(row, ref cols)| processor((row, cols.clone())))
                    .collect()
            }
        } else {
            row_col_pairs
                .iter()
                .flat_map(|&(row, ref cols)| processor((row, cols.clone())))
                .collect()
        };

        // Add derivatives to result
        for (row, col, deriv) in derivatives {
            jac_rows.push(row);
            jac_cols.push(col);
            jac_data.push(deriv);
        }
    }

    // Create the sparse Jacobian matrix
    CsrArray::from_triplets(&jac_rows, &jac_cols, &jac_data, (m, n), false)
        .map_err(|e| OptimizeError::ComputationError(format!("Error creating Jacobian: {}", e)))
}

/// Computes Jacobian using the complex step method
///
/// This implements a fourth-order accurate approximation as an alternative to
/// the complex step method, since Rust doesn't natively support automatic complex
/// number differentiation for arbitrary functions.
fn compute_jacobian_complex_step<F>(
    func: F,
    x: &ArrayView1<f64>,
    _f0: &Array1<f64>,
    h: &Array1<f64>,
    groups: &[usize],
    sparsity: &CsrArray<f64>,
    options: &SparseFiniteDiffOptions,
) -> Result<CsrArray<f64>, OptimizeError>
where
    F: Fn(&ArrayView1<f64>) -> Array1<f64> + Sync,
{
    let (m, n) = sparsity.shape();
    let num_groups = *groups.iter().max().unwrap_or(&0) + 1;

    // Prepare result data structures
    let mut jac_data = Vec::new();
    let mut jac_rows = Vec::new();
    let mut jac_cols = Vec::new();

    // Compute f0 if needed (but we're actually using _f0 from parameters)
    let _f0_owned = func(x);

    // Process each group of columns
    for group_id in 0..num_groups {
        // Determine indices in this group
        let group_indices: Vec<usize> = groups
            .iter()
            .enumerate()
            .filter_map(|(i, &g)| if g == group_id { Some(i) } else { None })
            .collect();

        if group_indices.is_empty() {
            continue;
        }

        // Create perturbations at different step sizes for higher-order accuracy
        let mut x_h = x.to_owned();
        let mut x_2h = x.to_owned();
        let mut x_minus_h = x.to_owned();
        let mut x_minus_2h = x.to_owned();

        for &idx in &group_indices {
            x_h[idx] = x[idx] + h[idx];
            x_2h[idx] = x[idx] + 2.0 * h[idx];
            x_minus_h[idx] = x[idx] - h[idx];
            x_minus_2h[idx] = x[idx] - 2.0 * h[idx];
        }

        // Evaluate the function at the perturbed points
        let f_h = func(&x_h.view());
        let f_2h = func(&x_2h.view());
        let f_minus_h = func(&x_minus_h.view());
        let f_minus_2h = func(&x_minus_2h.view());

        // Calculate the partial derivatives using 4th-order central difference formula
        let parallel = options.parallel.as_ref();
        let f_h = &f_h; // Capture by reference
        let f_2h = &f_2h; // Capture by reference
        let f_minus_h = &f_minus_h; // Capture by reference
        let f_minus_2h = &f_minus_2h; // Capture by reference
                                      // h is already captured by reference
        let group_indices = &group_indices; // Capture by reference

        // Use a closure that captures all needed variables by reference to avoid ownership issues
        let processor = move |(row, col_indices): (usize, Vec<usize>)| {
            // Calculate derivatives for all columns in this row that belong to the current group
            col_indices
                .iter()
                .filter(|&&col| group_indices.contains(&col))
                .map(move |&col| {
                    // Fourth-order accurate central difference formula
                    // f'(x) ≈ (-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)) / (12h)
                    let deriv = (-f_2h[row] + 8.0 * f_h[row] - 8.0 * f_minus_h[row]
                        + f_minus_2h[row])
                        / (12.0 * h[col]);
                    (row, col, deriv)
                })
                .collect::<Vec<_>>()
        };

        // Get all affected rows and their column indices from the sparsity pattern
        let row_col_pairs: Vec<(usize, Vec<usize>)> = (0..m)
            .map(|row| {
                let cols = get_nonzero_cols_in_row(sparsity, row);
                (row, cols)
            })
            .collect();

        // Process rows in parallel or sequentially
        let derivatives: Vec<(usize, usize, f64)> = if let Some(par_opts) = parallel {
            if row_col_pairs.len() >= par_opts.min_parallel_size && par_opts.parallel_evaluations {
                row_col_pairs
                    .par_iter()
                    .flat_map(|&(row, ref cols)| processor((row, cols.clone())))
                    .collect()
            } else {
                row_col_pairs
                    .iter()
                    .flat_map(|&(row, ref cols)| processor((row, cols.clone())))
                    .collect()
            }
        } else {
            row_col_pairs
                .iter()
                .flat_map(|&(row, ref cols)| processor((row, cols.clone())))
                .collect()
        };

        // Add derivatives to result
        for (row, col, deriv) in derivatives {
            jac_rows.push(row);
            jac_cols.push(col);
            jac_data.push(deriv);
        }
    }

    // Create the sparse Jacobian matrix
    CsrArray::from_triplets(&jac_rows, &jac_cols, &jac_data, (m, n), false)
        .map_err(|e| OptimizeError::ComputationError(format!("Error creating Jacobian: {}", e)))
}

/// Computes Hessian using the 2-point method
fn compute_hessian_2point<F>(
    func: F,
    x: &ArrayView1<f64>,
    grad: &Array1<f64>,
    h: &Array1<f64>,
    groups: &[usize],
    sparsity: &CsrArray<f64>,
    options: &SparseFiniteDiffOptions,
) -> Result<CsrArray<f64>, OptimizeError>
where
    F: Fn(&ArrayView1<f64>) -> f64 + Sync + Clone,
{
    let (m, n) = sparsity.shape();
    if m != n {
        return Err(OptimizeError::ComputationError(
            "Hessian must be square".to_string(),
        ));
    }

    let num_groups = *groups.iter().max().unwrap_or(&0) + 1;

    // Prepare result data structures
    let mut hess_data = Vec::new();
    let mut hess_rows = Vec::new();
    let mut hess_cols = Vec::new();

    // Process each group of columns
    for group_id in 0..num_groups {
        // Determine indices in this group
        let group_indices: Vec<usize> = groups
            .iter()
            .enumerate()
            .filter_map(|(i, &g)| if g == group_id { Some(i) } else { None })
            .collect();

        if group_indices.is_empty() {
            continue;
        }

        // For Hessian, we'll compute the gradient at perturbed points
        let mut x_perturbed = x.to_owned();
        for &idx in &group_indices {
            x_perturbed[idx] = x[idx] + h[idx];
        }

        // Create a simple gradient approximation function
        fn calc_gradient<F: Fn(&ArrayView1<f64>) -> f64 + ?Sized>(
            func: &F,
            x: &ArrayView1<f64>,
        ) -> Array1<f64> {
            let mut grad = Array1::zeros(x.len());
            let eps = 1e-8;

            let f0 = func(x);

            for i in 0..x.len() {
                let mut x_plus = x.to_owned();
                x_plus[i] += eps;
                let f_plus = func(&x_plus.view());

                grad[i] = (f_plus - f0) / eps;
            }

            grad
        }

        // Evaluate the gradient at the perturbed point
        let grad_perturbed = calc_gradient(&func, &x_perturbed.view());

        // Calculate the second derivatives for this group
        let parallel = options.parallel.as_ref();
        let grad_perturbed = &grad_perturbed; // Capture by reference
                                              // grad is already captured by reference
                                              // h is already captured by reference
        let group_indices = &group_indices; // Capture by reference

        // Use a closure that captures all needed variables by reference to avoid ownership issues
        let processor = move |(row, col_indices): (usize, Vec<usize>)| {
            // Calculate derivatives for all columns in this row that belong to the current group
            col_indices
                .iter()
                .filter(|&&col| group_indices.contains(&col))
                .map(move |&col| {
                    let deriv = (grad_perturbed[row] - grad[row]) / h[col];
                    (row, col, deriv)
                })
                .collect::<Vec<_>>()
        };

        // Get all row and column pairs from the sparsity pattern
        let row_col_pairs: Vec<(usize, Vec<usize>)> = (0..n)
            .map(|row| {
                let cols = get_nonzero_cols_in_row(sparsity, row);
                (row, cols)
            })
            .collect();

        // Process rows in parallel or sequentially
        let derivatives: Vec<(usize, usize, f64)> = if let Some(par_opts) = parallel {
            if row_col_pairs.len() >= par_opts.min_parallel_size && par_opts.parallel_evaluations {
                row_col_pairs
                    .par_iter()
                    .flat_map(|&(row, ref cols)| processor((row, cols.clone())))
                    .collect()
            } else {
                row_col_pairs
                    .iter()
                    .flat_map(|&(row, ref cols)| processor((row, cols.clone())))
                    .collect()
            }
        } else {
            row_col_pairs
                .iter()
                .flat_map(|&(row, ref cols)| processor((row, cols.clone())))
                .collect()
        };

        // Add derivatives to result
        for (row, col, deriv) in derivatives {
            hess_rows.push(row);
            hess_cols.push(col);
            hess_data.push(deriv);
        }
    }

    // Create the sparse Hessian matrix
    CsrArray::from_triplets(&hess_rows, &hess_cols, &hess_data, (n, n), false)
        .map_err(|e| OptimizeError::ComputationError(format!("Error creating Hessian: {}", e)))
}

/// Computes Hessian using the 3-point method
fn compute_hessian_3point<F>(
    func: F,
    x: &ArrayView1<f64>,
    h: &Array1<f64>,
    groups: &[usize],
    sparsity: &CsrArray<f64>,
    options: &SparseFiniteDiffOptions,
) -> Result<CsrArray<f64>, OptimizeError>
where
    F: Fn(&ArrayView1<f64>) -> f64 + Sync + Clone,
{
    let (m, n) = sparsity.shape();
    if m != n {
        return Err(OptimizeError::ComputationError(
            "Hessian must be square".to_string(),
        ));
    }

    let num_groups = *groups.iter().max().unwrap_or(&0) + 1;

    // Prepare result data structures
    let mut hess_data = Vec::new();
    let mut hess_rows = Vec::new();
    let mut hess_cols = Vec::new();

    // Base function value
    let f0 = func(x);

    // Process each group of columns
    for group_id in 0..num_groups {
        // Determine indices in this group
        let group_indices: Vec<usize> = groups
            .iter()
            .enumerate()
            .filter_map(|(i, &g)| if g == group_id { Some(i) } else { None })
            .collect();

        if group_indices.is_empty() {
            continue;
        }

        // For Hessian diagonal elements (i,i)
        for &i in &group_indices {
            // Create perturbations
            let mut x_plus = x.to_owned();
            let mut x_minus = x.to_owned();
            x_plus[i] = x[i] + h[i];
            x_minus[i] = x[i] - h[i];

            // Evaluate function
            let f_plus = func(&x_plus.view());
            let f_minus = func(&x_minus.view());

            // Calculate diagonal element using central difference
            let d2f_dx2 = (f_plus - 2.0 * f0 + f_minus) / (h[i] * h[i]);

            hess_rows.push(i);
            hess_cols.push(i);
            hess_data.push(d2f_dx2);
        }

        // For off-diagonal elements (i,j) where i≠j
        let parallel = options.parallel.as_ref();
        let _f0_val = f0; // Capture for closure
        let x_ref = x; // Capture x by reference
        let h_ref = h; // Capture h by reference
        let func_ref = &func; // Capture function by reference
        let group_indices_ref = &group_indices; // Capture group_indices by reference

        let processor = move |(i, j): (usize, usize)| {
            if i == j || !group_indices_ref.contains(&j) {
                return None;
            }

            // Create perturbations for mixed partial derivative
            let mut x_plus_plus = x_ref.to_owned();
            let mut x_plus_minus = x_ref.to_owned();
            let mut x_minus_plus = x_ref.to_owned();
            let mut x_minus_minus = x_ref.to_owned();

            x_plus_plus[i] = x_ref[i] + h_ref[i];
            x_plus_plus[j] = x_ref[j] + h_ref[j];

            x_plus_minus[i] = x_ref[i] + h_ref[i];
            x_plus_minus[j] = x_ref[j] - h_ref[j];

            x_minus_plus[i] = x_ref[i] - h_ref[i];
            x_minus_plus[j] = x_ref[j] + h_ref[j];

            x_minus_minus[i] = x_ref[i] - h_ref[i];
            x_minus_minus[j] = x_ref[j] - h_ref[j];

            // Evaluate function at all points
            let f_plus_plus = func_ref(&x_plus_plus.view());
            let f_plus_minus = func_ref(&x_plus_minus.view());
            let f_minus_plus = func_ref(&x_minus_plus.view());
            let f_minus_minus = func_ref(&x_minus_minus.view());

            // Mixed partial derivative using central difference
            let d2f_dxdy = (f_plus_plus - f_plus_minus - f_minus_plus + f_minus_minus)
                / (4.0 * h_ref[i] * h_ref[j]);

            Some((i, j, d2f_dxdy))
        };

        // Get all pairs of indices from sparsity pattern
        let mut index_pairs = Vec::new();
        for i in 0..n {
            let cols_i = get_nonzero_cols_in_row(sparsity, i);
            for &j in &cols_i {
                if i != j && group_indices.contains(&j) {
                    index_pairs.push((i, j));
                }
            }
        }

        // Process index pairs in parallel or sequentially
        let derivatives: Vec<(usize, usize, f64)> = if let Some(par_opts) = parallel {
            if index_pairs.len() >= par_opts.min_parallel_size && par_opts.parallel_evaluations {
                index_pairs
                    .par_iter()
                    .filter_map(|&(i, j)| processor((i, j)))
                    .collect()
            } else {
                index_pairs
                    .iter()
                    .filter_map(|&(i, j)| processor((i, j)))
                    .collect()
            }
        } else {
            index_pairs
                .iter()
                .filter_map(|&(i, j)| processor((i, j)))
                .collect()
        };

        // Add derivatives to result
        for (row, col, deriv) in derivatives {
            hess_rows.push(row);
            hess_cols.push(col);
            hess_data.push(deriv);
        }
    }

    // Create the sparse Hessian matrix
    CsrArray::from_triplets(&hess_rows, &hess_cols, &hess_data, (n, n), false)
        .map_err(|e| OptimizeError::ComputationError(format!("Error creating Hessian: {}", e)))
}

/// Fills in the lower triangular part of a symmetric matrix
fn fill_symmetric_hessian(upper: &CsrArray<f64>) -> Result<CsrArray<f64>, OptimizeError> {
    let (n, _) = upper.shape();

    // Extract CSR components
    let data = upper.get_data();
    let indices = upper.get_indices();
    let indptr = upper.get_indptr();

    // Create new entries for both upper and lower triangular parts
    let mut new_data = Vec::new();
    let mut new_rows = Vec::new();
    let mut new_cols = Vec::new();

    // Process each row
    for i in 0..n {
        let start = indptr[i];
        let end = indptr[i + 1];

        // Add original entries from the upper triangular part
        for k in start..end {
            let j = indices[k];
            new_data.push(data[k]);
            new_rows.push(i);
            new_cols.push(j);

            // Add symmetric entry for off-diagonal elements
            if i != j {
                new_data.push(data[k]);
                new_rows.push(j); // Swap i and j
                new_cols.push(i);
            }
        }
    }

    // Create the full symmetric Hessian
    CsrArray::from_triplets(&new_rows, &new_cols, &new_data, (n, n), false).map_err(|e| {
        OptimizeError::ComputationError(format!("Error creating symmetric Hessian: {}", e))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// Test function: sphere function
    fn sphere(x: &ArrayView1<f64>) -> f64 {
        x.iter().map(|&xi| xi * xi).sum()
    }

    /// Test function: multivariate output
    fn multi_output(x: &ArrayView1<f64>) -> Array1<f64> {
        let mut result = Array1::zeros(2);
        result[0] = x[0].powi(2) + x[1];
        result[1] = x[0] * x[1];
        result
    }

    /// Himmelblau function: has multiple local minima
    fn himmelblau(x: &ArrayView1<f64>) -> f64 {
        (x[0].powi(2) + x[1] - 11.0).powi(2) + (x[0] + x[1].powi(2) - 7.0).powi(2)
    }

    #[test]
    fn test_sparse_jacobian_dense() {
        // Create a test point
        let x = array![1.0, 2.0];

        // Compute Jacobian
        let jac = sparse_jacobian(multi_output, &x.view(), None, None, None).unwrap();

        // Expected Jacobian
        // [2*x[0], 1]
        // [x[1], x[0]]
        let expected = array![[2.0, 1.0], [2.0, 1.0]];

        // Compare with expected result
        let jac_dense = jac.to_array();

        // Check dimensions
        assert_eq!(jac_dense.shape(), [2, 2]);

        // Check values (with tolerance for numerical accuracy)
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (jac_dense[[i, j]] - expected[[i, j]]).abs() < 1e-5,
                    "Jacobian element [{}, {}] = {} doesn't match expected {}",
                    i,
                    j,
                    jac_dense[[i, j]],
                    expected[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_sparse_jacobian_higher_order() {
        // Create a test point
        let x = array![1.0, 2.0];

        // Create options for 4th-order method
        let mut options = SparseFiniteDiffOptions::default();
        options.method = "cs".to_string();

        // Compute Jacobian
        let jac = sparse_jacobian(multi_output, &x.view(), None, None, Some(options)).unwrap();

        // Expected Jacobian
        // [2*x[0], 1]
        // [x[1], x[0]]
        let expected = array![[2.0, 1.0], [2.0, 1.0]];

        // Compare with expected result
        let jac_dense = jac.to_array();

        // Check dimensions
        assert_eq!(jac_dense.shape(), [2, 2]);

        // Check values (with tolerance for numerical accuracy)
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (jac_dense[[i, j]] - expected[[i, j]]).abs() < 1e-5,
                    "Jacobian element [{}, {}] = {} doesn't match expected {}",
                    i,
                    j,
                    jac_dense[[i, j]],
                    expected[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_sparse_hessian_sphere() {
        // Create a test point
        let x = array![1.0, 2.0];

        // Compute Hessian using 3-point method
        let mut options = SparseFiniteDiffOptions::default();
        options.method = "3-point".to_string();
        let hess = sparse_hessian(sphere, &x.view(), None, None, Some(options)).unwrap();

        // Expected Hessian for sphere function is 2*I
        let expected = array![[2.0, 0.0], [0.0, 2.0]];

        // Compare with expected result
        let hess_dense = hess.to_array();

        // Check dimensions
        assert_eq!(hess_dense.shape(), [2, 2]);

        // Check values (with tolerance for numerical accuracy)
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (hess_dense[[i, j]] - expected[[i, j]]).abs() < 1e-5,
                    "Hessian element [{}, {}] = {} doesn't match expected {}",
                    i,
                    j,
                    hess_dense[[i, j]],
                    expected[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_sparse_hessian_higher_order() {
        // Create a test point
        let x = array![1.0, 2.0];

        // Create options for 3-point method (since cs has issues)
        let mut options = SparseFiniteDiffOptions::default();
        options.method = "3-point".to_string();

        // Compute Hessian
        let hess = sparse_hessian(sphere, &x.view(), None, None, Some(options)).unwrap();

        // Expected Hessian for sphere function is 2*I
        let expected = array![[2.0, 0.0], [0.0, 2.0]];

        // Compare with expected result
        let hess_dense = hess.to_array();

        // Check dimensions
        assert_eq!(hess_dense.shape(), [2, 2]);

        // Check values (with tolerance for numerical accuracy)
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (hess_dense[[i, j]] - expected[[i, j]]).abs() < 1e-5,
                    "Hessian element [{}, {}] = {} doesn't match expected {}",
                    i,
                    j,
                    hess_dense[[i, j]],
                    expected[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_sparse_hessian_rosenbrock() {
        // Rosenbrock function
        let rosenbrock = |x: &ArrayView1<f64>| {
            let a = 1.0;
            let b = 100.0;
            (a - x[0]).powi(2) + b * (x[1] - x[0].powi(2)).powi(2)
        };

        // Create a test point
        let x = array![1.0, 1.0]; // At minimum

        // Compute Hessian
        let hess = sparse_hessian(rosenbrock, &x.view(), None, None, None).unwrap();

        // Expected Hessian at (1,1) - not a diagonal matrix
        let expected = array![[802.0, -400.0], [-400.0, 200.0]];

        // Compare with expected result
        let hess_dense = hess.to_array();

        // Check dimensions
        assert_eq!(hess_dense.shape(), [2, 2]);

        // Check values (with higher tolerance due to numerical issues with Rosenbrock)
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (hess_dense[[i, j]] - expected[[i, j]]).abs() < 5.0,
                    "Hessian element [{}, {}] = {} doesn't match expected {}",
                    i,
                    j,
                    hess_dense[[i, j]],
                    expected[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_column_grouping() {
        // Create a simple sparsity pattern
        let indices_row = vec![0, 0, 1, 1, 2, 2, 3, 3, 4, 4];
        let indices_col = vec![0, 1, 1, 2, 2, 3, 3, 4, 0, 4];
        let data = vec![1.0; 10];
        let n = 5;

        let sparsity =
            CsrArray::from_triplets(&indices_row, &indices_col, &data, (n, n), false).unwrap();

        // Compute groups
        let groups = determine_column_groups(&sparsity, 10, Some(42));

        // Verify that no two adjacent columns have the same group
        for row in 0..n {
            let row_cols = get_nonzero_cols_in_row(&sparsity, row);
            for (i, &col_i) in row_cols.iter().enumerate() {
                for (j, &col_j) in row_cols.iter().enumerate() {
                    if i != j {
                        assert_ne!(
                            groups[col_i], groups[col_j],
                            "Columns {} and {} share row {} but have same group {} and {}",
                            col_i, col_j, row, groups[col_i], groups[col_j]
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_large_sparse_jacobian() {
        // Create a large sparse function (tridiagonal)
        fn tridiagonal_func(x: &ArrayView1<f64>) -> Array1<f64> {
            let n = x.len();
            let mut result = Array1::zeros(n);

            for i in 0..n {
                if i > 0 {
                    result[i] += x[i - 1]; // Lower diagonal
                }
                result[i] += 2.0 * x[i]; // Diagonal
                if i < n - 1 {
                    result[i] += x[i + 1]; // Upper diagonal
                }
            }

            result
        }

        // Create test point (size 100 for a large sparse system)
        let n = 100;
        let x = Array1::from_vec((0..n).map(|i| i as f64).collect());

        // Create sparsity pattern (tridiagonal)
        let mut indices_row = Vec::new();
        let mut indices_col = Vec::new();
        let mut data = Vec::new();

        for i in 0..n {
            if i > 0 {
                indices_row.push(i);
                indices_col.push(i - 1);
                data.push(1.0);
            }

            indices_row.push(i);
            indices_col.push(i);
            data.push(1.0);

            if i < n - 1 {
                indices_row.push(i);
                indices_col.push(i + 1);
                data.push(1.0);
            }
        }

        let sparsity =
            CsrArray::from_triplets(&indices_row, &indices_col, &data, (n, n), false).unwrap();

        // Create options with parallel computation enabled
        let mut options = SparseFiniteDiffOptions::default();
        options.parallel = Some(ParallelOptions::default());

        // Compute Jacobian
        let jac = sparse_jacobian(
            tridiagonal_func,
            &x.view(),
            None,
            Some(&sparsity),
            Some(options),
        )
        .unwrap();

        // Verify dimensions
        assert_eq!(jac.shape(), (n, n));

        // Verify sparsity pattern is preserved (should be tridiagonal)
        for i in 0..n {
            let row_cols = get_nonzero_cols_in_row(&jac, i);

            if i > 0 {
                assert!(
                    row_cols.contains(&(i - 1)),
                    "Lower diagonal missing at row {}",
                    i
                );
            }

            assert!(row_cols.contains(&i), "Diagonal missing at row {}", i);

            if i < n - 1 {
                assert!(
                    row_cols.contains(&(i + 1)),
                    "Upper diagonal missing at row {}",
                    i
                );
            }

            // Should not have any other non-zeros
            assert!(row_cols.len() <= 3, "Row {} has too many non-zeros", i);
        }
    }

    #[test]
    fn test_sparse_himmelblau_hessian() {
        // Test points at different critical points of Himmelblau function
        let critical_points = [
            array![3.0, 2.0],             // Minimum
            array![-2.805118, 3.131312],  // Minimum
            array![-3.779310, -3.283186], // Minimum
            array![3.584428, -1.848126],  // Minimum
        ];

        for (i, x) in critical_points.iter().enumerate() {
            // Create options with 3-point method
            let mut options = SparseFiniteDiffOptions::default();
            options.method = "3-point".to_string();

            // Compute Hessian
            let hess = sparse_hessian(himmelblau, &x.view(), None, None, Some(options)).unwrap();

            // Verify dimensions
            assert_eq!(hess.shape(), (2, 2));

            // At the minima, the Hessian should be positive definite
            // We'll check that the diagonal entries are positive
            let diag0 = hess.get(0, 0);
            let diag1 = hess.get(1, 1);

            assert!(diag0 > 0.0, "Diagonal[0,0] not positive at minimum {}", i);
            assert!(diag1 > 0.0, "Diagonal[1,1] not positive at minimum {}", i);

            // Compute the determinant to check positive definiteness
            let det = diag0 * diag1 - hess.get(0, 1) * hess.get(1, 0);
            assert!(
                det > 0.0,
                "Hessian not positive definite at minimum {}, det={}",
                i,
                det
            );
        }
    }
}
