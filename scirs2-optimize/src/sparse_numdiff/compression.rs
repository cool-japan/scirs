//! Compression techniques for sparse matrices in numerical differentiation
//!
//! This module provides implementations of compression algorithms for
//! sparse matrices to reduce the number of function evaluations required
//! for finite differences.

use crate::error::OptimizeError;
use scirs2_core::ndarray::{Array2, ArrayView2};
// Unused import: Array1
use scirs2_sparse::csr_array::CsrArray;
use scirs2_sparse::SparseArray;
use std::collections::HashSet;
// Unused import: HashMap

/// Type alias for the return type of compress_jacobian_pattern
pub type CompressedJacobianPattern = (CsrArray<f64>, Array2<f64>, Array2<f64>);

/// Compresses a sparse Jacobian pattern for more efficient finite differencing
///
/// Uses a simplified Curtis-Powell-Reid algorithm with graph coloring to group
/// columns that can be computed simultaneously without interference.
///
/// # Arguments
///
/// * `sparsity` - Sparse matrix representing the Jacobian sparsity pattern
///
/// # Returns
///
/// * Compressed sparsity pattern and compression matrices (original, B, C)
///   where B is the column compression matrix and C is the row compression matrix
#[allow(dead_code)]
pub fn compress_jacobian_pattern(
    sparsity: &CsrArray<f64>,
) -> Result<CompressedJacobianPattern, OptimizeError> {
    let (m, n) = sparsity.shape();

    // Perform column coloring: group columns that don't interfere
    let coloring = color_jacobian_columns(sparsity)?;
    let num_colors = coloring.iter().max().unwrap_or(&0) + 1;

    // Build column compression matrix B (n x num_colors)
    let mut b = Array2::zeros((n, num_colors));
    for (col, &color) in coloring.iter().enumerate() {
        b[[col, color]] = 1.0;
    }

    // Row compression matrix C is identity for Jacobian (m x m)
    let c = Array2::eye(m);

    Ok((sparsity.clone(), b, c))
}

/// Colors the columns of a Jacobian sparsity pattern using a greedy algorithm
///
/// Two columns can have the same color if they don't both have nonzeros in the same row
#[allow(dead_code)]
fn color_jacobian_columns(sparsity: &CsrArray<f64>) -> Result<Vec<usize>, OptimizeError> {
    let (_m, n) = sparsity.shape();
    let mut coloring = vec![0; n];

    // For each column, find the lowest color that doesn't conflict
    for col in 0..n {
        let mut used_colors = HashSet::new();

        // Find all rows where this column has a nonzero
        let col_rows = get_column_nonzero_rows(sparsity, col);

        // Check previously colored columns that share rows with this column
        for prev_col in 0..col {
            let prev_col_rows = get_column_nonzero_rows(sparsity, prev_col);

            // If columns share any row, they can't have the same color
            if col_rows.iter().any(|&row| prev_col_rows.contains(&row)) {
                used_colors.insert(coloring[prev_col]);
            }
        }

        // Assign the lowest available color
        let mut color = 0;
        while used_colors.contains(&color) {
            color += 1;
        }
        coloring[col] = color;
    }

    Ok(coloring)
}

/// Helper function to get rows where a column has nonzero entries
#[allow(dead_code)]
fn get_column_nonzero_rows(sparsity: &CsrArray<f64>, col: usize) -> HashSet<usize> {
    let mut rows = HashSet::new();
    let (m, _) = sparsity.shape();

    for row in 0..m {
        let val = sparsity.get(row, col);
        if val.abs() > 1e-15 {
            rows.insert(row);
        }
    }

    rows
}

/// Compresses a sparse Hessian pattern for more efficient finite differencing
///
/// Uses a distance-2 coloring algorithm suitable for symmetric Hessian matrices.
/// Two columns can have the same color if they are at distance > 2 in the sparsity graph.
///
/// # Arguments
///
/// * `sparsity` - Sparse matrix representing the Hessian sparsity pattern
///
/// # Returns
///
/// * Compressed sparsity pattern and compression matrix P (n x num_colors)
#[allow(dead_code)]
pub fn compress_hessian_pattern(
    sparsity: &CsrArray<f64>,
) -> Result<(CsrArray<f64>, Array2<f64>), OptimizeError> {
    let (n, _) = sparsity.shape();

    // Perform distance-2 coloring for Hessian
    let coloring = color_hessian_columns(sparsity)?;
    let num_colors = coloring.iter().max().unwrap_or(&0) + 1;

    // Build compression matrix P (n x num_colors)
    let mut p = Array2::zeros((n, num_colors));
    for (col, &color) in coloring.iter().enumerate() {
        p[[col, color]] = 1.0;
    }

    Ok((sparsity.clone(), p))
}

/// Colors the columns of a Hessian sparsity pattern using distance-2 coloring
///
/// For symmetric matrices, two columns i and j can have the same color if:
/// 1. They are not adjacent (H[i,j] = 0)
/// 2. They don't share any common neighbors
#[allow(dead_code)]
fn color_hessian_columns(sparsity: &CsrArray<f64>) -> Result<Vec<usize>, OptimizeError> {
    let (n, _) = sparsity.shape();
    let mut coloring = vec![0; n];

    // Build adjacency information for efficient neighbor lookup
    let adjacency = build_adjacency_list(sparsity);

    // For each column, find the lowest color that doesn't conflict
    for col in 0..n {
        let mut used_colors = HashSet::new();

        // Get neighbors of current column
        let neighbors = &adjacency[col];

        // Check all previously colored columns
        for prev_col in 0..col {
            let prev_neighbors = &adjacency[prev_col];

            // Check if columns are distance <= 2 apart
            let distance_conflict =
                // Distance 1: directly connected
                neighbors.contains(&prev_col) ||
                // Distance 2: share a common neighbor
                neighbors.iter().any(|&n| prev_neighbors.contains(&n));

            if distance_conflict {
                used_colors.insert(coloring[prev_col]);
            }
        }

        // Assign the lowest available color
        let mut color = 0;
        while used_colors.contains(&color) {
            color += 1;
        }
        coloring[col] = color;
    }

    Ok(coloring)
}

/// Build adjacency list representation of the sparsity pattern
#[allow(dead_code)]
fn build_adjacency_list(sparsity: &CsrArray<f64>) -> Vec<HashSet<usize>> {
    let (n, _) = sparsity.shape();
    let mut adjacency = vec![HashSet::new(); n];

    for i in 0..n {
        for j in 0..n {
            if i != j {
                let val = sparsity.get(i, j);
                if val.abs() > 1e-15 {
                    adjacency[i].insert(j);
                    adjacency[j].insert(i); // Symmetric
                }
            }
        }
    }

    adjacency
}

/// Reconstructs a sparse Jacobian from compressed gradient evaluations
///
/// Takes the compressed gradient evaluations and reconstructs the full sparse Jacobian
/// using the compression matrices. Each compressed gradient contains information about
/// multiple columns that were grouped together during compression.
///
/// # Arguments
///
/// * `gradients` - Matrix of compressed gradient evaluations (m x num_colors)
/// * `b` - Column compression matrix (n x num_colors)
/// * `c` - Row compression matrix (m x m), typically identity for Jacobian
/// * `sparsity` - Original sparsity pattern for reconstruction
///
/// # Returns
///
/// * Reconstructed sparse Jacobian
#[allow(dead_code)]
pub fn reconstruct_jacobian(
    gradients: &ArrayView2<f64>,
    b: &ArrayView2<f64>,
    _c: &ArrayView2<f64>,
    sparsity: &CsrArray<f64>,
) -> Result<CsrArray<f64>, OptimizeError> {
    let (m, n) = sparsity.shape();
    let (grad_m, num_colors) = gradients.dim();
    let (b_n, b_colors) = b.dim();

    // Validate dimensions
    if grad_m != m || b_n != n || b_colors != num_colors {
        return Err(OptimizeError::InvalidInput(
            "Dimension mismatch in reconstruction matrices".to_string(),
        ));
    }

    // Create dense matrix for reconstruction, then convert to sparse
    let mut jacobian_dense = Array2::zeros((m, n));

    // For each color group, extract the columns
    for color in 0..num_colors {
        // Find all columns that belong to this color
        let mut columns_in_color = Vec::new();
        for col in 0..n {
            if b[[col, color]].abs() > 1e-15 {
                columns_in_color.push(col);
            }
        }

        // The gradient for this color contains the sum of derivatives for all columns in the group
        // Since we used proper coloring, we can extract individual columns
        for &col in &columns_in_color {
            for row in 0..m {
                // Check if this position should be nonzero according to sparsity pattern
                let val = sparsity.get(row, col);
                if val.abs() > 1e-15 {
                    // The gradient[row, color] contains the derivative ∂f_row/∂x_col
                    jacobian_dense[[row, col]] = gradients[[row, color]];
                }
            }
        }
    }

    // Convert dense matrix to sparse, preserving only the sparsity pattern
    let mut row_indices = Vec::new();
    let mut col_indices = Vec::new();
    let mut values = Vec::new();

    for row in 0..m {
        for col in 0..n {
            let sparsity_val = sparsity.get(row, col);
            if sparsity_val.abs() > 1e-15 && jacobian_dense[[row, col]].abs() > 1e-15 {
                row_indices.push(row);
                col_indices.push(col);
                values.push(jacobian_dense[[row, col]]);
            }
        }
    }

    // Create sparse matrix from the reconstructed values
    CsrArray::from_triplets(&row_indices, &col_indices, &values, (m, n), false)
        .map_err(|_| OptimizeError::InvalidInput("Failed to create sparse matrix".to_string()))
}

/// Reconstructs a sparse Hessian from compressed gradient evaluations using central differences
///
/// Takes compressed gradient evaluations and reconstructs the symmetric sparse Hessian.
/// The reconstruction assumes that gradients were computed using central differences
/// with the compressed perturbation vectors.
///
/// # Arguments
///
/// * `gradients_forward` - Forward difference gradients (n x num_colors)
/// * `gradients_backward` - Backward difference gradients (n x num_colors)  
/// * `p` - Compression matrix (n x num_colors)
/// * `sparsity` - Original Hessian sparsity pattern
/// * `h` - Step size used in finite differences
///
/// # Returns
///
/// * Reconstructed sparse Hessian
#[allow(dead_code)]
pub fn reconstruct_hessian_central_diff(
    gradients_forward: &ArrayView2<f64>,
    gradients_backward: &ArrayView2<f64>,
    p: &ArrayView2<f64>,
    sparsity: &CsrArray<f64>,
    h: f64,
) -> Result<CsrArray<f64>, OptimizeError> {
    let (n, _) = sparsity.shape();
    let (grad_n, num_colors) = gradients_forward.dim();
    let (p_n, p_colors) = p.dim();

    // Validate dimensions
    if grad_n != n || p_n != n || p_colors != num_colors {
        return Err(OptimizeError::InvalidInput(
            "Dimension mismatch in Hessian reconstruction matrices".to_string(),
        ));
    }

    if gradients_backward.dim() != (n, num_colors) {
        return Err(OptimizeError::InvalidInput(
            "Gradient matrices must have the same dimensions".to_string(),
        ));
    }

    // Create dense matrix for reconstruction
    let mut hessian_dense = Array2::zeros((n, n));

    // Reconstruct using central differences: H = (grad_forward - grad_backward) / (2*h)
    for color in 0..num_colors {
        // Find columns that belong to this color
        let mut columns_in_color = Vec::new();
        for col in 0..n {
            if p[[col, color]].abs() > 1e-15 {
                columns_in_color.push(col);
            }
        }

        // For each pair of variables in the same color group
        for &col_i in &columns_in_color {
            for row in 0..n {
                // Check if (row, col_i) should be nonzero according to sparsity
                if sparsity.get(row, col_i).abs() > 1e-15 {
                    // Central difference approximation
                    let h_val = (gradients_forward[[row, color]]
                        - gradients_backward[[row, color]])
                        / (2.0 * h);
                    hessian_dense[[row, col_i]] = h_val;

                    // Ensure symmetry for Hessian
                    if row != col_i && sparsity.get(col_i, row).abs() > 1e-15 {
                        hessian_dense[[col_i, row]] = h_val;
                    }
                }
            }
        }
    }

    // Convert to sparse matrix, preserving sparsity pattern
    let mut row_indices = Vec::new();
    let mut col_indices = Vec::new();
    let mut values = Vec::new();

    for row in 0..n {
        for col in 0..n {
            if sparsity.get(row, col).abs() > 1e-15 && hessian_dense[[row, col]].abs() > 1e-15 {
                row_indices.push(row);
                col_indices.push(col);
                values.push(hessian_dense[[row, col]]);
            }
        }
    }

    // Create sparse Hessian
    CsrArray::from_triplets(&row_indices, &col_indices, &values, (n, n), false)
        .map_err(|_| OptimizeError::ValueError("Failed to create sparse Hessian".to_string()))
}

/// Simplified version of Hessian reconstruction for single gradient input
#[allow(dead_code)]
pub fn reconstruct_hessian(
    gradients: &ArrayView2<f64>,
    p: &ArrayView2<f64>,
    sparsity: &CsrArray<f64>,
) -> Result<CsrArray<f64>, OptimizeError> {
    // For backward compatibility, assume we have only forward differences
    // In practice, this would be less accurate than using central differences
    let h = 1e-8; // Default step size
    let gradients_backward = Array2::zeros(gradients.dim());
    let gradients_backward_view = gradients_backward.view();

    reconstruct_hessian_central_diff(gradients, &gradients_backward_view, p, sparsity, h)
}
