//! Graph coloring algorithms for efficient sparse differentiation
//!
//! This module provides implementations of graph coloring algorithms
//! used for grouping columns in sparse finite difference calculations.

use crate::error::OptimizeError;
use scirs2_core::random::{rngs::StdRng, seq::SliceRandom, SeedableRng};
use scirs2_sparse::csr_array::CsrArray;
use std::collections::{HashMap, HashSet};

/// Get the list of non-zero columns in a specific row of a sparse matrix
#[allow(dead_code)]
fn get_nonzero_cols_in_row(matrix: &CsrArray<f64>, row: usize) -> Vec<usize> {
    // Get row indices
    let row_start = matrix.get_indptr()[row];
    let row_end = matrix.get_indptr()[row + 1];

    // Collect all column indices in this row
    let indices = matrix.get_indices();
    let mut cols = Vec::new();
    for i in row_start..row_end {
        cols.push(indices[i]);
    }
    cols
}

/// Implements a greedy graph coloring algorithm to group columns that can be
/// perturbed simultaneously during sparse finite differences
///
/// # Arguments
///
/// * `sparsity` - Sparse matrix representing the sparsity pattern
/// * `seed` - Optional random seed for reproducibility
/// * `max_group_size` - Maximum number of columns per group
///
/// # Returns
///
/// * Vector of column groups, where each group contains columns that don't conflict
#[allow(dead_code)]
pub fn determine_column_groups(
    sparsity: &CsrArray<f64>,
    seed: Option<u64>,
    max_group_size: Option<usize>,
) -> Result<Vec<Vec<usize>>, OptimizeError> {
    let (m, n) = sparsity.shape();

    // Create a conflict graph represented as an adjacency list
    // Two columns conflict if they have nonzeros in the same row
    let mut conflicts: Vec<HashSet<usize>> = vec![HashSet::new(); n];

    // Build the conflict graph
    for row in 0..m {
        let cols = get_nonzero_cols_in_row(sparsity, row);

        // All columns with nonzeros in this row conflict with each other
        for &col1 in &cols {
            for &col2 in &cols {
                if col1 != col2 {
                    conflicts[col1].insert(col2);
                    conflicts[col2].insert(col1);
                }
            }
        }
    }

    // Order vertices for coloring (by degree, randomized)
    let mut order: Vec<usize> = (0..n).collect();

    // Sort vertices by degree (number of conflicts) for better coloring
    order.sort_by_key(|&v| conflicts[v].len());

    // Randomize the order of vertices with the same degree
    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => {
            // Use a constant seed for reproducibility in case rng fails
            // This is a fallback case, so using a fixed seed is acceptable
            StdRng::seed_from_u64(0)
        }
    };

    // Group vertices with the same degree and shuffle each group
    let mut i = 0;
    while i < order.len() {
        let degree = conflicts[order[i]].len();
        let mut j = i + 1;
        while j < order.len() && conflicts[order[j]].len() == degree {
            j += 1;
        }

        // Shuffle this group
        order[i..j].shuffle(&mut rng);

        i = j;
    }

    // Apply greedy coloring
    let mut vertex_colors: HashMap<usize, usize> = HashMap::new();

    for &v in &order {
        // Find the lowest color not used by neighbors
        let mut neighbor_colors: HashSet<usize> = HashSet::new();
        for &neighbor in &conflicts[v] {
            if let Some(&color) = vertex_colors.get(&neighbor) {
                neighbor_colors.insert(color);
            }
        }

        // Find smallest available color
        let mut color = 0;
        while neighbor_colors.contains(&color) {
            color += 1;
        }

        vertex_colors.insert(v, color);
    }

    // Group vertices by color
    let max_color = vertex_colors.values().max().cloned().unwrap_or(0);
    let mut color_groups: Vec<Vec<usize>> = vec![Vec::new(); max_color + 1];

    for (vertex, &color) in &vertex_colors {
        color_groups[color].push(*vertex);
    }

    // Apply max group _size constraint if specified
    let max_size = max_group_size.unwrap_or(usize::MAX);
    if max_size < n {
        let mut final_groups = Vec::new();

        for group in color_groups {
            if group.len() <= max_size {
                final_groups.push(group);
            } else {
                // Split into smaller groups
                for chunk in group.chunks(max_size) {
                    final_groups.push(chunk.to_vec());
                }
            }
        }

        Ok(final_groups)
    } else {
        // Filter out empty groups
        Ok(color_groups.into_iter().filter(|g| !g.is_empty()).collect())
    }
}
