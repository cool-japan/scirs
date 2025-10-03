//! Compressed sparse graph algorithms module
//!
//! This module provides graph algorithms optimized for sparse matrices,
//! similar to SciPy's `sparse.csgraph` module.
//!
//! ## Features
//!
//! * Shortest path algorithms (Dijkstra, Bellman-Ford)
//! * Connected components analysis  
//! * Graph traversal utilities (BFS, DFS)
//! * Laplacian matrix computation
//! * Minimum spanning tree algorithms
//! * Graph connectivity testing
//!
//! ## Examples
//!
//! ### Shortest Path
//!
//! ```
//! use scirs2_sparse::csgraph::shortest_path;
//! use scirs2_sparse::csr_array::CsrArray;
//!
//! // Create a graph as a sparse matrix
//! let rows = vec![0, 0, 1, 2];
//! let cols = vec![1, 2, 2, 0];
//! let data = vec![1.0, 4.0, 2.0, 3.0];
//! let graph = CsrArray::from_triplets(&rows, &cols, &data, (3, 3), false).unwrap();
//!
//! // Find shortest paths from vertex 0
//! let distances = shortest_path(&graph, Some(0), None, "dijkstra", true, false).unwrap();
//! ```
//!
//! ### Connected Components
//!
//! ```
//! use scirs2_sparse::csgraph::connected_components;
//! use scirs2_sparse::csr_array::CsrArray;
//!
//! // Create a graph
//! let rows = vec![0, 1, 2, 3];
//! let cols = vec![1, 0, 3, 2];
//! let data = vec![1.0, 1.0, 1.0, 1.0];
//! let graph = CsrArray::from_triplets(&rows, &cols, &data, (4, 4), false).unwrap();
//!
//! // Find connected components
//! let (n_components, labels) = connected_components(&graph, false, "weak", true).unwrap();
//! ```

use crate::error::{SparseError, SparseResult};
use crate::sparray::SparseArray;
use scirs2_core::numeric::Float;
use std::cmp::Ordering;
use std::fmt::Debug;

pub mod connected_components;
pub mod laplacian;
pub mod minimum_spanning_tree;
pub mod shortest_path;
pub mod traversal;

pub use connected_components::*;
pub use laplacian::*;
pub use minimum_spanning_tree::*;
pub use shortest_path::*;
pub use traversal::*;

/// Graph representation modes
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GraphMode {
    /// Treat the matrix as a directed graph
    Directed,
    /// Treat the matrix as an undirected graph
    Undirected,
}

/// Priority queue element for graph algorithms
#[derive(Debug, Clone)]
struct PriorityQueueNode<T>
where
    T: Float + PartialOrd,
{
    distance: T,
    node: usize,
}

impl<T> PartialEq for PriorityQueueNode<T>
where
    T: Float + PartialOrd,
{
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance && self.node == other.node
    }
}

impl<T> Eq for PriorityQueueNode<T> where T: Float + PartialOrd {}

impl<T> PartialOrd for PriorityQueueNode<T>
where
    T: Float + PartialOrd,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Ord for PriorityQueueNode<T>
where
    T: Float + PartialOrd,
{
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse the ordering for min-heap behavior
        other
            .distance
            .partial_cmp(&self.distance)
            .unwrap_or(Ordering::Equal)
    }
}

/// Check if a sparse matrix represents a valid graph
///
/// # Arguments
///
/// * `matrix` - The sparse matrix to check
/// * `directed` - Whether the graph is directed
///
/// # Returns
///
/// Result indicating if the matrix is a valid graph
#[allow(dead_code)]
pub fn validate_graph<T, S>(matrix: &S, directed: bool) -> SparseResult<()>
where
    T: Float + Debug + Copy + 'static,
    S: SparseArray<T>,
{
    let (rows, cols) = matrix.shape();

    // Graph matrices must be square
    if rows != cols {
        return Err(SparseError::ValueError(
            "Graph _matrix must be square".to_string(),
        ));
    }

    // Check for negative weights (not allowed in some algorithms)
    let (row_indices, col_indices, values) = matrix.find();
    for &value in values.iter() {
        if value < T::zero() {
            return Err(SparseError::ValueError(
                "Negative edge weights not supported".to_string(),
            ));
        }
    }

    // For undirected graphs, check symmetry
    if !directed {
        for (i, (&row, &col)) in row_indices.iter().zip(col_indices.iter()).enumerate() {
            if row != col {
                let weight = values[i];
                let reverse_weight = matrix.get(col, row);

                if (weight - reverse_weight).abs() > T::from(1e-10).unwrap() {
                    return Err(SparseError::ValueError(
                        "Undirected graph _matrix must be symmetric".to_string(),
                    ));
                }
            }
        }
    }

    Ok(())
}

/// Convert a sparse matrix to adjacency list representation
///
/// # Arguments
///
/// * `matrix` - The sparse matrix
/// * `directed` - Whether the graph is directed
///
/// # Returns
///
/// Adjacency list as a vector of vectors of (neighbor, weight) pairs
#[allow(dead_code)]
pub fn to_adjacency_list<T, S>(matrix: &S, directed: bool) -> SparseResult<Vec<Vec<(usize, T)>>>
where
    T: Float + Debug + Copy + 'static,
    S: SparseArray<T>,
{
    let (n_, _) = matrix.shape();
    let mut adj_list = vec![Vec::new(); n_];

    let (row_indices, col_indices, values) = matrix.find();

    for (i, (&row, &col)) in row_indices.iter().zip(col_indices.iter()).enumerate() {
        let weight = values[i];

        if !weight.is_zero() {
            adj_list[row].push((col, weight));

            // For undirected graphs, add the reverse edge only if it doesn't already exist
            if !directed && row != col {
                // Check if the reverse edge already exists in the _matrix
                let reverse_exists = row_indices
                    .iter()
                    .zip(col_indices.iter())
                    .any(|(r, c)| *r == col && *c == row);

                if !reverse_exists {
                    adj_list[col].push((row, weight));
                }
            }
        }
    }

    Ok(adj_list)
}

/// Get the number of vertices in a graph matrix
#[allow(dead_code)]
pub fn num_vertices<T, S>(matrix: &S) -> usize
where
    T: Float + Debug + Copy + 'static,
    S: SparseArray<T>,
{
    matrix.shape().0
}

/// Get the number of edges in a graph matrix
#[allow(dead_code)]
pub fn num_edges<T, S>(matrix: &S, directed: bool) -> SparseResult<usize>
where
    T: Float + Debug + Copy + 'static,
    S: SparseArray<T>,
{
    let nnz = matrix.nnz();

    if directed {
        Ok(nnz)
    } else {
        // For undirected graphs, count diagonal elements once and off-diagonal elements half
        let (row_indices, col_indices_, _) = matrix.find();
        let mut diagonal_count = 0;
        let mut off_diagonal_count = 0;

        for (&row, &col) in row_indices.iter().zip(col_indices_.iter()) {
            if row == col {
                diagonal_count += 1;
            } else {
                off_diagonal_count += 1;
            }
        }

        // Off-diagonal edges are counted twice in the _matrix (i,j) and (j,i)
        Ok(diagonal_count + off_diagonal_count / 2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::csr_array::CsrArray;

    fn create_test_graph() -> CsrArray<f64> {
        // Create a simple 4-vertex graph:
        //   0 -- 1
        //   |    |
        //   2 -- 3
        let rows = vec![0, 0, 1, 1, 2, 2, 3, 3];
        let cols = vec![1, 2, 0, 3, 0, 3, 1, 2];
        let data = vec![1.0, 2.0, 1.0, 3.0, 2.0, 1.0, 3.0, 1.0];

        CsrArray::from_triplets(&rows, &cols, &data, (4, 4), false).unwrap()
    }

    #[test]
    fn test_validate_graph_symmetric() {
        let graph = create_test_graph();

        // Should be valid as undirected graph
        assert!(validate_graph(&graph, false).is_ok());

        // Should also be valid as directed graph
        assert!(validate_graph(&graph, true).is_ok());
    }

    #[test]
    fn test_validate_graph_asymmetric() {
        // Create an asymmetric graph
        let rows = vec![0, 1];
        let cols = vec![1, 0];
        let data = vec![1.0, 2.0]; // Different weights

        let graph = CsrArray::from_triplets(&rows, &cols, &data, (2, 2), false).unwrap();

        // Should be valid as directed graph
        assert!(validate_graph(&graph, true).is_ok());

        // Should fail as undirected graph due to asymmetry
        assert!(validate_graph(&graph, false).is_err());
    }

    #[test]
    fn test_validate_graph_negative_weights() {
        let rows = vec![0, 1];
        let cols = vec![1, 0];
        let data = vec![-1.0, 1.0]; // Negative weight

        let graph = CsrArray::from_triplets(&rows, &cols, &data, (2, 2), false).unwrap();

        // Should fail due to negative weights
        assert!(validate_graph(&graph, true).is_err());
        assert!(validate_graph(&graph, false).is_err());
    }

    #[test]
    fn test_to_adjacency_list() {
        let graph = create_test_graph();
        let adj_list = to_adjacency_list(&graph, false).unwrap();

        assert_eq!(adj_list.len(), 4);

        // Vertex 0 should be connected to 1 and 2
        assert_eq!(adj_list[0].len(), 2);
        assert!(adj_list[0].contains(&(1, 1.0)));
        assert!(adj_list[0].contains(&(2, 2.0)));

        // Vertex 1 should be connected to 0 and 3
        assert_eq!(adj_list[1].len(), 2);
        assert!(adj_list[1].contains(&(0, 1.0)));
        assert!(adj_list[1].contains(&(3, 3.0)));
    }

    #[test]
    fn test_num_vertices() {
        let graph = create_test_graph();
        assert_eq!(num_vertices(&graph), 4);
    }

    #[test]
    fn test_num_edges() {
        let graph = create_test_graph();

        // Directed: all 8 edges
        assert_eq!(num_edges(&graph, true).unwrap(), 8);

        // Undirected: 4 unique edges
        assert_eq!(num_edges(&graph, false).unwrap(), 4);
    }
}
