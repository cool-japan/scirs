//! Shortest path algorithms for sparse graphs
//!
//! This module provides efficient implementations of shortest path algorithms
//! for sparse matrices representing graphs.

use super::{num_vertices, to_adjacency_list, validate_graph, PriorityQueueNode};
use crate::error::{SparseError, SparseResult};
use crate::sparray::SparseArray;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::Float;
use std::collections::BinaryHeap;
use std::fmt::Debug;

/// Shortest path algorithm types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ShortestPathMethod {
    /// Dijkstra's algorithm (non-negative weights only)
    Dijkstra,
    /// Bellman-Ford algorithm (handles negative weights)
    BellmanFord,
    /// Floyd-Warshall algorithm (all pairs shortest paths)
    FloydWarshall,
    /// Automatic selection based on graph properties
    Auto,
}

impl ShortestPathMethod {
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> SparseResult<Self> {
        match s.to_lowercase().as_str() {
            "dijkstra" | "dij" => Ok(Self::Dijkstra),
            "bellman-ford" | "bellman_ford" | "bf" => Ok(Self::BellmanFord),
            "floyd-warshall" | "floyd_warshall" | "fw" => Ok(Self::FloydWarshall),
            "auto" => Ok(Self::Auto),
            _ => Err(SparseError::ValueError(format!(
                "Unknown shortest path method: {s}"
            ))),
        }
    }
}

/// Compute shortest paths in a graph
///
/// # Arguments
///
/// * `graph` - The graph as a sparse matrix
/// * `from_vertex` - Source vertex (None for all pairs)
/// * `to_vertex` - Target vertex (None for all destinations)
/// * `method` - Algorithm to use
/// * `directed` - Whether the graph is directed
/// * `returnpredecessors` - Whether to return predecessor information
///
/// # Returns
///
/// A tuple containing:
/// - Distance matrix/array
/// - Optional predecessor matrix/array
///
/// # Examples
///
/// ```
/// use scirs2_sparse::csgraph::shortest_path;
/// use scirs2_sparse::csr_array::CsrArray;
///
/// // Create a simple graph
/// let rows = vec![0, 0, 1, 2];
/// let cols = vec![1, 2, 2, 0];
/// let data = vec![1.0, 4.0, 2.0, 3.0];
/// let graph = CsrArray::from_triplets(&rows, &cols, &data, (3, 3), false).unwrap();
///
/// // Find shortest paths from vertex 0
/// let (distances_) = shortest_path(
///     &graph, Some(0), None, "dijkstra", true, false
/// ).unwrap();
/// ```
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
pub fn shortest_path<T, S>(
    graph: &S,
    from_vertex: Option<usize>,
    to_vertex: Option<usize>,
    method: &str,
    directed: bool,
    returnpredecessors: bool,
) -> SparseResult<(Array2<T>, Option<Array2<isize>>)>
where
    T: Float + Debug + Copy + 'static,
    S: SparseArray<T>,
{
    validate_graph(graph, directed)?;
    let method = ShortestPathMethod::from_str(method)?;
    let n = num_vertices(graph);

    match (from_vertex, to_vertex) {
        (None, None) => {
            // All pairs shortest paths
            all_pairs_shortest_path(graph, method, directed, returnpredecessors)
        }
        (Some(source), None) => {
            // Single source shortest paths
            let (distances_, predecessors) =
                single_source_shortest_path(graph, source, method, directed, returnpredecessors)?;

            // Convert to matrix format
            let mut dist_matrix = Array2::from_elem((n, n), T::infinity());
            let mut pred_matrix = if returnpredecessors {
                Some(Array2::from_elem((n, n), -1isize))
            } else {
                None
            };

            for i in 0..n {
                dist_matrix[[source, i]] = distances_[i];
                if let Some(ref preds) = predecessors {
                    if let Some(ref mut pred_mat) = pred_matrix {
                        pred_mat[[source, i]] = preds[i];
                    }
                }
            }

            Ok((dist_matrix, pred_matrix))
        }
        (Some(source), Some(target)) => {
            // Single pair shortest path
            let (distances_, predecessors) =
                single_source_shortest_path(graph, source, method, directed, returnpredecessors)?;

            let dist_matrix = Array2::from_elem((1, 1), distances_[target]);
            let pred_matrix = if returnpredecessors {
                let mut pred_mat = Array2::from_elem((1, 1), -1isize);
                if let Some(ref preds) = predecessors {
                    pred_mat[[0, 0]] = preds[target];
                }
                Some(pred_mat)
            } else {
                None
            };

            Ok((dist_matrix, pred_matrix))
        }
        (None, Some(_)) => Err(SparseError::ValueError(
            "Cannot specify target _vertex without source _vertex".to_string(),
        )),
    }
}

/// Single source shortest paths
#[allow(dead_code)]
pub fn single_source_shortest_path<T, S>(
    graph: &S,
    source: usize,
    method: ShortestPathMethod,
    directed: bool,
    returnpredecessors: bool,
) -> SparseResult<(Array1<T>, Option<Array1<isize>>)>
where
    T: Float + Debug + Copy + 'static,
    S: SparseArray<T>,
{
    let n = num_vertices(graph);

    if source >= n {
        return Err(SparseError::ValueError(format!(
            "Source vertex {source} out of bounds for graph with {n} vertices"
        )));
    }

    let actual_method = match method {
        ShortestPathMethod::Auto => {
            // Check if graph has negative weights
            let (_, _, values) = graph.find();
            if values.iter().any(|&w| w < T::zero()) {
                ShortestPathMethod::BellmanFord
            } else {
                ShortestPathMethod::Dijkstra
            }
        }
        m => m,
    };

    match actual_method {
        ShortestPathMethod::Dijkstra => {
            dijkstra_single_source(graph, source, directed, returnpredecessors)
        }
        ShortestPathMethod::BellmanFord => {
            bellman_ford_single_source(graph, source, directed, returnpredecessors)
        }
        _ => Err(SparseError::ValueError(
            "Method not supported for single source shortest paths".to_string(),
        )),
    }
}

/// All pairs shortest paths
#[allow(dead_code)]
pub fn all_pairs_shortest_path<T, S>(
    graph: &S,
    method: ShortestPathMethod,
    directed: bool,
    returnpredecessors: bool,
) -> SparseResult<(Array2<T>, Option<Array2<isize>>)>
where
    T: Float + Debug + Copy + 'static,
    S: SparseArray<T>,
{
    let n = num_vertices(graph);

    let actual_method = match method {
        ShortestPathMethod::Auto => {
            if n <= 100 {
                ShortestPathMethod::FloydWarshall
            } else {
                ShortestPathMethod::Dijkstra
            }
        }
        m => m,
    };

    match actual_method {
        ShortestPathMethod::FloydWarshall => floyd_warshall(graph, directed, returnpredecessors),
        ShortestPathMethod::Dijkstra => {
            // Run Dijkstra from each vertex
            let mut distances = Array2::from_elem((n, n), T::infinity());
            let mut predecessors = if returnpredecessors {
                Some(Array2::from_elem((n, n), -1isize))
            } else {
                None
            };

            for source in 0..n {
                let (dist, pred) =
                    dijkstra_single_source(graph, source, directed, returnpredecessors)?;

                for target in 0..n {
                    distances[[source, target]] = dist[target];
                    if let Some(ref pred_vec) = pred {
                        if let Some(ref mut pred_matrix) = predecessors {
                            pred_matrix[[source, target]] = pred_vec[target];
                        }
                    }
                }
            }

            Ok((distances, predecessors))
        }
        _ => Err(SparseError::ValueError(
            "Method not supported for all pairs shortest paths".to_string(),
        )),
    }
}

/// Dijkstra's algorithm for single source shortest paths
#[allow(dead_code)]
pub fn dijkstra_single_source<T, S>(
    graph: &S,
    source: usize,
    directed: bool,
    returnpredecessors: bool,
) -> SparseResult<(Array1<T>, Option<Array1<isize>>)>
where
    T: Float + Debug + Copy + 'static,
    S: SparseArray<T>,
{
    let n = num_vertices(graph);
    let adj_list = to_adjacency_list(graph, directed)?;

    let mut distances = Array1::from_elem(n, T::infinity());
    let mut predecessors = if returnpredecessors {
        Some(Array1::from_elem(n, -1isize))
    } else {
        None
    };

    distances[source] = T::zero();

    let mut heap = BinaryHeap::new();
    heap.push(PriorityQueueNode {
        distance: T::zero(),
        node: source,
    });

    let mut visited = vec![false; n];

    while let Some(PriorityQueueNode { distance, node }) = heap.pop() {
        if visited[node] {
            continue;
        }

        visited[node] = true;

        // Early termination if we've processed all reachable nodes
        if distance == T::infinity() {
            break;
        }

        for &(neighbor, weight) in &adj_list[node] {
            if visited[neighbor] {
                continue;
            }

            let new_distance = distance + weight;

            if new_distance < distances[neighbor] {
                distances[neighbor] = new_distance;

                if let Some(ref mut preds) = predecessors {
                    preds[neighbor] = node as isize;
                }

                heap.push(PriorityQueueNode {
                    distance: new_distance,
                    node: neighbor,
                });
            }
        }
    }

    Ok((distances, predecessors))
}

/// Bellman-Ford algorithm for single source shortest paths
#[allow(dead_code)]
pub fn bellman_ford_single_source<T, S>(
    graph: &S,
    source: usize,
    directed: bool,
    returnpredecessors: bool,
) -> SparseResult<(Array1<T>, Option<Array1<isize>>)>
where
    T: Float + Debug + Copy + 'static,
    S: SparseArray<T>,
{
    let n = num_vertices(graph);
    let (row_indices, col_indices, values) = graph.find();

    let mut distances = Array1::from_elem(n, T::infinity());
    let mut predecessors = if returnpredecessors {
        Some(Array1::from_elem(n, -1isize))
    } else {
        None
    };

    distances[source] = T::zero();

    // Build edge list
    let mut edges = Vec::new();
    for (i, (&row, &col)) in row_indices.iter().zip(col_indices.iter()).enumerate() {
        let weight = values[i];
        if !weight.is_zero() {
            edges.push((row, col, weight));

            // For undirected graphs, add reverse edge
            if !directed && row != col {
                edges.push((col, row, weight));
            }
        }
    }

    // Relax edges n-1 times
    for _ in 0..(n - 1) {
        let mut updated = false;

        for &(u, v, weight) in &edges {
            if distances[u] != T::infinity() {
                let new_distance = distances[u] + weight;

                if new_distance < distances[v] {
                    distances[v] = new_distance;

                    if let Some(ref mut preds) = predecessors {
                        preds[v] = u as isize;
                    }

                    updated = true;
                }
            }
        }

        // Early termination if no updates
        if !updated {
            break;
        }
    }

    // Check for negative cycles
    for &(u, v, weight) in &edges {
        if distances[u] != T::infinity() && distances[u] + weight < distances[v] {
            return Err(SparseError::ValueError(
                "Graph contains negative cycles".to_string(),
            ));
        }
    }

    Ok((distances, predecessors))
}

/// Floyd-Warshall algorithm for all pairs shortest paths
#[allow(dead_code)]
pub fn floyd_warshall<T, S>(
    graph: &S,
    directed: bool,
    returnpredecessors: bool,
) -> SparseResult<(Array2<T>, Option<Array2<isize>>)>
where
    T: Float + Debug + Copy + 'static,
    S: SparseArray<T>,
{
    let n = num_vertices(graph);

    // Initialize distance matrix
    let mut distances = Array2::from_elem((n, n), T::infinity());
    let mut predecessors = if returnpredecessors {
        Some(Array2::from_elem((n, n), -1isize))
    } else {
        None
    };

    // Set diagonal to zero
    for i in 0..n {
        distances[[i, i]] = T::zero();
    }

    // Fill in the initial distances from the graph
    let (row_indices, col_indices, values) = graph.find();
    for (i, (&row, &col)) in row_indices.iter().zip(col_indices.iter()).enumerate() {
        let weight = values[i];
        if !weight.is_zero() {
            distances[[row, col]] = weight;

            if let Some(ref mut preds) = predecessors {
                if row != col {
                    preds[[row, col]] = row as isize;
                }
            }

            // For undirected graphs, set symmetric entries
            if !directed && row != col {
                distances[[col, row]] = weight;

                if let Some(ref mut preds) = predecessors {
                    preds[[col, row]] = col as isize;
                }
            }
        }
    }

    // Floyd-Warshall main loop
    for k in 0..n {
        for i in 0..n {
            for j in 0..n {
                let through_k = distances[[i, k]] + distances[[k, j]];

                if through_k < distances[[i, j]] {
                    distances[[i, j]] = through_k;

                    if let Some(ref mut preds) = predecessors {
                        preds[[i, j]] = preds[[k, j]];
                    }
                }
            }
        }
    }

    Ok((distances, predecessors))
}

/// Reconstruct shortest path from predecessor information
#[allow(dead_code)]
pub fn reconstruct_path(
    predecessors: &Array1<isize>,
    source: usize,
    target: usize,
) -> SparseResult<Vec<usize>> {
    let mut path = Vec::new();
    let mut current = target;

    // Check if target is reachable
    if predecessors[target] == -1 && source != target {
        return Ok(path); // Empty path means no path exists
    }

    while current != source {
        path.push(current);

        let pred = predecessors[current];
        if pred == -1 {
            return Err(SparseError::ValueError(
                "Invalid predecessor information".to_string(),
            ));
        }

        current = pred as usize;
    }

    path.push(source);
    path.reverse();

    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::csr_array::CsrArray;
    use approx::assert_relative_eq;

    fn create_test_graph() -> CsrArray<f64> {
        // Create a simple graph:
        //   0 --(1)-- 1
        //   |         |
        //  (2)       (3)
        //   |         |
        //   2 --(1)-- 3
        let rows = vec![0, 0, 1, 1, 2, 2, 3, 3];
        let cols = vec![1, 2, 0, 3, 0, 3, 1, 2];
        let data = vec![1.0, 2.0, 1.0, 3.0, 2.0, 1.0, 3.0, 1.0];

        CsrArray::from_triplets(&rows, &cols, &data, (4, 4), false).unwrap()
    }

    #[test]
    fn test_dijkstra_single_source() {
        let graph = create_test_graph();
        let (distances_, _) = dijkstra_single_source(&graph, 0, false, false).unwrap();

        assert_relative_eq!(distances_[0], 0.0);
        assert_relative_eq!(distances_[1], 1.0);
        assert_relative_eq!(distances_[2], 2.0);
        assert_relative_eq!(distances_[3], 3.0); // 0->2->3
    }

    #[test]
    fn test_dijkstra_withpredecessors() {
        let graph = create_test_graph();
        let (_distances, predecessors) = dijkstra_single_source(&graph, 0, false, true).unwrap();
        let preds = predecessors.unwrap();

        assert_eq!(preds[0], -1); // Source has no predecessor
        assert_eq!(preds[1], 0); // 1's predecessor is 0
        assert_eq!(preds[2], 0); // 2's predecessor is 0
        assert_eq!(preds[3], 2); // 3's predecessor is 2 (via shortest path 0->2->3)
    }

    #[test]
    fn test_bellman_ford() {
        let graph = create_test_graph();
        let (distances_, _) = bellman_ford_single_source(&graph, 0, false, false).unwrap();

        assert_relative_eq!(distances_[0], 0.0);
        assert_relative_eq!(distances_[1], 1.0);
        assert_relative_eq!(distances_[2], 2.0);
        assert_relative_eq!(distances_[3], 3.0);
    }

    #[test]
    fn test_floyd_warshall() {
        let graph = create_test_graph();
        let (distances_, _) = floyd_warshall(&graph, false, false).unwrap();

        // Check distances from vertex 0
        assert_relative_eq!(distances_[[0, 0]], 0.0);
        assert_relative_eq!(distances_[[0, 1]], 1.0);
        assert_relative_eq!(distances_[[0, 2]], 2.0);
        assert_relative_eq!(distances_[[0, 3]], 3.0);

        // Check distances from vertex 1
        assert_relative_eq!(distances_[[1, 0]], 1.0);
        assert_relative_eq!(distances_[[1, 1]], 0.0);
        assert_relative_eq!(distances_[[1, 2]], 3.0); // 1->0->2
        assert_relative_eq!(distances_[[1, 3]], 3.0); // 1->3
    }

    #[test]
    fn test_shortest_path_api() {
        let graph = create_test_graph();

        // Single source
        let (distances_, _) =
            shortest_path(&graph, Some(0), None, "dijkstra", false, false).unwrap();
        assert_relative_eq!(distances_[[0, 1]], 1.0);
        assert_relative_eq!(distances_[[0, 3]], 3.0);

        // Single pair
        let (distance, _) = shortest_path(&graph, Some(0), Some(3), "auto", false, false).unwrap();
        assert_relative_eq!(distance[[0, 0]], 3.0);
    }

    #[test]
    fn test_reconstruct_path() {
        let graph = create_test_graph();
        let (_, predecessors) = dijkstra_single_source(&graph, 0, false, true).unwrap();
        let preds = predecessors.unwrap();

        let path = reconstruct_path(&preds, 0, 3).unwrap();
        assert_eq!(path, vec![0, 2, 3]); // Shortest path 0->2->3
    }

    #[test]
    fn test_negative_cycle_detection() {
        // Create a graph with a negative cycle
        let rows = vec![0, 1, 2];
        let cols = vec![1, 2, 0];
        let data = vec![1.0, 1.0, -3.0]; // Cycle 0->1->2->0 with total weight -1

        let graph = CsrArray::from_triplets(&rows, &cols, &data, (3, 3), false).unwrap();

        // Bellman-Ford should detect the negative cycle
        let result = bellman_ford_single_source(&graph, 0, true, false);
        assert!(result.is_err());
    }
}
