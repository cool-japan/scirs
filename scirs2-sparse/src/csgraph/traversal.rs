//! Graph traversal algorithms for sparse graphs
//!
//! This module provides breadth-first search (BFS) and depth-first search (DFS)
//! algorithms for sparse matrices representing graphs.

use super::{num_vertices, to_adjacency_list, validate_graph};
use crate::error::{SparseError, SparseResult};
use crate::sparray::SparseArray;
use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::Float;
use std::collections::VecDeque;
use std::fmt::Debug;

/// Traversal order types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TraversalOrder {
    /// Breadth-first search
    BreadthFirst,
    /// Depth-first search
    DepthFirst,
}

impl TraversalOrder {
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> SparseResult<Self> {
        match s.to_lowercase().as_str() {
            "breadth_first" | "bfs" | "breadth-first" => Ok(Self::BreadthFirst),
            "depth_first" | "dfs" | "depth-first" => Ok(Self::DepthFirst),
            _ => Err(SparseError::ValueError(format!(
                "Unknown traversal order: {s}"
            ))),
        }
    }
}

/// Perform graph traversal from a starting vertex
///
/// # Arguments
///
/// * `graph` - The graph as a sparse matrix
/// * `start` - Starting vertex
/// * `directed` - Whether the graph is directed
/// * `order` - Traversal order (BFS or DFS)
/// * `return_predecessors` - Whether to return predecessor information
///
/// # Returns
///
/// A tuple containing:
/// - Traversal order as a vector of vertex indices
/// - Optional predecessor array
///
/// # Examples
///
/// ```
/// use scirs2_sparse::csgraph::traversegraph;
/// use scirs2_sparse::csr_array::CsrArray;
///
/// // Create a simple graph
/// let rows = vec![0, 1, 1, 2];
/// let cols = vec![1, 0, 2, 1];
/// let data = vec![1.0, 1.0, 1.0, 1.0];
/// let graph = CsrArray::from_triplets(&rows, &cols, &data, (3, 3), false).unwrap();
///
/// // Perform BFS from vertex 0
/// let (order, _) = traversegraph(&graph, 0, false, "bfs", false).unwrap();
/// ```
#[allow(dead_code)]
pub fn traversegraph<T, S>(
    graph: &S,
    start: usize,
    directed: bool,
    order: &str,
    return_predecessors: bool,
) -> SparseResult<(Vec<usize>, Option<Array1<isize>>)>
where
    T: Float + Debug + Copy + 'static,
    S: SparseArray<T>,
{
    validate_graph(graph, directed)?;
    let n = num_vertices(graph);

    if start >= n {
        return Err(SparseError::ValueError(format!(
            "Start vertex {start} out of bounds for graph with {n} vertices"
        )));
    }

    let traversal_order = TraversalOrder::from_str(order)?;

    match traversal_order {
        TraversalOrder::BreadthFirst => {
            breadth_first_search(graph, start, directed, return_predecessors)
        }
        TraversalOrder::DepthFirst => {
            depth_first_search(graph, start, directed, return_predecessors)
        }
    }
}

/// Breadth-first search traversal
#[allow(dead_code)]
pub fn breadth_first_search<T, S>(
    graph: &S,
    start: usize,
    directed: bool,
    return_predecessors: bool,
) -> SparseResult<(Vec<usize>, Option<Array1<isize>>)>
where
    T: Float + Debug + Copy + 'static,
    S: SparseArray<T>,
{
    let n = num_vertices(graph);
    let adj_list = to_adjacency_list(graph, directed)?;

    let mut visited = vec![false; n];
    let mut queue = VecDeque::new();
    let mut traversal_order = Vec::new();
    let mut predecessors = if return_predecessors {
        Some(Array1::from_elem(n, -1isize))
    } else {
        None
    };

    // Start BFS from the given vertex
    queue.push_back(start);
    visited[start] = true;

    while let Some(current) = queue.pop_front() {
        traversal_order.push(current);

        // Visit all unvisited neighbors
        for &(neighbor, _) in &adj_list[current] {
            if !visited[neighbor] {
                visited[neighbor] = true;
                queue.push_back(neighbor);

                if let Some(ref mut preds) = predecessors {
                    preds[neighbor] = current as isize;
                }
            }
        }
    }

    Ok((traversal_order, predecessors))
}

/// Depth-first search traversal
#[allow(dead_code)]
pub fn depth_first_search<T, S>(
    graph: &S,
    start: usize,
    directed: bool,
    return_predecessors: bool,
) -> SparseResult<(Vec<usize>, Option<Array1<isize>>)>
where
    T: Float + Debug + Copy + 'static,
    S: SparseArray<T>,
{
    let n = num_vertices(graph);
    let adj_list = to_adjacency_list(graph, directed)?;

    let mut visited = vec![false; n];
    let mut stack = Vec::new();
    let mut traversal_order = Vec::new();
    let mut predecessors = if return_predecessors {
        Some(Array1::from_elem(n, -1isize))
    } else {
        None
    };

    // Start DFS from the given vertex
    stack.push(start);

    while let Some(current) = stack.pop() {
        if visited[current] {
            continue;
        }

        visited[current] = true;
        traversal_order.push(current);

        // Add all unvisited neighbor_s to the stack (in reverse order for consistent ordering)
        let mut neighbor_s: Vec<_> = adj_list[current]
            .iter()
            .filter(|&(neighbor_, _)| !visited[*neighbor_])
            .collect();
        neighbor_s.reverse(); // Reverse to maintain left-to-right order when popping

        for &(neighbor_, _) in neighbor_s {
            if !visited[neighbor_] {
                stack.push(neighbor_);

                if let Some(ref mut preds) = predecessors {
                    if preds[neighbor_] == -1 {
                        preds[neighbor_] = current as isize;
                    }
                }
            }
        }
    }

    Ok((traversal_order, predecessors))
}

/// Recursive depth-first search traversal
#[allow(dead_code)]
pub fn depth_first_search_recursive<T, S>(
    graph: &S,
    start: usize,
    directed: bool,
    return_predecessors: bool,
) -> SparseResult<(Vec<usize>, Option<Array1<isize>>)>
where
    T: Float + Debug + Copy + 'static,
    S: SparseArray<T>,
{
    let n = num_vertices(graph);
    let adj_list = to_adjacency_list(graph, directed)?;

    let mut visited = vec![false; n];
    let mut traversal_order = Vec::new();
    let mut predecessors = if return_predecessors {
        Some(Array1::from_elem(n, -1isize))
    } else {
        None
    };

    dfs_recursive_helper::<T>(
        start,
        &adj_list,
        &mut visited,
        &mut traversal_order,
        &mut predecessors,
    );

    Ok((traversal_order, predecessors))
}

/// Helper function for recursive DFS
#[allow(dead_code)]
fn dfs_recursive_helper<T>(
    node: usize,
    adj_list: &[Vec<(usize, T)>],
    visited: &mut [bool],
    traversal_order: &mut Vec<usize>,
    predecessors: &mut Option<Array1<isize>>,
) where
    T: Float + Debug + Copy + 'static,
{
    visited[node] = true;
    traversal_order.push(node);

    for &(neighbor_, _) in &adj_list[node] {
        if !visited[neighbor_] {
            if let Some(ref mut preds) = predecessors {
                preds[neighbor_] = node as isize;
            }
            dfs_recursive_helper(neighbor_, adj_list, visited, traversal_order, predecessors);
        }
    }
}

/// Compute distances from a source vertex using BFS
///
/// # Arguments
///
/// * `graph` - The graph as a sparse matrix (unweighted)
/// * `start` - Starting vertex
/// * `directed` - Whether the graph is directed
///
/// # Returns
///
/// Array of distances from the start vertex to all other vertices
///
/// # Examples
///
/// ```
/// use scirs2_sparse::csgraph::bfs_distances;
/// use scirs2_sparse::csr_array::CsrArray;
///
/// let rows = vec![0, 1, 1, 2];
/// let cols = vec![1, 0, 2, 1];
/// let data = vec![1.0, 1.0, 1.0, 1.0];
/// let graph = CsrArray::from_triplets(&rows, &cols, &data, (3, 3), false).unwrap();
///
/// let distances = bfs_distances(&graph, 0, false).unwrap();
/// ```
#[allow(dead_code)]
pub fn bfs_distances<T, S>(graph: &S, start: usize, directed: bool) -> SparseResult<Array1<isize>>
where
    T: Float + Debug + Copy + 'static,
    S: SparseArray<T>,
{
    let n = num_vertices(graph);
    let adj_list = to_adjacency_list(graph, directed)?;

    if start >= n {
        return Err(SparseError::ValueError(format!(
            "Start vertex {start} out of bounds for graph with {n} vertices"
        )));
    }

    let mut distances = Array1::from_elem(n, -1isize);
    let mut queue = VecDeque::new();

    // Start BFS
    distances[start] = 0;
    queue.push_back(start);

    while let Some(current) = queue.pop_front() {
        let current_distance = distances[current];

        for &(neighbor_, _) in &adj_list[current] {
            if distances[neighbor_] == -1 {
                distances[neighbor_] = current_distance + 1;
                queue.push_back(neighbor_);
            }
        }
    }

    Ok(distances)
}

/// Check if there is a path between two vertices
///
/// # Arguments
///
/// * `graph` - The graph as a sparse matrix
/// * `source` - Source vertex
/// * `target` - Target vertex
/// * `directed` - Whether the graph is directed
///
/// # Returns
///
/// True if there is a path from source to target, false otherwise
///
/// # Examples
///
/// ```
/// use scirs2_sparse::csgraph::has_path;
/// use scirs2_sparse::csr_array::CsrArray;
///
/// let rows = vec![0, 1, 1, 2];
/// let cols = vec![1, 0, 2, 1];
/// let data = vec![1.0, 1.0, 1.0, 1.0];
/// let graph = CsrArray::from_triplets(&rows, &cols, &data, (3, 3), false).unwrap();
///
/// assert!(has_path(&graph, 0, 2, false).unwrap());
/// ```
#[allow(dead_code)]
pub fn has_path<T, S>(graph: &S, source: usize, target: usize, directed: bool) -> SparseResult<bool>
where
    T: Float + Debug + Copy + 'static,
    S: SparseArray<T>,
{
    let n = num_vertices(graph);

    if source >= n || target >= n {
        return Err(SparseError::ValueError(format!(
            "Vertex index out of bounds for graph with {n} vertices"
        )));
    }

    if source == target {
        return Ok(true);
    }

    let (traversal_order, _) = breadth_first_search(graph, source, directed, false)?;
    Ok(traversal_order.contains(&target))
}

/// Find all vertices reachable from a source vertex
///
/// # Arguments
///
/// * `graph` - The graph as a sparse matrix
/// * `source` - Source vertex
/// * `directed` - Whether the graph is directed
///
/// # Returns
///
/// Vector of all vertices reachable from the source
///
/// # Examples
///
/// ```
/// use scirs2_sparse::csgraph::reachable_vertices;
/// use scirs2_sparse::csr_array::CsrArray;
///
/// let rows = vec![0, 1, 1, 2];
/// let cols = vec![1, 0, 2, 1];
/// let data = vec![1.0, 1.0, 1.0, 1.0];
/// let graph = CsrArray::from_triplets(&rows, &cols, &data, (3, 3), false).unwrap();
///
/// let reachable = reachable_vertices(&graph, 0, false).unwrap();
/// ```
#[allow(dead_code)]
pub fn reachable_vertices<T, S>(
    graph: &S,
    source: usize,
    directed: bool,
) -> SparseResult<Vec<usize>>
where
    T: Float + Debug + Copy + 'static,
    S: SparseArray<T>,
{
    let (traversal_order, _) = breadth_first_search(graph, source, directed, false)?;
    Ok(traversal_order)
}

/// Topological sort of a directed acyclic graph (DAG)
///
/// # Arguments
///
/// * `graph` - The directed graph as a sparse matrix
///
/// # Returns
///
/// Topologically sorted order of vertices, or error if the graph has cycles
///
/// # Examples
///
/// ```
/// use scirs2_sparse::csgraph::topological_sort;
/// use scirs2_sparse::csr_array::CsrArray;
///
/// // Create a DAG: 0 -> 1 -> 2
/// let rows = vec![0, 1];
/// let cols = vec![1, 2];
/// let data = vec![1.0, 1.0];
/// let graph = CsrArray::from_triplets(&rows, &cols, &data, (3, 3), false).unwrap();
///
/// let topo_order = topological_sort(&graph).unwrap();
/// ```
#[allow(dead_code)]
pub fn topological_sort<T, S>(graph: &S) -> SparseResult<Vec<usize>>
where
    T: Float + Debug + Copy + 'static,
    S: SparseArray<T>,
{
    let n = num_vertices(graph);
    let adj_list = to_adjacency_list(graph, true)?; // Must be directed

    // Compute in-degrees
    let mut in_degree = vec![0; n];
    for adj in &adj_list {
        for &(neighbor_, _) in adj {
            in_degree[neighbor_] += 1;
        }
    }

    // Initialize queue with vertices having in-degree 0
    let mut queue = VecDeque::new();
    for (vertex, &degree) in in_degree.iter().enumerate() {
        if degree == 0 {
            queue.push_back(vertex);
        }
    }

    let mut topo_order = Vec::new();

    while let Some(vertex) = queue.pop_front() {
        topo_order.push(vertex);

        // Remove this vertex and update in-degrees of its neighbor_s
        for &(neighbor_, _) in &adj_list[vertex] {
            in_degree[neighbor_] -= 1;
            if in_degree[neighbor_] == 0 {
                queue.push_back(neighbor_);
            }
        }
    }

    // Check if all vertices were processed (no cycles)
    if topo_order.len() != n {
        return Err(SparseError::ValueError(
            "Graph contains cycles - topological sort not possible".to_string(),
        ));
    }

    Ok(topo_order)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::csr_array::CsrArray;

    fn create_testgraph() -> CsrArray<f64> {
        // Create a simple connected graph:
        //   0 -- 1
        //   |    |
        //   2 -- 3
        let rows = vec![0, 0, 1, 1, 2, 2, 3, 3];
        let cols = vec![1, 2, 0, 3, 0, 3, 1, 2];
        let data = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

        CsrArray::from_triplets(&rows, &cols, &data, (4, 4), false).unwrap()
    }

    fn create_dag() -> CsrArray<f64> {
        // Create a DAG: 0 -> 1 -> 3, 0 -> 2 -> 3
        let rows = vec![0, 0, 1, 2];
        let cols = vec![1, 2, 3, 3];
        let data = vec![1.0, 1.0, 1.0, 1.0];

        CsrArray::from_triplets(&rows, &cols, &data, (4, 4), false).unwrap()
    }

    #[test]
    fn test_bfs() {
        let graph = create_testgraph();
        let (order, predecessors) = breadth_first_search(&graph, 0, false, true).unwrap();

        // Should visit all vertices
        assert_eq!(order.len(), 4);
        assert!(order.contains(&0));
        assert!(order.contains(&1));
        assert!(order.contains(&2));
        assert!(order.contains(&3));

        // First vertex should be the start
        assert_eq!(order[0], 0);

        // Check predecessors
        let preds = predecessors.unwrap();
        assert_eq!(preds[0], -1); // Start vertex has no predecessor
        assert!(preds[1] == 0); // 1's predecessor should be 0
        assert!(preds[2] == 0); // 2's predecessor should be 0
    }

    #[test]
    fn test_dfs() {
        let graph = create_testgraph();
        let (order, _) = depth_first_search(&graph, 0, false, false).unwrap();

        // Should visit all vertices
        assert_eq!(order.len(), 4);
        assert!(order.contains(&0));
        assert!(order.contains(&1));
        assert!(order.contains(&2));
        assert!(order.contains(&3));

        // First vertex should be the start
        assert_eq!(order[0], 0);
    }

    #[test]
    fn test_dfs_recursive() {
        let graph = create_testgraph();
        let (order, _) = depth_first_search_recursive(&graph, 0, false, false).unwrap();

        // Should visit all vertices
        assert_eq!(order.len(), 4);
        assert!(order.contains(&0));
        assert!(order.contains(&1));
        assert!(order.contains(&2));
        assert!(order.contains(&3));

        // First vertex should be the start
        assert_eq!(order[0], 0);
    }

    #[test]
    fn test_traversegraph_api() {
        let graph = create_testgraph();

        // Test BFS
        let (bfs_order, _) = traversegraph(&graph, 0, false, "bfs", false).unwrap();
        assert_eq!(bfs_order[0], 0);
        assert_eq!(bfs_order.len(), 4);

        // Test DFS
        let (dfs_order, _) = traversegraph(&graph, 0, false, "dfs", false).unwrap();
        assert_eq!(dfs_order[0], 0);
        assert_eq!(dfs_order.len(), 4);
    }

    #[test]
    fn test_bfs_distances() {
        let graph = create_testgraph();
        let distances = bfs_distances(&graph, 0, false).unwrap();

        assert_eq!(distances[0], 0); // Distance to self is 0
        assert_eq!(distances[1], 1); // Direct neighbor_
        assert_eq!(distances[2], 1); // Direct neighbor_
        assert_eq!(distances[3], 2); // Via 1 or 2
    }

    #[test]
    fn test_has_path() {
        let graph = create_testgraph();

        // All vertices are connected
        assert!(has_path(&graph, 0, 3, false).unwrap());
        assert!(has_path(&graph, 1, 2, false).unwrap());
        assert!(has_path(&graph, 0, 0, false).unwrap()); // Self path

        // Test disconnected graph
        let rows = vec![0, 2];
        let cols = vec![1, 3];
        let data = vec![1.0, 1.0];
        let disconnected = CsrArray::from_triplets(&rows, &cols, &data, (4, 4), false).unwrap();

        assert!(has_path(&disconnected, 0, 1, false).unwrap());
        assert!(!has_path(&disconnected, 0, 2, false).unwrap());
    }

    #[test]
    fn test_reachable_vertices() {
        let graph = create_testgraph();
        let reachable = reachable_vertices(&graph, 0, false).unwrap();

        // All vertices should be reachable
        assert_eq!(reachable.len(), 4);
        assert!(reachable.contains(&0));
        assert!(reachable.contains(&1));
        assert!(reachable.contains(&2));
        assert!(reachable.contains(&3));
    }

    #[test]
    fn test_topological_sort() {
        let dag = create_dag();
        let topo_order = topological_sort(&dag).unwrap();

        assert_eq!(topo_order.len(), 4);

        // 0 should come before 1 and 2
        let pos_0 = topo_order.iter().position(|&x| x == 0).unwrap();
        let pos_1 = topo_order.iter().position(|&x| x == 1).unwrap();
        let pos_2 = topo_order.iter().position(|&x| x == 2).unwrap();
        let pos_3 = topo_order.iter().position(|&x| x == 3).unwrap();

        assert!(pos_0 < pos_1);
        assert!(pos_0 < pos_2);
        assert!(pos_1 < pos_3);
        assert!(pos_2 < pos_3);
    }

    #[test]
    fn test_topological_sort_cycle() {
        // Create a graph with a cycle: 0 -> 1 -> 2 -> 0
        let rows = vec![0, 1, 2];
        let cols = vec![1, 2, 0];
        let data = vec![1.0, 1.0, 1.0];
        let cyclic = CsrArray::from_triplets(&rows, &cols, &data, (3, 3), false).unwrap();

        // Should fail due to cycle
        assert!(topological_sort(&cyclic).is_err());
    }

    #[test]
    fn test_invalid_start_vertex() {
        let graph = create_testgraph();

        // Test out of bounds start vertex
        assert!(traversegraph(&graph, 10, false, "bfs", false).is_err());
        assert!(bfs_distances(&graph, 10, false).is_err());
    }
}
