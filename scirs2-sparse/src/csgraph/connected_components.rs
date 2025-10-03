//! Connected components analysis for sparse graphs
//!
//! This module provides efficient algorithms for finding connected components
//! in sparse graphs represented as matrices.

use super::{num_vertices, to_adjacency_list, validate_graph};
use crate::error::{SparseError, SparseResult};
use crate::sparray::SparseArray;
use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::Float;
use std::fmt::Debug;

/// Find connected components in a graph
///
/// # Arguments
///
/// * `graph` - The graph as a sparse matrix
/// * `directed` - Whether to treat the graph as directed
/// * `connection` - Type of connectivity for directed graphs ("weak" or "strong")
/// * `returnlabels` - Whether to return component labels for each vertex
///
/// # Returns
///
/// A tuple containing:
/// - Number of connected components
/// - Optional array of component labels for each vertex
///
/// # Examples
///
/// ```
/// use scirs2_sparse::csgraph::connected_components;
/// use scirs2_sparse::csr_array::CsrArray;
///
/// // Create a graph with two components
/// let rows = vec![0, 1, 2, 3];
/// let cols = vec![1, 0, 3, 2];
/// let data = vec![1.0, 1.0, 1.0, 1.0];
/// let graph = CsrArray::from_triplets(&rows, &cols, &data, (4, 4), false).unwrap();
///
/// let (n_components, labels) = connected_components(&graph, false, "weak", true).unwrap();
/// assert_eq!(n_components, 2);
/// ```
#[allow(dead_code)]
pub fn connected_components<T, S>(
    graph: &S,
    directed: bool,
    connection: &str,
    returnlabels: bool,
) -> SparseResult<(usize, Option<Array1<usize>>)>
where
    T: Float + Debug + Copy + 'static,
    S: SparseArray<T>,
{
    validate_graph(graph, directed)?;

    let connection_type = match connection.to_lowercase().as_str() {
        "weak" => ConnectionType::Weak,
        "strong" => ConnectionType::Strong,
        _ => {
            return Err(SparseError::ValueError(format!(
                "Unknown connection type: {connection}. Use 'weak' or 'strong'"
            )))
        }
    };

    if directed {
        match connection_type {
            ConnectionType::Weak => weakly_connected_components(graph, returnlabels),
            ConnectionType::Strong => strongly_connected_components(graph, returnlabels),
        }
    } else {
        // For undirected graphs, weak and strong connectivity are the same
        undirected_connected_components(graph, returnlabels)
    }
}

/// Connection type for directed graphs
#[derive(Debug, Clone, Copy, PartialEq)]
enum ConnectionType {
    /// Weak connectivity (ignore edge directions)
    Weak,
    /// Strong connectivity (respect edge directions)
    Strong,
}

/// Find connected components in an undirected graph using DFS
#[allow(dead_code)]
pub fn undirected_connected_components<T, S>(
    graph: &S,
    returnlabels: bool,
) -> SparseResult<(usize, Option<Array1<usize>>)>
where
    T: Float + Debug + Copy + 'static,
    S: SparseArray<T>,
{
    let n = num_vertices(graph);
    let adjlist = to_adjacency_list(graph, false)?; // Undirected

    let mut visited = vec![false; n];
    let mut labels = if returnlabels {
        Some(Array1::zeros(n))
    } else {
        None
    };

    let mut component_count = 0;

    for start in 0..n {
        if !visited[start] {
            // Start a new component
            dfs_component(&adjlist, start, &mut visited, component_count, &mut labels);
            component_count += 1;
        }
    }

    Ok((component_count, labels))
}

/// Find weakly connected components in a directed graph
#[allow(dead_code)]
pub fn weakly_connected_components<T, S>(
    graph: &S,
    returnlabels: bool,
) -> SparseResult<(usize, Option<Array1<usize>>)>
where
    T: Float + Debug + Copy + 'static,
    S: SparseArray<T>,
{
    // For weak connectivity, we treat the graph as undirected
    undirected_connected_components(graph, returnlabels)
}

/// Find strongly connected components in a directed graph using Tarjan's algorithm
#[allow(dead_code)]
pub fn strongly_connected_components<T, S>(
    graph: &S,
    returnlabels: bool,
) -> SparseResult<(usize, Option<Array1<usize>>)>
where
    T: Float + Debug + Copy + 'static,
    S: SparseArray<T>,
{
    let n = num_vertices(graph);
    let adjlist = to_adjacency_list(graph, true)?; // Directed

    let mut tarjan = TarjanSCC::<T>::new(n, returnlabels);

    for v in 0..n {
        if tarjan.indices[v] == -1 {
            tarjan.strongconnect(v, &adjlist);
        }
    }

    Ok((tarjan.component_count, tarjan._labels))
}

/// DFS helper for finding a connected component
#[allow(dead_code)]
fn dfs_component<T>(
    adjlist: &[Vec<(usize, T)>],
    start: usize,
    visited: &mut [bool],
    component_id: usize,
    labels: &mut Option<Array1<usize>>,
) where
    T: Float + Debug + Copy + 'static,
{
    let mut stack = vec![start];

    while let Some(node) = stack.pop() {
        if visited[node] {
            continue;
        }

        visited[node] = true;

        if let Some(ref mut label_array) = labels {
            label_array[node] = component_id;
        }

        // Add all unvisited neighbors to the stack
        for &(neighbor, _) in &adjlist[node] {
            if !visited[neighbor] {
                stack.push(neighbor);
            }
        }
    }
}

/// Tarjan's strongly connected components algorithm
struct TarjanSCC<T>
where
    T: Float + Debug + Copy + 'static,
{
    indices: Vec<isize>,
    lowlinks: Vec<isize>,
    on_stack: Vec<bool>,
    stack: Vec<usize>,
    index: isize,
    component_count: usize,
    _labels: Option<Array1<usize>>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> TarjanSCC<T>
where
    T: Float + Debug + Copy + 'static,
{
    fn new(n: usize, returnlabels: bool) -> Self {
        Self {
            indices: vec![-1; n],
            lowlinks: vec![-1; n],
            on_stack: vec![false; n],
            stack: Vec::new(),
            index: 0,
            component_count: 0,
            _labels: if returnlabels {
                Some(Array1::zeros(n))
            } else {
                None
            },
            _phantom: std::marker::PhantomData,
        }
    }

    fn strongconnect(&mut self, v: usize, adjlist: &[Vec<(usize, T)>]) {
        // Set the depth index for v to the smallest unused index
        self.indices[v] = self.index;
        self.lowlinks[v] = self.index;
        self.index += 1;
        self.stack.push(v);
        self.on_stack[v] = true;

        // Consider successors of v
        for &(w, _) in &adjlist[v] {
            if self.indices[w] == -1 {
                // Successor w has not yet been visited; recurse on it
                self.strongconnect(w, adjlist);
                self.lowlinks[v] = self.lowlinks[v].min(self.lowlinks[w]);
            } else if self.on_stack[w] {
                // Successor w is in stack S and hence in the current SCC
                self.lowlinks[v] = self.lowlinks[v].min(self.indices[w]);
            }
        }

        // If v is a root node, pop the stack and create an SCC
        if self.lowlinks[v] == self.indices[v] {
            loop {
                let w = self.stack.pop().unwrap();
                self.on_stack[w] = false;

                if let Some(ref mut labels) = self._labels {
                    labels[w] = self.component_count;
                }

                if w == v {
                    break;
                }
            }
            self.component_count += 1;
        }
    }
}

/// Check if a graph is connected
///
/// # Arguments
///
/// * `graph` - The graph as a sparse matrix
/// * `directed` - Whether the graph is directed
///
/// # Returns
///
/// True if the graph is connected, false otherwise
///
/// # Examples
///
/// ```
/// use scirs2_sparse::csgraph::is_connected;
/// use scirs2_sparse::csr_array::CsrArray;
///
/// // Create a connected graph
/// let rows = vec![0, 1, 1, 2];
/// let cols = vec![1, 0, 2, 1];
/// let data = vec![1.0, 1.0, 1.0, 1.0];
/// let graph = CsrArray::from_triplets(&rows, &cols, &data, (3, 3), false).unwrap();
///
/// assert!(is_connected(&graph, false).unwrap());
/// ```
#[allow(dead_code)]
pub fn is_connected<T, S>(graph: &S, directed: bool) -> SparseResult<bool>
where
    T: Float + Debug + Copy + 'static,
    S: SparseArray<T>,
{
    let (n_components_, _) = connected_components(graph, directed, "strong", false)?;
    Ok(n_components_ == 1)
}

/// Find the largest connected component
///
/// # Arguments
///
/// * `graph` - The graph as a sparse matrix
/// * `directed` - Whether the graph is directed
/// * `connection` - Type of connectivity for directed graphs
///
/// # Returns
///
/// A tuple containing:
/// - Size of the largest component
/// - Optional indices of vertices in the largest component
///
/// # Examples
///
/// ```
/// use scirs2_sparse::csgraph::largest_component;
/// use scirs2_sparse::csr_array::CsrArray;
///
/// // Create a symmetric graph with components of different sizes
/// let rows = vec![0, 1, 1, 2, 2, 3, 3, 4];
/// let cols = vec![1, 0, 2, 1, 3, 2, 4, 3];
/// let data = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
/// let graph = CsrArray::from_triplets(&rows, &cols, &data, (5, 5), false).unwrap();
///
/// let (size, indices) = largest_component(&graph, false, "weak").unwrap();
/// ```
#[allow(dead_code)]
pub fn largest_component<T, S>(
    graph: &S,
    directed: bool,
    connection: &str,
) -> SparseResult<(usize, Vec<usize>)>
where
    T: Float + Debug + Copy + 'static,
    S: SparseArray<T>,
{
    let (n_components, labels) = connected_components(graph, directed, connection, true)?;
    let labels = labels.unwrap();

    // Count the size of each component
    let mut component_sizes = vec![0; n_components];
    for &label in labels.iter() {
        component_sizes[label] += 1;
    }

    // Find the largest component
    let largest_component_id = component_sizes
        .iter()
        .enumerate()
        .max_by_key(|(_, &size)| size)
        .map(|(id_, _)| id_)
        .unwrap_or(0);

    let largest_size = component_sizes[largest_component_id];

    // Collect indices of vertices in the largest component
    let largest_indices: Vec<usize> = labels
        .iter()
        .enumerate()
        .filter_map(|(vertex, &label)| {
            if label == largest_component_id {
                Some(vertex)
            } else {
                None
            }
        })
        .collect();

    Ok((largest_size, largest_indices))
}

/// Extract a subgraph containing only the largest connected component
///
/// # Arguments
///
/// * `graph` - The original graph as a sparse matrix
/// * `directed` - Whether the graph is directed
/// * `connection` - Type of connectivity for directed graphs
///
/// # Returns
///
/// A tuple containing:
/// - The subgraph as a sparse matrix
/// - Mapping from new vertex indices to original vertex indices
///
#[allow(dead_code)]
pub fn extract_largest_component<T, S>(
    graph: &S,
    directed: bool,
    connection: &str,
) -> SparseResult<(S, Vec<usize>)>
where
    T: Float + Debug + Copy + 'static,
    S: SparseArray<T> + Clone,
{
    let (_, vertex_indices) = largest_component(graph, directed, connection)?;

    // Create mapping from old to new indices
    let mut old_to_new = vec![None; num_vertices(graph)];
    for (new_idx, &old_idx) in vertex_indices.iter().enumerate() {
        old_to_new[old_idx] = Some(new_idx);
    }

    // Extract edges within the largest component
    let (row_indices, col_indices, values) = graph.find();
    let mut new_rows = Vec::new();
    let mut new_cols = Vec::new();
    let mut new_values = Vec::new();

    for (i, (&old_row, &old_col)) in row_indices.iter().zip(col_indices.iter()).enumerate() {
        if let (Some(new_row), Some(new_col)) = (old_to_new[old_row], old_to_new[old_col]) {
            new_rows.push(new_row);
            new_cols.push(new_col);
            new_values.push(values[i]);
        }
    }

    // Create the subgraph
    // Note: This is a simplified approach. In a real implementation,
    // we would need to create the specific sparse array type.
    // For now, we'll return the original graph as a placeholder.
    let subgraph = graph.clone();

    Ok((subgraph, vertex_indices))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::csr_array::CsrArray;

    fn create_disconnected_graph() -> CsrArray<f64> {
        // Create a graph with two components:
        // Component 1: 0 -- 1
        // Component 2: 2 -- 3
        let rows = vec![0, 1, 2, 3];
        let cols = vec![1, 0, 3, 2];
        let data = vec![1.0, 1.0, 1.0, 1.0];

        CsrArray::from_triplets(&rows, &cols, &data, (4, 4), false).unwrap()
    }

    fn create_strongly_connected_graph() -> CsrArray<f64> {
        // Create a directed graph with strong connectivity:
        // 0 -> 1 -> 2 -> 0 (strongly connected)
        // 3 (isolated)
        let rows = vec![0, 1, 2];
        let cols = vec![1, 2, 0];
        let data = vec![1.0, 1.0, 1.0];

        CsrArray::from_triplets(&rows, &cols, &data, (4, 4), false).unwrap()
    }

    #[test]
    fn test_undirected_connected_components() {
        let graph = create_disconnected_graph();
        let (n_components, labels) = undirected_connected_components(&graph, true).unwrap();

        assert_eq!(n_components, 2);

        let labels = labels.unwrap();
        // Vertices 0 and 1 should be in the same component
        assert_eq!(labels[0], labels[1]);
        // Vertices 2 and 3 should be in the same component
        assert_eq!(labels[2], labels[3]);
        // But components should be different
        assert_ne!(labels[0], labels[2]);
    }

    #[test]
    fn test_connected_components_api() {
        let graph = create_disconnected_graph();

        // Test undirected
        let (n_components_, _) = connected_components(&graph, false, "weak", false).unwrap();
        assert_eq!(n_components_, 2);

        // Test directed weak connectivity
        let (n_components_, _) = connected_components(&graph, true, "weak", false).unwrap();
        assert_eq!(n_components_, 2);
    }

    #[test]
    fn test_strongly_connected_components() {
        let graph = create_strongly_connected_graph();
        let (n_components, labels) = strongly_connected_components(&graph, true).unwrap();

        // Should have 2 components: {0,1,2} and {3}
        assert_eq!(n_components, 2);

        let labels = labels.unwrap();
        // Vertices 0, 1, 2 should be in the same strongly connected component
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[1], labels[2]);
        // Vertex 3 should be in a different component
        assert_ne!(labels[0], labels[3]);
    }

    #[test]
    fn test_is_connected() {
        let disconnected = create_disconnected_graph();
        assert!(!is_connected(&disconnected, false).unwrap());

        // Create a connected graph
        let rows = vec![0, 1, 1, 2];
        let cols = vec![1, 0, 2, 1];
        let data = vec![1.0, 1.0, 1.0, 1.0];
        let connected = CsrArray::from_triplets(&rows, &cols, &data, (3, 3), false).unwrap();

        assert!(is_connected(&connected, false).unwrap());
    }

    #[test]
    fn test_largest_component() {
        // Create a graph with components of different sizes
        // Component 1: 0 -- 1 -- 2 (size 3)
        // Component 2: 3 -- 4 (size 2)
        // Component 3: 5 (size 1)
        let rows = vec![0, 1, 1, 2, 3, 4];
        let cols = vec![1, 0, 2, 1, 4, 3];
        let data = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let graph = CsrArray::from_triplets(&rows, &cols, &data, (6, 6), false).unwrap();

        let (size, indices) = largest_component(&graph, false, "weak").unwrap();

        assert_eq!(size, 3);
        assert_eq!(indices.len(), 3);
        // Should contain vertices 0, 1, 2
        assert!(indices.contains(&0));
        assert!(indices.contains(&1));
        assert!(indices.contains(&2));
    }

    #[test]
    fn test_single_component() {
        // Test a graph with only one component
        let rows = vec![0, 1, 1, 2];
        let cols = vec![1, 0, 2, 1];
        let data = vec![1.0, 1.0, 1.0, 1.0];
        let graph = CsrArray::from_triplets(&rows, &cols, &data, (3, 3), false).unwrap();

        let (n_components_, _) = connected_components(&graph, false, "weak", false).unwrap();
        assert_eq!(n_components_, 1);

        let (size, indices) = largest_component(&graph, false, "weak").unwrap();
        assert_eq!(size, 3);
        assert_eq!(indices, vec![0, 1, 2]);
    }

    #[test]
    fn test_isolated_vertices() {
        // Create a graph with isolated vertices (symmetric for undirected graph)
        let rows = vec![0, 1];
        let cols = vec![1, 0];
        let data = vec![1.0, 1.0];
        let graph = CsrArray::from_triplets(&rows, &cols, &data, (4, 4), false).unwrap();

        let (n_components, labels) = connected_components(&graph, false, "weak", true).unwrap();

        // Should have 3 components: {0,1}, {2}, {3}
        assert_eq!(n_components, 3);

        let labels = labels.unwrap();
        assert_eq!(labels[0], labels[1]); // 0 and 1 connected
        assert_ne!(labels[0], labels[2]); // 2 is isolated
        assert_ne!(labels[0], labels[3]); // 3 is isolated
        assert_ne!(labels[2], labels[3]); // 2 and 3 are in different components
    }
}
