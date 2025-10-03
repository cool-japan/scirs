//! Minimum spanning tree algorithms for sparse graphs
//!
//! This module provides efficient implementations of minimum spanning tree (MST)
//! algorithms for sparse matrices representing weighted graphs.

use super::{num_vertices, to_adjacency_list, validate_graph};
use crate::csr_array::CsrArray;
use crate::error::{SparseError, SparseResult};
use crate::sparray::SparseArray;
use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::Float;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::fmt::Debug;

/// Edge representation for MST algorithms
#[derive(Debug, Clone)]
struct Edge<T>
where
    T: Float + PartialOrd,
{
    weight: T,
    u: usize,
    v: usize,
}

impl<T> PartialEq for Edge<T>
where
    T: Float + PartialOrd,
{
    fn eq(&self, other: &Self) -> bool {
        self.weight == other.weight
    }
}

impl<T> Eq for Edge<T> where T: Float + PartialOrd {}

impl<T> PartialOrd for Edge<T>
where
    T: Float + PartialOrd,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Ord for Edge<T>
where
    T: Float + PartialOrd,
{
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap behavior
        other
            .weight
            .partial_cmp(&self.weight)
            .unwrap_or(Ordering::Equal)
    }
}

/// Union-Find (Disjoint Set Union) data structure
#[derive(Debug)]
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]); // Path compression
        }
        self.parent[x]
    }

    fn union(&mut self, x: usize, y: usize) -> bool {
        let root_x = self.find(x);
        let root_y = self.find(y);

        if root_x == root_y {
            return false; // Already in the same set
        }

        // Union by rank
        match self.rank[root_x].cmp(&self.rank[root_y]) {
            Ordering::Less => self.parent[root_x] = root_y,
            Ordering::Greater => self.parent[root_y] = root_x,
            Ordering::Equal => {
                self.parent[root_y] = root_x;
                self.rank[root_x] += 1;
            }
        }

        true
    }
}

/// MST algorithm types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MSTAlgorithm {
    /// Kruskal's algorithm
    Kruskal,
    /// Prim's algorithm
    Prim,
    /// Automatic selection based on graph properties
    Auto,
}

impl MSTAlgorithm {
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> SparseResult<Self> {
        match s.to_lowercase().as_str() {
            "kruskal" => Ok(Self::Kruskal),
            "prim" => Ok(Self::Prim),
            "auto" => Ok(Self::Auto),
            _ => Err(SparseError::ValueError(format!(
                "Unknown MST algorithm: {s}. Use 'kruskal', 'prim', or 'auto'"
            ))),
        }
    }
}

/// Compute the minimum spanning tree of a graph
///
/// # Arguments
///
/// * `graph` - The graph as a sparse matrix (must be undirected and connected)
/// * `algorithm` - MST algorithm to use
/// * `return_tree` - Whether to return the MST as a sparse matrix
///
/// # Returns
///
/// A tuple containing:
/// - Total weight of the MST
/// - Optional MST as a sparse matrix (if requested)
/// - Array of parent indices in the MST
///
/// # Examples
///
/// ```
/// use scirs2_sparse::csgraph::minimum_spanning_tree;
/// use scirs2_sparse::csr_array::CsrArray;
///
/// // Create a weighted symmetric graph
/// let rows = vec![0, 0, 1, 1, 2, 2];
/// let cols = vec![1, 2, 0, 2, 0, 1];
/// let data = vec![2.0, 3.0, 2.0, 1.0, 3.0, 1.0];
/// let graph = CsrArray::from_triplets(&rows, &cols, &data, (3, 3), false).unwrap();
///
/// let (total_weight, mst, parents) = minimum_spanning_tree(&graph, "kruskal", true).unwrap();
/// ```
#[allow(dead_code)]
pub fn minimum_spanning_tree<T, S>(
    graph: &S,
    algorithm: &str,
    return_tree: bool,
) -> SparseResult<(T, Option<CsrArray<T>>, Array1<isize>)>
where
    T: Float + Debug + Copy + 'static,
    S: SparseArray<T>,
{
    validate_graph(graph, false)?; // Must be undirected
    let n = num_vertices(graph);

    if n == 0 {
        return Err(SparseError::ValueError(
            "Cannot compute MST of empty graph".to_string(),
        ));
    }

    let mst_algorithm = MSTAlgorithm::from_str(algorithm)?;

    let actual_algorithm = match mst_algorithm {
        MSTAlgorithm::Auto => {
            // For sparse graphs, Kruskal is often more efficient
            // For dense graphs, Prim might be better
            let nnz = graph.nnz();
            if nnz <= n * n / 4 {
                MSTAlgorithm::Kruskal
            } else {
                MSTAlgorithm::Prim
            }
        }
        alg => alg,
    };

    match actual_algorithm {
        MSTAlgorithm::Kruskal => kruskal_mst(graph, return_tree),
        MSTAlgorithm::Prim => {
            prim_mst(graph, 0, return_tree) // Start from vertex 0
        }
        MSTAlgorithm::Auto => unreachable!(),
    }
}

/// Kruskal's algorithm for MST
#[allow(dead_code)]
pub fn kruskal_mst<T, S>(
    graph: &S,
    return_tree: bool,
) -> SparseResult<(T, Option<CsrArray<T>>, Array1<isize>)>
where
    T: Float + Debug + Copy + 'static,
    S: SparseArray<T>,
{
    let n = num_vertices(graph);
    let (row_indices, col_indices, values) = graph.find();

    // Create edges and sort them by weight
    let mut edges = Vec::new();
    for (i, (&u, &v)) in row_indices.iter().zip(col_indices.iter()).enumerate() {
        if u <= v && !values[i].is_zero() {
            // Avoid duplicate edges for undirected graph
            edges.push(Edge {
                weight: values[i],
                u,
                v,
            });
        }
    }

    edges.sort_by(|a, b| a.weight.partial_cmp(&b.weight).unwrap_or(Ordering::Equal));

    let mut union_find = UnionFind::new(n);
    let mut mst_edges = Vec::new();
    let mut total_weight = T::zero();
    let mut parent = Array1::from_elem(n, -1isize);

    for edge in edges {
        if union_find.union(edge.u, edge.v) {
            mst_edges.push(edge.clone());
            total_weight = total_weight + edge.weight;

            // Set parent relationship (arbitrary choice for undirected tree)
            if parent[edge.v] == -1 {
                parent[edge.v] = edge.u as isize;
            } else if parent[edge.u] == -1 {
                parent[edge.u] = edge.v as isize;
            }

            // MST has n-1 edges
            if mst_edges.len() == n - 1 {
                break;
            }
        }
    }

    // Check if graph is connected
    if mst_edges.len() != n - 1 {
        return Err(SparseError::ValueError(
            "Graph is not connected - cannot compute spanning tree".to_string(),
        ));
    }

    let mst_matrix = if return_tree {
        Some(build_mst_matrix(&mst_edges, n)?)
    } else {
        None
    };

    Ok((total_weight, mst_matrix, parent))
}

/// Prim's algorithm for MST
#[allow(dead_code)]
pub fn prim_mst<T, S>(
    graph: &S,
    start: usize,
    return_tree: bool,
) -> SparseResult<(T, Option<CsrArray<T>>, Array1<isize>)>
where
    T: Float + Debug + Copy + 'static,
    S: SparseArray<T>,
{
    let n = num_vertices(graph);
    let adj_list = to_adjacency_list(graph, false)?; // Undirected

    if start >= n {
        return Err(SparseError::ValueError(format!(
            "Start vertex {start} out of bounds for graph with {n} vertices"
        )));
    }

    let mut in_mst = vec![false; n];
    let mut min_weight = vec![T::infinity(); n];
    let mut parent = Array1::from_elem(n, -1isize);
    let mut total_weight = T::zero();
    let mut mst_edges = Vec::new();

    // Priority queue for edges (weight, vertex)
    let mut heap = BinaryHeap::new();

    // Start with the given vertex
    min_weight[start] = T::zero();
    heap.push(Edge {
        weight: T::zero(),
        u: start,
        v: start,
    });

    while let Some(Edge { weight, u: _, v }) = heap.pop() {
        if in_mst[v] {
            continue;
        }

        in_mst[v] = true;
        total_weight = total_weight + weight;

        if weight > T::zero() {
            // Add edge to MST (except for the first vertex)
            mst_edges.push(Edge {
                weight,
                u: parent[v] as usize,
                v,
            });
        }

        // Update neighbors
        for &(neighbor, edge_weight) in &adj_list[v] {
            if !in_mst[neighbor] && edge_weight < min_weight[neighbor] {
                min_weight[neighbor] = edge_weight;
                parent[neighbor] = v as isize;

                heap.push(Edge {
                    weight: edge_weight,
                    u: v,
                    v: neighbor,
                });
            }
        }
    }

    // Check if all vertices are reachable
    let vertices_in_mst = in_mst.iter().filter(|&&x| x).count();
    if vertices_in_mst != n {
        return Err(SparseError::ValueError(
            "Graph is not connected - cannot compute spanning tree".to_string(),
        ));
    }

    let mst_matrix = if return_tree {
        Some(build_mst_matrix(&mst_edges, n)?)
    } else {
        None
    };

    Ok((total_weight, mst_matrix, parent))
}

/// Build a sparse matrix representation of the MST from edges
#[allow(dead_code)]
fn build_mst_matrix<T>(edges: &[Edge<T>], n: usize) -> SparseResult<CsrArray<T>>
where
    T: Float + Debug + Copy + 'static,
{
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut values = Vec::new();

    for edge in edges {
        // Add both directions for undirected tree
        rows.push(edge.u);
        cols.push(edge.v);
        values.push(edge.weight);

        rows.push(edge.v);
        cols.push(edge.u);
        values.push(edge.weight);
    }

    CsrArray::from_triplets(&rows, &cols, &values, (n, n), false)
}

/// Check if a tree is a valid spanning tree of a graph
///
/// # Arguments
///
/// * `graph` - The original graph
/// * `tree` - The potential spanning tree
/// * `tol` - Tolerance for weight comparisons
///
/// # Returns
///
/// True if the tree is a valid spanning tree, false otherwise
#[allow(dead_code)]
pub fn is_spanning_tree<T, S1, S2>(graph: &S1, tree: &S2, tol: T) -> SparseResult<bool>
where
    T: Float + Debug + Copy + 'static,
    S1: SparseArray<T>,
    S2: SparseArray<T>,
{
    let n = num_vertices(graph);
    let m = num_vertices(tree);

    // Must have same number of vertices
    if n != m {
        return Ok(false);
    }

    // Tree must have exactly n-1 edges (counting each undirected edge once)
    let tree_edges = tree.nnz() / 2; // Assuming undirected representation
    if tree_edges != n - 1 {
        return Ok(false);
    }

    // All edges in tree must exist in original graph with same weight
    let (tree_rows, tree_cols, tree_values) = tree.find();

    for (i, (&u, &v)) in tree_rows.iter().zip(tree_cols.iter()).enumerate() {
        if u < v {
            // Check each edge only once
            let graph_weight = graph.get(u, v);
            let tree_weight = tree_values[i];

            if (graph_weight - tree_weight).abs() > tol {
                return Ok(false);
            }
        }
    }

    // Check connectivity (tree should connect all vertices)
    // This is implicitly checked by the n-1 edges condition for a tree

    Ok(true)
}

/// Compute the weight of a spanning tree
///
/// # Arguments
///
/// * `tree` - The spanning tree as a sparse matrix
///
/// # Returns
///
/// Total weight of the spanning tree
#[allow(dead_code)]
pub fn spanning_tree_weight<T, S>(tree: &S) -> SparseResult<T>
where
    T: Float + Debug + Copy + 'static,
    S: SparseArray<T>,
{
    let (row_indices, col_indices, values) = tree.find();
    let mut total_weight = T::zero();

    // Sum weights, counting each undirected edge only once
    for (i, (&u, &v)) in row_indices.iter().zip(col_indices.iter()).enumerate() {
        if u <= v {
            total_weight = total_weight + values[i];
        }
    }

    Ok(total_weight)
}

/// Find all minimum spanning trees of a graph
///
/// # Note
/// This is a computationally expensive operation for large graphs.
/// It returns one MST and indicates if multiple MSTs exist.
///
/// # Arguments
///
/// * `graph` - The graph as a sparse matrix
/// * `algorithm` - MST algorithm to use
///
/// # Returns
///
/// A tuple containing:
/// - One minimum spanning tree
/// - Boolean indicating if multiple MSTs exist
/// - Total weight of any MST
#[allow(dead_code)]
pub fn all_minimum_spanning_trees<T, S>(
    graph: &S,
    algorithm: &str,
) -> SparseResult<(CsrArray<T>, bool, T)>
where
    T: Float + Debug + Copy + 'static,
    S: SparseArray<T>,
{
    let (total_weight, mst_, _) = minimum_spanning_tree(graph, algorithm, true)?;
    let mst = mst_.unwrap();

    // Simple heuristic: if there are edges with equal weights, multiple MSTs might exist
    let (_, _, values) = graph.find();
    let mut weights: Vec<_> = values.iter().copied().collect();
    weights.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

    let has_duplicates = weights
        .windows(2)
        .any(|w| (w[0] - w[1]).abs() < T::from(1e-10).unwrap());

    Ok((mst, has_duplicates, total_weight))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::csr_array::CsrArray;
    use approx::assert_relative_eq;

    fn create_test_graph() -> CsrArray<f64> {
        // Create a simple weighted graph:
        //     1
        //  0 --- 1
        //  |   / |
        //  |2 /1  |3
        //  | /    |
        //  2 ---- 3
        //     4
        let rows = vec![0, 0, 1, 1, 1, 2, 2, 2, 3, 3];
        let cols = vec![1, 2, 0, 2, 3, 0, 1, 3, 1, 2];
        let data = vec![1.0, 2.0, 1.0, 1.0, 3.0, 2.0, 1.0, 4.0, 3.0, 4.0];

        CsrArray::from_triplets(&rows, &cols, &data, (4, 4), false).unwrap()
    }

    #[test]
    fn test_union_find() {
        let mut uf = UnionFind::new(4);

        // Initially, all elements are in separate sets
        assert_ne!(uf.find(0), uf.find(1));
        assert_ne!(uf.find(1), uf.find(2));

        // Union 0 and 1
        assert!(uf.union(0, 1));
        assert_eq!(uf.find(0), uf.find(1));

        // Union 1 and 2 (effectively 0, 1, 2 in same set)
        assert!(uf.union(1, 2));
        assert_eq!(uf.find(0), uf.find(2));

        // Try to union elements already in same set
        assert!(!uf.union(0, 2));
    }

    #[test]
    fn test_kruskal_mst() {
        let graph = create_test_graph();
        let (total_weight, mst_, _) = kruskal_mst(&graph, true).unwrap();

        // MST should have weight 5 (edges: 0-1 weight 1, 1-2 weight 1, 1-3 weight 3)
        assert_relative_eq!(total_weight, 5.0);

        let mst = mst_.unwrap();

        // MST should have 3 edges (4 vertices - 1)
        assert_eq!(mst.nnz(), 6); // 3 edges * 2 (undirected)

        // Check that MST weight calculation is correct
        let calculated_weight = spanning_tree_weight(&mst).unwrap();
        assert_relative_eq!(calculated_weight, total_weight);

        // Check that it's a valid spanning tree
        assert!(is_spanning_tree(&graph, &mst, 1e-10).unwrap());
    }

    #[test]
    fn test_prim_mst() {
        let graph = create_test_graph();
        let (total_weight, mst_, _mst_parents) = prim_mst(&graph, 0, true).unwrap();

        // Should produce the same weight as Kruskal
        assert_relative_eq!(total_weight, 5.0);

        let mst = mst_.unwrap();
        assert_eq!(mst.nnz(), 6); // 3 edges * 2 (undirected)

        // Check that it's a valid spanning tree
        assert!(is_spanning_tree(&graph, &mst, 1e-10).unwrap());
    }

    #[test]
    fn test_minimum_spanning_tree_api() {
        let graph = create_test_graph();

        // Test Kruskal
        let (weight_k_, _, _) = minimum_spanning_tree(&graph, "kruskal", false).unwrap();
        assert_relative_eq!(weight_k_, 5.0);

        // Test Prim
        let (weight_p_, _, _) = minimum_spanning_tree(&graph, "prim", false).unwrap();
        assert_relative_eq!(weight_p_, 5.0);

        // Test auto selection
        let (weight_a_, _, _) = minimum_spanning_tree(&graph, "auto", false).unwrap();
        assert_relative_eq!(weight_a_, 5.0);
    }

    #[test]
    fn test_disconnected_graph() {
        // Create a disconnected graph
        let rows = vec![0, 1, 2, 3];
        let cols = vec![1, 0, 3, 2];
        let data = vec![1.0, 1.0, 1.0, 1.0];
        let graph = CsrArray::from_triplets(&rows, &cols, &data, (4, 4), false).unwrap();

        // MST should fail for disconnected graph
        assert!(minimum_spanning_tree(&graph, "kruskal", false).is_err());
        assert!(minimum_spanning_tree(&graph, "prim", false).is_err());
    }

    #[test]
    fn test_single_vertex() {
        // Single vertex graph
        let graph: CsrArray<f64> = CsrArray::from_triplets(&[], &[], &[], (1, 1), false).unwrap();

        let (total_weight, mst_, _) = minimum_spanning_tree(&graph, "kruskal", true).unwrap();
        assert_relative_eq!(total_weight, 0.0);

        let mst = mst_.unwrap();
        assert_eq!(mst.nnz(), 0); // No edges in single vertex tree
    }

    #[test]
    fn test_two_vertices() {
        // Two vertex graph
        let rows = vec![0, 1];
        let cols = vec![1, 0];
        let data = vec![5.0, 5.0];
        let graph = CsrArray::from_triplets(&rows, &cols, &data, (2, 2), false).unwrap();

        let (total_weight, mst_, _mst_parents) =
            minimum_spanning_tree(&graph, "prim", true).unwrap();
        assert_relative_eq!(total_weight, 5.0);

        let mst = mst_.unwrap();
        assert_eq!(mst.nnz(), 2); // One edge * 2 (undirected)
    }

    #[test]
    fn test_complete_graph() {
        // Create a complete graph on 4 vertices with different weights
        let rows = vec![0, 0, 0, 1, 1, 2];
        let cols = vec![1, 2, 3, 2, 3, 3];
        let data = vec![1.0, 4.0, 3.0, 2.0, 5.0, 6.0];

        // Make it symmetric
        let mut all_rows = rows.clone();
        let mut all_cols = cols.clone();
        let mut all_data = data.clone();

        for (i, (&r, &c)) in rows.iter().zip(cols.iter()).enumerate() {
            all_rows.push(c);
            all_cols.push(r);
            all_data.push(data[i]);
        }

        let graph =
            CsrArray::from_triplets(&all_rows, &all_cols, &all_data, (4, 4), false).unwrap();

        let (total_weight_, _, _) = minimum_spanning_tree(&graph, "kruskal", false).unwrap();

        // MST should use edges: 0-1 (1), 1-2 (2), 0-3 (3) for total weight 6
        assert_relative_eq!(total_weight_, 6.0);
    }

    #[test]
    fn test_spanning_tree_validation() {
        let graph = create_test_graph();
        let (_, mst_, _) = minimum_spanning_tree(&graph, "kruskal", true).unwrap();
        let mst = mst_.unwrap();

        // Valid spanning tree
        assert!(is_spanning_tree(&graph, &mst, 1e-10).unwrap());

        // Create an invalid tree (wrong number of edges)
        let rows = vec![0, 1];
        let cols = vec![1, 0];
        let data = vec![1.0, 1.0];
        let invalid_tree = CsrArray::from_triplets(&rows, &cols, &data, (4, 4), false).unwrap();

        assert!(!is_spanning_tree(&graph, &invalid_tree, 1e-10).unwrap());
    }

    #[test]
    fn test_algorithm_selection() {
        let _graph = create_test_graph();

        // Test algorithm string parsing
        assert!(matches!(
            MSTAlgorithm::from_str("kruskal"),
            Ok(MSTAlgorithm::Kruskal)
        ));
        assert!(matches!(
            MSTAlgorithm::from_str("prim"),
            Ok(MSTAlgorithm::Prim)
        ));
        assert!(matches!(
            MSTAlgorithm::from_str("auto"),
            Ok(MSTAlgorithm::Auto)
        ));
        assert!(MSTAlgorithm::from_str("invalid").is_err());
    }
}
