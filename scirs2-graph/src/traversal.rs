//! Enhanced graph traversal algorithms on adjacency matrices.
//!
//! This module complements the typed-graph traversal in
//! `algorithms::traversal` by providing matrix-based versions that operate
//! directly on `Array2<f64>` weighted adjacency matrices.
//!
//! ## Algorithms
//! - **Topological sort** (Kahn's algorithm for DAGs)
//! - **All simple paths** (DFS-based, bounded by `max_length`)
//! - **Eulerian circuit / path** (Hierholzer's algorithm)
//! - **Hamiltonian path heuristic** (nearest-neighbour greedy)
//! - **Bipartite check and 2-colouring** (BFS-based)
//! - **k-core decomposition** (iterative degree peeling)
//!
//! ## Convention
//! - For **undirected** algorithms (Euler, bipartite, k-core, simple paths):
//!   the matrix is treated as symmetric and only the upper-triangle is
//!   meaningful if both `adj[i,j]` and `adj[j,i]` are non-zero.
//! - For **directed** algorithms (topological sort): only `adj[i,j] > 0`
//!   means an edge from `i → j`.
//!
//! ## Example
//! ```rust,no_run
//! use scirs2_core::ndarray::Array2;
//! use scirs2_graph::traversal::{topological_sort, is_bipartite, k_core_decomposition};
//!
//! // DAG: 0 → 1 → 2, 0 → 2
//! let adj = Array2::<f64>::from_shape_vec((3,3), vec![
//!     0.,1.,1., 0.,0.,1., 0.,0.,0.,
//! ]).unwrap();
//! let order = topological_sort(&adj).unwrap();
//! assert_eq!(order[0], 0);
//! ```

use std::collections::VecDeque;

use scirs2_core::ndarray::Array2;

use crate::error::{GraphError, Result};

// ─────────────────────────────────────────────────────────────────────────────
// Topological sort (Kahn's algorithm)
// ─────────────────────────────────────────────────────────────────────────────

/// Topological sort of a directed acyclic graph (Kahn's algorithm).
///
/// Treats `adj[i,j] > 0` as a directed edge `i → j`.
/// Returns nodes in topological order (sources first).
///
/// # Errors
/// Returns `GraphError::CycleDetected` if the graph has a cycle.
pub fn topological_sort(adj: &Array2<f64>) -> Result<Vec<usize>> {
    let n = adj.nrows();
    if n == 0 {
        return Ok(vec![]);
    }
    if adj.ncols() != n {
        return Err(GraphError::InvalidGraph(
            "adjacency matrix must be square".into(),
        ));
    }

    // Compute in-degrees
    let mut in_degree = vec![0usize; n];
    for j in 0..n {
        for i in 0..n {
            if adj[[i, j]] > 0.0 {
                in_degree[j] += 1;
            }
        }
    }

    // Enqueue all nodes with in-degree 0
    let mut queue: VecDeque<usize> = in_degree
        .iter()
        .enumerate()
        .filter(|(_, &d)| d == 0)
        .map(|(i, _)| i)
        .collect();

    let mut order = Vec::with_capacity(n);

    while let Some(u) = queue.pop_front() {
        order.push(u);
        for v in 0..n {
            if adj[[u, v]] > 0.0 {
                in_degree[v] -= 1;
                if in_degree[v] == 0 {
                    queue.push_back(v);
                }
            }
        }
    }

    if order.len() != n {
        return Err(GraphError::CycleDetected {
            start_node: "0".into(),
            cycle_length: n - order.len(),
        });
    }

    Ok(order)
}

// ─────────────────────────────────────────────────────────────────────────────
// All simple paths
// ─────────────────────────────────────────────────────────────────────────────

/// Enumerate all simple paths from `source` to `target` in an undirected graph,
/// with path length (number of edges) at most `max_length`.
///
/// A "simple" path visits each node at most once.  For `max_length = 0` or when
/// `source == target`, returns `[[source]]` (trivial path) if applicable.
///
/// **Warning**: exponential worst-case complexity — use only for small graphs or
/// small `max_length`.
///
/// Returns a (possibly empty) list of node sequences.
pub fn all_simple_paths(
    adj: &Array2<f64>,
    source: usize,
    target: usize,
    max_length: usize,
) -> Vec<Vec<usize>> {
    let n = adj.nrows();
    if n == 0 || source >= n || target >= n {
        return vec![];
    }
    if source == target {
        return vec![vec![source]];
    }

    let mut results = Vec::new();
    let mut path = vec![source];
    let mut visited = vec![false; n];
    visited[source] = true;

    dfs_paths(
        adj,
        source,
        target,
        max_length,
        &mut path,
        &mut visited,
        &mut results,
    );

    results
}

/// DFS helper for `all_simple_paths`.
fn dfs_paths(
    adj: &Array2<f64>,
    current: usize,
    target: usize,
    max_length: usize,
    path: &mut Vec<usize>,
    visited: &mut Vec<bool>,
    results: &mut Vec<Vec<usize>>,
) {
    // Path length is number of edges = path.len() - 1
    if path.len() > max_length {
        return;
    }

    let n = adj.nrows();
    for next in 0..n {
        if adj[[current, next]] <= 0.0 || visited[next] {
            continue;
        }
        path.push(next);
        if next == target {
            results.push(path.clone());
        } else {
            visited[next] = true;
            dfs_paths(adj, next, target, max_length, path, visited, results);
            visited[next] = false;
        }
        path.pop();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Eulerian circuit / path (Hierholzer's algorithm)
// ─────────────────────────────────────────────────────────────────────────────

/// Find an Eulerian **circuit** in an undirected graph (Hierholzer's algorithm).
///
/// An Eulerian circuit exists iff every node has even degree and the graph is
/// connected (ignoring isolated nodes).
///
/// Returns the circuit as a sequence of node indices (first == last).
///
/// # Errors
/// * `GraphError::GraphStructureError` – if the graph has no Eulerian circuit.
pub fn eulerian_circuit(adj: &Array2<f64>) -> Result<Vec<usize>> {
    let n = adj.nrows();
    if n == 0 {
        return Err(GraphError::InvalidGraph("empty adjacency matrix".into()));
    }

    // Check all degrees are even
    for i in 0..n {
        let deg = degree_of(adj, i);
        if !deg.is_multiple_of(2) {
            return Err(GraphError::GraphStructureError {
                expected: "all even degrees".into(),
                found: format!("node {i} has degree {deg}"),
                context: "eulerian_circuit".into(),
            });
        }
    }

    // Find a starting node with non-zero degree
    let start = (0..n).find(|&i| degree_of(adj, i) > 0).unwrap_or(0);

    // Check connectivity (among nodes with non-zero degree)
    if !is_eulerian_connected(adj) {
        return Err(GraphError::GraphStructureError {
            expected: "connected graph".into(),
            found: "graph is disconnected".into(),
            context: "eulerian_circuit".into(),
        });
    }

    hierholzer(adj, start)
}

/// Find an Eulerian **path** starting from `source` in an undirected graph.
///
/// An Eulerian path exists iff there are exactly 0 or 2 nodes of odd degree.
/// If 2 odd-degree nodes exist, one must be the source and the other the terminus.
///
/// # Errors
/// * `GraphError::GraphStructureError` – if no Eulerian path exists from `source`.
pub fn eulerian_path(adj: &Array2<f64>, source: usize) -> Result<Vec<usize>> {
    let n = adj.nrows();
    if n == 0 {
        return Err(GraphError::InvalidGraph("empty adjacency matrix".into()));
    }
    if source >= n {
        return Err(GraphError::InvalidParameter {
            param: "source".into(),
            value: source.to_string(),
            expected: format!("< {n}"),
            context: "eulerian_path".into(),
        });
    }

    let odd_nodes: Vec<usize> = (0..n)
        .filter(|&i| !degree_of(adj, i).is_multiple_of(2))
        .collect();

    match odd_nodes.len() {
        0 => {
            // Eulerian circuit — path is also valid
            eulerian_circuit(adj)
        }
        2 => {
            if !odd_nodes.contains(&source) {
                return Err(GraphError::GraphStructureError {
                    expected: format!("source {source} must be an odd-degree node"),
                    found: format!("odd nodes are {:?}", odd_nodes),
                    context: "eulerian_path".into(),
                });
            }
            if !is_eulerian_connected(adj) {
                return Err(GraphError::GraphStructureError {
                    expected: "connected graph".into(),
                    found: "graph is disconnected".into(),
                    context: "eulerian_path".into(),
                });
            }
            hierholzer(adj, source)
        }
        _ => Err(GraphError::GraphStructureError {
            expected: "0 or 2 odd-degree nodes".into(),
            found: format!("{} odd-degree nodes", odd_nodes.len()),
            context: "eulerian_path".into(),
        }),
    }
}

/// Undirected degree of node `i` (counts multi-edges via weight > 0 threshold).
fn degree_of(adj: &Array2<f64>, i: usize) -> usize {
    // Count discrete edges: each positive entry counts once.
    // For an undirected graph stored symmetrically, adj[i,j] > 0 ↔ edge present.
    (0..adj.ncols()).filter(|&j| adj[[i, j]] > 0.0).count()
}

/// Check that all non-isolated nodes form a single connected component.
fn is_eulerian_connected(adj: &Array2<f64>) -> bool {
    let n = adj.nrows();
    let non_isolated: Vec<usize> = (0..n).filter(|&i| degree_of(adj, i) > 0).collect();
    if non_isolated.is_empty() {
        return true;
    }
    let start = non_isolated[0];
    let mut visited = vec![false; n];
    let mut stack = vec![start];
    visited[start] = true;
    while let Some(u) = stack.pop() {
        for v in 0..n {
            if adj[[u, v]] > 0.0 && !visited[v] {
                visited[v] = true;
                stack.push(v);
            }
        }
    }
    non_isolated.iter().all(|&i| visited[i])
}

/// Hierholzer's algorithm on a multigraph represented by a mutable adjacency
/// matrix (edge counts / weights are decremented as edges are used).
fn hierholzer(adj: &Array2<f64>, start: usize) -> Result<Vec<usize>> {
    let n = adj.nrows();
    // Work on integer edge counts (treat weight > 0 as one edge)
    let mut edge_count = vec![vec![0i64; n]; n];
    let mut total_edges = 0i64;
    for i in 0..n {
        for j in 0..n {
            if adj[[i, j]] > 0.0 {
                edge_count[i][j] = 1;
                total_edges += 1;
            }
        }
    }
    // For undirected: each edge is counted twice in total_edges
    total_edges /= 2;

    let mut circuit = Vec::new();
    let mut stack = vec![start];

    while let Some(&u) = stack.last() {
        // Find an unused edge from u
        let next = (0..n).find(|&v| edge_count[u][v] > 0);
        match next {
            Some(v) => {
                stack.push(v);
                edge_count[u][v] -= 1;
                edge_count[v][u] -= 1;
            }
            None => {
                circuit.push(stack.pop().expect("stack not empty"));
            }
        }
    }

    // Verify we used all edges
    if circuit.len() as i64 != total_edges + 1 {
        return Err(GraphError::AlgorithmError(format!(
            "Hierholzer: expected {} nodes in circuit, got {}",
            total_edges + 1,
            circuit.len()
        )));
    }

    circuit.reverse();
    Ok(circuit)
}

// ─────────────────────────────────────────────────────────────────────────────
// Hamiltonian path heuristic (nearest-neighbour)
// ─────────────────────────────────────────────────────────────────────────────

/// Greedy nearest-neighbour heuristic for the Hamiltonian path problem.
///
/// Starting from `start`, always moves to the closest unvisited neighbour
/// (by edge weight — zero weight is treated as "no edge").
///
/// This is **not** guaranteed to find a Hamiltonian path if one exists, but
/// runs in O(n²) time and provides a good initial solution for TSP-style problems.
///
/// # Errors
/// Returns an error if the graph is empty or `start` is out of range.
pub fn hamiltonian_path_heuristic(adj: &Array2<f64>, start: usize) -> Result<Vec<usize>> {
    let n = adj.nrows();
    if n == 0 {
        return Err(GraphError::InvalidGraph("empty adjacency matrix".into()));
    }
    if start >= n {
        return Err(GraphError::InvalidParameter {
            param: "start".into(),
            value: start.to_string(),
            expected: format!("< {n}"),
            context: "hamiltonian_path_heuristic".into(),
        });
    }

    let mut visited = vec![false; n];
    let mut path = vec![start];
    visited[start] = true;

    for _ in 1..n {
        let current = *path.last().expect("path not empty");
        // Find nearest unvisited neighbour
        let mut best_next = None;
        let mut best_w = f64::NEG_INFINITY;
        for v in 0..n {
            if !visited[v] && adj[[current, v]] > 0.0 && adj[[current, v]] > best_w {
                best_w = adj[[current, v]];
                best_next = Some(v);
            }
        }
        match best_next {
            Some(v) => {
                visited[v] = true;
                path.push(v);
            }
            None => break, // no unvisited neighbour — partial path
        }
    }

    Ok(path)
}

// ─────────────────────────────────────────────────────────────────────────────
// Bipartite check and 2-colouring
// ─────────────────────────────────────────────────────────────────────────────

/// Check whether the graph is bipartite and return a 2-colouring if so.
///
/// Uses BFS (breadth-first 2-colouring) on the undirected graph.
/// Self-loops immediately make the graph non-bipartite.
///
/// # Returns
/// `(true, Some(colours))` where `colours[i] ∈ {0, 1}`, or `(false, None)`.
pub fn is_bipartite(adj: &Array2<f64>) -> (bool, Option<Vec<usize>>) {
    let n = adj.nrows();
    if n == 0 {
        return (true, Some(vec![]));
    }

    // Self-loop check
    for i in 0..n {
        if adj[[i, i]] > 0.0 {
            return (false, None);
        }
    }

    let mut colour = vec![usize::MAX; n];

    for start in 0..n {
        if colour[start] != usize::MAX {
            continue;
        }
        colour[start] = 0;
        let mut queue = VecDeque::new();
        queue.push_back(start);

        while let Some(u) = queue.pop_front() {
            let c = colour[u];
            for v in 0..n {
                if adj[[u, v]] <= 0.0 {
                    continue;
                }
                if colour[v] == usize::MAX {
                    colour[v] = 1 - c;
                    queue.push_back(v);
                } else if colour[v] == c {
                    return (false, None);
                }
            }
        }
    }

    (true, Some(colour))
}

// ─────────────────────────────────────────────────────────────────────────────
// k-core decomposition
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the **core number** (degeneracy shell) of every node.
///
/// The k-core of a graph is the maximal subgraph in which every node has degree
/// ≥ k within that subgraph.  The core number of a node is the largest k such
/// that the node belongs to the k-core.
///
/// Implements the efficient O(n + m) peeling algorithm.
/// Returns a `Vec<usize>` of length n.
pub fn k_core_decomposition(adj: &Array2<f64>) -> Vec<usize> {
    let n = adj.nrows();
    if n == 0 {
        return vec![];
    }

    // Current degrees (undirected, counting non-zero entries)
    let mut deg: Vec<usize> = (0..n).map(|i| degree_of(adj, i)).collect();
    let mut removed = vec![false; n];
    let mut core = vec![0usize; n];

    let max_deg = *deg.iter().max().unwrap_or(&0);

    // Bucket sort by degree for efficient peeling
    for k in 0..=max_deg {
        // Peel all nodes with current degree <= k
        let mut changed = true;
        while changed {
            changed = false;
            for u in 0..n {
                if removed[u] || deg[u] > k {
                    continue;
                }
                removed[u] = true;
                core[u] = k;
                // Reduce neighbours' effective degrees
                for v in 0..n {
                    if !removed[v] && adj[[u, v]] > 0.0 {
                        deg[v] = deg[v].saturating_sub(1);
                        changed = true;
                    }
                }
            }
        }
    }

    core
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;
    use std::collections::HashSet;

    // ── topological_sort ────────────────────────────────────────────────────

    #[test]
    fn test_topo_sort_dag() {
        // 0 → 1 → 3
        // 0 → 2 → 3
        let mut adj = Array2::<f64>::zeros((4, 4));
        adj[[0, 1]] = 1.0;
        adj[[0, 2]] = 1.0;
        adj[[1, 3]] = 1.0;
        adj[[2, 3]] = 1.0;
        let order = topological_sort(&adj).expect("topo sort");
        assert_eq!(order.len(), 4);
        // 0 must come before 1, 2, and 3
        let pos: Vec<usize> = {
            let mut p = vec![0usize; 4];
            for (i, &v) in order.iter().enumerate() {
                p[v] = i;
            }
            p
        };
        assert!(pos[0] < pos[1], "0 before 1");
        assert!(pos[0] < pos[2], "0 before 2");
        assert!(pos[1] < pos[3], "1 before 3");
        assert!(pos[2] < pos[3], "2 before 3");
    }

    #[test]
    fn test_topo_sort_cycle_error() {
        let mut adj = Array2::<f64>::zeros((3, 3));
        adj[[0, 1]] = 1.0;
        adj[[1, 2]] = 1.0;
        adj[[2, 0]] = 1.0;
        assert!(topological_sort(&adj).is_err());
    }

    #[test]
    fn test_topo_sort_empty() {
        let adj = Array2::<f64>::zeros((0, 0));
        assert_eq!(topological_sort(&adj).expect("empty"), vec![] as Vec<usize>);
    }

    #[test]
    fn test_topo_sort_linear_chain() {
        // 0 → 1 → 2 → 3 → 4
        let mut adj = Array2::<f64>::zeros((5, 5));
        for i in 0..4 {
            adj[[i, i + 1]] = 1.0;
        }
        let order = topological_sort(&adj).expect("chain");
        assert_eq!(order, vec![0, 1, 2, 3, 4]);
    }

    // ── all_simple_paths ────────────────────────────────────────────────────

    #[test]
    fn test_all_simple_paths_triangle() {
        // Complete triangle: 0-1, 1-2, 0-2
        let mut adj = Array2::<f64>::zeros((3, 3));
        adj[[0, 1]] = 1.0;
        adj[[1, 0]] = 1.0;
        adj[[1, 2]] = 1.0;
        adj[[2, 1]] = 1.0;
        adj[[0, 2]] = 1.0;
        adj[[2, 0]] = 1.0;

        let paths = all_simple_paths(&adj, 0, 2, 3);
        // Should find at least 2 paths: 0→2 and 0→1→2
        assert!(
            paths.len() >= 2,
            "should find at least 2 paths, got {}",
            paths.len()
        );
        // Direct path 0→2 must be present
        assert!(
            paths.iter().any(|p| p == &[0, 2]),
            "direct path 0→2 not found"
        );
        // Indirect 0→1→2 must be present
        assert!(
            paths.iter().any(|p| p == &[0, 1, 2]),
            "path 0→1→2 not found"
        );
    }

    #[test]
    fn test_all_simple_paths_max_length() {
        let mut adj = Array2::<f64>::zeros((4, 4));
        for i in 0..3 {
            adj[[i, i + 1]] = 1.0;
            adj[[i + 1, i]] = 1.0;
        }
        // With max_length=1 only the direct edge (if any) counts
        let paths = all_simple_paths(&adj, 0, 3, 1);
        // No direct edge 0-3 → should be empty
        assert!(paths.is_empty());

        let paths2 = all_simple_paths(&adj, 0, 3, 3);
        // Only path: 0→1→2→3
        assert_eq!(paths2.len(), 1);
        assert_eq!(paths2[0], vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_all_simple_paths_self_loop() {
        let adj = Array2::<f64>::zeros((4, 4));
        let paths = all_simple_paths(&adj, 2, 2, 5);
        // source == target: return trivial path
        assert_eq!(paths, vec![vec![2]]);
    }

    // ── eulerian_circuit ────────────────────────────────────────────────────

    #[test]
    fn test_eulerian_circuit_cycle4() {
        // Simple cycle 0-1-2-3-0
        let mut adj = Array2::<f64>::zeros((4, 4));
        adj[[0, 1]] = 1.0;
        adj[[1, 0]] = 1.0;
        adj[[1, 2]] = 1.0;
        adj[[2, 1]] = 1.0;
        adj[[2, 3]] = 1.0;
        adj[[3, 2]] = 1.0;
        adj[[3, 0]] = 1.0;
        adj[[0, 3]] = 1.0;

        let circuit = eulerian_circuit(&adj).expect("euler circuit");
        // Circuit must start and end at same node and have 5 elements (4 edges + 1)
        assert_eq!(circuit.len(), 5, "circuit length");
        assert_eq!(circuit.first(), circuit.last(), "start == end");
        // Must visit all 4 nodes
        let unique: HashSet<usize> = circuit.iter().cloned().collect();
        assert_eq!(unique.len(), 4);
    }

    #[test]
    fn test_eulerian_circuit_odd_degree_error() {
        // Path 0-1-2 has odd-degree nodes (0 and 2 have degree 1)
        let mut adj = Array2::<f64>::zeros((3, 3));
        adj[[0, 1]] = 1.0;
        adj[[1, 0]] = 1.0;
        adj[[1, 2]] = 1.0;
        adj[[2, 1]] = 1.0;
        assert!(eulerian_circuit(&adj).is_err());
    }

    // ── eulerian_path ───────────────────────────────────────────────────────

    #[test]
    fn test_eulerian_path_line() {
        // Path 0-1-2: Eulerian path from 0 to 2
        let mut adj = Array2::<f64>::zeros((3, 3));
        adj[[0, 1]] = 1.0;
        adj[[1, 0]] = 1.0;
        adj[[1, 2]] = 1.0;
        adj[[2, 1]] = 1.0;

        let path = eulerian_path(&adj, 0).expect("euler path");
        assert_eq!(path.len(), 3, "path should visit 3 nodes");
        assert_eq!(path[0], 0);
        assert_eq!(*path.last().expect("path last"), 2);
    }

    #[test]
    fn test_eulerian_path_wrong_start_error() {
        let mut adj = Array2::<f64>::zeros((3, 3));
        adj[[0, 1]] = 1.0;
        adj[[1, 0]] = 1.0;
        adj[[1, 2]] = 1.0;
        adj[[2, 1]] = 1.0;
        // Starting from node 1 (even degree) is invalid when odd nodes exist
        assert!(eulerian_path(&adj, 1).is_err());
    }

    // ── hamiltonian_path_heuristic ───────────────────────────────────────────

    #[test]
    fn test_hamiltonian_heuristic_complete() {
        // Complete graph K4: heuristic should find a full path of length 4
        let mut adj = Array2::<f64>::ones((4, 4));
        for i in 0..4 {
            adj[[i, i]] = 0.0;
        }
        let path = hamiltonian_path_heuristic(&adj, 0).expect("ham");
        assert_eq!(path.len(), 4);
        // All nodes distinct
        let unique: HashSet<usize> = path.iter().cloned().collect();
        assert_eq!(unique.len(), 4);
    }

    #[test]
    fn test_hamiltonian_heuristic_path_graph() {
        // Path 0-1-2-3-4 should be found starting at 0
        let mut adj = Array2::<f64>::zeros((5, 5));
        for i in 0..4 {
            adj[[i, i + 1]] = 1.0;
            adj[[i + 1, i]] = 1.0;
        }
        let path = hamiltonian_path_heuristic(&adj, 0).expect("ham");
        assert_eq!(path, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_hamiltonian_heuristic_error() {
        let adj = Array2::<f64>::zeros((0, 0));
        assert!(hamiltonian_path_heuristic(&adj, 0).is_err());
    }

    // ── is_bipartite ─────────────────────────────────────────────────────────

    #[test]
    fn test_bipartite_even_cycle() {
        // C4 (4-cycle) is bipartite
        let mut adj = Array2::<f64>::zeros((4, 4));
        adj[[0, 1]] = 1.0;
        adj[[1, 0]] = 1.0;
        adj[[1, 2]] = 1.0;
        adj[[2, 1]] = 1.0;
        adj[[2, 3]] = 1.0;
        adj[[3, 2]] = 1.0;
        adj[[3, 0]] = 1.0;
        adj[[0, 3]] = 1.0;

        let (bip, colours) = is_bipartite(&adj);
        assert!(bip, "C4 should be bipartite");
        let colours = colours.expect("should have colouring");
        // Check proper 2-colouring
        for i in 0..4 {
            for j in 0..4 {
                if adj[[i, j]] > 0.0 {
                    assert_ne!(colours[i], colours[j], "adjacent nodes same colour");
                }
            }
        }
    }

    #[test]
    fn test_bipartite_complete_bipartite() {
        // K_{3,3} is bipartite
        let mut adj = Array2::<f64>::zeros((6, 6));
        for i in 0..3 {
            for j in 3..6 {
                adj[[i, j]] = 1.0;
                adj[[j, i]] = 1.0;
            }
        }
        let (bip, _) = is_bipartite(&adj);
        assert!(bip, "K_{{3,3}} should be bipartite");
    }

    #[test]
    fn test_bipartite_triangle_not() {
        // Triangle K3 is NOT bipartite (odd cycle)
        let mut adj = Array2::<f64>::zeros((3, 3));
        adj[[0, 1]] = 1.0;
        adj[[1, 0]] = 1.0;
        adj[[1, 2]] = 1.0;
        adj[[2, 1]] = 1.0;
        adj[[0, 2]] = 1.0;
        adj[[2, 0]] = 1.0;

        let (bip, _) = is_bipartite(&adj);
        assert!(!bip, "K3 (triangle) should NOT be bipartite");
    }

    #[test]
    fn test_bipartite_self_loop_not() {
        let mut adj = Array2::<f64>::zeros((3, 3));
        adj[[1, 1]] = 1.0; // self-loop
        let (bip, _) = is_bipartite(&adj);
        assert!(!bip, "self-loop graph should NOT be bipartite");
    }

    // ── k_core_decomposition ─────────────────────────────────────────────────

    #[test]
    fn test_k_core_path_graph() {
        // Path graph P5: every node has degree <= 2, so max core = 1
        let mut adj = Array2::<f64>::zeros((5, 5));
        for i in 0..4 {
            adj[[i, i + 1]] = 1.0;
            adj[[i + 1, i]] = 1.0;
        }
        let cores = k_core_decomposition(&adj);
        assert_eq!(cores.len(), 5);
        // All cores should be 1 (path graph is 1-core)
        for (i, &c) in cores.iter().enumerate() {
            assert_eq!(c, 1, "path node {i} should have core 1, got {c}");
        }
    }

    #[test]
    fn test_k_core_triangle_clique() {
        // Triangle: all nodes have degree 2 within themselves → 2-core
        let mut adj = Array2::<f64>::zeros((3, 3));
        adj[[0, 1]] = 1.0;
        adj[[1, 0]] = 1.0;
        adj[[1, 2]] = 1.0;
        adj[[2, 1]] = 1.0;
        adj[[0, 2]] = 1.0;
        adj[[2, 0]] = 1.0;

        let cores = k_core_decomposition(&adj);
        for &c in &cores {
            assert_eq!(c, 2, "triangle node should have core 2");
        }
    }

    #[test]
    fn test_k_core_mixed_graph() {
        // K3 clique (0,1,2) + pendant node 3 attached to 0
        // Nodes 0,1,2 → 2-core; node 3 → 1-core
        let mut adj = Array2::<f64>::zeros((4, 4));
        adj[[0, 1]] = 1.0;
        adj[[1, 0]] = 1.0;
        adj[[1, 2]] = 1.0;
        adj[[2, 1]] = 1.0;
        adj[[0, 2]] = 1.0;
        adj[[2, 0]] = 1.0;
        adj[[0, 3]] = 1.0;
        adj[[3, 0]] = 1.0;

        let cores = k_core_decomposition(&adj);
        assert_eq!(cores[0], 2, "node 0 in K3 should be 2-core");
        assert_eq!(cores[1], 2, "node 1 in K3 should be 2-core");
        assert_eq!(cores[2], 2, "node 2 in K3 should be 2-core");
        assert_eq!(cores[3], 1, "pendant node 3 should be 1-core");
    }

    #[test]
    fn test_k_core_empty() {
        let adj = Array2::<f64>::zeros((0, 0));
        assert_eq!(k_core_decomposition(&adj), vec![] as Vec<usize>);
    }

    #[test]
    fn test_k_core_isolated_nodes() {
        // 3 isolated nodes → all core 0
        let adj = Array2::<f64>::zeros((3, 3));
        let cores = k_core_decomposition(&adj);
        for &c in &cores {
            assert_eq!(c, 0);
        }
    }
}
