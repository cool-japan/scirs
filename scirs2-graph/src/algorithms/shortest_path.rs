//! Shortest path algorithms
//!
//! This module provides various shortest path algorithms including:
//! - Dijkstra's algorithm
//! - A* search
//! - Floyd-Warshall all-pairs shortest paths
//! - K-shortest paths (Yen's algorithm)

use petgraph::algo::dijkstra;
use petgraph::visit::EdgeRef;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::hash::Hash;

use crate::base::{DiGraph, EdgeWeight, Graph, Node};
use crate::error::{GraphError, Result};

/// Path between two nodes in a graph
#[derive(Debug, Clone)]
pub struct Path<N: Node + std::fmt::Debug, E: EdgeWeight> {
    /// The nodes in the path, in order
    pub nodes: Vec<N>,
    /// The total weight of the path
    pub total_weight: E,
}

/// Result of A* search algorithm
#[derive(Debug, Clone)]
pub struct AStarResult<N: Node + std::fmt::Debug, E: EdgeWeight> {
    /// The path from source to target
    pub path: Vec<N>,
    /// The total cost of the path
    pub cost: E,
}

/// State for A* priority queue
#[derive(Clone)]
struct AStarState<N: Node + std::fmt::Debug, E: EdgeWeight> {
    node: N,
    cost: E,
    heuristic: E,
    path: Vec<N>,
}

impl<N: Node + std::fmt::Debug, E: EdgeWeight> PartialEq for AStarState<N, E> {
    fn eq(&self, other: &Self) -> bool {
        self.node == other.node
    }
}

impl<N: Node + std::fmt::Debug, E: EdgeWeight> Eq for AStarState<N, E> {}

impl<N: Node + std::fmt::Debug, E: EdgeWeight + std::ops::Add<Output = E> + Copy + PartialOrd> Ord
    for AStarState<N, E>
{
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse order for min-heap behavior
        let self_total = self.cost + self.heuristic;
        let other_total = other.cost + other.heuristic;
        other_total
            .partial_cmp(&self_total)
            .unwrap_or(Ordering::Equal)
            .then_with(|| {
                other
                    .cost
                    .partial_cmp(&self.cost)
                    .unwrap_or(Ordering::Equal)
            })
    }
}

impl<N: Node + std::fmt::Debug, E: EdgeWeight + std::ops::Add<Output = E> + Copy + PartialOrd>
    PartialOrd for AStarState<N, E>
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Finds the shortest path between source and target nodes using Dijkstra's algorithm
///
/// # Arguments
/// * `graph` - The graph to search in
/// * `source` - The source node
/// * `target` - The target node
///
/// # Returns
/// * `Ok(Some(Path))` - If a path exists
/// * `Ok(None)` - If no path exists
/// * `Err(GraphError)` - If the source or target node is not in the graph
///
/// # Time Complexity
/// O((V + E) log V) where V is the number of vertices and E is the number of edges.
/// Using a binary heap (min-heap) priority queue implementation.
///
/// # Space Complexity
/// O(V) for the distance array and predecessor tracking.
///
/// # Deprecation Notice
/// This function will be updated in v1.0 to return a standardized `PathResult` type
/// that provides more detailed information about the path and search process.
/// The current API will remain available but deprecated.
#[deprecated(
    since = "0.1.0-beta.2",
    note = "Use `dijkstra_path` for future compatibility. This function will return PathResult in v1.0"
)]
#[allow(dead_code)]
pub fn shortest_path<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    source: &N,
    target: &N,
) -> Result<Option<Path<N, E>>>
where
    N: Node + std::fmt::Debug,
    E: EdgeWeight
        + scirs2_core::numeric::Zero
        + scirs2_core::numeric::One
        + std::ops::Add<Output = E>
        + PartialOrd
        + std::marker::Copy
        + std::fmt::Debug
        + std::default::Default,
    Ix: petgraph::graph::IndexType,
{
    // Check if source and target are in the graph
    if !graph.has_node(source) {
        return Err(GraphError::InvalidGraph(format!(
            "Source node {source:?} not found"
        )));
    }
    if !graph.has_node(target) {
        return Err(GraphError::InvalidGraph(format!(
            "Target node {target:?} not found"
        )));
    }

    let source_idx = graph
        .inner()
        .node_indices()
        .find(|&idx| graph.inner()[idx] == *source)
        .unwrap();
    let target_idx = graph
        .inner()
        .node_indices()
        .find(|&idx| graph.inner()[idx] == *target)
        .unwrap();

    // Use petgraph's Dijkstra algorithm implementation
    let results = dijkstra(graph.inner(), source_idx, Some(target_idx), |e| *e.weight());

    // If target is not reachable, return None
    if !results.contains_key(&target_idx) {
        return Ok(None);
    }

    let total_weight = results[&target_idx];

    // Reconstruct the path
    let mut path = Vec::new();
    let mut current = target_idx;

    path.push(graph.inner()[current].clone());

    // Backtrack to find the path
    while current != source_idx {
        let min_prev = graph
            .inner()
            .edges_directed(current, petgraph::Direction::Incoming)
            .filter_map(|e| {
                let from = e.source();
                let edge_weight = *e.weight();

                // Check if this node is part of the shortest path
                if let Some(from_dist) = results.get(&from) {
                    // If this edge is part of the shortest path
                    if *from_dist + edge_weight == results[&current] {
                        return Some((from, *from_dist));
                    }
                }
                None
            })
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

        if let Some((prev, _)) = min_prev {
            current = prev;
            path.push(graph.inner()[current].clone());
        } else {
            // This shouldn't happen if Dijkstra's algorithm works correctly
            return Err(GraphError::AlgorithmError(
                "Failed to reconstruct path".to_string(),
            ));
        }
    }

    // Reverse the path to get it from source to target
    path.reverse();

    Ok(Some(Path {
        nodes: path,
        total_weight,
    }))
}

/// Finds the shortest path between source and target nodes using Dijkstra's algorithm (modern API)
///
/// This function provides the same functionality as `shortest_path` but with a clearer name
/// that explicitly indicates the algorithm being used. This is the recommended function to use
/// going forward.
///
/// # Arguments
/// * `graph` - The graph to search in
/// * `source` - The source node
/// * `target` - The target node
///
/// # Returns
/// * `Ok(Some(Path))` - If a path exists
/// * `Ok(None)` - If no path exists
/// * `Err(GraphError)` - If the source or target node is not in the graph
///
/// # Time Complexity
/// O((V + E) log V) where V is the number of vertices and E is the number of edges.
///
/// # Space Complexity
/// O(V) for the distance array and predecessor tracking.
///
/// # Example
/// ```rust
/// use scirs2_graph::{Graph, dijkstra_path};
///
/// let mut graph: Graph<String, f64> = Graph::new();
/// graph.add_node("A".to_string());
/// graph.add_node("B".to_string());
/// graph.add_edge("A".to_string(), "B".to_string(), 1.0).unwrap();
///
/// let path = dijkstra_path(&graph, &"A".to_string(), &"B".to_string()).unwrap();
/// assert!(path.is_some());
/// ```
#[allow(dead_code)]
pub fn dijkstra_path<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    source: &N,
    target: &N,
) -> Result<Option<Path<N, E>>>
where
    N: Node + std::fmt::Debug,
    E: EdgeWeight
        + scirs2_core::numeric::Zero
        + scirs2_core::numeric::One
        + std::ops::Add<Output = E>
        + PartialOrd
        + std::marker::Copy
        + std::fmt::Debug
        + std::default::Default,
    Ix: petgraph::graph::IndexType,
{
    #[allow(deprecated)]
    shortest_path(graph, source, target)
}

/// Finds the shortest path in a directed graph
///
/// # Deprecation Notice
/// This function will be updated in v1.0 to return a standardized `PathResult` type
/// that provides more detailed information about the path and search process.
/// The current API will remain available but deprecated.
#[deprecated(
    since = "0.1.0-beta.2",
    note = "Use `dijkstra_path_digraph` for future compatibility. This function will return PathResult in v1.0"
)]
#[allow(dead_code)]
pub fn shortest_path_digraph<N, E, Ix>(
    graph: &DiGraph<N, E, Ix>,
    source: &N,
    target: &N,
) -> Result<Option<Path<N, E>>>
where
    N: Node + std::fmt::Debug,
    E: EdgeWeight
        + scirs2_core::numeric::Zero
        + scirs2_core::numeric::One
        + std::ops::Add<Output = E>
        + PartialOrd
        + std::marker::Copy
        + std::fmt::Debug
        + std::default::Default,
    Ix: petgraph::graph::IndexType,
{
    // Implementation similar to undirected version
    // Check if source and target are in the graph
    if !graph.has_node(source) {
        return Err(GraphError::InvalidGraph(format!(
            "Source node {source:?} not found"
        )));
    }
    if !graph.has_node(target) {
        return Err(GraphError::InvalidGraph(format!(
            "Target node {target:?} not found"
        )));
    }

    let source_idx = graph
        .inner()
        .node_indices()
        .find(|&idx| graph.inner()[idx] == *source)
        .unwrap();
    let target_idx = graph
        .inner()
        .node_indices()
        .find(|&idx| graph.inner()[idx] == *target)
        .unwrap();

    // Use petgraph's Dijkstra algorithm implementation
    let results = dijkstra(graph.inner(), source_idx, Some(target_idx), |e| *e.weight());

    // If target is not reachable, return None
    if !results.contains_key(&target_idx) {
        return Ok(None);
    }

    let total_weight = results[&target_idx];

    // Reconstruct the path
    let mut path = Vec::new();
    let mut current = target_idx;

    path.push(graph.inner()[current].clone());

    // Backtrack to find the path
    while current != source_idx {
        let min_prev = graph
            .inner()
            .edges_directed(current, petgraph::Direction::Incoming)
            .filter_map(|e| {
                let from = e.source();
                let edge_weight = *e.weight();

                // Check if this node is part of the shortest path
                if let Some(from_dist) = results.get(&from) {
                    // If this edge is part of the shortest path
                    if *from_dist + edge_weight == results[&current] {
                        return Some((from, *from_dist));
                    }
                }
                None
            })
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

        if let Some((prev, _)) = min_prev {
            current = prev;
            path.push(graph.inner()[current].clone());
        } else {
            // This shouldn't happen if Dijkstra's algorithm works correctly
            return Err(GraphError::AlgorithmError(
                "Failed to reconstruct path".to_string(),
            ));
        }
    }

    // Reverse the path to get it from source to target
    path.reverse();

    Ok(Some(Path {
        nodes: path,
        total_weight,
    }))
}

/// Finds the shortest path between source and target nodes in a directed graph using Dijkstra's algorithm
///
/// This is the modern API that provides a clearer name for directed graph path finding.
/// This is the recommended function to use for directed graphs.
///
/// # Arguments
/// * `graph` - The directed graph to search in
/// * `source` - The source node
/// * `target` - The target node
///
/// # Returns
/// * `Ok(Some(Path))` - If a path exists
/// * `Ok(None)` - If no path exists
/// * `Err(GraphError)` - If the source or target node is not in the graph
///
/// # Time Complexity
/// O((V + E) log V) where V is the number of vertices and E is the number of edges.
///
/// # Space Complexity
/// O(V) for the distance array and predecessor tracking.
#[allow(dead_code)]
pub fn dijkstra_path_digraph<N, E, Ix>(
    graph: &DiGraph<N, E, Ix>,
    source: &N,
    target: &N,
) -> Result<Option<Path<N, E>>>
where
    N: Node + std::fmt::Debug,
    E: EdgeWeight
        + scirs2_core::numeric::Zero
        + scirs2_core::numeric::One
        + std::ops::Add<Output = E>
        + PartialOrd
        + std::marker::Copy
        + std::fmt::Debug
        + std::default::Default,
    Ix: petgraph::graph::IndexType,
{
    #[allow(deprecated)]
    shortest_path_digraph(graph, source, target)
}

/// Computes all-pairs shortest paths using the Floyd-Warshall algorithm
///
/// Returns a matrix where entry (i, j) contains the shortest distance from node i to node j.
/// If there's no path, the entry will be infinity.
///
/// # Arguments
/// * `graph` - The graph to analyze (works for both directed and undirected)
///
/// # Returns
/// * `Result<Array2<f64>>` - A matrix of shortest distances
///
/// # Time Complexity
/// O(V³) where V is the number of vertices. This algorithm computes shortest paths
/// between all pairs of vertices using dynamic programming.
///
/// # Space Complexity
/// O(V²) for the distance matrix.
///
/// # Note
/// This algorithm can detect negative cycles if the diagonal contains negative values
/// after completion. It works correctly with negative edge weights but not with
/// negative cycles.
#[allow(dead_code)]
pub fn floyd_warshall<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
) -> Result<scirs2_core::ndarray::Array2<f64>>
where
    N: Node + std::fmt::Debug,
    E: EdgeWeight + Into<f64> + scirs2_core::numeric::Zero + Copy,
    Ix: petgraph::graph::IndexType,
{
    let n = graph.node_count();

    if n == 0 {
        return Ok(scirs2_core::ndarray::Array2::zeros((0, 0)));
    }

    // Initialize distance matrix
    let mut dist = scirs2_core::ndarray::Array2::from_elem((n, n), f64::INFINITY);

    // Set diagonal to 0 (distance from a node to itself)
    for i in 0..n {
        dist[[i, i]] = 0.0;
    }

    // Initialize with direct edge weights
    for edge in graph.inner().edge_references() {
        let i = edge.source().index();
        let j = edge.target().index();
        let weight: f64 = (*edge.weight()).into();

        dist[[i, j]] = weight;
        // For undirected graphs, also set the reverse direction
        dist[[j, i]] = weight;
    }

    // Floyd-Warshall algorithm
    for k in 0..n {
        for i in 0..n {
            for j in 0..n {
                let alt = dist[[i, k]] + dist[[k, j]];
                if alt < dist[[i, j]] {
                    dist[[i, j]] = alt;
                }
            }
        }
    }

    Ok(dist)
}

/// Computes all-pairs shortest paths for a directed graph using Floyd-Warshall
#[allow(dead_code)]
pub fn floyd_warshall_digraph<N, E, Ix>(
    graph: &DiGraph<N, E, Ix>,
) -> Result<scirs2_core::ndarray::Array2<f64>>
where
    N: Node + std::fmt::Debug,
    E: EdgeWeight + Into<f64> + scirs2_core::numeric::Zero + Copy,
    Ix: petgraph::graph::IndexType,
{
    let n = graph.node_count();

    if n == 0 {
        return Ok(scirs2_core::ndarray::Array2::zeros((0, 0)));
    }

    // Initialize distance matrix
    let mut dist = scirs2_core::ndarray::Array2::from_elem((n, n), f64::INFINITY);

    // Set diagonal to 0 (distance from a node to itself)
    for i in 0..n {
        dist[[i, i]] = 0.0;
    }

    // Initialize with direct edge weights
    for edge in graph.inner().edge_references() {
        let i = edge.source().index();
        let j = edge.target().index();
        let weight: f64 = (*edge.weight()).into();

        dist[[i, j]] = weight;
    }

    // Floyd-Warshall algorithm
    for k in 0..n {
        for i in 0..n {
            for j in 0..n {
                let alt = dist[[i, k]] + dist[[k, j]];
                if alt < dist[[i, j]] {
                    dist[[i, j]] = alt;
                }
            }
        }
    }

    Ok(dist)
}

/// A* search algorithm for finding the shortest path with a heuristic
///
/// # Arguments
/// * `graph` - The graph to search
/// * `source` - Source node
/// * `target` - Target node
/// * `heuristic` - Heuristic function that estimates distance from a node to the target
///
/// # Returns
/// * `Result<AStarResult>` - The shortest path and its cost
///
/// # Time Complexity
/// O(b^d) where b is the branching factor and d is the depth of the solution.
/// In the worst case with an inadmissible heuristic: O(V²).
/// With a perfect heuristic: O(V).
/// Typical case with good heuristic: O(V log V).
///
/// # Space Complexity
/// O(V) for the open set, closed set, and path reconstruction.
///
/// # Note
/// The heuristic function must be admissible (never overestimate the actual cost)
/// and consistent (satisfy the triangle inequality) to guarantee finding the
/// optimal path.
#[allow(dead_code)]
pub fn astar_search<N, E, Ix, H>(
    graph: &Graph<N, E, Ix>,
    source: &N,
    target: &N,
    heuristic: H,
) -> Result<AStarResult<N, E>>
where
    N: Node + std::fmt::Debug + Clone + Hash + Eq,
    E: EdgeWeight
        + Clone
        + std::ops::Add<Output = E>
        + scirs2_core::numeric::Zero
        + PartialOrd
        + Copy,
    Ix: petgraph::graph::IndexType,
    H: Fn(&N) -> E,
{
    if !graph.contains_node(source) || !graph.contains_node(target) {
        return Err(GraphError::node_not_found("node"));
    }

    let mut open_set = BinaryHeap::new();
    let mut g_score: HashMap<N, E> = HashMap::new();
    let mut came_from: HashMap<N, N> = HashMap::new();

    g_score.insert(source.clone(), E::zero());

    open_set.push(AStarState {
        node: source.clone(),
        cost: E::zero(),
        heuristic: heuristic(source),
        path: vec![source.clone()],
    });

    while let Some(current_state) = open_set.pop() {
        let current = &current_state.node;

        if current == target {
            return Ok(AStarResult {
                path: current_state.path,
                cost: current_state.cost,
            });
        }

        let current_g = g_score.get(current).cloned().unwrap_or_else(E::zero);

        if let Ok(neighbors) = graph.neighbors(current) {
            for neighbor in neighbors {
                if let Ok(edge_weight) = graph.edge_weight(current, &neighbor) {
                    let tentative_g = current_g + edge_weight;

                    let current_neighbor_g = g_score.get(&neighbor);
                    if current_neighbor_g.is_none() || tentative_g < *current_neighbor_g.unwrap() {
                        came_from.insert(neighbor.clone(), current.clone());
                        g_score.insert(neighbor.clone(), tentative_g);

                        let mut new_path = current_state.path.clone();
                        new_path.push(neighbor.clone());

                        open_set.push(AStarState {
                            node: neighbor.clone(),
                            cost: tentative_g,
                            heuristic: heuristic(&neighbor),
                            path: new_path,
                        });
                    }
                }
            }
        }
    }

    Err(GraphError::NoPath {
        src_node: format!("{source:?}"),
        target: format!("{target:?}"),
        nodes: 0,
        edges: 0,
    })
}

/// A* search for directed graphs
#[allow(dead_code)]
pub fn astar_search_digraph<N, E, Ix, H>(
    graph: &DiGraph<N, E, Ix>,
    source: &N,
    target: &N,
    heuristic: H,
) -> Result<AStarResult<N, E>>
where
    N: Node + std::fmt::Debug + Clone + Hash + Eq,
    E: EdgeWeight
        + Clone
        + std::ops::Add<Output = E>
        + scirs2_core::numeric::Zero
        + PartialOrd
        + Copy,
    Ix: petgraph::graph::IndexType,
    H: Fn(&N) -> E,
{
    if !graph.contains_node(source) || !graph.contains_node(target) {
        return Err(GraphError::node_not_found("node"));
    }

    let mut open_set = BinaryHeap::new();
    let mut g_score: HashMap<N, E> = HashMap::new();
    let mut came_from: HashMap<N, N> = HashMap::new();

    g_score.insert(source.clone(), E::zero());

    open_set.push(AStarState {
        node: source.clone(),
        cost: E::zero(),
        heuristic: heuristic(source),
        path: vec![source.clone()],
    });

    while let Some(current_state) = open_set.pop() {
        let current = &current_state.node;

        if current == target {
            return Ok(AStarResult {
                path: current_state.path,
                cost: current_state.cost,
            });
        }

        let current_g = g_score.get(current).cloned().unwrap_or_else(E::zero);

        if let Ok(successors) = graph.successors(current) {
            for neighbor in successors {
                if let Ok(edge_weight) = graph.edge_weight(current, &neighbor) {
                    let tentative_g = current_g + edge_weight;

                    let current_neighbor_g = g_score.get(&neighbor);
                    if current_neighbor_g.is_none() || tentative_g < *current_neighbor_g.unwrap() {
                        came_from.insert(neighbor.clone(), current.clone());
                        g_score.insert(neighbor.clone(), tentative_g);

                        let mut new_path = current_state.path.clone();
                        new_path.push(neighbor.clone());

                        open_set.push(AStarState {
                            node: neighbor.clone(),
                            cost: tentative_g,
                            heuristic: heuristic(&neighbor),
                            path: new_path,
                        });
                    }
                }
            }
        }
    }

    Err(GraphError::NoPath {
        src_node: format!("{source:?}"),
        target: format!("{target:?}"),
        nodes: 0,
        edges: 0,
    })
}

/// Finds K shortest paths between two nodes using Yen's algorithm
///
/// Returns up to k shortest paths sorted by total weight.
/// Each path includes the total weight and the sequence of nodes.
#[allow(dead_code)]
pub fn k_shortest_paths<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    source: &N,
    target: &N,
    k: usize,
) -> Result<Vec<(f64, Vec<N>)>>
where
    N: Node + std::fmt::Debug + Clone + Hash + Eq + Ord,
    E: EdgeWeight
        + Into<f64>
        + Clone
        + scirs2_core::numeric::Zero
        + scirs2_core::numeric::One
        + std::ops::Add<Output = E>
        + PartialOrd
        + std::marker::Copy
        + std::fmt::Debug
        + std::default::Default,
    Ix: petgraph::graph::IndexType,
{
    if k == 0 {
        return Ok(vec![]);
    }

    // Check if nodes exist
    if !graph.contains_node(source) || !graph.contains_node(target) {
        return Err(GraphError::node_not_found("node"));
    }

    let mut paths = Vec::new();
    let mut candidates = std::collections::BinaryHeap::new();

    // Find the shortest path first
    match dijkstra_path(graph, source, target) {
        Ok(Some(path)) => {
            let weight: f64 = path.total_weight.into();
            paths.push((weight, path.nodes));
        }
        Ok(None) => return Ok(vec![]), // No path exists
        Err(e) => return Err(e),
    }

    // Find k-1 more paths
    for i in 0..k - 1 {
        if i >= paths.len() {
            break;
        }

        let (_, prev_path) = &paths[i];

        // For each node in the previous path (except the last)
        for j in 0..prev_path.len() - 1 {
            let spur_node = &prev_path[j];
            let root_path = &prev_path[..=j];

            // Store edges to remove temporarily
            let mut removed_edges = Vec::new();

            // Remove edges that are part of previous paths with same root
            for (_, path) in &paths {
                if path.len() > j && &path[..=j] == root_path && j + 1 < path.len() {
                    removed_edges.push((path[j].clone(), path[j + 1].clone()));
                }
            }

            // Find shortest path from spur_node to target avoiding removed edges
            if let Ok((spur_weight, spur_path)) =
                shortest_path_avoiding_edges(graph, spur_node, target, &removed_edges, root_path)
            {
                // Calculate total weight of the new path
                let mut total_weight = spur_weight;
                for idx in 0..j {
                    if let Ok(edge_weight) = graph.edge_weight(&prev_path[idx], &prev_path[idx + 1])
                    {
                        let weight: f64 = edge_weight.into();
                        total_weight += weight;
                    }
                }

                // Construct the complete path
                let mut complete_path = root_path[..j].to_vec();
                complete_path.extend(spur_path);

                // Add to candidates with negative weight for min-heap behavior
                candidates.push((
                    std::cmp::Reverse(ordered_float::OrderedFloat(total_weight)),
                    complete_path.clone(),
                ));
            }
        }
    }

    // Extract paths from candidates
    while paths.len() < k && !candidates.is_empty() {
        let (std::cmp::Reverse(ordered_float::OrderedFloat(weight)), path) =
            candidates.pop().unwrap();

        // Check if this path is already in our result
        let is_duplicate = paths.iter().any(|(_, p)| p == &path);
        if !is_duplicate {
            paths.push((weight, path));
        }
    }

    Ok(paths)
}

/// Helper function for k-shortest paths that finds shortest path avoiding certain edges
#[allow(dead_code)]
fn shortest_path_avoiding_edges<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    source: &N,
    target: &N,
    avoided_edges: &[(N, N)],
    excluded_nodes: &[N],
) -> Result<(f64, Vec<N>)>
where
    N: Node + std::fmt::Debug + Clone + Hash + Eq + Ord,
    E: EdgeWeight + Into<f64>,
    Ix: petgraph::graph::IndexType,
{
    use std::cmp::Reverse;

    let mut distances: HashMap<N, f64> = HashMap::new();
    let mut previous: HashMap<N, N> = HashMap::new();
    let mut heap = BinaryHeap::new();

    distances.insert(source.clone(), 0.0);
    heap.push((Reverse(ordered_float::OrderedFloat(0.0)), source.clone()));

    while let Some((Reverse(ordered_float::OrderedFloat(dist)), node)) = heap.pop() {
        if &node == target {
            // Reconstruct path
            let mut path = vec![target.clone()];
            let mut current = target.clone();

            while let Some(prev) = previous.get(&current) {
                path.push(prev.clone());
                current = prev.clone();
            }

            path.reverse();
            return Ok((dist, path));
        }

        if distances.get(&node).is_none_or(|&d| dist > d) {
            continue;
        }

        if let Ok(neighbors) = graph.neighbors(&node) {
            for neighbor in neighbors {
                // Skip if this edge should be avoided
                if avoided_edges.contains(&(node.clone(), neighbor.clone())) {
                    continue;
                }

                // Skip if this node is in excluded _nodes (except source and target)
                if &neighbor != source && &neighbor != target && excluded_nodes.contains(&neighbor)
                {
                    continue;
                }

                if let Ok(edge_weight) = graph.edge_weight(&node, &neighbor) {
                    let weight: f64 = edge_weight.into();
                    let new_dist = dist + weight;

                    if new_dist < *distances.get(&neighbor).unwrap_or(&f64::INFINITY) {
                        distances.insert(neighbor.clone(), new_dist);
                        previous.insert(neighbor.clone(), node.clone());
                        heap.push((Reverse(ordered_float::OrderedFloat(new_dist)), neighbor));
                    }
                }
            }
        }
    }

    Err(GraphError::NoPath {
        src_node: format!("{source:?}"),
        target: format!("{target:?}"),
        nodes: 0,
        edges: 0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(deprecated)]
    fn test_shortest_path() {
        let mut graph: Graph<i32, f64> = Graph::new();

        // Create a simple graph
        graph.add_edge(1, 2, 4.0).unwrap();
        graph.add_edge(1, 3, 2.0).unwrap();
        graph.add_edge(2, 3, 1.0).unwrap();
        graph.add_edge(2, 4, 5.0).unwrap();
        graph.add_edge(3, 4, 8.0).unwrap();

        let path = shortest_path(&graph, &1, &4).unwrap().unwrap();

        // The shortest path from 1 to 4 should be 1 -> 3 -> 2 -> 4 with total weight 8.0
        assert_eq!(path.total_weight, 8.0);
        assert_eq!(path.nodes, vec![1, 3, 2, 4]);
    }

    #[test]
    fn test_floyd_warshall() {
        let mut graph: Graph<i32, f64> = Graph::new();

        // Create a triangle
        graph.add_edge(0, 1, 1.0).unwrap();
        graph.add_edge(1, 2, 2.0).unwrap();
        graph.add_edge(2, 0, 3.0).unwrap();

        let distances = floyd_warshall(&graph).unwrap();

        // Check some distances
        assert_eq!(distances[[0, 0]], 0.0);
        assert_eq!(distances[[0, 1]], 1.0);
        assert_eq!(distances[[0, 2]], 3.0); // 0->1->2 is shorter than 0->2
        assert_eq!(distances[[1, 0]], 1.0); // Undirected graph
    }

    #[test]
    fn test_astar_search() {
        let mut graph: Graph<(i32, i32), f64> = Graph::new();

        // Create a simple grid
        graph.add_edge((0, 0), (0, 1), 1.0).unwrap();
        graph.add_edge((0, 1), (1, 1), 1.0).unwrap();
        graph.add_edge((1, 1), (1, 0), 1.0).unwrap();
        graph.add_edge((1, 0), (0, 0), 1.0).unwrap();

        // Manhattan distance heuristic
        let heuristic = |&(x, y): &(i32, i32)| -> f64 { ((1 - x).abs() + (1 - y).abs()) as f64 };

        let result = astar_search(&graph, &(0, 0), &(1, 1), heuristic);
        let result = result.unwrap();
        assert_eq!(result.cost, 2.0);
        assert_eq!(result.path.len(), 3); // Start, one intermediate, goal
    }

    #[test]
    fn test_k_shortest_paths() {
        let mut graph: Graph<char, f64> = Graph::new();

        // Create a graph with multiple paths
        graph.add_edge('A', 'B', 2.0).unwrap();
        graph.add_edge('B', 'D', 2.0).unwrap();
        graph.add_edge('A', 'C', 1.0).unwrap();
        graph.add_edge('C', 'D', 4.0).unwrap();
        graph.add_edge('B', 'C', 1.0).unwrap();

        let paths = k_shortest_paths(&graph, &'A', &'D', 3).unwrap();

        assert!(paths.len() >= 2);
        assert_eq!(paths[0].0, 4.0); // A->B->D
        assert_eq!(paths[0].1, vec!['A', 'B', 'D']);
    }
}
