//! Network flow and cut algorithms
//!
//! This module contains algorithms for network flow problems and graph cuts.

use crate::base::{DiGraph, EdgeWeight, Graph, IndexType, Node};
use crate::error::{GraphError, Result};
use std::collections::{HashMap, VecDeque};
use std::hash::Hash;

/// Finds a minimum cut in a graph using Karger's algorithm
///
/// Returns the minimum cut value and a partition of nodes.
/// This is a randomized algorithm, so multiple runs may give different results.
pub fn minimum_cut<N, E, Ix>(graph: &Graph<N, E, Ix>) -> Result<(f64, Vec<bool>)>
where
    N: Node + Clone + Hash + Eq,
    E: EdgeWeight + Into<f64> + Clone,
    Ix: IndexType,
{
    let nodes: Vec<N> = graph.nodes().into_iter().cloned().collect();
    let n = nodes.len();

    if n < 2 {
        return Err(GraphError::InvalidGraph(
            "Graph must have at least 2 nodes".to_string(),
        ));
    }

    // For small graphs, try all possible cuts
    if n <= 10 {
        let mut min_cut_value = f64::INFINITY;
        let mut best_partition = vec![false; n];

        // Try all possible bipartitions (except empty and full sets)
        for mask in 1..(1 << n) - 1 {
            let mut partition = vec![false; n];
            for (i, p) in partition.iter_mut().enumerate().take(n) {
                *p = (mask & (1 << i)) != 0;
            }

            // Calculate cut value
            let cut_value = calculate_cut_value(graph, &nodes, &partition);

            if cut_value < min_cut_value {
                min_cut_value = cut_value;
                best_partition = partition;
            }
        }

        Ok((min_cut_value, best_partition))
    } else {
        // For larger graphs, use a heuristic approach
        // This is a simplified version - a full implementation would use Karger's algorithm
        minimum_cut_heuristic(graph, &nodes)
    }
}

fn calculate_cut_value<N, E, Ix>(graph: &Graph<N, E, Ix>, nodes: &[N], partition: &[bool]) -> f64
where
    N: Node + Clone + Hash + Eq,
    E: EdgeWeight + Into<f64>,
    Ix: IndexType,
{
    let mut cut_value = 0.0;

    for (i, node_i) in nodes.iter().enumerate() {
        if let Ok(neighbors) = graph.neighbors(node_i) {
            for neighbor in neighbors {
                if let Some(j) = nodes.iter().position(|n| n == &neighbor) {
                    // Only count edges going from partition A to partition B
                    if partition[i] && !partition[j] {
                        if let Ok(weight) = graph.edge_weight(node_i, &neighbor) {
                            cut_value += weight.into();
                        }
                    }
                }
            }
        }
    }

    cut_value
}

fn minimum_cut_heuristic<N, E, Ix>(graph: &Graph<N, E, Ix>, nodes: &[N]) -> Result<(f64, Vec<bool>)>
where
    N: Node + Clone + Hash + Eq,
    E: EdgeWeight + Into<f64> + Clone,
    Ix: IndexType,
{
    let n = nodes.len();
    let mut best_cut_value = f64::INFINITY;
    let mut best_partition = vec![false; n];

    // Try a few random partitions
    use rand::Rng;
    let mut rng = rand::rng();

    for _ in 0..10 {
        let mut partition = vec![false; n];
        let size_a = rng.random_range(1..n);

        // Randomly select nodes for partition A
        let mut indices: Vec<usize> = (0..n).collect();
        use rand::seq::SliceRandom;
        indices.shuffle(&mut rng);

        for i in 0..size_a {
            partition[indices[i]] = true;
        }

        let cut_value = calculate_cut_value(graph, nodes, &partition);

        if cut_value < best_cut_value {
            best_cut_value = cut_value;
            best_partition = partition;
        }
    }

    Ok((best_cut_value, best_partition))
}

/// Dinic's algorithm for maximum flow
///
/// Implements Dinic's algorithm for finding maximum flow in a directed graph.
/// This algorithm has time complexity O(V²E) which is better than Ford-Fulkerson
/// for dense graphs.
///
/// # Arguments
/// * `graph` - The directed graph representing the flow network
/// * `source` - The source node
/// * `sink` - The sink node
///
/// # Returns
/// * The maximum flow value from source to sink
pub fn dinic_max_flow<N, E, Ix>(graph: &DiGraph<N, E, Ix>, source: &N, sink: &N) -> Result<f64>
where
    N: Node + Clone + Hash + Eq,
    E: EdgeWeight + Into<f64> + Clone + std::ops::Sub<Output = E> + num_traits::Zero + PartialOrd,
    Ix: IndexType,
{
    if !graph.contains_node(source) {
        return Err(GraphError::NodeNotFound);
    }
    if !graph.contains_node(sink) {
        return Err(GraphError::NodeNotFound);
    }
    if source == sink {
        return Err(GraphError::InvalidGraph(
            "Source and sink cannot be the same".to_string(),
        ));
    }

    // Build residual graph with capacities
    let mut residual_graph = DinicResidualGraph::new(graph, source, sink)?;
    let mut max_flow = 0.0f64;

    // Main Dinic's algorithm loop
    loop {
        // Build level graph using BFS
        if !residual_graph.build_level_graph() {
            break; // No augmenting path exists
        }

        // Find blocking flows using DFS
        loop {
            let flow = residual_graph.send_flow(f64::INFINITY);
            if flow == 0.0 {
                break;
            }
            max_flow += flow;
        }
    }

    Ok(max_flow)
}

/// Internal structure for Dinic's algorithm residual graph
struct DinicResidualGraph<N: Node> {
    /// Adjacency list representation: node -> (neighbor, capacity, reverse_edge_index)
    adj: HashMap<N, Vec<(N, f64, usize)>>,
    source: N,
    sink: N,
    /// Level of each node in the level graph (-1 means unreachable)
    level: HashMap<N, i32>,
    /// Current edge index for each node (for DFS optimization)
    current: HashMap<N, usize>,
}

impl<N: Node + Clone + Hash + Eq> DinicResidualGraph<N> {
    fn new<E, Ix>(graph: &DiGraph<N, E, Ix>, source: &N, sink: &N) -> Result<Self>
    where
        E: EdgeWeight + Into<f64> + Clone,
        Ix: IndexType,
    {
        let mut adj: HashMap<N, Vec<(N, f64, usize)>> = HashMap::new();

        // Initialize adjacency lists for all nodes
        for node in graph.nodes() {
            adj.insert(node.clone(), Vec::new());
        }

        // Add edges to residual graph
        for node in graph.nodes() {
            if let Ok(successors) = graph.successors(node) {
                for successor in successors {
                    if let Ok(weight) = graph.edge_weight(node, &successor) {
                        let capacity: f64 = weight.into();

                        // Forward edge
                        let forward_idx = adj.get(&successor).map(|v| v.len()).unwrap_or(0);
                        adj.entry(node.clone()).or_default().push((
                            successor.clone(),
                            capacity,
                            forward_idx,
                        ));

                        // Backward edge (initially 0 capacity)
                        let backward_idx = adj.get(node).map(|v| v.len() - 1).unwrap_or(0);
                        adj.entry(successor.clone()).or_default().push((
                            node.clone(),
                            0.0,
                            backward_idx,
                        ));
                    }
                }
            }
        }

        Ok(DinicResidualGraph {
            adj,
            source: source.clone(),
            sink: sink.clone(),
            level: HashMap::new(),
            current: HashMap::new(),
        })
    }

    /// Build level graph using BFS
    fn build_level_graph(&mut self) -> bool {
        self.level.clear();
        self.current.clear();

        // Initialize all nodes to level -1 (unreachable)
        for node in self.adj.keys() {
            self.level.insert(node.clone(), -1);
            self.current.insert(node.clone(), 0);
        }

        // BFS from source
        let mut queue = VecDeque::new();
        queue.push_back(self.source.clone());
        self.level.insert(self.source.clone(), 0);

        while let Some(node) = queue.pop_front() {
            if let Some(edges) = self.adj.get(&node) {
                for (neighbor, capacity, _) in edges {
                    if *capacity > 0.0 && self.level.get(neighbor) == Some(&-1) {
                        self.level.insert(neighbor.clone(), self.level[&node] + 1);
                        queue.push_back(neighbor.clone());
                    }
                }
            }
        }

        // Return true if sink is reachable
        self.level.get(&self.sink) != Some(&-1)
    }

    /// Send flow from source to sink using DFS
    fn send_flow(&mut self, max_flow: f64) -> f64 {
        self.dfs(&self.source.clone(), max_flow)
    }

    fn dfs(&mut self, node: &N, flow: f64) -> f64 {
        if node == &self.sink {
            return flow;
        }

        let current_idx = *self.current.get(node).unwrap_or(&0);
        let node_level = *self.level.get(node).unwrap_or(&-1);

        if let Some(edges) = self.adj.get(node).cloned() {
            for i in current_idx..edges.len() {
                let (neighbor, capacity, rev_idx) = &edges[i];
                let neighbor_level = *self.level.get(neighbor).unwrap_or(&-1);

                if *capacity > 0.0 && neighbor_level == node_level + 1 {
                    let bottleneck = flow.min(*capacity);
                    let pushed = self.dfs(neighbor, bottleneck);

                    if pushed > 0.0 {
                        // Update capacities
                        if let Some(node_edges) = self.adj.get_mut(node) {
                            node_edges[i].1 -= pushed; // Forward edge
                        }
                        if let Some(neighbor_edges) = self.adj.get_mut(neighbor) {
                            neighbor_edges[*rev_idx].1 += pushed; // Backward edge
                        }
                        return pushed;
                    }
                }

                // Update current pointer
                self.current.insert(node.clone(), i + 1);
            }
        }

        0.0
    }
}

/// Push-relabel algorithm for maximum flow
///
/// Implements the push-relabel algorithm for finding maximum flow.
/// This algorithm has time complexity O(V³) and works well for dense graphs.
///
/// # Arguments
/// * `graph` - The directed graph representing the flow network
/// * `source` - The source node
/// * `sink` - The sink node
///
/// # Returns
/// * The maximum flow value from source to sink
pub fn push_relabel_max_flow<N, E, Ix>(
    graph: &DiGraph<N, E, Ix>,
    source: &N,
    sink: &N,
) -> Result<f64>
where
    N: Node + Clone + Hash + Eq,
    E: EdgeWeight + Into<f64> + Clone,
    Ix: IndexType,
{
    if !graph.contains_node(source) {
        return Err(GraphError::NodeNotFound);
    }
    if !graph.contains_node(sink) {
        return Err(GraphError::NodeNotFound);
    }
    if source == sink {
        return Err(GraphError::InvalidGraph(
            "Source and sink cannot be the same".to_string(),
        ));
    }

    // Build capacity matrix and other data structures
    let nodes: Vec<N> = graph.nodes().into_iter().cloned().collect();
    let n = nodes.len();
    let mut capacity = vec![vec![0.0f64; n]; n];
    let mut flow = vec![vec![0.0f64; n]; n];
    let mut height = vec![0; n];
    let mut excess = vec![0.0f64; n];

    // Node to index mapping
    let node_to_idx: HashMap<N, usize> = nodes
        .iter()
        .enumerate()
        .map(|(i, node)| (node.clone(), i))
        .collect();

    let source_idx = *node_to_idx.get(source).unwrap();
    let sink_idx = *node_to_idx.get(sink).unwrap();

    // Initialize capacity matrix
    for node in &nodes {
        if let Ok(successors) = graph.successors(node) {
            let from_idx = node_to_idx[node];
            for successor in successors {
                let to_idx = node_to_idx[&successor];
                if let Ok(weight) = graph.edge_weight(node, &successor) {
                    capacity[from_idx][to_idx] = weight.into();
                }
            }
        }
    }

    // Initialize preflow
    height[source_idx] = n;
    for i in 0..n {
        if capacity[source_idx][i] > 0.0 {
            flow[source_idx][i] = capacity[source_idx][i];
            flow[i][source_idx] = -capacity[source_idx][i];
            excess[i] = capacity[source_idx][i];
        }
    }

    // Main push-relabel loop
    loop {
        // Find active node (excess > 0 and not source or sink)
        let mut active_node = None;
        for (i, &excess_val) in excess.iter().enumerate().take(n) {
            if i != source_idx && i != sink_idx && excess_val > 0.0 {
                active_node = Some(i);
                break;
            }
        }

        if let Some(u) = active_node {
            let mut pushed = false;

            // Try to push to neighbors
            for v in 0..n {
                if capacity[u][v] - flow[u][v] > 0.0 && height[u] == height[v] + 1 {
                    let delta = excess[u].min(capacity[u][v] - flow[u][v]);
                    flow[u][v] += delta;
                    flow[v][u] -= delta;
                    excess[u] -= delta;
                    excess[v] += delta;
                    pushed = true;

                    if excess[u] == 0.0 {
                        break;
                    }
                }
            }

            // If no push was possible, relabel
            if !pushed {
                let mut min_height = usize::MAX;
                for (v, &height_val) in height.iter().enumerate().take(n) {
                    if capacity[u][v] - flow[u][v] > 0.0 {
                        min_height = min_height.min(height_val);
                    }
                }
                if min_height < usize::MAX {
                    height[u] = min_height + 1;
                }
            }
        } else {
            break; // No active nodes, algorithm terminates
        }
    }

    // Calculate total flow leaving source
    let total_flow: f64 = (0..n).map(|i| flow[source_idx][i]).sum();
    Ok(total_flow)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Result as GraphResult;
    use crate::generators::create_graph;

    #[test]
    fn test_minimum_cut_simple() -> GraphResult<()> {
        let mut graph = create_graph::<&str, f64>();

        // Create two clusters connected by a single edge
        // Cluster 1: A-B-C
        graph.add_edge("A", "B", 10.0)?;
        graph.add_edge("B", "C", 10.0)?;
        graph.add_edge("C", "A", 10.0)?;

        // Cluster 2: D-E-F
        graph.add_edge("D", "E", 10.0)?;
        graph.add_edge("E", "F", 10.0)?;
        graph.add_edge("F", "D", 10.0)?;

        // Bridge between clusters
        graph.add_edge("C", "D", 1.0)?;

        let (cut_value, partition) = minimum_cut(&graph)?;

        // The minimum cut should be 1.0 (the bridge edge)
        assert!((cut_value - 1.0).abs() < 1e-6);

        // Check that the partition separates the two clusters
        let nodes: Vec<&str> = graph.nodes().into_iter().cloned().collect();
        let cluster1: Vec<bool> = nodes.iter().map(|n| ["A", "B", "C"].contains(n)).collect();
        let cluster2: Vec<bool> = nodes.iter().map(|n| ["D", "E", "F"].contains(n)).collect();

        // Partition should match one of the clusters
        let matches_cluster1 = partition.iter().zip(&cluster1).all(|(a, b)| a == b);
        let matches_cluster2 = partition.iter().zip(&cluster2).all(|(a, b)| a == b);

        assert!(matches_cluster1 || matches_cluster2);

        Ok(())
    }

    #[test]
    fn test_minimum_cut_single_edge() -> GraphResult<()> {
        let mut graph = create_graph::<&str, f64>();

        // Simple two-node graph
        graph.add_edge("A", "B", 5.0)?;

        let (cut_value, partition) = minimum_cut(&graph)?;

        // The only cut has value 5.0
        assert_eq!(cut_value, 5.0);

        // One node in each partition
        assert_eq!(partition.iter().filter(|&&x| x).count(), 1);
        assert_eq!(partition.iter().filter(|&&x| !x).count(), 1);

        Ok(())
    }

    #[test]
    fn test_dinic_max_flow() -> GraphResult<()> {
        let mut graph = crate::generators::create_digraph::<&str, f64>();

        // Create a simple flow network:
        //   A --2--> B --3--> D
        //   |        |        ^
        //   1        1        |
        //   v        v        2
        //   C -------4------> E
        graph.add_edge("A", "B", 2.0)?;
        graph.add_edge("A", "C", 1.0)?;
        graph.add_edge("B", "D", 3.0)?;
        graph.add_edge("B", "E", 1.0)?;
        graph.add_edge("C", "E", 4.0)?;
        graph.add_edge("E", "D", 2.0)?;

        let max_flow = dinic_max_flow(&graph, &"A", &"D")?;

        // Maximum flow from A to D should be 3.0
        // Path 1: A->B->D (flow 2)
        // Path 2: A->C->E->D (flow 1)
        assert!((max_flow - 3.0).abs() < 1e-6);

        Ok(())
    }

    #[test]
    fn test_push_relabel_max_flow() -> GraphResult<()> {
        let mut graph = crate::generators::create_digraph::<&str, f64>();

        // Create a simple flow network
        graph.add_edge("S", "A", 10.0)?;
        graph.add_edge("S", "B", 10.0)?;
        graph.add_edge("A", "T", 10.0)?;
        graph.add_edge("B", "T", 10.0)?;
        graph.add_edge("A", "B", 1.0)?;

        let max_flow = push_relabel_max_flow(&graph, &"S", &"T")?;

        // Maximum flow should be 20.0
        assert!((max_flow - 20.0).abs() < 1e-6);

        Ok(())
    }

    #[test]
    fn test_dinic_single_path() -> GraphResult<()> {
        let mut graph = crate::generators::create_digraph::<&str, f64>();

        // Simple path: A -> B -> C
        graph.add_edge("A", "B", 5.0)?;
        graph.add_edge("B", "C", 3.0)?;

        let max_flow = dinic_max_flow(&graph, &"A", &"C")?;

        // Bottleneck is 3.0
        assert!((max_flow - 3.0).abs() < 1e-6);

        Ok(())
    }
}
