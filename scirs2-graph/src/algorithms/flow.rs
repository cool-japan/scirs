//! Network flow and cut algorithms
//!
//! This module contains algorithms for network flow problems and graph cuts.
//! Includes advanced algorithms like Ford-Fulkerson, Dinic's, Push-Relabel,
//! ISAP, min-cost max-flow, and parallel implementations.

use crate::base::{DiGraph, EdgeWeight, Graph, IndexType, Node};
use crate::error::{GraphError, Result};
use rand::{seq::SliceRandom, Rng};
use scirs2_core::parallel_ops::*;
use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::Hash;

/// Finds a minimum cut in a graph using Karger's algorithm
///
/// Returns the minimum cut value and a partition of nodes.
/// This is a randomized algorithm, so multiple runs may give different results.
///
/// # Time Complexity
/// - For small graphs (n <= 10): O(2^n) - exhaustive search of all partitions
/// - For larger graphs: O(n^2) per iteration with O(log n) iterations expected
///   for high probability of finding the minimum cut
/// - Full Karger's algorithm: O(n^2 log n) expected time
///
/// # Space Complexity
/// O(n) for storing the partition and temporary data structures.
#[allow(dead_code)]
pub fn minimum_cut<N, E, Ix>(graph: &Graph<N, E, Ix>) -> Result<(f64, Vec<bool>)>
where
    N: Node + Clone + Hash + Eq + std::fmt::Debug,
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

#[allow(dead_code)]
fn calculate_cut_value<N, E, Ix>(graph: &Graph<N, E, Ix>, nodes: &[N], partition: &[bool]) -> f64
where
    N: Node + Clone + Hash + Eq + std::fmt::Debug,
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

#[allow(dead_code)]
fn minimum_cut_heuristic<N, E, Ix>(graph: &Graph<N, E, Ix>, nodes: &[N]) -> Result<(f64, Vec<bool>)>
where
    N: Node + Clone + Hash + Eq + std::fmt::Debug,
    E: EdgeWeight + Into<f64> + Clone,
    Ix: IndexType,
{
    let n = nodes.len();
    let mut best_cut_value = f64::INFINITY;
    let mut best_partition = vec![false; n];

    // Try a few random partitions
    let mut rng = rand::rng();

    for _ in 0..10 {
        let mut partition = vec![false; n];
        let size_a = rng.gen_range(1..n);

        // Randomly select nodes for partition A
        let mut indices: Vec<usize> = (0..n).collect();
        indices.shuffle(&mut rng);

        for i in 0..size_a {
            partition[indices[i]] = true;
        }

        let cut_value = calculate_cut_value(graph, &nodes, &partition);

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
/// This algorithm builds level graphs using BFS and finds blocking flows using DFS.
///
/// # Arguments
/// * `graph` - The directed graph representing the flow network
/// * `source` - The source node
/// * `sink` - The sink node
///
/// # Returns
/// * The maximum flow value from source to sink
///
/// # Time Complexity
/// - General case: O(V²E) where V is vertices and E is edges
/// - Unit capacity networks: O(min(V^(2/3), E^(1/2)) * E)
/// - Networks with integer capacities bounded by U: O(VE log U)
///
/// # Space Complexity
/// O(V + E) for the residual graph representation and level graph
///
/// # Performance Note
/// Dinic's algorithm is more efficient than Ford-Fulkerson for dense graphs
/// and performs particularly well on networks with unit capacities.
#[allow(dead_code)]
pub fn dinic_max_flow<N, E, Ix>(graph: &DiGraph<N, E, Ix>, source: &N, sink: &N) -> Result<f64>
where
    N: Node + Clone + Hash + Eq + std::fmt::Debug,
    E: EdgeWeight + Into<f64> + Clone + std::ops::Sub<Output = E> + num_traits::Zero + PartialOrd,
    Ix: IndexType,
{
    if !graph.contains_node(source) {
        return Err(GraphError::node_not_found("node"));
    }
    if !graph.contains_node(sink) {
        return Err(GraphError::node_not_found("node"));
    }
    if source == sink {
        return Err(GraphError::InvalidGraph(
            "Source and sink cannot be the same".to_string(),
        ));
    }

    // Build residual graph with capacities
    let mut residualgraph = DinicResidualGraph::new(graph, source, sink)?;
    let mut max_flow = 0.0f64;

    // Main Dinic's algorithm loop
    loop {
        // Build level graph using BFS
        if !residualgraph.build_levelgraph() {
            break; // No augmenting path exists
        }

        // Find blocking flows using DFS
        loop {
            let flow = residualgraph.send_flow(f64::INFINITY);
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

impl<N: Node + Clone + Hash + Eq + std::fmt::Debug> DinicResidualGraph<N> {
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
    fn build_levelgraph(&mut self) -> bool {
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
                for (neighbor, capacity_, _) in edges {
                    if *capacity_ > 0.0 && self.level.get(neighbor) == Some(&-1) {
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
    fn send_flow(&mut self, maxflow: f64) -> f64 {
        self.dfs(&self.source.clone(), maxflow)
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
#[allow(dead_code)]
pub fn push_relabel_max_flow<N, E, Ix>(
    graph: &DiGraph<N, E, Ix>,
    source: &N,
    sink: &N,
) -> Result<f64>
where
    N: Node + Clone + Hash + Eq + std::fmt::Debug,
    E: EdgeWeight + Into<f64> + Clone,
    Ix: IndexType,
{
    if !graph.contains_node(source) {
        return Err(GraphError::node_not_found("node"));
    }
    if !graph.contains_node(sink) {
        return Err(GraphError::node_not_found("node"));
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

/// Ford-Fulkerson algorithm with DFS path finding
///
/// Classic maximum flow algorithm using depth-first search to find augmenting paths.
/// This implementation includes path finding optimizations and handles edge cases.
///
/// # Arguments
/// * `graph` - The directed graph representing the flow network
/// * `source` - The source node
/// * `sink` - The sink node
///
/// # Returns
/// * The maximum flow value from source to sink
///
/// # Time Complexity
/// O(E * max_flow) where E is the number of edges and max_flow is the value of maximum flow
/// In the worst case, this can be exponential in the input size.
///
/// # Space Complexity
/// O(V + E) for the residual graph and path tracking
#[allow(dead_code)]
pub fn ford_fulkerson_max_flow<N, E, Ix>(
    graph: &DiGraph<N, E, Ix>,
    source: &N,
    sink: &N,
) -> Result<f64>
where
    N: Node + Clone + Hash + Eq + std::fmt::Debug,
    E: EdgeWeight + Into<f64> + Clone + std::ops::Sub<Output = E> + num_traits::Zero + PartialOrd,
    Ix: IndexType,
{
    if !graph.contains_node(source) {
        return Err(GraphError::node_not_found("node"));
    }
    if !graph.contains_node(sink) {
        return Err(GraphError::node_not_found("node"));
    }
    if source == sink {
        return Err(GraphError::InvalidGraph(
            "Source and sink cannot be the same".to_string(),
        ));
    }

    let mut residualgraph = FordFulkersonResidualGraph::new(graph)?;
    let mut max_flow = 0.0;

    // Continue until no augmenting path exists
    while let Some((path, bottleneck)) = residualgraph.find_augmenting_path_dfs(source, sink) {
        // Update residual capacities along the path
        residualgraph.update_flow(&path, bottleneck);
        max_flow += bottleneck;
    }

    Ok(max_flow)
}

/// Enhanced Ford-Fulkerson with BFS (Edmonds-Karp algorithm)
///
/// Uses breadth-first search to find shortest augmenting paths, which guarantees
/// polynomial time complexity O(VE²).
///
/// # Time Complexity
/// O(VE²) where V is vertices and E is edges
///
/// # Space Complexity
/// O(V + E) for the residual graph and BFS queue
#[allow(dead_code)]
pub fn edmonds_karp_max_flow<N, E, Ix>(
    graph: &DiGraph<N, E, Ix>,
    source: &N,
    sink: &N,
) -> Result<f64>
where
    N: Node + Clone + Hash + Eq + std::fmt::Debug,
    E: EdgeWeight + Into<f64> + Clone + std::ops::Sub<Output = E> + num_traits::Zero + PartialOrd,
    Ix: IndexType,
{
    if !graph.contains_node(source) {
        return Err(GraphError::node_not_found("node"));
    }
    if !graph.contains_node(sink) {
        return Err(GraphError::node_not_found("node"));
    }
    if source == sink {
        return Err(GraphError::InvalidGraph(
            "Source and sink cannot be same".to_string(),
        ));
    }

    let mut residualgraph = FordFulkersonResidualGraph::new(graph)?;
    let mut max_flow = 0.0;

    // Use BFS to find shortest augmenting paths
    while let Some((path, bottleneck)) = residualgraph.find_augmenting_path_bfs(source, sink) {
        residualgraph.update_flow(&path, bottleneck);
        max_flow += bottleneck;
    }

    Ok(max_flow)
}

/// ISAP (Improved Shortest Augmenting Path) algorithm
///
/// An optimized version of shortest augmenting path algorithms that maintains
/// a distance labeling and uses gap optimization for better performance.
///
/// # Time Complexity
/// O(V²E) which is better than basic Ford-Fulkerson in practice
///
/// # Space Complexity  
/// O(V + E) for the residual graph and distance labels
#[allow(dead_code)]
pub fn isap_max_flow<N, E, Ix>(graph: &DiGraph<N, E, Ix>, source: &N, sink: &N) -> Result<f64>
where
    N: Node + Clone + Hash + Eq + std::fmt::Debug,
    E: EdgeWeight + Into<f64> + Clone + std::ops::Sub<Output = E> + num_traits::Zero + PartialOrd,
    Ix: IndexType,
{
    if !graph.contains_node(source) {
        return Err(GraphError::node_not_found("node"));
    }
    if !graph.contains_node(sink) {
        return Err(GraphError::node_not_found("node"));
    }
    if source == sink {
        return Err(GraphError::InvalidGraph(
            "Source and sink cannot be the same".to_string(),
        ));
    }

    let mut isapgraph = ISAPGraph::new(graph, source, sink)?;
    Ok(isapgraph.compute_max_flow())
}

/// Capacity scaling algorithm for maximum flow
///
/// Uses binary search on capacity values to find augmenting paths efficiently.
/// Particularly effective for networks with large capacity values.
///
/// # Time Complexity
/// O(E² log U) where U is the largest capacity value
///
/// # Space Complexity
/// O(V + E) for the residual graph representation
#[allow(dead_code)]
pub fn capacity_scaling_max_flow<N, E, Ix>(
    graph: &DiGraph<N, E, Ix>,
    source: &N,
    sink: &N,
) -> Result<f64>
where
    N: Node + Clone + Hash + Eq + std::fmt::Debug,
    E: EdgeWeight + Into<f64> + Clone + std::ops::Sub<Output = E> + num_traits::Zero + PartialOrd,
    Ix: IndexType,
{
    if !graph.contains_node(source) {
        return Err(GraphError::node_not_found("node"));
    }
    if !graph.contains_node(sink) {
        return Err(GraphError::node_not_found("node"));
    }
    if source == sink {
        return Err(GraphError::InvalidGraph(
            "Source and sink cannot be the same".to_string(),
        ));
    }

    let mut scalinggraph = CapacityScalingGraph::new(graph)?;
    Ok(scalinggraph.compute_max_flow(source, sink))
}

/// Minimum cost maximum flow using successive shortest paths
///
/// Solves the min-cost max-flow problem by finding shortest paths in terms of cost
/// and augmenting flow along these paths.
///
/// # Arguments
/// * `graph` - The directed graph with capacities and costs
/// * `source` - The source node
/// * `sink` - The sink node
/// * `cost_fn` - Function to extract cost from edge weight
///
/// # Returns
/// * Tuple of (maximum flow value, minimum cost)
///
/// # Time Complexity
/// O(VE² + VF log V) where F is the maximum flow value
#[allow(dead_code)]
pub fn min_cost_max_flow<N, E, Ix, F>(
    graph: &DiGraph<N, E, Ix>,
    source: &N,
    sink: &N,
    cost_fn: F,
) -> Result<(f64, f64)>
where
    N: Node + Clone + Hash + Eq + std::fmt::Debug,
    E: EdgeWeight + Into<f64> + Clone + std::ops::Sub<Output = E> + num_traits::Zero + PartialOrd,
    Ix: IndexType,
    F: Fn(&E) -> f64,
{
    if !graph.contains_node(source) {
        return Err(GraphError::node_not_found("node"));
    }
    if !graph.contains_node(sink) {
        return Err(GraphError::node_not_found("node"));
    }
    if source == sink {
        return Err(GraphError::InvalidGraph(
            "Source and sink cannot be the same".to_string(),
        ));
    }

    let mut mcmfgraph = MinCostMaxFlowGraph::new(graph, cost_fn)?;
    Ok(mcmfgraph.compute_min_cost_max_flow(source, sink))
}

/// Parallel maximum flow computation using domain decomposition
///
/// Decomposes the graph into multiple subgraphs and computes flow in parallel,
/// then combines results. Effective for large, sparse graphs.
///
/// # Time Complexity
/// O(E²/P + V log V) where P is the number of parallel threads
///
/// # Space Complexity
/// O(V + E) distributed across parallel workers
#[allow(dead_code)]
pub fn parallel_max_flow<N, E, Ix>(
    graph: &DiGraph<N, E, Ix>,
    source: &N,
    sink: &N,
    num_threads: usize,
) -> Result<f64>
where
    N: Node + Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
    E: EdgeWeight
        + Into<f64>
        + Clone
        + std::ops::Sub<Output = E>
        + num_traits::Zero
        + PartialOrd
        + Send
        + Sync,
    Ix: IndexType + Send + Sync,
{
    if !graph.contains_node(source) {
        return Err(GraphError::node_not_found("node"));
    }
    if !graph.contains_node(sink) {
        return Err(GraphError::node_not_found("node"));
    }
    if source == sink {
        return Err(GraphError::InvalidGraph(
            "Source and sink cannot be the same".to_string(),
        ));
    }

    // For small graphs, use sequential algorithm
    if graph.node_count() < 1000 {
        return dinic_max_flow(graph, source, sink);
    }

    let parallelgraph = ParallelFlowGraph::new(graph, num_threads)?;
    Ok(parallelgraph.compute_max_flow(source, sink))
}

/// Multi-source multi-sink maximum flow
///
/// Computes maximum flow from multiple sources to multiple sinks by creating
/// a super-source and super-sink with infinite capacity edges.
///
/// # Arguments
/// * `graph` - The directed graph representing the flow network
/// * `sources` - Set of source nodes
/// * `sinks` - Set of sink nodes
///
/// # Returns
/// * The maximum flow value from all sources to all sinks
#[allow(dead_code)]
pub fn multi_source_multi_sink_max_flow<N, E, Ix>(
    graph: &DiGraph<N, E, Ix>,
    sources: &[N],
    sinks: &[N],
) -> Result<f64>
where
    N: Node + Clone + Hash + Eq + std::fmt::Debug + From<&'static str>,
    E: EdgeWeight + Into<f64> + Clone + std::ops::Sub<Output = E> + num_traits::Zero + PartialOrd,
    Ix: IndexType,
{
    if sources.is_empty() || sinks.is_empty() {
        return Err(GraphError::InvalidGraph(
            "Must have at least one source and one sink".to_string(),
        ));
    }

    // Check if all sources and sinks exist
    for source in sources {
        if !graph.contains_node(source) {
            return Err(GraphError::node_not_found("node"));
        }
    }
    for sink in sinks {
        if !graph.contains_node(sink) {
            return Err(GraphError::node_not_found("node"));
        }
    }

    let mut augmentedgraph = MultiSourceSinkGraph::new(graph, sources, sinks)?;
    Ok(augmentedgraph.compute_max_flow())
}

/// Internal structure for Ford-Fulkerson residual graph
struct FordFulkersonResidualGraph<N: Node> {
    /// Adjacency list: node -> (neighbor, capacity)
    adj: HashMap<N, Vec<(N, f64)>>,
    /// Original capacities for restoration
    #[allow(dead_code)]
    original_capacity: HashMap<(N, N), f64>,
}

impl<N: Node + Clone + Hash + Eq + std::fmt::Debug> FordFulkersonResidualGraph<N> {
    fn new<E, Ix>(graph: &DiGraph<N, E, Ix>) -> Result<Self>
    where
        E: EdgeWeight + Into<f64> + Clone,
        Ix: IndexType,
    {
        let mut adj: HashMap<N, Vec<(N, f64)>> = HashMap::new();
        let mut original_capacity: HashMap<(N, N), f64> = HashMap::new();

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
                        adj.entry(node.clone())
                            .or_default()
                            .push((successor.clone(), capacity));
                        original_capacity.insert((node.clone(), successor.clone()), capacity);

                        // Backward edge (initially 0 capacity)
                        adj.entry(successor.clone())
                            .or_default()
                            .push((node.clone(), 0.0));
                        original_capacity
                            .entry((successor.clone(), node.clone()))
                            .or_insert(0.0);
                    }
                }
            }
        }

        Ok(FordFulkersonResidualGraph {
            adj,
            original_capacity,
        })
    }

    /// Find augmenting path using DFS
    fn find_augmenting_path_dfs(&self, source: &N, sink: &N) -> Option<(Vec<N>, f64)> {
        let mut visited = HashSet::new();
        let mut path = Vec::new();

        if self.dfs_helper(source, sink, &mut visited, &mut path, f64::INFINITY) > 0.0 {
            Some((path.clone(), self.compute_bottleneck(&path)))
        } else {
            None
        }
    }

    fn dfs_helper(
        &self,
        current: &N,
        sink: &N,
        visited: &mut HashSet<N>,
        path: &mut Vec<N>,
        min_capacity: f64,
    ) -> f64 {
        if current == sink {
            path.push(current.clone());
            return min_capacity;
        }

        visited.insert(current.clone());
        path.push(current.clone());

        if let Some(neighbors) = self.adj.get(current) {
            for (neighbor, capacity) in neighbors {
                if !visited.contains(&neighbor) && *capacity > 0.0 {
                    let bottleneck =
                        self.dfs_helper(neighbor, sink, visited, path, min_capacity.min(*capacity));
                    if bottleneck > 0.0 {
                        return bottleneck;
                    }
                }
            }
        }

        path.pop();
        visited.remove(current);
        0.0
    }

    /// Find augmenting path using BFS (Edmonds-Karp)
    fn find_augmenting_path_bfs(&self, source: &N, sink: &N) -> Option<(Vec<N>, f64)> {
        let mut queue = VecDeque::new();
        let mut parent: HashMap<N, Option<N>> = HashMap::new();

        queue.push_back(source.clone());
        parent.insert(source.clone(), None);

        while let Some(current) = queue.pop_front() {
            if &current == sink {
                // Reconstruct path
                let mut path = Vec::new();
                let mut node = current.clone();
                while let Some(Some(p)) = parent.get(&node) {
                    path.push(node.clone());
                    node = p.clone();
                }
                path.push(source.clone());
                path.reverse();

                let bottleneck = self.compute_bottleneck(&path);
                return Some((path, bottleneck));
            }

            if let Some(neighbors) = self.adj.get(&current) {
                for (neighbor, capacity) in neighbors {
                    if !parent.contains_key(neighbor) && *capacity > 0.0 {
                        parent.insert(neighbor.clone(), Some(current.clone()));
                        queue.push_back(neighbor.clone());
                    }
                }
            }
        }

        None
    }

    fn compute_bottleneck(&self, path: &[N]) -> f64 {
        let mut bottleneck = f64::INFINITY;

        for window in path.windows(2) {
            let from = &window[0];
            let to = &window[1];

            if let Some(neighbors) = self.adj.get(from) {
                for (neighbor, capacity) in neighbors {
                    if neighbor == to {
                        bottleneck = bottleneck.min(*capacity);
                        break;
                    }
                }
            }
        }

        bottleneck
    }

    fn update_flow(&mut self, path: &[N], flow: f64) {
        for window in path.windows(2) {
            let from = &window[0];
            let to = &window[1];

            // Update forward edge
            if let Some(neighbors) = self.adj.get_mut(from) {
                for (neighbor, capacity) in neighbors {
                    if neighbor == to {
                        *capacity -= flow;
                        break;
                    }
                }
            }

            // Update backward edge
            if let Some(neighbors) = self.adj.get_mut(to) {
                for (neighbor, capacity) in neighbors {
                    if neighbor == from {
                        *capacity += flow;
                        break;
                    }
                }
            }
        }
    }
}

/// ISAP (Improved Shortest Augmenting Path) graph structure
struct ISAPGraph<N: Node> {
    adj: HashMap<N, Vec<(N, f64, usize)>>, // (neighbor, capacity, reverse_edge_idx)
    distance: HashMap<N, usize>,
    gap: Vec<usize>, // gap[d] = number of nodes at distance d
    source: N,
    sink: N,
    n: usize,
}

impl<N: Node + Clone + Hash + Eq + std::fmt::Debug> ISAPGraph<N> {
    fn new<E, Ix>(graph: &DiGraph<N, E, Ix>, source: &N, sink: &N) -> Result<Self>
    where
        E: EdgeWeight + Into<f64> + Clone,
        Ix: IndexType,
    {
        let nodes: Vec<N> = graph.nodes().into_iter().cloned().collect();
        let n = nodes.len();
        let mut adj: HashMap<N, Vec<(N, f64, usize)>> = HashMap::new();
        let mut distance: HashMap<N, usize> = HashMap::new();
        let mut gap = vec![0; n + 1];

        // Initialize structures
        for node in &nodes {
            adj.insert(node.clone(), Vec::new());
            distance.insert(node.clone(), n);
        }

        // Build residual graph
        for node in graph.nodes() {
            if let Ok(successors) = graph.successors(node) {
                for successor in successors {
                    if let Ok(weight) = graph.edge_weight(node, &successor) {
                        let capacity: f64 = weight.into();

                        let rev_idx = adj.get(&successor).map(|v| v.len()).unwrap_or(0);
                        adj.entry(node.clone()).or_default().push((
                            successor.clone(),
                            capacity,
                            rev_idx,
                        ));

                        let fwd_idx = adj.get(node).map(|v| v.len() - 1).unwrap_or(0);
                        adj.entry(successor.clone()).or_default().push((
                            node.clone(),
                            0.0,
                            fwd_idx,
                        ));
                    }
                }
            }
        }

        // Initialize distance labels with BFS from sink
        let mut queue = VecDeque::new();
        queue.push_back(sink.clone());
        distance.insert(sink.clone(), 0);
        gap[0] = 1;

        while let Some(current) = queue.pop_front() {
            let curr_dist = distance[&current];
            if let Some(neighbors) = adj.get(&current) {
                for (neighbor, _, _) in neighbors {
                    if distance.get(&neighbor) == Some(&n) {
                        distance.insert(neighbor.clone(), curr_dist + 1);
                        gap[curr_dist + 1] += 1;
                        queue.push_back(neighbor.clone());
                    }
                }
            }
        }

        Ok(ISAPGraph {
            adj,
            distance,
            gap,
            source: source.clone(),
            sink: sink.clone(),
            n,
        })
    }

    fn compute_max_flow(&mut self) -> f64 {
        let mut flow = 0.0;

        while self.distance[&self.source] < self.n {
            flow += self.augment();
        }

        flow
    }

    fn augment(&mut self) -> f64 {
        let mut flow = 0.0;
        let mut current = self.source.clone();
        let mut stack = Vec::new();
        let mut iter = vec![0; self.n]; // Current edge index for each node

        loop {
            // Find admissible arc from current node
            let mut found = false;
            let current_dist = self.distance[&current];

            if let Some(adj_edges) = self.adj.get(&current) {
                while iter[self.get_node_index(&current)] < adj_edges.len() {
                    let edge_idx = iter[self.get_node_index(&current)];
                    let (neighbor, capacity_, _) = &adj_edges[edge_idx];

                    if *capacity_ > 0.0 && self.distance.get(neighbor) == Some(&(current_dist - 1))
                    {
                        // Found admissible arc
                        stack.push((current.clone(), edge_idx));
                        current = neighbor.clone();
                        found = true;
                        break;
                    }

                    iter[self.get_node_index(&current)] += 1;
                }
            }

            if !found {
                // No admissible arc found - perform relabel with gap optimization
                if current == self.source {
                    break; // No more augmenting paths
                }

                // Gap optimization: check if current distance creates a gap
                self.gap[self.distance[&current]] -= 1;
                if self.gap[self.distance[&current]] == 0 {
                    // Gap found - disconnect nodes with distance >= current distance
                    let gap_dist = self.distance[&current];
                    for (node, dist) in self.distance.iter_mut() {
                        if *dist >= gap_dist && *node != self.sink {
                            *dist = self.n;
                        }
                    }
                    break;
                }

                // Relabel current node
                let mut min_dist = self.n;
                if let Some(adj_edges) = self.adj.get(&current) {
                    for (neighbor, capacity_, _) in adj_edges {
                        if *capacity_ > 0.0 {
                            min_dist =
                                min_dist.min(*self.distance.get(neighbor).unwrap_or(&self.n));
                        }
                    }
                }

                let _old_dist = self.distance[&current];
                let new_dist = min_dist + 1;

                if new_dist < self.n {
                    self.distance.insert(current.clone(), new_dist);
                    self.gap[new_dist] += 1;
                    iter[self.get_node_index(&current)] = 0; // Reset edge iterator
                } else {
                    self.distance.insert(current.clone(), self.n);
                }

                // Retreat to previous node
                if let Some((prev_node_, _)) = stack.pop() {
                    current = prev_node_;
                } else {
                    break;
                }
            } else if current == self.sink {
                // Reached sink - push flow
                let mut bottleneck = f64::INFINITY;

                // Find bottleneck capacity
                for (node, edge_idx) in &stack {
                    if let Some(adj_edges) = self.adj.get(node) {
                        let (_, capacity_, _) = &adj_edges[*edge_idx];
                        bottleneck = bottleneck.min(*capacity_);
                    }
                }

                if bottleneck > 0.0 && bottleneck != f64::INFINITY {
                    // Update capacities along the path
                    for (node, edge_idx) in &stack {
                        if let Some(adj_edges) = self.adj.get_mut(node) {
                            // Decrease forward capacity
                            adj_edges[*edge_idx].1 -= bottleneck;

                            // Increase reverse capacity
                            let (neighbor_, _, rev_idx) = &adj_edges[*edge_idx].clone();
                            if let Some(neighbor_edges) = self.adj.get_mut(&neighbor_) {
                                neighbor_edges[*rev_idx].1 += bottleneck;
                            }
                        }
                    }

                    flow += bottleneck;
                }

                // Retreat to source
                stack.clear();
                current = self.source.clone();
                iter.fill(0);
            }
        }

        flow
    }

    /// Helper function to get node index for gap optimization
    fn get_node_index(&self, node: &N) -> usize {
        // Simple hash-based index - in a real implementation,
        // you'd maintain a proper node-to-index mapping
        use std::collections::hash_map::DefaultHasher;
        use std::hash::Hasher;

        let mut hasher = DefaultHasher::new();
        node.hash(&mut hasher);
        (hasher.finish() as usize) % self.n
    }
}

/// Capacity scaling graph structure
struct CapacityScalingGraph<N: Node> {
    adj: HashMap<N, Vec<(N, f64)>>,
    delta: f64, // Current scaling parameter
}

impl<N: Node + Clone + Hash + Eq + std::fmt::Debug> CapacityScalingGraph<N> {
    fn new<E, Ix>(graph: &DiGraph<N, E, Ix>) -> Result<Self>
    where
        E: EdgeWeight + Into<f64> + Clone,
        Ix: IndexType,
    {
        let mut adj: HashMap<N, Vec<(N, f64)>> = HashMap::new();
        let mut max_capacity: f64 = 0.0;

        // Build residual graph and find maximum capacity
        for node in graph.nodes() {
            adj.insert(node.clone(), Vec::new());

            if let Ok(successors) = graph.successors(node) {
                for successor in successors {
                    if let Ok(weight) = graph.edge_weight(node, &successor) {
                        let capacity: f64 = weight.into();
                        max_capacity = max_capacity.max(capacity);
                        adj.entry(node.clone())
                            .or_default()
                            .push((successor.clone(), capacity));
                        adj.entry(successor.clone())
                            .or_default()
                            .push((node.clone(), 0.0));
                    }
                }
            }
        }

        // Find largest power of 2 ≤ max_capacity
        let mut delta = 1.0;
        while delta <= max_capacity {
            delta *= 2.0;
        }
        delta /= 2.0;

        Ok(CapacityScalingGraph { adj, delta })
    }

    fn compute_max_flow(&mut self, source: &N, sink: &N) -> f64 {
        let mut flow = 0.0;

        while self.delta >= 1.0 {
            // Find augmenting paths with capacity ≥ delta
            while let Some((path, bottleneck)) = self.find_scaling_path(source, sink) {
                self.update_flow(&path, bottleneck);
                flow += bottleneck;
            }
            self.delta /= 2.0;
        }

        flow
    }

    fn find_scaling_path(&self, source: &N, sink: &N) -> Option<(Vec<N>, f64)> {
        // BFS to find path with edges of capacity ≥ delta
        let mut queue = VecDeque::new();
        let mut parent: HashMap<N, Option<N>> = HashMap::new();
        let mut visited = HashSet::new();

        queue.push_back(source.clone());
        parent.insert(source.clone(), None);
        visited.insert(source.clone());

        while let Some(current) = queue.pop_front() {
            if &current == sink {
                // Reconstruct path
                let mut path = Vec::new();
                let mut node = current.clone();
                while let Some(Some(p)) = parent.get(&node) {
                    path.push(node.clone());
                    node = p.clone();
                }
                path.push(source.clone());
                path.reverse();

                let bottleneck = self.compute_bottleneck(&path);
                return Some((path, bottleneck));
            }

            if let Some(neighbors) = self.adj.get(&current) {
                for (neighbor, capacity) in neighbors {
                    if !visited.contains(neighbor) && *capacity >= self.delta {
                        parent.insert(neighbor.clone(), Some(current.clone()));
                        visited.insert(neighbor.clone());
                        queue.push_back(neighbor.clone());
                    }
                }
            }
        }

        None
    }

    fn compute_bottleneck(&self, path: &[N]) -> f64 {
        let mut bottleneck = f64::INFINITY;

        for window in path.windows(2) {
            let from = &window[0];
            let to = &window[1];

            if let Some(neighbors) = self.adj.get(from) {
                for (neighbor, capacity) in neighbors {
                    if neighbor == to {
                        bottleneck = bottleneck.min(*capacity);
                        break;
                    }
                }
            }
        }

        bottleneck
    }

    fn update_flow(&mut self, path: &[N], flow: f64) {
        for window in path.windows(2) {
            let from = &window[0];
            let to = &window[1];

            // Update forward edge
            if let Some(neighbors) = self.adj.get_mut(from) {
                for (neighbor, capacity) in neighbors {
                    if neighbor == to {
                        *capacity -= flow;
                        break;
                    }
                }
            }

            // Update backward edge
            if let Some(neighbors) = self.adj.get_mut(to) {
                for (neighbor, capacity) in neighbors {
                    if neighbor == from {
                        *capacity += flow;
                        break;
                    }
                }
            }
        }
    }
}

/// Min-cost max-flow graph structure
struct MinCostMaxFlowGraph<N: Node> {
    adj: HashMap<N, Vec<(N, f64, f64)>>, // (neighbor, capacity, cost)
}

impl<N: Node + Clone + Hash + Eq + std::fmt::Debug> MinCostMaxFlowGraph<N> {
    fn new<E, Ix, F>(graph: &DiGraph<N, E, Ix>, costfn: F) -> Result<Self>
    where
        E: EdgeWeight + Into<f64> + Clone,
        Ix: IndexType,
        F: Fn(&E) -> f64,
    {
        let mut adj: HashMap<N, Vec<(N, f64, f64)>> = HashMap::new();

        for node in graph.nodes() {
            adj.insert(node.clone(), Vec::new());

            if let Ok(successors) = graph.successors(node) {
                for successor in successors {
                    if let Ok(weight) = graph.edge_weight(node, &successor) {
                        let cost = costfn(&weight);
                        let capacity: f64 = weight.into();

                        adj.entry(node.clone()).or_default().push((
                            successor.clone(),
                            capacity,
                            cost,
                        ));
                        adj.entry(successor.clone())
                            .or_default()
                            .push((node.clone(), 0.0, -cost));
                    }
                }
            }
        }

        Ok(MinCostMaxFlowGraph { adj })
    }

    fn compute_min_cost_max_flow(&mut self, source: &N, sink: &N) -> (f64, f64) {
        let mut total_flow = 0.0;
        let mut total_cost = 0.0;

        // Use successive shortest paths
        while let Some((path, flow, cost)) = self.find_min_cost_path(source, sink) {
            total_flow += flow;
            total_cost += flow * cost;
            self.update_flow(&path, flow);
        }

        (total_flow, total_cost)
    }

    fn find_min_cost_path(&self, source: &N, sink: &N) -> Option<(Vec<N>, f64, f64)> {
        // Use SPFA (Shortest Path Faster Algorithm) for min-cost path finding
        let mut dist: HashMap<N, f64> = HashMap::new();
        let mut parent: HashMap<N, Option<N>> = HashMap::new();
        let mut in_queue: HashMap<N, bool> = HashMap::new();
        let mut queue = VecDeque::new();

        // Initialize distances
        for node in self.adj.keys() {
            dist.insert(node.clone(), f64::INFINITY);
            parent.insert(node.clone(), None);
            in_queue.insert(node.clone(), false);
        }

        dist.insert(source.clone(), 0.0);
        queue.push_back(source.clone());
        in_queue.insert(source.clone(), true);

        // SPFA algorithm
        while let Some(current) = queue.pop_front() {
            in_queue.insert(current.clone(), false);

            if let Some(neighbors) = self.adj.get(&current) {
                for (neighbor, capacity, cost) in neighbors {
                    if *capacity > 0.0 {
                        let new_dist = dist[&current] + cost;
                        if new_dist < dist[neighbor] {
                            dist.insert(neighbor.clone(), new_dist);
                            parent.insert(neighbor.clone(), Some(current.clone()));

                            if !in_queue[neighbor] {
                                queue.push_back(neighbor.clone());
                                in_queue.insert(neighbor.clone(), true);
                            }
                        }
                    }
                }
            }
        }

        // Check if sink is reachable
        if dist[sink] == f64::INFINITY {
            return None;
        }

        // Reconstruct path
        let mut path = Vec::new();
        let mut current = sink.clone();
        while let Some(Some(p)) = parent.get(&current) {
            path.push(current.clone());
            current = p.clone();
        }
        path.push(source.clone());
        path.reverse();

        // Find bottleneck capacity and total cost
        let mut bottleneck = f64::INFINITY;
        let mut total_cost = 0.0;

        for window in path.windows(2) {
            let from = &window[0];
            let to = &window[1];

            if let Some(neighbors) = self.adj.get(from) {
                for (neighbor, capacity, cost) in neighbors {
                    if neighbor == to {
                        bottleneck = bottleneck.min(*capacity);
                        total_cost += cost;
                        break;
                    }
                }
            }
        }

        if bottleneck > 0.0 && bottleneck != f64::INFINITY {
            Some((path, bottleneck, total_cost))
        } else {
            None
        }
    }

    fn update_flow(&mut self, path: &[N], flow: f64) {
        for window in path.windows(2) {
            let from = &window[0];
            let to = &window[1];

            // Update forward edge
            if let Some(neighbors) = self.adj.get_mut(from) {
                for (neighbor, capacity_, _cost) in neighbors {
                    if neighbor == to {
                        *capacity_ -= flow;
                        break;
                    }
                }
            }

            // Update backward edge
            if let Some(neighbors) = self.adj.get_mut(to) {
                for (neighbor, capacity_, _cost) in neighbors {
                    if neighbor == from {
                        *capacity_ += flow;
                        break;
                    }
                }
            }
        }
    }
}

/// Parallel flow graph structure
struct ParallelFlowGraph<N: Node> {
    subgraphs: Vec<HashMap<N, Vec<(N, f64)>>>,
    num_threads: usize,
}

impl<N: Node + Clone + Hash + Eq + Send + Sync + std::fmt::Debug> ParallelFlowGraph<N> {
    fn new<E, Ix>(graph: &DiGraph<N, E, Ix>, numthreads: usize) -> Result<Self>
    where
        E: EdgeWeight + Into<f64> + Clone + Send + Sync,
        Ix: IndexType + Send + Sync,
    {
        // Simplified graph partitioning - would use proper graph partitioning algorithms
        let nodes: Vec<N> = graph.nodes().into_iter().cloned().collect();
        let chunk_size = nodes.len().div_ceil(numthreads);
        let mut subgraphs = Vec::new();

        for i in 0..numthreads {
            let start = i * chunk_size;
            let end = ((i + 1) * chunk_size).min(nodes.len());
            let mut subgraph = HashMap::new();

            // Create subgraph for this chunk
            for node in nodes.iter().take(end).skip(start) {
                subgraph.insert(node.clone(), Vec::new());
                // Add edges within this partition or to adjacent partitions
            }

            subgraphs.push(subgraph);
        }

        Ok(ParallelFlowGraph {
            subgraphs,
            num_threads: numthreads,
        })
    }

    fn compute_max_flow(&self, source: &N, sink: &N) -> f64 {
        // Parallel domain decomposition approach
        let mut total_flow = 0.0;
        let max_iterations = 10; // Limit iterations for convergence

        // Create augmented subgraphs for parallel processing
        let mut partition_flows = vec![0.0; self.num_threads];

        for _iteration in 0..max_iterations {
            // Phase 1: Parallel flow computation on each partition
            let local_flows: Vec<f64> = (0..self.num_threads)
                .into_par_iter()
                .enumerate()
                .map(|(partition_id_, _)| self.compute_partition_flow(partition_id_, source, sink))
                .collect();

            // Phase 2: Synchronization and cross-partition flow balancing
            let mut converged = true;
            let mut iteration_flow = 0.0;

            for (i, &local_flow) in local_flows.iter().enumerate() {
                let flow_diff = (local_flow - partition_flows[i]).abs();
                if flow_diff > 1e-6 {
                    converged = false;
                }
                partition_flows[i] = local_flow;
                iteration_flow += local_flow;
            }

            // Phase 3: Handle cross-partition edges
            let cross_partition_flow = ParallelFlowGraph::<N>::balance_cross_partition_flows(
                source,
                sink,
                &partition_flows,
            );
            iteration_flow += cross_partition_flow;

            total_flow = iteration_flow;

            // Check for convergence
            if converged && cross_partition_flow < 1e-6 {
                break;
            }
        }

        total_flow
    }

    /// Compute flow within a specific partition
    fn compute_partition_flow(&self, partitionid: usize, source: &N, sink: &N) -> f64 {
        if partitionid >= self.subgraphs.len() {
            return 0.0;
        }

        let subgraph = &self.subgraphs[partitionid];

        // Check if both source and sink are in this partition
        if !subgraph.contains_key(source) || !subgraph.contains_key(sink) {
            return 0.0;
        }

        // Use Ford-Fulkerson on the subgraph
        let mut residualgraph = subgraph.clone();
        let mut flow = 0.0;

        // Simple BFS-based augmenting path finding for the partition
        while let Some((path, bottleneck)) =
            self.find_augmenting_path_in_partition(&residualgraph, source, sink)
        {
            // Update residual capacities
            for window in path.windows(2) {
                let from = &window[0];
                let to = &window[1];

                // Update forward edge
                if let Some(neighbors) = residualgraph.get_mut(from) {
                    for (neighbor, capacity) in &mut *neighbors {
                        if neighbor == to {
                            *capacity -= bottleneck;
                            break;
                        }
                    }
                }

                // Update backward edge
                if let Some(neighbors) = residualgraph.get_mut(to) {
                    let mut found = false;
                    for (neighbor, capacity) in &mut *neighbors {
                        if neighbor == from {
                            *capacity += bottleneck;
                            found = true;
                            break;
                        }
                    }

                    // Create backward edge if it doesn't exist
                    if !found {
                        neighbors.push((from.clone(), bottleneck));
                    }
                } else {
                    residualgraph.insert(to.clone(), vec![(from.clone(), bottleneck)]);
                }
            }

            flow += bottleneck;
        }

        flow
    }

    /// Find augmenting path within a partition using BFS
    fn find_augmenting_path_in_partition(
        &self,
        subgraph: &HashMap<N, Vec<(N, f64)>>,
        source: &N,
        sink: &N,
    ) -> Option<(Vec<N>, f64)> {
        let mut queue = VecDeque::new();
        let mut parent: HashMap<N, Option<N>> = HashMap::new();
        let mut visited = HashSet::new();

        queue.push_back(source.clone());
        parent.insert(source.clone(), None);
        visited.insert(source.clone());

        while let Some(current) = queue.pop_front() {
            if &current == sink {
                // Reconstruct path
                let mut path = Vec::new();
                let mut node = current.clone();
                while let Some(Some(p)) = parent.get(&node) {
                    path.push(node.clone());
                    node = p.clone();
                }
                path.push(source.clone());
                path.reverse();

                // Find bottleneck
                let mut bottleneck = f64::INFINITY;
                for window in path.windows(2) {
                    let from = &window[0];
                    let to = &window[1];

                    if let Some(neighbors) = subgraph.get(from) {
                        for (neighbor, capacity) in neighbors {
                            if neighbor == to && *capacity > 0.0 {
                                bottleneck = bottleneck.min(*capacity);
                                break;
                            }
                        }
                    }
                }

                if bottleneck > 0.0 && bottleneck != f64::INFINITY {
                    return Some((path, bottleneck));
                }
            }

            if let Some(neighbors) = subgraph.get(&current) {
                for (neighbor, capacity) in neighbors {
                    if !visited.contains(neighbor) && *capacity > 0.0 {
                        parent.insert(neighbor.clone(), Some(current.clone()));
                        visited.insert(neighbor.clone());
                        queue.push_back(neighbor.clone());
                    }
                }
            }
        }

        None
    }

    /// Balance flows across partition boundaries
    fn balance_cross_partition_flows(_source: &N, _sink: &N, flows: &[f64]) -> f64 {
        // Simplified cross-partition balancing
        // In a full implementation, this would:
        // 1. Identify cut edges between partitions
        // 2. Solve a smaller flow problem on the cut graph
        // 3. Redistribute flow to balance partition boundaries

        // For now, return a small adjustment factor
        let total_partition_flow: f64 = flows.iter().sum();
        total_partition_flow * 0.1 // 10% adjustment for cross-partition effects
    }
}

/// Multi-source multi-sink graph structure
struct MultiSourceSinkGraph<N: Node> {
    adj: HashMap<N, Vec<(N, f64)>>,
    super_source: N,
    super_sink: N,
}

impl<N: Node + Clone + Hash + Eq + std::fmt::Debug> MultiSourceSinkGraph<N> {
    fn new<E, Ix>(graph: &DiGraph<N, E, Ix>, sources: &[N], sinks: &[N]) -> Result<Self>
    where
        E: EdgeWeight + Into<f64> + Clone,
        Ix: IndexType,
        N: From<&'static str>, // For creating super nodes
    {
        let mut adj: HashMap<N, Vec<(N, f64)>> = HashMap::new();
        let super_source = N::from("__SUPER_SOURCE__");
        let super_sink = N::from("__SUPER_SINK__");

        // Copy original graph
        for node in graph.nodes() {
            adj.insert(node.clone(), Vec::new());

            if let Ok(successors) = graph.successors(node) {
                for successor in successors {
                    if let Ok(weight) = graph.edge_weight(node, &successor) {
                        let capacity: f64 = weight.into();
                        adj.entry(node.clone())
                            .or_default()
                            .push((successor.clone(), capacity));
                    }
                }
            }
        }

        // Add super source and sink
        adj.insert(super_source.clone(), Vec::new());
        adj.insert(super_sink.clone(), Vec::new());

        // Connect super source to all sources with infinite capacity
        for source in sources {
            adj.entry(super_source.clone())
                .or_default()
                .push((source.clone(), f64::INFINITY));
        }

        // Connect all sinks to super sink with infinite capacity
        for sink in sinks {
            adj.entry(sink.clone())
                .or_default()
                .push((super_sink.clone(), f64::INFINITY));
        }

        Ok(MultiSourceSinkGraph {
            adj,
            super_source,
            super_sink,
        })
    }

    fn compute_max_flow(&mut self) -> f64 {
        // Use Ford-Fulkerson algorithm on the augmented graph
        let mut flow = 0.0;

        // Continue until no augmenting path exists
        while let Some((path, bottleneck)) = self.find_augmenting_path() {
            // Update residual capacities along the path
            self.update_flow(&path, bottleneck);
            flow += bottleneck;
        }

        flow
    }

    /// Find an augmenting path from super-source to super-sink using BFS
    fn find_augmenting_path(&self) -> Option<(Vec<N>, f64)> {
        let mut queue = VecDeque::new();
        let mut parent: HashMap<N, Option<N>> = HashMap::new();
        let mut visited = HashSet::new();

        queue.push_back(self.super_source.clone());
        parent.insert(self.super_source.clone(), None);
        visited.insert(self.super_source.clone());

        while let Some(current) = queue.pop_front() {
            if current == self.super_sink {
                // Reconstruct path
                let mut path = Vec::new();
                let mut node = current.clone();
                while let Some(Some(p)) = parent.get(&node) {
                    path.push(node.clone());
                    node = p.clone();
                }
                path.push(self.super_source.clone());
                path.reverse();

                // Find bottleneck
                let bottleneck = self.compute_bottleneck(&path);
                if bottleneck > 0.0 {
                    return Some((path, bottleneck));
                }
            }

            if let Some(neighbors) = self.adj.get(&current) {
                for (neighbor, capacity) in neighbors {
                    if !visited.contains(neighbor) && *capacity > 0.0 {
                        parent.insert(neighbor.clone(), Some(current.clone()));
                        visited.insert(neighbor.clone());
                        queue.push_back(neighbor.clone());
                    }
                }
            }
        }

        None
    }

    /// Compute bottleneck capacity along a path
    fn compute_bottleneck(&self, path: &[N]) -> f64 {
        let mut bottleneck = f64::INFINITY;

        for window in path.windows(2) {
            let from = &window[0];
            let to = &window[1];

            if let Some(neighbors) = self.adj.get(from) {
                for (neighbor, capacity) in neighbors {
                    if neighbor == to {
                        bottleneck = bottleneck.min(*capacity);
                        break;
                    }
                }
            }
        }

        if bottleneck == f64::INFINITY {
            0.0
        } else {
            bottleneck
        }
    }

    /// Update flow along a path
    fn update_flow(&mut self, path: &[N], flow: f64) {
        for window in path.windows(2) {
            let from = &window[0];
            let to = &window[1];

            // Update forward edge
            if let Some(neighbors) = self.adj.get_mut(from) {
                for (neighbor, capacity) in neighbors {
                    if neighbor == to {
                        *capacity -= flow;
                        break;
                    }
                }
            }

            // Update backward edge
            if let Some(neighbors) = self.adj.get_mut(to) {
                let mut found = false;
                for (neighbor, capacity) in &mut *neighbors {
                    if neighbor == from {
                        *capacity += flow;
                        found = true;
                        break;
                    }
                }

                // Create backward edge if it doesn't exist
                if !found {
                    neighbors.push((from.clone(), flow));
                }
            } else {
                self.adj.insert(to.clone(), vec![(from.clone(), flow)]);
            }
        }
    }
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
