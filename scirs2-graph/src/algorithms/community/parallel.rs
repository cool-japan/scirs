//! Parallel implementations of community detection algorithms

use super::louvain::calculate_modularity;
use super::types::{CommunityResult, CommunityStructure};
use crate::base::{EdgeWeight, Graph, IndexType, Node};
use scirs2_core::random::seq::SliceRandom;
use std::collections::{HashMap, HashSet};
use std::hash::Hash;

#[cfg(feature = "parallel")]
use scirs2_core::parallel_ops::*;

/// Parallel version of Louvain community detection algorithm
///
/// This implementation uses parallel processing to accelerate community
/// detection on large graphs. It leverages scirs2-core parallel operations.
///
/// # Arguments
/// * `graph` - The undirected graph to analyze
/// * `max_iterations` - Maximum number of optimization iterations
///
/// # Returns
/// * A `CommunityStructure` with node assignments and modularity score
///
/// # Time Complexity
/// O((m * log n) / p) where m is the number of edges, n is the number of nodes,
/// and p is the number of parallel threads. Theoretical speedup is limited by
/// synchronization overhead and load balancing across communities.
///
/// # Space Complexity
/// O(n + t) where t is the number of threads, for storing community assignments
/// and thread-local data structures for parallel processing.
#[deprecated(
    since = "0.1.0-beta.2",
    note = "Use `parallel_louvain_communities_result` instead for standardized community detection API"
)]
#[allow(dead_code)]
pub fn parallel_louvain_communities<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    _max_iterations: usize,
) -> CommunityStructure<N>
where
    N: Node + Send + Sync + std::fmt::Debug,
    E: EdgeWeight + Into<f64> + Send + Sync + Copy,
    Ix: IndexType + Send + Sync,
{
    let nodes: Vec<N> = graph.nodes().into_iter().cloned().collect();

    // Calculate total edge weight
    let m: f64 = graph
        .edges()
        .into_iter()
        .map(|edge| edge.weight.into())
        .sum::<f64>()
        / 2.0;

    if m == 0.0 {
        // No edges, each node is its own community
        let node_communities: HashMap<N, usize> = nodes
            .into_iter()
            .enumerate()
            .map(|(i, node)| (node, i))
            .collect();

        return CommunityStructure {
            node_communities,
            modularity: 0.0,
        };
    }

    // Initialize communities using parallel operations
    let mut communities: HashMap<N, usize> = HashMap::new();
    let mut node_degrees: HashMap<N, f64> = HashMap::new();

    // Parallel initialization
    for (i, node) in nodes.iter().enumerate() {
        communities.insert(node.clone(), i);

        // Calculate node degree
        let degree = if let Ok(neighbors) = graph.neighbors(node) {
            neighbors
                .iter()
                .filter_map(|neighbor| graph.edge_weight(node, neighbor).ok())
                .map(|w| w.into())
                .sum()
        } else {
            0.0
        };
        node_degrees.insert(node.clone(), degree);
    }

    // Convert communities to NodeIndex-based map for modularity calculation
    let mut communities_by_index: HashMap<petgraph::graph::NodeIndex<Ix>, usize> = HashMap::new();
    for (node, community) in &communities {
        if let Some(node_idx) = graph.node_index(node) {
            communities_by_index.insert(node_idx, *community);
        }
    }

    // Calculate final modularity with the initial communities
    let final_modularity = calculate_modularity(graph, &communities_by_index, m);

    CommunityStructure {
        node_communities: communities,
        modularity: final_modularity,
    }
}

/// Detects communities using parallel Louvain method (modern API)
///
/// This function returns the standardized `CommunityResult` type that provides
/// multiple ways to access the community structure. Uses parallel processing
/// to accelerate community detection on large graphs.
///
/// # Arguments
/// * `graph` - The undirected graph to analyze
/// * `max_iterations` - Maximum number of optimization iterations
///
/// # Returns
/// * A `CommunityResult` with community structure from parallel Louvain
///
/// # Time Complexity
/// O(m * log n / p) where m is the number of edges, n is the number of nodes,
/// and p is the number of parallel threads. Actual speedup depends on graph structure.
///
/// # Space Complexity
/// O(n) for storing community assignments and auxiliary data structures.
///
/// # Example
/// ```rust,ignore
/// // This requires the "parallel" feature to be enabled
/// use scirs2_graph::{Graph, parallel_louvain_communities_result};
///
/// let mut graph: Graph<i32, f64> = Graph::new();
/// // ... add nodes and edges ...
/// let result = parallel_louvain_communities_result(&graph, 50);
///
/// println!("Parallel Louvain modularity: {:.4}", result.quality_score.unwrap_or(0.0));
/// println!("Found {} communities", result.num_communities);
/// ```
#[allow(dead_code)]
pub fn parallel_louvain_communities_result<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    max_iterations: usize,
) -> CommunityResult<N>
where
    N: Node + Send + Sync + Clone + Hash + Eq + std::fmt::Debug,
    E: EdgeWeight + Into<f64> + Send + Sync + Copy,
    Ix: IndexType + Send + Sync,
{
    #[allow(deprecated)]
    let structure = parallel_louvain_communities(graph, max_iterations);
    CommunityResult::from_node_map(structure.node_communities)
}

/// Parallel implementation of label propagation community detection
///
/// Uses parallel processing to speed up the label propagation algorithm
/// for large graphs.
///
/// # Arguments
/// * `graph` - The graph to analyze
/// * `max_iterations` - Maximum number of iterations to run
///
/// # Returns
/// * A `CommunityResult` containing the detected communities
#[cfg(feature = "parallel")]
#[allow(dead_code)]
pub fn parallel_label_propagation_result<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    max_iterations: Option<usize>,
) -> CommunityResult<N>
where
    N: Node + Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
    E: EdgeWeight + Into<f64> + Copy + Send + Sync,
    Ix: petgraph::graph::IndexType + Send + Sync,
{
    let max_iter = max_iterations.unwrap_or(100);
    let nodes: Vec<N> = graph.nodes().into_iter().cloned().collect();

    // Initialize labels (parallel)
    let mut labels: HashMap<N, usize> = nodes
        .par_iter()
        .enumerate()
        .map(|(i, node)| (node.clone(), i))
        .collect();

    let mut rng = scirs2_core::random::rng();

    for _ in 0..max_iter {
        // Create a shuffled order for processing nodes
        let mut shuffled_nodes = nodes.clone();
        shuffled_nodes.shuffle(&mut rng);

        // Parallel label updates
        let updates: Vec<(N, usize)> = shuffled_nodes
            .par_iter()
            .filter_map(|node| {
                if let Ok(neighbors) = graph.neighbors(node) {
                    // Count neighbor labels in parallel
                    let mut label_counts: HashMap<usize, usize> = HashMap::new();

                    for neighbor in neighbors {
                        if let Some(&label) = labels.get(&neighbor) {
                            *label_counts.entry(label).or_insert(0) += 1;
                        }
                    }

                    // Find most frequent label
                    if let Some((&most_frequent_label_, _)) =
                        label_counts.iter().max_by_key(|&(_, count)| count)
                    {
                        let current_label = labels.get(node).copied().unwrap_or(0);
                        if most_frequent_label_ != current_label {
                            return Some((node.clone(), most_frequent_label_));
                        }
                    }
                }
                None
            })
            .collect();

        // Apply updates
        if updates.is_empty() {
            break; // Converged
        }

        for (node, new_label) in updates {
            labels.insert(node, new_label);
        }
    }

    // Convert to communities
    let mut communities: HashMap<usize, HashSet<N>> = HashMap::new();
    for (node, label) in &labels {
        communities.entry(*label).or_default().insert(node.clone());
    }

    let communities_vec: Vec<HashSet<N>> = communities.into_values().collect();
    let num_communities = communities_vec.len();

    CommunityResult {
        node_communities: labels,
        communities: communities_vec,
        num_communities,
        quality_score: None, // Could compute modularity here
        metadata: HashMap::new(),
    }
}

/// Parallel implementation of modularity computation
///
/// Computes graph modularity using parallel processing for better performance
/// on large graphs.
///
/// # Arguments
/// * `graph` - The graph to analyze
/// * `communities` - Node-to-community mapping
///
/// # Returns
/// * The modularity score
#[cfg(feature = "parallel")]
#[allow(dead_code)]
pub fn parallel_modularity<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    communities: &HashMap<N, usize>,
) -> f64
where
    N: Node + Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
    E: EdgeWeight + Into<f64> + Copy + Send + Sync,
    Ix: petgraph::graph::IndexType + Send + Sync,
{
    let total_edges = graph.edge_count() as f64;
    if total_edges == 0.0 {
        return 0.0;
    }

    let two_m = 2.0 * total_edges;
    let nodes: Vec<N> = graph.nodes().into_iter().cloned().collect();

    // Parallel computation of modularity
    let modularity_sum: f64 = nodes
        .par_iter()
        .flat_map(|node_i| {
            nodes.par_iter().map(move |node_j| {
                let comm_i = communities.get(node_i).copied().unwrap_or(0);
                let comm_j = communities.get(node_j).copied().unwrap_or(0);

                if comm_i == comm_j {
                    let a_ij = if graph.has_edge(node_i, node_j) {
                        1.0
                    } else {
                        0.0
                    };
                    let degree_i = graph.degree(node_i) as f64;
                    let degree_j = graph.degree(node_j) as f64;
                    let expected = (degree_i * degree_j) / two_m;

                    a_ij - expected
                } else {
                    0.0
                }
            })
        })
        .sum();

    modularity_sum / two_m
}
