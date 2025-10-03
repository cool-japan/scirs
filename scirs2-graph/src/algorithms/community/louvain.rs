//! Louvain method for community detection

use super::types::{CommunityResult, CommunityStructure};
use crate::base::{EdgeWeight, Graph, Node};
use petgraph::visit::EdgeRef;
use std::collections::HashMap;
use std::hash::Hash;

/// Detects communities in a graph using the Louvain method (modern API)
///
/// This function returns the standardized `CommunityResult` type that provides
/// multiple ways to access the community structure.
///
/// # Arguments
/// * `graph` - The undirected graph to analyze
///
/// # Returns
/// * A `CommunityResult` with comprehensive community information
///
/// # Time Complexity
/// O(m * log n) where m is the number of edges and n is the number of nodes
/// in typical cases. Can be O(m * n) in worst case with many iterations.
/// The algorithm usually converges quickly in practice.
///
/// # Space Complexity
/// O(n) for storing community assignments and node degrees.
///
/// # Example
/// ```rust
/// use scirs2_graph::{Graph, louvain_communities_result};
///
/// let mut graph: Graph<i32, f64> = Graph::new();
/// // ... add nodes and edges ...
/// let result = louvain_communities_result(&graph);
///
/// println!("Found {} communities", result.num_communities);
/// for (i, community) in result.communities.iter().enumerate() {
///     println!("Community {}: {} members", i, community.len());
/// }
/// ```
#[allow(dead_code)]
pub fn louvain_communities_result<N, E, Ix>(graph: &Graph<N, E, Ix>) -> CommunityResult<N>
where
    N: Node + Clone + Hash + Eq + std::fmt::Debug,
    E: EdgeWeight + Into<f64> + scirs2_core::numeric::Zero + Copy,
    Ix: petgraph::graph::IndexType,
{
    let structure = louvain_communities_legacy(graph);
    CommunityResult::from_community_structure(structure)
}

/// Detects communities in a graph using the Louvain method (legacy API)
///
/// **Note**: This function is deprecated in favor of `louvain_communities_result`.
/// It will be removed in version 2.0.
#[deprecated(
    since = "0.1.0-beta.1",
    note = "Use `louvain_communities_result` instead"
)]
#[allow(dead_code)]
pub fn louvain_communities<N, E, Ix>(graph: &Graph<N, E, Ix>) -> CommunityStructure<N>
where
    N: Node + std::fmt::Debug,
    E: EdgeWeight + Into<f64> + scirs2_core::numeric::Zero + Copy,
    Ix: petgraph::graph::IndexType,
{
    louvain_communities_legacy(graph)
}

/// Internal implementation of Louvain method
#[allow(dead_code)]
fn louvain_communities_legacy<N, E, Ix>(graph: &Graph<N, E, Ix>) -> CommunityStructure<N>
where
    N: Node + std::fmt::Debug,
    E: EdgeWeight + Into<f64> + scirs2_core::numeric::Zero + Copy,
    Ix: petgraph::graph::IndexType,
{
    let n = graph.node_count();
    if n == 0 {
        return CommunityStructure {
            node_communities: HashMap::new(),
            modularity: 0.0,
        };
    }

    // Initialize each node in its own community
    let mut communities: HashMap<petgraph::graph::NodeIndex<Ix>, usize> = HashMap::new();
    let mut node_degrees: HashMap<petgraph::graph::NodeIndex<Ix>, f64> = HashMap::new();

    // Calculate node degrees and total weight
    let mut m = 0.0; // Total weight of edges (sum of all edge weights)
    for edge in graph.inner().edge_references() {
        m += (*edge.weight()).into();
    }

    // Handle edge case
    if m == 0.0 {
        m = 1.0;
    }

    // Calculate node degrees
    for node_idx in graph.inner().node_indices() {
        let mut degree = 0.0;
        for edge in graph.inner().edges(node_idx) {
            degree += (*edge.weight()).into();
        }
        node_degrees.insert(node_idx, degree);
        communities.insert(node_idx, node_idx.index());
    }

    // Optimization phase
    let mut improved = true;
    let mut iterations = 0;
    const MAX_ITERATIONS: usize = 100;

    while improved && iterations < MAX_ITERATIONS {
        improved = false;
        iterations += 1;

        // For each node, try to find a better community
        for node_idx in graph.inner().node_indices() {
            let current_community = communities[&node_idx];
            let k_i = node_degrees[&node_idx]; // Degree of node i

            // Remove node from its community first
            communities.insert(node_idx, node_idx.index());

            // Calculate sum of weights to each neighboring community
            let mut community_weights: HashMap<usize, f64> = HashMap::new();
            for edge in graph.inner().edges(node_idx) {
                let neighbor_idx = edge.target();
                let neighbor_community = communities[&neighbor_idx];
                let edge_weight: f64 = (*edge.weight()).into();
                *community_weights.entry(neighbor_community).or_insert(0.0) += edge_weight;
            }

            // Add current node as a possible community
            community_weights.entry(node_idx.index()).or_insert(0.0);

            // Find best community
            let mut best_community = node_idx.index();
            let mut best_delta_q = 0.0;

            for (&community, &k_i_in) in &community_weights {
                // Calculate sum of degrees of nodes in this community
                let mut sigma_tot = 0.0;
                for (&other_node, &other_community) in &communities {
                    if other_community == community && other_node != node_idx {
                        sigma_tot += node_degrees[&other_node];
                    }
                }

                // Calculate modularity gain
                let delta_q = k_i_in / m - (sigma_tot * k_i) / (2.0 * m * m);

                if delta_q > best_delta_q {
                    best_delta_q = delta_q;
                    best_community = community;
                }
            }

            // Move node to best community
            if best_community != current_community {
                improved = true;
            }
            communities.insert(node_idx, best_community);
        }
    }

    // Renumber communities to be consecutive
    let mut community_map: HashMap<usize, usize> = HashMap::new();
    let mut next_id = 0;
    for &comm in communities.values() {
        if let std::collections::hash_map::Entry::Vacant(e) = community_map.entry(comm) {
            e.insert(next_id);
            next_id += 1;
        }
    }

    // Apply renumbering
    for comm in communities.values_mut() {
        *comm = community_map[comm];
    }

    // Calculate final modularity
    let modularity = calculate_modularity(graph, &communities, m);

    // Convert to final result
    let node_communities: HashMap<N, usize> = communities
        .into_iter()
        .map(|(idx, comm)| (graph.inner()[idx].clone(), comm))
        .collect();

    CommunityStructure {
        node_communities,
        modularity,
    }
}

/// Calculate modularity for a given partition
#[allow(dead_code)]
pub fn calculate_modularity<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    communities: &HashMap<petgraph::graph::NodeIndex<Ix>, usize>,
    m: f64,
) -> f64
where
    N: Node + std::fmt::Debug,
    E: EdgeWeight + Into<f64> + Copy,
    Ix: petgraph::graph::IndexType,
{
    let mut modularity = 0.0;

    // Calculate node degrees
    let mut node_degrees: HashMap<petgraph::graph::NodeIndex<Ix>, f64> = HashMap::new();
    for node_idx in graph.inner().node_indices() {
        let degree: f64 = graph
            .inner()
            .edges(node_idx)
            .map(|e| (*e.weight()).into())
            .sum();
        node_degrees.insert(node_idx, degree);
    }

    // Sum over all pairs of nodes
    for node_i in graph.inner().node_indices() {
        for node_j in graph.inner().node_indices() {
            if communities[&node_i] == communities[&node_j] {
                // Check if edge exists between i and j
                let mut a_ij = 0.0;
                for edge in graph.inner().edges(node_i) {
                    if edge.target() == node_j {
                        a_ij = (*edge.weight()).into();
                        break;
                    }
                }

                let k_i = node_degrees[&node_i];
                let k_j = node_degrees[&node_j];

                modularity += a_ij - (k_i * k_j) / (2.0 * m);
            }
        }
    }

    modularity / (2.0 * m)
}
