//! Hierarchical community detection algorithms

use super::modularity::modularity;
use super::types::{CommunityResult, CommunityStructure};
use crate::base::{EdgeWeight, Graph, IndexType, Node};
use std::collections::{HashMap, HashSet};
use std::hash::Hash;

/// Hierarchical community structure using agglomerative clustering (legacy API)
///
/// **Note**: This function is deprecated in favor of `hierarchical_communities_result`.
/// It will be removed in version 2.0.
///
/// This algorithm starts with each node as its own community and iteratively
/// merges communities to maximize modularity. It builds a dendrogram-like
/// structure showing the hierarchy of communities.
///
/// # Arguments
/// * `graph` - The graph to analyze
/// * `linkage` - Linkage criterion ("single", "complete", "average")
///
/// # Returns
/// * A vector of community structures at different hierarchy levels
#[deprecated(
    since = "0.1.0-beta.2",
    note = "Use `hierarchical_communities_result` instead"
)]
#[allow(dead_code)]
pub fn hierarchical_communities<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    linkage: &str,
) -> Vec<CommunityStructure<N>>
where
    N: Node + Clone + Hash + Eq + std::fmt::Debug,
    E: EdgeWeight + Into<f64> + Copy,
    Ix: IndexType,
{
    let nodes: Vec<N> = graph.nodes().into_iter().cloned().collect();
    let n = nodes.len();

    if n == 0 {
        return vec![];
    }

    let mut results = Vec::new();

    // Start with each node as its own community
    let mut current_communities: HashMap<N, usize> = nodes
        .iter()
        .enumerate()
        .map(|(i, node)| (node.clone(), i))
        .collect();

    // Record initial state
    let initial_mod = modularity(graph, &current_communities);
    results.push(CommunityStructure {
        node_communities: current_communities.clone(),
        modularity: initial_mod,
    });

    // Keep track of which communities exist
    let mut active_communities: HashSet<usize> = (0..n).collect();

    // Agglomerative merging
    while active_communities.len() > 1 {
        let mut best_merge: Option<(usize, usize)> = None;
        let mut best_modularity = f64::NEG_INFINITY;

        // Try all possible merges
        let communities_vec: Vec<usize> = active_communities.iter().cloned().collect();
        for i in 0..communities_vec.len() {
            for j in (i + 1)..communities_vec.len() {
                let comm1 = communities_vec[i];
                let comm2 = communities_vec[j];

                // Check if these communities are connected
                if are_communities_connected(graph, &current_communities, comm1, comm2) {
                    // Try merging these communities
                    let mut test_communities = current_communities.clone();
                    for (_, community) in test_communities.iter_mut() {
                        if *community == comm2 {
                            *community = comm1;
                        }
                    }

                    let test_modularity = modularity(graph, &test_communities);

                    // Use different criteria based on linkage
                    let score = match linkage {
                        "single" => {
                            calculate_single_linkage(graph, &current_communities, comm1, comm2)
                        }
                        "complete" => {
                            calculate_complete_linkage(graph, &current_communities, comm1, comm2)
                        }
                        "average" => {
                            calculate_average_linkage(graph, &current_communities, comm1, comm2)
                        }
                        _ => test_modularity, // Default to modularity
                    };

                    if score > best_modularity {
                        best_modularity = score;
                        best_merge = Some((comm1, comm2));
                    }
                }
            }
        }

        // Perform best merge
        if let Some((comm1, comm2)) = best_merge {
            // Merge comm2 into comm1
            for (_, community) in current_communities.iter_mut() {
                if *community == comm2 {
                    *community = comm1;
                }
            }
            active_communities.remove(&comm2);

            // Record this level
            let current_mod = modularity(graph, &current_communities);
            results.push(CommunityStructure {
                node_communities: current_communities.clone(),
                modularity: current_mod,
            });
        } else {
            // No more valid merges
            break;
        }
    }

    // Renumber all community structures
    for result in &mut results {
        let mut community_map: HashMap<usize, usize> = HashMap::new();
        let mut next_id = 0;
        for &comm in result.node_communities.values() {
            if let std::collections::hash_map::Entry::Vacant(e) = community_map.entry(comm) {
                e.insert(next_id);
                next_id += 1;
            }
        }

        // Apply renumbering
        for (_, comm) in result.node_communities.iter_mut() {
            *comm = community_map[comm];
        }
    }

    results
}

/// Hierarchical community structure with standardized CommunityResult return type
///
/// This function provides the same functionality as `hierarchical_communities` but returns
/// a vector of standardized `CommunityResult` types that provide multiple ways to access
/// the community structure at each hierarchy level.
///
/// This algorithm starts with each node as its own community and iteratively
/// merges communities to maximize modularity. It builds a dendrogram-like
/// structure showing the hierarchy of communities.
///
/// # Arguments
/// * `graph` - The graph to analyze
/// * `linkage` - Linkage criterion ("single", "complete", "average")
///
/// # Returns
/// * A vector of `CommunityResult`s representing different hierarchy levels
///
/// # Time Complexity
/// O(n³) for complete linkage, O(n² * m) for single linkage, where n is the number
/// of nodes and m is the number of edges. The algorithm builds a complete dendrogram
/// by iteratively merging communities based on the specified linkage criterion.
///
/// # Space Complexity
/// O(n²) for storing the distance matrix and the hierarchical structure at all levels.
///
/// # Example
/// ```rust
/// use scirs2_graph::{Graph, hierarchical_communities_result};
///
/// let mut graph: Graph<i32, f64> = Graph::new();
/// // ... add nodes and edges ...
/// let results = hierarchical_communities_result(&graph, "average");
///
/// for (level, result) in results.iter().enumerate() {
///     println!("Level {}: {} communities", level, result.num_communities);
/// }
/// ```
#[allow(dead_code)]
pub fn hierarchical_communities_result<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    linkage: &str,
) -> Vec<CommunityResult<N>>
where
    N: Node + Clone + Hash + Eq + std::fmt::Debug,
    E: EdgeWeight + Into<f64> + Copy,
    Ix: IndexType,
{
    #[allow(deprecated)]
    let structures = hierarchical_communities(graph, linkage);
    structures
        .into_iter()
        .map(CommunityResult::from_community_structure)
        .collect()
}

/// Check if two communities are connected by at least one edge
#[allow(dead_code)]
fn are_communities_connected<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    communities: &HashMap<N, usize>,
    comm1: usize,
    comm2: usize,
) -> bool
where
    N: Node + std::fmt::Debug,
    E: EdgeWeight,
    Ix: IndexType,
{
    for (node, &node_comm) in communities {
        if node_comm == comm1 {
            if let Ok(neighbors) = graph.neighbors(node) {
                for neighbor in neighbors {
                    if let Some(&neighbor_comm) = communities.get(&neighbor) {
                        if neighbor_comm == comm2 {
                            return true;
                        }
                    }
                }
            }
        }
    }
    false
}

/// Calculate single linkage distance (minimum distance between communities)
#[allow(dead_code)]
fn calculate_single_linkage<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    communities: &HashMap<N, usize>,
    comm1: usize,
    comm2: usize,
) -> f64
where
    N: Node + std::fmt::Debug,
    E: EdgeWeight + Into<f64> + Copy,
    Ix: IndexType,
{
    let mut min_distance = f64::INFINITY;

    for (node1, &node1_comm) in communities {
        if node1_comm == comm1 {
            for (node2, &node2_comm) in communities {
                if node2_comm == comm2 {
                    if let Ok(weight) = graph.edge_weight(node1, node2) {
                        let distance = 1.0 / (1.0 + weight.into()); // Convert weight to distance
                        min_distance = min_distance.min(distance);
                    }
                }
            }
        }
    }

    if min_distance == f64::INFINITY {
        0.0 // No direct connection
    } else {
        1.0 / min_distance // Convert back to similarity
    }
}

/// Calculate complete linkage distance (maximum distance between communities)
#[allow(dead_code)]
fn calculate_complete_linkage<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    communities: &HashMap<N, usize>,
    comm1: usize,
    comm2: usize,
) -> f64
where
    N: Node + std::fmt::Debug,
    E: EdgeWeight + Into<f64> + Copy,
    Ix: IndexType,
{
    let mut max_distance: f64 = 0.0;
    let mut found_connection = false;

    for (node1, &node1_comm) in communities {
        if node1_comm == comm1 {
            for (node2, &node2_comm) in communities {
                if node2_comm == comm2 {
                    if let Ok(weight) = graph.edge_weight(node1, node2) {
                        let distance = 1.0 / (1.0 + weight.into());
                        max_distance = max_distance.max(distance);
                        found_connection = true;
                    }
                }
            }
        }
    }

    if found_connection {
        1.0 / max_distance
    } else {
        0.0
    }
}

/// Calculate average linkage distance (average distance between communities)
#[allow(dead_code)]
fn calculate_average_linkage<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    communities: &HashMap<N, usize>,
    comm1: usize,
    comm2: usize,
) -> f64
where
    N: Node + std::fmt::Debug,
    E: EdgeWeight + Into<f64> + Copy,
    Ix: IndexType,
{
    let mut total_distance = 0.0;
    let mut count = 0;

    for (node1, &node1_comm) in communities {
        if node1_comm == comm1 {
            for (node2, &node2_comm) in communities {
                if node2_comm == comm2 {
                    if let Ok(weight) = graph.edge_weight(node1, node2) {
                        let distance = 1.0 / (1.0 + weight.into());
                        total_distance += distance;
                        count += 1;
                    }
                }
            }
        }
    }

    if count > 0 {
        1.0 / (total_distance / count as f64)
    } else {
        0.0
    }
}
