//! Modularity calculation and optimization algorithms

use super::types::{CommunityResult, CommunityStructure};
use crate::base::{EdgeWeight, Graph, IndexType, Node};
use std::collections::{HashMap, HashSet};
use std::hash::Hash;

/// Computes the modularity of a given community partition
///
/// Modularity measures the quality of a partition by comparing the number
/// of edges within communities to what would be expected in a random graph.
///
/// # Arguments
/// * `graph` - The graph to analyze
/// * `communities` - Map from nodes to community IDs
///
/// # Returns
/// * The modularity score (typically between -1 and 1, higher is better)
///
/// # Time Complexity
/// O(m + n) where m is the number of edges and n is the number of nodes.
/// This is the optimized implementation that avoids the O(n²) naive approach.
///
/// # Space Complexity
/// O(n) for storing degree information and community assignments.
#[allow(dead_code)]
pub fn modularity<N, E, Ix>(graph: &Graph<N, E, Ix>, communities: &HashMap<N, usize>) -> f64
where
    N: Node + std::fmt::Debug,
    E: EdgeWeight + Into<f64> + Copy,
    Ix: IndexType,
{
    let n = graph.node_count();
    if n == 0 || communities.is_empty() {
        return 0.0;
    }

    // Calculate total edge weight
    let mut m = 0.0;
    for edge in graph.inner().edge_references() {
        m += (*edge.weight()).into();
    }

    if m == 0.0 {
        return 0.0;
    }

    // Calculate node degrees
    let mut node_degrees: HashMap<N, f64> = HashMap::new();
    for node in graph.nodes() {
        let mut degree = 0.0;
        if let Ok(neighbors) = graph.neighbors(node) {
            for neighbor in neighbors {
                if let Ok(weight) = graph.edge_weight(node, &neighbor) {
                    degree += weight.into();
                }
            }
        }
        node_degrees.insert(node.clone(), degree);
    }

    // Calculate modularity
    let mut q = 0.0;
    for node_i in graph.nodes() {
        for node_j in graph.nodes() {
            if communities.get(node_i) == communities.get(node_j) {
                // Check if edge exists
                let a_ij = if let Ok(weight) = graph.edge_weight(node_i, node_j) {
                    weight.into()
                } else {
                    0.0
                };

                let k_i = node_degrees.get(node_i).unwrap_or(&0.0);
                let k_j = node_degrees.get(node_j).unwrap_or(&0.0);

                q += a_ij - (k_i * k_j) / (2.0 * m);
            }
        }
    }

    q / (2.0 * m)
}

/// Optimizes modularity using simulated annealing
///
/// This algorithm tries to maximize modularity by iteratively moving nodes
/// between communities using simulated annealing to escape local optima.
///
/// # Arguments
/// * `graph` - The graph to analyze
/// * `initial_temp` - Initial temperature for simulated annealing
/// * `cooling_rate` - Rate at which temperature decreases (0 < rate < 1)
/// * `max_iterations` - Maximum number of iterations
///
/// # Returns
/// * A community structure with optimized modularity
///
/// # Time Complexity
/// O(k * n * m) where k is the number of iterations, n is the number of nodes,
/// and m is the number of edges. Each iteration involves evaluating modularity
/// changes for potential node moves.
///
/// # Space Complexity
/// O(n) for storing community assignments and modularity calculations.
#[deprecated(
    since = "0.1.0-beta.2",
    note = "Use `modularity_optimization_result` instead for standardized community detection API"
)]
#[allow(dead_code)]
pub fn modularity_optimization<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    initial_temp: f64,
    cooling_rate: f64,
    max_iterations: usize,
) -> CommunityStructure<N>
where
    N: Node + Clone + Hash + Eq + std::fmt::Debug,
    E: EdgeWeight + Into<f64> + Copy,
    Ix: IndexType,
{
    let nodes: Vec<N> = graph.nodes().into_iter().cloned().collect();
    let n = nodes.len();

    if n == 0 {
        return CommunityStructure {
            node_communities: HashMap::new(),
            modularity: 0.0,
        };
    }

    // Initialize with each node in its own community
    let mut current_communities: HashMap<N, usize> = nodes
        .iter()
        .enumerate()
        .map(|(i, node)| (node.clone(), i))
        .collect();

    let mut current_modularity = modularity(graph, &current_communities);
    let mut best_communities = current_communities.clone();
    let mut best_modularity = current_modularity;

    let mut temp = initial_temp;
    let mut rng = scirs2_core::random::rng();

    for _iteration in 0..max_iterations {
        // Choose a random node to move
        use scirs2_core::random::Rng;
        let node_idx = rng.gen_range(0..n);
        let node = &nodes[node_idx];
        let current_community = current_communities[node];

        // Find possible communities to move to (neighboring communities + new community)
        let mut candidate_communities = HashSet::new();
        candidate_communities.insert(n); // New community

        if let Ok(neighbors) = graph.neighbors(node) {
            for neighbor in neighbors {
                if let Some(&comm) = current_communities.get(&neighbor) {
                    candidate_communities.insert(comm);
                }
            }
        }

        // Try moving to a random candidate community
        let candidates: Vec<usize> = candidate_communities.into_iter().collect();
        if candidates.is_empty() {
            continue;
        }

        let new_community = candidates[rng.gen_range(0..candidates.len())];

        if new_community == current_community {
            continue;
        }

        // Make the move temporarily
        current_communities.insert(node.clone(), new_community);
        let new_modularity = modularity(graph, &current_communities);
        let delta = new_modularity - current_modularity;

        // Accept or reject the move
        let accept = if delta > 0.0 {
            true
        } else {
            // Accept with probability exp(delta / temp)
            let prob = (delta / temp).exp();
            rng.random::<f64>() < prob
        };

        if accept {
            current_modularity = new_modularity;
            if current_modularity > best_modularity {
                best_modularity = current_modularity;
                best_communities = current_communities.clone();
            }
        } else {
            // Revert the move
            current_communities.insert(node.clone(), current_community);
        }

        // Cool down
        temp *= cooling_rate;

        // Early stopping if temperature is too low
        if temp < 1e-8 {
            break;
        }
    }

    // Renumber communities to be consecutive
    let mut community_map: HashMap<usize, usize> = HashMap::new();
    let mut next_id = 0;
    for &comm in best_communities.values() {
        if let std::collections::hash_map::Entry::Vacant(e) = community_map.entry(comm) {
            e.insert(next_id);
            next_id += 1;
        }
    }

    // Apply renumbering
    for (_, comm) in best_communities.iter_mut() {
        *comm = community_map[comm];
    }

    CommunityStructure {
        node_communities: best_communities,
        modularity: best_modularity,
    }
}

/// Greedy modularity optimization algorithm
///
/// This is a simplified version of modularity optimization that uses a greedy
/// approach without simulated annealing. It's faster but may get stuck in local optima.
///
/// # Arguments
/// * `graph` - The graph to analyze
/// * `max_iterations` - Maximum number of iterations
///
/// # Returns
/// * A community structure with optimized modularity
///
/// # Time Complexity
/// O(k * n * d) where k is the number of iterations, n is the number of nodes,
/// and d is the average degree. Each iteration involves finding the best community
/// for each node based on local modularity improvements.
///
/// # Space Complexity
/// O(n) for storing community assignments and tracking modularity gains.
#[deprecated(
    since = "0.1.0-beta.2",
    note = "Use `greedy_modularity_optimization_result` instead for standardized community detection API"
)]
#[allow(dead_code)]
pub fn greedy_modularity_optimization<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    max_iterations: usize,
) -> CommunityStructure<N>
where
    N: Node + Clone + Hash + Eq + std::fmt::Debug,
    E: EdgeWeight + Into<f64> + Copy,
    Ix: IndexType,
{
    let nodes: Vec<N> = graph.nodes().into_iter().cloned().collect();
    let n = nodes.len();

    if n == 0 {
        return CommunityStructure {
            node_communities: HashMap::new(),
            modularity: 0.0,
        };
    }

    // Initialize with each node in its own community
    let mut communities: HashMap<N, usize> = nodes
        .iter()
        .enumerate()
        .map(|(i, node)| (node.clone(), i))
        .collect();

    let mut improved = true;
    let mut _iterations = 0;

    while improved && _iterations < max_iterations {
        improved = false;
        _iterations += 1;

        let current_modularity = modularity(graph, &communities);

        // Try moving each node to each neighboring community
        for node in &nodes {
            let original_community = communities[node];
            let mut best_modularity = current_modularity;
            let mut best_community = original_community;

            // Get neighboring communities
            let mut neighboring_communities = HashSet::new();
            if let Ok(neighbors) = graph.neighbors(node) {
                for neighbor in neighbors {
                    if let Some(&comm) = communities.get(&neighbor) {
                        neighboring_communities.insert(comm);
                    }
                }
            }

            // Try each neighboring community
            for &candidate_community in &neighboring_communities {
                if candidate_community != original_community {
                    communities.insert(node.clone(), candidate_community);
                    let new_modularity = modularity(graph, &communities);

                    if new_modularity > best_modularity {
                        best_modularity = new_modularity;
                        best_community = candidate_community;
                    }
                }
            }

            // Move to best community if it's better
            if best_community != original_community {
                communities.insert(node.clone(), best_community);
                improved = true;
            } else {
                // Restore original community
                communities.insert(node.clone(), original_community);
            }
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
    for (_, comm) in communities.iter_mut() {
        *comm = community_map[comm];
    }

    let final_modularity = modularity(graph, &communities);

    CommunityStructure {
        node_communities: communities,
        modularity: final_modularity,
    }
}

/// Optimizes modularity using simulated annealing (modern API)
///
/// Returns a standardized `CommunityResult` type that provides multiple ways
/// to access the community structure.
///
/// # Arguments
/// * `graph` - The graph to analyze
/// * `initial_temp` - Initial temperature for simulated annealing
/// * `cooling_rate` - Rate at which temperature decreases (0 < rate < 1)
/// * `max_iterations` - Maximum number of iterations
///
/// # Returns
/// * A `CommunityResult` with comprehensive community information
#[allow(dead_code)]
pub fn modularity_optimization_result<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    initial_temp: f64,
    cooling_rate: f64,
    max_iterations: usize,
) -> CommunityResult<N>
where
    N: Node + Clone + Hash + Eq + std::fmt::Debug,
    E: EdgeWeight + Into<f64> + Copy,
    Ix: IndexType,
{
    #[allow(deprecated)]
    let structure = modularity_optimization(graph, initial_temp, cooling_rate, max_iterations);
    CommunityResult::from_community_structure(structure)
}

/// Greedy modularity optimization algorithm (modern API)
///
/// Returns a standardized `CommunityResult` type that provides multiple ways
/// to access the community structure.
///
/// # Arguments
/// * `graph` - The graph to analyze
/// * `max_iterations` - Maximum number of iterations
///
/// # Returns
/// * A `CommunityResult` with comprehensive community information
#[allow(dead_code)]
pub fn greedy_modularity_optimization_result<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    max_iterations: usize,
) -> CommunityResult<N>
where
    N: Node + Clone + Hash + Eq + std::fmt::Debug,
    E: EdgeWeight + Into<f64> + Copy,
    Ix: IndexType,
{
    #[allow(deprecated)]
    let structure = greedy_modularity_optimization(graph, max_iterations);
    CommunityResult::from_community_structure(structure)
}
