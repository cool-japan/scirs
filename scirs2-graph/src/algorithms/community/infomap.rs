//! Infomap algorithm for community detection

use super::modularity::modularity;
use super::types::CommunityResult;
use crate::base::{EdgeWeight, Graph, IndexType, Node};
use std::collections::HashMap;
use std::hash::Hash;

/// Represents the result of the Infomap algorithm
#[derive(Debug, Clone)]
pub struct InfomapResult<N: Node> {
    /// Map from node to community ID
    pub node_communities: HashMap<N, usize>,
    /// The map equation (code length) of this partition - lower is better
    pub code_length: f64,
    /// The modularity score of this community structure
    pub modularity: f64,
}

/// Infomap algorithm for community detection
///
/// The Infomap algorithm uses information theory to find communities that minimize
/// the description length of random walks on the graph. It optimizes the map equation
/// which balances the cost of describing the partition with the cost of describing
/// random walks within and between communities.
///
/// # Arguments
/// * `graph` - The undirected graph to analyze
/// * `max_iterations` - Maximum number of optimization iterations
/// * `tolerance` - Convergence tolerance for code length improvement
///
/// # Returns
/// * An InfomapResult with node assignments, code length, and modularity
///
/// # Time Complexity
/// O(k * m * log n) where k is the number of iterations, m is the number of edges,
/// and n is the number of nodes. Each iteration involves computing flow probabilities
/// and optimizing the map equation across all nodes and edges.
///
/// # Space Complexity
/// O(n + m) for storing flow probabilities, community assignments, and the
/// hierarchical module structure required by the map equation.
#[allow(dead_code)]
pub fn infomap_communities<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    max_iterations: usize,
    tolerance: f64,
) -> InfomapResult<N>
where
    N: Node + Clone + Hash + Eq + std::fmt::Debug,
    E: EdgeWeight + Into<f64> + Copy,
    Ix: IndexType,
{
    let nodes: Vec<N> = graph.nodes().into_iter().cloned().collect();
    let n = nodes.len();

    if n == 0 {
        return InfomapResult {
            node_communities: HashMap::new(),
            code_length: 0.0,
            modularity: 0.0,
        };
    }

    // Build transition probability matrix and compute steady-state probabilities
    let (transition_matrix, node_weights) = build_transition_matrix(graph, &nodes);
    let stationary_probs = compute_stationary_distribution(&transition_matrix, &node_weights);

    // Initialize each node in its own community
    let mut communities: HashMap<N, usize> = nodes
        .iter()
        .enumerate()
        .map(|(i, node)| (node.clone(), i))
        .collect();

    let mut current_code_length = calculate_map_equation(
        graph,
        &communities,
        &transition_matrix,
        &stationary_probs,
        &nodes,
    );
    let mut best_communities = communities.clone();
    let mut best_code_length = current_code_length;

    let mut rng = scirs2_core::random::rng();

    for _iteration in 0..max_iterations {
        let mut improved = false;

        // Try moving each node to neighboring communities
        for node in &nodes {
            let current_community = communities[node];
            let mut best_community = current_community;
            let mut best_local_code_length = current_code_length;

            // Get neighboring communities
            let mut neighboring_communities = std::collections::HashSet::new();
            if let Ok(neighbors) = graph.neighbors(node) {
                for neighbor in neighbors {
                    if let Some(&comm) = communities.get(&neighbor) {
                        neighboring_communities.insert(comm);
                    }
                }
            }

            // Try moving to each neighboring community
            for &candidate_community in &neighboring_communities {
                if candidate_community != current_community {
                    communities.insert(node.clone(), candidate_community);
                    let new_code_length = calculate_map_equation(
                        graph,
                        &communities,
                        &transition_matrix,
                        &stationary_probs,
                        &nodes,
                    );

                    if new_code_length < best_local_code_length {
                        best_local_code_length = new_code_length;
                        best_community = candidate_community;
                    }
                }
            }

            // Make the best move
            if best_community != current_community {
                communities.insert(node.clone(), best_community);
                current_code_length = best_local_code_length;
                improved = true;

                if current_code_length < best_code_length {
                    best_code_length = current_code_length;
                    best_communities = communities.clone();
                }
            } else {
                // Restore original community
                communities.insert(node.clone(), current_community);
            }
        }

        // Check convergence
        if !improved || (best_code_length - current_code_length).abs() < tolerance {
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

    let final_modularity = modularity(graph, &best_communities);

    InfomapResult {
        node_communities: best_communities,
        code_length: best_code_length,
        modularity: final_modularity,
    }
}

/// Build transition probability matrix for random walks
#[allow(dead_code)]
fn build_transition_matrix<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    nodes: &[N],
) -> (Vec<Vec<f64>>, Vec<f64>)
where
    N: Node + std::fmt::Debug,
    E: EdgeWeight + Into<f64> + Copy,
    Ix: IndexType,
{
    let n = nodes.len();
    let mut transition_matrix = vec![vec![0.0; n]; n];
    let mut node_weights = vec![0.0; n];

    // Create node index mapping
    let node_to_idx: HashMap<&N, usize> = nodes.iter().enumerate().map(|(i, n)| (n, i)).collect();

    // Calculate transition probabilities
    for (i, node) in nodes.iter().enumerate() {
        let mut total_weight = 0.0;

        // First pass: calculate total outgoing weight
        if let Ok(neighbors) = graph.neighbors(node) {
            for neighbor in neighbors {
                if let Ok(weight) = graph.edge_weight(node, &neighbor) {
                    total_weight += weight.into();
                }
            }
        }

        node_weights[i] = total_weight;

        // Second pass: set transition probabilities
        if total_weight > 0.0 {
            if let Ok(neighbors) = graph.neighbors(node) {
                for neighbor in neighbors {
                    if let Some(&j) = node_to_idx.get(&neighbor) {
                        if let Ok(weight) = graph.edge_weight(node, &neighbor) {
                            transition_matrix[i][j] = weight.into() / total_weight;
                        }
                    }
                }
            }
        } else {
            // Teleport to random node if no outgoing edges (dangling node)
            for j in 0..n {
                transition_matrix[i][j] = 1.0 / n as f64;
            }
        }
    }

    (transition_matrix, node_weights)
}

/// Compute stationary distribution using power iteration
#[allow(dead_code)]
fn compute_stationary_distribution(
    transition_matrix: &[Vec<f64>],
    node_weights: &[f64],
) -> Vec<f64> {
    let n = transition_matrix.len();
    if n == 0 {
        return vec![];
    }

    // Initialize with degree-based probabilities
    let total_weight: f64 = node_weights.iter().sum();
    let mut pi = if total_weight > 0.0 {
        node_weights.iter().map(|&w| w / total_weight).collect()
    } else {
        vec![1.0 / n as f64; n]
    };

    // Power iteration to find stationary distribution
    for _ in 0..1000 {
        let mut new_pi = vec![0.0; n];

        for (i, new_pi_item) in new_pi.iter_mut().enumerate().take(n) {
            for j in 0..n {
                *new_pi_item += pi[j] * transition_matrix[j][i];
            }
        }

        // Normalize
        let sum: f64 = new_pi.iter().sum();
        if sum > 0.0 {
            for p in new_pi.iter_mut() {
                *p /= sum;
            }
        }

        // Check convergence
        let diff: f64 = pi
            .iter()
            .zip(&new_pi)
            .map(|(old, new)| (old - new).abs())
            .sum();

        pi = new_pi;

        if diff < 1e-10 {
            break;
        }
    }

    pi
}

/// Calculate the map equation (code length) for a given partition
#[allow(dead_code)]
fn calculate_map_equation<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    communities: &HashMap<N, usize>,
    transition_matrix: &[Vec<f64>],
    stationary_probs: &[f64],
    nodes: &[N],
) -> f64
where
    N: Node + std::fmt::Debug,
    E: EdgeWeight + Into<f64> + Copy,
    Ix: IndexType,
{
    let n = nodes.len();
    if n == 0 {
        return 0.0;
    }

    // Create node index mapping
    let node_to_idx: HashMap<&N, usize> = nodes.iter().enumerate().map(|(i, n)| (n, i)).collect();

    // Calculate exit probabilities and internal flow for each community
    let mut community_exit_prob: HashMap<usize, f64> = HashMap::new();
    let mut community_flow: HashMap<usize, f64> = HashMap::new();

    // Initialize community maps
    for &comm in communities.values() {
        community_exit_prob.insert(comm, 0.0);
        community_flow.insert(comm, 0.0);
    }

    // Calculate exit probabilities (flow leaving each community)
    for (node, &comm) in communities {
        if let Some(&i) = node_to_idx.get(node) {
            let pi_i = stationary_probs[i];
            *community_flow.get_mut(&comm).unwrap() += pi_i;

            // Add transitions to other communities
            if let Ok(neighbors) = graph.neighbors(node) {
                for neighbor in neighbors {
                    if let Some(&neighbor_comm) = communities.get(&neighbor) {
                        if neighbor_comm != comm {
                            if let Some(&j) = node_to_idx.get(&neighbor) {
                                *community_exit_prob.get_mut(&comm).unwrap() +=
                                    pi_i * transition_matrix[i][j];
                            }
                        }
                    }
                }
            }
        }
    }

    // Calculate map equation
    let mut code_length = 0.0;

    // Index codebook term: H(Q) = -sum(q_alpha * log(q_alpha))
    let total_exit_flow: f64 = community_exit_prob.values().sum();
    if total_exit_flow > 0.0 {
        for &q_alpha in community_exit_prob.values() {
            if q_alpha > 0.0 {
                code_length -= q_alpha * (q_alpha / total_exit_flow).ln();
            }
        }
    }

    // Module codebook terms: sum_alpha(q_alpha + p_alpha) * H(P_alpha)
    for (&comm, &q_alpha) in &community_exit_prob {
        let p_alpha = community_flow[&comm];
        let total_alpha = q_alpha + p_alpha;

        if total_alpha > 0.0 {
            // Entropy within community
            let mut h_alpha = 0.0;

            // Exit probability contribution
            if q_alpha > 0.0 {
                h_alpha -= (q_alpha / total_alpha) * (q_alpha / total_alpha).ln();
            }

            // Internal transition probabilities
            for (node, &node_comm) in communities {
                if node_comm == comm {
                    if let Some(&i) = node_to_idx.get(node) {
                        let pi_i = stationary_probs[i];
                        if pi_i > 0.0 {
                            let prob_in_module = pi_i / total_alpha;
                            h_alpha -= prob_in_module * prob_in_module.ln();
                        }
                    }
                }
            }

            code_length += total_alpha * h_alpha;
        }
    }

    code_length
}
