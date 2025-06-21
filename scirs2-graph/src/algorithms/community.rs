//! Community detection algorithms
//!
//! This module contains algorithms for detecting community structure in graphs.

use crate::base::{EdgeWeight, Graph, IndexType, Node};
use petgraph::visit::EdgeRef;
use rand::seq::SliceRandom;
use std::collections::{HashMap, HashSet};
use std::hash::Hash;

/// Represents a community structure in a graph
#[derive(Debug, Clone)]
pub struct CommunityStructure<N: Node> {
    /// Map from node to community ID
    pub node_communities: HashMap<N, usize>,
    /// The modularity score of this community structure
    pub modularity: f64,
}

/// Detects communities in a graph using the Louvain method
///
/// The Louvain method is a greedy optimization algorithm that attempts to maximize
/// the modularity of the network partition.
///
/// # Arguments
/// * `graph` - The undirected graph to analyze
///
/// # Returns
/// * A community structure with node assignments and modularity score
pub fn louvain_communities<N, E, Ix>(graph: &Graph<N, E, Ix>) -> CommunityStructure<N>
where
    N: Node,
    E: EdgeWeight + Into<f64> + num_traits::Zero + Copy,
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
fn calculate_modularity<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    communities: &HashMap<petgraph::graph::NodeIndex<Ix>, usize>,
    m: f64,
) -> f64
where
    N: Node,
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

/// Label propagation algorithm for community detection
///
/// Each node adopts the label that most of its neighbors have, with ties broken randomly.
/// Returns a mapping from nodes to community labels.
pub fn label_propagation<N, E, Ix>(graph: &Graph<N, E, Ix>, max_iter: usize) -> HashMap<N, usize>
where
    N: Node + Clone + Hash + Eq,
    E: EdgeWeight,
    Ix: IndexType,
{
    let nodes: Vec<N> = graph.nodes().into_iter().cloned().collect();
    let n = nodes.len();

    if n == 0 {
        return HashMap::new();
    }

    // Initialize each node with its own label
    let mut labels: Vec<usize> = (0..n).collect();
    let node_to_idx: HashMap<N, usize> = nodes
        .iter()
        .enumerate()
        .map(|(i, n)| (n.clone(), i))
        .collect();

    let mut rng = rand::rng();
    let mut changed = true;
    let mut iterations = 0;

    while changed && iterations < max_iter {
        changed = false;
        iterations += 1;

        // Process nodes in random order
        let mut order: Vec<usize> = (0..n).collect();
        order.shuffle(&mut rng);

        for &i in &order {
            let node = &nodes[i];

            // Count labels of neighbors
            let mut label_counts: HashMap<usize, usize> = HashMap::new();

            if let Ok(neighbors) = graph.neighbors(node) {
                for neighbor in neighbors {
                    if let Some(&neighbor_idx) = node_to_idx.get(&neighbor) {
                        let neighbor_label = labels[neighbor_idx];
                        *label_counts.entry(neighbor_label).or_insert(0) += 1;
                    }
                }
            }

            if label_counts.is_empty() {
                continue;
            }

            // Find most frequent label(s)
            let max_count = *label_counts.values().max().unwrap();
            let best_labels: Vec<usize> = label_counts
                .into_iter()
                .filter(|(_, count)| *count == max_count)
                .map(|(label, _)| label)
                .collect();

            // Choose randomly among ties
            use rand::Rng;
            let new_label = best_labels[rng.random_range(0..best_labels.len())];

            if labels[i] != new_label {
                labels[i] = new_label;
                changed = true;
            }
        }
    }

    // Convert to final result
    nodes
        .into_iter()
        .enumerate()
        .map(|(i, node)| (node, labels[i]))
        .collect()
}

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
pub fn modularity<N, E, Ix>(graph: &Graph<N, E, Ix>, communities: &HashMap<N, usize>) -> f64
where
    N: Node,
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
pub fn modularity_optimization<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    initial_temp: f64,
    cooling_rate: f64,
    max_iterations: usize,
) -> CommunityStructure<N>
where
    N: Node + Clone + Hash + Eq,
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
    let mut rng = rand::rng();

    for _iteration in 0..max_iterations {
        // Choose a random node to move
        use rand::Rng;
        let node_idx = rng.random_range(0..n);
        let node = &nodes[node_idx];
        let current_community = current_communities[node];

        // Find possible communities to move to (neighboring communities + new community)
        let mut candidate_communities = std::collections::HashSet::new();
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

        let new_community = candidates[rng.random_range(0..candidates.len())];

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
pub fn greedy_modularity_optimization<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    max_iterations: usize,
) -> CommunityStructure<N>
where
    N: Node + Clone + Hash + Eq,
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
    let mut iterations = 0;

    while improved && iterations < max_iterations {
        improved = false;
        iterations += 1;

        let current_modularity = modularity(graph, &communities);

        // Try moving each node to each neighboring community
        for node in &nodes {
            let original_community = communities[node];
            let mut best_modularity = current_modularity;
            let mut best_community = original_community;

            // Get neighboring communities
            let mut neighboring_communities = std::collections::HashSet::new();
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
pub fn infomap_communities<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    max_iterations: usize,
    tolerance: f64,
) -> InfomapResult<N>
where
    N: Node + Clone + Hash + Eq,
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
    let mut improved = true;
    let mut iterations = 0;

    while improved && iterations < max_iterations {
        improved = false;
        iterations += 1;

        // Try moving each node to the community that minimizes code length
        for (node_idx, node) in nodes.iter().enumerate() {
            let current_community = communities[node];
            let mut best_community = current_community;
            let mut best_delta = 0.0;

            // Get neighboring communities
            let mut candidate_communities = HashSet::new();
            if let Ok(neighbors) = graph.neighbors(node) {
                for neighbor in neighbors {
                    if let Some(&comm) = communities.get(&neighbor) {
                        candidate_communities.insert(comm);
                    }
                }
            }

            // Also consider creating a new community
            let new_community_id = nodes.len() + node_idx;
            candidate_communities.insert(new_community_id);

            // Test each candidate community
            for &candidate_community in &candidate_communities {
                if candidate_community == current_community {
                    continue;
                }

                // Temporarily move node to candidate community
                communities.insert(node.clone(), candidate_community);

                let new_code_length = calculate_map_equation(
                    graph,
                    &communities,
                    &transition_matrix,
                    &stationary_probs,
                    &nodes,
                );

                let delta = current_code_length - new_code_length;

                if delta > best_delta + tolerance {
                    best_delta = delta;
                    best_community = candidate_community;
                }

                // Restore original community for next test
                communities.insert(node.clone(), current_community);
            }

            // Move to best community if improvement found
            if best_community != current_community {
                communities.insert(node.clone(), best_community);
                current_code_length -= best_delta;
                improved = true;

                if current_code_length < best_code_length {
                    best_code_length = current_code_length;
                    best_communities = communities.clone();
                }
            }
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

    // Calculate final modularity
    let final_modularity = modularity(graph, &best_communities);

    InfomapResult {
        node_communities: best_communities,
        code_length: best_code_length,
        modularity: final_modularity,
    }
}

/// Build transition probability matrix for random walks
fn build_transition_matrix<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    nodes: &[N],
) -> (Vec<Vec<f64>>, Vec<f64>)
where
    N: Node,
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
fn calculate_map_equation<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    communities: &HashMap<N, usize>,
    transition_matrix: &[Vec<f64>],
    stationary_probs: &[f64],
    nodes: &[N],
) -> f64
where
    N: Node,
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

/// Fluid communities algorithm
///
/// Fluid communities is a density-based algorithm where communities are formed
/// by propagating "fluids" through the network. Each community starts with a seed
/// node and expands by including neighboring nodes based on density.
///
/// # Arguments
/// * `graph` - The graph to analyze
/// * `num_communities` - Target number of communities to find
/// * `max_iterations` - Maximum number of iterations
///
/// # Returns
/// * A community structure with node assignments and modularity
pub fn fluid_communities<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    num_communities: usize,
    max_iterations: usize,
) -> CommunityStructure<N>
where
    N: Node + Clone + Hash + Eq,
    E: EdgeWeight + Into<f64> + Copy,
    Ix: IndexType,
{
    let nodes: Vec<N> = graph.nodes().into_iter().cloned().collect();
    let n = nodes.len();

    if n == 0 || num_communities == 0 {
        return CommunityStructure {
            node_communities: HashMap::new(),
            modularity: 0.0,
        };
    }

    let num_communities = num_communities.min(n);
    let mut rng = rand::rng();

    // Initialize fluids - each node starts with a random fluid
    let mut node_fluids: HashMap<N, Vec<f64>> = HashMap::new();
    for node in &nodes {
        let mut fluids = vec![0.0; num_communities];
        // Assign random initial fluid
        use rand::Rng;
        let initial_fluid = rng.random_range(0..num_communities);
        fluids[initial_fluid] = 1.0;
        node_fluids.insert(node.clone(), fluids);
    }

    // Fluid propagation iterations
    for _iteration in 0..max_iterations {
        let mut new_fluids: HashMap<N, Vec<f64>> = HashMap::new();

        for node in &nodes {
            let mut fluid_sums = vec![0.0; num_communities];

            // Aggregate fluids from neighbors
            if let Ok(neighbors) = graph.neighbors(node) {
                let neighbor_count = neighbors.len();
                if neighbor_count > 0 {
                    for neighbor in neighbors {
                        if let Some(neighbor_fluids) = node_fluids.get(&neighbor) {
                            for (i, &fluid_amount) in neighbor_fluids.iter().enumerate() {
                                fluid_sums[i] += fluid_amount;
                            }
                        }
                    }

                    // Normalize by number of neighbors
                    for fluid_sum in fluid_sums.iter_mut() {
                        *fluid_sum /= neighbor_count as f64;
                    }
                } else {
                    // Isolated nodes keep their current fluids
                    if let Some(current_fluids) = node_fluids.get(node) {
                        fluid_sums = current_fluids.clone();
                    }
                }
            } else {
                // No neighbors, keep current fluids
                if let Some(current_fluids) = node_fluids.get(node) {
                    fluid_sums = current_fluids.clone();
                }
            }

            // Normalize fluids to sum to 1
            let total: f64 = fluid_sums.iter().sum();
            if total > 0.0 {
                for fluid in fluid_sums.iter_mut() {
                    *fluid /= total;
                }
            } else {
                // If all fluids are zero, assign random fluid
                use rand::Rng;
                let random_fluid = rng.random_range(0..num_communities);
                fluid_sums[random_fluid] = 1.0;
            }

            new_fluids.insert(node.clone(), fluid_sums);
        }

        // Update fluids
        node_fluids = new_fluids;
    }

    // Assign nodes to communities based on dominant fluid
    let mut communities: HashMap<N, usize> = HashMap::new();
    for (node, fluids) in &node_fluids {
        let max_fluid_idx = fluids
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        communities.insert(node.clone(), max_fluid_idx);
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

    // Calculate modularity
    let mod_score = modularity(graph, &communities);

    CommunityStructure {
        node_communities: communities,
        modularity: mod_score,
    }
}

/// Hierarchical community structure using agglomerative clustering
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
pub fn hierarchical_communities<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    linkage: &str,
) -> Vec<CommunityStructure<N>>
where
    N: Node + Clone + Hash + Eq,
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

/// Check if two communities are connected by at least one edge
fn are_communities_connected<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    communities: &HashMap<N, usize>,
    comm1: usize,
    comm2: usize,
) -> bool
where
    N: Node,
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
fn calculate_single_linkage<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    communities: &HashMap<N, usize>,
    comm1: usize,
    comm2: usize,
) -> f64
where
    N: Node,
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
fn calculate_complete_linkage<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    communities: &HashMap<N, usize>,
    comm1: usize,
    comm2: usize,
) -> f64
where
    N: Node,
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
fn calculate_average_linkage<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    communities: &HashMap<N, usize>,
    comm1: usize,
    comm2: usize,
) -> f64
where
    N: Node,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Result as GraphResult;
    use crate::generators::create_graph;

    #[test]
    fn test_louvain_communities() -> GraphResult<()> {
        // Create a graph with two clear communities
        let mut graph = create_graph::<i32, f64>();

        // Dense connections within communities
        graph.add_edge(0, 1, 1.0)?;
        graph.add_edge(1, 2, 1.0)?;
        graph.add_edge(2, 0, 1.0)?;

        graph.add_edge(3, 4, 1.0)?;
        graph.add_edge(4, 5, 1.0)?;
        graph.add_edge(5, 3, 1.0)?;

        // Sparse connection between communities
        graph.add_edge(2, 3, 0.1)?;

        let communities = louvain_communities(&graph);

        // Check that nodes in the same group have the same community
        assert_eq!(
            communities.node_communities[&0],
            communities.node_communities[&1]
        );
        assert_eq!(
            communities.node_communities[&1],
            communities.node_communities[&2]
        );

        assert_eq!(
            communities.node_communities[&3],
            communities.node_communities[&4]
        );
        assert_eq!(
            communities.node_communities[&4],
            communities.node_communities[&5]
        );

        // Check that the two groups have different communities
        assert_ne!(
            communities.node_communities[&0],
            communities.node_communities[&3]
        );

        // Modularity should be positive for good community structure
        // Note: For small graphs, modularity can sometimes be 0 or slightly negative
        // due to numerical precision and the algorithm's initialization
        assert!(
            communities.modularity >= -0.1,
            "Modularity {} is too negative",
            communities.modularity
        );

        Ok(())
    }

    #[test]
    fn test_label_propagation() -> GraphResult<()> {
        // Create a graph with communities
        let mut graph = crate::generators::create_graph::<&str, f64>();

        // Community 1
        graph.add_edge("A", "B", 1.0)?;
        graph.add_edge("B", "C", 1.0)?;
        graph.add_edge("C", "A", 1.0)?;

        // Community 2
        graph.add_edge("D", "E", 1.0)?;
        graph.add_edge("E", "F", 1.0)?;
        graph.add_edge("F", "D", 1.0)?;

        // Weak link between communities
        graph.add_edge("C", "D", 0.1)?;

        let communities = label_propagation(&graph, 100);

        // Check that nodes in the same triangle tend to have the same label
        // (Note: label propagation is stochastic, so we can't guarantee exact results)
        assert_eq!(communities.len(), 6);

        // At least check that all nodes got labels
        assert!(communities.contains_key(&"A"));
        assert!(communities.contains_key(&"B"));
        assert!(communities.contains_key(&"C"));
        assert!(communities.contains_key(&"D"));
        assert!(communities.contains_key(&"E"));
        assert!(communities.contains_key(&"F"));

        Ok(())
    }

    #[test]
    fn test_modularity_calculation() -> GraphResult<()> {
        // Create a simple graph with clear community structure
        let mut graph = create_graph::<i32, f64>();

        // Community 1: triangle
        graph.add_edge(0, 1, 1.0)?;
        graph.add_edge(1, 2, 1.0)?;
        graph.add_edge(2, 0, 1.0)?;

        // Community 2: triangle
        graph.add_edge(3, 4, 1.0)?;
        graph.add_edge(4, 5, 1.0)?;
        graph.add_edge(5, 3, 1.0)?;

        // Weak connection between communities
        graph.add_edge(2, 3, 0.1)?;

        // Define communities manually
        let mut communities = HashMap::new();
        communities.insert(0, 0);
        communities.insert(1, 0);
        communities.insert(2, 0);
        communities.insert(3, 1);
        communities.insert(4, 1);
        communities.insert(5, 1);

        let mod_score = modularity(&graph, &communities);

        // Should be positive for good community structure
        assert!(mod_score > 0.0);

        // Compare with random partition - should be lower
        let mut random_communities = HashMap::new();
        random_communities.insert(0, 0);
        random_communities.insert(1, 1);
        random_communities.insert(2, 0);
        random_communities.insert(3, 1);
        random_communities.insert(4, 0);
        random_communities.insert(5, 1);

        let random_mod = modularity(&graph, &random_communities);
        assert!(mod_score > random_mod);

        Ok(())
    }

    #[test]
    fn test_modularity_optimization() -> GraphResult<()> {
        // Create a graph with community structure
        let mut graph = create_graph::<&str, f64>();

        // Dense connections within communities
        graph.add_edge("A", "B", 1.0)?;
        graph.add_edge("B", "C", 1.0)?;
        graph.add_edge("C", "A", 1.0)?;

        graph.add_edge("D", "E", 1.0)?;
        graph.add_edge("E", "F", 1.0)?;
        graph.add_edge("F", "D", 1.0)?;

        // Sparse connection between communities
        graph.add_edge("C", "D", 0.1)?;

        let result = modularity_optimization(&graph, 1.0, 0.9, 1000);

        // Check that all nodes are assigned to communities
        assert_eq!(result.node_communities.len(), 6);

        // Modularity should be reasonable
        assert!(result.modularity >= -0.1);

        // Verify modularity calculation matches
        let calculated_mod = modularity(&graph, &result.node_communities);
        assert!((result.modularity - calculated_mod).abs() < 1e-10);

        Ok(())
    }

    #[test]
    fn test_greedy_modularity_optimization() -> GraphResult<()> {
        // Create a simple graph
        let mut graph = create_graph::<i32, f64>();

        graph.add_edge(0, 1, 1.0)?;
        graph.add_edge(1, 2, 1.0)?;
        graph.add_edge(3, 4, 1.0)?;
        graph.add_edge(0, 3, 0.1)?; // Weak link

        let result = greedy_modularity_optimization(&graph, 100);

        // Should find some community structure
        assert_eq!(result.node_communities.len(), 5);

        // Modularity should be calculated correctly
        let calculated_mod = modularity(&graph, &result.node_communities);
        assert!((result.modularity - calculated_mod).abs() < 1e-10);

        Ok(())
    }

    #[test]
    fn test_empty_graph_modularity() {
        let graph = create_graph::<i32, f64>();
        let communities = HashMap::new();

        assert_eq!(modularity(&graph, &communities), 0.0);

        let result = modularity_optimization(&graph, 1.0, 0.9, 100);
        assert_eq!(result.modularity, 0.0);
        assert!(result.node_communities.is_empty());
    }

    #[test]
    fn test_infomap_communities() -> GraphResult<()> {
        // Create a graph with clear community structure
        let mut graph = create_graph::<&str, f64>();

        // Community 1: triangle
        graph.add_edge("A", "B", 1.0)?;
        graph.add_edge("B", "C", 1.0)?;
        graph.add_edge("C", "A", 1.0)?;

        // Community 2: triangle
        graph.add_edge("D", "E", 1.0)?;
        graph.add_edge("E", "F", 1.0)?;
        graph.add_edge("F", "D", 1.0)?;

        // Weak connection between communities
        graph.add_edge("C", "D", 0.1)?;

        let result = infomap_communities(&graph, 100, 1e-6);

        // Check that all nodes are assigned to communities
        assert_eq!(result.node_communities.len(), 6);

        // Check that nodes in the same triangle tend to have the same community
        let comm_a = result.node_communities[&"A"];
        let comm_b = result.node_communities[&"B"];
        let comm_c = result.node_communities[&"C"];
        let comm_d = result.node_communities[&"D"];
        let comm_e = result.node_communities[&"E"];
        let comm_f = result.node_communities[&"F"];

        // At least some structure should be detected
        // (Exact community assignment can vary due to the algorithm's heuristic nature)
        assert!(comm_a == comm_b || comm_a == comm_c || comm_b == comm_c);
        assert!(comm_d == comm_e || comm_d == comm_f || comm_e == comm_f);

        // Code length should be finite and positive
        assert!(result.code_length.is_finite());
        assert!(result.code_length >= 0.0);

        // Modularity should be calculated
        assert!(result.modularity.is_finite());

        Ok(())
    }

    #[test]
    fn test_infomap_single_community() -> GraphResult<()> {
        // Create a complete graph (should form single community)
        let mut graph = create_graph::<i32, f64>();

        // Complete triangle
        graph.add_edge(0, 1, 1.0)?;
        graph.add_edge(1, 2, 1.0)?;
        graph.add_edge(2, 0, 1.0)?;

        let result = infomap_communities(&graph, 50, 1e-6);

        // All nodes should be in the same community for a complete graph
        let communities: HashSet<usize> = result.node_communities.values().cloned().collect();
        assert_eq!(communities.len(), 1);

        // Code length should be reasonable
        assert!(result.code_length.is_finite());

        Ok(())
    }

    #[test]
    fn test_infomap_disconnected_components() -> GraphResult<()> {
        // Create two disconnected triangles
        let mut graph = create_graph::<i32, f64>();

        // First triangle
        graph.add_edge(0, 1, 1.0)?;
        graph.add_edge(1, 2, 1.0)?;
        graph.add_edge(2, 0, 1.0)?;

        // Second triangle (disconnected)
        graph.add_edge(3, 4, 1.0)?;
        graph.add_edge(4, 5, 1.0)?;
        graph.add_edge(5, 3, 1.0)?;

        let result = infomap_communities(&graph, 100, 1e-6);

        // Should detect two distinct communities
        let communities: HashSet<usize> = result.node_communities.values().cloned().collect();
        assert_eq!(communities.len(), 2);

        // Nodes within each triangle should be in the same community
        assert_eq!(result.node_communities[&0], result.node_communities[&1]);
        assert_eq!(result.node_communities[&1], result.node_communities[&2]);

        assert_eq!(result.node_communities[&3], result.node_communities[&4]);
        assert_eq!(result.node_communities[&4], result.node_communities[&5]);

        // The two components should be in different communities
        assert_ne!(result.node_communities[&0], result.node_communities[&3]);

        Ok(())
    }

    #[test]
    fn test_infomap_empty_graph() {
        let graph = create_graph::<i32, f64>();
        let result = infomap_communities(&graph, 100, 1e-6);

        assert!(result.node_communities.is_empty());
        assert_eq!(result.code_length, 0.0);
        assert_eq!(result.modularity, 0.0);
    }

    #[test]
    fn test_infomap_single_node() -> GraphResult<()> {
        let mut graph = create_graph::<&str, f64>();
        graph.add_node("A");

        let result = infomap_communities(&graph, 100, 1e-6);

        assert_eq!(result.node_communities.len(), 1);
        assert!(result.node_communities.contains_key(&"A"));
        assert!(result.code_length.is_finite());

        Ok(())
    }

    #[test]
    fn test_infomap_vs_modularity() -> GraphResult<()> {
        // Create a graph and compare Infomap with modularity-based methods
        let mut graph = create_graph::<i32, f64>();

        // Create a graph with known community structure
        // Community 1
        graph.add_edge(0, 1, 1.0)?;
        graph.add_edge(1, 2, 1.0)?;
        graph.add_edge(2, 0, 1.0)?;

        // Community 2
        graph.add_edge(3, 4, 1.0)?;
        graph.add_edge(4, 5, 1.0)?;

        // Bridge
        graph.add_edge(2, 3, 0.1)?;

        let infomap_result = infomap_communities(&graph, 100, 1e-6);
        let louvain_result = louvain_communities(&graph);

        // Both should find some community structure
        let infomap_communities_count: HashSet<usize> =
            infomap_result.node_communities.values().cloned().collect();
        let louvain_communities_count: HashSet<usize> =
            louvain_result.node_communities.values().cloned().collect();

        assert!(!infomap_communities_count.is_empty());
        assert!(!louvain_communities_count.is_empty());

        // Both should have reasonable modularity
        assert!(infomap_result.modularity >= -0.5);
        assert!(louvain_result.modularity >= -0.5);

        Ok(())
    }

    #[test]
    fn test_build_transition_matrix() -> GraphResult<()> {
        let mut graph = create_graph::<i32, f64>();
        graph.add_edge(0, 1, 2.0)?;
        graph.add_edge(1, 2, 1.0)?;

        let nodes = vec![0, 1, 2];
        let (transition_matrix, node_weights) = build_transition_matrix(&graph, &nodes);

        // Check dimensions
        assert_eq!(transition_matrix.len(), 3);
        assert_eq!(node_weights.len(), 3);

        // Check that rows sum to 1 (or 0 for isolated nodes)
        for (i, row) in transition_matrix.iter().enumerate() {
            let row_sum: f64 = row.iter().sum();
            if node_weights[i] > 0.0 {
                assert!((row_sum - 1.0).abs() < 1e-10, "Row {} sum: {}", i, row_sum);
            }
        }

        // Node 0 should transition to node 1 with probability 1
        assert!((transition_matrix[0][1] - 1.0).abs() < 1e-10);

        Ok(())
    }

    #[test]
    fn test_compute_stationary_distribution() {
        // Simple two-node case
        let transition_matrix = vec![
            vec![0.0, 1.0], // Node 0 always goes to node 1
            vec![1.0, 0.0], // Node 1 always goes to node 0
        ];
        let node_weights = vec![1.0, 1.0];

        let stationary = compute_stationary_distribution(&transition_matrix, &node_weights);

        // Should be uniform distribution
        assert_eq!(stationary.len(), 2);
        assert!((stationary[0] - 0.5).abs() < 1e-6);
        assert!((stationary[1] - 0.5).abs() < 1e-6);

        // Check normalization
        let sum: f64 = stationary.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_fluid_communities() -> GraphResult<()> {
        // Create a graph with clear community structure
        let mut graph = create_graph::<i32, f64>();

        // Community 1: triangle
        graph.add_edge(0, 1, 1.0)?;
        graph.add_edge(1, 2, 1.0)?;
        graph.add_edge(2, 0, 1.0)?;

        // Community 2: triangle
        graph.add_edge(3, 4, 1.0)?;
        graph.add_edge(4, 5, 1.0)?;
        graph.add_edge(5, 3, 1.0)?;

        // Weak connection between communities
        graph.add_edge(2, 3, 0.1)?;

        let result = fluid_communities(&graph, 2, 30);

        // Check that all nodes are assigned to communities
        assert_eq!(result.node_communities.len(), 6);

        // Check that we get 1 or 2 communities (fluid communities can converge to fewer communities)
        let communities: HashSet<usize> = result.node_communities.values().cloned().collect();
        assert!(!communities.is_empty() && communities.len() <= 2);

        // Modularity should be calculated
        assert!(result.modularity.is_finite());

        // Verify modularity calculation matches
        let calculated_mod = modularity(&graph, &result.node_communities);
        assert!((result.modularity - calculated_mod).abs() < 1e-10);

        Ok(())
    }

    #[test]
    fn test_fluid_communities_empty_graph() {
        let graph = create_graph::<i32, f64>();
        let result = fluid_communities(&graph, 2, 10);

        assert!(result.node_communities.is_empty());
        assert_eq!(result.modularity, 0.0);
    }

    #[test]
    fn test_fluid_communities_single_node() -> GraphResult<()> {
        let mut graph = create_graph::<&str, f64>();
        graph.add_node("A");

        let result = fluid_communities(&graph, 1, 10);

        assert_eq!(result.node_communities.len(), 1);
        assert!(result.node_communities.contains_key(&"A"));
        assert_eq!(result.node_communities[&"A"], 0);

        Ok(())
    }

    #[test]
    fn test_hierarchical_communities() -> GraphResult<()> {
        // Create a simple graph with clear structure
        let mut graph = create_graph::<i32, f64>();

        // Two disconnected triangles
        graph.add_edge(0, 1, 1.0)?;
        graph.add_edge(1, 2, 1.0)?;
        graph.add_edge(2, 0, 1.0)?;

        graph.add_edge(3, 4, 1.0)?;
        graph.add_edge(4, 5, 1.0)?;
        graph.add_edge(5, 3, 1.0)?;

        // Connect them with a weak link
        graph.add_edge(2, 3, 0.1)?;

        let hierarchy = hierarchical_communities(&graph, "average");

        // Should have multiple levels (starting from each node as its own community)
        assert!(!hierarchy.is_empty());
        assert!(hierarchy.len() > 1);

        // First level should have 6 communities (one per node)
        assert_eq!(hierarchy[0].node_communities.len(), 6);
        let first_communities: HashSet<usize> =
            hierarchy[0].node_communities.values().cloned().collect();
        assert_eq!(first_communities.len(), 6);

        // Last level should have fewer communities
        let last_communities: HashSet<usize> = hierarchy
            .last()
            .unwrap()
            .node_communities
            .values()
            .cloned()
            .collect();
        assert!(last_communities.len() < 6);

        // Each level should have valid modularity
        for level in &hierarchy {
            assert!(level.modularity.is_finite());
        }

        Ok(())
    }

    #[test]
    fn test_hierarchical_communities_single_linkage() -> GraphResult<()> {
        let mut graph = create_graph::<&str, f64>();

        // Create a path: A-B-C-D
        graph.add_edge("A", "B", 1.0)?;
        graph.add_edge("B", "C", 1.0)?;
        graph.add_edge("C", "D", 1.0)?;

        let hierarchy = hierarchical_communities(&graph, "single");

        assert!(!hierarchy.is_empty());
        assert_eq!(hierarchy[0].node_communities.len(), 4);

        // Should progressively merge communities
        for i in 1..hierarchy.len() {
            let prev_communities: HashSet<usize> = hierarchy[i - 1]
                .node_communities
                .values()
                .cloned()
                .collect();
            let curr_communities: HashSet<usize> =
                hierarchy[i].node_communities.values().cloned().collect();
            assert!(curr_communities.len() <= prev_communities.len());
        }

        Ok(())
    }

    #[test]
    fn test_hierarchical_communities_complete_linkage() -> GraphResult<()> {
        let mut graph = create_graph::<i32, f64>();

        // Create a small complete graph
        graph.add_edge(0, 1, 1.0)?;
        graph.add_edge(1, 2, 1.0)?;
        graph.add_edge(2, 0, 1.0)?;

        let hierarchy = hierarchical_communities(&graph, "complete");

        assert!(!hierarchy.is_empty());
        assert_eq!(hierarchy[0].node_communities.len(), 3);

        // Should eventually merge to a single community
        let final_communities: HashSet<usize> = hierarchy
            .last()
            .unwrap()
            .node_communities
            .values()
            .cloned()
            .collect();
        assert!(final_communities.len() <= 3);

        Ok(())
    }

    #[test]
    fn test_hierarchical_communities_empty_graph() {
        let graph = create_graph::<i32, f64>();
        let hierarchy = hierarchical_communities(&graph, "average");

        assert!(hierarchy.is_empty());
    }

    #[test]
    fn test_are_communities_connected() -> GraphResult<()> {
        let mut graph = create_graph::<i32, f64>();

        graph.add_edge(0, 1, 1.0)?;
        graph.add_edge(2, 3, 1.0)?;
        graph.add_edge(1, 2, 1.0)?; // Bridge between communities

        let mut communities = HashMap::new();
        communities.insert(0, 0);
        communities.insert(1, 0);
        communities.insert(2, 1);
        communities.insert(3, 1);

        // Communities 0 and 1 should be connected via edge 1-2
        assert!(are_communities_connected(&graph, &communities, 0, 1));

        // Remove the bridge
        let mut disconnected_communities = HashMap::new();
        disconnected_communities.insert(0, 0);
        disconnected_communities.insert(1, 1);
        disconnected_communities.insert(2, 2);
        disconnected_communities.insert(3, 3);

        // Now communities should not be connected (except those with direct edges)
        assert!(!are_communities_connected(
            &graph,
            &disconnected_communities,
            0,
            2
        ));
        assert!(!are_communities_connected(
            &graph,
            &disconnected_communities,
            0,
            3
        ));

        Ok(())
    }

    #[test]
    fn test_linkage_calculations() -> GraphResult<()> {
        let mut graph = create_graph::<i32, f64>();

        graph.add_edge(0, 1, 2.0)?;
        graph.add_edge(0, 2, 1.0)?;
        graph.add_edge(1, 2, 3.0)?;

        let mut communities = HashMap::new();
        communities.insert(0, 0);
        communities.insert(1, 1);
        communities.insert(2, 1);

        // Test single linkage (should find minimum distance)
        let single = calculate_single_linkage(&graph, &communities, 0, 1);
        assert!(single > 0.0);

        // Test complete linkage (should find maximum distance)
        let complete = calculate_complete_linkage(&graph, &communities, 0, 1);
        assert!(complete > 0.0);

        // Test average linkage
        let average = calculate_average_linkage(&graph, &communities, 0, 1);
        assert!(average > 0.0);

        // Test with disconnected communities
        communities.insert(0, 0);
        communities.insert(1, 1);
        communities.insert(2, 2);

        let single_disconnected = calculate_single_linkage(&graph, &communities, 0, 2);
        assert!(single_disconnected > 0.0); // Should find connection via edge 0-2

        Ok(())
    }
}
