//! Social network analysis algorithms
//!
//! This module provides specialized algorithms for analyzing social graphs:
//!
//! - **Influence Maximization**: Find the top-k most influential seed nodes
//!   under Independent Cascade or Linear Threshold diffusion models
//! - **Role Detection**: Identify structurally equivalent node roles
//! - **Echo Chamber Detection**: Partition users into ideologically isolated groups
//! - **Polarization Index**: Quantify the degree of network polarization
//!
//! # References
//! - Kempe, Kleinberg & Tardos (2003) — influence maximization / IC / LT
//! - Lorrain & White (1971) — structural equivalence
//! - Del Vicario et al. (2016) — echo chamber detection

use std::cmp::Reverse;
use std::collections::{HashMap, HashSet, VecDeque};

use scirs2_core::random::{Rng, RngExt};

use crate::base::{EdgeWeight, Graph, Node};
use crate::error::{GraphError, Result};

/// Type alias for numeric node id used in social network operations
pub type NodeId = usize;

/// Spread estimation function type
type SpreadFn = Box<dyn Fn(&[NodeId]) -> f64>;

// ============================================================================
// Diffusion / cascade models
// ============================================================================

/// Diffusion model for influence propagation
#[derive(Debug, Clone, PartialEq, Default)]
pub enum CascadeModel {
    /// Independent Cascade (IC): each active node tries to activate each
    /// inactive neighbor independently with probability `edge_weight`
    #[default]
    IndependentCascade,
    /// Linear Threshold (LT): a node activates when the total weight of
    /// incoming active neighbors exceeds a per-node threshold drawn from
    /// Uniform(0, 1)
    LinearThreshold,
}

// ============================================================================
// Influence maximization
// ============================================================================

/// Estimate the expected spread of a seed set under the IC model
/// using Monte-Carlo simulation with `num_simulations` runs.
///
/// Returns the average number of activated nodes (including seeds).
fn estimate_spread_ic(
    adj: &HashMap<NodeId, Vec<(NodeId, f64)>>,
    seeds: &[NodeId],
    num_simulations: usize,
) -> f64 {
    let mut rng = scirs2_core::random::rng();
    let mut total = 0.0f64;

    for _ in 0..num_simulations {
        let mut active: HashSet<NodeId> = seeds.iter().cloned().collect();
        let mut queue: VecDeque<NodeId> = seeds.iter().cloned().collect();

        while let Some(node) = queue.pop_front() {
            if let Some(neighbors) = adj.get(&node) {
                for &(nbr, prob) in neighbors {
                    if !active.contains(&nbr) && rng.random::<f64>() < prob {
                        active.insert(nbr);
                        queue.push_back(nbr);
                    }
                }
            }
        }
        total += active.len() as f64;
    }

    total / num_simulations as f64
}

/// Estimate expected spread under the Linear Threshold model
fn estimate_spread_lt(
    adj: &HashMap<NodeId, Vec<(NodeId, f64)>>,
    n_nodes: usize,
    seeds: &[NodeId],
    num_simulations: usize,
) -> f64 {
    let mut rng = scirs2_core::random::rng();
    let mut total = 0.0f64;

    for _ in 0..num_simulations {
        // Draw random thresholds
        let thresholds: Vec<f64> = (0..n_nodes).map(|_| rng.random::<f64>()).collect();
        let mut active: HashSet<NodeId> = seeds.iter().cloned().collect();
        let mut changed = true;

        while changed {
            changed = false;
            for node in 0..n_nodes {
                if active.contains(&node) {
                    continue;
                }
                // Sum of weights from active in-neighbors
                let influence: f64 = adj
                    .get(&node)
                    .map(|nbrs| {
                        nbrs.iter()
                            .filter(|&&(nbr, _)| active.contains(&nbr))
                            .map(|&(_, w)| w)
                            .sum::<f64>()
                    })
                    .unwrap_or(0.0);

                if influence >= thresholds[node] {
                    active.insert(node);
                    changed = true;
                }
            }
        }
        total += active.len() as f64;
    }

    total / num_simulations as f64
}

/// Configuration for influence maximization
#[derive(Debug, Clone)]
pub struct InfluenceConfig {
    /// Diffusion model to use
    pub model: CascadeModel,
    /// Number of Monte-Carlo simulations per candidate
    pub num_simulations: usize,
    /// Default edge activation probability (used when weight ∉ (0,1))
    pub default_prob: f64,
}

impl Default for InfluenceConfig {
    fn default() -> Self {
        InfluenceConfig {
            model: CascadeModel::IndependentCascade,
            num_simulations: 100,
            default_prob: 0.1,
        }
    }
}

/// Select the top-k most influential seed nodes using a greedy hill-climbing
/// algorithm (Kempe, Kleinberg & Tardos 2003).
///
/// At each step, the node that maximises the marginal gain in expected spread
/// is added to the seed set.
///
/// # Arguments
/// * `graph` - Undirected or directed graph with edge weights as probabilities
/// * `k` - Number of seed nodes to select
/// * `config` - Model and simulation settings
///
/// # Returns
/// A vector of `k` node ids (0-indexed) in order of selection
pub fn influence_maximization<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    k: usize,
    config: &InfluenceConfig,
) -> Result<Vec<NodeId>>
where
    N: Node + Clone + std::fmt::Debug,
    E: EdgeWeight + Clone + Into<f64>,
    Ix: petgraph::graph::IndexType,
{
    let nodes: Vec<N> = graph.nodes().into_iter().cloned().collect();
    let n = nodes.len();

    if k == 0 {
        return Ok(Vec::new());
    }
    if k > n {
        return Err(GraphError::InvalidParameter {
            param: "k".to_string(),
            value: k.to_string(),
            expected: format!("<= n_nodes ({})", n),
            context: "influence_maximization".to_string(),
        });
    }

    // Build numeric adjacency list
    let node_to_idx: HashMap<N, NodeId> = nodes
        .iter()
        .enumerate()
        .map(|(i, nd)| (nd.clone(), i))
        .collect();

    let mut adj: HashMap<NodeId, Vec<(NodeId, f64)>> = HashMap::new();
    for edge in graph.edges() {
        let si = *node_to_idx
            .get(&edge.source)
            .ok_or_else(|| GraphError::node_not_found("source node"))?;
        let ti = *node_to_idx
            .get(&edge.target)
            .ok_or_else(|| GraphError::node_not_found("target node"))?;
        let w: f64 = edge.weight.clone().into();
        let prob = if w > 0.0 && w <= 1.0 {
            w
        } else {
            config.default_prob
        };
        adj.entry(si).or_default().push((ti, prob));
        adj.entry(ti).or_default().push((si, prob)); // undirected
    }

    let spread_fn: SpreadFn = match &config.model {
        CascadeModel::IndependentCascade => {
            let adj_ref = adj.clone();
            let sims = config.num_simulations;
            Box::new(move |seeds| estimate_spread_ic(&adj_ref, seeds, sims))
        }
        CascadeModel::LinearThreshold => {
            let adj_ref = adj.clone();
            let sims = config.num_simulations;
            Box::new(move |seeds| estimate_spread_lt(&adj_ref, n, seeds, sims))
        }
    };

    let mut seeds: Vec<NodeId> = Vec::with_capacity(k);
    let mut current_spread = 0.0f64;

    for _ in 0..k {
        let mut best_node = None;
        let mut best_gain = f64::NEG_INFINITY;

        for candidate in 0..n {
            if seeds.contains(&candidate) {
                continue;
            }
            let mut trial_seeds = seeds.clone();
            trial_seeds.push(candidate);
            let spread = spread_fn(&trial_seeds);
            let gain = spread - current_spread;

            if gain > best_gain {
                best_gain = gain;
                best_node = Some(candidate);
            }
        }

        if let Some(node) = best_node {
            seeds.push(node);
            current_spread += best_gain;
        }
    }

    Ok(seeds)
}

// ============================================================================
// Role detection
// ============================================================================

/// Structural role of a node in the network
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RoleType {
    /// High-degree central hub connecting many nodes
    Hub,
    /// Low-degree peripheral node connected mainly to hubs
    Peripheral,
    /// Node bridging between communities (high betweenness, moderate degree)
    Bridge,
    /// Ordinary member of a community
    Member,
    /// Isolated node with no connections
    Isolated,
}

impl std::fmt::Display for RoleType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RoleType::Hub => write!(f, "Hub"),
            RoleType::Peripheral => write!(f, "Peripheral"),
            RoleType::Bridge => write!(f, "Bridge"),
            RoleType::Member => write!(f, "Member"),
            RoleType::Isolated => write!(f, "Isolated"),
        }
    }
}

/// Detect structural roles for all nodes based on degree and local clustering.
///
/// The assignment uses thresholds derived from the graph's degree statistics:
/// - **Isolated**: degree 0
/// - **Hub**: degree > mean + std
/// - **Peripheral**: degree < mean - 0.5*std AND low local clustering
/// - **Bridge**: clustering coefficient much lower than graph average
/// - **Member**: otherwise
///
/// # Arguments
/// * `graph` - The graph to analyze
///
/// # Returns
/// Map from node index to its detected role
pub fn role_detection<N, E, Ix>(graph: &Graph<N, E, Ix>) -> Result<HashMap<NodeId, RoleType>>
where
    N: Node + Clone + std::fmt::Debug,
    E: EdgeWeight + Clone + Into<f64>,
    Ix: petgraph::graph::IndexType,
{
    let nodes: Vec<N> = graph.nodes().into_iter().cloned().collect();
    let n = nodes.len();

    if n == 0 {
        return Ok(HashMap::new());
    }

    // Compute degrees
    let degrees: Vec<f64> = nodes.iter().map(|nd| graph.degree(nd) as f64).collect();
    let mean_deg = degrees.iter().sum::<f64>() / n as f64;
    let var_deg = degrees.iter().map(|d| (d - mean_deg).powi(2)).sum::<f64>() / n as f64;
    let std_deg = var_deg.sqrt();

    // Compute local clustering coefficient per node
    let clustering: Vec<f64> = nodes
        .iter()
        .map(|nd| local_clustering_coefficient(graph, nd))
        .collect();

    let mean_clustering = if n > 0 {
        clustering.iter().sum::<f64>() / n as f64
    } else {
        0.0
    };

    let mut roles = HashMap::with_capacity(n);

    for (i, _node) in nodes.iter().enumerate() {
        let deg = degrees[i];
        let clust = clustering[i];

        let role = if deg == 0.0 {
            RoleType::Isolated
        } else if deg > mean_deg + std_deg {
            RoleType::Hub
        } else if deg < (mean_deg - 0.5 * std_deg).max(1.0) && clust < mean_clustering * 0.5 {
            RoleType::Peripheral
        } else if clust < mean_clustering * 0.4 && deg >= 2.0 {
            RoleType::Bridge
        } else {
            RoleType::Member
        };

        roles.insert(i, role);
    }

    Ok(roles)
}

/// Compute local clustering coefficient for a node
fn local_clustering_coefficient<N, E, Ix>(graph: &Graph<N, E, Ix>, node: &N) -> f64
where
    N: Node + Clone + std::fmt::Debug,
    E: EdgeWeight + Clone + Into<f64>,
    Ix: petgraph::graph::IndexType,
{
    let neighbors: Vec<N> = match graph.neighbors(node) {
        Ok(nbrs) => nbrs,
        Err(_) => return 0.0,
    };
    let k = neighbors.len();
    if k < 2 {
        return 0.0;
    }

    let mut triangles = 0usize;
    for i in 0..k {
        for j in i + 1..k {
            if graph.has_edge(&neighbors[i], &neighbors[j]) {
                triangles += 1;
            }
        }
    }

    let max_possible = k * (k - 1) / 2;
    if max_possible == 0 {
        0.0
    } else {
        triangles as f64 / max_possible as f64
    }
}

// ============================================================================
// Echo chamber detection
// ============================================================================

/// Detect echo chambers using a label-propagation-inspired algorithm
/// that respects node opinion features.
///
/// Nodes are partitioned into communities where internal edge density is
/// high and cross-community connections are few. The `features` argument
/// provides a numeric opinion/attribute vector per node (indexed 0..n).
///
/// # Arguments
/// * `graph` - The social graph
/// * `features` - Per-node feature vectors; must have `graph.node_count()` entries
///
/// # Returns
/// A list of echo chambers, each being a list of node indices
pub fn echo_chamber_detection<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    features: &[Vec<f64>],
) -> Result<Vec<Vec<NodeId>>>
where
    N: Node + Clone + std::fmt::Debug,
    E: EdgeWeight + Clone + Into<f64>,
    Ix: petgraph::graph::IndexType,
{
    let nodes: Vec<N> = graph.nodes().into_iter().cloned().collect();
    let n = nodes.len();

    if n == 0 {
        return Ok(Vec::new());
    }
    if features.len() != n {
        return Err(GraphError::InvalidParameter {
            param: "features".to_string(),
            value: format!("{} rows", features.len()),
            expected: format!("{} rows (one per node)", n),
            context: "echo_chamber_detection".to_string(),
        });
    }

    // Build numeric adjacency
    let node_to_idx: HashMap<N, NodeId> = nodes
        .iter()
        .enumerate()
        .map(|(i, nd)| (nd.clone(), i))
        .collect();

    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for edge in graph.edges() {
        if let (Some(&si), Some(&ti)) =
            (node_to_idx.get(&edge.source), node_to_idx.get(&edge.target))
        {
            adj[si].push(ti);
            adj[ti].push(si);
        }
    }

    // Feature-aware label propagation
    // Initialize: each node has its own label
    let mut labels: Vec<NodeId> = (0..n).collect();

    // Run multiple rounds of propagation
    for _round in 0..20 {
        let mut changed = false;

        // Randomized order via deterministic pseudo-random permutation
        let mut order: Vec<usize> = (0..n).collect();
        // Simple Fisher-Yates with deterministic seed
        for i in (1..n).rev() {
            let j = i
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407)
                % (i + 1);
            order.swap(i, j);
        }

        for &node in &order {
            let nbrs = &adj[node];
            if nbrs.is_empty() {
                continue;
            }

            // Score each candidate label by: frequency + feature similarity
            let mut label_scores: HashMap<NodeId, f64> = HashMap::new();
            for &nbr in nbrs {
                let lbl = labels[nbr];
                let sim = feature_similarity(&features[node], &features[nbr]);
                *label_scores.entry(lbl).or_default() += 1.0 + sim;
            }

            if let Some((&best_label, _)) = label_scores
                .iter()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            {
                if best_label != labels[node] {
                    labels[node] = best_label;
                    changed = true;
                }
            }
        }

        if !changed {
            break;
        }
    }

    // Group nodes by final label
    let mut chambers: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
    for (node, &lbl) in labels.iter().enumerate() {
        chambers.entry(lbl).or_default().push(node);
    }

    let mut result: Vec<Vec<NodeId>> = chambers.into_values().collect();
    result.sort_by_key(|b| Reverse(b.len())); // Largest first
    Ok(result)
}

/// Cosine-like feature similarity in [−1, 1]
fn feature_similarity(a: &[f64], b: &[f64]) -> f64 {
    if a.is_empty() || b.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-10);
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-10);
    dot / (norm_a * norm_b)
}

// ============================================================================
// Polarization index
// ============================================================================

/// Compute the polarization index of a social network.
///
/// The index combines three signals:
/// 1. **Modularity**: ratio of intra-community vs cross-community edges
///    (computed via echo chamber partition)
/// 2. **Homophily**: average feature similarity within vs across detected chambers
/// 3. **Fragmentation**: fraction of cut edges between chambers
///
/// The returned value is in [0, 1]: 0 = fully integrated, 1 = fully polarized.
///
/// # Arguments
/// * `graph` - The social graph
/// * `features` - Per-node opinion features (optional; pass empty vecs if unavailable)
///
/// # Returns
/// Polarization index ∈ [0, 1]
pub fn polarization_index<N, E, Ix>(graph: &Graph<N, E, Ix>, features: &[Vec<f64>]) -> Result<f64>
where
    N: Node + Clone + std::fmt::Debug,
    E: EdgeWeight + Clone + Into<f64>,
    Ix: petgraph::graph::IndexType,
{
    let nodes: Vec<N> = graph.nodes().into_iter().cloned().collect();
    let n = nodes.len();

    if n < 2 {
        return Ok(0.0);
    }

    let feat_len = features.first().map(|f| f.len()).unwrap_or(0);
    let feature_pad: Vec<Vec<f64>>;
    let features_ref: &[Vec<f64>] = if features.len() == n {
        features
    } else {
        feature_pad = vec![vec![0.0; feat_len.max(1)]; n];
        &feature_pad
    };

    // Get chamber assignments
    let chambers = echo_chamber_detection(graph, features_ref)?;
    let num_chambers = chambers.len();

    if num_chambers <= 1 {
        return Ok(0.0);
    }

    // Map node → chamber id
    let mut node_chamber: Vec<usize> = vec![0; n];
    for (cid, chamber) in chambers.iter().enumerate() {
        for &node in chamber {
            if node < n {
                node_chamber[node] = cid;
            }
        }
    }

    let node_to_idx: HashMap<N, NodeId> = nodes
        .iter()
        .enumerate()
        .map(|(i, nd)| (nd.clone(), i))
        .collect();

    let edges = graph.edges();
    let total_edges = edges.len() as f64;

    if total_edges == 0.0 {
        return Ok(0.0);
    }

    // Count intra- and cross-chamber edges
    let mut intra = 0.0f64;
    let mut cross = 0.0f64;
    let mut intra_sim = 0.0f64;
    let mut cross_sim = 0.0f64;

    for edge in &edges {
        if let (Some(&si), Some(&ti)) =
            (node_to_idx.get(&edge.source), node_to_idx.get(&edge.target))
        {
            let sim = feature_similarity(
                features_ref.get(si).map(|v| v.as_slice()).unwrap_or(&[]),
                features_ref.get(ti).map(|v| v.as_slice()).unwrap_or(&[]),
            );
            if node_chamber[si] == node_chamber[ti] {
                intra += 1.0;
                intra_sim += sim;
            } else {
                cross += 1.0;
                cross_sim += sim;
            }
        }
    }

    // Modularity component: high intra / total = high polarization
    let modularity_component = intra / total_edges;

    // Homophily component: if features available, compare similarities
    let homophily_component = if feat_len > 0 && (intra + cross) > 0.0 {
        let avg_intra_sim = if intra > 0.0 { intra_sim / intra } else { 0.0 };
        let avg_cross_sim = if cross > 0.0 { cross_sim / cross } else { 0.0 };
        // Normalize difference to [0, 1]
        ((avg_intra_sim - avg_cross_sim + 2.0) / 4.0).clamp(0.0, 1.0)
    } else {
        0.5 // neutral if no features
    };

    // Combine: weighted average
    let polarization = 0.6 * modularity_component + 0.4 * homophily_component;
    Ok(polarization.clamp(0.0, 1.0))
}

// ============================================================================
// Additional utility: spread simulation
// ============================================================================

/// Simulate information spread from a set of seed nodes and return all
/// activated nodes under the chosen diffusion model.
///
/// Useful for what-if analysis after running `influence_maximization`.
pub fn simulate_spread<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    seeds: &[NodeId],
    config: &InfluenceConfig,
) -> Result<HashSet<NodeId>>
where
    N: Node + Clone + std::fmt::Debug,
    E: EdgeWeight + Clone + Into<f64>,
    Ix: petgraph::graph::IndexType,
{
    let nodes: Vec<N> = graph.nodes().into_iter().cloned().collect();
    let n = nodes.len();
    let node_to_idx: HashMap<N, NodeId> = nodes
        .iter()
        .enumerate()
        .map(|(i, nd)| (nd.clone(), i))
        .collect();

    let mut adj: HashMap<NodeId, Vec<(NodeId, f64)>> = HashMap::new();
    for edge in graph.edges() {
        let si = *node_to_idx
            .get(&edge.source)
            .ok_or_else(|| GraphError::node_not_found("source"))?;
        let ti = *node_to_idx
            .get(&edge.target)
            .ok_or_else(|| GraphError::node_not_found("target"))?;
        let w: f64 = edge.weight.clone().into();
        let prob = if w > 0.0 && w <= 1.0 {
            w
        } else {
            config.default_prob
        };
        adj.entry(si).or_default().push((ti, prob));
        adj.entry(ti).or_default().push((si, prob));
    }

    let active_count = match &config.model {
        CascadeModel::IndependentCascade => {
            // Run a single full simulation
            let mut rng = scirs2_core::random::rng();
            let mut active: HashSet<NodeId> = seeds.iter().cloned().collect();
            let mut queue: VecDeque<NodeId> = seeds.iter().cloned().collect();
            while let Some(node) = queue.pop_front() {
                if let Some(neighbors) = adj.get(&node) {
                    for &(nbr, prob) in neighbors {
                        if !active.contains(&nbr) && rng.random::<f64>() < prob {
                            active.insert(nbr);
                            queue.push_back(nbr);
                        }
                    }
                }
            }
            active
        }
        CascadeModel::LinearThreshold => {
            let mut rng = scirs2_core::random::rng();
            let thresholds: Vec<f64> = (0..n).map(|_| rng.random::<f64>()).collect();
            let mut active: HashSet<NodeId> = seeds.iter().cloned().collect();
            let mut changed = true;
            while changed {
                changed = false;
                for node in 0..n {
                    if active.contains(&node) {
                        continue;
                    }
                    let influence: f64 = adj
                        .get(&node)
                        .map(|nbrs| {
                            nbrs.iter()
                                .filter(|&&(nbr, _)| active.contains(&nbr))
                                .map(|&(_, w)| w)
                                .sum::<f64>()
                        })
                        .unwrap_or(0.0);
                    if influence >= thresholds[node] {
                        active.insert(node);
                        changed = true;
                    }
                }
            }
            active
        }
    };

    Ok(active_count)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::Graph;

    fn make_social_graph() -> Graph<usize, f64> {
        let mut g: Graph<usize, f64> = Graph::new();
        // Two cliques connected by a bridge
        for i in 0..4 {
            for j in i + 1..4 {
                let _ = g.add_edge(i, j, 0.3);
            }
        }
        for i in 5..9 {
            for j in i + 1..9 {
                let _ = g.add_edge(i, j, 0.3);
            }
        }
        // Bridge node 4 connecting the two cliques
        let _ = g.add_edge(3, 4, 0.1);
        let _ = g.add_edge(4, 5, 0.1);
        g
    }

    #[test]
    fn test_influence_maximization_returns_k_seeds() {
        let g = make_social_graph();
        let config = InfluenceConfig {
            model: CascadeModel::IndependentCascade,
            num_simulations: 20,
            default_prob: 0.3,
        };
        let seeds = influence_maximization(&g, 3, &config).expect("IM failed");
        assert_eq!(seeds.len(), 3, "Should return exactly k seeds");
        // No duplicates
        let unique: HashSet<_> = seeds.iter().cloned().collect();
        assert_eq!(unique.len(), 3, "Seeds should be unique");
    }

    #[test]
    fn test_influence_maximization_linear_threshold() {
        let g = make_social_graph();
        let config = InfluenceConfig {
            model: CascadeModel::LinearThreshold,
            num_simulations: 20,
            default_prob: 0.3,
        };
        let seeds = influence_maximization(&g, 2, &config).expect("IM LT failed");
        assert_eq!(seeds.len(), 2);
    }

    #[test]
    fn test_influence_maximization_k_zero() {
        let g = make_social_graph();
        let config = InfluenceConfig::default();
        let seeds = influence_maximization(&g, 0, &config).expect("IM k=0");
        assert!(seeds.is_empty());
    }

    #[test]
    fn test_role_detection_identifies_hub() {
        let g = make_social_graph();
        let roles = role_detection(&g).expect("Role detection failed");
        // Node 4 is the bridge
        assert!(roles.contains_key(&4), "Node 4 should have a role");
        // At least one hub in each clique (high degree nodes)
        let hubs: Vec<_> = roles.values().filter(|r| **r == RoleType::Hub).collect();
        assert!(!hubs.is_empty(), "Should detect at least one hub");
    }

    #[test]
    fn test_role_detection_isolated() {
        let mut g: Graph<usize, f64> = Graph::new();
        g.add_node(0);
        g.add_node(1);
        let _ = g.add_edge(0, 1, 1.0);
        g.add_node(2); // isolated
        let roles = role_detection(&g).expect("Roles failed");
        assert_eq!(roles.get(&2), Some(&RoleType::Isolated));
    }

    #[test]
    fn test_echo_chamber_detection_two_groups() {
        let g = make_social_graph();
        // Feature: group A has opinion ~0, group B has opinion ~1
        let features: Vec<Vec<f64>> = (0..9)
            .map(|i| vec![if i < 4 { 0.1 } else { 0.9 }])
            .collect();
        let chambers = echo_chamber_detection(&g, &features).expect("Echo chamber failed");
        assert!(!chambers.is_empty(), "Should detect at least one chamber");
        // Total nodes across all chambers should equal graph node count
        let total: usize = chambers.iter().map(|c| c.len()).sum();
        assert_eq!(total, 9, "All nodes must be assigned to a chamber");
    }

    #[test]
    fn test_echo_chamber_feature_size_mismatch() {
        let g = make_social_graph();
        let features: Vec<Vec<f64>> = vec![vec![0.5]; 3]; // wrong size
        let result = echo_chamber_detection(&g, &features);
        assert!(
            result.is_err(),
            "Should return error for mismatched features"
        );
    }

    #[test]
    fn test_polarization_index_range() {
        let g = make_social_graph();
        let features: Vec<Vec<f64>> = (0..9)
            .map(|i| vec![if i < 4 { 0.0 } else { 1.0 }])
            .collect();
        let pi = polarization_index(&g, &features).expect("Polarization failed");
        assert!(
            (0.0..=1.0).contains(&pi),
            "Polarization index must be in [0,1], got {}",
            pi
        );
    }

    #[test]
    fn test_polarization_index_no_features() {
        let g = make_social_graph();
        let features: Vec<Vec<f64>> = vec![vec![0.0; 0]; 9];
        let pi = polarization_index(&g, &features).expect("Polarization (no feat)");
        assert!((0.0..=1.0).contains(&pi));
    }

    #[test]
    fn test_simulate_spread_ic() {
        let g = make_social_graph();
        let config = InfluenceConfig {
            model: CascadeModel::IndependentCascade,
            num_simulations: 10,
            default_prob: 0.3,
        };
        let activated = simulate_spread(&g, &[0], &config).expect("Spread failed");
        // At minimum, the seed itself is activated
        assert!(activated.contains(&0), "Seed must be in activated set");
    }
}
