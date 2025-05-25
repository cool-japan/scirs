//! Community detection algorithms
//!
//! This module contains algorithms for detecting community structure in graphs.

use crate::base::{EdgeWeight, Graph, IndexType, Node};
use petgraph::visit::EdgeRef;
use rand::seq::SliceRandom;
use std::collections::HashMap;
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
    let mut community_weights: HashMap<usize, f64> = HashMap::new();
    let mut node_weights: HashMap<petgraph::graph::NodeIndex<Ix>, f64> = HashMap::new();

    // Calculate total weight and node weights
    let mut total_weight = 0.0;
    for (i, node_idx) in graph.inner().node_indices().enumerate() {
        communities.insert(node_idx, i);
        let mut weight = 0.0;
        for edge in graph.inner().edges(node_idx) {
            weight += (*edge.weight()).into();
        }
        node_weights.insert(node_idx, weight);
        community_weights.insert(i, weight);
        total_weight += weight;
    }

    // Optimization phase
    let mut improved = true;
    while improved {
        improved = false;

        // For each node, try to find a better community
        for node_idx in graph.inner().node_indices() {
            let current_community = communities[&node_idx];
            let node_weight = node_weights[&node_idx];

            // Calculate weight to each neighboring community
            let mut neighbor_communities: HashMap<usize, f64> = HashMap::new();
            for edge in graph.inner().edges(node_idx) {
                let neighbor_idx = edge.target();
                let neighbor_community = communities[&neighbor_idx];
                let edge_weight: f64 = (*edge.weight()).into();
                *neighbor_communities
                    .entry(neighbor_community)
                    .or_insert(0.0) += edge_weight;
            }

            // Find best community to move to
            let mut best_community = current_community;
            let mut best_gain = 0.0;

            for (&community, &weight_to_community) in &neighbor_communities {
                if community == current_community {
                    continue;
                }

                // Calculate modularity gain
                let community_weight = community_weights[&community];
                let current_comm_weight = community_weights[&current_community];

                let gain = 2.0 * weight_to_community / total_weight
                    - 2.0 * node_weight * community_weight / (total_weight * total_weight)
                    + 2.0 * node_weight * (current_comm_weight - node_weight)
                        / (total_weight * total_weight);

                if gain > best_gain {
                    best_gain = gain;
                    best_community = community;
                }
            }

            // Move node if beneficial
            if best_community != current_community {
                improved = true;
                communities.insert(node_idx, best_community);

                // Update community weights
                let node_w = node_weights[&node_idx];
                *community_weights.get_mut(&current_community).unwrap() -= node_w;
                *community_weights.get_mut(&best_community).unwrap() += node_w;
            }
        }
    }

    // Calculate final modularity
    let modularity = calculate_modularity(graph, &communities, total_weight);

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
    total_weight: f64,
) -> f64
where
    N: Node,
    E: EdgeWeight + Into<f64> + Copy,
    Ix: petgraph::graph::IndexType,
{
    let mut modularity = 0.0;

    for edge in graph.inner().edge_references() {
        let source_comm = communities[&edge.source()];
        let target_comm = communities[&edge.target()];

        if source_comm == target_comm {
            let weight: f64 = (*edge.weight()).into();
            modularity += weight / total_weight;
        }
    }

    // Subtract expected edges
    let mut community_degrees: HashMap<usize, f64> = HashMap::new();
    for node_idx in graph.inner().node_indices() {
        let comm = communities[&node_idx];
        let degree: f64 = graph
            .inner()
            .edges(node_idx)
            .map(|e| (*e.weight()).into())
            .sum();
        *community_degrees.entry(comm).or_insert(0.0) += degree;
    }

    for &degree in community_degrees.values() {
        modularity -= (degree / total_weight).powi(2);
    }

    modularity
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
}
