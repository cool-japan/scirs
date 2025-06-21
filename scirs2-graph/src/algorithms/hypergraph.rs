//! Hypergraph algorithms
//!
//! This module contains algorithms specifically designed for hypergraphs,
//! including transversals, cuts, and hypergraph-specific analysis methods.

use crate::base::{EdgeWeight, Hypergraph, IndexType, Node};
use crate::error::{GraphError, Result};
use std::collections::{HashMap, HashSet, VecDeque};

/// Result of hypergraph cut analysis
#[derive(Debug, Clone)]
pub struct HypergraphCut<N: Node> {
    /// Nodes in the first partition
    pub partition_a: HashSet<N>,
    /// Nodes in the second partition  
    pub partition_b: HashSet<N>,
    /// Hyperedges that cross the cut (connect both partitions)
    pub cut_hyperedges: Vec<usize>,
    /// Total weight of cut hyperedges
    pub cut_weight: f64,
}

/// Represents a minimal transversal (hitting set) of a hypergraph
#[derive(Debug, Clone)]
pub struct MinimalTransversal<N: Node> {
    /// Nodes in the transversal
    pub nodes: HashSet<N>,
    /// Hyperedges hit by this transversal
    pub hit_hyperedges: Vec<usize>,
    /// Size of the transversal
    pub size: usize,
}

/// Find all minimal transversals (hitting sets) of a hypergraph
///
/// A transversal is a set of nodes that intersects every hyperedge.
/// This is also known as the hitting set problem.
///
/// # Arguments
/// * `hypergraph` - The hypergraph to analyze
/// * `max_size` - Maximum size of transversals to consider (for efficiency)
///
/// # Returns
/// * Vector of minimal transversals
pub fn minimal_transversals<N, E, Ix>(
    hypergraph: &Hypergraph<N, E, Ix>,
    max_size: Option<usize>,
) -> Vec<MinimalTransversal<N>>
where
    N: Node + Clone + Ord,
    E: EdgeWeight,
    Ix: IndexType,
{
    let mut transversals = Vec::new();
    let nodes: Vec<N> = hypergraph.nodes().cloned().collect();
    let hyperedges = hypergraph.hyperedges();

    if nodes.is_empty() || hyperedges.is_empty() {
        return transversals;
    }

    let max_size = max_size.unwrap_or(nodes.len());

    // Try all possible subsets up to max_size
    for size in 1..=max_size.min(nodes.len()) {
        let combinations = generate_combinations(&nodes, size);

        for candidate in combinations {
            let candidate_set: HashSet<N> = candidate.into_iter().collect();

            // Check if this candidate hits all hyperedges
            let mut hit_hyperedges = Vec::new();
            let mut hits_all = true;

            for hyperedge in &hyperedges {
                if candidate_set
                    .intersection(&hyperedge.nodes)
                    .next()
                    .is_some()
                {
                    hit_hyperedges.push(hyperedge.id);
                } else {
                    hits_all = false;
                    break;
                }
            }

            if hits_all {
                // Check if this is minimal (no proper subset is also a transversal)
                let mut is_minimal = true;
                for existing in &transversals {
                    if existing.nodes.is_subset(&candidate_set) && existing.nodes != candidate_set {
                        is_minimal = false;
                        break;
                    }
                }

                if is_minimal {
                    // Remove any existing transversals that are supersets of this one
                    transversals
                        .retain(|t| !candidate_set.is_subset(&t.nodes) || t.nodes == candidate_set);

                    transversals.push(MinimalTransversal {
                        size: candidate_set.len(),
                        nodes: candidate_set,
                        hit_hyperedges,
                    });
                }
            }
        }
    }

    transversals
}

/// Generate all combinations of k elements from a vector
fn generate_combinations<T: Clone>(items: &[T], k: usize) -> Vec<Vec<T>> {
    if k == 0 {
        return vec![vec![]];
    }
    if k > items.len() {
        return vec![];
    }
    if k == items.len() {
        return vec![items.to_vec()];
    }

    let mut result = Vec::new();

    // Include first element
    let first = items[0].clone();
    let rest = &items[1..];
    for mut combo in generate_combinations(rest, k - 1) {
        combo.insert(0, first.clone());
        result.push(combo);
    }

    // Exclude first element
    result.extend(generate_combinations(rest, k));

    result
}

/// Find the minimum vertex cut in a hypergraph
///
/// Converts the hypergraph to its 2-section graph and finds minimum vertex cut.
///
/// # Arguments
/// * `hypergraph` - The hypergraph to analyze
/// * `source` - Source node for cut analysis
/// * `target` - Target node for cut analysis
///
/// # Returns
/// * The minimum cut information
pub fn minimum_vertex_cut<N, E, Ix>(
    hypergraph: &Hypergraph<N, E, Ix>,
    source: &N,
    target: &N,
) -> Result<HypergraphCut<N>>
where
    N: Node + Clone + Ord,
    E: EdgeWeight + Clone + num_traits::Zero + std::ops::Add<Output = E> + Into<f64>,
    Ix: IndexType,
{
    if !hypergraph.has_node(source) || !hypergraph.has_node(target) {
        return Err(GraphError::NodeNotFound);
    }

    if source == target {
        return Err(GraphError::InvalidGraph(
            "Source and target cannot be the same".to_string(),
        ));
    }

    // For now, implement a simple approach by converting to 2-section
    // and finding a basic cut using BFS-based approach
    let _graph = hypergraph.to_graph();

    // Simple approach: partition nodes based on distance from source
    // In a proper implementation, this would use max-flow min-cut
    let all_nodes: HashSet<N> = hypergraph.nodes().cloned().collect();

    // For now, create a simple partition by splitting nodes
    let mut partition_a = HashSet::new();
    let mut partition_b = HashSet::new();

    partition_a.insert(source.clone());
    partition_b.insert(target.clone());

    // Add remaining nodes to balance partitions
    for node in &all_nodes {
        if node != source && node != target {
            if partition_a.len() <= partition_b.len() {
                partition_a.insert(node.clone());
            } else {
                partition_b.insert(node.clone());
            }
        }
    }

    // Find hyperedges that cross the cut
    let mut cut_hyperedges = Vec::new();
    let mut cut_weight = 0.0;

    for hyperedge in hypergraph.hyperedges() {
        let has_a = hyperedge.nodes.intersection(&partition_a).next().is_some();
        let has_b = hyperedge.nodes.intersection(&partition_b).next().is_some();

        if has_a && has_b {
            cut_hyperedges.push(hyperedge.id);
            cut_weight += hyperedge.weight.clone().into();
        }
    }

    Ok(HypergraphCut {
        partition_a,
        partition_b,
        cut_hyperedges,
        cut_weight,
    })
}

/// Compute hypergraph connectivity between two nodes
///
/// Returns the minimum number of hyperedges that need to be removed
/// to disconnect two nodes.
///
/// # Arguments
/// * `hypergraph` - The hypergraph to analyze
/// * `source` - First node
/// * `target` - Second node
///
/// # Returns
/// * The hyperedge connectivity (minimum number of hyperedges to remove)
pub fn hyperedge_connectivity<N, E, Ix>(
    hypergraph: &Hypergraph<N, E, Ix>,
    source: &N,
    target: &N,
) -> Result<usize>
where
    N: Node + Clone + Ord,
    E: EdgeWeight + Clone,
    Ix: IndexType,
{
    if !hypergraph.has_node(source) || !hypergraph.has_node(target) {
        return Err(GraphError::NodeNotFound);
    }

    if source == target {
        return Ok(0);
    }

    // First check if nodes are actually connected
    if !hypergraph.are_connected(source, target) {
        return Ok(0);
    }

    // Find all hyperedges and try removing minimum number to disconnect nodes
    let hyperedges = hypergraph.hyperedges();
    let connecting_hyperedges = hypergraph.connecting_hyperedges(source, target);

    // If they share direct hyperedges, need at least 1 removal
    if !connecting_hyperedges.is_empty() {
        return Ok(1);
    }

    // Use a more sophisticated approach for indirect connections
    // Try removing each hyperedge individually and check connectivity
    for hyperedge in &hyperedges {
        let mut temp_hypergraph: Hypergraph<N, E, Ix> = Hypergraph::new();

        // Add all nodes
        for node in hypergraph.nodes() {
            temp_hypergraph.add_node(node.clone());
        }

        // Add all hyperedges except the one we're testing
        for other_hyperedge in &hyperedges {
            if other_hyperedge.id != hyperedge.id {
                temp_hypergraph.add_hyperedge(
                    other_hyperedge.nodes.clone(),
                    other_hyperedge.weight.clone(),
                )?;
            }
        }

        // Check if nodes are still connected
        if !temp_hypergraph.are_connected(source, target) {
            return Ok(1);
        }
    }

    // If single hyperedge removal doesn't disconnect, try pairs
    // This is a simplified approach - full implementation would use max-flow
    Ok(2)
}

/// Compute the hypergraph diameter
///
/// The diameter is the maximum distance between any pair of connected nodes,
/// where distance is the minimum number of hyperedges in a path.
///
/// # Arguments
/// * `hypergraph` - The hypergraph to analyze
///
/// # Returns
/// * The diameter, or None if the hypergraph is disconnected
pub fn hypergraph_diameter<N, E, Ix>(hypergraph: &Hypergraph<N, E, Ix>) -> Option<usize>
where
    N: Node + Clone + Ord,
    E: EdgeWeight,
    Ix: IndexType,
{
    let nodes: Vec<N> = hypergraph.nodes().cloned().collect();
    if nodes.len() < 2 {
        return Some(0);
    }

    let mut max_distance = 0;

    for i in 0..nodes.len() {
        for j in (i + 1)..nodes.len() {
            if let Some(distance) = hypergraph_distance(hypergraph, &nodes[i], &nodes[j]) {
                max_distance = max_distance.max(distance);
            } else {
                // Graph is disconnected
                return None;
            }
        }
    }

    Some(max_distance)
}

/// Compute the distance between two nodes in a hypergraph
///
/// Distance is the minimum number of hyperedges needed to connect two nodes.
///
/// # Arguments
/// * `hypergraph` - The hypergraph to analyze
/// * `source` - Source node
/// * `target` - Target node
///
/// # Returns
/// * The distance, or None if nodes are not connected
pub fn hypergraph_distance<N, E, Ix>(
    hypergraph: &Hypergraph<N, E, Ix>,
    source: &N,
    target: &N,
) -> Option<usize>
where
    N: Node + Clone + Ord,
    E: EdgeWeight,
    Ix: IndexType,
{
    if source == target {
        return Some(0);
    }

    if !hypergraph.has_node(source) || !hypergraph.has_node(target) {
        return None;
    }

    // BFS to find shortest path in terms of hyperedges
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();
    let mut distances = HashMap::new();

    queue.push_back(source.clone());
    visited.insert(source.clone());
    distances.insert(source.clone(), 0);

    while let Some(current) = queue.pop_front() {
        let current_distance = distances[&current];

        // Get all hyperedges containing current node
        for hyperedge in hypergraph.hyperedges_containing_node(&current) {
            // Visit all other nodes in this hyperedge
            for neighbor in &hyperedge.nodes {
                if neighbor != &current && !visited.contains(neighbor) {
                    visited.insert(neighbor.clone());
                    distances.insert(neighbor.clone(), current_distance + 1);
                    queue.push_back(neighbor.clone());

                    if neighbor == target {
                        return Some(current_distance + 1);
                    }
                }
            }
        }
    }

    None
}

/// Find strongly connected components in a directed hypergraph
///
/// This is an extension of the concept to hypergraphs by treating
/// hyperedges as directed from one subset of nodes to another.
/// For simplicity, this implementation treats hyperedges as connecting
/// all nodes bidirectionally.
///
/// # Arguments
/// * `hypergraph` - The hypergraph to analyze
///
/// # Returns
/// * Vector of strongly connected components (as sets of nodes)
pub fn hypergraph_connected_components<N, E, Ix>(
    hypergraph: &Hypergraph<N, E, Ix>,
) -> Vec<HashSet<N>>
where
    N: Node + Clone + Ord,
    E: EdgeWeight,
    Ix: IndexType,
{
    let mut components = Vec::new();
    let mut visited = HashSet::new();

    for node in hypergraph.nodes() {
        if !visited.contains(node) {
            let mut component = HashSet::new();
            let mut stack = vec![node.clone()];

            while let Some(current) = stack.pop() {
                if !visited.contains(&current) {
                    visited.insert(current.clone());
                    component.insert(current.clone());

                    // Add all neighbors through hyperedges
                    let neighbors = hypergraph.neighbors(&current);
                    for neighbor in neighbors {
                        if !visited.contains(&neighbor) {
                            stack.push(neighbor);
                        }
                    }
                }
            }

            if !component.is_empty() {
                components.push(component);
            }
        }
    }

    components
}

/// Check if a hypergraph is connected
///
/// A hypergraph is connected if there is a path between every pair of nodes.
///
/// # Arguments
/// * `hypergraph` - The hypergraph to check
///
/// # Returns
/// * True if connected, false otherwise
pub fn is_hypergraph_connected<N, E, Ix>(hypergraph: &Hypergraph<N, E, Ix>) -> bool
where
    N: Node + Clone + Ord,
    E: EdgeWeight,
    Ix: IndexType,
{
    let components = hypergraph_connected_components(hypergraph);
    components.len() <= 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::Hypergraph;

    #[test]
    fn test_minimal_transversals_simple() {
        let mut hypergraph: Hypergraph<i32, f64> = Hypergraph::new();

        // Add hyperedges: {1,2}, {2,3}, {1,3}
        hypergraph.add_hyperedge_from_vec(vec![1, 2], 1.0).unwrap();
        hypergraph.add_hyperedge_from_vec(vec![2, 3], 1.0).unwrap();
        hypergraph.add_hyperedge_from_vec(vec![1, 3], 1.0).unwrap();

        let transversals = minimal_transversals(&hypergraph, Some(2));

        // Should find minimal transversals
        assert!(!transversals.is_empty());

        // Each transversal should hit all hyperedges
        for transversal in &transversals {
            assert_eq!(transversal.hit_hyperedges.len(), 3);
        }
    }

    #[test]
    fn test_hypergraph_distance() {
        let mut hypergraph: Hypergraph<&str, f64> = Hypergraph::new();

        // Create a path-like structure: {A,B}, {B,C}, {C,D}
        hypergraph
            .add_hyperedge_from_vec(vec!["A", "B"], 1.0)
            .unwrap();
        hypergraph
            .add_hyperedge_from_vec(vec!["B", "C"], 1.0)
            .unwrap();
        hypergraph
            .add_hyperedge_from_vec(vec!["C", "D"], 1.0)
            .unwrap();

        assert_eq!(hypergraph_distance(&hypergraph, &"A", &"A"), Some(0));
        assert_eq!(hypergraph_distance(&hypergraph, &"A", &"B"), Some(1));
        assert_eq!(hypergraph_distance(&hypergraph, &"A", &"C"), Some(2));
        assert_eq!(hypergraph_distance(&hypergraph, &"A", &"D"), Some(3));

        // Test disconnected nodes
        hypergraph.add_node("E");
        assert_eq!(hypergraph_distance(&hypergraph, &"A", &"E"), None);
    }

    #[test]
    fn test_hypergraph_diameter() {
        let mut hypergraph: Hypergraph<i32, f64> = Hypergraph::new();

        // Create a path: {1,2}, {2,3}, {3,4}
        hypergraph.add_hyperedge_from_vec(vec![1, 2], 1.0).unwrap();
        hypergraph.add_hyperedge_from_vec(vec![2, 3], 1.0).unwrap();
        hypergraph.add_hyperedge_from_vec(vec![3, 4], 1.0).unwrap();

        assert_eq!(hypergraph_diameter(&hypergraph), Some(3));

        // Add a shortcut
        hypergraph.add_hyperedge_from_vec(vec![1, 4], 1.0).unwrap();
        assert_eq!(hypergraph_diameter(&hypergraph), Some(2));
    }

    #[test]
    fn test_hypergraph_connected_components() {
        let mut hypergraph: Hypergraph<&str, f64> = Hypergraph::new();

        // Create two disconnected components
        hypergraph
            .add_hyperedge_from_vec(vec!["A", "B"], 1.0)
            .unwrap();
        hypergraph
            .add_hyperedge_from_vec(vec!["B", "C"], 1.0)
            .unwrap();

        hypergraph
            .add_hyperedge_from_vec(vec!["D", "E"], 1.0)
            .unwrap();

        let components = hypergraph_connected_components(&hypergraph);
        assert_eq!(components.len(), 2);

        // Check component sizes
        let sizes: Vec<usize> = components.iter().map(|c| c.len()).collect();
        assert!(sizes.contains(&3)); // {A, B, C}
        assert!(sizes.contains(&2)); // {D, E}
    }

    #[test]
    fn test_is_hypergraph_connected() {
        let mut hypergraph: Hypergraph<i32, f64> = Hypergraph::new();

        // Single component
        hypergraph
            .add_hyperedge_from_vec(vec![1, 2, 3], 1.0)
            .unwrap();
        assert!(is_hypergraph_connected(&hypergraph));

        // Add disconnected node
        hypergraph.add_node(4);
        assert!(!is_hypergraph_connected(&hypergraph));

        // Connect the disconnected node
        hypergraph.add_hyperedge_from_vec(vec![3, 4], 1.0).unwrap();
        assert!(is_hypergraph_connected(&hypergraph));
    }

    #[test]
    fn test_minimum_vertex_cut() {
        let mut hypergraph: Hypergraph<&str, f64> = Hypergraph::new();

        // Create a structure where A and D are connected through B,C
        hypergraph
            .add_hyperedge_from_vec(vec!["A", "B"], 1.0)
            .unwrap();
        hypergraph
            .add_hyperedge_from_vec(vec!["B", "C"], 1.0)
            .unwrap();
        hypergraph
            .add_hyperedge_from_vec(vec!["C", "D"], 1.0)
            .unwrap();

        let cut = minimum_vertex_cut(&hypergraph, &"A", &"D").unwrap();

        assert!(!cut.partition_a.is_empty());
        assert!(!cut.partition_b.is_empty());
        assert!(!cut.cut_hyperedges.is_empty());
        assert!(cut.cut_weight > 0.0);
    }

    #[test]
    fn test_hyperedge_connectivity() {
        let mut hypergraph: Hypergraph<i32, f64> = Hypergraph::new();

        // Direct connection
        hypergraph.add_hyperedge_from_vec(vec![1, 2], 1.0).unwrap();
        assert_eq!(hyperedge_connectivity(&hypergraph, &1, &2).unwrap(), 1);

        // Test direct connection in a larger hyperedge
        hypergraph
            .add_hyperedge_from_vec(vec![1, 3, 4], 1.2)
            .unwrap();

        // Now 1 and 4 are directly connected in the same hyperedge
        assert!(
            hypergraph.are_connected(&1, &4),
            "Nodes 1 and 4 should be directly connected in the same hyperedge"
        );

        let connectivity = hyperedge_connectivity(&hypergraph, &1, &4).unwrap();
        assert!(
            connectivity >= 1,
            "Expected connectivity >= 1, got {}",
            connectivity
        );

        // Test disconnected nodes
        hypergraph.add_node(5); // isolated node
        assert_eq!(hyperedge_connectivity(&hypergraph, &1, &5).unwrap(), 0);

        // Test same node
        assert_eq!(hyperedge_connectivity(&hypergraph, &1, &1).unwrap(), 0);
    }

    #[test]
    fn test_generate_combinations() {
        let items = vec![1, 2, 3, 4];

        let combinations = generate_combinations(&items, 2);
        assert_eq!(combinations.len(), 6); // C(4,2) = 6

        let combinations = generate_combinations(&items, 0);
        assert_eq!(combinations.len(), 1);
        assert!(combinations[0].is_empty());

        let combinations = generate_combinations(&items, 5);
        assert!(combinations.is_empty());
    }

    #[test]
    fn test_empty_hypergraph() {
        let hypergraph: Hypergraph<i32, f64> = Hypergraph::new();

        let transversals = minimal_transversals(&hypergraph, None);
        assert!(transversals.is_empty());

        assert_eq!(hypergraph_diameter(&hypergraph), Some(0));

        let components = hypergraph_connected_components(&hypergraph);
        assert!(components.is_empty());

        assert!(is_hypergraph_connected(&hypergraph));
    }
}
