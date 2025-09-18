//! Fluid communities algorithm for community detection

use super::types::{CommunityResult, CommunityStructure};
use crate::base::{EdgeWeight, Graph, IndexType, Node};
use std::collections::HashMap;
use std::hash::Hash;

/// Fluid communities algorithm (legacy API)
///
/// **Note**: This function is deprecated in favor of `fluid_communities_result`.
/// It will be removed in version 2.0.
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
#[deprecated(
    since = "0.1.0-beta.2",
    note = "Use `fluid_communities_result` instead"
)]
#[allow(dead_code)]
pub fn fluid_communities<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    num_communities: usize,
    max_iterations: usize,
) -> CommunityStructure<N>
where
    N: Node + Clone + Hash + Eq + std::fmt::Debug,
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
        let initial_fluid = rng.gen_range(0..num_communities);
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
                let random_fluid = rng.gen_range(0..num_communities);
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
    let mod_score = super::modularity(graph, &communities);

    CommunityStructure {
        node_communities: communities,
        modularity: mod_score,
    }
}

/// Fluid communities algorithm with standardized CommunityResult return type
///
/// This function provides the same functionality as `fluid_communities` but returns
/// a standardized `CommunityResult` type that provides multiple ways to access
/// the community structure.
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
/// * A `CommunityResult` with comprehensive community information
///
/// # Time Complexity
/// O(k * m * c) where k is the number of iterations, m is the number of edges,
/// and c is the target number of communities. Each iteration involves fluid
/// propagation across all edges and community density calculations.
///
/// # Space Complexity
/// O(n + c) for storing node assignments, fluid densities per community,
/// and tracking community membership changes.
///
/// # Example
/// ```rust
/// use scirs2_graph::{Graph, fluid_communities_result};
///
/// let mut graph: Graph<i32, f64> = Graph::new();
/// // ... add nodes and edges ...
/// let result = fluid_communities_result(&graph, 5, 100);
///
/// println!("Found {} communities", result.num_communities);
/// for (i, community) in result.communities.iter().enumerate() {
///     println!("Community {}: {} members", i, community.len());
/// }
/// ```
#[allow(dead_code)]
pub fn fluid_communities_result<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    num_communities: usize,
    max_iterations: usize,
) -> CommunityResult<N>
where
    N: Node + Clone + Hash + Eq + std::fmt::Debug,
    E: EdgeWeight + Into<f64> + Copy,
    Ix: IndexType,
{
    #[allow(deprecated)]
    let structure = fluid_communities(graph, num_communities, max_iterations);
    let mut result = CommunityResult::from_node_map(structure.node_communities.clone());

    // Calculate and set the quality score (modularity)
    let mod_score = super::modularity(graph, &structure.node_communities);
    result.quality_score = Some(mod_score);
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Result as GraphResult;
    use crate::generators::create_graph;

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

        let result = fluid_communities_result(&graph, 2, 30);

        // Check that all nodes are assigned to communities
        assert_eq!(result.node_communities.len(), 6);

        // Check that we found the expected number of communities (should be <= 2)
        assert!(result.num_communities <= 2);
        assert!(result.num_communities > 0);

        // Check that quality score was calculated
        assert!(result.quality_score.is_some());

        Ok(())
    }

    #[test]
    fn test_fluid_communities_empty_graph() {
        let graph = create_graph::<i32, f64>();
        let result = fluid_communities_result(&graph, 2, 10);

        assert!(result.node_communities.is_empty());
        assert_eq!(result.quality_score.unwrap_or(0.0), 0.0);
    }

    #[test]
    fn test_fluid_communities_single_node() -> GraphResult<()> {
        let mut graph = create_graph::<&str, f64>();
        graph.add_node("A");

        let result = fluid_communities_result(&graph, 1, 10);

        assert_eq!(result.node_communities.len(), 1);
        assert!(result.node_communities.contains_key(&"A"));
        assert_eq!(result.node_communities[&"A"], 0);

        Ok(())
    }
}
