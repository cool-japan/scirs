//! Label propagation algorithm for community detection

use super::types::CommunityResult;
use crate::base::{EdgeWeight, Graph, IndexType, Node};
use rand::seq::SliceRandom;
use std::collections::HashMap;
use std::hash::Hash;

/// Internal implementation of label propagation algorithm
#[allow(dead_code)]
fn label_propagation_internal<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    max_iterations: usize,
) -> HashMap<N, usize>
where
    N: Node + Clone + Hash + Eq + std::fmt::Debug,
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
    let mut _iterations = 0;

    while changed && _iterations < max_iterations {
        changed = false;
        _iterations += 1;

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
            let new_label = best_labels[rng.gen_range(0..best_labels.len())];

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

/// Label propagation algorithm for community detection (legacy API)
///
/// **Note**: This function is deprecated in favor of `label_propagation_result`.
/// It will be removed in version 2.0.
///
/// Each node adopts the label that most of its neighbors have, with ties broken randomly.
/// Returns a mapping from nodes to community labels.
///
/// # Time Complexity
/// O(k * m) where k is the number of iterations (typically small) and m is
/// the number of edges. The algorithm often converges in 5-10 iterations.
///
/// # Space Complexity
/// O(n) for storing labels and temporary data structures.
#[deprecated(
    since = "0.1.0-beta.2",
    note = "Use `label_propagation_result` instead"
)]
#[allow(dead_code)]
pub fn label_propagation<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    max_iterations: usize,
) -> HashMap<N, usize>
where
    N: Node + Clone + Hash + Eq + std::fmt::Debug,
    E: EdgeWeight,
    Ix: IndexType,
{
    label_propagation_internal(graph, max_iterations)
}

/// Label propagation algorithm with standardized CommunityResult return type
///
/// This function provides the same functionality as `label_propagation` but returns
/// a standardized `CommunityResult` type that provides multiple ways to access
/// the community structure.
///
/// # Arguments
/// * `graph` - The graph to analyze
/// * `max_iterations` - Maximum number of iterations (default: 100)
///
/// # Returns
/// * A `CommunityResult` with comprehensive community information
///
/// # Time Complexity
/// O(k * m) where k is the number of iterations and m is the number of edges.
/// In practice, the algorithm usually converges in a few iterations.
///
/// # Space Complexity
/// O(n) for storing node labels and community assignments.
///
/// # Example
/// ```rust
/// use scirs2_graph::{Graph, label_propagation_result};
///
/// let mut graph: Graph<i32, f64> = Graph::new();
/// // ... add nodes and edges ...
/// let result = label_propagation_result(&graph, 100);
///
/// println!("Found {} communities", result.num_communities);
/// for (i, community) in result.communities.iter().enumerate() {
///     println!("Community {}: {} members", i, community.len());
/// }
/// ```
#[allow(dead_code)]
pub fn label_propagation_result<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    max_iterations: usize,
) -> CommunityResult<N>
where
    N: Node + Clone + Hash + Eq + std::fmt::Debug,
    E: EdgeWeight,
    Ix: IndexType,
{
    let node_communities = label_propagation_internal(graph, max_iterations);
    CommunityResult::from_node_map(node_communities)
}
