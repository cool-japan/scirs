//! Graph coloring algorithms
//!
//! This module contains algorithms for graph coloring problems.

use crate::base::{EdgeWeight, Graph, Node};
use std::collections::{HashMap, HashSet};

/// Result of graph coloring
#[derive(Debug, Clone)]
pub struct GraphColoring<N: Node> {
    /// The coloring as a map from node to color (0-based)
    pub coloring: HashMap<N, usize>,
    /// The number of colors used
    pub num_colors: usize,
}

/// Colors a graph using a greedy algorithm
///
/// This algorithm assigns colors to vertices one by one, using the smallest
/// available color that hasn't been used by any adjacent vertex.
///
/// # Arguments
/// * `graph` - The graph to color
///
/// # Returns
/// * A graph coloring
pub fn greedy_coloring<N, E, Ix>(graph: &Graph<N, E, Ix>) -> GraphColoring<N>
where
    N: Node,
    E: EdgeWeight,
    Ix: petgraph::graph::IndexType,
{
    let mut coloring: HashMap<N, usize> = HashMap::new();
    let mut max_color = 0;

    // Color nodes in the order they appear
    for node_idx in graph.inner().node_indices() {
        // Find colors used by neighbors
        let mut used_colors = HashSet::new();
        for neighbor_idx in graph.inner().neighbors(node_idx) {
            if let Some(neighbor_node) = graph.inner().node_weight(neighbor_idx) {
                if let Some(&color) = coloring.get(neighbor_node) {
                    used_colors.insert(color);
                }
            }
        }

        // Find the smallest unused color
        let mut color = 0;
        while used_colors.contains(&color) {
            color += 1;
        }

        let node = graph.inner()[node_idx].clone();
        coloring.insert(node, color);
        max_color = max_color.max(color);
    }

    GraphColoring {
        coloring,
        num_colors: max_color + 1,
    }
}

/// Finds the chromatic number of a graph (minimum number of colors needed)
///
/// Uses an exhaustive search up to max_colors to find the minimum coloring.
/// This is an NP-complete problem, so it may be slow for large graphs.
///
/// # Arguments
/// * `graph` - The graph to analyze
/// * `max_colors` - Maximum number of colors to try
///
/// # Returns
/// * The chromatic number if found within max_colors, None otherwise
pub fn chromatic_number<N, E, Ix>(graph: &Graph<N, E, Ix>, max_colors: usize) -> Option<usize>
where
    N: Node,
    E: EdgeWeight,
    Ix: petgraph::graph::IndexType,
{
    if graph.inner().node_count() == 0 {
        return Some(0);
    }

    // Try coloring with 1, 2, 3, ... colors
    (1..=max_colors).find(|&num_colors| can_color_with_k_colors(graph, num_colors))
}

/// Helper function to check if a graph can be colored with k colors
fn can_color_with_k_colors<N, E, Ix>(graph: &Graph<N, E, Ix>, k: usize) -> bool
where
    N: Node,
    E: EdgeWeight,
    Ix: petgraph::graph::IndexType,
{
    let nodes: Vec<_> = graph.inner().node_indices().collect();
    let mut coloring = vec![0; nodes.len()];

    fn backtrack<N, E, Ix>(
        graph: &Graph<N, E, Ix>,
        nodes: &[petgraph::graph::NodeIndex<Ix>],
        coloring: &mut [usize],
        node_idx: usize,
        k: usize,
    ) -> bool
    where
        N: Node,
        E: EdgeWeight,
        Ix: petgraph::graph::IndexType,
    {
        if node_idx == nodes.len() {
            return true;
        }

        let node = nodes[node_idx];

        for color in 0..k {
            // Check if this color is valid (no adjacent node has the same color)
            let mut valid = true;
            for (i, &other_node) in nodes.iter().enumerate().take(node_idx) {
                if (graph.inner().contains_edge(node, other_node)
                    || graph.inner().contains_edge(other_node, node))
                    && coloring[i] == color
                {
                    valid = false;
                    break;
                }
            }

            if valid {
                coloring[node_idx] = color;
                if backtrack(graph, nodes, coloring, node_idx + 1, k) {
                    return true;
                }
            }
        }

        false
    }

    backtrack(graph, &nodes, &mut coloring, 0, k)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generators::create_graph;

    #[test]
    fn test_greedy_coloring() {
        // Create a triangle (needs 3 colors with greedy)
        let mut graph = create_graph::<char, ()>();
        graph.add_edge('A', 'B', ()).unwrap();
        graph.add_edge('B', 'C', ()).unwrap();
        graph.add_edge('C', 'A', ()).unwrap();

        let coloring = greedy_coloring(&graph);
        assert_eq!(coloring.num_colors, 3);

        // Check that adjacent nodes have different colors
        assert_ne!(coloring.coloring[&'A'], coloring.coloring[&'B']);
        assert_ne!(coloring.coloring[&'B'], coloring.coloring[&'C']);
        assert_ne!(coloring.coloring[&'C'], coloring.coloring[&'A']);
    }

    #[test]
    fn test_bipartite_graph_coloring() {
        // Create a bipartite graph (needs only 2 colors)
        let mut graph = create_graph::<i32, ()>();

        // Bipartite structure: 0-1, 0-3, 2-1, 2-3
        graph.add_edge(0, 1, ()).unwrap();
        graph.add_edge(0, 3, ()).unwrap();
        graph.add_edge(2, 1, ()).unwrap();
        graph.add_edge(2, 3, ()).unwrap();

        let coloring = greedy_coloring(&graph);
        assert!(coloring.num_colors <= 2);
    }

    #[test]
    fn test_chromatic_number() {
        // Triangle graph needs exactly 3 colors
        let mut triangle = create_graph::<i32, ()>();

        triangle.add_edge(0, 1, ()).unwrap();
        triangle.add_edge(1, 2, ()).unwrap();
        triangle.add_edge(2, 0, ()).unwrap();

        assert_eq!(chromatic_number(&triangle, 5), Some(3));

        // Bipartite graph needs exactly 2 colors
        let mut bipartite = create_graph::<i32, ()>();

        bipartite.add_edge(0, 1, ()).unwrap();
        bipartite.add_edge(1, 2, ()).unwrap();
        bipartite.add_edge(2, 3, ()).unwrap();
        bipartite.add_edge(3, 0, ()).unwrap();

        assert_eq!(chromatic_number(&bipartite, 5), Some(2));

        // Empty graph needs 0 colors
        let empty = create_graph::<i32, ()>();
        assert_eq!(chromatic_number(&empty, 5), Some(0));
    }
}
