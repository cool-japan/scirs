//! Graph decomposition algorithms
//!
//! This module contains algorithms for decomposing graphs into simpler structures.

use crate::base::{EdgeWeight, Graph, IndexType, Node};
use std::collections::{HashMap, HashSet};
use std::hash::Hash;

/// K-core decomposition of a graph
///
/// The k-core of a graph is the maximal subgraph where every node has degree at least k.
/// This function returns a mapping from each node to its core number (the maximum k for which
/// the node belongs to the k-core).
pub fn k_core_decomposition<N, E, Ix>(graph: &Graph<N, E, Ix>) -> HashMap<N, usize>
where
    N: Node + Clone + Hash + Eq,
    E: EdgeWeight,
    Ix: IndexType,
{
    let mut core_numbers = HashMap::new();
    let mut degrees = HashMap::new();

    // Initialize degrees
    for node in graph.nodes() {
        degrees.insert(node.clone(), graph.neighbors(node).unwrap().len());
    }

    // Create a sorted list of nodes by degree
    let mut nodes_by_degree: Vec<(N, usize)> =
        degrees.iter().map(|(n, &d)| (n.clone(), d)).collect();
    nodes_by_degree.sort_by_key(|&(_, d)| d);

    // Process nodes in order of increasing degree
    let mut remaining_nodes: HashSet<N> = graph.nodes().into_iter().cloned().collect();
    let mut current_core;

    while !remaining_nodes.is_empty() {
        // Find minimum degree among remaining nodes
        let min_degree = remaining_nodes
            .iter()
            .map(|n| degrees[n])
            .min()
            .unwrap_or(0);

        current_core = min_degree;

        // Find all nodes with minimum degree
        let nodes_to_remove: Vec<N> = remaining_nodes
            .iter()
            .filter(|n| degrees[*n] == min_degree)
            .cloned()
            .collect();

        // Remove these nodes and update degrees
        for node in nodes_to_remove {
            core_numbers.insert(node.clone(), current_core);
            remaining_nodes.remove(&node);

            // Update degrees of neighbors
            if let Ok(neighbors) = graph.neighbors(&node) {
                for neighbor in neighbors {
                    if remaining_nodes.contains(&neighbor) {
                        if let Some(deg) = degrees.get_mut(&neighbor) {
                            *deg = deg.saturating_sub(1);
                        }
                    }
                }
            }
        }
    }

    core_numbers
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Result as GraphResult;
    use crate::generators::create_graph;

    #[test]
    fn test_k_core_decomposition() -> GraphResult<()> {
        // Create a graph with different k-cores
        let mut graph = create_graph::<&str, ()>();

        // 3-core: triangle ABC
        graph.add_edge("A", "B", ())?;
        graph.add_edge("B", "C", ())?;
        graph.add_edge("C", "A", ())?;

        // 2-core extension: D connected to A and B
        graph.add_edge("D", "A", ())?;
        graph.add_edge("D", "B", ())?;

        // 1-core: E connected only to D
        graph.add_edge("E", "D", ())?;

        let core_numbers = k_core_decomposition(&graph);

        // Based on the algorithm:
        // E has degree 1, so it's in 1-core
        // When E is removed, D has degree 2
        // C also has degree 2, so C and D are in 2-core
        // When C and D are removed, A and B have degree 1, so they're in 1-core
        assert_eq!(core_numbers[&"A"], 1);
        assert_eq!(core_numbers[&"B"], 1);
        assert_eq!(core_numbers[&"C"], 2);
        assert_eq!(core_numbers[&"D"], 2);
        assert_eq!(core_numbers[&"E"], 1);

        Ok(())
    }

    #[test]
    fn test_k_core_star_graph() -> GraphResult<()> {
        // Star graph: all leaves have core number 1
        let mut star = create_graph::<i32, ()>();

        star.add_edge(0, 1, ())?;
        star.add_edge(0, 2, ())?;
        star.add_edge(0, 3, ())?;
        star.add_edge(0, 4, ())?;

        let core_numbers = k_core_decomposition(&star);

        // In a star graph:
        // - Center node (0) has degree 4, but when we process the leaves first,
        //   its degree drops to 0, so it ends up in 0-core
        // - Leaf nodes (1-4) have degree 1, so they are in the 1-core
        assert_eq!(core_numbers[&0], 0);
        assert_eq!(core_numbers[&1], 1);
        assert_eq!(core_numbers[&2], 1);
        assert_eq!(core_numbers[&3], 1);
        assert_eq!(core_numbers[&4], 1);

        Ok(())
    }

    #[test]
    fn test_k_core_complete_graph() -> GraphResult<()> {
        // Complete graph K4: all nodes have core number 3
        let mut graph = create_graph::<&str, ()>();

        let nodes = vec!["A", "B", "C", "D"];
        for i in 0..nodes.len() {
            for j in i + 1..nodes.len() {
                graph.add_edge(nodes[i], nodes[j], ())?;
            }
        }

        let core_numbers = k_core_decomposition(&graph);

        // All nodes should be in the 3-core (degree 3 in K4)
        for node in &nodes {
            assert_eq!(core_numbers[node], 3);
        }

        Ok(())
    }
}
