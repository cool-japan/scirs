//! Graph isomorphism and subgraph matching algorithms
//!
//! This module contains algorithms for graph isomorphism testing and subgraph matching.

use crate::base::{EdgeWeight, Graph, IndexType, Node};
use std::collections::HashMap;
use std::hash::Hash;

/// Find all subgraph matches of a pattern graph in a target graph
///
/// Returns a vector of mappings from pattern nodes to target nodes for each match found.
pub fn find_subgraph_matches<N1, N2, E, Ix>(
    pattern: &Graph<N1, E, Ix>,
    target: &Graph<N2, E, Ix>,
) -> Vec<HashMap<N1, N2>>
where
    N1: Node + Clone + Hash + Eq,
    N2: Node + Clone + Hash + Eq,
    E: EdgeWeight,
    Ix: IndexType,
{
    let pattern_nodes: Vec<N1> = pattern.nodes().into_iter().cloned().collect();
    let target_nodes: Vec<N2> = target.nodes().into_iter().cloned().collect();

    if pattern_nodes.is_empty() || pattern_nodes.len() > target_nodes.len() {
        return vec![];
    }

    let mut matches = Vec::new();
    let mut current_mapping = HashMap::new();

    // Try to match starting from each target node
    for start_node in &target_nodes {
        find_matches_recursive(
            &pattern_nodes,
            pattern,
            target,
            &mut current_mapping,
            0,
            start_node,
            &mut matches,
        );
    }

    matches
}

fn find_matches_recursive<N1, N2, E, Ix>(
    pattern_nodes: &[N1],
    pattern: &Graph<N1, E, Ix>,
    target: &Graph<N2, E, Ix>,
    current_mapping: &mut HashMap<N1, N2>,
    pattern_idx: usize,
    target_node: &N2,
    matches: &mut Vec<HashMap<N1, N2>>,
) where
    N1: Node + Clone + Hash + Eq,
    N2: Node + Clone + Hash + Eq,
    E: EdgeWeight,
    Ix: IndexType,
{
    if pattern_idx >= pattern_nodes.len() {
        // Found a complete match
        matches.push(current_mapping.clone());
        return;
    }

    let pattern_node = &pattern_nodes[pattern_idx];

    // Check if target_node is already mapped
    if current_mapping.values().any(|n| n == target_node) {
        return;
    }

    // Try to map pattern_node to target_node
    current_mapping.insert(pattern_node.clone(), target_node.clone());

    // Check if this mapping is consistent with edges
    if is_mapping_consistent(pattern, target, current_mapping) {
        if pattern_idx + 1 < pattern_nodes.len() {
            // Continue mapping with remaining nodes
            if let Ok(target_neighbors) = target.neighbors(target_node) {
                for next_target in target_neighbors {
                    find_matches_recursive(
                        pattern_nodes,
                        pattern,
                        target,
                        current_mapping,
                        pattern_idx + 1,
                        &next_target,
                        matches,
                    );
                }
            }

            // Also try non-neighbors
            for next_target in &target.nodes().into_iter().cloned().collect::<Vec<_>>() {
                if !current_mapping.values().any(|n| n == next_target) {
                    find_matches_recursive(
                        pattern_nodes,
                        pattern,
                        target,
                        current_mapping,
                        pattern_idx + 1,
                        next_target,
                        matches,
                    );
                }
            }
        } else {
            // Last node - check if complete mapping is valid
            find_matches_recursive(
                pattern_nodes,
                pattern,
                target,
                current_mapping,
                pattern_idx + 1,
                target_node,
                matches,
            );
        }
    }

    // Backtrack
    current_mapping.remove(pattern_node);
}

fn is_mapping_consistent<N1, N2, E, Ix>(
    pattern: &Graph<N1, E, Ix>,
    target: &Graph<N2, E, Ix>,
    mapping: &HashMap<N1, N2>,
) -> bool
where
    N1: Node + Hash + Eq,
    N2: Node + Hash + Eq,
    E: EdgeWeight,
    Ix: IndexType,
{
    // Check that all edges in the pattern exist in the target under the mapping
    for (n1, n2) in mapping {
        for (m1, m2) in mapping {
            if n1 != m1 {
                let pattern_has_edge = pattern.has_edge(n1, m1);
                let target_has_edge = target.has_edge(n2, m2);

                if pattern_has_edge && !target_has_edge {
                    return false;
                }
            }
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Result as GraphResult;
    use crate::generators::create_graph;

    #[test]
    fn test_find_subgraph_matches() -> GraphResult<()> {
        // Create a pattern graph (triangle)
        let mut pattern = create_graph::<&str, ()>();
        pattern.add_edge("A", "B", ())?;
        pattern.add_edge("B", "C", ())?;
        pattern.add_edge("C", "A", ())?;

        // Create a target graph with two triangles
        let mut target = create_graph::<&str, ()>();
        // First triangle
        target.add_edge("1", "2", ())?;
        target.add_edge("2", "3", ())?;
        target.add_edge("3", "1", ())?;
        // Second triangle
        target.add_edge("4", "5", ())?;
        target.add_edge("5", "6", ())?;
        target.add_edge("6", "4", ())?;
        // Connect them
        target.add_edge("3", "4", ())?;

        let matches = find_subgraph_matches(&pattern, &target);

        // Should find at least 2 triangles
        assert!(matches.len() >= 2);

        // Each match should have 3 mappings
        for match_map in &matches {
            assert_eq!(match_map.len(), 3);
        }

        Ok(())
    }

    #[test]
    fn test_no_subgraph_match() -> GraphResult<()> {
        // Create a pattern graph (triangle)
        let mut pattern = create_graph::<&str, ()>();
        pattern.add_edge("A", "B", ())?;
        pattern.add_edge("B", "C", ())?;
        pattern.add_edge("C", "A", ())?;

        // Create a target graph with no triangles (path)
        let mut target = create_graph::<&str, ()>();
        target.add_edge("1", "2", ())?;
        target.add_edge("2", "3", ())?;
        target.add_edge("3", "4", ())?;

        let matches = find_subgraph_matches(&pattern, &target);

        // Should find no matches
        assert_eq!(matches.len(), 0);

        Ok(())
    }
}
