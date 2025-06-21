//! Graph isomorphism and subgraph matching algorithms
//!
//! This module contains algorithms for graph isomorphism testing and subgraph matching.
//! Features both a naive backtracking algorithm and the efficient VF2 algorithm.

use crate::base::{EdgeWeight, Graph, IndexType, Node};
use std::collections::{HashMap, HashSet};
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

/// Check if two graphs are isomorphic
///
/// Two graphs are isomorphic if there exists a bijection between their vertices
/// that preserves the edge-adjacency relationship.
///
/// # Arguments
/// * `graph1` - The first graph
/// * `graph2` - The second graph
///
/// # Returns
/// * `bool` - True if the graphs are isomorphic, false otherwise
pub fn are_graphs_isomorphic<N1, N2, E, Ix>(
    graph1: &Graph<N1, E, Ix>,
    graph2: &Graph<N2, E, Ix>,
) -> bool
where
    N1: Node + Clone + Hash + Eq,
    N2: Node + Clone + Hash + Eq,
    E: EdgeWeight,
    Ix: IndexType,
{
    // Quick checks first
    if graph1.node_count() != graph2.node_count() || graph1.edge_count() != graph2.edge_count() {
        return false;
    }

    // Check degree sequence
    if !have_same_degree_sequence(graph1, graph2) {
        return false;
    }

    // If either graph is empty, they're isomorphic
    if graph1.node_count() == 0 {
        return true;
    }

    // Try to find an isomorphism
    find_isomorphism(graph1, graph2).is_some()
}

/// Find an isomorphism between two graphs if one exists
///
/// # Arguments
/// * `graph1` - The first graph
/// * `graph2` - The second graph
///
/// # Returns
/// * `Option<HashMap<N1, N2>>` - Mapping from graph1 nodes to graph2 nodes if isomorphic
pub fn find_isomorphism<N1, N2, E, Ix>(
    graph1: &Graph<N1, E, Ix>,
    graph2: &Graph<N2, E, Ix>,
) -> Option<HashMap<N1, N2>>
where
    N1: Node + Clone + Hash + Eq,
    N2: Node + Clone + Hash + Eq,
    E: EdgeWeight,
    Ix: IndexType,
{
    let nodes1: Vec<N1> = graph1.nodes().into_iter().cloned().collect();
    let nodes2: Vec<N2> = graph2.nodes().into_iter().cloned().collect();

    if nodes1.len() != nodes2.len() {
        return None;
    }

    let mut mapping = HashMap::new();
    if backtrack_isomorphism(&nodes1, &nodes2, graph1, graph2, &mut mapping, 0) {
        Some(mapping)
    } else {
        None
    }
}

/// Check if two graphs have the same degree sequence
fn have_same_degree_sequence<N1, N2, E, Ix>(
    graph1: &Graph<N1, E, Ix>,
    graph2: &Graph<N2, E, Ix>,
) -> bool
where
    N1: Node,
    N2: Node,
    E: EdgeWeight,
    Ix: IndexType,
{
    let mut degrees1: Vec<usize> = graph1
        .nodes()
        .iter()
        .map(|node| {
            graph1
                .neighbors(node)
                .map_or(0, |neighbors| neighbors.len())
        })
        .collect();

    let mut degrees2: Vec<usize> = graph2
        .nodes()
        .iter()
        .map(|node| {
            graph2
                .neighbors(node)
                .map_or(0, |neighbors| neighbors.len())
        })
        .collect();

    degrees1.sort_unstable();
    degrees2.sort_unstable();

    degrees1 == degrees2
}

/// Backtracking algorithm to find isomorphism
fn backtrack_isomorphism<N1, N2, E, Ix>(
    nodes1: &[N1],
    nodes2: &[N2],
    graph1: &Graph<N1, E, Ix>,
    graph2: &Graph<N2, E, Ix>,
    mapping: &mut HashMap<N1, N2>,
    depth: usize,
) -> bool
where
    N1: Node + Clone + Hash + Eq,
    N2: Node + Clone + Hash + Eq,
    E: EdgeWeight,
    Ix: IndexType,
{
    // Base case: all nodes mapped
    if depth == nodes1.len() {
        return is_valid_isomorphism(graph1, graph2, mapping);
    }

    let node1 = &nodes1[depth];

    for node2 in nodes2 {
        // Skip if this node2 is already mapped
        if mapping.values().any(|mapped| mapped == node2) {
            continue;
        }

        // Check degree compatibility
        let degree1 = graph1
            .neighbors(node1)
            .map_or(0, |neighbors| neighbors.len());
        let degree2 = graph2
            .neighbors(node2)
            .map_or(0, |neighbors| neighbors.len());

        if degree1 != degree2 {
            continue;
        }

        // Try this mapping
        mapping.insert(node1.clone(), node2.clone());

        // Check if current partial mapping is consistent
        if is_partial_mapping_valid(graph1, graph2, mapping, depth + 1)
            && backtrack_isomorphism(nodes1, nodes2, graph1, graph2, mapping, depth + 1)
        {
            return true;
        }

        // Backtrack
        mapping.remove(node1);
    }

    false
}

/// Check if a partial mapping is valid (preserves edges among mapped nodes)
fn is_partial_mapping_valid<N1, N2, E, Ix>(
    graph1: &Graph<N1, E, Ix>,
    graph2: &Graph<N2, E, Ix>,
    mapping: &HashMap<N1, N2>,
    _mapped_count: usize,
) -> bool
where
    N1: Node + Hash + Eq,
    N2: Node + Hash + Eq,
    E: EdgeWeight,
    Ix: IndexType,
{
    for (n1, n2) in mapping {
        for (m1, m2) in mapping {
            if n1 != m1 {
                let edge1_exists = graph1.has_edge(n1, m1);
                let edge2_exists = graph2.has_edge(n2, m2);

                if edge1_exists != edge2_exists {
                    return false;
                }
            }
        }
    }
    true
}

/// Check if a complete mapping is a valid isomorphism
fn is_valid_isomorphism<N1, N2, E, Ix>(
    graph1: &Graph<N1, E, Ix>,
    graph2: &Graph<N2, E, Ix>,
    mapping: &HashMap<N1, N2>,
) -> bool
where
    N1: Node + Hash + Eq,
    N2: Node + Hash + Eq,
    E: EdgeWeight,
    Ix: IndexType,
{
    // Check that the mapping preserves all edges
    for (n1, n2) in mapping {
        for (m1, m2) in mapping {
            if n1 != m1 {
                let edge1_exists = graph1.has_edge(n1, m1);
                let edge2_exists = graph2.has_edge(n2, m2);

                if edge1_exists != edge2_exists {
                    return false;
                }
            }
        }
    }
    true
}

/// VF2 Algorithm State for efficient graph isomorphism checking
///
/// The VF2 algorithm maintains state about the current mapping and feasible candidates
/// to efficiently prune the search space and achieve much better performance than naive backtracking.
#[derive(Debug, Clone)]
struct VF2State<N1, N2>
where
    N1: Node + Clone + Hash + Eq,
    N2: Node + Clone + Hash + Eq,
{
    /// Current mapping from graph1 to graph2
    mapping: HashMap<N1, N2>,
    /// Reverse mapping from graph2 to graph1  
    reverse_mapping: HashMap<N2, N1>,
    /// Terminal sets for graph1 (nodes adjacent to mapped nodes)
    terminal_1: HashSet<N1>,
    /// Terminal sets for graph2 (nodes adjacent to mapped nodes)
    terminal_2: HashSet<N2>,
    /// Mapped nodes in graph1
    mapped_1: HashSet<N1>,
    /// Mapped nodes in graph2
    mapped_2: HashSet<N2>,
    /// Depth of current state (number of mapped pairs)
    depth: usize,
}

impl<N1, N2> VF2State<N1, N2>
where
    N1: Node + Clone + Hash + Eq,
    N2: Node + Clone + Hash + Eq,
{
    /// Create a new empty VF2 state
    fn new() -> Self {
        VF2State {
            mapping: HashMap::new(),
            reverse_mapping: HashMap::new(),
            terminal_1: HashSet::new(),
            terminal_2: HashSet::new(),
            mapped_1: HashSet::new(),
            mapped_2: HashSet::new(),
            depth: 0,
        }
    }

    /// Add a new mapping pair to the state
    fn add_pair<E, Ix>(
        &mut self,
        n1: N1,
        n2: N2,
        graph1: &Graph<N1, E, Ix>,
        graph2: &Graph<N2, E, Ix>,
    ) where
        E: EdgeWeight,
        Ix: IndexType,
    {
        self.mapping.insert(n1.clone(), n2.clone());
        self.reverse_mapping.insert(n2.clone(), n1.clone());
        self.mapped_1.insert(n1.clone());
        self.mapped_2.insert(n2.clone());
        self.depth += 1;

        // Update terminal sets by adding neighbors of newly mapped nodes
        if let Ok(neighbors1) = graph1.neighbors(&n1) {
            for neighbor in neighbors1 {
                if !self.mapped_1.contains(&neighbor) {
                    self.terminal_1.insert(neighbor);
                }
            }
        }

        if let Ok(neighbors2) = graph2.neighbors(&n2) {
            for neighbor in neighbors2 {
                if !self.mapped_2.contains(&neighbor) {
                    self.terminal_2.insert(neighbor);
                }
            }
        }

        // Remove newly mapped nodes from terminal sets
        self.terminal_1.remove(&n1);
        self.terminal_2.remove(&n2);
    }

    /// Remove the last mapping pair (backtrack)
    fn remove_pair<E, Ix>(
        &mut self,
        n1: &N1,
        n2: &N2,
        graph1: &Graph<N1, E, Ix>,
        graph2: &Graph<N2, E, Ix>,
    ) where
        E: EdgeWeight,
        Ix: IndexType,
    {
        self.mapping.remove(n1);
        self.reverse_mapping.remove(n2);
        self.mapped_1.remove(n1);
        self.mapped_2.remove(n2);
        self.depth -= 1;

        // Restore terminal sets
        self.update_terminal_sets_after_removal(n1, n2, graph1, graph2);
    }

    /// Update terminal sets after removing a mapping
    fn update_terminal_sets_after_removal<E, Ix>(
        &mut self,
        _n1: &N1,
        _n2: &N2,
        graph1: &Graph<N1, E, Ix>,
        graph2: &Graph<N2, E, Ix>,
    ) where
        E: EdgeWeight,
        Ix: IndexType,
    {
        // Rebuild terminal sets from scratch for simplicity and correctness
        self.terminal_1.clear();
        self.terminal_2.clear();

        for mapped_node in &self.mapped_1 {
            if let Ok(neighbors) = graph1.neighbors(mapped_node) {
                for neighbor in neighbors {
                    if !self.mapped_1.contains(&neighbor) {
                        self.terminal_1.insert(neighbor);
                    }
                }
            }
        }

        for mapped_node in &self.mapped_2 {
            if let Ok(neighbors) = graph2.neighbors(mapped_node) {
                for neighbor in neighbors {
                    if !self.mapped_2.contains(&neighbor) {
                        self.terminal_2.insert(neighbor);
                    }
                }
            }
        }
    }

    /// Get feasible candidate pairs for the next mapping step
    fn get_candidate_pairs<E, Ix>(
        &self,
        graph1: &Graph<N1, E, Ix>,
        graph2: &Graph<N2, E, Ix>,
    ) -> Vec<(N1, N2)>
    where
        E: EdgeWeight,
        Ix: IndexType,
    {
        let mut candidates = Vec::new();

        // Priority 1: Terminal-Terminal pairs (both in terminal sets)
        if !self.terminal_1.is_empty() && !self.terminal_2.is_empty() {
            for n1 in &self.terminal_1 {
                for n2 in &self.terminal_2 {
                    candidates.push((n1.clone(), n2.clone()));
                }
            }
            return candidates;
        }

        // Priority 2: Terminal-Non-terminal pairs
        if !self.terminal_1.is_empty() {
            let all_nodes2: Vec<N2> = graph2.nodes().into_iter().cloned().collect();
            for n1 in &self.terminal_1 {
                for n2 in all_nodes2.iter() {
                    if !self.mapped_2.contains(n2) {
                        candidates.push((n1.clone(), n2.clone()));
                    }
                }
            }
            return candidates;
        }

        if !self.terminal_2.is_empty() {
            let all_nodes1: Vec<N1> = graph1.nodes().into_iter().cloned().collect();
            for n1 in all_nodes1.iter() {
                if !self.mapped_1.contains(n1) {
                    for n2 in &self.terminal_2 {
                        candidates.push((n1.clone(), n2.clone()));
                    }
                }
            }
            return candidates;
        }

        // Priority 3: Any unmapped pair
        let all_nodes1: Vec<N1> = graph1.nodes().into_iter().cloned().collect();
        let all_nodes2: Vec<N2> = graph2.nodes().into_iter().cloned().collect();

        for n1 in all_nodes1.iter() {
            if !self.mapped_1.contains(n1) {
                for n2 in all_nodes2.iter() {
                    if !self.mapped_2.contains(n2) {
                        candidates.push((n1.clone(), n2.clone()));
                        // For efficiency, return first candidate in this case
                        return candidates;
                    }
                }
            }
        }

        candidates
    }

    /// Check if the current state is feasible (VF2 feasibility rules)
    fn is_feasible<E, Ix>(
        &self,
        n1: &N1,
        n2: &N2,
        graph1: &Graph<N1, E, Ix>,
        graph2: &Graph<N2, E, Ix>,
    ) -> bool
    where
        E: EdgeWeight,
        Ix: IndexType,
    {
        // Rule 1: Degree compatibility (basic check)
        let degree1 = graph1.neighbors(n1).map_or(0, |neighbors| neighbors.len());
        let degree2 = graph2.neighbors(n2).map_or(0, |neighbors| neighbors.len());
        if degree1 != degree2 {
            return false;
        }

        // Rule 2: Predecessor rule - check edges to already mapped nodes
        for (mapped_n1, mapped_n2) in &self.mapping {
            let edge1_exists = graph1.has_edge(n1, mapped_n1) || graph1.has_edge(mapped_n1, n1);
            let edge2_exists = graph2.has_edge(n2, mapped_n2) || graph2.has_edge(mapped_n2, n2);

            if edge1_exists != edge2_exists {
                return false;
            }
        }

        // Rule 3: Successor rule - check potential future mappings
        let n1_terminal_neighbors = self.count_terminal_neighbors_1(n1, graph1);
        let n2_terminal_neighbors = self.count_terminal_neighbors_2(n2, graph2);

        if n1_terminal_neighbors != n2_terminal_neighbors {
            return false;
        }

        // Rule 4: Look-ahead rule for unmapped neighbors
        let n1_unmapped_neighbors = self.count_unmapped_neighbors_1(n1, graph1);
        let n2_unmapped_neighbors = self.count_unmapped_neighbors_2(n2, graph2);

        if n1_unmapped_neighbors != n2_unmapped_neighbors {
            return false;
        }

        true
    }

    /// Count neighbors in terminal set for graph1 nodes
    fn count_terminal_neighbors_1<E, Ix>(&self, node: &N1, graph: &Graph<N1, E, Ix>) -> usize
    where
        E: EdgeWeight,
        Ix: IndexType,
    {
        if let Ok(neighbors) = graph.neighbors(node) {
            neighbors
                .into_iter()
                .filter(|n| self.terminal_1.contains(n) && !self.mapped_1.contains(n))
                .count()
        } else {
            0
        }
    }

    /// Count neighbors in terminal set for graph2 nodes
    fn count_terminal_neighbors_2<E, Ix>(&self, node: &N2, graph: &Graph<N2, E, Ix>) -> usize
    where
        E: EdgeWeight,
        Ix: IndexType,
    {
        if let Ok(neighbors) = graph.neighbors(node) {
            neighbors
                .into_iter()
                .filter(|n| self.terminal_2.contains(n) && !self.mapped_2.contains(n))
                .count()
        } else {
            0
        }
    }

    /// Count unmapped neighbors for graph1 nodes (not in terminal or mapped sets)
    fn count_unmapped_neighbors_1<E, Ix>(&self, node: &N1, graph: &Graph<N1, E, Ix>) -> usize
    where
        E: EdgeWeight,
        Ix: IndexType,
    {
        if let Ok(neighbors) = graph.neighbors(node) {
            neighbors
                .into_iter()
                .filter(|n| !self.mapped_1.contains(n) && !self.terminal_1.contains(n))
                .count()
        } else {
            0
        }
    }

    /// Count unmapped neighbors for graph2 nodes (not in terminal or mapped sets)
    fn count_unmapped_neighbors_2<E, Ix>(&self, node: &N2, graph: &Graph<N2, E, Ix>) -> usize
    where
        E: EdgeWeight,
        Ix: IndexType,
    {
        if let Ok(neighbors) = graph.neighbors(node) {
            neighbors
                .into_iter()
                .filter(|n| !self.mapped_2.contains(n) && !self.terminal_2.contains(n))
                .count()
        } else {
            0
        }
    }
}

/// VF2 algorithm for graph isomorphism - enhanced performance version
///
/// This implementation uses the VF2 algorithm which provides significant performance
/// improvements over naive backtracking through intelligent state space exploration
/// and feasibility-based pruning.
pub fn find_isomorphism_vf2<N1, N2, E, Ix>(
    graph1: &Graph<N1, E, Ix>,
    graph2: &Graph<N2, E, Ix>,
) -> Option<HashMap<N1, N2>>
where
    N1: Node + Clone + Hash + Eq,
    N2: Node + Clone + Hash + Eq,
    E: EdgeWeight,
    Ix: IndexType,
{
    // Quick validation checks
    if graph1.node_count() != graph2.node_count() || graph1.edge_count() != graph2.edge_count() {
        return None;
    }

    // Check degree sequence
    if !have_same_degree_sequence(graph1, graph2) {
        return None;
    }

    // Handle empty graphs
    if graph1.node_count() == 0 {
        return Some(HashMap::new());
    }

    let mut state = VF2State::new();
    if vf2_match(&mut state, graph1, graph2) {
        Some(state.mapping)
    } else {
        None
    }
}

/// Core VF2 matching recursive function
fn vf2_match<N1, N2, E, Ix>(
    state: &mut VF2State<N1, N2>,
    graph1: &Graph<N1, E, Ix>,
    graph2: &Graph<N2, E, Ix>,
) -> bool
where
    N1: Node + Clone + Hash + Eq,
    N2: Node + Clone + Hash + Eq,
    E: EdgeWeight,
    Ix: IndexType,
{
    // Base case: all nodes mapped
    if state.depth == graph1.node_count() {
        return true;
    }

    // Get candidate pairs using VF2 ordering heuristics
    let candidates = state.get_candidate_pairs(graph1, graph2);

    for (n1, n2) in candidates {
        // Check VF2 feasibility rules
        if state.is_feasible(&n1, &n2, graph1, graph2) {
            // Add the pair to current state
            state.add_pair(n1.clone(), n2.clone(), graph1, graph2);

            // Recursively try to complete the mapping
            if vf2_match(state, graph1, graph2) {
                return true;
            }

            // Backtrack
            state.remove_pair(&n1, &n2, graph1, graph2);
        }
    }

    false
}

/// Enhanced isomorphism checking using VF2 algorithm with fallback
///
/// This function first attempts to use the efficient VF2 algorithm for isomorphism checking.
/// For very small graphs or edge cases, it may fall back to the simpler backtracking algorithm.
pub fn are_graphs_isomorphic_enhanced<N1, N2, E, Ix>(
    graph1: &Graph<N1, E, Ix>,
    graph2: &Graph<N2, E, Ix>,
) -> bool
where
    N1: Node + Clone + Hash + Eq,
    N2: Node + Clone + Hash + Eq,
    E: EdgeWeight,
    Ix: IndexType,
{
    // For very small graphs, use simple algorithm
    if graph1.node_count() <= 4 {
        return are_graphs_isomorphic(graph1, graph2);
    }

    // For larger graphs, use VF2 for better performance
    find_isomorphism_vf2(graph1, graph2).is_some()
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

    #[test]
    fn test_isomorphic_graphs() -> GraphResult<()> {
        // Create two isomorphic triangles with different node labels
        let mut graph1 = create_graph::<&str, ()>();
        graph1.add_edge("A", "B", ())?;
        graph1.add_edge("B", "C", ())?;
        graph1.add_edge("C", "A", ())?;

        let mut graph2 = create_graph::<i32, ()>();
        graph2.add_edge(1, 2, ())?;
        graph2.add_edge(2, 3, ())?;
        graph2.add_edge(3, 1, ())?;

        assert!(are_graphs_isomorphic(&graph1, &graph2));

        let isomorphism = find_isomorphism(&graph1, &graph2);
        assert!(isomorphism.is_some());

        Ok(())
    }

    #[test]
    fn test_non_isomorphic_graphs() -> GraphResult<()> {
        // Triangle vs path
        let mut triangle = create_graph::<i32, ()>();
        triangle.add_edge(1, 2, ())?;
        triangle.add_edge(2, 3, ())?;
        triangle.add_edge(3, 1, ())?;

        let mut path = create_graph::<i32, ()>();
        path.add_edge(1, 2, ())?;
        path.add_edge(2, 3, ())?;

        assert!(!are_graphs_isomorphic(&triangle, &path));
        assert!(find_isomorphism(&triangle, &path).is_none());

        Ok(())
    }

    #[test]
    fn test_different_size_graphs() -> GraphResult<()> {
        let mut small = create_graph::<i32, ()>();
        small.add_edge(1, 2, ())?;

        let mut large = create_graph::<i32, ()>();
        large.add_edge(1, 2, ())?;
        large.add_edge(2, 3, ())?;

        assert!(!are_graphs_isomorphic(&small, &large));
        assert!(find_isomorphism(&small, &large).is_none());

        Ok(())
    }

    #[test]
    fn test_empty_graphs() {
        let graph1 = create_graph::<i32, ()>();
        let graph2 = create_graph::<&str, ()>();

        assert!(are_graphs_isomorphic(&graph1, &graph2));
        assert!(find_isomorphism(&graph1, &graph2).is_some());
    }

    #[test]
    fn test_vf2_algorithm_triangles() -> GraphResult<()> {
        // Create two isomorphic triangles with different node labels
        let mut graph1 = create_graph::<&str, ()>();
        graph1.add_edge("A", "B", ())?;
        graph1.add_edge("B", "C", ())?;
        graph1.add_edge("C", "A", ())?;

        let mut graph2 = create_graph::<i32, ()>();
        graph2.add_edge(1, 2, ())?;
        graph2.add_edge(2, 3, ())?;
        graph2.add_edge(3, 1, ())?;

        // Test VF2 algorithm
        let vf2_mapping = find_isomorphism_vf2(&graph1, &graph2);
        assert!(vf2_mapping.is_some());

        // Test enhanced function
        assert!(are_graphs_isomorphic_enhanced(&graph1, &graph2));

        // Compare with original algorithm
        assert!(are_graphs_isomorphic(&graph1, &graph2));

        Ok(())
    }

    #[test]
    fn test_vf2_algorithm_larger_graph() -> GraphResult<()> {
        // Create a more complex graph (pentagon with internal connections)
        let mut graph1 = create_graph::<i32, ()>();
        // Pentagon
        graph1.add_edge(1, 2, ())?;
        graph1.add_edge(2, 3, ())?;
        graph1.add_edge(3, 4, ())?;
        graph1.add_edge(4, 5, ())?;
        graph1.add_edge(5, 1, ())?;
        // Internal connections
        graph1.add_edge(1, 3, ())?;
        graph1.add_edge(2, 4, ())?;

        // Create isomorphic graph with different labeling
        let mut graph2 = create_graph::<char, ()>();
        graph2.add_edge('A', 'B', ())?;
        graph2.add_edge('B', 'C', ())?;
        graph2.add_edge('C', 'D', ())?;
        graph2.add_edge('D', 'E', ())?;
        graph2.add_edge('E', 'A', ())?;
        graph2.add_edge('A', 'C', ())?;
        graph2.add_edge('B', 'D', ())?;

        // Test VF2 algorithm
        assert!(find_isomorphism_vf2(&graph1, &graph2).is_some());
        assert!(are_graphs_isomorphic_enhanced(&graph1, &graph2));

        Ok(())
    }

    #[test]
    fn test_vf2_non_isomorphic_graphs() -> GraphResult<()> {
        // Create two non-isomorphic graphs with same number of nodes and edges
        let mut graph1 = create_graph::<i32, ()>(); // Path graph
        graph1.add_edge(1, 2, ())?;
        graph1.add_edge(2, 3, ())?;
        graph1.add_edge(3, 4, ())?;

        let mut graph2 = create_graph::<i32, ()>(); // Star graph
        graph2.add_edge(1, 2, ())?;
        graph2.add_edge(1, 3, ())?;
        graph2.add_edge(1, 4, ())?;

        // Both algorithms should return false
        assert!(!are_graphs_isomorphic(&graph1, &graph2));
        assert!(!are_graphs_isomorphic_enhanced(&graph1, &graph2));
        assert!(find_isomorphism_vf2(&graph1, &graph2).is_none());

        Ok(())
    }

    #[test]
    fn test_vf2_single_node_graphs() -> GraphResult<()> {
        let mut graph1 = create_graph::<&str, ()>();
        graph1.add_node("A");

        let mut graph2 = create_graph::<i32, ()>();
        graph2.add_node(1);

        assert!(find_isomorphism_vf2(&graph1, &graph2).is_some());
        assert!(are_graphs_isomorphic_enhanced(&graph1, &graph2));

        Ok(())
    }

    #[test]
    fn test_vf2_empty_graphs() {
        let graph1 = create_graph::<i32, ()>();
        let graph2 = create_graph::<&str, ()>();

        assert!(find_isomorphism_vf2(&graph1, &graph2).is_some());
        assert!(are_graphs_isomorphic_enhanced(&graph1, &graph2));
    }

    #[test]
    fn test_vf2_algorithm_performance_comparison() -> GraphResult<()> {
        // Create a larger graph to test performance characteristics
        let mut graph1 = create_graph::<i32, ()>();
        let mut graph2 = create_graph::<i32, ()>();

        // Create two isomorphic complete graphs K4
        for i in 1..=4 {
            for j in (i + 1)..=4 {
                graph1.add_edge(i, j, ())?;
                graph2.add_edge(i, j, ())?; // Same structure for simplicity
            }
        }

        // Both algorithms should find isomorphism
        let naive_result = are_graphs_isomorphic(&graph1, &graph2);
        let vf2_result = are_graphs_isomorphic_enhanced(&graph1, &graph2);

        assert_eq!(naive_result, vf2_result);
        assert!(naive_result); // Should be isomorphic

        Ok(())
    }

    #[test]
    fn test_vf2_different_degree_sequences() -> GraphResult<()> {
        // Create graphs with different degree sequences (quick rejection test)
        let mut graph1 = create_graph::<i32, ()>(); // Star graph: 1 connects to 2,3
        graph1.add_edge(1, 2, ())?;
        graph1.add_edge(1, 3, ())?;
        // Degree sequence: [2, 1, 1] - node 1 has degree 2, nodes 2,3 have degree 1

        let mut graph2 = create_graph::<i32, ()>(); // Path graph: 1-2-3
        graph2.add_edge(1, 2, ())?;
        graph2.add_edge(2, 3, ())?;
        // Degree sequence: [1, 2, 1] - node 2 has degree 2, nodes 1,3 have degree 1

        // Both have same degree sequence [2,1,1] when sorted, so they might be isomorphic
        // Let's create truly different degree sequences instead
        let mut graph3 = create_graph::<i32, ()>(); // Triangle
        graph3.add_edge(1, 2, ())?;
        graph3.add_edge(2, 3, ())?;
        graph3.add_edge(3, 1, ())?;
        // Degree sequence: [2, 2, 2] - all nodes have degree 2

        // Compare star vs triangle (different degree sequences)
        assert!(!are_graphs_isomorphic(&graph1, &graph3));
        assert!(!are_graphs_isomorphic_enhanced(&graph1, &graph3));
        assert!(find_isomorphism_vf2(&graph1, &graph3).is_none());

        Ok(())
    }
}
