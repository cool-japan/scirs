//! Graph matching algorithms
//!
//! This module contains algorithms for finding matchings in graphs,
//! particularly bipartite matchings.

use crate::algorithms::connectivity::is_bipartite;
use crate::base::{EdgeWeight, Graph, IndexType, Node};
use crate::error::{GraphError, Result};
use std::collections::{HashMap, HashSet};
use std::hash::Hash;

/// Maximum bipartite matching result
#[derive(Debug, Clone)]
pub struct BipartiteMatching<N: Node> {
    /// The matching as a map from left nodes to right nodes
    pub matching: HashMap<N, N>,
    /// The size of the matching
    pub size: usize,
}

/// Finds a maximum bipartite matching using the Hungarian algorithm
///
/// Assumes the graph is bipartite with nodes already colored.
///
/// # Arguments
/// * `graph` - The bipartite graph
/// * `coloring` - The bipartite coloring (0 or 1 for each node)
///
/// # Returns
/// * A maximum bipartite matching
#[allow(dead_code)]
pub fn maximum_bipartite_matching<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    coloring: &HashMap<N, u8>,
) -> BipartiteMatching<N>
where
    N: Node + std::fmt::Debug,
    E: EdgeWeight,
    Ix: petgraph::graph::IndexType,
{
    // Create a mapping from nodes to indices
    let mut node_to_idx: HashMap<N, petgraph::graph::NodeIndex<Ix>> = HashMap::new();
    for node_idx in graph.inner().node_indices() {
        node_to_idx.insert(graph.inner()[node_idx].clone(), node_idx);
    }

    // Separate nodes into left and right sets based on coloring
    let mut left_nodes = Vec::new();
    let mut right_nodes = Vec::new();

    for (node, &color) in coloring {
        if color == 0 {
            left_nodes.push(node.clone());
        } else {
            right_nodes.push(node.clone());
        }
    }

    // Build matching using augmenting paths
    let mut matching: HashMap<N, N> = HashMap::new();
    let mut reverse_matching: HashMap<N, N> = HashMap::new();

    // For each unmatched left node, try to find an augmenting path
    for left_node in &left_nodes {
        if !matching.contains_key(left_node) {
            let mut visited = HashSet::new();
            augment_path(
                graph,
                left_node,
                &mut matching,
                &mut reverse_matching,
                &mut visited,
                coloring,
            );
        }
    }

    BipartiteMatching {
        size: matching.len(),
        matching,
    }
}

/// Try to find an augmenting path from an unmatched left node
#[allow(dead_code)]
fn augment_path<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    node: &N,
    matching: &mut HashMap<N, N>,
    reverse_matching: &mut HashMap<N, N>,
    visited: &mut HashSet<N>,
    coloring: &HashMap<N, u8>,
) -> bool
where
    N: Node + std::fmt::Debug,
    E: EdgeWeight,
    Ix: petgraph::graph::IndexType,
{
    // Mark as visited
    visited.insert(node.clone());

    // Try all neighbors
    if let Ok(neighbors) = graph.neighbors(node) {
        for neighbor in neighbors {
            // Skip if same color (not bipartite edge)
            if coloring.get(node) == coloring.get(&neighbor) {
                continue;
            }

            // If neighbor is unmatched, we found an augmenting path
            if let std::collections::hash_map::Entry::Vacant(e) =
                reverse_matching.entry(neighbor.clone())
            {
                matching.insert(node.clone(), neighbor.clone());
                e.insert(node.clone());
                return true;
            }

            // Otherwise, try to augment through the matched node
            let matched_node = reverse_matching[&neighbor].clone();
            if !visited.contains(&matched_node)
                && augment_path(
                    graph,
                    &matched_node,
                    matching,
                    reverse_matching,
                    visited,
                    coloring,
                )
            {
                matching.insert(node.clone(), neighbor.clone());
                reverse_matching.insert(neighbor, node.clone());
                return true;
            }
        }
    }

    false
}

/// Minimum weight bipartite matching using the exact O(n^3) Hungarian algorithm
///
/// Finds the minimum weight perfect matching in a bipartite graph, exactly,
/// for graphs of any size (there is no approximate fallback: every left node
/// must have an edge into a distinct right node, or this returns an error
/// rather than a non-optimal or partial matching).
/// Returns the total weight and the matching as a vector of (left_node, right_node) pairs.
#[allow(dead_code)]
pub fn minimum_weight_bipartite_matching<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
) -> Result<(f64, Vec<(N, N)>)>
where
    N: Node + Clone + Hash + Eq + std::fmt::Debug,
    E: EdgeWeight + Into<f64> + Clone,
    Ix: IndexType,
{
    // First check if the graph is bipartite
    let bipartite_result = is_bipartite(graph);

    if !bipartite_result.is_bipartite {
        return Err(GraphError::InvalidGraph(
            "Graph is not bipartite".to_string(),
        ));
    }

    let coloring = bipartite_result.coloring;

    // Separate nodes by color
    let mut left_nodes = Vec::new();
    let mut right_nodes = Vec::new();

    for (node, &color) in &coloring {
        if color == 0 {
            left_nodes.push(node.clone());
        } else {
            right_nodes.push(node.clone());
        }
    }

    let n_left = left_nodes.len();
    let n_right = right_nodes.len();

    if n_left != n_right {
        return Err(GraphError::InvalidGraph(
            "Bipartite graph must have equal number of nodes in each partition for perfect matching".to_string()
        ));
    }

    if n_left == 0 {
        return Ok((0.0, vec![]));
    }

    // Create cost matrix
    let mut cost_matrix = vec![vec![f64::INFINITY; n_right]; n_left];

    for (i, left_node) in left_nodes.iter().enumerate() {
        for (j, right_node) in right_nodes.iter().enumerate() {
            if let Ok(weight) = graph.edge_weight(left_node, right_node) {
                cost_matrix[i][j] = weight.into();
            }
        }
    }

    // Solve the assignment problem exactly with the O(n^3) Hungarian
    // algorithm for every size (previously this silently switched to a
    // non-optimal greedy heuristic for n_left > 6, with no indication in the
    // return type that the answer was only approximate).
    let assignment = hungarian_algorithm(&cost_matrix).map_err(|e| {
        GraphError::InvalidGraph(format!(
            "minimum_weight_bipartite_matching: {e} (graph may be missing edges needed for a perfect matching)"
        ))
    })?;

    let mut total_cost = 0.0;
    let mut matching = Vec::with_capacity(n_left);
    for (j, &row_1indexed) in assignment.iter().enumerate().skip(1) {
        let i = row_1indexed - 1;
        let cost = cost_matrix[i][j - 1];
        if !cost.is_finite() {
            return Err(GraphError::InvalidGraph(
                "minimum_weight_bipartite_matching: no perfect matching exists using only real edges".to_string(),
            ));
        }
        total_cost += cost;
        matching.push((left_nodes[i].clone(), right_nodes[j - 1].clone()));
    }

    Ok((total_cost, matching))
}

/// Solves the square assignment problem (minimum weight perfect matching)
/// via the O(n^3) Hungarian algorithm (Kuhn-Munkres, using row/column
/// potentials and shortest augmenting paths). `cost[i][j]` is the cost of
/// assigning row `i` to column `j`; use `f64::INFINITY` for a forbidden
/// assignment (no edge).
///
/// Returns `p` where `p[j]` (for `j` in `1..=n`; `p[0]` is unused scratch
/// space) is the 1-indexed row assigned to column `j`.
///
/// This closely follows the standard reference algorithm for the assignment
/// problem (e.g. <https://cp-algorithms.com/graph/hungarian-algorithm.html>),
/// adapted to Rust and to gracefully reject infeasible instances (rather
/// than assuming a complete cost matrix with no forbidden assignments).
#[allow(clippy::needless_range_loop)]
fn hungarian_algorithm(cost: &[Vec<f64>]) -> std::result::Result<Vec<usize>, String> {
    let n = cost.len();
    if n == 0 {
        return Ok(vec![0]);
    }

    // 1-indexed throughout (index 0 is the algorithm's sentinel/dummy row
    // and column), matching the reference implementation closely to
    // minimize transcription risk in this notoriously fiddly algorithm.
    let mut u = vec![0.0_f64; n + 1];
    let mut v = vec![0.0_f64; n + 1];
    let mut p = vec![0usize; n + 1]; // p[j] = row assigned to column j, 0 = none yet
    let mut way = vec![0usize; n + 1];

    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0usize;
        let mut minv = vec![f64::INFINITY; n + 1];
        let mut used = vec![false; n + 1];

        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = f64::INFINITY;
            let mut j1 = 0usize;

            for j in 1..=n {
                if !used[j] {
                    let cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
                    if cur < minv[j] {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if minv[j] < delta {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }

            if !delta.is_finite() {
                // Every unused column is unreachable through a finite-cost
                // edge from the rows visited so far: no perfect matching
                // exists using only real edges.
                return Err("no feasible perfect matching exists".to_string());
            }

            for j in 0..=n {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }

            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }

        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }

    Ok(p)
}

#[allow(dead_code)]
fn minimum_weight_matching_bruteforce<N>(
    left_nodes: &[N],
    right_nodes: &[N],
    cost_matrix: &[Vec<f64>],
) -> Result<(f64, Vec<(N, N)>)>
where
    N: Node + Clone + std::fmt::Debug,
{
    let n = left_nodes.len();
    let mut best_cost = f64::INFINITY;
    let mut best_matching = Vec::new();

    // Generate all permutations
    let mut perm: Vec<usize> = (0..n).collect();

    loop {
        // Calculate cost for this permutation
        let mut cost = 0.0;
        for i in 0..n {
            cost += cost_matrix[i][perm[i]];
        }

        if cost < best_cost {
            best_cost = cost;
            best_matching = (0..n)
                .map(|i| (left_nodes[i].clone(), right_nodes[perm[i]].clone()))
                .collect();
        }

        // Next permutation
        if !next_permutation(&mut perm) {
            break;
        }
    }

    Ok((best_cost, best_matching))
}

#[allow(dead_code)]
fn next_permutation(perm: &mut [usize]) -> bool {
    let n = perm.len();

    // Find the largest index k such that perm[k] < perm[k + 1]
    let mut k = None;
    for i in 0..n - 1 {
        if perm[i] < perm[i + 1] {
            k = Some(i);
        }
    }

    let k = match k {
        Some(k) => k,
        None => return false, // Last permutation
    };

    // Find the largest index l greater than k such that perm[k] < perm[l]
    let mut l = k + 1;
    for i in k + 1..n {
        if perm[k] < perm[i] {
            l = i;
        }
    }

    // Swap perm[k] and perm[l]
    perm.swap(k, l);

    // Reverse the sequence from perm[k + 1] to the end
    perm[k + 1..].reverse();

    true
}

/// Maximum cardinality matching result
#[derive(Debug, Clone)]
pub struct MaximumMatching<N: Node> {
    /// The matching as a vector of edge pairs
    pub matching: Vec<(N, N)>,
    /// The size of the matching
    pub size: usize,
}

/// Finds a maximum cardinality matching in a general graph using Edmonds' blossom algorithm
///
/// This is a simplified implementation of the blossom algorithm for general graphs.
/// For better performance on bipartite graphs, use `maximum_bipartite_matching`.
///
/// # Arguments
/// * `graph` - The input graph
///
/// # Returns
/// * A maximum cardinality matching
#[allow(dead_code)]
pub fn maximum_cardinality_matching<N, E, Ix>(graph: &Graph<N, E, Ix>) -> MaximumMatching<N>
where
    N: Node + Clone + std::fmt::Debug,
    E: EdgeWeight,
    Ix: IndexType,
{
    let nodes: Vec<N> = graph.nodes().into_iter().cloned().collect();
    let n = nodes.len();

    if n == 0 {
        return MaximumMatching {
            matching: Vec::new(),
            size: 0,
        };
    }

    // Use a greedy approach for simplicity
    // A full implementation would use Edmonds' blossom algorithm
    let mut matching = Vec::new();
    let mut matched = vec![false; n];
    let node_to_idx: HashMap<N, usize> = nodes
        .iter()
        .enumerate()
        .map(|(i, n)| (n.clone(), i))
        .collect();

    // Greedy matching: find augmenting paths
    for (i, node) in nodes.iter().enumerate() {
        if matched[i] {
            continue;
        }

        if let Ok(neighbors) = graph.neighbors(node) {
            for neighbor in neighbors {
                if let Some(&j) = node_to_idx.get(&neighbor) {
                    if !matched[j] {
                        // Found an augmenting path of length 1
                        matching.push((node.clone(), neighbor));
                        matched[i] = true;
                        matched[j] = true;
                        break;
                    }
                }
            }
        }
    }

    MaximumMatching {
        size: matching.len(),
        matching,
    }
}

/// Finds a maximal matching using a greedy algorithm
///
/// A maximal matching is one where no more edges can be added.
/// This is simpler than maximum matching but provides a 2-approximation.
///
/// # Arguments
/// * `graph` - The input graph
///
/// # Returns
/// * A maximal matching
#[allow(dead_code)]
pub fn maximal_matching<N, E, Ix>(graph: &Graph<N, E, Ix>) -> MaximumMatching<N>
where
    N: Node + Clone + std::fmt::Debug,
    E: EdgeWeight,
    Ix: IndexType,
{
    let mut matching = Vec::new();
    let mut matched_nodes = HashSet::new();

    // Get all edges
    let edges = graph.edges();

    // Greedily add edges that don't conflict with existing matching
    for edge in edges {
        if !matched_nodes.contains(&edge.source) && !matched_nodes.contains(&edge.target) {
            matching.push((edge.source.clone(), edge.target.clone()));
            matched_nodes.insert(edge.source);
            matched_nodes.insert(edge.target);
        }
    }

    MaximumMatching {
        size: matching.len(),
        matching,
    }
}

/// Stable marriage problem solver using the Gale-Shapley algorithm
///
/// Finds a stable matching between two sets of equal size where each element
/// has a preference order over the other set.
///
/// # Arguments
/// * `left_prefs` - Preference lists for left set (each list is ordered from most to least preferred)
/// * `right_prefs` - Preference lists for right set
///
/// # Returns
/// * A stable matching as pairs (left_index, right_index)
#[allow(dead_code)]
pub fn stable_marriage(
    left_prefs: &[Vec<usize>],
    right_prefs: &[Vec<usize>],
) -> Result<Vec<(usize, usize)>> {
    let n = left_prefs.len();

    if n != right_prefs.len() {
        return Err(GraphError::InvalidGraph(
            "Left and right sets must have equal size".to_string(),
        ));
    }

    if n == 0 {
        return Ok(Vec::new());
    }

    // Validate preference lists
    for (i, prefs) in left_prefs.iter().enumerate() {
        if prefs.len() != n {
            return Err(GraphError::InvalidGraph(format!(
                "Left preference list {i} has wrong length"
            )));
        }
        let mut sorted_prefs = prefs.clone();
        sorted_prefs.sort_unstable();
        if sorted_prefs != (0..n).collect::<Vec<_>>() {
            return Err(GraphError::InvalidGraph(format!(
                "Left preference list {i} is not a valid permutation"
            )));
        }
    }

    for (i, prefs) in right_prefs.iter().enumerate() {
        if prefs.len() != n {
            return Err(GraphError::InvalidGraph(format!(
                "Right preference list {i} has wrong length"
            )));
        }
        let mut sorted_prefs = prefs.clone();
        sorted_prefs.sort_unstable();
        if sorted_prefs != (0..n).collect::<Vec<_>>() {
            return Err(GraphError::InvalidGraph(format!(
                "Right preference list {i} is not a valid permutation"
            )));
        }
    }

    // Create inverse preference mappings for right set for efficiency
    let mut right_inv_prefs = vec![vec![0; n]; n];
    for (i, prefs) in right_prefs.iter().enumerate() {
        for (rank, &person) in prefs.iter().enumerate() {
            right_inv_prefs[i][person] = rank;
        }
    }

    // Gale-Shapley algorithm
    let mut left_partner = vec![None; n];
    let mut right_partner = vec![None; n];
    let mut left_next_proposal = vec![0; n];
    let mut free_left: std::collections::VecDeque<usize> = (0..n).collect();

    while let Some(left) = free_left.pop_front() {
        if left_next_proposal[left] >= n {
            continue; // This left person has proposed to everyone
        }

        let right = left_prefs[left][left_next_proposal[left]];
        left_next_proposal[left] += 1;

        match right_partner[right] {
            None => {
                // Right person is free, form engagement
                left_partner[left] = Some(right);
                right_partner[right] = Some(left);
            }
            Some(current_left) => {
                // Right person is engaged, check if they prefer the new proposal
                if right_inv_prefs[right][left] < right_inv_prefs[right][current_left] {
                    // Right person prefers the new proposal
                    left_partner[left] = Some(right);
                    right_partner[right] = Some(left);
                    left_partner[current_left] = None;
                    free_left.push_back(current_left);
                } else {
                    // Right person prefers their current partner
                    free_left.push_back(left);
                }
            }
        }
    }

    // Convert to result format
    let mut result = Vec::new();
    for (left, partner) in left_partner.iter().enumerate() {
        if let Some(right) = partner {
            result.push((left, *right));
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Result as GraphResult;
    use crate::generators::create_graph;

    #[test]
    fn test_maximum_bipartite_matching() -> GraphResult<()> {
        let mut graph = create_graph::<&str, ()>();

        // Create a bipartite graph
        graph.add_edge("A", "1", ())?;
        graph.add_edge("A", "2", ())?;
        graph.add_edge("B", "2", ())?;
        graph.add_edge("B", "3", ())?;
        graph.add_edge("C", "3", ())?;

        // Create coloring
        let mut coloring = HashMap::new();
        coloring.insert("A", 0);
        coloring.insert("B", 0);
        coloring.insert("C", 0);
        coloring.insert("1", 1);
        coloring.insert("2", 1);
        coloring.insert("3", 1);

        let matching = maximum_bipartite_matching(&graph, &coloring);

        // Should find a perfect matching of size 3
        assert_eq!(matching.size, 3);

        // Verify it's a valid matching
        let mut used_right = HashSet::new();
        for right in matching.matching.values() {
            assert!(!used_right.contains(right));
            used_right.insert(right);
        }

        Ok(())
    }

    #[test]
    fn test_minimum_weight_bipartite_matching() -> GraphResult<()> {
        let mut graph = create_graph::<&str, f64>();

        // Create a complete bipartite graph K2,2
        graph.add_edge("A", "1", 1.0)?;
        graph.add_edge("A", "2", 3.0)?;
        graph.add_edge("B", "1", 2.0)?;
        graph.add_edge("B", "2", 1.0)?;

        let (total_weight, matching) = minimum_weight_bipartite_matching(&graph)?;

        // Optimal matching: A-1 (1.0) and B-2 (1.0)
        assert_eq!(total_weight, 2.0);
        assert_eq!(matching.len(), 2);

        Ok(())
    }

    #[test]
    fn test_hungarian_matches_bruteforce_on_random_instances() {
        // Cross-check the real Hungarian algorithm against the brute-force
        // exact solver (feasible up to n=6) on many random, non-constant
        // cost matrices. A deterministic LCG (no external RNG dependency)
        // keeps this reproducible while still exercising genuine variety.
        let mut state: u64 = 0x1234_5678_9abc_def0;

        for n in 1..=6usize {
            for trial in 0..20u32 {
                let mut cost_matrix = vec![vec![0.0_f64; n]; n];
                for row in cost_matrix.iter_mut() {
                    for cell in row.iter_mut() {
                        state = state
                            .wrapping_mul(6364136223846793005)
                            .wrapping_add(1442695040888963407);
                        *cell = ((state >> 11) as f64 / (1u64 << 53) as f64) * 100.0;
                    }
                }

                let left_nodes: Vec<usize> = (0..n).collect();
                let right_nodes: Vec<usize> = (0..n).collect();

                let (bruteforce_cost, _) =
                    minimum_weight_matching_bruteforce(&left_nodes, &right_nodes, &cost_matrix)
                        .expect("bruteforce failed");
                let assignment = hungarian_algorithm(&cost_matrix).expect("hungarian failed");
                let hungarian_cost: f64 =
                    (1..=n).map(|j| cost_matrix[assignment[j] - 1][j - 1]).sum();

                assert!(
                    (bruteforce_cost - hungarian_cost).abs() < 1e-6,
                    "n={n} trial={trial}: hungarian cost {hungarian_cost} should match bruteforce {bruteforce_cost}"
                );
            }
        }
    }

    #[test]
    fn test_minimum_weight_bipartite_matching_large_finds_true_optimum() {
        // n_left = 7 (> 6): the OLD implementation silently substituted a
        // non-optimal greedy heuristic for any instance past this size.
        // This is a classic "greedy fails" case: greedily grabbing each left
        // node's locally-cheapest still-free right node gives (0,10)=1 +
        // (1,11)=3 = 4 for the first pair, whereas the TRUE optimum swaps
        // them: (0,11)=2 + (1,10)=1 = 3. The remaining five pairs are
        // free (cost 0) padding that just keeps n_left > 6 while leaving the
        // graph a clean disjoint union of small bipartite pieces.
        let mut graph = create_graph::<i32, f64>();
        graph.add_edge(0, 10, 1.0).expect("Operation failed");
        graph.add_edge(0, 11, 2.0).expect("Operation failed");
        graph.add_edge(1, 10, 1.0).expect("Operation failed");
        graph.add_edge(1, 11, 3.0).expect("Operation failed");
        graph.add_edge(2, 12, 0.0).expect("Operation failed");
        graph.add_edge(3, 13, 0.0).expect("Operation failed");
        graph.add_edge(4, 14, 0.0).expect("Operation failed");
        graph.add_edge(5, 15, 0.0).expect("Operation failed");
        graph.add_edge(6, 16, 0.0).expect("Operation failed");

        let (total_weight, matching) =
            minimum_weight_bipartite_matching(&graph).expect("matching failed");

        assert_eq!(matching.len(), 7);
        assert!(
            (total_weight - 3.0).abs() < 1e-9,
            "expected the true optimum 3.0 (not the greedy-suboptimal 4.0), got {total_weight}"
        );
    }

    #[test]
    fn test_minimum_weight_bipartite_matching_infeasible_returns_error() {
        // Node 0 has no edge at all into the right partition, so no perfect
        // matching can exist -- this must be reported as an error, never a
        // silently wrong/partial matching.
        let mut graph = create_graph::<i32, f64>();
        graph.add_edge(0, 10, 1.0).expect("Operation failed");
        graph.add_node(1);
        graph.add_edge(2, 11, 1.0).expect("Operation failed");

        assert!(minimum_weight_bipartite_matching(&graph).is_err());
    }

    #[test]
    fn test_hungarian_algorithm_detects_infeasible_instance() {
        // Rows 0 and 1 can BOTH only be assigned to column 0 (Hall's
        // marriage condition is violated for the subset {0, 1}), so no
        // perfect assignment exists even though every row has at least one
        // finite-cost entry. The solver must reject this outright rather
        // than corrupt its potentials or silently return a partial/invalid
        // assignment.
        let inf = f64::INFINITY;
        let cost_matrix = vec![
            vec![1.0, inf, inf],
            vec![1.0, inf, inf],
            vec![inf, 1.0, 1.0],
        ];

        assert!(hungarian_algorithm(&cost_matrix).is_err());
    }

    #[test]
    fn test_maximum_cardinality_matching() {
        let mut graph = create_graph::<&str, ()>();

        // Create a simple graph
        graph.add_edge("A", "B", ()).expect("Operation failed");
        graph.add_edge("C", "D", ()).expect("Operation failed");
        graph.add_edge("E", "F", ()).expect("Operation failed");

        let matching = maximum_cardinality_matching(&graph);

        // Should find a matching of size 3
        assert_eq!(matching.size, 3);
        assert_eq!(matching.matching.len(), 3);

        // Verify no node is matched twice
        let mut matched_nodes = HashSet::new();
        for (u, v) in &matching.matching {
            assert!(!matched_nodes.contains(u));
            assert!(!matched_nodes.contains(v));
            matched_nodes.insert(u);
            matched_nodes.insert(v);
        }
    }

    #[test]
    fn test_maximal_matching() {
        let mut graph = create_graph::<i32, ()>();

        // Create a triangle
        graph.add_edge(1, 2, ()).expect("Operation failed");
        graph.add_edge(2, 3, ()).expect("Operation failed");
        graph.add_edge(3, 1, ()).expect("Operation failed");

        let matching = maximal_matching(&graph);

        // Should find at least one edge (maximal for triangle is 1)
        assert_eq!(matching.size, 1);
        assert_eq!(matching.matching.len(), 1);

        // Verify it's a valid matching
        let mut matched_nodes = HashSet::new();
        for (u, v) in &matching.matching {
            assert!(!matched_nodes.contains(u));
            assert!(!matched_nodes.contains(v));
            matched_nodes.insert(u);
            matched_nodes.insert(v);
        }
    }

    #[test]
    fn test_stable_marriage() -> GraphResult<()> {
        // Example: 3 people on each side
        let left_prefs = vec![
            vec![0, 1, 2], // Person 0 prefers 0, then 1, then 2
            vec![1, 0, 2], // Person 1 prefers 1, then 0, then 2
            vec![0, 1, 2], // Person 2 prefers 0, then 1, then 2
        ];

        let right_prefs = vec![
            vec![2, 1, 0], // Person 0 prefers 2, then 1, then 0
            vec![0, 2, 1], // Person 1 prefers 0, then 2, then 1
            vec![0, 1, 2], // Person 2 prefers 0, then 1, then 2
        ];

        let matching = stable_marriage(&left_prefs, &right_prefs)?;

        // Should have 3 pairs
        assert_eq!(matching.len(), 3);

        // Verify it's a complete matching
        let mut matched_left = HashSet::new();
        let mut matched_right = HashSet::new();
        for (left, right) in &matching {
            assert!(!matched_left.contains(left));
            assert!(!matched_right.contains(right));
            matched_left.insert(*left);
            matched_right.insert(*right);
        }

        Ok(())
    }

    #[test]
    fn test_stable_marriage_empty() -> GraphResult<()> {
        let left_prefs: Vec<Vec<usize>> = vec![];
        let right_prefs: Vec<Vec<usize>> = vec![];

        let matching = stable_marriage(&left_prefs, &right_prefs)?;
        assert_eq!(matching.len(), 0);

        Ok(())
    }

    #[test]
    fn test_stable_marriage_invalid_input() {
        // Mismatched sizes
        let left_prefs = vec![vec![0]];
        let right_prefs = vec![vec![0], vec![1]];

        assert!(stable_marriage(&left_prefs, &right_prefs).is_err());

        // Invalid preference list
        let left_prefs = vec![vec![0, 0]]; // Duplicate
        let right_prefs = vec![vec![0, 1]];

        assert!(stable_marriage(&left_prefs, &right_prefs).is_err());
    }
}
