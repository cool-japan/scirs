//! Hopcroft-Karp maximum bipartite matching
//!
//! This module implements the Hopcroft-Karp algorithm for finding maximum matchings
//! in bipartite graphs in O(E * sqrt(V)) time. It is used as a component of the
//! Pantelides algorithm for DAE index reduction.

use std::collections::VecDeque;

/// Sentinel for "unmatched" or "unreachable".
const UNMATCHED: usize = usize::MAX;

/// Hopcroft-Karp maximum bipartite matching.
///
/// Finds a maximum-cardinality matching in the bipartite graph defined by `edges`.
/// Left vertices are indexed `0..left_n`; right vertices are `0..right_n`.
///
/// Returns `match_left[i] = Some(j)` if left vertex `i` is matched with right vertex `j`,
/// or `None` if left vertex `i` is unmatched.
///
/// # Algorithm
///
/// Hopcroft-Karp runs alternating BFS/DFS phases:
/// 1. BFS from all free left vertices builds a layered graph of shortest augmenting paths.
/// 2. DFS finds maximal set of vertex-disjoint augmenting paths along those layers.
/// 3. Repeat until no augmenting paths exist.
///
/// # Complexity
///
/// O(E * sqrt(V)) where E = `edges.len()` and V = `left_n + right_n`.
///
/// # Examples
///
/// ```
/// use scirs2_integrate::dae::bipartite_matching::hopcroft_karp;
///
/// // Perfect matching: (0,0),(1,1),(2,2)
/// let edges = vec![(0,0),(1,1),(2,2)];
/// let matching = hopcroft_karp(3, 3, &edges);
/// assert_eq!(matching[0], Some(0));
/// assert_eq!(matching[1], Some(1));
/// assert_eq!(matching[2], Some(2));
/// ```
pub fn hopcroft_karp(
    left_n: usize,
    right_n: usize,
    edges: &[(usize, usize)],
) -> Vec<Option<usize>> {
    if left_n == 0 || right_n == 0 {
        return vec![None; left_n];
    }

    // Build adjacency list: adj[i] = list of right vertices adjacent to left vertex i
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); left_n];
    for &(l, r) in edges {
        if l < left_n && r < right_n {
            adj[l].push(r);
        }
    }

    // match_left[i] = matched right vertex for left vertex i (UNMATCHED = free)
    let mut match_left: Vec<usize> = vec![UNMATCHED; left_n];
    // match_right[j] = matched left vertex for right vertex j (UNMATCHED = free)
    let mut match_right: Vec<usize> = vec![UNMATCHED; right_n];

    loop {
        // BFS phase: compute distances for the layered graph
        let dist = bfs_phase(&adj, &match_left, &match_right, left_n, right_n);

        // If no augmenting path reaches the sink, matching is maximum
        if dist[left_n] == UNMATCHED {
            break;
        }

        // DFS phase: augment along all found paths
        let mut dist_dfs = dist.clone();
        for u in 0..left_n {
            if match_left[u] == UNMATCHED {
                dfs_augment(
                    u,
                    &adj,
                    &mut match_left,
                    &mut match_right,
                    &mut dist_dfs,
                    left_n,
                );
            }
        }
    }

    // Convert UNMATCHED sentinel to None
    match_left
        .into_iter()
        .map(|v| if v == UNMATCHED { None } else { Some(v) })
        .collect()
}

/// BFS phase: builds the layered graph and returns `dist` array of length `left_n + 1`.
/// `dist[u]` = BFS layer of left vertex `u`; `dist[left_n]` is the sink layer.
/// Unreachable vertices have `dist[u] = UNMATCHED`.
fn bfs_phase(
    adj: &[Vec<usize>],
    match_left: &[usize],
    match_right: &[usize],
    left_n: usize,
    right_n: usize,
) -> Vec<usize> {
    let mut dist = vec![UNMATCHED; left_n + 1];
    let mut queue = VecDeque::new();

    // Seed BFS with all free (unmatched) left vertices
    for u in 0..left_n {
        if match_left[u] == UNMATCHED {
            dist[u] = 0;
            queue.push_back(u);
        }
    }

    while let Some(u) = queue.pop_front() {
        // Only explore if we haven't already found a shorter path to the sink
        let du = dist[u];
        if du == UNMATCHED {
            continue;
        }
        let sink_dist = dist[left_n];
        if sink_dist != UNMATCHED && du >= sink_dist {
            continue;
        }

        for &v in &adj[u] {
            if v >= right_n {
                continue;
            }
            let w = match_right[v];
            // w is the left vertex matched to right v (or UNMATCHED → this is the sink)
            let w_idx = if w == UNMATCHED { left_n } else { w };
            if dist[w_idx] == UNMATCHED {
                dist[w_idx] = du + 1;
                if w_idx != left_n {
                    queue.push_back(w_idx);
                }
            }
        }
    }

    dist
}

/// Iterative DFS augmentation: finds and augments an augmenting path from `start`.
/// Returns `true` if a path was found and augmented.
fn dfs_augment(
    start: usize,
    adj: &[Vec<usize>],
    match_left: &mut [usize],
    match_right: &mut [usize],
    dist: &mut [usize],
    left_n: usize,
) -> bool {
    // Iterative DFS: stack contains (left_vertex, edge_cursor).
    // We also track the right-vertices along the current path so we can augment.
    let mut call_stack: Vec<(usize, usize)> = vec![(start, 0)];
    // right_path[i] is the right vertex used to go from call_stack[i].0 to call_stack[i+1].0
    let mut right_path: Vec<usize> = Vec::new();

    'outer: while let Some(top) = call_stack.last_mut() {
        let (u, ref mut ei) = *top;

        // Try edges from u
        let edges = &adj[u];
        loop {
            if *ei >= edges.len() {
                // No more edges — backtrack
                call_stack.pop();
                right_path.pop();
                // Mark u as exhausted in this DFS phase
                dist[u] = UNMATCHED;
                continue 'outer;
            }

            let v = edges[*ei];
            *ei += 1;

            let w = match_right[v];
            let w_idx = if w == UNMATCHED { left_n } else { w };

            // Check if this edge is on the layered graph (dist constraint)
            let du = dist[u];
            if du == UNMATCHED {
                // u was marked exhausted, skip
                continue;
            }
            if dist[w_idx] != du + 1 {
                continue;
            }

            // w_idx is on the next layer
            if w_idx == left_n {
                // Reached the sink — augment along path (including this last edge)
                right_path.push(v);
                let path_len = right_path.len();
                for k in 0..path_len {
                    let lv = call_stack[k].0;
                    let rv = right_path[k];
                    match_left[lv] = rv;
                    match_right[rv] = lv;
                }
                return true;
            }

            // Mark w_idx as visited to prevent re-use
            dist[w_idx] = UNMATCHED;
            right_path.push(v);
            call_stack.push((w_idx, 0));
            continue 'outer;
        }
    }

    false
}

/// Returns the set of unmatched left vertices after running Hopcroft-Karp.
///
/// These unmatched vertices correspond to equations that cannot be matched
/// to a variable in the structural analysis of a DAE system.
pub fn find_unmatched_left(matching: &[Option<usize>]) -> Vec<usize> {
    matching
        .iter()
        .enumerate()
        .filter_map(|(i, m)| if m.is_none() { Some(i) } else { None })
        .collect()
}

/// Given the matching, find all left vertices reachable from unmatched left vertices
/// via alternating paths (unmatched edge → matched edge → ...).
///
/// Returns the set of all reachable left vertices (including the starting unmatched ones).
/// This set of equations forms the "singular subset" in the Pantelides algorithm.
pub fn alternating_reachable(
    left_n: usize,
    right_n: usize,
    adj: &[Vec<usize>],
    matching: &[Option<usize>],
) -> Vec<usize> {
    // Build match_right: right vertex → left vertex that matches it
    let mut match_right: Vec<Option<usize>> = vec![None; right_n];
    for (l, m) in matching.iter().enumerate() {
        if let Some(r) = m {
            if *r < right_n {
                match_right[*r] = Some(l);
            }
        }
    }

    let mut visited_left = vec![false; left_n];
    let mut visited_right = vec![false; right_n];
    let mut queue: VecDeque<usize> = VecDeque::new();

    // Seed BFS with all unmatched left vertices
    for u in 0..left_n {
        if matching[u].is_none() {
            visited_left[u] = true;
            queue.push_back(u);
        }
    }

    while let Some(u) = queue.pop_front() {
        // From left vertex u, follow all (unmatched) edges to right vertices
        if u < adj.len() {
            for &v in &adj[u] {
                if v < right_n && !visited_right[v] {
                    visited_right[v] = true;
                    // From right vertex v, follow the matched edge back to a left vertex
                    if let Some(w) = match_right[v] {
                        if !visited_left[w] {
                            visited_left[w] = true;
                            queue.push_back(w);
                        }
                    }
                }
            }
        }
    }

    (0..left_n).filter(|&i| visited_left[i]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perfect_matching_3x3() {
        let edges = vec![(0, 0), (1, 1), (2, 2)];
        let m = hopcroft_karp(3, 3, &edges);
        assert_eq!(m[0], Some(0));
        assert_eq!(m[1], Some(1));
        assert_eq!(m[2], Some(2));
    }

    #[test]
    fn test_empty_graph() {
        let m = hopcroft_karp(3, 3, &[]);
        assert!(m.iter().all(|x| x.is_none()));
    }

    #[test]
    fn test_partial_matching() {
        // Only 2 right vertices, 3 left: at most 2 can be matched
        let edges = vec![(0, 0), (1, 0), (2, 1)];
        let m = hopcroft_karp(3, 2, &edges);
        let matched_count = m.iter().filter(|x| x.is_some()).count();
        assert_eq!(matched_count, 2);
    }

    #[test]
    fn test_alternating_reachable_simple() {
        // Left 0,1 — Right 0,1
        // Only edge: (0,0),(1,1) — perfect matching, no reachable from unmatched
        let edges = vec![(0usize, 0usize), (1, 1)];
        let matching = hopcroft_karp(2, 2, &edges);
        let adj: Vec<Vec<usize>> = vec![vec![0], vec![1]];
        let reachable = alternating_reachable(2, 2, &adj, &matching);
        // Perfect match → no unmatched → empty reachable set
        assert!(
            reachable.is_empty(),
            "No unmatched vertices means nothing reachable: {:?}",
            reachable
        );
    }

    #[test]
    fn test_alternating_reachable_with_unmatched() {
        // 3 left, 2 right → 1 unmatched left
        let edges = vec![(0, 0), (1, 1)];
        let matching = hopcroft_karp(3, 2, &edges);
        let adj: Vec<Vec<usize>> = vec![vec![0], vec![1], vec![]];
        let reachable = alternating_reachable(3, 2, &adj, &matching);
        // Left vertex 2 has no edges and is unmatched — it is trivially reachable
        assert!(
            reachable.contains(&2),
            "Unmatched left-2 must be reachable: {:?}",
            reachable
        );
    }

    #[test]
    fn test_chain_augmenting() {
        // Forces Hopcroft-Karp to use augmenting paths through existing matches
        let edges = vec![(0, 0), (1, 0), (1, 1), (2, 1), (2, 2)];
        let m = hopcroft_karp(3, 3, &edges);
        let matched_count = m.iter().filter(|x| x.is_some()).count();
        assert_eq!(
            matched_count, 3,
            "Perfect matching should be found: {:?}",
            m
        );
        // Right vertices must all be distinct
        let mut rights: Vec<usize> = m.iter().filter_map(|x| *x).collect();
        rights.sort_unstable();
        rights.dedup();
        assert_eq!(rights.len(), 3);
    }
}
