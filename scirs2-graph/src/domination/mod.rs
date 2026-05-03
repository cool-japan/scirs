//! Graph domination and covering problems.
//!
//! Greedy algorithms for NP-hard problems:
//! - Minimum dominating set
//! - Minimum vertex cover
//! - Maximum independent set
//! - Minimum edge dominating set
//! - Feedback vertex set

use std::collections::{HashSet, VecDeque};

// ─────────────────────────────────────────────────────────────────────────────
// Minimum Dominating Set
// ─────────────────────────────────────────────────────────────────────────────

/// Greedy approximation for the minimum dominating set.
///
/// A dominating set D is a subset of vertices such that every vertex not in D
/// has a neighbour in D.  The greedy strategy repeatedly selects the vertex
/// that dominates the most uncovered vertices.
///
/// # Arguments
/// * `edges`   – undirected edge list (0-indexed)
/// * `n_nodes` – number of vertices
///
/// # Returns
/// A dominating set (may not be minimum, but is a valid O(log N) approximation).
pub fn minimum_dominating_set(edges: &[(usize, usize)], n_nodes: usize) -> Vec<usize> {
    if n_nodes == 0 {
        return vec![];
    }
    let adj = build_adj(edges, n_nodes);
    let mut dominated = vec![false; n_nodes];
    let mut in_set = vec![false; n_nodes];
    let mut result = Vec::new();

    // Helper: count how many vertices v would newly dominate
    fn vertex_coverage(v: usize, adj: &[Vec<usize>], dominated: &[bool]) -> usize {
        let mut cnt = if !dominated[v] { 1 } else { 0 };
        for &w in &adj[v] {
            if !dominated[w] {
                cnt += 1;
            }
        }
        cnt
    }

    loop {
        // Check if all vertices are dominated
        if dominated.iter().all(|&d| d) {
            break;
        }
        // Find vertex with maximum uncovered coverage
        let best = (0..n_nodes)
            .filter(|&v| !in_set[v])
            .max_by_key(|&v| vertex_coverage(v, &adj, &dominated));
        match best {
            None => break,
            Some(v) => {
                if vertex_coverage(v, &adj, &dominated) == 0 {
                    break;
                }
                in_set[v] = true;
                result.push(v);
                dominated[v] = true;
                for &w in &adj[v] {
                    dominated[w] = true;
                }
            }
        }
    }
    result.sort_unstable();
    result
}

// ─────────────────────────────────────────────────────────────────────────────
// Minimum Vertex Cover
// ─────────────────────────────────────────────────────────────────────────────

/// Greedy 2-approximation for the minimum vertex cover.
///
/// Iteratively selects a maximal matching and includes both endpoints of each
/// matched edge (guaranteed ≤ 2 × OPT).
///
/// # Returns
/// A vertex cover: a set S such that every edge has at least one endpoint in S.
pub fn minimum_vertex_cover(edges: &[(usize, usize)], n_nodes: usize) -> Vec<usize> {
    if n_nodes == 0 || edges.is_empty() {
        return vec![];
    }
    let mut covered = HashSet::new();
    let mut result = HashSet::new();

    // Maximal matching approach
    for &(u, v) in edges {
        if u >= n_nodes || v >= n_nodes || u == v {
            continue;
        }
        if !covered.contains(&u) && !covered.contains(&v) {
            result.insert(u);
            result.insert(v);
            covered.insert(u);
            covered.insert(v);
        }
    }
    let mut vec: Vec<usize> = result.into_iter().collect();
    vec.sort_unstable();
    vec
}

// ─────────────────────────────────────────────────────────────────────────────
// Maximum Independent Set
// ─────────────────────────────────────────────────────────────────────────────

/// Greedy approximation for a maximum independent set.
///
/// An independent set is a set of vertices with no two adjacent.  The greedy
/// strategy iteratively adds the minimum-degree uncovered vertex and removes
/// its neighbours.
///
/// # Returns
/// An independent set (may not be maximum).
pub fn maximum_independent_set(edges: &[(usize, usize)], n_nodes: usize) -> Vec<usize> {
    if n_nodes == 0 {
        return vec![];
    }
    let adj = build_adj(edges, n_nodes);
    let mut active = vec![true; n_nodes];
    let mut result = Vec::new();

    loop {
        // Pick active vertex with minimum degree among active vertices
        let best = (0..n_nodes)
            .filter(|&v| active[v])
            .min_by_key(|&v| adj[v].iter().filter(|&&w| active[w]).count());
        match best {
            None => break,
            Some(v) => {
                result.push(v);
                active[v] = false;
                for &w in &adj[v] {
                    active[w] = false;
                }
            }
        }
    }
    result.sort_unstable();
    result
}

// ─────────────────────────────────────────────────────────────────────────────
// Minimum Edge Dominating Set
// ─────────────────────────────────────────────────────────────────────────────

/// Greedy 2-approximation for the minimum edge dominating set.
///
/// An edge dominating set is a set F of edges such that every edge not in F
/// shares an endpoint with some edge in F.  A maximum matching is a 2-
/// approximation (every edge within a maximal matching constitutes a minimal
/// edge dominating set).
///
/// # Returns
/// A set of edges forming an edge dominating set.
pub fn minimum_edge_dominating_set(
    edges: &[(usize, usize)],
    n_nodes: usize,
) -> Vec<(usize, usize)> {
    if n_nodes == 0 || edges.is_empty() {
        return vec![];
    }
    // Maximal matching = minimal edge dominating set
    let mut matched = vec![false; n_nodes];
    let mut result = Vec::new();
    for &(u, v) in edges {
        if u >= n_nodes || v >= n_nodes || u == v {
            continue;
        }
        if !matched[u] && !matched[v] {
            result.push(if u < v { (u, v) } else { (v, u) });
            matched[u] = true;
            matched[v] = true;
        }
    }
    result
}

// ─────────────────────────────────────────────────────────────────────────────
// Feedback Vertex Set
// ─────────────────────────────────────────────────────────────────────────────

/// Greedy approximation for the feedback vertex set (FVS).
///
/// An FVS is a set of vertices whose removal makes the graph acyclic (forest).
/// The greedy strategy: repeatedly find a cycle (via DFS), pick the vertex with
/// the highest degree on that cycle, add it to the FVS, and remove it.
///
/// # Returns
/// A set of vertices whose removal leaves a forest.
pub fn feedback_vertex_set(edges: &[(usize, usize)], n_nodes: usize) -> Vec<usize> {
    if n_nodes == 0 {
        return vec![];
    }
    let mut adj = build_adj(edges, n_nodes);
    let mut removed = vec![false; n_nodes];
    let mut result = Vec::new();

    loop {
        // Check if acyclic
        match find_cycle(&adj, n_nodes, &removed) {
            None => break,
            Some(cycle) => {
                // Pick vertex in cycle with highest degree
                let best = cycle
                    .iter()
                    .max_by_key(|&&v| adj[v].iter().filter(|&&w| !removed[w]).count())
                    .copied();
                if let Some(v) = best {
                    removed[v] = true;
                    result.push(v);
                    // Remove v from adjacency lists
                    for w in 0..n_nodes {
                        adj[w].retain(|&x| x != v);
                    }
                    adj[v].clear();
                }
            }
        }
    }
    result.sort_unstable();
    result
}

/// Finds a cycle in the graph (excluding removed vertices) using DFS.
/// Returns the vertices on the cycle, or None if the graph is acyclic.
fn find_cycle(adj: &[Vec<usize>], n: usize, removed: &[bool]) -> Option<Vec<usize>> {
    let mut colour = vec![0u8; n]; // 0=white, 1=grey, 2=black
    let mut parent = vec![usize::MAX; n];

    for start in 0..n {
        if colour[start] != 0 || removed[start] {
            continue;
        }
        // Iterative DFS
        let mut stack: Vec<(usize, usize)> = vec![(start, usize::MAX)]; // (vertex, neighbour_idx)
        while let Some((v, ni)) = stack.last_mut().copied() {
            if colour[v] == 0 {
                colour[v] = 1;
            }
            // Find next unvisited / grey neighbour
            let neighbours: Vec<usize> = adj[v].iter().copied().filter(|&w| !removed[w]).collect();
            let mut found_next = false;
            for idx in ni..neighbours.len() {
                let w = neighbours[idx];
                if let Some((_, ni_ref)) = stack.last_mut() {
                    *ni_ref = idx + 1;
                }
                if colour[w] == 1 && parent[v] != w {
                    // Back edge → cycle found
                    // Reconstruct cycle from v back to w
                    let mut cycle = vec![w, v];
                    let mut cur = v;
                    while cur != w {
                        cur = parent[cur];
                        if cur == usize::MAX {
                            break;
                        }
                        cycle.push(cur);
                    }
                    return Some(cycle);
                }
                if colour[w] == 0 {
                    parent[w] = v;
                    stack.push((w, 0));
                    found_next = true;
                    break;
                }
            }
            if !found_next {
                colour[v] = 2;
                stack.pop();
            }
        }
    }
    None
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper
// ─────────────────────────────────────────────────────────────────────────────

fn build_adj(edges: &[(usize, usize)], n: usize) -> Vec<Vec<usize>> {
    let mut adj = vec![vec![]; n];
    for &(u, v) in edges {
        if u < n && v < n && u != v {
            adj[u].push(v);
            adj[v].push(u);
        }
    }
    adj
}

/// BFS-based connectivity check (used in tests).
#[allow(dead_code)]
fn is_connected_after_removal(adj: &[Vec<usize>], removed: &[bool], n: usize) -> bool {
    let start = (0..n).find(|&v| !removed[v]);
    let start = match start {
        Some(s) => s,
        None => return true,
    };
    let mut visited = vec![false; n];
    let mut queue = VecDeque::new();
    queue.push_back(start);
    visited[start] = true;
    let mut count = 1usize;
    while let Some(v) = queue.pop_front() {
        for &w in &adj[v] {
            if !visited[w] && !removed[w] {
                visited[w] = true;
                count += 1;
                queue.push_back(w);
            }
        }
    }
    count == n - removed.iter().filter(|&&r| r).count()
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    fn triangle() -> Vec<(usize, usize)> {
        vec![(0, 1), (1, 2), (0, 2)]
    }

    fn path4() -> Vec<(usize, usize)> {
        vec![(0, 1), (1, 2), (2, 3)]
    }

    fn k4() -> Vec<(usize, usize)> {
        vec![(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    }

    // ── Minimum dominating set ──────────────────────────────────────────────

    #[test]
    fn test_dominating_set_triangle() {
        let ds = minimum_dominating_set(&triangle(), 3);
        // Any single vertex dominates all others in a triangle
        assert!(!ds.is_empty(), "dominating set must be non-empty");
        // Verify it is a valid dominating set
        let ds_set: HashSet<usize> = ds.iter().copied().collect();
        for v in 0..3 {
            let dominated = ds_set.contains(&v)
                || triangle().iter().any(|&(u, w)| {
                    (u == v && ds_set.contains(&w)) || (w == v && ds_set.contains(&u))
                });
            assert!(dominated, "vertex {v} is not dominated");
        }
    }

    #[test]
    fn test_dominating_set_path4() {
        let ds = minimum_dominating_set(&path4(), 4);
        let ds_set: HashSet<usize> = ds.iter().copied().collect();
        // Verify validity
        for v in 0..4 {
            let dominated = ds_set.contains(&v)
                || path4().iter().any(|&(u, w)| {
                    (u == v && ds_set.contains(&w)) || (w == v && ds_set.contains(&u))
                });
            assert!(dominated, "vertex {v} not dominated in path4");
        }
    }

    #[test]
    fn test_dominating_set_empty() {
        let ds = minimum_dominating_set(&[], 0);
        assert!(ds.is_empty());
    }

    // ── Minimum vertex cover ────────────────────────────────────────────────

    #[test]
    fn test_vertex_cover_triangle() {
        let vc = minimum_vertex_cover(&triangle(), 3);
        let vc_set: HashSet<usize> = vc.iter().copied().collect();
        for &(u, v) in &triangle() {
            assert!(
                vc_set.contains(&u) || vc_set.contains(&v),
                "edge ({u},{v}) not covered"
            );
        }
    }

    #[test]
    fn test_vertex_cover_path4() {
        let vc = minimum_vertex_cover(&path4(), 4);
        let vc_set: HashSet<usize> = vc.iter().copied().collect();
        for &(u, v) in &path4() {
            assert!(vc_set.contains(&u) || vc_set.contains(&v));
        }
    }

    #[test]
    fn test_vertex_cover_empty_edges() {
        let vc = minimum_vertex_cover(&[], 5);
        assert!(vc.is_empty());
    }

    // ── Maximum independent set ─────────────────────────────────────────────

    #[test]
    fn test_independent_set_triangle() {
        let mis = maximum_independent_set(&triangle(), 3);
        // In a triangle the maximum independent set has size 1
        assert!(!mis.is_empty());
        let mis_set: HashSet<usize> = mis.iter().copied().collect();
        for &(u, v) in &triangle() {
            assert!(
                !(mis_set.contains(&u) && mis_set.contains(&v)),
                "edge ({u},{v}) violates independence"
            );
        }
    }

    #[test]
    fn test_independent_set_path4() {
        let mis = maximum_independent_set(&path4(), 4);
        let mis_set: HashSet<usize> = mis.iter().copied().collect();
        for &(u, v) in &path4() {
            assert!(!(mis_set.contains(&u) && mis_set.contains(&v)));
        }
        // Optimal MIS for 0-1-2-3 has size 2 (e.g. {0,2} or {1,3})
        assert!(mis.len() >= 2);
    }

    #[test]
    fn test_independent_set_k4() {
        let mis = maximum_independent_set(&k4(), 4);
        let mis_set: HashSet<usize> = mis.iter().copied().collect();
        for &(u, v) in &k4() {
            assert!(!(mis_set.contains(&u) && mis_set.contains(&v)));
        }
        // K4 MIS has size 1
        assert_eq!(mis.len(), 1);
    }

    // ── Minimum edge dominating set ─────────────────────────────────────────

    #[test]
    fn test_edge_dominating_set_triangle() {
        let eds = minimum_edge_dominating_set(&triangle(), 3);
        assert!(!eds.is_empty());
        let edge_set: HashSet<(usize, usize)> = eds.iter().copied().collect();
        // Every edge must be dominated
        for &(u, v) in &triangle() {
            let e = if u < v { (u, v) } else { (v, u) };
            let dominated = edge_set.contains(&e)
                || edge_set
                    .iter()
                    .any(|&(a, b)| a == u || b == u || a == v || b == v);
            assert!(dominated, "edge ({u},{v}) not dominated");
        }
    }

    #[test]
    fn test_edge_dominating_set_empty() {
        assert!(minimum_edge_dominating_set(&[], 3).is_empty());
    }

    // ── Feedback vertex set ─────────────────────────────────────────────────

    #[test]
    fn test_feedback_vertex_set_triangle() {
        let fvs = feedback_vertex_set(&triangle(), 3);
        // After removing FVS, graph should be acyclic
        let removed: Vec<bool> = (0..3).map(|v| fvs.contains(&v)).collect();
        let adj = build_adj(&triangle(), 3);
        assert!(
            find_cycle(&adj, 3, &removed).is_none(),
            "FVS should eliminate all cycles"
        );
    }

    #[test]
    fn test_feedback_vertex_set_tree_empty() {
        // A tree (path) has no cycles → FVS should be empty
        let fvs = feedback_vertex_set(&path4(), 4);
        assert!(fvs.is_empty(), "path has no cycles, FVS should be empty");
    }

    #[test]
    fn test_feedback_vertex_set_k4() {
        let fvs = feedback_vertex_set(&k4(), 4);
        let removed: Vec<bool> = (0..4).map(|v| fvs.contains(&v)).collect();
        let adj = build_adj(&k4(), 4);
        assert!(
            find_cycle(&adj, 4, &removed).is_none(),
            "FVS of K4 should eliminate all cycles"
        );
    }
}
