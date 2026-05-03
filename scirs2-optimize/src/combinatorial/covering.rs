//! Set cover, weighted set cover, vertex cover, and hitting set algorithms.
//!
//! - Greedy log-approximation for (weighted) set cover
//! - 2-approximation vertex cover via maximal matching
//! - König's theorem exact minimum vertex cover for bipartite graphs
//! - Greedy hitting set

use crate::error::OptimizeError;

/// Result type for covering operations.
pub type CoveringResult<T> = Result<T, OptimizeError>;

// ── Greedy set cover ──────────────────────────────────────────────────────────

/// Greedy set cover (unweighted): at each step, pick the set that covers the
/// most uncovered elements of `universe`.
///
/// Achieves O(ln n) approximation ratio.
///
/// Returns the indices of selected sets.
pub fn greedy_set_cover(universe: usize, sets: &[Vec<usize>]) -> Vec<usize> {
    if universe == 0 || sets.is_empty() {
        return vec![];
    }

    let mut uncovered = vec![true; universe];
    let mut remaining = universe;
    let mut selected = Vec::new();
    let mut used = vec![false; sets.len()];

    while remaining > 0 {
        // Find the set that covers the most uncovered elements
        let mut best_idx = None;
        let mut best_count = 0usize;

        for (i, set) in sets.iter().enumerate() {
            if used[i] {
                continue;
            }
            let count = set
                .iter()
                .filter(|&&e| e < universe && uncovered[e])
                .count();
            if count > best_count {
                best_count = count;
                best_idx = Some(i);
            }
        }

        match best_idx {
            Some(idx) => {
                used[idx] = true;
                selected.push(idx);
                for &e in &sets[idx] {
                    if e < universe && uncovered[e] {
                        uncovered[e] = false;
                        remaining -= 1;
                    }
                }
            }
            None => break, // remaining elements cannot be covered
        }
    }

    selected
}

// ── Weighted set cover ────────────────────────────────────────────────────────

/// Greedy weighted set cover: at each step, pick the set with the lowest
/// cost per newly-covered element.
///
/// Achieves O(ln n) approximation ratio.
///
/// Returns `(selected_indices, total_cost)`.
pub fn weighted_set_cover(
    universe: usize,
    sets: &[Vec<usize>],
    costs: &[f64],
) -> CoveringResult<(Vec<usize>, f64)> {
    if sets.len() != costs.len() {
        return Err(OptimizeError::InvalidInput(format!(
            "sets.len() = {} but costs.len() = {}",
            sets.len(),
            costs.len()
        )));
    }
    if universe == 0 || sets.is_empty() {
        return Ok((vec![], 0.0));
    }

    let mut uncovered = vec![true; universe];
    let mut remaining = universe;
    let mut selected = Vec::new();
    let mut total_cost = 0.0;
    let mut used = vec![false; sets.len()];

    while remaining > 0 {
        let mut best_idx = None;
        let mut best_ratio = f64::INFINITY;

        for (i, set) in sets.iter().enumerate() {
            if used[i] {
                continue;
            }
            let new_covers = set
                .iter()
                .filter(|&&e| e < universe && uncovered[e])
                .count();
            if new_covers == 0 {
                continue;
            }
            let ratio = costs[i] / new_covers as f64;
            if ratio < best_ratio {
                best_ratio = ratio;
                best_idx = Some(i);
            }
        }

        match best_idx {
            Some(idx) => {
                used[idx] = true;
                selected.push(idx);
                total_cost += costs[idx];
                for &e in &sets[idx] {
                    if e < universe && uncovered[e] {
                        uncovered[e] = false;
                        remaining -= 1;
                    }
                }
            }
            None => break,
        }
    }

    Ok((selected, total_cost))
}

// ── 2-approximation vertex cover ──────────────────────────────────────────────

/// 2-approximation vertex cover via maximal matching.
///
/// Repeatedly picks an arbitrary uncovered edge and adds both endpoints to the
/// cover.  Guarantees a cover of size ≤ 2 · OPT.
pub fn vertex_cover_2approx(n: usize, edges: &[(usize, usize)]) -> Vec<usize> {
    if n == 0 || edges.is_empty() {
        return vec![];
    }

    let mut in_cover = vec![false; n];
    let mut cover = Vec::new();

    for &(u, v) in edges {
        if u >= n || v >= n {
            continue;
        }
        // If neither endpoint is yet in the cover, add both
        if !in_cover[u] && !in_cover[v] {
            in_cover[u] = true;
            in_cover[v] = true;
            cover.push(u);
            cover.push(v);
        }
    }

    cover.sort_unstable();
    cover.dedup();
    cover
}

// ── König's theorem: exact min vertex cover for bipartite graphs ──────────────

/// Exact minimum vertex cover for a bipartite graph via König's theorem.
///
/// Uses a maximum bipartite matching (augmenting paths) followed by the
/// alternating-tree construction from König's theorem.
///
/// Vertices `0..n_left` are on the left side; vertices `n_left..n_left+n_right`
/// on the right.  Edges connect left vertex `u` to right vertex `v` (stored as
/// pairs `(u_left, v_right)` in 0-indexed form on each side).
///
/// Returns the vertex cover as a list of global vertex indices
/// (left side: 0..n_left, right side: n_left..n_left+n_right).
pub fn min_vertex_cover_bip(n_left: usize, n_right: usize, edges: &[(usize, usize)]) -> Vec<usize> {
    if n_left == 0 || n_right == 0 {
        return vec![];
    }

    // Build adjacency for left→right
    let mut adj_left: Vec<Vec<usize>> = vec![vec![]; n_left];
    for &(u, v) in edges {
        if u < n_left && v < n_right {
            adj_left[u].push(v);
        }
    }

    // Maximum matching via Hopcroft-Karp (simplified augmenting paths)
    let mut match_left = vec![usize::MAX; n_left];
    let mut match_right = vec![usize::MAX; n_right];

    for u in 0..n_left {
        let mut visited = vec![false; n_right];
        augment_bip(
            u,
            &adj_left,
            &mut match_left,
            &mut match_right,
            &mut visited,
        );
    }

    // König's construction:
    // Let U = unmatched left vertices.
    // Alternating BFS/DFS from U:
    //   - from left vertex: go to all right neighbours
    //   - from right vertex: go to its matched left vertex
    // T_L = left vertices reachable, T_R = right vertices reachable
    // Min vertex cover = (L \ T_L) ∪ T_R

    let mut reachable_left = vec![false; n_left];
    let mut reachable_right = vec![false; n_right];

    // Start from unmatched left vertices
    let mut stack: Vec<(bool, usize)> = Vec::new(); // (is_left, index)
    for u in 0..n_left {
        if match_left[u] == usize::MAX {
            stack.push((true, u));
            reachable_left[u] = true;
        }
    }

    while let Some((is_left, v)) = stack.pop() {
        if is_left {
            // Explore all right neighbours
            for &r in &adj_left[v] {
                if !reachable_right[r] {
                    reachable_right[r] = true;
                    stack.push((false, r));
                }
            }
        } else {
            // From right vertex v, follow matching edge to left
            let ml = match_right[v];
            if ml != usize::MAX && !reachable_left[ml] {
                reachable_left[ml] = true;
                stack.push((true, ml));
            }
        }
    }

    // Cover = (L \ T_L) ∪ T_R  (as global indices)
    let mut cover = Vec::new();
    for u in 0..n_left {
        if !reachable_left[u] {
            cover.push(u); // left side global index = u
        }
    }
    for v in 0..n_right {
        if reachable_right[v] {
            cover.push(n_left + v); // right side global index = n_left + v
        }
    }

    cover.sort_unstable();
    cover
}

/// Augmenting path DFS for bipartite matching.
fn augment_bip(
    u: usize,
    adj: &[Vec<usize>],
    match_left: &mut Vec<usize>,
    match_right: &mut Vec<usize>,
    visited: &mut Vec<bool>,
) -> bool {
    for &v in &adj[u] {
        if visited[v] {
            continue;
        }
        visited[v] = true;
        let prev = match_right[v];
        if prev == usize::MAX || augment_bip(prev, adj, match_left, match_right, visited) {
            match_left[u] = v;
            match_right[v] = u;
            return true;
        }
    }
    false
}

// ── Greedy hitting set ────────────────────────────────────────────────────────

/// Greedy hitting set: find a small set H ⊆ {0..universe-1} such that every
/// input set contains at least one element of H.
///
/// Equivalent to set cover on the dual hypergraph.  Greedily picks the element
/// that "hits" the most unhit sets.
///
/// Returns the hitting set as a sorted list of element indices.
pub fn hitting_set(universe: usize, sets: &[Vec<usize>]) -> Vec<usize> {
    if universe == 0 || sets.is_empty() {
        return vec![];
    }

    let m = sets.len();
    let mut hit = vec![false; m]; // whether set i has been hit
    let mut remaining = m;
    let mut chosen = Vec::new();
    let mut in_hitting_set = vec![false; universe];

    while remaining > 0 {
        // For each element, count how many unhit sets it appears in
        let mut best_elem = None;
        let mut best_count = 0usize;

        for e in 0..universe {
            if in_hitting_set[e] {
                continue;
            }
            let count = sets
                .iter()
                .enumerate()
                .filter(|(i, s)| !hit[*i] && s.contains(&e))
                .count();
            if count > best_count {
                best_count = count;
                best_elem = Some(e);
            }
        }

        match best_elem {
            Some(e) => {
                in_hitting_set[e] = true;
                chosen.push(e);
                for (i, s) in sets.iter().enumerate() {
                    if !hit[i] && s.contains(&e) {
                        hit[i] = true;
                        remaining -= 1;
                    }
                }
            }
            None => break, // some sets cannot be hit (empty sets)
        }
    }

    chosen.sort_unstable();
    chosen
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    fn is_set_cover(universe: usize, sets: &[Vec<usize>], selected: &[usize]) -> bool {
        let mut covered = vec![false; universe];
        for &idx in selected {
            for &e in &sets[idx] {
                if e < universe {
                    covered[e] = true;
                }
            }
        }
        covered.iter().all(|&c| c)
    }

    #[test]
    fn test_greedy_set_cover() {
        let sets = vec![vec![0, 1, 2], vec![1, 3], vec![2, 4], vec![3, 4]];
        let sel = greedy_set_cover(5, &sets);
        assert!(is_set_cover(5, &sets, &sel));
        assert!(!sel.is_empty());
    }

    #[test]
    fn test_weighted_set_cover() {
        let sets = vec![vec![0, 1, 2], vec![1, 3], vec![2, 4], vec![3, 4]];
        let costs = vec![3.0, 1.0, 1.0, 1.0];
        let (sel, cost) = weighted_set_cover(5, &sets, &costs).expect("unexpected None or Err");
        assert!(is_set_cover(5, &sets, &sel));
        assert!(cost > 0.0);
    }

    #[test]
    fn test_weighted_set_cover_mismatch() {
        let sets = vec![vec![0, 1]];
        let costs = vec![1.0, 2.0]; // length mismatch
        assert!(weighted_set_cover(2, &sets, &costs).is_err());
    }

    #[test]
    fn test_vertex_cover_2approx() {
        // Triangle: needs 2 vertices to cover
        let edges = vec![(0, 1), (1, 2), (2, 0)];
        let cover = vertex_cover_2approx(3, &edges);
        // Verify it's a valid cover
        for &(u, v) in &edges {
            assert!(cover.contains(&u) || cover.contains(&v));
        }
    }

    #[test]
    fn test_vertex_cover_path() {
        // Path 0-1-2-3: minimum cover = {1, 2} or {1, 3}
        let edges = vec![(0, 1), (1, 2), (2, 3)];
        let cover = vertex_cover_2approx(4, &edges);
        for &(u, v) in &edges {
            assert!(cover.contains(&u) || cover.contains(&v));
        }
        assert!(cover.len() <= 4); // at most 2 * OPT = 4
    }

    #[test]
    fn test_kings_bipartite() {
        // K_{2,2}: min vertex cover = 2 (by König = max matching = 2)
        let edges = vec![(0, 0), (0, 1), (1, 0), (1, 1)];
        let cover = min_vertex_cover_bip(2, 2, &edges);
        // Verify it covers all edges
        for &(u, v) in &edges {
            assert!(cover.contains(&u) || cover.contains(&(2 + v)));
        }
        assert_eq!(cover.len(), 2);
    }

    #[test]
    fn test_kings_path_bip() {
        // Path: left={0}, right={0}, edge (0,0) → min cover = 1
        let edges = vec![(0, 0)];
        let cover = min_vertex_cover_bip(1, 1, &edges);
        assert_eq!(cover.len(), 1);
        assert!(cover.contains(&0) || cover.contains(&1));
    }

    #[test]
    fn test_hitting_set() {
        // Universe 5, sets: {0,1},{1,2},{2,3},{3,4}
        let sets = vec![vec![0, 1], vec![1, 2], vec![2, 3], vec![3, 4]];
        let hs = hitting_set(5, &sets);
        // Verify every set is hit
        for s in &sets {
            assert!(s.iter().any(|e| hs.contains(e)));
        }
    }

    #[test]
    fn test_empty_inputs() {
        assert!(greedy_set_cover(0, &[]).is_empty());
        assert!(vertex_cover_2approx(0, &[]).is_empty());
        assert!(hitting_set(0, &[]).is_empty());
    }
}
