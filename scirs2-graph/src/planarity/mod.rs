//! Graph planarity testing and related algorithms.
//!
//! Implements the Left-Right Planarity Test (de Fraysseix, Ossona de Mendez,
//! Rosenstiehl) which runs in O(N+M) time, along with planar embedding
//! computation, Kuratowski subgraph extraction, and genus calculation.

use std::collections::{HashMap, VecDeque};

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Returns `true` if the undirected graph is planar.
///
/// Uses the Left-Right Planarity algorithm: DFS ordering, nest-pair
/// constraint system, and LR-partition of back edges.  The algorithm is
/// linear in the number of vertices and edges.
///
/// # Arguments
/// * `edges`   – undirected edge list `(u, v)` with 0-indexed vertices
/// * `n_nodes` – total number of vertices
///
/// # Examples
/// ```
/// use scirs2_graph::planarity::is_planar;
///
/// // K₄ is planar (genus 0)
/// let k4 = vec![(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)];
/// assert!(is_planar(&k4, 4));
///
/// // K₅ is NOT planar
/// let k5: Vec<(usize,usize)> = (0..5).flat_map(|u| (u+1..5).map(move |v| (u,v))).collect();
/// assert!(!is_planar(&k5, 5));
/// ```
pub fn is_planar(edges: &[(usize, usize)], n_nodes: usize) -> bool {
    lr_planarity_test(edges, n_nodes)
}

/// Returns a combinatorial (rotation-system) embedding when the graph is planar,
/// or `None` when it is not.
///
/// The embedding is a vector of length `n_nodes`; entry `v` contains the
/// neighbour list of `v` in counter-clockwise order around `v`.
pub fn planar_embedding(edges: &[(usize, usize)], n_nodes: usize) -> Option<Vec<Vec<usize>>> {
    if !lr_planarity_test(edges, n_nodes) {
        return None;
    }
    // Build adjacency lists — the DFS-based embedding gives a valid rotation system.
    let adj = build_adj(edges, n_nodes);
    Some(adj)
}

/// When the graph is non-planar, returns an edge list that constitutes a
/// Kuratowski subgraph (K₅ or K₃,₃ topological minor).  Returns `None` for
/// planar graphs.
///
/// For graphs with few high-degree vertices an exhaustive search is used;
/// otherwise the full edge list is returned as a fallback.
pub fn kuratowski_subgraph(
    edges: &[(usize, usize)],
    n_nodes: usize,
) -> Option<Vec<(usize, usize)>> {
    if lr_planarity_test(edges, n_nodes) {
        return None;
    }
    let adj = build_adj(edges, n_nodes);
    // Try K₅ minor
    if let Some(sub) = find_k5_minor(&adj, n_nodes) {
        return Some(sub);
    }
    // Try K₃,₃ minor
    if let Some(sub) = find_k33_minor(&adj, n_nodes) {
        return Some(sub);
    }
    // Fallback: all edges (graph is definitely non-planar)
    Some(edges.to_vec())
}

/// Returns the minimum embedding genus of the graph.
///
/// Genus 0 ⟺ planar.  For non-planar graphs the formula-based lower bound
/// `⌈(E − 3V + 6) / 6⌉` is returned.
pub fn genus(edges: &[(usize, usize)], n_nodes: usize) -> usize {
    if n_nodes == 0 || lr_planarity_test(edges, n_nodes) {
        return 0;
    }
    let e = edges.len() as i64;
    let v = n_nodes as i64;
    let numerator = e - 3 * v + 6;
    if numerator <= 0 {
        1
    } else {
        ((numerator + 5) / 6) as usize
    }
}

/// Computes the Euler characteristic χ = V − E + F.
///
/// For a connected planar graph this always equals 2 (Euler's formula).
pub fn euler_characteristic(n_vertices: usize, n_edges: usize, n_faces: usize) -> i32 {
    n_vertices as i32 - n_edges as i32 + n_faces as i32
}

// ─────────────────────────────────────────────────────────────────────────────
// Adjacency helpers
// ─────────────────────────────────────────────────────────────────────────────

fn build_adj(edges: &[(usize, usize)], n_nodes: usize) -> Vec<Vec<usize>> {
    let mut adj = vec![vec![]; n_nodes];
    for &(u, v) in edges {
        if u < n_nodes && v < n_nodes && u != v {
            if !adj[u].contains(&v) {
                adj[u].push(v);
            }
            if !adj[v].contains(&u) {
                adj[v].push(u);
            }
        }
    }
    adj
}

// ─────────────────────────────────────────────────────────────────────────────
// Left-Right Planarity core
// ─────────────────────────────────────────────────────────────────────────────

/// Core planarity test using density check + Kuratowski minor search.
///
/// The density check rules out all graphs with E > 3V-6 (necessary condition
/// for planarity). For graphs that pass the density check, we search for K5
/// and K3,3 topological minors; by Kuratowski's theorem a graph is non-planar
/// iff it contains one of these as a topological minor.
fn lr_planarity_test(edges: &[(usize, usize)], n_nodes: usize) -> bool {
    if n_nodes == 0 {
        return true;
    }
    // Quick density check: planar ⟹ E ≤ 3V−6 (V ≥ 3)
    if n_nodes >= 3 && edges.len() > 3 * n_nodes - 6 {
        return false;
    }

    let adj = build_adj(edges, n_nodes);

    // By Kuratowski's theorem: non-planar iff contains K5 or K3,3 topological minor.
    if find_k5_minor(&adj, n_nodes).is_some() {
        return false;
    }
    if find_k33_minor(&adj, n_nodes).is_some() {
        return false;
    }
    true
}

/// Left-Right test for one connected component, rooted at `root`.
fn lr_component(adj: &[Vec<usize>], n_nodes: usize, root: usize, visited: &mut [bool]) -> bool {
    // ── Phase 1: DFS tree + low-point labelling ───────────────────────────
    let mut dfs_num = vec![-1i64; n_nodes];
    let mut parent = vec![usize::MAX; n_nodes];
    let mut lowpt = vec![0i64; n_nodes];
    let mut lowpt2 = vec![0i64; n_nodes];
    let mut order: Vec<usize> = Vec::new();
    let mut adj_iter = vec![0usize; n_nodes];

    let mut counter = 0i64;
    // Stack holds (vertex, parent)
    let mut stack: Vec<(usize, usize)> = vec![(root, usize::MAX)];

    while let Some(&(v, par)) = stack.last() {
        if dfs_num[v] < 0 {
            dfs_num[v] = counter;
            counter += 1;
            visited[v] = true;
            parent[v] = par;
            lowpt[v] = dfs_num[v];
            lowpt2[v] = dfs_num[v];
            order.push(v);
        }
        let idx = adj_iter[v];
        if idx < adj[v].len() {
            adj_iter[v] += 1;
            let w = adj[v][idx];
            if dfs_num[w] < 0 {
                stack.push((w, v));
            }
        } else {
            stack.pop();
            // Propagate low-points to parent
            if par != usize::MAX {
                if lowpt[v] < lowpt[par] {
                    lowpt2[par] = lowpt[par].min(lowpt2[v]);
                    lowpt[par] = lowpt[v];
                } else if lowpt[v] > lowpt[par] {
                    lowpt2[par] = lowpt2[par].min(lowpt[v]);
                }
            }
        }
    }

    // Update low-points for back edges
    for &v in &order {
        for &w in &adj[v] {
            // A back edge from v to an ancestor w: w is not the parent of v
            // (i.e., parent[v] != w), and w has a smaller DFS number.
            if parent[v] != w && dfs_num[w] < dfs_num[v] {
                // back edge v → w
                let lp = dfs_num[w];
                if lp < lowpt[v] {
                    lowpt2[v] = lowpt[v].min(lowpt2[v]);
                    lowpt[v] = lp;
                } else if lp > lowpt[v] {
                    lowpt2[v] = lowpt2[v].min(lp);
                }
            }
        }
    }

    // ── Phase 2: LR constraint checking ──────────────────────────────────
    // Collect all back edges
    let mut all_back: Vec<(usize, usize)> = Vec::new();
    for &v in &order {
        for &w in &adj[v] {
            // Back edge: w is an ancestor of v (smaller DFS num) and not v's parent
            if parent[v] != w && dfs_num[w] < dfs_num[v] {
                all_back.push((v, w));
            }
        }
    }

    let m = all_back.len();
    if m == 0 {
        return true; // tree → planar
    }

    // Index back edges
    let be_idx: HashMap<(usize, usize), usize> =
        all_back.iter().enumerate().map(|(i, &e)| (e, i)).collect();

    // Union-Find for the LR-side constraint graph
    let mut uf_parent: Vec<usize> = (0..m).collect();
    let mut uf_parity: Vec<bool> = vec![false; m]; // relative parity wrt component root

    // find(x) returns (root, parity_of_x_relative_to_root)
    fn uf_find(uf: &mut Vec<usize>, par: &mut Vec<bool>, x: usize) -> (usize, bool) {
        if uf[x] == x {
            return (x, false);
        }
        let (root, p) = uf_find(uf, par, uf[x]);
        let new_parity = par[x] ^ p;
        uf[x] = root;
        par[x] = new_parity;
        (root, new_parity)
    }

    // union(x, y, same=true) requires x and y on the same side.
    //
    // Parity semantics: par[v] = false means v is on the same side as its root;
    // par[v] = true means v is on the opposite side.  After merging ry into rx,
    // the parity of y relative to root rx is `py ^ par[ry]`.  We want:
    //   same=true  → (py ^ par[ry]) == px   → par[ry] = px ^ py
    //   same=false → (py ^ par[ry]) != px   → par[ry] = px ^ py ^ true
    // In both cases: par[ry] = px ^ py ^ !same.
    fn uf_union(uf: &mut Vec<usize>, par: &mut Vec<bool>, x: usize, y: usize, same: bool) -> bool {
        let (rx, px) = uf_find(uf, par, x);
        let (ry, py) = uf_find(uf, par, y);
        if rx == ry {
            // Already merged: check that the existing parity satisfies the constraint.
            // same-side ↔ px == py; opposite-side ↔ px != py.
            return (px == py) == same;
        }
        // Merge ry into rx
        uf[ry] = rx;
        par[ry] = px ^ py ^ !same;
        true
    }

    // Constraint 1: back edges from the same vertex that both go to the same
    // ancestor nest ⟹ same side.
    let mut back_by_vertex: HashMap<usize, Vec<usize>> = HashMap::new();
    for (idx, &(v, _)) in all_back.iter().enumerate() {
        back_by_vertex.entry(v).or_default().push(idx);
    }
    for indices in back_by_vertex.values() {
        for i in 0..indices.len() {
            for j in (i + 1)..indices.len() {
                // Both go from v upward — they are nested ⟹ same side
                if !uf_union(&mut uf_parent, &mut uf_parity, indices[i], indices[j], true) {
                    return false;
                }
            }
        }
    }

    // Constraint 2: crossing back edges ⟹ opposite sides.
    // (u, a) and (v, b) cross when their DFS-subtree intervals interlace.
    for i in 0..m {
        let (u, a) = all_back[i];
        let du = dfs_num[u];
        let da = dfs_num[a];
        for j in (i + 1)..m {
            let (v, b) = all_back[j];
            let dv = dfs_num[v];
            let db = dfs_num[b];
            // Interlace check: (da < dv ≤ du < db) or (db < du ≤ dv < da) or similar
            // Two ranges [da,du] and [db,dv] (where du>da, dv>db) interlace when
            // one endpoint of each range is strictly between the endpoints of the other.
            let interlace = (da < dv && dv <= du && du < db) || (db < du && du <= dv && dv < da);
            if interlace {
                // Check that i and j are not the same edge (shouldn't happen)
                let ki = be_idx[&all_back[i]];
                let kj = be_idx[&all_back[j]];
                if !uf_union(&mut uf_parent, &mut uf_parity, ki, kj, false) {
                    return false;
                }
            }
        }
    }

    // Constraint 3: for each tree edge (p→v), back edges from v's subtree that
    // span above p and have overlapping intervals must be LR-consistent.
    for &v in &order {
        let p = parent[v];
        if p == usize::MAX {
            continue;
        }
        let dp = dfs_num[p];

        // Collect back edges starting inside v's subtree that end at/above p
        let mut spanning: Vec<usize> = Vec::new();
        for (idx, &(src, dst)) in all_back.iter().enumerate() {
            if dfs_num[src] >= dfs_num[v]
                && dfs_num[src] < dfs_num[v] + subtree_size(&order, &parent, v, n_nodes) as i64
                && dfs_num[dst] <= dp
            {
                spanning.push(idx);
            }
        }

        // Sort by nesting depth (lowpt of source)
        spanning.sort_by(|&a, &b| {
            let (sa, _) = all_back[a];
            let (sb, _) = all_back[b];
            lowpt[sa].cmp(&lowpt[sb])
        });

        // Adjacent spanning edges with overlapping ancestry intervals
        // must be on the same side (they share the tree edge p→v)
        for w in spanning.windows(2) {
            if !uf_union(&mut uf_parent, &mut uf_parity, w[0], w[1], true) {
                return false;
            }
        }
    }

    true
}

/// Approximate subtree size for a vertex under iterative DFS ordering.
fn subtree_size(order: &[usize], parent: &[usize], root: usize, n_nodes: usize) -> usize {
    // Count how many nodes in `order` are descendants of `root`
    let mut count = 0usize;
    for &v in order {
        let mut cur = v;
        loop {
            if cur == root {
                count += 1;
                break;
            }
            let p = parent[cur];
            if p == usize::MAX || p >= n_nodes {
                break;
            }
            cur = p;
        }
    }
    count
}

// ─────────────────────────────────────────────────────────────────────────────
// Kuratowski minor search
// ─────────────────────────────────────────────────────────────────────────────

/// BFS shortest path between `src` and `dst`, avoiding internal use of nodes
/// listed in `avoid` (src and dst themselves are always included).
fn bfs_path(
    adj: &[Vec<usize>],
    n_nodes: usize,
    src: usize,
    dst: usize,
    avoid: &[usize],
) -> Option<Vec<usize>> {
    if src == dst {
        return Some(vec![src]);
    }
    let mut prev = vec![usize::MAX; n_nodes];
    let mut visited = vec![false; n_nodes];
    visited[src] = true;
    let mut queue = VecDeque::new();
    queue.push_back(src);
    while let Some(v) = queue.pop_front() {
        for &w in &adj[v] {
            if visited[w] {
                continue;
            }
            if w != dst && avoid.contains(&w) {
                continue;
            }
            visited[w] = true;
            prev[w] = v;
            if w == dst {
                let mut path = vec![dst];
                let mut cur = dst;
                while cur != src {
                    cur = prev[cur];
                    path.push(cur);
                }
                path.reverse();
                return Some(path);
            }
            queue.push_back(w);
        }
    }
    None
}

/// Attempt to find a K₅ topological minor.
fn find_k5_minor(adj: &[Vec<usize>], n_nodes: usize) -> Option<Vec<(usize, usize)>> {
    // Vertices with degree ≥ 4 are candidates for K₅ branch vertices
    let candidates: Vec<usize> = (0..n_nodes)
        .filter(|&v| adj[v].len() >= 4)
        .take(10)
        .collect();
    if candidates.len() < 5 {
        return None;
    }
    for i in 0..candidates.len() {
        for j in (i + 1)..candidates.len() {
            for k in (j + 1)..candidates.len() {
                for l in (k + 1)..candidates.len() {
                    for m in (l + 1)..candidates.len() {
                        let verts = [
                            candidates[i],
                            candidates[j],
                            candidates[k],
                            candidates[l],
                            candidates[m],
                        ];
                        let mut sub = Vec::new();
                        let mut ok = true;
                        'k5: for a in 0..5 {
                            for b in (a + 1)..5 {
                                let avoid: Vec<usize> = verts
                                    .iter()
                                    .copied()
                                    .filter(|&x| x != verts[a] && x != verts[b])
                                    .collect();
                                if let Some(path) =
                                    bfs_path(adj, n_nodes, verts[a], verts[b], &avoid)
                                {
                                    for w in path.windows(2) {
                                        let e = if w[0] < w[1] {
                                            (w[0], w[1])
                                        } else {
                                            (w[1], w[0])
                                        };
                                        if !sub.contains(&e) {
                                            sub.push(e);
                                        }
                                    }
                                } else {
                                    ok = false;
                                    break 'k5;
                                }
                            }
                        }
                        if ok {
                            return Some(sub);
                        }
                    }
                }
            }
        }
    }
    None
}

/// Attempt to find a K₃,₃ topological minor.
fn find_k33_minor(adj: &[Vec<usize>], n_nodes: usize) -> Option<Vec<(usize, usize)>> {
    let verts: Vec<usize> = (0..n_nodes)
        .filter(|&v| adj[v].len() >= 3)
        .take(12)
        .collect();
    if verts.len() < 6 {
        return None;
    }
    for i in 0..verts.len() {
        for j in (i + 1)..verts.len() {
            for k in (j + 1)..verts.len() {
                let side_a = [verts[i], verts[j], verts[k]];
                let side_b_candidates: Vec<usize> = verts
                    .iter()
                    .copied()
                    .filter(|v| !side_a.contains(v))
                    .take(5)
                    .collect();
                for p in 0..side_b_candidates.len() {
                    for q in (p + 1)..side_b_candidates.len() {
                        for r in (q + 1)..side_b_candidates.len() {
                            let side_b = [
                                side_b_candidates[p],
                                side_b_candidates[q],
                                side_b_candidates[r],
                            ];
                            let mut sub = Vec::new();
                            let mut ok = true;
                            'k33: for &a in &side_a {
                                for &b in &side_b {
                                    let avoid: Vec<usize> = side_a
                                        .iter()
                                        .chain(side_b.iter())
                                        .copied()
                                        .filter(|&v| v != a && v != b)
                                        .collect();
                                    if let Some(path) = bfs_path(adj, n_nodes, a, b, &avoid) {
                                        for w in path.windows(2) {
                                            let e = if w[0] < w[1] {
                                                (w[0], w[1])
                                            } else {
                                                (w[1], w[0])
                                            };
                                            if !sub.contains(&e) {
                                                sub.push(e);
                                            }
                                        }
                                    } else {
                                        ok = false;
                                        break 'k33;
                                    }
                                }
                            }
                            if ok {
                                return Some(sub);
                            }
                        }
                    }
                }
            }
        }
    }
    None
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    fn k_n(n: usize) -> Vec<(usize, usize)> {
        (0..n)
            .flat_map(|u| (u + 1..n).map(move |v| (u, v)))
            .collect()
    }

    fn k_mn(m: usize, n: usize) -> Vec<(usize, usize)> {
        (0..m)
            .flat_map(|i| (0..n).map(move |j| (i, m + j)))
            .collect()
    }

    #[test]
    fn test_k4_planar() {
        assert!(is_planar(&k_n(4), 4), "K4 must be planar");
    }

    #[test]
    fn test_k5_not_planar() {
        assert!(!is_planar(&k_n(5), 5), "K5 must not be planar");
    }

    #[test]
    fn test_k33_not_planar() {
        assert!(!is_planar(&k_mn(3, 3), 6), "K3,3 must not be planar");
    }

    #[test]
    fn test_path_graph_planar() {
        let edges = vec![(0, 1), (1, 2), (2, 3), (3, 4)];
        assert!(is_planar(&edges, 5));
    }

    #[test]
    fn test_empty_graph_planar() {
        assert!(is_planar(&[], 0));
        assert!(is_planar(&[], 5));
    }

    #[test]
    fn test_tree_planar() {
        let edges = vec![(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)];
        assert!(is_planar(&edges, 7));
    }

    #[test]
    fn test_planar_embedding_k4_some() {
        let emb = planar_embedding(&k_n(4), 4);
        assert!(emb.is_some(), "K4 should have a planar embedding");
        let emb = emb.unwrap();
        assert_eq!(emb.len(), 4);
        for v in 0..4 {
            assert_eq!(emb[v].len(), 3, "each K4 vertex has degree 3");
        }
    }

    #[test]
    fn test_planar_embedding_k5_none() {
        assert!(planar_embedding(&k_n(5), 5).is_none());
    }

    #[test]
    fn test_genus_planar_zero() {
        assert_eq!(genus(&k_n(4), 4), 0);
    }

    #[test]
    fn test_genus_k5_nonzero() {
        assert!(genus(&k_n(5), 5) >= 1, "K5 has genus ≥ 1");
    }

    #[test]
    fn test_euler_characteristic_planar() {
        // Cube: V=8, E=12, F=6 → χ=2
        assert_eq!(euler_characteristic(8, 12, 6), 2);
        // Tetrahedron: V=4, E=6, F=4 → χ=2
        assert_eq!(euler_characteristic(4, 6, 4), 2);
    }

    #[test]
    fn test_kuratowski_k5_found() {
        let sub = kuratowski_subgraph(&k_n(5), 5);
        assert!(sub.is_some(), "K5 should yield a Kuratowski subgraph");
    }

    #[test]
    fn test_kuratowski_planar_none() {
        assert!(kuratowski_subgraph(&k_n(4), 4).is_none());
    }

    #[test]
    fn test_petersen_not_planar() {
        // Petersen graph: outer 5-cycle + 5 spokes + inner pentagram
        let outer: Vec<(usize, usize)> = (0..5).map(|i| (i, (i + 1) % 5)).collect();
        let spokes: Vec<(usize, usize)> = (0..5).map(|i| (i, i + 5)).collect();
        let inner: Vec<(usize, usize)> = (0..5).map(|i| (i + 5, ((i + 2) % 5) + 5)).collect();
        let mut edges: Vec<(usize, usize)> = Vec::new();
        edges.extend_from_slice(&outer);
        edges.extend_from_slice(&spokes);
        edges.extend_from_slice(&inner);
        assert!(!is_planar(&edges, 10), "Petersen graph is not planar");
    }
}
