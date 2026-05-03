//! Additional network reliability algorithms: factoring-based exact computation,
//! k-connectivity, and reliability bounds.
//!
//! These supplement the Monte-Carlo and BDD methods in `network_reliability.rs`.

use std::collections::{HashMap, VecDeque};

/// Memoisation cache key: canonical sorted edge list paired with node count.
type FactoringMemo = HashMap<(Vec<(usize, usize)>, usize), f64>;

// ─────────────────────────────────────────────────────────────────────────────
// All-terminal reliability via factoring
// ─────────────────────────────────────────────────────────────────────────────

/// Computes all-terminal reliability by the factoring algorithm:
/// R(G) = p · R(G/e) + (1 − p) · R(G − e)
///
/// where G/e contracts edge e and G − e deletes edge e.
/// Base cases: R(K_n) = 1, R(disconnected) = 0.
///
/// Memoisation is keyed on the canonical edge-sorted representation of the
/// current multi-graph.
///
/// # Arguments
/// * `edges`   – undirected edge list (may contain parallel edges)
/// * `n_nodes` – number of vertices
/// * `p`       – per-edge survival probability
pub fn all_terminal_reliability_factoring(edges: &[(usize, usize)], n_nodes: usize, p: f64) -> f64 {
    if n_nodes <= 1 {
        return 1.0;
    }
    // Normalise edge representation (remove self-loops, canonicalise direction)
    let clean: Vec<(usize, usize)> = edges
        .iter()
        .filter(|&&(u, v)| u != v && u < n_nodes && v < n_nodes)
        .map(|&(u, v)| if u < v { (u, v) } else { (v, u) })
        .collect();

    if !is_connected(&clean, n_nodes) {
        return 0.0;
    }

    // Memo key must include n_nodes: the same edge set with fewer nodes represents
    // a different graph (e.g., empty edges for n=1 are connected, for n=2 are not).
    let mut memo: FactoringMemo = HashMap::new();
    factoring_recurse(&clean, n_nodes, p, &mut memo)
}

fn factoring_recurse(
    edges: &[(usize, usize)],
    n_nodes: usize,
    p: f64,
    memo: &mut FactoringMemo,
) -> f64 {
    // Sort for canonical key; include n_nodes to distinguish graphs with the
    // same edge multiset but different vertex counts (arising after contractions).
    let mut sorted = edges.to_vec();
    sorted.sort_unstable();
    let key = (sorted.clone(), n_nodes);

    if let Some(&cached) = memo.get(&key) {
        return cached;
    }

    // Base cases
    if n_nodes <= 1 {
        let result = 1.0;
        memo.insert(key, result);
        return result;
    }
    if !is_connected(edges, n_nodes) {
        memo.insert(key, 0.0);
        return 0.0;
    }
    if edges.is_empty() {
        // Connected with no edges only possible for n=1
        let result = if n_nodes == 1 { 1.0 } else { 0.0 };
        memo.insert(key, result);
        return result;
    }

    // Pick first edge to factor on
    let (eu, ev) = sorted[0];
    let rest: Vec<(usize, usize)> = sorted[1..].to_vec();

    // G − e: delete edge (eu, ev)
    let rel_delete = factoring_recurse(&rest, n_nodes, p, memo);

    // G / e: contract edge (eu, ev) — merge ev into eu
    let contracted = contract_edge(&rest, n_nodes, eu, ev);
    let new_n = n_nodes - 1; // one vertex merged away
    let rel_contract = factoring_recurse(&contracted, new_n, p, memo);

    let result = p * rel_contract + (1.0 - p) * rel_delete;
    memo.insert(key, result);
    result
}

/// Contract edge (eu, ev): replace all occurrences of ev with eu and
/// remove resulting self-loops.
fn contract_edge(
    edges: &[(usize, usize)],
    _n_nodes: usize,
    keep: usize,
    remove: usize,
) -> Vec<(usize, usize)> {
    // Remap: vertices > remove shift down by 1
    let remap = |v: usize| -> usize {
        let v2 = if v == remove { keep } else { v };
        // Shift down vertices that are above the removed vertex
        if v2 > remove {
            v2 - 1
        } else {
            v2
        }
    };
    let keep_mapped = remap(keep);

    edges
        .iter()
        .filter_map(|&(u, v)| {
            let u2 = remap(u);
            let v2 = remap(v);
            if u2 == v2 {
                None // self-loop
            } else {
                let e = if u2 < v2 { (u2, v2) } else { (v2, u2) };
                Some(e)
            }
        })
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Two-terminal reliability via path enumeration
// ─────────────────────────────────────────────────────────────────────────────

/// Computes two-terminal reliability P(s connected to t) using inclusion-
/// exclusion over all simple paths between s and t (exact for small graphs).
///
/// For larger graphs (many paths) an upper bound is returned based on the
/// most reliable single path.
pub fn two_terminal_reliability(
    edges: &[(usize, usize)],
    n_nodes: usize,
    source: usize,
    target: usize,
    p: f64,
) -> f64 {
    if source == target {
        return 1.0;
    }
    if source >= n_nodes || target >= n_nodes {
        return 0.0;
    }
    let adj = build_adj(edges, n_nodes);
    // Enumerate simple paths (up to a limit)
    let paths = enumerate_simple_paths(&adj, source, target, n_nodes, 500);
    if paths.is_empty() {
        return 0.0;
    }
    // Inclusion-exclusion over path events
    // For large numbers of paths, use an approximation
    if paths.len() <= 20 {
        inclusion_exclusion_reliability(&paths, edges, n_nodes, p)
    } else {
        // Upper bound: 1 − Π(1 − P(path_i reliable))
        let approx: f64 = paths
            .iter()
            .map(|path| {
                path.windows(2)
                    .map(|w| {
                        if edge_exists(edges, w[0], w[1]) {
                            p
                        } else {
                            0.0
                        }
                    })
                    .product::<f64>()
            })
            .fold(1.0f64, |acc, pp| acc * (1.0 - pp));
        1.0 - approx
    }
}

/// Exact inclusion-exclusion over path events.
fn inclusion_exclusion_reliability(
    paths: &[Vec<usize>],
    edges: &[(usize, usize)],
    n_nodes: usize,
    p: f64,
) -> f64 {
    let m = paths.len();
    let mut result = 0.0f64;
    for mask in 1u64..(1u64 << m) {
        // Collect union of edges in the selected paths
        let mut edge_set: std::collections::HashSet<(usize, usize)> =
            std::collections::HashSet::new();
        let mut count = 0u32;
        for i in 0..m {
            if mask & (1 << i) != 0 {
                count += 1;
                let path = &paths[i];
                for w in path.windows(2) {
                    let e = if w[0] < w[1] {
                        (w[0], w[1])
                    } else {
                        (w[1], w[0])
                    };
                    edge_set.insert(e);
                }
            }
        }
        // Probability all edges in the union survive
        let prob: f64 = edge_set
            .iter()
            .map(|&(u, v)| if edge_exists(edges, u, v) { p } else { 0.0 })
            .product();
        // Inclusion-exclusion sign
        if count % 2 == 1 {
            result += prob;
        } else {
            result -= prob;
        }
    }
    // Clamp to [0,1]
    result.clamp(0.0, 1.0)
}

fn edge_exists(edges: &[(usize, usize)], u: usize, v: usize) -> bool {
    edges
        .iter()
        .any(|&(a, b)| (a == u && b == v) || (a == v && b == u))
}

/// Enumerate simple paths from `src` to `dst` (up to `limit` paths).
fn enumerate_simple_paths(
    adj: &[Vec<usize>],
    src: usize,
    dst: usize,
    n: usize,
    limit: usize,
) -> Vec<Vec<usize>> {
    let mut paths = Vec::new();
    let mut visited = vec![false; n];
    let mut current_path = vec![src];
    visited[src] = true;
    dfs_paths(
        adj,
        src,
        dst,
        &mut visited,
        &mut current_path,
        &mut paths,
        limit,
    );
    paths
}

fn dfs_paths(
    adj: &[Vec<usize>],
    v: usize,
    dst: usize,
    visited: &mut Vec<bool>,
    path: &mut Vec<usize>,
    paths: &mut Vec<Vec<usize>>,
    limit: usize,
) {
    if paths.len() >= limit {
        return;
    }
    if v == dst {
        paths.push(path.clone());
        return;
    }
    for &w in &adj[v] {
        if !visited[w] {
            visited[w] = true;
            path.push(w);
            dfs_paths(adj, w, dst, visited, path, paths, limit);
            path.pop();
            visited[w] = false;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Reliability polynomial (exact, spanning subgraphs)
// ─────────────────────────────────────────────────────────────────────────────

/// Computes the coefficients of the reliability polynomial R(G, p) =
/// Σ_i  c_i · p^i  for small graphs via spanning subgraph enumeration.
///
/// Returns coefficient vector `[c_0, c_1, ..., c_m]` (length = m+1 where
/// m = number of edges).
///
/// # Complexity
/// O(2^m · (N + M)) — practical only for m ≤ 25.
pub fn reliability_polynomial(edges: &[(usize, usize)], n_nodes: usize) -> Vec<f64> {
    let clean: Vec<(usize, usize)> = edges
        .iter()
        .filter(|&&(u, v)| u != v && u < n_nodes && v < n_nodes)
        .map(|&(u, v)| if u < v { (u, v) } else { (v, u) })
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    let m = clean.len();
    if m > 25 {
        // Too large for exact enumeration — return empty
        return vec![];
    }
    let mut coeffs = vec![0i64; m + 1];
    for mask in 0u32..(1u32 << m) {
        let k = mask.count_ones() as usize;
        let sub: Vec<(usize, usize)> = (0..m)
            .filter(|&i| mask & (1 << i) != 0)
            .map(|i| clean[i])
            .collect();
        if is_connected_subset(&sub, n_nodes) {
            coeffs[k] += 1;
        }
    }
    // Polynomial: R(p) = sum_k coeffs[k] * p^k * (1-p)^(m-k)
    // Expand into standard polynomial coefficients
    let mut poly = vec![0.0f64; m + 1];
    for k in 0..=m {
        if coeffs[k] == 0 {
            continue;
        }
        // p^k * (1-p)^(m-k) = sum_j binom(m-k, j) (-1)^j p^{k+j}
        for j in 0..=(m - k) {
            let sign = if j % 2 == 0 { 1.0 } else { -1.0 };
            let binom = binom_coeff(m - k, j) as f64;
            poly[k + j] += coeffs[k] as f64 * sign * binom;
        }
    }
    poly
}

fn binom_coeff(n: usize, k: usize) -> u64 {
    if k > n {
        return 0;
    }
    if k == 0 || k == n {
        return 1;
    }
    let k = k.min(n - k);
    let mut result = 1u64;
    for i in 0..k {
        result = result * (n - i) as u64 / (i + 1) as u64;
    }
    result
}

fn is_connected_subset(edges: &[(usize, usize)], n: usize) -> bool {
    if n <= 1 {
        return true;
    }
    is_connected(edges, n)
}

// ─────────────────────────────────────────────────────────────────────────────
// Reliability bound via minimum cut
// ─────────────────────────────────────────────────────────────────────────────

/// Returns an upper bound on all-terminal reliability:
/// R(G) ≤ p^{λ} where λ is the edge-connectivity.
pub fn min_cut_reliability_bound(edges: &[(usize, usize)], n_nodes: usize, p: f64) -> f64 {
    let lambda = k_edge_connectivity(edges, n_nodes);
    p.powi(lambda as i32)
}

// ─────────────────────────────────────────────────────────────────────────────
// Edge and vertex connectivity
// ─────────────────────────────────────────────────────────────────────────────

/// Returns the edge connectivity κ'(G): minimum number of edges whose removal
/// disconnects the graph.
///
/// Computed by repeated max-flow (Ford-Fulkerson) from a fixed source to every
/// other vertex; the minimum flow value is the edge connectivity.
pub fn k_edge_connectivity(edges: &[(usize, usize)], n_nodes: usize) -> usize {
    if n_nodes <= 1 {
        return 0;
    }
    if !is_connected(edges, n_nodes) {
        return 0;
    }
    // Edge connectivity = min over all t ≠ 0 of max_flow(0, t)
    let mut min_flow = usize::MAX;
    for t in 1..n_nodes {
        let f = max_flow_bfs(edges, n_nodes, 0, t);
        if f < min_flow {
            min_flow = f;
        }
    }
    if min_flow == usize::MAX {
        0
    } else {
        min_flow
    }
}

/// Returns the vertex connectivity κ(G): minimum number of vertices whose
/// removal disconnects the graph (or reduces it to a single vertex).
///
/// Uses node-splitting: each vertex v is split into v_in and v_out with a
/// unit-capacity edge; then max-flow gives the minimum vertex cut.
pub fn k_vertex_connectivity(edges: &[(usize, usize)], n_nodes: usize) -> usize {
    if n_nodes <= 1 {
        return 0;
    }
    if !is_connected(edges, n_nodes) {
        return 0;
    }
    // For complete graphs the vertex connectivity is n-1
    let edge_set: std::collections::HashSet<(usize, usize)> = edges
        .iter()
        .map(|&(u, v)| if u < v { (u, v) } else { (v, u) })
        .collect();
    let max_edges = n_nodes * (n_nodes - 1) / 2;
    if edge_set.len() == max_edges {
        return n_nodes - 1;
    }

    // Node-splitting: vertex i → (i, i+n) with capacity 1
    // For source s=0: s_in=0, s_out=n
    // For target t: t_in=t, t_out=t+n
    // Edge (u,v) → (u_out, v_in) and (v_out, u_in) with capacity ∞
    let n2 = 2 * n_nodes;
    let mut min_cut = usize::MAX;
    for t in 1..n_nodes {
        let f = max_flow_node_split(edges, n_nodes, n2, 0, t);
        if f < min_cut {
            min_cut = f;
        }
    }
    if min_cut == usize::MAX {
        0
    } else {
        min_cut
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Max-flow (BFS / Edmonds-Karp)
// ─────────────────────────────────────────────────────────────────────────────

/// Edmonds-Karp max-flow for unit-capacity undirected edges.
fn max_flow_bfs(edges: &[(usize, usize)], n: usize, source: usize, sink: usize) -> usize {
    // Build residual capacity matrix
    let mut cap = vec![vec![0i64; n]; n];
    for &(u, v) in edges {
        if u < n && v < n && u != v {
            cap[u][v] += 1;
            cap[v][u] += 1;
        }
    }
    let mut flow = 0usize;
    loop {
        // BFS to find augmenting path
        let mut prev = vec![usize::MAX; n];
        let mut visited = vec![false; n];
        visited[source] = true;
        let mut queue = VecDeque::new();
        queue.push_back(source);
        while let Some(v) = queue.pop_front() {
            for w in 0..n {
                if !visited[w] && cap[v][w] > 0 {
                    visited[w] = true;
                    prev[w] = v;
                    if w == sink {
                        break;
                    }
                    queue.push_back(w);
                }
            }
        }
        if !visited[sink] {
            break;
        }
        // Find bottleneck
        let mut bot = i64::MAX;
        let mut cur = sink;
        while cur != source {
            let pr = prev[cur];
            bot = bot.min(cap[pr][cur]);
            cur = pr;
        }
        // Update residual
        cur = sink;
        while cur != source {
            let pr = prev[cur];
            cap[pr][cur] -= bot;
            cap[cur][pr] += bot;
            cur = pr;
        }
        flow += bot as usize;
    }
    flow
}

/// Max-flow on the node-split network for vertex connectivity.
fn max_flow_node_split(
    edges: &[(usize, usize)],
    n: usize,
    n2: usize,
    source: usize,
    sink: usize,
) -> usize {
    // Node-split network: 2n nodes
    // v_in = v, v_out = v + n
    // Internal edge (v_in, v_out): capacity 1 for all v ≠ source, ≠ sink
    // Graph edge (u,v): (u_out → v_in) and (v_out → u_in) with capacity ∞=n
    let mut cap = vec![vec![0i64; n2]; n2];
    for v in 0..n {
        if v != source && v != sink {
            cap[v][v + n] = 1;
        } else {
            // Source and sink get infinite internal capacity
            cap[v][v + n] = n as i64;
        }
    }
    let inf = n as i64;
    for &(u, v) in edges {
        if u < n && v < n && u != v {
            cap[u + n][v] += inf;
            cap[v + n][u] += inf;
        }
    }
    // Source is source_out (can emit freely), sink is sink_in
    let s = source + n;
    let t = sink;
    max_flow_generic(&mut cap, n2, s, t)
}

fn max_flow_generic(cap: &mut [Vec<i64>], n: usize, source: usize, sink: usize) -> usize {
    let mut flow = 0usize;
    loop {
        let mut prev = vec![usize::MAX; n];
        let mut visited = vec![false; n];
        visited[source] = true;
        let mut queue = VecDeque::new();
        queue.push_back(source);
        'bfs: while let Some(v) = queue.pop_front() {
            for w in 0..n {
                if !visited[w] && cap[v][w] > 0 {
                    visited[w] = true;
                    prev[w] = v;
                    if w == sink {
                        break 'bfs;
                    }
                    queue.push_back(w);
                }
            }
        }
        if !visited[sink] {
            break;
        }
        let mut bot = i64::MAX;
        let mut cur = sink;
        while cur != source {
            let pr = prev[cur];
            bot = bot.min(cap[pr][cur]);
            cur = pr;
        }
        cur = sink;
        while cur != source {
            let pr = prev[cur];
            cap[pr][cur] -= bot;
            cap[cur][pr] += bot;
            cur = pr;
        }
        flow += bot as usize;
    }
    flow
}

// ─────────────────────────────────────────────────────────────────────────────
// Graph connectivity helpers
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

fn is_connected(edges: &[(usize, usize)], n: usize) -> bool {
    if n == 0 {
        return true;
    }
    let adj = build_adj(edges, n);
    let mut visited = vec![false; n];
    let mut queue = VecDeque::new();
    queue.push_back(0usize);
    visited[0] = true;
    let mut count = 1usize;
    while let Some(v) = queue.pop_front() {
        for &w in &adj[v] {
            if !visited[w] {
                visited[w] = true;
                count += 1;
                queue.push_back(w);
            }
        }
    }
    count == n
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

    fn path3() -> Vec<(usize, usize)> {
        vec![(0, 1), (1, 2)]
    }

    #[test]
    fn test_all_terminal_reliability_complete_graph() {
        // Complete graph K3 — all edges present with p=1 → reliability = 1
        let r = all_terminal_reliability_factoring(&triangle(), 3, 1.0);
        assert!((r - 1.0).abs() < 1e-9, "r = {r}");
    }

    #[test]
    fn test_all_terminal_reliability_zero_probability() {
        let r = all_terminal_reliability_factoring(&triangle(), 3, 0.0);
        assert!(r < 0.01, "r = {r}");
    }

    #[test]
    fn test_two_terminal_reliability_path3() {
        // Path 0-1-2: only one path, each edge survives with p
        let r = two_terminal_reliability(&path3(), 3, 0, 2, 0.9);
        assert!((r - 0.81).abs() < 0.01, "r = {r}");
    }

    #[test]
    fn test_two_terminal_reliability_same_node() {
        assert_eq!(two_terminal_reliability(&triangle(), 3, 0, 0, 0.5), 1.0);
    }

    #[test]
    fn test_reliability_polynomial_path2() {
        // Single edge: connected iff edge survives → R(p) = p
        let poly = reliability_polynomial(&[(0, 1)], 2);
        assert!(!poly.is_empty());
        // Evaluate at p=0.5 → should be 0.5
        let val: f64 = poly
            .iter()
            .enumerate()
            .map(|(i, &c)| c * 0.5f64.powi(i as i32))
            .sum();
        assert!((val - 0.5).abs() < 1e-9, "R(0.5) = {val}");
    }

    #[test]
    fn test_k_edge_connectivity_triangle() {
        assert_eq!(k_edge_connectivity(&triangle(), 3), 2);
    }

    #[test]
    fn test_k_edge_connectivity_path() {
        assert_eq!(k_edge_connectivity(&path3(), 3), 1);
    }

    #[test]
    fn test_k_vertex_connectivity_triangle() {
        assert_eq!(k_vertex_connectivity(&triangle(), 3), 2);
    }

    #[test]
    fn test_k_vertex_connectivity_path() {
        assert_eq!(k_vertex_connectivity(&path3(), 3), 1);
    }

    #[test]
    fn test_min_cut_reliability_bound() {
        // Lambda=2 for triangle, bound = p^2
        let bound = min_cut_reliability_bound(&triangle(), 3, 0.9);
        assert!((bound - 0.81).abs() < 1e-9);
    }

    #[test]
    fn test_disconnected_graph_reliability_zero() {
        let r = all_terminal_reliability_factoring(&[(0, 1)], 3, 0.9);
        assert!(r < 1e-9, "disconnected graph has reliability 0");
    }
}
