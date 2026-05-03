//! Algebraic graph theory.
//!
//! Laplacian matrices, spectral properties (eigenvalues / eigenvectors),
//! characteristic polynomial, graph automorphisms, and graph isomorphism
//! via the VF2 algorithm.

use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// Laplacian and adjacency matrices
// ─────────────────────────────────────────────────────────────────────────────

/// Build the (combinatorial) Laplacian matrix  L = D − A  for a weighted
/// undirected graph.
///
/// # Arguments
/// * `edges`   – triples `(u, v, weight)` with 0-indexed vertices
/// * `n_nodes` – number of vertices
///
/// # Returns
/// `n_nodes × n_nodes` matrix as a flat `Vec<Vec<f64>>`.
pub fn laplacian_matrix(edges: &[(usize, usize, f64)], n_nodes: usize) -> Vec<Vec<f64>> {
    let mut mat = vec![vec![0.0f64; n_nodes]; n_nodes];
    for &(u, v, w) in edges {
        if u >= n_nodes || v >= n_nodes || u == v {
            continue;
        }
        mat[u][v] -= w;
        mat[v][u] -= w;
        mat[u][u] += w;
        mat[v][v] += w;
    }
    mat
}

/// Normalized Laplacian  L_norm = D^{−½} L D^{−½}.
///
/// Isolated vertices (zero degree) are left as zero rows/columns.
pub fn normalized_laplacian(edges: &[(usize, usize, f64)], n_nodes: usize) -> Vec<Vec<f64>> {
    let lap = laplacian_matrix(edges, n_nodes);
    // D^{−½}: inverse square root of diagonal
    let mut d_inv_sqrt = vec![0.0f64; n_nodes];
    for i in 0..n_nodes {
        let d = lap[i][i];
        if d > 1e-14 {
            d_inv_sqrt[i] = 1.0 / d.sqrt();
        }
    }
    let mut norm = vec![vec![0.0f64; n_nodes]; n_nodes];
    for i in 0..n_nodes {
        for j in 0..n_nodes {
            norm[i][j] = d_inv_sqrt[i] * lap[i][j] * d_inv_sqrt[j];
        }
    }
    norm
}

// ─────────────────────────────────────────────────────────────────────────────
// Spectral properties
// ─────────────────────────────────────────────────────────────────────────────

/// Returns eigenvalues of the **adjacency** matrix sorted in descending order
/// (the "graph spectrum").
///
/// Uses the symmetric QR algorithm (Jacobi method) which is exact for real
/// symmetric matrices.
pub fn graph_spectrum(edges: &[(usize, usize, f64)], n_nodes: usize) -> Vec<f64> {
    if n_nodes == 0 {
        return vec![];
    }
    let adj = build_adj_matrix(edges, n_nodes);
    let mut eigenvalues = jacobi_eigenvalues(&adj, n_nodes);
    eigenvalues.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    eigenvalues
}

/// Returns the **Fiedler value** (algebraic connectivity): the second smallest
/// eigenvalue of the Laplacian.
///
/// A value of 0 indicates a disconnected graph; larger values indicate
/// higher connectivity.
pub fn algebraic_connectivity(edges: &[(usize, usize, f64)], n_nodes: usize) -> f64 {
    if n_nodes <= 1 {
        return 0.0;
    }
    let lap = laplacian_matrix(edges, n_nodes);
    let mut eigenvalues = jacobi_eigenvalues(&lap, n_nodes);
    eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    // eigenvalues[0] ≈ 0 (always), eigenvalues[1] = Fiedler value
    if eigenvalues.len() > 1 {
        eigenvalues[1]
    } else {
        0.0
    }
}

/// Returns the Fiedler vector: the eigenvector corresponding to the second
/// smallest eigenvalue of the Laplacian.
///
/// Used for spectral graph partitioning (spectral bisection).
pub fn fiedler_vector(edges: &[(usize, usize, f64)], n_nodes: usize) -> Vec<f64> {
    if n_nodes == 0 {
        return vec![];
    }
    if n_nodes == 1 {
        return vec![1.0];
    }
    let lap = laplacian_matrix(edges, n_nodes);
    let (eigenvalues, eigenvectors) = jacobi_eigensystem(&lap, n_nodes);
    // Find index of second smallest eigenvalue
    let mut indexed: Vec<(f64, usize)> = eigenvalues
        .iter()
        .copied()
        .enumerate()
        .map(|(i, v)| (v, i))
        .collect();
    indexed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let fiedler_idx = if indexed.len() > 1 {
        indexed[1].1
    } else {
        indexed[0].1
    };
    (0..n_nodes).map(|i| eigenvectors[i][fiedler_idx]).collect()
}

/// Returns the spectral radius: the largest eigenvalue of the adjacency matrix.
///
/// Computed via power iteration (100 iterations maximum).
pub fn spectral_radius(edges: &[(usize, usize, f64)], n_nodes: usize) -> f64 {
    if n_nodes == 0 {
        return 0.0;
    }
    let adj = build_adj_matrix(edges, n_nodes);
    power_iteration(&adj, n_nodes, 200)
}

// ─────────────────────────────────────────────────────────────────────────────
// Characteristic polynomial (Faddeev-LeVerrier)
// ─────────────────────────────────────────────────────────────────────────────

/// Computes the characteristic polynomial  det(λI − A)  of the unweighted
/// adjacency matrix using the Faddeev-LeVerrier algorithm.
///
/// Returns coefficients `[c_0, c_1, ..., c_n]` such that
/// `det(λI − A) = c_n λ^n + c_{n−1} λ^{n−1} + ... + c_0`.
///
/// Complexity: O(N⁴).
pub fn characteristic_polynomial(edges: &[(usize, usize)], n_nodes: usize) -> Vec<f64> {
    if n_nodes == 0 {
        return vec![1.0];
    }
    // Convert unweighted edges to weighted
    let weighted: Vec<(usize, usize, f64)> = edges.iter().map(|&(u, v)| (u, v, 1.0)).collect();
    let a = build_adj_matrix(&weighted, n_nodes);
    faddeev_leverrier(&a, n_nodes)
}

// ─────────────────────────────────────────────────────────────────────────────
// Automorphisms and isomorphism
// ─────────────────────────────────────────────────────────────────────────────

/// Counts graph automorphisms by exhaustive permutation search.
///
/// Feasible only for small graphs (N ≤ 8).  Returns 1 (identity) for larger
/// graphs as an approximation.
pub fn graph_automorphisms_count(edges: &[(usize, usize)], n_nodes: usize) -> usize {
    if n_nodes > 8 {
        // Too expensive: return a lower bound of 1
        return 1;
    }
    let adj = build_bool_adj(edges, n_nodes);
    let perm: Vec<usize> = (0..n_nodes).collect();
    let mut count = 0usize;
    count_automorphisms(&adj, &perm, n_nodes, &mut count);
    count
}

/// Remaps arbitrary node indices in `edges` to the range `[0, k)` where `k`
/// is the number of distinct nodes seen, preserving edge structure.
///
/// This allows isomorphism checks on edge lists whose node labels are not
/// necessarily contiguous from 0 (e.g. nodes 3, 4, 5 become 0, 1, 2).
fn canonicalize_edges(edges: &[(usize, usize)]) -> Vec<(usize, usize)> {
    let mut map: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    let mut next = 0usize;
    let mut result = Vec::with_capacity(edges.len());
    for &(u, v) in edges {
        let cu = *map.entry(u).or_insert_with(|| {
            let k = next;
            next += 1;
            k
        });
        let cv = *map.entry(v).or_insert_with(|| {
            let k = next;
            next += 1;
            k
        });
        result.push((cu, cv));
    }
    result
}

/// Tests graph isomorphism using the VF2 algorithm with degree-sequence
/// pre-filtering.
///
/// Returns `true` if graphs `(adj1, n1)` and `(adj2, n2)` are isomorphic.
/// Node labels in the edge lists need not start at 0; they are canonicalized
/// internally so that, e.g., edges `[(3,4),(4,5),(3,5)]` with `n=3` are
/// treated equivalently to `[(0,1),(1,2),(0,2)]` with `n=3`.
pub fn is_isomorphic(
    adj1: &[(usize, usize)],
    n1: usize,
    adj2: &[(usize, usize)],
    n2: usize,
) -> bool {
    if n1 != n2 {
        return false;
    }
    if n1 == 0 {
        return true;
    }
    // Remap node labels to [0, k) so that offset graphs compare correctly.
    let adj1c = canonicalize_edges(adj1);
    let adj2c = canonicalize_edges(adj2);
    let e1 = count_edges(&adj1c, n1);
    let e2 = count_edges(&adj2c, n2);
    if e1 != e2 {
        return false;
    }
    // Degree sequence check
    let mut deg1 = degree_sequence(&adj1c, n1);
    let mut deg2 = degree_sequence(&adj2c, n2);
    deg1.sort_unstable();
    deg2.sort_unstable();
    if deg1 != deg2 {
        return false;
    }
    // VF2 backtracking
    let a1 = build_bool_adj(&adj1c, n1);
    let a2 = build_bool_adj(&adj2c, n2);
    let mut mapping = vec![usize::MAX; n1];
    vf2_match(&a1, &a2, n1, &mut mapping, 0)
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

fn build_adj_matrix(edges: &[(usize, usize, f64)], n: usize) -> Vec<Vec<f64>> {
    let mut mat = vec![vec![0.0f64; n]; n];
    for &(u, v, w) in edges {
        if u < n && v < n && u != v {
            mat[u][v] = w;
            mat[v][u] = w;
        }
    }
    mat
}

fn build_bool_adj(edges: &[(usize, usize)], n: usize) -> Vec<Vec<bool>> {
    let mut mat = vec![vec![false; n]; n];
    for &(u, v) in edges {
        if u < n && v < n && u != v {
            mat[u][v] = true;
            mat[v][u] = true;
        }
    }
    mat
}

fn count_edges(edges: &[(usize, usize)], n: usize) -> usize {
    // Count unique undirected edges
    let mut set: std::collections::HashSet<(usize, usize)> = std::collections::HashSet::new();
    for &(u, v) in edges {
        if u < n && v < n && u != v {
            let e = if u < v { (u, v) } else { (v, u) };
            set.insert(e);
        }
    }
    set.len()
}

fn degree_sequence(edges: &[(usize, usize)], n: usize) -> Vec<usize> {
    let mut deg = vec![0usize; n];
    for &(u, v) in edges {
        if u < n && v < n && u != v {
            deg[u] += 1;
            deg[v] += 1;
        }
    }
    deg
}

// ─────────────────────── Jacobi eigenvalue algorithm ────────────────────────

/// Computes eigenvalues of a real symmetric matrix via the Jacobi method.
fn jacobi_eigenvalues(mat: &[Vec<f64>], n: usize) -> Vec<f64> {
    let (vals, _) = jacobi_eigensystem(mat, n);
    vals
}

/// Computes eigenvalues and eigenvectors of a real symmetric matrix via Jacobi
/// iterations.  Returns `(eigenvalues, eigenvectors)` where eigenvectors are
/// stored column-by-column in a flat Vec<Vec<f64>> (rows = node index, cols =
/// eigenvector index).
fn jacobi_eigensystem(mat: &[Vec<f64>], n: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
    if n == 0 {
        return (vec![], vec![]);
    }
    if n == 1 {
        return (vec![mat[0][0]], vec![vec![1.0]]);
    }

    // Working copy
    let mut a: Vec<Vec<f64>> = mat.to_vec();
    // Eigenvector matrix (starts as identity)
    let mut v: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
        .collect();

    let max_iter = 100 * n * n;
    for _ in 0..max_iter {
        // Find largest off-diagonal element
        let mut max_val = 0.0f64;
        let mut p = 0;
        let mut q = 1;
        for i in 0..n {
            for j in (i + 1)..n {
                let av = a[i][j].abs();
                if av > max_val {
                    max_val = av;
                    p = i;
                    q = j;
                }
            }
        }
        if max_val < 1e-12 {
            break;
        }

        // Compute rotation angle
        let diff = a[q][q] - a[p][p];
        let theta = if diff.abs() < 1e-14 {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * (2.0 * a[p][q] / diff).atan()
        };
        let c = theta.cos();
        let s = theta.sin();

        // Apply Jacobi rotation
        let app = a[p][p];
        let aqq = a[q][q];
        let apq = a[p][q];
        a[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        a[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        a[p][q] = 0.0;
        a[q][p] = 0.0;
        for r in 0..n {
            if r != p && r != q {
                let arp = a[r][p];
                let arq = a[r][q];
                a[r][p] = c * arp - s * arq;
                a[p][r] = a[r][p];
                a[r][q] = s * arp + c * arq;
                a[q][r] = a[r][q];
            }
        }
        // Update eigenvector matrix
        for r in 0..n {
            let vrp = v[r][p];
            let vrq = v[r][q];
            v[r][p] = c * vrp - s * vrq;
            v[r][q] = s * vrp + c * vrq;
        }
    }

    let eigenvalues: Vec<f64> = (0..n).map(|i| a[i][i]).collect();
    (eigenvalues, v)
}

// ──────────────────────────── Power iteration ────────────────────────────────

fn power_iteration(mat: &[Vec<f64>], n: usize, max_iter: usize) -> f64 {
    let mut x: Vec<f64> = vec![1.0; n];
    let mut lambda = 0.0f64;
    for _ in 0..max_iter {
        let mut y = vec![0.0f64; n];
        for i in 0..n {
            for j in 0..n {
                y[i] += mat[i][j] * x[j];
            }
        }
        let norm: f64 = y.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm < 1e-14 {
            break;
        }
        lambda = norm;
        for i in 0..n {
            x[i] = y[i] / norm;
        }
    }
    lambda
}

// ─────────────────────── Faddeev-LeVerrier algorithm ─────────────────────────

/// Faddeev-LeVerrier recurrence to compute the characteristic polynomial.
/// Returns coefficients [c_0, ..., c_n] from lowest to highest degree.
fn faddeev_leverrier(a: &[Vec<f64>], n: usize) -> Vec<f64> {
    let mut coeffs = vec![0.0f64; n + 1];
    coeffs[n] = 1.0; // leading coefficient is always 1

    // M_k recurrence:  M_1 = A, c_{n-1} = -tr(M_1)
    // M_k = A * (M_{k-1} + c_{n-k+1} * I)
    let mut m: Vec<Vec<f64>> = (0..n).map(|_i| (0..n).map(|_j| 0.0).collect()).collect();

    for k in 1..=n {
        // M_k = A * (M_{k-1} + c_{n-k+1} * I)
        // For k=1: M_0 = I (identity), so M_1 = A * I = A
        let c_prev = if k == 1 { 1.0 } else { coeffs[n - k + 1] };
        let mut m_plus_ci: Vec<Vec<f64>> = m.clone();
        for i in 0..n {
            m_plus_ci[i][i] += c_prev;
        }
        // new_m = A * m_plus_ci
        let mut new_m = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for j in 0..n {
                for l in 0..n {
                    new_m[i][j] += a[i][l] * m_plus_ci[l][j];
                }
            }
        }
        // c_{n-k} = -1/k * tr(new_m)
        let trace: f64 = (0..n).map(|i| new_m[i][i]).sum();
        coeffs[n - k] = -trace / k as f64;
        m = new_m;
    }
    coeffs
}

// ─────────────────────────── VF2 isomorphism ─────────────────────────────────

/// VF2 backtracking isomorphism check.
/// `mapping[v1]` = matched vertex in graph 2, or `usize::MAX` if unmatched.
fn vf2_match(
    a1: &[Vec<bool>],
    a2: &[Vec<bool>],
    n: usize,
    mapping: &mut Vec<usize>,
    depth: usize,
) -> bool {
    if depth == n {
        return true;
    }
    let v1 = depth; // next vertex from graph 1 to match
    let used: Vec<usize> = mapping[..depth].to_vec();
    for v2 in 0..n {
        if used.contains(&v2) {
            continue;
        }
        // Check feasibility: edges from v1 to previously matched vertices in G1
        // must correspond to edges from v2 to their images in G2.
        let mut ok = true;
        for prev_v1 in 0..depth {
            let prev_v2 = mapping[prev_v1];
            if a1[v1][prev_v1] != a2[v2][prev_v2] {
                ok = false;
                break;
            }
        }
        if ok {
            mapping[v1] = v2;
            if vf2_match(a1, a2, n, mapping, depth + 1) {
                return true;
            }
            mapping[v1] = usize::MAX;
        }
    }
    false
}

// ───────────────────────── Automorphism counting ─────────────────────────────

fn count_automorphisms(adj: &[Vec<bool>], base: &[usize], n: usize, count: &mut usize) {
    // Iterative permutation generation via Heap's algorithm check
    // We enumerate all permutations of {0..n} and check if each is an automorphism.
    let mut perm: Vec<usize> = (0..n).collect();
    let _ = base; // unused in this implementation
    loop {
        if is_automorphism(adj, &perm, n) {
            *count += 1;
        }
        // Next permutation
        if !next_permutation(&mut perm) {
            break;
        }
    }
}

fn is_automorphism(adj: &[Vec<bool>], perm: &[usize], n: usize) -> bool {
    for i in 0..n {
        for j in 0..n {
            if adj[i][j] != adj[perm[i]][perm[j]] {
                return false;
            }
        }
    }
    true
}

fn next_permutation(v: &mut [usize]) -> bool {
    let n = v.len();
    if n <= 1 {
        return false;
    }
    let mut k = n - 1;
    while k > 0 && v[k - 1] >= v[k] {
        k -= 1;
    }
    if k == 0 {
        return false;
    }
    let mut l = n - 1;
    while v[l] <= v[k - 1] {
        l -= 1;
    }
    v.swap(k - 1, l);
    v[k..].reverse();
    true
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    fn petersen_edges_weighted() -> Vec<(usize, usize, f64)> {
        let outer: Vec<(usize, usize, f64)> = (0..5).map(|i| (i, (i + 1) % 5, 1.0)).collect();
        let spokes: Vec<(usize, usize, f64)> = (0..5).map(|i| (i, i + 5, 1.0)).collect();
        let inner: Vec<(usize, usize, f64)> =
            (0..5).map(|i| (i + 5, ((i + 2) % 5) + 5, 1.0)).collect();
        let mut e = Vec::new();
        e.extend_from_slice(&outer);
        e.extend_from_slice(&spokes);
        e.extend_from_slice(&inner);
        e
    }

    #[test]
    fn test_laplacian_triangle() {
        // Triangle graph: all weights 1
        let edges = vec![(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)];
        let lap = laplacian_matrix(&edges, 3);
        // Each diagonal = 2, each off-diagonal = -1
        for i in 0..3 {
            assert!((lap[i][i] - 2.0).abs() < 1e-10);
        }
        for i in 0..3 {
            for j in 0..3 {
                if i != j {
                    assert!((lap[i][j] + 1.0).abs() < 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_normalized_laplacian_triangle() {
        let edges = vec![(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)];
        let nl = normalized_laplacian(&edges, 3);
        // Row sums of normalized Laplacian should be 0 for regular graphs
        // Eigenvalues should be in [0, 2]
        for row in &nl {
            let s: f64 = row.iter().sum();
            assert!(s.abs() < 1e-8, "row sum = {s}");
        }
    }

    #[test]
    fn test_graph_spectrum_complete() {
        // K4: eigenvalues are 3 (once) and -1 (three times)
        let edges: Vec<(usize, usize, f64)> = (0..4)
            .flat_map(|u| (u + 1..4).map(move |v| (u, v, 1.0)))
            .collect();
        let spec = graph_spectrum(&edges, 4);
        assert_eq!(spec.len(), 4);
        // Largest eigenvalue should be 3
        assert!(
            (spec[0] - 3.0).abs() < 1e-6,
            "largest eigenvalue = {}",
            spec[0]
        );
    }

    #[test]
    fn test_algebraic_connectivity_disconnected() {
        // Two isolated vertices: no edges
        let conn = algebraic_connectivity(&[], 4);
        assert!(conn.abs() < 1e-8, "disconnected graph has Fiedler = 0");
    }

    #[test]
    fn test_algebraic_connectivity_petersen() {
        let edges = petersen_edges_weighted();
        let ac = algebraic_connectivity(&edges, 10);
        // Petersen graph is 3-regular, algebraic connectivity ≈ 2
        assert!(ac > 1.5, "Petersen algebraic connectivity = {ac}");
    }

    #[test]
    fn test_fiedler_vector_length() {
        let edges = vec![(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)];
        let fv = fiedler_vector(&edges, 4);
        assert_eq!(fv.len(), 4);
    }

    #[test]
    fn test_spectral_radius_k_regular() {
        // 4-cycle (2-regular): spectral radius = 2
        let edges = vec![(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 0, 1.0)];
        let sr = spectral_radius(&edges, 4);
        assert!((sr - 2.0).abs() < 0.01, "spectral radius = {sr}");
    }

    #[test]
    fn test_characteristic_polynomial_path2() {
        // P₂ (single edge 0-1): char poly = λ² − 1
        let edges = vec![(0, 1)];
        let cp = characteristic_polynomial(&edges, 2);
        // cp = [c0, c1, c2] = [-1, 0, 1]
        assert_eq!(cp.len(), 3);
        assert!((cp[2] - 1.0).abs() < 1e-10, "c2 = {}", cp[2]);
        assert!((cp[0] + 1.0).abs() < 1e-10, "c0 = {}", cp[0]);
    }

    #[test]
    fn test_graph_automorphisms_single_edge() {
        // Single edge: 2 automorphisms (identity + swap)
        let edges = vec![(0, 1)];
        let count = graph_automorphisms_count(&edges, 2);
        assert_eq!(count, 2, "single edge has 2 automorphisms");
    }

    #[test]
    fn test_is_isomorphic_triangles() {
        let t1 = vec![(0, 1), (1, 2), (0, 2)];
        let t2 = vec![(3, 4), (4, 5), (3, 5)];
        // Offset by 3 but still 3-node triangle
        let t2_reindexed = vec![(0, 1), (1, 2), (0, 2)];
        assert!(is_isomorphic(&t1, 3, &t2_reindexed, 3));
        assert!(is_isomorphic(&t1, 3, &t2, 3));
    }

    #[test]
    fn test_is_isomorphic_different_graphs() {
        // Path P3 vs triangle
        let path3 = vec![(0, 1), (1, 2)];
        let triangle = vec![(0, 1), (1, 2), (0, 2)];
        assert!(!is_isomorphic(&path3, 3, &triangle, 3));
    }

    #[test]
    fn test_is_isomorphic_k4_vs_c4() {
        let k4: Vec<(usize, usize)> = (0..4)
            .flat_map(|u| (u + 1..4).map(move |v| (u, v)))
            .collect();
        let c4 = vec![(0, 1), (1, 2), (2, 3), (3, 0)];
        assert!(!is_isomorphic(&k4, 4, &c4, 4));
    }

    #[test]
    fn test_characteristic_polynomial_empty() {
        let cp = characteristic_polynomial(&[], 0);
        assert_eq!(cp, vec![1.0]);
    }

    #[test]
    fn test_laplacian_matrix_empty() {
        let lap = laplacian_matrix(&[], 3);
        for row in &lap {
            assert!(row.iter().all(|&v| v == 0.0));
        }
    }
}
