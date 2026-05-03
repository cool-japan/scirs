//! Higher-order network analysis.
//!
//! Implements:
//! * **Motif tensors** – third-order adjacency tensors capturing 3-node motifs.
//! * **Topological features** – Betti numbers from a simplicial complex bridge.
//! * **Cellular sheaves** – basic sheaf theory on graphs (restriction maps,
//!   coboundary operator, Hodge Laplacian).
//!
//! # References
//! - Benson et al., "Three hypergraph eigenvector centralities", SIAM J. Math.
//!   Data Sci. 2019.
//! - Hansen & Ghrist, "Toward a spectral theory of cellular sheaves", J. Applied
//!   and Computational Topology, 2019.
//! - Battiston et al., "Networks beyond pairwise interactions", Physics Reports
//!   2020.

use super::simplicial::SimplicialComplex;
use crate::base::Graph;
use crate::error::{GraphError, Result};
use scirs2_core::ndarray::{Array2, Array3};
use std::collections::HashMap;

// ============================================================================
// Motif tensors
// ============================================================================

/// A third-order motif tensor T[i,j,k] encoding 3-node interaction patterns.
///
/// For an undirected graph, entry T[i,j,k] = 1 iff the induced subgraph on
/// {i,j,k} forms a **triangle** (all three edges present).
///
/// For directed/weighted variants, use [`directed_motif_tensor`].
pub struct MotifTensor {
    /// Shape: (n, n, n) where n = number of nodes
    pub data: Array3<f64>,
    /// Number of nodes
    pub n: usize,
}

impl MotifTensor {
    /// Build the **triangle motif tensor** from a graph adjacency matrix.
    ///
    /// T[i,j,k] = A[i,j] * A[j,k] * A[i,k]  (all three edges present).
    ///
    /// For a 0/1 adjacency matrix this is exactly 1 for triangles and 0
    /// otherwise.  For weighted graphs the entry equals the product of the
    /// three edge weights.
    ///
    /// # Arguments
    /// * `adj` – symmetric adjacency matrix (shape n × n)
    pub fn from_adjacency(adj: &Array2<f64>) -> Self {
        let n = adj.nrows();
        let mut data = Array3::<f64>::zeros((n, n, n));
        for i in 0..n {
            for j in 0..n {
                if adj[[i, j]] == 0.0 {
                    continue;
                }
                for k in 0..n {
                    if i == j || j == k || i == k {
                        continue;
                    }
                    let val = adj[[i, j]] * adj[[j, k]] * adj[[i, k]];
                    if val != 0.0 {
                        data[[i, j, k]] = val;
                    }
                }
            }
        }
        MotifTensor { data, n }
    }

    /// Count the total number of directed triangles encoded in the tensor.
    ///
    /// Each undirected triangle {i,j,k} contributes 6 directed entries
    /// (all permutations), so divide by 6 to get the undirected count.
    pub fn triangle_count(&self) -> f64 {
        self.data.iter().sum::<f64>() / 6.0
    }

    /// Compute the **tensor-vector product** T ×_1 x ×_2 x:
    ///
    /// (T x^2)[i] = Σ_{j,k} T[i,j,k] x[j] x[k]
    ///
    /// This is the "mode-1 multilinear product" used in tensor eigenvector
    /// centrality.
    pub fn tensor_vec_product(&self, x: &[f64]) -> Vec<f64> {
        let n = self.n;
        let mut result = vec![0.0f64; n];
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    result[i] += self.data[[i, j, k]] * x[j] * x[k];
                }
            }
        }
        result
    }

    /// Compute the **Z-eigenvector centrality** of the motif tensor using the
    /// shifted power method.
    ///
    /// Returns `(eigenvalue, eigenvector)` where the eigenvector satisfies
    /// T x^2 = λ x.  All entries of the returned vector are non-negative.
    ///
    /// # Errors
    /// Returns `GraphError::InvalidGraph` if n = 0.
    pub fn z_eigenvector_centrality(&self) -> Result<(f64, Vec<f64>)> {
        let n = self.n;
        if n == 0 {
            return Err(GraphError::InvalidGraph("empty motif tensor".to_string()));
        }

        // Initialise uniformly
        let mut x = vec![1.0 / (n as f64).sqrt(); n];
        let max_iter = 2000;
        let tol = 1e-9;

        for _ in 0..max_iter {
            let mut tx2 = self.tensor_vec_product(&x);
            // Compute Rayleigh quotient λ = x^T (T x^2)
            let lambda: f64 = tx2.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
            // Project to non-negative orthant
            for v in &mut tx2 {
                if *v < 0.0 {
                    *v = 0.0;
                }
            }
            let norm: f64 = tx2.iter().map(|v| v * v).sum::<f64>().sqrt();
            if norm < 1e-15 {
                return Ok((0.0, vec![0.0; n]));
            }
            let x_new: Vec<f64> = tx2.iter().map(|v| v / norm).collect();
            let diff: f64 = x_new.iter().zip(x.iter()).map(|(a, b)| (a - b).abs()).sum();
            x = x_new;
            if diff < tol {
                return Ok((lambda, x));
            }
        }
        let lambda: f64 = {
            let tx2 = self.tensor_vec_product(&x);
            tx2.iter().zip(x.iter()).map(|(a, b)| a * b).sum()
        };
        Ok((lambda, x))
    }
}

/// Build a **directed motif tensor** that distinguishes different 3-node motif
/// types (feed-forward, cycle, etc.) in a directed graph.
///
/// The function encodes motif type as integer codes in the returned `Array3<u8>`:
/// * 0 – no directed edges
/// * 1 – cycle (i→j, j→k, k→i)
/// * 2 – feed-forward chain (i→j, j→k)
/// * 3 – other configurations
///
/// # Arguments
/// * `adj_directed` – (possibly asymmetric) adjacency matrix (n × n)
pub fn directed_motif_tensor(adj: &Array2<f64>) -> Array3<u8> {
    let n = adj.nrows();
    let mut tensor = Array3::<u8>::zeros((n, n, n));
    for i in 0..n {
        for j in 0..n {
            if i == j || adj[[i, j]] == 0.0 {
                continue;
            }
            for k in 0..n {
                if k == i || k == j {
                    continue;
                }
                if adj[[j, k]] > 0.0 {
                    if adj[[k, i]] > 0.0 {
                        // Directed 3-cycle: i→j→k→i
                        tensor[[i, j, k]] = 1;
                    } else {
                        // Feed-forward: i→j→k (no back edge k→i)
                        tensor[[i, j, k]] = 2;
                    }
                } else if adj[[i, k]] > 0.0 || adj[[k, j]] > 0.0 {
                    tensor[[i, j, k]] = 3;
                }
            }
        }
    }
    tensor
}

// ============================================================================
// Topological features bridge
// ============================================================================

/// Topological feature vector derived from a [`SimplicialComplex`].
#[derive(Debug, Clone)]
pub struct TopologicalFeatures {
    /// Betti numbers β_0, β_1, …
    pub betti_numbers: Vec<usize>,
    /// Euler characteristic χ
    pub euler_characteristic: i64,
    /// Number of simplices per dimension
    pub simplex_counts: Vec<usize>,
    /// Maximum dimension
    pub max_dim: usize,
}

impl TopologicalFeatures {
    /// Extract topological features from a [`SimplicialComplex`].
    pub fn from_complex(sc: &SimplicialComplex) -> Self {
        let max_dim = sc.max_dim().unwrap_or(0);
        let betti_numbers = sc.betti_numbers();
        let euler_characteristic = sc.euler_characteristic();
        let simplex_counts = (0..=max_dim).map(|d| sc.num_simplices(d)).collect();
        TopologicalFeatures {
            betti_numbers,
            euler_characteristic,
            simplex_counts,
            max_dim,
        }
    }

    /// Compute a simple **topological signature** vector of fixed length `len`.
    ///
    /// The signature concatenates (betti_numbers padded/truncated to `len/2`)
    /// and (simplex_counts padded/truncated to `len/2`), normalised to unit
    /// L2 norm.
    pub fn signature(&self, len: usize) -> Vec<f64> {
        let half = len / 2;
        let mut sig = Vec::with_capacity(len);

        // First half: Betti numbers
        for i in 0..half {
            sig.push(self.betti_numbers.get(i).copied().unwrap_or(0) as f64);
        }
        // Second half: simplex counts
        for i in 0..(len - half) {
            sig.push(self.simplex_counts.get(i).copied().unwrap_or(0) as f64);
        }

        // Normalise
        let norm: f64 = sig.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 0.0 {
            for v in &mut sig {
                *v /= norm;
            }
        }
        sig
    }
}

// ============================================================================
// Cellular sheaves (basic implementation)
// ============================================================================

/// A **cellular sheaf** on an undirected graph.
///
/// A sheaf assigns:
/// * A vector space `ℝ^{d_v}` to each vertex `v` (vertex stalk).
/// * A vector space `ℝ^{d_e}` to each edge `e` (edge stalk).
/// * **Restriction maps** `F_{v ◁ e} : ℝ^{d_e} → ℝ^{d_v}` for each
///   vertex-edge incidence.
///
/// The **coboundary operator** δ : C^0 → C^1 and the resulting **Hodge-0
/// sheaf Laplacian** L = δ^T δ capture the sheaf's cohomology.
///
/// Reference: Hansen & Ghrist (2019).
#[derive(Debug, Clone)]
pub struct CellularSheaf {
    /// Number of nodes
    pub n_nodes: usize,
    /// Number of edges (oriented pairs (u,v) with u < v)
    pub n_edges: usize,
    /// Stalks at nodes: node_stalks[v] = dimension of the stalk at v
    pub node_stalks: Vec<usize>,
    /// Stalks at edges: edge_stalks[e] = dimension of the stalk at edge e
    pub edge_stalks: Vec<usize>,
    /// Oriented edge list: edges[e] = (tail, head) with tail < head
    pub edges: Vec<(usize, usize)>,
    /// Restriction maps: maps[(v, e)] = the matrix F_{v◁e} : ℝ^{d_e} → ℝ^{d_v}
    /// stored row-major as Vec<Vec<f64>> of shape (d_v, d_e)
    pub restriction_maps: HashMap<(usize, usize), Vec<Vec<f64>>>,
}

impl CellularSheaf {
    /// Create a trivial sheaf on a graph where all stalks have dimension 1.
    ///
    /// The restriction maps are initialised to the identity (or ±1 depending
    /// on orientation).
    ///
    /// # Arguments
    /// * `n_nodes` – number of nodes
    /// * `edges`   – oriented edge list `(tail, head)`, both < n_nodes
    pub fn trivial(n_nodes: usize, edges: Vec<(usize, usize)>) -> Result<Self> {
        for &(u, v) in &edges {
            if u >= n_nodes || v >= n_nodes {
                return Err(GraphError::InvalidGraph(format!(
                    "edge ({u},{v}) references node ≥ n_nodes={n_nodes}"
                )));
            }
        }
        let n_edges = edges.len();
        let node_stalks = vec![1; n_nodes];
        let edge_stalks = vec![1; n_edges];
        let mut restriction_maps: HashMap<(usize, usize), Vec<Vec<f64>>> = HashMap::new();
        for (eid, &(u, v)) in edges.iter().enumerate() {
            // F_{v ◁ e} = [[1]] for all incidences; the sign comes from the
            // coboundary formula δ(x)_e = F_{head◁e} x_{head} - F_{tail◁e} x_{tail}.
            restriction_maps.insert((u, eid), vec![vec![1.0]]);
            restriction_maps.insert((v, eid), vec![vec![1.0]]);
        }
        Ok(CellularSheaf {
            n_nodes,
            n_edges,
            node_stalks,
            edge_stalks,
            edges,
            restriction_maps,
        })
    }

    /// Create a sheaf with user-specified stalk dimensions and restriction maps.
    ///
    /// `node_dim` and `edge_dim` give the stalk dimensions.
    /// `maps` is a map from `(node_id, edge_id)` to a matrix of shape
    /// `(node_dim, edge_dim)`.
    pub fn new(
        n_nodes: usize,
        node_dim: usize,
        edges: Vec<(usize, usize)>,
        edge_dim: usize,
        maps: HashMap<(usize, usize), Vec<Vec<f64>>>,
    ) -> Result<Self> {
        let n_edges = edges.len();
        for &(u, v) in &edges {
            if u >= n_nodes || v >= n_nodes {
                return Err(GraphError::InvalidGraph(format!(
                    "edge ({u},{v}) out of range"
                )));
            }
        }
        Ok(CellularSheaf {
            n_nodes,
            n_edges,
            node_stalks: vec![node_dim; n_nodes],
            edge_stalks: vec![edge_dim; n_edges],
            edges,
            restriction_maps: maps,
        })
    }

    /// Set the restriction map for vertex `v` and edge `e`.
    ///
    /// The matrix `map` should have shape `(stalk_dim[v], edge_stalk[e])`.
    pub fn set_restriction(&mut self, v: usize, e: usize, map: Vec<Vec<f64>>) -> Result<()> {
        if v >= self.n_nodes {
            return Err(GraphError::InvalidGraph(format!(
                "vertex {v} >= n_nodes {}",
                self.n_nodes
            )));
        }
        if e >= self.n_edges {
            return Err(GraphError::InvalidGraph(format!(
                "edge {e} >= n_edges {}",
                self.n_edges
            )));
        }
        self.restriction_maps.insert((v, e), map);
        Ok(())
    }

    /// Compute the **coboundary matrix** δ : C^0 → C^1.
    ///
    /// The full coboundary operator has shape
    /// `(sum_e d_e) × (sum_v d_v)`.
    ///
    /// For a section x = (x_v)_{v}, the coboundary δ(x) at edge e = (u,v) is:
    /// `δ(x)_e = F_{v◁e} x_v − F_{u◁e} x_u`
    ///
    /// Returns the matrix as `Array2<f64>`.
    pub fn coboundary_operator(&self) -> Array2<f64> {
        let total_node = self.node_stalks.iter().sum::<usize>();
        let total_edge = self.edge_stalks.iter().sum::<usize>();

        if total_node == 0 || total_edge == 0 {
            return Array2::zeros((total_edge.max(1), total_node.max(1)));
        }

        // Compute offset arrays
        let mut node_offset = vec![0usize; self.n_nodes + 1];
        for i in 0..self.n_nodes {
            node_offset[i + 1] = node_offset[i] + self.node_stalks[i];
        }
        let mut edge_offset = vec![0usize; self.n_edges + 1];
        for i in 0..self.n_edges {
            edge_offset[i + 1] = edge_offset[i] + self.edge_stalks[i];
        }

        let mut delta = Array2::<f64>::zeros((total_edge, total_node));

        for (eid, &(u, v)) in self.edges.iter().enumerate() {
            let e_rows = self.edge_stalks[eid];
            let e_off = edge_offset[eid];

            // δ(x)_e = F_{head◁e} x_{head} - F_{tail◁e} x_{tail}
            // (standard orientation: positive end = head = v, negative end = tail = u)
            if let Some(fu) = self.restriction_maps.get(&(u, eid)) {
                let u_off = node_offset[u];
                let u_cols = self.node_stalks[u];
                // Tail contributes negatively
                for r in 0..e_rows.min(fu.len()) {
                    for c in 0..u_cols.min(fu[r].len()) {
                        delta[[e_off + r, u_off + c]] -= fu[r][c];
                    }
                }
            }
            // Head contributes positively
            if let Some(fv) = self.restriction_maps.get(&(v, eid)) {
                let v_off = node_offset[v];
                let v_cols = self.node_stalks[v];
                for r in 0..e_rows.min(fv.len()) {
                    for c in 0..v_cols.min(fv[r].len()) {
                        delta[[e_off + r, v_off + c]] += fv[r][c];
                    }
                }
            }
        }
        delta
    }

    /// Compute the **Hodge-0 sheaf Laplacian** L_0 = δ^T δ.
    ///
    /// Shape: `(sum_v d_v) × (sum_v d_v)`.
    pub fn hodge_laplacian_0(&self) -> Array2<f64> {
        let delta = self.coboundary_operator();
        let rows = delta.nrows();
        let cols = delta.ncols();
        // L = delta^T * delta
        let mut l = Array2::<f64>::zeros((cols, cols));
        for i in 0..cols {
            for j in 0..cols {
                let mut val = 0.0f64;
                for r in 0..rows {
                    val += delta[[r, i]] * delta[[r, j]];
                }
                l[[i, j]] = val;
            }
        }
        l
    }

    /// Compute the dimension of the **0th sheaf cohomology** H^0(G; F).
    ///
    /// dim H^0 = dim ker(δ) = total_node_dim − rank(δ).
    ///
    /// A dimension of 1 corresponds to a "consistent" global section (like a
    /// constant function on a constant sheaf).
    pub fn cohomology_h0(&self) -> usize {
        let delta = self.coboundary_operator();
        let total_node: usize = self.node_stalks.iter().sum();
        let rank = matrix_rank_f64(&delta);
        total_node.saturating_sub(rank)
    }

    /// Compute the dimension of the **1st sheaf cohomology** H^1(G; F).
    ///
    /// dim H^1 = dim ker(δ^T) restricted to the image complement
    ///          = total_edge_dim − rank(δ).
    pub fn cohomology_h1(&self) -> usize {
        let delta = self.coboundary_operator();
        let total_edge: usize = self.edge_stalks.iter().sum();
        let rank = matrix_rank_f64(&delta);
        total_edge.saturating_sub(rank)
    }
}

/// Build a trivial sheaf from an existing graph (uses usize-indexed nodes).
///
/// This is a convenience constructor for the common case of constant stalks.
pub fn trivial_sheaf_from_graph(n: usize, edge_list: &[(usize, usize)]) -> Result<CellularSheaf> {
    let mut oriented: Vec<(usize, usize)> = edge_list
        .iter()
        .map(|&(u, v)| (u.min(v), u.max(v)))
        .collect();
    oriented.sort_unstable();
    oriented.dedup();
    CellularSheaf::trivial(n, oriented)
}

// ============================================================================
// Helpers
// ============================================================================

/// Gaussian elimination rank of a real matrix (with floating-point pivoting).
fn matrix_rank_f64(mat: &Array2<f64>) -> usize {
    let (rows, cols) = (mat.nrows(), mat.ncols());
    if rows == 0 || cols == 0 {
        return 0;
    }
    let mut m: Vec<Vec<f64>> = (0..rows)
        .map(|i| (0..cols).map(|j| mat[[i, j]]).collect())
        .collect();

    let tol = 1e-10;
    let mut rank = 0usize;
    let mut pivot_row = 0usize;
    for col in 0..cols {
        // Find max pivot
        let (best, best_val) = (pivot_row..rows).fold((pivot_row, 0.0f64), |(bi, bv), r| {
            let v = m[r][col].abs();
            if v > bv {
                (r, v)
            } else {
                (bi, bv)
            }
        });
        if best_val < tol {
            continue;
        }
        m.swap(pivot_row, best);
        let piv = m[pivot_row][col];
        for c in 0..cols {
            m[pivot_row][c] /= piv;
        }
        for r in 0..rows {
            if r == pivot_row {
                continue;
            }
            let factor = m[r][col];
            if factor.abs() < tol {
                continue;
            }
            for c in 0..cols {
                let sub = factor * m[pivot_row][c];
                m[r][c] -= sub;
            }
        }
        pivot_row += 1;
        rank += 1;
    }
    rank
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::array;

    // ----- MotifTensor -------------------------------------------------------

    #[test]
    fn test_motif_tensor_triangle_count() {
        // Complete graph K3: 1 triangle
        let adj = array![[0.0_f64, 1., 1.], [1., 0., 1.], [1., 1., 0.]];
        let mt = MotifTensor::from_adjacency(&adj);
        assert_relative_eq!(mt.triangle_count(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_motif_tensor_no_triangle() {
        // Path P3: no triangle
        let adj = array![[0.0_f64, 1., 0.], [1., 0., 1.], [0., 1., 0.]];
        let mt = MotifTensor::from_adjacency(&adj);
        assert_relative_eq!(mt.triangle_count(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_z_eigenvector_centrality_k3() {
        let adj = array![[0.0_f64, 1., 1.], [1., 0., 1.], [1., 1., 0.]];
        let mt = MotifTensor::from_adjacency(&adj);
        let (lambda, x) = mt.z_eigenvector_centrality().expect("ok");
        assert!(lambda >= 0.0);
        // Symmetric graph → uniform eigenvector
        for &xi in &x {
            assert!(xi >= 0.0);
        }
        let norm: f64 = x.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert_relative_eq!(norm, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_directed_motif_tensor_cycle() {
        // Directed 3-cycle: 0→1→2→0
        let mut adj = Array2::<f64>::zeros((3, 3));
        adj[[0, 1]] = 1.0;
        adj[[1, 2]] = 1.0;
        adj[[2, 0]] = 1.0;
        let mt = directed_motif_tensor(&adj);
        assert_eq!(mt[[0, 1, 2]], 1); // cycle motif
    }

    // ----- TopologicalFeatures -----------------------------------------------

    #[test]
    fn test_topological_features_triangle() {
        let mut sc = SimplicialComplex::new();
        sc.add_simplex(vec![0, 1, 2]);
        let tf = TopologicalFeatures::from_complex(&sc);
        assert_eq!(tf.betti_numbers[0], 1);
        assert_eq!(tf.euler_characteristic, 1); // 3-3+1=1
    }

    #[test]
    fn test_signature_length() {
        let mut sc = SimplicialComplex::new();
        sc.add_simplex(vec![0, 1]);
        let tf = TopologicalFeatures::from_complex(&sc);
        let sig = tf.signature(8);
        assert_eq!(sig.len(), 8);
    }

    // ----- CellularSheaf -----------------------------------------------------

    #[test]
    fn test_trivial_sheaf_path() {
        // Path 0-1-2 as a trivial sheaf with constant stalks ℝ^1
        let sheaf = CellularSheaf::trivial(3, vec![(0, 1), (1, 2)]).expect("ok");
        let delta = sheaf.coboundary_operator();
        // delta should be 2×3
        assert_eq!(delta.nrows(), 2);
        assert_eq!(delta.ncols(), 3);
    }

    #[test]
    fn test_trivial_sheaf_h0_path() {
        // Path graph: H^0 = 1 (connected)
        let sheaf = CellularSheaf::trivial(3, vec![(0, 1), (1, 2)]).expect("ok");
        assert_eq!(sheaf.cohomology_h0(), 1);
    }

    #[test]
    fn test_trivial_sheaf_h0_two_components() {
        // Two isolated edges: H^0 = 2
        let sheaf = CellularSheaf::trivial(4, vec![(0, 1), (2, 3)]).expect("ok");
        assert_eq!(sheaf.cohomology_h0(), 2);
    }

    #[test]
    fn test_trivial_sheaf_h1_triangle() {
        // Triangle with constant sheaf: H^1 = 1 (one loop)
        let sheaf = CellularSheaf::trivial(3, vec![(0, 1), (0, 2), (1, 2)]).expect("ok");
        assert_eq!(sheaf.cohomology_h1(), 1);
    }

    #[test]
    fn test_hodge_laplacian_symmetric() {
        let sheaf = CellularSheaf::trivial(3, vec![(0, 1), (1, 2)]).expect("ok");
        let l = sheaf.hodge_laplacian_0();
        // Symmetric
        for i in 0..l.nrows() {
            for j in 0..l.ncols() {
                assert_relative_eq!(l[[i, j]], l[[j, i]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_trivial_sheaf_from_graph() {
        let sheaf = trivial_sheaf_from_graph(4, &[(0, 1), (1, 2), (2, 3)]).expect("ok");
        assert_eq!(sheaf.n_nodes, 4);
        assert_eq!(sheaf.n_edges, 3);
    }

    #[test]
    fn test_sheaf_invalid_node() {
        let err = CellularSheaf::trivial(3, vec![(0, 5)]);
        assert!(err.is_err());
    }
}
