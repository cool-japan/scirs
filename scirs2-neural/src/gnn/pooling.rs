//! Graph pooling operations for graph-level readout.
//!
//! Pooling reduces a variable-size node feature matrix `[N, F]` to a
//! fixed-size graph-level representation:
//!
//! - `GlobalMeanPool`  — element-wise mean over nodes.
//! - `GlobalMaxPool`   — element-wise maximum over nodes.
//! - `GlobalAddPool`   — element-wise sum over nodes.
//! - `DiffPool`        — hierarchical differentiable pooling (Ying et al., 2018).
//!
//! All operations are allocation-minimal and operate on plain `Vec<Vec<f32>>`
//! slices matching the GNN layer conventions in this module.

use crate::error::{NeuralError, Result};
use crate::gnn::gcn::{Activation, GCNLayer};
use crate::gnn::graph::Graph;

/// Type alias for DiffPool forward pass result:
/// `(coarsened_features, coarsened_adjacency, link_pred_loss, entropy_loss)`
type DiffPoolForwardResult = (Vec<Vec<f32>>, Vec<Vec<f32>>, f32, f32);

// ──────────────────────────────────────────────────────────────────────────────
// Global pooling operators
// ──────────────────────────────────────────────────────────────────────────────

/// Global mean pooling: `readout[f] = (1/N) Σ_v h_v[f]`.
#[derive(Debug, Clone, Default)]
pub struct GlobalMeanPool;

impl GlobalMeanPool {
    pub fn new() -> Self {
        GlobalMeanPool
    }

    /// Compute the mean node embedding.
    ///
    /// # Returns
    /// A vector of length `F` or an error if `h` is empty.
    pub fn forward(&self, h: &[Vec<f32>]) -> Result<Vec<f32>> {
        let n = h.len();
        if n == 0 {
            return Err(NeuralError::InvalidArgument(
                "Cannot pool over empty node set".to_string(),
            ));
        }
        let f = h[0].len();
        let mut out = vec![0.0_f32; f];
        for row in h {
            if row.len() != f {
                return Err(NeuralError::DimensionMismatch(
                    "Inconsistent feature dimensions in node matrix".to_string(),
                ));
            }
            for (k, &v) in row.iter().enumerate() {
                out[k] += v;
            }
        }
        let scale = 1.0 / n as f32;
        out.iter_mut().for_each(|v| *v *= scale);
        Ok(out)
    }
}

/// Global max pooling: `readout[f] = max_v h_v[f]`.
#[derive(Debug, Clone, Default)]
pub struct GlobalMaxPool;

impl GlobalMaxPool {
    pub fn new() -> Self {
        GlobalMaxPool
    }

    /// Compute the element-wise maximum over all nodes.
    pub fn forward(&self, h: &[Vec<f32>]) -> Result<Vec<f32>> {
        let n = h.len();
        if n == 0 {
            return Err(NeuralError::InvalidArgument(
                "Cannot pool over empty node set".to_string(),
            ));
        }
        let f = h[0].len();
        let mut out = vec![f32::NEG_INFINITY; f];
        for row in h {
            if row.len() != f {
                return Err(NeuralError::DimensionMismatch(
                    "Inconsistent feature dimensions in node matrix".to_string(),
                ));
            }
            for (k, &v) in row.iter().enumerate() {
                if v > out[k] {
                    out[k] = v;
                }
            }
        }
        // Replace any remaining NEG_INFINITY (shouldn't happen if n > 0 and f > 0)
        out.iter_mut().for_each(|v| {
            if v.is_infinite() {
                *v = 0.0;
            }
        });
        Ok(out)
    }
}

/// Global add (sum) pooling: `readout[f] = Σ_v h_v[f]`.
#[derive(Debug, Clone, Default)]
pub struct GlobalAddPool;

impl GlobalAddPool {
    pub fn new() -> Self {
        GlobalAddPool
    }

    /// Compute the element-wise sum over all nodes.
    pub fn forward(&self, h: &[Vec<f32>]) -> Result<Vec<f32>> {
        let n = h.len();
        if n == 0 {
            return Err(NeuralError::InvalidArgument(
                "Cannot pool over empty node set".to_string(),
            ));
        }
        let f = h[0].len();
        let mut out = vec![0.0_f32; f];
        for row in h {
            if row.len() != f {
                return Err(NeuralError::DimensionMismatch(
                    "Inconsistent feature dimensions in node matrix".to_string(),
                ));
            }
            for (k, &v) in row.iter().enumerate() {
                out[k] += v;
            }
        }
        Ok(out)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// DiffPool
// ──────────────────────────────────────────────────────────────────────────────

/// Differentiable Graph Pooling (Ying et al., 2018).
///
/// Uses two GCN networks:
/// - `gnn_pool`  — produces a soft node-to-cluster assignment matrix `S [N, K]`.
/// - `gnn_embed` — produces node embeddings `Z [N, F]`.
///
/// The coarsened graph is computed as:
///
/// ```text
/// S         = softmax( GNN_pool(X, A) )    [N, K]
/// X_pooled  = S^T Z                         [K, F]
/// A_pooled  = S^T A S                       [K, K]
/// ```
///
/// Additionally, two auxiliary losses are returned:
/// - **Link prediction loss**: `‖ A − S Sᵀ ‖_F`
/// - **Entropy regularisation**: `−(1/N) Σ_{v,k} S_{vk} log S_{vk}`
///
/// # Example
/// ```rust
/// use scirs2_neural::gnn::pooling::DiffPool;
/// use scirs2_neural::gnn::graph::Graph;
///
/// let mut g = Graph::new(6, 4);
/// for i in 0..5 { g.add_undirected_edge(i, i + 1).expect("operation should succeed"); }
/// for i in 0..6 { g.set_node_features(i, vec![1.0; 4]).expect("operation should succeed"); }
///
/// let pool = DiffPool::new(4, 3); // 4-dim features, coarsen to 3 clusters
/// let (x_coarse, a_coarse, lp_loss, ent_loss) =
///     pool.forward(&g, &g.node_features).expect("diffpool ok");
/// assert_eq!(x_coarse.len(), 3);
/// assert_eq!(a_coarse.len(), 3);
/// ```
#[derive(Debug, Clone)]
pub struct DiffPool {
    /// GNN for computing soft cluster assignments (output dim = `n_clusters`).
    gnn_pool: GCNLayer,
    /// GNN for computing node embeddings (output dim = `in_features`).
    gnn_embed: GCNLayer,
    n_clusters: usize,
}

impl DiffPool {
    /// Create a new `DiffPool` layer.
    ///
    /// # Arguments
    /// * `in_features` — dimension of input node features.
    /// * `n_clusters`  — number of output clusters (coarsened graph nodes).
    pub fn new(in_features: usize, n_clusters: usize) -> Self {
        let gnn_pool = GCNLayer::new(in_features, n_clusters, Activation::None);
        let gnn_embed = GCNLayer::new(in_features, in_features, Activation::ReLU);
        DiffPool {
            gnn_pool,
            gnn_embed,
            n_clusters,
        }
    }

    /// Forward pass.
    ///
    /// # Returns
    /// `(coarsened_features, coarsened_adjacency, link_pred_loss, entropy_loss)`
    pub fn forward(&self, graph: &Graph, h: &[Vec<f32>]) -> Result<DiffPoolForwardResult> {
        let n = graph.num_nodes;
        if h.len() != n {
            return Err(NeuralError::InvalidArgument(format!(
                "h.len() ({}) must equal graph.num_nodes ({})",
                h.len(),
                n
            )));
        }

        // 1. Compute raw assignment logits S_raw [N, K]
        let s_raw = self.gnn_pool.forward(graph, h)?;

        // 2. Softmax row-wise to get S [N, K]
        let s = s_raw
            .iter()
            .map(|row| Self::softmax_row(row))
            .collect::<Vec<_>>();

        // 3. Compute node embeddings Z [N, F]
        let z = self.gnn_embed.forward(graph, h)?;
        let z_dim = if z.is_empty() { 0 } else { z[0].len() };

        // 4. X_pooled = S^T Z   [K, F]
        let mut x_pooled: Vec<Vec<f32>> = vec![vec![0.0_f32; z_dim]; self.n_clusters];
        for i in 0..n {
            for k in 0..self.n_clusters {
                let s_ik = s[i][k];
                for f in 0..z_dim {
                    x_pooled[k][f] += s_ik * z[i][f];
                }
            }
        }

        // 5. A_pooled = S^T A S   [K, K]
        let adj = &graph.adjacency;

        // Intermediate: (S^T A) [K, N]
        let mut sta: Vec<Vec<f32>> = vec![vec![0.0_f32; n]; self.n_clusters];
        for k in 0..self.n_clusters {
            for i in 0..n {
                for j in 0..n {
                    sta[k][j] += s[i][k] * adj[i][j];
                }
            }
        }
        // (S^T A) S [K, K]
        let mut a_pooled: Vec<Vec<f32>> = vec![vec![0.0_f32; self.n_clusters]; self.n_clusters];
        for k in 0..self.n_clusters {
            for l in 0..self.n_clusters {
                for j in 0..n {
                    a_pooled[k][l] += sta[k][j] * s[j][l];
                }
            }
        }

        // 6. Link prediction loss: ‖ A − S Sᵀ ‖_F²  (Frobenius norm squared)
        //    S S^T [N, N]
        let mut ss_t: Vec<Vec<f32>> = vec![vec![0.0_f32; n]; n];
        for i in 0..n {
            for j in 0..n {
                for (s_ik, s_jk) in s[i].iter().zip(s[j].iter()).take(self.n_clusters) {
                    ss_t[i][j] += s_ik * s_jk;
                }
            }
        }
        let mut lp_loss_sq: f32 = 0.0;
        for i in 0..n {
            for j in 0..n {
                let diff = adj[i][j] - ss_t[i][j];
                lp_loss_sq += diff * diff;
            }
        }
        let lp_loss: f32 = lp_loss_sq.sqrt();

        // 7. Entropy regularisation: −(1/N) Σ_{v,k} S_{vk} log(S_{vk} + ε)
        let ent_loss: f32 = s
            .iter()
            .flat_map(|row| row.iter().map(|&v| -v * (v + 1e-12).ln()))
            .sum::<f32>()
            / n as f32;

        Ok((x_pooled, a_pooled, lp_loss, ent_loss))
    }

    fn softmax_row(row: &[f32]) -> Vec<f32> {
        if row.is_empty() {
            return Vec::new();
        }
        let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_row: Vec<f32> = row.iter().map(|&v| (v - max_val).exp()).collect();
        let sum: f32 = exp_row.iter().sum();
        if sum < 1e-12 {
            vec![1.0 / row.len() as f32; row.len()]
        } else {
            exp_row.iter().map(|&e| e / sum).collect()
        }
    }

    pub fn n_clusters(&self) -> usize {
        self.n_clusters
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_node_features(n: usize, f: usize, val: f32) -> Vec<Vec<f32>> {
        vec![vec![val; f]; n]
    }

    fn chain_graph(n: usize, fdim: usize) -> Graph {
        let mut g = Graph::new(n, fdim);
        for i in 0..n.saturating_sub(1) {
            g.add_undirected_edge(i, i + 1).expect("edge ok");
        }
        for i in 0..n {
            g.set_node_features(i, vec![1.0_f32; fdim]).expect("ok");
        }
        g
    }

    // ── GlobalMeanPool tests ──────────────────────────────────────────────────

    #[test]
    fn test_mean_pool_correctness() {
        let h = vec![vec![1.0_f32, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let pool = GlobalMeanPool::new();
        let out = pool.forward(&h).expect("mean pool ok");
        assert!((out[0] - 3.0).abs() < 1e-6, "mean[0] = {}", out[0]);
        assert!((out[1] - 4.0).abs() < 1e-6, "mean[1] = {}", out[1]);
    }

    #[test]
    fn test_mean_pool_single_node() {
        let h = vec![vec![7.0_f32, -3.0]];
        let pool = GlobalMeanPool::new();
        let out = pool.forward(&h).expect("single node ok");
        assert!((out[0] - 7.0).abs() < 1e-6);
        assert!((out[1] - (-3.0)).abs() < 1e-6);
    }

    #[test]
    fn test_mean_pool_empty_error() {
        let pool = GlobalMeanPool::new();
        assert!(pool.forward(&[]).is_err());
    }

    // ── GlobalMaxPool tests ───────────────────────────────────────────────────

    #[test]
    fn test_max_pool_correctness() {
        let h = vec![vec![1.0_f32, 5.0], vec![3.0, 2.0], vec![2.0, 4.0]];
        let pool = GlobalMaxPool::new();
        let out = pool.forward(&h).expect("max pool ok");
        assert!((out[0] - 3.0).abs() < 1e-6, "max[0] = {}", out[0]);
        assert!((out[1] - 5.0).abs() < 1e-6, "max[1] = {}", out[1]);
    }

    #[test]
    fn test_max_pool_negative_values() {
        let h = vec![vec![-5.0_f32, -1.0], vec![-3.0, -2.0]];
        let pool = GlobalMaxPool::new();
        let out = pool.forward(&h).expect("max pool ok");
        assert!((out[0] - (-3.0)).abs() < 1e-6);
        assert!((out[1] - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_max_pool_empty_error() {
        let pool = GlobalMaxPool::new();
        assert!(pool.forward(&[]).is_err());
    }

    // ── GlobalAddPool tests ───────────────────────────────────────────────────

    #[test]
    fn test_add_pool_correctness() {
        let h = vec![vec![1.0_f32, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let pool = GlobalAddPool::new();
        let out = pool.forward(&h).expect("add pool ok");
        assert!((out[0] - 9.0).abs() < 1e-6, "sum[0] = {}", out[0]);
        assert!((out[1] - 12.0).abs() < 1e-6, "sum[1] = {}", out[1]);
    }

    #[test]
    fn test_add_pool_empty_error() {
        let pool = GlobalAddPool::new();
        assert!(pool.forward(&[]).is_err());
    }

    // ── DiffPool tests ────────────────────────────────────────────────────────

    #[test]
    fn test_diffpool_output_shapes() {
        let g = chain_graph(6, 4);
        let pool = DiffPool::new(4, 3);
        let (x_coarse, a_coarse, _lp, _ent) =
            pool.forward(&g, &g.node_features).expect("diffpool ok");
        assert_eq!(x_coarse.len(), 3, "coarsened nodes");
        assert_eq!(x_coarse[0].len(), 4, "feature dim preserved");
        assert_eq!(a_coarse.len(), 3, "coarsened adj rows");
        assert_eq!(a_coarse[0].len(), 3, "coarsened adj cols");
    }

    #[test]
    fn test_diffpool_losses_non_negative() {
        let g = chain_graph(4, 4);
        let pool = DiffPool::new(4, 2);
        let (_, _, lp_loss, ent_loss) = pool.forward(&g, &g.node_features).expect("diffpool ok");
        assert!(lp_loss >= 0.0, "link pred loss must be >= 0, got {lp_loss}");
        assert!(ent_loss >= 0.0, "entropy loss must be >= 0, got {ent_loss}");
    }

    #[test]
    fn test_diffpool_assignment_stochastic() {
        // Soft assignment S should sum to ≈ 1.0 along cluster axis per node
        let g = chain_graph(5, 3);
        // We can't directly access S, but we can check coarsened adjacency
        // is symmetric (A_pooled = S^T A S where A is symmetric)
        let pool = DiffPool::new(3, 2);
        let (_, a_coarse, _, _) = pool.forward(&g, &g.node_features).expect("ok");
        // A_pooled[i][j] and A_pooled[j][i] should be close (approx symmetric)
        assert!(
            (a_coarse[0][1] - a_coarse[1][0]).abs() < 1e-4,
            "A_pooled should be symmetric"
        );
    }

    #[test]
    fn test_diffpool_all_finite() {
        let g = chain_graph(8, 4);
        let pool = DiffPool::new(4, 3);
        let (x_coarse, a_coarse, lp_loss, ent_loss) =
            pool.forward(&g, &g.node_features).expect("diffpool ok");
        assert!(
            x_coarse
                .iter()
                .flat_map(|r| r.iter())
                .all(|v| v.is_finite()),
            "x_coarse contains non-finite"
        );
        assert!(
            a_coarse
                .iter()
                .flat_map(|r| r.iter())
                .all(|v| v.is_finite()),
            "a_coarse contains non-finite"
        );
        assert!(lp_loss.is_finite());
        assert!(ent_loss.is_finite());
    }

    // ── Combined pipeline test ────────────────────────────────────────────────

    #[test]
    fn test_global_pool_pipeline() {
        // Simulate a typical GNN + global pooling pipeline
        let h = make_node_features(5, 8, 1.0);
        let mean_out = GlobalMeanPool::new().forward(&h).expect("mean ok");
        let max_out = GlobalMaxPool::new().forward(&h).expect("max ok");
        let add_out = GlobalAddPool::new().forward(&h).expect("add ok");

        assert_eq!(mean_out.len(), 8);
        assert_eq!(max_out.len(), 8);
        assert_eq!(add_out.len(), 8);

        // Mean of [1.0; 8] should be [1.0; 8]
        assert!(mean_out.iter().all(|&v| (v - 1.0).abs() < 1e-6));
        // Max of [1.0; 8] should be [1.0; 8]
        assert!(max_out.iter().all(|&v| (v - 1.0).abs() < 1e-6));
        // Sum of 5 × [1.0; 8] should be [5.0; 8]
        assert!(add_out.iter().all(|&v| (v - 5.0).abs() < 1e-6));
    }
}
