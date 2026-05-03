//! Hypergraph Neural Network (HGNN) convolution layers.
//!
//! Implements HGNN (Feng et al. 2019) and its attention-based variant HGAT.
//!
//! ## HGNN Convolution
//!
//! ```text
//! X' = D_v^{-1/2} H W_e D_e^{-1} H^T D_v^{-1/2} X Θ
//! ```
//!
//! where:
//! - `H` is the incidence matrix [n_nodes × n_hyperedges]
//! - `W_e` is the diagonal hyperedge weight matrix
//! - `D_v[i] = Σ_e H[i,e] * w[e]` (weighted node degree)
//! - `D_e[e] = Σ_i H[i,e]` (hyperedge degree)
//! - `Θ` is the learnable weight matrix [in_dim × out_dim]
//!
//! ## HGAT Attention Variant
//!
//! When `use_attention = true`, attention coefficients are computed per node-edge
//! pair before aggregation, allowing the model to focus on the most relevant
//! hyperedge memberships.

use crate::error::{GraphError, Result};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::{Rng, RngExt, SeedableRng};

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for a single HGNN or HGAT convolution layer.
#[derive(Debug, Clone)]
pub struct HgnnLayerConfig {
    /// Input feature dimension.
    pub in_dim: usize,
    /// Output feature dimension.
    pub out_dim: usize,
    /// Whether to use attention-weighted aggregation (HGAT variant).
    pub use_attention: bool,
    /// Number of attention heads (only used when `use_attention = true`).
    pub n_heads: usize,
    /// Dropout rate applied to output features during training (0.0 = no dropout).
    pub dropout: f64,
}

impl Default for HgnnLayerConfig {
    fn default() -> Self {
        Self {
            in_dim: 64,
            out_dim: 64,
            use_attention: false,
            n_heads: 1,
            dropout: 0.0,
        }
    }
}

// ============================================================================
// HgnnLayer
// ============================================================================

/// A single HGNN convolution layer.
///
/// Stores the learnable weight matrix `Θ` of shape `[in_dim × out_dim]`.
/// Optionally stores attention weight vectors when `use_attention = true`.
pub struct HgnnLayer {
    /// Learnable weight matrix: [in_dim × out_dim]
    theta: Array2<f64>,
    /// Attention weight vector: [in_dim] (used only when use_attention = true)
    attn_vec: Array1<f64>,
    config: HgnnLayerConfig,
}

impl HgnnLayer {
    /// Create a new HGNN layer with Xavier-uniform initialised weights.
    ///
    /// # Arguments
    /// * `config` – layer configuration
    /// * `seed`   – RNG seed for reproducible weight initialisation
    pub fn new(config: HgnnLayerConfig, seed: u64) -> Self {
        let mut rng = scirs2_core::random::ChaCha20Rng::seed_from_u64(seed);

        // Xavier uniform scale: sqrt(6 / (fan_in + fan_out))
        let scale = (6.0 / (config.in_dim + config.out_dim) as f64).sqrt();
        let theta = Array2::from_shape_fn((config.in_dim, config.out_dim), |_| {
            rng.random::<f64>() * 2.0 * scale - scale
        });

        let attn_scale = (6.0 / (config.in_dim + 1) as f64).sqrt();
        let attn_vec = Array1::from_shape_fn(config.in_dim, |_| {
            rng.random::<f64>() * 2.0 * attn_scale - attn_scale
        });

        HgnnLayer {
            theta,
            attn_vec,
            config,
        }
    }

    /// Forward pass: `X' = Ã X Θ`
    ///
    /// `Ã = D_v^{-1/2} H diag(w / D_e) H^T D_v^{-1/2}`
    ///
    /// # Arguments
    /// * `incidence`   – incidence matrix H of shape `[n_nodes × n_edges]`
    /// * `node_feats`  – node feature matrix X of shape `[n_nodes × in_dim]`
    /// * `edge_weights` – per-hyperedge weight vector (length = n_edges); if
    ///   `None`, all weights default to 1.0
    ///
    /// # Returns
    /// Node feature matrix of shape `[n_nodes × out_dim]`
    pub fn forward(
        &self,
        incidence: &Array2<f64>,
        node_feats: &Array2<f64>,
        edge_weights: Option<&Array1<f64>>,
    ) -> Result<Array2<f64>> {
        let (n_nodes, n_edges) = incidence.dim();
        let (feat_n, in_dim) = node_feats.dim();

        if feat_n != n_nodes {
            return Err(GraphError::InvalidParameter {
                param: "node_feats".to_string(),
                value: format!("rows={feat_n}"),
                expected: format!("rows={n_nodes} (matching incidence rows)"),
                context: "HgnnLayer::forward".to_string(),
            });
        }
        if in_dim != self.config.in_dim {
            return Err(GraphError::InvalidParameter {
                param: "node_feats".to_string(),
                value: format!("cols={in_dim}"),
                expected: format!("cols={}", self.config.in_dim),
                context: "HgnnLayer::forward".to_string(),
            });
        }

        // Default uniform edge weights
        let default_w = Array1::ones(n_edges);
        let w: &Array1<f64> = edge_weights.unwrap_or(&default_w);

        if self.config.use_attention {
            self.forward_attention(incidence, node_feats, w)
        } else {
            self.forward_standard(incidence, node_feats, w)
        }
    }

    /// Standard HGNN forward (no attention).
    fn forward_standard(
        &self,
        incidence: &Array2<f64>,
        node_feats: &Array2<f64>,
        w: &Array1<f64>,
    ) -> Result<Array2<f64>> {
        let (n_nodes, n_edges) = incidence.dim();

        // D_v[i] = Σ_e H[i,e] * w[e]  (node degree)
        let mut dv: Array1<f64> = Array1::zeros(n_nodes);
        for i in 0..n_nodes {
            for e in 0..n_edges {
                dv[i] += incidence[[i, e]] * w[e];
            }
        }

        // D_e[e] = Σ_i H[i,e]  (hyperedge degree)
        let mut de: Array1<f64> = Array1::zeros(n_edges);
        for e in 0..n_edges {
            for i in 0..n_nodes {
                de[e] += incidence[[i, e]];
            }
        }

        // Dv^{-1/2}: treat zero degrees as 0 (isolated nodes)
        let dv_inv_sqrt: Array1<f64> =
            dv.mapv(|d: f64| if d > 1e-12 { 1.0 / d.sqrt() } else { 0.0 });

        // De^{-1}: treat zero as 0
        let de_inv: Array1<f64> = de.mapv(|d: f64| if d > 1e-12 { 1.0 / d } else { 0.0 });

        // Ã = Dv^{-1/2} * H * diag(w * de_inv) * H^T * Dv^{-1/2}
        //
        // Step 1: T1 = H * diag(w * de_inv)   [n_nodes × n_edges]
        let mut t1: Array2<f64> = Array2::zeros((n_nodes, n_edges));
        for i in 0..n_nodes {
            for e in 0..n_edges {
                t1[[i, e]] = incidence[[i, e]] * w[e] * de_inv[e];
            }
        }

        // Step 2: T2 = T1 * H^T              [n_nodes × n_nodes]
        let mut t2: Array2<f64> = Array2::zeros((n_nodes, n_nodes));
        for i in 0..n_nodes {
            for j in 0..n_nodes {
                let mut val = 0.0_f64;
                for e in 0..n_edges {
                    val += t1[[i, e]] * incidence[[j, e]];
                }
                t2[[i, j]] = val;
            }
        }

        // Step 3: Ã = diag(Dv^{-1/2}) * T2 * diag(Dv^{-1/2})
        let mut a_tilde: Array2<f64> = Array2::zeros((n_nodes, n_nodes));
        for i in 0..n_nodes {
            for j in 0..n_nodes {
                a_tilde[[i, j]] = dv_inv_sqrt[i] * t2[[i, j]] * dv_inv_sqrt[j];
            }
        }

        // Step 4: Output = Ã * X * Θ
        //   First: Z = Ã * X   [n_nodes × in_dim]
        let in_dim = node_feats.dim().1;
        let out_dim = self.config.out_dim;
        let mut z: Array2<f64> = Array2::zeros((n_nodes, in_dim));
        for i in 0..n_nodes {
            for k in 0..in_dim {
                let mut val = 0.0_f64;
                for j in 0..n_nodes {
                    val += a_tilde[[i, j]] * node_feats[[j, k]];
                }
                z[[i, k]] = val;
            }
        }

        //   Then: Output = Z * Θ   [n_nodes × out_dim]
        let mut output: Array2<f64> = Array2::zeros((n_nodes, out_dim));
        for i in 0..n_nodes {
            for k in 0..out_dim {
                let mut val = 0.0;
                for d in 0..in_dim {
                    val += z[[i, d]] * self.theta[[d, k]];
                }
                output[[i, k]] = val;
            }
        }

        Ok(output)
    }

    /// HGAT forward: attention-weighted aggregation over hyperedge memberships.
    ///
    /// For each node v and hyperedge e (where v ∈ e):
    ///   - Compute raw attention score: e_{ve} = LeakyReLU(attn_vec · h_v)
    ///   - Normalise over hyperedges containing v: α_{ve} = softmax_e(e_{ve})
    ///   - Aggregate: h'_v = Σ_e α_{ve} * (mean over u∈e of h_u) * w[e] / D_e[e]
    ///   - Apply θ: output_v = h'_v * Θ
    fn forward_attention(
        &self,
        incidence: &Array2<f64>,
        node_feats: &Array2<f64>,
        w: &Array1<f64>,
    ) -> Result<Array2<f64>> {
        let (n_nodes, n_edges) = incidence.dim();
        let in_dim = node_feats.dim().1;
        let out_dim = self.config.out_dim;

        // D_e[e] = Σ_i H[i,e]
        let mut de: Array1<f64> = Array1::zeros(n_edges);
        for e in 0..n_edges {
            for i in 0..n_nodes {
                de[e] += incidence[[i, e]];
            }
        }
        let de_inv: Array1<f64> = de.mapv(|d: f64| if d > 1e-12 { 1.0 / d } else { 0.0 });

        // Compute mean feature per hyperedge: m[e] = (1/D_e[e]) Σ_{u∈e} h_u
        let mut m_edge: Array2<f64> = Array2::zeros((n_edges, in_dim));
        for e in 0..n_edges {
            let mut sum: Array1<f64> = Array1::zeros(in_dim);
            for i in 0..n_nodes {
                if incidence[[i, e]] > 0.0 {
                    for d in 0..in_dim {
                        sum[d] += node_feats[[i, d]];
                    }
                }
            }
            for d in 0..in_dim {
                m_edge[[e, d]] = sum[d] * de_inv[e];
            }
        }

        // Compute raw attention scores using LeakyReLU(attn_vec · h_v) per node
        let leaky_alpha = 0.2_f64;
        let mut output: Array2<f64> = Array2::zeros((n_nodes, out_dim));

        for v in 0..n_nodes {
            // Collect hyperedges containing v
            let edges_of_v: Vec<usize> =
                (0..n_edges).filter(|&e| incidence[[v, e]] > 0.0).collect();

            if edges_of_v.is_empty() {
                // Isolated node: no message, output remains zero
                continue;
            }

            // Compute raw attention scores e_{ve} = LeakyReLU(attn_vec · h_v)
            // (same for all edges since it only depends on the query node here)
            let score_v: f64 = {
                let raw: f64 = (0..in_dim)
                    .map(|d| self.attn_vec[d] * node_feats[[v, d]])
                    .sum();
                if raw >= 0.0 {
                    raw
                } else {
                    leaky_alpha * raw
                }
            };

            // Softmax is trivially 1/n_edges_of_v when scores are uniform;
            // expand to per-edge scores incorporating edge weights.
            let edge_scores: Vec<f64> = edges_of_v
                .iter()
                .map(|&e| {
                    let raw: f64 = (0..in_dim).map(|d| self.attn_vec[d] * m_edge[[e, d]]).sum();
                    let s = raw + score_v;
                    let leaky = if s >= 0.0 { s } else { leaky_alpha * s };
                    leaky * w[e]
                })
                .collect();

            // Numerical-stable softmax
            let max_s = edge_scores
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            let exps: Vec<f64> = edge_scores.iter().map(|&s| (s - max_s).exp()).collect();
            let sum_exp: f64 = exps.iter().sum();
            let alphas: Vec<f64> = exps.iter().map(|&e| e / sum_exp).collect();

            // Aggregate: agg_v = Σ_e alpha_{ve} * m[e]
            let mut agg: Array1<f64> = Array1::zeros(in_dim);
            for (k, &e) in edges_of_v.iter().enumerate() {
                for d in 0..in_dim {
                    agg[d] += alphas[k] * m_edge[[e, d]];
                }
            }

            // Apply Θ: output_v = agg_v * Θ
            for k in 0..out_dim {
                let mut val = 0.0_f64;
                for d in 0..in_dim {
                    val += agg[d] * self.theta[[d, k]];
                }
                output[[v, k]] = val;
            }
        }

        Ok(output)
    }

    /// Element-wise ReLU activation.
    pub fn relu(x: Array2<f64>) -> Array2<f64> {
        x.mapv(|v| if v > 0.0 { v } else { 0.0 })
    }

    /// Number of learnable parameters in this layer.
    pub fn n_params(&self) -> usize {
        self.theta.len() + self.attn_vec.len()
    }
}

// ============================================================================
// HgnnNetwork
// ============================================================================

/// Multi-layer HGNN network.
///
/// Layers alternate convolution and ReLU activation; the final layer has no
/// activation applied (the caller is free to apply softmax / sigmoid as
/// appropriate for the downstream task).
pub struct HgnnNetwork {
    layers: Vec<HgnnLayer>,
}

impl HgnnNetwork {
    /// Build a multi-layer HGNN.
    ///
    /// # Arguments
    /// * `dims` – sequence of dimensions `[in_dim, h1, h2, ..., out_dim]`.
    ///   Must have at least two elements.
    /// * `use_attention` – whether every layer uses the HGAT attention variant.
    /// * `seed` – base RNG seed; each layer receives `seed + layer_index`.
    ///
    /// # Panics
    /// Panics if `dims.len() < 2`.
    pub fn new(dims: &[usize], use_attention: bool, seed: u64) -> Self {
        assert!(
            dims.len() >= 2,
            "dims must have at least 2 elements (in_dim, out_dim)"
        );
        let layers = dims
            .windows(2)
            .enumerate()
            .map(|(i, w)| {
                let cfg = HgnnLayerConfig {
                    in_dim: w[0],
                    out_dim: w[1],
                    use_attention,
                    n_heads: 1,
                    dropout: 0.0,
                };
                HgnnLayer::new(cfg, seed + i as u64)
            })
            .collect();
        HgnnNetwork { layers }
    }

    /// Forward pass through all layers.
    ///
    /// ReLU is applied after every hidden layer; the output of the last layer
    /// is returned without activation.
    ///
    /// # Arguments
    /// * `incidence`    – incidence matrix H: `[n_nodes × n_edges]`
    /// * `node_feats`   – node feature matrix X: `[n_nodes × in_dim]`
    /// * `edge_weights` – optional per-hyperedge weight vector
    ///
    /// # Returns
    /// Output feature matrix: `[n_nodes × out_dim]`
    pub fn forward(
        &self,
        incidence: &Array2<f64>,
        node_feats: &Array2<f64>,
        edge_weights: Option<&Array1<f64>>,
    ) -> Result<Array2<f64>> {
        let n_layers = self.layers.len();
        let mut x = node_feats.clone();

        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(incidence, &x, edge_weights)?;
            // Apply ReLU after all but the last layer
            if i + 1 < n_layers {
                x = HgnnLayer::relu(x);
            }
        }

        Ok(x)
    }

    /// Total number of learnable parameters across all layers.
    pub fn n_params(&self) -> usize {
        self.layers.iter().map(|l| l.n_params()).sum()
    }

    /// Number of layers.
    pub fn depth(&self) -> usize {
        self.layers.len()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    /// Build a small incidence matrix: 4 nodes, 3 hyperedges
    ///   e0 = {0,1}, e1 = {1,2}, e2 = {2,3}
    fn small_incidence() -> Array2<f64> {
        let mut h = Array2::zeros((4, 3));
        h[[0, 0]] = 1.0;
        h[[1, 0]] = 1.0;
        h[[1, 1]] = 1.0;
        h[[2, 1]] = 1.0;
        h[[2, 2]] = 1.0;
        h[[3, 2]] = 1.0;
        h
    }

    /// Build an identity incidence: each node belongs to exactly one hyperedge.
    fn identity_incidence(n: usize) -> Array2<f64> {
        let mut h = Array2::zeros((n, n));
        for i in 0..n {
            h[[i, i]] = 1.0;
        }
        h
    }

    #[test]
    fn test_output_shape() {
        let h = small_incidence();
        let x = Array2::ones((4, 8));
        let cfg = HgnnLayerConfig {
            in_dim: 8,
            out_dim: 16,
            ..Default::default()
        };
        let layer = HgnnLayer::new(cfg, 42);
        let out = layer.forward(&h, &x, None).expect("forward ok");
        assert_eq!(out.dim(), (4, 16));
    }

    #[test]
    fn test_identity_incidence_is_diagonal() {
        // With identity incidence H=I and unit edge weights:
        // D_v[i]=1, D_e[e]=1, Ã=I, so output = X * Θ
        let n = 4;
        let h = identity_incidence(n);
        // identity feature matrix scaled to in_dim
        let in_dim = 4;
        let out_dim = 4;
        let x = Array2::eye(n);
        let cfg = HgnnLayerConfig {
            in_dim,
            out_dim,
            ..Default::default()
        };
        let layer = HgnnLayer::new(cfg, 7);
        let out = layer.forward(&h, &x, None).expect("forward ok");
        // With Ã = I, output = X @ theta = I @ theta = theta
        // Compare: out[i, :] ≈ theta[i, :]
        for i in 0..n {
            for k in 0..out_dim {
                let diff = (out[[i, k]] - layer.theta[[i, k]]).abs();
                assert!(
                    diff < 1e-10,
                    "out[{i},{k}]={} != theta[{i},{k}]={}",
                    out[[i, k]],
                    layer.theta[[i, k]]
                );
            }
        }
    }

    #[test]
    fn test_output_bounded_with_unit_features() {
        // With unit features and reasonable weights the output should be finite
        let h = small_incidence();
        let x = Array2::ones((4, 4));
        let cfg = HgnnLayerConfig {
            in_dim: 4,
            out_dim: 4,
            ..Default::default()
        };
        let layer = HgnnLayer::new(cfg, 99);
        let out = layer.forward(&h, &x, None).expect("forward ok");
        for v in out.iter() {
            assert!(v.is_finite(), "output contains non-finite value: {v}");
        }
    }

    #[test]
    fn test_zero_dropout_is_identity_of_forward() {
        // dropout=0.0 means no masking; two calls with same input return same output
        let h = small_incidence();
        let x = Array2::from_shape_fn((4, 8), |(i, j)| (i + j) as f64 * 0.1);
        let cfg = HgnnLayerConfig {
            in_dim: 8,
            out_dim: 4,
            dropout: 0.0,
            ..Default::default()
        };
        let layer = HgnnLayer::new(cfg, 1);
        let out1 = layer.forward(&h, &x, None).expect("ok");
        let out2 = layer.forward(&h, &x, None).expect("ok");
        for (a, b) in out1.iter().zip(out2.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn test_n_params_counts_correctly() {
        let cfg = HgnnLayerConfig {
            in_dim: 8,
            out_dim: 16,
            use_attention: false,
            ..Default::default()
        };
        let layer = HgnnLayer::new(cfg, 0);
        // theta: 8*16=128  attn_vec: 8  total: 136
        assert_eq!(layer.n_params(), 8 * 16 + 8);
    }

    #[test]
    fn test_multi_layer_output_shape() {
        let h = small_incidence(); // 4 nodes
        let x = Array2::ones((4, 16));
        let net = HgnnNetwork::new(&[16, 32, 8], false, 42);
        let out = net.forward(&h, &x, None).expect("ok");
        assert_eq!(out.dim(), (4, 8));
    }

    #[test]
    fn test_network_n_params() {
        let net = HgnnNetwork::new(&[8, 16, 4], false, 0);
        // Layer 0: theta 8*16=128, attn 8  → 136
        // Layer 1: theta 16*4=64, attn 16 → 80
        // Total: 216
        assert_eq!(net.n_params(), 8 * 16 + 8 + 16 * 4 + 16);
    }

    #[test]
    fn test_theta_small_init() {
        let cfg = HgnnLayerConfig {
            in_dim: 64,
            out_dim: 64,
            ..Default::default()
        };
        let layer = HgnnLayer::new(cfg, 1234);
        let scale = (6.0_f64 / (64.0 + 64.0)).sqrt();
        for v in layer.theta.iter() {
            assert!(
                v.abs() <= scale + 1e-9,
                "theta value {v} exceeds Xavier bound {scale}"
            );
        }
    }

    #[test]
    fn test_attention_output_shape() {
        let h = small_incidence();
        let x = Array2::ones((4, 8));
        let cfg = HgnnLayerConfig {
            in_dim: 8,
            out_dim: 4,
            use_attention: true,
            n_heads: 1,
            dropout: 0.0,
        };
        let layer = HgnnLayer::new(cfg, 5);
        let out = layer.forward(&h, &x, None).expect("ok");
        assert_eq!(out.dim(), (4, 4));
    }

    #[test]
    fn test_relu_zeros_negatives() {
        let x = Array2::from_shape_vec((2, 3), vec![-1.0, 0.0, 1.0, -0.5, 2.0, -3.0]).expect("ok");
        let r = HgnnLayer::relu(x);
        assert_eq!(r[[0, 0]], 0.0);
        assert_eq!(r[[0, 2]], 1.0);
        assert_eq!(r[[1, 1]], 2.0);
        assert_eq!(r[[1, 2]], 0.0);
    }

    #[test]
    fn test_edge_weights_change_output() {
        let h = small_incidence();
        let x = Array2::ones((4, 4));
        let cfg = HgnnLayerConfig {
            in_dim: 4,
            out_dim: 4,
            ..Default::default()
        };
        let layer = HgnnLayer::new(cfg, 42);
        let w1 = Array1::ones(3);
        let w2 = Array1::from_vec(vec![2.0, 1.0, 0.5]);
        let out1 = layer.forward(&h, &x, Some(&w1)).expect("ok");
        let out2 = layer.forward(&h, &x, Some(&w2)).expect("ok");
        // Different weights → different output
        let diff: f64 = out1
            .iter()
            .zip(out2.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff > 1e-10,
            "different edge weights should produce different output"
        );
    }
}
