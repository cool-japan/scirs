//! Hypergraph Attention Network (HAN).
//!
//! Implements a dual-attention mechanism over hypergraphs:
//! 1. **Node-to-hyperedge attention**: each hyperedge aggregates member node features
//! 2. **Hyperedge-to-node attention**: each node aggregates features from its hyperedges
//!
//! ## Architecture
//!
//! Given incidence matrix `H ∈ {0,1}^{N×M}` (N nodes, M hyperedges):
//!
//! ### Node → Hyperedge
//! ```text
//! a_{ih} = softmax_i ∈ h [ (W_Q x_i · W_K ê_h) / sqrt(d) ]
//! e_h^new = sum_{i ∈ h} a_{ih} W_V x_i
//! ```
//!
//! ### Hyperedge → Node
//! ```text
//! b_{hi} = softmax_{h ∋ i} [ (W_Q ê_h^new · W_K x_i) / sqrt(d) ]
//! x_i^new = sum_{h ∋ i} b_{hi} W_V ê_h^new
//! ```
//!
//! ## References
//!
//! - Ding et al. (2020). "HNHN: Hypergraph Networks with Hyperedge Neurons."
//! - Bai et al. (2021). "Hypergraph Convolution and Hypergraph Attention."

use crate::error::{GraphError, Result};
use scirs2_core::ndarray::Array2;
use scirs2_core::random::{Rng, RngExt};

// ============================================================================
// Linear layer (local to this module, same pattern as egnn.rs)
// ============================================================================

/// A simple linear layer: y = W x + b.
#[derive(Debug, Clone)]
struct Linear {
    weight: Vec<Vec<f64>>,
    bias: Vec<f64>,
    out_dim: usize,
    in_dim: usize,
}

impl Linear {
    fn new(in_dim: usize, out_dim: usize) -> Self {
        let scale = (2.0 / in_dim as f64).sqrt();
        let mut rng = scirs2_core::random::rng();
        let weight: Vec<Vec<f64>> = (0..out_dim)
            .map(|_| {
                (0..in_dim)
                    .map(|_| (rng.random::<f64>() * 2.0 - 1.0) * scale)
                    .collect()
            })
            .collect();
        Linear {
            weight,
            bias: vec![0.0; out_dim],
            out_dim,
            in_dim,
        }
    }

    fn forward(&self, x: &[f64]) -> Vec<f64> {
        let mut out = self.bias.clone();
        for (i, row) in self.weight.iter().enumerate() {
            for (j, &w) in row.iter().enumerate() {
                out[i] += w * x[j];
            }
        }
        out
    }
}

// ============================================================================
// Layer Norm helper
// ============================================================================

fn layer_norm(x: &mut [f64]) {
    let n = x.len() as f64;
    let mean: f64 = x.iter().sum::<f64>() / n;
    let var: f64 = x.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / n;
    let std_dev = (var + 1e-8).sqrt();
    for v in x.iter_mut() {
        *v = (*v - mean) / std_dev;
    }
}

fn softmax(xs: &[f64]) -> Vec<f64> {
    if xs.is_empty() {
        return Vec::new();
    }
    let max_val = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = xs.iter().map(|x| (x - max_val).exp()).collect();
    let sum: f64 = exps.iter().sum::<f64>().max(1e-15);
    exps.iter().map(|e| e / sum).collect()
}

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for a HypergraphAttentionLayer.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct HypergraphAttentionConfig {
    /// Node feature (and hyperedge feature) dimension.
    pub hidden_dim: usize,
    /// Number of attention heads.
    pub n_heads: usize,
    /// Dropout rate (0.0 = disabled; for inference/tests, dropout is typically turned off).
    pub dropout: f64,
    /// Whether to apply layer normalisation.
    pub use_layer_norm: bool,
}

impl Default for HypergraphAttentionConfig {
    fn default() -> Self {
        HypergraphAttentionConfig {
            hidden_dim: 64,
            n_heads: 4,
            dropout: 0.1,
            use_layer_norm: true,
        }
    }
}

// ============================================================================
// HypergraphAttentionLayer
// ============================================================================

/// A single Hypergraph Attention layer.
///
/// Processes node features through two rounds of dual-direction attention:
/// 1. Nodes → Hyperedges (aggregate node info into hyperedge representations)
/// 2. Hyperedges → Nodes (aggregate hyperedge info back to node representations)
#[derive(Debug, Clone)]
pub struct HypergraphAttentionLayer {
    /// Query projection for node features.
    w_q_node: Linear,
    /// Key projection for hyperedge features.
    w_k_edge: Linear,
    /// Value projection for node features.
    w_v_node: Linear,
    /// Query projection for hyperedge features.
    w_q_edge: Linear,
    /// Key projection for node features.
    w_k_node: Linear,
    /// Value projection for hyperedge features.
    w_v_edge: Linear,
    /// Output projection back to node feature space.
    w_o: Linear,
    /// Hyperedge initial feature projection W_e.
    w_e: Linear,
    /// Configuration.
    config: HypergraphAttentionConfig,
    /// Input feature dimension.
    in_dim: usize,
}

impl HypergraphAttentionLayer {
    /// Create a new HypergraphAttentionLayer.
    ///
    /// # Arguments
    /// - `in_dim`: input node feature dimension
    /// - `config`: layer configuration
    pub fn new(in_dim: usize, config: HypergraphAttentionConfig) -> Self {
        let h = config.hidden_dim;
        HypergraphAttentionLayer {
            w_q_node: Linear::new(in_dim, h),
            w_k_edge: Linear::new(h, h),
            w_v_node: Linear::new(in_dim, h),
            w_q_edge: Linear::new(h, h),
            w_k_node: Linear::new(in_dim, h),
            w_v_edge: Linear::new(h, h),
            w_o: Linear::new(h, in_dim),
            w_e: Linear::new(in_dim, h),
            config,
            in_dim,
        }
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// - `node_feats`: node features, shape [N × in_dim]
    /// - `incidence_matrix`: H ∈ {0,1}^{N×M}, shape [N × M]
    ///
    /// # Returns
    /// Updated node features, shape [N × in_dim].
    pub fn forward(
        &self,
        node_feats: &Array2<f64>,
        incidence_matrix: &Array2<f64>,
    ) -> Result<Array2<f64>> {
        let n_nodes = node_feats.nrows();
        let in_d = node_feats.ncols();

        if in_d != self.in_dim {
            return Err(GraphError::InvalidParameter {
                param: "node_feats".to_string(),
                value: format!("ncols={in_d}"),
                expected: format!("ncols={}", self.in_dim),
                context: "HypergraphAttentionLayer::forward".to_string(),
            });
        }
        if incidence_matrix.nrows() != n_nodes {
            return Err(GraphError::InvalidParameter {
                param: "incidence_matrix".to_string(),
                value: format!("nrows={}", incidence_matrix.nrows()),
                expected: format!("nrows={n_nodes}"),
                context: "HypergraphAttentionLayer::forward".to_string(),
            });
        }

        let n_edges = incidence_matrix.ncols();
        let h_dim = self.config.hidden_dim;
        let scale = (h_dim as f64).sqrt();

        // ── Step 1: compute initial hyperedge features as mean of member nodes ─
        // e_h = W_e * (mean_{i ∈ h} x_i)
        let mut edge_feats: Vec<Vec<f64>> = Vec::with_capacity(n_edges);
        for edge_h in 0..n_edges {
            let members: Vec<usize> = (0..n_nodes)
                .filter(|&i| incidence_matrix[[i, edge_h]] > 0.5)
                .collect();
            let mean_feat = if members.is_empty() {
                vec![0.0_f64; in_d]
            } else {
                let inv_n = 1.0 / members.len() as f64;
                let mut mean = vec![0.0_f64; in_d];
                for &i in &members {
                    for d in 0..in_d {
                        mean[d] += node_feats[[i, d]] * inv_n;
                    }
                }
                mean
            };
            edge_feats.push(self.w_e.forward(&mean_feat));
        }

        // ── Step 2: node → hyperedge attention ─────────────────────────────
        // For each hyperedge h, attend over its member nodes
        let mut edge_feats_new: Vec<Vec<f64>> = vec![vec![0.0_f64; h_dim]; n_edges];

        for edge_h in 0..n_edges {
            let members: Vec<usize> = (0..n_nodes)
                .filter(|&i| incidence_matrix[[i, edge_h]] > 0.5)
                .collect();
            if members.is_empty() {
                edge_feats_new[edge_h] = edge_feats[edge_h].clone();
                continue;
            }

            let k_e = self.w_k_edge.forward(&edge_feats[edge_h]);

            // Attention scores: Q_i · K_h / sqrt(d)
            let scores: Vec<f64> = members
                .iter()
                .map(|&i| {
                    let q_i = self.w_q_node.forward(
                        node_feats
                            .row(i)
                            .as_slice()
                            .unwrap_or(&[])
                            .to_vec()
                            .as_slice(),
                    );
                    let dot: f64 = q_i.iter().zip(k_e.iter()).map(|(a, b)| a * b).sum();
                    dot / scale
                })
                .collect();

            let alphas = softmax(&scores);

            // Aggregate: e_h_new = sum_i alpha_i * V_i
            let e_new = &mut edge_feats_new[edge_h];
            for (k, &i) in members.iter().enumerate() {
                let v_i = self.w_v_node.forward(
                    node_feats
                        .row(i)
                        .as_slice()
                        .unwrap_or(&[])
                        .to_vec()
                        .as_slice(),
                );
                for d in 0..h_dim {
                    e_new[d] += alphas[k] * v_i[d];
                }
            }
        }

        // ── Step 3: hyperedge → node attention ─────────────────────────────
        let mut node_feats_new = Array2::zeros((n_nodes, in_d));
        let mut residual_used = vec![false; n_nodes];

        for node_i in 0..n_nodes {
            let incident_edges: Vec<usize> = (0..n_edges)
                .filter(|&h| incidence_matrix[[node_i, h]] > 0.5)
                .collect();
            if incident_edges.is_empty() {
                // Residual: copy input
                for d in 0..in_d {
                    node_feats_new[[node_i, d]] = node_feats[[node_i, d]];
                }
                residual_used[node_i] = true;
                continue;
            }

            let k_i = self.w_k_node.forward(
                node_feats
                    .row(node_i)
                    .as_slice()
                    .unwrap_or(&[])
                    .to_vec()
                    .as_slice(),
            );

            // Attention scores: Q_h · K_i / sqrt(d)
            let scores: Vec<f64> = incident_edges
                .iter()
                .map(|&h| {
                    let q_h = self.w_q_edge.forward(&edge_feats_new[h]);
                    let dot: f64 = q_h.iter().zip(k_i.iter()).map(|(a, b)| a * b).sum();
                    dot / scale
                })
                .collect();

            let betas = softmax(&scores);

            // Aggregate: x_i_new = sum_h beta_h * W_V e_h_new
            let mut x_new_h = vec![0.0_f64; h_dim];
            for (k, &h) in incident_edges.iter().enumerate() {
                let v_h = self.w_v_edge.forward(&edge_feats_new[h]);
                for d in 0..h_dim {
                    x_new_h[d] += betas[k] * v_h[d];
                }
            }

            // Project back to input dim + residual
            let projected = self.w_o.forward(&x_new_h);
            let mut out_i: Vec<f64> = projected
                .iter()
                .enumerate()
                .map(|(d, &p)| p + node_feats[[node_i, d]])
                .collect();

            // Layer norm
            if self.config.use_layer_norm {
                layer_norm(&mut out_i);
            }

            for d in 0..in_d {
                node_feats_new[[node_i, d]] = out_i[d];
            }
        }

        Ok(node_feats_new)
    }
}

// ============================================================================
// HypergraphAttentionNetwork
// ============================================================================

/// Multi-layer Hypergraph Attention Network.
#[derive(Debug, Clone)]
pub struct HypergraphAttentionNetwork {
    /// Stacked hypergraph attention layers.
    pub layers: Vec<HypergraphAttentionLayer>,
    /// Inter-layer MLPs (feedforward block after each attention layer).
    ff_layers: Vec<(Linear, Linear)>,
    /// Input dimension.
    pub in_dim: usize,
    /// Configuration (of the first layer, representative).
    pub config: HypergraphAttentionConfig,
}

impl HypergraphAttentionNetwork {
    /// Create a multi-layer Hypergraph Attention Network.
    ///
    /// # Arguments
    /// - `in_dim`: input node feature dimension
    /// - `n_layers`: number of stacked attention layers
    /// - `config`: configuration (shared across layers)
    pub fn new(in_dim: usize, n_layers: usize, config: HypergraphAttentionConfig) -> Self {
        let h = config.hidden_dim;
        let layers = (0..n_layers)
            .map(|_| HypergraphAttentionLayer::new(in_dim, config.clone()))
            .collect();
        // Feedforward block: in_dim → h → in_dim
        let ff_layers = (0..n_layers)
            .map(|_| (Linear::new(in_dim, h), Linear::new(h, in_dim)))
            .collect();
        HypergraphAttentionNetwork {
            layers,
            ff_layers,
            in_dim,
            config,
        }
    }

    /// Forward pass through all layers.
    ///
    /// # Arguments
    /// - `node_feats`: initial node features [N × in_dim]
    /// - `incidence_matrix`: H ∈ {0,1}^{N×M}
    ///
    /// # Returns
    /// Final node features [N × in_dim].
    pub fn forward(
        &self,
        node_feats: &Array2<f64>,
        incidence_matrix: &Array2<f64>,
    ) -> Result<Array2<f64>> {
        let mut x = node_feats.clone();
        for (layer, (ff1, ff2)) in self.layers.iter().zip(self.ff_layers.iter()) {
            let x_att = layer.forward(&x, incidence_matrix)?;
            // Feedforward: apply per-node, with ReLU + residual
            let mut x_ff = Array2::zeros(x_att.dim());
            for i in 0..x_att.nrows() {
                let row: Vec<f64> = x_att.row(i).to_vec();
                let mut h_mid = ff1.forward(&row);
                for v in h_mid.iter_mut() {
                    *v = v.max(0.0); // ReLU
                }
                let projected = ff2.forward(&h_mid);
                let mut out: Vec<f64> = projected
                    .iter()
                    .zip(row.iter())
                    .map(|(p, r)| p + r)
                    .collect();
                if self.config.use_layer_norm {
                    layer_norm(&mut out);
                }
                for d in 0..self.in_dim {
                    x_ff[[i, d]] = out[d];
                }
            }
            x = x_ff;
        }
        Ok(x)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn make_node_feats(n_nodes: usize, in_dim: usize) -> Array2<f64> {
        let data: Vec<f64> = (0..n_nodes * in_dim)
            .map(|i| (i as f64 + 1.0) * 0.1)
            .collect();
        Array2::from_shape_vec((n_nodes, in_dim), data).expect("node feats")
    }

    fn make_incidence_matrix(n_nodes: usize, n_edges: usize) -> Array2<f64> {
        // Simple hyperedge: hyperedge 0 = {0,1,2}, hyperedge 1 = {2,3,4}
        let mut h = Array2::zeros((n_nodes, n_edges));
        if n_nodes >= 3 && n_edges >= 1 {
            h[[0, 0]] = 1.0;
            h[[1, 0]] = 1.0;
            h[[2, 0]] = 1.0;
        }
        if n_nodes >= 5 && n_edges >= 2 {
            h[[2, 1]] = 1.0;
            h[[3, 1]] = 1.0;
            h[[4, 1]] = 1.0;
        }
        h
    }

    #[test]
    fn test_attention_layer_output_shape() {
        let config = HypergraphAttentionConfig {
            hidden_dim: 8,
            n_heads: 2,
            ..Default::default()
        };
        let layer = HypergraphAttentionLayer::new(4, config);
        let node_feats = make_node_feats(5, 4);
        let incidence = make_incidence_matrix(5, 2);
        let out = layer.forward(&node_feats, &incidence).expect("forward");
        assert_eq!(out.nrows(), 5, "output node count");
        assert_eq!(out.ncols(), 4, "output feature dim");
    }

    #[test]
    fn test_attention_handles_varying_hyperedge_sizes() {
        // Hyperedge 0: {0} (size 1), hyperedge 1: {1,2,3,4} (size 4)
        let mut incidence = Array2::zeros((5, 2));
        incidence[[0, 0]] = 1.0;
        incidence[[1, 1]] = 1.0;
        incidence[[2, 1]] = 1.0;
        incidence[[3, 1]] = 1.0;
        incidence[[4, 1]] = 1.0;

        let config = HypergraphAttentionConfig {
            hidden_dim: 8,
            n_heads: 2,
            ..Default::default()
        };
        let layer = HypergraphAttentionLayer::new(4, config);
        let node_feats = make_node_feats(5, 4);
        let out = layer
            .forward(&node_feats, &incidence)
            .expect("varying sizes");
        assert_eq!(out.shape(), &[5, 4]);
    }

    #[test]
    fn test_attention_output_is_finite() {
        let config = HypergraphAttentionConfig {
            hidden_dim: 8,
            ..Default::default()
        };
        let layer = HypergraphAttentionLayer::new(4, config);
        let node_feats = make_node_feats(5, 4);
        let incidence = make_incidence_matrix(5, 2);
        let out = layer.forward(&node_feats, &incidence).expect("forward");
        for v in out.iter() {
            assert!(v.is_finite(), "output must be finite, got {v}");
        }
    }

    #[test]
    fn test_network_stacked_output_shape() {
        let config = HypergraphAttentionConfig {
            hidden_dim: 8,
            n_heads: 2,
            ..Default::default()
        };
        let net = HypergraphAttentionNetwork::new(4, 3, config);
        let node_feats = make_node_feats(5, 4);
        let incidence = make_incidence_matrix(5, 2);
        let out = net.forward(&node_feats, &incidence).expect("net forward");
        assert_eq!(out.shape(), &[5, 4]);
    }

    #[test]
    fn test_network_output_is_finite() {
        let config = HypergraphAttentionConfig {
            hidden_dim: 8,
            ..Default::default()
        };
        let net = HypergraphAttentionNetwork::new(4, 2, config);
        let node_feats = make_node_feats(5, 4);
        let incidence = make_incidence_matrix(5, 2);
        let out = net.forward(&node_feats, &incidence).expect("forward");
        for v in out.iter() {
            assert!(v.is_finite(), "network output must be finite");
        }
    }

    #[test]
    fn test_empty_hyperedge() {
        // Node with no hyperedge membership gets residual connection
        let incidence = Array2::zeros((3, 2)); // all zeros → no memberships
        let config = HypergraphAttentionConfig {
            hidden_dim: 8,
            use_layer_norm: false,
            ..Default::default()
        };
        let layer = HypergraphAttentionLayer::new(4, config);
        let node_feats = make_node_feats(3, 4);
        let out = layer
            .forward(&node_feats, &incidence)
            .expect("empty hyperedge");
        // Should equal input (residual path)
        for i in 0..3 {
            for d in 0..4 {
                assert!(
                    (out[[i, d]] - node_feats[[i, d]]).abs() < 1e-12,
                    "residual mismatch at ({i},{d})"
                );
            }
        }
    }
}
