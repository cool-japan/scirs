//! Graph Isomorphism Network (Xu et al., 2019).
//!
//! GIN uses a 2-layer MLP to process node features after neighbourhood sum
//! aggregation:
//!
//! ```text
//! h_v' = MLP( (1 + ε) · h_v + Σ_{u ∈ N(v)} h_u )
//! ```
//!
//! GIN is maximally expressive among message-passing GNNs that use injective
//! aggregation functions, and can distinguish any graphs that the
//! Weisfeiler-Lehman (WL) graph isomorphism test can distinguish.

use crate::error::{NeuralError, Result};
use crate::gnn::graph::Graph;

// ──────────────────────────────────────────────────────────────────────────────
// LCG weight initialisation
// ──────────────────────────────────────────────────────────────────────────────

fn xavier_init(fan_in: usize, fan_out: usize, seed: u64) -> Vec<Vec<f32>> {
    let limit = (6.0_f64 / (fan_in + fan_out) as f64).sqrt() as f32;
    let mut state: u64 = 55443322110099887_u64.wrapping_add(seed);
    let mut rng = || -> f32 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let v = (state >> 33) as u32 as f64 / u32::MAX as f64;
        v as f32 * 2.0 * limit - limit
    };
    (0..fan_in)
        .map(|_| (0..fan_out).map(|_| rng()).collect())
        .collect()
}

// ──────────────────────────────────────────────────────────────────────────────
// GINLayer
// ──────────────────────────────────────────────────────────────────────────────

/// Graph Isomorphism Network layer.
///
/// The MLP has two linear layers with a ReLU and batch-normalisation-free
/// design:
///
/// ```text
/// MLP(x) = W2 · ReLU( W1 · x + b1 ) + b2
/// ```
///
/// # Example
/// ```rust
/// use scirs2_neural::gnn::gin::GINLayer;
/// use scirs2_neural::gnn::graph::Graph;
///
/// let mut g = Graph::new(4, 5);
/// g.add_undirected_edge(0, 1).expect("operation should succeed");
/// g.add_undirected_edge(1, 2).expect("operation should succeed");
/// g.add_undirected_edge(2, 3).expect("operation should succeed");
/// for i in 0..4 { g.set_node_features(i, vec![1.0; 5]).expect("operation should succeed"); }
///
/// let layer = GINLayer::new(5, 10, 16, 0.0);
/// let out = layer.forward(&g, &g.node_features).expect("forward ok");
/// assert_eq!(out.len(), 4);
/// assert_eq!(out[0].len(), 10);
/// ```
#[derive(Debug, Clone)]
pub struct GINLayer {
    in_features: usize,
    out_features: usize,
    /// ε — learnable or fixed scaling factor for the self embedding.
    epsilon: f32,
    /// First MLP layer weights `[in_features, hidden]`.
    mlp_w1: Vec<Vec<f32>>,
    /// First MLP layer bias `[hidden]`.
    mlp_b1: Vec<f32>,
    /// Second MLP layer weights `[hidden, out_features]`.
    mlp_w2: Vec<Vec<f32>>,
    /// Second MLP layer bias `[out_features]`.
    mlp_b2: Vec<f32>,
    hidden: usize,
}

impl GINLayer {
    /// Create a new `GINLayer`.
    ///
    /// # Arguments
    /// * `in_features`  — input node embedding dimension.
    /// * `out_features` — output node embedding dimension.
    /// * `hidden`       — hidden dimension of the 2-layer MLP.
    /// * `epsilon`      — ε value added to the self-embedding coefficient.
    ///   Set to 0.0 for an un-weighted self-loop, or learn it via gradient.
    pub fn new(in_features: usize, out_features: usize, hidden: usize, epsilon: f32) -> Self {
        let hidden = hidden.max(1);
        let mlp_w1 = xavier_init(in_features, hidden, 0);
        let mlp_b1 = vec![0.0_f32; hidden];
        let mlp_w2 = xavier_init(hidden, out_features, 1);
        let mlp_b2 = vec![0.0_f32; out_features];
        GINLayer {
            in_features,
            out_features,
            epsilon,
            mlp_w1,
            mlp_b1,
            mlp_w2,
            mlp_b2,
            hidden,
        }
    }

    /// Forward pass.
    ///
    /// 1. For each node *v*, compute neighbourhood sum: `agg_v = Σ_{u ∈ N(v)} h_u`.
    /// 2. Compute input to MLP: `x_v = (1 + ε) h_v + agg_v`.
    /// 3. Apply 2-layer MLP: `h_v' = MLP(x_v)`.
    pub fn forward(&self, graph: &Graph, h: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        let n = graph.num_nodes;
        if h.len() != n {
            return Err(NeuralError::InvalidArgument(format!(
                "h.len() ({}) must equal graph.num_nodes ({})",
                h.len(),
                n
            )));
        }
        for (i, row) in h.iter().enumerate() {
            if row.len() != self.in_features {
                return Err(NeuralError::DimensionMismatch(format!(
                    "h[{i}].len() ({}) != in_features ({})",
                    row.len(),
                    self.in_features
                )));
            }
        }

        // Step 1: neighbourhood sum aggregation
        let mut agg: Vec<Vec<f32>> = vec![vec![0.0_f32; self.in_features]; n];
        for (i, agg_i) in agg.iter_mut().enumerate().take(n) {
            for &j in &graph.neighbors(i) {
                for (f, agg_if) in agg_i.iter_mut().enumerate().take(self.in_features) {
                    *agg_if += h[j][f];
                }
            }
        }

        // Step 2 + 3: MLP on (1+ε)h_v + agg_v
        let out: Vec<Vec<f32>> = (0..n)
            .map(|i| {
                // Combine self-feature and aggregation
                let x: Vec<f32> = (0..self.in_features)
                    .map(|f| (1.0 + self.epsilon) * h[i][f] + agg[i][f])
                    .collect();

                // First MLP layer + ReLU
                let hidden_out: Vec<f32> = (0..self.hidden)
                    .map(|d| {
                        let mut val = self.mlp_b1[d];
                        for (f, &xf) in x.iter().enumerate().take(self.in_features) {
                            val += xf * self.mlp_w1[f][d];
                        }
                        val.max(0.0) // ReLU
                    })
                    .collect();

                // Second MLP layer (no activation — caller may apply one)
                (0..self.out_features)
                    .map(|o| {
                        let mut val = self.mlp_b2[o];
                        for (d, &hd) in hidden_out.iter().enumerate().take(self.hidden) {
                            val += hd * self.mlp_w2[d][o];
                        }
                        val
                    })
                    .collect()
            })
            .collect();

        Ok(out)
    }

    /// Apply the MLP to a single input vector (without graph context).
    /// Useful for testing the MLP sub-network independently.
    pub fn mlp_forward(&self, x: &[f32]) -> Result<Vec<f32>> {
        if x.len() != self.in_features {
            return Err(NeuralError::DimensionMismatch(format!(
                "x.len() ({}) != in_features ({})",
                x.len(),
                self.in_features
            )));
        }
        let hidden_out: Vec<f32> = (0..self.hidden)
            .map(|d| {
                let mut val = self.mlp_b1[d];
                for (f, &xf) in x.iter().enumerate().take(self.in_features) {
                    val += xf * self.mlp_w1[f][d];
                }
                val.max(0.0)
            })
            .collect();
        let out: Vec<f32> = (0..self.out_features)
            .map(|o| {
                let mut val = self.mlp_b2[o];
                for (d, &hd) in hidden_out.iter().enumerate().take(self.hidden) {
                    val += hd * self.mlp_w2[d][o];
                }
                val
            })
            .collect();
        Ok(out)
    }

    /// Total number of trainable parameters.
    pub fn num_parameters(&self) -> usize {
        let w1 = self.in_features * self.hidden;
        let b1 = self.hidden;
        let w2 = self.hidden * self.out_features;
        let b2 = self.out_features;
        w1 + b1 + w2 + b2
    }

    pub fn in_features(&self) -> usize {
        self.in_features
    }

    pub fn out_features(&self) -> usize {
        self.out_features
    }

    pub fn epsilon(&self) -> f32 {
        self.epsilon
    }

    /// Update ε (e.g. after a gradient step).
    pub fn set_epsilon(&mut self, eps: f32) {
        self.epsilon = eps;
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn chain_graph(n: usize, fdim: usize, val: f32) -> Graph {
        let mut g = Graph::new(n, fdim);
        for i in 0..n.saturating_sub(1) {
            g.add_undirected_edge(i, i + 1).expect("edge ok");
        }
        for i in 0..n {
            g.set_node_features(i, vec![val; fdim]).expect("feat ok");
        }
        g
    }

    #[test]
    fn test_gin_output_shape() {
        let g = chain_graph(5, 4, 0.5);
        let layer = GINLayer::new(4, 8, 16, 0.0);
        let out = layer.forward(&g, &g.node_features).expect("forward ok");
        assert_eq!(out.len(), 5);
        assert_eq!(out[0].len(), 8);
    }

    #[test]
    fn test_gin_all_finite() {
        let g = chain_graph(4, 3, 1.0);
        let layer = GINLayer::new(3, 6, 12, 0.5);
        let out = layer.forward(&g, &g.node_features).expect("forward");
        assert!(out.iter().flat_map(|r| r.iter()).all(|v| v.is_finite()));
    }

    #[test]
    fn test_gin_permutation_invariance() {
        // Two isomorphic graphs (same structure, nodes relabelled) should
        // produce the same **multiset** of node embeddings after one GIN layer
        // if all initial node features are identical.
        let g1 = chain_graph(3, 2, 1.0);

        // Permuted: same adjacency but nodes 0 ↔ 2
        let mut g2 = Graph::new(3, 2);
        g2.add_undirected_edge(2, 1).expect("ok"); // was 0--1
        g2.add_undirected_edge(1, 0).expect("ok"); // was 1--2
        for i in 0..3 {
            g2.set_node_features(i, vec![1.0; 2]).expect("ok");
        }

        let layer = GINLayer::new(2, 4, 8, 0.0);
        let out1 = layer.forward(&g1, &g1.node_features).expect("g1 forward");
        let out2 = layer.forward(&g2, &g2.node_features).expect("g2 forward");

        // Both graphs have one centre node with 2 neighbours and two leaf nodes
        // with 1 neighbour.  Collect sorted norms to compare as multisets.
        let mut norms1: Vec<f32> = out1
            .iter()
            .map(|r| r.iter().map(|&v| v * v).sum::<f32>().sqrt())
            .collect();
        let mut norms2: Vec<f32> = out2
            .iter()
            .map(|r| r.iter().map(|&v| v * v).sum::<f32>().sqrt())
            .collect();
        norms1.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        norms2.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        for (a, b) in norms1.iter().zip(norms2.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "Permutation invariance violated: {a} vs {b}"
            );
        }
    }

    #[test]
    fn test_gin_epsilon_scaling() {
        // With epsilon = 1.0 and isolated nodes (no neighbours), the MLP input
        // is (1 + ε) * h_v.  We use negative features so that the xavier-init
        // weights (which are all negative) produce positive pre-activations and
        // survive the ReLU, making the outputs non-zero and ε-sensitive.
        let mut g = Graph::new(2, 2); // no edges
        g.set_node_features(0, vec![-1.0, -1.0]).expect("ok");
        g.set_node_features(1, vec![-1.0, -0.5]).expect("ok");

        let layer_eps0 = GINLayer::new(2, 2, 4, 0.0);
        let layer_eps1 = GINLayer::new(2, 2, 4, 1.0);

        // Both layers have identical weights but different ε.  The MLP input
        // for eps=0 is 1·h and for eps=1 it is 2·h, so after the MLP the
        // outputs must differ.
        let out0 = layer_eps0
            .forward(&g, &g.node_features)
            .expect("eps0 forward");
        let out1 = layer_eps1
            .forward(&g, &g.node_features)
            .expect("eps1 forward");

        // The outputs should differ (ε=1 scales input by 2×)
        let differs = out0.iter().zip(out1.iter()).any(|(r0, r1)| {
            r0.iter()
                .zip(r1.iter())
                .any(|(&a, &b)| (a - b).abs() > 1e-6)
        });
        assert!(
            differs,
            "Different ε values should produce different outputs"
        );
    }

    #[test]
    fn test_gin_num_parameters() {
        let layer = GINLayer::new(4, 8, 16, 0.0);
        // W1: 4*16=64, b1: 16, W2: 16*8=128, b2: 8  →  216
        assert_eq!(layer.num_parameters(), 4 * 16 + 16 + 16 * 8 + 8);
    }

    #[test]
    fn test_gin_mlp_forward() {
        let layer = GINLayer::new(3, 5, 8, 0.0);
        let x = vec![1.0_f32, 0.5, -0.3];
        let out = layer.mlp_forward(&x).expect("mlp ok");
        assert_eq!(out.len(), 5);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_gin_dimension_mismatch_error() {
        let g = chain_graph(3, 4, 1.0);
        let layer = GINLayer::new(2, 4, 8, 0.0); // expects 2, graph has 4
        let result = layer.forward(&g, &g.node_features);
        assert!(result.is_err());
    }
}
