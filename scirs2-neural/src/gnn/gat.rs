//! Graph Attention Network (Veličković et al., 2018).
//!
//! Multi-head self-attention over the graph neighbourhood:
//!
//! ```text
//! e_{ij}   = LeakyReLU( a^T [ W h_i || W h_j ] )
//! α_{ij}   = softmax_j( e_{ij} )
//! h_v'     = σ( Σ_{j ∈ N(v) ∪ {v}} α_{vj} W h_j )
//! ```
//!
//! Multiple attention heads are either concatenated (for hidden layers) or
//! averaged (for the final output layer).

use crate::error::{NeuralError, Result};
use crate::gnn::graph::Graph;

// ──────────────────────────────────────────────────────────────────────────────
// Xavier + LCG helpers
// ──────────────────────────────────────────────────────────────────────────────

fn lcg_rand(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 33) as u32) as f32 / u32::MAX as f32
}

fn xavier_matrix(fan_in: usize, fan_out: usize, state: &mut u64) -> Vec<Vec<f32>> {
    let limit = (6.0_f64 / (fan_in + fan_out) as f64).sqrt() as f32;
    (0..fan_in)
        .map(|_| {
            (0..fan_out)
                .map(|_| lcg_rand(state) * 2.0 * limit - limit)
                .collect()
        })
        .collect()
}

fn xavier_vec(len: usize, state: &mut u64) -> Vec<f32> {
    let limit = (6.0_f64 / (len * 2) as f64).sqrt() as f32;
    (0..len)
        .map(|_| lcg_rand(state) * 2.0 * limit - limit)
        .collect()
}

// ──────────────────────────────────────────────────────────────────────────────
// GATLayer
// ──────────────────────────────────────────────────────────────────────────────

/// Graph Attention Network layer with multi-head attention.
///
/// # Example
/// ```rust
/// use scirs2_neural::gnn::gat::GATLayer;
/// use scirs2_neural::gnn::graph::Graph;
///
/// let mut g = Graph::new(4, 8);
/// g.add_undirected_edge(0, 1).expect("operation should succeed");
/// g.add_undirected_edge(1, 2).expect("operation should succeed");
/// g.add_undirected_edge(2, 3).expect("operation should succeed");
/// for i in 0..4 { g.set_node_features(i, vec![1.0; 8]).expect("operation should succeed"); }
///
/// let layer = GATLayer::new(8, 16, 4); // in=8, out=16, heads=4
/// let out = layer.forward(&g, &g.node_features).expect("forward ok");
/// assert_eq!(out.len(), 4);
/// assert_eq!(out[0].len(), 16);
/// ```
#[derive(Debug, Clone)]
pub struct GATLayer {
    in_features: usize,
    out_features: usize,
    num_heads: usize,
    /// Per-head output dimension.  When concatenating heads:
    /// `head_dim * num_heads == out_features`.
    head_dim: usize,
    /// Per-head linear projection `[num_heads][in_features][head_dim]`.
    w: Vec<Vec<Vec<f32>>>,
    /// Per-head attention vector `[num_heads][2 * head_dim]`.
    a: Vec<Vec<f32>>,
    /// LeakyReLU negative slope.
    negative_slope: f32,
    /// If `true`, concatenate head outputs; if `false`, average them.
    concat: bool,
    /// Bias term of length `out_features`.
    bias: Vec<f32>,
}

impl GATLayer {
    /// Create a new `GATLayer`.
    ///
    /// # Arguments
    /// * `in_features`  — input feature dimension.
    /// * `out_features` — output feature dimension (must be divisible by
    ///   `num_heads` when `concat == true`).
    /// * `num_heads`    — number of attention heads.
    ///
    /// # Panics
    /// This constructor uses `num_heads = 1` if `out_features < num_heads`.
    pub fn new(in_features: usize, out_features: usize, num_heads: usize) -> Self {
        let num_heads = num_heads.max(1);
        let head_dim = (out_features / num_heads).max(1);
        let mut state: u64 = 11223344556677889_u64;

        let w: Vec<Vec<Vec<f32>>> = (0..num_heads)
            .map(|_| xavier_matrix(in_features, head_dim, &mut state))
            .collect();

        let a: Vec<Vec<f32>> = (0..num_heads)
            .map(|_| xavier_vec(2 * head_dim, &mut state))
            .collect();

        let actual_out = head_dim * num_heads; // may differ from out_features
        let bias = vec![0.0_f32; actual_out];

        GATLayer {
            in_features,
            out_features: actual_out,
            num_heads,
            head_dim,
            w,
            a,
            negative_slope: 0.2,
            concat: true,
            bias,
        }
    }

    /// Create a `GATLayer` that averages head outputs (suitable for final layer).
    pub fn new_averaging(in_features: usize, out_features: usize, num_heads: usize) -> Self {
        let mut layer = Self::new(in_features, out_features * num_heads, num_heads);
        layer.concat = false;
        layer.out_features = out_features;
        layer.bias = vec![0.0_f32; out_features];
        layer
    }

    /// Set the LeakyReLU negative slope (default 0.2).
    pub fn with_negative_slope(mut self, slope: f32) -> Result<Self> {
        if !(0.0..1.0).contains(&slope) {
            return Err(NeuralError::InvalidArgument(format!(
                "negative_slope must be in [0, 1), got {slope}"
            )));
        }
        self.negative_slope = slope;
        Ok(self)
    }

    /// Forward pass.
    ///
    /// For each attention head:
    /// 1. Project node features: `Wh[i] = W * h[i]`  `[N, head_dim]`
    /// 2. Compute unnormalised attention: `e[i][j] = LeakyReLU(a^T [Wh_i || Wh_j])`
    ///    restricted to `j ∈ N(i) ∪ {i}`.
    /// 3. Normalise with softmax over the neighbourhood.
    /// 4. Compute weighted sum: `out_head[i] = Σ_j α[i][j] * Wh[j]`.
    ///
    /// Heads are concatenated or averaged based on `self.concat`.
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

        // Gather per-head outputs
        // head_outputs[head] = [N, head_dim]
        let mut head_outputs: Vec<Vec<Vec<f32>>> = Vec::with_capacity(self.num_heads);

        for k in 0..self.num_heads {
            // Step 1: project all nodes  Wh[i] = W_k h[i]
            let wh: Vec<Vec<f32>> = (0..n).map(|i| self.project_node(&h[i], k)).collect();

            // Step 2 + 3 + 4: attention + aggregate
            let head_out: Vec<Vec<f32>> = (0..n)
                .map(|i| {
                    // Neighbourhood = neighbours ∪ {self}
                    let mut nb = graph.neighbors(i);
                    if !nb.contains(&i) {
                        nb.push(i);
                    }

                    // Compute raw attention scores e_{i,j}
                    let raw: Vec<f32> = nb
                        .iter()
                        .map(|&j| {
                            let eij = self.attention_score(&wh[i], &wh[j], k);
                            self.leaky_relu(eij)
                        })
                        .collect();

                    // Softmax normalisation
                    let alpha = self.softmax_vec(&raw);

                    // Weighted aggregation
                    let mut agg = vec![0.0_f32; self.head_dim];
                    for (idx, &j) in nb.iter().enumerate() {
                        for d in 0..self.head_dim {
                            agg[d] += alpha[idx] * wh[j][d];
                        }
                    }
                    agg
                })
                .collect();

            head_outputs.push(head_out);
        }

        // Combine heads
        let out_dim = self.out_features;
        let mut out: Vec<Vec<f32>> = vec![vec![0.0_f32; out_dim]; n];

        if self.concat {
            for i in 0..n {
                for (k, head_out_k) in head_outputs.iter().enumerate().take(self.num_heads) {
                    let start = k * self.head_dim;
                    for d in 0..self.head_dim {
                        out[i][start + d] = head_out_k[i][d] + self.bias[start + d];
                    }
                }
            }
        } else {
            // Average heads
            let scale = 1.0 / self.num_heads as f32;
            for i in 0..n {
                for head_out_k in head_outputs.iter().take(self.num_heads) {
                    for d in 0..self.head_dim.min(out_dim) {
                        out[i][d] += head_out_k[i][d] * scale;
                    }
                }
                for (d, od) in out[i].iter_mut().enumerate().take(out_dim) {
                    *od += self.bias[d];
                }
            }
        }

        Ok(out)
    }

    /// Project node feature `h_i` using head-`k` weight matrix.
    fn project_node(&self, hi: &[f32], k: usize) -> Vec<f32> {
        let mut wh = vec![0.0_f32; self.head_dim];
        for (f, &hf) in hi.iter().enumerate() {
            for (d, wh_d) in wh.iter_mut().enumerate().take(self.head_dim) {
                *wh_d += hf * self.w[k][f][d];
            }
        }
        wh
    }

    /// Compute the unnormalised attention logit
    /// `e = a_k^T [Wh_i || Wh_j]` (before LeakyReLU).
    fn attention_score(&self, whi: &[f32], whj: &[f32], k: usize) -> f32 {
        let mut score = 0.0_f32;
        for (d, &whi_d) in whi.iter().enumerate().take(self.head_dim) {
            score += self.a[k][d] * whi_d;
        }
        for (d, &whj_d) in whj.iter().enumerate().take(self.head_dim) {
            score += self.a[k][self.head_dim + d] * whj_d;
        }
        score
    }

    fn leaky_relu(&self, x: f32) -> f32 {
        if x >= 0.0 {
            x
        } else {
            self.negative_slope * x
        }
    }

    fn softmax_vec(&self, vals: &[f32]) -> Vec<f32> {
        if vals.is_empty() {
            return Vec::new();
        }
        let max_val = vals.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_vals: Vec<f32> = vals.iter().map(|&v| (v - max_val).exp()).collect();
        let sum: f32 = exp_vals.iter().sum();
        if sum < 1e-12 {
            vec![1.0 / vals.len() as f32; vals.len()]
        } else {
            exp_vals.iter().map(|&e| e / sum).collect()
        }
    }

    /// Total number of trainable parameters.
    pub fn num_parameters(&self) -> usize {
        let w_params = self.num_heads * self.in_features * self.head_dim;
        let a_params = self.num_heads * 2 * self.head_dim;
        let b_params = self.out_features;
        w_params + a_params + b_params
    }

    pub fn in_features(&self) -> usize {
        self.in_features
    }

    pub fn out_features(&self) -> usize {
        self.out_features
    }

    pub fn num_heads(&self) -> usize {
        self.num_heads
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_graph(n: usize, fdim: usize) -> Graph {
        let mut g = Graph::new(n, fdim);
        for i in 0..n.saturating_sub(1) {
            g.add_undirected_edge(i, i + 1).expect("edge ok");
        }
        for i in 0..n {
            g.set_node_features(i, vec![0.5_f32; fdim])
                .expect("feat ok");
        }
        g
    }

    #[test]
    fn test_gat_output_shape_concat() {
        let g = make_graph(4, 8);
        let layer = GATLayer::new(8, 16, 4); // 4 heads × 4 dims = 16
        let out = layer.forward(&g, &g.node_features).expect("forward ok");
        assert_eq!(out.len(), 4);
        assert_eq!(out[0].len(), 16);
    }

    #[test]
    fn test_gat_all_finite() {
        let g = make_graph(5, 4);
        let layer = GATLayer::new(4, 8, 2);
        let out = layer.forward(&g, &g.node_features).expect("forward");
        assert!(out.iter().flat_map(|r| r.iter()).all(|v| v.is_finite()));
    }

    #[test]
    fn test_gat_attention_weights_sum_to_one() {
        // Verify that softmax normalisation produces valid probability distributions
        let layer = GATLayer::new(2, 4, 1);
        let vals = vec![1.0_f32, 2.0, 3.0];
        let alpha = layer.softmax_vec(&vals);
        let sum: f32 = alpha.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax sum = {sum}");
        assert!(alpha.iter().all(|&v| (0.0..=1.0).contains(&v)));
    }

    #[test]
    fn test_gat_single_node() {
        let mut g = Graph::new(1, 4);
        g.set_node_features(0, vec![1.0; 4]).expect("ok");
        let layer = GATLayer::new(4, 8, 2);
        let out = layer.forward(&g, &g.node_features).expect("forward");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].len(), 8);
    }

    #[test]
    fn test_gat_negative_slope_validation() {
        let layer = GATLayer::new(4, 8, 2);
        assert!(layer.clone().with_negative_slope(0.2).is_ok());
        assert!(layer.clone().with_negative_slope(-0.1).is_err());
        assert!(layer.with_negative_slope(1.5).is_err());
    }

    #[test]
    fn test_gat_leaky_relu() {
        let layer = GATLayer::new(2, 4, 1);
        assert!((layer.leaky_relu(1.0) - 1.0).abs() < 1e-6);
        assert!((layer.leaky_relu(-1.0) - (-0.2)).abs() < 1e-6);
        assert!((layer.leaky_relu(0.0)).abs() < 1e-6);
    }

    #[test]
    fn test_gat_num_parameters() {
        // in=4, out=8, heads=2  →  head_dim=4
        // W: 2 * 4*4 = 32, a: 2 * 2*4 = 16, bias: 8  →  56
        let layer = GATLayer::new(4, 8, 2);
        assert_eq!(layer.num_parameters(), 2 * 4 * 4 + 2 * 2 * 4 + 8);
    }

    #[test]
    fn test_gat_dimension_mismatch_error() {
        let g = make_graph(3, 8);
        let layer = GATLayer::new(4, 8, 2); // expects in=4, graph has 8
        let result = layer.forward(&g, &g.node_features);
        assert!(result.is_err());
    }
}
