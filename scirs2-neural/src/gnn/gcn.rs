//! Graph Convolutional Network layer (Kipf & Welling, 2017).
//!
//! Implements the spectral graph convolution:
//!
//! ```text
//! H' = σ( D̃^{-½} Ã D̃^{-½} H W + b )
//! ```
//!
//! where `Ã = A + I` (self-loops added) and `D̃` is the corresponding degree
//! matrix.  The normalised adjacency is pre-computed from the supplied `Graph`
//! at each forward pass, so the layer itself only stores trainable parameters.

use crate::error::{NeuralError, Result};
use crate::gnn::graph::Graph;

// ──────────────────────────────────────────────────────────────────────────────
// Activation
// ──────────────────────────────────────────────────────────────────────────────

/// Activation function applied after the linear transform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    /// Rectified Linear Unit: max(0, x)
    ReLU,
    /// Sigmoid: 1 / (1 + exp(−x))
    Sigmoid,
    /// Hyperbolic tangent
    Tanh,
    /// Identity (no activation)
    None,
}

impl Activation {
    fn apply(&self, x: f32) -> f32 {
        match self {
            Activation::ReLU => x.max(0.0),
            Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Activation::Tanh => x.tanh(),
            Activation::None => x,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Xavier init helper
// ──────────────────────────────────────────────────────────────────────────────

/// Xavier / Glorot uniform initialisation.
///
/// Fills a `fan_in × fan_out` matrix with values drawn from
/// U(−limit, limit) where `limit = sqrt(6 / (fan_in + fan_out))`.
fn xavier_init(fan_in: usize, fan_out: usize, seed_offset: u64) -> Vec<Vec<f32>> {
    let limit = (6.0_f64 / (fan_in + fan_out) as f64).sqrt() as f32;
    // Simple LCG PRNG seeded deterministically so tests are reproducible.
    let mut state: u64 = 12345678901234567_u64.wrapping_add(seed_offset);
    let lcg_next = |s: &mut u64| -> f32 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bits = ((*s >> 33) as u32) as f64 / u32::MAX as f64;
        (bits as f32) * 2.0 * limit - limit
    };
    (0..fan_in)
        .map(|_| (0..fan_out).map(|_| lcg_next(&mut state)).collect())
        .collect()
}

// ──────────────────────────────────────────────────────────────────────────────
// GCNLayer
// ──────────────────────────────────────────────────────────────────────────────

/// A single Graph Convolutional Network (GCN) layer.
///
/// # Example
/// ```rust
/// use scirs2_neural::gnn::gcn::{GCNLayer, Activation};
/// use scirs2_neural::gnn::graph::Graph;
///
/// let mut g = Graph::new(3, 4);
/// g.add_undirected_edge(0, 1).expect("operation should succeed");
/// g.add_undirected_edge(1, 2).expect("operation should succeed");
///
/// // Assign random features
/// for i in 0..3 {
///     g.set_node_features(i, vec![1.0; 4]).expect("operation should succeed");
/// }
///
/// let layer = GCNLayer::new(4, 8, Activation::ReLU);
/// let out = layer.forward(&g, &g.node_features).expect("forward ok");
/// assert_eq!(out.len(), 3);
/// assert_eq!(out[0].len(), 8);
/// ```
#[derive(Debug, Clone)]
pub struct GCNLayer {
    in_features: usize,
    out_features: usize,
    /// Weight matrix W of shape `[in_features, out_features]`.
    weights: Vec<Vec<f32>>,
    /// Bias vector of length `out_features`.
    bias: Vec<f32>,
    use_bias: bool,
    activation: Activation,
}

impl GCNLayer {
    /// Create a new `GCNLayer` with Xavier-initialised weights and zero bias.
    pub fn new(in_features: usize, out_features: usize, activation: Activation) -> Self {
        let weights = xavier_init(in_features, out_features, 0);
        let bias = vec![0.0_f32; out_features];
        GCNLayer {
            in_features,
            out_features,
            weights,
            bias,
            use_bias: true,
            activation,
        }
    }

    /// Create a `GCNLayer` without a bias term.
    pub fn new_no_bias(in_features: usize, out_features: usize, activation: Activation) -> Self {
        let mut layer = Self::new(in_features, out_features, activation);
        layer.use_bias = false;
        layer
    }

    /// Forward pass.
    ///
    /// Computes `H' = σ(Â H W + b)` where `Â = D̃^{-½}(A+I)D̃^{-½}`.
    ///
    /// # Arguments
    /// * `graph` — graph topology (adjacency used to derive normalised Â).
    /// * `h`     — node feature matrix `[N, in_features]`.
    ///
    /// # Returns
    /// Output feature matrix `[N, out_features]`.
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

        // Step 1: normalised adjacency Â = D̃^{-½}(A+I)D̃^{-½}
        let a_hat = graph.normalized_adjacency();

        // Step 2: Â @ H  →  [N, in_features]
        let mut agg: Vec<Vec<f32>> = vec![vec![0.0_f32; self.in_features]; n];
        for i in 0..n {
            for j in 0..n {
                let a_ij = a_hat[i][j];
                if a_ij.abs() < 1e-12 {
                    continue;
                }
                for f in 0..self.in_features {
                    agg[i][f] += a_ij * h[j][f];
                }
            }
        }

        // Step 3: (Â H) @ W + b, then apply activation
        let mut out: Vec<Vec<f32>> = vec![vec![0.0_f32; self.out_features]; n];
        for i in 0..n {
            for (o, out_io) in out[i].iter_mut().enumerate().take(self.out_features) {
                let mut val = if self.use_bias { self.bias[o] } else { 0.0 };
                for (f, &agg_if) in agg[i].iter().enumerate().take(self.in_features) {
                    val += agg_if * self.weights[f][o];
                }
                *out_io = self.activation.apply(val);
            }
        }
        Ok(out)
    }

    /// Return a reference to the weight matrix `[in_features, out_features]`.
    pub fn weights(&self) -> &Vec<Vec<f32>> {
        &self.weights
    }

    /// Replace the weight matrix.
    ///
    /// # Errors
    /// Returns an error if the shape does not match `[in_features, out_features]`.
    pub fn set_weights(&mut self, w: Vec<Vec<f32>>) -> Result<()> {
        if w.len() != self.in_features || w.iter().any(|r| r.len() != self.out_features) {
            return Err(NeuralError::DimensionMismatch(format!(
                "Expected weight shape [{}, {}]",
                self.in_features, self.out_features
            )));
        }
        self.weights = w;
        Ok(())
    }

    /// Return a reference to the bias vector.
    pub fn bias(&self) -> &Vec<f32> {
        &self.bias
    }

    /// Set bias vector.
    pub fn set_bias(&mut self, b: Vec<f32>) -> Result<()> {
        if b.len() != self.out_features {
            return Err(NeuralError::DimensionMismatch(format!(
                "Expected bias length {}, got {}",
                self.out_features,
                b.len()
            )));
        }
        self.bias = b;
        Ok(())
    }

    /// Number of trainable parameters.
    pub fn num_parameters(&self) -> usize {
        let w_params = self.in_features * self.out_features;
        let b_params = if self.use_bias { self.out_features } else { 0 };
        w_params + b_params
    }

    pub fn in_features(&self) -> usize {
        self.in_features
    }

    pub fn out_features(&self) -> usize {
        self.out_features
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn small_chain_graph(n: usize, feat_dim: usize) -> Graph {
        let mut g = Graph::new(n, feat_dim);
        for i in 0..n.saturating_sub(1) {
            g.add_undirected_edge(i, i + 1).expect("edge ok");
        }
        for i in 0..n {
            g.set_node_features(i, vec![1.0_f32; feat_dim])
                .expect("feat ok");
        }
        g
    }

    #[test]
    fn test_gcn_output_shape() {
        let g = small_chain_graph(5, 4);
        let layer = GCNLayer::new(4, 8, Activation::ReLU);
        let out = layer.forward(&g, &g.node_features).expect("forward ok");
        assert_eq!(out.len(), 5);
        assert_eq!(out[0].len(), 8);
    }

    #[test]
    fn test_gcn_all_finite() {
        let g = small_chain_graph(4, 3);
        let layer = GCNLayer::new(3, 6, Activation::ReLU);
        let out = layer.forward(&g, &g.node_features).expect("forward");
        assert!(out.iter().flat_map(|r| r.iter()).all(|v| v.is_finite()));
    }

    #[test]
    fn test_gcn_relu_non_negative() {
        let g = small_chain_graph(3, 2);
        let layer = GCNLayer::new(2, 4, Activation::ReLU);
        let out = layer.forward(&g, &g.node_features).expect("forward");
        assert!(
            out.iter().flat_map(|r| r.iter()).all(|&v| v >= 0.0),
            "ReLU output should be non-negative"
        );
    }

    #[test]
    fn test_gcn_no_activation() {
        let g = small_chain_graph(3, 2);
        let layer = GCNLayer::new(2, 2, Activation::None);
        let out = layer.forward(&g, &g.node_features).expect("forward");
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn test_gcn_wrong_feature_dim_error() {
        let g = small_chain_graph(3, 4);
        let layer = GCNLayer::new(2, 4, Activation::ReLU); // expects 2, graph has 4
        let result = layer.forward(&g, &g.node_features);
        assert!(result.is_err());
    }

    #[test]
    fn test_gcn_single_node() {
        let mut g = Graph::new(1, 3);
        g.set_node_features(0, vec![0.5, -0.3, 1.2]).expect("ok");
        let layer = GCNLayer::new(3, 5, Activation::Sigmoid);
        let out = layer.forward(&g, &g.node_features).expect("forward");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].len(), 5);
        // All sigmoid outputs must be in (0, 1)
        assert!(out[0].iter().all(|&v| v > 0.0 && v < 1.0));
    }

    #[test]
    fn test_gcn_set_weights() {
        let mut layer = GCNLayer::new(2, 3, Activation::None);
        let new_w = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        layer.set_weights(new_w.clone()).expect("set_weights ok");
        assert_eq!(layer.weights(), &new_w);
    }

    #[test]
    fn test_gcn_num_parameters() {
        let layer = GCNLayer::new(4, 8, Activation::ReLU);
        assert_eq!(layer.num_parameters(), 4 * 8 + 8);
    }
}
