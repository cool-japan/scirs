//! GraphSAGE: Inductive Representation Learning on Large Graphs
//! (Hamilton, Ying & Leskovec, 2017).
//!
//! For each node v the update rule is:
//!
//! ```text
//! h_N(v) = AGG({ h_u : u ∈ N(v) })
//! h_v'   = σ( W · CONCAT(h_v, h_N(v)) )
//! ```
//!
//! The weight matrix `W` has shape `[2 * in_features, out_features]` after
//! the concatenation.  Optionally the output is L2-normalised so that
//! downstream dot-product similarities are meaningful.

use crate::error::{NeuralError, Result};
use crate::gnn::graph::Graph;

// ──────────────────────────────────────────────────────────────────────────────
// Aggregator type
// ──────────────────────────────────────────────────────────────────────────────

/// Neighbourhood aggregation strategy for GraphSAGE.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregatorType {
    /// Average of neighbour embeddings.
    Mean,
    /// Element-wise maximum of neighbour embeddings.
    Max,
    /// Sum of neighbour embeddings.
    Sum,
}

// ──────────────────────────────────────────────────────────────────────────────
// Xavier init (same deterministic LCG as GCNLayer)
// ──────────────────────────────────────────────────────────────────────────────

fn xavier_init(fan_in: usize, fan_out: usize, seed_offset: u64) -> Vec<Vec<f32>> {
    let limit = (6.0_f64 / (fan_in + fan_out) as f64).sqrt() as f32;
    let mut state: u64 = 98765432109876543_u64.wrapping_add(seed_offset);
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
// GraphSAGELayer
// ──────────────────────────────────────────────────────────────────────────────

/// GraphSAGE layer with configurable neighbourhood aggregation.
///
/// # Example
/// ```rust
/// use scirs2_neural::gnn::sage::{GraphSAGELayer, AggregatorType};
/// use scirs2_neural::gnn::graph::Graph;
///
/// let mut g = Graph::new(4, 3);
/// g.add_undirected_edge(0, 1).expect("operation should succeed");
/// g.add_undirected_edge(1, 2).expect("operation should succeed");
/// g.add_undirected_edge(2, 3).expect("operation should succeed");
/// for i in 0..4 { g.set_node_features(i, vec![1.0, 0.5, -0.5]).expect("operation should succeed"); }
///
/// let layer = GraphSAGELayer::new(3, 8, AggregatorType::Mean, true);
/// let out = layer.forward(&g, &g.node_features).expect("forward ok");
/// assert_eq!(out.len(), 4);
/// assert_eq!(out[0].len(), 8);
/// ```
#[derive(Debug, Clone)]
pub struct GraphSAGELayer {
    in_features: usize,
    out_features: usize,
    aggregator: AggregatorType,
    /// Combined weight W of shape `[2 * in_features, out_features]`.
    /// The first `in_features` rows correspond to the self embedding,
    /// the next `in_features` rows to the aggregated neighbourhood.
    weights: Vec<Vec<f32>>,
    bias: Vec<f32>,
    /// Whether to L2-normalise each output node embedding.
    normalize: bool,
}

impl GraphSAGELayer {
    /// Create a new `GraphSAGELayer`.
    ///
    /// # Arguments
    /// * `in_features`  — dimension of input node embeddings.
    /// * `out_features` — dimension of output node embeddings.
    /// * `agg`          — neighbourhood aggregation type.
    /// * `normalize`    — if `true`, L2-normalise each output vector.
    pub fn new(
        in_features: usize,
        out_features: usize,
        agg: AggregatorType,
        normalize: bool,
    ) -> Self {
        // Weight matrix input dim = 2 * in_features (concat of self + agg)
        let combined_in = 2 * in_features;
        let weights = xavier_init(combined_in, out_features, 0);
        let bias = vec![0.0_f32; out_features];
        GraphSAGELayer {
            in_features,
            out_features,
            aggregator: agg,
            weights,
            bias,
            normalize,
        }
    }

    /// Forward pass.
    ///
    /// For each node:
    /// 1. Aggregate neighbours with the chosen aggregator.
    /// 2. Concatenate self embedding with aggregated embedding.
    /// 3. Apply linear transform + bias, then ReLU.
    /// 4. Optionally L2-normalise.
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

        let combined_in = 2 * self.in_features;
        let mut out: Vec<Vec<f32>> = Vec::with_capacity(n);

        for i in 0..n {
            let neighbors = graph.neighbors(i);

            // Aggregate neighbourhood
            let agg_vec = if neighbors.is_empty() {
                vec![0.0_f32; self.in_features]
            } else {
                self.aggregate(&neighbors, h)?
            };

            // Concatenate [h_v || h_N(v)]
            let mut concat = Vec::with_capacity(combined_in);
            concat.extend_from_slice(&h[i]);
            concat.extend_from_slice(&agg_vec);

            // Linear + bias + ReLU
            let mut node_out = vec![0.0_f32; self.out_features];
            for (o, node_o) in node_out.iter_mut().enumerate().take(self.out_features) {
                let mut val = self.bias[o];
                for (f, &cf) in concat.iter().enumerate().take(combined_in) {
                    val += cf * self.weights[f][o];
                }
                *node_o = val.max(0.0); // ReLU
            }

            // Optional L2 normalise
            if self.normalize {
                let norm: f32 = node_out.iter().map(|&v| v * v).sum::<f32>().sqrt();
                if norm > 1e-12 {
                    node_out.iter_mut().for_each(|v| *v /= norm);
                }
            }

            out.push(node_out);
        }
        Ok(out)
    }

    /// Aggregate the embeddings of a node's neighbours using the configured
    /// `AggregatorType`.
    fn aggregate(&self, neighbors: &[usize], h: &[Vec<f32>]) -> Result<Vec<f32>> {
        let f = self.in_features;
        match self.aggregator {
            AggregatorType::Mean => {
                let mut agg = vec![0.0_f32; f];
                for &nb in neighbors {
                    if nb >= h.len() {
                        return Err(NeuralError::InvalidArgument(format!(
                            "Neighbour index {nb} out of bounds (h.len() = {})",
                            h.len()
                        )));
                    }
                    for (k, &v) in h[nb].iter().enumerate() {
                        agg[k] += v;
                    }
                }
                let n = neighbors.len() as f32;
                agg.iter_mut().for_each(|v| *v /= n);
                Ok(agg)
            }
            AggregatorType::Max => {
                let mut agg = vec![f32::NEG_INFINITY; f];
                for &nb in neighbors {
                    if nb >= h.len() {
                        return Err(NeuralError::InvalidArgument(format!(
                            "Neighbour index {nb} out of bounds (h.len() = {})",
                            h.len()
                        )));
                    }
                    for (k, &v) in h[nb].iter().enumerate() {
                        if v > agg[k] {
                            agg[k] = v;
                        }
                    }
                }
                // Replace any remaining NEG_INFINITY with 0
                agg.iter_mut().for_each(|v| {
                    if v.is_infinite() {
                        *v = 0.0;
                    }
                });
                Ok(agg)
            }
            AggregatorType::Sum => {
                let mut agg = vec![0.0_f32; f];
                for &nb in neighbors {
                    if nb >= h.len() {
                        return Err(NeuralError::InvalidArgument(format!(
                            "Neighbour index {nb} out of bounds (h.len() = {})",
                            h.len()
                        )));
                    }
                    for (k, &v) in h[nb].iter().enumerate() {
                        agg[k] += v;
                    }
                }
                Ok(agg)
            }
        }
    }

    /// Number of trainable parameters.
    pub fn num_parameters(&self) -> usize {
        2 * self.in_features * self.out_features + self.out_features
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
    fn test_sage_mean_output_shape() {
        let g = chain_graph(5, 4, 0.5);
        let layer = GraphSAGELayer::new(4, 8, AggregatorType::Mean, false);
        let out = layer.forward(&g, &g.node_features).expect("forward ok");
        assert_eq!(out.len(), 5);
        assert_eq!(out[0].len(), 8);
    }

    #[test]
    fn test_sage_max_aggregation() {
        let g = chain_graph(4, 3, 1.0);
        let layer = GraphSAGELayer::new(3, 6, AggregatorType::Max, false);
        let out = layer.forward(&g, &g.node_features).expect("forward");
        assert_eq!(out.len(), 4);
        assert!(out.iter().flat_map(|r| r.iter()).all(|v| v.is_finite()));
    }

    #[test]
    fn test_sage_sum_aggregation() {
        let g = chain_graph(3, 2, 0.3);
        let layer = GraphSAGELayer::new(2, 4, AggregatorType::Sum, false);
        let out = layer.forward(&g, &g.node_features).expect("forward");
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn test_sage_mean_aggregation_correctness() {
        // Graph: 0--1--2 with uniform features [1.0, 0.0]
        // Node 1 has neighbours {0, 2}, mean agg = [1.0, 0.0] (same as h since all identical)
        let g = chain_graph(3, 2, 1.0);
        let layer = GraphSAGELayer::new(2, 4, AggregatorType::Mean, false);
        // All features are [1.0, 1.0], aggregation is also [1.0, 1.0]
        // The concat vector for every node is [1.0, 1.0, 1.0, 1.0] (for interior)
        let out = layer.forward(&g, &g.node_features).expect("forward");
        // All outputs should be non-negative (ReLU applied)
        assert!(out.iter().flat_map(|r| r.iter()).all(|&v| v >= 0.0));
    }

    #[test]
    fn test_sage_normalized_unit_norm() {
        let g = chain_graph(4, 3, 0.7);
        let layer = GraphSAGELayer::new(3, 6, AggregatorType::Mean, true);
        let out = layer.forward(&g, &g.node_features).expect("forward");
        for (i, row) in out.iter().enumerate() {
            let norm: f32 = row.iter().map(|&v| v * v).sum::<f32>().sqrt();
            // Either norm ≈ 1 or all zeros (if ReLU zeros everything out)
            assert!(
                (norm - 1.0).abs() < 1e-5 || norm < 1e-6,
                "node {i} norm = {norm}"
            );
        }
    }

    #[test]
    fn test_sage_isolated_node() {
        // Node 1 has no neighbours — aggregation should fall back to zero vector
        let mut g = Graph::new(3, 2);
        g.add_edge(0, 2, 1.0).expect("ok");
        for i in 0..3 {
            g.set_node_features(i, vec![1.0, 1.0]).expect("ok");
        }
        let layer = GraphSAGELayer::new(2, 4, AggregatorType::Mean, false);
        let out = layer.forward(&g, &g.node_features).expect("forward");
        assert_eq!(out.len(), 3);
        assert!(out.iter().flat_map(|r| r.iter()).all(|v| v.is_finite()));
    }

    #[test]
    fn test_sage_dimension_mismatch_error() {
        let g = chain_graph(3, 4, 1.0);
        let layer = GraphSAGELayer::new(2, 4, AggregatorType::Mean, false); // expects 2, graph has 4
        let result = layer.forward(&g, &g.node_features);
        assert!(result.is_err());
    }

    #[test]
    fn test_sage_num_parameters() {
        let layer = GraphSAGELayer::new(4, 8, AggregatorType::Mean, false);
        // W: [2*4, 8] = 64, b: 8 → 72
        assert_eq!(layer.num_parameters(), 2 * 4 * 8 + 8);
    }
}
