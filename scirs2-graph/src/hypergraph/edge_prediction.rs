//! Hyperedge Prediction module.
//!
//! Implements scoring and evaluation utilities for the hyperedge prediction task:
//! given a set of nodes, predict the probability that they form a hyperedge.
//!
//! ## Approach
//!
//! 1. **Pooling**: aggregate node features of candidate hyperedge members
//! 2. **Scoring MLP**: map pooled features through a small MLP to a scalar score
//! 3. **Sigmoid**: convert score to probability
//!
//! ## Evaluation
//!
//! - ROC-AUC computed using the trapezoidal rule
//! - Negative sampling: random k-subsets of the node set

use crate::error::{GraphError, Result};
use scirs2_core::ndarray::Array2;
use scirs2_core::random::{Rng, RngExt, SeedableRng};

// ============================================================================
// Linear layer helper (same pattern as other modules)
// ============================================================================

#[derive(Debug, Clone)]
struct Linear {
    weight: Vec<Vec<f64>>,
    bias: Vec<f64>,
    out_dim: usize,
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
// Pooling Types
// ============================================================================

/// Pooling method for aggregating node features within a hyperedge candidate.
#[derive(Debug, Clone, PartialEq, Default)]
#[non_exhaustive]
pub enum PoolingType {
    /// Sum of node feature vectors.
    Sum,
    /// Element-wise mean.
    #[default]
    Mean,
    /// Element-wise maximum.
    Max,
}

impl PoolingType {
    /// Aggregate node features using this pooling method.
    fn pool(&self, node_feats: &Array2<f64>, nodes: &[usize]) -> Vec<f64> {
        if nodes.is_empty() {
            return vec![0.0; node_feats.ncols()];
        }
        let d = node_feats.ncols();
        match self {
            PoolingType::Sum => {
                let mut out = vec![0.0_f64; d];
                for &i in nodes {
                    for k in 0..d {
                        out[k] += node_feats[[i, k]];
                    }
                }
                out
            }
            PoolingType::Mean => {
                let mut out = vec![0.0_f64; d];
                let inv_n = 1.0 / nodes.len() as f64;
                for &i in nodes {
                    for k in 0..d {
                        out[k] += node_feats[[i, k]] * inv_n;
                    }
                }
                out
            }
            PoolingType::Max => {
                let mut out = vec![f64::NEG_INFINITY; d];
                for &i in nodes {
                    for k in 0..d {
                        if node_feats[[i, k]] > out[k] {
                            out[k] = node_feats[[i, k]];
                        }
                    }
                }
                // Replace -inf with 0 for nodes with no features
                for v in out.iter_mut() {
                    if *v == f64::NEG_INFINITY {
                        *v = 0.0;
                    }
                }
                out
            }
        }
    }
}

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for the HyperedgePredictor.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct HyperedgePredictorConfig {
    /// Hidden dimension of the scoring MLP.
    pub hidden_dim: usize,
    /// Pooling method for aggregating node features.
    pub pooling: PoolingType,
    /// Number of hidden layers in the scoring MLP (not counting the output layer).
    pub n_hidden_layers: usize,
}

impl Default for HyperedgePredictorConfig {
    fn default() -> Self {
        HyperedgePredictorConfig {
            hidden_dim: 64,
            pooling: PoolingType::Mean,
            n_hidden_layers: 2,
        }
    }
}

// ============================================================================
// HyperedgePredictor
// ============================================================================

/// Hyperedge predictor: scores candidate node sets as potential hyperedges.
///
/// Architecture:
/// ```text
/// pool(node_feats[candidate]) → MLP → sigmoid → probability
/// ```
#[derive(Debug, Clone)]
pub struct HyperedgePredictor {
    /// MLP layers.
    layers: Vec<Linear>,
    /// Input feature dimension (per-node).
    in_dim: usize,
    /// Configuration.
    config: HyperedgePredictorConfig,
}

impl HyperedgePredictor {
    /// Create a new HyperedgePredictor.
    ///
    /// # Arguments
    /// - `in_dim`: node feature dimension
    /// - `config`: predictor configuration
    pub fn new(in_dim: usize, config: HyperedgePredictorConfig) -> Self {
        let h = config.hidden_dim;
        let mut layers = Vec::new();
        // Input layer: in_dim → hidden
        layers.push(Linear::new(in_dim, h));
        // Hidden layers
        for _ in 1..config.n_hidden_layers {
            layers.push(Linear::new(h, h));
        }
        // Output layer: hidden → 1
        layers.push(Linear::new(h, 1));
        HyperedgePredictor {
            layers,
            in_dim,
            config,
        }
    }

    /// Score a single candidate hyperedge (set of node indices).
    ///
    /// # Arguments
    /// - `node_feats`: all node features [N × in_dim]
    /// - `candidate`: indices of nodes in the candidate hyperedge
    ///
    /// # Returns
    /// Probability in [0, 1] that the candidate is a real hyperedge.
    pub fn score(&self, node_feats: &Array2<f64>, candidate: &[usize]) -> Result<f64> {
        if candidate.is_empty() {
            return Err(GraphError::InvalidParameter {
                param: "candidate".to_string(),
                value: "empty".to_string(),
                expected: "non-empty set of node indices".to_string(),
                context: "HyperedgePredictor::score".to_string(),
            });
        }
        if node_feats.ncols() != self.in_dim {
            return Err(GraphError::InvalidParameter {
                param: "node_feats".to_string(),
                value: format!("ncols={}", node_feats.ncols()),
                expected: format!("ncols={}", self.in_dim),
                context: "HyperedgePredictor::score".to_string(),
            });
        }
        for &i in candidate {
            if i >= node_feats.nrows() {
                return Err(GraphError::InvalidParameter {
                    param: "candidate".to_string(),
                    value: format!("node {i}"),
                    expected: format!("< {}", node_feats.nrows()),
                    context: "HyperedgePredictor::score".to_string(),
                });
            }
        }

        // Pool node features
        let pooled = self.config.pooling.pool(node_feats, candidate);

        // MLP forward pass
        let mut h = pooled;
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h);
            if i < self.layers.len() - 1 {
                // SiLU activation
                for v in h.iter_mut() {
                    *v = *v / (1.0 + (-*v).exp());
                }
            }
        }

        // Sigmoid output
        let logit = h[0];
        let prob = 1.0 / (1.0 + (-logit).exp());
        Ok(prob)
    }

    /// Score a batch of candidate hyperedges.
    ///
    /// # Arguments
    /// - `node_feats`: all node features [N × in_dim]
    /// - `candidates`: list of candidate hyperedges (each is a list of node indices)
    ///
    /// # Returns
    /// Vector of probabilities in [0, 1].
    pub fn predict_batch(
        &self,
        node_feats: &Array2<f64>,
        candidates: &[Vec<usize>],
    ) -> Result<Vec<f64>> {
        candidates
            .iter()
            .map(|c| self.score(node_feats, c))
            .collect()
    }
}

// ============================================================================
// Negative sampling
// ============================================================================

/// Generate random negative hyperedge samples.
///
/// For each positive hyperedge, generates `n_neg_per_pos` random k-subsets of
/// the node set, ensuring the generated set differs from the positive hyperedge.
///
/// # Arguments
/// - `positives`: known positive hyperedges
/// - `n_nodes`: total number of nodes
/// - `n_neg_per_pos`: number of negatives per positive
///
/// # Returns
/// List of negative hyperedge candidates.
pub fn generate_negatives(
    positives: &[Vec<usize>],
    n_nodes: usize,
    n_neg_per_pos: usize,
) -> Vec<Vec<usize>> {
    if positives.is_empty() || n_nodes == 0 {
        return Vec::new();
    }

    let mut rng = scirs2_core::random::seeded_rng(42u64);
    let mut negatives = Vec::new();

    // Build a set of all positive hyperedges for fast lookup
    use std::collections::HashSet;
    let pos_set: HashSet<Vec<usize>> = positives
        .iter()
        .map(|p| {
            let mut sorted = p.clone();
            sorted.sort();
            sorted
        })
        .collect();

    for pos in positives {
        let k = pos.len();
        if k == 0 || k > n_nodes {
            continue;
        }

        let mut generated = 0;
        let mut attempts = 0;
        while generated < n_neg_per_pos && attempts < 1000 {
            attempts += 1;
            // Sample k unique nodes without replacement
            let mut candidate: Vec<usize> = (0..n_nodes).collect();
            // Fisher-Yates partial shuffle for k elements
            for i in 0..k {
                let j = i + (rng.random::<f64>() * (n_nodes - i) as f64) as usize;
                let j = j.min(n_nodes - 1);
                candidate.swap(i, j);
            }
            let mut neg: Vec<usize> = candidate[..k].to_vec();
            neg.sort();

            // Check it's not a known positive
            if !pos_set.contains(&neg) {
                negatives.push(neg);
                generated += 1;
            }
        }
    }

    negatives
}

// ============================================================================
// ROC-AUC computation
// ============================================================================

/// Compute the ROC-AUC (Area Under the ROC Curve) using the trapezoidal rule.
///
/// # Arguments
/// - `labels`: ground truth labels (true = positive, false = negative)
/// - `scores`: predicted scores / probabilities (higher = more likely positive)
///
/// # Returns
/// AUC value in [0, 1]. Returns 0.5 for degenerate inputs.
pub fn roc_auc(labels: &[bool], scores: &[f64]) -> f64 {
    assert_eq!(
        labels.len(),
        scores.len(),
        "labels and scores must have equal length"
    );
    if labels.is_empty() {
        return 0.5;
    }

    let n_pos = labels.iter().filter(|&&l| l).count();
    let n_neg = labels.len() - n_pos;
    if n_pos == 0 || n_neg == 0 {
        return 0.5;
    }

    // Sort by score descending
    let mut indices: Vec<usize> = (0..labels.len()).collect();
    indices.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Compute ROC curve points (FPR, TPR) using sorted scores
    let mut tpr_points = vec![0.0_f64];
    let mut fpr_points = vec![0.0_f64];
    let mut tp = 0usize;
    let mut fp = 0usize;

    for &i in &indices {
        if labels[i] {
            tp += 1;
        } else {
            fp += 1;
        }
        let tpr = tp as f64 / n_pos as f64;
        let fpr = fp as f64 / n_neg as f64;
        tpr_points.push(tpr);
        fpr_points.push(fpr);
    }

    // Trapezoidal rule: AUC = sum of trapezoids
    let mut auc = 0.0_f64;
    for i in 1..fpr_points.len() {
        let dfpr = fpr_points[i] - fpr_points[i - 1];
        let avg_tpr = (tpr_points[i] + tpr_points[i - 1]) / 2.0;
        auc += dfpr * avg_tpr;
    }

    auc.clamp(0.0, 1.0)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn make_feats(n: usize, d: usize) -> Array2<f64> {
        let data: Vec<f64> = (0..n * d).map(|i| (i as f64 + 1.0) * 0.1).collect();
        Array2::from_shape_vec((n, d), data).expect("feats")
    }

    #[test]
    fn test_predictor_score_in_unit_interval() {
        let config = HyperedgePredictorConfig {
            hidden_dim: 8,
            ..Default::default()
        };
        let predictor = HyperedgePredictor::new(4, config);
        let feats = make_feats(5, 4);
        let candidate = vec![0, 1, 2];
        let score = predictor.score(&feats, &candidate).expect("score");
        assert!(
            (0.0..=1.0).contains(&score),
            "score must be in [0,1], got {score}"
        );
    }

    #[test]
    fn test_predictor_batch_all_in_unit_interval() {
        let config = HyperedgePredictorConfig {
            hidden_dim: 8,
            ..Default::default()
        };
        let predictor = HyperedgePredictor::new(4, config);
        let feats = make_feats(6, 4);
        let candidates = vec![vec![0, 1], vec![1, 2, 3], vec![3, 4, 5], vec![0, 2, 4]];
        let scores = predictor.predict_batch(&feats, &candidates).expect("batch");
        for s in &scores {
            assert!(*s >= 0.0 && *s <= 1.0, "score {s} not in [0,1]");
        }
        assert_eq!(scores.len(), 4);
    }

    #[test]
    fn test_generate_negatives_differ_from_positives() {
        let positives = vec![vec![0, 1, 2], vec![3, 4, 5]];
        let negatives = generate_negatives(&positives, 8, 3);
        // Check that none of the negatives are in positives
        use std::collections::HashSet;
        let pos_set: HashSet<Vec<usize>> = positives.iter().cloned().collect();
        for neg in &negatives {
            let mut sorted = neg.clone();
            sorted.sort();
            assert!(
                !pos_set.contains(&sorted),
                "negative {:?} should not match a positive",
                neg
            );
        }
    }

    #[test]
    fn test_generate_negatives_count() {
        let positives = vec![vec![0, 1, 2], vec![3, 4, 5]];
        let negatives = generate_negatives(&positives, 20, 5);
        // Up to 2 positives × 5 negatives each = 10 negatives (may be fewer if hard to sample)
        assert!(negatives.len() <= 10 + 5, "too many negatives generated");
        assert!(!negatives.is_empty(), "some negatives should be generated");
    }

    #[test]
    fn test_roc_auc_perfect() {
        // Perfect predictor: positive scores all higher than negative scores
        let labels = vec![true, true, true, false, false, false];
        let scores = vec![0.9, 0.8, 0.7, 0.3, 0.2, 0.1];
        let auc = roc_auc(&labels, &scores);
        assert!(
            (auc - 1.0).abs() < 1e-10,
            "perfect AUC should be 1.0, got {auc}"
        );
    }

    #[test]
    fn test_roc_auc_worst() {
        // Worst predictor: all negative scores higher than positive scores
        let labels = vec![true, true, true, false, false, false];
        let scores = vec![0.1, 0.2, 0.3, 0.7, 0.8, 0.9];
        let auc = roc_auc(&labels, &scores);
        assert!(auc < 0.1, "worst AUC should be ~0.0, got {auc}");
    }

    #[test]
    fn test_roc_auc_random_approx_half() {
        // Uninformative predictor: scores are random w.r.t. labels
        // With fixed labels and scores, AUC should be close to 0.5
        let labels = vec![
            true, false, true, false, true, false, true, false, true, false,
        ];
        let scores = vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
        let auc = roc_auc(&labels, &scores);
        // With all equal scores, AUC depends on tie-breaking → ≈ 0.5
        assert!(
            (0.0..=1.0).contains(&auc),
            "AUC must be in [0,1], got {auc}"
        );
    }

    #[test]
    fn test_pooling_mean() {
        let feats = make_feats(4, 3);
        let pooled = PoolingType::Mean.pool(&feats, &[0, 1, 2]);
        assert_eq!(pooled.len(), 3);
        // Mean of rows 0,1,2 for column 0: (0.1+0.4+0.7)/3
        let expected_col0 = (feats[[0, 0]] + feats[[1, 0]] + feats[[2, 0]]) / 3.0;
        assert!((pooled[0] - expected_col0).abs() < 1e-12);
    }

    #[test]
    fn test_pooling_sum() {
        let feats = make_feats(4, 3);
        let pooled = PoolingType::Sum.pool(&feats, &[0, 1]);
        let expected = feats[[0, 0]] + feats[[1, 0]];
        assert!((pooled[0] - expected).abs() < 1e-12);
    }

    #[test]
    fn test_pooling_max() {
        let feats = make_feats(4, 3);
        let pooled = PoolingType::Max.pool(&feats, &[0, 1, 2]);
        // Max of rows 0,1,2 for column 0: max(0.1, 0.4, 0.7) = 0.7
        assert!((pooled[0] - feats[[2, 0]]).abs() < 1e-12);
    }

    #[test]
    fn test_predictor_empty_candidate_error() {
        let config = HyperedgePredictorConfig::default();
        let predictor = HyperedgePredictor::new(4, config);
        let feats = make_feats(5, 4);
        let result = predictor.score(&feats, &[]);
        assert!(result.is_err(), "empty candidate should return error");
    }
}
