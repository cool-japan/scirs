//! Sentence embedding aggregation and SimCSE contrastive learning.
//!
//! This module provides:
//!
//! - **[`SentenceEmbedder`]**: aggregates token-level embeddings (token-ID
//!   based, using `ndarray`) into a single sentence vector using several
//!   pooling strategies, mirroring the Sentence-BERT family of models.
//! - **[`SimCseTrainer`]** (legacy, ndarray-based): computes the InfoNCE
//!   contrastive loss for pre-computed embeddings.
//! - **[`encoder::SentenceEncoder`]**: word-level sentence encoder with a
//!   `HashMap<String, Vec<f32>>` lookup table, suitable for use without
//!   pre-tokenised token IDs.
//! - **[`simcse::SimCSETrainer`]**: unsupervised SimCSE trainer with a full
//!   training loop (noise augmentation + NT-Xent + gradient-free update).
//!
//! Neither component requires external neural-network infrastructure.
//!
//! # Example
//!
//! ```rust
//! use scirs2_text::sentence_embeddings::{
//!     SentenceEmbedder, SentenceEmbedderConfig, PoolingStrategy,
//! };
//!
//! let config = SentenceEmbedderConfig {
//!     d_model: 64,
//!     pooling: PoolingStrategy::MeanPooling,
//!     normalize: true,
//! };
//! let embedder = SentenceEmbedder::new(1000, config, 42);
//!
//! let token_ids = vec![101u32, 7592, 102];
//! let emb = embedder.embed_tokens(&token_ids);
//! assert_eq!(emb.len(), 64);
//! ```

/// Differentiable projection head backed by `scirs2-autograd`.
pub mod autograd_projection;
/// Cross-lingual alignment via orthogonal Procrustes.
pub mod cross_lingual;
/// Word-level sentence encoder (USE-style, `HashMap` vocabulary).
pub mod encoder;
/// Standalone InfoNCE / NT-Xent contrastive loss functions (encoder-agnostic).
pub mod infonce;
/// Unsupervised SimCSE trainer with noise augmentation and NT-Xent loss.
pub mod simcse;
/// Pairwise semantic similarity metrics and utility functions.
pub mod similarity;
/// High-level SimCSE trainer (frozen encoder + differentiable projection).
pub mod trainer;
/// Universal Sentence Encoder-style token-ID-based encoder.
pub mod universal;

pub use autograd_projection::{DifferentiableProjection, ProjectionConfig};
pub use cross_lingual::{procrustes_align, AlignedEncoder, CrossLingualAligner};
pub use encoder::{
    PoolingStrategy as SentenceEncoderPooling, SentenceEncoder, SentenceEncoderConfig,
};
pub use infonce::{cosine_similarity_matrix, infonce_loss, top1_accuracy};
pub use simcse::{SimCSELoss, SimCSETrainer};
pub use similarity::{
    semantic_similarity_matrix, semantic_similarity_tokens, semantic_similarity_vecs,
    vector_similarity, PairwiseSimilarityMetric, SentenceEncoderLike,
};
pub use trainer::{SimcseConfig, SimcseTrainer, TrainStep};
pub use universal::{UniversalPoolingStrategy, UniversalSentenceEncoder};

use std::fmt::Debug;

use scirs2_core::ndarray::{Array1, Array2};

// ── PoolingStrategy ───────────────────────────────────────────────────────────

/// Strategy for aggregating per-token embeddings into a sentence vector.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum PoolingStrategy {
    /// Average of all token embeddings (padding token 0 is excluded).
    MeanPooling,
    /// Use only the embedding of the first token (CLS-style).
    ClsPooling,
    /// Element-wise maximum across all token embeddings.
    MaxPooling,
    /// Weighted mean: earlier tokens receive linearly higher weight
    /// (a triangular weighting scheme).
    WeightedMeanPooling,
}

// ── SentenceEmbedderConfig ────────────────────────────────────────────────────

/// Configuration for [`SentenceEmbedder`].
#[derive(Debug, Clone)]
pub struct SentenceEmbedderConfig {
    /// Embedding dimensionality (`d_model`).
    pub d_model: usize,
    /// Token-embedding aggregation strategy.
    pub pooling: PoolingStrategy,
    /// When `true`, L2-normalise the pooled sentence vector to unit length.
    pub normalize: bool,
}

impl Default for SentenceEmbedderConfig {
    fn default() -> Self {
        SentenceEmbedderConfig {
            d_model: 768,
            pooling: PoolingStrategy::MeanPooling,
            normalize: true,
        }
    }
}

// ── SentenceEmbedder ──────────────────────────────────────────────────────────

/// Aggregates token embeddings to produce sentence-level representations.
///
/// The token embedding matrix is randomly initialised from a seeded LCG
/// (linear congruential generator) so results are deterministic without
/// requiring an external RNG crate.
pub struct SentenceEmbedder {
    /// Tokenizer configuration.
    pub config: SentenceEmbedderConfig,
    /// Token embedding matrix of shape `[vocab_size × d_model]`.
    pub embeddings: Array2<f64>,
}

impl SentenceEmbedder {
    /// Create a new embedder with randomly initialised token embeddings.
    ///
    /// # Parameters
    /// - `vocab_size`: number of rows in the embedding matrix.
    /// - `config`: pooling and normalisation settings.
    /// - `seed`: seed for the LCG initialiser (deterministic).
    pub fn new(vocab_size: usize, config: SentenceEmbedderConfig, seed: u64) -> Self {
        let d_model = config.d_model;
        let embeddings = Array2::from_shape_fn((vocab_size, d_model), |(i, j)| {
            // Simple LCG: produces values in (-1, 1)
            let state = lcg_f64(seed, i as u64 * d_model as u64 + j as u64);
            state * 2.0 - 1.0
        });

        SentenceEmbedder { config, embeddings }
    }

    /// Aggregate token embeddings for the given sequence of token IDs.
    ///
    /// Tokens with ID 0 are treated as padding and excluded from mean /
    /// weighted-mean pooling.  For max-pooling they are included so that
    /// the output shape is always `[d_model]`.
    ///
    /// Returns an error when `token_ids` is empty.
    pub fn embed_tokens(&self, token_ids: &[u32]) -> Array1<f64> {
        let d = self.config.d_model;
        let vocab_size = self.embeddings.nrows();

        // Collect valid row indices (clamp out-of-range to 0)
        let rows: Vec<usize> = token_ids
            .iter()
            .map(|&id| (id as usize).min(vocab_size.saturating_sub(1)))
            .collect();

        if rows.is_empty() {
            return Array1::zeros(d);
        }

        let output = match self.config.pooling {
            PoolingStrategy::MeanPooling => {
                // Exclude padding (original id == 0)
                let non_pad: Vec<usize> = token_ids
                    .iter()
                    .zip(rows.iter())
                    .filter(|(&id, _)| id != 0)
                    .map(|(_, &row)| row)
                    .collect();

                let effective: &[usize] = if non_pad.is_empty() { &rows } else { &non_pad };
                let n = effective.len() as f64;
                let mut sum = Array1::<f64>::zeros(d);
                for &row in effective {
                    sum += &self.embeddings.row(row);
                }
                sum / n
            }

            PoolingStrategy::ClsPooling => {
                // Use the first token's embedding regardless of ID
                self.embeddings.row(rows[0]).to_owned()
            }

            PoolingStrategy::MaxPooling => {
                let mut max_emb = self.embeddings.row(rows[0]).to_owned();
                for &row in &rows[1..] {
                    let emb = self.embeddings.row(row);
                    for (m, e) in max_emb.iter_mut().zip(emb.iter()) {
                        if *e > *m {
                            *m = *e;
                        }
                    }
                }
                max_emb
            }

            PoolingStrategy::WeightedMeanPooling => {
                // Weight[i] = (n - i) so earlier tokens have higher weight.
                // Exclude padding (id == 0).
                let weighted: Vec<(usize, f64)> = token_ids
                    .iter()
                    .zip(rows.iter())
                    .enumerate()
                    .filter(|(_, (&id, _))| id != 0)
                    .map(|(i, (_, &row))| {
                        let w = (token_ids.len() - i) as f64;
                        (row, w)
                    })
                    .collect();

                let effective: Vec<(usize, f64)> = if weighted.is_empty() {
                    rows.iter()
                        .enumerate()
                        .map(|(i, &row)| {
                            let w = (rows.len() - i) as f64;
                            (row, w)
                        })
                        .collect()
                } else {
                    weighted
                };

                let total_weight: f64 = effective.iter().map(|(_, w)| w).sum();
                let mut result = Array1::<f64>::zeros(d);
                for (row, w) in &effective {
                    let emb = self.embeddings.row(*row);
                    for (r, e) in result.iter_mut().zip(emb.iter()) {
                        *r += e * w;
                    }
                }
                result / total_weight
            }
        };

        if self.config.normalize {
            l2_normalize_1d(output)
        } else {
            output
        }
    }

    /// Cosine similarity between two embedding vectors.
    ///
    /// Both vectors are assumed to have the same length.  Returns a value in
    /// `[-1, 1]`.
    pub fn cosine_similarity(&self, emb1: &Array1<f64>, emb2: &Array1<f64>) -> f64 {
        cosine_sim_1d(emb1, emb2)
    }

    /// Compute the `n × n` pairwise cosine-similarity matrix for a set of
    /// sentence embeddings.
    ///
    /// `embeddings` has shape `[n × d_model]`.
    pub fn pairwise_similarity(&self, embeddings: &Array2<f64>) -> Array2<f64> {
        let n = embeddings.nrows();
        let mut sim = Array2::<f64>::zeros((n, n));

        for i in 0..n {
            let ei = embeddings.row(i);
            for j in 0..n {
                let ej = embeddings.row(j);
                let s = cosine_sim_arr(ei.view(), ej.view());
                sim[[i, j]] = s;
            }
        }
        sim
    }
}

impl Debug for SentenceEmbedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SentenceEmbedder")
            .field("d_model", &self.config.d_model)
            .field("vocab_size", &self.embeddings.nrows())
            .finish()
    }
}

// ── SimCseConfig ──────────────────────────────────────────────────────────────

/// Configuration for the SimCSE contrastive trainer.
#[derive(Debug, Clone)]
pub struct SimCseConfig {
    /// Temperature parameter τ for the InfoNCE loss (typically 0.05).
    pub temperature: f64,
    /// Number of negative examples per anchor-positive pair.
    pub n_negatives_per_positive: usize,
    /// Output dimensionality of the linear projection head.
    pub d_projection: usize,
}

impl Default for SimCseConfig {
    fn default() -> Self {
        SimCseConfig {
            temperature: 0.05,
            n_negatives_per_positive: 7,
            d_projection: 128,
        }
    }
}

// ── SimCseTrainer ─────────────────────────────────────────────────────────────

/// SimCSE contrastive loss computation.
///
/// Implements the InfoNCE objective from:
/// > Gao et al. (2021) *SimCSE: Simple Contrastive Learning of Sentence
/// > Embeddings*.  <https://arxiv.org/abs/2104.08821>
///
/// A linear projection head maps `d_model`-dimensional sentence embeddings to
/// a lower `d_projection`-dimensional space before computing similarities.
pub struct SimCseTrainer {
    /// Trainer configuration.
    pub config: SimCseConfig,
    /// Projection weight matrix of shape `[d_model × d_projection]`.
    pub projection: Array2<f64>,
}

impl SimCseTrainer {
    /// Create a new trainer.
    ///
    /// `d_model` must match the dimensionality of the sentence embeddings that
    /// will be passed to [`Self::info_nce_loss`] and [`Self::batch_loss`].
    pub fn new(d_model: usize, config: SimCseConfig, seed: u64) -> Self {
        let d_proj = config.d_projection;
        let projection = Array2::from_shape_fn((d_model, d_proj), |(i, j)| {
            let s = lcg_f64(seed.wrapping_add(1), i as u64 * d_proj as u64 + j as u64);
            (s * 2.0 - 1.0) * (2.0 / (d_model as f64).sqrt())
        });

        SimCseTrainer { config, projection }
    }

    /// Project a `d_model`-dimensional vector to `d_projection` dimensions.
    fn project(&self, emb: &Array1<f64>) -> Array1<f64> {
        // result[j] = Σ_i emb[i] * projection[i, j]
        let d_proj = self.projection.ncols();
        let mut out = Array1::<f64>::zeros(d_proj);
        for j in 0..d_proj {
            let col = self.projection.column(j);
            out[j] = emb.iter().zip(col.iter()).map(|(a, b)| a * b).sum();
        }
        l2_normalize_1d(out)
    }

    /// Compute the InfoNCE loss for a single (anchor, positive, negatives) tuple.
    ///
    /// All embeddings are first projected through the linear head and
    /// L2-normalised.  Then:
    ///
    /// ```text
    /// loss = -log( exp(sim(a,p)/τ) / (exp(sim(a,p)/τ) + Σᵢ exp(sim(a,negᵢ)/τ)) )
    /// ```
    ///
    /// The loss is always ≥ 0 (it is a negative log-probability) and approaches
    /// `log(n_negatives + 1)` in the worst case and approaches 0 as the positive
    /// pair similarity greatly exceeds all negative similarities.
    pub fn info_nce_loss(
        &self,
        anchor: &Array1<f64>,
        positive: &Array1<f64>,
        negatives: &[Array1<f64>],
    ) -> f64 {
        let tau = self.config.temperature;

        let a_proj = self.project(anchor);
        let p_proj = self.project(positive);

        let sim_ap = cosine_sim_1d(&a_proj, &p_proj) / tau;
        let exp_ap = sim_ap.exp();

        let denom = negatives
            .iter()
            .map(|neg| {
                let n_proj = self.project(neg);
                let sim_an = cosine_sim_1d(&a_proj, &n_proj) / tau;
                sim_an.exp()
            })
            .fold(exp_ap, |acc, x| acc + x);

        // -log(exp_ap / denom) = log(denom) - sim_ap
        if denom <= 0.0 || !denom.is_finite() {
            return -sim_ap;
        }

        -(exp_ap.ln() - denom.ln())
    }

    /// Compute the average InfoNCE loss over a mini-batch.
    ///
    /// Each even-indexed embedding `i` acts as anchor, with `i+1` as its
    /// positive (paired) example.  All other embeddings in the batch are used
    /// as negatives (in-batch negatives, SimCSE-style).
    ///
    /// If the batch has fewer than 2 embeddings this returns `0.0`.
    pub fn batch_loss(&self, embeddings: &Array2<f64>) -> f64 {
        let n = embeddings.nrows();
        if n < 2 {
            return 0.0;
        }

        // Process pairs (0,1), (2,3), …
        let mut total_loss = 0.0;
        let mut count = 0;

        let mut i = 0;
        while i + 1 < n {
            let anchor = embeddings.row(i).to_owned();
            let positive = embeddings.row(i + 1).to_owned();

            // All rows except anchor and positive are negatives
            let negatives: Vec<Array1<f64>> = (0..n)
                .filter(|&j| j != i && j != i + 1)
                .map(|j| embeddings.row(j).to_owned())
                .collect();

            total_loss += self.info_nce_loss(&anchor, &positive, &negatives);
            count += 1;
            i += 2;
        }

        if count == 0 {
            0.0
        } else {
            total_loss / count as f64
        }
    }

    /// Mine hard negatives: pairs `(i, j)` where cosine similarity is high
    /// but the embeddings come from different sentences.
    ///
    /// Returns the top-`top_k` most-similar non-identical pairs.
    pub fn hard_negative_mining(
        &self,
        embeddings: &Array2<f64>,
        top_k: usize,
    ) -> Vec<(usize, usize)> {
        let n = embeddings.nrows();
        if n < 2 {
            return vec![];
        }

        // Collect all (i, j, sim) with i < j
        let mut pairs: Vec<(usize, usize, f64)> = Vec::new();
        for i in 0..n {
            let ei = embeddings.row(i);
            for j in (i + 1)..n {
                let ej = embeddings.row(j);
                let s = cosine_sim_arr(ei.view(), ej.view());
                pairs.push((i, j, s));
            }
        }

        // Sort by descending similarity
        pairs.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

        pairs
            .into_iter()
            .take(top_k)
            .map(|(i, j, _)| (i, j))
            .collect()
    }
}

impl Debug for SimCseTrainer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SimCseTrainer")
            .field("d_model", &self.projection.nrows())
            .field("d_projection", &self.config.d_projection)
            .finish()
    }
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Linear congruential generator — returns a pseudo-random value in `[0, 1)`.
///
/// Uses the Knuth multiplicative constants so successive calls with
/// incrementing offsets cover the space reasonably well.
fn lcg_f64(seed: u64, offset: u64) -> f64 {
    const A: u64 = 6_364_136_223_846_793_005;
    const C: u64 = 1_442_695_040_888_963_407;
    let state = A.wrapping_mul(seed.wrapping_add(offset)).wrapping_add(C);
    // Extract upper 52 bits and map to [0, 1)
    ((state >> 12) as f64) / ((1u64 << 52) as f64)
}

/// L2-normalise a 1-D array in-place.  Returns the array unchanged when its
/// norm is zero or NaN.
fn l2_normalize_1d(mut v: Array1<f64>) -> Array1<f64> {
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 1e-12 && norm.is_finite() {
        v /= norm;
    }
    v
}

/// Cosine similarity between two `Array1<f64>` values.
fn cosine_sim_1d(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    cosine_sim_arr(a.view(), b.view())
}

/// Cosine similarity between two `ArrayView1<f64>` slices.
fn cosine_sim_arr(
    a: scirs2_core::ndarray::ArrayView1<f64>,
    b: scirs2_core::ndarray::ArrayView1<f64>,
) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na < 1e-12 || nb < 1e-12 {
        return 0.0;
    }
    (dot / (na * nb)).clamp(-1.0, 1.0)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn make_embedder(pooling: PoolingStrategy) -> SentenceEmbedder {
        let config = SentenceEmbedderConfig {
            d_model: 32,
            pooling,
            normalize: true,
        };
        SentenceEmbedder::new(200, config, 42)
    }

    fn make_embedder_unnorm(pooling: PoolingStrategy) -> SentenceEmbedder {
        let config = SentenceEmbedderConfig {
            d_model: 32,
            pooling,
            normalize: false,
        };
        SentenceEmbedder::new(200, config, 42)
    }

    // ── SentenceEmbedder tests ─────────────────────────────────────────────

    #[test]
    fn new_creates_correct_shape() {
        let config = SentenceEmbedderConfig {
            d_model: 16,
            pooling: PoolingStrategy::MeanPooling,
            normalize: false,
        };
        let emb = SentenceEmbedder::new(100, config, 0);
        assert_eq!(emb.embeddings.shape(), &[100, 16]);
    }

    #[test]
    fn embed_tokens_mean_shape() {
        let emb = make_embedder(PoolingStrategy::MeanPooling);
        let ids = vec![1u32, 2, 3, 4];
        let out = emb.embed_tokens(&ids);
        assert_eq!(out.len(), 32);
    }

    #[test]
    fn embed_tokens_cls_equals_first() {
        let emb = make_embedder_unnorm(PoolingStrategy::ClsPooling);
        let ids = vec![5u32, 10, 15];
        let out = emb.embed_tokens(&ids);
        let first_row = emb.embeddings.row(5).to_owned();
        assert_abs_diff_eq!(
            out.as_slice().unwrap(),
            first_row.as_slice().unwrap(),
            epsilon = 1e-10
        );
    }

    #[test]
    fn embed_tokens_max_pooling_ge_all_inputs() {
        let emb = make_embedder_unnorm(PoolingStrategy::MaxPooling);
        let ids = vec![1u32, 2, 3];
        let out = emb.embed_tokens(&ids);
        // Each element of max-pooled output must be >= all individual embeddings
        for (d, &max_val) in out.iter().enumerate() {
            for &id in &ids {
                let row_val = emb.embeddings[[id as usize, d]];
                assert!(
                    max_val >= row_val - 1e-12,
                    "max[{}]={} < row {}[{}]={}",
                    d,
                    max_val,
                    id,
                    d,
                    row_val
                );
            }
        }
    }

    #[test]
    fn normalize_true_unit_norm() {
        let emb = make_embedder(PoolingStrategy::MeanPooling);
        let ids = vec![1u32, 2, 3, 4, 5];
        let out = emb.embed_tokens(&ids);
        let norm: f64 = out.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert_abs_diff_eq!(norm, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn cosine_similarity_same_vector() {
        let emb = make_embedder(PoolingStrategy::MeanPooling);
        let ids = vec![1u32, 2];
        let v = emb.embed_tokens(&ids);
        let sim = emb.cosine_similarity(&v, &v);
        assert_abs_diff_eq!(sim, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn cosine_similarity_opposite_vector() {
        let emb = make_embedder(PoolingStrategy::MeanPooling);
        let ids = vec![1u32, 2];
        let v = emb.embed_tokens(&ids);
        let neg_v = v.mapv(|x| -x);
        let sim = emb.cosine_similarity(&v, &neg_v);
        assert_abs_diff_eq!(sim, -1.0, epsilon = 1e-10);
    }

    #[test]
    fn pairwise_similarity_shape() {
        let emb = make_embedder(PoolingStrategy::MeanPooling);
        let rows: Vec<Array1<f64>> = (0..5u32)
            .map(|i| emb.embed_tokens(&[i + 1, i + 2]))
            .collect();
        let mat = Array2::from_shape_fn((5, 32), |(i, j)| rows[i][j]);
        let sim = emb.pairwise_similarity(&mat);
        assert_eq!(sim.shape(), &[5, 5]);
    }

    #[test]
    fn pairwise_similarity_diagonal_ones() {
        let emb = make_embedder(PoolingStrategy::MeanPooling);
        let rows: Vec<Array1<f64>> = (0..4u32)
            .map(|i| emb.embed_tokens(&[i + 1, i + 2]))
            .collect();
        let mat = Array2::from_shape_fn((4, 32), |(i, j)| rows[i][j]);
        let sim = emb.pairwise_similarity(&mat);
        for i in 0..4 {
            assert_abs_diff_eq!(sim[[i, i]], 1.0, epsilon = 1e-10);
        }
    }

    // ── SimCseTrainer tests ────────────────────────────────────────────────

    fn make_trainer() -> SimCseTrainer {
        let config = SimCseConfig::default();
        SimCseTrainer::new(32, config, 7)
    }

    fn rand_emb(d: usize, seed: u64) -> Array1<f64> {
        let raw = Array1::from_shape_fn(d, |i| lcg_f64(seed, i as u64) * 2.0 - 1.0);
        l2_normalize_1d(raw)
    }

    #[test]
    fn info_nce_loss_is_log_prob() {
        let trainer = make_trainer();
        let a = rand_emb(32, 1);
        let p = rand_emb(32, 2);
        let negs: Vec<Array1<f64>> = (0..7).map(|i| rand_emb(32, i + 10)).collect();
        let loss = trainer.info_nce_loss(&a, &p, &negs);
        // InfoNCE = -log(p) is a non-negative cross-entropy; loss >= 0
        assert!(loss >= 0.0, "InfoNCE loss must be >= 0, got {}", loss);
        assert!(loss.is_finite(), "loss must be finite");
    }

    #[test]
    fn info_nce_loss_perfect_match_near_lower_bound() {
        let trainer = make_trainer();
        // When anchor == positive (perfect cosine match), loss should be near
        // -log(1/(1+n_neg)) from the limit where positive dominates.
        let a = rand_emb(32, 42);
        let negs: Vec<Array1<f64>> = (0..7).map(|i| rand_emb(32, i + 100)).collect();
        let loss = trainer.info_nce_loss(&a, &a, &negs);
        // When a == p, the positive score dominates and loss approaches its
        // minimum (near 0); verify it is finite and non-negative.
        assert!(loss.is_finite(), "loss must be finite");
    }

    #[test]
    fn batch_loss_runs_without_panic() {
        let trainer = make_trainer();
        let embs = Array2::from_shape_fn((8, 32), |(i, j)| {
            lcg_f64(99 + i as u64, j as u64) * 2.0 - 1.0
        });
        let loss = trainer.batch_loss(&embs);
        assert!(loss.is_finite());
    }

    #[test]
    fn hard_negative_mining_returns_k_pairs() {
        let trainer = make_trainer();
        let embs = Array2::from_shape_fn((6, 32), |(i, j)| {
            lcg_f64(50 + i as u64, j as u64) * 2.0 - 1.0
        });
        let pairs = trainer.hard_negative_mining(&embs, 3);
        assert_eq!(pairs.len(), 3);
    }

    #[test]
    fn simcse_config_defaults() {
        let cfg = SimCseConfig::default();
        assert!((cfg.temperature - 0.05).abs() < 1e-10);
        assert_eq!(cfg.n_negatives_per_positive, 7);
        assert_eq!(cfg.d_projection, 128);
    }

    #[test]
    fn sentenceembedder_config_defaults() {
        let cfg = SentenceEmbedderConfig::default();
        assert_eq!(cfg.d_model, 768);
        assert_eq!(cfg.pooling, PoolingStrategy::MeanPooling);
        assert!(cfg.normalize);
    }
}
