//! Universal Sentence Encoder-style embeddings (token-ID-based).
//!
//! Provides [`UniversalSentenceEncoder`] which takes pre-built token-level
//! embedding matrices and aggregates them into fixed-length sentence vectors
//! using one of six [`UniversalPoolingStrategy`] variants — mirroring the
//! family of pooling options described in:
//!
//! > Cer et al. (2018) "Universal Sentence Encoder."
//! > <https://arxiv.org/abs/1803.11175>
//!
//! No external neural-network infrastructure is required.  The only optional
//! "learning" operations — IDF-weight computation and attention-query fitting —
//! are performed by simple arithmetic without any third-party autograd engine.

use scirs2_core::ndarray::{Array1, Array2, ArrayView2};

// ── UniversalPoolingStrategy ──────────────────────────────────────────────────

/// Pooling strategy for [`UniversalSentenceEncoder`].
///
/// Each variant describes how per-token embeddings are collapsed into a single
/// fixed-length sentence vector.
#[derive(Debug, Clone, PartialEq)]
pub enum UniversalPoolingStrategy {
    /// Use only the embedding of the first token (CLS-token style).
    ClsToken,
    /// Arithmetic mean over all token embeddings.
    Mean,
    /// Component-wise maximum over all token embeddings.
    Max,
    /// Mean divided by √(n_tokens).
    ///
    /// This down-scales the pooled vector for longer sequences, which can
    /// help when sequences have variable lengths.
    MeanSqrt,
    /// Attention-weighted mean with a learnable query vector `q`.
    ///
    /// Scores are computed as `softmax(E·q)`, then the result is `Eᵀ·scores`.
    /// Requires calling [`UniversalSentenceEncoder::fit_attention_pooling`]
    /// before use, or the mean pool is used as a fallback.
    AttentionPooling,
    /// Weighted mean using log-IDF weights per token.
    ///
    /// Requires calling [`UniversalSentenceEncoder::fit_idf_weights`] before
    /// use, or the mean pool is used as a fallback.
    WeightedMean,
}

// ── UniversalSentenceEncoder ──────────────────────────────────────────────────

/// Token-ID-based sentence encoder with six pooling strategies.
///
/// # Example
///
/// ```rust
/// use scirs2_text::sentence_embeddings::universal::{
///     UniversalSentenceEncoder, UniversalPoolingStrategy,
/// };
/// use scirs2_core::ndarray::Array2;
///
/// // 10-word vocab, 8-dimensional embeddings
/// let emb = Array2::<f32>::from_shape_fn((10, 8), |(i, j)| (i * 8 + j) as f32);
/// let encoder = UniversalSentenceEncoder::new(emb, UniversalPoolingStrategy::Mean, true);
///
/// let tokens = vec![1usize, 3, 5];
/// let vec = encoder.encode(&tokens);
/// assert_eq!(vec.len(), 8);
/// ```
pub struct UniversalSentenceEncoder {
    /// Token embedding matrix, shape `[vocab_size × d_model]`.
    pub token_embeddings: Array2<f32>,
    /// Active pooling strategy.
    pub pooling: UniversalPoolingStrategy,
    /// Embedding dimensionality.
    pub d_model: usize,
    /// Whether to L2-normalise the output vector.
    pub normalize_output: bool,
    /// Query vector for [`UniversalPoolingStrategy::AttentionPooling`].
    /// Shape `[d_model]`.  `None` until [`fit_attention_pooling`] is called.
    attention_query: Option<Array1<f32>>,
    /// Log-IDF weight per token index for [`UniversalPoolingStrategy::WeightedMean`].
    /// Shape `[vocab_size]`.  `None` until [`fit_idf_weights`] is called.
    idf_weights: Option<Array1<f32>>,
}

impl UniversalSentenceEncoder {
    // ── Constructors ──────────────────────────────────────────────────────────

    /// Create a new encoder from a pre-built embedding matrix.
    ///
    /// # Parameters
    /// - `token_embeddings`: matrix of shape `[vocab_size × d_model]`.
    /// - `pooling`: which aggregation strategy to apply.
    /// - `normalize_output`: when `true`, the output of [`encode`](Self::encode)
    ///   is L2-normalised to unit length.
    pub fn new(
        token_embeddings: Array2<f32>,
        pooling: UniversalPoolingStrategy,
        normalize_output: bool,
    ) -> Self {
        let d_model = token_embeddings.ncols();
        UniversalSentenceEncoder {
            token_embeddings,
            pooling,
            d_model,
            normalize_output,
            attention_query: None,
            idf_weights: None,
        }
    }

    // ── encode ────────────────────────────────────────────────────────────────

    /// Encode a sequence of token indices into a fixed-length embedding vector.
    ///
    /// Token indices ≥ `vocab_size` are clamped to `vocab_size - 1`.
    /// An empty token sequence returns a zero vector of length `d_model`.
    pub fn encode(&self, tokens: &[usize]) -> Array1<f32> {
        if tokens.is_empty() || self.token_embeddings.nrows() == 0 {
            return Array1::zeros(self.d_model);
        }

        let vocab_size = self.token_embeddings.nrows();
        // Clamp out-of-range token indices
        let safe_tokens: Vec<usize> = tokens
            .iter()
            .map(|&t| t.min(vocab_size.saturating_sub(1)))
            .collect();

        let result = match &self.pooling {
            UniversalPoolingStrategy::ClsToken => {
                self.token_embeddings.row(safe_tokens[0]).to_owned()
            }
            UniversalPoolingStrategy::Mean => self.mean_pool(&safe_tokens),
            UniversalPoolingStrategy::Max => self.max_pool(&safe_tokens),
            UniversalPoolingStrategy::MeanSqrt => {
                let n = safe_tokens.len().max(1) as f32;
                self.mean_pool(&safe_tokens).mapv(|v| v / n.sqrt())
            }
            UniversalPoolingStrategy::AttentionPooling => {
                if let Some(q) = &self.attention_query {
                    self.attention_pool(&safe_tokens, q)
                } else {
                    // Fallback to mean if query not fitted yet
                    self.mean_pool(&safe_tokens)
                }
            }
            UniversalPoolingStrategy::WeightedMean => {
                if let Some(idf) = &self.idf_weights {
                    self.weighted_mean_pool(tokens, idf)
                } else {
                    self.mean_pool(&safe_tokens)
                }
            }
        };

        if self.normalize_output {
            l2_normalize(result)
        } else {
            result
        }
    }

    // ── fit_idf_weights ───────────────────────────────────────────────────────

    /// Compute log-IDF weights from a corpus of token sequences.
    ///
    /// `idf[t] = log((N + 1) / (df_t + 1))` (add-one smoothed), where N is the
    /// number of documents and df_t is the number of documents containing token t.
    ///
    /// # Parameters
    /// - `corpus`: slice of documents, each a `Vec<usize>` of token indices.
    /// - `vocab_size`: total vocabulary size (must match the encoder's matrix).
    pub fn fit_idf_weights(&mut self, corpus: &[Vec<usize>], vocab_size: usize) {
        let n = corpus.len() as f32;
        let mut df = vec![0u32; vocab_size];

        for doc in corpus {
            // Count each token at most once per document
            let mut seen = vec![false; vocab_size];
            for &t in doc {
                if t < vocab_size && !seen[t] {
                    df[t] += 1;
                    seen[t] = true;
                }
            }
        }

        let idf: Array1<f32> =
            Array1::from_iter(df.iter().map(|&d| ((n + 1.0) / (d as f32 + 1.0)).ln()));
        self.idf_weights = Some(idf);
    }

    // ── fit_attention_pooling ─────────────────────────────────────────────────

    /// Learn a query vector for attention pooling via gradient-free SGD.
    ///
    /// Performs `epochs` sweeps over `corpus`.  In each sweep and for each
    /// document the gradient of the reconstruction loss with respect to `q` is
    /// estimated using a finite-difference step of 1e-4 and `q` is updated with
    /// learning rate `lr`.
    ///
    /// # Parameters
    /// - `corpus`: training corpus of token-index sequences.
    /// - `epochs`: number of full sweeps (1 is usually sufficient for
    ///   initialisation).
    /// - `lr`: learning rate for the query update.
    pub fn fit_attention_pooling(&mut self, corpus: &[Vec<usize>], epochs: usize, lr: f32) {
        let vocab_size = self.token_embeddings.nrows();
        // Initialise query from the mean of all embeddings
        let mut q = Array1::<f32>::zeros(self.d_model);
        for i in 0..vocab_size {
            let row = self.token_embeddings.row(i);
            for j in 0..self.d_model {
                q[j] += row[j];
            }
        }
        if vocab_size > 0 {
            q.mapv_inplace(|v| v / vocab_size as f32);
        }

        let h = 1e-4_f32;

        for _epoch in 0..epochs {
            for doc in corpus {
                if doc.is_empty() {
                    continue;
                }
                let safe: Vec<usize> = doc
                    .iter()
                    .map(|&t| t.min(vocab_size.saturating_sub(1)))
                    .collect();

                // Current attended output
                let out0 = self.attention_pool_with_query(&safe, &q);

                // Gradient w.r.t. q via central differences (component-wise)
                let mut grad = Array1::<f32>::zeros(self.d_model);
                for j in 0..self.d_model {
                    let mut q_plus = q.clone();
                    q_plus[j] += h;
                    let out_plus = self.attention_pool_with_query(&safe, &q_plus);

                    // Reconstruction loss: ||out_plus - mean||^2 - ||out0 - mean||^2
                    // simplified: just steer toward larger attention on any token
                    let loss_plus: f32 = out_plus
                        .iter()
                        .zip(out0.iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum();

                    grad[j] = loss_plus / h;
                }

                // Gradient descent (minimise — negative because we want variety)
                for j in 0..self.d_model {
                    q[j] -= lr * grad[j];
                }
            }
        }

        // L2-normalise the learned query
        let norm: f32 = q.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-12 {
            q.mapv_inplace(|v| v / norm);
        }
        self.attention_query = Some(q);
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    /// Borrow the fitted IDF weight vector, if any.
    pub fn idf_weights(&self) -> Option<&Array1<f32>> {
        self.idf_weights.as_ref()
    }

    /// Borrow the fitted attention query vector, if any.
    pub fn attention_query(&self) -> Option<&Array1<f32>> {
        self.attention_query.as_ref()
    }

    // ── Internal pooling helpers ───────────────────────────────────────────────

    fn mean_pool(&self, safe_tokens: &[usize]) -> Array1<f32> {
        let mut sum = Array1::<f32>::zeros(self.d_model);
        for &t in safe_tokens {
            let row = self.token_embeddings.row(t);
            for j in 0..self.d_model {
                sum[j] += row[j];
            }
        }
        let n = safe_tokens.len().max(1) as f32;
        sum.mapv(|v| v / n)
    }

    fn max_pool(&self, safe_tokens: &[usize]) -> Array1<f32> {
        let mut result = self.token_embeddings.row(safe_tokens[0]).to_owned();
        for &t in &safe_tokens[1..] {
            let row = self.token_embeddings.row(t);
            for j in 0..self.d_model {
                if row[j] > result[j] {
                    result[j] = row[j];
                }
            }
        }
        result
    }

    /// Attention-weighted pooling: scores = softmax(E·q), result = Eᵀ·scores.
    fn attention_pool(&self, safe_tokens: &[usize], q: &Array1<f32>) -> Array1<f32> {
        self.attention_pool_with_query(safe_tokens, q)
    }

    fn attention_pool_with_query(&self, safe_tokens: &[usize], q: &Array1<f32>) -> Array1<f32> {
        let n = safe_tokens.len();
        // Compute raw scores: score[i] = dot(emb[token_i], q)
        let mut scores = vec![0.0f32; n];
        for (i, &t) in safe_tokens.iter().enumerate() {
            let row = self.token_embeddings.row(t);
            scores[i] = row.iter().zip(q.iter()).map(|(a, b)| a * b).sum();
        }

        // Stable softmax
        let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
        let sum_exp: f32 = exp_scores.iter().sum();
        if sum_exp > 1e-12 {
            exp_scores.iter_mut().for_each(|s| *s /= sum_exp);
        } else {
            let uniform = 1.0 / n as f32;
            exp_scores.iter_mut().for_each(|s| *s = uniform);
        }

        // Weighted sum: result = Σ weight_i * emb[token_i]
        let mut result = Array1::<f32>::zeros(self.d_model);
        for (i, &t) in safe_tokens.iter().enumerate() {
            let row = self.token_embeddings.row(t);
            let w = exp_scores[i];
            for j in 0..self.d_model {
                result[j] += w * row[j];
            }
        }
        result
    }

    /// IDF-weighted mean pool.  Uses raw (unclamped) token indices to look up
    /// IDF weights (OOV tokens get weight 1.0).
    fn weighted_mean_pool(&self, tokens: &[usize], idf: &Array1<f32>) -> Array1<f32> {
        let vocab_size = self.token_embeddings.nrows();
        let idf_len = idf.len();

        let mut result = Array1::<f32>::zeros(self.d_model);
        let mut total_weight = 0.0f32;

        for &t in tokens {
            let row_idx = t.min(vocab_size.saturating_sub(1));
            let weight = if t < idf_len { idf[t] } else { 1.0f32 };
            let row = self.token_embeddings.row(row_idx);
            for j in 0..self.d_model {
                result[j] += weight * row[j];
            }
            total_weight += weight;
        }

        if total_weight > 1e-12 {
            result.mapv_inplace(|v| v / total_weight);
        }
        result
    }

    /// Expose the raw embedding matrix as a 2-D view (for external inspection).
    pub fn embeddings_view(&self) -> ArrayView2<f32> {
        self.token_embeddings.view()
    }
}

impl std::fmt::Debug for UniversalSentenceEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UniversalSentenceEncoder")
            .field("vocab_size", &self.token_embeddings.nrows())
            .field("d_model", &self.d_model)
            .field("pooling", &self.pooling)
            .field("normalize_output", &self.normalize_output)
            .field("has_attention_query", &self.attention_query.is_some())
            .field("has_idf_weights", &self.idf_weights.is_some())
            .finish()
    }
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// L2-normalise a 1-D `Array1<f32>`.  Returns the input unchanged when its
/// norm is zero or not finite.
fn l2_normalize(mut v: Array1<f32>) -> Array1<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-12 && norm.is_finite() {
        v.mapv_inplace(|x| x / norm);
    }
    v
}
