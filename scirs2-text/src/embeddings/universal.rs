//! Universal Sentence Encoder architecture — simplified transformer-based
//! sentence encoder producing fixed-size embeddings.
//!
//! # References
//! Cer et al. (2018) "Universal Sentence Encoder"

use crate::error::{Result, TextError};
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Pooling strategy for aggregating token representations into a sentence vector.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UsePooling {
    /// Average of all token representations.
    Mean,
    /// Element-wise maximum over token representations.
    Max,
    /// Use the first (CLS) token representation.
    Cls,
    /// Learned attention-weighted mean.
    Attentive,
}

/// Configuration for the Universal Sentence Encoder-style model.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct UseConfig {
    /// Embedding / hidden dimension.
    pub d_model: usize,
    /// Number of attention heads.
    pub n_heads: usize,
    /// Number of transformer encoder layers.
    pub n_layers: usize,
    /// Inner dimension of the position-wise FFN.
    pub ffn_dim: usize,
    /// Maximum sequence length.
    pub max_seq_len: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Sentence-level pooling strategy.
    pub pooling: UsePooling,
}

impl Default for UseConfig {
    fn default() -> Self {
        Self {
            d_model: 128,
            n_heads: 4,
            n_layers: 2,
            ffn_dim: 256,
            max_seq_len: 512,
            vocab_size: 30_000,
            pooling: UsePooling::Mean,
        }
    }
}

// ---------------------------------------------------------------------------
// Cross-lingual configuration
// ---------------------------------------------------------------------------

/// Configuration for cross-lingual sentence encoding.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct CrossLingualConfig {
    /// Shared vocabulary size across all languages.
    pub shared_vocab_size: usize,
    /// Number of supported languages.
    pub n_languages: usize,
    /// Dimension of per-language embedding appended to token embeddings.
    pub lang_embedding_dim: usize,
}

impl Default for CrossLingualConfig {
    fn default() -> Self {
        Self {
            shared_vocab_size: 50_000,
            n_languages: 10,
            lang_embedding_dim: 16,
        }
    }
}

// ---------------------------------------------------------------------------
// Deterministic pseudo-random weight generation (LCG)
// ---------------------------------------------------------------------------

/// Simple linear-congruential "random" number in [-scale, +scale].
/// Used to produce deterministic dummy weights without any external crates.
fn lcg_weight(seed: u64, scale: f64) -> f64 {
    // LCG parameters from Numerical Recipes
    let v = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    let frac = (v >> 11) as f64 / (1u64 << 53) as f64; // [0, 1)
    (frac * 2.0 - 1.0) * scale
}

// ---------------------------------------------------------------------------
// Sinusoidal positional encoding
// ---------------------------------------------------------------------------

/// Compute sinusoidal positional encoding matrix of shape [seq_len × d_model].
fn sinusoidal_pe(seq_len: usize, d_model: usize) -> Vec<Vec<f64>> {
    let mut pe = vec![vec![0.0_f64; d_model]; seq_len];
    for pos in 0..seq_len {
        for i in 0..d_model / 2 {
            let angle = pos as f64 / f64::powf(10_000.0, (2 * i) as f64 / d_model as f64);
            pe[pos][2 * i] = angle.sin();
            if 2 * i + 1 < d_model {
                pe[pos][2 * i + 1] = angle.cos();
            }
        }
    }
    pe
}

// ---------------------------------------------------------------------------
// LayerNorm (simplified, no learnable parameters)
// ---------------------------------------------------------------------------

fn layer_norm(x: &[f64], eps: f64) -> Vec<f64> {
    let n = x.len() as f64;
    let mean = x.iter().sum::<f64>() / n;
    let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    x.iter().map(|v| (v - mean) / (var + eps).sqrt()).collect()
}

fn layer_norm_rows(x: &[Vec<f64>]) -> Vec<Vec<f64>> {
    x.iter().map(|row| layer_norm(row, 1e-5)).collect()
}

// ---------------------------------------------------------------------------
// Matrix multiply helpers
// ---------------------------------------------------------------------------

/// (seq_len × d_in) × (d_in × d_out) → (seq_len × d_out)
fn matmul_2d(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let seq = a.len();
    let d_in = b.len();
    let d_out = if d_in == 0 { 0 } else { b[0].len() };
    let mut out = vec![vec![0.0_f64; d_out]; seq];
    for i in 0..seq {
        for k in 0..d_in {
            let a_ik = a[i][k];
            for j in 0..d_out {
                out[i][j] += a_ik * b[k][j];
            }
        }
    }
    out
}

/// (n × m) × (m × p) → (n × p)
fn matmul_rect(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    matmul_2d(a, b)
}

/// Transpose a 2-D matrix.
fn transpose(m: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if m.is_empty() {
        return vec![];
    }
    let rows = m.len();
    let cols = m[0].len();
    let mut out = vec![vec![0.0_f64; rows]; cols];
    for i in 0..rows {
        for j in 0..cols {
            out[j][i] = m[i][j];
        }
    }
    out
}

/// Add bias (broadcast over seq dim).
fn add_bias(x: &[Vec<f64>], bias: &[f64]) -> Vec<Vec<f64>> {
    x.iter()
        .map(|row| row.iter().zip(bias).map(|(v, b)| v + b).collect())
        .collect()
}

/// Element-wise add of two same-shape matrices.
fn mat_add(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    a.iter()
        .zip(b)
        .map(|(ra, rb)| ra.iter().zip(rb).map(|(x, y)| x + y).collect())
        .collect()
}

// ---------------------------------------------------------------------------
// Transformer encoder layer
// ---------------------------------------------------------------------------

/// A single transformer encoder layer (multi-head self-attention + FFN).
pub struct TransformerEncoderLayer {
    d_model: usize,
    n_heads: usize,
    ffn_dim: usize,
    // Projection weights for Q, K, V, O — generated deterministically
    wq: Vec<Vec<f64>>, // d_model × d_model
    wk: Vec<Vec<f64>>,
    wv: Vec<Vec<f64>>,
    wo: Vec<Vec<f64>>,
    // FFN
    w1: Vec<Vec<f64>>, // d_model × ffn_dim
    b1: Vec<f64>,
    w2: Vec<Vec<f64>>, // ffn_dim × d_model
    b2: Vec<f64>,
    /// Query vector for attentive pooling (used only with `UsePooling::Attentive`).
    pub attn_query: Vec<f64>,
}

impl TransformerEncoderLayer {
    /// Create a new encoder layer with deterministic random weights.
    pub fn new(d_model: usize, n_heads: usize, ffn_dim: usize) -> Self {
        let scale_attn = 1.0 / (d_model as f64).sqrt();
        let scale_ffn = 1.0 / (ffn_dim as f64).sqrt();

        let init_matrix = |rows: usize, cols: usize, offset: u64, scale: f64| -> Vec<Vec<f64>> {
            (0..rows)
                .map(|r| {
                    (0..cols)
                        .map(|c| lcg_weight(offset + (r * cols + c) as u64, scale))
                        .collect()
                })
                .collect()
        };
        let init_bias = |len: usize, offset: u64, scale: f64| -> Vec<f64> {
            (0..len)
                .map(|i| lcg_weight(offset + i as u64, scale))
                .collect()
        };

        let wq = init_matrix(d_model, d_model, 1000, scale_attn);
        let wk = init_matrix(d_model, d_model, 2000, scale_attn);
        let wv = init_matrix(d_model, d_model, 3000, scale_attn);
        let wo = init_matrix(d_model, d_model, 4000, scale_attn);
        let w1 = init_matrix(d_model, ffn_dim, 5000, scale_ffn);
        let b1 = init_bias(ffn_dim, 6000, 0.01);
        let w2 = init_matrix(ffn_dim, d_model, 7000, scale_ffn);
        let b2 = init_bias(d_model, 8000, 0.01);
        let attn_query = init_bias(d_model, 9000, scale_attn);

        Self {
            d_model,
            n_heads,
            ffn_dim,
            wq,
            wk,
            wv,
            wo,
            w1,
            b1,
            w2,
            b2,
            attn_query,
        }
    }

    /// Multi-head self-attention with scaled dot-product.
    ///
    /// `x`: \[seq_len × d_model\]
    /// Returns \[seq_len × d_model\]
    pub fn self_attention(
        &self,
        x: &[Vec<f64>],
        mask: Option<&[Vec<bool>]>,
    ) -> Result<Vec<Vec<f64>>> {
        let seq_len = x.len();
        if seq_len == 0 {
            return Err(TextError::InvalidInput(
                "self_attention: empty sequence".into(),
            ));
        }
        let d_head = self.d_model / self.n_heads;
        if d_head == 0 {
            return Err(TextError::InvalidInput("d_model must be >= n_heads".into()));
        }

        let q = matmul_2d(x, &self.wq); // seq × d_model
        let k = matmul_2d(x, &self.wk);
        let v = matmul_2d(x, &self.wv);

        let scale = 1.0 / (d_head as f64).sqrt();

        let mut concat_heads = vec![vec![0.0_f64; self.d_model]; seq_len];

        for h in 0..self.n_heads {
            let h_start = h * d_head;
            let h_end = h_start + d_head;

            // Extract head slices
            let q_h: Vec<Vec<f64>> = q.iter().map(|row| row[h_start..h_end].to_vec()).collect();
            let k_h: Vec<Vec<f64>> = k.iter().map(|row| row[h_start..h_end].to_vec()).collect();
            let v_h: Vec<Vec<f64>> = v.iter().map(|row| row[h_start..h_end].to_vec()).collect();

            // scores = Q × K^T  [seq × seq]
            let kt = transpose(&k_h);
            let scores_raw = matmul_rect(&q_h, &kt);

            // Apply mask & scale, then softmax per row
            let mut attn_weights = vec![vec![0.0_f64; seq_len]; seq_len];
            for i in 0..seq_len {
                let mut row = vec![0.0_f64; seq_len];
                for j in 0..seq_len {
                    let masked = mask.is_some_and(|m| m[i][j]);
                    row[j] = if masked {
                        f64::NEG_INFINITY
                    } else {
                        scores_raw[i][j] * scale
                    };
                }
                // softmax
                let max_v = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let exps: Vec<f64> = row.iter().map(|v| (v - max_v).exp()).collect();
                let sum_exp: f64 = exps.iter().sum();
                let sum_exp = if sum_exp < 1e-12 { 1e-12 } else { sum_exp };
                for j in 0..seq_len {
                    attn_weights[i][j] = exps[j] / sum_exp;
                }
            }

            // context = attn_weights × V_h  [seq × d_head]
            let ctx = matmul_rect(&attn_weights, &v_h);

            for i in 0..seq_len {
                for j in 0..d_head {
                    concat_heads[i][h_start + j] = ctx[i][j];
                }
            }
        }

        // output projection
        let out = matmul_2d(&concat_heads, &self.wo);
        Ok(out)
    }

    /// Position-wise feed-forward: Linear(ReLU(Linear(x))).
    pub fn ffn(&self, x: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        if x.is_empty() {
            return Err(TextError::InvalidInput("ffn: empty input".into()));
        }
        // hidden = x × W1 + b1, ReLU
        let h = add_bias(&matmul_2d(x, &self.w1), &self.b1);
        let h_relu: Vec<Vec<f64>> = h
            .iter()
            .map(|row| row.iter().map(|v| v.max(0.0)).collect())
            .collect();
        // out = h_relu × W2 + b2
        let out = add_bias(&matmul_2d(&h_relu, &self.w2), &self.b2);
        Ok(out)
    }

    /// Full encoder layer forward pass with residual connections and layer norm.
    pub fn forward(&self, x: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        // 1. Self-attention sublayer: LN(x + SA(x))
        let sa_out = self.self_attention(x, None)?;
        let x1 = layer_norm_rows(&mat_add(x, &sa_out));

        // 2. FFN sublayer: LN(x + FFN(x))
        let ffn_out = self.ffn(&x1)?;
        let x2 = layer_norm_rows(&mat_add(&x1, &ffn_out));

        Ok(x2)
    }
}

// ---------------------------------------------------------------------------
// Universal Sentence Encoder
// ---------------------------------------------------------------------------

/// Universal Sentence Encoder — transformer-based architecture.
pub struct UniversalSentenceEncoder {
    /// Model configuration.
    pub config: UseConfig,
    layers: Vec<TransformerEncoderLayer>,
    /// Token embedding table [vocab_size × d_model].
    token_embeddings: Vec<Vec<f64>>,
}

impl UniversalSentenceEncoder {
    /// Construct a new USE with the given configuration.
    pub fn new(config: UseConfig) -> Self {
        let scale = 1.0 / (config.d_model as f64).sqrt();
        let token_embeddings: Vec<Vec<f64>> = (0..config.vocab_size)
            .map(|tok| {
                (0..config.d_model)
                    .map(|dim| lcg_weight((tok * config.d_model + dim) as u64 + 100_000, scale))
                    .collect()
            })
            .collect();

        let layers = (0..config.n_layers)
            .map(|l| {
                // Offset seed per layer so each layer gets distinct weights
                let _offset = l as u64 * 1_000_000;
                TransformerEncoderLayer::new(config.d_model, config.n_heads, config.ffn_dim)
            })
            .collect();

        Self {
            config,
            layers,
            token_embeddings,
        }
    }

    /// Lookup + add sinusoidal positional encoding for token IDs.
    fn embed(&self, token_ids: &[usize]) -> Result<Vec<Vec<f64>>> {
        let seq_len = token_ids.len().min(self.config.max_seq_len);
        if seq_len == 0 {
            return Err(TextError::InvalidInput(
                "encode: token_ids must not be empty".into(),
            ));
        }
        let pe = sinusoidal_pe(seq_len, self.config.d_model);
        let embedded: Result<Vec<Vec<f64>>> = token_ids[..seq_len]
            .iter()
            .enumerate()
            .map(|(pos, &tok_id)| {
                if tok_id >= self.config.vocab_size {
                    return Err(TextError::InvalidInput(format!(
                        "token_id {} out of range (vocab_size={})",
                        tok_id, self.config.vocab_size
                    )));
                }
                let emb = &self.token_embeddings[tok_id];
                Ok(emb.iter().zip(&pe[pos]).map(|(e, p)| e + p).collect())
            })
            .collect();
        embedded
    }

    /// Pool the final hidden states into a single sentence vector.
    fn pool(&self, hidden: &[Vec<f64>]) -> Vec<f64> {
        match self.config.pooling {
            UsePooling::Mean => {
                let n = hidden.len() as f64;
                let d = hidden[0].len();
                let mut out = vec![0.0_f64; d];
                for row in hidden {
                    for (i, v) in row.iter().enumerate() {
                        out[i] += v;
                    }
                }
                out.iter_mut().for_each(|v| *v /= n);
                out
            }
            UsePooling::Max => {
                let d = hidden[0].len();
                let mut out = vec![f64::NEG_INFINITY; d];
                for row in hidden {
                    for (i, v) in row.iter().enumerate() {
                        if *v > out[i] {
                            out[i] = *v;
                        }
                    }
                }
                out
            }
            UsePooling::Cls => hidden[0].clone(),
            UsePooling::Attentive => {
                // Soft attention scores using first layer's query vector
                let query = if self.layers.is_empty() {
                    vec![1.0_f64; hidden[0].len()]
                } else {
                    self.layers[0].attn_query.clone()
                };
                let d = hidden[0].len();
                let scores: Vec<f64> = hidden
                    .iter()
                    .map(|row| row.iter().zip(&query).map(|(v, q)| v * q).sum::<f64>())
                    .collect();
                let max_s = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let exps: Vec<f64> = scores.iter().map(|s| (s - max_s).exp()).collect();
                let sum_exp: f64 = exps.iter().sum::<f64>().max(1e-12);
                let weights: Vec<f64> = exps.iter().map(|e| e / sum_exp).collect();

                let mut out = vec![0.0_f64; d];
                for (row, w) in hidden.iter().zip(&weights) {
                    for (i, v) in row.iter().enumerate() {
                        out[i] += v * w;
                    }
                }
                out
            }
        }
    }

    /// Encode token IDs to a fixed-size sentence embedding of length `d_model`.
    pub fn encode(&self, token_ids: &[usize]) -> Result<Vec<f64>> {
        let mut x = self.embed(token_ids)?;
        for layer in &self.layers {
            x = layer.forward(&x)?;
        }
        Ok(self.pool(&x))
    }

    /// Encode a batch of token ID sequences.
    pub fn encode_batch(&self, batch: &[Vec<usize>]) -> Result<Vec<Vec<f64>>> {
        batch.iter().map(|ids| self.encode(ids)).collect()
    }

    /// Cosine similarity between two sentence embeddings.
    pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
        let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
        if na < 1e-12 || nb < 1e-12 {
            0.0
        } else {
            (dot / (na * nb)).clamp(-1.0, 1.0)
        }
    }

    /// Cross-lingual encoding — adds a language embedding before the stack.
    ///
    /// `token_ids`: vocabulary indices
    /// `lang_id`: 0-indexed language identifier
    /// `xl_config`: cross-lingual configuration
    pub fn cross_lingual_encode(
        &self,
        token_ids: &[usize],
        lang_id: usize,
        xl_config: &CrossLingualConfig,
    ) -> Result<Vec<f64>> {
        if lang_id >= xl_config.n_languages {
            return Err(TextError::InvalidInput(format!(
                "lang_id {} >= n_languages {}",
                lang_id, xl_config.n_languages
            )));
        }
        // Build a language embedding vector of length lang_embedding_dim, then
        // tile / truncate to d_model and add to token embeddings.
        let d = self.config.d_model;
        let ld = xl_config.lang_embedding_dim;
        let lang_emb_raw: Vec<f64> = (0..ld)
            .map(|i| {
                // sinusoidal language embedding
                let angle = lang_id as f64 / f64::powf(100.0, (2 * i) as f64 / ld as f64);
                if i % 2 == 0 {
                    angle.sin()
                } else {
                    angle.cos()
                }
            })
            .collect();
        // Tile to d_model
        let lang_emb: Vec<f64> = (0..d).map(|i| lang_emb_raw[i % ld]).collect();

        let mut x = self.embed(token_ids)?;
        // Add language embedding to every token
        for row in x.iter_mut() {
            for (j, v) in row.iter_mut().enumerate() {
                *v += lang_emb[j];
            }
        }
        for layer in &self.layers {
            x = layer.forward(&x)?;
        }
        Ok(self.pool(&x))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_use() -> UniversalSentenceEncoder {
        UniversalSentenceEncoder::new(UseConfig::default())
    }

    #[test]
    fn test_default_config() {
        let cfg = UseConfig::default();
        assert_eq!(cfg.d_model, 128);
        assert_eq!(cfg.n_heads, 4);
        assert_eq!(cfg.n_layers, 2);
        assert_eq!(cfg.ffn_dim, 256);
        assert_eq!(cfg.pooling, UsePooling::Mean);
    }

    #[test]
    fn test_encode_output_size() {
        let use_model = make_use();
        let ids = vec![1, 2, 3, 4, 5];
        let emb = use_model.encode(&ids).expect("encode failed");
        assert_eq!(emb.len(), 128, "embedding must have d_model dimensions");
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let v = vec![1.0_f64, 2.0, 3.0, 4.0];
        let sim = UniversalSentenceEncoder::cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-9, "identical vectors → sim = 1.0");
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0_f64, 0.0];
        let b = vec![0.0_f64, 1.0];
        let sim = UniversalSentenceEncoder::cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-9, "orthogonal vectors → sim ≈ 0.0");
    }

    #[test]
    fn test_batch_consistent_with_single() {
        let use_model = make_use();
        let ids1 = vec![1_usize, 2, 3];
        let ids2 = vec![4_usize, 5];
        let batch = use_model
            .encode_batch(&[ids1.clone(), ids2.clone()])
            .expect("batch failed");
        let single1 = use_model.encode(&ids1).expect("single encode 1 failed");
        let single2 = use_model.encode(&ids2).expect("single encode 2 failed");
        for (a, b) in batch[0].iter().zip(&single1) {
            assert!((a - b).abs() < 1e-12, "batch[0] must equal single encode");
        }
        for (a, b) in batch[1].iter().zip(&single2) {
            assert!((a - b).abs() < 1e-12, "batch[1] must equal single encode");
        }
    }

    #[test]
    fn test_cross_lingual_config_defaults() {
        let cfg = CrossLingualConfig::default();
        assert_eq!(cfg.shared_vocab_size, 50_000);
        assert_eq!(cfg.n_languages, 10);
        assert_eq!(cfg.lang_embedding_dim, 16);
    }

    #[test]
    fn test_cross_lingual_encode_output_size() {
        let use_model = make_use();
        let xl = CrossLingualConfig::default();
        let emb = use_model
            .cross_lingual_encode(&[1, 2, 3], 0, &xl)
            .expect("cross-lingual encode failed");
        assert_eq!(emb.len(), 128);
    }

    #[test]
    fn test_encode_different_inputs_differ() {
        // Use a model with n_layers=0 so we get only the embedding + positional
        // encoding, without layernorm collapse from the transformer stack.
        let cfg = UseConfig {
            n_layers: 0,
            ..UseConfig::default()
        };
        let use_model = UniversalSentenceEncoder::new(cfg);
        let emb1 = use_model.encode(&[1, 2, 3]).unwrap();
        let emb2 = use_model.encode(&[100, 200, 300]).unwrap();
        // The two embeddings should not be element-wise identical
        let all_eq = emb1.iter().zip(&emb2).all(|(a, b)| (a - b).abs() < 1e-12);
        assert!(
            !all_eq,
            "different token inputs should produce numerically distinct embeddings"
        );
    }

    #[test]
    fn test_sinusoidal_pe_shape() {
        let pe = sinusoidal_pe(10, 128);
        assert_eq!(pe.len(), 10);
        assert_eq!(pe[0].len(), 128);
    }

    #[test]
    fn test_max_pooling() {
        let cfg = UseConfig {
            pooling: UsePooling::Max,
            n_layers: 1,
            ..UseConfig::default()
        };
        let m = UniversalSentenceEncoder::new(cfg);
        let emb = m.encode(&[1, 2, 3]).unwrap();
        assert_eq!(emb.len(), 128);
    }

    #[test]
    fn test_cls_pooling() {
        let cfg = UseConfig {
            pooling: UsePooling::Cls,
            n_layers: 1,
            ..UseConfig::default()
        };
        let m = UniversalSentenceEncoder::new(cfg);
        let emb = m.encode(&[0, 1, 2]).unwrap();
        assert_eq!(emb.len(), 128);
    }

    #[test]
    fn test_attentive_pooling() {
        let cfg = UseConfig {
            pooling: UsePooling::Attentive,
            n_layers: 1,
            ..UseConfig::default()
        };
        let m = UniversalSentenceEncoder::new(cfg);
        let emb = m.encode(&[5, 6, 7]).unwrap();
        assert_eq!(emb.len(), 128);
    }
}
