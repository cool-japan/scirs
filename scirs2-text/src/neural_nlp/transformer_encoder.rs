//! Pure-ndarray multi-head self-attention transformer encoder for text.
//!
//! Implements a minimal BERT-style encoder: embedding table + sinusoidal
//! position encoding + N × (pre-norm MHA + pre-norm FFN) layers using `f32`.

use crate::error::{Result, TextError};
use scirs2_core::ndarray::{s, Array1, Array2, Array3, Axis};

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for [`TransformerTextEncoder`].
#[derive(Debug, Clone)]
pub struct TransformerEncoderConfig {
    /// Vocabulary size (number of distinct token IDs).
    pub vocab_size: usize,
    /// Dimensionality of token + position embeddings.
    pub hidden_size: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Number of encoder layers.
    pub num_layers: usize,
    /// Maximum sequence length supported.
    pub max_seq_len: usize,
    /// Dropout probability (applied during training; unused at inference).
    pub dropout: f32,
    /// PRNG seed for weight initialisation.
    pub seed: u64,
}

impl Default for TransformerEncoderConfig {
    fn default() -> Self {
        Self {
            vocab_size: 30000,
            hidden_size: 256,
            num_heads: 4,
            num_layers: 2,
            max_seq_len: 512,
            dropout: 0.1,
            seed: 42,
        }
    }
}

// ─── Attention Layer ──────────────────────────────────────────────────────────

/// Single multi-head self-attention layer (f32).
struct MhsaLayer {
    /// Q projection: [hidden, hidden]
    w_q: Array2<f32>,
    /// K projection: [hidden, hidden]
    w_k: Array2<f32>,
    /// V projection: [hidden, hidden]
    w_v: Array2<f32>,
    /// Output projection: [hidden, hidden]
    w_o: Array2<f32>,
    /// Pre-attention LayerNorm scale
    ln1_scale: Array1<f32>,
    /// Pre-attention LayerNorm bias
    ln1_bias: Array1<f32>,
    n_heads: usize,
    d_k: usize,
}

/// Feed-forward sub-layer (two linear + GELU).
struct FfnLayer {
    /// W1: [hidden, 4*hidden]
    w1: Array2<f32>,
    b1: Array1<f32>,
    /// W2: [4*hidden, hidden]
    w2: Array2<f32>,
    b2: Array1<f32>,
    /// Pre-FFN LayerNorm scale
    ln2_scale: Array1<f32>,
    ln2_bias: Array1<f32>,
}

// ─── LCG-based weight initialisation ─────────────────────────────────────────

fn next_lcg(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let bits = (*seed >> 33) as f32 / (u32::MAX as f32);
    (bits - 0.5) * 2.0 // uniform in [-1, 1]
}

fn xavier_init(rows: usize, cols: usize, seed: &mut u64) -> Array2<f32> {
    let scale = (6.0_f32 / (rows + cols) as f32).sqrt();
    Array2::from_shape_fn((rows, cols), |_| next_lcg(seed) * scale)
}

fn zeros1(n: usize) -> Array1<f32> {
    Array1::zeros(n)
}

fn ones1(n: usize) -> Array1<f32> {
    Array1::ones(n)
}

// ─── Math helpers ─────────────────────────────────────────────────────────────

/// Row-wise softmax in place.
fn softmax_rows(x: &mut Array2<f32>) {
    let (rows, cols) = x.dim();
    for i in 0..rows {
        let max_val = x.row(i).fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let mut sum = 0.0_f32;
        for j in 0..cols {
            x[[i, j]] = (x[[i, j]] - max_val).exp();
            sum += x[[i, j]];
        }
        if sum > 0.0 {
            for j in 0..cols {
                x[[i, j]] /= sum;
            }
        }
    }
}

/// GELU approximation: 0.5x(1+tanh(√(2/π)(x+0.044715x³))).
#[inline]
fn gelu(x: f32) -> f32 {
    let inner = (2.0_f32 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x * x * x);
    0.5 * x * (1.0 + inner.tanh())
}

/// Layer normalisation: (x - μ) / (σ + ε) * scale + bias.
fn layer_norm(x: &Array2<f32>, scale: &Array1<f32>, bias: &Array1<f32>) -> Array2<f32> {
    let eps = 1e-5_f32;
    let (seq, hidden) = x.dim();
    let mut out = Array2::zeros((seq, hidden));
    for i in 0..seq {
        let row = x.row(i);
        let mean = row.sum() / hidden as f32;
        let var = row.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / hidden as f32;
        let inv_std = 1.0 / (var + eps).sqrt();
        for j in 0..hidden {
            out[[i, j]] = (x[[i, j]] - mean) * inv_std * scale[j] + bias[j];
        }
    }
    out
}

// ─── MhsaLayer impl ──────────────────────────────────────────────────────────

impl MhsaLayer {
    fn new(hidden: usize, n_heads: usize, seed: &mut u64) -> Result<Self> {
        if !hidden.is_multiple_of(n_heads) {
            return Err(TextError::InvalidInput(format!(
                "hidden_size {hidden} must be divisible by num_heads {n_heads}"
            )));
        }
        let d_k = hidden / n_heads;
        Ok(Self {
            w_q: xavier_init(hidden, hidden, seed),
            w_k: xavier_init(hidden, hidden, seed),
            w_v: xavier_init(hidden, hidden, seed),
            w_o: xavier_init(hidden, hidden, seed),
            ln1_scale: ones1(hidden),
            ln1_bias: zeros1(hidden),
            n_heads,
            d_k,
        })
    }

    /// Forward pass; returns (output [seq, hidden], attention [n_heads, seq, seq]).
    fn forward_with_attn(&self, x: &Array2<f32>) -> Result<(Array2<f32>, Array2<f32>)> {
        let (seq, hidden) = x.dim();

        // Pre-norm
        let xn = layer_norm(x, &self.ln1_scale, &self.ln1_bias);

        // Q, K, V projections: [seq, hidden]
        let q = xn.dot(&self.w_q);
        let k = xn.dot(&self.w_k);
        let v = xn.dot(&self.w_v);

        let scale = (self.d_k as f32).sqrt();

        // Compute attention per head, accumulate output
        let mut out = Array2::zeros((seq, hidden));
        // averaged attention weights [seq, seq]
        let mut avg_attn = Array2::zeros((seq, seq));

        for h in 0..self.n_heads {
            let start = h * self.d_k;
            let end = start + self.d_k;

            let q_h = q.slice(s![.., start..end]).to_owned(); // [seq, d_k]
            let k_h = k.slice(s![.., start..end]).to_owned(); // [seq, d_k]
            let v_h = v.slice(s![.., start..end]).to_owned(); // [seq, d_k]

            // Attention scores [seq, seq]
            let mut scores = q_h.dot(&k_h.t()) / scale; // [seq, seq]
            softmax_rows(&mut scores);

            // Add to avg_attn
            avg_attn += &scores;

            // Context [seq, d_k]
            let ctx = scores.dot(&v_h);
            out.slice_mut(s![.., start..end]).assign(&ctx);
        }

        // Average across heads
        let n_heads_f = self.n_heads as f32;
        avg_attn.mapv_inplace(|v| v / n_heads_f);

        // Output projection + residual
        let proj = out.dot(&self.w_o);
        let result = x + &proj;

        Ok((result, avg_attn))
    }

    /// Forward pass for all n_heads attention maps: returns [n_heads, seq, seq].
    fn forward_all_heads(&self, x: &Array2<f32>) -> Result<(Array2<f32>, Array3<f32>)> {
        let (seq, hidden) = x.dim();

        let xn = layer_norm(x, &self.ln1_scale, &self.ln1_bias);

        let q = xn.dot(&self.w_q);
        let k = xn.dot(&self.w_k);
        let v = xn.dot(&self.w_v);

        let scale = (self.d_k as f32).sqrt();

        let mut out = Array2::zeros((seq, hidden));
        let mut all_attn = Array3::zeros((self.n_heads, seq, seq));

        for h in 0..self.n_heads {
            let start = h * self.d_k;
            let end = start + self.d_k;

            let q_h = q.slice(s![.., start..end]).to_owned();
            let k_h = k.slice(s![.., start..end]).to_owned();
            let v_h = v.slice(s![.., start..end]).to_owned();

            let mut scores = q_h.dot(&k_h.t()) / scale;
            softmax_rows(&mut scores);

            all_attn.slice_mut(s![h, .., ..]).assign(&scores);

            let ctx = scores.dot(&v_h);
            out.slice_mut(s![.., start..end]).assign(&ctx);
        }

        let proj = out.dot(&self.w_o);
        let result = x + &proj;

        Ok((result, all_attn))
    }
}

// ─── FfnLayer impl ───────────────────────────────────────────────────────────

impl FfnLayer {
    fn new(hidden: usize, seed: &mut u64) -> Self {
        let ffn_dim = 4 * hidden;
        Self {
            w1: xavier_init(hidden, ffn_dim, seed),
            b1: zeros1(ffn_dim),
            w2: xavier_init(ffn_dim, hidden, seed),
            b2: zeros1(hidden),
            ln2_scale: ones1(hidden),
            ln2_bias: zeros1(hidden),
        }
    }

    fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        // Pre-norm
        let xn = layer_norm(x, &self.ln2_scale, &self.ln2_bias);

        // W1 + bias + GELU
        let h1 = xn.dot(&self.w1) + &self.b1;
        let h1 = h1.mapv(gelu);

        // W2 + bias + residual
        let h2 = h1.dot(&self.w2) + &self.b2;
        x + &h2
    }
}

// ─── Sinusoidal positional encoding ──────────────────────────────────────────

fn sinusoidal_pe(max_seq: usize, hidden: usize) -> Array2<f32> {
    let mut pe = Array2::zeros((max_seq, hidden));
    for pos in 0..max_seq {
        for i in (0..hidden).step_by(2) {
            let angle = pos as f32 / 10000.0_f32.powf(i as f32 / hidden as f32);
            pe[[pos, i]] = angle.sin();
            if i + 1 < hidden {
                pe[[pos, i + 1]] = angle.cos();
            }
        }
    }
    pe
}

// ─── TransformerTextEncoder ───────────────────────────────────────────────────

/// Transformer-based text encoder that maps token-ID sequences to contextual embeddings.
///
/// Uses pure-ndarray f32 multi-head self-attention with sinusoidal position encoding.
pub struct TransformerTextEncoder {
    config: TransformerEncoderConfig,
    /// Token embedding table [vocab_size, hidden]
    embedding: Array2<f32>,
    /// Positional encoding table [max_seq_len, hidden]
    position_enc: Array2<f32>,
    /// Attention sub-layers (one per encoder layer)
    attn_layers: Vec<MhsaLayer>,
    /// FFN sub-layers (one per encoder layer)
    ffn_layers: Vec<FfnLayer>,
}

impl TransformerTextEncoder {
    /// Create a new encoder from the given config.
    pub fn new(config: TransformerEncoderConfig) -> Result<Self> {
        let mut seed = config.seed;

        let scale = (config.hidden_size as f32).sqrt();
        let embedding = Array2::from_shape_fn((config.vocab_size, config.hidden_size), |_| {
            next_lcg(&mut seed) / scale
        });

        let position_enc = sinusoidal_pe(config.max_seq_len, config.hidden_size);

        let mut attn_layers = Vec::with_capacity(config.num_layers);
        let mut ffn_layers = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            attn_layers.push(MhsaLayer::new(
                config.hidden_size,
                config.num_heads,
                &mut seed,
            )?);
            ffn_layers.push(FfnLayer::new(config.hidden_size, &mut seed));
        }

        Ok(Self {
            config,
            embedding,
            position_enc,
            attn_layers,
            ffn_layers,
        })
    }

    /// Look up embeddings + add positional encoding for the given token IDs.
    fn embed_tokens(&self, tokens: &[usize]) -> Result<Array2<f32>> {
        let seq = tokens.len();
        if seq == 0 {
            return Err(TextError::InvalidInput("Empty token sequence".to_string()));
        }
        if seq > self.config.max_seq_len {
            return Err(TextError::InvalidInput(format!(
                "Sequence length {seq} exceeds max_seq_len {}",
                self.config.max_seq_len
            )));
        }

        let hidden = self.config.hidden_size;
        let mut x = Array2::zeros((seq, hidden));
        for (i, &tok) in tokens.iter().enumerate() {
            if tok >= self.config.vocab_size {
                return Err(TextError::InvalidInput(format!(
                    "Token ID {tok} out of vocab range {}",
                    self.config.vocab_size
                )));
            }
            let emb_row = self.embedding.row(tok);
            let pe_row = self.position_enc.row(i);
            for j in 0..hidden {
                x[[i, j]] = emb_row[j] + pe_row[j];
            }
        }
        Ok(x)
    }

    /// Encode token IDs to contextual embeddings `[seq_len, hidden_size]`.
    pub fn encode_tokens(&self, tokens: &[usize]) -> Result<Array2<f32>> {
        let mut x = self.embed_tokens(tokens)?;
        for (attn, ffn) in self.attn_layers.iter().zip(self.ffn_layers.iter()) {
            let (out, _) = attn.forward_with_attn(&x)?;
            x = ffn.forward(&out);
        }
        Ok(x)
    }

    /// Pool contextual embeddings to a single sentence embedding `[hidden_size]`.
    /// Uses mean pooling across all token positions.
    pub fn encode_sentence(&self, tokens: &[usize]) -> Result<Array1<f32>> {
        let ctx = self.encode_tokens(tokens)?;
        ctx.mean_axis(Axis(0))
            .ok_or_else(|| TextError::InvalidInput("Cannot mean-pool empty context".to_string()))
    }

    /// Encode tokens and expose per-layer per-head attention weights.
    ///
    /// Returns `(embeddings [seq, hidden], attention_weights)` where
    /// `attention_weights[layer]` has shape `[n_heads, seq, seq]`.
    pub fn encode_with_attention(
        &self,
        tokens: &[usize],
    ) -> Result<(Array2<f32>, Vec<Array3<f32>>)> {
        let mut x = self.embed_tokens(tokens)?;
        let mut all_attn = Vec::with_capacity(self.config.num_layers);

        for (attn, ffn) in self.attn_layers.iter().zip(self.ffn_layers.iter()) {
            let (out, layer_attn) = attn.forward_all_heads(&x)?;
            x = ffn.forward(&out);
            all_attn.push(layer_attn);
        }

        Ok((x, all_attn))
    }

    /// Access the encoder configuration.
    pub fn config(&self) -> &TransformerEncoderConfig {
        &self.config
    }

    /// Access the embedding table (read-only).
    pub fn embedding(&self) -> &Array2<f32> {
        &self.embedding
    }

    /// Mutably access the embedding table (for fine-tuning).
    pub fn embedding_mut(&mut self) -> &mut Array2<f32> {
        &mut self.embedding
    }
}
