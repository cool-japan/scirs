//! PatchTST: Patch-based Time Series Transformer
//!
//! Implementation of *"A Time Series is Worth 64 Words: Long-term Forecasting with Transformers"*
//! (Nie et al., 2023). Key innovations:
//!
//! - **Patching**: Unfolds the input time series into non-overlapping or overlapping patches,
//!   reducing the number of tokens quadratically and enabling the transformer to capture
//!   local semantic information.
//!
//! - **Channel Independence (CI)**: Each variate (channel) is processed independently through
//!   the same transformer backbone, avoiding cross-channel interference and enabling
//!   better per-variate representation learning.
//!
//! - **Instance Normalization**: Reversible instance normalization to handle distribution shift.
//!
//! The architecture: Input → Patch Embedding → Positional Encoding → Transformer Encoder
//! → Flatten Head → Prediction.

use scirs2_core::ndarray::{Array1, Array2, Array3};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::Debug;

use super::nn_utils;
use crate::error::{Result, TimeSeriesError};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the PatchTST model.
#[derive(Debug, Clone)]
pub struct PatchTSTConfig {
    /// Input sequence length.
    pub seq_len: usize,
    /// Prediction horizon.
    pub pred_len: usize,
    /// Number of input variates (channels).
    pub n_channels: usize,
    /// Patch length: each patch covers this many time steps.
    pub patch_len: usize,
    /// Stride between consecutive patches.
    pub stride: usize,
    /// Transformer model dimension.
    pub d_model: usize,
    /// Number of attention heads.
    pub n_heads: usize,
    /// Number of transformer encoder layers.
    pub n_layers: usize,
    /// Feed-forward hidden dimension.
    pub d_ff: usize,
    /// Dropout probability (used conceptually; deterministic in inference).
    pub dropout: f64,
    /// Whether to use channel-independent mode (default: true).
    pub channel_independent: bool,
    /// Random seed for weight initialization.
    pub seed: u32,
}

impl Default for PatchTSTConfig {
    fn default() -> Self {
        Self {
            seq_len: 96,
            pred_len: 24,
            n_channels: 7,
            patch_len: 16,
            stride: 8,
            d_model: 64,
            n_heads: 4,
            n_layers: 3,
            d_ff: 128,
            dropout: 0.1,
            channel_independent: true,
            seed: 42,
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute the number of patches given sequence length, patch length, and stride.
fn compute_num_patches(seq_len: usize, patch_len: usize, stride: usize) -> usize {
    if seq_len <= patch_len {
        1
    } else {
        (seq_len - patch_len) / stride + 1
    }
}

/// Element-wise addition of two identically shaped 2-D arrays.
fn add_2d<F: Float>(a: &Array2<F>, b: &Array2<F>) -> Array2<F> {
    let (rows, cols) = a.dim();
    let mut out = Array2::zeros((rows, cols));
    for r in 0..rows {
        for c in 0..cols {
            out[[r, c]] = a[[r, c]] + b[[r, c]];
        }
    }
    out
}

/// GELU activation for 2-D arrays.
fn gelu_2d<F: Float + FromPrimitive>(x: &Array2<F>) -> Array2<F> {
    let half = F::from(0.5).unwrap_or_else(|| F::zero());
    let sqrt_2_pi = F::from(0.7978845608).unwrap_or_else(|| F::one());
    let coeff = F::from(0.044715).unwrap_or_else(|| F::zero());
    x.mapv(|v| half * v * (F::one() + (sqrt_2_pi * (v + coeff * v * v * v)).tanh()))
}

// ---------------------------------------------------------------------------
// Sinusoidal positional encoding
// ---------------------------------------------------------------------------

/// Sinusoidal positional encoding table of shape `(max_len, d_model)`.
#[derive(Debug)]
struct SinusoidalPE<F: Float> {
    table: Array2<F>,
}

impl<F: Float + FromPrimitive + Debug> SinusoidalPE<F> {
    fn new(max_len: usize, d_model: usize) -> Self {
        let mut table = Array2::zeros((max_len, d_model));
        let base = F::from(10000.0).unwrap_or_else(|| F::one());
        let dim_f = F::from(d_model as f64).unwrap_or_else(|| F::one());
        for pos in 0..max_len {
            let pos_f = F::from(pos as f64).unwrap_or_else(|| F::zero());
            for d in 0..d_model {
                let d_f = F::from(d as f64).unwrap_or_else(|| F::zero());
                let two = F::from(2.0).unwrap_or_else(|| F::one());
                let angle = pos_f / base.powf(two * (d_f / dim_f));
                table[[pos, d]] = if d % 2 == 0 { angle.sin() } else { angle.cos() };
            }
        }
        Self { table }
    }

    /// Add positional encoding to input of shape `(seq_len, d_model)`.
    fn add(&self, input: &Array2<F>) -> Array2<F> {
        let (seq_len, d) = input.dim();
        let max_len = self.table.nrows();
        let mut output = input.clone();
        for t in 0..seq_len.min(max_len) {
            for j in 0..d {
                output[[t, j]] = output[[t, j]] + self.table[[t, j]];
            }
        }
        output
    }
}

// ---------------------------------------------------------------------------
// Multi-head self-attention
// ---------------------------------------------------------------------------

/// Multi-head self-attention with Q, K, V projections.
#[derive(Debug)]
struct MultiHeadAttention<F: Float> {
    d_model: usize,
    n_heads: usize,
    head_dim: usize,
    w_q: Array2<F>,
    w_k: Array2<F>,
    w_v: Array2<F>,
    w_o: Array2<F>,
    b_o: Array1<F>,
}

impl<F: Float + FromPrimitive + Debug> MultiHeadAttention<F> {
    fn new(d_model: usize, n_heads: usize, seed: u32) -> Result<Self> {
        if d_model == 0 || n_heads == 0 {
            return Err(TimeSeriesError::InvalidInput(
                "d_model and n_heads must be positive".to_string(),
            ));
        }
        if d_model % n_heads != 0 {
            return Err(TimeSeriesError::InvalidInput(
                "d_model must be divisible by n_heads".to_string(),
            ));
        }
        let head_dim = d_model / n_heads;
        Ok(Self {
            d_model,
            n_heads,
            head_dim,
            w_q: nn_utils::xavier_matrix(d_model, d_model, seed),
            w_k: nn_utils::xavier_matrix(d_model, d_model, seed.wrapping_add(100)),
            w_v: nn_utils::xavier_matrix(d_model, d_model, seed.wrapping_add(200)),
            w_o: nn_utils::xavier_matrix(d_model, d_model, seed.wrapping_add(300)),
            b_o: nn_utils::zero_bias(d_model),
        })
    }

    /// Self-attention forward: input shape `(seq_len, d_model)` -> `(seq_len, d_model)`.
    fn forward(&self, input: &Array2<F>) -> Result<Array2<F>> {
        let (seq_len, _) = input.dim();
        let b_zero = nn_utils::zero_bias::<F>(self.d_model);
        let q = nn_utils::dense_forward(input, &self.w_q, &b_zero);
        let k = nn_utils::dense_forward(input, &self.w_k, &b_zero);
        let v = nn_utils::dense_forward(input, &self.w_v, &b_zero);

        let scale = F::from(self.head_dim as f64)
            .unwrap_or_else(|| F::one())
            .sqrt();

        // Multi-head: split, attend, concatenate
        let mut concat_out = Array2::zeros((seq_len, self.d_model));

        for h in 0..self.n_heads {
            let offset = h * self.head_dim;
            // Compute attention scores for this head
            let mut scores = Array2::zeros((seq_len, seq_len));
            for i in 0..seq_len {
                for j in 0..seq_len {
                    let mut dot = F::zero();
                    for d in 0..self.head_dim {
                        dot = dot + q[[i, offset + d]] * k[[j, offset + d]];
                    }
                    scores[[i, j]] = dot / scale;
                }
            }
            let attn = nn_utils::softmax_rows(&scores);

            // Weighted sum of values
            for i in 0..seq_len {
                for d in 0..self.head_dim {
                    let mut acc = F::zero();
                    for j in 0..seq_len {
                        acc = acc + attn[[i, j]] * v[[j, offset + d]];
                    }
                    concat_out[[i, offset + d]] = acc;
                }
            }
        }

        let output = nn_utils::dense_forward(&concat_out, &self.w_o, &self.b_o);
        Ok(output)
    }
}

// ---------------------------------------------------------------------------
// Transformer encoder layer
// ---------------------------------------------------------------------------

/// Transformer encoder layer: self-attention + FFN + layer norms + residuals.
#[derive(Debug)]
struct TransformerEncoderLayer<F: Float> {
    self_attn: MultiHeadAttention<F>,
    w1: Array2<F>,
    b1: Array1<F>,
    w2: Array2<F>,
    b2: Array1<F>,
    ln1_gamma: Array1<F>,
    ln1_beta: Array1<F>,
    ln2_gamma: Array1<F>,
    ln2_beta: Array1<F>,
}

impl<F: Float + FromPrimitive + Debug> TransformerEncoderLayer<F> {
    fn new(d_model: usize, n_heads: usize, d_ff: usize, seed: u32) -> Result<Self> {
        Ok(Self {
            self_attn: MultiHeadAttention::new(d_model, n_heads, seed)?,
            w1: nn_utils::xavier_matrix(d_ff, d_model, seed.wrapping_add(500)),
            b1: nn_utils::zero_bias(d_ff),
            w2: nn_utils::xavier_matrix(d_model, d_ff, seed.wrapping_add(600)),
            b2: nn_utils::zero_bias(d_model),
            ln1_gamma: Array1::ones(d_model),
            ln1_beta: Array1::zeros(d_model),
            ln2_gamma: Array1::ones(d_model),
            ln2_beta: Array1::zeros(d_model),
        })
    }

    fn forward(&self, input: &Array2<F>) -> Result<Array2<F>> {
        // Pre-norm: layer norm -> self-attention -> residual
        let normed1 = nn_utils::layer_norm(input, &self.ln1_gamma, &self.ln1_beta);
        let attn_out = self.self_attn.forward(&normed1)?;
        let residual1 = add_2d(input, &attn_out);

        // Pre-norm: layer norm -> FFN -> residual
        let normed2 = nn_utils::layer_norm(&residual1, &self.ln2_gamma, &self.ln2_beta);
        let h = gelu_2d(&nn_utils::dense_forward(&normed2, &self.w1, &self.b1));
        let ffn_out = nn_utils::dense_forward(&h, &self.w2, &self.b2);
        let residual2 = add_2d(&residual1, &ffn_out);

        Ok(residual2)
    }
}

// ---------------------------------------------------------------------------
// Patch embedding
// ---------------------------------------------------------------------------

/// Patch embedding: unfolds input into patches and projects to d_model.
#[derive(Debug)]
struct PatchEmbedding<F: Float> {
    patch_len: usize,
    stride: usize,
    /// Linear projection: (d_model, patch_len)
    w_proj: Array2<F>,
    b_proj: Array1<F>,
}

impl<F: Float + FromPrimitive + Debug> PatchEmbedding<F> {
    fn new(patch_len: usize, stride: usize, d_model: usize, seed: u32) -> Self {
        Self {
            patch_len,
            stride,
            w_proj: nn_utils::xavier_matrix(d_model, patch_len, seed),
            b_proj: nn_utils::zero_bias(d_model),
        }
    }

    /// Unfold a 1-D series of length `seq_len` into patches, then project.
    ///
    /// Input: shape `(seq_len,)` -> Output: shape `(num_patches, d_model)`.
    fn forward(&self, series: &Array1<F>) -> Array2<F> {
        let seq_len = series.len();
        let num_patches = compute_num_patches(seq_len, self.patch_len, self.stride);
        let mut patches = Array2::zeros((num_patches, self.patch_len));

        for p in 0..num_patches {
            let start = p * self.stride;
            for j in 0..self.patch_len {
                let idx = start + j;
                if idx < seq_len {
                    patches[[p, j]] = series[idx];
                }
                // else stays zero (padding)
            }
        }

        // Project: (num_patches, patch_len) * W^T + b -> (num_patches, d_model)
        nn_utils::dense_forward(&patches, &self.w_proj, &self.b_proj)
    }
}

// ---------------------------------------------------------------------------
// Flatten head
// ---------------------------------------------------------------------------

/// Flatten head: concatenate all patch embeddings and project to pred_len.
#[derive(Debug)]
struct FlattenHead<F: Float> {
    /// Linear: (pred_len, num_patches * d_model)
    w_out: Array2<F>,
    b_out: Array1<F>,
}

impl<F: Float + FromPrimitive + Debug> FlattenHead<F> {
    fn new(num_patches: usize, d_model: usize, pred_len: usize, seed: u32) -> Self {
        let in_dim = num_patches * d_model;
        Self {
            w_out: nn_utils::xavier_matrix(pred_len, in_dim, seed),
            b_out: nn_utils::zero_bias(pred_len),
        }
    }

    /// Input: shape `(num_patches, d_model)` -> Output: shape `(pred_len,)`.
    fn forward(&self, encoded: &Array2<F>) -> Array1<F> {
        let (num_patches, d_model) = encoded.dim();
        // Flatten to 1-D vector of length num_patches * d_model
        let flat_len = num_patches * d_model;
        let mut flat = Array1::zeros(flat_len);
        for p in 0..num_patches {
            for d in 0..d_model {
                flat[p * d_model + d] = encoded[[p, d]];
            }
        }
        nn_utils::dense_forward_vec(&flat, &self.w_out, &self.b_out)
    }
}

// ---------------------------------------------------------------------------
// Instance normalization (RevIN-like)
// ---------------------------------------------------------------------------

/// Per-channel instance normalization statistics.
struct InstanceNormStats<F: Float> {
    mean: F,
    std_dev: F,
}

/// Compute per-channel mean and std for reversible instance normalization.
fn instance_norm_forward<F: Float + FromPrimitive>(
    series: &Array1<F>,
) -> (Array1<F>, InstanceNormStats<F>) {
    let n = series.len();
    let n_f = F::from(n as f64).unwrap_or_else(|| F::one());
    let eps = F::from(1e-5).unwrap_or_else(|| F::zero());

    let mean = series.iter().cloned().fold(F::zero(), |a, b| a + b) / n_f;
    let var = series.iter().cloned().fold(F::zero(), |a, v| {
        let d = v - mean;
        a + d * d
    }) / n_f;
    let std_dev = (var + eps).sqrt();

    let normed = series.mapv(|v| (v - mean) / std_dev);
    (normed, InstanceNormStats { mean, std_dev })
}

/// Reverse instance normalization on prediction.
fn instance_norm_reverse<F: Float>(pred: &Array1<F>, stats: &InstanceNormStats<F>) -> Array1<F> {
    pred.mapv(|v| v * stats.std_dev + stats.mean)
}

// ---------------------------------------------------------------------------
// PatchTST model
// ---------------------------------------------------------------------------

/// PatchTST model for multivariate time series forecasting.
///
/// Processes each channel independently through a shared transformer backbone
/// (in channel-independent mode) or jointly (in channel-dependent mode).
///
/// Input shape: `[batch, n_channels, seq_len]`
/// Output shape: `[batch, n_channels, pred_len]`
#[derive(Debug)]
pub struct PatchTSTModel<F: Float + Debug> {
    config: PatchTSTConfig,
    num_patches: usize,
    patch_embed: PatchEmbedding<F>,
    pos_enc: SinusoidalPE<F>,
    encoder_layers: Vec<TransformerEncoderLayer<F>>,
    head: FlattenHead<F>,
    /// Final layer norm
    ln_gamma: Array1<F>,
    ln_beta: Array1<F>,
}

impl<F: Float + FromPrimitive + Debug> PatchTSTModel<F> {
    /// Create a new PatchTST model from configuration.
    ///
    /// # Errors
    ///
    /// Returns error if configuration parameters are invalid (e.g., d_model not
    /// divisible by n_heads, zero-length dimensions).
    pub fn new(config: PatchTSTConfig) -> Result<Self> {
        if config.seq_len == 0 || config.pred_len == 0 {
            return Err(TimeSeriesError::InvalidInput(
                "seq_len and pred_len must be positive".to_string(),
            ));
        }
        if config.patch_len == 0 || config.stride == 0 {
            return Err(TimeSeriesError::InvalidInput(
                "patch_len and stride must be positive".to_string(),
            ));
        }
        if config.d_model == 0 || config.n_heads == 0 {
            return Err(TimeSeriesError::InvalidInput(
                "d_model and n_heads must be positive".to_string(),
            ));
        }
        if config.d_model % config.n_heads != 0 {
            return Err(TimeSeriesError::InvalidInput(
                "d_model must be divisible by n_heads".to_string(),
            ));
        }
        if config.n_channels == 0 {
            return Err(TimeSeriesError::InvalidInput(
                "n_channels must be positive".to_string(),
            ));
        }

        let num_patches = compute_num_patches(config.seq_len, config.patch_len, config.stride);
        let seed = config.seed;
        let dm = config.d_model;

        let patch_embed = PatchEmbedding::new(config.patch_len, config.stride, dm, seed);
        let pos_enc = SinusoidalPE::new(num_patches, dm);

        let mut encoder_layers = Vec::with_capacity(config.n_layers);
        for i in 0..config.n_layers {
            encoder_layers.push(TransformerEncoderLayer::new(
                dm,
                config.n_heads,
                config.d_ff,
                seed.wrapping_add(1000 + i as u32 * 500),
            )?);
        }

        let head = FlattenHead::new(num_patches, dm, config.pred_len, seed.wrapping_add(8000));

        Ok(Self {
            config,
            num_patches,
            patch_embed,
            pos_enc,
            encoder_layers,
            head,
            ln_gamma: Array1::ones(dm),
            ln_beta: Array1::zeros(dm),
        })
    }

    /// Forward pass for a single channel's time series.
    ///
    /// Input: shape `(seq_len,)` -> Output: shape `(pred_len,)`.
    fn forward_channel(&self, channel_data: &Array1<F>) -> Result<Array1<F>> {
        // Instance normalization
        let (normed, stats) = instance_norm_forward(channel_data);

        // Patch embedding: (seq_len,) -> (num_patches, d_model)
        let patches = self.patch_embed.forward(&normed);

        // Add positional encoding
        let positioned = self.pos_enc.add(&patches);

        // Transformer encoder
        let mut hidden = positioned;
        for layer in &self.encoder_layers {
            hidden = layer.forward(&hidden)?;
        }

        // Final layer norm
        let normed_out = nn_utils::layer_norm(&hidden, &self.ln_gamma, &self.ln_beta);

        // Flatten head -> prediction
        let pred = self.head.forward(&normed_out);

        // Reverse instance normalization
        let denormed = instance_norm_reverse(&pred, &stats);

        Ok(denormed)
    }

    /// Forward pass for a batch of multivariate time series.
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor of shape `[batch, n_channels, seq_len]`.
    ///
    /// # Returns
    ///
    /// Output tensor of shape `[batch, n_channels, pred_len]`.
    ///
    /// # Errors
    ///
    /// Returns error if input dimensions don't match the configuration.
    pub fn forward(&self, x: &Array3<F>) -> Result<Array3<F>> {
        let (batch, n_ch, sl) = x.dim();

        if sl != self.config.seq_len {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: self.config.seq_len,
                actual: sl,
            });
        }

        let expected_channels = if self.config.channel_independent {
            // In CI mode, any number of channels is fine (we iterate them)
            n_ch
        } else {
            self.config.n_channels
        };

        if !self.config.channel_independent && n_ch != self.config.n_channels {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: self.config.n_channels,
                actual: n_ch,
            });
        }

        let mut output = Array3::zeros((batch, expected_channels, self.config.pred_len));

        for b in 0..batch {
            for ch in 0..expected_channels {
                // Extract channel data: shape (seq_len,)
                let mut channel_data = Array1::zeros(sl);
                for t in 0..sl {
                    channel_data[t] = x[[b, ch, t]];
                }

                let pred = self.forward_channel(&channel_data)?;

                for t in 0..self.config.pred_len {
                    output[[b, ch, t]] = pred[t];
                }
            }
        }

        Ok(output)
    }

    /// Get the number of patches computed from the configuration.
    pub fn num_patches(&self) -> usize {
        self.num_patches
    }

    /// Get a reference to the configuration.
    pub fn config(&self) -> &PatchTSTConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array3;

    #[test]
    fn test_default_config_produces_valid_model() {
        let config = PatchTSTConfig::default();
        let model = PatchTSTModel::<f64>::new(config);
        assert!(model.is_ok());
    }

    #[test]
    fn test_forward_shape() {
        let config = PatchTSTConfig {
            seq_len: 96,
            pred_len: 24,
            n_channels: 3,
            patch_len: 16,
            stride: 8,
            d_model: 32,
            n_heads: 4,
            n_layers: 2,
            d_ff: 64,
            dropout: 0.0,
            channel_independent: true,
            seed: 42,
        };
        let model = PatchTSTModel::<f64>::new(config).expect("model creation failed");

        let x = Array3::zeros((2, 3, 96));
        let out = model.forward(&x).expect("forward failed");
        assert_eq!(out.dim(), (2, 3, 24));
    }

    #[test]
    fn test_patch_count() {
        // num_patches = ceil((seq_len - patch_len) / stride) + 1
        // = (96 - 16) / 8 + 1 = 10 + 1 = 11
        assert_eq!(compute_num_patches(96, 16, 8), 11);
        // seq_len == patch_len -> 1 patch
        assert_eq!(compute_num_patches(16, 16, 8), 1);
        // seq_len < patch_len -> 1 patch
        assert_eq!(compute_num_patches(10, 16, 8), 1);
        // non-overlapping: stride == patch_len
        assert_eq!(compute_num_patches(64, 16, 16), 4);
    }

    #[test]
    fn test_channel_independence() {
        let config = PatchTSTConfig {
            seq_len: 32,
            pred_len: 8,
            n_channels: 2,
            patch_len: 8,
            stride: 4,
            d_model: 16,
            n_heads: 4,
            n_layers: 1,
            d_ff: 32,
            dropout: 0.0,
            channel_independent: true,
            seed: 42,
        };
        let model = PatchTSTModel::<f64>::new(config).expect("model creation failed");

        // Create input where channel 0 has ones and channel 1 has twos
        let mut x1 = Array3::zeros((1, 2, 32));
        for t in 0..32 {
            x1[[0, 0, t]] = 1.0;
            x1[[0, 1, t]] = 2.0;
        }
        let out1 = model.forward(&x1).expect("forward failed");

        // Modify channel 1 and verify channel 0 output is unchanged
        let mut x2 = x1.clone();
        for t in 0..32 {
            x2[[0, 1, t]] = 5.0;
        }
        let out2 = model.forward(&x2).expect("forward failed");

        // Channel 0 outputs should be identical in CI mode
        for t in 0..8 {
            let diff = (out1[[0, 0, t]] - out2[[0, 0, t]]).abs();
            assert!(
                diff < 1e-10,
                "Channel independence violated at t={}: diff={}",
                t,
                diff
            );
        }
    }

    #[test]
    fn test_single_patch() {
        let config = PatchTSTConfig {
            seq_len: 16,
            pred_len: 4,
            n_channels: 1,
            patch_len: 16,
            stride: 16,
            d_model: 16,
            n_heads: 4,
            n_layers: 1,
            d_ff: 32,
            dropout: 0.0,
            channel_independent: true,
            seed: 42,
        };
        let model = PatchTSTModel::<f64>::new(config).expect("model creation failed");
        assert_eq!(model.num_patches(), 1);

        let x = Array3::zeros((1, 1, 16));
        let out = model.forward(&x).expect("forward failed");
        assert_eq!(out.dim(), (1, 1, 4));
    }

    #[test]
    fn test_zero_input_finite_output() {
        let config = PatchTSTConfig {
            seq_len: 32,
            pred_len: 8,
            n_channels: 2,
            patch_len: 8,
            stride: 4,
            d_model: 16,
            n_heads: 4,
            n_layers: 1,
            d_ff: 32,
            dropout: 0.0,
            channel_independent: true,
            seed: 42,
        };
        let model = PatchTSTModel::<f64>::new(config).expect("model creation failed");

        let x = Array3::zeros((1, 2, 32));
        let out = model.forward(&x).expect("forward failed");

        for b in 0..1 {
            for ch in 0..2 {
                for t in 0..8 {
                    assert!(
                        out[[b, ch, t]].is_finite(),
                        "Non-finite output at [{}, {}, {}]",
                        b,
                        ch,
                        t
                    );
                }
            }
        }
    }

    #[test]
    fn test_batch_dimension_preserved() {
        let config = PatchTSTConfig {
            seq_len: 32,
            pred_len: 8,
            n_channels: 3,
            patch_len: 8,
            stride: 8,
            d_model: 16,
            n_heads: 4,
            n_layers: 1,
            d_ff: 32,
            dropout: 0.0,
            channel_independent: true,
            seed: 42,
        };
        let model = PatchTSTModel::<f64>::new(config).expect("model creation failed");

        for batch_size in [1, 4, 8] {
            let x = Array3::zeros((batch_size, 3, 32));
            let out = model.forward(&x).expect("forward failed");
            assert_eq!(
                out.dim().0,
                batch_size,
                "batch size {} not preserved",
                batch_size
            );
        }
    }

    #[test]
    fn test_invalid_config_errors() {
        // d_model not divisible by n_heads
        let config = PatchTSTConfig {
            d_model: 33,
            n_heads: 4,
            ..PatchTSTConfig::default()
        };
        assert!(PatchTSTModel::<f64>::new(config).is_err());

        // zero seq_len
        let config = PatchTSTConfig {
            seq_len: 0,
            ..PatchTSTConfig::default()
        };
        assert!(PatchTSTModel::<f64>::new(config).is_err());
    }

    #[test]
    fn test_dimension_mismatch_error() {
        let config = PatchTSTConfig {
            seq_len: 32,
            pred_len: 8,
            n_channels: 2,
            patch_len: 8,
            stride: 8,
            d_model: 16,
            n_heads: 4,
            n_layers: 1,
            d_ff: 32,
            dropout: 0.0,
            channel_independent: true,
            seed: 42,
        };
        let model = PatchTSTModel::<f64>::new(config).expect("model creation failed");

        // Wrong seq_len
        let x = Array3::zeros((1, 2, 64));
        assert!(model.forward(&x).is_err());
    }
}
