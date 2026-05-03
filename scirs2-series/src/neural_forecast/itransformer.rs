//! iTransformer: Inverted Transformer for Time Series Forecasting
//!
//! Implementation of *"iTransformer: Inverted Transformers Are Effective for Time Series
//! Forecasting"* (Liu et al., 2024). The key insight is to **invert** the roles of
//! tokens and features:
//!
//! - In standard transformers, each time step is a token and features are embedded.
//! - In iTransformer, each **variate** (channel) is a token, and its entire time series
//!   is the feature that gets embedded.
//!
//! This means:
//! - **Attention** captures cross-variate (multivariate) correlations.
//! - **Feed-forward networks** learn temporal patterns per variate.
//!
//! Architecture: Variate Embedding (seq_len → d_model per channel) → Transformer Encoder
//! (tokens = variates) → Projection Head (d_model → pred_len per channel).

use scirs2_core::ndarray::{Array1, Array2, Array3};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::Debug;

use super::nn_utils;
use crate::error::{Result, TimeSeriesError};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the iTransformer model.
#[derive(Debug, Clone)]
pub struct ITransformerConfig {
    /// Input sequence length.
    pub seq_len: usize,
    /// Prediction horizon.
    pub pred_len: usize,
    /// Number of variates (channels). Each variate becomes one token.
    pub n_channels: usize,
    /// Transformer model dimension.
    pub d_model: usize,
    /// Number of attention heads.
    pub n_heads: usize,
    /// Number of transformer encoder layers.
    pub n_layers: usize,
    /// Feed-forward hidden dimension.
    pub d_ff: usize,
    /// Dropout probability (conceptual; deterministic in inference).
    pub dropout: f64,
    /// Random seed for weight initialization.
    pub seed: u32,
}

impl Default for ITransformerConfig {
    fn default() -> Self {
        Self {
            seq_len: 96,
            pred_len: 24,
            n_channels: 7,
            d_model: 64,
            n_heads: 4,
            n_layers: 3,
            d_ff: 128,
            dropout: 0.1,
            seed: 42,
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

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
// Variate embedding
// ---------------------------------------------------------------------------

/// Embeds each variate's full time series into the model dimension.
///
/// For each variate (token), projects from `seq_len` to `d_model`.
#[derive(Debug)]
struct VariateEmbedding<F: Float> {
    /// Linear projection: (d_model, seq_len)
    w: Array2<F>,
    b: Array1<F>,
}

impl<F: Float + FromPrimitive + Debug> VariateEmbedding<F> {
    fn new(seq_len: usize, d_model: usize, seed: u32) -> Self {
        Self {
            w: nn_utils::xavier_matrix(d_model, seq_len, seed),
            b: nn_utils::zero_bias(d_model),
        }
    }

    /// Embed variates: input `(n_channels, seq_len)` -> `(n_channels, d_model)`.
    fn forward(&self, x: &Array2<F>) -> Array2<F> {
        nn_utils::dense_forward(x, &self.w, &self.b)
    }
}

// ---------------------------------------------------------------------------
// Multi-head self-attention
// ---------------------------------------------------------------------------

/// Multi-head self-attention for variate tokens.
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

    /// Self-attention: input `(n_tokens, d_model)` -> `(n_tokens, d_model)`.
    ///
    /// In iTransformer, tokens are variates and attention captures cross-variate correlations.
    fn forward(&self, input: &Array2<F>) -> Result<Array2<F>> {
        let (n_tokens, _) = input.dim();
        let b_zero = nn_utils::zero_bias::<F>(self.d_model);
        let q = nn_utils::dense_forward(input, &self.w_q, &b_zero);
        let k = nn_utils::dense_forward(input, &self.w_k, &b_zero);
        let v = nn_utils::dense_forward(input, &self.w_v, &b_zero);

        let scale = F::from(self.head_dim as f64)
            .unwrap_or_else(|| F::one())
            .sqrt();

        let mut concat_out = Array2::zeros((n_tokens, self.d_model));

        for h in 0..self.n_heads {
            let offset = h * self.head_dim;
            let mut scores = Array2::zeros((n_tokens, n_tokens));
            for i in 0..n_tokens {
                for j in 0..n_tokens {
                    let mut dot = F::zero();
                    for d in 0..self.head_dim {
                        dot = dot + q[[i, offset + d]] * k[[j, offset + d]];
                    }
                    scores[[i, j]] = dot / scale;
                }
            }
            let attn = nn_utils::softmax_rows(&scores);

            for i in 0..n_tokens {
                for d in 0..self.head_dim {
                    let mut acc = F::zero();
                    for j in 0..n_tokens {
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

/// Encoder layer: self-attention + FFN + layer norms + residuals.
///
/// In iTransformer, the FFN operates on each variate's embedding independently,
/// learning temporal patterns.
#[derive(Debug)]
struct EncoderLayer<F: Float> {
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

impl<F: Float + FromPrimitive + Debug> EncoderLayer<F> {
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
        // Pre-norm variant
        let normed1 = nn_utils::layer_norm(input, &self.ln1_gamma, &self.ln1_beta);
        let attn_out = self.self_attn.forward(&normed1)?;
        let residual1 = add_2d(input, &attn_out);

        let normed2 = nn_utils::layer_norm(&residual1, &self.ln2_gamma, &self.ln2_beta);
        let h = gelu_2d(&nn_utils::dense_forward(&normed2, &self.w1, &self.b1));
        let ffn_out = nn_utils::dense_forward(&h, &self.w2, &self.b2);
        let residual2 = add_2d(&residual1, &ffn_out);

        Ok(residual2)
    }
}

// ---------------------------------------------------------------------------
// Projection head
// ---------------------------------------------------------------------------

/// Projects from d_model back to pred_len for each variate.
#[derive(Debug)]
struct ProjectionHead<F: Float> {
    /// Linear: (pred_len, d_model)
    w: Array2<F>,
    b: Array1<F>,
}

impl<F: Float + FromPrimitive + Debug> ProjectionHead<F> {
    fn new(d_model: usize, pred_len: usize, seed: u32) -> Self {
        Self {
            w: nn_utils::xavier_matrix(pred_len, d_model, seed),
            b: nn_utils::zero_bias(pred_len),
        }
    }

    /// Project: `(n_channels, d_model)` -> `(n_channels, pred_len)`.
    fn forward(&self, x: &Array2<F>) -> Array2<F> {
        nn_utils::dense_forward(x, &self.w, &self.b)
    }
}

// ---------------------------------------------------------------------------
// Instance normalization
// ---------------------------------------------------------------------------

/// Per-channel normalization statistics.
struct ChannelStats<F: Float> {
    means: Array1<F>,
    stds: Array1<F>,
}

/// Instance normalize each channel of the input.
///
/// Input: `(n_channels, seq_len)` -> normalized + stats for reversal.
fn instance_normalize<F: Float + FromPrimitive>(x: &Array2<F>) -> (Array2<F>, ChannelStats<F>) {
    let (n_ch, seq_len) = x.dim();
    let eps = F::from(1e-5).unwrap_or_else(|| F::zero());
    let sl_f = F::from(seq_len as f64).unwrap_or_else(|| F::one());
    let mut normed = Array2::zeros((n_ch, seq_len));
    let mut means = Array1::zeros(n_ch);
    let mut stds = Array1::zeros(n_ch);

    for ch in 0..n_ch {
        let mut mean = F::zero();
        for t in 0..seq_len {
            mean = mean + x[[ch, t]];
        }
        mean = mean / sl_f;

        let mut var = F::zero();
        for t in 0..seq_len {
            let d = x[[ch, t]] - mean;
            var = var + d * d;
        }
        var = var / sl_f;
        let std = (var + eps).sqrt();

        means[ch] = mean;
        stds[ch] = std;

        for t in 0..seq_len {
            normed[[ch, t]] = (x[[ch, t]] - mean) / std;
        }
    }

    (normed, ChannelStats { means, stds })
}

/// Reverse instance normalization on predictions.
fn instance_denormalize<F: Float>(pred: &Array2<F>, stats: &ChannelStats<F>) -> Array2<F> {
    let (n_ch, pred_len) = pred.dim();
    let mut out = Array2::zeros((n_ch, pred_len));
    for ch in 0..n_ch {
        for t in 0..pred_len {
            out[[ch, t]] = pred[[ch, t]] * stats.stds[ch] + stats.means[ch];
        }
    }
    out
}

// ---------------------------------------------------------------------------
// iTransformer model
// ---------------------------------------------------------------------------

/// iTransformer model for multivariate time series forecasting.
///
/// Inverts the standard transformer paradigm: variates are tokens, and their
/// full time series are features. Attention captures cross-variate correlations
/// while FFN learns temporal patterns.
///
/// Input shape: `[batch, n_channels, seq_len]`
/// Output shape: `[batch, n_channels, pred_len]`
#[derive(Debug)]
pub struct ITransformerModel<F: Float + Debug> {
    config: ITransformerConfig,
    variate_embed: VariateEmbedding<F>,
    encoder_layers: Vec<EncoderLayer<F>>,
    projection: ProjectionHead<F>,
    /// Final layer norm
    ln_gamma: Array1<F>,
    ln_beta: Array1<F>,
}

impl<F: Float + FromPrimitive + Debug> ITransformerModel<F> {
    /// Create a new iTransformer model from configuration.
    ///
    /// # Errors
    ///
    /// Returns error if configuration parameters are invalid.
    pub fn new(config: ITransformerConfig) -> Result<Self> {
        if config.seq_len == 0 || config.pred_len == 0 {
            return Err(TimeSeriesError::InvalidInput(
                "seq_len and pred_len must be positive".to_string(),
            ));
        }
        if config.n_channels == 0 {
            return Err(TimeSeriesError::InvalidInput(
                "n_channels must be positive".to_string(),
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

        let seed = config.seed;
        let dm = config.d_model;

        let variate_embed = VariateEmbedding::new(config.seq_len, dm, seed);

        let mut encoder_layers = Vec::with_capacity(config.n_layers);
        for i in 0..config.n_layers {
            encoder_layers.push(EncoderLayer::new(
                dm,
                config.n_heads,
                config.d_ff,
                seed.wrapping_add(1000 + i as u32 * 500),
            )?);
        }

        let projection = ProjectionHead::new(dm, config.pred_len, seed.wrapping_add(8000));

        Ok(Self {
            config,
            variate_embed,
            encoder_layers,
            projection,
            ln_gamma: Array1::ones(dm),
            ln_beta: Array1::zeros(dm),
        })
    }

    /// Forward pass for a single sample.
    ///
    /// Input: `(n_channels, seq_len)` -> Output: `(n_channels, pred_len)`.
    fn forward_single(&self, x: &Array2<F>) -> Result<Array2<F>> {
        let (n_ch, _sl) = x.dim();

        // Instance normalization per channel
        let (normed, stats) = instance_normalize(x);

        // Variate embedding: (n_channels, seq_len) -> (n_channels, d_model)
        // Each channel's full time series is projected to d_model
        let embedded = self.variate_embed.forward(&normed);

        // Transformer encoder: tokens = variates
        let mut hidden = embedded;
        for layer in &self.encoder_layers {
            hidden = layer.forward(&hidden)?;
        }

        // Final layer norm
        let normed_out = nn_utils::layer_norm(&hidden, &self.ln_gamma, &self.ln_beta);

        // Projection: (n_channels, d_model) -> (n_channels, pred_len)
        let pred = self.projection.forward(&normed_out);

        // Reverse instance normalization
        let denormed = instance_denormalize(&pred, &stats);

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
        if n_ch != self.config.n_channels {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: self.config.n_channels,
                actual: n_ch,
            });
        }

        let mut output = Array3::zeros((batch, n_ch, self.config.pred_len));

        for b in 0..batch {
            // Extract sample: (n_channels, seq_len)
            let mut sample = Array2::zeros((n_ch, sl));
            for ch in 0..n_ch {
                for t in 0..sl {
                    sample[[ch, t]] = x[[b, ch, t]];
                }
            }

            let pred = self.forward_single(&sample)?;

            for ch in 0..n_ch {
                for t in 0..self.config.pred_len {
                    output[[b, ch, t]] = pred[[ch, t]];
                }
            }
        }

        Ok(output)
    }

    /// Get a reference to the configuration.
    pub fn config(&self) -> &ITransformerConfig {
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
        let config = ITransformerConfig::default();
        let model = ITransformerModel::<f64>::new(config);
        assert!(model.is_ok());
    }

    #[test]
    fn test_forward_shape() {
        let config = ITransformerConfig {
            seq_len: 96,
            pred_len: 24,
            n_channels: 5,
            d_model: 32,
            n_heads: 4,
            n_layers: 2,
            d_ff: 64,
            dropout: 0.0,
            seed: 42,
        };
        let model = ITransformerModel::<f64>::new(config).expect("model creation failed");

        let x = Array3::zeros((2, 5, 96));
        let out = model.forward(&x).expect("forward failed");
        assert_eq!(out.dim(), (2, 5, 24));
    }

    #[test]
    fn test_variate_attention_cross_channel() {
        // In iTransformer, changing one channel SHOULD affect others
        // because attention is across variates.
        let config = ITransformerConfig {
            seq_len: 32,
            pred_len: 8,
            n_channels: 3,
            d_model: 16,
            n_heads: 4,
            n_layers: 1,
            d_ff: 32,
            dropout: 0.0,
            seed: 42,
        };
        let model = ITransformerModel::<f64>::new(config).expect("model creation failed");

        // Input 1: channels have distinct patterns so instance norm preserves differences
        let mut x1 = Array3::zeros((1, 3, 32));
        for t in 0..32 {
            x1[[0, 0, t]] = (t as f64) * 0.1; // linear ramp
            x1[[0, 1, t]] = (t as f64 * 0.3).sin(); // sine
            x1[[0, 2, t]] = 1.0; // constant
        }
        let out1 = model.forward(&x1).expect("forward failed");

        // Input 2: change channel 2 from constant to different pattern
        let mut x2 = x1.clone();
        for t in 0..32 {
            x2[[0, 2, t]] = (t as f64) * 0.5; // steeper ramp
        }
        let out2 = model.forward(&x2).expect("forward failed");

        // Channel 0 output should differ because attention is cross-variate
        let mut any_diff = false;
        for t in 0..8 {
            let diff = (out1[[0, 0, t]] - out2[[0, 0, t]]).abs();
            if diff > 1e-6 {
                any_diff = true;
                break;
            }
        }
        assert!(
            any_diff,
            "Expected cross-channel interaction but outputs were identical"
        );
    }

    #[test]
    fn test_single_channel() {
        let config = ITransformerConfig {
            seq_len: 48,
            pred_len: 12,
            n_channels: 1,
            d_model: 16,
            n_heads: 4,
            n_layers: 1,
            d_ff: 32,
            dropout: 0.0,
            seed: 42,
        };
        let model = ITransformerModel::<f64>::new(config).expect("model creation failed");

        let mut x = Array3::zeros((1, 1, 48));
        for t in 0..48 {
            x[[0, 0, t]] = (t as f64) * 0.1;
        }
        let out = model.forward(&x).expect("forward failed");
        assert_eq!(out.dim(), (1, 1, 12));

        // Output should be finite
        for t in 0..12 {
            assert!(out[[0, 0, t]].is_finite());
        }
    }

    #[test]
    fn test_zero_input_finite_output() {
        let config = ITransformerConfig {
            seq_len: 32,
            pred_len: 8,
            n_channels: 3,
            d_model: 16,
            n_heads: 4,
            n_layers: 1,
            d_ff: 32,
            dropout: 0.0,
            seed: 42,
        };
        let model = ITransformerModel::<f64>::new(config).expect("model creation failed");

        let x = Array3::zeros((1, 3, 32));
        let out = model.forward(&x).expect("forward failed");

        for ch in 0..3 {
            for t in 0..8 {
                assert!(
                    out[[0, ch, t]].is_finite(),
                    "Non-finite at ch={}, t={}",
                    ch,
                    t
                );
            }
        }
    }

    #[test]
    fn test_batch_dimension_preserved() {
        let config = ITransformerConfig {
            seq_len: 32,
            pred_len: 8,
            n_channels: 2,
            d_model: 16,
            n_heads: 4,
            n_layers: 1,
            d_ff: 32,
            dropout: 0.0,
            seed: 42,
        };
        let model = ITransformerModel::<f64>::new(config).expect("model creation failed");

        for batch_size in [1, 4, 8] {
            let x = Array3::zeros((batch_size, 2, 32));
            let out = model.forward(&x).expect("forward failed");
            assert_eq!(out.dim().0, batch_size);
        }
    }

    #[test]
    fn test_invalid_config_errors() {
        let config = ITransformerConfig {
            d_model: 33,
            n_heads: 4,
            ..ITransformerConfig::default()
        };
        assert!(ITransformerModel::<f64>::new(config).is_err());

        let config = ITransformerConfig {
            n_channels: 0,
            ..ITransformerConfig::default()
        };
        assert!(ITransformerModel::<f64>::new(config).is_err());
    }

    #[test]
    fn test_dimension_mismatch_error() {
        let config = ITransformerConfig {
            seq_len: 32,
            pred_len: 8,
            n_channels: 3,
            d_model: 16,
            n_heads: 4,
            n_layers: 1,
            d_ff: 32,
            dropout: 0.0,
            seed: 42,
        };
        let model = ITransformerModel::<f64>::new(config).expect("model creation failed");

        // Wrong seq_len
        let x = Array3::zeros((1, 3, 64));
        assert!(model.forward(&x).is_err());

        // Wrong n_channels
        let x = Array3::zeros((1, 5, 32));
        assert!(model.forward(&x).is_err());
    }
}
