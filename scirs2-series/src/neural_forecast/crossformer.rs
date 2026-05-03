//! Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate
//! Time Series Forecasting.
//!
//! Implementation of *"Crossformer: Transformer Utilizing Cross-Dimension Dependency for
//! Multivariate Time Series Forecasting"* (Zhang & Yan, 2022).
//!
//! Key innovations:
//!
//! - **Segment Merging**: Divides the time series into segments (patches), allowing the
//!   model to capture local temporal patterns while reducing sequence length.
//!
//! - **Cross-Time Stage**: Self-attention over time segments within each variate (dimension),
//!   capturing temporal dependencies at the segment level.
//!
//! - **Cross-Dimension Stage**: Router-based cross-attention across variates for each time
//!   segment. A small set of router tokens (router_size << n_channels) aggregate information
//!   from all dimensions, then redistribute it back — achieving O(D·R) complexity instead
//!   of O(D²).
//!
//! Architecture:
//! ```text
//! Input [L, D] → Segment Merging → (n_segs, D, d_model)
//!              → N × CrossformerLayer:
//!                  cross_time: per-dim self-attn across segments
//!                  cross_dim: router-based cross-attn across dims
//!              → Linear projection → [pred_len, D]
//! ```

use scirs2_core::ndarray::{Array1, Array2, Array3};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::Debug;

use super::nn_utils;
use crate::error::{Result, TimeSeriesError};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the Crossformer model.
#[derive(Debug, Clone)]
pub struct CrossformerConfig {
    /// Input sequence length.
    pub seq_len: usize,
    /// Prediction horizon.
    pub pred_len: usize,
    /// Number of input variates (dimensions/channels).
    pub n_channels: usize,
    /// Segment length: each segment covers this many time steps.
    pub seg_len: usize,
    /// Model embedding dimension.
    pub d_model: usize,
    /// Number of attention heads for cross-time and cross-dim attention.
    pub n_heads: usize,
    /// Number of Crossformer layers.
    pub n_layers: usize,
    /// Number of router tokens for cross-dimension attention.
    pub router_size: usize,
    /// Feed-forward hidden dimension.
    pub d_ff: usize,
    /// Random seed for weight initialization.
    pub seed: u32,
}

impl Default for CrossformerConfig {
    fn default() -> Self {
        Self {
            seq_len: 96,
            pred_len: 24,
            n_channels: 7,
            seg_len: 6,
            d_model: 64,
            n_heads: 4,
            n_layers: 2,
            router_size: 10,
            d_ff: 128,
            seed: 42,
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute number of segments for given seq_len and seg_len.
///
/// Uses ceiling division so the last segment may be padded.
fn num_segments(seq_len: usize, seg_len: usize) -> usize {
    (seq_len + seg_len - 1) / seg_len
}

/// Element-wise add for 2D arrays of matching shape.
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

/// GELU approximation for 2D arrays.
fn gelu_2d<F: Float + FromPrimitive>(x: &Array2<F>) -> Array2<F> {
    let half = F::from(0.5).unwrap_or_else(|| F::zero());
    let sqrt_2_pi = F::from(0.7978845608).unwrap_or_else(|| F::one());
    let coeff = F::from(0.044715).unwrap_or_else(|| F::zero());
    x.mapv(|v| half * v * (F::one() + (sqrt_2_pi * (v + coeff * v * v * v)).tanh()))
}

// ---------------------------------------------------------------------------
// Segment Merging
// ---------------------------------------------------------------------------

/// Segment Merging: divides time series into segments and embeds each segment.
///
/// For an input `[L, D]`, produces a `(n_segs, D, d_model)` tensor where
/// each segment for each dimension is projected to `d_model`.
#[derive(Debug)]
pub struct SegmentMerging<F: Float + Debug> {
    seg_len: usize,
    /// Projection weight: (d_model, seg_len)
    w_proj: Array2<F>,
    b_proj: Array1<F>,
}

impl<F: Float + FromPrimitive + Debug> SegmentMerging<F> {
    /// Create a new SegmentMerging layer.
    pub fn new(seg_len: usize, d_model: usize, seed: u32) -> Self {
        Self {
            seg_len,
            w_proj: nn_utils::xavier_matrix(d_model, seg_len, seed),
            b_proj: nn_utils::zero_bias(d_model),
        }
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// * `input` - shape `[seq_len, n_channels]`
    ///
    /// # Returns
    /// Tensor of shape `(n_segs, n_channels, d_model)`.
    pub fn forward(&self, input: &Array2<F>) -> Array3<F> {
        let (seq_len, n_ch) = input.dim();
        let n_segs = num_segments(seq_len, self.seg_len);
        let d_model = self.w_proj.nrows();

        let mut output = Array3::zeros((n_segs, n_ch, d_model));

        for ch in 0..n_ch {
            for s in 0..n_segs {
                // Extract segment (with zero padding if needed)
                let mut seg_vec: Array1<F> = Array1::zeros(self.seg_len);
                for k in 0..self.seg_len {
                    let t = s * self.seg_len + k;
                    if t < seq_len {
                        seg_vec[k] = input[[t, ch]];
                    }
                }
                // Project segment to d_model
                let embedded = nn_utils::dense_forward_vec(&seg_vec, &self.w_proj, &self.b_proj);
                for d in 0..d_model {
                    output[[s, ch, d]] = embedded[d];
                }
            }
        }

        output
    }
}

// ---------------------------------------------------------------------------
// Cross-Time Attention
// ---------------------------------------------------------------------------

/// Cross-Time Attention: self-attention for each dimension independently across
/// the time segment axis.
///
/// For each dimension `d`, treats the `n_segs` segment embeddings as a sequence
/// and runs scaled dot-product self-attention with multi-head support.
#[derive(Debug)]
pub struct CrossTimeAttention<F: Float + Debug> {
    d_model: usize,
    n_heads: usize,
    head_dim: usize,
    // Shared Q/K/V projections across all dimensions
    w_q: Array2<F>,
    w_k: Array2<F>,
    w_v: Array2<F>,
    w_o: Array2<F>,
    b_o: Array1<F>,
}

impl<F: Float + FromPrimitive + Debug> CrossTimeAttention<F> {
    /// Create a new CrossTimeAttention layer.
    pub fn new(d_model: usize, n_heads: usize, seed: u32) -> Result<Self> {
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

    /// Forward pass.
    ///
    /// # Arguments
    /// * `x` - Input of shape `(n_segs, n_channels, d_model)`
    ///
    /// # Returns
    /// Output of same shape `(n_segs, n_channels, d_model)`.
    pub fn forward(&self, x: &Array3<F>) -> Result<Array3<F>> {
        let (n_segs, n_ch, d_model) = x.dim();
        if d_model != self.d_model {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: self.d_model,
                actual: d_model,
            });
        }

        let mut output = Array3::zeros((n_segs, n_ch, d_model));
        let zero_bias = nn_utils::zero_bias::<F>(d_model);
        let scale = F::from(self.head_dim as f64)
            .unwrap_or_else(|| F::one())
            .sqrt();

        // Process each dimension (channel) independently
        for ch in 0..n_ch {
            // Extract [n_segs, d_model] for this channel
            let mut ch_seq: Array2<F> = Array2::zeros((n_segs, d_model));
            for s in 0..n_segs {
                for d in 0..d_model {
                    ch_seq[[s, d]] = x[[s, ch, d]];
                }
            }

            // Compute Q, K, V projections
            let q = nn_utils::dense_forward(&ch_seq, &self.w_q, &zero_bias);
            let k = nn_utils::dense_forward(&ch_seq, &self.w_k, &zero_bias);
            let v = nn_utils::dense_forward(&ch_seq, &self.w_v, &zero_bias);

            let mut concat_out: Array2<F> = Array2::zeros((n_segs, d_model));

            for h in 0..self.n_heads {
                let offset = h * self.head_dim;
                // Compute attention scores
                let mut scores: Array2<F> = Array2::zeros((n_segs, n_segs));
                for i in 0..n_segs {
                    for j in 0..n_segs {
                        let mut dot = F::zero();
                        for dd in 0..self.head_dim {
                            dot = dot + q[[i, offset + dd]] * k[[j, offset + dd]];
                        }
                        scores[[i, j]] = dot / scale;
                    }
                }
                let attn = nn_utils::softmax_rows(&scores);

                // Weighted sum of values
                for i in 0..n_segs {
                    for dd in 0..self.head_dim {
                        let mut acc = F::zero();
                        for j in 0..n_segs {
                            acc = acc + attn[[i, j]] * v[[j, offset + dd]];
                        }
                        concat_out[[i, offset + dd]] = acc;
                    }
                }
            }

            // Output projection
            let proj_out = nn_utils::dense_forward(&concat_out, &self.w_o, &self.b_o);

            // Write back
            for s in 0..n_segs {
                for d in 0..d_model {
                    output[[s, ch, d]] = proj_out[[s, d]];
                }
            }
        }

        Ok(output)
    }
}

// ---------------------------------------------------------------------------
// Cross-Dimension Attention (Router-based)
// ---------------------------------------------------------------------------

/// Cross-Dimension Attention with router mechanism.
///
/// For each time segment, cross-dimension attention is performed:
/// 1. All dimension embeddings are projected to a small set of router tokens
///    (router_size << n_channels) — reducing O(D²) to O(D·R).
/// 2. Each dimension then attends to the router tokens to aggregate
///    cross-dimensional information.
///
/// Two-stage process:
/// - Stage 1 (D→R): Router tokens attend to all dimension embeddings
/// - Stage 2 (R→D): Each dimension embedding attends to router tokens
#[derive(Debug)]
pub struct CrossDimAttention<F: Float + Debug> {
    d_model: usize,
    n_heads: usize,
    head_dim: usize,
    router_size: usize,
    /// Stage 1: Router queries - shape (router_size, d_model)
    router_queries: Array2<F>,
    /// Stage 1: Keys from dimensions - (d_model, d_model)
    w_k1: Array2<F>,
    /// Stage 1: Values from dimensions - (d_model, d_model)
    w_v1: Array2<F>,
    /// Stage 2: Query from dimension embeddings - (d_model, d_model)
    w_q2: Array2<F>,
    /// Stage 2: Keys from router - (d_model, d_model)
    w_k2: Array2<F>,
    /// Stage 2: Values from router - (d_model, d_model)
    w_v2: Array2<F>,
    /// Output projection
    w_o: Array2<F>,
    b_o: Array1<F>,
}

impl<F: Float + FromPrimitive + Debug> CrossDimAttention<F> {
    /// Create a new CrossDimAttention layer.
    pub fn new(d_model: usize, n_heads: usize, router_size: usize, seed: u32) -> Result<Self> {
        if d_model == 0 || n_heads == 0 || router_size == 0 {
            return Err(TimeSeriesError::InvalidInput(
                "d_model, n_heads, and router_size must be positive".to_string(),
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
            router_size,
            router_queries: nn_utils::xavier_matrix(router_size, d_model, seed),
            w_k1: nn_utils::xavier_matrix(d_model, d_model, seed.wrapping_add(100)),
            w_v1: nn_utils::xavier_matrix(d_model, d_model, seed.wrapping_add(200)),
            w_q2: nn_utils::xavier_matrix(d_model, d_model, seed.wrapping_add(300)),
            w_k2: nn_utils::xavier_matrix(d_model, d_model, seed.wrapping_add(400)),
            w_v2: nn_utils::xavier_matrix(d_model, d_model, seed.wrapping_add(500)),
            w_o: nn_utils::xavier_matrix(d_model, d_model, seed.wrapping_add(600)),
            b_o: nn_utils::zero_bias(d_model),
        })
    }

    /// Scaled dot-product attention.
    ///
    /// # Arguments
    /// * `q` - Queries of shape `(q_len, d_model)`
    /// * `k` - Keys of shape `(k_len, d_model)`
    /// * `v` - Values of shape `(k_len, d_model)`
    ///
    /// # Returns
    /// Output of shape `(q_len, d_model)`
    fn scaled_dot_product_attention(
        &self,
        q: &Array2<F>,
        k: &Array2<F>,
        v: &Array2<F>,
    ) -> Array2<F> {
        let q_len = q.nrows();
        let k_len = k.nrows();
        let scale = F::from(self.head_dim as f64)
            .unwrap_or_else(|| F::one())
            .sqrt();
        let mut concat_out = Array2::zeros((q_len, self.d_model));

        for h in 0..self.n_heads {
            let offset = h * self.head_dim;
            let mut scores: Array2<F> = Array2::zeros((q_len, k_len));
            for i in 0..q_len {
                for j in 0..k_len {
                    let mut dot = F::zero();
                    for dd in 0..self.head_dim {
                        dot = dot + q[[i, offset + dd]] * k[[j, offset + dd]];
                    }
                    scores[[i, j]] = dot / scale;
                }
            }
            let attn = nn_utils::softmax_rows(&scores);
            for i in 0..q_len {
                for dd in 0..self.head_dim {
                    let mut acc = F::zero();
                    for j in 0..k_len {
                        acc = acc + attn[[i, j]] * v[[j, offset + dd]];
                    }
                    concat_out[[i, offset + dd]] = acc;
                }
            }
        }

        concat_out
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// * `x` - Input of shape `(n_segs, n_channels, d_model)`
    ///
    /// # Returns
    /// Output of same shape `(n_segs, n_channels, d_model)`.
    pub fn forward(&self, x: &Array3<F>) -> Result<Array3<F>> {
        let (n_segs, n_ch, d_model) = x.dim();
        if d_model != self.d_model {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: self.d_model,
                actual: d_model,
            });
        }

        let zero_bias_d = nn_utils::zero_bias::<F>(d_model);
        let zero_bias_r = nn_utils::zero_bias::<F>(d_model);
        let mut output = Array3::zeros((n_segs, n_ch, d_model));

        // Process each time segment independently
        for s in 0..n_segs {
            // Extract dimension embeddings for segment s: shape (n_ch, d_model)
            let mut seg_embeds: Array2<F> = Array2::zeros((n_ch, d_model));
            for ch in 0..n_ch {
                for d in 0..d_model {
                    seg_embeds[[ch, d]] = x[[s, ch, d]];
                }
            }

            // ---------------------------------------------------------------
            // Stage 1: Router tokens attend to dimension embeddings (D → R)
            // Keys and Values come from dimension embeddings
            // Queries are the learned router tokens
            // ---------------------------------------------------------------
            let k1 = nn_utils::dense_forward(&seg_embeds, &self.w_k1, &zero_bias_d);
            let v1 = nn_utils::dense_forward(&seg_embeds, &self.w_v1, &zero_bias_d);
            // Router queries are broadcast: shape (router_size, d_model)
            let router_out = self.scaled_dot_product_attention(&self.router_queries, &k1, &v1);

            // ---------------------------------------------------------------
            // Stage 2: Dimensions attend to router tokens (R → D)
            // Queries come from dimension embeddings, Keys/Values from router
            // ---------------------------------------------------------------
            let q2 = nn_utils::dense_forward(&seg_embeds, &self.w_q2, &zero_bias_d);
            let k2 = nn_utils::dense_forward(&router_out, &self.w_k2, &zero_bias_r);
            let v2 = nn_utils::dense_forward(&router_out, &self.w_v2, &zero_bias_r);
            let dim_out = self.scaled_dot_product_attention(&q2, &k2, &v2);

            // Output projection with residual
            let proj_out = nn_utils::dense_forward(&dim_out, &self.w_o, &self.b_o);

            for ch in 0..n_ch {
                for d in 0..d_model {
                    output[[s, ch, d]] = proj_out[[ch, d]];
                }
            }
        }

        Ok(output)
    }
}

// ---------------------------------------------------------------------------
// Feed-Forward Network (used within CrossformerLayer)
// ---------------------------------------------------------------------------

/// Feed-forward network applied per-token (segment-dimension pair).
#[derive(Debug)]
struct FeedForward<F: Float + Debug> {
    w1: Array2<F>,
    b1: Array1<F>,
    w2: Array2<F>,
    b2: Array1<F>,
}

impl<F: Float + FromPrimitive + Debug> FeedForward<F> {
    fn new(d_model: usize, d_ff: usize, seed: u32) -> Self {
        Self {
            w1: nn_utils::xavier_matrix(d_ff, d_model, seed),
            b1: nn_utils::zero_bias(d_ff),
            w2: nn_utils::xavier_matrix(d_model, d_ff, seed.wrapping_add(100)),
            b2: nn_utils::zero_bias(d_model),
        }
    }

    /// Forward pass on a 2D token matrix of shape `(n_tokens, d_model)`.
    fn forward(&self, x: &Array2<F>) -> Array2<F> {
        let h = gelu_2d(&nn_utils::dense_forward(x, &self.w1, &self.b1));
        nn_utils::dense_forward(&h, &self.w2, &self.b2)
    }
}

// ---------------------------------------------------------------------------
// Crossformer Layer
// ---------------------------------------------------------------------------

/// A single Crossformer layer combining cross-time and cross-dimension attention.
///
/// Each layer:
/// 1. Cross-Time: self-attention across segments for each dimension
/// 2. Cross-Dimension: router-based attention across dimensions for each segment
/// Both stages have residual connections and layer normalization.
#[derive(Debug)]
pub struct CrossformerLayer<F: Float + Debug> {
    cross_time: CrossTimeAttention<F>,
    cross_dim: CrossDimAttention<F>,
    ffn: FeedForward<F>,
    d_model: usize,
    // Layer norm parameters (gamma=1, beta=0 for each of 3 norms)
    ln_gamma1: Array1<F>,
    ln_beta1: Array1<F>,
    ln_gamma2: Array1<F>,
    ln_beta2: Array1<F>,
    ln_gamma3: Array1<F>,
    ln_beta3: Array1<F>,
}

impl<F: Float + FromPrimitive + Debug> CrossformerLayer<F> {
    /// Create a new Crossformer layer.
    pub fn new(
        d_model: usize,
        n_heads: usize,
        router_size: usize,
        d_ff: usize,
        seed: u32,
    ) -> Result<Self> {
        Ok(Self {
            cross_time: CrossTimeAttention::new(d_model, n_heads, seed)?,
            cross_dim: CrossDimAttention::new(
                d_model,
                n_heads,
                router_size,
                seed.wrapping_add(1000),
            )?,
            ffn: FeedForward::new(d_model, d_ff, seed.wrapping_add(2000)),
            d_model,
            ln_gamma1: Array1::ones(d_model),
            ln_beta1: Array1::zeros(d_model),
            ln_gamma2: Array1::ones(d_model),
            ln_beta2: Array1::zeros(d_model),
            ln_gamma3: Array1::ones(d_model),
            ln_beta3: Array1::zeros(d_model),
        })
    }

    /// Apply layer norm to each (segment, channel) token in a 3D tensor.
    fn layer_norm_3d(&self, x: &Array3<F>, gamma: &Array1<F>, beta: &Array1<F>) -> Array3<F> {
        let (n_segs, n_ch, d_model) = x.dim();
        let mut out = Array3::zeros((n_segs, n_ch, d_model));
        let eps = F::from(1e-5).unwrap_or_else(|| F::zero());
        let d_f = F::from(d_model).unwrap_or_else(|| F::one());
        for s in 0..n_segs {
            for ch in 0..n_ch {
                let mut mean = F::zero();
                for d in 0..d_model {
                    mean = mean + x[[s, ch, d]];
                }
                mean = mean / d_f;
                let mut var = F::zero();
                for d in 0..d_model {
                    let diff = x[[s, ch, d]] - mean;
                    var = var + diff * diff;
                }
                var = var / d_f;
                let inv_std = F::one() / (var + eps).sqrt();
                for d in 0..d_model {
                    out[[s, ch, d]] = (x[[s, ch, d]] - mean) * inv_std * gamma[d] + beta[d];
                }
            }
        }
        out
    }

    /// Reshape 3D tensor to 2D for FFN application, then back.
    fn apply_ffn(&self, x: &Array3<F>) -> Array3<F> {
        let (n_segs, n_ch, d_model) = x.dim();
        // Flatten (n_segs * n_ch, d_model)
        let n_tokens = n_segs * n_ch;
        let mut flat: Array2<F> = Array2::zeros((n_tokens, d_model));
        for s in 0..n_segs {
            for ch in 0..n_ch {
                for d in 0..d_model {
                    flat[[s * n_ch + ch, d]] = x[[s, ch, d]];
                }
            }
        }
        let ffn_out = self.ffn.forward(&flat);
        let mut out = Array3::zeros((n_segs, n_ch, d_model));
        for s in 0..n_segs {
            for ch in 0..n_ch {
                for d in 0..d_model {
                    out[[s, ch, d]] = ffn_out[[s * n_ch + ch, d]];
                }
            }
        }
        out
    }

    /// Element-wise add for 3D arrays.
    fn add_3d(a: &Array3<F>, b: &Array3<F>) -> Array3<F> {
        let (n_segs, n_ch, d_model) = a.dim();
        let mut out = Array3::zeros((n_segs, n_ch, d_model));
        for s in 0..n_segs {
            for ch in 0..n_ch {
                for d in 0..d_model {
                    out[[s, ch, d]] = a[[s, ch, d]] + b[[s, ch, d]];
                }
            }
        }
        out
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// * `x` - Input of shape `(n_segs, n_channels, d_model)`
    ///
    /// # Returns
    /// Output of same shape.
    pub fn forward(&self, x: &Array3<F>) -> Result<Array3<F>> {
        // Cross-Time stage with residual + layer norm
        let normed1 = self.layer_norm_3d(x, &self.ln_gamma1, &self.ln_beta1);
        let ct_out = self.cross_time.forward(&normed1)?;
        let residual1 = Self::add_3d(x, &ct_out);

        // Cross-Dimension stage with residual + layer norm
        let normed2 = self.layer_norm_3d(&residual1, &self.ln_gamma2, &self.ln_beta2);
        let cd_out = self.cross_dim.forward(&normed2)?;
        let residual2 = Self::add_3d(&residual1, &cd_out);

        // FFN with residual + layer norm
        let normed3 = self.layer_norm_3d(&residual2, &self.ln_gamma3, &self.ln_beta3);
        let ffn_out = self.apply_ffn(&normed3);
        let output = Self::add_3d(&residual2, &ffn_out);

        Ok(output)
    }
}

// ---------------------------------------------------------------------------
// Crossformer Model
// ---------------------------------------------------------------------------

/// Crossformer model for multivariate time series forecasting.
///
/// # Input/Output
/// - Input: `[seq_len, n_channels]`
/// - Output: `[pred_len, n_channels]`
#[derive(Debug)]
pub struct CrossformerModel<F: Float + Debug> {
    config: CrossformerConfig,
    segment_merging: SegmentMerging<F>,
    layers: Vec<CrossformerLayer<F>>,
    /// Output projection: (pred_len, n_segs * d_model) — applied per channel
    w_out: Array2<F>,
    b_out: Array1<F>,
    n_segs: usize,
}

impl<F: Float + FromPrimitive + Debug> CrossformerModel<F> {
    /// Create a new Crossformer model from configuration.
    ///
    /// # Errors
    ///
    /// Returns error if configuration parameters are invalid.
    pub fn new(config: CrossformerConfig) -> Result<Self> {
        if config.seq_len == 0 {
            return Err(TimeSeriesError::InvalidInput(
                "seq_len must be positive".to_string(),
            ));
        }
        if config.pred_len == 0 {
            return Err(TimeSeriesError::InvalidInput(
                "pred_len must be positive".to_string(),
            ));
        }
        if config.n_channels == 0 {
            return Err(TimeSeriesError::InvalidInput(
                "n_channels must be positive".to_string(),
            ));
        }
        if config.seg_len == 0 {
            return Err(TimeSeriesError::InvalidInput(
                "seg_len must be positive".to_string(),
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
        if config.n_layers == 0 {
            return Err(TimeSeriesError::InvalidInput(
                "n_layers must be positive".to_string(),
            ));
        }
        if config.router_size == 0 {
            return Err(TimeSeriesError::InvalidInput(
                "router_size must be positive".to_string(),
            ));
        }

        let seed = config.seed;
        let n_segs = num_segments(config.seq_len, config.seg_len);
        let dm = config.d_model;

        let segment_merging = SegmentMerging::new(config.seg_len, dm, seed);

        let mut layers = Vec::with_capacity(config.n_layers);
        for i in 0..config.n_layers {
            layers.push(CrossformerLayer::new(
                dm,
                config.n_heads,
                config.router_size,
                config.d_ff,
                seed.wrapping_add(3000 + i as u32 * 2000),
            )?);
        }

        // Output head: for each channel, flatten segment embeddings and project to pred_len
        // Input: (n_segs * d_model) for each channel, Output: pred_len
        let w_out = nn_utils::xavier_matrix(config.pred_len, n_segs * dm, seed.wrapping_add(50000));
        let b_out = nn_utils::zero_bias(config.pred_len);

        Ok(Self {
            config,
            segment_merging,
            layers,
            w_out,
            b_out,
            n_segs,
        })
    }

    /// Forecast future values for a multivariate input.
    ///
    /// # Arguments
    /// * `input` - Input array of shape `[seq_len, n_channels]`
    ///
    /// # Returns
    /// Forecast array of shape `[pred_len, n_channels]`
    ///
    /// # Errors
    ///
    /// Returns error if input shape doesn't match configuration.
    pub fn forecast(&self, input: &Array2<F>) -> Result<Array2<F>> {
        let (seq_len, n_ch) = input.dim();
        if seq_len != self.config.seq_len {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: self.config.seq_len,
                actual: seq_len,
            });
        }
        if n_ch != self.config.n_channels {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: self.config.n_channels,
                actual: n_ch,
            });
        }

        // Step 1: Segment merging → (n_segs, n_ch, d_model)
        let seg_embed = self.segment_merging.forward(input);

        // Step 2: Crossformer layers
        let mut hidden = seg_embed;
        for layer in &self.layers {
            hidden = layer.forward(&hidden)?;
        }

        // Step 3: Output projection per channel
        // For each channel, flatten (n_segs, d_model) → (n_segs * d_model,) and project to pred_len
        let dm = self.config.d_model;
        let flat_size = self.n_segs * dm;
        let mut output = Array2::zeros((self.config.pred_len, n_ch));

        for ch in 0..n_ch {
            let mut flat: Array1<F> = Array1::zeros(flat_size);
            for s in 0..self.n_segs {
                for d in 0..dm {
                    flat[s * dm + d] = hidden[[s, ch, d]];
                }
            }
            let pred = nn_utils::dense_forward_vec(&flat, &self.w_out, &self.b_out);
            for t in 0..self.config.pred_len {
                output[[t, ch]] = pred[t];
            }
        }

        Ok(output)
    }

    /// Get a reference to the configuration.
    pub fn config(&self) -> &CrossformerConfig {
        &self.config
    }

    /// Get the number of time segments.
    pub fn n_segs(&self) -> usize {
        self.n_segs
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_input(seq_len: usize, n_channels: usize) -> Array2<f64> {
        let mut arr = Array2::zeros((seq_len, n_channels));
        for t in 0..seq_len {
            for c in 0..n_channels {
                arr[[t, c]] = (t as f64 * 0.05 + c as f64 * 0.1).sin();
            }
        }
        arr
    }

    #[test]
    fn test_default_config_values() {
        let cfg = CrossformerConfig::default();
        assert_eq!(cfg.seq_len, 96);
        assert_eq!(cfg.pred_len, 24);
        assert_eq!(cfg.n_channels, 7);
        assert_eq!(cfg.seg_len, 6);
        assert_eq!(cfg.d_model, 64);
        assert_eq!(cfg.n_heads, 4);
        assert_eq!(cfg.n_layers, 2);
        assert_eq!(cfg.router_size, 10);
        assert_eq!(cfg.d_ff, 128);
    }

    #[test]
    fn test_num_segments_calculation() {
        assert_eq!(num_segments(96, 6), 16);
        assert_eq!(num_segments(96, 8), 12);
        assert_eq!(num_segments(97, 8), 13); // ceil
        assert_eq!(num_segments(6, 6), 1);
    }

    #[test]
    fn test_model_creation_default() {
        let model = CrossformerModel::<f64>::new(CrossformerConfig::default());
        assert!(model.is_ok());
    }

    #[test]
    fn test_model_creation_invalid_d_model_not_divisible() {
        let cfg = CrossformerConfig {
            d_model: 33,
            n_heads: 4,
            ..CrossformerConfig::default()
        };
        assert!(CrossformerModel::<f64>::new(cfg).is_err());
    }

    #[test]
    fn test_model_creation_invalid_zero_seq_len() {
        let cfg = CrossformerConfig {
            seq_len: 0,
            ..CrossformerConfig::default()
        };
        assert!(CrossformerModel::<f64>::new(cfg).is_err());
    }

    #[test]
    fn test_segment_merging_output_shape() {
        let seg_len = 6;
        let d_model = 32;
        let n_channels = 4;
        let seq_len = 24;
        let sm = SegmentMerging::<f64>::new(seg_len, d_model, 42);
        let input = make_input(seq_len, n_channels);
        let out = sm.forward(&input);
        let expected_segs = num_segments(seq_len, seg_len);
        assert_eq!(out.dim(), (expected_segs, n_channels, d_model));
    }

    #[test]
    fn test_cross_time_attention_output_shape() {
        let n_segs = 8;
        let n_ch = 4;
        let d_model = 32;
        let n_heads = 4;
        let cta = CrossTimeAttention::<f64>::new(d_model, n_heads, 42).expect("creation failed");
        let x = Array3::zeros((n_segs, n_ch, d_model));
        let out = cta.forward(&x).expect("forward failed");
        assert_eq!(out.dim(), (n_segs, n_ch, d_model));
    }

    #[test]
    fn test_cross_dim_attention_output_shape() {
        let n_segs = 4;
        let n_ch = 7;
        let d_model = 32;
        let n_heads = 4;
        let router_size = 5;
        let cda = CrossDimAttention::<f64>::new(d_model, n_heads, router_size, 42)
            .expect("creation failed");
        let x = Array3::zeros((n_segs, n_ch, d_model));
        let out = cda.forward(&x).expect("forward failed");
        assert_eq!(out.dim(), (n_segs, n_ch, d_model));
    }

    #[test]
    fn test_crossformer_layer_output_shape() {
        let n_segs = 8;
        let n_ch = 4;
        let d_model = 32;
        let n_heads = 4;
        let router_size = 3;
        let d_ff = 64;
        let layer = CrossformerLayer::<f64>::new(d_model, n_heads, router_size, d_ff, 42)
            .expect("layer creation failed");
        let x = Array3::zeros((n_segs, n_ch, d_model));
        let out = layer.forward(&x).expect("forward failed");
        assert_eq!(out.dim(), (n_segs, n_ch, d_model));
    }

    #[test]
    fn test_forecast_shape_standard() {
        let cfg = CrossformerConfig {
            seq_len: 48,
            pred_len: 12,
            n_channels: 4,
            seg_len: 6,
            d_model: 32,
            n_heads: 4,
            n_layers: 1,
            router_size: 5,
            d_ff: 64,
            seed: 42,
        };
        let model = CrossformerModel::<f64>::new(cfg).expect("model creation failed");
        let input = make_input(48, 4);
        let output = model.forecast(&input).expect("forecast failed");
        assert_eq!(output.dim(), (12, 4));
    }

    #[test]
    fn test_forecast_shape_default_config() {
        let model = CrossformerModel::<f64>::new(CrossformerConfig::default())
            .expect("model creation failed");
        let input = make_input(96, 7);
        let output = model.forecast(&input).expect("forecast failed");
        assert_eq!(output.dim(), (24, 7));
    }

    #[test]
    fn test_forecast_output_is_finite() {
        let cfg = CrossformerConfig {
            seq_len: 24,
            pred_len: 6,
            n_channels: 3,
            seg_len: 4,
            d_model: 16,
            n_heads: 4,
            n_layers: 1,
            router_size: 3,
            d_ff: 32,
            seed: 7,
        };
        let model = CrossformerModel::<f64>::new(cfg).expect("model creation failed");
        let input = make_input(24, 3);
        let output = model.forecast(&input).expect("forecast failed");
        for t in 0..6 {
            for ch in 0..3 {
                assert!(output[[t, ch]].is_finite(), "Non-finite at [{t},{ch}]");
            }
        }
    }

    #[test]
    fn test_wrong_seq_len_returns_error() {
        let cfg = CrossformerConfig {
            seq_len: 48,
            pred_len: 12,
            n_channels: 3,
            seg_len: 6,
            d_model: 32,
            n_heads: 4,
            n_layers: 1,
            router_size: 3,
            d_ff: 64,
            seed: 1,
        };
        let model = CrossformerModel::<f64>::new(cfg).expect("model creation failed");
        let bad_input = make_input(32, 3); // wrong seq_len
        assert!(model.forecast(&bad_input).is_err());
    }

    #[test]
    fn test_wrong_n_channels_returns_error() {
        let cfg = CrossformerConfig {
            seq_len: 24,
            pred_len: 6,
            n_channels: 4,
            seg_len: 4,
            d_model: 16,
            n_heads: 4,
            n_layers: 1,
            router_size: 3,
            d_ff: 32,
            seed: 1,
        };
        let model = CrossformerModel::<f64>::new(cfg).expect("model creation failed");
        let bad_input = make_input(24, 7); // wrong n_channels
        assert!(model.forecast(&bad_input).is_err());
    }

    #[test]
    fn test_n_segs_accessor() {
        let cfg = CrossformerConfig {
            seq_len: 48,
            seg_len: 6,
            ..CrossformerConfig::default()
        };
        let model = CrossformerModel::<f64>::new(cfg).expect("model creation failed");
        assert_eq!(model.n_segs(), 8); // 48 / 6 = 8
    }

    #[test]
    fn test_router_size_smaller_than_n_channels() {
        // router_size=3 << n_channels=7
        let cfg = CrossformerConfig {
            seq_len: 24,
            pred_len: 6,
            n_channels: 7,
            seg_len: 4,
            d_model: 16,
            n_heads: 4,
            n_layers: 1,
            router_size: 3,
            d_ff: 32,
            seed: 42,
        };
        let model = CrossformerModel::<f64>::new(cfg).expect("model creation failed");
        let input = make_input(24, 7);
        let output = model.forecast(&input).expect("forecast failed");
        assert_eq!(output.dim(), (6, 7));
    }

    #[test]
    fn test_multiple_layers() {
        let cfg = CrossformerConfig {
            seq_len: 24,
            pred_len: 6,
            n_channels: 3,
            seg_len: 4,
            d_model: 16,
            n_heads: 4,
            n_layers: 3,
            router_size: 3,
            d_ff: 32,
            seed: 42,
        };
        let model = CrossformerModel::<f64>::new(cfg).expect("model creation failed");
        let input = make_input(24, 3);
        let output = model.forecast(&input).expect("forecast failed");
        assert_eq!(output.dim(), (6, 3));
    }
}
