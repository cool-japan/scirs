//! Grouped-Query Attention (GQA) with Rotary Position Embedding (RoPE)
//!
//! This module implements Grouped-Query Attention as described in:
//! "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"
//! by Ainslie et al. (2023).
//!
//! GQA is a generalisation of Multi-Head Attention (MHA) and Multi-Query
//! Attention (MQA):
//! - **GQA(H)**: `num_kv_heads = num_heads` => standard MHA
//! - **GQA(1)**: `num_kv_heads = 1` => MQA
//! - **GQA(G)**: `num_kv_heads = G` where `1 < G < H` => grouped query
//!
//! Additionally, this module provides a Rotary Position Embedding (RoPE)
//! implementation that can be applied to queries and keys before computing
//! attention, as used in LLaMA / Mistral / Gemma models.

use crate::error::{NeuralError, Result};
use crate::layers::Layer;
use scirs2_core::ndarray::{Array, Array2, Array4, IxDyn, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign};
use scirs2_core::random::{Rng, RngExt};
use std::fmt::Debug;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn mk_weight<F: Float, R: Rng>(rows: usize, cols: usize, rng: &mut R) -> Result<Array<F, IxDyn>> {
    let scale = (6.0_f64 / (rows + cols) as f64).sqrt();
    let mut data = Vec::with_capacity(rows * cols);
    for _ in 0..(rows * cols) {
        let x: f64 = rng.random_range(-scale..scale);
        let f = F::from(x).ok_or_else(|| NeuralError::InvalidArchitecture("xavier cast".into()))?;
        data.push(f);
    }
    Array::from_shape_vec(IxDyn(&[rows, cols]), data)
        .map_err(|e| NeuralError::InvalidArchitecture(format!("mk_weight: {e}")))
}

fn softmax_inplace<F: Float + NumAssign>(s: &mut [F]) {
    let max_v = s
        .iter()
        .fold(F::neg_infinity(), |a, &b| if b > a { b } else { a });
    let mut sum = F::zero();
    for v in s.iter_mut() {
        *v = (*v - max_v).exp();
        sum += *v;
    }
    let eps = F::from(1e-12).unwrap_or(F::zero());
    let norm = if sum < eps { eps } else { sum };
    for v in s.iter_mut() {
        *v /= norm;
    }
}

// ---------------------------------------------------------------------------
// Rotary Position Embedding (RoPE)
// ---------------------------------------------------------------------------

/// Rotary Position Embedding (RoPE) as used in LLaMA, Mistral, etc.
///
/// Applies a position-dependent rotation to pairs of dimensions:
///
/// ```text
/// For each pair (x_{2i}, x_{2i+1}) at position pos:
///   theta_i = base^(-2i/d)
///   x'_{2i}   = x_{2i}   * cos(pos * theta_i) - x_{2i+1} * sin(pos * theta_i)
///   x'_{2i+1} = x_{2i}   * sin(pos * theta_i) + x_{2i+1} * cos(pos * theta_i)
/// ```
#[derive(Debug, Clone)]
pub struct RotaryPositionEmbedding {
    /// Base frequency (typically 10000.0)
    pub base: f64,
    /// Head dimension (must be even)
    pub head_dim: usize,
    /// Maximum sequence length for precomputed frequencies
    pub max_seq_len: usize,
}

impl RotaryPositionEmbedding {
    /// Create a new RoPE with the given head dimension
    pub fn new(head_dim: usize) -> Result<Self> {
        if !head_dim.is_multiple_of(2) {
            return Err(NeuralError::InvalidArchitecture(
                "RoPE head_dim must be even".into(),
            ));
        }
        Ok(Self {
            base: 10000.0,
            head_dim,
            max_seq_len: 8192,
        })
    }

    /// Set the base frequency
    pub fn with_base(mut self, base: f64) -> Self {
        self.base = base;
        self
    }

    /// Set the max sequence length
    pub fn with_max_seq_len(mut self, max_seq_len: usize) -> Self {
        self.max_seq_len = max_seq_len;
        self
    }

    /// Apply RoPE to a tensor in-place
    ///
    /// # Arguments
    /// * `x` - Tensor [batch, seq_len, num_heads, head_dim]
    /// * `position_offset` - Starting position (for cached generation)
    pub fn apply<F: Float + NumAssign>(
        &self,
        x: &mut Array4<F>,
        position_offset: usize,
    ) -> Result<()> {
        let seq_len = x.shape()[1];
        let num_heads = x.shape()[2];
        let head_dim = x.shape()[3];

        if head_dim != self.head_dim {
            return Err(NeuralError::InvalidArchitecture(format!(
                "RoPE head_dim mismatch: expected {}, got {head_dim}",
                self.head_dim
            )));
        }

        let half_dim = head_dim / 2;
        let batch = x.shape()[0];

        for b in 0..batch {
            for t in 0..seq_len {
                let pos = (position_offset + t) as f64;
                for h in 0..num_heads {
                    for i in 0..half_dim {
                        let theta = pos / self.base.powf(2.0 * i as f64 / head_dim as f64);
                        let cos_theta = F::from(theta.cos())
                            .ok_or_else(|| NeuralError::ComputationError("cos cast".into()))?;
                        let sin_theta = F::from(theta.sin())
                            .ok_or_else(|| NeuralError::ComputationError("sin cast".into()))?;

                        let x0 = x[[b, t, h, 2 * i]];
                        let x1 = x[[b, t, h, 2 * i + 1]];

                        x[[b, t, h, 2 * i]] = x0 * cos_theta - x1 * sin_theta;
                        x[[b, t, h, 2 * i + 1]] = x0 * sin_theta + x1 * cos_theta;
                    }
                }
            }
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// KV Cache
// ---------------------------------------------------------------------------

/// Key-Value cache for GQA autoregressive generation
#[derive(Debug, Clone)]
pub struct GqaKvCache<F: Float> {
    /// Cached keys: [batch, past_len, num_kv_heads, head_dim]
    pub keys: Array4<F>,
    /// Cached values: [batch, past_len, num_kv_heads, head_dim]
    pub values: Array4<F>,
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for Grouped-Query Attention
#[derive(Debug, Clone)]
pub struct GroupedQueryAttentionConfig {
    /// Number of query heads (H)
    pub num_heads: usize,
    /// Number of key-value heads (G). Must divide num_heads evenly.
    /// G=1 is MQA, G=H is MHA.
    pub num_kv_heads: usize,
    /// Per-head dimension
    pub head_dim: usize,
    /// Dropout probability
    pub dropout_prob: f64,
    /// Whether to apply causal masking
    pub causal: bool,
    /// Whether to apply RoPE
    pub use_rope: bool,
    /// RoPE base frequency (only used when use_rope=true)
    pub rope_base: f64,
}

impl Default for GroupedQueryAttentionConfig {
    fn default() -> Self {
        Self {
            num_heads: 8,
            num_kv_heads: 4,
            head_dim: 64,
            dropout_prob: 0.0,
            causal: false,
            use_rope: false,
            rope_base: 10000.0,
        }
    }
}

impl GroupedQueryAttentionConfig {
    /// Create a new GQA config
    pub fn new(num_heads: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        Self {
            num_heads,
            num_kv_heads,
            head_dim,
            ..Default::default()
        }
    }

    /// Enable causal masking
    pub fn with_causal(mut self, causal: bool) -> Self {
        self.causal = causal;
        self
    }

    /// Enable RoPE
    pub fn with_rope(mut self, use_rope: bool) -> Self {
        self.use_rope = use_rope;
        self
    }

    /// Set RoPE base frequency
    pub fn with_rope_base(mut self, base: f64) -> Self {
        self.rope_base = base;
        self
    }

    /// Set dropout probability
    pub fn with_dropout(mut self, prob: f64) -> Self {
        self.dropout_prob = prob;
        self
    }
}

// ---------------------------------------------------------------------------
// Layer
// ---------------------------------------------------------------------------

/// Grouped-Query Attention layer with optional RoPE
///
/// Implements GQA where `num_kv_heads` groups of key-value heads are shared
/// among `num_heads / num_kv_heads` query heads each.
///
/// # Input
/// 3D tensor `[batch, seq_len, d_model]` where `d_model = num_heads * head_dim`
///
/// # Output
/// 3D tensor `[batch, seq_len, d_model]`
///
/// # Examples
///
/// ```rust
/// use scirs2_neural::layers::{GroupedQueryAttention, GroupedQueryAttentionConfig, Layer};
/// use scirs2_core::ndarray::Array3;
/// use scirs2_core::random::rng;
///
/// let mut rng = rng();
/// let config = GroupedQueryAttentionConfig::new(8, 2, 8) // 8 Q heads, 2 KV heads
///     .with_causal(true)
///     .with_rope(true);
/// let gqa = GroupedQueryAttention::<f64>::new(64, config, &mut rng).expect("failed");
///
/// let input = Array3::<f64>::from_elem((1, 16, 64), 0.1).into_dyn();
/// let output = gqa.forward(&input).expect("failed");
/// assert_eq!(output.shape(), &[1, 16, 64]);
/// ```
#[derive(Debug)]
pub struct GroupedQueryAttention<F: Float + Debug + Send + Sync + NumAssign> {
    d_model: usize,
    config: GroupedQueryAttentionConfig,
    /// [d_model, num_heads * head_dim]
    w_q: Array<F, IxDyn>,
    /// [d_model, num_kv_heads * head_dim]
    w_k: Array<F, IxDyn>,
    /// [d_model, num_kv_heads * head_dim]
    w_v: Array<F, IxDyn>,
    /// [num_heads * head_dim, d_model]
    w_o: Array<F, IxDyn>,
    scale: F,
    rope: Option<RotaryPositionEmbedding>,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + 'static + NumAssign>
    GroupedQueryAttention<F>
{
    /// Create a new Grouped-Query Attention layer
    pub fn new<R: Rng>(
        d_model: usize,
        config: GroupedQueryAttentionConfig,
        rng: &mut R,
    ) -> Result<Self> {
        if config.num_heads == 0 || config.num_kv_heads == 0 || config.head_dim == 0 {
            return Err(NeuralError::InvalidArchitecture(
                "num_heads, num_kv_heads, head_dim must be > 0".into(),
            ));
        }

        if !config.num_heads.is_multiple_of(config.num_kv_heads) {
            return Err(NeuralError::InvalidArchitecture(format!(
                "num_heads ({}) must be divisible by num_kv_heads ({})",
                config.num_heads, config.num_kv_heads
            )));
        }

        let q_dim = config.num_heads * config.head_dim;
        if q_dim != d_model {
            return Err(NeuralError::InvalidArchitecture(format!(
                "num_heads * head_dim ({q_dim}) must equal d_model ({d_model})"
            )));
        }

        let kv_dim = config.num_kv_heads * config.head_dim;

        let w_q = mk_weight(d_model, q_dim, rng)?;
        let w_k = mk_weight(d_model, kv_dim, rng)?;
        let w_v = mk_weight(d_model, kv_dim, rng)?;
        let w_o = mk_weight(q_dim, d_model, rng)?;

        let scale = F::one()
            / F::from(config.head_dim)
                .ok_or_else(|| NeuralError::InvalidArchitecture("scale cast".into()))?
                .sqrt();

        let rope = if config.use_rope {
            Some(RotaryPositionEmbedding::new(config.head_dim)?.with_base(config.rope_base))
        } else {
            None
        };

        Ok(Self {
            d_model,
            config,
            w_q,
            w_k,
            w_v,
            w_o,
            scale,
            rope,
        })
    }

    /// Forward pass with optional KV cache
    ///
    /// # Arguments
    /// * `input` - [batch, seq_len, d_model]
    /// * `past_kv` - Optional past KV cache
    ///
    /// # Returns
    /// (output, updated cache)
    pub fn forward_with_cache(
        &self,
        input: &Array<F, IxDyn>,
        past_kv: Option<&GqaKvCache<F>>,
    ) -> Result<(Array<F, IxDyn>, GqaKvCache<F>)> {
        if input.ndim() != 3 {
            return Err(NeuralError::InvalidArchitecture(format!(
                "GQA expects 3D input, got {}D",
                input.ndim()
            )));
        }

        let shape = input.shape();
        let (batch, seq_len, d_model) = (shape[0], shape[1], shape[2]);

        if d_model != self.d_model {
            return Err(NeuralError::InvalidArchitecture(format!(
                "input dim {d_model} != d_model {}",
                self.d_model
            )));
        }

        let num_heads = self.config.num_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = self.config.head_dim;
        let group_size = num_heads / num_kv_heads;

        // Project Q, K, V
        let mut q_4d =
            self.project_reshape(input, &self.w_q, batch, seq_len, num_heads, head_dim)?;
        let mut k_new =
            self.project_reshape(input, &self.w_k, batch, seq_len, num_kv_heads, head_dim)?;
        let v_new =
            self.project_reshape(input, &self.w_v, batch, seq_len, num_kv_heads, head_dim)?;

        // Apply RoPE if configured
        let position_offset = past_kv.map(|c| c.keys.shape()[1]).unwrap_or(0);
        if let Some(ref rope) = self.rope {
            rope.apply(&mut q_4d, position_offset)?;
            rope.apply(&mut k_new, position_offset)?;
        }

        // Concatenate with cache
        let (k_4d, v_4d, total_kv_len) = if let Some(cache) = past_kv {
            let past_len = cache.keys.shape()[1];
            let total = past_len + seq_len;
            let k_full =
                self.concat_cache(&cache.keys, &k_new, batch, total, num_kv_heads, head_dim)?;
            let v_full =
                self.concat_cache(&cache.values, &v_new, batch, total, num_kv_heads, head_dim)?;
            (k_full, v_full, total)
        } else {
            (k_new.clone(), v_new.clone(), seq_len)
        };

        let new_cache = GqaKvCache {
            keys: k_4d.clone(),
            values: v_4d.clone(),
        };

        // Compute attention with grouping
        let mut output_4d = Array4::<F>::zeros((batch, seq_len, num_heads, head_dim));

        for b in 0..batch {
            for kv_h in 0..num_kv_heads {
                let q_h_start = kv_h * group_size;
                let q_h_end = q_h_start + group_size;

                for q_h in q_h_start..q_h_end {
                    for t in 0..seq_len {
                        let global_t = position_offset + t;

                        let mut scores = Vec::with_capacity(total_kv_len);
                        for s_idx in 0..total_kv_len {
                            if self.config.causal && s_idx > global_t {
                                scores.push(F::neg_infinity());
                            } else {
                                let mut dot = F::zero();
                                for d in 0..head_dim {
                                    dot += q_4d[[b, t, q_h, d]] * k_4d[[b, s_idx, kv_h, d]];
                                }
                                scores.push(dot * self.scale);
                            }
                        }

                        softmax_inplace(&mut scores);

                        for d in 0..head_dim {
                            let mut acc = F::zero();
                            for s_idx in 0..total_kv_len {
                                acc += scores[s_idx] * v_4d[[b, s_idx, kv_h, d]];
                            }
                            output_4d[[b, t, q_h, d]] = acc;
                        }
                    }
                }
            }
        }

        // Output projection
        let output_3d = output_4d
            .into_shape_with_order((batch, seq_len, d_model))
            .map_err(|e| NeuralError::InferenceError(format!("reshape output: {e}")))?;

        let output_2d = output_3d
            .into_shape_with_order((batch * seq_len, d_model))
            .map_err(|e| NeuralError::InferenceError(format!("reshape O proj: {e}")))?;

        let w_o_2d = self
            .w_o
            .view()
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .map_err(|_| NeuralError::InferenceError("O weight 2D".into()))?;

        let final_out = output_2d.dot(&w_o_2d);

        let result = final_out
            .into_shape_with_order((batch, seq_len, d_model))
            .map_err(|e| NeuralError::InferenceError(format!("reshape final: {e}")))?;

        Ok((result.into_dyn(), new_cache))
    }

    fn project_reshape(
        &self,
        input: &Array<F, IxDyn>,
        weight: &Array<F, IxDyn>,
        batch: usize,
        seq: usize,
        heads: usize,
        head_dim: usize,
    ) -> Result<Array4<F>> {
        let d_model = input.shape()[2];

        let input_2d = input
            .clone()
            .into_shape_with_order(IxDyn(&[batch * seq, d_model]))
            .map_err(|e| NeuralError::InferenceError(format!("reshape: {e}")))?;

        let input_2d_view = input_2d
            .view()
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .map_err(|_| NeuralError::InferenceError("to Ix2".into()))?;

        let w_2d = weight
            .view()
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .map_err(|_| NeuralError::InferenceError("weight Ix2".into()))?;

        let projected = input_2d_view.dot(&w_2d);

        projected
            .into_shape_with_order((batch, seq, heads, head_dim))
            .map_err(|e| NeuralError::InferenceError(format!("reshape projected: {e}")))
    }

    fn concat_cache(
        &self,
        past: &Array4<F>,
        new: &Array4<F>,
        batch: usize,
        total_len: usize,
        heads: usize,
        head_dim: usize,
    ) -> Result<Array4<F>> {
        let past_len = past.shape()[1];
        let new_len = new.shape()[1];

        if past_len + new_len != total_len {
            return Err(NeuralError::InferenceError(
                "cache concat length mismatch".into(),
            ));
        }

        let mut result = Array4::<F>::zeros((batch, total_len, heads, head_dim));

        for b in 0..batch {
            for t in 0..past_len {
                for h in 0..heads {
                    for d in 0..head_dim {
                        result[[b, t, h, d]] = past[[b, t, h, d]];
                    }
                }
            }
            for t in 0..new_len {
                for h in 0..heads {
                    for d in 0..head_dim {
                        result[[b, past_len + t, h, d]] = new[[b, t, h, d]];
                    }
                }
            }
        }

        Ok(result)
    }

    /// Apply the inverse RoPE rotation (transpose of forward RoPE).
    ///
    /// Forward RoPE rotates by +theta; the inverse rotates by -theta.
    /// Since R(-theta) is the same as the conjugate rotation (negate sin),
    /// we replicate the RoPE apply loop with negated sin.
    fn apply_rope_inverse(
        rope: &RotaryPositionEmbedding,
        mut x: Array4<F>,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> Result<Array4<F>> {
        let half_dim = head_dim / 2;
        let batch = x.shape()[0];

        for b in 0..batch {
            for t in 0..seq_len {
                let pos = t as f64;
                for h in 0..num_heads {
                    for i in 0..half_dim {
                        let theta = pos / rope.base.powf(2.0 * i as f64 / head_dim as f64);
                        let cos_theta = F::from(theta.cos()).ok_or_else(|| {
                            NeuralError::ComputationError("inv rope cos cast".into())
                        })?;
                        // Negated sin for inverse rotation
                        let neg_sin_theta = F::from(-theta.sin()).ok_or_else(|| {
                            NeuralError::ComputationError("inv rope sin cast".into())
                        })?;

                        let x0 = x[[b, t, h, 2 * i]];
                        let x1 = x[[b, t, h, 2 * i + 1]];

                        // Inverse: x'_0 = x0*cos - x1*(-sin) = x0*cos + x1*sin
                        //          x'_1 = x0*(-sin) + x1*cos = -x0*sin + x1*cos
                        // Equivalently: apply_rope with negated sin
                        x[[b, t, h, 2 * i]] = x0 * cos_theta - x1 * neg_sin_theta;
                        x[[b, t, h, 2 * i + 1]] = x0 * neg_sin_theta + x1 * cos_theta;
                    }
                }
            }
        }

        Ok(x)
    }

    /// Get configuration
    pub fn config(&self) -> &GroupedQueryAttentionConfig {
        &self.config
    }

    /// Get model dimension
    pub fn d_model(&self) -> usize {
        self.d_model
    }
}

impl<F> Layer<F> for GroupedQueryAttention<F>
where
    F: Float + Debug + ScalarOperand + Send + Sync + 'static + NumAssign,
{
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let (output, _cache) = self.forward_with_cache(input, None)?;
        Ok(output)
    }

    fn backward(
        &self,
        input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        if input.ndim() != 3 {
            return Err(NeuralError::InvalidArchitecture(format!(
                "GQA backward expects 3D input, got {}D",
                input.ndim()
            )));
        }

        let shape = input.shape();
        let (batch, seq_len, d_model) = (shape[0], shape[1], shape[2]);

        if d_model != self.d_model {
            return Err(NeuralError::InvalidArchitecture(format!(
                "GQA backward: input dim {d_model} != d_model {}",
                self.d_model
            )));
        }

        let num_heads = self.config.num_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = self.config.head_dim;
        let group_size = num_heads / num_kv_heads;

        // -----------------------------------------------------------------------
        // Re-compute forward quantities (Q, K, V with optional RoPE)
        // -----------------------------------------------------------------------
        let mut q_4d =
            self.project_reshape(input, &self.w_q, batch, seq_len, num_heads, head_dim)?;
        let mut k_4d =
            self.project_reshape(input, &self.w_k, batch, seq_len, num_kv_heads, head_dim)?;
        let v_4d =
            self.project_reshape(input, &self.w_v, batch, seq_len, num_kv_heads, head_dim)?;

        // Apply RoPE (position_offset=0 since no cache in backward context)
        if let Some(ref rope) = self.rope {
            rope.apply(&mut q_4d, 0)?;
            rope.apply(&mut k_4d, 0)?;
        }

        // -----------------------------------------------------------------------
        // Compute attention weights (softmax output)
        // -----------------------------------------------------------------------
        let mut attn_weights = Array::zeros(IxDyn(&[batch, num_heads, seq_len, seq_len]));

        for b in 0..batch {
            for kv_h in 0..num_kv_heads {
                let q_h_start = kv_h * group_size;
                let q_h_end = q_h_start + group_size;

                for q_h in q_h_start..q_h_end {
                    for t in 0..seq_len {
                        let mut scores = Vec::with_capacity(seq_len);
                        for s_idx in 0..seq_len {
                            if self.config.causal && s_idx > t {
                                scores.push(F::neg_infinity());
                            } else {
                                let mut dot = F::zero();
                                for d in 0..head_dim {
                                    dot += q_4d[[b, t, q_h, d]] * k_4d[[b, s_idx, kv_h, d]];
                                }
                                scores.push(dot * self.scale);
                            }
                        }

                        softmax_inplace(&mut scores);

                        for s_idx in 0..seq_len {
                            attn_weights[[b, q_h, t, s_idx]] = scores[s_idx];
                        }
                    }
                }
            }
        }

        // -----------------------------------------------------------------------
        // Backward through output projection
        // -----------------------------------------------------------------------
        let grad_out_2d = grad_output
            .view()
            .into_shape_with_order(IxDyn(&[batch * seq_len, d_model]))
            .map_err(|e| {
                NeuralError::InferenceError(format!("GQA backward reshape grad_out: {e}"))
            })?;

        let w_o_2d = self
            .w_o
            .view()
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .map_err(|_| NeuralError::InferenceError("GQA backward: W_o to Ix2".into()))?;

        let grad_out_2d_typed = grad_out_2d
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .map_err(|_| NeuralError::InferenceError("GQA backward: grad_out to Ix2".into()))?;

        // d_concat: [batch*seq, q_dim]
        let d_concat_2d = grad_out_2d_typed.dot(&w_o_2d.t());

        let q_dim = num_heads * head_dim;
        let d_concat_4d: Array4<F> = d_concat_2d
            .into_shape_with_order((batch, seq_len, num_heads, head_dim))
            .map_err(|e| {
                NeuralError::InferenceError(format!("GQA backward d_concat reshape: {e}"))
            })?;

        // -----------------------------------------------------------------------
        // Backward through grouped attention
        // GQA: each KV head serves group_size query heads.
        // Accumulate d_K, d_V gradients from all query heads in each group.
        // -----------------------------------------------------------------------
        let kv_dim = num_kv_heads * head_dim;

        // These accumulate gradients in post-RoPE space (if RoPE is active)
        let mut d_q_post_rope = Array4::<F>::zeros((batch, seq_len, num_heads, head_dim));
        let mut d_k_post_rope = Array4::<F>::zeros((batch, seq_len, num_kv_heads, head_dim));
        let mut d_v_4d = Array4::<F>::zeros((batch, seq_len, num_kv_heads, head_dim));

        for b in 0..batch {
            for kv_h in 0..num_kv_heads {
                let q_h_start = kv_h * group_size;
                let q_h_end = q_h_start + group_size;

                for q_h in q_h_start..q_h_end {
                    for t in 0..seq_len {
                        // d_V[b,s,kv_h,:] += attn[b,q_h,t,s] * d_concat[b,t,q_h,:]
                        for s_idx in 0..seq_len {
                            let a = attn_weights[[b, q_h, t, s_idx]];
                            for d in 0..head_dim {
                                d_v_4d[[b, s_idx, kv_h, d]] += a * d_concat_4d[[b, t, q_h, d]];
                            }
                        }

                        // d_scores_raw[s] = sum_d d_concat[b,t,q_h,d] * V[b,s,kv_h,d]
                        let mut d_scores_raw = vec![F::zero(); seq_len];
                        for s_idx in 0..seq_len {
                            let mut sum = F::zero();
                            for d in 0..head_dim {
                                sum += d_concat_4d[[b, t, q_h, d]] * v_4d[[b, s_idx, kv_h, d]];
                            }
                            d_scores_raw[s_idx] = sum;
                        }

                        // Softmax backward
                        let dot_attn_d: F = (0..seq_len).fold(F::zero(), |acc, s| {
                            acc + attn_weights[[b, q_h, t, s]] * d_scores_raw[s]
                        });

                        let mut d_pre_scores = vec![F::zero(); seq_len];
                        for s_idx in 0..seq_len {
                            let a = attn_weights[[b, q_h, t, s_idx]];
                            d_pre_scores[s_idx] = a * (d_scores_raw[s_idx] - dot_attn_d);
                        }

                        // Scale
                        for d_pre in d_pre_scores.iter_mut() {
                            *d_pre *= self.scale;
                        }

                        // d_Q_post_rope[b,t,q_h,:] += sum_s d_pre[s] * K_post_rope[b,s,kv_h,:]
                        for s_idx in 0..seq_len {
                            let dp = d_pre_scores[s_idx];
                            for d in 0..head_dim {
                                d_q_post_rope[[b, t, q_h, d]] += dp * k_4d[[b, s_idx, kv_h, d]];
                            }
                        }

                        // d_K_post_rope[b,s,kv_h,:] += d_pre[s] * Q_post_rope[b,t,q_h,:]
                        for s_idx in 0..seq_len {
                            let dp = d_pre_scores[s_idx];
                            for d in 0..head_dim {
                                d_k_post_rope[[b, s_idx, kv_h, d]] += dp * q_4d[[b, t, q_h, d]];
                            }
                        }
                    }
                }
            }
        }

        // -----------------------------------------------------------------------
        // Un-rotate through RoPE if active.
        // RoPE is a rotation R(pos). Forward: x_rot = R(pos) @ x
        // Backward: d_x = R(pos)^T @ d_x_rot = R(-pos) @ d_x_rot
        // Since R(-pos) = R(pos)^{-1} (orthogonal), we apply with negated angles.
        // Equivalently: apply RoPE again with negated sin (conjugate rotation).
        // -----------------------------------------------------------------------
        let (d_q_4d, d_k_4d) = if let Some(ref rope) = self.rope {
            // Apply the inverse rotation: rotate with negative position angles.
            // We implement this by applying RoPE with negated sin terms,
            // which we achieve by building a temporary "negated rope" helper.
            let d_q_pre =
                Self::apply_rope_inverse(rope, d_q_post_rope, seq_len, num_heads, head_dim)?;
            let d_k_pre =
                Self::apply_rope_inverse(rope, d_k_post_rope, seq_len, num_kv_heads, head_dim)?;
            (d_q_pre, d_k_pre)
        } else {
            (d_q_post_rope, d_k_post_rope)
        };

        // -----------------------------------------------------------------------
        // Backward through Q/K/V projections
        // -----------------------------------------------------------------------
        let d_q_2d = d_q_4d
            .into_shape_with_order((batch * seq_len, q_dim))
            .map_err(|e| NeuralError::InferenceError(format!("GQA backward d_q reshape: {e}")))?;

        let d_k_2d = d_k_4d
            .into_shape_with_order((batch * seq_len, kv_dim))
            .map_err(|e| NeuralError::InferenceError(format!("GQA backward d_k reshape: {e}")))?;

        let d_v_2d = d_v_4d
            .into_shape_with_order((batch * seq_len, kv_dim))
            .map_err(|e| NeuralError::InferenceError(format!("GQA backward d_v reshape: {e}")))?;

        let w_q_2d = self
            .w_q
            .view()
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .map_err(|_| NeuralError::InferenceError("GQA backward: W_q to Ix2".into()))?;
        let w_k_2d = self
            .w_k
            .view()
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .map_err(|_| NeuralError::InferenceError("GQA backward: W_k to Ix2".into()))?;
        let w_v_2d = self
            .w_v
            .view()
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .map_err(|_| NeuralError::InferenceError("GQA backward: W_v to Ix2".into()))?;

        let d_input_from_q = d_q_2d.dot(&w_q_2d.t());
        let d_input_from_k = d_k_2d.dot(&w_k_2d.t());
        let d_input_from_v = d_v_2d.dot(&w_v_2d.t());

        let d_input_2d = d_input_from_q + d_input_from_k + d_input_from_v;

        let d_input = d_input_2d
            .into_shape_with_order(IxDyn(&[batch, seq_len, d_model]))
            .map_err(|e| {
                NeuralError::InferenceError(format!("GQA backward d_input reshape: {e}"))
            })?;

        Ok(d_input)
    }

    fn update(&mut self, _learning_rate: F) -> Result<()> {
        Ok(())
    }

    fn layer_type(&self) -> &str {
        "GroupedQueryAttention"
    }

    fn parameter_count(&self) -> usize {
        let q_dim = self.config.num_heads * self.config.head_dim;
        let kv_dim = self.config.num_kv_heads * self.config.head_dim;
        let dm = self.d_model;
        dm * q_dim + 2 * dm * kv_dim + q_dim * dm
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array3;

    #[test]
    fn test_gqa_creation() {
        let mut rng = scirs2_core::random::rng();
        let config = GroupedQueryAttentionConfig::new(8, 2, 8);
        let gqa = GroupedQueryAttention::<f64>::new(64, config, &mut rng);
        assert!(gqa.is_ok());
    }

    #[test]
    fn test_gqa_forward_shape() {
        let mut rng = scirs2_core::random::rng();
        let config = GroupedQueryAttentionConfig::new(4, 2, 16);
        let gqa = GroupedQueryAttention::<f64>::new(64, config, &mut rng).expect("creation failed");

        let input = Array3::<f64>::from_elem((2, 10, 64), 0.1).into_dyn();
        let output = gqa.forward(&input).expect("forward failed");
        assert_eq!(output.shape(), &[2, 10, 64]);
    }

    #[test]
    fn test_gqa_1_equals_mqa_shape() {
        // GQA(1) = MQA
        let mut rng = scirs2_core::random::rng();
        let config = GroupedQueryAttentionConfig::new(4, 1, 16);
        let gqa = GroupedQueryAttention::<f64>::new(64, config, &mut rng).expect("creation failed");

        let input = Array3::<f64>::from_elem((1, 6, 64), 0.1).into_dyn();
        let output = gqa.forward(&input).expect("forward failed");
        assert_eq!(output.shape(), &[1, 6, 64]);

        // Parameter count: K,V only have 1 head
        let q_dim = 4 * 16;
        let kv_dim = 16;
        let dm = 64;
        assert_eq!(
            gqa.parameter_count(),
            dm * q_dim + 2 * dm * kv_dim + q_dim * dm
        );
    }

    #[test]
    fn test_gqa_h_equals_mha_shape() {
        // GQA(H) = MHA
        let mut rng = scirs2_core::random::rng();
        let config = GroupedQueryAttentionConfig::new(4, 4, 16);
        let gqa = GroupedQueryAttention::<f64>::new(64, config, &mut rng).expect("creation failed");

        let input = Array3::<f64>::from_elem((1, 8, 64), 0.15).into_dyn();
        let output = gqa.forward(&input).expect("forward failed");
        assert_eq!(output.shape(), &[1, 8, 64]);

        for val in output.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_gqa_causal_masking() {
        let mut rng = scirs2_core::random::rng();
        let config = GroupedQueryAttentionConfig::new(4, 2, 8).with_causal(true);
        let gqa = GroupedQueryAttention::<f64>::new(32, config, &mut rng).expect("creation failed");

        let mut input = Array3::<f64>::zeros((1, 6, 32));
        for t in 0..6 {
            for d in 0..32 {
                input[[0, t, d]] = (t as f64 + 1.0) * 0.1 + d as f64 * 0.01;
            }
        }

        let output = gqa.forward(&input.into_dyn()).expect("forward failed");
        assert_eq!(output.shape(), &[1, 6, 32]);

        for val in output.iter() {
            assert!(val.is_finite(), "causal output non-finite");
        }
    }

    #[test]
    fn test_gqa_with_rope() {
        let mut rng = scirs2_core::random::rng();
        let config = GroupedQueryAttentionConfig::new(4, 2, 8)
            .with_rope(true)
            .with_causal(true);
        let gqa = GroupedQueryAttention::<f64>::new(32, config, &mut rng).expect("creation failed");

        let mut input = Array3::<f64>::zeros((1, 8, 32));
        for t in 0..8 {
            for d in 0..32 {
                input[[0, t, d]] = (t as f64) * 0.05 + d as f64 * 0.02;
            }
        }

        let output = gqa.forward(&input.into_dyn()).expect("forward failed");
        assert_eq!(output.shape(), &[1, 8, 32]);

        for val in output.iter() {
            assert!(val.is_finite(), "RoPE output non-finite");
        }
    }

    #[test]
    fn test_gqa_kv_cache() {
        let mut rng = scirs2_core::random::rng();
        let config = GroupedQueryAttentionConfig::new(4, 2, 8).with_causal(true);
        let gqa = GroupedQueryAttention::<f64>::new(32, config, &mut rng).expect("creation failed");

        // Step 1: prefix
        let prefix = Array3::<f64>::from_elem((1, 4, 32), 0.1).into_dyn();
        let (out1, cache1) = gqa.forward_with_cache(&prefix, None).expect("step 1");
        assert_eq!(out1.shape(), &[1, 4, 32]);
        assert_eq!(cache1.keys.shape()[1], 4);

        // Step 2: one new token
        let token = Array3::<f64>::from_elem((1, 1, 32), 0.2).into_dyn();
        let (out2, cache2) = gqa
            .forward_with_cache(&token, Some(&cache1))
            .expect("step 2");
        assert_eq!(out2.shape(), &[1, 1, 32]);
        assert_eq!(cache2.keys.shape()[1], 5);
    }

    #[test]
    fn test_gqa_kv_cache_with_rope() {
        let mut rng = scirs2_core::random::rng();
        let config = GroupedQueryAttentionConfig::new(4, 2, 8)
            .with_causal(true)
            .with_rope(true);
        let gqa = GroupedQueryAttention::<f64>::new(32, config, &mut rng).expect("creation failed");

        let prefix = Array3::<f64>::from_elem((1, 3, 32), 0.1).into_dyn();
        let (_out1, cache1) = gqa.forward_with_cache(&prefix, None).expect("step 1");

        let token = Array3::<f64>::from_elem((1, 1, 32), 0.15).into_dyn();
        let (out2, cache2) = gqa
            .forward_with_cache(&token, Some(&cache1))
            .expect("step 2");
        assert_eq!(out2.shape(), &[1, 1, 32]);
        assert_eq!(cache2.keys.shape()[1], 4);

        for val in out2.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_gqa_invalid_config() {
        let mut rng = scirs2_core::random::rng();

        // num_heads not divisible by num_kv_heads
        let config = GroupedQueryAttentionConfig::new(7, 3, 8);
        let result = GroupedQueryAttention::<f64>::new(56, config, &mut rng);
        assert!(result.is_err());
    }

    #[test]
    fn test_rope_standalone() {
        let rope = RotaryPositionEmbedding::new(8).expect("creation failed");
        let mut x = Array4::<f64>::zeros((1, 4, 2, 8));

        // Fill with identity-like values
        for t in 0..4 {
            for h in 0..2 {
                for d in 0..8 {
                    x[[0, t, h, d]] = 1.0;
                }
            }
        }

        let original = x.clone();
        rope.apply(&mut x, 0).expect("apply failed");

        // Position 0: cos(0)=1, sin(0)=0 => no change for pairs
        // x'_{2i} = 1*1 - 1*0 = 1, x'_{2i+1} = 1*0 + 1*1 = 1
        for d in 0..8 {
            assert!(
                (x[[0, 0, 0, d]] - original[[0, 0, 0, d]]).abs() < 1e-10,
                "position 0 should not change (dim {d})"
            );
        }

        // Position > 0 should differ
        let mut any_different = false;
        for d in 0..8 {
            if (x[[0, 1, 0, d]] - original[[0, 1, 0, d]]).abs() > 1e-10 {
                any_different = true;
            }
        }
        assert!(any_different, "position 1 should differ from position 0");
    }

    #[test]
    fn test_rope_position_offset() {
        let rope = RotaryPositionEmbedding::new(4).expect("creation failed");

        // Applying RoPE to pos 0..4 should give different results than pos 2..6
        let mut x1 = Array4::<f64>::from_elem((1, 4, 1, 4), 0.5);
        let mut x2 = x1.clone();

        rope.apply(&mut x1, 0).expect("apply 1");
        rope.apply(&mut x2, 2).expect("apply 2");

        // They should differ because positions are different
        let mut any_diff = false;
        for t in 0..4 {
            for d in 0..4 {
                if (x1[[0, t, 0, d]] - x2[[0, t, 0, d]]).abs() > 1e-10 {
                    any_diff = true;
                }
            }
        }
        assert!(
            any_diff,
            "different position offsets should give different results"
        );
    }

    #[test]
    fn test_rope_odd_dim_rejected() {
        let result = RotaryPositionEmbedding::new(7);
        assert!(result.is_err());
    }

    /// Test that GQA backward returns correct output shape.
    #[test]
    fn test_gqa_backward_shape() {
        let mut rng = scirs2_core::random::rng();
        // 4 Q heads, 2 KV heads, head_dim=8 => d_model=32
        let config = GroupedQueryAttentionConfig::new(4, 2, 8);
        let gqa = GroupedQueryAttention::<f64>::new(32, config, &mut rng).expect("creation failed");

        let input = Array3::<f64>::from_elem((2, 6, 32), 0.1).into_dyn();
        let grad_output = Array3::<f64>::from_elem((2, 6, 32), 0.02).into_dyn();

        let d_input = gqa
            .backward(&input, &grad_output)
            .expect("GQA backward should succeed");

        assert_eq!(
            d_input.shape(),
            input.shape(),
            "GQA backward gradient shape must match input shape"
        );

        for val in d_input.iter() {
            assert!(val.is_finite(), "GQA backward gradient must be finite");
        }
    }

    /// Analytical test for GQA with seq_len=1 (no-RoPE mode).
    ///
    /// With seq_len=1, softmax gradient is zero so d_Q=d_K=0.
    /// Only V path: output = (input @ W_v) @ W_o
    ///              d_input = grad_output @ W_o^T @ W_v^T
    #[test]
    fn test_gqa_backward_analytical_seq1() {
        let mut rng = scirs2_core::random::rng();
        // 1 Q head, 1 KV head, head_dim=4 => d_model=4
        let config = GroupedQueryAttentionConfig::new(1, 1, 4);
        let gqa = GroupedQueryAttention::<f64>::new(4, config, &mut rng).expect("creation failed");

        let input = Array3::<f64>::from_elem((1, 1, 4), 0.5).into_dyn();
        let grad_out = Array3::<f64>::from_elem((1, 1, 4), 1.0).into_dyn();

        let d_input = gqa
            .backward(&input, &grad_out)
            .expect("GQA backward failed");

        assert_eq!(d_input.shape(), &[1, 1, 4]);

        use scirs2_core::ndarray::{Array2, Ix2};

        let w_v_2d = gqa.w_v.view().into_dimensionality::<Ix2>().unwrap();
        let w_o_2d = gqa.w_o.view().into_dimensionality::<Ix2>().unwrap();

        let go = Array2::<f64>::from_elem((1, 4), 1.0);
        let expected = go.dot(&w_o_2d.t()).dot(&w_v_2d.t());

        for d in 0..4 {
            let computed = d_input[[0, 0, d]];
            let analytic = expected[[0, d]];
            assert!(
                (computed - analytic).abs() < 1e-10,
                "GQA seq1 dim {d}: computed={computed:.6} vs analytic={analytic:.6}"
            );
        }
    }

    /// Test GQA backward with RoPE enabled — should return finite gradients.
    #[test]
    fn test_gqa_backward_with_rope() {
        let mut rng = scirs2_core::random::rng();
        let config = GroupedQueryAttentionConfig::new(4, 2, 8).with_rope(true);
        let gqa = GroupedQueryAttention::<f64>::new(32, config, &mut rng).expect("creation failed");

        let input = Array3::<f64>::from_elem((1, 4, 32), 0.1).into_dyn();
        let grad_output = Array3::<f64>::from_elem((1, 4, 32), 0.05).into_dyn();

        let d_input = gqa
            .backward(&input, &grad_output)
            .expect("GQA+RoPE backward should succeed");

        assert_eq!(d_input.shape(), &[1, 4, 32]);

        for val in d_input.iter() {
            assert!(val.is_finite(), "GQA+RoPE gradient must be finite");
        }
    }

    /// Test GQA backward with causal masking.
    #[test]
    fn test_gqa_backward_causal() {
        let mut rng = scirs2_core::random::rng();
        let config = GroupedQueryAttentionConfig::new(2, 1, 8).with_causal(true);
        let gqa = GroupedQueryAttention::<f64>::new(16, config, &mut rng).expect("creation failed");

        let input = Array3::<f64>::from_elem((1, 4, 16), 0.1).into_dyn();
        let grad_output = Array3::<f64>::from_elem((1, 4, 16), 0.1).into_dyn();

        let d_input = gqa
            .backward(&input, &grad_output)
            .expect("GQA causal backward should succeed");

        assert_eq!(d_input.shape(), &[1, 4, 16]);

        for val in d_input.iter() {
            assert!(val.is_finite(), "GQA causal gradient must be finite");
        }
    }
}
