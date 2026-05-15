//! Multi-Query Attention (MQA) implementation
//!
//! This module implements Multi-Query Attention as described in:
//! "Fast Transformer Decoding: One Write-Head is All You Need"
//! by Noam Shazeer (2019).
//!
//! In MQA, all query heads share a single set of key and value projections.
//! This drastically reduces the KV cache size during autoregressive generation
//! (by a factor of `num_heads`), while maintaining most of the quality of
//! standard multi-head attention.
//!
//! When `num_kv_heads == num_heads`, MQA degenerates to standard MHA.

use crate::error::{NeuralError, Result};
use crate::layers::Layer;
use scirs2_core::ndarray::{s, Array, Array2, Array4, IxDyn, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign};
use scirs2_core::random::{Rng, RngExt};
use std::fmt::Debug;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Xavier-uniform weight initialisation returning an IxDyn array.
fn mk_weight<F: Float, R: Rng>(rows: usize, cols: usize, rng: &mut R) -> Result<Array<F, IxDyn>> {
    let scale = (6.0_f64 / (rows + cols) as f64).sqrt();
    let mut data = Vec::with_capacity(rows * cols);
    for _ in 0..(rows * cols) {
        let x: f64 = rng.random_range(-scale..scale);
        let f = F::from(x)
            .ok_or_else(|| NeuralError::InvalidArchitecture("xavier cast failed".into()))?;
        data.push(f);
    }
    Array::from_shape_vec(IxDyn(&[rows, cols]), data)
        .map_err(|e| NeuralError::InvalidArchitecture(format!("mk_weight: {e}")))
}

/// Softmax over a mutable slice (in-place, numerically stable).
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
// KV Cache
// ---------------------------------------------------------------------------

/// Key-Value cache for autoregressive generation
///
/// Stores past key and value tensors so they do not need to be recomputed
/// during incremental decoding.
#[derive(Debug, Clone)]
pub struct KvCache<F: Float> {
    /// Cached keys: [batch, past_len, num_kv_heads, head_dim]
    pub keys: Array<F, IxDyn>,
    /// Cached values: [batch, past_len, num_kv_heads, head_dim]
    pub values: Array<F, IxDyn>,
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for Multi-Query Attention
#[derive(Debug, Clone)]
pub struct MultiQueryAttentionConfig {
    /// Number of query heads
    pub num_heads: usize,
    /// Number of KV heads (1 = pure MQA, num_heads = standard MHA)
    pub num_kv_heads: usize,
    /// Per-head dimension
    pub head_dim: usize,
    /// Dropout probability
    pub dropout_prob: f64,
    /// Whether to apply causal masking
    pub causal: bool,
}

impl Default for MultiQueryAttentionConfig {
    fn default() -> Self {
        Self {
            num_heads: 8,
            num_kv_heads: 1,
            head_dim: 64,
            dropout_prob: 0.0,
            causal: false,
        }
    }
}

impl MultiQueryAttentionConfig {
    /// Create a pure MQA config (1 KV head)
    pub fn new(num_heads: usize, head_dim: usize) -> Self {
        Self {
            num_heads,
            num_kv_heads: 1,
            head_dim,
            ..Default::default()
        }
    }

    /// Set number of KV heads (1 = MQA, num_heads = MHA)
    pub fn with_num_kv_heads(mut self, n: usize) -> Self {
        self.num_kv_heads = n;
        self
    }

    /// Enable or disable causal masking
    pub fn with_causal(mut self, causal: bool) -> Self {
        self.causal = causal;
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

/// Multi-Query Attention layer
///
/// Projects queries with `num_heads` independent heads but uses only
/// `num_kv_heads` (default 1) shared key/value heads.
///
/// # Input
/// 3D tensor `[batch, seq_len, d_model]`
///
/// # Output
/// 3D tensor `[batch, seq_len, d_model]`
///
/// # Examples
///
/// ```rust
/// use scirs2_neural::layers::{MultiQueryAttention, MultiQueryAttentionConfig, Layer};
/// use scirs2_core::ndarray::Array3;
/// use scirs2_core::random::rng;
///
/// let mut rng = rng();
/// let config = MultiQueryAttentionConfig::new(4, 16); // 4 Q heads, 1 KV head
/// let mqa = MultiQueryAttention::<f64>::new(64, config, &mut rng).expect("failed");
///
/// let input = Array3::<f64>::from_elem((2, 8, 64), 0.1).into_dyn();
/// let output = mqa.forward(&input).expect("failed");
/// assert_eq!(output.shape(), &[2, 8, 64]);
/// ```
#[derive(Debug)]
pub struct MultiQueryAttention<F: Float + Debug + Send + Sync + NumAssign> {
    d_model: usize,
    config: MultiQueryAttentionConfig,
    /// [d_model, num_heads * head_dim]
    w_q: Array<F, IxDyn>,
    /// [d_model, num_kv_heads * head_dim]
    w_k: Array<F, IxDyn>,
    /// [d_model, num_kv_heads * head_dim]
    w_v: Array<F, IxDyn>,
    /// [num_heads * head_dim, d_model]
    w_o: Array<F, IxDyn>,
    scale: F,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + 'static + NumAssign> MultiQueryAttention<F> {
    /// Create a new Multi-Query Attention layer
    pub fn new<R: Rng>(
        d_model: usize,
        config: MultiQueryAttentionConfig,
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
        let kv_dim = config.num_kv_heads * config.head_dim;

        if q_dim != d_model {
            return Err(NeuralError::InvalidArchitecture(format!(
                "num_heads * head_dim ({q_dim}) must equal d_model ({d_model})"
            )));
        }

        let w_q = mk_weight(d_model, q_dim, rng)?;
        let w_k = mk_weight(d_model, kv_dim, rng)?;
        let w_v = mk_weight(d_model, kv_dim, rng)?;
        let w_o = mk_weight(q_dim, d_model, rng)?;

        let scale = F::one()
            / F::from(config.head_dim)
                .ok_or_else(|| NeuralError::InvalidArchitecture("scale cast".into()))?
                .sqrt();

        Ok(Self {
            d_model,
            config,
            w_q,
            w_k,
            w_v,
            w_o,
            scale,
        })
    }

    /// Forward pass with optional KV cache for autoregressive generation
    ///
    /// # Arguments
    /// * `input` - [batch, seq_len, d_model]
    /// * `past_kv` - Optional past KV cache
    ///
    /// # Returns
    /// (output [batch, seq_len, d_model], updated KV cache)
    pub fn forward_with_cache(
        &self,
        input: &Array<F, IxDyn>,
        past_kv: Option<&KvCache<F>>,
    ) -> Result<(Array<F, IxDyn>, KvCache<F>)> {
        if input.ndim() != 3 {
            return Err(NeuralError::InvalidArchitecture(format!(
                "MQA expects 3D input, got {}D",
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
        let q_4d =
            self.project_and_reshape(input, &self.w_q, batch, seq_len, num_heads, head_dim)?;
        let k_new =
            self.project_and_reshape(input, &self.w_k, batch, seq_len, num_kv_heads, head_dim)?;
        let v_new =
            self.project_and_reshape(input, &self.w_v, batch, seq_len, num_kv_heads, head_dim)?;

        // Concatenate with past cache if provided
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

        // Build updated cache
        let new_cache = KvCache {
            keys: k_4d.clone().into_dyn(),
            values: v_4d.clone().into_dyn(),
        };

        // Compute attention
        // Q: [batch, seq_len, num_heads, head_dim]
        // K, V: [batch, total_kv_len, num_kv_heads, head_dim]
        let mut output_4d = Array4::<F>::zeros((batch, seq_len, num_heads, head_dim));

        for b in 0..batch {
            for kv_h in 0..num_kv_heads {
                let q_h_start = kv_h * group_size;
                let q_h_end = q_h_start + group_size;

                for q_h in q_h_start..q_h_end {
                    for t in 0..seq_len {
                        // Compute attention scores
                        let global_t = if past_kv.is_some() {
                            let past_len = past_kv.map(|c| c.keys.shape()[1]).unwrap_or(0);
                            past_len + t
                        } else {
                            t
                        };

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

                        // Weighted sum of values
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

        // Reshape to [batch, seq_len, d_model] and project output
        let output_3d = output_4d
            .into_shape_with_order((batch, seq_len, d_model))
            .map_err(|e| NeuralError::InferenceError(format!("reshape output: {e}")))?;

        let output_2d = output_3d
            .into_shape_with_order((batch * seq_len, d_model))
            .map_err(|e| NeuralError::InferenceError(format!("reshape for O proj: {e}")))?;

        let w_o_2d = self
            .w_o
            .view()
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .map_err(|_| NeuralError::InferenceError("O weights 2D".into()))?;

        let final_out = output_2d.dot(&w_o_2d);

        let result = final_out
            .into_shape_with_order((batch, seq_len, d_model))
            .map_err(|e| NeuralError::InferenceError(format!("reshape final: {e}")))?;

        Ok((result.into_dyn(), new_cache))
    }

    /// Project input and reshape to [batch, seq, heads, head_dim]
    fn project_and_reshape(
        &self,
        input: &Array<F, IxDyn>,
        weight: &Array<F, IxDyn>,
        batch: usize,
        seq: usize,
        heads: usize,
        head_dim: usize,
    ) -> Result<Array4<F>> {
        let d_model = input.shape()[2];
        let proj_dim = heads * head_dim;

        // [batch * seq, d_model] @ [d_model, proj_dim] = [batch * seq, proj_dim]
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
            .map_err(|_| NeuralError::InferenceError("weight to Ix2".into()))?;

        let projected = input_2d_view.dot(&w_2d);

        projected
            .into_shape_with_order((batch, seq, heads, head_dim))
            .map_err(|e| NeuralError::InferenceError(format!("reshape projected: {e}")))
    }

    /// Concatenate past cache with new KV along the seq dimension
    fn concat_cache(
        &self,
        past: &Array<F, IxDyn>,
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

        // Copy past
        for b in 0..batch {
            for t in 0..past_len {
                for h in 0..heads {
                    for d in 0..head_dim {
                        result[[b, t, h, d]] = past[[b, t, h, d]];
                    }
                }
            }
            // Copy new
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

    /// Get configuration
    pub fn config(&self) -> &MultiQueryAttentionConfig {
        &self.config
    }

    /// Get model dimension
    pub fn d_model(&self) -> usize {
        self.d_model
    }
}

impl<F> Layer<F> for MultiQueryAttention<F>
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
                "MQA backward expects 3D input, got {}D",
                input.ndim()
            )));
        }

        let shape = input.shape();
        let (batch, seq_len, d_model) = (shape[0], shape[1], shape[2]);

        if d_model != self.d_model {
            return Err(NeuralError::InvalidArchitecture(format!(
                "MQA backward: input dim {d_model} != d_model {}",
                self.d_model
            )));
        }

        let num_heads = self.config.num_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = self.config.head_dim;
        let group_size = num_heads / num_kv_heads;

        // -----------------------------------------------------------------------
        // Re-compute forward quantities (Q, K, V, scores, attn_weights)
        // -----------------------------------------------------------------------
        let q_4d =
            self.project_and_reshape(input, &self.w_q, batch, seq_len, num_heads, head_dim)?;
        let k_4d =
            self.project_and_reshape(input, &self.w_k, batch, seq_len, num_kv_heads, head_dim)?;
        let v_4d =
            self.project_and_reshape(input, &self.w_v, batch, seq_len, num_kv_heads, head_dim)?;

        // Attention weights: [batch, num_heads, seq_q, seq_k]
        let mut attn_weights = Array::zeros(IxDyn(&[batch, num_heads, seq_len, seq_len]));

        for b in 0..batch {
            for kv_h in 0..num_kv_heads {
                let q_h_start = kv_h * group_size;
                let q_h_end = q_h_start + group_size;

                for q_h in q_h_start..q_h_end {
                    for t in 0..seq_len {
                        // Scores: dot(Q[b,t,q_h,:], K[b,:,kv_h,:]) * scale
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

                        // Store softmax output
                        for s_idx in 0..seq_len {
                            attn_weights[[b, q_h, t, s_idx]] = scores[s_idx];
                        }
                    }
                }
            }
        }

        // -----------------------------------------------------------------------
        // Backward through output projection: output = concat @ W_o
        // d_concat = grad_output @ W_o^T  [batch*seq, d_model] @ [d_model, q_dim] => [batch*seq, q_dim]
        // -----------------------------------------------------------------------
        let grad_out_2d = grad_output
            .view()
            .into_shape_with_order(IxDyn(&[batch * seq_len, d_model]))
            .map_err(|e| NeuralError::InferenceError(format!("backward reshape grad_out: {e}")))?;

        let w_o_2d = self
            .w_o
            .view()
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .map_err(|_| NeuralError::InferenceError("backward: W_o to Ix2".into()))?;

        let grad_out_2d_typed = grad_out_2d
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .map_err(|_| NeuralError::InferenceError("backward: grad_out to Ix2".into()))?;

        // d_concat: [batch*seq, q_dim]
        let d_concat_2d = grad_out_2d_typed.dot(&w_o_2d.t());

        let q_dim = num_heads * head_dim;
        // d_concat reshaped: [batch, seq_len, num_heads, head_dim]
        let d_concat_4d: Array4<F> = d_concat_2d
            .into_shape_with_order((batch, seq_len, num_heads, head_dim))
            .map_err(|e| NeuralError::InferenceError(format!("backward d_concat reshape: {e}")))?;

        // -----------------------------------------------------------------------
        // Backward through attention for each head
        // For each (b, kv_h, q_h, t):
        //   d_V += attn_weights[b,q_h,t,:]^T @ d_concat[b,t,q_h,:]   (outer)
        //   d_scores = d_concat[b,t,q_h,:] @ V^T
        //   d_softmax = softmax_backward(attn, d_scores)
        //   d_Q[b,t,q_h,:] += d_softmax @ K[b,:,kv_h,:] * scale
        //   d_K[b,:,kv_h,:] += d_softmax^T @ Q[b,t,q_h,:] * scale   (outer)
        // -----------------------------------------------------------------------
        let kv_dim = num_kv_heads * head_dim;

        let mut d_q_4d = Array4::<F>::zeros((batch, seq_len, num_heads, head_dim));
        let mut d_k_4d = Array4::<F>::zeros((batch, seq_len, num_kv_heads, head_dim));
        let mut d_v_4d = Array4::<F>::zeros((batch, seq_len, num_kv_heads, head_dim));

        for b in 0..batch {
            for kv_h in 0..num_kv_heads {
                let q_h_start = kv_h * group_size;
                let q_h_end = q_h_start + group_size;

                for q_h in q_h_start..q_h_end {
                    for t in 0..seq_len {
                        // Gradient of attn_out[b,t,q_h,d] = sum_s attn[b,q_h,t,s] * V[b,s,kv_h,d]
                        // d_attn_out: [head_dim]  (from d_concat_4d[b,t,q_h,:])
                        // d_V[b,s,kv_h,d] += attn[b,q_h,t,s] * d_attn_out[d]
                        for s_idx in 0..seq_len {
                            let a = attn_weights[[b, q_h, t, s_idx]];
                            for d in 0..head_dim {
                                d_v_4d[[b, s_idx, kv_h, d]] += a * d_concat_4d[[b, t, q_h, d]];
                            }
                        }

                        // d_scores_raw[s] = sum_d d_attn_out[d] * V[b,s,kv_h,d]   (before softmax)
                        let mut d_scores_raw = vec![F::zero(); seq_len];
                        for s_idx in 0..seq_len {
                            let mut sum = F::zero();
                            for d in 0..head_dim {
                                sum += d_concat_4d[[b, t, q_h, d]] * v_4d[[b, s_idx, kv_h, d]];
                            }
                            d_scores_raw[s_idx] = sum;
                        }

                        // Softmax backward: d_pre[s] = attn[s] * (d_raw[s] - dot(attn, d_raw))
                        let dot_attn_d: F = (0..seq_len).fold(F::zero(), |acc, s| {
                            acc + attn_weights[[b, q_h, t, s]] * d_scores_raw[s]
                        });

                        let mut d_pre_scores = vec![F::zero(); seq_len];
                        for s_idx in 0..seq_len {
                            let a = attn_weights[[b, q_h, t, s_idx]];
                            // For causal masked positions, attn=0 so this contributes 0
                            d_pre_scores[s_idx] = a * (d_scores_raw[s_idx] - dot_attn_d);
                        }

                        // d_pre_scores = d_pre_scores * scale  (since scores = dot * scale)
                        for d_pre in d_pre_scores.iter_mut() {
                            *d_pre *= self.scale;
                        }

                        // d_Q[b,t,q_h,:] += sum_s d_pre[s] * K[b,s,kv_h,:]
                        for s_idx in 0..seq_len {
                            let dp = d_pre_scores[s_idx];
                            for d in 0..head_dim {
                                d_q_4d[[b, t, q_h, d]] += dp * k_4d[[b, s_idx, kv_h, d]];
                            }
                        }

                        // d_K[b,s,kv_h,:] += d_pre[s] * Q[b,t,q_h,:]
                        for s_idx in 0..seq_len {
                            let dp = d_pre_scores[s_idx];
                            for d in 0..head_dim {
                                d_k_4d[[b, s_idx, kv_h, d]] += dp * q_4d[[b, t, q_h, d]];
                            }
                        }
                    }
                }
            }
        }

        // -----------------------------------------------------------------------
        // Backward through Q/K/V projections: d_input = d_Q @ W_q^T + d_K @ W_k^T + d_V @ W_v^T
        // Each projection: x [batch*seq, d_model] @ W [d_model, proj_dim]
        // backward: d_x [batch*seq, proj_dim] @ W^T [proj_dim, d_model] => [batch*seq, d_model]
        // -----------------------------------------------------------------------

        // Reshape d_q: [batch*seq, q_dim]
        let d_q_2d = d_q_4d
            .into_shape_with_order((batch * seq_len, q_dim))
            .map_err(|e| NeuralError::InferenceError(format!("backward d_q reshape: {e}")))?;

        // Reshape d_k: [batch*seq, kv_dim]
        let d_k_2d = d_k_4d
            .into_shape_with_order((batch * seq_len, kv_dim))
            .map_err(|e| NeuralError::InferenceError(format!("backward d_k reshape: {e}")))?;

        // Reshape d_v: [batch*seq, kv_dim]
        let d_v_2d = d_v_4d
            .into_shape_with_order((batch * seq_len, kv_dim))
            .map_err(|e| NeuralError::InferenceError(format!("backward d_v reshape: {e}")))?;

        let w_q_2d = self
            .w_q
            .view()
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .map_err(|_| NeuralError::InferenceError("backward: W_q to Ix2".into()))?;
        let w_k_2d = self
            .w_k
            .view()
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .map_err(|_| NeuralError::InferenceError("backward: W_k to Ix2".into()))?;
        let w_v_2d = self
            .w_v
            .view()
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .map_err(|_| NeuralError::InferenceError("backward: W_v to Ix2".into()))?;

        // d_input: [batch*seq, d_model]
        let d_input_from_q = d_q_2d.dot(&w_q_2d.t());
        let d_input_from_k = d_k_2d.dot(&w_k_2d.t());
        let d_input_from_v = d_v_2d.dot(&w_v_2d.t());

        let d_input_2d = d_input_from_q + d_input_from_k + d_input_from_v;

        let d_input = d_input_2d
            .into_shape_with_order(IxDyn(&[batch, seq_len, d_model]))
            .map_err(|e| NeuralError::InferenceError(format!("backward d_input reshape: {e}")))?;

        Ok(d_input)
    }

    fn update(&mut self, _learning_rate: F) -> Result<()> {
        Ok(())
    }

    fn layer_type(&self) -> &str {
        "MultiQueryAttention"
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
    fn test_mqa_creation() {
        let mut rng = scirs2_core::random::rng();
        let config = MultiQueryAttentionConfig::new(4, 16); // 4 Q heads, 1 KV head
        let mqa = MultiQueryAttention::<f64>::new(64, config, &mut rng);
        assert!(mqa.is_ok());
    }

    #[test]
    fn test_mqa_forward_shape() {
        let mut rng = scirs2_core::random::rng();
        let config = MultiQueryAttentionConfig::new(4, 16);
        let mqa = MultiQueryAttention::<f64>::new(64, config, &mut rng).expect("creation failed");

        let input = Array3::<f64>::from_elem((2, 8, 64), 0.1).into_dyn();
        let output = mqa.forward(&input).expect("forward failed");
        assert_eq!(output.shape(), &[2, 8, 64]);
    }

    #[test]
    fn test_mqa_kv_cache() {
        let mut rng = scirs2_core::random::rng();
        let config = MultiQueryAttentionConfig::new(4, 16).with_causal(true);
        let mqa = MultiQueryAttention::<f64>::new(64, config, &mut rng).expect("creation failed");

        // First step: process prefix
        let prefix = Array3::<f64>::from_elem((1, 4, 64), 0.1).into_dyn();
        let (out1, cache1) = mqa
            .forward_with_cache(&prefix, None)
            .expect("step 1 failed");
        assert_eq!(out1.shape(), &[1, 4, 64]);
        assert_eq!(cache1.keys.shape()[1], 4);
        assert_eq!(cache1.values.shape()[1], 4);

        // Second step: process one new token with cache
        let new_token = Array3::<f64>::from_elem((1, 1, 64), 0.2).into_dyn();
        let (out2, cache2) = mqa
            .forward_with_cache(&new_token, Some(&cache1))
            .expect("step 2 failed");
        assert_eq!(out2.shape(), &[1, 1, 64]);
        assert_eq!(cache2.keys.shape()[1], 5); // 4 + 1
        assert_eq!(cache2.values.shape()[1], 5);
    }

    #[test]
    fn test_mqa_with_num_heads_equals_mha() {
        // When num_kv_heads == num_heads, MQA should behave like MHA
        let mut rng = scirs2_core::random::rng();
        let config = MultiQueryAttentionConfig::new(4, 16).with_num_kv_heads(4); // same as num_heads = MHA
        let mqa = MultiQueryAttention::<f64>::new(64, config, &mut rng).expect("creation failed");

        let input = Array3::<f64>::from_elem((1, 6, 64), 0.15).into_dyn();
        let output = mqa.forward(&input).expect("forward failed");
        assert_eq!(output.shape(), &[1, 6, 64]);

        // Output should be finite
        for val in output.iter() {
            assert!(val.is_finite(), "MHA-mode output has non-finite value");
        }
    }

    #[test]
    fn test_mqa_causal_masking() {
        let mut rng = scirs2_core::random::rng();
        let config = MultiQueryAttentionConfig::new(2, 8).with_causal(true);
        let mqa = MultiQueryAttention::<f64>::new(16, config, &mut rng).expect("creation failed");

        let mut input = Array3::<f64>::zeros((1, 6, 16));
        for t in 0..6 {
            for d in 0..16 {
                input[[0, t, d]] = (t as f64 + 1.0) * 0.1 + d as f64 * 0.01;
            }
        }

        let output = mqa.forward(&input.into_dyn()).expect("forward failed");
        assert_eq!(output.shape(), &[1, 6, 16]);

        for val in output.iter() {
            assert!(val.is_finite(), "causal output non-finite");
        }
    }

    #[test]
    fn test_mqa_invalid_config() {
        let mut rng = scirs2_core::random::rng();

        // num_heads not divisible by num_kv_heads
        let config = MultiQueryAttentionConfig::new(5, 16).with_num_kv_heads(3);
        let result = MultiQueryAttention::<f64>::new(80, config, &mut rng);
        assert!(result.is_err());
    }

    #[test]
    fn test_mqa_parameter_count() {
        let mut rng = scirs2_core::random::rng();
        let config = MultiQueryAttentionConfig::new(4, 16); // 1 KV head
        let mqa = MultiQueryAttention::<f64>::new(64, config, &mut rng).expect("creation failed");

        // Q: 64 * 64 = 4096
        // K: 64 * 16 = 1024
        // V: 64 * 16 = 1024
        // O: 64 * 64 = 4096
        assert_eq!(mqa.parameter_count(), 4096 + 1024 + 1024 + 4096);
    }

    /// Test that backward returns correct output shape matching the input shape.
    #[test]
    fn test_mqa_backward_shape() {
        let mut rng = scirs2_core::random::rng();
        // num_heads=2, head_dim=8 => d_model=16, num_kv_heads=1 (pure MQA)
        let config = MultiQueryAttentionConfig::new(2, 8);
        let mqa = MultiQueryAttention::<f64>::new(16, config, &mut rng).expect("creation failed");

        let input = Array3::<f64>::from_elem((2, 5, 16), 0.1).into_dyn();
        let grad_output = Array3::<f64>::from_elem((2, 5, 16), 0.01).into_dyn();

        let d_input = mqa
            .backward(&input, &grad_output)
            .expect("backward should succeed");

        // Gradient w.r.t. input must have the same shape as input
        assert_eq!(
            d_input.shape(),
            input.shape(),
            "backward gradient shape must match input shape"
        );

        // All gradients must be finite
        for val in d_input.iter() {
            assert!(val.is_finite(), "backward gradient must be finite");
        }
    }

    /// Analytical test for the degenerate seq_len=1 case.
    ///
    /// When seq_len=1, softmax of a single element is exactly 1.0.
    /// The softmax Jacobian is `diag(a) - a @ a^T` = 0 for a scalar,
    /// so d_scores = 0, and d_Q = 0, d_K = 0.
    ///
    /// Only the V path survives:
    ///   output = (input @ W_v) @ W_o   (with trivial attention weight=1)
    ///   d_input_from_v = grad_output @ W_o^T @ W_v^T
    #[test]
    fn test_mqa_backward_analytical_seq1() {
        let mut rng = scirs2_core::random::rng();
        // 1 head, 1 kv head, head_dim=4 => d_model=4
        let config = MultiQueryAttentionConfig::new(1, 4).with_num_kv_heads(1);
        let mqa = MultiQueryAttention::<f64>::new(4, config, &mut rng).expect("creation failed");

        // batch=1, seq=1, d_model=4
        let input = Array3::<f64>::from_elem((1, 1, 4), 0.5).into_dyn();
        let grad_out = Array3::<f64>::from_elem((1, 1, 4), 1.0).into_dyn();

        let d_input = mqa.backward(&input, &grad_out).expect("backward failed");

        assert_eq!(d_input.shape(), &[1, 1, 4]);

        // Analytical: d_input_from_v = grad_out @ W_o^T @ W_v^T
        // (d_input_from_q and d_input_from_k are 0 for seq_len=1)
        use scirs2_core::ndarray::{Array2, Ix2};

        let w_v_2d = mqa.w_v.view().into_dimensionality::<Ix2>().unwrap();
        let w_o_2d = mqa.w_o.view().into_dimensionality::<Ix2>().unwrap();

        // grad_out: [1, 4], W_o: [4,4], W_v: [4,4]
        let go = Array2::<f64>::from_elem((1, 4), 1.0);
        let expected = go.dot(&w_o_2d.t()).dot(&w_v_2d.t());

        for d in 0..4 {
            let computed = d_input[[0, 0, d]];
            let analytic = expected[[0, d]];
            assert!(
                (computed - analytic).abs() < 1e-10,
                "dim {d}: computed={computed:.6} vs analytic={analytic:.6}"
            );
        }
    }

    /// Test backward with num_kv_heads > 1 (GQA-mode within MQA layer).
    #[test]
    fn test_mqa_backward_multi_kv_heads() {
        let mut rng = scirs2_core::random::rng();
        // 4 Q heads, 2 KV heads, head_dim=8 => d_model=32
        let config = MultiQueryAttentionConfig::new(4, 8).with_num_kv_heads(2);
        let mqa = MultiQueryAttention::<f64>::new(32, config, &mut rng).expect("creation failed");

        let input = Array3::<f64>::from_elem((1, 3, 32), 0.1).into_dyn();
        let grad_output = Array3::<f64>::from_elem((1, 3, 32), 0.05).into_dyn();

        let d_input = mqa
            .backward(&input, &grad_output)
            .expect("backward should succeed");

        assert_eq!(d_input.shape(), &[1, 3, 32]);

        for val in d_input.iter() {
            assert!(val.is_finite(), "gradient must be finite");
        }
    }
}
