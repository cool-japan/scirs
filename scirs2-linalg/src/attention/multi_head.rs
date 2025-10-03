//! Multi-Head Attention Implementation
//!
//! This module provides the multi-head attention mechanism that computes
//! multiple attention heads in parallel and concatenates the results.

use scirs2_core::ndarray::{Array3, ArrayView2, ArrayView3};
use scirs2_core::numeric::{Float, NumAssignOps, Zero};
use std::ops::{Add, Div, Mul, Sub};

use super::utils::{attention, AttentionConfig, AttentionMask};
use crate::error::{LinalgError, LinalgResult};

/// Multi-Head Attention
///
/// Computes multiple attention heads in parallel and concatenates the results:
/// MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
/// where head_i = Attention(QW^Q_i, KW^K_i, VW^V_i)
///
/// # Arguments
///
/// * `query` - Query tensor of shape [batchsize, seq_len_q, d_model]
/// * `key` - Key tensor of shape [batchsize, seq_len_k, d_model]
/// * `value` - Value tensor of shape [batchsize, seq_len_v, d_model]
/// * `wq` - Query projection weights [d_model, d_model]
/// * `wk` - Key projection weights [d_model, d_model]
/// * `wv` - Value projection weights [d_model, d_model]
/// * `wo` - Output projection weights [d_model, d_model]
/// * `mask` - Optional mask to apply to attention weights
/// * `config` - Configuration for the attention mechanism
///
/// # Returns
///
/// * Output tensor of shape [batchsize, seq_len_q, d_model]
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub fn multi_head_attention<F>(
    query: &ArrayView3<F>,
    key: &ArrayView3<F>,
    value: &ArrayView3<F>,
    wq: &ArrayView2<F>,
    wk: &ArrayView2<F>,
    wv: &ArrayView2<F>,
    wo: &ArrayView2<F>,
    mask: Option<&AttentionMask>,
    config: &AttentionConfig,
) -> LinalgResult<Array3<F>>
where
    F: Float + Add + Mul + Div + Sub + NumAssignOps + Zero + std::fmt::Debug,
{
    // Extract dimensions
    let (batchsize, seq_len_q, d_model) = (query.shape()[0], query.shape()[1], query.shape()[2]);
    let seq_len_k = key.shape()[1];
    let seq_len_v = value.shape()[1];

    // Validate dimensions
    if key.shape()[2] != d_model || value.shape()[2] != d_model {
        return Err(LinalgError::DimensionError(format!(
            "Model dimensions must match: {}, {}, {}",
            d_model,
            key.shape()[2],
            value.shape()[2]
        )));
    }

    if wq.shape() != [d_model, d_model]
        || wk.shape() != [d_model, d_model]
        || wv.shape() != [d_model, d_model]
        || wo.shape() != [d_model, d_model]
    {
        return Err(LinalgError::DimensionError(
            "Weight matrices must have shape [d_model, d_model]".to_string(),
        ));
    }

    // Extract attention configuration
    let num_heads = config.num_heads;
    let head_dim = config.head_dim;
    let scale = match config.scale {
        Some(s) => F::from(s).ok_or_else(|| {
            LinalgError::ValueError("Failed to convert scale to target type".to_string())
        })?,
        None => {
            let head_dim_f64 = head_dim as f64;
            if head_dim_f64 <= 0.0 {
                return Err(LinalgError::ValueError(
                    "Head dimension must be positive".to_string(),
                ));
            }
            let default_scale = 1.0 / head_dim_f64.sqrt();
            F::from(default_scale).ok_or_else(|| {
                LinalgError::ValueError(
                    "Failed to convert default scale to target type".to_string(),
                )
            })?
        }
    };

    // Verify that d_model is compatible with num_heads and head_dim
    if d_model != num_heads * head_dim {
        return Err(LinalgError::ValueError(format!(
            "Model dimension ({d_model}) must equal num_heads ({num_heads}) * head_dim ({head_dim})"
        )));
    }

    // Project query, key, and value
    let mut q_proj = Array3::<F>::zeros((batchsize, seq_len_q, d_model));
    let mut k_proj = Array3::<F>::zeros((batchsize, seq_len_k, d_model));
    let mut v_proj = Array3::<F>::zeros((batchsize, seq_len_v, d_model));

    // Perform projections batch by batch
    for b in 0..batchsize {
        // Project query
        for i in 0..seq_len_q {
            for j in 0..d_model {
                let mut sum = F::zero();
                for k in 0..d_model {
                    sum += query[[b, i, k]] * wq[[k, j]];
                }
                q_proj[[b, i, j]] = sum;
            }
        }

        // Project key
        for i in 0..seq_len_k {
            for j in 0..d_model {
                let mut sum = F::zero();
                for k in 0..d_model {
                    sum += key[[b, i, k]] * wk[[k, j]];
                }
                k_proj[[b, i, j]] = sum;
            }
        }

        // Project value
        for i in 0..seq_len_v {
            for j in 0..d_model {
                let mut sum = F::zero();
                for k in 0..d_model {
                    sum += value[[b, i, k]] * wv[[k, j]];
                }
                v_proj[[b, i, j]] = sum;
            }
        }
    }

    // Reshape for multi-head attention
    // We need to effectively reshape to [batchsize, num_heads, seq_len, head_dim]
    // but will use separate tensors for each head to avoid complex reshaping
    let mut head_outputs = Vec::with_capacity(num_heads);

    for h in 0..num_heads {
        // Extract head-specific portions of the projected tensors
        let start_idx = h * head_dim;
        let _end_idx = start_idx + head_dim; // Used for debugging/clarity

        let q_head = q_proj.slice(scirs2_core::ndarray::s![
            ..,
            ..,
            start_idx..(start_idx + head_dim)
        ]);
        let k_head = k_proj.slice(scirs2_core::ndarray::s![
            ..,
            ..,
            start_idx..(start_idx + head_dim)
        ]);
        let v_head = v_proj.slice(scirs2_core::ndarray::s![
            ..,
            ..,
            start_idx..(start_idx + head_dim)
        ]);

        // Compute attention for this head
        let head_output = attention(&q_head, &k_head, &v_head, mask, scale)?;
        head_outputs.push(head_output);
    }

    // Concatenate head outputs along the last dimension
    let mut concat_output = Array3::<F>::zeros((batchsize, seq_len_q, d_model));

    for (h, head_output) in head_outputs.iter().enumerate().take(num_heads) {
        let start_idx = h * head_dim;
        let _end_idx = start_idx + head_dim; // Used for clarity

        for b in 0..batchsize {
            for i in 0..seq_len_q {
                for j in 0..head_dim {
                    concat_output[[b, i, start_idx + j]] = head_output[[b, i, j]];
                }
            }
        }
    }

    // Apply output projection
    let mut output = Array3::<F>::zeros((batchsize, seq_len_q, d_model));

    for b in 0..batchsize {
        for i in 0..seq_len_q {
            for j in 0..d_model {
                let mut sum = F::zero();
                for k in 0..d_model {
                    sum += concat_output[[b, i, k]] * wo[[k, j]];
                }
                output[[b, i, j]] = sum;
            }
        }
    }

    Ok(output)
}

/// Grouped Query Attention (GQA)
///
/// Implements grouped query attention as described in papers like
/// "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"
///
/// # Arguments
///
/// * `query` - Query tensor of shape [batchsize, seq_len_q, d_model]
/// * `key` - Key tensor of shape [batchsize, seq_len_k, d_model]
/// * `value` - Value tensor of shape [batchsize, seq_len_k, d_model]
/// * `wq` - Query projection weights [d_model, d_model]
/// * `wk` - Key projection weights [d_model, kv_dim]
/// * `wv` - Value projection weights [d_model, kv_dim]
/// * `wo` - Output projection weights [d_model, d_model]
/// * `mask` - Optional mask to apply to attention weights
/// * `num_heads` - Number of query heads
/// * `num_kv_heads` - Number of key/value heads
/// * `scale` - Scaling factor for dot product
///
/// # Returns
///
/// * Output tensor of shape [batchsize, seq_len_q, d_model]
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub fn grouped_query_attention<F>(
    query: &ArrayView3<F>,
    key: &ArrayView3<F>,
    value: &ArrayView3<F>,
    wq: &ArrayView2<F>,
    wk: &ArrayView2<F>,
    wv: &ArrayView2<F>,
    wo: &ArrayView2<F>,
    mask: Option<&AttentionMask>,
    num_heads: usize,
    num_kv_heads: usize,
    scale: F,
) -> LinalgResult<Array3<F>>
where
    F: Float + Add + Mul + Div + Sub + NumAssignOps + Zero + std::fmt::Debug,
{
    // Validate dimensions
    let (batchsize, seq_len_q, d_model) = (query.shape()[0], query.shape()[1], query.shape()[2]);
    let seq_len_k = key.shape()[1];

    // Validate kv head configuration
    if !num_heads.is_multiple_of(num_kv_heads) {
        return Err(LinalgError::ValueError(format!(
            "Number of query heads ({num_heads}) must be divisible by number of KV heads ({num_kv_heads})"
        )));
    }

    if !num_heads.is_multiple_of(num_kv_heads) {
        return Err(LinalgError::ValueError(format!(
            "Number of heads ({num_heads}) must be divisible by number of key-value heads ({num_kv_heads})"
        )));
    }
    let heads_per_kv = num_heads / num_kv_heads;
    let head_dim = d_model / num_heads;
    let kv_dim = num_kv_heads * head_dim;

    // Validate weight dimensions
    if wq.shape() != [d_model, d_model]
        || wk.shape() != [d_model, kv_dim]
        || wv.shape() != [d_model, kv_dim]
        || wo.shape() != [d_model, d_model]
    {
        return Err(LinalgError::DimensionError(
            "Weight matrices have incorrect dimensions".to_string(),
        ));
    }

    // Project query, key, and value
    let mut q_proj = Array3::<F>::zeros((batchsize, seq_len_q, d_model));
    let mut k_proj = Array3::<F>::zeros((batchsize, seq_len_k, kv_dim));
    let mut v_proj = Array3::<F>::zeros((batchsize, seq_len_k, kv_dim));

    // Perform projections batch by batch
    for b in 0..batchsize {
        // Project query
        for i in 0..seq_len_q {
            for j in 0..d_model {
                let mut sum = F::zero();
                for k in 0..d_model {
                    sum += query[[b, i, k]] * wq[[k, j]];
                }
                q_proj[[b, i, j]] = sum;
            }
        }

        // Project key
        for i in 0..seq_len_k {
            for j in 0..kv_dim {
                let mut sum = F::zero();
                for k in 0..d_model {
                    sum += key[[b, i, k]] * wk[[k, j]];
                }
                k_proj[[b, i, j]] = sum;
            }
        }

        // Project value
        for i in 0..seq_len_k {
            for j in 0..kv_dim {
                let mut sum = F::zero();
                for k in 0..d_model {
                    sum += value[[b, i, k]] * wv[[k, j]];
                }
                v_proj[[b, i, j]] = sum;
            }
        }
    }

    // Initialize output
    let mut concat_output = Array3::<F>::zeros((batchsize, seq_len_q, d_model));

    // Process each head
    for h in 0..num_heads {
        let kv_head_idx = h / heads_per_kv;

        // Extract head-specific portions of the projected tensors
        let q_start = h * head_dim;
        let q_end = q_start + head_dim;

        let kv_start = kv_head_idx * head_dim;
        let kv_end = kv_start + head_dim;

        // Extract query head
        let q_head = q_proj.slice(scirs2_core::ndarray::s![.., .., q_start..q_end]);

        // Extract key and value heads (shared across multiple query heads)
        let k_head = k_proj.slice(scirs2_core::ndarray::s![.., .., kv_start..kv_end]);
        let v_head = v_proj.slice(scirs2_core::ndarray::s![.., .., kv_start..kv_end]);

        // Compute attention for this head
        let head_output = attention(&q_head, &k_head, &v_head, mask, scale)?;

        // Add to concatenated output
        for b in 0..batchsize {
            for i in 0..seq_len_q {
                for j in 0..head_dim {
                    concat_output[[b, i, q_start + j]] = head_output[[b, i, j]];
                }
            }
        }
    }

    // Apply output projection
    let mut output = Array3::<F>::zeros((batchsize, seq_len_q, d_model));

    for b in 0..batchsize {
        for i in 0..seq_len_q {
            for j in 0..d_model {
                let mut sum = F::zero();
                for k in 0..d_model {
                    sum += concat_output[[b, i, k]] * wo[[k, j]];
                }
                output[[b, i, j]] = sum;
            }
        }
    }

    Ok(output)
}
