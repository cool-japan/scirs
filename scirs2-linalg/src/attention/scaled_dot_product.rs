//! Scaled Dot-Product Attention Implementation
//!
//! This module provides the core scaled dot-product attention mechanism
//! used in Transformer models, including optimized implementations for f32.

use scirs2_core::ndarray::{Array3, ArrayView3};
use scirs2_core::numeric::{Float, NumAssignOps, Zero};
use std::ops::{Add, Div, Mul, Sub};

use super::utils::{attention, AttentionMask};
use crate::blas_accelerated;
use crate::error::{check_dimensions, LinalgError, LinalgResult};

/// Scaled Dot-Product Attention
///
/// The standard attention mechanism used in Transformer models:
/// Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
///
/// # Arguments
///
/// * `query` - Query tensor of shape [batchsize, seq_len_q, d_model]
/// * `key` - Key tensor of shape [batchsize, seq_len_k, d_model]
/// * `value` - Value tensor of shape [batchsize, seq_len_k, d_model]
/// * `mask` - Optional mask to apply to attention weights
/// * `scale` - Scaling factor for dot product (default is 1/sqrt(d_k))
///
/// # Returns
///
/// * Output tensor of shape [batchsize, seq_len_q, d_model]
#[allow(dead_code)]
pub fn scaled_dot_product_attention<F>(
    query: &ArrayView3<F>,
    key: &ArrayView3<F>,
    value: &ArrayView3<F>,
    mask: Option<&AttentionMask>,
    scale: F,
) -> LinalgResult<Array3<F>>
where
    F: Float + Add + Mul + Div + Sub + NumAssignOps + Zero + std::fmt::Debug + 'static,
{
    // Special case for f32 - using runtime type checking
    if let Some(f32_result) = try_f32_attention(query, key, value, mask, scale) {
        return f32_result;
    }

    // Fall back to the generic implementation
    attention(query, key, value, mask, scale)
}

/// Try to use an optimized implementation for f32 type
#[allow(dead_code)]
fn try_f32_attention<F>(
    query: &ArrayView3<F>,
    key: &ArrayView3<F>,
    value: &ArrayView3<F>,
    mask: Option<&AttentionMask>,
    scale: F,
) -> Option<LinalgResult<Array3<F>>>
where
    F: Float + Add + Mul + Div + Sub + NumAssignOps + Zero + std::fmt::Debug + 'static,
{
    // Check if F is f32 and if we can use BLAS acceleration
    if std::any::TypeId::of::<F>() == std::any::TypeId::of::<f32>() && mask.is_none() {
        // SAFETY: We've already verified that F is f32, so this transmute is safe
        let query_f32: &ArrayView3<f32> = unsafe { std::mem::transmute(query) };
        let key_f32: &ArrayView3<f32> = unsafe { std::mem::transmute(key) };
        let value_f32: &ArrayView3<f32> = unsafe { std::mem::transmute(value) };
        let scale_f32: f32 = unsafe { *(&scale as *const F as *const f32) };

        // Use BLAS-accelerated attention for unmasked attention with f32
        let result = blas_attention_f32(query_f32, key_f32, value_f32, scale_f32);

        // Convert the result back to F type
        return Some(unsafe {
            std::mem::transmute::<Result<Array3<f32>, LinalgError>, Result<Array3<F>, LinalgError>>(
                result,
            )
        });
    }

    None
}

/// BLAS-accelerated attention implementation for f32
///
/// Uses BLAS for matrix multiplications to speed up the attention computation
#[allow(dead_code)]
fn blas_attention_f32(
    query: &ArrayView3<f32>,
    key: &ArrayView3<f32>,
    value: &ArrayView3<f32>,
    scale: f32,
) -> LinalgResult<Array3<f32>> {
    // Validate input dimensions
    let (batchsize, seq_len_q, d_model_q) = (query.shape()[0], query.shape()[1], query.shape()[2]);
    let (batchsize_k, seq_len_k, d_model_k) = (key.shape()[0], key.shape()[1], key.shape()[2]);
    let (batchsize_v, seq_len_v, d_model_v) =
        (value.shape()[0], value.shape()[1], value.shape()[2]);

    check_dimensions(
        batchsize == batchsize_k && batchsize == batchsize_v,
        format!("Batch sizes must match: {batchsize}, {batchsize_k}, {batchsize_v}"),
    )?;

    check_dimensions(
        seq_len_k == seq_len_v,
        format!("Key and value sequence lengths must match: {seq_len_k}, {seq_len_v}"),
    )?;

    check_dimensions(
        d_model_q == d_model_k,
        format!("Query and key dimensions must match: {d_model_q}, {d_model_k}"),
    )?;

    let mut result = Array3::<f32>::zeros((batchsize, seq_len_q, d_model_v));

    for b in 0..batchsize {
        // Extract batch slices
        let q_b = query.slice(scirs2_core::ndarray::s![b, .., ..]);
        let k_b = key.slice(scirs2_core::ndarray::s![b, .., ..]);
        let v_b = value.slice(scirs2_core::ndarray::s![b, .., ..]);

        // Compute scores using BLAS matrix multiplication: QK^T
        // First transpose the key matrix
        let k_b_t = k_b.t();

        // Use BLAS to compute Q @ K^T
        // We need to convert our views to the correct type for BLAS
        let q_b_view = q_b.view();
        let k_b_t_view = k_b_t.view();
        let scores = blas_accelerated::matmul(&q_b_view, &k_b_t_view)?;

        // Scale the scores
        let mut scores_scaled = scores.mapv(|x| x * scale);

        // Apply softmax along the last dimension
        for i in 0..seq_len_q {
            let mut row = scores_scaled.slice_mut(scirs2_core::ndarray::s![i, ..]);

            // Find the maximum value for numerical stability
            let max_val = row.fold(f32::NEG_INFINITY, |max, &x| max.max(x));

            // Subtract max value and exponentiate
            let mut sum = 0.0;
            for j in 0..seq_len_k {
                let exp_val = (row[j] - max_val).exp();
                row[j] = exp_val;
                sum += exp_val;
            }

            // Normalize
            if sum > 0.0 {
                for j in 0..seq_len_k {
                    row[j] /= sum;
                }
            }
        }

        // Use BLAS to compute attention_weights @ V
        let scores_view = scores_scaled.view();
        let v_b_view = v_b.view();
        let output = blas_accelerated::matmul(&scores_view, &v_b_view)?;

        // Store the result for this batch
        result
            .slice_mut(scirs2_core::ndarray::s![b, .., ..])
            .assign(&output);
    }

    Ok(result)
}

/// Masked Attention - Applies a custom mask to attention
///
/// # Arguments
///
/// * `query` - Query tensor of shape [batchsize, seq_len_q, d_model]
/// * `key` - Key tensor of shape [batchsize, seq_len_k, d_model]
/// * `value` - Value tensor of shape [batchsize, seq_len_k, d_model]
/// * `mask` - The mask to apply to attention weights
/// * `scale` - Scaling factor for dot product
///
/// # Returns
///
/// * Output tensor of shape [batchsize, seq_len_q, d_model]
#[allow(dead_code)]
pub fn masked_attention<F>(
    query: &ArrayView3<F>,
    key: &ArrayView3<F>,
    value: &ArrayView3<F>,
    mask: &AttentionMask,
    scale: F,
) -> LinalgResult<Array3<F>>
where
    F: Float + Add + Mul + Div + Sub + NumAssignOps + Zero + std::fmt::Debug,
{
    attention(query, key, value, Some(mask), scale)
}

/// Causal Attention - Implements attention with causal masking
///
/// Ensures each position can only attend to previous positions (and itself),
/// which is necessary for autoregressive models like GPT.
///
/// # Arguments
///
/// * `query` - Query tensor of shape [batchsize, seq_len_q, d_model]
/// * `key` - Key tensor of shape [batchsize, seq_len_k, d_model]
/// * `value` - Value tensor of shape [batchsize, seq_len_k, d_model]
/// * `scale` - Scaling factor for dot product
///
/// # Returns
///
/// * Output tensor of shape [batchsize, seq_len_q, d_model]
#[allow(dead_code)]
pub fn causal_attention<F>(
    query: &ArrayView3<F>,
    key: &ArrayView3<F>,
    value: &ArrayView3<F>,
    scale: F,
) -> LinalgResult<Array3<F>>
where
    F: Float + Add + Mul + Div + Sub + NumAssignOps + Zero + std::fmt::Debug,
{
    let mask = AttentionMask::Causal;
    attention(query, key, value, Some(&mask), scale)
}
