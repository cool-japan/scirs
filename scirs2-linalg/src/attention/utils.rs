//! Utility functions for attention mechanisms
//!
//! This module provides basic building blocks and utility functions
//! used by various attention implementations.

use scirs2_core::ndarray::{Array2, Array3, ArrayView3};
use scirs2_core::numeric::{Float, NumAssignOps, Zero};
use std::ops::{Add, Div, Mul, Sub};

use crate::error::{check_dimensions, LinalgError, LinalgResult};

/// Mask types for attention mechanisms
#[derive(Debug, Clone)]
pub enum AttentionMask {
    /// Additive mask (added to attention scores before softmax)
    /// Shape: [batchsize, seq_len_q, seq_len_k] or [1, seq_len_q, seq_len_k]
    Additive(Array3<f32>),

    /// Multiplicative mask (multiplied with attention scores after softmax)
    /// Shape: [batchsize, seq_len_q, seq_len_k] or [1, seq_len_q, seq_len_k]
    Multiplicative(Array3<f32>),

    /// Boolean mask (True means attend, False means don't attend)
    /// Shape: [batchsize, seq_len_q, seq_len_k] or [1, seq_len_q, seq_len_k]
    Boolean(Array3<bool>),

    /// Causal mask (upper triangular with -inf)
    /// Automatically sized to match sequence lengths
    Causal,
}

/// Configuration for attention mechanisms
#[derive(Debug, Clone)]
pub struct AttentionConfig {
    /// Number of attention heads
    pub num_heads: usize,

    /// Dimension of each attention head
    pub head_dim: usize,

    /// Dropout probability (0.0 means no dropout)
    pub dropout_prob: f32,

    /// Whether to use causal masking (for autoregressive models)
    pub causal: bool,

    /// Custom scaling factor (default is 1/sqrt(d_k))
    pub scale: Option<f32>,
}

impl Default for AttentionConfig {
    fn default() -> Self {
        Self {
            num_heads: 8,
            head_dim: 64,
            dropout_prob: 0.0,
            causal: false,
            scale: None,
        }
    }
}

/// Basic attention function - the building block for all other attention variants
///
/// This implements the standard attention mechanism: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
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
pub fn attention<F>(
    query: &ArrayView3<F>,
    key: &ArrayView3<F>,
    value: &ArrayView3<F>,
    mask: Option<&AttentionMask>,
    scale: F,
) -> LinalgResult<Array3<F>>
where
    F: Float + Add + Mul + Div + Sub + NumAssignOps + Zero + std::fmt::Debug,
{
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

    let mut result = Array3::<F>::zeros((batchsize, seq_len_q, d_model_v));

    for b in 0..batchsize {
        // Calculate attention scores: QK^T [seq_len_q, seq_len_k]
        let q_b = query.slice(scirs2_core::ndarray::s![b, .., ..]);
        let k_b = key.slice(scirs2_core::ndarray::s![b, .., ..]);
        let v_b = value.slice(scirs2_core::ndarray::s![b, .., ..]);

        // Compute scores as matrix multiplication: query @ key.transpose()
        let mut scores = Array2::<F>::zeros((seq_len_q, seq_len_k));

        for i in 0..seq_len_q {
            for j in 0..seq_len_k {
                let mut dot_product = F::zero();
                for k in 0..d_model_q {
                    dot_product += q_b[[i, k]] * k_b[[j, k]];
                }
                scores[[i, j]] = dot_product * scale;
            }
        }

        // Apply mask if provided
        if let Some(mask_ref) = mask {
            apply_mask(&mut scores, mask_ref, b)?;
        }

        // Apply softmax along the last dimension
        for i in 0..seq_len_q {
            let mut row = scores.slice_mut(scirs2_core::ndarray::s![i, ..]);

            // Compute softmax manually for numerical stability
            // First find the maximum value for numerical stability
            let max_val = row.fold(F::neg_infinity(), |max, &x| if x > max { x } else { max });

            // Subtract max value and exponentiate
            let mut sum = F::zero();
            for j in 0..seq_len_k {
                let exp_val = (row[j] - max_val).exp();
                row[j] = exp_val;
                sum += exp_val;
            }

            // Normalize
            if sum > F::zero() {
                for j in 0..seq_len_k {
                    row[j] /= sum;
                }
            }
        }

        // Calculate output: scores @ value
        let mut output = Array2::<F>::zeros((seq_len_q, d_model_v));

        for i in 0..seq_len_q {
            for j in 0..d_model_v {
                let mut sum = F::zero();
                for k in 0..seq_len_k {
                    sum += scores[[i, k]] * v_b[[k, j]];
                }
                output[[i, j]] = sum;
            }
        }

        // Store the result for this batch
        result
            .slice_mut(scirs2_core::ndarray::s![b, .., ..])
            .assign(&output);
    }

    Ok(result)
}

/// Apply attention mask to scores
#[allow(dead_code)]
pub fn apply_mask<F>(
    scores: &mut Array2<F>,
    mask: &AttentionMask,
    batch_idx: usize,
) -> LinalgResult<()>
where
    F: Float + Add + Mul + Div + Sub + NumAssignOps + Zero + std::fmt::Debug,
{
    let (seq_len_q, seq_len_k) = (scores.shape()[0], scores.shape()[1]);

    match mask {
        AttentionMask::Additive(mask_tensor) => {
            let batch_dim = mask_tensor.shape()[0];
            let mask_idx = if batch_dim == 1 { 0 } else { batch_idx };

            if mask_tensor.shape()[1] != seq_len_q || mask_tensor.shape()[2] != seq_len_k {
                return Err(LinalgError::DimensionError(format!(
                    "Mask shape {:?} doesn't match scores shape [{}, {}]",
                    mask_tensor.shape(),
                    seq_len_q,
                    seq_len_k
                )));
            }

            let mask_slice = mask_tensor.slice(scirs2_core::ndarray::s![mask_idx, .., ..]);

            for i in 0..seq_len_q {
                for j in 0..seq_len_k {
                    // Convert f32 to F (handling float type conversion)
                    let mask_val = F::from(mask_slice[[i, j]]).unwrap_or(F::zero());
                    scores[[i, j]] += mask_val;
                }
            }
        }

        AttentionMask::Multiplicative(mask_tensor) => {
            let batch_dim = mask_tensor.shape()[0];
            let mask_idx = if batch_dim == 1 { 0 } else { batch_idx };

            if mask_tensor.shape()[1] != seq_len_q || mask_tensor.shape()[2] != seq_len_k {
                return Err(LinalgError::DimensionError(format!(
                    "Mask shape {:?} doesn't match scores shape [{}, {}]",
                    mask_tensor.shape(),
                    seq_len_q,
                    seq_len_k
                )));
            }

            let mask_slice = mask_tensor.slice(scirs2_core::ndarray::s![mask_idx, .., ..]);

            for i in 0..seq_len_q {
                for j in 0..seq_len_k {
                    // Convert f32 to F
                    let mask_val = F::from(mask_slice[[i, j]]).unwrap_or(F::zero());
                    scores[[i, j]] *= mask_val;
                }
            }
        }

        AttentionMask::Boolean(mask_tensor) => {
            let batch_dim = mask_tensor.shape()[0];
            let mask_idx = if batch_dim == 1 { 0 } else { batch_idx };

            if mask_tensor.shape()[1] != seq_len_q || mask_tensor.shape()[2] != seq_len_k {
                return Err(LinalgError::DimensionError(format!(
                    "Mask shape {:?} doesn't match scores shape [{}, {}]",
                    mask_tensor.shape(),
                    seq_len_q,
                    seq_len_k
                )));
            }

            let mask_slice = mask_tensor.slice(scirs2_core::ndarray::s![mask_idx, .., ..]);

            for i in 0..seq_len_q {
                for j in 0..seq_len_k {
                    if !mask_slice[[i, j]] {
                        scores[[i, j]] = F::neg_infinity();
                    }
                }
            }
        }

        AttentionMask::Causal => {
            for i in 0..seq_len_q {
                for j in 0..seq_len_k {
                    if j > i {
                        scores[[i, j]] = F::neg_infinity();
                    }
                }
            }
        }
    }

    Ok(())
}
