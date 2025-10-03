//! Batched attention mechanisms for transformer models
//!
//! This module provides optimized implementations of attention mechanisms
//! for processing batches of sequences, which is especially useful for
//! transformer-based models in machine learning.

use scirs2_core::ndarray::{Array1, Array2, Array3, ArrayView2, ArrayView3};
use scirs2_core::numeric::{Float, NumAssignOps, Zero};
use std::ops::{Add, Div, Mul, Sub};

use crate::attention::{AttentionConfig, AttentionMask};
use crate::error::{check_dimensions, LinalgError, LinalgResult};

/// Multi-query batched attention
///
/// Computes attention for a batch of sequences where each sequence has its own
/// query matrix but shares the same key and value matrices. This is useful for
/// decoder-only architectures where each token needs to attend to all previous tokens.
///
/// # Arguments
///
/// * `batch_query` - Batch of query matrices [batchsize, seq_len_q, d_model]
/// * `key` - Key matrix [seq_len_k, d_model]
/// * `value` - Value matrix [seq_len_k, d_model]
/// * `mask` - Optional attention mask
/// * `scale` - Scaling factor for attention scores (typically 1/sqrt(d_model))
///
/// # Returns
///
/// * Batch of attention outputs [batchsize, seq_len_q, d_model]
#[allow(dead_code)]
pub fn batch_multi_query_attention<F>(
    batch_query: &ArrayView3<F>,
    key: &ArrayView2<F>,
    value: &ArrayView2<F>,
    mask: Option<&AttentionMask>,
    scale: F,
) -> LinalgResult<Array3<F>>
where
    F: Float + Add + Mul + Div + Sub + NumAssignOps + Zero + std::fmt::Debug + 'static,
{
    // Check dimensions
    let (batchsize, seq_len_q, d_model_q) = batch_query.dim();
    let (seq_len_k, d_model_k) = key.dim();
    let (seq_len_v, d_model_v) = value.dim();

    check_dimensions(
        d_model_q == d_model_k,
        format!("Query and key dimensions must match: {d_model_q} vs {d_model_k}"),
    )?;

    check_dimensions(
        seq_len_k == seq_len_v,
        format!("Key and value sequence lengths must match: {seq_len_k} vs {seq_len_v}"),
    )?;

    // Initialize output
    let mut result = Array3::<F>::zeros((batchsize, seq_len_q, d_model_v));

    // Process each batch independently
    for b in 0..batchsize {
        // Extract _query for this batch
        let query_b = batch_query.slice(scirs2_core::ndarray::s![b, .., ..]);

        // Calculate attention scores: Q * K^T
        let mut scores = Array2::<F>::zeros((seq_len_q, seq_len_k));
        for i in 0..seq_len_q {
            for j in 0..seq_len_k {
                let mut dot_product = F::zero();
                for k in 0..d_model_q {
                    dot_product += query_b[[i, k]] * key[[j, k]];
                }
                scores[[i, j]] = dot_product * scale;
            }
        }

        // Apply mask if provided
        if let Some(mask_ref) = mask {
            match mask_ref {
                AttentionMask::Causal => {
                    // Apply causal mask (upper triangular with -inf)
                    for i in 0..seq_len_q {
                        for j in 0..seq_len_k {
                            if j > i {
                                scores[[i, j]] = F::neg_infinity();
                            }
                        }
                    }
                }
                // Other mask types not implemented for batched version yet
                _ => {
                    return Err(LinalgError::NotImplementedError(
                        "Only causal masks are currently supported for batch_multi_query_attention"
                            .to_string(),
                    ))
                }
            }
        }

        // Apply softmax to each row
        for i in 0..seq_len_q {
            let mut row = scores.slice_mut(scirs2_core::ndarray::s![i, ..]);

            // Find max for numerical stability
            let max_val = row.fold(F::neg_infinity(), |max, &x| if x > max { x } else { max });

            // Compute exp and sum
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

        // Calculate weighted sums: scores * V
        for i in 0..seq_len_q {
            for j in 0..d_model_v {
                let mut sum = F::zero();
                for k in 0..seq_len_k {
                    sum += scores[[i, k]] * value[[k, j]];
                }
                result[[b, i, j]] = sum;
            }
        }
    }

    Ok(result)
}

/// Batch multi-head attention
///
/// Computes multi-head attention for a batch of sequences, with optimized
/// performance for parallel processing across the batch dimension.
///
/// # Arguments
///
/// * `batch_query` - Batch of query matrices [batchsize, seq_len_q, d_model]
/// * `batch_key` - Batch of key matrices [batchsize, seq_len_k, d_model]
/// * `batch_value` - Batch of value matrices [batchsize, seq_len_k, d_model]
/// * `wq` - Query projection weights [d_model, d_model]
/// * `wk` - Key projection weights [d_model, d_model]
/// * `wv` - Value projection weights [d_model, d_model]
/// * `wo` - Output projection weights [d_model, d_model]
/// * `mask` - Optional attention mask (causal or padding)
/// * `config` - Attention configuration
///
/// # Returns
///
/// * Batch of attention outputs [batchsize, seq_len_q, d_model]
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub fn batch_multi_head_attention<F>(
    batch_query: &ArrayView3<F>,
    batch_key: &ArrayView3<F>,
    batch_value: &ArrayView3<F>,
    wq: &ArrayView2<F>,
    wk: &ArrayView2<F>,
    wv: &ArrayView2<F>,
    wo: &ArrayView2<F>,
    mask: Option<&AttentionMask>,
    config: &AttentionConfig,
) -> LinalgResult<Array3<F>>
where
    F: Float + Add + Mul + Div + Sub + NumAssignOps + Zero + std::fmt::Debug + 'static,
{
    // Extract dimensions
    let (batchsize, seq_len_q, d_model) = batch_query.dim();
    let (batchsize_k, seq_len_k, d_model_k) = batch_key.dim();
    let (batchsize_v, seq_len_v, d_model_v) = batch_value.dim();

    // Validate dimensions
    check_dimensions(
        batchsize == batchsize_k && batchsize == batchsize_v,
        format!("Batch sizes must match: {batchsize}, {batchsize_k}, {batchsize_v}"),
    )?;

    check_dimensions(
        d_model == d_model_k && d_model == d_model_v,
        format!("Model dimensions must match: {d_model}, {d_model_k}, {d_model_v}"),
    )?;

    check_dimensions(
        seq_len_k == seq_len_v,
        format!("Key and _value sequence lengths must match: {seq_len_k} vs {seq_len_v}"),
    )?;

    // Check weight matrix dimensions
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

    // Convert scale from f32 to F
    let scale = match config.scale {
        Some(s) => F::from(s).unwrap_or_else(|| {
            // If conversion fails, compute a default scale
            F::from(1.0 / (head_dim as f64).sqrt())
                .unwrap_or_else(|| F::one() / F::from(head_dim).unwrap_or(F::one()).sqrt())
        }),
        None => {
            // No scale provided, compute default
            F::from(1.0 / (head_dim as f64).sqrt())
                .unwrap_or_else(|| F::one() / F::from(head_dim).unwrap_or(F::one()).sqrt())
        }
    };

    // Verify that d_model is compatible with num_heads and head_dim
    if d_model != num_heads * head_dim {
        return Err(LinalgError::ValueError(format!(
            "Model dimension ({d_model}) must equal num_heads ({num_heads}) * head_dim ({head_dim})"
        )));
    }

    // Project queries, keys, and values for all batches
    let mut q_proj = Array3::<F>::zeros((batchsize, seq_len_q, d_model));
    let mut k_proj = Array3::<F>::zeros((batchsize, seq_len_k, d_model));
    let mut v_proj = Array3::<F>::zeros((batchsize, seq_len_v, d_model));

    // Apply projections for all batches
    for b in 0..batchsize {
        // Project _query
        for i in 0..seq_len_q {
            for j in 0..d_model {
                let mut sum = F::zero();
                for k in 0..d_model {
                    sum += batch_query[[b, i, k]] * wq[[k, j]];
                }
                q_proj[[b, i, j]] = sum;
            }
        }

        // Project _key
        for i in 0..seq_len_k {
            for j in 0..d_model {
                let mut sum = F::zero();
                for k in 0..d_model {
                    sum += batch_key[[b, i, k]] * wk[[k, j]];
                }
                k_proj[[b, i, j]] = sum;
            }
        }

        // Project _value
        for i in 0..seq_len_v {
            for j in 0..d_model {
                let mut sum = F::zero();
                for k in 0..d_model {
                    sum += batch_value[[b, i, k]] * wv[[k, j]];
                }
                v_proj[[b, i, j]] = sum;
            }
        }
    }

    // Initialize output tensor
    let mut result = Array3::<F>::zeros((batchsize, seq_len_q, d_model));

    // Process each batch and head
    for b in 0..batchsize {
        // Initialize concatenated outputs for this batch
        let mut concat_outputs = Array2::<F>::zeros((seq_len_q, d_model));

        // Process each attention head
        for h in 0..num_heads {
            let start_idx = h * head_dim;
            let end_idx = start_idx + head_dim;

            // Extract head-specific portions for this batch
            let q_head = q_proj.slice(scirs2_core::ndarray::s![b, .., start_idx..end_idx]);
            let k_head = k_proj.slice(scirs2_core::ndarray::s![b, .., start_idx..end_idx]);
            let v_head = v_proj.slice(scirs2_core::ndarray::s![b, .., start_idx..end_idx]);

            // Calculate attention scores: Q * K^T
            let mut scores = Array2::<F>::zeros((seq_len_q, seq_len_k));
            for i in 0..seq_len_q {
                for j in 0..seq_len_k {
                    let mut dot_product = F::zero();
                    for k in 0..head_dim {
                        dot_product += q_head[[i, k]] * k_head[[j, k]];
                    }
                    scores[[i, j]] = dot_product * scale;
                }
            }

            // Apply mask if provided
            if let Some(mask_ref) = mask {
                match mask_ref {
                    AttentionMask::Causal => {
                        if config.causal {
                            // Apply causal mask (upper triangular with -inf)
                            for i in 0..seq_len_q {
                                for j in 0..seq_len_k {
                                    if j > i {
                                        scores[[i, j]] = F::neg_infinity();
                                    }
                                }
                            }
                        }
                    }
                    // Other mask types not implemented for batched version yet
                    _ => return Err(LinalgError::NotImplementedError(
                        "Only causal masks are currently supported for batch_multi_head_attention"
                            .to_string(),
                    )),
                }
            } else if config.causal {
                // Apply causal mask if specified in config
                for i in 0..seq_len_q {
                    for j in 0..seq_len_k {
                        if j > i {
                            scores[[i, j]] = F::neg_infinity();
                        }
                    }
                }
            }

            // Apply softmax to each row
            for i in 0..seq_len_q {
                let mut row = scores.slice_mut(scirs2_core::ndarray::s![i, ..]);

                // Find max for numerical stability
                let max_val = row.fold(F::neg_infinity(), |max, &x| if x > max { x } else { max });

                // Compute exp and sum
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

            // Calculate head output: scores * V
            let mut head_output = Array2::<F>::zeros((seq_len_q, head_dim));
            for i in 0..seq_len_q {
                for j in 0..head_dim {
                    let mut sum = F::zero();
                    for k in 0..seq_len_k {
                        sum += scores[[i, k]] * v_head[[k, j]];
                    }
                    head_output[[i, j]] = sum;
                }
            }

            // Store head output in the concatenated output
            for i in 0..seq_len_q {
                for j in 0..head_dim {
                    concat_outputs[[i, start_idx + j]] = head_output[[i, j]];
                }
            }
        }

        // Apply output projection for this batch
        for i in 0..seq_len_q {
            for j in 0..d_model {
                let mut sum = F::zero();
                for k in 0..d_model {
                    sum += concat_outputs[[i, k]] * wo[[k, j]];
                }
                result[[b, i, j]] = sum;
            }
        }
    }

    Ok(result)
}

/// Flash Attention for batched sequences
///
/// Memory-efficient implementation of attention for a batch of sequences.
/// This implementation avoids materializing the full attention matrix.
///
/// # Arguments
///
/// * `batch_query` - Batch of query matrices [batchsize, seq_len_q, d_model]
/// * `batch_key` - Batch of key matrices [batchsize, seq_len_k, d_model]
/// * `batch_value` - Batch of value matrices [batchsize, seq_len_k, d_model]
/// * `mask` - Optional attention mask
/// * `scale` - Scaling factor for attention scores
/// * `blocksize` - Block size for tiling (affects performance but not results)
///
/// # Returns
///
/// * Batch of attention outputs [batchsize, seq_len_q, d_model]
#[allow(dead_code)]
pub fn batch_flash_attention<F>(
    batch_query: &ArrayView3<F>,
    batch_key: &ArrayView3<F>,
    batch_value: &ArrayView3<F>,
    mask: Option<&AttentionMask>,
    scale: F,
    blocksize: usize,
) -> LinalgResult<Array3<F>>
where
    F: Float + Add + Mul + Div + Sub + NumAssignOps + Zero + std::fmt::Debug + 'static,
{
    // Check dimensions
    let (batchsize, seq_len_q, d_model) = batch_query.dim();
    let (batchsize_k, seq_len_k, d_model_k) = batch_key.dim();
    let (batchsize_v, seq_len_v, d_model_v) = batch_value.dim();

    check_dimensions(
        batchsize == batchsize_k && batchsize == batchsize_v,
        format!("Batch sizes must match: {batchsize}, {batchsize_k}, {batchsize_v}"),
    )?;

    check_dimensions(
        d_model == d_model_k,
        format!("Query and _key dimensions must match: {d_model} vs {d_model_k}"),
    )?;

    check_dimensions(
        seq_len_k == seq_len_v,
        format!("Key and _value sequence lengths must match: {seq_len_k} vs {seq_len_v}"),
    )?;

    // Determine block sizes
    let blocksize_q = blocksize.min(seq_len_q);
    let blocksize_k = blocksize.min(seq_len_k);

    // Initialize output
    let mut result = Array3::<F>::zeros((batchsize, seq_len_q, d_model_v));

    // Process batch by batch
    for b in 0..batchsize {
        let query_b = batch_query.slice(scirs2_core::ndarray::s![b, .., ..]);
        let key_b = batch_key.slice(scirs2_core::ndarray::s![b, .., ..]);
        let value_b = batch_value.slice(scirs2_core::ndarray::s![b, .., ..]);

        // Process _query blocks
        for q_start in (0..seq_len_q).step_by(blocksize_q) {
            let q_end = (q_start + blocksize_q).min(seq_len_q);
            let q_block = query_b.slice(scirs2_core::ndarray::s![q_start..q_end, ..]);

            // For each _query block, process all _key/_value blocks
            let mut m_block = Array1::<F>::from_elem(q_end - q_start, F::neg_infinity());
            let mut l_block = Array1::<F>::zeros(q_end - q_start);

            for k_start in (0..seq_len_k).step_by(blocksize_k) {
                let k_end = (k_start + blocksize_k).min(seq_len_k);
                let k_block = key_b.slice(scirs2_core::ndarray::s![k_start..k_end, ..]);
                let v_block = value_b.slice(scirs2_core::ndarray::s![k_start..k_end, ..]);

                // Compute block of attention scores
                let mut scores_block = Array2::<F>::zeros((q_end - q_start, k_end - k_start));

                for i in 0..(q_end - q_start) {
                    for j in 0..(k_end - k_start) {
                        let mut dot_product = F::zero();
                        for k in 0..d_model {
                            dot_product += q_block[[i, k]] * k_block[[j, k]];
                        }
                        scores_block[[i, j]] = dot_product * scale;
                    }
                }

                // Apply mask if provided
                if let Some(mask_ref) = mask {
                    match mask_ref {
                        AttentionMask::Causal => {
                            for i in 0..(q_end - q_start) {
                                let q_idx = q_start + i;
                                for j in 0..(k_end - k_start) {
                                    let k_idx = k_start + j;
                                    if k_idx > q_idx {
                                        scores_block[[i, j]] = F::neg_infinity();
                                    }
                                }
                            }
                        }_ => return Err(LinalgError::NotImplementedError(
                            "Flash attention currently only supports causal masks for batched operations".to_string()
                        )),
                    }
                }

                // Update max values and compute exponentials
                for i in 0..(q_end - q_start) {
                    let row = scores_block.slice(scirs2_core::ndarray::s![i, ..]);
                    let max_val =
                        row.fold(F::neg_infinity(), |max, &x| if x > max { x } else { max });

                    if max_val > m_block[i] {
                        // Update scaling factors
                        let m_prev = m_block[i];
                        let m_new = max_val;

                        // Update l_i <- l_i * exp(m_i - m'_i)
                        if m_prev != F::neg_infinity() {
                            l_block[i] *= (m_prev - m_new).exp();
                        }

                        // Update output with scaling
                        if l_block[i] > F::zero() {
                            let scale_factor = (m_prev - m_new).exp() / l_block[i];
                            for j in 0..d_model_v {
                                result[[b, q_start + i, j]] *= scale_factor;
                            }
                        }

                        // Update m_i
                        m_block[i] = m_new;
                    }

                    // Compute contribution of current _key-_value block
                    let mut block_sum = F::zero();
                    let mut block_output = Array1::<F>::zeros(d_model_v);

                    for j in 0..(k_end - k_start) {
                        let exp_val = (scores_block[[i, j]] - m_block[i]).exp();
                        block_sum += exp_val;

                        // Update output
                        for k in 0..d_model_v {
                            block_output[k] += exp_val * v_block[[j, k]];
                        }
                    }

                    // Update l_i and output
                    l_block[i] += block_sum;
                    for j in 0..d_model_v {
                        result[[b, q_start + i, j]] += block_output[j];
                    }
                }
            }

            // Normalize output by l_block
            for i in 0..(q_end - q_start) {
                if l_block[i] > F::zero() {
                    for j in 0..d_model_v {
                        result[[b, q_start + i, j]] /= l_block[i];
                    }
                }
            }
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::Array;

    #[test]
    fn test_batch_multi_query_attention() {
        // Create a batch of 2 query matrices, each 2x2
        let batch_query = Array3::from_shape_vec(
            (2, 2, 2),
            vec![
                1.0, 1.0, // First batch, first query vector
                1.0, 1.0, // First batch, second query vector
                1.0, 1.0, // Second batch, first query vector
                1.0, 1.0, // Second batch, second query vector
            ],
        )
        .unwrap();

        // Shared key matrix 2x2
        let key = Array::from_shape_vec((2, 2), vec![1.0, 1.0, 1.0, 1.0]).unwrap();

        // Shared value matrix 2x2
        let value = Array::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        // Scale factor (1/sqrt(d_k))
        let scale = 1.0 / (2.0f64).sqrt();

        // Compute batch attention
        let result = batch_multi_query_attention(
            &batch_query.view(),
            &key.view(),
            &value.view(),
            None,
            scale,
        )
        .unwrap();

        // Check output shape
        assert_eq!(result.shape(), &[2, 2, 2]);

        // Since the inputs are uniform, each query should get an equal weighted average of the values
        // Expected: Each position should have [2.0, 3.0] (average of [1.0, 2.0] and [3.0, 4.0])
        assert_relative_eq!(result[[0, 0, 0]], 2.0, epsilon = 1e-5);
        assert_relative_eq!(result[[0, 0, 1]], 3.0, epsilon = 1e-5);
        assert_relative_eq!(result[[0, 1, 0]], 2.0, epsilon = 1e-5);
        assert_relative_eq!(result[[0, 1, 1]], 3.0, epsilon = 1e-5);
        assert_relative_eq!(result[[1, 0, 0]], 2.0, epsilon = 1e-5);
        assert_relative_eq!(result[[1, 0, 1]], 3.0, epsilon = 1e-5);
        assert_relative_eq!(result[[1, 1, 0]], 2.0, epsilon = 1e-5);
        assert_relative_eq!(result[[1, 1, 1]], 3.0, epsilon = 1e-5);
    }

    #[test]
    fn test_batch_multi_head_attention() {
        // Create a batch of 2 sequences, each with 2 tokens and embedding dim 4
        let batchsize = 2;
        let seq_len = 2;
        let d_model = 4;
        let num_heads = 2;
        let head_dim = d_model / num_heads;

        // Create query, key, value tensors with simple values for testing
        let batch_query = Array3::from_shape_fn((batchsize, seq_len, d_model), |_| 0.1f64);
        let batch_key = Array3::from_shape_fn((batchsize, seq_len, d_model), |_| 0.1f64);
        let batch_value = Array3::from_shape_fn((batchsize, seq_len, d_model), |_| 0.1f64);

        // Weight matrices
        let wq = Array2::from_shape_fn((d_model, d_model), |_| 0.1f64);
        let wk = Array2::from_shape_fn((d_model, d_model), |_| 0.1f64);
        let wv = Array2::from_shape_fn((d_model, d_model), |_| 0.1f64);
        let wo = Array2::from_shape_fn((d_model, d_model), |_| 0.1f64);

        // Attention config
        let config = AttentionConfig {
            num_heads,
            head_dim,
            dropout_prob: 0.0,
            causal: false,
            scale: Some(1.0 / (head_dim as f32).sqrt()),
        };

        // Compute batched multi-head attention
        let result = batch_multi_head_attention(
            &batch_query.view(),
            &batch_key.view(),
            &batch_value.view(),
            &wq.view(),
            &wk.view(),
            &wv.view(),
            &wo.view(),
            None,
            &config,
        )
        .unwrap();

        // Check output shape
        assert_eq!(result.shape(), &[batchsize, seq_len, d_model]);

        // For uniform inputs, should get uniform outputs
        // The exact value depends on the complex matrix multiplications, but we can check they're all the same
        let first_value = result[[0, 0, 0]];
        for b in 0..batchsize {
            for i in 0..seq_len {
                for j in 0..d_model {
                    assert_relative_eq!(result[[b, i, j]], first_value, epsilon = 1e-5);
                }
            }
        }
    }

    #[test]
    fn test_batch_flash_attention() {
        // Create a batch of 2 sequences, each with 3 tokens and embedding dim 4
        let batchsize = 2;
        let seq_len = 3;
        let d_model = 4;

        // Create query, key, value tensors with simple values for testing
        let batch_query = Array3::from_shape_fn((batchsize, seq_len, d_model), |_| 0.1f64);
        let batch_key = Array3::from_shape_fn((batchsize, seq_len, d_model), |_| 0.1f64);
        let batch_value = Array3::from_shape_fn((batchsize, seq_len, d_model), |_| 0.1f64);

        // Compute batched flash attention with block size 2
        let result = batch_flash_attention(
            &batch_query.view(),
            &batch_key.view(),
            &batch_value.view(),
            None,
            1.0 / (d_model as f64).sqrt(),
            2,
        )
        .unwrap();

        // Check output shape
        assert_eq!(result.shape(), &[batchsize, seq_len, d_model]);

        // For uniform inputs, should get uniform outputs
        let first_value = result[[0, 0, 0]];
        for b in 0..batchsize {
            for i in 0..seq_len {
                for j in 0..d_model {
                    assert_relative_eq!(result[[b, i, j]], first_value, epsilon = 1e-5);
                }
            }
        }
    }

    #[test]
    fn test_batch_multi_query_attention_causal() {
        // Create a batch of query matrices
        let batch_query = Array3::from_shape_vec(
            (2, 3, 2), // 2 batches, 3 tokens, 2 dimensions
            vec![
                1.0, 1.0, // Batch 0, Token 0
                1.0, 1.0, // Batch 0, Token 1
                1.0, 1.0, // Batch 0, Token 2
                1.0, 1.0, // Batch 1, Token 0
                1.0, 1.0, // Batch 1, Token 1
                1.0, 1.0, // Batch 1, Token 2
            ],
        )
        .unwrap();

        // Shared key matrix
        let key = Array::from_shape_vec(
            (3, 2),
            vec![
                1.0, 1.0, // Token 0
                1.0, 1.0, // Token 1
                1.0, 1.0, // Token 2
            ],
        )
        .unwrap();

        // Shared value matrix with different values for each position
        let value = Array::from_shape_vec(
            (3, 2),
            vec![
                1.0, 2.0, // Token 0 values
                3.0, 4.0, // Token 1 values
                5.0, 6.0, // Token 2 values
            ],
        )
        .unwrap();

        // Scale factor
        let scale = 1.0 / (2.0f64).sqrt();

        // Causal mask
        let mask = AttentionMask::Causal;

        // Compute batch attention with causal mask
        let result = batch_multi_query_attention(
            &batch_query.view(),
            &key.view(),
            &value.view(),
            Some(&mask),
            scale,
        )
        .unwrap();

        // Check output shape
        assert_eq!(result.shape(), &[2, 3, 2]);

        // With causal masking:
        // - Token 0 can only attend to Token 0: [1.0, 2.0]
        // - Token 1 can attend to Tokens 0 and 1: average of [1.0, 2.0] and [3.0, 4.0] = [2.0, 3.0]
        // - Token 2 can attend to all tokens: average of [1.0, 2.0], [3.0, 4.0], and [5.0, 6.0] = [3.0, 4.0]

        // Check first batch
        assert_relative_eq!(result[[0, 0, 0]], 1.0, epsilon = 1e-5);
        assert_relative_eq!(result[[0, 0, 1]], 2.0, epsilon = 1e-5);

        assert_relative_eq!(result[[0, 1, 0]], 2.0, epsilon = 1e-5);
        assert_relative_eq!(result[[0, 1, 1]], 3.0, epsilon = 1e-5);

        assert_relative_eq!(result[[0, 2, 0]], 3.0, epsilon = 1e-5);
        assert_relative_eq!(result[[0, 2, 1]], 4.0, epsilon = 1e-5);

        // Second batch should match first batch for this simple example
        assert_relative_eq!(result[[1, 0, 0]], 1.0, epsilon = 1e-5);
        assert_relative_eq!(result[[1, 0, 1]], 2.0, epsilon = 1e-5);

        assert_relative_eq!(result[[1, 1, 0]], 2.0, epsilon = 1e-5);
        assert_relative_eq!(result[[1, 1, 1]], 3.0, epsilon = 1e-5);

        assert_relative_eq!(result[[1, 2, 0]], 3.0, epsilon = 1e-5);
        assert_relative_eq!(result[[1, 2, 1]], 4.0, epsilon = 1e-5);
    }
}
