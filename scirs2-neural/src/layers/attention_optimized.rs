//! Optimized attention mechanism implementation for neural networks
//!
//! This module provides high-performance implementation of attention mechanisms
//! using efficient matrix operations instead of nested loops.

use crate::error::{NeuralError, Result};
use crate::layers::{Layer, ParamLayer};
use scirs2_core::ndarray::{Array, ArrayView, IxDyn, ScalarOperand, s, Array2, Axis};
use scirs2_core::numeric::Float;
use scirs2_core::random::Rng;
use std::sync::RwLock;
use std::fmt::Debug;
// SIMD optimizations using scirs2-core
use scirs2_core::simd_ops::{PlatformCapabilities, SimdUnifiedOps};
/// Optimized forward pass implementation for MultiHeadAttention
/// This replaces the inefficient nested loops with proper matrix operations
#[allow(dead_code)]
pub fn optimized_attention_forward<F>(
    input: &Array<F, IxDyn>,
    w_query: &Array<F, IxDyn>,
    w_key: &Array<F, IxDyn>, 
    w_value: &Array<F, IxDyn>,
    w_output: &Array<F, IxDyn>,
    d_model: usize,
    config: &crate::layers::attention::AttentionConfig,
    scale: F,
) -> Result<Array<F, IxDyn>>
where
    F: Float + Debug + ScalarOperand + Send + Sync + 'static + SimdUnifiedOps + scirs2_core::ndarray::LinalgScalar,
{
    // Check input shape
    if input.ndim() < 3 {
        return Err(NeuralError::InferenceError(
            "Input must have at least 3 dimensions [batch, seq_len, features]".to_string(),
        ));
    }
    let inputshape = input.shape();
    let batch_size = inputshape[0];
    let seq_len = inputshape[1];
    // Reshape input if necessary to 3D [batch, seq_len, features]
    let input_3d = if input.ndim() > 3 {
        let flat_batch_size = inputshape.iter().take(input.ndim() - 2).product();
        let features = inputshape[input.ndim() - 1];
        input
            .clone()
            .into_shape_with_order(IxDyn(&[flat_batch_size, seq_len, features]))
            .map_err(|e| {
                NeuralError::InferenceError(format!("Failed to reshape input: {}", e))
            })?
    } else {
        input.clone()
    };
    // Convert to 2D for efficient batched matrix multiplication
    let input_2d = input_3d
        .clone()
        .into_shape((batch_size * seq_len, d_model))
        .map_err(|e| {
            NeuralError::InferenceError(format!("Failed to reshape input for matmul: {}", e))
        })?;
    // Convert weights to 2D arrays
    let w_query_2d = w_query.view().into_dimensionality::<scirs2_core::ndarray::Ix2>()
        .map_err(|_| NeuralError::InferenceError("Failed to convert query weights to 2D".to_string()))?;
    let w_key_2d = w_key.view().into_dimensionality::<scirs2_core::ndarray::Ix2>()
        .map_err(|_| NeuralError::InferenceError("Failed to convert key weights to 2D".to_string()))?;
    let w_value_2d = w_value.view().into_dimensionality::<scirs2_core::ndarray::Ix2>()
        .map_err(|_| NeuralError::InferenceError("Failed to convert value weights to 2D".to_string()))?;
    let w_output_2d = w_output.view().into_dimensionality::<scirs2_core::ndarray::Ix2>()
        .map_err(|_| NeuralError::InferenceError("Failed to convert output weights to 2D".to_string()))?;
    let input_2d_view = input_2d.view().into_dimensionality::<scirs2_core::ndarray::Ix2>()
        .map_err(|_| NeuralError::InferenceError("Failed to convert input to 2D view".to_string()))?;
    // Efficient Q, K, V projections using matrix multiplication
    let q_proj_2d = input_2d_view.dot(&w_query_2d);
    let k_proj_2d = input_2d_view.dot(&w_key_2d);  
    let v_proj_2d = input_2d_view.dot(&w_value_2d);
    // Reshape back to [batch, seq_len, d_model]
    let q_proj = q_proj_2d.into_shape((batch_size, seq_len, d_model))
        .map_err(|e| NeuralError::InferenceError(format!("Failed to reshape Q projection: {}", e)))?;
    let k_proj = k_proj_2d.into_shape((batch_size, seq_len, d_model))
        .map_err(|e| NeuralError::InferenceError(format!("Failed to reshape K projection: {}", e)))?;
    let v_proj = v_proj_2d.into_shape((batch_size, seq_len, d_model))
        .map_err(|e| NeuralError::InferenceError(format!("Failed to reshape V projection: {}", e)))?;
    // Reshape for multi-head attention: [batch, seq, d_model] -> [batch, seq, heads, head_dim]
    let num_heads = config.num_heads;
    let head_dim = config.head_dim;
    
    let q_multihead = q_proj.into_shape((batch_size, seq_len, num_heads, head_dim))
        .map_err(|e| NeuralError::InferenceError(format!("Failed to reshape Q for multihead: {}", e)))?;
    let k_multihead = k_proj.into_shape((batch_size, seq_len, num_heads, head_dim))
        .map_err(|e| NeuralError::InferenceError(format!("Failed to reshape K for multihead: {}", e)))?;
    let v_multihead = v_proj.into_shape((batch_size, seq_len, num_heads, head_dim))
        .map_err(|e| NeuralError::InferenceError(format!("Failed to reshape V for multihead: {}", e)))?;
    // Transpose for efficient attention computation: [batch, seq, heads, head_dim] -> [batch, heads, seq, head_dim]
    let q_transposed = q_multihead.permuted_axes([0, 2, 1, 3]);
    let k_transposed = k_multihead.permuted_axes([0, 2, 1, 3]);
    let v_transposed = v_multihead.permuted_axes([0, 2, 1, 3]);
    // Compute attention scores using batched matrix multiplication
    // scores = Q @ K^T with shape [batch, heads, seq_q, seq_k]
    let mut attention_scores = Array::zeros((batch_size, num_heads, seq_len, seq_len));
    for b in 0..batch_size {
        for h in 0..num_heads {
            let q_slice = q_transposed.slice(s![b, h, .., ..]).into_dimensionality::<scirs2_core::ndarray::Ix2>()
                .map_err(|_| NeuralError::InferenceError("Failed to get Q slice".to_string()))?;
            let k_slice = k_transposed.slice(s![b, h, .., ..]).into_dimensionality::<scirs2_core::ndarray::Ix2>()
                .map_err(|_| NeuralError::InferenceError("Failed to get K slice".to_string()))?;
            
            // Q @ K^T
            let scores = q_slice.dot(&k_slice.t()) * scale;
            for i in 0..seq_len {
                for j in 0..seq_len {
                    attention_scores[[b, h, i, j]] = scores[[i, j]];
                }
            }
        }
    // Apply causal mask if needed
    if config.causal {
        for b in 0..batch_size {
            for h in 0..num_heads {
                for i in 0..seq_len {
                    for j in (i+1)..seq_len {
                        attention_scores[[b, h, i, j]] = F::neg_infinity();
                    }
    // Apply softmax to get attention weights
    let mut attention_weights = attention_scores.clone();
                // Find max for numerical stability
                let mut max_val = F::neg_infinity();
                    if attention_weights[[b, h, i, j]] > max_val {
                        max_val = attention_weights[[b, h, i, j]];
                // Compute exp and sum
                let mut sum = F::zero();
                    let exp_val = (attention_weights[[b, h, i, j]] - max_val).exp();
                    attention_weights[[b, h, i, j]] = exp_val;
                    sum = sum + exp_val;
                // Normalize
                if sum > F::zero() {
                    for j in 0..seq_len {
                        attention_weights[[b, h, i, j]] = attention_weights[[b, h, i, j]] / sum;
    // Apply attention weights to values: attention_weights @ V
    let mut attention_output = Array::zeros((batch_size, num_heads, seq_len, head_dim));
            let attn_slice = attentionweights.slice(s![b, h, .., ..]).into_dimensionality::<scirs2_core::ndarray::Ix2>()
                .map_err(|_| NeuralError::InferenceError("Failed to get attention slice".to_string()))?;
            let v_slice = v_transposed.slice(s![b, h, .., ..]).into_dimensionality::<scirs2_core::ndarray::Ix2>()
                .map_err(|_| NeuralError::InferenceError("Failed to get V slice".to_string()))?;
            // attention_weights @ V
            let output = attn_slice.dot(&v_slice);
                for d in 0..head_dim {
                    attention_output[[b, h, i, d]] = output[[i, d]];
    // Transpose back and reshape: [batch, heads, seq, head_dim] -> [batch, seq, heads, head_dim] -> [batch, seq, d_model]
    let attention_output_transposed = attention_output.permuted_axes([0, 2, 1, 3]);
    let attention_output_reshaped = attention_output_transposed.into_shape((batch_size, seq_len, d_model))
        .map_err(|e| NeuralError::InferenceError(format!("Failed to reshape attention output: {}", e)))?;
    // Apply output projection using efficient matrix multiplication
    let attention_output_2d = attention_output_reshaped.view()
        .map_err(|e| NeuralError::InferenceError(format!("Failed to reshape for output projection: {}", e)))?;
    let attention_output_2d_view = attention_output_2d.into_dimensionality::<scirs2_core::ndarray::Ix2>()
        .map_err(|_| NeuralError::InferenceError("Failed to convert attention output to 2D view".to_string()))?;
    let output_2d = attention_output_2d_view.dot(&w_output_2d);
    let output = output_2d.into_shape((batch_size, seq_len, d_model))
        .map_err(|e| NeuralError::InferenceError(format!("Failed to reshape output: {}", e)))?
        .into_dyn();
    // Reshape output to match input shape if necessary
    if input.ndim() > 3 {
        let mut originalshape = inputshape.to_vec();
        originalshape[originalshape.len() - 1] = d_model;
        output
            .into_shape_with_order(IxDyn(&originalshape))
                NeuralError::InferenceError(format!("Failed to reshape output: {}", e))
            })
        Ok(output)
}
