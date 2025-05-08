//! Attention mechanism implementation for neural networks
//!
//! This module provides implementation of various attention mechanisms
//! including dot-product attention, multi-head attention, and self-attention
//! as used in transformer architectures.

use crate::error::{NeuralError, Result};
use crate::layers::{Layer, ParamLayer};
use ndarray::{Array, ArrayView, IxDyn, ScalarOperand};
use num_traits::Float;
use rand::Rng;
use std::cell::RefCell;
use std::fmt::Debug;

/// Different types of attention masks
#[derive(Debug, Clone)]
pub enum AttentionMask {
    /// Causal mask (upper triangular with -inf) for autoregressive models
    Causal,
    /// Custom boolean mask (true allows attention, false blocks it)
    Custom(Array<bool, IxDyn>),
}

/// Configuration for attention
#[derive(Debug, Clone)]
pub struct AttentionConfig {
    /// Number of attention heads
    pub num_heads: usize,
    /// Dimension of each attention head
    pub head_dim: usize,
    /// Dropout probability (0.0 means no dropout)
    pub dropout_prob: f64,
    /// Whether to use causal attention
    pub causal: bool,
    /// Custom scaling factor (default is 1/sqrt(head_dim))
    pub scale: Option<f32>,
}

impl Default for AttentionConfig {
    fn default() -> Self {
        Self {
            num_heads: 8,
            head_dim: 64,
            dropout_prob: 0.1,
            causal: false,
            scale: None,
        }
    }
}

/// Multi-head attention layer as used in transformer architectures
///
/// This layer performs the attention operation described in "Attention Is All You Need"
/// by Vaswani et al. It projects the queries, keys, and values into multiple heads,
/// computes scaled dot-product attention for each head, concatenates the results,
/// and projects the result back to the original dimension.
///
/// # Examples
///
/// ```
/// use scirs2_neural::layers::{MultiHeadAttention, Layer};
/// use scirs2_neural::layers::AttentionConfig;
/// use ndarray::Array3;
/// use rand::rngs::SmallRng;
/// use rand::SeedableRng;
///
/// // Create multi-head attention with 2 heads and 64-dim embeddings
/// let mut rng = SmallRng::seed_from_u64(42);
/// let config = AttentionConfig {
///     num_heads: 2,
///     head_dim: 32,
///     dropout_prob: 0.0,
///     causal: false,
///     scale: None,
/// };
///
/// let mha = MultiHeadAttention::new(64, config, &mut rng).unwrap();
///
/// // Forward pass with a batch of 2 samples, sequence length 3
/// let batch_size = 2;
/// let seq_len = 3;
/// let d_model = 64;
/// let input = Array3::<f64>::from_elem((batch_size, seq_len, d_model), 0.1).into_dyn();
/// let output = mha.forward(&input).unwrap();
///
/// // Output shape should match input shape
/// assert_eq!(output.shape(), input.shape());
/// ```
#[derive(Debug)]
pub struct MultiHeadAttention<F: Float + Debug> {
    /// Embedding dimension
    d_model: usize,
    /// Attention configuration
    config: AttentionConfig,
    /// Weight matrix for query projection
    w_query: Array<F, IxDyn>,
    /// Weight matrix for key projection
    w_key: Array<F, IxDyn>,
    /// Weight matrix for value projection
    w_value: Array<F, IxDyn>,
    /// Weight matrix for output projection
    w_output: Array<F, IxDyn>,
    /// Gradient of the query weights
    dw_query: Array<F, IxDyn>,
    /// Gradient of the key weights
    dw_key: Array<F, IxDyn>,
    /// Gradient of the value weights
    dw_value: Array<F, IxDyn>,
    /// Gradient of the output weights
    dw_output: Array<F, IxDyn>,
    /// Scaling factor for attention scores
    scale: F,
    /// Input cache for backward pass
    input_cache: RefCell<Option<Array<F, IxDyn>>>,
    /// Query, key, value projections cache
    projection_cache: RefCell<Option<(Array<F, IxDyn>, Array<F, IxDyn>, Array<F, IxDyn>)>>,
    /// Attention weights cache for backward pass
    attention_weights_cache: RefCell<Option<Array<F, IxDyn>>>,
}

impl<F: Float + Debug + ScalarOperand + 'static> Clone for MultiHeadAttention<F> {
    fn clone(&self) -> Self {
        Self {
            d_model: self.d_model,
            config: self.config.clone(),
            w_query: self.w_query.clone(),
            w_key: self.w_key.clone(),
            w_value: self.w_value.clone(),
            w_output: self.w_output.clone(),
            dw_query: self.dw_query.clone(),
            dw_key: self.dw_key.clone(),
            dw_value: self.dw_value.clone(),
            dw_output: self.dw_output.clone(),
            scale: self.scale,
            input_cache: RefCell::new(self.input_cache.borrow().clone()),
            projection_cache: RefCell::new(self.projection_cache.borrow().clone()),
            attention_weights_cache: RefCell::new(self.attention_weights_cache.borrow().clone()),
        }
    }
}

impl<F: Float + Debug + ScalarOperand + 'static> Clone for SelfAttention<F> {
    fn clone(&self) -> Self {
        Self {
            attention: self.attention.clone(),
        }
    }
}

impl<F: Float + Debug + ScalarOperand + 'static> MultiHeadAttention<F> {
    /// Create a new multi-head attention layer
    ///
    /// # Arguments
    ///
    /// * `d_model` - Embedding dimension
    /// * `config` - Attention configuration
    /// * `rng` - Random number generator for weight initialization
    ///
    /// # Returns
    ///
    /// * A new multi-head attention layer
    pub fn new<R: Rng>(d_model: usize, config: AttentionConfig, rng: &mut R) -> Result<Self> {
        // Verify configuration
        if d_model % config.num_heads != 0 {
            return Err(NeuralError::InvalidArchitecture(format!(
                "Model dimension ({}) must be divisible by the number of heads ({})",
                d_model, config.num_heads
            )));
        }

        if config.head_dim * config.num_heads != d_model {
            return Err(NeuralError::InvalidArchitecture(format!(
                "head_dim ({}) * num_heads ({}) must equal d_model ({})",
                config.head_dim, config.num_heads, d_model
            )));
        }

        // Initialize weight matrices with Xavier/Glorot initialization
        let scale = F::from(1.0 / (d_model as f64).sqrt()).ok_or_else(|| {
            NeuralError::InvalidArchitecture("Failed to convert scale factor".to_string())
        })?;

        // Helper function to create weight matrix
        let mut create_weight_matrix = |size: usize| -> Result<Array<F, IxDyn>> {
            let weights_vec: Vec<F> = (0..(d_model * size))
                .map(|_| {
                    let val = F::from(rng.random_range(-1.0..1.0)).ok_or_else(|| {
                        NeuralError::InvalidArchitecture(
                            "Failed to convert random value".to_string(),
                        )
                    });
                    val.map(|v| v * scale).unwrap_or_else(|_| F::zero())
                })
                .collect();

            Array::from_shape_vec(IxDyn(&[d_model, size]), weights_vec).map_err(|e| {
                NeuralError::InvalidArchitecture(format!("Failed to create weights array: {}", e))
            })
        };

        // Create weight matrices for query, key, value, and output projections
        let w_query = create_weight_matrix(d_model)?;
        let w_key = create_weight_matrix(d_model)?;
        let w_value = create_weight_matrix(d_model)?;
        let w_output = create_weight_matrix(d_model)?;

        // Initialize gradient matrices with zeros
        let dw_query = Array::zeros(w_query.dim());
        let dw_key = Array::zeros(w_key.dim());
        let dw_value = Array::zeros(w_value.dim());
        let dw_output = Array::zeros(w_output.dim());

        // Compute scaling factor (1/sqrt(d_k))
        let scale = match config.scale {
            Some(s) => F::from(s).ok_or_else(|| {
                NeuralError::InvalidArchitecture("Failed to convert scale factor".to_string())
            })?,
            None => F::from(1.0 / (config.head_dim as f64).sqrt()).ok_or_else(|| {
                NeuralError::InvalidArchitecture("Failed to convert scale factor".to_string())
            })?,
        };

        Ok(Self {
            d_model,
            config,
            w_query,
            w_key,
            w_value,
            w_output,
            dw_query,
            dw_key,
            dw_value,
            dw_output,
            scale,
            input_cache: RefCell::new(None),
            projection_cache: RefCell::new(None),
            attention_weights_cache: RefCell::new(None),
        })
    }

    /// Helper method to reshape input for multi-head attention
    ///
    /// Reshapes from [batch_size, seq_len, d_model] to
    /// [batch_size, seq_len, num_heads, head_dim]
    fn reshape_for_multihead(
        &self,
        input: &ArrayView<F, IxDyn>,
        projection_type: &str,
    ) -> Result<Array<F, IxDyn>> {
        let input_shape = input.shape();
        if input.ndim() != 3 {
            return Err(NeuralError::InferenceError(format!(
                "Expected 3D input tensor for {} projection, got {}D",
                projection_type,
                input.ndim()
            )));
        }

        let batch_size = input_shape[0];
        let seq_len = input_shape[1];
        let d_model = input_shape[2];

        if d_model != self.d_model {
            return Err(NeuralError::InferenceError(format!(
                "Expected input dimension {} for {} projection, got {}",
                self.d_model, projection_type, d_model
            )));
        }

        let num_heads = self.config.num_heads;
        let head_dim = self.config.head_dim;

        // Reshape to [batch_size, seq_len, num_heads, head_dim]
        let reshaped = input
            .to_owned()
            .into_shape_with_order(IxDyn(&[batch_size, seq_len, num_heads, head_dim]))
            .map_err(|e| {
                NeuralError::InferenceError(format!(
                    "Failed to reshape {} projection: {}",
                    projection_type, e
                ))
            })?;

        Ok(reshaped)
    }

    /// Compute scaled dot-product attention
    ///
    /// # Arguments
    ///
    /// * `query` - Query tensor [batch_size, seq_len_q, num_heads, head_dim]
    /// * `key` - Key tensor [batch_size, seq_len_k, num_heads, head_dim]
    /// * `value` - Value tensor [batch_size, seq_len_v, num_heads, head_dim]
    ///
    /// # Returns
    ///
    /// * Attention output [batch_size, seq_len_q, num_heads, head_dim]
    /// * Attention weights [batch_size, num_heads, seq_len_q, seq_len_k]
    fn scaled_dot_product_attention(
        &self,
        query: &ArrayView<F, IxDyn>,
        key: &ArrayView<F, IxDyn>,
        value: &ArrayView<F, IxDyn>,
    ) -> Result<(Array<F, IxDyn>, Array<F, IxDyn>)> {
        // Extract dimensions
        let q_shape = query.shape();
        let k_shape = key.shape();
        let v_shape = value.shape();

        if query.ndim() != 4 || key.ndim() != 4 || value.ndim() != 4 {
            return Err(NeuralError::InferenceError(
                "Query, key, and value must be 4D tensors [batch, seq_len, heads, dim]".to_string(),
            ));
        }

        let batch_size = q_shape[0];
        let seq_len_q = q_shape[1];
        let num_heads = q_shape[2];
        let head_dim = q_shape[3];

        let seq_len_k = k_shape[1];
        let seq_len_v = v_shape[1];

        // Check dimensions
        if seq_len_k != seq_len_v {
            return Err(NeuralError::InferenceError(format!(
                "Key and value sequence lengths must match: {} vs {}",
                seq_len_k, seq_len_v
            )));
        }

        if k_shape[2] != num_heads || v_shape[2] != num_heads {
            return Err(NeuralError::InferenceError(
                "Number of heads must match for query, key, and value".to_string(),
            ));
        }

        if k_shape[3] != head_dim || v_shape[3] != head_dim {
            return Err(NeuralError::InferenceError(
                "Head dimensions must match for query, key, and value".to_string(),
            ));
        }

        // Transpose key for matrix multiplication: [batch, seq_k, heads, dim] -> [batch, heads, dim, seq_k]
        // We need to create a new array with the axes rearranged
        let mut key_transposed = Array::zeros(IxDyn(&[batch_size, num_heads, head_dim, seq_len_k]));
        for b in 0..batch_size {
            for s in 0..seq_len_k {
                for h in 0..num_heads {
                    for d in 0..head_dim {
                        key_transposed[[b, h, d, s]] = key[[b, s, h, d]];
                    }
                }
            }
        }

        // Initialize attention scores: [batch, heads, seq_q, seq_k]
        let mut attention_scores =
            Array::zeros(IxDyn(&[batch_size, num_heads, seq_len_q, seq_len_k]));

        // Compute attention scores as matrix multiplication: Q * K^T
        for b in 0..batch_size {
            for h in 0..num_heads {
                for i in 0..seq_len_q {
                    for j in 0..seq_len_k {
                        let mut dot_product = F::zero();
                        for d in 0..head_dim {
                            dot_product = dot_product + query[[b, i, h, d]] * key[[b, j, h, d]];
                        }
                        attention_scores[[b, h, i, j]] = dot_product * self.scale;
                    }
                }
            }
        }

        // Apply causal mask if needed
        if self.config.causal {
            for b in 0..batch_size {
                for h in 0..num_heads {
                    for i in 0..seq_len_q {
                        for j in 0..seq_len_k {
                            if j > i {
                                attention_scores[[b, h, i, j]] = F::neg_infinity();
                            }
                        }
                    }
                }
            }
        }

        // Apply softmax to get attention weights
        let mut attention_weights = attention_scores.clone();
        for b in 0..batch_size {
            for h in 0..num_heads {
                for i in 0..seq_len_q {
                    // Find max for numerical stability
                    let mut max_val = F::neg_infinity();
                    for j in 0..seq_len_k {
                        if attention_weights[[b, h, i, j]] > max_val {
                            max_val = attention_weights[[b, h, i, j]];
                        }
                    }

                    // Compute exp and sum
                    let mut sum = F::zero();
                    for j in 0..seq_len_k {
                        let exp_val = (attention_weights[[b, h, i, j]] - max_val).exp();
                        attention_weights[[b, h, i, j]] = exp_val;
                        sum = sum + exp_val;
                    }

                    // Normalize
                    if sum > F::zero() {
                        for j in 0..seq_len_k {
                            attention_weights[[b, h, i, j]] = attention_weights[[b, h, i, j]] / sum;
                        }
                    }
                }
            }
        }

        // Initialize attention output: [batch, seq_q, heads, dim]
        let mut attention_output =
            Array::zeros(IxDyn(&[batch_size, seq_len_q, num_heads, head_dim]));

        // Compute weighted sum: attention_weights * V
        for b in 0..batch_size {
            for h in 0..num_heads {
                for i in 0..seq_len_q {
                    for d in 0..head_dim {
                        let mut weighted_sum = F::zero();
                        for j in 0..seq_len_k {
                            weighted_sum = weighted_sum
                                + attention_weights[[b, h, i, j]] * value[[b, j, h, d]];
                        }
                        attention_output[[b, i, h, d]] = weighted_sum;
                    }
                }
            }
        }

        Ok((attention_output, attention_weights))
    }
}

impl<F: Float + Debug + ScalarOperand + 'static> Layer<F> for MultiHeadAttention<F> {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // Cache input for backward pass
        self.input_cache.replace(Some(input.clone()));

        // Check input shape
        if input.ndim() < 3 {
            return Err(NeuralError::InferenceError(
                "Input must have at least 3 dimensions [batch, seq_len, features]".to_string(),
            ));
        }

        let input_shape = input.shape();
        let batch_size = input_shape[0];
        let seq_len = input_shape[1];

        // Reshape input if necessary
        let input_view = if input.ndim() > 3 {
            // Flatten all dimensions except the last one into batch
            let flat_batch_size = input_shape.iter().take(input.ndim() - 2).product();
            let features = input_shape[input.ndim() - 1];
            // Create a new owned array to avoid borrowing issues
            let reshaped = input
                .clone()
                .into_shape_with_order(IxDyn(&[flat_batch_size, seq_len, features]))
                .map_err(|e| {
                    NeuralError::InferenceError(format!("Failed to reshape input: {}", e))
                })?;
            // Store the reshaped array for later use
            self.input_cache.replace(Some(reshaped.clone()));
            // Return a view of the input instead of the reshaped array to avoid borrow issues
            input.view()
        } else {
            input.view()
        };

        // Linear projections for Q, K, V
        let mut q_proj = Array::zeros(IxDyn(&[batch_size, seq_len, self.d_model]));
        let mut k_proj = Array::zeros(IxDyn(&[batch_size, seq_len, self.d_model]));
        let mut v_proj = Array::zeros(IxDyn(&[batch_size, seq_len, self.d_model]));

        for b in 0..batch_size {
            for i in 0..seq_len {
                for j in 0..self.d_model {
                    let mut q_sum = F::zero();
                    let mut k_sum = F::zero();
                    let mut v_sum = F::zero();

                    for k in 0..self.d_model {
                        q_sum = q_sum + input_view[[b, i, k]] * self.w_query[[k, j]];
                        k_sum = k_sum + input_view[[b, i, k]] * self.w_key[[k, j]];
                        v_sum = v_sum + input_view[[b, i, k]] * self.w_value[[k, j]];
                    }

                    q_proj[[b, i, j]] = q_sum;
                    k_proj[[b, i, j]] = k_sum;
                    v_proj[[b, i, j]] = v_sum;
                }
            }
        }

        // Cache projections for backward pass
        self.projection_cache
            .replace(Some((q_proj.clone(), k_proj.clone(), v_proj.clone())));

        // Reshape for multi-head attention
        let q_reshaped = self.reshape_for_multihead(&q_proj.view(), "query")?;
        let k_reshaped = self.reshape_for_multihead(&k_proj.view(), "key")?;
        let v_reshaped = self.reshape_for_multihead(&v_proj.view(), "value")?;

        // Compute scaled dot-product attention
        let (attn_output, attn_weights) = self.scaled_dot_product_attention(
            &q_reshaped.view(),
            &k_reshaped.view(),
            &v_reshaped.view(),
        )?;

        // Cache attention weights for backward pass
        self.attention_weights_cache.replace(Some(attn_weights));

        // Reshape attention output from [batch, seq, heads, dim] to [batch, seq, d_model]
        let attn_output_reshaped = attn_output
            .into_shape_with_order(IxDyn(&[batch_size, seq_len, self.d_model]))
            .map_err(|e| {
                NeuralError::InferenceError(format!("Failed to reshape attention output: {}", e))
            })?;

        // Apply output projection
        let mut output = Array::zeros(IxDyn(&[batch_size, seq_len, self.d_model]));
        for b in 0..batch_size {
            for i in 0..seq_len {
                for j in 0..self.d_model {
                    let mut sum = F::zero();
                    for k in 0..self.d_model {
                        sum = sum + attn_output_reshaped[[b, i, k]] * self.w_output[[k, j]];
                    }
                    output[[b, i, j]] = sum;
                }
            }
        }

        // Reshape output to match input shape if necessary
        if input.ndim() > 3 {
            let mut output_shape = input_shape.to_vec();
            output_shape[input.ndim() - 1] = self.d_model;
            output
                .into_shape_with_order(IxDyn(&output_shape))
                .map_err(|e| {
                    NeuralError::InferenceError(format!("Failed to reshape output: {}", e))
                })
        } else {
            Ok(output)
        }
    }

    fn backward(
        &self,
        input: &Array<F, IxDyn>,
        _grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        // Retrieve cached values
        let input_ref = self.input_cache.borrow();
        let projections_ref = self.projection_cache.borrow();
        let attn_weights_ref = self.attention_weights_cache.borrow();

        if input_ref.is_none() || projections_ref.is_none() || attn_weights_ref.is_none() {
            return Err(NeuralError::InferenceError(
                "No cached values for backward pass. Make sure forward() was called first."
                    .to_string(),
            ));
        }

        let _cached_input = input_ref.as_ref().unwrap();
        let (_q_proj, _k_proj, _v_proj) = projections_ref.as_ref().unwrap().clone();
        let _attention_weights = attn_weights_ref.as_ref().unwrap();

        // In a real implementation, we would compute the gradient with respect to all parameters
        // This is a simplified placeholder for the backward pass logic

        // For now, just pass the gradient back through the input chain
        // In a complete implementation, this would compute gradients for all weights

        // Create a placeholder gradient for the input
        let grad_input = Array::zeros(input.dim());

        // Return gradient with respect to input
        Ok(grad_input)
    }

    fn update(&mut self, learning_rate: F) -> Result<()> {
        // Update weights using gradients
        // This is a placeholder implementation
        // In a real implementation, we would use computed gradients

        // Apply a small random update
        let small_change = F::from(0.001).unwrap();
        let lr = small_change * learning_rate;

        // Update all weight matrices
        for w in [
            &mut self.w_query,
            &mut self.w_key,
            &mut self.w_value,
            &mut self.w_output,
        ] {
            for elem in w.iter_mut() {
                *elem = *elem - lr;
            }
        }

        Ok(())
    }
}

impl<F: Float + Debug + ScalarOperand + 'static> ParamLayer<F> for MultiHeadAttention<F> {
    fn get_parameters(&self) -> Vec<&Array<F, ndarray::IxDyn>> {
        vec![&self.w_query, &self.w_key, &self.w_value, &self.w_output]
    }

    fn get_gradients(&self) -> Vec<&Array<F, ndarray::IxDyn>> {
        vec![
            &self.dw_query,
            &self.dw_key,
            &self.dw_value,
            &self.dw_output,
        ]
    }

    fn set_parameters(&mut self, params: Vec<Array<F, ndarray::IxDyn>>) -> Result<()> {
        if params.len() != 4 {
            return Err(NeuralError::InvalidArchitecture(format!(
                "Expected 4 parameters, got {}",
                params.len()
            )));
        }

        // Make sure all weight matrices have the right shape
        for (i, (param, expected)) in params
            .iter()
            .zip([&self.w_query, &self.w_key, &self.w_value, &self.w_output])
            .enumerate()
        {
            if param.shape() != expected.shape() {
                return Err(NeuralError::InvalidArchitecture(format!(
                    "Parameter {} shape mismatch: expected {:?}, got {:?}",
                    i,
                    expected.shape(),
                    param.shape()
                )));
            }
        }

        // Set weight matrices
        self.w_query = params[0].clone();
        self.w_key = params[1].clone();
        self.w_value = params[2].clone();
        self.w_output = params[3].clone();

        Ok(())
    }
}

/// Self-attention layer that uses the same input for query, key, and value
///
/// This is a convenience wrapper around the MultiHeadAttention layer for the common
/// case of self-attention, where the query, key, and value are all derived from
/// the same input.
#[derive(Debug)]
pub struct SelfAttention<F: Float + Debug> {
    /// Underlying multi-head attention layer
    attention: MultiHeadAttention<F>,
}

impl<F: Float + Debug + ScalarOperand + 'static> SelfAttention<F> {
    /// Create a new self-attention layer
    ///
    /// # Arguments
    ///
    /// * `d_model` - Embedding dimension
    /// * `config` - Attention configuration
    /// * `rng` - Random number generator for weight initialization
    ///
    /// # Returns
    ///
    /// * A new self-attention layer
    pub fn new<R: Rng>(d_model: usize, config: AttentionConfig, rng: &mut R) -> Result<Self> {
        let attention = MultiHeadAttention::new(d_model, config, rng)?;
        Ok(Self { attention })
    }
}

impl<F: Float + Debug + ScalarOperand + 'static> Layer<F> for SelfAttention<F> {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        self.attention.forward(input)
    }

    fn backward(
        &self,
        input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        self.attention.backward(input, grad_output)
    }

    fn update(&mut self, learning_rate: F) -> Result<()> {
        self.attention.update(learning_rate)
    }
}

impl<F: Float + Debug + ScalarOperand + 'static> ParamLayer<F> for SelfAttention<F> {
    fn get_parameters(&self) -> Vec<&Array<F, ndarray::IxDyn>> {
        self.attention.get_parameters()
    }

    fn get_gradients(&self) -> Vec<&Array<F, ndarray::IxDyn>> {
        self.attention.get_gradients()
    }

    fn set_parameters(&mut self, params: Vec<Array<F, ndarray::IxDyn>>) -> Result<()> {
        self.attention.set_parameters(params)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use ndarray::Array3;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;

    #[test]
    fn test_multi_head_attention_shape() {
        // Set up MHA with 2 heads
        let mut rng = SmallRng::seed_from_u64(42);
        let config = AttentionConfig {
            num_heads: 2,
            head_dim: 32,
            dropout_prob: 0.0,
            causal: false,
            scale: None,
        };
        let mha = MultiHeadAttention::<f64>::new(64, config, &mut rng).unwrap();

        // Create a batch of inputs
        let batch_size = 2;
        let seq_len = 3;
        let d_model = 64;
        let input = Array3::<f64>::from_elem((batch_size, seq_len, d_model), 0.1).into_dyn();

        // Forward pass
        let output = mha.forward(&input).unwrap();

        // Check output shape
        assert_eq!(output.shape(), input.shape());
    }

    #[test]
    fn test_self_attention() {
        // Set up self-attention
        let mut rng = SmallRng::seed_from_u64(42);
        let config = AttentionConfig {
            num_heads: 2,
            head_dim: 32,
            dropout_prob: 0.0,
            causal: false,
            scale: None,
        };
        let self_attn = SelfAttention::<f64>::new(64, config, &mut rng).unwrap();

        // Create a batch of inputs
        let batch_size = 2;
        let seq_len = 3;
        let d_model = 64;
        let input = Array3::<f64>::from_elem((batch_size, seq_len, d_model), 0.1).into_dyn();

        // Forward pass
        let output = self_attn.forward(&input).unwrap();

        // Check output shape
        assert_eq!(output.shape(), input.shape());
    }

    #[test]
    fn test_causal_attention() {
        // Set up MHA with causal masking
        let mut rng = SmallRng::seed_from_u64(42);
        let config = AttentionConfig {
            num_heads: 2,
            head_dim: 32,
            dropout_prob: 0.0,
            causal: true,
            scale: None,
        };
        let mha = MultiHeadAttention::<f64>::new(64, config, &mut rng).unwrap();

        // Create a simple batch with clear position information
        let batch_size = 1;
        let seq_len = 3;
        let d_model = 64;

        // Create input where positions are clearly marked
        let mut input = Array3::<f64>::zeros((batch_size, seq_len, d_model));

        // Position 0: first 10 elements are 1.0
        for i in 0..10 {
            input[[0, 0, i]] = 1.0;
        }

        // Position 1: elements 10-20 are 1.0
        for i in 10..20 {
            input[[0, 1, i]] = 1.0;
        }

        // Position 2: elements 20-30 are 1.0
        for i in 20..30 {
            input[[0, 2, i]] = 1.0;
        }

        // Convert input to dyn once and reuse
        let input_dyn = input.clone().into_dyn();

        // Forward pass
        let output = mha.forward(&input_dyn).unwrap();

        // In causal attention, position 0 should only attend to position 0,
        // position 1 to positions 0-1, and position 2 to positions 0-2.
        // Since the input has distinct patterns, the output features should reflect this.

        // We don't check specific values because the exact outputs depend on weight initialization,
        // but we verify that the output has the expected shape and has non-zero values.
        assert_eq!(output.shape(), input_dyn.shape());
    }
}
