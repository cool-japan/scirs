//! Multi-head attention mechanisms for transformer architecture

use ndarray::{Array1, Array2, Array3, Axis};
use num_traits::Float;
use std::f64::consts::PI;
use crate::error::Result;

/// Multi-head attention mechanism
pub struct MultiHeadAttention<T: Float> {
    /// Number of attention heads
    num_heads: usize,

    /// Dimension per head
    head_dimension: usize,

    /// Model dimension
    model_dimension: usize,

    /// Query projection weights
    query_weights: Array2<T>,

    /// Key projection weights
    key_weights: Array2<T>,

    /// Value projection weights
    value_weights: Array2<T>,

    /// Output projection weights
    output_weights: Array2<T>,

    /// Attention dropout rate
    dropout_rate: f64,

    /// Cached attention weights for analysis
    attention_weights: Option<Array3<T>>,

    /// Scale factor for attention scores
    scale_factor: T,
}

impl<T: Float + 'static> MultiHeadAttention<T> {
    /// Create new multi-head attention
    pub fn new(
        num_heads: usize,
        model_dimension: usize,
        head_dimension: usize,
    ) -> Result<Self> {
        if model_dimension % num_heads != 0 {
            return Err(crate::error::OptimError::Other(
                "Model dimension must be divisible by number of heads".to_string()
            ));
        }

        let scale_factor = T::from(1.0 / (head_dimension as f64).sqrt()).unwrap();

        // Initialize weights with Xavier/Glorot initialization
        let xavier_std = (2.0 / (model_dimension + head_dimension) as f64).sqrt();

        let query_weights = Self::initialize_weights(model_dimension, model_dimension, xavier_std);
        let key_weights = Self::initialize_weights(model_dimension, model_dimension, xavier_std);
        let value_weights = Self::initialize_weights(model_dimension, model_dimension, xavier_std);
        let output_weights = Self::initialize_weights(model_dimension, model_dimension, xavier_std);

        Ok(Self {
            num_heads,
            head_dimension,
            model_dimension,
            query_weights,
            key_weights,
            value_weights,
            output_weights,
            dropout_rate: 0.1,
            attention_weights: None,
            scale_factor,
        })
    }

    /// Initialize weight matrix with Xavier initialization
    fn initialize_weights(rows: usize, cols: usize, std: f64) -> Array2<T> {
        let mut weights = Array2::<T>::zeros((rows, cols));

        for i in 0..rows {
            for j in 0..cols {
                let random_val = (rand::random::<f64>() - 0.5) * 2.0 * std;
                weights[[i, j]] = T::from(random_val).unwrap();
            }
        }

        weights
    }

    /// Forward pass through multi-head attention
    pub fn forward(
        &mut self,
        query: &Array2<T>,
        key: &Array2<T>,
        value: &Array2<T>,
    ) -> Result<Array2<T>> {
        let batch_size = query.shape()[0];
        let seq_length = query.shape()[1];

        // Linear projections
        let q = query.dot(&self.query_weights);
        let k = key.dot(&self.key_weights);
        let v = value.dot(&self.value_weights);

        // Reshape for multi-head attention
        let q_heads = self.reshape_for_heads(&q, batch_size, seq_length)?;
        let k_heads = self.reshape_for_heads(&k, batch_size, seq_length)?;
        let v_heads = self.reshape_for_heads(&v, batch_size, seq_length)?;

        // Compute attention
        let attention_output = self.compute_attention(&q_heads, &k_heads, &v_heads)?;

        // Reshape back and apply output projection
        let reshaped_output = self.reshape_from_heads(&attention_output, batch_size, seq_length)?;
        let output = reshaped_output.dot(&self.output_weights);

        Ok(output)
    }

    /// Reshape tensor for multi-head attention
    fn reshape_for_heads(
        &self,
        tensor: &Array2<T>,
        batch_size: usize,
        seq_length: usize,
    ) -> Result<Array3<T>> {
        // Reshape from (batch_size * seq_length, model_dim) to (batch_size, seq_length, num_heads, head_dim)
        // Then transpose to (batch_size, num_heads, seq_length, head_dim)

        let mut reshaped = Array3::<T>::zeros((batch_size, self.num_heads, seq_length * self.head_dimension));

        for batch in 0..batch_size {
            for head in 0..self.num_heads {
                for seq in 0..seq_length {
                    for dim in 0..self.head_dimension {
                        let input_idx = batch * seq_length + seq;
                        let input_dim = head * self.head_dimension + dim;
                        let output_idx = seq * self.head_dimension + dim;

                        if input_idx < tensor.shape()[0] && input_dim < tensor.shape()[1] {
                            reshaped[[batch, head, output_idx]] = tensor[[input_idx, input_dim]];
                        }
                    }
                }
            }
        }

        Ok(reshaped)
    }

    /// Reshape tensor back from multi-head format
    fn reshape_from_heads(
        &self,
        tensor: &Array3<T>,
        batch_size: usize,
        seq_length: usize,
    ) -> Result<Array2<T>> {
        let mut output = Array2::<T>::zeros((batch_size * seq_length, self.model_dimension));

        for batch in 0..batch_size {
            for head in 0..self.num_heads {
                for seq in 0..seq_length {
                    for dim in 0..self.head_dimension {
                        let input_idx = seq * self.head_dimension + dim;
                        let output_row = batch * seq_length + seq;
                        let output_col = head * self.head_dimension + dim;

                        if input_idx < tensor.shape()[2] && output_row < output.shape()[0] && output_col < output.shape()[1] {
                            output[[output_row, output_col]] = tensor[[batch, head, input_idx]];
                        }
                    }
                }
            }
        }

        Ok(output)
    }

    /// Compute scaled dot-product attention
    fn compute_attention(
        &mut self,
        query: &Array3<T>,
        key: &Array3<T>,
        value: &Array3<T>,
    ) -> Result<Array3<T>> {
        let batch_size = query.shape()[0];
        let num_heads = query.shape()[1];
        let seq_length = query.shape()[2] / self.head_dimension;

        let mut attention_output = Array3::<T>::zeros(query.raw_dim());
        let mut attention_weights = Array3::<T>::zeros((batch_size, num_heads, seq_length * seq_length));

        for batch in 0..batch_size {
            for head in 0..num_heads {
                // Extract Q, K, V for this batch and head
                let q_slice = self.extract_head_slice(query, batch, head, seq_length)?;
                let k_slice = self.extract_head_slice(key, batch, head, seq_length)?;
                let v_slice = self.extract_head_slice(value, batch, head, seq_length)?;

                // Compute attention scores: Q * K^T / sqrt(d_k)
                let scores = self.compute_attention_scores(&q_slice, &k_slice)?;

                // Apply softmax
                let attention_probs = self.softmax(&scores)?;

                // Store attention weights for analysis
                for i in 0..seq_length {
                    for j in 0..seq_length {
                        if i * seq_length + j < attention_weights.shape()[2] {
                            attention_weights[[batch, head, i * seq_length + j]] = attention_probs[[i, j]];
                        }
                    }
                }

                // Apply attention to values: Attention * V
                let attended_values = attention_probs.dot(&v_slice);

                // Store result
                for i in 0..seq_length {
                    for j in 0..self.head_dimension {
                        let output_idx = i * self.head_dimension + j;
                        if output_idx < attention_output.shape()[2] {
                            attention_output[[batch, head, output_idx]] = attended_values[[i, j]];
                        }
                    }
                }
            }
        }

        // Cache attention weights
        self.attention_weights = Some(attention_weights);

        Ok(attention_output)
    }

    /// Extract slice for specific batch and head
    fn extract_head_slice(
        &self,
        tensor: &Array3<T>,
        batch: usize,
        head: usize,
        seq_length: usize,
    ) -> Result<Array2<T>> {
        let mut slice = Array2::<T>::zeros((seq_length, self.head_dimension));

        for seq in 0..seq_length {
            for dim in 0..self.head_dimension {
                let tensor_idx = seq * self.head_dimension + dim;
                if tensor_idx < tensor.shape()[2] {
                    slice[[seq, dim]] = tensor[[batch, head, tensor_idx]];
                }
            }
        }

        Ok(slice)
    }

    /// Compute attention scores
    fn compute_attention_scores(
        &self,
        query: &Array2<T>,
        key: &Array2<T>,
    ) -> Result<Array2<T>> {
        let seq_length = query.shape()[0];
        let mut scores = Array2::<T>::zeros((seq_length, seq_length));

        for i in 0..seq_length {
            for j in 0..seq_length {
                let mut score = T::zero();
                for k in 0..self.head_dimension {
                    score = score + query[[i, k]] * key[[j, k]];
                }
                scores[[i, j]] = score * self.scale_factor;
            }
        }

        Ok(scores)
    }

    /// Apply softmax to attention scores
    fn softmax(&self, scores: &Array2<T>) -> Result<Array2<T>> {
        let seq_length = scores.shape()[0];
        let mut probs = Array2::<T>::zeros((seq_length, seq_length));

        for i in 0..seq_length {
            // Find max for numerical stability
            let mut max_score = T::neg_infinity();
            for j in 0..seq_length {
                if scores[[i, j]] > max_score {
                    max_score = scores[[i, j]];
                }
            }

            // Compute exponentials and sum
            let mut exp_sum = T::zero();
            let mut exp_scores = vec![T::zero(); seq_length];

            for j in 0..seq_length {
                let exp_val = (scores[[i, j]] - max_score).to_f64().unwrap().exp();
                exp_scores[j] = T::from(exp_val).unwrap();
                exp_sum = exp_sum + exp_scores[j];
            }

            // Normalize
            for j in 0..seq_length {
                probs[[i, j]] = exp_scores[j] / exp_sum;
            }
        }

        Ok(probs)
    }

    /// Get cached attention weights
    pub fn get_attention_weights(&self) -> Option<Array3<T>> {
        self.attention_weights.clone()
    }

    /// Reset parameters
    pub fn reset(&mut self) -> Result<()> {
        let xavier_std = (2.0 / (self.model_dimension + self.head_dimension) as f64).sqrt();

        self.query_weights = Self::initialize_weights(self.model_dimension, self.model_dimension, xavier_std);
        self.key_weights = Self::initialize_weights(self.model_dimension, self.model_dimension, xavier_std);
        self.value_weights = Self::initialize_weights(self.model_dimension, self.model_dimension, xavier_std);
        self.output_weights = Self::initialize_weights(self.model_dimension, self.model_dimension, xavier_std);

        self.attention_weights = None;

        Ok(())
    }

    /// Get parameter count
    pub fn parameter_count(&self) -> usize {
        self.query_weights.len() +
        self.key_weights.len() +
        self.value_weights.len() +
        self.output_weights.len()
    }

    /// Set dropout rate
    pub fn set_dropout_rate(&mut self, rate: f64) {
        self.dropout_rate = rate.clamp(0.0, 1.0);
    }
}

/// Attention mechanism trait for different attention types
pub trait AttentionMechanism<T: Float> {
    fn compute_attention(
        &mut self,
        query: &Array2<T>,
        key: &Array2<T>,
        value: &Array2<T>,
    ) -> Result<Array2<T>>;

    fn get_attention_weights(&self) -> Option<Array3<T>>;
    fn parameter_count(&self) -> usize;
    fn reset(&mut self) -> Result<()>;
}

impl<T: Float + 'static> AttentionMechanism<T> for MultiHeadAttention<T> {
    fn compute_attention(
        &mut self,
        query: &Array2<T>,
        key: &Array2<T>,
        value: &Array2<T>,
    ) -> Result<Array2<T>> {
        self.forward(query, key, value)
    }

    fn get_attention_weights(&self) -> Option<Array3<T>> {
        self.attention_weights.clone()
    }

    fn parameter_count(&self) -> usize {
        self.parameter_count()
    }

    fn reset(&mut self) -> Result<()> {
        self.reset()
    }
}

/// Self-attention specific implementation
pub struct SelfAttention<T: Float> {
    multi_head_attention: MultiHeadAttention<T>,
}

impl<T: Float + 'static> SelfAttention<T> {
    pub fn new(
        num_heads: usize,
        model_dimension: usize,
        head_dimension: usize,
    ) -> Result<Self> {
        let multi_head_attention = MultiHeadAttention::new(num_heads, model_dimension, head_dimension)?;

        Ok(Self {
            multi_head_attention,
        })
    }

    pub fn forward(&mut self, input: &Array2<T>) -> Result<Array2<T>> {
        // Self-attention: Q = K = V = input
        self.multi_head_attention.forward(input, input, input)
    }
}

/// Cross-attention implementation
pub struct CrossAttention<T: Float> {
    multi_head_attention: MultiHeadAttention<T>,
}

impl<T: Float + 'static> CrossAttention<T> {
    pub fn new(
        num_heads: usize,
        model_dimension: usize,
        head_dimension: usize,
    ) -> Result<Self> {
        let multi_head_attention = MultiHeadAttention::new(num_heads, model_dimension, head_dimension)?;

        Ok(Self {
            multi_head_attention,
        })
    }

    pub fn forward(
        &mut self,
        query: &Array2<T>,
        context: &Array2<T>,
    ) -> Result<Array2<T>> {
        // Cross-attention: Q = query, K = V = context
        self.multi_head_attention.forward(query, context, context)
    }
}

/// Attention visualization utilities
pub struct AttentionVisualizer<T: Float> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float + 'static> AttentionVisualizer<T> {
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }

    /// Extract attention patterns for analysis
    pub fn extract_attention_patterns(
        &self,
        attention_weights: &Array3<T>,
    ) -> AttentionPatterns<T> {
        let batch_size = attention_weights.shape()[0];
        let num_heads = attention_weights.shape()[1];
        let seq_length_squared = attention_weights.shape()[2];
        let seq_length = (seq_length_squared as f64).sqrt() as usize;

        let mut head_entropies = vec![T::zero(); num_heads];
        let mut attention_diversity = T::zero();

        for head in 0..num_heads {
            let mut entropy = T::zero();
            for i in 0..seq_length {
                for j in 0..seq_length {
                    let idx = i * seq_length + j;
                    if idx < seq_length_squared {
                        let prob = attention_weights[[0, head, idx]]; // Use first batch
                        if prob > T::zero() {
                            let log_prob = prob.to_f64().unwrap().ln();
                            entropy = entropy - prob * T::from(log_prob).unwrap();
                        }
                    }
                }
            }
            head_entropies[head] = entropy;
            attention_diversity = attention_diversity + entropy;
        }

        AttentionPatterns {
            head_entropies,
            attention_diversity: attention_diversity / T::from(num_heads).unwrap(),
            sequence_length: seq_length,
            num_heads,
        }
    }
}

/// Attention pattern analysis results
#[derive(Debug, Clone)]
pub struct AttentionPatterns<T: Float> {
    pub head_entropies: Vec<T>,
    pub attention_diversity: T,
    pub sequence_length: usize,
    pub num_heads: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_head_attention_creation() {
        let attention = MultiHeadAttention::<f32>::new(8, 512, 64);
        assert!(attention.is_ok());

        let mha = attention.unwrap();
        assert_eq!(mha.num_heads, 8);
        assert_eq!(mha.model_dimension, 512);
        assert_eq!(mha.head_dimension, 64);
    }

    #[test]
    fn test_attention_forward_pass() {
        let mut attention = MultiHeadAttention::<f32>::new(4, 128, 32).unwrap();

        let seq_length = 10;
        let batch_size = 2;
        let input = Array2::<f32>::zeros((batch_size * seq_length, 128));

        let result = attention.forward(&input, &input, &input);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.shape(), &[batch_size * seq_length, 128]);
    }

    #[test]
    fn test_self_attention() {
        let mut self_attention = SelfAttention::<f32>::new(4, 128, 32).unwrap();

        let input = Array2::<f32>::zeros((20, 128)); // batch_size * seq_length = 20
        let result = self_attention.forward(&input);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.shape(), input.shape());
    }

    #[test]
    fn test_parameter_count() {
        let attention = MultiHeadAttention::<f32>::new(8, 512, 64).unwrap();
        let param_count = attention.parameter_count();

        // 4 weight matrices of size 512x512
        let expected = 4 * 512 * 512;
        assert_eq!(param_count, expected);
    }

    #[test]
    fn test_attention_visualization() {
        let visualizer = AttentionVisualizer::<f32>::new();
        let attention_weights = Array3::<f32>::zeros((1, 4, 100)); // 1 batch, 4 heads, 10x10 sequence

        let patterns = visualizer.extract_attention_patterns(&attention_weights);
        assert_eq!(patterns.num_heads, 4);
        assert_eq!(patterns.sequence_length, 10);
        assert_eq!(patterns.head_entropies.len(), 4);
    }
}