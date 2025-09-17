//! Transformer-based Time Series Forecasting
//!
//! This module provides transformer architectures specifically designed for time series
//! forecasting tasks, incorporating attention mechanisms and positional encoding.

use ndarray::{Array1, Array2};
use num_traits::{Float, FromPrimitive};
use std::fmt::Debug;

use crate::error::Result;

/// Transformer model for time series forecasting with multi-head attention
#[derive(Debug)]
pub struct TimeSeriesTransformer<F: Float + Debug + ndarray::ScalarOperand> {
    /// Number of transformer layers
    num_layers: usize,
    /// Attention heads per layer
    num_heads: usize,
    /// Model dimension
    d_model: usize,
    /// Feed-forward dimension
    d_ff: usize,
    /// Sequence length
    seq_len: usize,
    /// Prediction horizon
    pred_len: usize,
    /// Model parameters
    parameters: Array2<F>,
    /// Positional encoding
    positional_encoding: Array2<F>,
}

impl<F: Float + Debug + Clone + FromPrimitive + ndarray::ScalarOperand> TimeSeriesTransformer<F> {
    /// Create new Transformer for time series forecasting
    pub fn new(
        seq_len: usize,
        pred_len: usize,
        d_model: usize,
        num_heads: usize,
        num_layers: usize,
        d_ff: usize,
    ) -> Self {
        // Calculate total parameter count
        let attention_params_per_layer = 4 * d_model * d_model; // Q, K, V, O projections
        let ff_params_per_layer = 2 * d_model * d_ff + d_ff + d_model; // Two linear _layers + biases
        let layer_norm_params_per_layer = 2 * d_model * 2; // Two layer norms per layer
        let embedding_params = seq_len * d_model; // Input embeddings
        let output_params = d_model * pred_len; // Final projection

        let total_params = num_layers
            * (attention_params_per_layer + ff_params_per_layer + layer_norm_params_per_layer)
            + embedding_params
            + output_params;

        // Initialize parameters
        let scale = F::from(2.0).unwrap() / F::from(d_model).unwrap();
        let std_dev = scale.sqrt();

        let mut parameters = Array2::zeros((1, total_params));
        for i in 0..total_params {
            let val = ((i * 13) % 1000) as f64 / 1000.0 - 0.5;
            parameters[[0, i]] = F::from(val).unwrap() * std_dev;
        }

        // Create positional encoding
        let mut positional_encoding = Array2::zeros((seq_len, d_model));
        for pos in 0..seq_len {
            for i in 0..d_model {
                let angle = F::from(pos).unwrap()
                    / F::from(10000.0)
                        .unwrap()
                        .powf(F::from(2 * (i / 2)).unwrap() / F::from(d_model).unwrap());
                if i % 2 == 0 {
                    positional_encoding[[pos, i]] = angle.sin();
                } else {
                    positional_encoding[[pos, i]] = angle.cos();
                }
            }
        }

        Self {
            num_layers,
            num_heads,
            d_model,
            d_ff,
            seq_len,
            pred_len,
            parameters,
            positional_encoding,
        }
    }

    /// Forward pass through transformer
    pub fn forward(&self, input: &Array2<F>) -> Result<Array2<F>> {
        let batch_size_ = input.nrows();

        // Input embedding + positional encoding
        let mut x = self.input_embedding(input)?;

        // Add positional encoding
        for i in 0..batch_size_ {
            for j in 0..self.seq_len {
                for k in 0..self.d_model {
                    x[[i * self.seq_len + j, k]] =
                        x[[i * self.seq_len + j, k]] + self.positional_encoding[[j, k]];
                }
            }
        }

        // Pass through transformer layers
        for layer in 0..self.num_layers {
            x = self.transformer_layer(&x, layer)?;
        }

        // Final projection to prediction horizon
        self.output_projection(&x, batch_size_)
    }

    /// Input embedding layer
    fn input_embedding(&self, input: &Array2<F>) -> Result<Array2<F>> {
        let batch_size_ = input.nrows();
        let input_dim = input.ncols();

        // Simple linear projection to d_model
        let mut embedded = Array2::zeros((batch_size_ * self.seq_len, self.d_model));

        // Extract embedding weights (simplified)
        let param_start = 0;

        for i in 0..batch_size_ {
            for j in 0..self.seq_len.min(input_dim) {
                for k in 0..self.d_model {
                    let weight_idx = (j * self.d_model + k) % (self.seq_len * self.d_model);
                    let weight = if param_start + weight_idx < self.parameters.ncols() {
                        self.parameters[[0, param_start + weight_idx]]
                    } else {
                        F::zero()
                    };
                    embedded[[i * self.seq_len + j, k]] = input[[i, j]] * weight;
                }
            }
        }

        Ok(embedded)
    }

    /// Single transformer layer
    fn transformer_layer(&self, input: &Array2<F>, layeridx: usize) -> Result<Array2<F>> {
        // Multi-head attention
        let attention_output = self.multi_head_attention(input, layeridx)?;

        // Add & Norm 1
        let norm1_output =
            self.layer_norm(&self.add_residual(input, &attention_output)?, layeridx, 0)?;

        // Feed-forward
        let ff_output = self.feed_forward(&norm1_output, layeridx)?;

        // Add & Norm 2
        let final_output =
            self.layer_norm(&self.add_residual(&norm1_output, &ff_output)?, layeridx, 1)?;

        Ok(final_output)
    }

    /// Multi-head attention mechanism
    fn multi_head_attention(&self, input: &Array2<F>, layeridx: usize) -> Result<Array2<F>> {
        let seq_len = input.nrows();
        let head_dim = self.d_model / self.num_heads;

        // Simplified attention computation
        let mut output = Array2::zeros((seq_len, self.d_model));

        for head in 0..self.num_heads {
            // Compute Q, K, V for this head (simplified)
            let q = self.compute_qkv_projection(input, layeridx, head, 0)?; // Query
            let k = self.compute_qkv_projection(input, layeridx, head, 1)?; // Key
            let v = self.compute_qkv_projection(input, layeridx, head, 2)?; // Value

            // Attention scores
            let attention_scores = self.compute_attention_scores(&q, &k)?;

            // Apply attention to values
            let head_output = self.apply_attention(&attention_scores, &v)?;

            // Combine heads
            for i in 0..seq_len {
                for j in 0..head_dim {
                    if head * head_dim + j < self.d_model {
                        output[[i, head * head_dim + j]] = head_output[[i, j]];
                    }
                }
            }
        }

        Ok(output)
    }

    /// Compute Q, K, V projections
    fn compute_qkv_projection(
        &self,
        input: &Array2<F>,
        layer_idx: usize,
        head: usize,
        projection_type: usize,
    ) -> Result<Array2<F>> {
        let seq_len = input.nrows();
        let head_dim = self.d_model / self.num_heads;
        let mut output = Array2::zeros((seq_len, head_dim));

        // Simplified projection computation
        for i in 0..seq_len {
            for j in 0..head_dim {
                let mut sum = F::zero();
                for k in 0..self.d_model {
                    // Compute weight index (simplified)
                    let weight_idx = (layer_idx * 1000
                        + head * 100
                        + projection_type * 10
                        + j * self.d_model
                        + k)
                        % self.parameters.ncols();
                    let weight = self.parameters[[0, weight_idx]];
                    sum = sum + input[[i, k]] * weight;
                }
                output[[i, j]] = sum;
            }
        }

        Ok(output)
    }

    /// Compute attention scores
    fn compute_attention_scores(&self, q: &Array2<F>, k: &Array2<F>) -> Result<Array2<F>> {
        let seq_len = q.nrows();
        let head_dim = q.ncols();
        let scale = F::one() / F::from(head_dim).unwrap().sqrt();

        let mut scores = Array2::zeros((seq_len, seq_len));

        for i in 0..seq_len {
            for j in 0..seq_len {
                let mut dot_product = F::zero();
                for dim in 0..head_dim {
                    dot_product = dot_product + q[[i, dim]] * k[[j, dim]];
                }
                scores[[i, j]] = dot_product * scale;
            }
        }

        // Apply softmax
        self.softmax_2d(&scores)
    }

    /// Apply attention weights to values
    fn apply_attention(&self, attention: &Array2<F>, values: &Array2<F>) -> Result<Array2<F>> {
        let seq_len = attention.nrows();
        let head_dim = values.ncols();
        let mut output = Array2::zeros((seq_len, head_dim));

        for i in 0..seq_len {
            for j in 0..head_dim {
                let mut sum = F::zero();
                for k in 0..seq_len {
                    sum = sum + attention[[i, k]] * values[[k, j]];
                }
                output[[i, j]] = sum;
            }
        }

        Ok(output)
    }

    /// Feed-forward network
    fn feed_forward(&self, input: &Array2<F>, layeridx: usize) -> Result<Array2<F>> {
        let seq_len = input.nrows();

        // First linear layer
        let mut hidden = Array2::zeros((seq_len, self.d_ff));
        for i in 0..seq_len {
            for j in 0..self.d_ff {
                let mut sum = F::zero();
                for k in 0..self.d_model {
                    let weight_idx =
                        (layeridx * 2000 + j * self.d_model + k) % self.parameters.ncols();
                    let weight = self.parameters[[0, weight_idx]];
                    sum = sum + input[[i, k]] * weight;
                }
                hidden[[i, j]] = self.relu(sum);
            }
        }

        // Second linear layer
        let mut output = Array2::zeros((seq_len, self.d_model));
        for i in 0..seq_len {
            for j in 0..self.d_model {
                let mut sum = F::zero();
                for k in 0..self.d_ff {
                    let weight_idx =
                        (layeridx * 3000 + j * self.d_ff + k) % self.parameters.ncols();
                    let weight = self.parameters[[0, weight_idx]];
                    sum = sum + hidden[[i, k]] * weight;
                }
                output[[i, j]] = sum;
            }
        }

        Ok(output)
    }

    /// Layer normalization
    fn layer_norm(
        &self,
        input: &Array2<F>,
        layer_idx: usize,
        norm_idx: usize,
    ) -> Result<Array2<F>> {
        let seq_len = input.nrows();
        let mut output = Array2::zeros(input.dim());

        for i in 0..seq_len {
            // Compute mean and variance
            let mut sum = F::zero();
            for j in 0..self.d_model {
                sum = sum + input[[i, j]];
            }
            let mean = sum / F::from(self.d_model).unwrap();

            let mut var_sum = F::zero();
            for j in 0..self.d_model {
                let diff = input[[i, j]] - mean;
                var_sum = var_sum + diff * diff;
            }
            let variance = var_sum / F::from(self.d_model).unwrap();
            let std_dev = (variance + F::from(1e-5).unwrap()).sqrt();

            // Normalize
            for j in 0..self.d_model {
                let normalized = (input[[i, j]] - mean) / std_dev;

                // Apply learnable parameters (gamma and beta)
                let gamma_idx = (layer_idx * 100 + norm_idx * 50 + j) % self.parameters.ncols();
                let beta_idx = (layer_idx * 100 + norm_idx * 50 + j + 25) % self.parameters.ncols();

                let gamma = self.parameters[[0, gamma_idx]];
                let beta = self.parameters[[0, beta_idx]];

                output[[i, j]] = gamma * normalized + beta;
            }
        }

        Ok(output)
    }

    /// Add residual connection
    fn add_residual(&self, input1: &Array2<F>, input2: &Array2<F>) -> Result<Array2<F>> {
        let mut output = Array2::zeros(input1.dim());

        for i in 0..input1.nrows() {
            for j in 0..input1.ncols() {
                output[[i, j]] = input1[[i, j]] + input2[[i, j]];
            }
        }

        Ok(output)
    }

    /// Output projection to prediction horizon
    fn output_projection(&self, input: &Array2<F>, batchsize: usize) -> Result<Array2<F>> {
        let mut output = Array2::zeros((batchsize, self.pred_len));

        // Use last token representation for prediction
        for i in 0..batchsize {
            let last_token_idx = i * self.seq_len + self.seq_len - 1;

            for j in 0..self.pred_len {
                let mut sum = F::zero();
                for k in 0..self.d_model {
                    let weight_idx = (j * self.d_model + k) % self.parameters.ncols();
                    let weight = self.parameters[[0, weight_idx]];
                    sum = sum + input[[last_token_idx, k]] * weight;
                }
                output[[i, j]] = sum;
            }
        }

        Ok(output)
    }

    /// 2D Softmax function
    fn softmax_2d(&self, input: &Array2<F>) -> Result<Array2<F>> {
        let mut output = Array2::zeros(input.dim());

        for i in 0..input.nrows() {
            // Find max for numerical stability
            let mut max_val = input[[i, 0]];
            for j in 1..input.ncols() {
                if input[[i, j]] > max_val {
                    max_val = input[[i, j]];
                }
            }

            // Compute exponentials and sum
            let mut sum = F::zero();
            for j in 0..input.ncols() {
                let exp_val = (input[[i, j]] - max_val).exp();
                output[[i, j]] = exp_val;
                sum = sum + exp_val;
            }

            // Normalize
            for j in 0..input.ncols() {
                output[[i, j]] = output[[i, j]] / sum;
            }
        }

        Ok(output)
    }

    /// ReLU activation
    fn relu(&self, x: F) -> F {
        x.max(F::zero())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_transformer_creation() {
        let transformer = TimeSeriesTransformer::<f64>::new(10, 5, 64, 8, 4, 256);
        assert_eq!(transformer.seq_len, 10);
        assert_eq!(transformer.pred_len, 5);
        assert_eq!(transformer.d_model, 64);
        assert_eq!(transformer.num_heads, 8);
        assert_eq!(transformer.num_layers, 4);
        assert_eq!(transformer.d_ff, 256);
    }

    #[test]
    fn test_positional_encoding() {
        let transformer = TimeSeriesTransformer::<f64>::new(8, 4, 16, 4, 2, 64);
        let pe = &transformer.positional_encoding;

        assert_eq!(pe.dim(), (8, 16));

        // Check that positional encoding values are bounded
        for &val in pe.iter() {
            assert!(val >= -1.0 && val <= 1.0);
        }
    }

    #[test]
    fn test_transformer_forward() {
        let transformer = TimeSeriesTransformer::<f64>::new(6, 3, 32, 4, 2, 128);
        let input =
            Array2::from_shape_vec((2, 6), (0..12).map(|i| i as f64 * 0.1).collect()).unwrap();

        let output = transformer.forward(&input).unwrap();
        assert_eq!(output.dim(), (2, 3)); // batch_size x pred_len

        // Check that output is finite
        for &val in output.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_softmax_properties() {
        let transformer = TimeSeriesTransformer::<f64>::new(4, 2, 8, 2, 1, 32);
        let input =
            Array2::from_shape_vec((3, 3), vec![1.0, 2.0, 3.0, 0.5, 1.5, 2.5, 2.0, 1.0, 0.5])
                .unwrap();

        let output = transformer.softmax_2d(&input).unwrap();

        // Check that each row sums to approximately 1.0
        for i in 0..output.nrows() {
            let row_sum: f64 = (0..output.ncols()).map(|j| output[[i, j]]).sum();
            assert_abs_diff_eq!(row_sum, 1.0, epsilon = 1e-10);
        }

        // Check that all values are non-negative
        for &val in output.iter() {
            assert!(val >= 0.0);
        }
    }

    #[test]
    fn test_input_embedding() {
        let transformer = TimeSeriesTransformer::<f64>::new(5, 3, 16, 4, 2, 64);
        let input =
            Array2::from_shape_vec((2, 5), (0..10).map(|i| i as f64 * 0.2).collect()).unwrap();

        let embedded = transformer.input_embedding(&input).unwrap();
        assert_eq!(embedded.dim(), (10, 16)); // (batch_size * seq_len, d_model)

        // Check that embedding is finite
        for &val in embedded.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_multi_head_attention() {
        let transformer = TimeSeriesTransformer::<f64>::new(4, 2, 16, 4, 1, 64);
        let input = Array2::zeros((4, 16)); // seq_len x d_model

        let output = transformer.multi_head_attention(&input, 0).unwrap();
        assert_eq!(output.dim(), (4, 16));

        // Check that output is finite
        for &val in output.iter() {
            assert!(val.is_finite());
        }
    }
}
