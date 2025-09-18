//! Core transformer layer implementations

use ndarray::{Array1, Array2, Array3, Axis};
use num_traits::Float;
use crate::error::Result;
use super::config::ActivationFunction;

/// Embedding layer for input vectors
pub struct EmbeddingLayer<T: Float> {
    /// Embedding matrix
    embedding_matrix: Array2<T>,

    /// Input dimension
    input_dimension: usize,

    /// Output dimension (model dimension)
    output_dimension: usize,
}

impl<T: Float> EmbeddingLayer<T> {
    /// Create new embedding layer
    pub fn new(input_dimension: usize, output_dimension: usize) -> Result<Self> {
        let embedding_matrix = Array2::zeros((input_dimension, output_dimension));

        Ok(Self {
            embedding_matrix,
            input_dimension,
            output_dimension,
        })
    }

    /// Forward pass through embedding layer
    pub fn forward(&self, input: &Array2<T>) -> Result<Array2<T>> {
        let batch_size = input.shape()[0];
        let seq_len = input.shape()[1];

        let mut output = Array2::zeros((batch_size, self.output_dimension));

        for i in 0..batch_size {
            for j in 0..seq_len {
                let embedding_idx = input[[i, j]].to_usize().unwrap_or(0) % self.input_dimension;
                let embedding = self.embedding_matrix.row(embedding_idx);

                for k in 0..self.output_dimension {
                    output[[i, k]] = output[[i, k]] + embedding[k];
                }
            }
        }

        Ok(output)
    }

    /// Get parameter count
    pub fn parameter_count(&self) -> usize {
        self.input_dimension * self.output_dimension
    }

    /// Reset embedding parameters
    pub fn reset(&mut self) -> Result<()> {
        self.embedding_matrix.fill(T::zero());
        Ok(())
    }
}

/// Layer normalization implementation
pub struct LayerNormalization<T: Float> {
    /// Layer dimension
    dimension: usize,

    /// Learnable scale parameters
    gamma: Array1<T>,

    /// Learnable shift parameters
    beta: Array1<T>,

    /// Small constant for numerical stability
    epsilon: T,
}

impl<T: Float> LayerNormalization<T> {
    /// Create new layer normalization
    pub fn new(dimension: usize) -> Result<Self> {
        let gamma = Array1::ones(dimension);
        let beta = Array1::zeros(dimension);
        let epsilon = num_traits::cast::cast(1e-5).unwrap_or_else(|| T::zero());

        Ok(Self {
            dimension,
            gamma,
            beta,
            epsilon,
        })
    }

    /// Forward pass through layer normalization
    pub fn forward(&self, input: &Array2<T>) -> Result<Array2<T>> {
        let mut output = input.clone();
        let batch_size = input.shape()[0];

        for i in 0..batch_size {
            let row = input.row(i);

            // Calculate mean
            let mean = row.sum() / T::from(row.len()).unwrap();

            // Calculate variance
            let variance = row.iter()
                .map(|&x| {
                    let diff = x - mean;
                    diff * diff
                })
                .fold(T::zero(), |acc, x| acc + x) / T::from(row.len()).unwrap();

            // Normalize
            let std_dev = (variance + self.epsilon).sqrt();

            for j in 0..self.dimension {
                let normalized = (input[[i, j]] - mean) / std_dev;
                output[[i, j]] = self.gamma[j] * normalized + self.beta[j];
            }
        }

        Ok(output)
    }

    /// Get parameter count
    pub fn parameter_count(&self) -> usize {
        2 * self.dimension // gamma + beta
    }

    /// Reset normalization parameters
    pub fn reset(&mut self) -> Result<()> {
        self.gamma.fill(T::one());
        self.beta.fill(T::zero());
        Ok(())
    }
}

/// Dropout layer for regularization
pub struct DropoutLayer {
    /// Dropout probability
    dropout_rate: f64,

    /// Training mode flag
    training: bool,
}

impl DropoutLayer {
    /// Create new dropout layer
    pub fn new(dropout_rate: f64) -> Self {
        Self {
            dropout_rate,
            training: true,
        }
    }

    /// Forward pass through dropout
    pub fn forward<T: Float>(&self, input: &Array2<T>) -> Array2<T> {
        if !self.training || self.dropout_rate == 0.0 {
            return input.clone();
        }

        let mut output = input.clone();
        let keep_prob = 1.0 - self.dropout_rate;
        let scale = num_traits::cast::cast(1.0 / keep_prob).unwrap_or_else(|| T::zero());

        for elem in output.iter_mut() {
            if rand::random::<f64>() < self.dropout_rate {
                *elem = T::zero();
            } else {
                *elem = *elem * scale;
            }
        }

        output
    }

    /// Set training mode
    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    /// Get training mode
    pub fn is_training(&self) -> bool {
        self.training
    }
}

/// Output projection layer
pub struct OutputProjection<T: Float> {
    /// Weight matrix
    weight: Array2<T>,

    /// Bias vector
    bias: Array1<T>,

    /// Input dimension
    input_dim: usize,

    /// Output dimension
    output_dim: usize,
}

impl<T: Float> OutputProjection<T> {
    /// Create new output projection
    pub fn new(input_dim: usize, output_dim: usize) -> Result<Self> {
        let weight = Array2::zeros((input_dim, output_dim));
        let bias = Array1::zeros(output_dim);

        Ok(Self {
            weight,
            bias,
            input_dim,
            output_dim,
        })
    }

    /// Forward pass through projection
    pub fn forward(&self, input: &Array2<T>) -> Result<Array2<T>> {
        let batch_size = input.shape()[0];
        let mut output = Array2::zeros((batch_size, self.output_dim));

        // Matrix multiplication: input @ weight + bias
        for i in 0..batch_size {
            for j in 0..self.output_dim {
                let mut sum = self.bias[j];
                for k in 0..self.input_dim {
                    sum = sum + input[[i, k]] * self.weight[[k, j]];
                }
                output[[i, j]] = sum;
            }
        }

        Ok(output)
    }

    /// Get parameter count
    pub fn parameter_count(&self) -> usize {
        self.input_dim * self.output_dim + self.output_dim
    }

    /// Reset projection parameters
    pub fn reset(&mut self) -> Result<()> {
        self.weight.fill(T::zero());
        self.bias.fill(T::zero());
        Ok(())
    }
}

/// Residual connections manager
pub struct ResidualConnections<T: Float> {
    /// Model dimension
    dimension: usize,

    /// Optional learnable scaling factor
    scale_factor: Option<T>,
}

impl<T: Float> ResidualConnections<T> {
    /// Create new residual connections
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            scale_factor: None,
        }
    }

    /// Create with learnable scaling
    pub fn new_with_scaling(dimension: usize, initial_scale: T) -> Self {
        Self {
            dimension,
            scale_factor: Some(initial_scale),
        }
    }

    /// Add residual connection
    pub fn add(&self, input: &Array2<T>, residual: &Array2<T>) -> Result<Array2<T>> {
        if input.shape() != residual.shape() {
            return Err(crate::error::OptimError::Other(
                "Shape mismatch in residual connection".to_string()
            ));
        }

        let mut output = input + residual;

        if let Some(scale) = self.scale_factor {
            output.mapv_inplace(|x| x * scale);
        }

        Ok(output)
    }

    /// Set scaling factor
    pub fn set_scale_factor(&mut self, scale: T) {
        self.scale_factor = Some(scale);
    }

    /// Get scaling factor
    pub fn get_scale_factor(&self) -> Option<T> {
        self.scale_factor
    }
}

/// Activation layer with various activation functions
pub struct ActivationLayer;

impl ActivationLayer {
    /// Apply activation function
    pub fn apply<T: Float>(input: &Array2<T>, activation: ActivationFunction) -> Array2<T> {
        match activation {
            ActivationFunction::ReLU => Self::relu(input),
            ActivationFunction::GELU => Self::gelu(input),
            ActivationFunction::Swish => Self::swish(input),
            ActivationFunction::Tanh => Self::tanh(input),
            ActivationFunction::Sigmoid => Self::sigmoid(input),
            ActivationFunction::LeakyReLU => Self::leaky_relu(input, num_traits::cast::cast(0.01).unwrap_or_else(|| T::zero())),
        }
    }

    /// ReLU activation
    fn relu<T: Float>(input: &Array2<T>) -> Array2<T> {
        input.map(|&x| if x > T::zero() { x } else { T::zero() })
    }

    /// GELU activation (approximation)
    fn gelu<T: Float>(input: &Array2<T>) -> Array2<T> {
        input.map(|&x| {
            let half = num_traits::cast::cast(0.5).unwrap_or_else(|| T::zero());
            let one = T::one();
            let sqrt_2_pi = num_traits::cast::cast(0.797884560802865).unwrap_or_else(|| T::zero()); // sqrt(2/π)
            let coeff = num_traits::cast::cast(0.044715).unwrap_or_else(|| T::zero());

            let tanh_arg = sqrt_2_pi * (x + coeff * x * x * x);
            let tanh_val = tanh_arg.tanh();

            half * x * (one + tanh_val)
        })
    }

    /// Swish activation
    fn swish<T: Float>(input: &Array2<T>) -> Array2<T> {
        input.map(|&x| x * Self::sigmoid_scalar(x))
    }

    /// Tanh activation
    fn tanh<T: Float>(input: &Array2<T>) -> Array2<T> {
        input.map(|&x| x.tanh())
    }

    /// Sigmoid activation
    fn sigmoid<T: Float>(input: &Array2<T>) -> Array2<T> {
        input.map(|&x| Self::sigmoid_scalar(x))
    }

    /// Leaky ReLU activation
    fn leaky_relu<T: Float>(input: &Array2<T>, alpha: T) -> Array2<T> {
        input.map(|&x| if x > T::zero() { x } else { alpha * x })
    }

    /// Sigmoid scalar function
    fn sigmoid_scalar<T: Float>(x: T) -> T {
        let one = T::one();
        one / (one + (-x).exp())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_layer() {
        let embedding = EmbeddingLayer::<f32>::new(100, 64);
        assert!(embedding.is_ok());

        let emb = embedding.unwrap();
        assert_eq!(emb.parameter_count(), 100 * 64);
    }

    #[test]
    fn test_layer_normalization() {
        let layer_norm = LayerNormalization::<f32>::new(128);
        assert!(layer_norm.is_ok());

        let ln = layer_norm.unwrap();
        let input = Array2::<f32>::ones((2, 128));
        let result = ln.forward(&input);
        assert!(result.is_ok());
    }

    #[test]
    fn test_dropout_layer() {
        let mut dropout = DropoutLayer::new(0.5);
        let input = Array2::<f32>::ones((4, 128));

        dropout.set_training(false);
        let output = dropout.forward(&input);
        assert_eq!(output, input);

        dropout.set_training(true);
        let output = dropout.forward(&input);
        assert_eq!(output.shape(), input.shape());
    }

    #[test]
    fn test_output_projection() {
        let projection = OutputProjection::<f32>::new(128, 64);
        assert!(projection.is_ok());

        let proj = projection.unwrap();
        let input = Array2::<f32>::zeros((2, 128));
        let result = proj.forward(&input);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.shape(), &[2, 64]);
    }

    #[test]
    fn test_residual_connections() {
        let residual = ResidualConnections::<f32>::new(64);
        let input = Array2::<f32>::ones((2, 64));
        let res_input = Array2::<f32>::ones((2, 64)) * 0.5;

        let result = residual.add(&input, &res_input);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output[[0, 0]], 1.5);
    }

    #[test]
    fn test_activation_functions() {
        let input = Array2::<f32>::from_shape_vec((2, 2), vec![-1.0, 0.0, 0.5, 1.0]).unwrap();

        let relu_output = ActivationLayer::apply(&input, ActivationFunction::ReLU);
        assert_eq!(relu_output[[0, 0]], 0.0);
        assert_eq!(relu_output[[1, 1]], 1.0);

        let gelu_output = ActivationLayer::apply(&input, ActivationFunction::GELU);
        assert_eq!(gelu_output.shape(), input.shape());

        let sigmoid_output = ActivationLayer::apply(&input, ActivationFunction::Sigmoid);
        assert!(sigmoid_output.iter().all(|&x| x >= 0.0 && x <= 1.0));
    }
}