//! Feed-forward network implementations for transformer layers

use ndarray::{Array1, Array2, Axis};
use num_traits::Float;
use crate::error::Result;
use super::layers::ActivationLayer;

/// Feed-forward network implementation
pub struct FeedForwardNetwork<T: Float> {
    /// First linear layer (expansion)
    linear1: LinearLayer<T>,

    /// Second linear layer (projection)
    linear2: LinearLayer<T>,

    /// Activation function
    activation: ActivationFunction,

    /// Input dimension
    input_dimension: usize,

    /// Hidden dimension (typically 4x input dimension)
    hidden_dimension: usize,

    /// Dropout layer for regularization
    dropout: super::layers::DropoutLayer,
}

impl<T: Float> FeedForwardNetwork<T> {
    /// Create new feed-forward network
    pub fn new(
        input_dimension: usize,
        hidden_dimension: usize,
        activation: ActivationFunction,
    ) -> Result<Self> {
        let linear1 = LinearLayer::new(input_dimension, hidden_dimension)?;
        let linear2 = LinearLayer::new(hidden_dimension, input_dimension)?;
        let dropout = super::layers::DropoutLayer::new(0.1);

        Ok(Self {
            linear1,
            linear2,
            activation,
            input_dimension,
            hidden_dimension,
            dropout,
        })
    }

    /// Create with custom dropout rate
    pub fn new_with_dropout(
        input_dimension: usize,
        hidden_dimension: usize,
        activation: ActivationFunction,
        dropout_rate: f64,
    ) -> Result<Self> {
        let linear1 = LinearLayer::new(input_dimension, hidden_dimension)?;
        let linear2 = LinearLayer::new(hidden_dimension, input_dimension)?;
        let dropout = super::layers::DropoutLayer::new(dropout_rate);

        Ok(Self {
            linear1,
            linear2,
            activation,
            input_dimension,
            hidden_dimension,
            dropout,
        })
    }

    /// Forward pass through feed-forward network
    pub fn forward(&mut self, input: &Array2<T>) -> Result<Array2<T>> {
        // First linear transformation
        let hidden = self.linear1.forward(input)?;

        // Apply activation function
        let activated = ActivationLayer::apply(&hidden, self.activation);

        // Apply dropout
        let dropout_output = self.dropout.forward(&activated);

        // Second linear transformation
        let output = self.linear2.forward(&dropout_output)?;

        Ok(output)
    }

    /// Get parameter count
    pub fn parameter_count(&self) -> usize {
        self.linear1.parameter_count() + self.linear2.parameter_count()
    }

    /// Reset all parameters
    pub fn reset(&mut self) -> Result<()> {
        self.linear1.reset()?;
        self.linear2.reset()?;
        Ok(())
    }

    /// Set training mode
    pub fn set_training(&mut self, training: bool) {
        self.dropout.set_training(training);
    }

    /// Get activation function
    pub fn get_activation(&self) -> ActivationFunction {
        self.activation
    }

    /// Set activation function
    pub fn set_activation(&mut self, activation: ActivationFunction) {
        self.activation = activation;
    }
}

/// Linear layer implementation
pub struct LinearLayer<T: Float> {
    /// Weight matrix
    weight: Array2<T>,

    /// Bias vector
    bias: Array1<T>,

    /// Input dimension
    input_dim: usize,

    /// Output dimension
    output_dim: usize,
}

impl<T: Float> LinearLayer<T> {
    /// Create new linear layer
    pub fn new(input_dim: usize, output_dim: usize) -> Result<Self> {
        // Xavier/Glorot initialization
        let scale = T::from(2.0 / (input_dim + output_dim) as f64).unwrap().sqrt();
        let mut weight = Array2::zeros((input_dim, output_dim));
        let bias = Array1::zeros(output_dim);

        // Initialize weights with Xavier initialization
        for i in 0..input_dim {
            for j in 0..output_dim {
                let random_val = T::from(rand::random::<f64>() * 2.0 - 1.0).unwrap();
                weight[[i, j]] = random_val * scale;
            }
        }

        Ok(Self {
            weight,
            bias,
            input_dim,
            output_dim,
        })
    }

    /// Create with He initialization (better for ReLU)
    pub fn new_he_init(input_dim: usize, output_dim: usize) -> Result<Self> {
        let scale = T::from(2.0 / input_dim as f64).unwrap().sqrt();
        let mut weight = Array2::zeros((input_dim, output_dim));
        let bias = Array1::zeros(output_dim);

        for i in 0..input_dim {
            for j in 0..output_dim {
                let random_val = T::from(rand::random::<f64>() * 2.0 - 1.0).unwrap();
                weight[[i, j]] = random_val * scale;
            }
        }

        Ok(Self {
            weight,
            bias,
            input_dim,
            output_dim,
        })
    }

    /// Forward pass through linear layer
    pub fn forward(&self, input: &Array2<T>) -> Result<Array2<T>> {
        let batch_size = input.shape()[0];
        let input_features = input.shape()[1];

        if input_features != self.input_dim {
            return Err(crate::error::OptimError::Other(
                format!("Input dimension mismatch: expected {}, got {}",
                       self.input_dim, input_features)
            ));
        }

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

    /// Reset parameters
    pub fn reset(&mut self) -> Result<()> {
        // Re-initialize with Xavier
        let scale = T::from(2.0 / (self.input_dim + self.output_dim) as f64).unwrap().sqrt();

        for i in 0..self.input_dim {
            for j in 0..self.output_dim {
                let random_val = T::from(rand::random::<f64>() * 2.0 - 1.0).unwrap();
                self.weight[[i, j]] = random_val * scale;
            }
        }

        self.bias.fill(T::zero());
        Ok(())
    }

    /// Get weight matrix reference
    pub fn get_weights(&self) -> &Array2<T> {
        &self.weight
    }

    /// Get bias vector reference
    pub fn get_bias(&self) -> &Array1<T> {
        &self.bias
    }

    /// Update weights (for training)
    pub fn update_weights(&mut self, weight_delta: &Array2<T>, bias_delta: &Array1<T>) -> Result<()> {
        if weight_delta.shape() != self.weight.shape() {
            return Err(crate::error::OptimError::Other(
                "Weight delta shape mismatch".to_string()
            ));
        }

        if bias_delta.len() != self.bias.len() {
            return Err(crate::error::OptimError::Other(
                "Bias delta shape mismatch".to_string()
            ));
        }

        self.weight = &self.weight - weight_delta;
        self.bias = &self.bias - bias_delta;

        Ok(())
    }
}

/// Gated Linear Unit (GLU) implementation
pub struct GatedLinearUnit<T: Float> {
    /// Linear layer for gate
    gate_linear: LinearLayer<T>,

    /// Linear layer for values
    value_linear: LinearLayer<T>,

    /// Input dimension
    input_dimension: usize,

    /// Hidden dimension
    hidden_dimension: usize,
}

impl<T: Float> GatedLinearUnit<T> {
    /// Create new GLU
    pub fn new(input_dimension: usize, hidden_dimension: usize) -> Result<Self> {
        let gate_linear = LinearLayer::new(input_dimension, hidden_dimension)?;
        let value_linear = LinearLayer::new(input_dimension, hidden_dimension)?;

        Ok(Self {
            gate_linear,
            value_linear,
            input_dimension,
            hidden_dimension,
        })
    }

    /// Forward pass through GLU
    pub fn forward(&self, input: &Array2<T>) -> Result<Array2<T>> {
        let gate = self.gate_linear.forward(input)?;
        let value = self.value_linear.forward(input)?;

        // Apply sigmoid to gate and element-wise multiply
        let sigmoid_gate = ActivationLayer::apply(&gate, ActivationFunction::Sigmoid);
        let output = &sigmoid_gate * &value;

        Ok(output)
    }

    /// Get parameter count
    pub fn parameter_count(&self) -> usize {
        self.gate_linear.parameter_count() + self.value_linear.parameter_count()
    }

    /// Reset parameters
    pub fn reset(&mut self) -> Result<()> {
        self.gate_linear.reset()?;
        self.value_linear.reset()?;
        Ok(())
    }
}

/// Swish GLU variant
pub struct SwiGLU<T: Float> {
    /// Linear layer for gate
    gate_linear: LinearLayer<T>,

    /// Linear layer for values
    value_linear: LinearLayer<T>,

    /// Input dimension
    input_dimension: usize,

    /// Hidden dimension
    hidden_dimension: usize,
}

impl<T: Float> SwiGLU<T> {
    /// Create new SwiGLU
    pub fn new(input_dimension: usize, hidden_dimension: usize) -> Result<Self> {
        let gate_linear = LinearLayer::new(input_dimension, hidden_dimension)?;
        let value_linear = LinearLayer::new(input_dimension, hidden_dimension)?;

        Ok(Self {
            gate_linear,
            value_linear,
            input_dimension,
            hidden_dimension,
        })
    }

    /// Forward pass through SwiGLU
    pub fn forward(&self, input: &Array2<T>) -> Result<Array2<T>> {
        let gate = self.gate_linear.forward(input)?;
        let value = self.value_linear.forward(input)?;

        // Apply Swish to gate and element-wise multiply
        let swish_gate = ActivationLayer::apply(&gate, ActivationFunction::Swish);
        let output = &swish_gate * &value;

        Ok(output)
    }

    /// Get parameter count
    pub fn parameter_count(&self) -> usize {
        self.gate_linear.parameter_count() + self.value_linear.parameter_count()
    }

    /// Reset parameters
    pub fn reset(&mut self) -> Result<()> {
        self.gate_linear.reset()?;
        self.value_linear.reset()?;
        Ok(())
    }
}

/// Expert mixture for sparse feed-forward networks
pub struct MixtureOfExperts<T: Float> {
    /// Individual expert networks
    experts: Vec<FeedForwardNetwork<T>>,

    /// Gating network
    gate: LinearLayer<T>,

    /// Number of experts
    num_experts: usize,

    /// Number of experts to activate (top-k)
    top_k: usize,

    /// Input dimension
    input_dimension: usize,

    /// Hidden dimension per expert
    hidden_dimension: usize,
}

impl<T: Float> MixtureOfExperts<T> {
    /// Create new mixture of experts
    pub fn new(
        input_dimension: usize,
        hidden_dimension: usize,
        num_experts: usize,
        top_k: usize,
        activation: ActivationFunction,
    ) -> Result<Self> {
        let mut experts = Vec::new();
        for _ in 0..num_experts {
            experts.push(FeedForwardNetwork::new(
                input_dimension,
                hidden_dimension,
                activation,
            )?);
        }

        let gate = LinearLayer::new(input_dimension, num_experts)?;

        Ok(Self {
            experts,
            gate,
            num_experts,
            top_k: top_k.min(num_experts),
            input_dimension,
            hidden_dimension,
        })
    }

    /// Forward pass through mixture of experts
    pub fn forward(&mut self, input: &Array2<T>) -> Result<Array2<T>> {
        let batch_size = input.shape()[0];

        // Compute gating scores
        let gate_scores = self.gate.forward(input)?;
        let gate_probs = self.softmax(&gate_scores);

        // Find top-k experts for each sample
        let mut output = Array2::zeros((batch_size, self.input_dimension));

        for i in 0..batch_size {
            let sample_input = input.row(i).insert_axis(Axis(0)).to_owned();
            let sample_probs = gate_probs.row(i);

            // Get top-k indices
            let mut prob_indices: Vec<(usize, T)> = sample_probs
                .iter()
                .enumerate()
                .map(|(idx, &prob)| (idx, prob))
                .collect();

            prob_indices.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            let top_k_indices: Vec<usize> = prob_indices
                .iter()
                .take(self.top_k)
                .map(|(idx, _)| *idx)
                .collect();

            // Compute weighted sum from top-k experts
            let mut sample_output = Array1::zeros(self.input_dimension);
            let mut total_weight = T::zero();

            for &expert_idx in &top_k_indices {
                let expert_output = self.experts[expert_idx].forward(&sample_input)?;
                let weight = sample_probs[expert_idx];

                total_weight = total_weight + weight;

                for j in 0..self.input_dimension {
                    sample_output[j] = sample_output[j] + weight * expert_output[[0, j]];
                }
            }

            // Normalize by total weight
            if total_weight > T::zero() {
                for j in 0..self.input_dimension {
                    sample_output[j] = sample_output[j] / total_weight;
                    output[[i, j]] = sample_output[j];
                }
            }
        }

        Ok(output)
    }

    /// Softmax activation for gating
    fn softmax(&self, input: &Array2<T>) -> Array2<T> {
        let mut output = Array2::zeros(input.raw_dim());
        let batch_size = input.shape()[0];

        for i in 0..batch_size {
            let row = input.row(i);
            let max_val = row.iter().fold(T::neg_infinity(), |a, &b| a.max(b));

            let mut exp_sum = T::zero();
            let mut exp_row = Array1::zeros(row.len());

            for (j, &val) in row.iter().enumerate() {
                exp_row[j] = (val - max_val).exp();
                exp_sum = exp_sum + exp_row[j];
            }

            for (j, &exp_val) in exp_row.iter().enumerate() {
                output[[i, j]] = exp_val / exp_sum;
            }
        }

        output
    }

    /// Get parameter count
    pub fn parameter_count(&self) -> usize {
        let expert_params: usize = self.experts.iter()
            .map(|expert| expert.parameter_count())
            .sum();

        expert_params + self.gate.parameter_count()
    }

    /// Reset all parameters
    pub fn reset(&mut self) -> Result<()> {
        for expert in &mut self.experts {
            expert.reset()?;
        }
        self.gate.reset()?;
        Ok(())
    }

    /// Set training mode for all experts
    pub fn set_training(&mut self, training: bool) {
        for expert in &mut self.experts {
            expert.set_training(training);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feedforward_network() {
        let mut ffn = FeedForwardNetwork::<f32>::new(128, 512, crate::learned_optimizers::transformer_based_optimizer::config::ActivationFunction::ReLU);
        assert!(ffn.is_ok());

        let mut network = ffn.unwrap();
        let input = Array2::<f32>::ones((4, 128));
        let result = network.forward(&input);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.shape(), &[4, 128]);
    }

    #[test]
    fn test_linear_layer() {
        let linear = LinearLayer::<f32>::new(64, 128);
        assert!(linear.is_ok());

        let layer = linear.unwrap();
        let input = Array2::<f32>::zeros((2, 64));
        let result = layer.forward(&input);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.shape(), &[2, 128]);
        assert_eq!(layer.parameter_count(), 64 * 128 + 128);
    }

    #[test]
    fn test_gated_linear_unit() {
        let glu = GatedLinearUnit::<f32>::new(128, 256);
        assert!(glu.is_ok());

        let unit = glu.unwrap();
        let input = Array2::<f32>::ones((2, 128));
        let result = unit.forward(&input);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.shape(), &[2, 256]);
    }

    #[test]
    fn test_swiglu() {
        let swiglu = SwiGLU::<f32>::new(128, 256);
        assert!(swiglu.is_ok());

        let unit = swiglu.unwrap();
        let input = Array2::<f32>::ones((2, 128));
        let result = unit.forward(&input);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.shape(), &[2, 256]);
    }

    #[test]
    fn test_mixture_of_experts() {
        let mut moe = MixtureOfExperts::<f32>::new(
            128, 256, 4, 2, ActivationFunction::ReLU
        );
        assert!(moe.is_ok());

        let mut mixture = moe.unwrap();
        let input = Array2::<f32>::ones((3, 128));
        let result = mixture.forward(&input);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.shape(), &[3, 128]);
    }

    #[test]
    fn test_linear_layer_initialization() {
        let xavier_layer = LinearLayer::<f32>::new(64, 128).unwrap();
        let he_layer = LinearLayer::<f32>::new_he_init(64, 128).unwrap();

        assert_eq!(xavier_layer.parameter_count(), he_layer.parameter_count());
        assert_eq!(xavier_layer.get_weights().shape(), he_layer.get_weights().shape());
    }
}

// Re-export for backward compatibility
pub use super::config::ActivationFunction;