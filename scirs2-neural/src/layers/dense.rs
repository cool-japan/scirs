//! Dense (fully connected) layer implementation

use crate::activations_minimal::Activation;
use crate::error::{NeuralError, Result};
use crate::layers::{Layer, ParamLayer};
use scirs2_core::ndarray::{Array, IxDyn, ScalarOperand};
use scirs2_core::numeric::Float;
use scirs2_core::random::{Distribution, Uniform};
use std::fmt::Debug;

/// Dense (fully connected) layer for neural networks.
///
/// A dense layer performs the operation: y = activation(W * x + b), where W is the weight matrix,
/// x is the input vector, b is the bias vector, and activation is the activation function.
pub struct Dense<F: Float + Debug + Send + Sync> {
    /// Number of input features
    input_dim: usize,
    /// Number of output features
    output_dim: usize,
    /// Weight matrix
    weights: Array<F, IxDyn>,
    /// Bias vector
    biases: Array<F, IxDyn>,
    /// Gradient of the weights
    dweights: std::sync::RwLock<Array<F, IxDyn>>,
    /// Gradient of the biases
    dbiases: std::sync::RwLock<Array<F, IxDyn>>,
    /// Activation function, if any
    activation: Option<Box<dyn Activation<F> + Send + Sync>>,
    /// Input from the forward pass, needed in backward pass
    input: std::sync::RwLock<Option<Array<F, IxDyn>>>,
    /// Output before activation, needed in backward pass
    output_pre_activation: std::sync::RwLock<Option<Array<F, IxDyn>>>,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + 'static> std::fmt::Debug for Dense<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Dense")
            .field("input_dim", &self.input_dim)
            .field("output_dim", &self.output_dim)
            .field("weightsshape", &self.weights.shape())
            .field("biasesshape", &self.biases.shape())
            .field("has_activation", &self.activation.is_some())
            .finish()
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + 'static> Clone for Dense<F> {
    fn clone(&self) -> Self {
        Self {
            input_dim: self.input_dim,
            output_dim: self.output_dim,
            weights: self.weights.clone(),
            biases: self.biases.clone(),
            dweights: std::sync::RwLock::new(self.dweights.read().unwrap().clone()),
            dbiases: std::sync::RwLock::new(self.dbiases.read().unwrap().clone()),
            // We can't clone trait objects, so we skip the activation
            activation: None,
            input: std::sync::RwLock::new(self.input.read().unwrap().clone()),
            output_pre_activation: std::sync::RwLock::new(
                self.output_pre_activation.read().unwrap().clone(),
            ),
        }
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + 'static> Dense<F> {
    /// Create a new dense layer.
    ///
    /// # Arguments
    /// * `input_dim` - Number of input features
    /// * `output_dim` - Number of output features
    /// * `activation_name` - Optional activation function name
    /// * `rng` - Random number generator for weight initialization
    pub fn new<R: scirs2_core::random::Rng + scirs2_core::random::RngCore>(
        input_dim: usize,
        output_dim: usize,
        activation_name: Option<&str>,
        rng: &mut R,
    ) -> Result<Self> {
        // Create activation function from _name
        let activation = if let Some(name) = activation_name {
            match name.to_lowercase().as_str() {
                "relu" => Some(Box::new(crate::activations_minimal::ReLU::new())
                    as Box<dyn Activation<F> + Send + Sync>),
                "sigmoid" => Some(Box::new(crate::activations_minimal::Sigmoid::new())
                    as Box<dyn Activation<F> + Send + Sync>),
                "tanh" => Some(Box::new(crate::activations_minimal::Tanh::new())
                    as Box<dyn Activation<F> + Send + Sync>),
                "softmax" => Some(Box::new(crate::activations_minimal::Softmax::new(-1))
                    as Box<dyn Activation<F> + Send + Sync>),
                "gelu" => Some(Box::new(crate::activations_minimal::GELU::new())
                    as Box<dyn Activation<F> + Send + Sync>),
                _ => None,
            }
        } else {
            None
        };

        // Initialize weights with Xavier/Glorot initialization
        let scale = F::from(1.0 / f64::sqrt(input_dim as f64)).ok_or_else(|| {
            NeuralError::InvalidArchitecture("Failed to convert scale factor".to_string())
        })?;

        // Create a 2D weights array
        let uniform = Uniform::new(-1.0, 1.0).map_err(|e| {
            NeuralError::InvalidArchitecture(format!("Failed to create uniform distribution: {e}"))
        })?;
        let weights_vec: Vec<F> = (0..(input_dim * output_dim))
            .map(|_| {
                let val = F::from(uniform.sample(rng)).ok_or_else(|| {
                    NeuralError::InvalidArchitecture("Failed to convert random value".to_string())
                });
                val.map(|v| v * scale).unwrap_or_else(|_| F::zero())
            })
            .collect();

        let weights =
            Array::from_shape_vec(IxDyn(&[input_dim, output_dim]), weights_vec).map_err(|e| {
                NeuralError::InvalidArchitecture(format!("Failed to create weights array: {e}"))
            })?;

        // Initialize biases with zeros
        let biases = Array::zeros(IxDyn(&[output_dim]));

        // Initialize gradient arrays with zeros
        let dweights = std::sync::RwLock::new(Array::zeros(weights.dim()));
        let dbiases = std::sync::RwLock::new(Array::zeros(biases.dim()));

        Ok(Self {
            input_dim,
            output_dim,
            weights,
            biases,
            dweights,
            dbiases,
            activation,
            input: std::sync::RwLock::new(None),
            output_pre_activation: std::sync::RwLock::new(None),
        })
    }

    /// Get the input dimension
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// Get the output dimension
    pub fn output_dim(&self) -> usize {
        self.output_dim
    }

    /// Simple matrix multiplication for forward pass
    fn compute_forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let batch_size = input.shape()[0];
        let mut output = Array::zeros(IxDyn(&[batch_size, self.output_dim]));

        // Matrix multiplication: output = input @ weights
        for batch in 0..batch_size {
            for out_idx in 0..self.output_dim {
                let mut sum = F::zero();
                for in_idx in 0..self.input_dim {
                    sum = sum + input[[batch, in_idx]] * self.weights[[in_idx, out_idx]];
                }
                // Add bias
                output[[batch, out_idx]] = sum + self.biases[out_idx];
            }
        }

        Ok(output)
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + 'static> Layer<F> for Dense<F> {
    fn forward(
        &self,
        input: &Array<F, scirs2_core::ndarray::IxDyn>,
    ) -> Result<Array<F, scirs2_core::ndarray::IxDyn>> {
        // Cache input for backward pass
        {
            let mut input_cache = self.input.write().unwrap();
            *input_cache = Some(input.clone());
        }

        // Ensure input is 2D
        let input_2d = if input.ndim() == 1 {
            input
                .clone()
                .into_shape_with_order(IxDyn(&[1, self.input_dim]))
                .map_err(|e| NeuralError::InferenceError(format!("Failed to reshape input: {e}")))?
        } else {
            input.clone()
        };

        // Validate input dimensions
        if input_2d.shape()[1] != self.input_dim {
            return Err(NeuralError::InvalidArgument(format!(
                "Input dimension mismatch: expected {}, got {}",
                self.input_dim,
                input_2d.shape()[1]
            )));
        }

        // Compute linear transformation
        let output = self.compute_forward(&input_2d)?;

        // Cache pre-activation output
        {
            let mut pre_activation_cache = self.output_pre_activation.write().unwrap();
            *pre_activation_cache = Some(output.clone());
        }

        // Apply activation function if present
        if let Some(ref activation) = self.activation {
            activation.forward(&output)
        } else {
            Ok(output)
        }
    }

    fn backward(
        &self,
        _input: &Array<F, scirs2_core::ndarray::IxDyn>,
        grad_output: &Array<F, scirs2_core::ndarray::IxDyn>,
    ) -> Result<Array<F, scirs2_core::ndarray::IxDyn>> {
        // Get cached data
        let cached_input = {
            let cache = self.input.read().unwrap();
            cache.clone().ok_or_else(|| {
                NeuralError::InferenceError("No cached _input for backward pass".to_string())
            })?
        };

        let pre_activation = {
            let cache = self.output_pre_activation.read().unwrap();
            cache.clone().ok_or_else(|| {
                NeuralError::InferenceError(
                    "No cached pre-activation _output for backward pass".to_string(),
                )
            })?
        };

        // Apply activation gradient if present
        let grad_pre_activation = if let Some(ref activation) = self.activation {
            activation.backward(grad_output, &pre_activation)?
        } else {
            grad_output.clone()
        };

        // Ensure gradients are 2D
        let grad_2d = if grad_pre_activation.ndim() == 1 {
            grad_pre_activation
                .into_shape_with_order(IxDyn(&[1, self.output_dim]))
                .map_err(|e| {
                    NeuralError::InferenceError(format!("Failed to reshape gradient: {e}"))
                })?
        } else {
            grad_pre_activation
        };

        let input_2d = if cached_input.ndim() == 1 {
            cached_input
                .into_shape_with_order(IxDyn(&[1, self.input_dim]))
                .map_err(|e| {
                    NeuralError::InferenceError(format!("Failed to reshape cached input: {e}"))
                })?
        } else {
            cached_input
        };

        let batch_size = grad_2d.shape()[0];

        // Compute weight gradients: dW = input.T @ grad_output
        let mut dweights = Array::zeros(IxDyn(&[self.input_dim, self.output_dim]));
        for i in 0..self.input_dim {
            for j in 0..self.output_dim {
                let mut sum = F::zero();
                for b in 0..batch_size {
                    sum = sum + input_2d[[b, i]] * grad_2d[[b, j]];
                }
                dweights[[i, j]] = sum;
            }
        }

        // Compute bias gradients: db = sum(grad_output, axis=0)
        let mut dbiases = Array::zeros(IxDyn(&[self.output_dim]));
        for j in 0..self.output_dim {
            let mut sum = F::zero();
            for b in 0..batch_size {
                sum = sum + grad_2d[[b, j]];
            }
            dbiases[j] = sum;
        }

        // Update internal gradients
        {
            let mut dweights_guard = self.dweights.write().unwrap();
            *dweights_guard = dweights;
        }
        {
            let mut dbiases_guard = self.dbiases.write().unwrap();
            *dbiases_guard = dbiases;
        }

        // Compute gradient w.r.t. _input: grad_input = grad_output @ weights.T
        let mut grad_input = Array::zeros(IxDyn(&[batch_size, self.input_dim]));
        for b in 0..batch_size {
            for i in 0..self.input_dim {
                let mut sum = F::zero();
                for j in 0..self.output_dim {
                    sum = sum + grad_2d[[b, j]] * self.weights[[i, j]];
                }
                grad_input[[b, i]] = sum;
            }
        }

        Ok(grad_input)
    }

    fn update(&mut self, learningrate: F) -> Result<()> {
        let dweights = {
            let dweights_guard = self.dweights.read().unwrap();
            dweights_guard.clone()
        };
        let dbiases = {
            let dbiases_guard = self.dbiases.read().unwrap();
            dbiases_guard.clone()
        };

        // Update weights and biases using gradient descent
        for i in 0..self.input_dim {
            for j in 0..self.output_dim {
                self.weights[[i, j]] = self.weights[[i, j]] - learningrate * dweights[[i, j]];
            }
        }

        for j in 0..self.output_dim {
            self.biases[j] = self.biases[j] - learningrate * dbiases[j];
        }

        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn layer_type(&self) -> &str {
        "Dense"
    }

    fn parameter_count(&self) -> usize {
        self.weights.len() + self.biases.len()
    }

    fn layer_description(&self) -> String {
        format!(
            "type:Dense, input, _dim:{}, output, _dim:{}, params:{}",
            self.input_dim,
            self.output_dim,
            self.parameter_count()
        )
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + 'static> ParamLayer<F> for Dense<F> {
    fn get_parameters(&self) -> Vec<Array<F, scirs2_core::ndarray::IxDyn>> {
        vec![self.weights.clone(), self.biases.clone()]
    }

    fn get_gradients(&self) -> Vec<Array<F, scirs2_core::ndarray::IxDyn>> {
        // This method has limitations with RwLock - in practice this would need redesign
        vec![]
    }

    fn set_parameters(&mut self, params: Vec<Array<F, scirs2_core::ndarray::IxDyn>>) -> Result<()> {
        if params.len() != 2 {
            return Err(NeuralError::InvalidArchitecture(format!(
                "Expected 2 parameters (weights, biases), got {}",
                params.len()
            )));
        }

        let weights = &params[0];
        let biases = &params[1];

        if weights.shape() != self.weights.shape() {
            return Err(NeuralError::InvalidArchitecture(format!(
                "Weights shape mismatch: expected {:?}, got {:?}",
                self.weights.shape(),
                weights.shape()
            )));
        }

        if biases.shape() != self.biases.shape() {
            return Err(NeuralError::InvalidArchitecture(format!(
                "Biases shape mismatch: expected {:?}, got {:?}",
                self.biases.shape(),
                biases.shape()
            )));
        }

        self.weights = weights.clone();
        self.biases = biases.clone();

        Ok(())
    }
}

// Explicit Send + Sync implementations for Dense layer
unsafe impl<F: Float + Debug + Send + Sync> Send for Dense<F> {}
unsafe impl<F: Float + Debug + Send + Sync> Sync for Dense<F> {}
