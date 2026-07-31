//! Neural network components for sparse matrix optimization
//!
//! This module contains the neural network architectures used in the adaptive
//! sparse matrix processing system.

use scirs2_core::numeric::{Float, SparseElement};
use scirs2_core::random::{Rng, RngExt};
use std::collections::HashMap;

/// Neural network layer for sparse matrix optimization
#[derive(Debug, Clone)]
pub(crate) struct NeuralLayer {
    pub weights: Vec<Vec<f64>>,
    pub biases: Vec<f64>,
    pub activation: ActivationFunction,
}

/// Activation functions for neural network layers
#[derive(Debug, Clone, Copy)]
pub enum ActivationFunction {
    ReLU,
    Sigmoid,
    #[allow(dead_code)]
    Tanh,
    #[allow(dead_code)]
    Swish,
    #[allow(dead_code)]
    Gelu,
}

/// Neural network for sparse matrix optimization
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub(crate) struct NeuralNetwork {
    pub layers: Vec<NeuralLayer>,
    pub attention_weights: Vec<Vec<f64>>,
    /// Multi-head attention mechanisms
    pub attention_heads: Vec<AttentionHead>,
    /// Layer normalization parameters
    pub layer_norms: Vec<LayerNorm>,
}

/// Multi-head attention mechanism
#[derive(Debug, Clone)]
pub(crate) struct AttentionHead {
    pub query_weights: Vec<Vec<f64>>,
    pub key_weights: Vec<Vec<f64>>,
    pub value_weights: Vec<Vec<f64>>,
    pub output_weights: Vec<Vec<f64>>,
    pub head_dim: usize,
}

/// Layer normalization
#[derive(Debug, Clone)]
pub(crate) struct LayerNorm {
    pub gamma: Vec<f64>,
    pub beta: Vec<f64>,
    pub eps: f64,
}

/// Forward cache for neural network computations
#[derive(Debug, Clone)]
pub(crate) struct ForwardCache {
    pub layer_outputs: Vec<Vec<f64>>,
    pub attention_outputs: Vec<Vec<f64>>,
    pub normalized_outputs: Vec<Vec<f64>>,
}

/// Network gradients for backpropagation
#[derive(Debug, Clone)]
pub(crate) struct NetworkGradients {
    pub weight_gradients: Vec<Vec<Vec<f64>>>,
    pub bias_gradients: Vec<Vec<f64>>,
}

impl NeuralNetwork {
    /// Create a new neural network with specified architecture
    pub fn new(
        input_size: usize,
        hidden_layers: usize,
        neurons_per_layer: usize,
        output_size: usize,
        attention_heads: usize,
    ) -> Self {
        let mut layers = Vec::new();
        let mut layer_norms = Vec::new();

        // Input layer
        let input_layer = NeuralLayer {
            weights: Self::initialize_weights(input_size, neurons_per_layer),
            biases: vec![0.0; neurons_per_layer],
            activation: ActivationFunction::ReLU,
        };
        layers.push(input_layer);
        layer_norms.push(LayerNorm::new(neurons_per_layer));

        // Hidden layers
        for _ in 0..hidden_layers.saturating_sub(1) {
            let layer = NeuralLayer {
                weights: Self::initialize_weights(neurons_per_layer, neurons_per_layer),
                biases: vec![0.0; neurons_per_layer],
                activation: ActivationFunction::ReLU,
            };
            layers.push(layer);
            layer_norms.push(LayerNorm::new(neurons_per_layer));
        }

        // Output layer
        let output_layer = NeuralLayer {
            weights: Self::initialize_weights(neurons_per_layer, output_size),
            biases: vec![0.0; output_size],
            activation: ActivationFunction::Sigmoid,
        };
        layers.push(output_layer);
        layer_norms.push(LayerNorm::new(output_size));

        // Initialize attention heads
        let mut attention_heads_vec = Vec::new();
        for _ in 0..attention_heads {
            attention_heads_vec.push(AttentionHead::new(neurons_per_layer));
        }

        Self {
            layers,
            attention_weights: vec![vec![1.0; neurons_per_layer]; attention_heads],
            attention_heads: attention_heads_vec,
            layer_norms,
        }
    }

    /// Initialize weights using Xavier initialization
    fn initialize_weights(input_size: usize, output_size: usize) -> Vec<Vec<f64>> {
        let mut rng = scirs2_core::random::thread_rng();
        let bound = (6.0 / (input_size + output_size) as f64).sqrt();

        (0..output_size)
            .map(|_| {
                (0..input_size)
                    .map(|_| rng.random_range(-bound..bound))
                    .collect()
            })
            .collect()
    }

    /// Forward pass through the network
    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        let mut current_input = input.to_vec();

        for (i, layer) in self.layers.iter().enumerate() {
            let mut output = vec![0.0; layer.biases.len()];

            // Linear transformation
            for (j, neuron_weights) in layer.weights.iter().enumerate() {
                let mut sum = layer.biases[j];
                for (k, &input_val) in current_input.iter().enumerate() {
                    sum += neuron_weights[k] * input_val;
                }
                output[j] = sum;
            }

            // Apply activation function
            for val in &mut output {
                *val = Self::apply_activation(*val, layer.activation);
            }

            // Apply layer normalization
            if i < self.layer_norms.len() {
                output = self.layer_norms[i].normalize(&output);
            }

            current_input = output;
        }

        current_input
    }

    /// Forward pass with caching for backpropagation
    pub fn forward_with_cache(&self, input: &[f64]) -> (Vec<f64>, ForwardCache) {
        let mut layer_outputs = Vec::new();
        let mut attention_outputs = Vec::new();
        let mut normalized_outputs = Vec::new();
        let mut current_input = input.to_vec();

        for (i, layer) in self.layers.iter().enumerate() {
            let mut output = vec![0.0; layer.biases.len()];

            // Linear transformation
            for (j, neuron_weights) in layer.weights.iter().enumerate() {
                let mut sum = layer.biases[j];
                for (k, &input_val) in current_input.iter().enumerate() {
                    sum += neuron_weights[k] * input_val;
                }
                output[j] = sum;
            }

            layer_outputs.push(output.clone());

            // Apply activation function
            for val in &mut output {
                *val = Self::apply_activation(*val, layer.activation);
            }

            // Apply attention if not the last layer
            if i < self.layers.len() - 1 && !self.attention_heads.is_empty() {
                let attention_output = self.apply_attention(&output, i);
                attention_outputs.push(attention_output.clone());
                output = attention_output;
            }

            // Apply layer normalization
            if i < self.layer_norms.len() {
                output = self.layer_norms[i].normalize(&output);
                normalized_outputs.push(output.clone());
            }

            current_input = output;
        }

        let cache = ForwardCache {
            layer_outputs,
            attention_outputs,
            normalized_outputs,
        };

        (current_input, cache)
    }

    /// Apply attention mechanism
    fn apply_attention(&self, input: &[f64], layer_idx: usize) -> Vec<f64> {
        if self.attention_heads.is_empty() {
            return input.to_vec();
        }

        let mut attention_output = vec![0.0; input.len()];
        let num_heads = self.attention_heads.len();

        for head in &self.attention_heads {
            let head_output = head.forward(input);
            for (i, &val) in head_output.iter().enumerate() {
                if i < attention_output.len() {
                    attention_output[i] += val / num_heads as f64;
                }
            }
        }

        // Add residual connection
        for (i, &input_val) in input.iter().enumerate() {
            if i < attention_output.len() {
                attention_output[i] += input_val;
            }
        }

        attention_output
    }

    /// Apply activation function
    fn apply_activation(x: f64, activation: ActivationFunction) -> f64 {
        match activation {
            ActivationFunction::ReLU => x.max(0.0),
            ActivationFunction::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            ActivationFunction::Tanh => x.tanh(),
            ActivationFunction::Swish => x * (1.0 / (1.0 + (-x).exp())),
            ActivationFunction::Gelu => 0.5 * x * (1.0 + (x * 0.7978845608028654).tanh()),
        }
    }

    /// Update weights using gradients
    pub fn update_weights(&mut self, gradients: &NetworkGradients, learning_rate: f64) {
        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            if layer_idx < gradients.weight_gradients.len() {
                let layer_weight_grads = &gradients.weight_gradients[layer_idx];
                for (neuron_idx, neuron_weights) in layer.weights.iter_mut().enumerate() {
                    if neuron_idx < layer_weight_grads.len() {
                        let neuron_grads = &layer_weight_grads[neuron_idx];
                        for (weight_idx, weight) in neuron_weights.iter_mut().enumerate() {
                            if weight_idx < neuron_grads.len() {
                                *weight -= learning_rate * neuron_grads[weight_idx];
                            }
                        }
                    }
                }
            }

            if layer_idx < gradients.bias_gradients.len() {
                let bias_grads = &gradients.bias_gradients[layer_idx];
                for (bias_idx, bias) in layer.biases.iter_mut().enumerate() {
                    if bias_idx < bias_grads.len() {
                        *bias -= learning_rate * bias_grads[bias_idx];
                    }
                }
            }
        }
    }

    /// Derivative of the activation function evaluated at pre-activation value `x`.
    fn activation_derivative(x: f64, activation: ActivationFunction) -> f64 {
        match activation {
            ActivationFunction::ReLU => {
                if x > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            ActivationFunction::Sigmoid => {
                let s = 1.0 / (1.0 + (-x).exp());
                s * (1.0 - s)
            }
            ActivationFunction::Tanh => {
                let t = x.tanh();
                1.0 - t * t
            }
            ActivationFunction::Swish => {
                let s = 1.0 / (1.0 + (-x).exp());
                s + x * s * (1.0 - s)
            }
            ActivationFunction::Gelu => {
                let c = 0.7978845608028654;
                let t = (c * x).tanh();
                0.5 * (1.0 + t) + 0.5 * x * c * (1.0 - t * t)
            }
        }
    }

    /// Compute network gradients via backpropagation.
    ///
    /// Uses cached pre-activation outputs to compute dL/dW and dL/db
    /// for each layer, where L = 0.5 * ||output - target||^2.
    pub fn compute_gradients(
        &self,
        input: &[f64],
        target: &[f64],
        cache: &ForwardCache,
    ) -> NetworkGradients {
        let num_layers = self.layers.len();
        let mut weight_gradients: Vec<Vec<Vec<f64>>> = Vec::with_capacity(num_layers);
        let mut bias_gradients: Vec<Vec<f64>> = Vec::with_capacity(num_layers);

        // Collect the input to each layer during the forward pass.
        let mut layer_inputs: Vec<Vec<f64>> = Vec::with_capacity(num_layers);
        {
            let mut current = input.to_vec();
            for (l, layer) in self.layers.iter().enumerate() {
                layer_inputs.push(current.clone());
                let mut output = vec![0.0; layer.biases.len()];
                for (j, neuron_w) in layer.weights.iter().enumerate() {
                    let mut s = layer.biases[j];
                    for (k, &iv) in current.iter().enumerate() {
                        if k < neuron_w.len() {
                            s += neuron_w[k] * iv;
                        }
                    }
                    output[j] = Self::apply_activation(s, layer.activation);
                }
                if l < self.layer_norms.len() {
                    output = self.layer_norms[l].normalize(&output);
                }
                current = output;
            }
        }

        // Compute output from cache
        let last_output = if !cache.layer_outputs.is_empty() {
            let pre_act = &cache.layer_outputs[num_layers - 1];
            let act = self.layers[num_layers - 1].activation;
            pre_act
                .iter()
                .map(|&z| Self::apply_activation(z, act))
                .collect::<Vec<_>>()
        } else {
            self.forward(input)
        };

        let output_pre_act = if num_layers <= cache.layer_outputs.len() {
            cache.layer_outputs[num_layers - 1].clone()
        } else {
            last_output.clone()
        };

        let out_activation = self.layers[num_layers - 1].activation;
        let output_size = self.layers[num_layers - 1].biases.len();
        let mut delta = vec![0.0; output_size];
        for i in 0..output_size {
            let z = if i < output_pre_act.len() {
                output_pre_act[i]
            } else {
                0.0
            };
            let o = if i < last_output.len() {
                last_output[i]
            } else {
                0.0
            };
            let t = if i < target.len() { target[i] } else { 0.0 };
            delta[i] = (o - t) * Self::activation_derivative(z, out_activation);
        }

        // Pre-allocate gradient storage
        for layer in &self.layers {
            let n_out = layer.biases.len();
            let mut wg = Vec::with_capacity(n_out);
            for neuron_w in &layer.weights {
                wg.push(vec![0.0; neuron_w.len()]);
            }
            weight_gradients.push(wg);
            bias_gradients.push(vec![0.0; n_out]);
        }

        // Backpropagate through layers in reverse
        for l in (0..num_layers).rev() {
            let layer = &self.layers[l];
            let layer_in = &layer_inputs[l];

            for j in 0..layer.biases.len() {
                if j < delta.len() {
                    bias_gradients[l][j] = delta[j];
                    for k in 0..layer.weights[j].len() {
                        let inp_val = if k < layer_in.len() { layer_in[k] } else { 0.0 };
                        weight_gradients[l][j][k] = delta[j] * inp_val;
                    }
                }
            }

            if l > 0 {
                let prev_layer = &self.layers[l - 1];
                let prev_pre_act = if l - 1 < cache.layer_outputs.len() {
                    &cache.layer_outputs[l - 1]
                } else {
                    &layer_inputs[l]
                };
                let prev_activation = prev_layer.activation;
                let prev_size = prev_layer.biases.len();

                let mut new_delta = vec![0.0; prev_size];
                for k in 0..prev_size {
                    let mut sum = 0.0;
                    for j in 0..layer.biases.len() {
                        if j < delta.len() && k < layer.weights[j].len() {
                            sum += delta[j] * layer.weights[j][k];
                        }
                    }
                    let z = if k < prev_pre_act.len() {
                        prev_pre_act[k]
                    } else {
                        0.0
                    };
                    new_delta[k] = sum * Self::activation_derivative(z, prev_activation);
                }
                delta = new_delta;
            }
        }

        NetworkGradients {
            weight_gradients,
            bias_gradients,
        }
    }

    /// Get network parameters for serialization
    pub fn get_parameters(&self) -> HashMap<String, Vec<f64>> {
        let mut params = HashMap::new();

        for (i, layer) in self.layers.iter().enumerate() {
            // Flatten weights
            let mut weights = Vec::new();
            for neuron_weights in &layer.weights {
                weights.extend(neuron_weights.iter());
            }
            params.insert(format!("layer_{}_weights", i), weights);
            params.insert(format!("layer_{}_biases", i), layer.biases.clone());
        }

        params
    }

    /// Set network parameters from serialized data
    pub fn set_parameters(&mut self, params: &HashMap<String, Vec<f64>>) {
        for (i, layer) in self.layers.iter_mut().enumerate() {
            if let Some(weights) = params.get(&format!("layer_{}_weights", i)) {
                let mut weight_idx = 0;
                for neuron_weights in &mut layer.weights {
                    for weight in neuron_weights {
                        if weight_idx < weights.len() {
                            *weight = weights[weight_idx];
                            weight_idx += 1;
                        }
                    }
                }
            }

            if let Some(biases) = params.get(&format!("layer_{}_biases", i)) {
                for (j, bias) in layer.biases.iter_mut().enumerate() {
                    if j < biases.len() {
                        *bias = biases[j];
                    }
                }
            }
        }
    }
}

impl AttentionHead {
    /// Create a new attention head
    pub fn new(model_dim: usize) -> Self {
        let head_dim = model_dim / 8; // Typical head dimension

        Self {
            query_weights: NeuralNetwork::initialize_weights(model_dim, head_dim),
            key_weights: NeuralNetwork::initialize_weights(model_dim, head_dim),
            value_weights: NeuralNetwork::initialize_weights(model_dim, head_dim),
            output_weights: NeuralNetwork::initialize_weights(head_dim, model_dim),
            head_dim,
        }
    }

    /// Forward pass through attention head
    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        self.forward_with_score(input).0
    }

    /// Forward pass through attention head, also returning the real,
    /// input-dependent attention weight computed for this call. Callers that
    /// need a genuine attention measurement (e.g. reported processor
    /// statistics) should use this instead of fabricating a constant.
    pub fn forward_with_score(&self, input: &[f64]) -> (Vec<f64>, f64) {
        // Simplified attention mechanism
        let query = self.linear_transform(input, &self.query_weights);
        let key = self.linear_transform(input, &self.key_weights);
        let value = self.linear_transform(input, &self.value_weights);

        // Compute attention scores (simplified)
        let attention_score = self.dot_product(&query, &key) / (self.head_dim as f64).sqrt();
        let attention_weight = (attention_score).exp() / (1.0 + (attention_score).exp());

        // Apply attention to values
        let mut attended_value = value;
        for val in &mut attended_value {
            *val *= attention_weight;
        }

        // Output projection
        (
            self.linear_transform(&attended_value, &self.output_weights),
            attention_weight,
        )
    }

    /// Linear transformation
    fn linear_transform(&self, input: &[f64], weights: &[Vec<f64>]) -> Vec<f64> {
        let mut output = vec![0.0; weights.len()];

        for (i, neuron_weights) in weights.iter().enumerate() {
            let mut sum = 0.0;
            for (j, &input_val) in input.iter().enumerate() {
                if j < neuron_weights.len() {
                    sum += neuron_weights[j] * input_val;
                }
            }
            output[i] = sum;
        }

        output
    }

    /// Dot product of two vectors
    fn dot_product(&self, a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }
}

impl LayerNorm {
    /// Create a new layer normalization
    pub fn new(size: usize) -> Self {
        Self {
            gamma: vec![1.0; size],
            beta: vec![0.0; size],
            eps: 1e-5,
        }
    }

    /// Normalize input
    pub fn normalize(&self, input: &[f64]) -> Vec<f64> {
        let mean = input.iter().sum::<f64>() / input.len() as f64;
        let variance = input.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / input.len() as f64;
        let std_dev = (variance + self.eps).sqrt();

        input
            .iter()
            .zip(&self.gamma)
            .zip(&self.beta)
            .map(|((x, gamma), beta)| gamma * ((x - mean) / std_dev) + beta)
            .collect()
    }
}
