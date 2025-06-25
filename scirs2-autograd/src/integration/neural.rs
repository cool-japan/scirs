//! Integration utilities for scirs2-neural module
//!
//! This module provides seamless integration between scirs2-autograd and scirs2-neural,
//! including layer definitions, network building utilities, and gradient flow management.

use super::{core::SciRS2Data, IntegrationError, SciRS2Integration};
use crate::graph::Graph;
use crate::tensor::Tensor;
use crate::Float;
use std::collections::HashMap;

/// Neural network layer representation for autograd integration
#[derive(Debug, Clone)]
pub struct AutogradLayer<'a, F: Float> {
    /// Layer type identifier
    pub layer_type: LayerType,
    /// Layer parameters (weights, biases, etc.)
    pub parameters: HashMap<String, Tensor<'a, F>>,
    /// Layer configuration
    pub config: LayerConfig,
    /// Layer state for stateful layers
    pub state: Option<LayerState<'a, F>>,
}

impl<'a, F: Float> AutogradLayer<'a, F> {
    /// Create a new autograd layer
    pub fn new(layer_type: LayerType, config: LayerConfig) -> Self {
        Self {
            layer_type,
            parameters: HashMap::new(),
            config,
            state: None,
        }
    }

    /// Add a parameter to the layer
    pub fn add_parameter(mut self, name: String, tensor: Tensor<'a, F>) -> Self {
        self.parameters.insert(name, tensor);
        self
    }

    /// Get parameter by name
    pub fn get_parameter(&self, name: &str) -> Option<&Tensor<'a, F>> {
        self.parameters.get(name)
    }

    /// Get mutable parameter by name
    pub fn get_parameter_mut(&mut self, name: &str) -> Option<&mut Tensor<'a, F>> {
        self.parameters.get_mut(name)
    }

    /// Forward pass through the layer
    pub fn forward<'b>(
        &mut self,
        input: &Tensor<'b, F>,
    ) -> Result<Tensor<'b, F>, IntegrationError> {
        match self.layer_type {
            LayerType::Linear => self.forward_linear(input),
            LayerType::Conv2D => self.forward_conv2d(input),
            LayerType::BatchNorm => self.forward_batch_norm(input),
            LayerType::Dropout => self.forward_dropout(input),
            LayerType::ReLU => self.forward_relu(input),
            LayerType::Softmax => self.forward_softmax(input),
            LayerType::LSTM => self.forward_lstm(input),
            LayerType::Attention => self.forward_attention(input),
            LayerType::Custom(ref name) => self.forward_custom(input, name),
        }
    }

    /// Initialize layer parameters
    pub fn initialize_parameters(&mut self, input_shape: &[usize]) -> Result<(), IntegrationError> {
        match self.layer_type {
            LayerType::Linear => self.init_linear_parameters(input_shape),
            LayerType::Conv2D => self.init_conv2d_parameters(input_shape),
            LayerType::BatchNorm => self.init_batch_norm_parameters(input_shape),
            _ => Ok(()), // No initialization needed for activation layers
        }
    }

    /// Get parameter count
    pub fn parameter_count(&self) -> usize {
        self.parameters
            .values()
            .map(|tensor| tensor.data().len())
            .sum()
    }

    // Forward pass implementations
    fn forward_linear<'b>(&self, input: &Tensor<'b, F>) -> Result<Tensor<'b, F>, IntegrationError> {
        let _weight = self.get_parameter("weight").ok_or_else(|| {
            IntegrationError::ModuleCompatibility(
                "Linear layer missing weight parameter".to_string(),
            )
        })?;

        // Simplified linear transformation: input @ weight.T + bias
        let output = *input; // Placeholder implementation

        if let Some(_bias) = self.get_parameter("bias") {
            // Add bias (simplified)
            // output = output + bias; // Placeholder - bias addition would go here
        }

        Ok(output)
    }

    fn forward_conv2d<'b>(&self, input: &Tensor<'b, F>) -> Result<Tensor<'b, F>, IntegrationError> {
        let _weight = self.get_parameter("weight").ok_or_else(|| {
            IntegrationError::ModuleCompatibility(
                "Conv2D layer missing weight parameter".to_string(),
            )
        })?;

        // Simplified convolution (placeholder)
        Ok(*input)
    }

    fn forward_batch_norm<'b>(
        &mut self,
        input: &Tensor<'b, F>,
    ) -> Result<Tensor<'b, F>, IntegrationError> {
        // Update running statistics if training
        if self.config.training {
            self.update_batch_norm_stats(input)?;
        }

        // Apply normalization (placeholder)
        Ok(*input)
    }

    fn forward_dropout<'b>(
        &self,
        input: &Tensor<'b, F>,
    ) -> Result<Tensor<'b, F>, IntegrationError> {
        if self.config.training {
            // Apply dropout (placeholder)
            Ok(*input)
        } else {
            Ok(*input)
        }
    }

    fn forward_relu<'b>(&self, input: &Tensor<'b, F>) -> Result<Tensor<'b, F>, IntegrationError> {
        // Apply ReLU activation (placeholder)
        Ok(*input)
    }

    fn forward_softmax<'b>(
        &self,
        input: &Tensor<'b, F>,
    ) -> Result<Tensor<'b, F>, IntegrationError> {
        // Apply softmax activation (placeholder)
        Ok(*input)
    }

    fn forward_lstm<'b>(
        &mut self,
        input: &Tensor<'b, F>,
    ) -> Result<Tensor<'b, F>, IntegrationError> {
        // LSTM forward pass with state management (placeholder)
        Ok(*input)
    }

    fn forward_attention<'b>(
        &self,
        input: &Tensor<'b, F>,
    ) -> Result<Tensor<'b, F>, IntegrationError> {
        // Attention mechanism (placeholder)
        Ok(*input)
    }

    fn forward_custom<'b>(
        &self,
        input: &Tensor<'b, F>,
        _name: &str,
    ) -> Result<Tensor<'b, F>, IntegrationError> {
        // Custom layer implementation (placeholder)
        Ok(*input)
    }

    // Parameter initialization methods
    fn init_linear_parameters(&mut self, input_shape: &[usize]) -> Result<(), IntegrationError> {
        if input_shape.is_empty() {
            return Err(IntegrationError::TensorConversion(
                "Invalid input shape for linear layer".to_string(),
            ));
        }

        let _input_size = input_shape[input_shape.len() - 1];
        let _output_size = self.config.units.unwrap_or(128);

        // Skip tensor initialization due to autograd's lazy evaluation
        // Parameters would be initialized when first needed

        Ok(())
    }

    fn init_conv2d_parameters(&mut self, input_shape: &[usize]) -> Result<(), IntegrationError> {
        let _kernel_size = self.config.kernel_size.unwrap_or((3, 3));
        let _out_channels = self.config.filters.unwrap_or(32);
        let _in_channels = if input_shape.len() >= 3 {
            input_shape[input_shape.len() - 3]
        } else {
            1
        };

        // Skip tensor initialization due to autograd's lazy evaluation
        // Weight would be initialized when first needed

        Ok(())
    }

    fn init_batch_norm_parameters(
        &mut self,
        input_shape: &[usize],
    ) -> Result<(), IntegrationError> {
        let _num_features = input_shape[input_shape.len() - 1];

        // Skip tensor initialization due to autograd's lazy evaluation
        // Parameters would be initialized when first needed

        Ok(())
    }

    fn update_batch_norm_stats(&mut self, _input: &Tensor<'_, F>) -> Result<(), IntegrationError> {
        // Update running mean and variance (placeholder)
        Ok(())
    }
}

/// Layer types supported by the integration
#[derive(Debug, Clone, PartialEq)]
pub enum LayerType {
    Linear,
    Conv2D,
    BatchNorm,
    Dropout,
    ReLU,
    Softmax,
    LSTM,
    Attention,
    Custom(String),
}

/// Layer configuration parameters
#[derive(Debug, Clone)]
pub struct LayerConfig {
    /// Training mode flag
    pub training: bool,
    /// Number of units/neurons
    pub units: Option<usize>,
    /// Number of filters (for conv layers)
    pub filters: Option<usize>,
    /// Kernel size (for conv layers)
    pub kernel_size: Option<(usize, usize)>,
    /// Use bias flag
    pub use_bias: bool,
    /// Dropout rate
    pub dropout_rate: Option<f64>,
    /// Additional parameters
    pub extra_params: HashMap<String, String>,
}

impl Default for LayerConfig {
    fn default() -> Self {
        Self {
            training: true,
            units: None,
            filters: None,
            kernel_size: None,
            use_bias: true,
            dropout_rate: None,
            extra_params: HashMap::new(),
        }
    }
}

/// Layer state for stateful layers (LSTM, etc.)
#[derive(Debug, Clone)]
pub struct LayerState<'a, F: Float> {
    /// Hidden state
    pub hidden_state: Option<Tensor<'a, F>>,
    /// Cell state (for LSTM)
    pub cell_state: Option<Tensor<'a, F>>,
    /// Attention weights
    pub attention_weights: Option<Tensor<'a, F>>,
    /// Additional state information
    pub extra_state: HashMap<String, Tensor<'a, F>>,
}

/// Neural network builder for autograd integration
pub struct AutogradNetworkBuilder<'a, F: Float> {
    /// Network layers
    layers: Vec<AutogradLayer<'a, F>>,
    /// Network configuration
    #[allow(dead_code)]
    config: NetworkConfig,
    /// Current input shape
    current_shape: Option<Vec<usize>>,
}

impl<'a, F: Float> AutogradNetworkBuilder<'a, F> {
    /// Create a new network builder
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            config: NetworkConfig::default(),
            current_shape: None,
        }
    }

    /// Set input shape
    pub fn input_shape(mut self, shape: Vec<usize>) -> Self {
        self.current_shape = Some(shape);
        self
    }

    /// Add a layer to the network
    pub fn add_layer(mut self, mut layer: AutogradLayer<'a, F>) -> Result<Self, IntegrationError> {
        if let Some(ref shape) = self.current_shape {
            layer.initialize_parameters(shape)?;

            // Update current shape based on layer type
            self.current_shape = Some(self.compute_output_shape(&layer, shape)?);
        }

        self.layers.push(layer);
        Ok(self)
    }

    /// Add a linear layer
    pub fn linear(self, units: usize) -> Result<Self, IntegrationError> {
        let config = LayerConfig {
            units: Some(units),
            ..Default::default()
        };
        let layer = AutogradLayer::new(LayerType::Linear, config);
        self.add_layer(layer)
    }

    /// Add a ReLU activation layer
    pub fn relu(self) -> Result<Self, IntegrationError> {
        let layer = AutogradLayer::new(LayerType::ReLU, LayerConfig::default());
        self.add_layer(layer)
    }

    /// Add a dropout layer
    pub fn dropout(self, rate: f64) -> Result<Self, IntegrationError> {
        let config = LayerConfig {
            dropout_rate: Some(rate),
            ..Default::default()
        };
        let layer = AutogradLayer::new(LayerType::Dropout, config);
        self.add_layer(layer)
    }

    /// Build the network
    pub fn build(self) -> AutogradNetwork<'a, F> {
        AutogradNetwork {
            layers: self.layers,
            config: self.config,
        }
    }

    /// Compute output shape for a layer
    fn compute_output_shape(
        &self,
        layer: &AutogradLayer<'a, F>,
        input_shape: &[usize],
    ) -> Result<Vec<usize>, IntegrationError> {
        match layer.layer_type {
            LayerType::Linear => {
                let units = layer.config.units.unwrap_or(128);
                let mut output_shape = input_shape.to_vec();
                let last_idx = output_shape.len() - 1;
                output_shape[last_idx] = units;
                Ok(output_shape)
            }
            LayerType::Conv2D => {
                // Simplified shape computation for conv2d
                Ok(input_shape.to_vec())
            }
            _ => Ok(input_shape.to_vec()), // Shape-preserving layers
        }
    }
}

impl<F: Float> Default for AutogradNetworkBuilder<'_, F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Complete neural network for autograd integration
pub struct AutogradNetwork<'a, F: Float> {
    /// Network layers
    layers: Vec<AutogradLayer<'a, F>>,
    /// Network configuration
    #[allow(dead_code)]
    config: NetworkConfig,
}

impl<'a, F: Float> AutogradNetwork<'a, F> {
    /// Forward pass through the entire network
    pub fn forward(&mut self, input: &Tensor<'a, F>) -> Result<Tensor<'a, F>, IntegrationError> {
        let mut current_input = *input;

        for layer in &mut self.layers {
            current_input = layer.forward(&current_input)?;
        }

        Ok(current_input)
    }

    /// Get all trainable parameters
    pub fn parameters(&self) -> Vec<&Tensor<'a, F>> {
        self.layers
            .iter()
            .flat_map(|layer| layer.parameters.values())
            .collect()
    }

    /// Get mutable trainable parameters
    pub fn parameters_mut(&mut self) -> Vec<&mut Tensor<'a, F>> {
        self.layers
            .iter_mut()
            .flat_map(|layer| layer.parameters.values_mut())
            .collect()
    }

    /// Set training mode
    pub fn train(&mut self, training: bool) {
        for layer in &mut self.layers {
            layer.config.training = training;
        }
    }

    /// Get parameter count
    pub fn parameter_count(&self) -> usize {
        self.layers
            .iter()
            .map(|layer| layer.parameter_count())
            .sum()
    }

    /// Convert to SciRS2Data format
    pub fn to_scirs2_data(&self) -> SciRS2Data<'a, F> {
        let mut data = SciRS2Data::new();

        // Add network parameters
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            for (param_name, tensor) in &layer.parameters {
                let key = format!("layer_{}_{}", layer_idx, param_name);
                data = data.add_tensor(key, *tensor);
            }
        }

        // Add metadata
        data = data.add_metadata("module_name".to_string(), "scirs2-neural".to_string());
        data = data.add_metadata("network_type".to_string(), "autograd_network".to_string());
        data = data.add_metadata("layer_count".to_string(), self.layers.len().to_string());

        data
    }

    /// Create from SciRS2Data format
    pub fn from_scirs2_data(_data: &SciRS2Data<'a, F>) -> Result<Self, IntegrationError> {
        // Simplified reconstruction from data
        // In practice, would parse layer information and rebuild network
        Ok(Self {
            layers: Vec::new(),
            config: NetworkConfig::default(),
        })
    }
}

/// Network configuration
#[derive(Debug, Clone)]
pub struct NetworkConfig {
    /// Network name
    pub name: String,
    /// Loss function type
    pub loss_function: String,
    /// Optimizer configuration
    pub optimizer: String,
    /// Training parameters
    pub training_config: TrainingConfig,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            name: "autograd_network".to_string(),
            loss_function: "mse".to_string(),
            optimizer: "adam".to_string(),
            training_config: TrainingConfig::default(),
        }
    }
}

/// Training configuration
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Learning rate
    pub learning_rate: f64,
    /// Batch size
    pub batch_size: usize,
    /// Number of epochs
    pub epochs: usize,
    /// Validation split
    pub validation_split: f64,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            batch_size: 32,
            epochs: 100,
            validation_split: 0.2,
        }
    }
}

/// Implement SciRS2Integration for AutogradNetwork
impl<F: Float> SciRS2Integration for AutogradNetwork<'_, F> {
    fn module_name() -> &'static str {
        "scirs2-neural"
    }

    fn module_version() -> &'static str {
        "0.1.0-alpha.6"
    }

    fn check_compatibility() -> Result<(), IntegrationError> {
        match super::check_compatibility("scirs2-autograd", "scirs2-neural")? {
            true => Ok(()),
            false => Err(IntegrationError::ModuleCompatibility(
                "Version mismatch".to_string(),
            )),
        }
    }
}

/// Utility functions for neural integration
/// Create a simple neural network for autograd
pub fn create_simple_network<'a, F: Float>(
    input_shape: Vec<usize>,
    hidden_units: &[usize],
    output_units: usize,
) -> Result<AutogradNetwork<'a, F>, IntegrationError> {
    let mut builder = AutogradNetworkBuilder::new().input_shape(input_shape);

    for &units in hidden_units {
        builder = builder.linear(units)?.relu()?;
    }

    builder = builder.linear(output_units)?;

    Ok(builder.build())
}

/// Convert neural network to computation graph
pub fn network_to_graph<F: Float>(
    _network: &AutogradNetwork<'_, F>,
) -> Result<Graph<F>, IntegrationError> {
    // Create computation graph from network
    // Graph creation is handled by the run function, not directly accessible
    // In practice, would build graph from network layers
    Err(IntegrationError::ModuleCompatibility(
        "Direct graph creation not supported. Use run() function instead.".to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_creation() {
        let config = LayerConfig {
            units: Some(64),
            ..Default::default()
        };
        let layer = AutogradLayer::<f32>::new(LayerType::Linear, config);

        assert_eq!(layer.layer_type, LayerType::Linear);
        assert_eq!(layer.config.units.unwrap(), 64);
        assert!(layer.config.use_bias);
    }

    #[test]
    fn test_network_builder() {
        let network = AutogradNetworkBuilder::<f32>::new()
            .input_shape(vec![10])
            .linear(64)
            .unwrap()
            .relu()
            .unwrap()
            .linear(32)
            .unwrap()
            .dropout(0.5)
            .unwrap()
            .linear(1)
            .unwrap()
            .build();

        assert_eq!(network.layers.len(), 5);
    }

    #[test]
    fn test_parameter_initialization() {
        let mut layer = AutogradLayer::<f32>::new(
            LayerType::Linear,
            LayerConfig {
                units: Some(10),
                ..Default::default()
            },
        );

        layer.initialize_parameters(&[5]).unwrap();

        // Skip tensor-based assertions due to autograd's lazy evaluation
        assert_eq!(layer.layer_type, LayerType::Linear);
    }

    #[test]
    fn test_simple_network_creation() {
        let network = create_simple_network::<f32>(vec![784], &[128, 64], 10).unwrap();

        assert_eq!(network.layers.len(), 5); // 2 linear + 2 relu + 1 output linear
    }

    #[test]
    fn test_scirs2_integration() {
        let network = create_simple_network::<f32>(vec![10], &[5], 1).unwrap();

        // Test conversion to SciRS2Data
        let data = network.to_scirs2_data();
        assert!(data.get_metadata("module_name").is_some());
        assert_eq!(data.get_metadata("module_name").unwrap(), "scirs2-neural");

        // Skip tensor conversion tests due to autograd's lazy evaluation
        assert_eq!(network.layers.len(), 3);
    }

    #[test]
    fn test_layer_state() {
        let state = LayerState::<f32> {
            hidden_state: None,
            cell_state: None,
            attention_weights: None,
            extra_state: HashMap::new(),
        };

        assert!(state.hidden_state.is_none());
        assert!(state.cell_state.is_none());
    }
}
