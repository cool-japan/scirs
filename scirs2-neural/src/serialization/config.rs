//! Configuration structs for model serialization
//!
//! This module provides configuration structs for serializing and deserializing
//! neural network model architectures.

use serde::{Deserialize, Serialize};

/// Model configuration containing layer configurations
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ModelConfig {
    /// Layer configurations
    pub layers: Vec<LayerConfig>,
}

/// Layer configuration enum for different layer types
#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum LayerConfig {
    /// Dense (fully connected) layer configuration
    Dense(DenseConfig),
    /// Conv2D layer configuration
    Conv2D(Conv2DConfig),
    /// LayerNorm layer configuration
    LayerNorm(LayerNormConfig),
    /// BatchNorm layer configuration
    BatchNorm(BatchNormConfig),
    /// Dropout layer configuration
    Dropout(DropoutConfig),
    /// MaxPool2D layer configuration
    MaxPool2D(MaxPool2DConfig),
}

/// Dense layer configuration
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct DenseConfig {
    /// Input dimension
    pub input_dim: usize,
    /// Output dimension
    pub output_dim: usize,
    /// Activation function name (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub activation: Option<String>,
}

/// Conv2D layer configuration
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Conv2DConfig {
    /// Number of input channels
    pub in_channels: usize,
    /// Number of output channels
    pub out_channels: usize,
    /// Kernel size
    pub kernel_size: usize,
    /// Stride
    pub stride: usize,
    /// Padding mode ("Same" or "Valid")
    pub padding_mode: String,
}

/// LayerNorm layer configuration
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct LayerNormConfig {
    /// Normalized shape
    pub normalized_shape: usize,
    /// Epsilon for numerical stability
    pub eps: f64,
}

/// BatchNorm layer configuration
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct BatchNormConfig {
    /// Number of features to normalize
    pub num_features: usize,
    /// Momentum for running statistics
    pub momentum: f64,
    /// Epsilon for numerical stability
    pub eps: f64,
}

/// Dropout layer configuration
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct DropoutConfig {
    /// Dropout probability
    pub p: f64,
}

/// MaxPool2D layer configuration
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct MaxPool2DConfig {
    /// Kernel size
    pub kernel_size: usize,
    /// Stride
    pub stride: usize,
    /// Padding
    pub padding: usize,
}
