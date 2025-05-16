//! Module for model serialization and deserialization

use crate::activations::*;
use crate::error::{NeuralError, Result};
use crate::layers::*;
use crate::models::sequential::Sequential;
use ndarray::{Array, ScalarOperand};
use num_traits::Float;
use num_traits::ToPrimitive;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::Debug;
use std::fs;
use std::path::Path;

/// Model serialization format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SerializationFormat {
    /// JSON serialization format
    JSON,
    /// CBOR serialization format
    CBOR,
    /// MessagePack serialization format
    MessagePack,
}

/// Layer type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LayerType {
    /// Dense (fully connected) layer
    Dense,
    /// Convolutional 2D layer
    Conv2D,
    /// Layer normalization
    LayerNorm,
    /// Batch normalization
    BatchNorm,
    /// Dropout layer
    Dropout,
    /// Max pooling 2D layer
    MaxPool2D,
}

/// Layer configuration for serialization
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum LayerConfig {
    /// Dense layer configuration
    #[serde(rename = "Dense")]
    Dense(DenseConfig),
    /// Conv2D layer configuration
    #[serde(rename = "Conv2D")]
    Conv2D(Conv2DConfig),
    /// LayerNorm layer configuration
    #[serde(rename = "LayerNorm")]
    LayerNorm(LayerNormConfig),
    /// BatchNorm layer configuration
    #[serde(rename = "BatchNorm")]
    BatchNorm(BatchNormConfig),
    /// Dropout layer configuration
    #[serde(rename = "Dropout")]
    Dropout(DropoutConfig),
    /// MaxPool2D layer configuration
    #[serde(rename = "MaxPool2D")]
    MaxPool2D(MaxPool2DConfig),
}

/// Dense layer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DenseConfig {
    /// Input dimension
    pub input_dim: usize,
    /// Output dimension
    pub output_dim: usize,
    /// Activation function name
    pub activation: Option<String>,
}

/// Conv2D layer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conv2DConfig {
    /// Number of input channels
    pub in_channels: usize,
    /// Number of output channels
    pub out_channels: usize,
    /// Kernel size (square)
    pub kernel_size: usize,
    /// Stride
    pub stride: usize,
    /// Padding mode
    pub padding_mode: String,
}

/// LayerNorm layer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerNormConfig {
    /// Normalized shape
    pub normalized_shape: usize,
    /// Epsilon for numerical stability
    pub eps: f64,
}

/// BatchNorm layer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchNormConfig {
    /// Number of features
    pub num_features: usize,
    /// Momentum
    pub momentum: f64,
    /// Epsilon for numerical stability
    pub eps: f64,
}

/// Dropout layer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DropoutConfig {
    /// Dropout probability
    pub p: f64,
}

/// MaxPool2D layer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaxPool2DConfig {
    /// Kernel size
    pub kernel_size: (usize, usize),
    /// Stride
    pub stride: (usize, usize),
    /// Padding
    pub padding: Option<(usize, usize)>,
}

/// Serialized model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedModel {
    /// Model name
    pub name: String,
    /// Model version
    pub version: String,
    /// Model layers configuration
    pub layers: Vec<LayerConfig>,
    /// Model parameters (weights and biases)
    pub parameters: Vec<Vec<Vec<f64>>>,
}

/// Save model to file
pub fn save_model<F: Float + Debug + ScalarOperand + Send + Sync + 'static, P: AsRef<Path>>(
    model: &Sequential<F>,
    path: P,
    format: SerializationFormat,
) -> Result<()> {
    let serialized = serialize_model(model)?;

    let bytes = match format {
        SerializationFormat::JSON => serde_json::to_vec_pretty(&serialized)
            .map_err(|e| NeuralError::SerializationError(e.to_string()))?,
        SerializationFormat::CBOR => {
            let mut buf = Vec::new();
            let mut serializer = serde_cbor::Serializer::new(&mut buf);
            serialized
                .serialize(&mut serializer)
                .map_err(|e| NeuralError::SerializationError(e.to_string()))?;
            buf
        }
        SerializationFormat::MessagePack => {
            let mut buf = Vec::new();
            let mut serializer = rmp_serde::Serializer::new(&mut buf);
            serialized
                .serialize(&mut serializer)
                .map_err(|e| NeuralError::SerializationError(e.to_string()))?;
            buf
        }
    };

    fs::write(path, bytes).map_err(|e| NeuralError::IOError(e.to_string()))?;
    Ok(())
}

/// Load model from file
pub fn load_model<F: Float + Debug + ScalarOperand + Send + Sync + 'static, P: AsRef<Path>>(
    path: P,
    format: SerializationFormat,
) -> Result<Sequential<F>> {
    let bytes = fs::read(path).map_err(|e| NeuralError::IOError(e.to_string()))?;
    let serialized: SerializedModel = match format {
        SerializationFormat::JSON => serde_json::from_slice(&bytes)
            .map_err(|e| NeuralError::SerializationError(e.to_string()))?,
        SerializationFormat::CBOR => serde_cbor::from_slice(&bytes)
            .map_err(|e| NeuralError::SerializationError(e.to_string()))?,
        SerializationFormat::MessagePack => rmp_serde::from_slice(&bytes)
            .map_err(|e| NeuralError::SerializationError(e.to_string()))?,
    };

    deserialize_model(&serialized)
}

/// Serialize model to SerializedModel
fn serialize_model<F: Float + Debug + ScalarOperand + Send + Sync + 'static>(
    model: &Sequential<F>,
) -> Result<SerializedModel> {
    let mut layers = Vec::new();
    let mut parameters = Vec::new();

    for layer in model.layers() {
        if let Some(dense) = layer.as_any().downcast_ref::<Dense<F>>() {
            let config = LayerConfig::Dense(DenseConfig {
                input_dim: dense.input_dim(),
                output_dim: dense.output_dim(),
                activation: dense.activation_name().map(|s| s.to_string()),
            });
            layers.push(config);

            // Get parameters (weights and biases)
            let layer_params = dense.get_parameters();

            // The Dense layer stores weights with shape [input_dim, output_dim]
            // We maintain the same shape when serializing

            // Extract parameters - weights and biases
            let params = extract_parameters(layer_params)?;
            parameters.push(params);
        } else if let Some(conv) = layer.as_any().downcast_ref::<Conv2D<F>>() {
            let config = LayerConfig::Conv2D(Conv2DConfig {
                in_channels: conv.in_channels(),
                out_channels: conv.out_channels(),
                kernel_size: conv.kernel_size(),
                stride: conv.stride(),
                padding_mode: match conv.padding_mode() {
                    PaddingMode::Same => "Same".to_string(),
                    PaddingMode::Valid => "Valid".to_string(),
                    PaddingMode::Custom(size) => format!("Custom({})", size),
                },
            });
            layers.push(config);

            // Extract parameters
            let params = extract_parameters(conv.get_parameters())?;
            parameters.push(params);
        } else if let Some(ln) = layer.as_any().downcast_ref::<LayerNorm<F>>() {
            let config = LayerConfig::LayerNorm(LayerNormConfig {
                normalized_shape: ln.normalized_shape(),
                eps: ln.eps().to_f64().unwrap(),
            });
            layers.push(config);

            // Extract parameters
            let params = extract_parameters(ln.get_parameters())?;
            parameters.push(params);
        } else if let Some(bn) = layer.as_any().downcast_ref::<BatchNorm<F>>() {
            let config = LayerConfig::BatchNorm(BatchNormConfig {
                num_features: bn.num_features(),
                momentum: bn.momentum().to_f64().unwrap(),
                eps: bn.eps().to_f64().unwrap(),
            });
            layers.push(config);

            // Extract parameters
            let params = extract_parameters(bn.get_parameters())?;
            parameters.push(params);
        } else if let Some(dropout) = layer.as_any().downcast_ref::<Dropout<F>>() {
            let config = LayerConfig::Dropout(DropoutConfig {
                p: dropout.p().to_f64().unwrap(),
            });
            layers.push(config);

            // Dropout has no parameters
            parameters.push(Vec::new());
        } else if let Some(maxpool) = layer.as_any().downcast_ref::<MaxPool2D<F>>() {
            let config = LayerConfig::MaxPool2D(MaxPool2DConfig {
                kernel_size: (maxpool.kernel_size(), maxpool.kernel_size()),
                stride: (maxpool.stride(), maxpool.stride()),
                padding: Some((maxpool.padding(), maxpool.padding())),
            });
            layers.push(config);

            // MaxPool2D has no parameters
            parameters.push(Vec::new());
        } else {
            return Err(NeuralError::SerializationError(
                "Unsupported layer type".to_string(),
            ));
        }
    }

    Ok(SerializedModel {
        name: "SciRS2 Sequential Model".to_string(),
        version: "0.1.0".to_string(),
        layers,
        parameters,
    })
}

/// Extract parameters from layer
fn extract_parameters<F: Float + Debug + ScalarOperand + Send + Sync>(
    params: Vec<&Array<F, ndarray::IxDyn>>,
) -> Result<Vec<Vec<f64>>> {
    let mut result = Vec::new();

    for param in params.iter() {
        let f64_vec: Vec<f64> = param
            .iter()
            .map(|&x| {
                x.to_f64().ok_or_else(|| {
                    NeuralError::SerializationError("Cannot convert parameter to f64".to_string())
                })
            })
            .collect::<Result<Vec<f64>>>()?;

        result.push(f64_vec);
    }

    Ok(result)
}

/// Deserialize model from SerializedModel
fn deserialize_model<F: Float + Debug + ScalarOperand + Send + Sync + 'static>(
    serialized: &SerializedModel,
) -> Result<Sequential<F>> {
    let mut layers: Vec<Box<dyn Layer<F>>> = Vec::new();

    for (i, layer_config) in serialized.layers.iter().enumerate() {
        let params = if i < serialized.parameters.len() {
            &serialized.parameters[i]
        } else {
            &Vec::<Vec<f64>>::new()
        };

        match layer_config {
            LayerConfig::Dense(config) => {
                let layer = create_dense_layer::<F>(config, params)?;
                layers.push(Box::new(layer));
            }
            LayerConfig::Conv2D(config) => {
                let layer = create_conv2d_layer::<F>(config, params)?;
                layers.push(Box::new(layer));
            }
            LayerConfig::LayerNorm(config) => {
                let layer = create_layer_norm::<F>(config, params)?;
                layers.push(Box::new(layer));
            }
            LayerConfig::BatchNorm(config) => {
                let layer = create_batch_norm::<F>(config, params)?;
                layers.push(Box::new(layer));
            }
            LayerConfig::Dropout(config) => {
                let layer = create_dropout::<F>(config)?;
                layers.push(Box::new(layer));
            }
            LayerConfig::MaxPool2D(config) => {
                let layer = create_maxpool2d::<F>(config)?;
                layers.push(Box::new(layer));
            }
        }
    }

    // Convert layers to include Send + Sync bounds
    let mut bound_layers: Vec<Box<dyn Layer<F> + Send + Sync>> = Vec::new();

    for layer in layers {
        // Need to reboxing the layer with proper bounds
        let layer_ref = &*layer as &dyn Layer<F>;
        if let Some(dense) = layer_ref.as_any().downcast_ref::<Dense<F>>() {
            bound_layers.push(Box::new(dense.clone()));
        } else if let Some(conv) = layer_ref.as_any().downcast_ref::<Conv2D<F>>() {
            bound_layers.push(Box::new(conv.clone()));
        } else if let Some(bn) = layer_ref.as_any().downcast_ref::<BatchNorm<F>>() {
            // Create a new instance rather than cloning to avoid RefCell issues
            let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
            let new_bn = BatchNorm::new(
                bn.num_features(),
                bn.momentum().to_f64().unwrap(),
                bn.eps().to_f64().unwrap(),
                &mut rng,
            )?;
            bound_layers.push(Box::new(new_bn));
        } else if let Some(ln) = layer_ref.as_any().downcast_ref::<LayerNorm<F>>() {
            bound_layers.push(Box::new(ln.clone()));
        } else if let Some(dropout) = layer_ref.as_any().downcast_ref::<Dropout<F>>() {
            bound_layers.push(Box::new(dropout.clone()));
        } else if let Some(maxpool) = layer_ref.as_any().downcast_ref::<MaxPool2D<F>>() {
            // Create a new MaxPool2D instead of cloning
            let new_maxpool = MaxPool2D::new(
                (maxpool.kernel_size(), maxpool.kernel_size()),
                (maxpool.stride(), maxpool.stride()),
                Some((maxpool.padding(), maxpool.padding())),
            )?;
            bound_layers.push(Box::new(new_maxpool));
        } else {
            return Err(NeuralError::DeserializationError(format!(
                "Unsupported layer type for deserialization: {:?}",
                std::any::type_name::<F>()
            )));
        }
    }

    Ok(Sequential::from_layers(bound_layers))
}

/// Create a Dense layer from configuration and parameters
fn create_dense_layer<F: Float + Debug + ScalarOperand + Send + Sync + 'static>(
    config: &DenseConfig,
    params: &[Vec<f64>],
) -> Result<Dense<F>> {
    let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
    let mut layer = Dense::new(
        config.input_dim,
        config.output_dim,
        config.activation.as_deref(),
        &mut rng,
    )?;

    // Load parameters if available
    if params.len() >= 2 {
        // We need to calculate the expected weights shape based on the config
        let weights_shape = [config.input_dim, config.output_dim];
        let bias_shape = [config.output_dim];

        // Check if the array has the correct number of elements
        if params[0].len() == config.output_dim * config.input_dim {
            // We have the right number of elements, proceed with caution
            let weights_array = match array_from_vec::<F>(&params[0], &weights_shape) {
                Ok(arr) => arr,
                Err(_) => {
                    // If we get an error with the expected shape, try the transposed shape
                    let transposed_shape = [config.output_dim, config.input_dim];
                    let transposed_arr = array_from_vec::<F>(&params[0], &transposed_shape)?;
                    // Transpose the array to get the correct shape
                    transposed_arr.t().to_owned().into_dyn()
                }
            };

            let bias_array = array_from_vec::<F>(&params[1], &bias_shape)?;

            layer.set_parameters(vec![weights_array, bias_array])?;
        } else {
            return Err(NeuralError::DeserializationError(format!(
                "Weight vector length ({}) doesn't match expected shape size ({})",
                params[0].len(),
                config.input_dim * config.output_dim
            )));
        }
    }

    Ok(layer)
}

/// Create a Conv2D layer from configuration and parameters
fn create_conv2d_layer<F: Float + Debug + ScalarOperand + Send + Sync + 'static>(
    config: &Conv2DConfig,
    params: &[Vec<f64>],
) -> Result<Conv2D<F>> {
    let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
    let padding_mode = match config.padding_mode.as_str() {
        "Same" => crate::layers::PaddingMode::Same,
        "Valid" => crate::layers::PaddingMode::Valid,
        _ => {
            return Err(NeuralError::SerializationError(format!(
                "Unsupported padding mode: {}",
                config.padding_mode
            )))
        }
    };

    let kernel_size = (config.kernel_size, config.kernel_size);
    let stride = (config.stride, config.stride);

    let mut layer = Conv2D::new(
        config.in_channels,
        config.out_channels,
        kernel_size,
        stride,
        padding_mode,
        &mut rng,
    )?;

    // Load parameters if available
    if params.len() >= 2 {
        // Ensure the weight shape matches what Conv2D expects
        let weights_shape = [
            config.out_channels,
            config.in_channels,
            config.kernel_size,
            config.kernel_size,
        ];
        let bias_shape = [config.out_channels];

        let weights_array = array_from_vec::<F>(&params[0], &weights_shape)?;
        let bias_array = array_from_vec::<F>(&params[1], &bias_shape)?;

        layer.set_parameters(vec![weights_array, bias_array])?;
    }

    Ok(layer)
}

/// Create a LayerNorm layer from configuration and parameters
fn create_layer_norm<F: Float + Debug + ScalarOperand + Send + Sync + 'static>(
    config: &LayerNormConfig,
    params: &[Vec<f64>],
) -> Result<LayerNorm<F>> {
    let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
    let mut layer = LayerNorm::new(config.normalized_shape, config.eps, &mut rng)?;

    // Load parameters if available
    if params.len() >= 2 {
        let gamma_shape = [config.normalized_shape];
        let beta_shape = [config.normalized_shape];

        let gamma_array = array_from_vec::<F>(&params[0], &gamma_shape)?;
        let beta_array = array_from_vec::<F>(&params[1], &beta_shape)?;

        layer.set_parameters(vec![gamma_array, beta_array])?;
    }

    Ok(layer)
}

/// Create a BatchNorm layer from configuration and parameters
fn create_batch_norm<F: Float + Debug + ScalarOperand + Send + Sync + 'static>(
    config: &BatchNormConfig,
    params: &[Vec<f64>],
) -> Result<BatchNorm<F>> {
    let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
    let mut layer = BatchNorm::new(config.num_features, config.momentum, config.eps, &mut rng)?;

    // Load parameters if available
    if params.len() >= 2 {
        let gamma_shape = [config.num_features];
        let beta_shape = [config.num_features];

        let gamma_array = array_from_vec::<F>(&params[0], &gamma_shape)?;
        let beta_array = array_from_vec::<F>(&params[1], &beta_shape)?;

        layer.set_parameters(vec![gamma_array, beta_array])?;
    }

    Ok(layer)
}

/// Create a Dropout layer from configuration
fn create_dropout<F: Float + Debug + ScalarOperand + Send + Sync + 'static>(
    config: &DropoutConfig,
) -> Result<Dropout<F>> {
    let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
    Dropout::new(config.p, &mut rng)
}

/// Create a MaxPool2D layer from configuration
fn create_maxpool2d<F: Float + Debug + ScalarOperand + Send + Sync + 'static>(
    config: &MaxPool2DConfig,
) -> Result<MaxPool2D<F>> {
    MaxPool2D::new(config.kernel_size, config.stride, config.padding)
}

/// Convert a vector of f64 values to an ndarray with the given shape
fn array_from_vec<F: Float + Debug + ScalarOperand + Send + Sync + 'static>(
    vec: &[f64],
    shape: &[usize],
) -> Result<Array<F, ndarray::IxDyn>> {
    let shape_size: usize = shape.iter().product();
    if vec.len() != shape_size {
        return Err(NeuralError::SerializationError(format!(
            "Parameter vector length ({}) doesn't match expected shape size ({})",
            vec.len(),
            shape_size
        )));
    }

    let f_vec: Vec<F> = vec
        .iter()
        .map(|&x| {
            F::from(x).ok_or_else(|| {
                NeuralError::SerializationError(format!("Cannot convert {} to target type", x))
            })
        })
        .collect::<Result<Vec<F>>>()?;

    let shape_ix = ndarray::IxDyn(shape);
    Ok(Array::from_shape_vec(shape_ix, f_vec)?)
}

/// Serializable activation function
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActivationFunction {
    /// ReLU activation
    ReLU,
    /// Sigmoid activation
    Sigmoid,
    /// Tanh activation
    Tanh,
    /// Softmax activation
    Softmax,
    /// LeakyReLU activation
    LeakyReLU(f64),
    /// ELU activation
    ELU(f64),
    /// GELU activation
    GELU,
    /// Swish activation
    Swish,
    /// Mish activation
    Mish,
}

impl ActivationFunction {
    /// Convert activation function name to ActivationFunction enum
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "relu" | "ReLU" => Some(ActivationFunction::ReLU),
            "sigmoid" | "Sigmoid" => Some(ActivationFunction::Sigmoid),
            "tanh" | "Tanh" => Some(ActivationFunction::Tanh),
            "softmax" | "Softmax" => Some(ActivationFunction::Softmax),
            "gelu" | "GELU" => Some(ActivationFunction::GELU),
            "swish" | "Swish" => Some(ActivationFunction::Swish),
            "mish" | "Mish" => Some(ActivationFunction::Mish),
            _ => {
                if name.starts_with("leaky_relu") || name.starts_with("LeakyReLU") {
                    // Extract alpha value
                    let parts: Vec<&str> = name.split('(').collect();
                    if parts.len() == 2 {
                        let alpha_str = parts[1].trim_end_matches(')');
                        if let Ok(alpha) = alpha_str.parse::<f64>() {
                            return Some(ActivationFunction::LeakyReLU(alpha));
                        }
                    }
                    Some(ActivationFunction::LeakyReLU(0.01)) // Default alpha
                } else if name.starts_with("elu") || name.starts_with("ELU") {
                    // Extract alpha value
                    let parts: Vec<&str> = name.split('(').collect();
                    if parts.len() == 2 {
                        let alpha_str = parts[1].trim_end_matches(')');
                        if let Ok(alpha) = alpha_str.parse::<f64>() {
                            return Some(ActivationFunction::ELU(alpha));
                        }
                    }
                    Some(ActivationFunction::ELU(1.0)) // Default alpha
                } else {
                    None
                }
            }
        }
    }

    /// Convert ActivationFunction enum to activation function name
    pub fn to_name(&self) -> String {
        match self {
            ActivationFunction::ReLU => "relu".to_string(),
            ActivationFunction::Sigmoid => "sigmoid".to_string(),
            ActivationFunction::Tanh => "tanh".to_string(),
            ActivationFunction::Softmax => "softmax".to_string(),
            ActivationFunction::LeakyReLU(alpha) => format!("leaky_relu({})", alpha),
            ActivationFunction::ELU(alpha) => format!("elu({})", alpha),
            ActivationFunction::GELU => "gelu".to_string(),
            ActivationFunction::Swish => "swish".to_string(),
            ActivationFunction::Mish => "mish".to_string(),
        }
    }

    /// Create activation function from enum
    pub fn create<F: Float + Debug + ScalarOperand + Send + Sync>(&self) -> Box<dyn Activation<F>> {
        match self {
            ActivationFunction::ReLU => Box::new(ReLU::new()),
            ActivationFunction::Sigmoid => Box::new(Sigmoid::new()),
            ActivationFunction::Tanh => Box::new(Tanh::new()),
            ActivationFunction::Softmax => Box::new(Softmax::new(1)), // Default axis is 1
            ActivationFunction::LeakyReLU(alpha) => Box::new(LeakyReLU::new(*alpha)),
            ActivationFunction::ELU(alpha) => Box::new(ELU::new(*alpha)),
            ActivationFunction::GELU => Box::new(GELU::new()),
            ActivationFunction::Swish => Box::new(Swish::new(1.0)),
            ActivationFunction::Mish => Box::new(Mish::new()),
        }
    }
}

/// Activation function factory
pub struct ActivationFactory;

impl ActivationFactory {
    /// Create activation function from name
    pub fn create<F: Float + Debug + ScalarOperand + Send + Sync>(
        name: &str,
    ) -> Option<Box<dyn Activation<F>>> {
        ActivationFunction::from_name(name).map(|af| af.create::<F>())
    }

    /// Get activation function names
    pub fn get_activation_names() -> HashMap<&'static str, &'static str> {
        let mut names = HashMap::new();
        names.insert("relu", "ReLU activation function");
        names.insert("sigmoid", "Sigmoid activation function");
        names.insert("tanh", "Tanh activation function");
        names.insert("softmax", "Softmax activation function");
        names.insert("leaky_relu", "Leaky ReLU activation function");
        names.insert("elu", "ELU activation function");
        names.insert("gelu", "GELU activation function");
        names.insert("swish", "Swish activation function");
        names.insert("mish", "Mish activation function");
        names
    }
}
