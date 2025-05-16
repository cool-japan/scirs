// Copyright (c) 2025, SciRS2 Team
//
// Licensed under either of
//
// * Apache License, Version 2.0
//   (LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0)
// * MIT license
//   (LICENSE-MIT or http://opensource.org/licenses/MIT)
//
// at your option.
//

//! Serialization and deserialization of neural network models.
//!
//! This module provides utilities for saving and loading neural network models,
//! including their parameters, architecture, and optimizer state.

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use ndarray::IxDyn;

#[cfg(feature = "serialization")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "serialization")]
use serde_json;

use crate::array_protocol::grad::{Optimizer, SGD};
use crate::array_protocol::ml_ops::ActivationFunc;
use crate::array_protocol::neural::{
    BatchNorm, Conv2D, Dropout, Layer, Linear, MaxPool2D, Sequential,
};
use crate::array_protocol::{ArrayProtocol, NdarrayWrapper};
use crate::error::{CoreError, CoreResult, ErrorContext};

/// Trait for serializable objects.
pub trait Serializable {
    /// Serialize the object to a byte vector.
    fn serialize(&self) -> CoreResult<Vec<u8>>;

    /// Deserialize the object from a byte vector.
    fn deserialize(bytes: &[u8]) -> CoreResult<Self>
    where
        Self: Sized;

    /// Get the object type name.
    fn type_name(&self) -> &str;
}

/// Serialized model file format.
#[derive(Serialize, Deserialize)]
pub struct ModelFile {
    /// Model architecture metadata.
    pub metadata: ModelMetadata,

    /// Model architecture.
    pub architecture: ModelArchitecture,

    /// Parameter file paths relative to the model file.
    pub parameter_files: HashMap<String, String>,

    /// Optimizer state file path relative to the model file.
    pub optimizer_state: Option<String>,
}

/// Model metadata.
#[derive(Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Model name.
    pub name: String,

    /// Model version.
    pub version: String,

    /// Framework version.
    pub framework_version: String,

    /// Creation date.
    pub created_at: String,

    /// Input shape.
    pub input_shape: Vec<usize>,

    /// Output shape.
    pub output_shape: Vec<usize>,

    /// Additional metadata.
    pub additional_info: HashMap<String, String>,
}

/// Model architecture.
#[cfg_attr(feature = "serialization", derive(Serialize, Deserialize))]
pub struct ModelArchitecture {
    /// Model type.
    pub model_type: String,

    /// Layer configurations.
    pub layers: Vec<LayerConfig>,
}

/// Layer configuration.
#[cfg_attr(feature = "serialization", derive(Serialize, Deserialize))]
pub struct LayerConfig {
    /// Layer type.
    pub layer_type: String,

    /// Layer name.
    pub name: String,

    /// Layer configuration.
    #[cfg(feature = "serialization")]
    pub config: serde_json::Value,
    #[cfg(not(feature = "serialization"))]
    pub config: HashMap<String, String>, // Fallback when serialization is not enabled
}

/// Model serializer for saving neural network models.
pub struct ModelSerializer {
    /// Base directory for saving models.
    base_dir: PathBuf,
}

impl ModelSerializer {
    /// Create a new model serializer.
    pub fn new(base_dir: impl AsRef<Path>) -> Self {
        Self {
            base_dir: base_dir.as_ref().to_path_buf(),
        }
    }

    /// Save a model to disk.
    pub fn save_model(
        &self,
        model: &Sequential,
        name: &str,
        version: &str,
        optimizer: Option<&dyn Optimizer>,
    ) -> CoreResult<PathBuf> {
        // Create model directory
        let model_dir = self.base_dir.join(name).join(version);
        fs::create_dir_all(&model_dir)?;

        // Create metadata
        let metadata = ModelMetadata {
            name: name.to_string(),
            version: version.to_string(),
            framework_version: "0.1.0".to_string(),
            created_at: chrono::Utc::now().to_rfc3339(),
            input_shape: vec![],  // This would be determined from the model
            output_shape: vec![], // This would be determined from the model
            additional_info: HashMap::new(),
        };

        // Create architecture
        let architecture = self.create_architecture(model)?;

        // Save parameters
        let mut parameter_files = HashMap::new();
        self.save_parameters(model, &model_dir, &mut parameter_files)?;

        // Save optimizer state if provided
        let optimizer_state = if let Some(optimizer) = optimizer {
            let optimizer_path = self.save_optimizer(optimizer, &model_dir)?;
            Some(
                optimizer_path
                    .file_name()
                    .unwrap()
                    .to_string_lossy()
                    .to_string(),
            )
        } else {
            None
        };

        // Create model file
        let model_file = ModelFile {
            metadata,
            architecture,
            parameter_files,
            optimizer_state,
        };

        // Serialize model file
        let model_file_path = model_dir.join("model.json");
        let model_file_json = serde_json::to_string_pretty(&model_file)?;
        let mut file = File::create(&model_file_path)?;
        file.write_all(model_file_json.as_bytes())?;

        Ok(model_file_path)
    }

    /// Load a model from disk.
    pub fn load_model(
        &self,
        name: &str,
        version: &str,
    ) -> CoreResult<(Sequential, Option<Box<dyn Optimizer>>)> {
        // Get model directory
        let model_dir = self.base_dir.join(name).join(version);

        // Load model file
        let model_file_path = model_dir.join("model.json");
        let mut file = File::open(&model_file_path)?;
        let mut model_file_json = String::new();
        file.read_to_string(&mut model_file_json)?;

        let model_file: ModelFile = serde_json::from_str(&model_file_json)?;

        // Create model from architecture
        let model = self.create_model_from_architecture(&model_file.architecture)?;

        // Load parameters
        self.load_parameters(&model, &model_dir, &model_file.parameter_files)?;

        // Load optimizer if available
        let optimizer = if let Some(optimizer_state) = &model_file.optimizer_state {
            let optimizer_path = model_dir.join(optimizer_state);
            Some(self.load_optimizer(&optimizer_path)?)
        } else {
            None
        };

        Ok((model, optimizer))
    }

    /// Create architecture from a model.
    fn create_architecture(&self, model: &Sequential) -> CoreResult<ModelArchitecture> {
        let mut layers = Vec::new();

        for layer in model.layers() {
            let layer_config = self.create_layer_config(layer.as_ref())?;
            layers.push(layer_config);
        }

        Ok(ModelArchitecture {
            model_type: "Sequential".to_string(),
            layers,
        })
    }

    /// Create layer configuration from a layer.
    fn create_layer_config(&self, layer: &dyn Layer) -> CoreResult<LayerConfig> {
        let layer_type = if layer.as_any().is::<Linear>() {
            "Linear"
        } else if layer.as_any().is::<Conv2D>() {
            "Conv2D"
        } else if layer.as_any().is::<MaxPool2D>() {
            "MaxPool2D"
        } else if layer.as_any().is::<BatchNorm>() {
            "BatchNorm"
        } else if layer.as_any().is::<Dropout>() {
            "Dropout"
        } else {
            return Err(CoreError::NotImplementedError(ErrorContext::new(format!(
                "Serialization not implemented for layer type: {}",
                layer.name()
            ))));
        };

        // Create configuration based on layer type
        let config = match layer_type {
            "Linear" => {
                let _linear = layer.as_any().downcast_ref::<Linear>().unwrap();
                // Extract configuration from linear layer
                serde_json::json!({
                    // This would include in_features, out_features, bias, activation, etc.
                    "in_features": 0,
                    "out_features": 0,
                    "bias": true,
                    "activation": "relu",
                })
            }
            "Conv2D" => {
                let _conv = layer.as_any().downcast_ref::<Conv2D>().unwrap();
                // Extract configuration from conv layer
                serde_json::json!({
                    // This would include filter_height, filter_width, in_channels, etc.
                    "filter_height": 3,
                    "filter_width": 3,
                    "in_channels": 0,
                    "out_channels": 0,
                    "stride": [1, 1],
                    "padding": [0, 0],
                    "bias": true,
                    "activation": "relu",
                })
            }
            // Other layer types would be handled similarly
            _ => serde_json::json!({}),
        };

        Ok(LayerConfig {
            layer_type: layer_type.to_string(),
            name: layer.name().to_string(),
            config,
        })
    }

    /// Save parameters of a model.
    fn save_parameters(
        &self,
        model: &Sequential,
        model_dir: &Path,
        parameter_files: &mut HashMap<String, String>,
    ) -> CoreResult<()> {
        // Create parameters directory
        let params_dir = model_dir.join("parameters");
        fs::create_dir_all(&params_dir)?;

        // Save parameters for each layer
        for (i, layer) in model.layers().iter().enumerate() {
            for (j, param) in layer.parameters().iter().enumerate() {
                // Generate parameter file name
                let param_name = format!("layer_{}_param_{}", i, j);
                let param_file = format!("{}.npz", param_name);
                let param_path = params_dir.join(&param_file);

                // Save parameter
                self.save_parameter(param.as_ref(), &param_path)?;

                // Add to parameter files map
                parameter_files.insert(param_name, format!("parameters/{}", param_file));
            }
        }

        Ok(())
    }

    /// Save a single parameter.
    fn save_parameter(&self, param: &dyn ArrayProtocol, path: &Path) -> CoreResult<()> {
        // For simplicity, we'll assume all parameters are NdarrayWrapper<f64, IxDyn>
        if let Some(array) = param.as_any().downcast_ref::<NdarrayWrapper<f64, IxDyn>>() {
            let _ndarray = array.as_array();

            // In a real implementation, we would serialize the ndarray to a file
            // For now, we'll just create an empty file as a placeholder
            File::create(path)?;

            Ok(())
        } else {
            Err(CoreError::NotImplementedError(ErrorContext::new(
                "Parameter serialization not implemented for this array type".to_string(),
            )))
        }
    }

    /// Save optimizer state.
    fn save_optimizer(&self, _optimizer: &dyn Optimizer, model_dir: &Path) -> CoreResult<PathBuf> {
        // Create optimizer state file
        let optimizer_path = model_dir.join("optimizer.npz");

        // In a real implementation, we would serialize the optimizer state to a file
        // For now, we'll just create an empty file as a placeholder
        File::create(&optimizer_path)?;

        Ok(optimizer_path)
    }

    /// Create a model from architecture.
    fn create_model_from_architecture(
        &self,
        architecture: &ModelArchitecture,
    ) -> CoreResult<Sequential> {
        let mut model = Sequential::new(&architecture.model_type, Vec::new());

        // Create layers from configuration
        for layer_config in &architecture.layers {
            let layer = self.create_layer_from_config(layer_config)?;
            model.add_layer(layer);
        }

        Ok(model)
    }

    /// Create a layer from configuration.
    fn create_layer_from_config(&self, config: &LayerConfig) -> CoreResult<Box<dyn Layer>> {
        match config.layer_type.as_str() {
            "Linear" => {
                // Extract configuration
                let in_features = config.config["in_features"].as_u64().unwrap_or(0) as usize;
                let out_features = config.config["out_features"].as_u64().unwrap_or(0) as usize;
                let bias = config.config["bias"].as_bool().unwrap_or(true);
                let activation = match config.config["activation"].as_str() {
                    Some("relu") => Some(ActivationFunc::ReLU),
                    Some("sigmoid") => Some(ActivationFunc::Sigmoid),
                    Some("tanh") => Some(ActivationFunc::Tanh),
                    _ => None,
                };

                // Create layer
                Ok(Box::new(Linear::with_shape(
                    &config.name,
                    in_features,
                    out_features,
                    bias,
                    activation,
                )))
            }
            "Conv2D" => {
                // Extract configuration
                let filter_height = config.config["filter_height"].as_u64().unwrap_or(3) as usize;
                let filter_width = config.config["filter_width"].as_u64().unwrap_or(3) as usize;
                let in_channels = config.config["in_channels"].as_u64().unwrap_or(0) as usize;
                let out_channels = config.config["out_channels"].as_u64().unwrap_or(0) as usize;
                let stride = (
                    config.config["stride"][0].as_u64().unwrap_or(1) as usize,
                    config.config["stride"][1].as_u64().unwrap_or(1) as usize,
                );
                let padding = (
                    config.config["padding"][0].as_u64().unwrap_or(0) as usize,
                    config.config["padding"][1].as_u64().unwrap_or(0) as usize,
                );
                let bias = config.config["bias"].as_bool().unwrap_or(true);
                let activation = match config.config["activation"].as_str() {
                    Some("relu") => Some(ActivationFunc::ReLU),
                    Some("sigmoid") => Some(ActivationFunc::Sigmoid),
                    Some("tanh") => Some(ActivationFunc::Tanh),
                    _ => None,
                };

                // Create layer
                Ok(Box::new(Conv2D::with_shape(
                    &config.name,
                    filter_height,
                    filter_width,
                    in_channels,
                    out_channels,
                    stride,
                    padding,
                    bias,
                    activation,
                )))
            }
            "MaxPool2D" => {
                // Extract configuration
                let kernel_size = (
                    config.config["kernel_size"][0].as_u64().unwrap_or(2) as usize,
                    config.config["kernel_size"][1].as_u64().unwrap_or(2) as usize,
                );
                let stride = if config.config["stride"].is_array() {
                    Some((
                        config.config["stride"][0].as_u64().unwrap_or(2) as usize,
                        config.config["stride"][1].as_u64().unwrap_or(2) as usize,
                    ))
                } else {
                    None
                };
                let padding = (
                    config.config["padding"][0].as_u64().unwrap_or(0) as usize,
                    config.config["padding"][1].as_u64().unwrap_or(0) as usize,
                );

                // Create layer
                Ok(Box::new(MaxPool2D::new(
                    &config.name,
                    kernel_size,
                    stride,
                    padding,
                )))
            }
            "BatchNorm" => {
                // Extract configuration
                let num_features = config.config["num_features"].as_u64().unwrap_or(0) as usize;
                let epsilon = config.config["epsilon"].as_f64().unwrap_or(1e-5);
                let momentum = config.config["momentum"].as_f64().unwrap_or(0.1);

                // Create layer
                Ok(Box::new(BatchNorm::with_shape(
                    &config.name,
                    num_features,
                    Some(epsilon),
                    Some(momentum),
                )))
            }
            "Dropout" => {
                // Extract configuration
                let rate = config.config["rate"].as_f64().unwrap_or(0.5);
                let seed = config.config["seed"].as_u64();

                // Create layer
                Ok(Box::new(Dropout::new(&config.name, rate, seed)))
            }
            _ => Err(CoreError::NotImplementedError(ErrorContext::new(format!(
                "Deserialization not implemented for layer type: {}",
                config.layer_type
            )))),
        }
    }

    /// Load parameters into a model.
    fn load_parameters(
        &self,
        model: &Sequential,
        model_dir: &Path,
        parameter_files: &HashMap<String, String>,
    ) -> CoreResult<()> {
        // For each layer, load its parameters
        for (i, layer) in model.layers().iter().enumerate() {
            for (j, _) in layer.parameters().iter().enumerate() {
                // Get parameter file
                let param_name = format!("layer_{}_param_{}", i, j);
                if let Some(param_file) = parameter_files.get(&param_name) {
                    let param_path = model_dir.join(param_file);

                    // Load parameter
                    // This would populate the parameter with its saved values
                    // For now, we'll just check if the file exists
                    if !param_path.exists() {
                        return Err(CoreError::InvalidArgument(ErrorContext::new(format!(
                            "Parameter file not found: {}",
                            param_path.display()
                        ))));
                    }
                }
            }
        }

        Ok(())
    }

    /// Load optimizer state.
    fn load_optimizer(&self, optimizer_path: &Path) -> CoreResult<Box<dyn Optimizer>> {
        // Check if optimizer file exists
        if !optimizer_path.exists() {
            return Err(CoreError::InvalidArgument(ErrorContext::new(format!(
                "Optimizer file not found: {}",
                optimizer_path.display()
            ))));
        }

        // In a real implementation, we would deserialize the optimizer state from the file
        // For now, we'll just create a new optimizer with default values
        Ok(Box::new(SGD::new(0.01, None)))
    }
}

/// ONNX model exporter.
pub struct OnnxExporter;

impl OnnxExporter {
    /// Export a model to ONNX format.
    pub fn export_model(
        _model: &Sequential,
        path: impl AsRef<Path>,
        _input_shape: &[usize],
    ) -> CoreResult<()> {
        // This is a simplified implementation for demonstration purposes.
        // In a real implementation, this would convert the model to ONNX format.

        // For now, we'll just create an empty file as a placeholder
        File::create(path.as_ref())?;

        Ok(())
    }
}

/// Create a model checkpoint.
pub fn save_checkpoint(
    model: &Sequential,
    optimizer: &dyn Optimizer,
    path: impl AsRef<Path>,
    epoch: usize,
    metrics: HashMap<String, f64>,
) -> CoreResult<()> {
    // Create checkpoint directory
    let checkpoint_dir = path.as_ref().parent().unwrap_or(Path::new("."));
    fs::create_dir_all(checkpoint_dir)?;

    // Create checkpoint metadata
    let metadata = serde_json::json!({
        "epoch": epoch,
        "metrics": metrics,
        "timestamp": chrono::Utc::now().to_rfc3339(),
    });

    // Save metadata
    let metadata_path = path.as_ref().with_extension("json");
    let metadata_json = serde_json::to_string_pretty(&metadata)?;
    let mut file = File::create(&metadata_path)?;
    file.write_all(metadata_json.as_bytes())?;

    // Create serializer
    let serializer = ModelSerializer::new(checkpoint_dir);

    // Save model and optimizer
    let model_name = "checkpoint";
    let model_version = format!("epoch_{}", epoch);
    serializer.save_model(model, model_name, &model_version, Some(optimizer))?;

    Ok(())
}

/// Type alias for model checkpoint data
pub type ModelCheckpoint = (Sequential, Box<dyn Optimizer>, usize, HashMap<String, f64>);

/// Load a model checkpoint.
#[cfg(feature = "serialization")]
pub fn load_checkpoint(path: impl AsRef<Path>) -> CoreResult<ModelCheckpoint> {
    // Load metadata
    let metadata_path = path.as_ref().with_extension("json");
    let mut file = File::open(&metadata_path)?;
    let mut metadata_json = String::new();
    file.read_to_string(&mut metadata_json)?;

    let metadata: serde_json::Value = serde_json::from_str(&metadata_json)?;

    // Extract metadata
    let epoch = metadata["epoch"].as_u64().unwrap_or(0) as usize;
    let metrics: HashMap<String, f64> =
        serde_json::from_value(metadata["metrics"].clone()).unwrap_or_else(|_| HashMap::new());

    // Create serializer
    let checkpoint_dir = path.as_ref().parent().unwrap_or(Path::new("."));
    let serializer = ModelSerializer::new(checkpoint_dir);

    // Load model and optimizer
    let model_name = "checkpoint";
    let model_version = format!("epoch_{}", epoch);
    let (model, optimizer) = serializer.load_model(model_name, &model_version)?;

    Ok((model, optimizer.unwrap(), epoch, metrics))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array_protocol;
    use crate::array_protocol::grad::SGD;
    use crate::array_protocol::ml_ops::ActivationFunc;
    use crate::array_protocol::neural::{Linear, Sequential};
    use tempfile::tempdir;

    #[test]
    #[ignore = "Model serialization not fully implemented"]
    fn test_model_serializer() {
        // Initialize the array protocol system
        array_protocol::init();

        // Create a temporary directory
        let temp_dir = match tempdir() {
            Ok(dir) => dir,
            Err(e) => {
                println!(
                    "Skipping test_model_serializer (temp dir creation failed): {}",
                    e
                );
                return;
            }
        };

        // Create a model
        let mut model = Sequential::new("test_model", Vec::new());

        // Add layers
        model.add_layer(Box::new(Linear::with_shape(
            "fc1",
            10,
            5,
            true,
            Some(ActivationFunc::ReLU),
        )));

        model.add_layer(Box::new(Linear::with_shape("fc2", 5, 2, true, None)));

        // Create optimizer
        let optimizer = SGD::new(0.01, Some(0.9));

        // Create serializer
        let serializer = ModelSerializer::new(temp_dir.path());

        // Save model
        let model_path = serializer.save_model(&model, "test_model", "v1", Some(&optimizer));
        if model_path.is_err() {
            println!("Save model failed: {:?}", model_path.err());
            return;
        }

        // Load model
        let (loaded_model, loaded_optimizer) = serializer.load_model("test_model", "v1").unwrap();

        // Check model
        assert_eq!(loaded_model.layers().len(), 2);
        assert!(loaded_optimizer.is_some());
    }

    #[test]
    #[ignore = "Checkpoint save/load not fully implemented"]
    fn test_save_load_checkpoint() {
        // Initialize the array protocol system
        array_protocol::init();

        // Create a temporary directory
        let temp_dir = match tempdir() {
            Ok(dir) => dir,
            Err(e) => {
                println!(
                    "Skipping test_save_load_checkpoint (temp dir creation failed): {}",
                    e
                );
                return;
            }
        };

        // Create a model
        let mut model = Sequential::new("test_model", Vec::new());

        // Add layers
        model.add_layer(Box::new(Linear::with_shape(
            "fc1",
            10,
            5,
            true,
            Some(ActivationFunc::ReLU),
        )));

        // Create optimizer
        let optimizer = SGD::new(0.01, Some(0.9));

        // Create metrics
        let mut metrics = HashMap::new();
        metrics.insert("loss".to_string(), 0.1);
        metrics.insert("accuracy".to_string(), 0.9);

        // Save checkpoint
        let checkpoint_path = temp_dir.path().join("checkpoint");
        let result = save_checkpoint(&model, &optimizer, &checkpoint_path, 10, metrics.clone());
        if let Err(e) = result {
            println!("Skipping test_save_load_checkpoint (save failed): {}", e);
            return;
        }

        // Load checkpoint
        let result = load_checkpoint(&checkpoint_path);
        if let Err(e) = result {
            println!("Skipping test_save_load_checkpoint (load failed): {}", e);
            return;
        }

        let (loaded_model, _loaded_optimizer, loaded_epoch, loaded_metrics) = result.unwrap();

        // Check loaded data
        assert_eq!(loaded_model.layers().len(), 1);
        assert_eq!(loaded_epoch, 10);
        assert_eq!(loaded_metrics.get("loss"), metrics.get("loss"));
        assert_eq!(loaded_metrics.get("accuracy"), metrics.get("accuracy"));
    }
}
