// Copyright (c) 2025, `SciRS2` Team
//
// Licensed under the Apache License, Version 2.0
// (LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0)
//

//! Serialization and deserialization of neural network models.
//!
//! This module provides utilities for saving and loading neural network models,
//! including their parameters, architecture, and optimizer state.

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use ::ndarray::{Array, Ix1, Ix2, IxDyn};

#[cfg(feature = "serialization")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "serialization")]
use serde_json;

use chrono;

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
    pub inputshape: Vec<usize>,

    /// Output shape.
    pub outputshape: Vec<usize>,

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
    basedir: PathBuf,
}

impl ModelSerializer {
    /// Create a new model serializer.
    pub fn new(basedir: impl AsRef<Path>) -> Self {
        Self {
            basedir: basedir.as_ref().to_path_buf(),
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
        let modeldir = self.basedir.join(name).join(version);
        fs::create_dir_all(&modeldir)?;

        // Create metadata
        let metadata = ModelMetadata {
            name: name.to_string(),
            version: version.to_string(),
            framework_version: "0.1.0".to_string(),
            created_at: chrono::Utc::now().to_rfc3339(),
            inputshape: vec![],  // This would be determined from the model
            outputshape: vec![], // This would be determined from the model
            additional_info: HashMap::new(),
        };

        // Create architecture
        let architecture = self.create_architecture(model)?;

        // Save parameters
        let mut parameter_files = HashMap::new();
        self.save_parameters(model, &modeldir, &mut parameter_files)?;

        // Save optimizer state if provided
        let optimizer_state = if let Some(optimizer) = optimizer {
            let optimizerpath = self.save_optimizer(optimizer, &modeldir)?;
            Some(
                optimizerpath
                    .file_name()
                    .expect("Operation failed")
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
        let model_file_path = modeldir.join("model.json");
        let model_file_json = serde_json::to_string_pretty(&model_file)?;
        let mut file = File::create(&model_file_path)?;
        file.write_all(model_file_json.as_bytes())?;

        Ok(model_file_path)
    }

    /// Load a model from disk.
    pub fn loadmodel(
        &self,
        name: &str,
        version: &str,
    ) -> CoreResult<(Sequential, Option<Box<dyn Optimizer>>)> {
        // Get model directory
        let modeldir = self.basedir.join(name).join(version);

        // Load model file
        let model_file_path = modeldir.join("model.json");
        let mut file = File::open(&model_file_path)?;
        let mut model_file_json = String::new();
        file.read_to_string(&mut model_file_json)?;

        let model_file: ModelFile = serde_json::from_str(&model_file_json)?;

        // Create model from architecture
        let mut model = self.create_model_from_architecture(&model_file.architecture)?;

        // Load parameters
        self.load_parameters(&mut model, &modeldir, &model_file.parameter_files)?;

        // Load optimizer if available
        let optimizer = if let Some(optimizer_state) = &model_file.optimizer_state {
            let optimizerpath = modeldir.join(optimizer_state);
            Some(self.load_optimizer(&optimizerpath)?)
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
        let layer_type = layer.layer_type();
        if !["Linear", "Conv2D", "MaxPool2D", "BatchNorm", "Dropout"].contains(&layer_type) {
            return Err(CoreError::NotImplementedError(ErrorContext::new(format!(
                "Serialization not implemented for layer type: {}",
                layer.name()
            ))));
        };

        // Create configuration based on layer type
        let config = match layer_type {
            "Linear" => {
                // Without downcasting, we can't extract the actual configuration
                // This would need to be stored in the layer itself
                serde_json::json!({
                    "in_features": 0,
                    "out_features": 0,
                    "bias": true,
                    "activation": "relu",
                })
            }
            "Conv2D" => {
                serde_json::json!({
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
            "MaxPool2D" => {
                serde_json::json!({
                    "kernel_size": [2, 2],
                    "stride": [2, 2],
                    "padding": [0, 0],
                })
            }
            "BatchNorm" => {
                serde_json::json!({
                    "num_features": 0,
                    "epsilon": 1e-5,
                    "momentum": 0.1,
                })
            }
            "Dropout" => {
                serde_json::json!({
                    "rate": 0.5,
                    "seed": null,
                })
            }
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
        modeldir: &Path,
        parameter_files: &mut HashMap<String, String>,
    ) -> CoreResult<()> {
        // Create parameters directory
        let params_dir = modeldir.join("parameters");
        fs::create_dir_all(&params_dir)?;

        // Save parameters for each layer
        for (i, layer) in model.layers().iter().enumerate() {
            for (j, param) in layer.parameters().iter().enumerate() {
                // Generate parameter file name
                let param_name = format!("layer_{i}_param_{j}");
                let param_file = format!("{param_name}.npz");
                let param_path = params_dir.join(&param_file);

                // Save parameter
                self.save_parameter(param.as_ref(), &param_path)?;

                // Add to parameter files map
                parameter_files.insert(param_name, format!("parameters/{param_file}"));
            }
        }

        Ok(())
    }

    /// Save a single parameter.
    fn save_parameter(&self, param: &dyn ArrayProtocol, path: &Path) -> CoreResult<()> {
        // Layer parameters (weight matrices, bias vectors) are naturally
        // Ix2/Ix1, not IxDyn — downcast regardless of the concrete dimension
        // type rather than assuming IxDyn (see `super::downcast_arg_to_ixdyn`).
        if let Some(ndarray) = super::downcast_arg_to_ixdyn::<f64>(param.as_any()) {
            // Save the array shape and data
            let shape: Vec<usize> = ndarray.shape().to_vec();
            let data: Vec<f64> = ndarray.iter().cloned().collect();

            let save_data = serde_json::json!({
                "shape": shape,
                "data": data,
            });

            let mut file = File::create(path)?;
            let json_str = serde_json::to_string(&save_data)?;
            file.write_all(json_str.as_bytes())?;

            Ok(())
        } else {
            Err(CoreError::NotImplementedError(ErrorContext::new(
                "Parameter serialization not implemented for this array type".to_string(),
            )))
        }
    }

    /// Save optimizer state.
    fn save_optimizer(&self, _optimizer: &dyn Optimizer, modeldir: &Path) -> CoreResult<PathBuf> {
        // Create optimizer state file
        let optimizerpath = modeldir.join("optimizer.json");

        // Save basic optimizer metadata
        // Since the Optimizer trait doesn't have methods to extract its type or config,
        // we'll just save a placeholder for now
        let optimizer_data = serde_json::json!({
            "type": "SGD", // Default to SGD for now
            "config": {
                "learningrate": 0.01,
                "momentum": null
            },
            "state": {} // Optimizer state would be saved here
        });

        let mut file = File::create(&optimizerpath)?;
        let json_str = serde_json::to_string_pretty(&optimizer_data)?;
        file.write_all(json_str.as_bytes())?;

        Ok(optimizerpath)
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
                Ok(Box::new(Linear::new_random(
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
                Ok(Box::new(Conv2D::withshape(
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
                Ok(Box::new(BatchNorm::withshape(
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
                "Deserialization not implemented for layer type: {layer_type}",
                layer_type = config.layer_type
            )))),
        }
    }

    /// Rebuilds a loaded parameter as the same concrete `NdarrayWrapper<f64, D>`
    /// dimensionality that `existing` (the pre-load parameter value) uses,
    /// mirroring `save_parameter`'s IxDyn/Ix2/Ix1 scope so the reloaded value
    /// stays dispatch-compatible with operations (like `matmul`/`add`) that
    /// require both operands to share the same concrete `D`.
    fn rebuild_parameter(
        existing: &dyn ArrayProtocol,
        shape: &[usize],
        data: Vec<f64>,
    ) -> CoreResult<Box<dyn ArrayProtocol>> {
        let dyn_arr = Array::from_shape_vec(IxDyn(shape), data).map_err(|e| {
            CoreError::InvalidArgument(ErrorContext::new(format!(
                "Parameter shape mismatch on load: {e}"
            )))
        })?;

        if existing
            .as_any()
            .downcast_ref::<NdarrayWrapper<f64, Ix1>>()
            .is_some()
        {
            let arr = dyn_arr.into_dimensionality::<Ix1>().map_err(|e| {
                CoreError::InvalidArgument(ErrorContext::new(format!(
                    "expected a 1-D parameter on load: {e}"
                )))
            })?;
            return Ok(Box::new(NdarrayWrapper::new(arr)));
        }
        if existing
            .as_any()
            .downcast_ref::<NdarrayWrapper<f64, Ix2>>()
            .is_some()
        {
            let arr = dyn_arr.into_dimensionality::<Ix2>().map_err(|e| {
                CoreError::InvalidArgument(ErrorContext::new(format!(
                    "expected a 2-D parameter on load: {e}"
                )))
            })?;
            return Ok(Box::new(NdarrayWrapper::new(arr)));
        }
        if existing
            .as_any()
            .downcast_ref::<NdarrayWrapper<f64, IxDyn>>()
            .is_some()
        {
            return Ok(Box::new(NdarrayWrapper::new(dyn_arr)));
        }

        Err(CoreError::NotImplementedError(ErrorContext::new(
            "Parameter loading not implemented for this array type (e.g. Conv2D's Ix4 filters — \
             matches save_parameter's existing IxDyn/Ix2/Ix1 scope)"
                .to_string(),
        )))
    }

    /// Load parameters into a model.
    fn load_parameters(
        &self,
        model: &mut Sequential,
        modeldir: &Path,
        parameter_files: &HashMap<String, String>,
    ) -> CoreResult<()> {
        // For each layer, load its parameters and actually write them back
        // (via `update_parameter`), instead of only reading and discarding
        // them.
        for (i, layer) in model.layers_mut().iter_mut().enumerate() {
            // Snapshot the existing parameters/names before mutating the
            // layer, since `parameter_names()` must be called before (or
            // independently of) `update_parameter()`.
            let existing_params = layer.parameters();
            let names = layer.parameter_names();

            for (j, existing) in existing_params.iter().enumerate() {
                let param_name = format!("layer_{i}_param_{j}");
                let Some(param_file) = parameter_files.get(&param_name) else {
                    continue;
                };
                let param_path = modeldir.join(param_file);

                if !param_path.exists() {
                    return Err(CoreError::InvalidArgument(ErrorContext::new(format!(
                        "Parameter file not found: {path}",
                        path = param_path.display()
                    ))));
                }

                let mut file = File::open(&param_path)?;
                let mut json_str = String::new();
                file.read_to_string(&mut json_str)?;

                let load_data: serde_json::Value = serde_json::from_str(&json_str)?;
                let shape: Vec<usize> = serde_json::from_value(load_data["shape"].clone())?;
                let data: Vec<f64> = serde_json::from_value(load_data["data"].clone())?;

                let new_value = Self::rebuild_parameter(existing.as_ref(), &shape, data)?;

                let name = names.get(j).ok_or_else(|| {
                    CoreError::InvalidArgument(ErrorContext::new(format!(
                        "layer {i} has {nparams} parameter(s) but only {nnames} name(s)",
                        nparams = existing_params.len(),
                        nnames = names.len()
                    )))
                })?;
                layer.update_parameter(name, new_value).map_err(|e| {
                    CoreError::InvalidArgument(ErrorContext::new(format!(
                        "Failed to update parameter '{name}' on layer {i}: {e}"
                    )))
                })?;
            }
        }

        Ok(())
    }

    /// Load optimizer state.
    fn load_optimizer(&self, optimizerpath: &Path) -> CoreResult<Box<dyn Optimizer>> {
        // Check if optimizer file exists
        if !optimizerpath.exists() {
            return Err(CoreError::InvalidArgument(ErrorContext::new(format!(
                "Optimizer file not found: {path}",
                path = optimizerpath.display()
            ))));
        }

        // Load optimizer metadata
        let mut file = File::open(optimizerpath)?;
        let mut json_str = String::new();
        file.read_to_string(&mut json_str)?;

        let optimizer_data: serde_json::Value = serde_json::from_str(&json_str)?;

        // Create optimizer based on type
        match optimizer_data["type"].as_str() {
            Some("SGD") => {
                let config = &optimizer_data["config"];
                let learningrate = config["learningrate"].as_f64().unwrap_or(0.01);
                let momentum = config["momentum"].as_f64();
                Ok(Box::new(SGD::new(learningrate, momentum)))
            }
            _ => {
                // Default to SGD for unknown types
                Ok(Box::new(SGD::new(0.01, None)))
            }
        }
    }
}

/// ONNX model exporter.
pub struct OnnxExporter;

impl OnnxExporter {
    /// Export a model to ONNX format.
    pub fn export(
        &self,
        _model: &Sequential,
        path: impl AsRef<Path>,
        _inputshape: &[usize],
    ) -> CoreResult<()> {
        // This is a simplified implementation for demonstration purposes.
        // In a real implementation, this would convert the model to ONNX format.

        // For now, we'll just create an empty file as a placeholder
        File::create(path.as_ref())?;

        Ok(())
    }
}

/// Create a model checkpoint.
#[allow(dead_code)]
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
    let model_version = format!("epoch_{epoch}");
    serializer.save_model(model, model_name, &model_version, Some(optimizer))?;

    Ok(())
}

/// Type alias for model checkpoint data
pub type ModelCheckpoint = (Sequential, Box<dyn Optimizer>, usize, HashMap<String, f64>);

/// Load a model checkpoint.
#[cfg(feature = "serialization")]
#[allow(dead_code)]
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
    let model_version = format!("epoch_{epoch}");
    let (model, optimizer) = serializer.loadmodel(model_name, &model_version)?;

    Ok((model, optimizer.expect("Operation failed"), epoch, metrics))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array_protocol;
    use crate::array_protocol::grad::SGD;
    use crate::array_protocol::ml_ops::ActivationFunc;
    use crate::array_protocol::neural::{Linear, Sequential};
    use tempfile::tempdir;

    /// Extracts a layer's `param_idx`-th parameter as an owned `Array2<f64>`.
    fn param_ix2(model: &Sequential, layer_idx: usize, param_idx: usize) -> ::ndarray::Array2<f64> {
        model.layers()[layer_idx].parameters()[param_idx]
            .as_any()
            .downcast_ref::<NdarrayWrapper<f64, ::ndarray::Ix2>>()
            .expect("parameter should be a NdarrayWrapper<f64, Ix2>")
            .as_array()
            .clone()
    }

    /// Extracts a layer's `param_idx`-th parameter as an owned `Array1<f64>`.
    fn param_ix1(model: &Sequential, layer_idx: usize, param_idx: usize) -> ::ndarray::Array1<f64> {
        model.layers()[layer_idx].parameters()[param_idx]
            .as_any()
            .downcast_ref::<NdarrayWrapper<f64, ::ndarray::Ix1>>()
            .expect("parameter should be a NdarrayWrapper<f64, Ix1>")
            .as_array()
            .clone()
    }

    #[test]
    fn test_model_serializer() {
        // Initialize the array protocol system
        array_protocol::init();

        // Create a temporary directory
        let temp_dir = tempdir().expect("failed to create temp dir");

        // Create a model
        let mut model = Sequential::new("test_model", Vec::new());

        // Add layers. `new_random`'s Xavier-initialized weights are
        // non-constant by construction (this is what makes the round-trip
        // check below meaningful: a no-op `load_parameters` — the bug this
        // regression-tests — would leave `loadedmodel`'s weights at
        // whatever *fresh* random values `create_model_from_architecture`
        // happened to draw, which would not match `model`'s original values).
        model.add_layer(Box::new(Linear::new_random(
            "fc1",
            10,
            5,
            true,
            Some(ActivationFunc::ReLU),
        )));

        model.add_layer(Box::new(Linear::new_random("fc2", 5, 2, true, None)));

        // Capture the original parameter values before saving.
        let orig_fc1_weights = param_ix2(&model, 0, 0);
        let orig_fc1_bias = param_ix1(&model, 0, 1);
        let orig_fc2_weights = param_ix2(&model, 1, 0);
        let orig_fc2_bias = param_ix1(&model, 1, 1);

        // Create optimizer
        let optimizer = SGD::new(0.01, Some(0.9));

        // Create serializer
        let serializer = ModelSerializer::new(temp_dir.path());

        // Save model
        serializer
            .save_model(&model, "test_model", "v1", Some(&optimizer))
            .expect("save_model should succeed");

        // Load model
        let (loadedmodel, loaded_optimizer) = serializer
            .loadmodel("test_model", "v1")
            .expect("loadmodel should succeed");

        // Check model
        assert_eq!(loadedmodel.layers().len(), 2);
        assert!(loaded_optimizer.is_some());

        // The actual regression check: loaded parameter VALUES must match
        // what was saved, element-for-element — not just shape/count. Uses a
        // tight-but-nonzero tolerance rather than `assert_eq!`: round-tripping
        // an f64 through `serde_json`'s text-based JSON representation can
        // perturb the last bit or two for specific values (verified directly:
        // `serde_json::to_string` emits the exact shortest round-trippable
        // decimal for e.g. `-0.11303865380207907`, but
        // `serde_json::from_str`/`from_value` parses that same text back to
        // `-0.11303865380207909`, one ULP off — a `serde_json`
        // float-parsing characteristic, not something `save_parameter`/
        // `load_parameters` control). `1e-9` is ~1e8 times looser than that
        // single-ULP noise, while still ~1e6 times tighter than the smallest
        // Xavier-initialized weight magnitude here — nowhere near loose
        // enough to hide a real bug (wrong shape, wrong values, or a no-op
        // load would all fail this by many orders of magnitude).
        assert_arrays_close(&param_ix2(&loadedmodel, 0, 0), &orig_fc1_weights, 1e-9);
        assert_arrays_close_1d(&param_ix1(&loadedmodel, 0, 1), &orig_fc1_bias, 1e-9);
        assert_arrays_close(&param_ix2(&loadedmodel, 1, 0), &orig_fc2_weights, 1e-9);
        assert_arrays_close_1d(&param_ix1(&loadedmodel, 1, 1), &orig_fc2_bias, 1e-9);

        // Guard against a degenerate all-zero/all-equal round trip
        // trivially "passing" (Xavier init makes this exceedingly unlikely,
        // but assert it outright rather than relying on that alone).
        assert!(orig_fc1_weights
            .iter()
            .any(|&v| v != orig_fc1_weights[[0, 0]]));
    }

    /// Asserts two 2-D arrays match within `tol` per element (see
    /// `test_model_serializer` for why this isn't `assert_eq!`).
    fn assert_arrays_close(
        actual: &::ndarray::Array2<f64>,
        expected: &::ndarray::Array2<f64>,
        tol: f64,
    ) {
        assert_eq!(actual.dim(), expected.dim());
        for ((i, j), &e) in expected.indexed_iter() {
            let a = actual[[i, j]];
            assert!(
                (a - e).abs() < tol,
                "mismatch at [{i},{j}]: actual={a}, expected={e}"
            );
        }
    }

    /// 1-D counterpart of [`assert_arrays_close`].
    fn assert_arrays_close_1d(
        actual: &::ndarray::Array1<f64>,
        expected: &::ndarray::Array1<f64>,
        tol: f64,
    ) {
        assert_eq!(actual.dim(), expected.dim());
        for (i, &e) in expected.indexed_iter() {
            let a = actual[i];
            assert!(
                (a - e).abs() < tol,
                "mismatch at [{i}]: actual={a}, expected={e}"
            );
        }
    }

    #[test]
    fn test_save_load_checkpoint() {
        // Initialize the array protocol system
        array_protocol::init();

        // Create a temporary directory
        let temp_dir = tempdir().expect("failed to create temp dir");

        // Create a model
        let mut model = Sequential::new("test_model", Vec::new());

        // Add layers
        model.add_layer(Box::new(Linear::new_random(
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
        save_checkpoint(&model, &optimizer, &checkpoint_path, 10, metrics.clone())
            .expect("save_checkpoint should succeed");

        // Load checkpoint
        let (loadedmodel, loaded_optimizer, loaded_epoch, loaded_metrics) =
            load_checkpoint(&checkpoint_path).expect("load_checkpoint should succeed");

        // Check loaded data
        assert_eq!(loadedmodel.layers().len(), 1);
        assert_eq!(loaded_epoch, 10);
        assert_eq!(loaded_metrics.get("loss"), metrics.get("loss"));
        assert_eq!(loaded_metrics.get("accuracy"), metrics.get("accuracy"));
    }
}
