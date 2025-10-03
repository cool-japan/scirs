//! JAX format converter
#![allow(dead_code)]

use crate::error::{IoError, Result};
use crate::ml_framework::converters::MLFrameworkConverter;
use crate::ml_framework::types::{MLFramework, MLModel, MLTensor};
use scirs2_core::ndarray::{ArrayD, IxDyn};
use std::fs::File;
use std::path::Path;

/// JAX format converter
pub struct JAXConverter;

impl MLFrameworkConverter for JAXConverter {
    fn save_model(&self, model: &MLModel, path: &Path) -> Result<()> {
        // JAX uses a simpler pickle-like format
        let jax_model = serde_json::json!({
            "format": "jax",
            "version": "0.4.0",
            "pytree": {
                "params": model.weights.iter().map(|(name, tensor)| {
                    (name.clone(), serde_json::json!({
                        "shape": tensor.metadata.shape,
                        "dtype": format!("{:?}", tensor.metadata.dtype),
                        "data": tensor.data.as_slice().unwrap().to_vec()
                    }))
                }).collect::<serde_json::Map<String, serde_json::Value>>(),
                "config": model.config
            },
            "metadata": model.metadata
        });

        let file = File::create(path).map_err(IoError::Io)?;
        serde_json::to_writer_pretty(file, &jax_model)
            .map_err(|e| IoError::SerializationError(e.to_string()))
    }

    fn load_model(&self, path: &Path) -> Result<MLModel> {
        let file = File::open(path).map_err(IoError::Io)?;
        let jax_model: serde_json::Value = serde_json::from_reader(file)
            .map_err(|e| IoError::SerializationError(e.to_string()))?;

        let mut model = MLModel::new(MLFramework::JAX);

        if let Some(metadata) = jax_model.get("metadata") {
            model.metadata = serde_json::from_value(metadata.clone())
                .map_err(|e| IoError::SerializationError(e.to_string()))?;
        }

        if let Some(pytree) = jax_model.get("pytree") {
            if let Some(params) = pytree.get("params").and_then(|v| v.as_object()) {
                for (name, param_data) in params {
                    let shape: Vec<usize> = serde_json::from_value(param_data["shape"].clone())
                        .map_err(|e| IoError::SerializationError(e.to_string()))?;

                    let data: Vec<f32> = serde_json::from_value(param_data["data"].clone())
                        .map_err(|e| IoError::SerializationError(e.to_string()))?;

                    let array = ArrayD::from_shape_vec(IxDyn(&shape), data)
                        .map_err(|e| IoError::Other(e.to_string()))?;

                    model
                        .weights
                        .insert(name.clone(), MLTensor::new(array, Some(name.clone())));
                }
            }

            if let Some(config) = pytree.get("config") {
                model.config = serde_json::from_value(config.clone())
                    .map_err(|e| IoError::SerializationError(e.to_string()))?;
            }
        }

        Ok(model)
    }

    fn save_tensor(&self, tensor: &MLTensor, path: &Path) -> Result<()> {
        let tensor_data = serde_json::json!({
            "jax_array": {
                "shape": tensor.metadata.shape,
                "dtype": format!("{:?}", tensor.metadata.dtype),
                "data": tensor.data.as_slice().unwrap().to_vec(),
                "weak_type": false
            }
        });

        let file = File::create(path).map_err(IoError::Io)?;
        serde_json::to_writer_pretty(file, &tensor_data)
            .map_err(|e| IoError::SerializationError(e.to_string()))
    }

    fn load_tensor(&self, path: &Path) -> Result<MLTensor> {
        let file = File::open(path).map_err(IoError::Io)?;
        let tensor_data: serde_json::Value = serde_json::from_reader(file)
            .map_err(|e| IoError::SerializationError(e.to_string()))?;

        if let Some(jax_array) = tensor_data.get("jax_array") {
            let shape: Vec<usize> = serde_json::from_value(jax_array["shape"].clone())
                .map_err(|e| IoError::SerializationError(e.to_string()))?;

            let data: Vec<f32> = serde_json::from_value(jax_array["data"].clone())
                .map_err(|e| IoError::SerializationError(e.to_string()))?;

            let array = ArrayD::from_shape_vec(IxDyn(&shape), data)
                .map_err(|e| IoError::Other(e.to_string()))?;

            return Ok(MLTensor::new(array, None));
        }

        Err(IoError::Other("Invalid JAX tensor format".to_string()))
    }
}
