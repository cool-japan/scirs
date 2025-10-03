//! SafeTensors format converter
#![allow(dead_code)]

use crate::error::{IoError, Result};
use crate::ml_framework::converters::MLFrameworkConverter;
use crate::ml_framework::types::{MLFramework, MLModel, MLTensor};
use scirs2_core::ndarray::{ArrayD, IxDyn};
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

/// SafeTensors format converter
pub struct SafeTensorsConverter;

impl MLFrameworkConverter for SafeTensorsConverter {
    fn save_model(&self, model: &MLModel, path: &Path) -> Result<()> {
        // SafeTensors-like format
        let mut tensors = HashMap::new();

        for (name, tensor) in &model.weights {
            tensors.insert(
                name.clone(),
                serde_json::json!({
                    "shape": tensor.metadata.shape,
                    "dtype": format!("{:?}", tensor.metadata.dtype),
                    "data": tensor.data.as_slice().unwrap().to_vec(),
                }),
            );
        }

        let safetensors = serde_json::json!({
            "tensors": tensors,
            "metadata": model.metadata,
        });

        let file = File::create(path).map_err(IoError::Io)?;
        serde_json::to_writer(file, &safetensors)
            .map_err(|e| IoError::SerializationError(e.to_string()))
    }

    fn load_model(&self, path: &Path) -> Result<MLModel> {
        let file = File::open(path).map_err(IoError::Io)?;
        let safetensors: serde_json::Value = serde_json::from_reader(file)
            .map_err(|e| IoError::SerializationError(e.to_string()))?;

        let mut model = MLModel::new(MLFramework::SafeTensors);

        if let Some(metadata) = safetensors.get("metadata") {
            model.metadata = serde_json::from_value(metadata.clone())
                .map_err(|e| IoError::SerializationError(e.to_string()))?;
        }

        if let Some(tensors) = safetensors.get("tensors").and_then(|v| v.as_object()) {
            for (name, tensor_data) in tensors {
                let shape: Vec<usize> = serde_json::from_value(tensor_data["shape"].clone())
                    .map_err(|e| IoError::SerializationError(e.to_string()))?;

                let data: Vec<f32> = serde_json::from_value(tensor_data["data"].clone())
                    .map_err(|e| IoError::SerializationError(e.to_string()))?;

                let array = ArrayD::from_shape_vec(IxDyn(&shape), data)
                    .map_err(|e| IoError::Other(e.to_string()))?;

                model
                    .weights
                    .insert(name.clone(), MLTensor::new(array, Some(name.clone())));
            }
        }

        Ok(model)
    }

    fn save_tensor(&self, tensor: &MLTensor, path: &Path) -> Result<()> {
        let tensor_data = serde_json::json!({
            "shape": tensor.metadata.shape,
            "dtype": format!("{:?}", tensor.metadata.dtype),
            "data": tensor.data.as_slice().unwrap().to_vec(),
        });

        let file = File::create(path).map_err(IoError::Io)?;
        serde_json::to_writer(file, &tensor_data)
            .map_err(|e| IoError::SerializationError(e.to_string()))
    }

    fn load_tensor(&self, path: &Path) -> Result<MLTensor> {
        let file = File::open(path).map_err(IoError::Io)?;
        let tensor_data: serde_json::Value = serde_json::from_reader(file)
            .map_err(|e| IoError::SerializationError(e.to_string()))?;

        let shape: Vec<usize> = serde_json::from_value(tensor_data["shape"].clone())
            .map_err(|e| IoError::SerializationError(e.to_string()))?;

        let data: Vec<f32> = serde_json::from_value(tensor_data["data"].clone())
            .map_err(|e| IoError::SerializationError(e.to_string()))?;

        let array = ArrayD::from_shape_vec(IxDyn(&shape), data)
            .map_err(|e| IoError::Other(e.to_string()))?;

        Ok(MLTensor::new(array, None))
    }
}
