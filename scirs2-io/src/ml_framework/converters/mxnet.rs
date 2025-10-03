//! MXNet format converter
#![allow(dead_code)]

use crate::error::{IoError, Result};
use crate::ml_framework::converters::MLFrameworkConverter;
use crate::ml_framework::types::{MLFramework, MLModel, MLTensor};
use scirs2_core::ndarray::{ArrayD, IxDyn};
use std::fs::File;
use std::path::Path;

/// MXNet format converter
pub struct MXNetConverter;

impl MLFrameworkConverter for MXNetConverter {
    fn save_model(&self, model: &MLModel, path: &Path) -> Result<()> {
        // MXNet uses symbol and params files
        let mxnet_model = serde_json::json!({
            "format": "mxnet",
            "version": "1.9.0",
            "symbol": {
                "nodes": [],
                "arg_nodes": model.weights.keys().enumerate().map(|(i, name)| i).collect::<Vec<_>>(),
                "node_row_ptr": [0, model.weights.len()],
                "attrs": {
                    "mxnet_version": ["1.9.0", "int"]
                }
            },
            "params": model.weights.iter().map(|(name, tensor)| {
                (name.clone(), serde_json::json!({
                    "shape": tensor.metadata.shape,
                    "dtype": format!("{:?}", tensor.metadata.dtype),
                    "data": tensor.data.as_slice().unwrap().to_vec()
                }))
            }).collect::<serde_json::Map<String, serde_json::Value>>(),
            "metadata": model.metadata
        });

        let file = File::create(path).map_err(IoError::Io)?;
        serde_json::to_writer_pretty(file, &mxnet_model)
            .map_err(|e| IoError::SerializationError(e.to_string()))
    }

    fn load_model(&self, path: &Path) -> Result<MLModel> {
        let file = File::open(path).map_err(IoError::Io)?;
        let mxnet_model: serde_json::Value = serde_json::from_reader(file)
            .map_err(|e| IoError::SerializationError(e.to_string()))?;

        let mut model = MLModel::new(MLFramework::MXNet);

        if let Some(metadata) = mxnet_model.get("metadata") {
            model.metadata = serde_json::from_value(metadata.clone())
                .map_err(|e| IoError::SerializationError(e.to_string()))?;
        }

        if let Some(params) = mxnet_model.get("params").and_then(|v| v.as_object()) {
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

        Ok(model)
    }

    fn save_tensor(&self, tensor: &MLTensor, path: &Path) -> Result<()> {
        let tensor_data = serde_json::json!({
            "mxnet_ndarray": {
                "shape": tensor.metadata.shape,
                "dtype": format!("{:?}", tensor.metadata.dtype),
                "data": tensor.data.as_slice().unwrap().to_vec(),
                "context": "cpu(0)"
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

        if let Some(ndarray) = tensor_data.get("mxnet_ndarray") {
            let shape: Vec<usize> = serde_json::from_value(ndarray["shape"].clone())
                .map_err(|e| IoError::SerializationError(e.to_string()))?;

            let data: Vec<f32> = serde_json::from_value(ndarray["data"].clone())
                .map_err(|e| IoError::SerializationError(e.to_string()))?;

            let array = ArrayD::from_shape_vec(IxDyn(&shape), data)
                .map_err(|e| IoError::Other(e.to_string()))?;

            return Ok(MLTensor::new(array, None));
        }

        Err(IoError::Other("Invalid MXNet tensor format".to_string()))
    }
}
