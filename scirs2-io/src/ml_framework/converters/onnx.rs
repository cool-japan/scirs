//! ONNX format converter
#![allow(dead_code)]

use crate::error::{IoError, Result};
use crate::ml_framework::converters::MLFrameworkConverter;
use crate::ml_framework::types::{MLFramework, MLModel, MLTensor};
use scirs2_core::ndarray::{ArrayD, IxDyn};
use std::fs::File;
use std::path::Path;

/// ONNX format converter
pub struct ONNXConverter;

impl MLFrameworkConverter for ONNXConverter {
    fn save_model(&self, model: &MLModel, path: &Path) -> Result<()> {
        // Simplified ONNX-like format
        let onnx_model = serde_json::json!({
            "format": "onnx",
            "version": "1.0",
            "graph": {
                "name": model.metadata.model_name,
                "inputs": model.metadata.inputshapes,
                "outputs": model.metadata.outputshapes,
                "initializers": model.weights.iter().map(|(name, tensor)| {
                    serde_json::json!({
                        "name": name,
                        "shape": tensor.metadata.shape,
                        "dtype": tensor.metadata.dtype,
                    })
                }).collect::<Vec<_>>(),
            },
            "metadata": model.metadata,
        });

        let file = File::create(path).map_err(IoError::Io)?;
        serde_json::to_writer_pretty(file, &onnx_model)
            .map_err(|e| IoError::SerializationError(e.to_string()))
    }

    fn load_model(&self, path: &Path) -> Result<MLModel> {
        let file = File::open(path).map_err(IoError::Io)?;
        let onnx_model: serde_json::Value = serde_json::from_reader(file)
            .map_err(|e| IoError::SerializationError(e.to_string()))?;

        let mut model = MLModel::new(MLFramework::ONNX);

        // Parse ONNX model metadata
        if let Some(graph) = onnx_model.get("graph") {
            if let Some(name) = graph.get("name").and_then(|v| v.as_str()) {
                model.metadata.model_name = Some(name.to_string());
            }

            // Parse inputs and outputs
            if let Some(inputs) = graph.get("inputs").and_then(|v| v.as_object()) {
                for (name, shape_val) in inputs {
                    if let Some(shape) = shape_val.as_array() {
                        let shape_vec: Vec<usize> = shape
                            .iter()
                            .filter_map(|v| v.as_u64().map(|u| u as usize))
                            .collect();
                        model.metadata.inputshapes.insert(name.clone(), shape_vec);
                    }
                }
            }

            if let Some(outputs) = graph.get("outputs").and_then(|v| v.as_object()) {
                for (name, shape_val) in outputs {
                    if let Some(shape) = shape_val.as_array() {
                        let shape_vec: Vec<usize> = shape
                            .iter()
                            .filter_map(|v| v.as_u64().map(|u| u as usize))
                            .collect();
                        model.metadata.outputshapes.insert(name.clone(), shape_vec);
                    }
                }
            }

            // Parse initializers (weights)
            if let Some(initializers) = graph.get("initializers").and_then(|v| v.as_array()) {
                for init in initializers {
                    if let Some(init_obj) = init.as_object() {
                        if let (Some(name), Some(shape), Some(_dtype)) = (
                            init_obj.get("name").and_then(|v| v.as_str()),
                            init_obj.get("shape").and_then(|v| v.as_array()),
                            init_obj.get("dtype"),
                        ) {
                            let shape_vec: Vec<usize> = shape
                                .iter()
                                .filter_map(|v| v.as_u64().map(|u| u as usize))
                                .collect();

                            // Read actual tensor data from the JSON
                            let data = if let Some(data_array) =
                                init_obj.get("data").and_then(|v| v.as_array())
                            {
                                // Extract actual data values
                                data_array
                                    .iter()
                                    .filter_map(|v| v.as_f64().map(|f| f as f32))
                                    .collect::<Vec<f32>>()
                            } else {
                                // Fallback to zeros if no data is provided
                                let total_elements: usize = shape_vec.iter().product();
                                vec![0.0f32; total_elements]
                            };

                            if let Ok(array) = ArrayD::from_shape_vec(IxDyn(&shape_vec), data) {
                                model.weights.insert(
                                    name.to_string(),
                                    MLTensor::new(array, Some(name.to_string())),
                                );
                            }
                        }
                    }
                }
            }
        }

        Ok(model)
    }

    fn save_tensor(&self, tensor: &MLTensor, path: &Path) -> Result<()> {
        let tensor_data = serde_json::json!({
            "name": tensor.metadata.name,
            "shape": tensor.metadata.shape,
            "dtype": "float32",
            "data": tensor.data.as_slice().unwrap().to_vec(),
        });

        let file = File::create(path).map_err(IoError::Io)?;
        serde_json::to_writer_pretty(file, &tensor_data)
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

        let name = tensor_data
            .get("name")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let array = ArrayD::from_shape_vec(IxDyn(&shape), data)
            .map_err(|e| IoError::Other(e.to_string()))?;

        Ok(MLTensor::new(array, name))
    }
}
