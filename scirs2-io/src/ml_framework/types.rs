//! Core types for ML framework compatibility
#![allow(dead_code)]
#![allow(missing_docs)]

use scirs2_core::ndarray::{ArrayD, IxDyn};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Supported ML framework formats
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MLFramework {
    /// PyTorch tensor format
    PyTorch,
    /// TensorFlow SavedModel format
    TensorFlow,
    /// ONNX (Open Neural Network Exchange) format
    ONNX,
    /// Core ML format (Apple)
    CoreML,
    /// JAX format
    JAX,
    /// MXNet format
    MXNet,
    /// Hugging Face format
    HuggingFace,
    /// SafeTensors format
    SafeTensors,
}

/// Tensor metadata for ML frameworks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorMetadata {
    pub name: Option<String>,
    pub shape: Vec<usize>,
    pub dtype: DataType,
    pub device: Option<String>,
    pub requires_grad: bool,
    pub is_parameter: bool,
}

/// Data types supported by ML frameworks
#[derive(Debug, Clone, Copy, Hash, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum DataType {
    Float32,
    Float64,
    Float16,
    BFloat16,
    Int32,
    Int64,
    Int16,
    Int8,
    UInt8,
    Bool,
}

impl DataType {
    /// Get byte size of the data type
    pub fn byte_size(&self) -> usize {
        match self {
            Self::Float64 | Self::Int64 => 8,
            Self::Float32 | Self::Int32 => 4,
            Self::Float16 | Self::BFloat16 | Self::Int16 => 2,
            Self::Int8 | Self::UInt8 | Self::Bool => 1,
        }
    }
}

/// Model metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub framework: String,
    pub framework_version: Option<String>,
    pub model_name: Option<String>,
    pub model_version: Option<String>,
    pub architecture: Option<String>,
    pub inputshapes: HashMap<String, Vec<usize>>,
    pub outputshapes: HashMap<String, Vec<usize>>,
    pub parameters: HashMap<String, serde_json::Value>,
}

/// ML tensor container
#[derive(Debug, Clone)]
pub struct MLTensor {
    pub data: ArrayD<f32>,
    pub metadata: TensorMetadata,
}

impl MLTensor {
    /// Create new ML tensor
    pub fn new(data: ArrayD<f32>, name: Option<String>) -> Self {
        let shape = data.shape().to_vec();
        Self {
            data,
            metadata: TensorMetadata {
                name,
                shape,
                dtype: DataType::Float32,
                device: None,
                requires_grad: false,
                is_parameter: false,
            },
        }
    }

    /// Convert to different data type
    pub fn to_dtype(&self, dtype: DataType) -> crate::error::Result<Self> {
        use crate::error::IoError;
        // For simplicity, we'll just handle float conversions
        match dtype {
            DataType::Float32 => Ok(self.clone()),
            DataType::Float64 => {
                let data = self.data.mapv(|x| x as f64);
                Ok(Self {
                    data: data.mapv(|x| x as f32).into_dyn(),
                    metadata: TensorMetadata {
                        dtype,
                        ..self.metadata.clone()
                    },
                })
            }
            _ => Err(IoError::UnsupportedFormat(format!(
                "Unsupported dtype conversion: {:?}",
                dtype
            ))),
        }
    }
}

/// ML model container
#[derive(Clone)]
pub struct MLModel {
    pub metadata: ModelMetadata,
    pub weights: HashMap<String, MLTensor>,
    pub config: HashMap<String, serde_json::Value>,
}

impl MLModel {
    /// Create new ML model
    pub fn new(framework: MLFramework) -> Self {
        Self {
            metadata: ModelMetadata {
                framework: format!("{:?}", framework),
                framework_version: None,
                model_name: None,
                model_version: None,
                architecture: None,
                inputshapes: HashMap::new(),
                outputshapes: HashMap::new(),
                parameters: HashMap::new(),
            },
            weights: HashMap::new(),
            config: HashMap::new(),
        }
    }

    /// Add weight tensor
    pub fn add_weight(&mut self, name: impl Into<String>, tensor: MLTensor) {
        self.weights.insert(name.into(), tensor);
    }

    /// Get weight tensor
    pub fn get_weight(&self, name: &str) -> Option<&MLTensor> {
        self.weights.get(name)
    }

    /// Save model to file
    pub fn save(
        &self,
        framework: MLFramework,
        path: impl AsRef<std::path::Path>,
    ) -> crate::error::Result<()> {
        use crate::ml_framework::converters::get_converter;
        let converter = get_converter(framework);
        converter.save_model(self, path.as_ref())
    }

    /// Load model from file
    pub fn load(
        framework: MLFramework,
        path: impl AsRef<std::path::Path>,
    ) -> crate::error::Result<Self> {
        use crate::ml_framework::converters::get_converter;
        let converter = get_converter(framework);
        converter.load_model(path.as_ref())
    }
}
