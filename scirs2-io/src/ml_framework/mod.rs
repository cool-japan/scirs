//! Machine learning framework compatibility layer
//!
//! Provides conversion utilities and interfaces for seamless integration with
//! popular machine learning frameworks, enabling easy data exchange and model I/O.

#![allow(dead_code)]
#![allow(missing_docs)]

pub mod batch_processing;
pub mod converters;
pub mod datasets;
pub mod optimization;
pub mod quantization;
pub mod serving;
pub mod types;
pub mod utils;
pub mod validation;

// Re-export core types for backward compatibility
pub use batch_processing::{BatchProcessor, DataLoader};
pub use converters::{
    get_converter, CoreMLConverter, HuggingFaceConverter, JAXConverter, MLFrameworkConverter,
    MXNetConverter, ONNXConverter, PyTorchConverter, SafeTensorsConverter, TensorFlowConverter,
};
pub use datasets::MLDataset;
pub use optimization::{ModelOptimizer, OptimizationTechnique};
pub use quantization::{ModelQuantizer, QuantizationMethod, QuantizedModel, QuantizedTensor};
pub use serving::{
    ApiConfig, HealthStatus, InferenceRequest, InferenceResponse, LoadBalancer, ModelInfo,
    ModelServer, ResponseStatus, ServerConfig, ServerMetrics,
};
pub use types::{DataType, MLFramework, MLModel, MLTensor, ModelMetadata, TensorMetadata};
pub use validation::{BatchValidator, ModelValidator, ValidationConfig, ValidationReport};

// Import required dependencies for the remaining modules
use crate::error::{IoError, Result};
use scirs2_core::ndarray::{Array2, ArrayD, ArrayView2, IxDyn};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

// Additional imports for async support
#[cfg(feature = "async")]
use tokio::{
    fs,
    sync::{Mutex, RwLock},
    time::{sleep, Duration, Instant},
};

// Additional imports for binary operations
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

// Additional imports for networking
#[cfg(feature = "async")]
use std::collections::VecDeque;

// Include remaining modules from the parent file
// Note: The following modules are not available in the root and have been commented out
// pub use super::converters as common_converters;
// pub use super::serving;
// pub use super::model_hub;
// pub use super::pytorch_enhanced;
// pub use super::tensorflow_enhanced;
// pub use super::onnx_enhanced;
// pub use super::versioning;
// pub use super::error_handling;
