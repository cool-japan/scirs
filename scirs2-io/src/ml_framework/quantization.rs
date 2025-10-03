//! Model quantization support

use crate::error::{IoError, Result};
use crate::ml_framework::{MLModel, MLTensor, ModelMetadata, TensorMetadata};
use scirs2_core::ndarray::{ArrayD, IxDyn};
use std::collections::HashMap;

/// Quantization methods
#[derive(Debug, Clone, Copy)]
pub enum QuantizationMethod {
    /// Dynamic quantization
    Dynamic,
    /// Static quantization with calibration
    Static,
    /// Quantization-aware training
    QAT,
    /// Post-training quantization
    PTQ,
}

/// Quantized tensor
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    pub data: Vec<u8>,
    pub scale: f32,
    pub zero_point: i32,
    pub metadata: TensorMetadata,
}

impl QuantizedTensor {
    /// Quantize a floating-point tensor
    pub fn from_float_tensor(tensor: &MLTensor, bits: u8) -> Result<Self> {
        let data = tensor.data.as_slice().unwrap();
        let (min_val, max_val) = data
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), &x| {
                (min.min(x), max.max(x))
            });

        let qmax = (1 << bits) - 1;
        let scale = (max_val - min_val) / qmax as f32;
        let zero_point = (-min_val / scale).round() as i32;

        let quantized: Vec<u8> = data
            .iter()
            .map(|&x| (x / scale + zero_point as f32).round() as u8)
            .collect();

        Ok(Self {
            data: quantized,
            scale,
            zero_point,
            metadata: tensor.metadata.clone(),
        })
    }

    /// Dequantize to floating-point
    pub fn to_float_tensor(&self) -> Result<MLTensor> {
        let data: Vec<f32> = self
            .data
            .iter()
            .map(|&q| (q as i32 - self.zero_point) as f32 * self.scale)
            .collect();

        let array = ArrayD::from_shape_vec(IxDyn(&self.metadata.shape), data)
            .map_err(|e| IoError::Other(e.to_string()))?;

        Ok(MLTensor::new(array, self.metadata.name.clone()))
    }
}

/// Model quantizer
pub struct ModelQuantizer {
    method: QuantizationMethod,
    bits: u8,
}

impl ModelQuantizer {
    pub fn new(method: QuantizationMethod, bits: u8) -> Self {
        Self { method, bits }
    }

    /// Quantize entire model
    pub fn quantize_model(&self, model: &MLModel) -> Result<QuantizedModel> {
        let mut quantized_weights = HashMap::new();

        for (name, tensor) in &model.weights {
            let quantized = QuantizedTensor::from_float_tensor(tensor, self.bits)?;
            quantized_weights.insert(name.clone(), quantized);
        }

        Ok(QuantizedModel {
            metadata: model.metadata.clone(),
            weights: quantized_weights,
            config: model.config.clone(),
            quantization_info: QuantizationInfo {
                method: self.method,
                bits: self.bits,
            },
        })
    }
}

/// Quantized model
#[derive(Debug, Clone)]
pub struct QuantizedModel {
    pub metadata: ModelMetadata,
    pub weights: HashMap<String, QuantizedTensor>,
    pub config: HashMap<String, serde_json::Value>,
    pub quantization_info: QuantizationInfo,
}

#[derive(Debug, Clone)]
pub struct QuantizationInfo {
    pub method: QuantizationMethod,
    pub bits: u8,
}
