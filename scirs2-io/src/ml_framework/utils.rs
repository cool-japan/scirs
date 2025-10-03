//! Utility functions for ML framework operations
#![allow(dead_code)]

use crate::error::{IoError, Result};
use crate::ml_framework::types::{DataType, MLTensor};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use scirs2_core::ndarray::{ArrayD, IxDyn};
use std::fs::File;
use std::io::Read;

/// Convert tensor to Python-compatible dictionary
pub fn tensor_to_python_dict(tensor: &MLTensor) -> Result<serde_json::Value> {
    Ok(serde_json::json!({
        "data": tensor.data.as_slice().unwrap().to_vec(),
        "shape": tensor.metadata.shape,
        "dtype": format!("{:?}", tensor.metadata.dtype),
        "requires_grad": tensor.metadata.requires_grad,
    }))
}

/// Convert Python dictionary to tensor
pub fn python_dict_to_tensor(dict: &serde_json::Value) -> Result<MLTensor> {
    let shape: Vec<usize> = serde_json::from_value(dict["shape"].clone())
        .map_err(|e| IoError::SerializationError(e.to_string()))?;

    let data: Vec<f32> = serde_json::from_value(dict["data"].clone())
        .map_err(|e| IoError::SerializationError(e.to_string()))?;

    let array =
        ArrayD::from_shape_vec(IxDyn(&shape), data).map_err(|e| IoError::Other(e.to_string()))?;

    let mut tensor = MLTensor::new(array, None);

    if let Some(requires_grad) = dict.get("requires_grad").and_then(|v| v.as_bool()) {
        tensor.metadata.requires_grad = requires_grad;
    }

    Ok(tensor)
}

/// SafeTensors header structure
#[derive(Debug, Clone)]
pub struct SafeTensorsHeader {
    pub tensors: std::collections::HashMap<String, TensorInfo>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TensorInfo {
    pub dtype: DataType,
    pub shape: Vec<usize>,
    pub data_offsets: (usize, usize),
}

/// Write SafeTensors header
pub fn write_safetensors_header<W: std::io::Write>(
    writer: &mut W,
    header: &SafeTensorsHeader,
) -> Result<()> {
    let header_json = serde_json::to_string(&header.tensors)
        .map_err(|e| IoError::SerializationError(e.to_string()))?;

    writer
        .write_u64::<LittleEndian>(header_json.len() as u64)
        .map_err(IoError::Io)?;
    writer
        .write_all(header_json.as_bytes())
        .map_err(IoError::Io)?;

    Ok(())
}

/// Read SafeTensors header
pub fn read_safetensors_header<R: std::io::Read>(reader: &mut R) -> Result<SafeTensorsHeader> {
    let header_size = reader.read_u64::<LittleEndian>().map_err(IoError::Io)?;
    let mut header_bytes = vec![0u8; header_size as usize];
    reader.read_exact(&mut header_bytes).map_err(IoError::Io)?;

    let header_str = String::from_utf8(header_bytes).map_err(|e| IoError::Other(e.to_string()))?;

    let tensors: std::collections::HashMap<String, TensorInfo> = serde_json::from_str(&header_str)
        .map_err(|e| IoError::SerializationError(e.to_string()))?;

    Ok(SafeTensorsHeader { tensors })
}
