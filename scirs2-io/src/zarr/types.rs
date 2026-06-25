//! Zarr v3 types: data types, compressors, array metadata, and array structs.

use serde::{Deserialize, Serialize};

/// Supported Zarr data types.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ZarrDataType {
    /// 32-bit floating point
    Float32,
    /// 64-bit floating point
    Float64,
    /// 32-bit signed integer
    Int32,
    /// 64-bit signed integer
    Int64,
    /// 8-bit unsigned integer
    UInt8,
    /// 16-bit unsigned integer
    UInt16,
    /// 32-bit unsigned integer
    UInt32,
}

impl ZarrDataType {
    /// Returns the number of bytes per element.
    pub fn byte_size(&self) -> usize {
        match self {
            ZarrDataType::Float32 => 4,
            ZarrDataType::Float64 => 8,
            ZarrDataType::Int32 => 4,
            ZarrDataType::Int64 => 8,
            ZarrDataType::UInt8 => 1,
            ZarrDataType::UInt16 => 2,
            ZarrDataType::UInt32 => 4,
        }
    }

    /// Returns the Zarr v3 data type string.
    pub fn as_str(&self) -> &'static str {
        match self {
            ZarrDataType::Float32 => "float32",
            ZarrDataType::Float64 => "float64",
            ZarrDataType::Int32 => "int32",
            ZarrDataType::Int64 => "int64",
            ZarrDataType::UInt8 => "uint8",
            ZarrDataType::UInt16 => "uint16",
            ZarrDataType::UInt32 => "uint32",
        }
    }
}

/// Zarr compressor configuration.
///
/// Actual compression is handled via oxiarc-* crates (COOLJAPAN Policy).
/// For `None`, raw bytes are stored.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ZarrCompressor {
    /// No compression — store raw bytes.
    None,
    /// Blosc compression (placeholder; compression via oxiarc).
    Blosc {
        /// Blosc codec name (e.g., "lz4", "zstd").
        cname: String,
        /// Compression level (1–9).
        clevel: u8,
    },
    /// GZip/Deflate compression via oxiarc-deflate.
    GZip {
        /// Compression level (1–9).
        level: u8,
    },
    /// Zstandard compression via oxiarc-zstd.
    Zstd {
        /// Compression level (1–22).
        level: u8,
    },
}

/// Zarr v3 array metadata (zarr.json).
#[non_exhaustive]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZarrArrayMeta {
    /// Shape of the array.
    pub shape: Vec<usize>,
    /// Chunk shape (must have same length as `shape`).
    pub chunks: Vec<usize>,
    /// Element data type.
    pub dtype: ZarrDataType,
    /// Compressor configuration.
    pub compressor: ZarrCompressor,
    /// Fill value for missing/unwritten chunks.
    pub fill_value: f64,
    /// Zarr format version (must be 3).
    pub zarr_format: u8,
    /// Separator character used in chunk keys ('/' for v3).
    pub dimension_separator: char,
}

impl Default for ZarrArrayMeta {
    fn default() -> Self {
        Self {
            shape: vec![],
            chunks: vec![],
            dtype: ZarrDataType::Float64,
            compressor: ZarrCompressor::None,
            fill_value: 0.0,
            zarr_format: 3,
            dimension_separator: '/',
        }
    }
}

/// An in-memory Zarr array with flat f64 storage.
pub struct ZarrArray {
    /// Array metadata.
    pub meta: ZarrArrayMeta,
    /// Flat data buffer in row-major order.
    pub(crate) data: Vec<f64>,
}

impl ZarrArray {
    /// Create a new ZarrArray with given metadata and flat data.
    pub fn new(meta: ZarrArrayMeta, data: Vec<f64>) -> Self {
        Self { meta, data }
    }

    /// Total number of elements.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` if the array contains no elements.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Access data as a flat slice.
    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }
}
