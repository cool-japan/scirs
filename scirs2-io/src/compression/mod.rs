//! Compression utilities for scientific data
//!
//! This module provides functionality for compressing and decompressing data using
//! various algorithms suitable for scientific computing. It focuses on lossless
//! compression to ensure data integrity while reducing storage requirements.
//!
//! ## Features
//!
//! - Multiple compression algorithms (GZIP, ZSTD, LZ4, BZIP2)
//! - Configurable compression levels
//! - Memory-efficient compression of large datasets
//! - Metadata preservation during compression
//! - Array-specific compression optimizations
//!
//! ## Sub-modules
//!
//! - `ndarray`: Specialized compression utilities for ndarray types
//!
//! ## Examples
//!
//! ```rust,no_run
//! use scirs2_io::compression::{compress_data, decompress_data, CompressionAlgorithm};
//! use std::fs::File;
//! use std::io::prelude::*;
//!
//! // Compress some data using ZSTD with default compression level
//! let data = b"Large scientific dataset with repetitive patterns";
//! let compressed = compress_data(data, CompressionAlgorithm::Zstd, None).unwrap();
//!
//! // Save the compressed data to a file
//! let mut file = File::create("data.zst").unwrap();
//! file.write_all(&compressed).unwrap();
//!
//! // Later, read and decompress the data
//! let mut compressed_data = Vec::new();
//! File::open("data.zst").unwrap().read_to_end(&mut compressed_data).unwrap();
//! let original = decompress_data(&compressed_data, CompressionAlgorithm::Zstd).unwrap();
//! assert_eq!(original, data);
//! ```

use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

use brotli::{enc::BrotliEncoderParams, BrotliCompress, BrotliDecompress};
use bzip2::read::{BzDecoder, BzEncoder};
use bzip2::Compression as Bzip2Compression;
use flate2::read::{GzDecoder, GzEncoder};
use flate2::Compression as GzipCompression;
use lz4::{Decoder, EncoderBuilder};
use snap::{raw::Decoder as SnapDecoder, raw::Encoder as SnapEncoder};
use zstd::{decode_all, encode_all};

// Re-export ndarray submodule
pub mod ndarray;

use crate::error::{IoError, Result};

/// Compression algorithms supported by the library
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionAlgorithm {
    /// GZIP compression (good balance of speed and compression ratio)
    Gzip,
    /// Zstandard compression (excellent compression ratio, fast decompression)
    Zstd,
    /// LZ4 compression (extremely fast, moderate compression ratio)
    Lz4,
    /// BZIP2 compression (high compression ratio, slower speed)
    Bzip2,
    /// Brotli compression (excellent compression ratio, web-optimized)
    Brotli,
    /// Snappy compression (very fast, Google-developed)
    Snappy,
    /// Floating-point specific compression optimized for scientific data
    FpZip,
    /// Delta + LZ4 compression for time series data
    DeltaLz4,
}

impl CompressionAlgorithm {
    /// Get the file extension associated with this compression algorithm
    pub fn extension(&self) -> &'static str {
        match self {
            CompressionAlgorithm::Gzip => "gz",
            CompressionAlgorithm::Zstd => "zst",
            CompressionAlgorithm::Lz4 => "lz4",
            CompressionAlgorithm::Bzip2 => "bz2",
            CompressionAlgorithm::Brotli => "br",
            CompressionAlgorithm::Snappy => "snappy",
            CompressionAlgorithm::FpZip => "fpz",
            CompressionAlgorithm::DeltaLz4 => "dlz4",
        }
    }

    /// Try to determine the compression algorithm from a file extension
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "gz" | "gzip" => Some(CompressionAlgorithm::Gzip),
            "zst" | "zstd" => Some(CompressionAlgorithm::Zstd),
            "lz4" => Some(CompressionAlgorithm::Lz4),
            "bz2" | "bzip2" => Some(CompressionAlgorithm::Bzip2),
            "br" | "brotli" => Some(CompressionAlgorithm::Brotli),
            "snappy" | "snp" => Some(CompressionAlgorithm::Snappy),
            "fpz" | "fpzip" => Some(CompressionAlgorithm::FpZip),
            "dlz4" | "delta-lz4" => Some(CompressionAlgorithm::DeltaLz4),
            _ => None,
        }
    }
}

/// Convert a compression level (0-9) to the appropriate internal level for each algorithm
#[allow(dead_code)]
fn normalize_level(level: Option<u32>, algorithm: CompressionAlgorithm) -> Result<u32> {
    let _level = level.unwrap_or(6); // Default compression level

    if _level > 9 {
        return Err(IoError::CompressionError(format!(
            "Compression level must be between 0 and 9, got {_level}"
        )));
    }

    // Each compression library has different ranges for compression levels
    match algorithm {
        CompressionAlgorithm::Gzip => Ok(_level),
        CompressionAlgorithm::Zstd => {
            // ZSTD supports levels 1-22, map our 0-9 to 1-22
            Ok(1 + (_level * 21) / 9)
        }
        CompressionAlgorithm::Lz4 => Ok(_level),
        CompressionAlgorithm::Bzip2 => Ok(_level),
        CompressionAlgorithm::Brotli => {
            // Brotli supports levels 0-11, map our 0-9 to 0-11
            Ok((_level * 11) / 9)
        }
        CompressionAlgorithm::Snappy => Ok(0), // Snappy has no compression levels
        CompressionAlgorithm::FpZip => Ok(_level), // Will implement custom handling
        CompressionAlgorithm::DeltaLz4 => Ok(_level), // Uses LZ4 internally
    }
}

/// Compress data using the specified algorithm and compression level
///
/// # Arguments
///
/// * `data` - The data to compress
/// * `algorithm` - The compression algorithm to use
/// * `level` - The compression level (0-9, where 0 is no compression, 9 is maximum compression)
///
/// # Returns
///
/// The compressed data as a `Vec<u8>`
#[allow(dead_code)]
pub fn compress_data(
    mut data: &[u8],
    algorithm: CompressionAlgorithm,
    level: Option<u32>,
) -> Result<Vec<u8>> {
    let normalized_level = normalize_level(level, algorithm)?;

    match algorithm {
        CompressionAlgorithm::Gzip => {
            let mut encoder = GzEncoder::new(data, GzipCompression::new(normalized_level as u32));
            let mut compressed = Vec::new();
            encoder
                .read_to_end(&mut compressed)
                .map_err(|e| IoError::CompressionError(e.to_string()))?;
            Ok(compressed)
        }
        CompressionAlgorithm::Zstd => encode_all(data, normalized_level as i32)
            .map_err(|e| IoError::CompressionError(e.to_string())),
        CompressionAlgorithm::Lz4 => {
            let mut encoder = EncoderBuilder::new()
                .level(normalized_level as u32)
                .build(Vec::new())
                .map_err(|e| IoError::CompressionError(e.to_string()))?;

            encoder
                .write_all(data)
                .map_err(|e| IoError::CompressionError(e.to_string()))?;

            let (compressed, result) = encoder.finish();
            result.map_err(|e| IoError::CompressionError(e.to_string()))?;

            Ok(compressed)
        }
        CompressionAlgorithm::Bzip2 => {
            let mut encoder = BzEncoder::new(data, Bzip2Compression::new(normalized_level as u32));
            let mut compressed = Vec::new();
            encoder
                .read_to_end(&mut compressed)
                .map_err(|e| IoError::CompressionError(e.to_string()))?;
            Ok(compressed)
        }
        CompressionAlgorithm::Brotli => {
            let mut compressed = Vec::new();
            let params = BrotliEncoderParams {
                quality: normalized_level as i32,
                ..Default::default()
            };
            BrotliCompress(&mut data, &mut compressed, &params).map_err(|e| {
                IoError::CompressionError(format!("Brotli compression failed: {e}"))
            })?;
            Ok(compressed)
        }
        CompressionAlgorithm::Snappy => {
            let mut encoder = SnapEncoder::new();
            let compressed = encoder.compress_vec(data).map_err(|e| {
                IoError::CompressionError(format!("Snappy compression failed: {e:?}"))
            })?;
            Ok(compressed)
        }
        CompressionAlgorithm::FpZip => {
            // Implement floating-point specific compression
            compress_fpzip(data, normalized_level)
        }
        CompressionAlgorithm::DeltaLz4 => {
            // Implement delta + LZ4 compression for time series
            compress_delta_lz4(data, normalized_level)
        }
    }
}

/// Decompress data using the specified algorithm
///
/// # Arguments
///
/// * `data` - The compressed data
/// * `algorithm` - The compression algorithm used
///
/// # Returns
///
/// The decompressed data as a `Vec<u8>`
#[allow(dead_code)]
pub fn decompress_data(mut data: &[u8], algorithm: CompressionAlgorithm) -> Result<Vec<u8>> {
    match algorithm {
        CompressionAlgorithm::Gzip => {
            let mut decoder = GzDecoder::new(data);
            let mut decompressed = Vec::new();
            decoder
                .read_to_end(&mut decompressed)
                .map_err(|e| IoError::DecompressionError(e.to_string()))?;
            Ok(decompressed)
        }
        CompressionAlgorithm::Zstd => {
            decode_all(data).map_err(|e| IoError::DecompressionError(e.to_string()))
        }
        CompressionAlgorithm::Lz4 => {
            let mut decoder =
                Decoder::new(data).map_err(|e| IoError::DecompressionError(e.to_string()))?;
            let mut decompressed = Vec::new();
            decoder
                .read_to_end(&mut decompressed)
                .map_err(|e| IoError::DecompressionError(e.to_string()))?;
            Ok(decompressed)
        }
        CompressionAlgorithm::Bzip2 => {
            let mut decoder = BzDecoder::new(data);
            let mut decompressed = Vec::new();
            decoder
                .read_to_end(&mut decompressed)
                .map_err(|e| IoError::DecompressionError(e.to_string()))?;
            Ok(decompressed)
        }
        CompressionAlgorithm::Brotli => {
            let mut decompressed = Vec::new();
            BrotliDecompress(&mut data, &mut decompressed).map_err(|e| {
                IoError::DecompressionError(format!("Brotli decompression failed: {e}"))
            })?;
            Ok(decompressed)
        }
        CompressionAlgorithm::Snappy => {
            let mut decoder = SnapDecoder::new();
            let decompressed = decoder.decompress_vec(data).map_err(|e| {
                IoError::DecompressionError(format!("Snappy decompression failed: {e:?}"))
            })?;
            Ok(decompressed)
        }
        CompressionAlgorithm::FpZip => {
            // Implement floating-point specific decompression
            decompress_fpzip(data)
        }
        CompressionAlgorithm::DeltaLz4 => {
            // Implement delta + LZ4 decompression for time series
            decompress_delta_lz4(data)
        }
    }
}

/// Compress a file using the specified algorithm and save it to a new file
///
/// # Arguments
///
/// * `input_path` - Path to the file to compress
/// * `output_path` - Path to save the compressed file (if None, appends algorithm extension to input_path)
/// * `algorithm` - The compression algorithm to use
/// * `level` - The compression level (0-9, where 0 is no compression, 9 is maximum compression)
///
/// # Returns
///
/// The path to the compressed file
#[allow(dead_code)]
pub fn compress_file<P: AsRef<Path>>(
    input_path: P,
    output_path: Option<P>,
    algorithm: CompressionAlgorithm,
    level: Option<u32>,
) -> Result<String> {
    // Read input file
    let mut input_data = Vec::new();
    File::open(input_path.as_ref())
        .map_err(|e| IoError::FileError(format!("Failed to open input file: {e}")))?
        .read_to_end(&mut input_data)
        .map_err(|e| IoError::FileError(format!("Failed to read input file: {e}")))?;

    // Compress the data
    let compressed_data = compress_data(&input_data, algorithm, level)?;

    // Determine output _path
    let output_path_string = match output_path {
        Some(path) => path.as_ref().to_string_lossy().to_string(),
        None => {
            // Generate output _path by appending algorithm extension
            let mut path_buf = input_path.as_ref().to_path_buf();
            let ext = algorithm.extension();

            // Get the file name as a string
            let file_name = path_buf
                .file_name()
                .ok_or_else(|| IoError::FileError("Invalid input file _path".to_string()))?
                .to_string_lossy()
                .to_string();

            // Append the extension and update the file name
            let new_file_name = format!("{file_name}.{ext}");
            path_buf.set_file_name(new_file_name);

            path_buf.to_string_lossy().to_string()
        }
    };

    // Write the compressed data to the output file
    File::create(&output_path_string)
        .map_err(|e| IoError::FileError(format!("Failed to create output file: {e}")))?
        .write_all(&compressed_data)
        .map_err(|e| IoError::FileError(format!("Failed to write to output file: {e}")))?;

    Ok(output_path_string)
}

/// Decompress a file using the specified algorithm and save it to a new file
///
/// # Arguments
///
/// * `input_path` - Path to the compressed file
/// * `output_path` - Path to save the decompressed file (if None, removes algorithm extension from input_path)
/// * `algorithm` - The compression algorithm to use (if None, tries to determine from file extension)
///
/// # Returns
///
/// The path to the decompressed file
#[allow(dead_code)]
pub fn decompress_file<P: AsRef<Path>>(
    input_path: P,
    output_path: Option<P>,
    algorithm: Option<CompressionAlgorithm>,
) -> Result<String> {
    // Determine the compression algorithm
    let algorithm = match algorithm {
        Some(algo) => algo,
        None => {
            // Try to determine from the file extension
            let ext = input_path
                .as_ref()
                .extension()
                .ok_or_else(|| {
                    IoError::DecompressionError("Unable to determine file extension".to_string())
                })?
                .to_string_lossy()
                .to_string();

            CompressionAlgorithm::from_extension(&ext)
                .ok_or(IoError::UnsupportedCompressionAlgorithm(ext))?
        }
    };

    // Read input file
    let mut input_data = Vec::new();
    File::open(input_path.as_ref())
        .map_err(|e| IoError::FileError(format!("Failed to open input file: {e}")))?
        .read_to_end(&mut input_data)
        .map_err(|e| IoError::FileError(format!("Failed to read input file: {e}")))?;

    // Decompress the data
    let decompressed_data = decompress_data(&input_data, algorithm)?;

    // Determine output _path
    let output_path_string = match output_path {
        Some(path) => path.as_ref().to_string_lossy().to_string(),
        None => {
            // Generate output _path by removing algorithm extension
            let path_str = input_path.as_ref().to_string_lossy().to_string();
            let ext = algorithm.extension();

            if path_str.ends_with(&format!(".{ext}")) {
                // Remove the extension
                path_str[0..path_str.len() - ext.len() - 1].to_string()
            } else {
                // If the extension doesn't match, add a ".decompressed" suffix
                format!("{path_str}.decompressed")
            }
        }
    };

    // Write the decompressed data to the output file
    File::create(&output_path_string)
        .map_err(|e| IoError::FileError(format!("Failed to create output file: {e}")))?
        .write_all(&decompressed_data)
        .map_err(|e| IoError::FileError(format!("Failed to write to output file: {e}")))?;

    Ok(output_path_string)
}

/// Calculate the compression ratio for the given data and algorithm
///
/// # Arguments
///
/// * `data` - The original data
/// * `algorithm` - The compression algorithm to use
/// * `level` - The compression level (0-9, optional)
///
/// # Returns
///
/// The compression ratio (original size / compressed size)
#[allow(dead_code)]
pub fn compression_ratio(
    data: &[u8],
    algorithm: CompressionAlgorithm,
    level: Option<u32>,
) -> Result<f64> {
    let compressed = compress_data(data, algorithm, level)?;
    let original_size = data.len() as f64;
    let compressed_size = compressed.len() as f64;

    // Avoid division by zero
    if compressed_size == 0.0 {
        return Err(IoError::CompressionError(
            "Compressed data has zero size".to_string(),
        ));
    }

    Ok(original_size / compressed_size)
}

/// Information about a compression algorithm
pub struct CompressionInfo {
    /// Name of the compression algorithm
    pub name: String,
    /// Brief description of the algorithm
    pub description: String,
    /// Typical compression ratio for scientific data (higher is better)
    pub typical_compression_ratio: f64,
    /// Relative compression speed (1-10, higher is faster)
    pub compression_speed: u8,
    /// Relative decompression speed (1-10, higher is faster)
    pub decompression_speed: u8,
    /// File extension associated with this compression
    pub file_extension: String,
}

/// Get information about a specific compression algorithm
#[allow(dead_code)]
pub fn algorithm_info(algorithm: CompressionAlgorithm) -> CompressionInfo {
    match algorithm {
        CompressionAlgorithm::Gzip => CompressionInfo {
            name: "GZIP".to_string(),
            description: "General-purpose compression _algorithm with good balance of speed and compression ratio".to_string(),
            typical_compression_ratio: 2.5,
            compression_speed: 6,
            decompression_speed: 7,
            file_extension: "gz".to_string(),
        },
        CompressionAlgorithm::Zstd => CompressionInfo {
            name: "Zstandard".to_string(),
            description: "Modern compression _algorithm with excellent compression ratio and fast decompression".to_string(),
            typical_compression_ratio: 3.2,
            compression_speed: 7,
            decompression_speed: 9,
            file_extension: "zst".to_string(),
        },
        CompressionAlgorithm::Lz4 => CompressionInfo {
            name: "LZ4".to_string(),
            description: "Extremely fast compression _algorithm with moderate compression ratio".to_string(),
            typical_compression_ratio: 1.8,
            compression_speed: 10,
            decompression_speed: 10,
            file_extension: "lz4".to_string(),
        },
        CompressionAlgorithm::Bzip2 => CompressionInfo {
            name: "BZIP2".to_string(),
            description: "High compression ratio but slower speed, good for archival storage".to_string(),
            typical_compression_ratio: 3.5,
            compression_speed: 3,
            decompression_speed: 4,
            file_extension: "bz2".to_string(),
        },
        CompressionAlgorithm::Brotli => CompressionInfo {
            name: "Brotli".to_string(),
            description: "Web-optimized compression _algorithm with excellent compression ratio".to_string(),
            typical_compression_ratio: 3.8,
            compression_speed: 5,
            decompression_speed: 8,
            file_extension: "br".to_string(),
        },
        CompressionAlgorithm::Snappy => CompressionInfo {
            name: "Snappy".to_string(),
            description: "Very fast compression _algorithm developed by Google".to_string(),
            typical_compression_ratio: 1.5,
            compression_speed: 10,
            decompression_speed: 10,
            file_extension: "snappy".to_string(),
        },
        CompressionAlgorithm::FpZip => CompressionInfo {
            name: "FPZip".to_string(),
            description: "Floating-point specific compression optimized for scientific data".to_string(),
            typical_compression_ratio: 4.0,
            compression_speed: 7,
            decompression_speed: 8,
            file_extension: "fpz".to_string(),
        },
        CompressionAlgorithm::DeltaLz4 => CompressionInfo {
            name: "Delta+LZ4".to_string(),
            description: "Delta encoding followed by LZ4 compression, optimized for time series data".to_string(),
            typical_compression_ratio: 5.0,
            compression_speed: 8,
            decompression_speed: 9,
            file_extension: "dlz4".to_string(),
        },
    }
}

/// Magic bytes for different compression formats
const GZIP_MAGIC: &[u8] = &[0x1f, 0x8b];
const ZSTD_MAGIC: &[u8] = &[0x28, 0xb5, 0x2f, 0xfd];
const LZ4_MAGIC: &[u8] = &[0x04, 0x22, 0x4d, 0x18];
const BZIP2_MAGIC: &[u8] = &[0x42, 0x5a, 0x68];
const BROTLI_MAGIC: &[u8] = &[0xce, 0xb2, 0xcf, 0x81]; // Custom magic for our Brotli implementation
const SNAPPY_MAGIC: &[u8] = &[0x73, 0x4e, 0x61, 0x50]; // "sNaP"
const FPZIP_MAGIC: &[u8] = &[0x46, 0x50, 0x5a, 0x49]; // "FPZI"
const DELTA_LZ4_MAGIC: &[u8] = &[0x44, 0x4c, 0x5a, 0x34]; // "DLZ4"

/// Detect compression algorithm from magic bytes
#[allow(dead_code)]
pub fn detect_compression_from_bytes(data: &[u8]) -> Option<CompressionAlgorithm> {
    if data.starts_with(GZIP_MAGIC) {
        Some(CompressionAlgorithm::Gzip)
    } else if data.starts_with(ZSTD_MAGIC) {
        Some(CompressionAlgorithm::Zstd)
    } else if data.starts_with(LZ4_MAGIC) {
        Some(CompressionAlgorithm::Lz4)
    } else if data.starts_with(BZIP2_MAGIC) {
        Some(CompressionAlgorithm::Bzip2)
    } else if data.starts_with(BROTLI_MAGIC) {
        Some(CompressionAlgorithm::Brotli)
    } else if data.starts_with(SNAPPY_MAGIC) {
        Some(CompressionAlgorithm::Snappy)
    } else if data.starts_with(FPZIP_MAGIC) {
        Some(CompressionAlgorithm::FpZip)
    } else if data.starts_with(DELTA_LZ4_MAGIC) {
        Some(CompressionAlgorithm::DeltaLz4)
    } else {
        None
    }
}

/// Transparent file handler that automatically handles compression/decompression
pub struct TransparentFileHandler {
    /// Automatically detect compression from file extension
    pub auto_detect_extension: bool,
    /// Automatically detect compression from file content
    pub auto_detect_content: bool,
    /// Default compression algorithm when creating new files
    pub default_algorithm: CompressionAlgorithm,
    /// Default compression level
    pub default_level: Option<u32>,
}

impl Default for TransparentFileHandler {
    fn default() -> Self {
        Self {
            auto_detect_extension: true,
            auto_detect_content: true,
            default_algorithm: CompressionAlgorithm::Zstd,
            default_level: Some(6),
        }
    }
}

impl TransparentFileHandler {
    /// Create a new transparent file handler with custom settings
    pub fn new(
        auto_detect_extension: bool,
        auto_detect_content: bool,
        default_algorithm: CompressionAlgorithm,
        default_level: Option<u32>,
    ) -> Self {
        Self {
            auto_detect_extension,
            auto_detect_content,
            default_algorithm,
            default_level,
        }
    }

    /// Read a file with automatic decompression
    pub fn read_file<P: AsRef<Path>>(&self, path: P) -> Result<Vec<u8>> {
        let mut file_data = Vec::new();
        File::open(path.as_ref())
            .map_err(|e| IoError::FileError(format!("Failed to open file: {e}")))?
            .read_to_end(&mut file_data)
            .map_err(|e| IoError::FileError(format!("Failed to read file: {e}")))?;

        // Try to detect compression
        let mut algorithm = None;

        // Check file extension first if enabled
        if self.auto_detect_extension {
            if let Some(ext) = path.as_ref().extension() {
                algorithm = CompressionAlgorithm::from_extension(&ext.to_string_lossy());
            }
        }

        // Check content magic bytes if enabled and extension detection failed
        if algorithm.is_none() && self.auto_detect_content {
            algorithm = detect_compression_from_bytes(&file_data);
        }

        // Decompress if compression was detected
        match algorithm {
            Some(algo) => decompress_data(&file_data, algo),
            None => Ok(file_data), // Return as-is if no compression detected
        }
    }

    /// Write a file with automatic compression
    pub fn write_file<P: AsRef<Path>>(&self, path: P, data: &[u8]) -> Result<()> {
        let mut algorithm = None;
        let level = self.default_level;

        // Check if we should compress based on file extension
        if self.auto_detect_extension {
            if let Some(ext) = path.as_ref().extension() {
                algorithm = CompressionAlgorithm::from_extension(&ext.to_string_lossy());
            }
        }

        // Use default algorithm if no compression detected but we want to compress
        if algorithm.is_none() && self.should_compress_by_default(&path) {
            algorithm = Some(self.default_algorithm);
        }

        // Compress data if needed
        let output_data = match algorithm {
            Some(algo) => compress_data(data, algo, level)?,
            None => data.to_vec(),
        };

        // Write to file
        File::create(path.as_ref())
            .map_err(|e| IoError::FileError(format!("Failed to create file: {e}")))?
            .write_all(&output_data)
            .map_err(|e| IoError::FileError(format!("Failed to write file: {e}")))?;

        Ok(())
    }

    /// Determine if file should be compressed by default based on path
    fn should_compress_by_default<P: AsRef<Path>>(&self, path: P) -> bool {
        // Don't compress if file already has a compression extension
        if let Some(ext) = path.as_ref().extension() {
            let ext_str = ext.to_string_lossy().to_lowercase();
            matches!(
                ext_str.as_str(),
                "gz" | "gzip" | "zst" | "zstd" | "lz4" | "bz2" | "bzip2"
            )
        } else {
            false
        }
    }

    /// Copy a file with transparent compression/decompression
    pub fn copy_file<P: AsRef<Path>, Q: AsRef<Path>>(
        &self,
        source: P,
        destination: Q,
    ) -> Result<()> {
        let data = self.read_file(source)?;
        self.write_file(destination, &data)?;
        Ok(())
    }

    /// Get file info including compression details
    pub fn file_info<P: AsRef<Path>>(&self, path: P) -> Result<FileCompressionInfo> {
        let mut file_data = Vec::new();
        File::open(path.as_ref())
            .map_err(|e| IoError::FileError(format!("Failed to open file: {e}")))?
            .read_to_end(&mut file_data)
            .map_err(|e| IoError::FileError(format!("Failed to read file: {e}")))?;

        let original_size = file_data.len();

        // Detect compression
        let detected_algorithm = if self.auto_detect_content {
            detect_compression_from_bytes(&file_data)
        } else {
            None
        };

        let extension_algorithm = if self.auto_detect_extension {
            path.as_ref()
                .extension()
                .and_then(|ext| CompressionAlgorithm::from_extension(&ext.to_string_lossy()))
        } else {
            None
        };

        let is_compressed = detected_algorithm.is_some() || extension_algorithm.is_some();
        let algorithm = detected_algorithm.or(extension_algorithm);

        let uncompressed_size = if let Some(algo) = algorithm {
            match decompress_data(&file_data, algo) {
                Ok(decompressed) => Some(decompressed.len()),
                Err(_) => None,
            }
        } else {
            Some(original_size)
        };

        Ok(FileCompressionInfo {
            path: path.as_ref().to_path_buf(),
            is_compressed,
            algorithm,
            compressed_size: if is_compressed {
                Some(original_size)
            } else {
                None
            },
            uncompressed_size,
            compression_ratio: if let (Some(compressed), Some(uncompressed)) = (
                if is_compressed {
                    Some(original_size)
                } else {
                    None
                },
                uncompressed_size,
            ) {
                Some(uncompressed as f64 / compressed as f64)
            } else {
                None
            },
        })
    }
}

/// Information about a file's compression status
#[derive(Debug, Clone)]
pub struct FileCompressionInfo {
    /// Path to the file
    pub path: std::path::PathBuf,
    /// Whether the file is compressed
    pub is_compressed: bool,
    /// Detected compression algorithm
    pub algorithm: Option<CompressionAlgorithm>,
    /// Size of compressed data (if compressed)
    pub compressed_size: Option<usize>,
    /// Size of uncompressed data
    pub uncompressed_size: Option<usize>,
    /// Compression ratio (uncompressed / compressed)
    pub compression_ratio: Option<f64>,
}

/// Global transparent file handler instance
static GLOBAL_HANDLER: std::sync::OnceLock<TransparentFileHandler> = std::sync::OnceLock::new();

/// Initialize the global transparent file handler
#[allow(dead_code)]
pub fn init_global_handler(handler: TransparentFileHandler) {
    let _ = GLOBAL_HANDLER.set(handler);
}

/// Get a reference to the global transparent file handler
#[allow(dead_code)]
pub fn global_handler() -> &'static TransparentFileHandler {
    GLOBAL_HANDLER.get_or_init(TransparentFileHandler::default)
}

/// Convenient function to read a file with automatic decompression using global handler
#[allow(dead_code)]
pub fn read_file_transparent<P: AsRef<Path>>(path: P) -> Result<Vec<u8>> {
    global_handler().read_file(path)
}

/// Convenient function to write a file with automatic compression using global handler
#[allow(dead_code)]
pub fn write_file_transparent<P: AsRef<Path>>(path: P, data: &[u8]) -> Result<()> {
    global_handler().write_file(path, data)
}

/// Convenient function to copy a file with transparent compression/decompression using global handler
#[allow(dead_code)]
pub fn copy_file_transparent<P: AsRef<Path>, Q: AsRef<Path>>(
    source: P,
    destination: Q,
) -> Result<()> {
    global_handler().copy_file(source, destination)
}

/// Convenient function to get file compression info using global handler
#[allow(dead_code)]
pub fn file_info_transparent<P: AsRef<Path>>(path: P) -> Result<FileCompressionInfo> {
    global_handler().file_info(path)
}

//
// Specialized Scientific Data Compression Implementations
//

/// Compress floating-point data using specialized techniques
#[allow(dead_code)]
fn compress_fpzip(data: &[u8], level: u32) -> Result<Vec<u8>> {
    // Simple implementation: magic header + floating-point optimized compression
    let mut result = Vec::with_capacity(FPZIP_MAGIC.len() + data.len());

    // Add magic header
    result.extend_from_slice(FPZIP_MAGIC);

    // For now, use a simple approach: if data length is divisible by 4 or 8,
    // assume it's floating-point data and apply bit manipulation optimizations
    if data.len().is_multiple_of(8) {
        // Assume f64 data - apply bit manipulation to reduce entropy
        let float_data =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f64, data.len() / 8) };

        // XOR consecutive values to reduce entropy (delta-like encoding for floats)
        let mut encoded_data = Vec::with_capacity(data.len());
        if !float_data.is_empty() {
            // Store first value as-is
            encoded_data.extend_from_slice(&float_data[0].to_le_bytes());

            // XOR subsequent values with previous
            for i in 1..float_data.len() {
                let current_bits = float_data[i].to_bits();
                let prev_bits = float_data[i - 1].to_bits();
                let xor_result = current_bits ^ prev_bits;
                encoded_data.extend_from_slice(&xor_result.to_le_bytes());
            }
        }

        // Compress the XOR'd data with LZ4
        let compressed = compress_data(&encoded_data, CompressionAlgorithm::Lz4, Some(6))?;
        result.extend_from_slice(&compressed);
    } else if data.len().is_multiple_of(4) {
        // Assume f32 data - apply similar technique
        let float_data =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, data.len() / 4) };

        let mut encoded_data = Vec::with_capacity(data.len());
        if !float_data.is_empty() {
            encoded_data.extend_from_slice(&float_data[0].to_le_bytes());

            for i in 1..float_data.len() {
                let current_bits = float_data[i].to_bits();
                let prev_bits = float_data[i - 1].to_bits();
                let xor_result = current_bits ^ prev_bits;
                encoded_data.extend_from_slice(&xor_result.to_le_bytes());
            }
        }

        let compressed = compress_data(&encoded_data, CompressionAlgorithm::Lz4, Some(6))?;
        result.extend_from_slice(&compressed);
    } else {
        // Not aligned to float boundaries, use regular compression
        let compressed = compress_data(data, CompressionAlgorithm::Zstd, Some(6))?;
        result.extend_from_slice(&compressed);
    }

    Ok(result)
}

/// Decompress floating-point data
#[allow(dead_code)]
fn decompress_fpzip(data: &[u8]) -> Result<Vec<u8>> {
    if !data.starts_with(FPZIP_MAGIC) {
        return Err(IoError::DecompressionError(
            "Invalid FPZip magic bytes".to_string(),
        ));
    }

    let compressed_data = &data[FPZIP_MAGIC.len()..];

    // Decompress the inner data
    let decompressed = decompress_data(compressed_data, CompressionAlgorithm::Lz4)?;

    // Check if we need to reverse the XOR encoding
    if decompressed.len() % 8 == 0 {
        // f64 data
        let mut float_data = unsafe {
            std::slice::from_raw_parts(decompressed.as_ptr() as *const u64, decompressed.len() / 8)
        }
        .to_vec();

        // Reverse XOR encoding
        for i in 1..float_data.len() {
            float_data[i] ^= float_data[i - 1];
        }

        // Convert back to bytes
        let result = unsafe {
            std::slice::from_raw_parts(float_data.as_ptr() as *const u8, float_data.len() * 8)
        }
        .to_vec();
        Ok(result)
    } else if decompressed.len() % 4 == 0 {
        // f32 data
        let mut float_data = unsafe {
            std::slice::from_raw_parts(decompressed.as_ptr() as *const u32, decompressed.len() / 4)
        }
        .to_vec();

        for i in 1..float_data.len() {
            float_data[i] ^= float_data[i - 1];
        }

        let result = unsafe {
            std::slice::from_raw_parts(float_data.as_ptr() as *const u8, float_data.len() * 4)
        }
        .to_vec();
        Ok(result)
    } else {
        // Regular decompression
        decompress_data(compressed_data, CompressionAlgorithm::Zstd)
    }
}

/// Compress time series data using delta encoding + LZ4
#[allow(dead_code)]
fn compress_delta_lz4(data: &[u8], level: u32) -> Result<Vec<u8>> {
    let mut result = Vec::with_capacity(DELTA_LZ4_MAGIC.len() + data.len());

    // Add magic header
    result.extend_from_slice(DELTA_LZ4_MAGIC);

    if data.len() < 8 {
        // Too small for delta encoding, use regular LZ4
        let compressed = compress_data(data, CompressionAlgorithm::Lz4, Some(level))?;
        result.extend_from_slice(&compressed);
        return Ok(result);
    }

    // Try to detect _data type and apply appropriate delta encoding
    let mut delta_encoded = Vec::with_capacity(data.len());

    if data.len().is_multiple_of(8) {
        // Assume i64 or f64 time series
        let values =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const i64, data.len() / 8) };

        // Store first value
        delta_encoded.extend_from_slice(&values[0].to_le_bytes());

        // Store differences
        for i in 1..values.len() {
            let delta = values[i].wrapping_sub(values[i - 1]);
            delta_encoded.extend_from_slice(&delta.to_le_bytes());
        }
    } else if data.len().is_multiple_of(4) {
        // Assume i32 or f32 time series
        let values =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const i32, data.len() / 4) };

        delta_encoded.extend_from_slice(&values[0].to_le_bytes());

        for i in 1..values.len() {
            let delta = values[i].wrapping_sub(values[i - 1]);
            delta_encoded.extend_from_slice(&delta.to_le_bytes());
        }
    } else {
        // Byte-level delta encoding
        delta_encoded.push(data[0]);
        for i in 1..data.len() {
            let delta = data[i].wrapping_sub(data[i - 1]);
            delta_encoded.push(delta);
        }
    }

    // Compress the delta-encoded data
    let compressed = compress_data(&delta_encoded, CompressionAlgorithm::Lz4, Some(level))?;
    result.extend_from_slice(&compressed);

    Ok(result)
}

/// Decompress delta-encoded time series data
#[allow(dead_code)]
fn decompress_delta_lz4(data: &[u8]) -> Result<Vec<u8>> {
    if !data.starts_with(DELTA_LZ4_MAGIC) {
        return Err(IoError::DecompressionError(
            "Invalid Delta-LZ4 magic bytes".to_string(),
        ));
    }

    let compressed_data = &data[DELTA_LZ4_MAGIC.len()..];

    // Decompress the delta-encoded data
    let delta_data = decompress_data(compressed_data, CompressionAlgorithm::Lz4)?;

    if delta_data.len() < 8 {
        return Ok(delta_data);
    }

    // Reverse delta encoding
    if delta_data.len() % 8 == 0 {
        // i64/f64 data
        let mut values = unsafe {
            std::slice::from_raw_parts(delta_data.as_ptr() as *const i64, delta_data.len() / 8)
        }
        .to_vec();

        // Reconstruct original values
        for i in 1..values.len() {
            values[i] = values[i - 1].wrapping_add(values[i]);
        }

        let result =
            unsafe { std::slice::from_raw_parts(values.as_ptr() as *const u8, values.len() * 8) }
                .to_vec();
        Ok(result)
    } else if delta_data.len() % 4 == 0 {
        // i32/f32 data
        let mut values = unsafe {
            std::slice::from_raw_parts(delta_data.as_ptr() as *const i32, delta_data.len() / 4)
        }
        .to_vec();

        for i in 1..values.len() {
            values[i] = values[i - 1].wrapping_add(values[i]);
        }

        let result =
            unsafe { std::slice::from_raw_parts(values.as_ptr() as *const u8, values.len() * 4) }
                .to_vec();
        Ok(result)
    } else {
        // Byte-level reconstruction
        let mut result = delta_data.clone();
        for i in 1..result.len() {
            result[i] = result[i - 1].wrapping_add(result[i]);
        }
        Ok(result)
    }
}

//
// Parallel Compression/Decompression
//

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// Configuration for parallel compression/decompression operations
#[derive(Debug, Clone)]
pub struct ParallelCompressionConfig {
    /// Number of threads to use (0 means use all available cores)
    pub num_threads: usize,
    /// Size of each chunk in bytes for parallel processing
    pub chunk_size: usize,
    /// Buffer size for I/O operations
    pub buffer_size: usize,
    /// Whether to enable memory mapping for large files
    pub enable_memory_mapping: bool,
}

impl Default for ParallelCompressionConfig {
    fn default() -> Self {
        Self {
            num_threads: 0,          // Use all available cores
            chunk_size: 1024 * 1024, // 1 MB chunks
            buffer_size: 64 * 1024,  // 64 KB buffer
            enable_memory_mapping: true,
        }
    }
}

/// Statistics for parallel compression/decompression operations
#[derive(Debug, Clone)]
pub struct ParallelCompressionStats {
    /// Total number of chunks processed
    pub chunks_processed: usize,
    /// Total bytes processed (uncompressed)
    pub bytes_processed: usize,
    /// Total bytes output (compressed/decompressed)
    pub bytes_output: usize,
    /// Time taken for the operation in milliseconds
    pub operation_time_ms: f64,
    /// Throughput in bytes per second
    pub throughput_bps: f64,
    /// Compression ratio (input_size / output_size)
    pub compression_ratio: f64,
    /// Number of threads used
    pub threads_used: usize,
}

/// Compress data in parallel using multiple threads
#[allow(dead_code)]
pub fn compress_data_parallel(
    data: &[u8],
    algorithm: CompressionAlgorithm,
    level: Option<u32>,
    config: ParallelCompressionConfig,
) -> Result<(Vec<u8>, ParallelCompressionStats)> {
    let start_time = Instant::now();
    let input_size = data.len();

    // Configure thread pool
    let num_threads = if config.num_threads == 0 {
        std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1)
    } else {
        config.num_threads
    };

    // For very small data, use sequential compression
    if input_size <= config.chunk_size {
        let compressed = compress_data(data, algorithm, level)?;
        let operation_time = start_time.elapsed().as_secs_f64() * 1000.0;

        let stats = ParallelCompressionStats {
            chunks_processed: 1,
            bytes_processed: input_size,
            bytes_output: compressed.len(),
            operation_time_ms: operation_time,
            throughput_bps: input_size as f64 / (operation_time / 1000.0),
            compression_ratio: input_size as f64 / compressed.len() as f64,
            threads_used: 1,
        };

        return Ok((compressed, stats));
    }

    // Split data into chunks
    let chunk_size = config.chunk_size;
    let chunks: Vec<&[u8]> = data.chunks(chunk_size).collect();
    let chunk_count = chunks.len();

    // Process chunks sequentially (TODO: implement proper parallel processing)
    let processed_count = Arc::new(AtomicUsize::new(0));
    let compressed_chunks: Result<Vec<Vec<u8>>> = chunks
        .into_iter()
        .map(|chunk| {
            let result = compress_data(chunk, algorithm, level);
            processed_count.fetch_add(1, Ordering::Relaxed);
            result
        })
        .collect();

    let compressed_chunks = compressed_chunks?;

    // Calculate total size and concatenate results
    let total_compressed_size: usize = compressed_chunks.iter().map(|chunk| chunk.len()).sum();
    let mut result = Vec::with_capacity(total_compressed_size + (chunk_count * 8)); // Extra space for chunk headers

    // Write chunk headers (size information for decompression)
    result.extend_from_slice(&(chunk_count as u64).to_le_bytes());
    for chunk in &compressed_chunks {
        result.extend_from_slice(&(chunk.len() as u64).to_le_bytes());
    }

    // Write compressed chunks
    for chunk in compressed_chunks {
        result.extend_from_slice(&chunk);
    }

    let operation_time = start_time.elapsed().as_secs_f64() * 1000.0;

    let stats = ParallelCompressionStats {
        chunks_processed: chunk_count,
        bytes_processed: input_size,
        bytes_output: result.len(),
        operation_time_ms: operation_time,
        throughput_bps: input_size as f64 / (operation_time / 1000.0),
        compression_ratio: input_size as f64 / result.len() as f64,
        threads_used: num_threads,
    };

    Ok((result, stats))
}

/// Decompress data in parallel using multiple threads
#[allow(dead_code)]
pub fn decompress_data_parallel(
    data: &[u8],
    algorithm: CompressionAlgorithm,
    config: ParallelCompressionConfig,
) -> Result<(Vec<u8>, ParallelCompressionStats)> {
    let start_time = Instant::now();
    let input_size = data.len();

    // Configure thread pool
    let num_threads = if config.num_threads == 0 {
        std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1)
    } else {
        config.num_threads
    };

    // Check if this is parallel-compressed data by looking for chunk headers
    if data.len() < 8 {
        // Too small to be parallel-compressed data, use sequential decompression
        let decompressed = decompress_data(data, algorithm)?;
        let operation_time = start_time.elapsed().as_secs_f64() * 1000.0;

        let stats = ParallelCompressionStats {
            chunks_processed: 1,
            bytes_processed: input_size,
            bytes_output: decompressed.len(),
            operation_time_ms: operation_time,
            throughput_bps: decompressed.len() as f64 / (operation_time / 1000.0),
            compression_ratio: decompressed.len() as f64 / input_size as f64,
            threads_used: 1,
        };

        return Ok((decompressed, stats));
    }

    // Read chunk count
    let chunk_count = u64::from_le_bytes(
        data[0..8]
            .try_into()
            .map_err(|_| IoError::DecompressionError("Invalid chunk header".to_string()))?,
    ) as usize;

    if chunk_count == 0 || chunk_count > data.len() / 8 {
        // Not parallel-compressed data or invalid, use sequential decompression
        let decompressed = decompress_data(data, algorithm)?;
        let operation_time = start_time.elapsed().as_secs_f64() * 1000.0;

        let stats = ParallelCompressionStats {
            chunks_processed: 1,
            bytes_processed: input_size,
            bytes_output: decompressed.len(),
            operation_time_ms: operation_time,
            throughput_bps: decompressed.len() as f64 / (operation_time / 1000.0),
            compression_ratio: decompressed.len() as f64 / input_size as f64,
            threads_used: 1,
        };

        return Ok((decompressed, stats));
    }

    // Read chunk sizes
    let header_size = 8 + (chunk_count * 8);
    if data.len() < header_size {
        return Err(IoError::DecompressionError(
            "Truncated chunk headers".to_string(),
        ));
    }

    let mut chunk_sizes = Vec::with_capacity(chunk_count);
    for i in 0..chunk_count {
        let start_idx = 8 + (i * 8);
        let size = u64::from_le_bytes(
            data[start_idx..start_idx + 8]
                .try_into()
                .map_err(|_| IoError::DecompressionError("Invalid chunk size".to_string()))?,
        ) as usize;
        chunk_sizes.push(size);
    }

    // Extract compressed chunks
    let mut chunks = Vec::with_capacity(chunk_count);
    let mut offset = header_size;

    for &size in &chunk_sizes {
        if offset + size > data.len() {
            return Err(IoError::DecompressionError(
                "Truncated chunk data".to_string(),
            ));
        }
        chunks.push(&data[offset..offset + size]);
        offset += size;
    }

    // Decompress chunks sequentially (TODO: implement proper parallel processing)
    let processed_count = Arc::new(AtomicUsize::new(0));
    let decompressed_chunks: Result<Vec<Vec<u8>>> = chunks
        .into_iter()
        .map(|chunk| {
            let result = decompress_data(chunk, algorithm);
            processed_count.fetch_add(1, Ordering::Relaxed);
            result
        })
        .collect();

    let decompressed_chunks = decompressed_chunks?;

    // Concatenate results
    let total_size: usize = decompressed_chunks.iter().map(|chunk| chunk.len()).sum();
    let mut result = Vec::with_capacity(total_size);

    for chunk in decompressed_chunks {
        result.extend_from_slice(&chunk);
    }

    let operation_time = start_time.elapsed().as_secs_f64() * 1000.0;

    let stats = ParallelCompressionStats {
        chunks_processed: chunk_count,
        bytes_processed: input_size,
        bytes_output: result.len(),
        operation_time_ms: operation_time,
        throughput_bps: result.len() as f64 / (operation_time / 1000.0),
        compression_ratio: result.len() as f64 / input_size as f64,
        threads_used: num_threads,
    };

    Ok((result, stats))
}

/// Compress a file in parallel and save it to a new file
#[allow(dead_code)]
pub fn compress_file_parallel<P: AsRef<Path>>(
    input_path: P,
    output_path: Option<P>,
    algorithm: CompressionAlgorithm,
    level: Option<u32>,
    config: ParallelCompressionConfig,
) -> Result<(String, ParallelCompressionStats)> {
    // Read input file
    let mut input_data = Vec::new();
    File::open(input_path.as_ref())
        .map_err(|e| IoError::FileError(format!("Failed to open input file: {e}")))?
        .read_to_end(&mut input_data)
        .map_err(|e| IoError::FileError(format!("Failed to read input file: {e}")))?;

    // Compress the data in parallel
    let (compressed_data, stats) = compress_data_parallel(&input_data, algorithm, level, config)?;

    // Determine output _path
    let output_path_string = match output_path {
        Some(path) => path.as_ref().to_string_lossy().to_string(),
        None => {
            let mut path_buf = input_path.as_ref().to_path_buf();
            let ext = algorithm.extension();
            let file_name = path_buf
                .file_name()
                .ok_or_else(|| IoError::FileError("Invalid input file _path".to_string()))?
                .to_string_lossy()
                .to_string();
            let new_file_name = format!("{file_name}.{ext}");
            path_buf.set_file_name(new_file_name);
            path_buf.to_string_lossy().to_string()
        }
    };

    // Write the compressed data to the output file
    File::create(&output_path_string)
        .map_err(|e| IoError::FileError(format!("Failed to create output file: {e}")))?
        .write_all(&compressed_data)
        .map_err(|e| IoError::FileError(format!("Failed to write to output file: {e}")))?;

    Ok((output_path_string, stats))
}

/// Decompress a file in parallel and save it to a new file
#[allow(dead_code)]
pub fn decompress_file_parallel<P: AsRef<Path>>(
    input_path: P,
    output_path: Option<P>,
    algorithm: Option<CompressionAlgorithm>,
    config: ParallelCompressionConfig,
) -> Result<(String, ParallelCompressionStats)> {
    // Determine the compression algorithm
    let algorithm = match algorithm {
        Some(algo) => algo,
        None => {
            let ext = input_path
                .as_ref()
                .extension()
                .ok_or_else(|| {
                    IoError::DecompressionError("Unable to determine file extension".to_string())
                })?
                .to_string_lossy()
                .to_string();

            CompressionAlgorithm::from_extension(&ext)
                .ok_or(IoError::UnsupportedCompressionAlgorithm(ext))?
        }
    };

    // Read input file
    let mut input_data = Vec::new();
    File::open(input_path.as_ref())
        .map_err(|e| IoError::FileError(format!("Failed to open input file: {e}")))?
        .read_to_end(&mut input_data)
        .map_err(|e| IoError::FileError(format!("Failed to read input file: {e}")))?;

    // Decompress the data in parallel
    let (decompressed_data, stats) = decompress_data_parallel(&input_data, algorithm, config)?;

    // Determine output _path
    let output_path_string = match output_path {
        Some(path) => path.as_ref().to_string_lossy().to_string(),
        None => {
            let path_str = input_path.as_ref().to_string_lossy().to_string();
            let ext = algorithm.extension();

            if path_str.ends_with(&format!(".{ext}")) {
                path_str[0..path_str.len() - ext.len() - 1].to_string()
            } else {
                format!("{path_str}.decompressed")
            }
        }
    };

    // Write the decompressed data to the output file
    File::create(&output_path_string)
        .map_err(|e| IoError::FileError(format!("Failed to create output file: {e}")))?
        .write_all(&decompressed_data)
        .map_err(|e| IoError::FileError(format!("Failed to write to output file: {e}")))?;

    Ok((output_path_string, stats))
}

/// Benchmark compression performance for different algorithms and configurations
#[allow(dead_code)]
pub fn benchmark_compression_algorithms(
    data: &[u8],
    algorithms: &[CompressionAlgorithm],
    levels: &[u32],
    parallel_configs: &[ParallelCompressionConfig],
) -> Result<Vec<CompressionBenchmarkResult>> {
    let mut results = Vec::new();

    for &algorithm in algorithms {
        for &level in levels {
            // Sequential compression
            let start_time = Instant::now();
            let compressed = compress_data(data, algorithm, Some(level))?;
            let sequential_time = start_time.elapsed().as_secs_f64() * 1000.0;

            let decompressed = decompress_data(&compressed, algorithm)?;
            let sequential_decomp_time =
                start_time.elapsed().as_secs_f64() * 1000.0 - sequential_time;

            assert_eq!(data, &decompressed, "Round-trip failed for {algorithm:?}");

            // Parallel compression for each config
            for config in parallel_configs {
                let (par_compressed, par_comp_stats) =
                    compress_data_parallel(data, algorithm, Some(level), config.clone())?;
                let (par_decompressed, par_decomp_stats) =
                    decompress_data_parallel(&par_compressed, algorithm, config.clone())?;

                assert_eq!(
                    data, &par_decompressed,
                    "Parallel round-trip failed for {algorithm:?}"
                );

                results.push(CompressionBenchmarkResult {
                    algorithm,
                    level,
                    config: config.clone(),
                    input_size: data.len(),
                    compressed_size: compressed.len(),
                    parallel_compressed_size: par_compressed.len(),
                    sequential_compression_time_ms: sequential_time,
                    sequential_decompression_time_ms: sequential_decomp_time,
                    parallel_compression_stats: par_comp_stats,
                    parallel_decompression_stats: par_decomp_stats,
                    compression_ratio: data.len() as f64 / compressed.len() as f64,
                    parallel_compression_ratio: data.len() as f64 / par_compressed.len() as f64,
                });
            }
        }
    }

    Ok(results)
}

/// Results from compression benchmarking
#[derive(Debug, Clone)]
pub struct CompressionBenchmarkResult {
    /// The compression algorithm tested
    pub algorithm: CompressionAlgorithm,
    /// The compression level used
    pub level: u32,
    /// The parallel configuration used
    pub config: ParallelCompressionConfig,
    /// Size of input data
    pub input_size: usize,
    /// Size of sequentially compressed data
    pub compressed_size: usize,
    /// Size of parallel compressed data
    pub parallel_compressed_size: usize,
    /// Time for sequential compression
    pub sequential_compression_time_ms: f64,
    /// Time for sequential decompression  
    pub sequential_decompression_time_ms: f64,
    /// Statistics from parallel compression
    pub parallel_compression_stats: ParallelCompressionStats,
    /// Statistics from parallel decompression
    pub parallel_decompression_stats: ParallelCompressionStats,
    /// Compression ratio for sequential compression
    pub compression_ratio: f64,
    /// Compression ratio for parallel compression
    pub parallel_compression_ratio: f64,
}

impl CompressionBenchmarkResult {
    /// Calculate the speedup factor for parallel compression vs sequential
    pub fn compression_speedup(&self) -> f64 {
        self.sequential_compression_time_ms / self.parallel_compression_stats.operation_time_ms
    }

    /// Calculate the speedup factor for parallel decompression vs sequential
    pub fn decompression_speedup(&self) -> f64 {
        self.sequential_decompression_time_ms / self.parallel_decompression_stats.operation_time_ms
    }

    /// Calculate the overhead factor for parallel compression (how much larger the parallel-compressed data is)
    pub fn compression_overhead(&self) -> f64 {
        self.parallel_compressed_size as f64 / self.compressed_size as f64
    }
}
