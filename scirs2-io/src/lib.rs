//! Input/Output utilities module for SciRS2
//!
//! This module provides functionality for reading and writing various file formats
//! commonly used in scientific computing, including MATLAB, WAV, ARFF, and CSV files.
//!
//! ## Features
//!
//! - **MATLAB Support**: Read and write MATLAB `.mat` files
//! - **WAV File Support**: Read and write WAV audio files
//! - **ARFF Support**: Read and write Weka ARFF files
//! - **CSV Support**: Read and write CSV files with flexible configuration options
//! - **Error Handling**: Robust error handling with detailed error information
//!
//! ## Modules
//!
//! - `arff`: Support for ARFF (Attribute-Relation File Format) files
//! - `compression`: Utilities for data compression and decompression
//! - `csv`: Support for CSV (Comma-Separated Values) files
//! - `image`: Support for image file formats (PNG, JPEG, BMP, TIFF)
//! - `matlab`: Support for MATLAB (.mat) files
//! - `matrix_market`: Support for Matrix Market sparse and dense matrix files
//! - `netcdf`: Support for NetCDF scientific data files
//! - `serialize`: Utilities for data serialization and deserialization
//! - `validation`: Utilities for data validation and integrity checking
//! - `wavfile`: Support for WAV audio files
//! - `error`: Error types for the IO module

#![warn(missing_docs)]
// Allow specific Clippy warnings with justifications
#![allow(clippy::manual_div_ceil)] // Manual div_ceil implementation for compatibility with Rust versions without div_ceil
#![allow(clippy::should_implement_trait)] // from_str methods are used consistently across modules
#![allow(clippy::type_complexity)] // Complex type is necessary for format validators

pub mod arff;
/// Data compression module
///
/// Provides utilities for compressing and decompressing scientific data:
/// - Lossless compression algorithms (GZIP, ZSTD, LZ4, BZIP2)
/// - Array compression with metadata preservation
/// - Chunked compression for large datasets
/// - Compression level configuration
pub mod compression;
/// CSV (Comma-Separated Values) file format module
///
/// Provides functionality for reading and writing CSV files with various options:
/// - Basic CSV reading and writing
/// - Type conversion and automatic type detection
/// - Missing value handling with customizable options
/// - Memory-efficient processing of large files using chunked reading
/// - Support for specialized data types (date, time, complex numbers)
/// - Column-based operations with flexible configuration
pub mod csv;
pub mod error;
/// HDF5 file format module
///
/// Provides functionality for reading and writing HDF5 (Hierarchical Data Format) files:
/// - Reading and writing HDF5 groups and datasets
/// - Support for attributes on groups and datasets
/// - Multiple datatypes (integers, floats, strings, compound types)
/// - Chunking and compression options
/// - Integration with ndarray for efficient array operations
pub mod hdf5;
/// Image file format module
///
/// Provides functionality for reading and writing common image formats:
/// - Reading and writing PNG, JPEG, BMP, and TIFF images
/// - Metadata extraction and manipulation
/// - Conversion between different image formats
/// - Basic image processing operations
pub mod image;
pub mod matlab;
/// Matrix Market file format module
///
/// Provides functionality for reading and writing Matrix Market files:
/// - Support for sparse matrix coordinate format (COO)
/// - Support for dense array format
/// - Real, complex, integer, and pattern data types
/// - Different matrix symmetry types (general, symmetric, hermitian, skew-symmetric)
/// - Integration with ndarray for efficient matrix operations
pub mod matrix_market;
/// NetCDF file format module
///
/// Provides functionality for reading and writing NetCDF files:
/// - Reading and writing NetCDF3 files
/// - Support for dimensions, variables, and attributes
/// - Conversion between NetCDF and ndarray data structures
/// - Memory-efficient access to large datasets
pub mod netcdf;
/// Data serialization utilities
///
/// Provides functionality for serializing and deserializing scientific data:
/// - Binary, JSON, and MessagePack serialization formats
/// - Array serialization with metadata
/// - Structured data serialization
/// - Sparse matrix serialization
pub mod serialize;
/// Data validation and integrity checking module
///
/// Provides functionality for validating data integrity through checksums,
/// format validation, and other verification methods:
/// - File integrity validation with multiple checksum algorithms (CRC32, SHA256, BLAKE3)
/// - Format-specific validation for scientific data formats
/// - Directory manifests for data validation
/// - Integrity metadata for tracking data provenance
pub mod validation;
pub mod wavfile;
