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
/// Harwell-Boeing sparse matrix format module
///
/// Provides functionality for reading and writing Harwell-Boeing sparse matrix files:
/// - Support for real and complex matrices
/// - Different matrix symmetry types (general, symmetric, hermitian, skew-symmetric)
/// - Pattern matrices (structure only, no values)
/// - Conversion to/from column-compressed sparse (CCS) format
/// - Integration with ndarray for efficient matrix operations
pub mod harwell_boeing;
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
/// Memory-mapped file I/O module
///
/// Provides memory-mapped file operations for efficient handling of large arrays:
/// - Memory-mapped arrays for minimal memory usage
/// - Read-only and read-write access modes
/// - Support for multi-dimensional arrays
/// - Cross-platform compatibility (Unix and Windows)
/// - Type-safe operations with generic numeric types
///
/// # Examples
///
/// ```rust,no_run
/// use scirs2_io::mmap::{MmapArray, create_mmap_array};
/// use ndarray::Array2;
///
/// // Create a large array file
/// let data = Array2::from_shape_fn((1000, 1000), |(i, j)| (i + j) as f64);
/// let file_path = "large_array.bin";
///
/// // Write array to file
/// create_mmap_array(file_path, &data)?;
///
/// // Memory-map the array for reading
/// let mmap_array: MmapArray<f64> = MmapArray::open(file_path)?;
/// let shape = mmap_array.shape()?;
/// let array_view = mmap_array.as_array_view(&shape)?;
///
/// // Access data without loading entire file into memory
/// let slice = mmap_array.as_slice()?;
/// let value = slice[500 * 1000 + 500]; // Access element at (500, 500)
/// println!("Value at (500, 500): {}", value);
/// # Ok::<(), scirs2_io::error::IoError>(())
/// ```
pub mod mmap;
/// NetCDF file format module
///
/// Provides functionality for reading and writing NetCDF files:
/// - Reading and writing NetCDF3 files
/// - Support for dimensions, variables, and attributes
/// - Conversion between NetCDF and ndarray data structures
/// - Memory-efficient access to large datasets
pub mod netcdf;
/// Network I/O and cloud storage integration
///
/// Provides functionality for reading and writing files over network protocols
/// and integrating with cloud storage services:
/// - HTTP/HTTPS file download and upload with progress tracking
/// - Cloud storage integration (AWS S3, Google Cloud Storage, Azure Blob Storage)
/// - Streaming I/O for efficient handling of large files over network
/// - Authentication and secure credential management
/// - Retry logic and error recovery for network operations
/// - Local caching for offline access and performance optimization
///
/// # Examples
///
/// ```rust,no_run
/// use scirs2_io::network::NetworkClient;
///
/// // Create a network client for downloading files
/// let client = NetworkClient::new();
/// println!("Network client created for file operations");
/// ```
pub mod network;
/// Data serialization utilities
///
/// Provides functionality for serializing and deserializing scientific data:
/// - Binary, JSON, and MessagePack serialization formats
/// - Array serialization with metadata
/// - Structured data serialization
/// - Sparse matrix serialization
pub mod serialize;
/// Comprehensive sparse matrix format support
///
/// Provides unified support for common sparse matrix formats:
/// - COO (Coordinate), CSR (Compressed Sparse Row), and CSC (Compressed Sparse Column) formats
/// - Efficient format conversion algorithms
/// - Matrix operations (addition, multiplication, transpose)
/// - I/O support with Matrix Market integration
/// - Performance-optimized algorithms for large sparse matrices
/// - Memory-efficient sparse data handling
///
/// # Examples
///
/// ```rust,no_run
/// use scirs2_io::sparse::SparseMatrix;
/// use ndarray::Array2;
///
/// // Create a sparse matrix from a dense array
/// let dense = Array2::from_shape_vec((3, 3), vec![
///     1.0_f64, 0.0_f64, 2.0_f64,
///     0.0_f64, 3.0_f64, 0.0_f64,
///     4.0_f64, 0.0_f64, 5.0_f64
/// ]).unwrap();
///
/// let mut sparse = SparseMatrix::from_dense_2d(&dense, 0.0_f64)?;
/// println!("Sparse matrix: {} non-zeros", sparse.nnz());
///
/// // Convert to different formats
/// let _csr = sparse.to_csr()?;
/// let _csc = sparse.to_csc()?;
///
/// // Save to file
/// sparse.save_matrix_market("matrix.mtx")?;
/// # Ok::<(), scirs2_io::error::IoError>(())
/// ```
pub mod sparse;
/// Streaming and iterator interfaces for large data handling
///
/// Provides memory-efficient streaming interfaces for processing large datasets:
/// - Chunked reading for processing files in configurable chunks
/// - Iterator-based APIs for seamless integration with Rust's iterator ecosystem
/// - Streaming CSV processing with header support
/// - Memory-efficient processing without loading entire files
/// - Performance monitoring and statistics tracking
///
/// # Examples
///
/// ```rust,no_run
/// use scirs2_io::streaming::{StreamingConfig, process_file_chunked};
///
/// // Process a large file in chunks
/// let config = StreamingConfig::default().chunk_size(64 * 1024);
///
/// let (result, stats) = process_file_chunked("large_file.dat", config, |chunk_data, chunk_id| {
///     println!("Processing chunk {}: {} bytes", chunk_id, chunk_data.len());
///     Ok(())
/// })?;
/// # Ok::<(), scirs2_io::error::IoError>(())
/// ```
pub mod streaming;

/// Async I/O support for streaming capabilities
///
/// Provides asynchronous I/O interfaces for non-blocking processing of large datasets:
/// - Async file reading and writing with tokio
/// - Asynchronous stream processing with backpressure
/// - Concurrent processing with configurable concurrency levels
/// - Network I/O support for remote data access
/// - Cancellation support for long-running operations
/// - Real-time progress monitoring for async operations
#[cfg(feature = "async")]
pub mod async_io;
/// Thread pool for parallel I/O operations
///
/// Provides a high-performance thread pool optimized for I/O operations:
/// - Separate thread pools for I/O-bound and CPU-bound tasks
/// - Work stealing for load balancing
/// - Performance monitoring and statistics
/// - Configurable thread counts and queue sizes
/// - Global thread pool for convenience
pub mod thread_pool;
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
