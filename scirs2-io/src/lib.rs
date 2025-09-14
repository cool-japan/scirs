#![allow(deprecated)]
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
//! - `fortran`: Support for Fortran unformatted files

#![warn(missing_docs)]
// Allow specific Clippy warnings with justifications
#![allow(clippy::manual_div_ceil)] // Manual div_ceil implementation for compatibility with Rust versions without div_ceil
#![allow(clippy::should_implement_trait)] // from_str methods are used consistently across modules
#![allow(clippy::type_complexity)] // Complex type is necessary for format validators

/// Advanced Mode Coordinator - Unified Intelligence for I/O Operations
///
/// Provides the highest level of intelligent I/O processing by coordinating multiple advanced systems:
/// - Neural adaptive optimization with reinforcement learning
/// - Quantum-inspired parallel processing with superposition algorithms
/// - GPU acceleration with multi-backend support
/// - Advanced memory management and resource allocation
/// - Real-time performance monitoring and self-optimization
/// - Meta-learning for cross-domain adaptation
/// - Emergent behavior detection and autonomous system improvement
pub mod advanced_coordinator;
pub mod arff;
/// Enhanced algorithms for Advanced Mode
///
/// Provides advanced algorithmic enhancements for the Advanced coordinator:
/// - Advanced pattern recognition with deep learning capabilities
/// - Multi-scale feature extraction and analysis
/// - Emergent pattern detection and meta-pattern recognition
/// - Sophisticated optimization recommendation systems
/// - Self-improving algorithmic components with adaptive learning
pub mod enhanced_algorithms;

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
/// Database connectivity
///
/// Provides interfaces for database operations:
/// - Support for SQL databases (PostgreSQL, MySQL, SQLite)
/// - NoSQL database support (MongoDB, Redis, Cassandra)
/// - Time series databases (InfluxDB)
/// - Query builder and ORM-like features
/// - Bulk loading and export capabilities
/// - Integration with scientific data formats
pub mod database;
/// Distributed I/O processing
///
/// Provides infrastructure for distributed processing of large datasets:
/// - Distributed file reading with partitioning strategies
/// - Parallel writing with merge capabilities
/// - Distributed array operations
/// - Load balancing and fault tolerance
/// - Progress tracking for distributed operations
pub mod distributed;
pub mod error;
/// Domain-specific file formats
///
/// Provides specialized support for scientific file formats:
/// - Bioinformatics: FASTA, FASTQ, SAM/BAM, VCF
/// - Geospatial: GeoTIFF, Shapefile, GeoJSON, KML
/// - Astronomical: FITS, VOTable
pub mod formats;
/// Fortran unformatted file format module
///
/// Provides functionality for reading and writing Fortran unformatted files:
/// - Sequential, direct, and stream access modes
/// - Support for different endianness and record marker sizes
/// - Automatic format detection
/// - Arrays stored in column-major order (Fortran convention)
/// - Support for all common Fortran data types
pub mod fortran;
/// GPU-accelerated I/O operations
///
/// Provides GPU-accelerated implementations of I/O operations using the scirs2-core GPU abstraction:
/// - GPU-accelerated compression and decompression
/// - GPU-accelerated data type conversions
/// - GPU-accelerated matrix operations for file I/O
/// - GPU-accelerated checksum computation
/// - Support for multiple GPU backends (CUDA, Metal, OpenCL)
/// - Automatic fallback to CPU when GPU is not available
#[cfg(feature = "gpu")]
/// GPU-accelerated I/O operations
///
/// Provides comprehensive GPU acceleration for I/O operations including:
/// - Multi-backend GPU support (CUDA, Metal, OpenCL)
/// - GPU-accelerated compression and decompression
/// - Advanced GPU memory management with pooling
/// - Performance monitoring and optimization
/// - Intelligent backend selection and workload optimization
pub mod gpu;
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
/// IDL (Interactive Data Language) save file format module
///
/// Provides functionality for reading and writing IDL save files (.sav):
/// - Support for all standard IDL data types
/// - Arrays, strings, structures, and complex numbers
/// - Automatic endianness detection and handling
/// - Compatible with IDL 8.0 format
/// - Commonly used in astronomy and remote sensing
pub mod idl;
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
/// Advanced metadata management
///
/// Provides comprehensive metadata handling across different file formats:
/// - Unified metadata interface for all formats
/// - Metadata validation with schemas
/// - Processing history tracking
/// - Format conversion between JSON, YAML, TOML
/// - Format-specific extensions
/// - Standard metadata keys for scientific data
pub mod metadata;
/// Machine learning framework compatibility
///
/// Provides conversion utilities and interfaces for ML frameworks:
/// - Support for PyTorch, TensorFlow, ONNX, SafeTensors formats
/// - Model and tensor serialization/deserialization
/// - Data type conversions between frameworks
/// - Dataset utilities for ML pipelines
/// - Seamless integration with ndarray
pub mod ml_framework;
/// Data pipeline APIs
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
/// Neural-adaptive I/O optimization with advanced-level intelligence
///
/// Provides AI-driven adaptive optimization for I/O operations:
/// - Machine learning-based performance optimization
/// - Dynamic parameter adaptation based on system metrics
/// - Neural network-driven decision making for resource allocation
/// - Real-time performance feedback and learning
/// - Advanced-high performance processing with adaptive algorithms
/// - SIMD-accelerated neural inference for low-latency decisions
pub mod neural_adaptive_io;
/// Out-of-core processing for terabyte-scale datasets
///
/// Provides infrastructure for processing datasets too large for memory:
/// - Memory-mapped arrays with virtual memory management
/// - Chunked processing with configurable chunk sizes
/// - Disk-based algorithms for sorting and aggregation
/// - Virtual arrays combining multiple data sources
/// - Sliding window iterators for streaming operations
pub mod out_of_core;
/// Data pipeline APIs
///
/// Provides a flexible framework for building data processing pipelines:
/// - Composable pipeline stages for reading, transforming, and writing data
/// - Multiple execution strategies (sequential, parallel, streaming, async)
/// - Built-in transformations (normalization, encoding, aggregation)
/// - Error handling and recovery mechanisms
/// - Progress tracking and monitoring
/// - Caching and checkpointing for long-running pipelines
pub mod pipeline;
/// Quantum-inspired I/O processing algorithms with advanced capabilities
///
/// Provides quantum-inspired algorithms for advanced-high performance I/O:
/// - Quantum superposition for parallel processing paths
/// - Quantum entanglement for correlated data operations
/// - Quantum annealing for parameter optimization
/// - Quantum interference patterns for data compression
/// - Quantum tunneling for barrier-free processing
/// - Quantum measurement for adaptive decision making
pub mod quantum_inspired_io;
/// Real-time data streaming protocols
///
/// Provides infrastructure for real-time data streaming and processing:
/// - WebSocket and Server-Sent Events support
/// - gRPC and MQTT streaming protocols
/// - Backpressure handling and flow control
/// - Stream transformations and filtering
/// - Multi-stream synchronization
/// - Time series buffering and aggregation
#[cfg(feature = "async")]
pub mod realtime;
/// Data serialization utilities
///
/// Provides functionality for serializing and deserializing scientific data:
/// - Binary, JSON, and MessagePack serialization formats
/// - Array serialization with metadata
/// - Structured data serialization
/// - Sparse matrix serialization
pub mod serialize;
/// SIMD-accelerated I/O operations
///
/// Provides SIMD-optimized implementations of common I/O operations:
/// - Data type conversions with SIMD
/// - Audio normalization and processing
/// - CSV parsing acceleration
/// - Compression utilities with SIMD
/// - Checksum calculations
pub mod simd_io;
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
/// Visualization tool integration
///
/// Provides interfaces for integrating with visualization libraries:
/// - Export to multiple visualization formats (Plotly, Matplotlib, Gnuplot, Vega-Lite)
/// - Fluent API for building plots
/// - Support for various plot types (line, scatter, histogram, heatmap)
/// - Quick plotting functions for common use cases
/// - Configurable styling and theming
pub mod visualization;
pub mod wavfile;
/// Workflow automation tools
///
/// Provides framework for building automated data processing workflows:
/// - Task definition and dependency management
/// - Workflow scheduling and execution
/// - Resource management and allocation
/// - Retry policies and error handling
/// - Progress monitoring and notifications
/// - Common workflow templates (ETL, batch processing)
pub mod workflow;
/// Zero-copy I/O optimizations
///
/// Provides zero-copy implementations for various I/O operations:
/// - Memory-mapped file access
/// - Zero-copy array views
/// - CSV parsing without allocation
/// - Binary data reading without copying
/// - Minimized memory allocations for large datasets
pub mod zero_copy;

// Re-export commonly used functionality
pub use advanced_coordinator::{
    AdaptiveImprovements, AdvancedCoordinator, AdvancedStatistics, IntelligenceLevel,
    PerformanceIntelligenceStats, ProcessingResult, QualityMetrics, StrategyType,
};
pub use enhanced_algorithms::{
    AdvancedPatternAnalysis, AdvancedPatternRecognizer, DataCharacteristics, EmergentPattern,
    MetaPattern, OptimizationRecommendation, SynergyType,
};
