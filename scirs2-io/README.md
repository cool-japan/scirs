# SciRS2 IO

[![crates.io](https://img.shields.io/crates/v/scirs2-io.svg)](https://crates.io/crates/scirs2-io)
[[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)]](../LICENSE)
[![Documentation](https://img.shields.io/docsrs/scirs2-io)](https://docs.rs/scirs2-io)

Input/Output module for the SciRS2 scientific computing library. This module provides functionality for reading and writing various scientific and numerical data formats.

## Features

### File Format Support
- **MATLAB Support**: Read and write MATLAB `.mat` files
- **WAV File Support**: Read and write WAV audio files  
- **ARFF Support**: Read and write Weka ARFF files
- **CSV Support**: Read and write CSV files with flexible configuration options
- **NetCDF Support**: Read and write NetCDF scientific data files with dimensions, variables, and attributes
- **HDF5 Support**: Hierarchical data format with groups, datasets, and compression options
- **Matrix Market**: Read and write sparse and dense matrices in Matrix Market format

### Data Processing
- **Image Support**: Read and write common image formats (PNG, JPEG, BMP, TIFF) with basic EXIF metadata
- **Data Serialization**: Serialize and deserialize arrays, structs, and enhanced sparse matrices (COO, CSR, CSC)
- **Data Compression**: Compress and decompress data using multiple algorithms (GZIP, ZSTD, LZ4, BZIP2)
- **Data Validation**: Verify data integrity through checksums and format validation
- **Sparse Matrix Operations**: Advanced sparse matrix support with format conversion and operations

### Additional Features
- **Error Handling**: Robust error handling with detailed error information
- **Memory Efficiency**: Chunked processing for large files
- **Type Safety**: Strong typing with Rust's type system

## Installation

Add the following to your `Cargo.toml`:

```toml
[dependencies]
scirs2-io = "0.1.0-alpha.4"
```

To enable specific features:

```toml
[dependencies]
scirs2-io = { version = "0.1.0-alpha.4", features = ["matlab", "image", "compression"] }
```

Available features:
- `matlab`: MATLAB file support
- `image`: Image file support
- `compression`: Advanced compression algorithms
- `validation`: File format validation tools
- `all`: All features

## Usage

Basic usage examples:

```rust
use scirs2_io::{matlab, wavfile, arff, csv, compression, validation};
use scirs2_core::error::CoreResult;
use ndarray::Array2;

// Read and write MATLAB .mat files
fn matlab_example() -> CoreResult<()> {
    // Read a MATLAB file
    let data = matlab::loadmat("data.mat")?;
    
    // Access a variable from the file
    let x = data.get_array::<f64>("x")?;
    println!("Variable x has shape {:?}", x.shape());
    
    // Create a new MATLAB file
    let mut mat_file = matlab::MatFile::new();
    
    // Add variables
    let array = Array2::<f64>::zeros((3, 3));
    mat_file.add_array("zeros", &array)?;
    
    // Save to file
    mat_file.save("output.mat")?;
    
    Ok(())
}

// Read and write WAV audio files
fn wav_example() -> CoreResult<()> {
    // Read a WAV file
    let (rate, data) = wavfile::read("audio.wav")?;
    println!("Audio file: {} Hz, {} samples", rate, data.len());
    
    // Create a simple sine wave
    let rate = 44100;
    let freq = 440.0;  // 440 Hz (A4)
    let duration = 2.0;  // 2 seconds
    
    let samples = (rate as f64 * duration) as usize;
    let mut sine_wave = Vec::with_capacity(samples);
    
    for i in 0..samples {
        let t = i as f64 / rate as f64;
        let sample = (2.0 * std::f64::consts::PI * freq * t).sin();
        // Scale to i16 range (-32768 to 32767)
        let sample_i16 = (sample * 32767.0) as i16;
        sine_wave.push(sample_i16);
    }
    
    // Write to WAV file
    wavfile::write("sine_440hz.wav", rate, &sine_wave)?;
    
    Ok(())
}

// Read and write ARFF files
fn arff_example() -> CoreResult<()> {
    // Read an ARFF file
    let dataset = arff::load_arff("data.arff")?;
    
    println!("Dataset name: {}", dataset.relation);
    println!("Attributes: {:?}", dataset.attributes.iter().map(|attr| &attr.name).collect::<Vec<_>>());
    println!("Number of instances: {}", dataset.data.len());
    
    // Create a new ARFF dataset
    let mut new_dataset = arff::ArffDataset::new("my_dataset");
    
    // Add attributes
    new_dataset.add_numeric_attribute("attr1")?;
    new_dataset.add_numeric_attribute("attr2")?;
    new_dataset.add_nominal_attribute("class", &["class1", "class2"])?;
    
    // Add instances
    new_dataset.add_instance(&[arff::ArffValue::Numeric(1.0), 
                               arff::ArffValue::Numeric(2.0), 
                               arff::ArffValue::Nominal("class1".to_string())])?;
    
    // Save to file
    arff::save_arff(&new_dataset, "output.arff")?;
    
    Ok(())
}

// Read and write CSV files
fn csv_example() -> CoreResult<()> {
    // Read a CSV file with default settings
    let (headers, data) = csv::read_csv("data.csv", None)?;
    println!("Headers: {:?}", headers);
    println!("Data shape: {:?}", data.shape());
    println!("First row: {:?}", data.row(0));
    
    // Read CSV with custom configuration
    let config = csv::CsvReaderConfig {
        delimiter: ';',
        has_header: true,
        comment_char: Some('#'),
        ..Default::default()
    };
    let (headers, data) = csv::read_csv("data.csv", Some(config))?;
    
    // Read as numeric data
    let (headers, numeric_data) = csv::read_csv_numeric("data.csv", None)?;
    println!("Numeric data shape: {:?}", numeric_data.shape());
    
    // Write data to CSV
    let data_to_write = Array2::<f64>::zeros((3, 4));
    let headers = vec!["A".to_string(), "B".to_string(), "C".to_string(), "D".to_string()];
    
    csv::write_csv("output.csv", &data_to_write, Some(&headers), None)?;
    
    // Write with custom configuration
    let write_config = csv::CsvWriterConfig {
        delimiter: ';',
        always_quote: true,
        ..Default::default()
    };
    
    csv::write_csv("custom_output.csv", &data_to_write, Some(&headers), Some(write_config))?;
    
    Ok(())
}
```

## Components

### MATLAB File Support

Functions for working with MATLAB `.mat` files:

```rust
use scirs2_io::matlab::{
    loadmat,                // Load MATLAB .mat file
    savemat,                // Save variables to MATLAB .mat file
    MatFile,                // MATLAB file representation
    MatVar,                 // MATLAB variable representation
};
```

### WAV File Support

Functions for working with WAV audio files:

```rust
use scirs2_io::wavfile::{
    read,                   // Read WAV file
    write,                  // Write WAV file
    WavInfo,                // WAV file information
};
```

### ARFF File Support

Functions for working with Weka ARFF files:

```rust
use scirs2_io::arff::{
    load_arff,              // Load ARFF file
    save_arff,              // Save ARFF file
    ArffDataset,            // ARFF dataset representation
    ArffAttribute,          // ARFF attribute representation
    ArffValue,              // ARFF value type (Numeric, Nominal, etc.)
};
```

### CSV File Support

Functions for working with CSV files:

```rust
use scirs2_io::csv::{
    read_csv,               // Read CSV file as string data
    read_csv_numeric,       // Read CSV file as numeric data
    read_csv_typed,         // Read CSV file with type conversion
    read_csv_chunked,       // Read large CSV files in chunks
    write_csv,              // Write 2D array to CSV file
    write_csv_columns,      // Write columns to CSV file
    write_csv_typed,        // Write typed data to CSV file
    CsvReaderConfig,        // Configuration for CSV reading
    CsvWriterConfig,        // Configuration for CSV writing
    DataValue,              // Mixed type value representation
    ColumnType,             // Column data type specification
    MissingValueOptions,    // Missing value handling options
};
```

### Image File Support

Functions for working with image files:

```rust
use scirs2_io::image::{
    read_image,             // Read image file
    write_image,            // Write image file
    read_image_metadata,    // Read image metadata
    convert_image,          // Convert between image formats
    get_grayscale,          // Convert image to grayscale
    ImageFormat,            // Image format (PNG, JPEG, BMP, TIFF)
    ColorMode,              // Image color mode (Grayscale, RGB, RGBA)
    ImageMetadata,          // Image metadata
};
```

### Data Serialization

Functions for data serialization:

```rust
use scirs2_io::serialize::{
    serialize_array,             // Serialize ndarray
    deserialize_array,           // Deserialize ndarray
    serialize_array_with_metadata, // Serialize ndarray with metadata
    deserialize_array_with_metadata, // Deserialize ndarray with metadata
    serialize_struct,            // Serialize struct
    deserialize_struct,          // Deserialize struct
    serialize_sparse_matrix,     // Serialize sparse matrix
    deserialize_sparse_matrix,   // Deserialize sparse matrix
    SerializationFormat,         // Format (Binary, JSON, MessagePack)
    SparseMatrixCOO,             // Sparse matrix in COO format
};
```

### Data Compression

Functions for data compression:

```rust
use scirs2_io::compression::{
    compress_data,              // Compress raw bytes
    decompress_data,            // Decompress raw bytes
    compress_file,              // Compress a file
    decompress_file,            // Decompress a file
    compression_ratio,          // Calculate compression ratio
    algorithm_info,             // Get information about compression algorithm
    CompressionAlgorithm,       // Compression algorithm (Gzip, Zstd, Lz4, Bzip2)
    CompressionInfo,            // Information about a compression algorithm
};

// For ndarray-specific compression
use scirs2_io::compression::ndarray::{
    compress_array,             // Compress ndarray
    decompress_array,           // Decompress ndarray
    compress_array_chunked,     // Compress large ndarray in chunks
    decompress_array_chunked,   // Decompress chunked ndarray
    compare_compression_algorithms, // Compare compression algorithms on a dataset
    CompressedArray,            // Container for compressed array data
    CompressedArrayMetadata,    // Metadata for compressed array
};

// For data validation
use scirs2_io::validation::{
    calculate_checksum,         // Calculate checksum for data
    verify_checksum,            // Verify data against checksum
    calculate_file_checksum,    // Calculate checksum for a file
    verify_file_checksum,       // Verify file against checksum
    generate_file_integrity_metadata, // Generate integrity metadata
    validate_file_integrity,    // Validate file against metadata
    create_directory_manifest,  // Create manifest for a directory
    ChecksumAlgorithm,          // Checksum algorithm types
    IntegrityMetadata,          // File integrity metadata
    ValidationReport,           // Validation result report
};

// For format validation
use scirs2_io::validation::formats::{
    validate_format,            // Validate specific file format
    detect_file_format,         // Detect file format automatically
    validate_file_format,       // Detailed format validation
    DataFormat,                 // Common scientific data formats
};
```

## Format Details

### MATLAB File Format

The module supports:
- MATLAB level 5 MAT-File Format
- Both compressed and uncompressed files
- Basic MATLAB types: double, single, int8/16/32/64, uint8/16/32/64, logical, char
- Multidimensional arrays
- Structure arrays
- Cell arrays

### WAV File Format

Supported WAV features:
- PCM format with various bit depths (8, 16, 24, 32 bit)
- Float format (32 and 64 bit)
- Mono and stereo files
- Sample rates from 8kHz to 192kHz
- Reading extended WAV format information

### ARFF File Format

The ARFF implementation supports:
- Attribute types: numeric, nominal, string, date
- Sparse ARFF format
- Comment handling
- Missing values
- Relation metadata

### CSV File Format

The CSV implementation supports:
- Standard comma-separated values with configurable delimiters
- Reading and writing with various options:
  - Custom delimiters (comma, semicolon, tab, etc.)
  - Quoted fields handling
  - Comment lines
  - Header row handling
  - Whitespace trimming
  - Line ending control (LF, CRLF)
- Conversion to and from numeric arrays
- Column-based I/O operations
- Type conversion and automatic type inference
- Missing value handling with customizable missing value markers
- Memory-efficient processing of large files using chunked reading
- Support for mixed data types (string, integer, float, boolean)

### Image File Format

The image module supports:
- Common image formats:
  - PNG (Portable Network Graphics)
  - JPEG (Joint Photographic Experts Group)
  - BMP (Bitmap)
  - TIFF (Tagged Image File Format)
- Image operations:
  - Reading and writing images with ndarray integration
  - Converting between different image formats
  - Extracting image metadata (dimensions, color type, etc.)
  - Converting color images to grayscale
- Image representation:
  - Images are represented as 3D arrays (height × width × channels)
  - Support for different color modes (Grayscale, RGB, RGBA)
  - Integration with ndarray for efficient image manipulation

### Data Serialization Format

The serialize module supports:
- Multiple serialization formats:
  - Binary format (compact, efficient using bincode)
  - JSON format (human-readable, widely compatible)
  - MessagePack format (compact binary, cross-language)
- Data types:
  - Ndarray arrays with automatic shape preservation
  - Arrays with additional metadata (units, descriptions, etc.)
  - Structured data (any Rust struct implementing Serialize/Deserialize)
  - Sparse matrices in COO (Coordinate) format
- Features:
  - Automatic type detection and conversion
  - Metadata preservation
  - Efficient storage for both dense and sparse data
  - Cross-platform compatibility

### Data Compression Format

The compression module supports:
- Multiple compression algorithms:
  - GZIP (good balance of speed and compression ratio)
  - Zstandard (excellent compression ratio, fast decompression)
  - LZ4 (extremely fast, moderate compression ratio)
  - BZIP2 (high compression ratio, slower speed)
- Compression features:
  - Raw data compression and decompression
  - File-based compression operations
  - Configurable compression levels
  - Algorithm comparison and benchmarking
- Ndarray-specific compression:
  - Memory-efficient compression of large arrays
  - Chunked processing for arrays that don't fit in memory
  - Metadata preservation during compression
  - Optimized for scientific data patterns

### Data Validation Features

The validation module provides:
- Checksum and integrity verification:
  - Multiple checksum algorithms (CRC32, SHA-256, BLAKE3)
  - File integrity metadata generation and validation
  - Array integrity verification
  - Directory manifests with checksums
- Format validation:
  - Automatic format detection
  - Format-specific validators for scientific data formats
  - Detailed validation reports
  - Structure validation for formats like CSV, JSON, and ARFF
- Integration features:
  - Validation reports with machine-readable output
  - Checksum file generation and verification
  - Directory traversal and recursive validation

### NetCDF File Format

The NetCDF module supports:
- NetCDF3 Classic format:
  - Dimensions (fixed and unlimited)
  - Variables with multiple dimensions
  - Global and variable attributes
  - Common data types (byte, short, int, float, double)
- Features:
  - Create new NetCDF files with hierarchical structure
  - Define dimensions and variables
  - Write array data to variables
  - Add metadata through attributes
  - Read existing NetCDF files

### HDF5 File Format

The HDF5 module provides:
- Hierarchical data organization:
  - Groups (similar to directories)
  - Datasets (n-dimensional arrays)
  - Attributes on groups and datasets
- Data types:
  - Integers, floats, strings
  - Compound types and arrays
- Advanced features:
  - Compression options (gzip, szip, lzf)
  - Chunked storage for large datasets
  - Fletcher32 checksums
  - Metadata management

### Matrix Market Format

The Matrix Market module supports:
- Sparse matrix formats:
  - Coordinate (COO) format
  - Support for symmetric, hermitian, and skew-symmetric matrices
- Data types:
  - Real, complex, integer, and pattern matrices
- Features:
  - Read and write Matrix Market files
  - Automatic format detection
  - Conversion between 0-based and 1-based indexing
  - Integration with sparse matrix operations

### Enhanced Sparse Matrix Support

The serialization module now includes:
- Multiple sparse matrix formats:
  - COO (Coordinate) - flexible, easy to construct
  - CSR (Compressed Sparse Row) - efficient row operations
  - CSC (Compressed Sparse Column) - efficient column operations
- Features:
  - Automatic format conversion with caching
  - Sparse matrix operations (addition, transpose, multiplication)
  - Memory efficiency analysis
  - Integration with Matrix Market format
  - Serialization in multiple formats (binary, JSON, MessagePack)

## Contributing

See the [CONTRIBUTING.md](../CONTRIBUTING.md) file for contribution guidelines.

## License

This project is dual-licensed under:

- [MIT License](../LICENSE-MIT)
- [Apache License Version 2.0](../LICENSE-APACHE)

You can choose to use either license. See the [LICENSE](../LICENSE) file for details.
