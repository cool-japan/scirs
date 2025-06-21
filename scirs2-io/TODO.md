# scirs2-io TODO

This module provides input/output functionality for scientific data formats similar to SciPy's io module.

## Current Status

- [x] Set up module structure
- [x] Error handling
- [x] Initial implementation in progress
- [x] Implemented ARFF file format support (Attribute-Relation File Format)
- [x] Implemented MATLAB file format support (.mat)
- [x] Implemented WAV audio file support
- [x] **NEW**: Parallel compression/decompression capabilities with significant performance improvements
- [x] **NEW**: Comprehensive schema-based validation system with JSON Schema compatibility
- [x] **NEW**: NetCDF4/HDF5 integration with enhanced features and backward compatibility
  - Enhanced NetCDF module with dual backend support (Classic NetCDF3 + NetCDF4/HDF5)
  - Added NetCDFFormat enum (Classic, NetCDF4, NetCDF4Classic) for format selection
  - Convenient write_array() API for NetCDF4 with automatic dimension/variable creation
  - HDF5 backend integration providing large file support and compression capabilities
  - Auto-detection of format when reading NetCDF files
  - Comprehensive example demonstrating both formats and advanced features
  - Enhanced NetCDFOptions with compression and chunking support
- [x] **NEW**: Iterator interfaces for large data handling with comprehensive streaming support
  - ChunkedReader for memory-efficient file processing in configurable chunks
  - LineChunkedReader for line-based processing with batching support
  - StreamingCsvReader for streaming CSV processing with header support
  - Comprehensive performance monitoring and statistics tracking
  - Integration with Rust's iterator ecosystem for seamless data processing
- [x] **NEW**: Async I/O support with tokio integration and concurrent processing
  - AsyncChunkedReader for non-blocking file reading with configurable concurrency
  - AsyncLineReader for asynchronous line-based processing
  - Concurrent processing with backpressure handling and timeout support
  - Cancellation tokens for graceful operation termination
  - Process multiple chunks concurrently with configurable worker pools
  - Real-time progress monitoring for async operations

## File Format Support

- [x] Basic file format support
  - [x] ARFF (Attribute-Relation File Format)
  - [x] MATLAB .mat file format
  - [x] WAV audio file format
- [ ] Enhanced existing format support
  - [ ] Expand MATLAB format support
    - [ ] Support for newer MATLAB versions (v7.3+)
    - [ ] Improved handling of sparse matrices
    - [ ] Support for cell arrays and structs
  - [ ] Enhance WAV file handling
    - [ ] Support for additional compression formats
    - [ ] Metadata handling improvements
    - [ ] Multi-channel processing utilities
  - [ ] Extend ARFF format functionality
    - [ ] Improved sparse data support
    - [ ] Custom type mapping
    - [ ] Enhanced error handling
- [ ] Additional scientific file formats
  - [x] CSV and delimited text files
    - [x] Basic CSV reading/writing
    - [x] Type conversion and detection
    - [x] Missing value handling
    - [x] Processing large files in chunks
    - [x] More data type support (date, time, complex numbers)
  - [x] Matrix Market format
    - [x] High-performance implementation
    - [x] Parallel processing for large matrices
    - [x] Support for both dense and sparse matrices
  - [x] Harwell-Boeing sparse matrix format
    - [x] Reading/writing support
    - [x] Conversion to/from other sparse formats
  - [x] NetCDF file format
    - [x] Basic NetCDF3 reading/writing (skeleton implemented, needs refinement)
    - [x] **COMPLETED**: NetCDF4/HDF5 integration with enhanced features
    - [x] Dimension and attribute handling (basic support)
  - [x] HDF5 file format
    - [x] Reading/writing support
    - [x] Group and dataset management
    - [x] Attribute handling
  - [ ] IDL file format
    - [ ] Reading support for IDL save files
    - [ ] Conversion to/from native Rust types
  - [ ] Fortran unformatted files
    - [ ] Sequential file support
    - [ ] Record-based file structure

## Matrix and Array I/O

- [ ] Matrix format handling
  - [ ] Common matrix formats (CSR, CSC, COO)
  - [ ] Integration with scirs2-sparse
  - [ ] Efficient matrix serialization
- [ ] Array serialization
  - [x] Basic array serialization
  - [ ] Memory-mapped array I/O
  - [ ] Chunked array reading/writing
  - [ ] Parallel I/O for large arrays
- [ ] Binary array format
  - [ ] Native binary format with metadata
  - [ ] Versioning support
  - [ ] Cross-platform compatibility

## Image File Support

- [x] Basic image file support
  - [x] Read/write common image formats (PNG, JPEG, BMP, TIFF)
  - [x] Metadata handling
  - [x] Image sequence handling
- [ ] Enhanced image capabilities
  - [ ] Multi-scale image support
  - [ ] Lossless compression options
  - [ ] Color space conversions
  - [ ] ICC profile handling
  - [ ] EXIF metadata handling

## Data Compression and Optimization

- [x] Basic data compression
  - [x] Lossless compression for scientific data
  - [x] Dimensionality reduction for storage
- [x] Enhanced compression capabilities
  - [x] Transparent handling of compressed files (.gz, .bz2, .xz)
  - [x] Compression level control
  - [x] Memory-efficient compression/decompression
  - [x] Parallel compression/decompression
- [x] Performance optimizations
  - [x] Thread pool for parallel I/O operations
  - [ ] Streaming I/O for large files
  - [ ] Zero-copy optimizations where possible
  - [ ] Memory mapping for large files

## Data Exchange and Network I/O

- [ ] Network data exchange
  - [ ] Data transfer protocols
  - [ ] Remote data access APIs
  - [ ] HTTP/HTTPS client for data retrieval
  - [ ] WebSocket support for streaming data
- [ ] Cloud storage integration
  - [ ] Amazon S3 support
  - [ ] Google Cloud Storage support
  - [ ] Azure Blob Storage support
- [ ] Distributed data handling
  - [ ] Parallel data loading across nodes
  - [ ] Coordinated I/O operations

## Data Validation and Integrity

- [x] Basic validation utilities
  - [x] Checksum and integrity checking
  - [x] Format validation
- [ ] Enhanced validation features
  - [x] Schema-based validation
  - [ ] Content validation rules
  - [ ] Error recovery options
  - [ ] Corruption detection and handling

## Streaming and Large Data Handling

- [x] Basic streaming capabilities
  - [x] Processing large files in chunks
  - [x] Memory-efficient I/O operations
- [x] **NEW**: Advanced streaming features
  - [x] **COMPLETED**: Iterator interfaces for large data handling with comprehensive streaming support
  - [x] **COMPLETED**: Async I/O support with tokio integration and concurrent processing
  - [ ] Event-based parsing
  - [ ] Resumable I/O operations

## API Design and Usability

- [ ] Consistent API design
  - [ ] Common patterns across format handlers
  - [ ] Fluent builder APIs where appropriate
  - [ ] Error handling consistency
- [ ] User experience improvements
  - [ ] Detailed progress reporting
  - [ ] Cancellation support
  - [ ] Resource management utilities

## Documentation and Examples

- [x] Basic documentation and examples
  - [x] Tutorial for common I/O operations
  - [x] Examples for different file formats
- [ ] Enhanced documentation
  - [ ] Comprehensive API reference
  - [ ] Performance considerations and guidelines
  - [ ] Common patterns and best practices
  - [ ] Troubleshooting guide

## Testing and Quality Assurance

- [x] Enhanced testing
  - [x] Round-trip testing (write→read→compare)
  - [x] Performance benchmarks (comprehensive parallel compression benchmarking)
  - [ ] Comparison with reference implementations
  - [ ] Edge case handling verification
  - [x] Fixed warnings and code quality issues in NetCDF implementation
  - [x] Comprehensive testing for parallel compression and schema validation
  - [x] NetCDF4/HDF5 integration testing with format comparison and advanced features

## Long-term Goals

- [ ] Comprehensive I/O support for scientific data formats
  - [ ] Parity with SciPy's io module
  - [ ] Support for domain-specific formats
- [ ] Efficient handling of large datasets
  - [ ] Out-of-core processing capabilities
  - [ ] Streaming processing for TB-scale data
- [ ] Integration with other modules for seamless data flow
  - [ ] Integration with visualization tools
  - [ ] Pipeline APIs for data processing workflows
- [ ] Support for cloud storage and distributed file systems
  - [ ] Adapters for various storage backends
  - [ ] Consistent API across storage types
- [ ] Performance optimizations for I/O-bound operations
  - [ ] Rust-specific optimizations
  - [ ] SIMD acceleration where applicable
- [ ] Domain-specific I/O utilities for various scientific fields
  - [ ] Bioinformatics file formats
  - [ ] Geospatial data formats
  - [ ] Astronomical data formats