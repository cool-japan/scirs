# scirs2-datasets TODO

This module provides dataset loading utilities similar to scikit-learn's datasets module and SciPy's datasets module.

## Current Status

- [x] Set up module structure
- [x] Error handling
- [x] Initial implementation completed
- [x] Core functionality working
- [x] Examples added

## Implemented Features

- [x] Implement standard test datasets
  - [x] Toy datasets for testing algorithms
  - [x] Small sample datasets for examples
  - [x] Synthetic data generators
- [x] Implement dataset loading utilities
  - [x] Downloading and caching (feature-gated)
  - [x] Memory-efficient loading
  - [x] Data preprocessing
- [x] Add dataset manipulation tools
  - [x] Splitting (train/test)
- [x] Implement dataset conversion utilities
  - [x] Format conversion (CSV, JSON)
  - [x] Normalization
- [x] Add data generation utilities
  - [x] Random data generators
  - [x] Structured data generators (blobs, classification, regression)
  - [x] Time series generators
- [x] Implement metadata handling
  - [x] Dataset descriptions
  - [x] Feature information
  - [x] Target information
- [x] Add examples and documentation
  - [x] Examples for loading toy datasets
  - [x] Examples for data generation
  - [x] Examples for CSV loading

## Data Management and Distribution

- [ ] Remote dataset management
  - [ ] GitHub-based dataset storage model
  - [ ] SHA256 verification for dataset integrity
  - [ ] Create registry system for mapping datasets to files
  - [ ] Version tracking for dataset updates
- [ ] Local cache management
  - [ ] Platform-specific cache locations
  - [ ] Cache clearing and management utilities
  - [ ] Download control for offline environments
  - [ ] Cache size monitoring and limits
- [ ] Dataset download system
  - [ ] Robust download handling with retries
  - [ ] Resumable downloads for large datasets
  - [ ] Progress reporting during downloads
  - [ ] Proxy and authentication support

## Standard Datasets

- [ ] Test image datasets
  - [ ] Face dataset
  - [ ] Ascent dataset (or equivalent)
  - [ ] Sample photography dataset
- [x] Time series datasets
  - [x] Electrocardiogram dataset
  - [x] Stock market dataset
  - [x] Weather time series
- [ ] Add real-world datasets
  - [x] Classic machine learning datasets (iris, breast cancer, digits, boston)
  - [ ] Domain-specific scientific datasets
    - [ ] Astronomy datasets
    - [ ] Materials science datasets
    - [ ] Environmental datasets
  - [ ] Specialized time series datasets
    - [ ] Physiological signals
    - [ ] Economic indicators
    - [ ] Sensor data
- [ ] Image datasets
  - [ ] Test images for processing algorithms
  - [ ] Small labeled image collections
  - [ ] Multi-resolution image samples

## Dataset Manipulation and Processing

- [ ] Extend dataset manipulation tools
  - [ ] Cross-validation utilities
    - [ ] K-fold cross-validation
    - [ ] Stratified cross-validation
    - [ ] Time series cross-validation
  - [ ] Sampling and bootstrapping
    - [ ] Random sampling
    - [ ] Stratified sampling
    - [ ] Importance sampling
  - [ ] Balancing utilities
    - [ ] Oversampling
    - [ ] Undersampling
    - [ ] Synthetic sample generation
- [ ] Enhance dataset conversion utilities
  - [ ] Feature extraction
    - [ ] Basic transformations
    - [ ] Statistical feature generation
    - [ ] Domain-specific feature extraction
  - [ ] Format transformations
    - [ ] Support for specialized formats
    - [ ] Schema conversion
    - [ ] Type adaptation

## Data Generation

- [ ] Enhanced synthetic data generators
  - [ ] Complex pattern generation
    - [ ] Non-linear patterns
    - [ ] Clustered data with specific properties
    - [ ] Hierarchical structure generation
  - [ ] Domain-specific generators
    - [ ] Scientific data generators
    - [ ] Physical simulation data
    - [ ] Network/graph data
  - [ ] Realistic noise models
    - [ ] Missing data generation
    - [ ] Outlier injection
    - [ ] Domain-specific noise patterns

## API and Usability

- [ ] Consistent API design
  - [ ] Streamlined dataset interfaces
  - [ ] Common accessor patterns
  - [ ] Unified metadata format
- [ ] Batch operations
  - [ ] Batch download capability for all datasets
  - [ ] Batch processing utilities
  - [ ] Selective cache management
- [ ] Integration interfaces
  - [ ] Integration with array libraries
  - [ ] Integration with data frame implementations
  - [ ] Streaming interfaces for large datasets

## Performance Optimization

- [ ] Memory usage optimization
  - [ ] Lazy loading for large datasets
  - [ ] Memory-mapped data access
  - [ ] Progressive loading options
- [ ] Processing optimization
  - [ ] Parallel processing for data transformations
  - [ ] SIMD acceleration where applicable
  - [ ] GPU utilization for large datasets
- [ ] Storage optimization
  - [ ] Compressed storage formats
  - [ ] Sparse representation support
  - [ ] Efficient serialization/deserialization

## Documentation

- [ ] Comprehensive documentation
  - [ ] Dataset catalog with descriptions
  - [ ] API reference documentation
  - [ ] Usage tutorials and examples
  - [ ] Performance considerations
- [ ] Dataset visualization
  - [ ] Preview utilities for quick exploration
  - [ ] Statistical summaries
  - [ ] Interactive examples (where applicable)

## Code Cleanup Tasks

- [x] Fix warnings about unused imports 
- [x] Fix warning about dead code in time_series.rs
- [x] Mark doc tests with remote dependencies as ignored
- [ ] Add missing documentation for public items
- [ ] Improve error handling for edge cases
- [ ] Add more unit tests
- [ ] Optimize serialization for larger datasets

## Long-term Goals

- [ ] Comprehensive dataset collection for scientific computing
  - [ ] Parity with scikit-learn and SciPy datasets
  - [ ] Rust-specific dataset collections
- [ ] Efficient handling of large datasets
  - [ ] Distributed loading capabilities
  - [ ] Out-of-core processing
  - [ ] Cloud storage integration
- [ ] Integration with other modules for seamless data flow
  - [ ] Pipeline APIs for data processing
  - [ ] Integration with visualization tools
  - [ ] Export/import with I/O module
- [ ] Support for streaming and online datasets
  - [ ] Real-time data sources
  - [ ] Streaming data processors
  - [ ] Online learning utilities
- [ ] Performance optimizations for data loading
  - [ ] Custom data formats for fast loading
  - [ ] Indexed access for partial loading
  - [ ] SIMD and multi-threading
- [ ] Domain-specific datasets for various scientific fields
  - [ ] Field-specific preprocessors
  - [ ] Specialized metadata handlers
  - [ ] Domain validation utilities