# SciRS2 Datasets

[![crates.io](https://img.shields.io/crates/v/scirs2-datasets.svg)](https://crates.io/crates/scirs2-datasets)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](../LICENSE)
[![Documentation](https://img.shields.io/docsrs/scirs2-datasets)](https://docs.rs/scirs2-datasets)

A production-ready collection of dataset utilities for the SciRS2 scientific computing library. This module provides comprehensive functionality for loading, generating, and working with datasets commonly used in scientific computing, machine learning, and statistical analysis.

## 🚀 Production Status - Final Alpha (0.1.0-alpha.6)

This is the final alpha release with all core functionality implemented, thoroughly tested (117+ tests), and production-ready. The API is stable and follows Rust best practices with zero-warning builds.

## ✨ Features

- **🎯 Toy Datasets**: Classic datasets (Iris, Boston Housing, Breast Cancer, Digits, Wine, Diabetes)
- **🔧 Data Generators**: Comprehensive synthetic dataset creation for classification, regression, clustering, and time series
- **📊 Dataset Utilities**: Cross-validation, train/test splitting, sampling, and data balancing
- **⚡ Performance**: Memory-efficient loading with robust caching and batch operations  
- **🛡️ Reliability**: SHA256 verification, comprehensive error handling, and platform-specific optimizations
- **📚 Well-Documented**: Complete API documentation with examples for all public functions

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
scirs2-datasets = "0.1.0-alpha.6"
```

For remote dataset downloading capabilities:

```toml
[dependencies]
scirs2-datasets = { version = "0.1.0-alpha.6", features = ["download"] }
```

## Quick Start

### Load Classic Datasets

```rust
use scirs2_datasets::{load_iris, load_boston, Dataset};

// Load the Iris dataset
let iris = load_iris()?;
println!("Iris: {} samples, {} features", iris.n_samples(), iris.n_features());

// Load Boston housing dataset  
let boston = load_boston()?;
println!("Boston: {} samples, {} features", boston.n_samples(), boston.n_features());
```

### Generate Synthetic Data

```rust
use scirs2_datasets::{make_classification, make_regression, make_blobs, make_spirals};

// Classification dataset
let dataset = make_classification(1000, 10, 3, 2, 4, Some(42))?;
println!("Classification: {} samples, {} features", dataset.n_samples(), dataset.n_features());

// Non-linear patterns
let spirals = make_spirals(500, 2, 0.1, Some(42))?;
let blobs = make_blobs(300, 2, 4, 1.0, Some(42))?;
```

### Cross-Validation and Splitting

```rust
use scirs2_datasets::{load_iris, k_fold_split, stratified_k_fold_split, train_test_split};

let iris = load_iris()?;

// K-fold cross-validation
let folds = k_fold_split(iris.n_samples(), 5, true, Some(42))?;

// Stratified splitting with targets
if let Some(target) = &iris.target {
    let stratified_folds = stratified_k_fold_split(target, 5, true, Some(42))?;
    let (train_idx, test_idx) = train_test_split(iris.n_samples(), 0.8, Some(42))?;
}
```

## Core Components

### 🎯 Toy Datasets
Pre-loaded classic datasets for immediate use:

```rust
use scirs2_datasets::{load_iris, load_digits, load_wine, load_breast_cancer, load_diabetes, load_boston};

// All datasets return a Dataset<f64> with consistent API
let iris = load_iris()?;          // 150 samples, 4 features, 3 classes
let digits = load_digits()?;      // 1797 samples, 64 features, 10 classes  
let wine = load_wine()?;          // 178 samples, 13 features, 3 classes
let cancer = load_breast_cancer()?; // 569 samples, 30 features, 2 classes
let diabetes = load_diabetes()?;  // 442 samples, 10 features, regression
let boston = load_boston()?;      // 506 samples, 13 features, regression
```

### 🔧 Data Generators
Comprehensive synthetic dataset creation:

```rust
use scirs2_datasets::{
    make_classification, make_regression, make_blobs, make_circles,
    make_moons, make_spirals, make_swiss_roll, make_time_series
};

// Linear and non-linear patterns
let classification = make_classification(500, 8, 2, 1, 2, Some(42))?;
let regression = make_regression(400, 5, 3, 0.1, Some(42))?;
let circles = make_circles(300, 0.1, Some(42))?;
let moons = make_moons(200, 0.05, Some(42))?;

// Complex patterns
let spirals = make_spirals(600, 3, 0.2, Some(42))?;
let swiss_roll = make_swiss_roll(800, 0.1, Some(42))?;

// Time series
let ts = make_time_series(1000, 24, 0.1, Some(42))?;
```

### 📊 Dataset Utilities
Complete toolkit for dataset manipulation:

```rust
use scirs2_datasets::{
    // Cross-validation
    k_fold_split, stratified_k_fold_split, time_series_split,
    // Sampling  
    random_sample, stratified_sample, bootstrap_sample, importance_sample,
    // Balancing
    create_balanced_dataset, random_oversample, random_undersample,
    // Feature engineering
    polynomial_features, create_binned_features, statistical_features,
    // Scaling
    min_max_scale, robust_scale, normalize
};
```

### ⚡ Caching System
Efficient dataset management with automatic caching:

```rust
use scirs2_datasets::{CacheManager, DatasetCache};

let cache = CacheManager::new()?;
let stats = cache.get_statistics()?;
println!("Cache contains {} datasets using {} MB", 
         stats.total_files, stats.total_size_mb);
```

## Performance & Reliability

- **Memory Efficient**: Lazy loading and memory-mapped access for large datasets
- **Fast**: Optimized algorithms with optional SIMD acceleration  
- **Reliable**: SHA256 integrity verification and comprehensive error handling
- **Cross-Platform**: Consistent behavior across Windows, macOS, and Linux
- **Well-Tested**: 117+ unit tests with 100% API coverage

## API Stability

The API is stable and production-ready. All public functions are thoroughly documented with examples. Breaking changes will only occur in major version updates (1.0.0+).

## Integration

Seamlessly integrates with other SciRS2 modules:

```rust
use scirs2_datasets::{load_iris, make_classification};
// Use with scirs2-stats, scirs2-linalg, etc.
```

## Contributing

See the [project CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines. Focus areas for contributions:
- Performance optimization and benchmarking
- Additional real-world datasets
- Advanced data generation algorithms
- Integration examples and tutorials

## License

Dual-licensed under [MIT](../LICENSE-MIT) or [Apache License 2.0](../LICENSE-APACHE).