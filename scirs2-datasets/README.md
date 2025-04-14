# SciRS2 Datasets

[![crates.io](https://img.shields.io/crates/v/scirs2-datasets.svg)](https://crates.io/crates/scirs2-datasets)
[![License](https://img.shields.io/crates/l/scirs2-datasets.svg)](../LICENSE)
[![Documentation](https://img.shields.io/docsrs/scirs2-datasets)](https://docs.rs/scirs2-datasets)

A collection of dataset utilities for the SciRS2 scientific computing library. This module provides functionality for loading, generating, and working with common datasets used in scientific computing, machine learning, and statistical analysis.

## Features

- **Data Loaders**: Functions for loading datasets from various sources
- **Dataset Generators**: Utilities to generate synthetic datasets
- **Toy Datasets**: Pre-defined small datasets for testing and examples
- **Caching**: Efficient caching mechanism for dataset loading
- **Data Sampling**: Tools for sampling from datasets

## Installation

Add the following to your `Cargo.toml`:

```toml
[dependencies]
scirs2-datasets = "0.1.0-alpha.1"
```

To enable additional download and caching features:

```toml
[dependencies]
scirs2-datasets = { version = "0.1.0-alpha.1", features = ["remote-datasets"] }
```

## Usage

Basic usage examples:

```rust
use scirs2_datasets::{loaders, generators, toy};
use scirs2_core::error::CoreResult;

// Load CSV data
fn example_csv_loading() -> CoreResult<()> {
    let csv_path = "data/example.csv";
    let data = loaders::load_csv(csv_path, true)?;
    println!("Loaded {} rows from CSV", data.nrows());
    Ok(())
}

// Generate synthetic data
fn example_data_generation() -> CoreResult<()> {
    // Generate a random classification dataset
    let (features, labels) = generators::make_classification(
        100,    // n_samples
        2,      // n_features
        2,      // n_classes
        1,      // n_clusters_per_class
        0.8,    // class_sep
    )?;
    
    println!("Generated dataset with {} samples", features.nrows());
    Ok(())
}

// Use toy datasets
fn example_toy_dataset() -> CoreResult<()> {
    // Load the iris dataset
    let iris = toy::load_iris()?;
    println!("Iris dataset: {} samples, {} features", 
             iris.data.nrows(), iris.data.ncols());
    println!("Feature names: {:?}", iris.feature_names);
    println!("Target names: {:?}", iris.target_names);
    Ok(())
}
```

## Components

### Loaders

Functions for loading data from various file formats:

```rust
use scirs2_datasets::loaders::{
    load_csv,        // Load data from CSV files
    load_json,       // Load data from JSON files
    load_arff,       // Load data from ARFF files
    load_libsvm,     // Load data from LIBSVM/SVMLight format
};
```

### Generators

Functions for generating synthetic datasets:

```rust
use scirs2_datasets::generators::{
    make_classification,  // Generate a random n-class classification problem
    make_regression,      // Generate a random regression problem
    make_blobs,           // Generate isotropic Gaussian blobs
    make_moons,           // Generate two interleaving half circles
    make_circles,         // Generate a large circle containing a smaller circle
    make_s_curve,         // Generate an S curve dataset
    make_swiss_roll,      // Generate a swiss roll dataset
};
```

### Toy Datasets

Pre-defined datasets for testing and examples:

```rust
use scirs2_datasets::toy::{
    load_iris,        // The classic Iris dataset
    load_digits,      // Handwritten digits dataset
    load_wine,        // Wine recognition dataset
    load_boston,      // Boston house prices dataset
    load_diabetes,    // Diabetes dataset
    load_breast_cancer, // Breast cancer wisconsin dataset
};
```

### Sampling

Utilities for data sampling:

```rust
use scirs2_datasets::sample::{
    train_test_split,       // Split arrays into random train and test subsets
    stratified_split,       // Split preserving the percentage of samples for each class
    bootstrap_sample,       // Generate a bootstrap sample
    resample,               // Resample arrays or matrices
};
```

## Contributing

See the [CONTRIBUTING.md](../CONTRIBUTING.md) file for contribution guidelines.

## License

This project is licensed under the Apache License, Version 2.0 - see the [LICENSE](../LICENSE) file for details.