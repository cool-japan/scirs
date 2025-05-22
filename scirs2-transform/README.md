# SciRS2 Transform

[![crates.io](https://img.shields.io/crates/v/scirs2-transform.svg)](https://crates.io/crates/scirs2-transform)
[[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)]](../LICENSE)
[![Documentation](https://img.shields.io/docsrs/scirs2-transform)](https://docs.rs/scirs2-transform)

This crate provides data transformation utilities for the SciRS2 library, designed to mimic the functionality of scikit-learn's preprocessing and decomposition modules.

## Features

### Data Normalization and Standardization

- Min-max scaling
- Z-score standardization (mean 0, std 1)
- Max-absolute scaling
- L1 and L2 normalization
- Normalizer class for fitting and transforming

```rust
use ndarray::array;
use scirs2_transform::normalize::{normalize_array, NormalizationMethod};

// Normalize columns using min-max normalization
let data = array![[1.0, 2.0, 3.0], 
                 [4.0, 5.0, 6.0],
                 [7.0, 8.0, 9.0]];
let normalized = normalize_array(&data, NormalizationMethod::MinMax, 0).unwrap();

// Using the Normalizer for fit-transform workflows
let mut normalizer = Normalizer::new(NormalizationMethod::ZScore, 0);
let train_data = normalizer.fit_transform(&training_data).unwrap();
let test_data = normalizer.transform(&test_data).unwrap();
```

### Feature Engineering

- Polynomial features generation
- Binarization
- Discretization (equal-width and equal-frequency binning)
- Power transformations (Box-Cox and Yeo-Johnson)
- Log transformations

```rust
use ndarray::array;
use scirs2_transform::features::{PolynomialFeatures, binarize};

// Generate polynomial features
let data = array![[1.0, 2.0], [3.0, 4.0]];
let poly = PolynomialFeatures::new(2, false, true);
let poly_features = poly.transform(&data).unwrap();

// Binarize features with a threshold
let data = array![[1.0, -1.0, 2.0], [2.0, 0.0, -3.0]];
let binary = binarize(&data, 0.0).unwrap();
```

### Dimensionality Reduction

- Principal Component Analysis (PCA)
- Truncated Singular Value Decomposition (SVD)
- Linear Discriminant Analysis (LDA)

```rust
use ndarray::array;
use scirs2_transform::reduction::PCA;

// Reduce dimensionality with PCA
let data = array![[1.0, 2.0, 3.0], 
                 [4.0, 5.0, 6.0],
                 [7.0, 8.0, 9.0],
                 [10.0, 11.0, 12.0]];
                 
let mut pca = PCA::new(2, true, false);
let reduced = pca.fit_transform(&data).unwrap();

// Access explained variance ratio
let explained_variance = pca.explained_variance_ratio().unwrap();
println!("Explained variance ratio: {:?}", explained_variance);
```

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
scirs2-transform = "0.1.0-alpha.4"
```

To enable optimizations through the core module, add feature flags:

```toml
[dependencies]
scirs2-transform = { version = "0.1.0-alpha.4", features = ["parallel"] }
```

## Examples

See the documentation for detailed examples of each transformation.

## License

This project is dual-licensed under:

- [MIT License](../LICENSE-MIT)
- [Apache License Version 2.0](../LICENSE-APACHE)

You can choose to use either license. See the [LICENSE](../LICENSE) file for details.
