# scirs2-transform

[![crates.io](https://img.shields.io/crates/v/scirs2-transform.svg)](https://crates.io/crates/scirs2-transform)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](../LICENSE)
[![Documentation](https://img.shields.io/docsrs/scirs2-transform)](https://docs.rs/scirs2-transform)
[![Version](https://img.shields.io/badge/version-0.4.3-green)]()

Data transformation, dimensionality reduction, and feature engineering library for machine learning in Rust, part of the [SciRS2](https://github.com/cool-japan/scirs) scientific computing ecosystem.

## Overview

`scirs2-transform` provides comprehensive data preprocessing and transformation utilities following scikit-learn's `fit` / `transform` / `fit_transform` API pattern. v0.4.3 carries forward the full suite of UMAP, Barnes-Hut t-SNE, persistent homology / TDA, metric learning, kernel methods, optimal transport, and advanced NMF variants introduced earlier in the v0.4.x line.

## Features

### Normalization and Scaling
- Min-Max scaling to `[0, 1]` or custom ranges
- Z-score standardization (zero mean, unit variance)
- Robust scaling (median and IQR; outlier-resistant)
- Max-absolute scaling
- L1 / L2 vector normalization
- Quantile normalization
- Reusable `Normalizer` with `fit` / `transform` / `inverse_transform`

### Feature Engineering
- Polynomial features (degree 2+, with/without interaction-only mode)
- Box-Cox and Yeo-Johnson power transformations with optimal lambda estimation
- Equal-width and equal-frequency discretization (binning)
- Binarization with configurable thresholds
- Log transformations with epsilon handling
- Interaction terms, custom function transformers

### Dimensionality Reduction
- PCA (Principal Component Analysis) with centering/scaling, explained variance ratio
- Truncated SVD (memory-efficient for sparse data)
- Linear Discriminant Analysis (LDA) for supervised reduction
- t-SNE with Barnes-Hut approximation (O(n log n), multicore)
- UMAP (Uniform Manifold Approximation and Projection)
- Isomap (geodesic-distance manifold learning)
- Locally Linear Embedding (LLE)
- Kernel PCA (RBF, polynomial, sigmoid kernels)
- Probabilistic PCA (PPCA)
- Factor analysis

### Independent Component Analysis
- FastICA (fixed-point iteration)
- Spatial ICA
- Infomax ICA

### Non-Negative Matrix Factorization (NMF) Variants
- Standard NMF (multiplicative update rules)
- Sparse NMF
- Convex NMF
- Semi-NMF
- Online NMF for streaming data

### Sparse PCA and Dictionary Learning
- Sparse PCA via LASSO-based encoding
- Dictionary learning (K-SVD style)
- Sparse coding and reconstruction

### Metric Learning
- Mahalanobis distance learning
- LMNN (Large Margin Nearest Neighbor)
- NCA (Neighborhood Components Analysis)
- Metric learning extensions (`metric_learning_ext/`)

### Kernel Methods
- Kernel PCA with multiple kernels
- Deep kernel learning
- Random Fourier Features (RFF) for large-scale kernel approximation
- Orthogonal Random Features (ORF)
- Nystrom approximation

### Optimal Transport
- Wasserstein distance computation
- Sinkhorn-Knopp regularized OT
- Sliced Wasserstein distance
- OT-based domain adaptation

### Topological Data Analysis (TDA)
- Vietoris-Rips complex construction
- Persistent homology computation (Betti numbers, persistence diagrams)
- Persistence landscape features
- Topological feature vectorization
- Persistent diagram analysis

### Archetypal Analysis
- Archetypal analysis (convex hull vertex finding)
- Simplex-based data representation

### Autoencoder-Based Reduction
- Linear autoencoder as PCA surrogate
- Nonlinear autoencoder reduction

### Categorical Encoding
- One-hot encoding (sparse and dense)
- Ordinal / label encoding
- Target encoding with regularization
- Binary encoding for high-cardinality features
- Unknown category handling strategies

### Missing Value Imputation
- Simple imputation: mean, median, mode, constant
- KNN imputation with multiple distance metrics
- Iterative imputation (MICE algorithm)
- Missing indicator tracking

### Feature Selection
- Variance threshold filter
- Recursive Feature Elimination (RFE)
- Mutual information-based selection

### Pipeline API
- Sequential transformation chains
- `ColumnTransformer` for per-column transforms
- `fit` / `transform` / `fit_transform` / `inverse_transform` throughout

### Signal Transforms (Integrated)
- Discrete Wavelet Transform (DWT): Haar, Daubechies, Symlet, Coiflet; multi-level
- 2D DWT for image decomposition (LL, LH, HL, HH subbands)
- Continuous Wavelet Transform (CWT): Morlet, Mexican Hat, Gaussian
- Wavelet Packet Transform (WPT) with best-basis selection
- Short-Time Fourier Transform (STFT) with multiple window functions
- Spectrograms (power, magnitude, dB scale)
- MFCC with mel filterbank, delta and delta-delta features
- Constant-Q Transform (CQT) for musical analysis
- Chromagram (12-bin pitch class profiles)

### Multi-View Learning
- Multi-view PCA and CCA
- Consensus multi-view embedding

### Online / Incremental Learning
- Incremental PCA (chunk-by-chunk update)
- Online NMF
- Online t-SNE approximations

### Out-of-Core Processing
- Chunked array reader/writer for datasets larger than RAM
- Streaming normalizer with partial-fit

### Structure Learning
- Covariance structure estimation
- Graphical LASSO for sparse inverse covariance

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
scirs2-transform = "0.4.3"
```

With SIMD and parallel features:

```toml
[dependencies]
scirs2-transform = { version = "0.4.3", features = ["parallel", "simd"] }
```

### Normalization

```rust
use scirs2_transform::normalize::{normalize_array, NormalizationMethod, Normalizer};
use scirs2_core::ndarray::Array2;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data = Array2::<f64>::from_shape_vec((4, 3), vec![
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
        10.0, 11.0, 12.0,
    ])?;

    // One-shot Z-score normalization
    let normalized = normalize_array(&data, NormalizationMethod::ZScore, 0)?;

    // Reusable normalizer (fit on train, apply to test)
    let mut scaler = Normalizer::new(NormalizationMethod::MinMax, 0);
    let train_scaled = scaler.fit_transform(&data)?;
    // let test_scaled = scaler.transform(&test_data)?;

    println!("Normalized shape: {:?}", normalized.shape());
    Ok(())
}
```

### PCA

```rust
use scirs2_transform::reduction::PCA;
use scirs2_core::ndarray::Array2;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data = Array2::<f64>::zeros((200, 50)); // high-dimensional data

    let mut pca = PCA::new(10, true, false); // 10 components, center=true
    let reduced = pca.fit_transform(&data)?;

    if let Some(evr) = pca.explained_variance_ratio() {
        let total: f64 = evr.iter().take(10).sum();
        println!("Explained variance (10 components): {:.1}%", total * 100.0);
    }
    println!("Reduced shape: {:?}", reduced.shape()); // (200, 10)
    Ok(())
}
```

### UMAP

```rust
use scirs2_transform::umap::UMAP;
use scirs2_core::ndarray::Array2;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data = Array2::<f64>::zeros((500, 100));

    let mut umap = UMAP::new(2)     // 2D embedding
        .with_n_neighbors(15)
        .with_min_dist(0.1);
    let embedding = umap.fit_transform(&data)?;

    println!("UMAP embedding shape: {:?}", embedding.shape()); // (500, 2)
    Ok(())
}
```

### Persistent Homology (TDA)

```rust
use scirs2_transform::tda::{VietorisRips, PersistentHomology};
use scirs2_core::ndarray::Array2;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let points = Array2::<f64>::zeros((50, 3));

    let vr = VietorisRips::new(2.0, 1); // max_radius, max_dim
    let complex = vr.build(points.view())?;

    let ph = PersistentHomology::new();
    let diagrams = ph.compute(&complex)?;

    println!("H0 features: {}", diagrams[0].len());
    println!("H1 features: {}", diagrams[1].len());
    Ok(())
}
```

### Optimal Transport

```rust
use scirs2_transform::optimal_transport::{sinkhorn, wasserstein_distance};
use scirs2_core::ndarray::{Array1, Array2};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let source = Array1::<f64>::from_vec(vec![0.5, 0.5]);
    let target = Array1::<f64>::from_vec(vec![0.3, 0.7]);
    let cost = Array2::<f64>::from_shape_vec((2, 2), vec![0.0, 1.0, 1.0, 0.0])?;

    // Sinkhorn regularized OT (reg=0.1)
    let (transport_plan, ot_cost) = sinkhorn(
        source.view(), target.view(), cost.view(), 0.1, 100
    )?;
    println!("OT cost: {:.4}", ot_cost);
    Ok(())
}
```

## Feature Flags

| Flag | Description |
|------|-------------|
| `parallel` | Enable Rayon-based multi-threaded transforms |
| `simd` | SIMD-accelerated normalization and distance operations |

## Related Crates

- [`scirs2-cluster`](../scirs2-cluster) - Clustering algorithms
- [`scirs2-linalg`](../scirs2-linalg) - Linear algebra (SVD, eigendecomposition)
- [`scirs2-fft`](../scirs2-fft) - FFT operations (used by signal transforms)
- [SciRS2 project](https://github.com/cool-japan/scirs)

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](../LICENSE) for details.
