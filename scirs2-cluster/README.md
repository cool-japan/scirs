# scirs2-cluster

[![crates.io](https://img.shields.io/crates/v/scirs2-cluster.svg)](https://crates.io/crates/scirs2-cluster)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](../LICENSE)
[![Documentation](https://img.shields.io/docsrs/scirs2-cluster)](https://docs.rs/scirs2-cluster)
[![Status](https://img.shields.io/badge/status-stable-brightgreen.svg)]()

Comprehensive clustering algorithms for unsupervised learning in Rust, part of the [SciRS2](https://github.com/cool-japan/scirs) scientific computing ecosystem.

## Overview

`scirs2-cluster` provides production-ready implementations of classical and modern clustering algorithms with SciPy/scikit-learn compatible APIs. Since v0.5.0, it has significantly expanded beyond the core algorithms with Gaussian Mixture Models, Self-Organizing Maps, topological clustering, streaming/online methods, fuzzy clustering, deep clustering, Bayesian nonparametric methods, and advanced validation tools. Current release: v0.6.2.

Validated by `cargo nextest`: 962/962 tests passing with default features, 1061/1061 passing with all features enabled (0 failures either way).

## Features

### Partitional Clustering (Vector Quantization)
- K-means with multiple initialization strategies
- K-means++ smart initialization (faster convergence)
- Mini-batch K-means for large-scale datasets
- Parallel K-means using Rayon
- `kmeans2` with SciPy-compatible interface
- Data whitening / normalization utilities

### Hierarchical Clustering
- Agglomerative clustering with full linkage method suite: single, complete, average, Ward, centroid, median, weighted
- Optimized Ward's method: O(n^2 log n) vs naive O(n^3)
- Dendrogram utilities and flat cluster extraction (`fcluster`)
- Dendrogram export (Newick, JSON)

### Density-Based Clustering
- DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
- OPTICS (Ordering Points To Identify the Clustering Structure)
- HDBSCAN (Hierarchical DBSCAN)
- Density peaks algorithm
- Density ratio estimation clustering

### Probabilistic and Mixture Models
- Gaussian Mixture Models (GMM) with full EM algorithm
- Bayesian GMM with variational inference
- Dirichlet Process mixture models (nonparametric Bayesian)
- Probabilistic soft assignments

### Prototype-Based and Competitive Learning
- Self-Organizing Maps (SOM) with hexagonal and rectangular topologies
- Competitive learning networks
- Prototype-enhanced clustering (Neural Gas, Growing Neural Gas, LVQ/GLVQ)
- Leader algorithm (single-pass with hierarchical tree)

### Spectral and Graph-Based
- Spectral clustering with multiple Laplacian variants
- Affinity propagation (exemplar-based)
- BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies)
- Mean-shift clustering

### Subspace Clustering
- Subspace clustering for high-dimensional data
- Projected clustering and axis-aligned subspace search
- Advanced subspace methods: Sparse Subspace Clustering (SSC), Low-Rank Subspace Clustering (`subspace_enhanced.rs`)

### Fuzzy and Soft Clustering
- Fuzzy c-means (FCM) with membership degree outputs
- Soft clustering with probabilistic assignments
- Possibilistic c-means

### Topological Clustering
- Topological data analysis applied to clustering
- Persistent homology-based cluster boundary detection
- Mapper algorithm integration

### Streaming and Online Clustering
- Online k-means (incremental updates)
- ADWIN-based streaming cluster detection
- CluStream and DenStream for data streams
- Reservoir sampling for large data streams

### Time Series Clustering
- DTW-based distance for time series k-means
- Temporal pattern clustering

### Ensemble and Consensus
- Consensus clustering via co-association matrices
- Evidence Accumulation Clustering (EAC)
- Bagging-based and weighted voting ensembles
- Stability-based cluster selection

### Deep Clustering
- Deep embedding via autoencoder
- DEC (Deep Embedded Clustering)
- Transformer-based cluster embeddings

### Biclustering and Co-clustering
- Biclustering for simultaneous row/column clustering
- Co-clustering (information-theoretic)

### Evaluation Metrics
- Silhouette coefficient (individual and average)
- Davies-Bouldin index
- Calinski-Harabasz index
- Gap statistic for optimal k selection
- Adjusted Rand Index (ARI)
- Normalized Mutual Information (NMI)
- Homogeneity, Completeness, V-measure
- Stability analysis across bootstrap samples

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
scirs2-cluster = "0.6.2"
```

Rayon-based parallel processing (parallel K-means, parallel linkage, etc.) is included by default — no extra feature flag is required. To additionally enable SIMD-accelerated distance computations:

```toml
[dependencies]
scirs2-cluster = { version = "0.6.2", features = ["simd"] }
```

### K-means Clustering

```rust
use scirs2_cluster::vq::kmeans;
use scirs2_core::ndarray::Array2;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data = Array2::from_shape_vec((6, 2), vec![
        1.0, 2.0,  1.2, 1.8,  0.8, 1.9,
        3.7, 4.2,  3.9, 3.9,  4.2, 4.1,
    ])?;

    let (centroids, labels) = kmeans(data.view(), 2, None, None, None, None)?;

    println!("Centroids: {:?}", centroids);
    println!("Labels: {:?}", labels);
    Ok(())
}
```

### Hierarchical Clustering

```rust
use scirs2_cluster::hierarchy::{linkage, fcluster, LinkageMethod, Metric};
use scirs2_core::ndarray::Array2;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data = Array2::from_shape_vec((6, 2), vec![
        1.0, 2.0,  1.2, 1.8,  0.8, 1.9,
        3.7, 4.2,  3.9, 3.9,  4.2, 4.1,
    ])?;

    let z = linkage(data.view(), LinkageMethod::Ward, Metric::Euclidean)?;
    let labels = fcluster(&z, 2, None)?;

    println!("Cluster assignments: {:?}", labels);
    Ok(())
}
```

### DBSCAN

```rust
use scirs2_cluster::density::dbscan;
use scirs2_core::ndarray::Array2;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data = Array2::from_shape_vec((8, 2), vec![
        1.0, 2.0,  1.5, 1.8,  1.3, 1.9,
        5.0, 7.0,  5.1, 6.8,  5.2, 7.1,
        0.0, 10.0, 10.0, 0.0,
    ])?;

    // eps=0.8, min_samples=2
    let labels = dbscan(data.view(), 0.8, 2, None)?;
    println!("Labels (-1 = noise): {:?}", labels);
    Ok(())
}
```

### Gaussian Mixture Model

```rust
use scirs2_cluster::soft_clustering::GaussianMixtureModel;
use scirs2_core::ndarray::Array2;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data = Array2::<f64>::zeros((100, 2)); // replace with real data

    // fit(data, n_components, max_iter, tol)
    let gmm = GaussianMixtureModel::fit(data.view(), 3, 100, 1e-6)?;

    let labels = gmm.predict(data.view())?;
    let responsibilities = gmm.predict_proba(data.view())?;
    println!("Soft assignments shape: {:?}", responsibilities.shape());
    Ok(())
}
```

### Cluster Validation

```rust
use scirs2_cluster::metrics::{
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
};
use scirs2_core::ndarray::{Array2, Array1};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data = Array2::<f64>::zeros((100, 5));
    let labels = Array1::<i32>::zeros(100);

    let sil = silhouette_score(data.view(), labels.view())?;
    let db  = davies_bouldin_score(data.view(), labels.view())?;
    let ch  = calinski_harabasz_score(data.view(), labels.view())?;

    println!("Silhouette: {:.4}", sil);
    println!("Davies-Bouldin: {:.4}", db);
    println!("Calinski-Harabasz: {:.4}", ch);
    Ok(())
}
```

## Feature Flags

| Flag | Description |
|------|-------------|
| `simd` | SIMD-accelerated distance computations |

Rayon-based parallel processing is always compiled in (via `scirs2-core`); there is no separate `parallel` opt-in flag.

## Related Crates

- [`scirs2-stats`](../scirs2-stats) - Statistical distributions and tests
- [`scirs2-transform`](../scirs2-transform) - Dimensionality reduction and preprocessing
- [`scirs2-spatial`](../scirs2-spatial) - Spatial indexing (KD-tree, Ball-tree)
- [SciRS2 project](https://github.com/cool-japan/scirs)

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](../LICENSE) for details.
