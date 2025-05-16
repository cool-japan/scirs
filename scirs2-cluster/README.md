# SciRS2 Clustering Module

[![crates.io](https://img.shields.io/crates/v/scirs2-cluster.svg)](https://crates.io/crates/scirs2-cluster)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](../LICENSE)
[![Documentation](https://img.shields.io/docsrs/scirs2-cluster)](https://docs.rs/scirs2-cluster)

A comprehensive clustering module for the SciRS2 scientific computing library in Rust. This crate provides implementations of various clustering algorithms with a focus on performance, flexibility, and idiomatic Rust code.

## Features

* **Vector Quantization**
  * K-means clustering with multiple initialization methods
  * K-means++ smart initialization
  * Enhanced kmeans2 with SciPy-compatible interface
  * Mini-batch K-means for large datasets
  * Data whitening/normalization utilities

* **Hierarchical Clustering**
  * Agglomerative clustering with multiple linkage methods:
    * Single linkage (minimum distance)
    * Complete linkage (maximum distance)
    * Average linkage
    * Ward's method (minimizes variance)
    * Centroid method (distance between centroids)
    * Median method
    * Weighted average
  * Dendrogram utilities and flat cluster extraction
  * Cluster distance metrics (Euclidean, Manhattan, Chebyshev, Correlation)

* **Density-Based Clustering**
  * DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
  * OPTICS (Ordering Points To Identify the Clustering Structure)
  * HDBSCAN (Hierarchical DBSCAN)
  * Support for custom distance metrics

* **Other Algorithms**
  * Mean-shift clustering
  * Spectral clustering
  * Affinity propagation

* **Evaluation Metrics**
  * Silhouette coefficient
  * Davies-Bouldin index
  * Calinski-Harabasz index

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
scirs2-cluster = "0.1.0-alpha.3"
ndarray = "0.15"
```

To enable optimizations through the core module, add feature flags:

```toml
[dependencies]
scirs2-cluster = { version = "0.1.0-alpha.3", features = ["parallel", "simd"] }
```

## Usage

### K-means Example

```rust
use ndarray::Array2;
use scirs2_cluster::vq::{kmeans, KMeansOptions, KMeansInit};

// Create a dataset
let data = Array2::from_shape_vec((6, 2), vec![
    1.0, 2.0,
    1.2, 1.8,
    0.8, 1.9,
    3.7, 4.2,
    3.9, 3.9,
    4.2, 4.1,
]).unwrap();

// Configure K-means
let options = KMeansOptions {
    init_method: KMeansInit::KMeansPlusPlus,
    max_iter: 300,
    ..Default::default()
};

// Run k-means with k=2
let (centroids, labels) = kmeans(data.view(), 2, Some(options)).unwrap();

println!("Centroids: {:?}", centroids);
println!("Cluster assignments: {:?}", labels);
```

### Enhanced kmeans2 (SciPy-compatible)

```rust
use scirs2_cluster::vq::{kmeans2, MinitMethod, MissingMethod, whiten};

// Whiten the data for better clustering
let whitened_data = whiten(&data).unwrap();

// Run kmeans2 with different initialization methods
let (centroids, labels) = kmeans2(
    whitened_data.view(),
    3,                             // k clusters
    Some(10),                      // iterations
    Some(1e-4),                    // threshold
    Some(MinitMethod::PlusPlus),   // K-means++ initialization
    Some(MissingMethod::Warn),     // warn on empty clusters
    Some(true),                    // check finite values
    Some(42),                      // random seed
).unwrap();
```

### Mini-batch K-means

```rust
use scirs2_cluster::vq::{minibatch_kmeans, MiniBatchKMeansOptions};

// Configure mini-batch K-means
let options = MiniBatchKMeansOptions {
    batch_size: 1024,
    max_iter: 100,
    ..Default::default()
};

// Run clustering on large dataset
let (centroids, labels) = minibatch_kmeans(large_data.view(), 5, Some(options)).unwrap();
```

### Hierarchical Clustering Example

```rust
use ndarray::Array2;
use scirs2_cluster::hierarchy::{linkage, fcluster, LinkageMethod};

// Create a dataset
let data = Array2::from_shape_vec((6, 2), vec![
    1.0, 2.0,
    1.2, 1.8,
    0.8, 1.9,
    3.7, 4.2,
    3.9, 3.9,
    4.2, 4.1,
]).unwrap();

// Calculate linkage matrix using Ward's method
let linkage_matrix = linkage(data.view(), LinkageMethod::Ward, None).unwrap();

// Form flat clusters by cutting the dendrogram
let num_clusters = 2;
let labels = fcluster(&linkage_matrix, num_clusters, None).unwrap();

println!("Cluster assignments: {:?}", labels);
```

### Evaluation Metrics

```rust
use scirs2_cluster::metrics::{silhouette_score, davies_bouldin_score, calinski_harabasz_score};

// Evaluate clustering quality
let silhouette = silhouette_score(data.view(), labels.view()).unwrap();
let db_score = davies_bouldin_score(data.view(), labels.view()).unwrap();
let ch_score = calinski_harabasz_score(data.view(), labels.view()).unwrap();

println!("Silhouette score: {}", silhouette);
println!("Davies-Bouldin score: {}", db_score);
println!("Calinski-Harabasz score: {}", ch_score);
```

### DBSCAN Example

```rust
use ndarray::Array2;
use scirs2_cluster::density::{dbscan, labels};

// Create a dataset with clusters and noise
let data = Array2::from_shape_vec((8, 2), vec![
    1.0, 2.0,   // Cluster 1
    1.5, 1.8,   // Cluster 1
    1.3, 1.9,   // Cluster 1
    5.0, 7.0,   // Cluster 2
    5.1, 6.8,   // Cluster 2
    5.2, 7.1,   // Cluster 2
    0.0, 10.0,  // Noise
    10.0, 0.0,  // Noise
]).unwrap();

// Run DBSCAN with eps=0.8 and min_samples=2
let cluster_labels = dbscan(data.view(), 0.8, 2, None).unwrap();

// Count noise points
let noise_count = cluster_labels.iter().filter(|&&label| label == labels::NOISE).count();

println!("Cluster assignments: {:?}", cluster_labels);
println!("Number of noise points: {}", noise_count);
```

## Key Enhancements

### SciPy Compatibility
- APIs designed to match SciPy's cluster module
- Compatible initialization methods and parameters
- Similar return value formats

### Performance Optimizations
- SIMD-accelerated distance computations (when enabled)
- Parallel implementations via Rayon
- Memory-efficient algorithms for large datasets

### Rust-specific Features
- Type-safe APIs with compile-time guarantees
- Zero-copy array views where possible
- Error handling via Result types
- Generic implementations over floating-point types

## License

This project is dual-licensed under:

- [MIT License](../LICENSE-MIT)
- [Apache License Version 2.0](../LICENSE-APACHE)

You can choose to use either license. See the [LICENSE](../LICENSE) file for details.

## Contributing

Contributions are welcome! Please see the project's [CONTRIBUTING.md](../CONTRIBUTING.md) file for guidelines.