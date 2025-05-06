# SciRS2 Clustering Module

[![crates.io](https://img.shields.io/crates/v/scirs2-cluster.svg)](https://crates.io/crates/scirs2-cluster)
[[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)]](../LICENSE)
[![Documentation](https://img.shields.io/docsrs/scirs2-cluster)](https://docs.rs/scirs2-cluster)

A comprehensive clustering module for the SciRS2 scientific computing library in Rust. This crate provides implementations of various clustering algorithms with a focus on performance, flexibility, and idiomatic Rust code.

## Features

* **Vector Quantization**
  * K-means clustering with customizable initialization
  * K-means++ smart initialization method
  * Vector quantization utilities

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
  * Support for custom distance metrics

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
scirs2-cluster = "0.1.0-alpha.2"
ndarray = "0.15"
```

To enable optimizations through the core module, add feature flags:

```toml
[dependencies]
scirs2-cluster = { version = "0.1.0-alpha.2", features = ["parallel"] }
```

## Usage

### K-means Example

```rust
use ndarray::{Array2, ArrayView2};
use scirs2_cluster::vq::kmeans;

// Create a dataset
let data = Array2::from_shape_vec((6, 2), vec![
    1.0, 2.0,
    1.2, 1.8,
    0.8, 1.9,
    3.7, 4.2,
    3.9, 3.9,
    4.2, 4.1,
]).unwrap();

// Run k-means with k=2
let (centroids, labels) = kmeans(data.view(), 2, None, None, None).unwrap();

// Print the results
println!("Centroids: {:?}", centroids);
println!("Cluster assignments: {:?}", labels);
```

### Hierarchical Clustering Example

```rust
use ndarray::{Array2, ArrayView2};
use scirs2_cluster::hierarchy::{linkage, fcluster, LinkageMethod, Metric};

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
let linkage_matrix = linkage(data.view(), LinkageMethod::Ward, Metric::Euclidean).unwrap();

// Form flat clusters by cutting the dendrogram
let num_clusters = 2;
let labels = fcluster(&linkage_matrix, num_clusters, None).unwrap();

// Print the results
println!("Cluster assignments: {:?}", labels);
```

### DBSCAN Example

```rust
use ndarray::{Array2, ArrayView2};
use scirs2_cluster::density::{dbscan, labels};
use scirs2_spatial::distance::DistanceMetric;

// Create a dataset
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
let cluster_labels = dbscan(data.view(), 0.8, 2, Some(DistanceMetric::Euclidean)).unwrap();

// Count noise points
let noise_count = cluster_labels.iter().filter(|&&label| label == labels::NOISE).count();

// Print the results
println!("Cluster assignments: {:?}", cluster_labels);
println!("Number of noise points: {}", noise_count);
```

## License

This project is dual-licensed under:

- [MIT License](../LICENSE-MIT)
- [Apache License Version 2.0](../LICENSE-APACHE)

You can choose to use either license. See the [LICENSE](../LICENSE) file for details.
## Contributing

Contributions are welcome! Please see the project's [CONTRIBUTING.md](../CONTRIBUTING.md) file for guidelines.
