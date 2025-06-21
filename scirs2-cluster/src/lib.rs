//! Clustering algorithms module for SciRS2
//!
//! This module provides implementations of various clustering algorithms such as:
//! - Vector quantization (k-means, etc.)
//! - Hierarchical clustering
//! - Density-based clustering (DBSCAN, OPTICS, etc.)
//! - Mean Shift clustering
//! - Spectral clustering
//! - Affinity Propagation
//!
//! ## Features
//!
//! * **Vector Quantization**: K-means and K-means++ for partitioning data
//! * **Hierarchical Clustering**: Agglomerative clustering with various linkage methods
//! * **Density-based Clustering**: DBSCAN and OPTICS for finding clusters of arbitrary shape
//! * **Mean Shift**: Non-parametric clustering based on density estimation
//! * **Spectral Clustering**: Graph-based clustering using eigenvectors of the graph Laplacian
//! * **Affinity Propagation**: Message-passing based clustering that identifies exemplars
//! * **Evaluation Metrics**: Silhouette coefficient, Davies-Bouldin index, and other measures to evaluate clustering quality
//! * **Data Preprocessing**: Utilities for normalizing, standardizing, and whitening data before clustering
//!
//! ## Examples
//!
//! ```
//! use ndarray::{Array2, ArrayView2};
//! use scirs2_cluster::vq::kmeans;
//! use scirs2_cluster::preprocess::standardize;
//!
//! // Example data with two clusters
//! let data = Array2::from_shape_vec((6, 2), vec![
//!     1.0, 2.0,
//!     1.2, 1.8,
//!     0.8, 1.9,
//!     3.7, 4.2,
//!     3.9, 3.9,
//!     4.2, 4.1,
//! ]).unwrap();
//!
//! // Standardize the data
//! let standardized = standardize(data.view(), true).unwrap();
//!
//! // Run k-means with k=2
//! let (centroids, labels) = kmeans(standardized.view(), 2, None, None, None, None).unwrap();
//!
//! // Print the results
//! println!("Centroids: {:?}", centroids);
//! println!("Cluster assignments: {:?}", labels);
//! ```

#![warn(missing_docs)]

pub mod affinity;
pub mod birch;
pub mod density;
pub mod error;
pub mod gmm;
pub mod hierarchy;
pub mod input_validation;
/// Mean Shift clustering implementation.
///
/// This module provides the Mean Shift clustering algorithm, which is a centroid-based
/// algorithm that works by updating candidates for centroids to be the mean of the points
/// within a given region. These candidates are then filtered in a post-processing stage to
/// eliminate near-duplicates, forming the final set of centroids.
///
/// Mean Shift is a non-parametric clustering technique that doesn't require specifying the
/// number of clusters in advance and can find clusters of arbitrary shapes.
pub mod meanshift;
pub mod metrics;
pub mod neighbor_search;
pub mod preprocess;
pub mod sparse;
pub mod spectral;
pub mod stability;
pub mod streaming;
pub mod vq;

// Re-exports
pub use affinity::{affinity_propagation, AffinityPropagationOptions};
pub use birch::{birch, Birch, BirchOptions};
pub use density::hdbscan::{
    dbscan_clustering, hdbscan, ClusterSelectionMethod, HDBSCANOptions, HDBSCANResult, StoreCenter,
};
pub use density::optics::{extract_dbscan_clustering, extract_xi_clusters, OPTICSResult};
pub use density::*;
pub use gmm::{gaussian_mixture, CovarianceType, GMMInit, GMMOptions, GaussianMixture};
pub use hierarchy::*;
pub use input_validation::{
    check_duplicate_points, suggest_clustering_algorithm, validate_clustering_data,
    validate_convergence_parameters, validate_distance_parameter, validate_integer_parameter,
    validate_n_clusters, validate_sample_weights, ValidationConfig,
};
pub use meanshift::{estimate_bandwidth, get_bin_seeds, mean_shift, MeanShift, MeanShiftOptions};
pub use metrics::{
    adjusted_rand_index, calinski_harabasz_score, davies_bouldin_score,
    homogeneity_completeness_v_measure, normalized_mutual_info, silhouette_samples,
    silhouette_score,
};
pub use neighbor_search::{
    create_neighbor_searcher, BallTree, BruteForceSearch, KDTree, NeighborResult,
    NeighborSearchAlgorithm, NeighborSearchConfig, NeighborSearcher,
};
pub use preprocess::{min_max_scale, normalize, standardize, whiten, NormType};
pub use sparse::{
    sparse_epsilon_graph, sparse_knn_graph, SparseDistanceMatrix, SparseHierarchicalClustering,
};
pub use spectral::{
    spectral_bipartition, spectral_clustering, AffinityMode, SpectralClusteringOptions,
};
pub use stability::{
    BootstrapValidator, ConsensusClusterer, OptimalKSelector, StabilityConfig, StabilityResult,
};
pub use streaming::{
    ChunkedDistanceMatrix, ProgressiveHierarchical, StreamingConfig, StreamingKMeans,
};
pub use vq::*;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod property_tests;
