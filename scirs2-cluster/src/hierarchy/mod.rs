//! Hierarchical clustering functions
//!
//! This module provides hierarchical clustering algorithms for agglomerative clustering,
//! linkage methods, and dendrogram visualization.
//!
//! ## Examples
//!
//! ```
//! use ndarray::{Array2, ArrayView2};
//! use scirs2_cluster::hierarchy::{linkage, fcluster, LinkageMethod, Metric};
//!
//! // Example data
//! let data = Array2::from_shape_vec((6, 2), vec![
//!     1.0, 2.0,
//!     1.2, 1.8,
//!     0.8, 1.9,
//!     3.7, 4.2,
//!     3.9, 3.9,
//!     4.2, 4.1,
//! ]).unwrap();
//!
//! // Calculate linkage matrix using Ward's method
//! let linkage_matrix = linkage(data.view(), LinkageMethod::Ward, Metric::Euclidean).unwrap();
//!
//! // Form flat clusters by cutting the dendrogram at a height that gives 2 clusters
//! let num_clusters = 2;
//! let labels = fcluster(&linkage_matrix, num_clusters, None).unwrap();
//!
//! // Print the results
//! println!("Cluster assignments: {:?}", labels);
//! ```

use ndarray::{Array1, Array2, ArrayView2};
use num_traits::{Float, FromPrimitive};
use std::fmt::Debug;

use crate::error::{ClusteringError, Result};

// Module definitions
pub mod agglomerative;
pub mod cluster_extraction;
pub mod condensed_matrix;
pub mod dendrogram;
pub mod disjoint_set;
pub mod leaf_ordering;
pub mod linkage;
pub mod optimized_ward;
pub mod parallel_linkage;
pub mod validation;
pub mod visualization;

// Re-exports
pub use self::agglomerative::{cut_tree_by_distance, cut_tree_by_inconsistency};
pub use self::cluster_extraction::{
    estimate_optimal_clusters, extract_clusters_multi_criteria, prune_clusters,
};
pub use self::condensed_matrix::{
    condensed_size, condensed_to_square, get_distance, points_from_condensed_size,
    square_to_condensed, validate_condensed_matrix,
};
pub use self::dendrogram::{cophenet, dendrogram, inconsistent, optimal_leaf_ordering};
pub use self::disjoint_set::DisjointSet;
pub use self::leaf_ordering::{
    apply_leaf_ordering, optimal_leaf_ordering_exact, optimal_leaf_ordering_heuristic,
};
pub use self::optimized_ward::{
    lance_williams_ward_update, memory_efficient_ward_linkage, optimized_ward_linkage,
};
pub use self::validation::{
    validate_cluster_consistency, validate_cluster_extraction_params, validate_distance_matrix,
    validate_linkage_matrix, validate_monotonic_distances, validate_square_distance_matrix,
};
pub use self::visualization::{
    create_dendrogram_plot, get_color_palette, Branch, ColorScheme, ColorThreshold,
    DendrogramConfig, DendrogramOrientation, DendrogramPlot, Leaf, LegendEntry, TruncateMode,
};

/// Linkage methods for hierarchical clustering
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkageMethod {
    /// Single linkage (minimum distance between any two points in clusters)
    Single,

    /// Complete linkage (maximum distance between any two points in clusters)
    Complete,

    /// Average linkage (average distance between all points in clusters)
    Average,

    /// Ward's method (minimizes variance of merged clusters)
    Ward,

    /// Centroid method (distance between cluster centroids)
    Centroid,

    /// Median method (uses weighted centroids)
    Median,

    /// Weighted average (weights by cluster size)
    Weighted,
}

/// Distance metrics for hierarchical clustering
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Metric {
    /// Euclidean distance (L2 norm)
    Euclidean,

    /// Manhattan distance (L1 norm)
    Manhattan,

    /// Maximum distance (L∞ norm)
    Chebyshev,

    /// Correlation distance (1 - correlation)
    Correlation,
}

/// Criterion for forming flat clusters from a hierarchical clustering
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClusterCriterion {
    /// Number of clusters desired
    MaxClust,

    /// Distance threshold for clusters
    Distance,

    /// Inconsistency threshold
    Inconsistent,
}

/// Computes distances between observations
fn compute_distances<F: Float + FromPrimitive>(data: ArrayView2<F>, metric: Metric) -> Array1<F> {
    let n_samples = data.shape()[0];
    let n_features = data.shape()[1];

    // For n samples, we need n*(n-1)/2 distances (condensed distance matrix)
    let num_distances = n_samples * (n_samples - 1) / 2;
    let mut distances = Array1::zeros(num_distances);

    let mut idx = 0;
    for i in 0..n_samples {
        for j in (i + 1)..n_samples {
            let dist = match metric {
                Metric::Euclidean => {
                    // Euclidean distance
                    let mut sum = F::zero();
                    for k in 0..n_features {
                        let diff = data[[i, k]] - data[[j, k]];
                        sum = sum + diff * diff;
                    }
                    sum.sqrt()
                }
                Metric::Manhattan => {
                    // Manhattan distance
                    let mut sum = F::zero();
                    for k in 0..n_features {
                        let diff = (data[[i, k]] - data[[j, k]]).abs();
                        sum = sum + diff;
                    }
                    sum
                }
                Metric::Chebyshev => {
                    // Chebyshev distance
                    let mut max_diff = F::zero();
                    for k in 0..n_features {
                        let diff = (data[[i, k]] - data[[j, k]]).abs();
                        if diff > max_diff {
                            max_diff = diff;
                        }
                    }
                    max_diff
                }
                Metric::Correlation => {
                    // Correlation distance
                    // Formula: 1 - correlation coefficient

                    // Compute means for both vectors
                    let mut mean_i = F::zero();
                    let mut mean_j = F::zero();

                    for k in 0..n_features {
                        mean_i = mean_i + data[[i, k]];
                        mean_j = mean_j + data[[j, k]];
                    }

                    mean_i = mean_i / F::from_usize(n_features).unwrap();
                    mean_j = mean_j / F::from_usize(n_features).unwrap();

                    // Compute correlation coefficient
                    let mut numerator = F::zero();
                    let mut denom_i = F::zero();
                    let mut denom_j = F::zero();

                    for k in 0..n_features {
                        let diff_i = data[[i, k]] - mean_i;
                        let diff_j = data[[j, k]] - mean_j;

                        numerator = numerator + diff_i * diff_j;
                        denom_i = denom_i + diff_i * diff_i;
                        denom_j = denom_j + diff_j * diff_j;
                    }

                    let denom = (denom_i * denom_j).sqrt();

                    if denom < F::from_f64(1e-10).unwrap() {
                        // If vectors are constant, distance is 0
                        F::zero()
                    } else {
                        F::one() - (numerator / denom)
                    }
                }
            };

            distances[idx] = dist;
            idx += 1;
        }
    }

    distances
}

/// Converts a condensed distance matrix index to (i, j) coordinates
pub fn condensed_index_to_coords(n: usize, idx: usize) -> (usize, usize) {
    // Find i and j from the condensed index
    let mut i = 0;
    let mut j = 0;
    let mut k = 0;

    for i_temp in 0..n {
        for j_temp in (i_temp + 1)..n {
            if k == idx {
                i = i_temp;
                j = j_temp;
                break;
            }
            k += 1;
        }

        if k == idx {
            break;
        }
    }

    (i, j)
}

/// Converts (i, j) coordinates to a condensed distance matrix index
pub fn coords_to_condensed_index(n: usize, i: usize, j: usize) -> usize {
    if i == j {
        panic!("Cannot compute diagonal index in condensed matrix");
    }

    let (i_min, j_min) = if i < j { (i, j) } else { (j, i) };
    (n * i_min) - ((i_min * (i_min + 1)) / 2) + (j_min - i_min - 1)
}

/// Performs hierarchical clustering using the specified linkage method
///
/// # Arguments
///
/// * `data` - The input data as a 2D array (n_samples x n_features)
/// * `method` - The linkage method to use
/// * `metric` - The distance metric to use
///
/// # Returns
///
/// * `Result<Array2<F>>` - The linkage matrix, which describes the dendrogram
pub fn linkage<
    F: Float + FromPrimitive + Debug + PartialOrd + Send + Sync + ndarray::ScalarOperand + 'static,
>(
    data: ArrayView2<F>,
    method: LinkageMethod,
    metric: Metric,
) -> Result<Array2<F>> {
    let n_samples = data.shape()[0];

    if n_samples < 2 {
        return Err(ClusteringError::InvalidInput(
            "Need at least 2 samples for hierarchical clustering".into(),
        ));
    }

    if n_samples > 10000 {
        // Hierarchical clustering on large datasets can be very memory-intensive
        // and slow. We'll add a warning here.
        eprintln!("Warning: Performing hierarchical clustering on {} samples. This may be slow and memory-intensive.", n_samples);
    }

    // Use optimized Ward's method if requested
    if method == LinkageMethod::Ward {
        return optimized_ward::optimized_ward_linkage(data, metric);
    }

    // Calculate distances between observations
    let distances = compute_distances(data, metric);

    // Run the clustering
    linkage::hierarchical_clustering(&distances, n_samples, method)
}

/// Performs parallel hierarchical clustering using the specified linkage method
///
/// This function uses parallelization to speed up the clustering process,
/// particularly beneficial for large datasets and computationally intensive linkage methods.
///
/// # Arguments
///
/// * `data` - The input data as a 2D array (n_samples x n_features)
/// * `method` - The linkage method to use
/// * `metric` - The distance metric to use
///
/// # Returns
///
/// * `Result<Array2<F>>` - The linkage matrix, which describes the dendrogram
///
/// # Examples
///
/// ```
/// use ndarray::{Array2, ArrayView2};
/// use scirs2_cluster::hierarchy::{parallel_linkage, LinkageMethod, Metric};
///
/// // Example data
/// let data = Array2::from_shape_vec((6, 2), vec![
///     1.0, 2.0,
///     1.2, 1.8,
///     0.8, 1.9,
///     3.7, 4.2,
///     3.9, 3.9,
///     4.2, 4.1,
/// ]).unwrap();
///
/// // Calculate linkage matrix using parallel Ward's method
/// let linkage_matrix = parallel_linkage(data.view(), LinkageMethod::Ward, Metric::Euclidean).unwrap();
///
/// println!("Linkage matrix shape: {:?}", linkage_matrix.shape());
/// ```
pub fn parallel_linkage<
    F: Float
        + FromPrimitive
        + Debug
        + PartialOrd
        + Send
        + Sync
        + std::iter::Sum
        + ndarray::ScalarOperand
        + 'static,
>(
    data: ArrayView2<F>,
    method: LinkageMethod,
    metric: Metric,
) -> Result<Array2<F>> {
    let n_samples = data.shape()[0];

    if n_samples < 2 {
        return Err(ClusteringError::InvalidInput(
            "Need at least 2 samples for hierarchical clustering".into(),
        ));
    }

    if n_samples > 10000 {
        // Hierarchical clustering on large datasets can be very memory-intensive
        // and slow. We'll add a warning here.
        eprintln!("Warning: Performing parallel hierarchical clustering on {} samples. This may still be slow for very large datasets.", n_samples);
    }

    // Use optimized Ward's method if requested (already parallel-optimized)
    if method == LinkageMethod::Ward {
        return optimized_ward::optimized_ward_linkage(data, metric);
    }

    // Calculate distances between observations
    let distances = compute_distances(data, metric);

    // Run the parallel clustering
    parallel_linkage::parallel_hierarchical_clustering(&distances, n_samples, method)
}

/// Forms flat clusters from a hierarchical clustering result
///
/// # Arguments
///
/// * `z` - The linkage matrix from the `linkage` function
/// * `t` - The threshold or number of clusters (depends on criterion)
/// * `criterion` - The criterion to use for forming clusters (default: MaxClust)
///
/// # Returns
///
/// * `Result<Array1<usize>>` - Cluster assignments (0-indexed)
pub fn fcluster<F: Float + FromPrimitive + PartialOrd>(
    z: &Array2<F>,
    t: usize,
    criterion: Option<ClusterCriterion>,
) -> Result<Array1<usize>> {
    let n_samples = z.shape()[0] + 1;
    let crit = criterion.unwrap_or(ClusterCriterion::MaxClust);

    match crit {
        ClusterCriterion::MaxClust => {
            // t represents the number of clusters
            if t == 0 || t > n_samples {
                return Err(ClusteringError::InvalidInput(format!(
                    "Number of clusters must be between 1 and {}",
                    n_samples
                )));
            }

            agglomerative::cut_tree(z, t)
        }
        ClusterCriterion::Distance => {
            // t represents a distance threshold
            let t_float = F::from_usize(t).unwrap();
            agglomerative::cut_tree_by_distance(z, t_float)
        }
        ClusterCriterion::Inconsistent => {
            // Not implemented yet
            Err(ClusteringError::InvalidInput(
                "Inconsistent criterion not yet implemented".into(),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_linkage_simple() {
        // Simple dataset with two clear clusters
        let data = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 1.2, 1.8, 0.8, 1.9, 3.7, 4.2, 3.9, 3.9, 4.2, 4.1],
        )
        .unwrap();

        // Run with Ward's method
        let linkage_matrix = linkage(data.view(), LinkageMethod::Ward, Metric::Euclidean).unwrap();

        // Check dimensions
        assert_eq!(linkage_matrix.shape(), &[5, 4]);

        // Check the first row
        // (We can't check exact values due to implementation differences, but we can check ranges)
        assert!(linkage_matrix[[0, 2]] > 0.0); // Distance should be positive
        assert_eq!(linkage_matrix[[0, 3]] as usize, 2); // Size should be 2
    }

    #[test]
    fn test_fcluster() {
        // Create a simple linkage matrix
        let data = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 1.2, 1.8, 0.8, 1.9, 3.7, 4.2, 3.9, 3.9, 4.2, 4.1],
        )
        .unwrap();

        let linkage_matrix = linkage(data.view(), LinkageMethod::Ward, Metric::Euclidean).unwrap();

        // Get 2 clusters
        let labels = fcluster(&linkage_matrix, 2, None).unwrap();

        // Should have 6 labels
        assert_eq!(labels.len(), 6);

        // Verify clusters make sense - first 3 and last 3 should be different clusters
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[1], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_eq!(labels[4], labels[5]);
        assert_ne!(labels[0], labels[3]);
    }

    #[test]
    fn test_distance_metrics() {
        // Test data
        let data =
            Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();

        // Calculate distances with different metrics
        let euclidean_distances = compute_distances(data.view(), Metric::Euclidean);
        let manhattan_distances = compute_distances(data.view(), Metric::Manhattan);
        let chebyshev_distances = compute_distances(data.view(), Metric::Chebyshev);

        // Check dimensions
        assert_eq!(euclidean_distances.len(), 6); // (4 choose 2)

        // Check specific distances
        // Distance between (0,0) and (1,0) should be 1.0 for Euclidean
        assert_abs_diff_eq!(euclidean_distances[0], 1.0, epsilon = 1e-10);

        // Distance between (0,0) and (1,1) should be sqrt(2) for Euclidean
        assert_abs_diff_eq!(
            euclidean_distances[2],
            std::f64::consts::SQRT_2,
            epsilon = 1e-10
        );

        // Distance between (0,0) and (1,0) should be 1.0 for Manhattan
        assert_abs_diff_eq!(manhattan_distances[0], 1.0, epsilon = 1e-10);

        // Distance between (0,0) and (1,1) should be 2.0 for Manhattan
        assert_abs_diff_eq!(manhattan_distances[2], 2.0, epsilon = 1e-10);

        // Distance between (0,0) and (1,1) should be 1.0 for Chebyshev
        assert_abs_diff_eq!(chebyshev_distances[2], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_hierarchy_with_different_linkage_methods() {
        // Simple dataset
        let data = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 1.2, 1.8, 0.8, 1.9, 3.7, 4.2, 3.9, 3.9, 4.2, 4.1],
        )
        .unwrap();

        // Test different linkage methods
        let methods = vec![
            LinkageMethod::Single,
            LinkageMethod::Complete,
            LinkageMethod::Average,
            LinkageMethod::Ward,
        ];

        for method in methods {
            let linkage_matrix = linkage(data.view(), method, Metric::Euclidean).unwrap();

            // Check dimensions
            assert_eq!(linkage_matrix.shape(), &[5, 4]);

            // Get 2 clusters
            let labels = fcluster(&linkage_matrix, 2, None).unwrap();

            // Should have 6 labels
            assert_eq!(labels.len(), 6);
        }
    }
}
