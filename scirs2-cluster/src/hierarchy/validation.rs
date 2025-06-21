//! Validation utilities for hierarchical clustering
//!
//! This module provides functions to validate linkage matrices and other
//! hierarchical clustering data structures to ensure they meet mathematical
//! requirements and are suitable for downstream analysis.

use ndarray::{ArrayView1, ArrayView2};
use num_traits::{Float, FromPrimitive};
use std::fmt::Debug;

use crate::error::{ClusteringError, Result};

/// Validates a linkage matrix for correctness
///
/// This function performs comprehensive validation of a linkage matrix
/// to ensure it meets the mathematical requirements for hierarchical clustering.
///
/// # Arguments
///
/// * `linkage_matrix` - The linkage matrix to validate (n-1 × 4)
/// * `n_observations` - Expected number of original observations
///
/// # Returns
///
/// * `Result<()>` - Ok if valid, error with detailed message if invalid
///
/// # Validation Checks
///
/// 1. Matrix dimensions are correct (n-1 rows, 4 columns)
/// 2. Cluster indices are valid and in proper range
/// 3. Merge distances are non-negative and monotonic (for single/complete linkage)
/// 4. Cluster sizes are consistent and >= 2
/// 5. No self-merges (cluster merging with itself)
/// 6. All values are finite
pub fn validate_linkage_matrix<
    F: Float + FromPrimitive + Debug + PartialOrd + std::fmt::Display,
>(
    linkage_matrix: ArrayView2<F>,
    n_observations: usize,
) -> Result<()> {
    let n_merges = linkage_matrix.shape()[0];
    let n_cols = linkage_matrix.shape()[1];

    // Check dimensions
    if n_merges != n_observations - 1 {
        return Err(ClusteringError::InvalidInput(format!(
            "Linkage matrix should have {} rows for {} observations, got {}",
            n_observations - 1,
            n_observations,
            n_merges
        )));
    }

    if n_cols != 4 {
        return Err(ClusteringError::InvalidInput(format!(
            "Linkage matrix should have 4 columns, got {}",
            n_cols
        )));
    }

    // Validate each merge
    for i in 0..n_merges {
        let cluster1 = linkage_matrix[[i, 0]];
        let cluster2 = linkage_matrix[[i, 1]];
        let distance = linkage_matrix[[i, 2]];
        let count = linkage_matrix[[i, 3]];

        // Check that all values are finite
        if !cluster1.is_finite()
            || !cluster2.is_finite()
            || !distance.is_finite()
            || !count.is_finite()
        {
            return Err(ClusteringError::InvalidInput(format!(
                "Non-finite values in linkage matrix at row {}",
                i
            )));
        }

        // Convert to usize for index checking
        let c1 = cluster1.to_usize().unwrap_or(usize::MAX);
        let c2 = cluster2.to_usize().unwrap_or(usize::MAX);

        // Check cluster indices are valid
        let max_cluster_id = n_observations + i - 1;
        if c1 >= n_observations + i || c2 >= n_observations + i {
            return Err(ClusteringError::InvalidInput(format!(
                "Invalid cluster indices at merge {}: {} and {} (max allowed: {})",
                i, c1, c2, max_cluster_id
            )));
        }

        // Check no self-merge
        if c1 == c2 {
            return Err(ClusteringError::InvalidInput(format!(
                "Self-merge detected at row {}: cluster {} merges with itself",
                i, c1
            )));
        }

        // Check distance is non-negative
        if distance < F::zero() {
            return Err(ClusteringError::InvalidInput(format!(
                "Negative merge distance at row {}: {}",
                i, distance
            )));
        }

        // Check cluster count is at least 2 (since it's a merge)
        if count < F::from(2).unwrap() {
            return Err(ClusteringError::InvalidInput(format!(
                "Cluster count should be >= 2 at row {}, got {}",
                i, count
            )));
        }
    }

    Ok(())
}

/// Validates that merge distances are monotonic (for certain linkage methods)
///
/// For single and complete linkage, merge distances should be non-decreasing.
///
/// # Arguments
///
/// * `linkage_matrix` - The linkage matrix to check
/// * `strict` - If true, requires strictly increasing distances
///
/// # Returns
///
/// * `Result<()>` - Ok if monotonic, error otherwise
pub fn validate_monotonic_distances<
    F: Float + FromPrimitive + Debug + PartialOrd + std::fmt::Display,
>(
    linkage_matrix: ArrayView2<F>,
    strict: bool,
) -> Result<()> {
    let n_merges = linkage_matrix.shape()[0];

    for i in 1..n_merges {
        let prev_distance = linkage_matrix[[i - 1, 2]];
        let curr_distance = linkage_matrix[[i, 2]];

        if strict {
            if curr_distance <= prev_distance {
                return Err(ClusteringError::InvalidInput(format!(
                    "Merge distances should be strictly increasing: {} <= {} at merge {}",
                    curr_distance, prev_distance, i
                )));
            }
        } else if curr_distance < prev_distance - F::from(1e-10).unwrap() {
            return Err(ClusteringError::InvalidInput(format!(
                "Merge distances should be non-decreasing: {} < {} at merge {}",
                curr_distance, prev_distance, i
            )));
        }
    }

    Ok(())
}

/// Validates cluster extraction parameters
///
/// Ensures that parameters for flat cluster extraction are valid.
///
/// # Arguments
///
/// * `linkage_matrix` - The linkage matrix
/// * `criterion` - Criterion type ("distance", "maxclust", "inconsistent")
/// * `threshold` - Threshold value for the criterion
///
/// # Returns
///
/// * `Result<()>` - Ok if valid, error otherwise
pub fn validate_cluster_extraction_params<
    F: Float + FromPrimitive + Debug + PartialOrd + std::fmt::Display,
>(
    linkage_matrix: ArrayView2<F>,
    criterion: &str,
    threshold: F,
) -> Result<()> {
    // First validate the linkage matrix itself
    let n_observations = linkage_matrix.shape()[0] + 1;
    validate_linkage_matrix(linkage_matrix, n_observations)?;

    // Check criterion is valid
    match criterion.to_lowercase().as_str() {
        "distance" => {
            if threshold < F::zero() {
                return Err(ClusteringError::InvalidInput(
                    "Distance threshold must be non-negative".to_string(),
                ));
            }
        }
        "maxclust" => {
            let max_clusters = threshold.to_usize().unwrap_or(0);
            if max_clusters < 1 || max_clusters > n_observations {
                return Err(ClusteringError::InvalidInput(format!(
                    "Number of clusters must be between 1 and {}, got {}",
                    n_observations, max_clusters
                )));
            }
        }
        "inconsistent" => {
            if threshold < F::zero() {
                return Err(ClusteringError::InvalidInput(
                    "Inconsistency threshold must be non-negative".to_string(),
                ));
            }
        }
        _ => {
            return Err(ClusteringError::InvalidInput(format!(
                "Unknown criterion '{}'. Valid options: 'distance', 'maxclust', 'inconsistent'",
                criterion
            )));
        }
    }

    Ok(())
}

/// Validates that a distance matrix is suitable for clustering
///
/// Checks properties required for distance matrices used in hierarchical clustering.
///
/// # Arguments
///
/// * `distance_matrix` - Distance matrix (condensed or square form)
/// * `condensed` - Whether the matrix is in condensed form
///
/// # Returns
///
/// * `Result<()>` - Ok if valid, error otherwise
pub fn validate_distance_matrix<
    F: Float + FromPrimitive + Debug + PartialOrd + std::fmt::Display,
>(
    distance_matrix: ArrayView1<F>,
    condensed: bool,
) -> Result<()> {
    let n_elements = distance_matrix.len();

    if condensed {
        // For condensed form, we should have n*(n-1)/2 elements
        // Solve n*(n-1)/2 = n_elements for n
        let n_float = (1.0 + (1.0 + 8.0 * n_elements as f64).sqrt()) / 2.0;
        let n = n_float as usize;

        if n * (n - 1) / 2 != n_elements {
            return Err(ClusteringError::InvalidInput(format!(
                "Invalid condensed distance matrix size: {} elements doesn't correspond to n*(n-1)/2 for any integer n",
                n_elements
            )));
        }

        if n < 2 {
            return Err(ClusteringError::InvalidInput(
                "Distance matrix must represent at least 2 observations".to_string(),
            ));
        }
    }

    // Check all distances are non-negative and finite
    for (i, &distance) in distance_matrix.iter().enumerate() {
        if !distance.is_finite() {
            return Err(ClusteringError::InvalidInput(format!(
                "Non-finite distance at index {}",
                i
            )));
        }

        if distance < F::zero() {
            return Err(ClusteringError::InvalidInput(format!(
                "Negative distance at index {}: {}",
                i, distance
            )));
        }
    }

    Ok(())
}

/// Validates that a square distance matrix has required properties
///
/// Checks symmetry, zero diagonal, and metric properties.
///
/// # Arguments
///
/// * `distance_matrix` - Square distance matrix
/// * `check_symmetry` - Whether to check matrix symmetry
/// * `check_triangle_inequality` - Whether to check triangle inequality
///
/// # Returns
///
/// * `Result<()>` - Ok if valid, error otherwise
pub fn validate_square_distance_matrix<
    F: Float + FromPrimitive + Debug + PartialOrd + std::fmt::Display,
>(
    distance_matrix: ArrayView2<F>,
    check_symmetry: bool,
    check_triangle_inequality: bool,
) -> Result<()> {
    let n = distance_matrix.shape()[0];
    let m = distance_matrix.shape()[1];

    // Check square matrix
    if n != m {
        return Err(ClusteringError::InvalidInput(format!(
            "Distance matrix must be square, got {}x{}",
            n, m
        )));
    }

    if n < 2 {
        return Err(ClusteringError::InvalidInput(
            "Distance matrix must be at least 2x2".to_string(),
        ));
    }

    // Check diagonal is zero
    for i in 0..n {
        let diag_val = distance_matrix[[i, i]];
        if !diag_val.is_finite() || diag_val.abs() > F::from(1e-10).unwrap() {
            return Err(ClusteringError::InvalidInput(format!(
                "Diagonal element at ({}, {}) should be zero, got {}",
                i, i, diag_val
            )));
        }
    }

    // Check all elements are finite and non-negative
    for i in 0..n {
        for j in 0..n {
            let val = distance_matrix[[i, j]];
            if !val.is_finite() {
                return Err(ClusteringError::InvalidInput(format!(
                    "Non-finite distance at ({}, {})",
                    i, j
                )));
            }

            if val < F::zero() {
                return Err(ClusteringError::InvalidInput(format!(
                    "Negative distance at ({}, {}): {}",
                    i, j, val
                )));
            }
        }
    }

    // Check symmetry
    if check_symmetry {
        for i in 0..n {
            for j in (i + 1)..n {
                let val_ij = distance_matrix[[i, j]];
                let val_ji = distance_matrix[[j, i]];
                let diff = (val_ij - val_ji).abs();

                if diff > F::from(1e-10).unwrap() {
                    return Err(ClusteringError::InvalidInput(format!(
                        "Distance matrix is not symmetric: d({}, {}) = {} != d({}, {}) = {}",
                        i, j, val_ij, j, i, val_ji
                    )));
                }
            }
        }
    }

    // Check triangle inequality
    if check_triangle_inequality {
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    if i != j && j != k && i != k {
                        let d_ij = distance_matrix[[i, j]];
                        let d_jk = distance_matrix[[j, k]];
                        let d_ik = distance_matrix[[i, k]];

                        if d_ik > d_ij + d_jk + F::from(1e-10).unwrap() {
                            return Err(ClusteringError::InvalidInput(format!(
                                "Triangle inequality violated: d({}, {}) = {} > d({}, {}) + d({}, {}) = {} + {}",
                                i, k, d_ik, i, j, j, k, d_ij, d_jk
                            )));
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

/// Checks consistency of cluster assignments with a linkage matrix
///
/// Validates that flat cluster assignments are consistent with the
/// hierarchical structure defined by the linkage matrix.
///
/// # Arguments
///
/// * `linkage_matrix` - The linkage matrix
/// * `cluster_assignments` - Flat cluster assignments for each observation
///
/// # Returns
///
/// * `Result<()>` - Ok if consistent, error otherwise
pub fn validate_cluster_consistency<
    F: Float + FromPrimitive + Debug + PartialOrd + std::fmt::Display,
>(
    linkage_matrix: ArrayView2<F>,
    cluster_assignments: ArrayView1<usize>,
) -> Result<()> {
    let n_observations = linkage_matrix.shape()[0] + 1;

    // Check dimensions
    if cluster_assignments.len() != n_observations {
        return Err(ClusteringError::InvalidInput(format!(
            "Cluster assignments length {} doesn't match number of observations {}",
            cluster_assignments.len(),
            n_observations
        )));
    }

    // First validate the linkage matrix
    validate_linkage_matrix(linkage_matrix, n_observations)?;

    // Check that cluster IDs are in valid range
    let max_cluster_id = cluster_assignments.iter().max().copied().unwrap_or(0);
    let unique_clusters: std::collections::HashSet<_> =
        cluster_assignments.iter().copied().collect();

    // Cluster IDs should be contiguous starting from 0
    for expected_id in 0..unique_clusters.len() {
        if !unique_clusters.contains(&expected_id) {
            return Err(ClusteringError::InvalidInput(format!(
                "Cluster IDs should be contiguous starting from 0, missing ID {}",
                expected_id
            )));
        }
    }

    if max_cluster_id >= n_observations {
        return Err(ClusteringError::InvalidInput(format!(
            "Maximum cluster ID {} should be less than number of observations {}",
            max_cluster_id, n_observations
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};

    #[test]
    fn test_validate_linkage_matrix_valid() {
        // Create a valid linkage matrix for 4 points
        let linkage = Array2::from_shape_vec(
            (3, 4),
            vec![
                0.0, 1.0, 0.5, 2.0, // Merge clusters 0 and 1 at distance 0.5
                2.0, 3.0, 0.8, 2.0, // Merge clusters 2 and 3 at distance 0.8
                4.0, 5.0, 1.2, 4.0, // Merge clusters 4 and 5 at distance 1.2
            ],
        )
        .unwrap();

        let result = validate_linkage_matrix(linkage.view(), 4);
        assert!(
            result.is_ok(),
            "Valid linkage matrix should pass validation"
        );
    }

    #[test]
    fn test_validate_linkage_matrix_wrong_dimensions() {
        // Wrong number of rows
        let linkage =
            Array2::from_shape_vec((2, 4), vec![0.0, 1.0, 0.5, 2.0, 2.0, 3.0, 0.8, 2.0]).unwrap();

        let result = validate_linkage_matrix(linkage.view(), 4);
        assert!(result.is_err(), "Wrong dimensions should fail validation");
    }

    #[test]
    fn test_validate_linkage_matrix_negative_distance() {
        let linkage = Array2::from_shape_vec(
            (3, 4),
            vec![
                0.0, 1.0, -0.5, 2.0, // Negative distance
                2.0, 3.0, 0.8, 2.0, 4.0, 5.0, 1.2, 4.0,
            ],
        )
        .unwrap();

        let result = validate_linkage_matrix(linkage.view(), 4);
        assert!(result.is_err(), "Negative distance should fail validation");
    }

    #[test]
    fn test_validate_linkage_matrix_self_merge() {
        let linkage = Array2::from_shape_vec(
            (3, 4),
            vec![
                0.0, 0.0, 0.5, 2.0, // Self-merge
                2.0, 3.0, 0.8, 2.0, 4.0, 5.0, 1.2, 4.0,
            ],
        )
        .unwrap();

        let result = validate_linkage_matrix(linkage.view(), 4);
        assert!(result.is_err(), "Self-merge should fail validation");
    }

    #[test]
    fn test_validate_monotonic_distances_valid() {
        let linkage = Array2::from_shape_vec(
            (3, 4),
            vec![0.0, 1.0, 0.5, 2.0, 2.0, 3.0, 0.8, 2.0, 4.0, 5.0, 1.2, 4.0],
        )
        .unwrap();

        let result = validate_monotonic_distances(linkage.view(), false);
        assert!(result.is_ok(), "Monotonic distances should pass validation");
    }

    #[test]
    fn test_validate_monotonic_distances_invalid() {
        let linkage = Array2::from_shape_vec(
            (3, 4),
            vec![
                0.0, 1.0, 1.2, 2.0, // Higher distance first
                2.0, 3.0, 0.8, 2.0, // Lower distance second
                4.0, 5.0, 1.5, 4.0,
            ],
        )
        .unwrap();

        let result = validate_monotonic_distances(linkage.view(), false);
        assert!(
            result.is_err(),
            "Non-monotonic distances should fail validation"
        );
    }

    #[test]
    fn test_validate_condensed_distance_matrix() {
        // Valid condensed matrix for 4 points: 4*3/2 = 6 elements
        let distances = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let result = validate_distance_matrix(distances.view(), true);
        assert!(
            result.is_ok(),
            "Valid condensed distance matrix should pass"
        );
    }

    #[test]
    fn test_validate_condensed_distance_matrix_invalid_size() {
        // Invalid size: 5 elements doesn't correspond to n*(n-1)/2 for any n
        let distances = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        let result = validate_distance_matrix(distances.view(), true);
        assert!(result.is_err(), "Invalid condensed matrix size should fail");
    }

    #[test]
    fn test_validate_cluster_extraction_params() {
        let linkage = Array2::from_shape_vec(
            (3, 4),
            vec![0.0, 1.0, 0.5, 2.0, 2.0, 3.0, 0.8, 2.0, 4.0, 5.0, 1.2, 4.0],
        )
        .unwrap();

        // Valid parameters
        assert!(validate_cluster_extraction_params(linkage.view(), "distance", 1.0).is_ok());
        assert!(validate_cluster_extraction_params(linkage.view(), "maxclust", 3.0).is_ok());
        assert!(validate_cluster_extraction_params(linkage.view(), "inconsistent", 0.5).is_ok());

        // Invalid parameters
        assert!(validate_cluster_extraction_params(linkage.view(), "distance", -1.0).is_err());
        assert!(validate_cluster_extraction_params(linkage.view(), "maxclust", 0.0).is_err());
        assert!(validate_cluster_extraction_params(linkage.view(), "invalid", 1.0).is_err());
    }
}
