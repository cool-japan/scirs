//! Density-based metrics for clustering evaluation
//!
//! This module contains metrics for evaluating clustering results based on
//! density characteristics of the clusters. These metrics help assess how
//! well clusters represent dense regions in the data space.

use ndarray::{ArrayBase, Data, Dimension, Ix1, Ix2};
use num_traits::{Float, NumCast};
use std::collections::HashMap;
use std::ops::{AddAssign, DivAssign};

use crate::error::{MetricsError, Result};

/// Calculate the Local Density Factor (LDF) for all clusters
///
/// The Local Density Factor measures how dense clusters are compared to their
/// surrounding space. Higher values indicate better clustering of dense regions.
///
/// # Arguments
///
/// * `x` - Array of shape (n_samples, n_features) - The data
/// * `labels` - Array of shape (n_samples,) - Predicted labels for each sample
/// * `k` - Number of neighbors to consider for density calculation (default: 5)
///
/// # Returns
///
/// * HashMap mapping each cluster label to its local density factor
///
/// # Examples
///
/// ```
/// use ndarray::{array, Array2};
/// use scirs2_metrics::clustering::density::local_density_factor;
///
/// // Create a simple dataset with 2 clusters
/// let x = Array2::from_shape_vec((6, 2), vec![
///     1.0, 2.0, 1.5, 1.8, 1.2, 2.2,
///     5.0, 6.0, 5.2, 5.8, 5.5, 6.2,
/// ]).unwrap();
///
/// let labels = array![0, 0, 0, 1, 1, 1];
///
/// let factors = local_density_factor(&x, &labels, None).unwrap();
/// ```
pub fn local_density_factor<F, S1, S2, D>(
    x: &ArrayBase<S1, Ix2>,
    labels: &ArrayBase<S2, D>,
    k: Option<usize>,
) -> Result<HashMap<usize, F>>
where
    F: Float + NumCast + std::fmt::Debug + ndarray::ScalarOperand + AddAssign + DivAssign,
    S1: Data<Elem = F>,
    S2: Data<Elem = usize>,
    D: Dimension,
{
    // Check that x and labels have the same number of samples
    let n_samples = x.shape()[0];
    if n_samples != labels.len() {
        return Err(MetricsError::InvalidInput(format!(
            "x has {} samples, but labels has {} samples",
            n_samples,
            labels.len()
        )));
    }

    // Default k to a safe value based on n_samples
    let k = k.unwrap_or_else(|| {
        if n_samples <= 1 {
            1
        } else if n_samples < 10 {
            std::cmp::min(2, n_samples - 1)
        } else {
            std::cmp::min(5, n_samples / 10)
        }
    });

    // Get unique cluster labels
    let mut unique_labels = Vec::new();
    for &label in labels.iter() {
        if !unique_labels.contains(&label) {
            unique_labels.push(label);
        }
    }

    // Sort labels for consistent results
    unique_labels.sort();

    // For each sample, compute the k-nearest neighbors distance (density estimate)
    let mut all_knn_distances = Vec::new();
    let mut cluster_idx = HashMap::new();

    for label in &unique_labels {
        cluster_idx.insert(*label, Vec::new());
    }

    // Collect indices by cluster
    for (i, &label) in labels.iter().enumerate() {
        if let Some(indices) = cluster_idx.get_mut(&label) {
            indices.push(i);
        }
    }

    // Calculate k-distance for each point
    for i in 0..n_samples {
        let current_point = x.row(i);

        // Calculate distances to all other points
        let mut distances = Vec::new();
        for j in 0..n_samples {
            if i != j {
                let other_point = x.row(j);
                let dist = calculate_euclidean_distance(&current_point, &other_point);
                distances.push(dist);
            }
        }

        // Sort distances and take the k-th nearest
        distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let k_distance = if distances.len() >= k {
            distances[k - 1] // k-th nearest neighbor distance
        } else if !distances.is_empty() {
            distances[distances.len() - 1] // Use farthest if k is too large
        } else {
            F::zero() // Fallback if no distances
        };

        all_knn_distances.push((i, k_distance));
    }

    // Calculate average k-distance per cluster
    let mut ldf = HashMap::new();

    for &label in &unique_labels {
        let cluster_indices = cluster_idx.get(&label).unwrap();
        if cluster_indices.is_empty() {
            continue;
        }

        // Average k-distance within the cluster
        let mut cluster_knn_sum = F::zero();
        let mut count = 0;

        for &idx in cluster_indices {
            cluster_knn_sum += all_knn_distances[idx].1;
            count += 1;
        }

        let avg_knn = if count > 0 {
            cluster_knn_sum / F::from(count).unwrap()
        } else {
            F::zero()
        };

        // Calculate LDF as the inverse of the average k-distance
        // (smaller distances = higher density)
        let factor = if avg_knn > F::zero() {
            F::one() / avg_knn
        } else {
            F::zero()
        };

        ldf.insert(label, factor);
    }

    Ok(ldf)
}

/// Calculate the Relative Density Index (RDI) for clustering evaluation
///
/// The Relative Density Index measures the ratio of intra-cluster density
/// to inter-cluster density. Higher values indicate better separation
/// between clusters in terms of density.
///
/// # Arguments
///
/// * `x` - Array of shape (n_samples, n_features) - The data
/// * `labels` - Array of shape (n_samples,) - Predicted labels for each sample
/// * `k` - Number of neighbors to consider for density calculation (default: 5)
///
/// # Returns
///
/// * Relative Density Index value
///
/// # Examples
///
/// ```
/// use ndarray::{array, Array2};
/// use scirs2_metrics::clustering::density::relative_density_index;
///
/// // Create a simple dataset with 2 clusters
/// let x = Array2::from_shape_vec((6, 2), vec![
///     1.0, 2.0, 1.5, 1.8, 1.2, 2.2,
///     5.0, 6.0, 5.2, 5.8, 5.5, 6.2,
/// ]).unwrap();
///
/// let labels = array![0, 0, 0, 1, 1, 1];
///
/// let rdi = relative_density_index(&x, &labels, None).unwrap();
/// ```
pub fn relative_density_index<F, S1, S2, D>(
    x: &ArrayBase<S1, Ix2>,
    labels: &ArrayBase<S2, D>,
    k: Option<usize>,
) -> Result<F>
where
    F: Float + NumCast + std::fmt::Debug + ndarray::ScalarOperand + AddAssign + DivAssign,
    S1: Data<Elem = F>,
    S2: Data<Elem = usize>,
    D: Dimension,
{
    // Check that x and labels have the same number of samples
    let n_samples = x.shape()[0];
    if n_samples != labels.len() {
        return Err(MetricsError::InvalidInput(format!(
            "x has {} samples, but labels has {} samples",
            n_samples,
            labels.len()
        )));
    }

    // Get unique cluster labels
    let mut unique_labels = Vec::new();
    for &label in labels.iter() {
        if !unique_labels.contains(&label) {
            unique_labels.push(label);
        }
    }

    // Sort labels for consistent results
    unique_labels.sort();

    // Default k to a safe value based on n_samples
    let k = k.unwrap_or_else(|| {
        if n_samples <= 1 {
            1
        } else if n_samples < 10 {
            std::cmp::min(2, n_samples - 1)
        } else {
            std::cmp::min(5, n_samples / 10)
        }
    });

    // Calculate intra-cluster and inter-cluster density estimates
    let mut intra_density_sum = F::zero();
    let mut inter_density_sum = F::zero();

    for (i, &label_i) in labels.iter().enumerate() {
        // Calculate distances to all other points
        let mut intra_distances = Vec::new();
        let mut inter_distances = Vec::new();

        for (j, &label_j) in labels.iter().enumerate() {
            if i != j {
                let dist = calculate_euclidean_distance(&x.row(i), &x.row(j));

                if label_i == label_j {
                    intra_distances.push(dist);
                } else {
                    inter_distances.push(dist);
                }
            }
        }

        // Calculate intra-cluster density (k-nearest neighbors within cluster)
        intra_distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let intra_k = std::cmp::min(k, intra_distances.len());

        if intra_k > 0 {
            let knn_intra_dist = intra_distances[intra_k - 1];
            let intra_density = if knn_intra_dist > F::zero() {
                F::one() / knn_intra_dist
            } else {
                F::zero()
            };
            intra_density_sum += intra_density;
        }

        // Calculate inter-cluster density (k-nearest neighbors from other clusters)
        inter_distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let inter_k = std::cmp::min(k, inter_distances.len());

        if inter_k > 0 {
            let knn_inter_dist = inter_distances[inter_k - 1];
            let inter_density = if knn_inter_dist > F::zero() {
                F::one() / knn_inter_dist
            } else {
                F::zero()
            };
            inter_density_sum += inter_density;
        }
    }

    // Calculate average densities
    let avg_intra_density = if n_samples > 0 {
        intra_density_sum / F::from(n_samples).unwrap()
    } else {
        F::zero()
    };

    let avg_inter_density = if n_samples > 0 {
        inter_density_sum / F::from(n_samples).unwrap()
    } else {
        F::zero()
    };

    // Calculate RDI (avoiding division by zero)
    let rdi = if avg_inter_density > F::zero() {
        avg_intra_density / avg_inter_density
    } else if avg_intra_density > F::zero() {
        F::max_value() // If inter-density is 0 but intra-density is not
    } else {
        F::one() // Default case if both are zero
    };

    Ok(rdi)
}

/// Calculate Density-Based Cluster Validity (DBCV) index
///
/// The DBCV index measures the validity of a clustering based on the relative density
/// of clusters. It accounts for variations in cluster densities and shapes.
/// Values closer to 1 indicate better clustering.
///
/// # Arguments
///
/// * `x` - Array of shape (n_samples, n_features) - The data
/// * `labels` - Array of shape (n_samples,) - Predicted labels for each sample
/// * `k` - Number of neighbors to consider for density calculation (default: 5)
///
/// # Returns
///
/// * DBCV index value (between -1 and 1)
///
/// # Examples
///
/// ```
/// use ndarray::{array, Array2};
/// use scirs2_metrics::clustering::density::density_based_cluster_validity;
///
/// // Create a simple dataset with 2 clusters
/// let x = Array2::from_shape_vec((6, 2), vec![
///     1.0, 2.0, 1.5, 1.8, 1.2, 2.2,
///     5.0, 6.0, 5.2, 5.8, 5.5, 6.2,
/// ]).unwrap();
///
/// let labels = array![0, 0, 0, 1, 1, 1];
///
/// let dbcv = density_based_cluster_validity(&x, &labels, None).unwrap();
/// ```
pub fn density_based_cluster_validity<F, S1, S2, D>(
    x: &ArrayBase<S1, Ix2>,
    labels: &ArrayBase<S2, D>,
    k: Option<usize>,
) -> Result<F>
where
    F: Float + NumCast + std::fmt::Debug + ndarray::ScalarOperand + AddAssign + DivAssign,
    S1: Data<Elem = F>,
    S2: Data<Elem = usize>,
    D: Dimension,
{
    // Check that x and labels have the same number of samples
    let n_samples = x.shape()[0];
    if n_samples != labels.len() {
        return Err(MetricsError::InvalidInput(format!(
            "x has {} samples, but labels has {} samples",
            n_samples,
            labels.len()
        )));
    }

    // Default k to a safe value based on n_samples
    let k = k.unwrap_or_else(|| {
        if n_samples <= 1 {
            1
        } else if n_samples < 10 {
            std::cmp::min(2, n_samples - 1)
        } else {
            std::cmp::min(5, n_samples / 10)
        }
    });

    // Get unique cluster labels
    let mut unique_labels = Vec::new();
    for &label in labels.iter() {
        if !unique_labels.contains(&label) {
            unique_labels.push(label);
        }
    }

    if unique_labels.len() < 2 {
        return Err(MetricsError::InvalidInput(
            "At least two clusters are required to calculate DBCV".to_string(),
        ));
    }

    // Sort labels for consistent results
    unique_labels.sort();

    // Collect indices by cluster
    let mut cluster_indices = HashMap::new();
    for label in &unique_labels {
        cluster_indices.insert(*label, Vec::new());
    }

    for (i, &label) in labels.iter().enumerate() {
        if let Some(indices) = cluster_indices.get_mut(&label) {
            indices.push(i);
        }
    }

    // Calculate density sparseness for each cluster
    let mut sparseness_values = Vec::new();

    for &label in &unique_labels {
        let indices = cluster_indices.get(&label).unwrap();
        if indices.len() <= 1 {
            // Single point clusters have zero sparseness
            sparseness_values.push(F::zero());
            continue;
        }

        // Calculate core distance (k-nearest neighbor distance) for each point in the cluster
        let mut core_distances = Vec::new();

        for &i in indices {
            let mut distances = Vec::new();
            for &j in indices {
                if i != j {
                    let dist = calculate_euclidean_distance(&x.row(i), &x.row(j));
                    distances.push(dist);
                }
            }

            distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let k_actual = std::cmp::min(k, distances.len());

            if k_actual > 0 {
                core_distances.push(distances[k_actual - 1]);
            } else {
                core_distances.push(F::zero());
            }
        }

        // Calculate density sparseness (mutual reachability distance)
        let avg_core_distance = if !core_distances.is_empty() {
            core_distances.iter().fold(F::zero(), |sum, &val| sum + val)
                / F::from(core_distances.len()).unwrap()
        } else {
            F::zero()
        };

        // Calculate the variance of core distances
        let variance = if core_distances.len() > 1 {
            let mean = avg_core_distance;
            let sum_squared_diff = core_distances
                .iter()
                .fold(F::zero(), |sum, &val| sum + (val - mean) * (val - mean));
            sum_squared_diff / F::from(core_distances.len() - 1).unwrap()
        } else {
            F::zero()
        };

        // Sparseness is a function of the average core distance and variance
        let sparseness = avg_core_distance * (F::one() + variance.sqrt());
        sparseness_values.push(sparseness);
    }

    // Calculate cluster separation (density separation)
    let mut separation_matrix = vec![vec![F::zero(); unique_labels.len()]; unique_labels.len()];

    for (i, &label_i) in unique_labels.iter().enumerate() {
        let indices_i = cluster_indices.get(&label_i).unwrap();

        for (j, &label_j) in unique_labels.iter().enumerate() {
            if i == j {
                continue;
            }

            let indices_j = cluster_indices.get(&label_j).unwrap();

            // Calculate minimum distance between clusters
            let mut min_dist = F::max_value();

            for &idx_i in indices_i {
                for &idx_j in indices_j {
                    let dist = calculate_euclidean_distance(&x.row(idx_i), &x.row(idx_j));
                    min_dist = F::min(min_dist, dist);
                }
            }

            separation_matrix[i][j] = min_dist;
        }
    }

    // Calculate the DBCV for each cluster
    let mut cluster_validity = Vec::new();

    for (i, &_) in unique_labels.iter().enumerate() {
        let cluster_sparseness = sparseness_values[i];

        // Find minimum separation to other clusters
        let mut min_separation = F::max_value();
        for j in 0..unique_labels.len() {
            if i != j && separation_matrix[i][j] < min_separation {
                min_separation = separation_matrix[i][j];
            }
        }

        // If no other clusters or all separations are max_value
        if min_separation == F::max_value() {
            min_separation = F::zero();
        }

        // Calculate validity for this cluster
        let validity = if min_separation > F::zero() || cluster_sparseness > F::zero() {
            (min_separation - cluster_sparseness) / F::max(min_separation, cluster_sparseness)
        } else {
            F::zero()
        };

        cluster_validity.push(validity);
    }

    // Calculate DBCV as weighted average of cluster validities
    let mut weighted_sum = F::zero();
    let mut weight_sum = F::zero();

    for (i, &label) in unique_labels.iter().enumerate() {
        let weight = F::from(cluster_indices.get(&label).unwrap().len()).unwrap();
        weighted_sum += weight * cluster_validity[i];
        weight_sum += weight;
    }

    let dbcv = if weight_sum > F::zero() {
        weighted_sum / weight_sum
    } else {
        F::zero()
    };

    // DBCV ranges from -1 to 1
    Ok(dbcv)
}

/// Helper function to calculate Euclidean distance between two vectors
fn calculate_euclidean_distance<F, S1, S2>(a: &ArrayBase<S1, Ix1>, b: &ArrayBase<S2, Ix1>) -> F
where
    F: Float,
    S1: Data<Elem = F>,
    S2: Data<Elem = F>,
{
    let mut sum = F::zero();
    for (x, y) in a.iter().zip(b.iter()) {
        let diff = *x - *y;
        sum = sum + diff * diff;
    }
    sum.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_local_density_factor() {
        // Create a simple dataset with two well-separated clusters
        let well_separated = Array2::from_shape_vec(
            (6, 2),
            vec![
                1.0, 2.0, 1.5, 1.8, 1.2, 2.2, 10.0, 12.0, 10.2, 11.8, 10.5, 12.2,
            ],
        )
        .unwrap();

        let labels = ndarray::array![0, 0, 0, 1, 1, 1];

        // Calculate LDF with k=2 (small enough for this test case)
        let factors = local_density_factor(&well_separated, &labels, Some(2)).unwrap();

        // Both clusters should have similar density factors
        assert!(factors.len() == 2);
        assert!(factors.contains_key(&0));
        assert!(factors.contains_key(&1));

        // Create a dataset with varying density
        let varying_density = Array2::from_shape_vec(
            (6, 2),
            vec![
                1.0, 1.1, 1.05, 1.05, 1.1, 1.0, // Dense cluster
                5.0, 5.0, 6.0, 6.0, 7.0, 7.0, // Sparse cluster
            ],
        )
        .unwrap();

        let labels = ndarray::array![0, 0, 0, 1, 1, 1];

        // Calculate LDF with k=2
        let factors = local_density_factor(&varying_density, &labels, Some(2)).unwrap();

        // Dense cluster should have higher factor than sparse cluster
        assert!(*factors.get(&0).unwrap() > *factors.get(&1).unwrap());
    }

    #[test]
    fn test_relative_density_index() {
        // Create a simple dataset with two well-separated clusters
        let well_separated = Array2::from_shape_vec(
            (6, 2),
            vec![
                1.0, 2.0, 1.5, 1.8, 1.2, 2.2, 10.0, 12.0, 10.2, 11.8, 10.5, 12.2,
            ],
        )
        .unwrap();

        let labels = ndarray::array![0, 0, 0, 1, 1, 1];

        // Calculate RDI with k=2
        let rdi = relative_density_index(&well_separated, &labels, Some(2)).unwrap();

        // Well-separated clusters should have high RDI
        assert!(rdi > 1.0);

        // Create a dataset with overlapping clusters
        let overlapping = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 1.5, 1.8, 3.0, 3.0, 3.0, 3.0, 4.0, 4.5, 5.0, 5.5],
        )
        .unwrap();

        let labels = ndarray::array![0, 0, 0, 1, 1, 1];

        // Calculate RDI for overlapping clusters
        let rdi_overlapping = relative_density_index(&overlapping, &labels, Some(2)).unwrap();

        // Overlapping clusters should have lower RDI
        assert!(rdi > rdi_overlapping);
    }

    #[test]
    fn test_density_based_cluster_validity() {
        // Create a simple dataset with two well-separated clusters
        let well_separated = Array2::from_shape_vec(
            (6, 2),
            vec![
                1.0, 2.0, 1.5, 1.8, 1.2, 2.2, 10.0, 12.0, 10.2, 11.8, 10.5, 12.2,
            ],
        )
        .unwrap();

        let labels = ndarray::array![0, 0, 0, 1, 1, 1];

        // Calculate DBCV with k=2
        let dbcv = density_based_cluster_validity(&well_separated, &labels, Some(2)).unwrap();

        // DBCV should be positive for well-separated clusters
        assert!(dbcv > 0.0);

        // Create a dataset with poor clustering
        let poor_clustering = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 8.0, 9.0, 1.2, 2.2, 8.0, 9.0, 1.0, 2.0, 8.0, 9.0],
        )
        .unwrap();

        // Labels not matching the natural clusters
        let bad_labels = ndarray::array![0, 0, 0, 1, 1, 1];

        // Calculate DBCV for poor clustering
        let bad_dbcv =
            density_based_cluster_validity(&poor_clustering, &bad_labels, Some(2)).unwrap();

        // DBCV should be lower for poorly defined clusters
        assert!(dbcv > bad_dbcv);
    }
}
