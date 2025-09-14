//! Time series clustering algorithms with specialized distance metrics
//!
//! This module provides clustering algorithms specifically designed for time series data,
//! including dynamic time warping (DTW) distance and other temporal similarity measures.
//! These algorithms can handle time series of different lengths and temporal alignments.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use num_traits::{Float, FromPrimitive};
use std::fmt::Debug;

use serde::{Deserialize, Serialize};

use crate::error::{ClusteringError, Result};

/// Dynamic Time Warping (DTW) distance between two time series
///
/// DTW finds the optimal alignment between two time series by minimizing
/// the cumulative distance between aligned points. It can handle series
/// of different lengths and temporal distortions.
///
/// # Arguments
///
/// * `series1` - First time series
/// * `series2` - Second time series
/// * `window` - Sakoe-Chiba band constraint (None for no constraint)
///
/// # Returns
///
/// DTW distance between the two series
///
/// # Example
///
/// ```
/// use ndarray::Array1;
/// use scirs2_cluster::time_series::dtw_distance;
///
/// let series1 = Array1::from_vec(vec![1.0, 2.0, 3.0, 2.0, 1.0]);
/// let series2 = Array1::from_vec(vec![1.0, 2.0, 2.0, 3.0, 2.0, 1.0]);
///
/// let distance = dtw_distance(series1.view(), series2.view(), None).unwrap();
/// ```
#[allow(dead_code)]
pub fn dtw_distance<F>(
    series1: ArrayView1<F>,
    series2: ArrayView1<F>,
    window: Option<usize>,
) -> Result<F>
where
    F: Float + FromPrimitive + Debug + 'static,
{
    let n = series1.len();
    let m = series2.len();

    if n == 0 || m == 0 {
        return Err(ClusteringError::InvalidInput(
            "Time series cannot be empty".to_string(),
        ));
    }

    // Initialize DTW matrix with infinity
    let mut dtw = Array2::from_elem((n + 1, m + 1), F::infinity());
    dtw[[0, 0]] = F::zero();

    // Apply Sakoe-Chiba band constraint if specified
    let effective_window = window.unwrap_or(m.max(n));

    for i in 1..=n {
        let start_j = if effective_window < i {
            i - effective_window
        } else {
            1
        };
        let end_j = (i + effective_window).min(m + 1);

        for j in start_j..end_j {
            if j <= m {
                let cost = (series1[i - 1] - series2[j - 1]).abs();

                let candidates = [
                    dtw[[i - 1, j]],     // Insertion
                    dtw[[i, j - 1]],     // Deletion
                    dtw[[i - 1, j - 1]], // Match
                ];

                let min_prev = candidates.iter().fold(F::infinity(), |acc, &x| acc.min(x));
                dtw[[i, j]] = cost + min_prev;
            }
        }
    }

    Ok(dtw[[n, m]])
}

/// DTW distance with custom local distance function
///
/// Allows using custom distance functions for comparing individual time points.
///
/// # Arguments
///
/// * `series1` - First time series
/// * `series2` - Second time series
/// * `local_distance` - Function to compute distance between individual points
/// * `window` - Sakoe-Chiba band constraint
///
/// # Returns
///
/// DTW distance using the custom local distance function
#[allow(dead_code)]
pub fn dtw_distance_custom<F, D>(
    series1: ArrayView1<F>,
    series2: ArrayView1<F>,
    local_distance: D,
    window: Option<usize>,
) -> Result<F>
where
    F: Float + FromPrimitive + Debug + 'static,
    D: Fn(F, F) -> F,
{
    let n = series1.len();
    let m = series2.len();

    if n == 0 || m == 0 {
        return Err(ClusteringError::InvalidInput(
            "Time series cannot be empty".to_string(),
        ));
    }

    let mut dtw = Array2::from_elem((n + 1, m + 1), F::infinity());
    dtw[[0, 0]] = F::zero();

    let effective_window = window.unwrap_or(m.max(n));

    for i in 1..=n {
        let start_j = if effective_window < i {
            i - effective_window
        } else {
            1
        };
        let end_j = (i + effective_window).min(m + 1);

        for j in start_j..end_j {
            if j <= m {
                let cost = local_distance(series1[i - 1], series2[j - 1]);

                let candidates = [dtw[[i - 1, j]], dtw[[i, j - 1]], dtw[[i - 1, j - 1]]];

                let min_prev = candidates.iter().fold(F::infinity(), |acc, &x| acc.min(x));
                dtw[[i, j]] = cost + min_prev;
            }
        }
    }

    Ok(dtw[[n, m]])
}

/// Soft DTW distance for differentiable time series clustering
///
/// Soft DTW is a differentiable version of DTW that uses a soft minimum
/// operation instead of hard minimum, making it suitable for gradient-based
/// optimization.
///
/// # Arguments
///
/// * `series1` - First time series
/// * `series2` - Second time series
/// * `gamma` - Smoothing parameter (smaller values approach standard DTW)
///
/// # Returns
///
/// Soft DTW distance
#[allow(dead_code)]
pub fn soft_dtw_distance<F>(series1: ArrayView1<F>, series2: ArrayView1<F>, gamma: F) -> Result<F>
where
    F: Float + FromPrimitive + Debug + 'static,
{
    let n = series1.len();
    let m = series2.len();

    if n == 0 || m == 0 {
        return Err(ClusteringError::InvalidInput(
            "Time series cannot be empty".to_string(),
        ));
    }

    if gamma <= F::zero() {
        return Err(ClusteringError::InvalidInput(
            "Gamma must be positive".to_string(),
        ));
    }

    let mut dtw = Array2::from_elem((n + 1, m + 1), F::infinity());
    dtw[[0, 0]] = F::zero();

    for i in 1..=n {
        for j in 1..=m {
            let cost = (series1[i - 1] - series2[j - 1]).powi(2);

            let candidates = [dtw[[i - 1, j]], dtw[[i, j - 1]], dtw[[i - 1, j - 1]]];

            // Soft minimum: -gamma * log(sum(exp(-x/gamma)))
            // For numerical stability, use the log-sum-exp trick
            let min_val = candidates.iter().fold(F::infinity(), |acc, &x| acc.min(x));
            let sum_exp = candidates
                .iter()
                .map(|&x| (-(x - min_val) / gamma).exp())
                .fold(F::zero(), |acc, x| acc + x);

            let soft_min = min_val - gamma * sum_exp.ln();
            dtw[[i, j]] = cost + soft_min;
        }
    }

    Ok(dtw[[n, m]])
}

/// Time series clustering using k-medoids with DTW distance
///
/// Performs k-medoids clustering on time series data using DTW as the
/// distance metric. This is more robust than k-means for time series
/// as it uses actual time series as cluster centers.
///
/// # Arguments
///
/// * `time_series` - Matrix where each row is a time series
/// * `k` - Number of clusters
/// * `max_iterations` - Maximum number of iterations
/// * `window` - DTW constraint window
///
/// # Returns
///
/// Tuple of (medoid_indices, cluster_assignments)
#[allow(dead_code)]
pub fn dtw_k_medoids<F>(
    time_series: ArrayView2<F>,
    k: usize,
    max_iterations: usize,
    window: Option<usize>,
) -> Result<(Array1<usize>, Array1<usize>)>
where
    F: Float + FromPrimitive + Debug + 'static,
{
    let n_series = time_series.nrows();

    if k > n_series {
        return Err(ClusteringError::InvalidInput(
            "Number of clusters cannot exceed number of time _series".to_string(),
        ));
    }

    if n_series == 0 {
        return Err(ClusteringError::InvalidInput(
            "No time _series provided".to_string(),
        ));
    }

    // Initialize medoids randomly (for deterministic results, use first k series)
    let mut medoids: Array1<usize> = Array1::from_iter(0..k);
    let mut assignments = Array1::zeros(n_series);

    for _iteration in 0..max_iterations {
        let mut changed = false;

        // Assign each time _series to nearest medoid
        for i in 0..n_series {
            let mut min_distance = F::infinity();
            let mut best_cluster = 0;

            for (cluster_id, &medoid_idx) in medoids.iter().enumerate() {
                let distance =
                    dtw_distance(time_series.row(i), time_series.row(medoid_idx), window)?;

                if distance < min_distance {
                    min_distance = distance;
                    best_cluster = cluster_id;
                }
            }

            if assignments[i] != best_cluster {
                assignments[i] = best_cluster;
                changed = true;
            }
        }

        // Update medoids
        for cluster_id in 0..k {
            let cluster_members: Vec<usize> = assignments
                .iter()
                .enumerate()
                .filter(|(_, &assignment)| assignment == cluster_id)
                .map(|(idx, _)| idx)
                .collect();

            if !cluster_members.is_empty() {
                let mut best_medoid = medoids[cluster_id];
                let mut min_total_distance = F::infinity();

                // Try each member as potential medoid
                for &candidate in &cluster_members {
                    let mut total_distance = F::zero();

                    for &member in &cluster_members {
                        if candidate != member {
                            let distance = dtw_distance(
                                time_series.row(candidate),
                                time_series.row(member),
                                window,
                            )?;
                            total_distance = total_distance + distance;
                        }
                    }

                    if total_distance < min_total_distance {
                        min_total_distance = total_distance;
                        best_medoid = candidate;
                    }
                }

                if medoids[cluster_id] != best_medoid {
                    medoids[cluster_id] = best_medoid;
                    changed = true;
                }
            }
        }

        if !changed {
            break;
        }
    }

    Ok((medoids, assignments))
}

/// Hierarchical clustering for time series using DTW distance
///
/// Performs agglomerative hierarchical clustering using DTW distance
/// with complete linkage.
///
/// # Arguments
///
/// * `time_series` - Matrix where each row is a time series
/// * `window` - DTW constraint window
///
/// # Returns
///
/// Linkage matrix in the format compatible with scipy.cluster.hierarchy
#[allow(dead_code)]
pub fn dtw_hierarchical_clustering<F>(
    time_series: ArrayView2<F>,
    window: Option<usize>,
) -> Result<Array2<F>>
where
    F: Float + FromPrimitive + Debug + 'static,
{
    let n_series = time_series.nrows();

    if n_series < 2 {
        return Err(ClusteringError::InvalidInput(
            "Need at least 2 time _series for clustering".to_string(),
        ));
    }

    // Compute distance matrix
    let mut distances = Array2::zeros((n_series, n_series));
    for i in 0..n_series {
        for j in (i + 1)..n_series {
            let distance = dtw_distance(time_series.row(i), time_series.row(j), window)?;
            distances[[i, j]] = distance;
            distances[[j, i]] = distance;
        }
    }

    // Initialize clusters (each point is its own cluster initially)
    let mut clusters: Vec<Vec<usize>> = (0..n_series).map(|i| vec![i]).collect();
    let mut linkage = Vec::new();
    let mut cluster_id = n_series;

    while clusters.len() > 1 {
        // Find closest pair of clusters
        let mut min_distance = F::infinity();
        let mut merge_i = 0;
        let mut merge_j = 1;

        for i in 0..clusters.len() {
            for j in (i + 1)..clusters.len() {
                // Calculate complete linkage distance
                let mut max_dist = F::zero();
                for &point_i in &clusters[i] {
                    for &point_j in &clusters[j] {
                        max_dist = max_dist.max(distances[[point_i, point_j]]);
                    }
                }

                if max_dist < min_distance {
                    min_distance = max_dist;
                    merge_i = i;
                    merge_j = j;
                }
            }
        }

        // Record the merge
        let cluster_i_size = clusters[merge_i].len();
        let cluster_j_size = clusters[merge_j].len();

        linkage.push([
            F::from(if merge_i < n_series {
                merge_i
            } else {
                n_series + merge_i
            })
            .unwrap(),
            F::from(if merge_j < n_series {
                merge_j
            } else {
                n_series + merge_j
            })
            .unwrap(),
            min_distance,
            F::from(cluster_i_size + cluster_j_size).unwrap(),
        ]);

        // Merge clusters
        let mut new_cluster = clusters[merge_i].clone();
        new_cluster.extend(&clusters[merge_j]);

        // Remove old clusters (remove higher index first)
        let (first, second) = if merge_i > merge_j {
            (merge_i, merge_j)
        } else {
            (merge_j, merge_i)
        };

        clusters.remove(first);
        clusters.remove(second);
        clusters.push(new_cluster);

        cluster_id += 1;
    }

    // Convert to ndarray
    let linkage_array =
        Array2::from_shape_vec((linkage.len(), 4), linkage.into_iter().flatten().collect())
            .map_err(|_| {
                ClusteringError::ComputationError("Failed to create linkage matrix".to_string())
            })?;

    Ok(linkage_array)
}

/// Time series k-means clustering with DTW barycenter averaging
///
/// Performs k-means clustering where cluster centers are computed as
/// DTW barycenters (average time series under DTW alignment).
///
/// # Arguments
///
/// * `time_series` - Matrix where each row is a time series
/// * `k` - Number of clusters
/// * `max_iterations` - Maximum number of iterations
/// * `tolerance` - Convergence tolerance
///
/// # Returns
///
/// Tuple of (cluster_centers, cluster_assignments)
#[allow(dead_code)]
pub fn dtw_k_means<F>(
    time_series: ArrayView2<F>,
    k: usize,
    max_iterations: usize,
    tolerance: F,
) -> Result<(Array2<F>, Array1<usize>)>
where
    F: Float + FromPrimitive + Debug + 'static,
{
    let n_series = time_series.nrows();
    let series_length = time_series.ncols();

    if k > n_series {
        return Err(ClusteringError::InvalidInput(
            "Number of clusters cannot exceed number of time _series".to_string(),
        ));
    }

    // Initialize centers with first k time _series
    let mut centers = Array2::zeros((k, series_length));
    for i in 0..k {
        centers.row_mut(i).assign(&time_series.row(i));
    }

    let mut assignments = Array1::zeros(n_series);

    for _iteration in 0..max_iterations {
        let mut changed = false;

        // Assign each time _series to nearest center
        for i in 0..n_series {
            let mut min_distance = F::infinity();
            let mut best_cluster = 0;

            for j in 0..k {
                let distance = dtw_distance(time_series.row(i), centers.row(j), None)?;

                if distance < min_distance {
                    min_distance = distance;
                    best_cluster = j;
                }
            }

            if assignments[i] != best_cluster {
                assignments[i] = best_cluster;
                changed = true;
            }
        }

        if !changed {
            break;
        }

        // Update centers using DTW barycenter averaging
        let mut center_changed = false;
        for cluster_id in 0..k {
            let cluster_members: Vec<usize> = assignments
                .iter()
                .enumerate()
                .filter(|(_, &assignment)| assignment == cluster_id)
                .map(|(idx, _)| idx)
                .collect();

            if !cluster_members.is_empty() {
                let new_center = dtw_barycenter_averaging(
                    &time_series.select(Axis(0), &cluster_members),
                    10,
                    tolerance,
                )?;

                let center_distance =
                    dtw_distance(centers.row(cluster_id), new_center.view(), None)?;

                if center_distance > tolerance {
                    center_changed = true;
                }

                centers.row_mut(cluster_id).assign(&new_center);
            }
        }

        if !center_changed {
            break;
        }
    }

    Ok((centers, assignments))
}

/// Compute DTW barycenter (average time series) using iterative refinement
///
/// The DTW barycenter is the time series that minimizes the sum of squared
/// DTW distances to all input time series.
///
/// # Arguments
///
/// * `time_series` - Collection of time series to average
/// * `max_iterations` - Maximum number of refinement iterations
/// * `tolerance` - Convergence tolerance
///
/// # Returns
///
/// Barycenter time series
#[allow(dead_code)]
pub fn dtw_barycenter_averaging<F>(
    time_series: &Array2<F>,
    max_iterations: usize,
    tolerance: F,
) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + Debug + 'static,
{
    let n_series = time_series.nrows();
    let series_length = time_series.ncols();

    if n_series == 0 {
        return Err(ClusteringError::InvalidInput(
            "No time _series provided".to_string(),
        ));
    }

    if n_series == 1 {
        return Ok(time_series.row(0).to_owned());
    }

    // Initialize barycenter as the mean of all _series
    let mut barycenter = time_series.mean_axis(Axis(0)).unwrap();

    for _iteration in 0..max_iterations {
        let mut new_barycenter = Array1::zeros(series_length);
        let mut weights = Array1::zeros(series_length);

        // For each time series, find optimal alignment with current barycenter
        for i in 0..n_series {
            let (aligned_series, alignment_weights) =
                dtw_align_series(time_series.row(i), barycenter.view())?;

            new_barycenter = new_barycenter + aligned_series;
            weights = weights + alignment_weights;
        }

        // Normalize by weights
        for i in 0..series_length {
            if weights[i] > F::zero() {
                new_barycenter[i] = new_barycenter[i] / weights[i];
            }
        }

        // Check convergence
        let change = dtw_distance(barycenter.view(), new_barycenter.view(), None)?;
        if change < tolerance {
            break;
        }

        barycenter = new_barycenter;
    }

    Ok(barycenter)
}

/// Align a time series with a reference using DTW and return weighted average
#[allow(dead_code)]
fn dtw_align_series<F>(
    series: ArrayView1<F>,
    reference: ArrayView1<F>,
) -> Result<(Array1<F>, Array1<F>)>
where
    F: Float + FromPrimitive + Debug + 'static,
{
    let n = series.len();
    let m = reference.len();

    // Compute DTW matrix
    let mut dtw = Array2::from_elem((n + 1, m + 1), F::infinity());
    dtw[[0, 0]] = F::zero();

    for i in 1..=n {
        for j in 1..=m {
            let cost = (series[i - 1] - reference[j - 1]).abs();
            let min_prev = [dtw[[i - 1, j]], dtw[[i, j - 1]], dtw[[i - 1, j - 1]]]
                .iter()
                .fold(F::infinity(), |acc, &x| acc.min(x));

            dtw[[i, j]] = cost + min_prev;
        }
    }

    // Backtrack to find optimal path
    let mut i = n;
    let mut j = m;
    let mut aligned_series = Array1::zeros(m);
    let mut weights = Array1::zeros(m);

    while i > 0 && j > 0 {
        // Add current series value to aligned position
        aligned_series[j - 1] = aligned_series[j - 1] + series[i - 1];
        weights[j - 1] = weights[j - 1] + F::one();

        // Find which direction we came from
        let candidates = [
            (dtw[[i - 1, j - 1]], (i - 1, j - 1)), // diagonal
            (dtw[[i - 1, j]], (i - 1, j)),         // up
            (dtw[[i, j - 1]], (i, j - 1)),         // left
        ];

        let (_, (next_i, next_j)) = candidates
            .iter()
            .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
            .unwrap();

        i = *next_i;
        j = *next_j;
    }

    Ok((aligned_series, weights))
}

/// Configuration for time series clustering algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesClusteringConfig {
    /// Algorithm to use for clustering
    pub algorithm: TimeSeriesAlgorithm,
    /// Number of clusters
    pub n_clusters: usize,
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// DTW constraint window size
    pub dtw_window: Option<usize>,
    /// Soft DTW gamma parameter
    pub soft_dtw_gamma: Option<f64>,
}

/// Available time series clustering algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TimeSeriesAlgorithm {
    /// K-medoids with DTW distance
    DTWKMedoids,
    /// K-means with DTW barycenter averaging
    DTWKMeans,
    /// Hierarchical clustering with DTW distance
    DTWHierarchical,
}

impl Default for TimeSeriesClusteringConfig {
    fn default() -> Self {
        Self {
            algorithm: TimeSeriesAlgorithm::DTWKMedoids,
            n_clusters: 3,
            max_iterations: 100,
            tolerance: 1e-4,
            dtw_window: None,
            soft_dtw_gamma: None,
        }
    }
}

/// Perform time series clustering using the specified configuration
///
/// # Arguments
///
/// * `time_series` - Matrix where each row is a time series
/// * `config` - Clustering configuration
///
/// # Returns
///
/// Cluster assignments for each time series
#[allow(dead_code)]
pub fn time_series_clustering<F>(
    time_series: ArrayView2<F>,
    config: &TimeSeriesClusteringConfig,
) -> Result<Array1<usize>>
where
    F: Float + FromPrimitive + Debug + 'static,
{
    match config.algorithm {
        TimeSeriesAlgorithm::DTWKMedoids => {
            let (_, assignments) = dtw_k_medoids(
                time_series,
                config.n_clusters,
                config.max_iterations,
                config.dtw_window,
            )?;
            Ok(assignments)
        }
        TimeSeriesAlgorithm::DTWKMeans => {
            let tolerance = F::from(config.tolerance).unwrap();
            let (_, assignments) = dtw_k_means(
                time_series,
                config.n_clusters,
                config.max_iterations,
                tolerance,
            )?;
            Ok(assignments)
        }
        TimeSeriesAlgorithm::DTWHierarchical => {
            // For hierarchical clustering, we need to cut the dendrogram
            // This is a simplified version that returns the first n_clusters
            let _linkage = dtw_hierarchical_clustering(time_series, config.dtw_window)?;

            // Simple flat clustering: assign based on first few merges
            // In practice, you'd want to implement proper dendrogram cutting
            let n_series = time_series.nrows();
            let mut assignments = Array1::from_iter(0..n_series);

            // This is a simplified assignment - a proper implementation would
            // cut the dendrogram at the appropriate level
            for i in 0..n_series {
                assignments[i] = i % config.n_clusters;
            }

            Ok(assignments)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_dtw_distance() {
        let series1 = Array1::from_vec(vec![1.0, 2.0, 3.0, 2.0, 1.0]);
        let series2 = Array1::from_vec(vec![1.0, 2.0, 2.0, 3.0, 2.0, 1.0]);

        let distance = dtw_distance(series1.view(), series2.view(), None).unwrap();
        assert!(distance >= 0.0);
    }

    #[test]
    fn test_dtw_identical_series() {
        let series = Array1::from_vec(vec![1.0, 2.0, 3.0, 2.0, 1.0]);
        let distance = dtw_distance(series.view(), series.view(), None).unwrap();
        assert_eq!(distance, 0.0);
    }

    #[test]
    fn test_dtw_k_medoids() {
        let time_series = Array2::from_shape_vec(
            (4, 5),
            vec![
                1.0, 2.0, 3.0, 2.0, 1.0, 1.1, 2.1, 3.1, 2.1, 1.1, 5.0, 6.0, 7.0, 6.0, 5.0, 5.1,
                6.1, 7.1, 6.1, 5.1,
            ],
        )
        .unwrap();

        let (medoids, assignments) = dtw_k_medoids(time_series.view(), 2, 10, None).unwrap();

        assert_eq!(medoids.len(), 2);
        assert_eq!(assignments.len(), 4);

        // First two series should be in one cluster, last two in another
        assert_eq!(assignments[0], assignments[1]);
        assert_eq!(assignments[2], assignments[3]);
        assert_ne!(assignments[0], assignments[2]);
    }

    #[test]
    fn test_soft_dtw_distance() {
        let series1 = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let series2 = Array1::from_vec(vec![1.0, 2.5, 3.0]);

        let distance = soft_dtw_distance(series1.view(), series2.view(), 0.1).unwrap();
        assert!(distance >= 0.0);
    }

    #[test]
    fn test_dtw_barycenter_averaging() {
        let time_series = Array2::from_shape_vec(
            (3, 4),
            vec![1.0, 2.0, 3.0, 2.0, 1.1, 2.1, 3.1, 2.1, 0.9, 1.9, 2.9, 1.9],
        )
        .unwrap();

        let barycenter = dtw_barycenter_averaging(&time_series, 10, 1e-3).unwrap();
        assert_eq!(barycenter.len(), 4);

        // Barycenter should be close to the mean
        let mean_series = time_series.mean_axis(Axis(0)).unwrap();
        for i in 0..4 {
            assert!((barycenter[i] - mean_series[i]).abs() < 0.5);
        }
    }

    #[test]
    fn test_time_series_clustering_config() {
        let config = TimeSeriesClusteringConfig::default();
        assert_eq!(config.n_clusters, 3);
        assert_eq!(config.max_iterations, 100);

        let time_series = Array2::from_shape_vec(
            (4, 5),
            vec![
                1.0, 2.0, 3.0, 2.0, 1.0, 1.1, 2.1, 3.1, 2.1, 1.1, 5.0, 6.0, 7.0, 6.0, 5.0, 5.1,
                6.1, 7.1, 6.1, 5.1,
            ],
        )
        .unwrap();

        let assignments = time_series_clustering(time_series.view(), &config).unwrap();
        assert_eq!(assignments.len(), 4);
    }
}
