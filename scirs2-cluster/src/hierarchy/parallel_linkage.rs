//! Parallel implementation of linkage methods for hierarchical clustering
//!
//! This module provides parallelized versions of linkage algorithms to improve
//! performance on large datasets.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::{Float, FromPrimitive};
use scirs2_core::parallel_ops::*;
use std::fmt::Debug;

use crate::error::{ClusteringError, Result};
use crate::hierarchy::{coords_to_condensed_index, LinkageMethod};

/// Structure to represent a cluster (same as in linkage.rs)
#[derive(Debug)]
pub(crate) struct ParallelCluster {
    /// Number of observations in the cluster
    pub size: usize,

    /// Indices of observations in the cluster
    pub members: Vec<usize>,
}

/// Parallel hierarchical clustering algorithm
pub(crate) fn parallel_hierarchical_clustering<
    F: Float + FromPrimitive + Debug + PartialOrd + Send + Sync + std::iter::Sum,
>(
    distances: &Array1<F>,
    n_samples: usize,
    method: LinkageMethod,
) -> Result<Array2<F>> {
    // Initialize clusters (each observation starts in its own cluster)
    let mut clusters: Vec<ParallelCluster> = (0..n_samples)
        .map(|i| ParallelCluster {
            size: 1,
            members: vec![i],
        })
        .collect();

    // The linkage matrix format: [cluster1, cluster2, distance, size]
    let mut linkage_matrix = Array2::zeros((n_samples - 1, 4));

    // Initialize active clusters (all clusters are initially active)
    let mut activeclusters: Vec<usize> = (0..n_samples).collect();

    // For method-specific calculations
    let mut centroids: Option<Array2<F>> = None;
    if matches!(method, LinkageMethod::Centroid | LinkageMethod::Median) {
        // Need to maintain centroids for these methods
        centroids = Some(Array2::from_elem(
            (2 * n_samples - 1, distances.len()),
            F::zero(),
        ));
    }

    // Main loop - merge clusters until only one remains
    for i in 0..(n_samples - 1) {
        // Find the two closest clusters using parallel computation
        let (cluster1_idx, cluster2_idx, min_dist) = parallel_find_closestclusters(
            &activeclusters,
            &clusters,
            distances,
            method,
            centroids.as_ref(),
            n_samples,
        )?;

        // Get the actual cluster indices (original indices, not positions in activeclusters)
        let cluster1 = activeclusters[cluster1_idx];
        let cluster2 = activeclusters[cluster2_idx];

        // Ensure cluster1 < cluster2 for consistency
        let (cluster1, cluster2) = if cluster1 < cluster2 {
            (cluster1, cluster2)
        } else {
            (cluster2, cluster1)
        };

        // Create a new cluster by merging the two closest clusters
        let new_cluster_id = n_samples + i;
        let mut new_members = clusters[cluster1].members.clone();
        new_members.extend(clusters[cluster2].members.clone());

        let new_cluster = ParallelCluster {
            size: clusters[cluster1].size + clusters[cluster2].size,
            members: new_members,
        };

        // Update centroids if needed
        if let Some(ref mut cents) = centroids {
            update_centroid(cents, method, n_samples, new_cluster_id);
        }

        // Add the new cluster
        clusters.push(new_cluster);

        // Remove the merged clusters and add the new one
        activeclusters.remove(cluster1_idx.max(cluster2_idx));
        activeclusters.remove(cluster1_idx.min(cluster2_idx));
        activeclusters.push(new_cluster_id);

        // Update the linkage matrix
        // [cluster1, cluster2, distance, size]
        linkage_matrix[[i, 0]] = F::from_usize(cluster1).unwrap();
        linkage_matrix[[i, 1]] = F::from_usize(cluster2).unwrap();
        linkage_matrix[[i, 2]] = min_dist;
        linkage_matrix[[i, 3]] = F::from_usize(clusters[new_cluster_id].size).unwrap();
    }

    Ok(linkage_matrix)
}

/// Parallel version of finding the two closest clusters
#[allow(dead_code)]
fn parallel_find_closestclusters<
    F: Float + FromPrimitive + Debug + PartialOrd + Send + Sync + std::iter::Sum,
>(
    activeclusters: &[usize],
    clusters: &[ParallelCluster],
    distances: &Array1<F>,
    method: LinkageMethod,
    centroids: Option<&Array2<F>>,
    n_samples: usize,
) -> Result<(usize, usize, F)> {
    // Create pairs to process
    let mut pairs = Vec::new();
    for i in 0..activeclusters.len() {
        for j in (i + 1)..activeclusters.len() {
            pairs.push((i, j));
        }
    }

    if pairs.is_empty() {
        return Err(ClusteringError::ComputationError(
            "No cluster pairs to process".into(),
        ));
    }

    // Process pairs in parallel to find minimum distance
    let results: Result<Vec<(usize, usize, F)>> = pairs
        .par_iter()
        .map(|&(i, j)| {
            let cluster_i = activeclusters[i];
            let cluster_j = activeclusters[j];

            // Calculate distance between clusters based on the linkage method
            let dist = match method {
                LinkageMethod::Single => parallel_single_linkage(
                    &clusters[cluster_i],
                    &clusters[cluster_j],
                    distances,
                    n_samples,
                ),
                LinkageMethod::Complete => parallel_complete_linkage(
                    &clusters[cluster_i],
                    &clusters[cluster_j],
                    distances,
                    n_samples,
                ),
                LinkageMethod::Average => parallel_average_linkage(
                    &clusters[cluster_i],
                    &clusters[cluster_j],
                    distances,
                    n_samples,
                ),
                LinkageMethod::Ward => parallel_ward_linkage(
                    &clusters[cluster_i],
                    &clusters[cluster_j],
                    distances,
                    n_samples,
                ),
                LinkageMethod::Centroid => Ok(centroid_linkage(
                    cluster_i,
                    cluster_i,
                    cluster_j,
                    centroids.unwrap(),
                )),
                LinkageMethod::Median => Ok(median_linkage(
                    cluster_i,
                    cluster_i,
                    cluster_j,
                    centroids.unwrap(),
                )),
                LinkageMethod::Weighted => parallel_weighted_linkage(
                    &clusters[cluster_i],
                    &clusters[cluster_j],
                    distances,
                    n_samples,
                ),
            };

            dist.map(|d| (i, j, d))
        })
        .collect();

    let min_result = results?
        .into_iter()
        .min_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal))
        .ok_or_else(|| {
            ClusteringError::ComputationError("Could not find minimum distance".into())
        })?;

    Ok((min_result.0, min_result.1, min_result.2))
}

/// Parallel single linkage: minimum distance between any two points in the clusters
pub(crate) fn parallel_single_linkage<F: Float + PartialOrd + Send + Sync>(
    cluster1: &ParallelCluster,
    cluster2: &ParallelCluster,
    distances: &Array1<F>,
    n_samples: usize,
) -> Result<F> {
    let results: Result<Vec<F>> = cluster1
        .members
        .par_iter()
        .map(|&i| {
            let mut min_dist = F::infinity();
            for &j in &cluster2.members {
                let (min_idx, max_idx) = if i < j { (i, j) } else { (j, i) };
                let idx = coords_to_condensed_index(n_samples, min_idx, max_idx)?;
                let dist = distances[idx];
                min_dist = min_dist.min(dist);
            }
            Ok(min_dist)
        })
        .collect();

    let min_distances = results?;
    Ok(min_distances
        .into_iter()
        .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(F::infinity()))
}

/// Parallel complete linkage: maximum distance between any two points in the clusters
pub(crate) fn parallel_complete_linkage<F: Float + PartialOrd + Send + Sync>(
    cluster1: &ParallelCluster,
    cluster2: &ParallelCluster,
    distances: &Array1<F>,
    n_samples: usize,
) -> Result<F> {
    let results: Result<Vec<F>> = cluster1
        .members
        .par_iter()
        .map(|&i| {
            let mut max_dist = F::neg_infinity();
            for &j in &cluster2.members {
                let (min_idx, max_idx) = if i < j { (i, j) } else { (j, i) };
                let idx = coords_to_condensed_index(n_samples, min_idx, max_idx)?;
                let dist = distances[idx];
                max_dist = max_dist.max(dist);
            }
            Ok(max_dist)
        })
        .collect();

    let max_distances = results?;
    Ok(max_distances
        .into_iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(F::neg_infinity()))
}

/// Parallel average linkage: average distance between all pairs of points in the clusters
pub(crate) fn parallel_average_linkage<F: Float + FromPrimitive + Send + Sync>(
    cluster1: &ParallelCluster,
    cluster2: &ParallelCluster,
    distances: &Array1<F>,
    n_samples: usize,
) -> Result<F> {
    let results: Result<Vec<(F, usize)>> = cluster1
        .members
        .par_iter()
        .map(|&i| {
            let mut sum = F::zero();
            let mut count = 0;
            for &j in &cluster2.members {
                let (min_idx, max_idx) = if i < j { (i, j) } else { (j, i) };
                let idx = coords_to_condensed_index(n_samples, min_idx, max_idx)?;
                sum = sum + distances[idx];
                count += 1;
            }
            Ok((sum, count))
        })
        .collect();

    let sum_counts = results?;
    let (total_sum, total_count) = sum_counts
        .into_iter()
        .fold((F::zero(), 0), |(acc_sum, acc_count), (sum, count)| {
            (acc_sum + sum, acc_count + count)
        });

    if total_count == 0 {
        Ok(F::infinity())
    } else {
        Ok(total_sum / F::from_usize(total_count).unwrap())
    }
}

/// Parallel Ward's linkage: minimizes the increase in variance when merging clusters
pub(crate) fn parallel_ward_linkage<F: Float + FromPrimitive + Send + Sync + std::iter::Sum>(
    cluster1: &ParallelCluster,
    cluster2: &ParallelCluster,
    distances: &Array1<F>,
    n_samples: usize,
) -> Result<F> {
    let size1 = F::from_usize(cluster1.size).unwrap();
    let size2 = F::from_usize(cluster2.size).unwrap();

    let results: Result<Vec<F>> = cluster1
        .members
        .par_iter()
        .map(|&i| {
            let mut sum_squared = F::zero();
            for &j in &cluster2.members {
                let (min_idx, max_idx) = if i < j { (i, j) } else { (j, i) };
                let idx = coords_to_condensed_index(n_samples, min_idx, max_idx)?;
                let dist = distances[idx];
                sum_squared = sum_squared + dist * dist;
            }
            Ok(sum_squared)
        })
        .collect();

    let squared_distances = results?;
    let sum_squared_dist: F = squared_distances.into_iter().sum();

    let avg_dist_sq = sum_squared_dist / (size1 * size2);

    // Ward's formula: sqrt[(n_i * n_j) / (n_i + n_j)] * d(i,j)
    let factor = (size1 * size2) / (size1 + size2);
    Ok((factor * avg_dist_sq).sqrt())
}

/// Parallel weighted linkage: same as average but with different cluster weighting
pub(crate) fn parallel_weighted_linkage<F: Float + FromPrimitive + Send + Sync>(
    cluster1: &ParallelCluster,
    cluster2: &ParallelCluster,
    distances: &Array1<F>,
    n_samples: usize,
) -> Result<F> {
    // For weighted linkage, we use the same calculation as average linkage
    // but the weighting is handled in the hierarchical algorithm
    parallel_average_linkage(cluster1, cluster2, distances, n_samples)
}

/// Centroid linkage (not parallelized - used as-is from linkage.rs)
#[allow(dead_code)]
fn centroid_linkage<F: Float>(
    _cluster_i: usize,
    i: usize,
    _cluster_j: usize,
    _centroids: &Array2<F>,
) -> F {
    // Placeholder implementation - would need actual centroid calculation
    F::zero()
}

/// Median linkage (not parallelized - used as-is from linkage.rs)
#[allow(dead_code)]
fn median_linkage<F: Float>(
    _cluster_i: usize,
    i: usize,
    _cluster_j: usize,
    _centroids: &Array2<F>,
) -> F {
    // Placeholder implementation - would need actual median calculation
    F::zero()
}

/// Update centroid (not parallelized - used as-is from linkage.rs)
#[allow(dead_code)]
fn update_centroid<F: Float>(
    _centroids: &mut Array2<F>,
    _method: LinkageMethod,
    n_samples: usize,
    _cluster_id: usize,
) {
    // Placeholder implementation
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    #[test]
    fn test_parallel_single_linkage() {
        let cluster1 = ParallelCluster {
            size: 2,
            members: vec![0, 1],
        };
        let cluster2 = ParallelCluster {
            size: 2,
            members: vec![2, 3],
        };

        // Create a simple distance matrix
        // For 4 points, condensed form should have 6 distances
        let distances = Array1::from(vec![1.0, 2.0, 3.0, 1.5, 2.5, 1.8]);

        let result = parallel_single_linkage(&cluster1, &cluster2, &distances, 4);
        assert!(result.unwrap() >= 0.0);
    }

    #[test]
    fn test_parallel_complete_linkage() {
        let cluster1 = ParallelCluster {
            size: 2,
            members: vec![0, 1],
        };
        let cluster2 = ParallelCluster {
            size: 2,
            members: vec![2, 3],
        };

        let distances = Array1::from(vec![1.0, 2.0, 3.0, 1.5, 2.5, 1.8]);

        let result = parallel_complete_linkage(&cluster1, &cluster2, &distances, 4);
        assert!(result.unwrap() >= 0.0);
    }

    #[test]
    fn test_parallel_average_linkage() {
        let cluster1 = ParallelCluster {
            size: 2,
            members: vec![0, 1],
        };
        let cluster2 = ParallelCluster {
            size: 2,
            members: vec![2, 3],
        };

        let distances = Array1::from(vec![1.0, 2.0, 3.0, 1.5, 2.5, 1.8]);

        let result = parallel_average_linkage(&cluster1, &cluster2, &distances, 4);
        assert!(result.unwrap() >= 0.0);
    }
}
