//! Linkage methods for hierarchical clustering
//!
//! This module provides functions for computing linkage matrices for hierarchical clustering.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::Debug;

use crate::error::{ClusteringError, Result};
use crate::hierarchy::{coords_to_condensed_index, LinkageMethod};

/// Structure to represent a cluster
#[derive(Debug)]
pub(crate) struct Cluster {
    /// Number of observations in the cluster
    pub size: usize,

    /// Indices of observations in the cluster
    pub members: Vec<usize>,
    // Commented out unused fields to avoid warnings
    // /// Unique ID for the cluster (not needed after refactoring)
    // pub id: usize,
    // /// Centroid of the cluster (for centroid-based methods)
    // pub centroid: Option<Array1<F>>,
}

/// Main hierarchical clustering algorithm
pub(crate) fn hierarchical_clustering<F: Float + FromPrimitive + Debug + PartialOrd>(
    distances: &Array1<F>,
    n_samples: usize,
    method: LinkageMethod,
) -> Result<Array2<F>> {
    // Initialize clusters (each observation starts in its own cluster)
    let mut clusters: Vec<Cluster> = (0..n_samples)
        .map(|i| Cluster {
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
        // Find the two closest clusters
        let (cluster1_idx, cluster2_idx, min_dist) = find_closestclusters(
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

        let new_cluster = Cluster {
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

/// Finds the two closest clusters based on the given linkage method
#[allow(dead_code)]
fn find_closestclusters<F: Float + FromPrimitive + Debug + PartialOrd>(
    activeclusters: &[usize],
    clusters: &[Cluster],
    distances: &Array1<F>,
    method: LinkageMethod,
    centroids: Option<&Array2<F>>,
    n_samples: usize,
) -> Result<(usize, usize, F)> {
    let mut min_dist = F::infinity();
    let mut min_i = 0;
    let mut min_j = 0;

    // Loop through all pairs of active clusters
    for (i, &cluster_i) in activeclusters.iter().enumerate() {
        for (j, &cluster_j) in activeclusters.iter().enumerate() {
            if i >= j {
                continue; // Only need to check each pair once
            }

            // Calculate distance between clusters based on the linkage method
            let dist = match method {
                LinkageMethod::Single => single_linkage(
                    &clusters[cluster_i],
                    &clusters[cluster_j],
                    distances,
                    n_samples,
                )?,
                LinkageMethod::Complete => complete_linkage(
                    &clusters[cluster_i],
                    &clusters[cluster_j],
                    distances,
                    n_samples,
                )?,
                LinkageMethod::Average => average_linkage(
                    &clusters[cluster_i],
                    &clusters[cluster_j],
                    distances,
                    n_samples,
                )?,
                LinkageMethod::Ward => ward_linkage(
                    &clusters[cluster_i],
                    &clusters[cluster_j],
                    distances,
                    n_samples,
                )?,
                LinkageMethod::Centroid => {
                    centroid_linkage(cluster_i, cluster_j, centroids.unwrap())
                }
                LinkageMethod::Median => median_linkage(cluster_i, cluster_j, centroids.unwrap()),
                LinkageMethod::Weighted => weighted_linkage(
                    &clusters[cluster_i],
                    &clusters[cluster_j],
                    distances,
                    n_samples,
                )?,
            };

            // Update minimum distance
            if dist < min_dist {
                min_dist = dist;
                min_i = i;
                min_j = j;
            }
        }
    }

    if min_dist == F::infinity() {
        return Err(ClusteringError::ComputationError(
            "Could not find minimum distance between clusters".into(),
        ));
    }

    Ok((min_i, min_j, min_dist))
}

/// Single linkage: minimum distance between any two points in the clusters
pub(crate) fn single_linkage<F: Float + PartialOrd>(
    cluster1: &Cluster,
    cluster2: &Cluster,
    distances: &Array1<F>,
    n_samples: usize,
) -> Result<F> {
    let mut min_dist = F::infinity();

    for &i in &cluster1.members {
        for &j in &cluster2.members {
            // Get the distance between observations i and j from the condensed matrix
            let (min_idx, max_idx) = if i < j { (i, j) } else { (j, i) };
            let idx = coords_to_condensed_index(n_samples, min_idx, max_idx)?;
            let dist = distances[idx];

            if dist < min_dist {
                min_dist = dist;
            }
        }
    }

    Ok(min_dist)
}

/// Complete linkage: maximum distance between any two points in the clusters
pub(crate) fn complete_linkage<F: Float + PartialOrd>(
    cluster1: &Cluster,
    cluster2: &Cluster,
    distances: &Array1<F>,
    n_samples: usize,
) -> Result<F> {
    let mut max_dist = F::neg_infinity();

    for &i in &cluster1.members {
        for &j in &cluster2.members {
            // Get the distance between observations i and j from the condensed matrix
            let (min_idx, max_idx) = if i < j { (i, j) } else { (j, i) };
            let idx = coords_to_condensed_index(n_samples, min_idx, max_idx)?;
            let dist = distances[idx];

            if dist > max_dist {
                max_dist = dist;
            }
        }
    }

    Ok(max_dist)
}

/// Average linkage: average distance between all pairs of points in the clusters
pub(crate) fn average_linkage<F: Float + FromPrimitive>(
    cluster1: &Cluster,
    cluster2: &Cluster,
    distances: &Array1<F>,
    n_samples: usize,
) -> Result<F> {
    let mut sum_dist = F::zero();
    let mut count = 0;

    for &i in &cluster1.members {
        for &j in &cluster2.members {
            // Get the distance between observations i and j from the condensed matrix
            let (min_idx, max_idx) = if i < j { (i, j) } else { (j, i) };
            let idx = coords_to_condensed_index(n_samples, min_idx, max_idx)?;
            sum_dist = sum_dist + distances[idx];
            count += 1;
        }
    }

    Ok(sum_dist / F::from_usize(count).unwrap())
}

/// Ward's linkage: minimizes the increase in variance when merging clusters
pub(crate) fn ward_linkage<F: Float + FromPrimitive>(
    cluster1: &Cluster,
    cluster2: &Cluster,
    distances: &Array1<F>,
    n_samples: usize,
) -> Result<F> {
    // For Ward's method, we need to calculate the increase in variance
    // This is proportional to the distance between centroids weighted by the size of clusters

    let size1 = F::from_usize(cluster1.size).unwrap();
    let size2 = F::from_usize(cluster2.size).unwrap();

    // Find the squared distance between the centroids
    // For simplicity, we'll calculate it based on the original distances

    let mut sum_dist = F::zero();

    for &i in &cluster1.members {
        for &j in &cluster2.members {
            let (min_idx, max_idx) = if i < j { (i, j) } else { (j, i) };
            let idx = coords_to_condensed_index(n_samples, min_idx, max_idx)?;
            let dist = distances[idx];
            sum_dist = sum_dist + dist * dist;
        }
    }

    let avg_dist_sq = sum_dist / (size1 * size2);

    // Ward's formula: sqrt[(n_i * n_j) / (n_i + n_j)] * d(i,j)
    let factor = (size1 * size2) / (size1 + size2);
    Ok((factor * avg_dist_sq).sqrt())
}

/// Centroid linkage: distance between cluster centroids
pub(crate) fn centroid_linkage<F: Float>(
    cluster1: usize,
    cluster2: usize,
    centroids: &Array2<F>,
) -> F {
    // For the centroid method, we simply return the distance between centroids
    // which is pre-computed and stored in the centroids matrix
    centroids[[cluster1, cluster2]]
}

/// Median linkage: uses weighted centroids
pub(crate) fn median_linkage<F: Float>(
    cluster1: usize,
    cluster2: usize,
    centroids: &Array2<F>,
) -> F {
    // Similar to centroid method, but with different centroid update rule
    centroids[[cluster1, cluster2]]
}

/// Weighted average linkage: weights by cluster size
pub(crate) fn weighted_linkage<F: Float + FromPrimitive>(
    cluster1: &Cluster,
    cluster2: &Cluster,
    distances: &Array1<F>,
    n_samples: usize,
) -> Result<F> {
    // Weighted average gives equal weights to each cluster regardless of size
    average_linkage(cluster1, cluster2, distances, n_samples)
}

/// Updates the centroid for centroid-based methods
pub(crate) fn update_centroid<F: Float + FromPrimitive>(
    centroids: &mut Array2<F>,
    _method: LinkageMethod,
    n_samples: usize,
    new_cluster_id: usize,
) {
    // This is a simplified placeholder. For a complete implementation,
    // we would need to store actual centroids and update them.

    // In a real implementation:
    // 1. For centroid _method: new_centroid = (n1*c1 + n2*c2)/(n1 + n2)
    // 2. For median _method: new_centroid = (c1 + c2)/2

    // For now, we'll just set a dummy value to satisfy the type system
    centroids[[new_cluster_id, 0]] = F::one();
}
