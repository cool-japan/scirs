//! OPTICS (Ordering Points To Identify the Clustering Structure) algorithm
//!
//! This module provides an implementation of the OPTICS clustering algorithm,
//! which is an extension of DBSCAN that addresses one of DBSCAN's major weaknesses:
//! the inability to detect meaningful clusters in data with varying densities.
//!
//! OPTICS does not explicitly produce clusters; instead, it creates an ordering
//! of points that is a compact representation of the cluster structure, along with
//! a reachability plot that can be used to extract clusters at different density levels.

use ndarray::{Array1, Array2, ArrayView2};
use num_traits::{Float, FromPrimitive};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::fmt::Debug;

use crate::error::{ClusteringError, Result};

use super::distance;
use super::DistanceMetric;

/// Point data structure for OPTICS algorithm
#[derive(Debug, Clone)]
struct OPTICSPoint {
    /// Index of the point in the original data
    #[allow(dead_code)]
    index: usize,
    /// Core distance (minimum radius required to be a core point)
    core_distance: Option<f64>,
    /// Reachability distance from previous point in the ordering
    reachability_distance: Option<f64>,
    /// Whether the point has been processed
    processed: bool,
}

/// Priority queue element for OPTICS algorithm
#[derive(Debug, Clone, PartialEq)]
struct PriorityQueueElement {
    /// Index of the point in the original data
    point_index: usize,
    /// Reachability distance from a core point
    reachability_distance: f64,
}

impl Eq for PriorityQueueElement {}

impl PartialOrd for PriorityQueueElement {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PriorityQueueElement {
    fn cmp(&self, other: &Self) -> Ordering {
        // Use reverse ordering for min-heap (smaller distances have higher priority)
        other
            .reachability_distance
            .partial_cmp(&self.reachability_distance)
            .unwrap_or(Ordering::Equal)
    }
}

/// Result of the OPTICS algorithm
#[derive(Debug, Clone)]
pub struct OPTICSResult {
    /// Ordering of points according to the OPTICS algorithm
    pub ordering: Vec<usize>,
    /// Reachability distances for each point in the ordering
    pub reachability: Vec<Option<f64>>,
    /// Core distances for each point
    pub core_distances: Vec<Option<f64>>,
    /// Predecessor points for each point in the ordering (used for visualization)
    pub predecessor: Vec<Option<usize>>,
}

/// Extracts DBSCAN-like clusters from OPTICS ordering using a specific epsilon value
///
/// # Arguments
///
/// * `optics_result` - The result from the OPTICS algorithm
/// * `eps` - The maximum distance to consider for extracting clusters
///
/// # Returns
///
/// * `Array1<i32>` - Cluster labels starting from 0, with -1 for noise points
pub fn extract_dbscan_clustering(optics_result: &OPTICSResult, eps: f64) -> Array1<i32> {
    let n_samples = optics_result.ordering.len();
    let mut labels = vec![-1; n_samples];
    let mut cluster_label = 0;

    for i in 0..n_samples {
        let point_idx = optics_result.ordering[i];
        let reachability = optics_result.reachability[i];

        // Points with reachability distance > eps are noise
        if reachability.is_none() || reachability.unwrap() > eps {
            // Could be the start of a new cluster if it's a core point
            if let Some(core_dist) = optics_result.core_distances[point_idx] {
                if core_dist <= eps {
                    // Start a new cluster
                    labels[point_idx] = cluster_label;
                    cluster_label += 1;
                }
            }
        } else {
            // Point is density-reachable from previous points in the ordering
            // Assign it to the same cluster as its predecessor
            if let Some(pred_idx) = optics_result.predecessor[point_idx] {
                if labels[pred_idx] != -1 {
                    labels[point_idx] = labels[pred_idx];
                } else {
                    // If predecessor is noise, start a new cluster
                    labels[point_idx] = cluster_label;
                    cluster_label += 1;
                }
            }
        }
    }

    Array1::from(labels)
}

/// Implements the OPTICS (Ordering Points To Identify the Clustering Structure) algorithm
///
/// # Arguments
///
/// * `data` - Input data as a 2D array (n_samples × n_features)
/// * `min_samples` - Minimum number of samples in a neighborhood to be considered a core point
/// * `max_eps` - Maximum distance to consider (defaults to infinity)
/// * `metric` - Distance metric to use (default: Euclidean)
///
/// # Returns
///
/// * `Result<OPTICSResult>` - The OPTICS ordering and associated distances
///
/// # Examples
///
/// ```ignore
/// use ndarray::{Array2, ArrayView2};
/// use scirs2_cluster::density::{optics, DistanceMetric};
///
/// // Example data with two clusters of different densities
/// let data = Array2::from_shape_vec((10, 2), vec![
///     1.0, 2.0,   // Cluster 1 (dense)
///     1.2, 1.8,   // Cluster 1
///     0.9, 1.9,   // Cluster 1
///     1.1, 2.1,   // Cluster 1
///     6.0, 8.0,   // Cluster 2 (sparse)
///     6.9, 7.5,   // Cluster 2
///     7.1, 8.2,   // Cluster 2
///     3.0, 3.0,   // Noise
///     9.0, 9.0,   // Noise
///     0.0, 10.0,  // Noise
/// ]).unwrap();
///
/// // Run OPTICS with min_samples=2
/// let result = optics(data.view(), 2, None, Some(DistanceMetric::Euclidean)).unwrap();
///
/// // Access the ordering and reachability distances
/// println!("Ordering: {:?}", result.ordering);
/// println!("Reachability: {:?}", result.reachability);
///
/// println!("Ordering: {:?}", result.ordering);
/// println!("Reachability: {:?}", result.reachability);
/// println!("Cluster assignments: {:?}", labels);
/// ```
pub fn optics<F: Float + FromPrimitive + Debug + PartialOrd>(
    data: ArrayView2<F>,
    min_samples: usize,
    max_eps: Option<F>,
    metric: Option<DistanceMetric>,
) -> Result<OPTICSResult> {
    let n_samples = data.shape()[0];

    if n_samples == 0 {
        return Err(ClusteringError::InvalidInput("Empty input data".into()));
    }

    if min_samples < 2 {
        return Err(ClusteringError::InvalidInput(
            "min_samples must be at least 2".into(),
        ));
    }

    // Convert max_eps to f64 for internal calculations
    let max_eps_f64 = match max_eps {
        Some(eps) => {
            if eps <= F::zero() {
                return Err(ClusteringError::InvalidInput(
                    "max_eps must be positive".into(),
                ));
            }
            eps.to_f64().unwrap()
        }
        None => f64::INFINITY,
    };

    // Initialize data structures
    let mut optics_points: Vec<OPTICSPoint> = (0..n_samples)
        .map(|i| OPTICSPoint {
            index: i,
            core_distance: None,
            reachability_distance: None,
            processed: false,
        })
        .collect();

    let mut ordering = Vec::with_capacity(n_samples);
    let mut reachability = Vec::with_capacity(n_samples);
    let mut predecessor = vec![None; n_samples];

    // Pre-compute pairwise distances to avoid repeated calculations
    let mut distance_matrix = Array2::<f64>::zeros((n_samples, n_samples));

    // Calculate pairwise distances
    for i in 0..n_samples {
        for j in (i + 1)..n_samples {
            let point1 = data.row(i).to_vec();
            let point2 = data.row(j).to_vec();

            let dist = match metric.unwrap_or(DistanceMetric::Euclidean) {
                DistanceMetric::Euclidean => {
                    distance::euclidean(&point1, &point2).to_f64().unwrap()
                }
                DistanceMetric::Manhattan => {
                    distance::manhattan(&point1, &point2).to_f64().unwrap()
                }
                DistanceMetric::Chebyshev => {
                    distance::chebyshev(&point1, &point2).to_f64().unwrap()
                }
                DistanceMetric::Minkowski => {
                    distance::minkowski(&point1, &point2, F::from(3.0).unwrap())
                        .to_f64()
                        .unwrap()
                }
            };

            distance_matrix[[i, j]] = dist;
            distance_matrix[[j, i]] = dist; // Symmetric matrix
        }
    }

    // Main OPTICS algorithm
    for point_idx in 0..n_samples {
        if optics_points[point_idx].processed {
            continue;
        }

        // Find neighbors of the current point within max_eps
        let neighbors = get_neighbors(point_idx, &distance_matrix, max_eps_f64);

        // Mark the point as processed
        optics_points[point_idx].processed = true;

        // Calculate core distance
        let core_distance = if neighbors.len() >= min_samples - 1 {
            // Sort distances to neighbors
            let mut distances: Vec<f64> = neighbors
                .iter()
                .map(|&n| distance_matrix[[point_idx, n]])
                .collect();
            distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

            // Core distance is the distance to the min_samples-1 th neighbor
            Some(distances[min_samples - 2])
        } else {
            None
        };

        optics_points[point_idx].core_distance = core_distance;

        // Add the point to the ordering
        ordering.push(point_idx);
        reachability.push(optics_points[point_idx].reachability_distance);

        // Continue only if this is a core point
        if let Some(core_dist) = core_distance {
            let mut seeds = BinaryHeap::new();

            // Update reachability distances for neighbors
            update_seeds(
                point_idx,
                &neighbors,
                &mut seeds,
                &mut optics_points,
                &distance_matrix,
                core_dist,
                &mut predecessor,
            );

            // Process the priority queue
            while let Some(element) = seeds.pop() {
                let current_idx = element.point_index;

                if optics_points[current_idx].processed {
                    continue;
                }

                // Find neighbors of current point
                let current_neighbors = get_neighbors(current_idx, &distance_matrix, max_eps_f64);

                // Mark point as processed
                optics_points[current_idx].processed = true;

                // Add to ordering
                ordering.push(current_idx);
                reachability.push(Some(element.reachability_distance));

                // Calculate core distance for current point
                let current_core_dist = if current_neighbors.len() >= min_samples - 1 {
                    let mut distances: Vec<f64> = current_neighbors
                        .iter()
                        .map(|&n| distance_matrix[[current_idx, n]])
                        .collect();
                    distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

                    Some(distances[min_samples - 2])
                } else {
                    None
                };

                optics_points[current_idx].core_distance = current_core_dist;

                // If it's a core point, update its neighbors
                if let Some(core_dist) = current_core_dist {
                    update_seeds(
                        current_idx,
                        &current_neighbors,
                        &mut seeds,
                        &mut optics_points,
                        &distance_matrix,
                        core_dist,
                        &mut predecessor,
                    );
                }
            }
        }
    }

    // Collect core distances in the original order
    let core_distances = optics_points.iter().map(|p| p.core_distance).collect();

    Ok(OPTICSResult {
        ordering,
        reachability,
        core_distances,
        predecessor,
    })
}

/// Get neighbors of a point within the specified epsilon radius
fn get_neighbors(point_idx: usize, distance_matrix: &Array2<f64>, max_eps: f64) -> Vec<usize> {
    let n_samples = distance_matrix.shape()[0];
    let mut neighbors = Vec::new();

    for j in 0..n_samples {
        if point_idx != j && distance_matrix[[point_idx, j]] <= max_eps {
            neighbors.push(j);
        }
    }

    neighbors
}

/// Update seeds with new reachability distances
fn update_seeds(
    point_idx: usize,
    neighbors: &[usize],
    seeds: &mut BinaryHeap<PriorityQueueElement>,
    optics_points: &mut [OPTICSPoint],
    distance_matrix: &Array2<f64>,
    core_distance: f64,
    predecessor: &mut [Option<usize>],
) {
    for &neighbor_idx in neighbors {
        if optics_points[neighbor_idx].processed {
            continue;
        }

        // Calculate new reachability distance
        let new_reachability_distance =
            core_distance.max(distance_matrix[[point_idx, neighbor_idx]]);

        // Update reachability distance if needed
        match optics_points[neighbor_idx].reachability_distance {
            None => {
                // Point has not been seen before
                optics_points[neighbor_idx].reachability_distance = Some(new_reachability_distance);
                predecessor[neighbor_idx] = Some(point_idx);

                seeds.push(PriorityQueueElement {
                    point_index: neighbor_idx,
                    reachability_distance: new_reachability_distance,
                });
            }
            Some(old_distance) => {
                // Point has been seen before, update if new distance is smaller
                if new_reachability_distance < old_distance {
                    optics_points[neighbor_idx].reachability_distance =
                        Some(new_reachability_distance);
                    predecessor[neighbor_idx] = Some(point_idx);

                    // Since BinaryHeap doesn't support decreasing key, we add a new element
                    // The old one will be skipped when processed since the point will be marked as processed
                    seeds.push(PriorityQueueElement {
                        point_index: neighbor_idx,
                        reachability_distance: new_reachability_distance,
                    });
                }
            }
        }
    }
}

/// Extract DBSCAN-like clusters from OPTICS ordering using a xi threshold for steepness-based clustering
///
/// # Arguments
///
/// * `optics_result` - The result from the OPTICS algorithm
/// * `xi` - The steepness threshold (between 0 and 1) for detecting cluster boundaries
/// * `min_cluster_size` - Minimum number of points needed for a cluster
///
/// # Returns
///
/// * `Array1<i32>` - Cluster labels starting from 0, with -1 for noise points
pub fn extract_xi_clusters(
    optics_result: &OPTICSResult,
    xi: f64,
    min_cluster_size: usize,
) -> Result<Array1<i32>> {
    if xi <= 0.0 || xi >= 1.0 {
        return Err(ClusteringError::InvalidInput(
            "xi must be between 0 and 1 (exclusive)".into(),
        ));
    }

    if min_cluster_size < 2 {
        return Err(ClusteringError::InvalidInput(
            "min_cluster_size must be at least 2".into(),
        ));
    }

    let n_samples = optics_result.ordering.len();

    // Initialize all points as noise
    let mut labels = vec![-1; n_samples];

    // Get reachability distances, handling None values
    let reachability: Vec<f64> = optics_result
        .reachability
        .iter()
        .map(|&r| r.unwrap_or(f64::INFINITY))
        .collect();

    // Find steep up and down areas
    let steep_areas = find_steep_areas(&reachability, xi);

    // Extract clusters from steep areas
    let mut cluster_id = 0;

    // Process each pair of steep down and steep up areas
    for i in 0..steep_areas.len() {
        if steep_areas[i].0 == "steep_down" {
            // Find the matching steep up area
            let mut found_match = false;
            for j in i + 1..steep_areas.len() {
                if steep_areas[j].0 == "steep_up" {
                    // We found a potential cluster
                    let start_idx = steep_areas[i].1;
                    let end_idx = steep_areas[j].1;

                    // Check cluster size
                    if end_idx - start_idx + 1 >= min_cluster_size {
                        // Valid cluster, assign points to it
                        for k in start_idx..=end_idx {
                            let point_idx = optics_result.ordering[k];
                            labels[point_idx] = cluster_id;
                        }
                        cluster_id += 1;
                    }

                    found_match = true;
                    break;
                }
            }

            if !found_match && i + 1 < steep_areas.len() {
                // No matching steep up area, try with next steep area
                continue;
            }
        }
    }

    Ok(Array1::from(labels))
}

/// Find steep up and down areas in reachability plot
fn find_steep_areas(reachability: &[f64], xi: f64) -> Vec<(String, usize)> {
    let mut steep_areas = Vec::new();
    let n = reachability.len();

    // Skip first point since it doesn't have a predecessor
    for i in 1..n {
        let prev = reachability[i - 1];
        let curr = reachability[i];

        // Skip infinity values
        if prev.is_infinite() || curr.is_infinite() {
            continue;
        }

        // Calculate steepness
        let steepness = (prev - curr) / prev.max(curr);

        if steepness > xi {
            // Steep down area (decreasing reachability)
            steep_areas.push(("steep_down".to_string(), i));
        } else if steepness < -xi {
            // Steep up area (increasing reachability)
            steep_areas.push(("steep_up".to_string(), i));
        }
    }

    steep_areas
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_optics_basic() {
        // Sample data with two clusters
        let data = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 2.0, // Cluster 1
                1.5, 1.8, // Cluster 1
                1.3, 1.9, // Cluster 1
                5.0, 7.0, // Cluster 2
                5.1, 6.8, // Cluster 2
                5.2, 7.1, // Cluster 2
                0.0, 10.0, // Noise
                10.0, 0.0, // Noise
            ],
        )
        .unwrap();

        // Run OPTICS with min_samples=2
        let result = optics(data.view(), 2, None, Some(DistanceMetric::Euclidean)).unwrap();

        // Check that the ordering contains all points
        assert_eq!(result.ordering.len(), 8);
        assert_eq!(result.reachability.len(), 8);

        // Extract clusters with eps = 0.8 (similar to DBSCAN)
        let labels = extract_dbscan_clustering(&result, 0.8);

        // Check dimensions of results
        assert_eq!(labels.len(), 8);

        // Count number of clusters (excluding noise)
        let num_clusters = labels.iter().filter(|&&x| x >= 0).count();
        assert!(num_clusters > 0, "There should be at least one cluster");
    }
}
