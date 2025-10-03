//! Agglomerative clustering methods
//!
//! This module provides functions for creating flat clusters from hierarchical clustering results.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::{Float, FromPrimitive};

use crate::error::{ClusteringError, Result};

/// Cuts the dendrogram to produce a specific number of clusters
pub(crate) fn cut_tree<F: Float + FromPrimitive + PartialOrd>(
    z: &Array2<F>,
    nclusters: usize,
) -> Result<Array1<usize>> {
    let n_samples = z.shape()[0] + 1;

    // Start with each observation in its own cluster
    let mut labels = Array1::from_iter((0..n_samples).map(|_| 0));

    // Initialize the cluster memberships
    let mut clusters: Vec<Vec<usize>> = (0..n_samples).map(|i| vec![i]).collect();

    // Track which clusters are active
    let mut activeclusters: Vec<usize> = (0..n_samples).collect();

    // Number of merges needed
    let n_merges = n_samples - nclusters;

    // Perform merges
    for i in 0..n_merges {
        let cluster1 = z[[i, 0]].to_usize().unwrap();
        let cluster2 = z[[i, 1]].to_usize().unwrap();

        // Check if cluster IDs are valid
        if cluster1 >= n_samples + i || cluster2 >= n_samples + i {
            return Err(ClusteringError::ComputationError(
                "Invalid cluster indices in linkage matrix".into(),
            ));
        }

        // Create a new cluster
        let new_cluster_id = n_samples + i;
        let mut new_members = clusters[cluster1].clone();
        new_members.extend(clusters[cluster2].clone());
        clusters.push(new_members);

        // Update active clusters
        if let Some(pos) = activeclusters.iter().position(|&x| x == cluster1) {
            activeclusters.remove(pos);
        }
        if let Some(pos) = activeclusters.iter().position(|&x| x == cluster2) {
            activeclusters.remove(pos);
        }
        activeclusters.push(new_cluster_id);
    }

    // Assign cluster labels
    for (i, &cluster_id) in activeclusters.iter().enumerate() {
        for &sample in &clusters[cluster_id] {
            if sample < n_samples {
                labels[sample] = i;
            }
        }
    }

    Ok(labels)
}

/// Cuts the dendrogram at a specific distance threshold
#[allow(dead_code)]
pub fn cut_tree_by_distance<F: Float + FromPrimitive + PartialOrd>(
    z: &Array2<F>,
    threshold: F,
) -> Result<Array1<usize>> {
    let n_samples = z.shape()[0] + 1;

    // Start with each observation in its own cluster
    let mut labels = Array1::from_iter(0..n_samples);

    // Process the linkage matrix
    for i in 0..z.shape()[0] {
        let cluster1 = z[[i, 0]].to_usize().unwrap();
        let cluster2 = z[[i, 1]].to_usize().unwrap();
        let distance = z[[i, 2]];

        if distance < threshold {
            // Merge clusters if distance is below threshold
            let label1 = labels[cluster1.min(n_samples - 1)];
            let label2 = labels[cluster2.min(n_samples - 1)];

            // Replace all occurrences of label2 with label1
            for j in 0..n_samples {
                if labels[j] == label2 {
                    labels[j] = label1;
                }
            }
        }
    }

    // Renumber clusters to be 0-indexed and contiguous
    let mut unique_labels: Vec<usize> = Vec::new();
    for &label in labels.iter() {
        if !unique_labels.contains(&label) {
            unique_labels.push(label);
        }
    }

    let mut label_map: Vec<usize> = vec![0; *unique_labels.iter().max().unwrap_or(&0) + 1];
    for (i, &label) in unique_labels.iter().enumerate() {
        label_map[label] = i;
    }

    let remapped_labels = Array1::from_iter(labels.iter().map(|&l| label_map[l]));

    Ok(remapped_labels)
}

/// Forms flat clusters from a hierarchical clustering result by cutting at a specified inconsistency
///
/// # Arguments
///
/// * `z` - The linkage matrix from the `linkage` function
/// * `threshold` - The inconsistency threshold
/// * `inconsistency` - The inconsistency matrix from `inconsistent` function
///
/// # Returns
///
/// * `Result<Array1<usize>>` - Cluster assignments (0-indexed)
#[allow(dead_code)]
pub fn cut_tree_by_inconsistency<F: Float + FromPrimitive + PartialOrd>(
    z: &Array2<F>,
    threshold: F,
    inconsistency: &Array2<F>,
) -> Result<Array1<usize>> {
    let n_samples = z.shape()[0] + 1;

    // Start with each observation in its own cluster
    let mut labels = Array1::from_iter(0..n_samples);

    // Process the linkage matrix
    for i in 0..z.shape()[0] {
        let cluster1 = z[[i, 0]].to_usize().unwrap();
        let cluster2 = z[[i, 1]].to_usize().unwrap();
        let inconsistency_value = inconsistency[[i, 3]];

        if inconsistency_value < threshold {
            // Merge clusters if inconsistency is below threshold
            let label1 = labels[cluster1.min(n_samples - 1)];
            let label2 = labels[cluster2.min(n_samples - 1)];

            // Replace all occurrences of label2 with label1
            for j in 0..n_samples {
                if labels[j] == label2 {
                    labels[j] = label1;
                }
            }
        }
    }

    // Renumber clusters to be 0-indexed and contiguous
    let mut unique_labels: Vec<usize> = Vec::new();
    for &label in labels.iter() {
        if !unique_labels.contains(&label) {
            unique_labels.push(label);
        }
    }

    let mut label_map: Vec<usize> = vec![0; *unique_labels.iter().max().unwrap_or(&0) + 1];
    for (i, &label) in unique_labels.iter().enumerate() {
        label_map[label] = i;
    }

    let remapped_labels = Array1::from_iter(labels.iter().map(|&l| label_map[l]));

    Ok(remapped_labels)
}
