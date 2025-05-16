//! Silhouette coefficient for evaluating clustering quality

use ndarray::{Array1, ArrayView1, ArrayView2};
use num_traits::{Float, FromPrimitive};

use crate::error::{ClusteringError, Result};
use crate::vq::euclidean_distance;

/// Computes the Silhouette coefficient for each sample
///
/// The Silhouette coefficient is a measure of how similar an object is to its own
/// cluster (cohesion) compared to other clusters (separation). The Silhouette ranges
/// from -1 to +1, where a high value indicates that the object is well matched to
/// its own cluster and poorly matched to neighboring clusters.
///
/// # Arguments
///
/// * `data` - Input data (n_samples × n_features)
/// * `labels` - Cluster labels for each sample (n_samples,)
///
/// # Returns
///
/// * Array of silhouette scores for each sample (n_samples,)
pub fn silhouette_samples<F>(data: ArrayView2<F>, labels: ArrayView1<i32>) -> Result<Array1<F>>
where
    F: Float + FromPrimitive,
{
    let n_samples = data.shape()[0];

    if n_samples != labels.len() {
        return Err(ClusteringError::InvalidInput(
            "Data and labels must have the same number of samples".to_string(),
        ));
    }

    // Find the unique labels (excluding noise points with label -1)
    let mut unique_labels = Vec::new();
    for &label in labels.iter() {
        if label >= 0 && !unique_labels.contains(&label) {
            unique_labels.push(label);
        }
    }

    let n_clusters = unique_labels.len();

    if n_clusters < 2 || n_clusters >= n_samples {
        return Err(ClusteringError::InvalidInput(
            "Silhouette score requires at least 2 clusters and fewer clusters than samples"
                .to_string(),
        ));
    }

    let mut silhouette_scores = Array1::<F>::zeros(n_samples);

    // Compute intra-cluster and inter-cluster distances
    for i in 0..n_samples {
        let label_i = labels[i];

        // Skip noise points (label -1)
        if label_i < 0 {
            continue;
        }

        let sample_i = data.slice(ndarray::s![i, ..]);

        // a(i): Mean distance to other samples in the same cluster
        let mut intra_dist_sum = F::zero();
        let mut intra_count = 0;

        // b(i): Mean distance to samples in the nearest cluster
        let mut inter_cluster_dists = Vec::new();

        for &cluster_label in &unique_labels {
            if cluster_label == label_i {
                // Same cluster - compute intra-cluster distance
                for j in 0..n_samples {
                    if i != j && labels[j] == label_i {
                        let sample_j = data.slice(ndarray::s![j, ..]);
                        intra_dist_sum = intra_dist_sum + euclidean_distance(sample_i, sample_j);
                        intra_count += 1;
                    }
                }
            } else {
                // Different cluster - compute mean distance to this cluster
                let mut cluster_dist_sum = F::zero();
                let mut cluster_count = 0;

                for j in 0..n_samples {
                    if labels[j] == cluster_label {
                        let sample_j = data.slice(ndarray::s![j, ..]);
                        cluster_dist_sum =
                            cluster_dist_sum + euclidean_distance(sample_i, sample_j);
                        cluster_count += 1;
                    }
                }

                if cluster_count > 0 {
                    let mean_dist = cluster_dist_sum / F::from(cluster_count).unwrap();
                    inter_cluster_dists.push(mean_dist);
                }
            }
        }

        // Calculate a(i)
        let a_i = if intra_count > 0 {
            intra_dist_sum / F::from(intra_count).unwrap()
        } else {
            F::zero()
        };

        // Calculate b(i) - minimum mean distance to other clusters
        let b_i = if !inter_cluster_dists.is_empty() {
            *inter_cluster_dists
                .iter()
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap()
        } else {
            F::infinity()
        };

        // Calculate silhouette score
        if a_i.is_zero() && b_i.is_zero() {
            silhouette_scores[i] = F::zero();
        } else {
            let max_val = if a_i > b_i { a_i } else { b_i };
            silhouette_scores[i] = (b_i - a_i) / max_val;
        }
    }

    Ok(silhouette_scores)
}

/// Computes the mean Silhouette coefficient of all samples
///
/// # Arguments
///
/// * `data` - Input data (n_samples × n_features)
/// * `labels` - Cluster labels for each sample (n_samples,)
///
/// # Returns
///
/// * Mean silhouette score across all samples
pub fn silhouette_score<F>(data: ArrayView2<F>, labels: ArrayView1<i32>) -> Result<F>
where
    F: Float + FromPrimitive,
{
    let sample_scores = silhouette_samples(data, labels)?;
    let n_samples = sample_scores.len();

    if n_samples == 0 {
        return Ok(F::zero());
    }

    let sum: F = sample_scores.iter().fold(F::zero(), |acc, &val| acc + val);
    Ok(sum / F::from(n_samples).unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_silhouette_samples() {
        // Create simple clustered data
        let data = array![[1.0, 1.0], [1.5, 1.5], [5.0, 5.0], [5.5, 5.5],];

        // Two clear clusters
        let labels = array![0, 0, 1, 1];

        let scores = silhouette_samples(data.view(), labels.view()).unwrap();

        // All scores should be positive for well-separated clusters
        for score in scores.iter() {
            assert!(*score > 0.0);
        }
    }

    #[test]
    fn test_silhouette_score() {
        let data = array![[1.0, 1.0], [1.5, 1.5], [5.0, 5.0], [5.5, 5.5],];

        let labels = array![0, 0, 1, 1];

        let score = silhouette_score(data.view(), labels.view()).unwrap();

        assert!(score > 0.0);
        assert!(score <= 1.0);
    }

    #[test]
    fn test_single_cluster_error() {
        let data = array![[1.0, 1.0], [1.5, 1.5],];

        let labels = array![0, 0];

        let result = silhouette_score(data.view(), labels.view());
        assert!(result.is_err());
    }
}
