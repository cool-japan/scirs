//! Streaming and memory-efficient clustering algorithms
//!
//! This module provides implementations of clustering algorithms that can handle
//! large datasets that don't fit entirely in memory, using streaming and
//! progressive processing techniques.

use ndarray::{Array1, Array2, ArrayView2};
use num_traits::{Float, FromPrimitive};
use std::collections::VecDeque;
use std::fmt::Debug;

use crate::error::{ClusteringError, Result};
use crate::vq::euclidean_distance;

/// Configuration for streaming clustering algorithms
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Maximum number of samples to keep in memory at once
    pub max_memory_samples: usize,
    /// Batch size for processing chunks of data
    pub batch_size: usize,
    /// Number of cluster centers to maintain
    pub n_centers: usize,
    /// Convergence threshold for iterative algorithms
    pub tolerance: f64,
    /// Maximum number of iterations
    pub max_iterations: usize,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            max_memory_samples: 10000,
            batch_size: 1000,
            n_centers: 10,
            tolerance: 1e-4,
            max_iterations: 100,
        }
    }
}

/// Streaming K-means clustering for large datasets
///
/// This implementation processes data in chunks and maintains a fixed number
/// of cluster centers, updating them incrementally as new data arrives.
pub struct StreamingKMeans<F: Float> {
    config: StreamingConfig,
    centers: Option<Array2<F>>,
    weights: Option<Array1<F>>,
    n_samples_processed: usize,
    initialized: bool,
}

impl<F: Float + FromPrimitive + Debug> StreamingKMeans<F> {
    /// Create a new streaming K-means instance
    pub fn new(config: StreamingConfig) -> Self {
        Self {
            config,
            centers: None,
            weights: None,
            n_samples_processed: 0,
            initialized: false,
        }
    }

    /// Initialize the clustering with the first batch of data
    pub fn initialize(&mut self, data: ArrayView2<F>) -> Result<()> {
        let n_samples = data.shape()[0];
        let n_features = data.shape()[1];

        if n_samples == 0 {
            return Err(ClusteringError::InvalidInput(
                "Cannot initialize with empty data".into(),
            ));
        }

        let k = self.config.n_centers.min(n_samples);

        // Initialize centers using K-means++ method
        let mut centers = Array2::zeros((k, n_features));
        let weights = Array1::ones(k);

        // Choose first center randomly
        let first_center_idx = 0; // For deterministic behavior, choose first point
        centers.row_mut(0).assign(&data.row(first_center_idx));

        // Choose remaining centers using K-means++ initialization
        for i in 1..k {
            let mut distances = Array1::zeros(n_samples);
            let mut total_distance = F::zero();

            // Calculate distances to nearest existing center
            for j in 0..n_samples {
                let mut min_dist = F::infinity();
                for center_idx in 0..i {
                    let dist = euclidean_distance(data.row(j), centers.row(center_idx));
                    if dist < min_dist {
                        min_dist = dist;
                    }
                }
                distances[j] = min_dist * min_dist; // Squared distance for K-means++
                total_distance = total_distance + distances[j];
            }

            // Choose next center with probability proportional to squared distance
            let mut cumsum = F::zero();
            let target = total_distance * F::from(0.5).unwrap(); // Simplified selection

            for j in 0..n_samples {
                cumsum = cumsum + distances[j];
                if cumsum >= target {
                    centers.row_mut(i).assign(&data.row(j));
                    break;
                }
            }
        }

        self.centers = Some(centers);
        self.weights = Some(weights);
        self.n_samples_processed = n_samples;
        self.initialized = true;

        Ok(())
    }

    /// Process a new batch of data and update cluster centers
    pub fn partial_fit(&mut self, data: ArrayView2<F>) -> Result<()> {
        if !self.initialized {
            return self.initialize(data);
        }

        let n_samples = data.shape()[0];
        if n_samples == 0 {
            return Ok(());
        }

        let centers = self.centers.as_mut().unwrap();
        let weights = self.weights.as_mut().unwrap();

        // Assign each point to the nearest center and update centers
        for i in 0..n_samples {
            let point = data.row(i);

            // Find nearest center
            let mut min_dist = F::infinity();
            let mut nearest_center = 0;

            for j in 0..centers.shape()[0] {
                let dist = euclidean_distance(point, centers.row(j));
                if dist < min_dist {
                    min_dist = dist;
                    nearest_center = j;
                }
            }

            // Update center using online mean update
            let weight = weights[nearest_center];
            let new_weight = weight + F::one();
            let learning_rate = F::one() / new_weight;

            // Update center: new_center = old_center + lr * (point - old_center)
            let mut center_row = centers.row_mut(nearest_center);
            for k in 0..center_row.len() {
                let diff = point[k] - center_row[k];
                center_row[k] = center_row[k] + learning_rate * diff;
            }

            weights[nearest_center] = new_weight;
        }

        self.n_samples_processed += n_samples;
        Ok(())
    }

    /// Get the current cluster centers
    pub fn cluster_centers(&self) -> Option<&Array2<F>> {
        self.centers.as_ref()
    }

    /// Predict cluster assignments for new data
    pub fn predict(&self, data: ArrayView2<F>) -> Result<Array1<usize>> {
        if !self.initialized {
            return Err(ClusteringError::InvalidInput(
                "Model must be initialized before prediction".into(),
            ));
        }

        let centers = self.centers.as_ref().unwrap();
        let n_samples = data.shape()[0];
        let mut labels = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let point = data.row(i);
            let mut min_dist = F::infinity();
            let mut nearest_center = 0;

            for j in 0..centers.shape()[0] {
                let dist = euclidean_distance(point, centers.row(j));
                if dist < min_dist {
                    min_dist = dist;
                    nearest_center = j;
                }
            }

            labels[i] = nearest_center;
        }

        Ok(labels)
    }

    /// Get the number of samples processed so far
    pub fn n_samples_seen(&self) -> usize {
        self.n_samples_processed
    }
}

/// Progressive hierarchical clustering for large datasets
///
/// This implementation builds a hierarchy incrementally by processing data
/// in chunks and maintaining a compressed representation of the clustering.
pub struct ProgressiveHierarchical<F: Float> {
    #[allow(dead_code)]
    config: StreamingConfig,
    representative_points: VecDeque<Array1<F>>,
    cluster_sizes: VecDeque<usize>,
    max_representatives: usize,
}

impl<F: Float + FromPrimitive + Debug> ProgressiveHierarchical<F> {
    /// Create a new progressive hierarchical clustering instance
    pub fn new(config: StreamingConfig) -> Self {
        let max_representatives = config.max_memory_samples / 10; // Keep 10% as representatives

        Self {
            config,
            representative_points: VecDeque::new(),
            cluster_sizes: VecDeque::new(),
            max_representatives,
        }
    }

    /// Process a new batch of data
    pub fn partial_fit(&mut self, data: ArrayView2<F>) -> Result<()> {
        let n_samples = data.shape()[0];
        if n_samples == 0 {
            return Ok(());
        }

        // If this is the first batch, just add some representative points
        if self.representative_points.is_empty() {
            let step_size = (n_samples / self.max_representatives.min(n_samples)).max(1);

            for i in (0..n_samples).step_by(step_size) {
                self.representative_points.push_back(data.row(i).to_owned());
                self.cluster_sizes.push_back(1);

                if self.representative_points.len() >= self.max_representatives {
                    break;
                }
            }
            return Ok(());
        }

        // For subsequent batches, merge new points with existing representatives
        let mut new_representatives = Vec::new();
        let mut new_sizes = Vec::new();

        // Process new data points
        for i in 0..n_samples {
            let point = data.row(i);

            // Find closest representative
            let mut min_dist = F::infinity();
            let mut closest_idx = 0;

            for (j, repr) in self.representative_points.iter().enumerate() {
                let dist = euclidean_distance(point, repr.view());
                if dist < min_dist {
                    min_dist = dist;
                    closest_idx = j;
                }
            }

            // Merge with closest representative or create new one
            let threshold = F::from(0.1).unwrap(); // Distance threshold for merging

            if min_dist < threshold && closest_idx < self.representative_points.len() {
                // Merge with existing representative
                let old_size = self.cluster_sizes[closest_idx];
                let new_size = old_size + 1;
                let weight = F::from(old_size).unwrap() / F::from(new_size).unwrap();

                // Update representative as weighted average
                let mut repr = self.representative_points[closest_idx].clone();
                for k in 0..repr.len() {
                    repr[k] = weight * repr[k] + (F::one() - weight) * point[k];
                }

                new_representatives.push(repr);
                new_sizes.push(new_size);
            } else {
                // Create new representative
                new_representatives.push(point.to_owned());
                new_sizes.push(1);
            }
        }

        // Replace old representatives with updated ones
        self.representative_points.clear();
        self.cluster_sizes.clear();

        for (repr, size) in new_representatives.into_iter().zip(new_sizes.into_iter()) {
            self.representative_points.push_back(repr);
            self.cluster_sizes.push_back(size);
        }

        // If we have too many representatives, compress by merging similar ones
        if self.representative_points.len() > self.max_representatives {
            self.compress_representatives()?;
        }

        Ok(())
    }

    /// Compress the representation by merging similar representative points
    fn compress_representatives(&mut self) -> Result<()> {
        let _n_repr = self.representative_points.len();
        let target_size = self.max_representatives * 3 / 4; // Reduce to 75% of max

        while self.representative_points.len() > target_size {
            // Find the two closest representatives to merge
            let mut min_dist = F::infinity();
            let mut merge_i = 0;
            let mut merge_j = 1;

            for i in 0..self.representative_points.len() {
                for j in (i + 1)..self.representative_points.len() {
                    let dist = euclidean_distance(
                        self.representative_points[i].view(),
                        self.representative_points[j].view(),
                    );
                    if dist < min_dist {
                        min_dist = dist;
                        merge_i = i;
                        merge_j = j;
                    }
                }
            }

            // Merge the two closest representatives
            let size_i = self.cluster_sizes[merge_i];
            let size_j = self.cluster_sizes[merge_j];
            let total_size = size_i + size_j;

            let weight_i = F::from(size_i).unwrap() / F::from(total_size).unwrap();
            let weight_j = F::from(size_j).unwrap() / F::from(total_size).unwrap();

            // Create merged representative
            let repr_i = &self.representative_points[merge_i];
            let repr_j = &self.representative_points[merge_j];
            let mut merged_repr = Array1::zeros(repr_i.len());

            for k in 0..merged_repr.len() {
                merged_repr[k] = weight_i * repr_i[k] + weight_j * repr_j[k];
            }

            // Remove the two old representatives (remove larger index first)
            if merge_j > merge_i {
                self.representative_points.remove(merge_j);
                self.cluster_sizes.remove(merge_j);
                self.representative_points.remove(merge_i);
                self.cluster_sizes.remove(merge_i);
            } else {
                self.representative_points.remove(merge_i);
                self.cluster_sizes.remove(merge_i);
                self.representative_points.remove(merge_j);
                self.cluster_sizes.remove(merge_j);
            }

            // Add the merged representative
            self.representative_points.push_back(merged_repr);
            self.cluster_sizes.push_back(total_size);
        }

        Ok(())
    }

    /// Get the current representative points
    pub fn get_representatives(&self) -> (Vec<Array1<F>>, Vec<usize>) {
        (
            self.representative_points.iter().cloned().collect(),
            self.cluster_sizes.iter().cloned().collect(),
        )
    }

    /// Get the number of representative points
    pub fn n_representatives(&self) -> usize {
        self.representative_points.len()
    }
}

/// Memory-efficient distance matrix computation
///
/// Computes distances between points in chunks to avoid storing the full
/// distance matrix in memory.
pub struct ChunkedDistanceMatrix<F: Float> {
    chunk_size: usize,
    n_samples: usize,
    _phantom: std::marker::PhantomData<F>,
}

impl<F: Float + FromPrimitive> ChunkedDistanceMatrix<F> {
    /// Create a new chunked distance matrix
    pub fn new(n_samples: usize, max_memory_mb: usize) -> Self {
        // Estimate chunk size based on memory limit
        let memory_per_float = std::mem::size_of::<F>();
        let max_elements = (max_memory_mb * 1024 * 1024) / memory_per_float;
        let chunk_size = (max_elements / n_samples).max(1).min(n_samples);

        Self {
            chunk_size,
            n_samples,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Process distances in chunks and apply a function to each chunk
    pub fn process_chunks<Func>(&self, data: ArrayView2<F>, mut processor: Func) -> Result<()>
    where
        Func: FnMut(usize, usize, F) -> Result<()>,
    {
        for i in (0..self.n_samples).step_by(self.chunk_size) {
            let end_i = (i + self.chunk_size).min(self.n_samples);

            for j in (i..self.n_samples).step_by(self.chunk_size) {
                let end_j = (j + self.chunk_size).min(self.n_samples);

                // Process this chunk of distances
                for row in i..end_i {
                    for col in j.max(row + 1)..end_j {
                        let dist = euclidean_distance(data.row(row), data.row(col));
                        processor(row, col, dist)?;
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_streaming_kmeans() {
        let config = StreamingConfig {
            max_memory_samples: 100,
            batch_size: 10,
            n_centers: 2,
            tolerance: 1e-4,
            max_iterations: 10,
        };

        let mut streaming_kmeans = StreamingKMeans::new(config);

        // First batch
        let batch1 =
            Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 0.1, 0.1, 1.0, 1.0, 1.1, 1.1]).unwrap();

        streaming_kmeans.partial_fit(batch1.view()).unwrap();
        assert!(streaming_kmeans.cluster_centers().is_some());

        // Second batch
        let batch2 =
            Array2::from_shape_vec((4, 2), vec![0.2, 0.2, 0.0, 0.1, 1.2, 1.0, 1.0, 1.2]).unwrap();

        streaming_kmeans.partial_fit(batch2.view()).unwrap();

        // Test prediction
        let test_data = Array2::from_shape_vec((2, 2), vec![0.05, 0.05, 1.05, 1.05]).unwrap();

        let labels = streaming_kmeans.predict(test_data.view()).unwrap();
        assert_eq!(labels.len(), 2);

        // Points should be assigned to different clusters
        assert_ne!(labels[0], labels[1]);
    }

    #[test]
    fn test_progressive_hierarchical() {
        let config = StreamingConfig::default();
        let mut progressive = ProgressiveHierarchical::new(config);

        // Process first batch
        let batch1 = Array2::from_shape_vec(
            (6, 2),
            vec![0.0, 0.0, 0.1, 0.1, 0.2, 0.2, 5.0, 5.0, 5.1, 5.1, 5.2, 5.2],
        )
        .unwrap();

        progressive.partial_fit(batch1.view()).unwrap();
        let (representatives, sizes) = progressive.get_representatives();

        assert!(!representatives.is_empty());
        assert_eq!(representatives.len(), sizes.len());
        assert!(progressive.n_representatives() > 0);
    }

    #[test]
    fn test_chunked_distance_matrix() {
        let data =
            Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();

        let chunked_matrix = ChunkedDistanceMatrix::new(4, 1); // 1 MB limit
        let mut distance_count = 0;

        chunked_matrix
            .process_chunks(data.view(), |i, j, dist| {
                assert!(i < j);
                assert!(dist >= 0.0);
                distance_count += 1;
                Ok(())
            })
            .unwrap();

        // Should process 6 distances for 4 points: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
        assert_eq!(distance_count, 6);
    }

    #[test]
    fn test_streaming_config_default() {
        let config = StreamingConfig::default();
        assert_eq!(config.max_memory_samples, 10000);
        assert_eq!(config.batch_size, 1000);
        assert_eq!(config.n_centers, 10);
        assert_eq!(config.tolerance, 1e-4);
        assert_eq!(config.max_iterations, 100);
    }
}
