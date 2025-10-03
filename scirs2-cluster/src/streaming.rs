//! Streaming and memory-efficient clustering algorithms
//!
//! This module provides implementations of clustering algorithms that can handle
//! large datasets that don't fit entirely in memory, using streaming and
//! progressive processing techniques.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

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
    use scirs2_core::ndarray::Array2;

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

/// Enhanced memory management for out-of-core clustering
pub mod memory_management {
    use super::*;

    /// Adaptive memory manager that monitors system resources
    #[derive(Debug, Clone)]
    pub struct AdaptiveMemoryManager {
        /// Current memory usage estimate (bytes)
        current_usage: usize,
        /// Maximum allowed memory usage (bytes)
        max_memory: usize,
        /// Memory pressure threshold (0.0 to 1.0)
        pressure_threshold: f64,
        /// Enable disk-based storage when memory is full
        enable_disk_storage: bool,
        /// Temporary directory for disk storage
        temp_dir: Option<PathBuf>,
    }

    impl AdaptiveMemoryManager {
        /// Create a new adaptive memory manager
        pub fn new(max_memory_mb: usize) -> Self {
            Self {
                current_usage: 0,
                max_memory: max_memory_mb * 1024 * 1024,
                pressure_threshold: 0.8,
                enable_disk_storage: true,
                temp_dir: std::env::temp_dir().into(),
            }
        }

        /// Check if memory pressure is high
        pub fn is_memory_pressure_high(&self) -> bool {
            self.current_usage as f64 / self.max_memory as f64 > self.pressure_threshold
        }

        /// Estimate memory usage for storing data
        pub fn estimate_memory_usage<F: Float>(
            &self,
            n_samples: usize,
            n_features: usize,
        ) -> usize {
            std::mem::size_of::<F>() * n_samples * n_features
        }

        /// Allocate memory for data
        pub fn allocate<F: Float>(&mut self, n_samples: usize, n_features: usize) -> Result<()> {
            let required = self.estimate_memory_usage::<F>(n_samples, n_features);

            if self.current_usage + required > self.max_memory {
                if self.enable_disk_storage {
                    // Allow allocation but mark that we need disk storage
                    Ok(())
                } else {
                    Err(ClusteringError::InvalidInput(
                        "Not enough memory and disk storage is disabled".to_string(),
                    ))
                }
            } else {
                self.current_usage += required;
                Ok(())
            }
        }

        /// Deallocate memory
        pub fn deallocate(&mut self, amount: usize) {
            self.current_usage = self.current_usage.saturating_sub(amount);
        }

        /// Get available memory
        pub fn available_memory(&self) -> usize {
            self.max_memory.saturating_sub(self.current_usage)
        }

        /// Get optimal batch size based on available memory
        pub fn optimal_batch_size<F: Float>(&self, n_features: usize) -> usize {
            let available = self.available_memory();
            let bytes_per_sample = std::mem::size_of::<F>() * n_features;

            if bytes_per_sample == 0 {
                1000 // Default fallback
            } else {
                (available / bytes_per_sample).max(1).min(10000)
            }
        }
    }

    /// Disk-based storage for large intermediate results
    #[derive(Debug)]
    pub struct DiskBackedStorage<F: Float + FromPrimitive> {
        temp_files: Vec<PathBuf>,
        temp_dir: PathBuf,
        buffer_size: usize,
        _phantom: std::marker::PhantomData<F>,
    }

    impl<F: Float + FromPrimitive> DiskBackedStorage<F> {
        /// Create a new disk-backed storage
        pub fn new(temp_dir: Option<PathBuf>, buffer_size: usize) -> Self {
            let temp_dir = temp_dir.unwrap_or_else(std::env::temp_dir);

            Self {
                temp_files: Vec::new(),
                temp_dir,
                buffer_size,
                _phantom: std::marker::PhantomData,
            }
        }

        /// Write data chunk to disk
        pub fn write_chunk(&mut self, data: ArrayView2<F>) -> Result<usize> {
            let chunk_id = self.temp_files.len();
            let file_path = self
                .temp_dir
                .join(format!("cluster_chunk_{}.bin", chunk_id));

            let file = File::create(&file_path).map_err(|e| {
                ClusteringError::InvalidInput(format!("Failed to create temp file: {}", e))
            })?;
            let mut writer = BufWriter::new(file);

            // Write dimensions
            let n_rows = data.shape()[0] as u64;
            let n_cols = data.shape()[1] as u64;
            writer.write_all(&n_rows.to_le_bytes()).map_err(|e| {
                ClusteringError::InvalidInput(format!("Failed to write dimensions: {}", e))
            })?;
            writer.write_all(&n_cols.to_le_bytes()).map_err(|e| {
                ClusteringError::InvalidInput(format!("Failed to write dimensions: {}", e))
            })?;

            // Write data (simplified - in practice, you'd want more sophisticated serialization)
            for row in data.rows() {
                for &value in row.iter() {
                    let bytes = value.to_f64().unwrap_or(0.0).to_le_bytes();
                    writer.write_all(&bytes).map_err(|e| {
                        ClusteringError::InvalidInput(format!("Failed to write data: {}", e))
                    })?;
                }
            }

            writer.flush().map_err(|e| {
                ClusteringError::InvalidInput(format!("Failed to flush data: {}", e))
            })?;

            self.temp_files.push(file_path);
            Ok(chunk_id)
        }

        /// Read data chunk from disk
        pub fn read_chunk(&self, chunk_id: usize) -> Result<Array2<F>> {
            if chunk_id >= self.temp_files.len() {
                return Err(ClusteringError::InvalidInput(
                    "Invalid chunk ID".to_string(),
                ));
            }

            let file = File::open(&self.temp_files[chunk_id]).map_err(|e| {
                ClusteringError::InvalidInput(format!("Failed to open temp file: {}", e))
            })?;
            let mut reader = BufReader::new(file);

            // Read dimensions
            let mut dim_bytes = [0u8; 8];
            reader.read_exact(&mut dim_bytes).map_err(|e| {
                ClusteringError::InvalidInput(format!("Failed to read dimensions: {}", e))
            })?;
            let n_rows = u64::from_le_bytes(dim_bytes) as usize;

            reader.read_exact(&mut dim_bytes).map_err(|e| {
                ClusteringError::InvalidInput(format!("Failed to read dimensions: {}", e))
            })?;
            let n_cols = u64::from_le_bytes(dim_bytes) as usize;

            // Read data
            let mut data = Array2::zeros((n_rows, n_cols));
            for mut row in data.rows_mut() {
                for element in row.iter_mut() {
                    let mut value_bytes = [0u8; 8];
                    reader.read_exact(&mut value_bytes).map_err(|e| {
                        ClusteringError::InvalidInput(format!("Failed to read data: {}", e))
                    })?;
                    let value = f64::from_le_bytes(value_bytes);
                    *element = F::from(value).unwrap_or(F::zero());
                }
            }

            Ok(data)
        }

        /// Clean up temporary files
        pub fn cleanup(&mut self) -> Result<()> {
            for file_path in &self.temp_files {
                if file_path.exists() {
                    std::fs::remove_file(file_path).map_err(|e| {
                        ClusteringError::InvalidInput(format!("Failed to remove temp file: {}", e))
                    })?;
                }
            }
            self.temp_files.clear();
            Ok(())
        }

        /// Get number of chunks stored
        pub fn num_chunks(&self) -> usize {
            self.temp_files.len()
        }
    }

    impl<F: Float + FromPrimitive> Drop for DiskBackedStorage<F> {
        fn drop(&mut self) {
            let _ = self.cleanup(); // Best effort cleanup
        }
    }
}

/// Advanced streaming algorithms for out-of-core processing
pub mod advanced_streaming {
    use super::*;

    /// Count-Min Sketch for approximate frequency counting
    /// Useful for identifying heavy hitters in streaming data
    #[derive(Debug, Clone)]
    pub struct CountMinSketch {
        /// Hash tables (width x depth)
        tables: Vec<Vec<u64>>,
        /// Width of each hash table
        width: usize,
        /// Depth (number of hash tables)
        depth: usize,
        /// Hash functions (simple linear congruential generators)
        hash_params: Vec<(u64, u64)>,
    }

    impl CountMinSketch {
        /// Create a new Count-Min Sketch
        pub fn new(epsilon: f64, delta: f64) -> Self {
            let width = (std::f64::consts::E / epsilon).ceil() as usize;
            let depth = (1.0 / delta).ln().ceil() as usize;

            let mut tables = Vec::new();
            let mut hash_params = Vec::new();

            for i in 0..depth {
                tables.push(vec![0u64; width]);
                // Simple hash parameters (in practice, use better hash functions)
                hash_params.push((
                    1000000007 + i as u64 * 1000000009,
                    1000000021 + i as u64 * 1000000033,
                ));
            }

            Self {
                tables,
                width,
                depth,
                hash_params,
            }
        }

        /// Add an item to the sketch
        pub fn add(&mut self, item: u64) {
            for i in 0..self.depth {
                let hash = self.hash(item, i);
                let idx = (hash as usize) % self.width;
                self.tables[i][idx] += 1;
            }
        }

        /// Estimate the frequency of an item
        pub fn estimate(&self, item: u64) -> u64 {
            let mut min_count = u64::MAX;

            for i in 0..self.depth {
                let hash = self.hash(item, i);
                let idx = (hash as usize) % self.width;
                min_count = min_count.min(self.tables[i][idx]);
            }

            min_count
        }

        /// Simple hash function
        fn hash(&self, item: u64, table_idx: usize) -> u64 {
            let (a, b) = self.hash_params[table_idx];
            a.wrapping_mul(item).wrapping_add(b)
        }

        /// Get heavy hitters (items with frequency above threshold)
        pub fn heavy_hitters(&self, threshold: u64) -> Vec<u64> {
            // This is a simplified implementation
            // In practice, you'd need to track candidates more carefully
            Vec::new()
        }
    }

    /// Reservoir sampling for maintaining a random sample from a stream
    #[derive(Debug, Clone)]
    pub struct ReservoirSampler<T> {
        reservoir: Vec<T>,
        capacity: usize,
        seen_count: usize,
    }

    impl<T: Clone> ReservoirSampler<T> {
        /// Create a new reservoir sampler
        pub fn new(capacity: usize) -> Self {
            Self {
                reservoir: Vec::with_capacity(capacity),
                capacity,
                seen_count: 0,
            }
        }

        /// Add an item to the reservoir
        pub fn add(&mut self, item: T) {
            self.seen_count += 1;

            if self.reservoir.len() < self.capacity {
                self.reservoir.push(item);
            } else {
                // Replace random item with probability k/n
                let random_idx = (self.seen_count - 1) % self.capacity; // Simplified random selection
                if random_idx < self.capacity {
                    self.reservoir[random_idx] = item;
                }
            }
        }

        /// Get the current sample
        pub fn sample(&self) -> &[T] {
            &self.reservoir
        }

        /// Get the number of items seen
        pub fn items_seen(&self) -> usize {
            self.seen_count
        }
    }

    /// Progressive learning framework for online clustering
    #[derive(Debug)]
    pub struct ProgressiveLearner<F: Float> {
        /// Current model state
        model_state: HashMap<String, Vec<F>>,
        /// Learning rate schedule
        learning_rate: F,
        /// Decay factor for learning rate
        decay_factor: F,
        /// Number of updates performed
        update_count: usize,
        /// Memory for recent gradients (for momentum)
        gradient_memory: HashMap<String, Vec<F>>,
        /// Momentum coefficient
        momentum: F,
    }

    impl<F: Float + FromPrimitive + std::fmt::Debug> ProgressiveLearner<F> {
        /// Create a new progressive learner
        pub fn new(initial_lr: F, decay: F, momentum: F) -> Self {
            Self {
                model_state: HashMap::new(),
                learning_rate: initial_lr,
                decay_factor: decay,
                update_count: 0,
                gradient_memory: HashMap::new(),
                momentum,
            }
        }

        /// Update model parameters with gradient
        pub fn update(&mut self, param_name: &str, gradient: &[F]) -> Result<()> {
            self.update_count += 1;

            // Update learning rate with decay
            if self.update_count.is_multiple_of(100) {
                self.learning_rate = self.learning_rate * self.decay_factor;
            }

            // Initialize parameter if not exists
            if !self.model_state.contains_key(param_name) {
                self.model_state
                    .insert(param_name.to_string(), vec![F::zero(); gradient.len()]);
                self.gradient_memory
                    .insert(param_name.to_string(), vec![F::zero(); gradient.len()]);
            }

            let params = self.model_state.get_mut(param_name).unwrap();
            let momentum_grad = self.gradient_memory.get_mut(param_name).unwrap();

            // Update with momentum
            for i in 0..params.len() {
                momentum_grad[i] = self.momentum * momentum_grad[i] + gradient[i];
                params[i] = params[i] - self.learning_rate * momentum_grad[i];
            }

            Ok(())
        }

        /// Get current parameter values
        pub fn get_parameters(&self, param_name: &str) -> Option<&[F]> {
            self.model_state.get(param_name).map(|v| v.as_slice())
        }

        /// Get current learning rate
        pub fn current_learning_rate(&self) -> F {
            self.learning_rate
        }

        /// Get update count
        pub fn update_count(&self) -> usize {
            self.update_count
        }
    }
}

/// Intelligent data loading and preprocessing for streaming
pub mod intelligent_loading {
    use super::*;

    /// Adaptive data loader that adjusts batch sizes based on system performance
    #[derive(Debug)]
    pub struct AdaptiveDataLoader {
        /// Current batch size
        current_batch_size: usize,
        /// Minimum batch size
        min_batch_size: usize,
        /// Maximum batch size
        max_batch_size: usize,
        /// Performance history (processing times)
        performance_history: VecDeque<f64>,
        /// Target processing time per batch (seconds)
        target_time: f64,
        /// Adjustment factor for batch size changes
        adjustment_factor: f64,
    }

    impl AdaptiveDataLoader {
        /// Create a new adaptive data loader
        pub fn new(initial_batch_size: usize, target_time_seconds: f64) -> Self {
            Self {
                current_batch_size: initial_batch_size,
                min_batch_size: initial_batch_size / 10,
                max_batch_size: initial_batch_size * 10,
                performance_history: VecDeque::with_capacity(10),
                target_time: target_time_seconds,
                adjustment_factor: 0.1,
            }
        }

        /// Report batch processing time and adjust batch size
        pub fn report_batch_time(&mut self, processing_time: f64) {
            self.performance_history.push_back(processing_time);
            if self.performance_history.len() > 10 {
                self.performance_history.pop_front();
            }

            // Calculate moving average
            let avg_time = self.performance_history.iter().sum::<f64>()
                / self.performance_history.len() as f64;

            // Adjust batch size based on performance
            if avg_time > self.target_time * 1.2 {
                // Too slow, reduce batch size
                let new_size =
                    (self.current_batch_size as f64 * (1.0 - self.adjustment_factor)) as usize;
                self.current_batch_size = new_size.max(self.min_batch_size);
            } else if avg_time < self.target_time * 0.8 {
                // Too fast, increase batch size
                let new_size =
                    (self.current_batch_size as f64 * (1.0 + self.adjustment_factor)) as usize;
                self.current_batch_size = new_size.min(self.max_batch_size);
            }
        }

        /// Get current optimal batch size
        pub fn current_batch_size(&self) -> usize {
            self.current_batch_size
        }

        /// Get performance statistics
        pub fn get_stats(&self) -> (f64, f64, usize) {
            let avg_time = if self.performance_history.is_empty() {
                0.0
            } else {
                self.performance_history.iter().sum::<f64>() / self.performance_history.len() as f64
            };

            let efficiency = if avg_time > 0.0 {
                self.target_time / avg_time
            } else {
                1.0
            };

            (avg_time, efficiency, self.current_batch_size)
        }
    }

    /// Smart preprocessing pipeline for streaming data
    #[derive(Debug, Clone)]
    pub struct StreamingPreprocessor<F: Float> {
        /// Running statistics for normalization
        running_mean: Option<Array1<F>>,
        running_var: Option<Array1<F>>,
        sample_count: usize,
        /// Enable online normalization
        normalize: bool,
        /// Outlier detection threshold (standard deviations)
        outlier_threshold: F,
        /// Missing value strategy
        missing_value_strategy: MissingValueStrategy,
    }

    #[derive(Debug, Clone)]
    pub enum MissingValueStrategy {
        Drop,
        FillMean,
        FillZero,
        Interpolate,
    }

    impl<F: Float + FromPrimitive + std::fmt::Debug> StreamingPreprocessor<F> {
        /// Create a new streaming preprocessor
        pub fn new(normalize: bool, outlier_threshold: F) -> Self {
            Self {
                running_mean: None,
                running_var: None,
                sample_count: 0,
                normalize,
                outlier_threshold,
                missing_value_strategy: MissingValueStrategy::FillMean,
            }
        }

        /// Process a batch of data
        pub fn process_batch(&mut self, mut data: Array2<F>) -> Result<Array2<F>> {
            let (n_samples, n_features) = (data.shape()[0], data.shape()[1]);

            if n_samples == 0 {
                return Ok(data);
            }

            // Initialize statistics if first batch
            if self.running_mean.is_none() {
                self.running_mean = Some(Array1::zeros(n_features));
                self.running_var = Some(Array1::zeros(n_features));
            }

            // Update running statistics
            if self.normalize {
                self.update_statistics(&data)?;
            }

            // Handle missing values (simplified - assumes NaN for missing)
            self.handle_missing_values(&mut data)?;

            // Apply normalization
            if self.normalize {
                self.apply_normalization(&mut data)?;
            }

            // Detect and handle outliers
            self.handle_outliers(&mut data)?;

            Ok(data)
        }

        /// Update running mean and variance
        fn update_statistics(&mut self, data: &Array2<F>) -> Result<()> {
            let (n_samples, n_features) = (data.shape()[0], data.shape()[1]);
            let mean = self.running_mean.as_mut().unwrap();
            let var = self.running_var.as_mut().unwrap();

            for i in 0..n_samples {
                self.sample_count += 1;
                let sample = data.row(i);

                for j in 0..n_features {
                    if sample[j].is_finite() {
                        // Online update of mean and variance (Welford's algorithm)
                        let delta = sample[j] - mean[j];
                        mean[j] = mean[j] + delta / F::from(self.sample_count).unwrap();
                        let delta2 = sample[j] - mean[j];
                        var[j] = var[j] + delta * delta2;
                    }
                }
            }

            Ok(())
        }

        /// Handle missing values
        fn handle_missing_values(&self, data: &mut Array2<F>) -> Result<()> {
            match self.missing_value_strategy {
                MissingValueStrategy::FillZero => {
                    for elem in data.iter_mut() {
                        if !elem.is_finite() {
                            *elem = F::zero();
                        }
                    }
                }
                MissingValueStrategy::FillMean => {
                    if let Some(ref mean) = self.running_mean {
                        for mut row in data.rows_mut().into_iter() {
                            for (j, elem) in row.iter_mut().enumerate() {
                                if !elem.is_finite() && j < mean.len() {
                                    *elem = mean[j];
                                }
                            }
                        }
                    }
                }
                _ => {} // Other strategies not implemented
            }
            Ok(())
        }

        /// Apply normalization
        fn apply_normalization(&self, data: &mut Array2<F>) -> Result<()> {
            if let (Some(ref mean), Some(ref var)) = (&self.running_mean, &self.running_var) {
                if self.sample_count > 1 {
                    for mut row in data.rows_mut().into_iter() {
                        for (j, elem) in row.iter_mut().enumerate() {
                            if j < mean.len() && var[j] > F::zero() {
                                let std_dev =
                                    (var[j] / F::from(self.sample_count - 1).unwrap()).sqrt();
                                if std_dev > F::zero() {
                                    *elem = (*elem - mean[j]) / std_dev;
                                }
                            }
                        }
                    }
                }
            }
            Ok(())
        }

        /// Handle outliers
        fn handle_outliers(&self, data: &mut Array2<F>) -> Result<()> {
            // Simple outlier detection: clip values beyond threshold standard deviations
            for elem in data.iter_mut() {
                if elem.abs() > self.outlier_threshold {
                    *elem = if *elem > F::zero() {
                        self.outlier_threshold
                    } else {
                        -self.outlier_threshold
                    };
                }
            }
            Ok(())
        }

        /// Get current statistics
        pub fn get_statistics(&self) -> Option<(Array1<F>, Array1<F>)> {
            if let (Some(ref mean), Some(ref var)) = (&self.running_mean, &self.running_var) {
                Some((mean.clone(), var.clone()))
            } else {
                None
            }
        }
    }
}

/// Advanced online clustering algorithms for streaming data
pub mod online_algorithms {
    use super::*;
    use std::collections::VecDeque;

    /// Online K-means with adaptive learning rate
    #[derive(Debug, Clone)]
    pub struct AdaptiveOnlineKMeans<F: Float> {
        /// Current cluster centers
        centers: Array2<F>,
        /// Learning rate schedule
        learning_rate_schedule: LearningRateSchedule,
        /// Current iteration count
        iteration: usize,
        /// Adaptive parameters
        adaptive_params: AdaptiveParams<F>,
        /// Performance metrics
        metrics: OnlineMetrics,
    }

    /// Learning rate scheduling strategies
    #[derive(Debug, Clone)]
    pub enum LearningRateSchedule {
        /// Constant learning rate
        Constant(f64),
        /// Decreasing with iteration: lr / (1 + decay * iteration)
        Decay { initial_lr: f64, decay: f64 },
        /// Step decay: lr * factor every step_size iterations
        StepDecay {
            initial_lr: f64,
            factor: f64,
            step_size: usize,
        },
        /// Adaptive based on cluster stability
        Adaptive {
            min_lr: f64,
            max_lr: f64,
            stability_window: usize,
        },
    }

    /// Adaptive parameters for online learning
    #[derive(Debug, Clone)]
    pub struct AdaptiveParams<F: Float> {
        /// Momentum for center updates
        pub momentum: F,
        /// Cluster stability tracking
        pub stability_scores: Vec<F>,
        /// Recent center movements
        pub center_movements: VecDeque<F>,
        /// Automatic cluster count adjustment
        pub auto_k_adjustment: bool,
        /// Split threshold for creating new clusters
        pub split_threshold: F,
        /// Merge threshold for combining clusters
        pub merge_threshold: F,
    }

    /// Online performance metrics
    #[derive(Debug, Clone, Default)]
    pub struct OnlineMetrics {
        /// Running estimate of within-cluster sum of squares
        pub wcss: f64,
        /// Number of samples processed
        pub samples_processed: usize,
        /// Center update frequency
        pub update_frequency: f64,
        /// Cluster assignments distribution
        pub cluster_distribution: Vec<usize>,
        /// Processing time per batch
        pub batch_processing_times: VecDeque<f64>,
    }

    impl<F: Float + FromPrimitive + Debug> AdaptiveOnlineKMeans<F> {
        /// Create a new adaptive online K-means instance
        pub fn new(
            initial_centers: Array2<F>,
            learning_rate_schedule: LearningRateSchedule,
        ) -> Self {
            let n_clusters = initial_centers.nrows();
            let adaptive_params = AdaptiveParams {
                momentum: F::from(0.9).unwrap(),
                stability_scores: vec![F::zero(); n_clusters],
                center_movements: VecDeque::with_capacity(100),
                auto_k_adjustment: false,
                split_threshold: F::from(2.0).unwrap(),
                merge_threshold: F::from(0.5).unwrap(),
            };

            Self {
                centers: initial_centers,
                learning_rate_schedule,
                iteration: 0,
                adaptive_params,
                metrics: OnlineMetrics::default(),
            }
        }

        /// Process a new sample and update clusters
        pub fn update(&mut self, sample: ArrayView1<F>) -> Result<usize> {
            let start_time = std::time::Instant::now();

            // Find nearest cluster
            let (nearest_cluster, min_distance) = self.find_nearest_cluster(sample)?;

            // Get current learning rate
            let lr = self.get_current_learning_rate();

            // Update cluster center
            let old_center = self.centers.row(nearest_cluster).to_owned();
            self.update_center(nearest_cluster, sample, lr)?;

            // Track center movement for adaptive learning
            let movement = euclidean_distance(old_center.view(), self.centers.row(nearest_cluster));
            self.adaptive_params.center_movements.push_back(movement);
            if self.adaptive_params.center_movements.len() > 100 {
                self.adaptive_params.center_movements.pop_front();
            }

            // Update metrics
            self.update_metrics(
                nearest_cluster,
                min_distance,
                start_time.elapsed().as_secs_f64(),
            );

            // Check for adaptive cluster adjustments
            if self.adaptive_params.auto_k_adjustment {
                self.maybe_adjust_clusters(sample, min_distance)?;
            }

            self.iteration += 1;
            Ok(nearest_cluster)
        }

        /// Adaptively adjust cluster count based on distance thresholds
        fn maybe_adjust_clusters(&mut self, sample: ArrayView1<F>, min_distance: F) -> Result<()> {
            // Split: Create new cluster if sample is far from all existing clusters
            if min_distance > self.adaptive_params.split_threshold {
                // Add new cluster at sample location
                let n_clusters = self.centers.nrows();
                let n_features = self.centers.ncols();

                let mut new_centers = Array2::<F>::zeros((n_clusters + 1, n_features));

                // Copy existing centers
                for i in 0..n_clusters {
                    for j in 0..n_features {
                        new_centers[[i, j]] = self.centers[[i, j]];
                    }
                }

                // Add new center at sample location
                for (j, &val) in sample.iter().enumerate() {
                    if j < n_features {
                        new_centers[[n_clusters, j]] = val;
                    }
                }

                self.centers = new_centers;
                self.adaptive_params.stability_scores.push(F::zero());
            }

            // Merge: Combine clusters that are too close
            let n_clusters = self.centers.nrows();
            if n_clusters > 1 {
                let mut clusters_to_merge: Option<(usize, usize)> = None;
                let mut min_inter_distance = F::infinity();

                // Find closest pair of clusters
                for i in 0..n_clusters {
                    for j in (i + 1)..n_clusters {
                        let dist = euclidean_distance(self.centers.row(i), self.centers.row(j));

                        if dist < min_inter_distance {
                            min_inter_distance = dist;
                            clusters_to_merge = Some((i, j));
                        }
                    }
                }

                // Merge if below threshold
                if let Some((i, j)) = clusters_to_merge {
                    if min_inter_distance < self.adaptive_params.merge_threshold {
                        let n_features = self.centers.ncols();
                        let mut new_centers = Array2::<F>::zeros((n_clusters - 1, n_features));

                        // Merge clusters i and j by averaging
                        let mut merged_center = Array1::<F>::zeros(n_features);
                        for k in 0..n_features {
                            merged_center[k] = (self.centers[[i, k]] + self.centers[[j, k]])
                                / (F::one() + F::one());
                        }

                        // Build new center matrix
                        let mut new_idx = 0;
                        for old_idx in 0..n_clusters {
                            if old_idx == i {
                                // Add merged center
                                for k in 0..n_features {
                                    new_centers[[new_idx, k]] = merged_center[k];
                                }
                                new_idx += 1;
                            } else if old_idx != j {
                                // Copy unchanged center
                                for k in 0..n_features {
                                    new_centers[[new_idx, k]] = self.centers[[old_idx, k]];
                                }
                                new_idx += 1;
                            }
                        }

                        self.centers = new_centers;

                        // Update stability scores (remove merged cluster)
                        if j < self.adaptive_params.stability_scores.len() {
                            self.adaptive_params.stability_scores.remove(j);
                        }
                    }
                }
            }

            Ok(())
        }

        /// Find the nearest cluster to a sample
        fn find_nearest_cluster(&self, sample: ArrayView1<F>) -> Result<(usize, F)> {
            let mut min_distance = F::infinity();
            let mut nearest_cluster = 0;

            for (i, center) in self.centers.rows().into_iter().enumerate() {
                let distance = euclidean_distance(sample, center);
                if distance < min_distance {
                    min_distance = distance;
                    nearest_cluster = i;
                }
            }

            Ok((nearest_cluster, min_distance))
        }

        /// Update a cluster center using momentum
        fn update_center(
            &mut self,
            cluster_idx: usize,
            sample: ArrayView1<F>,
            lr: f64,
        ) -> Result<()> {
            let learning_rate = F::from(lr).unwrap();
            let momentum = self.adaptive_params.momentum;

            let mut center = self.centers.row_mut(cluster_idx);
            for (i, &sample_val) in sample.iter().enumerate() {
                if i < center.len() {
                    let old_val = center[i];
                    let gradient = sample_val - old_val;
                    let update = learning_rate * gradient;
                    center[i] = momentum * old_val + (F::one() - momentum) * (old_val + update);
                }
            }

            Ok(())
        }

        /// Get current learning rate based on schedule
        fn get_current_learning_rate(&self) -> f64 {
            match &self.learning_rate_schedule {
                LearningRateSchedule::Constant(lr) => *lr,
                LearningRateSchedule::Decay { initial_lr, decay } => {
                    initial_lr / (1.0 + decay * self.iteration as f64)
                }
                LearningRateSchedule::StepDecay {
                    initial_lr,
                    factor,
                    step_size,
                } => {
                    let steps = self.iteration / step_size;
                    initial_lr * factor.powi(steps as i32)
                }
                LearningRateSchedule::Adaptive {
                    min_lr,
                    max_lr,
                    stability_window,
                } => {
                    let recent_movements: Vec<F> = self
                        .adaptive_params
                        .center_movements
                        .iter()
                        .rev()
                        .take(*stability_window)
                        .cloned()
                        .collect();

                    if recent_movements.is_empty() {
                        return *max_lr;
                    }

                    let avg_movement = recent_movements.iter().fold(F::zero(), |acc, x| acc + *x)
                        / F::from(recent_movements.len()).unwrap();
                    let stability = F::one() / (F::one() + avg_movement);

                    // High stability = low learning rate, low stability = high learning rate
                    let adaptive_lr =
                        min_lr + (max_lr - min_lr) * (F::one() - stability).to_f64().unwrap();
                    adaptive_lr.clamp(*min_lr, *max_lr)
                }
            }
        }

        /// Update performance metrics
        fn update_metrics(&mut self, cluster_idx: usize, distance: F, processing_time: f64) {
            self.metrics.samples_processed += 1;

            // Update WCSS estimate
            let distance_sq = distance.to_f64().unwrap().powi(2);
            let n = self.metrics.samples_processed as f64;
            self.metrics.wcss = ((n - 1.0) * self.metrics.wcss + distance_sq) / n;

            // Update cluster distribution
            if cluster_idx >= self.metrics.cluster_distribution.len() {
                self.metrics.cluster_distribution.resize(cluster_idx + 1, 0);
            }
            self.metrics.cluster_distribution[cluster_idx] += 1;

            // Track processing times
            self.metrics
                .batch_processing_times
                .push_back(processing_time);
            if self.metrics.batch_processing_times.len() > 1000 {
                self.metrics.batch_processing_times.pop_front();
            }

            // Update frequency calculation
            let total_updates = self.metrics.cluster_distribution.iter().sum::<usize>() as f64;
            self.metrics.update_frequency = total_updates / self.iteration.max(1) as f64;
        }

        /// Get current cluster centers
        pub fn get_centers(&self) -> &Array2<F> {
            &self.centers
        }

        /// Get current metrics
        pub fn get_metrics(&self) -> &OnlineMetrics {
            &self.metrics
        }

        /// Predict cluster for new samples
        pub fn predict(&self, samples: ArrayView2<F>) -> Result<Array1<usize>> {
            let n_samples = samples.nrows();
            let mut predictions = Array1::zeros(n_samples);

            for (i, sample) in samples.rows().into_iter().enumerate() {
                let (cluster_id, _distance) = self.find_nearest_cluster(sample)?;
                predictions[i] = cluster_id;
            }

            Ok(predictions)
        }
    }
}
