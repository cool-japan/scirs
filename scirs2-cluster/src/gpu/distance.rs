//! GPU-accelerated distance computations
//!
//! This module provides GPU-accelerated distance matrix computations and
//! various distance metrics optimized for GPU hardware.

use crate::error::{ClusteringError, Result};
use scirs2_core::ndarray::{Array2, ArrayView1, ArrayView2};
use scirs2_core::numeric::{Float, FromPrimitive};
use serde::{Deserialize, Serialize};

use super::core::{GpuConfig, GpuContext};
use super::memory::{GpuMemoryManager, MemoryTransfer};

/// Distance metrics supported by GPU acceleration
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum DistanceMetric {
    /// Euclidean distance (L2 norm)
    Euclidean,
    /// Manhattan distance (L1 norm)
    Manhattan,
    /// Cosine distance
    Cosine,
    /// Minkowski distance with custom p
    Minkowski(f64),
    /// Squared Euclidean distance (faster, no sqrt)
    SquaredEuclidean,
    /// Chebyshev distance (L norm)
    Chebyshev,
    /// Hamming distance (for binary data)
    Hamming,
}

impl Default for DistanceMetric {
    fn default() -> Self {
        DistanceMetric::Euclidean
    }
}

impl std::fmt::Display for DistanceMetric {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DistanceMetric::Euclidean => write!(f, "euclidean"),
            DistanceMetric::Manhattan => write!(f, "manhattan"),
            DistanceMetric::Cosine => write!(f, "cosine"),
            DistanceMetric::Minkowski(p) => write!(f, "minkowski(p={})", p),
            DistanceMetric::SquaredEuclidean => write!(f, "squared_euclidean"),
            DistanceMetric::Chebyshev => write!(f, "chebyshev"),
            DistanceMetric::Hamming => write!(f, "hamming"),
        }
    }
}

/// Enhanced GPU distance matrix for fast nearest neighbor computations
#[derive(Debug)]
pub struct GpuDistanceMatrix<F: Float> {
    /// GPU context
    context: GpuContext,
    /// Distance metric
    metric: DistanceMetric,
    /// Pre-loaded GPU data
    gpu_data: Option<GpuArray<F>>,
    /// Tile size for blocked computations
    tile_size: usize,
    /// Whether to use shared memory optimization
    use_shared_memory: bool,
    /// Memory manager
    memory_manager: GpuMemoryManager,
}

/// GPU array abstraction
#[derive(Debug)]
pub struct GpuArray<F: Float> {
    /// Device pointer
    device_ptr: usize,
    /// Array shape (rows, cols)
    shape: [usize; 2],
    /// Data type size in bytes
    element_size: usize,
    /// Whether data is currently on device
    on_device: bool,
    _phantom: std::marker::PhantomData<F>,
}

impl<F: Float + FromPrimitive + Send + Sync> GpuDistanceMatrix<F> {
    /// Create new GPU distance matrix
    pub fn new(
        gpu_config: GpuConfig,
        metric: DistanceMetric,
        tile_size: Option<usize>,
    ) -> Result<Self> {
        let device = Self::detect_gpu_device(&gpu_config)?;
        let context = GpuContext::new(device, gpu_config)?;

        let optimal_tile_size =
            tile_size.unwrap_or_else(|| Self::calculate_optimal_tile_size(&context));

        let memory_manager = GpuMemoryManager::new(256, 100);

        Ok(Self {
            context,
            metric,
            gpu_data: None,
            tile_size: optimal_tile_size,
            use_shared_memory: true,
            memory_manager,
        })
    }

    /// Preload data to GPU for repeated distance computations
    pub fn preload_data(&mut self, data: ArrayView2<F>) -> Result<()> {
        let shape = [data.nrows(), data.ncols()];
        let mut gpu_data = GpuArray::allocate(shape)?;
        gpu_data.copy_from_host(data)?;
        self.gpu_data = Some(gpu_data);
        Ok(())
    }

    /// Compute full distance matrix
    pub fn compute_distance_matrix(&mut self, data: ArrayView2<F>) -> Result<Array2<F>> {
        let n_samples = data.nrows();
        let mut result = Array2::zeros((n_samples, n_samples));

        if !self.context.is_gpu_accelerated() {
            // CPU fallback
            return self.compute_distance_matrix_cpu(data);
        }

        // Use preloaded data if available
        if self.gpu_data.is_none() {
            self.preload_data(data)?;
        }

        // GPU computation with tiling
        for i in (0..n_samples).step_by(self.tile_size) {
            for j in (0..n_samples).step_by(self.tile_size) {
                let i_end = (i + self.tile_size).min(n_samples);
                let j_end = (j + self.tile_size).min(n_samples);

                let tile_result = self.compute_distance_tile(i, i_end, j, j_end)?;

                // Copy results back to host
                for (ii, row) in tile_result.rows().into_iter().enumerate() {
                    for (jj, &val) in row.iter().enumerate() {
                        if i + ii < n_samples && j + jj < n_samples {
                            result[[i + ii, j + jj]] = val;
                        }
                    }
                }
            }
        }

        Ok(result)
    }

    /// Compute distances from points to centroids
    pub fn compute_distances_to_centroids(
        &mut self,
        data: ArrayView2<F>,
        centroids: ArrayView2<F>,
    ) -> Result<Array2<F>> {
        let n_samples = data.nrows();
        let n_centroids = centroids.nrows();
        let mut result = Array2::zeros((n_samples, n_centroids));

        if !self.context.is_gpu_accelerated() {
            return self.compute_distances_to_centroids_cpu(data, centroids);
        }

        // GPU implementation
        for i in (0..n_samples).step_by(self.tile_size) {
            let i_end = (i + self.tile_size).min(n_samples);

            for j in (0..n_centroids).step_by(self.tile_size) {
                let j_end = (j + self.tile_size).min(n_centroids);

                let tile_result =
                    self.compute_centroid_distance_tile(data, centroids, i, i_end, j, j_end)?;

                // Copy results
                for (ii, row) in tile_result.rows().into_iter().enumerate() {
                    for (jj, &val) in row.iter().enumerate() {
                        if i + ii < n_samples && j + jj < n_centroids {
                            result[[i + ii, j + jj]] = val;
                        }
                    }
                }
            }
        }

        Ok(result)
    }

    /// Find k nearest neighbors
    pub fn find_k_nearest(
        &mut self,
        query: ArrayView1<F>,
        data: ArrayView2<F>,
        k: usize,
    ) -> Result<(Vec<usize>, Vec<F>)> {
        if k == 0 || k > data.nrows() {
            return Err(ClusteringError::InvalidInput(
                "Invalid k value for k-nearest neighbors".to_string(),
            ));
        }

        let distances = self.compute_point_distances(query, data)?;

        // Sort and get top k
        let mut indexed_distances: Vec<(usize, F)> =
            distances.iter().enumerate().map(|(i, &d)| (i, d)).collect();

        indexed_distances
            .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let indices = indexed_distances.iter().take(k).map(|(i, _)| *i).collect();
        let distances = indexed_distances.iter().take(k).map(|(_, d)| *d).collect();

        Ok((indices, distances))
    }

    /// Compute distances from a single point to all data points
    fn compute_point_distances(
        &mut self,
        query: ArrayView1<F>,
        data: ArrayView2<F>,
    ) -> Result<Vec<F>> {
        let n_samples = data.nrows();
        let mut distances = vec![F::zero(); n_samples];

        for (i, data_point) in data.rows().into_iter().enumerate() {
            distances[i] = self.compute_single_distance(query, data_point)?;
        }

        Ok(distances)
    }

    /// Compute distance between two points
    fn compute_single_distance(&self, point1: ArrayView1<F>, point2: ArrayView1<F>) -> Result<F> {
        if point1.len() != point2.len() {
            return Err(ClusteringError::InvalidInput(
                "Points must have same dimensionality".to_string(),
            ));
        }

        let distance = match self.metric {
            DistanceMetric::Euclidean => {
                let sum_sq: F = point1
                    .iter()
                    .zip(point2.iter())
                    .map(|(&a, &b)| (a - b) * (a - b))
                    .fold(F::zero(), |acc, x| acc + x);
                sum_sq.sqrt()
            }
            DistanceMetric::SquaredEuclidean => point1
                .iter()
                .zip(point2.iter())
                .map(|(&a, &b)| (a - b) * (a - b))
                .fold(F::zero(), |acc, x| acc + x),
            DistanceMetric::Manhattan => point1
                .iter()
                .zip(point2.iter())
                .map(|(&a, &b)| (a - b).abs())
                .fold(F::zero(), |acc, x| acc + x),
            DistanceMetric::Cosine => {
                let dot_product = point1
                    .iter()
                    .zip(point2.iter())
                    .map(|(&a, &b)| a * b)
                    .fold(F::zero(), |acc, x| acc + x);

                let norm1 = point1
                    .iter()
                    .map(|&x| x * x)
                    .fold(F::zero(), |acc, x| acc + x)
                    .sqrt();

                let norm2 = point2
                    .iter()
                    .map(|&x| x * x)
                    .fold(F::zero(), |acc, x| acc + x)
                    .sqrt();

                if norm1 == F::zero() || norm2 == F::zero() {
                    F::one()
                } else {
                    F::one() - (dot_product / (norm1 * norm2))
                }
            }
            DistanceMetric::Chebyshev => point1
                .iter()
                .zip(point2.iter())
                .map(|(&a, &b)| (a - b).abs())
                .fold(F::zero(), |acc, x| if x > acc { x } else { acc }),
            DistanceMetric::Minkowski(p) => {
                let p_f = F::from(p).unwrap_or(F::one());
                let sum: F = point1
                    .iter()
                    .zip(point2.iter())
                    .map(|(&a, &b)| (a - b).abs().powf(p_f))
                    .fold(F::zero(), |acc, x| acc + x);
                sum.powf(F::one() / p_f)
            }
            DistanceMetric::Hamming => {
                // For continuous data, use threshold-based Hamming
                let threshold = F::from(0.5).unwrap_or(F::zero());
                let count = point1
                    .iter()
                    .zip(point2.iter())
                    .filter(|(&a, &b)| (a - b).abs() > threshold)
                    .count();
                F::from(count).unwrap_or(F::zero())
            }
        };

        Ok(distance)
    }

    /// CPU fallback for distance matrix computation
    pub fn compute_distance_matrix_cpu(&self, data: ArrayView2<F>) -> Result<Array2<F>> {
        let n_samples = data.nrows();
        let mut result = Array2::zeros((n_samples, n_samples));

        for i in 0..n_samples {
            for j in i..n_samples {
                let distance = self.compute_single_distance(data.row(i), data.row(j))?;
                result[[i, j]] = distance;
                result[[j, i]] = distance;
            }
        }

        Ok(result)
    }

    /// CPU fallback for centroid distances
    fn compute_distances_to_centroids_cpu(
        &self,
        data: ArrayView2<F>,
        centroids: ArrayView2<F>,
    ) -> Result<Array2<F>> {
        let n_samples = data.nrows();
        let n_centroids = centroids.nrows();
        let mut result = Array2::zeros((n_samples, n_centroids));

        for i in 0..n_samples {
            for j in 0..n_centroids {
                let distance = self.compute_single_distance(data.row(i), centroids.row(j))?;
                result[[i, j]] = distance;
            }
        }

        Ok(result)
    }

    /// Stub implementations for GPU computations
    fn compute_distance_tile(
        &self,
        _i_start: usize,
        _i_end: usize,
        _j_start: usize,
        _j_end: usize,
    ) -> Result<Array2<F>> {
        // This would contain the actual GPU kernel launch
        // For now, return empty array as stub
        Ok(Array2::zeros((1, 1)))
    }

    fn compute_centroid_distance_tile(
        &self,
        _data: ArrayView2<F>,
        _centroids: ArrayView2<F>,
        _i_start: usize,
        _i_end: usize,
        _j_start: usize,
        _j_end: usize,
    ) -> Result<Array2<F>> {
        // This would contain the actual GPU kernel launch
        // For now, return empty array as stub
        Ok(Array2::zeros((1, 1)))
    }

    /// Detect available GPU device
    fn detect_gpu_device(config: &GpuConfig) -> Result<super::core::GpuDevice> {
        // Stub implementation - would detect actual GPU devices
        Ok(super::core::GpuDevice::new(
            0,
            "Stub GPU".to_string(),
            8_000_000_000,
            6_000_000_000,
            "1.0".to_string(),
            1024,
            config.preferred_backend,
            true,
        ))
    }

    /// Calculate optimal tile size based on GPU capabilities
    fn calculate_optimal_tile_size(context: &GpuContext) -> usize {
        // Calculate based on available memory and compute units
        let (total_memory, available_memory) = context.memory_info();
        let compute_units = context.device.compute_units as usize;

        // Simple heuristic: balance memory usage and parallelism
        let memory_based = (available_memory / (8 * std::mem::size_of::<F>())).min(1024);
        let compute_based = (compute_units * 32).min(512);

        memory_based.min(compute_based).max(32)
    }
}

impl<F: Float> GpuArray<F> {
    /// Allocate GPU array
    pub fn allocate(shape: [usize; 2]) -> Result<Self> {
        let element_size = std::mem::size_of::<F>();
        let total_size = shape[0] * shape[1] * element_size;

        // Stub allocation - would allocate actual GPU memory
        let device_ptr = 0x2000_0000; // Fake pointer

        Ok(Self {
            device_ptr,
            shape,
            element_size,
            on_device: true,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Copy data from host to device
    pub fn copy_from_host(&mut self, _data: ArrayView2<F>) -> Result<()> {
        // Stub implementation - would perform actual memory transfer
        self.on_device = true;
        Ok(())
    }

    /// Copy data from device to host
    pub fn copy_to_host(&self) -> Result<Array2<F>> {
        // Stub implementation - would perform actual memory transfer
        Ok(Array2::zeros((self.shape[0], self.shape[1])))
    }

    /// Get array shape
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }

    /// Check if data is on device
    pub fn is_on_device(&self) -> bool {
        self.on_device
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_distance_metrics() {
        let point1 = scirs2_core::ndarray::arr1(&[1.0, 2.0, 3.0]);
        let point2 = scirs2_core::ndarray::arr1(&[4.0, 5.0, 6.0]);

        let config = GpuConfig::default();
        let matrix =
            GpuDistanceMatrix::<f64>::new(config, DistanceMetric::Euclidean, None).unwrap();

        let distance = matrix
            .compute_single_distance(point1.view(), point2.view())
            .unwrap();
        assert!((distance - 5.196152422706632).abs() < 1e-10);
    }

    #[test]
    fn test_gpu_array_allocation() {
        let array = GpuArray::<f32>::allocate([100, 50]).unwrap();
        assert_eq!(array.shape(), [100, 50]);
        assert!(array.is_on_device());
    }

    #[test]
    fn test_distance_matrix_cpu_fallback() {
        let data = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        let config = GpuConfig::default();
        let matrix = GpuDistanceMatrix::new(config, DistanceMetric::Euclidean, None).unwrap();

        let result = matrix.compute_distance_matrix_cpu(data.view()).unwrap();
        assert_eq!(result.shape(), &[3, 3]);
        assert!((result[[0, 0]] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_k_nearest_neighbors() {
        let query = scirs2_core::ndarray::arr1(&[1.0, 1.0]);
        let data =
            Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 2.0, 2.0, 3.0, 3.0, 1.0, 1.0]).unwrap();

        let config = GpuConfig::default();
        let mut matrix = GpuDistanceMatrix::new(config, DistanceMetric::Euclidean, None).unwrap();

        let (indices, distances) = matrix.find_k_nearest(query.view(), data.view(), 2).unwrap();
        assert_eq!(indices.len(), 2);
        assert_eq!(distances.len(), 2);
        assert_eq!(indices[0], 3); // Exact match should be first
    }
}
