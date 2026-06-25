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

/// GPU array abstraction.
///
/// When a native device runtime is bound, `device_ptr` addresses real device
/// memory. In CPU-fallback builds (the default), the array additionally retains
/// the host-side buffer so that round-trips (`copy_from_host` / `copy_to_host`)
/// preserve the real data instead of fabricating zeros. This keeps every
/// downstream computation (tiled distance assembly, etc.) numerically correct.
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
    /// Host-resident copy of the data (authoritative in CPU-fallback builds)
    host_data: Option<Array2<F>>,
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

    /// Compute the distance sub-matrix for a tile `[i_start, i_end) x [j_start, j_end)`.
    ///
    /// A native GPU backend would launch a tiled kernel here. Since this build ships
    /// without a bound device runtime, we compute the tile on the CPU using the exact
    /// same metric as [`Self::compute_single_distance`]. This returns real pairwise
    /// distances for the requested block instead of a fabricated empty array, so the
    /// assembled distance matrix is mathematically correct on every backend.
    fn compute_distance_tile(
        &self,
        i_start: usize,
        i_end: usize,
        j_start: usize,
        j_end: usize,
    ) -> Result<Array2<F>> {
        let data = self.gpu_data.as_ref().ok_or_else(|| {
            ClusteringError::ComputationError(
                "Distance tile requested before data was loaded to the device".to_string(),
            )
        })?;
        let host = data.copy_to_host()?;

        let n_rows = i_end.saturating_sub(i_start);
        let n_cols = j_end.saturating_sub(j_start);
        let mut tile = Array2::zeros((n_rows, n_cols));

        for (ti, i) in (i_start..i_end).enumerate() {
            for (tj, j) in (j_start..j_end).enumerate() {
                tile[[ti, tj]] = self.compute_single_distance(host.row(i), host.row(j))?;
            }
        }

        Ok(tile)
    }

    /// Compute the point-to-centroid distance sub-matrix for a tile.
    ///
    /// CPU computation of the real block of distances between samples
    /// `[i_start, i_end)` and centroids `[j_start, j_end)`, using the configured
    /// metric. Replaces the previous empty-array stub that silently zeroed the
    /// centroid distance matrix.
    fn compute_centroid_distance_tile(
        &self,
        data: ArrayView2<F>,
        centroids: ArrayView2<F>,
        i_start: usize,
        i_end: usize,
        j_start: usize,
        j_end: usize,
    ) -> Result<Array2<F>> {
        let n_rows = i_end.saturating_sub(i_start);
        let n_cols = j_end.saturating_sub(j_start);
        let mut tile = Array2::zeros((n_rows, n_cols));

        for (ti, i) in (i_start..i_end).enumerate() {
            for (tj, j) in (j_start..j_end).enumerate() {
                tile[[ti, tj]] = self.compute_single_distance(data.row(i), centroids.row(j))?;
            }
        }

        Ok(tile)
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
        let _total_size = shape[0] * shape[1] * element_size;

        // In a native-device build this would request real device memory; in the
        // CPU-fallback build the host buffer (populated by `copy_from_host`) is
        // authoritative, so the pointer is only a placeholder handle.
        let device_ptr = 0x2000_0000;

        Ok(Self {
            device_ptr,
            shape,
            element_size,
            on_device: false,
            host_data: None,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Copy data from host to device.
    ///
    /// Retains a host-resident copy of the real data so it can be read back
    /// faithfully on CPU-fallback backends. A native backend would additionally
    /// issue the host-to-device transfer here.
    pub fn copy_from_host(&mut self, data: ArrayView2<F>) -> Result<()> {
        if data.nrows() != self.shape[0] || data.ncols() != self.shape[1] {
            return Err(ClusteringError::InvalidInput(format!(
                "Host data shape {:?} does not match allocated GPU array shape {:?}",
                [data.nrows(), data.ncols()],
                self.shape
            )));
        }
        self.host_data = Some(data.to_owned());
        self.on_device = true;
        Ok(())
    }

    /// Copy data from device to host.
    ///
    /// Returns the real data previously uploaded with [`Self::copy_from_host`].
    /// Errors honestly if nothing was uploaded rather than fabricating zeros.
    pub fn copy_to_host(&self) -> Result<Array2<F>> {
        self.host_data.clone().ok_or_else(|| {
            ClusteringError::ComputationError(
                "copy_to_host called before any data was uploaded to the GPU array".to_string(),
            )
        })
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
        let matrix = GpuDistanceMatrix::<f64>::new(config, DistanceMetric::Euclidean, None)
            .expect("Operation failed");

        let distance = matrix
            .compute_single_distance(point1.view(), point2.view())
            .expect("Operation failed");
        assert!((distance - 5.196152422706632).abs() < 1e-10);
    }

    #[test]
    fn test_gpu_array_allocation() {
        let mut array = GpuArray::<f32>::allocate([100, 50]).expect("Operation failed");
        assert_eq!(array.shape(), [100, 50]);
        // Freshly allocated array holds no data yet.
        assert!(!array.is_on_device());

        // After uploading, the array reports on-device and round-trips the real data.
        let data = Array2::<f32>::from_elem((100, 50), 1.5);
        array.copy_from_host(data.view()).expect("Operation failed");
        assert!(array.is_on_device());
        let back = array.copy_to_host().expect("Operation failed");
        assert_eq!(back, data);
    }

    #[test]
    fn test_distance_matrix_cpu_fallback() {
        let data = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("Operation failed");

        let config = GpuConfig::default();
        let matrix = GpuDistanceMatrix::new(config, DistanceMetric::Euclidean, None)
            .expect("Operation failed");

        let result = matrix
            .compute_distance_matrix_cpu(data.view())
            .expect("Operation failed");
        assert_eq!(result.shape(), &[3, 3]);
        assert!((result[[0, 0]] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_gpu_tile_path_matches_cpu() {
        // Force the tiled "GPU" code path (preferred backend != CpuFallback makes the
        // stub context report itself as accelerated) and verify the assembled matrix
        // equals the direct CPU computation -- i.e. tiles carry real distances.
        use super::super::core::GpuBackend;

        let data = Array2::from_shape_vec(
            (5, 3),
            vec![
                0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 1.0, 1.0, 1.0, 3.0, 0.0, 4.0,
            ],
        )
        .expect("Operation failed");

        let mut gpu_config = GpuConfig::new(GpuBackend::Cuda);
        gpu_config.auto_fallback = true;
        let mut gpu_matrix =
            GpuDistanceMatrix::<f64>::new(gpu_config, DistanceMetric::Euclidean, Some(2))
                .expect("Operation failed");
        assert!(gpu_matrix.context.is_gpu_accelerated());
        let gpu_result = gpu_matrix
            .compute_distance_matrix(data.view())
            .expect("Operation failed");

        let cpu_matrix =
            GpuDistanceMatrix::<f64>::new(GpuConfig::default(), DistanceMetric::Euclidean, None)
                .expect("Operation failed");
        let cpu_result = cpu_matrix
            .compute_distance_matrix_cpu(data.view())
            .expect("Operation failed");

        assert_eq!(gpu_result.shape(), &[5, 5]);
        for i in 0..5 {
            for j in 0..5 {
                assert!(
                    (gpu_result[[i, j]] - cpu_result[[i, j]]).abs() < 1e-10,
                    "tile mismatch at ({i},{j}): gpu={} cpu={}",
                    gpu_result[[i, j]],
                    cpu_result[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_k_nearest_neighbors() {
        let query = scirs2_core::ndarray::arr1(&[1.0, 1.0]);
        let data = Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 2.0, 2.0, 3.0, 3.0, 1.0, 1.0])
            .expect("Operation failed");

        let config = GpuConfig::default();
        let mut matrix = GpuDistanceMatrix::new(config, DistanceMetric::Euclidean, None)
            .expect("Operation failed");

        let (indices, distances) = matrix
            .find_k_nearest(query.view(), data.view(), 2)
            .expect("Operation failed");
        assert_eq!(indices.len(), 2);
        assert_eq!(distances.len(), 2);
        assert_eq!(indices[0], 3); // Exact match should be first
    }
}
