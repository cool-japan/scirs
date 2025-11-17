//! Hardware acceleration utilities for metrics computation
//!
//! This module provides hardware-accelerated implementations of common metrics
//! using unified SIMD operations from scirs2-core for improved performance
//! and cross-platform compatibility.

use crate::error::{MetricsError, Result};
use scirs2_core::ndarray::ArrayStatCompat;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use scirs2_core::numeric::{Float, One, Zero};
use scirs2_core::parallel_ops::*;
use scirs2_core::simd_ops::{AutoOptimizer, PlatformCapabilities, SimdUnifiedOps};
use statrs::statistics::Statistics;

/// Configuration for hardware acceleration
#[derive(Debug, Clone)]
pub struct HardwareAccelConfig {
    /// Enable SIMD vectorization
    pub enable_simd: bool,
    /// Enable GPU acceleration (when available)
    pub enable_gpu: bool,
    /// Minimum data size to use hardware acceleration
    pub min_data_size: usize,
    /// Preferred vector width for SIMD operations
    pub vector_width: VectorWidth,
    /// GPU memory threshold for offloading
    pub gpu_memory_threshold: usize,
}

/// Vector width for SIMD operations
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VectorWidth {
    /// 128-bit vectors (SSE)
    V128,
    /// 256-bit vectors (AVX)
    V256,
    /// 512-bit vectors (AVX-512)
    V512,
    /// Auto-detect best available
    Auto,
}

impl Default for HardwareAccelConfig {
    fn default() -> Self {
        Self {
            enable_simd: true,
            enable_gpu: false, // Disabled by default until GPU backend is ready
            min_data_size: 1000,
            vector_width: VectorWidth::Auto,
            gpu_memory_threshold: 1024 * 1024, // 1MB
        }
    }
}

impl HardwareAccelConfig {
    /// Create new hardware acceleration configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable/disable SIMD vectorization
    pub fn with_simd_enabled(mut self, enabled: bool) -> Self {
        self.enable_simd = enabled;
        self
    }

    /// Enable/disable GPU acceleration
    pub fn with_gpu_enabled(mut self, enabled: bool) -> Self {
        self.enable_gpu = enabled;
        self
    }

    /// Set minimum data size for hardware acceleration
    pub fn with_min_data_size(mut self, size: usize) -> Self {
        self.min_data_size = size;
        self
    }

    /// Set preferred vector width
    pub fn with_vector_width(mut self, width: VectorWidth) -> Self {
        self.vector_width = width;
        self
    }

    /// Set GPU memory threshold
    pub fn with_gpu_memory_threshold(mut self, threshold: usize) -> Self {
        self.gpu_memory_threshold = threshold;
        self
    }
}

/// Hardware capabilities detector (using core platform capabilities)
#[derive(Debug)]
pub struct HardwareCapabilities {
    pub has_sse: bool,
    pub has_sse2: bool,
    pub has_sse3: bool,
    pub has_ssse3: bool,
    pub has_sse41: bool,
    pub has_sse42: bool,
    pub has_avx: bool,
    pub has_avx2: bool,
    pub has_avx512f: bool,
    pub has_fma: bool,
    pub has_gpu: bool,
    pub gpu_memory: Option<usize>,
    // Store individual capabilities instead of the entire struct
    pub simd_available: bool,
    pub avx2_available: bool,
    pub avx512_available: bool,
    pub gpu_available: bool,
}

impl HardwareCapabilities {
    /// Detect hardware capabilities using core platform detection
    pub fn detect() -> Self {
        let core_caps = PlatformCapabilities::detect();

        Self {
            has_sse: true, // Assume SSE is available if we're on x86_64 (legacy compatibility)
            has_sse2: core_caps.simd_available,
            has_sse3: core_caps.simd_available,
            has_ssse3: core_caps.simd_available,
            has_sse41: core_caps.simd_available,
            has_sse42: core_caps.simd_available,
            has_avx: core_caps.avx2_available,
            has_avx2: core_caps.avx2_available,
            has_avx512f: core_caps.avx512_available,
            has_fma: core_caps.simd_available, // FMA is typically available with modern SIMD
            has_gpu: core_caps.gpu_available,
            gpu_memory: None, // GPU memory detection not implemented in core yet
            simd_available: core_caps.simd_available,
            avx2_available: core_caps.avx2_available,
            avx512_available: core_caps.avx512_available,
            gpu_available: core_caps.gpu_available,
        }
    }

    /// Get optimal vector width for current hardware
    pub fn optimal_vector_width(&self) -> VectorWidth {
        if self.avx512_available {
            VectorWidth::V512
        } else if self.avx2_available {
            VectorWidth::V256
        } else {
            VectorWidth::V128 // Default for SIMD or fallback
        }
    }

    /// Check if SIMD is available
    pub fn simd_available(&self) -> bool {
        self.simd_available
    }
}

/// SIMD-accelerated distance computations
pub struct SimdDistanceMetrics {
    config: HardwareAccelConfig,
    capabilities: HardwareCapabilities,
}

impl SimdDistanceMetrics {
    /// Create new SIMD distance metrics calculator
    pub fn new() -> Self {
        Self {
            config: HardwareAccelConfig::default(),
            capabilities: HardwareCapabilities::detect(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: HardwareAccelConfig) -> Self {
        Self {
            config,
            capabilities: HardwareCapabilities::detect(),
        }
    }

    /// Compute Euclidean distance using SIMD
    pub fn euclidean_distance_simd<F>(&self, a: &Array1<F>, b: &Array1<F>) -> Result<F>
    where
        F: Float + SimdUnifiedOps + std::fmt::Debug + 'static,
    {
        if a.len() != b.len() {
            return Err(MetricsError::InvalidInput(
                "Arrays must have the same length".to_string(),
            ));
        }

        if !self.config.enable_simd
            || !self.capabilities.simd_available()
            || a.len() < self.config.min_data_size
        {
            // Fallback to standard implementation
            return self.euclidean_distance_standard(a, b);
        }

        // Use unified SIMD operations for distance calculation
        let diff = F::simd_sub(&a.view(), &b.view());
        let squared_diff = F::simd_mul(&diff.view(), &diff.view());
        let distance = F::simd_sum(&squared_diff.view()).sqrt();
        Ok(distance)
    }

    /// Compute Manhattan distance using SIMD
    pub fn manhattan_distance_simd<F>(&self, a: &Array1<F>, b: &Array1<F>) -> Result<F>
    where
        F: Float + SimdUnifiedOps + std::fmt::Debug + 'static + std::iter::Sum,
    {
        if a.len() != b.len() {
            return Err(MetricsError::InvalidInput(
                "Arrays must have the same length".to_string(),
            ));
        }

        if !self.config.enable_simd
            || !self.capabilities.simd_available()
            || a.len() < self.config.min_data_size
        {
            return self.manhattan_distance_standard(a, b);
        }

        // Use unified SIMD operations for Manhattan distance
        let diff = F::simd_sub(&a.view(), &b.view());
        let abs_diff = F::simd_abs(&diff.view());
        let distance = F::simd_sum(&abs_diff.view());
        Ok(distance)
    }

    /// Compute cosine distance using SIMD
    pub fn cosine_distance_simd<F>(&self, a: &Array1<F>, b: &Array1<F>) -> Result<F>
    where
        F: Float + SimdUnifiedOps + std::fmt::Debug + PartialEq + 'static,
    {
        if a.len() != b.len() {
            return Err(MetricsError::InvalidInput(
                "Arrays must have the same length".to_string(),
            ));
        }

        if !self.config.enable_simd
            || !self.capabilities.simd_available()
            || a.len() < self.config.min_data_size
        {
            return self.cosine_distance_standard(a, b);
        }

        // For cosine distance, we need dot product and norms
        let dot_product = self.dot_product_simd(a, b)?;
        let norm_a = self.euclidean_norm_simd(a)?;
        let norm_b = self.euclidean_norm_simd(b)?;

        if norm_a == F::zero() || norm_b == F::zero() {
            return Ok(F::one()); // Maximum distance
        }

        let cosine_similarity = dot_product / (norm_a * norm_b);
        Ok(F::one() - cosine_similarity)
    }

    /// Compute dot product using SIMD
    pub fn dot_product_simd<F>(&self, a: &Array1<F>, b: &Array1<F>) -> Result<F>
    where
        F: Float + SimdUnifiedOps + std::fmt::Debug + 'static,
    {
        if a.len() != b.len() {
            return Err(MetricsError::InvalidInput(
                "Arrays must have the same length".to_string(),
            ));
        }

        if !self.config.enable_simd
            || !self.capabilities.simd_available()
            || a.len() < self.config.min_data_size
        {
            return Ok(a.dot(b));
        }

        // Use unified SIMD operations for dot product
        let dot_product = F::simd_dot(&a.view(), &b.view());
        Ok(dot_product)
    }

    /// Compute Euclidean norm using SIMD
    pub fn euclidean_norm_simd<F>(&self, a: &Array1<F>) -> Result<F>
    where
        F: Float + SimdUnifiedOps + std::fmt::Debug + 'static,
    {
        if !self.config.enable_simd
            || !self.capabilities.simd_available()
            || a.len() < self.config.min_data_size
        {
            return Ok(a.dot(a).sqrt());
        }

        let norm_squared = self.dot_product_simd(a, a)?;
        Ok(norm_squared.sqrt())
    }

    // Private implementation methods

    fn euclidean_distance_standard<F>(&self, a: &Array1<F>, b: &Array1<F>) -> Result<F>
    where
        F: Float + std::fmt::Debug + 'static,
    {
        let diff = a - b;
        Ok(diff.dot(&diff).sqrt())
    }

    fn manhattan_distance_standard<F>(&self, a: &Array1<F>, b: &Array1<F>) -> Result<F>
    where
        F: Float + std::fmt::Debug + std::iter::Sum,
    {
        let sum: F = a.iter().zip(b.iter()).map(|(x, y)| (*x - *y).abs()).sum();
        Ok(sum)
    }

    fn cosine_distance_standard<F>(&self, a: &Array1<F>, b: &Array1<F>) -> Result<F>
    where
        F: Float + std::fmt::Debug + Zero + One + 'static,
    {
        let dot_product = a.dot(b);
        let norm_a = a.dot(a).sqrt();
        let norm_b = b.dot(b).sqrt();

        if norm_a == F::zero() || norm_b == F::zero() {
            return Ok(F::one());
        }

        let cosine_similarity = dot_product / (norm_a * norm_b);
        Ok(F::one() - cosine_similarity)
    }
}

impl Default for SimdDistanceMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// SIMD-accelerated statistical computations
pub struct SimdStatistics {
    config: HardwareAccelConfig,
    capabilities: HardwareCapabilities,
}

impl SimdStatistics {
    /// Create new SIMD statistics calculator
    pub fn new() -> Self {
        Self {
            config: HardwareAccelConfig::default(),
            capabilities: HardwareCapabilities::detect(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: HardwareAccelConfig) -> Self {
        Self {
            config,
            capabilities: HardwareCapabilities::detect(),
        }
    }

    /// Compute mean using SIMD
    pub fn mean_simd(&self, data: &Array1<f64>) -> Result<f64> {
        if !self.config.enable_simd
            || !self.capabilities.simd_available()
            || data.len() < self.config.min_data_size
        {
            return Ok(data.mean_or(0.0));
        }

        let sum = self.sum_simd(data)?;
        Ok(sum / data.len() as f64)
    }

    /// Compute variance using SIMD
    pub fn variance_simd(&self, data: &Array1<f64>) -> Result<f64> {
        if data.len() < 2 {
            return Ok(0.0);
        }

        if !self.config.enable_simd
            || !self.capabilities.simd_available()
            || data.len() < self.config.min_data_size
        {
            let mean = data.mean_or(0.0);
            let var =
                data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (data.len() - 1) as f64;
            return Ok(var);
        }

        let mean = self.mean_simd(data)?;
        let sum_squared_diff = self.sum_squared_differences_simd(data, mean)?;
        Ok(sum_squared_diff / (data.len() - 1) as f64)
    }

    /// Compute standard deviation using SIMD
    pub fn std_simd(&self, data: &Array1<f64>) -> Result<f64> {
        let variance = self.variance_simd(data)?;
        Ok(variance.sqrt())
    }

    /// Compute sum using SIMD
    pub fn sum_simd(&self, data: &Array1<f64>) -> Result<f64> {
        if !self.config.enable_simd
            || !self.capabilities.simd_available()
            || data.len() < self.config.min_data_size
        {
            return Ok(data.sum());
        }

        // Use unified SIMD operations for sum
        let sum = f64::simd_sum(&data.view());
        Ok(sum)
    }

    /// Compute sum of squared differences from mean using SIMD
    pub fn sum_squared_differences_simd(&self, data: &Array1<f64>, mean: f64) -> Result<f64> {
        if !self.config.enable_simd
            || !self.capabilities.simd_available()
            || data.len() < self.config.min_data_size
        {
            let sum = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>();
            return Ok(sum);
        }

        // Use unified SIMD operations for sum of squared differences
        let mean_array = Array1::from_elem(data.len(), mean);
        let diff = f64::simd_sub(&data.view(), &mean_array.view());
        let squared = f64::simd_mul(&diff.view(), &diff.view());
        let sum = f64::simd_sum(&squared.view());
        Ok(sum)
    }
}

impl Default for SimdStatistics {
    fn default() -> Self {
        Self::new()
    }
}

/// Matrix operations with hardware acceleration
pub struct HardwareAcceleratedMatrix {
    config: HardwareAccelConfig,
    capabilities: HardwareCapabilities,
}

impl HardwareAcceleratedMatrix {
    /// Create new hardware-accelerated matrix operations
    pub fn new() -> Self {
        Self {
            config: HardwareAccelConfig::default(),
            capabilities: HardwareCapabilities::detect(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: HardwareAccelConfig) -> Self {
        Self {
            config,
            capabilities: HardwareCapabilities::detect(),
        }
    }

    /// Matrix-vector multiplication with hardware acceleration
    pub fn matvec_accelerated(
        &self,
        matrix: &Array2<f64>,
        vector: &Array1<f64>,
    ) -> Result<Array1<f64>> {
        let (rows, cols) = matrix.dim();
        if cols != vector.len() {
            return Err(MetricsError::InvalidInput(
                "Matrix columns must match vector length".to_string(),
            ));
        }

        if !self.config.enable_simd
            || !self.capabilities.simd_available()
            || rows * cols < self.config.min_data_size
        {
            // Fallback to standard implementation
            return Ok(matrix.dot(vector));
        }

        let mut result = Array1::zeros(rows);
        let simd_distances = SimdDistanceMetrics::with_config(self.config.clone());

        // Compute dot product for each row
        for (i, row) in matrix.rows().into_iter().enumerate() {
            result[i] = simd_distances.dot_product_simd(&row.to_owned(), vector)?;
        }

        Ok(result)
    }

    /// Pairwise distance matrix computation with hardware acceleration
    pub fn pairwise_distances_accelerated(
        &self,
        data: &Array2<f64>,
        metric: &str,
    ) -> Result<Array2<f64>> {
        let n_samples = data.nrows();
        let mut distances = Array2::zeros((n_samples, n_samples));

        if !self.config.enable_simd || !self.capabilities.simd_available() {
            // Fallback to standard implementation
            return self.pairwise_distances_standard(data, metric);
        }

        let simd_distances = SimdDistanceMetrics::with_config(self.config.clone());

        for i in 0..n_samples {
            for j in (i + 1)..n_samples {
                let row_i = data.row(i).to_owned();
                let row_j = data.row(j).to_owned();

                let distance = match metric {
                    "euclidean" => simd_distances.euclidean_distance_simd(&row_i, &row_j)?,
                    "manhattan" => simd_distances.manhattan_distance_simd(&row_i, &row_j)?,
                    "cosine" => simd_distances.cosine_distance_simd(&row_i, &row_j)?,
                    _ => {
                        return Err(MetricsError::InvalidInput(format!(
                            "Unsupported metric: {}",
                            metric
                        )))
                    }
                };

                distances[[i, j]] = distance;
                distances[[j, i]] = distance; // Symmetric
            }
        }

        Ok(distances)
    }

    /// Correlation matrix computation with hardware acceleration
    pub fn correlation_matrix_accelerated(&self, data: &Array2<f64>) -> Result<Array2<f64>> {
        let (_n_samples, n_features) = data.dim();
        let mut correlation_matrix = Array2::zeros((n_features, n_features));

        if !self.config.enable_simd || !self.capabilities.simd_available() {
            return self.correlation_matrix_standard(data);
        }

        let simd_stats = SimdStatistics::with_config(self.config.clone());

        // Compute means for each feature
        let mut means = Array1::zeros(n_features);
        for j in 0..n_features {
            let column = data.column(j).to_owned();
            means[j] = simd_stats.mean_simd(&column)?;
        }

        // Compute correlations
        for i in 0..n_features {
            for j in (i + 1)..n_features {
                let col_i = data.column(i).to_owned();
                let col_j = data.column(j).to_owned();

                let correlation =
                    self.compute_correlation_simd(&col_i, &col_j, means[i], means[j], &simd_stats)?;

                correlation_matrix[[i, j]] = correlation;
                correlation_matrix[[j, i]] = correlation;
            }
            // Diagonal elements are 1.0
            correlation_matrix[[i, i]] = 1.0;
        }

        Ok(correlation_matrix)
    }

    // Private helper methods

    fn pairwise_distances_standard(&self, data: &Array2<f64>, metric: &str) -> Result<Array2<f64>> {
        let n_samples = data.nrows();
        let mut distances = Array2::zeros((n_samples, n_samples));

        for i in 0..n_samples {
            for j in (i + 1)..n_samples {
                let row_i = data.row(i);
                let row_j = data.row(j);

                let distance = match metric {
                    "euclidean" => {
                        let diff = &row_i.to_owned() - &row_j.to_owned();
                        diff.dot(&diff).sqrt()
                    }
                    "manhattan" => row_i
                        .iter()
                        .zip(row_j.iter())
                        .map(|(a, b)| (a - b).abs())
                        .sum(),
                    "cosine" => {
                        let dot_product = row_i.dot(&row_j);
                        let norm_i = row_i.dot(&row_i).sqrt();
                        let norm_j = row_j.dot(&row_j).sqrt();
                        if norm_i == 0.0 || norm_j == 0.0 {
                            1.0
                        } else {
                            1.0 - dot_product / (norm_i * norm_j)
                        }
                    }
                    _ => {
                        return Err(MetricsError::InvalidInput(format!(
                            "Unsupported metric: {}",
                            metric
                        )))
                    }
                };

                distances[[i, j]] = distance;
                distances[[j, i]] = distance;
            }
        }

        Ok(distances)
    }

    fn correlation_matrix_standard(&self, data: &Array2<f64>) -> Result<Array2<f64>> {
        let (n_samples, n_features) = data.dim();
        let mut correlation_matrix = Array2::zeros((n_features, n_features));

        // Compute means
        let means: Array1<f64> = data.mean_axis(Axis(0)).unwrap();

        for i in 0..n_features {
            for j in (i + 1)..n_features {
                let col_i = data.column(i);
                let col_j = data.column(j);

                let mean_i = means[i];
                let mean_j = means[j];

                // Compute covariance and standard deviations
                let mut cov = 0.0;
                let mut var_i = 0.0;
                let mut var_j = 0.0;

                for k in 0..n_samples {
                    let diff_i = col_i[k] - mean_i;
                    let diff_j = col_j[k] - mean_j;
                    cov += diff_i * diff_j;
                    var_i += diff_i * diff_i;
                    var_j += diff_j * diff_j;
                }

                let correlation = if var_i > 1e-10 && var_j > 1e-10 {
                    cov / (var_i * var_j).sqrt()
                } else {
                    0.0
                };

                correlation_matrix[[i, j]] = correlation;
                correlation_matrix[[j, i]] = correlation;
            }
            correlation_matrix[[i, i]] = 1.0;
        }

        Ok(correlation_matrix)
    }

    fn compute_correlation_simd(
        &self,
        col_i: &Array1<f64>,
        col_j: &Array1<f64>,
        mean_i: f64,
        mean_j: f64,
        simd_stats: &SimdStatistics,
    ) -> Result<f64> {
        // Compute centered vectors
        let centered_i = col_i.mapv(|x| x - mean_i);
        let centered_j = col_j.mapv(|x| x - mean_j);

        // Compute covariance using SIMD dot product
        let simd_distances = SimdDistanceMetrics::with_config(self.config.clone());
        let covariance = simd_distances.dot_product_simd(&centered_i, &centered_j)?;

        // Compute standard deviations using SIMD
        let var_i = simd_stats.sum_squared_differences_simd(col_i, mean_i)?;
        let var_j = simd_stats.sum_squared_differences_simd(col_j, mean_j)?;

        if var_i > 1e-10 && var_j > 1e-10 {
            Ok(covariance / (var_i * var_j).sqrt())
        } else {
            Ok(0.0)
        }
    }
}

impl Default for HardwareAcceleratedMatrix {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array;

    #[test]
    fn test_hardware_capabilities() {
        let capabilities = HardwareCapabilities::detect();
        println!("Hardware capabilities: {:?}", capabilities);

        // Just verify detection works - actual capabilities depend on hardware
        assert!(
            capabilities.optimal_vector_width() != VectorWidth::V512 || capabilities.has_avx512f
        );
    }

    #[test]
    fn test_simd_distance_metrics() {
        let metrics = SimdDistanceMetrics::new();
        let a = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let b = Array1::from_vec(vec![2.0, 3.0, 4.0, 5.0, 6.0]);

        // Test Euclidean distance
        let euclidean = metrics.euclidean_distance_simd(&a, &b).unwrap();
        let expected_euclidean = (5.0_f64).sqrt(); // sqrt(5 * 1^2)
        assert!((euclidean - expected_euclidean).abs() < 1e-10);

        // Test Manhattan distance
        let manhattan = metrics.manhattan_distance_simd(&a, &b).unwrap();
        assert!((manhattan - 5.0).abs() < 1e-10);

        // Test dot product
        let dot = metrics.dot_product_simd(&a, &b).unwrap();
        let expected_dot = 70.0; // 1*2 + 2*3 + 3*4 + 4*5 + 5*6
        assert!((dot - expected_dot).abs() < 1e-10);
    }

    #[test]
    fn test_simd_statistics() {
        let stats = SimdStatistics::new();
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        // Test mean
        let mean = stats.mean_simd(&data).unwrap();
        assert!((mean - 3.0).abs() < 1e-10);

        // Test sum
        let sum = stats.sum_simd(&data).unwrap();
        assert!((sum - 15.0).abs() < 1e-10);

        // Test variance
        let variance = stats.variance_simd(&data).unwrap();
        let expected_variance = 2.5; // Sample variance
        assert!((variance - expected_variance).abs() < 1e-10);
    }

    #[test]
    fn test_hardware_accelerated_matrix() {
        let matrix_ops = HardwareAcceleratedMatrix::new();

        // Test matrix-vector multiplication
        let matrix =
            Array2::from_shape_vec((3, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
                .unwrap();
        let vector = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        let result = matrix_ops.matvec_accelerated(&matrix, &vector).unwrap();
        let expected = Array1::from_vec(vec![14.0, 32.0, 50.0]); // 1*1+2*2+3*3, etc.

        for (actual, expected) in result.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-10);
        }

        // Test pairwise distances
        let data = Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0]).unwrap();

        let distances = matrix_ops
            .pairwise_distances_accelerated(&data, "euclidean")
            .unwrap();

        // Check symmetry
        assert!((distances[[0, 1]] - distances[[1, 0]]).abs() < 1e-10);
        assert!((distances[[0, 2]] - distances[[2, 0]]).abs() < 1e-10);
        assert!((distances[[1, 2]] - distances[[2, 1]]).abs() < 1e-10);

        // Check diagonal is zero
        assert!(distances[[0, 0]].abs() < 1e-10);
        assert!(distances[[1, 1]].abs() < 1e-10);
        assert!(distances[[2, 2]].abs() < 1e-10);
    }

    #[test]
    fn test_config_builder() {
        let config = HardwareAccelConfig::new()
            .with_simd_enabled(true)
            .with_gpu_enabled(false)
            .with_min_data_size(500)
            .with_vector_width(VectorWidth::V256)
            .with_gpu_memory_threshold(2048);

        assert!(config.enable_simd);
        assert!(!config.enable_gpu);
        assert_eq!(config.min_data_size, 500);
        assert_eq!(config.gpu_memory_threshold, 2048);
    }
}
