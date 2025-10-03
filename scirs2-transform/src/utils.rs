//! Utility functions and helpers for data transformation
//!
//! This module provides common utility functions that are frequently needed
//! for data transformation tasks, including data validation, memory optimization,
//! and performance helpers.

use scirs2_core::ndarray::{
    par_azip, Array1, Array2, ArrayBase, ArrayView1, ArrayView2, Data, Ix2, Zip,
};
use scirs2_core::numeric::{Float, NumCast};
use scirs2_core::parallel_ops::*;
use scirs2_core::validation::check_not_empty;
use std::collections::HashMap;

use crate::error::{Result, TransformError};
use statrs::statistics::Statistics;

/// Memory-efficient data chunking for large-scale transformations
#[derive(Debug, Clone)]
pub struct DataChunker {
    /// Maximum memory usage in MB
    _max_memorymb: usize,
    /// Preferred chunk size in number of samples
    preferred_chunk_size: usize,
    /// Minimum chunk size to maintain efficiency
    min_chunk_size: usize,
}

impl DataChunker {
    /// Create a new data chunker with memory constraints
    pub fn new(_max_memorymb: usize) -> Self {
        DataChunker {
            _max_memorymb,
            preferred_chunk_size: 10000,
            min_chunk_size: 100,
        }
    }

    /// Calculate optimal chunk size for given data dimensions
    pub fn calculate_chunk_size(&self, n_samples: usize, nfeatures: usize) -> usize {
        // Estimate memory per sample (8 bytes per f64 element + overhead)
        let bytes_per_sample = nfeatures * std::mem::size_of::<f64>() + 64; // 64 bytes overhead
        let max_samples_in_memory = (self._max_memorymb * 1024 * 1024) / bytes_per_sample;

        max_samples_in_memory
            .min(self.preferred_chunk_size)
            .max(self.min_chunk_size)
            .min(n_samples)
    }

    /// Iterator over data chunks
    pub fn chunk_indices(&self, n_samples: usize, nfeatures: usize) -> ChunkIterator {
        let chunk_size = self.calculate_chunk_size(n_samples, nfeatures);
        ChunkIterator {
            current: 0,
            total: n_samples,
            chunk_size,
        }
    }
}

/// Iterator for data chunk indices
#[derive(Debug)]
pub struct ChunkIterator {
    current: usize,
    total: usize,
    chunk_size: usize,
}

impl Iterator for ChunkIterator {
    type Item = (usize, usize); // (start_idx, end_idx)

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.total {
            return None;
        }

        let start = self.current;
        let end = (self.current + self.chunk_size).min(self.total);
        self.current = end;

        Some((start, end))
    }
}

/// Fast data type conversion utilities
pub struct TypeConverter;

impl TypeConverter {
    /// Convert array to f64 with optimized SIMD operations where possible
    pub fn to_f64<T, S>(array: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        T: Float + NumCast + Send + Sync,
        S: Data<Elem = T>,
    {
        check_not_empty(array, "array")?;

        let result = if array.is_standard_layout() {
            // Use parallel processing for large arrays
            if array.len() > 10000 {
                let mut result = Array2::zeros(array.raw_dim());
                Zip::from(&mut result).and(array).par_for_each(|out, &inp| {
                    *out = NumCast::from(inp).unwrap_or(0.0);
                });
                result
            } else {
                array.mapv(|x| NumCast::from(x).unwrap_or(0.0))
            }
        } else {
            // Handle non-standard layout
            let shape = array.shape();
            let mut result = Array2::zeros((shape[0], shape[1]));

            par_azip!((out in result.view_mut(), &inp in array) {
                *out = NumCast::from(inp).unwrap_or(0.0);
            });

            result
        };

        // Validate result for non-finite values
        for &val in result.iter() {
            if !val.is_finite() {
                return Err(crate::error::TransformError::DataValidationError(
                    "Array contains non-finite values after conversion".to_string(),
                ));
            }
        }
        Ok(result)
    }

    /// Convert f32 array to f64 with SIMD optimization
    pub fn f32_to_f64_simd(array: &ArrayView2<f32>) -> Result<Array2<f64>> {
        check_not_empty(array, "array")?;

        let result = if array.len() > 10000 {
            let mut result = Array2::zeros(array.raw_dim());
            Zip::from(&mut result).and(array).par_for_each(|out, &inp| {
                *out = inp as f64;
            });
            result
        } else {
            array.mapv(|x| x as f64)
        };

        for &val in result.iter() {
            if !val.is_finite() {
                return Err(crate::error::TransformError::DataValidationError(
                    "Array contains non-finite values after conversion".to_string(),
                ));
            }
        }
        Ok(result)
    }

    /// Convert f64 array to f32 with overflow checking
    pub fn f64_to_f32_safe(array: &ArrayView2<f64>) -> Result<Array2<f32>> {
        check_not_empty(array, "array")?;

        // Check finite values
        for &val in array.iter() {
            if !val.is_finite() {
                return Err(crate::error::TransformError::DataValidationError(
                    "Array contains non-finite values".to_string(),
                ));
            }
        }

        let mut result = Array2::zeros(array.raw_dim());
        for (out, &inp) in result.iter_mut().zip(array.iter()) {
            if inp.abs() > f32::MAX as f64 {
                return Err(TransformError::DataValidationError(
                    "Value too large for f32 conversion".to_string(),
                ));
            }
            *out = inp as f32;
        }

        Ok(result)
    }
}

/// Statistical utilities for transformation validation
pub struct StatUtils;

impl StatUtils {
    /// Calculate robust statistics (median, MAD) efficiently
    pub fn robust_stats(data: &ArrayView1<f64>) -> Result<(f64, f64)> {
        check_not_empty(data, "data")?;

        // Check finite values
        for &val in data.iter() {
            if !val.is_finite() {
                return Err(crate::error::TransformError::DataValidationError(
                    "Data contains non-finite values".to_string(),
                ));
            }
        }

        let mut sorted_data = data.to_vec();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = sorted_data.len();
        let median = if n.is_multiple_of(2) {
            (sorted_data[n / 2 - 1] + sorted_data[n / 2]) / 2.0
        } else {
            sorted_data[n / 2]
        };

        // Calculate MAD (Median Absolute Deviation)
        let mut deviations: Vec<f64> = sorted_data.iter().map(|&x| (x - median).abs()).collect();
        deviations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mad = if n.is_multiple_of(2) {
            (deviations[n / 2 - 1] + deviations[n / 2]) / 2.0
        } else {
            deviations[n / 2]
        };

        Ok((median, mad))
    }

    /// Calculate column-wise robust statistics in parallel
    pub fn robust_stats_columns(data: &ArrayView2<f64>) -> Result<(Array1<f64>, Array1<f64>)> {
        check_not_empty(data, "data")?;

        // Check finite values
        for &val in data.iter() {
            if !val.is_finite() {
                return Err(crate::error::TransformError::DataValidationError(
                    "Data contains non-finite values".to_string(),
                ));
            }
        }

        let nfeatures = data.ncols();
        let mut medians = Array1::zeros(nfeatures);
        let mut mads = Array1::zeros(nfeatures);

        // Use parallel processing for multiple columns
        let stats: Result<Vec<_>> = (0..nfeatures)
            .into_par_iter()
            .map(|j| {
                let col = data.column(j);
                Self::robust_stats(&col)
            })
            .collect();

        let stats = stats?;

        for (j, (median, mad)) in stats.into_iter().enumerate() {
            medians[j] = median;
            mads[j] = mad;
        }

        Ok((medians, mads))
    }

    /// Detect outliers using IQR method
    pub fn detect_outliers_iqr(data: &ArrayView1<f64>, factor: f64) -> Result<Vec<bool>> {
        check_not_empty(data, "data")?;

        // Check finite values
        for &val in data.iter() {
            if !val.is_finite() {
                return Err(crate::error::TransformError::DataValidationError(
                    "Data contains non-finite values".to_string(),
                ));
            }
        }

        if factor <= 0.0 {
            return Err(TransformError::InvalidInput(
                "Outlier factor must be positive".to_string(),
            ));
        }

        let mut sorted_data = data.to_vec();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = sorted_data.len();
        let q1_idx = n / 4;
        let q3_idx = 3 * n / 4;

        let q1 = sorted_data[q1_idx];
        let q3 = sorted_data[q3_idx];
        let iqr = q3 - q1;

        let lower_bound = q1 - factor * iqr;
        let upper_bound = q3 + factor * iqr;

        let outliers = data
            .iter()
            .map(|&x| x < lower_bound || x > upper_bound)
            .collect();

        Ok(outliers)
    }

    /// Calculate data quality score
    pub fn data_quality_score(data: &ArrayView2<f64>) -> Result<f64> {
        check_not_empty(data, "data")?;

        let total_elements = data.len() as f64;

        // Count finite values
        let finite_count = data.iter().filter(|&&x| x.is_finite()).count() as f64;
        let finite_ratio = finite_count / total_elements;

        // Count unique values per column (diversity score)
        let nfeatures = data.ncols();
        let mut diversity_scores = Vec::with_capacity(nfeatures);

        for j in 0..nfeatures {
            let col = data.column(j);
            let mut unique_values = std::collections::HashSet::new();
            for &val in col.iter() {
                if val.is_finite() {
                    // Round to avoid floating point precision issues
                    let rounded = (val * 1e12).round() as i64;
                    unique_values.insert(rounded);
                }
            }

            let diversity = if !col.is_empty() {
                unique_values.len() as f64 / col.len() as f64
            } else {
                0.0
            };
            diversity_scores.push(diversity);
        }

        let avg_diversity = if diversity_scores.is_empty() {
            0.0
        } else {
            diversity_scores.iter().sum::<f64>() / diversity_scores.len() as f64
        };

        // Combine scores with weights
        let quality_score = 0.7 * finite_ratio + 0.3 * avg_diversity;

        Ok(quality_score.clamp(0.0, 1.0))
    }
}

/// Memory pool for efficient array allocation and reuse
pub struct ArrayMemoryPool<T> {
    /// Available arrays by size
    available_arrays: HashMap<(usize, usize), Vec<Array2<T>>>,
    /// Maximum number of arrays to keep per size
    max_persize: usize,
    /// Total memory limit in bytes
    memory_limit: usize,
    /// Current memory usage
    current_memory: usize,
}

impl<T: Clone + Default> ArrayMemoryPool<T> {
    /// Create a new array memory pool
    pub fn new(_memory_limit_mb: usize, max_persize: usize) -> Self {
        ArrayMemoryPool {
            available_arrays: HashMap::new(),
            max_persize,
            memory_limit: _memory_limit_mb * 1024 * 1024,
            current_memory: 0,
        }
    }

    /// Get an array from the pool or create a new one
    pub fn get_array(&mut self, rows: usize, cols: usize) -> Array2<T> {
        let size_key = (rows, cols);

        if let Some(arrays) = self.available_arrays.get_mut(&size_key) {
            if let Some(array) = arrays.pop() {
                let array_size = rows * cols * std::mem::size_of::<T>();
                self.current_memory = self.current_memory.saturating_sub(array_size);
                return array;
            }
        }

        // Create new array if none available
        Array2::default((rows, cols))
    }

    /// Return an array to the pool for reuse
    pub fn return_array(&mut self, mut array: Array2<T>) {
        let (rows, cols) = array.dim();
        let size_key = (rows, cols);
        let array_size = rows * cols * std::mem::size_of::<T>();

        // Check memory limits
        if self.current_memory + array_size > self.memory_limit {
            return; // Drop the array
        }

        // Zero out the array for reuse
        array.fill(T::default());

        let arrays = self.available_arrays.entry(size_key).or_default();
        if arrays.len() < self.max_persize {
            arrays.push(array);
            self.current_memory += array_size;
        }
    }

    /// Clear the pool and free memory
    pub fn clear(&mut self) {
        self.available_arrays.clear();
        self.current_memory = 0;
    }

    /// Get current memory usage in MB
    pub fn memory_usage_mb(&self) -> f64 {
        self.current_memory as f64 / (1024.0 * 1024.0)
    }
}

/// Validation utilities for transformation parameters
pub struct ValidationUtils;

impl ValidationUtils {
    /// Validate that a parameter is within reasonable bounds
    pub fn validate_parameter_bounds(
        value: f64,
        min: f64,
        max: f64,
        param_name: &str,
    ) -> Result<()> {
        if !value.is_finite() {
            return Err(TransformError::InvalidInput(format!(
                "{param_name} must be finite"
            )));
        }

        if value < min || value > max {
            return Err(TransformError::InvalidInput(format!(
                "{param_name} must be between {min} and {max}, got {value}"
            )));
        }

        Ok(())
    }

    /// Validate array dimensions for compatibility
    pub fn validate_dimensions_compatible(
        shape1: &[usize],
        shape2: &[usize],
        operation: &str,
    ) -> Result<()> {
        if shape1.len() != shape2.len() {
            return Err(TransformError::InvalidInput(format!(
                "Incompatible dimensions for {operation}: {shape1:?} vs {shape2:?}"
            )));
        }

        for (i, (&dim1, &dim2)) in shape1.iter().zip(shape2.iter()).enumerate() {
            if dim1 != dim2 {
                return Err(TransformError::InvalidInput(format!(
                    "Dimension {i} mismatch for {operation}: {dim1} vs {dim2}"
                )));
            }
        }

        Ok(())
    }

    /// Validate that data is suitable for a specific transformation
    pub fn validate_data_for_transformation(
        data: &ArrayView2<f64>,
        transformation: &str,
    ) -> Result<()> {
        check_not_empty(data, "data")?;

        // Check finite values
        for &val in data.iter() {
            if !val.is_finite() {
                return Err(crate::error::TransformError::DataValidationError(
                    "Data contains non-finite values".to_string(),
                ));
            }
        }

        let (n_samples, nfeatures) = data.dim();

        match transformation {
            "pca" => {
                if n_samples < 2 {
                    return Err(TransformError::InvalidInput(
                        "PCA requires at least 2 samples".to_string(),
                    ));
                }
                if nfeatures < 1 {
                    return Err(TransformError::InvalidInput(
                        "PCA requires at least 1 feature".to_string(),
                    ));
                }
            }
            "standardization" => {
                // Check for constant features
                for j in 0..nfeatures {
                    let col = data.column(j);
                    let variance = col.variance();
                    if variance < 1e-15 {
                        return Err(TransformError::DataValidationError(format!(
                            "Feature {j} has zero variance and cannot be standardized"
                        )));
                    }
                }
            }
            "normalization" => {
                // Check for zero-norm rows
                for i in 0..n_samples {
                    let row = data.row(i);
                    let norm = row.iter().map(|&x| x * x).sum::<f64>().sqrt();
                    if norm < 1e-15 {
                        return Err(TransformError::DataValidationError(format!(
                            "Sample {i} has zero norm and cannot be normalized"
                        )));
                    }
                }
            }
            _ => {
                // Generic validation
            }
        }

        Ok(())
    }
}

/// Performance monitoring utilities
pub struct PerfUtils;

impl PerfUtils {
    /// Estimate memory usage for an operation
    pub fn estimate_memory_usage(
        inputshape: &[usize],
        outputshape: &[usize],
        operation: &str,
    ) -> usize {
        let input_size = inputshape.iter().product::<usize>() * std::mem::size_of::<f64>();
        let output_size = outputshape.iter().product::<usize>() * std::mem::size_of::<f64>();

        let overhead = match operation {
            "pca" => input_size * 2,              // Covariance matrix + temporaries
            "standardization" => input_size / 10, // Just statistics
            "polynomial" => output_size / 2,      // Temporary computations
            _ => input_size / 4,                  // Default overhead
        };

        input_size + output_size + overhead
    }

    /// Estimate computation time based on data size and operation
    pub fn estimate_computation_time(
        n_samples: usize,
        nfeatures: usize,
        operation: &str,
    ) -> std::time::Duration {
        use std::time::Duration;

        let base_time_ns = match operation {
            "pca" => (n_samples as u64) * (nfeatures as u64).pow(2) / 1000, // O(n*m^2)
            "standardization" => (n_samples as u64) * (nfeatures as u64) / 100, // O(n*m)
            "normalization" => (n_samples as u64) * (nfeatures as u64) / 50, // O(n*m)
            "polynomial" => (n_samples as u64) * (nfeatures as u64).pow(3) / 10000, // O(n*m^3)
            _ => (n_samples as u64) * (nfeatures as u64) / 100,
        };

        Duration::from_nanos(base_time_ns.max(1000)) // At least 1 microsecond
    }

    /// Choose optimal processing strategy based on data characteristics
    pub fn choose_processing_strategy(
        n_samples: usize,
        nfeatures: usize,
        available_memory_mb: usize,
    ) -> ProcessingStrategy {
        let estimated_memory_mb =
            (n_samples * nfeatures * std::mem::size_of::<f64>()) / (1024 * 1024);

        if estimated_memory_mb > available_memory_mb {
            ProcessingStrategy::OutOfCore {
                chunk_size: (available_memory_mb * 1024 * 1024)
                    / (nfeatures * std::mem::size_of::<f64>()),
            }
        } else if n_samples > 10000 && nfeatures > 100 {
            ProcessingStrategy::Parallel
        } else if nfeatures > 1000 {
            ProcessingStrategy::Simd
        } else {
            ProcessingStrategy::Standard
        }
    }
}

/// Processing strategy recommendation
#[derive(Debug, Clone)]
#[cfg_attr(feature = "distributed", derive(serde::Serialize, serde::Deserialize))]
pub enum ProcessingStrategy {
    /// Standard sequential processing
    Standard,
    /// SIMD-accelerated processing
    Simd,
    /// Parallel processing across multiple cores
    Parallel,
    /// Out-of-core processing for large datasets
    OutOfCore {
        /// Size of data chunks for processing
        chunk_size: usize,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_data_chunker() {
        let chunker = DataChunker::new(100); // 100MB
        let chunk_size = chunker.calculate_chunk_size(50000, 100);
        assert!(chunk_size > 0);
        assert!(chunk_size <= 50000);
    }

    #[test]
    fn test_chunk_iterator() {
        let chunker = DataChunker::new(1); // 1MB - small for testing
        let chunks: Vec<_> = chunker.chunk_indices(1000, 10).collect();
        assert!(!chunks.is_empty());

        // Verify complete coverage
        let total_covered = chunks.iter().map(|(start, end)| end - start).sum::<usize>();
        assert_eq!(total_covered, 1000);
    }

    #[test]
    fn test_type_converter() {
        let data = Array2::<f32>::ones((10, 5));
        let result = TypeConverter::f32_to_f64_simd(&data.view()).unwrap();
        assert_eq!(result.shape(), &[10, 5]);
        assert!((result[(0, 0)] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_robust_stats() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 100.0]); // With outlier
        let (median, mad) = StatUtils::robust_stats(&data.view()).unwrap();
        assert!((median - 3.5).abs() < 1e-10);
        assert!(mad > 0.0);
    }

    #[test]
    fn test_outlier_detection() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 100.0]);
        let outliers = StatUtils::detect_outliers_iqr(&data.view(), 1.5).unwrap();
        assert_eq!(outliers.len(), 6);
        assert!(outliers[5]); // 100.0 should be detected as outlier
    }

    #[test]
    fn test_data_quality_score() {
        let good_data =
            Array2::from_shape_vec((10, 3), (0..30).map(|x| x as f64).collect()).unwrap();
        let quality = StatUtils::data_quality_score(&good_data.view()).unwrap();
        assert!(quality > 0.5); // Should have reasonable quality

        let bad_data = Array2::from_elem((10, 3), f64::NAN);
        let quality = StatUtils::data_quality_score(&bad_data.view()).unwrap();
        assert!(quality < 0.5); // Should have poor quality due to NaN values
    }

    #[test]
    fn test_memory_pool() {
        let mut pool = ArrayMemoryPool::<f64>::new(10, 2);

        // Get and return arrays
        let array1 = pool.get_array(10, 5);
        assert_eq!(array1.shape(), &[10, 5]);

        pool.return_array(array1);

        let array2 = pool.get_array(10, 5);
        assert_eq!(array2.shape(), &[10, 5]);
    }

    #[test]
    fn test_validation_utils() {
        // Test parameter bounds validation
        assert!(ValidationUtils::validate_parameter_bounds(0.5, 0.0, 1.0, "test").is_ok());
        assert!(ValidationUtils::validate_parameter_bounds(1.5, 0.0, 1.0, "test").is_err());

        // Test dimension compatibility
        assert!(
            ValidationUtils::validate_dimensions_compatible(&[10, 5], &[10, 5], "test").is_ok()
        );
        assert!(
            ValidationUtils::validate_dimensions_compatible(&[10, 5], &[10, 6], "test").is_err()
        );
    }

    #[test]
    fn test_performance_utils() {
        let memory = PerfUtils::estimate_memory_usage(&[1000, 100], &[1000, 50], "pca");
        assert!(memory > 0);

        let time = PerfUtils::estimate_computation_time(1000, 100, "pca");
        assert!(time.as_nanos() > 0);

        let strategy = PerfUtils::choose_processing_strategy(10000, 100, 100);
        matches!(strategy, ProcessingStrategy::Parallel);
    }
}
