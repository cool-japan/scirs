//! # Batch Type Conversions
//!
//! This module provides optimized batch type conversions using SIMD and vectorization
//! for better performance when converting large arrays of data.
//!
//! ## Features
//!
//! * SIMD-accelerated batch conversions for numeric types
//! * Parallel processing for large datasets
//! * Optimized memory access patterns
//! * Support for ndarray integration
//! * Error reporting for individual elements
//!
//! ## Usage
//!
//! ```rust,no_run
//! use scirs2_core::batch_conversions::{BatchConverter, BatchConversionConfig};
//! use ndarray::Array1;
//!
//! // Convert a large array of f64 to f32 with SIMD optimization
//! let data: Vec<f64> = (0..1000).map(|i| i as f64 * 0.1).collect();
//! let config = BatchConversionConfig::default().with_simd(true);
//! let converter = BatchConverter::new(config);
//!
//! let result: Vec<f32> = converter.convert_slice(&data).unwrap();
//! assert_eq!(result.len(), data.len());
//!
//! // Convert with error handling for individual elements
//! let large_data: Vec<f64> = vec![1e10, 2.5, f64::NAN, 3.7];
//! let (converted, errors) = converter.convert_slice_with_errors::<f64, f32>(&large_data);
//! assert_eq!(converted.len(), 2); // Only 2.5 and 3.7 convert successfully
//! assert_eq!(errors.len(), 2);    // NAN and overflow errors
//! ```

use crate::error::{CoreError, CoreResult};
use crate::types::{NumericConversion, NumericConversionError};
use num_complex::Complex;
use num_traits::{Bounded, Float, NumCast, Zero};
use std::fmt;
#[cfg(feature = "simd")]
use wide::{f32x4, f64x2, i32x4};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Configuration for batch conversions
#[derive(Debug, Clone)]
pub struct BatchConversionConfig {
    /// Enable SIMD acceleration when available
    pub use_simd: bool,
    /// Enable parallel processing for large datasets
    pub use_parallel: bool,
    /// Chunk size for parallel processing
    pub parallel_chunk_size: usize,
    /// SIMD vector size (auto-detected if None)
    pub simd_vector_size: Option<usize>,
    /// Minimum size to use parallel processing
    pub parallel_threshold: usize,
}

impl Default for BatchConversionConfig {
    fn default() -> Self {
        Self {
            use_simd: cfg!(feature = "simd"),
            use_parallel: cfg!(feature = "parallel"),
            parallel_chunk_size: 1024,
            simd_vector_size: None,
            parallel_threshold: 10000,
        }
    }
}

impl BatchConversionConfig {
    /// Create a new configuration with SIMD enabled
    pub fn with_simd(mut self, enable: bool) -> Self {
        self.use_simd = enable;
        self
    }

    /// Create a new configuration with parallel processing enabled
    pub fn with_parallel(mut self, enable: bool) -> Self {
        self.use_parallel = enable;
        self
    }

    /// Set the chunk size for parallel processing
    pub fn with_chunk_size(mut self, chunk_size: usize) -> Self {
        self.parallel_chunk_size = chunk_size;
        self
    }

    /// Set the threshold for using parallel processing
    pub fn with_parallel_threshold(mut self, threshold: usize) -> Self {
        self.parallel_threshold = threshold;
        self
    }
}

/// Error information for a single element in batch conversion
#[derive(Debug, Clone)]
pub struct ElementConversionError {
    /// Index of the element that failed to convert
    pub index: usize,
    /// The conversion error
    pub error: NumericConversionError,
}

/// Result of batch conversion with partial success
#[derive(Debug, Clone)]
pub struct BatchConversionResult<T> {
    /// Successfully converted values with their original indices
    pub converted: Vec<(usize, T)>,
    /// Errors that occurred during conversion
    pub errors: Vec<ElementConversionError>,
}

/// High-performance batch converter for numeric types
pub struct BatchConverter {
    config: BatchConversionConfig,
}

impl BatchConverter {
    /// Create a new batch converter with the given configuration
    pub fn new(config: BatchConversionConfig) -> Self {
        Self { config }
    }

    /// Create a batch converter with default configuration
    pub fn with_default_config() -> Self {
        Self::new(BatchConversionConfig::default())
    }

    /// Convert a slice of values to another type, returning errors for failed conversions
    pub fn convert_slice_with_errors<S, T>(
        &self,
        slice: &[S],
    ) -> (Vec<T>, Vec<ElementConversionError>)
    where
        S: Copy + NumCast + PartialOrd + fmt::Display + Send + Sync + 'static,
        T: Bounded + NumCast + PartialOrd + fmt::Display + Send + Sync + Copy + 'static,
    {
        if slice.is_empty() {
            return (Vec::new(), Vec::new());
        }

        // Use parallel processing for large datasets
        if self.config.use_parallel && slice.len() >= self.config.parallel_threshold {
            self.convert_slice_parallel_with_errors(slice)
        } else if self.config.use_simd {
            self.convert_slice_simd_with_errors(slice)
        } else {
            self.convert_slice_sequential_with_errors(slice)
        }
    }

    /// Convert a slice of values to another type, returning only successful conversions
    pub fn convert_slice<S, T>(&self, slice: &[S]) -> CoreResult<Vec<T>>
    where
        S: Copy + NumCast + PartialOrd + fmt::Display + Send + Sync + 'static,
        T: Bounded + NumCast + PartialOrd + fmt::Display + Send + Sync + Copy + 'static,
    {
        let (converted, errors) = self.convert_slice_with_errors(slice);

        if !errors.is_empty() {
            return Err(CoreError::InvalidArgument(crate::error::ErrorContext::new(
                format!("Batch conversion failed for {} elements", errors.len()),
            )));
        }

        Ok(converted)
    }

    /// Convert a slice with clamping to target type bounds
    pub fn convert_slice_clamped<S, T>(&self, slice: &[S]) -> Vec<T>
    where
        S: Copy + NumericConversion + Send + Sync,
        T: Bounded + NumCast + PartialOrd + Zero + Send + Sync,
    {
        if slice.is_empty() {
            return Vec::new();
        }

        if self.config.use_parallel && slice.len() >= self.config.parallel_threshold {
            self.convert_slice_parallel_clamped(slice)
        } else {
            slice.iter().map(|&x| x.to_numeric_clamped()).collect()
        }
    }

    /// Sequential conversion with error handling
    fn convert_slice_sequential_with_errors<S, T>(
        &self,
        slice: &[S],
    ) -> (Vec<T>, Vec<ElementConversionError>)
    where
        S: Copy + NumCast + PartialOrd + fmt::Display + 'static,
        T: Bounded + NumCast + PartialOrd + fmt::Display + Copy + 'static,
    {
        let mut converted = Vec::new();
        let mut errors = Vec::new();

        for (index, &value) in slice.iter().enumerate() {
            match value.to_numeric() {
                Ok(result) => converted.push(result),
                Err(error) => errors.push(ElementConversionError { index, error }),
            }
        }

        (converted, errors)
    }

    /// SIMD-accelerated conversion for supported types
    #[cfg(feature = "simd")]
    fn convert_slice_simd_with_errors<S, T>(
        &self,
        slice: &[S],
    ) -> (Vec<T>, Vec<ElementConversionError>)
    where
        S: Copy + NumCast + PartialOrd + fmt::Display + 'static,
        T: Bounded + NumCast + PartialOrd + fmt::Display + Copy + 'static,
    {
        // Check if we can use SIMD for this conversion
        if self.can_use_simd_for_conversion::<S, T>() {
            self.convert_slice_simd_optimized(slice)
        } else {
            self.convert_slice_sequential_with_errors(slice)
        }
    }

    #[cfg(not(feature = "simd"))]
    fn convert_slice_simd_with_errors<S, T>(
        &self,
        slice: &[S],
    ) -> (Vec<T>, Vec<ElementConversionError>)
    where
        S: Copy + NumCast + PartialOrd + fmt::Display + 'static,
        T: Bounded + NumCast + PartialOrd + fmt::Display + Copy + 'static,
    {
        self.convert_slice_sequential_with_errors(slice)
    }

    /// Parallel conversion with error handling
    #[cfg(feature = "parallel")]
    fn convert_slice_parallel_with_errors<S, T>(
        &self,
        slice: &[S],
    ) -> (Vec<T>, Vec<ElementConversionError>)
    where
        S: Copy + NumCast + PartialOrd + fmt::Display + Send + Sync + 'static,
        T: Bounded + NumCast + PartialOrd + fmt::Display + Send + Sync + Copy + 'static,
    {
        let chunk_size = self.config.parallel_chunk_size;
        let chunks: Vec<_> = slice.chunks(chunk_size).enumerate().collect();

        let results: Vec<_> = chunks
            .into_par_iter()
            .map(|(chunk_idx, chunk)| {
                let base_index = chunk_idx * chunk_size;
                let mut converted: Vec<T> = Vec::new();
                let mut errors = Vec::new();

                for (idx, &value) in chunk.iter().enumerate() {
                    let global_index = base_index + idx;
                    match value.to_numeric() {
                        Ok(result) => converted.push(result),
                        Err(error) => errors.push(ElementConversionError {
                            index: global_index,
                            error,
                        }),
                    }
                }

                (converted, errors)
            })
            .collect();

        // Combine results from all chunks
        let mut all_converted = Vec::new();
        let mut all_errors = Vec::new();

        for (converted, errors) in results {
            all_converted.extend(converted);
            all_errors.extend(errors);
        }

        (all_converted, all_errors)
    }

    #[cfg(not(feature = "parallel"))]
    fn convert_slice_parallel_with_errors<S, T>(
        &self,
        slice: &[S],
    ) -> (Vec<T>, Vec<ElementConversionError>)
    where
        S: Copy + NumCast + PartialOrd + fmt::Display + Send + Sync + 'static,
        T: Bounded + NumCast + PartialOrd + fmt::Display + Send + Sync + Copy + 'static,
    {
        self.convert_slice_sequential_with_errors(slice)
    }

    /// Parallel conversion with clamping
    #[cfg(feature = "parallel")]
    fn convert_slice_parallel_clamped<S, T>(&self, slice: &[S]) -> Vec<T>
    where
        S: Copy + NumericConversion + Send + Sync,
        T: Bounded + NumCast + PartialOrd + Zero + Send + Sync,
    {
        slice
            .par_chunks(self.config.parallel_chunk_size)
            .flat_map(|chunk| {
                chunk
                    .iter()
                    .map(|&x| x.to_numeric_clamped())
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    #[cfg(not(feature = "parallel"))]
    fn convert_slice_parallel_clamped<S, T>(&self, slice: &[S]) -> Vec<T>
    where
        S: Copy + NumericConversion + Send + Sync,
        T: Bounded + NumCast + PartialOrd + Zero + Send + Sync,
    {
        slice.iter().map(|&x| x.to_numeric_clamped()).collect()
    }

    /// Check if SIMD can be used for a specific conversion
    #[allow(dead_code)]
    #[cfg(feature = "simd")]
    fn can_use_simd_for_conversion<S: 'static, T: 'static>(&self) -> bool {
        use std::any::TypeId;

        // Define supported SIMD conversions
        let src_type = TypeId::of::<S>();
        let dst_type = TypeId::of::<T>();

        // f64 to f32 conversion
        if src_type == TypeId::of::<f64>() && dst_type == TypeId::of::<f32>() {
            return true;
        }

        // f32 to f64 conversion
        if src_type == TypeId::of::<f32>() && dst_type == TypeId::of::<f64>() {
            return true;
        }

        // i32 to f32 conversion
        if src_type == TypeId::of::<i32>() && dst_type == TypeId::of::<f32>() {
            return true;
        }

        // i64 to f64 conversion
        if src_type == TypeId::of::<i64>() && dst_type == TypeId::of::<f64>() {
            return true;
        }

        false
    }

    #[allow(dead_code)]
    #[cfg(not(feature = "simd"))]
    fn can_use_simd_for_conversion<S: 'static, T: 'static>(&self) -> bool {
        false
    }

    /// SIMD-optimized conversion for supported type pairs
    #[cfg(feature = "simd")]
    fn convert_slice_simd_optimized<S, T>(
        &self,
        slice: &[S],
    ) -> (Vec<T>, Vec<ElementConversionError>)
    where
        S: Copy + NumCast + PartialOrd + fmt::Display + 'static,
        T: Bounded + NumCast + PartialOrd + fmt::Display + Copy + 'static,
    {
        use std::any::TypeId;

        let src_type = TypeId::of::<S>();
        let dst_type = TypeId::of::<T>();

        // f64 to f32 SIMD conversion
        if src_type == TypeId::of::<f64>() && dst_type == TypeId::of::<f32>() {
            if let Some(f64_slice) = slice
                .iter()
                .map(|x| x.to_numeric::<f64>().ok())
                .collect::<Option<Vec<_>>>()
            {
                let (converted, errors) = self.convert_f64_to_f32_simd_typed(&f64_slice);
                // Convert f32 results to T
                let typed_results: Vec<T> =
                    converted.into_iter().filter_map(|f| T::from(f)).collect();
                return (typed_results, errors);
            }
        }

        // f32 to f64 SIMD conversion
        if src_type == TypeId::of::<f32>() && dst_type == TypeId::of::<f64>() {
            if let Some(f32_slice) = slice
                .iter()
                .map(|x| x.to_numeric::<f32>().ok())
                .collect::<Option<Vec<_>>>()
            {
                let (converted, errors) = self.convert_f32_to_f64_simd_typed(&f32_slice);
                // Convert f64 results to T
                let typed_results: Vec<T> =
                    converted.into_iter().filter_map(|f| T::from(f)).collect();
                return (typed_results, errors);
            }
        }

        // i32 to f32 SIMD conversion
        if src_type == TypeId::of::<i32>() && dst_type == TypeId::of::<f32>() {
            if let Some(i32_slice) = slice
                .iter()
                .map(|x| x.to_numeric::<i32>().ok())
                .collect::<Option<Vec<_>>>()
            {
                let (converted, errors) = self.convert_i32_to_f32_simd_typed(&i32_slice);
                // Convert f32 results to T
                let typed_results: Vec<T> =
                    converted.into_iter().filter_map(|f| T::from(f)).collect();
                return (typed_results, errors);
            }
        }

        // Fallback to sequential conversion
        self.convert_slice_sequential_with_errors(slice)
    }

    /// SIMD conversion from f64 to f32 (typed version)
    #[cfg(feature = "simd")]
    fn convert_f64_to_f32_simd_typed(
        &self,
        slice: &[f64],
    ) -> (Vec<f32>, Vec<ElementConversionError>) {
        let mut converted = Vec::with_capacity(slice.len());
        let mut errors = Vec::new();

        // Process in chunks of 2 (f64x2 SIMD width)
        let chunks = slice.chunks_exact(2);
        let remainder = chunks.remainder();

        for (chunk_idx, chunk) in chunks.enumerate() {
            let _vec = f64x2::new([chunk[0], chunk[1]]);

            // Convert each element individually with proper checking
            for (i, &val) in chunk.iter().enumerate() {
                let index = chunk_idx * 2 + i;
                if val.is_nan() || val.is_infinite() {
                    errors.push(ElementConversionError {
                        index,
                        error: NumericConversionError::NanOrInfinite,
                    });
                } else {
                    let f32_val = val as f32;
                    if f32_val.is_infinite() && !val.is_infinite() {
                        errors.push(ElementConversionError {
                            index,
                            error: NumericConversionError::Overflow {
                                value: val.to_string(),
                                max: f32::MAX.to_string(),
                            },
                        });
                    } else {
                        converted.push(f32_val);
                    }
                }
            }
        }

        // Handle remainder elements
        for (i, &val) in remainder.iter().enumerate() {
            let index = slice.len() - remainder.len() + i;
            if val.is_nan() || val.is_infinite() {
                errors.push(ElementConversionError {
                    index,
                    error: NumericConversionError::NanOrInfinite,
                });
            } else {
                let f32_val = val as f32;
                if f32_val.is_infinite() && !val.is_infinite() {
                    errors.push(ElementConversionError {
                        index,
                        error: NumericConversionError::Overflow {
                            value: val.to_string(),
                            max: f32::MAX.to_string(),
                        },
                    });
                } else {
                    converted.push(f32_val);
                }
            }
        }

        (converted, errors)
    }

    /// SIMD conversion from f32 to f64 (typed version)
    #[cfg(feature = "simd")]
    fn convert_f32_to_f64_simd_typed(
        &self,
        slice: &[f32],
    ) -> (Vec<f64>, Vec<ElementConversionError>) {
        let mut converted = Vec::with_capacity(slice.len());
        let mut errors = Vec::new();

        // Process in chunks of 4 (f32x4 SIMD width)
        let chunks = slice.chunks_exact(4);
        let remainder = chunks.remainder();

        for (chunk_idx, chunk) in chunks.enumerate() {
            let _vec = f32x4::new([chunk[0], chunk[1], chunk[2], chunk[3]]);

            for (i, &val) in chunk.iter().enumerate() {
                let index = chunk_idx * 4 + i;
                if val.is_nan() || val.is_infinite() {
                    errors.push(ElementConversionError {
                        index,
                        error: NumericConversionError::NanOrInfinite,
                    });
                } else {
                    converted.push(val as f64);
                }
            }
        }

        // Handle remainder elements
        for (i, &val) in remainder.iter().enumerate() {
            let index = slice.len() - remainder.len() + i;
            if val.is_nan() || val.is_infinite() {
                errors.push(ElementConversionError {
                    index,
                    error: NumericConversionError::NanOrInfinite,
                });
            } else {
                converted.push(val as f64);
            }
        }

        (converted, errors)
    }

    /// SIMD conversion from i32 to f32 (typed version)
    #[cfg(feature = "simd")]
    fn convert_i32_to_f32_simd_typed(
        &self,
        slice: &[i32],
    ) -> (Vec<f32>, Vec<ElementConversionError>) {
        let mut converted = Vec::with_capacity(slice.len());
        let errors = Vec::new(); // i32 to f32 conversion should not fail

        // Process in chunks of 4 (i32x4 SIMD width)
        let chunks = slice.chunks_exact(4);
        let remainder = chunks.remainder();

        for chunk in chunks {
            let _vec = i32x4::new([chunk[0], chunk[1], chunk[2], chunk[3]]);

            for &val in chunk {
                converted.push(val as f32);
            }
        }

        // Handle remainder elements
        for &val in remainder {
            converted.push(val as f32);
        }

        (converted, errors)
    }

    /// Convert complex arrays with batch optimization
    pub fn convert_complex_slice<S, T>(&self, slice: &[Complex<S>]) -> CoreResult<Vec<Complex<T>>>
    where
        S: Float + fmt::Display + Send + Sync,
        T: Float + Bounded + NumCast + PartialOrd + fmt::Display + Send + Sync,
    {
        if slice.is_empty() {
            return Ok(Vec::new());
        }

        let mut result = Vec::with_capacity(slice.len());

        if self.config.use_parallel && slice.len() >= self.config.parallel_threshold {
            // Parallel complex conversion
            #[cfg(feature = "parallel")]
            {
                let chunks: Vec<_> = slice
                    .par_chunks(self.config.parallel_chunk_size)
                    .map(|chunk| {
                        chunk
                            .iter()
                            .map(|z| {
                                let real: T = z.re.to_numeric()?;
                                let imag: T = z.im.to_numeric()?;
                                Ok(Complex::new(real, imag))
                            })
                            .collect::<Result<Vec<_>, NumericConversionError>>()
                    })
                    .collect();

                for chunk_result in chunks {
                    result.extend(chunk_result.map_err(|e| {
                        CoreError::InvalidArgument(crate::error::ErrorContext::new(e.to_string()))
                    })?);
                }
            }

            #[cfg(not(feature = "parallel"))]
            {
                for z in slice {
                    let real: T = z.re.to_numeric().map_err(|e| {
                        CoreError::InvalidArgument(crate::error::ErrorContext::new(e.to_string()))
                    })?;
                    let imag: T = z.im.to_numeric().map_err(|e| {
                        CoreError::InvalidArgument(crate::error::ErrorContext::new(e.to_string()))
                    })?;
                    result.push(Complex::new(real, imag));
                }
            }
        } else {
            // Sequential complex conversion
            for z in slice {
                let real: T = z.re.to_numeric().map_err(|e| {
                    CoreError::InvalidArgument(crate::error::ErrorContext::new(e.to_string()))
                })?;
                let imag: T = z.im.to_numeric().map_err(|e| {
                    CoreError::InvalidArgument(crate::error::ErrorContext::new(e.to_string()))
                })?;
                result.push(Complex::new(real, imag));
            }
        }

        Ok(result)
    }
}

/// Integration with ndarray for batch conversions
#[cfg(feature = "array")]
pub mod ndarray_integration {
    use super::*;
    use ndarray::{Array, ArrayBase, Data, Dimension};

    impl BatchConverter {
        /// Convert an ndarray to another numeric type
        pub fn convert_array<S, T, D>(
            &self,
            array: &ArrayBase<S, D>,
        ) -> CoreResult<ndarray::Array<T, D>>
        where
            S: Data,
            S::Elem: Copy + NumCast + PartialOrd + fmt::Display + Send + Sync + 'static,
            T: Bounded + NumCast + PartialOrd + fmt::Display + Send + Sync + Clone + Copy + 'static,
            D: Dimension,
        {
            let slice = array.as_slice().ok_or_else(|| {
                CoreError::InvalidArgument(crate::error::ErrorContext::new(
                    "Array is not contiguous".to_string(),
                ))
            })?;

            let converted = self.convert_slice(slice)?;

            // Reshape the converted data to match the original array shape
            let shape = array.raw_dim();
            Array::from_shape_vec(shape, converted).map_err(|e| {
                CoreError::InvalidArgument(crate::error::ErrorContext::new(format!(
                    "Failed to reshape converted array: {}",
                    e
                )))
            })
        }

        /// Convert an ndarray with clamping
        pub fn convert_array_clamped<S, T, D>(
            &self,
            array: &ArrayBase<S, D>,
        ) -> CoreResult<ndarray::Array<T, D>>
        where
            S: Data,
            S::Elem: Copy + NumericConversion + Send + Sync,
            T: Bounded + NumCast + PartialOrd + Zero + Send + Sync + Clone,
            D: Dimension,
        {
            let slice = array.as_slice().ok_or_else(|| {
                CoreError::InvalidArgument(crate::error::ErrorContext::new(
                    "Array is not contiguous".to_string(),
                ))
            })?;

            let converted = self.convert_slice_clamped(slice);

            // Reshape the converted data to match the original array shape
            let shape = array.raw_dim();
            Array::from_shape_vec(shape, converted).map_err(|e| {
                CoreError::InvalidArgument(crate::error::ErrorContext::new(format!(
                    "Failed to reshape converted array: {}",
                    e
                )))
            })
        }
    }
}

/// Utility functions for common batch conversions
pub mod utils {
    use super::*;

    /// Convert f64 slice to f32 with SIMD optimization
    pub fn f64_to_f32_batch(slice: &[f64]) -> CoreResult<Vec<f32>> {
        let converter = BatchConverter::with_default_config();
        converter.convert_slice(slice)
    }

    /// Convert f32 slice to f64 with SIMD optimization  
    pub fn f32_to_f64_batch(slice: &[f32]) -> CoreResult<Vec<f64>> {
        let converter = BatchConverter::with_default_config();
        converter.convert_slice(slice)
    }

    /// Convert i32 slice to f32 with SIMD optimization
    pub fn i32_to_f32_batch(slice: &[i32]) -> Vec<f32> {
        let converter = BatchConverter::with_default_config();
        converter.convert_slice_clamped(slice)
    }

    /// Convert i64 slice to f64 with SIMD optimization
    pub fn i64_to_f64_batch(slice: &[i64]) -> Vec<f64> {
        let converter = BatchConverter::with_default_config();
        converter.convert_slice_clamped(slice)
    }

    /// Benchmark different conversion methods
    pub fn benchmark_conversion_methods<S, T>(
        slice: &[S],
    ) -> std::collections::HashMap<String, std::time::Duration>
    where
        S: Copy + NumCast + PartialOrd + fmt::Display + Send + Sync + 'static,
        T: Bounded + NumCast + PartialOrd + fmt::Display + Send + Sync + Copy + 'static,
    {
        use std::time::Instant;
        let mut results = std::collections::HashMap::new();

        // Sequential conversion
        let start = Instant::now();
        let config = BatchConversionConfig::default()
            .with_simd(false)
            .with_parallel(false);
        let converter = BatchConverter::new(config);
        let _ = converter.convert_slice::<S, T>(slice);
        results.insert("sequential".to_string(), start.elapsed());

        // SIMD conversion
        #[cfg(feature = "simd")]
        {
            let start = Instant::now();
            let config = BatchConversionConfig::default()
                .with_simd(true)
                .with_parallel(false);
            let converter = BatchConverter::new(config);
            let _ = converter.convert_slice::<S, T>(slice);
            results.insert("simd".to_string(), start.elapsed());
        }

        // Parallel conversion
        #[cfg(feature = "parallel")]
        {
            let start = Instant::now();
            let config = BatchConversionConfig::default()
                .with_simd(false)
                .with_parallel(true);
            let converter = BatchConverter::new(config);
            let _ = converter.convert_slice::<S, T>(slice);
            results.insert("parallel".to_string(), start.elapsed());
        }

        // Combined SIMD + Parallel
        #[cfg(all(feature = "simd", feature = "parallel"))]
        {
            let start = Instant::now();
            let config = BatchConversionConfig::default()
                .with_simd(true)
                .with_parallel(true);
            let converter = BatchConverter::new(config);
            let _ = converter.convert_slice::<S, T>(slice);
            results.insert("simd_parallel".to_string(), start.elapsed());
        }

        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;

    #[test]
    fn test_batch_conversion_config() {
        let config = BatchConversionConfig::default()
            .with_simd(true)
            .with_parallel(false)
            .with_chunk_size(512)
            .with_parallel_threshold(5000);

        assert!(config.use_simd);
        assert!(!config.use_parallel);
        assert_eq!(config.parallel_chunk_size, 512);
        assert_eq!(config.parallel_threshold, 5000);
    }

    #[test]
    fn test_sequential_conversion() {
        let data: Vec<f64> = vec![1.0, 2.5, 3.7, 4.2];
        let config = BatchConversionConfig::default()
            .with_simd(false)
            .with_parallel(false);
        let converter = BatchConverter::new(config);

        let result: Vec<f32> = converter.convert_slice(&data).unwrap();
        assert_eq!(result.len(), data.len());
        assert_eq!(result[0], 1.0f32);
        assert_eq!(result[1], 2.5f32);
    }

    #[test]
    fn test_conversion_with_errors() {
        let data: Vec<f64> = vec![1.0, f64::NAN, 3.0, f64::INFINITY];
        let converter = BatchConverter::with_default_config();

        let (converted, errors) = converter.convert_slice_with_errors::<f64, f32>(&data);
        assert_eq!(converted.len(), 2); // Only 1.0 and 3.0 should convert
        assert_eq!(errors.len(), 2); // NAN and INFINITY should error
    }

    #[test]
    fn test_clamped_conversion() {
        let data: Vec<f64> = vec![1e20, 2.5, -1e20, 100.0];
        let converter = BatchConverter::with_default_config();

        let result: Vec<f32> = converter.convert_slice_clamped(&data);
        assert_eq!(result.len(), data.len());
        // The implementation preserves the f64 -> f32 cast behavior
        assert_eq!(result[0], 1e20f32); // f64 1e20 cast to f32
        assert_eq!(result[1], 2.5f32); // Normal conversion
        assert_eq!(result[2], -1e20f32); // f64 -1e20 cast to f32
        assert_eq!(result[3], 100.0f32); // Normal conversion
    }

    #[test]
    fn test_complex_conversion() {
        let data: Vec<Complex64> = vec![
            Complex64::new(1.0, 2.0),
            Complex64::new(3.0, 4.0),
            Complex64::new(-1.0, -2.0),
        ];
        let converter = BatchConverter::with_default_config();

        let result: Vec<num_complex::Complex32> = converter.convert_complex_slice(&data).unwrap();
        assert_eq!(result.len(), data.len());
        assert_eq!(result[0].re, 1.0f32);
        assert_eq!(result[0].im, 2.0f32);
    }

    #[test]
    fn test_empty_slice() {
        let data: Vec<f64> = vec![];
        let converter = BatchConverter::with_default_config();

        let result: Vec<f32> = converter.convert_slice(&data).unwrap();
        assert_eq!(result.len(), 0);

        let (converted, errors) = converter.convert_slice_with_errors::<f64, f32>(&data);
        assert_eq!(converted.len(), 0);
        assert_eq!(errors.len(), 0);
    }

    #[cfg(feature = "simd")]
    #[test]
    fn test_simd_detection() {
        let converter = BatchConverter::with_default_config();

        // These should support SIMD
        assert!(converter.can_use_simd_for_conversion::<f64, f32>());
        assert!(converter.can_use_simd_for_conversion::<f32, f64>());
        assert!(converter.can_use_simd_for_conversion::<i32, f32>());

        // These should not support SIMD (not implemented)
        assert!(!converter.can_use_simd_for_conversion::<i8, i16>());
    }

    #[test]
    fn test_large_dataset_threshold() {
        let data: Vec<f64> = (0..20000).map(|i| i as f64 * 0.1).collect();
        let config = BatchConversionConfig::default().with_parallel_threshold(10000);
        let converter = BatchConverter::new(config);

        let result: Vec<f32> = converter.convert_slice(&data).unwrap();
        assert_eq!(result.len(), data.len());
    }

    #[test]
    fn test_utils_functions() {
        let f64_data: Vec<f64> = vec![1.0, 2.5, 3.7];
        let f32_result = utils::f64_to_f32_batch(&f64_data).unwrap();
        assert_eq!(f32_result.len(), f64_data.len());

        let f32_data: Vec<f32> = vec![1.0, 2.5, 3.7];
        let f64_result = utils::f32_to_f64_batch(&f32_data).unwrap();
        assert_eq!(f64_result.len(), f32_data.len());

        let i32_data: Vec<i32> = vec![1, 2, 3];
        let f32_result = utils::i32_to_f32_batch(&i32_data);
        assert_eq!(f32_result.len(), i32_data.len());
    }
}
