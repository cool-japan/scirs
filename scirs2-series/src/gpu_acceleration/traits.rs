//! GPU acceleration traits for time series operations
//!
//! This module defines the core traits for GPU-accelerated time series processing,
//! including data transfer, forecasting, decomposition, and feature extraction.

use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::Float;
use std::fmt::Debug;

use super::config::GpuConfig;
use crate::error::Result;

/// Trait for GPU-accelerated time series operations
pub trait GpuAccelerated<F: Float + Debug> {
    /// Transfer data to GPU
    fn to_gpu(&self, config: &GpuConfig) -> Result<Self>
    where
        Self: Sized;

    /// Transfer data from GPU to CPU
    fn to_cpu(&self) -> Result<Self>
    where
        Self: Sized;

    /// Check if data is on GPU
    fn is_on_gpu(&self) -> bool;

    /// Get GPU memory usage in bytes
    fn gpu_memory_usage(&self) -> usize;
}

/// Type alias for decomposition result (trend, seasonal, residual)
pub type DecompositionResult<F> = (Array1<F>, Array1<F>, Array1<F>);

/// GPU-accelerated forecasting operations
pub trait GpuForecasting<F: Float + Debug> {
    /// Perform forecasting on GPU
    fn forecast_gpu(&self, steps: usize, config: &GpuConfig) -> Result<Array1<F>>;

    /// Batch forecasting for multiple series
    fn batch_forecast_gpu(
        &self,
        data: &[Array1<F>],
        steps: usize,
        config: &GpuConfig,
    ) -> Result<Vec<Array1<F>>>;
}

/// GPU-accelerated decomposition operations
pub trait GpuDecomposition<F: Float + Debug> {
    /// Perform decomposition on GPU
    fn decompose_gpu(&self, config: &GpuConfig) -> Result<DecompositionResult<F>>;

    /// Batch decomposition for multiple series
    fn batch_decompose_gpu(
        &self,
        data: &[Array1<F>],
        config: &GpuConfig,
    ) -> Result<Vec<DecompositionResult<F>>>;
}

/// GPU-accelerated feature extraction
pub trait GpuFeatureExtraction<F: Float + Debug> {
    /// Extract features on GPU
    fn extract_features_gpu(&self, config: &GpuConfig) -> Result<Array1<F>>;

    /// Batch feature extraction for multiple series
    fn batch_extract_features_gpu(
        &self,
        data: &[Array1<F>],
        config: &GpuConfig,
    ) -> Result<Vec<Array1<F>>>;
}
