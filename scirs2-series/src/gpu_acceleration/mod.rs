//! GPU acceleration infrastructure for time series operations
//!
//! This module provides the foundation for GPU-accelerated time series processing,
//! including forecasting, decomposition, and feature extraction.

// Module declarations
pub mod algorithms;
pub mod array;
pub mod blas;
pub mod config;
pub mod convolution;
pub mod device_manager;
pub mod fft;
pub mod traits;
pub mod utils;

// Re-export all public items for backward compatibility
pub use config::{
    GpuBackend, GpuCapabilities, GpuConfig, GraphOptimizationLevel, MemoryStrategy,
    TensorCoresConfig, TensorCoresGeneration, TensorDataType,
};

pub use traits::{
    DecompositionResult, GpuAccelerated, GpuDecomposition, GpuFeatureExtraction, GpuForecasting,
};

pub use array::GpuArray;

pub use device_manager::GpuDeviceManager;

// Re-export utility functions
pub use utils::{
    estimate_memory_usage, get_recommended_batch_size, is_gpu_supported, optimize_gpu_config,
};

// Re-export FFT functionality
pub use fft::GpuFFT;

// Re-export convolution functionality
pub use convolution::GpuConvolution;

// Re-export BLAS functionality
pub use blas::{GpuBLAS, TensorCoresBLAS};

// Re-export algorithms functionality
pub use algorithms::{
    FeatureConfig, ForecastMethod, GpuFeatureExtractor, GpuTimeSeriesProcessor, WindowStatistic,
};

// For backward compatibility, also re-export everything at the module level
use scirs2_core::ndarray::{s, Array1};
use scirs2_core::numeric::Float;
use std::fmt::Debug;

use crate::error::{Result, TimeSeriesError};

// Re-export commonly used imports for convenience
pub use scirs2_core::ndarray;
pub use scirs2_core::numeric;

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    #[test]
    fn test_gpu_config_default() {
        let config = GpuConfig::default();
        assert_eq!(config.device_id, 0);
        assert_eq!(config.batch_size, 1024);
        assert!(config.enable_memory_optimization);
    }

    #[test]
    fn test_gpu_array_creation() {
        let config = GpuConfig::default();
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let gpu_array = GpuArray::from_cpu(data, config);
        assert_eq!(gpu_array.len(), 5);
        assert!(!gpu_array.is_on_gpu());
    }

    #[test]
    fn test_device_manager_creation() {
        let device_manager = GpuDeviceManager::new();
        assert!(device_manager.is_ok());
    }

    #[test]
    fn test_tensor_cores_config_default() {
        let config = TensorCoresConfig::default();
        assert!(config.enabled);
        assert_eq!(config.data_type, TensorDataType::FP16);
        assert_eq!(config.tile_size, (16, 16, 16));
    }

    #[test]
    fn test_tensor_cores_generation_capabilities() {
        let gen_v1 = TensorCoresGeneration::V1;
        let supported_types = gen_v1.supported_data_types();
        assert!(supported_types.contains(&TensorDataType::FP16));

        let dimensions = gen_v1.supported_matrix_dimensions();
        assert!(dimensions.contains(&(16, 16, 16)));
    }

    #[test]
    fn test_utils_functions() {
        assert!(utils::is_gpu_supported() || !utils::is_gpu_supported()); // Should not panic

        let batch_size = utils::get_recommended_batch_size(1000, 1024 * 1024);
        assert!(batch_size > 0);

        let memory_usage = utils::estimate_memory_usage(1000, 0.5);
        assert!(memory_usage > 0);

        let config = utils::optimize_gpu_config(10000, 256 * 1024 * 1024);
        assert!(config.batch_size > 0);
    }
}
