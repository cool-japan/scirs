//! GPU-accelerated operations modules
//!
//! This module has been refactored into focused sub-modules for better organization:
//! - `context`: GPU context management and backend selection
//! - `basic_operations`: Basic image processing operations (convolution, blur, gradients)
//! - `feature_detection`: Feature detection algorithms (Sobel, Harris corners)
//! - `batch_processing`: Batch operations, memory management, async processing, performance profiling
//! - `neural_operations`: Neural network operations, transformers, feature matching

pub mod context;
pub mod basic_operations;
pub mod feature_detection;
pub mod batch_processing;
pub mod neural_operations;

// Re-export main public API from context module
pub use context::{GpuMemoryStats, GpuVisionContext};

// Re-export basic operations
pub use basic_operations::{gpu_convolve_2d, gpu_gaussian_blur};

// Re-export feature detection functionality
pub use feature_detection::{gpu_element_wise_multiply, gpu_harris_corners, gpu_sobel_gradients};

// Re-export batch processing functionality
pub use batch_processing::{
    gpu_batch_convolve_2d, gpu_batch_process, AsyncGpuProcessor, GpuBenchmark, GpuMemoryPool,
    GpuOperation, GpuPerformanceProfiler,
};

// Re-export neural operations functionality
pub use neural_operations::{
    gpu_batch_matmul_transformer, gpu_feature_matching_advanced, gpu_multi_head_attention,
    gpu_neural_feature_extraction, LayerConfig, LayerType,
};