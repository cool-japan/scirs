//! GPU-accelerated operations for computer vision
//!
//! This module provides GPU-optimized implementations of vision operations
//! using the scirs2-core GPU abstraction layer.
//!
//! # Performance
//!
//! GPU operations can provide significant speedup for:
//! - Large-scale image processing
//! - Batch operations on multiple images
//! - Complex convolutions and filters
//! - Real-time video processing
//!
//! # Supported Backends
//!
//! - CUDA (NVIDIA GPUs)
//! - Metal (Apple Silicon and Intel Macs)
//! - OpenCL (Cross-platform)
//! - WebGPU (Future web deployment)
//! - CPU fallback for compatibility
//!
//! **Note**: This module has been refactored into smaller, focused sub-modules
//! for better maintainability. The original API is preserved for backward compatibility.

// Import the modular implementation
#[path = "gpu_modules/context.rs"]
pub mod context;

#[path = "gpu_modules/basic_operations.rs"]
pub mod basic_operations;

#[path = "gpu_modules/feature_detection.rs"]
pub mod feature_detection;

#[path = "gpu_modules/batch_processing.rs"]
pub mod batch_processing;

#[path = "gpu_modules/neural_operations.rs"]
pub mod neural_operations;

// Re-export types for backward compatibility
pub use context::{GpuMemoryStats, GpuVisionContext};

pub use basic_operations::{gpu_convolve_2d, gpu_gaussian_blur};

pub use feature_detection::{gpu_element_wise_multiply, gpu_harris_corners, gpu_sobel_gradients};

pub use batch_processing::{
    gpu_batch_convolve_2d, gpu_batch_process, AsyncGpuProcessor, GpuBenchmark, GpuMemoryPool,
    GpuOperation, GpuPerformanceProfiler,
};

pub use neural_operations::{
    gpu_batch_matmul_transformer, gpu_feature_matching_advanced, gpu_multi_head_attention,
    gpu_neural_feature_extraction, LayerConfig, LayerType,
};

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_gpu_context_creation() {
        let result = GpuVisionContext::new();
        // Should succeed with at least CPU backend
        assert!(result.is_ok());

        let ctx = result.unwrap();
        println!("GPU backend: {}", ctx.backend_name());
    }

    #[test]
    fn test_gpu_memory_info() {
        if let Ok(ctx) = GpuVisionContext::new() {
            if let Some(stats) = ctx.memory_stats() {
                println!("GPU Memory Stats:");
                println!("  Total: {} MB", stats.total_memory / (1024 * 1024));
                println!("  Available: {} MB", stats.available_memory / (1024 * 1024));
                println!("  Used: {} MB", stats.used_memory / (1024 * 1024));
                println!("  Utilization: {:.1}%", stats.utilization_percent);
            }
        }
    }

    #[test]
    #[ignore = "timeout"]
    fn test_gpu_convolution() {
        if let Ok(ctx) = GpuVisionContext::new() {
            let image = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);

            let kernel = arr2(&[[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]]);

            let result = gpu_convolve_2d(&ctx, &image.view(), &kernel.view());
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_backend_selection() {
        // Test CPU backend explicitly
        let cpu_ctx = GpuVisionContext::with_backend(scirs2_core::gpu::GpuBackend::Cpu);
        assert!(cpu_ctx.is_ok());

        let ctx = cpu_ctx.unwrap();
        assert_eq!(ctx.backend(), scirs2_core::gpu::GpuBackend::Cpu);
        assert!(!ctx.is_gpu_available());
    }
}
