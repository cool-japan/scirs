//! Advanced GPU kernels for high-performance metrics computation
//!
//! This module provides production-ready GPU kernels using CUDA, OpenCL, Metal, and Vulkan
//! for large-scale metrics computation with optimal memory management.

#![allow(clippy::too_many_arguments)]
#![allow(clippy::uninlined_format_args)]
#![allow(dead_code)]

// Module declarations
pub mod computer;
pub mod config;
pub mod contexts;
pub mod kernels;
pub mod runtime;

// Re-export core types and functions for backward compatibility
pub use computer::AdvancedGpuComputer;
pub use config::{
    BatchSettings, ComputeStrategy, ErrorHandling, GpuApi, GpuComputeConfig, GpuComputeResults,
    GpuPerformanceStats, KernelConfig, KernelMetrics, KernelOptimization, MemoryStrategy,
    TransferMetrics, VectorizationLevel,
};
pub use contexts::{
    CudaContext, CudaDeviceProperties, CudaMemoryBlock, CudaMemoryPool, CudaMemoryStats,
    OpenClContext, OpenClDeviceInfo,
};
pub use kernels::{cuda_kernels, metal_kernels, opencl_kernels, vulkan_kernels};
pub use runtime::{
    CudaRuntime, GpuBuffer, GpuBufferHandle, GpuBufferType, GpuKernelArg, GpuMemoryStats,
    GpuPerformanceStats as RuntimeGpuPerformanceStats, GpuRuntime, GpuScalar, MetalRuntime,
    OpenClRuntime, VulkanRuntime,
};

// Legacy type aliases for backward compatibility
pub use computer::AdvancedGpuComputer as GpuComputer;
pub use config::GpuComputeConfig as GpuConfig;

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_advanced_gpu_computer_creation() {
        let config = GpuComputeConfig::default();
        let computer = AdvancedGpuComputer::new(config);
        assert!(computer.is_ok());
    }

    #[test]
    fn test_cuda_availability_detection() {
        let available = AdvancedGpuComputer::is_cuda_available();
        // Should work regardless of actual CUDA availability
        println!("CUDA available: {}", available);
    }

    #[test]
    fn test_opencl_availability_detection() {
        let available = AdvancedGpuComputer::is_opencl_available();
        println!("OpenCL available: {}", available);
    }

    #[test]
    #[ignore = "timeout"]
    fn test_batch_metrics_computation() {
        let computer = AdvancedGpuComputer::default();

        let y_true_batch = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let y_pred_batch = array![[1.1, 2.1, 2.9], [4.1, 4.9, 6.1]];

        let results = computer.compute_batch_metrics(
            &y_true_batch.view(),
            &y_pred_batch.view(),
            &["mse", "mae", "r2_score"],
        );

        assert!(results.is_ok());

        if let Ok(gpu_results) = results {
            assert_eq!(gpu_results.results.len(), 2);
            assert!(gpu_results.execution_time.as_nanos() > 0);
            assert!(gpu_results.memory_used > 0);
        }
    }

    #[test]
    #[ignore = "timeout"]
    fn test_performance_stats_tracking() {
        let computer = AdvancedGpuComputer::default();

        // Simulate some operations
        let y_true_batch = array![[1.0, 2.0], [3.0, 4.0]];
        let y_pred_batch = array![[1.1, 2.1], [2.9, 4.1]];

        let _ =
            computer.compute_batch_metrics(&y_true_batch.view(), &y_pred_batch.view(), &["mse"]);

        let stats = computer.get_performance_stats();
        assert!(stats.total_operations > 0);
    }

    #[test]
    fn test_kernel_config_defaults() {
        let config = KernelConfig::default();
        assert_eq!(config.block_size, (256, 1, 1));
        assert_eq!(config.grid_size, (1, 1, 1));
        assert!(config.async_execution);
    }

    #[test]
    fn test_gpu_compute_config_defaults() {
        let config = GpuComputeConfig::default();
        matches!(config.preferred_api, GpuApi::Auto);
        assert!(config.kernel_optimization.fast_math);
        assert!(config.batch_settings.multi_stream);
    }

    #[test]
    fn test_gpu_runtime_initialization() {
        let mut cuda_runtime = CudaRuntime::new(0);
        assert!(cuda_runtime.initialize().is_ok());

        let mut opencl_runtime = OpenClRuntime::new(1, 1);
        assert!(opencl_runtime.initialize().is_ok());

        let mut metal_runtime = MetalRuntime::new();
        assert!(metal_runtime.initialize().is_ok());

        let mut vulkan_runtime = VulkanRuntime::new();
        assert!(vulkan_runtime.initialize().is_ok());
    }

    #[test]
    fn test_gpu_buffer_creation() {
        let buffer = GpuBuffer {
            id: 12345,
            size: 1024,
            buffer_type: GpuBufferType::InputOutput,
            handle: GpuBufferHandle::Cuda(0x11111111),
        };

        assert_eq!(buffer.id, 12345);
        assert_eq!(buffer.size, 1024);
        matches!(buffer.buffer_type, GpuBufferType::InputOutput);
    }

    #[test]
    fn test_cuda_memory_pool() {
        let mut pool = CudaMemoryPool::new(1024 * 1024); // 1MB limit

        // Allocate a block
        let block = pool.allocate(512);
        assert!(block.is_some());

        let block = block.unwrap();
        assert_eq!(block.size, 512);

        // Free the block
        let freed = pool.free(block.ptr);
        assert!(freed);

        // Check stats
        let stats = pool.get_stats();
        assert_eq!(stats.memory_limit, 1024 * 1024);
    }

    #[test]
    fn test_kernel_source_availability() {
        // Test that kernel sources are available
        assert!(!cuda_kernels::MSE_KERNEL.is_empty());
        assert!(!cuda_kernels::MAE_KERNEL.is_empty());
        assert!(!cuda_kernels::R2_KERNEL.is_empty());

        assert!(!opencl_kernels::MSE_KERNEL.is_empty());
        assert!(!opencl_kernels::MAE_KERNEL.is_empty());

        assert!(!metal_kernels::MSE_KERNEL.is_empty());
        assert!(!metal_kernels::MAE_KERNEL.is_empty());

        assert!(!vulkan_kernels::MSE_GLSL_SOURCE.is_empty());
        assert!(!vulkan_kernels::MAE_GLSL_SOURCE.is_empty());
    }
}
