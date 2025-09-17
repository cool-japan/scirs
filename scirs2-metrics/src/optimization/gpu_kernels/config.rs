//! GPU kernel configuration and performance metrics
//!
//! This module provides configuration structures and performance tracking
//! for GPU kernel execution across different backends.

#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

use std::time::Duration;

/// GPU kernel configuration
#[derive(Debug, Clone)]
pub struct KernelConfig {
    /// Block size for CUDA / Work group size for OpenCL
    pub block_size: (u32, u32, u32),
    /// Grid size for CUDA / Global work size for OpenCL
    pub grid_size: (u32, u32, u32),
    /// Shared memory size
    pub shared_memory_size: u32,
    /// Use asynchronous execution
    pub async_execution: bool,
    /// Memory transfer optimization
    pub use_pinned_memory: bool,
    /// Kernel optimization level
    pub optimization_level: u8,
}

/// GPU compute configuration
#[derive(Debug, Clone)]
pub struct GpuComputeConfig {
    /// Preferred API (CUDA, OpenCL, Auto)
    pub preferred_api: GpuApi,
    /// Memory allocation strategy
    pub memory_strategy: MemoryStrategy,
    /// Kernel optimization settings
    pub kernel_optimization: KernelOptimization,
    /// Batch processing settings
    pub batch_settings: BatchSettings,
    /// Error handling strategy
    pub error_handling: ErrorHandling,
}

/// GPU API preference
#[derive(Debug, Clone, Copy)]
pub enum GpuApi {
    Auto,
    Cuda,
    OpenCl,
    Metal,  // For macOS support
    Vulkan, // For advanced compute
}

/// Memory allocation strategy
#[derive(Debug, Clone)]
pub enum MemoryStrategy {
    /// Pool pre-allocated blocks
    Pool {
        initial_size: usize,
        max_size: usize,
    },
    /// Allocate on demand
    OnDemand,
    /// Use unified memory (CUDA)
    Unified,
    /// Memory mapping
    Mapped,
}

/// Kernel optimization settings
#[derive(Debug, Clone)]
pub struct KernelOptimization {
    /// Use fast math operations
    pub fast_math: bool,
    /// Vectorization level
    pub vectorization: VectorizationLevel,
    /// Occupancy optimization
    pub optimize_occupancy: bool,
    /// Use shared memory optimizations
    pub use_shared_memory: bool,
    /// Memory coalescing optimization
    pub memory_coalescing: bool,
}

/// Vectorization level
#[derive(Debug, Clone, Copy)]
pub enum VectorizationLevel {
    None,
    Float2,
    Float4,
    Float8,
    Auto,
}

/// Batch processing settings
#[derive(Debug, Clone)]
pub struct BatchSettings {
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Minimum batch size for GPU usage
    pub min_batch_size: usize,
    /// Use multi-stream processing
    pub multi_stream: bool,
    /// Stream count
    pub stream_count: usize,
    /// Overlap computation and memory transfer
    pub overlap_computation: bool,
}

/// Error handling strategy
#[derive(Debug, Clone, Copy)]
pub enum ErrorHandling {
    /// Fail fast on any error
    FailFast,
    /// Retry with fallback
    RetryFallback,
    /// Graceful degradation
    GracefulFallback,
}

/// GPU performance statistics
#[derive(Debug, Default, Clone)]
pub struct GpuPerformanceStats {
    /// Total GPU operations performed
    pub total_operations: u64,
    /// Total GPU time
    pub total_gpu_time: Duration,
    /// Memory transfers performed
    pub memory_transfers: u64,
    /// Total memory transferred (bytes)
    pub total_memory_transferred: usize,
    /// Kernel launch count
    pub kernel_launches: u64,
    /// Average kernel execution time
    pub avg_kernel_time: Duration,
    /// Cache hit rate
    pub cache_hit_rate: f64,
    /// Memory bandwidth utilization
    pub memory_bandwidth_utilization: f64,
}

/// GPU computation results with detailed metrics
#[derive(Debug)]
pub struct GpuComputeResults<T> {
    /// Computation results
    pub results: T,
    /// Execution time
    pub execution_time: Duration,
    /// Memory usage
    pub memory_used: usize,
    /// Kernel performance metrics
    pub kernel_metrics: KernelMetrics,
    /// Transfer metrics
    pub transfer_metrics: TransferMetrics,
}

/// Kernel execution metrics
#[derive(Debug)]
pub struct KernelMetrics {
    /// Kernel launch time
    pub launch_time: Duration,
    /// Kernel execution time
    pub execution_time: Duration,
    /// Occupancy achieved
    pub occupancy: f32,
    /// Memory bandwidth achieved
    pub memory_bandwidth: f64,
    /// FLOPS achieved
    pub flops: f64,
}

/// Memory transfer metrics
#[derive(Debug)]
pub struct TransferMetrics {
    /// Host to device transfer time
    pub h2d_time: Duration,
    /// Device to host transfer time
    pub d2h_time: Duration,
    /// Bytes transferred H2D
    pub h2d_bytes: usize,
    /// Bytes transferred D2H
    pub d2h_bytes: usize,
    /// Transfer bandwidth achieved
    pub bandwidth: f64,
}

/// Compute strategy selection
#[derive(Debug, Clone, Copy)]
pub enum ComputeStrategy {
    Cuda,
    OpenCl,
    Fallback,
}

impl Default for GpuComputeConfig {
    fn default() -> Self {
        Self {
            preferred_api: GpuApi::Auto,
            memory_strategy: MemoryStrategy::Pool {
                initial_size: 256 * 1024 * 1024,  // 256MB
                max_size: 2 * 1024 * 1024 * 1024, // 2GB
            },
            kernel_optimization: KernelOptimization {
                fast_math: true,
                vectorization: VectorizationLevel::Auto,
                optimize_occupancy: true,
                use_shared_memory: true,
                memory_coalescing: true,
            },
            batch_settings: BatchSettings {
                max_batch_size: 1024 * 1024,
                min_batch_size: 1000,
                multi_stream: true,
                stream_count: 4,
                overlap_computation: true,
            },
            error_handling: ErrorHandling::RetryFallback,
        }
    }
}

impl Default for KernelConfig {
    fn default() -> Self {
        Self {
            block_size: (256, 1, 1),
            grid_size: (1, 1, 1),
            shared_memory_size: 0,
            async_execution: true,
            use_pinned_memory: true,
            optimization_level: 2,
        }
    }
}
