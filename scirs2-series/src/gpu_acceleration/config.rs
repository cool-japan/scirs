//! GPU acceleration configuration and capabilities
//!
//! This module defines the configuration structures and enums for GPU acceleration,
//! including device capabilities, tensor cores configuration, and optimization settings.

use std::fmt::Debug;

/// GPU device configuration
#[derive(Debug, Clone)]
pub struct GpuConfig {
    /// Device ID to use
    pub device_id: usize,
    /// Memory pool size in bytes
    pub memory_pool_size: Option<usize>,
    /// Enable memory optimization
    pub enable_memory_optimization: bool,
    /// Batch size for GPU operations
    pub batch_size: usize,
    /// Use half precision (FP16) for faster computation
    pub use_half_precision: bool,
    /// Enable asynchronous execution
    pub enable_async: bool,
    /// Tensor cores configuration
    pub tensor_cores: TensorCoresConfig,
    /// Memory allocation strategy
    pub memory_strategy: MemoryStrategy,
    /// Enable dynamic batch sizing
    pub dynamic_batching: bool,
    /// Graph optimization level
    pub graph_optimization: GraphOptimizationLevel,
}

/// Graph optimization levels for GPU computation
#[derive(Debug, Clone, Copy)]
pub enum GraphOptimizationLevel {
    /// No optimization
    None,
    /// Basic optimization
    Basic,
    /// Extended optimization
    Extended,
    /// Maximum optimization (may increase compile time)
    Maximum,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            device_id: 0,
            memory_pool_size: None,
            enable_memory_optimization: true,
            batch_size: 1024,
            use_half_precision: false,
            enable_async: true,
            tensor_cores: TensorCoresConfig::default(),
            memory_strategy: MemoryStrategy::OnDemand,
            dynamic_batching: true,
            graph_optimization: GraphOptimizationLevel::Extended,
        }
    }
}

/// GPU memory management strategy
#[derive(Debug, Clone)]
pub enum MemoryStrategy {
    /// Allocate memory on-demand
    OnDemand,
    /// Pre-allocate memory pool
    PreAllocated {
        /// Size of the memory pool in bytes
        pool_size: usize,
    },
    /// Use unified memory (if available)
    Unified,
    /// Use pinned host memory for transfers
    Pinned,
}

/// GPU computation backend
#[derive(Debug, Clone, PartialEq)]
pub enum GpuBackend {
    /// CUDA backend for NVIDIA GPUs
    Cuda,
    /// ROCm backend for AMD GPUs
    Rocm,
    /// OpenCL backend for cross-platform support
    OpenCL,
    /// Metal backend for Apple Silicon
    Metal,
    /// CPU fallback (no GPU acceleration)
    CpuFallback,
}

/// GPU acceleration capabilities
#[derive(Debug, Clone)]
pub struct GpuCapabilities {
    /// Available backend
    pub backend: GpuBackend,
    /// Compute capability (for CUDA)
    pub compute_capability: Option<(u32, u32)>,
    /// Available memory in bytes
    pub memory: usize,
    /// Number of multiprocessors
    pub multiprocessors: usize,
    /// Supports half precision
    pub supports_fp16: bool,
    /// Supports tensor cores
    pub supports_tensor_cores: bool,
    /// Maximum threads per block
    pub max_threads_per_block: usize,
    /// Tensor cores generation
    pub tensor_cores_generation: Option<TensorCoresGeneration>,
    /// Memory bandwidth (GB/s)
    pub memory_bandwidth: f64,
    /// Peak tensor performance (TOPS)
    pub tensor_performance: Option<f64>,
}

/// Tensor cores generation and capabilities
#[derive(Debug, Clone, Copy)]
pub enum TensorCoresGeneration {
    /// First generation (V100)
    V1,
    /// Second generation (T4, RTX 20xx)
    V2,
    /// Third generation (A100, RTX 30xx)
    V3,
    /// Fourth generation (H100, RTX 40xx)
    V4,
}

impl TensorCoresGeneration {
    /// Get supported data types for this generation
    pub fn supported_data_types(&self) -> Vec<TensorDataType> {
        match self {
            TensorCoresGeneration::V1 => vec![TensorDataType::FP16],
            TensorCoresGeneration::V2 => vec![TensorDataType::FP16, TensorDataType::INT8],
            TensorCoresGeneration::V3 => vec![
                TensorDataType::FP16,
                TensorDataType::BF16,
                TensorDataType::INT8,
                TensorDataType::INT4,
                TensorDataType::FP64,
            ],
            TensorCoresGeneration::V4 => vec![
                TensorDataType::FP16,
                TensorDataType::BF16,
                TensorDataType::INT8,
                TensorDataType::INT4,
                TensorDataType::FP8,
                TensorDataType::FP64,
            ],
        }
    }

    /// Get matrix dimensions supported by tensor cores
    pub fn supported_matrix_dimensions(&self) -> Vec<(usize, usize, usize)> {
        match self {
            TensorCoresGeneration::V1 => vec![(16, 16, 16)],
            TensorCoresGeneration::V2 => vec![(16, 16, 16), (8, 32, 16), (32, 8, 16)],
            TensorCoresGeneration::V3 | TensorCoresGeneration::V4 => vec![
                (16, 16, 16),
                (8, 32, 16),
                (32, 8, 16),
                (16, 8, 8),
                (8, 8, 4),
            ],
        }
    }
}

/// Tensor data types supported by tensor cores
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TensorDataType {
    /// 16-bit floating point
    FP16,
    /// 16-bit brain floating point
    BF16,
    /// 8-bit floating point (FP8)
    FP8,
    /// 64-bit floating point
    FP64,
    /// 8-bit integer
    INT8,
    /// 4-bit integer
    INT4,
}

/// Tensor cores optimization configuration
#[derive(Debug, Clone)]
pub struct TensorCoresConfig {
    /// Enable tensor cores acceleration
    pub enabled: bool,
    /// Preferred data type for computation
    pub data_type: TensorDataType,
    /// Matrix dimensions to use for tiling
    pub tile_size: (usize, usize, usize),
    /// Enable mixed precision training
    pub mixed_precision: bool,
    /// Loss scaling for mixed precision
    pub loss_scale: f32,
    /// Enable automatic mixed precision
    pub auto_mixed_precision: bool,
    /// Minimum matrix size to use tensor cores
    pub min_matrix_size: usize,
}

impl Default for TensorCoresConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            data_type: TensorDataType::FP16,
            tile_size: (16, 16, 16),
            mixed_precision: true,
            loss_scale: 65536.0,
            auto_mixed_precision: true,
            min_matrix_size: 512,
        }
    }
}
