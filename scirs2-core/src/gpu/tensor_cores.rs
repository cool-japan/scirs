//! Tensor core acceleration support for modern GPUs
//!
//! This module provides support for hardware-accelerated tensor operations using
//! specialized tensor processing units available on modern GPUs (NVIDIA Tensor Cores,
//! AMD Matrix Cores, etc.).

use crate::gpu::{GpuBackend, GpuBuffer, GpuError};
use std::fmt;
use thiserror::Error;

/// Supported tensor data types for hardware acceleration
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TensorDataType {
    /// 16-bit floating point (half precision)
    Float16,
    /// Brain floating point 16-bit
    BFloat16,
    /// 32-bit floating point (single precision)
    Float32,
    /// 64-bit floating point (double precision)
    Float64,
    /// 8-bit signed integer
    Int8,
    /// 4-bit integer (packed)
    Int4,
    /// 1-bit binary
    Binary,
    /// Mixed precision (accumulation in higher precision)
    Mixed(Box<TensorDataType>, Box<TensorDataType>), // (input, accumulator)
}

impl fmt::Display for TensorDataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TensorDataType::Float16 => write!(f, "f16"),
            TensorDataType::BFloat16 => write!(f, "bf16"),
            TensorDataType::Float32 => write!(f, "f32"),
            TensorDataType::Float64 => write!(f, "f64"),
            TensorDataType::Int8 => write!(f, "i8"),
            TensorDataType::Int4 => write!(f, "i4"),
            TensorDataType::Binary => write!(f, "binary"),
            TensorDataType::Mixed(input, accum) => write!(f, "mixed({}, {})", input, accum),
        }
    }
}

/// Tensor core operation types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorCoreOp {
    /// Matrix multiplication (GEMM)
    MatrixMultiply,
    /// Convolution operation
    Convolution,
    /// Attention mechanism (scaled dot-product attention)
    Attention,
    /// Sparse matrix operations
    SparseOps,
    /// Custom tensor operation
    Custom(&'static str),
}

/// Tensor core capabilities for different GPU architectures
#[derive(Debug, Clone, Default)]
pub struct TensorCoreCapabilities {
    /// Whether tensor cores are available
    pub available: bool,
    /// Supported data types
    pub supported_types: Vec<TensorDataType>,
    /// Supported operations
    pub supported_ops: Vec<TensorCoreOp>,
    /// Matrix dimensions supported (M, N, K)
    pub supported_dimensions: Vec<(usize, usize, usize)>,
    /// Peak throughput in TOPS (Tera-Operations Per Second)
    pub peak_tops: Option<f64>,
    /// Memory bandwidth in GB/s
    pub memory_bandwidth_gbps: Option<f64>,
    /// Architecture-specific features
    pub arch_features: Vec<String>,
}

/// Tensor core configuration for optimized operations
#[derive(Debug, Clone)]
pub struct TensorCoreConfig {
    /// Preferred data type for computations
    pub data_type: TensorDataType,
    /// Use mixed precision if available
    pub use_mixed_precision: bool,
    /// Enable automatic type conversion
    pub auto_convert: bool,
    /// Tile size for operations
    pub tile_size: (usize, usize),
    /// Use sparse operations if beneficial
    pub use_sparse: bool,
    /// Architecture-specific optimizations
    pub arch_optimizations: Vec<String>,
}

impl Default for TensorCoreConfig {
    fn default() -> Self {
        Self {
            data_type: TensorDataType::Float16,
            use_mixed_precision: true,
            auto_convert: true,
            tile_size: (16, 16),
            use_sparse: false,
            arch_optimizations: Vec::new(),
        }
    }
}

/// Error types for tensor core operations
#[derive(Error, Debug)]
pub enum TensorCoreError {
    /// Tensor cores not available on this device
    #[error("Tensor cores not available on this device")]
    NotAvailable,

    /// Unsupported data type for tensor core operations
    #[error("Unsupported data type: {0}")]
    UnsupportedDataType(TensorDataType),

    /// Unsupported operation
    #[error("Unsupported operation: {0:?}")]
    UnsupportedOperation(TensorCoreOp),

    /// Invalid matrix dimensions
    #[error("Invalid matrix dimensions: {m}x{n}x{k}")]
    InvalidDimensions { m: usize, n: usize, k: usize },

    /// Memory alignment error
    #[error("Memory alignment error: {0}")]
    MemoryAlignment(String),

    /// Performance hint
    #[error("Performance warning: {0}")]
    PerformanceWarning(String),

    /// Underlying GPU error
    #[error("GPU error: {0}")]
    GpuError(#[from] GpuError),
}

/// Tensor core manager for handling hardware acceleration
pub struct TensorCoreManager {
    backend: GpuBackend,
    capabilities: TensorCoreCapabilities,
    config: TensorCoreConfig,
}

impl TensorCoreManager {
    /// Create a new tensor core manager for the given backend
    pub fn new(backend: GpuBackend) -> Result<Self, TensorCoreError> {
        let capabilities = Self::detect_capabilities(backend)?;

        if !capabilities.available {
            return Err(TensorCoreError::NotAvailable);
        }

        let config = Self::optimal_config(&capabilities);

        Ok(Self {
            backend,
            capabilities,
            config,
        })
    }

    /// Get tensor core capabilities for the current device
    pub const fn capabilities(&self) -> &TensorCoreCapabilities {
        &self.capabilities
    }

    /// Get current configuration
    pub const fn config(&self) -> &TensorCoreConfig {
        &self.config
    }

    /// Update tensor core configuration
    pub fn set_config(&mut self, config: TensorCoreConfig) -> Result<(), TensorCoreError> {
        // Validate configuration against capabilities
        if !self
            .capabilities
            .supported_types
            .contains(&config.data_type)
        {
            return Err(TensorCoreError::UnsupportedDataType(config.data_type));
        }

        self.config = config;
        Ok(())
    }

    /// Check if an operation is supported with current configuration
    pub fn is_operation_supported(&self, op: TensorCoreOp) -> bool {
        self.capabilities.supported_ops.contains(&op)
    }

    /// Check if dimensions are optimal for tensor core operations
    pub fn are_dimensions_optimal(&self, m: usize, n: usize, k: usize) -> bool {
        // Check if dimensions are multiples of tensor core tile sizes
        match self.backend {
            GpuBackend::Cuda => {
                // NVIDIA tensor cores typically work best with multiples of 16
                m % 16 == 0 && n % 16 == 0 && k % 16 == 0
            }
            GpuBackend::Rocm => {
                // AMD matrix cores typically work best with multiples of 32
                m % 32 == 0 && n % 32 == 0 && k % 32 == 0
            }
            _ => false,
        }
    }

    /// Get performance hints for given dimensions
    pub fn get_performance_hints(&self, m: usize, n: usize, k: usize) -> Vec<String> {
        let mut hints = Vec::new();

        if !self.are_dimensions_optimal(m, n, k) {
            hints.push(format!(
                "Consider padding dimensions to multiples of {} for optimal performance",
                match self.backend {
                    GpuBackend::Cuda => 16,
                    GpuBackend::Rocm => 32,
                    _ => 16,
                }
            ));
        }

        if self.config.use_mixed_precision && self.config.data_type == TensorDataType::Float32 {
            hints.push(
                "Consider using Float16 or BFloat16 for better tensor core utilization".to_string(),
            );
        }

        if m * n * k < 1024 * 1024 {
            hints.push("Small matrices may not fully utilize tensor cores".to_string());
        }

        hints
    }

    /// Suggest optimal data type for given operation
    pub fn suggest_optimal_type(&self, op: TensorCoreOp) -> Option<TensorDataType> {
        match op {
            TensorCoreOp::MatrixMultiply => {
                if self
                    .capabilities
                    .supported_types
                    .contains(&TensorDataType::BFloat16)
                {
                    Some(TensorDataType::BFloat16)
                } else if self
                    .capabilities
                    .supported_types
                    .contains(&TensorDataType::Float16)
                {
                    Some(TensorDataType::Float16)
                } else {
                    Some(TensorDataType::Float32)
                }
            }
            TensorCoreOp::Convolution => {
                if self
                    .capabilities
                    .supported_types
                    .contains(&TensorDataType::Int8)
                {
                    Some(TensorDataType::Int8)
                } else if self
                    .capabilities
                    .supported_types
                    .contains(&TensorDataType::Float16)
                {
                    Some(TensorDataType::Float16)
                } else {
                    Some(TensorDataType::Float32)
                }
            }
            TensorCoreOp::Attention => {
                // Attention typically benefits from higher precision
                if self
                    .capabilities
                    .supported_types
                    .contains(&TensorDataType::BFloat16)
                {
                    Some(TensorDataType::BFloat16)
                } else {
                    Some(TensorDataType::Float32)
                }
            }
            _ => self.capabilities.supported_types.first().cloned(),
        }
    }

    /// Detect tensor core capabilities for the given backend
    fn detect_capabilities(backend: GpuBackend) -> Result<TensorCoreCapabilities, TensorCoreError> {
        match backend {
            GpuBackend::Cuda => Ok(Self::nvidia_tensor_capabilities()),
            GpuBackend::Rocm => Ok(Self::amd_matrix_capabilities()),
            GpuBackend::Metal => Ok(Self::apple_neural_capabilities()),
            _ => Ok(TensorCoreCapabilities::default()),
        }
    }

    /// NVIDIA Tensor Core capabilities (Volta, Turing, Ampere, Hopper)
    fn nvidia_tensor_capabilities() -> TensorCoreCapabilities {
        TensorCoreCapabilities {
            available: true,
            supported_types: vec![
                TensorDataType::Float16,
                TensorDataType::BFloat16,
                TensorDataType::Float32,
                TensorDataType::Int8,
                TensorDataType::Int4,
                TensorDataType::Binary,
                TensorDataType::Mixed(
                    Box::new(TensorDataType::Float16),
                    Box::new(TensorDataType::Float32),
                ),
            ],
            supported_ops: vec![
                TensorCoreOp::MatrixMultiply,
                TensorCoreOp::Convolution,
                TensorCoreOp::Attention,
                TensorCoreOp::SparseOps,
            ],
            supported_dimensions: vec![
                (16, 16, 16), // Basic tensor core size
                (32, 8, 16),  // Alternative configurations
                (8, 32, 16),
            ],
            peak_tops: Some(312.0),              // Example for A100
            memory_bandwidth_gbps: Some(2039.0), // Example for A100 HBM2e
            arch_features: vec![
                "Sparsity 2:4".to_string(),
                "Multi-precision".to_string(),
                "Transformer Engine".to_string(),
            ],
        }
    }

    /// AMD Matrix Core capabilities (CDNA, RDNA)
    fn amd_matrix_capabilities() -> TensorCoreCapabilities {
        TensorCoreCapabilities {
            available: true,
            supported_types: vec![
                TensorDataType::Float16,
                TensorDataType::BFloat16,
                TensorDataType::Float32,
                TensorDataType::Int8,
                TensorDataType::Mixed(
                    Box::new(TensorDataType::Float16),
                    Box::new(TensorDataType::Float32),
                ),
            ],
            supported_ops: vec![TensorCoreOp::MatrixMultiply, TensorCoreOp::Convolution],
            supported_dimensions: vec![
                (32, 32, 8), // MFMA instruction size
                (16, 16, 16),
            ],
            peak_tops: Some(383.0),              // Example for MI250X
            memory_bandwidth_gbps: Some(3276.0), // Example for MI250X HBM2e
            arch_features: vec!["MFMA instructions".to_string(), "Matrix cores".to_string()],
        }
    }

    /// Apple Neural Engine capabilities
    fn apple_neural_capabilities() -> TensorCoreCapabilities {
        TensorCoreCapabilities {
            available: true,
            supported_types: vec![
                TensorDataType::Float16,
                TensorDataType::Float32,
                TensorDataType::Int8,
            ],
            supported_ops: vec![
                TensorCoreOp::MatrixMultiply,
                TensorCoreOp::Convolution,
                TensorCoreOp::Attention,
            ],
            supported_dimensions: vec![(16, 16, 16)],
            peak_tops: Some(15.8),              // Example for M1 Neural Engine
            memory_bandwidth_gbps: Some(68.25), // Example for M1 unified memory
            arch_features: vec!["Neural Engine".to_string(), "Unified memory".to_string()],
        }
    }

    /// Determine optimal configuration based on capabilities
    fn optimal_config(capabilities: &TensorCoreCapabilities) -> TensorCoreConfig {
        let data_type = if capabilities
            .supported_types
            .contains(&TensorDataType::BFloat16)
        {
            TensorDataType::BFloat16
        } else if capabilities
            .supported_types
            .contains(&TensorDataType::Float16)
        {
            TensorDataType::Float16
        } else {
            TensorDataType::Float32
        };

        let tile_size = capabilities
            .supported_dimensions
            .first()
            .map(|(m, n, _)| (*m, *n))
            .unwrap_or((16, 16));

        TensorCoreConfig {
            data_type,
            use_mixed_precision: capabilities
                .supported_types
                .iter()
                .any(|t| matches!(t, TensorDataType::Mixed(_, _))),
            auto_convert: true,
            tile_size,
            use_sparse: capabilities
                .arch_features
                .iter()
                .any(|f| f.contains("Sparsity")),
            arch_optimizations: capabilities.arch_features.clone(),
        }
    }
}

/// Tensor core operation descriptor
#[derive(Debug, Clone)]
pub struct TensorOperation {
    /// Type of operation
    pub op_type: TensorCoreOp,
    /// Input data type
    pub input_type: TensorDataType,
    /// Output data type
    pub output_type: TensorDataType,
    /// Matrix dimensions (M, N, K)
    pub dimensions: (usize, usize, usize),
    /// Whether to use mixed precision
    pub mixed_precision: bool,
    /// Sparsity pattern if applicable
    pub sparsity: Option<SparsePattern>,
}

/// Sparsity patterns for sparse tensor operations
#[derive(Debug, Clone)]
pub enum SparsePattern {
    /// 2:4 structured sparsity (2 out of every 4 elements are zero)
    Structured2_4,
    /// Random sparsity with given ratio
    Random(f32),
    /// Block sparsity
    Block {
        block_size: (usize, usize),
        sparsity: f32,
    },
    /// Custom sparsity pattern
    Custom(String),
}

/// Tensor core optimized matrix multiplication
pub fn tensor_core_gemm<T>(
    manager: &TensorCoreManager,
    _a: &GpuBuffer<T>,
    _b: &GpuBuffer<T>,
    _c: &mut GpuBuffer<T>,
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), TensorCoreError>
where
    T: crate::gpu::GpuDataType,
{
    // Validate dimensions
    if !manager.are_dimensions_optimal(m, n, k) {
        let hints = manager.get_performance_hints(m, n, k);
        for hint in hints {
            eprintln!("Performance hint: {}", hint);
        }
    }

    // Check if operation is supported
    if !manager.is_operation_supported(TensorCoreOp::MatrixMultiply) {
        return Err(TensorCoreError::UnsupportedOperation(
            TensorCoreOp::MatrixMultiply,
        ));
    }

    // In a real implementation, we would:
    // 1. Convert data types if needed
    // 2. Set up tensor core kernels
    // 3. Launch optimized GEMM operation
    // 4. Handle mixed precision accumulation

    // For now, this is a placeholder
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_data_type_display() {
        assert_eq!(TensorDataType::Float16.to_string(), "f16");
        assert_eq!(TensorDataType::BFloat16.to_string(), "bf16");
        assert_eq!(TensorDataType::Int8.to_string(), "i8");
    }

    #[test]
    fn test_nvidia_capabilities() {
        let caps = TensorCoreManager::nvidia_tensor_capabilities();
        assert!(caps.available);
        assert!(caps.supported_types.contains(&TensorDataType::Float16));
        assert!(caps.supported_ops.contains(&TensorCoreOp::MatrixMultiply));
    }

    #[test]
    fn test_amd_capabilities() {
        let caps = TensorCoreManager::amd_matrix_capabilities();
        assert!(caps.available);
        assert!(caps.supported_types.contains(&TensorDataType::BFloat16));
        assert!(caps.supported_ops.contains(&TensorCoreOp::MatrixMultiply));
    }

    #[test]
    fn test_optimal_config() {
        let caps = TensorCoreManager::nvidia_tensor_capabilities();
        let config = TensorCoreManager::optimal_config(&caps);
        assert_eq!(config.data_type, TensorDataType::BFloat16);
        assert!(config.use_mixed_precision);
    }

    #[test]
    fn test_dimension_optimization() {
        // This would require a real GPU context, so we'll test the logic only
        let caps = TensorCoreManager::nvidia_tensor_capabilities();
        let config = TensorCoreManager::optimal_config(&caps);

        // Test that we can create a config
        assert!(config.auto_convert);
        assert_eq!(config.tile_size, (16, 16));
    }
}
