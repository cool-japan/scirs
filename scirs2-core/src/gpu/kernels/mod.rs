//! GPU kernel library for common scientific computing operations
//!
//! This module provides optimized GPU kernels for various operations used in
//! scientific computing, with support for multiple GPU backends.

use std::collections::HashMap;
use std::fmt;

pub mod blas;
pub mod complex;
pub mod ml;
pub mod reduction;
pub mod transform;

use crate::gpu::{GpuBackend, GpuError};

/// Supported data types for GPU kernels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DataType {
    /// 32-bit floating point (f32)
    Float32,
    /// 64-bit floating point (f64)
    Float64,
    /// 32-bit signed integer (i32)
    Int32,
    /// 32-bit unsigned integer (u32)
    UInt32,
    /// 16-bit floating point (f16)
    Float16,
    /// Brain floating point (bfloat16)
    BFloat16,
}

impl fmt::Display for DataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DataType::Float32 => write!(f, "f32"),
            DataType::Float64 => write!(f, "f64"),
            DataType::Int32 => write!(f, "i32"),
            DataType::UInt32 => write!(f, "u32"),
            DataType::Float16 => write!(f, "f16"),
            DataType::BFloat16 => write!(f, "bf16"),
        }
    }
}

/// The type of operation performed by the kernel
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperationType {
    /// Primarily compute-intensive operations
    ComputeIntensive,
    /// Primarily memory-intensive operations
    MemoryIntensive,
    /// Balanced between compute and memory
    Balanced,
}

/// Metadata for kernel execution
#[derive(Debug, Clone)]
pub struct KernelMetadata {
    /// Recommended workgroup size
    pub workgroup_size: [u32; 3],
    /// Local memory usage in bytes
    pub local_memory_usage: usize,
    /// Whether the kernel supports tensor cores (NVIDIA) or similar
    pub supports_tensor_cores: bool,
    /// Operation type (compute intensive, memory intensive, balanced)
    pub operation_type: OperationType,
    /// Additional backend-specific metadata
    pub backend_metadata: HashMap<String, String>,
}

impl Default for KernelMetadata {
    fn default() -> Self {
        Self {
            workgroup_size: [16, 16, 1],
            local_memory_usage: 0,
            supports_tensor_cores: false,
            operation_type: OperationType::Balanced,
            backend_metadata: HashMap::new(),
        }
    }
}

/// Parameters for kernel specialization
#[derive(Debug, Clone)]
pub struct KernelParams {
    /// Numeric type (f32, f64, etc.)
    pub data_type: DataType,
    /// Input dimensions
    pub input_dims: Vec<usize>,
    /// Output dimensions
    pub output_dims: Vec<usize>,
    /// Additional numeric parameters
    pub numeric_params: HashMap<String, f64>,
    /// Additional string parameters
    pub string_params: HashMap<String, String>,
}

impl KernelParams {
    /// Create new kernel parameters
    pub fn new(data_type: DataType) -> Self {
        Self {
            data_type,
            input_dims: Vec::new(),
            output_dims: Vec::new(),
            numeric_params: HashMap::new(),
            string_params: HashMap::new(),
        }
    }

    /// Set input dimensions
    pub fn with_input_dims(mut self, dims: Vec<usize>) -> Self {
        self.input_dims = dims;
        self
    }

    /// Set output dimensions
    pub fn with_output_dims(mut self, dims: Vec<usize>) -> Self {
        self.output_dims = dims;
        self
    }

    /// Add a numeric parameter
    pub fn with_numeric_param(mut self, name: &str, value: f64) -> Self {
        self.numeric_params.insert(name.to_string(), value);
        self
    }

    /// Add a string parameter
    pub fn with_string_param(mut self, name: &str, value: &str) -> Self {
        self.string_params
            .insert(name.to_string(), value.to_string());
        self
    }
}

/// GPU Kernel interface
pub trait GpuKernel: Send + Sync {
    /// The name of the kernel
    fn name(&self) -> &str;

    /// Get kernel source for the specified backend
    fn source_for_backend(&self, backend: GpuBackend) -> Result<String, GpuError>;

    /// Get kernel metadata (workgroup size, memory requirements, etc.)
    fn metadata(&self) -> KernelMetadata;

    /// Can this kernel be specialized for the given parameters?
    fn can_specialize(&self, params: &KernelParams) -> bool;

    /// Create a specialized version of this kernel for the given parameters
    fn specialize(&self, params: &KernelParams) -> Result<Box<dyn GpuKernel>, GpuError>;
}

/// Base kernel implementation that can be used by specialized kernels
pub struct BaseKernel {
    name: String,
    cuda_source: String,
    rocm_source: String,
    wgpu_source: String,
    metal_source: String,
    opencl_source: String,
    metadata: KernelMetadata,
}

impl BaseKernel {
    /// Create a new base kernel
    pub fn new(
        name: &str,
        cuda_source: &str,
        rocm_source: &str,
        wgpu_source: &str,
        metal_source: &str,
        opencl_source: &str,
        metadata: KernelMetadata,
    ) -> Self {
        Self {
            name: name.to_string(),
            cuda_source: cuda_source.to_string(),
            rocm_source: rocm_source.to_string(),
            wgpu_source: wgpu_source.to_string(),
            metal_source: metal_source.to_string(),
            opencl_source: opencl_source.to_string(),
            metadata,
        }
    }
}

impl GpuKernel for BaseKernel {
    fn name(&self) -> &str {
        &self.name
    }

    fn source_for_backend(&self, backend: GpuBackend) -> Result<String, GpuError> {
        match backend {
            GpuBackend::Cuda => Ok(self.cuda_source.clone()),
            GpuBackend::Rocm => Ok(self.rocm_source.clone()),
            GpuBackend::Wgpu => Ok(self.wgpu_source.clone()),
            GpuBackend::Metal => Ok(self.metal_source.clone()),
            GpuBackend::OpenCL => Ok(self.opencl_source.clone()),
            _ => Err(GpuError::UnsupportedBackend(backend)),
        }
    }

    fn metadata(&self) -> KernelMetadata {
        self.metadata.clone()
    }

    fn can_specialize(&self, _params: &KernelParams) -> bool {
        false // Base implementation doesn't support specialization
    }

    fn specialize(&self, _params: &KernelParams) -> Result<Box<dyn GpuKernel>, GpuError> {
        Err(GpuError::SpecializationNotSupported)
    }
}

/// Registry of available GPU kernels
pub struct KernelRegistry {
    kernels: HashMap<String, Box<dyn GpuKernel>>,
}

impl KernelRegistry {
    /// Create a new kernel registry
    pub fn new() -> Self {
        Self {
            kernels: HashMap::new(),
        }
    }

    /// Create a registry with all default kernels
    pub fn with_default_kernels() -> Self {
        let mut registry = Self::new();

        // Register BLAS kernels
        registry.register(Box::new(blas::gemm::GemmKernel::new()));
        registry.register(Box::new(blas::axpy::AxpyKernel::new()));

        // Register transform kernels
        registry.register(Box::new(transform::fft::FftKernel::new()));
        registry.register(Box::new(transform::convolution::Conv1dKernel::new()));
        registry.register(Box::new(transform::convolution::Conv2dKernel::new()));

        // Register reduction kernels
        registry.register(Box::new(reduction::sum::SumKernel::new()));
        registry.register(Box::new(reduction::norm::NormKernel::new()));
        registry.register(Box::new(reduction::min_max::MinKernel::new()));
        registry.register(Box::new(reduction::min_max::MaxKernel::new()));
        registry.register(Box::new(reduction::mean::MeanKernel::new()));
        registry.register(Box::new(reduction::std_dev::StdDevKernel::new()));

        // Register ML kernels
        registry.register(Box::new(ml::activation::ReluKernel::new()));
        registry.register(Box::new(ml::activation::SigmoidKernel::new()));
        registry.register(Box::new(ml::activation::TanhKernel::new()));
        registry.register(Box::new(ml::softmax::SoftmaxKernel::new()));
        registry.register(Box::new(ml::pooling::MaxPoolKernel::new()));
        registry.register(Box::new(ml::pooling::AvgPoolKernel::new()));

        // Register complex number kernels
        registry.register(Box::new(complex::ComplexMultiplyKernel::new()));
        registry.register(Box::new(complex::ComplexConjugateKernel::new()));
        registry.register(Box::new(complex::ComplexMatMulKernel::new()));

        registry
    }

    /// Register a kernel
    pub fn register(&mut self, kernel: Box<dyn GpuKernel>) {
        self.kernels.insert(kernel.name().to_string(), kernel);
    }

    /// Get a kernel by name
    pub fn get(&self, name: &str) -> Option<&dyn GpuKernel> {
        self.kernels.get(name).map(|k| k.as_ref())
    }

    /// Get a specialized kernel
    pub fn get_specialized(
        &self,
        name: &str,
        params: &KernelParams,
    ) -> Result<Box<dyn GpuKernel>, GpuError> {
        let kernel = self
            .get(name)
            .ok_or_else(|| GpuError::KernelNotFound(name.to_string()))?;

        if kernel.can_specialize(params) {
            kernel.specialize(params)
        } else {
            Err(GpuError::SpecializationNotSupported)
        }
    }
}

impl Default for KernelRegistry {
    fn default() -> Self {
        Self::with_default_kernels()
    }
}
