//! Backend delegation system for GPU acceleration
//!
//! This module provides a unified interface for delegating operations
//! to different computational backends (CPU, CUDA, OpenCL, etc.).
//! It allows seamless switching between implementations based on
//! hardware availability and performance characteristics.

pub mod concrete_gpu_backends;
pub mod device_detection;
pub mod gpu_acceleration_framework;
pub mod kernels;

#[cfg(feature = "cuda")]
pub mod cuda;

pub use device_detection::{DeviceCapability, DeviceManager, MemoryManager, SystemCapabilities};
pub use gpu_acceleration_framework::{
    CompiledKernel, GpuAccelerationManager, GpuKernelCache, GpuMemoryPool, GpuPerformanceReport,
    KernelPerformanceStats, MemoryPoolConfig, MemoryPoolStatistics,
};
pub use kernels::{GpuBuffer, GpuKernelExecutor, KernelInfo};

#[cfg(feature = "cuda")]
pub use concrete_gpu_backends::CudaContext;
#[cfg(feature = "opencl")]
pub use concrete_gpu_backends::OpenCLContext;
// TODO: Implement MetalContext in concrete_gpu_backends.rs
// #[cfg(all(target_os = "macos", feature = "metal"))]
// pub use concrete_gpu_backends::MetalContext;

use crate::error::{NdimageError, NdimageResult};
use scirs2_core::ndarray::{Array, ArrayView, Dimension};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::Debug;
use std::sync::Arc;

/// Available computation backends
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Backend {
    /// CPU-based implementation (default)
    Cpu,
    /// NVIDIA CUDA GPU acceleration
    #[cfg(feature = "cuda")]
    Cuda,
    /// OpenCL GPU acceleration
    #[cfg(feature = "opencl")]
    OpenCL,
    /// Apple Metal GPU acceleration
    #[cfg(all(target_os = "macos", feature = "metal"))]
    Metal,
    /// Automatic selection based on operation and data size
    Auto,
}

impl Default for Backend {
    fn default() -> Self {
        Backend::Cpu
    }
}

/// Configuration for backend selection and operation
#[derive(Debug, Clone)]
pub struct BackendConfig {
    /// Preferred backend
    pub backend: Backend,
    /// Minimum array size for GPU acceleration (elements)
    pub gpu_threshold: usize,
    /// Maximum GPU memory to use (bytes)
    pub gpu_memory_limit: Option<usize>,
    /// Whether to allow fallback to CPU if GPU fails
    pub allow_fallback: bool,
    /// Device ID for multi-GPU systems
    pub device_id: Option<usize>,
}

impl Default for BackendConfig {
    fn default() -> Self {
        Self {
            backend: Backend::default(),
            gpu_threshold: 100_000, // 100k elements minimum for GPU
            gpu_memory_limit: None,
            allow_fallback: true,
            device_id: None,
        }
    }
}

/// Trait for operations that can be delegated to different backends
pub trait BackendOp<T, D>: Send + Sync
where
    T: Float + FromPrimitive + Debug + Clone,
    D: Dimension,
{
    /// Execute operation on CPU
    fn execute_cpu(&self, input: &ArrayView<T, D>) -> NdimageResult<Array<T, D>>;

    /// Execute operation on GPU (if available)
    #[cfg(feature = "gpu")]
    fn execute_gpu(&self, input: &ArrayView<T, D>, backend: Backend) -> NdimageResult<Array<T, D>>;

    /// Get estimated memory requirements
    fn memory_requirement(&self, input_shape: &[usize]) -> usize;

    /// Check if this operation benefits from GPU acceleration
    fn benefits_from_gpu(&self, array_size: usize) -> bool {
        array_size > 50_000 // Default threshold
    }
}

/// Backend executor that handles delegation
pub struct BackendExecutor {
    config: BackendConfig,
    #[cfg(feature = "gpu")]
    gpu_context: Option<Arc<dyn GpuContext>>,
}

impl BackendExecutor {
    pub fn new(config: BackendConfig) -> NdimageResult<Self> {
        #[cfg(feature = "gpu")]
        let gpu_context: Option<Arc<dyn GpuContext>> = match config.backend {
            #[cfg(feature = "cuda")]
            Backend::Cuda => Some(Arc::new(CudaContext::new(config.device_id)?)),
            #[cfg(feature = "opencl")]
            Backend::OpenCL => Some(Arc::new(OpenCLContext::new(config.device_id)?)),
            // TODO: Implement Metal backend
            // #[cfg(all(target_os = "macos", feature = "metal"))]
            // Backend::Metal => Some(Arc::new(MetalContext::new(_config.device_id)?),
            _ => None,
        };

        Ok(Self {
            config,
            #[cfg(feature = "gpu")]
            gpu_context,
        })
    }

    /// Execute an operation with automatic backend selection
    pub fn execute<T, D, Op>(&self, input: &ArrayView<T, D>, op: Op) -> NdimageResult<Array<T, D>>
    where
        T: Float + FromPrimitive + Debug + Clone + Send + Sync + 'static,
        D: Dimension,
        Op: BackendOp<T, D>,
    {
        let array_size = input.len();
        let backend = self.select_backend(&op, array_size)?;

        match backend {
            Backend::Cpu => op.execute_cpu(input),
            #[cfg(feature = "gpu")]
            _ => match op.execute_gpu(input, backend) {
                Ok(result) => Ok(result),
                Err(e) if self.config.allow_fallback => {
                    eprintln!("GPU execution failed, falling back to CPU: {}", e);
                    op.execute_cpu(input)
                }
                Err(e) => Err(e),
            },
            #[cfg(not(feature = "gpu"))]
            _ => op.execute_cpu(input),
        }
    }

    /// Select the best backend for an operation
    fn select_backend<T, D, Op>(&self, op: &Op, array_size: usize) -> NdimageResult<Backend>
    where
        T: Float + FromPrimitive + Debug + Clone,
        D: Dimension,
        Op: BackendOp<T, D>,
    {
        match self.config.backend {
            Backend::Auto => {
                // Automatic selection based on heuristics
                if array_size < self.config.gpu_threshold {
                    Ok(Backend::Cpu)
                } else if op.benefits_from_gpu(array_size) {
                    // Check available GPU backends
                    #[cfg(feature = "cuda")]
                    if self.is_cuda_available() {
                        return Ok(Backend::Cuda);
                    }
                    #[cfg(feature = "opencl")]
                    if self.is_opencl_available() {
                        return Ok(Backend::OpenCL);
                    }
                    #[cfg(all(target_os = "macos", feature = "metal"))]
                    if self.is_metal_available() {
                        return Ok(Backend::Metal);
                    }
                    Ok(Backend::Cpu)
                } else {
                    Ok(Backend::Cpu)
                }
            }
            backend => Ok(backend),
        }
    }

    #[cfg(feature = "cuda")]
    fn is_cuda_available(&self) -> bool {
        device_detection::get_device_manager()
            .map(|manager| manager.lock().unwrap().is_backend_available(Backend::Cuda))
            .unwrap_or(false)
    }

    #[cfg(feature = "opencl")]
    fn is_opencl_available(&self) -> bool {
        device_detection::get_device_manager()
            .map(|manager| {
                manager
                    .lock()
                    .unwrap()
                    .is_backend_available(Backend::OpenCL)
            })
            .unwrap_or(false)
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn is_metal_available(&self) -> bool {
        device_detection::get_device_manager()
            .map(|manager| manager.lock().unwrap().is_backend_available(Backend::Metal))
            .unwrap_or(false)
    }
}

/// GPU context trait for different GPU backends
#[cfg(feature = "gpu")]
pub trait GpuContext: Send + Sync {
    fn name(&self) -> &str;
    fn device_count(&self) -> usize;
    fn current_device(&self) -> usize;
    fn memory_info(&self) -> (usize, usize); // (used, total)
}

/// Example: Gaussian filter with backend support
pub struct GaussianFilterOp<T> {
    sigma: Vec<T>,
    truncate: Option<T>,
}

impl<T: Float + FromPrimitive + Debug + Clone> GaussianFilterOp<T> {
    pub fn new(sigma: Vec<T>, truncate: Option<T>) -> Self {
        Self { sigma, truncate }
    }
}

impl<T, D> BackendOp<T, D> for GaussianFilterOp<T>
where
    T: Float + FromPrimitive + Debug + Clone + Default + Send + Sync + 'static,
    D: Dimension + 'static,
{
    fn execute_cpu(&self, input: &ArrayView<T, D>) -> NdimageResult<Array<T, D>> {
        // Call the existing CPU implementation
        crate::filters::gaussian_filter_chunked(
            &input.to_owned(),
            &self.sigma,
            self.truncate,
            crate::filters::BorderMode::Reflect,
            None,
        )
    }

    #[cfg(feature = "gpu")]
    fn execute_gpu(&self, input: &ArrayView<T, D>, backend: Backend) -> NdimageResult<Array<T, D>> {
        match backend {
            #[cfg(feature = "cuda")]
            Backend::Cuda => {
                // Would call CUDA implementation
                cuda_gaussian_filter(input, &self.sigma, self.truncate)
            }
            _ => self.execute_cpu(input),
        }
    }

    fn memory_requirement(&self, input_shape: &[usize]) -> usize {
        let elements: usize = input_shape.iter().product();
        // Input + output + temporary buffers
        elements * std::mem::size_of::<T>() * 3
    }

    fn benefits_from_gpu(&self, array_size: usize) -> bool {
        // Gaussian filter benefits from GPU for large arrays
        array_size > 100_000
    }
}

#[cfg(feature = "cuda")]
#[allow(dead_code)]
fn cuda_gaussian_filter<T, D>(
    input: &ArrayView<T, D>,
    sigma: &[T],
    _truncate: Option<T>,
) -> NdimageResult<Array<T, D>>
where
    T: Float + FromPrimitive + Debug + Clone + Default + Send + Sync + 'static,
    D: Dimension,
{
    // Currently only support 2D arrays for GPU acceleration
    if input.ndim() == 2 {
        let input_2d = input
            .view()
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .map_err(|_| NdimageError::DimensionError("Failed to convert to 2D array".into()))?;

        if sigma.len() >= 2 {
            let sigma_2d = [sigma[0], sigma[1]];
            let cuda_ops = cuda::CudaOperations::new(None)?;
            let result_2d = cuda_ops.gaussian_filter_2d(&input_2d, sigma_2d)?;

            // Convert back to original dimension
            let result = result_2d.into_dimensionality::<D>().map_err(|_| {
                NdimageError::DimensionError("Failed to convert result dimension".into())
            })?;
            return Ok(result);
        }
    }

    // Fallback for non-2D or unsupported cases
    Err(NdimageError::NotImplementedError(
        "CUDA Gaussian filter currently only supports 2D arrays".into(),
    ))
}

/// Builder for creating backend executors
pub struct BackendBuilder {
    config: BackendConfig,
}

impl BackendBuilder {
    pub fn new() -> Self {
        Self {
            config: BackendConfig::default(),
        }
    }

    pub fn backend(mut self, backend: Backend) -> Self {
        self.config.backend = backend;
        self
    }

    pub fn gpu_threshold(mut self, threshold: usize) -> Self {
        self.config.gpu_threshold = threshold;
        self
    }

    pub fn gpu_memory_limit(mut self, limit: usize) -> Self {
        self.config.gpu_memory_limit = Some(limit);
        self
    }

    pub fn allow_fallback(mut self, allow: bool) -> Self {
        self.config.allow_fallback = allow;
        self
    }

    pub fn device_id(mut self, id: usize) -> Self {
        self.config.device_id = Some(id);
        self
    }

    pub fn build(self) -> NdimageResult<BackendExecutor> {
        BackendExecutor::new(self.config)
    }
}

/// Convenience function to create an auto-selecting backend executor
#[allow(dead_code)]
pub fn auto_backend() -> NdimageResult<BackendExecutor> {
    BackendBuilder::new().backend(Backend::Auto).build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::arr2;

    #[test]
    fn test_backend_selection() {
        let config = BackendConfig {
            backend: Backend::Auto,
            gpu_threshold: 1000,
            ..Default::default()
        };

        let executor = BackendExecutor::new(config).unwrap();
        let small_array = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let op = GaussianFilterOp::new(vec![1.0, 1.0], None);

        // Small array should use CPU
        let _result = executor.execute(&small_array.view(), op).unwrap();
    }

    #[test]
    fn test_backend_builder() {
        let executor = BackendBuilder::new()
            .backend(Backend::Cpu)
            .gpu_threshold(50_000)
            .allow_fallback(true)
            .build()
            .unwrap();

        assert_eq!(executor.config.backend, Backend::Cpu);
        assert_eq!(executor.config.gpu_threshold, 50_000);
        assert!(executor.config.allow_fallback);
    }
}
