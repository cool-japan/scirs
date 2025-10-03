//! GPU-accelerated operations for sparse matrices
//!
//! This module provides GPU acceleration for sparse matrix operations
//! using the scirs2-core GPU backend system. The implementation has been
//! modularized for better maintainability and vendor-specific optimizations.

// Re-export all GPU operations from the modular structure
pub use crate::gpu::*;

// For backward compatibility, re-export the main functions and types

// Common GPU types and traits
#[cfg(feature = "gpu")]
pub use scirs2_core::gpu::{GpuBackend, GpuBuffer, GpuContext, GpuDataType, GpuKernelHandle};

#[cfg(feature = "gpu")]
pub use scirs2_core::GpuError;

// Fallback types when GPU feature is not enabled

// Fallback trait for GpuDataType when GPU feature is not enabled
#[cfg(not(feature = "gpu"))]
pub trait GpuDataType: Copy + Send + Sync + 'static {}

// Implement GpuDataType for common numeric types
#[cfg(not(feature = "gpu"))]
impl GpuDataType for f32 {}
#[cfg(not(feature = "gpu"))]
impl GpuDataType for f64 {}
#[cfg(not(feature = "gpu"))]
impl GpuDataType for i32 {}
#[cfg(not(feature = "gpu"))]
impl GpuDataType for i64 {}
#[cfg(not(feature = "gpu"))]
impl GpuDataType for u32 {}
#[cfg(not(feature = "gpu"))]
impl GpuDataType for u64 {}

#[cfg(not(feature = "gpu"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GpuBackend {
    #[default]
    Cpu,
    Cuda,
    OpenCL,
    Metal,
    Rocm,
    Wgpu,
}

#[cfg(not(feature = "gpu"))]
#[derive(Debug, Clone)]
pub struct GpuError(String);

#[cfg(not(feature = "gpu"))]
impl GpuError {
    pub fn new(msg: &str) -> Self {
        Self(msg.to_string())
    }

    pub fn invalid_buffer(msg: String) -> Self {
        Self(msg)
    }

    pub fn invalid_parameter(msg: String) -> Self {
        Self(msg)
    }

    pub fn kernel_compilation_error(msg: String) -> Self {
        Self(msg)
    }

    pub fn other(msg: String) -> Self {
        Self(msg)
    }
}

#[cfg(not(feature = "gpu"))]
impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[cfg(not(feature = "gpu"))]
impl std::error::Error for GpuError {}

#[cfg(not(feature = "gpu"))]
pub struct GpuBuffer<T> {
    data: Vec<T>,
}

#[cfg(not(feature = "gpu"))]
impl<T: Clone + Copy> GpuBuffer<T> {
    pub fn from_vec(data: Vec<T>) -> Self {
        Self { data }
    }

    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    pub fn to_vec(&self) -> Vec<T> {
        self.data.clone()
    }

    pub fn to_host(&self) -> Result<Vec<T>, GpuError> {
        Ok(self.data.clone())
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

#[cfg(not(feature = "gpu"))]
#[derive(Debug, Clone)]
pub struct GpuKernelHandle;

#[cfg(not(feature = "gpu"))]
pub struct GpuDevice {
    backend: GpuBackend,
}

#[cfg(not(feature = "gpu"))]
impl GpuDevice {
    pub fn new(_backend: GpuBackend) -> Result<Self, GpuError> {
        Ok(Self {
            backend: GpuBackend::Cpu,
        })
    }

    pub fn get_default(_backend: GpuBackend) -> Result<Self, GpuError> {
        Self::new(_backend)
    }

    pub fn backend(&self) -> GpuBackend {
        self.backend
    }

    pub fn create_buffer<T>(&self, data: &[T]) -> Result<GpuBuffer<T>, GpuError>
    where
        T: Clone + Copy,
    {
        Ok(GpuBuffer {
            data: data.to_vec(),
        })
    }

    pub fn create_buffer_zeros<T>(&self, size: usize) -> Result<GpuBuffer<T>, GpuError>
    where
        T: Clone + Copy + Default,
    {
        Ok(GpuBuffer {
            data: vec![T::default(); size],
        })
    }
}

// GPU data type implementations for compatibility with scirs2-core
// Note: GpuDataType trait is provided by scirs2-core

// Re-export unified GPU interface
pub use crate::gpu::{BackendInfo, GpuSpMatVec, OptimizationHint};

// Re-export convenience functions for backward compatibility
pub use crate::gpu::convenience::{available_backends, gpu_spmv, gpu_spmv_optimized};

// Legacy types for backward compatibility
pub struct AdvancedGpuOps {
    gpu_handler: GpuSpMatVec,
}

#[derive(Debug, Clone)]
pub struct GpuKernelScheduler {
    backend: GpuBackend,
}

#[derive(Debug, Clone)]
pub struct GpuMemoryManager {
    backend: GpuBackend,
}

#[derive(Debug, Clone)]
pub struct GpuOptions {
    pub backend: GpuBackend,
    pub optimization: OptimizationHint,
}

#[derive(Debug, Clone)]
pub struct GpuProfiler {
    enabled: bool,
}

pub struct OptimizedGpuOps {
    gpu_handler: GpuSpMatVec,
}

// Legacy function names for backward compatibility
use crate::csr_array::CsrArray;
use crate::error::SparseResult;
use scirs2_core::ndarray::{Array1, ArrayView1};
use scirs2_core::numeric::Float;
use std::fmt::Debug;

// GpuDataType is already defined above in this module

/// GPU sparse matrix-vector multiplication (legacy interface)
///
/// This function provides backward compatibility with the original API.
/// For new code, consider using the unified `GpuSpMatVec` interface.
#[allow(dead_code)]
pub fn gpu_sparse_matvec<T>(
    matrix: &CsrArray<T>,
    vector: &ArrayView1<T>,
    backend: Option<GpuBackend>,
) -> SparseResult<Array1<T>>
where
    T: Float + Debug + Copy + GpuDataType + std::iter::Sum,
{
    let gpu_handler = if let Some(backend) = backend {
        GpuSpMatVec::with_backend(backend)?
    } else {
        GpuSpMatVec::new()?
    };

    // For Metal backend on macOS, create a dummy device
    #[cfg(all(target_os = "macos", feature = "gpu"))]
    let device = if matches!(backend, Some(GpuBackend::Metal) | None) {
        Some(GpuDevice::new(GpuBackend::Metal, 0))
    } else {
        None
    };
    #[cfg(all(target_os = "macos", not(feature = "gpu")))]
    let device = if matches!(backend, Some(GpuBackend::Metal) | None) {
        GpuDevice::new(GpuBackend::Metal).ok()
    } else {
        None
    };
    #[cfg(not(target_os = "macos"))]
    let device = None;

    gpu_handler.spmv(matrix, vector, device.as_ref())
}

/// GPU symmetric sparse matrix-vector multiplication (legacy interface)
#[allow(dead_code)]
pub fn gpu_sym_sparse_matvec<T>(
    matrix: &CsrArray<T>,
    vector: &ArrayView1<T>,
    backend: Option<GpuBackend>,
) -> SparseResult<Array1<T>>
where
    T: Float + Debug + Copy + GpuDataType + std::iter::Sum,
{
    let gpu_handler = if let Some(backend) = backend {
        GpuSpMatVec::with_backend(backend)?
    } else {
        GpuSpMatVec::new()?
    };

    // For Metal backend on macOS, create a dummy device
    #[cfg(all(target_os = "macos", feature = "gpu"))]
    let device = if matches!(backend, Some(GpuBackend::Metal) | None) {
        Some(GpuDevice::new(GpuBackend::Metal, 0))
    } else {
        None
    };
    #[cfg(all(target_os = "macos", not(feature = "gpu")))]
    let device = if matches!(backend, Some(GpuBackend::Metal) | None) {
        GpuDevice::new(GpuBackend::Metal).ok()
    } else {
        None
    };
    #[cfg(not(target_os = "macos"))]
    let device = None;

    gpu_handler.spmv(matrix, vector, device.as_ref())
}

/// Advanced GPU sparse matrix-vector multiplication with optimization hints
#[allow(dead_code)]
pub fn gpu_advanced_spmv<T>(
    matrix: &CsrArray<T>,
    vector: &ArrayView1<T>,
    backend: Option<GpuBackend>,
    optimization: OptimizationHint,
) -> SparseResult<Array1<T>>
where
    T: Float + Debug + Copy + GpuDataType + std::iter::Sum,
{
    let gpu_handler = if let Some(backend) = backend {
        GpuSpMatVec::with_backend(backend)?
    } else {
        GpuSpMatVec::new()?
    };

    // For Metal backend on macOS, create a dummy device
    #[cfg(all(target_os = "macos", feature = "gpu"))]
    let device = if matches!(backend, Some(GpuBackend::Metal) | None) {
        Some(GpuDevice::new(GpuBackend::Metal, 0))
    } else {
        None
    };
    #[cfg(all(target_os = "macos", not(feature = "gpu")))]
    let device = if matches!(backend, Some(GpuBackend::Metal) | None) {
        GpuDevice::new(GpuBackend::Metal).ok()
    } else {
        None
    };
    #[cfg(not(target_os = "macos"))]
    let device = None;

    gpu_handler.spmv_optimized(matrix, vector, device.as_ref(), optimization)
}

// Legacy kernel and device management structures
#[allow(dead_code)]
pub struct SpMVKernel {
    gpu_handler: GpuSpMatVec,
}

impl SpMVKernel {
    pub fn new(_device: &GpuDevice, _workgroupsize: [u32; 3]) -> Result<Self, GpuError> {
        let gpu_handler = GpuSpMatVec::new().map_err(|e| {
            #[cfg(feature = "gpu")]
            return GpuError::Other(format!("{:?}", e));
            #[cfg(not(feature = "gpu"))]
            return GpuError::other(format!("{:?}", e));
        })?;
        Ok(Self { gpu_handler })
    }

    pub fn execute<T>(
        &self,
        matrix: &CsrArray<T>,
        vector: &ArrayView1<T>,
        device: &GpuDevice,
    ) -> Result<Array1<T>, GpuError>
    where
        T: Float + Debug + Copy + GpuDataType + std::iter::Sum,
    {
        self.gpu_handler
            .spmv(matrix, vector, Some(device))
            .map_err(|e| {
                #[cfg(feature = "gpu")]
                return GpuError::Other(format!("{:?}", e));
                #[cfg(not(feature = "gpu"))]
                return GpuError::other(format!("{:?}", e));
            })
    }
}

// GPU buffer extension trait for compatibility
pub trait GpuBufferExt<T: GpuDataType> {
    fn to_host(&self) -> Result<Vec<T>, GpuError>;
    fn to_host_range(&self, range: std::ops::Range<usize>) -> Result<Vec<T>, GpuError>;
}

impl<T: GpuDataType> GpuBufferExt<T> for GpuBuffer<T> {
    fn to_host(&self) -> Result<Vec<T>, GpuError> {
        Ok(self.to_vec())
    }

    fn to_host_range(&self, range: std::ops::Range<usize>) -> Result<Vec<T>, GpuError> {
        let full_data = self.to_vec();
        if range.end <= full_data.len() {
            Ok(full_data[range].to_vec())
        } else {
            #[cfg(feature = "gpu")]
            return Err(GpuError::InvalidParameter(
                "Range out of bounds".to_string(),
            ));
            #[cfg(not(feature = "gpu"))]
            return Err(GpuError::invalid_parameter(
                "Range out of bounds".to_string(),
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    #[test]
    fn test_backward_compatibility_gpu_sparse_matvec() {
        // Create a simple CSR matrix for testing
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let indices = vec![0, 1, 0, 1];
        let indptr = vec![0, 2, 4];
        let matrix = CsrArray::new(data.into(), indices.into(), indptr.into(), (2, 2)).unwrap();

        let vector = Array1::from_vec(vec![1.0, 2.0]);

        // Test with automatic backend selection
        // This may fail if GPU hardware is not available, which is expected in CI
        let result = gpu_sparse_matvec(&matrix, &vector.view(), None);
        if let Err(e) = &result {
            eprintln!("Error from gpu_sparse_matvec: {:?}", e);
            // If GPU hardware is not available, we should get a specific error
            // and that's acceptable for testing purposes
            let error_msg = format!("{:?}", e);
            assert!(
                error_msg.contains("GPU device required")
                    || error_msg.contains("not initialized")
                    || error_msg.contains("not available"),
                "Unexpected error: {:?}",
                e
            );
        } else {
            // If it succeeds, that's also fine
            assert!(result.is_ok());
        }

        // Test with specific CPU backend - this should always work
        let result = gpu_sparse_matvec(&matrix, &vector.view(), Some(GpuBackend::Cpu));
        if let Err(e) = &result {
            eprintln!("Error from gpu_sparse_matvec with CPU backend: {:?}", e);
        }
        assert!(result.is_ok(), "CPU backend should always work");
    }

    #[test]
    fn test_gpu_spmv_kernel() {
        #[cfg(feature = "gpu")]
        let device = scirs2_core::gpu::GpuDevice::new(GpuBackend::Cpu, 0);
        #[cfg(not(feature = "gpu"))]
        let device = GpuDevice::new(GpuBackend::Cpu).unwrap();

        let kernel = SpMVKernel::new(&device, [1, 1, 1]);
        assert!(kernel.is_ok());
    }

    #[test]
    fn test_gpu_buffer_ext() {
        #[cfg(not(feature = "gpu"))]
        {
            let buffer = GpuBuffer {
                data: vec![1.0, 2.0, 3.0, 4.0],
            };
            let host_data = buffer.to_host().unwrap();
            assert_eq!(host_data, vec![1.0, 2.0, 3.0, 4.0]);

            let range_data = buffer.to_host_range(1..3).unwrap();
            assert_eq!(range_data, vec![2.0, 3.0]);
        }
    }

    #[test]
    fn test_gpu_data_types() {
        // Test that the trait is implemented for expected types
        fn is_gpu_data_type<T: GpuDataType>() {}

        is_gpu_data_type::<f32>();
        is_gpu_data_type::<f64>();
        is_gpu_data_type::<u32>();
        is_gpu_data_type::<u64>();
        is_gpu_data_type::<i32>();
        is_gpu_data_type::<i64>();
    }

    #[test]
    fn test_gpu_backend_enum() {
        let backends = [
            GpuBackend::Cpu,
            GpuBackend::Cuda,
            GpuBackend::OpenCL,
            GpuBackend::Metal,
            GpuBackend::Rocm,
            GpuBackend::Wgpu,
        ];

        for backend in &backends {
            match backend {
                GpuBackend::Cpu => (),
                GpuBackend::Cuda => (),
                GpuBackend::OpenCL => (),
                GpuBackend::Metal => (),
                GpuBackend::Rocm => (),
                GpuBackend::Wgpu => (),
            }
        }
    }

    #[test]
    fn test_available_backends() {
        let backends = available_backends();
        assert!(!backends.is_empty());
        assert!(backends.contains(&GpuBackend::Cpu)); // CPU should always be available
    }
}
