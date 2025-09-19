//! CUDA backend for sparse matrix GPU operations
//!
//! This module provides CUDA-specific implementations for sparse matrix operations.

use crate::csr_array::CsrArray;
use crate::error::{SparseError, SparseResult};
use crate::sparray::SparseArray;
use ndarray::{Array1, ArrayView1};
use num_traits::Float;
use std::fmt::Debug;

#[cfg(feature = "gpu")]
use crate::gpu_kernel_execution::{GpuKernelConfig, MemoryStrategy};

#[cfg(feature = "gpu")]
pub use scirs2_core::gpu::{GpuBackend, GpuBuffer, GpuContext, GpuDataType, GpuKernelHandle};

#[cfg(feature = "gpu")]
pub use scirs2_core::GpuError;

/// CUDA kernel source code for sparse matrix-vector multiplication
pub const CUDA_SPMV_KERNEL_SOURCE: &str = r#"
extern "C" __global__ void spmv_csr_kernel(
    int rows,
    const int* __restrict__ indptr,
    const int* __restrict__ indices,
    const float* __restrict__ data,
    const float* __restrict__ x,
    float* __restrict__ y
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    
    float sum = 0.0f;
    int start = indptr[row];
    int end = indptr[row + 1];
    
    // Vectorized loop for better memory access patterns
    for (int j = start; j < end; j++) {
        sum += data[j] * x[indices[j]];
    }
    
    y[row] = sum;
}

extern "C" __global__ void spmv_csr_vectorized_kernel(
    int rows,
    const int* __restrict__ indptr,
    const int* __restrict__ indices,
    const float* __restrict__ data,
    const float* __restrict__ x,
    float* __restrict__ y
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    
    float sum = 0.0f;
    int start = indptr[row];
    int end = indptr[row + 1];
    
    // Use shared memory for better performance
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    
    sdata[tid] = 0.0f;
    __syncthreads();
    
    for (int j = start; j < end; j++) {
        sdata[tid] += data[j] * x[indices[j]];
    }
    
    __syncthreads();
    y[row] = sdata[tid];
}
"#;

/// CUDA warp-level sparse matrix-vector multiplication kernel
pub const CUDA_WARP_SPMV_KERNEL_SOURCE: &str = r#"
extern "C" __global__ void spmv_csr_warp_kernel(
    int rows,
    const int* __restrict__ indptr,
    const int* __restrict__ indices,
    const float* __restrict__ data,
    const float* __restrict__ x,
    float* __restrict__ y
) {
    int warp_id = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = threadIdx.x % 32;
    int row = warp_id / 32;
    
    if (row >= rows) return;
    
    int start = indptr[row];
    int end = indptr[row + 1];
    float sum = 0.0f;
    
    // Warp-level parallelization
    for (int j = start + lane_id; j < end; j += 32) {
        sum += data[j] * x[indices[j]];
    }
    
    // Warp reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    if (lane_id == 0) {
        y[row] = sum;
    }
}
"#;

/// CUDA sparse matrix operations
#[derive(Debug, Clone)]
pub struct CudaSpMatVec {
    kernel_handle: Option<super::GpuKernelHandle>,
    vectorized_kernel: Option<super::GpuKernelHandle>,
    warp_kernel: Option<super::GpuKernelHandle>,
}

impl CudaSpMatVec {
    /// Create a new CUDA sparse matrix-vector multiplication handler
    pub fn new() -> SparseResult<Self> {
        Ok(Self {
            kernel_handle: None,
            vectorized_kernel: None,
            warp_kernel: None,
        })
    }

    /// Compile CUDA kernels for sparse matrix operations
    #[cfg(feature = "gpu")]
    pub fn compile_kernels(&mut self, device: &super::GpuDevice) -> Result<(), super::GpuError> {
        // Compile standard kernel
        self.kernel_handle =
            Some(device.compile_kernel(CUDA_SPMV_KERNEL_SOURCE, "spmv_csr_kernel")?);

        // Compile vectorized kernel
        self.vectorized_kernel =
            Some(device.compile_kernel(CUDA_SPMV_KERNEL_SOURCE, "spmv_csr_vectorized_kernel")?);

        // Compile warp-level kernel
        self.warp_kernel =
            Some(device.compile_kernel(CUDA_WARP_SPMV_KERNEL_SOURCE, "spmv_csr_warp_kernel")?);

        Ok(())
    }

    /// Execute CUDA sparse matrix-vector multiplication
    #[cfg(feature = "gpu")]
    pub fn execute_spmv<T>(
        &self,
        matrix: &CsrArray<T>,
        vector: &ArrayView1<T>,
        device: &super::GpuDevice,
    ) -> SparseResult<Array1<T>>
    where
        T: Float + Debug + Copy + super::GpuDataType,
    {
        let (rows, cols) = matrix.shape();
        if cols != vector.len() {
            return Err(SparseError::DimensionMismatch {
                expected: cols,
                found: vector.len(),
            });
        }

        // Upload data to GPU
        let indptr_gpu = device.create_buffer(&matrix.indptr)?;
        let indices_gpu = device.create_buffer(&matrix.indices)?;
        let data_gpu = device.create_buffer(&matrix.data)?;
        let vector_gpu = device.create_buffer(vector.as_slice().unwrap())?;
        let mut result_gpu = device.create_buffer(&vec![T::zero(); rows])?;

        // Launch kernel
        if let Some(ref kernel) = self.kernel_handle {
            let grid_size = (rows + 255) / 256;
            let block_size = 256;

            device.launch_kernel(
                kernel,
                [grid_size as u32, 1, 1],
                [block_size as u32, 1, 1],
                &[
                    &(rows as i32),
                    &indptr_gpu,
                    &indices_gpu,
                    &data_gpu,
                    &vector_gpu,
                    &mut result_gpu,
                ],
            )?;
        } else {
            return Err(SparseError::ComputationError(
                "CUDA kernel not compiled".to_string(),
            ));
        }

        // Download result from GPU
        let result_host = result_gpu.to_host()?;
        Ok(Array1::from_vec(result_host))
    }

    /// Execute optimized CUDA sparse matrix-vector multiplication
    #[cfg(feature = "gpu")]
    pub fn execute_optimized_spmv<T>(
        &self,
        matrix: &CsrArray<T>,
        vector: &ArrayView1<T>,
        device: &super::GpuDevice,
        optimization_level: CudaOptimizationLevel,
    ) -> SparseResult<Array1<T>>
    where
        T: Float + Debug + Copy + super::GpuDataType,
    {
        let (rows, cols) = matrix.shape();
        if cols != vector.len() {
            return Err(SparseError::DimensionMismatch {
                expected: cols,
                found: vector.len(),
            });
        }

        // Choose kernel based on optimization level
        let kernel = match optimization_level {
            CudaOptimizationLevel::Basic => &self.kernel_handle,
            CudaOptimizationLevel::Vectorized => &self.vectorized_kernel,
            CudaOptimizationLevel::WarpLevel => &self.warp_kernel,
        };

        if let Some(ref k) = kernel {
            self.execute_kernel_with_optimization(matrix, vector, device, k, optimization_level)
        } else {
            Err(SparseError::ComputationError(
                "CUDA kernel not available for requested optimization level".to_string(),
            ))
        }
    }

    #[cfg(feature = "gpu")]
    fn execute_kernel_with_optimization<T>(
        &self,
        matrix: &CsrArray<T>,
        vector: &ArrayView1<T>,
        device: &super::GpuDevice,
        kernel: &super::GpuKernelHandle,
        optimization_level: CudaOptimizationLevel,
    ) -> SparseResult<Array1<T>>
    where
        T: Float + Debug + Copy + super::GpuDataType,
    {
        let (rows, _) = matrix.shape();

        // Upload data to GPU with memory optimization
        let indptr_gpu = device.create_buffer(&matrix.indptr)?;
        let indices_gpu = device.create_buffer(&matrix.indices)?;
        let data_gpu = device.create_buffer(&matrix.data)?;
        let vector_gpu = device.create_buffer(vector.as_slice().unwrap())?;
        let mut result_gpu = device.create_buffer(&vec![T::zero(); rows])?;

        // Configure launch parameters based on optimization level
        let (grid_size, block_size, shared_memory) = match optimization_level {
            CudaOptimizationLevel::Basic => ((rows + 255) / 256, 256, 0),
            CudaOptimizationLevel::Vectorized => {
                ((rows + 127) / 128, 128, 128 * std::mem::size_of::<f32>())
            }
            CudaOptimizationLevel::WarpLevel => ((rows * 32 + 255) / 256, 256, 0),
        };

        // Launch kernel with optimized parameters
        device.launch_kernel_with_shared_memory(
            kernel,
            [grid_size as u32, 1, 1],
            [block_size as u32, 1, 1],
            shared_memory,
            &[
                &(rows as i32),
                &indptr_gpu,
                &indices_gpu,
                &data_gpu,
                &vector_gpu,
                &mut result_gpu,
            ],
        )?;

        // Download result
        let result_host = result_gpu.to_host()?;
        Ok(Array1::from_vec(result_host))
    }

    /// CPU fallback implementation
    #[cfg(not(feature = "gpu"))]
    pub fn execute_spmv_cpu<T>(
        &self,
        matrix: &CsrArray<T>,
        vector: &ArrayView1<T>,
    ) -> SparseResult<Array1<T>>
    where
        T: Float + Debug + Copy + std::iter::Sum,
    {
        matrix.dot_vector(vector)
    }
}

impl Default for CudaSpMatVec {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_| Self {
            kernel_handle: None,
            vectorized_kernel: None,
            warp_kernel: None,
        })
    }
}

/// CUDA optimization levels for sparse matrix operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaOptimizationLevel {
    /// Basic thread-per-row implementation
    Basic,
    /// Vectorized implementation with shared memory
    Vectorized,
    /// Warp-level implementation for better memory coalescing
    WarpLevel,
}

impl Default for CudaOptimizationLevel {
    fn default() -> Self {
        Self::Basic
    }
}

/// CUDA memory management for sparse matrices
pub struct CudaMemoryManager {
    #[allow(dead_code)]
    allocated_buffers: Vec<String>,
}

impl CudaMemoryManager {
    /// Create a new CUDA memory manager
    pub fn new() -> Self {
        Self {
            allocated_buffers: Vec::new(),
        }
    }

    /// Allocate GPU memory for sparse matrix data
    #[cfg(feature = "gpu")]
    pub fn allocate_sparse_matrix<T>(
        &mut self,
        matrix: &CsrArray<T>,
        device: &super::GpuDevice,
    ) -> Result<CudaMatrixBuffers<T>, super::GpuError>
    where
        T: super::GpuDataType + Copy + Float + Debug,
    {
        let indptr_buffer = device.create_buffer(&matrix.indptr)?;
        let indices_buffer = device.create_buffer(&matrix.indices)?;
        let data_buffer = device.create_buffer(&matrix.data)?;

        Ok(CudaMatrixBuffers {
            indptr: indptr_buffer,
            indices: indices_buffer,
            data: data_buffer,
        })
    }

    /// Allocate GPU memory with optimal memory layout
    #[cfg(feature = "gpu")]
    pub fn allocate_optimized<T>(
        &mut self,
        data: &[T],
        device: &super::GpuDevice,
        access_pattern: MemoryAccessPattern,
    ) -> Result<super::GpuBuffer<T>, super::GpuError>
    where
        T: super::GpuDataType + Copy + Float + Debug,
    {
        match access_pattern {
            MemoryAccessPattern::Sequential => {
                // Use default allocation for sequential access
                device.create_buffer(data)
            }
            MemoryAccessPattern::Random => {
                // Use cache-optimized allocation for random access
                device.create_buffer_cached(data)
            }
            MemoryAccessPattern::Strided => {
                // Use aligned allocation for strided access
                device.create_buffer_aligned(data, 128)
            }
        }
    }
}

impl Default for CudaMemoryManager {
    fn default() -> Self {
        Self::new()
    }
}

/// GPU memory buffers for sparse matrix data
#[cfg(feature = "gpu")]
pub struct CudaMatrixBuffers<T: super::GpuDataType> {
    pub indptr: super::GpuBuffer<usize>,
    pub indices: super::GpuBuffer<usize>,
    pub data: super::GpuBuffer<T>,
}

/// Memory access patterns for optimization
#[derive(Debug, Clone, Copy)]
pub enum MemoryAccessPattern {
    /// Sequential memory access
    Sequential,
    /// Random memory access
    Random,
    /// Strided memory access
    Strided,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_cuda_spmv_creation() {
        let cuda_spmv = CudaSpMatVec::new();
        assert!(cuda_spmv.is_ok());
    }

    #[test]
    fn test_cuda_optimization_levels() {
        let basic = CudaOptimizationLevel::Basic;
        let vectorized = CudaOptimizationLevel::Vectorized;
        let warp = CudaOptimizationLevel::WarpLevel;

        assert_ne!(basic, vectorized);
        assert_ne!(vectorized, warp);
        assert_eq!(
            CudaOptimizationLevel::default(),
            CudaOptimizationLevel::Basic
        );
    }

    #[test]
    fn test_cuda_memory_manager() {
        let manager = CudaMemoryManager::new();
        assert_eq!(manager.allocated_buffers.len(), 0);
    }

    #[test]
    fn test_memory_access_patterns() {
        let patterns = [
            MemoryAccessPattern::Sequential,
            MemoryAccessPattern::Random,
            MemoryAccessPattern::Strided,
        ];

        // Test that all patterns are defined
        for pattern in &patterns {
            match pattern {
                MemoryAccessPattern::Sequential => (),
                MemoryAccessPattern::Random => (),
                MemoryAccessPattern::Strided => (),
            }
        }
    }

    #[test]
    fn test_kernel_sources() {
        assert!(!CUDA_SPMV_KERNEL_SOURCE.is_empty());
        assert!(!CUDA_WARP_SPMV_KERNEL_SOURCE.is_empty());

        // Check that kernels contain expected function names
        assert!(CUDA_SPMV_KERNEL_SOURCE.contains("spmv_csr_kernel"));
        assert!(CUDA_SPMV_KERNEL_SOURCE.contains("spmv_csr_vectorized_kernel"));
        assert!(CUDA_WARP_SPMV_KERNEL_SOURCE.contains("spmv_csr_warp_kernel"));
    }
}
