//! OpenCL backend for sparse matrix GPU operations
//!
//! This module provides OpenCL-specific implementations for sparse matrix operations.

use crate::csr_array::CsrArray;
use crate::error::{SparseError, SparseResult};
use crate::sparray::SparseArray;
use scirs2_core::ndarray::{Array1, ArrayView1};
use scirs2_core::numeric::Float;
use std::fmt::Debug;

#[cfg(feature = "gpu")]
use crate::gpu_kernel_execution::{GpuKernelConfig, MemoryStrategy};

#[cfg(feature = "gpu")]
pub use scirs2_core::gpu::{GpuBackend, GpuBuffer, GpuContext, GpuDataType, GpuKernelHandle};

#[cfg(feature = "gpu")]
pub use scirs2_core::GpuError;

/// OpenCL kernel source code for sparse matrix-vector multiplication
pub const OPENCL_SPMV_KERNEL_SOURCE: &str = r#"
__kernel void spmv_csr_kernel(
    int rows,
    __global const int* restrict indptr,
    __global const int* restrict indices,
    __global const float* restrict data,
    __global const float* restrict x,
    __global float* restrict y
) {
    int row = get_global_id(0);
    if (row >= rows) return;
    
    float sum = 0.0f;
    int start = indptr[row];
    int end = indptr[row + 1];
    
    for (int j = start; j < end; j++) {
        sum += data[j] * x[indices[j]];
    }
    
    y[row] = sum;
}

__kernel void spmv_csr_workgroup_kernel(
    int rows,
    __global const int* restrict indptr,
    __global const int* restrict indices,
    __global const float* restrict data,
    __global const float* restrict x,
    __global float* restrict y,
    __local float* local_mem
) {
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int group_size = get_local_size(0);
    
    if (gid >= rows) return;
    
    int start = indptr[gid];
    int end = indptr[gid + 1];
    
    local_mem[lid] = 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for (int j = start; j < end; j++) {
        local_mem[lid] += data[j] * x[indices[j]];
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    y[gid] = local_mem[lid];
}
"#;

/// OpenCL vectorized kernel for better performance
pub const OPENCL_VECTORIZED_KERNEL_SOURCE: &str = r#"
__kernel void spmv_csr_vectorized_kernel(
    int rows,
    __global const int* restrict indptr,
    __global const int* restrict indices,
    __global const float* restrict data,
    __global const float* restrict x,
    __global float* restrict y
) {
    int row = get_global_id(0);
    if (row >= rows) return;
    
    int start = indptr[row];
    int end = indptr[row + 1];
    int nnz = end - start;
    
    float4 sum = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    
    // Process 4 elements at a time when possible
    int j;
    for (j = start; j + 3 < end; j += 4) {
        float4 data_vec = (float4)(data[j], data[j+1], data[j+2], data[j+3]);
        float4 x_vec = (float4)(
            x[indices[j]], 
            x[indices[j+1]], 
            x[indices[j+2]], 
            x[indices[j+3]]
        );
        sum += data_vec * x_vec;
    }
    
    float scalar_sum = sum.x + sum.y + sum.z + sum.w;
    
    // Handle remaining elements
    for (; j < end; j++) {
        scalar_sum += data[j] * x[indices[j]];
    }
    
    y[row] = scalar_sum;
}
"#;

/// OpenCL sparse matrix operations
pub struct OpenCLSpMatVec {
    context: Option<scirs2_core::gpu::GpuContext>,
    kernel_handle: Option<scirs2_core::gpu::GpuKernelHandle>,
    workgroup_kernel: Option<scirs2_core::gpu::GpuKernelHandle>,
    vectorized_kernel: Option<scirs2_core::gpu::GpuKernelHandle>,
    platform_info: OpenCLPlatformInfo,
}

impl OpenCLSpMatVec {
    /// Create a new OpenCL sparse matrix-vector multiplication handler
    pub fn new() -> SparseResult<Self> {
        // Try to create OpenCL context
        #[cfg(feature = "gpu")]
        let context = match scirs2_core::gpu::GpuContext::new(scirs2_core::gpu::GpuBackend::OpenCL)
        {
            Ok(ctx) => Some(ctx),
            Err(_) => None, // OpenCL not available, will use CPU fallback
        };
        #[cfg(not(feature = "gpu"))]
        let context = None;

        let mut handler = Self {
            context,
            kernel_handle: None,
            workgroup_kernel: None,
            vectorized_kernel: None,
            platform_info: OpenCLPlatformInfo::detect(),
        };

        // Compile kernels if context is available
        #[cfg(feature = "gpu")]
        if handler.context.is_some() {
            let _ = handler.compile_kernels();
        }

        Ok(handler)
    }

    /// Compile OpenCL kernels for sparse matrix operations
    #[cfg(feature = "gpu")]
    pub fn compile_kernels(&mut self) -> Result<(), scirs2_core::gpu::GpuError> {
        if let Some(ref context) = self.context {
            // Compile basic kernel
            self.kernel_handle =
                context.execute(|compiler| compiler.compile(OPENCL_SPMV_KERNEL_SOURCE).ok());

            // Compile workgroup-optimized kernel
            self.workgroup_kernel =
                context.execute(|compiler| compiler.compile(OPENCL_SPMV_KERNEL_SOURCE).ok());

            // Compile vectorized kernel
            self.vectorized_kernel =
                context.execute(|compiler| compiler.compile(OPENCL_VECTORIZED_KERNEL_SOURCE).ok());

            if self.kernel_handle.is_some() {
                Ok(())
            } else {
                Err(scirs2_core::gpu::GpuError::KernelCompilationError(
                    "Failed to compile OpenCL kernels".to_string(),
                ))
            }
        } else {
            Err(scirs2_core::gpu::GpuError::BackendNotAvailable(
                "OpenCL".to_string(),
            ))
        }
    }

    /// Execute OpenCL sparse matrix-vector multiplication
    #[cfg(feature = "gpu")]
    pub fn execute_spmv<T>(
        &self,
        matrix: &CsrArray<T>,
        vector: &ArrayView1<T>,
        _device: &super::GpuDevice,
    ) -> SparseResult<Array1<T>>
    where
        T: Float + Debug + Copy + scirs2_core::gpu::GpuDataType,
    {
        let (rows, cols) = matrix.shape();
        if cols != vector.len() {
            return Err(SparseError::DimensionMismatch {
                expected: cols,
                found: vector.len(),
            });
        }

        if let Some(ref context) = self.context {
            if let Some(ref kernel) = self.kernel_handle {
                // Upload data to GPU
                let indptr_buffer =
                    context.create_buffer_from_slice(matrix.get_indptr().as_slice().unwrap());
                let indices_buffer =
                    context.create_buffer_from_slice(matrix.get_indices().as_slice().unwrap());
                let data_buffer =
                    context.create_buffer_from_slice(matrix.get_data().as_slice().unwrap());
                let vector_buffer = context.create_buffer_from_slice(vector.as_slice().unwrap());
                let result_buffer = context.create_buffer::<T>(rows);

                // Set kernel parameters
                kernel.set_buffer("indptr", &indptr_buffer);
                kernel.set_buffer("indices", &indices_buffer);
                kernel.set_buffer("data", &data_buffer);
                kernel.set_buffer("vector", &vector_buffer);
                kernel.set_buffer("result", &result_buffer);
                kernel.set_u32("rows", rows as u32);

                // Configure work group size based on platform
                let work_group_size = self.platform_info.max_work_group_size.min(256);
                let grid_size = ((rows + work_group_size - 1) / work_group_size, 1, 1);
                let block_size = (work_group_size, 1, 1);

                // Execute kernel
                let args = vec![scirs2_core::gpu::DynamicKernelArg::U32(rows as u32)];

                context
                    .launch_kernel("spmv_csr_kernel", grid_size, block_size, &args)
                    .map_err(|e| {
                        SparseError::ComputationError(format!(
                            "OpenCL kernel execution failed: {:?}",
                            e
                        ))
                    })?;

                // Read result back
                let mut result_vec = vec![T::zero(); rows];
                result_buffer.copy_to_host(&mut result_vec).map_err(|e| {
                    SparseError::ComputationError(format!(
                        "Failed to copy result from GPU: {:?}",
                        e
                    ))
                })?;
                Ok(Array1::from_vec(result_vec))
            } else {
                Err(SparseError::ComputationError(
                    "OpenCL kernel not compiled".to_string(),
                ))
            }
        } else {
            // Fallback to CPU implementation
            matrix.dot_vector(vector)
        }
    }

    /// Execute optimized OpenCL sparse matrix-vector multiplication
    #[cfg(feature = "gpu")]
    pub fn execute_optimized_spmv<T>(
        &self,
        matrix: &CsrArray<T>,
        vector: &ArrayView1<T>,
        device: &super::GpuDevice,
        optimization_level: OpenCLOptimizationLevel,
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
            OpenCLOptimizationLevel::Basic => &self.kernel_handle,
            OpenCLOptimizationLevel::Workgroup => &self.workgroup_kernel,
            OpenCLOptimizationLevel::Vectorized => &self.vectorized_kernel,
        };

        if let Some(ref k) = kernel {
            self.execute_kernel_with_optimization(matrix, vector, device, k, optimization_level)
        } else {
            Err(SparseError::ComputationError(
                "OpenCL kernel not available for requested optimization level".to_string(),
            ))
        }
    }

    #[cfg(feature = "gpu")]
    fn execute_kernel_with_optimization<T>(
        &self,
        matrix: &CsrArray<T>,
        vector: &ArrayView1<T>,
        _device: &super::GpuDevice,
        _kernel: &super::GpuKernelHandle,
        optimization_level: OpenCLOptimizationLevel,
    ) -> SparseResult<Array1<T>>
    where
        T: Float + Debug + Copy + super::GpuDataType,
    {
        let (rows, _) = matrix.shape();

        if let Some(ref context) = self.context {
            // Upload data to GPU using context
            let indptr_gpu =
                context.create_buffer_from_slice(matrix.get_indptr().as_slice().unwrap());
            let indices_gpu =
                context.create_buffer_from_slice(matrix.get_indices().as_slice().unwrap());
            let data_gpu = context.create_buffer_from_slice(matrix.get_data().as_slice().unwrap());
            let vector_gpu = context.create_buffer_from_slice(vector.as_slice().unwrap());
            let result_gpu = context.create_buffer::<T>(rows);

            // Configure work group parameters based on optimization level
            let (work_group_size, _local_memory_size) = match optimization_level {
                OpenCLOptimizationLevel::Basic => {
                    (self.platform_info.max_work_group_size.min(64), 0)
                }
                OpenCLOptimizationLevel::Workgroup => {
                    let wg_size = self.platform_info.max_work_group_size.min(128);
                    (wg_size, wg_size * std::mem::size_of::<f32>())
                }
                OpenCLOptimizationLevel::Vectorized => {
                    (self.platform_info.max_work_group_size.min(256), 0)
                }
            };

            let global_work_size =
                ((rows + work_group_size - 1) / work_group_size) * work_group_size;

            // Launch kernel using context
            let args = vec![scirs2_core::gpu::DynamicKernelArg::U32(rows as u32)];

            // Use appropriate kernel based on optimization level
            let kernel_name = match optimization_level {
                OpenCLOptimizationLevel::Basic => "spmv_csr_kernel",
                OpenCLOptimizationLevel::Workgroup => "spmv_csr_workgroup_kernel",
                OpenCLOptimizationLevel::Vectorized => "spmv_csr_vectorized_kernel",
            };

            context
                .launch_kernel(
                    kernel_name,
                    (global_work_size, 1, 1),
                    (work_group_size, 1, 1),
                    &args,
                )
                .map_err(|e| {
                    SparseError::ComputationError(format!(
                        "OpenCL kernel execution failed: {:?}",
                        e
                    ))
                })?;

            // Download result
            let mut result_vec = vec![T::zero(); rows];
            result_gpu.copy_to_host(&mut result_vec).map_err(|e| {
                SparseError::ComputationError(format!("Failed to copy result from GPU: {:?}", e))
            })?;
            Ok(Array1::from_vec(result_vec))
        } else {
            // Fallback to CPU implementation
            matrix.dot_vector(vector)
        }
    }

    /// Select optimal kernel based on matrix characteristics
    #[cfg(feature = "gpu")]
    fn select_optimal_kernel<T>(
        &self,
        rows: usize,
        matrix: &CsrArray<T>,
    ) -> SparseResult<super::GpuKernelHandle>
    where
        T: Float + Debug + Copy,
    {
        // Calculate average non-zeros per row
        let avg_nnz_per_row = matrix.get_data().len() as f64 / rows as f64;

        // Select kernel based on sparsity pattern and platform capabilities
        if avg_nnz_per_row < 5.0 && self.platform_info.supports_vectorization {
            // Sparse matrix, use vectorized kernel if available
            if let Some(ref kernel) = self.vectorized_kernel {
                Ok(kernel.clone())
            } else if let Some(ref kernel) = self.kernel_handle {
                Ok(kernel.clone())
            } else {
                Err(SparseError::ComputationError(
                    "No OpenCL kernels available".to_string(),
                ))
            }
        } else if avg_nnz_per_row > 20.0 && self.platform_info.max_work_group_size >= 128 {
            // Dense-ish matrix, use workgroup kernel
            if let Some(ref kernel) = self.workgroup_kernel {
                Ok(kernel.clone())
            } else if let Some(ref kernel) = self.kernel_handle {
                Ok(kernel.clone())
            } else {
                Err(SparseError::ComputationError(
                    "No OpenCL kernels available".to_string(),
                ))
            }
        } else {
            // Default to basic kernel
            if let Some(ref kernel) = self.kernel_handle {
                Ok(kernel.clone())
            } else {
                Err(SparseError::ComputationError(
                    "No OpenCL kernels available".to_string(),
                ))
            }
        }
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

impl Default for OpenCLSpMatVec {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_| Self {
            context: None,
            kernel_handle: None,
            workgroup_kernel: None,
            vectorized_kernel: None,
            platform_info: OpenCLPlatformInfo::default(),
        })
    }
}

/// OpenCL optimization levels for sparse matrix operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenCLOptimizationLevel {
    /// Basic thread-per-row implementation
    Basic,
    /// Workgroup-optimized implementation with local memory
    Workgroup,
    /// Vectorized implementation using float4 operations
    Vectorized,
}

impl Default for OpenCLOptimizationLevel {
    fn default() -> Self {
        Self::Basic
    }
}

/// OpenCL platform information for optimization
#[derive(Debug)]
pub struct OpenCLPlatformInfo {
    pub max_work_group_size: usize,
    pub local_memory_size: usize,
    pub supports_vectorization: bool,
    pub compute_units: usize,
    pub device_type: OpenCLDeviceType,
}

impl OpenCLPlatformInfo {
    /// Detect OpenCL platform capabilities
    pub fn detect() -> Self {
        // In a real implementation, this would query the OpenCL runtime
        // For now, return sensible defaults
        Self {
            max_work_group_size: 256,
            local_memory_size: 32768, // 32KB
            supports_vectorization: true,
            compute_units: 8,
            device_type: OpenCLDeviceType::GPU,
        }
    }
}

impl Default for OpenCLPlatformInfo {
    fn default() -> Self {
        Self::detect()
    }
}

/// OpenCL device types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenCLDeviceType {
    /// CPU device
    CPU,
    /// GPU device
    GPU,
    /// Accelerator device
    Accelerator,
}

/// OpenCL memory management for sparse matrices
pub struct OpenCLMemoryManager {
    platform_info: OpenCLPlatformInfo,
    #[allow(dead_code)]
    allocated_buffers: Vec<String>,
}

impl OpenCLMemoryManager {
    /// Create a new OpenCL memory manager
    pub fn new() -> Self {
        Self {
            platform_info: OpenCLPlatformInfo::detect(),
            allocated_buffers: Vec::new(),
        }
    }

    /// Allocate GPU memory for sparse matrix data with optimal layout
    #[cfg(feature = "gpu")]
    pub fn allocate_sparse_matrix<T>(
        &mut self,
        _matrix: &CsrArray<T>,
        _device: &super::GpuDevice,
    ) -> Result<OpenCLMatrixBuffers<T>, super::GpuError>
    where
        T: super::GpuDataType + Copy + Float + Debug,
    {
        // This functionality should use GpuContext instead of GpuDevice
        // For now, return an error indicating this needs proper implementation
        Err(super::GpuError::BackendNotImplemented(
            super::GpuBackend::OpenCL,
        ))
    }

    /// Get optimal work group size for the current platform
    pub fn optimal_work_group_size(&self, problem_size: usize) -> usize {
        let max_wg_size = self.platform_info.max_work_group_size;

        // For small problems, use smaller work groups
        if problem_size < 1000 {
            max_wg_size.min(64)
        } else if problem_size < 10000 {
            max_wg_size.min(128)
        } else {
            max_wg_size
        }
    }

    /// Check if vectorization is beneficial for the given matrix
    pub fn should_use_vectorization<T>(&self, matrix: &CsrArray<T>) -> bool
    where
        T: Float + Debug + Copy,
    {
        if !self.platform_info.supports_vectorization {
            return false;
        }

        let avg_nnz_per_row = matrix.nnz() as f64 / matrix.shape().0 as f64;

        // Vectorization is beneficial for sparse matrices with moderate sparsity
        (4.0..=32.0).contains(&avg_nnz_per_row)
    }
}

impl Default for OpenCLMemoryManager {
    fn default() -> Self {
        Self::new()
    }
}

/// GPU memory buffers for OpenCL sparse matrix data
#[cfg(feature = "gpu")]
pub struct OpenCLMatrixBuffers<T: super::GpuDataType> {
    pub indptr: super::GpuBuffer<usize>,
    pub indices: super::GpuBuffer<usize>,
    pub data: super::GpuBuffer<T>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    #[test]
    fn test_opencl_spmv_creation() {
        let opencl_spmv = OpenCLSpMatVec::new();
        assert!(opencl_spmv.is_ok());
    }

    #[test]
    fn test_opencl_optimization_levels() {
        let basic = OpenCLOptimizationLevel::Basic;
        let workgroup = OpenCLOptimizationLevel::Workgroup;
        let vectorized = OpenCLOptimizationLevel::Vectorized;

        assert_ne!(basic, workgroup);
        assert_ne!(workgroup, vectorized);
        assert_eq!(
            OpenCLOptimizationLevel::default(),
            OpenCLOptimizationLevel::Basic
        );
    }

    #[test]
    fn test_opencl_platform_info() {
        let info = OpenCLPlatformInfo::detect();
        assert!(info.max_work_group_size > 0);
        assert!(info.local_memory_size > 0);
        assert!(info.compute_units > 0);
    }

    #[test]
    fn test_opencl_device_types() {
        let device_types = [
            OpenCLDeviceType::CPU,
            OpenCLDeviceType::GPU,
            OpenCLDeviceType::Accelerator,
        ];

        for device_type in &device_types {
            match device_type {
                OpenCLDeviceType::CPU => (),
                OpenCLDeviceType::GPU => (),
                OpenCLDeviceType::Accelerator => (),
            }
        }
    }

    #[test]
    fn test_opencl_memory_manager() {
        let manager = OpenCLMemoryManager::new();
        assert_eq!(manager.allocated_buffers.len(), 0);
        assert!(manager.platform_info.max_work_group_size > 0);

        // Test work group size selection
        let wg_size_small = manager.optimal_work_group_size(500);
        let wg_size_large = manager.optimal_work_group_size(50000);
        assert!(wg_size_small <= wg_size_large);
    }

    #[test]
    #[allow(clippy::const_is_empty)]
    fn test_kernel_sources() {
        assert!(!OPENCL_SPMV_KERNEL_SOURCE.is_empty());
        assert!(!OPENCL_VECTORIZED_KERNEL_SOURCE.is_empty());

        // Check that kernels contain expected function names
        assert!(OPENCL_SPMV_KERNEL_SOURCE.contains("spmv_csr_kernel"));
        assert!(OPENCL_SPMV_KERNEL_SOURCE.contains("spmv_csr_workgroup_kernel"));
        assert!(OPENCL_VECTORIZED_KERNEL_SOURCE.contains("spmv_csr_vectorized_kernel"));
    }
}
