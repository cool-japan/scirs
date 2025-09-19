//! Metal backend for sparse matrix GPU operations on Apple platforms
//!
//! This module provides Metal-specific implementations for sparse matrix operations
//! optimized for Apple Silicon and Intel Macs with discrete GPUs.

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

/// Metal shader source code for sparse matrix-vector multiplication
pub const METAL_SPMV_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void spmv_csr_kernel(
    device const int* indptr [[buffer(0)]],
    device const int* indices [[buffer(1)]],
    device const float* data [[buffer(2)]],
    device const float* x [[buffer(3)]],
    device float* y [[buffer(4)]],
    constant int& rows [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= uint(rows)) return;
    
    float sum = 0.0f;
    int start = indptr[gid];
    int end = indptr[gid + 1];
    
    for (int j = start; j < end; j++) {
        sum += data[j] * x[indices[j]];
    }
    
    y[gid] = sum;
}

kernel void spmv_csr_simdgroup_kernel(
    device const int* indptr [[buffer(0)]],
    device const int* indices [[buffer(1)]],
    device const float* data [[buffer(2)]],
    device const float* x [[buffer(3)]],
    device float* y [[buffer(4)]],
    constant int& rows [[buffer(5)]],
    uint gid [[thread_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    if (gid >= uint(rows)) return;
    
    int start = indptr[gid];
    int end = indptr[gid + 1];
    float sum = 0.0f;
    
    // Use SIMD group for better performance on Apple Silicon
    for (int j = start + simd_lane_id; j < end; j += 32) {
        sum += data[j] * x[indices[j]];
    }
    
    // SIMD group reduction
    sum = simd_sum(sum);
    
    if (simd_lane_id == 0) {
        y[gid] = sum;
    }
}
"#;

/// Metal shader for Apple Silicon optimized operations
pub const METAL_APPLE_SILICON_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void spmv_csr_apple_silicon_kernel(
    device const int* indptr [[buffer(0)]],
    device const int* indices [[buffer(1)]],
    device const float* data [[buffer(2)]],
    device const float* x [[buffer(3)]],
    device float* y [[buffer(4)]],
    constant int& rows [[buffer(5)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    threadgroup float* shared_data [[threadgroup(0)]]
) {
    if (gid >= uint(rows)) return;
    
    int start = indptr[gid];
    int end = indptr[gid + 1];
    
    // Use unified memory architecture efficiently
    shared_data[lid] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (int j = start; j < end; j++) {
        shared_data[lid] += data[j] * x[indices[j]];
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    y[gid] = shared_data[lid];
}

kernel void spmv_csr_neural_engine_prep_kernel(
    device const int* indptr [[buffer(0)]],
    device const int* indices [[buffer(1)]],
    device const float* data [[buffer(2)]],
    device const float* x [[buffer(3)]],
    device float* y [[buffer(4)]],
    constant int& rows [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    // Prepare data layout for potential Neural Engine acceleration
    if (gid >= uint(rows)) return;
    
    int start = indptr[gid];
    int end = indptr[gid + 1];
    float sum = 0.0f;
    
    // Use float4 for better throughput on Apple Silicon
    int j = start;
    for (; j + 3 < end; j += 4) {
        float4 data_vec = float4(data[j], data[j+1], data[j+2], data[j+3]);
        float4 x_vec = float4(
            x[indices[j]], 
            x[indices[j+1]], 
            x[indices[j+2]], 
            x[indices[j+3]]
        );
        float4 prod = data_vec * x_vec;
        sum += prod.x + prod.y + prod.z + prod.w;
    }
    
    // Handle remaining elements
    for (; j < end; j++) {
        sum += data[j] * x[indices[j]];
    }
    
    y[gid] = sum;
}
"#;

/// Metal sparse matrix operations
pub struct MetalSpMatVec {
    context: Option<scirs2_core::gpu::GpuContext>,
    kernel_handle: Option<scirs2_core::gpu::GpuKernelHandle>,
    simdgroup_kernel: Option<scirs2_core::gpu::GpuKernelHandle>,
    apple_silicon_kernel: Option<scirs2_core::gpu::GpuKernelHandle>,
    neural_engine_kernel: Option<scirs2_core::gpu::GpuKernelHandle>,
    device_info: MetalDeviceInfo,
}

impl MetalSpMatVec {
    /// Create a new Metal sparse matrix-vector multiplication handler
    pub fn new() -> SparseResult<Self> {
        // Try to create Metal context
        #[cfg(feature = "gpu")]
        let context = match scirs2_core::gpu::GpuContext::new(scirs2_core::gpu::GpuBackend::Metal) {
            Ok(ctx) => Some(ctx),
            Err(_) => None, // Metal not available, will use CPU fallback
        };
        #[cfg(not(feature = "gpu"))]
        let context = None;

        let mut handler = Self {
            context,
            kernel_handle: None,
            simdgroup_kernel: None,
            apple_silicon_kernel: None,
            neural_engine_kernel: None,
            device_info: MetalDeviceInfo::detect(),
        };

        // Compile kernels if context is available
        #[cfg(feature = "gpu")]
        if handler.context.is_some() {
            let _ = handler.compile_kernels();
        }

        Ok(handler)
    }

    /// Compile Metal shaders for sparse matrix operations
    #[cfg(feature = "gpu")]
    pub fn compile_kernels(&mut self) -> Result<(), scirs2_core::gpu::GpuError> {
        if let Some(ref context) = self.context {
            // Compile kernels using the context
            self.kernel_handle =
                context.execute(|compiler| compiler.compile(METAL_SPMV_SHADER_SOURCE).ok());

            self.simdgroup_kernel =
                context.execute(|compiler| compiler.compile(METAL_SPMV_SHADER_SOURCE).ok());

            // Apple Silicon specific optimizations
            if self.device_info.is_apple_silicon {
                self.apple_silicon_kernel = context
                    .execute(|compiler| compiler.compile(METAL_APPLE_SILICON_SHADER_SOURCE).ok());

                // Neural Engine kernel would compile the same shader separately
                if self.device_info.has_neural_engine {
                    self.neural_engine_kernel = context.execute(|compiler| {
                        compiler.compile(METAL_APPLE_SILICON_SHADER_SOURCE).ok()
                    });
                }
            }

            if self.kernel_handle.is_some() {
                Ok(())
            } else {
                Err(scirs2_core::gpu::GpuError::KernelCompilationError(
                    "Failed to compile Metal kernels".to_string(),
                ))
            }
        } else {
            Err(scirs2_core::gpu::GpuError::BackendNotAvailable(
                "Metal".to_string(),
            ))
        }
    }

    /// Execute Metal sparse matrix-vector multiplication
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
            // Select the best kernel based on device capabilities
            let kernel = if self.device_info.is_apple_silicon {
                self.apple_silicon_kernel
                    .as_ref()
                    .or(self.simdgroup_kernel.as_ref())
                    .or(self.kernel_handle.as_ref())
            } else {
                self.simdgroup_kernel
                    .as_ref()
                    .or(self.kernel_handle.as_ref())
            };

            if let Some(kernel) = kernel {
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
                kernel.set_buffer("x", &vector_buffer);
                kernel.set_buffer("y", &result_buffer);
                kernel.set_u32("num_rows", rows as u32);

                // Configure threadgroup size for Metal
                let threadgroup_size = self.device_info.max_threadgroup_size.min(256);
                let grid_size = ((rows + threadgroup_size - 1) / threadgroup_size, 1, 1);
                let block_size = (threadgroup_size, 1, 1);

                // Execute kernel
                let args = vec![scirs2_core::gpu::DynamicKernelArg::U32(rows as u32)];

                context
                    .launch_kernel("spmv_csr_kernel", grid_size, block_size, &args)
                    .map_err(|e| {
                        SparseError::ComputationError(format!(
                            "Metal kernel execution failed: {:?}",
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
                    "Metal kernel not compiled".to_string(),
                ))
            }
        } else {
            // Fallback to CPU implementation
            matrix.dot_vector(vector)
        }
    }

    /// Execute optimized Metal sparse matrix-vector multiplication
    #[cfg(feature = "gpu")]
    pub fn execute_optimized_spmv<T>(
        &self,
        matrix: &CsrArray<T>,
        vector: &ArrayView1<T>,
        device: &super::GpuDevice,
        optimization_level: MetalOptimizationLevel,
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

        // Choose kernel based on optimization level and device capabilities
        let kernel = match optimization_level {
            MetalOptimizationLevel::Basic => &self.kernel_handle,
            MetalOptimizationLevel::SimdGroup => &self.simdgroup_kernel,
            MetalOptimizationLevel::AppleSilicon => &self.apple_silicon_kernel,
            MetalOptimizationLevel::NeuralEngine => &self.neural_engine_kernel,
        };

        if let Some(ref k) = kernel {
            self.execute_kernel_with_optimization(matrix, vector, device, k, optimization_level)
        } else {
            // Fallback to basic kernel if specific optimization not available
            if let Some(ref basic_kernel) = self.kernel_handle {
                self.execute_kernel_with_optimization(
                    matrix,
                    vector,
                    device,
                    basic_kernel,
                    MetalOptimizationLevel::Basic,
                )
            } else {
                Err(SparseError::ComputationError(
                    "No Metal kernels available".to_string(),
                ))
            }
        }
    }

    #[cfg(feature = "gpu")]
    fn execute_kernel_with_optimization<T>(
        &self,
        matrix: &CsrArray<T>,
        vector: &ArrayView1<T>,
        _device: &super::GpuDevice,
        _kernel: &super::GpuKernelHandle,
        optimization_level: MetalOptimizationLevel,
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

            // Configure launch parameters based on optimization level
            let (threadgroup_size, _uses_shared_memory) = match optimization_level {
                MetalOptimizationLevel::Basic => {
                    (self.device_info.max_threadgroup_size.min(64), false)
                }
                MetalOptimizationLevel::SimdGroup => {
                    (self.device_info.max_threadgroup_size.min(128), false)
                }
                MetalOptimizationLevel::AppleSilicon => {
                    (self.device_info.max_threadgroup_size.min(256), true)
                }
                MetalOptimizationLevel::NeuralEngine => {
                    // Optimize for Neural Engine pipeline
                    (self.device_info.max_threadgroup_size.min(128), false)
                }
            };

            let grid_size = (rows + threadgroup_size - 1) / threadgroup_size;

            // Launch kernel using context
            let args = vec![scirs2_core::gpu::DynamicKernelArg::U32(rows as u32)];

            // Use appropriate kernel based on optimization level
            let kernel_name = match optimization_level {
                MetalOptimizationLevel::Basic => "spmv_csr_kernel",
                MetalOptimizationLevel::SimdGroup => "spmv_csr_simdgroup_kernel",
                MetalOptimizationLevel::AppleSilicon => "spmv_csr_apple_silicon_kernel",
                MetalOptimizationLevel::NeuralEngine => "spmv_csr_neural_engine_kernel",
            };

            context
                .launch_kernel(
                    kernel_name,
                    (grid_size, 1, 1),
                    (threadgroup_size, 1, 1),
                    &args,
                )
                .map_err(|e| {
                    SparseError::ComputationError(format!("Metal kernel execution failed: {:?}", e))
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

    /// Select optimal kernel based on device and matrix characteristics
    #[cfg(feature = "gpu")]
    fn select_optimal_kernel<T>(
        &self,
        rows: usize,
        matrix: &CsrArray<T>,
    ) -> SparseResult<super::GpuKernelHandle>
    where
        T: Float + Debug + Copy,
    {
        let avg_nnz_per_row = matrix.get_data().len() as f64 / rows as f64;

        // Select kernel based on device capabilities and matrix characteristics
        if self.device_info.is_apple_silicon && avg_nnz_per_row > 16.0 {
            // Use Apple Silicon optimized kernel for dense-ish matrices
            if let Some(ref kernel) = self.apple_silicon_kernel {
                Ok(kernel.clone())
            } else if let Some(ref kernel) = self.simdgroup_kernel {
                Ok(kernel.clone())
            } else if let Some(ref kernel) = self.kernel_handle {
                Ok(kernel.clone())
            } else {
                Err(SparseError::ComputationError(
                    "No Metal kernels available".to_string(),
                ))
            }
        } else if self.device_info.supports_simdgroups && avg_nnz_per_row > 5.0 {
            // Use SIMD group kernel for moderate sparsity
            if let Some(ref kernel) = self.simdgroup_kernel {
                Ok(kernel.clone())
            } else if let Some(ref kernel) = self.kernel_handle {
                Ok(kernel.clone())
            } else {
                Err(SparseError::ComputationError(
                    "No Metal kernels available".to_string(),
                ))
            }
        } else {
            // Use basic kernel for very sparse matrices
            if let Some(ref kernel) = self.kernel_handle {
                Ok(kernel.clone())
            } else {
                Err(SparseError::ComputationError(
                    "No Metal kernels available".to_string(),
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

impl Default for MetalSpMatVec {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_| Self {
            context: None,
            kernel_handle: None,
            simdgroup_kernel: None,
            apple_silicon_kernel: None,
            neural_engine_kernel: None,
            device_info: MetalDeviceInfo::default(),
        })
    }
}

/// Metal optimization levels for sparse matrix operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetalOptimizationLevel {
    /// Basic thread-per-row implementation
    Basic,
    /// SIMD group optimized implementation
    SimdGroup,
    /// Apple Silicon specific optimizations
    AppleSilicon,
    /// Neural Engine preparation (future feature)
    NeuralEngine,
}

impl Default for MetalOptimizationLevel {
    fn default() -> Self {
        Self::Basic
    }
}

/// Metal device information for optimization
#[derive(Debug)]
pub struct MetalDeviceInfo {
    pub max_threadgroup_size: usize,
    pub shared_memory_size: usize,
    pub supports_simdgroups: bool,
    pub is_apple_silicon: bool,
    pub has_neural_engine: bool,
    pub device_name: String,
}

impl MetalDeviceInfo {
    /// Detect Metal device capabilities
    pub fn detect() -> Self {
        // In a real implementation, this would query the Metal runtime
        // For now, return sensible defaults for Apple Silicon
        Self {
            max_threadgroup_size: 1024,
            shared_memory_size: 32768, // 32KB
            supports_simdgroups: true,
            is_apple_silicon: Self::detect_apple_silicon(),
            has_neural_engine: Self::detect_neural_engine(),
            device_name: "Apple GPU".to_string(),
        }
    }

    fn detect_apple_silicon() -> bool {
        // Simple detection based on architecture
        #[cfg(target_arch = "aarch64")]
        {
            #[cfg(target_os = "macos")]
            return true;
        }
        false
    }

    fn detect_neural_engine() -> bool {
        // Neural Engine is available on M1 and later
        Self::detect_apple_silicon()
    }
}

impl Default for MetalDeviceInfo {
    fn default() -> Self {
        Self::detect()
    }
}

/// Metal memory management for sparse matrices
pub struct MetalMemoryManager {
    device_info: MetalDeviceInfo,
    #[allow(dead_code)]
    allocated_buffers: Vec<String>,
}

impl MetalMemoryManager {
    /// Create a new Metal memory manager
    pub fn new() -> Self {
        Self {
            device_info: MetalDeviceInfo::detect(),
            allocated_buffers: Vec::new(),
        }
    }

    /// Allocate GPU memory for sparse matrix data with Metal-specific optimizations
    #[cfg(feature = "gpu")]
    pub fn allocate_sparse_matrix<T>(
        &mut self,
        _matrix: &CsrArray<T>,
        _device: &super::GpuDevice,
    ) -> Result<MetalMatrixBuffers<T>, super::GpuError>
    where
        T: super::GpuDataType + Copy + Float + Debug,
    {
        // This functionality should use GpuContext instead of GpuDevice
        // For now, return an error indicating this needs proper implementation
        Err(super::GpuError::BackendNotImplemented(
            super::GpuBackend::Metal,
        ))
    }

    /// Get optimal threadgroup size for the current device
    pub fn optimal_threadgroup_size(&self, problem_size: usize) -> usize {
        let max_tg_size = self.device_info.max_threadgroup_size;

        if self.device_info.is_apple_silicon {
            // Apple Silicon prefers larger threadgroups
            if problem_size < 1000 {
                max_tg_size.min(128)
            } else {
                max_tg_size.min(256)
            }
        } else {
            // Intel/AMD GPUs prefer smaller threadgroups
            if problem_size < 1000 {
                max_tg_size.min(64)
            } else {
                max_tg_size.min(128)
            }
        }
    }

    /// Check if SIMD group operations are beneficial
    pub fn should_use_simdgroups<T>(&self, matrix: &CsrArray<T>) -> bool
    where
        T: Float + Debug + Copy,
    {
        if !self.device_info.supports_simdgroups {
            return false;
        }

        let avg_nnz_per_row = matrix.nnz() as f64 / matrix.shape().0 as f64;

        // SIMD groups are beneficial for matrices with moderate to high sparsity
        avg_nnz_per_row >= 5.0
    }
}

impl Default for MetalMemoryManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Metal storage modes for optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetalStorageMode {
    /// Shared between CPU and GPU (Apple Silicon)
    Shared,
    /// Managed by Metal (discrete GPUs)
    Managed,
    /// Private to GPU only
    Private,
}

/// GPU memory buffers for Metal sparse matrix data
#[cfg(feature = "gpu")]
pub struct MetalMatrixBuffers<T: super::GpuDataType> {
    pub indptr: super::GpuBuffer<usize>,
    pub indices: super::GpuBuffer<usize>,
    pub data: super::GpuBuffer<T>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metal_spmv_creation() {
        let metal_spmv = MetalSpMatVec::new();
        assert!(metal_spmv.is_ok());
    }

    #[test]
    fn test_metal_optimization_levels() {
        let basic = MetalOptimizationLevel::Basic;
        let simdgroup = MetalOptimizationLevel::SimdGroup;
        let apple_silicon = MetalOptimizationLevel::AppleSilicon;
        let neural_engine = MetalOptimizationLevel::NeuralEngine;

        assert_ne!(basic, simdgroup);
        assert_ne!(simdgroup, apple_silicon);
        assert_ne!(apple_silicon, neural_engine);
        assert_eq!(
            MetalOptimizationLevel::default(),
            MetalOptimizationLevel::Basic
        );
    }

    #[test]
    fn test_metal_device_info() {
        let info = MetalDeviceInfo::detect();
        assert!(info.max_threadgroup_size > 0);
        assert!(info.shared_memory_size > 0);
        assert!(!info.device_name.is_empty());
    }

    #[test]
    fn test_apple_silicon_detection() {
        let info = MetalDeviceInfo::detect();

        // Test that detection logic runs without errors
        #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
        assert!(info.is_apple_silicon);

        #[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
        assert!(!info.is_apple_silicon);
    }

    #[test]
    fn test_metal_memory_manager() {
        let manager = MetalMemoryManager::new();
        assert_eq!(manager.allocated_buffers.len(), 0);
        assert!(manager.device_info.max_threadgroup_size > 0);

        // Test threadgroup size selection
        let tg_size_small = manager.optimal_threadgroup_size(500);
        let tg_size_large = manager.optimal_threadgroup_size(50000);
        assert!(tg_size_small > 0);
        assert!(tg_size_large > 0);
    }

    #[test]
    fn test_metal_storage_modes() {
        let modes = [
            MetalStorageMode::Shared,
            MetalStorageMode::Managed,
            MetalStorageMode::Private,
        ];

        for mode in &modes {
            match mode {
                MetalStorageMode::Shared => (),
                MetalStorageMode::Managed => (),
                MetalStorageMode::Private => (),
            }
        }
    }

    #[test]
    fn test_shader_sources() {
        assert!(!METAL_SPMV_SHADER_SOURCE.is_empty());
        assert!(!METAL_APPLE_SILICON_SHADER_SOURCE.is_empty());

        // Check that shaders contain expected function names
        assert!(METAL_SPMV_SHADER_SOURCE.contains("spmv_csr_kernel"));
        assert!(METAL_SPMV_SHADER_SOURCE.contains("spmv_csr_simdgroup_kernel"));
        assert!(METAL_APPLE_SILICON_SHADER_SOURCE.contains("spmv_csr_apple_silicon_kernel"));
        assert!(METAL_APPLE_SILICON_SHADER_SOURCE.contains("spmv_csr_neural_engine_prep_kernel"));
    }
}
