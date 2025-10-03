//! Enhanced GPU SpMV Implementation for scirs2-sparse
//!
//! This module provides production-ready GPU-accelerated sparse matrix-vector multiplication
//! with proper error handling, memory management, and multi-backend support.

use crate::error::{SparseError, SparseResult};
use scirs2_core::gpu::{GpuBackend, GpuContext, GpuDataType};
use scirs2_core::numeric::{Float, NumAssign};
use scirs2_core::simd_ops::SimdUnifiedOps;
use std::fmt::Debug;

/// Enhanced GPU-accelerated Sparse Matrix-Vector multiplication implementation
pub struct GpuSpMV {
    #[allow(dead_code)]
    context: GpuContext,
    backend: GpuBackend,
}

impl GpuSpMV {
    /// Create a new GPU SpMV instance with automatic backend detection
    pub fn new() -> SparseResult<Self> {
        // Try to initialize GPU backends in order of preference
        let (context, backend) = Self::initialize_best_backend()?;

        Ok(Self { context, backend })
    }

    /// Create a new GPU SpMV instance with specified backend
    pub fn with_backend(backend: GpuBackend) -> SparseResult<Self> {
        let context = GpuContext::new(backend).map_err(|e| {
            SparseError::ComputationError(format!("Failed to initialize GPU context: {e}"))
        })?;

        Ok(Self { context, backend })
    }

    /// Initialize the best available GPU backend
    fn initialize_best_backend() -> SparseResult<(GpuContext, GpuBackend)> {
        // Try backends in order of performance preference
        let backends_to_try = [
            GpuBackend::Cuda,   // Best performance on NVIDIA GPUs
            GpuBackend::Metal,  // Best performance on Apple Silicon
            GpuBackend::OpenCL, // Good cross-platform compatibility
            GpuBackend::Cpu,    // Fallback option
        ];

        for &backend in &backends_to_try {
            if let Ok(context) = GpuContext::new(backend) {
                return Ok((context, backend));
            }
        }

        Err(SparseError::ComputationError(
            "No GPU backend available".to_string(),
        ))
    }

    /// Execute sparse matrix-vector multiplication: y = A * x
    #[allow(clippy::too_many_arguments)]
    pub fn spmv<T>(
        &self,
        rows: usize,
        cols: usize,
        indptr: &[usize],
        indices: &[usize],
        data: &[T],
        x: &[T],
    ) -> SparseResult<Vec<T>>
    where
        T: Float
            + Debug
            + Copy
            + Default
            + GpuDataType
            + Send
            + Sync
            + 'static
            + NumAssign
            + SimdUnifiedOps
            + std::iter::Sum,
    {
        // Validate input dimensions
        self.validate_spmv_inputs(rows, cols, indptr, indices, data, x)?;

        // Execute GPU-accelerated SpMV based on backend
        match self.backend {
            GpuBackend::Cuda => self.spmv_cuda(rows, indptr, indices, data, x),
            GpuBackend::OpenCL => self.spmv_opencl(rows, indptr, indices, data, x),
            GpuBackend::Metal => self.spmv_metal(rows, indptr, indices, data, x),
            GpuBackend::Cpu => self.spmv_cpu_optimized(rows, indptr, indices, data, x),
            GpuBackend::Rocm | GpuBackend::Wgpu => {
                // For now, use CPU fallback for Rocm and Wgpu until implemented
                self.spmv_cpu_optimized(rows, indptr, indices, data, x)
            }
        }
    }

    /// Validate SpMV input parameters
    fn validate_spmv_inputs<T>(
        &self,
        rows: usize,
        cols: usize,
        indptr: &[usize],
        indices: &[usize],
        data: &[T],
        x: &[T],
    ) -> SparseResult<()>
    where
        T: Float + Debug,
    {
        if indptr.len() != rows + 1 {
            return Err(SparseError::InvalidFormat(format!(
                "indptr length {} does not match rows + 1 = {}",
                indptr.len(),
                rows + 1
            )));
        }

        if indices.len() != data.len() {
            return Err(SparseError::InvalidFormat(format!(
                "indices length {} does not match data length {}",
                indices.len(),
                data.len()
            )));
        }

        if x.len() != cols {
            return Err(SparseError::InvalidFormat(format!(
                "x length {} does not match cols {}",
                x.len(),
                cols
            )));
        }

        // Validate that all indices are within bounds
        for &idx in indices {
            if idx >= cols {
                return Err(SparseError::InvalidFormat(format!(
                    "Column index {idx} exceeds cols {cols}"
                )));
            }
        }

        Ok(())
    }

    /// CUDA-accelerated SpMV implementation
    fn spmv_cuda<T>(
        &self,
        rows: usize,
        indptr: &[usize],
        indices: &[usize],
        data: &[T],
        x: &[T],
    ) -> SparseResult<Vec<T>>
    where
        T: Float
            + Debug
            + Copy
            + Default
            + GpuDataType
            + Send
            + Sync
            + 'static
            + NumAssign
            + SimdUnifiedOps
            + std::iter::Sum,
    {
        #[cfg(feature = "gpu")]
        {
            use crate::gpu_ops::{GpuBufferExt, SpMVKernel};

            // Create GPU buffers
            let indptr_buffer = self.context.create_buffer_from_slice(indptr);
            let indices_buffer = self.context.create_buffer_from_slice(indices);
            let data_buffer = self.context.create_buffer_from_slice(data);
            let x_buffer = self.context.create_buffer_from_slice(x);
            let mut y_buffer = self.context.create_buffer::<T>(rows);

            // Use unified GPU interface
            use crate::csr_array::CsrArray;
            use crate::gpu::GpuSpMatVec;

            // Create CSR matrix from components for unified interface
            let csr_matrix = CsrArray::new(
                data.to_vec().into(),
                indices.to_vec().into(),
                indptr.to_vec().into(),
                (rows, x.len()),
            )?;

            let gpu_handler = GpuSpMatVec::with_backend(self.backend)?;
            let result = gpu_handler.spmv(
                &csr_matrix,
                &scirs2_core::ndarray::ArrayView1::from(x),
                None,
            )?;
            Ok(result.to_vec())
        }

        #[cfg(not(feature = "gpu"))]
        {
            // Fall back to optimized CPU implementation when GPU feature is not enabled
            self.spmv_cpu_optimized(rows, indptr, indices, data, x)
        }
    }

    /// OpenCL-accelerated SpMV implementation
    fn spmv_opencl<T>(
        &self,
        rows: usize,
        indptr: &[usize],
        indices: &[usize],
        data: &[T],
        x: &[T],
    ) -> SparseResult<Vec<T>>
    where
        T: Float
            + Debug
            + Copy
            + Default
            + GpuDataType
            + Send
            + Sync
            + 'static
            + NumAssign
            + SimdUnifiedOps
            + std::iter::Sum,
    {
        #[cfg(feature = "gpu")]
        {
            use crate::gpu_ops::{GpuBufferExt, SpMVKernel};

            // Use unified GPU interface instead of low-level OpenCL
            use crate::csr_array::CsrArray;
            use crate::gpu::GpuSpMatVec;

            // Create CSR matrix from components for unified interface
            let csr_matrix = CsrArray::new(
                data.to_vec().into(),
                indices.to_vec().into(),
                indptr.to_vec().into(),
                (rows, x.len()),
            )?;

            let gpu_handler = GpuSpMatVec::with_backend(self.backend)?;
            let result = gpu_handler.spmv(
                &csr_matrix,
                &scirs2_core::ndarray::ArrayView1::from(x),
                None,
            )?;
            Ok(result.to_vec())
        }

        #[cfg(not(feature = "gpu"))]
        {
            // Fall back to optimized CPU implementation when GPU feature is not enabled
            self.spmv_cpu_optimized(rows, indptr, indices, data, x)
        }
    }

    /// Metal-accelerated SpMV implementation  
    fn spmv_metal<T>(
        &self,
        rows: usize,
        indptr: &[usize],
        indices: &[usize],
        data: &[T],
        x: &[T],
    ) -> SparseResult<Vec<T>>
    where
        T: Float
            + Debug
            + Copy
            + Default
            + GpuDataType
            + Send
            + Sync
            + 'static
            + NumAssign
            + SimdUnifiedOps
            + std::iter::Sum,
    {
        #[cfg(feature = "gpu")]
        {
            use crate::gpu_ops::{GpuBufferExt, SpMVKernel};

            // Create GPU buffers for Metal
            let indptr_buffer = self.context.create_buffer_from_slice(indptr);
            let indices_buffer = self.context.create_buffer_from_slice(indices);
            let data_buffer = self.context.create_buffer_from_slice(data);
            let x_buffer = self.context.create_buffer_from_slice(x);
            let mut y_buffer = self.context.create_buffer::<T>(rows);

            // Use unified GPU interface instead of low-level kernel
            use crate::csr_array::CsrArray;
            use crate::gpu::GpuSpMatVec;

            // Create CSR matrix from components for unified interface
            let csr_matrix = CsrArray::new(
                data.to_vec().into(),
                indices.to_vec().into(),
                indptr.to_vec().into(),
                (rows, x.len()),
            )?;

            let gpu_handler = GpuSpMatVec::with_backend(self.backend)?;
            let result = gpu_handler.spmv(
                &csr_matrix,
                &scirs2_core::ndarray::ArrayView1::from(x),
                None,
            )?;
            Ok(result.to_vec())
        }

        #[cfg(not(feature = "gpu"))]
        {
            // Fall back to optimized CPU implementation when GPU feature is not enabled
            self.spmv_cpu_optimized(rows, indptr, indices, data, x)
        }
    }

    /// Optimized CPU fallback implementation
    fn spmv_cpu_optimized<T>(
        &self,
        rows: usize,
        indptr: &[usize],
        indices: &[usize],
        data: &[T],
        x: &[T],
    ) -> SparseResult<Vec<T>>
    where
        T: Float + Debug + Copy + Default + Send + Sync + NumAssign + SimdUnifiedOps,
    {
        let mut y = vec![T::zero(); rows];

        // Use parallel processing for CPU implementation
        #[cfg(feature = "parallel")]
        {
            use crate::parallel_vector_ops::parallel_sparse_matvec_csr;
            parallel_sparse_matvec_csr(&mut y, rows, indptr, indices, data, x, None);
        }

        #[cfg(not(feature = "parallel"))]
        {
            for row in 0..rows {
                let mut sum = T::zero();
                let start = indptr[row];
                let end = indptr[row + 1];

                for idx in start..end {
                    let col = indices[idx];
                    sum = sum + data[idx] * x[col];
                }
                y[row] = sum;
            }
        }

        Ok(y)
    }

    /// Get CUDA kernel source code
    #[allow(dead_code)]
    fn get_cuda_spmv_kernel_source(&self) -> String {
        r#"
        extern "C" _global_ void spmv_csr_kernel(
            int rows,
            const int* _restrict_ indptr,
            const int* _restrict_ indices,
            const float* _restrict_ data,
            const float* _restrict_ x,
            float* _restrict_ y
        ) {
            int row = blockIdx.x * blockDim.x + threadIdx.x;
            if (row >= rows) return;
            
            float sum = 0.0f;
            int start = indptr[row];
            int end = indptr[row + 1];
            
            // Optimized loop with memory coalescing
            for (int j = start; j < end; j++) {
                sum += data[j] * x[indices[j]];
            }
            
            y[row] = sum;
        }
        "#
        .to_string()
    }

    /// Get OpenCL kernel source code
    #[allow(dead_code)]
    fn get_opencl_spmv_kernel_source(&self) -> String {
        r#"
        _kernel void spmv_csr_kernel(
            const int rowsglobal const int* restrict indptr_global const int* restrict indices_global const float* restrict data_global const float* restrict x_global float* restrict y
        ) {
            int row = get_global_id(0);
            if (row >= rows) return;
            
            float sum = 0.0f;
            int start = indptr[row];
            int end = indptr[row + 1];
            
            // Vectorized loop with memory coalescing
            for (int j = start; j < end; j++) {
                sum += data[j] * x[indices[j]];
            }
            
            y[row] = sum;
        }
        "#
        .to_string()
    }

    /// Get Metal kernel source code
    #[allow(dead_code)]
    fn get_metal_spmv_kernel_source(&self) -> String {
        r#"
        #include <metal_stdlib>
        using namespace metal;
        
        kernel void spmv_csr_kernel(
            constant int& rows [[buffer(0)]],
            constant int* indptr [[buffer(1)]],
            constant int* indices [[buffer(2)]],
            constant float* data [[buffer(3)]],
            constant float* x [[buffer(4)]],
            device float* y [[buffer(5)]],
            uint row [[thread_position_in_grid]]
        ) {
            if (row >= rows) return;
            
            float sum = 0.0f;
            int start = indptr[row];
            int end = indptr[row + 1];
            
            // Vectorized loop optimized for Metal
            for (int j = start; j < end; j++) {
                sum += data[j] * x[indices[j]];
            }
            
            y[row] = sum;
        }
        "#
        .to_string()
    }

    /// Get information about the current GPU backend
    pub fn backend_info(&self) -> (GpuBackend, String) {
        let backend_name = match self.backend {
            GpuBackend::Cuda => "NVIDIA CUDA",
            GpuBackend::OpenCL => "OpenCL",
            GpuBackend::Metal => "Apple Metal",
            GpuBackend::Cpu => "CPU Fallback",
            GpuBackend::Rocm => "AMD ROCm",
            GpuBackend::Wgpu => "WebGPU",
        };

        (self.backend, backend_name.to_string())
    }
}

impl Default for GpuSpMV {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_| {
            // If GPU initialization fails, create CPU-only version
            Self {
                context: GpuContext::new(GpuBackend::Cpu).unwrap(),
                backend: GpuBackend::Cpu,
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_spmv_creation() {
        let gpu_spmv = GpuSpMV::new();
        assert!(
            gpu_spmv.is_ok(),
            "Should be able to create GPU SpMV instance"
        );
    }

    #[test]
    fn test_cpu_fallback_spmv() {
        let gpu_spmv = GpuSpMV::with_backend(GpuBackend::Cpu).unwrap();

        // Simple test matrix: [[1, 2], [0, 3]]
        let indptr = vec![0, 2, 3];
        let indices = vec![0, 1, 1];
        let data = vec![1.0, 2.0, 3.0];
        let x = vec![1.0, 1.0];

        let result = gpu_spmv.spmv(2, 2, &indptr, &indices, &data, &x).unwrap();
        assert_eq!(result, vec![3.0, 3.0]); // [1*1 + 2*1, 3*1] = [3, 3]
    }

    #[test]
    fn test_backend_info() {
        let gpu_spmv = GpuSpMV::default();
        let (_backend, name) = gpu_spmv.backend_info();
        assert!(!name.is_empty(), "Backend name should not be empty");
    }
}
