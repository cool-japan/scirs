//! Metal Performance Shaders (MPS) integration for accelerated operations
//!
//! This module provides access to Apple's optimized GPU primitives through
//! Metal Performance Shaders, offering high-performance implementations of
//! common operations like matrix multiplication, convolution, and more.

#![cfg(all(feature = "metal", target_os = "macos"))]

use crate::gpu::{GpuBufferImpl, GpuError};
use metal::{Buffer, CommandQueue, Device};
use objc2_metal_performance_shaders::{
    MPSCNNConvolution, MPSCNNPoolingAverage, MPSCNNPoolingMax,
    MPSImageGaussianBlur, MPSMatrix, MPSMatrixDescriptor, MPSMatrixFindTopK,
    MPSMatrixMultiplication, MPSMatrixSoftMax, MPSMatrixSum,
};
use std::sync::Arc;

/// Metal Performance Shaders context
pub struct MPSContext {
    device: Device,
    command_queue: CommandQueue,
}

impl MPSContext {
    /// Create a new MPS context
    pub fn new(device: Device, command_queue: CommandQueue) -> Self {
        Self {
            device,
            command_queue,
        }
    }

    /// Create a matrix multiplication operation
    pub fn create_matrix_multiplication(
        &self,
        transpose_left: bool,
        transpose_right: bool,
        result_rows: usize,
        result_cols: usize,
        inner_dimension: usize,
        alpha: f32,
        beta: f32,
    ) -> Result<MPSMatrixMultiplication, GpuError> {
        let matmul = MPSMatrixMultiplication::init_with_device(
            &self.device,
            transpose_left,
            transpose_right,
            result_rows,
            result_cols,
            inner_dimension,
            alpha as f64,
            beta as f64,
        )
        .ok_or_else(|| GpuError::Other("Failed to create MPSMatrixMultiplication".to_string()))?;

        Ok(matmul)
    }

    /// Create a matrix descriptor
    pub fn create_matrix_descriptor(
        rows: usize,
        columns: usize,
        row_bytes: usize,
        data_type: MPSDataType,
    ) -> MPSMatrixDescriptor {
        MPSMatrixDescriptor::matrix_descriptor_with_dimensions(rows, columns, row_bytes, data_type)
    }

    /// Create an MPS matrix from a Metal buffer
    pub fn create_matrix(
        &self,
        buffer: &Buffer,
        descriptor: &MPSMatrixDescriptor,
    ) -> Result<MPSMatrix, GpuError> {
        let matrix = MPSMatrix::init_with_buffer(buffer, descriptor)
            .ok_or_else(|| GpuError::Other("Failed to create MPSMatrix".to_string()))?;
        Ok(matrix)
    }

    /// Perform matrix multiplication using MPS
    pub fn matrix_multiply(
        &self,
        left: &MPSMatrix,
        right: &MPSMatrix,
        result: &MPSMatrix,
        matmul: &MPSMatrixMultiplication,
    ) -> Result<(), GpuError> {
        let command_buffer = self.command_queue.new_command_buffer();

        matmul.encode_to_command_buffer(&command_buffer, left, right, result);

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(())
    }

    /// Create a softmax operation
    pub fn create_softmax(&self, axis: i32) -> Result<MPSMatrixSoftMax, GpuError> {
        let softmax = MPSMatrixSoftMax::init_with_device(&self.device)
            .ok_or_else(|| GpuError::Other("Failed to create MPSMatrixSoftMax".to_string()))?;
        softmax.set_source_columns(axis);
        Ok(softmax)
    }

    /// Create a sum reduction operation
    pub fn create_sum(&self) -> Result<MPSMatrixSum, GpuError> {
        let sum = MPSMatrixSum::init_row_with_device(&self.device)
            .ok_or_else(|| GpuError::Other("Failed to create MPSMatrixSum".to_string()))?;
        Ok(sum)
    }

    // Note: MPSMatrixMeanAndVariance is not available in current objc2 bindings
    // This functionality would need to be implemented using other MPS operations

    /// Create a top-k operation
    pub fn create_find_top_k(&self, k: usize) -> Result<MPSMatrixFindTopK, GpuError> {
        let topk = MPSMatrixFindTopK::init_with_device(&self.device, k)
            .ok_or_else(|| GpuError::Other("Failed to create MPSMatrixFindTopK".to_string()))?;
        Ok(topk)
    }

    /// Create a 2D convolution operation
    pub fn create_convolution_2d(
        &self,
        _kernel_width: usize,
        _kernel_height: usize,
        _input_channels: usize,
        _output_channels: usize,
    ) -> Result<MPSCNNConvolution, GpuError> {
        // This is a placeholder - actual implementation would require
        // proper initialization with weights and biases
        Err(GpuError::Other(
            "CNN convolution requires weight initialization".to_string(),
        ))
    }

    /// Create a max pooling operation
    pub fn create_max_pool_2d(
        &self,
        kernel_width: usize,
        kernel_height: usize,
        stride_x: usize,
        stride_y: usize,
    ) -> Result<MPSCNNPoolingMax, GpuError> {
        let pool = MPSCNNPoolingMax::init_with_device(
            &self.device,
            kernel_width,
            kernel_height,
            stride_x,
            stride_y,
        )
        .ok_or_else(|| GpuError::Other("Failed to create MPSCNNPoolingMax".to_string()))?;
        Ok(pool)
    }

    /// Create an average pooling operation
    pub fn create_avg_pool_2d(
        &self,
        kernel_width: usize,
        kernel_height: usize,
        stride_x: usize,
        stride_y: usize,
    ) -> Result<MPSCNNPoolingAverage, GpuError> {
        let pool = MPSCNNPoolingAverage::init_with_device(
            &self.device,
            kernel_width,
            kernel_height,
            stride_x,
            stride_y,
        )
        .ok_or_else(|| GpuError::Other("Failed to create MPSCNNPoolingAverage".to_string()))?;
        Ok(pool)
    }

    /// Create a Gaussian blur operation
    pub fn create_gaussian_blur(&self, sigma: f32) -> Result<MPSImageGaussianBlur, GpuError> {
        let blur = MPSImageGaussianBlur::init_with_device(&self.device, sigma)
            .ok_or_else(|| GpuError::Other("Failed to create MPSImageGaussianBlur".to_string()))?;
        Ok(blur)
    }
}

/// MPS data type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MPSDataType {
    Float16,
    Float32,
    Int8,
    UInt8,
    Int16,
    UInt16,
    Int32,
    UInt32,
}

impl MPSDataType {
    /// Convert to Metal Performance Shaders data type
    pub fn to_mps_data_type(&self) -> objc2_metal_performance_shaders::MPSDataType {
        match self {
            MPSDataType::Float16 => objc2_metal_performance_shaders::MPSDataType::Float16,
            MPSDataType::Float32 => objc2_metal_performance_shaders::MPSDataType::Float32,
            MPSDataType::Int8 => objc2_metal_performance_shaders::MPSDataType::Int8,
            MPSDataType::UInt8 => objc2_metal_performance_shaders::MPSDataType::UInt8,
            MPSDataType::Int16 => objc2_metal_performance_shaders::MPSDataType::Int16,
            MPSDataType::UInt16 => objc2_metal_performance_shaders::MPSDataType::UInt16,
            MPSDataType::Int32 => objc2_metal_performance_shaders::MPSDataType::Int32,
            MPSDataType::UInt32 => objc2_metal_performance_shaders::MPSDataType::UInt32,
        }
    }
}

/// High-level wrapper for common MPS operations
pub struct MPSOperations {
    context: MPSContext,
}

impl MPSOperations {
    /// Create new MPS operations wrapper
    pub fn new(device: Device, command_queue: CommandQueue) -> Self {
        Self {
            context: MPSContext::new(device, command_queue),
        }
    }

    /// Perform optimized matrix multiplication
    pub fn matmul(
        &self,
        _a: &Arc<dyn GpuBufferImpl>,
        _b: &Arc<dyn GpuBufferImpl>,
        _c: &Arc<dyn GpuBufferImpl>,
        _m: usize,
        _n: usize,
        _k: usize,
        _alpha: f32,
        _beta: f32,
    ) -> Result<(), GpuError> {
        // This would require proper buffer casting and matrix creation
        // For now, return a placeholder error
        Err(GpuError::Other(
            "MPS matmul not fully implemented".to_string(),
        ))
    }

    /// Perform softmax operation
    pub fn softmax(
        &self,
        _input: &Arc<dyn GpuBufferImpl>,
        _output: &Arc<dyn GpuBufferImpl>,
        _axis: i32,
    ) -> Result<(), GpuError> {
        // Placeholder implementation
        Err(GpuError::Other(
            "MPS softmax not fully implemented".to_string(),
        ))
    }

    /// Calculate mean and variance
    pub fn mean_and_variance(
        &self,
        _input: &Arc<dyn GpuBufferImpl>,
        _mean: &Arc<dyn GpuBufferImpl>,
        _variance: &Arc<dyn GpuBufferImpl>,
    ) -> Result<(), GpuError> {
        // Placeholder implementation
        Err(GpuError::Other(
            "MPS mean_and_variance not fully implemented".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mps_data_type_conversion() {
        assert_eq!(
            MPSDataType::Float32.to_mps_data_type(),
            objc2_metal_performance_shaders::MPSDataType::Float32
        );
        assert_eq!(
            MPSDataType::Float16.to_mps_data_type(),
            objc2_metal_performance_shaders::MPSDataType::Float16
        );
    }
}
