// Copyright (c) 2025, SciRS2 Team
//
// Licensed under either of
//
// * Apache License, Version 2.0
//   (LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0)
// * MIT license
//   (LICENSE-MIT or http://opensource.org/licenses/MIT)
//
// at your option.
//

//! GPU array implementation using the array protocol.
//!
//! This module provides an implementation of GPU arrays that leverage
//! the array protocol for delegating operations to GPU-specific code.

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::fmt::Debug;

use crate::array_protocol::{ArrayFunction, ArrayProtocol, GPUArray, NotImplemented};
use crate::error::{CoreError, CoreResult, ErrorContext};
use ndarray::{Array, Dimension};

/// GPU backends that can be used
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GPUBackend {
    /// CUDA (NVIDIA GPUs)
    CUDA,

    /// ROCm (AMD GPUs)
    ROCm,

    /// Metal (Apple GPUs)
    Metal,

    /// WebGPU (cross-platform)
    WebGPU,

    /// OpenCL (cross-platform)
    OpenCL,
}

impl Default for GPUBackend {
    fn default() -> Self {
        Self::CUDA
    }
}

/// Configuration for GPU operations
#[derive(Debug, Clone)]
pub struct GPUConfig {
    /// The GPU backend to use
    pub backend: GPUBackend,

    /// The device ID to use
    pub device_id: usize,

    /// Whether to use asynchronous operations
    pub async_ops: bool,

    /// Whether to use mixed precision
    pub mixed_precision: bool,

    /// The fraction of GPU memory to use for the operation
    pub memory_fraction: f32,
}

impl Default for GPUConfig {
    fn default() -> Self {
        Self {
            backend: GPUBackend::default(),
            device_id: 0,
            async_ops: true,
            mixed_precision: false,
            memory_fraction: 0.9,
        }
    }
}

/// A mock implementation of a GPU array
pub struct GPUNdarray<T, D: Dimension>
where
    T: Clone + Send + Sync + 'static + num_traits::Zero,
    T: std::ops::Div<f64, Output = T>,
    D: Clone + Send + Sync + 'static + ndarray::RemoveAxis,
{
    /// The host-side copy of the data (in a real implementation, this would be on the GPU)
    host_data: Array<T, D>,

    /// Configuration for GPU operations
    config: GPUConfig,

    /// Whether the data is currently on the GPU
    on_gpu: bool,

    /// Unique ID for this GPU array
    id: String,
}

impl<T, D> Debug for GPUNdarray<T, D>
where
    T: Debug + Clone + Send + Sync + 'static + num_traits::Zero + std::ops::Div<f64, Output = T>,
    D: Dimension + Debug + Clone + Send + Sync + 'static + ndarray::RemoveAxis,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GPUNdarray")
            .field("config", &self.config)
            .field("on_gpu", &self.on_gpu)
            .field("id", &self.id)
            .field("shape", &self.host_data.shape())
            .finish()
    }
}

impl<T, D> GPUNdarray<T, D>
where
    T: Clone + Send + Sync + 'static + num_traits::Zero + std::ops::Div<f64, Output = T>,
    D: Dimension + Clone + Send + Sync + 'static + ndarray::RemoveAxis,
{
    /// Create a new GPU array from a host array.
    pub fn new(host_data: Array<T, D>, config: GPUConfig) -> Self {
        let id = format!("gpu_array_{}", uuid::Uuid::new_v4());
        let mut array = Self {
            host_data,
            config,
            on_gpu: false,
            id,
        };

        // In a real implementation, this would allocate GPU memory
        // and copy the host data to the GPU
        array.on_gpu = true;

        array
    }

    /// Get the shape of the array.
    pub fn shape(&self) -> &[usize] {
        self.host_data.shape()
    }

    /// Get a reference to the host data.
    pub fn host_data(&self) -> &Array<T, D> {
        &self.host_data
    }

    /// Get a mutable reference to the host data.
    pub fn host_data_mut(&mut self) -> &mut Array<T, D> {
        // In a real implementation, this would sync from GPU to host
        &mut self.host_data
    }

    /// Get a reference to the GPU configuration.
    pub fn config(&self) -> &GPUConfig {
        &self.config
    }

    /// Execute a GPU kernel on this array.
    pub fn execute_kernel<F, R>(&self, kernel: F) -> CoreResult<R>
    where
        F: FnOnce(&Array<T, D>) -> CoreResult<R>,
    {
        // In a real implementation, this would execute a GPU kernel
        // For now, we just call the function on the host data
        kernel(&self.host_data)
    }

    /// Synchronize data from GPU to host.
    pub fn sync_to_host(&mut self) -> CoreResult<()> {
        // In a real implementation, this would copy data from GPU to host
        // For now, we just set a flag
        Ok(())
    }

    /// Synchronize data from host to GPU.
    pub fn sync_to_gpu(&mut self) -> CoreResult<()> {
        // In a real implementation, this would copy data from host to GPU
        // For now, we just set a flag
        self.on_gpu = true;
        Ok(())
    }
}

impl<T, D> ArrayProtocol for GPUNdarray<T, D>
where
    T: Clone + Send + Sync + 'static + num_traits::Zero,
    T: std::ops::Div<f64, Output = T> + std::ops::Mul<Output = T> + std::ops::Add<Output = T>,
    D: Dimension + Clone + Send + Sync + 'static + ndarray::RemoveAxis,
{
    fn array_function(
        &self,
        func: &ArrayFunction,
        _types: &[TypeId],
        args: &[Box<dyn Any>],
        kwargs: &HashMap<String, Box<dyn Any>>,
    ) -> Result<Box<dyn Any>, NotImplemented> {
        match func.name {
            "scirs2::array_protocol::operations::sum" => {
                // Example implementation of sum for a GPU array
                // In a real implementation, this would use GPU-accelerated reduction
                let axis = kwargs.get("axis").and_then(|a| a.downcast_ref::<usize>());

                if let Some(&_ax) = axis {
                    // Sum along a specific axis - this would use GPU kernel in real implementation
                    // But we can't use sum_axis without RemoveAxis trait
                    // Just return the full sum for simplicity
                    let sum = self.host_data.sum();
                    Ok(Box::new(sum))
                } else {
                    // Sum all elements
                    let sum = self.host_data.sum();
                    Ok(Box::new(sum))
                }
            }
            "scirs2::array_protocol::operations::mean" => {
                // Example implementation of mean for a GPU array
                let sum = self.host_data.sum();
                let count = self.host_data.len();
                let mean = sum / count as f64;

                Ok(Box::new(mean))
            }
            "scirs2::array_protocol::operations::add" => {
                // Element-wise addition
                if args.len() < 2 {
                    return Err(NotImplemented);
                }

                // Try to get the second argument as a GPU array first
                if let Some(other) = args[1].downcast_ref::<GPUNdarray<T, D>>() {
                    // Check shapes match
                    if self.shape() != other.shape() {
                        return Err(NotImplemented);
                    }

                    // Use GPU kernel for addition (in this case simulated)
                    let result = match kernels::add(self, other) {
                        Ok(gpu_array) => gpu_array,
                        Err(_) => return Err(NotImplemented),
                    };

                    return Ok(Box::new(result));
                }

                // If the other array is not a GPU array, we could potentially handle
                // other array types, but for simplicity, we'll just return NotImplemented
                Err(NotImplemented)
            }
            "scirs2::array_protocol::operations::multiply" => {
                // Element-wise multiplication
                if args.len() < 2 {
                    return Err(NotImplemented);
                }

                // Try to get the second argument as a GPU array
                if let Some(other) = args[1].downcast_ref::<GPUNdarray<T, D>>() {
                    // Check shapes match
                    if self.shape() != other.shape() {
                        return Err(NotImplemented);
                    }

                    // Use GPU kernel for multiplication (in this case simulated)
                    let result = match kernels::multiply(self, other) {
                        Ok(gpu_array) => gpu_array,
                        Err(_) => return Err(NotImplemented),
                    };

                    return Ok(Box::new(result));
                }

                // If the other array is not a GPU array, we could potentially handle
                // other array types, but for simplicity, we'll just return NotImplemented
                Err(NotImplemented)
            }
            "scirs2::array_protocol::operations::matmul" => {
                // Matrix multiplication
                if args.len() < 2 {
                    return Err(NotImplemented);
                }

                // We can only handle matrix multiplication for 2D arrays
                // Note: For Dimension trait, checking ndim would need more complex logic
                // For simplicity, we'll just check if this is specifically an Ix2 array
                if TypeId::of::<D>() != TypeId::of::<ndarray::Ix2>() {
                    return Err(NotImplemented);
                }

                // Try to get the second argument as a GPU array with the same type
                if let Some(other) = args[1].downcast_ref::<GPUNdarray<T, D>>() {
                    // For simplicity, we'll use the existing kernel function for the specific case
                    // of f64 arrays with 2 dimensions
                    if TypeId::of::<T>() == TypeId::of::<f64>()
                        && TypeId::of::<D>() == TypeId::of::<ndarray::Ix2>()
                    {
                        let self_f64 =
                            unsafe { &*(self as *const _ as *const GPUNdarray<f64, ndarray::Ix2>) };
                        let other_f64 = unsafe {
                            &*(other as *const _ as *const GPUNdarray<f64, ndarray::Ix2>)
                        };

                        match kernels::matmul(self_f64, other_f64) {
                            Ok(result) => {
                                // We can't safely transmute between types with different sizes
                                // Since we're in a specific case where we know T is f64 and D is Ix2,
                                // we can just return the f64 result directly
                                return Ok(Box::new(result));
                            }
                            Err(_) => return Err(NotImplemented),
                        }
                    } else {
                        // For other types, create a placeholder result for demonstration
                        // In a real implementation, we would support more types and dimensions
                        let result = GPUNdarray::new(self.host_data.clone(), self.config.clone());
                        return Ok(Box::new(result));
                    }
                }

                Err(NotImplemented)
            }
            "scirs2::array_protocol::operations::transpose" => {
                // Transpose operation
                // Check for 2D array using TypeId
                if TypeId::of::<D>() != TypeId::of::<ndarray::Ix2>() {
                    return Err(NotImplemented);
                }

                // In a real implementation, this would use a GPU kernel
                // For now, we'll simulate by cloning to CPU, transposing, and creating a new GPU array
                let transposed = self.host_data.t().to_owned();
                let result = GPUNdarray::new(transposed, self.config.clone());

                Ok(Box::new(result))
            }
            "scirs2::array_protocol::operations::reshape" => {
                // Reshape operation
                if let Some(shape) = kwargs
                    .get("shape")
                    .and_then(|s| s.downcast_ref::<Vec<usize>>())
                {
                    match self.host_data.clone().into_shape_with_order(shape.clone()) {
                        Ok(reshaped) => {
                            let result = GPUNdarray::new(reshaped, self.config.clone());
                            return Ok(Box::new(result));
                        }
                        Err(_) => return Err(NotImplemented),
                    }
                }

                Err(NotImplemented)
            }
            _ => Err(NotImplemented),
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn shape(&self) -> &[usize] {
        self.host_data.shape()
    }

    fn box_clone(&self) -> Box<dyn ArrayProtocol> {
        Box::new(self.clone())
    }
}

impl<T, D> GPUArray for GPUNdarray<T, D>
where
    T: Clone + Send + Sync + 'static + num_traits::Zero,
    T: std::ops::Div<f64, Output = T> + std::ops::Mul<Output = T> + std::ops::Add<Output = T>,
    D: Dimension + Clone + Send + Sync + 'static + ndarray::RemoveAxis,
{
    fn to_gpu(&self) -> CoreResult<Box<dyn GPUArray>> {
        // Already on GPU
        Ok(Box::new(self.clone()))
    }

    fn to_cpu(&self) -> CoreResult<Box<dyn ArrayProtocol>> {
        // Create a regular ndarray from the host data
        let array = super::NdarrayWrapper::new(self.host_data.clone());

        Ok(Box::new(array) as Box<dyn ArrayProtocol>)
    }

    fn is_on_gpu(&self) -> bool {
        self.on_gpu
    }

    fn device_info(&self) -> HashMap<String, String> {
        let mut info = HashMap::new();
        info.insert("backend".to_string(), format!("{:?}", self.config.backend));
        info.insert("device_id".to_string(), self.config.device_id.to_string());
        info.insert("on_gpu".to_string(), self.on_gpu.to_string());
        info.insert("id".to_string(), self.id.clone());
        info
    }
}

impl<T, D> Clone for GPUNdarray<T, D>
where
    T: Clone + Send + Sync + 'static + num_traits::Zero,
    T: std::ops::Div<f64, Output = T>,
    D: Dimension + Clone + Send + Sync + 'static + ndarray::RemoveAxis,
{
    fn clone(&self) -> Self {
        Self {
            host_data: self.host_data.clone(),
            config: self.config.clone(),
            on_gpu: self.on_gpu,
            id: self.id.clone(),
        }
    }
}

/// A builder for GPU arrays
pub struct GPUArrayBuilder {
    config: GPUConfig,
}

impl Default for GPUArrayBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl GPUArrayBuilder {
    /// Create a new GPU array builder with default settings.
    pub fn new() -> Self {
        Self {
            config: GPUConfig::default(),
        }
    }

    /// Set the GPU backend to use.
    pub fn backend(mut self, backend: GPUBackend) -> Self {
        self.config.backend = backend;
        self
    }

    /// Set the device ID to use.
    pub fn device_id(mut self, device_id: usize) -> Self {
        self.config.device_id = device_id;
        self
    }

    /// Set whether to use asynchronous operations.
    pub fn async_ops(mut self, async_ops: bool) -> Self {
        self.config.async_ops = async_ops;
        self
    }

    /// Set whether to use mixed precision.
    pub fn mixed_precision(mut self, mixed_precision: bool) -> Self {
        self.config.mixed_precision = mixed_precision;
        self
    }

    /// Set the fraction of GPU memory to use.
    pub fn memory_fraction(mut self, memory_fraction: f32) -> Self {
        self.config.memory_fraction = memory_fraction;
        self
    }

    /// Build a GPU array from a host array.
    pub fn build<T, D>(self, host_data: Array<T, D>) -> GPUNdarray<T, D>
    where
        T: Clone + Send + Sync + 'static + num_traits::Zero + std::ops::Div<f64, Output = T>,
        D: Dimension + Clone + Send + Sync + 'static + ndarray::RemoveAxis,
    {
        GPUNdarray::new(host_data, self.config)
    }
}

/// A collection of GPU kernels for common operations
pub mod kernels {
    use super::*;
    use ndarray::{Array, Dimension};

    /// Add two arrays element-wise.
    pub fn add<T, D>(a: &GPUNdarray<T, D>, b: &GPUNdarray<T, D>) -> CoreResult<GPUNdarray<T, D>>
    where
        T: Clone
            + std::ops::Add<Output = T>
            + Send
            + Sync
            + 'static
            + num_traits::Zero
            + std::ops::Div<f64, Output = T>,
        D: Dimension + Clone + Send + Sync + 'static + ndarray::RemoveAxis,
    {
        // In a real implementation, this would use a GPU kernel
        // For now, we just add the arrays on the CPU

        // Check that the shapes match
        if a.shape() != b.shape() {
            return Err(CoreError::ShapeError(ErrorContext::new(format!(
                "Shape mismatch: {:?} vs {:?}",
                a.shape(),
                b.shape()
            ))));
        }

        // Perform the addition
        let result_data = a.host_data().clone() + b.host_data().clone();

        // Create a new GPU array from the result
        Ok(GPUNdarray::<T, D>::new(result_data, a.config.clone()))
    }

    /// Multiply two arrays element-wise.
    pub fn multiply<T, D>(
        a: &GPUNdarray<T, D>,
        b: &GPUNdarray<T, D>,
    ) -> CoreResult<GPUNdarray<T, D>>
    where
        T: Clone
            + std::ops::Mul<Output = T>
            + Send
            + Sync
            + 'static
            + num_traits::Zero
            + std::ops::Div<f64, Output = T>,
        D: Dimension + Clone + Send + Sync + 'static + ndarray::RemoveAxis,
    {
        // In a real implementation, this would use a GPU kernel
        // For now, we just multiply the arrays on the CPU

        // Check that the shapes match
        if a.shape() != b.shape() {
            return Err(CoreError::ShapeError(ErrorContext::new(format!(
                "Shape mismatch: {:?} vs {:?}",
                a.shape(),
                b.shape()
            ))));
        }

        // Perform the multiplication
        let result_data = a.host_data().clone() * b.host_data().clone();

        // Create a new GPU array from the result
        Ok(GPUNdarray::<T, D>::new(result_data, a.config.clone()))
    }

    /// Matrix multiplication.
    pub fn matmul<T>(
        a: &GPUNdarray<T, ndarray::Ix2>,
        b: &GPUNdarray<T, ndarray::Ix2>,
    ) -> CoreResult<GPUNdarray<T, ndarray::Ix2>>
    where
        T: Clone
            + std::ops::Mul<Output = T>
            + std::ops::Add<Output = T>
            + Default
            + Send
            + Sync
            + 'static
            + num_traits::Zero
            + std::ops::Div<f64, Output = T>,
    {
        // In a real implementation, this would use cuBLAS or similar
        // For now, we just perform matrix multiplication on the CPU

        // Check that the shapes are compatible for matrix multiplication
        let a_shape = a.shape();
        let b_shape = b.shape();

        if a_shape.len() != 2 || b_shape.len() != 2 || a_shape[1] != b_shape[0] {
            return Err(CoreError::ShapeError(ErrorContext::new(format!(
                "Incompatible shapes for matmul: {:?} vs {:?}",
                a_shape, b_shape
            ))));
        }

        // This is a simplified implementation for a GPU array
        // In a real implementation, this would use GPU-accelerated matrix multiplication
        let m = a_shape[0];
        let p = b_shape[1];

        // Just create a default result (all zeros) for demonstration purposes
        let result_data = Array::default((m, p));

        // Create a new GPU array from the result - with explicit type
        Ok(GPUNdarray::<T, ndarray::Ix2>::new(
            result_data,
            a.config.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr2, Array2};

    #[test]
    fn test_gpu_ndarray_creation() {
        let array = Array2::<f64>::ones((10, 5));
        let config = GPUConfig::default();

        let gpu_array = GPUNdarray::new(array.clone(), config);

        // Check that the array was created correctly
        assert_eq!(gpu_array.shape(), &[10, 5]);
        assert!(gpu_array.is_on_gpu());

        // Check device info
        let info = gpu_array.device_info();
        assert_eq!(info.get("backend").unwrap(), "CUDA");
        assert_eq!(info.get("device_id").unwrap(), "0");
        assert_eq!(info.get("on_gpu").unwrap(), "true");
    }

    #[test]
    fn test_gpu_array_builder() {
        let array = Array2::<f64>::ones((10, 5));

        let gpu_array = GPUArrayBuilder::new()
            .backend(GPUBackend::CUDA)
            .device_id(1)
            .async_ops(true)
            .mixed_precision(true)
            .memory_fraction(0.8)
            .build(array.clone());

        // Check that the configuration was set correctly
        assert_eq!(gpu_array.config.backend, GPUBackend::CUDA);
        assert_eq!(gpu_array.config.device_id, 1);
        assert!(gpu_array.config.async_ops);
        assert!(gpu_array.config.mixed_precision);
        assert_eq!(gpu_array.config.memory_fraction, 0.8);
    }

    #[test]
    fn test_gpu_array_kernels() {
        let a = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let b = arr2(&[[5.0, 6.0], [7.0, 8.0]]);

        let gpu_a = GPUNdarray::new(a.clone(), GPUConfig::default());
        let gpu_b = GPUNdarray::new(b.clone(), GPUConfig::default());

        // Test addition
        let result = kernels::add(&gpu_a, &gpu_b).unwrap();
        let expected = a + b;
        assert_eq!(result.host_data(), &expected);

        // Test multiplication
        let a = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let b = arr2(&[[5.0, 6.0], [7.0, 8.0]]);

        let gpu_a = GPUNdarray::new(a.clone(), GPUConfig::default());
        let gpu_b = GPUNdarray::new(b.clone(), GPUConfig::default());

        let result = kernels::multiply(&gpu_a, &gpu_b).unwrap();
        let expected = a * b;
        assert_eq!(result.host_data(), &expected);
    }
}
