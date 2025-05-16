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

//! Automatic device placement for array operations.
//!
//! This module provides functionality for automatically determining the best
//! device (CPU, GPU, distributed) for array operations based on array size,
//! available hardware, and operation complexity.

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::RwLock;

use ndarray::{Array, Dim, Dimension, SliceArg, SliceInfo, SliceInfoElem};
use num_traits;

use crate::array_protocol::{
    ArrayFunction, ArrayProtocol, DistributedBackend, DistributedConfig, DistributedNdarray,
    DistributionStrategy, GPUBackend, GPUConfig, GPUNdarray, NdarrayWrapper, NotImplemented,
};
use crate::error::CoreResult;

/// Configuration for automatic device placement.
#[derive(Debug, Clone)]
pub struct AutoDeviceConfig {
    /// Minimum array size (total elements) to consider GPU placement.
    pub gpu_threshold: usize,

    /// Minimum array size to consider distributed placement.
    pub distributed_threshold: usize,

    /// Enable mixed-precision operations.
    pub enable_mixed_precision: bool,

    /// Prefer memory efficiency over speed.
    pub prefer_memory_efficiency: bool,

    /// Automatically transfer arrays between devices when needed.
    pub auto_transfer: bool,

    /// Prefer device data locality (avoid transfers).
    pub prefer_data_locality: bool,

    /// Preferred GPU backend.
    pub preferred_gpu_backend: GPUBackend,

    /// Fallback to CPU if GPU is not available.
    pub fallback_to_cpu: bool,
}

impl Default for AutoDeviceConfig {
    fn default() -> Self {
        Self {
            gpu_threshold: 1_000_000,           // 1M elements
            distributed_threshold: 100_000_000, // 100M elements
            enable_mixed_precision: false,
            prefer_memory_efficiency: false,
            auto_transfer: true,
            prefer_data_locality: true,
            preferred_gpu_backend: GPUBackend::CUDA,
            fallback_to_cpu: true,
        }
    }
}

/// Global auto device configuration.
pub static AUTO_DEVICE_CONFIG: RwLock<AutoDeviceConfig> = RwLock::new(AutoDeviceConfig {
    gpu_threshold: 1_000_000,
    distributed_threshold: 100_000_000,
    enable_mixed_precision: false,
    prefer_memory_efficiency: false,
    auto_transfer: true,
    prefer_data_locality: true,
    preferred_gpu_backend: GPUBackend::CUDA,
    fallback_to_cpu: true,
});

/// Set the global auto device configuration.
pub fn set_auto_device_config(config: AutoDeviceConfig) {
    if let Ok(mut global_config) = AUTO_DEVICE_CONFIG.write() {
        *global_config = config;
    }
}

/// Get the current auto device configuration.
pub fn get_auto_device_config() -> AutoDeviceConfig {
    AUTO_DEVICE_CONFIG
        .read()
        .map(|c| c.clone())
        .unwrap_or_default()
}

/// Determine the best device for an array.
///
/// This function determines the best device (CPU, GPU, distributed) for an array
/// based on its size, the operation being performed, and the current configuration.
pub fn determine_best_device<T, D>(array: &Array<T, D>) -> DeviceType
where
    T: Clone + Send + Sync + 'static + num_traits::Zero + std::ops::Div<f64, Output = T>,
    D: Dimension + ndarray::RemoveAxis,
{
    let config = get_auto_device_config();
    let size = array.len();

    if size >= config.distributed_threshold {
        DeviceType::Distributed
    } else if size >= config.gpu_threshold {
        DeviceType::GPU
    } else {
        DeviceType::CPU
    }
}

/// Determine the best device for an operation with multiple arrays.
///
/// This function determines the best device for an operation based on
/// the arrays involved and the operation being performed.
pub fn determine_best_device_for_operation<T, D>(
    arrays: &[&Array<T, D>],
    operation: &str,
) -> DeviceType
where
    T: Clone + Send + Sync + 'static + num_traits::Zero + std::ops::Div<f64, Output = T>,
    D: Dimension + ndarray::RemoveAxis,
{
    let config = get_auto_device_config();

    // Complex operations (matrix multiplication, SVD, etc.) benefit more from GPU
    let is_complex_operation = matches!(operation, "matmul" | "svd" | "inverse" | "conv2d");

    // Compute total size of all arrays
    let total_size: usize = arrays.iter().map(|arr| arr.len()).sum();

    // Adjust thresholds based on operation complexity
    let gpu_threshold = if is_complex_operation {
        config.gpu_threshold / 10 // Lower threshold for complex operations
    } else {
        config.gpu_threshold
    };

    let distributed_threshold = if is_complex_operation {
        config.distributed_threshold / 2 // Lower threshold for complex operations
    } else {
        config.distributed_threshold
    };

    if total_size >= distributed_threshold {
        DeviceType::Distributed
    } else if total_size >= gpu_threshold {
        DeviceType::GPU
    } else {
        DeviceType::CPU
    }
}

/// Available device types for array operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceType {
    /// CPU-based computation.
    CPU,

    /// GPU-accelerated computation.
    GPU,

    /// Distributed computation across multiple machines/processes.
    Distributed,
}

/// Convert an array to the specified device type.
///
/// This function converts an array to the specified device type, creating
/// the appropriate array wrapper for the target device.
pub fn convert_to_device<T, D>(array: Array<T, D>, device: DeviceType) -> Box<dyn ArrayProtocol>
where
    T: Clone + Send + Sync + 'static + num_traits::Zero + std::ops::Div<f64, Output = T> + Default,
    D: Dimension + ndarray::RemoveAxis + 'static,
    SliceInfo<[SliceInfoElem; 1], Dim<[usize; 1]>, Dim<[usize; 1]>>: SliceArg<D>,
    SliceInfo<[SliceInfoElem; 2], Dim<[usize; 2]>, Dim<[usize; 2]>>: SliceArg<D>,
{
    match device {
        DeviceType::CPU => Box::new(NdarrayWrapper::new(array)),
        DeviceType::GPU => {
            let config = get_auto_device_config();
            let gpu_config = GPUConfig {
                backend: config.preferred_gpu_backend,
                device_id: 0,
                async_ops: true,
                mixed_precision: config.enable_mixed_precision,
                memory_fraction: 0.9,
            };

            Box::new(GPUNdarray::new(array, gpu_config))
        }
        DeviceType::Distributed => {
            let dist_config = DistributedConfig {
                chunks: 2, // Using 2 chunks as a default instead of num_cpus / 2
                balance: true,
                strategy: DistributionStrategy::RowWise,
                backend: DistributedBackend::Threaded,
            };

            Box::new(DistributedNdarray::from_array(array, dist_config))
        }
    }
}

/// A wrapper for arrays that automatically chooses the best device.
///
/// This wrapper automatically places arrays on the most appropriate device
/// based on their size and the operations being performed.
pub struct AutoDevice<T, D>
where
    T: Clone + Send + Sync + 'static + num_traits::Zero + std::ops::Div<f64, Output = T> + Default,
    D: Dimension + ndarray::RemoveAxis,
    SliceInfo<[SliceInfoElem; 1], Dim<[usize; 1]>, Dim<[usize; 1]>>: SliceArg<D>,
    SliceInfo<[SliceInfoElem; 2], Dim<[usize; 2]>, Dim<[usize; 2]>>: SliceArg<D>,
{
    /// The underlying array.
    array: Array<T, D>,

    /// The current device the array is on.
    device: DeviceType,

    /// The array on the current device.
    device_array: Option<Box<dyn ArrayProtocol>>,
}

// Manually implement Debug for AutoDevice since Box<dyn ArrayProtocol> doesn't implement Debug
impl<T, D> std::fmt::Debug for AutoDevice<T, D>
where
    T: Clone
        + Send
        + Sync
        + std::fmt::Debug
        + 'static
        + num_traits::Zero
        + std::ops::Div<f64, Output = T>
        + Default,
    D: Dimension + ndarray::RemoveAxis + std::fmt::Debug + 'static,
    SliceInfo<[SliceInfoElem; 1], Dim<[usize; 1]>, Dim<[usize; 1]>>: SliceArg<D>,
    SliceInfo<[SliceInfoElem; 2], Dim<[usize; 2]>, Dim<[usize; 2]>>: SliceArg<D>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AutoDevice")
            .field("array", &self.array)
            .field("device", &self.device)
            .field("device_array", &self.device_array.is_some())
            .finish()
    }
}

impl<T, D> AutoDevice<T, D>
where
    T: Clone + Send + Sync + 'static + num_traits::Zero + std::ops::Div<f64, Output = T> + Default,
    D: Dimension + ndarray::RemoveAxis + 'static,
    SliceInfo<[SliceInfoElem; 1], Dim<[usize; 1]>, Dim<[usize; 1]>>: SliceArg<D>,
    SliceInfo<[SliceInfoElem; 2], Dim<[usize; 2]>, Dim<[usize; 2]>>: SliceArg<D>,
{
    /// Create a new auto-device array.
    pub fn new(array: Array<T, D>) -> Self {
        let device = determine_best_device(&array);
        let device_array = None; // Lazily initialized

        Self {
            array,
            device,
            device_array,
        }
    }

    /// Get the array on the specified device.
    pub fn on_device(&mut self, device: DeviceType) -> &dyn ArrayProtocol {
        if self.device != device || self.device_array.is_none() {
            // Convert to the requested device
            self.device = device;
            self.device_array = Some(convert_to_device(self.array.clone(), device));
        }

        self.device_array.as_ref().unwrap().as_ref()
    }

    /// Get the current device.
    pub fn device(&self) -> DeviceType {
        self.device
    }

    /// Get the underlying array.
    pub fn array(&self) -> &Array<T, D> {
        &self.array
    }
}

impl<T, D> Clone for AutoDevice<T, D>
where
    T: Clone + Send + Sync + 'static + num_traits::Zero + std::ops::Div<f64, Output = T> + Default,
    D: Dimension + ndarray::RemoveAxis + 'static,
    SliceInfo<[SliceInfoElem; 1], Dim<[usize; 1]>, Dim<[usize; 1]>>: SliceArg<D>,
    SliceInfo<[SliceInfoElem; 2], Dim<[usize; 2]>, Dim<[usize; 2]>>: SliceArg<D>,
{
    fn clone(&self) -> Self {
        Self {
            array: self.array.clone(),
            device: self.device,
            device_array: self.device_array.clone(),
        }
    }
}

impl<T, D> ArrayProtocol for AutoDevice<T, D>
where
    T: Clone + Send + Sync + 'static + num_traits::Zero + std::ops::Div<f64, Output = T> + Default,
    D: Dimension + ndarray::RemoveAxis + 'static,
    SliceInfo<[SliceInfoElem; 1], Dim<[usize; 1]>, Dim<[usize; 1]>>: SliceArg<D>,
    SliceInfo<[SliceInfoElem; 2], Dim<[usize; 2]>, Dim<[usize; 2]>>: SliceArg<D>,
{
    fn array_function(
        &self,
        func: &ArrayFunction,
        types: &[TypeId],
        args: &[Box<dyn Any>],
        kwargs: &HashMap<String, Box<dyn Any>>,
    ) -> Result<Box<dyn Any>, NotImplemented> {
        // If we already have a device array, delegate to it
        if let Some(device_array) = &self.device_array {
            device_array.array_function(func, types, args, kwargs)
        } else {
            // Otherwise, create a temporary array on the appropriate device
            let device = determine_best_device(&self.array);
            let temp_array = convert_to_device(self.array.clone(), device);
            temp_array.array_function(func, types, args, kwargs)
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn shape(&self) -> &[usize] {
        self.array.shape()
    }

    fn dtype(&self) -> TypeId {
        TypeId::of::<T>()
    }

    fn box_clone(&self) -> Box<dyn ArrayProtocol> {
        Box::new(self.clone())
    }
}

/// Execute an operation with automatic device selection.
///
/// This function automatically selects the best device for the operation
/// based on the arrays involved and the operation being performed.
pub fn auto_execute<T, D, F, R>(
    arrays: &mut [&mut AutoDevice<T, D>],
    operation: &str,
    executor: F,
) -> CoreResult<R>
where
    T: Clone + Send + Sync + 'static + num_traits::Zero + std::ops::Div<f64, Output = T> + Default,
    D: Dimension + ndarray::RemoveAxis + 'static,
    SliceInfo<[SliceInfoElem; 1], Dim<[usize; 1]>, Dim<[usize; 1]>>: SliceArg<D>,
    SliceInfo<[SliceInfoElem; 2], Dim<[usize; 2]>, Dim<[usize; 2]>>: SliceArg<D>,
    F: FnOnce(&[&dyn ArrayProtocol]) -> CoreResult<R>,
    R: 'static,
{
    // Determine the best device for this operation
    let best_device = determine_best_device_for_operation(
        &arrays.iter().map(|a| &a.array).collect::<Vec<_>>(),
        operation,
    );

    // Get arrays on the selected device
    let device_arrays: Vec<&dyn ArrayProtocol> = arrays
        .iter_mut()
        .map(|a| a.on_device(best_device))
        .collect();

    // Execute the operation
    executor(&device_arrays)
}

/// Implementation of common array operations with automatic device selection.
pub mod ops {
    use super::*;
    use crate::array_protocol::operations as ap_ops;
    use crate::error::{CoreError, ErrorContext};

    /// Matrix multiplication with automatic device selection.
    pub fn matmul<T, D>(
        a: &mut AutoDevice<T, D>,
        b: &mut AutoDevice<T, D>,
    ) -> CoreResult<Box<dyn ArrayProtocol>>
    where
        T: Clone
            + Send
            + Sync
            + 'static
            + num_traits::Zero
            + std::ops::Div<f64, Output = T>
            + Default,
        D: Dimension + ndarray::RemoveAxis + 'static,
        SliceInfo<[SliceInfoElem; 1], Dim<[usize; 1]>, Dim<[usize; 1]>>: SliceArg<D>,
        SliceInfo<[SliceInfoElem; 2], Dim<[usize; 2]>, Dim<[usize; 2]>>: SliceArg<D>,
    {
        auto_execute(&mut [a, b], "matmul", |arrays| {
            // Convert OperationError to CoreError
            match ap_ops::matmul(arrays[0], arrays[1]) {
                Ok(result) => Ok(result),
                Err(e) => Err(CoreError::NotImplementedError(ErrorContext::new(
                    e.to_string(),
                ))),
            }
        })
    }

    /// Element-wise addition with automatic device selection.
    pub fn add<T, D>(
        a: &mut AutoDevice<T, D>,
        b: &mut AutoDevice<T, D>,
    ) -> CoreResult<Box<dyn ArrayProtocol>>
    where
        T: Clone
            + Send
            + Sync
            + 'static
            + num_traits::Zero
            + std::ops::Div<f64, Output = T>
            + Default,
        D: Dimension + ndarray::RemoveAxis + 'static,
        SliceInfo<[SliceInfoElem; 1], Dim<[usize; 1]>, Dim<[usize; 1]>>: SliceArg<D>,
        SliceInfo<[SliceInfoElem; 2], Dim<[usize; 2]>, Dim<[usize; 2]>>: SliceArg<D>,
    {
        auto_execute(&mut [a, b], "add", |arrays| {
            // Convert OperationError to CoreError
            match ap_ops::add(arrays[0], arrays[1]) {
                Ok(result) => Ok(result),
                Err(e) => Err(CoreError::NotImplementedError(ErrorContext::new(
                    e.to_string(),
                ))),
            }
        })
    }

    /// Element-wise multiplication with automatic device selection.
    pub fn multiply<T, D>(
        a: &mut AutoDevice<T, D>,
        b: &mut AutoDevice<T, D>,
    ) -> CoreResult<Box<dyn ArrayProtocol>>
    where
        T: Clone
            + Send
            + Sync
            + 'static
            + num_traits::Zero
            + std::ops::Div<f64, Output = T>
            + Default,
        D: Dimension + ndarray::RemoveAxis + 'static,
        SliceInfo<[SliceInfoElem; 1], Dim<[usize; 1]>, Dim<[usize; 1]>>: SliceArg<D>,
        SliceInfo<[SliceInfoElem; 2], Dim<[usize; 2]>, Dim<[usize; 2]>>: SliceArg<D>,
    {
        auto_execute(&mut [a, b], "multiply", |arrays| {
            // Convert OperationError to CoreError
            match ap_ops::multiply(arrays[0], arrays[1]) {
                Ok(result) => Ok(result),
                Err(e) => Err(CoreError::NotImplementedError(ErrorContext::new(
                    e.to_string(),
                ))),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr2, Array2};

    #[test]
    fn test_auto_device_selection() {
        // Initialize the array protocol
        crate::array_protocol::init();

        // Create a small array (should be placed on CPU)
        let small_array = Array2::<f64>::ones((10, 10));
        let device = determine_best_device(&small_array);
        assert_eq!(device, DeviceType::CPU);

        // Modify config to place smaller arrays on GPU
        let mut config = get_auto_device_config();
        config.gpu_threshold = 50; // 50 elements
        set_auto_device_config(config);

        // Check device selection with new config
        let device = determine_best_device(&small_array);
        assert_eq!(device, DeviceType::GPU);

        // Reset config
        set_auto_device_config(AutoDeviceConfig::default());
    }

    #[test]
    fn test_auto_device_wrapper() {
        // Initialize the array protocol
        crate::array_protocol::init();

        // Create a small array - using IxDyn to match the trait bounds
        let array_2d = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let array = array_2d.into_dyn();
        let mut auto_array = AutoDevice::new(array.clone());

        // Check initial device (should be CPU for small array)
        assert_eq!(auto_array.device(), DeviceType::CPU);

        // Get array on GPU
        let gpu_array = auto_array.on_device(DeviceType::GPU);
        assert!(gpu_array
            .as_any()
            .downcast_ref::<GPUNdarray<f64, ndarray::IxDyn>>()
            .is_some());

        // Check that device was updated
        assert_eq!(auto_array.device(), DeviceType::GPU);
    }
}
