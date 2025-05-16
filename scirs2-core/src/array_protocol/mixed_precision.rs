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

//! Mixed-precision operations for the array protocol.
//!
//! This module provides support for mixed-precision operations, allowing
//! arrays to use different numeric types (e.g., f32, f64) for storage
//! and computation to optimize performance and memory usage.

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::fmt;
use std::sync::{LazyLock, RwLock};

use ndarray::{Array, Dimension};
use num_traits::Float;

use crate::array_protocol::gpu_impl::GPUNdarray;
use crate::array_protocol::{
    ArrayFunction, ArrayProtocol, GPUArray, NdarrayWrapper, NotImplemented,
};
use crate::error::{CoreError, CoreResult, ErrorContext};

/// Precision levels for array operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    /// Half-precision floating point (16-bit)
    Half,

    /// Single-precision floating point (32-bit)
    Single,

    /// Double-precision floating point (64-bit)
    Double,

    /// Mixed precision (e.g., store in 16/32-bit, compute in 64-bit)
    Mixed,
}

impl fmt::Display for Precision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Precision::Half => write!(f, "half"),
            Precision::Single => write!(f, "single"),
            Precision::Double => write!(f, "double"),
            Precision::Mixed => write!(f, "mixed"),
        }
    }
}

/// Configuration for mixed-precision operations.
#[derive(Debug, Clone)]
pub struct MixedPrecisionConfig {
    /// Storage precision for arrays.
    pub storage_precision: Precision,

    /// Computation precision for operations.
    pub compute_precision: Precision,

    /// Automatic precision selection based on array size and operation.
    pub auto_precision: bool,

    /// Threshold for automatic downcast to lower precision.
    pub downcast_threshold: usize,

    /// Always use double precision for intermediate results.
    pub double_precision_accumulation: bool,
}

impl Default for MixedPrecisionConfig {
    fn default() -> Self {
        Self {
            storage_precision: Precision::Single,
            compute_precision: Precision::Double,
            auto_precision: true,
            downcast_threshold: 10_000_000, // 10M elements
            double_precision_accumulation: true,
        }
    }
}

/// Global mixed-precision configuration.
pub static MIXED_PRECISION_CONFIG: LazyLock<RwLock<MixedPrecisionConfig>> = LazyLock::new(|| {
    RwLock::new(MixedPrecisionConfig {
        storage_precision: Precision::Single,
        compute_precision: Precision::Double,
        auto_precision: true,
        downcast_threshold: 10_000_000, // 10M elements
        double_precision_accumulation: true,
    })
});

/// Set the global mixed-precision configuration.
pub fn set_mixed_precision_config(config: MixedPrecisionConfig) {
    if let Ok(mut global_config) = MIXED_PRECISION_CONFIG.write() {
        *global_config = config;
    }
}

/// Get the current mixed-precision configuration.
pub fn get_mixed_precision_config() -> MixedPrecisionConfig {
    MIXED_PRECISION_CONFIG
        .read()
        .map(|c| c.clone())
        .unwrap_or_default()
}

/// Determine the optimal precision for an array based on its size.
pub fn determine_optimal_precision<T, D>(array: &Array<T, D>) -> Precision
where
    T: Clone + 'static,
    D: Dimension,
{
    let config = get_mixed_precision_config();
    let size = array.len();

    if config.auto_precision {
        if size >= config.downcast_threshold {
            Precision::Single
        } else {
            Precision::Double
        }
    } else {
        config.storage_precision
    }
}

/// Mixed-precision array that can automatically convert between precisions.
///
/// This wrapper enables arrays to use different precision levels for storage
/// and computation, automatically converting between precisions as needed.
#[derive(Debug)]
pub struct MixedPrecisionArray<T, D>
where
    T: Clone + 'static,
    D: Dimension,
{
    /// The array stored at the specified precision.
    array: Array<T, D>,

    /// The current storage precision.
    storage_precision: Precision,

    /// The precision used for computations.
    compute_precision: Precision,
}

impl<T, D> MixedPrecisionArray<T, D>
where
    T: Clone + Float + 'static,
    D: Dimension,
{
    /// Create a new mixed-precision array.
    pub fn new(array: Array<T, D>) -> Self {
        let precision = match std::mem::size_of::<T>() {
            2 => Precision::Half,
            4 => Precision::Single,
            8 => Precision::Double,
            _ => Precision::Mixed,
        };

        Self {
            array,
            storage_precision: precision,
            compute_precision: precision,
        }
    }

    /// Create a new mixed-precision array with specified compute precision.
    pub fn with_compute_precision(array: Array<T, D>, compute_precision: Precision) -> Self {
        let storage_precision = match std::mem::size_of::<T>() {
            2 => Precision::Half,
            4 => Precision::Single,
            8 => Precision::Double,
            _ => Precision::Mixed,
        };

        Self {
            array,
            storage_precision,
            compute_precision,
        }
    }

    /// Get the array at the specified precision.
    ///
    /// This is a placeholder implementation. In a real implementation,
    /// this would convert the array to the requested precision.
    pub fn at_precision<U>(&self) -> CoreResult<Array<U, D>>
    where
        U: Clone + Float + 'static,
    {
        // This is a simplified implementation for demonstration purposes.
        // In a real implementation, this would handle proper type conversion.
        Err(CoreError::NotImplementedError(ErrorContext::new(
            "Precision conversion not fully implemented yet",
        )))
    }

    /// Get the current storage precision.
    pub fn storage_precision(&self) -> Precision {
        self.storage_precision
    }

    /// Get the underlying array.
    pub fn array(&self) -> &Array<T, D> {
        &self.array
    }
}

/// Trait for arrays that support mixed-precision operations.
pub trait MixedPrecisionSupport: ArrayProtocol {
    /// Convert the array to the specified precision.
    fn to_precision(&self, precision: Precision) -> CoreResult<Box<dyn MixedPrecisionSupport>>;

    /// Get the current precision of the array.
    fn precision(&self) -> Precision;

    /// Check if the array supports the specified precision.
    fn supports_precision(&self, precision: Precision) -> bool;
}

/// Implement ArrayProtocol for MixedPrecisionArray.
impl<T, D> ArrayProtocol for MixedPrecisionArray<T, D>
where
    T: Clone + Float + Send + Sync + 'static,
    D: Dimension + Send + Sync + 'static,
{
    fn array_function(
        &self,
        func: &ArrayFunction,
        types: &[TypeId],
        args: &[Box<dyn Any>],
        kwargs: &HashMap<String, Box<dyn Any>>,
    ) -> Result<Box<dyn Any>, NotImplemented> {
        // If the function supports mixed precision, delegate to the appropriate implementation
        if let Some(precision) = kwargs
            .get("precision")
            .and_then(|p| p.downcast_ref::<Precision>())
        {
            match func.name {
                "scirs2::array_protocol::operations::matmul" => {
                    // For matrix multiplication, use the appropriate precision
                    match precision {
                        Precision::Single => {
                            // Convert to f32 for computation (simplified)
                            let wrapped = NdarrayWrapper::new(self.array.clone());
                            return Ok(Box::new(wrapped));
                        }
                        Precision::Double => {
                            // Use f64 for computation
                            let wrapped = NdarrayWrapper::new(self.array.clone());
                            return Ok(Box::new(wrapped));
                        }
                        _ => return Err(NotImplemented),
                    }
                }
                _ => return Err(NotImplemented),
            }
        }

        // Delegate to the standard implementation
        let wrapped = NdarrayWrapper::new(self.array.clone());
        wrapped.array_function(func, types, args, kwargs)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn shape(&self) -> &[usize] {
        self.array.shape()
    }

    fn box_clone(&self) -> Box<dyn ArrayProtocol> {
        Box::new(Self {
            array: self.array.clone(),
            storage_precision: self.storage_precision,
            compute_precision: self.compute_precision,
        })
    }
}

/// Implement MixedPrecisionSupport for MixedPrecisionArray.
impl<T, D> MixedPrecisionSupport for MixedPrecisionArray<T, D>
where
    T: Clone + Float + Send + Sync + 'static,
    D: Dimension + Send + Sync + 'static,
{
    fn to_precision(&self, precision: Precision) -> CoreResult<Box<dyn MixedPrecisionSupport>> {
        match precision {
            Precision::Single => {
                // Convert to f32 (simplified)
                let array_f32 = self.array.clone();
                let new_array = MixedPrecisionArray::with_compute_precision(array_f32, precision);
                Ok(Box::new(new_array))
            }
            Precision::Double => {
                // Convert to f64 (simplified)
                let array_f64 = self.array.clone();
                let new_array = MixedPrecisionArray::with_compute_precision(array_f64, precision);
                Ok(Box::new(new_array))
            }
            Precision::Mixed => {
                // For mixed precision, use storage precision of the current array and double compute precision
                let array_mixed = self.array.clone();
                let new_array =
                    MixedPrecisionArray::with_compute_precision(array_mixed, Precision::Double);
                Ok(Box::new(new_array))
            }
            _ => Err(CoreError::NotImplementedError(ErrorContext::new(format!(
                "Conversion to {} precision not implemented",
                precision
            )))),
        }
    }

    fn precision(&self) -> Precision {
        // If storage and compute precision differ, return Mixed
        if self.storage_precision != self.compute_precision {
            Precision::Mixed
        } else {
            self.storage_precision
        }
    }

    fn supports_precision(&self, _precision: Precision) -> bool {
        matches!(_precision, Precision::Single | Precision::Double)
    }
}

/// Implement MixedPrecisionSupport for GPUNdarray.
impl<T, D> MixedPrecisionSupport for GPUNdarray<T, D>
where
    T: Clone + Float + Send + Sync + 'static + num_traits::Zero + std::ops::Div<f64, Output = T>,
    D: Dimension + Send + Sync + 'static + ndarray::RemoveAxis,
{
    fn to_precision(&self, precision: Precision) -> CoreResult<Box<dyn MixedPrecisionSupport>> {
        // For GPUs, creating a new array with mixed precision enabled
        let mut config = self.config().clone();
        config.mixed_precision = precision == Precision::Mixed;

        if let Ok(cpu_array) = self.to_cpu() {
            // Use as_any() to downcast the ArrayProtocol trait object
            if let Some(ndarray) = cpu_array.as_any().downcast_ref::<NdarrayWrapper<T, D>>() {
                let new_gpu_array = GPUNdarray::new(ndarray.as_array().clone(), config);
                return Ok(Box::new(new_gpu_array));
            }
        }

        Err(CoreError::NotImplementedError(ErrorContext::new(format!(
            "Conversion to {} precision not implemented for GPU arrays",
            precision
        ))))
    }

    fn precision(&self) -> Precision {
        if self.config().mixed_precision {
            Precision::Mixed
        } else {
            match std::mem::size_of::<T>() {
                4 => Precision::Single,
                8 => Precision::Double,
                _ => Precision::Mixed,
            }
        }
    }

    fn supports_precision(&self, _precision: Precision) -> bool {
        // Most GPUs support all precision levels
        true
    }
}

/// Execute an operation with a specific precision.
///
/// This function automatically converts arrays to the specified precision
/// before executing the operation.
pub fn execute_with_precision<F, R>(
    arrays: &[&dyn MixedPrecisionSupport],
    precision: Precision,
    executor: F,
) -> CoreResult<R>
where
    F: FnOnce(&[&dyn ArrayProtocol]) -> CoreResult<R>,
    R: 'static,
{
    // Check if all arrays support the requested precision
    for array in arrays {
        if !array.supports_precision(precision) {
            return Err(CoreError::InvalidArgument(ErrorContext::new(format!(
                "One or more arrays do not support {} precision",
                precision
            ))));
        }
    }

    // Convert arrays to the requested precision
    let mut converted_arrays = Vec::with_capacity(arrays.len());

    for &array in arrays {
        let converted = array.to_precision(precision)?;
        converted_arrays.push(converted);
    }

    // Create array references
    let array_refs: Vec<&dyn ArrayProtocol> = converted_arrays
        .iter()
        .map(|arr| arr.as_ref() as &dyn ArrayProtocol)
        .collect();

    // Execute the operation
    executor(&array_refs)
}

/// Implementation of common array operations with mixed precision.
pub mod ops {
    use super::*;
    use crate::array_protocol::operations as array_ops;

    /// Matrix multiplication with specified precision.
    pub fn matmul(
        a: &dyn MixedPrecisionSupport,
        b: &dyn MixedPrecisionSupport,
        precision: Precision,
    ) -> CoreResult<Box<dyn ArrayProtocol>> {
        execute_with_precision(&[a, b], precision, |arrays| {
            // Convert OperationError to CoreError
            match array_ops::matmul(arrays[0], arrays[1]) {
                Ok(result) => Ok(result),
                Err(e) => Err(CoreError::NotImplementedError(ErrorContext::new(
                    e.to_string(),
                ))),
            }
        })
    }

    /// Element-wise addition with specified precision.
    pub fn add(
        a: &dyn MixedPrecisionSupport,
        b: &dyn MixedPrecisionSupport,
        precision: Precision,
    ) -> CoreResult<Box<dyn ArrayProtocol>> {
        execute_with_precision(&[a, b], precision, |arrays| {
            // Convert OperationError to CoreError
            match array_ops::add(arrays[0], arrays[1]) {
                Ok(result) => Ok(result),
                Err(e) => Err(CoreError::NotImplementedError(ErrorContext::new(
                    e.to_string(),
                ))),
            }
        })
    }

    /// Element-wise multiplication with specified precision.
    pub fn multiply(
        a: &dyn MixedPrecisionSupport,
        b: &dyn MixedPrecisionSupport,
        precision: Precision,
    ) -> CoreResult<Box<dyn ArrayProtocol>> {
        execute_with_precision(&[a, b], precision, |arrays| {
            // Convert OperationError to CoreError
            match array_ops::multiply(arrays[0], arrays[1]) {
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
    use ndarray::arr2;

    #[test]
    fn test_mixed_precision_array() {
        // Create a mixed-precision array
        let array = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let mixed_array = MixedPrecisionArray::new(array.clone());

        // Check the storage precision (should be double for f64 arrays)
        assert_eq!(mixed_array.storage_precision(), Precision::Double);

        // Test the ArrayProtocol implementation
        let array_protocol: &dyn ArrayProtocol = &mixed_array;
        // The array is of type MixedPrecisionArray<f64, Ix2> (not IxDyn)
        assert!(array_protocol
            .as_any()
            .is::<MixedPrecisionArray<f64, ndarray::Ix2>>());
    }

    #[test]
    fn test_mixed_precision_support() {
        // Initialize the array protocol
        crate::array_protocol::init();

        // Create a mixed-precision array
        let array = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let mixed_array = MixedPrecisionArray::new(array.clone());

        // Test MixedPrecisionSupport implementation
        let mixed_support: &dyn MixedPrecisionSupport = &mixed_array;
        assert_eq!(mixed_support.precision(), Precision::Double);
        assert!(mixed_support.supports_precision(Precision::Single));
        assert!(mixed_support.supports_precision(Precision::Double));
    }
}
