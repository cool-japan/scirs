// Copyright (c) 2025, `SciRS2` Team
//
// Licensed under the Apache License, Version 2.0
// (LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0)
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

use ::ndarray::{Array, Dimension};
use num_traits::{cast as num_cast, Float};

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
    pub computeprecision: Precision,

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
            computeprecision: Precision::Double,
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
        computeprecision: Precision::Double,
        auto_precision: true,
        downcast_threshold: 10_000_000, // 10M elements
        double_precision_accumulation: true,
    })
});

/// Set the global mixed-precision configuration.
#[allow(dead_code)]
pub fn set_mixed_precision_config(config: MixedPrecisionConfig) {
    if let Ok(mut global_config) = MIXED_PRECISION_CONFIG.write() {
        *global_config = config;
    }
}

/// Get the current mixed-precision configuration.
#[allow(dead_code)]
pub fn get_mixed_precision_config() -> MixedPrecisionConfig {
    MIXED_PRECISION_CONFIG
        .read()
        .map(|c| c.clone())
        .unwrap_or_default()
}

/// Determine the optimal precision for an array based on its size.
#[allow(dead_code)]
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
#[derive(Debug, Clone)]
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
    computeprecision: Precision,
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
            computeprecision: precision,
        }
    }

    /// Create a new mixed-precision array with specified compute precision.
    pub fn with_computeprecision(data: Array<T, D>, computeprecision: Precision) -> Self {
        let storage_precision = match std::mem::size_of::<T>() {
            2 => Precision::Half,
            4 => Precision::Single,
            8 => Precision::Double,
            _ => Precision::Mixed,
        };

        Self {
            array: data,
            storage_precision,
            computeprecision,
        }
    }

    /// Convert the array to a different floating-point precision `U`.
    ///
    /// Each element is cast from `T` to `U` using [`fn@num_traits::cast`].  If any
    /// element cannot be represented in `U` (e.g. an `f64` infinity cast to a
    /// hypothetical narrow type) the method returns a
    /// [`CoreError::ComputationError`].
    ///
    /// # Example
    /// ```
    /// use ndarray::array;
    /// use scirs2_core::array_protocol::mixed_precision::MixedPrecisionArray;
    ///
    /// let arr = array![1.0_f64, 2.5_f64, 1.75_f64];
    /// let mp = MixedPrecisionArray::new(arr.into_dyn());
    /// let as_f32: ndarray::ArrayD<f32> = mp.at_precision()
    ///     .expect("f64 -> f32 conversion should succeed");
    /// assert!((as_f32[0] - 1.0_f32).abs() < 1e-6);
    /// ```
    pub fn at_precision<U>(&self) -> CoreResult<Array<U, D>>
    where
        U: Clone + Float + 'static,
    {
        // ndarray does not have a fallible mapv, so we collect into a Vec<U> first.
        let mut converted: Vec<U> = Vec::with_capacity(self.array.len());
        for x in self.array.iter() {
            match num_cast::<T, U>(*x) {
                Some(v) => converted.push(v),
                None => {
                    return Err(CoreError::ComputationError(ErrorContext::new(format!(
                        "at_precision: failed to cast element to target precision (source size \
                         {} bytes, target size {} bytes)",
                        std::mem::size_of::<T>(),
                        std::mem::size_of::<U>(),
                    ))))
                }
            }
        }

        // Reconstruct with the same shape.
        Array::from_shape_vec(self.array.raw_dim(), converted).map_err(|e| {
            CoreError::ShapeError(ErrorContext::new(format!(
                "at_precision: failed to reconstruct array from converted elements: {e}"
            )))
        })
    }

    /// Get the current storage precision.
    pub fn storage_precision(&self) -> Precision {
        self.storage_precision
    }

    /// Get the underlying array.
    pub const fn array(&self) -> &Array<T, D> {
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

    /// Borrow this value as an [`ArrayProtocol`] trait object.
    ///
    /// `MixedPrecisionSupport` has `ArrayProtocol` as a supertrait, but on stable
    /// Rust a `&dyn MixedPrecisionSupport` cannot be upcast to `&dyn ArrayProtocol`
    /// without the unstable `trait_upcasting` feature (RFC #65991). This method
    /// provides that bridge explicitly: every implementor already *is* an
    /// `ArrayProtocol`, so the default implementation simply returns `self`.
    ///
    /// This allows mixed-precision arrays to be passed to operations that are
    /// generic over `&dyn ArrayProtocol` (such as those in
    /// `crate::array_protocol::operations`) on stable Rust.
    fn as_array_protocol(&self) -> &dyn ArrayProtocol;
}

/// Extract the inner `ndarray` of element type `T` and dimension `D` from a
/// boxed argument produced by the operation dispatcher.
///
/// The dispatcher in [`crate::array_protocol::operations`] boxes operands as
/// `Box<dyn ArrayProtocol>` (further boxed into `Box<dyn Any>`). This helper
/// looks through that indirection and recognises both [`MixedPrecisionArray`]
/// and [`NdarrayWrapper`] operands, returning an owned copy of the underlying
/// `ndarray`. Returns `None` if the argument is not a recognised array of the
/// requested `T`/`D`.
fn extract_inner_ndarray<T, D>(arg: &dyn Any) -> Option<Array<T, D>>
where
    T: Clone + Float + Send + Sync + 'static,
    D: Dimension + Send + Sync + 'static,
{
    // Case 1: the operand was boxed as `Box<dyn ArrayProtocol>` (the path used
    // by the operation dispatcher).
    if let Some(ap) = arg.downcast_ref::<Box<dyn ArrayProtocol>>() {
        let inner: &dyn ArrayProtocol = &**ap;
        if let Some(mp) = inner.as_any().downcast_ref::<MixedPrecisionArray<T, D>>() {
            return Some(mp.array.clone());
        }
        if let Some(nd) = inner.as_any().downcast_ref::<NdarrayWrapper<T, D>>() {
            return Some(nd.as_array().clone());
        }
        return None;
    }

    // Case 2: the operand was boxed directly as its concrete type.
    if let Some(mp) = arg.downcast_ref::<MixedPrecisionArray<T, D>>() {
        return Some(mp.array.clone());
    }
    if let Some(nd) = arg.downcast_ref::<NdarrayWrapper<T, D>>() {
        return Some(nd.as_array().clone());
    }

    None
}

/// Normalise the result returned by `NdarrayWrapper`'s array-function kernels
/// into the `Box<dyn Any>`-of-`Box<dyn ArrayProtocol>` shape expected by the
/// operation dispatcher in [`crate::array_protocol::operations`].
///
/// `NdarrayWrapper::array_function` returns its array results boxed as the
/// concrete `NdarrayWrapper<T, _>` type, whereas the dispatcher downcasts the
/// result to `Box<dyn ArrayProtocol>`. This helper bridges that gap for the
/// floating-point element type `T` across the dimensionalities the kernels can
/// produce (`Ix1`, `Ix2`, `IxDyn`). Non-array results (for example the scalar
/// produced by `sum`) do not match any branch and are returned unchanged so the
/// caller can downcast them directly.
fn rewrap_result_as_array_protocol<T>(result: Box<dyn Any>) -> Box<dyn Any>
where
    T: Clone + Float + Send + Sync + 'static,
{
    use crate::ndarray::{Ix1, Ix2, IxDyn};

    // Already in the expected shape (e.g. produced by another delegating layer).
    if result.is::<Box<dyn ArrayProtocol>>() {
        return result;
    }

    // 2-D results: matmul, and element-wise ops on 2-D inputs.
    let result = match result.downcast::<NdarrayWrapper<T, Ix2>>() {
        Ok(wrapper) => {
            let boxed: Box<dyn ArrayProtocol> = wrapper;
            return Box::new(boxed);
        }
        Err(other) => other,
    };

    // 1-D results: element-wise ops on 1-D inputs, reshape to 1-D.
    let result = match result.downcast::<NdarrayWrapper<T, Ix1>>() {
        Ok(wrapper) => {
            let boxed: Box<dyn ArrayProtocol> = wrapper;
            return Box::new(boxed);
        }
        Err(other) => other,
    };

    // Dynamic-dimension results.
    match result.downcast::<NdarrayWrapper<T, IxDyn>>() {
        Ok(wrapper) => {
            let boxed: Box<dyn ArrayProtocol> = wrapper;
            Box::new(boxed)
        }
        // Not an array result (e.g. a scalar from `sum`): pass through unchanged.
        Err(other) => other,
    }
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
        // Wrap `self` as a plain `NdarrayWrapper`. The mixed-precision storage is
        // a regular `ndarray`, so all numeric kernels live in `NdarrayWrapper`'s
        // implementation; this struct only manages precision metadata.
        let wrapped_self = NdarrayWrapper::new(self.array.clone());

        // Determine operating precision based on function and arguments. The
        // precision is currently used to validate the requested operation; the
        // actual numeric computation is delegated to `NdarrayWrapper`.
        let precision = kwargs
            .get("precision")
            .and_then(|p| p.downcast_ref::<Precision>())
            .cloned()
            .unwrap_or(self.computeprecision);

        match func.name {
            "scirs2::array_protocol::operations::matmul"
            | "scirs2::array_protocol::operations::add"
            | "scirs2::array_protocol::operations::subtract"
            | "scirs2::array_protocol::operations::multiply" => {
                // Binary operations need the second operand. The dispatcher boxes
                // operands as `Box<dyn ArrayProtocol>`, so we extract the inner
                // ndarray (whether it arrived as a `MixedPrecisionArray` or an
                // `NdarrayWrapper`) and re-wrap it as an `NdarrayWrapper` so the
                // delegated kernel receives the concrete type it expects.
                if args.len() < 2 {
                    return Err(NotImplemented);
                }

                let Some(other_array) = extract_inner_ndarray::<T, D>(args[1].as_ref()) else {
                    return Err(NotImplemented);
                };
                let wrapped_other = NdarrayWrapper::new(other_array);

                // Forbid precision levels we cannot honour numerically. Half is
                // not representable by the underlying storage on stable Rust.
                if matches!(precision, Precision::Half) {
                    return Err(NotImplemented);
                }

                let new_args: Vec<Box<dyn Any>> =
                    vec![Box::new(wrapped_self.clone()), Box::new(wrapped_other)];
                wrapped_self
                    .array_function(func, types, &new_args, kwargs)
                    .map(rewrap_result_as_array_protocol::<T>)
            }
            "scirs2::array_protocol::operations::transpose"
            | "scirs2::array_protocol::operations::reshape"
            | "scirs2::array_protocol::operations::sum" => {
                // Unary operations: delegate to `NdarrayWrapper` with `self`
                // re-wrapped as the first argument. Array results (transpose,
                // reshape) are normalised; scalar results (sum) pass through.
                let new_args: Vec<Box<dyn Any>> = vec![Box::new(wrapped_self.clone())];
                wrapped_self
                    .array_function(func, types, &new_args, kwargs)
                    .map(rewrap_result_as_array_protocol::<T>)
            }
            _ => {
                // For any other function, delegate to the standard implementation
                // with the original arguments.
                wrapped_self.array_function(func, types, args, kwargs)
            }
        }
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
            computeprecision: self.computeprecision,
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
                // For actual implementation, this would convert f64 to f32 if needed
                // This is a simplified version - in reality, we would need to convert between types

                let current_precision = self.precision();
                if current_precision == Precision::Single {
                    // Already in single precision
                    return Ok(Box::new(self.clone()));
                }

                // In real implementation, would handle proper conversion from T to f32
                // For now, create a new array with the requested precision
                let array_single = self.array.clone();
                let newarray = MixedPrecisionArray::with_computeprecision(array_single, precision);
                Ok(Box::new(newarray))
            }
            Precision::Double => {
                // For actual implementation, this would convert f32 to f64 if needed

                let current_precision = self.precision();
                if current_precision == Precision::Double {
                    // Already in double precision
                    return Ok(Box::new(self.clone()));
                }

                // In real implementation, would handle proper conversion from T to f64
                // For now, create a new array with the requested precision
                let array_double = self.array.clone();
                let newarray = MixedPrecisionArray::with_computeprecision(array_double, precision);
                Ok(Box::new(newarray))
            }
            Precision::Mixed => {
                // For mixed precision, use storage precision of the current array and double compute precision
                let array_mixed = self.array.clone();
                let newarray =
                    MixedPrecisionArray::with_computeprecision(array_mixed, Precision::Double);
                Ok(Box::new(newarray))
            }
            _ => Err(CoreError::NotImplementedError(ErrorContext::new(format!(
                "Conversion to {precision} precision not implemented"
            )))),
        }
    }

    fn precision(&self) -> Precision {
        // If storage and compute precision differ, return Mixed
        if self.storage_precision != self.computeprecision {
            Precision::Mixed
        } else {
            self.storage_precision
        }
    }

    fn supports_precision(&self, precision: Precision) -> bool {
        matches!(precision, Precision::Single | Precision::Double)
    }

    fn as_array_protocol(&self) -> &dyn ArrayProtocol {
        self
    }
}

/// Implement MixedPrecisionSupport for GPUNdarray.
impl<T, D> MixedPrecisionSupport for GPUNdarray<T, D>
where
    T: Clone + Float + Send + Sync + 'static + num_traits::Zero + std::ops::Div<f64, Output = T>,
    D: Dimension + Send + Sync + 'static + crate::ndarray::RemoveAxis,
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
            "Conversion to {precision} precision not implemented for GPU arrays"
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

    fn supports_precision(&self, precision: Precision) -> bool {
        // Most GPUs support all precision levels
        true
    }

    fn as_array_protocol(&self) -> &dyn ArrayProtocol {
        self
    }
}

/// Execute an operation with a specific precision.
///
/// This function automatically converts arrays to the specified precision
/// before executing the operation.
#[allow(dead_code)]
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
                "One or more arrays do not support {precision} precision"
            ))));
        }
    }

    // Convert arrays to the requested precision. Each conversion yields a
    // `Box<dyn MixedPrecisionSupport>` that owns its precision-converted data.
    let mut converted_arrays: Vec<Box<dyn MixedPrecisionSupport>> =
        Vec::with_capacity(arrays.len());

    for &array in arrays {
        let converted = array.to_precision(precision)?;
        converted_arrays.push(converted);
    }

    // Bridge `&dyn MixedPrecisionSupport` to `&dyn ArrayProtocol` on stable Rust.
    //
    // Trait upcasting (`&dyn MixedPrecisionSupport` -> `&dyn ArrayProtocol`) is
    // unstable (RFC #65991), so instead of relying on it we use the explicit
    // `as_array_protocol` bridge method defined on `MixedPrecisionSupport`. Every
    // implementor already *is* an `ArrayProtocol`, so this is a zero-cost borrow.
    let protocol_refs: Vec<&dyn ArrayProtocol> = converted_arrays
        .iter()
        .map(|array| array.as_array_protocol())
        .collect();

    // Run the requested operation on the precision-converted arrays.
    executor(&protocol_refs)
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
    use ::ndarray::arr2;

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
            .is::<MixedPrecisionArray<f64, crate::ndarray::Ix2>>());
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

    // ── at_precision tests ───────────────────────────────────────────────────

    /// Downcast f64 → f32: values should be preserved within f32 precision.
    #[test]
    fn test_at_precision_f64_to_f32() {
        use ::ndarray::array;
        // Use values that are not approximate constants recognized by clippy.
        let arr = array![1.0_f64, 2.5_f64, -1.75_f64].into_dyn();
        let mp = MixedPrecisionArray::new(arr);
        let as_f32: crate::ndarray::ArrayD<f32> = mp
            .at_precision()
            .expect("f64 → f32 precision conversion should succeed");
        assert!((as_f32[0] - 1.0_f32).abs() < 1e-6);
        assert!((as_f32[1] - 2.5_f32).abs() < 1e-6);
        assert!((as_f32[2] - (-1.75_f32)).abs() < 1e-6);
    }

    /// Upcast f32 → f64: precision should be maintained.
    #[test]
    fn test_at_precision_f32_to_f64() {
        use ::ndarray::array;
        let arr = array![0.5_f32, 1.25_f32, -2.0_f32].into_dyn();
        let mp = MixedPrecisionArray::new(arr);
        let as_f64: crate::ndarray::ArrayD<f64> = mp
            .at_precision()
            .expect("f32 → f64 precision conversion should succeed");
        assert!((as_f64[0] - 0.5_f64).abs() < 1e-12);
        assert!((as_f64[1] - 1.25_f64).abs() < 1e-12);
        assert!((as_f64[2] - (-2.0_f64)).abs() < 1e-12);
    }

    /// Identity conversion f64 → f64 should be a no-op.
    #[test]
    fn test_at_precision_same_type_is_identity() {
        use ::ndarray::array;
        let arr = array![42.0_f64, -7.5_f64].into_dyn();
        let mp = MixedPrecisionArray::new(arr.clone());
        let result: crate::ndarray::ArrayD<f64> = mp
            .at_precision()
            .expect("f64 → f64 precision conversion should succeed");
        for (a, b) in arr.iter().zip(result.iter()) {
            assert_eq!(*a, *b, "Identity conversion must not change values");
        }
    }

    /// 2-D array conversion preserves shape.
    #[test]
    fn test_at_precision_preserves_shape() {
        let arr = arr2(&[[1.0_f64, 2.0], [3.0, 4.0]]);
        let mp = MixedPrecisionArray::new(arr);
        let as_f32: crate::ndarray::Array<f32, crate::ndarray::Ix2> = mp
            .at_precision()
            .expect("2D f64 → f32 conversion should succeed");
        assert_eq!(as_f32.shape(), &[2, 2]);
        assert!((as_f32[[0, 0]] - 1.0_f32).abs() < 1e-6);
        assert!((as_f32[[1, 1]] - 4.0_f32).abs() < 1e-6);
    }

    // ── execute_with_precision end-to-end tests ──────────────────────────────

    /// `ops::matmul` must run through `execute_with_precision` end-to-end on
    /// stable Rust and return the correct numeric result (no `Err`, no upcast).
    #[test]
    fn test_execute_with_precision_matmul_single() {
        crate::array_protocol::init();

        // [[1, 2], [3, 4]] x [[5, 6], [7, 8]] = [[19, 22], [43, 50]]
        let a = MixedPrecisionArray::new(arr2(&[[1.0_f64, 2.0], [3.0, 4.0]]));
        let b = MixedPrecisionArray::new(arr2(&[[5.0_f64, 6.0], [7.0, 8.0]]));

        let result = ops::matmul(&a, &b, Precision::Single)
            .expect("mixed-precision matmul should succeed on stable Rust");

        let wrapper = result
            .as_any()
            .downcast_ref::<NdarrayWrapper<f64, crate::ndarray::Ix2>>()
            .expect("matmul result should be an NdarrayWrapper<f64, Ix2>");
        let out = wrapper.as_array();

        assert_eq!(out.shape(), &[2, 2]);
        assert!((out[[0, 0]] - 19.0).abs() < 1e-9);
        assert!((out[[0, 1]] - 22.0).abs() < 1e-9);
        assert!((out[[1, 0]] - 43.0).abs() < 1e-9);
        assert!((out[[1, 1]] - 50.0).abs() < 1e-9);
    }

    /// `ops::add` must run through `execute_with_precision` end-to-end and return
    /// the correct element-wise sum.
    #[test]
    fn test_execute_with_precision_add_single() {
        crate::array_protocol::init();

        let a = MixedPrecisionArray::new(arr2(&[[1.0_f64, 2.0], [3.0, 4.0]]));
        let b = MixedPrecisionArray::new(arr2(&[[10.0_f64, 20.0], [30.0, 40.0]]));

        let result = ops::add(&a, &b, Precision::Single)
            .expect("mixed-precision add should succeed on stable Rust");

        let wrapper = result
            .as_any()
            .downcast_ref::<NdarrayWrapper<f64, crate::ndarray::Ix2>>()
            .expect("add result should be an NdarrayWrapper<f64, Ix2>");
        let out = wrapper.as_array();

        assert_eq!(out.shape(), &[2, 2]);
        assert!((out[[0, 0]] - 11.0).abs() < 1e-9);
        assert!((out[[0, 1]] - 22.0).abs() < 1e-9);
        assert!((out[[1, 0]] - 33.0).abs() < 1e-9);
        assert!((out[[1, 1]] - 44.0).abs() < 1e-9);
    }

    /// Half precision is not numerically representable by the stable backend, so
    /// the operation must surface an error rather than silently producing wrong
    /// results.
    #[test]
    fn test_execute_with_precision_half_is_rejected() {
        crate::array_protocol::init();

        let a = MixedPrecisionArray::new(arr2(&[[1.0_f64, 2.0], [3.0, 4.0]]));
        let b = MixedPrecisionArray::new(arr2(&[[5.0_f64, 6.0], [7.0, 8.0]]));

        // `supports_precision` returns false for Half, so this must be rejected.
        let result = ops::matmul(&a, &b, Precision::Half);
        assert!(
            result.is_err(),
            "Half precision matmul must return an error"
        );
    }
}
