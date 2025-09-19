//! Core Universal Function implementation
//!
//! This module provides the foundational infrastructure for the universal function
//! (ufunc) system, including trait definitions, registration, and dispatching.

use ndarray::{
    Array, ArrayBase, ArrayView, ArrayViewMut, Data, DataMut, Dimension, Ix1, IxDyn, RawData,
};
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::sync::RwLock;

/// Enum defining the different kinds of universal functions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UFuncKind {
    /// Unary function (takes one input array)
    Unary,
    /// Binary function (takes two input arrays)
    Binary,
    /// Reduction function (reduces array along an axis)
    Reduction,
}

/// Trait for universal function implementation
pub trait UFunc: Send + Sync {
    /// Get the name of the ufunc
    fn name(&self) -> &str;

    /// Get the kind of ufunc (unary, binary, reduction)
    fn kind(&self) -> UFuncKind;

    /// Apply the ufunc to array(s) and store the result in the output array
    fn apply(
        &self,
        inputs: &[ArrayView<f64, IxDyn>],
        output: &mut ArrayViewMut<f64, IxDyn>,
    ) -> Result<(), &'static str>;

    /// Use SIMD acceleration if available
    fn use_simd(&self) -> bool {
        #[cfg(feature = "simd")]
        return true;

        #[cfg(not(feature = "simd"))]
        return false;
    }

    /// Use parallel execution if available
    fn use_parallel(&self) -> bool {
        #[cfg(feature = "parallel")]
        return true;

        #[cfg(not(feature = "parallel"))]
        return false;
    }
}

/// Global registry for universal functions
static UFUNC_REGISTRY: Lazy<RwLock<HashMap<String, Box<dyn UFunc>>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));

/// Register a universal function in the global registry
#[allow(dead_code)]
pub fn register_ufunc(ufunc: Box<dyn UFunc>) -> Result<(), &'static str> {
    let name = ufunc.name().to_string();

    let mut registry = UFUNC_REGISTRY.write().unwrap();

    if registry.contains_key(&name) {
        return Err("UFunc with this name already exists");
    }

    registry.insert(name, ufunc);
    Ok(())
}

/// Get a universal function from the registry by name
#[allow(dead_code)]
pub fn get_ufunc(name: &str) -> Option<Box<dyn UFunc>> {
    let registry = UFUNC_REGISTRY.read().unwrap();

    registry.get(name).map(|ufunc| {
        // Clone the UFunc implementation
        let ufunc_clone: Box<dyn UFunc> = Box::new(UFuncWrapper {
            name: ufunc.name().to_string(),
            kind: ufunc.kind(),
        });

        ufunc_clone
    })
}

/// A wrapper for UFunc implementations to allow cloning
struct UFuncWrapper {
    name: String,
    kind: UFuncKind,
}

impl UFunc for UFuncWrapper {
    fn name(&self) -> &str {
        &self.name
    }

    fn kind(&self) -> UFuncKind {
        self.kind
    }

    fn apply(
        &self,
        inputs: &[ArrayView<f64, IxDyn>],
        output: &mut ArrayViewMut<f64, IxDyn>,
    ) -> Result<(), &'static str> {
        // This is a wrapper that delegates to the actual implementation
        // Get the real UFunc from the registry
        let registry = UFUNC_REGISTRY.read().unwrap();

        if let Some(real_ufunc) = registry.get(&self.name) {
            real_ufunc.apply(inputs, output)
        } else {
            Err("UFunc not found in registry")
        }
    }
}

/// Helper function to apply a unary operation element-wise
#[allow(dead_code)]
pub fn apply_unary<T, F, O, S1, S2, D>(
    input: &ArrayBase<S1, D>,
    output: &mut ArrayBase<S2, D>,
    op: F,
) -> Result<(), &'static str>
where
    S1: Data<Elem = T>,
    S2: Data<Elem = O> + DataMut,
    T: Clone + Send + Sync,
    O: Clone + Send + Sync,
    F: Fn(&T) -> O + Send + Sync,
    D: Dimension,
{
    // Check that the output shape matches the input shape
    if input.shape() != output.shape() {
        return Err("Output shape must match input shape for unary operations");
    }

    // Apply the operation element-wise
    #[cfg(feature = "parallel")]
    {
        use crate::parallel_ops::*;
        // For simplicity, we convert to vectors, process in parallel, then convert back
        // A more efficient implementation would operate directly on array iterators
        let input_slice = input.as_slice().unwrap();
        let output_slice = output.as_slice_mut().unwrap();

        output_slice
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, out)| {
                let in_val = unsafe { input_slice.get_unchecked(i) };
                *out = op(in_val);
            });
    }

    #[cfg(not(feature = "parallel"))]
    {
        output.iter_mut().zip(input.iter()).for_each(|(out, inp)| {
            *out = op(inp);
        });
    }

    Ok(())
}

/// Helper function to apply a binary operation element-wise with broadcasting
#[allow(dead_code)]
pub fn apply_binary<T, F, O, S1, S2, S3, D>(
    input1: &ArrayBase<S1, D>,
    input2: &ArrayBase<S2, D>,
    output: &mut ArrayBase<S3, D>,
    op: F,
) -> Result<(), &'static str>
where
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
    S3: Data<Elem = O> + DataMut,
    T: Clone + Send + Sync,
    O: Clone + Send + Sync,
    F: Fn(&T, &T) -> O + Send + Sync,
    D: Dimension,
{
    // This is a simplified implementation without full broadcasting support
    // For a complete implementation, we would need to use the broadcasting module

    // For now, just check that all arrays have the same shape
    if input1.shape() != output.shape() || input2.shape() != output.shape() {
        return Err("All arrays must have the same shape for binary operations");
    }

    // Apply the operation element-wise
    #[cfg(feature = "parallel")]
    {
        use crate::parallel_ops::*;

        let input1_slice = input1.as_slice().unwrap();
        let input2_slice = input2.as_slice().unwrap();
        let output_slice = output.as_slice_mut().unwrap();

        output_slice
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, out)| {
                let in1 = unsafe { input1_slice.get_unchecked(i) };
                let in2 = unsafe { input2_slice.get_unchecked(i) };
                *out = op(in1, in2);
            });
    }

    #[cfg(not(feature = "parallel"))]
    {
        output
            .iter_mut()
            .zip(input1.iter().zip(input2.iter()))
            .for_each(|(out, (in1, in2))| {
                *out = op(in1, in2);
            });
    }

    Ok(())
}

/// Helper function to apply a reduction operation along an axis
#[allow(dead_code)]
pub fn apply_reduction<T, F, S1, S2, D>(
    input: &ArrayBase<S1, D>,
    output: &mut ArrayBase<S2, Ix1>,
    axis: Option<usize>,
    initial: Option<T>,
    op: F,
) -> Result<(), &'static str>
where
    S1: Data<Elem = T>,
    S2: Data<Elem = T> + DataMut,
    T: Clone + Send + Sync,
    F: Fn(T, &T) -> T + Send + Sync,
    D: Dimension,
{
    // This is a simplified implementation for reduction along an axis
    // In a complete implementation, we would handle all reduction patterns

    match axis {
        Some(ax) => {
            // Reduction along a specific axis
            if ax >= input.ndim() {
                return Err("Axis index out of bounds");
            }

            let axis_size = input.len_of(ndarray::Axis(ax));
            let othershape = input
                .shape()
                .iter()
                .enumerate()
                .filter_map(|(i, &s)| if i != ax { Some(s) } else { None })
                .collect::<Vec<_>>();

            // Check that the output shape matches the expected shape
            if output.shape() != othershape.as_slice() {
                return Err("Output shape does not match the expected shape for reduction");
            }

            // For simplicity, this implementation only handles 2D arrays and axis 0 or 1
            // A complete implementation would handle arbitrary dimensions
            if input.ndim() != 2 {
                return Err("This simplified implementation only supports 2D arrays");
            }

            let (rows, cols) = (input.shape()[0], input.shape()[1]);

            // Convert to slice for linear indexing
            if let Some(input_slice) = input.as_slice() {
                if ax == 0 {
                    // Reduce along rows
                    for j in 0..cols {
                        let mut acc = initial.clone().unwrap_or_else(|| {
                            // Get first element in this column
                            input_slice[j].clone()
                        });
                        let start_i = if initial.is_some() { 0 } else { 1 };
                        for i in start_i..rows {
                            // Use linear indexing for 2D array in row-major order
                            let val = &input_slice[i * cols + j];
                            acc = op(acc, val);
                        }
                        output[j] = acc;
                    }
                } else {
                    // Reduce along columns
                    for i in 0..rows {
                        let mut acc = initial.clone().unwrap_or_else(|| {
                            // Get first element in this row
                            input_slice[i * cols].clone()
                        });
                        let start_j = if initial.is_some() { 0 } else { 1 };
                        for j in start_j..cols {
                            // Use linear indexing for 2D array in row-major order
                            let val = &input_slice[i * cols + j];
                            acc = op(acc, val);
                        }
                        output[i] = acc;
                    }
                }
            } else {
                return Err("Input array is not contiguous");
            }
        }
        None => {
            // Reduction over the entire array
            if output.len() != 1 {
                return Err("Output array must have length 1 for full reduction");
            }

            let mut iter = input.iter();
            let mut acc = initial
                .clone()
                .unwrap_or_else(|| iter.next().unwrap().clone());

            for val in iter {
                acc = op(acc, val);
            }

            output[0] = acc;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array1, Array2};

    // Create a simple unary ufunc for testing
    struct TestUnaryUFunc;

    impl UFunc for TestUnaryUFunc {
        fn name(&self) -> &str {
            "test_unary"
        }

        fn kind(&self) -> UFuncKind {
            UFuncKind::Unary
        }

        fn apply(
            &self,
            inputs: &[ArrayView<f64, IxDyn>],
            output: &mut ArrayViewMut<f64, IxDyn>,
        ) -> Result<(), &'static str> {
            if inputs.len() != 1 {
                return Err("Unary ufunc requires exactly one input array");
            }

            // Square each element
            let input = &inputs[0];
            for (inp, out) in input.iter().zip(output.iter_mut()) {
                *out = inp * inp;
            }
            Ok(())
        }
    }

    // Create a simple binary ufunc for testing
    struct TestBinaryUFunc;

    impl UFunc for TestBinaryUFunc {
        fn name(&self) -> &str {
            "test_binary"
        }

        fn kind(&self) -> UFuncKind {
            UFuncKind::Binary
        }

        fn apply(
            &self,
            inputs: &[ArrayView<f64, IxDyn>],
            output: &mut ArrayViewMut<f64, IxDyn>,
        ) -> Result<(), &'static str> {
            if inputs.len() != 2 {
                return Err("Binary ufunc requires exactly two input arrays");
            }

            // Add the elements
            let input1 = &inputs[0];
            let input2 = &inputs[1];
            for ((a, b), out) in input1.iter().zip(input2.iter()).zip(output.iter_mut()) {
                *out = a + b;
            }
            Ok(())
        }
    }

    #[test]
    fn test_ufunc_registry() {
        // Register a test ufunc
        let ufunc = Box::new(TestUnaryUFunc);
        register_ufunc(ufunc).unwrap();

        // Get the ufunc from the registry
        let ufunc = get_ufunc("test_unary").unwrap();
        assert_eq!(ufunc.name(), "test_unary");
        assert_eq!(ufunc.kind(), UFuncKind::Unary);
    }

    #[test]
    fn test_apply_unary() {
        let input = array![1.0, 2.0, 3.0, 4.0];
        let mut output = Array1::<f64>::zeros(4);

        apply_unary(&input, &mut output, |&x: &f64| x * x).unwrap();

        assert_eq!(output, array![1.0, 4.0, 9.0, 16.0]);
    }

    #[test]
    fn test_apply_binary() {
        let input1 = array![1.0, 2.0, 3.0, 4.0];
        let input2 = array![5.0, 6.0, 7.0, 8.0];
        let mut output = Array1::<f64>::zeros(4);

        apply_binary(&input1, &input2, &mut output, |&x: &f64, &y: &f64| x + y).unwrap();

        assert_eq!(output, array![6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_apply_reduction() {
        let input = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

        // Reduction along axis 0 (sum of columns)
        let mut output = Array1::<f64>::zeros(3);
        apply_reduction(&input, &mut output, Some(0), Some(0.0), |acc, &x| acc + x).unwrap();
        assert_eq!(output, array![5.0, 7.0, 9.0]);

        // Reduction along axis 1 (sum of rows)
        let mut output = Array1::<f64>::zeros(2);
        apply_reduction(&input, &mut output, Some(1), Some(0.0), |acc, &x| acc + x).unwrap();
        assert_eq!(output, array![6.0, 15.0]);

        // Full reduction (sum of all elements)
        let mut output = Array1::<f64>::zeros(1);
        apply_reduction(&input, &mut output, None, Some(0.0), |acc, &x| acc + x).unwrap();
        assert_eq!(output, array![21.0]);
    }
}
