//! Binary universal functions
//!
//! This module provides implementation of common binary operations
//! (addition, subtraction, etc.) as universal functions for efficient
//! vectorized operations with broadcasting support.

use ndarray::{Array, ArrayView, Dimension, IxDyn, ShapeBuilder};
use crate::ufuncs::core::{UFunc, UFuncKind, apply_binary, register_ufunc};
use crate::ndarray_ext::broadcasting::{broadcast_arrays, broadcast_apply};
use std::sync::Once;

static INIT: Once = Once::new();

// Initialize the ufunc registry with binary operations
fn init_binary_ufuncs() {
    INIT.call_once(|| {
        // Register all the binary ufuncs
        let _ = register_ufunc(Box::new(AddUFunc));
        let _ = register_ufunc(Box::new(SubtractUFunc));
        let _ = register_ufunc(Box::new(MultiplyUFunc));
        let _ = register_ufunc(Box::new(DivideUFunc));
        let _ = register_ufunc(Box::new(PowerUFunc));
    });
}

// Define the binary ufuncs

/// Addition universal function
pub struct AddUFunc;

impl UFunc for AddUFunc {
    fn name(&self) -> &str {
        "add"
    }

    fn kind(&self) -> UFuncKind {
        UFuncKind::Binary
    }

    fn apply<D>(&self, inputs: &[&ndarray::ArrayBase<ndarray::Data, D>], output: &mut ndarray::ArrayBase<ndarray::Data, D>) -> Result<(), &'static str>
    where
        D: Dimension,
    {
        if inputs.len() != 2 {
            return Err("Add requires exactly two input arrays");
        }

        // Apply addition element-wise
        apply_binary(inputs[0], inputs[1], output, |&x: &f64, &y: &f64| x + y)
    }
}

/// Subtraction universal function
pub struct SubtractUFunc;

impl UFunc for SubtractUFunc {
    fn name(&self) -> &str {
        "subtract"
    }

    fn kind(&self) -> UFuncKind {
        UFuncKind::Binary
    }

    fn apply<D>(&self, inputs: &[&ndarray::ArrayBase<ndarray::Data, D>], output: &mut ndarray::ArrayBase<ndarray::Data, D>) -> Result<(), &'static str>
    where
        D: Dimension,
    {
        if inputs.len() != 2 {
            return Err("Subtract requires exactly two input arrays");
        }

        // Apply subtraction element-wise
        apply_binary(inputs[0], inputs[1], output, |&x: &f64, &y: &f64| x - y)
    }
}

/// Multiplication universal function
pub struct MultiplyUFunc;

impl UFunc for MultiplyUFunc {
    fn name(&self) -> &str {
        "multiply"
    }

    fn kind(&self) -> UFuncKind {
        UFuncKind::Binary
    }

    fn apply<D>(&self, inputs: &[&ndarray::ArrayBase<ndarray::Data, D>], output: &mut ndarray::ArrayBase<ndarray::Data, D>) -> Result<(), &'static str>
    where
        D: Dimension,
    {
        if inputs.len() != 2 {
            return Err("Multiply requires exactly two input arrays");
        }

        // Apply multiplication element-wise
        apply_binary(inputs[0], inputs[1], output, |&x: &f64, &y: &f64| x * y)
    }
}

/// Division universal function
pub struct DivideUFunc;

impl UFunc for DivideUFunc {
    fn name(&self) -> &str {
        "divide"
    }

    fn kind(&self) -> UFuncKind {
        UFuncKind::Binary
    }

    fn apply<D>(&self, inputs: &[&ndarray::ArrayBase<ndarray::Data, D>], output: &mut ndarray::ArrayBase<ndarray::Data, D>) -> Result<(), &'static str>
    where
        D: Dimension,
    {
        if inputs.len() != 2 {
            return Err("Divide requires exactly two input arrays");
        }

        // Apply division element-wise
        apply_binary(inputs[0], inputs[1], output, |&x: &f64, &y: &f64| {
            if y == 0.0 {
                f64::NAN // Return NaN for division by zero
            } else {
                x / y
            }
        })
    }
}

/// Power (exponentiation) universal function
pub struct PowerUFunc;

impl UFunc for PowerUFunc {
    fn name(&self) -> &str {
        "power"
    }

    fn kind(&self) -> UFuncKind {
        UFuncKind::Binary
    }

    fn apply<D>(&self, inputs: &[&ndarray::ArrayBase<ndarray::Data, D>], output: &mut ndarray::ArrayBase<ndarray::Data, D>) -> Result<(), &'static str>
    where
        D: Dimension,
    {
        if inputs.len() != 2 {
            return Err("Power requires exactly two input arrays");
        }

        // Apply power function element-wise
        apply_binary(inputs[0], inputs[1], output, |&x: &f64, &y: &f64| x.powf(y))
    }
}

// Convenience functions for applying binary ufuncs

/// Add arrays element-wise with broadcasting
///
/// # Arguments
///
/// * `a` - First input array
/// * `b` - Second input array
///
/// # Returns
///
/// An array with the sum of the two input arrays
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_core::ufuncs::add;
///
/// let a = array![1.0, 2.0, 3.0];
/// let b = array![4.0, 5.0, 6.0];
/// let result = add(&a, &b);
/// assert_eq!(result, array![5.0, 7.0, 9.0]);
///
/// // With broadcasting
/// let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
/// let b = array![10.0, 20.0, 30.0];
/// let result = add(&a, &b);
/// assert_eq!(result, array![[11.0, 22.0, 33.0], [14.0, 25.0, 36.0]]);
/// ```
pub fn add<D1, D2, S1, S2>(a: &ndarray::ArrayBase<S1, D1>, b: &ndarray::ArrayBase<S2, D2>) -> Array<f64, IxDyn>
where
    D1: Dimension,
    D2: Dimension,
    S1: ndarray::Data<Elem = f64>,
    S2: ndarray::Data<Elem = f64>,
{
    // Initialize the ufuncs registry if needed
    init_binary_ufuncs();

    // Use broadcasting to handle arrays of different shapes
    // We need to convert to dynamic dimension for broadcasting
    let a_view = a.view().into_dyn();
    let b_view = b.view().into_dyn();

    // Try to broadcast the arrays
    broadcast_apply(a_view, b_view, |x, y| x + y).unwrap_or_else(|_| {
        // If broadcasting fails, assume arrays are the same shape
        // and apply operation directly
        let mut result = Array::<f64, _>::zeros(a.raw_dim().into_dyn());

        let add_ufunc = AddUFunc;
        if let Err(_) = add_ufunc.apply(&[&a.view(), &b.view()], &mut result) {
            panic!("Arrays are not compatible for addition");
        }

        result
    })
}

/// Subtract arrays element-wise with broadcasting
///
/// # Arguments
///
/// * `a` - First input array
/// * `b` - Second input array
///
/// # Returns
///
/// An array with the difference of the two input arrays (a - b)
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_core::ufuncs::subtract;
///
/// let a = array![5.0, 7.0, 9.0];
/// let b = array![1.0, 2.0, 3.0];
/// let result = subtract(&a, &b);
/// assert_eq!(result, array![4.0, 5.0, 6.0]);
///
/// // With broadcasting
/// let a = array![[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]];
/// let b = array![1.0, 2.0, 3.0];
/// let result = subtract(&a, &b);
/// assert_eq!(result, array![[9.0, 18.0, 27.0], [39.0, 48.0, 57.0]]);
/// ```
pub fn subtract<D1, D2, S1, S2>(a: &ndarray::ArrayBase<S1, D1>, b: &ndarray::ArrayBase<S2, D2>) -> Array<f64, IxDyn>
where
    D1: Dimension,
    D2: Dimension,
    S1: ndarray::Data<Elem = f64>,
    S2: ndarray::Data<Elem = f64>,
{
    // Initialize the ufuncs registry if needed
    init_binary_ufuncs();

    // Use broadcasting to handle arrays of different shapes
    let a_view = a.view().into_dyn();
    let b_view = b.view().into_dyn();

    // Try to broadcast the arrays
    broadcast_apply(a_view, b_view, |x, y| x - y).unwrap_or_else(|_| {
        // If broadcasting fails, assume arrays are the same shape
        // and apply operation directly
        let mut result = Array::<f64, _>::zeros(a.raw_dim().into_dyn());

        let subtract_ufunc = SubtractUFunc;
        if let Err(_) = subtract_ufunc.apply(&[&a.view(), &b.view()], &mut result) {
            panic!("Arrays are not compatible for subtraction");
        }

        result
    })
}

/// Multiply arrays element-wise with broadcasting
///
/// # Arguments
///
/// * `a` - First input array
/// * `b` - Second input array
///
/// # Returns
///
/// An array with the product of the two input arrays
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_core::ufuncs::multiply;
///
/// let a = array![1.0, 2.0, 3.0];
/// let b = array![4.0, 5.0, 6.0];
/// let result = multiply(&a, &b);
/// assert_eq!(result, array![4.0, 10.0, 18.0]);
///
/// // With broadcasting
/// let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
/// let b = array![10.0, 20.0, 30.0];
/// let result = multiply(&a, &b);
/// assert_eq!(result, array![[10.0, 40.0, 90.0], [40.0, 100.0, 180.0]]);
/// ```
pub fn multiply<D1, D2, S1, S2>(a: &ndarray::ArrayBase<S1, D1>, b: &ndarray::ArrayBase<S2, D2>) -> Array<f64, IxDyn>
where
    D1: Dimension,
    D2: Dimension,
    S1: ndarray::Data<Elem = f64>,
    S2: ndarray::Data<Elem = f64>,
{
    // Initialize the ufuncs registry if needed
    init_binary_ufuncs();

    // Use broadcasting to handle arrays of different shapes
    let a_view = a.view().into_dyn();
    let b_view = b.view().into_dyn();

    // Try to broadcast the arrays
    broadcast_apply(a_view, b_view, |x, y| x * y).unwrap_or_else(|_| {
        // If broadcasting fails, assume arrays are the same shape
        // and apply operation directly
        let mut result = Array::<f64, _>::zeros(a.raw_dim().into_dyn());

        let multiply_ufunc = MultiplyUFunc;
        if let Err(_) = multiply_ufunc.apply(&[&a.view(), &b.view()], &mut result) {
            panic!("Arrays are not compatible for multiplication");
        }

        result
    })
}

/// Divide arrays element-wise with broadcasting
///
/// # Arguments
///
/// * `a` - First input array (numerator)
/// * `b` - Second input array (denominator)
///
/// # Returns
///
/// An array with the quotient of the two input arrays (a / b)
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_core::ufuncs::divide;
///
/// let a = array![4.0, 10.0, 18.0];
/// let b = array![1.0, 2.0, 3.0];
/// let result = divide(&a, &b);
/// assert_eq!(result, array![4.0, 5.0, 6.0]);
///
/// // With broadcasting
/// let a = array![[10.0, 40.0, 90.0], [40.0, 100.0, 180.0]];
/// let b = array![10.0, 20.0, 30.0];
/// let result = divide(&a, &b);
/// assert_eq!(result, array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
/// ```
pub fn divide<D1, D2, S1, S2>(a: &ndarray::ArrayBase<S1, D1>, b: &ndarray::ArrayBase<S2, D2>) -> Array<f64, IxDyn>
where
    D1: Dimension,
    D2: Dimension,
    S1: ndarray::Data<Elem = f64>,
    S2: ndarray::Data<Elem = f64>,
{
    // Initialize the ufuncs registry if needed
    init_binary_ufuncs();

    // Use broadcasting to handle arrays of different shapes
    let a_view = a.view().into_dyn();
    let b_view = b.view().into_dyn();

    // Try to broadcast the arrays
    broadcast_apply(a_view, b_view, |x, y| {
        if *y == 0.0 {
            f64::NAN // Return NaN for division by zero
        } else {
            x / y
        }
    }).unwrap_or_else(|_| {
        // If broadcasting fails, assume arrays are the same shape
        // and apply operation directly
        let mut result = Array::<f64, _>::zeros(a.raw_dim().into_dyn());

        let divide_ufunc = DivideUFunc;
        if let Err(_) = divide_ufunc.apply(&[&a.view(), &b.view()], &mut result) {
            panic!("Arrays are not compatible for division");
        }

        result
    })
}

/// Raise arrays element-wise to a power with broadcasting
///
/// # Arguments
///
/// * `a` - First input array (base)
/// * `b` - Second input array (exponent)
///
/// # Returns
///
/// An array with each element of `a` raised to the power of the corresponding element of `b`
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_core::ufuncs::power;
///
/// let a = array![2.0, 3.0, 4.0];
/// let b = array![2.0, 2.0, 2.0];
/// let result = power(&a, &b);
/// assert_eq!(result, array![4.0, 9.0, 16.0]);
///
/// // With broadcasting
/// let a = array![[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]];
/// let b = array![2.0, 3.0, 2.0];
/// let result = power(&a, &b);
/// assert_eq!(result, array![[4.0, 27.0, 16.0], [25.0, 216.0, 49.0]]);
/// ```
pub fn power<D1, D2, S1, S2>(a: &ndarray::ArrayBase<S1, D1>, b: &ndarray::ArrayBase<S2, D2>) -> Array<f64, IxDyn>
where
    D1: Dimension,
    D2: Dimension,
    S1: ndarray::Data<Elem = f64>,
    S2: ndarray::Data<Elem = f64>,
{
    // Initialize the ufuncs registry if needed
    init_binary_ufuncs();

    // Use broadcasting to handle arrays of different shapes
    let a_view = a.view().into_dyn();
    let b_view = b.view().into_dyn();

    // Try to broadcast the arrays
    broadcast_apply(a_view, b_view, |x, y| x.powf(*y)).unwrap_or_else(|_| {
        // If broadcasting fails, assume arrays are the same shape
        // and apply operation directly
        let mut result = Array::<f64, _>::zeros(a.raw_dim().into_dyn());

        let power_ufunc = PowerUFunc;
        if let Err(_) = power_ufunc.apply(&[&a.view(), &b.view()], &mut result) {
            panic!("Arrays are not compatible for power operation");
        }

        result
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_add() {
        let a = array![1.0, 2.0, 3.0];
        let b = array![4.0, 5.0, 6.0];
        let result = add(&a, &b);
        assert_eq!(result, array![5.0, 7.0, 9.0]);

        // Test broadcasting
        let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let b = array![10.0, 20.0, 30.0];
        let result = add(&a, &b);
        assert_eq!(result, array![[11.0, 22.0, 33.0], [14.0, 25.0, 36.0]]);
    }

    #[test]
    fn test_subtract() {
        let a = array![5.0, 7.0, 9.0];
        let b = array![1.0, 2.0, 3.0];
        let result = subtract(&a, &b);
        assert_eq!(result, array![4.0, 5.0, 6.0]);

        // Test broadcasting
        let a = array![[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]];
        let b = array![1.0, 2.0, 3.0];
        let result = subtract(&a, &b);
        assert_eq!(result, array![[9.0, 18.0, 27.0], [39.0, 48.0, 57.0]]);
    }

    #[test]
    fn test_multiply() {
        let a = array![1.0, 2.0, 3.0];
        let b = array![4.0, 5.0, 6.0];
        let result = multiply(&a, &b);
        assert_eq!(result, array![4.0, 10.0, 18.0]);

        // Test broadcasting
        let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let b = array![10.0, 20.0, 30.0];
        let result = multiply(&a, &b);
        assert_eq!(result, array![[10.0, 40.0, 90.0], [40.0, 100.0, 180.0]]);
    }

    #[test]
    fn test_divide() {
        let a = array![4.0, 10.0, 18.0];
        let b = array![1.0, 2.0, 3.0];
        let result = divide(&a, &b);
        assert_eq!(result, array![4.0, 5.0, 6.0]);

        // Test broadcasting
        let a = array![[10.0, 40.0, 90.0], [40.0, 100.0, 180.0]];
        let b = array![10.0, 20.0, 30.0];
        let result = divide(&a, &b);
        assert_eq!(result, array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);

        // Test division by zero
        let a = array![1.0, 2.0, 3.0];
        let b = array![1.0, 0.0, 3.0];
        let result = divide(&a, &b);
        assert_eq!(result[0], 1.0);
        assert!(result[1].is_nan());
        assert_eq!(result[2], 1.0);
    }

    #[test]
    fn test_power() {
        let a = array![2.0, 3.0, 4.0];
        let b = array![2.0, 2.0, 2.0];
        let result = power(&a, &b);
        assert_eq!(result, array![4.0, 9.0, 16.0]);

        // Test broadcasting
        let a = array![[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]];
        let b = array![2.0, 3.0, 2.0];
        let result = power(&a, &b);
        assert_eq!(result, array![[4.0, 27.0, 16.0], [25.0, 216.0, 49.0]]);
    }
}