//! Mathematical unary universal functions
//!
//! This module provides implementation of common mathematical functions
//! as universal functions for efficient vectorized operations.

use ndarray::{Array, ArrayView, Dimension, IxDyn, ShapeBuilder};
use crate::ufuncs::core::{UFunc, UFuncKind, apply_unary, register_ufunc};
use std::sync::Once;

static INIT: Once = Once::new();

// Initialize the ufunc registry with mathematical functions
fn init_math_ufuncs() {
    INIT.call_once(|| {
        // Register all the mathematical ufuncs
        let _ = register_ufunc(Box::new(SinUFunc));
        let _ = register_ufunc(Box::new(CosUFunc));
        let _ = register_ufunc(Box::new(TanUFunc));
        let _ = register_ufunc(Box::new(ExpUFunc));
        let _ = register_ufunc(Box::new(LogUFunc));
        let _ = register_ufunc(Box::new(SqrtUFunc));
        let _ = register_ufunc(Box::new(AbsUFunc));
    });
}

// Define the unary ufuncs

/// Sine universal function
pub struct SinUFunc;

impl UFunc for SinUFunc {
    fn name(&self) -> &str {
        "sin"
    }

    fn kind(&self) -> UFuncKind {
        UFuncKind::Unary
    }

    fn apply<D>(&self, inputs: &[&ndarray::ArrayBase<ndarray::Data, D>], output: &mut ndarray::ArrayBase<ndarray::Data, D>) -> Result<(), &'static str>
    where
        D: Dimension,
    {
        if inputs.len() != 1 {
            return Err("Sin requires exactly one input array");
        }

        // Apply the sine function element-wise
        apply_unary(inputs[0], output, |&x: &f64| x.sin())
    }
}

/// Cosine universal function
pub struct CosUFunc;

impl UFunc for CosUFunc {
    fn name(&self) -> &str {
        "cos"
    }

    fn kind(&self) -> UFuncKind {
        UFuncKind::Unary
    }

    fn apply<D>(&self, inputs: &[&ndarray::ArrayBase<ndarray::Data, D>], output: &mut ndarray::ArrayBase<ndarray::Data, D>) -> Result<(), &'static str>
    where
        D: Dimension,
    {
        if inputs.len() != 1 {
            return Err("Cos requires exactly one input array");
        }

        // Apply the cosine function element-wise
        apply_unary(inputs[0], output, |&x: &f64| x.cos())
    }
}

/// Tangent universal function
pub struct TanUFunc;

impl UFunc for TanUFunc {
    fn name(&self) -> &str {
        "tan"
    }

    fn kind(&self) -> UFuncKind {
        UFuncKind::Unary
    }

    fn apply<D>(&self, inputs: &[&ndarray::ArrayBase<ndarray::Data, D>], output: &mut ndarray::ArrayBase<ndarray::Data, D>) -> Result<(), &'static str>
    where
        D: Dimension,
    {
        if inputs.len() != 1 {
            return Err("Tan requires exactly one input array");
        }

        // Apply the tangent function element-wise
        apply_unary(inputs[0], output, |&x: &f64| x.tan())
    }
}

/// Exponential universal function
pub struct ExpUFunc;

impl UFunc for ExpUFunc {
    fn name(&self) -> &str {
        "exp"
    }

    fn kind(&self) -> UFuncKind {
        UFuncKind::Unary
    }

    fn apply<D>(&self, inputs: &[&ndarray::ArrayBase<ndarray::Data, D>], output: &mut ndarray::ArrayBase<ndarray::Data, D>) -> Result<(), &'static str>
    where
        D: Dimension,
    {
        if inputs.len() != 1 {
            return Err("Exp requires exactly one input array");
        }

        // Apply the exponential function element-wise
        apply_unary(inputs[0], output, |&x: &f64| x.exp())
    }
}

/// Natural logarithm universal function
pub struct LogUFunc;

impl UFunc for LogUFunc {
    fn name(&self) -> &str {
        "log"
    }

    fn kind(&self) -> UFuncKind {
        UFuncKind::Unary
    }

    fn apply<D>(&self, inputs: &[&ndarray::ArrayBase<ndarray::Data, D>], output: &mut ndarray::ArrayBase<ndarray::Data, D>) -> Result<(), &'static str>
    where
        D: Dimension,
    {
        if inputs.len() != 1 {
            return Err("Log requires exactly one input array");
        }

        // Apply the natural logarithm function element-wise
        apply_unary(inputs[0], output, |&x: &f64| x.ln())
    }
}

/// Square root universal function
pub struct SqrtUFunc;

impl UFunc for SqrtUFunc {
    fn name(&self) -> &str {
        "sqrt"
    }

    fn kind(&self) -> UFuncKind {
        UFuncKind::Unary
    }

    fn apply<D>(&self, inputs: &[&ndarray::ArrayBase<ndarray::Data, D>], output: &mut ndarray::ArrayBase<ndarray::Data, D>) -> Result<(), &'static str>
    where
        D: Dimension,
    {
        if inputs.len() != 1 {
            return Err("Sqrt requires exactly one input array");
        }

        // Apply the square root function element-wise
        apply_unary(inputs[0], output, |&x: &f64| x.sqrt())
    }
}

/// Absolute value universal function
pub struct AbsUFunc;

impl UFunc for AbsUFunc {
    fn name(&self) -> &str {
        "abs"
    }

    fn kind(&self) -> UFuncKind {
        UFuncKind::Unary
    }

    fn apply<D>(&self, inputs: &[&ndarray::ArrayBase<ndarray::Data, D>], output: &mut ndarray::ArrayBase<ndarray::Data, D>) -> Result<(), &'static str>
    where
        D: Dimension,
    {
        if inputs.len() != 1 {
            return Err("Abs requires exactly one input array");
        }

        // Apply the absolute value function element-wise
        apply_unary(inputs[0], output, |&x: &f64| x.abs())
    }
}

// Convenience functions for applying ufuncs

/// Apply the sine function to each element of the array
///
/// # Arguments
///
/// * `array` - The input array
///
/// # Returns
///
/// An array with the sine of each element of the input array
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_core::ufuncs::sin;
/// use std::f64::consts::PI;
///
/// let a = array![0.0, PI/2.0, PI];
/// let result = sin(&a);
/// assert!((result[0] - 0.0).abs() < 1e-10);
/// assert!((result[1] - 1.0).abs() < 1e-10);
/// assert!((result[2] - 0.0).abs() < 1e-10);
/// ```
pub fn sin<D>(array: &ndarray::ArrayBase<ndarray::Data, D>) -> Array<f64, D>
where
    D: Dimension,
{
    // Initialize the ufuncs registry if needed
    init_math_ufuncs();

    // Create output array
    let mut result = Array::zeros(array.raw_dim());

    // Apply the sine function
    let sin_ufunc = SinUFunc;
    sin_ufunc.apply(&[array], &mut result).unwrap();

    result
}

/// Apply the cosine function to each element of the array
///
/// # Arguments
///
/// * `array` - The input array
///
/// # Returns
///
/// An array with the cosine of each element of the input array
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_core::ufuncs::cos;
/// use std::f64::consts::PI;
///
/// let a = array![0.0, PI/2.0, PI];
/// let result = cos(&a);
/// assert!((result[0] - 1.0).abs() < 1e-10);
/// assert!((result[1] - 0.0).abs() < 1e-10);
/// assert!((result[2] + 1.0).abs() < 1e-10);
/// ```
pub fn cos<D>(array: &ndarray::ArrayBase<ndarray::Data, D>) -> Array<f64, D>
where
    D: Dimension,
{
    // Initialize the ufuncs registry if needed
    init_math_ufuncs();

    // Create output array
    let mut result = Array::zeros(array.raw_dim());

    // Apply the cosine function
    let cos_ufunc = CosUFunc;
    cos_ufunc.apply(&[array], &mut result).unwrap();

    result
}

/// Apply the tangent function to each element of the array
///
/// # Arguments
///
/// * `array` - The input array
///
/// # Returns
///
/// An array with the tangent of each element of the input array
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_core::ufuncs::tan;
/// use std::f64::consts::PI;
///
/// let a = array![0.0, PI/4.0];
/// let result = tan(&a);
/// assert!((result[0] - 0.0).abs() < 1e-10);
/// assert!((result[1] - 1.0).abs() < 1e-10);
/// ```
pub fn tan<D>(array: &ndarray::ArrayBase<ndarray::Data, D>) -> Array<f64, D>
where
    D: Dimension,
{
    // Initialize the ufuncs registry if needed
    init_math_ufuncs();

    // Create output array
    let mut result = Array::zeros(array.raw_dim());

    // Apply the tangent function
    let tan_ufunc = TanUFunc;
    tan_ufunc.apply(&[array], &mut result).unwrap();

    result
}

/// Apply the exponential function to each element of the array
///
/// # Arguments
///
/// * `array` - The input array
///
/// # Returns
///
/// An array with the exponential of each element of the input array
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_core::ufuncs::exp;
///
/// let a = array![0.0, 1.0];
/// let result = exp(&a);
/// assert!((result[0] - 1.0).abs() < 1e-10);
/// assert!((result[1] - std::f64::consts::E).abs() < 1e-10);
/// ```
pub fn exp<D>(array: &ndarray::ArrayBase<ndarray::Data, D>) -> Array<f64, D>
where
    D: Dimension,
{
    // Initialize the ufuncs registry if needed
    init_math_ufuncs();

    // Create output array
    let mut result = Array::zeros(array.raw_dim());

    // Apply the exponential function
    let exp_ufunc = ExpUFunc;
    exp_ufunc.apply(&[array], &mut result).unwrap();

    result
}

/// Apply the natural logarithm function to each element of the array
///
/// # Arguments
///
/// * `array` - The input array
///
/// # Returns
///
/// An array with the natural logarithm of each element of the input array
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_core::ufuncs::log;
///
/// let a = array![1.0, std::f64::consts::E];
/// let result = log(&a);
/// assert!((result[0] - 0.0).abs() < 1e-10);
/// assert!((result[1] - 1.0).abs() < 1e-10);
/// ```
pub fn log<D>(array: &ndarray::ArrayBase<ndarray::Data, D>) -> Array<f64, D>
where
    D: Dimension,
{
    // Initialize the ufuncs registry if needed
    init_math_ufuncs();

    // Create output array
    let mut result = Array::zeros(array.raw_dim());

    // Apply the natural logarithm function
    let log_ufunc = LogUFunc;
    log_ufunc.apply(&[array], &mut result).unwrap();

    result
}

/// Apply the square root function to each element of the array
///
/// # Arguments
///
/// * `array` - The input array
///
/// # Returns
///
/// An array with the square root of each element of the input array
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_core::ufuncs::sqrt;
///
/// let a = array![1.0, 4.0, 9.0];
/// let result = sqrt(&a);
/// assert_eq!(result, array![1.0, 2.0, 3.0]);
/// ```
pub fn sqrt<D>(array: &ndarray::ArrayBase<ndarray::Data, D>) -> Array<f64, D>
where
    D: Dimension,
{
    // Initialize the ufuncs registry if needed
    init_math_ufuncs();

    // Create output array
    let mut result = Array::zeros(array.raw_dim());

    // Apply the square root function
    let sqrt_ufunc = SqrtUFunc;
    sqrt_ufunc.apply(&[array], &mut result).unwrap();

    result
}

/// Apply the absolute value function to each element of the array
///
/// # Arguments
///
/// * `array` - The input array
///
/// # Returns
///
/// An array with the absolute value of each element of the input array
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_core::ufuncs::abs;
///
/// let a = array![-1.0, 0.0, 1.0];
/// let result = abs(&a);
/// assert_eq!(result, array![1.0, 0.0, 1.0]);
/// ```
pub fn abs<D>(array: &ndarray::ArrayBase<ndarray::Data, D>) -> Array<f64, D>
where
    D: Dimension,
{
    // Initialize the ufuncs registry if needed
    init_math_ufuncs();

    // Create output array
    let mut result = Array::zeros(array.raw_dim());

    // Apply the absolute value function
    let abs_ufunc = AbsUFunc;
    abs_ufunc.apply(&[array], &mut result).unwrap();

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use std::f64::consts::PI;

    #[test]
    fn test_sin() {
        let a = array![0.0, PI/2.0, PI];
        let result = sin(&a);
        assert!((result[0] - 0.0).abs() < 1e-10);
        assert!((result[1] - 1.0).abs() < 1e-10);
        assert!((result[2] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_cos() {
        let a = array![0.0, PI/2.0, PI];
        let result = cos(&a);
        assert!((result[0] - 1.0).abs() < 1e-10);
        assert!((result[1] - 0.0).abs() < 1e-10);
        assert!((result[2] + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_sqrt() {
        let a = array![1.0, 4.0, 9.0];
        let result = sqrt(&a);
        assert_eq!(result, array![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_abs() {
        let a = array![-1.0, 0.0, 1.0];
        let result = abs(&a);
        assert_eq!(result, array![1.0, 0.0, 1.0]);
    }
}