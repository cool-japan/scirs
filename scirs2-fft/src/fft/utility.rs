//! Utility functions for FFT operations
//!
//! This module contains helper functions used by the FFT implementation.

use crate::error::{FFTError, FFTResult};
use ndarray::{Array, ArrayD, IxDyn};
use num_complex::Complex64;
use std::fmt::Debug;

/// Try to convert a value to Complex64
///
/// Attempts to interpret the given value as a Complex64. Currently only handles
/// direct Complex64 values, but can be extended to support more types.
///
/// # Arguments
///
/// * `val` - The value to convert
///
/// # Returns
///
/// * `Some(Complex64)` if the conversion was successful
/// * `None` if the conversion failed
pub(crate) fn try_as_complex<T: Copy + Debug + 'static>(val: T) -> Option<Complex64> {
    // Check if the type is Complex64
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<Complex64>() {
        unsafe {
            // This is a bit of a hack, but necessary for type detection at runtime
            let ptr = &val as *const T as *const Complex64;
            return Some(*ptr);
        }
    }

    None
}

/// Check if a number is a power of two
///
/// # Arguments
///
/// * `n` - The number to check
///
/// # Returns
///
/// `true` if the number is a power of 2, `false` otherwise
#[inline]
pub fn is_power_of_two(n: usize) -> bool {
    n != 0 && (n & (n - 1)) == 0
}

/// Calculate the next power of two greater than or equal to `n`
///
/// # Arguments
///
/// * `n` - The input number
///
/// # Returns
///
/// The next power of 2 that is greater than or equal to `n`
#[inline]
pub fn next_power_of_two(n: usize) -> usize {
    if n == 0 {
        return 1;
    }

    // If n is already a power of 2, return it
    if is_power_of_two(n) {
        return n;
    }

    // Find the position of the most significant bit
    let msb_pos = usize::BITS - n.leading_zeros();

    // Return 2^msb_pos
    1 << msb_pos
}

/// Validate FFT arguments and determine output size
///
/// # Arguments
///
/// * `input_size` - The size of the input data
/// * `n` - The requested output size (optional)
///
/// # Returns
///
/// The validated FFT size
///
/// # Errors
///
/// Returns an error if the input size is zero or n is zero
pub fn validate_fft_size(input_size: usize, n: Option<usize>) -> FFTResult<usize> {
    if input_size == 0 {
        return Err(FFTError::ValueError("Input cannot be empty".to_string()));
    }

    match n {
        Some(0) => Err(FFTError::ValueError("FFT size cannot be zero".to_string())),
        Some(size) => Ok(size),
        None => Ok(next_power_of_two(input_size)),
    }
}

/// Determine the N-dimensional shapes for FFT operations
///
/// # Arguments
///
/// * `input_shape` - The shape of the input array
/// * `shape` - The requested output shape (optional)
///
/// # Returns
///
/// The validated output shape
///
/// # Errors
///
/// Returns an error if the dimensions don't match
pub fn validate_fft_shapes(
    input_shape: &[usize],
    shape: Option<&[usize]>,
) -> FFTResult<Vec<usize>> {
    match shape {
        Some(output_shape) => {
            if output_shape.len() != input_shape.len() {
                return Err(FFTError::ValueError(
                    "Output shape must have the same number of dimensions as input".to_string(),
                ));
            }
            Ok(output_shape.to_vec())
        }
        None => Ok(input_shape.to_vec()),
    }
}

/// Validate axes for FFT operations
///
/// # Arguments
///
/// * `ndim` - The number of dimensions in the array
/// * `axes` - The requested axes for FFT (optional)
///
/// # Returns
///
/// The validated axes
///
/// # Errors
///
/// Returns an error if any axis is out of bounds
pub fn validate_fft_axes(ndim: usize, axes: Option<&[usize]>) -> FFTResult<Vec<usize>> {
    match axes {
        Some(axes) => {
            // Check for axes out of bounds
            for &axis in axes {
                if axis >= ndim {
                    return Err(FFTError::ValueError(format!(
                        "Axis {} out of bounds for array of dimension {}",
                        axis, ndim
                    )));
                }
            }
            Ok(axes.to_vec())
        }
        None => Ok((0..ndim).collect()),
    }
}

/// Create a complex array filled with zeros with the given shape
///
/// # Arguments
///
/// * `shape` - The shape of the array as a slice
///
/// # Returns
///
/// A new complex array with the specified shape
/// Marked as dead code but kept for API consistency
#[allow(dead_code)]
pub fn zeros_like_complex(shape: &[usize]) -> ArrayD<Complex64> {
    ArrayD::<Complex64>::zeros(IxDyn(shape))
}

/// Expand a real array into a complex array with zero imaginary part
///
/// # Arguments
///
/// * `real_array` - The real-valued array
///
/// # Returns
///
/// A complex array with the same shape
pub fn real_to_complex<D>(real_array: &Array<f64, D>) -> Array<Complex64, D>
where
    D: ndarray::Dimension,
{
    real_array.mapv(|x| Complex64::new(x, 0.0))
}

/// Extract the real part of a complex array
///
/// # Arguments
///
/// * `complex_array` - The complex-valued array
///
/// # Returns
///
/// A real array with the same shape
pub fn complex_to_real<D>(complex_array: &Array<Complex64, D>) -> Array<f64, D>
where
    D: ndarray::Dimension,
{
    complex_array.mapv(|x| x.re)
}

/// Calculate the magnitude of a complex array (absolute values)
///
/// # Arguments
///
/// * `complex_array` - The complex-valued array
///
/// # Returns
///
/// A real array with the magnitude (absolute value) of each element
pub fn complex_magnitude<D>(complex_array: &Array<Complex64, D>) -> Array<f64, D>
where
    D: ndarray::Dimension,
{
    complex_array.mapv(|x| x.norm())
}

/// Calculate the phase angle of a complex array (in radians)
///
/// # Arguments
///
/// * `complex_array` - The complex-valued array
///
/// # Returns
///
/// A real array with the phase angle of each element
pub fn complex_angle<D>(complex_array: &Array<Complex64, D>) -> Array<f64, D>
where
    D: ndarray::Dimension,
{
    complex_array.mapv(|x| x.arg())
}

/// Calculate the power spectrum of a complex array (squared magnitude)
///
/// # Arguments
///
/// * `complex_array` - The complex-valued array
///
/// # Returns
///
/// A real array with the power spectrum (squared magnitude) of each element
pub fn power_spectrum<D>(complex_array: &Array<Complex64, D>) -> Array<f64, D>
where
    D: ndarray::Dimension,
{
    complex_array.mapv(|x| x.norm_sqr())
}

/// Convert an array slice to an index vector
///
/// # Arguments
///
/// * `slice` - The array slice
///
/// # Returns
///
/// A vector of indices for the slice
/// Marked as dead code but kept for API consistency
#[allow(dead_code)]
pub fn slice_to_indices(slice: &[usize]) -> Vec<usize> {
    slice.to_vec()
}
