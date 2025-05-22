//! Utility functions for numerical operations
//!
//! This module provides common utility functions used throughout SciRS2.

use ndarray::{Array, Array1, Array2, ArrayBase, Data, Dimension};
use num_traits::{Float, FromPrimitive, Num, NumCast};
use std::fmt::Debug;

/// Checks if two floating-point values are approximately equal
///
/// # Arguments
///
/// * `a` - First value
/// * `b` - Second value
/// * `abs_tol` - Absolute tolerance
/// * `rel_tol` - Relative tolerance
///
/// # Returns
///
/// * `true` if the values are approximately equal, `false` otherwise
pub fn is_close<F: Float>(a: F, b: F, abs_tol: F, rel_tol: F) -> bool {
    let abs_diff = (a - b).abs();

    if abs_diff <= abs_tol {
        return true;
    }

    let abs_a = a.abs();
    let abs_b = b.abs();
    let max_abs = if abs_a > abs_b { abs_a } else { abs_b };

    abs_diff <= max_abs * rel_tol
}

/// Check if two points are equal within a tolerance
///
/// Compares each element of the points to determine if they are
/// approximately equal within a specified tolerance.
///
/// # Arguments
///
/// * `point1` - First point as a slice
/// * `point2` - Second point as a slice
/// * `tol` - Tolerance (default: 1e-8)
///
/// # Returns
///
/// * True if points are equal within tolerance
///
/// # Examples
///
/// ```
/// use scirs2_core::utils::points_equal;
///
/// let point1 = [1.0, 2.0, 3.0];
/// let point2 = [1.0, 2.0, 3.0];
/// let point3 = [1.0, 2.0, 3.001];
///
/// assert!(points_equal(&point1, &point2, None));
/// assert!(!points_equal(&point1, &point3, None));
/// assert!(points_equal(&point1, &point3, Some(0.01)));
/// ```
pub fn points_equal<T>(point1: &[T], point2: &[T], tol: Option<T>) -> bool
where
    T: PartialOrd + std::ops::Sub<Output = T> + Copy + FromPrimitive,
{
    // Default tolerance as 1e-8 converted to type T
    let tol = match tol {
        Some(t) => t,
        None => match T::from_f64(1e-8) {
            Some(t) => t,
            None => panic!("Could not convert 1e-8 to generic type"),
        },
    };

    if point1.len() != point2.len() {
        return false;
    }

    for i in 0..point1.len() {
        if point1[i] > point2[i] && point1[i] - point2[i] > tol {
            return false;
        }
        if point2[i] > point1[i] && point2[i] - point1[i] > tol {
            return false;
        }
    }

    true
}

/// Compare arrays within a tolerance
///
/// Compares each element of the arrays to determine if they are
/// approximately equal within the specified tolerance.
///
/// # Arguments
///
/// * `array1` - First array
/// * `array2` - Second array
/// * `tol` - Tolerance (default: 1e-8)
///
/// # Returns
///
/// * True if arrays are equal within tolerance
///
/// # Examples
///
/// ```
/// use scirs2_core::utils::arrays_equal;
/// use ndarray::array;
///
/// let arr1 = array![[1.0, 2.0], [3.0, 4.0]];
/// let arr2 = array![[1.0, 2.0], [3.0, 4.0]];
/// let arr3 = array![[1.0, 2.0], [3.0, 4.001]];
///
/// assert!(arrays_equal(&arr1, &arr2, None));
/// assert!(!arrays_equal(&arr1, &arr3, None));
/// assert!(arrays_equal(&arr1, &arr3, Some(0.01)));
/// ```
pub fn arrays_equal<S1, S2, D, T>(
    array1: &ArrayBase<S1, D>,
    array2: &ArrayBase<S2, D>,
    tol: Option<T>,
) -> bool
where
    S1: Data<Elem = T>,
    S2: Data<Elem = T>,
    D: Dimension,
    T: PartialOrd + std::ops::Sub<Output = T> + Copy + FromPrimitive,
{
    if array1.shape() != array2.shape() {
        return false;
    }

    let points1: Vec<T> = array1.iter().copied().collect();
    let points2: Vec<T> = array2.iter().copied().collect();

    points_equal(&points1, &points2, tol)
}

/// Fills the diagonal of a matrix with a value
///
/// # Arguments
///
/// * `mut a` - Matrix to modify
/// * `val` - Value to set on the diagonal
///
/// # Returns
///
/// * The modified matrix
pub fn fill_diagonal<T: Clone>(mut a: Array2<T>, val: T) -> Array2<T> {
    let min_dim = a.nrows().min(a.ncols());

    for i in 0..min_dim {
        a[[i, i]] = val.clone();
    }

    a
}

/// Computes the product of all elements in an iterable
///
/// # Arguments
///
/// * `iter` - Iterable of values
///
/// # Returns
///
/// * Product of all elements
pub fn prod<I, T>(iter: I) -> T
where
    I: IntoIterator<Item = T>,
    T: std::ops::Mul<Output = T> + From<u8>,
{
    iter.into_iter().fold(T::from(1), |a, b| a * b)
}

/// Creates a range of values with a specified step size
///
/// # Arguments
///
/// * `start` - Start value (inclusive)
/// * `stop` - Stop value (exclusive)
/// * `step` - Step size
///
/// # Returns
///
/// * Vector of values
pub fn arange<F: Float + std::iter::Sum>(start: F, stop: F, step: F) -> Vec<F> {
    if step == F::zero() {
        panic!("Step size cannot be zero");
    }

    let mut result = Vec::new();
    let mut current = start;

    if step > F::zero() {
        while current < stop {
            result.push(current);
            current = current + step;
        }
    } else {
        while current > stop {
            result.push(current);
            current = current + step;
        }
    }

    result
}

/// Checks if all elements in an iterable satisfy a predicate
///
/// # Arguments
///
/// * `iter` - Iterable of values
/// * `predicate` - Function to check each value
///
/// # Returns
///
/// * `true` if all elements satisfy the predicate, `false` otherwise
pub fn all<I, T, F>(iter: I, predicate: F) -> bool
where
    I: IntoIterator<Item = T>,
    F: Fn(T) -> bool,
{
    iter.into_iter().all(predicate)
}

/// Checks if any element in an iterable satisfies a predicate
///
/// # Arguments
///
/// * `iter` - Iterable of values
/// * `predicate` - Function to check each value
///
/// # Returns
///
/// * `true` if any element satisfies the predicate, `false` otherwise
pub fn any<I, T, F>(iter: I, predicate: F) -> bool
where
    I: IntoIterator<Item = T>,
    F: Fn(T) -> bool,
{
    iter.into_iter().any(predicate)
}

/// Creates a linearly spaced array between start and end (inclusive)
///
/// This function uses parallel processing when available and
/// appropriate for better performance.
///
/// # Arguments
///
/// * `start` - Start value
/// * `end` - End value (inclusive)
/// * `num` - Number of points
///
/// # Returns
///
/// * Array of linearly spaced values
pub fn linspace<F: Float + std::iter::Sum>(start: F, end: F, num: usize) -> Array1<F> {
    if num < 2 {
        return Array::from_vec(vec![start]);
    }

    // Use parallel implementation for larger arrays
    #[cfg(feature = "parallel")]
    {
        if num >= 1000 {
            // TODO: Re-enable when par_linspace is implemented
            // use crate::parallel::par_linspace;

            // Use numeric conversion for f64 values
            let start_f64: Option<f64> = NumCast::from(start);
            let end_f64: Option<f64> = NumCast::from(end);

            if let (Some(_start_64), Some(_end_64)) = (start_f64, end_f64) {
                // TODO: Implement par_linspace
                // let result = par_linspace(start_64, end_64, num);
                // Convert back to F using safe numeric conversion
                // let mut result_f = Array1::zeros(result.raw_dim());
                // for (i, val) in result.iter().enumerate() {
                //     if let Some(val_f) = NumCast::from(*val) {
                //         result_f[i] = val_f;
                //     } else {
                //         // If conversion fails for any value, fall back to serial implementation
                //         break;
                //     }
                // }
                // return result_f;
            }
        }
    }

    // Fall back to standard implementation
    let step = (end - start) / F::from(num - 1).unwrap();
    let mut result = Vec::with_capacity(num);

    for i in 0..num {
        let value = start + step * F::from(i).unwrap();
        result.push(value);
    }

    // Make sure the last value is exactly end to avoid floating point precision issues
    if let Some(last) = result.last_mut() {
        *last = end;
    }

    Array::from_vec(result)
}

/// Creates a logarithmically spaced array between base^start and base^end (inclusive)
///
/// # Arguments
///
/// * `start` - Start exponent
/// * `end` - End exponent (inclusive)
/// * `num` - Number of points
/// * `base` - Base of the logarithm (default: 10.0)
///
/// # Returns
///
/// * Array of logarithmically spaced values
pub fn logspace<F: Float + std::iter::Sum>(
    start: F,
    end: F,
    num: usize,
    base: Option<F>,
) -> Array1<F> {
    let base = base.unwrap_or_else(|| F::from(10.0).unwrap());

    // Generate linearly spaced values in the exponent space
    let linear = linspace(start, end, num);

    // Convert to logarithmic space
    linear.mapv(|x| base.powf(x))
}

/// Compute the element-wise maximum of two arrays
///
/// This function uses parallel processing when available and
/// appropriate for the input arrays.
///
/// # Arguments
///
/// * `a` - First array
/// * `b` - Second array
///
/// # Returns
///
/// * Element-wise maximum
///
/// # Panics
///
/// * If the arrays have different shapes
pub fn maximum<S1, S2, D, T>(
    a: &ndarray::ArrayBase<S1, D>,
    b: &ndarray::ArrayBase<S2, D>,
) -> Array<T, D>
where
    S1: ndarray::Data<Elem = T>,
    S2: ndarray::Data<Elem = T>,
    D: Dimension,
    T: Num + PartialOrd + Copy + Send + Sync,
{
    assert_eq!(
        a.shape(),
        b.shape(),
        "Arrays must have the same shape for element-wise maximum"
    );

    // Use parallel implementation for larger arrays
    #[cfg(feature = "parallel")]
    {
        // TODO: Re-enable when par_maximum is implemented
        // use crate::parallel::par_maximum;

        // Check if array is large enough
        if a.len() > 1000 {
            // TODO: Implement par_maximum
            // return par_maximum(a, b);
        }
    }

    // Fall back to standard implementation
    let mut result = a.to_owned();
    for (i, elem) in result.iter_mut().enumerate() {
        if let Some(b_slice) = b.as_slice() {
            let b_val = b_slice[i];
            if b_val > *elem {
                *elem = b_val;
            }
        } else {
            // Handle case where b cannot be converted to slice
            let b_val = b.iter().nth(i).unwrap();
            if *b_val > *elem {
                *elem = *b_val;
            }
        }
    }

    result
}

/// Compute the element-wise minimum of two arrays
///
/// This function uses parallel processing when available and
/// appropriate for the input arrays.
///
/// # Arguments
///
/// * `a` - First array
/// * `b` - Second array
///
/// # Returns
///
/// * Element-wise minimum
///
/// # Panics
///
/// * If the arrays have different shapes
pub fn minimum<S1, S2, D, T>(
    a: &ndarray::ArrayBase<S1, D>,
    b: &ndarray::ArrayBase<S2, D>,
) -> Array<T, D>
where
    S1: ndarray::Data<Elem = T>,
    S2: ndarray::Data<Elem = T>,
    D: Dimension,
    T: Num + PartialOrd + Copy + Send + Sync,
{
    assert_eq!(
        a.shape(),
        b.shape(),
        "Arrays must have the same shape for element-wise minimum"
    );

    // Use parallel implementation for larger arrays
    #[cfg(feature = "parallel")]
    {
        // TODO: Re-enable when par_minimum is implemented
        // use crate::parallel::par_minimum;

        // Check if array is large enough
        if a.len() > 1000 {
            // TODO: Implement par_minimum
            // return par_minimum(a, b);
        }
    }

    // Fall back to standard implementation
    let mut result = a.to_owned();
    for (i, elem) in result.iter_mut().enumerate() {
        if let Some(b_slice) = b.as_slice() {
            let b_val = b_slice[i];
            if b_val < *elem {
                *elem = b_val;
            }
        } else {
            // Handle case where b cannot be converted to slice
            let b_val = b.iter().nth(i).unwrap();
            if *b_val < *elem {
                *elem = *b_val;
            }
        }
    }

    result
}

/// Normalize a vector to have unit energy or unit peak amplitude.
///
/// # Arguments
///
/// * `x` - Input vector
/// * `norm` - Normalization type: "energy", "peak", "sum", or "max"
///
/// # Returns
///
/// * Normalized vector as `Vec<f64>`
///
/// # Examples
///
/// ```
/// use scirs2_core::utils::normalize;
///
/// // Normalize a vector to unit energy
/// let signal = vec![1.0, 2.0, 3.0, 4.0];
/// let normalized = normalize(&signal, "energy").unwrap();
///
/// // Sum of squares should be 1.0
/// let sum_of_squares: f64 = normalized.iter().map(|&x| x * x).sum();
/// assert!((sum_of_squares - 1.0).abs() < 1e-10);
/// ```
pub fn normalize<T>(x: &[T], norm: &str) -> Result<Vec<f64>, &'static str>
where
    T: Float + NumCast + Debug,
{
    if x.is_empty() {
        return Err("Input signal is empty");
    }

    // Convert to f64 for internal processing
    let x_f64: Vec<f64> = x
        .iter()
        .map(|&val| NumCast::from(val).ok_or("Could not convert value to f64"))
        .collect::<Result<Vec<_>, _>>()?;

    // Normalize based on type
    match norm.to_lowercase().as_str() {
        "energy" => {
            // Normalize to unit energy (sum of squares = 1.0)
            let sum_of_squares: f64 = x_f64.iter().map(|&x| x * x).sum();

            if sum_of_squares.abs() < f64::EPSILON {
                return Err("Signal has zero energy, cannot normalize");
            }

            let scale = 1.0 / sum_of_squares.sqrt();
            let normalized = x_f64.iter().map(|&x| x * scale).collect();

            Ok(normalized)
        }
        "peak" => {
            // Normalize to unit peak amplitude (max absolute value = 1.0)
            let peak = x_f64.iter().fold(0.0, |a, &b| a.max(b.abs()));

            if peak.abs() < f64::EPSILON {
                return Err("Signal has zero peak, cannot normalize");
            }

            let scale = 1.0 / peak;
            let normalized = x_f64.iter().map(|&x| x * scale).collect();

            Ok(normalized)
        }
        "sum" => {
            // Normalize to unit sum
            let sum: f64 = x_f64.iter().sum();

            if sum.abs() < f64::EPSILON {
                return Err("Signal has zero sum, cannot normalize");
            }

            let scale = 1.0 / sum;
            let normalized = x_f64.iter().map(|&x| x * scale).collect();

            Ok(normalized)
        }
        "max" => {
            // Normalize to max value = 1.0 (preserves sign)
            let max_val = x_f64.iter().fold(0.0, |a, &b| a.max(b.abs()));

            if max_val.abs() < f64::EPSILON {
                return Err("Signal has zero maximum, cannot normalize");
            }

            let scale = 1.0 / max_val;
            let normalized = x_f64.iter().map(|&x| x * scale).collect();

            Ok(normalized)
        }
        _ => Err("Unknown normalization type. Supported types: 'energy', 'peak', 'sum', 'max'"),
    }
}

/// Pad an array with values according to the specified mode.
///
/// # Arguments
///
/// * `input` - Input array
/// * `pad_width` - Width of padding in each dimension (before, after)
/// * `mode` - Padding mode: "constant", "edge", "linear_ramp", "maximum", "mean", "median", "minimum", "reflect", "symmetric", "wrap"
/// * `constant_value` - Value to use for constant padding (only used for "constant" mode)
///
/// # Returns
///
/// * Padded array
///
/// # Examples
///
/// ```
/// // Example with a 1D array
/// use scirs2_core::utils::pad_array;
/// use ndarray::{Array1, array};
///
/// let arr = array![1.0, 2.0, 3.0];
/// let padded = pad_array(&arr, &[(1, 2)], "constant", Some(0.0)).unwrap();
/// assert_eq!(padded.shape(), &[6]);
/// assert_eq!(padded, array![0.0, 1.0, 2.0, 3.0, 0.0, 0.0]);
/// ```
pub fn pad_array<T, D>(
    input: &Array<T, D>,
    pad_width: &[(usize, usize)],
    mode: &str,
    constant_value: Option<T>,
) -> Result<Array<T, D>, String>
where
    T: Float + FromPrimitive + Debug + Clone,
    D: Dimension,
{
    // Validate inputs
    if input.ndim() == 0 {
        return Err("Input array cannot be 0-dimensional".to_string());
    }

    if pad_width.len() != input.ndim() {
        return Err(format!(
            "Pad width must have same length as input dimensions (got {} expected {})",
            pad_width.len(),
            input.ndim()
        ));
    }

    // No padding needed - return copy of input
    if pad_width.iter().all(|&(a, b)| a == 0 && b == 0) {
        return Ok(input.to_owned());
    }

    // Calculate new shape
    let mut new_shape = Vec::with_capacity(input.ndim());
    for (dim, &(pad_before, pad_after)) in pad_width.iter().enumerate().take(input.ndim()) {
        new_shape.push(input.shape()[dim] + pad_before + pad_after);
    }

    // Create output array with default constant value
    let const_val = constant_value.unwrap_or_else(|| T::zero());
    let mut output = Array::<T, D>::from_elem(
        D::from_dimension(&ndarray::IxDyn(&new_shape))
            .expect("Could not create dimension from shape"),
        const_val,
    );

    // For 1D arrays
    if input.ndim() == 1 {
        // Convert to Array1 for easier manipulation
        let input_array1 = input
            .view()
            .into_dimensionality::<ndarray::Ix1>()
            .map_err(|_| "Failed to convert to 1D array".to_string())?;
        let mut output_array1 = output
            .view_mut()
            .into_dimensionality::<ndarray::Ix1>()
            .map_err(|_| "Failed to convert output to 1D array".to_string())?;

        let input_len = input_array1.len();
        let start = pad_width[0].0;

        // First copy the input to the center region
        for i in 0..input_len {
            output_array1[start + i] = input_array1[i];
        }

        // Then pad the borders based on the mode
        match mode.to_lowercase().as_str() {
            "constant" => {
                // Already filled with constant value
            }
            "edge" => {
                // Pad left side with first value
                for i in 0..pad_width[0].0 {
                    output_array1[i] = input_array1[0];
                }
                // Pad right side with last value
                let offset = start + input_len;
                for i in 0..pad_width[0].1 {
                    output_array1[offset + i] = input_array1[input_len - 1];
                }
            }
            "reflect" => {
                // Pad left side
                for i in 0..pad_width[0].0 {
                    let src_idx = pad_width[0].0 - i;
                    if src_idx < input_len {
                        output_array1[i] = input_array1[src_idx];
                    }
                }
                // Pad right side
                let offset = start + input_len;
                for i in 0..pad_width[0].1 {
                    let src_idx = input_len - 2 - i;
                    if src_idx < input_len {
                        output_array1[offset + i] = input_array1[src_idx];
                    }
                }
            }
            "wrap" => {
                // Pad left side
                for i in 0..pad_width[0].0 {
                    let src_idx = (input_len - (pad_width[0].0 - i) % input_len) % input_len;
                    output_array1[i] = input_array1[src_idx];
                }
                // Pad right side
                let offset = start + input_len;
                for i in 0..pad_width[0].1 {
                    let src_idx = i % input_len;
                    output_array1[offset + i] = input_array1[src_idx];
                }
            }
            "maximum" => {
                // Find maximum value
                let max_val = input_array1
                    .iter()
                    .fold(T::neg_infinity(), |a, &b| a.max(b));

                // Pad with maximum value
                for i in 0..pad_width[0].0 {
                    output_array1[i] = max_val;
                }
                let offset = start + input_len;
                for i in 0..pad_width[0].1 {
                    output_array1[offset + i] = max_val;
                }
            }
            "minimum" => {
                // Find minimum value
                let min_val = input_array1.iter().fold(T::infinity(), |a, &b| a.min(b));

                // Pad with minimum value
                for i in 0..pad_width[0].0 {
                    output_array1[i] = min_val;
                }
                let offset = start + input_len;
                for i in 0..pad_width[0].1 {
                    output_array1[offset + i] = min_val;
                }
            }
            "mean" => {
                // Calculate mean value
                let sum = input_array1.iter().fold(T::zero(), |a, &b| a + b);
                let mean_val = sum / T::from_usize(input_len).unwrap();

                // Pad with mean value
                for i in 0..pad_width[0].0 {
                    output_array1[i] = mean_val;
                }
                let offset = start + input_len;
                for i in 0..pad_width[0].1 {
                    output_array1[offset + i] = mean_val;
                }
            }
            _ => return Err(format!("Unsupported padding mode: {}", mode)),
        }

        return Ok(output);
    }

    // For 2D arrays, we could add specific implementation similar to the above
    // For now, we'll just return the output with constant padding for 2D and higher

    if mode.to_lowercase() != "constant" {
        return Err(format!(
            "Padding mode '{}' is not yet implemented for arrays with more than 1 dimension",
            mode
        ));
    }

    // For higher dimensions, we'll just return a more simplified implementation with
    // constant padding only for now, and a note that it needs more work

    // We've already created the padded array with constant values,
    // now just return it since other padding modes for higher dimensions
    // would require more complex implementation
    //
    // NOTE: This is a placeholder implementation that needs to be improved
    // in the future to support all padding modes for higher dimensions

    Ok(output)
}

/// Create window functions of various types.
///
/// # Arguments
///
/// * `window_type` - Type of window function ("hamming", "hanning", "blackman", etc.)
/// * `length` - Length of the window
/// * `periodic` - Whether the window should be periodic (default: false)
///
/// # Returns
///
/// * Window function values as a vector
pub fn get_window(window_type: &str, length: usize, periodic: bool) -> Result<Vec<f64>, String> {
    if length == 0 {
        return Err("Window length must be positive".to_string());
    }

    let mut window = Vec::with_capacity(length);

    // Adjust length for periodic case
    let n = if periodic { length + 1 } else { length };

    // Generate window based on type
    match window_type.to_lowercase().as_str() {
        "hamming" => {
            // Hamming window: 0.54 - 0.46 * cos(2πn/(N-1))
            for i in 0..length {
                let w =
                    0.54 - 0.46 * (2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64).cos();
                window.push(w);
            }
        }
        "hanning" | "hann" => {
            // Hann window: 0.5 * (1 - cos(2πn/(N-1)))
            for i in 0..length {
                let w =
                    0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64).cos());
                window.push(w);
            }
        }
        "blackman" => {
            // Blackman window: 0.42 - 0.5 * cos(2πn/(N-1)) + 0.08 * cos(4πn/(N-1))
            for i in 0..length {
                let w = 0.42 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64).cos()
                    + 0.08 * (4.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64).cos();
                window.push(w);
            }
        }
        "bartlett" => {
            // Bartlett window (triangular window)
            let m = (n - 1) as f64 / 2.0;
            for i in 0..length {
                let w = 1.0 - ((i as f64 - m) / m).abs();
                window.push(w);
            }
        }
        "boxcar" | "rectangular" => {
            // Rectangular window (all ones)
            window.extend(std::iter::repeat_n(1.0, length));
        }
        "triang" => {
            // Triangular window (slightly different from Bartlett)
            let m = (length - 1) as f64 / 2.0;
            for i in 0..length {
                let w = 1.0 - ((i as f64 - m) / (m + 1.0)).abs();
                window.push(w);
            }
        }
        _ => {
            return Err(format!("Unknown window type: {}", window_type));
        }
    }

    Ok(window)
}

/// Differentiate a function using central difference method.
///
/// # Arguments
///
/// * `x` - Point at which to evaluate the derivative
/// * `h` - Step size for the finite difference
/// * `eval_fn` - Function that evaluates the function at a point
///
/// # Returns
///
/// * Derivative of the function at x
///
/// # Examples
///
/// ```
/// use scirs2_core::utils::differentiate;
///
/// // Differentiate f(x) = x^2 at x = 3
/// let f = |x: f64| -> Result<f64, String> { Ok(x * x) };
/// let derivative = differentiate(3.0, 0.001, f).unwrap();
///
/// // The exact derivative is 2x = 6 at x = 3
/// assert!((derivative - 6.0).abs() < 1e-5);
/// ```
pub fn differentiate<F, Func>(x: F, h: F, eval_fn: Func) -> Result<F, String>
where
    F: Float + FromPrimitive + Debug,
    Func: Fn(F) -> Result<F, String>,
{
    // Use central difference for better accuracy
    let f_plus = eval_fn(x + h).map_err(|e| format!("Error evaluating function at x+h: {}", e))?;
    let f_minus = eval_fn(x - h).map_err(|e| format!("Error evaluating function at x-h: {}", e))?;
    let derivative = (f_plus - f_minus) / (F::from(2.0).unwrap() * h);
    Ok(derivative)
}

/// Integrate a function using composite Simpson's rule.
///
/// # Arguments
///
/// * `a` - Lower bound of integration
/// * `b` - Upper bound of integration
/// * `n` - Number of intervals for the quadrature (must be even)
/// * `eval_fn` - Function that evaluates the function at a point
///
/// # Returns
///
/// * Definite integral of the function from a to b
///
/// # Examples
///
/// ```
/// use scirs2_core::utils::integrate;
///
/// // Integrate f(x) = x^2 from 0 to 1
/// let f = |x: f64| -> Result<f64, String> { Ok(x * x) };
/// let integral = integrate(0.0, 1.0, 100, f).unwrap();
///
/// // The exact integral is x^3/3 = 1/3 from 0 to 1
/// assert!((integral - 1.0/3.0).abs() < 1e-5);
/// ```
pub fn integrate<F, Func>(a: F, b: F, n: usize, eval_fn: Func) -> Result<F, String>
where
    F: Float + FromPrimitive + Debug,
    Func: Fn(F) -> Result<F, String>,
{
    if a > b {
        return integrate(b, a, n, eval_fn).map(|result| -result);
    }

    // Use composite Simpson's rule for integration
    if n < 2 {
        return Err("number of intervals must be at least 2".to_string());
    }

    if n % 2 != 0 {
        return Err("number of intervals must be even".to_string());
    }

    let h = (b - a) / F::from_usize(n).unwrap();
    let mut sum = eval_fn(a).map_err(|e| format!("Error evaluating function at a: {}", e))?
        + eval_fn(b).map_err(|e| format!("Error evaluating function at b: {}", e))?;

    // Even-indexed points (except endpoints)
    for i in 1..n {
        if i % 2 == 0 {
            let x_i = a + F::from_usize(i).unwrap() * h;
            sum = sum
                + F::from(2.0).unwrap()
                    * eval_fn(x_i)
                        .map_err(|e| format!("Error evaluating function at x_{}: {}", i, e))?;
        }
    }

    // Odd-indexed points
    for i in 1..n {
        if i % 2 == 1 {
            let x_i = a + F::from_usize(i).unwrap() * h;
            sum = sum
                + F::from(4.0).unwrap()
                    * eval_fn(x_i)
                        .map_err(|e| format!("Error evaluating function at x_{}: {}", i, e))?;
        }
    }

    let integral = h * sum / F::from(3.0).unwrap();
    Ok(integral)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::{arr1, array};

    #[test]
    fn test_is_close() {
        assert!(is_close(1.0, 1.0, 1e-10, 1e-10));
        assert!(is_close(1.0, 1.0 + 1e-11, 1e-10, 1e-10));
        assert!(!is_close(1.0, 1.1, 1e-10, 1e-10));
        assert!(is_close(1e10, 1e10 + 1.0, 1e-10, 1e-9));
    }

    #[test]
    fn test_fill_diagonal() {
        let a = Array2::<f64>::zeros((3, 3));
        let a_diag = fill_diagonal(a, 5.0);

        assert_relative_eq!(a_diag[[0, 0]], 5.0);
        assert_relative_eq!(a_diag[[1, 1]], 5.0);
        assert_relative_eq!(a_diag[[2, 2]], 5.0);
        assert_relative_eq!(a_diag[[0, 1]], 0.0);
    }

    #[test]
    fn test_prod() {
        assert_eq!(prod(vec![1, 2, 3, 4]), 24);
        assert_eq!(prod(vec![2.0, 3.0, 4.0]), 24.0);
    }

    #[test]
    fn test_arange() {
        let result = arange(0.0, 5.0, 1.0);
        assert_eq!(result, vec![0.0, 1.0, 2.0, 3.0, 4.0]);

        let result = arange(5.0, 0.0, -1.0);
        assert_eq!(result, vec![5.0, 4.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    #[should_panic]
    fn test_arange_zero_step() {
        let _result = arange(0.0, 5.0, 0.0);
    }

    #[test]
    fn test_all() {
        assert!(all(vec![2, 4, 6, 8], |x| x % 2 == 0));
        assert!(!all(vec![2, 4, 5, 8], |x| x % 2 == 0));
    }

    #[test]
    fn test_any() {
        assert!(any(vec![1, 2, 3, 4], |x| x % 2 == 0));
        assert!(!any(vec![1, 3, 5, 7], |x| x % 2 == 0));
    }

    #[test]
    fn test_linspace() {
        let result = linspace(0.0, 1.0, 5);
        let expected = arr1(&[0.0, 0.25, 0.5, 0.75, 1.0]);
        assert_eq!(result.len(), 5);

        for (a, b) in result.iter().zip(expected.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-14);
        }

        // Test with single value
        let result = linspace(5.0, 5.0, 1);
        assert_eq!(result.len(), 1);
        assert_relative_eq!(result[0], 5.0);

        // Test endpoints
        let result = linspace(-10.0, 10.0, 5);
        assert_relative_eq!(result[0], -10.0);
        assert_relative_eq!(result[4], 10.0);
    }

    #[test]
    fn test_logspace() {
        let result = logspace(0.0, 3.0, 4, None);
        let expected = arr1(&[1.0, 10.0, 100.0, 1000.0]);
        assert_eq!(result.len(), 4);

        for (a, b) in result.iter().zip(expected.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-10);
        }

        // Test with custom base
        let result = logspace(0.0, 3.0, 4, Some(2.0));
        let expected = arr1(&[1.0, 2.0, 4.0, 8.0]);

        for (a, b) in result.iter().zip(expected.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_maximum() {
        let a = array![[1, 2], [3, 4]];
        let b = array![[5, 1], [7, 2]];

        let result = maximum(&a, &b);
        let expected = array![[5, 2], [7, 4]];

        assert_eq!(result, expected);
    }

    #[test]
    fn test_minimum() {
        let a = array![[1, 2], [3, 4]];
        let b = array![[5, 1], [7, 2]];

        let result = minimum(&a, &b);
        let expected = array![[1, 1], [3, 2]];

        assert_eq!(result, expected);
    }

    #[test]
    #[should_panic]
    fn test_maximum_different_shapes() {
        let a = array![[1, 2], [3, 4]];
        let b = array![[5, 1, 2], [7, 2, 3]];

        let _result = maximum(&a, &b);
    }

    #[test]
    fn test_points_equal() {
        let point1 = [1.0, 2.0, 3.0];
        let point2 = [1.0, 2.0, 3.0];
        let point3 = [1.0, 2.0, 3.001];
        let point4 = [1.0, 2.0, 4.0];

        assert!(points_equal(&point1, &point2, None));
        assert!(!points_equal(&point1, &point3, None));
        assert!(points_equal(&point1, &point3, Some(0.01)));
        assert!(!points_equal(&point1, &point4, Some(0.01)));
    }

    #[test]
    fn test_arrays_equal() {
        let arr1 = array![[1.0, 2.0], [3.0, 4.0]];
        let arr2 = array![[1.0, 2.0], [3.0, 4.0]];
        let arr3 = array![[1.0, 2.0], [3.0, 4.001]];

        assert!(arrays_equal(&arr1, &arr2, None));
        assert!(!arrays_equal(&arr1, &arr3, None));
        assert!(arrays_equal(&arr1, &arr3, Some(0.01)));
    }

    #[test]
    fn test_normalize() {
        // Test energy normalization
        let signal = vec![1.0, 2.0, 3.0, 4.0];
        let normalized = normalize(&signal, "energy").unwrap();

        // Sum of squares should be 1.0
        let sum_of_squares: f64 = normalized.iter().map(|&x| x * x).sum();
        assert_relative_eq!(sum_of_squares, 1.0, epsilon = 1e-10);

        // Test peak normalization
        let signal = vec![1.0, -2.0, 3.0, -4.0];
        let normalized = normalize(&signal, "peak").unwrap();

        // Max absolute value should be 1.0
        let peak = normalized.iter().fold(0.0, |a, &b| a.max(b.abs()));
        assert_relative_eq!(peak, 1.0, epsilon = 1e-10);
        assert_relative_eq!(normalized[3], -1.0, epsilon = 1e-10);

        // Test sum normalization
        let signal = vec![1.0, 2.0, 3.0, 4.0];
        let normalized = normalize(&signal, "sum").unwrap();

        // Sum should be 1.0
        let sum: f64 = normalized.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_pad_array() {
        // Test constant padding on 1D array
        let arr = array![1.0, 2.0, 3.0];
        let padded = pad_array(&arr, &[(1, 2)], "constant", Some(0.0)).unwrap();

        assert_eq!(padded.shape(), &[6]);
        assert_eq!(padded, array![0.0, 1.0, 2.0, 3.0, 0.0, 0.0]);

        // Test edge padding
        let arr = array![1.0, 2.0, 3.0];
        let padded = pad_array(&arr, &[(2, 2)], "edge", None).unwrap();

        assert_eq!(padded.shape(), &[7]);
        assert_eq!(padded, array![1.0, 1.0, 1.0, 2.0, 3.0, 3.0, 3.0]);

        // Test maximum padding
        let arr = array![1.0, 2.0, 3.0];
        let padded = pad_array(&arr, &[(1, 1)], "maximum", None).unwrap();

        assert_eq!(padded.shape(), &[5]);
        assert_eq!(padded, array![3.0, 1.0, 2.0, 3.0, 3.0]);
    }

    #[test]
    fn test_get_window() {
        // Test Hamming window
        let window = get_window("hamming", 5, false).unwrap();

        assert_eq!(window.len(), 5);
        assert!(window[0] > 0.0 && window[0] < 0.6); // First value around 0.54
        assert!(window[2] > 0.9); // Middle value close to 1.0

        // Test Hann window
        let window = get_window("hann", 5, false).unwrap();

        assert_eq!(window.len(), 5);
        assert_relative_eq!(window[0], 0.0, epsilon = 1e-10);
        assert!(window[2] > 0.9); // Middle value close to 1.0

        // Test rectangular window
        let window = get_window("rectangular", 5, false).unwrap();

        assert_eq!(window.len(), 5);
        assert!(window.iter().all(|&x| (x - 1.0).abs() < 1e-10));
    }

    #[test]
    fn test_differentiate_integrate() {
        // Test differentiation of x^2
        let f = |x: f64| -> Result<f64, String> { Ok(x * x) };

        let derivative = differentiate(3.0, 0.001, f).unwrap();
        assert_relative_eq!(derivative, 6.0, epsilon = 1e-3); // f'(x) = 2x => f'(3) = 6

        // Test integration of x^2 from 0 to 1
        let integral = integrate(0.0, 1.0, 100, f).unwrap();
        assert_relative_eq!(integral, 1.0 / 3.0, epsilon = 1e-5); // ∫x^2 dx = x^3/3 => [0,1] = 1/3

        // Test integration of x^2 from 0 to 2
        let integral = integrate(0.0, 2.0, 100, f).unwrap();
        assert_relative_eq!(integral, 8.0 / 3.0, epsilon = 1e-5); // ∫x^2 dx = x^3/3 => [0,2] = 8/3
    }
}
