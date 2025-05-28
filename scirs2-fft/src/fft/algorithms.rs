/*!
 * FFT algorithm implementations
 *
 * This module provides implementations of the Fast Fourier Transform (FFT)
 * and its inverse (IFFT) in 1D, 2D, and N-dimensional cases.
 */

use crate::error::{FFTError, FFTResult};
use ndarray::{Array2, ArrayD, Axis, IxDyn};
use num_complex::Complex64;
use num_traits::NumCast;
use rustfft::{num_complex::Complex as RustComplex, FftPlanner};
use std::fmt::Debug;

// We're using the serial implementation even with parallel feature enabled,
// since we're not using parallelism at this level

/// Normalization mode for FFT operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormMode {
    /// No normalization (default for forward transforms)
    None,
    /// Normalize by 1/n (default for inverse transforms)
    Backward,
    /// Normalize by 1/sqrt(n) (unitary transform)
    Ortho,
    /// Normalize by 1/n for both forward and inverse transforms
    Forward,
}

impl From<&str> for NormMode {
    fn from(s: &str) -> Self {
        match s {
            "backward" => NormMode::Backward,
            "ortho" => NormMode::Ortho,
            "forward" => NormMode::Forward,
            _ => NormMode::None,
        }
    }
}

/// Convert a normalization mode string to NormMode enum
pub fn parse_norm_mode(norm: Option<&str>, is_inverse: bool) -> NormMode {
    match norm {
        Some(s) => NormMode::from(s),
        None if is_inverse => NormMode::Backward, // Default for inverse transforms
        None => NormMode::None,                   // Default for forward transforms
    }
}

/// Apply normalization to FFT results based on the specified mode
fn apply_normalization(data: &mut [Complex64], n: usize, mode: NormMode) {
    match mode {
        NormMode::None => {} // No normalization
        NormMode::Backward => {
            let scale = 1.0 / (n as f64);
            data.iter_mut().for_each(|c| *c *= scale);
        }
        NormMode::Ortho => {
            let scale = 1.0 / (n as f64).sqrt();
            data.iter_mut().for_each(|c| *c *= scale);
        }
        NormMode::Forward => {
            let scale = 1.0 / (n as f64);
            data.iter_mut().for_each(|c| *c *= scale);
        }
    }
}

/// Convert a single value to Complex64
fn convert_to_complex<T>(val: T) -> FFTResult<Complex64>
where
    T: NumCast + Copy + Debug + 'static,
{
    // First try to cast directly to f64 (for real numbers)
    if let Some(real) = num_traits::cast::<T, f64>(val) {
        return Ok(Complex64::new(real, 0.0));
    }

    // If direct casting fails, check if it's already a Complex64
    use std::any::Any;
    if let Some(complex) = (&val as &dyn Any).downcast_ref::<Complex64>() {
        return Ok(*complex);
    }

    // Try to handle f32 complex numbers
    if let Some(complex32) = (&val as &dyn Any).downcast_ref::<num_complex::Complex<f32>>() {
        return Ok(Complex64::new(complex32.re as f64, complex32.im as f64));
    }

    Err(FFTError::ValueError(format!(
        "Could not convert {val:?} to numeric type"
    )))
}

/// Convert input data to complex values
fn to_complex<T>(input: &[T]) -> FFTResult<Vec<Complex64>>
where
    T: NumCast + Copy + Debug + 'static,
{
    input.iter().map(|&val| convert_to_complex(val)).collect()
}

/// Compute the 1-dimensional Fast Fourier Transform
///
/// # Arguments
///
/// * `input` - Input data array
/// * `n` - Length of the output (optional)
///
/// # Returns
///
/// A vector of complex values representing the FFT result
///
/// # Examples
///
/// ```
/// use scirs2_fft::fft;
/// use num_complex::Complex64;
///
/// // Generate a simple signal
/// let signal = vec![1.0, 2.0, 3.0, 4.0];
///
/// // Compute the FFT
/// let spectrum = fft(&signal, None).unwrap();
///
/// // The DC component should be the sum of the input
/// assert!((spectrum[0].re - 10.0).abs() < 1e-10);
/// assert!(spectrum[0].im.abs() < 1e-10);
/// ```
pub fn fft<T>(input: &[T], n: Option<usize>) -> FFTResult<Vec<Complex64>>
where
    T: NumCast + Copy + Debug + 'static,
{
    // Input validation
    if input.is_empty() {
        return Err(FFTError::ValueError("Input cannot be empty".to_string()));
    }

    // Determine the FFT size (n or next power of 2 if n is None)
    let input_len = input.len();
    let fft_size = n.unwrap_or_else(|| input_len.next_power_of_two());

    // Convert input to complex numbers
    let mut data = to_complex(input)?;

    // Pad or truncate data to match fft_size
    if fft_size != input_len {
        if fft_size > input_len {
            // Pad with zeros
            data.resize(fft_size, Complex64::new(0.0, 0.0));
        } else {
            // Truncate
            data.truncate(fft_size);
        }
    }

    // Use rustfft library for computation
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_size);

    // Convert to rustfft-compatible complex type
    let mut buffer: Vec<RustComplex<f64>> =
        data.iter().map(|c| RustComplex::new(c.re, c.im)).collect();

    // Perform FFT in-place
    fft.process(&mut buffer);

    // Convert back to our Complex64 type
    let result: Vec<Complex64> = buffer
        .into_iter()
        .map(|c| Complex64::new(c.re, c.im))
        .collect();

    Ok(result)
}

/// Compute the inverse 1-dimensional Fast Fourier Transform
///
/// # Arguments
///
/// * `input` - Input data array
/// * `n` - Length of the output (optional)
///
/// # Returns
///
/// A vector of complex values representing the inverse FFT result
///
/// # Examples
///
/// ```
/// use scirs2_fft::{fft, ifft};
/// use num_complex::Complex64;
///
/// // Generate a simple signal
/// let signal = vec![1.0, 2.0, 3.0, 4.0];
///
/// // Compute the FFT
/// let spectrum = fft(&signal, None).unwrap();
///
/// // Compute the inverse FFT
/// let reconstructed = ifft(&spectrum, None).unwrap();
///
/// // The reconstructed signal should match the original
/// for (i, val) in signal.iter().enumerate() {
///     assert!((*val - reconstructed[i].re).abs() < 1e-10);
///     assert!(reconstructed[i].im.abs() < 1e-10);
/// }
/// ```
pub fn ifft<T>(input: &[T], n: Option<usize>) -> FFTResult<Vec<Complex64>>
where
    T: NumCast + Copy + Debug + 'static,
{
    // Input validation
    if input.is_empty() {
        return Err(FFTError::ValueError("Input cannot be empty".to_string()));
    }

    // Determine the FFT size
    let input_len = input.len();
    let fft_size = n.unwrap_or_else(|| input_len.next_power_of_two());

    // Convert input to complex numbers
    let mut data = to_complex(input)?;

    // Pad or truncate data to match fft_size
    if fft_size != input_len {
        if fft_size > input_len {
            // Pad with zeros
            data.resize(fft_size, Complex64::new(0.0, 0.0));
        } else {
            // Truncate
            data.truncate(fft_size);
        }
    }

    // Create FFT planner and plan
    let mut planner = FftPlanner::new();
    let ifft = planner.plan_fft_inverse(fft_size);

    // Convert to rustfft-compatible complex type
    let mut buffer: Vec<RustComplex<f64>> =
        data.iter().map(|c| RustComplex::new(c.re, c.im)).collect();

    // Perform inverse FFT in-place
    ifft.process(&mut buffer);

    // Convert back to our Complex64 type with normalization
    let mut result: Vec<Complex64> = buffer
        .into_iter()
        .map(|c| Complex64::new(c.re, c.im))
        .collect();

    // Apply 1/N normalization (standard for IFFT)
    apply_normalization(&mut result, fft_size, NormMode::Backward);

    // Truncate if necessary to match the original input length
    if n.is_none() && fft_size > input_len {
        result.truncate(input_len);
    }

    Ok(result)
}

/// Compute the 2-dimensional Fast Fourier Transform
///
/// # Arguments
///
/// * `input` - Input 2D array
/// * `shape` - Shape of the output (optional)
/// * `axes` - Axes along which to compute the FFT (optional)
/// * `norm` - Normalization mode: "backward", "ortho", or "forward" (optional)
///
/// # Returns
///
/// A 2D array of complex values representing the FFT result
///
/// # Examples
///
/// ```
/// use scirs2_fft::fft2;
/// use ndarray::{array, Array2};
///
/// // Create a simple 2x2 array
/// let input = array![[1.0, 2.0], [3.0, 4.0]];
///
/// // Compute the 2D FFT
/// let result = fft2(&input, None, None, None).unwrap();
///
/// // The DC component should be the sum of all elements
/// assert!((result[[0, 0]].re - 10.0).abs() < 1e-10);
/// ```
pub fn fft2<T>(
    input: &Array2<T>,
    shape: Option<(usize, usize)>,
    axes: Option<(i32, i32)>,
    norm: Option<&str>,
) -> FFTResult<Array2<Complex64>>
where
    T: NumCast + Copy + Debug + 'static,
{
    // Get input array shape
    let input_shape = input.shape();

    // Determine output shape
    let output_shape = shape.unwrap_or((input_shape[0], input_shape[1]));

    // Determine axes to perform FFT on
    let axes = axes.unwrap_or((0, 1));

    // Validate axes
    if axes.0 < 0 || axes.0 > 1 || axes.1 < 0 || axes.1 > 1 || axes.0 == axes.1 {
        return Err(FFTError::ValueError("Invalid axes for 2D FFT".to_string()));
    }

    // Parse normalization mode
    let norm_mode = parse_norm_mode(norm, false);

    // Create the output array
    let mut output = Array2::<Complex64>::zeros(output_shape);

    // Convert input array to complex numbers
    let mut complex_input = Array2::<Complex64>::zeros((input_shape[0], input_shape[1]));
    for i in 0..input_shape[0] {
        for j in 0..input_shape[1] {
            let val = input[[i, j]];

            // Convert using the unified conversion function
            complex_input[[i, j]] = convert_to_complex(val)?;
        }
    }

    // Pad or truncate to match output shape if necessary
    let mut padded_input = if input_shape != [output_shape.0, output_shape.1] {
        let mut padded = Array2::<Complex64>::zeros((output_shape.0, output_shape.1));
        let copy_rows = std::cmp::min(input_shape[0], output_shape.0);
        let copy_cols = std::cmp::min(input_shape[1], output_shape.1);

        for i in 0..copy_rows {
            for j in 0..copy_cols {
                padded[[i, j]] = complex_input[[i, j]];
            }
        }
        padded
    } else {
        complex_input
    };

    // Create FFT planner
    let mut planner = FftPlanner::new();

    // Perform FFT along each row
    let row_fft = planner.plan_fft_forward(output_shape.1);
    for mut row in padded_input.rows_mut() {
        // Convert to rustfft compatible format
        let mut buffer: Vec<RustComplex<f64>> =
            row.iter().map(|&c| RustComplex::new(c.re, c.im)).collect();

        // Perform FFT
        row_fft.process(&mut buffer);

        // Update row with FFT result
        for (i, val) in buffer.iter().enumerate() {
            row[i] = Complex64::new(val.re, val.im);
        }
    }

    // Perform FFT along each column
    let col_fft = planner.plan_fft_forward(output_shape.0);
    for mut col in padded_input.columns_mut() {
        // Convert to rustfft compatible format
        let mut buffer: Vec<RustComplex<f64>> =
            col.iter().map(|&c| RustComplex::new(c.re, c.im)).collect();

        // Perform FFT
        col_fft.process(&mut buffer);

        // Update column with FFT result
        for (i, val) in buffer.iter().enumerate() {
            col[i] = Complex64::new(val.re, val.im);
        }
    }

    // Apply normalization if needed
    if norm_mode != NormMode::None {
        let total_elements = output_shape.0 * output_shape.1;
        let scale = match norm_mode {
            NormMode::Backward => 1.0 / (total_elements as f64),
            NormMode::Ortho => 1.0 / (total_elements as f64).sqrt(),
            NormMode::Forward => 1.0 / (total_elements as f64),
            NormMode::None => 1.0, // Should not happen due to earlier check
        };

        padded_input.mapv_inplace(|x| x * scale);
    }

    // Copy result to output
    output.assign(&padded_input);

    Ok(output)
}

/// Compute the inverse 2-dimensional Fast Fourier Transform
///
/// # Arguments
///
/// * `input` - Input 2D array
/// * `shape` - Shape of the output (optional)
/// * `axes` - Axes along which to compute the inverse FFT (optional)
/// * `norm` - Normalization mode: "backward", "ortho", or "forward" (optional)
///
/// # Returns
///
/// A 2D array of complex values representing the inverse FFT result
///
/// # Examples
///
/// ```
/// use scirs2_fft::{fft2, ifft2};
/// use ndarray::{array, Array2};
///
/// // Create a simple 2x2 array
/// let input = array![[1.0, 2.0], [3.0, 4.0]];
///
/// // Compute the 2D FFT
/// let spectrum = fft2(&input, None, None, None).unwrap();
///
/// // Compute the inverse 2D FFT
/// let reconstructed = ifft2(&spectrum, None, None, None).unwrap();
///
/// // The reconstructed signal should match the original
/// for i in 0..2 {
///     for j in 0..2 {
///         assert!((input[[i, j]] - reconstructed[[i, j]].re).abs() < 1e-10);
///         assert!(reconstructed[[i, j]].im.abs() < 1e-10);
///     }
/// }
/// ```
pub fn ifft2<T>(
    input: &Array2<T>,
    shape: Option<(usize, usize)>,
    axes: Option<(i32, i32)>,
    norm: Option<&str>,
) -> FFTResult<Array2<Complex64>>
where
    T: NumCast + Copy + Debug + 'static,
{
    // Get input array shape
    let input_shape = input.shape();

    // Determine output shape
    let output_shape = shape.unwrap_or((input_shape[0], input_shape[1]));

    // Determine axes to perform FFT on
    let axes = axes.unwrap_or((0, 1));

    // Validate axes
    if axes.0 < 0 || axes.0 > 1 || axes.1 < 0 || axes.1 > 1 || axes.0 == axes.1 {
        return Err(FFTError::ValueError("Invalid axes for 2D IFFT".to_string()));
    }

    // Parse normalization mode (default is "backward" for inverse FFT)
    let norm_mode = parse_norm_mode(norm, true);

    // Convert input to complex and copy to output shape
    let mut complex_input = Array2::<Complex64>::zeros((input_shape[0], input_shape[1]));
    for i in 0..input_shape[0] {
        for j in 0..input_shape[1] {
            let val = input[[i, j]];

            // Convert using the unified conversion function
            complex_input[[i, j]] = convert_to_complex(val)?;
        }
    }

    // Pad or truncate to match output shape if necessary
    let mut padded_input = if input_shape != [output_shape.0, output_shape.1] {
        let mut padded = Array2::<Complex64>::zeros((output_shape.0, output_shape.1));
        let copy_rows = std::cmp::min(input_shape[0], output_shape.0);
        let copy_cols = std::cmp::min(input_shape[1], output_shape.1);

        for i in 0..copy_rows {
            for j in 0..copy_cols {
                padded[[i, j]] = complex_input[[i, j]];
            }
        }
        padded
    } else {
        complex_input
    };

    // Create FFT planner
    let mut planner = FftPlanner::new();

    // Perform inverse FFT along each row
    let row_ifft = planner.plan_fft_inverse(output_shape.1);
    for mut row in padded_input.rows_mut() {
        // Convert to rustfft compatible format
        let mut buffer: Vec<RustComplex<f64>> =
            row.iter().map(|&c| RustComplex::new(c.re, c.im)).collect();

        // Perform inverse FFT
        row_ifft.process(&mut buffer);

        // Update row with IFFT result
        for (i, val) in buffer.iter().enumerate() {
            row[i] = Complex64::new(val.re, val.im);
        }
    }

    // Perform inverse FFT along each column
    let col_ifft = planner.plan_fft_inverse(output_shape.0);
    for mut col in padded_input.columns_mut() {
        // Convert to rustfft compatible format
        let mut buffer: Vec<RustComplex<f64>> =
            col.iter().map(|&c| RustComplex::new(c.re, c.im)).collect();

        // Perform inverse FFT
        col_ifft.process(&mut buffer);

        // Update column with IFFT result
        for (i, val) in buffer.iter().enumerate() {
            col[i] = Complex64::new(val.re, val.im);
        }
    }

    // Apply appropriate normalization
    let total_elements = output_shape.0 * output_shape.1;
    let scale = match norm_mode {
        NormMode::Backward => 1.0 / (total_elements as f64),
        NormMode::Ortho => 1.0 / (total_elements as f64).sqrt(),
        NormMode::Forward => 1.0, // No additional normalization for forward mode in IFFT
        NormMode::None => 1.0,    // No normalization
    };

    if scale != 1.0 {
        padded_input.mapv_inplace(|x| x * scale);
    }

    Ok(padded_input)
}

/// Compute the N-dimensional Fast Fourier Transform
///
/// # Arguments
///
/// * `input` - Input N-dimensional array
/// * `shape` - Shape of the output (optional)
/// * `axes` - Axes along which to compute the FFT (optional)
/// * `norm` - Normalization mode: "backward", "ortho", or "forward" (optional)
/// * `overwrite_x` - Whether to overwrite the input array (optional)
/// * `workers` - Number of worker threads to use (optional)
///
/// # Returns
///
/// An N-dimensional array of complex values representing the FFT result
///
/// # Examples
///
/// ```
/// use scirs2_fft::fftn;
/// use ndarray::{Array, IxDyn};
///
/// // Create a 3D array
/// let mut data = Array::zeros(IxDyn(&[2, 2, 2]));
/// data[[0, 0, 0]] = 1.0;
/// data[[1, 1, 1]] = 1.0;
///
/// // Compute the N-dimensional FFT
/// let result = fftn(&data, None, None, None, None, None).unwrap();
///
/// // Check dimensions
/// assert_eq!(result.shape(), &[2, 2, 2]);
/// ```
#[allow(clippy::too_many_arguments)]
pub fn fftn<T>(
    input: &ArrayD<T>,
    shape: Option<Vec<usize>>,
    axes: Option<Vec<usize>>,
    norm: Option<&str>,
    _overwrite_x: Option<bool>,
    _workers: Option<usize>,
) -> FFTResult<ArrayD<Complex64>>
where
    T: NumCast + Copy + Debug + 'static,
{
    let input_shape = input.shape().to_vec();
    let input_ndim = input_shape.len();

    // Determine output shape
    let output_shape = shape.unwrap_or_else(|| input_shape.clone());

    // Validate output shape
    if output_shape.len() != input_ndim {
        return Err(FFTError::ValueError(
            "Output shape must have the same number of dimensions as input".to_string(),
        ));
    }

    // Determine axes to perform FFT on
    let axes = axes.unwrap_or_else(|| (0..input_ndim).collect());

    // Validate axes
    for &axis in &axes {
        if axis >= input_ndim {
            return Err(FFTError::ValueError(format!(
                "Axis {} out of bounds for array of dimension {}",
                axis, input_ndim
            )));
        }
    }

    // Parse normalization mode
    let norm_mode = parse_norm_mode(norm, false);

    // Convert input array to complex
    let mut complex_input = ArrayD::<Complex64>::zeros(IxDyn(&input_shape));
    for (idx, &val) in input.iter().enumerate() {
        let mut idx_vec = Vec::with_capacity(input_ndim);
        let mut remaining = idx;

        for &dim in input.shape().iter().rev() {
            idx_vec.push(remaining % dim);
            remaining /= dim;
        }

        idx_vec.reverse();

        complex_input[IxDyn(&idx_vec)] = convert_to_complex(val)?;
    }

    // Pad or truncate to match output shape if necessary
    let mut result = if input_shape != output_shape {
        let mut padded = ArrayD::<Complex64>::zeros(IxDyn(&output_shape));

        // Copy all elements that fit within both arrays
        for (idx, &val) in complex_input.iter().enumerate() {
            let mut idx_vec = Vec::with_capacity(input_ndim);
            let mut remaining = idx;

            for &dim in input.shape().iter().rev() {
                idx_vec.push(remaining % dim);
                remaining /= dim;
            }

            idx_vec.reverse();

            let mut in_bounds = true;
            for (dim, &idx_val) in idx_vec.iter().enumerate() {
                if idx_val >= output_shape[dim] {
                    in_bounds = false;
                    break;
                }
            }

            if in_bounds {
                padded[IxDyn(&idx_vec)] = val;
            }
        }

        padded
    } else {
        complex_input
    };

    // Create FFT planner
    let mut planner = FftPlanner::new();

    // Perform FFT along each axis
    for &axis in &axes {
        let axis_len = output_shape[axis];
        let fft = planner.plan_fft_forward(axis_len);

        // For each slice along the current axis
        let axis = Axis(axis);

        for mut lane in result.lanes_mut(axis) {
            // Convert to rustfft compatible format
            let mut buffer: Vec<RustComplex<f64>> =
                lane.iter().map(|&c| RustComplex::new(c.re, c.im)).collect();

            // Perform FFT
            fft.process(&mut buffer);

            // Update lane with FFT result
            for (i, val) in buffer.iter().enumerate() {
                lane[i] = Complex64::new(val.re, val.im);
            }
        }
    }

    // Apply normalization if needed
    if norm_mode != NormMode::None {
        let total_elements: usize = output_shape.iter().product();
        let scale = match norm_mode {
            NormMode::Backward => 1.0 / (total_elements as f64),
            NormMode::Ortho => 1.0 / (total_elements as f64).sqrt(),
            NormMode::Forward => 1.0 / (total_elements as f64),
            NormMode::None => 1.0, // Should not happen due to earlier check
        };

        result.mapv_inplace(|x| x * scale);
    }

    Ok(result)
}

/// Compute the inverse N-dimensional Fast Fourier Transform
///
/// # Arguments
///
/// * `input` - Input N-dimensional array
/// * `shape` - Shape of the output (optional)
/// * `axes` - Axes along which to compute the inverse FFT (optional)
/// * `norm` - Normalization mode: "backward", "ortho", or "forward" (optional)
/// * `overwrite_x` - Whether to overwrite the input array (optional)
/// * `workers` - Number of worker threads to use (optional)
///
/// # Returns
///
/// An N-dimensional array of complex values representing the inverse FFT result
///
/// # Examples
///
/// ```
/// use scirs2_fft::{fftn, ifftn};
/// use ndarray::{Array, IxDyn};
/// use num_complex::Complex64;
///
/// // Create a 3D array
/// let mut data = Array::zeros(IxDyn(&[2, 2, 2]));
/// data[[0, 0, 0]] = 1.0;
/// data[[1, 1, 1]] = 1.0;
///
/// // Compute the N-dimensional FFT
/// let spectrum = fftn(&data, None, None, None, None, None).unwrap();
///
/// // Compute the inverse N-dimensional FFT
/// let result = ifftn(&spectrum, None, None, None, None, None).unwrap();
///
/// // Check if the original data is recovered
/// for i in 0..2 {
///     for j in 0..2 {
///         for k in 0..2 {
///             let expected = if (i == 0 && j == 0 && k == 0) || (i == 1 && j == 1 && k == 1) {
///                 1.0
///             } else {
///                 0.0
///             };
///             assert!((result[[i, j, k]].re - expected).abs() < 1e-10);
///             assert!(result[[i, j, k]].im.abs() < 1e-10);
///         }
///     }
/// }
/// ```
#[allow(clippy::too_many_arguments)]
pub fn ifftn<T>(
    input: &ArrayD<T>,
    shape: Option<Vec<usize>>,
    axes: Option<Vec<usize>>,
    norm: Option<&str>,
    _overwrite_x: Option<bool>,
    _workers: Option<usize>,
) -> FFTResult<ArrayD<Complex64>>
where
    T: NumCast + Copy + Debug + 'static,
{
    let input_shape = input.shape().to_vec();
    let input_ndim = input_shape.len();

    // Determine output shape
    let output_shape = shape.unwrap_or_else(|| input_shape.clone());

    // Validate output shape
    if output_shape.len() != input_ndim {
        return Err(FFTError::ValueError(
            "Output shape must have the same number of dimensions as input".to_string(),
        ));
    }

    // Determine axes to perform FFT on
    let axes = axes.unwrap_or_else(|| (0..input_ndim).collect());

    // Validate axes
    for &axis in &axes {
        if axis >= input_ndim {
            return Err(FFTError::ValueError(format!(
                "Axis {} out of bounds for array of dimension {}",
                axis, input_ndim
            )));
        }
    }

    // Parse normalization mode (default is "backward" for inverse FFT)
    let norm_mode = parse_norm_mode(norm, true);

    // Create workspace array - convert input to complex first
    let mut complex_input = ArrayD::<Complex64>::zeros(IxDyn(&input_shape));
    for (idx, &val) in input.iter().enumerate() {
        let mut idx_vec = Vec::with_capacity(input_ndim);
        let mut remaining = idx;

        for &dim in input.shape().iter().rev() {
            idx_vec.push(remaining % dim);
            remaining /= dim;
        }

        idx_vec.reverse();

        // Try to convert to Complex64
        complex_input[IxDyn(&idx_vec)] = convert_to_complex(val)?;
    }

    // Now handle padding/resizing if needed
    let mut result = if input_shape != output_shape {
        let mut padded = ArrayD::<Complex64>::zeros(IxDyn(&output_shape));

        // Copy all elements that fit within both arrays
        for (idx, &val) in complex_input.iter().enumerate() {
            let mut idx_vec = Vec::with_capacity(input_ndim);
            let mut remaining = idx;

            for &dim in input.shape().iter().rev() {
                idx_vec.push(remaining % dim);
                remaining /= dim;
            }

            idx_vec.reverse();

            let mut in_bounds = true;
            for (dim, &idx_val) in idx_vec.iter().enumerate() {
                if idx_val >= output_shape[dim] {
                    in_bounds = false;
                    break;
                }
            }

            if in_bounds {
                padded[IxDyn(&idx_vec)] = val;
            }
        }

        padded
    } else {
        complex_input
    };

    // Create FFT planner
    let mut planner = FftPlanner::new();

    // Perform inverse FFT along each axis
    for &axis in &axes {
        let axis_len = output_shape[axis];
        let ifft = planner.plan_fft_inverse(axis_len);

        // For each slice along the current axis
        let axis = Axis(axis);

        for mut lane in result.lanes_mut(axis) {
            // Convert to rustfft compatible format
            let mut buffer: Vec<RustComplex<f64>> =
                lane.iter().map(|&c| RustComplex::new(c.re, c.im)).collect();

            // Perform inverse FFT
            ifft.process(&mut buffer);

            // Update lane with IFFT result
            for (i, val) in buffer.iter().enumerate() {
                lane[i] = Complex64::new(val.re, val.im);
            }
        }
    }

    // Apply appropriate normalization
    if norm_mode != NormMode::None {
        let total_elements: usize = axes.iter().map(|&a| output_shape[a]).product();
        let scale = match norm_mode {
            NormMode::Backward => 1.0 / (total_elements as f64),
            NormMode::Ortho => 1.0 / (total_elements as f64).sqrt(),
            NormMode::Forward => 1.0, // No additional normalization
            NormMode::None => 1.0,    // No normalization
        };

        if scale != 1.0 {
            result.mapv_inplace(|x| x * scale);
        }
    }

    Ok(result)
}
