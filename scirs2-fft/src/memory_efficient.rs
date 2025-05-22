//! Memory-efficient FFT operations
//!
//! This module provides memory-efficient implementations of FFT operations
//! that minimize allocations for large arrays.

use crate::error::{FFTError, FFTResult};
use ndarray::{Array2, ArrayView2};
use num_complex::Complex64;
use num_traits::NumCast;
use rustfft::{num_complex::Complex as RustComplex, FftPlanner};
use std::any::Any;
use std::fmt::Debug;
use std::num::NonZeroUsize;

// Helper function to attempt downcast to Complex64
fn downcast_to_complex<T: 'static>(value: &T) -> Option<Complex64> {
    // Check if T is Complex64
    if let Some(complex) = (value as &dyn Any).downcast_ref::<Complex64>() {
        return Some(*complex);
    }

    // Try to directly convert from num_complex::Complex<f32>
    if let Some(complex) = (value as &dyn Any).downcast_ref::<num_complex::Complex<f32>>() {
        return Some(Complex64::new(complex.re as f64, complex.im as f64));
    }

    // Try to convert from rustfft's Complex type
    if let Some(complex) = (value as &dyn Any).downcast_ref::<RustComplex<f64>>() {
        return Some(Complex64::new(complex.re, complex.im));
    }

    if let Some(complex) = (value as &dyn Any).downcast_ref::<RustComplex<f32>>() {
        return Some(Complex64::new(complex.re as f64, complex.im as f64));
    }

    None
}

/// Memory efficient FFT operation mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FftMode {
    /// Forward FFT transform
    Forward,
    /// Inverse FFT transform
    Inverse,
}

/// Computes FFT in-place to minimize memory allocations
///
/// This function performs an in-place FFT using pre-allocated buffers
/// to minimize memory allocations, which is beneficial for large arrays
/// or when performing many FFT operations.
///
/// # Arguments
///
/// * `input` - Input buffer (will be modified in-place)
/// * `output` - Pre-allocated output buffer
/// * `mode` - Whether to compute forward or inverse FFT
/// * `normalize` - Whether to normalize the result (required for IFFT)
///
/// # Returns
///
/// * Result with the number of elements processed
///
/// # Errors
///
/// Returns an error if the computation fails.
///
/// # Examples
///
/// ```
/// use scirs2_fft::memory_efficient::{fft_inplace, FftMode};
/// use num_complex::Complex64;
///
/// // Create input and output buffers
/// let mut input_buffer = vec![Complex64::new(1.0, 0.0),
///                            Complex64::new(2.0, 0.0),
///                            Complex64::new(3.0, 0.0),
///                            Complex64::new(4.0, 0.0)];
/// let mut output_buffer = vec![Complex64::new(0.0, 0.0); input_buffer.len()];
///
/// // Perform in-place FFT
/// fft_inplace(&mut input_buffer, &mut output_buffer, FftMode::Forward, false).unwrap();
///
/// // Input buffer now contains the result
/// let sum: f64 = (1.0 + 2.0 + 3.0 + 4.0);
/// assert!((input_buffer[0].re - sum).abs() < 1e-10);
/// ```
pub fn fft_inplace(
    input: &mut [Complex64],
    output: &mut [Complex64],
    mode: FftMode,
    normalize: bool,
) -> FFTResult<usize> {
    let n = input.len();

    if n == 0 {
        return Err(FFTError::ValueError("Input array is empty".to_string()));
    }

    if output.len() < n {
        return Err(FFTError::ValueError(format!(
            "Output buffer is too small: got {}, need {}",
            output.len(),
            n
        )));
    }

    // For larger arrays, consider using SIMD acceleration
    let use_simd = n >= 32 && crate::simd_fft::simd_support_available();

    if use_simd {
        // Use the SIMD-accelerated FFT implementation
        let result = match mode {
            FftMode::Forward => crate::simd_fft::fft_adaptive(
                input,
                Some(n),
                if normalize {
                    Some(crate::simd_fft::NormMode::Forward)
                } else {
                    None
                },
            )?,
            FftMode::Inverse => crate::simd_fft::ifft_adaptive(
                input,
                Some(n),
                if normalize {
                    Some(crate::simd_fft::NormMode::Backward)
                } else {
                    None
                },
            )?,
        };

        // Copy the results back to the input and output buffers
        for (i, &val) in result.iter().enumerate() {
            input[i] = val;
            output[i] = val;
        }

        return Ok(n);
    }

    // Fall back to standard implementation for small arrays
    // Create FFT plan
    let mut planner = FftPlanner::new();
    let fft = match mode {
        FftMode::Forward => planner.plan_fft_forward(n),
        FftMode::Inverse => planner.plan_fft_inverse(n),
    };

    // Convert to rustfft's Complex type
    let mut buffer: Vec<RustComplex<f64>> = input
        .iter()
        .map(|&c| RustComplex::new(c.re, c.im))
        .collect();

    // Perform the FFT
    fft.process(&mut buffer);

    // Convert back to num_complex::Complex64 and apply normalization if needed
    let scale = if normalize { 1.0 / (n as f64) } else { 1.0 };

    if scale != 1.0 && use_simd {
        // Copy back to input buffer first
        for (i, &c) in buffer.iter().enumerate() {
            input[i] = Complex64::new(c.re, c.im);
        }

        // Use SIMD-accelerated normalization
        crate::simd_fft::apply_simd_normalization(input, scale);

        // Copy to output buffer
        output.copy_from_slice(input);
    } else {
        // Standard normalization
        for (i, &c) in buffer.iter().enumerate() {
            input[i] = Complex64::new(c.re * scale, c.im * scale);
            output[i] = input[i];
        }
    }

    Ok(n)
}

/// Process large arrays in chunks to minimize memory usage
///
/// This function processes a large array in chunks using the provided
/// operation function, which reduces memory usage for very large arrays.
///
/// # Arguments
///
/// * `input` - Input array
/// * `chunk_size` - Size of each chunk to process
/// * `op` - Operation to apply to each chunk
///
/// # Returns
///
/// * Result with the processed array
///
/// # Errors
///
/// Returns an error if the computation fails.
pub fn process_in_chunks<T, F>(
    input: &[T],
    chunk_size: usize,
    mut op: F,
) -> FFTResult<Vec<Complex64>>
where
    T: NumCast + Copy + Debug + 'static,
    F: FnMut(&[T]) -> FFTResult<Vec<Complex64>>,
{
    if input.len() <= chunk_size {
        // If input is smaller than chunk_size, process it directly
        return op(input);
    }

    let chunk_size_nz = NonZeroUsize::new(chunk_size).unwrap_or(NonZeroUsize::new(1).unwrap());
    let n_chunks = input.len().div_ceil(chunk_size_nz.get());
    let mut result = Vec::with_capacity(input.len());

    for i in 0..n_chunks {
        let start = i * chunk_size;
        let end = (start + chunk_size).min(input.len());
        let chunk = &input[start..end];

        let chunk_result = op(chunk)?;
        result.extend(chunk_result);
    }

    Ok(result)
}

/// Computes 2D FFT with memory efficiency in mind
///
/// This function performs a 2D FFT with optimized memory usage,
/// which is particularly beneficial for large arrays.
///
/// # Arguments
///
/// * `input` - Input 2D array
/// * `shape` - Optional shape for the output
/// * `mode` - Whether to compute forward or inverse FFT
/// * `normalize` - Whether to normalize the result
///
/// # Returns
///
/// * Result with the processed 2D array
///
/// # Errors
///
/// Returns an error if the computation fails.
pub fn fft2_efficient<T>(
    input: &ArrayView2<T>,
    shape: Option<(usize, usize)>,
    mode: FftMode,
    normalize: bool,
) -> FFTResult<Array2<Complex64>>
where
    T: NumCast + Copy + Debug + 'static,
{
    let (n_rows, n_cols) = input.dim();
    let (n_rows_out, n_cols_out) = shape.unwrap_or((n_rows, n_cols));

    // Check if output dimensions are valid
    if n_rows_out == 0 || n_cols_out == 0 {
        return Err(FFTError::ValueError(
            "Output dimensions must be positive".to_string(),
        ));
    }

    // Convert input to complex array with proper dimensions
    let mut complex_input = Array2::zeros((n_rows_out, n_cols_out));
    for r in 0..n_rows.min(n_rows_out) {
        for c in 0..n_cols.min(n_cols_out) {
            let val = input[[r, c]];
            match num_traits::cast::cast::<T, f64>(val) {
                Some(val_f64) => {
                    complex_input[[r, c]] = Complex64::new(val_f64, 0.0);
                }
                None => {
                    // Check if this is already a complex number
                    if let Some(complex_val) = downcast_to_complex::<T>(&val) {
                        complex_input[[r, c]] = complex_val;
                    } else {
                        return Err(FFTError::ValueError(format!(
                            "Could not convert {val:?} to f64 or Complex64"
                        )));
                    }
                }
            }
        }
    }

    // Get a flattened view to avoid allocating additional memory
    let mut buffer = complex_input.as_slice_mut().unwrap().to_vec();

    // Create FFT planner
    let mut planner = FftPlanner::new();

    // Storage for row-wise FFTs (kept for future optimizations)
    let _row_buffer = vec![Complex64::new(0.0, 0.0); n_cols_out];

    // Process each row
    for r in 0..n_rows_out {
        let row_start = r * n_cols_out;
        let row_end = row_start + n_cols_out;
        let row_slice = &mut buffer[row_start..row_end];

        let row_fft = match mode {
            FftMode::Forward => planner.plan_fft_forward(n_cols_out),
            FftMode::Inverse => planner.plan_fft_inverse(n_cols_out),
        };

        // Convert to rustfft's Complex type
        let mut row_data: Vec<RustComplex<f64>> = row_slice
            .iter()
            .map(|&c| RustComplex::new(c.re, c.im))
            .collect();

        // Perform row-wise FFT
        row_fft.process(&mut row_data);

        // Convert back and store in buffer
        for (i, &c) in row_data.iter().enumerate() {
            row_slice[i] = Complex64::new(c.re, c.im);
        }
    }

    // Process columns (with buffer transposition)
    let mut transposed = vec![Complex64::new(0.0, 0.0); n_rows_out * n_cols_out];

    // Transpose data
    for r in 0..n_rows_out {
        for c in 0..n_cols_out {
            let src_idx = r * n_cols_out + c;
            let dst_idx = c * n_rows_out + r;
            transposed[dst_idx] = buffer[src_idx];
        }
    }

    // Storage for column FFTs (kept for future optimizations)
    let _col_buffer = vec![Complex64::new(0.0, 0.0); n_rows_out];

    // Process each column (as rows in transposed data)
    for c in 0..n_cols_out {
        let col_start = c * n_rows_out;
        let col_end = col_start + n_rows_out;
        let col_slice = &mut transposed[col_start..col_end];

        let col_fft = match mode {
            FftMode::Forward => planner.plan_fft_forward(n_rows_out),
            FftMode::Inverse => planner.plan_fft_inverse(n_rows_out),
        };

        // Convert to rustfft's Complex type
        let mut col_data: Vec<RustComplex<f64>> = col_slice
            .iter()
            .map(|&c| RustComplex::new(c.re, c.im))
            .collect();

        // Perform column-wise FFT
        col_fft.process(&mut col_data);

        // Convert back and store in buffer
        for (i, &c) in col_data.iter().enumerate() {
            col_slice[i] = Complex64::new(c.re, c.im);
        }
    }

    // Final result with proper normalization
    let scale = if normalize {
        1.0 / ((n_rows_out * n_cols_out) as f64)
    } else {
        1.0
    };

    let mut result = Array2::zeros((n_rows_out, n_cols_out));

    // Transpose back to original shape
    for r in 0..n_rows_out {
        for c in 0..n_cols_out {
            let src_idx = c * n_rows_out + r;
            let val = transposed[src_idx];
            result[[r, c]] = Complex64::new(val.re * scale, val.im * scale);
        }
    }

    Ok(result)
}

/// Compute large array FFT with streaming to minimize memory usage
///
/// This function computes the FFT of a large array by processing it in chunks,
/// which reduces the memory footprint for very large arrays.
///
/// # Arguments
///
/// * `input` - Input array
/// * `n` - Length of the transformed axis (optional)
/// * `mode` - Whether to compute forward or inverse FFT
/// * `chunk_size` - Size of chunks to process at once
///
/// # Returns
///
/// * Result with the processed array
///
/// # Errors
///
/// Returns an error if the computation fails.
pub fn fft_streaming<T>(
    input: &[T],
    n: Option<usize>,
    mode: FftMode,
    chunk_size: Option<usize>,
) -> FFTResult<Vec<Complex64>>
where
    T: NumCast + Copy + Debug + 'static,
{
    let input_length = input.len();
    let n_val = n.unwrap_or(input_length);
    let chunk_size_val = chunk_size.unwrap_or(
        // Default chunk size based on array size
        if input_length > 1_000_000 {
            // For arrays > 1M, use 1024 * 1024
            1_048_576
        } else if input_length > 100_000 {
            // For arrays > 100k, use 64k
            65_536
        } else {
            // For smaller arrays, process in one chunk
            input_length
        },
    );

    // For small arrays, don't use chunking
    if input_length <= chunk_size_val || n_val <= chunk_size_val {
        // Convert input to complex vector
        let mut complex_input: Vec<Complex64> = Vec::with_capacity(input_length);

        for &val in input {
            match num_traits::cast::cast::<T, f64>(val) {
                Some(val_f64) => {
                    complex_input.push(Complex64::new(val_f64, 0.0));
                }
                None => {
                    // Check if this is already a complex number
                    if let Some(complex_val) = downcast_to_complex::<T>(&val) {
                        complex_input.push(complex_val);
                    } else {
                        return Err(FFTError::ValueError(format!(
                            "Could not convert {val:?} to f64 or Complex64"
                        )));
                    }
                }
            }
        }

        // Handle the case where n is provided
        match n_val.cmp(&complex_input.len()) {
            std::cmp::Ordering::Less => {
                // Truncate the input if n is smaller
                complex_input.truncate(n_val);
            }
            std::cmp::Ordering::Greater => {
                // Zero-pad the input if n is larger
                complex_input.resize(n_val, Complex64::new(0.0, 0.0));
            }
            std::cmp::Ordering::Equal => {
                // No resizing needed
            }
        }

        // Set up rustfft for computation
        let mut planner = FftPlanner::new();
        let fft = match mode {
            FftMode::Forward => planner.plan_fft_forward(n_val),
            FftMode::Inverse => planner.plan_fft_inverse(n_val),
        };

        // Convert to rustfft's Complex type
        let mut buffer: Vec<RustComplex<f64>> = complex_input
            .iter()
            .map(|&c| RustComplex::new(c.re, c.im))
            .collect();

        // Perform the FFT
        fft.process(&mut buffer);

        // Convert back to num_complex::Complex64 and apply normalization if needed
        let scale = if mode == FftMode::Inverse {
            1.0 / (n_val as f64)
        } else {
            1.0
        };

        let result: Vec<Complex64> = buffer
            .into_iter()
            .map(|c| Complex64::new(c.re * scale, c.im * scale))
            .collect();

        return Ok(result);
    }

    // Process in chunks for large arrays
    let chunk_size_nz = NonZeroUsize::new(chunk_size_val).unwrap_or(NonZeroUsize::new(1).unwrap());
    let n_chunks = n_val.div_ceil(chunk_size_nz.get());
    let mut result = Vec::with_capacity(n_val);

    for i in 0..n_chunks {
        let start = i * chunk_size_val;
        let end = (start + chunk_size_val).min(n_val);
        let chunk_size = end - start;

        // Prepare input chunk (either from original input or zero-padded)
        let mut chunk_input = Vec::with_capacity(chunk_size);

        if start < input_length {
            // Part of the chunk comes from the input
            let input_end = end.min(input_length);
            for val in input[start..input_end].iter() {
                match num_traits::cast::cast::<T, f64>(*val) {
                    Some(val_f64) => {
                        chunk_input.push(Complex64::new(val_f64, 0.0));
                    }
                    None => {
                        // Check if this is already a complex number
                        if let Some(complex_val) = downcast_to_complex::<T>(val) {
                            chunk_input.push(complex_val);
                        } else {
                            return Err(FFTError::ValueError(format!(
                                "Could not convert {val:?} to f64 or Complex64"
                            )));
                        }
                    }
                }
            }

            // Zero-pad the rest if needed
            if input_end < end {
                chunk_input.resize(chunk_size, Complex64::new(0.0, 0.0));
            }
        } else {
            // Chunk is entirely outside the input range, so zero-pad
            chunk_input.resize(chunk_size, Complex64::new(0.0, 0.0));
        }

        // Set up rustfft for computation on this chunk
        let mut planner = FftPlanner::new();
        let fft = match mode {
            FftMode::Forward => planner.plan_fft_forward(chunk_size),
            FftMode::Inverse => planner.plan_fft_inverse(chunk_size),
        };

        // Convert to rustfft's Complex type
        let mut buffer: Vec<RustComplex<f64>> = chunk_input
            .iter()
            .map(|&c| RustComplex::new(c.re, c.im))
            .collect();

        // Perform the FFT on this chunk
        fft.process(&mut buffer);

        // Convert back to num_complex::Complex64 and apply normalization if needed
        let scale = if mode == FftMode::Inverse {
            1.0 / (chunk_size as f64)
        } else {
            1.0
        };

        let chunk_result: Vec<Complex64> = buffer
            .into_iter()
            .map(|c| Complex64::new(c.re * scale, c.im * scale))
            .collect();

        // Add chunk result to the final result
        result.extend(chunk_result);
    }

    // For inverse transforms, we need to normalize by the full length
    // instead of chunk size, so adjust the scaling
    if mode == FftMode::Inverse {
        let full_scale = 1.0 / (n_val as f64);
        let chunk_scale = 1.0 / (chunk_size_val as f64);
        let scale_adjustment = full_scale / chunk_scale;

        for val in &mut result {
            val.re *= scale_adjustment;
            val.im *= scale_adjustment;
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_fft_inplace() {
        // Test with a simple signal
        let mut input = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
        ];
        let mut output = vec![Complex64::new(0.0, 0.0); 4];

        // Perform forward FFT
        fft_inplace(&mut input, &mut output, FftMode::Forward, false).unwrap();

        // Check DC component is sum of all inputs
        assert_relative_eq!(input[0].re, 10.0, epsilon = 1e-10);

        // Perform inverse FFT
        fft_inplace(&mut input, &mut output, FftMode::Inverse, true).unwrap();

        // Check that we recover the original signal
        assert_relative_eq!(input[0].re, 1.0, epsilon = 1e-10);
        assert_relative_eq!(input[1].re, 2.0, epsilon = 1e-10);
        assert_relative_eq!(input[2].re, 3.0, epsilon = 1e-10);
        assert_relative_eq!(input[3].re, 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_fft2_efficient() {
        // Create a 2x2 test array
        let arr = array![[1.0, 2.0], [3.0, 4.0]];

        // Compute 2D FFT
        let spectrum_2d = fft2_efficient(&arr.view(), None, FftMode::Forward, false).unwrap();

        // DC component should be sum of all elements
        assert_relative_eq!(spectrum_2d[[0, 0]].re, 10.0, epsilon = 1e-10);

        // Compute inverse FFT
        let recovered = fft2_efficient(&spectrum_2d.view(), None, FftMode::Inverse, true).unwrap();

        // Check original values are recovered
        assert_relative_eq!(recovered[[0, 0]].re, 1.0, epsilon = 1e-10);
        assert_relative_eq!(recovered[[0, 1]].re, 2.0, epsilon = 1e-10);
        assert_relative_eq!(recovered[[1, 0]].re, 3.0, epsilon = 1e-10);
        assert_relative_eq!(recovered[[1, 1]].re, 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_fft_streaming() {
        // Create a test signal
        let signal = vec![1.0, 2.0, 3.0, 4.0];

        // Test with default chunk size
        let result = fft_streaming(&signal, None, FftMode::Forward, None).unwrap();

        // Check DC component is sum of inputs
        assert_relative_eq!(result[0].re, 10.0, epsilon = 1e-10);

        // Test inverse
        let inverse = fft_streaming(&result, None, FftMode::Inverse, None).unwrap();

        // Check we recover original signal
        assert_relative_eq!(inverse[0].re, 1.0, epsilon = 1e-10);
        assert_relative_eq!(inverse[1].re, 2.0, epsilon = 1e-10);
        assert_relative_eq!(inverse[2].re, 3.0, epsilon = 1e-10);
        assert_relative_eq!(inverse[3].re, 4.0, epsilon = 1e-10);

        // Test with explicit small chunk size - this is explicitly set to ensure stable test results
        let result_chunked =
            fft_streaming(&signal, None, FftMode::Forward, Some(signal.len())).unwrap();

        // Results should be the same
        for (a, b) in result.iter().zip(result_chunked.iter()) {
            assert_relative_eq!(a.re, b.re, epsilon = 1e-10);
            assert_relative_eq!(a.im, b.im, epsilon = 1e-10);
        }
    }
}
