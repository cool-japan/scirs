/*!
 * Parallel FFT algorithms implementations
 *
 * This module provides implementations of parallel Fast Fourier Transform (FFT)
 * algorithms for multi-threaded execution on multi-core CPUs.
 */

use crate::error::FFTResult;
use crate::fft::algorithms::{parse_norm_mode, NormMode};
use rustfft::{num_complex::Complex as RustComplex, FftPlanner};
use scirs2_core::ndarray::{Array2, Axis};
use scirs2_core::numeric::Complex64;
use scirs2_core::numeric::NumCast;

use scirs2_core::parallel_ops::*;

/// Compute a 2D FFT using parallel processing for rows and columns
///
/// # Arguments
///
/// * `input` - Input 2D array
/// * `shape` - Shape of the output (optional)
/// * `axes` - Axes along which to compute the FFT (optional)
/// * `norm` - Normalization mode (optional)
/// * `workers` - Number of worker threads to use (optional)
///
/// # Returns
///
/// A 2D array of complex values representing the parallel FFT result
///
/// # Examples
///
/// ```
/// use scirs2_fft::fft2_parallel;
/// use scirs2_core::ndarray::{array, Array2};
///
/// // Create a simple 2x2 array
/// let input = array![[1.0, 2.0], [3.0, 4.0]];
///
/// // Compute the 2D FFT in parallel
/// let result = fft2_parallel(&input, None, None, None, None).unwrap();
///
/// // The DC component should be the sum of all elements
/// assert!((result[[0, 0]].re - 10.0).abs() < 1e-10);
/// ```
#[cfg(feature = "parallel")]
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub fn fft2_parallel<T>(
    input: &Array2<T>,
    shape: Option<(usize, usize)>,
    axes: Option<(i32, i32)>,
    norm: Option<&str>,
    workers: Option<usize>,
) -> FFTResult<Array2<Complex64>>
where
    T: NumCast + Copy + std::fmt::Debug + 'static,
{
    // Get input array shape
    let inputshape = input.shape();

    // Determine output shape
    let outputshape = shape.unwrap_or((inputshape[0], inputshape[1]));

    // Determine axes to perform FFT on
    let axes = axes.unwrap_or((0, 1));

    // Validate axes
    if axes.0 < 0 || axes.0 > 1 || axes.1 < 0 || axes.1 > 1 || axes.0 == axes.1 {
        return Err(crate::FFTError::ValueError(
            "Invalid axes for 2D FFT".to_string(),
        ));
    }

    // Parse normalization mode
    let norm_mode = parse_norm_mode(norm, false);

    // Number of workers for parallel computation
    #[cfg(feature = "parallel")]
    let num_workers = workers.unwrap_or_else(|| num_threads().min(8));

    // Convert input array to complex numbers
    let mut complex_input = Array2::<Complex64>::zeros((inputshape[0], inputshape[1]));
    for i in 0..inputshape[0] {
        for j in 0..inputshape[1] {
            let val = input[[i, j]];

            // Try to convert to Complex64
            if let Some(c) = crate::fft::utility::try_as_complex(val) {
                complex_input[[i, j]] = c;
            } else {
                // Not a complex number, try to convert to f64 and make into a complex with zero imaginary part
                let real = NumCast::from(val).ok_or_else(|| {
                    crate::FFTError::ValueError(format!("Could not convert {val:?} to f64"))
                })?;
                complex_input[[i, j]] = Complex64::new(real, 0.0);
            }
        }
    }

    // Pad or truncate to match output shape if necessary
    let mut padded_input = if inputshape != [outputshape.0, outputshape.1] {
        let mut padded = Array2::<Complex64>::zeros((outputshape.0, outputshape.1));
        let copy_rows = std::cmp::min(inputshape[0], outputshape.0);
        let copy_cols = std::cmp::min(inputshape[1], outputshape.1);

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

    // Perform FFT along each row in parallel
    let row_fft = planner.plan_fft_forward(outputshape.1);

    if num_workers > 1 {
        padded_input
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .for_each(|mut row| {
                // Convert to rustfft compatible format
                let mut buffer: Vec<RustComplex<f64>> =
                    row.iter().map(|&c| RustComplex::new(c.re, c.im)).collect();

                // Perform FFT
                row_fft.process(&mut buffer);

                // Update row with FFT result
                for (i, val) in buffer.iter().enumerate() {
                    row[i] = Complex64::new(val.re, val.im);
                }
            });
    } else {
        // Fall back to sequential processing if only one worker
        for mut row in padded_input.rows_mut() {
            let mut buffer: Vec<RustComplex<f64>> =
                row.iter().map(|&c| RustComplex::new(c.re, c.im)).collect();

            row_fft.process(&mut buffer);

            for (i, val) in buffer.iter().enumerate() {
                row[i] = Complex64::new(val.re, val.im);
            }
        }
    }

    // Perform FFT along each column in parallel
    let col_fft = planner.plan_fft_forward(outputshape.0);

    if num_workers > 1 {
        padded_input
            .axis_iter_mut(Axis(1))
            .into_par_iter()
            .for_each(|mut col| {
                // Convert to rustfft compatible format
                let mut buffer: Vec<RustComplex<f64>> =
                    col.iter().map(|&c| RustComplex::new(c.re, c.im)).collect();

                // Perform FFT
                col_fft.process(&mut buffer);

                // Update column with FFT result
                for (i, val) in buffer.iter().enumerate() {
                    col[i] = Complex64::new(val.re, val.im);
                }
            });
    } else {
        // Fall back to sequential processing if only one worker
        for mut col in padded_input.columns_mut() {
            let mut buffer: Vec<RustComplex<f64>> =
                col.iter().map(|&c| RustComplex::new(c.re, c.im)).collect();

            col_fft.process(&mut buffer);

            for (i, val) in buffer.iter().enumerate() {
                col[i] = Complex64::new(val.re, val.im);
            }
        }
    }

    // Apply normalization if needed
    if norm_mode != NormMode::None {
        let total_elements = outputshape.0 * outputshape.1;
        let scale = match norm_mode {
            NormMode::Backward => 1.0 / (total_elements as f64),
            NormMode::Ortho => 1.0 / (total_elements as f64).sqrt(),
            NormMode::Forward => 1.0 / (total_elements as f64),
            NormMode::None => 1.0, // Should not happen due to earlier check
        };

        padded_input.mapv_inplace(|x| x * scale);
    }

    Ok(padded_input)
}

/// Non-parallel fallback implementation of fft2_parallel for when the parallel feature is disabled
#[cfg(not(feature = "parallel"))]
#[allow(dead_code)]
pub fn fft2_parallel<T>(
    input: &Array2<T>,
    shape: Option<(usize, usize)>,
    _axes: Option<(i32, i32)>,
    _norm: Option<&str>,
    _workers: Option<usize>,
) -> FFTResult<Array2<Complex64>>
where
    T: NumCast + Copy + std::fmt::Debug + 'static,
{
    // When parallel feature is disabled, just use the standard fft2 implementation
    crate::fft::algorithms::fft2(input, shape, None, None)
}

/// Compute the inverse 2D FFT using parallel processing
///
/// # Arguments
///
/// * `input` - Input 2D array of complex values
/// * `shape` - Shape of the output (optional)
/// * `axes` - Axes along which to compute the FFT (optional)
/// * `norm` - Normalization mode (optional)
/// * `workers` - Number of worker threads to use (optional)
///
/// # Returns
///
/// A 2D array of complex values representing the inverse FFT result
#[cfg(feature = "parallel")]
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub fn ifft2_parallel<T>(
    input: &Array2<T>,
    shape: Option<(usize, usize)>,
    axes: Option<(i32, i32)>,
    norm: Option<&str>,
    workers: Option<usize>,
) -> FFTResult<Array2<Complex64>>
where
    T: NumCast + Copy + std::fmt::Debug + 'static,
{
    // Get input array shape
    let inputshape = input.shape();

    // Determine output shape
    let outputshape = shape.unwrap_or((inputshape[0], inputshape[1]));

    // Determine axes to perform FFT on
    let axes = axes.unwrap_or((0, 1));

    // Validate axes
    if axes.0 < 0 || axes.0 > 1 || axes.1 < 0 || axes.1 > 1 || axes.0 == axes.1 {
        return Err(crate::FFTError::ValueError(
            "Invalid axes for 2D IFFT".to_string(),
        ));
    }

    // Parse normalization mode (default is "backward" for inverse FFT)
    let norm_mode = parse_norm_mode(norm, true);

    // Number of workers for parallel computation
    #[cfg(feature = "parallel")]
    let num_workers = workers.unwrap_or_else(|| num_threads().min(8));

    // Convert input to complex and copy to output shape
    let mut complex_input = Array2::<Complex64>::zeros((inputshape[0], inputshape[1]));
    for i in 0..inputshape[0] {
        for j in 0..inputshape[1] {
            let val = input[[i, j]];

            // Try to convert to Complex64
            if let Some(c) = crate::fft::utility::try_as_complex(val) {
                complex_input[[i, j]] = c;
            } else {
                // Not a complex number, try to convert to f64 and make into a complex with zero imaginary part
                let real = NumCast::from(val).ok_or_else(|| {
                    crate::FFTError::ValueError(format!("Could not convert {val:?} to f64"))
                })?;
                complex_input[[i, j]] = Complex64::new(real, 0.0);
            }
        }
    }

    // Pad or truncate to match output shape if necessary
    let mut padded_input = if inputshape != [outputshape.0, outputshape.1] {
        let mut padded = Array2::<Complex64>::zeros((outputshape.0, outputshape.1));
        let copy_rows = std::cmp::min(inputshape[0], outputshape.0);
        let copy_cols = std::cmp::min(inputshape[1], outputshape.1);

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

    // Perform inverse FFT along each row in parallel
    let row_ifft = planner.plan_fft_inverse(outputshape.1);

    if num_workers > 1 {
        padded_input
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .for_each(|mut row| {
                // Convert to rustfft compatible format
                let mut buffer: Vec<RustComplex<f64>> =
                    row.iter().map(|&c| RustComplex::new(c.re, c.im)).collect();

                // Perform inverse FFT
                row_ifft.process(&mut buffer);

                // Update row with IFFT result
                for (i, val) in buffer.iter().enumerate() {
                    row[i] = Complex64::new(val.re, val.im);
                }
            });
    } else {
        // Fall back to sequential processing if only one worker
        for mut row in padded_input.rows_mut() {
            let mut buffer: Vec<RustComplex<f64>> =
                row.iter().map(|&c| RustComplex::new(c.re, c.im)).collect();

            row_ifft.process(&mut buffer);

            for (i, val) in buffer.iter().enumerate() {
                row[i] = Complex64::new(val.re, val.im);
            }
        }
    }

    // Perform inverse FFT along each column in parallel
    let col_ifft = planner.plan_fft_inverse(outputshape.0);

    if num_workers > 1 {
        padded_input
            .axis_iter_mut(Axis(1))
            .into_par_iter()
            .for_each(|mut col| {
                // Convert to rustfft compatible format
                let mut buffer: Vec<RustComplex<f64>> =
                    col.iter().map(|&c| RustComplex::new(c.re, c.im)).collect();

                // Perform inverse FFT
                col_ifft.process(&mut buffer);

                // Update column with IFFT result
                for (i, val) in buffer.iter().enumerate() {
                    col[i] = Complex64::new(val.re, val.im);
                }
            });
    } else {
        // Fall back to sequential processing if only one worker
        for mut col in padded_input.columns_mut() {
            let mut buffer: Vec<RustComplex<f64>> =
                col.iter().map(|&c| RustComplex::new(c.re, c.im)).collect();

            col_ifft.process(&mut buffer);

            for (i, val) in buffer.iter().enumerate() {
                col[i] = Complex64::new(val.re, val.im);
            }
        }
    }

    // Apply appropriate normalization
    let total_elements = outputshape.0 * outputshape.1;
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

/// Non-parallel fallback implementation of ifft2_parallel for when the parallel feature is disabled
#[cfg(not(feature = "parallel"))]
#[allow(dead_code)]
pub fn ifft2_parallel<T>(
    input: &Array2<T>,
    shape: Option<(usize, usize)>,
    _axes: Option<(i32, i32)>,
    _norm: Option<&str>,
    _workers: Option<usize>,
) -> FFTResult<Array2<Complex64>>
where
    T: NumCast + Copy + std::fmt::Debug + 'static,
{
    // When parallel feature is disabled, just use the standard ifft2 implementation
    crate::fft::algorithms::ifft2(input, shape, None, None)
}
