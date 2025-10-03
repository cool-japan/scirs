//! Parallel FFT-based filtering operations
//!
//! This module provides parallel implementations of FFT-based filtering
//! for efficient frequency-domain processing of signals.

use crate::error::{SignalError, SignalResult};
use rustfft::{num_complex::Complex, FftPlanner};
use scirs2_core::parallel_ops::*;

/// Helper for parallel operations (temporary replacement)
fn par_iter_with_setup<I, IT, S, F, R, RF, E>(
    items: I,
    _setup: S,
    map_fn: F,
    reduce_fn: RF,
) -> Result<Vec<R>, E>
where
    I: IntoIterator<Item = IT>,
    IT: Copy,
    S: Fn(),
    F: Fn((), IT) -> Result<R, E>,
    RF: Fn(&mut Vec<R>, Result<R, E>) -> Result<(), E>,
    E: std::fmt::Debug,
{
    let mut results = Vec::new();
    for item in items {
        let result = map_fn((), item);
        reduce_fn(&mut results, result)?;
    }
    Ok(results)
}

/// Parallel frequency-domain filtering using FFT convolution
///
/// Efficient for long filters using overlap-add method with parallel FFTs.
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `impulse_response` - Filter impulse response
/// * `chunk_size` - FFT chunk size (should be power of 2)
///
/// # Returns
///
/// * Filtered signal
pub fn parallel_fft_filter(
    signal: &[f64],
    impulse_response: &[f64],
    chunk_size: Option<usize>,
) -> SignalResult<Vec<f64>> {
    let fft_size = chunk_size.unwrap_or(4096);
    let ir_len = impulse_response.len();

    if fft_size < ir_len {
        return Err(SignalError::ValueError(
            "FFT size too small for impulse response length".to_string(),
        ));
    }

    let useful_size = fft_size - ir_len + 1;

    // Prepare FFT planner
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_size);
    let ifft = planner.plan_fft_inverse(fft_size);

    // Zero-pad and FFT the impulse response
    let mut ir_padded: Vec<Complex<f64>> = impulse_response
        .iter()
        .map(|&x| Complex::new(x, 0.0))
        .collect();
    ir_padded.resize(fft_size, Complex::new(0.0, 0.0));

    let mut ir_fft = ir_padded.clone();
    fft.process(&mut ir_fft);

    // Calculate number of chunks needed
    let n_chunks = signal.len().div_ceil(useful_size);

    // Process chunks in parallel
    let chunk_results: Vec<Vec<f64>> = par_iter_with_setup(
        0..n_chunks,
        || {},
        |_, chunk_idx| {
            let start = chunk_idx * useful_size;
            let end = (start + useful_size).min(signal.len());

            // Prepare input chunk
            let mut chunk_data: Vec<Complex<f64>> =
                (start..end).map(|i| Complex::new(signal[i], 0.0)).collect();
            chunk_data.resize(fft_size, Complex::new(0.0, 0.0));

            // FFT of input chunk
            let mut chunk_fft = chunk_data;
            fft.process(&mut chunk_fft);

            // Frequency domain multiplication
            for i in 0..fft_size {
                chunk_fft[i] *= ir_fft[i];
            }

            // IFFT to get time domain result
            ifft.process(&mut chunk_fft);

            // Extract real part and normalize
            let chunk_result: Vec<f64> = chunk_fft.iter().map(|c| c.re / fft_size as f64).collect();

            Ok(chunk_result)
        },
        |results, chunk_result: Result<Vec<f64>, SignalError>| {
            results.push(chunk_result?);
            Ok(())
        },
    )?;

    // Overlap-add to combine results
    let output_len = signal.len() + ir_len - 1;
    let mut output = vec![0.0; output_len];

    for (chunk_idx, chunk_result) in chunk_results.iter().enumerate() {
        let start = chunk_idx * useful_size;
        for (i, &val) in chunk_result.iter().enumerate() {
            if start + i < output_len {
                output[start + i] += val;
            }
        }
    }

    // Trim to input signal length for "same" mode
    output.truncate(signal.len());
    Ok(output)
}

/// Parallel FFT-based filter design
///
/// Design a filter in frequency domain and convert to time domain.
///
/// # Arguments
///
/// * `frequency_response` - Desired frequency response (complex)
/// * `filter_length` - Length of resulting FIR filter
/// * `window_type` - Window function to apply ("rectangular", "hamming", "hanning", "blackman")
///
/// # Returns
///
/// * FIR filter coefficients
pub fn parallel_fft_filter_design(
    frequency_response: &[Complex<f64>],
    filter_length: usize,
    window_type: &str,
) -> SignalResult<Vec<f64>> {
    if frequency_response.is_empty() || filter_length == 0 {
        return Err(SignalError::ValueError(
            "Frequency response and filter length must be non-zero".to_string(),
        ));
    }

    let fft_size = frequency_response.len();

    // Prepare IFFT planner
    let mut planner = FftPlanner::new();
    let ifft = planner.plan_fft_inverse(fft_size);

    // Copy frequency response for IFFT
    let mut freq_data = frequency_response.to_vec();
    ifft.process(&mut freq_data);

    // Extract real part and normalize
    let mut impulse_response: Vec<f64> = freq_data
        .iter()
        .take(filter_length)
        .map(|c| c.re / fft_size as f64)
        .collect();

    // Apply window function
    apply_window(&mut impulse_response, window_type)?;

    Ok(impulse_response)
}

/// Parallel overlap-add FFT convolution
///
/// Performs convolution using overlap-add method with parallel FFT processing.
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `kernel` - Convolution kernel
/// * `fft_size` - Size of FFT blocks
///
/// # Returns
///
/// * Convolution result
pub fn parallel_overlap_add_convolution(
    signal: &[f64],
    kernel: &[f64],
    fft_size: usize,
) -> SignalResult<Vec<f64>> {
    let kernel_len = kernel.len();
    let useful_size = fft_size - kernel_len + 1;

    if fft_size < kernel_len {
        return Err(SignalError::ValueError(
            "FFT size must be at least as large as kernel length".to_string(),
        ));
    }

    // Prepare FFT planners
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_size);
    let ifft = planner.plan_fft_inverse(fft_size);

    // Zero-pad and FFT the kernel
    let mut kernel_padded: Vec<Complex<f64>> =
        kernel.iter().map(|&x| Complex::new(x, 0.0)).collect();
    kernel_padded.resize(fft_size, Complex::new(0.0, 0.0));

    let mut kernel_fft = kernel_padded;
    fft.process(&mut kernel_fft);

    // Calculate number of chunks
    let n_chunks = signal.len().div_ceil(useful_size);

    // Process chunks in parallel
    let chunk_results: Vec<Vec<f64>> = par_iter_with_setup(
        0..n_chunks,
        || {},
        |_, chunk_idx| {
            let start = chunk_idx * useful_size;
            let end = (start + useful_size).min(signal.len());

            // Prepare input chunk
            let mut chunk_data: Vec<Complex<f64>> =
                (start..end).map(|i| Complex::new(signal[i], 0.0)).collect();
            chunk_data.resize(fft_size, Complex::new(0.0, 0.0));

            // FFT of input chunk
            let mut chunk_fft = chunk_data;
            fft.process(&mut chunk_fft);

            // Frequency domain multiplication
            for i in 0..fft_size {
                chunk_fft[i] *= kernel_fft[i];
            }

            // IFFT to get result
            ifft.process(&mut chunk_fft);

            // Extract real part and normalize
            let chunk_result: Vec<f64> = chunk_fft.iter().map(|c| c.re / fft_size as f64).collect();

            Ok(chunk_result)
        },
        |results, chunk_result: Result<Vec<f64>, SignalError>| {
            results.push(chunk_result?);
            Ok(())
        },
    )?;

    // Overlap-add to combine results
    let output_len = signal.len() + kernel_len - 1;
    let mut output = vec![0.0; output_len];

    for (chunk_idx, chunk_result) in chunk_results.iter().enumerate() {
        let start = chunk_idx * useful_size;
        for (i, &val) in chunk_result.iter().enumerate() {
            if start + i < output_len {
                output[start + i] += val;
            }
        }
    }

    Ok(output)
}

/// Parallel overlap-save FFT convolution
///
/// Performs convolution using overlap-save method with parallel FFT processing.
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `kernel` - Convolution kernel
/// * `fft_size` - Size of FFT blocks
///
/// # Returns
///
/// * Convolution result
pub fn parallel_overlap_save_convolution(
    signal: &[f64],
    kernel: &[f64],
    fft_size: usize,
) -> SignalResult<Vec<f64>> {
    let kernel_len = kernel.len();
    let overlap_len = kernel_len - 1;
    let useful_size = fft_size - overlap_len;

    if fft_size < kernel_len {
        return Err(SignalError::ValueError(
            "FFT size must be at least as large as kernel length".to_string(),
        ));
    }

    // Prepare FFT planners
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_size);
    let ifft = planner.plan_fft_inverse(fft_size);

    // Zero-pad and FFT the kernel
    let mut kernel_padded: Vec<Complex<f64>> =
        kernel.iter().map(|&x| Complex::new(x, 0.0)).collect();
    kernel_padded.resize(fft_size, Complex::new(0.0, 0.0));

    let mut kernel_fft = kernel_padded;
    fft.process(&mut kernel_fft);

    // Calculate number of chunks
    let n_chunks = signal.len().div_ceil(useful_size);

    // Process chunks in parallel
    let chunk_results: Vec<Vec<f64>> = par_iter_with_setup(
        0..n_chunks,
        || {},
        |_, chunk_idx| {
            let start = chunk_idx * useful_size;
            let end = (start + fft_size).min(signal.len() + overlap_len);

            // Prepare input chunk with overlap
            let mut chunk_data = vec![Complex::new(0.0, 0.0); fft_size];
            for i in 0..fft_size {
                let signal_idx = start + i;
                if signal_idx < signal.len() && signal_idx >= chunk_idx * useful_size {
                    chunk_data[i] = Complex::new(signal[signal_idx], 0.0);
                } else if signal_idx < signal.len() {
                    chunk_data[i] = Complex::new(signal[signal_idx], 0.0);
                }
            }

            // FFT of input chunk
            let mut chunk_fft = chunk_data;
            fft.process(&mut chunk_fft);

            // Frequency domain multiplication
            for i in 0..fft_size {
                chunk_fft[i] *= kernel_fft[i];
            }

            // IFFT to get result
            ifft.process(&mut chunk_fft);

            // Extract useful part (discard overlap)
            let useful_start = if chunk_idx == 0 { 0 } else { overlap_len };
            let chunk_result: Vec<f64> = chunk_fft[useful_start..]
                .iter()
                .take(useful_size)
                .map(|c| c.re / fft_size as f64)
                .collect();

            Ok(chunk_result)
        },
        |results, chunk_result: Result<Vec<f64>, SignalError>| {
            results.push(chunk_result?);
            Ok(())
        },
    )?;

    // Concatenate results
    let mut output = Vec::with_capacity(signal.len());
    for chunk_result in chunk_results {
        output.extend(chunk_result);
    }

    // Trim to input signal length
    output.truncate(signal.len());
    Ok(output)
}

/// Apply window function to filter coefficients
fn apply_window(coefficients: &mut [f64], window_type: &str) -> SignalResult<()> {
    let n = coefficients.len();

    match window_type {
        "rectangular" => {
            // No modification needed
        }
        "hamming" => {
            for (i, coeff) in coefficients.iter_mut().enumerate() {
                let window_val =
                    0.54 - 0.46 * (2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64).cos();
                *coeff *= window_val;
            }
        }
        "hanning" => {
            for (i, coeff) in coefficients.iter_mut().enumerate() {
                let window_val =
                    0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64).cos());
                *coeff *= window_val;
            }
        }
        "blackman" => {
            for (i, coeff) in coefficients.iter_mut().enumerate() {
                let t = 2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64;
                let window_val = 0.42 - 0.5 * t.cos() + 0.08 * (2.0 * t).cos();
                *coeff *= window_val;
            }
        }
        _ => {
            return Err(SignalError::ValueError(format!(
                "Unknown window type: {}",
                window_type
            )));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_parallel_fft_filter() {
        let signal: Vec<f64> = (0..512)
            .map(|i| (2.0 * PI * i as f64 / 32.0).sin())
            .collect();

        let impulse_response = vec![0.25, 0.5, 0.25]; // Simple lowpass filter

        let result = parallel_fft_filter(&signal, &impulse_response, Some(256));
        assert!(result.is_ok());
        let filtered = result.unwrap();
        assert_eq!(filtered.len(), signal.len());
    }

    #[test]
    fn test_parallel_fft_filter_design() {
        let fft_size = 64;

        // Create a simple lowpass frequency response
        let mut frequency_response = vec![Complex::new(0.0, 0.0); fft_size];
        // Set lowpass response (keep low frequencies, zero high frequencies)
        for i in 0..fft_size / 8 {
            frequency_response[i] = Complex::new(1.0, 0.0);
            frequency_response[fft_size - i - 1] = Complex::new(1.0, 0.0);
        }

        let result = parallel_fft_filter_design(&frequency_response, 16, "hamming");
        assert!(result.is_ok());
        let coefficients = result.unwrap();
        assert_eq!(coefficients.len(), 16);
    }

    #[test]
    fn test_parallel_overlap_add_convolution() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0];
        let kernel = vec![0.25, 0.5, 0.25];

        let result = parallel_overlap_add_convolution(&signal, &kernel, 8);
        assert!(result.is_ok());
        let convolved = result.unwrap();
        assert_eq!(convolved.len(), signal.len() + kernel.len() - 1);
    }

    #[test]
    fn test_parallel_overlap_save_convolution() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let kernel = vec![0.5, 0.5];

        let result = parallel_overlap_save_convolution(&signal, &kernel, 4);
        assert!(result.is_ok());
        let convolved = result.unwrap();
        assert_eq!(convolved.len(), signal.len());
    }

    #[test]
    fn test_apply_window() {
        let mut coefficients = vec![1.0; 8];

        // Test Hamming window
        let result = apply_window(&mut coefficients, "hamming");
        assert!(result.is_ok());

        // Check that window was applied (values should not all be 1.0)
        let all_ones = coefficients.iter().all(|&x| (x - 1.0).abs() < 1e-10);
        assert!(!all_ones);

        // Test invalid window
        let mut coefficients = vec![1.0; 4];
        let result = apply_window(&mut coefficients, "invalid");
        assert!(result.is_err());
    }

    #[test]
    fn test_fft_filter_error_conditions() {
        let signal = vec![1.0, 2.0, 3.0];
        let impulse_response = vec![0.5, 0.5, 0.5, 0.5]; // Longer than FFT size

        // FFT size too small
        let result = parallel_fft_filter(&signal, &impulse_response, Some(2));
        assert!(result.is_err());
    }
}
