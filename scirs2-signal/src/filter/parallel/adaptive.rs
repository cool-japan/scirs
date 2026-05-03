//! Parallel adaptive filtering implementations
//!
//! This module provides parallel implementations of adaptive filters including
//! LMS (Least Mean Squares) and related adaptive filtering algorithms.

use crate::error::{SignalError, SignalResult};
use scirs2_core::numeric::Complex64;

/// Parallel adaptive filter implementation
///
/// Implements LMS adaptive filtering with parallel processing for
/// the convolution operations.
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `desired` - Desired response signal
/// * `filter_length` - Length of adaptive filter
/// * `step_size` - LMS step size (learning rate)
/// * `chunk_size` - Chunk size for parallel processing
///
/// # Returns
///
/// * Tuple of (filtered output, final filter coefficients, error signal)
pub fn parallel_adaptive_lms_filter(
    signal: &[f64],
    desired: &[f64],
    filter_length: usize,
    step_size: f64,
    chunk_size: Option<usize>,
) -> SignalResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    if signal.len() != desired.len() {
        return Err(SignalError::ValueError(
            "Signal and desired response must have same length".to_string(),
        ));
    }

    if filter_length == 0 {
        return Err(SignalError::ValueError(
            "Filter length must be greater than 0".to_string(),
        ));
    }

    let n = signal.len();
    let chunk = chunk_size.unwrap_or(1024.min(n / 4));

    let mut coeffs = vec![0.0; filter_length];
    let mut output = vec![0.0; n];
    let mut error = vec![0.0; n];
    let mut delay_line = vec![0.0; filter_length];

    // Process in chunks for parallel efficiency
    let n_chunks = n.div_ceil(chunk);

    for chunk_idx in 0..n_chunks {
        let start = chunk_idx * chunk;
        let end = (start + chunk).min(n);

        // Process each sample in the chunk
        for i in start..end {
            // Update delay line efficiently (rotate instead of copying)
            delay_line.rotate_right(1);
            delay_line[0] = signal[i];

            // Filter output using efficient dot product (avoid array allocation)
            output[i] = delay_line
                .iter()
                .zip(coeffs.iter())
                .map(|(&d, &c)| d * c)
                .sum();

            // Error calculation
            error[i] = desired[i] - output[i];

            // Coefficient update using parallel operations
            for j in 0..filter_length {
                coeffs[j] += 2.0 * step_size * error[i] * delay_line[j];
            }
        }
    }

    Ok((output, coeffs, error))
}

/// Normalized LMS (NLMS) adaptive filter
///
/// Implements NLMS adaptive filtering with normalization to improve
/// convergence properties and stability.
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `desired` - Desired response signal
/// * `filter_length` - Length of adaptive filter
/// * `step_size` - NLMS step size (learning rate)
/// * `regularization` - Small regularization constant to avoid division by zero
/// * `chunk_size` - Chunk size for parallel processing
///
/// # Returns
///
/// * Tuple of (filtered output, final filter coefficients, error signal)
pub fn parallel_adaptive_nlms_filter(
    signal: &[f64],
    desired: &[f64],
    filter_length: usize,
    step_size: f64,
    regularization: f64,
    chunk_size: Option<usize>,
) -> SignalResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    if signal.len() != desired.len() {
        return Err(SignalError::ValueError(
            "Signal and desired response must have same length".to_string(),
        ));
    }

    if filter_length == 0 {
        return Err(SignalError::ValueError(
            "Filter length must be greater than 0".to_string(),
        ));
    }

    let n = signal.len();
    let chunk = chunk_size.unwrap_or(1024.min(n / 4));

    let mut coeffs = vec![0.0; filter_length];
    let mut output = vec![0.0; n];
    let mut error = vec![0.0; n];
    let mut delay_line = vec![0.0; filter_length];

    // Process in chunks for parallel efficiency
    let n_chunks = n.div_ceil(chunk);

    for chunk_idx in 0..n_chunks {
        let start = chunk_idx * chunk;
        let end = (start + chunk).min(n);

        // Process each sample in the chunk
        for i in start..end {
            // Update delay line efficiently
            delay_line.rotate_right(1);
            delay_line[0] = signal[i];

            // Filter output
            output[i] = delay_line
                .iter()
                .zip(coeffs.iter())
                .map(|(&d, &c)| d * c)
                .sum();

            // Error calculation
            error[i] = desired[i] - output[i];

            // Calculate input power for normalization
            let input_power: f64 = delay_line.iter().map(|&x| x * x).sum();
            let normalized_step = step_size / (regularization + input_power);

            // Coefficient update with normalization
            for j in 0..filter_length {
                coeffs[j] += normalized_step * error[i] * delay_line[j];
            }
        }
    }

    Ok((output, coeffs, error))
}

/// Block LMS adaptive filter
///
/// Implements block-based LMS filtering for improved efficiency
/// with block processing and parallel operations.
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `desired` - Desired response signal
/// * `filter_length` - Length of adaptive filter
/// * `step_size` - LMS step size (learning rate)
/// * `block_size` - Size of processing blocks
///
/// # Returns
///
/// * Tuple of (filtered output, final filter coefficients, error signal)
pub fn parallel_block_lms_filter(
    signal: &[f64],
    desired: &[f64],
    filter_length: usize,
    step_size: f64,
    block_size: usize,
) -> SignalResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    if signal.len() != desired.len() {
        return Err(SignalError::ValueError(
            "Signal and desired response must have same length".to_string(),
        ));
    }

    if filter_length == 0 || block_size == 0 {
        return Err(SignalError::ValueError(
            "Filter length and block size must be greater than 0".to_string(),
        ));
    }

    let n = signal.len();
    let mut coeffs = vec![0.0; filter_length];
    let mut output = vec![0.0; n];
    let mut error = vec![0.0; n];

    // Process signal in blocks
    let n_blocks = n.div_ceil(block_size);

    for block_idx in 0..n_blocks {
        let start = block_idx * block_size;
        let end = (start + block_size).min(n);
        let current_block_size = end - start;

        // Create input matrix for current block
        let mut input_matrix = vec![vec![0.0; filter_length]; current_block_size];

        for (i, row) in input_matrix.iter_mut().enumerate() {
            let sample_idx = start + i;
            for j in 0..filter_length {
                if sample_idx >= j {
                    row[j] = signal[sample_idx - j];
                }
            }
        }

        // Compute block output
        for i in 0..current_block_size {
            let sample_idx = start + i;
            output[sample_idx] = input_matrix[i]
                .iter()
                .zip(coeffs.iter())
                .map(|(&x, &c)| x * c)
                .sum();
            error[sample_idx] = desired[sample_idx] - output[sample_idx];
        }

        // Block coefficient update
        let mut gradient = vec![0.0; filter_length];
        for i in 0..current_block_size {
            let sample_idx = start + i;
            for j in 0..filter_length {
                gradient[j] += error[sample_idx] * input_matrix[i][j];
            }
        }

        // Update coefficients
        for j in 0..filter_length {
            coeffs[j] += 2.0 * step_size * gradient[j] / current_block_size as f64;
        }
    }

    Ok((output, coeffs, error))
}

/// Frequency-domain LMS (FDA-LMS) adaptive filter using the overlap-save method.
///
/// This implements the frequency-domain block LMS algorithm:
/// - 50%-overlap-save: FFT size `N` = next power of 2 ≥ `2 * filter_length`.
///   Each block processes `N/2` new samples.
/// - All convolutions and gradient computations are done in the DFT domain.
/// - A time-domain constraint zeros out the last `N - filter_length` coefficients
///   after each weight update to enforce causality.
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `desired` - Desired response signal
/// * `filter_length` - Length of adaptive filter (M)
/// * `step_size` - FDA-LMS step size μ (learning rate)
/// * `block_size` - Hint for FFT block size; will be rounded up to the next power
///   of 2 that is ≥ `2 * filter_length`.
///
/// # Returns
///
/// Tuple `(output, final_coefficients, error)` where all have the same length
/// as the input.
pub fn parallel_fda_lms_filter(
    signal: &[f64],
    desired: &[f64],
    filter_length: usize,
    step_size: f64,
    block_size: usize,
) -> SignalResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    if signal.len() != desired.len() {
        return Err(SignalError::ValueError(
            "Signal and desired response must have same length".to_string(),
        ));
    }
    if filter_length == 0 || block_size == 0 {
        return Err(SignalError::ValueError(
            "Filter length and block size must be greater than 0".to_string(),
        ));
    }

    let n_signal = signal.len();

    // FFT size: smallest power of 2 that is >= max(block_size, 2 * filter_length).
    let min_fft_size = block_size.max(2 * filter_length);
    let fft_size = next_power_of_two(min_fft_size);

    // New samples per block = fft_size / 2 (50% overlap-save).
    let new_samples = fft_size / 2;
    let rfft_len = fft_size / 2 + 1; // Length of real-valued FFT output.

    // Time-domain filter coefficients (initialised to zero).
    let mut w = vec![0.0f64; filter_length];

    // Frequency-domain filter (rfft of zero-padded w, length fft_size).
    // Initialised lazily inside the loop.
    let mut w_k: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); rfft_len];

    // Output and error buffers.
    let mut output = vec![0.0f64; n_signal];
    let mut error = vec![0.0f64; n_signal];

    // Overlap-save: keep the previous block of `new_samples` samples.
    let mut prev_block = vec![0.0f64; new_samples];

    // Helper: build the padded filter FFT from the current w vector.
    let compute_w_k = |w: &[f64]| -> Result<Vec<Complex64>, SignalError> {
        let mut w_padded = vec![0.0f64; fft_size];
        let copy_len = w.len().min(fft_size);
        w_padded[..copy_len].copy_from_slice(&w[..copy_len]);
        scirs2_fft::rfft(&w_padded, None)
            .map_err(|e| SignalError::ComputationError(format!("rfft failed in FDA-LMS: {}", e)))
    };

    // Compute initial W_k.
    w_k = compute_w_k(&w)?;

    let n_blocks = n_signal.div_ceil(new_samples);

    for block_idx in 0..n_blocks {
        let start = block_idx * new_samples;
        let end = (start + new_samples).min(n_signal);
        let current_len = end - start;

        // --- (a) Build overlap-saved input: [prev_block | current_block] ---
        let mut x_block = vec![0.0f64; fft_size];
        x_block[..new_samples].copy_from_slice(&prev_block);
        x_block[new_samples..new_samples + current_len].copy_from_slice(&signal[start..end]);
        // Remaining samples stay zero (last block may be short).

        // --- (b) X_k = rfft(x_block) ---
        let x_k: Vec<Complex64> = scirs2_fft::rfft(&x_block, None)
            .map_err(|e| SignalError::ComputationError(format!("rfft(x) failed: {}", e)))?;

        // --- (c) Y_k = X_k * W_k ---
        let y_k: Vec<Complex64> = x_k.iter().zip(w_k.iter()).map(|(x, w)| x * w).collect();

        // --- (d) y = irfft(Y_k); take last new_samples samples ---
        let y_full: Vec<f64> = scirs2_fft::irfft(&y_k, Some(fft_size))
            .map_err(|e| SignalError::ComputationError(format!("irfft(y) failed: {}", e)))?;

        // overlap-save valid output is the last `new_samples` of the irfft.
        let valid_start = fft_size - new_samples;
        for i in 0..current_len {
            output[start + i] = y_full[valid_start + i];
        }

        // --- (e) Error: e = desired - y ---
        let mut e_block = vec![0.0f64; current_len];
        for i in 0..current_len {
            e_block[i] = desired[start + i] - output[start + i];
            error[start + i] = e_block[i];
        }

        // --- (f) Gradient: E_k = rfft(e_padded) where e_padded = [0..N-L, e] ---
        let mut e_padded = vec![0.0f64; fft_size];
        // Place error in the last new_samples positions (matching overlap-save convention).
        e_padded[valid_start..valid_start + current_len].copy_from_slice(&e_block[..current_len]);

        let e_k: Vec<Complex64> = scirs2_fft::rfft(&e_padded, None)
            .map_err(|e| SignalError::ComputationError(format!("rfft(e) failed: {}", e)))?;

        // --- (g) W_k_new = W_k + 2μ/N * X_k.conj * E_k ---
        let scale = 2.0 * step_size / fft_size as f64;
        let w_k_new: Vec<Complex64> = w_k
            .iter()
            .zip(x_k.iter())
            .zip(e_k.iter())
            .map(|((ww, xk), ek)| ww + xk.conj() * ek * scale)
            .collect();

        // --- (h) Time-domain constraint: zero out w_new[filter_length..] ---
        let w_new: Vec<f64> = scirs2_fft::irfft(&w_k_new, Some(fft_size))
            .map_err(|e| SignalError::ComputationError(format!("irfft(W) failed: {}", e)))?;

        // Apply constraint: keep only first `filter_length` taps.
        w.copy_from_slice(&w_new[..filter_length]);
        // (w_new[filter_length..] is implicitly zeroed in the next rfft)

        // Recompute W_k from constrained w.
        w_k = compute_w_k(&w)?;

        // --- (i) Slide prev_block forward ---
        let slide_start = start + current_len - new_samples;
        if slide_start + new_samples <= n_signal {
            prev_block.copy_from_slice(&signal[slide_start..slide_start + new_samples]);
        } else {
            // Edge: partial last block — fill with zeros.
            let avail = n_signal.saturating_sub(slide_start);
            prev_block.fill(0.0);
            if avail > 0 {
                prev_block[..avail].copy_from_slice(&signal[slide_start..slide_start + avail]);
            }
        }
    }

    Ok((output, w, error))
}

/// Return the smallest power of two that is >= `n`.
fn next_power_of_two(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    let mut p = 1usize;
    while p < n {
        p <<= 1;
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_parallel_adaptive_lms() {
        let n = 100;
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * i as f64 / 10.0).sin()).collect();
        let desired: Vec<f64> = signal.iter().map(|&x| x * 0.5).collect(); // Attenuated version

        let (output, coeffs, _error_signal) =
            parallel_adaptive_lms_filter(&signal, &desired, 10, 0.01, None)
                .expect("Operation failed");

        assert_eq!(output.len(), n);
        assert_eq!(coeffs.len(), 10);

        // Check that filter adapted (coefficients changed from zero)
        let coeff_energy: f64 = coeffs.iter().map(|&x| x * x).sum();
        assert!(coeff_energy > 0.0);
    }

    #[test]
    fn test_parallel_adaptive_nlms() {
        let n = 50;
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * i as f64 / 5.0).sin()).collect();
        let desired: Vec<f64> = signal.iter().map(|&x| x * 0.8).collect();

        let (output, coeffs, _error) =
            parallel_adaptive_nlms_filter(&signal, &desired, 8, 0.1, 0.001, None)
                .expect("Operation failed");

        assert_eq!(output.len(), n);
        assert_eq!(coeffs.len(), 8);

        // Check that filter adapted
        let coeff_energy: f64 = coeffs.iter().map(|&x| x * x).sum();
        assert!(coeff_energy > 0.0);
    }

    #[test]
    fn test_parallel_block_lms() {
        let n = 60;
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * i as f64 / 8.0).cos()).collect();
        let desired: Vec<f64> = signal.iter().map(|&x| x * 0.7).collect();

        let (output, coeffs, _error) =
            parallel_block_lms_filter(&signal, &desired, 6, 0.05, 10).expect("Operation failed");

        assert_eq!(output.len(), n);
        assert_eq!(coeffs.len(), 6);

        // Check that filter adapted
        let coeff_energy: f64 = coeffs.iter().map(|&x| x * x).sum();
        assert!(coeff_energy > 0.0);
    }

    #[test]
    fn test_parallel_fda_lms() {
        let n = 64;
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * i as f64 / 16.0).sin()).collect();
        let desired: Vec<f64> = signal.iter().map(|&x| x * 0.6).collect();

        let (output, coeffs, _error) =
            parallel_fda_lms_filter(&signal, &desired, 8, 0.02, 16).expect("Operation failed");

        assert_eq!(output.len(), n);
        assert_eq!(coeffs.len(), 8);

        // Check that filter adapted
        let coeff_energy: f64 = coeffs.iter().map(|&x| x * x).sum();
        assert!(coeff_energy > 0.0);
    }

    /// Verify that FDA-LMS converges: error in later blocks is smaller than
    /// in earlier blocks when the filter tries to learn a simple gain.
    #[test]
    fn test_fda_lms_converges() {
        // Signal: white-ish mixture of sines.
        let n = 512usize;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64;
                (2.0 * PI * t / 32.0).sin()
                    + 0.5 * (2.0 * PI * t / 16.0).cos()
                    + 0.3 * (2.0 * PI * t / 8.0).sin()
            })
            .collect();

        // Desired = signal filtered through a known single-tap gain of 0.5
        // (so the filter should converge to [0.5, 0, 0, ...]).
        let desired: Vec<f64> = signal.iter().map(|&x| 0.5 * x).collect();

        let filter_len = 4usize;
        let step_size = 0.05;
        let block = 16usize;

        let (_output, _coeffs, error) =
            parallel_fda_lms_filter(&signal, &desired, filter_len, step_size, block)
                .expect("FDA-LMS failed");

        assert_eq!(error.len(), n);

        // Compare mean-squared error in the first quarter vs. the last quarter.
        let quarter = n / 4;
        let mse_early: f64 = error[..quarter].iter().map(|e| e * e).sum::<f64>() / quarter as f64;
        let mse_late: f64 =
            error[3 * quarter..].iter().map(|e| e * e).sum::<f64>() / quarter as f64;

        assert!(
            mse_late < mse_early,
            "FDA-LMS should converge: early MSE={:.6}, late MSE={:.6}",
            mse_early,
            mse_late,
        );
    }

    #[test]
    fn test_fda_lms_output_length() {
        let n = 100usize;
        let signal = vec![1.0f64; n];
        let desired = vec![0.5f64; n];

        let (output, coeffs, error) =
            parallel_fda_lms_filter(&signal, &desired, 8, 0.01, 16).expect("FDA-LMS failed");

        assert_eq!(output.len(), n);
        assert_eq!(coeffs.len(), 8);
        assert_eq!(error.len(), n);
    }

    #[test]
    fn test_adaptive_filter_error_conditions() {
        let signal = vec![1.0, 2.0, 3.0];
        let desired = vec![1.0, 2.0]; // Different length

        let result = parallel_adaptive_lms_filter(&signal, &desired, 2, 0.01, None);
        assert!(result.is_err());

        let signal = vec![1.0, 2.0, 3.0];
        let desired = vec![1.0, 2.0, 3.0];

        // Zero filter length
        let result = parallel_adaptive_lms_filter(&signal, &desired, 0, 0.01, None);
        assert!(result.is_err());
    }
}
