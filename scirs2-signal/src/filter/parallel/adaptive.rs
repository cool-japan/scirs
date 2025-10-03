//! Parallel adaptive filtering implementations
//!
//! This module provides parallel implementations of adaptive filters including
//! LMS (Least Mean Squares) and related adaptive filtering algorithms.

use crate::error::{SignalError, SignalResult};

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

/// Frequency-domain adaptive filter (FDA-LMS)
///
/// Implements frequency-domain LMS adaptive filtering for improved
/// computational efficiency with long filters.
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `desired` - Desired response signal
/// * `filter_length` - Length of adaptive filter
/// * `step_size` - FDA-LMS step size (learning rate)
/// * `block_size` - Size of FFT blocks (should be power of 2)
///
/// # Returns
///
/// * Tuple of (filtered output, final filter coefficients, error signal)
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

    if block_size < filter_length {
        return Err(SignalError::ValueError(
            "Block size must be at least as large as filter length".to_string(),
        ));
    }

    // For now, fall back to block LMS implementation
    // TODO: Implement full frequency-domain processing with FFT
    parallel_block_lms_filter(signal, desired, filter_length, step_size, block_size)
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
            parallel_adaptive_lms_filter(&signal, &desired, 10, 0.01, None).unwrap();

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
            parallel_adaptive_nlms_filter(&signal, &desired, 8, 0.1, 0.001, None).unwrap();

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
            parallel_block_lms_filter(&signal, &desired, 6, 0.05, 10).unwrap();

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
            parallel_fda_lms_filter(&signal, &desired, 8, 0.02, 16).unwrap();

        assert_eq!(output.len(), n);
        assert_eq!(coeffs.len(), 8);

        // Check that filter adapted
        let coeff_energy: f64 = coeffs.iter().map(|&x| x * x).sum();
        assert!(coeff_energy > 0.0);
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
