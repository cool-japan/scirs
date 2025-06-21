//! Spectrum reconstruction utilities for Sparse FFT
//!
//! This module provides functions for reconstructing signals and spectra
//! from sparse FFT results.

use crate::error::{FFTError, FFTResult};
use crate::fft::ifft;
use num_complex::Complex64;

use super::algorithms::SparseFFTResult;

/// Reconstruct the full spectrum from sparse FFT result
///
/// # Arguments
///
/// * `sparse_result` - The sparse FFT result
/// * `n` - Length of the full spectrum
///
/// # Returns
///
/// * Full spectrum
///
/// # Examples
///
/// ```rust
/// use scirs2_fft::sparse_fft::{sparse_fft, reconstruct_spectrum};
///
/// // Generate a sparse signal
/// let n = 64;
/// let mut signal = vec![0.0; n];
/// for i in 0..n {
///     let t = 2.0 * std::f64::consts::PI * (i as f64) / (n as f64);
///     signal[i] = 1.0 * (3.0 * t).sin() + 0.5 * (7.0 * t).sin();
/// }
///
/// // Compute sparse FFT with k=4
/// let sparse_result = sparse_fft(&signal, 4, None, None).unwrap();
///
/// // Reconstruct full spectrum
/// let full_spectrum = reconstruct_spectrum(&sparse_result, n).unwrap();
///
/// // The reconstructed spectrum should have length n
/// assert_eq!(full_spectrum.len(), n);
/// ```
pub fn reconstruct_spectrum(
    sparse_result: &SparseFFTResult,
    n: usize,
) -> FFTResult<Vec<Complex64>> {
    // Create full spectrum from sparse representation
    let mut spectrum = vec![Complex64::new(0.0, 0.0); n];
    for (value, &index) in sparse_result
        .values
        .iter()
        .zip(sparse_result.indices.iter())
    {
        spectrum[index] = *value;
    }
    Ok(spectrum)
}

/// Reconstructs time-domain signal from sparse frequency components
///
/// # Arguments
///
/// * `sparse_result` - The sparse FFT result
/// * `n` - Length of the signal
///
/// # Returns
///
/// * Reconstructed time-domain signal
///
/// # Examples
///
/// ```rust
/// use scirs2_fft::sparse_fft::{sparse_fft, reconstruct_time_domain};
///
/// // Generate a sparse signal
/// let n = 64;
/// let mut original_signal = vec![0.0; n];
/// for i in 0..n {
///     let t = 2.0 * std::f64::consts::PI * (i as f64) / (n as f64);
///     original_signal[i] = 1.0 * (3.0 * t).sin() + 0.5 * (7.0 * t).sin();
/// }
///
/// // Compute sparse FFT
/// let sparse_result = sparse_fft(&original_signal, 4, None, None).unwrap();
///
/// // Reconstruct time-domain signal
/// let reconstructed = reconstruct_time_domain(&sparse_result, n).unwrap();
///
/// // The reconstructed signal should have the same length
/// assert_eq!(reconstructed.len(), n);
/// ```
pub fn reconstruct_time_domain(
    sparse_result: &SparseFFTResult,
    n: usize,
) -> FFTResult<Vec<Complex64>> {
    // Create full spectrum from sparse representation
    let mut spectrum = vec![Complex64::new(0.0, 0.0); n];
    for (value, &index) in sparse_result
        .values
        .iter()
        .zip(sparse_result.indices.iter())
    {
        spectrum[index] = *value;
    }

    // Perform inverse FFT to get time-domain signal
    ifft(&spectrum, None)
}

/// Reconstructs a signal with enhanced frequency resolution by zero padding
///
/// This method allows reconstructing a signal with higher frequency resolution
/// by zero-padding the sparse spectrum before performing the inverse FFT.
///
/// # Arguments
///
/// * `sparse_result` - The sparse FFT result containing frequency components
/// * `original_length` - The original length of the signal
/// * `target_length` - The desired length after zero padding (must be >= original_length)
///
/// # Returns
///
/// * The reconstructed signal with enhanced frequency resolution
///
/// # Examples
///
/// ```rust
/// use scirs2_fft::sparse_fft::{sparse_fft, reconstruct_high_resolution};
///
/// // Generate a sparse signal
/// let n = 32;
/// let mut signal = vec![0.0; n];
/// for i in 0..n {
///     let t = 2.0 * std::f64::consts::PI * (i as f64) / (n as f64);
///     signal[i] = 1.0 * (3.0 * t).sin();
/// }
///
/// // Compute sparse FFT
/// let sparse_result = sparse_fft(&signal, 2, None, None).unwrap();
///
/// // Reconstruct with higher resolution (2x the original length)
/// let high_res = reconstruct_high_resolution(&sparse_result, n, 2 * n).unwrap();
///
/// // The high-resolution signal should have the target length
/// assert_eq!(high_res.len(), 2 * n);
/// ```
pub fn reconstruct_high_resolution(
    sparse_result: &SparseFFTResult,
    original_length: usize,
    target_length: usize,
) -> FFTResult<Vec<Complex64>> {
    if target_length < original_length {
        return Err(FFTError::DimensionError(format!(
            "Target length {} must be greater than or equal to original length {}",
            target_length, original_length
        )));
    }

    // First reconstruct the spectrum at original resolution
    let mut spectrum = vec![Complex64::new(0.0, 0.0); original_length];
    for (value, &index) in sparse_result
        .values
        .iter()
        .zip(sparse_result.indices.iter())
    {
        spectrum[index] = *value;
    }

    // Scale the frequencies to the new length
    let mut high_res_spectrum = vec![Complex64::new(0.0, 0.0); target_length];

    // For components below the Nyquist frequency
    let original_nyquist = original_length / 2;
    let target_nyquist = target_length / 2;

    // Copy DC component
    high_res_spectrum[0] = spectrum[0];

    // Scale positive frequencies (0 to Nyquist)
    #[allow(clippy::needless_range_loop)]
    for i in 1..=original_nyquist {
        // Calculate the scaled frequency index in the new spectrum
        let new_idx =
            ((i as f64) * (target_nyquist as f64) / (original_nyquist as f64)).round() as usize;
        if new_idx < target_length {
            high_res_spectrum[new_idx] = spectrum[i];
        }
    }

    // Handle the negative frequencies (those above Nyquist in the original spectrum)
    if original_length % 2 == 0 {
        // Even length case - map original negative frequencies to the new negative frequencies
        #[allow(clippy::needless_range_loop)]
        for i in (original_nyquist + 1)..original_length {
            // Calculate the relative position in the negative frequency range
            let rel_pos = original_length - i;
            let new_idx = target_length - rel_pos;
            if new_idx < target_length {
                high_res_spectrum[new_idx] = spectrum[i];
            }
        }

        // If even length, also copy the Nyquist component
        if original_length % 2 == 0 && target_length % 2 == 0 {
            high_res_spectrum[target_nyquist] = spectrum[original_nyquist];
        }
    } else {
        // Odd length case
        #[allow(clippy::needless_range_loop)]
        for i in (original_nyquist + 1)..original_length {
            // Calculate the relative position in the negative frequency range
            let rel_pos = original_length - i;
            let new_idx = target_length - rel_pos;
            if new_idx < target_length {
                high_res_spectrum[new_idx] = spectrum[i];
            }
        }
    }

    // Compute the inverse FFT to get the high-resolution time-domain signal
    ifft(&high_res_spectrum, None)
}

/// Reconstructs a filtered version of the signal by frequency-domain filtering
///
/// # Arguments
///
/// * `sparse_result` - The sparse FFT result
/// * `n` - The length of the original signal
/// * `filter_fn` - A function that takes a frequency index and returns a scaling factor (0.0 to 1.0)
///
/// # Returns
///
/// * The filtered signal
///
/// # Examples
///
/// ```rust
/// use scirs2_fft::sparse_fft::{sparse_fft, reconstruct_filtered};
///
/// // Generate a sparse signal
/// let n = 64;
/// let mut signal = vec![0.0; n];
/// for i in 0..n {
///     let t = 2.0 * std::f64::consts::PI * (i as f64) / (n as f64);
///     signal[i] = 1.0 * (3.0 * t).sin() + 0.5 * (7.0 * t).sin() + 0.25 * (15.0 * t).sin();
/// }
///
/// // Compute sparse FFT
/// let sparse_result = sparse_fft(&signal, 6, None, None).unwrap();
///
/// // Create a low-pass filter (keep only low frequencies)
/// let low_pass_filter = |freq_index: usize, n: usize| -> f64 {
///     if freq_index <= n / 8 || freq_index >= 7 * n / 8 {
///         1.0  // Keep low frequencies
///     } else {
///         0.0  // Remove high frequencies
///     }
/// };
///
/// // Reconstruct with filtering
/// let filtered = reconstruct_filtered(&sparse_result, n, low_pass_filter).unwrap();
///
/// assert_eq!(filtered.len(), n);
/// ```
pub fn reconstruct_filtered<F>(
    sparse_result: &SparseFFTResult,
    n: usize,
    filter_fn: F,
) -> FFTResult<Vec<Complex64>>
where
    F: Fn(usize, usize) -> f64,
{
    // Create full spectrum from sparse representation
    let mut spectrum = vec![Complex64::new(0.0, 0.0); n];

    // Apply the filter function to each component
    for (value, &index) in sparse_result
        .values
        .iter()
        .zip(sparse_result.indices.iter())
    {
        let scale = filter_fn(index, n);
        spectrum[index] = *value * scale;
    }

    // Perform inverse FFT to get filtered time-domain signal
    ifft(&spectrum, None)
}

#[cfg(test)]
mod tests {
    use super::super::algorithms::sparse_fft;
    use super::*;
    use std::f64::consts::PI;

    fn create_test_signal(n: usize) -> Vec<f64> {
        let mut signal = vec![0.0; n];
        for i in 0..n {
            let t = 2.0 * PI * (i as f64) / (n as f64);
            signal[i] = 1.0 * (3.0 * t).sin() + 0.5 * (7.0 * t).sin();
        }
        signal
    }

    #[test]
    fn test_reconstruct_spectrum() {
        let signal = create_test_signal(64);
        let sparse_result = sparse_fft(&signal, 4, None, None).unwrap();
        let spectrum = reconstruct_spectrum(&sparse_result, 64).unwrap();

        assert_eq!(spectrum.len(), 64);
        // Verify that the reconstructed spectrum has the expected sparse structure
        // Count non-zero components
        let non_zero_count = spectrum.iter().filter(|&c| c.norm() > 1e-10).count();
        assert_eq!(non_zero_count, sparse_result.values.len());

        // Verify that all sparse components are present in the spectrum
        for (&index, &_value) in sparse_result
            .indices
            .iter()
            .zip(sparse_result.values.iter())
        {
            assert!(index < spectrum.len());
            // The value should be non-zero at this index
            assert!(
                spectrum[index].norm() > 1e-10,
                "Expected non-zero value at index {}",
                index
            );
        }
    }

    #[test]
    fn test_reconstruct_time_domain() {
        let signal = create_test_signal(64);
        let sparse_result = sparse_fft(&signal, 4, None, None).unwrap();
        let reconstructed = reconstruct_time_domain(&sparse_result, 64).unwrap();

        assert_eq!(reconstructed.len(), 64);
    }

    #[test]
    fn test_reconstruct_high_resolution() {
        let signal = create_test_signal(32);
        let sparse_result = sparse_fft(&signal, 4, None, None).unwrap();
        let high_res = reconstruct_high_resolution(&sparse_result, 32, 64).unwrap();

        assert_eq!(high_res.len(), 64);
    }

    #[test]
    fn test_reconstruct_high_resolution_invalid_length() {
        let signal = create_test_signal(32);
        let sparse_result = sparse_fft(&signal, 4, None, None).unwrap();
        let result = reconstruct_high_resolution(&sparse_result, 32, 16);

        assert!(result.is_err());
    }

    #[test]
    fn test_reconstruct_filtered() {
        let signal = create_test_signal(64);
        let sparse_result = sparse_fft(&signal, 6, None, None).unwrap();

        // Low-pass filter
        let filter = |freq_index: usize, n: usize| -> f64 {
            if freq_index <= n / 8 || freq_index >= 7 * n / 8 {
                1.0
            } else {
                0.0
            }
        };

        let filtered = reconstruct_filtered(&sparse_result, 64, filter).unwrap();
        assert_eq!(filtered.len(), 64);
    }
}
