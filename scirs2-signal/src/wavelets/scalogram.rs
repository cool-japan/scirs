// Scalogram and related visualizations

use super::transform::cwt;
use crate::error::{SignalError, SignalResult};
use scirs2_core::numeric::Complex64;
use scirs2_core::numeric::{Float, NumCast};
use std::fmt::Debug;

#[allow(unused_imports)]
/// Computes a scalogram (squared magnitude of CWT coefficients) of a signal.
///
/// A scalogram is a visual representation of the energy density of a signal
/// across different scales and times, computed from the continuous wavelet transform.
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `wavelet_fn` - Function to generate wavelet (can be a closure)
/// * `scales` - Array of scales at which to compute the transform
/// * `normalize` - Whether to normalize the scalogram (default = false)
///
/// # Returns
///
/// * 2D array of scalogram values (scales × signal_length)
///
/// # Example
///
/// ```
/// use scirs2_signal::wavelets::{scalogram, morlet};
/// use std::f64::consts::PI;
///
/// // Generate a chirp signal
/// let n = 1024;
/// let dt = 0.01;
/// let time: Vec<f64> = (0..n).map(|i| i as f64 * dt).collect();
/// let signal: Vec<f64> = time.iter().map(|&t| {
///     let freq = 5.0 + 15.0 * t; // Increase from 5 Hz to 20 Hz
///     (2.0 * PI * freq * t).sin()
/// }).collect();
///
/// // Define logarithmically spaced scales
/// let scales: Vec<f64> = (1..64).map(|i| 2.0_f64.powf(i as f64 / 8.0)).collect();
///
/// // Compute scalogram with normalization
/// let scalo = scalogram(&signal, |points, scale| morlet(points, 6.0, scale), &scales, Some(true)).unwrap();
///
/// // The shape should match the expected dimensions
/// assert_eq!(scalo.len(), scales.len());
/// assert_eq!(scalo[0].len(), signal.len());
/// ```
#[allow(dead_code)]
pub fn scalogram<T, F, W>(
    signal: &[T],
    wavelet_fn: F,
    scales: &[f64],
    normalize: Option<bool>,
) -> SignalResult<Vec<Vec<f64>>>
where
    T: Float + NumCast + Debug,
    F: Fn(usize, f64) -> SignalResult<Vec<W>> + Copy,
    W: Into<Complex64> + Copy,
{
    // Compute CWT
    let cwt_coeffs = cwt(signal, wavelet_fn, scales)?;

    // Default parameter
    let normalize_val = normalize.unwrap_or(false);

    // Convert to scalogram (squared magnitude)
    let mut scalogram = Vec::with_capacity(cwt_coeffs.len());

    for row in &cwt_coeffs {
        let magnitude_row: Vec<f64> = row.iter().map(|val| val.norm_sqr()).collect();
        scalogram.push(magnitude_row);
    }

    // Normalize if requested
    if normalize_val {
        // Find global maximum
        let mut max_val = 0.0;
        for row in &scalogram {
            for &val in row {
                if val > max_val {
                    max_val = val;
                }
            }
        }

        // Apply normalization if max is not zero
        if max_val > 0.0 {
            for row in &mut scalogram {
                for val in row.iter_mut() {
                    *val /= max_val;
                }
            }
        }
    }

    Ok(scalogram)
}

/// Computes the magnitude of CWT coefficients of a signal.
///
/// Similar to scalogram but returns the magnitude (absolute value) rather than
/// squared magnitude of the coefficients.
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `wavelet_fn` - Function to generate wavelet (can be a closure)
/// * `scales` - Array of scales at which to compute the transform
/// * `normalize` - Whether to normalize the result (default = false)
///
/// # Returns
///
/// * 2D array of magnitude values (scales × signal_length)
///
/// # Example
///
/// ```
/// use scirs2_signal::wavelets::{cwt_magnitude, morlet};
/// use std::f64::consts::PI;
///
/// // Generate a simple signal
/// let n = 1024;
/// let dt = 0.01;
/// let time: Vec<f64> = (0..n).map(|i| i as f64 * dt).collect();
/// let signal: Vec<f64> = time.iter().map(|&t| (2.0 * PI * 5.0 * t).sin()).collect();
///
/// // Define scales
/// let scales: Vec<f64> = (1..32).map(|i| i as f64).collect();
///
/// // Compute CWT magnitude
/// let mag = cwt_magnitude(&signal, |points, scale| morlet(points, 6.0, scale), &scales, Some(false)).unwrap();
///
/// // Check dimensions
/// assert_eq!(mag.len(), scales.len());
/// assert_eq!(mag[0].len(), signal.len());
/// ```
#[allow(dead_code)]
pub fn cwt_magnitude<T, F, W>(
    signal: &[T],
    wavelet_fn: F,
    scales: &[f64],
    normalize: Option<bool>,
) -> SignalResult<Vec<Vec<f64>>>
where
    T: Float + NumCast + Debug,
    F: Fn(usize, f64) -> SignalResult<Vec<W>> + Copy,
    W: Into<Complex64> + Copy,
{
    // Compute CWT
    let cwt_coeffs = cwt(signal, wavelet_fn, scales)?;

    // Default parameter
    let normalize_val = normalize.unwrap_or(false);

    // Convert to magnitude
    let mut magnitude = Vec::with_capacity(cwt_coeffs.len());

    for row in &cwt_coeffs {
        let magnitude_row: Vec<f64> = row.iter().map(|val| val.norm()).collect();
        magnitude.push(magnitude_row);
    }

    // Normalize if requested
    if normalize_val {
        // Find global maximum
        let mut max_val = 0.0;
        for row in &magnitude {
            for &val in row {
                if val > max_val {
                    max_val = val;
                }
            }
        }

        // Apply normalization if max is not zero
        if max_val > 0.0 {
            for row in &mut magnitude {
                for val in row.iter_mut() {
                    *val /= max_val;
                }
            }
        }
    }

    Ok(magnitude)
}

/// Computes the phase of CWT coefficients of a signal.
///
/// This function returns the phase (argument) of the complex CWT coefficients.
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `wavelet_fn` - Function to generate wavelet (can be a closure)
/// * `scales` - Array of scales at which to compute the transform
///
/// # Returns
///
/// * 2D array of phase values (scales × signal_length)
///
/// # Example
///
/// ```
/// use scirs2_signal::wavelets::{cwt_phase, morlet};
/// use std::f64::consts::PI;
///
/// // Generate a simple signal
/// let n = 1024;
/// let dt = 0.01;
/// let time: Vec<f64> = (0..n).map(|i| i as f64 * dt).collect();
/// let signal: Vec<f64> = time.iter().map(|&t| (2.0 * PI * 5.0 * t).sin()).collect();
///
/// // Define scales
/// let scales: Vec<f64> = (1..32).map(|i| i as f64).collect();
///
/// // Compute CWT phase
/// let phase = cwt_phase(&signal, |points, scale| morlet(points, 6.0, scale), &scales).unwrap();
///
/// // Check dimensions
/// assert_eq!(phase.len(), scales.len());
/// assert_eq!(phase[0].len(), signal.len());
///
/// // Phase values should be between -π and π
/// for row in &phase {
///     for &val in row {
///         assert!(val >= -PI && val <= PI);
///     }
/// }
/// ```
#[allow(dead_code)]
pub fn cwt_phase<T, F, W>(
    signal: &[T],
    wavelet_fn: F,
    scales: &[f64],
) -> SignalResult<Vec<Vec<f64>>>
where
    T: Float + NumCast + Debug,
    F: Fn(usize, f64) -> SignalResult<Vec<W>> + Copy,
    W: Into<Complex64> + Copy,
{
    // Compute CWT
    let cwt_coeffs = cwt(signal, wavelet_fn, scales)?;

    // Extract phase
    let mut phase = Vec::with_capacity(cwt_coeffs.len());

    for row in &cwt_coeffs {
        let phase_row: Vec<f64> = row.iter().map(|val| val.arg()).collect();
        phase.push(phase_row);
    }

    Ok(phase)
}

/// Estimates the frequencies corresponding to the scales used in CWT.
///
/// This function converts scales to approximate frequency values based on the
/// wavelet's central frequency.
///
/// # Arguments
///
/// * `scales` - Array of scales used in CWT
/// * `central_frequency` - Central frequency of the wavelet
/// * `sampling_period` - Sampling period (1/sampling_frequency) in seconds
///
/// # Returns
///
/// * Array of frequency values corresponding to each scale
///
/// # Example
///
/// ```
/// use scirs2_signal::wavelets::{scale_to_frequency};
///
/// // Define scales
/// let scales: Vec<f64> = (1..32).map(|i| i as f64).collect();
/// let dt = 0.01; // 100 Hz sampling rate
/// let central_freq = 0.85 / (2.0 * std::f64::consts::PI); // For Morlet with w0=6.0
///
/// // Compute corresponding frequencies
/// let freqs = scale_to_frequency(&scales, central_freq, dt).unwrap();
///
/// // Should have same length as scales
/// assert_eq!(freqs.len(), scales.len());
///
/// // Frequencies should decrease as scales increase
/// for i in 1..freqs.len() {
///     assert!(freqs[i] < freqs[i-1]);
/// }
/// ```
#[allow(dead_code)]
pub fn scale_to_frequency(
    scales: &[f64],
    central_frequency: f64,
    sampling_period: f64,
) -> SignalResult<Vec<f64>> {
    if scales.is_empty() {
        return Err(crate::error::SignalError::ValueError(
            "Scales array is empty".to_string(),
        ));
    }

    if sampling_period <= 0.0 {
        return Err(crate::error::SignalError::ValueError(
            "Sampling _period must be positive".to_string(),
        ));
    }

    if central_frequency <= 0.0 {
        return Err(crate::error::SignalError::ValueError(
            "Central _frequency must be positive".to_string(),
        ));
    }

    // Get the central _frequency for this wavelet
    let central_freq = central_frequency;

    // Convert scales to frequencies using the formula: f = central_freq / (scale * dt)
    let freqs: Vec<f64> = scales
        .iter()
        .map(|&s| central_freq / (s * sampling_period))
        .collect();

    Ok(freqs)
}
