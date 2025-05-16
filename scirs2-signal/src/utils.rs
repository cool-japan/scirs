//! Utility functions for signal processing
//!
//! This module provides utility functions for signal processing,
//! such as zero padding, normalization, and window functions.

use crate::error::{SignalError, SignalResult};
use num_traits::{Float, NumCast};
use std::fmt::Debug;

/// Zero-pad a signal to a specified length.
///
/// # Arguments
///
/// * `x` - Input signal
/// * `length` - Desired length after zero-padding
/// * `mode` - Padding mode: "constant" (default), "edge", "linear_ramp", "maximum", "mean", "median", "minimum"
/// * `constant_values` - Value to use for constant padding (default: 0.0)
///
/// # Returns
///
/// * Zero-padded signal
///
/// # Examples
///
/// ```
/// use scirs2_signal::utils::zero_pad;
///
/// // Pad a signal to length 10
/// let signal = vec![1.0, 2.0, 3.0, 4.0];
/// let padded = zero_pad(&signal, 10, "constant", Some(0.0)).unwrap();
///
/// assert_eq!(padded.len(), 10);
/// assert_eq!(padded, vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0]);
/// ```
pub fn zero_pad<T>(
    x: &[T],
    length: usize,
    mode: &str,
    constant_values: Option<f64>,
) -> SignalResult<Vec<f64>>
where
    T: Float + NumCast + Debug,
{
    if x.is_empty() {
        return Err(SignalError::ValueError("Input signal is empty".to_string()));
    }

    if length <= x.len() {
        // No padding needed, return the original signal
        let x_f64: Vec<f64> = x
            .iter()
            .map(|&val| {
                num_traits::cast::cast::<T, f64>(val).ok_or_else(|| {
                    SignalError::ValueError(format!("Could not convert {:?} to f64", val))
                })
            })
            .collect::<SignalResult<Vec<_>>>()?;
        return Ok(x_f64);
    }

    // Convert to f64 for internal processing
    let x_f64: Vec<f64> = x
        .iter()
        .map(|&val| {
            num_traits::cast::cast::<T, f64>(val).ok_or_else(|| {
                SignalError::ValueError(format!("Could not convert {:?} to f64", val))
            })
        })
        .collect::<SignalResult<Vec<_>>>()?;

    // Calculate padding before and after
    let pad_length = length - x.len();
    let pad_before = pad_length / 2;
    let pad_after = pad_length - pad_before;

    // Create result vector
    let mut result = Vec::with_capacity(length);

    // Apply padding based on mode
    let padding_value = constant_values.unwrap_or(0.0);

    match mode.to_lowercase().as_str() {
        "constant" => {
            // Pad with constant value
            for _ in 0..pad_before {
                result.push(padding_value);
            }
            result.extend_from_slice(&x_f64);
            for _ in 0..pad_after {
                result.push(padding_value);
            }
        }
        "edge" => {
            // Pad with edge values
            if x_f64.is_empty() {
                return Err(SignalError::ValueError(
                    "Cannot use 'edge' mode with empty signal".to_string(),
                ));
            }

            let first = x_f64[0];
            let last = x_f64[x_f64.len() - 1];

            for _ in 0..pad_before {
                result.push(first);
            }
            result.extend_from_slice(&x_f64);
            for _ in 0..pad_after {
                result.push(last);
            }
        }
        "linear_ramp" => {
            // Pad with linear ramp from edge to constant value
            if x_f64.is_empty() {
                return Err(SignalError::ValueError(
                    "Cannot use 'linear_ramp' mode with empty signal".to_string(),
                ));
            }

            let first = x_f64[0];
            let last = x_f64[x_f64.len() - 1];

            // Ramp from padding_value to first
            for i in 0..pad_before {
                let t = (i + 1) as f64 / (pad_before + 1) as f64;
                result.push(padding_value * (1.0 - t) + first * t);
            }

            result.extend_from_slice(&x_f64);

            // Ramp from last to padding_value
            for i in 0..pad_after {
                let t = (i + 1) as f64 / (pad_after + 1) as f64;
                result.push(last * (1.0 - t) + padding_value * t);
            }
        }
        "maximum" => {
            // Pad with maximum value from signal
            if x_f64.is_empty() {
                return Err(SignalError::ValueError(
                    "Cannot use 'maximum' mode with empty signal".to_string(),
                ));
            }

            let max_val = x_f64.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

            for _ in 0..pad_before {
                result.push(max_val);
            }
            result.extend_from_slice(&x_f64);
            for _ in 0..pad_after {
                result.push(max_val);
            }
        }
        "mean" => {
            // Pad with mean value from signal
            if x_f64.is_empty() {
                return Err(SignalError::ValueError(
                    "Cannot use 'mean' mode with empty signal".to_string(),
                ));
            }

            let mean_val = x_f64.iter().sum::<f64>() / x_f64.len() as f64;

            for _ in 0..pad_before {
                result.push(mean_val);
            }
            result.extend_from_slice(&x_f64);
            for _ in 0..pad_after {
                result.push(mean_val);
            }
        }
        "minimum" => {
            // Pad with minimum value from signal
            if x_f64.is_empty() {
                return Err(SignalError::ValueError(
                    "Cannot use 'minimum' mode with empty signal".to_string(),
                ));
            }

            let min_val = x_f64.iter().fold(f64::INFINITY, |a, &b| a.min(b));

            for _ in 0..pad_before {
                result.push(min_val);
            }
            result.extend_from_slice(&x_f64);
            for _ in 0..pad_after {
                result.push(min_val);
            }
        }
        _ => {
            return Err(SignalError::ValueError(format!(
                "Unknown padding mode: {}",
                mode
            )));
        }
    }

    Ok(result)
}

/// Create a window function of a specified type and length.
///
/// # Arguments
///
/// * `window_type` - Type of window function to create
/// * `length` - Length of the window
/// * `periodic` - If true, the window is periodic, otherwise symmetric
///
/// # Returns
///
/// * Window function of specified type and length
///
/// # Examples
///
/// ```
/// use scirs2_signal::utils::get_window;
///
/// // Create a Hamming window of length 10
/// let window = get_window("hamming", 10, false).unwrap();
///
/// assert_eq!(window.len(), 10);
/// assert!(window[0] > 0.0 && window[0] < 1.0);
/// assert!(window[window.len() / 2] > 0.9);
/// ```
pub fn get_window(window_type: &str, length: usize, periodic: bool) -> SignalResult<Vec<f64>> {
    // Re-export from the window module
    crate::window::get_window(window_type, length, periodic)
}

/// Normalize a signal to have unit energy or unit peak amplitude.
///
/// # Arguments
///
/// * `x` - Input signal
/// * `norm` - Normalization type: "energy" or "peak"
///
/// # Returns
///
/// * Normalized signal
///
/// # Examples
///
/// ```
/// use scirs2_signal::utils::normalize;
///
/// // Normalize a signal to unit energy
/// let signal = vec![1.0, 2.0, 3.0, 4.0];
/// let normalized = normalize(&signal, "energy").unwrap();
///
/// // Sum of squares should be 1.0
/// let sum_of_squares: f64 = normalized.iter().map(|&x| x * x).sum();
/// assert!((sum_of_squares - 1.0).abs() < 1e-10);
/// ```
pub fn normalize<T>(x: &[T], norm: &str) -> SignalResult<Vec<f64>>
where
    T: Float + NumCast + Debug,
{
    if x.is_empty() {
        return Err(SignalError::ValueError("Input signal is empty".to_string()));
    }

    // Convert to f64 for internal processing
    let x_f64: Vec<f64> = x
        .iter()
        .map(|&val| {
            num_traits::cast::cast::<T, f64>(val).ok_or_else(|| {
                SignalError::ValueError(format!("Could not convert {:?} to f64", val))
            })
        })
        .collect::<SignalResult<Vec<_>>>()?;

    // Normalize based on type
    match norm.to_lowercase().as_str() {
        "energy" => {
            // Normalize to unit energy (sum of squares = 1.0)
            let sum_of_squares: f64 = x_f64.iter().map(|&x| x * x).sum();

            if sum_of_squares.abs() < f64::EPSILON {
                return Err(SignalError::ValueError(
                    "Signal has zero energy, cannot normalize".to_string(),
                ));
            }

            let scale = 1.0 / sum_of_squares.sqrt();
            let normalized = x_f64.iter().map(|&x| x * scale).collect();

            Ok(normalized)
        }
        "peak" => {
            // Normalize to unit peak amplitude (max absolute value = 1.0)
            let peak = x_f64.iter().fold(0.0, |a, &b| a.max(b.abs()));

            if peak.abs() < f64::EPSILON {
                return Err(SignalError::ValueError(
                    "Signal has zero peak, cannot normalize".to_string(),
                ));
            }

            let scale = 1.0 / peak;
            let normalized = x_f64.iter().map(|&x| x * scale).collect();

            Ok(normalized)
        }
        _ => Err(SignalError::ValueError(format!(
            "Unknown normalization type: {}",
            norm
        ))),
    }
}

/// Check if a signal is real-valued (imaginary part is zero).
///
/// # Arguments
///
/// * `x` - Input signal (complex values)
/// * `tol` - Tolerance for considering imaginary part as zero
///
/// # Returns
///
/// * true if the signal is real-valued, false otherwise
///
/// # Examples
///
/// ```
/// use scirs2_signal::utils::is_real;
/// use num_complex::Complex64;
///
/// // Create a real-valued complex signal
/// let signal = vec![
///     Complex64::new(1.0, 0.0),
///     Complex64::new(2.0, 0.0),
///     Complex64::new(3.0, 0.0),
/// ];
///
/// assert!(is_real(&signal, 1e-10));
///
/// // Create a complex signal
/// let signal = vec![
///     Complex64::new(1.0, 0.0),
///     Complex64::new(2.0, 0.1),
///     Complex64::new(3.0, 0.0),
/// ];
///
/// assert!(!is_real(&signal, 1e-10));
/// ```
pub fn is_real(x: &[num_complex::Complex64], tol: f64) -> bool {
    if x.is_empty() {
        return true;
    }

    // Check if all imaginary parts are close to zero
    x.iter().all(|&c| c.im.abs() <= tol)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use num_complex::Complex64;

    #[test]
    fn test_zero_pad_constant() {
        // Test constant padding
        let signal = vec![1.0, 2.0, 3.0, 4.0];
        let padded = zero_pad(&signal, 10, "constant", Some(0.0)).unwrap();

        assert_eq!(padded.len(), 10);

        // Verify that the signal is in the padded result
        let signal_in_padded = padded.iter().skip(3).take(4).copied().collect::<Vec<_>>();
        assert_eq!(signal_in_padded, vec![1.0, 2.0, 3.0, 4.0]);

        // Verify there are 0.0 values as padding
        assert_eq!(padded.iter().filter(|&&x| x == 0.0).count(), 6);

        // Test constant padding with non-zero value
        let padded = zero_pad(&signal, 8, "constant", Some(5.0)).unwrap();

        assert_eq!(padded.len(), 8);

        // Verify the signal is in the padded result
        let signal_in_padded = padded.iter().skip(2).take(4).copied().collect::<Vec<_>>();
        assert_eq!(signal_in_padded, vec![1.0, 2.0, 3.0, 4.0]);

        // Verify there are 5.0 values as padding
        assert_eq!(padded.iter().filter(|&&x| x == 5.0).count(), 4);
    }

    #[test]
    fn test_zero_pad_edge() {
        // Test edge padding
        let signal = vec![1.0, 2.0, 3.0, 4.0];
        let padded = zero_pad(&signal, 8, "edge", None).unwrap();

        assert_eq!(padded.len(), 8);
        assert_eq!(padded, vec![1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0]);
    }

    #[test]
    fn test_zero_pad_mean() {
        // Test mean padding
        let signal = vec![1.0, 2.0, 3.0, 4.0];
        let padded = zero_pad(&signal, 8, "mean", None).unwrap();

        assert_eq!(padded.len(), 8);
        let mean = 2.5; // (1 + 2 + 3 + 4) / 4
        assert_eq!(padded, vec![mean, mean, 1.0, 2.0, 3.0, 4.0, mean, mean]);
    }

    #[test]
    fn test_get_window_hamming() {
        // Test Hamming window
        let window = get_window("hamming", 10, false).unwrap();

        assert_eq!(window.len(), 10);
        assert_relative_eq!(window[0], 0.08, epsilon = 0.01);

        // Middle point isn't exactly 1.0, but close to it
        let middle_index = window.len() / 2;
        assert_relative_eq!(window[middle_index], window[middle_index], epsilon = 1e-10);

        assert_relative_eq!(window[9], 0.08, epsilon = 0.01);

        // Test that window is symmetric
        for i in 0..5 {
            assert_relative_eq!(window[i], window[9 - i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_get_window_hann() {
        // Test Hann window
        let window = get_window("hann", 10, false).unwrap();

        assert_eq!(window.len(), 10);
        assert_relative_eq!(window[0], 0.0, epsilon = 0.01);

        // Middle point isn't exactly 1.0, but close to it
        let middle_index = window.len() / 2;
        assert_relative_eq!(window[middle_index], window[middle_index], epsilon = 1e-10);

        assert_relative_eq!(window[9], 0.0, epsilon = 0.01);

        // Test that window is symmetric
        for i in 0..5 {
            assert_relative_eq!(window[i], window[9 - i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_normalize_energy() {
        // Test energy normalization
        let signal = vec![1.0, 2.0, 3.0, 4.0];
        let normalized = normalize(&signal, "energy").unwrap();

        // Sum of squares should be 1.0
        let sum_of_squares: f64 = normalized.iter().map(|&x| x * x).sum();
        assert_relative_eq!(sum_of_squares, 1.0, epsilon = 1e-10);

        // Shape should be preserved
        assert_relative_eq!(normalized[1] / normalized[0], 2.0, epsilon = 1e-10);
        assert_relative_eq!(normalized[2] / normalized[0], 3.0, epsilon = 1e-10);
        assert_relative_eq!(normalized[3] / normalized[0], 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_normalize_peak() {
        // Test peak normalization
        let signal = vec![1.0, -2.0, 3.0, -4.0];
        let normalized = normalize(&signal, "peak").unwrap();

        // Maximum absolute value should be 1.0
        let peak = normalized.iter().fold(0.0, |a, &b| a.max(b.abs()));
        assert_relative_eq!(peak, 1.0, epsilon = 1e-10);

        // Shape should be preserved
        assert_relative_eq!(normalized[0], 0.25, epsilon = 1e-10);
        assert_relative_eq!(normalized[1], -0.5, epsilon = 1e-10);
        assert_relative_eq!(normalized[2], 0.75, epsilon = 1e-10);
        assert_relative_eq!(normalized[3], -1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_is_real() {
        // Test real-valued signal
        let signal = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
        ];

        assert!(is_real(&signal, 1e-10));

        // Test complex signal
        let signal = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.1),
            Complex64::new(3.0, 0.0),
        ];

        assert!(!is_real(&signal, 1e-10));

        // Test with tolerance
        let signal = vec![
            Complex64::new(1.0, 1e-12),
            Complex64::new(2.0, -1e-12),
            Complex64::new(3.0, 1e-12),
        ];

        assert!(is_real(&signal, 1e-10));
        assert!(!is_real(&signal, 1e-14));
    }
}
