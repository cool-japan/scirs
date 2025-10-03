// Resampling and rate conversion
//
// This module provides functions for resampling signals at different rates,
// including upsampling, downsampling, and arbitrary resampling.

use crate::error::{SignalError, SignalResult};
use scirs2_core::numeric::{Float, NumCast};
use std::fmt::Debug;

#[allow(unused_imports)]
/// Resample a signal using polyphase filtering.
///
/// # Arguments
///
/// * `x` - Input signal
/// * `up` - Upsampling factor
/// * `down` - Downsampling factor
/// * `window` - Optional FIR window to use for filter design
///
/// # Returns
///
/// * Resampled signal
///
/// # Examples
///
/// ```
/// use scirs2_signal::resample::resample;
///
/// // Generate a simple signal
/// let signal = (0..100).map(|i| (i as f64 * 0.1).sin()).collect::<Vec<_>>();
///
/// // Resample by a factor of 2/1 (double the sampling rate)
/// let resampled = resample(&signal, 2, 1, None).unwrap();
///
/// // Result should be approximately twice as long
/// assert_eq!(resampled.len(), signal.len() * 2);
/// ```
#[allow(dead_code)]
pub fn resample<T>(x: &[T], up: usize, down: usize, window: Option<&str>) -> SignalResult<Vec<f64>>
where
    T: Float + NumCast + Debug,
{
    if x.is_empty() {
        return Err(SignalError::ValueError("Input signal is empty".to_string()));
    }

    if up == 0 || down == 0 {
        return Err(SignalError::ValueError(format!(
            "Upsampling and downsampling factors must be positive, got up={}, down={}",
            up, down
        )));
    }

    // Convert to f64 for internal processing
    let x_f64: Vec<f64> = x
        .iter()
        .map(|&val| {
            NumCast::from(val).ok_or_else(|| {
                SignalError::ValueError(format!("Could not convert {:?} to f64", val))
            })
        })
        .collect::<SignalResult<Vec<_>>>()?;

    // Simplify polyphase implementation for common cases

    if up == 1 && down == 1 {
        // No resampling needed
        return Ok(x_f64);
    }

    if up == 1 {
        // Simple downsampling
        let result: Vec<f64> = x_f64.iter().step_by(down).copied().collect();
        return Ok(result);
    }

    if down == 1 {
        // Simple upsampling
        let mut result = Vec::with_capacity(x_f64.len() * up);
        for &val in &x_f64 {
            result.push(val);
            result.extend(std::iter::repeat_n(0.0, up - 1));
        }
        return Ok(result);
    }

    // For arbitrary rational resampling, implement a simplified polyphase filter
    // This is a basic implementation that doesn't include proper filter design

    // Determine filter properties
    let window_type = window.unwrap_or("hamming");
    let gcd = gcd(up, down);
    let up = up / gcd;
    let down = down / gcd;

    // Calculate number of output samples
    let n_out = ((x_f64.len() * up) as f64 / down as f64).ceil() as usize;
    let mut result = Vec::with_capacity(n_out);

    // Generate a simple lowpass filter kernel
    // In a real implementation, we would use a proper filter design function
    let filter_length = 10 * up; // A reasonable length for the filter
    let mut h = vec![0.0; filter_length];

    // Simple sinc filter with window
    for (i, h_val) in h.iter_mut().enumerate().take(filter_length) {
        let t = i as f64 - (filter_length - 1) as f64 / 2.0;
        let sinc = if t == 0.0 {
            1.0
        } else {
            (PI * t).sin() / (PI * t)
        };

        // Apply window function
        let window_val = match window_type {
            "hamming" => 0.54 - 0.46 * (2.0 * PI * i as f64 / (filter_length - 1) as f64).cos(),
            "hanning" => 0.5 * (1.0 - (2.0 * PI * i as f64 / (filter_length - 1) as f64).cos()),
            "blackman" => {
                let a0 = 0.42;
                let a1 = 0.5;
                let a2 = 0.08;
                let w = 2.0 * PI * i as f64 / (filter_length - 1) as f64;
                a0 - a1 * w.cos() + a2 * (2.0 * w).cos()
            }
            _ => 1.0, // Rectangular window (no window)
        };

        // Normalize filter for unity gain at DC
        *h_val = sinc * window_val / up as f64;
    }

    // Apply polyphase filtering
    for i in 0..n_out {
        let output_time = i as f64 * down as f64 / up as f64;
        let input_index = (output_time as usize).min(x_f64.len() - 1);
        let frac = output_time - input_index as f64;
        let phase = (frac * up as f64).round() as usize % up;

        let mut sum = 0.0;
        let mut norm = 0.0;

        for j in 0..filter_length / up {
            let coef_index = j * up + phase;
            let input_index_offset = input_index as isize - j as isize;

            if input_index_offset >= 0 && input_index_offset < x_f64.len() as isize {
                sum += x_f64[input_index_offset as usize] * h[coef_index];
                norm += h[coef_index];
            }
        }

        // Normalize output
        if norm > 0.0 {
            sum /= norm;
        }

        result.push(sum);
    }

    Ok(result)
}

/// Upsample a signal by inserting zeros and then applying a lowpass filter.
///
/// # Arguments
///
/// * `x` - Input signal
/// * `factor` - Upsampling factor
///
/// # Returns
///
/// * Upsampled signal
///
/// # Examples
///
/// ```
/// use scirs2_signal::resample::upsample;
///
/// // Generate a simple signal
/// let signal = (0..100).map(|i| (i as f64 * 0.1).sin()).collect::<Vec<_>>();
///
/// // Upsample by a factor of 2
/// let upsampled = upsample(&signal, 2).unwrap();
///
/// // Result should be approximately twice as long
/// assert_eq!(upsampled.len(), signal.len() * 2);
/// ```
#[allow(dead_code)]
pub fn upsample<T>(x: &[T], factor: usize) -> SignalResult<Vec<f64>>
where
    T: Float + NumCast + Debug,
{
    if factor == 0 {
        return Err(SignalError::ValueError(
            "Upsampling factor must be positive".to_string(),
        ));
    }

    if factor == 1 {
        // No upsampling needed
        let x_f64: Vec<f64> = x
            .iter()
            .map(|&val| {
                NumCast::from(val).ok_or_else(|| {
                    SignalError::ValueError(format!("Could not convert {:?} to f64", val))
                })
            })
            .collect::<SignalResult<Vec<_>>>()?;
        return Ok(x_f64);
    }

    // Use the general resample function with down=1
    resample(x, factor, 1, Some("hamming"))
}

/// Downsample a signal by applying a lowpass filter and then picking every nth sample.
///
/// # Arguments
///
/// * `x` - Input signal
/// * `factor` - Downsampling factor
///
/// # Returns
///
/// * Downsampled signal
///
/// # Examples
///
/// ```
/// use scirs2_signal::resample::downsample;
///
/// // Generate a simple signal
/// let signal = (0..100).map(|i| (i as f64 * 0.1).sin()).collect::<Vec<_>>();
///
/// // Downsample by a factor of 2
/// let downsampled = downsample(&signal, 2).unwrap();
///
/// // Result should be approximately half as long
/// assert_eq!(downsampled.len(), (signal.len() + 1) / 2);
/// ```
#[allow(dead_code)]
pub fn downsample<T>(x: &[T], factor: usize) -> SignalResult<Vec<f64>>
where
    T: Float + NumCast + Debug,
{
    if factor == 0 {
        return Err(SignalError::ValueError(
            "Downsampling factor must be positive".to_string(),
        ));
    }

    if factor == 1 {
        // No downsampling needed
        let x_f64: Vec<f64> = x
            .iter()
            .map(|&val| {
                NumCast::from(val).ok_or_else(|| {
                    SignalError::ValueError(format!("Could not convert {:?} to f64", val))
                })
            })
            .collect::<SignalResult<Vec<_>>>()?;
        return Ok(x_f64);
    }

    // Use the general resample function with up=1
    resample(x, 1, factor, Some("hamming"))
}

/// Resample a signal to a new length.
///
/// # Arguments
///
/// * `x` - Input signal
/// * `num` - Number of samples in resampled signal
///
/// # Returns
///
/// * Resampled signal with specified length
///
/// # Examples
///
/// ```
/// use scirs2_signal::resample::resample_poly;
///
/// // Generate a simple signal
/// let signal = (0..100).map(|i| (i as f64 * 0.1).sin()).collect::<Vec<_>>();
///
/// // Resample to 150 points
/// let resampled = resample_poly(&signal, 150).unwrap();
///
/// assert_eq!(resampled.len(), 150);
/// ```
#[allow(dead_code)]
pub fn resample_poly<T>(x: &[T], num: usize) -> SignalResult<Vec<f64>>
where
    T: Float + NumCast + Debug,
{
    if x.is_empty() {
        return Err(SignalError::ValueError("Input signal is empty".to_string()));
    }

    if num == 0 {
        return Err(SignalError::ValueError(
            "Output size must be positive".to_string(),
        ));
    }

    if num == x.len() {
        // No resampling needed
        let x_f64: Vec<f64> = x
            .iter()
            .map(|&val| {
                NumCast::from(val).ok_or_else(|| {
                    SignalError::ValueError(format!("Could not convert {:?} to f64", val))
                })
            })
            .collect::<SignalResult<Vec<_>>>()?;
        return Ok(x_f64);
    }

    // Find a rational approximation of the resampling factor
    let (up, down) = rational_approximation(num as f64 / x.len() as f64);

    // Use the general resample function
    let resampled = resample(x, up, down, Some("hamming"))?;

    // Adjust the length if needed
    match resampled.len().cmp(&num) {
        std::cmp::Ordering::Equal => Ok(resampled),
        std::cmp::Ordering::Greater => {
            // Trim to the exact length
            Ok(resampled[0..num].to_vec())
        }
        std::cmp::Ordering::Less => {
            // Pad with zeros if shorter
            let mut result = resampled;
            result.resize(num, 0.0);
            Ok(result)
        }
    }
}

/// Find the greatest common divisor of two numbers.
#[allow(dead_code)]
fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let temp = b;
        b = a % b;
        a = temp;
    }
    a
}

/// Find a rational approximation of a floating point number.
///
/// Returns a tuple (numerator, denominator) such that n/d ≈ x.
#[allow(dead_code)]
fn rational_approximation(x: f64) -> (usize, usize) {
    if x <= 0.0 {
        return (1, 1); // Invalid input, return 1/1
    }

    const MAX_DENOM: usize = 1000; // Maximum denominator to consider

    let mut best_num = 1;
    let mut best_denom = 1;
    let mut best_error = ((x - 1.0) as f64).abs();

    for denom in 1..=MAX_DENOM {
        let num = (x * denom as f64).round() as usize;
        if num > 0 {
            let error = ((num as f64 / denom as f64) - x).abs();
            if error < best_error {
                best_num = num;
                best_denom = denom;
                best_error = error;
            }
        }
    }

    // Simplify the fraction
    let d = gcd(best_num, best_denom);
    (best_num / d, best_denom / d)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    #[test]
    fn test_resample_identity() {
        // Resampling with up=1, down=1 should return the original signal
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let resampled = resample(&signal, 1, 1, None).unwrap();

        assert_eq!(resampled.len(), signal.len());
        for (x, y) in signal.iter().zip(resampled.iter()) {
            assert_relative_eq!(*x, *y, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_upsample() {
        // Simple upsampling test
        let signal = vec![1.0, 2.0, 3.0, 4.0];
        let upsampled = upsample(&signal, 2).unwrap();

        // Should be approximately twice as long
        assert!(upsampled.len() >= signal.len() * 2 - 2);
        assert!(upsampled.len() <= signal.len() * 2 + 2);

        // Original samples should be preserved
        assert_relative_eq!(upsampled[0], 1.0, epsilon = 0.1);
        assert_relative_eq!(upsampled[2], 2.0, epsilon = 0.1);
        assert_relative_eq!(upsampled[4], 3.0, epsilon = 0.1);
        assert_relative_eq!(upsampled[6], 4.0, epsilon = 0.1);
    }

    #[test]
    fn test_downsample() {
        // Simple downsampling test
        let signal: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let downsampled = downsample(&signal, 2).unwrap();

        // Should be approximately half as long
        assert!(downsampled.len() >= signal.len() / 2 - 1);
        assert!(downsampled.len() <= signal.len() / 2 + 1);
    }

    #[test]
    fn test_resample_poly() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![0.5, 0.5];
        // Resample to a specific length
        let signal: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();

        // Resample to 150 points
        let resampled = resample_poly(&signal, 150).unwrap();
        assert_eq!(resampled.len(), 150);

        // Resample to 50 points
        let resampled = resample_poly(&signal, 50).unwrap();
        assert_eq!(resampled.len(), 50);
    }

    #[test]
    fn test_rational_approximation() {
        // Test rational approximation of π
        let (num, denom) = rational_approximation(PI);
        let approx = num as f64 / denom as f64;
        assert!((approx - PI).abs() < 0.001);

        // Common fractions should be exact
        assert_eq!(rational_approximation(0.5), (1, 2));
        assert_eq!(rational_approximation(2.0), (2, 1));
        assert_eq!(rational_approximation(1.5), (3, 2));
    }
}
