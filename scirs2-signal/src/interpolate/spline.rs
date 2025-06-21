//! Spline-based interpolation methods for signal processing
//!
//! This module provides cubic spline and cubic Hermite spline (PCHIP) interpolation
//! algorithms for filling missing values in signals with smooth curves.

use super::basic::linear_interpolate;
use super::core::{enforce_monotonicity, smooth_signal, InterpolationConfig};
use crate::error::{SignalError, SignalResult};
use ndarray::{Array1, Array2};
use scirs2_linalg::solve;

/// Applies cubic spline interpolation to fill missing values in a signal
///
/// Cubic spline interpolation creates smooth curves using piecewise cubic polynomials
/// that are twice differentiable at the joining points. This method provides very
/// smooth interpolation suitable for continuous data.
///
/// # Arguments
///
/// * `signal` - Input signal with missing values (NaN)
/// * `config` - Interpolation configuration
///
/// # Returns
///
/// * Interpolated signal where missing values are filled using cubic splines
///
/// # Example
///
/// ```rust
/// use ndarray::Array1;
/// use scirs2_signal::interpolate::{spline::cubic_spline_interpolate, core::InterpolationConfig};
///
/// let mut signal = Array1::from_vec(vec![1.0, f64::NAN, f64::NAN, 4.0, f64::NAN, 6.0]);
/// let config = InterpolationConfig::default();
/// let result = cubic_spline_interpolate(&signal, &config).unwrap();
/// // Result will contain smoothly interpolated values
/// ```
pub fn cubic_spline_interpolate(
    signal: &Array1<f64>,
    config: &InterpolationConfig,
) -> SignalResult<Array1<f64>> {
    let n = signal.len();

    // Check if input has any missing values
    let has_missing = signal.iter().any(|&x| x.is_nan());
    if !has_missing {
        return Ok(signal.clone());
    }

    // Find all non-missing points
    let mut valid_indices = Vec::new();
    let mut valid_values = Vec::new();

    for i in 0..n {
        if !signal[i].is_nan() {
            valid_indices.push(i as f64);
            valid_values.push(signal[i]);
        }
    }

    if valid_indices.is_empty() {
        return Err(SignalError::ValueError(
            "All values are missing in the input signal".to_string(),
        ));
    }

    if valid_indices.len() < 4 {
        // Not enough points for cubic spline, fall back to linear
        return linear_interpolate(signal);
    }

    // Convert to ndarray for matrix operations
    let x = Array1::from_vec(valid_indices);
    let y = Array1::from_vec(valid_values);

    // Number of valid points
    let n_valid = x.len();

    // Set up the system of equations for cubic spline
    let mut matrix = Array2::zeros((n_valid, n_valid));
    let mut rhs = Array1::zeros(n_valid);

    // First and last points have second derivative = 0 (natural spline)
    matrix[[0, 0]] = 2.0;
    matrix[[0, 1]] = 1.0;
    rhs[0] = 0.0;

    matrix[[n_valid - 1, n_valid - 2]] = 1.0;
    matrix[[n_valid - 1, n_valid - 1]] = 2.0;
    rhs[n_valid - 1] = 0.0;

    // Interior points satisfy continuity of first and second derivatives
    for i in 1..n_valid - 1 {
        let h_i = x[i] - x[i - 1];
        let h_i1 = x[i + 1] - x[i];

        matrix[[i, i - 1]] = h_i;
        matrix[[i, i]] = 2.0 * (h_i + h_i1);
        matrix[[i, i + 1]] = h_i1;

        rhs[i] = 3.0 * ((y[i + 1] - y[i]) / h_i1 - (y[i] - y[i - 1]) / h_i);
    }

    // Add regularization for stability
    for i in 0..n_valid {
        matrix[[i, i]] += config.regularization;
    }

    // Solve the system to get second derivatives at each point
    let second_derivatives = match solve(&matrix.view(), &rhs.view(), None) {
        Ok(solution) => solution,
        Err(_) => {
            return Err(SignalError::Compute(
                "Failed to solve spline equation system".to_string(),
            ));
        }
    };

    // Now we can evaluate the spline at each point in the original signal
    let mut result = signal.clone();

    for i in 0..n {
        if signal[i].is_nan() {
            let t = i as f64;

            // Find the spline segment containing t
            let mut segment_idx = 0;
            while segment_idx < n_valid - 1 && x[segment_idx + 1] < t {
                segment_idx += 1;
            }

            // If t is outside the range of valid points
            if t < x[0] {
                if config.extrapolate {
                    // Extrapolate using the first segment
                    segment_idx = 0;
                } else {
                    // Use the first valid value
                    result[i] = y[0];
                    continue;
                }
            } else if t > x[n_valid - 1] {
                if config.extrapolate {
                    // Extrapolate using the last segment
                    segment_idx = n_valid - 2;
                } else {
                    // Use the last valid value
                    result[i] = y[n_valid - 1];
                    continue;
                }
            }

            // Evaluate the cubic spline
            let x1 = x[segment_idx];
            let x2 = x[segment_idx + 1];
            let y1 = y[segment_idx];
            let y2 = y[segment_idx + 1];
            let d1 = second_derivatives[segment_idx];
            let d2 = second_derivatives[segment_idx + 1];

            let h = x2 - x1;
            let t_norm = (t - x1) / h;

            // Cubic spline formula
            let a = (1.0 - t_norm) * y1
                + t_norm * y2
                + t_norm
                    * (1.0 - t_norm)
                    * ((1.0 - t_norm) * h * h * d1 / 6.0 + t_norm * h * h * d2 / 6.0);

            result[i] = a;
        }
    }

    // Apply monotonicity constraint if requested
    if config.monotonic {
        result = enforce_monotonicity(&result);
    }

    // Apply smoothing if requested
    if config.smoothing {
        result = smooth_signal(&result, config.smoothing_factor);
    }

    Ok(result)
}

/// Applies cubic Hermite spline interpolation (PCHIP) to fill missing values in a signal
///
/// PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) produces a smooth curve
/// that preserves monotonicity and avoids overshooting. It's particularly suitable
/// for data where shape preservation is important.
///
/// # Arguments
///
/// * `signal` - Input signal with missing values (NaN)
/// * `config` - Interpolation configuration
///
/// # Returns
///
/// * Interpolated signal where missing values are filled using PCHIP
///
/// # Example
///
/// ```rust
/// use ndarray::Array1;
/// use scirs2_signal::interpolate::{spline::cubic_hermite_interpolate, core::InterpolationConfig};
///
/// let mut signal = Array1::from_vec(vec![1.0, f64::NAN, 3.0, f64::NAN, 2.0]);
/// let config = InterpolationConfig::default();
/// let result = cubic_hermite_interpolate(&signal, &config).unwrap();
/// // Result will preserve monotonicity and avoid overshooting
/// ```
pub fn cubic_hermite_interpolate(
    signal: &Array1<f64>,
    config: &InterpolationConfig,
) -> SignalResult<Array1<f64>> {
    let n = signal.len();

    // Check if input has any missing values
    let has_missing = signal.iter().any(|&x| x.is_nan());
    if !has_missing {
        return Ok(signal.clone());
    }

    // Find all non-missing points
    let mut valid_indices = Vec::new();
    let mut valid_values = Vec::new();

    for i in 0..n {
        if !signal[i].is_nan() {
            valid_indices.push(i as f64);
            valid_values.push(signal[i]);
        }
    }

    if valid_indices.is_empty() {
        return Err(SignalError::ValueError(
            "All values are missing in the input signal".to_string(),
        ));
    }

    if valid_indices.len() < 2 {
        // Not enough points for PCHIP, fill with constant value
        let value = valid_values[0];
        let mut result = signal.clone();
        for i in 0..n {
            if result[i].is_nan() {
                result[i] = value;
            }
        }
        return Ok(result);
    }

    // Convert to ndarray for easier manipulation
    let x = Array1::from_vec(valid_indices);
    let y = Array1::from_vec(valid_values);

    // Number of valid points
    let n_valid = x.len();

    // Calculate slopes that preserve monotonicity
    let mut slopes = Array1::zeros(n_valid);

    // Interior points
    for i in 1..n_valid - 1 {
        let h1 = x[i] - x[i - 1];
        let h2 = x[i + 1] - x[i];
        let delta1 = (y[i] - y[i - 1]) / h1;
        let delta2 = (y[i + 1] - y[i]) / h2;

        if delta1 * delta2 <= 0.0 {
            // If slopes have opposite signs or one is zero, set slope to zero
            slopes[i] = 0.0;
        } else {
            // Harmonic mean of slopes
            let w1 = 2.0 * h2 + h1;
            let w2 = h2 + 2.0 * h1;
            slopes[i] = (w1 + w2) / (w1 / delta1 + w2 / delta2);
        }
    }

    // End points
    let h1 = x[1] - x[0];
    let h2 = x[n_valid - 1] - x[n_valid - 2];
    let delta1 = (y[1] - y[0]) / h1;
    let delta2 = (y[n_valid - 1] - y[n_valid - 2]) / h2;

    // Secant slope for end points
    slopes[0] = delta1;
    slopes[n_valid - 1] = delta2;

    // Now we can evaluate the PCHIP at each point in the original signal
    let mut result = signal.clone();

    for i in 0..n {
        if signal[i].is_nan() {
            let t = i as f64;

            // Find the spline segment containing t
            let mut segment_idx = 0;
            while segment_idx < n_valid - 1 && x[segment_idx + 1] < t {
                segment_idx += 1;
            }

            // If t is outside the range of valid points
            if t < x[0] {
                if config.extrapolate {
                    // Extrapolate using the first segment
                    segment_idx = 0;
                } else {
                    // Use the first valid value
                    result[i] = y[0];
                    continue;
                }
            } else if t > x[n_valid - 1] {
                if config.extrapolate {
                    // Extrapolate using the last segment
                    segment_idx = n_valid - 2;
                } else {
                    // Use the last valid value
                    result[i] = y[n_valid - 1];
                    continue;
                }
            }

            // Evaluate the cubic Hermite spline
            let x1 = x[segment_idx];
            let x2 = x[segment_idx + 1];
            let y1 = y[segment_idx];
            let y2 = y[segment_idx + 1];
            let m1 = slopes[segment_idx];
            let m2 = slopes[segment_idx + 1];

            let h = x2 - x1;
            let t_norm = (t - x1) / h;

            // Hermite basis functions
            let h00 = 2.0 * t_norm.powi(3) - 3.0 * t_norm.powi(2) + 1.0;
            let h10 = t_norm.powi(3) - 2.0 * t_norm.powi(2) + t_norm;
            let h01 = -2.0 * t_norm.powi(3) + 3.0 * t_norm.powi(2);
            let h11 = t_norm.powi(3) - t_norm.powi(2);

            // PCHIP formula
            let value = h00 * y1 + h10 * h * m1 + h01 * y2 + h11 * h * m2;

            result[i] = value;
        }
    }

    // Apply smoothing if requested
    if config.smoothing {
        result = smooth_signal(&result, config.smoothing_factor);
    }

    Ok(result)
}

/// Unit tests for spline interpolation methods
#[cfg(test)]
mod tests {
    use super::*;
    use crate::interpolate::core::InterpolationConfig;
    use ndarray::Array1;

    #[test]
    fn test_cubic_spline_interpolate_no_missing() {
        let signal = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let config = InterpolationConfig::default();
        let result = cubic_spline_interpolate(&signal, &config).unwrap();
        assert_eq!(result, signal);
    }

    #[test]
    fn test_cubic_spline_interpolate_simple() {
        let signal = Array1::from_vec(vec![1.0, f64::NAN, f64::NAN, f64::NAN, 5.0]);
        let config = InterpolationConfig::default();
        let result = cubic_spline_interpolate(&signal, &config).unwrap();

        // Should produce smooth interpolation
        assert_eq!(result[0], 1.0);
        assert_eq!(result[4], 5.0);
        assert!(result.iter().all(|&x| !x.is_nan()));

        // Should be monotonically increasing
        for i in 1..result.len() {
            assert!(result[i] >= result[i - 1]);
        }
    }

    #[test]
    fn test_cubic_spline_insufficient_points() {
        let signal = Array1::from_vec(vec![1.0, f64::NAN, 3.0]);
        let config = InterpolationConfig::default();
        let result = cubic_spline_interpolate(&signal, &config).unwrap();

        // Should fall back to linear interpolation
        assert_eq!(result[0], 1.0);
        assert_eq!(result[1], 2.0);
        assert_eq!(result[2], 3.0);
    }

    #[test]
    fn test_cubic_spline_all_missing() {
        let signal = Array1::from_vec(vec![f64::NAN, f64::NAN, f64::NAN]);
        let config = InterpolationConfig::default();
        let result = cubic_spline_interpolate(&signal, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_cubic_hermite_interpolate_no_missing() {
        let signal = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let config = InterpolationConfig::default();
        let result = cubic_hermite_interpolate(&signal, &config).unwrap();
        assert_eq!(result, signal);
    }

    #[test]
    fn test_cubic_hermite_interpolate_simple() {
        let signal = Array1::from_vec(vec![1.0, f64::NAN, 3.0, f64::NAN, 2.0]);
        let config = InterpolationConfig::default();
        let result = cubic_hermite_interpolate(&signal, &config).unwrap();

        // All values should be valid
        assert!(result.iter().all(|&x| !x.is_nan()));

        // Original values should be preserved
        assert_eq!(result[0], 1.0);
        assert_eq!(result[2], 3.0);
        assert_eq!(result[4], 2.0);
    }

    #[test]
    fn test_cubic_hermite_single_point() {
        let signal = Array1::from_vec(vec![f64::NAN, 2.0, f64::NAN]);
        let config = InterpolationConfig::default();
        let result = cubic_hermite_interpolate(&signal, &config).unwrap();

        // Should fill with constant value
        assert_eq!(result[0], 2.0);
        assert_eq!(result[1], 2.0);
        assert_eq!(result[2], 2.0);
    }

    #[test]
    fn test_cubic_hermite_all_missing() {
        let signal = Array1::from_vec(vec![f64::NAN, f64::NAN, f64::NAN]);
        let config = InterpolationConfig::default();
        let result = cubic_hermite_interpolate(&signal, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_extrapolation_config() {
        let signal = Array1::from_vec(vec![f64::NAN, 2.0, 3.0, f64::NAN]);
        let mut config = InterpolationConfig::default();
        config.extrapolate = true;

        let result = cubic_spline_interpolate(&signal, &config).unwrap();
        assert!(result.iter().all(|&x| !x.is_nan()));

        let result = cubic_hermite_interpolate(&signal, &config).unwrap();
        assert!(result.iter().all(|&x| !x.is_nan()));
    }

    #[test]
    fn test_smoothing_config() {
        let signal = Array1::from_vec(vec![1.0, f64::NAN, 10.0, f64::NAN, 1.0]);
        let mut config = InterpolationConfig::default();
        config.smoothing = true;
        config.smoothing_factor = 0.3;

        let result = cubic_spline_interpolate(&signal, &config).unwrap();
        assert!(result.iter().all(|&x| !x.is_nan()));

        let result = cubic_hermite_interpolate(&signal, &config).unwrap();
        assert!(result.iter().all(|&x| !x.is_nan()));
    }

    #[test]
    fn test_monotonicity_config() {
        let signal = Array1::from_vec(vec![1.0, f64::NAN, f64::NAN, 4.0, f64::NAN, 6.0]);
        let mut config = InterpolationConfig::default();
        config.monotonic = true;

        let result = cubic_spline_interpolate(&signal, &config).unwrap();

        // Should be monotonically increasing
        for i in 1..result.len() {
            assert!(result[i] >= result[i - 1]);
        }
    }
}
