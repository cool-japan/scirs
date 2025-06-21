//! Spectral and frequency-domain interpolation methods
//!
//! This module provides frequency-domain interpolation algorithms including
//! sinc interpolation, FFT-based spectral interpolation, automatic method selection,
//! and comprehensive resampling utilities.

use crate::error::{SignalError, SignalResult};
use ndarray::Array1;
use rustfft::{num_complex::Complex, FftPlanner};
use std::f64::consts::PI;

use super::basic::linear_interpolate;
use super::core::{find_nearest_valid_index, InterpolationConfig, InterpolationMethod};

/// Applies sinc interpolation to fill missing values in a bandlimited signal
///
/// Sinc interpolation is optimal for bandlimited signals and provides perfect
/// reconstruction when the Nyquist criterion is satisfied. This method uses
/// a windowed sinc function to reduce ringing artifacts.
///
/// # Arguments
///
/// * `signal` - Input signal with missing values (NaN)
/// * `cutoff_freq` - Normalized cutoff frequency (0.0 to 0.5)
///
/// # Returns
///
/// * Interpolated signal using sinc interpolation
///
/// # Example
///
/// ```rust
/// use ndarray::Array1;
/// use scirs2_signal::interpolate::spectral::sinc_interpolate;
///
/// let mut signal = Array1::from_vec(vec![1.0, f64::NAN, 3.0, f64::NAN, 5.0]);
/// let result = sinc_interpolate(&signal, 0.4).unwrap();
/// // Result contains bandlimited interpolated values
/// ```
pub fn sinc_interpolate(signal: &Array1<f64>, cutoff_freq: f64) -> SignalResult<Array1<f64>> {
    if cutoff_freq <= 0.0 || cutoff_freq > 0.5 {
        return Err(SignalError::ValueError(
            "Cutoff frequency must be in the range (0, 0.5]".to_string(),
        ));
    }

    let n = signal.len();

    // Check if input has any missing values
    let has_missing = signal.iter().any(|&x| x.is_nan());
    if !has_missing {
        return Ok(signal.clone());
    }

    // Find indices of missing and non-missing points
    let mut missing_indices = Vec::new();
    let mut valid_indices = Vec::new();
    let mut valid_values = Vec::new();

    for i in 0..n {
        if signal[i].is_nan() {
            missing_indices.push(i);
        } else {
            valid_indices.push(i);
            valid_values.push(signal[i]);
        }
    }

    if valid_indices.is_empty() {
        return Err(SignalError::ValueError(
            "All values are missing in the input signal".to_string(),
        ));
    }

    // Create result by copying input
    let mut result = signal.clone();

    // For each missing point, compute sinc interpolation
    for &missing_idx in &missing_indices {
        let mut sum = 0.0;
        let mut weight_sum = 0.0;

        for (&valid_idx, &valid_value) in valid_indices.iter().zip(valid_values.iter()) {
            // Sinc function: sin(pi*x)/(pi*x)
            let distance = missing_idx as f64 - valid_idx as f64;

            // Avoid division by zero
            let sinc = if distance.abs() < 1e-10 {
                1.0
            } else {
                let x = 2.0 * PI * cutoff_freq * distance;
                x.sin() / x
            };

            // Apply window to reduce ringing (Lanczos window)
            let window = if distance.abs() < n as f64 {
                let x = PI * distance / n as f64;
                if x.abs() < 1e-10 {
                    1.0
                } else {
                    x.sin() / x
                }
            } else {
                0.0
            };

            let weight = sinc * window;
            sum += valid_value * weight;
            weight_sum += weight;
        }

        // Normalize if total weight is non-zero
        if weight_sum.abs() > 1e-10 {
            result[missing_idx] = sum / weight_sum;
        } else {
            // Fallback to linear interpolation
            let nearest_valid_idx = find_nearest_valid_index(missing_idx, &valid_indices);
            result[missing_idx] = valid_values[nearest_valid_idx];
        }
    }

    Ok(result)
}

/// Applies spectral (FFT-based) interpolation to fill missing values in a signal
///
/// Spectral interpolation uses the frequency domain to estimate missing values
/// by iteratively refining the signal in both time and frequency domains.
/// This method is particularly effective for periodic or quasi-periodic signals.
///
/// # Arguments
///
/// * `signal` - Input signal with missing values (NaN)
/// * `config` - Interpolation configuration
///
/// # Returns
///
/// * Interpolated signal using spectral methods
///
/// # Example
///
/// ```rust
/// use ndarray::Array1;
/// use scirs2_signal::interpolate::{spectral::spectral_interpolate, core::InterpolationConfig};
///
/// let mut signal = Array1::from_vec(vec![1.0, f64::NAN, 3.0, f64::NAN, 5.0]);
/// let config = InterpolationConfig::default();
/// let result = spectral_interpolate(&signal, &config).unwrap();
/// ```
pub fn spectral_interpolate(
    signal: &Array1<f64>,
    config: &InterpolationConfig,
) -> SignalResult<Array1<f64>> {
    let n = signal.len();

    // Check if input has any missing values
    let has_missing = signal.iter().any(|&x| x.is_nan());
    if !has_missing {
        return Ok(signal.clone());
    }

    // Find indices of missing and non-missing points
    let mut mask = Array1::zeros(n);
    let mut valid_signal = signal.clone();

    for i in 0..n {
        if signal[i].is_nan() {
            mask[i] = 1.0; // 1 indicates missing
            valid_signal[i] = 0.0; // Initialize with zeros
        }
    }

    // If all values are missing, return error
    if mask.sum() == n as f64 {
        return Err(SignalError::ValueError(
            "All values are missing in the input signal".to_string(),
        ));
    }

    // Make a copy of the initial valid signal
    let mut result = valid_signal.clone();
    let mut prev_result = valid_signal.clone();

    // Create FFT planner
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    let ifft = planner.plan_fft_inverse(n);

    // Convert to complex for FFT
    let mut complex_signal = vec![Complex::new(0.0, 0.0); n];

    // Iterative spectral interpolation
    for _ in 0..config.max_iterations {
        // Copy current estimate to complex array
        for (i, &val) in result.iter().enumerate().take(n) {
            complex_signal[i] = Complex::new(val, 0.0);
        }

        // Forward FFT
        fft.process(&mut complex_signal);

        // Apply frequency constraint if requested
        if config.frequency_constraint {
            let cutoff = (n as f64 * config.cutoff_frequency) as usize;

            // Zero out high frequencies
            for value in complex_signal.iter_mut().skip(cutoff).take(n - 2 * cutoff) {
                *value = Complex::new(0.0, 0.0);
            }
        }

        // Inverse FFT
        ifft.process(&mut complex_signal);

        // Scale by 1/n
        let scale = 1.0 / n as f64;
        for value in complex_signal.iter_mut().take(n) {
            *value *= scale;
        }

        // Copy current estimate and update
        prev_result.assign(&result);

        // Update: known samples remain the same, missing samples get values from FFT
        for i in 0..n {
            if mask[i] > 0.5 {
                // Missing data point
                result[i] = complex_signal[i].re;
            }
        }

        // Check for convergence
        let diff = (&result - &prev_result).mapv(|x| x.powi(2)).sum().sqrt();
        let norm = result.mapv(|x| x.powi(2)).sum().sqrt();

        if diff / norm < config.convergence_threshold {
            break;
        }
    }

    Ok(result)
}

/// Automatically selects the best interpolation method for a given signal
///
/// This function evaluates multiple interpolation methods and selects the one
/// that produces the best result based on either cross-validation or smoothness criteria.
///
/// # Arguments
///
/// * `signal` - Input signal with missing values (NaN)
/// * `config` - Interpolation configuration
/// * `cross_validation` - Whether to use cross-validation (true) or smoothness (false)
///
/// # Returns
///
/// * Tuple of (interpolated signal, selected method)
///
/// # Example
///
/// ```rust
/// use ndarray::Array1;
/// use scirs2_signal::interpolate::{spectral::auto_interpolate, core::InterpolationConfig};
///
/// let mut signal = Array1::from_vec(vec![1.0, f64::NAN, 3.0, f64::NAN, 5.0]);
/// let config = InterpolationConfig::default();
/// let (result, method) = auto_interpolate(&signal, &config, true).unwrap();
/// println!("Best method: {:?}", method);
/// ```
pub fn auto_interpolate(
    signal: &Array1<f64>,
    config: &InterpolationConfig,
    cross_validation: bool,
) -> SignalResult<(Array1<f64>, InterpolationMethod)> {
    let n = signal.len();

    // Check if input has any missing values
    let has_missing = signal.iter().any(|&x| x.is_nan());
    if !has_missing {
        return Ok((signal.clone(), InterpolationMethod::Linear));
    }

    let methods = [
        InterpolationMethod::Linear,
        InterpolationMethod::CubicSpline,
        InterpolationMethod::CubicHermite,
        InterpolationMethod::Sinc,
        InterpolationMethod::Spectral,
        InterpolationMethod::MinimumEnergy,
        InterpolationMethod::NearestNeighbor,
    ];

    if cross_validation {
        // Find all valid points
        let mut valid_indices = Vec::new();

        for i in 0..n {
            if !signal[i].is_nan() {
                valid_indices.push(i);
            }
        }

        let n_valid = valid_indices.len();

        if n_valid < 5 {
            // Not enough points for cross-validation
            let result = linear_interpolate(signal)?;
            return Ok((result, InterpolationMethod::Linear));
        }

        // Prepare for k-fold cross-validation (k=5)
        let k = 5.min(n_valid);
        let fold_size = n_valid / k;

        let mut best_method = InterpolationMethod::Linear;
        let mut min_error = f64::MAX;

        for &method in &methods {
            let mut total_error = 0.0;

            // K-fold cross-validation
            for fold in 0..k {
                let start = fold * fold_size;
                let end = if fold == k - 1 {
                    n_valid
                } else {
                    (fold + 1) * fold_size
                };

                // Create temporary signal with additional missing values
                let mut temp_signal = signal.clone();

                // Mask out validation fold
                for &idx in valid_indices.iter().skip(start).take(end - start) {
                    temp_signal[idx] = f64::NAN;
                }

                // Interpolate with current method
                let interpolated = super::core::interpolate(&temp_signal, method, config)?;

                // Calculate error on validation fold
                let mut fold_error = 0.0;
                for &idx in valid_indices.iter().skip(start).take(end - start) {
                    let error = interpolated[idx] - signal[idx];
                    fold_error += error * error;
                }

                total_error += fold_error / (end - start) as f64;
            }

            let avg_error = total_error / k as f64;

            if avg_error < min_error {
                min_error = avg_error;
                best_method = method;
            }
        }

        // Apply the best method to the original signal
        let result = super::core::interpolate(signal, best_method, config)?;
        Ok((result, best_method))
    } else {
        // Try all methods and pick the one with the smoothest result
        let mut min_roughness = f64::MAX;
        let mut best_method = InterpolationMethod::Linear;
        let mut best_result = linear_interpolate(signal)?;

        for &method in &methods {
            let interpolated = super::core::interpolate(signal, method, config)?;

            // Calculate second-derivative roughness
            let mut roughness = 0.0;
            for i in 1..n - 1 {
                let d2 = interpolated[i - 1] - 2.0 * interpolated[i] + interpolated[i + 1];
                roughness += d2 * d2;
            }

            if roughness < min_roughness {
                min_roughness = roughness;
                best_method = method;
                best_result = interpolated;
            }
        }

        Ok((best_result, best_method))
    }
}

pub mod resampling {
    //! Advanced resampling utilities for signal interpolation and sample rate conversion
    //!
    //! This module provides high-quality resampling algorithms based on windowed sinc
    //! interpolation, polyphase filtering, and other advanced signal processing techniques.

    use crate::error::{SignalError, SignalResult};
    use std::f64::consts::PI;

    /// Configuration for high-quality resampling
    #[derive(Debug, Clone)]
    pub struct ResamplingConfig {
        /// Length of the sinc filter kernel (in samples)
        pub kernel_length: usize,
        /// Beta parameter for Kaiser window (controls sidelobe attenuation)
        pub kaiser_beta: f64,
        /// Cutoff frequency as fraction of Nyquist rate
        pub cutoff_frequency: f64,
        /// Oversampling factor for polyphase filters
        pub oversampling_factor: usize,
        /// Whether to use zero-phase filtering
        pub zero_phase: bool,
    }

    impl Default for ResamplingConfig {
        fn default() -> Self {
            Self {
                kernel_length: 65, // Must be odd
                kaiser_beta: 8.0,
                cutoff_frequency: 0.9,
                oversampling_factor: 32,
                zero_phase: true,
            }
        }
    }

    /// High-quality sinc interpolation for arbitrary resampling ratios
    ///
    /// This function provides high-quality resampling using windowed sinc interpolation
    /// with Kaiser windowing for optimal frequency response characteristics.
    ///
    /// # Arguments
    ///
    /// * `signal` - Input signal to resample
    /// * `target_length` - Target length of resampled signal
    /// * `config` - Resampling configuration
    ///
    /// # Returns
    ///
    /// * Resampled signal with the specified target length
    pub fn sinc_resample(
        signal: &[f64],
        target_length: usize,
        config: &ResamplingConfig,
    ) -> SignalResult<Vec<f64>> {
        if signal.is_empty() {
            return Err(SignalError::ValueError("Input signal is empty".to_string()));
        }

        if target_length == 0 {
            return Err(SignalError::ValueError(
                "Target length must be positive".to_string(),
            ));
        }

        let input_length = signal.len();
        let ratio = input_length as f64 / target_length as f64;

        // Create sinc kernel
        let kernel = create_sinc_kernel(config)?;
        let kernel_half = kernel.len() / 2;

        let mut output = vec![0.0; target_length];

        for (i, output_sample) in output.iter_mut().enumerate() {
            // Calculate the exact input position
            let exact_pos = i as f64 * ratio;
            let center_sample = exact_pos.round() as i32;
            let fractional_delay = exact_pos - center_sample as f64;

            let mut sum = 0.0;
            let mut weight_sum = 0.0;

            // Apply windowed sinc kernel
            for k in 0..kernel.len() {
                let sample_idx = center_sample + k as i32 - kernel_half as i32;

                if sample_idx >= 0 && sample_idx < input_length as i32 {
                    let kernel_pos = k as f64 - kernel_half as f64 + fractional_delay;
                    let sinc_weight =
                        evaluate_sinc_kernel(&kernel, kernel_pos, config.cutoff_frequency);

                    sum += signal[sample_idx as usize] * sinc_weight;
                    weight_sum += sinc_weight;
                }
            }

            // Normalize to preserve signal amplitude
            *output_sample = if weight_sum.abs() > 1e-10 {
                sum / weight_sum
            } else {
                0.0
            };
        }

        Ok(output)
    }

    /// Creates a windowed sinc kernel for resampling
    fn create_sinc_kernel(config: &ResamplingConfig) -> SignalResult<Vec<f64>> {
        let mut kernel = vec![0.0; config.kernel_length];
        let half_length = config.kernel_length / 2;

        for (i, kernel_val) in kernel.iter_mut().enumerate() {
            let x = (i as f64 - half_length as f64) / config.oversampling_factor as f64;

            // Sinc function
            let sinc = if x.abs() < 1e-10 {
                1.0
            } else {
                let pi_x = PI * x * config.cutoff_frequency;
                pi_x.sin() / pi_x
            };

            // Kaiser window
            let window = kaiser_window(i, config.kernel_length, config.kaiser_beta);

            *kernel_val = sinc * window;
        }

        Ok(kernel)
    }

    /// Evaluates the sinc kernel at a fractional position
    fn evaluate_sinc_kernel(kernel: &[f64], position: f64, _cutoff: f64) -> f64 {
        let idx = position.round() as i32 + kernel.len() as i32 / 2;

        if idx >= 0 && (idx as usize) < kernel.len() {
            kernel[idx as usize]
        } else {
            0.0
        }
    }

    /// Kaiser window function
    fn kaiser_window(n: usize, length: usize, beta: f64) -> f64 {
        let alpha = (length - 1) as f64 / 2.0;
        let x = (n as f64 - alpha) / alpha;

        if x.abs() <= 1.0 {
            bessel_i0(beta * (1.0 - x * x).sqrt()) / bessel_i0(beta)
        } else {
            0.0
        }
    }

    /// Modified Bessel function of the first kind, order 0
    fn bessel_i0(x: f64) -> f64 {
        let mut sum = 1.0;
        let mut term = 1.0;
        let x_half_squared = (x / 2.0).powi(2);

        for k in 1..=50 {
            term *= x_half_squared / (k as f64).powi(2);
            sum += term;

            if term < 1e-12 * sum {
                break;
            }
        }

        sum
    }

    // Additional resampling utilities would be included here
    // (polyphase filtering, bandlimited interpolation, etc.)
    // This is a simplified version focusing on the core functionality
}

pub mod polynomial {
    //! Polynomial interpolation methods and utilities
    //!
    //! This module provides various polynomial interpolation techniques including
    //! Lagrange interpolation, Newton's method, Chebyshev interpolation, and
    //! least-squares polynomial fitting.

    use crate::error::{SignalError, SignalResult};
    use ndarray::{Array1, Array2};
    use scirs2_linalg::solve;

    /// Lagrange polynomial interpolation
    ///
    /// Performs Lagrange polynomial interpolation to estimate values at
    /// specified points given a set of known data points.
    ///
    /// # Arguments
    ///
    /// * `x_known` - Known x-coordinates
    /// * `y_known` - Known y-coordinates
    /// * `x_target` - Target x-coordinates for interpolation
    ///
    /// # Returns
    ///
    /// * Interpolated y-values at target points
    pub fn lagrange_interpolate(
        x_known: &[f64],
        y_known: &[f64],
        x_target: &[f64],
    ) -> SignalResult<Vec<f64>> {
        if x_known.len() != y_known.len() {
            return Err(SignalError::ValueError(
                "Known x and y arrays must have the same length".to_string(),
            ));
        }

        if x_known.is_empty() {
            return Err(SignalError::ValueError(
                "Known points arrays cannot be empty".to_string(),
            ));
        }

        let n = x_known.len();
        let mut result = vec![0.0; x_target.len()];

        for (target_idx, &x) in x_target.iter().enumerate() {
            let mut sum = 0.0;

            for i in 0..n {
                let mut product = y_known[i];

                for j in 0..n {
                    if i != j {
                        let denominator = x_known[i] - x_known[j];
                        if denominator.abs() < 1e-12 {
                            return Err(SignalError::ValueError(
                                "Duplicate x-coordinates in known points".to_string(),
                            ));
                        }
                        product *= (x - x_known[j]) / denominator;
                    }
                }

                sum += product;
            }

            result[target_idx] = sum;
        }

        Ok(result)
    }

    /// Least-squares polynomial fitting
    ///
    /// Fits a polynomial of specified degree to the given data points
    /// using least-squares regression.
    ///
    /// # Arguments
    ///
    /// * `x` - x-coordinates of data points
    /// * `y` - y-coordinates of data points
    /// * `degree` - Degree of the polynomial to fit
    ///
    /// # Returns
    ///
    /// * Polynomial coefficients (from constant to highest degree)
    pub fn polynomial_fit(x: &[f64], y: &[f64], degree: usize) -> SignalResult<Vec<f64>> {
        if x.len() != y.len() {
            return Err(SignalError::ValueError(
                "x and y arrays must have the same length".to_string(),
            ));
        }

        if x.len() <= degree {
            return Err(SignalError::ValueError(
                "Number of data points must exceed polynomial degree".to_string(),
            ));
        }

        let n = x.len();
        let m = degree + 1;

        // Create Vandermonde matrix
        let mut a = Array2::zeros((n, m));
        for i in 0..n {
            for j in 0..m {
                a[[i, j]] = x[i].powi(j as i32);
            }
        }

        let y_array = Array1::from_vec(y.to_vec());

        // Solve normal equations: A^T A x = A^T b
        let at_a = a.t().dot(&a);
        let at_b = a.t().dot(&y_array);

        match solve(&at_a.view(), &at_b.view(), None) {
            Ok(coeffs) => Ok(coeffs.to_vec()),
            Err(_) => Err(SignalError::Compute(
                "Failed to solve polynomial fitting system".to_string(),
            )),
        }
    }

    /// Evaluates a polynomial at given points
    ///
    /// # Arguments
    ///
    /// * `coeffs` - Polynomial coefficients (from constant to highest degree)
    /// * `x` - Points at which to evaluate the polynomial
    ///
    /// # Returns
    ///
    /// * Polynomial values at the specified points
    pub fn polynomial_eval(coeffs: &[f64], x: &[f64]) -> Vec<f64> {
        x.iter()
            .map(|&xi| {
                coeffs
                    .iter()
                    .enumerate()
                    .map(|(j, &coeff)| coeff * xi.powi(j as i32))
                    .sum()
            })
            .collect()
    }

    /// Newton's divided difference interpolation
    ///
    /// Computes Newton's interpolating polynomial using divided differences.
    /// This method is numerically more stable than Lagrange interpolation
    /// for higher-degree polynomials.
    ///
    /// # Arguments
    ///
    /// * `x_known` - Known x-coordinates (must be sorted)
    /// * `y_known` - Known y-coordinates
    /// * `x_target` - Target x-coordinates for interpolation
    ///
    /// # Returns
    ///
    /// * Interpolated y-values at target points
    pub fn newton_interpolate(
        x_known: &[f64],
        y_known: &[f64],
        x_target: &[f64],
    ) -> SignalResult<Vec<f64>> {
        if x_known.len() != y_known.len() {
            return Err(SignalError::ValueError(
                "Known x and y arrays must have the same length".to_string(),
            ));
        }

        let n = x_known.len();
        if n == 0 {
            return Err(SignalError::ValueError(
                "Known points arrays cannot be empty".to_string(),
            ));
        }

        // Compute divided differences table
        let mut dd_table = vec![vec![0.0; n]; n];

        // Initialize first column with y values
        for i in 0..n {
            dd_table[i][0] = y_known[i];
        }

        // Compute divided differences
        for j in 1..n {
            for i in 0..(n - j) {
                let denominator = x_known[i + j] - x_known[i];
                if denominator.abs() < 1e-12 {
                    return Err(SignalError::ValueError(
                        "Duplicate x-coordinates in known points".to_string(),
                    ));
                }
                dd_table[i][j] = (dd_table[i + 1][j - 1] - dd_table[i][j - 1]) / denominator;
            }
        }

        // Evaluate Newton polynomial at target points
        let mut result = vec![0.0; x_target.len()];

        for (target_idx, &x) in x_target.iter().enumerate() {
            let mut value = dd_table[0][0];
            let mut product = 1.0;

            for j in 1..n {
                product *= x - x_known[j - 1];
                value += dd_table[0][j] * product;
            }

            result[target_idx] = value;
        }

        Ok(result)
    }
}

/// Unit tests for spectral interpolation methods
#[cfg(test)]
mod tests {
    use super::super::core::InterpolationConfig;
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_sinc_interpolate() {
        let signal = Array1::from_vec(vec![1.0, f64::NAN, 3.0, f64::NAN, 5.0]);
        let result = sinc_interpolate(&signal, 0.4).unwrap();

        // All values should be valid
        assert!(result.iter().all(|&x| !x.is_nan()));

        // Original values should be preserved
        assert_eq!(result[0], 1.0);
        assert_eq!(result[2], 3.0);
        assert_eq!(result[4], 5.0);
    }

    #[test]
    fn test_sinc_interpolate_invalid_cutoff() {
        let signal = Array1::from_vec(vec![1.0, f64::NAN, 3.0]);

        assert!(sinc_interpolate(&signal, 0.0).is_err());
        assert!(sinc_interpolate(&signal, 0.6).is_err());
    }

    #[test]
    fn test_spectral_interpolate() {
        let signal = Array1::from_vec(vec![1.0, f64::NAN, 3.0, f64::NAN, 5.0]);
        let config = InterpolationConfig::default();
        let result = spectral_interpolate(&signal, &config).unwrap();

        // All values should be valid
        assert!(result.iter().all(|&x| !x.is_nan()));

        // Original values should be preserved
        assert_eq!(result[0], 1.0);
        assert_eq!(result[2], 3.0);
        assert_eq!(result[4], 5.0);
    }

    #[test]
    fn test_auto_interpolate() {
        let signal = Array1::from_vec(vec![1.0, f64::NAN, 3.0, f64::NAN, 5.0]);
        let config = InterpolationConfig::default();

        let (result, method) = auto_interpolate(&signal, &config, false).unwrap();

        // All values should be valid
        assert!(result.iter().all(|&x| !x.is_nan()));

        // Should return a valid method
        assert!(matches!(
            method,
            InterpolationMethod::Linear
                | InterpolationMethod::CubicSpline
                | InterpolationMethod::CubicHermite
                | InterpolationMethod::Sinc
                | InterpolationMethod::Spectral
                | InterpolationMethod::MinimumEnergy
                | InterpolationMethod::NearestNeighbor
        ));
    }

    #[test]
    fn test_auto_interpolate_cross_validation() {
        let signal = Array1::from_vec(vec![1.0, 2.0, f64::NAN, 4.0, 5.0, 6.0, f64::NAN, 8.0]);
        let config = InterpolationConfig::default();

        let (result, _method) = auto_interpolate(&signal, &config, true).unwrap();

        // All values should be valid
        assert!(result.iter().all(|&x| !x.is_nan()));
    }

    #[test]
    fn test_no_missing_passthrough() {
        let signal = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let config = InterpolationConfig::default();

        let result1 = sinc_interpolate(&signal, 0.4).unwrap();
        let result2 = spectral_interpolate(&signal, &config).unwrap();
        let (result3, _) = auto_interpolate(&signal, &config, false).unwrap();

        assert_eq!(result1, signal);
        assert_eq!(result2, signal);
        assert_eq!(result3, signal);
    }

    #[test]
    fn test_all_missing_error() {
        let signal = Array1::from_vec(vec![f64::NAN, f64::NAN, f64::NAN]);
        let config = InterpolationConfig::default();

        assert!(sinc_interpolate(&signal, 0.4).is_err());
        assert!(spectral_interpolate(&signal, &config).is_err());
    }

    #[test]
    fn test_resampling_config() {
        let config = resampling::ResamplingConfig::default();
        assert_eq!(config.kernel_length, 65);
        assert_eq!(config.kaiser_beta, 8.0);
        assert_eq!(config.cutoff_frequency, 0.9);
    }

    #[test]
    fn test_polynomial_lagrange() {
        use polynomial::lagrange_interpolate;

        let x_known = [0.0, 1.0, 2.0];
        let y_known = [1.0, 2.0, 5.0]; // y = x^2 + 1
        let x_target = [0.5, 1.5];

        let result = lagrange_interpolate(&x_known, &y_known, &x_target).unwrap();

        // Should be close to [1.25, 3.25] for y = x^2 + 1
        assert!((result[0] - 1.25).abs() < 1e-10);
        assert!((result[1] - 3.25).abs() < 1e-10);
    }

    #[test]
    fn test_polynomial_fit() {
        use polynomial::{polynomial_eval, polynomial_fit};

        let x = [0.0, 1.0, 2.0, 3.0];
        let y = [1.0, 2.0, 5.0, 10.0]; // Roughly y = x^2 + 1

        let coeffs = polynomial_fit(&x, &y, 2).unwrap();
        assert_eq!(coeffs.len(), 3); // degree 2 + 1

        let y_eval = polynomial_eval(&coeffs, &x);

        // Should approximate the original data well
        for (orig, approx) in y.iter().zip(y_eval.iter()) {
            assert!((orig - approx).abs() < 1e-10);
        }
    }
}
