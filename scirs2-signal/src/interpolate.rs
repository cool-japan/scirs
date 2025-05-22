// Missing data interpolation module
//
// This module implements various techniques for interpolating missing data in signals,
// including linear, spline, Gaussian process, spectral, and iterative methods.

use crate::error::{SignalError, SignalResult};
use ndarray::{s, Array1, Array2};
use rustfft::{num_complex::Complex, FftPlanner};
use scirs2_linalg::{cholesky, solve, solve_triangular};
use std::f64::consts::PI;

/// Configuration for interpolation algorithms
#[derive(Debug, Clone)]
pub struct InterpolationConfig {
    /// Maximum number of iterations for iterative methods
    pub max_iterations: usize,
    /// Convergence threshold for iterative methods
    pub convergence_threshold: f64,
    /// Regularization parameter for solving systems
    pub regularization: f64,
    /// Window size for local methods
    pub window_size: usize,
    /// Whether to extrapolate beyond boundaries
    pub extrapolate: bool,
    /// Whether to enforce monotonicity constraints
    pub monotonic: bool,
    /// Whether to apply smoothing
    pub smoothing: bool,
    /// Smoothing factor
    pub smoothing_factor: f64,
    /// Whether to use frequency-domain constraints
    pub frequency_constraint: bool,
    /// Cutoff frequency ratio for bandlimited signals
    pub cutoff_frequency: f64,
}

impl Default for InterpolationConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            convergence_threshold: 1e-6,
            regularization: 1e-6,
            window_size: 10,
            extrapolate: false,
            monotonic: false,
            smoothing: false,
            smoothing_factor: 0.5,
            frequency_constraint: false,
            cutoff_frequency: 0.5,
        }
    }
}

/// Interpolation methods
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InterpolationMethod {
    /// Linear interpolation
    Linear,
    /// Cubic spline interpolation
    CubicSpline,
    /// Cubic Hermite spline interpolation (PCHIP)
    CubicHermite,
    /// Gaussian process interpolation
    GaussianProcess,
    /// Sinc interpolation (for bandlimited signals)
    Sinc,
    /// Spectral interpolation (FFT-based)
    Spectral,
    /// Minimum energy interpolation
    MinimumEnergy,
    /// Kriging interpolation
    Kriging,
    /// Radial basis function interpolation
    RBF,
    /// Nearest neighbor interpolation
    NearestNeighbor,
}

/// Applies linear interpolation to fill missing values in a signal
///
/// # Arguments
///
/// * `signal` - Input signal with missing values (NaN)
///
/// # Returns
///
/// * Interpolated signal
pub fn linear_interpolate(signal: &Array1<f64>) -> SignalResult<Array1<f64>> {
    let n = signal.len();

    // Check if input has any missing values
    let has_missing = signal.iter().any(|&x| x.is_nan());
    if !has_missing {
        return Ok(signal.clone());
    }

    let mut result = signal.clone();

    // Find all non-missing points
    let mut valid_indices = Vec::new();
    let mut valid_values = Vec::new();

    for i in 0..n {
        if !signal[i].is_nan() {
            valid_indices.push(i);
            valid_values.push(signal[i]);
        }
    }

    if valid_indices.is_empty() {
        return Err(SignalError::ValueError(
            "All values are missing in the input signal".to_string(),
        ));
    }

    if valid_indices.len() == 1 {
        // If only one valid point, fill with that value
        let value = valid_values[0];
        for i in 0..n {
            result[i] = value;
        }
        return Ok(result);
    }

    // Interpolate missing values
    for i in 0..n {
        if signal[i].is_nan() {
            // Find the valid points surrounding the missing value
            let mut left_idx = None;
            let mut right_idx = None;

            for (j, &valid_idx) in valid_indices.iter().enumerate() {
                match valid_idx.cmp(&i) {
                    std::cmp::Ordering::Less => left_idx = Some(j),
                    std::cmp::Ordering::Greater => {
                        right_idx = Some(j);
                        break;
                    }
                    std::cmp::Ordering::Equal => {} // No action needed for exact match
                }
            }

            match (left_idx, right_idx) {
                (Some(left), Some(right)) => {
                    // Interpolate between left and right valid points
                    let x1 = valid_indices[left] as f64;
                    let x2 = valid_indices[right] as f64;
                    let y1 = valid_values[left];
                    let y2 = valid_values[right];
                    let x = i as f64;

                    // Linear interpolation formula
                    result[i] = y1 + (y2 - y1) * (x - x1) / (x2 - x1);
                }
                (Some(left), None) => {
                    // Extrapolate using the last valid point
                    result[i] = valid_values[left];
                }
                (None, Some(right)) => {
                    // Extrapolate using the first valid point
                    result[i] = valid_values[right];
                }
                (None, None) => {
                    // This shouldn't happen if we have at least one valid point
                    result[i] = 0.0;
                }
            }
        }
    }

    Ok(result)
}

/// Applies cubic spline interpolation to fill missing values in a signal
///
/// # Arguments
///
/// * `signal` - Input signal with missing values (NaN)
/// * `config` - Interpolation configuration
///
/// # Returns
///
/// * Interpolated signal
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
    let second_derivatives = match solve(&matrix.view(), &rhs.view()) {
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
/// # Arguments
///
/// * `signal` - Input signal with missing values (NaN)
/// * `config` - Interpolation configuration
///
/// # Returns
///
/// * Interpolated signal
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

/// Applies Gaussian process interpolation to fill missing values in a signal
///
/// # Arguments
///
/// * `signal` - Input signal with missing values (NaN)
/// * `kernel_length` - Length scale parameter for RBF kernel
/// * `kernel_sigma` - Signal variance parameter for RBF kernel
/// * `noise_level` - Noise variance parameter
///
/// # Returns
///
/// * Interpolated signal
pub fn gaussian_process_interpolate(
    signal: &Array1<f64>,
    kernel_length: f64,
    kernel_sigma: f64,
    noise_level: f64,
) -> SignalResult<Array1<f64>> {
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

    if missing_indices.is_empty() {
        return Ok(signal.clone());
    }

    // RBF Kernel function
    let kernel = |x1: f64, x2: f64| -> f64 {
        kernel_sigma * (-0.5 * (x1 - x2).powi(2) / (kernel_length * kernel_length)).exp()
    };

    // Create covariance matrix for observed points
    let n_valid = valid_indices.len();
    let mut k_xx = Array2::zeros((n_valid, n_valid));

    for i in 0..n_valid {
        for j in 0..n_valid {
            k_xx[[i, j]] = kernel(valid_indices[i] as f64, valid_indices[j] as f64);

            // Add noise variance to diagonal
            if i == j {
                k_xx[[i, j]] += noise_level;
            }
        }
    }

    // Create cross-covariance matrix between test points and observed points
    let n_missing = missing_indices.len();
    let mut k_star_x = Array2::zeros((n_missing, n_valid));

    for i in 0..n_missing {
        for j in 0..n_valid {
            k_star_x[[i, j]] = kernel(missing_indices[i] as f64, valid_indices[j] as f64);
        }
    }

    // Compute self-covariance matrix for test points
    let mut k_star_star = Array2::zeros((n_missing, n_missing));

    for i in 0..n_missing {
        for j in 0..n_missing {
            k_star_star[[i, j]] = kernel(missing_indices[i] as f64, missing_indices[j] as f64);

            // Add noise variance to diagonal
            if i == j {
                k_star_star[[i, j]] += noise_level;
            }
        }
    }

    // Compute the Cholesky decomposition of K_xx
    let l = match cholesky(&k_xx.view()) {
        Ok(l) => l,
        Err(_) => {
            return Err(SignalError::Compute(
                "Failed to compute Cholesky decomposition of covariance matrix".to_string(),
            ));
        }
    };

    // Solve for alpha = K_xx^(-1) * y
    let y = Array1::from_vec(valid_values);
    let alpha = match solve_triangular(&l.view(), &y.view(), true, false) {
        Ok(a) => a,
        Err(_) => {
            return Err(SignalError::Compute(
                "Failed to solve triangular system in Gaussian process".to_string(),
            ));
        }
    };

    // Predict mean for missing points: mu = K_*x * K_xx^(-1) * y
    let mu = k_star_x.dot(&alpha);

    // Create result by copying input and filling missing values
    let mut result = signal.clone();

    for i in 0..n_missing {
        result[missing_indices[i]] = mu[i];
    }

    Ok(result)
}

/// Applies sinc interpolation to fill missing values in a bandlimited signal
///
/// # Arguments
///
/// * `signal` - Input signal with missing values (NaN)
/// * `cutoff_freq` - Normalized cutoff frequency (0.0 to 0.5)
///
/// # Returns
///
/// * Interpolated signal
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
/// # Arguments
///
/// * `signal` - Input signal with missing values (NaN)
/// * `config` - Interpolation configuration
///
/// # Returns
///
/// * Interpolated signal
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

/// Applies minimum energy interpolation to fill missing values in a signal
///
/// # Arguments
///
/// * `signal` - Input signal with missing values (NaN)
/// * `config` - Interpolation configuration
///
/// # Returns
///
/// * Interpolated signal
pub fn minimum_energy_interpolate(
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

    // Create finite difference matrix for second derivative
    let mut d2 = Array2::zeros((n - 2, n));
    for i in 0..n - 2 {
        d2[[i, i]] = 1.0;
        d2[[i, i + 1]] = -2.0;
        d2[[i, i + 2]] = 1.0;
    }

    // Split the problem into known and unknown parts
    let n_missing = missing_indices.len();
    let n_valid = valid_indices.len();

    // Create selection matrices
    let mut s_known = Array2::zeros((n_valid, n));
    let mut s_unknown = Array2::zeros((n_missing, n));

    for (i, &idx) in valid_indices.iter().enumerate() {
        s_known[[i, idx]] = 1.0;
    }

    for (i, &idx) in missing_indices.iter().enumerate() {
        s_unknown[[i, idx]] = 1.0;
    }

    // Known values vector
    let y_known = Array1::from_vec(valid_values);

    // Calculate the regularization matrix
    let h = d2.t().dot(&d2);

    // Calculate the matrices for the linear system
    let a = s_unknown.dot(&h).dot(&s_unknown.t());
    let b = s_unknown.dot(&h).dot(&s_known.t()).dot(&y_known);

    // Add regularization for stability
    let mut a_reg = a.clone();
    for i in 0..a_reg.dim().0 {
        a_reg[[i, i]] += config.regularization;
    }

    // Solve the system to get the unknown values
    let y_unknown = match solve(&a_reg.view(), &b.view()) {
        Ok(solution) => -solution, // Negative because of how we set up the system
        Err(_) => {
            return Err(SignalError::Compute(
                "Failed to solve minimum energy interpolation system".to_string(),
            ));
        }
    };

    // Create result by copying input and filling missing values
    let mut result = signal.clone();

    for (i, &idx) in missing_indices.iter().enumerate() {
        result[idx] = y_unknown[i];
    }

    Ok(result)
}

/// Applies Kriging interpolation to fill missing values in a signal
///
/// # Arguments
///
/// * `signal` - Input signal with missing values (NaN)
/// * `variogram_model` - Variogram model function (distance -> semivariance)
/// * `config` - Interpolation configuration
///
/// # Returns
///
/// * Interpolated signal
pub fn kriging_interpolate<F>(
    signal: &Array1<f64>,
    variogram_model: F,
    config: &InterpolationConfig,
) -> SignalResult<Array1<f64>>
where
    F: Fn(f64) -> f64,
{
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

    // Number of valid points
    let n_valid = valid_indices.len();

    // Create the variogram matrix
    let mut gamma = Array2::zeros((n_valid + 1, n_valid + 1));

    // Fill variogram matrix
    for i in 0..n_valid {
        for j in 0..n_valid {
            let dist = (valid_indices[i] as f64 - valid_indices[j] as f64).abs();
            gamma[[i, j]] = variogram_model(dist);
        }
    }

    // Add Lagrange multiplier row and column
    for i in 0..n_valid {
        gamma[[i, n_valid]] = 1.0;
        gamma[[n_valid, i]] = 1.0;
    }
    gamma[[n_valid, n_valid]] = 0.0;

    // Add small regularization to diagonal for numerical stability
    for i in 0..n_valid {
        gamma[[i, i]] += config.regularization;
    }

    // Create result array
    let mut result = signal.clone();

    // For each missing point, solve the Kriging system
    for &missing_idx in &missing_indices {
        // Create the right-hand side vector (variogram values to prediction point)
        let mut rhs = Array1::zeros(n_valid + 1);

        for i in 0..n_valid {
            let dist = (valid_indices[i] as f64 - missing_idx as f64).abs();
            rhs[i] = variogram_model(dist);
        }
        rhs[n_valid] = 1.0;

        // Solve the Kriging system
        let weights = match solve(&gamma.view(), &rhs.view()) {
            Ok(w) => w,
            Err(_) => {
                return Err(SignalError::Compute(
                    "Failed to solve Kriging system".to_string(),
                ));
            }
        };

        // Compute the interpolated value
        let mut value = 0.0;
        for i in 0..n_valid {
            value += weights[i] * valid_values[i];
        }

        result[missing_idx] = value;
    }

    Ok(result)
}

/// Applies Radial Basis Function (RBF) interpolation to fill missing values
///
/// # Arguments
///
/// * `signal` - Input signal with missing values (NaN)
/// * `rbf_function` - Radial basis function (distance -> value)
/// * `config` - Interpolation configuration
///
/// # Returns
///
/// * Interpolated signal
pub fn rbf_interpolate<F>(
    signal: &Array1<f64>,
    rbf_function: F,
    config: &InterpolationConfig,
) -> SignalResult<Array1<f64>>
where
    F: Fn(f64) -> f64,
{
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

    // Number of valid points
    let n_valid = valid_indices.len();

    // Create the RBF matrix
    let mut phi = Array2::zeros((n_valid, n_valid));

    // Fill RBF matrix
    for i in 0..n_valid {
        for j in 0..n_valid {
            let dist = (valid_indices[i] as f64 - valid_indices[j] as f64).abs();
            phi[[i, j]] = rbf_function(dist);
        }
    }

    // Add regularization for stability
    for i in 0..n_valid {
        phi[[i, i]] += config.regularization;
    }

    // Solve for RBF weights
    let y = Array1::from_vec(valid_values);
    let weights = match solve(&phi.view(), &y.view()) {
        Ok(w) => w,
        Err(_) => {
            return Err(SignalError::Compute(
                "Failed to solve RBF system".to_string(),
            ));
        }
    };

    // Create result array
    let mut result = signal.clone();

    // For each missing point, compute the RBF interpolation
    for &missing_idx in &missing_indices {
        let mut value = 0.0;

        for i in 0..n_valid {
            let dist = (valid_indices[i] as f64 - missing_idx as f64).abs();
            value += weights[i] * rbf_function(dist);
        }

        result[missing_idx] = value;
    }

    Ok(result)
}

/// Applies nearest neighbor interpolation to fill missing values
///
/// # Arguments
///
/// * `signal` - Input signal with missing values (NaN)
///
/// # Returns
///
/// * Interpolated signal
pub fn nearest_neighbor_interpolate(signal: &Array1<f64>) -> SignalResult<Array1<f64>> {
    let n = signal.len();

    // Check if input has any missing values
    let has_missing = signal.iter().any(|&x| x.is_nan());
    if !has_missing {
        return Ok(signal.clone());
    }

    // Find indices of non-missing points
    let mut valid_indices = Vec::new();
    let mut valid_values = Vec::new();

    for i in 0..n {
        if !signal[i].is_nan() {
            valid_indices.push(i);
            valid_values.push(signal[i]);
        }
    }

    if valid_indices.is_empty() {
        return Err(SignalError::ValueError(
            "All values are missing in the input signal".to_string(),
        ));
    }

    // Create result by filling missing values with nearest valid value
    let mut result = signal.clone();

    for i in 0..n {
        if signal[i].is_nan() {
            let nearest_idx = find_nearest_valid_index(i, &valid_indices);
            result[i] = valid_values[nearest_idx];
        }
    }

    Ok(result)
}

/// Interpolate missing values using the specified method
///
/// # Arguments
///
/// * `signal` - Input signal with missing values (NaN)
/// * `method` - Interpolation method to use
/// * `config` - Interpolation configuration
///
/// # Returns
///
/// * Interpolated signal
pub fn interpolate(
    signal: &Array1<f64>,
    method: InterpolationMethod,
    config: &InterpolationConfig,
) -> SignalResult<Array1<f64>> {
    match method {
        InterpolationMethod::Linear => linear_interpolate(signal),
        InterpolationMethod::CubicSpline => cubic_spline_interpolate(signal, config),
        InterpolationMethod::CubicHermite => cubic_hermite_interpolate(signal, config),
        InterpolationMethod::GaussianProcess => {
            gaussian_process_interpolate(signal, 10.0, 1.0, 1e-3)
        }
        InterpolationMethod::Sinc => sinc_interpolate(signal, config.cutoff_frequency),
        InterpolationMethod::Spectral => spectral_interpolate(signal, config),
        InterpolationMethod::MinimumEnergy => minimum_energy_interpolate(signal, config),
        InterpolationMethod::Kriging => {
            // Use exponential variogram model
            let variogram = |h: f64| -> f64 { 1.0 - (-h / 10.0).exp() };

            kriging_interpolate(signal, variogram, config)
        }
        InterpolationMethod::RBF => {
            // Use Gaussian RBF
            let rbf = |r: f64| -> f64 { (-r * r / (2.0 * 10.0 * 10.0)).exp() };

            rbf_interpolate(signal, rbf, config)
        }
        InterpolationMethod::NearestNeighbor => nearest_neighbor_interpolate(signal),
    }
}

/// Interpolates a 2D array (image) with missing values
///
/// # Arguments
///
/// * `image` - Input image with missing values (NaN)
/// * `method` - Interpolation method to use
/// * `config` - Interpolation configuration
///
/// # Returns
///
/// * Interpolated image
pub fn interpolate_2d(
    image: &Array2<f64>,
    method: InterpolationMethod,
    config: &InterpolationConfig,
) -> SignalResult<Array2<f64>> {
    let (n_rows, n_cols) = image.dim();

    // Check if input has any missing values
    let has_missing = image.iter().any(|&x| x.is_nan());
    if !has_missing {
        return Ok(image.clone());
    }

    // Initialize result
    let mut result = image.clone();

    match method {
        InterpolationMethod::NearestNeighbor => {
            // For nearest neighbor, we can efficiently process the entire image at once
            nearest_neighbor_interpolate_2d(image)
        }
        _ => {
            // First interpolate along rows
            for i in 0..n_rows {
                let row = image.slice(s![i, ..]).to_owned();
                let interpolated_row = interpolate(&row, method, config)?;
                result.slice_mut(s![i, ..]).assign(&interpolated_row);
            }

            // Then interpolate along columns for any remaining missing values
            for j in 0..n_cols {
                let col = result.slice(s![.., j]).to_owned();

                // Only interpolate column if it still has missing values
                if col.iter().any(|&x| x.is_nan()) {
                    let interpolated_col = interpolate(&col, method, config)?;
                    result.slice_mut(s![.., j]).assign(&interpolated_col);
                }
            }

            // Check if there are still missing values and try one more pass if needed
            if result.iter().any(|&x| x.is_nan()) {
                // One more row pass
                for i in 0..n_rows {
                    let row = result.slice(s![i, ..]).to_owned();

                    // Only interpolate row if it still has missing values
                    if row.iter().any(|&x| x.is_nan()) {
                        let interpolated_row = interpolate(&row, method, config)?;
                        result.slice_mut(s![i, ..]).assign(&interpolated_row);
                    }
                }
            }

            Ok(result)
        }
    }
}

/// Applies nearest neighbor interpolation to a 2D image with missing values
///
/// # Arguments
///
/// * `image` - Input image with missing values (NaN)
///
/// # Returns
///
/// * Interpolated image
fn nearest_neighbor_interpolate_2d(image: &Array2<f64>) -> SignalResult<Array2<f64>> {
    let (n_rows, n_cols) = image.dim();

    // Find all valid points
    let mut valid_points = Vec::new();

    for i in 0..n_rows {
        for j in 0..n_cols {
            if !image[[i, j]].is_nan() {
                valid_points.push(((i, j), image[[i, j]]));
            }
        }
    }

    if valid_points.is_empty() {
        return Err(SignalError::ValueError(
            "All values are missing in the input image".to_string(),
        ));
    }

    // Create result with all missing values initially
    let mut result = image.clone();

    // Find nearest valid point for each missing point
    for i in 0..n_rows {
        for j in 0..n_cols {
            if image[[i, j]].is_nan() {
                // Find nearest valid point
                let mut min_dist = f64::MAX;
                let mut min_value = 0.0;

                for &((vi, vj), value) in &valid_points {
                    let dist =
                        ((i as f64 - vi as f64).powi(2) + (j as f64 - vj as f64).powi(2)).sqrt();

                    if dist < min_dist {
                        min_dist = dist;
                        min_value = value;
                    }
                }

                result[[i, j]] = min_value;
            }
        }
    }

    Ok(result)
}

/// Helper function to smooth a signal using moving average
fn smooth_signal(signal: &Array1<f64>, factor: f64) -> Array1<f64> {
    let n = signal.len();
    let window_size = (n as f64 * factor).ceil() as usize;
    let half_window = window_size / 2;

    let mut result = signal.clone();

    for i in 0..n {
        let mut count = 0;
        let mut sum = 0.0;

        let start = i.saturating_sub(half_window);
        let end = (i + half_window + 1).min(n);

        for j in start..end {
            if !signal[j].is_nan() {
                sum += signal[j];
                count += 1;
            }
        }

        if count > 0 {
            result[i] = sum / count as f64;
        }
    }

    result
}

/// Helper function to enforce monotonicity constraints
fn enforce_monotonicity(signal: &Array1<f64>) -> Array1<f64> {
    let n = signal.len();
    let mut result = signal.clone();

    // Find all valid points
    let mut valid_indices = Vec::new();

    for i in 0..n {
        if !signal[i].is_nan() {
            valid_indices.push(i);
        }
    }

    if valid_indices.len() < 2 {
        return result;
    }

    // Check if valid samples are monotonically increasing or decreasing
    let mut increasing = true;
    let mut decreasing = true;

    for i in 1..valid_indices.len() {
        let prev = signal[valid_indices[i - 1]];
        let curr = signal[valid_indices[i]];

        if curr < prev {
            increasing = false;
        }

        if curr > prev {
            decreasing = false;
        }
    }

    // If neither increasing nor decreasing, no monotonicity to enforce
    if !increasing && !decreasing {
        return result;
    }

    // Enforce monotonicity
    if increasing {
        for i in 1..n {
            if result[i] < result[i - 1] {
                result[i] = result[i - 1];
            }
        }
    } else if decreasing {
        for i in 1..n {
            if result[i] > result[i - 1] {
                result[i] = result[i - 1];
            }
        }
    }

    result
}

/// Helper function to find the index of the nearest valid point
fn find_nearest_valid_index(idx: usize, valid_indices: &[usize]) -> usize {
    if valid_indices.is_empty() {
        return 0;
    }

    let mut nearest_idx = 0;
    let mut min_dist = usize::MAX;

    for (i, &valid_idx) in valid_indices.iter().enumerate() {
        let dist = if valid_idx > idx {
            valid_idx - idx
        } else {
            idx - valid_idx
        };

        if dist < min_dist {
            min_dist = dist;
            nearest_idx = i;
        }
    }

    nearest_idx
}

/// Generate standard variogram models for Kriging interpolation
pub mod variogram_models {
    /// Spherical variogram model
    pub fn spherical(range: f64, sill: f64, nugget: f64) -> impl Fn(f64) -> f64 {
        move |h: f64| {
            if h <= 0.0 {
                return 0.0;
            }

            if h >= range {
                return sill;
            }

            let h_norm = h / range;
            nugget + (sill - nugget) * (1.5 * h_norm - 0.5 * h_norm.powi(3))
        }
    }

    /// Exponential variogram model
    pub fn exponential(range: f64, sill: f64, nugget: f64) -> impl Fn(f64) -> f64 {
        move |h: f64| {
            if h <= 0.0 {
                return 0.0;
            }

            nugget + (sill - nugget) * (1.0 - (-3.0 * h / range).exp())
        }
    }

    /// Gaussian variogram model
    pub fn gaussian(range: f64, sill: f64, nugget: f64) -> impl Fn(f64) -> f64 {
        move |h: f64| {
            if h <= 0.0 {
                return 0.0;
            }

            nugget + (sill - nugget) * (1.0 - (-9.0 * h * h / (range * range)).exp())
        }
    }

    /// Linear variogram model
    pub fn linear(slope: f64, nugget: f64) -> impl Fn(f64) -> f64 {
        move |h: f64| {
            if h <= 0.0 {
                return 0.0;
            }

            nugget + slope * h
        }
    }
}

/// Generate standard RBF functions for interpolation
pub mod rbf_functions {
    /// Gaussian RBF
    pub fn gaussian(epsilon: f64) -> impl Fn(f64) -> f64 {
        move |r: f64| (-epsilon * r * r).exp()
    }

    /// Multiquadric RBF
    pub fn multiquadric(epsilon: f64) -> impl Fn(f64) -> f64 {
        move |r: f64| (1.0 + epsilon * r * r).sqrt()
    }

    /// Inverse multiquadric RBF
    pub fn inverse_multiquadric(epsilon: f64) -> impl Fn(f64) -> f64 {
        move |r: f64| 1.0 / (1.0 + epsilon * r * r).sqrt()
    }

    /// Thin plate spline RBF
    pub fn thin_plate_spline() -> impl Fn(f64) -> f64 {
        move |r: f64| {
            if r.abs() < 1e-10 {
                0.0
            } else {
                r * r * r.ln()
            }
        }
    }
}

/// Computes multiple interpolation methods and selects the best one
///
/// # Arguments
///
/// * `signal` - Input signal with missing values (NaN)
/// * `config` - Interpolation configuration
/// * `cross_validation` - Whether to use cross-validation for method selection
///
/// # Returns
///
/// * Tuple containing (interpolated signal, selected method)
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
                let interpolated = interpolate(&temp_signal, method, config)?;

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
        let result = interpolate(signal, best_method, config)?;
        Ok((result, best_method))
    } else {
        // Try all methods and pick the one with the smoothest result
        let mut min_roughness = f64::MAX;
        let mut best_method = InterpolationMethod::Linear;
        let mut best_result = linear_interpolate(signal)?;

        for &method in &methods {
            let interpolated = interpolate(signal, method, config)?;

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
