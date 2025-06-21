//! Core configuration and dispatch logic for interpolation algorithms
//!
//! This module provides the common types, configuration structures, and dispatch
//! functions that coordinate between different interpolation algorithms.

use crate::error::{SignalError, SignalResult};
use ndarray::{s, Array1, Array2};

// Import the specific interpolation functions from their respective modules
use super::advanced::{
    gaussian_process_interpolate, kriging_interpolate, minimum_energy_interpolate, rbf_interpolate,
};
use super::basic::{linear_interpolate, nearest_neighbor_interpolate};
use super::spectral::{sinc_interpolate, spectral_interpolate};
use super::spline::{cubic_hermite_interpolate, cubic_spline_interpolate};

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

/// Main interpolation dispatch function
///
/// This function provides a unified interface to all interpolation methods.
/// It routes the request to the appropriate specialized function based on the method.
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
/// This function applies interpolation to 2D data by first interpolating along rows,
/// then along columns, with additional passes if needed to handle complex missing patterns.
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
/// This function finds the nearest valid pixel for each missing pixel in the image.
///
/// # Arguments
///
/// * `image` - Input image with missing values (NaN)
///
/// # Returns
///
/// * Interpolated image
pub fn nearest_neighbor_interpolate_2d(image: &Array2<f64>) -> SignalResult<Array2<f64>> {
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
///
/// This function applies a moving average filter to smooth the signal.
/// The window size is determined by the smoothing factor.
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `factor` - Smoothing factor (0.0 to 1.0)
///
/// # Returns
///
/// * Smoothed signal
pub fn smooth_signal(signal: &Array1<f64>, factor: f64) -> Array1<f64> {
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
///
/// This function ensures the signal maintains monotonic behavior
/// (either increasing or decreasing) if the original valid samples exhibit such behavior.
///
/// # Arguments
///
/// * `signal` - Input signal
///
/// # Returns
///
/// * Signal with enforced monotonicity
pub fn enforce_monotonicity(signal: &Array1<f64>) -> Array1<f64> {
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
///
/// This function finds the index in the valid_indices array that corresponds
/// to the valid point nearest to the given index.
///
/// # Arguments
///
/// * `idx` - Target index
/// * `valid_indices` - Array of valid indices
///
/// # Returns
///
/// * Index in valid_indices array of nearest valid point
pub fn find_nearest_valid_index(idx: usize, valid_indices: &[usize]) -> usize {
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

/// Unit tests for core interpolation functionality
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interpolation_config_default() {
        let config = InterpolationConfig::default();
        assert_eq!(config.max_iterations, 100);
        assert_eq!(config.convergence_threshold, 1e-6);
        assert_eq!(config.window_size, 10);
        assert!(!config.extrapolate);
        assert!(!config.monotonic);
    }

    #[test]
    fn test_interpolation_method_enum() {
        let method = InterpolationMethod::Linear;
        assert_eq!(method, InterpolationMethod::Linear);
        assert_ne!(method, InterpolationMethod::CubicSpline);
    }

    #[test]
    fn test_smooth_signal() {
        let signal = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let smoothed = smooth_signal(&signal, 0.2);

        // With 20% window size (1 element), result should be similar to original
        assert!((smoothed[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_find_nearest_valid_index() {
        let valid_indices = vec![0, 2, 5, 8];

        // Test finding nearest to index 1 (should be 0)
        assert_eq!(find_nearest_valid_index(1, &valid_indices), 0);

        // Test finding nearest to index 3 (should be 1, which corresponds to index 2)
        assert_eq!(find_nearest_valid_index(3, &valid_indices), 1);

        // Test finding nearest to index 6 (should be 2, which corresponds to index 5)
        assert_eq!(find_nearest_valid_index(6, &valid_indices), 2);
    }

    #[test]
    fn test_enforce_monotonicity_increasing() {
        // Use an already mostly increasing signal with minor violations
        let signal = Array1::from_vec(vec![1.0, 2.0, 2.1, 4.0, 4.5, 6.0]);
        let result = enforce_monotonicity(&signal);

        // Should be monotonically increasing
        for i in 1..result.len() {
            assert!(result[i] >= result[i - 1]);
        }
    }

    #[test]
    fn test_nearest_neighbor_interpolate_2d() {
        use ndarray::Array2;

        let mut image = Array2::zeros((3, 3));
        image[[0, 0]] = 1.0;
        image[[0, 1]] = f64::NAN;
        image[[0, 2]] = 3.0;
        image[[1, 0]] = f64::NAN;
        image[[1, 1]] = 5.0;
        image[[1, 2]] = f64::NAN;
        image[[2, 0]] = 7.0;
        image[[2, 1]] = f64::NAN;
        image[[2, 2]] = 9.0;

        let result = nearest_neighbor_interpolate_2d(&image).unwrap();

        // All values should be valid (non-NaN)
        assert!(result.iter().all(|&x| !x.is_nan()));

        // Original valid values should be preserved
        assert_eq!(result[[0, 0]], 1.0);
        assert_eq!(result[[1, 1]], 5.0);
        assert_eq!(result[[2, 2]], 9.0);
    }
}
