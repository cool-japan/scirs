//! Advanced interpolation methods for signal processing
//!
//! This module provides sophisticated interpolation algorithms including
//! Gaussian process interpolation, Kriging, Radial Basis Functions (RBF),
//! and minimum energy interpolation.

use super::core::InterpolationConfig;
use crate::error::{SignalError, SignalResult};
use ndarray::{Array1, Array2};
use scirs2_linalg::{cholesky, solve, solve_triangular};

/// Applies Gaussian process interpolation to fill missing values in a signal
///
/// Gaussian process interpolation provides a probabilistic approach to interpolation
/// that models the signal as a realization of a Gaussian random process. This method
/// is particularly effective for smooth, continuous data.
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
/// * Interpolated signal using Gaussian process regression
///
/// # Example
///
/// ```rust
/// use ndarray::Array1;
/// use scirs2_signal::interpolate::advanced::gaussian_process_interpolate;
///
/// let mut signal = Array1::from_vec(vec![1.0, f64::NAN, 3.0, f64::NAN, 5.0]);
/// let result = gaussian_process_interpolate(&signal, 2.0, 1.0, 0.01).unwrap();
/// // Result contains probabilistically interpolated values
/// ```
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
    let l = match cholesky(&k_xx.view(), None) {
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

/// Applies Kriging interpolation to fill missing values in a signal
///
/// Kriging is a geostatistical interpolation technique that uses a variogram model
/// to describe the spatial correlation structure. It provides optimal unbiased
/// linear predictions with minimum variance.
///
/// # Arguments
///
/// * `signal` - Input signal with missing values (NaN)
/// * `variogram_model` - Variogram model function (distance -> semivariance)
/// * `config` - Interpolation configuration
///
/// # Returns
///
/// * Interpolated signal using Kriging
///
/// # Example
///
/// ```rust
/// use ndarray::Array1;
/// use scirs2_signal::interpolate::{advanced::kriging_interpolate, core::InterpolationConfig};
///
/// let mut signal = Array1::from_vec(vec![1.0, f64::NAN, 3.0]);
/// let config = InterpolationConfig::default();
/// let variogram = |h: f64| 1.0 - (-h / 2.0).exp(); // Exponential model
/// let result = kriging_interpolate(&signal, variogram, &config).unwrap();
/// ```
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
        let weights = match solve(&gamma.view(), &rhs.view(), None) {
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
/// RBF interpolation represents the signal as a weighted sum of radially symmetric
/// basis functions centered at the known data points. This method is particularly
/// effective for scattered data interpolation.
///
/// # Arguments
///
/// * `signal` - Input signal with missing values (NaN)
/// * `rbf_function` - Radial basis function (distance -> value)
/// * `config` - Interpolation configuration
///
/// # Returns
///
/// * Interpolated signal using RBF interpolation
///
/// # Example
///
/// ```rust
/// use ndarray::Array1;
/// use scirs2_signal::interpolate::{advanced::rbf_interpolate, core::InterpolationConfig};
///
/// let mut signal = Array1::from_vec(vec![1.0, f64::NAN, 3.0]);
/// let config = InterpolationConfig::default();
/// let rbf = |r: f64| (-r * r).exp(); // Gaussian RBF
/// let result = rbf_interpolate(&signal, rbf, &config).unwrap();
/// ```
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
    let weights = match solve(&phi.view(), &y.view(), None) {
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

/// Applies minimum energy interpolation to fill missing values in a signal
///
/// Minimum energy interpolation finds the smoothest possible interpolation by
/// minimizing the energy (sum of squared second derivatives) of the result.
/// This produces very smooth interpolations suitable for continuous data.
///
/// # Arguments
///
/// * `signal` - Input signal with missing values (NaN)
/// * `config` - Interpolation configuration
///
/// # Returns
///
/// * Interpolated signal with minimum energy constraint
///
/// # Example
///
/// ```rust
/// use ndarray::Array1;
/// use scirs2_signal::interpolate::{advanced::minimum_energy_interpolate, core::InterpolationConfig};
///
/// let mut signal = Array1::from_vec(vec![1.0, f64::NAN, f64::NAN, 4.0]);
/// let config = InterpolationConfig::default();
/// let result = minimum_energy_interpolate(&signal, &config).unwrap();
/// // Result contains the smoothest possible interpolation
/// ```
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
    let y_unknown = match solve(&a_reg.view(), &b.view(), None) {
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

/// Generate standard variogram models for Kriging interpolation
pub mod variogram_models {
    /// Spherical variogram model
    ///
    /// The spherical model is one of the most commonly used variogram models.
    /// It reaches its sill at the range parameter.
    ///
    /// # Arguments
    ///
    /// * `range` - The range parameter (distance at which sill is reached)
    /// * `sill` - The sill parameter (maximum semivariance)
    /// * `nugget` - The nugget parameter (semivariance at distance 0)
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
    ///
    /// The exponential model approaches its sill asymptotically.
    ///
    /// # Arguments
    ///
    /// * `range` - The range parameter (practical range is 3 times this value)
    /// * `sill` - The sill parameter (maximum semivariance)
    /// * `nugget` - The nugget parameter (semivariance at distance 0)
    pub fn exponential(range: f64, sill: f64, nugget: f64) -> impl Fn(f64) -> f64 {
        move |h: f64| {
            if h <= 0.0 {
                return 0.0;
            }

            nugget + (sill - nugget) * (1.0 - (-3.0 * h / range).exp())
        }
    }

    /// Gaussian variogram model
    ///
    /// The Gaussian model provides very smooth interpolation.
    ///
    /// # Arguments
    ///
    /// * `range` - The range parameter (practical range is sqrt(3) times this value)
    /// * `sill` - The sill parameter (maximum semivariance)
    /// * `nugget` - The nugget parameter (semivariance at distance 0)
    pub fn gaussian(range: f64, sill: f64, nugget: f64) -> impl Fn(f64) -> f64 {
        move |h: f64| {
            if h <= 0.0 {
                return 0.0;
            }

            nugget + (sill - nugget) * (1.0 - (-9.0 * h * h / (range * range)).exp())
        }
    }

    /// Linear variogram model
    ///
    /// The linear model increases linearly with distance.
    ///
    /// # Arguments
    ///
    /// * `slope` - The slope parameter (rate of increase)
    /// * `nugget` - The nugget parameter (semivariance at distance 0)
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
    ///
    /// The Gaussian RBF is smooth and has compact support.
    ///
    /// # Arguments
    ///
    /// * `epsilon` - Shape parameter controlling the width of the basis function
    pub fn gaussian(epsilon: f64) -> impl Fn(f64) -> f64 {
        move |r: f64| (-epsilon * r * r).exp()
    }

    /// Multiquadric RBF
    ///
    /// The multiquadric RBF is globally supported and often provides good results.
    ///
    /// # Arguments
    ///
    /// * `epsilon` - Shape parameter
    pub fn multiquadric(epsilon: f64) -> impl Fn(f64) -> f64 {
        move |r: f64| (1.0 + epsilon * r * r).sqrt()
    }

    /// Inverse multiquadric RBF
    ///
    /// The inverse multiquadric RBF has global support but decays at infinity.
    ///
    /// # Arguments
    ///
    /// * `epsilon` - Shape parameter
    pub fn inverse_multiquadric(epsilon: f64) -> impl Fn(f64) -> f64 {
        move |r: f64| 1.0 / (1.0 + epsilon * r * r).sqrt()
    }

    /// Thin plate spline RBF
    ///
    /// The thin plate spline minimizes bending energy and is commonly used
    /// for smooth interpolation.
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

/// Unit tests for advanced interpolation methods
#[cfg(test)]
mod tests {
    use super::*;
    use crate::interpolate::core::InterpolationConfig;
    use ndarray::Array1;

    #[test]
    fn test_gaussian_process_interpolate() {
        let signal = Array1::from_vec(vec![1.0, f64::NAN, 3.0, f64::NAN, 5.0]);
        let result = gaussian_process_interpolate(&signal, 2.0, 1.0, 0.01).unwrap();

        // All values should be valid
        assert!(result.iter().all(|&x| !x.is_nan()));

        // Original values should be preserved
        assert_eq!(result[0], 1.0);
        assert_eq!(result[2], 3.0);
        assert_eq!(result[4], 5.0);
    }

    #[test]
    fn test_kriging_interpolate() {
        let signal = Array1::from_vec(vec![1.0, f64::NAN, 3.0]);
        let config = InterpolationConfig::default();
        let variogram = |h: f64| 1.0 - (-h / 2.0).exp();

        let result = kriging_interpolate(&signal, variogram, &config).unwrap();

        assert!(result.iter().all(|&x| !x.is_nan()));
        assert_eq!(result[0], 1.0);
        assert_eq!(result[2], 3.0);
    }

    #[test]
    fn test_rbf_interpolate() {
        let signal = Array1::from_vec(vec![1.0, f64::NAN, 3.0]);
        let config = InterpolationConfig::default();
        let rbf = |r: f64| (-r * r).exp();

        let result = rbf_interpolate(&signal, rbf, &config).unwrap();

        assert!(result.iter().all(|&x| !x.is_nan()));
        assert_eq!(result[0], 1.0);
        assert_eq!(result[2], 3.0);
    }

    #[test]
    fn test_minimum_energy_interpolate() {
        let signal = Array1::from_vec(vec![1.0, f64::NAN, f64::NAN, 4.0]);
        let config = InterpolationConfig::default();

        let result = minimum_energy_interpolate(&signal, &config).unwrap();

        assert!(result.iter().all(|&x| !x.is_nan()));
        assert_eq!(result[0], 1.0);
        assert_eq!(result[3], 4.0);

        // Should produce smooth interpolation
        assert!(result[1] > 1.0 && result[1] < 4.0);
        assert!(result[2] > 1.0 && result[2] < 4.0);
        assert!(result[1] < result[2]); // Should be monotonic
    }

    #[test]
    fn test_variogram_models() {
        let spherical = variogram_models::spherical(10.0, 1.0, 0.1);
        assert_eq!(spherical(0.0), 0.0);
        assert!(spherical(5.0) > 0.1);
        assert!(spherical(10.0) <= 1.0);

        let exponential = variogram_models::exponential(10.0, 1.0, 0.1);
        assert_eq!(exponential(0.0), 0.0);
        assert!(exponential(5.0) > 0.1);

        let gaussian = variogram_models::gaussian(10.0, 1.0, 0.1);
        assert_eq!(gaussian(0.0), 0.0);
        assert!(gaussian(5.0) > 0.1);

        let linear = variogram_models::linear(0.1, 0.05);
        assert_eq!(linear(0.0), 0.0);
        assert_eq!(linear(10.0), 1.05);
    }

    #[test]
    fn test_rbf_functions() {
        let gaussian = rbf_functions::gaussian(1.0);
        assert_eq!(gaussian(0.0), 1.0);
        assert!(gaussian(1.0) < 1.0);

        let multiquadric = rbf_functions::multiquadric(1.0);
        assert_eq!(multiquadric(0.0), 1.0);
        assert!(multiquadric(1.0) > 1.0);

        let inverse_multiquadric = rbf_functions::inverse_multiquadric(1.0);
        assert_eq!(inverse_multiquadric(0.0), 1.0);
        assert!(inverse_multiquadric(1.0) < 1.0);

        let thin_plate = rbf_functions::thin_plate_spline();
        assert_eq!(thin_plate(0.0), 0.0);
        assert!(thin_plate(2.0) > 0.0);
    }

    #[test]
    fn test_all_missing_error() {
        let signal = Array1::from_vec(vec![f64::NAN, f64::NAN, f64::NAN]);
        let config = InterpolationConfig::default();

        assert!(gaussian_process_interpolate(&signal, 1.0, 1.0, 0.01).is_err());
        assert!(kriging_interpolate(&signal, |_| 1.0, &config).is_err());
        assert!(rbf_interpolate(&signal, |_| 1.0, &config).is_err());
        assert!(minimum_energy_interpolate(&signal, &config).is_err());
    }

    #[test]
    fn test_no_missing_passthrough() {
        let signal = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let config = InterpolationConfig::default();

        let result1 = gaussian_process_interpolate(&signal, 1.0, 1.0, 0.01).unwrap();
        let result2 = kriging_interpolate(&signal, |_| 1.0, &config).unwrap();
        let result3 = rbf_interpolate(&signal, |_| 1.0, &config).unwrap();
        let result4 = minimum_energy_interpolate(&signal, &config).unwrap();

        assert_eq!(result1, signal);
        assert_eq!(result2, signal);
        assert_eq!(result3, signal);
        assert_eq!(result4, signal);
    }
}
