//! Signal interpolation and missing data imputation
//!
//! This module provides a comprehensive suite of interpolation algorithms for filling
//! missing values in signals, including linear, spline, spectral, and advanced statistical methods.
//!
//! # Module Organization
//!
//! The interpolation functionality is organized into focused submodules:
//!
//! - [`core`] - Core configuration types and dispatch functions
//! - [`basic`] - Simple interpolation methods (linear, nearest neighbor)
//! - [`spline`] - Spline-based methods (cubic spline, Hermite)
//! - [`advanced`] - Statistical methods (Gaussian process, Kriging, RBF, minimum energy)
//! - [`spectral`] - Frequency-domain methods (sinc, spectral, auto-selection)
//!
//! # Quick Start
//!
//! For simple interpolation tasks, use the main dispatch function:
//!
//! ```rust
//! use ndarray::Array1;
//! use scirs2_signal::interpolate::{interpolate, InterpolationMethod, InterpolationConfig};
//!
//! let signal = Array1::from_vec(vec![1.0, f64::NAN, 3.0, f64::NAN, 5.0]);
//! let config = InterpolationConfig::default();
//! let result = interpolate(&signal, InterpolationMethod::Linear, &config).unwrap();
//! ```
//!
//! For automatic method selection:
//!
//! ```rust
//! use ndarray::Array1;
//! use scirs2_signal::interpolate::{auto_interpolate, InterpolationConfig};
//!
//! let signal = Array1::from_vec(vec![1.0, f64::NAN, 3.0, f64::NAN, 5.0]);
//! let config = InterpolationConfig::default();
//! let (result, method) = auto_interpolate(&signal, &config, false).unwrap();
//! ```
//!
//! # Algorithm Selection Guide
//!
//! | Method | Best For | Pros | Cons |
//! |--------|----------|------|------|
//! | Linear | Simple, fast interpolation | Fast, stable | Not smooth |
//! | Cubic Spline | Smooth curves | Very smooth | Can overshoot |
//! | PCHIP | Shape-preserving | Preserves monotonicity | More complex |
//! | Gaussian Process | Statistical modeling | Uncertainty quantification | Computationally intensive |
//! | Kriging | Spatial data | Optimal for spatial correlation | Requires variogram model |
//! | RBF | Scattered data | Flexible basis functions | Parameter tuning needed |
//! | Sinc | Bandlimited signals | Optimal for bandlimited | Requires knowledge of bandwidth |
//! | Spectral | Periodic signals | Good for frequency content | Iterative process |

// Re-export all submodules
pub mod advanced;
pub mod basic;
pub mod core;
pub mod spectral;
pub mod spline;

// Re-export all main types and functions for API compatibility
pub use core::{
    enforce_monotonicity, find_nearest_valid_index, interpolate, interpolate_2d,
    nearest_neighbor_interpolate_2d, smooth_signal, InterpolationConfig, InterpolationMethod,
};

pub use basic::{linear_interpolate, nearest_neighbor_interpolate};

pub use spline::{cubic_hermite_interpolate, cubic_spline_interpolate};

pub use advanced::{
    gaussian_process_interpolate, kriging_interpolate, minimum_energy_interpolate, rbf_functions,
    rbf_interpolate, variogram_models,
};

pub use spectral::{
    auto_interpolate, polynomial, resampling, sinc_interpolate, spectral_interpolate,
};

// Re-export the comprehensive variogram and RBF function collections
pub use advanced::variogram_models::{
    exponential as exponential_variogram, gaussian as gaussian_variogram,
    linear as linear_variogram, spherical as spherical_variogram,
};

pub use advanced::rbf_functions::{
    gaussian as gaussian_rbf, inverse_multiquadric as inverse_multiquadric_rbf,
    multiquadric as multiquadric_rbf, thin_plate_spline as thin_plate_spline_rbf,
};

// Re-export resampling utilities
pub use spectral::resampling::{sinc_resample, ResamplingConfig};

// Re-export polynomial utilities
pub use spectral::polynomial::{
    lagrange_interpolate, newton_interpolate, polynomial_eval, polynomial_fit,
};

/// Convenience function for linear interpolation (most commonly used)
///
/// This is a direct alias to [`basic::linear_interpolate`] for convenience.
///
/// # Arguments
///
/// * `signal` - Input signal with missing values (NaN)
///
/// # Returns
///
/// * Linearly interpolated signal
///
/// # Example
///
/// ```rust
/// use ndarray::Array1;
/// use scirs2_signal::interpolate::linear;
///
/// let signal = Array1::from_vec(vec![1.0, f64::NAN, 3.0]);
/// let result = linear(&signal).unwrap();
/// assert_eq!(result[1], 2.0); // Linear interpolation between 1.0 and 3.0
/// ```
pub fn linear(signal: &ndarray::Array1<f64>) -> crate::error::SignalResult<ndarray::Array1<f64>> {
    linear_interpolate(signal)
}

/// Convenience function for cubic spline interpolation
///
/// This function uses default configuration for cubic spline interpolation.
///
/// # Arguments
///
/// * `signal` - Input signal with missing values (NaN)
///
/// # Returns
///
/// * Cubic spline interpolated signal
///
/// # Example
///
/// ```rust
/// use ndarray::Array1;
/// use scirs2_signal::interpolate::cubic_spline;
///
/// let signal = Array1::from_vec(vec![1.0, f64::NAN, f64::NAN, 4.0]);
/// let result = cubic_spline(&signal).unwrap();
/// // Result contains smooth interpolated values
/// ```
pub fn cubic_spline(
    signal: &ndarray::Array1<f64>,
) -> crate::error::SignalResult<ndarray::Array1<f64>> {
    let config = InterpolationConfig::default();
    cubic_spline_interpolate(signal, &config)
}

/// Convenience function for automatic interpolation method selection
///
/// This function automatically selects the best interpolation method based on smoothness criteria.
///
/// # Arguments
///
/// * `signal` - Input signal with missing values (NaN)
///
/// # Returns
///
/// * Tuple of (interpolated signal, selected method)
///
/// # Example
///
/// ```rust
/// use ndarray::Array1;
/// use scirs2_signal::interpolate::auto;
///
/// let signal = Array1::from_vec(vec![1.0, f64::NAN, 3.0, f64::NAN, 5.0]);
/// let (result, method) = auto(&signal).unwrap();
/// println!("Selected method: {:?}", method);
/// ```
pub fn auto(
    signal: &ndarray::Array1<f64>,
) -> crate::error::SignalResult<(ndarray::Array1<f64>, InterpolationMethod)> {
    let config = InterpolationConfig::default();
    auto_interpolate(signal, &config, false)
}

/// Builder pattern for configuring interpolation
///
/// This struct provides a fluent interface for configuring interpolation parameters.
///
/// # Example
///
/// ```rust
/// use ndarray::Array1;
/// use scirs2_signal::interpolate::{InterpolationBuilder, InterpolationMethod};
///
/// let signal = Array1::from_vec(vec![1.0, f64::NAN, 3.0]);
/// let result = InterpolationBuilder::new()
///     .method(InterpolationMethod::CubicSpline)
///     .smoothing(true)
///     .smoothing_factor(0.3)
///     .monotonic(true)
///     .interpolate(&signal)
///     .unwrap();
/// ```
pub struct InterpolationBuilder {
    config: InterpolationConfig,
    method: InterpolationMethod,
}

impl InterpolationBuilder {
    /// Creates a new interpolation builder with default settings
    pub fn new() -> Self {
        Self {
            config: InterpolationConfig::default(),
            method: InterpolationMethod::Linear,
        }
    }

    /// Sets the interpolation method
    pub fn method(mut self, method: InterpolationMethod) -> Self {
        self.method = method;
        self
    }

    /// Sets the maximum number of iterations for iterative methods
    pub fn max_iterations(mut self, max_iterations: usize) -> Self {
        self.config.max_iterations = max_iterations;
        self
    }

    /// Sets the convergence threshold for iterative methods
    pub fn convergence_threshold(mut self, threshold: f64) -> Self {
        self.config.convergence_threshold = threshold;
        self
    }

    /// Sets the regularization parameter
    pub fn regularization(mut self, regularization: f64) -> Self {
        self.config.regularization = regularization;
        self
    }

    /// Sets the window size for local methods
    pub fn window_size(mut self, window_size: usize) -> Self {
        self.config.window_size = window_size;
        self
    }

    /// Enables or disables extrapolation beyond boundaries
    pub fn extrapolate(mut self, extrapolate: bool) -> Self {
        self.config.extrapolate = extrapolate;
        self
    }

    /// Enables or disables monotonicity constraints
    pub fn monotonic(mut self, monotonic: bool) -> Self {
        self.config.monotonic = monotonic;
        self
    }

    /// Enables or disables smoothing
    pub fn smoothing(mut self, smoothing: bool) -> Self {
        self.config.smoothing = smoothing;
        self
    }

    /// Sets the smoothing factor
    pub fn smoothing_factor(mut self, factor: f64) -> Self {
        self.config.smoothing_factor = factor;
        self
    }

    /// Enables or disables frequency-domain constraints
    pub fn frequency_constraint(mut self, constraint: bool) -> Self {
        self.config.frequency_constraint = constraint;
        self
    }

    /// Sets the cutoff frequency for bandlimited signals
    pub fn cutoff_frequency(mut self, cutoff: f64) -> Self {
        self.config.cutoff_frequency = cutoff;
        self
    }

    /// Performs interpolation with the configured settings
    pub fn interpolate(
        self,
        signal: &ndarray::Array1<f64>,
    ) -> crate::error::SignalResult<ndarray::Array1<f64>> {
        interpolate(signal, self.method, &self.config)
    }

    /// Performs 2D interpolation with the configured settings
    pub fn interpolate_2d(
        self,
        image: &ndarray::Array2<f64>,
    ) -> crate::error::SignalResult<ndarray::Array2<f64>> {
        interpolate_2d(image, self.method, &self.config)
    }

    /// Performs automatic method selection and interpolation
    pub fn auto_interpolate(
        self,
        signal: &ndarray::Array1<f64>,
        cross_validation: bool,
    ) -> crate::error::SignalResult<(ndarray::Array1<f64>, InterpolationMethod)> {
        auto_interpolate(signal, &self.config, cross_validation)
    }
}

impl Default for InterpolationBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Collection of all available interpolation methods for easy discovery
pub struct InterpolationMethods;

impl InterpolationMethods {
    /// All available interpolation methods
    pub const ALL: &'static [InterpolationMethod] = &[
        InterpolationMethod::Linear,
        InterpolationMethod::CubicSpline,
        InterpolationMethod::CubicHermite,
        InterpolationMethod::GaussianProcess,
        InterpolationMethod::Sinc,
        InterpolationMethod::Spectral,
        InterpolationMethod::MinimumEnergy,
        InterpolationMethod::Kriging,
        InterpolationMethod::RBF,
        InterpolationMethod::NearestNeighbor,
    ];

    /// Basic interpolation methods (fast, simple)
    pub const BASIC: &'static [InterpolationMethod] = &[
        InterpolationMethod::Linear,
        InterpolationMethod::NearestNeighbor,
    ];

    /// Spline-based methods (smooth curves)
    pub const SPLINE: &'static [InterpolationMethod] = &[
        InterpolationMethod::CubicSpline,
        InterpolationMethod::CubicHermite,
    ];

    /// Advanced statistical methods
    pub const ADVANCED: &'static [InterpolationMethod] = &[
        InterpolationMethod::GaussianProcess,
        InterpolationMethod::Kriging,
        InterpolationMethod::RBF,
        InterpolationMethod::MinimumEnergy,
    ];

    /// Frequency-domain methods
    pub const SPECTRAL: &'static [InterpolationMethod] =
        &[InterpolationMethod::Sinc, InterpolationMethod::Spectral];
}

/// Unit tests for the unified interpolation API
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_convenience_functions() {
        let signal = Array1::from_vec(vec![1.0, f64::NAN, 3.0]);

        let result1 = linear(&signal).unwrap();
        assert_eq!(result1[1], 2.0);

        let result2 = cubic_spline(&signal).unwrap();
        assert!(!result2[1].is_nan());

        let (result3, _method) = auto(&signal).unwrap();
        assert!(!result3[1].is_nan());
    }

    #[test]
    fn test_interpolation_builder() {
        let signal = Array1::from_vec(vec![1.0, f64::NAN, 3.0]);

        let result = InterpolationBuilder::new()
            .method(InterpolationMethod::Linear)
            .smoothing(false)
            .extrapolate(false)
            .interpolate(&signal)
            .unwrap();

        assert_eq!(result[1], 2.0);
    }

    #[test]
    fn test_builder_methods() {
        let builder = InterpolationBuilder::new()
            .method(InterpolationMethod::CubicSpline)
            .max_iterations(200)
            .convergence_threshold(1e-8)
            .regularization(1e-5)
            .window_size(15)
            .extrapolate(true)
            .monotonic(true)
            .smoothing(true)
            .smoothing_factor(0.2)
            .frequency_constraint(true)
            .cutoff_frequency(0.4);

        // Builder should be configured correctly
        assert_eq!(builder.method, InterpolationMethod::CubicSpline);
        assert_eq!(builder.config.max_iterations, 200);
        assert_eq!(builder.config.convergence_threshold, 1e-8);
    }

    #[test]
    fn test_interpolation_methods_collections() {
        assert_eq!(InterpolationMethods::ALL.len(), 10);
        assert_eq!(InterpolationMethods::BASIC.len(), 2);
        assert_eq!(InterpolationMethods::SPLINE.len(), 2);
        assert_eq!(InterpolationMethods::ADVANCED.len(), 4);
        assert_eq!(InterpolationMethods::SPECTRAL.len(), 2);
    }

    #[test]
    fn test_api_compatibility() {
        let signal = Array1::from_vec(vec![1.0, f64::NAN, 3.0, f64::NAN, 5.0]);
        let config = InterpolationConfig::default();

        // Test that all main API functions work
        let result1 = interpolate(&signal, InterpolationMethod::Linear, &config).unwrap();
        let result2 = linear_interpolate(&signal).unwrap();
        let result3 = cubic_spline_interpolate(&signal, &config).unwrap();
        let result4 = sinc_interpolate(&signal, 0.4).unwrap();
        let (result5, _) = auto_interpolate(&signal, &config, false).unwrap();

        // All results should have no NaN values
        assert!(result1.iter().all(|&x| !x.is_nan()));
        assert!(result2.iter().all(|&x| !x.is_nan()));
        assert!(result3.iter().all(|&x| !x.is_nan()));
        assert!(result4.iter().all(|&x| !x.is_nan()));
        assert!(result5.iter().all(|&x| !x.is_nan()));
    }

    #[test]
    fn test_variogram_rbf_exports() {
        // Test that exported variogram models work
        let spherical = spherical_variogram(10.0, 1.0, 0.1);
        assert_eq!(spherical(0.0), 0.0);

        let gaussian_rbf_fn = gaussian_rbf(1.0);
        assert_eq!(gaussian_rbf_fn(0.0), 1.0);
    }

    #[test]
    fn test_resampling_polynomial_exports() {
        // Test resampling config
        let config = ResamplingConfig::default();
        assert_eq!(config.kernel_length, 65);

        // Test polynomial fitting
        let x = [0.0, 1.0, 2.0];
        let y = [1.0, 2.0, 5.0];
        let coeffs = polynomial_fit(&x, &y, 2).unwrap();
        assert_eq!(coeffs.len(), 3);
    }
}
