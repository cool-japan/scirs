// Shared types and data structures for system identification

use crate::lombscargle_enhanced::WindowType;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::Complex64;

#[allow(unused_imports)]
/// Methods for transfer function estimation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TfEstimationMethod {
    /// Least squares in time domain
    LeastSquares,
    /// Frequency domain estimation using spectral methods
    FrequencyDomain,
    /// Instrumental variable method
    InstrumentalVariable,
    /// Subspace-based estimation
    Subspace,
}

/// Methods for frequency response estimation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FreqResponseMethod {
    /// Welch's method using overlapping windows
    Welch,
    /// Simple periodogram
    Periodogram,
    /// H1 estimator (minimize input noise)
    H1,
    /// H2 estimator (minimize output noise)
    H2,
    /// Coherence-weighted estimator
    CoherenceWeighted,
}

/// Configuration for system identification
#[derive(Debug, Clone)]
pub struct SysIdConfig {
    /// Sampling frequency
    pub fs: f64,
    /// Window type for spectral estimation
    pub window: String,
    /// Window overlap for spectral methods (0.0 to 1.0)
    pub overlap: f64,
    /// Number of FFT points for spectral estimation
    pub nfft: Option<usize>,
    /// Regularization parameter for least squares
    pub regularization: Option<f64>,
    /// Maximum number of iterations for iterative methods
    pub max_iterations: usize,
    /// Convergence tolerance for iterative methods
    pub tolerance: f64,
}

impl Default for SysIdConfig {
    fn default() -> Self {
        Self {
            fs: 1.0,
            window: WindowType::Hann.to_string(),
            overlap: 0.5,
            nfft: None,
            regularization: None,
            max_iterations: 100,
            tolerance: 1e-6,
        }
    }
}

/// Result structure for transfer function estimation
#[derive(Debug, Clone)]
pub struct TfEstimationResult {
    /// Estimated transfer function numerator coefficients
    pub numerator: Array1<f64>,
    /// Estimated transfer function denominator coefficients
    pub denominator: Array1<f64>,
    /// Model fit percentage (0-100)
    pub fit_percentage: f64,
    /// Final prediction error variance
    pub error_variance: f64,
    /// Frequency response at estimation frequencies
    pub frequency_response: Option<Array1<Complex64>>,
    /// Frequencies used for estimation
    pub frequencies: Option<Array1<f64>>,
}

/// Result structure for frequency response estimation
#[derive(Debug, Clone)]
pub struct FreqResponseResult {
    /// Estimated frequency response
    pub frequency_response: Array1<Complex64>,
    /// Frequencies
    pub frequencies: Array1<f64>,
    /// Coherence function
    pub coherence: Array1<f64>,
    /// Confidence bounds (if available)
    pub confidence_bounds: Option<Array2<f64>>,
}

/// Structure for AR/ARMA identification results
#[derive(Debug, Clone)]
pub struct ParametricResult {
    /// AR coefficients
    pub ar_coefficients: Array1<f64>,
    /// MA coefficients (if ARMA)
    pub ma_coefficients: Option<Array1<f64>>,
    /// Noise variance
    pub noise_variance: f64,
    /// Reflection coefficients (if available)
    pub reflection_coefficients: Option<Array1<f64>>,
    /// Information criterion value
    pub information_criterion: f64,
    /// Model order
    pub model_order: (usize, usize), // (AR order, MA order)
}

/// Model validation result
#[derive(Debug, Clone)]
pub struct ModelValidation {
    /// Model fit percentage
    pub fit_percentage: f64,
    /// Mean squared error
    pub mse: f64,
    /// R-squared coefficient
    pub r_squared: f64,
    /// Final prediction error
    pub fpe: f64,
    /// Akaike Information Criterion
    pub aic: f64,
    /// Bayesian Information Criterion
    pub bic: f64,
    /// Residual whiteness test p-value
    pub whiteness_test: f64,
    /// Cross-validation error (if performed)
    pub cv_error: Option<f64>,
}
