//! Type definitions for advanced-enhanced parametric spectral estimation
//!
//! This module contains all the data structures, enums, and configuration types
//! used throughout the parametric spectral estimation framework.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::Complex64;

/// Advanced-enhanced ARMA estimation result with comprehensive diagnostics
#[derive(Debug, Clone)]
pub struct AdvancedEnhancedARMAResult {
    /// AR coefficients [1, a1, a2, ..., ap]
    pub ar_coeffs: Array1<f64>,
    /// MA coefficients [1, b1, b2, ..., bq]
    pub ma_coeffs: Array1<f64>,
    /// Noise variance estimate
    pub noise_variance: f64,
    /// Model residuals
    pub residuals: Array1<f64>,
    /// Convergence information
    pub convergence_info: ConvergenceInfo,
    /// Model diagnostics
    pub diagnostics: ModelDiagnostics,
    /// Computational statistics
    pub performance_stats: PerformanceStats,
}

/// Convergence information for iterative algorithms
#[derive(Debug, Clone)]
pub struct ConvergenceInfo {
    pub converged: bool,
    pub iterations: usize,
    pub final_residual: f64,
    pub convergence_history: Vec<f64>,
    pub method_used: String,
}

/// Comprehensive model diagnostics
#[derive(Debug, Clone)]
pub struct ModelDiagnostics {
    /// Model stability (roots inside unit circle)
    pub is_stable: bool,
    /// Condition number of coefficient matrix
    pub condition_number: f64,
    /// Akaike Information Criterion
    pub aic: f64,
    /// Bayesian Information Criterion
    pub bic: f64,
    /// Likelihood value
    pub log_likelihood: f64,
    /// Prediction error variance
    pub prediction_error_variance: f64,
    /// Residual autocorrelation (Ljung-Box test p-value)
    pub ljung_box_p_value: Option<f64>,
}

/// Performance statistics for SIMD operations
#[derive(Debug, Clone)]
pub struct PerformanceStats {
    pub total_time_ms: f64,
    pub simd_time_ms: f64,
    pub parallel_time_ms: f64,
    pub memory_usage_mb: f64,
    pub simd_utilization: f64,
}

/// Configuration for advanced-enhanced ARMA estimation
#[derive(Debug, Clone)]
pub struct AdvancedEnhancedConfig {
    /// Maximum iterations for iterative methods
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Use SIMD acceleration
    pub use_simd: bool,
    /// Use parallel processing
    pub use_parallel: bool,
    /// Parallel processing threshold
    pub parallel_threshold: usize,
    /// Memory optimization mode
    pub memory_optimized: bool,
    /// Regularization parameter for numerical stability
    pub regularization: f64,
    /// Enable detailed diagnostics
    pub detailed_diagnostics: bool,
}

impl Default for AdvancedEnhancedConfig {
    fn default() -> Self {
        Self {
            max_iterations: 500,
            tolerance: 1e-10,
            use_simd: true,
            use_parallel: true,
            parallel_threshold: 2048,
            memory_optimized: false,
            regularization: 1e-12,
            detailed_diagnostics: true,
        }
    }
}

/// Configuration for adaptive AR estimation
#[derive(Debug, Clone)]
pub struct AdaptiveARConfig {
    /// Initial AR order
    pub initial_order: usize,
    /// Maximum AR order to consider
    pub max_order: usize,
    /// Order selection criterion
    pub order_selection: OrderSelectionCriterion,
    /// Adaptation window size
    pub adaptation_window: usize,
    /// Overlap between consecutive windows
    pub overlap_ratio: f64,
    /// Forgetting factor for recursive updates
    pub forgetting_factor: f64,
    /// Use parallel processing
    pub use_parallel: bool,
    /// SIMD acceleration
    pub use_simd: bool,
}

/// Configuration for robust parametric estimation
#[derive(Debug, Clone)]
pub struct RobustParametricConfig {
    /// AR order
    pub ar_order: usize,
    /// MA order
    pub ma_order: usize,
    /// Scale estimator for robust estimation
    pub scale_estimator: ScaleEstimator,
    /// Robust weight function
    pub weight_function: RobustWeightFunction,
    /// Maximum iterations for robust estimation
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Tuning constant for robust estimation
    pub tuning_constant: f64,
    /// Outlier detection threshold
    pub outlier_threshold: f64,
}

/// Scale estimator types for robust estimation
#[derive(Debug, Clone, PartialEq)]
pub enum ScaleEstimator {
    MAD,
    Rousseeuw,
    Huber,
    Bisquare,
}

/// Robust weight function types
#[derive(Debug, Clone)]
pub enum RobustWeightFunction {
    Huber,
    Bisquare,
    Andrews,
    Hampel,
}

/// Configuration for high-resolution spectral estimation
#[derive(Debug, Clone)]
pub struct HighResolutionConfig {
    /// Method to use for high-resolution estimation
    pub method: HighResolutionMethod,
    /// Order of the model
    pub order: usize,
    /// Number of frequency bins
    pub frequency_bins: usize,
    /// Frequency range (normalized)
    pub frequency_range: (f64, f64),
    /// Forward-backward averaging
    pub forward_backward: bool,
    /// Singular value threshold for MUSIC
    pub svd_threshold: f64,
}

/// High-resolution spectral estimation methods
#[derive(Debug, Clone, PartialEq)]
pub enum HighResolutionMethod {
    MUSIC,
    ESPRIT,
    Pisarenko,
    EigenFilter,
    MinNorm,
}

/// Configuration for multitaper parametric estimation
#[derive(Debug, Clone)]
pub struct MultitaperParametricConfig {
    /// Number of tapers
    pub num_tapers: usize,
    /// Time-bandwidth product
    pub time_bandwidth: f64,
    /// AR order for each taper
    pub ar_order: usize,
    /// Method for combining spectral estimates
    pub combination_method: CombinationMethod,
    /// Bias correction
    pub bias_correction: bool,
    /// Jackknife variance estimation
    pub jackknife_variance: bool,
}

/// Methods for combining multitaper spectral estimates
#[derive(Debug, Clone)]
pub enum CombinationMethod {
    ArithmeticMean,
    GeometricMean,
    MedianCombination,
    WeightedAverage,
    AdaptiveWeighting,
}

/// Order selection criteria
#[derive(Debug, Clone, PartialEq)]
pub enum OrderSelectionCriterion {
    AIC,
    BIC,
    MDL,
    FPE,
    CAT,
}

/// Result structure for adaptive AR estimation
#[derive(Debug, Clone)]
pub struct AdaptiveARResult {
    /// Time-varying AR coefficients
    pub ar_coeffs_time: Array2<f64>,
    /// Time-varying order selection
    pub orders: Array1<usize>,
    /// Time vector
    pub time_vector: Array1<f64>,
    /// Prediction error variance over time
    pub error_variance: Array1<f64>,
    /// Spectral estimates over time and frequency
    pub spectral_estimates: Array2<f64>,
    /// Frequency vector
    pub frequencies: Array1<f64>,
    /// Convergence information for each time step
    pub convergence_info: Vec<ConvergenceInfo>,
}

/// Result structure for robust parametric estimation
#[derive(Debug, Clone)]
pub struct RobustParametricResult {
    /// Robust AR coefficients
    pub ar_coeffs: Array1<f64>,
    /// Robust MA coefficients
    pub ma_coeffs: Array1<f64>,
    /// Robust scale estimate
    pub robust_scale: f64,
    /// Outlier weights
    pub outlier_weights: Array1<f64>,
    /// Detected outliers
    pub outliers: Array1<bool>,
    /// Standard ARMA result for comparison
    pub standard_result: AdvancedEnhancedARMAResult,
    /// Convergence information
    pub convergence_info: ConvergenceInfo,
}

/// Result structure for high-resolution estimation
#[derive(Debug, Clone)]
pub struct HighResolutionResult {
    /// Frequency estimates
    pub frequency_estimates: Array1<f64>,
    /// Amplitude estimates
    pub amplitude_estimates: Array1<Complex64>,
    /// Phase estimates
    pub phase_estimates: Array1<f64>,
    /// Power spectral density
    pub power_spectrum: Array1<f64>,
    /// Frequency vector for spectrum
    pub frequencies: Array1<f64>,
    /// Noise subspace dimension
    pub noise_subspace_dim: usize,
    /// Signal subspace dimension
    pub signal_subspace_dim: usize,
    /// Eigenvalues
    pub eigenvalues: Array1<f64>,
}

/// Spectral estimate structure
#[derive(Debug, Clone)]
pub struct SpectralEstimate {
    /// Power spectral density
    pub psd: Array1<f64>,
    /// Frequency vector
    pub frequencies: Array1<f64>,
    /// Method used for estimation
    pub method: String,
    /// Model order
    pub order: usize,
    /// Confidence intervals (if available)
    pub confidence_intervals: Option<Array2<f64>>,
}

/// Result structure for multitaper parametric estimation
#[derive(Debug, Clone)]
pub struct MultitaperParametricResult {
    /// Combined spectral estimate
    pub combined_spectrum: SpectralEstimate,
    /// Individual taper estimates
    pub individual_estimates: Vec<SpectralEstimate>,
    /// Coherence estimates between tapers
    pub coherence_estimates: Array1<f64>,
    /// Variance estimates (if jackknife used)
    pub variance_estimates: Option<Array1<f64>>,
    /// Degrees of freedom
    pub degrees_of_freedom: f64,
    /// Bias correction factors
    pub bias_correction_factors: Array1<f64>,
}

impl Default for AdaptiveARConfig {
    fn default() -> Self {
        Self {
            initial_order: 10,
            max_order: 50,
            order_selection: OrderSelectionCriterion::AIC,
            adaptation_window: 256,
            overlap_ratio: 0.5,
            forgetting_factor: 0.99,
            use_parallel: true,
            use_simd: true,
        }
    }
}

impl Default for RobustParametricConfig {
    fn default() -> Self {
        Self {
            ar_order: 10,
            ma_order: 5,
            scale_estimator: ScaleEstimator::MAD,
            weight_function: RobustWeightFunction::Huber,
            max_iterations: 100,
            tolerance: 1e-6,
            tuning_constant: 1.345,
            outlier_threshold: 2.5,
        }
    }
}

impl Default for HighResolutionConfig {
    fn default() -> Self {
        Self {
            method: HighResolutionMethod::MUSIC,
            order: 20,
            frequency_bins: 1024,
            frequency_range: (0.0, 0.5),
            forward_backward: true,
            svd_threshold: 1e-10,
        }
    }
}

impl Default for MultitaperParametricConfig {
    fn default() -> Self {
        Self {
            num_tapers: 7,
            time_bandwidth: 4.0,
            ar_order: 20,
            combination_method: CombinationMethod::ArithmeticMean,
            bias_correction: true,
            jackknife_variance: true,
        }
    }
}
