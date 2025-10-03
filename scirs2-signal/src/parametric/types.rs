//! Type definitions for parametric spectral estimation methods
//!
//! This module contains all the struct and enum definitions used throughout
//! the parametric spectral estimation module, including:
//! - AR/MA/ARMA estimation options and results
//! - Order selection criteria and results
//! - Diagnostic and validation structures
//! - Robust estimation types
//! - State-space and advanced model types

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::Complex64;
use std::collections::HashMap;

/// Method for estimating AR model parameters
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ARMethod {
    /// Yule-Walker method using autocorrelation
    YuleWalker,

    /// Burg method (minimizes forward and backward prediction errors)
    Burg,

    /// Covariance method (uses covariance estimate)
    Covariance,

    /// Modified covariance method (forward and backward predictions)
    ModifiedCovariance,

    /// Least squares method
    LeastSquares,
}

/// Method for selecting the optimal model order
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderSelection {
    /// Akaike Information Criterion
    AIC,

    /// Bayesian Information Criterion (more penalty for model complexity)
    BIC,

    /// Final Prediction Error
    FPE,

    /// Minimum Description Length
    MDL,

    /// Corrected Akaike Information Criterion (for small samples)
    AICc,
}

/// Methods for MA estimation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MAMethod {
    Innovations,
    MaximumLikelihood,
    Durbin,
}

/// Order selection enhancements
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum OrderSelectionCriterion {
    AIC,
    BIC,
    HQC,
    FPE,
    AICc,
    CrossValidation,
    PredictionError,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationMethod {
    LevenbergMarquardt,
    GaussNewton,
    BFGS,
    NelderMead,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InitializationMethod {
    MethodOfMoments,
    Hannan,
    LeastSquares,
    Random,
}

#[derive(Debug, Clone, Copy)]
pub enum RobustWeightFunction {
    Huber,
    Bisquare,
    Andrews,
    Hampel,
}

#[derive(Debug, Clone, Copy)]
pub enum RobustScaleMethod {
    MAD, // Median Absolute Deviation
    Qn,  // Rousseeuw-Croux Qn estimator
    Sn,  // Rousseeuw-Croux Sn estimator
}

/// Options for enhanced ARMA estimation
#[derive(Debug, Clone)]
pub struct ARMAOptions {
    pub max_iterations: usize,
    pub tolerance: f64,
    pub optimization_method: OptimizationMethod,
    pub initial_method: InitializationMethod,
    pub compute_standard_errors: bool,
    pub confidence_level: f64,
    pub learning_rate: f64,
    pub ljung_box_lags: Option<usize>,
    pub arch_lags: Option<usize>,
}

impl Default for ARMAOptions {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            tolerance: 1e-8,
            optimization_method: OptimizationMethod::LevenbergMarquardt,
            initial_method: InitializationMethod::MethodOfMoments,
            compute_standard_errors: true,
            confidence_level: 0.95,
            learning_rate: 0.01,
            ljung_box_lags: None,
            arch_lags: None,
        }
    }
}

/// Enhanced ARMA estimation result with comprehensive diagnostics
#[derive(Debug, Clone)]
pub struct EnhancedARMAResult {
    pub ar_coeffs: Array1<f64>,
    pub ma_coeffs: Array1<f64>,
    pub variance: f64,
    pub likelihood: f64,
    pub aic: f64,
    pub bic: f64,
    pub standard_errors: Option<ARMAStandardErrors>,
    pub confidence_intervals: Option<ARMAConfidenceIntervals>,
    pub residuals: Array1<f64>,
    pub diagnostics: ARMADiagnostics,
    pub validation: ARMAValidation,
    pub convergence_info: ConvergenceInfo,
}

/// MA estimation result
#[derive(Debug, Clone)]
pub struct MAResult {
    pub ma_coeffs: Array1<f64>,
    pub variance: f64,
    pub residuals: Array1<f64>,
    pub likelihood: f64,
}

/// Options for spectrum computation
#[derive(Debug, Clone)]
pub struct SpectrumOptions {
    pub compute_confidence_bands: bool,
    pub confidence_level: f64,
    pub detect_peaks: bool,
    pub peak_threshold: f64,
    pub bootstrap_samples: usize,
}

impl Default for SpectrumOptions {
    fn default() -> Self {
        Self {
            compute_confidence_bands: false,
            confidence_level: 0.95,
            detect_peaks: false,
            peak_threshold: 0.1,
            bootstrap_samples: 1000,
        }
    }
}

/// Enhanced spectrum result with analysis
#[derive(Debug, Clone)]
pub struct EnhancedSpectrumResult {
    pub frequencies: Array1<f64>,
    pub spectrum: Array1<f64>,
    pub confidence_bands: Option<(Array1<f64>, Array1<f64>)>,
    pub pole_zero_analysis: PoleZeroAnalysis,
    pub peaks: Option<Vec<SpectralPeak>>,
    pub metrics: SpectrumMetrics,
}

/// VARMA options and result structures
#[derive(Debug, Clone)]
pub struct VARMAOptions {
    pub max_iterations: usize,
    pub tolerance: f64,
    pub test_cointegration: bool,
    pub compute_impulse_responses: bool,
}

impl Default for VARMAOptions {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            tolerance: 1e-8,
            test_cointegration: false,
            compute_impulse_responses: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct VARMAResult {
    pub ar_coeffs: Array2<f64>,
    pub ma_coeffs: Array2<f64>,
    pub variance_matrix: Array2<f64>,
    pub likelihood: f64,
    pub cointegration_test: Option<CointegrationTest>,
    pub impulse_responses: Option<Array2<f64>>,
}

#[derive(Debug, Clone)]
pub struct OrderSelectionOptions {
    pub use_cross_validation: bool,
    pub cv_folds: usize,
    pub penalty_factor: f64,
    pub stability_weight: f64,
}

impl Default for OrderSelectionOptions {
    fn default() -> Self {
        Self {
            use_cross_validation: true,
            cv_folds: 5,
            penalty_factor: 1.0,
            stability_weight: 0.1,
        }
    }
}

#[derive(Debug, Clone)]
pub struct EnhancedOrderSelectionResult {
    pub best_models: HashMap<OrderSelectionCriterion, OrderSelectionCandidate>,
    pub all_candidates: Vec<OrderSelectionCandidate>,
    pub recommendations: OrderRecommendations,
}

#[derive(Debug, Clone)]
pub struct OrderSelectionCandidate {
    pub arorder: usize,
    pub maorder: usize,
    pub criterion_values: HashMap<OrderSelectionCriterion, f64>,
    pub cv_score: Option<f64>,
    pub stability: StabilityAnalysis,
    pub model_result: EnhancedARMAResult,
}

/// Adaptive estimation structures
#[derive(Debug, Clone)]
pub struct AdaptationOptions {
    pub forgetting_factor: f64,
    pub adaptation_rate: f64,
    pub change_detection_threshold: f64,
    pub buffer_size: usize,
}

impl Default for AdaptationOptions {
    fn default() -> Self {
        Self {
            forgetting_factor: 0.98,
            adaptation_rate: 0.01,
            change_detection_threshold: 3.0,
            buffer_size: 1000,
        }
    }
}

#[derive(Debug)]
pub struct AdaptiveARMAEstimator {
    pub arorder: usize,
    pub maorder: usize,
    pub current_ar_coeffs: Array1<f64>,
    pub current_ma_coeffs: Array1<f64>,
    pub current_variance: f64,
    pub forgetting_factor: f64,
    pub adaptation_rate: f64,
    pub change_detection_threshold: f64,
    pub buffer: CircularBuffer<f64>,
    pub update_count: usize,
    pub last_update_time: std::time::Instant,
}

/// Placeholder structures for comprehensive API
#[derive(Debug, Clone)]
pub struct ARMAStandardErrors {
    pub ar_se: Array1<f64>,
    pub ma_se: Array1<f64>,
    pub variance_se: f64,
}

#[derive(Debug, Clone)]
pub struct ARMAConfidenceIntervals {
    pub ar_ci: Array2<f64>,
    pub ma_ci: Array2<f64>,
    pub variance_ci: (f64, f64),
}

#[derive(Debug, Clone)]
pub struct ARMADiagnostics {
    pub aic: f64,
    pub bic: f64,
    pub ljung_box_test: LjungBoxTest,
    pub jarque_bera_test: JarqueBeraTest,
    pub arch_test: ARCHTest,
}

#[derive(Debug, Clone)]
pub struct ARMAValidation {
    pub residual_autocorrelation: Array1<f64>,
    pub normality_tests: NormalityTests,
    pub heteroskedasticity_tests: HeteroskedasticityTests,
    pub stability_tests: StabilityTests,
}

#[derive(Debug, Clone)]
pub struct ConvergenceInfo {
    pub converged: bool,
    pub iterations: usize,
    pub final_gradient_norm: f64,
    pub final_step_size: f64,
}

#[derive(Debug, Clone)]
pub struct PoleZeroAnalysis {
    pub poles: Vec<Complex64>,
    pub zeros: Vec<Complex64>,
    pub stability_margin: f64,
    pub frequency_peaks: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct SpectralPeak {
    pub frequency: f64,
    pub power: f64,
    pub prominence: f64,
    pub bandwidth: f64,
}

#[derive(Debug, Clone)]
pub struct SpectrumMetrics {
    pub total_power: f64,
    pub peak_frequency: f64,
    pub bandwidth_3db: f64,
    pub spectral_entropy: f64,
}

#[derive(Debug, Clone)]
pub struct CointegrationTest {
    pub test_statistic: f64,
    pub p_value: f64,
    pub cointegrating_vectors: Array2<f64>,
}

#[derive(Debug, Clone)]
pub struct StabilityAnalysis {
    pub is_stable: bool,
    pub stability_margin: f64,
    pub critical_frequencies: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct OrderRecommendations {
    pub recommended_ar: usize,
    pub recommended_ma: usize,
    pub confidence_level: f64,
    pub rationale: String,
}

#[derive(Debug)]
pub struct CircularBuffer<T> {
    buffer: Vec<T>,
    capacity: usize,
    head: usize,
    tail: usize,
    full: bool,
}

impl<T: Clone> CircularBuffer<T> {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(capacity),
            capacity,
            head: 0,
            tail: 0,
            full: false,
        }
    }
}

/// Statistical test result structures
#[derive(Debug, Clone)]
pub struct LjungBoxTest {
    pub statistic: f64,
    pub p_value: f64,
    pub lags: usize,
}

impl Default for LjungBoxTest {
    fn default() -> Self {
        Self {
            statistic: 0.0,
            p_value: 1.0,
            lags: 1,
        }
    }
}

#[derive(Debug, Clone)]
pub struct JarqueBeraTest {
    pub statistic: f64,
    pub p_value: f64,
}

impl Default for JarqueBeraTest {
    fn default() -> Self {
        Self {
            statistic: 0.0,
            p_value: 1.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ARCHTest {
    pub statistic: f64,
    pub p_value: f64,
    pub lags: usize,
}

impl Default for ARCHTest {
    fn default() -> Self {
        Self {
            statistic: 0.0,
            p_value: 1.0,
            lags: 1,
        }
    }
}

#[derive(Debug, Clone)]
pub struct NormalityTests {
    pub jarque_bera: JarqueBeraTest,
    pub kolmogorov_smirnov: f64,
    pub anderson_darling: f64,
}

impl Default for NormalityTests {
    fn default() -> Self {
        Self {
            jarque_bera: Default::default(),
            kolmogorov_smirnov: 1.0,
            anderson_darling: 1.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct HeteroskedasticityTests {
    pub arch_test: ARCHTest,
    pub white_test: f64,
    pub breusch_pagan: f64,
}

impl Default for HeteroskedasticityTests {
    fn default() -> Self {
        Self {
            arch_test: Default::default(),
            white_test: 1.0,
            breusch_pagan: 1.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct StabilityTests {
    pub chow_test: f64,
    pub cusum_test: f64,
    pub recursive_residuals: Array1<f64>,
}

impl Default for StabilityTests {
    fn default() -> Self {
        Self {
            chow_test: 1.0,
            cusum_test: 1.0,
            recursive_residuals: Array1::zeros(1),
        }
    }
}

#[derive(Debug, Clone)]
pub struct RobustEstimationOptions {
    pub max_iterations: usize,
    pub tolerance: f64,
    pub weight_function: RobustWeightFunction,
    pub scale_method: RobustScaleMethod,
    pub outlier_threshold: f64,
    pub breakdown_point: f64,
}

impl Default for RobustEstimationOptions {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            tolerance: 1e-6,
            weight_function: RobustWeightFunction::Huber,
            scale_method: RobustScaleMethod::MAD,
            outlier_threshold: 0.1,
            breakdown_point: 0.5,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RobustARResult {
    pub ar_coefficients: Array1<f64>,
    pub error_variance: f64,
    pub robust_scale: f64,
    pub outlier_indices: Vec<usize>,
    pub outlier_weights: Array1<f64>,
    pub breakdown_point: f64,
    pub efficiency: f64,
    pub iterations_needed: usize,
}

#[derive(Debug, Clone)]
pub struct StateSpaceOptions {
    pub max_em_iterations: usize,
    pub em_tolerance: f64,
    pub initial_process_variance: f64,
    pub initial_observation_variance: f64,
    pub compute_arma_equivalent: bool,
}

impl Default for StateSpaceOptions {
    fn default() -> Self {
        Self {
            max_em_iterations: 100,
            em_tolerance: 1e-6,
            initial_process_variance: 1.0,
            initial_observation_variance: 1.0,
            compute_arma_equivalent: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct StateSpaceParametricResult {
    pub state_transition_matrix: Array2<f64>,
    pub observation_vector: Array1<f64>,
    pub process_noise_covariance: Array2<f64>,
    pub observation_noise_variance: f64,
    pub state_estimates: Array2<f64>,
    pub state_covariances: Vec<Array2<f64>>,
    pub innovations: Array1<f64>,
    pub innovation_covariances: Array1<f64>,
    pub log_likelihood: f64,
    pub aic: f64,
    pub bic: f64,
    pub arma_equivalent: Option<(Array1<f64>, Array1<f64>)>, // (AR, MA) coefficients
    pub convergence_iterations: usize,
}

#[derive(Debug, Clone)]
pub struct FARIMAOptions {
    pub gph_bandwidth: Option<usize>,
    pub truncation_lag: usize,
    pub spectrum_points: usize,
}

impl Default for FARIMAOptions {
    fn default() -> Self {
        Self {
            gph_bandwidth: None, // Will be set automatically based on signal length
            truncation_lag: 100,
            spectrum_points: 512,
        }
    }
}

#[derive(Debug, Clone)]
pub struct FARIMAResult {
    pub ar_coefficients: Array1<f64>,
    pub ma_coefficients: Array1<f64>,
    pub fractional_d: f64,
    pub error_variance: f64,
    pub hurst_exponent: f64,
    pub spectrum: Array1<f64>,
    pub residuals: Array1<f64>,
    pub aic: f64,
    pub bic: f64,
    pub log_likelihood: f64,
    pub fractional_d_standard_error: f64,
}

#[derive(Debug, Clone)]
pub struct VAROptions {
    pub compute_granger_causality: bool,
    pub compute_impulse_responses: bool,
    pub compute_cross_spectrum: bool,
    pub impulse_horizon: usize,
    pub spectrum_points: usize,
}

impl Default for VAROptions {
    fn default() -> Self {
        Self {
            compute_granger_causality: true,
            compute_impulse_responses: true,
            compute_cross_spectrum: true,
            impulse_horizon: 20,
            spectrum_points: 512,
        }
    }
}

#[derive(Debug, Clone)]
pub struct VARResult {
    pub coefficient_matrices: Vec<Array2<f64>>,
    pub error_covariance: Array2<f64>,
    pub residuals: Array2<f64>,
    pub log_likelihood: f64,
    pub aic: f64,
    pub bic: f64,
    pub granger_causality_tests: Option<Array2<f64>>, // P-values matrix
    pub impulse_response_functions: Option<Array2<f64>>,
    pub cross_spectral_density: Option<Array2<Complex64>>,
    pub stability_eigenvalues: Array1<Complex64>,
}

/// Private struct for internal ARMA parameter representation
#[derive(Debug, Clone)]
pub(crate) struct ARMAParameters {
    pub ar_coeffs: Array1<f64>,
    pub ma_coeffs: Array1<f64>,
    pub variance: f64,
    pub noise_variance: f64,
    pub likelihood: f64,
    pub convergence_info: ConvergenceInfo,
}
