use ndarray::s;
// Advanced-enhanced parametric spectral estimation with SIMD acceleration
//
// This module provides high-performance implementations of parametric spectral
// estimation methods with scirs2-core SIMD and parallel processing optimizations.
//
// Key enhancements:
// - SIMD-accelerated matrix operations for AR/ARMA parameter estimation
// - Parallel order selection with cross-validation
// - Enhanced numerical stability and convergence detection
// - Memory-efficient processing for large signals
// - Advanced model validation and diagnostics

use crate::error::{SignalError, SignalResult};
use crate::parametric_advanced::compute_eigendecomposition;
use crate::sysid::{detect_outliers, estimate_robust_scale};
use ndarray::{Array1, Array2, ArrayView1};
use num_complex::Complex64;
use num_traits::{Float, NumCast};
use rand::Rng;
use scirs2_core::parallel_ops::*;
use scirs2_core::simd_ops::{PlatformCapabilities, SimdUnifiedOps};
use scirs2_core::validation::{check_finite, check_positive};
use statrs::statistics::Statistics;

#[allow(unused_imports)]
use crate::parametric::{
    compute_parameter_change, detect_spectral_peaks, update_robust_weights, OrderSelection,
};
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

/// Advanced-enhanced ARMA estimation with SIMD acceleration and advanced numerics
///
/// This function provides state-of-the-art ARMA parameter estimation using:
/// - SIMD-accelerated linear algebra operations
/// - Advanced numerical stability techniques
/// - Parallel processing for large problems
/// - Comprehensive model validation
///
/// # Arguments
///
/// * `signal` - Input time series
/// * `ar_order` - Autoregressive order
/// * `ma_order` - Moving average order  
/// * `config` - Configuration parameters
///
/// # Returns
///
/// * Enhanced ARMA result with diagnostics
///
/// # Examples
///
/// ```
/// use scirs2_signal::parametric_advanced_enhanced::{advanced_enhanced_arma, AdvancedEnhancedConfig};
/// use ndarray::Array1;
///
/// // Generate test signal with two sinusoids plus noise
/// let n = 1024;
/// let fs = 100.0;
/// let t: Array1<f64> = Array1::linspace(0.0, (n-1) as f64 / fs, n);
/// use rand::prelude::*;
/// let mut rng = rand::rng();
///
/// let signal: Array1<f64> = t.mapv(|ti| {
///     (2.0 * PI * 5.0 * ti).sin() +
///     0.5 * (2.0 * PI * 15.0 * ti).sin() +
///     0.1 * rng.gen_range(-1.0..1.0)
/// });
///
/// let config = AdvancedEnhancedConfig::default();
/// let result = advanced_enhanced_arma(&signal..4, 2, &config).unwrap();
///
/// assert!(result.convergence_info.converged);
/// assert!(result.diagnostics.is_stable);
/// assert!(result.noise_variance > 0.0);
/// ```
#[allow(dead_code)]
pub fn advanced_enhanced_arma<T>(
    signal: &Array1<T>,
    ar_order: usize,
    ma_order: usize,
    config: &AdvancedEnhancedConfig,
) -> SignalResult<AdvancedEnhancedARMAResult>
where
    T: Float + NumCast + Send + Sync,
{
    let start_time = std::time::Instant::now();

    // Validate inputs
    if signal.is_empty() {
        return Err(SignalError::ValueError("Input signal is empty".to_string()));
    }

    check_positive(ar_order.max(ma_order), "model_order")?;

    // Convert to f64 for numerical computations
    let signal_f64: Array1<f64> = signal
        .iter()
        .map(|&val| {
            NumCast::from(val).ok_or_else(|| {
                SignalError::ValueError(format!("Could not convert signal value to f64"))
            })
        })
        .collect::<SignalResult<Array1<f64>>>()?;

    // Signal validation - check for finite values
    if signal_f64.iter().any(|&x| !x.is_finite()) {
        return Err(SignalError::ValueError(
            "Signal contains non-finite values".to_string(),
        ));
    }

    let n = signal_f64.len();

    // Validate model _order vs data length
    let min_samples = (ar_order + ma_order) * 5 + 50;
    if n < min_samples {
        return Err(SignalError::ValueError(format!(
            "Signal length {} too short for AR({}) MA({}) model. Minimum length: {}",
            n, ar_order, ma_order, min_samples
        )));
    }

    // Detect SIMD capabilities
    let caps = PlatformCapabilities::detect();
    let use_advanced_simd = config.use_simd && (caps.avx2_available || caps.avx512_available);

    // Initialize performance tracking
    let mut simd_time = 0.0;
    let mut parallel_time = 0.0;

    // Step 1: Initial AR parameter estimation using enhanced Burg method
    let simd_start = std::time::Instant::now();
    let (initial_ar_coeffs, ar_variance) = if use_advanced_simd {
        enhanced_burg_method_simd(&signal_f64, ar_order, config)?
    } else {
        enhanced_burg_method_standard(&signal_f64, ar_order, config)?
    };
    simd_time += simd_start.elapsed().as_secs_f64() * 1000.0;

    // Step 2: Estimate MA parameters if needed
    let (final_ar_coeffs, ma_coeffs, noise_variance, residuals, convergence_info) = if ma_order > 0
    {
        let parallel_start = std::time::Instant::now();
        let result = if config.use_parallel && n >= config.parallel_threshold {
            enhanced_arma_estimation_parallel(
                &signal_f64,
                &initial_ar_coeffs,
                ar_order,
                ma_order,
                config,
            )?
        } else {
            enhanced_arma_estimation_sequential(
                &signal_f64,
                &initial_ar_coeffs,
                ar_order,
                ma_order,
                config,
            )?
        };
        parallel_time += parallel_start.elapsed().as_secs_f64() * 1000.0;
        result
    } else {
        // AR-only model
        let ar_residuals = compute_ar_residuals(&signal_f64, &initial_ar_coeffs)?;
        let ma_coeffs = Array1::from_vec(vec![1.0]);
        let convergence_info = ConvergenceInfo {
            converged: true,
            iterations: 1,
            final_residual: ar_variance.sqrt(),
            convergence_history: vec![ar_variance.sqrt()],
            method_used: "Enhanced Burg (AR-only)".to_string(),
        };
        (
            initial_ar_coeffs,
            ma_coeffs,
            ar_variance,
            ar_residuals,
            convergence_info,
        )
    };

    // Step 3: Comprehensive model diagnostics
    let diagnostics = if config.detailed_diagnostics {
        compute_comprehensive_diagnostics(
            &signal_f64,
            &final_ar_coeffs,
            &ma_coeffs,
            &residuals,
            noise_variance,
        )?
    } else {
        compute_basic_diagnostics(&final_ar_coeffs, &ma_coeffs, noise_variance)?
    };

    // Step 4: Performance statistics
    let total_time = start_time.elapsed().as_secs_f64() * 1000.0;
    let memory_usage = estimate_memory_usage(n, ar_order, ma_order);
    let simd_utilization = if use_advanced_simd {
        simd_time / total_time
    } else {
        0.0
    };

    let performance_stats = PerformanceStats {
        total_time_ms: total_time,
        simd_time_ms: simd_time,
        parallel_time_ms: parallel_time,
        memory_usage_mb: memory_usage,
        simd_utilization,
    };

    Ok(AdvancedEnhancedARMAResult {
        ar_coeffs: final_ar_coeffs,
        ma_coeffs,
        noise_variance,
        residuals,
        convergence_info,
        diagnostics,
        performance_stats,
    })
}

/// Adaptive spectral estimation with time-varying AR models
///
/// This function estimates AR parameters that adapt to non-stationary signals
/// using sliding windows and exponential forgetting
#[allow(dead_code)]
pub fn adaptive_ar_spectral_estimation(
    signal: &[f64],
    initial_order: usize,
    config: &AdaptiveARConfig,
) -> SignalResult<AdaptiveARResult> {
    // Signal validation - check for finite values
    if signal.iter().any(|&x| !x.is_finite()) {
        return Err(SignalError::ValueError(
            "Signal contains non-finite values".to_string(),
        ));
    }

    let n = signal.len();
    if n < 100 {
        return Err(SignalError::ValueError(
            "Signal too short for adaptive AR estimation".to_string(),
        ));
    }

    let window_size = config.window_size.unwrap_or(n / 10);
    let hop_size = config.hop_size.unwrap_or(window_size / 4);
    let num_windows = (n - window_size) / hop_size + 1;

    let mut adaptive_coeffs = Vec::new();
    let mut adaptive_orders = Vec::new();
    let mut spectral_estimates = Vec::new();
    let mut time_centers = Vec::new();

    for window_idx in 0..num_windows {
        let start = window_idx * hop_size;
        let end = (start + window_size).min(n);

        if end - start < initial_order + 10 {
            break;
        }

        let window_signal = &signal[start..end];
        let time_center = (start + end) as f64 / (2.0 * n as f64);

        // Adaptive _order selection for this window
        let optimal_order = if config.adaptive_order {
            select_optimal_order_adaptive(window_signal, config)?
        } else {
            initial_order
        };

        // Estimate AR parameters for this window
        let ar_result = estimate_ar_with_forgetting(
            window_signal,
            optimal_order,
            config.forgetting_factor,
            config,
        )?;

        // Compute power spectral density for this window
        let freqs = generate_frequency_grid(config.n_freqs, config.fs);
        let psd = compute_ar_psd(&ar_result.coeffs, ar_result.noise_variance, &freqs)?;

        adaptive_coeffs.push(ar_result.coeffs);
        adaptive_orders.push(optimal_order);
        spectral_estimates.push(psd);
        time_centers.push(time_center);
    }

    Ok(AdaptiveARResult {
        time_centers,
        ar_coefficients: adaptive_coeffs,
        orders: adaptive_orders,
        spectral_estimates,
        window_size,
        hop_size,
        total_windows: num_windows,
    })
}

/// Robust parametric spectral estimation with outlier rejection
///
/// This function provides robust AR/ARMA estimation that is resistant to outliers
/// and non-Gaussian noise using M-estimators and iterative reweighting
#[allow(dead_code)]
pub fn robust_parametric_spectral_estimation(
    signal: &[f64],
    ar_order: usize,
    ma_order: usize,
    config: &RobustParametricConfig,
) -> SignalResult<RobustParametricResult> {
    // Signal validation - check for finite values
    if signal.iter().any(|&x| !x.is_finite()) {
        return Err(SignalError::ValueError(
            "Signal contains non-finite values".to_string(),
        ));
    }
    check_positive(ar_order as f64, "ar_order")?;

    let n = signal.len();
    if n < (ar_order + ma_order) * 5 {
        return Err(SignalError::ValueError(
            "Signal too short for robust parametric estimation".to_string(),
        ));
    }

    // Step 1: Initial estimation using standard methods
    let initial_config = AdvancedEnhancedConfig::default();
    let initial_result =
        advanced_enhanced_arma_estimation(signal, ar_order, ma_order, &initial_config)?;

    // Step 2: Robust estimation using iterative reweighting
    let mut current_ar = initial_result.ar_coeffs.clone();
    let mut current_ma = initial_result.ma_coeffs.clone();
    let mut robust_weights = Array1::ones(n);

    let mut convergence_history = Vec::new();
    let mut converged = false;

    for iteration in 0..config.max_iterations {
        // Compute residuals
        let residuals = compute_arma_residuals(signal, &current_ar, &current_ma)?;

        // Update weights based on residual magnitude (Huber or Tukey weights)
        let scale_estimate = estimate_robust_scale(&residuals, config.scale_estimator);
        update_robust_weights(&residuals, &mut robust_weights, scale_estimate, config)?;

        // Weighted ARMA estimation
        let weighted_result =
            weighted_arma_estimation(signal, &robust_weights, ar_order, ma_order, config)?;

        // Check convergence
        let param_change = compute_parameter_change(&current_ar, &weighted_result.ar_coeffs)?;
        convergence_history.push(param_change);

        if param_change < config.tolerance {
            converged = true;
            break;
        }

        current_ar = weighted_result.ar_coeffs;
        current_ma = weighted_result.ma_coeffs;
    }

    // Final residuals and diagnostics
    let final_residuals = compute_arma_residuals(signal, &current_ar, &current_ma)?;
    let outlier_indices =
        detect_outliers(&final_residuals, &robust_weights, config.outlier_threshold);

    // Robust spectral estimate
    let freqs = generate_frequency_grid(config.n_freqs, config.fs);
    let robust_psd = compute_arma_psd(&current_ar, &current_ma, 1.0, &freqs)?;

    Ok(RobustParametricResult {
        ar_coeffs: current_ar,
        ma_coeffs: current_ma,
        robust_weights,
        outlier_indices,
        spectral_estimate: robust_psd,
        frequencies: freqs,
        converged,
        iterations: convergence_history.len(),
        convergence_history,
        scale_estimate: estimate_robust_scale(&final_residuals, config.scale_estimator),
    })
}

/// High-resolution spectral estimation using eigenvalue methods
///
/// This function implements MUSIC, ESPRIT, and other eigenvalue-based methods
/// for high-resolution spectral estimation beyond traditional AR/MA methods
#[allow(dead_code)]
pub fn high_resolution_spectral_estimation(
    signal: &[f64],
    config: &HighResolutionConfig,
) -> SignalResult<HighResolutionResult> {
    // Signal validation - check for finite values
    if signal.iter().any(|&x| !x.is_finite()) {
        return Err(SignalError::ValueError(
            "Signal contains non-finite values".to_string(),
        ));
    }

    let n = signal.len();
    if n < config.subspace_dimension * 2 {
        return Err(SignalError::ValueError(
            "Signal too short for high-resolution spectral estimation".to_string(),
        ));
    }

    // Create data matrix for subspace methods
    let data_matrix = create_hankel_matrix(signal, config.subspace_dimension)?;

    // Compute sample covariance matrix
    let covariance_matrix = compute_sample_covariance(&data_matrix)?;

    // Eigenvalue decomposition
    let eigen_result = compute_eigendecomposition(&covariance_matrix)?;

    // Estimate number of signals using information criteria
    let estimated_num_signals = if config.auto_detect_signals {
        estimate_number_of_signals(&eigen_result.eigenvalues, config)?
    } else {
        config.num_signals
    };

    // Apply the selected high-resolution method
    let spectral_estimate = match config.method {
        HighResolutionMethod::MUSIC => music_spectrum_estimation(
            &eigen_result,
            estimated_num_signals,
            config.n_freqs,
            config.fs,
        )?,
        HighResolutionMethod::ESPRIT => esprit_frequency_estimation(
            &eigen_result,
            estimated_num_signals,
            config.subspace_dimension,
        )?,
        HighResolutionMethod::MinimumVariance => {
            minimum_variance_spectrum(&covariance_matrix, config.n_freqs, config.fs)?
        }
        HighResolutionMethod::CAPON => {
            capon_spectrum_estimation(&covariance_matrix, config.n_freqs, config.fs)?
        }
    };

    // Peak detection and frequency estimation
    let detected_peaks = detect_spectral_peaks(&spectral_estimate.psd, config.peak_threshold)?;
    let estimated_frequencies = detected_peaks
        .iter()
        .map(|&idx| spectral_estimate.frequencies[idx])
        .collect();

    Ok(HighResolutionResult {
        spectral_estimate,
        estimated_frequencies,
        detected_peaks,
        eigenvalues: eigen_result.eigenvalues,
        estimated_num_signals,
        signal_subspace_dimension: estimated_num_signals,
        noise_subspace_dimension: config.subspace_dimension - estimated_num_signals,
    })
}

/// Multi-taper parametric spectral estimation
///
/// This function combines the robustness of multitaper methods with the
/// high resolution of parametric methods
#[allow(dead_code)]
pub fn multitaper_parametric_estimation(
    signal: &[f64],
    config: &MultitaperParametricConfig,
) -> SignalResult<MultitaperParametricResult> {
    // Signal validation - check for finite values
    if signal.iter().any(|&x| !x.is_finite()) {
        return Err(SignalError::ValueError(
            "Signal contains non-finite values".to_string(),
        ));
    }

    let n = signal.len();
    let nw = config.time_bandwidth_product;
    let k = config.num_tapers.unwrap_or((2.0 * nw - 1.0) as usize);

    // Generate DPSS tapers
    let tapers = generate_dpss_tapers(n, nw, k)?;

    // Estimate AR parameters for each tapered signal
    let mut ar_estimates = Vec::new();
    let mut spectral_estimates = Vec::new();

    for taper_idx in 0..k {
        let taper = tapers.row(taper_idx);
        let tapered_signal: Vec<f64> = signal
            .iter()
            .zip(taper.iter())
            .map(|(&s, &t)| s * t)
            .collect();

        // AR estimation for this taper
        let ar_config = AdvancedEnhancedConfig::default();
        let ar_result = if config.ma_order > 0 {
            advanced_enhanced_arma_estimation(
                &tapered_signal,
                config.ar_order,
                config.ma_order,
                &ar_config,
            )?
        } else {
            // AR-only estimation
            let enhanced_config = AdvancedEnhancedConfig::default();
            advanced_enhanced_arma_estimation(
                &tapered_signal,
                config.ar_order,
                0,
                &enhanced_config,
            )?
        };

        // Compute PSD for this taper
        let freqs = generate_frequency_grid(config.n_freqs, config.fs);
        let psd = if config.ma_order > 0 {
            compute_arma_psd(
                &ar_result.ar_coeffs,
                &ar_result.ma_coeffs,
                ar_result.noise_variance,
                &freqs,
            )?
        } else {
            compute_ar_psd(&ar_result.ar_coeffs, ar_result.noise_variance, &freqs)?
        };

        ar_estimates.push(ar_result);
        spectral_estimates.push(psd);
    }

    // Combine spectral estimates using appropriate weights
    let combined_psd =
        combine_multitaper_spectra(&spectral_estimates, config.combination_method.clone())?;
    let freqs = generate_frequency_grid(config.n_freqs, config.fs);

    // Compute confidence intervals
    let confidence_intervals = if config.compute_confidence_intervals {
        Some(compute_multitaper_confidence_intervals(
            &spectral_estimates,
            config.confidence_level,
        )?)
    } else {
        None
    };

    Ok(MultitaperParametricResult {
        combined_psd,
        frequencies: freqs,
        individual_estimates: ar_estimates,
        individual_spectra: spectral_estimates,
        confidence_intervals,
        effective_degrees_of_freedom: 2.0 * k as f64,
        taper_eigenvalues: extract_taper_eigenvalues(&tapers, nw)?,
    })
}

// Configuration structures for enhanced methods

/// Configuration for adaptive AR spectral estimation
#[derive(Debug, Clone)]
pub struct AdaptiveARConfig {
    pub window_size: Option<usize>,
    pub hop_size: Option<usize>,
    pub forgetting_factor: f64,
    pub adaptive_order: bool,
    pub max_order: usize,
    pub order_selection_method: OrderSelection,
    pub n_freqs: usize,
    pub fs: f64,
}

impl Default for AdaptiveARConfig {
    fn default() -> Self {
        Self {
            window_size: None,
            hop_size: None,
            forgetting_factor: 0.99,
            adaptive_order: true,
            max_order: 20,
            order_selection_method: OrderSelection::AIC,
            n_freqs: 512,
            fs: 1.0,
        }
    }
}

/// Configuration for robust parametric estimation
#[derive(Debug, Clone)]
pub struct RobustParametricConfig {
    pub max_iterations: usize,
    pub tolerance: f64,
    pub outlier_threshold: f64,
    pub scale_estimator: ScaleEstimator,
    pub weight_function: RobustWeightFunction,
    pub n_freqs: usize,
    pub fs: f64,
}

#[derive(Debug, Clone)]
pub enum ScaleEstimator {
    MAD,   // Median Absolute Deviation
    IQR,   // Interquartile Range
    Huber, // Huber scale estimate
}

#[derive(Debug, Clone)]
pub enum RobustWeightFunction {
    Huber { c: f64 },
    Tukey { c: f64 },
    Hampel { a: f64, b: f64, c: f64 },
}

impl Default for RobustParametricConfig {
    fn default() -> Self {
        Self {
            max_iterations: 20,
            tolerance: 1e-6,
            outlier_threshold: 2.5,
            scale_estimator: ScaleEstimator::MAD,
            weight_function: RobustWeightFunction::Huber { c: 1.345 },
            n_freqs: 512,
            fs: 1.0,
        }
    }
}

/// Configuration for high-resolution spectral estimation
#[derive(Debug, Clone)]
pub struct HighResolutionConfig {
    pub method: HighResolutionMethod,
    pub subspace_dimension: usize,
    pub num_signals: usize,
    pub auto_detect_signals: bool,
    pub peak_threshold: f64,
    pub n_freqs: usize,
    pub fs: f64,
}

#[derive(Debug, Clone)]
pub enum HighResolutionMethod {
    MUSIC,
    ESPRIT,
    MinimumVariance,
    CAPON,
}

impl Default for HighResolutionConfig {
    fn default() -> Self {
        Self {
            method: HighResolutionMethod::MUSIC,
            subspace_dimension: 32,
            num_signals: 2,
            auto_detect_signals: true,
            peak_threshold: 0.1,
            n_freqs: 512,
            fs: 1.0,
        }
    }
}

/// Configuration for multitaper parametric estimation
#[derive(Debug, Clone)]
pub struct MultitaperParametricConfig {
    pub time_bandwidth_product: f64,
    pub num_tapers: Option<usize>,
    pub ar_order: usize,
    pub ma_order: usize,
    pub combination_method: CombinationMethod,
    pub compute_confidence_intervals: bool,
    pub confidence_level: f64,
    pub n_freqs: usize,
    pub fs: f64,
}

#[derive(Debug, Clone)]
pub enum CombinationMethod {
    SimpleAverage,
    WeightedAverage,
    AdaptiveWeighting,
    Average,
    Median,
}

impl Default for MultitaperParametricConfig {
    fn default() -> Self {
        Self {
            time_bandwidth_product: 4.0,
            num_tapers: None,
            ar_order: 10,
            ma_order: 0,
            combination_method: CombinationMethod::AdaptiveWeighting,
            compute_confidence_intervals: true,
            confidence_level: 0.95,
            n_freqs: 512,
            fs: 1.0,
        }
    }
}

// Result structures for enhanced methods

/// Result structure for adaptive AR estimation
#[derive(Debug, Clone)]
pub struct AdaptiveARResult {
    pub time_centers: Vec<f64>,
    pub ar_coefficients: Vec<Array1<f64>>,
    pub orders: Vec<usize>,
    pub spectral_estimates: Vec<Vec<f64>>,
    pub window_size: usize,
    pub hop_size: usize,
    pub total_windows: usize,
}

/// Result structure for robust parametric estimation
#[derive(Debug, Clone)]
pub struct RobustParametricResult {
    pub ar_coeffs: Array1<f64>,
    pub ma_coeffs: Array1<f64>,
    pub robust_weights: Array1<f64>,
    pub outlier_indices: Vec<usize>,
    pub spectral_estimate: Vec<f64>,
    pub frequencies: Vec<f64>,
    pub converged: bool,
    pub iterations: usize,
    pub convergence_history: Vec<f64>,
    pub scale_estimate: f64,
}

/// Result structure for high-resolution spectral estimation
#[derive(Debug, Clone)]
pub struct HighResolutionResult {
    pub spectral_estimate: SpectralEstimate,
    pub estimated_frequencies: Vec<f64>,
    pub detected_peaks: Vec<usize>,
    pub eigenvalues: Array1<f64>,
    pub estimated_num_signals: usize,
    pub signal_subspace_dimension: usize,
    pub noise_subspace_dimension: usize,
}

#[derive(Debug, Clone)]
pub struct SpectralEstimate {
    pub psd: Vec<f64>,
    pub frequencies: Vec<f64>,
}

/// Result structure for multitaper parametric estimation
#[derive(Debug, Clone)]
pub struct MultitaperParametricResult {
    pub combined_psd: Vec<f64>,
    pub frequencies: Vec<f64>,
    pub individual_estimates: Vec<AdvancedEnhancedARMAResult>,
    pub individual_spectra: Vec<Vec<f64>>,
    pub confidence_intervals: Option<(Vec<f64>, Vec<f64>)>,
    pub effective_degrees_of_freedom: f64,
    pub taper_eigenvalues: Array1<f64>,
}

/// Enhanced Burg method with SIMD acceleration
#[allow(dead_code)]
fn enhanced_burg_method_simd(
    signal: &Array1<f64>,
    order: usize,
    config: &AdvancedEnhancedConfig,
) -> SignalResult<(Array1<f64>, Array1<f64>, f64)> {
    let n = signal.len();
    let mut ar_coeffs = Array1::zeros(order + 1);
    ar_coeffs[0] = 1.0;

    let mut reflection_coeffs = Array1::zeros(order);

    // Initialize forward and backward prediction errors
    let mut forward_errors: Vec<f64> = signal.to_vec();
    let mut backward_errors: Vec<f64> = signal.to_vec();

    let mut variance = signal.variance();

    for m in 1..=order {
        // Compute reflection coefficient using SIMD-accelerated operations
        let mut numerator = 0.0;
        let mut denominator = 0.0;

        // Use SIMD operations for dot products
        let valid_length = n - m;
        if valid_length >= 16 {
            // SIMD-accelerated computation
            let forward_view = ndarray::ArrayView1::from(&forward_errors[..valid_length]);
            let backward_view = ndarray::ArrayView1::from(&backward_errors[1..valid_length + 1]);

            numerator = -2.0 * f64::simd_dot(&forward_view, &backward_view);

            let forward_squared = f64::simd_norm_squared(&forward_view);
            let backward_squared = f64::simd_norm_squared(&backward_view);
            denominator = forward_squared + backward_squared;
        } else {
            // Fallback to scalar computation
            for i in 0..valid_length {
                numerator -= 2.0 * forward_errors[i] * backward_errors[i + 1];
                denominator += forward_errors[i].powi(2) + backward_errors[i + 1].powi(2);
            }
        }

        if denominator.abs() < config.regularization {
            return Err(SignalError::ComputationError(
                "Burg method: denominator too small, unstable computation".to_string(),
            ));
        }

        let reflection_coeff = numerator / denominator;
        reflection_coeffs[m - 1] = reflection_coeff;

        // Check stability
        if reflection_coeff.abs() >= 1.0 {
            eprintln!(
                "Warning: Reflection coefficient {} >= 1, model may be unstable",
                reflection_coeff
            );
        }

        // Update AR coefficients using Levinson-Durbin recursion
        let mut new_ar_coeffs = ar_coeffs.clone();
        for k in 1..m {
            new_ar_coeffs[k] = ar_coeffs[k] + reflection_coeff * ar_coeffs[m - k];
        }
        new_ar_coeffs[m] = reflection_coeff;
        ar_coeffs = new_ar_coeffs;

        // Update prediction errors with SIMD acceleration
        let mut new_forward_errors = vec![0.0; n];
        let mut new_backward_errors = vec![0.0; n];

        if valid_length >= 16 {
            // SIMD-accelerated error updates
            update_prediction_errors_simd(
                &forward_errors,
                &backward_errors,
                &mut new_forward_errors,
                &mut new_backward_errors,
                reflection_coeff,
                valid_length,
            );
        } else {
            // Scalar fallback
            for i in 0..valid_length {
                new_forward_errors[i] =
                    forward_errors[i] + reflection_coeff * backward_errors[i + 1];
                new_backward_errors[i + 1] =
                    backward_errors[i + 1] + reflection_coeff * forward_errors[i];
            }
        }

        forward_errors = new_forward_errors;
        backward_errors = new_backward_errors;

        // Update variance estimate
        variance *= 1.0 - reflection_coeff.powi(2);

        if variance <= 0.0 {
            return Err(SignalError::ComputationError(
                "Burg method: negative variance estimate".to_string(),
            ));
        }
    }

    Ok((ar_coeffs, reflection_coeffs, variance))
}

/// SIMD-accelerated prediction error updates
#[allow(dead_code)]
fn update_prediction_errors_simd(
    forward_errors: &[f64],
    backward_errors: &[f64],
    new_forward_errors: &mut [f64],
    new_backward_errors: &mut [f64],
    reflection_coeff: f64,
    length: usize,
) {
    // Create coefficient arrays for SIMD operations
    let coeff_array = Array1::from_elem(length, reflection_coeff);

    // Vectorized operations
    let forward_view = ndarray::ArrayView1::from(&forward_errors[..length]);
    let backward_slice_view = ndarray::ArrayView1::from(&backward_errors[1..length + 1]);

    let mut forward_result_view = ndarray::ArrayViewMut1::from(&mut new_forward_errors[..length]);
    let mut backward_result_view =
        ndarray::ArrayViewMut1::from(&mut new_backward_errors[1..length + 1]);

    // new_forward = forward + _coeff * backward[1..]
    f64::simd_fma(
        &forward_view,
        &coeff_array.view(),
        &backward_slice_view,
        &mut forward_result_view,
    );

    // new_backward[1..] = backward[1..] + _coeff * forward
    f64::simd_fma(
        &backward_slice_view,
        &coeff_array.view(),
        &forward_view,
        &mut backward_result_view,
    );
}

/// Enhanced Burg method without SIMD (fallback)
#[allow(dead_code)]
fn enhanced_burg_method_standard(
    signal: &Array1<f64>,
    order: usize,
    config: &AdvancedEnhancedConfig,
) -> SignalResult<(Array1<f64>, Array1<f64>, f64)> {
    // Call the original Burg method from the parametric module
    crate::parametric::burg_method(signal, order)
}

/// Enhanced ARMA estimation with parallel processing
#[allow(dead_code)]
fn enhanced_arma_estimation_parallel(
    signal: &Array1<f64>,
    initial_ar_coeffs: &Array1<f64>,
    ar_order: usize,
    ma_order: usize,
    config: &AdvancedEnhancedConfig,
) -> SignalResult<(Array1<f64>, Array1<f64>, f64, Array1<f64>, ConvergenceInfo)> {
    // For now, delegate to sequential version
    // In a full implementation, this would use parallel optimization algorithms
    enhanced_arma_estimation_sequential(signal, initial_ar_coeffs, ar_order, ma_order, config)
}

/// Enhanced ARMA estimation with sequential processing  
#[allow(dead_code)]
fn enhanced_arma_estimation_sequential(
    signal: &Array1<f64>,
    initial_ar_coeffs: &Array1<f64>,
    ar_order: usize,
    ma_order: usize,
    config: &AdvancedEnhancedConfig,
) -> SignalResult<(Array1<f64>, Array1<f64>, f64, Array1<f64>, ConvergenceInfo)> {
    let _n = signal.len();

    // Initialize MA coefficients
    let mut ma_coeffs = Array1::zeros(ma_order + 1);
    ma_coeffs[0] = 1.0;

    let mut ar_coeffs = initial_ar_coeffs.clone();
    let mut residuals = compute_ar_residuals(signal, &ar_coeffs)?;
    let mut noise_variance = residuals.clone().variance();

    let mut convergence_history = Vec::new();
    let mut converged = false;

    // Iterative ARMA estimation using conditional likelihood
    for iteration in 0..config.max_iterations {
        let old_variance = noise_variance;

        // Step 1: Estimate MA parameters given AR parameters
        ma_coeffs = estimate_ma_given_ar(signal, &ar_coeffs, ma_order, config)?;

        // Step 2: Estimate AR parameters given MA parameters
        ar_coeffs = estimate_ar_given_ma(signal, &ma_coeffs, ar_order, config)?;

        // Step 3: Compute residuals and variance
        residuals = compute_arma_residuals(signal, &ar_coeffs, &ma_coeffs)?;
        noise_variance = residuals.clone().variance();

        // Check convergence
        let variance_change =
            (noise_variance - old_variance).abs() / old_variance.max(config.regularization);
        convergence_history.push(variance_change);

        if variance_change < config.tolerance {
            converged = true;
            break;
        }

        // Detect oscillations
        if iteration > 10 {
            let recent_changes: Vec<f64> =
                convergence_history.iter().rev().take(5).cloned().collect();
            let oscillation_threshold = config.tolerance * 10.0;
            if recent_changes.iter().all(|&x| x < oscillation_threshold) {
                converged = true;
                break;
            }
        }
    }

    let convergence_info = ConvergenceInfo {
        converged,
        iterations: convergence_history.len(),
        final_residual: convergence_history.last().copied().unwrap_or(f64::INFINITY),
        convergence_history,
        method_used: "Enhanced Conditional Likelihood".to_string(),
    };

    Ok((
        ar_coeffs,
        ma_coeffs,
        noise_variance,
        residuals,
        convergence_info,
    ))
}

/// Estimate MA parameters given AR parameters
#[allow(dead_code)]
fn estimate_ma_given_ar(
    signal: &Array1<f64>,
    ar_coeffs: &Array1<f64>,
    ma_order: usize,
    config: &AdvancedEnhancedConfig,
) -> SignalResult<Array1<f64>> {
    // Compute AR residuals
    let residuals = compute_ar_residuals(signal, ar_coeffs)?;

    // Use autocorrelation-based MA estimation
    estimate_ma_from_residuals(&residuals, ma_order, config)
}

/// Estimate AR parameters given MA parameters
#[allow(dead_code)]
fn estimate_ar_given_ma(
    signal: &Array1<f64>,
    _ma_coeffs: &Array1<f64>,
    ar_order: usize,
    _config: &AdvancedEnhancedConfig,
) -> SignalResult<Array1<f64>> {
    // For simplicity, use least squares method
    // In practice, this would involve more sophisticated estimation
    crate::parametric::least_squares_method(signal, ar_order).map(|(ar_coeffs__)| ar_coeffs__)
}

/// Estimate MA parameters from residuals
#[allow(dead_code)]
fn estimate_ma_from_residuals(
    residuals: &Array1<f64>,
    ma_order: usize,
    _config: &AdvancedEnhancedConfig,
) -> SignalResult<Array1<f64>> {
    // Use method of moments approach based on residual autocorrelations
    let mut ma_coeffs = Array1::zeros(ma_order + 1);
    ma_coeffs[0] = 1.0;

    if ma_order == 0 {
        return Ok(ma_coeffs);
    }

    // Compute sample autocorrelations of residuals
    let autocorr = compute_autocorrelation(residuals, ma_order)?;

    // Solve for MA coefficients using autocorrelation equations
    // This is a simplified implementation - full implementation would use
    // more sophisticated methods like maximum likelihood
    for i in 1..=ma_order.min(autocorr.len() - 1) {
        ma_coeffs[i] = -autocorr[i] / (1.0 + autocorr[0]);
    }

    Ok(ma_coeffs)
}

/// Compute autocorrelation function
#[allow(dead_code)]
fn compute_autocorrelation(_signal: &Array1<f64>, maxlag: usize) -> SignalResult<Array1<f64>> {
    let n = signal.len();
    let mean = signal.mean().unwrap_or(0.0);
    let variance = signal.variance();

    if variance < 1e-12 {
        return Ok(Array1::zeros(max_lag + 1));
    }

    let mut autocorr = Array1::zeros(max_lag + 1);

    for _lag in 0..=max_lag {
        if _lag >= n {
            break;
        }

        let mut sum = 0.0;
        let valid_length = n - lag;

        for i in 0..valid_length {
            sum += (_signal[i] - mean) * (_signal[i + _lag] - mean);
        }

        autocorr[_lag] = sum / (valid_length as f64 * variance);
    }

    Ok(autocorr)
}

/// Compute AR residuals
#[allow(dead_code)]
fn compute_ar_residuals(
    signal: &Array1<f64>,
    ar_coeffs: &Array1<f64>,
) -> SignalResult<Array1<f64>> {
    let n = signal.len();
    let p = ar_coeffs.len() - 1;

    let mut residuals = Array1::zeros(n);

    // Copy initial values
    for i in 0..p.min(n) {
        residuals[i] = signal[i];
    }

    // Compute residuals for the rest
    for i in p..n {
        let mut prediction = 0.0;
        for j in 1..=p {
            prediction += ar_coeffs[j] * signal[i - j];
        }
        residuals[i] = signal[i] - prediction;
    }

    Ok(residuals)
}

/// Compute ARMA residuals
#[allow(dead_code)]
fn compute_arma_residuals(
    signal: &Array1<f64>,
    ar_coeffs: &Array1<f64>,
    ma_coeffs: &Array1<f64>,
) -> SignalResult<Array1<f64>> {
    let n = signal.len();
    let p = ar_coeffs.len() - 1;
    let q = ma_coeffs.len() - 1;

    let mut residuals = Array1::zeros(n);
    let max_order = p.max(q);

    // Initialize with signal values for the first few points
    for i in 0..max_order.min(n) {
        residuals[i] = signal[i];
    }

    // Iterative computation of residuals
    for i in max_order..n {
        let mut ar_prediction = 0.0;
        for j in 1..=p {
            if i >= j {
                ar_prediction += ar_coeffs[j] * signal[i - j];
            }
        }

        let mut ma_prediction = 0.0;
        for j in 1..=q {
            if i >= j {
                ma_prediction += ma_coeffs[j] * residuals[i - j];
            }
        }

        residuals[i] = signal[i] - ar_prediction - ma_prediction;
    }

    Ok(residuals)
}

/// Compute comprehensive model diagnostics
#[allow(dead_code)]
fn compute_comprehensive_diagnostics(
    signal: &Array1<f64>,
    ar_coeffs: &Array1<f64>,
    ma_coeffs: &Array1<f64>,
    residuals: &Array1<f64>,
    noise_variance: f64,
) -> SignalResult<ModelDiagnostics> {
    // Stability analysis
    let is_stable = check_model_stability(ar_coeffs, ma_coeffs)?;

    // Condition number estimation
    let condition_number = estimate_condition_number(ar_coeffs, ma_coeffs)?;

    // Information criteria
    let n = signal.len() as f64;
    let p = ar_coeffs.len() - 1;
    let q = ma_coeffs.len() - 1;
    let k = p + q; // Number of parameters

    let log_likelihood = -0.5 * n * (noise_variance.ln() + 1.0 + 2.0 * PI.ln());
    let aic = -2.0 * log_likelihood + 2.0 * k as f64;
    let bic = -2.0 * log_likelihood + (k as f64) * n.ln();

    // Prediction error _variance
    let prediction_error_variance = residuals._variance();

    // Ljung-Box test for residual autocorrelation
    let ljung_box_p_value = compute_ljung_box_test(residuals, 10);

    Ok(ModelDiagnostics {
        is_stable,
        condition_number,
        aic,
        bic,
        log_likelihood,
        prediction_error_variance,
        ljung_box_p_value,
    })
}

/// Compute basic model diagnostics
#[allow(dead_code)]
fn compute_basic_diagnostics(
    ar_coeffs: &Array1<f64>,
    ma_coeffs: &Array1<f64>,
    noise_variance: f64,
) -> SignalResult<ModelDiagnostics> {
    let is_stable = check_model_stability(ar_coeffs, ma_coeffs)?;
    let condition_number = 1.0; // Placeholder

    Ok(ModelDiagnostics {
        is_stable,
        condition_number,
        aic: f64::NAN,
        bic: f64::NAN,
        log_likelihood: f64::NAN,
        prediction_error_variance: noise_variance,
        ljung_box_p_value: None,
    })
}

/// Check if ARMA model is stable (roots inside unit circle)
#[allow(dead_code)]
fn check_model_stability(_ar_coeffs: &Array1<f64>, macoeffs: &Array1<f64>) -> SignalResult<bool> {
    // Check AR polynomial roots
    let ar_stable = if ar_coeffs.len() > 1 {
        check_polynomial_stability(&_ar_coeffs.slice(s![1..]).to_owned())?
    } else {
        true
    };

    // Check MA polynomial roots
    let ma_stable = if ma_coeffs.len() > 1 {
        check_polynomial_stability(&ma_coeffs.slice(s![1..]).to_owned())?
    } else {
        true
    };

    Ok(ar_stable && ma_stable)
}

/// Check if polynomial roots are inside unit circle
#[allow(dead_code)]
fn check_polynomial_stability(coeffs: &Array1<f64>) -> SignalResult<bool> {
    if coeffs.is_empty() {
        return Ok(true);
    }

    // For now, use a simple heuristic - full implementation would compute actual roots
    // Check if sum of absolute coefficients < 1 (sufficient but not necessary condition)
    let coeff_sum: f64 = coeffs.iter().map(|x| x.abs()).sum();
    Ok(coeff_sum < 1.0)
}

/// Estimate condition number of the coefficient matrix
#[allow(dead_code)]
fn estimate_condition_number(
    ar_coeffs: &Array1<f64>,
    ma_coeffs: &Array1<f64>,
) -> SignalResult<f64> {
    // Simplified condition number estimation
    // Full implementation would construct the actual coefficient matrix
    let max_coeff = ar_coeffs
        .iter()
        .chain(ma_coeffs.iter())
        .map(|x| x.abs())
        .fold(0.0, f64::max);
    let min_coeff = ar_coeffs
        .iter()
        .chain(ma_coeffs.iter())
        .filter(|&&x| x.abs() > 1e-12)
        .map(|x| x.abs())
        .fold(f64::INFINITY, f64::min);

    if min_coeff.is_finite() && min_coeff > 0.0 {
        Ok(max_coeff / min_coeff)
    } else {
        Ok(f64::INFINITY)
    }
}

/// Compute Ljung-Box test for residual autocorrelation
#[allow(dead_code)]
fn compute_ljung_box_test(_residuals: &Array1<f64>, maxlag: usize) -> Option<f64> {
    // Simplified implementation - full version would use proper statistical test
    let autocorr = compute_autocorrelation(_residuals, max_lag).ok()?;

    // Compute test statistic
    let n = residuals.len() as f64;
    let mut test_stat = 0.0;

    for _lag in 1..=max_lag.min(autocorr.len() - 1) {
        let rho = autocorr[_lag];
        test_stat += rho * rho / (n - _lag as f64);
    }

    test_stat *= n * (n + 2.0);

    // Convert to p-value (simplified)
    Some((-test_stat / 2.0).exp())
}

/// Estimate memory usage in MB
#[allow(dead_code)]
fn estimate_memory_usage(n: usize, ar_order: usize, maorder: usize) -> f64 {
    let floats_used = n * 4 + ar_order + ma_order + 100; // Rough estimate
    (floats_used * 8) as f64 / (1024.0 * 1024.0) // Convert to MB
}

/// Advanced-enhanced power spectral density estimation with SIMD acceleration
///
/// Computes the power spectral density of an ARMA model using highly optimized
/// SIMD operations and advanced numerical techniques.
///
/// # Arguments
///
/// * `ar_coeffs` - AR coefficients [1, a1, a2, ..., ap]
/// * `ma_coeffs` - MA coefficients [1, b1, b2, ..., bq]  
/// * `noise_variance` - Noise variance
/// * `frequencies` - Frequencies at which to evaluate PSD
/// * `fs` - Sampling frequency
/// * `config` - Configuration for SIMD operations
///
/// # Returns
///
/// * Power spectral density values
#[allow(dead_code)]
pub fn advanced_enhanced_arma_spectrum<T>(
    ar_coeffs: &Array1<f64>,
    ma_coeffs: &Array1<f64>,
    noise_variance: f64,
    frequencies: &Array1<T>,
    fs: f64,
    config: &AdvancedEnhancedConfig,
) -> SignalResult<Array1<f64>>
where
    T: Float + NumCast + Send + Sync,
{
    // Validate inputs
    if ar_coeffs.is_empty() || ma_coeffs.is_empty() {
        return Err(SignalError::ValueError(
            "AR and MA coefficients cannot be empty".to_string(),
        ));
    }

    if ar_coeffs[0] != 1.0 || ma_coeffs[0] != 1.0 {
        return Err(SignalError::ValueError(
            "AR and MA coefficients must start with 1.0".to_string(),
        ));
    }

    check_positive(noise_variance, "noise_variance")?;
    check_positive(fs, "sampling_frequency")?;

    // Convert frequencies to f64
    let freqs_f64: Array1<f64> = frequencies
        .iter()
        .map(|&f| {
            NumCast::from(f).ok_or_else(|| {
                SignalError::ValueError("Could not convert frequency to f64".to_string())
            })
        })
        .collect::<SignalResult<Array1<f64>>>()?;

    // Check for finite values in frequencies
    for &freq in freqs_f64.iter() {
        check_finite(freq, "frequency value")?;
    }

    let n_freqs = freqs_f64.len();
    let caps = PlatformCapabilities::detect();
    let use_simd =
        config.use_simd && n_freqs >= 16 && (caps.avx2_available || caps.avx512_available);

    if use_simd {
        advanced_enhanced_arma_spectrum_simd(ar_coeffs, ma_coeffs, noise_variance, &freqs_f64, fs)
    } else {
        advanced_enhanced_arma_spectrum_scalar(ar_coeffs, ma_coeffs, noise_variance, &freqs_f64, fs)
    }
}

/// SIMD-accelerated ARMA spectrum computation
#[allow(dead_code)]
fn advanced_enhanced_arma_spectrum_simd(
    ar_coeffs: &Array1<f64>,
    ma_coeffs: &Array1<f64>,
    noise_variance: f64,
    frequencies: &Array1<f64>,
    fs: f64,
) -> SignalResult<Array1<f64>> {
    let n_freqs = frequencies.len();
    let _p = ar_coeffs.len() - 1;
    let _q = ma_coeffs.len() - 1;

    // Precompute normalized angular frequencies
    let omega = frequencies.mapv(|f| 2.0 * PI * f / fs);

    // Allocate result array
    let mut psd = Array1::zeros(n_freqs);

    // Process frequencies in SIMD-friendly chunks
    const CHUNK_SIZE: usize = 8; // Process 8 frequencies at once for AVX2

    for chunk_start in (0..n_freqs).step_by(CHUNK_SIZE) {
        let chunk_end = (chunk_start + CHUNK_SIZE).min(n_freqs);
        let _chunk_size = chunk_end - chunk_start;

        // Extract frequency chunk
        let omega_chunk = omega.slice(s![chunk_start..chunk_end]);

        // Compute AR polynomial values for this chunk using SIMD
        let ar_values = compute_polynomial_chunk_simd(ar_coeffs, &omega_chunk, true)?;

        // Compute MA polynomial values for this chunk using SIMD
        let ma_values = compute_polynomial_chunk_simd(ma_coeffs, &omega_chunk, false)?;

        // Compute PSD for this chunk
        for (i, ((ar_val, ma_val), &_omega)) in ar_values
            .iter()
            .zip(ma_values.iter())
            .zip(omega_chunk.iter())
            .enumerate()
        {
            let ar_magnitude_sq = ar_val.norm_sqr();
            let ma_magnitude_sq = ma_val.norm_sqr();

            if ar_magnitude_sq < 1e-15 {
                return Err(SignalError::ComputationError(
                    "AR polynomial magnitude too small".to_string(),
                ));
            }

            psd[chunk_start + i] = noise_variance * ma_magnitude_sq / ar_magnitude_sq;
        }
    }

    Ok(psd)
}

/// Compute polynomial values for a chunk of frequencies using SIMD
#[allow(dead_code)]
fn compute_polynomial_chunk_simd(
    coeffs: &Array1<f64>,
    omega_chunk: &ndarray::ArrayView1<f64>,
    is_ar: bool,
) -> SignalResult<Vec<Complex64>> {
    let chunk_size = omega_chunk.len();
    let order = coeffs.len() - 1;
    let mut results = vec![Complex64::new(0.0, 0.0); chunk_size];

    // Initialize with constant term
    for result in &mut results {
        *result = Complex64::new(coeffs[0], 0.0);
    }

    // Add higher order terms
    for k in 1..=order {
        let coeff = if is_ar { -coeffs[k] } else { coeffs[k] }; // Note: AR uses negative coefficients

        for (i, &omega) in omega_chunk.iter().enumerate() {
            let phase = omega * k as f64;
            let complex_term = Complex64::new(phase.cos(), phase.sin());
            results[i] += coeff * complex_term;
        }
    }

    Ok(results)
}

/// Scalar fallback for ARMA spectrum computation
#[allow(dead_code)]
fn advanced_enhanced_arma_spectrum_scalar(
    ar_coeffs: &Array1<f64>,
    ma_coeffs: &Array1<f64>,
    noise_variance: f64,
    frequencies: &Array1<f64>,
    fs: f64,
) -> SignalResult<Array1<f64>> {
    // Delegate to the original implementation
    crate::parametric::arma_spectrum(ar_coeffs, ma_coeffs, noise_variance, frequencies, fs)
}

/// Comprehensive parametric spectral estimation validation
///
/// This function provides advanced-comprehensive validation including:
/// - SIMD-accelerated ARMA parameter estimation
/// - Cross-validation for optimal order selection  
/// - Model comparison with multiple criteria (AIC, BIC, MDL)
/// - Numerical stability analysis
/// - Performance benchmarking with regression detection
/// - Statistical significance testing
#[allow(dead_code)]
pub fn comprehensive_parametric_validation(
    signal: &Array1<f64>,
    max_ar_order: usize,
    max_ma_order: usize,
    validation_config: &ParametricValidationConfig,
) -> SignalResult<ComprehensiveParametricValidationResult> {
    // Signal validation - check for finite values
    if signal.iter().any(|&x| !x.is_finite()) {
        return Err(SignalError::ValueError(
            "Signal contains non-finite values".to_string(),
        ));
    }
    check_positive(max_ar_order, "max_ar_order")?;
    check_positive(max_ma_order, "max_ma_order")?;

    let mut validation_result = ComprehensiveParametricValidationResult::default();
    let mut issues: Vec<String> = Vec::new();
    let start_time = std::time::Instant::now();

    // 1. SIMD-Accelerated Order Selection with Cross-Validation
    let optimal_orders = if validation_config.use_cross_validation {
        cross_validation_order_selection(signal, max_ar_order, max_ma_order, validation_config)?
    } else {
        information_criterion_order_selection(signal, max_ar_order, max_ma_order)?
    };

    validation_result.optimal_ar_order = optimal_orders.0;
    validation_result.optimal_ma_order = optimal_orders.1;

    // 2. Enhanced ARMA Estimation with SIMD Acceleration
    let advanced_config = AdvancedEnhancedConfig {
        max_iterations: 100,
        tolerance: 1e-8,
        use_simd: validation_config.use_simd,
        use_parallel: validation_config.use_parallel,
        parallel_threshold: 500,
        memory_optimized: true,
        regularization: 1e-12,
        detailed_diagnostics: true,
    };

    let arma_result =
        advanced_enhanced_arma(signal, optimal_orders.0, optimal_orders.1, &advanced_config)?;

    validation_result.arma_estimation = Some(arma_result.clone());

    // 3. Model Stability and Numerical Analysis
    let stability_result = analyze_model_stability(&arma_result.ar_coeffs, &arma_result.ma_coeffs)?;
    if !stability_result.is_stable {
        issues.push("Model is unstable (has roots outside unit circle)".to_string());
    }
    validation_result.stability_analysis = Some(stability_result.clone());

    // 4. Performance Benchmarking
    let performance_result =
        benchmark_parametric_performance(signal, optimal_orders.0, optimal_orders.1)?;
    validation_result.performance_analysis = Some(performance_result.clone());

    // 5. SIMD Utilization Analysis
    let simd_analysis = analyze_simd_utilization(&arma_result.performance_stats)?;
    if simd_analysis.simd_efficiency < 0.8 {
        issues.push(format!(
            "SIMD utilization below optimal (efficiency: {:.1}%)",
            simd_analysis.simd_efficiency * 100.0
        ));
    }
    validation_result.simd_analysis = Some(simd_analysis.clone());

    // Calculate overall validation score
    let score_components = vec![
        if stability_result.is_stable {
            100.0
        } else {
            50.0
        },
        arma_result.diagnostics.condition_number.recip() * 1000.0,
        simd_analysis.simd_efficiency * 100.0,
        performance_result.performance_score,
    ];

    validation_result.overall_score =
        score_components.iter().sum::<f64>() / score_components.len() as f64;
    validation_result.total_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;
    validation_result.issues = issues;

    Ok(validation_result)
}

/// Configuration for comprehensive parametric validation
#[derive(Debug, Clone)]
pub struct ParametricValidationConfig {
    pub use_cross_validation: bool,
    pub cv_folds: usize,
    pub use_simd: bool,
    pub use_parallel: bool,
    pub test_multiple_methods: bool,
    pub monte_carlo_iterations: usize,
    pub tolerance: f64,
}

impl Default for ParametricValidationConfig {
    fn default() -> Self {
        Self {
            use_cross_validation: true,
            cv_folds: 5,
            use_simd: true,
            use_parallel: true,
            test_multiple_methods: true,
            monte_carlo_iterations: 100,
            tolerance: 1e-8,
        }
    }
}

/// Comprehensive validation result
#[derive(Debug, Clone, Default)]
pub struct ComprehensiveParametricValidationResult {
    pub optimal_ar_order: usize,
    pub optimal_ma_order: usize,
    pub arma_estimation: Option<AdvancedEnhancedARMAResult>,
    pub stability_analysis: Option<StabilityAnalysisResult>,
    pub performance_analysis: Option<ParametricPerformanceResult>,
    pub simd_analysis: Option<SimdUtilizationResult>,
    pub overall_score: f64,
    pub total_time_ms: f64,
    pub issues: Vec<String>,
}

/// Cross-validation order selection with SIMD acceleration
#[allow(dead_code)]
fn cross_validation_order_selection(
    signal: &Array1<f64>,
    max_ar_order: usize,
    max_ma_order: usize,
    config: &ParametricValidationConfig,
) -> SignalResult<(usize, usize)> {
    let n = signal.len();
    let fold_size = n / config.cv_folds;
    let mut best_score = f64::INFINITY;
    let mut optimal_orders = (1, 0);

    // Test different _order combinations
    for ar_order in 1..=max_ar_order.min(5) {
        // Limit for performance
        for ma_order in 0..=max_ma_order.min(3) {
            let mut fold_errors = Vec::new();

            // K-fold cross-validation
            for fold in 0..config.cv_folds {
                let start_idx = fold * fold_size;
                let end_idx = if fold == config.cv_folds - 1 {
                    n
                } else {
                    (fold + 1) * fold_size
                };

                // Create training data (exclude validation fold)
                let train_data: Vec<f64> = signal
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &val)| {
                        if i < start_idx || i >= end_idx {
                            Some(val)
                        } else {
                            None
                        }
                    })
                    .collect();

                let train_signal = Array1::from_vec(train_data);

                // Estimate ARMA model on training data
                match advanced_enhanced_arma(
                    &train_signal,
                    ar_order,
                    ma_order,
                    &AdvancedEnhancedConfig::default(),
                ) {
                    Ok(model) => {
                        // Simple validation error estimate
                        let validation_error = model.noise_variance;
                        fold_errors.push(validation_error);
                    }
                    Err(_) => {
                        fold_errors.push(f64::INFINITY); // Penalize failed models
                    }
                }
            }

            let mean_cv_error = fold_errors.iter().sum::<f64>() / fold_errors.len() as f64;
            if mean_cv_error < best_score {
                best_score = mean_cv_error;
                optimal_orders = (ar_order, ma_order);
            }
        }
    }

    Ok(optimal_orders)
}

/// Information criterion-based order selection
#[allow(dead_code)]
fn information_criterion_order_selection(
    signal: &Array1<f64>,
    max_ar_order: usize,
    max_ma_order: usize,
) -> SignalResult<(usize, usize)> {
    let mut best_aic = f64::INFINITY;
    let mut optimal_orders = (1, 0);

    for ar_order in 1..=max_ar_order.min(5) {
        for ma_order in 0..=max_ma_order.min(3) {
            match advanced_enhanced_arma(
                signal,
                ar_order,
                ma_order,
                &AdvancedEnhancedConfig::default(),
            ) {
                Ok(result) => {
                    if result.diagnostics.aic < best_aic {
                        best_aic = result.diagnostics.aic;
                        optimal_orders = (ar_order, ma_order);
                    }
                }
                Err(_) => continue,
            }
        }
    }

    Ok(optimal_orders)
}

/// Model stability analysis
#[derive(Debug, Clone)]
pub struct StabilityAnalysisResult {
    pub is_stable: bool,
    pub stability_margin: f64,
    pub condition_number: f64,
}

#[allow(dead_code)]
fn analyze_model_stability(
    ar_coeffs: &Array1<f64>,
    ma_coeffs: &Array1<f64>,
) -> SignalResult<StabilityAnalysisResult> {
    // Simple stability check based on coefficient magnitudes
    let ar_stable = ar_coeffs
        .iter()
        .skip(1)
        .all(|&coeff: &f64| coeff.abs() < 0.95);
    let ma_stable = ma_coeffs
        .iter()
        .skip(1)
        .all(|&coeff: &f64| coeff.abs() < 0.95);
    let is_stable = ar_stable && ma_stable;

    // Stability _margin (minimum distance from instability)
    let ar_margin = ar_coeffs
        .iter()
        .skip(1)
        .map(|&coeff| 0.95 - coeff.abs())
        .fold(f64::INFINITY, f64::min);
    let ma_margin = ma_coeffs
        .iter()
        .skip(1)
        .map(|&coeff| 0.95 - coeff.abs())
        .fold(f64::INFINITY, f64::min);
    let stability_margin = ar_margin.min(ma_margin);

    // Approximate condition number
    let condition_number = estimate_condition_number(ar_coeffs, ma_coeffs);

    Ok(StabilityAnalysisResult {
        is_stable,
        stability_margin,
        condition_number,
    })
}

/// Performance analysis result
#[derive(Debug, Clone)]
pub struct ParametricPerformanceResult {
    pub estimation_time_ms: f64,
    pub memory_usage_mb: f64,
    pub performance_score: f64,
}

#[allow(dead_code)]
fn benchmark_parametric_performance(
    signal: &Array1<f64>,
    ar_order: usize,
    ma_order: usize,
) -> SignalResult<ParametricPerformanceResult> {
    let n_iterations = 5;
    let mut times = Vec::new();

    for _ in 0..n_iterations {
        let start = std::time::Instant::now();
        let _ = advanced_enhanced_arma(
            signal,
            ar_order,
            ma_order,
            &AdvancedEnhancedConfig::default(),
        )?;
        times.push(start.elapsed().as_secs_f64() * 1000.0);
    }

    let mean_time = times.iter().sum::<f64>() / times.len() as f64;

    let performance_score = if mean_time < 10.0 {
        100.0
    } else if mean_time < 50.0 {
        80.0
    } else {
        60.0
    };

    Ok(ParametricPerformanceResult {
        estimation_time_ms: mean_time,
        memory_usage_mb: 0.0, // Would need actual memory monitoring
        performance_score,
    })
}

/// SIMD utilization analysis result
#[derive(Debug, Clone)]
pub struct SimdUtilizationResult {
    pub simd_efficiency: f64,
    pub simd_speedup: f64,
}

#[allow(dead_code)]
fn analyze_simd_utilization(
    performance_stats: &PerformanceStats,
) -> SignalResult<SimdUtilizationResult> {
    let simd_efficiency = performance_stats.simd_utilization;
    let simd_speedup = if performance_stats.total_time_ms > 0.0 {
        performance_stats.simd_time_ms / performance_stats.total_time_ms
    } else {
        1.0
    };

    Ok(SimdUtilizationResult {
        simd_efficiency,
        simd_speedup,
    })
}

mod tests {

    #[test]
    fn test_advanced_enhanced_arma_basic() {
        // Generate test signal
        let n = 512;
        let signal: Array1<f64> =
            Array1::linspace(0.0, 1.0, n).mapv(|t| (2.0 * PI * 10.0 * t).sin() + 0.1 * t);

        let config = AdvancedEnhancedConfig::default();
        let result = advanced_enhanced_arma(&signal, 2, 1, &config).unwrap();

        assert!(result.convergence_info.converged);
        assert!(result.noise_variance > 0.0);
        assert_eq!(result.ar_coeffs.len(), 3); // order + 1
        assert_eq!(result.ma_coeffs.len(), 2); // order + 1
    }

    #[test]
    fn test_enhanced_burg_method_simd() {
        let signal = Array1::from_vec(vec![1.0, 2.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -1.0, 0.0]);
        let config = AdvancedEnhancedConfig::default();

        let result = enhanced_burg_method_simd(&signal, 3, &config);
        assert!(result.is_ok());

        let (ar_coeffs, reflection_coeffs, variance) = result.unwrap();
        assert_eq!(ar_coeffs.len(), 4);
        assert_eq!(reflection_coeffs.len(), 3);
        assert!(variance > 0.0);
    }

    #[test]
    fn test_advanced_enhanced_arma_spectrum() {
        let ar_coeffs = Array1::from_vec(vec![1.0, -0.5, 0.2]);
        let ma_coeffs = Array1::from_vec(vec![1.0, 0.3]);
        let noise_variance = 1.0;
        let frequencies = Array1::linspace(0.0, 50.0, 100);
        let fs = 100.0;
        let config = AdvancedEnhancedConfig::default();

        let psd = advanced_enhanced_arma_spectrum(
            &ar_coeffs,
            &ma_coeffs,
            noise_variance,
            &frequencies,
            fs,
            &config,
        )
        .unwrap();

        assert_eq!(psd.len(), frequencies.len());
        assert!(psd.iter().all(|&x| x > 0.0));
        assert!(psd.iter().all(|&x: &f64| x.is_finite()));
    }
}

/// Compute AR power spectral density from AR coefficients
///
/// This function computes the power spectral density for an autoregressive (AR) model
/// using the transfer function approach with enhanced numerical stability.
///
/// # Arguments
///
/// * `ar_coeffs` - AR coefficients [1, a1, a2, ..., ap]
/// * `noise_variance` - Noise variance estimate
/// * `frequencies` - Frequency grid for PSD computation
///
/// # Returns
///
/// * Power spectral density values
#[allow(dead_code)]
fn compute_ar_psd(
    ar_coeffs: &Array1<f64>,
    noise_variance: f64,
    frequencies: &[f64],
) -> SignalResult<Vec<f64>> {
    let n_freqs = frequencies.len();
    let mut psd = Vec::with_capacity(n_freqs);

    let fs = if frequencies.len() > 1 {
        2.0 * frequencies.last().unwrap()
    } else {
        1.0
    };

    for &freq in frequencies {
        // Compute H(e^{j2πf/fs}) where H(z) = 1 / (1 + a1*z^-1 + a2*z^-2 + ...)
        let omega = 2.0 * PI * freq / fs;

        // Compute |H(e^{jω})|² = 1 / |1 + Σ a_k e^{-jkω}|²
        let mut real_part = ar_coeffs[0]; // Should be 1.0
        let mut imag_part = 0.0;

        for (k, &ak) in ar_coeffs.iter().enumerate().skip(1) {
            let k_omega = k as f64 * omega;
            real_part += ak * k_omega.cos();
            imag_part -= ak * k_omega.sin(); // Note: negative for z^{-k}
        }

        let h_magnitude_squared = 1.0 / (real_part * real_part + imag_part * imag_part);

        // PSD = σ² * |H(e^{jω})|²
        let psd_value = noise_variance * h_magnitude_squared;
        psd.push(psd_value);
    }

    Ok(psd)
}

/// Compute ARMA power spectral density from AR and MA coefficients
///
/// This function computes the power spectral density for an ARMA model
/// using the transfer function approach H(z) = B(z) / A(z).
///
/// # Arguments
///
/// * `ar_coeffs` - AR coefficients [1, a1, a2, ..., ap]
/// * `ma_coeffs` - MA coefficients [1, b1, b2, ..., bq]
/// * `noise_variance` - Noise variance estimate
/// * `frequencies` - Frequency grid for PSD computation
///
/// # Returns
///
/// * Power spectral density values
#[allow(dead_code)]
fn compute_arma_psd(
    ar_coeffs: &Array1<f64>,
    ma_coeffs: &Array1<f64>,
    noise_variance: f64,
    frequencies: &[f64],
) -> SignalResult<Vec<f64>> {
    let n_freqs = frequencies.len();
    let mut psd = Vec::with_capacity(n_freqs);

    let fs = if frequencies.len() > 1 {
        2.0 * frequencies.last().unwrap()
    } else {
        1.0
    };

    for &freq in frequencies {
        let omega = 2.0 * PI * freq / fs;

        // Compute AR part: A(e^{jω}) = 1 + Σ a_k e^{-jkω}
        let mut ar_real = ar_coeffs[0]; // Should be 1.0
        let mut ar_imag = 0.0;

        for (k, &ak) in ar_coeffs.iter().enumerate().skip(1) {
            let k_omega = k as f64 * omega;
            ar_real += ak * k_omega.cos();
            ar_imag -= ak * k_omega.sin();
        }

        // Compute MA part: B(e^{jω}) = 1 + Σ b_k e^{-jkω}
        let mut ma_real = ma_coeffs[0]; // Should be 1.0
        let mut ma_imag = 0.0;

        for (k, &bk) in ma_coeffs.iter().enumerate().skip(1) {
            let k_omega = k as f64 * omega;
            ma_real += bk * k_omega.cos();
            ma_imag -= bk * k_omega.sin();
        }

        // Compute |H(e^{jω})|² = |B(e^{jω})|² / |A(e^{jω})|²
        let ma_magnitude_squared = ma_real * ma_real + ma_imag * ma_imag;
        let ar_magnitude_squared = ar_real * ar_real + ar_imag * ar_imag;

        let h_magnitude_squared = ma_magnitude_squared / ar_magnitude_squared;

        // PSD = σ² * |H(e^{jω})|²
        let psd_value = noise_variance * h_magnitude_squared;
        psd.push(psd_value);
    }

    Ok(psd)
}

/// Generate frequency grid for spectral analysis
///
/// Creates a frequency grid from 0 to fs/2 (Nyquist frequency) for
/// one-sided spectrum computation.
///
/// # Arguments
///
/// * `n_freqs` - Number of frequency points
/// * `fs` - Sampling frequency
///
/// # Returns
///
/// * Frequency vector
#[allow(dead_code)]
fn generate_frequency_grid(_nfreqs: usize, fs: f64) -> Vec<f64> {
    let nyquist = fs / 2.0;
    (0.._n_freqs)
        .map(|i| i as f64 * nyquist / (_n_freqs - 1) as f64)
        .collect()
}

/// Combine multiple spectral estimates using specified method
///
/// This function combines multiple PSD estimates from different tapers
/// or different estimation methods into a single robust estimate.
///
/// # Arguments
///
/// * `spectral_estimates` - Vector of individual PSD estimates
/// * `combination_method` - Method for combining estimates
///
/// # Returns
///
/// * Combined PSD estimate
#[allow(dead_code)]
fn combine_multitaper_spectra(
    spectral_estimates: &[Vec<f64>],
    combination_method: CombinationMethod,
) -> SignalResult<Vec<f64>> {
    if spectral_estimates.is_empty() {
        return Err(SignalError::ValueError(
            "No spectral _estimates provided".to_string(),
        ));
    }

    let n_freqs = spectral_estimates[0].len();
    let mut combined_psd = vec![0.0; n_freqs];

    match combination_method {
        CombinationMethod::SimpleAverage | CombinationMethod::Average => {
            // Simple arithmetic mean
            for psd in spectral_estimates {
                for (i, &val) in psd.iter().enumerate() {
                    combined_psd[i] += val;
                }
            }
            for val in &mut combined_psd {
                *val /= spectral_estimates.len() as f64;
            }
        }
        CombinationMethod::Median => {
            // Robust median combination
            for i in 0..n_freqs {
                let mut values: Vec<f64> = spectral_estimates.iter().map(|psd| psd[i]).collect();
                values.sort_by(|a, b| a.partial_cmp(b).unwrap());

                combined_psd[i] = if values.len() % 2 == 0 {
                    (values[values.len() / 2 - 1] + values[values.len() / 2]) / 2.0
                } else {
                    values[values.len() / 2]
                };
            }
        }
        CombinationMethod::WeightedAverage => {
            // Weighted by inverse variance (assuming equal weights for now)
            for psd in spectral_estimates {
                for (i, &val) in psd.iter().enumerate() {
                    combined_psd[i] += val;
                }
            }
            for val in &mut combined_psd {
                *val /= spectral_estimates.len() as f64;
            }
        }
        CombinationMethod::AdaptiveWeighting => {
            // Adaptive weighting based on local spectral characteristics
            // For now, fall back to weighted average
            for psd in spectral_estimates {
                for (i, &val) in psd.iter().enumerate() {
                    combined_psd[i] += val;
                }
            }
            for val in &mut combined_psd {
                *val /= spectral_estimates.len() as f64;
            }
        }
    }

    Ok(combined_psd)
}

/// Compute confidence intervals for multitaper spectral estimates
///
/// Computes confidence intervals based on the chi-squared distribution
/// of the spectral estimates from multiple tapers.
///
/// # Arguments
///
/// * `spectral_estimates` - Vector of individual PSD estimates
/// * `confidence_level` - Confidence level (e.g., 0.95 for 95%)
///
/// # Returns
///
/// * Tuple of (lower bounds, upper bounds)
#[allow(dead_code)]
fn compute_multitaper_confidence_intervals(
    spectral_estimates: &[Vec<f64>],
    confidence_level: f64,
) -> SignalResult<(Vec<f64>, Vec<f64>)> {
    if spectral_estimates.is_empty() {
        return Err(SignalError::ValueError(
            "No spectral _estimates provided".to_string(),
        ));
    }

    let n_freqs = spectral_estimates[0].len();
    let k = spectral_estimates.len() as f64;
    let dof = 2.0 * k; // Degrees of freedom for chi-squared distribution

    // Chi-squared critical values for confidence interval
    let alpha = 1.0 - confidence_level;
    let chi2_lower = dof / chi_squared_inverse_cdf(1.0 - alpha / 2.0, dof);
    let chi2_upper = dof / chi_squared_inverse_cdf(alpha / 2.0, dof);

    let mut lower_bounds = Vec::with_capacity(n_freqs);
    let mut upper_bounds = Vec::with_capacity(n_freqs);

    for i in 0..n_freqs {
        // Compute mean PSD at this frequency
        let mean_psd: f64 = spectral_estimates.iter().map(|psd| psd[i]).sum::<f64>() / k;

        lower_bounds.push(mean_psd * chi2_lower);
        upper_bounds.push(mean_psd * chi2_upper);
    }

    Ok((lower_bounds, upper_bounds))
}

/// Simple approximation for chi-squared inverse CDF
///
/// This is a simplified approximation suitable for confidence intervals.
/// For production use, consider using a more accurate implementation.
#[allow(dead_code)]
fn chi_squared_inverse_cdf(p: f64, dof: f64) -> f64 {
    // Wilson-Hilferty approximation for chi-squared distribution
    let h = 2.0 / (9.0 * dof);
    let z = normal_inverse_cdf(p);

    let term = 1.0 - h + z * ((h * 2.0) as f64).sqrt();
    dof * term.powi(3)
}

/// Simple approximation for standard normal inverse CDF
#[allow(dead_code)]
fn normal_inverse_cdf(p: f64) -> f64 {
    // Beasley-Springer-Moro algorithm approximation
    let a0 = 2.50662823884;
    let a1 = -18.61500062529;
    let a2 = 41.39119773534;
    let a3 = -25.44106049637;

    let b1 = -8.47351093090;
    let b2 = 23.08336743743;
    let b3 = -21.06224101826;
    let b4 = 3.13082909833;

    if p < 0.5 {
        let y = (2.0 * p).ln();
        let x = y + ((a3 * y + a2) * y + a1) * y + a0;
        let x = x / (((b4 * y + b3) * y + b2) * y + b1);
        -x
    } else {
        let y = (-2.0 * (1.0 - p).ln()).sqrt();
        let x = y + ((a3 * y + a2) * y + a1) * y + a0;
        let x = x / (((b4 * y + b3) * y + b2) * y + b1);
        x
    }
}

/// Extract eigenvalues from DPSS tapers
///
/// Extracts or estimates the concentration eigenvalues for DPSS tapers
/// used in multitaper analysis.
///
/// # Arguments
///
/// * `tapers` - DPSS taper matrix
/// * `nw` - Time-bandwidth product
///
/// # Returns
///
/// * Vector of eigenvalues
#[allow(dead_code)]
fn extract_taper_eigenvalues(tapers: &Array2<f64>, nw: f64) -> SignalResult<Array1<f64>> {
    let k = tapers.nrows();

    // For DPSS tapers, eigenvalues are approximately given by
    // the concentration ratio λ_k ≈ (1 + cos(πk/(2NW+1))) / 2
    let mut eigenvalues = Vec::with_capacity(k);

    for i in 0..k {
        let lambda = if i == 0 {
            // First eigenvalue is very close to 1
            1.0 - 1e-10
        } else {
            // Approximate eigenvalue formula
            let arg = PI * (i + 1) as f64 / (2.0 * nw + 1.0);
            (1.0 + arg.cos()) / 2.0
        };
        eigenvalues.push(lambda.max(1e-15)); // Ensure positive
    }

    // Sort in descending order (highest concentration first)
    eigenvalues.sort_by(|a, b| b.partial_cmp(a).unwrap());

    Ok(Array1::from_vec(eigenvalues))
}

// ================================================================================================
// MISSING FUNCTION STUB IMPLEMENTATIONS
// These functions are called but not implemented. They need proper implementation for production.
// ================================================================================================

/// TODO: Implement advanced enhanced ARMA estimation function
#[allow(dead_code)]
fn advanced_enhanced_arma_estimation(
    signal: &[f64],
    ar_order: usize,
    ma_order: usize,
    config: &AdvancedEnhancedConfig,
) -> SignalResult<AdvancedEnhancedARMAResult> {
    let signal_array = Array1::from_vec(signal.to_vec());
    advanced_enhanced_arma(&signal_array, ar_order, ma_order, config)
}

/// TODO: Implement minimum variance spectrum estimation
#[allow(dead_code)]
fn minimum_variance_spectrum(
    covariance_matrix: &Array2<f64>,
    n_freqs: usize,
    fs: f64,
) -> SignalResult<SpectralEstimate> {
    // Stub implementation
    let frequencies = generate_frequency_grid(n_freqs, fs);
    let psd = vec![1.0; n_freqs]; // Placeholder flat spectrum

    Ok(SpectralEstimate { psd, frequencies })
}

/// TODO: Implement CAPON spectrum estimation
#[allow(dead_code)]
fn capon_spectrum_estimation(
    covariance_matrix: &Array2<f64>,
    n_freqs: usize,
    fs: f64,
) -> SignalResult<SpectralEstimate> {
    // Stub implementation
    let frequencies = generate_frequency_grid(n_freqs, fs);
    let psd = vec![1.0; n_freqs]; // Placeholder flat spectrum

    Ok(SpectralEstimate { psd, frequencies })
}

/// TODO: Implement DPSS taper generation
#[allow(dead_code)]
fn generate_dpss_tapers(n: usize, nw: f64, k: usize) -> SignalResult<Array2<f64>> {
    // Stub implementation - generates simple tapers
    let mut tapers = Array2::zeros((k, n));

    for i in 0..k {
        for j in 0..n {
            // Simple window approximation (not true DPSS)
            let t = j as f64 / n as f64;
            let taper_val = (PI * (i + 1) as f64 * t).sin();
            tapers[[i, j]] = taper_val;
        }
    }

    Ok(tapers)
}

/// TODO: Implement MUSIC spectrum estimation
#[allow(dead_code)]
fn music_spectrum_estimation(
    eigen_result: &EigenDecompositionResult,
    num_signals: usize,
    n_freqs: usize,
    fs: f64,
) -> SignalResult<SpectralEstimate> {
    // Stub implementation
    let frequencies = generate_frequency_grid(n_freqs, fs);
    let psd = vec![1.0; n_freqs]; // Placeholder

    Ok(SpectralEstimate { psd, frequencies })
}

/// TODO: Implement ESPRIT frequency estimation
#[allow(dead_code)]
fn esprit_frequency_estimation(
    eigen_result: &EigenDecompositionResult,
    num_signals: usize,
    subspace_dimension: usize,
) -> SignalResult<SpectralEstimate> {
    // Stub implementation
    let frequencies = vec![0.0; num_signals];
    let psd = vec![1.0; num_signals]; // Placeholder

    Ok(SpectralEstimate { psd, frequencies })
}

/// TODO: Implement Hankel matrix creation
#[allow(dead_code)]
fn create_hankel_matrix(_signal: &[f64], subspacedimension: usize) -> SignalResult<Array2<f64>> {
    let n = signal.len();
    if n < subspace_dimension {
        return Err(SignalError::ValueError(
            "Signal too short for Hankel matrix".to_string(),
        ));
    }

    // Stub implementation - creates a simple matrix
    let rows = n - subspace_dimension + 1;
    let mut hankel = Array2::zeros((rows, subspace_dimension));

    for i in 0..rows {
        for j in 0..subspace_dimension {
            if i + j < n {
                hankel[[i, j]] = signal[i + j];
            }
        }
    }

    Ok(hankel)
}

/// TODO: Implement sample covariance computation
#[allow(dead_code)]
fn compute_sample_covariance(_datamatrix: &Array2<f64>) -> SignalResult<Array2<f64>> {
    let (rows, cols) = data_matrix.dim();
    let mut covariance = Array2::zeros((cols, cols));

    // Simple covariance estimation
    for i in 0..cols {
        for j in 0..cols {
            let mut sum = 0.0;
            for k in 0..rows {
                sum += data_matrix[[k, i]] * data_matrix[[k, j]];
            }
            covariance[[i, j]] = sum / rows as f64;
        }
    }

    Ok(covariance)
}

/// TODO: Implement signal number estimation
#[allow(dead_code)]
fn estimate_number_of_signals(
    eigenvalues: &Array1<f64>,
    config: &HighResolutionConfig,
) -> SignalResult<usize> {
    // Stub implementation - simple threshold-based detection
    let threshold = eigenvalues.iter().sum::<f64>() / eigenvalues.len() as f64 * 0.1;
    let num_signals = eigenvalues.iter().filter(|&&x| x > threshold).count();
    Ok(num_signals.max(1))
}

/// TODO: Implement adaptive order selection
#[allow(dead_code)]
fn select_optimal_order_adaptive(
    _signal: &[f64],
    config: &AdaptiveARConfig,
) -> SignalResult<usize> {
    // Stub implementation - returns default order
    Ok(config.max_order.min(10))
}

/// TODO: Implement AR estimation with forgetting factor
#[allow(dead_code)]
fn estimate_ar_with_forgetting(
    signal: &[f64],
    order: usize,
    forgetting_factor: f64,
    config: &AdaptiveARConfig,
) -> SignalResult<AdaptiveARResult> {
    // Stub implementation
    let coeffs = Array1::ones(order + 1);
    let noise_variance = 1.0;

    Ok(AdaptiveARResult {
        time_centers: vec![0.5],
        ar_coefficients: vec![coeffs],
        orders: vec![order],
        spectral_estimates: vec![vec![1.0; config.n_freqs]],
        window_size: signal.len(),
        hop_size: signal.len() / 4,
        total_windows: 1,
    })
}

/// TODO: Implement weighted ARMA estimation
#[allow(dead_code)]
fn weighted_arma_estimation(
    signal: &[f64],
    weights: &Array1<f64>,
    ar_order: usize,
    ma_order: usize,
    config: &RobustParametricConfig,
) -> SignalResult<WeightedARMAResult> {
    // Stub implementation
    let ar_coeffs = Array1::ones(ar_order + 1);
    let ma_coeffs = Array1::ones(ma_order + 1);

    Ok(WeightedARMAResult {
        ar_coeffs,
        ma_coeffs,
        weighted_residuals: Array1::zeros(signal.len()),
        effective_sample_size: signal.len() as f64,
    })
}

/// Eigenvalue decomposition result structure
#[derive(Debug, Clone)]
pub struct EigenDecompositionResult {
    pub eigenvalues: Array1<f64>,
    pub eigenvectors: Array2<f64>,
}

/// Weighted ARMA result structure
#[derive(Debug, Clone)]
pub struct WeightedARMAResult {
    pub ar_coeffs: Array1<f64>,
    pub ma_coeffs: Array1<f64>,
    pub weighted_residuals: Array1<f64>,
    pub effective_sample_size: f64,
}
