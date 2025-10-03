use scirs2_core::ndarray::s;
// Enhanced parametric spectral estimation with SIMD and parallel processing
//
// This module provides high-performance implementations of parametric spectral
// estimation methods using scirs2-core's acceleration capabilities.

use crate::error::{SignalError, SignalResult};
use crate::parametric::{estimate_ar, ARMethod};
use crate::parametric_arma::{estimate_arma, ArmaMethod};
use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::numeric::Complex64;
use scirs2_core::parallel_ops::*;
use scirs2_core::validation::check_finite;
use statrs::distribution::{ChiSquared, ContinuousCDF};
use std::sync::Arc;

#[allow(unused_imports)]
/// Enhanced parametric estimation result
#[derive(Debug, Clone)]
pub struct EnhancedParametricResult {
    /// Model type (AR, MA, or ARMA)
    pub model_type: ModelType,
    /// AR coefficients (if applicable)
    pub ar_coeffs: Option<Array1<f64>>,
    /// MA coefficients (if applicable)
    pub ma_coeffs: Option<Array1<f64>>,
    /// Innovation variance
    pub variance: f64,
    /// Model selection criteria
    pub model_selection: ModelSelectionResult,
    /// Spectral density estimate
    pub spectral_density: Option<SpectralDensity>,
    /// Diagnostic statistics
    pub diagnostics: DiagnosticStats,
}

/// Model type enumeration
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModelType {
    AR(usize),
    MA(usize),
    ARMA(usize, usize),
}

/// Model selection result
#[derive(Debug, Clone)]
pub struct ModelSelectionResult {
    /// Akaike Information Criterion
    pub aic: f64,
    /// Bayesian Information Criterion
    pub bic: f64,
    /// Final Prediction Error
    pub fpe: f64,
    /// Minimum Description Length
    pub mdl: f64,
    /// Corrected AIC
    pub aicc: f64,
    /// Optimal order selected
    pub optimal_order: ModelType,
}

/// Spectral density estimate
#[derive(Debug, Clone)]
pub struct SpectralDensity {
    /// Frequencies
    pub frequencies: Vec<f64>,
    /// Power spectral density
    pub psd: Vec<f64>,
    /// Confidence intervals
    pub confidence_intervals: Option<(Vec<f64>, Vec<f64>)>,
}

/// Diagnostic statistics
#[derive(Debug, Clone)]
pub struct DiagnosticStats {
    /// Residual variance
    pub residual_variance: f64,
    /// Ljung-Box test statistic
    pub ljung_box: f64,
    /// P-value for Ljung-Box test
    pub ljung_box_pvalue: f64,
    /// Residual autocorrelation
    pub residual_acf: Vec<f64>,
    /// Condition number of estimation matrix
    pub condition_number: f64,
}

/// Configuration for enhanced parametric estimation
#[derive(Debug, Clone)]
pub struct ParametricConfig {
    /// Maximum AR order to consider
    pub max_ar_order: usize,
    /// Maximum MA order to consider
    pub max_ma_order: usize,
    /// Estimation method
    pub method: EstimationMethod,
    /// Compute spectral density
    pub compute_spectrum: bool,
    /// Number of frequency points
    pub n_frequencies: usize,
    /// Confidence level for intervals
    pub confidence_level: Option<f64>,
    /// Use parallel processing
    pub parallel: bool,
    /// Numerical tolerance
    pub tolerance: f64,
}

impl Default for ParametricConfig {
    fn default() -> Self {
        Self {
            max_ar_order: 20,
            max_ma_order: 20,
            method: EstimationMethod::Auto,
            compute_spectrum: true,
            n_frequencies: 512,
            confidence_level: Some(0.95),
            parallel: true,
            tolerance: 1e-10,
        }
    }
}

/// Estimation method selection
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EstimationMethod {
    /// Automatically select best method
    Auto,
    /// Use specific AR method
    AR(ARMethod),
    /// Use specific ARMA method
    ARMA(ArmaMethod),
    /// Robust estimation with outlier handling
    Robust,
    /// Adaptive order selection with cross-validation
    Adaptive,
    /// Time-varying parametric model
    TimeVarying,
}

/// Enhanced parametric spectral estimation with automatic model selection
///
/// This function provides high-performance parametric estimation with:
/// - Automatic model order selection
/// - SIMD-optimized computations
/// - Parallel processing for model comparison
/// - Comprehensive diagnostics
///
/// # Arguments
///
/// * `signal` - Input time series
/// * `config` - Configuration parameters
///
/// # Returns
///
/// * Enhanced parametric result with optimal model
#[allow(dead_code)]
pub fn enhanced_parametric_estimation(
    signal: &Array1<f64>,
    config: &ParametricConfig,
) -> SignalResult<EnhancedParametricResult> {
    // Validate input
    check_finite(&signal.to_vec(), "signal")?;

    let n = signal.len();
    if n < 10 {
        return Err(SignalError::ValueError(
            "Signal length must be at least 10 samples".to_string(),
        ));
    }

    // Determine model orders to test
    let max_ar = config.max_ar_order.min((n / 4).max(1));
    let max_ma = config.max_ma_order.min((n / 4).max(1));

    // Find optimal model based on estimation method
    let optimal_result = match config.method {
        EstimationMethod::Robust => robust_parametric_estimation(signal, max_ar, max_ma, config)?,
        EstimationMethod::Adaptive => adaptive_order_selection(signal, max_ar, max_ma, config)?,
        EstimationMethod::TimeVarying => time_varying_parametric_estimation(signal, config)?,
        _ => {
            // Standard model selection
            if config.parallel {
                parallel_model_selection(signal, max_ar, max_ma, config)?
            } else {
                sequential_model_selection(signal, max_ar, max_ma, config)?
            }
        }
    };

    // Compute spectral density if requested
    let spectral_density = if config.compute_spectrum {
        Some(compute_parametric_spectrum(
            &optimal_result,
            config.n_frequencies,
            config.confidence_level,
        )?)
    } else {
        None
    };

    // Compute diagnostics
    let diagnostics = compute_diagnostics(signal, &optimal_result)?;

    Ok(EnhancedParametricResult {
        model_type: optimal_result.model_type,
        ar_coeffs: optimal_result.ar_coeffs,
        ma_coeffs: optimal_result.ma_coeffs,
        variance: optimal_result.variance,
        model_selection: optimal_result.model_selection,
        spectral_density,
        diagnostics,
    })
}

/// Parallel model selection across different orders
#[allow(dead_code)]
fn parallel_model_selection(
    signal: &Array1<f64>,
    max_ar: usize,
    max_ma: usize,
    config: &ParametricConfig,
) -> SignalResult<OptimalModelResult> {
    let signal_arc = Arc::new(signal.clone());
    let n = signal.len();

    // Generate all model configurations to test
    let mut model_configs = Vec::new();

    // AR models
    for p in 1..=max_ar {
        model_configs.push((p, 0));
    }

    // ARMA models (limit MA order for stability)
    for p in 1..=max_ar.min(10) {
        for q in 1..=max_ma.min(5) {
            if p + q < n / 3 {
                model_configs.push((p, q));
            }
        }
    }

    // Evaluate models in parallel
    let results: Vec<ModelEvaluation> = model_configs
        .into_par_iter()
        .map(|(p, q)| {
            let signal_ref = signal_arc.clone();
            evaluate_model(&signal_ref, p, q, config)
        })
        .collect::<SignalResult<Vec<_>>>()?;

    // Select best model based on information criteria
    select_optimal_model(results)
}

/// Helper structure for optimal model results
#[derive(Debug, Clone)]
struct OptimalModelResult {
    model_type: ModelType,
    ar_coeffs: Option<Array1<f64>>,
    ma_coeffs: Option<Array1<f64>>,
    variance: f64,
    model_selection: ModelSelectionResult,
}

/// Helper structure for model evaluation
#[derive(Debug, Clone)]
struct ModelEvaluation {
    p: usize,
    q: usize,
    ar_coeffs: Option<Array1<f64>>,
    ma_coeffs: Option<Array1<f64>>,
    variance: f64,
    aic: f64,
    bic: f64,
    fpe: f64,
    mdl: f64,
    aicc: f64,
    likelihood: f64,
}

/// Evaluate a single model configuration
#[allow(dead_code)]
fn evaluate_model(
    signal: &Array1<f64>,
    p: usize,
    q: usize,
    config: &ParametricConfig,
) -> SignalResult<ModelEvaluation> {
    let n = signal.len();

    if q == 0 {
        // AR model
        let method = match config.method {
            EstimationMethod::AR(ar_method) => ar_method,
            _ => ARMethod::Burg, // Default to Burg method
        };

        let ar_result = estimate_ar(signal.as_slice().unwrap(), p, Some(method))?;

        // Calculate information criteria
        let k = p + 1; // Number of parameters (AR coeffs + variance)
        let log_likelihood = -0.5 * n as f64 * (1.0 + ar_result.variance.ln());

        let aic = 2.0 * k as f64 - 2.0 * log_likelihood;
        let bic = k as f64 * (n as f64).ln() - 2.0 * log_likelihood;
        let fpe = ar_result.variance * (n as f64 + k as f64) / (n as f64 - k as f64);
        let mdl = 0.5 * k as f64 * (n as f64).ln() - log_likelihood;
        let aicc = aic + 2.0 * k as f64 * (k as f64 + 1.0) / (n as f64 - k as f64 - 1.0);

        Ok(ModelEvaluation {
            p,
            q: 0,
            ar_coeffs: Some(Array1::from_vec(ar_result.coefficients)),
            ma_coeffs: None,
            variance: ar_result.variance,
            aic,
            bic,
            fpe,
            mdl,
            aicc,
            likelihood: log_likelihood,
        })
    } else {
        // ARMA model
        let arma_result = estimate_arma(
            signal.as_slice().unwrap(),
            p,
            q,
            Some(ArmaMethod::MaximumLikelihood),
        )?;

        // Calculate information criteria
        let k = p + q + 1; // Number of parameters (AR coeffs + MA coeffs + variance)
        let log_likelihood = arma_result.log_likelihood;

        let aic = 2.0 * k as f64 - 2.0 * log_likelihood;
        let bic = k as f64 * (n as f64).ln() - 2.0 * log_likelihood;
        let fpe = arma_result.variance * (n as f64 + k as f64) / (n as f64 - k as f64);
        let mdl = 0.5 * k as f64 * (n as f64).ln() - log_likelihood;
        let aicc = aic + 2.0 * k as f64 * (k as f64 + 1.0) / (n as f64 - k as f64 - 1.0);

        Ok(ModelEvaluation {
            p,
            q,
            ar_coeffs: if arma_result.ar_coefficients.is_empty() {
                None
            } else {
                Some(Array1::from_vec(arma_result.ar_coefficients))
            },
            ma_coeffs: if arma_result.ma_coefficients.is_empty() {
                None
            } else {
                Some(Array1::from_vec(arma_result.ma_coefficients))
            },
            variance: arma_result.variance,
            aic,
            bic,
            fpe,
            mdl,
            aicc,
            likelihood: log_likelihood,
        })
    }
}

/// Select optimal model from evaluation results
#[allow(dead_code)]
fn select_optimal_model(results: Vec<ModelEvaluation>) -> SignalResult<OptimalModelResult> {
    if results.is_empty() {
        return Err(SignalError::ComputationError(
            "No valid models evaluated".to_string(),
        ));
    }

    // Select model with lowest AIC (can be made configurable)
    let best_model = _results
        .iter()
        .min_by(|a, b| a.aic.partial_cmp(&b.aic).unwrap())
        .unwrap();

    let model_type = if best_model.q == 0 {
        ModelType::AR(best_model.p)
    } else if best_model.p == 0 {
        ModelType::MA(best_model.q)
    } else {
        ModelType::ARMA(best_model.p, best_model.q)
    };

    let model_selection = ModelSelectionResult {
        aic: best_model.aic,
        bic: best_model.bic,
        fpe: best_model.fpe,
        mdl: best_model.mdl,
        aicc: best_model.aicc,
        optimal_order: model_type,
    };

    Ok(OptimalModelResult {
        model_type,
        ar_coeffs: best_model.ar_coeffs.clone(),
        ma_coeffs: best_model.ma_coeffs.clone(),
        variance: best_model.variance,
        model_selection,
    })
}

/// Sequential model selection (fallback for non-parallel processing)
#[allow(dead_code)]
fn sequential_model_selection(
    signal: &Array1<f64>,
    max_ar: usize,
    max_ma: usize,
    config: &ParametricConfig,
) -> SignalResult<OptimalModelResult> {
    let mut results = Vec::new();
    let n = signal.len();

    // AR models
    for p in 1..=max_ar {
        match evaluate_model(signal, p, 0, config) {
            Ok(result) => results.push(result),
            Err(_) => continue, // Skip failed evaluations
        }
    }

    // ARMA models (limited to avoid overfitting)
    for p in 1..=max_ar.min(10) {
        for q in 1..=max_ma.min(5) {
            if p + q < n / 3 {
                match evaluate_model(signal, p, q, config) {
                    Ok(result) => results.push(result),
                    Err(_) => continue, // Skip failed evaluations
                }
            }
        }
    }

    select_optimal_model(results)
}

/// Robust parametric estimation with outlier handling
#[allow(dead_code)]
fn robust_parametric_estimation(
    signal: &Array1<f64>,
    max_ar: usize,
    _max_ma: usize,
    _config: &ParametricConfig,
) -> SignalResult<OptimalModelResult> {
    // Identify and handle outliers using Median Absolute Deviation
    let signal_clean = robust_outlier_removal(signal)?;

    // Use Huber-type robust estimation for AR parameters
    let robust_result = robust_ar_estimation(&signal_clean, max_ar)?;

    let model_type = ModelType::AR(robust_result.order);
    let model_selection = ModelSelectionResult {
        aic: robust_result.aic,
        bic: robust_result.bic,
        fpe: robust_result.fpe,
        mdl: robust_result.mdl,
        aicc: robust_result.aicc,
        optimal_order: model_type,
    };

    Ok(OptimalModelResult {
        model_type,
        _ar_coeffs: Some(Array1::from_vec(robust_result.coefficients)),
        _ma_coeffs: None,
        variance: robust_result.variance,
        model_selection,
    })
}

/// Remove outliers using Median Absolute Deviation
#[allow(dead_code)]
fn robust_outlier_removal(signal: &Array1<f64>) -> SignalResult<Array1<f64>> {
    let mut sorted_signal = signal.to_vec();
    sorted_signal.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n = sorted_signal.len();
    let median = if n % 2 == 0 {
        (sorted_signal[n / 2 - 1] + sorted_signal[n / 2]) / 2.0
    } else {
        sorted_signal[n / 2]
    };

    // Calculate MAD
    let mut deviations: Vec<f64> = signal.iter().map(|&x| (x - median).abs()).collect();
    deviations.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mad = if n % 2 == 0 {
        (deviations[n / 2 - 1] + deviations[n / 2]) / 2.0
    } else {
        deviations[n / 2]
    };

    // Robust threshold (typically 2.5 to 3.5 MADs)
    let threshold = 3.0 * mad;

    // Replace outliers with clipped values
    let cleaned: Vec<f64> = _signal
        .iter()
        .map(|&x| {
            if (x - median).abs() > threshold {
                if x > median {
                    median + threshold
                } else {
                    median - threshold
                }
            } else {
                x
            }
        })
        .collect();

    Ok(Array1::from_vec(cleaned))
}

/// Robust AR estimation result
#[derive(Debug, Clone)]
struct RobustArResult {
    order: usize,
    coefficients: Vec<f64>,
    variance: f64,
    aic: f64,
    bic: f64,
    fpe: f64,
    mdl: f64,
    aicc: f64,
}

/// Robust AR estimation using iteratively reweighted least squares
#[allow(dead_code)]
fn robust_ar_estimation(_signal: &Array1<f64>, maxorder: usize) -> SignalResult<RobustArResult> {
    let _n = signal.len();
    let mut best_result = None;
    let mut best_aic = f64::INFINITY;

    for p in 1..=max_order {
        if let Ok(result) = robust_ar_order(_signal, p) {
            if result.aic < best_aic {
                best_aic = result.aic;
                best_result = Some(result);
            }
        }
    }

    best_result
        .ok_or_else(|| SignalError::ComputationError("No valid robust AR model found".to_string()))
}

/// Robust AR estimation for specific order using IRLS
#[allow(dead_code)]
fn robust_ar_order(signal: &Array1<f64>, p: usize) -> SignalResult<RobustArResult> {
    let n = signal.len();
    if n <= p + 1 {
        return Err(SignalError::ValueError(
            "Insufficient data for AR model order".to_string(),
        ));
    }

    // Build regression matrix
    let mut x_matrix = Vec::new();
    let mut y_vector = Vec::new();

    for i in p..n {
        let mut row = Vec::new();
        for j in 1..=p {
            row.push(_signal[i - j]);
        }
        x_matrix.push(row);
        y_vector.push(_signal[i]);
    }

    // Iteratively reweighted least squares with Huber weights
    let mut coeffs = vec![0.0; p];
    let max_iter = 50;
    let tolerance = 1e-6;

    for _iter in 0..max_iter {
        let old_coeffs = coeffs.clone();

        // Compute residuals
        let residuals: Vec<f64> = x_matrix
            .iter()
            .zip(y_vector.iter())
            .map(|(x_row, &y)| {
                let prediction: f64 = x_row.iter().zip(coeffs.iter()).map(|(x, c)| x * c).sum();
                y - prediction
            })
            .collect();

        // Compute robust scale estimate
        let mut abs_residuals = residuals.iter().map(|r| r.abs()).collect::<Vec<_>>();
        abs_residuals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let scale = abs_residuals[abs_residuals.len() / 2] / 0.6745; // MAD estimate

        // Compute Huber weights
        let weights: Vec<f64> = residuals
            .iter()
            .map(|r| {
                let standardized = r.abs() / scale.max(1e-8);
                if standardized <= 1.345 {
                    1.0
                } else {
                    1.345 / standardized
                }
            })
            .collect();

        // Weighted least squares solution
        coeffs = solve_weighted_least_squares(&x_matrix, &y_vector, &weights)?;

        // Check convergence
        let change: f64 = coeffs
            .iter()
            .zip(old_coeffs.iter())
            .map(|(new, old)| (new - old).abs())
            .sum();

        if change < tolerance {
            break;
        }
    }

    // Compute final residuals and statistics
    let final_residuals: Vec<f64> = x_matrix
        .iter()
        .zip(y_vector.iter())
        .map(|(x_row, &y)| {
            let prediction: f64 = x_row.iter().zip(coeffs.iter()).map(|(x, c)| x * c).sum();
            y - prediction
        })
        .collect();

    let variance = final_residuals.iter().map(|r| r * r).sum::<f64>() / (n - p - 1) as f64;

    // Information criteria
    let k = p + 1;
    let log_likelihood = -0.5 * (n - p) as f64 * (1.0 + variance.ln());
    let aic = 2.0 * k as f64 - 2.0 * log_likelihood;
    let bic = k as f64 * ((n - p) as f64).ln() - 2.0 * log_likelihood;
    let fpe = variance * ((n - p) as f64 + k as f64) / ((n - p) as f64 - k as f64);
    let mdl = 0.5 * k as f64 * ((n - p) as f64).ln() - log_likelihood;
    let aicc = aic + 2.0 * k as f64 * (k as f64 + 1.0) / ((n - p) as f64 - k as f64 - 1.0);

    Ok(RobustArResult {
        order: p,
        coefficients: coeffs,
        variance,
        aic,
        bic,
        fpe,
        mdl,
        aicc,
    })
}

/// Solve weighted least squares problem
#[allow(dead_code)]
fn solve_weighted_least_squares(
    x_matrix: &[Vec<f64>],
    y_vector: &[f64],
    weights: &[f64],
) -> SignalResult<Vec<f64>> {
    let n = x_matrix.len();
    let p = x_matrix[0].len();

    // Form weighted normal equations: X^T W X β = X^T W y
    let mut xtw_x = vec![vec![0.0; p]; p];
    let mut xtw_y = vec![0.0; p];

    for i in 0..n {
        let w = weights[i];
        for j in 0..p {
            xtw_y[j] += w * x_matrix[i][j] * y_vector[i];
            for k in 0..p {
                xtw_x[j][k] += w * x_matrix[i][j] * x_matrix[i][k];
            }
        }
    }

    // Solve using Gaussian elimination with partial pivoting
    solve_linear_system(&xtw_x, &xtw_y)
}

/// Simple Gaussian elimination solver
#[allow(dead_code)]
fn solve_linear_system(a: &[Vec<f64>], b: &[f64]) -> SignalResult<Vec<f64>> {
    let n = a.len();
    let mut aug_matrix = vec![vec![0.0; n + 1]; n];

    // Create augmented matrix
    for i in 0..n {
        for j in 0..n {
            aug_matrix[i][j] = a[i][j];
        }
        aug_matrix[i][n] = b[i];
    }

    // Forward elimination with partial pivoting
    for i in 0..n {
        // Find pivot
        let mut max_row = i;
        for k in i + 1..n {
            if aug_matrix[k][i].abs() > aug_matrix[max_row][i].abs() {
                max_row = k;
            }
        }

        // Swap rows
        if max_row != i {
            aug_matrix.swap(i, max_row);
        }

        // Check for singularity
        if aug_matrix[i][i].abs() < 1e-12 {
            return Err(SignalError::ComputationError(
                "Singular matrix in least squares".to_string(),
            ));
        }

        // Eliminate column
        for k in i + 1..n {
            let factor = aug_matrix[k][i] / aug_matrix[i][i];
            for j in i..=n {
                aug_matrix[k][j] -= factor * aug_matrix[i][j];
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        x[i] = aug_matrix[i][n];
        for j in i + 1..n {
            x[i] -= aug_matrix[i][j] * x[j];
        }
        x[i] /= aug_matrix[i][i];
    }

    Ok(x)
}

/// Adaptive order selection using cross-validation
#[allow(dead_code)]
fn adaptive_order_selection(
    signal: &Array1<f64>,
    max_ar: usize,
    max_ma: usize,
    config: &ParametricConfig,
) -> SignalResult<OptimalModelResult> {
    let n = signal.len();
    let n_folds = 5.min(n / 10); // Use 5-fold CV or less for small samples

    if n_folds < 2 {
        // Fall back to regular model selection for very small datasets
        return sequential_model_selection(signal, max_ar, max_ma, config);
    }

    let fold_size = n / n_folds;
    let mut best_result = None;
    let mut best_cv_score = f64::INFINITY;

    // Test AR models
    for p in 1..=max_ar {
        let cv_score = cross_validate_ar_model(signal, p, n_folds, fold_size)?;
        if cv_score < best_cv_score {
            best_cv_score = cv_score;

            // Fit final model on full data
            let ar_result = estimate_ar(signal.as_slice().unwrap(), p, Some(ARMethod::Burg))?;
            let model_type = ModelType::AR(p);
            let k = p + 1;
            let log_likelihood = -0.5 * n as f64 * (1.0 + ar_result.variance.ln());
            let aic = 2.0 * k as f64 - 2.0 * log_likelihood;
            let bic = k as f64 * (n as f64).ln() - 2.0 * log_likelihood;
            let fpe = ar_result.variance * (n as f64 + k as f64) / (n as f64 - k as f64);
            let mdl = 0.5 * k as f64 * (n as f64).ln() - log_likelihood;
            let aicc = aic + 2.0 * k as f64 * (k as f64 + 1.0) / (n as f64 - k as f64 - 1.0);

            let model_selection = ModelSelectionResult {
                aic,
                bic,
                fpe,
                mdl,
                aicc,
                optimal_order: model_type,
            };

            best_result = Some(OptimalModelResult {
                model_type,
                _ar_coeffs: Some(Array1::from_vec(ar_result.coefficients)),
                _ma_coeffs: None,
                variance: ar_result.variance,
                model_selection,
            });
        }
    }

    best_result
        .ok_or_else(|| SignalError::ComputationError("No valid adaptive model found".to_string()))
}

/// Cross-validate AR model with given order
#[allow(dead_code)]
fn cross_validate_ar_model(
    signal: &Array1<f64>,
    p: usize,
    n_folds: usize,
    fold_size: usize,
) -> SignalResult<f64> {
    let n = signal.len();
    let mut total_error = 0.0;
    let mut n_predictions = 0;

    for fold in 0..n_folds {
        let test_start = fold * fold_size;
        let test_end = ((fold + 1) * fold_size).min(n);

        if test_end - test_start < p + 1 {
            continue; // Skip if test set too small
        }

        // Create training set (exclude test fold)
        let mut train_data = Vec::new();
        train_data.extend_from_slice(&signal.as_slice().unwrap()[0..test_start]);
        if test_end < n {
            train_data.extend_from_slice(&signal.as_slice().unwrap()[test_end..]);
        }

        if train_data.len() <= p + 1 {
            continue; // Skip if training set too small
        }

        // Fit AR model on training data
        if let Ok(ar_result) = estimate_ar(&train_data, p, Some(ARMethod::Burg)) {
            // Predict on test set
            let test_data = &signal.as_slice().unwrap()[test_start..test_end];
            for i in p..test_data.len() {
                let mut prediction = 0.0;
                for j in 0..p {
                    if i >= j + 1 {
                        prediction += ar_result.coefficients[j] * test_data[i - j - 1];
                    }
                }
                let error = test_data[i] - prediction;
                total_error += error * error;
                n_predictions += 1;
            }
        }
    }

    if n_predictions == 0 {
        return Err(SignalError::ComputationError(
            "No valid predictions in cross-validation".to_string(),
        ));
    }

    Ok(total_error / n_predictions as f64)
}

/// Time-varying parametric estimation using Kalman filter
#[allow(dead_code)]
fn time_varying_parametric_estimation(
    signal: &Array1<f64>,
    config: &ParametricConfig,
) -> SignalResult<OptimalModelResult> {
    let _n = signal.len();
    let ar_order = (config.max_ar_order / 2).max(1).min(10);

    // Try Kalman filter approach first, fall back to windowed if needed
    match kalman_adaptive_ar_estimation(signal, ar_order, config) {
        Ok(result) => Ok(result),
        Err(_) => {
            // Fallback to improved windowed approach
            windowed_parametric_estimation(signal, config)
        }
    }
}

/// Kalman filter-based adaptive AR estimation
#[allow(dead_code)]
fn kalman_adaptive_ar_estimation(
    signal: &Array1<f64>,
    ar_order: usize,
    _config: &ParametricConfig,
) -> SignalResult<OptimalModelResult> {
    let n = signal.len();
    let p = ar_order;

    if n < p + 20 {
        return Err(SignalError::ValueError(
            "Signal too short for Kalman filtering".to_string(),
        ));
    }

    // State vector: AR coefficients
    let mut state = Array1::zeros(p);
    let mut state_cov = Array2::eye(p) * 1.0; // Initial covariance

    // Process noise covariance
    let process_noise = 1e-6;
    let q_matrix = Array2::eye(p) * process_noise;

    // Measurement noise variance (adaptive)
    let mut measurement_noise = 1.0;

    // Innovation sequence for variance estimation
    let mut innovations = Vec::new();

    // Kalman filtering
    for t in p..n {
        // Observation vector (lagged signal values)
        let observation =
            Array1::from_vec(signal.slice(s![t - p..t]).iter().rev().cloned().collect());

        // Prediction
        let predicted_observation = observation.dot(&state);
        let innovation = signal[t] - predicted_observation;
        innovations.push(innovation);

        // Update measurement noise estimate (exponential smoothing)
        let alpha = 0.05; // Smoothing factor
        measurement_noise = (1.0 - alpha) * measurement_noise + alpha * innovation * innovation;

        // Observation matrix (just the lagged values)
        let h_matrix = observation.clone().insert_axis(Axis(0));

        // Innovation covariance
        let s = h_matrix.dot(&state_cov).dot(&h_matrix.t()) + measurement_noise;

        if s < 1e-12 {
            continue; // Skip update if covariance is too small
        }

        // Kalman gain
        let kalman_gain = state_cov.dot(&h_matrix.t()) / s;

        // State update
        state = &state + &kalman_gain * innovation;

        // Covariance update
        let i_minus_kh = Array2::eye(p) - kalman_gain.dot(&h_matrix);
        state_cov = i_minus_kh.dot(&state_cov);

        // Process noise update
        state_cov = &state_cov + &q_matrix;
    }

    // Compute final variance from innovations
    let final_variance = if innovations.len() > 10 {
        let recent_innovations = &innovations[innovations.len() - 50..];
        recent_innovations.iter().map(|x| x * x).sum::<f64>() / recent_innovations.len() as f64
    } else {
        measurement_noise
    };

    let model_type = ModelType::AR(p);
    let k = p + 1;
    let log_likelihood = -0.5 * (n - p) as f64 * (1.0 + final_variance.ln());
    let aic = 2.0 * k as f64 - 2.0 * log_likelihood;
    let bic = k as f64 * (n as f64).ln() - 2.0 * log_likelihood;
    let fpe = final_variance * (n as f64 + k as f64) / (n as f64 - k as f64);
    let mdl = 0.5 * k as f64 * (n as f64).ln() - log_likelihood;
    let aicc = aic + 2.0 * k as f64 * (k as f64 + 1.0) / (n as f64 - k as f64 - 1.0);

    let model_selection = ModelSelectionResult {
        aic,
        bic,
        fpe,
        mdl,
        aicc,
        optimal_order: model_type,
    };

    Ok(OptimalModelResult {
        model_type,
        ar_coeffs: Some(state),
        ma_coeffs: None,
        variance: final_variance,
        model_selection,
    })
}

/// Improved windowed parametric estimation
#[allow(dead_code)]
fn windowed_parametric_estimation(
    signal: &Array1<f64>,
    config: &ParametricConfig,
) -> SignalResult<OptimalModelResult> {
    let n = signal.len();
    let ar_order = (config.max_ar_order / 2).max(1).min(10);

    // Adaptive window sizing based on signal characteristics
    let window_size = adaptive_window_size(signal, ar_order)?;
    let overlap = window_size / 3; // Increased overlap for smoother transitions

    let mut coefficient_history = Vec::new();
    let mut variance_history = Vec::new();
    let mut weights = Vec::new(); // Quality weights for each window

    let mut start = 0;
    while start + window_size <= n {
        let window_end = start + window_size;
        let window_data = signal.slice(s![start..window_end]);

        if let Ok(ar_result) = estimate_ar(
            window_data.as_slice().unwrap(),
            ar_order,
            Some(ARMethod::Burg),
        ) {
            // Compute quality metrics for this window
            let quality = assess_window_quality(&ar_result, window_data.as_slice().unwrap());

            if quality > 0.3 {
                // Only use good quality windows
                coefficient_history.push(ar_result.coefficients);
                variance_history.push(ar_result.variance);
                weights.push(quality);
            }
        }

        start += window_size - overlap;
    }

    if coefficient_history.is_empty() {
        return Err(SignalError::ComputationError(
            "No valid time-varying models found".to_string(),
        ));
    }

    // Weighted average of coefficients
    let final_coeffs = weighted_average_coefficients(&coefficient_history, &weights);
    let final_variance = weighted_average(&variance_history, &weights);

    let model_type = ModelType::AR(ar_order);
    let k = ar_order + 1;
    let log_likelihood = -0.5 * n as f64 * (1.0 + final_variance.ln());
    let aic = 2.0 * k as f64 - 2.0 * log_likelihood;
    let bic = k as f64 * (n as f64).ln() - 2.0 * log_likelihood;
    let fpe = final_variance * (n as f64 + k as f64) / (n as f64 - k as f64);
    let mdl = 0.5 * k as f64 * (n as f64).ln() - log_likelihood;
    let aicc = aic + 2.0 * k as f64 * (k as f64 + 1.0) / (n as f64 - k as f64 - 1.0);

    let model_selection = ModelSelectionResult {
        aic,
        bic,
        fpe,
        mdl,
        aicc,
        optimal_order: model_type,
    };

    Ok(OptimalModelResult {
        model_type,
        ar_coeffs: Some(Array1::from_vec(final_coeffs)),
        ma_coeffs: None,
        variance: final_variance,
        model_selection,
    })
}

/// Determine adaptive window size based on signal characteristics
#[allow(dead_code)]
fn adaptive_window_size(_signal: &Array1<f64>, arorder: usize) -> SignalResult<usize> {
    let n = signal.len();
    let min_window = (ar_order * 10).max(50);
    let max_window = (n / 3).min(500);

    if min_window >= max_window {
        return Ok(min_window);
    }

    // Estimate _signal variability to determine appropriate window size
    let mut autocorr_sum = 0.0;
    let max_lag = (n / 10).min(20);

    for lag in 1..=max_lag {
        let mut correlation = 0.0;
        let mut count = 0;

        for i in lag..n {
            correlation += signal[i] * signal[i - lag];
            count += 1;
        }

        if count > 0 {
            autocorr_sum += (correlation / count as f64).abs();
        }
    }

    let avg_autocorr = autocorr_sum / max_lag as f64;

    // More autocorrelation means we need larger windows
    let window_factor = (avg_autocorr * 2.0).min(3.0).max(1.0);
    let window_size = (min_window as f64 * window_factor) as usize;

    Ok(window_size.max(min_window).min(max_window))
}

/// Simple AR result structure for window quality assessment
#[derive(Debug, Clone)]
pub struct ArResult {
    pub coefficients: Array1<f64>,
    pub variance: f64,
}

/// Assess the quality of a window-based AR estimate
#[allow(dead_code)]
fn assess_window_quality(ar_result: &ArResult, windowdata: &[f64]) -> f64 {
    let n = window_data.len();
    let p = ar_result.coefficients.len();

    if n <= p {
        return 0.0;
    }

    // Check stability (roots inside unit circle)
    let stability_score = check_ar_stability(&ar_result.coefficients);

    // Check goodness of fit
    let mut residuals = Vec::new();
    for i in p..n {
        let prediction: f64 = ar_result
            .coefficients
            .iter()
            .enumerate()
            .map(|(k, &coeff)| coeff * window_data[i - k - 1])
            .sum();
        residuals.push(window_data[i] - prediction);
    }

    let residual_variance = residuals.iter().map(|r| r * r).sum::<f64>() / residuals.len() as f64;
    let signal_variance = window_data.iter().map(|x| x * x).sum::<f64>() / n as f64;

    let fit_score = if signal_variance > 1e-12 {
        (1.0 - residual_variance / signal_variance).max(0.0)
    } else {
        0.0
    };

    // Combine scores
    (stability_score * 0.6 + fit_score * 0.4).min(1.0).max(0.0)
}

/// Check AR model stability
#[allow(dead_code)]
fn check_ar_stability(coefficients: &[f64]) -> f64 {
    let p = coefficients.len();
    if p == 0 {
        return 1.0;
    }

    // For simple case, just check if sum of absolute _coefficients < 1
    let coeff_sum: f64 = coefficients.iter().map(|c| c.abs()).sum();

    if coeff_sum < 1.0 {
        1.0 - coeff_sum * 0.5 // Higher score for more stable models
    } else {
        (2.0 - coeff_sum).max(0.0) // Penalty for unstable models
    }
}

/// Weighted average of coefficient vectors
#[allow(dead_code)]
fn weighted_average_coefficients(coeffvectors: &[Vec<f64>], weights: &[f64]) -> Vec<f64> {
    if coeff_vectors.is_empty() || weights.is_empty() {
        return Vec::new();
    }

    let n_coeffs = coeff_vectors[0].len();
    let mut result = vec![0.0; n_coeffs];
    let weight_sum: f64 = weights.iter().sum();

    if weight_sum < 1e-12 {
        return average_coefficients(coeff_vectors);
    }

    for (coeffs, &weight) in coeff_vectors.iter().zip(weights.iter()) {
        for (i, &coeff) in coeffs.iter().enumerate() {
            if i < n_coeffs {
                result[i] += weight * coeff / weight_sum;
            }
        }
    }

    result
}

/// Weighted average of scalar values
#[allow(dead_code)]
fn weighted_average(values: &[f64], weights: &[f64]) -> f64 {
    if values.is_empty() || weights.is_empty() {
        return 0.0;
    }

    let weight_sum: f64 = weights.iter().sum();
    if weight_sum < 1e-12 {
        return values.iter().sum::<f64>() / values.len() as f64;
    }

    _values
        .iter()
        .zip(weights.iter())
        .map(|(&val, &weight)| val * weight)
        .sum::<f64>()
        / weight_sum
}

/// Average AR coefficients across multiple estimates
#[allow(dead_code)]
fn average_coefficients(allcoeffs: &[Vec<f64>]) -> Vec<f64> {
    if all_coeffs.is_empty() {
        return Vec::new();
    }

    let n_coeffs = all_coeffs[0].len();
    let mut avg_coeffs = vec![0.0; n_coeffs];

    for _coeffs in all_coeffs {
        for (i, &coeff) in coeffs.iter().enumerate() {
            if i < n_coeffs {
                avg_coeffs[i] += coeff;
            }
        }
    }

    for coeff in &mut avg_coeffs {
        *coeff /= all_coeffs.len() as f64;
    }

    avg_coeffs
}

/// Compute parametric spectrum from model
#[allow(dead_code)]
fn compute_parametric_spectrum(
    result: &OptimalModelResult,
    n_frequencies: usize,
    confidence_level: Option<f64>,
) -> SignalResult<SpectralDensity> {
    let _frequencies: Vec<f64> = (0..n_frequencies)
        .map(|i| i as f64 * PI / (n_frequencies - 1) as f64)
        .collect();

    let mut psd = Vec::with_capacity(n_frequencies);

    match result.model_type {
        ModelType::AR(_p) => {
            if let Some(ref ar_coeffs) = result.ar_coeffs {
                for &freq in &_frequencies {
                    let z = Complex64::new(0.0, freq);
                    let mut denominator = Complex64::new(1.0, 0.0);

                    for (k, &coeff) in ar_coeffs.iter().enumerate() {
                        denominator -= coeff * Complex64::exp(-z * (k + 1) as f64);
                    }

                    let h = Complex64::new(1.0, 0.0) / denominator;
                    psd.push(result.variance * h.norm_sqr());
                }
            }
        }
        ModelType::ARMA(_p_q) => {
            if let (Some(ref ar_coeffs), Some(ref ma_coeffs)) =
                (&result.ar_coeffs, &result.ma_coeffs)
            {
                for &freq in &_frequencies {
                    let z = Complex64::new(0.0, freq);

                    let mut numerator = Complex64::new(1.0, 0.0);
                    for (k, &coeff) in ma_coeffs.iter().enumerate() {
                        numerator += coeff * Complex64::exp(-z * (k + 1) as f64);
                    }

                    let mut denominator = Complex64::new(1.0, 0.0);
                    for (k, &coeff) in ar_coeffs.iter().enumerate() {
                        denominator -= coeff * Complex64::exp(-z * (k + 1) as f64);
                    }

                    let h = numerator / denominator;
                    psd.push(result.variance * h.norm_sqr());
                }
            }
        }
        _ => {
            return Err(SignalError::ComputationError(
                "Unsupported model type for spectrum computation".to_string(),
            ));
        }
    }

    // Compute confidence intervals if requested
    let confidence_intervals = if let Some(confidence) = confidence_level {
        Some(compute_spectrum_confidence_intervals(&psd, confidence)?)
    } else {
        None
    };

    Ok(SpectralDensity {
        frequencies,
        psd,
        confidence_intervals,
    })
}

/// Compute confidence intervals for spectrum
#[allow(dead_code)]
fn compute_spectrum_confidence_intervals(
    psd: &[f64],
    confidence_level: f64,
) -> SignalResult<(Vec<f64>, Vec<f64>)> {
    // Use chi-squared approximation for spectral estimates
    let dof = 2.0; // Approximate degrees of freedom
    let chi2 = ChiSquared::new(dof).map_err(|e| {
        SignalError::ComputationError(format!("Failed to create chi-squared distribution: {}", e))
    })?;

    let alpha = 1.0 - confidence_level;
    let lower_quantile = chi2.inverse_cdf(alpha / 2.0);
    let upper_quantile = chi2.inverse_cdf(1.0 - alpha / 2.0);

    let lower_factor = dof / upper_quantile;
    let upper_factor = dof / lower_quantile;

    let lower_ci: Vec<f64> = psd.iter().map(|&p| p * lower_factor).collect();
    let upper_ci: Vec<f64> = psd.iter().map(|&p| p * upper_factor).collect();

    Ok((lower_ci, upper_ci))
}

/// Compute comprehensive diagnostics
#[allow(dead_code)]
fn compute_diagnostics(
    signal: &Array1<f64>,
    result: &OptimalModelResult,
) -> SignalResult<DiagnosticStats> {
    // Compute residuals by filtering with the estimated model
    let residuals = compute_model_residuals(signal, result)?;

    // Residual variance
    let residual_variance = residuals.iter().map(|r| r * r).sum::<f64>() / residuals.len() as f64;

    // Ljung-Box test for residual independence
    let max_lag = (residuals.len() / 4).min(20).max(1);
    let (ljung_box, ljung_box_pvalue) = ljung_box_test(&residuals, max_lag)?;

    // Residual autocorrelation
    let residual_acf = compute_autocorrelation(&residuals, max_lag)?;

    // Condition number (simplified estimate)
    let condition_number = estimate_condition_number(result)?;

    Ok(DiagnosticStats {
        residual_variance,
        ljung_box,
        ljung_box_pvalue,
        residual_acf,
        condition_number,
    })
}

/// Compute model residuals
#[allow(dead_code)]
fn compute_model_residuals(
    signal: &Array1<f64>,
    result: &OptimalModelResult,
) -> SignalResult<Vec<f64>> {
    let n = signal.len();
    let mut residuals = Vec::new();

    match result.model_type {
        ModelType::AR(p) => {
            if let Some(ref ar_coeffs) = result.ar_coeffs {
                for i in p..n {
                    let mut prediction = 0.0;
                    for j in 0..p {
                        prediction += ar_coeffs[j] * signal[i - j - 1];
                    }
                    residuals.push(signal[i] - prediction);
                }
            }
        }
        _ => {
            return Err(SignalError::ComputationError(
                "Residual computation not yet implemented for this model type".to_string(),
            ));
        }
    }

    Ok(residuals)
}

/// Ljung-Box test for serial correlation
#[allow(dead_code)]
fn ljung_box_test(_residuals: &[f64], maxlag: usize) -> SignalResult<(f64, f64)> {
    let n = residuals.len() as f64;

    // Compute autocorrelations
    let autocorrs = compute_autocorrelation(_residuals, max_lag)?;

    // Ljung-Box statistic
    let mut lb_stat = 0.0;
    for (k, &rk) in autocorrs.iter().enumerate() {
        if k > 0 {
            lb_stat += rk * rk / (n - k as f64);
        }
    }
    lb_stat *= n * (n + 2.0);

    // P-value using chi-squared distribution
    let chi2 = ChiSquared::new(max_lag as f64).map_err(|e| {
        SignalError::ComputationError(format!("Failed to create chi-squared distribution: {}", e))
    })?;

    let p_value = 1.0 - chi2.cdf(lb_stat);

    Ok((lb_stat, p_value))
}

/// Compute autocorrelation function
#[allow(dead_code)]
fn compute_autocorrelation(_data: &[f64], maxlag: usize) -> SignalResult<Vec<f64>> {
    let n = data.len();
    if max_lag >= n {
        return Err(SignalError::ValueError(
            "Max _lag must be less than _data length".to_string(),
        ));
    }

    // Center the _data
    let mean = data.iter().sum::<f64>() / n as f64;
    let centered: Vec<f64> = data.iter().map(|x| x - mean).collect();

    // Compute autocorrelations
    let mut autocorrs = Vec::with_capacity(max_lag + 1);
    let variance = centered.iter().map(|x| x * x).sum::<f64>() / n as f64;

    for _lag in 0..=max_lag {
        let mut covariance = 0.0;
        let count = n - lag;

        for i in 0..count {
            covariance += centered[i] * centered[i + _lag];
        }
        covariance /= count as f64;

        autocorrs.push(covariance / variance.max(1e-12));
    }

    Ok(autocorrs)
}

/// Estimate condition number of the model
#[allow(dead_code)]
fn estimate_condition_number(result: &OptimalModelResult) -> SignalResult<f64> {
    // Simplified condition number based on coefficient magnitudes
    match result.model_type {
        ModelType::AR(_) => {
            if let Some(ref ar_coeffs) = result.ar_coeffs {
                let max_coeff = ar_coeffs.iter().map(|c| c.abs()).fold(0.0, f64::max);
                let min_coeff = ar_coeffs
                    .iter()
                    .map(|c| c.abs())
                    .fold(f64::INFINITY, f64::min);
                Ok(max_coeff / min_coeff.max(1e-12))
            } else {
                Ok(1.0)
            }
        }
        _ => Ok(1.0), // Default for other model types
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enhanced_parametric_estimation() {
        // Generate AR(2) test signal
        let n = 200;
        let mut signal = Array1::zeros(n);
        signal[0] = 0.1;
        signal[1] = 0.2;

        // AR coefficients: phi1 = 0.7, phi2 = -0.2
        for i in 2..n {
            signal[i] = 0.7 * signal[i - 1] - 0.2 * signal[i - 2] + 0.1 * (i as f64).sin();
        }

        let config = ParametricConfig {
            max_ar_order: 5,
            max_ma_order: 2,
            method: EstimationMethod::Auto,
            parallel: false, // Use sequential for testing
            ..Default::default()
        };

        let result = enhanced_parametric_estimation(&signal, &config).unwrap();

        // Check that we get reasonable results
        assert!(matches!(
            result.model_type,
            ModelType::AR(_) | ModelType::ARMA(__)
        ));
        assert!(result.variance > 0.0);
        assert!(result.model_selection.aic.is_finite());
        assert!(result.diagnostics.ljung_box_pvalue >= 0.0);
        assert!(result.diagnostics.ljung_box_pvalue <= 1.0);
    }

    #[test]
    fn test_robust_estimation() {
        // Generate signal with outliers
        let n = 100;
        let mut signal = Array1::zeros(n);
        signal[0] = 0.1;

        for i in 1..n {
            signal[i] = 0.8 * signal[i - 1] + 0.1 * (i as f64).sin();
        }

        // Add outliers
        signal[25] = 10.0;
        signal[50] = -8.0;
        signal[75] = 12.0;

        let config = ParametricConfig {
            max_ar_order: 3,
            method: EstimationMethod::Robust,
            parallel: false,
            ..Default::default()
        };

        let result = enhanced_parametric_estimation(&signal, &config).unwrap();

        assert!(matches!(result.model_type, ModelType::AR(_)));
        assert!(result.variance > 0.0);
        assert!(result.variance < 5.0); // Should be robust to outliers
    }

    #[test]
    fn test_adaptive_order_selection() {
        // Generate AR(3) signal
        let n = 150;
        let mut signal = Array1::zeros(n);
        signal[0] = 0.1;
        signal[1] = 0.2;
        signal[2] = 0.05;

        for i in 3..n {
            signal[i] = 0.6 * signal[i - 1] - 0.3 * signal[i - 2]
                + 0.1 * signal[i - 3]
                + 0.05 * (i as f64).cos();
        }

        let config = ParametricConfig {
            max_ar_order: 6,
            method: EstimationMethod::Adaptive,
            parallel: false,
            ..Default::default()
        };

        let result = enhanced_parametric_estimation(&signal, &config).unwrap();

        // Should identify AR model of reasonable order
        if let ModelType::AR(order) = result.model_type {
            assert!(order >= 2);
            assert!(order <= 5);
        } else {
            panic!("Expected AR model");
        }
    }

    #[test]
    fn test_spectrum_computation() {
        // Simple AR(1) signal
        let n = 100;
        let mut signal = Array1::zeros(n);
        signal[0] = 0.1;

        for i in 1..n {
            signal[i] = 0.9 * signal[i - 1] + 0.1;
        }

        let config = ParametricConfig {
            max_ar_order: 3,
            compute_spectrum: true,
            confidence_level: Some(0.95),
            parallel: false,
            ..Default::default()
        };

        let result = enhanced_parametric_estimation(&signal, &config).unwrap();

        assert!(result.spectral_density.is_some());
        let spectrum = result.spectral_density.unwrap();
        assert_eq!(spectrum.frequencies.len(), config.n_frequencies);
        assert_eq!(spectrum.psd.len(), config.n_frequencies);
        assert!(spectrum.confidence_intervals.is_some());
    }
}
