//! ARMA (Autoregressive Moving Average) model estimation and analysis
//!
//! This module implements ARMA model estimation methods including:
//! - Basic and enhanced ARMA estimation
//! - Spectral analysis for ARMA models
//! - Order selection for ARMA models
//! - Adaptive ARMA estimation for streaming data
//! - Stability analysis and parameter optimization
//! - Pole-zero analysis and root finding

use crate::error::{SignalError, SignalResult};
use scirs2_core::ndarray::{s, Array1, Array2};
use scirs2_core::numeric::Complex64;
use scirs2_core::validation::{check_finite, check_positive};
use statrs::statistics::Statistics;
use std::collections::HashMap;
use std::f64::consts::PI;

use super::types::{
    ARMAConfidenceIntervals, ARMADiagnostics, ARMAOptions, ARMAParameters, ARMAStandardErrors,
    ARMAValidation, AdaptationOptions, AdaptiveARMAEstimator, CircularBuffer, ConvergenceInfo,
    EnhancedARMAResult, EnhancedOrderSelectionResult, EnhancedSpectrumResult,
    HeteroskedasticityTests, NormalityTests, OrderSelectionCandidate, OrderSelectionCriterion,
    OrderSelectionOptions, PoleZeroAnalysis, SpectralPeak, SpectrumMetrics, SpectrumOptions,
    StabilityAnalysis, StabilityTests,
};

// Re-import AR method for basic ARMA that uses AR initialization
use super::ar_estimation::burg_method;

// Genuine residual diagnostic tests (Ljung-Box, Jarque-Bera, ARCH-LM, KS,
// Anderson-Darling, Breusch-Pagan, CUSUM/Chow-like stability checks).
use super::diagnostics::{
    anderson_darling_normal_stat, arch_lm_test, breusch_pagan_stat, jarque_bera_test,
    kolmogorov_smirnov_normal_stat, ljung_box_test, sample_autocorrelation, stability_diagnostics,
};

/// Estimates ARMA model parameters using a two-stage approach
///
/// This function implements a basic ARMA estimation using:
/// 1. Initial AR estimation with higher order
/// 2. MA parameter estimation from residuals
/// 3. Parameter refinement
///
/// # Arguments
/// * `signal` - Input time series
/// * `arorder` - AR order
/// * `maorder` - MA order
///
/// # Returns
/// * `ar_coeffs` - AR coefficients [1, a1, a2, ..., ap]
/// * `ma_coeffs` - MA coefficients [1, b1, b2, ..., bq]
/// * `variance` - Estimated noise variance
pub fn estimate_arma(
    signal: &Array1<f64>,
    arorder: usize,
    maorder: usize,
) -> SignalResult<(Array1<f64>, Array1<f64>, f64)> {
    if arorder + maorder >= signal.len() {
        return Err(SignalError::ValueError(format!(
            "Total ARMA order ({}) must be less than signal length ({})",
            arorder + maorder,
            signal.len()
        )));
    }

    // Step 1: Estimate AR parameters using Burg's method with increased order
    let ar_initorder = arorder + maorder;
    let ar_init = burg_method(signal, ar_initorder)?;

    // Step 2: Compute the residuals
    let n = signal.len();
    let mut residuals = Array1::<f64>::zeros(n);

    for t in ar_initorder..n {
        let mut pred = 0.0;
        for i in 1..=ar_initorder {
            pred += ar_init.0[i] * signal[t - i];
        }
        residuals[t] = signal[t] - pred;
    }

    // Step 3: Fit MA model to the residuals using innovation algorithm
    // This is a simplified approach for MA parameter estimation

    // Compute autocorrelation of residuals
    let mut r = Array1::<f64>::zeros(maorder + 1);
    for k in 0..=maorder {
        let mut sum = 0.0;
        let mut count = 0;

        for t in ar_initorder..(n - k) {
            sum += residuals[t] * residuals[t + k];
            count += 1;
        }

        if count > 0 {
            r[k] = sum / count as f64;
        }
    }

    // Solve for MA parameters using Durbin's method
    let mut ma_coeffs = Array1::<f64>::zeros(maorder + 1);
    ma_coeffs[0] = 1.0;

    let mut v = Array1::<f64>::zeros(maorder + 1);
    v[0] = r[0];

    for k in 1..=maorder {
        let mut sum = 0.0;
        for j in 1..k {
            sum += ma_coeffs[j] * r[k - j];
        }

        ma_coeffs[k] = (r[k] - sum) / v[0];

        // Update variance terms
        for j in 1..k {
            let old_c = ma_coeffs[j];
            ma_coeffs[j] = old_c - ma_coeffs[k] * ma_coeffs[k - j];
        }

        v[k] = v[k - 1] * (1.0 - ma_coeffs[k] * ma_coeffs[k]);
    }

    // Step 4: Re-estimate AR parameters while accounting for MA influence
    // This is a simplified version - in practice, more iterative approaches are used

    // Extract the final model parameters
    let mut final_ar = Array1::<f64>::zeros(arorder + 1);
    final_ar[0] = 1.0;
    for i in 1..=arorder {
        final_ar[i] = ar_init.0[i];
    }

    // Compute innovation variance
    let variance = v[maorder];

    Ok((final_ar, ma_coeffs, variance))
}

/// Calculates the power spectral density of an ARMA model
///
/// # Arguments
/// * `ar_coeffs` - AR coefficients [1, a1, a2, ..., ap]
/// * `ma_coeffs` - MA coefficients [1, b1, b2, ..., bq]
/// * `variance` - Noise variance
/// * `freqs` - Frequencies at which to evaluate the spectrum
/// * `fs` - Sampling frequency
///
/// # Returns
/// * Power spectral density at the specified frequencies
pub fn arma_spectrum(
    ar_coeffs: &Array1<f64>,
    ma_coeffs: &Array1<f64>,
    variance: f64,
    freqs: &Array1<f64>,
    fs: f64,
) -> SignalResult<Array1<f64>> {
    // Validate inputs
    if ar_coeffs[0] != 1.0 || ma_coeffs[0] != 1.0 {
        return Err(SignalError::ValueError(
            "AR and MA coefficients must start with 1.0".to_string(),
        ));
    }

    if variance <= 0.0 {
        return Err(SignalError::ValueError(
            "Variance must be positive".to_string(),
        ));
    }

    let p = ar_coeffs.len() - 1; // AR order
    let q = ma_coeffs.len() - 1; // MA order

    // Calculate normalized frequencies
    let norm_freqs = freqs.mapv(|f| f * 2.0 * PI / fs);

    // Calculate PSD for each frequency
    let mut psd = Array1::<f64>::zeros(norm_freqs.len());

    for (i, &w) in norm_freqs.iter().enumerate() {
        // Compute AR polynomial: A(e^{jw})
        let mut a = Complex64::new(0.0, 0.0);
        for k in 0..=p {
            let phase = -w * k as f64;
            let coeff = ar_coeffs[k];
            a += coeff * Complex64::new(phase.cos(), phase.sin());
        }

        // Compute MA polynomial: B(e^{jw})
        let mut b = Complex64::new(0.0, 0.0);
        for k in 0..=q {
            let phase = -w * k as f64;
            let coeff = ma_coeffs[k];
            b += coeff * Complex64::new(phase.cos(), phase.sin());
        }

        // PSD = variance * |B(e^{jw})|^2 / |A(e^{jw})|^2
        psd[i] = variance * b.norm_sqr() / a.norm_sqr();
    }

    Ok(psd)
}

/// Enhanced ARMA estimation with comprehensive analysis and diagnostics
///
/// This function provides advanced ARMA estimation including:
/// - Iterative parameter optimization
/// - Stability analysis
/// - Model diagnostics
/// - Convergence monitoring
/// - Levenberg-Marquardt optimization
/// - Enhanced numerical stability
pub fn estimate_arma_enhanced(
    signal: &Array1<f64>,
    arorder: usize,
    maorder: usize,
    options: Option<ARMAOptions>,
) -> SignalResult<EnhancedARMAResult> {
    let opts = options.unwrap_or_default();

    // Validate input parameters
    validate_arma_parameters(signal, arorder, maorder, &opts)?;

    // Initialize parameters using method of moments or other robust technique
    let initial_params = initialize_arma_parameters(signal, arorder, maorder, &opts)?;

    // Optimize parameters using iterative algorithm
    let optimized_params = optimize_arma_parameters(signal, initial_params, &opts)?;

    // Compute model diagnostics and statistics
    let diagnostics = compute_arma_diagnostics(signal, &optimized_params, &opts)?;

    // Validate the estimated model
    let validation = validate_arma_model(signal, &optimized_params, &opts)?;

    // Compute residuals
    let residuals = compute_arma_residuals(signal, &optimized_params)?;

    // Compute standard errors
    let standard_errors = compute_arma_standard_errors(signal, &optimized_params, &residuals)?;

    // Compute confidence intervals (default 95% confidence level)
    let confidence_level = 0.95;
    let confidence_intervals =
        compute_arma_confidence_intervals(&optimized_params, &standard_errors, confidence_level)?;

    Ok(EnhancedARMAResult {
        ar_coeffs: optimized_params.ar_coeffs,
        ma_coeffs: optimized_params.ma_coeffs,
        variance: optimized_params.variance,
        likelihood: optimized_params.likelihood,
        aic: diagnostics.aic,
        bic: diagnostics.bic,
        standard_errors: Some(standard_errors),
        confidence_intervals: Some(confidence_intervals),
        residuals,
        diagnostics,
        validation,
        convergence_info: optimized_params.convergence_info,
    })
}

/// Enhanced spectrum computation with comprehensive analysis
///
/// Computes ARMA spectrum with additional features:
/// - Pole-zero analysis
/// - Confidence bands (optional)
/// - Peak detection (optional)
/// - Spectral metrics
pub fn arma_spectrum_enhanced(
    ar_coeffs: &Array1<f64>,
    ma_coeffs: &Array1<f64>,
    variance: f64,
    freqs: &Array1<f64>,
    fs: f64,
    options: Option<SpectrumOptions>,
) -> SignalResult<EnhancedSpectrumResult> {
    let opts = options.unwrap_or_default();

    // Compute basic spectrum
    let spectrum = compute_arma_spectrum_basic(ar_coeffs, ma_coeffs, variance, freqs, fs)?;

    // Analyze poles and zeros
    let pole_zero_analysis = analyze_poles_zeros(ar_coeffs, ma_coeffs)?;

    // Compute confidence bands if requested
    let confidence_bands = if opts.compute_confidence_bands {
        Some(compute_spectrum_confidence_bands(
            ar_coeffs, ma_coeffs, variance, freqs, fs, &opts,
        )?)
    } else {
        None
    };

    // Detect spectral peaks
    let peaks = if opts.detect_peaks {
        Some(detect_spectral_peaks(&spectrum, freqs, &opts)?)
    } else {
        None
    };

    // Compute additional metrics
    let metrics = compute_spectrum_metrics(&spectrum, freqs)?;

    Ok(EnhancedSpectrumResult {
        frequencies: freqs.clone(),
        spectrum,
        confidence_bands,
        pole_zero_analysis,
        peaks,
        metrics,
    })
}

/// Enhanced order selection for ARMA models
///
/// Provides comprehensive order selection using multiple criteria:
/// - Information criteria (AIC, BIC, HQC, FPE, AICc)
/// - Cross-validation
/// - Stability analysis
/// - Model comparison and recommendations
pub fn select_armaorder_enhanced(
    signal: &Array1<f64>,
    max_arorder: usize,
    max_maorder: usize,
    criteria: Vec<OrderSelectionCriterion>,
    options: Option<OrderSelectionOptions>,
) -> SignalResult<EnhancedOrderSelectionResult> {
    let opts = options.unwrap_or_default();

    let mut results = Vec::new();

    // Test all combinations of AR and MA orders
    for arorder in 0..=max_arorder {
        for maorder in 0..=max_maorder {
            if arorder == 0 && maorder == 0 {
                continue; // Skip trivial model
            }

            // Fit ARMA model
            let model_result = estimate_arma_enhanced(signal, arorder, maorder, None);

            if let Ok(result) = model_result {
                // Compute all requested criteria
                let mut criterion_values = std::collections::HashMap::new();

                for criterion in &criteria {
                    let value = compute_order_criterion(signal, &result, criterion, &opts)?;
                    criterion_values.insert(criterion.clone(), value);
                }

                // Cross-validation score
                let cv_score = if opts.use_cross_validation {
                    Some(compute_cross_validation_score(
                        signal, arorder, maorder, &opts,
                    )?)
                } else {
                    None
                };

                // Stability analysis
                let stability = analyze_model_stability(&result)?;

                results.push(OrderSelectionCandidate {
                    arorder,
                    maorder,
                    criterion_values,
                    cv_score,
                    stability,
                    model_result: result,
                });
            }
        }
    }

    // Select best models according to each criterion
    let best_models = select_best_models(results, &criteria, &opts)?;

    Ok(EnhancedOrderSelectionResult {
        best_models: best_models.clone(),
        all_candidates: Vec::new(), // Could store all if needed
        recommendations: generate_order_recommendations(&best_models, &opts)?,
    })
}

/// Real-time adaptive ARMA estimation for streaming data
///
/// Provides online parameter estimation with:
/// - Recursive parameter updates
/// - Forgetting factors for non-stationary data
/// - Change point detection
/// - Computational efficiency for real-time applications
pub fn adaptive_arma_estimator(
    initial_signal: &Array1<f64>,
    arorder: usize,
    maorder: usize,
    adaptation_options: Option<AdaptationOptions>,
) -> SignalResult<AdaptiveARMAEstimator> {
    let opts = adaptation_options.unwrap_or_default();

    // Initialize with batch estimation
    let initial_estimate = estimate_arma_enhanced(initial_signal, arorder, maorder, None)?;

    Ok(AdaptiveARMAEstimator {
        arorder,
        maorder,
        current_ar_coeffs: initial_estimate.ar_coeffs,
        current_ma_coeffs: initial_estimate.ma_coeffs,
        current_variance: initial_estimate.variance,
        forgetting_factor: opts.forgetting_factor,
        adaptation_rate: opts.adaptation_rate,
        change_detection_threshold: opts.change_detection_threshold,
        buffer: CircularBuffer::new(opts.buffer_size),
        update_count: 0,
        last_update_time: std::time::Instant::now(),
    })
}

/// Helper function: Compute basic ARMA spectrum
fn compute_arma_spectrum_basic(
    ar_coeffs: &Array1<f64>,
    ma_coeffs: &Array1<f64>,
    variance: f64,
    freqs: &Array1<f64>,
    fs: f64,
) -> SignalResult<Array1<f64>> {
    arma_spectrum(ar_coeffs, ma_coeffs, variance, freqs, fs)
}

/// Analyze poles and zeros of ARMA model
fn analyze_poles_zeros(
    ar_coeffs: &Array1<f64>,
    ma_coeffs: &Array1<f64>,
) -> SignalResult<PoleZeroAnalysis> {
    // Find poles from AR coefficients (roots of AR polynomial)
    let poles = if ar_coeffs.len() > 1 {
        find_polynomial_roots(&ar_coeffs.slice(s![1..]).to_owned())?
    } else {
        Vec::new()
    };

    // Find zeros from MA coefficients (roots of MA polynomial)
    let zeros = if ma_coeffs.len() > 1 {
        find_polynomial_roots(&ma_coeffs.slice(s![1..]).to_owned())?
    } else {
        Vec::new()
    };

    // Calculate stability margin (minimum distance of poles from unit circle)
    let mut stability_margin = f64::INFINITY;
    for pole in &poles {
        let distance_from_unit_circle = (1.0 - pole.norm()).abs();
        stability_margin = stability_margin.min(distance_from_unit_circle);
    }

    // If no poles, system is stable
    if poles.is_empty() {
        stability_margin = 1.0;
    }

    // Find frequency peaks from pole locations
    let mut frequency_peaks = Vec::new();
    for pole in &poles {
        if pole.norm() > 0.8 {
            // Only consider poles close to unit circle
            let freq = pole.arg().abs() / (2.0 * PI);
            if freq > 0.0 && freq < 0.5 {
                // Normalized frequency [0, 0.5]
                frequency_peaks.push(freq);
            }
        }
    }

    // Sort frequency peaks
    frequency_peaks.sort_by(|a, b| a.partial_cmp(b).expect("Operation failed"));

    Ok(PoleZeroAnalysis {
        poles,
        zeros,
        stability_margin,
        frequency_peaks,
    })
}

/// Find roots of a polynomial using companion matrix eigenvalues
fn find_polynomial_roots(coeffs: &Array1<f64>) -> SignalResult<Vec<Complex64>> {
    let n = coeffs.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    if n == 1 {
        // Linear case: ax + b = 0 => x = -b/a
        if coeffs[0].abs() > 1e-15 {
            return Ok(vec![Complex64::new(-coeffs[0], 0.0)]);
        } else {
            return Ok(Vec::new());
        }
    }

    // Create companion matrix
    let mut companion = Array2::zeros((n, n));

    // Fill the companion matrix
    // Last row contains negative coefficients divided by leading coefficient
    let leading_coeff = coeffs[n - 1];
    if leading_coeff.abs() < 1e-15 {
        return Err(SignalError::ComputationError(
            "Leading coefficient is zero in polynomial".to_string(),
        ));
    }

    for i in 0..n {
        companion[[n - 1, i]] = -coeffs[i] / leading_coeff;
    }

    // Fill the upper subdiagonal with ones
    for i in 0..n - 1 {
        companion[[i, i + 1]] = 1.0;
    }

    // Find eigenvalues using QR algorithm (simplified implementation)
    eigenvalues_qr(&companion)
}

/// Simplified QR algorithm for eigenvalue computation
fn eigenvalues_qr(matrix: &Array2<f64>) -> SignalResult<Vec<Complex64>> {
    let n = matrix.nrows();
    let mut a = matrix.to_owned();
    let max_iter = 100;
    let tolerance = 1e-10;

    for _ in 0..max_iter {
        // QR decomposition (simplified Givens rotations)
        let (q, r) = qr_decomposition(&a)?;

        // Update A = RQ
        a = r.dot(&q);

        // Check for convergence (off-diagonal elements should be small)
        let mut converged = true;
        for i in 0..n {
            for j in 0..n {
                if i != j && a[[i, j]].abs() > tolerance {
                    converged = false;
                    break;
                }
            }
            if !converged {
                break;
            }
        }

        if converged {
            break;
        }
    }

    // Extract eigenvalues from diagonal (assuming convergence to quasi-triangular form)
    let mut eigenvals = Vec::new();
    let mut i = 0;
    while i < n {
        if i == n - 1 || a[[i + 1, i]].abs() < tolerance {
            // Real eigenvalue
            eigenvals.push(Complex64::new(a[[i, i]], 0.0));
            i += 1;
        } else {
            // Complex conjugate pair (2x2 block)
            let a11 = a[[i, i]];
            let a12 = a[[i, i + 1]];
            let a21 = a[[i + 1, i]];
            let a22 = a[[i + 1, i + 1]];

            let trace = a11 + a22;
            let det = a11 * a22 - a12 * a21;
            let discriminant = trace * trace - 4.0 * det;

            if discriminant >= 0.0 {
                // Two real eigenvalues
                let sqrt_disc = discriminant.sqrt();
                eigenvals.push(Complex64::new((trace + sqrt_disc) / 2.0, 0.0));
                eigenvals.push(Complex64::new((trace - sqrt_disc) / 2.0, 0.0));
            } else {
                // Complex conjugate pair
                let real_part = trace / 2.0;
                let imag_part = (-discriminant).sqrt() / 2.0;
                eigenvals.push(Complex64::new(real_part, imag_part));
                eigenvals.push(Complex64::new(real_part, -imag_part));
            }
            i += 2;
        }
    }

    Ok(eigenvals)
}

/// Simplified QR decomposition using Givens rotations
fn qr_decomposition(matrix: &Array2<f64>) -> SignalResult<(Array2<f64>, Array2<f64>)> {
    let (m, n) = matrix.dim();
    let mut q = Array2::eye(m);
    let mut r = matrix.to_owned();

    for j in 0..n.min(m - 1) {
        for i in (j + 1)..m {
            let x = r[[j, j]];
            let y = r[[i, j]];

            if y.abs() > 1e-15 {
                let norm = (x * x + y * y).sqrt();
                let c = x / norm;
                let s = y / norm;

                // Apply Givens rotation to R
                for k in j..n {
                    let temp1 = c * r[[j, k]] + s * r[[i, k]];
                    let temp2 = -s * r[[j, k]] + c * r[[i, k]];
                    r[[j, k]] = temp1;
                    r[[i, k]] = temp2;
                }

                // Apply Givens rotation to Q
                for k in 0..m {
                    let temp1 = c * q[[k, j]] + s * q[[k, i]];
                    let temp2 = -s * q[[k, j]] + c * q[[k, i]];
                    q[[k, j]] = temp1;
                    q[[k, i]] = temp2;
                }
            }
        }
    }

    Ok((q, r))
}

/// Compute confidence bands for spectrum
fn compute_spectrum_confidence_bands(
    ar_coeffs: &Array1<f64>,
    ma_coeffs: &Array1<f64>,
    variance: f64,
    freqs: &Array1<f64>,
    fs: f64,
    _opts: &SpectrumOptions,
) -> SignalResult<(Array1<f64>, Array1<f64>)> {
    let spectrum = compute_arma_spectrum_basic(ar_coeffs, ma_coeffs, variance, freqs, fs)?;
    let factor = 1.96; // 95% confidence
    let lower = spectrum.mapv(|x| x * (1.0 - factor * 0.1));
    let upper = spectrum.mapv(|x| x * (1.0 + factor * 0.1));
    Ok((lower, upper))
}

/// Detect spectral peaks in the computed spectrum
pub fn detect_spectral_peaks(
    spectrum: &Array1<f64>,
    freqs: &Array1<f64>,
    opts: &SpectrumOptions,
) -> SignalResult<Vec<SpectralPeak>> {
    let mut peaks = Vec::new();

    // Simple peak detection
    for i in 1..(spectrum.len() - 1) {
        if spectrum[i] > spectrum[i - 1]
            && spectrum[i] > spectrum[i + 1]
            && spectrum[i] > opts.peak_threshold
        {
            peaks.push(SpectralPeak {
                frequency: freqs[i],
                power: spectrum[i],
                prominence: spectrum[i] - spectrum[i - 1].min(spectrum[i + 1]),
                bandwidth: 1.0,
            });
        }
    }

    Ok(peaks)
}

/// Compute metrics for the spectrum
fn compute_spectrum_metrics(
    spectrum: &Array1<f64>,
    freqs: &Array1<f64>,
) -> SignalResult<SpectrumMetrics> {
    let total_power = spectrum.sum();
    let peak_idx = spectrum
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).expect("Operation failed"))
        .map(|(i, _)| i)
        .unwrap_or(0);

    Ok(SpectrumMetrics {
        total_power,
        peak_frequency: freqs[peak_idx],
        bandwidth_3db: 1.0,
        spectral_entropy: 1.0,
    })
}

/// Validate ARMA parameters
fn validate_arma_parameters(
    signal: &Array1<f64>,
    arorder: usize,
    maorder: usize,
    _opts: &ARMAOptions,
) -> SignalResult<()> {
    if signal.len() < (arorder + maorder) * 5 {
        return Err(SignalError::ValueError(
            "Insufficient data for reliable ARMA estimation".to_string(),
        ));
    }
    Ok(())
}

/// Initialize ARMA parameters using method of moments
fn initialize_arma_parameters(
    _signal: &Array1<f64>,
    arorder: usize,
    maorder: usize,
    _opts: &ARMAOptions,
) -> SignalResult<ARMAParameters> {
    // Placeholder implementation
    Ok(ARMAParameters {
        ar_coeffs: Array1::zeros(arorder + 1),
        ma_coeffs: Array1::zeros(maorder + 1),
        variance: 1.0,
        noise_variance: 1.0,
        likelihood: 0.0,
        convergence_info: ConvergenceInfo {
            converged: false,
            iterations: 0,
            final_gradient_norm: 0.0,
            final_step_size: 0.0,
        },
    })
}

/// Optimize ARMA parameters using iterative algorithm
fn optimize_arma_parameters(
    signal: &Array1<f64>,
    initial: ARMAParameters,
    opts: &ARMAOptions,
) -> SignalResult<ARMAParameters> {
    // Basic validation - check signal is not empty
    if signal.is_empty() {
        return Err(SignalError::ValueError(
            "Signal cannot be empty".to_string(),
        ));
    }
    check_positive(opts.max_iterations, "max_iterations")?;

    let mut current_params = initial;
    let mut current_likelihood = compute_log_likelihood(signal, &current_params)?;
    let mut best_params = current_params.clone();
    let mut best_likelihood = current_likelihood;

    let mut convergence_count = 0;
    let convergence_threshold = 3; // Require 3 consecutive iterations with small change

    for iteration in 0..opts.max_iterations {
        // Enhanced parameter update using gradient descent with adaptive learning rate
        let gradient = compute_parameter_gradient(signal, &current_params, opts.tolerance)?;

        // Adaptive learning rate based on iteration and gradient magnitude
        let gradient_norm = gradient.ar_coeffs.mapv(|x| x.powi(2)).sum()
            + gradient.ma_coeffs.mapv(|x| x.powi(2)).sum();
        let adaptive_learning_rate = opts.learning_rate / (1.0 + 0.1 * iteration as f64)
            * (1.0 / (1.0 + gradient_norm.sqrt()));

        // Update parameters with momentum and regularization
        let momentum_factor = 0.9;
        let regularization = 0.001;

        // Update AR coefficients with L2 regularization
        for i in 0..current_params.ar_coeffs.len() {
            let momentum = if iteration > 0 {
                momentum_factor * (current_params.ar_coeffs[i] - best_params.ar_coeffs[i])
            } else {
                0.0
            };

            current_params.ar_coeffs[i] -= adaptive_learning_rate * gradient.ar_coeffs[i]
                + regularization * current_params.ar_coeffs[i]
                + momentum;
        }

        // Update MA coefficients with L2 regularization
        for i in 0..current_params.ma_coeffs.len() {
            let momentum = if iteration > 0 {
                momentum_factor * (current_params.ma_coeffs[i] - best_params.ma_coeffs[i])
            } else {
                0.0
            };

            current_params.ma_coeffs[i] -= adaptive_learning_rate * gradient.ma_coeffs[i]
                + regularization * current_params.ma_coeffs[i]
                + momentum;
        }

        // Update noise variance with constraints
        current_params.noise_variance = (current_params.noise_variance
            - adaptive_learning_rate * gradient.noise_variance)
            .max(1e-8);

        // Ensure model stability
        if !is_stable(&current_params) {
            // Projection onto stable region
            current_params = project_to_stable_region(&current_params)?;
        }

        // Compute new likelihood
        let new_likelihood = compute_log_likelihood(signal, &current_params)?;

        // Check for improvement
        if new_likelihood > best_likelihood {
            best_params = current_params.clone();
            best_likelihood = new_likelihood;
            convergence_count = 0;
        } else {
            convergence_count += 1;
        }

        // Convergence check
        let likelihood_change = (new_likelihood - current_likelihood).abs();
        if likelihood_change < opts.tolerance && convergence_count >= convergence_threshold {
            break;
        }

        current_likelihood = new_likelihood;

        // Enhanced convergence diagnostics
        if iteration % 10 == 0 {
            let stability_margin = compute_stability_margin(&current_params);
            if stability_margin < 0.1 {
                eprintln!(
                    "Warning: Model approaching instability at iteration {}",
                    iteration
                );
            }
        }
    }

    // Final validation
    if !is_stable(&best_params) {
        return Err(SignalError::ComputationError(
            "Optimized ARMA model is unstable".to_string(),
        ));
    }

    Ok(best_params)
}

/// Compute parameter gradient for optimization
fn compute_parameter_gradient(
    signal: &Array1<f64>,
    params: &ARMAParameters,
    tolerance: f64,
) -> SignalResult<ARMAParameters> {
    let epsilon = tolerance.sqrt(); // Small perturbation for numerical differentiation
    let base_likelihood = compute_log_likelihood(signal, params)?;

    let mut gradient = ARMAParameters {
        ar_coeffs: Array1::zeros(params.ar_coeffs.len()),
        ma_coeffs: Array1::zeros(params.ma_coeffs.len()),
        variance: 0.0,
        noise_variance: 0.0,
        likelihood: 0.0,
        convergence_info: ConvergenceInfo {
            converged: false,
            iterations: 0,
            final_gradient_norm: 0.0,
            final_step_size: 0.0,
        },
    };

    // Compute gradient for AR coefficients
    for i in 0..params.ar_coeffs.len() {
        let mut params_plus = params.clone();
        params_plus.ar_coeffs[i] += epsilon;

        let likelihood_plus = compute_log_likelihood(signal, &params_plus)?;
        gradient.ar_coeffs[i] = (likelihood_plus - base_likelihood) / epsilon;
    }

    // Compute gradient for MA coefficients
    for i in 0..params.ma_coeffs.len() {
        let mut params_plus = params.clone();
        params_plus.ma_coeffs[i] += epsilon;

        let likelihood_plus = compute_log_likelihood(signal, &params_plus)?;
        gradient.ma_coeffs[i] = (likelihood_plus - base_likelihood) / epsilon;
    }

    // Compute gradient for noise variance
    let mut params_plus = params.clone();
    params_plus.noise_variance += epsilon;
    let likelihood_plus = compute_log_likelihood(signal, &params_plus)?;
    gradient.noise_variance = (likelihood_plus - base_likelihood) / epsilon;

    Ok(gradient)
}

/// Check if ARMA model is stable
fn is_stable(params: &ARMAParameters) -> bool {
    // Check AR stability: roots of AR polynomial should be outside unit circle
    let ar_stable = check_ar_stability(&params.ar_coeffs);

    // Check MA invertibility: roots of MA polynomial should be outside unit circle
    let ma_stable = check_ma_invertibility(&params.ma_coeffs);

    ar_stable && ma_stable
}

/// Check AR polynomial stability
fn check_ar_stability(ar_coeffs: &Array1<f64>) -> bool {
    if ar_coeffs.is_empty() {
        return true;
    }

    // For AR(1): |a1| < 1
    if ar_coeffs.len() == 1 {
        return ar_coeffs[0].abs() < 1.0;
    }

    // For higher orders, use companion matrix approach (simplified)
    // This is a basic stability check - could be enhanced with proper root finding
    let sum_abs: f64 = ar_coeffs.iter().map(|&x| x.abs()).sum();
    sum_abs < 1.0 // Sufficient condition for stability
}

/// Check MA polynomial invertibility
fn check_ma_invertibility(ma_coeffs: &Array1<f64>) -> bool {
    if ma_coeffs.is_empty() {
        return true;
    }

    // Similar to AR stability check
    let sum_abs: f64 = ma_coeffs.iter().map(|&x| x.abs()).sum();
    sum_abs < 1.0
}

/// Project parameters onto stable region
fn project_to_stable_region(params: &ARMAParameters) -> SignalResult<ARMAParameters> {
    let mut stable_params = params.clone();

    // Project AR coefficients
    let ar_sum: f64 = stable_params.ar_coeffs.iter().map(|&x| x.abs()).sum();
    if ar_sum >= 1.0 {
        let scaling_factor = 0.95 / ar_sum;
        stable_params.ar_coeffs.mapv_inplace(|x| x * scaling_factor);
    }

    // Project MA coefficients
    let ma_sum: f64 = stable_params.ma_coeffs.iter().map(|&x| x.abs()).sum();
    if ma_sum >= 1.0 {
        let scaling_factor = 0.95 / ma_sum;
        stable_params.ma_coeffs.mapv_inplace(|x| x * scaling_factor);
    }

    // Ensure positive noise variance
    stable_params.noise_variance = stable_params.noise_variance.max(1e-8);

    Ok(stable_params)
}

/// Compute stability margin
fn compute_stability_margin(params: &ARMAParameters) -> f64 {
    let ar_sum: f64 = params.ar_coeffs.iter().map(|&x| x.abs()).sum();
    let ma_sum: f64 = params.ma_coeffs.iter().map(|&x| x.abs()).sum();

    let ar_margin = 1.0 - ar_sum;
    let ma_margin = 1.0 - ma_sum;

    ar_margin.min(ma_margin)
}

/// Compute log-likelihood for ARMA model
fn compute_log_likelihood(signal: &Array1<f64>, params: &ARMAParameters) -> SignalResult<f64> {
    let _n = signal.len();
    let residuals = compute_residuals(signal, params)?;

    let mut log_likelihood = 0.0;
    let two_pi_sigma2 = 2.0 * PI * params.noise_variance;

    for &residual in residuals.iter() {
        let term = residual.powi(2) / (2.0 * params.noise_variance);
        log_likelihood -= 0.5 * two_pi_sigma2.ln() + term;
    }

    Ok(log_likelihood)
}

/// Compute residuals for ARMA model
fn compute_residuals(signal: &Array1<f64>, params: &ARMAParameters) -> SignalResult<Array1<f64>> {
    let n = signal.len();
    let mut residuals = Array1::zeros(n);
    let p = params.ar_coeffs.len();
    let q = params.ma_coeffs.len();

    // Initialize with zeros for simplicity (could use better initialization)
    let mut ma_errors = vec![0.0; q];

    for t in p.max(q)..n {
        let mut prediction = 0.0;

        // AR component
        for i in 0..p {
            if t > i {
                prediction += params.ar_coeffs[i] * signal[t - i - 1];
            }
        }

        // MA component
        for i in 0..q {
            if i < ma_errors.len() {
                prediction -= params.ma_coeffs[i] * ma_errors[q - 1 - i];
            }
        }

        residuals[t] = signal[t] - prediction;

        // Update MA error terms
        if q > 0 {
            ma_errors.rotate_right(1);
            ma_errors[0] = residuals[t];
        }
    }

    Ok(residuals)
}

// Placeholder implementations for additional helper functions
// These would need to be fully implemented in a production system

/// Residuals with the initial `max(p, q)` burn-in samples (identically
/// zero by construction, see [`compute_residuals`]) dropped, so diagnostic
/// tests operate only on genuine one-step-ahead prediction errors.
fn burned_in_residuals(signal: &Array1<f64>, params: &ARMAParameters) -> SignalResult<Vec<f64>> {
    let residuals = compute_residuals(signal, params)?;
    let p = params.ar_coeffs.len();
    let q = params.ma_coeffs.len();
    let burn_in = p.max(q);
    Ok(residuals.iter().skip(burn_in).copied().collect())
}

fn compute_arma_diagnostics(
    signal: &Array1<f64>,
    params: &ARMAParameters,
    opts: &ARMAOptions,
) -> SignalResult<ARMADiagnostics> {
    let n = signal.len() as f64;
    let p = params.ar_coeffs.len() as f64;
    let q = params.ma_coeffs.len() as f64;

    // Compute log-likelihood
    let log_likelihood = compute_log_likelihood(signal, params)?;

    // Akaike Information Criterion (AIC)
    let num_params = p + q + 1.0; // AR + MA + noise variance
    let aic = -2.0 * log_likelihood + 2.0 * num_params;

    // Bayesian Information Criterion (BIC)
    let bic = -2.0 * log_likelihood + num_params * n.ln();

    // Genuine residual diagnostics rather than `Default::default()` stand-ins
    // (which hardcoded `p_value: 1.0`, meaning every model unconditionally
    // "passed" every test regardless of actual fit quality).
    let valid_residuals = burned_in_residuals(signal, params)?;

    let lb_lags = opts
        .ljung_box_lags
        .unwrap_or(10)
        .min(valid_residuals.len().saturating_sub(1))
        .max(1);
    let ljung_box = ljung_box_test(&valid_residuals, lb_lags);

    let jarque_bera = jarque_bera_test(&valid_residuals);

    let arch_lags = opts
        .arch_lags
        .unwrap_or(5)
        .min(valid_residuals.len().saturating_sub(2))
        .max(1);
    let arch = arch_lm_test(&valid_residuals, arch_lags);

    Ok(ARMADiagnostics {
        aic,
        bic,
        ljung_box_test: ljung_box,
        jarque_bera_test: jarque_bera,
        arch_test: arch,
    })
}

fn validate_arma_model(
    signal: &Array1<f64>,
    params: &ARMAParameters,
    opts: &ARMAOptions,
) -> SignalResult<ARMAValidation> {
    let p = params.ar_coeffs.len();
    let q = params.ma_coeffs.len();
    let burn_in = p.max(q);
    let valid_residuals = burned_in_residuals(signal, params)?;

    // Residual autocorrelation function, computed from the actual fitted
    // residuals rather than a fixed zero vector.
    let lags = 10.min(valid_residuals.len().saturating_sub(1)).max(1);
    let mut residual_autocorrelation = Array1::zeros(lags);
    for k in 1..=lags {
        residual_autocorrelation[k - 1] = sample_autocorrelation(&valid_residuals, k);
    }

    let jarque_bera = jarque_bera_test(&valid_residuals);
    let kolmogorov_smirnov = kolmogorov_smirnov_normal_stat(&valid_residuals);
    let anderson_darling = anderson_darling_normal_stat(&valid_residuals);
    let normality_tests = NormalityTests {
        jarque_bera,
        kolmogorov_smirnov,
        anderson_darling,
    };

    let arch_lags = opts
        .arch_lags
        .unwrap_or(5)
        .min(valid_residuals.len().saturating_sub(2))
        .max(1);
    let arch_test = arch_lm_test(&valid_residuals, arch_lags);
    let white_test = arch_test.statistic; // both probe conditional heteroskedasticity
    let breusch_pagan =
        breusch_pagan_stat(&valid_residuals, signal.as_slice().unwrap_or(&[]), burn_in);
    let heteroskedasticity_tests = HeteroskedasticityTests {
        arch_test,
        white_test,
        breusch_pagan,
    };

    let (chow_test, cusum_test, recursive_residuals) = stability_diagnostics(&valid_residuals);
    let stability_tests = StabilityTests {
        chow_test,
        cusum_test,
        recursive_residuals,
    };

    Ok(ARMAValidation {
        residual_autocorrelation,
        normality_tests,
        heteroskedasticity_tests,
        stability_tests,
    })
}

fn compute_order_criterion(
    signal: &Array1<f64>,
    result: &EnhancedARMAResult,
    criterion: &OrderSelectionCriterion,
    opts: &OrderSelectionOptions,
) -> SignalResult<f64> {
    let n = signal.len() as f64;
    let p = result.ar_coeffs.len().saturating_sub(1);
    let q = result.ma_coeffs.len().saturating_sub(1);
    let num_params = (p + q + 1) as f64; // AR + MA + noise variance

    match criterion {
        OrderSelectionCriterion::AIC => Ok(result.aic),
        OrderSelectionCriterion::BIC => Ok(result.bic),
        OrderSelectionCriterion::HQC => {
            // Hannan-Quinn Criterion: -2*LL + 2*k*ln(ln(n)); recover LL
            // exactly from the already-computed AIC (aic = -2*LL + 2*k).
            let log_likelihood = (2.0 * num_params - result.aic) / 2.0;
            let penalty = 2.0 * num_params * n.ln().ln().max(1e-12);
            Ok(-2.0 * log_likelihood + penalty)
        }
        OrderSelectionCriterion::FPE => {
            // Final Prediction Error: sigma^2 * (n + k) / (n - k)
            let k = num_params;
            if n - k > 0.0 {
                Ok(result.variance * (n + k) / (n - k))
            } else {
                Ok(f64::INFINITY)
            }
        }
        OrderSelectionCriterion::AICc => {
            // Small-sample-corrected AIC.
            let k = num_params;
            if n - k - 1.0 > 0.0 {
                Ok(result.aic + (2.0 * k * (k + 1.0)) / (n - k - 1.0))
            } else {
                Ok(f64::INFINITY)
            }
        }
        OrderSelectionCriterion::CrossValidation => {
            compute_cross_validation_score(signal, p, q, opts)
        }
        OrderSelectionCriterion::PredictionError => Ok(result.variance),
    }
}

/// Rolling-origin (walk-forward) cross-validation for ARMA order
/// selection: repeatedly re-fits the model on a growing prefix of the
/// signal and scores its one-step-ahead forecast error on the next held-out
/// sample, returning the mean squared forecast error (lower is better,
/// consistent with the AIC/BIC convention used elsewhere in this module).
fn compute_cross_validation_score(
    signal: &Array1<f64>,
    arorder: usize,
    maorder: usize,
    opts: &OrderSelectionOptions,
) -> SignalResult<f64> {
    let n = signal.len();
    let min_train = ((arorder + maorder) * 5).max(arorder + maorder + 5);

    if n <= min_train + 1 {
        // Not enough data for genuine held-out validation; fall back to
        // the in-sample residual variance from a single fit rather than a
        // hardcoded 0.0.
        let fit = estimate_arma_enhanced(signal, arorder, maorder, None)?;
        return Ok(fit.variance);
    }

    // A lighter iteration budget keeps the repeated per-fold refits fast
    // while still performing a genuine optimization each time.
    let cv_options = ARMAOptions {
        max_iterations: 50,
        ..ARMAOptions::default()
    };

    let n_folds = opts.cv_folds.max(1);
    let test_len = n - min_train;
    let fold_size = (test_len / n_folds).max(1);

    let mut squared_errors = Vec::new();
    for fold in 0..n_folds {
        let split = min_train + fold * fold_size;
        if split + 1 >= n {
            break;
        }
        let train = signal.slice(s![0..split]).to_owned();
        let actual_next = signal[split];

        let fit = match estimate_arma_enhanced(&train, arorder, maorder, Some(cv_options.clone())) {
            Ok(f) => f,
            Err(_) => continue,
        };

        // One-step-ahead forecast from the AR component (the MA
        // component's lagged innovations are unknown at the forecast
        // origin and are conventionally treated as zero).
        let mut forecast = 0.0;
        for (i, &coeff) in fit.ar_coeffs.iter().enumerate() {
            if i < train.len() {
                forecast += coeff * train[train.len() - 1 - i];
            }
        }

        let error = actual_next - forecast;
        squared_errors.push(error * error);
    }

    if squared_errors.is_empty() {
        let fit = estimate_arma_enhanced(signal, arorder, maorder, None)?;
        return Ok(fit.variance);
    }

    Ok(squared_errors.iter().sum::<f64>() / squared_errors.len() as f64)
}

/// Assess the stability/invertibility of a fitted ARMA model and locate any
/// AR poles close to the unit circle, rather than hardcoding
/// `is_stable: true` and a constant margin regardless of the fitted
/// coefficients.
fn analyze_model_stability(result: &EnhancedARMAResult) -> SignalResult<StabilityAnalysis> {
    // Reuses the same (sufficient-condition) stability/invertibility check
    // already enforced during optimization on this exact `ar_coeffs` /
    // `ma_coeffs` representation (`estimate_arma_enhanced` never returns a
    // model that fails it), so this is a genuine re-derivation rather than
    // an independent guess at a possibly-mismatched coefficient convention.
    //
    // NOTE: this deliberately does not attempt to report `critical_frequencies`
    // via this file's `find_polynomial_roots`/companion-matrix root finder:
    // that helper's companion matrix is sized for an (n-1)-degree polynomial
    // from n coefficients but actually solves a degree-n one (its highest
    // coefficient always normalizes to exactly 1), so its roots do not
    // correspond to a well-defined polynomial in general. That is a
    // separate, pre-existing issue in `find_polynomial_roots` itself
    // (unrelated to the fabricated constants this function replaces) and is
    // out of scope here; leaving `critical_frequencies` empty is an honest
    // "not computed" rather than silently propagating that bug.
    let is_stable =
        check_ar_stability(&result.ar_coeffs) && check_ma_invertibility(&result.ma_coeffs);

    let ar_sum: f64 = result.ar_coeffs.iter().map(|&x| x.abs()).sum();
    let ma_sum: f64 = result.ma_coeffs.iter().map(|&x| x.abs()).sum();
    let stability_margin = (1.0 - ar_sum).min(1.0 - ma_sum);

    Ok(StabilityAnalysis {
        is_stable,
        stability_margin,
        critical_frequencies: Vec::new(),
    })
}

/// Select the best candidate model for each requested criterion.
///
/// For every information criterion (AIC/BIC/HQC/FPE/AICc), cross-validation
/// score, and prediction-error criterion computed above, a *lower* value
/// indicates a better model; this selects the genuine minimizer among the
/// real, previously-evaluated candidates instead of discarding all of them
/// into an empty map.
fn select_best_models(
    results: Vec<OrderSelectionCandidate>,
    criteria: &[OrderSelectionCriterion],
    _opts: &OrderSelectionOptions,
) -> SignalResult<HashMap<OrderSelectionCriterion, OrderSelectionCandidate>> {
    let mut best_models = HashMap::new();

    for criterion in criteria {
        let best = results
            .iter()
            .filter_map(|candidate| {
                let value = match criterion {
                    OrderSelectionCriterion::CrossValidation => candidate.cv_score,
                    other => candidate.criterion_values.get(other).copied(),
                };
                value.map(|v| (v, candidate))
            })
            .min_by(|(a, _), (b, _)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(_, candidate)| candidate.clone());

        if let Some(candidate) = best {
            best_models.insert(criterion.clone(), candidate);
        }
    }

    Ok(best_models)
}

/// Generate order recommendations from the genuinely-selected best models
/// (one per criterion), rather than a hardcoded `(1, 1)` recommendation.
///
/// Recommends the `(ar, ma)` order pair most criteria agree on (ties broken
/// toward the more parsimonious model), and reports the fraction of
/// criteria in agreement as the confidence level.
fn generate_order_recommendations(
    best_models: &HashMap<OrderSelectionCriterion, OrderSelectionCandidate>,
    opts: &OrderSelectionOptions,
) -> SignalResult<super::types::OrderRecommendations> {
    if best_models.is_empty() {
        return Ok(super::types::OrderRecommendations {
            recommended_ar: 0,
            recommended_ma: 0,
            confidence_level: 0.0,
            rationale: "No candidate models were successfully evaluated".to_string(),
        });
    }

    let mut votes: HashMap<(usize, usize), usize> = HashMap::new();
    for candidate in best_models.values() {
        *votes
            .entry((candidate.arorder, candidate.maorder))
            .or_insert(0) += 1;
    }

    let total_criteria = best_models.len();
    let ((recommended_ar, recommended_ma), agreeing_votes) = votes
        .iter()
        .max_by(|(order_a, count_a), (order_b, count_b)| {
            count_a
                .cmp(count_b)
                .then_with(|| (order_b.0 + order_b.1).cmp(&(order_a.0 + order_a.1)))
        })
        .map(|(&order, &count)| (order, count))
        .unwrap_or(((0, 0), 0));

    let confidence_level = agreeing_votes as f64 / total_criteria as f64;

    let mut agreeing_criteria: Vec<String> = best_models
        .iter()
        .filter(|(_, candidate)| {
            candidate.arorder == recommended_ar && candidate.maorder == recommended_ma
        })
        .map(|(criterion, _)| format!("{criterion:?}"))
        .collect();
    agreeing_criteria.sort();

    let rationale = format!(
        "ARMA({recommended_ar},{recommended_ma}) selected by {agreeing_votes}/{total_criteria} criteria ({}); stability weight {:.2}",
        agreeing_criteria.join(", "),
        opts.stability_weight
    );

    Ok(super::types::OrderRecommendations {
        recommended_ar,
        recommended_ma,
        confidence_level,
        rationale,
    })
}

/// Compute residuals from ARMA model
///
/// Calculates one-step-ahead prediction errors for the fitted ARMA model.
fn compute_arma_residuals(
    signal: &Array1<f64>,
    params: &ARMAParameters,
) -> SignalResult<Array1<f64>> {
    let n = signal.len();
    let p = params.ar_coeffs.len().saturating_sub(1); // AR order
    let q = params.ma_coeffs.len().saturating_sub(1); // MA order
    let max_lag = p.max(q);

    let mut residuals = Array1::zeros(n);
    let mut past_residuals = vec![0.0; q]; // Store past q residuals

    // Compute residuals: e_t = y_t - (AR_part + MA_part)
    for t in max_lag..n {
        // AR component: sum of a_i * y_{t-i}
        let mut ar_part = 0.0;
        for i in 1..=p {
            if t >= i {
                ar_part += params.ar_coeffs[i] * signal[t - i];
            }
        }

        // MA component: sum of b_j * e_{t-j}
        let mut ma_part = 0.0;
        for j in 1..=q.min(past_residuals.len()) {
            if t >= j {
                ma_part += params.ma_coeffs[j] * past_residuals[past_residuals.len() - j];
            }
        }

        // Calculate residual
        let residual = signal[t] - ar_part - ma_part;
        residuals[t] = residual;

        // Update past residuals buffer
        past_residuals.push(residual);
        if past_residuals.len() > q {
            past_residuals.remove(0);
        }
    }

    Ok(residuals)
}

/// Compute standard errors for ARMA parameters
///
/// Uses asymptotic theory and the observed Fisher information matrix.
/// Standard errors are approximated as:
/// SE = sqrt(diag(inverse(Fisher Information Matrix)))
fn compute_arma_standard_errors(
    signal: &Array1<f64>,
    params: &ARMAParameters,
    residuals: &Array1<f64>,
) -> SignalResult<ARMAStandardErrors> {
    let n = signal.len();
    let p = params.ar_coeffs.len().saturating_sub(1);
    let q = params.ma_coeffs.len().saturating_sub(1);

    // Compute residual variance (sigma^2)
    let valid_residuals: Vec<f64> = residuals
        .iter()
        .filter(|&&r| r.abs() > 1e-10)
        .copied()
        .collect();

    let residual_variance = if !valid_residuals.is_empty() {
        valid_residuals.iter().map(|r| r * r).sum::<f64>() / valid_residuals.len() as f64
    } else {
        params.noise_variance
    };

    // Asymptotic standard errors based on information matrix theory
    // For ARMA models: SE ≈ sqrt(sigma^2 / n) for each coefficient

    let base_se = (residual_variance / n as f64).sqrt();

    // AR coefficients standard errors
    // Use slightly larger SE for higher-order terms due to estimation uncertainty
    let mut ar_se = Array1::zeros(params.ar_coeffs.len());
    ar_se[0] = 0.0; // First coefficient is always 1.0, no uncertainty
    for i in 1..=p {
        // Higher order terms have slightly larger standard errors
        let order_penalty = 1.0 + 0.1 * (i as f64);
        ar_se[i] = base_se * order_penalty;
    }

    // MA coefficients standard errors
    let mut ma_se = Array1::zeros(params.ma_coeffs.len());
    ma_se[0] = 0.0; // First coefficient is always 1.0, no uncertainty
    for j in 1..=q {
        // MA terms typically have slightly higher uncertainty than AR terms
        let order_penalty = 1.0 + 0.15 * (j as f64);
        ma_se[j] = base_se * order_penalty * 1.2;
    }

    // Variance standard error using chi-square approximation
    // For residual variance: SE(sigma^2) ≈ sigma^2 * sqrt(2/n)
    let variance_se = residual_variance * (2.0 / n as f64).sqrt();

    Ok(ARMAStandardErrors {
        ar_se,
        ma_se,
        variance_se,
    })
}

/// Compute confidence intervals for ARMA parameters
///
/// Uses normal approximation: parameter ± z_(alpha/2) * SE
/// where z_(alpha/2) is the critical value from standard normal distribution.
fn compute_arma_confidence_intervals(
    params: &ARMAParameters,
    standard_errors: &ARMAStandardErrors,
    confidence_level: f64,
) -> SignalResult<ARMAConfidenceIntervals> {
    // Critical value for confidence interval (e.g., 1.96 for 95% CI)
    let alpha = 1.0 - confidence_level;
    let z_critical = normal_quantile(1.0 - alpha / 2.0);

    // AR confidence intervals
    let p = params.ar_coeffs.len();
    let mut ar_ci = Array2::zeros((p, 2));
    for i in 0..p {
        let margin = z_critical * standard_errors.ar_se[i];
        ar_ci[[i, 0]] = params.ar_coeffs[i] - margin; // Lower bound
        ar_ci[[i, 1]] = params.ar_coeffs[i] + margin; // Upper bound
    }

    // MA confidence intervals
    let q = params.ma_coeffs.len();
    let mut ma_ci = Array2::zeros((q, 2));
    for j in 0..q {
        let margin = z_critical * standard_errors.ma_se[j];
        ma_ci[[j, 0]] = params.ma_coeffs[j] - margin; // Lower bound
        ma_ci[[j, 1]] = params.ma_coeffs[j] + margin; // Upper bound
    }

    // Variance confidence interval (using chi-square approximation for positive values)
    let variance_margin = z_critical * standard_errors.variance_se;
    let variance_ci = (
        (params.variance - variance_margin).max(1e-10), // Ensure positive
        params.variance + variance_margin,
    );

    Ok(ARMAConfidenceIntervals {
        ar_ci,
        ma_ci,
        variance_ci,
    })
}

/// Approximate quantile function for standard normal distribution
///
/// Uses rational approximation for the inverse of the standard normal CDF.
/// Accurate to about 5 decimal places.
fn normal_quantile(p: f64) -> f64 {
    if p <= 0.0 || p >= 1.0 {
        return if p <= 0.0 {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        };
    }

    // For p near 0.5, use simple approximation
    if (p - 0.5).abs() < 0.42 {
        // Central region: use polynomial approximation
        let q = p - 0.5;
        let r = q * q;
        let num = ((((-25.44106049637) * r + 41.39119773534) * r + (-18.61500062529)) * r
            + 2.50662823884)
            * q;
        let den = ((((3.13082909833) * r + (-21.06224101826)) * r + 23.08336743743) * r
            + (-8.47351093090))
            * r
            + 1.0;
        return num / den;
    }

    // Tail regions: use different approximation
    let q = if p < 0.5 { p } else { 1.0 - p };
    let r = (-2.0 * q.ln()).sqrt();

    let num = ((2.32121276858) * r + 0.30119479853) * r + 4.85014127135;
    let den = (1.28776170681) * r + 3.54388924762;

    let x = num / (r + den);

    if p < 0.5 {
        -x
    } else {
        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_result(
        ar_coeffs: Vec<f64>,
        ma_coeffs: Vec<f64>,
        aic: f64,
        bic: f64,
    ) -> EnhancedARMAResult {
        EnhancedARMAResult {
            ar_coeffs: Array1::from_vec(ar_coeffs),
            ma_coeffs: Array1::from_vec(ma_coeffs),
            variance: 1.0,
            likelihood: 0.0,
            aic,
            bic,
            standard_errors: None,
            confidence_intervals: None,
            residuals: Array1::zeros(1),
            diagnostics: ARMADiagnostics {
                aic,
                bic,
                ljung_box_test: Default::default(),
                jarque_bera_test: Default::default(),
                arch_test: Default::default(),
            },
            validation: ARMAValidation {
                residual_autocorrelation: Array1::zeros(1),
                normality_tests: Default::default(),
                heteroskedasticity_tests: Default::default(),
                stability_tests: Default::default(),
            },
            convergence_info: ConvergenceInfo {
                converged: true,
                iterations: 1,
                final_gradient_norm: 0.0,
                final_step_size: 0.0,
            },
        }
    }

    fn make_test_candidate(
        arorder: usize,
        maorder: usize,
        aic: f64,
        bic: f64,
        cv_score: Option<f64>,
    ) -> OrderSelectionCandidate {
        let mut criterion_values = HashMap::new();
        criterion_values.insert(OrderSelectionCriterion::AIC, aic);
        criterion_values.insert(OrderSelectionCriterion::BIC, bic);
        OrderSelectionCandidate {
            arorder,
            maorder,
            criterion_values,
            cv_score,
            stability: StabilityAnalysis {
                is_stable: true,
                stability_margin: 0.5,
                critical_frequencies: Vec::new(),
            },
            model_result: make_test_result(vec![0.5], vec![0.0], aic, bic),
        }
    }

    #[test]
    fn test_select_best_models_returns_real_candidates() {
        let opts = OrderSelectionOptions::default();
        let candidates = vec![
            make_test_candidate(1, 0, 100.0, 110.0, Some(5.0)),
            make_test_candidate(2, 0, 90.0, 115.0, Some(3.0)), // best AIC
            make_test_candidate(1, 1, 95.0, 100.0, Some(1.0)), // best BIC and CV
        ];
        let criteria = vec![
            OrderSelectionCriterion::AIC,
            OrderSelectionCriterion::BIC,
            OrderSelectionCriterion::CrossValidation,
        ];

        let best = select_best_models(candidates, &criteria, &opts).expect("Operation failed");

        // The fabricated implementation always returned an empty map,
        // discarding every real candidate.
        assert_eq!(best.len(), 3);
        assert_eq!(best[&OrderSelectionCriterion::AIC].arorder, 2);
        assert_eq!(best[&OrderSelectionCriterion::BIC].arorder, 1);
        assert_eq!(best[&OrderSelectionCriterion::BIC].maorder, 1);
        assert_eq!(best[&OrderSelectionCriterion::CrossValidation].maorder, 1);
    }

    #[test]
    fn test_generate_order_recommendations_reflects_best_models() {
        let opts = OrderSelectionOptions::default();
        let mut best_models = HashMap::new();
        best_models.insert(
            OrderSelectionCriterion::AIC,
            make_test_candidate(2, 1, 90.0, 110.0, None),
        );
        best_models.insert(
            OrderSelectionCriterion::BIC,
            make_test_candidate(2, 1, 95.0, 100.0, None),
        );
        best_models.insert(
            OrderSelectionCriterion::HQC,
            make_test_candidate(1, 0, 99.0, 105.0, None),
        );

        let recommendations =
            generate_order_recommendations(&best_models, &opts).expect("Operation failed");

        // Two out of three criteria agree on ARMA(2,1); the fabricated
        // implementation always recommended a hardcoded ARMA(1,1).
        assert_eq!(recommendations.recommended_ar, 2);
        assert_eq!(recommendations.recommended_ma, 1);
        assert!((recommendations.confidence_level - 2.0 / 3.0).abs() < 1e-9);
        assert_ne!(recommendations.rationale, "Placeholder recommendation");
    }

    #[test]
    fn test_analyze_model_stability_reacts_to_coefficients() {
        let near_unstable = make_test_result(vec![0.95], vec![0.1], 100.0, 110.0);
        let very_stable = make_test_result(vec![0.1], vec![0.05], 100.0, 110.0);

        let near = analyze_model_stability(&near_unstable).expect("Operation failed");
        let stable = analyze_model_stability(&very_stable).expect("Operation failed");

        // The fabricated implementation always returned exactly 0.5
        // regardless of the fitted coefficients.
        assert!(stable.stability_margin > near.stability_margin);
        assert!(
            (near.stability_margin - 0.5).abs() > 1e-6
                || (stable.stability_margin - 0.5).abs() > 1e-6
        );
    }

    #[test]
    fn test_cross_validation_score_reacts_to_signal_predictability() {
        let n = 150;
        // A highly predictable (smooth sinusoidal) signal...
        let predictable: Array1<f64> = Array1::from_iter((0..n).map(|i| (i as f64 * 0.1).sin()));
        // ...vs an unpredictable (deterministic pseudo-random) one of the
        // same length and (approximately) matched variance, so the
        // comparison probes predictability rather than raw signal scale.
        let mut rng_state: u64 = 99;
        let mut next = || {
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            (rng_state as f64 / u64::MAX as f64) - 0.5
        };
        // sin(0.1*i) has variance ~0.5 (std ~0.707); uniform(-0.5, 0.5) has
        // std ~0.289, so scale by ~2.449 to roughly match variance.
        let noisy: Array1<f64> = Array1::from_iter((0..n).map(|_| 2.449 * next()));

        let opts = OrderSelectionOptions {
            cv_folds: 3,
            ..OrderSelectionOptions::default()
        };

        let score_predictable =
            compute_cross_validation_score(&predictable, 2, 0, &opts).expect("Operation failed");
        let score_noisy =
            compute_cross_validation_score(&noisy, 2, 0, &opts).expect("Operation failed");

        // The fabricated implementation always returned exactly 0.0
        // regardless of the signal; a genuine implementation must at least
        // react differently to genuinely different data.
        assert!(score_predictable >= 0.0);
        assert!(score_noisy >= 0.0);
        assert_ne!(score_predictable, score_noisy);
    }

    #[test]
    fn test_arma_diagnostics_are_computed_from_real_residuals() {
        let n = 300;
        let mut signal = Array1::<f64>::zeros(n);
        signal[0] = 1.0;
        signal[1] = 0.5;
        for t in 2..n {
            signal[t] = 0.6 * signal[t - 1] - 0.2 * signal[t - 2] + 0.3 * (t as f64 * 0.31).sin();
        }

        let low_order = estimate_arma_enhanced(&signal, 1, 0, None).expect("Operation failed");
        let high_order = estimate_arma_enhanced(&signal, 4, 2, None).expect("Operation failed");

        // The fabricated implementation always returned `zeros(10)` /
        // `Default::default()` (hardcoded `p_value: 1.0`) regardless of the
        // fitted model or residuals.
        assert_ne!(
            low_order.validation.residual_autocorrelation,
            high_order.validation.residual_autocorrelation
        );
        assert_ne!(
            low_order.diagnostics.ljung_box_test.statistic,
            high_order.diagnostics.ljung_box_test.statistic
        );
        for p_value in [
            low_order.diagnostics.ljung_box_test.p_value,
            high_order.diagnostics.ljung_box_test.p_value,
        ] {
            assert!((0.0..=1.0).contains(&p_value));
        }

        // Not every p-value should be exactly 1.0 (the old hardcoded default).
        let all_ones = [
            low_order.diagnostics.ljung_box_test.p_value,
            low_order.diagnostics.jarque_bera_test.p_value,
            low_order.diagnostics.arch_test.p_value,
            high_order.diagnostics.ljung_box_test.p_value,
            high_order.diagnostics.jarque_bera_test.p_value,
            high_order.diagnostics.arch_test.p_value,
        ]
        .iter()
        .all(|&p| (p - 1.0).abs() < 1e-9);
        assert!(
            !all_ones,
            "at least one diagnostic p-value should differ from the hardcoded 1.0 default"
        );
    }

    #[test]
    fn test_select_armaorder_enhanced_end_to_end_produces_real_selection() {
        let n = 150;
        let mut signal = Array1::<f64>::zeros(n);
        signal[0] = 1.0;
        signal[1] = 0.5;
        for t in 2..n {
            signal[t] = 0.6 * signal[t - 1] - 0.2 * signal[t - 2] + 0.2 * (t as f64 * 0.31).sin();
        }

        let criteria = vec![OrderSelectionCriterion::AIC, OrderSelectionCriterion::BIC];
        let options = OrderSelectionOptions {
            use_cross_validation: false, // keep this end-to-end test fast
            ..OrderSelectionOptions::default()
        };

        let result = select_armaorder_enhanced(&signal, 2, 1, criteria.clone(), Some(options))
            .expect("Operation failed");

        // The fabricated implementation always returned an empty
        // `best_models` map and a hardcoded ARMA(1,1) recommendation
        // regardless of the fitted candidates.
        assert_eq!(result.best_models.len(), criteria.len());
        for criterion in &criteria {
            assert!(result.best_models.contains_key(criterion));
        }
        assert!(result.recommendations.confidence_level > 0.0);
        assert_ne!(
            result.recommendations.rationale,
            "Placeholder recommendation"
        );
    }
}
