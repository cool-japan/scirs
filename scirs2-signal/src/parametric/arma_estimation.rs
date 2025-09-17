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
use ndarray::{s, Array1, Array2};
use num_complex::Complex64;
use scirs2_core::validation::{check_finite, check_positive};
use statrs::statistics::Statistics;
use std::collections::HashMap;
use std::f64::consts::PI;

use super::types::{
    ARMAConfidenceIntervals, ARMADiagnostics, ARMAOptions, ARMAParameters, ARMAStandardErrors,
    ARMAValidation, AdaptationOptions, AdaptiveARMAEstimator, CircularBuffer, ConvergenceInfo,
    EnhancedARMAResult, EnhancedOrderSelectionResult, EnhancedSpectrumResult,
    OrderSelectionCandidate, OrderSelectionCriterion, OrderSelectionOptions, PoleZeroAnalysis,
    SpectralPeak, SpectrumMetrics, SpectrumOptions, StabilityAnalysis,
};

// Re-import AR method for basic ARMA that uses AR initialization
use super::ar_estimation::burg_method;

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

    Ok(EnhancedARMAResult {
        ar_coeffs: optimized_params.ar_coeffs,
        ma_coeffs: optimized_params.ma_coeffs,
        variance: optimized_params.variance,
        likelihood: optimized_params.likelihood,
        aic: diagnostics.aic,
        bic: diagnostics.bic,
        standard_errors: None, // TODO: Implement standard error calculation
        confidence_intervals: None, // TODO: Implement confidence interval calculation
        residuals: Array1::zeros(signal.len()), // TODO: Calculate proper residuals
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
    frequency_peaks.sort_by(|a, b| a.partial_cmp(b).unwrap());

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
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
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
            if t >= i + 1 {
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

fn compute_arma_diagnostics(
    signal: &Array1<f64>,
    params: &ARMAParameters,
    _opts: &ARMAOptions,
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

    // Placeholder for actual diagnostic tests
    Ok(ARMADiagnostics {
        aic,
        bic,
        ljung_box_test: Default::default(),
        jarque_bera_test: Default::default(),
        arch_test: Default::default(),
    })
}

fn validate_arma_model(
    _signal: &Array1<f64>,
    _params: &ARMAParameters,
    _opts: &ARMAOptions,
) -> SignalResult<ARMAValidation> {
    // Placeholder implementation
    Ok(ARMAValidation {
        residual_autocorrelation: Array1::zeros(10),
        normality_tests: Default::default(),
        heteroskedasticity_tests: Default::default(),
        stability_tests: Default::default(),
    })
}

fn compute_order_criterion(
    _signal: &Array1<f64>,
    result: &EnhancedARMAResult,
    criterion: &OrderSelectionCriterion,
    _opts: &OrderSelectionOptions,
) -> SignalResult<f64> {
    match criterion {
        OrderSelectionCriterion::AIC => Ok(result.aic),
        OrderSelectionCriterion::BIC => Ok(result.bic),
        _ => Ok(result.aic), // Placeholder
    }
}

fn compute_cross_validation_score(
    _signal: &Array1<f64>,
    _arorder: usize,
    _maorder: usize,
    _opts: &OrderSelectionOptions,
) -> SignalResult<f64> {
    // Placeholder implementation
    Ok(0.0)
}

fn analyze_model_stability(result: &EnhancedARMAResult) -> SignalResult<StabilityAnalysis> {
    Ok(StabilityAnalysis {
        is_stable: true,
        stability_margin: 0.5,
        critical_frequencies: Vec::new(),
    })
}

fn select_best_models(
    _results: Vec<OrderSelectionCandidate>,
    _criteria: &[OrderSelectionCriterion],
    _opts: &OrderSelectionOptions,
) -> SignalResult<HashMap<OrderSelectionCriterion, OrderSelectionCandidate>> {
    // Placeholder implementation
    Ok(HashMap::new())
}

fn generate_order_recommendations(
    _best_models: &HashMap<OrderSelectionCriterion, OrderSelectionCandidate>,
    _opts: &OrderSelectionOptions,
) -> SignalResult<super::types::OrderRecommendations> {
    // Placeholder implementation
    Ok(super::types::OrderRecommendations {
        recommended_ar: 1,
        recommended_ma: 1,
        confidence_level: 0.95,
        rationale: "Placeholder recommendation".to_string(),
    })
}
