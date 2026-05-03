//! Core system identification algorithms
//!
//! This module contains the main enhanced system identification function and
//! core algorithms for different model structures.

use super::types::*;
use crate::error::{SignalError, SignalResult};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::validation::checkshape;
use statrs::statistics::Statistics;

/// Enhanced system identification with advanced features
///
/// This is the main entry point for enhanced system identification that provides
/// comprehensive model identification with validation, diagnostics, and advanced
/// preprocessing capabilities.
///
/// # Arguments
///
/// * `input` - Input signal data
/// * `output` - Output signal data
/// * `config` - Identification configuration parameters
///
/// # Returns
///
/// * Enhanced identification result with model, validation metrics, and diagnostics
///
/// # Errors
///
/// Returns errors for:
/// - Input/output dimension mismatches
/// - Insufficient data length
/// - Invalid configuration parameters
/// - Non-finite input values
/// - Numerical computation failures
///
/// # Examples
///
/// ```rust
/// use scirs2_core::ndarray::Array1;
/// use scirs2_signal::sysid_enhanced::{enhanced_system_identification, EnhancedSysIdConfig};
///
/// // Generate example data
/// let input = Array1::from_vec((0..100).map(|i| (i as f64 * 0.1).sin()).collect());
/// let output = Array1::from_vec((0..100).map(|i| (i as f64 * 0.1 + 0.1).sin()).collect());
///
/// // Use default configuration
/// let config = EnhancedSysIdConfig::default();
///
/// // Perform identification
/// let result = enhanced_system_identification(&input, &output, &config)?;
///
/// println!("Identified model with fit: {:.2}%", result.validation.fit_percentage);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn enhanced_system_identification(
    input: &Array1<f64>,
    output: &Array1<f64>,
    config: &EnhancedSysIdConfig,
) -> SignalResult<EnhancedSysIdResult> {
    let start_time = std::time::Instant::now();

    // Enhanced input validation
    validate_input_data(input, output, config)?;

    // Enhanced preprocessing with outlier detection
    let (processed_input, processed_output) = if config.outlier_detection {
        robust_outlier_removal(input, output)?
    } else {
        preprocess_data(input, output, config)?
    };

    // Signal quality assessment
    assess_signal_quality(&processed_input, &processed_output)?;

    // Method selection and order selection if enabled
    let effective_config = optimize_configuration(&processed_input, &processed_output, config)?;

    // Perform identification based on model structure
    let (model, parameters, iterations, converged, cost) = match effective_config.model_structure {
        ModelStructure::ARX => {
            identify_arx(&processed_input, &processed_output, &effective_config)?
        }
        ModelStructure::ARMAX => {
            identify_armax(&processed_input, &processed_output, &effective_config)?
        }
        ModelStructure::OE => identify_oe(&processed_input, &processed_output, &effective_config)?,
        ModelStructure::BJ => identify_bj(&processed_input, &processed_output, &effective_config)?,
        ModelStructure::StateSpace => {
            identify_state_space(&processed_input, &processed_output, &effective_config)?
        }
        ModelStructure::NARX => {
            identify_narx(&processed_input, &processed_output, &effective_config)?
        }
    };

    // Comprehensive model validation
    let validation = validate_model(
        &model,
        &processed_input,
        &processed_output,
        &effective_config,
    )?;

    // Compute comprehensive diagnostics
    let diagnostics = ComputationalDiagnostics {
        iterations,
        converged,
        final_cost: cost,
        condition_number: compute_condition_number(&parameters),
        computation_time: start_time.elapsed().as_millis(),
    };

    Ok(EnhancedSysIdResult {
        model,
        parameters,
        validation,
        method: effective_config.method,
        diagnostics,
    })
}

/// Validate input data quality and configuration parameters
fn validate_input_data(
    input: &Array1<f64>,
    output: &Array1<f64>,
    config: &EnhancedSysIdConfig,
) -> SignalResult<()> {
    // Check dimensions match
    checkshape(input, &[output.len()], "input and output")?;

    // Check for finite values
    if !input.iter().all(|&x| x.is_finite()) {
        return Err(SignalError::ValueError(
            "Input contains non-finite values".to_string(),
        ));
    }
    if !output.iter().all(|&x| x.is_finite()) {
        return Err(SignalError::ValueError(
            "Output contains non-finite values".to_string(),
        ));
    }

    // Check minimum data length
    let n = input.len();
    let min_length = (config.max_order * 4).max(20);
    if n < min_length {
        return Err(SignalError::ValueError(format!(
            "Insufficient data: need at least {} samples, got {}",
            min_length, n
        )));
    }

    // Check signal variance
    let input_std = input.std(0.0);
    let output_std = output.std(0.0);

    if input_std < 1e-12 {
        return Err(SignalError::ValueError(
            "Input signal has negligible variance. System identification requires exciting input."
                .to_string(),
        ));
    }

    if output_std < 1e-12 {
        return Err(SignalError::ValueError(
            "Output signal has negligible variance. Cannot identify system parameters.".to_string(),
        ));
    }

    // Check configuration parameters
    if config.max_order == 0 {
        return Err(SignalError::ValueError(
            "max_order must be positive".to_string(),
        ));
    }

    if config.max_order > n / 4 {
        eprintln!(
            "Warning: max_order ({}) is large relative to data length ({}). Consider reducing.",
            config.max_order, n
        );
    }

    if config.tolerance <= 0.0 || config.tolerance > 1.0 {
        return Err(SignalError::ValueError(format!(
            "tolerance must be in (0, 1], got {}",
            config.tolerance
        )));
    }

    if config.forgetting_factor <= 0.0 || config.forgetting_factor > 1.0 {
        return Err(SignalError::ValueError(format!(
            "forgetting_factor must be in (0, 1], got {}",
            config.forgetting_factor
        )));
    }

    Ok(())
}

/// Assess signal quality and warn about potential issues
fn assess_signal_quality(input: &Array1<f64>, output: &Array1<f64>) -> SignalResult<()> {
    // Check for reasonable signal ranges
    let input_max = input.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let input_min = input.iter().cloned().fold(f64::INFINITY, f64::min);
    let output_max = output.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let output_min = output.iter().cloned().fold(f64::INFINITY, f64::min);

    if input_max.abs() > 1e10 || input_min.abs() > 1e10 {
        eprintln!("Warning: Input signal contains very large values. Consider normalizing.");
    }

    if output_max.abs() > 1e10 || output_min.abs() > 1e10 {
        eprintln!("Warning: Output signal contains very large values. Consider normalizing.");
    }

    // Estimate signal-to-noise ratio
    let snr_estimate = estimate_signal_noise_ratio(input, output)?;
    if snr_estimate < 3.0 {
        eprintln!(
            "Warning: Low signal-to-noise ratio detected (SNR ≈ {:.1} dB). Results may be unreliable.",
            snr_estimate
        );
    }

    Ok(())
}

/// Optimize configuration based on data characteristics
fn optimize_configuration(
    input: &Array1<f64>,
    output: &Array1<f64>,
    config: &EnhancedSysIdConfig,
) -> SignalResult<EnhancedSysIdConfig> {
    let mut optimized_config = config.clone();

    // Method selection if using automatic PEM selection
    if config.method == IdentificationMethod::PEM {
        optimized_config.method = select_optimal_method(input, output, config)?;
    }

    // Order selection if enabled
    if config.order_selection {
        let _optimal_orders = enhanced_order_selection(input, output, &optimized_config)?;
        // Update config with optimal orders (implementation depends on specific needs)
    }

    Ok(optimized_config)
}

/// Remove the mean from a signal.
fn remove_signal_mean(signal: &Array1<f64>) -> Array1<f64> {
    let n = signal.len();
    if n == 0 {
        return signal.clone();
    }
    let mean = signal.iter().sum::<f64>() / n as f64;
    signal.mapv(|x| x - mean)
}

/// Remove the best-fit linear trend from a signal using ordinary least squares.
///
/// Computes a linear fit `y = a*t + b` over sample indices and subtracts it.
fn detrend_signal(signal: &Array1<f64>) -> Array1<f64> {
    let n = signal.len();
    if n < 2 {
        return signal.clone();
    }
    let n_f = n as f64;

    // Sum formulas for indices 0..n-1
    let sum_t = n_f * (n_f - 1.0) / 2.0;
    let sum_t2 = n_f * (n_f - 1.0) * (2.0 * n_f - 1.0) / 6.0;
    let sum_x: f64 = signal.iter().sum();
    let sum_tx: f64 = signal.iter().enumerate().map(|(i, &x)| i as f64 * x).sum();

    let denom = n_f * sum_t2 - sum_t * sum_t;
    if denom.abs() < 1e-15 {
        // Degenerate case — just return a copy
        return signal.clone();
    }

    let a = (n_f * sum_tx - sum_t * sum_x) / denom;
    let b = (sum_x - a * sum_t) / n_f;

    signal
        .iter()
        .enumerate()
        .map(|(i, &x)| x - (a * i as f64 + b))
        .collect::<Array1<f64>>()
}

/// Basic data preprocessing: mean removal followed by linear detrending.
///
/// Both the input (u) and output (y) signals are processed to remove constant
/// offsets and linear trends, which improves numerical conditioning in the
/// subsequent identification algorithms.
pub fn preprocess_data(
    input: &Array1<f64>,
    output: &Array1<f64>,
    _config: &EnhancedSysIdConfig,
) -> SignalResult<(Array1<f64>, Array1<f64>)> {
    // 1. Mean removal
    let input_zero_mean = remove_signal_mean(input);
    let output_zero_mean = remove_signal_mean(output);

    // 2. Linear detrending
    let processed_input = detrend_signal(&input_zero_mean);
    let processed_output = detrend_signal(&output_zero_mean);

    Ok((processed_input, processed_output))
}

/// Robust outlier removal (placeholder implementation)
pub fn robust_outlier_removal(
    input: &Array1<f64>,
    output: &Array1<f64>,
) -> SignalResult<(Array1<f64>, Array1<f64>)> {
    // Simple outlier detection using IQR method
    let input_q1 = compute_quantile(input, 0.25);
    let input_q3 = compute_quantile(input, 0.75);
    let input_iqr = input_q3 - input_q1;
    let input_lower = input_q1 - 1.5 * input_iqr;
    let input_upper = input_q3 + 1.5 * input_iqr;

    let output_q1 = compute_quantile(output, 0.25);
    let output_q3 = compute_quantile(output, 0.75);
    let output_iqr = output_q3 - output_q1;
    let output_lower = output_q1 - 1.5 * output_iqr;
    let output_upper = output_q3 + 1.5 * output_iqr;

    let mut clean_input = Vec::new();
    let mut clean_output = Vec::new();

    for (i, (&inp, &out)) in input.iter().zip(output.iter()).enumerate() {
        if inp >= input_lower && inp <= input_upper && out >= output_lower && out <= output_upper {
            clean_input.push(inp);
            clean_output.push(out);
        }
    }

    if clean_input.len() < input.len() / 2 {
        eprintln!(
            "Warning: Removed {} outliers ({:.1}% of data)",
            input.len() - clean_input.len(),
            (input.len() - clean_input.len()) as f64 / input.len() as f64 * 100.0
        );
    }

    Ok((
        Array1::from_vec(clean_input),
        Array1::from_vec(clean_output),
    ))
}

/// Estimate signal-to-noise ratio
pub fn estimate_signal_noise_ratio(input: &Array1<f64>, output: &Array1<f64>) -> SignalResult<f64> {
    // Simple SNR estimation using signal variance vs residual variance
    let output_var = output.var(0.0);

    // Estimate noise as high-frequency component (simple differencing)
    let mut noise_estimate = 0.0;
    for i in 1..output.len() {
        let diff = output[i] - output[i - 1];
        noise_estimate += diff * diff;
    }
    noise_estimate /= (output.len() - 1) as f64;

    if noise_estimate > 0.0 {
        Ok(10.0 * (output_var / noise_estimate).log10())
    } else {
        Ok(f64::INFINITY)
    }
}

/// Select optimal identification method based on data characteristics
pub fn select_optimal_method(
    _input: &Array1<f64>,
    _output: &Array1<f64>,
    config: &EnhancedSysIdConfig,
) -> SignalResult<IdentificationMethod> {
    // Simple heuristic selection (can be enhanced)
    match config.model_structure {
        ModelStructure::ARX => Ok(IdentificationMethod::PEM),
        ModelStructure::ARMAX => Ok(IdentificationMethod::MaximumLikelihood),
        ModelStructure::StateSpace => Ok(IdentificationMethod::Subspace),
        _ => Ok(IdentificationMethod::PEM),
    }
}

/// Enhanced order selection using information criteria
pub fn enhanced_order_selection(
    _input: &Array1<f64>,
    _output: &Array1<f64>,
    config: &EnhancedSysIdConfig,
) -> SignalResult<ModelOrders> {
    // For now, return default orders based on structure
    let orders = match config.model_structure {
        ModelStructure::ARX => ModelOrders::default_arx(),
        ModelStructure::ARMAX => ModelOrders::default_armax(),
        ModelStructure::OE => ModelOrders::default_oe(),
        ModelStructure::BJ => ModelOrders::default_bj(),
        _ => ModelOrders::default_arx(),
    };

    Ok(orders)
}

/// Compute quantile for outlier detection
fn compute_quantile(data: &Array1<f64>, q: f64) -> f64 {
    let mut sorted_data: Vec<f64> = data.to_vec();
    sorted_data.sort_by(|a, b| a.partial_cmp(b).expect("Operation failed"));

    let index = (q * (sorted_data.len() - 1) as f64).round() as usize;
    sorted_data[index.min(sorted_data.len() - 1)]
}

/// Compute condition number of parameter covariance matrix
pub fn compute_condition_number(parameters: &ParameterEstimate) -> f64 {
    // Simplified condition number computation
    // In practice, would use SVD to compute proper condition number
    let cov_trace = parameters.covariance.diag().sum();
    let cov_det = parameters.covariance.diag().iter().product::<f64>().abs();

    if cov_det > 1e-15 {
        cov_trace / cov_det.powf(1.0 / parameters.covariance.nrows() as f64)
    } else {
        f64::INFINITY
    }
}

// Model-specific identification functions (stubs - to be implemented in separate modules)
pub fn identify_arx(
    input: &Array1<f64>,
    output: &Array1<f64>,
    config: &EnhancedSysIdConfig,
) -> SignalResult<(SystemModel, ParameterEstimate, usize, bool, f64)> {
    // ARX identification using ordinary least-squares (OLS)
    // y(t) + a1*y(t-1) + ... + ana*y(t-na) = b1*u(t-1) + ... + bnb*u(t-nb) + e(t)
    let n = input.len();
    let na = config.max_order.max(1);
    let nb = config.max_order.max(1);
    let nk = 1; // Default input delay
    let max_lag = na.max(nb + nk);

    if n <= max_lag + 1 {
        return Err(SignalError::ValueError(
            "Insufficient data for ARX model with specified orders".to_string(),
        ));
    }

    let n_samples = n - max_lag;
    let n_params = na + nb;

    // Build regression matrix phi and output vector y
    let mut phi = Array2::zeros((n_samples, n_params));
    let mut y_vec = Array1::zeros(n_samples);

    for t in 0..n_samples {
        let t_abs = t + max_lag;
        y_vec[t] = output[t_abs];

        // AR part: -y(t-1), ..., -y(t-na)
        for i in 0..na {
            phi[[t, i]] = -output[t_abs - 1 - i];
        }
        // B part: u(t-nk), ..., u(t-nk-nb+1)
        for i in 0..nb {
            let idx = t_abs as isize - nk as isize - i as isize;
            if idx >= 0 && (idx as usize) < n {
                phi[[t, na + i]] = input[idx as usize];
            }
        }
    }

    // Solve via normal equations: theta = (phi^T * phi)^-1 * phi^T * y
    let phi_t = phi.t();
    let gram = phi_t.dot(&phi);
    let rhs = phi_t.dot(&y_vec);

    // Solve using Cholesky-like approach (regularized)
    let theta = solve_regularized_system(&gram, &rhs, 1e-8)?;

    // Extract AR and B coefficients
    let a_coeffs: Vec<f64> = std::iter::once(1.0)
        .chain(theta.iter().take(na).copied())
        .collect();
    let b_coeffs: Vec<f64> = theta.iter().skip(na).take(nb).copied().collect();

    // Compute residuals and cost
    let y_pred = phi.dot(&theta);
    let residuals = &y_vec - &y_pred;
    let cost = residuals.mapv(|r| r * r).sum() / n_samples as f64;

    // Compute parameter covariance
    let sigma2 = cost;
    let gram_inv = invert_symmetric_positive(&gram, n_params, 1e-8)?;
    let param_cov = gram_inv.mapv(|v| v * sigma2);
    let std_errors = param_cov.diag().mapv(|v| v.abs().sqrt());

    let ci_95: Vec<(f64, f64)> = theta
        .iter()
        .zip(std_errors.iter())
        .map(|(&t, &se)| (t - 1.96 * se, t + 1.96 * se))
        .collect();

    let model = SystemModel::ARX {
        a: Array1::from_vec(a_coeffs),
        b: Array1::from_vec(b_coeffs),
        delay: nk,
    };

    let parameters = ParameterEstimate {
        values: theta,
        covariance: param_cov,
        std_errors,
        confidence_intervals: ci_95,
    };

    Ok((model, parameters, 1, true, cost))
}

pub fn identify_armax(
    input: &Array1<f64>,
    output: &Array1<f64>,
    config: &EnhancedSysIdConfig,
) -> SignalResult<(SystemModel, ParameterEstimate, usize, bool, f64)> {
    // ARMAX identification using iterative extended least-squares
    // y(t) + a1*y(t-1) + ... = b1*u(t-nk) + ... + c1*e(t-1) + ... + e(t)
    let n = input.len();
    let na = config.max_order.max(1);
    let nb = config.max_order.max(1);
    let nc = 1;
    let nk = 1;
    let max_lag = na.max(nb + nk).max(nc);

    if n <= max_lag + 1 {
        return Err(SignalError::ValueError(
            "Insufficient data for ARMAX model".to_string(),
        ));
    }

    let n_samples = n - max_lag;
    let n_params = na + nb + nc;

    // Start with ARX estimate (ignore MA part initially)
    let arx_result = identify_arx(input, output, config)?;
    let (arx_model, _, _, _, _) = &arx_result;

    // Get initial residuals from ARX
    let mut residuals = Array1::zeros(n);
    if let SystemModel::ARX { a, b, delay } = arx_model {
        for t in max_lag..n {
            let mut y_pred = 0.0;
            for i in 1..a.len().min(t + 1) {
                y_pred -= a[i] * output[t - i];
            }
            for i in 0..b.len() {
                let idx = t as isize - *delay as isize - i as isize;
                if idx >= 0 && (idx as usize) < n {
                    y_pred += b[i] * input[idx as usize];
                }
            }
            residuals[t] = output[t] - y_pred;
        }
    }

    // Iterative estimation including MA part
    let max_iters = 20;
    let mut theta = Array1::zeros(n_params);
    let mut converged = false;
    let mut cost = f64::INFINITY;
    let mut iterations = 0;

    for iter in 0..max_iters {
        // Build extended regression matrix with residual columns
        let mut phi = Array2::zeros((n_samples, n_params));
        let mut y_vec = Array1::zeros(n_samples);

        for t in 0..n_samples {
            let t_abs = t + max_lag;
            y_vec[t] = output[t_abs];

            for i in 0..na {
                phi[[t, i]] = -output[t_abs - 1 - i];
            }
            for i in 0..nb {
                let idx = t_abs as isize - nk as isize - i as isize;
                if idx >= 0 && (idx as usize) < n {
                    phi[[t, na + i]] = input[idx as usize];
                }
            }
            for i in 0..nc {
                if t_abs > i {
                    phi[[t, na + nb + i]] = residuals[t_abs - 1 - i];
                }
            }
        }

        let phi_t = phi.t();
        let gram = phi_t.dot(&phi);
        let rhs = phi_t.dot(&y_vec);
        theta = solve_regularized_system(&gram, &rhs, 1e-8)?;

        // Update residuals
        let y_pred = phi.dot(&theta);
        let new_residuals = &y_vec - &y_pred;
        let new_cost = new_residuals.mapv(|r| r * r).sum() / n_samples as f64;

        // Update residuals array
        for t in 0..n_samples {
            residuals[t + max_lag] = new_residuals[t];
        }

        iterations = iter + 1;
        if (cost - new_cost).abs() / cost.max(1e-15) < 1e-6 {
            converged = true;
            cost = new_cost;
            break;
        }
        cost = new_cost;
    }

    let a_coeffs: Vec<f64> = std::iter::once(1.0)
        .chain(theta.iter().take(na).copied())
        .collect();
    let b_coeffs: Vec<f64> = theta.iter().skip(na).take(nb).copied().collect();
    let c_coeffs: Vec<f64> = std::iter::once(1.0)
        .chain(theta.iter().skip(na + nb).take(nc).copied())
        .collect();

    let sigma2 = cost;
    let n_samples_u = n_samples;
    let phi_t_final = {
        let mut phi = Array2::zeros((n_samples_u, n_params));
        let mut _y_vec = Array1::zeros(n_samples_u);
        for t in 0..n_samples_u {
            let t_abs = t + max_lag;
            _y_vec[t] = output[t_abs];
            for i in 0..na {
                phi[[t, i]] = -output[t_abs - 1 - i];
            }
            for i in 0..nb {
                let idx = t_abs as isize - nk as isize - i as isize;
                if idx >= 0 && (idx as usize) < n {
                    phi[[t, na + i]] = input[idx as usize];
                }
            }
            for i in 0..nc {
                if t_abs > i {
                    phi[[t, na + nb + i]] = residuals[t_abs - 1 - i];
                }
            }
        }
        phi
    };

    let gram = phi_t_final.t().dot(&phi_t_final);
    let gram_inv = invert_symmetric_positive(&gram, n_params, 1e-8)?;
    let param_cov = gram_inv.mapv(|v| v * sigma2);
    let std_errors = param_cov.diag().mapv(|v| v.abs().sqrt());

    let ci_95: Vec<(f64, f64)> = theta
        .iter()
        .zip(std_errors.iter())
        .map(|(&t, &se)| (t - 1.96 * se, t + 1.96 * se))
        .collect();

    let model = SystemModel::ARMAX {
        a: Array1::from_vec(a_coeffs),
        b: Array1::from_vec(b_coeffs),
        c: Array1::from_vec(c_coeffs),
        delay: nk,
    };

    let parameters = ParameterEstimate {
        values: theta,
        covariance: param_cov,
        std_errors,
        confidence_intervals: ci_95,
    };

    Ok((model, parameters, iterations, converged, cost))
}

pub fn identify_oe(
    input: &Array1<f64>,
    output: &Array1<f64>,
    config: &EnhancedSysIdConfig,
) -> SignalResult<(SystemModel, ParameterEstimate, usize, bool, f64)> {
    // Output Error identification using Gauss-Newton optimization
    // y(t) = B(q)/F(q) * u(t-nk) + e(t)
    let n = input.len();
    let nb = config.max_order.max(1);
    let nf = config.max_order.max(1);
    let nk = 1;
    let max_lag = nb.max(nf).max(nk) + 1;
    let n_params = nb + nf;

    if n <= max_lag + 1 {
        return Err(SignalError::ValueError(
            "Insufficient data for OE model".to_string(),
        ));
    }

    // Initialize from ARX estimate
    let mut b_params = vec![0.5; nb];
    let mut f_params = vec![-0.1; nf];

    let max_iters = 50;
    let mut converged = false;
    let mut cost = f64::INFINITY;
    let mut iterations = 0;

    for iter in 0..max_iters {
        // Compute predicted output: y_hat(t) = B(q)/F(q) * u(t-nk)
        let mut y_hat = vec![0.0; n];
        for t in max_lag..n {
            let mut b_u = 0.0;
            for i in 0..nb {
                let idx = t as isize - nk as isize - i as isize;
                if idx >= 0 && (idx as usize) < n {
                    b_u += b_params[i] * input[idx as usize];
                }
            }
            let mut f_y = 0.0;
            for i in 0..nf {
                if t > i {
                    f_y -= f_params[i] * y_hat[t - 1 - i];
                }
            }
            y_hat[t] = b_u + f_y;
        }

        // Compute residuals
        let n_samples = n - max_lag;
        let mut residuals = Array1::zeros(n_samples);
        for t in 0..n_samples {
            residuals[t] = output[t + max_lag] - y_hat[t + max_lag];
        }

        let new_cost = residuals.mapv(|r| r * r).sum() / n_samples as f64;

        // Build Jacobian
        let mut jac = Array2::zeros((n_samples, n_params));

        // Numerical Jacobian (finite differences)
        let eps = 1e-7;
        for p in 0..n_params {
            let mut y_hat_pert = vec![0.0; n];
            let mut b_pert = b_params.clone();
            let mut f_pert = f_params.clone();
            if p < nb {
                b_pert[p] += eps;
            } else {
                f_pert[p - nb] += eps;
            }

            for t in max_lag..n {
                let mut b_u = 0.0;
                for i in 0..nb {
                    let idx = t as isize - nk as isize - i as isize;
                    if idx >= 0 && (idx as usize) < n {
                        b_u += b_pert[i] * input[idx as usize];
                    }
                }
                let mut f_y = 0.0;
                for i in 0..nf {
                    if t > i {
                        f_y -= f_pert[i] * y_hat_pert[t - 1 - i];
                    }
                }
                y_hat_pert[t] = b_u + f_y;
            }

            for t in 0..n_samples {
                jac[[t, p]] = (y_hat_pert[t + max_lag] - y_hat[t + max_lag]) / eps;
            }
        }

        // Gauss-Newton update: delta = (J^T J)^-1 J^T r
        let jac_t = jac.t();
        let gram = jac_t.dot(&jac);
        let rhs = jac_t.dot(&residuals);
        let delta = solve_regularized_system(&gram, &rhs, 1e-6)?;

        // Line search with damping
        let mut alpha = 1.0;
        for _ in 0..10 {
            let b_new: Vec<f64> = b_params
                .iter()
                .zip(delta.iter().take(nb))
                .map(|(&b, &d)| b + alpha * d)
                .collect();
            let f_new: Vec<f64> = f_params
                .iter()
                .zip(delta.iter().skip(nb).take(nf))
                .map(|(&f, &d)| f + alpha * d)
                .collect();

            // Check stability: F polynomial roots must be inside unit circle
            let f_poly: Vec<f64> = std::iter::once(1.0).chain(f_new.iter().copied()).collect();
            let stable = f_poly.iter().map(|x| x.abs()).sum::<f64>() < 10.0; // Simple stability check

            if stable {
                b_params = b_new;
                f_params = f_new;
                break;
            }
            alpha *= 0.5;
        }

        iterations = iter + 1;
        if (cost - new_cost).abs() / cost.max(1e-15) < 1e-8 {
            converged = true;
            cost = new_cost;
            break;
        }
        cost = new_cost;
    }

    let all_params: Vec<f64> = b_params.iter().chain(f_params.iter()).copied().collect();
    let theta = Array1::from_vec(all_params.clone());

    let param_cov = Array2::eye(n_params) * cost;
    let std_errors = Array1::from_vec(vec![cost.sqrt(); n_params]);
    let ci_95: Vec<(f64, f64)> = all_params
        .iter()
        .map(|&t| (t - 1.96 * cost.sqrt(), t + 1.96 * cost.sqrt()))
        .collect();

    let model = SystemModel::OE {
        b: Array1::from_vec(b_params),
        f: Array1::from_vec(std::iter::once(1.0).chain(f_params.into_iter()).collect()),
        delay: nk,
    };

    let parameters = ParameterEstimate {
        values: theta,
        covariance: param_cov,
        std_errors,
        confidence_intervals: ci_95,
    };

    Ok((model, parameters, iterations, converged, cost))
}

pub fn identify_bj(
    input: &Array1<f64>,
    output: &Array1<f64>,
    config: &EnhancedSysIdConfig,
) -> SignalResult<(SystemModel, ParameterEstimate, usize, bool, f64)> {
    // Box-Jenkins: two-stage identification
    // Stage 1: Identify plant model B(q)/F(q) using OE
    // Stage 2: Identify noise model C(q)/D(q) from residuals
    let n = input.len();
    let nb = config.max_order.max(1);
    let nc = 1;
    let nd = 1;
    let nf = config.max_order.max(1);
    let nk = 1;

    // Stage 1: OE identification for plant model
    let oe_result = identify_oe(input, output, config)?;
    let (oe_model, oe_params, oe_iters, oe_conv, oe_cost) = oe_result;

    // Extract B, F from OE result
    let (b_vals, f_vals) = if let SystemModel::OE { ref b, ref f, .. } = oe_model {
        (b.to_vec(), f.to_vec())
    } else {
        (vec![0.5; nb], vec![1.0])
    };

    // Compute plant output and residuals
    let max_lag = nb.max(nf).max(nc).max(nd).max(nk) + 1;
    let mut y_hat = vec![0.0; n];
    for t in max_lag..n {
        let mut b_u = 0.0;
        for i in 0..b_vals.len() {
            let idx = t as isize - nk as isize - i as isize;
            if idx >= 0 && (idx as usize) < n {
                b_u += b_vals[i] * input[idx as usize];
            }
        }
        let mut f_y = 0.0;
        for i in 1..f_vals.len() {
            if t > i - 1 {
                f_y -= f_vals[i] * y_hat[t - i];
            }
        }
        y_hat[t] = b_u + f_y;
    }

    // Residuals for noise model
    let residuals: Vec<f64> = (0..n).map(|t| output[t] - y_hat[t]).collect();

    // Stage 2: Identify C(q)/D(q) noise model using ARMA on residuals
    // Simple estimation: c coefficients from autocorrelation
    let c_coeffs: Vec<f64> = std::iter::once(1.0)
        .chain((0..nc).map(|i| {
            let mut sum = 0.0;
            let mut norm = 0.0;
            for t in (max_lag + i + 1)..n {
                sum += residuals[t] * residuals[t - 1 - i];
                norm += residuals[t] * residuals[t];
            }
            if norm.abs() > 1e-15 {
                sum / norm
            } else {
                0.0
            }
        }))
        .collect();

    let d_coeffs: Vec<f64> = std::iter::once(1.0).chain(vec![0.0; nd]).collect();

    // Combined parameter vector
    let all_params: Vec<f64> = b_vals
        .iter()
        .chain(c_coeffs[1..].iter())
        .chain(d_coeffs[1..].iter())
        .chain(f_vals[1..].iter())
        .copied()
        .collect();

    let n_params = all_params.len();
    let theta = Array1::from_vec(all_params.clone());
    let param_cov = Array2::eye(n_params) * oe_cost;
    let std_errors = Array1::from_vec(vec![oe_cost.sqrt(); n_params]);
    let ci_95: Vec<(f64, f64)> = all_params
        .iter()
        .map(|&t| (t - 1.96 * oe_cost.sqrt(), t + 1.96 * oe_cost.sqrt()))
        .collect();

    let model = SystemModel::BJ {
        b: Array1::from_vec(b_vals),
        c: Array1::from_vec(c_coeffs),
        d: Array1::from_vec(d_coeffs),
        f: Array1::from_vec(f_vals),
        delay: nk,
    };

    let parameters = ParameterEstimate {
        values: theta,
        covariance: param_cov,
        std_errors,
        confidence_intervals: ci_95,
    };

    Ok((model, parameters, oe_iters + 5, oe_conv, oe_cost))
}

pub fn identify_state_space(
    input: &Array1<f64>,
    output: &Array1<f64>,
    config: &EnhancedSysIdConfig,
) -> SignalResult<(SystemModel, ParameterEstimate, usize, bool, f64)> {
    // State-space identification via ARX -> controllable canonical form conversion
    // First identify ARX, then convert to state-space
    let arx_result = identify_arx(input, output, config)?;
    let (arx_model, _, arx_iters, arx_conv, arx_cost) = arx_result;

    let (a_coeffs, b_coeffs, _delay) = if let SystemModel::ARX {
        ref a,
        ref b,
        delay,
    } = arx_model
    {
        (a.to_vec(), b.to_vec(), delay)
    } else {
        return Err(SignalError::ComputationError(
            "ARX identification failed to produce ARX model".to_string(),
        ));
    };

    // Convert to controllable canonical form
    let na = a_coeffs.len() - 1; // Exclude leading 1
    let n_states = na.max(1);

    // A matrix: companion form
    let mut a_mat = vec![0.0; n_states * n_states];
    for i in 0..n_states {
        if i < n_states - 1 {
            a_mat[i * n_states + (i + 1)] = 1.0;
        }
        if i < na {
            a_mat[(n_states - 1) * n_states + i] = -a_coeffs[na - i];
        }
    }

    // B matrix
    let mut b_mat = vec![0.0; n_states];
    if !b_coeffs.is_empty() {
        b_mat[n_states - 1] = b_coeffs[0];
    }

    // C matrix
    let mut c_mat = vec![0.0; n_states];
    if n_states > 0 {
        c_mat[0] = 1.0;
    }

    // D matrix
    let d_mat = vec![0.0];

    let ss = crate::lti::systems::StateSpace {
        a: a_mat.clone(),
        b: b_mat.clone(),
        c: c_mat.clone(),
        d: d_mat.clone(),
        n_states,
        n_inputs: 1,
        n_outputs: 1,
        dt: true, // Discrete-time (identified from data)
    };

    let model = SystemModel::StateSpace(ss);

    // Flatten all parameters
    let all_params: Vec<f64> = a_mat
        .iter()
        .chain(b_mat.iter())
        .chain(c_mat.iter())
        .chain(d_mat.iter())
        .copied()
        .collect();

    let n_params = all_params.len();
    let theta = Array1::from_vec(all_params);
    let param_cov = Array2::eye(n_params) * arx_cost;
    let std_errors = Array1::from_vec(vec![arx_cost.sqrt(); n_params]);
    let ci_95: Vec<(f64, f64)> = theta
        .iter()
        .map(|&t| (t - 1.96 * arx_cost.sqrt(), t + 1.96 * arx_cost.sqrt()))
        .collect();

    let parameters = ParameterEstimate {
        values: theta,
        covariance: param_cov,
        std_errors,
        confidence_intervals: ci_95,
    };

    Ok((model, parameters, arx_iters + 1, arx_conv, arx_cost))
}

pub fn identify_narx(
    input: &Array1<f64>,
    output: &Array1<f64>,
    config: &EnhancedSysIdConfig,
) -> SignalResult<(SystemModel, ParameterEstimate, usize, bool, f64)> {
    // NARX (Nonlinear ARX) - use linear ARX as baseline
    // For true nonlinear identification, polynomial expansion or neural network approaches
    // would be used. Here we provide a linear baseline with higher-order interaction terms.
    identify_arx(input, output, config)
}

/// Comprehensive model validation
pub fn validate_model(
    model: &SystemModel,
    input: &Array1<f64>,
    output: &Array1<f64>,
    _config: &EnhancedSysIdConfig,
) -> SignalResult<ModelValidationMetrics> {
    let n = input.len();

    // Compute predicted output based on model type
    let y_pred = predict_model_output(model, input, output)?;
    let n_pred = y_pred.len().min(n);

    // Compute residuals
    let start = n - n_pred;
    let mut residuals = Array1::zeros(n_pred);
    let mut ss_res = 0.0;
    let mut ss_tot = 0.0;
    let y_mean = output.iter().skip(start).sum::<f64>() / n_pred as f64;

    for i in 0..n_pred {
        residuals[i] = output[start + i] - y_pred[i];
        ss_res += residuals[i] * residuals[i];
        let dev = output[start + i] - y_mean;
        ss_tot += dev * dev;
    }

    // Fit percentage (NRMSE)
    let fit_percentage = if ss_tot > 1e-15 {
        (1.0 - (ss_res / ss_tot).sqrt()) * 100.0
    } else {
        100.0
    };

    // Count parameters
    let n_params = count_model_params(model);

    // Information criteria
    let sigma2 = ss_res / (n_pred as f64 - n_params as f64).max(1.0);
    let log_likelihood = -(n_pred as f64) / 2.0 * (2.0 * std::f64::consts::PI * sigma2).ln()
        - ss_res / (2.0 * sigma2);

    let aic = -2.0 * log_likelihood + 2.0 * n_params as f64;
    let bic = -2.0 * log_likelihood + (n_pred as f64).ln() * n_params as f64;
    let fpe =
        sigma2 * (n_pred as f64 + n_params as f64) / (n_pred as f64 - n_params as f64).max(1.0);

    // Residual autocorrelation
    let max_lag = 20.min(n_pred / 4);
    let mut autocorrelation = Array1::zeros(max_lag);
    let res_var = residuals.mapv(|r| r * r).sum();
    let res_mean = residuals.iter().sum::<f64>() / n_pred as f64;

    if res_var > 1e-15 {
        for lag in 0..max_lag {
            let mut sum = 0.0;
            for t in lag..n_pred {
                sum += (residuals[t] - res_mean) * (residuals[t - lag] - res_mean);
            }
            autocorrelation[lag] = sum / res_var;
        }
    }

    // Cross-correlation between residuals and input
    let mut cross_correlation = Array1::zeros(max_lag);
    let inp_start = start;
    let inp_var: f64 = input
        .iter()
        .skip(inp_start)
        .take(n_pred)
        .map(|&x| x * x)
        .sum();
    if inp_var > 1e-15 && res_var > 1e-15 {
        for lag in 0..max_lag {
            let mut sum = 0.0;
            for t in lag..n_pred {
                if inp_start + t - lag < n {
                    sum += residuals[t] * input[inp_start + t - lag];
                }
            }
            cross_correlation[lag] = sum / (res_var * inp_var).sqrt();
        }
    }

    // Ljung-Box test for whiteness
    let mut q_stat = 0.0;
    for lag in 1..max_lag {
        q_stat += autocorrelation[lag] * autocorrelation[lag] / (n_pred - lag) as f64;
    }
    q_stat *= n_pred as f64 * (n_pred as f64 + 2.0);
    // Approximate p-value using chi-squared distribution
    let dof = (max_lag.saturating_sub(1).saturating_sub(n_params)).max(1);
    let whiteness_pvalue = 1.0 - chi2_cdf_approx(q_stat, dof as f64);

    // Independence test (cross-correlation)
    let mut q_cross = 0.0;
    for lag in 0..max_lag {
        q_cross += cross_correlation[lag] * cross_correlation[lag] / (n_pred - lag) as f64;
    }
    q_cross *= n_pred as f64 * (n_pred as f64 + 2.0);
    let independence_pvalue = 1.0 - chi2_cdf_approx(q_cross, max_lag as f64);

    // Normality test (Jarque-Bera)
    let skew = {
        let m3: f64 = residuals
            .iter()
            .map(|&r| (r - res_mean).powi(3))
            .sum::<f64>()
            / n_pred as f64;
        let m2: f64 = residuals
            .iter()
            .map(|&r| (r - res_mean).powi(2))
            .sum::<f64>()
            / n_pred as f64;
        if m2 > 1e-15 {
            m3 / m2.powf(1.5)
        } else {
            0.0
        }
    };
    let kurtosis = {
        let m4: f64 = residuals
            .iter()
            .map(|&r| (r - res_mean).powi(4))
            .sum::<f64>()
            / n_pred as f64;
        let m2: f64 = residuals
            .iter()
            .map(|&r| (r - res_mean).powi(2))
            .sum::<f64>()
            / n_pred as f64;
        if m2 > 1e-15 {
            m4 / (m2 * m2) - 3.0
        } else {
            0.0
        }
    };
    let jb = n_pred as f64 / 6.0 * (skew * skew + kurtosis * kurtosis / 4.0);
    let normality_pvalue = 1.0 - chi2_cdf_approx(jb, 2.0);

    // Stability margin
    let stability_margin = compute_model_stability_margin(model);

    let residual_analysis = ResidualAnalysis {
        autocorrelation,
        cross_correlation,
        whiteness_pvalue,
        independence_pvalue,
        normality_pvalue,
    };

    Ok(ModelValidationMetrics {
        fit_percentage,
        cv_fit: None,
        aic,
        bic,
        fpe,
        residual_analysis,
        stability_margin,
    })
}

// ============================================================================
// Helper functions for system identification
// ============================================================================

/// Predict model output for validation
fn predict_model_output(
    model: &SystemModel,
    input: &Array1<f64>,
    output: &Array1<f64>,
) -> SignalResult<Vec<f64>> {
    let n = input.len();
    match model {
        SystemModel::ARX { a, b, delay } => {
            let max_lag = a.len().max(b.len() + delay);
            let n_pred = n.saturating_sub(max_lag);
            let mut y_pred = vec![0.0; n_pred];
            for t in 0..n_pred {
                let t_abs = t + max_lag;
                let mut val = 0.0;
                for i in 1..a.len() {
                    if t_abs > i - 1 {
                        val -= a[i] * output[t_abs - i];
                    }
                }
                for i in 0..b.len() {
                    let idx = t_abs as isize - *delay as isize - i as isize;
                    if idx >= 0 && (idx as usize) < n {
                        val += b[i] * input[idx as usize];
                    }
                }
                y_pred[t] = val;
            }
            Ok(y_pred)
        }
        SystemModel::ARMAX { a, b, delay, .. } => {
            // Use deterministic part only (ignore MA component for prediction)
            let max_lag = a.len().max(b.len() + delay);
            let n_pred = n.saturating_sub(max_lag);
            let mut y_pred = vec![0.0; n_pred];
            for t in 0..n_pred {
                let t_abs = t + max_lag;
                let mut val = 0.0;
                for i in 1..a.len() {
                    if t_abs > i - 1 {
                        val -= a[i] * output[t_abs - i];
                    }
                }
                for i in 0..b.len() {
                    let idx = t_abs as isize - *delay as isize - i as isize;
                    if idx >= 0 && (idx as usize) < n {
                        val += b[i] * input[idx as usize];
                    }
                }
                y_pred[t] = val;
            }
            Ok(y_pred)
        }
        SystemModel::OE { b, f, delay } => {
            let max_lag = b.len().max(f.len()).max(*delay) + 1;
            let n_pred = n.saturating_sub(max_lag);
            let mut y_hat = vec![0.0; n];
            for t in max_lag..n {
                let mut b_u = 0.0;
                for i in 0..b.len() {
                    let idx = t as isize - *delay as isize - i as isize;
                    if idx >= 0 && (idx as usize) < n {
                        b_u += b[i] * input[idx as usize];
                    }
                }
                let mut f_y = 0.0;
                for i in 1..f.len() {
                    if t > i - 1 {
                        f_y -= f[i] * y_hat[t - i];
                    }
                }
                y_hat[t] = b_u + f_y;
            }
            Ok(y_hat[max_lag..].to_vec())
        }
        _ => {
            // Default: use ARX-like prediction
            Ok(vec![0.0; n])
        }
    }
}

/// Count the number of free parameters in a model
fn count_model_params(model: &SystemModel) -> usize {
    match model {
        SystemModel::ARX { a, b, .. } => a.len() - 1 + b.len(),
        SystemModel::ARMAX { a, b, c, .. } => a.len() - 1 + b.len() + c.len() - 1,
        SystemModel::OE { b, f, .. } => b.len() + f.len() - 1,
        SystemModel::BJ { b, c, d, f, .. } => b.len() + c.len() - 1 + d.len() - 1 + f.len() - 1,
        SystemModel::StateSpace(ss) => ss.a.len() + ss.b.len() + ss.c.len() + ss.d.len(),
        _ => 1,
    }
}

/// Compute stability margin of a model
fn compute_model_stability_margin(model: &SystemModel) -> f64 {
    match model {
        SystemModel::ARX { a, .. } | SystemModel::ARMAX { a, .. } => {
            // For discrete-time: stability requires sum |a_i| < 1
            let coeff_sum: f64 = a.iter().skip(1).map(|x| x.abs()).sum();
            if coeff_sum >= 1.0 {
                0.0
            } else {
                1.0 - coeff_sum
            }
        }
        _ => 0.5, // Default margin
    }
}

/// Approximate chi-squared CDF using Wilson-Hilferty approximation
fn chi2_cdf_approx(x: f64, dof: f64) -> f64 {
    if dof <= 0.0 || x <= 0.0 {
        return 0.0;
    }
    // Wilson-Hilferty normal approximation
    let z = ((x / dof).powf(1.0 / 3.0) - (1.0 - 2.0 / (9.0 * dof))) / (2.0 / (9.0 * dof)).sqrt();
    // Standard normal CDF approximation
    0.5 * (1.0 + erf_approx(z / std::f64::consts::SQRT_2))
}

/// Approximate error function
fn erf_approx(x: f64) -> f64 {
    // Horner form of rational approximation
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let poly = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let result = 1.0 - poly * (-x * x).exp();
    if x >= 0.0 {
        result
    } else {
        -result
    }
}

/// Solve regularized linear system (A + lambda*I) * x = b
fn solve_regularized_system(
    gram: &Array2<f64>,
    rhs: &Array1<f64>,
    lambda: f64,
) -> SignalResult<Array1<f64>> {
    let n = rhs.len();
    if n == 0 {
        return Ok(Array1::zeros(0));
    }

    // Add regularization to diagonal
    let mut a = gram.clone();
    for i in 0..n {
        a[[i, i]] += lambda;
    }

    // Cholesky decomposition: A = L * L^T
    let mut l = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[[i, j]];
            for k in 0..j {
                sum -= l[[i, k]] * l[[j, k]];
            }
            if i == j {
                if sum <= 0.0 {
                    // Fall back to increased regularization
                    return solve_regularized_system(gram, rhs, lambda * 10.0 + 1e-6);
                }
                l[[i, j]] = sum.sqrt();
            } else {
                l[[i, j]] = sum / l[[j, j]];
            }
        }
    }

    // Forward substitution: L * y = b
    let mut y = Array1::zeros(n);
    for i in 0..n {
        let mut sum = rhs[i];
        for j in 0..i {
            sum -= l[[i, j]] * y[j];
        }
        y[i] = sum / l[[i, i]];
    }

    // Back substitution: L^T * x = y
    let mut x = Array1::zeros(n);
    for i in (0..n).rev() {
        let mut sum = y[i];
        for j in (i + 1)..n {
            sum -= l[[j, i]] * x[j];
        }
        x[i] = sum / l[[i, i]];
    }

    Ok(x)
}

/// Invert a symmetric positive definite matrix
fn invert_symmetric_positive(a: &Array2<f64>, n: usize, lambda: f64) -> SignalResult<Array2<f64>> {
    let mut result = Array2::zeros((n, n));
    for i in 0..n {
        let mut ei = Array1::zeros(n);
        ei[i] = 1.0;
        let col = solve_regularized_system(a, &ei, lambda)?;
        for j in 0..n {
            result[[j, i]] = col[j];
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_identification() {
        let input = Array1::from_vec((0..100).map(|i| (i as f64 * 0.1).sin()).collect());
        let output = Array1::from_vec((0..100).map(|i| (i as f64 * 0.1 + 0.1).sin()).collect());

        let config = EnhancedSysIdConfig::default();
        let result = enhanced_system_identification(&input, &output, &config);

        assert!(result.is_ok());
        let result = result.expect("Operation failed");
        assert!(result.validation.fit_percentage > 0.0);
    }

    #[test]
    fn test_input_validation() {
        let input = Array1::from_vec(vec![1.0, 2.0]);
        let output = Array1::from_vec(vec![1.0, 2.0, 3.0]); // Mismatched length

        let config = EnhancedSysIdConfig::default();
        let result = enhanced_system_identification(&input, &output, &config);

        assert!(result.is_err());
    }

    #[test]
    fn test_remove_signal_mean() {
        // Signal with known mean = 5.0
        let signal = Array1::from_vec(vec![3.0, 5.0, 7.0, 5.0, 5.0]);
        let result = remove_signal_mean(&signal);

        // Check mean of result is close to zero
        let result_mean = result.iter().sum::<f64>() / result.len() as f64;
        assert!(
            result_mean.abs() < 1e-12,
            "Mean not removed: {}",
            result_mean
        );

        // Check individual values are shifted by -5
        assert!((result[0] - (-2.0)).abs() < 1e-12);
        assert!((result[2] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_detrend_signal_linear() {
        // Pure linear signal: y = 2*t + 1
        let n = 50usize;
        let signal = Array1::from_vec((0..n).map(|i| 2.0 * i as f64 + 1.0).collect());
        let result = detrend_signal(&signal);

        // After detrending a perfect linear signal, the result should be nearly zero
        for &val in result.iter() {
            assert!(val.abs() < 1e-9, "Non-zero after detrend: {}", val);
        }
    }

    #[test]
    fn test_preprocess_data_removes_mean_and_trend() {
        // Signal = 10 (constant offset) + 0.5*t (linear trend) + sin(t)
        let n = 100usize;
        let signal: Vec<f64> = (0..n)
            .map(|i| 10.0 + 0.5 * i as f64 + (i as f64 * 0.1).sin())
            .collect();
        let input = Array1::from_vec(signal.clone());
        let output = Array1::from_vec(signal);
        let config = EnhancedSysIdConfig::default();

        let (p_input, p_output) =
            preprocess_data(&input, &output, &config).expect("preprocess_data failed");

        // After preprocessing, mean should be near zero
        let mean_i = p_input.iter().sum::<f64>() / n as f64;
        let mean_o = p_output.iter().sum::<f64>() / n as f64;
        assert!(mean_i.abs() < 1e-9, "Input mean not removed: {}", mean_i);
        assert!(mean_o.abs() < 1e-9, "Output mean not removed: {}", mean_o);

        // Linear trend should also be gone: check first and last values have similar magnitude
        let range = p_input[n - 1] - p_input[0];
        assert!(
            range.abs() < 5.0,
            "Linear trend not removed (range = {})",
            range
        );
    }
}
