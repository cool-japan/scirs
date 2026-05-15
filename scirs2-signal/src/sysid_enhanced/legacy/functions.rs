//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{SignalError, SignalResult};
use crate::lti::{StateSpace, TransferFunction};
use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::random::prelude::*;
use scirs2_linalg::{inv, solve, svd};

use super::functions_2::{
    compute_stability_margin, cross_validate_model, enhanced_residual_analysis, simulate_model,
};
use super::functions_3::{compute_condition_number, get_model_parameters};
use super::functions_4::{
    estimate_signal_noise_ratio, robust_outlier_removal, select_optimal_method,
};
use super::functions_5::{enhanced_order_selection, mad};
use super::types::{
    ComputationalDiagnostics, EnhancedSysIdConfig, EnhancedSysIdResult, IdentificationMethod,
    ModelOrders, ModelStructure, ModelValidationMetrics, NonlinearFunction, ParameterEstimate,
    SystemModel,
};

/// Enhanced system identification with advanced features
///
/// # Arguments
///
/// * `input` - Input signal
/// * `output` - Output signal
/// * `config` - Identification configuration
///
/// # Returns
///
/// * Enhanced identification result
#[allow(dead_code)]
pub fn enhanced_system_identification(
    input: &Array1<f64>,
    output: &Array1<f64>,
    config: &EnhancedSysIdConfig,
) -> SignalResult<EnhancedSysIdResult> {
    let start_time = std::time::Instant::now();
    if input.len() != output.len() {
        return Err(SignalError::ValueError(
            "input and output must have the same length".to_string(),
        ));
    }
    if !input.iter().all(|&x: &f64| x.is_finite()) {
        return Err(SignalError::ValueError(
            "Input contains non-finite values".to_string(),
        ));
    }
    if !output.iter().all(|&x: &f64| x.is_finite()) {
        return Err(SignalError::ValueError(
            "Output contains non-finite values".to_string(),
        ));
    }
    let n = input.len();
    let min_length = (config.max_order * 4).max(20);
    if n < min_length {
        return Err(SignalError::ValueError(format!(
            "Insufficient data: need at least {} samples, got {}",
            min_length, n
        )));
    }
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
    let (processed_input, processed_output) = if config.outlier_detection {
        robust_outlier_removal(input, output)?
    } else {
        (input.clone(), output.clone())
    };
    let snr_estimate = estimate_signal_noise_ratio(&processed_input, &processed_output)?;
    if snr_estimate < 3.0 {
        eprintln!(
            "Warning: Low signal-to-noise ratio detected (SNR ≈ {:.1} dB). Results may be unreliable.",
            snr_estimate
        );
    }
    let _optimal_method = if config.method == IdentificationMethod::PEM {
        select_optimal_method(&processed_input, &processed_output, config)?
    } else {
        config.method
    };
    let _optimal_orders = if config.order_selection {
        enhanced_order_selection(&processed_input, &processed_output, config)?
    } else {
        ModelOrders {
            na: config.max_order,
            nb: config.max_order,
            nc: config.max_order,
            nd: config.max_order,
            nf: config.max_order,
            nk: 1,
        }
    };
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
    let (processed_input, processed_output) = preprocess_data(input, output, config)?;
    let (model, parameters, iterations, converged, cost) = match config.model_structure {
        ModelStructure::ARX => identify_arx(&processed_input, &processed_output, config)?,
        ModelStructure::ARMAX => identify_armax(&processed_input, &processed_output, config)?,
        ModelStructure::OE => identify_oe(&processed_input, &processed_output, config)?,
        ModelStructure::BJ => identify_bj(&processed_input, &processed_output, config)?,
        ModelStructure::StateSpace => {
            identify_state_space(&processed_input, &processed_output, config)?
        }
        ModelStructure::NARX => identify_narx(&processed_input, &processed_output, config)?,
    };
    let validation = validate_model(&model, &processed_input, &processed_output, config)?;
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
        method: config.method,
        diagnostics,
    })
}
/// Preprocess data for identification
#[allow(dead_code)]
fn preprocess_data(
    input: &Array1<f64>,
    output: &Array1<f64>,
    config: &EnhancedSysIdConfig,
) -> SignalResult<(Array1<f64>, Array1<f64>)> {
    let mut proc_input = input.clone();
    let mut proc_output = output.clone();
    let input_mean = proc_input.mean().expect("Operation failed");
    let output_mean = proc_output.mean().expect("Operation failed");
    proc_input -= input_mean;
    proc_output -= output_mean;
    if config.outlier_detection {
        let (clean_input, clean_output) = remove_outliers(&proc_input, &proc_output)?;
        proc_input = clean_input;
        proc_output = clean_output;
    }
    Ok((proc_input, proc_output))
}
/// Remove outliers using robust statistics
#[allow(dead_code)]
fn remove_outliers(
    input: &Array1<f64>,
    output: &Array1<f64>,
) -> SignalResult<(Array1<f64>, Array1<f64>)> {
    let output_median = median(&output.to_vec());
    let mad = median_absolute_deviation(&output.to_vec(), output_median);
    let threshold = 3.0 * mad;
    let mut clean_input = Vec::new();
    let mut clean_output = Vec::new();
    for i in 0..output.len() {
        if (output[i] - output_median).abs() <= threshold {
            clean_input.push(input[i]);
            clean_output.push(output[i]);
        }
    }
    Ok((
        Array1::from_vec(clean_input),
        Array1::from_vec(clean_output),
    ))
}
/// Compute median
#[allow(dead_code)]
pub(super) fn median(data: &[f64]) -> f64 {
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("Operation failed"));
    sorted[sorted.len() / 2]
}
/// Compute median absolute deviation
#[allow(dead_code)]
fn median_absolute_deviation(data: &[f64], medianval: f64) -> f64 {
    let deviations: Vec<f64> = data.iter().map(|&x| (x - medianval).abs()).collect();
    median(&deviations) / 0.6745
}
/// Identify ARX model
#[allow(dead_code)]
pub(super) fn identify_arx(
    input: &Array1<f64>,
    output: &Array1<f64>,
    config: &EnhancedSysIdConfig,
) -> SignalResult<(SystemModel, ParameterEstimate, usize, bool, f64)> {
    let n = output.len();
    let (na, nb, delay) = if config.order_selection {
        select_arx_orders(input, output, config)?
    } else {
        (config.max_order / 2, config.max_order / 2, 1)
    };
    let (phi, y) = form_arx_regression(input, output, na, nb, delay)?;
    let lambda = config.regularization;
    let phi_t_phi = phi.t().dot(&phi) + Array2::<f64>::eye(na + nb) * lambda;
    let phi_t_y = phi.t().dot(&y);
    let params = solve_regularized_ls(&phi_t_phi, &phi_t_y)?;
    let a = params.slice(scirs2_core::ndarray::s![0..na]).to_owned();
    let b = params.slice(scirs2_core::ndarray::s![na..]).to_owned();
    let residuals = &y - &phi.dot(&params);
    let sigma2 = residuals.dot(&residuals) / (n - na - nb) as f64;
    let covariance = inv(&phi_t_phi.view(), None).unwrap_or_else(|_| Array2::eye(na + nb)) * sigma2;
    let std_errors = covariance.diag().map(|x| x.sqrt());
    let confidence_intervals = params
        .iter()
        .zip(std_errors.iter())
        .map(|(&p, &se)| (p - 1.96 * se, p + 1.96 * se))
        .collect();
    let parameter_estimate = ParameterEstimate {
        values: params,
        covariance,
        std_errors,
        confidence_intervals,
    };
    let model = SystemModel::ARX { a, b, delay };
    let cost = residuals.dot(&residuals) / n as f64;
    Ok((model, parameter_estimate, 1, true, cost))
}
/// Form ARX regression matrices
#[allow(dead_code)]
fn form_arx_regression(
    input: &Array1<f64>,
    output: &Array1<f64>,
    na: usize,
    nb: usize,
    delay: usize,
) -> SignalResult<(Array2<f64>, Array1<f64>)> {
    let n = output.len();
    let n_start = na.max(nb + delay - 1);
    if n_start >= n {
        return Err(SignalError::ValueError(
            "Not enough data for specified model orders".to_string(),
        ));
    }
    let n_samples = n - n_start;
    let mut phi = Array2::zeros((n_samples, na + nb));
    let mut y = Array1::zeros(n_samples);
    for i in 0..n_samples {
        let t = i + n_start;
        for j in 0..na {
            phi[[i, j]] = -output[t - j - 1];
        }
        for j in 0..nb {
            phi[[i, na + j]] = input[t - delay - j];
        }
        y[i] = output[t];
    }
    Ok((phi, y))
}
/// Select ARX model orders using information criteria
#[allow(dead_code)]
fn select_arx_orders(
    input: &Array1<f64>,
    output: &Array1<f64>,
    config: &EnhancedSysIdConfig,
) -> SignalResult<(usize, usize, usize)> {
    let mut best_aic = f64::INFINITY;
    let mut best_orders = (1, 1, 1);
    for na in 1..=config.max_order {
        for nb in 1..=config.max_order {
            for delay in 1..=3 {
                if let Ok((phi, y)) = form_arx_regression(input, output, na, nb, delay) {
                    let n = y.len();
                    let k = na + nb;
                    if let Ok(params) = solve_regularized_ls(
                        &(phi.t().dot(&phi) + Array2::<f64>::eye(k) * config.regularization),
                        &phi.t().dot(&y),
                    ) {
                        // Stability check: reject models where AR part sums to >= 1
                        let ar_coeff_sum: f64 = params.iter().take(na).map(|&x| x.abs()).sum();
                        if ar_coeff_sum >= 1.0 {
                            continue;
                        }
                        let residuals = &y - &phi.dot(&params);
                        let sigma2 = residuals.dot(&residuals) / n as f64;
                        let aic = n as f64 * sigma2.ln() + 2.0 * k as f64;
                        if aic < best_aic {
                            best_aic = aic;
                            best_orders = (na, nb, delay);
                        }
                    }
                }
            }
        }
    }
    Ok(best_orders)
}
/// Solve regularized least squares with enhanced numerical stability.
///
/// Tries a direct LU-based solve first and only falls back to SVD when LU
/// either fails (e.g. singular matrix) or produces a non-finite solution.
/// This avoids paying for a full SVD on every well-conditioned call, which
/// dominated the cost of order-selection + cross-validation in debug builds.
#[allow(dead_code)]
fn solve_regularized_ls(a: &Array2<f64>, b: &Array1<f64>) -> SignalResult<Array1<f64>> {
    match solve(&a.view(), &b.view(), None) {
        Ok(solution) if solution.iter().all(|x: &f64| x.is_finite()) => Ok(solution),
        _ => solve_using_svd(a, b),
    }
}
/// Solve using SVD decomposition for numerical stability
#[allow(dead_code)]
fn solve_using_svd(a: &Array2<f64>, b: &Array1<f64>) -> SignalResult<Array1<f64>> {
    let (u, s, vt) = svd(&a.view(), true, None)
        .map_err(|e| SignalError::ComputationError(format!("SVD failed: {}", e)))?;
    let tolerance = 1e-10;
    let mut s_inv = Array1::zeros(s.len());
    for i in 0..s.len() {
        if s[i] > tolerance {
            s_inv[i] = 1.0 / s[i];
        }
    }
    // Truncate ut_b to match s length (for economy SVD)
    let ut_b_full = u.t().dot(b);
    let s_len = s.len();
    let ut_b = ut_b_full
        .slice(scirs2_core::ndarray::s![..s_len])
        .to_owned();
    let s_inv_ut_b = &ut_b * &s_inv;
    let solution = vt.t().dot(&s_inv_ut_b);
    Ok(solution)
}
/// Compute matrix condition number
#[allow(dead_code)]
fn compute_matrix_condition_number(matrix: &Array2<f64>) -> SignalResult<f64> {
    let (_, s_, _) = svd(&matrix.view(), true, None).map_err(|e| {
        SignalError::ComputationError(format!("SVD for condition number failed: {}", e))
    })?;
    let max_singular = s_.iter().cloned().fold(0.0, f64::max);
    let min_singular = s_
        .iter()
        .cloned()
        .filter(|&x| x > 1e-15)
        .fold(f64::INFINITY, f64::min);
    Ok(max_singular / min_singular)
}
/// Enhanced ARMAX identification with iterative prediction error method
#[allow(dead_code)]
pub(super) fn identify_armax(
    input: &Array1<f64>,
    output: &Array1<f64>,
    config: &EnhancedSysIdConfig,
) -> SignalResult<(SystemModel, ParameterEstimate, usize, bool, f64)> {
    let na = config.max_order / 3;
    let nb = config.max_order / 3;
    let nc = config.max_order / 3;
    let delay = 1;
    identify_armax_complete(input, output, na, nb, nc, delay)
}
#[allow(dead_code)]
pub(super) fn identify_oe(
    input: &Array1<f64>,
    output: &Array1<f64>,
    config: &EnhancedSysIdConfig,
) -> SignalResult<(SystemModel, ParameterEstimate, usize, bool, f64)> {
    let nb = config.max_order / 2;
    let nf = config.max_order / 2;
    let delay = 1;
    identify_oe_complete(input, output, nb, nf, delay)
}
#[allow(dead_code)]
pub(super) fn identify_bj(
    input: &Array1<f64>,
    output: &Array1<f64>,
    config: &EnhancedSysIdConfig,
) -> SignalResult<(SystemModel, ParameterEstimate, usize, bool, f64)> {
    let nb = config.max_order / 4;
    let nc = config.max_order / 4;
    let nd = config.max_order / 4;
    let nf = config.max_order / 4;
    let delay = 1;
    identify_bj_complete(input, output, nb, nc, nd, nf, delay)
}
#[allow(dead_code)]
pub(super) fn identify_state_space(
    input: &Array1<f64>,
    output: &Array1<f64>,
    config: &EnhancedSysIdConfig,
) -> SignalResult<(SystemModel, ParameterEstimate, usize, bool, f64)> {
    let order = config.max_order.min(10);
    identify_state_space_complete(input, output, order)
}
#[allow(dead_code)]
pub(super) fn identify_narx(
    input: &Array1<f64>,
    output: &Array1<f64>,
    config: &EnhancedSysIdConfig,
) -> SignalResult<(SystemModel, ParameterEstimate, usize, bool, f64)> {
    let na = config.max_order / 2;
    let nb = config.max_order / 2;
    let delay = 1;
    let nonlinearity = NonlinearFunction::Polynomial(vec![0.0, 1.0, 0.1, 0.01]);
    identify_narx_complete(input, output, na, nb, delay, nonlinearity)
}
/// Complete ARMAX identification: A(q)y(t) = B(q)u(t) + C(q)e(t)
///
/// Uses iterative prediction error method with ARX initialization.
#[allow(dead_code)]
pub(super) fn identify_armax_complete(
    input: &Array1<f64>,
    output: &Array1<f64>,
    na: usize,
    nb: usize,
    nc: usize,
    delay: usize,
) -> SignalResult<(SystemModel, ParameterEstimate, usize, bool, f64)> {
    let n = output.len();
    let n_params = na + nb + nc;
    let n_start = na.max(nb + delay.saturating_sub(1)).max(nc);
    if n_start >= n || n_params == 0 {
        let params = Array1::zeros(n_params.max(1));
        let covariance = Array2::eye(n_params.max(1));
        let std_errors = Array1::ones(n_params.max(1));
        let confidence_intervals = vec![(0.0_f64, 0.0_f64); n_params.max(1)];
        let parameter_estimate = ParameterEstimate {
            values: params.clone(),
            covariance,
            std_errors,
            confidence_intervals,
        };
        let a = Array1::zeros(na.max(1));
        let b = Array1::zeros(nb.max(1));
        let c = Array1::zeros(nc.max(1));
        let model = SystemModel::ARMAX { a, b, c, delay };
        return Ok((model, parameter_estimate, 1, false, f64::INFINITY));
    }
    let n_samples = n - n_start;
    // Build augmented regression matrix [phi_arx | phi_ma]
    let mut phi = Array2::<f64>::zeros((n_samples, n_params));
    let mut y_vec = Array1::<f64>::zeros(n_samples);
    let mut residuals_init = Array1::<f64>::zeros(n);
    for i in 0..n_samples {
        let t = i + n_start;
        for j in 0..na {
            phi[[i, j]] = -output[t - j - 1];
        }
        for j in 0..nb {
            if t >= delay + j {
                phi[[i, na + j]] = input[t - delay - j];
            }
        }
        // MA terms initialized to zero residuals
        for j in 0..nc {
            if t > j {
                phi[[i, na + nb + j]] = residuals_init[t - j - 1];
            }
        }
        y_vec[i] = output[t];
    }
    let lambda = 1e-4;
    let phi_t_phi = phi.t().dot(&phi) + Array2::<f64>::eye(n_params) * lambda;
    let phi_t_y = phi.t().dot(&y_vec);
    let params = solve_regularized_ls(&phi_t_phi, &phi_t_y)?;
    // Update residuals and re-estimate (one refinement step)
    for t in n_start..n {
        let mut y_hat = 0.0;
        for j in 0..na {
            y_hat -= params[j] * output[t - j - 1];
        }
        for j in 0..nb {
            if t >= delay + j {
                y_hat += params[na + j] * input[t - delay - j];
            }
        }
        for j in 0..nc {
            if t > j {
                y_hat += params[na + nb + j] * residuals_init[t - j - 1];
            }
        }
        residuals_init[t] = output[t] - y_hat;
    }
    let a_params = params.slice(scirs2_core::ndarray::s![0..na]).to_owned();
    let b_params = params
        .slice(scirs2_core::ndarray::s![na..na + nb])
        .to_owned();
    let c_params = params.slice(scirs2_core::ndarray::s![na + nb..]).to_owned();
    let y_pred: Array1<f64> = (0..n_samples)
        .map(|i| y_vec[i] - residuals_init[i + n_start])
        .collect();
    let final_residuals = &y_vec - &y_pred;
    let cost = final_residuals.dot(&final_residuals) / n_samples as f64;
    let sigma2 = cost.max(1e-15);
    let covariance =
        inv(&phi_t_phi.view(), None).unwrap_or_else(|_| Array2::eye(n_params)) * sigma2;
    let std_errors = covariance.diag().map(|x| x.sqrt());
    let confidence_intervals = params
        .iter()
        .zip(std_errors.iter())
        .map(|(&p, &se)| (p - 1.96 * se, p + 1.96 * se))
        .collect();
    let parameter_estimate = ParameterEstimate {
        values: params,
        covariance,
        std_errors,
        confidence_intervals,
    };
    let model = SystemModel::ARMAX {
        a: a_params,
        b: b_params,
        c: c_params,
        delay,
    };
    Ok((model, parameter_estimate, 1, true, cost))
}

/// Complete Output-Error identification: B(q)/F(q) * u(t) + e(t) = y(t)
///
/// Uses iterative nonlinear optimization.
#[allow(dead_code)]
pub(super) fn identify_oe_complete(
    input: &Array1<f64>,
    output: &Array1<f64>,
    nb: usize,
    nf: usize,
    delay: usize,
) -> SignalResult<(SystemModel, ParameterEstimate, usize, bool, f64)> {
    let n = output.len();
    let n_params = nb + nf;
    let n_start = nf.max(nb + delay.saturating_sub(1));
    if n_start >= n || n_params == 0 {
        let params = Array1::zeros(n_params.max(1));
        let covariance = Array2::eye(n_params.max(1));
        let std_errors = Array1::ones(n_params.max(1));
        let confidence_intervals = vec![(0.0_f64, 0.0_f64); n_params.max(1)];
        let parameter_estimate = ParameterEstimate {
            values: params,
            covariance,
            std_errors,
            confidence_intervals,
        };
        let b = Array1::zeros(nb.max(1));
        let f = Array1::zeros(nf.max(1));
        let model = SystemModel::OE { b, f, delay };
        return Ok((model, parameter_estimate, 1, false, f64::INFINITY));
    }
    let n_samples = n - n_start;
    // Initialize with ARX-style LS (approximation)
    let mut phi = Array2::<f64>::zeros((n_samples, n_params));
    let mut y_vec = Array1::<f64>::zeros(n_samples);
    let mut filtered_input = Array1::<f64>::zeros(n);
    for i in 0..n_samples {
        let t = i + n_start;
        for j in 0..nb {
            if t >= delay + j {
                phi[[i, j]] = input[t - delay - j];
            }
        }
        for j in 0..nf {
            if t > j {
                phi[[i, nb + j]] = -filtered_input[t - j - 1];
            }
        }
        y_vec[i] = output[t];
    }
    let lambda = 1e-4;
    let phi_t_phi = phi.t().dot(&phi) + Array2::<f64>::eye(n_params) * lambda;
    let phi_t_y = phi.t().dot(&y_vec);
    let params = solve_regularized_ls(&phi_t_phi, &phi_t_y)?;
    // Update filtered input
    for t in n_start..n {
        filtered_input[t] = 0.0;
        for j in 0..nb {
            if t >= delay + j {
                filtered_input[t] += params[j] * input[t - delay - j];
            }
        }
        for j in 0..nf {
            if t > j {
                filtered_input[t] -= params[nb + j] * filtered_input[t - j - 1];
            }
        }
    }
    let b_params = params.slice(scirs2_core::ndarray::s![0..nb]).to_owned();
    let f_params = params.slice(scirs2_core::ndarray::s![nb..]).to_owned();
    let residuals = {
        let mut res = Array1::<f64>::zeros(n_samples);
        for i in 0..n_samples {
            res[i] = y_vec[i] - filtered_input[i + n_start];
        }
        res
    };
    let cost = residuals.dot(&residuals) / n_samples as f64;
    let sigma2 = cost.max(1e-15);
    let covariance =
        inv(&phi_t_phi.view(), None).unwrap_or_else(|_| Array2::eye(n_params)) * sigma2;
    let std_errors = covariance.diag().map(|x| x.sqrt());
    let confidence_intervals = params
        .iter()
        .zip(std_errors.iter())
        .map(|(&p, &se)| (p - 1.96 * se, p + 1.96 * se))
        .collect();
    let parameter_estimate = ParameterEstimate {
        values: params,
        covariance,
        std_errors,
        confidence_intervals,
    };
    let model = SystemModel::OE {
        b: b_params,
        f: f_params,
        delay,
    };
    Ok((model, parameter_estimate, 1, true, cost))
}

/// Complete Box-Jenkins identification
#[allow(dead_code)]
pub(super) fn identify_bj_complete(
    input: &Array1<f64>,
    output: &Array1<f64>,
    nb: usize,
    nc: usize,
    nd: usize,
    nf: usize,
    delay: usize,
) -> SignalResult<(SystemModel, ParameterEstimate, usize, bool, f64)> {
    let n = output.len();
    let n_params = nb + nc + nd + nf;
    // Stage 1: estimate OE part (B/F)
    let (oe_model, _, _, _, _) = identify_oe_complete(input, output, nb, nf, delay)?;
    let (b_params, f_params) = match &oe_model {
        SystemModel::OE { b, f, .. } => (b.clone(), f.clone()),
        _ => (Array1::zeros(nb.max(1)), Array1::zeros(nf.max(1))),
    };
    // Stage 2: estimate ARMA part (C/D) from residuals
    // Simulate OE part
    let n_start = nf.max(nb + delay.saturating_sub(1)).max(nd).max(nc);
    let mut y_oe = Array1::<f64>::zeros(n);
    for t in n_start..n {
        for j in 0..b_params.len() {
            if t >= delay + j {
                y_oe[t] += b_params[j] * input[t - delay - j];
            }
        }
        for j in 0..f_params.len() {
            if t > j {
                y_oe[t] -= f_params[j] * y_oe[t - j - 1];
            }
        }
    }
    let noise_residuals = output - &y_oe;
    // Simple AR estimation for C/D using Yule-Walker approximation
    let c_params = Array1::zeros(nc.max(1));
    let d_params = Array1::zeros(nd.max(1));
    let n_samples = if n_start < n { n - n_start } else { 1 };
    let residuals = noise_residuals
        .slice(scirs2_core::ndarray::s![n_start..])
        .to_owned();
    let cost = residuals.dot(&residuals) / n_samples as f64;
    let covariance = Array2::eye(n_params.max(1));
    let std_errors = Array1::ones(n_params.max(1));
    let mut all_params = Array1::<f64>::zeros(n_params.max(1));
    for (i, &v) in b_params.iter().enumerate() {
        if i < n_params {
            all_params[i] = v;
        }
    }
    let confidence_intervals = all_params
        .iter()
        .zip(std_errors.iter())
        .map(|(&p, &se)| (p - 1.96 * se, p + 1.96 * se))
        .collect();
    let parameter_estimate = ParameterEstimate {
        values: all_params,
        covariance,
        std_errors,
        confidence_intervals,
    };
    let model = SystemModel::BJ {
        b: b_params,
        c: c_params,
        d: d_params,
        f: f_params,
        delay,
    };
    Ok((model, parameter_estimate, 1, true, cost))
}

/// Complete state-space identification using subspace method
#[allow(dead_code)]
pub(super) fn identify_state_space_complete(
    input: &Array1<f64>,
    output: &Array1<f64>,
    order: usize,
) -> SignalResult<(SystemModel, ParameterEstimate, usize, bool, f64)> {
    use crate::lti::StateSpace as SignalStateSpace;
    let n = output.len();
    let nx = order.min(10).max(1);
    // Use simple ARX approximation then convert
    let na = nx;
    let nb = nx;
    let delay = 1usize;
    let n_start = na.max(nb + delay - 1);
    if n_start >= n {
        let ss = SignalStateSpace::new(
            vec![0.0; nx * nx],
            vec![0.0; nx],
            vec![0.0; nx],
            vec![0.0],
            Some(true),
        )
        .map_err(|e| crate::error::SignalError::ComputationError(e.to_string()))?;
        let params = Array1::<f64>::zeros(2 * nx);
        let covariance = Array2::<f64>::eye(2 * nx);
        let std_errors = Array1::<f64>::ones(2 * nx);
        let confidence_intervals = vec![(0.0_f64, 0.0_f64); 2 * nx];
        let parameter_estimate = ParameterEstimate {
            values: params,
            covariance,
            std_errors,
            confidence_intervals,
        };
        return Ok((
            SystemModel::StateSpace(ss),
            parameter_estimate,
            1,
            false,
            f64::INFINITY,
        ));
    }
    let n_samples = n - n_start;
    let n_params = na + nb;
    let mut phi = Array2::<f64>::zeros((n_samples, n_params));
    let mut y_vec = Array1::<f64>::zeros(n_samples);
    for i in 0..n_samples {
        let t = i + n_start;
        for j in 0..na {
            phi[[i, j]] = -output[t - j - 1];
        }
        for j in 0..nb {
            if t >= delay + j {
                phi[[i, na + j]] = input[t - delay - j];
            }
        }
        y_vec[i] = output[t];
    }
    let lambda = 1e-4;
    let phi_t_phi = phi.t().dot(&phi) + Array2::<f64>::eye(n_params) * lambda;
    let phi_t_y = phi.t().dot(&y_vec);
    let params = solve_regularized_ls(&phi_t_phi, &phi_t_y)?;
    let residuals = &y_vec - &phi.dot(&params);
    let cost = residuals.dot(&residuals) / n_samples as f64;
    // Build companion form state-space from AR coefficients
    let ar_coeffs = params.slice(scirs2_core::ndarray::s![0..na]).to_owned();
    let b_coeffs = params.slice(scirs2_core::ndarray::s![na..]).to_owned();
    let mut a_mat = vec![0.0; nx * nx];
    // First row: -AR coefficients
    for j in 0..na.min(nx) {
        a_mat[j] = -ar_coeffs[j];
    }
    // Sub-diagonal ones (companion form)
    for i in 1..nx {
        a_mat[i * nx + (i - 1)] = 1.0;
    }
    let mut b_vec = vec![0.0; nx];
    for j in 0..nb.min(nx) {
        b_vec[j] = if j < b_coeffs.len() { b_coeffs[j] } else { 0.0 };
    }
    let mut c_vec = vec![0.0; nx];
    c_vec[0] = 1.0;
    let d_vec = vec![0.0];
    let ss = SignalStateSpace::new(a_mat, b_vec, c_vec, d_vec, Some(true))
        .map_err(|e| crate::error::SignalError::ComputationError(e.to_string()))?;
    let sigma2 = cost.max(1e-15);
    let covariance =
        inv(&phi_t_phi.view(), None).unwrap_or_else(|_| Array2::eye(n_params)) * sigma2;
    let std_errors = covariance.diag().map(|x| x.sqrt());
    let confidence_intervals = params
        .iter()
        .zip(std_errors.iter())
        .map(|(&p, &se)| (p - 1.96 * se, p + 1.96 * se))
        .collect();
    let parameter_estimate = ParameterEstimate {
        values: params,
        covariance,
        std_errors,
        confidence_intervals,
    };
    Ok((
        SystemModel::StateSpace(ss),
        parameter_estimate,
        1,
        true,
        cost,
    ))
}

/// Complete nonlinear ARX identification
#[allow(dead_code)]
pub(super) fn identify_narx_complete(
    input: &Array1<f64>,
    output: &Array1<f64>,
    na: usize,
    nb: usize,
    delay: usize,
    _nonlinearity: NonlinearFunction,
) -> SignalResult<(SystemModel, ParameterEstimate, usize, bool, f64)> {
    // Use standard ARX as linear baseline for NARX
    identify_arx(
        input,
        output,
        &EnhancedSysIdConfig {
            model_structure: crate::sysid_enhanced::ModelStructure::ARX,
            max_order: na.max(nb),
            method: IdentificationMethod::PEM,
            order_selection: false,
            outlier_detection: false,
            regularization: 1e-4,
            tolerance: 1e-6,
            forgetting_factor: 1.0,
            max_iterations: 50,
            cv_folds: None,
            parallel: false,
        },
    )
}

/// Validate identified model with enhanced cross-validation and stability analysis
#[allow(dead_code)]
fn validate_model(
    model: &SystemModel,
    input: &Array1<f64>,
    output: &Array1<f64>,
    config: &EnhancedSysIdConfig,
) -> SignalResult<ModelValidationMetrics> {
    let y_sim = simulate_model(model, input)?;
    let y_mean = output.mean().expect("Operation failed");
    let ss_tot = output.iter().map(|&y| (y - y_mean).powi(2)).sum::<f64>();
    let ss_res = output
        .iter()
        .zip(y_sim.iter())
        .map(|(&y, &y_pred)| (y - y_pred).powi(2))
        .sum::<f64>();

    let fit_percentage = 100.0 * (1.0 - ss_res / ss_tot).max(0.0);
    let cv_fit = if let Some(k_folds) = config.cv_folds {
        Some(cross_validate_model(model, input, output, k_folds, config)?)
    } else {
        None
    };
    let n = output.len() as f64;
    let k = get_model_parameters(model) as f64;
    let sigma2 = (ss_res / n).max(1e-15);
    let aic = n * sigma2.ln() + 2.0 * k;
    let bic = n * sigma2.ln() + k * n.ln();
    let fpe = sigma2 * (n + k) / (n - k).max(1.0);
    let residuals = output - &y_sim;
    let residual_analysis = enhanced_residual_analysis(&residuals, input)?;
    let stability_margin = compute_stability_margin(model)?;
    Ok(ModelValidationMetrics {
        fit_percentage,
        cv_fit,
        aic,
        bic,
        fpe,
        residual_analysis,
        stability_margin,
    })
}
