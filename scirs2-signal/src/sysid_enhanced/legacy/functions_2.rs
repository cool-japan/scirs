//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{SignalError, SignalResult};
use crate::lti::{StateSpace, TransferFunction};
use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::numeric::Complex64;
use scirs2_core::random::prelude::*;
use scirs2_linalg::eig;

use super::functions::{
    identify_armax, identify_arx, identify_bj, identify_narx, identify_oe, identify_state_space,
};
use super::functions_3::{simulate_state_space, transfer_function_to_state_space};
use super::types::{
    EnhancedSysIdConfig, ModelStructure, NonlinearFunction, ResidualAnalysis, SystemModel,
};

/// Cross-validate model performance
#[allow(dead_code)]
pub(super) fn cross_validate_model(
    _model: &SystemModel,
    input: &Array1<f64>,
    output: &Array1<f64>,
    k_folds: usize,
    config: &EnhancedSysIdConfig,
) -> SignalResult<f64> {
    let n = output.len();
    let fold_size = n / k_folds;
    let mut cv_scores = Vec::with_capacity(k_folds);
    for fold in 0..k_folds {
        let test_start = fold * fold_size;
        let test_end = if fold == k_folds - 1 {
            n
        } else {
            (fold + 1) * fold_size
        };
        let mut train_input = Vec::new();
        let mut train_output = Vec::new();
        for i in 0..n {
            if i < test_start || i >= test_end {
                train_input.push(input[i]);
                train_output.push(output[i]);
            }
        }
        let train_input_arr = Array1::from_vec(train_input);
        let train_output_arr = Array1::from_vec(train_output);
        let cv_model = match config.model_structure {
            ModelStructure::ARX => identify_arx(&train_input_arr, &train_output_arr, config)?,
            ModelStructure::ARMAX => identify_armax(&train_input_arr, &train_output_arr, config)?,
            ModelStructure::OE => identify_oe(&train_input_arr, &train_output_arr, config)?,
            ModelStructure::BJ => identify_bj(&train_input_arr, &train_output_arr, config)?,
            ModelStructure::StateSpace => {
                identify_state_space(&train_input_arr, &train_output_arr, config)?
            }
            ModelStructure::NARX => identify_narx(&train_input_arr, &train_output_arr, config)?,
        };
        let test_input = input
            .slice(scirs2_core::ndarray::s![test_start..test_end])
            .to_owned();
        let test_output = output
            .slice(scirs2_core::ndarray::s![test_start..test_end])
            .to_owned();
        let y_pred = simulate_model(&cv_model.0, &test_input)?;
        let y_mean = test_output.mean().expect("Operation failed");
        let ss_tot = test_output
            .iter()
            .map(|&y| (y - y_mean).powi(2))
            .sum::<f64>();
        let ss_res = test_output
            .iter()
            .zip(y_pred.iter())
            .map(|(&y, &y_pred)| (y - y_pred).powi(2))
            .sum::<f64>();
        let score = 100.0 * (1.0 - ss_res / ss_tot).max(0.0);
        cv_scores.push(score);
    }
    Ok(cv_scores.iter().sum::<f64>() / k_folds as f64)
}
/// Enhanced residual analysis with statistical tests
#[allow(dead_code)]
pub(super) fn enhanced_residual_analysis(
    residuals: &Array1<f64>,
    input: &Array1<f64>,
) -> SignalResult<ResidualAnalysis> {
    let max_lag = 20.min(residuals.len() / 4);
    let mut autocorrelation = Array1::zeros(max_lag);
    let r_mean = residuals.mean().expect("Operation failed");
    let r_var =
        residuals.iter().map(|&r| (r - r_mean).powi(2)).sum::<f64>() / residuals.len() as f64;
    for lag in 0..max_lag {
        let mut sum = 0.0;
        let mut count = 0;
        for i in lag..residuals.len() {
            sum += (residuals[i] - r_mean) * (residuals[i - lag] - r_mean);
            count += 1;
        }
        autocorrelation[lag] = if count > 0 {
            sum / (count as f64 * r_var)
        } else {
            0.0
        };
    }
    let mut cross_correlation = Array1::zeros(max_lag);
    let i_mean = input.mean().expect("Operation failed");
    let i_var = input.iter().map(|&i| (i - i_mean).powi(2)).sum::<f64>() / input.len() as f64;
    for lag in 0..max_lag {
        let mut sum = 0.0;
        let mut count = 0;
        for i in lag..residuals.len().min(input.len()) {
            sum += (residuals[i] - r_mean) * (input[i - lag] - i_mean);
            count += 1;
        }
        cross_correlation[lag] = if count > 0 {
            sum / (count as f64 * (r_var * i_var).sqrt())
        } else {
            0.0
        };
    }
    let whiteness_pvalue = ljung_box_test(&autocorrelation);
    let independence_pvalue = cross_correlation_test(&cross_correlation);
    let normality_pvalue = jarque_bera_test(residuals);
    Ok(ResidualAnalysis {
        autocorrelation,
        cross_correlation,
        whiteness_pvalue,
        independence_pvalue,
        normality_pvalue,
    })
}
/// Ljung-Box test for whiteness
#[allow(dead_code)]
fn ljung_box_test(autocorr: &Array1<f64>) -> f64 {
    let n = autocorr.len() as f64;
    let h = autocorr.len().min(10);
    let mut lb_stat = 0.0;
    for k in 1..h {
        let rho_k = autocorr[k];
        lb_stat += rho_k * rho_k / (n - k as f64);
    }
    lb_stat *= n * (n + 2.0);
    chi_square_pvalue(lb_stat, h - 1)
}
/// Cross-correlation independence test
#[allow(dead_code)]
fn cross_correlation_test(crosscorr: &Array1<f64>) -> f64 {
    let max_corr = crosscorr.iter().map(|&x: &f64| x.abs()).fold(0.0, f64::max);
    let n = crosscorr.len() as f64;
    let test_stat = max_corr * n.sqrt();
    2.0 * (1.0 - standard_normal_cdf(test_stat))
}
/// Jarque-Bera test for normality
#[allow(dead_code)]
pub fn jarque_bera_test(data: &Array1<f64>) -> f64 {
    let n = data.len() as f64;
    let mean = data.mean().expect("Operation failed");
    let mut m2 = 0.0;
    let mut m3 = 0.0;
    let mut m4 = 0.0;
    for &x in data.iter() {
        let diff = x - mean;
        let diff2 = diff * diff;
        m2 += diff2;
        m3 += diff2 * diff;
        m4 += diff2 * diff2;
    }
    m2 /= n;
    m3 /= n;
    m4 /= n;
    let skewness = m3 / m2.powf(1.5);
    let kurtosis = m4 / (m2 * m2) - 3.0;
    let jb_stat = n / 6.0 * (skewness * skewness + kurtosis * kurtosis / 4.0);
    chi_square_pvalue(jb_stat, 2)
}
/// Chi-square p-value approximation
#[allow(dead_code)]
fn chi_square_pvalue(x: f64, df: usize) -> f64 {
    if df == 1 {
        2.0 * (1.0 - standard_normal_cdf(x.sqrt()))
    } else if df == 2 {
        (-x / 2.0).exp()
    } else {
        let mean = df as f64;
        let variance = 2.0 * df as f64;
        let z = (x - mean) / variance.sqrt();
        1.0 - standard_normal_cdf(z)
    }
}
/// Standard normal CDF approximation
#[allow(dead_code)]
fn standard_normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / 2.0_f64.sqrt()))
}
/// Error function approximation
#[allow(dead_code)]
fn erf(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;
    let sign = x.signum();
    let x = x.abs();
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}
/// Compute stability margin for different model types
#[allow(dead_code)]
pub(super) fn compute_stability_margin(model: &SystemModel) -> SignalResult<f64> {
    match model {
        SystemModel::ARX { a, .. } => {
            let roots = compute_polynomial_roots(a)?;
            let min_margin = roots
                .iter()
                .map(|r| r.norm())
                .fold(f64::NEG_INFINITY, f64::max);
            // Stability margin is 1 - max_eigenvalue_magnitude
            Ok((1.0 - min_margin).max(0.0))
        }
        SystemModel::StateSpace(ss) => {
            // Build the state matrix as Array2 from the Vec<f64>
            let n = ss.n_states;
            if n == 0 {
                return Ok(1.0);
            }
            let a_mat = Array2::from_shape_vec((n, n), ss.a.clone())
                .map_err(|e| SignalError::ComputationError(format!("Matrix shape error: {}", e)))?;
            let (eigenvalues, _) = eig(&a_mat.view(), None).map_err(|e| {
                SignalError::ComputationError(format!("Eigenvalue computation failed: {}", e))
            })?;
            let max_magnitude = eigenvalues
                .iter()
                .map(|&lambda| (lambda.re * lambda.re + lambda.im * lambda.im).sqrt())
                .fold(f64::NEG_INFINITY, f64::max);
            Ok((1.0 - max_magnitude).max(0.0))
        }
        _ => Ok(0.5),
    }
}
/// Compute polynomial roots (simplified for stability analysis)
#[allow(dead_code)]
fn compute_polynomial_roots(coeffs: &Array1<f64>) -> SignalResult<Vec<Complex64>> {
    let n = coeffs.len() - 1;
    if n == 0 {
        return Ok(vec![]);
    }
    let mut companion = Array2::zeros((n, n));
    let leading_coeff = coeffs[0];
    for i in 0..n {
        companion[[0, i]] = -coeffs[i + 1] / leading_coeff;
    }
    for i in 1..n {
        companion[[i, i - 1]] = 1.0;
    }
    match eig(&companion.view(), None) {
        Ok((eigenvals, _)) => Ok(eigenvals
            .iter()
            .map(|&c| Complex64::new(c.re, c.im))
            .collect()),
        Err(_) => {
            let sum_abs_coeffs: f64 = coeffs.iter().skip(1).map(|&c: &f64| c.abs()).sum();
            let leading_abs = coeffs[0].abs();
            if sum_abs_coeffs < leading_abs {
                Ok(vec![Complex64::new(0.5, 0.0)])
            } else {
                Ok(vec![Complex64::new(1.1, 0.0)])
            }
        }
    }
}
/// Simulate model response
#[allow(dead_code)]
pub(super) fn simulate_model(
    model: &SystemModel,
    input: &Array1<f64>,
) -> SignalResult<Array1<f64>> {
    match model {
        SystemModel::ARX { a, b, delay } => {
            let n = input.len();
            let mut output = Array1::zeros(n);
            for t in (*delay + b.len()).max(a.len())..n {
                for i in 0..a.len() {
                    output[t] -= a[i] * output[t - i - 1];
                }
                for i in 0..b.len() {
                    if t >= *delay + i {
                        output[t] += b[i] * input[t - delay - i];
                    }
                }
            }
            Ok(output)
        }
        SystemModel::ARMAX { a, b, c, delay } => {
            let n = input.len();
            let mut output = Array1::zeros(n);
            let mut noise = Array1::zeros(n);
            let mut rng = scirs2_core::random::rng();
            for i in 0..n {
                noise[i] = rng.random_range(-1.0..1.0) * 0.1;
            }
            for t in (*delay + b.len().max(c.len())).max(a.len())..n {
                for i in 0..a.len() {
                    output[t] -= a[i] * output[t - i - 1];
                }
                for i in 0..b.len() {
                    if t >= *delay + i {
                        output[t] += b[i] * input[t - delay - i];
                    }
                }
                for i in 0..c.len() {
                    if t >= i {
                        output[t] += c[i] * noise[t - i];
                    }
                }
            }
            Ok(output)
        }
        SystemModel::OE { b, f, delay } => {
            let n = input.len();
            let mut output = Array1::zeros(n);
            let mut filtered_input = Array1::zeros(n);
            for t in (*delay + b.len()).max(f.len())..n {
                for i in 0..b.len() {
                    if t >= *delay + i {
                        filtered_input[t] += b[i] * input[t - delay - i];
                    }
                }
                for i in 0..f.len() {
                    if i > 0 && t >= i {
                        filtered_input[t] -= f[i] * filtered_input[t - i];
                    }
                }
                output[t] = filtered_input[t];
            }
            Ok(output)
        }
        SystemModel::BJ { b, c, d, f, delay } => {
            let n = input.len();
            let mut output = Array1::<f64>::zeros(n);
            let mut filtered_input = Array1::<f64>::zeros(n);
            let mut noise = Array1::<f64>::zeros(n);
            let mut filtered_noise = Array1::<f64>::zeros(n);
            let mut rng = scirs2_core::random::rng();
            for i in 0..n {
                noise[i] = rng.random_range(-1.0..1.0) * 0.1;
            }
            for t in (*delay + b.len()).max(f.len()).max(c.len()).max(d.len())..n {
                for i in 0..b.len() {
                    if t >= *delay + i {
                        filtered_input[t] += b[i] * input[t - delay - i];
                    }
                }
                for i in 1..f.len() {
                    if t >= i {
                        filtered_input[t] -= f[i] * filtered_input[t - i];
                    }
                }
                for i in 0..c.len() {
                    if t >= i {
                        filtered_noise[t] += c[i] * noise[t - i];
                    }
                }
                for i in 1..d.len() {
                    if t >= i {
                        filtered_noise[t] -= d[i] * filtered_noise[t - i];
                    }
                }
                output[t] = filtered_input[t] + filtered_noise[t];
            }
            Ok(output)
        }
        SystemModel::HammersteinWiener {
            linear,
            input_nonlinearity,
            output_nonlinearity,
        } => {
            let n = input.len();
            let mut nonlinear_input = Array1::zeros(n);
            for i in 0..n {
                nonlinear_input[i] = apply_nonlinear_function(input[i], input_nonlinearity)?;
            }
            let linear_output = simulate_model(linear, &nonlinear_input)?;
            let mut output = Array1::zeros(n);
            for i in 0..n {
                output[i] = apply_nonlinear_function(linear_output[i], output_nonlinearity)?;
            }
            Ok(output)
        }
        SystemModel::TransferFunction(tf) => {
            let ss = transfer_function_to_state_space(tf)?;
            simulate_state_space(&ss, input)
        }
        SystemModel::StateSpace(ss) => simulate_state_space(ss, input),
    }
}
/// Apply nonlinear function
#[allow(dead_code)]
fn apply_nonlinear_function(input: f64, func: &NonlinearFunction) -> SignalResult<f64> {
    match func {
        NonlinearFunction::Polynomial(coeffs) => {
            let mut result = 0.0;
            for (i, &coeff) in coeffs.iter().enumerate() {
                result += coeff * input.powi(i as i32);
            }
            Ok(result)
        }
        NonlinearFunction::PiecewiseLinear {
            breakpoints,
            slopes,
        } => {
            if breakpoints.len() != slopes.len() + 1 {
                return Err(SignalError::ValueError(
                    "Breakpoints and slopes length mismatch".to_string(),
                ));
            }
            for i in 0..breakpoints.len() - 1 {
                if input >= breakpoints[i] && input < breakpoints[i + 1] {
                    let x0 = breakpoints[i];
                    let y0 = if i == 0 {
                        0.0
                    } else {
                        let mut y = 0.0;
                        for j in 0..i {
                            y += slopes[j] * (breakpoints[j + 1] - breakpoints[j]);
                        }
                        y
                    };
                    return Ok(y0 + slopes[i] * (input - x0));
                }
            }
            if input < breakpoints[0] {
                Ok(slopes[0] * (input - breakpoints[0]))
            } else {
                let last_idx = slopes.len() - 1;
                let last_break = breakpoints[last_idx];
                Ok(slopes[last_idx] * (input - last_break))
            }
        }
        NonlinearFunction::Sigmoid { scale, offset } => {
            Ok(1.0 / (1.0 + (-scale * (input - offset)).exp()))
        }
        NonlinearFunction::DeadZone { threshold } => {
            if input.abs() <= *threshold {
                Ok(0.0)
            } else if input > *threshold {
                Ok(input - threshold)
            } else {
                Ok(input + threshold)
            }
        }
        NonlinearFunction::Saturation { lower, upper } => Ok(input.max(*lower).min(*upper)),
        NonlinearFunction::Custom(name) => match name.as_str() {
            "tanh" => Ok(input.tanh()),
            "relu" => Ok(input.max(0.0)),
            "leaky_relu" => Ok(if input > 0.0 { input } else { 0.01 * input }),
            "identity" => Ok(input),
            _ => Err(SignalError::NotImplemented(format!(
                "Custom function '{}' not implemented",
                name
            ))),
        },
    }
}
