use scirs2_core::ndarray::s;
// ARMA (Autoregressive Moving Average) model estimation and analysis
//
// This module implements ARMA model estimation methods including:
// - Maximum likelihood estimation
// - Hannan-Rissanen method
// - Innovation algorithm
// - Spectral analysis for ARMA models

use crate::error::{SignalError, SignalResult};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::Complex64;
use scirs2_core::numeric::Float;
use scirs2_core::validation::check_finite;
use statrs::statistics::Statistics;

#[allow(unused_imports)]
/// ARMA model parameters
#[derive(Debug, Clone)]
pub struct ArmaModel {
    /// AR coefficients [1, a1, a2, ..., ap]
    pub ar_coeffs: Array1<f64>,
    /// MA coefficients [1, b1, b2, ..., bq]
    pub ma_coeffs: Array1<f64>,
    /// Innovation variance
    pub variance: f64,
    /// Log-likelihood (if computed)
    pub log_likelihood: Option<f64>,
}

/// ARMA estimation method
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ArmaMethod {
    /// Hannan-Rissanen two-stage method
    HannanRissanen,
    /// Innovation algorithm
    Innovation,
    /// Maximum likelihood estimation
    MaximumLikelihood,
    /// Conditional sum of squares
    ConditionalSumOfSquares,
    /// Burg's method for AR models
    Burg,
    /// Modified covariance method
    ModifiedCovariance,
    /// Forward-backward linear prediction
    ForwardBackward,
    /// Kalman filter-based estimation
    KalmanFilter,
}

/// Estimate ARMA model parameters
///
/// # Arguments
///
/// * `signal` - Input time series
/// * `p` - AR order
/// * `q` - MA order
/// * `method` - Estimation method
///
/// # Returns
///
/// * ARMA model with estimated parameters
#[allow(dead_code)]
pub fn estimate_arma(
    signal: &Array1<f64>,
    p: usize,
    q: usize,
    method: ArmaMethod,
) -> SignalResult<ArmaModel> {
    let n = signal.len();

    if p + q >= n {
        return Err(SignalError::ValueError(format!(
            "Model order (p={}, q={}) too large for signal length ({})",
            p, q, n
        )));
    }

    // Check for finite values
    for (i, &val) in signal.iter().enumerate() {
        check_finite(val, &format!("signal[{}]", i))?;
    }

    match method {
        ArmaMethod::HannanRissanen => hannan_rissanen(signal, p, q),
        ArmaMethod::Innovation => innovation_algorithm(signal, p, q),
        ArmaMethod::ConditionalSumOfSquares => conditional_sum_of_squares(signal, p, q),
        ArmaMethod::MaximumLikelihood => maximum_likelihood(signal, p, q),
        ArmaMethod::Burg => burg_method(signal, p),
        ArmaMethod::ModifiedCovariance => modified_covariance_method(signal, p),
        ArmaMethod::ForwardBackward => forward_backward_method(signal, p),
        ArmaMethod::KalmanFilter => kalman_filter_estimation(signal, p, q),
    }
}

/// Hannan-Rissanen two-stage method for ARMA estimation
///
/// This is a computationally efficient method that provides good initial estimates
#[allow(dead_code)]
fn hannan_rissanen(signal: &Array1<f64>, p: usize, q: usize) -> SignalResult<ArmaModel> {
    let n = signal.len();

    // Stage 1: Fit a high-order AR model to estimate innovations
    let ar_order = ((n as f64).sqrt() as usize).max(p + q);

    // Use Yule-Walker for AR estimation
    let (ar_coeffs_high, var) = yule_walker_ar(_signal, ar_order)?;

    // Compute residuals (innovations)
    let mut innovations = Array1::zeros(n);
    for t in ar_order..n {
        let mut pred = 0.0;
        for i in 1..=ar_order {
            pred += ar_coeffs_high[i] * signal[t - i];
        }
        innovations[t] = signal[t] - pred;
    }

    // Stage 2: Regression to estimate ARMA parameters
    // Build regression matrix
    let start = (p + q).max(ar_order);
    let n_obs = n - start;

    let mut x = Array2::zeros((n_obs, p + q));
    let mut y = Array1::zeros(n_obs);

    for i in 0..n_obs {
        let t = i + start;
        y[i] = signal[t];

        // AR terms
        for j in 0..p {
            x[[i, j]] = signal[t - j - 1];
        }

        // MA terms (using estimated innovations)
        for j in 0..q {
            x[[i, p + j]] = innovations[t - j - 1];
        }
    }

    // Least squares estimation
    let xt_x = x.t().dot(&x);
    let xt_y = x.t().dot(&y);

    let params = solve_linear_system(&xt_x, &xt_y)?;

    // Extract AR and MA coefficients
    let mut ar_coeffs = Array1::zeros(p + 1);
    ar_coeffs[0] = 1.0;
    for i in 0..p {
        ar_coeffs[i + 1] = -params[i];
    }

    let mut ma_coeffs = Array1::zeros(q + 1);
    ma_coeffs[0] = 1.0;
    for i in 0..q {
        ma_coeffs[i + 1] = params[p + i];
    }

    // Compute final residuals and variance
    let residuals = compute_arma_residuals(_signal, &ar_coeffs, &ma_coeffs)?;
    let variance = residuals.mapv(|r| r * r).sum() / (n - p - q) as f64;

    Ok(ArmaModel {
        ar_coeffs,
        ma_coeffs,
        variance,
        log_likelihood: None,
    })
}

/// Innovation algorithm for ARMA estimation
///
/// Uses recursive prediction error minimization
#[allow(dead_code)]
fn innovation_algorithm(signal: &Array1<f64>, p: usize, q: usize) -> SignalResult<ArmaModel> {
    let n = signal.len();

    // Initialize parameters
    let mut ar_coeffs = Array1::zeros(p + 1);
    ar_coeffs[0] = 1.0;

    let mut ma_coeffs = Array1::zeros(q + 1);
    ma_coeffs[0] = 1.0;

    // Center the _signal
    let mean = signal.mean().unwrap_or(0.0);
    let centered = signal.mapv(|x| x - mean);

    // Innovation algorithm iterations
    let max_iter = 20;
    let tolerance = 1e-6;
    let mut prev_variance = f64::INFINITY;

    for _iter in 0..max_iter {
        // E-step: Compute innovations
        let innovations = compute_innovations(&centered, &ar_coeffs, &ma_coeffs)?;

        // M-step: Update parameters
        // Update AR coefficients
        if p > 0 {
            let mut r_matrix = Array2::zeros((p, p));
            let mut r_vec = Array1::zeros(p);

            for i in 0..p {
                for j in 0..p {
                    let mut sum = 0.0;
                    for t in p.max(q)..n {
                        sum += centered[t - i - 1] * centered[t - j - 1];
                    }
                    r_matrix[[i, j]] = sum;
                }

                let mut sum = 0.0;
                for t in p.max(q)..n {
                    sum += centered[t - i - 1] * centered[t];
                }
                r_vec[i] = sum;
            }

            let ar_params = solve_linear_system(&r_matrix, &r_vec)?;
            for i in 0..p {
                ar_coeffs[i + 1] = -ar_params[i];
            }
        }

        // Update MA coefficients
        if q > 0 {
            let mut m_matrix = Array2::zeros((q, q));
            let mut m_vec = Array1::zeros(q);

            for i in 0..q {
                for j in 0..q {
                    let mut sum = 0.0;
                    for t in p.max(q)..n {
                        if t > i && t > j {
                            sum += innovations[t - i - 1] * innovations[t - j - 1];
                        }
                    }
                    m_matrix[[i, j]] = sum;
                }

                let mut sum = 0.0;
                for t in p.max(q)..n {
                    if t > i {
                        sum += innovations[t - i - 1] * centered[t];
                    }
                }
                m_vec[i] = sum;
            }

            if let Ok(ma_params) = solve_linear_system(&m_matrix, &m_vec) {
                for i in 0..q {
                    ma_coeffs[i + 1] = ma_params[i];
                }
            }
        }

        // Compute variance
        let variance = innovations.mapv(|e| e * e).sum() / (n - p - q) as f64;

        // Check convergence
        if (variance - prev_variance).abs() < tolerance {
            return Ok(ArmaModel {
                ar_coeffs,
                ma_coeffs,
                variance,
                log_likelihood: None,
            });
        }

        prev_variance = variance;
    }

    // Final variance computation
    let residuals = compute_arma_residuals(&centered, &ar_coeffs, &ma_coeffs)?;
    let variance = residuals.mapv(|r| r * r).sum() / (n - p - q) as f64;

    Ok(ArmaModel {
        ar_coeffs,
        ma_coeffs,
        variance,
        log_likelihood: None,
    })
}

/// Conditional sum of squares estimation
#[allow(dead_code)]
fn conditional_sum_of_squares(
    _signal: &Array1<f64>,
    p: usize,
    q: usize,
) -> SignalResult<ArmaModel> {
    let n = signal.len();

    // Initialize with Hannan-Rissanen estimates
    let initial = hannan_rissanen(_signal, p, q)?;
    let mut ar_coeffs = initial.ar_coeffs;
    let mut ma_coeffs = initial.ma_coeffs;

    // Optimization parameters
    let max_iter = 50;
    let tolerance = 1e-6;
    let mut prev_css = f64::INFINITY;

    for _iter in 0..max_iter {
        // Compute conditional residuals
        let residuals = compute_arma_residuals(_signal, &ar_coeffs, &ma_coeffs)?;

        // Compute conditional sum of squares
        let css: f64 = residuals.slice(s![p.max(q)..]).mapv(|r| r * r).sum();

        // Check convergence
        if (css - prev_css).abs() < tolerance {
            break;
        }

        // Update parameters using gradient descent
        let step_size = 0.01;

        // Gradient with respect to AR parameters
        for i in 1..=p {
            let mut grad = 0.0;
            for t in p.max(q)..n {
                let mut deriv = 0.0;
                for j in 0..i {
                    deriv += ar_coeffs[j] * signal[t - j];
                }
                grad += 2.0 * residuals[t] * deriv;
            }
            ar_coeffs[i] -= step_size * grad / n as f64;
        }

        // Gradient with respect to MA parameters
        for i in 1..=q {
            let mut grad = 0.0;
            for t in p.max(q)..n {
                if t >= i {
                    grad += 2.0 * residuals[t] * residuals[t - i];
                }
            }
            ma_coeffs[i] -= step_size * grad / n as f64;
        }

        prev_css = css;
    }

    // Final variance estimate
    let residuals = compute_arma_residuals(_signal, &ar_coeffs, &ma_coeffs)?;
    let variance = residuals.mapv(|r| r * r).sum() / (n - p - q) as f64;

    Ok(ArmaModel {
        ar_coeffs,
        ma_coeffs,
        variance,
        log_likelihood: None,
    })
}

/// Maximum likelihood estimation for ARMA models
#[allow(dead_code)]
fn maximum_likelihood(signal: &Array1<f64>, p: usize, q: usize) -> SignalResult<ArmaModel> {
    // Start with CSS estimates
    let initial = conditional_sum_of_squares(_signal, p, q)?;
    let mut model = initial;

    // Compute log-likelihood
    let n = signal.len();
    let residuals = compute_arma_residuals(_signal, &model.ar_coeffs, &model.ma_coeffs)?;

    // Gaussian log-likelihood
    let log_likelihood = -0.5 * n as f64 * (2.0 * PI * model.variance).ln()
        - 0.5 * residuals.mapv(|r| r * r).sum() / model.variance;

    model.log_likelihood = Some(log_likelihood);

    Ok(model)
}

/// Compute ARMA model residuals (innovations)
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

    for t in 0..n {
        let mut pred = 0.0;

        // AR part
        for i in 1..=p.min(t) {
            pred -= ar_coeffs[i] * signal[t - i];
        }

        // MA part
        for i in 1..=q.min(t) {
            pred += ma_coeffs[i] * residuals[t - i];
        }

        residuals[t] = signal[t] - pred;
    }

    Ok(residuals)
}

/// Compute innovations for given ARMA parameters
#[allow(dead_code)]
fn compute_innovations(
    signal: &Array1<f64>,
    ar_coeffs: &Array1<f64>,
    ma_coeffs: &Array1<f64>,
) -> SignalResult<Array1<f64>> {
    compute_arma_residuals(signal, ar_coeffs, ma_coeffs)
}

/// Yule-Walker method for AR estimation (helper function)
#[allow(dead_code)]
fn yule_walker_ar(
    signal: &Array1<f64>,
    order: usize,
) -> SignalResult<(Array1<f64>, Option<Array1<f64>>, f64)> {
    let n = signal.len();

    // Compute autocorrelation
    let mut r = vec![0.0; order + 1];
    for k in 0..=order {
        let mut sum = 0.0;
        for i in 0..(n - k) {
            sum += signal[i] * signal[i + k];
        }
        r[k] = sum / n as f64;
    }

    // Form Toeplitz matrix
    let mut toeplitz = Array2::zeros((order, order));
    for i in 0..order {
        for j in 0..order {
            toeplitz[[i, j]] = r[(i as i32 - j as i32).abs() as usize];
        }
    }

    // Right-hand side vector
    let mut rhs = Array1::zeros(order);
    for i in 0..order {
        rhs[i] = r[i + 1];
    }

    // Solve for AR parameters
    let ar_params = solve_linear_system(&toeplitz, &rhs)?;

    // Build coefficient array
    let mut ar_coeffs = Array1::zeros(order + 1);
    ar_coeffs[0] = 1.0;
    for i in 0..order {
        ar_coeffs[i + 1] = -ar_params[i];
    }

    // Compute variance
    let mut variance = r[0];
    for i in 1..=order {
        variance += ar_coeffs[i] * r[i];
    }

    Ok((ar_coeffs, None, variance))
}

/// Solve linear system (helper function)
#[allow(dead_code)]
fn solve_linear_system(a: &Array2<f64>, b: &Array1<f64>) -> SignalResult<Array1<f64>> {
    // Simple Gaussian elimination for small systems
    let n = a.nrows();
    let mut aug = a.to_owned();
    let mut rhs = b.to_owned();

    // Forward elimination
    for i in 0..n {
        // Find pivot
        let mut max_val = aug[[i, i]].abs();
        let mut max_row = i;

        for k in (i + 1)..n {
            if aug[[k, i]].abs() > max_val {
                max_val = aug[[k, i]].abs();
                max_row = k;
            }
        }

        if max_val < 1e-10 {
            return Err(SignalError::ComputationError(
                "Singular matrix in linear system".to_string(),
            ));
        }

        // Swap rows
        if max_row != i {
            for j in 0..n {
                let tmp = aug[[i, j]];
                aug[[i, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = tmp;
            }
            let tmp = rhs[i];
            rhs[i] = rhs[max_row];
            rhs[max_row] = tmp;
        }

        // Eliminate column
        for k in (i + 1)..n {
            let factor = aug[[k, i]] / aug[[i, i]];
            for j in i..n {
                aug[[k, j]] -= factor * aug[[i, j]];
            }
            rhs[k] -= factor * rhs[i];
        }
    }

    // Back substitution
    let mut x = Array1::zeros(n);
    for i in (0..n).rev() {
        x[i] = rhs[i];
        for j in (i + 1)..n {
            x[i] -= aug[[i, j]] * x[j];
        }
        x[i] /= aug[[i, i]];
    }

    Ok(x)
}

/// Compute power spectral density for ARMA model
///
/// # Arguments
///
/// * `model` - ARMA model
/// * `frequencies` - Frequencies at which to evaluate PSD
/// * `fs` - Sampling frequency
///
/// # Returns
///
/// * Power spectral density values
#[allow(dead_code)]
pub fn arma_spectrum(
    model: &ArmaModel,
    frequencies: &Array1<f64>,
    fs: f64,
) -> SignalResult<Array1<f64>> {
    let mut psd = Array1::zeros(frequencies.len());

    for (i, &freq) in frequencies.iter().enumerate() {
        let w = 2.0 * PI * freq / fs;

        // Compute AR polynomial at frequency
        let mut ar_poly = Complex64::new(0.0, 0.0);
        for (k, &coeff) in model.ar_coeffs.iter().enumerate() {
            let phase = -(k as f64) * w;
            ar_poly += coeff * Complex64::new(phase.cos(), phase.sin());
        }

        // Compute MA polynomial at frequency
        let mut ma_poly = Complex64::new(0.0, 0.0);
        for (k, &coeff) in model.ma_coeffs.iter().enumerate() {
            let phase = -(k as f64) * w;
            ma_poly += coeff * Complex64::new(phase.cos(), phase.sin());
        }

        // PSD = variance * |MA(w)|^2 / |AR(w)|^2
        psd[i] = model.variance * ma_poly.norm_sqr() / ar_poly.norm_sqr();
    }

    Ok(psd)
}

/// Forecast future values using ARMA model
///
/// # Arguments
///
/// * `model` - Fitted ARMA model
/// * `signal` - Historical signal values
/// * `n_ahead` - Number of steps to forecast
/// * `confidence` - Confidence level for prediction intervals (e.g., 0.95)
///
/// # Returns
///
/// * `forecasts` - Point forecasts
/// * `lower_bounds` - Lower confidence bounds
/// * `upper_bounds` - Upper confidence bounds
#[allow(dead_code)]
pub fn arma_forecast(
    model: &ArmaModel,
    signal: &Array1<f64>,
    n_ahead: usize,
    confidence: Option<f64>,
) -> SignalResult<(Array1<f64>, Option<Array1<f64>>, Option<Array1<f64>>)> {
    let n = signal.len();
    let p = model.ar_coeffs.len() - 1;
    let q = model.ma_coeffs.len() - 1;

    // Compute residuals for the historical data
    let residuals = compute_arma_residuals(signal, &model.ar_coeffs, &model.ma_coeffs)?;

    let mut forecasts = Array1::zeros(n_ahead);
    let mut forecast_errors = Array1::zeros(n_ahead);

    // Extended arrays for recursive forecasting
    let mut extended_signal = signal.to_owned();
    let mut extended_residuals = residuals.to_owned();

    for h in 0..n_ahead {
        let mut forecast = 0.0;

        // AR part
        for i in 1..=p {
            let idx = extended_signal.len() - i;
            if idx < extended_signal.len() {
                forecast -= model.ar_coeffs[i] * extended_signal[idx];
            }
        }

        // MA part (only use past residuals)
        for i in 1..=q.min(extended_residuals.len() - n) {
            let idx = extended_residuals.len() - i;
            if idx < extended_residuals.len() {
                forecast += model.ma_coeffs[i] * extended_residuals[idx];
            }
        }

        forecasts[h] = forecast;

        // Extend arrays
        extended_signal = extended_signal.clone().intoshape(n + h + 1).unwrap();
        extended_signal[n + h] = forecast;

        extended_residuals = extended_residuals.clone().intoshape(n + h + 1).unwrap();
        extended_residuals[n + h] = 0.0; // Future residuals are zero in expectation

        // Forecast error variance (simplified - assumes normal innovations)
        forecast_errors[h] = model.variance.sqrt() * ((1.0 + h as f64 / 10.0) as f64).sqrt();
    }

    // Compute confidence intervals if requested
    let (lower_bounds, upper_bounds) = if let Some(conf) = confidence {
        if conf <= 0.0 || conf >= 1.0 {
            return Err(SignalError::ValueError(
                "Confidence level must be between 0 and 1".to_string(),
            ));
        }

        // Normal quantile for confidence interval
        let z = inverse_normal_cdf((1.0 + conf) / 2.0);

        let lower = &forecasts - z * &forecast_errors;
        let upper = &forecasts + z * &forecast_errors;

        (Some(lower), Some(upper))
    } else {
        (None, None)
    };

    Ok((forecasts, lower_bounds, upper_bounds))
}

/// Approximate inverse normal CDF (for confidence intervals)
#[allow(dead_code)]
fn inverse_normal_cdf(p: f64) -> f64 {
    // Approximation for standard normal quantile
    if p <= 0.0 || p >= 1.0 {
        panic!("Probability must be between 0 and 1");
    }

    // Common quantiles
    match p {
        x if ((x - 0.975) as f64).abs() < 1e-10 => 1.96,
        x if ((x - 0.95) as f64).abs() < 1e-10 => 1.645,
        x if ((x - 0.99) as f64).abs() < 1e-10 => 2.576,
        _ => {
            // Approximate using rational approximation
            let a = [
                2.50662823884,
                -18.61500062529,
                41.39119773534,
                -25.44106049637,
            ];
            let b = [
                -8.47351093090,
                23.08336743743,
                -21.06224101826,
                3.13082909833,
            ];

            let x = p - 0.5;
            let r = x * x;

            let num = ((a[3] * r + a[2]) * r + a[1]) * r + a[0];
            let den = (((b[3] * r + b[2]) * r + b[1]) * r + b[0]) * r + 1.0;

            x * num / den
        }
    }
}

/// Burg's method for AR parameter estimation
///
/// This method provides good estimates for AR models with improved numerical stability
/// compared to the autocorrelation method. It minimizes forward and backward prediction
/// errors simultaneously.
#[allow(dead_code)]
fn burg_method(signal: &Array1<f64>, p: usize) -> SignalResult<ArmaModel> {
    let n = signal.len();

    if p >= n {
        return Err(SignalError::ValueError(format!(
            "AR order p={} must be less than _signal length n={}",
            p, n
        )));
    }

    // Initialize reflection coefficients and prediction errors
    let mut k = vec![0.0; p];
    let mut forward_error = signal.to_vec();
    let mut backward_error = signal.to_vec();
    let mut total_error = signal.mapv(|x| x * x).sum();

    let mut ar_coeffs = vec![0.0; p + 1];
    ar_coeffs[0] = 1.0;

    for m in 0..p {
        // Compute reflection coefficient
        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for i in m..n - 1 {
            numerator += forward_error[i + 1] * backward_error[i];
            denominator +=
                forward_error[i + 1] * forward_error[i + 1] + backward_error[i] * backward_error[i];
        }

        if denominator.abs() < 1e-15 {
            return Err(SignalError::ComputationError(
                "Burg method: denominator near zero".to_string(),
            ));
        }

        k[m] = -2.0 * numerator / denominator;

        // Check stability condition
        if k[m].abs() >= 1.0 {
            return Err(SignalError::ComputationError(format!(
                "Burg method: unstable reflection coefficient k[{}] = {}",
                m, k[m]
            )));
        }

        // Update AR coefficients using Levinson recursion
        let mut new_ar = vec![0.0; m + 2];
        new_ar[0] = 1.0;
        new_ar[m + 1] = k[m];

        for i in 1..=m {
            new_ar[i] = ar_coeffs[i] + k[m] * ar_coeffs[m + 1 - i];
        }

        ar_coeffs = new_ar;

        // Update forward and backward prediction errors
        let mut new_forward = vec![0.0; n];
        let mut new_backward = vec![0.0; n];

        for i in (m + 1)..n {
            new_forward[i] = forward_error[i] + k[m] * backward_error[i - 1];
        }

        for i in 0..(n - m - 1) {
            new_backward[i] = backward_error[i] + k[m] * forward_error[i + 1];
        }

        forward_error = new_forward;
        backward_error = new_backward;

        // Update total prediction error
        total_error *= 1.0 - k[m] * k[m];
    }

    let variance = total_error / (n - p) as f64;

    // Convert to standard AR representation [1, a1, a2, ..., ap]
    let mut final_ar_coeffs = Array1::zeros(p + 1);
    final_ar_coeffs[0] = 1.0;
    for i in 1..=p {
        final_ar_coeffs[i] = -ar_coeffs[i];
    }

    // No MA component for pure AR model
    let ma_coeffs = Array1::ones(1);

    Ok(ArmaModel {
        ar_coeffs: final_ar_coeffs,
        ma_coeffs,
        variance,
        log_likelihood: None,
    })
}

/// Modified covariance method for AR parameter estimation
///
/// This method minimizes the forward and backward prediction errors over a modified
/// covariance matrix, providing improved estimates for short data sequences.
#[allow(dead_code)]
fn modified_covariance_method(signal: &Array1<f64>, p: usize) -> SignalResult<ArmaModel> {
    let n = signal.len();

    if p >= n {
        return Err(SignalError::ValueError(format!(
            "AR order p={} must be less than _signal length n={}",
            p, n
        )));
    }

    // Build modified covariance matrix
    let matrix_size = 2 * (n - p);
    let mut data_matrix = Array2::zeros((matrix_size, p));
    let mut target_vector = Array1::zeros(matrix_size);

    // Forward prediction equations
    for i in 0..(n - p) {
        for j in 0..p {
            data_matrix[[i, j]] = signal[i + j];
        }
        target_vector[i] = -_signal[i + p];
    }

    // Backward prediction equations
    for i in 0..(n - p) {
        for j in 0..p {
            data_matrix[[n - p + i, j]] = signal[n - 1 - i - j];
        }
        target_vector[n - p + i] = -_signal[n - 1 - i - p];
    }

    // Solve normal equations: (X^T X) a = X^T y
    let xtx = data_matrix.t().dot(&data_matrix);
    let xty = data_matrix.t().dot(&target_vector);

    let ar_params = solve_linear_system(&xtx, &xty)?;

    // Build AR coefficient array [1, a1, a2, ..., ap]
    let mut ar_coeffs = Array1::zeros(p + 1);
    ar_coeffs[0] = 1.0;
    for i in 0..p {
        ar_coeffs[i + 1] = ar_params[i];
    }

    // Compute prediction error variance
    let residuals = data_matrix.dot(&ar_params) + target_vector;
    let variance = residuals.mapv(|r| r * r).sum() / matrix_size as f64;

    // No MA component for pure AR model
    let ma_coeffs = Array1::ones(1);

    Ok(ArmaModel {
        ar_coeffs,
        ma_coeffs,
        variance,
        log_likelihood: None,
    })
}

/// Forward-backward linear prediction method
///
/// Combines forward and backward linear prediction for improved AR parameter estimation,
/// particularly effective for signals with non-stationary characteristics.
#[allow(dead_code)]
fn forward_backward_method(signal: &Array1<f64>, p: usize) -> SignalResult<ArmaModel> {
    let n = signal.len();

    if p >= n {
        return Err(SignalError::ValueError(format!(
            "AR order p={} must be less than _signal length n={}",
            p, n
        )));
    }

    // Forward prediction
    let forward_result = solve_forward_prediction(_signal, p)?;

    // Backward prediction
    let backward_result = solve_backward_prediction(_signal, p)?;

    // Combine forward and backward estimates using weighted average
    let forward_weight = 0.5;
    let backward_weight = 0.5;

    let mut combined_ar = Array1::zeros(p + 1);
    combined_ar[0] = 1.0;

    for i in 1..=p {
        combined_ar[i] = forward_weight * forward_result.ar_coeffs[i]
            + backward_weight * backward_result.ar_coeffs[i];
    }

    let combined_variance =
        forward_weight * forward_result.variance + backward_weight * backward_result.variance;

    // No MA component for pure AR model
    let ma_coeffs = Array1::ones(1);

    Ok(ArmaModel {
        ar_coeffs: combined_ar,
        ma_coeffs,
        variance: combined_variance,
        log_likelihood: None,
    })
}

/// Kalman filter-based ARMA estimation
///
/// Uses Kalman filtering approach for robust estimation of ARMA parameters,
/// particularly suitable for non-stationary or noisy data.
#[allow(dead_code)]
fn kalman_filter_estimation(signal: &Array1<f64>, p: usize, q: usize) -> SignalResult<ArmaModel> {
    let n = signal.len();

    if p + q >= n {
        return Err(SignalError::ValueError(format!(
            "Model order (p={}, q={}) too large for _signal length ({})",
            p, q, n
        )));
    }

    // Initialize with Hannan-Rissanen estimates
    let initial = hannan_rissanen(_signal, p, q)?;
    let mut ar_coeffs = initial.ar_coeffs.clone();
    let mut ma_coeffs = initial.ma_coeffs.clone();
    let mut variance = initial.variance;

    // Kalman filter parameters
    let state_dim = p + q;
    if state_dim == 0 {
        return Ok(initial);
    }

    let mut state = Array1::zeros(state_dim);
    let mut state_covariance = Array2::eye(state_dim) * 1000.0; // Large initial uncertainty
    let process_noise = Array2::eye(state_dim) * 0.01;
    let observation_noise = variance;

    // Adaptive estimation using Kalman filtering
    let max_iter = 20;
    let tolerance = 1e-6;
    let mut prev_likelihood = f64::NEG_INFINITY;

    for _iter in 0..max_iter {
        let mut log_likelihood = 0.0;

        // Forward pass
        for t in (p.max(q))..n {
            // Construct observation vector
            let mut observation_vector = Array1::zeros(state_dim);

            // AR terms
            for i in 0..p {
                observation_vector[i] = signal[t - i - 1];
            }

            // MA terms (using current residuals)
            for i in 0..q {
                let residual = if t > i {
                    compute_residual_at_time(_signal, &ar_coeffs, &ma_coeffs, t - i - 1)?
                } else {
                    0.0
                };
                observation_vector[p + i] = residual;
            }

            // Kalman filter prediction step
            let predicted_state = state.clone();
            let predicted_covariance = &state_covariance + &process_noise;

            // Kalman filter update step
            let innovation = signal[t] - observation_vector.dot(&predicted_state);
            let innovation_covariance = observation_vector
                .dot(&predicted_covariance.dot(&observation_vector))
                + observation_noise;

            if innovation_covariance > 1e-15 {
                let kalman_gain =
                    predicted_covariance.dot(&observation_vector) / innovation_covariance;

                state = predicted_state + &kalman_gain * innovation;
                let identity = Array2::eye(state_dim);
                state_covariance = (&identity
                    - &kalman_gain
                        .insert_axis(scirs2_core::ndarray::Axis(1))
                        .dot(&observation_vector.insert_axis(scirs2_core::ndarray::Axis(0))))
                    .dot(&predicted_covariance);

                // Update log-likelihood
                log_likelihood += -0.5
                    * (innovation * innovation / innovation_covariance
                        + innovation_covariance.ln());
            }
        }

        // Update model parameters from state
        for i in 0..p {
            if i + 1 < ar_coeffs.len() {
                ar_coeffs[i + 1] = -state[i];
            }
        }

        for i in 0..q {
            if i + 1 < ma_coeffs.len() {
                ma_coeffs[i + 1] = state[p + i];
            }
        }

        // Check convergence
        if (log_likelihood - prev_likelihood).abs() < tolerance {
            break;
        }
        prev_likelihood = log_likelihood;
    }

    // Final variance estimation
    let residuals = compute_arma_residuals(_signal, &ar_coeffs, &ma_coeffs)?;
    variance = residuals.mapv(|r| r * r).sum() / (n - p - q) as f64;

    Ok(ArmaModel {
        ar_coeffs,
        ma_coeffs,
        variance,
        log_likelihood: Some(prev_likelihood),
    })
}

/// Solve forward prediction problem for AR estimation
#[allow(dead_code)]
fn solve_forward_prediction(signal: &Array1<f64>, p: usize) -> SignalResult<ArmaModel> {
    let n = signal.len();
    let mut data_matrix = Array2::zeros((n - p, p));
    let mut target_vector = Array1::zeros(n - p);

    for i in 0..(n - p) {
        for j in 0..p {
            data_matrix[[i, j]] = signal[i + j];
        }
        target_vector[i] = signal[i + p];
    }

    let xtx = data_matrix.t().dot(&data_matrix);
    let xty = data_matrix.t().dot(&target_vector);

    let ar_params = solve_linear_system(&xtx, &xty)?;

    let mut ar_coeffs = Array1::zeros(p + 1);
    ar_coeffs[0] = 1.0;
    for i in 0..p {
        ar_coeffs[i + 1] = -ar_params[i];
    }

    let residuals = data_matrix.dot(&ar_params) - target_vector;
    let variance = residuals.mapv(|r| r * r).sum() / (n - p) as f64;

    Ok(ArmaModel {
        ar_coeffs,
        ma_coeffs: Array1::ones(1),
        variance,
        log_likelihood: None,
    })
}

/// Solve backward prediction problem for AR estimation
#[allow(dead_code)]
fn solve_backward_prediction(signal: &Array1<f64>, p: usize) -> SignalResult<ArmaModel> {
    let n = signal.len();
    let mut data_matrix = Array2::zeros((n - p, p));
    let mut target_vector = Array1::zeros(n - p);

    for i in 0..(n - p) {
        for j in 0..p {
            data_matrix[[i, j]] = signal[n - 1 - i - j];
        }
        target_vector[i] = signal[n - 1 - i - p];
    }

    let xtx = data_matrix.t().dot(&data_matrix);
    let xty = data_matrix.t().dot(&target_vector);

    let ar_params = solve_linear_system(&xtx, &xty)?;

    let mut ar_coeffs = Array1::zeros(p + 1);
    ar_coeffs[0] = 1.0;
    for i in 0..p {
        ar_coeffs[i + 1] = -ar_params[i];
    }

    let residuals = data_matrix.dot(&ar_params) - target_vector;
    let variance = residuals.mapv(|r| r * r).sum() / (n - p) as f64;

    Ok(ArmaModel {
        ar_coeffs,
        ma_coeffs: Array1::ones(1),
        variance,
        log_likelihood: None,
    })
}

/// Compute residual at a specific time index
#[allow(dead_code)]
fn compute_residual_at_time(
    signal: &Array1<f64>,
    ar_coeffs: &Array1<f64>,
    ma_coeffs: &Array1<f64>,
    t: usize,
) -> SignalResult<f64> {
    let p = ar_coeffs.len() - 1;
    let q = ma_coeffs.len() - 1;

    if t < p.max(q) {
        return Ok(0.0); // Not enough history
    }

    let mut prediction = 0.0;

    // AR contribution
    for i in 1..=p {
        if t >= i {
            prediction += ar_coeffs[i] * signal[t - i];
        }
    }

    // MA contribution (simplified - would need previous residuals)
    // For now, approximate with zero since we don't have residual history

    Ok(signal[t] - prediction)
}
