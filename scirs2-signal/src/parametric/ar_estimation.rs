//! Autoregressive (AR) model estimation methods
//!
//! This module implements various methods for estimating AR model parameters including:
//! - Yule-Walker method (autocorrelation method)
//! - Burg's method (minimizes forward and backward prediction errors)
//! - Covariance method
//! - Modified covariance method
//! - Least squares method
//! - Model order selection using information criteria
//! - AR spectrum computation
//!
//! # Example
//! ```
//! use scirs2_core::ndarray::Array1;
//! use scirs2_signal::parametric::{estimate_ar, ARMethod, burg_method};
//!
//! // Create a signal with spectral peaks
//! let n = 256;
//! let t = Array1::linspace(0.0, 1.0, n);
//! let f1 = 50.0;
//! let f2 = 120.0;
//! let signal = t.mapv(|ti| (2.0 * std::f64::consts::PI * f1 * ti).sin() +
//!                          0.5 * (2.0 * std::f64::consts::PI * f2 * ti).sin());
//!
//! // Estimate AR parameters using Burg's method (order 10)
//! let (ar_coeffs, reflection_coeffs, variance) = burg_method(&signal, 10).unwrap();
//!
//! // Check that we got valid outputs
//! assert_eq!(ar_coeffs.len(), 11); // order + 1 coefficients
//! assert!(variance > 0.0);
//! assert!(reflection_coeffs.is_some());
//! ```

use super::types::{ARMethod, OrderSelection};
use crate::error::{SignalError, SignalResult};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::Complex64;
use scirs2_core::validation::{check_finite, check_positive};
use statrs::statistics::Statistics;
use std::f64::consts::PI;

/// Estimates the autoregressive (AR) parameters of a signal using the specified method
///
/// # Arguments
/// * `signal` - Input signal
/// * `order` - AR model order
/// * `method` - Method to use for AR parameter estimation
///
/// # Returns
/// * `ar_coeffs` - AR coefficients [1, a1, a2, ..., ap]
/// * `reflection_coeffs` - Reflection coefficients (if applicable)
/// * `variance` - Estimated noise variance
///
/// # Example
/// ```
/// use scirs2_core::ndarray::Array1;
/// use scirs2_signal::parametric::{estimate_ar, ARMethod};
///
/// let signal = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0]);
/// let order = 4;
/// let (ar_coeffs, reflection_coeffs, variance) =
///     estimate_ar(&signal, order, ARMethod::Burg).unwrap();
/// ```
pub fn estimate_ar(
    signal: &Array1<f64>,
    order: usize,
    method: ARMethod,
) -> SignalResult<(Array1<f64>, Option<Array1<f64>>, f64)> {
    if order >= signal.len() {
        return Err(SignalError::ValueError(format!(
            "AR model order ({}) must be less than signal length ({})",
            order,
            signal.len()
        )));
    }

    match method {
        ARMethod::YuleWalker => yule_walker(signal, order),
        ARMethod::Burg => burg_method(signal, order),
        ARMethod::Covariance => covariance_method(signal, order),
        ARMethod::ModifiedCovariance => modified_covariance_method(signal, order),
        ARMethod::LeastSquares => least_squares_method(signal, order),
    }
}

/// Estimates AR parameters using the Yule-Walker equations (autocorrelation method)
///
/// # Arguments
/// * `signal` - Input signal
/// * `order` - AR model order
///
/// # Returns
/// * `ar_coeffs` - AR coefficients [1, a1, a2, ..., ap]
/// * `reflection_coeffs` - Reflection coefficients (Levinson-Durbin algorithm byproduct)
/// * `variance` - Estimated noise variance
pub fn yule_walker(
    signal: &Array1<f64>,
    order: usize,
) -> SignalResult<(Array1<f64>, Option<Array1<f64>>, f64)> {
    if order >= signal.len() {
        return Err(SignalError::ValueError(format!(
            "AR model order ({}) must be less than signal length ({})",
            order,
            signal.len()
        )));
    }

    // Calculate autocorrelation up to lag 'order'
    let n = signal.len();
    let mut autocorr = Array1::<f64>::zeros(order + 1);

    for lag in 0..=order {
        let mut sum = 0.0;
        for i in 0..(n - lag) {
            sum += signal[i] * signal[i + lag];
        }
        autocorr[lag] = sum / (n - lag) as f64;
    }

    // Normalize by lag-0 autocorrelation
    let r0 = autocorr[0];
    if r0.abs() < 1e-10 {
        return Err(SignalError::ComputationError(
            "Signal has zero autocorrelation at lag 0".to_string(),
        ));
    }

    // Apply Levinson-Durbin algorithm to solve Yule-Walker equations
    let (ar_coeffs, reflection_coeffs, variance) = levinson_durbin(&autocorr, order)?;

    // Return AR coefficients with a leading 1
    let mut full_ar_coeffs = Array1::<f64>::zeros(order + 1);
    full_ar_coeffs[0] = 1.0;
    for i in 0..order {
        full_ar_coeffs[i + 1] = -ar_coeffs[i]; // Note: Negation of coefficients for standard form
    }

    Ok((full_ar_coeffs, Some(reflection_coeffs), variance))
}

/// Implements the Levinson-Durbin algorithm to solve Toeplitz system of equations
///
/// # Arguments
/// * `autocorr` - Autocorrelation sequence [r0, r1, ..., rp]
/// * `order` - AR model order
///
/// # Returns
/// * `ar_coeffs` - AR coefficients [a1, a2, ..., ap]
/// * `reflection_coeffs` - Reflection coefficients (partial correlation coefficients)
/// * `variance` - Estimated prediction error variance
fn levinson_durbin(
    autocorr: &Array1<f64>,
    order: usize,
) -> SignalResult<(Array1<f64>, Array1<f64>, f64)> {
    let p = order;
    let mut a = Array1::<f64>::zeros(p);
    let mut reflection = Array1::<f64>::zeros(p);

    // Initial error is the zero-lag autocorrelation
    let mut e = autocorr[0];

    for k in 0..p {
        // Compute reflection coefficient
        let mut err = 0.0;
        for j in 0..k {
            err += a[j] * autocorr[k - j];
        }

        let k_reflection = (autocorr[k + 1] - err) / e;
        reflection[k] = k_reflection;

        // Update AR coefficients based on the reflection coefficient
        a[k] = k_reflection;
        if k > 0 {
            let a_prev = a.slice(scirs2_core::ndarray::s![0..k]).to_owned();
            for j in 0..k {
                a[j] = a_prev[j] - k_reflection * a_prev[k - 1 - j];
            }
        }

        // Update prediction error
        e *= 1.0 - k_reflection * k_reflection;

        // Check for algorithm instability
        if e <= 0.0 {
            return Err(SignalError::ComputationError(
                "Levinson-Durbin algorithm became unstable with negative error variance"
                    .to_string(),
            ));
        }
    }

    Ok((a, reflection, e))
}

/// Estimates AR parameters using Burg's method
///
/// Burg's method minimizes the forward and backward prediction errors
/// while maintaining the Levinson recursion.
///
/// # Arguments
/// * `signal` - Input signal
/// * `order` - AR model order
///
/// # Returns
/// * `ar_coeffs` - AR coefficients [1, a1, a2, ..., ap]
/// * `reflection_coeffs` - Reflection coefficients
/// * `variance` - Estimated noise variance
pub fn burg_method(
    signal: &Array1<f64>,
    order: usize,
) -> SignalResult<(Array1<f64>, Option<Array1<f64>>, f64)> {
    if order >= signal.len() {
        return Err(SignalError::ValueError(format!(
            "AR model order ({}) must be less than signal length ({})",
            order,
            signal.len()
        )));
    }

    let n = signal.len();

    // Initialize forward and backward prediction errors
    let mut f = signal.clone();
    let mut b = signal.clone();

    // Initialize AR coefficients and reflection coefficients
    let mut a = Array2::<f64>::eye(order + 1);
    let mut k = Array1::<f64>::zeros(order);

    // Initial prediction error power
    let mut e = signal.iter().map(|&x| x * x).sum::<f64>() / n as f64;

    for m in 0..order {
        // Calculate reflection coefficient
        let mut num = 0.0;
        let mut den = 0.0;

        for i in 0..(n - m - 1) {
            num += f[i + m + 1] * b[i];
            den += f[i + m + 1].powi(2) + b[i].powi(2);
        }

        if den.abs() < 1e-10 {
            return Err(SignalError::ComputationError(
                "Burg algorithm encountered a division by near-zero value".to_string(),
            ));
        }

        let k_m = -2.0 * num / den;
        k[m] = k_m;

        // Update AR coefficients
        for i in 1..=(m + 1) {
            a[[m + 1, i]] = a[[m, i]] + k_m * a[[m, m + 1 - i]];
        }

        // Update prediction error power
        e *= 1.0 - k_m * k_m;

        // Check for algorithm instability
        if e <= 0.0 {
            return Err(SignalError::ComputationError(
                "Burg algorithm became unstable with negative error variance".to_string(),
            ));
        }

        // Update forward and backward prediction errors
        if m < order - 1 {
            for i in 0..(n - m - 1) {
                let f_old = f[i + m + 1];
                let b_old = b[i];

                f[i + m + 1] = f_old + k_m * b_old;
                b[i] = b_old + k_m * f_old;
            }
        }
    }

    // Extract the final AR coefficients and ensure leading coefficient is 1
    let raw_coeffs = a.row(order).to_owned();
    let mut ar_coeffs = Array1::<f64>::zeros(order + 1);
    ar_coeffs[0] = 1.0;
    for i in 1..=order {
        ar_coeffs[i] = raw_coeffs[i];
    }

    Ok((ar_coeffs, Some(k), e))
}

/// Estimates AR parameters using the covariance method
///
/// # Arguments
/// * `signal` - Input signal
/// * `order` - AR model order
///
/// # Returns
/// * `ar_coeffs` - AR coefficients [1, a1, a2, ..., ap]
/// * `reflection_coeffs` - None (not computed in this method)
/// * `variance` - Estimated noise variance
pub fn covariance_method(
    signal: &Array1<f64>,
    order: usize,
) -> SignalResult<(Array1<f64>, Option<Array1<f64>>, f64)> {
    if order >= signal.len() {
        return Err(SignalError::ValueError(format!(
            "AR model order ({}) must be less than signal length ({})",
            order,
            signal.len()
        )));
    }

    let n = signal.len();

    // Form the covariance matrix and vector
    let mut r = Array2::<f64>::zeros((order, order));
    let mut r_vec = Array1::<f64>::zeros(order);

    for i in 0..order {
        for j in 0..order {
            let mut sum = 0.0;
            for k in 0..(n - order) {
                sum += signal[k + i] * signal[k + j];
            }
            r[[i, j]] = sum;
        }

        let mut sum = 0.0;
        for k in 0..(n - order) {
            sum += signal[k + i] * signal[k + order];
        }
        r_vec[i] = sum;
    }

    // Solve the linear system to get AR coefficients
    let ar_params = solve_linear_system(&r, &r_vec)?;

    // Calculate prediction error variance
    let mut variance = 0.0;
    for t in order..n {
        let mut pred = 0.0;
        for i in 0..order {
            pred += ar_params[i] * signal[t - i - 1];
        }
        variance += (signal[t] - pred).powi(2);
    }
    variance /= (n - order) as f64;

    // Create full AR coefficients with leading 1
    let mut full_ar_coeffs = Array1::<f64>::zeros(order + 1);
    full_ar_coeffs[0] = 1.0;
    for i in 0..order {
        full_ar_coeffs[i + 1] = -ar_params[i]; // Note: Negation for standard form
    }

    Ok((full_ar_coeffs, None, variance))
}

/// Estimates AR parameters using the modified covariance method
///
/// The modified covariance method minimizes both forward and backward
/// prediction errors.
///
/// # Arguments
/// * `signal` - Input signal
/// * `order` - AR model order
///
/// # Returns
/// * `ar_coeffs` - AR coefficients [1, a1, a2, ..., ap]
/// * `reflection_coeffs` - None (not computed in this method)
/// * `variance` - Estimated noise variance
pub fn modified_covariance_method(
    signal: &Array1<f64>,
    order: usize,
) -> SignalResult<(Array1<f64>, Option<Array1<f64>>, f64)> {
    if order >= signal.len() {
        return Err(SignalError::ValueError(format!(
            "AR model order ({}) must be less than signal length ({})",
            order,
            signal.len()
        )));
    }

    let n = signal.len();

    // Form the covariance matrix and vector for both forward and backward predictions
    let mut r = Array2::<f64>::zeros((order, order));
    let mut r_vec = Array1::<f64>::zeros(order);

    for i in 0..order {
        for j in 0..order {
            let mut sum_forward = 0.0;
            let mut sum_backward = 0.0;

            for k in 0..(n - order) {
                // Forward prediction error correlation
                sum_forward += signal[k + i] * signal[k + j];

                // Backward prediction error correlation
                sum_backward += signal[n - 1 - k - i] * signal[n - 1 - k - j];
            }

            r[[i, j]] = sum_forward + sum_backward;
        }

        let mut sum_forward = 0.0;
        let mut sum_backward = 0.0;

        for k in 0..(n - order) {
            sum_forward += signal[k + i] * signal[k + order];
            sum_backward += signal[n - 1 - k - i] * signal[n - 1 - k - order];
        }

        r_vec[i] = sum_forward + sum_backward;
    }

    // Solve the linear system to get AR coefficients
    let ar_params = solve_linear_system(&r, &r_vec)?;

    // Calculate prediction error variance
    let mut variance = 0.0;
    let mut count = 0;

    // Forward prediction errors
    for t in order..n {
        let mut pred = 0.0;
        for i in 0..order {
            pred += ar_params[i] * signal[t - i - 1];
        }
        variance += (signal[t] - pred).powi(2);
        count += 1;
    }

    // Backward prediction errors
    for t in 0..(n - order) {
        let mut pred = 0.0;
        for i in 0..order {
            pred += ar_params[i] * signal[n - 1 - t - i - 1];
        }
        variance += (signal[n - 1 - t] - pred).powi(2);
        count += 1;
    }

    variance /= count as f64;

    // Create full AR coefficients with leading 1
    let mut full_ar_coeffs = Array1::<f64>::zeros(order + 1);
    full_ar_coeffs[0] = 1.0;
    for i in 0..order {
        full_ar_coeffs[i + 1] = -ar_params[i]; // Note: Negation for standard form
    }

    Ok((full_ar_coeffs, None, variance))
}

/// Estimates AR parameters using the least squares method
///
/// # Arguments
/// * `signal` - Input signal
/// * `order` - AR model order
///
/// # Returns
/// * `ar_coeffs` - AR coefficients [1, a1, a2, ..., ap]
/// * `reflection_coeffs` - None (not computed in this method)
/// * `variance` - Estimated noise variance
pub fn least_squares_method(
    signal: &Array1<f64>,
    order: usize,
) -> SignalResult<(Array1<f64>, Option<Array1<f64>>, f64)> {
    if order >= signal.len() {
        return Err(SignalError::ValueError(format!(
            "AR model order ({}) must be less than signal length ({})",
            order,
            signal.len()
        )));
    }

    let n = signal.len();

    // Build design matrix for least squares
    let mut x = Array2::<f64>::zeros((n - order, order));
    let mut y = Array1::<f64>::zeros(n - order);

    for i in 0..(n - order) {
        for j in 0..order {
            x[[i, j]] = signal[order - 1 - j + i];
        }
        y[i] = signal[order + i];
    }

    // Solve least squares problem: x^T * x * a = x^T * y
    let xtx = x.t().dot(&x);
    let xty = x.t().dot(&y);

    let ar_params = solve_linear_system(&xtx, &xty)?;

    // Calculate prediction error variance
    let predictions = x.dot(&ar_params);
    let residuals = &y - &predictions;
    let variance = residuals.mapv(|x| x * x).sum() / (n - order) as f64;

    // Create full AR coefficients with leading 1
    let mut full_ar_coeffs = Array1::<f64>::zeros(order + 1);
    full_ar_coeffs[0] = 1.0;
    for i in 0..order {
        full_ar_coeffs[i + 1] = -ar_params[i]; // Note: Negation for standard form
    }

    Ok((full_ar_coeffs, None, variance))
}

/// Solves a linear system Ax = b using LU decomposition
///
/// # Arguments
/// * `a` - Coefficient matrix
/// * `b` - Right-hand side vector
///
/// # Returns
/// * Solution vector x
fn solve_linear_system(a: &Array2<f64>, b: &Array1<f64>) -> SignalResult<Array1<f64>> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(SignalError::ValueError("Matrix must be square".to_string()));
    }
    if b.len() != n {
        return Err(SignalError::ValueError(
            "Vector dimension must match matrix dimension".to_string(),
        ));
    }

    // Simple Gaussian elimination with partial pivoting
    let mut aug = Array2::<f64>::zeros((n, n + 1));

    // Copy matrix and vector into augmented matrix
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
        }
        aug[[i, n]] = b[i];
    }

    // Forward elimination with partial pivoting
    for k in 0..n {
        // Find pivot
        let mut max_row = k;
        for i in (k + 1)..n {
            if aug[[i, k]].abs() > aug[[max_row, k]].abs() {
                max_row = i;
            }
        }

        // Swap rows if needed
        if max_row != k {
            for j in 0..=n {
                let temp = aug[[k, j]];
                aug[[k, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = temp;
            }
        }

        // Check for singular matrix
        if aug[[k, k]].abs() < 1e-14 {
            return Err(SignalError::ComputationError(
                "Matrix is singular or near-singular".to_string(),
            ));
        }

        // Eliminate
        for i in (k + 1)..n {
            let factor = aug[[i, k]] / aug[[k, k]];
            for j in k..=n {
                aug[[i, j]] -= factor * aug[[k, j]];
            }
        }
    }

    // Back substitution
    let mut x = Array1::<f64>::zeros(n);
    for i in (0..n).rev() {
        let mut sum = aug[[i, n]];
        for j in (i + 1)..n {
            sum -= aug[[i, j]] * x[j];
        }
        x[i] = sum / aug[[i, i]];
    }

    Ok(x)
}

/// Calculates the power spectral density (PSD) from AR model coefficients
///
/// # Arguments
/// * `ar_coeffs` - AR coefficients [1, a1, a2, ..., ap]
/// * `variance` - Estimated noise variance
/// * `freqs` - Frequency points at which to evaluate the PSD
/// * `fs` - Sampling frequency
///
/// # Returns
/// * Power spectral density values at the specified frequencies
pub fn ar_spectrum(
    ar_coeffs: &Array1<f64>,
    variance: f64,
    freqs: &Array1<f64>,
    fs: f64,
) -> SignalResult<Array1<f64>> {
    let p = ar_coeffs.len() - 1; // AR order

    // Validate inputs
    if ar_coeffs[0] != 1.0 {
        return Err(SignalError::ValueError(
            "AR coefficients must start with 1.0".to_string(),
        ));
    }

    if variance <= 0.0 {
        return Err(SignalError::ValueError(
            "Variance must be positive".to_string(),
        ));
    }

    // Calculate normalized frequencies
    let norm_freqs = freqs.mapv(|f| f * 2.0 * PI / fs);

    // Calculate PSD for each frequency
    let mut psd = Array1::<f64>::zeros(norm_freqs.len());

    for (i, &w) in norm_freqs.iter().enumerate() {
        // Compute frequency response: H(w) = 1 / A(e^{jw})
        let mut h = Complex64::new(0.0, 0.0);

        for k in 0..=p {
            let phase = -w * k as f64;
            let coeff = ar_coeffs[k];
            h += coeff * Complex64::new(phase.cos(), phase.sin());
        }

        // PSD = variance / |H(w)|^2
        psd[i] = variance / h.norm_sqr();
    }

    Ok(psd)
}

/// Selects the optimal AR model order using an information criterion
///
/// # Arguments
/// * `signal` - Input signal
/// * `maxorder` - Maximum order to consider
/// * `criterion` - Information criterion to use for selection
/// * `ar_method` - Method to use for AR parameter estimation
///
/// # Returns
/// * Optimal order
/// * Criterion values for all tested orders
pub fn select_arorder(
    signal: &Array1<f64>,
    maxorder: usize,
    criterion: OrderSelection,
    ar_method: ARMethod,
) -> SignalResult<(usize, Array1<f64>)> {
    if maxorder >= signal.len() / 2 {
        return Err(SignalError::ValueError(format!(
            "Maximum AR order ({}) should be less than half the signal length ({})",
            maxorder,
            signal.len()
        )));
    }

    let n = signal.len() as f64;
    let mut criteria = Array1::<f64>::zeros(maxorder + 1);

    for order in 0..=maxorder {
        if order == 0 {
            // Special case for order 0: just use the signal variance
            let variance = signal.iter().map(|&x| x * x).sum::<f64>() / n;

            // Compute information criteria based on variance
            match criterion {
                OrderSelection::AIC => criteria[order] = n * variance.ln() + 2.0,
                OrderSelection::BIC => criteria[order] = n * variance.ln() + (0 as f64).ln() * n,
                OrderSelection::FPE => criteria[order] = variance * (n + 1.0) / (n - 1.0),
                OrderSelection::MDL => {
                    criteria[order] = n * variance.ln() + 0.5 * (0 as f64).ln() * n
                }
                OrderSelection::AICc => criteria[order] = n * variance.ln() + 2.0,
            }
        } else {
            // Estimate AR parameters
            let result = estimate_ar(signal, order, ar_method)?;
            let variance = result.2;

            // Compute information criteria based on the method
            match criterion {
                OrderSelection::AIC => {
                    criteria[order] = n * variance.ln() + 2.0 * order as f64;
                }
                OrderSelection::BIC => {
                    criteria[order] = n * variance.ln() + order as f64 * n.ln();
                }
                OrderSelection::FPE => {
                    criteria[order] = variance * (n + order as f64) / (n - order as f64);
                }
                OrderSelection::MDL => {
                    criteria[order] = n * variance.ln() + 0.5 * order as f64 * n.ln();
                }
                OrderSelection::AICc => {
                    // Corrected AIC for small samples
                    criteria[order] =
                        n * variance.ln() + 2.0 * order as f64 * (n / (n - order as f64 - 1.0));
                }
            }
        }
    }

    // Find the order with the minimum criterion value
    let mut min_idx = 0;
    let mut min_val = criteria[0];

    for (i, &val) in criteria.iter().enumerate().skip(1) {
        if val < min_val {
            min_idx = i;
            min_val = val;
        }
    }

    Ok((min_idx, criteria))
}
