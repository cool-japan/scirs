//! Utility functions for parametric spectral estimation
//!
//! This module contains helper functions for spectral analysis, diagnostics,
//! and mathematical computations used throughout the parametric estimation framework.

use super::types::*;
use crate::error::{SignalError, SignalResult};
// Remove unused import - compute_eigendecomposition is used directly in high_resolution module
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::Complex64;
use statrs::statistics::Statistics;
use std::f64::consts::PI;

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
pub fn compute_ar_psd(
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
pub fn compute_arma_psd(
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

        if ar_magnitude_squared < 1e-12 {
            return Err(SignalError::ComputationError(
                "AR polynomial has near-zero magnitude".to_string(),
            ));
        }

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
pub fn generate_frequency_grid(n_freqs: usize, fs: f64) -> Vec<f64> {
    let nyquist = fs / 2.0;
    (0..n_freqs)
        .map(|i| i as f64 * nyquist / (n_freqs - 1) as f64)
        .collect()
}

/// Compute AR residuals from signal and AR coefficients
pub fn compute_ar_residuals(
    signal: &Array1<f64>,
    ar_coeffs: &Array1<f64>,
) -> SignalResult<Array1<f64>> {
    let n = signal.len();
    let ar_order = ar_coeffs.len() - 1;
    let mut residuals = Array1::zeros(n);

    for t in ar_order..n {
        let mut prediction = 0.0;
        for i in 1..=ar_order {
            prediction += ar_coeffs[i] * signal[t - i];
        }
        residuals[t] = signal[t] - prediction;
    }

    Ok(residuals)
}

/// Compute ARMA residuals from signal and coefficients
pub fn compute_arma_residuals(
    signal: &[f64],
    ar_coeffs: &Array1<f64>,
    ma_coeffs: &Array1<f64>,
) -> SignalResult<Array1<f64>> {
    let n = signal.len();
    let ar_order = ar_coeffs.len() - 1;
    let ma_order = ma_coeffs.len() - 1;

    let mut residuals = Array1::zeros(n);
    let mut innovations = Array1::zeros(n);

    for t in ar_order.max(ma_order)..n {
        // AR component
        let mut ar_prediction = 0.0;
        for i in 1..=ar_order {
            if t >= i {
                ar_prediction += ar_coeffs[i] * signal[t - i];
            }
        }

        // MA component
        let mut ma_prediction = 0.0;
        for i in 1..=ma_order {
            if t >= i {
                ma_prediction += ma_coeffs[i] * innovations[t - i];
            }
        }

        innovations[t] = signal[t] - ar_prediction - ma_prediction;
        residuals[t] = innovations[t];
    }

    Ok(residuals)
}

/// Compute comprehensive model diagnostics
pub fn compute_comprehensive_diagnostics(
    signal: &Array1<f64>,
    ar_coeffs: &Array1<f64>,
    ma_coeffs: &Array1<f64>,
    residuals: &Array1<f64>,
    noise_variance: f64,
) -> SignalResult<ModelDiagnostics> {
    let n = signal.len() as f64;
    let p = ar_coeffs.len() - 1; // AR order
    let q = ma_coeffs.len() - 1; // MA order

    // Check model stability (AR roots inside unit circle)
    let is_stable = check_ar_stability(ar_coeffs)?;

    // Estimate condition number
    let condition_number = estimate_condition_number(ar_coeffs, ma_coeffs)?;

    // Compute log-likelihood
    let log_likelihood = -0.5 * n * (noise_variance.ln() + 2.0 * PI.ln())
        - 0.5 * residuals.mapv(|r| r * r).sum() / noise_variance;

    // Information criteria
    let aic = -2.0 * log_likelihood + 2.0 * (p + q) as f64;
    let bic = -2.0 * log_likelihood + (p + q) as f64 * n.ln();

    // Prediction error variance (simplified)
    let prediction_error_variance = residuals.var(0.0);

    // Ljung-Box test p-value (simplified)
    let ljung_box_p_value = compute_ljung_box_test(residuals, (n / 10.0) as usize);

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
pub fn compute_basic_diagnostics(
    ar_coeffs: &Array1<f64>,
    ma_coeffs: &Array1<f64>,
    noise_variance: f64,
) -> SignalResult<ModelDiagnostics> {
    let p = ar_coeffs.len() - 1;
    let q = ma_coeffs.len() - 1;

    let is_stable = check_ar_stability(ar_coeffs)?;
    let condition_number = estimate_condition_number(ar_coeffs, ma_coeffs)?;

    // Simplified criteria (no data available)
    let log_likelihood = -100.0; // Placeholder
    let aic = 2.0 * (p + q) as f64;
    let bic = (p + q) as f64 * 100.0_f64.ln();

    Ok(ModelDiagnostics {
        is_stable,
        condition_number,
        aic,
        bic,
        log_likelihood,
        prediction_error_variance: noise_variance,
        ljung_box_p_value: None,
    })
}

/// Check AR model stability by examining roots
fn check_ar_stability(ar_coeffs: &Array1<f64>) -> SignalResult<bool> {
    if ar_coeffs.len() <= 1 {
        return Ok(true);
    }

    // For simplicity, check if any AR coefficient has magnitude >= 1
    // In practice, we would find polynomial roots and check unit circle
    for &coeff in ar_coeffs.iter().skip(1) {
        if coeff.abs() >= 0.99 {
            return Ok(false);
        }
    }

    Ok(true)
}

/// Estimate condition number of coefficient matrix
pub fn estimate_condition_number(
    ar_coeffs: &Array1<f64>,
    _ma_coeffs: &Array1<f64>,
) -> SignalResult<f64> {
    // Simplified condition number estimate based on AR coefficients
    let max_coeff = ar_coeffs
        .iter()
        .skip(1)
        .fold(0.0f64, |acc, &x| acc.max(x.abs()));
    let min_coeff = ar_coeffs
        .iter()
        .skip(1)
        .fold(1.0f64, |acc, &x| acc.min(x.abs()));

    if min_coeff > 0.0 {
        Ok(max_coeff / min_coeff)
    } else {
        Ok(1e6) // Large condition number for singular case
    }
}

/// Compute Ljung-Box test p-value (simplified)
fn compute_ljung_box_test(_residuals: &Array1<f64>, _maxlag: usize) -> Option<f64> {
    // Simplified implementation - return None for now
    // In practice, this would compute the Ljung-Box Q statistic
    // and return the corresponding p-value from chi-squared distribution
    None
}

/// Estimate memory usage for ARMA computation
pub fn estimate_memory_usage(n: usize, ar_order: usize, ma_order: usize) -> f64 {
    // Estimate in MB based on array sizes
    let signal_size = n * 8; // f64 bytes
    let ar_size = (ar_order + 1) * 8;
    let ma_size = (ma_order + 1) * 8;
    let residuals_size = n * 8;
    let working_memory = n * 8 * 4; // Temporary arrays

    let total_bytes = signal_size + ar_size + ma_size + residuals_size + working_memory;
    total_bytes as f64 / (1024.0 * 1024.0)
}

/// Compute autocorrelation function
pub fn compute_autocorrelation(signal: &Array1<f64>, maxlag: usize) -> SignalResult<Array1<f64>> {
    let n = signal.len();
    let mut autocorr = Array1::zeros(maxlag + 1);

    let mean = signal.mean().unwrap_or(0.0);
    let variance = signal.var(0.0);

    if variance <= 0.0 {
        return Ok(autocorr);
    }

    for lag in 0..=maxlag {
        if lag < n {
            let mut sum = 0.0;
            let mut count = 0;

            for i in lag..n {
                sum += (signal[i] - mean) * (signal[i - lag] - mean);
                count += 1;
            }

            if count > 0 {
                autocorr[lag] = sum / (count as f64 * variance);
            }
        }
    }

    Ok(autocorr)
}

/// Create autocorrelation matrix (Toeplitz)
pub fn create_autocorrelation_matrix(autocorr: &Array1<f64>, order: usize) -> Array2<f64> {
    let mut matrix = Array2::zeros((order, order));

    for i in 0..order {
        for j in 0..order {
            let lag = i.abs_diff(j);
            if lag < autocorr.len() {
                matrix[(i, j)] = autocorr[lag];
            }
        }
    }

    matrix
}

/// Solve Yule-Walker equations
pub fn solve_yule_walker(
    autocorr_matrix: &Array2<f64>,
    autocorr_vector: &Array1<f64>,
) -> SignalResult<Array1<f64>> {
    // Simplified implementation using basic linear algebra
    // In practice, would use specialized Toeplitz solvers like Levinson-Durbin

    if autocorr_matrix.nrows() != autocorr_matrix.ncols() {
        return Err(SignalError::ValueError(
            "Autocorrelation matrix must be square".to_string(),
        ));
    }

    if autocorr_matrix.nrows() != autocorr_vector.len() {
        return Err(SignalError::ValueError(
            "Matrix and vector dimensions must match".to_string(),
        ));
    }

    // Simple Gaussian elimination (not optimized for Toeplitz structure)
    let n = autocorr_matrix.nrows();
    let mut a = autocorr_matrix.clone();
    let mut b = autocorr_vector.clone();

    // Forward elimination
    for i in 0..n {
        // Find pivot
        let mut max_row = i;
        for k in i + 1..n {
            if a[(k, i)].abs() > a[(max_row, i)].abs() {
                max_row = k;
            }
        }

        // Swap rows
        if max_row != i {
            for j in 0..n {
                let temp = a[(i, j)];
                a[(i, j)] = a[(max_row, j)];
                a[(max_row, j)] = temp;
            }
            let temp = b[i];
            b[i] = b[max_row];
            b[max_row] = temp;
        }

        // Check for singular matrix
        if a[(i, i)].abs() < 1e-12 {
            return Err(SignalError::ComputationError(
                "Singular autocorrelation matrix".to_string(),
            ));
        }

        // Eliminate column
        for k in i + 1..n {
            let factor = a[(k, i)] / a[(i, i)];
            for j in i..n {
                a[(k, j)] -= factor * a[(i, j)];
            }
            b[k] -= factor * b[i];
        }
    }

    // Back substitution
    let mut x = Array1::zeros(n);
    for i in (0..n).rev() {
        x[i] = b[i];
        for j in i + 1..n {
            x[i] -= a[(i, j)] * x[j];
        }
        x[i] /= a[(i, i)];
    }

    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_ar_psd() {
        let ar_coeffs = Array1::from_vec(vec![1.0, -0.5, 0.2]);
        let noise_variance = 1.0;
        let frequencies = vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5];

        let result = compute_ar_psd(&ar_coeffs, noise_variance, &frequencies);
        assert!(result.is_ok());

        let psd = result.unwrap();
        assert_eq!(psd.len(), frequencies.len());
        assert!(psd.iter().all(|&x| x > 0.0 && x.is_finite()));
    }

    #[test]
    fn test_compute_arma_psd() {
        let ar_coeffs = Array1::from_vec(vec![1.0, -0.5]);
        let ma_coeffs = Array1::from_vec(vec![1.0, 0.3]);
        let noise_variance = 1.0;
        let frequencies = vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5];

        let result = compute_arma_psd(&ar_coeffs, &ma_coeffs, noise_variance, &frequencies);
        assert!(result.is_ok());

        let psd = result.unwrap();
        assert_eq!(psd.len(), frequencies.len());
        assert!(psd.iter().all(|&x| x > 0.0 && x.is_finite()));
    }

    #[test]
    fn test_generate_frequency_grid() {
        let n_freqs = 10;
        let fs = 2.0;

        let frequencies = generate_frequency_grid(n_freqs, fs);
        assert_eq!(frequencies.len(), n_freqs);
        assert_eq!(frequencies[0], 0.0);
        assert_eq!(frequencies[n_freqs - 1], 1.0); // Nyquist frequency
    }

    #[test]
    fn test_compute_ar_residuals() {
        let signal = Array1::from_vec(vec![1.0, 2.0, 1.5, 2.5, 1.8, 2.2]);
        let ar_coeffs = Array1::from_vec(vec![1.0, -0.5]);

        let result = compute_ar_residuals(&signal, &ar_coeffs);
        assert!(result.is_ok());

        let residuals = result.unwrap();
        assert_eq!(residuals.len(), signal.len());
    }

    #[test]
    fn test_compute_autocorrelation() {
        let signal = Array1::from_vec(vec![1.0, 2.0, 1.5, 2.5, 1.8, 2.2, 1.7, 2.3]);
        let maxlag = 3;

        let result = compute_autocorrelation(&signal, maxlag);
        assert!(result.is_ok());

        let autocorr = result.unwrap();
        assert_eq!(autocorr.len(), maxlag + 1);
        assert!((autocorr[0] - 1.0).abs() < 1e-10); // R[0] should be 1 (normalized)
    }
}
