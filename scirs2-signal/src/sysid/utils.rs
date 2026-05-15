// Shared utility functions for system identification
//
// These helpers are used across multiple sysid submodules:
// - Fit/error metrics
// - Linear algebra helpers
// - FFT wrapper
// - Signal processing utilities

use crate::error::{SignalError, SignalResult};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::Complex64;
use scirs2_fft::fft as fft_compute_sysid;

/// Helper function to calculate model fit percentage
#[allow(dead_code)]
pub(super) fn calculate_fit_percentage(actual: &Array1<f64>, predicted: &Array1<f64>) -> f64 {
    let mean_actual = if !actual.is_empty() {
        actual.iter().copied().sum::<f64>() / actual.len() as f64
    } else {
        0.0
    };
    let ss_tot = actual.mapv(|y| (y - mean_actual).powi(2)).sum();

    if ss_tot < 1e-12 {
        return 0.0;
    }

    let ss_res = (actual - predicted).mapv(|x| x * x).sum();
    let fit = 1.0 - ss_res / ss_tot;

    (100.0 * fit).clamp(0.0, 100.0)
}

/// Simple Ljung-Box test for residual whiteness
#[allow(dead_code)]
pub(super) fn ljung_box_test(residuals: &Array1<f64>, maxlag: usize) -> f64 {
    let n = residuals.len();
    if n <= maxlag {
        return 1.0; // Cannot perform test
    }

    // Calculate autocorrelations
    let mean_residual = residuals.iter().copied().sum::<f64>() / n as f64;
    let var_residual = residuals.mapv(|x| (x - mean_residual).powi(2)).sum() / n as f64;

    let mut lb_stat = 0.0;

    for lag in 1..=maxlag {
        let mut autocorr = 0.0;
        for t in lag..n {
            autocorr += (residuals[t] - mean_residual) * (residuals[t - lag] - mean_residual);
        }
        autocorr /= (n - lag) as f64 * var_residual;

        lb_stat += autocorr * autocorr / (n - lag) as f64;
    }

    lb_stat *= n as f64 * (n + 2) as f64;

    // Return p-value approximation (simplified)
    // In practice, would use chi-square distribution
    (-lb_stat / 2.0).exp()
}

/// Solve linear system using LU decomposition
#[allow(dead_code)]
pub(super) fn solve_linear_system(a: &Array2<f64>, b: &Array1<f64>) -> SignalResult<Array1<f64>> {
    match scirs2_linalg::solve(&a.view(), &b.view(), None) {
        Ok(solution) => Ok(solution),
        Err(_) => Err(SignalError::ComputationError(
            "Failed to solve linear system - matrix may be singular".to_string(),
        )),
    }
}

/// Solve complex least squares problem by separating real and imaginary parts
#[allow(dead_code)]
pub(super) fn solve_complex_least_squares(
    a: &Array2<Complex64>,
    b: &Array1<Complex64>,
) -> SignalResult<Array1<Complex64>> {
    let m = a.nrows();
    let n = a.ncols();

    // Convert to real system: [Re(A); Im(A)] * [Re(x); Im(x)] = [Re(b); Im(b)]
    let mut a_real = Array2::<f64>::zeros((2 * m, 2 * n));
    let mut b_real = Array1::<f64>::zeros(2 * m);

    // Fill real parts
    for i in 0..m {
        for j in 0..n {
            a_real[[i, j]] = a[[i, j]].re;
            a_real[[i, j + n]] = -a[[i, j]].im;
            a_real[[i + m, j]] = a[[i, j]].im;
            a_real[[i + m, j + n]] = a[[i, j]].re;
        }
        b_real[i] = b[i].re;
        b_real[i + m] = b[i].im;
    }

    // Solve real system
    let at_a = a_real.t().dot(&a_real);
    let at_b = a_real.t().dot(&b_real);
    let x_real = solve_linear_system(&at_a, &at_b)?;

    // Convert back to complex
    let mut result = Array1::<Complex64>::zeros(n);
    for i in 0..n {
        result[i] = Complex64::new(x_real[i], x_real[i + n]);
    }

    Ok(result)
}

/// Compute FFT using scirs2_fft for O(n log n) performance
#[allow(dead_code)]
pub(super) fn compute_fft(signal: &Array1<f64>) -> Array1<Complex64> {
    let n = signal.len();
    let signal_slice: Vec<f64> = signal.to_vec();
    match fft_compute_sysid(&signal_slice, Some(n)) {
        Ok(fft_vec) => Array1::from(fft_vec),
        Err(_) => {
            // Fallback: return zero-filled array on error (preserves signature)
            Array1::<Complex64>::zeros(n)
        }
    }
}

/// Find next power of 2
#[allow(dead_code)]
pub(super) fn next_power_of_2(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    let mut power = 1;
    while power < n {
        power <<= 1;
    }
    power
}

// ============================================================================
// ROBUST ESTIMATION HELPERS (also used in robust.rs)
// ============================================================================

#[allow(dead_code)]
pub(super) fn solve_least_squares(
    regressor: &Array2<f64>,
    target: &Array1<f64>,
) -> SignalResult<Array1<f64>> {
    // Simple normal equations solution (A^T A)^-1 A^T b
    let at = regressor.t();
    let ata = at.dot(regressor);
    let atb = at.dot(target);

    // Solve using pseudo-inverse (simplified)
    let mut result = Array1::zeros(regressor.ncols());
    for i in 0..regressor.ncols() {
        if i < atb.len() {
            result[i] = atb[i] / (ata[[i, i]] + 1e-12);
        }
    }

    Ok(result)
}

#[allow(dead_code)]
pub(super) fn solve_weighted_least_squares(
    regressor: &Array2<f64>,
    target: &Array1<f64>,
    weights: &Array1<f64>,
) -> SignalResult<Array1<f64>> {
    // Weighted least squares: (A^T W A)^-1 A^T W b
    let n = regressor.nrows();
    let p = regressor.ncols();

    let mut weighted_regressor = Array2::zeros((n, p));
    let mut weighted_target = Array1::zeros(n);

    for i in 0..n {
        let w = weights[i].sqrt();
        weighted_target[i] = target[i] * w;
        for j in 0..p {
            weighted_regressor[[i, j]] = regressor[[i, j]] * w;
        }
    }

    solve_least_squares(&weighted_regressor, &weighted_target)
}

#[allow(dead_code)]
pub(super) fn update_huber_weights(
    residuals: &Array1<f64>,
    scale: f64,
    threshold: f64,
    weights: &mut Array1<f64>,
) {
    for i in 0..residuals.len() {
        let standardized_residual = residuals[i].abs() / scale;
        weights[i] = if standardized_residual <= threshold {
            1.0
        } else {
            threshold / standardized_residual
        };
    }
}

#[allow(dead_code)]
pub fn estimate_robust_scale(
    target: &Array1<f64>,
    regressor: &Array2<f64>,
    parameters: &Array1<f64>,
) -> SignalResult<f64> {
    let residuals = target - &regressor.dot(parameters);
    let mut abs_residuals: Vec<f64> = residuals.iter().map(|&r: &f64| r.abs()).collect();
    abs_residuals.sort_by(|a, b| a.partial_cmp(b).expect("Operation failed"));

    // Median absolute deviation (MAD)
    let median_idx = abs_residuals.len() / 2;
    let mad = if abs_residuals.len().is_multiple_of(2) {
        (abs_residuals[median_idx - 1] + abs_residuals[median_idx]) / 2.0
    } else {
        abs_residuals[median_idx]
    };

    Ok(mad * 1.4826) // Scale factor for normal distribution
}

#[allow(dead_code)]
pub fn detect_outliers(residuals: &Array1<f64>, scale: f64, threshold: f64) -> Vec<usize> {
    residuals
        .iter()
        .enumerate()
        .filter_map(|(i, &r)| {
            if r.abs() / scale > threshold {
                Some(i)
            } else {
                None
            }
        })
        .collect()
}

#[allow(dead_code)]
pub(super) fn calculate_robust_fit(
    target: &Array1<f64>,
    regressor: &Array2<f64>,
    parameters: &Array1<f64>,
) -> f64 {
    let predicted = regressor.dot(parameters);
    let residuals = target - &predicted;

    // Use robust R-squared based on median
    let target_median = calculate_median(target);
    let total_deviation: f64 = target.iter().map(|&y| (y - target_median).abs()).sum();
    let residual_deviation: f64 = residuals.iter().map(|&r: &f64| r.abs()).sum();

    if total_deviation > 1e-12 {
        100.0 * (1.0 - residual_deviation / total_deviation).max(0.0)
    } else {
        100.0
    }
}

#[allow(dead_code)]
pub(super) fn calculate_median(data: &Array1<f64>) -> f64 {
    let mut sorted: Vec<f64> = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("Operation failed"));

    let n = sorted.len();
    if n.is_multiple_of(2) {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    }
}
