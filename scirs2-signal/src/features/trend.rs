use crate::error::SignalResult;
use crate::features::statistical::calculate_std;
use std::collections::HashMap;

/// Extract trend features from a time series
pub fn extract_trend_features(
    signal: &[f64],
    features: &mut HashMap<String, f64>,
) -> SignalResult<()> {
    let n = signal.len();
    let x: Vec<f64> = (0..n).map(|i| i as f64).collect();

    // Calculate linear regression parameters
    let (slope, intercept, r_squared) = linear_regression(&x, signal);

    features.insert("trend_slope".to_string(), slope);
    features.insert("trend_intercept".to_string(), intercept);
    features.insert("trend_r_squared".to_string(), r_squared);

    // Calculate how much the signal deviates from its trend
    let detrended_signal: Vec<f64> = signal
        .iter()
        .zip(x.iter())
        .map(|(&y, &x)| y - (slope * x + intercept))
        .collect();

    let detrended_std = calculate_std(&detrended_signal);
    features.insert("trend_residual_std".to_string(), detrended_std);

    // Estimate non-linearity by calculating the ratio of variance
    // explained by quadratic fit vs. linear fit
    let (_, _, r_squared_quad) = quadratic_regression(&x, signal);

    if r_squared < 1.0 {
        // Non-linearity score is how much better the quadratic fit is
        let non_linearity = (r_squared_quad - r_squared) / (1.0 - r_squared);
        features.insert("non_linearity".to_string(), non_linearity);
    } else {
        features.insert("non_linearity".to_string(), 0.0);
    }

    Ok(())
}

/// Perform linear regression on two vectors
pub fn linear_regression(x: &[f64], y: &[f64]) -> (f64, f64, f64) {
    if x.is_empty() || y.is_empty() || x.len() != y.len() {
        return (0.0, 0.0, 0.0);
    }

    let n = x.len() as f64;

    let sum_x: f64 = x.iter().sum();
    let sum_y: f64 = y.iter().sum();

    let sum_xx: f64 = x.iter().map(|&xi| xi * xi).sum();
    let sum_xy: f64 = x.iter().zip(y.iter()).map(|(&xi, &yi)| xi * yi).sum();

    let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
    let intercept = (sum_y - slope * sum_x) / n;

    // Calculate R-squared
    let mean_y = sum_y / n;

    let ss_total: f64 = y.iter().map(|&yi| (yi - mean_y).powi(2)).sum();
    let ss_residual: f64 = y
        .iter()
        .zip(x.iter())
        .map(|(&yi, &xi)| (yi - (slope * xi + intercept)).powi(2))
        .sum();

    let r_squared = 1.0 - ss_residual / ss_total;

    (slope, intercept, r_squared)
}

/// Perform quadratic regression on two vectors
pub fn quadratic_regression(x: &[f64], y: &[f64]) -> (f64, f64, f64) {
    if x.is_empty() || y.is_empty() || x.len() != y.len() {
        return (0.0, 0.0, 0.0);
    }

    let n = x.len();

    let sum_x: f64 = x.iter().sum();
    let sum_x2: f64 = x.iter().map(|&xi| xi * xi).sum();
    let sum_x3: f64 = x.iter().map(|&xi| xi * xi * xi).sum();
    let sum_x4: f64 = x.iter().map(|&xi| xi * xi * xi * xi).sum();

    let sum_y: f64 = y.iter().sum();
    let sum_xy: f64 = x.iter().zip(y.iter()).map(|(&xi, &yi)| xi * yi).sum();
    let sum_x2y: f64 = x.iter().zip(y.iter()).map(|(&xi, &yi)| xi * xi * yi).sum();

    // Solve the normal equations
    // [n     sum_x   sum_x2] [a]   [sum_y]
    // [sum_x sum_x2  sum_x3] [b] = [sum_xy]
    // [sum_x2 sum_x3 sum_x4] [c]   [sum_x2y]

    let det = n as f64 * sum_x2 * sum_x4 + sum_x * sum_x3 * sum_x2 + sum_x2 * sum_x * sum_x3
        - sum_x2 * sum_x2 * sum_x2
        - sum_x3 * sum_x3 * n as f64
        - sum_x4 * sum_x * sum_x;

    if det.abs() < 1e-10 {
        // Determinant is close to zero, matrix is singular
        return (0.0, 0.0, 0.0);
    }

    let a = (sum_y * sum_x2 * sum_x4 + sum_x * sum_x3 * sum_x2y + sum_xy * sum_x3 * sum_x2
        - sum_x2 * sum_x2 * sum_x2y
        - sum_x3 * sum_x3 * sum_y
        - sum_x4 * sum_xy * sum_x)
        / det;

    let b = (n as f64 * sum_xy * sum_x4 + sum_y * sum_x3 * sum_x2 + sum_x2 * sum_x * sum_x2y
        - sum_x2 * sum_xy * sum_x2
        - sum_x3 * sum_y * sum_x
        - sum_x4 * n as f64 * sum_x2y)
        / det;

    let c = (n as f64 * sum_x2 * sum_x2y + sum_x * sum_xy * sum_x2 + sum_y * sum_x * sum_x3
        - sum_x2 * sum_x * sum_x2y
        - sum_xy * sum_x3 * n as f64
        - sum_x2 * sum_y * sum_x2)
        / det;

    // Calculate R-squared
    let mean_y = sum_y / n as f64;

    let ss_total: f64 = y.iter().map(|&yi| (yi - mean_y).powi(2)).sum();
    let ss_residual: f64 = y
        .iter()
        .zip(x.iter())
        .map(|(&yi, &xi)| (yi - (a + b * xi + c * xi * xi)).powi(2))
        .sum();

    let r_squared = 1.0 - ss_residual / ss_total;

    (b, a, r_squared) // Return in the same format as linear_regression
}
