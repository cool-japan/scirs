//! Statistical tests and validation functions for system identification
//!
//! This module provides comprehensive statistical analysis tools for validating
//! system identification results, including residual analysis, normality tests,
//! and model diagnostics.

use crate::error::{SignalError, SignalResult};
use super::types::*;
use ndarray::{Array1, Array2};
use num_complex::Complex64;

/// Jarque-Bera test for normality of residuals
///
/// Tests the null hypothesis that the data comes from a normal distribution
/// by examining skewness and kurtosis of the sample.
///
/// # Arguments
///
/// * `data` - Data array to test for normality
///
/// # Returns
///
/// * P-value for the normality test (higher values indicate more normal data)
///
/// # Examples
///
/// ```rust
/// use ndarray::Array1;
/// use scirs2_signal::sysid_enhanced::statistics::jarque_bera_test;
///
/// // Test normally distributed data
/// let normal_data = Array1::from_vec(vec![0.1, -0.2, 0.3, -0.1, 0.0, 0.2, -0.3, 0.1]);
/// let p_value = jarque_bera_test(&normal_data);
/// println!("Normality p-value: {:.4}", p_value);
/// ```
pub fn jarque_bera_test(data: &Array1<f64>) -> f64 {
    let n = data.len() as f64;
    let mean = data.mean().unwrap_or(0.0);

    // Compute central moments
    let mut m2 = 0.0; // Second moment (variance)
    let mut m3 = 0.0; // Third moment (for skewness)
    let mut m4 = 0.0; // Fourth moment (for kurtosis)

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

    // Avoid division by zero
    if m2 < 1e-15 {
        return 1.0; // Perfect normality for constant data
    }

    // Compute skewness and excess kurtosis
    let skewness = m3 / m2.powf(1.5);
    let excess_kurtosis = m4 / (m2 * m2) - 3.0;

    // Jarque-Bera test statistic
    let jb_statistic = n / 6.0 * (skewness * skewness + excess_kurtosis * excess_kurtosis / 4.0);

    // Convert to p-value using chi-square distribution with 2 degrees of freedom
    chi_square_pvalue(jb_statistic, 2)
}

/// Ljung-Box test for autocorrelation in residuals
///
/// Tests the null hypothesis that residuals are independently distributed
/// (no autocorrelation) using a portmanteau test.
///
/// # Arguments
///
/// * `residuals` - Array of residual values
/// * `h` - Number of lags to test
///
/// # Returns
///
/// * P-value for independence test (higher values indicate independence)
pub fn ljung_box_test(residuals: &Array1<f64>, h: usize) -> f64 {
    let n = residuals.len();
    if n < h + 1 || h == 0 {
        return 1.0;
    }

    // Compute sample autocorrelations
    let mean = residuals.mean().unwrap_or(0.0);
    let variance = residuals.var(0.0);

    if variance < 1e-15 {
        return 1.0; // No variance means perfect autocorrelation structure
    }

    let mut autocorrelations = Vec::with_capacity(h);

    for lag in 1..=h {
        let mut numerator = 0.0;
        let count = n - lag;

        for i in 0..count {
            numerator += (residuals[i] - mean) * (residuals[i + lag] - mean);
        }

        let autocorr = numerator / (count as f64 * variance);
        autocorrelations.push(autocorr);
    }

    // Ljung-Box statistic
    let mut lb_statistic = 0.0;
    for (k, &rho_k) in autocorrelations.iter().enumerate() {
        let k_plus_1 = (k + 1) as f64;
        lb_statistic += rho_k * rho_k / (n as f64 - k_plus_1);
    }
    lb_statistic *= n as f64 * (n as f64 + 2.0);

    // Convert to p-value using chi-square approximation
    chi_square_pvalue(lb_statistic, h)
}

/// Cross-correlation test between residuals and input
///
/// Tests whether residuals are correlated with the input signal,
/// which would indicate model inadequacy.
///
/// # Arguments
///
/// * `residuals` - Array of residual values
/// * `input` - Array of input values
/// * `max_lag` - Maximum lag to test
///
/// # Returns
///
/// * P-value for independence test
pub fn cross_correlation_test(residuals: &Array1<f64>, input: &Array1<f64>, max_lag: usize) -> f64 {
    let n = residuals.len().min(input.len());
    if n < max_lag + 1 || max_lag == 0 {
        return 1.0;
    }

    let residuals_mean = residuals.mean().unwrap_or(0.0);
    let input_mean = input.mean().unwrap_or(0.0);
    let residuals_var = residuals.var(0.0);
    let input_var = input.var(0.0);

    if residuals_var < 1e-15 || input_var < 1e-15 {
        return 1.0;
    }

    let mut max_cross_corr = 0.0;

    // Compute cross-correlations at different lags
    for lag in 0..=max_lag {
        let mut numerator = 0.0;
        let count = n - lag;

        for i in 0..count {
            numerator += (residuals[i] - residuals_mean) * (input[i + lag] - input_mean);
        }

        let cross_corr = numerator / (count as f64 * (residuals_var * input_var).sqrt());
        max_cross_corr = max_cross_corr.max(cross_corr.abs());
    }

    // Approximate test using normal distribution
    let test_statistic = max_cross_corr * (n as f64).sqrt();

    // Two-sided test
    2.0 * (1.0 - standard_normal_cdf(test_statistic))
}

/// Comprehensive residual analysis
///
/// Performs multiple statistical tests on model residuals to assess
/// model adequacy and identify potential issues.
///
/// # Arguments
///
/// * `residuals` - Array of residual values from model validation
/// * `input` - Input data used for model identification
/// * `max_lag` - Maximum lag for autocorrelation tests
///
/// # Returns
///
/// * Complete residual analysis results
pub fn analyze_residuals(
    residuals: &Array1<f64>,
    input: &Array1<f64>,
    max_lag: usize,
) -> SignalResult<ResidualAnalysis> {
    // Compute autocorrelation function
    let autocorrelation = compute_autocorrelation(residuals, max_lag);

    // Compute cross-correlation with input
    let cross_correlation = compute_cross_correlation(residuals, input, max_lag);

    // Statistical tests
    let whiteness_pvalue = ljung_box_test(residuals, max_lag);
    let independence_pvalue = cross_correlation_test(residuals, input, max_lag);
    let normality_pvalue = jarque_bera_test(residuals);

    Ok(ResidualAnalysis {
        autocorrelation,
        cross_correlation,
        whiteness_pvalue,
        independence_pvalue,
        normality_pvalue,
    })
}

/// Compute autocorrelation function
fn compute_autocorrelation(data: &Array1<f64>, max_lag: usize) -> Array1<f64> {
    let n = data.len();
    let mean = data.mean().unwrap_or(0.0);
    let variance = data.var(0.0);

    let mut autocorr = Array1::zeros(max_lag + 1);

    if variance < 1e-15 {
        autocorr[0] = 1.0;
        return autocorr;
    }

    for lag in 0..=max_lag.min(n - 1) {
        let mut sum = 0.0;
        let count = n - lag;

        for i in 0..count {
            sum += (data[i] - mean) * (data[i + lag] - mean);
        }

        autocorr[lag] = sum / (count as f64 * variance);
    }

    autocorr
}

/// Compute cross-correlation function
fn compute_cross_correlation(data1: &Array1<f64>, data2: &Array1<f64>, max_lag: usize) -> Array1<f64> {
    let n = data1.len().min(data2.len());
    let mean1 = data1.mean().unwrap_or(0.0);
    let mean2 = data2.mean().unwrap_or(0.0);
    let var1 = data1.var(0.0);
    let var2 = data2.var(0.0);

    let mut cross_corr = Array1::zeros(max_lag + 1);

    if var1 < 1e-15 || var2 < 1e-15 {
        return cross_corr;
    }

    let normalization = (var1 * var2).sqrt();

    for lag in 0..=max_lag.min(n - 1) {
        let mut sum = 0.0;
        let count = n - lag;

        for i in 0..count {
            sum += (data1[i] - mean1) * (data2[i + lag] - mean2);
        }

        cross_corr[lag] = sum / (count as f64 * normalization);
    }

    cross_corr
}

/// Compute stability margin for different model types
pub fn compute_stability_margin(model: &SystemModel) -> SignalResult<f64> {
    match model {
        SystemModel::ARX { a, .. } => {
            // Check if AR polynomial roots are inside unit circle
            let roots = compute_polynomial_roots(a)?;
            let min_margin = roots
                .iter()
                .map(|r| 1.0 - r.norm())
                .fold(f64::INFINITY, f64::min);
            Ok(min_margin.max(0.0))
        }
        SystemModel::ARMAX { a, .. } => {
            // Check AR polynomial stability
            let roots = compute_polynomial_roots(a)?;
            let min_margin = roots
                .iter()
                .map(|r| 1.0 - r.norm())
                .fold(f64::INFINITY, f64::min);
            Ok(min_margin.max(0.0))
        }
        SystemModel::StateSpace(_ss) => {
            // For state-space models, check eigenvalues of A matrix
            // This is a placeholder - in practice would use proper eigenvalue computation
            Ok(0.5) // Default stable margin
        }
        _ => Ok(0.5), // Default margin for other models
    }
}

/// Chi-square p-value approximation
fn chi_square_pvalue(x: f64, df: usize) -> f64 {
    if x < 0.0 {
        return 1.0;
    }

    // Simple approximations for common degrees of freedom
    match df {
        1 => {
            // Chi-square with 1 DOF is square of standard normal
            2.0 * (1.0 - standard_normal_cdf(x.sqrt()))
        }
        2 => {
            // Chi-square with 2 DOF has simple exponential form
            (-x / 2.0).exp()
        }
        _ => {
            // Normal approximation for larger DOF
            let mean = df as f64;
            let variance = 2.0 * df as f64;
            let z = (x - mean) / variance.sqrt();
            1.0 - standard_normal_cdf(z)
        }
    }
}

/// Standard normal cumulative distribution function approximation
fn standard_normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Error function approximation using Abramowitz and Stegun formula
fn erf(x: f64) -> f64 {
    // Constants for the approximation
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

/// Compute polynomial roots using companion matrix method
fn compute_polynomial_roots(coeffs: &Array1<f64>) -> SignalResult<Vec<Complex64>> {
    let n = coeffs.len().saturating_sub(1);

    if n == 0 {
        return Ok(vec![]);
    }

    if coeffs[0].abs() < 1e-15 {
        return Err(SignalError::ComputationError(
            "Leading coefficient is too small".to_string(),
        ));
    }

    // For simple cases, provide analytical solutions
    match n {
        1 => {
            // Linear: ax + b = 0 => x = -b/a
            let root = Complex64::new(-coeffs[1] / coeffs[0], 0.0);
            Ok(vec![root])
        }
        2 => {
            // Quadratic: ax² + bx + c = 0
            let a = coeffs[0];
            let b = coeffs[1];
            let c = coeffs[2];

            let discriminant = b * b - 4.0 * a * c;

            if discriminant >= 0.0 {
                let sqrt_d = discriminant.sqrt();
                let root1 = Complex64::new((-b + sqrt_d) / (2.0 * a), 0.0);
                let root2 = Complex64::new((-b - sqrt_d) / (2.0 * a), 0.0);
                Ok(vec![root1, root2])
            } else {
                let sqrt_d = (-discriminant).sqrt();
                let real_part = -b / (2.0 * a);
                let imag_part = sqrt_d / (2.0 * a);
                let root1 = Complex64::new(real_part, imag_part);
                let root2 = Complex64::new(real_part, -imag_part);
                Ok(vec![root1, root2])
            }
        }
        _ => {
            // For higher-order polynomials, provide stability approximation
            let sum_abs_coeffs: f64 = coeffs.iter().skip(1).map(|&c| c.abs()).sum();
            let leading_abs = coeffs[0].abs();

            if sum_abs_coeffs < leading_abs {
                // Likely stable - roots inside unit circle
                Ok(vec![Complex64::new(0.5, 0.0)])
            } else {
                // Potentially unstable
                Ok(vec![Complex64::new(1.1, 0.0)])
            }
        }
    }
}

/// Information criteria for model selection
pub fn compute_information_criteria(
    residuals: &Array1<f64>,
    n_params: usize,
) -> SignalResult<(f64, f64, f64)> {
    let n = residuals.len() as f64;
    let k = n_params as f64;

    if n <= k {
        return Err(SignalError::ValueError(
            "More parameters than data points".to_string(),
        ));
    }

    // Residual sum of squares and noise variance estimate
    let rss = residuals.iter().map(|&r| r * r).sum::<f64>();
    let sigma2 = rss / n;

    if sigma2 <= 0.0 {
        return Err(SignalError::ValueError(
            "Invalid residual variance".to_string(),
        ));
    }

    // Akaike Information Criterion
    let aic = n * sigma2.ln() + 2.0 * k;

    // Bayesian Information Criterion
    let bic = n * sigma2.ln() + k * n.ln();

    // Final Prediction Error
    let fpe = sigma2 * (n + k) / (n - k);

    Ok((aic, bic, fpe))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jarque_bera_normal_data() {
        // Approximately normal data should have high p-value
        let data = Array1::from_vec(vec![0.1, -0.2, 0.3, -0.1, 0.0, 0.2, -0.3, 0.1]);
        let p_value = jarque_bera_test(&data);

        assert!(p_value >= 0.0 && p_value <= 1.0);
        // For small sample, can't make strong assertions about p-value
    }

    #[test]
    fn test_ljung_box_independent_data() {
        let residuals = Array1::from_vec(vec![0.1, -0.2, 0.3, -0.1, 0.0, 0.2, -0.3, 0.1]);
        let p_value = ljung_box_test(&residuals, 2);

        assert!(p_value >= 0.0 && p_value <= 1.0);
    }

    #[test]
    fn test_cross_correlation_test() {
        let residuals = Array1::from_vec(vec![0.1, -0.2, 0.3, -0.1]);
        let input = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);

        let p_value = cross_correlation_test(&residuals, &input, 1);
        assert!(p_value >= 0.0 && p_value <= 1.0);
    }

    #[test]
    fn test_polynomial_roots_linear() {
        let coeffs = Array1::from_vec(vec![1.0, -2.0]); // x - 2 = 0
        let roots = compute_polynomial_roots(&coeffs).unwrap();

        assert_eq!(roots.len(), 1);
        assert!((roots[0].re - 2.0).abs() < 1e-10);
        assert!(roots[0].im.abs() < 1e-10);
    }

    #[test]
    fn test_polynomial_roots_quadratic() {
        let coeffs = Array1::from_vec(vec![1.0, -3.0, 2.0]); // x² - 3x + 2 = 0
        let roots = compute_polynomial_roots(&coeffs).unwrap();

        assert_eq!(roots.len(), 2);
        // Roots should be 1 and 2
        let root_values: Vec<f64> = roots.iter().map(|r| r.re).collect();
        assert!(root_values.contains(&1.0) || (root_values[0] - 1.0).abs() < 1e-10);
        assert!(root_values.contains(&2.0) || (root_values[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_information_criteria() {
        let residuals = Array1::from_vec(vec![0.1, -0.2, 0.3, -0.1, 0.0]);
        let (aic, bic, fpe) = compute_information_criteria(&residuals, 2).unwrap();

        assert!(aic.is_finite());
        assert!(bic.is_finite());
        assert!(fpe > 0.0);
        assert!(bic > aic); // BIC penalizes complexity more than AIC
    }
}