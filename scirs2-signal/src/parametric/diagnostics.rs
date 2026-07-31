//! Residual diagnostic tests for fitted ARMA models
//!
//! This module implements genuine statistical diagnostic tests used to
//! validate a fitted ARMA model's residuals:
//! - Ljung-Box portmanteau test for residual autocorrelation
//! - Jarque-Bera test for residual normality
//! - ARCH-LM test for residual heteroskedasticity (conditional variance)
//! - Kolmogorov-Smirnov / Anderson-Darling goodness-of-fit against a normal
//! - Breusch-Pagan-style level-dependence heteroskedasticity test
//! - Split-sample ("Chow-like") variance-ratio and CUSUM structural-break
//!   diagnostics
//!
//! These replace a previous set of stand-ins that returned
//! `Default::default()` (hardcoded `p_value: 1.0`, meaning every model
//! unconditionally "passed" every diagnostic test regardless of residual
//! quality).

use scirs2_core::ndarray::{Array1, Array2};
use statrs::distribution::{ChiSquared, ContinuousCDF, Normal};

use super::types::{ARCHTest, JarqueBeraTest, LjungBoxTest};

/// Sample autocorrelation of `x` at lag `k` (0 for out-of-range `k`, or
/// when the series has (numerically) zero variance).
pub(super) fn sample_autocorrelation(x: &[f64], k: usize) -> f64 {
    let n = x.len();
    if k >= n || n == 0 {
        return 0.0;
    }
    let mean = x.iter().sum::<f64>() / n as f64;

    let den: f64 = x.iter().map(|&v| (v - mean).powi(2)).sum();
    if den.abs() < 1e-300 {
        return 0.0;
    }

    let num: f64 = (0..(n - k))
        .map(|t| (x[t] - mean) * (x[t + k] - mean))
        .sum();
    num / den
}

/// Ljung-Box portmanteau test for residual autocorrelation up to `lags`.
///
/// `Q = n(n+2) * sum_{k=1}^{h} rho_k^2 / (n-k)`, asymptotically
/// chi-square distributed with `h` degrees of freedom under the null
/// hypothesis of no residual autocorrelation.
pub(super) fn ljung_box_test(residuals: &[f64], lags: usize) -> LjungBoxTest {
    let n = residuals.len();
    let h = lags.min(n.saturating_sub(1)).max(1);
    if n <= h {
        return LjungBoxTest {
            statistic: 0.0,
            p_value: 1.0,
            lags: h,
        };
    }

    let mut q = 0.0;
    for k in 1..=h {
        let rho_k = sample_autocorrelation(residuals, k);
        q += rho_k * rho_k / (n - k) as f64;
    }
    q *= n as f64 * (n as f64 + 2.0);

    let p_value = ChiSquared::new(h as f64)
        .ok()
        .map(|dist| 1.0 - dist.cdf(q))
        .unwrap_or(1.0)
        .clamp(0.0, 1.0);

    LjungBoxTest {
        statistic: q,
        p_value,
        lags: h,
    }
}

/// Jarque-Bera test for residual normality based on sample skewness and
/// excess kurtosis: `JB = n/6 * (S^2 + (K-3)^2/4)`, chi-square(2) under the
/// null hypothesis of normally distributed residuals.
pub(super) fn jarque_bera_test(residuals: &[f64]) -> JarqueBeraTest {
    let n = residuals.len();
    if n < 3 {
        return JarqueBeraTest {
            statistic: 0.0,
            p_value: 1.0,
        };
    }
    let nf = n as f64;
    let mean = residuals.iter().sum::<f64>() / nf;

    let mut m2 = 0.0;
    let mut m3 = 0.0;
    let mut m4 = 0.0;
    for &r in residuals {
        let d = r - mean;
        m2 += d * d;
        m3 += d * d * d;
        m4 += d * d * d * d;
    }
    m2 /= nf;
    m3 /= nf;
    m4 /= nf;

    if m2 < 1e-300 {
        return JarqueBeraTest {
            statistic: 0.0,
            p_value: 1.0,
        };
    }

    let skewness = m3 / m2.powf(1.5);
    let kurtosis = m4 / (m2 * m2);

    let jb = (nf / 6.0) * (skewness.powi(2) + (kurtosis - 3.0).powi(2) / 4.0);
    let p_value = ChiSquared::new(2.0)
        .ok()
        .map(|dist| 1.0 - dist.cdf(jb))
        .unwrap_or(1.0)
        .clamp(0.0, 1.0);

    JarqueBeraTest {
        statistic: jb,
        p_value,
    }
}

/// Solve the OLS normal equations `(X^T X) beta = X^T y` and report the
/// coefficient of determination `R^2` of the fit; returns `None` if the
/// design matrix is (numerically) singular.
fn ols_r_squared(x: &Array2<f64>, y: &Array1<f64>) -> Option<f64> {
    let n = y.len();
    if n == 0 {
        return None;
    }
    let xtx = x.t().dot(x);
    let xty = x.t().dot(y);
    let beta = scirs2_linalg::solve(&xtx.view(), &xty.view(), None).ok()?;

    let y_mean = y.iter().sum::<f64>() / n as f64;
    let mut ss_res = 0.0;
    let mut ss_tot = 0.0;
    for i in 0..n {
        let pred: f64 = (0..beta.len()).map(|k| beta[k] * x[[i, k]]).sum();
        ss_res += (y[i] - pred).powi(2);
        ss_tot += (y[i] - y_mean).powi(2);
    }

    if ss_tot < 1e-300 {
        Some(0.0)
    } else {
        Some((1.0 - ss_res / ss_tot).clamp(0.0, 1.0))
    }
}

/// ARCH-LM test for conditional heteroskedasticity: regresses squared
/// residuals on their own `lags` past values via OLS and forms the test
/// statistic `n * R^2`, asymptotically chi-square(`lags`) under the null
/// hypothesis of no ARCH effects.
pub(super) fn arch_lm_test(residuals: &[f64], lags: usize) -> ARCHTest {
    let n = residuals.len();
    let m = lags.min(n.saturating_sub(2)).max(1);
    if n <= m + 2 {
        return ARCHTest {
            statistic: 0.0,
            p_value: 1.0,
            lags: m,
        };
    }

    let sq: Vec<f64> = residuals.iter().map(|&r| r * r).collect();
    let n_eff = n - m;

    let mut x = Array2::<f64>::zeros((n_eff, m + 1));
    let mut y = Array1::<f64>::zeros(n_eff);
    for (row, t) in (m..n).enumerate() {
        x[[row, 0]] = 1.0;
        for lag in 1..=m {
            x[[row, lag]] = sq[t - lag];
        }
        y[row] = sq[t];
    }

    let statistic = match ols_r_squared(&x, &y) {
        Some(r_squared) => n_eff as f64 * r_squared,
        None => {
            return ARCHTest {
                statistic: 0.0,
                p_value: 1.0,
                lags: m,
            }
        }
    };

    let p_value = ChiSquared::new(m as f64)
        .ok()
        .map(|dist| 1.0 - dist.cdf(statistic))
        .unwrap_or(1.0)
        .clamp(0.0, 1.0);

    ARCHTest {
        statistic,
        p_value,
        lags: m,
    }
}

/// Standardize `residuals` (zero mean, unit variance); returns `None` when
/// the sample has fewer than 2 points or (numerically) zero variance.
fn standardized_sorted(residuals: &[f64]) -> Option<Vec<f64>> {
    let n = residuals.len();
    if n < 2 {
        return None;
    }
    let mean = residuals.iter().sum::<f64>() / n as f64;
    let variance = residuals.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n as f64;
    let std_dev = variance.sqrt();
    if std_dev < 1e-12 {
        return None;
    }
    let mut standardized: Vec<f64> = residuals.iter().map(|&r| (r - mean) / std_dev).collect();
    standardized.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Some(standardized)
}

/// Kolmogorov-Smirnov statistic comparing the empirical distribution of
/// (standardized) `residuals` against the standard normal distribution.
pub(super) fn kolmogorov_smirnov_normal_stat(residuals: &[f64]) -> f64 {
    let Some(standardized) = standardized_sorted(residuals) else {
        return 0.0;
    };
    let Ok(normal) = Normal::new(0.0, 1.0) else {
        return 0.0;
    };

    let n = standardized.len();
    let mut max_diff: f64 = 0.0;
    for (i, &x) in standardized.iter().enumerate() {
        let empirical_upper = (i + 1) as f64 / n as f64;
        let empirical_lower = i as f64 / n as f64;
        let theoretical = normal.cdf(x);
        max_diff = max_diff
            .max((empirical_upper - theoretical).abs())
            .max((theoretical - empirical_lower).abs());
    }
    max_diff
}

/// Anderson-Darling statistic comparing the empirical distribution of
/// (standardized) `residuals` against the standard normal distribution
/// (more sensitive to tail deviations than Kolmogorov-Smirnov).
pub(super) fn anderson_darling_normal_stat(residuals: &[f64]) -> f64 {
    let Some(standardized) = standardized_sorted(residuals) else {
        return 0.0;
    };
    let Ok(normal) = Normal::new(0.0, 1.0) else {
        return 0.0;
    };

    let n = standardized.len();
    let nf = n as f64;
    let mut sum = 0.0;
    for i in 0..n {
        let f_i = normal.cdf(standardized[i]).clamp(1e-12, 1.0 - 1e-12);
        let f_rev = normal
            .cdf(standardized[n - 1 - i])
            .clamp(1e-12, 1.0 - 1e-12);
        sum += (2.0 * (i as f64 + 1.0) - 1.0) * (f_i.ln() + (1.0 - f_rev).ln());
    }
    (-nf - sum / nf).max(0.0)
}

/// Breusch-Pagan-style heteroskedasticity statistic: regresses squared
/// residuals on the concurrent signal level via OLS and forms `n * R^2`
/// (asymptotically chi-square(1) under homoskedasticity).
pub(super) fn breusch_pagan_stat(residuals: &[f64], signal: &[f64], burn_in: usize) -> f64 {
    let n = residuals.len();
    if n < 3 {
        return 0.0;
    }

    let mut x = Array2::<f64>::zeros((n, 2));
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let t = burn_in + i;
        x[[i, 0]] = 1.0;
        x[[i, 1]] = if t < signal.len() { signal[t] } else { 0.0 };
        y[i] = residuals[i] * residuals[i];
    }

    match ols_r_squared(&x, &y) {
        Some(r_squared) => n as f64 * r_squared,
        None => 0.0,
    }
}

/// Split-sample structural-stability diagnostics computed directly from
/// the residual series (no model refitting required):
/// * a "Chow-like" variance-ratio statistic comparing the first vs second
///   half of the residuals (near 1.0 for a genuinely stable model);
/// * a CUSUM statistic: the maximum absolute cumulative sum of standardized
///   residuals, normalized by `sqrt(n)`;
/// * the standardized residuals themselves (the "recursive residuals").
///
/// Returns `(chow_test, cusum_test, recursive_residuals)`.
pub(super) fn stability_diagnostics(residuals: &[f64]) -> (f64, f64, Array1<f64>) {
    let n = residuals.len();
    if n < 4 {
        return (1.0, 0.0, Array1::zeros(n.max(1)));
    }

    let mean = residuals.iter().sum::<f64>() / n as f64;
    let variance = residuals.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n as f64;
    let std_dev = variance.sqrt().max(1e-12);

    let half = n / 2;
    let (first, second) = residuals.split_at(half);
    let sample_variance = |xs: &[f64]| -> f64 {
        if xs.is_empty() {
            return 0.0;
        }
        let m = xs.iter().sum::<f64>() / xs.len() as f64;
        xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / xs.len() as f64
    };
    let var1 = sample_variance(first).max(1e-300);
    let var2 = sample_variance(second).max(1e-300);
    let chow_test = var1.max(var2) / var1.min(var2);

    let mut cumsum = 0.0;
    let mut recursive_residuals = Array1::zeros(n);
    let mut max_abs_cusum: f64 = 0.0;
    for (i, &r) in residuals.iter().enumerate() {
        let standardized = (r - mean) / std_dev;
        recursive_residuals[i] = standardized;
        cumsum += standardized;
        max_abs_cusum = max_abs_cusum.max(cumsum.abs());
    }
    let cusum_test = max_abs_cusum / (n as f64).sqrt();

    (chow_test, cusum_test, recursive_residuals)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ljung_box_detects_autocorrelation() {
        // Strongly autocorrelated (smooth sinusoidal) series should be
        // flagged with a tiny p-value.
        let n = 200;
        let autocorrelated: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * i as f64 / 20.0).sin())
            .collect();
        let result = ljung_box_test(&autocorrelated, 10);
        assert!(result.p_value < 0.01, "p_value = {}", result.p_value);

        // Independent (alternating-sign, non-constant) noise-like series
        // should not be flagged as strongly.
        let mut rng_state: u64 = 123456789;
        let mut next = || {
            // xorshift64
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            (rng_state as f64 / u64::MAX as f64) - 0.5
        };
        let white_noise: Vec<f64> = (0..n).map(|_| next()).collect();
        let result2 = ljung_box_test(&white_noise, 10);
        assert!(
            result2.p_value > result.p_value,
            "white noise p_value ({}) should exceed autocorrelated p_value ({})",
            result2.p_value,
            result.p_value
        );
    }

    #[test]
    fn test_jarque_bera_flags_skewed_data() {
        // Exponential-like (heavily skewed) data should have a much lower
        // p-value than symmetric, non-constant data.
        let skewed: Vec<f64> = (1..201).map(|i| (i as f64).powi(3)).collect();
        let symmetric: Vec<f64> = (0..200)
            .map(|i| (2.0 * std::f64::consts::PI * i as f64 / 37.0).sin())
            .collect();

        let skewed_result = jarque_bera_test(&skewed);
        let symmetric_result = jarque_bera_test(&symmetric);

        assert!(skewed_result.statistic > symmetric_result.statistic);
        assert!(skewed_result.p_value < symmetric_result.p_value);
    }

    #[test]
    fn test_arch_lm_detects_conditional_heteroskedasticity() {
        // Volatility-clustered series: variance alternates between low and
        // high in blocks, which is exactly what ARCH effects look like.
        let mut rng_state: u64 = 42;
        let mut next = || {
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            (rng_state as f64 / u64::MAX as f64) - 0.5
        };
        let clustered: Vec<f64> = (0..300)
            .map(|i| {
                let scale = if (i / 20) % 2 == 0 { 0.1 } else { 5.0 };
                scale * next()
            })
            .collect();
        let homoskedastic: Vec<f64> = (0..300).map(|_| next()).collect();

        let clustered_result = arch_lm_test(&clustered, 5);
        let homoskedastic_result = arch_lm_test(&homoskedastic, 5);

        assert!(clustered_result.statistic > homoskedastic_result.statistic);
    }

    #[test]
    fn test_stability_diagnostics_detects_variance_break() {
        let mut rng_state: u64 = 7;
        let mut next = || {
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            (rng_state as f64 / u64::MAX as f64) - 0.5
        };
        let stable: Vec<f64> = (0..200).map(|_| next()).collect();
        let mut broken: Vec<f64> = (0..100).map(|_| 0.1 * next()).collect();
        broken.extend((0..100).map(|_| 10.0 * next()));

        let (chow_stable, _, _) = stability_diagnostics(&stable);
        let (chow_broken, _, _) = stability_diagnostics(&broken);

        assert!(chow_broken > chow_stable);
        assert!(chow_broken > 10.0);
    }
}
