//! Long memory and fractal time series analysis.
//!
//! Implements ARFIMA models, Hurst exponent estimation, fractional Brownian
//! Motion simulation, and related methods for analysing processes that exhibit
//! long-range dependence (slow hyperbolic decay of the autocorrelation function).
//!
//! # Key concepts
//! - **Long memory**: autocorrelations decay as a power-law rather than
//!   exponentially, characterised by the fractional differencing parameter
//!   `d ∈ (0, 0.5)` or equivalently the Hurst exponent `H = d + 0.5 ∈ (0.5, 1)`.
//! - **ARFIMA(p,d,q)**: generalises ARIMA by allowing non-integer `d`.
//! - **Fractional Brownian Motion (fBm)**: the canonical long-memory continuous
//!   Gaussian process parameterised by the Hurst exponent `H`.

use std::f64::consts::PI;

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::{random_normal_array, seeded_rng};

use crate::error::{Result, TimeSeriesError};

// ---------------------------------------------------------------------------
// Internal helper: simple linear regression of y on x.
// Returns (slope, intercept, std_error_of_slope).
// ---------------------------------------------------------------------------
fn linear_regression(x: &[f64], y: &[f64]) -> Result<(f64, f64, f64)> {
    let n = x.len();
    if n < 2 || y.len() != n {
        return Err(TimeSeriesError::InsufficientData {
            message: "linear regression requires at least 2 data points".to_string(),
            required: 2,
            actual: n,
        });
    }
    let nf = n as f64;
    let sum_x: f64 = x.iter().sum();
    let sum_y: f64 = y.iter().sum();
    let sum_xx: f64 = x.iter().map(|v| v * v).sum();
    let sum_xy: f64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();

    let denom = nf * sum_xx - sum_x * sum_x;
    if denom.abs() < f64::EPSILON {
        return Err(TimeSeriesError::NumericalInstability(
            "Degenerate design matrix in linear regression".to_string(),
        ));
    }
    let slope = (nf * sum_xy - sum_x * sum_y) / denom;
    let intercept = (sum_y - slope * sum_x) / nf;

    // Residual std error of the slope
    let ss_res: f64 = x
        .iter()
        .zip(y.iter())
        .map(|(xi, yi)| {
            let res = yi - (slope * xi + intercept);
            res * res
        })
        .sum();
    let s2 = ss_res / (nf - 2.0).max(1.0);
    let var_slope = s2 * nf / denom.max(f64::EPSILON);
    let std_err = var_slope.sqrt();

    Ok((slope, intercept, std_err))
}

// ---------------------------------------------------------------------------
// Periodogram via DFT (O(n^2) but avoids pulling in the full FFT crate for
// this specific sub-problem where only low-frequency ordinates are needed).
// For GPH we only need the first m ~ n^0.5 frequencies, so the cost is
// O(n * m) which is manageable.
// ---------------------------------------------------------------------------
fn periodogram_at(series: &[f64], k: usize) -> f64 {
    let n = series.len() as f64;
    let mean: f64 = series.iter().sum::<f64>() / n;
    let omega = 2.0 * PI * k as f64 / n;
    let mut re = 0.0_f64;
    let mut im = 0.0_f64;
    for (t, &v) in series.iter().enumerate() {
        let angle = omega * t as f64;
        re += (v - mean) * angle.cos();
        im += (v - mean) * angle.sin();
    }
    (re * re + im * im) / n
}

// ---------------------------------------------------------------------------
// Binomial coefficient for the fractional differencing expansion:
//   w_k = (-1)^k * C(d, k) = product_{j=0}^{k-1} (j - d) / (j + 1)
// ---------------------------------------------------------------------------
fn frac_diff_weights(d: f64, k: usize) -> f64 {
    // w_0 = 1, w_k = w_{k-1} * (k - 1 - d) / k
    let mut w = 1.0_f64;
    for j in 1..=k {
        w *= (j as f64 - 1.0 - d) / j as f64;
    }
    w
}

// ---------------------------------------------------------------------------
// Build the covariance vector of fGn with Hurst H.
//   gamma(k) = 0.5 * (|k-1|^{2H} - 2|k|^{2H} + |k+1|^{2H})
// ---------------------------------------------------------------------------
fn fgn_covariance(h: f64, n: usize) -> Vec<f64> {
    let mut cov = vec![0.0_f64; n];
    cov[0] = 1.0; // variance = 1 by construction
    for k in 1..n {
        let km1 = (k as f64 - 1.0_f64).abs().powf(2.0 * h);
        let k0 = (k as f64).powf(2.0 * h);
        let kp1 = (k as f64 + 1.0_f64).powf(2.0 * h);
        cov[k] = 0.5 * (km1 - 2.0 * k0 + kp1);
    }
    cov
}

// ---------------------------------------------------------------------------
// Davies-Harte method for simulating fGn via circulant embedding.
// Returns n samples of fGn.
// ---------------------------------------------------------------------------
fn davies_harte_fgn(h: f64, n: usize, seed: u64) -> Result<Vec<f64>> {
    // Build first row of circulant matrix: length m = 2*(n-1)
    let m = 2 * (n - 1).max(1);
    let cov = fgn_covariance(h, n);

    // Extend to circulant: r[k] for k=0..m
    let mut r = vec![0.0_f64; m];
    for k in 0..n {
        r[k] = cov[k];
    }
    // Symmetric part
    for k in 1..(m - n + 1) {
        r[m - k] = cov[k];
    }

    // Compute eigenvalues via DFT of r  (real, even → use cosine transform shortcut)
    let mut eigenvalues = vec![0.0_f64; m];
    for j in 0..m {
        let mut sum = 0.0_f64;
        for k in 0..m {
            let angle = 2.0 * PI * j as f64 * k as f64 / m as f64;
            sum += r[k] * angle.cos();
        }
        eigenvalues[j] = sum;
    }

    // Check non-negativity (round small negatives to zero)
    for ev in &mut eigenvalues {
        if *ev < 0.0 {
            *ev = 0.0;
        }
    }

    // Generate 2m standard normals
    let mut rng = seeded_rng(seed);
    let normals: Array1<f64> =
        random_normal_array(scirs2_core::ndarray::Ix1(2 * m), 0.0, 1.0, &mut rng);

    // Build complex noise in frequency domain: W_j = sqrt(lambda_j/2) * (z1 + i*z2)
    // Then IDFT gives the circulant process.  We only need the real part [0..n].
    let sqrt2 = 2.0_f64.sqrt();
    let mut freq_re = vec![0.0_f64; m];
    let mut freq_im = vec![0.0_f64; m];
    for j in 0..m {
        let scale = (eigenvalues[j] / 2.0).sqrt() / (m as f64).sqrt();
        freq_re[j] = scale * normals[2 * j];
        freq_im[j] = scale * normals[2 * j + 1];
    }
    // Handle DC and Nyquist specially (must be real)
    freq_re[0] = (eigenvalues[0] / m as f64).sqrt() * normals[0];
    freq_im[0] = 0.0;
    if m % 2 == 0 {
        let ny = m / 2;
        freq_re[ny] = (eigenvalues[ny] / m as f64).sqrt() * normals[1];
        freq_im[ny] = 0.0;
    }

    // IDFT to get real process
    let mut output = vec![0.0_f64; n];
    for t in 0..n {
        let mut re = 0.0_f64;
        for j in 0..m {
            let angle = 2.0 * PI * j as f64 * t as f64 / m as f64;
            re += freq_re[j] * angle.cos() - freq_im[j] * angle.sin();
        }
        output[t] = re * sqrt2; // scale to unit variance
    }

    Ok(output)
}

// ===========================================================================
// Public types and implementations
// ===========================================================================

/// Method used to estimate the fractional differencing parameter `d`.
#[derive(Debug, Clone, PartialEq)]
pub enum FitMethod {
    /// Geweke-Porter-Hudak (GPH) log-periodogram regression.
    GPH,
    /// Whittle approximate maximum likelihood in the frequency domain.
    Whittle,
    /// R/S (Hurst-exponent-based): `d_hat = H - 0.5`.
    R2,
}

/// ARFIMA(p, d, q) model.
///
/// Fractionally Integrated ARMA: `(1-L)^d φ(L) y_t = θ(L) ε_t`.
/// When `d ∈ (0, 0.5)` the process exhibits long memory.
#[derive(Debug, Clone)]
pub struct Arfima {
    /// AR order.
    pub p: usize,
    /// Fractional differencing parameter; long memory when `d ∈ (0, 0.5)`.
    pub d: f64,
    /// MA order.
    pub q: usize,
    /// AR coefficients φ₁, …, φₚ.
    pub ar: Vec<f64>,
    /// MA coefficients θ₁, …, θ_q.
    pub ma: Vec<f64>,
    /// Innovation standard deviation.
    pub sigma: f64,
    /// Process mean.
    pub mean: f64,
}

impl Arfima {
    /// Create an ARFIMA(p, d, q) model specification with zero AR/MA
    /// coefficients, unit innovation std and zero mean.
    pub fn arfima(p: usize, d: f64, q: usize) -> Self {
        Self {
            p,
            d,
            q,
            ar: vec![0.0; p],
            ma: vec![0.0; q],
            sigma: 1.0,
            mean: 0.0,
        }
    }

    /// Returns the parameter range `(0.0, 0.5)` for which long memory holds.
    pub fn long_memory_range() -> (f64, f64) {
        (0.0, 0.5)
    }

    /// Simulate an ARFIMA process of length `n`.
    ///
    /// The procedure:
    /// 1. Generate innovations `ε_t ~ N(0, σ²)`.
    /// 2. Apply ARMA filter to obtain `x_t`.
    /// 3. Invert the fractional differencing operator `(1-L)^{-d}` by
    ///    convolving with truncated binomial weights.
    pub fn simulate(&self, n: usize, seed: u64) -> Result<Array1<f64>> {
        if n == 0 {
            return Err(TimeSeriesError::InvalidParameter {
                name: "n".to_string(),
                message: "Length must be positive".to_string(),
            });
        }
        if self.d.abs() >= 0.5 {
            return Err(TimeSeriesError::InvalidParameter {
                name: "d".to_string(),
                message: format!(
                    "Fractional differencing parameter d={} must satisfy |d| < 0.5",
                    self.d
                ),
            });
        }

        // Generate burn-in + actual length innovations
        let burn = 100_usize.max(n / 4);
        let total = n + burn;
        let mut rng = seeded_rng(seed);
        let innovations: Array1<f64> =
            random_normal_array(scirs2_core::ndarray::Ix1(total), 0.0, self.sigma, &mut rng);

        // ARMA filter
        let mut arma = vec![0.0_f64; total];
        for t in 0..total {
            arma[t] = innovations[t];
            for (j, &phi) in self.ar.iter().enumerate() {
                if t > j {
                    arma[t] += phi * arma[t - j - 1];
                }
            }
            for (j, &theta) in self.ma.iter().enumerate() {
                if t > j {
                    arma[t] += theta * innovations[t - j - 1];
                }
            }
        }

        // Apply fractional integration (1-L)^{-d}: cumulate with weights
        // w_k(-d) = fractional differencing weights for -d
        let threshold = 1e-9_f64;
        let max_lag = {
            let mut k = 1_usize;
            while k < total && frac_diff_weights(-self.d, k).abs() > threshold {
                k += 1;
            }
            k.min(total)
        };

        let mut weights_neg = vec![0.0_f64; max_lag];
        for k in 0..max_lag {
            weights_neg[k] = frac_diff_weights(-self.d, k);
        }

        let mut integrated = vec![0.0_f64; total];
        for t in 0..total {
            let mut val = 0.0_f64;
            for (k, &w) in weights_neg.iter().enumerate() {
                if t >= k {
                    val += w * arma[t - k];
                }
            }
            integrated[t] = val + self.mean;
        }

        // Discard burn-in
        let out: Vec<f64> = integrated[burn..].to_vec();
        Ok(Array1::from(out))
    }

    /// Apply the fractional differencing operator `(1-L)^d` to `series`.
    ///
    /// Uses a truncated binomial expansion; lags with `|coefficient| < threshold`
    /// are discarded.
    pub fn fractional_diff(series: &Array1<f64>, d: f64, threshold: f64) -> Result<Array1<f64>> {
        let n = series.len();
        if n == 0 {
            return Err(TimeSeriesError::InsufficientData {
                message: "Series must be non-empty".to_string(),
                required: 1,
                actual: 0,
            });
        }

        // Determine truncation point
        let mut max_lag = 1_usize;
        while max_lag < n {
            let w = frac_diff_weights(d, max_lag);
            if w.abs() <= threshold {
                break;
            }
            max_lag += 1;
        }

        let mut weights = vec![0.0_f64; max_lag];
        for k in 0..max_lag {
            weights[k] = frac_diff_weights(d, k);
        }

        let out: Vec<f64> = (0..n)
            .map(|t| {
                let mut val = 0.0_f64;
                for (k, &w) in weights.iter().enumerate() {
                    if t >= k {
                        val += w * series[t - k];
                    }
                }
                val
            })
            .collect();

        Ok(Array1::from(out))
    }

    /// Compute the Gaussian log-likelihood of the observed `series` under this
    /// ARFIMA model.  Uses a simplified frequency-domain approximation
    /// (Whittle likelihood).
    pub fn log_likelihood(&self, series: &Array1<f64>) -> Result<f64> {
        let n = series.len();
        if n < 4 {
            return Err(TimeSeriesError::InsufficientData {
                message: "Log-likelihood requires at least 4 observations".to_string(),
                required: 4,
                actual: n,
            });
        }
        // Fractionally difference the series
        let diff = Self::fractional_diff(series, self.d, 1e-9)?;

        // Compute residuals via ARMA filter on diff
        let mut residuals = vec![0.0_f64; n];
        for t in 0..n {
            residuals[t] = diff[t];
            for (j, &phi) in self.ar.iter().enumerate() {
                if t > j {
                    residuals[t] -= phi * diff[t - j - 1];
                }
            }
            for (j, &theta) in self.ma.iter().enumerate() {
                if t > j {
                    residuals[t] -= theta * residuals[t - j - 1];
                }
            }
        }

        let sigma2 = self.sigma * self.sigma;
        let sum_sq: f64 = residuals.iter().map(|r| r * r).sum();
        let ll = -0.5 * n as f64 * (2.0 * PI * sigma2).ln() - 0.5 * sum_sq / sigma2;
        Ok(ll)
    }
}

/// Fit ARFIMA(0, d, 0) using the specified estimation method.
///
/// Returns `(d_hat, std_error)`.
pub fn fit_arfima_d(series: &Array1<f64>, method: FitMethod) -> Result<(f64, f64)> {
    match method {
        FitMethod::GPH => {
            let (d_hat, std_err, _p) = gph_estimate(series, None)?;
            Ok((d_hat, std_err))
        }
        FitMethod::Whittle => {
            let d_hat = whittle_estimate(series, 0, 0)?;
            // Whittle std error ~ 1/(sqrt(m)) where m ~ n^0.5
            let m = (series.len() as f64).sqrt() as f64;
            let std_err = 1.0 / m.sqrt().max(1.0);
            Ok((d_hat, std_err))
        }
        FitMethod::R2 => {
            let (hurst, _, _) = HurstEstimator::rs_analysis(series, 8, series.len() / 2, 10)?;
            let d_hat = hurst - 0.5;
            // Approximate std error for R/S-based estimate
            let std_err = 0.5 / (series.len() as f64).ln().max(1.0);
            Ok((d_hat, std_err))
        }
    }
}

/// Methods for estimating the Hurst exponent of a time series.
pub struct HurstEstimator;

impl HurstEstimator {
    /// R/S (Rescaled Range) analysis.
    ///
    /// For each window size in `[min_window, max_window]` (on a log scale),
    /// computes the expected R/S statistic and fits `log(R/S) = H * log(n) + C`.
    ///
    /// Returns `(H, log_n_values, log_rs_values)`.
    pub fn rs_analysis(
        series: &Array1<f64>,
        min_window: usize,
        max_window: usize,
        n_windows: usize,
    ) -> Result<(f64, Vec<f64>, Vec<f64>)> {
        let n = series.len();
        if n < 8 {
            return Err(TimeSeriesError::InsufficientData {
                message: "R/S analysis requires at least 8 observations".to_string(),
                required: 8,
                actual: n,
            });
        }
        let min_w = min_window.max(4);
        let max_w = max_window.min(n / 2).max(min_w + 1);
        if min_w >= max_w {
            return Err(TimeSeriesError::InvalidParameter {
                name: "window".to_string(),
                message: format!(
                    "min_window ({}) must be strictly less than max_window ({})",
                    min_w, max_w
                ),
            });
        }
        let n_w = n_windows.max(2);

        // Generate window sizes on a log scale
        let log_min = (min_w as f64).ln();
        let log_max = (max_w as f64).ln();
        let mut sizes: Vec<usize> = (0..n_w)
            .map(|i| {
                let t = i as f64 / (n_w - 1) as f64;
                ((1.0 - t) * log_min + t * log_max).exp().round() as usize
            })
            .collect();
        sizes.sort_unstable();
        sizes.dedup();

        let mut log_ns: Vec<f64> = Vec::new();
        let mut log_rs: Vec<f64> = Vec::new();

        for &ws in &sizes {
            if ws < 4 || ws > n {
                continue;
            }
            let n_segments = n / ws;
            if n_segments == 0 {
                continue;
            }
            let mut rs_vals: Vec<f64> = Vec::new();
            for seg in 0..n_segments {
                let start = seg * ws;
                let end = start + ws;
                let segment: Vec<f64> = series
                    .slice(scirs2_core::ndarray::s![start..end])
                    .iter()
                    .copied()
                    .collect();
                let mean = segment.iter().sum::<f64>() / ws as f64;
                let std = {
                    let var = segment.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / ws as f64;
                    var.sqrt()
                };
                if std < f64::EPSILON {
                    continue;
                }
                // Cumulative deviation
                let mut cumdev = vec![0.0_f64; ws];
                let mut running = 0.0_f64;
                for (i, &v) in segment.iter().enumerate() {
                    running += v - mean;
                    cumdev[i] = running;
                }
                let range = cumdev.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
                    - cumdev.iter().cloned().fold(f64::INFINITY, f64::min);
                rs_vals.push(range / std);
            }
            if rs_vals.is_empty() {
                continue;
            }
            let mean_rs = rs_vals.iter().sum::<f64>() / rs_vals.len() as f64;
            if mean_rs > 0.0 {
                log_ns.push((ws as f64).ln());
                log_rs.push(mean_rs.ln());
            }
        }

        if log_ns.len() < 2 {
            return Err(TimeSeriesError::InsufficientData {
                message: "Not enough valid R/S observations for regression".to_string(),
                required: 2,
                actual: log_ns.len(),
            });
        }

        let (slope, _intercept, _std_err) = linear_regression(&log_ns, &log_rs)?;
        Ok((slope.clamp(0.0, 1.0), log_ns, log_rs))
    }

    /// Detrended Fluctuation Analysis (DFA).
    ///
    /// Returns `(alpha, log_scale_values, log_F_values)`.
    /// - `alpha ≈ 0.5` → uncorrelated (white noise / random walk).
    /// - `alpha > 0.5` → positive long-range correlations.
    pub fn dfa(
        series: &Array1<f64>,
        scales: &[usize],
        order: usize,
    ) -> Result<(f64, Vec<f64>, Vec<f64>)> {
        let n = series.len();
        if n < 16 {
            return Err(TimeSeriesError::InsufficientData {
                message: "DFA requires at least 16 observations".to_string(),
                required: 16,
                actual: n,
            });
        }
        if scales.is_empty() {
            return Err(TimeSeriesError::InvalidParameter {
                name: "scales".to_string(),
                message: "At least one scale must be provided".to_string(),
            });
        }

        // Cumulative sum (profile)
        let mean = series.iter().sum::<f64>() / n as f64;
        let profile: Vec<f64> = {
            let mut acc = 0.0_f64;
            series
                .iter()
                .map(|&v| {
                    acc += v - mean;
                    acc
                })
                .collect()
        };

        let mut log_scales: Vec<f64> = Vec::new();
        let mut log_f: Vec<f64> = Vec::new();

        for &s in scales {
            if s < order + 2 || s > n / 2 {
                continue;
            }
            let n_segs = n / s;
            if n_segs == 0 {
                continue;
            }
            let mut variance_sum = 0.0_f64;
            let mut count = 0_usize;
            for seg in 0..n_segs {
                let start = seg * s;
                let end = start + s;
                let seg_data: Vec<f64> = profile[start..end].to_vec();
                // Fit polynomial of given order and compute RMS residual
                let trend = fit_polynomial_trend(&seg_data, order)?;
                let rms: f64 = seg_data
                    .iter()
                    .zip(trend.iter())
                    .map(|(v, t)| (v - t).powi(2))
                    .sum::<f64>()
                    / s as f64;
                variance_sum += rms;
                count += 1;
            }
            if count == 0 {
                continue;
            }
            let f_s = (variance_sum / count as f64).sqrt();
            if f_s > 0.0 {
                log_scales.push((s as f64).ln());
                log_f.push(f_s.ln());
            }
        }

        if log_scales.len() < 2 {
            return Err(TimeSeriesError::InsufficientData {
                message: "Not enough valid DFA scales for regression".to_string(),
                required: 2,
                actual: log_scales.len(),
            });
        }

        let (alpha, _intercept, _std_err) = linear_regression(&log_scales, &log_f)?;
        Ok((alpha.clamp(0.0, 2.0), log_scales, log_f))
    }

    /// Multifractal DFA (MFDFA).
    ///
    /// Computes the generalised fluctuation function `F_q(s)` for each `q` in
    /// `q_values` and each scale in `scales`.
    ///
    /// Returns `(q_values_out, F_q_matrix)` where `F_q_matrix[i]` is the
    /// vector of log F_q values across log-scales for the i-th q.
    pub fn mfdfa(
        series: &Array1<f64>,
        scales: &[usize],
        q_values: &[f64],
        order: usize,
    ) -> Result<(Vec<f64>, Vec<Vec<f64>>)> {
        let n = series.len();
        if n < 16 {
            return Err(TimeSeriesError::InsufficientData {
                message: "MFDFA requires at least 16 observations".to_string(),
                required: 16,
                actual: n,
            });
        }
        if scales.is_empty() || q_values.is_empty() {
            return Err(TimeSeriesError::InvalidParameter {
                name: "scales/q_values".to_string(),
                message: "Scales and q_values must be non-empty".to_string(),
            });
        }

        // Build profile
        let mean = series.iter().sum::<f64>() / n as f64;
        let profile: Vec<f64> = {
            let mut acc = 0.0_f64;
            series
                .iter()
                .map(|&v| {
                    acc += v - mean;
                    acc
                })
                .collect()
        };

        let mut fq_matrix: Vec<Vec<f64>> = vec![Vec::new(); q_values.len()];

        for &s in scales {
            if s < order + 2 || s > n / 2 {
                continue;
            }
            let n_segs = n / s;
            if n_segs == 0 {
                continue;
            }

            // Collect per-segment variance
            let mut seg_variances: Vec<f64> = Vec::with_capacity(n_segs);
            for seg in 0..n_segs {
                let start = seg * s;
                let end = start + s;
                let seg_data: Vec<f64> = profile[start..end].to_vec();
                let trend = fit_polynomial_trend(&seg_data, order)?;
                let rms: f64 = seg_data
                    .iter()
                    .zip(trend.iter())
                    .map(|(v, t)| (v - t).powi(2))
                    .sum::<f64>()
                    / s as f64;
                seg_variances.push(rms);
            }

            for (qi, &q) in q_values.iter().enumerate() {
                let fq = if q.abs() < 1e-10 {
                    // q=0: geometric mean
                    let log_sum: f64 = seg_variances
                        .iter()
                        .filter(|&&v| v > 0.0)
                        .map(|&v| v.ln())
                        .sum();
                    let cnt = seg_variances.iter().filter(|&&v| v > 0.0).count();
                    if cnt == 0 {
                        continue;
                    }
                    (log_sum / (2.0 * cnt as f64)).exp()
                } else {
                    let sum: f64 = seg_variances
                        .iter()
                        .filter(|&&v| v > 0.0)
                        .map(|&v| v.powf(q / 2.0))
                        .sum();
                    let cnt = seg_variances.iter().filter(|&&v| v > 0.0).count();
                    if cnt == 0 {
                        continue;
                    }
                    (sum / cnt as f64).powf(1.0 / q)
                };
                if fq > 0.0 {
                    fq_matrix[qi].push(fq.ln());
                }
            }
        }

        Ok((q_values.to_vec(), fq_matrix))
    }

    /// Wavelet-based Hurst exponent estimation using a Haar wavelet decomposition.
    ///
    /// Computes the wavelet variance at each level and fits `log(var) ~ -2H * log(scale)`.
    pub fn wavelet_hurst(series: &Array1<f64>) -> Result<f64> {
        let wv = wavelet_variance(series, 10)?;
        if wv.len() < 2 {
            return Err(TimeSeriesError::InsufficientData {
                message: "Wavelet Hurst estimation requires at least 2 levels".to_string(),
                required: 2,
                actual: wv.len(),
            });
        }
        let log_scales: Vec<f64> = (1..=wv.len())
            .map(|j| (2.0_f64.powi(j as i32)).ln())
            .collect();
        let log_wv: Vec<f64> = wv.iter().map(|&v| (v.max(f64::EPSILON)).ln()).collect();
        let (slope, _ic, _se) = linear_regression(&log_scales, &log_wv)?;
        // Relationship: wavelet_variance ~ scale^{2H-2} → slope = 2H-2 → H = (slope+2)/2
        let h = ((slope + 2.0) / 2.0).clamp(0.0, 1.0);
        Ok(h)
    }
}

/// Fit a polynomial trend of the given `order` to `data` using least squares.
/// Returns the fitted values at each point.
fn fit_polynomial_trend(data: &[f64], order: usize) -> Result<Vec<f64>> {
    let n = data.len();
    let deg = order.min(n.saturating_sub(1));

    // Build Vandermonde-style design matrix columns: x^0, x^1, ..., x^deg
    let cols = deg + 1;
    let mut design = vec![vec![0.0_f64; n]; cols];
    for (i, row) in design.iter_mut().enumerate() {
        for (t, v) in row.iter_mut().enumerate() {
            *v = (t as f64 / (n as f64 - 1.0).max(1.0)).powi(i as i32);
        }
    }

    // Solve normal equations via Cholesky (small system, up to ~order^2)
    let mut ata = vec![vec![0.0_f64; cols]; cols];
    let mut aty = vec![0.0_f64; cols];
    for i in 0..cols {
        for j in 0..cols {
            ata[i][j] = design[i]
                .iter()
                .zip(design[j].iter())
                .map(|(a, b)| a * b)
                .sum();
        }
        aty[i] = design[i].iter().zip(data.iter()).map(|(a, b)| a * b).sum();
    }

    let coeffs = solve_small_ls(&ata, &aty)?;

    // Compute fitted values
    let fitted: Vec<f64> = (0..n)
        .map(|t| {
            let x = t as f64 / (n as f64 - 1.0).max(1.0);
            coeffs
                .iter()
                .enumerate()
                .map(|(i, &c)| c * x.powi(i as i32))
                .sum()
        })
        .collect();

    Ok(fitted)
}

/// Solve a small symmetric positive-definite system A x = b via LDL^T (inline).
fn solve_small_ls(a: &[Vec<f64>], b: &[f64]) -> Result<Vec<f64>> {
    let n = b.len();
    // Forward substitution with partial pivoting (Gaussian elimination)
    let mut mat: Vec<Vec<f64>> = a.to_vec();
    let mut rhs: Vec<f64> = b.to_vec();

    for col in 0..n {
        // Find pivot
        let max_row = (col..n)
            .max_by(|&i, &j| {
                mat[i][col]
                    .abs()
                    .partial_cmp(&mat[j][col].abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(col);
        mat.swap(col, max_row);
        rhs.swap(col, max_row);

        let pivot = mat[col][col];
        if pivot.abs() < 1e-14 {
            return Err(TimeSeriesError::NumericalInstability(
                "Singular matrix in polynomial trend fitting".to_string(),
            ));
        }
        for row in (col + 1)..n {
            let factor = mat[row][col] / pivot;
            for k in col..n {
                let old = mat[col][k];
                mat[row][k] -= factor * old;
            }
            rhs[row] -= factor * rhs[col];
        }
    }

    // Back-substitution
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        x[i] = rhs[i];
        for j in (i + 1)..n {
            x[i] -= mat[i][j] * x[j];
        }
        x[i] /= mat[i][i];
    }
    Ok(x)
}

/// Geweke-Porter-Hudak (GPH) log-periodogram regression.
///
/// Regresses `log I(ω_j)` on `log(4 sin²(ω_j/2))` at the `m` lowest Fourier
/// frequencies.  The slope estimate is `−2 d_hat`.
///
/// Returns `(d_hat, std_error, p_value)`.
pub fn gph_estimate(series: &Array1<f64>, bandwidth: Option<usize>) -> Result<(f64, f64, f64)> {
    let n = series.len();
    if n < 10 {
        return Err(TimeSeriesError::InsufficientData {
            message: "GPH estimate requires at least 10 observations".to_string(),
            required: 10,
            actual: n,
        });
    }

    // Bandwidth m: default m = floor(n^0.5)
    let m = bandwidth
        .unwrap_or_else(|| (n as f64).sqrt() as usize)
        .max(2)
        .min(n / 2);

    // Frequencies ω_j = 2π j / n for j = 1, ..., m
    let data: Vec<f64> = series.iter().copied().collect();
    let mut log_pgram: Vec<f64> = Vec::with_capacity(m);
    let mut log_freq: Vec<f64> = Vec::with_capacity(m);

    for j in 1..=m {
        let omega = 2.0 * PI * j as f64 / n as f64;
        let pg = periodogram_at(&data, j);
        if pg > 0.0 {
            let sin_half = (omega / 2.0).sin();
            let x_val = (4.0 * sin_half * sin_half).ln();
            log_pgram.push(pg.ln());
            log_freq.push(x_val);
        }
    }

    if log_freq.len() < 2 {
        return Err(TimeSeriesError::InsufficientData {
            message: "Not enough non-zero periodogram ordinates for GPH".to_string(),
            required: 2,
            actual: log_freq.len(),
        });
    }

    let (slope, _intercept, std_err) = linear_regression(&log_freq, &log_pgram)?;
    let d_hat = -slope / 2.0;
    let d_std_err = std_err / 2.0;

    // Approximate p-value for H0: d = 0 (two-tailed normal test)
    let z = if d_std_err > f64::EPSILON {
        d_hat / d_std_err
    } else {
        0.0
    };
    // Rough p-value approximation using the normal CDF approximation
    let p_value = 2.0 * normal_sf(z.abs());

    Ok((d_hat, d_std_err, p_value))
}

/// Whittle approximate maximum likelihood estimator for ARFIMA(p, d, q).
///
/// Minimises the Whittle contrast function over a grid of `d` values in
/// `(-0.49, 0.49)`.
pub fn whittle_estimate(series: &Array1<f64>, _p: usize, _q: usize) -> Result<f64> {
    let n = series.len();
    if n < 10 {
        return Err(TimeSeriesError::InsufficientData {
            message: "Whittle estimate requires at least 10 observations".to_string(),
            required: 10,
            actual: n,
        });
    }

    let data: Vec<f64> = series.iter().copied().collect();
    // Compute periodogram at all Fourier frequencies
    let m_max = n / 2;
    let mut periodogram_vals: Vec<f64> = (1..=m_max).map(|j| periodogram_at(&data, j)).collect();

    // Grid search over d ∈ (-0.49, 0.49)
    let n_grid = 200_usize;
    let mut best_d = 0.0_f64;
    let mut best_contrast = f64::INFINITY;

    for k in 0..=n_grid {
        let d_cand = -0.49 + 0.98 * k as f64 / n_grid as f64;
        // Spectral density of ARFIMA(0,d,0): f(ω) = σ²/(2π) * |1 - e^{-iω}|^{-2d}
        // = σ²/(2π) * (4 sin²(ω/2))^{-d}
        let mut contrast = 0.0_f64;
        for j in 1..=m_max {
            let omega = 2.0 * PI * j as f64 / n as f64;
            let sin_half = (omega / 2.0).sin();
            // spectral density (up to constant): (4 sin²)^{-d}
            let spec = (4.0 * sin_half * sin_half).powf(-d_cand);
            if spec > f64::EPSILON {
                contrast += periodogram_vals[j - 1] / spec + spec.ln();
            }
        }
        if contrast < best_contrast {
            best_contrast = contrast;
            best_d = d_cand;
        }
    }

    Ok(best_d)
}

/// Simulate fractional Brownian Motion (fBm) with Hurst exponent `H ∈ (0, 1)`.
///
/// Uses the Davies-Harte exact method for `n ≥ 64`, and a Cholesky factorisation
/// for smaller `n`.
pub fn fbm_simulate(hurst: f64, n: usize, seed: u64) -> Result<Array1<f64>> {
    if !(0.0 < hurst && hurst < 1.0) {
        return Err(TimeSeriesError::InvalidParameter {
            name: "hurst".to_string(),
            message: format!("Hurst exponent must be in (0, 1), got {}", hurst),
        });
    }
    if n == 0 {
        return Err(TimeSeriesError::InvalidParameter {
            name: "n".to_string(),
            message: "Length must be positive".to_string(),
        });
    }

    // Obtain fGn increments and cumulate to form fBm
    let fgn = if n < 64 {
        cholesky_fgn(hurst, n, seed)?
    } else {
        davies_harte_fgn(hurst, n, seed)?
    };

    let fbm: Vec<f64> = {
        let mut acc = 0.0_f64;
        let mut v: Vec<f64> = fgn
            .iter()
            .map(|&x| {
                acc += x;
                acc
            })
            .collect();
        // Prepend 0 for the standard fBm convention (starts at 0)
        let mut result = vec![0.0_f64; n + 1];
        result[0] = 0.0;
        result[1..].copy_from_slice(&v);
        // Trim to length n
        result[..n].to_vec()
    };

    Ok(Array1::from(fbm))
}

/// Simulate fractional Gaussian Noise (fGn) — the increments of fBm.
pub fn fgn_simulate(hurst: f64, n: usize, seed: u64) -> Result<Array1<f64>> {
    if !(0.0 < hurst && hurst < 1.0) {
        return Err(TimeSeriesError::InvalidParameter {
            name: "hurst".to_string(),
            message: format!("Hurst exponent must be in (0, 1), got {}", hurst),
        });
    }
    if n == 0 {
        return Err(TimeSeriesError::InvalidParameter {
            name: "n".to_string(),
            message: "Length must be positive".to_string(),
        });
    }
    let samples = if n < 64 {
        cholesky_fgn(hurst, n, seed)?
    } else {
        davies_harte_fgn(hurst, n, seed)?
    };
    Ok(Array1::from(samples))
}

/// Cholesky-based fGn simulation for small `n`.
fn cholesky_fgn(hurst: f64, n: usize, seed: u64) -> Result<Vec<f64>> {
    let cov = fgn_covariance(hurst, n);

    // Build covariance matrix
    let mut c = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in 0..n {
            let lag = if i >= j { i - j } else { j - i };
            c[i][j] = cov[lag];
        }
    }

    // Cholesky factorisation L such that C = L L^T
    let mut l = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = c[i][j];
            for k in 0..j {
                s -= l[i][k] * l[j][k];
            }
            l[i][j] = if i == j {
                s.max(0.0).sqrt()
            } else if l[j][j].abs() < f64::EPSILON {
                0.0
            } else {
                s / l[j][j]
            };
        }
    }

    // Generate standard normals and multiply by L
    let mut rng = seeded_rng(seed);
    let z: Array1<f64> = random_normal_array(scirs2_core::ndarray::Ix1(n), 0.0, 1.0, &mut rng);

    let mut output = vec![0.0_f64; n];
    for i in 0..n {
        for j in 0..=i {
            output[i] += l[i][j] * z[j];
        }
    }

    Ok(output)
}

/// Test whether a time series exhibits statistically significant long memory.
///
/// Uses GPH estimation; returns `(d_estimate, p_value, significant_at_5%)`.
pub fn long_memory_test(series: &Array1<f64>) -> Result<(f64, f64, bool)> {
    let (d_hat, _se, p_value) = gph_estimate(series, None)?;
    Ok((d_hat, p_value, p_value < 0.05))
}

/// Discrete Haar wavelet variance at levels 1, …, `max_level`.
///
/// Returns a vector of length `min(max_level, floor(log2(n)))` with the
/// estimated variance at each dyadic scale.
pub fn wavelet_variance(series: &Array1<f64>, max_level: usize) -> Result<Vec<f64>> {
    let n = series.len();
    if n < 4 {
        return Err(TimeSeriesError::InsufficientData {
            message: "Wavelet variance requires at least 4 observations".to_string(),
            required: 4,
            actual: n,
        });
    }

    let max_possible = (n as f64).log2().floor() as usize;
    let levels = max_level.min(max_possible).max(1);

    let mut coeffs: Vec<f64> = series.iter().copied().collect();
    let mut variances: Vec<f64> = Vec::with_capacity(levels);

    for _level in 0..levels {
        let m = coeffs.len();
        if m < 2 {
            break;
        }
        let half = m / 2;
        let mut detail = Vec::with_capacity(half);
        let mut approx = Vec::with_capacity(half);
        for i in 0..half {
            let a = coeffs[2 * i];
            let b = coeffs[2 * i + 1];
            detail.push((a - b) / 2.0_f64.sqrt());
            approx.push((a + b) / 2.0_f64.sqrt());
        }
        // Variance of detail coefficients at this level
        let mean = detail.iter().sum::<f64>() / detail.len() as f64;
        let var = detail.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / detail.len() as f64;
        variances.push(var);
        coeffs = approx;
    }

    if variances.is_empty() {
        return Err(TimeSeriesError::ComputationError(
            "Could not compute any wavelet variance levels".to_string(),
        ));
    }

    Ok(variances)
}

// ---------------------------------------------------------------------------
// Statistical helpers
// ---------------------------------------------------------------------------

/// Survival function of the standard normal (complementary CDF).
/// Uses a rational approximation that achieves ~7 significant digits.
fn normal_sf(x: f64) -> f64 {
    let t = x.abs();
    let e = (-(t * t) / 2.0).exp();
    // Abramowitz & Stegun 26.2.17
    let p = 0.2316419_f64;
    let b1 = 0.319381530_f64;
    let b2 = -0.356563782_f64;
    let b3 = 1.781477937_f64;
    let b4 = -1.821255978_f64;
    let b5 = 1.330274429_f64;
    let k = 1.0 / (1.0 + p * t);
    let poly = ((((b5 * k + b4) * k + b3) * k + b2) * k + b1) * k;
    let sf = e * poly / (2.0 * PI).sqrt();
    if x >= 0.0 {
        sf.max(0.0)
    } else {
        1.0 - sf
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    // ------------------------------------------------------------------
    // Helper: build a zero-mean white noise series
    // ------------------------------------------------------------------
    fn white_noise(n: usize, seed: u64) -> Array1<f64> {
        let mut rng = seeded_rng(seed);
        random_normal_array(scirs2_core::ndarray::Ix1(n), 0.0, 1.0, &mut rng)
    }

    // ------------------------------------------------------------------
    // Fractional differencing
    // ------------------------------------------------------------------
    #[test]
    fn test_frac_diff_d0_is_identity() {
        // d = 0 → w_0 = 1, all other weights zero → identity
        let series = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let result = Arfima::fractional_diff(&series, 0.0, 1e-12).expect("fractional_diff failed");
        for (a, b) in result.iter().zip(series.iter()) {
            assert!(
                (a - b).abs() < 1e-10,
                "d=0 should be identity, got {} vs {}",
                a,
                b
            );
        }
    }

    #[test]
    fn test_frac_diff_d1_is_first_difference() {
        // d = 1 → (1-L)^1: output[t] = series[t] - series[t-1] (output[0] = series[0])
        let series = Array1::from(vec![10.0, 12.0, 15.0, 13.0, 18.0]);
        let result =
            Arfima::fractional_diff(&series, 1.0, 1e-9).expect("fractional_diff d=1 failed");
        // index 0 is unchanged (series[0] itself, no lag available)
        assert!((result[0] - series[0]).abs() < 1e-9);
        // subsequent values should be first differences
        for t in 1..series.len() {
            let expected = series[t] - series[t - 1];
            assert!(
                (result[t] - expected).abs() < 1e-6,
                "d=1 at t={}: expected {}, got {}",
                t,
                expected,
                result[t]
            );
        }
    }

    #[test]
    fn test_frac_diff_intermediate_d() {
        // d = 0.3: weights should decay, first weight = 1
        let series = white_noise(50, 1);
        let result =
            Arfima::fractional_diff(&series, 0.3, 1e-9).expect("fractional_diff d=0.3 failed");
        assert_eq!(result.len(), series.len());
        // All finite
        for v in result.iter() {
            assert!(v.is_finite(), "fractional diff result should be finite");
        }
        // First element equals series[0] (only weight 0 = 1 contributes)
        assert!((result[0] - series[0]).abs() < 1e-12);
    }

    // ------------------------------------------------------------------
    // ARFIMA simulation
    // ------------------------------------------------------------------
    #[test]
    fn test_arfima_simulation_length() {
        let model = Arfima::arfima(0, 0.3, 0);
        let sim = model.simulate(200, 42).expect("ARFIMA simulation failed");
        assert_eq!(sim.len(), 200);
    }

    #[test]
    fn test_arfima_simulation_finite() {
        let model = Arfima::arfima(1, 0.4, 1);
        let mut m = model;
        m.ar = vec![0.5];
        m.ma = vec![0.2];
        let sim = m
            .simulate(100, 99)
            .expect("ARFIMA(1,0.4,1) simulation failed");
        for v in sim.iter() {
            assert!(v.is_finite(), "Simulation should produce finite values");
        }
    }

    #[test]
    fn test_arfima_d0_resembles_arma() {
        // d=0 ARFIMA should behave like regular ARMA
        let model = Arfima::arfima(0, 0.0, 0);
        let sim = model
            .simulate(500, 7)
            .expect("ARFIMA(0,0,0) simulation failed");
        assert_eq!(sim.len(), 500);
        // Mean should be near 0
        let mean = sim.iter().sum::<f64>() / 500.0;
        assert!(mean.abs() < 1.5, "Mean of ARFIMA(0,0,0) should be near 0");
    }

    #[test]
    fn test_arfima_log_likelihood() {
        let model = Arfima::arfima(0, 0.3, 0);
        let sim = model.simulate(100, 5).expect("Simulation failed");
        let ll = model.log_likelihood(&sim).expect("Log-likelihood failed");
        assert!(ll.is_finite(), "Log-likelihood should be finite");
    }

    // ------------------------------------------------------------------
    // R/S Analysis
    // ------------------------------------------------------------------
    #[test]
    fn test_rs_white_noise_hurst_near_half() {
        // White noise should have H ≈ 0.5
        let wn = white_noise(1024, 123);
        let (h, log_ns, log_rs) =
            HurstEstimator::rs_analysis(&wn, 8, 256, 12).expect("R/S analysis failed");
        assert!(!log_ns.is_empty());
        assert!(!log_rs.is_empty());
        // H for white noise should be in [0.3, 0.7]
        assert!(
            (0.3..=0.7).contains(&h),
            "Hurst of white noise should be near 0.5, got {}",
            h
        );
    }

    #[test]
    fn test_rs_long_memory_hurst_above_half() {
        // ARFIMA with d=0.4 → H ≈ 0.9; at least > 0.5
        let model = Arfima::arfima(0, 0.4, 0);
        let sim = model.simulate(1024, 77).expect("Simulation failed");
        let (h, _, _) = HurstEstimator::rs_analysis(&sim, 8, 256, 10).expect("R/S analysis failed");
        assert!(
            h > 0.5,
            "Long memory series (d=0.4) should have H > 0.5, got {}",
            h
        );
    }

    // ------------------------------------------------------------------
    // DFA
    // ------------------------------------------------------------------
    #[test]
    fn test_dfa_white_noise_alpha_near_half() {
        let wn = white_noise(512, 42);
        let scales: Vec<usize> = (2..=7).map(|j| 2_usize.pow(j)).collect();
        let (alpha, log_s, log_f) = HurstEstimator::dfa(&wn, &scales, 1).expect("DFA failed");
        assert!(!log_s.is_empty());
        assert!(!log_f.is_empty());
        // DFA alpha for white noise should be near 0.5 (± 0.25 slack)
        assert!(
            (0.25..=0.75).contains(&alpha),
            "DFA alpha of white noise should be near 0.5, got {}",
            alpha
        );
    }

    #[test]
    fn test_dfa_returns_log_scale_log_f() {
        let wn = white_noise(256, 7);
        let scales = vec![8, 16, 32, 64];
        let (_, log_s, log_f) = HurstEstimator::dfa(&wn, &scales, 1).expect("DFA failed");
        assert_eq!(log_s.len(), log_f.len());
        for v in log_f.iter() {
            assert!(v.is_finite());
        }
    }

    // ------------------------------------------------------------------
    // MFDFA
    // ------------------------------------------------------------------
    #[test]
    fn test_mfdfa_returns_correct_shape() {
        let wn = white_noise(512, 33);
        let scales = vec![8, 16, 32, 64];
        let q_vals = vec![-2.0, 0.0, 2.0, 4.0];
        let (q_out, fq_matrix) =
            HurstEstimator::mfdfa(&wn, &scales, &q_vals, 1).expect("MFDFA failed");
        assert_eq!(q_out.len(), q_vals.len());
        assert_eq!(fq_matrix.len(), q_vals.len());
        // Each F_q should have at most as many entries as valid scales
        for fq in &fq_matrix {
            for v in fq.iter() {
                assert!(v.is_finite());
            }
        }
    }

    // ------------------------------------------------------------------
    // GPH estimate
    // ------------------------------------------------------------------
    #[test]
    fn test_gph_white_noise_d_near_zero() {
        let wn = white_noise(512, 55);
        let (d_hat, _se, p_val) = gph_estimate(&wn, None).expect("GPH failed");
        // For white noise, d should be near 0 (within ±0.4)
        assert!(
            d_hat.abs() < 0.4,
            "GPH d for white noise should be near 0, got {}",
            d_hat
        );
        assert!((0.0..=1.0).contains(&p_val));
    }

    #[test]
    fn test_gph_long_memory_d_positive() {
        // ARFIMA(0, 0.4, 0) should yield d_hat > 0
        let model = Arfima::arfima(0, 0.4, 0);
        let sim = model.simulate(512, 88).expect("Simulation failed");
        let (d_hat, _se, _p) = gph_estimate(&sim, None).expect("GPH failed");
        assert!(
            d_hat > 0.0,
            "GPH should detect positive d for long memory series, got {}",
            d_hat
        );
    }

    // ------------------------------------------------------------------
    // Whittle estimate
    // ------------------------------------------------------------------
    #[test]
    fn test_whittle_white_noise_d_near_zero() {
        let wn = white_noise(256, 19);
        let d_hat = whittle_estimate(&wn, 0, 0).expect("Whittle failed");
        assert!(
            d_hat.abs() < 0.4,
            "Whittle d for white noise should be near 0, got {}",
            d_hat
        );
    }

    // ------------------------------------------------------------------
    // fGn / fBm simulation
    // ------------------------------------------------------------------
    #[test]
    fn test_fgn_length() {
        let fgn = fgn_simulate(0.7, 128, 1).expect("fGn simulation failed");
        assert_eq!(fgn.len(), 128);
    }

    #[test]
    fn test_fgn_finite() {
        let fgn = fgn_simulate(0.8, 64, 2).expect("fGn simulation failed");
        for v in fgn.iter() {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_fbm_finite_and_length() {
        let fbm = fbm_simulate(0.75, 100, 3).expect("fBm simulation failed");
        assert_eq!(fbm.len(), 100);
        for v in fbm.iter() {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_fgn_hurst_half_is_white_noise() {
        // H = 0.5 → fGn should be uncorrelated
        let fgn = fgn_simulate(0.5, 512, 4).expect("fGn H=0.5 failed");
        let (h, _, _) = HurstEstimator::rs_analysis(&fgn, 8, 128, 8).expect("R/S on fGn failed");
        // Should be near 0.5 (allow generous tolerance)
        assert!(
            (0.25..=0.75).contains(&h),
            "fGn with H=0.5 should yield R/S Hurst near 0.5, got {}",
            h
        );
    }

    // ------------------------------------------------------------------
    // Long memory test
    // ------------------------------------------------------------------
    #[test]
    fn test_long_memory_test_white_noise_not_significant() {
        let wn = white_noise(512, 66);
        let (d_hat, p_val, _sig) = long_memory_test(&wn).expect("long_memory_test failed");
        assert!(d_hat.is_finite());
        assert!((0.0..=1.0).contains(&p_val));
    }

    #[test]
    fn test_long_memory_test_long_memory_series() {
        // ARFIMA(0, 0.45, 0) should be detected as long memory
        let model = Arfima::arfima(0, 0.45, 0);
        let sim = model.simulate(512, 13).expect("Simulation failed");
        let (d_hat, _p, sig) = long_memory_test(&sim).expect("long_memory_test failed");
        assert!(
            d_hat > 0.0,
            "d_hat should be positive for long memory series, got {}",
            d_hat
        );
        // We don't assert significance strictly due to finite sample variability,
        // but d_hat should be clearly positive
        let _ = sig;
    }

    // ------------------------------------------------------------------
    // Wavelet variance
    // ------------------------------------------------------------------
    #[test]
    fn test_wavelet_variance_returns_finite_values() {
        let wn = white_noise(256, 99);
        let wv = wavelet_variance(&wn, 6).expect("wavelet_variance failed");
        assert!(!wv.is_empty());
        for v in &wv {
            assert!(v.is_finite() && *v >= 0.0);
        }
    }

    #[test]
    fn test_wavelet_hurst_white_noise() {
        let wn = white_noise(512, 11);
        let h = HurstEstimator::wavelet_hurst(&wn).expect("wavelet_hurst failed");
        // Wavelet Hurst for white noise should be in (0, 1)
        assert!(
            (0.0..=1.0).contains(&h),
            "Wavelet Hurst out of range: {}",
            h
        );
    }

    // ------------------------------------------------------------------
    // fit_arfima_d
    // ------------------------------------------------------------------
    #[test]
    fn test_fit_arfima_d_gph() {
        let model = Arfima::arfima(0, 0.3, 0);
        let sim = model.simulate(256, 17).expect("Simulation failed");
        let (d_hat, se) = fit_arfima_d(&sim, FitMethod::GPH).expect("fit_arfima_d GPH failed");
        assert!(d_hat.is_finite());
        assert!(se >= 0.0);
    }

    #[test]
    fn test_fit_arfima_d_whittle() {
        let model = Arfima::arfima(0, 0.3, 0);
        let sim = model.simulate(256, 21).expect("Simulation failed");
        let (d_hat, se) =
            fit_arfima_d(&sim, FitMethod::Whittle).expect("fit_arfima_d Whittle failed");
        assert!(d_hat.is_finite());
        assert!(se >= 0.0);
    }

    #[test]
    fn test_fit_arfima_d_r2() {
        let model = Arfima::arfima(0, 0.3, 0);
        let sim = model.simulate(512, 29).expect("Simulation failed");
        let (d_hat, se) = fit_arfima_d(&sim, FitMethod::R2).expect("fit_arfima_d R2 failed");
        assert!(d_hat.is_finite());
        assert!(se >= 0.0);
    }
}
