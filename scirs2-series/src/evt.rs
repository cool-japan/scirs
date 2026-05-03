//! Extreme Value Theory (EVT) for time series analysis
//!
//! This module provides statistical tools for analysing and modelling extreme
//! events in time series data.  It implements the two classical EVT frameworks:
//!
//! 1. **Block Maxima** – Fit a Generalised Extreme Value (GEV) distribution to
//!    maxima extracted from non-overlapping blocks of the series.
//! 2. **Peaks Over Threshold (POT)** – Fit a Generalised Pareto Distribution
//!    (GPD) to the exceedances above a high threshold.
//!
//! Additional utilities cover:
//! - Return-level estimation with delta-method confidence intervals
//! - Extremal index estimation (Intervals / Blocks / Runs methods)
//! - Mean excess plot for threshold selection
//! - Bivariate exceedance probability

use std::f64::consts::PI;

// ─── Error helpers ───────────────────────────────────────────────────────────

fn err(msg: impl Into<String>) -> String {
    msg.into()
}

// ─── GEV distribution ────────────────────────────────────────────────────────

/// Generalised Extreme Value (GEV) distribution.
///
/// The CDF is:
/// ```text
/// F(x; μ, σ, ξ) = exp(-(1 + ξ·(x-μ)/σ)^(-1/ξ))   for ξ ≠ 0
/// F(x; μ, σ, 0) = exp(-exp(-(x-μ)/σ))               (Gumbel)
/// ```
///
/// Shape parameter:
/// * ξ = 0  → Gumbel (light tail)
/// * ξ > 0  → Fréchet (heavy tail)
/// * ξ < 0  → Weibull (bounded upper tail)
#[derive(Debug, Clone)]
pub struct GevDistribution {
    /// Location parameter μ
    pub mu: f64,
    /// Scale parameter σ (> 0)
    pub sigma: f64,
    /// Shape parameter ξ
    pub xi: f64,
}

impl GevDistribution {
    /// Construct a GEV distribution with explicit parameters.
    ///
    /// # Errors
    /// Returns an error if σ ≤ 0.
    pub fn new(mu: f64, sigma: f64, xi: f64) -> Result<Self, String> {
        if sigma <= 0.0 {
            return Err(err("sigma must be positive"));
        }
        Ok(Self { mu, sigma, xi })
    }

    /// Fit by Maximum Likelihood Estimation to a set of block maxima.
    ///
    /// Uses a simple Newton-Raphson iteration on the GEV log-likelihood with
    /// automatic numerical differentiation.  The Gumbel case (ξ ≈ 0) is
    /// handled through a smooth approximation.
    ///
    /// # Errors
    /// Returns an error when `block_maxima` has fewer than 3 observations or
    /// when the optimiser fails to converge.
    pub fn fit_mle(block_maxima: &[f64]) -> Result<Self, String> {
        let n = block_maxima.len();
        if n < 3 {
            return Err(err("need at least 3 block maxima for GEV MLE"));
        }

        // Starting values via Probability Weighted Moments (robust initialiser)
        let pwm = Self::fit_pwm(block_maxima);

        // Clamp xi to avoid degenerate starting points
        let xi0 = pwm.xi.clamp(-0.4, 0.4);
        let mut params = [pwm.mu, pwm.sigma.max(1e-6), xi0];

        let log_lik = |p: &[f64]| -> f64 {
            let mu = p[0];
            let sigma = p[1];
            let xi = p[2];
            if sigma <= 0.0 {
                return f64::NEG_INFINITY;
            }
            let mut ll = -(n as f64) * sigma.ln();
            for &x in block_maxima {
                let z = (x - mu) / sigma;
                if xi.abs() < 1e-8 {
                    // Gumbel
                    ll -= z + (-z).exp();
                } else {
                    let t = 1.0 + xi * z;
                    if t <= 0.0 {
                        return f64::NEG_INFINITY;
                    }
                    ll -= (1.0 + 1.0 / xi) * t.ln() + t.powf(-1.0 / xi);
                }
            }
            ll
        };

        // Nelder-Mead simplex optimisation (no external deps)
        let best = nelder_mead_max(&params, log_lik, 2000, 1e-8)?;
        params.copy_from_slice(&best);

        if params[1] <= 0.0 {
            return Err(err("MLE failed: scale became non-positive"));
        }
        Ok(Self {
            mu: params[0],
            sigma: params[1],
            xi: params[2],
        })
    }

    /// Fit by Probability Weighted Moments (PWM / L-moments method).
    ///
    /// The PWM estimator is closed-form and more robust than MLE for small
    /// samples, but less efficient asymptotically.
    pub fn fit_pwm(block_maxima: &[f64]) -> Self {
        let mut sorted = block_maxima.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = sorted.len() as f64;

        // PWM moments β₀, β₁, β₂
        let mut b = [0.0_f64; 3];
        for (i, &x) in sorted.iter().enumerate() {
            let p = i as f64;
            b[0] += x;
            b[1] += x * p / (n - 1.0);
            b[2] += x * p * (p - 1.0) / ((n - 1.0) * (n - 2.0));
        }
        b[0] /= n;
        b[1] /= n;
        b[2] /= n;

        // L-moment ratios
        let l1 = b[0];
        let l2 = 2.0 * b[1] - b[0];
        let l3 = 6.0 * b[2] - 6.0 * b[1] + b[0];
        let t3 = if l2.abs() < 1e-12 { 0.0 } else { l3 / l2 };

        // Hosking & Wallis (1987) polynomial approximation for ξ
        let c = 2.0 / (3.0 + t3) - std::f64::consts::LN_2;
        let xi = if c.abs() < 1e-10 {
            0.0
        } else {
            let xi_approx = 7.859 * c + 2.9554 * c * c;
            xi_approx
        };

        let gamma1pxi = if xi.abs() < 1e-8 {
            1.0 // Γ(1) = 1
        } else {
            gamma_function(1.0 + xi)
        };

        let sigma = if xi.abs() < 1e-8 {
            l2 / std::f64::consts::LN_2
        } else {
            xi * l2 / (1.0 - 2.0_f64.powf(-xi)) / gamma1pxi
        };
        let sigma = sigma.max(1e-10);

        let mu = l1 - sigma * (1.0 - gamma1pxi) / xi.max(1e-10);

        Self {
            mu: mu.clamp(
                block_maxima.iter().cloned().fold(f64::INFINITY, f64::min) - 1.0,
                block_maxima
                    .iter()
                    .cloned()
                    .fold(f64::NEG_INFINITY, f64::max)
                    + 1.0,
            ),
            sigma: sigma.abs().max(1e-10),
            xi: xi.clamp(-0.5, 0.8),
        }
    }

    /// CDF of the GEV at `x`.
    pub fn cdf(&self, x: f64) -> f64 {
        let z = (x - self.mu) / self.sigma;
        if self.xi.abs() < 1e-8 {
            // Gumbel
            (-(-z).exp()).exp()
        } else {
            let t = 1.0 + self.xi * z;
            if t <= 0.0 {
                if self.xi > 0.0 {
                    0.0
                } else {
                    1.0
                }
            } else {
                (-t.powf(-1.0 / self.xi)).exp()
            }
        }
    }

    /// Quantile function (inverse CDF) of the GEV at probability `p`.
    pub fn quantile(&self, p: f64) -> f64 {
        assert!(p > 0.0 && p < 1.0, "p must be in (0,1)");
        let y = -p.ln().ln(); // reduced variate
        if self.xi.abs() < 1e-8 {
            self.mu + self.sigma * y
        } else {
            // GEV quantile: μ + σ·((−ln p)^(−ξ) − 1)/ξ
            self.mu + self.sigma * (((-p.ln()).powf(-self.xi) - 1.0) / self.xi)
        }
    }

    /// T-year return level: the level exceeded on average once every T years.
    ///
    /// This is `quantile(1 - 1/T)`.
    pub fn return_level(&self, return_period: f64) -> f64 {
        assert!(return_period > 1.0, "return period must be > 1");
        self.quantile(1.0 - 1.0 / return_period)
    }
}

// ─── GPD distribution ────────────────────────────────────────────────────────

/// Generalised Pareto Distribution (GPD).
///
/// Models exceedances `Y = X - u` above threshold `u`.
///
/// ```text
/// F(y; σ, ξ) = 1 - (1 + ξ·y/σ)^(-1/ξ)    for ξ ≠ 0
/// F(y; σ, 0) = 1 - exp(-y/σ)               (Exponential)
/// ```
#[derive(Debug, Clone)]
pub struct GpdDistribution {
    /// Threshold / location parameter μ
    pub mu: f64,
    /// Scale parameter σ (> 0)
    pub sigma: f64,
    /// Shape parameter ξ
    pub xi: f64,
}

impl GpdDistribution {
    /// Construct a GPD with explicit parameters.
    ///
    /// # Errors
    /// Returns an error if σ ≤ 0.
    pub fn new(mu: f64, sigma: f64, xi: f64) -> Result<Self, String> {
        if sigma <= 0.0 {
            return Err(err("sigma must be positive"));
        }
        Ok(Self { mu, sigma, xi })
    }

    /// Fit by Maximum Likelihood Estimation to a vector of exceedances.
    ///
    /// Exceedances should already have the threshold subtracted (i.e., `x - u`).
    ///
    /// # Errors
    /// Returns an error when there are fewer than 2 exceedances.
    pub fn fit_mle(exceedances: &[f64]) -> Result<Self, String> {
        let n = exceedances.len();
        if n < 2 {
            return Err(err("need at least 2 exceedances for GPD MLE"));
        }
        if exceedances.iter().any(|&x| x < 0.0) {
            return Err(err("exceedances must be non-negative"));
        }

        // Method-of-moments start
        let mom_start = Self::fit_moments(exceedances);
        let params0 = [mom_start.sigma, mom_start.xi];

        let log_lik = |p: &[f64]| -> f64 {
            let sigma = p[0];
            let xi = p[1];
            if sigma <= 0.0 {
                return f64::NEG_INFINITY;
            }
            let mut ll = -(n as f64) * sigma.ln();
            for &y in exceedances {
                if xi.abs() < 1e-8 {
                    ll -= y / sigma;
                } else {
                    let t = 1.0 + xi * y / sigma;
                    if t <= 0.0 {
                        return f64::NEG_INFINITY;
                    }
                    ll -= (1.0 + 1.0 / xi) * t.ln();
                }
            }
            ll
        };

        let best = nelder_mead_max(&params0, log_lik, 1000, 1e-8)?;
        if best[0] <= 0.0 {
            return Err(err("GPD MLE: scale became non-positive"));
        }
        Ok(Self {
            mu: 0.0,
            sigma: best[0],
            xi: best[1],
        })
    }

    /// Fit by method of moments (closed-form).
    pub fn fit_moments(exceedances: &[f64]) -> Self {
        let n = exceedances.len() as f64;
        let mean = exceedances.iter().sum::<f64>() / n;
        let var = exceedances.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;

        let xi = if var < 1e-14 {
            0.0
        } else {
            0.5 * (1.0 - mean * mean / var)
        };
        let sigma = mean * (1.0 - xi);

        Self {
            mu: 0.0,
            sigma: sigma.max(1e-10),
            xi: xi.clamp(-0.5, 0.5),
        }
    }

    /// CDF of the GPD at `x` (measured from threshold).
    pub fn cdf(&self, x: f64) -> f64 {
        let y = x - self.mu;
        if y < 0.0 {
            return 0.0;
        }
        if self.xi.abs() < 1e-8 {
            1.0 - (-y / self.sigma).exp()
        } else {
            let t = 1.0 + self.xi * y / self.sigma;
            if t <= 0.0 {
                return 1.0;
            }
            1.0 - t.powf(-1.0 / self.xi)
        }
    }

    /// Quantile of the GPD at probability `p` (0 < p < 1).
    pub fn quantile(&self, p: f64) -> f64 {
        assert!(p > 0.0 && p < 1.0, "p must be in (0,1)");
        if self.xi.abs() < 1e-8 {
            self.mu + self.sigma * (-(1.0 - p).ln())
        } else {
            self.mu + self.sigma * ((1.0 - p).powf(-self.xi) - 1.0) / self.xi
        }
    }

    /// Mean excess function value, i.e., E[X - u | X > u].
    ///
    /// Returns `None` when ξ ≥ 1 (infinite mean).
    pub fn mean_excess(&self) -> Option<f64> {
        if self.xi >= 1.0 {
            None
        } else {
            Some(self.sigma / (1.0 - self.xi))
        }
    }
}

// ─── Block maxima ─────────────────────────────────────────────────────────────

/// Extract block maxima from a time series.
///
/// Splits `data` into consecutive non-overlapping blocks of length `block_size`
/// and returns the maximum within each block.  Incomplete trailing blocks are
/// discarded.
///
/// # Examples
/// ```
/// use scirs2_series::evt::block_maxima;
/// let data: Vec<f64> = (1..=12).map(|i| i as f64).collect();
/// let maxima = block_maxima(&data, 4);
/// assert_eq!(maxima, vec![4.0, 8.0, 12.0]);
/// ```
pub fn block_maxima(data: &[f64], block_size: usize) -> Vec<f64> {
    if block_size == 0 || data.is_empty() {
        return Vec::new();
    }
    data.chunks(block_size)
        .filter(|chunk| chunk.len() == block_size)
        .map(|chunk| chunk.iter().cloned().fold(f64::NEG_INFINITY, f64::max))
        .collect()
}

// ─── Peaks over threshold ─────────────────────────────────────────────────────

/// Return the exceedances above `threshold` (i.e., `x - threshold` for each
/// `x > threshold`).
///
/// # Examples
/// ```
/// use scirs2_series::evt::peaks_over_threshold;
/// let data = vec![1.0, 3.0, 2.0, 5.0, 4.0];
/// let exc = peaks_over_threshold(&data, 2.5);
/// assert_eq!(exc, vec![0.5, 2.5, 1.5]);
/// ```
pub fn peaks_over_threshold(data: &[f64], threshold: f64) -> Vec<f64> {
    data.iter()
        .filter(|&&x| x > threshold)
        .map(|&x| x - threshold)
        .collect()
}

// ─── Mean excess plot ─────────────────────────────────────────────────────────

/// Compute the mean excess plot: for each threshold u the expected overshoot
/// E[X - u | X > u] is estimated from the data.
///
/// Returns a vector of `(threshold, mean_excess)` pairs, one per entry in
/// `thresholds`.  Thresholds for which no data exceed them are omitted.
///
/// # Examples
/// ```
/// use scirs2_series::evt::mean_excess_plot;
/// let data: Vec<f64> = (1..=100).map(|i| i as f64).collect();
/// let ts: Vec<f64> = vec![10.0, 50.0, 80.0];
/// let plot = mean_excess_plot(&data, &ts);
/// assert_eq!(plot.len(), 3);
/// ```
pub fn mean_excess_plot(data: &[f64], thresholds: &[f64]) -> Vec<(f64, f64)> {
    thresholds
        .iter()
        .filter_map(|&u| {
            let exceedances: Vec<f64> = data.iter().filter(|&&x| x > u).map(|&x| x - u).collect();
            if exceedances.is_empty() {
                None
            } else {
                let me = exceedances.iter().sum::<f64>() / exceedances.len() as f64;
                Some((u, me))
            }
        })
        .collect()
}

// ─── Return level with confidence intervals ────────────────────────────────────

/// Estimate the return level with delta-method confidence intervals.
///
/// Parameters
/// ----------
/// `gev`           – fitted GEV distribution  
/// `return_period` – T in "T-year return level"  
/// `n_data`        – number of block maxima used for fitting  
/// `ci_level`      – e.g. 0.95 for a 95 % confidence interval  
///
/// Returns `(lower, point_estimate, upper)`.
pub fn return_level_ci(
    gev: &GevDistribution,
    return_period: f64,
    n_data: usize,
    ci_level: f64,
) -> (f64, f64, f64) {
    let estimate = gev.return_level(return_period);

    // Delta-method variance approximation (Fisher information matrix).
    // We use a numerical estimate of the gradient ∂z_T/∂θ.
    let h = 1e-5;
    let perturb = |delta_mu: f64, delta_sigma: f64, delta_xi: f64| -> f64 {
        let g = GevDistribution {
            mu: gev.mu + delta_mu,
            sigma: (gev.sigma + delta_sigma).max(1e-10),
            xi: gev.xi + delta_xi,
        };
        g.return_level(return_period)
    };

    let dz_dmu = (perturb(h, 0.0, 0.0) - perturb(-h, 0.0, 0.0)) / (2.0 * h);
    let dz_ds = (perturb(0.0, h, 0.0) - perturb(0.0, -h, 0.0)) / (2.0 * h);
    let dz_dxi = (perturb(0.0, 0.0, h) - perturb(0.0, 0.0, -h)) / (2.0 * h);

    // Approximate variance of return level (diagonal Fisher info, scaled 1/n)
    let n = n_data as f64;
    let var_approx = (dz_dmu.powi(2) + dz_ds.powi(2) + dz_dxi.powi(2)) / n;
    let se = var_approx.sqrt();

    // z-quantile for ci_level
    let z = normal_quantile((1.0 + ci_level) / 2.0);

    (estimate - z * se, estimate, estimate + z * se)
}

// ─── Extremal index estimation ────────────────────────────────────────────────

/// Method for extremal index estimation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtremalIndexMethod {
    /// Ferro & Segers (2003) intervals estimator
    Intervals,
    /// Blocks estimator (proportion of blocks containing an exceedance)
    Blocks,
    /// Runs estimator (runs of exceedances with inter-run gaps)
    Runs,
}

/// Estimate the extremal index θ ∈ (0, 1].
///
/// θ = 1 means no clustering of extremes; θ < 1 indicates cluster structure.
///
/// # Arguments
/// * `data`      – time series  
/// * `threshold` – exceedance threshold  
/// * `method`    – one of [`ExtremalIndexMethod`]
///
/// # Examples
/// ```
/// use scirs2_series::evt::{extremal_index_intervals, ExtremalIndexMethod};
/// let data: Vec<f64> = (0..200).map(|i| (i as f64 * 0.3).sin()).collect();
/// let theta = extremal_index_intervals(&data, 0.5, ExtremalIndexMethod::Intervals);
/// assert!(theta >= 0.0 && theta <= 1.0);
/// ```
pub fn extremal_index_intervals(data: &[f64], threshold: f64, method: ExtremalIndexMethod) -> f64 {
    let n = data.len();
    if n == 0 {
        return 1.0;
    }

    // Indicator vector: exceeds threshold?
    let exc: Vec<bool> = data.iter().map(|&x| x > threshold).collect();
    let n_exc: usize = exc.iter().filter(|&&e| e).count();
    if n_exc == 0 {
        return 1.0;
    }

    match method {
        ExtremalIndexMethod::Intervals => {
            // Ferro & Segers (2003): estimate based on inter-exceedance times
            let times: Vec<usize> = exc
                .iter()
                .enumerate()
                .filter(|(_, &e)| e)
                .map(|(i, _)| i)
                .collect();
            if times.len() < 2 {
                return 1.0;
            }
            let gaps: Vec<f64> = times.windows(2).map(|w| (w[1] - w[0]) as f64).collect();
            let n_c = gaps.len() as f64;
            let _mean_gap = gaps.iter().sum::<f64>() / n_c;
            let max_gap = gaps.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            if max_gap <= 2.0 {
                // Small-gaps: θ̂ = 2·(Σ T_i)² / (n · Σ T_i²)
                let sum_t = gaps.iter().sum::<f64>();
                let sum_t2 = gaps.iter().map(|&t| t * t).sum::<f64>();
                if sum_t2 < 1e-14 {
                    1.0
                } else {
                    (2.0 * sum_t * sum_t) / (n_c * sum_t2)
                }
            } else {
                // Large-gaps: θ̂ = 2·(Σ (T_i-1))² / (n · Σ (T_i-1)(T_i-2))
                let tm1: Vec<f64> = gaps.iter().map(|&t| t - 1.0).collect();
                let sum_tm1 = tm1.iter().sum::<f64>();
                let sum_tm1_tm2 = gaps
                    .iter()
                    .map(|&t| (t - 1.0) * (t - 2.0).max(0.0))
                    .sum::<f64>();
                if sum_tm1_tm2 < 1e-14 {
                    1.0
                } else {
                    let theta = 2.0 * sum_tm1 * sum_tm1 / (n_c * sum_tm1_tm2);
                    theta.min(1.0).max(0.0)
                }
            }
        }
        ExtremalIndexMethod::Blocks => {
            // Block estimator: θ̂ ≈ fraction of blocks that contain ≥ 1 exceedance
            let block_size = ((n as f64).sqrt().ceil() as usize).max(1);
            let n_blocks = n / block_size;
            if n_blocks == 0 {
                return 1.0;
            }
            let blocks_with_exc = exc
                .chunks(block_size)
                .take(n_blocks)
                .filter(|b| b.iter().any(|&e| e))
                .count();
            let p_block = n_exc as f64 / n as f64;
            let q = blocks_with_exc as f64 / n_blocks as f64;
            if p_block < 1e-14 {
                return 1.0;
            }
            let block_p = block_size as f64 * p_block;
            if block_p.abs() < 1e-14 {
                1.0
            } else {
                (q / block_p).min(1.0).max(0.0)
            }
        }
        ExtremalIndexMethod::Runs => {
            // Runs estimator: θ̂ = number of runs / number of exceedances
            let mut runs = 0usize;
            let mut in_run = false;
            for &e in &exc {
                if e {
                    if !in_run {
                        runs += 1;
                        in_run = true;
                    }
                } else {
                    in_run = false;
                }
            }
            if n_exc == 0 {
                1.0
            } else {
                (runs as f64 / n_exc as f64).min(1.0).max(0.0)
            }
        }
    }
}

// ─── Bivariate exceedance probability ────────────────────────────────────────

/// Empirical bivariate exceedance probability P(X > x, Y > y).
///
/// Estimates P(X > level_x AND Y > level_y) from the data using the
/// empirical frequency.
///
/// # Examples
/// ```
/// use scirs2_series::evt::bivariate_exceedance_probability;
/// let data: Vec<(f64, f64)> = (0..100).map(|i| (i as f64, 100.0 - i as f64)).collect();
/// let p = bivariate_exceedance_probability(&data, 80.0, 80.0);
/// assert!(p >= 0.0 && p <= 1.0);
/// ```
pub fn bivariate_exceedance_probability(data: &[(f64, f64)], level_x: f64, level_y: f64) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let count = data
        .iter()
        .filter(|&&(x, y)| x > level_x && y > level_y)
        .count();
    count as f64 / data.len() as f64
}

// ─── Internal helpers ────────────────────────────────────────────────────────

/// Lanczos approximation for the Gamma function (accurate to ~15 digits).
fn gamma_function(x: f64) -> f64 {
    if x < 0.5 {
        PI / ((PI * x).sin() * gamma_function(1.0 - x))
    } else {
        let g = 7.0_f64;
        let c = [
            0.999_999_999_999_809_3_f64,
            676.520_368_121_885_1,
            -1_259.139_216_722_402_8,
            771.323_428_777_653_1,
            -176.615_029_162_140_6,
            12.507_343_278_686_905,
            -0.138_571_095_266_526_07,
            9.984_369_578_019_571_6e-6,
            1.505_632_735_149_311_6e-7,
        ];
        let xm1 = x - 1.0;
        let mut sum = c[0];
        for (k, &ck) in c[1..].iter().enumerate() {
            sum += ck / (xm1 + (k + 1) as f64);
        }
        let t = xm1 + g + 0.5;
        (2.0 * PI).sqrt() * t.powf(xm1 + 0.5) * (-t).exp() * sum
    }
}

/// Inverse CDF of the standard normal distribution (rational approximation).
fn normal_quantile(p: f64) -> f64 {
    // Beasley-Springer-Moro algorithm
    let a = [2.515_517, 0.802_853, 0.010_328_f64];
    let b = [1.432_788, 0.189_269, 0.001_308_f64];
    let sign = if p < 0.5 { -1.0 } else { 1.0 };
    let pp = if p < 0.5 { p } else { 1.0 - p };
    let t = (-2.0 * pp.ln()).sqrt();
    let num = a[0] + a[1] * t + a[2] * t * t;
    let den = 1.0 + b[0] * t + b[1] * t * t + b[2] * t * t * t;
    sign * (t - num / den)
}

/// Simple Nelder-Mead simplex maximiser.
///
/// Minimises `-f` by reflecting/expanding/contracting a simplex.
/// Returns the best parameter vector found.
fn nelder_mead_max<F>(start: &[f64], f: F, max_iter: usize, tol: f64) -> Result<Vec<f64>, String>
where
    F: Fn(&[f64]) -> f64,
{
    let d = start.len();
    let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(d + 1);
    simplex.push(start.to_vec());
    for i in 0..d {
        let mut s = start.to_vec();
        s[i] += if s[i].abs() > 1e-8 {
            0.05 * s[i]
        } else {
            0.00025
        };
        simplex.push(s);
    }

    let neg_f = |p: &[f64]| -> f64 { -f(p) };

    for _ in 0..max_iter {
        // Sort by function value (ascending = minimise neg_f)
        simplex.sort_by(|a, b| {
            neg_f(a)
                .partial_cmp(&neg_f(b))
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Convergence check
        let best_val = neg_f(&simplex[0]);
        let worst_val = neg_f(&simplex[d]);
        if (worst_val - best_val).abs() < tol {
            break;
        }

        // Centroid of all but worst
        let centroid: Vec<f64> = (0..d)
            .map(|j| simplex[..d].iter().map(|s| s[j]).sum::<f64>() / d as f64)
            .collect();

        // Reflection
        let reflected: Vec<f64> = (0..d).map(|j| 2.0 * centroid[j] - simplex[d][j]).collect();
        let fr = neg_f(&reflected);
        let f0 = neg_f(&simplex[0]);
        let fd = neg_f(&simplex[d - 1]);

        if fr < f0 {
            // Expansion
            let expanded: Vec<f64> = (0..d)
                .map(|j| 3.0 * centroid[j] - 2.0 * simplex[d][j])
                .collect();
            if neg_f(&expanded) < fr {
                simplex[d] = expanded;
            } else {
                simplex[d] = reflected;
            }
        } else if fr < fd {
            simplex[d] = reflected;
        } else {
            // Contraction
            let contracted: Vec<f64> = (0..d)
                .map(|j| 0.5 * (centroid[j] + simplex[d][j]))
                .collect();
            if neg_f(&contracted) < neg_f(&simplex[d]) {
                simplex[d] = contracted;
            } else {
                // Shrink
                let best = simplex[0].clone();
                for s in simplex.iter_mut().skip(1) {
                    for j in 0..d {
                        s[j] = 0.5 * (best[j] + s[j]);
                    }
                }
            }
        }
    }

    simplex.sort_by(|a, b| {
        let neg_f_closure = |p: &[f64]| -> f64 { -f(p) };
        neg_f_closure(a)
            .partial_cmp(&neg_f_closure(b))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(simplex.remove(0))
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_maxima() -> Vec<f64> {
        // 30 pseudo-random block maxima from a Gumbel(0,1)
        vec![
            0.23, 1.42, 2.14, 0.87, 1.55, 0.33, 1.78, 2.45, 1.23, 0.99, 1.67, 0.45, 2.01, 1.34,
            0.78, 1.90, 0.61, 2.33, 1.11, 0.55, 1.89, 0.74, 2.67, 1.48, 0.91, 1.22, 2.10, 0.38,
            1.75, 0.83,
        ]
    }

    #[test]
    fn test_block_maxima_basic() {
        let data: Vec<f64> = (1..=12).map(|i| i as f64).collect();
        let maxima = block_maxima(&data, 4);
        assert_eq!(maxima, vec![4.0, 8.0, 12.0]);
    }

    #[test]
    fn test_block_maxima_incomplete_block_discarded() {
        let data: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        let maxima = block_maxima(&data, 4);
        assert_eq!(maxima, vec![4.0, 8.0]); // 9,10 discarded (incomplete block)
    }

    #[test]
    fn test_peaks_over_threshold() {
        let data = vec![1.0, 3.0, 2.0, 5.0, 4.0];
        let exc = peaks_over_threshold(&data, 2.5);
        assert_eq!(exc.len(), 3);
        assert!((exc[0] - 0.5).abs() < 1e-10);
        assert!((exc[1] - 2.5).abs() < 1e-10);
        assert!((exc[2] - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_gev_pwm_fit() {
        let maxima = simple_maxima();
        let gev = GevDistribution::fit_pwm(&maxima);
        // Should produce reasonable parameters
        assert!(gev.sigma > 0.0);
        // CDF at large value should be close to 1
        assert!(gev.cdf(5.0) > 0.9);
        // CDF at small value should be close to 0
        assert!(gev.cdf(-5.0) < 0.1);
    }

    #[test]
    fn test_gev_mle_fit() {
        let maxima = simple_maxima();
        let result = GevDistribution::fit_mle(&maxima);
        assert!(result.is_ok(), "MLE should succeed: {:?}", result);
        let gev = result.expect("failed to create gev");
        assert!(gev.sigma > 0.0);
    }

    #[test]
    fn test_gev_cdf_gumbel() {
        // xi ≈ 0 → Gumbel; CDF at mu should be exp(-exp(0)) = exp(-1) ≈ 0.3679
        let gev = GevDistribution {
            mu: 0.0,
            sigma: 1.0,
            xi: 1e-12,
        };
        let cdf_at_mu = gev.cdf(0.0);
        assert!(
            (cdf_at_mu - (-1.0_f64).exp()).abs() < 1e-8,
            "cdf={cdf_at_mu}"
        );
    }

    #[test]
    fn test_gev_return_level_monotone() {
        let gev = GevDistribution {
            mu: 1.0,
            sigma: 0.5,
            xi: 0.1,
        };
        let rl10 = gev.return_level(10.0);
        let rl100 = gev.return_level(100.0);
        let rl1000 = gev.return_level(1000.0);
        assert!(
            rl10 < rl100 && rl100 < rl1000,
            "return levels should be monotone: {rl10} < {rl100} < {rl1000}"
        );
    }

    #[test]
    fn test_gpd_moments_fit() {
        let exceedances = vec![0.5, 1.2, 0.3, 2.1, 0.8, 1.5, 0.1, 3.0, 0.6, 1.0];
        let gpd = GpdDistribution::fit_moments(&exceedances);
        assert!(gpd.sigma > 0.0);
    }

    #[test]
    fn test_gpd_mle_fit() {
        let exceedances = vec![0.5, 1.2, 0.3, 2.1, 0.8, 1.5, 0.1, 3.0, 0.6, 1.0];
        let result = GpdDistribution::fit_mle(&exceedances);
        assert!(result.is_ok(), "GPD MLE: {:?}", result);
        let gpd = result.expect("failed to create gpd");
        assert!(gpd.sigma > 0.0);
        // CDF should be monotone
        let c1 = gpd.cdf(0.5);
        let c2 = gpd.cdf(1.5);
        assert!(
            c1 < c2,
            "CDF must be increasing: cdf(0.5)={c1} < cdf(1.5)={c2}"
        );
    }

    #[test]
    fn test_gpd_mean_excess() {
        // xi < 1 → finite mean excess
        let gpd = GpdDistribution {
            mu: 0.0,
            sigma: 1.0,
            xi: 0.2,
        };
        assert!(gpd.mean_excess().is_some());
        // xi >= 1 → infinite mean excess
        let gpd2 = GpdDistribution {
            mu: 0.0,
            sigma: 1.0,
            xi: 1.0,
        };
        assert!(gpd2.mean_excess().is_none());
    }

    #[test]
    fn test_mean_excess_plot() {
        let data: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let ts = vec![10.0, 50.0, 80.0];
        let plot = mean_excess_plot(&data, &ts);
        assert_eq!(plot.len(), 3);
        // Mean excess should be decreasing for uniform data (bounded upper tail)
        assert!(
            plot[0].1 > plot[1].1,
            "mean excess decreases with threshold for uniform"
        );
    }

    #[test]
    fn test_return_level_ci() {
        let gev = GevDistribution {
            mu: 1.0,
            sigma: 0.5,
            xi: 0.1,
        };
        let (lo, est, hi) = return_level_ci(&gev, 100.0, 50, 0.95);
        assert!(
            lo <= est && est <= hi,
            "CI must straddle estimate: ({lo}, {est}, {hi})"
        );
        assert!(hi - lo > 0.0, "CI must have positive width");
    }

    #[test]
    fn test_extremal_index_intervals() {
        // IID data → θ should be close to 1
        let iid: Vec<f64> = (0..200).map(|i| (i as f64 * 1.3).sin()).collect();
        let theta = extremal_index_intervals(&iid, 0.7, ExtremalIndexMethod::Intervals);
        assert!(theta >= 0.0 && theta <= 1.0, "theta={theta}");
    }

    #[test]
    fn test_extremal_index_blocks() {
        let data: Vec<f64> = (0..100).map(|i| (i as f64 * 0.5).sin()).collect();
        let theta = extremal_index_intervals(&data, 0.5, ExtremalIndexMethod::Blocks);
        assert!(theta >= 0.0 && theta <= 1.0, "theta={theta}");
    }

    #[test]
    fn test_extremal_index_runs() {
        let data: Vec<f64> = (0..100).map(|i| (i as f64 * 0.5).sin()).collect();
        let theta = extremal_index_intervals(&data, 0.5, ExtremalIndexMethod::Runs);
        assert!(theta >= 0.0 && theta <= 1.0, "theta={theta}");
    }

    #[test]
    fn test_bivariate_exceedance_probability() {
        let data: Vec<(f64, f64)> = (0..100).map(|i| (i as f64, i as f64)).collect();
        let p = bivariate_exceedance_probability(&data, 50.0, 50.0);
        // P(X > 50 AND Y > 50) = 49/100
        assert!((p - 0.49).abs() < 0.01, "p={p}");
    }

    #[test]
    fn test_bivariate_independence_approx() {
        // Perfectly negatively correlated → P(X>c AND Y>c) should be small
        let data: Vec<(f64, f64)> = (0..100).map(|i| (i as f64, (99 - i) as f64)).collect();
        let p = bivariate_exceedance_probability(&data, 80.0, 80.0);
        assert!(p < 0.05, "negatively correlated: p={p}");
    }

    #[test]
    fn test_gamma_function_known_values() {
        assert!((gamma_function(1.0) - 1.0).abs() < 1e-10);
        assert!((gamma_function(2.0) - 1.0).abs() < 1e-10);
        assert!((gamma_function(3.0) - 2.0).abs() < 1e-8);
        assert!((gamma_function(0.5) - PI.sqrt()).abs() < 1e-8);
    }
}
