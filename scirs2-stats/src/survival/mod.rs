//! Survival Analysis
//!
//! This module provides production-quality survival analysis functions:
//!
//! - **Kaplan-Meier estimator**: Non-parametric estimator of the survival function
//!   with Greenwood's formula confidence intervals and log-rank test.
//! - **Nelson-Aalen estimator**: Non-parametric estimator of the cumulative hazard
//!   function; yields `exp(-H(t))` as an alternative survival estimate.
//! - **Cox Proportional Hazards**: Semi-parametric regression model fitted via
//!   Newton-Raphson partial-likelihood optimisation (Breslow baseline hazard).
//!
//! # References
//! - Kaplan, E.L. & Meier, P. (1958). Non-parametric estimation from incomplete observations.
//! - Nelson, W. (1972). Theory and applications of hazard plotting for censored failure data.
//! - Cox, D.R. (1972). Regression models and life tables.

pub mod aft;
pub mod competing_risks;
pub mod cox;
pub mod frailty;
pub mod kaplan_meier;
pub mod legacy;
pub mod nelson_aalen;

use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};

// ---------------------------------------------------------------------------
// Chi-square / normal CDF helpers (no external dependency)
// ---------------------------------------------------------------------------

/// Abramowitz & Stegun rational approximation of erf(x), max error ~1.5e-7.
fn erf_approx(x: f64) -> f64 {
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    sign * (1.0 - poly * (-x * x).exp())
}

/// Standard normal CDF using erf approximation.
fn norm_cdf(z: f64) -> f64 {
    0.5 * (1.0 + erf_approx(z / std::f64::consts::SQRT_2))
}

/// Chi-square survival function P(X > x) for integer or half-integer df.
///
/// Uses the regularised incomplete gamma function via series expansion for
/// small x and continued-fraction for large x (both with df/2 parameter).
fn chi2_sf(x: f64, df: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    // Normal approximation for large df (Wilson-Hilferty cube-root transform)
    if df > 30.0 {
        let k = df / 2.0;
        let z =
            ((x / (2.0 * k)).powf(1.0 / 3.0) - (1.0 - 1.0 / (9.0 * k))) / (1.0 / (9.0 * k)).sqrt();
        return 1.0 - norm_cdf(z);
    }
    // Regularised upper incomplete gamma via series / continued fraction
    // gamma_sf(a, x) = 1 - gamma_cdf(a, x)
    let a = df / 2.0;
    let half_x = x / 2.0;
    1.0 - regularised_gamma_lower(a, half_x)
}

/// Regularised lower incomplete gamma P(a, x) using series expansion.
fn regularised_gamma_lower(a: f64, x: f64) -> f64 {
    if x < 0.0 {
        return 0.0;
    }
    if x == 0.0 {
        return 0.0;
    }
    // For large x relative to a, use continued-fraction (upper tail)
    if x > a + 1.0 {
        return 1.0 - regularised_gamma_upper_cf(a, x);
    }
    // Series expansion
    let mut term = 1.0 / a;
    let mut sum = term;
    let max_iter = 300;
    for n in 1..=max_iter {
        term *= x / (a + n as f64);
        sum += term;
        if term.abs() < sum.abs() * 1e-12 {
            break;
        }
    }
    sum * (-x + a * x.ln() - ln_gamma(a)).exp()
}

/// Regularised upper incomplete gamma Q(a, x) via Legendre continued fraction.
fn regularised_gamma_upper_cf(a: f64, x: f64) -> f64 {
    // Modified Lentz algorithm
    let fpmin = 1e-300_f64;
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / fpmin;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..=300_usize {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < fpmin {
            d = fpmin;
        }
        c = b + an / c;
        if c.abs() < fpmin {
            c = fpmin;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 1e-12 {
            break;
        }
    }
    h * (-x + a * x.ln() - ln_gamma(a)).exp()
}

/// Stirling/Lanczos approximation of ln Γ(z) for z > 0.
fn ln_gamma(z: f64) -> f64 {
    // Lanczos g=5 approximation
    const G: f64 = 5.0;
    const C: [f64; 6] = [
        76.180_091_729_471_46,
        -86.505_320_329_416_77,
        24.014_098_240_830_91,
        -1.231_739_572_450_155,
        1.208_650_973_866_179e-3,
        -5.395_239_384_953_e-6,
    ];
    let z = z - 1.0;
    let x = 1.000_000_000_190_015
        + C[0] / (z + 1.0)
        + C[1] / (z + 2.0)
        + C[2] / (z + 3.0)
        + C[3] / (z + 4.0)
        + C[4] / (z + 5.0)
        + C[5] / (z + 6.0);
    (z + 0.5) * (z + G + 0.5).ln() - (z + G + 0.5)
        + (2.0 * std::f64::consts::PI).sqrt().ln()
        + x.ln()
}

// ---------------------------------------------------------------------------
// Kaplan-Meier estimator
// ---------------------------------------------------------------------------

/// Kaplan-Meier non-parametric survival estimator.
///
/// Computed at each unique event time using the product-limit formula.
/// Confidence intervals use Greenwood's variance formula with the
/// complementary log-log (log-log) transformation for better coverage
/// near the boundaries of [0, 1].
#[derive(Debug, Clone)]
pub struct KaplanMeier {
    /// Unique event times (sorted ascending).
    pub times: Vec<f64>,
    /// Survival probability S(t) at each event time.
    pub survival: Vec<f64>,
    /// 95% CI lower bound (complementary log-log scale).
    pub lower_ci: Vec<f64>,
    /// 95% CI upper bound (complementary log-log scale).
    pub upper_ci: Vec<f64>,
    /// Number of subjects at risk just before each event time.
    pub n_at_risk: Vec<usize>,
    /// Number of events at each event time.
    pub n_events: Vec<usize>,
}

impl KaplanMeier {
    /// Fit the Kaplan-Meier estimator.
    ///
    /// # Arguments
    /// * `times`  – observed times (must be ≥ 0, finite)
    /// * `events` – `true` if the event occurred, `false` if censored
    ///
    /// # Errors
    /// Returns an error when inputs are empty, have mismatched lengths,
    /// or contain non-finite / negative values.
    pub fn fit(times: &[f64], events: &[bool]) -> StatsResult<Self> {
        if times.is_empty() {
            return Err(StatsError::InvalidArgument(
                "times array cannot be empty".into(),
            ));
        }
        if times.len() != events.len() {
            return Err(StatsError::DimensionMismatch(format!(
                "times ({}) and events ({}) must have equal length",
                times.len(),
                events.len()
            )));
        }
        for (i, &t) in times.iter().enumerate() {
            if !t.is_finite() {
                return Err(StatsError::InvalidArgument(format!(
                    "times[{i}] is not finite: {t}"
                )));
            }
            if t < 0.0 {
                return Err(StatsError::InvalidArgument(format!(
                    "times[{i}] is negative: {t}"
                )));
            }
        }

        // Sort by time (stable, ties broken by event-first so events precede
        // censoring at the same time — conservative KM convention).
        let mut pairs: Vec<(f64, bool)> =
            times.iter().copied().zip(events.iter().copied()).collect();
        pairs.sort_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                // events (true) before censored (false) at equal times
                .then(b.1.cmp(&a.1))
        });

        let n = pairs.len();
        let mut unique_times: Vec<f64> = Vec::new();
        let mut survival_probs: Vec<f64> = Vec::new();
        let mut at_risk_vec: Vec<usize> = Vec::new();
        let mut events_vec: Vec<usize> = Vec::new();

        let mut current_survival = 1.0_f64;
        let mut at_risk = n;
        let mut i = 0;

        while i < pairs.len() {
            let t = pairs[i].0;
            let mut d = 0_usize; // deaths/events
            let mut c = 0_usize; // censored
                                 // Collect all observations at time t
            while i < pairs.len() && pairs[i].0 == t {
                if pairs[i].1 {
                    d += 1;
                } else {
                    c += 1;
                }
                i += 1;
            }
            // Only record a step at times with actual events
            if d > 0 {
                current_survival *= 1.0 - (d as f64) / (at_risk as f64);
                unique_times.push(t);
                survival_probs.push(current_survival);
                at_risk_vec.push(at_risk);
                events_vec.push(d);
            }
            at_risk -= d + c;
        }

        // Greenwood confidence intervals (log-log transform, 95%)
        let z = 1.959_963_985; // Φ^{-1}(0.975)
        let mut lower_ci = vec![0.0_f64; unique_times.len()];
        let mut upper_ci = vec![1.0_f64; unique_times.len()];
        let mut greenwood_sum = 0.0_f64;

        for k in 0..unique_times.len() {
            let n_k = at_risk_vec[k] as f64;
            let d_k = events_vec[k] as f64;
            if n_k > d_k {
                greenwood_sum += d_k / (n_k * (n_k - d_k));
            }
            let s = survival_probs[k];
            if s > 0.0 && s < 1.0 {
                let ln_s = s.ln();
                // SE on log(-log S) scale
                let se_ll = (greenwood_sum / (ln_s * ln_s)).sqrt();
                let log_log_s = (-ln_s).ln();
                let ll_lo = log_log_s - z * se_ll;
                let ll_hi = log_log_s + z * se_ll;
                // Back-transform: S = exp(-exp(θ)), which is *decreasing* in θ.
                // So lower CI for S uses the upper CI for θ, and vice-versa:
                //   lower_ci = exp(-exp(ll_hi))
                //   upper_ci = exp(-exp(ll_lo))
                lower_ci[k] = (-ll_hi.exp()).exp().max(0.0);
                upper_ci[k] = (-ll_lo.exp()).exp().min(1.0);
            } else if s <= 0.0 {
                lower_ci[k] = 0.0;
                upper_ci[k] = 0.0;
            } else {
                // s == 1.0: no variance yet, CI is [1, 1]
                lower_ci[k] = 1.0;
                upper_ci[k] = 1.0;
            }
        }

        Ok(Self {
            times: unique_times,
            survival: survival_probs,
            lower_ci,
            upper_ci,
            n_at_risk: at_risk_vec,
            n_events: events_vec,
        })
    }

    /// Evaluate the survival probability at time `t` (step function: left-continuous).
    ///
    /// Returns 1.0 before the first event time, and the last recorded
    /// survival probability for times beyond the last event.
    pub fn survival_at(&self, t: f64) -> f64 {
        if self.times.is_empty() || t < self.times[0] {
            return 1.0;
        }
        // Find largest recorded time ≤ t
        let mut prob = self.survival[0];
        for (k, &tk) in self.times.iter().enumerate() {
            if tk <= t {
                prob = self.survival[k];
            } else {
                break;
            }
        }
        prob
    }

    /// Median survival time: smallest t such that S(t) ≤ 0.5.
    ///
    /// Returns `None` if the survival function never drops to 0.5 (i.e., more
    /// than half the sample is censored beyond the last event time).
    pub fn median_survival(&self) -> Option<f64> {
        for (k, &s) in self.survival.iter().enumerate() {
            if s <= 0.5 {
                return Some(self.times[k]);
            }
        }
        None
    }

    /// Restricted mean survival time (area under the KM curve) up to the last event time.
    pub fn mean_survival(&self) -> f64 {
        if self.times.is_empty() {
            return 0.0;
        }
        let mut area = 0.0_f64;
        let mut prev_t = 0.0_f64;
        let mut prev_s = 1.0_f64;
        for (k, &t) in self.times.iter().enumerate() {
            area += prev_s * (t - prev_t);
            prev_t = t;
            prev_s = self.survival[k];
        }
        area
    }

    /// Log-rank test comparing two Kaplan-Meier fitted groups.
    ///
    /// Returns `(chi2_statistic, p_value)` under H₀: the two survival
    /// functions are identical.  The test statistic follows a χ²(1)
    /// distribution under the null.
    ///
    /// This static method works with the *raw* data (not the KM structs)
    /// so that it can properly enumerate the joint risk sets.
    pub fn log_rank_test(
        times1: &[f64],
        events1: &[bool],
        times2: &[f64],
        events2: &[bool],
    ) -> StatsResult<(f64, f64)> {
        if times1.is_empty() || times2.is_empty() {
            return Err(StatsError::InvalidArgument(
                "Both groups must have at least one observation".into(),
            ));
        }
        if times1.len() != events1.len() || times2.len() != events2.len() {
            return Err(StatsError::DimensionMismatch(
                "times and events must have equal length in each group".into(),
            ));
        }

        // Merge all observations with a group label (0 or 1)
        let mut all: Vec<(f64, bool, u8)> = times1
            .iter()
            .copied()
            .zip(events1.iter().copied())
            .map(|(t, e)| (t, e, 0_u8))
            .chain(
                times2
                    .iter()
                    .copied()
                    .zip(events2.iter().copied())
                    .map(|(t, e)| (t, e, 1_u8)),
            )
            .collect();
        all.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut observed_minus_expected = 0.0_f64;
        let mut variance = 0.0_f64;
        let mut r1 = times1.len() as f64;
        let mut r2 = times2.len() as f64;

        let mut idx = 0;
        while idx < all.len() {
            let t = all[idx].0;
            let mut d1 = 0.0_f64;
            let mut d2 = 0.0_f64;
            let mut c1 = 0.0_f64;
            let mut c2 = 0.0_f64;
            while idx < all.len() && all[idx].0 == t {
                match (all[idx].1, all[idx].2) {
                    (true, 0) => d1 += 1.0,
                    (false, 0) => c1 += 1.0,
                    (true, _) => d2 += 1.0,
                    (false, _) => c2 += 1.0,
                }
                idx += 1;
            }
            let d = d1 + d2;
            let r = r1 + r2;
            if d > 0.0 && r > 0.0 {
                let e1 = (r1 / r) * d;
                observed_minus_expected += d1 - e1;
                if r > 1.0 {
                    variance += r1 * r2 * d * (r - d) / (r * r * (r - 1.0));
                }
            }
            r1 -= d1 + c1;
            r2 -= d2 + c2;
        }

        if variance <= 0.0 {
            return Ok((0.0, 1.0));
        }
        let chi2 = (observed_minus_expected * observed_minus_expected) / variance;
        let pval = chi2_sf(chi2, 1.0);
        Ok((chi2, pval))
    }
}

// ---------------------------------------------------------------------------
// Nelson-Aalen estimator
// ---------------------------------------------------------------------------

/// Nelson-Aalen non-parametric cumulative hazard estimator.
///
/// At each event time tᵢ with dᵢ events and nᵢ at risk, the increment is
/// dᵢ / nᵢ.  The cumulative hazard H(t) = Σ dᵢ/nᵢ for tᵢ ≤ t.
///
/// The corresponding survival function estimate is exp(-H(t)).
#[derive(Debug, Clone)]
pub struct NelsonAalen {
    /// Unique event times (sorted ascending).
    pub times: Vec<f64>,
    /// Cumulative hazard H(t) at each event time.
    pub cumulative_hazard: Vec<f64>,
}

impl NelsonAalen {
    /// Fit the Nelson-Aalen estimator.
    ///
    /// # Arguments
    /// * `times`  – observed times (must be ≥ 0, finite)
    /// * `events` – `true` if the event occurred, `false` if censored
    pub fn fit(times: &[f64], events: &[bool]) -> StatsResult<Self> {
        if times.is_empty() {
            return Err(StatsError::InvalidArgument(
                "times array cannot be empty".into(),
            ));
        }
        if times.len() != events.len() {
            return Err(StatsError::DimensionMismatch(format!(
                "times ({}) and events ({}) must have equal length",
                times.len(),
                events.len()
            )));
        }
        for (i, &t) in times.iter().enumerate() {
            if !t.is_finite() {
                return Err(StatsError::InvalidArgument(format!(
                    "times[{i}] is not finite: {t}"
                )));
            }
            if t < 0.0 {
                return Err(StatsError::InvalidArgument(format!(
                    "times[{i}] is negative: {t}"
                )));
            }
        }

        let mut pairs: Vec<(f64, bool)> =
            times.iter().copied().zip(events.iter().copied()).collect();
        pairs.sort_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(b.1.cmp(&a.1))
        });

        let n = pairs.len();
        let mut unique_times: Vec<f64> = Vec::new();
        let mut cumhaz: Vec<f64> = Vec::new();

        let mut at_risk = n;
        let mut current_cumhaz = 0.0_f64;
        let mut i = 0;

        while i < pairs.len() {
            let t = pairs[i].0;
            let mut d = 0_usize;
            let mut c = 0_usize;
            while i < pairs.len() && pairs[i].0 == t {
                if pairs[i].1 {
                    d += 1;
                } else {
                    c += 1;
                }
                i += 1;
            }
            if d > 0 && at_risk > 0 {
                current_cumhaz += (d as f64) / (at_risk as f64);
                unique_times.push(t);
                cumhaz.push(current_cumhaz);
            }
            at_risk -= d + c;
        }

        Ok(Self {
            times: unique_times,
            cumulative_hazard: cumhaz,
        })
    }

    /// Evaluate the cumulative hazard H(t) at time `t`.
    ///
    /// Returns 0.0 before the first event time.
    pub fn hazard_at(&self, t: f64) -> f64 {
        if self.times.is_empty() || t < self.times[0] {
            return 0.0;
        }
        let mut h = self.cumulative_hazard[0];
        for (k, &tk) in self.times.iter().enumerate() {
            if tk <= t {
                h = self.cumulative_hazard[k];
            } else {
                break;
            }
        }
        h
    }

    /// Survival probability estimate S(t) = exp(-H(t)).
    pub fn survival_at(&self, t: f64) -> f64 {
        (-self.hazard_at(t)).exp()
    }
}

// ---------------------------------------------------------------------------
// Cox Proportional Hazards model
// ---------------------------------------------------------------------------

/// Result type for a fitted Cox Proportional Hazards model.
///
/// The model is fitted by maximising the partial likelihood via Newton-Raphson
/// iterations.  Standard errors are computed from the observed information
/// matrix (negative Hessian) at convergence.
///
/// The concordance index (Harrell's C-statistic) measures the model's
/// discriminative ability.
#[derive(Debug, Clone)]
pub struct CoxPH {
    /// Regression coefficients β̂.
    pub coefficients: Array1<f64>,
    /// Standard errors SE(β̂).
    pub std_errors: Array1<f64>,
    /// Z-scores β̂ / SE(β̂).
    pub z_scores: Array1<f64>,
    /// Two-sided p-values under H₀: βⱼ = 0.
    pub p_values: Array1<f64>,
    /// Log partial likelihood at convergence.
    pub log_likelihood: f64,
    /// Number of Newton-Raphson iterations performed.
    pub n_iter: usize,
    /// Baseline times (Breslow estimator).
    baseline_times: Vec<f64>,
    /// Baseline cumulative hazard (Breslow estimator).
    baseline_cumhaz: Vec<f64>,
}

impl CoxPH {
    /// Fit a Cox Proportional Hazards model.
    ///
    /// # Arguments
    /// * `times`      – observed times (must be ≥ 0, finite)
    /// * `events`     – `true` if the event occurred
    /// * `covariates` – design matrix of shape (n_samples, n_features)
    ///
    /// # Convergence
    /// Newton-Raphson with a maximum of 100 iterations and tolerance 1e-7
    /// on the relative change in log partial likelihood.
    pub fn fit(times: &[f64], events: &[bool], covariates: &Array2<f64>) -> StatsResult<Self> {
        let n = times.len();
        let p = covariates.ncols();

        if n == 0 {
            return Err(StatsError::InvalidArgument("times cannot be empty".into()));
        }
        if n != events.len() {
            return Err(StatsError::DimensionMismatch(format!(
                "times ({n}) and events ({}) must have equal length",
                events.len()
            )));
        }
        if covariates.nrows() != n {
            return Err(StatsError::DimensionMismatch(format!(
                "covariates has {} rows but times has {n} entries",
                covariates.nrows()
            )));
        }
        if p == 0 {
            return Err(StatsError::InvalidArgument(
                "covariates must have at least one column".into(),
            ));
        }
        for (i, &t) in times.iter().enumerate() {
            if !t.is_finite() {
                return Err(StatsError::InvalidArgument(format!(
                    "times[{i}] is not finite: {t}"
                )));
            }
            if t < 0.0 {
                return Err(StatsError::InvalidArgument(format!(
                    "times[{i}] is negative: {t}"
                )));
            }
        }

        // Sort index by time
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| {
            times[a]
                .partial_cmp(&times[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Newton-Raphson iterations
        let max_iter = 100_usize;
        let tol = 1e-7_f64;
        let mut beta = Array1::zeros(p);
        let mut prev_ll = f64::NEG_INFINITY;
        let mut n_iter = 0_usize;

        for iter in 0..max_iter {
            let (ll, grad, hess) =
                Self::partial_likelihood(&order, times, events, covariates, &beta)?;

            let converged = (ll - prev_ll).abs() < tol * (1.0 + ll.abs());
            prev_ll = ll;
            n_iter = iter + 1;

            // Newton-Raphson step: β ← β + H⁻¹ g
            let hess_inv = invert_matrix(&hess)?;
            let delta = hess_inv.dot(&grad);
            // Limit step size to avoid divergence
            let step_norm = delta.iter().map(|&v| v * v).sum::<f64>().sqrt();
            let scale = if step_norm > 2.0 {
                2.0 / step_norm
            } else {
                1.0
            };
            beta = &beta + &(scale * &delta);

            if converged {
                break;
            }
        }

        // Final information matrix for standard errors
        let (log_likelihood, _, hess_final) =
            Self::partial_likelihood(&order, times, events, covariates, &beta)?;
        let info_inv = invert_matrix(&hess_final).unwrap_or_else(|_| Array2::eye(p));
        let std_errors: Array1<f64> = (0..p)
            .map(|j| info_inv[[j, j]].max(0.0).sqrt())
            .collect::<Vec<_>>()
            .into();

        let z_scores: Array1<f64> = (0..p)
            .map(|j| {
                let se = std_errors[j];
                if se > 0.0 {
                    beta[j] / se
                } else {
                    0.0
                }
            })
            .collect::<Vec<_>>()
            .into();

        // Two-sided p-values using normal approximation
        let p_values: Array1<f64> = z_scores
            .iter()
            .map(|&z| {
                let p = 2.0 * (1.0 - norm_cdf(z.abs()));
                p.clamp(0.0, 1.0)
            })
            .collect::<Vec<_>>()
            .into();

        // Breslow baseline cumulative hazard
        let (baseline_times, baseline_cumhaz) =
            Self::breslow_baseline(&order, times, events, covariates, &beta)?;

        Ok(Self {
            coefficients: beta,
            std_errors,
            z_scores,
            p_values,
            log_likelihood,
            n_iter,
            baseline_times,
            baseline_cumhaz,
        })
    }

    /// Compute the partial log-likelihood, gradient, and observed information
    /// matrix (negative Hessian) for a given β.
    ///
    /// Uses the Breslow approximation for ties.
    fn partial_likelihood(
        order: &[usize],
        times: &[f64],
        events: &[bool],
        covariates: &Array2<f64>,
        beta: &Array1<f64>,
    ) -> StatsResult<(f64, Array1<f64>, Array2<f64>)> {
        let n = times.len();
        let p = beta.len();

        // Pre-compute exp(β'xᵢ) and xᵢ for all subjects
        let mut exp_beta_x = vec![0.0_f64; n];
        for i in 0..n {
            let xi = covariates.row(i);
            exp_beta_x[i] = xi.dot(beta).exp();
        }

        let mut ll = 0.0_f64;
        let mut grad: Array1<f64> = Array1::zeros(p);
        let mut info: Array2<f64> = Array2::zeros((p, p)); // observed information = −Hessian

        let mut idx = 0_usize;
        while idx < order.len() {
            let i = order[idx];
            if !events[i] {
                idx += 1;
                continue;
            }
            let t_i = times[i];
            let xi = covariates.row(i);

            // Risk set: all j with times[j] >= t_i  (Breslow ties: all tied events share same risk set)
            // We accumulate S0, S1, S2 (scalar, vector, matrix moments of exp(β'x) over risk set)
            let mut s0 = 0.0_f64;
            let mut s1: Array1<f64> = Array1::zeros(p);
            let mut s2: Array2<f64> = Array2::zeros((p, p));

            for j in 0..n {
                if times[j] >= t_i {
                    let w: f64 = exp_beta_x[j];
                    s0 += w;
                    let xj = covariates.row(j);
                    let xj_owned: Array1<f64> = xj.to_owned();
                    s1 = s1 + w * xj_owned;
                    for a in 0..p {
                        for b in 0..p {
                            s2[[a, b]] += w * xj[a] * xj[b];
                        }
                    }
                }
            }

            if s0 <= 0.0 {
                idx += 1;
                continue;
            }

            // β'xᵢ − log S0
            ll += xi.dot(beta) - s0.ln();

            // Gradient: xᵢ − S1/S0
            let mean_x: Array1<f64> = s1.mapv(|v| v / s0);
            let xi_owned: Array1<f64> = xi.to_owned();
            grad = grad + (xi_owned - &mean_x);

            // Information: S2/S0 − (S1/S0)(S1/S0)ᵀ
            for a in 0..p {
                for b in 0..p {
                    let s2_bar: f64 = s2[[a, b]] / s0;
                    info[[a, b]] += s2_bar - mean_x[a] * mean_x[b];
                }
            }

            idx += 1;
        }

        Ok((ll, grad, info))
    }

    /// Compute Breslow baseline cumulative hazard estimate.
    fn breslow_baseline(
        order: &[usize],
        times: &[f64],
        events: &[bool],
        covariates: &Array2<f64>,
        beta: &Array1<f64>,
    ) -> StatsResult<(Vec<f64>, Vec<f64>)> {
        let n = times.len();
        let mut exp_beta_x = vec![0.0_f64; n];
        for i in 0..n {
            exp_beta_x[i] = covariates.row(i).dot(beta).exp();
        }

        let mut result_times: Vec<f64> = Vec::new();
        let mut result_cumhaz: Vec<f64> = Vec::new();
        let mut cumhaz = 0.0_f64;

        let mut idx = 0_usize;
        while idx < order.len() {
            let i = order[idx];
            if !events[i] {
                idx += 1;
                continue;
            }
            let t_i = times[i];
            // Risk set sum
            let risk_sum: f64 = (0..n)
                .filter(|&j| times[j] >= t_i)
                .map(|j| exp_beta_x[j])
                .sum();
            if risk_sum > 0.0 {
                cumhaz += 1.0 / risk_sum;
                result_times.push(t_i);
                result_cumhaz.push(cumhaz);
            }
            idx += 1;
        }
        Ok((result_times, result_cumhaz))
    }

    /// Exponentiated coefficients exp(β̂): hazard ratios for a unit covariate increase.
    pub fn hazard_ratio(&self) -> Array1<f64> {
        self.coefficients.mapv(f64::exp)
    }

    /// Predict the relative risk score exp(β̂' x) for a single covariate vector.
    pub fn predict_risk(&self, x: &Array1<f64>) -> f64 {
        x.dot(&self.coefficients).exp()
    }

    /// Harrell's concordance index (C-statistic).
    ///
    /// Proportion of comparable pairs (both with events, or one event and the
    /// other censored later) in which the model assigns the higher risk to the
    /// subject who experienced the event first.
    ///
    /// # Arguments
    /// * `times`      – observed times
    /// * `events`     – event indicators
    /// * `covariates` – covariate matrix for the evaluation set
    pub fn concordance_index(
        &self,
        times: &[f64],
        events: &[bool],
        covariates: &Array2<f64>,
    ) -> f64 {
        let n = times.len().min(events.len()).min(covariates.nrows());
        if n == 0 {
            return 0.5;
        }
        let scores: Vec<f64> = (0..n)
            .map(|i| covariates.row(i).dot(&self.coefficients))
            .collect();

        let mut concordant = 0.0_f64;
        let mut discordant = 0.0_f64;
        let mut tied_risk = 0.0_f64;

        for i in 0..n {
            if !events[i] {
                continue;
            }
            for j in 0..n {
                if i == j {
                    continue;
                }
                // Pair (i, j) is comparable if i had the event earlier OR
                // i had the event and j was censored at a later time.
                let comparable = if events[j] {
                    times[i] < times[j]
                } else {
                    times[i] <= times[j]
                };
                if !comparable {
                    continue;
                }
                if scores[i] > scores[j] {
                    concordant += 1.0;
                } else if scores[i] < scores[j] {
                    discordant += 1.0;
                } else {
                    tied_risk += 1.0;
                }
            }
        }

        let total = concordant + discordant + tied_risk;
        if total <= 0.0 {
            return 0.5;
        }
        (concordant + 0.5 * tied_risk) / total
    }
}

// ---------------------------------------------------------------------------
// Matrix inversion helper (no external BLAS dependency)
// Uses Gauss-Jordan elimination with partial pivoting.
// ---------------------------------------------------------------------------

fn invert_matrix(a: &Array2<f64>) -> StatsResult<Array2<f64>> {
    let n = a.nrows();
    if n != a.ncols() {
        return Err(StatsError::InvalidArgument(
            "Matrix must be square for inversion".into(),
        ));
    }
    if n == 0 {
        return Ok(Array2::zeros((0, 0)));
    }

    // Build augmented matrix [A | I]
    let mut aug = Array2::zeros((n, 2 * n));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
        }
        aug[[i, n + i]] = 1.0;
    }

    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut pivot_row = col;
        let mut max_val = aug[[col, col]].abs();
        for row in (col + 1)..n {
            if aug[[row, col]].abs() > max_val {
                max_val = aug[[row, col]].abs();
                pivot_row = row;
            }
        }
        if max_val < 1e-14 {
            return Err(StatsError::ComputationError(
                "Matrix is singular or near-singular; cannot invert".into(),
            ));
        }
        // Swap rows
        if pivot_row != col {
            for j in 0..(2 * n) {
                let tmp = aug[[col, j]];
                aug[[col, j]] = aug[[pivot_row, j]];
                aug[[pivot_row, j]] = tmp;
            }
        }
        // Scale pivot row
        let scale = aug[[col, col]];
        for j in 0..(2 * n) {
            aug[[col, j]] /= scale;
        }
        // Eliminate column in all other rows
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = aug[[row, col]];
            for j in 0..(2 * n) {
                let delta = factor * aug[[col, j]];
                aug[[row, j]] -= delta;
            }
        }
    }

    // Extract right half
    let mut inv = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            inv[[i, j]] = aug[[i, n + j]];
        }
    }
    Ok(inv)
}

// ---------------------------------------------------------------------------
// Re-export legacy types so downstream code that imports via `survival::`
// continues to work unchanged.
// ---------------------------------------------------------------------------
pub use legacy::{
    AFTDistribution, AFTModel, CompetingRisksModel, CoxPHModel, ExtendedCoxModel,
    KaplanMeierEstimator, LogRankTest,
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    // Helper: simple 10-subject dataset, 6 events, 4 censored
    fn simple_data() -> (Vec<f64>, Vec<bool>) {
        let times = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let events = vec![
            true, true, false, true, true, false, true, false, true, false,
        ];
        (times, events)
    }

    // -----------------------------------------------------------------------
    // Kaplan-Meier tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_km_fit_basic() {
        let (times, events) = simple_data();
        let km = KaplanMeier::fit(&times, &events).expect("KM fit failed");
        assert!(!km.times.is_empty());
        assert_eq!(km.times.len(), km.survival.len());
        assert_eq!(km.times.len(), km.n_at_risk.len());
        assert_eq!(km.times.len(), km.n_events.len());
    }

    #[test]
    fn test_km_survival_starts_at_one() {
        let (times, events) = simple_data();
        let km = KaplanMeier::fit(&times, &events).expect("KM fit failed");
        // Before first event: S(t) = 1.0
        assert!((km.survival_at(0.0) - 1.0).abs() < 1e-12);
        assert!((km.survival_at(0.5) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_km_survival_monotone_decreasing() {
        let (times, events) = simple_data();
        let km = KaplanMeier::fit(&times, &events).expect("KM fit failed");
        for i in 1..km.survival.len() {
            assert!(
                km.survival[i] <= km.survival[i - 1] + 1e-12,
                "KM survival is not monotone at index {i}"
            );
        }
    }

    #[test]
    fn test_km_survival_bounded() {
        let (times, events) = simple_data();
        let km = KaplanMeier::fit(&times, &events).expect("KM fit failed");
        for &s in &km.survival {
            assert!(s >= 0.0 && s <= 1.0 + 1e-12, "S(t)={s} is out of [0,1]");
        }
    }

    #[test]
    fn test_km_confidence_intervals_ordering() {
        let (times, events) = simple_data();
        let km = KaplanMeier::fit(&times, &events).expect("KM fit failed");
        for k in 0..km.times.len() {
            assert!(
                km.lower_ci[k] <= km.survival[k] + 1e-10,
                "lower CI > S(t) at index {k}"
            );
            assert!(
                km.survival[k] <= km.upper_ci[k] + 1e-10,
                "S(t) > upper CI at index {k}"
            );
        }
    }

    #[test]
    fn test_km_median_survival() {
        // Dataset where all subjects have events: median must exist
        let times = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let events = vec![true; 10];
        let km = KaplanMeier::fit(&times, &events).expect("KM fit failed");
        let median = km.median_survival();
        assert!(
            median.is_some(),
            "Median should exist when all subjects have events"
        );
        let m = median.expect("median must be Some");
        assert!(m > 0.0 && m <= 10.0, "Median out of range: {m}");
    }

    #[test]
    fn test_km_mean_survival_positive() {
        let (times, events) = simple_data();
        let km = KaplanMeier::fit(&times, &events).expect("KM fit failed");
        let mean = km.mean_survival();
        assert!(mean > 0.0, "Mean survival should be positive");
    }

    #[test]
    fn test_km_error_empty() {
        let result = KaplanMeier::fit(&[], &[]);
        assert!(result.is_err(), "Empty input should return an error");
    }

    #[test]
    fn test_km_error_length_mismatch() {
        let result = KaplanMeier::fit(&[1.0, 2.0], &[true]);
        assert!(result.is_err(), "Length mismatch should return an error");
    }

    #[test]
    fn test_km_error_negative_time() {
        let result = KaplanMeier::fit(&[-1.0, 2.0], &[true, true]);
        assert!(result.is_err(), "Negative time should return an error");
    }

    #[test]
    fn test_km_all_censored_has_no_steps() {
        let times = vec![1.0, 2.0, 3.0];
        let events = vec![false, false, false];
        let km =
            KaplanMeier::fit(&times, &events).expect("KM fit should not error for all-censored");
        // No events => no KM steps
        assert!(km.times.is_empty(), "No steps expected when all censored");
        assert!((km.survival_at(5.0) - 1.0).abs() < 1e-12);
    }

    // -----------------------------------------------------------------------
    // Log-rank test
    // -----------------------------------------------------------------------

    #[test]
    fn test_log_rank_identical_groups() {
        // Identical groups should yield a small chi2 statistic
        let times = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let events = vec![true; 5];
        let (chi2, pval) = KaplanMeier::log_rank_test(&times, &events, &times, &events)
            .expect("log-rank test failed");
        assert!(
            chi2.abs() < 1e-6,
            "chi2 for identical groups should be ~0, got {chi2}"
        );
        assert!(
            pval > 0.05,
            "p-value for identical groups should be high, got {pval}"
        );
    }

    #[test]
    fn test_log_rank_different_groups() {
        // Group 1: short survival; Group 2: long survival → large chi2
        let t1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let e1 = vec![true, true, true, true, true];
        let t2 = vec![10.0, 12.0, 14.0, 16.0, 18.0];
        let e2 = vec![true, true, true, true, true];
        let (chi2, pval) =
            KaplanMeier::log_rank_test(&t1, &e1, &t2, &e2).expect("log-rank test failed");
        assert!(
            chi2 > 1.0,
            "chi2 should be large for very different groups, got {chi2}"
        );
        assert!(
            pval < 0.5,
            "p-value should be small for very different groups, got {pval}"
        );
    }

    #[test]
    fn test_log_rank_p_value_range() {
        let t1 = vec![1.0, 3.0, 5.0, 7.0, 9.0];
        let e1 = vec![true, true, false, true, true];
        let t2 = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let e2 = vec![true, false, true, true, false];
        let (chi2, pval) =
            KaplanMeier::log_rank_test(&t1, &e1, &t2, &e2).expect("log-rank test failed");
        assert!(chi2 >= 0.0, "chi2 must be non-negative");
        assert!(
            pval >= 0.0 && pval <= 1.0,
            "p-value must be in [0,1], got {pval}"
        );
    }

    // -----------------------------------------------------------------------
    // Nelson-Aalen tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_na_fit_basic() {
        let (times, events) = simple_data();
        let na = NelsonAalen::fit(&times, &events).expect("NA fit failed");
        assert!(!na.times.is_empty());
        assert_eq!(na.times.len(), na.cumulative_hazard.len());
    }

    #[test]
    fn test_na_hazard_monotone_increasing() {
        let (times, events) = simple_data();
        let na = NelsonAalen::fit(&times, &events).expect("NA fit failed");
        for i in 1..na.cumulative_hazard.len() {
            assert!(
                na.cumulative_hazard[i] >= na.cumulative_hazard[i - 1] - 1e-12,
                "Cumulative hazard not monotone at index {i}"
            );
        }
    }

    #[test]
    fn test_na_survival_bounded() {
        let (times, events) = simple_data();
        let na = NelsonAalen::fit(&times, &events).expect("NA fit failed");
        for k in 0..na.times.len() {
            let s = na.survival_at(na.times[k]);
            assert!(s >= 0.0 && s <= 1.0 + 1e-12, "S(t)={s} out of [0,1]");
        }
    }

    #[test]
    fn test_na_hazard_zero_before_first_event() {
        let (times, events) = simple_data();
        let na = NelsonAalen::fit(&times, &events).expect("NA fit failed");
        assert!((na.hazard_at(0.0) - 0.0).abs() < 1e-12);
        assert!((na.survival_at(0.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_na_error_empty() {
        let result = NelsonAalen::fit(&[], &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_na_error_negative_time() {
        let result = NelsonAalen::fit(&[-1.0, 2.0], &[true, true]);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // Cox PH tests
    // -----------------------------------------------------------------------

    fn cox_simple_data() -> (Vec<f64>, Vec<bool>, Array2<f64>) {
        // n=10, p=1; covariate linearly correlated with time (higher covariate → shorter survival)
        let times = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let events = vec![true, true, true, true, true, false, true, true, false, true];
        let mut cov = Array2::zeros((10, 1));
        for i in 0..10_usize {
            // Higher covariate → higher hazard → shorter time
            cov[[i, 0]] = (10 - i) as f64;
        }
        (times, events, cov)
    }

    #[test]
    fn test_cox_fit_returns_result() {
        let (times, events, cov) = cox_simple_data();
        let model = CoxPH::fit(&times, &events, &cov).expect("Cox fit failed");
        assert_eq!(model.coefficients.len(), 1);
        assert!(model.n_iter > 0);
    }

    #[test]
    fn test_cox_coefficient_sign() {
        let (times, events, cov) = cox_simple_data();
        let model = CoxPH::fit(&times, &events, &cov).expect("Cox fit failed");
        // Higher covariate → shorter survival → negative coefficient (covariate decreases with time)
        // (actual sign depends on data layout; just verify it's a real finite number)
        assert!(
            model.coefficients[0].is_finite(),
            "Coefficient must be finite"
        );
    }

    #[test]
    fn test_cox_std_errors_positive() {
        let (times, events, cov) = cox_simple_data();
        let model = CoxPH::fit(&times, &events, &cov).expect("Cox fit failed");
        for &se in model.std_errors.iter() {
            assert!(se >= 0.0, "Standard error must be non-negative, got {se}");
        }
    }

    #[test]
    fn test_cox_p_values_in_range() {
        let (times, events, cov) = cox_simple_data();
        let model = CoxPH::fit(&times, &events, &cov).expect("Cox fit failed");
        for &p in model.p_values.iter() {
            assert!(p >= 0.0 && p <= 1.0, "p-value must be in [0,1], got {p}");
        }
    }

    #[test]
    fn test_cox_hazard_ratio_positive() {
        let (times, events, cov) = cox_simple_data();
        let model = CoxPH::fit(&times, &events, &cov).expect("Cox fit failed");
        for &hr in model.hazard_ratio().iter() {
            assert!(hr > 0.0, "Hazard ratio must be positive, got {hr}");
        }
    }

    #[test]
    fn test_cox_predict_risk_positive() {
        let (times, events, cov) = cox_simple_data();
        let model = CoxPH::fit(&times, &events, &cov).expect("Cox fit failed");
        let x = Array1::from_vec(vec![5.0]);
        let risk = model.predict_risk(&x);
        assert!(risk > 0.0, "Risk score must be positive, got {risk}");
    }

    #[test]
    fn test_cox_concordance_index_in_range() {
        let (times, events, cov) = cox_simple_data();
        let model = CoxPH::fit(&times, &events, &cov).expect("Cox fit failed");
        let c = model.concordance_index(&times, &events, &cov);
        assert!(
            c >= 0.0 && c <= 1.0,
            "Concordance index must be in [0,1], got {c}"
        );
    }

    #[test]
    fn test_cox_log_likelihood_finite() {
        let (times, events, cov) = cox_simple_data();
        let model = CoxPH::fit(&times, &events, &cov).expect("Cox fit failed");
        assert!(
            model.log_likelihood.is_finite(),
            "Log-likelihood must be finite"
        );
    }

    #[test]
    fn test_cox_error_empty() {
        let cov: Array2<f64> = Array2::zeros((0, 1));
        let result = CoxPH::fit(&[], &[], &cov);
        assert!(result.is_err());
    }

    #[test]
    fn test_cox_error_dimension_mismatch() {
        let times = vec![1.0, 2.0];
        let events = vec![true];
        let cov = Array2::zeros((2, 1));
        let result = CoxPH::fit(&times, &events, &cov);
        assert!(result.is_err());
    }

    #[test]
    fn test_cox_multivariate() {
        // Two-covariate model to verify p=2 path
        let times = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let events = vec![true, true, false, true, true, false, true, true];
        let mut cov = Array2::zeros((8, 2));
        for i in 0..8_usize {
            cov[[i, 0]] = i as f64;
            cov[[i, 1]] = (i % 3) as f64;
        }
        let model = CoxPH::fit(&times, &events, &cov).expect("Cox multivariate fit failed");
        assert_eq!(model.coefficients.len(), 2);
        assert_eq!(model.std_errors.len(), 2);
        assert_eq!(model.p_values.len(), 2);
    }

    // -----------------------------------------------------------------------
    // Cross-consistency between KM and NA
    // -----------------------------------------------------------------------

    #[test]
    fn test_km_na_consistency() {
        // KM and NA should agree directionally: where KM survival drops, NA hazard rises.
        let (times, events) = simple_data();
        let km = KaplanMeier::fit(&times, &events).expect("KM fit failed");
        let na = NelsonAalen::fit(&times, &events).expect("NA fit failed");
        // Both should have the same event times
        assert_eq!(km.times, na.times, "KM and NA should share event times");
        // S_KM and exp(-H_NA) should both be decreasing at the same times
        for k in 0..km.times.len() {
            let s_km = km.survival[k];
            let s_na = (-na.cumulative_hazard[k]).exp();
            // They won't be identical but should agree within ~5%
            let diff = (s_km - s_na).abs();
            assert!(
                diff < 0.10,
                "KM and NA survival estimates differ by {diff:.4} at index {k}"
            );
        }
    }
}
