//! Canonical Survival Analysis API
//!
//! Wraps the core survival analysis types in `survival::*` with a standardised
//! interface that matches the SciRS2 public API requirements:
//!
//! - [`KMCurve`]       – Kaplan-Meier result with `survival_function(t)` and
//!                       `confidence_interval(t, alpha)` (Greenwood formula)
//! - [`NACurve`]       – Nelson-Aalen result with `survival_function(t)` and
//!                       `confidence_interval(t, alpha)`
//! - [`log_rank_test`] – two-sample log-rank test → `(statistic, p_value)`
//! - [`CoxPHModel`]    – Cox proportional hazards fitted via Newton-Raphson

use crate::error::{StatsError, StatsResult};
use crate::survival::{CoxPH, KaplanMeier, NelsonAalen};
use scirs2_core::ndarray::Array2;

// ---------------------------------------------------------------------------
// Normal quantile (Beasley-Springer-Moro rational approximation)
// ---------------------------------------------------------------------------

fn norm_ppf(p: f64) -> f64 {
    let p = p.clamp(1e-15, 1.0 - 1e-15);
    let q = p - 0.5;
    if q.abs() <= 0.42 {
        let r = q * q;
        q * ((((-25.445_87 * r + 41.391_663) * r - 18.615_43) * r + 2.506_628)
            / ((((3.130_347 * r - 21.060_244) * r + 23.083_928) * r - 8.476_377) * r + 1.0))
    } else {
        let r = if q < 0.0 { p } else { 1.0 - p };
        let r = (-r.ln()).sqrt();
        let x = (((2.321_213_5 * r + 4.850_091_7) * r - 2.297_460_0) * r - 2.787_688_0)
            / ((1.637_547_9 * r + 3.543_889_2) * r + 1.0);
        if q < 0.0 {
            -x
        } else {
            x
        }
    }
}

// ---------------------------------------------------------------------------
// KMCurve
// ---------------------------------------------------------------------------

/// The result of fitting the Kaplan-Meier estimator.
///
/// Exposes `survival_function(t)` and `confidence_interval(t, alpha)`.
pub struct KMCurve {
    km: KaplanMeier,
}

impl KMCurve {
    /// Fit the Kaplan-Meier estimator.
    ///
    /// # Arguments
    /// * `times`  – observed event/censoring times (must be ≥ 0, finite).
    /// * `events` – `true` if the observation is an actual event (uncensored).
    pub fn fit(times: &[f64], events: &[bool]) -> StatsResult<Self> {
        let km = KaplanMeier::fit(times, events)?;
        Ok(Self { km })
    }

    /// Evaluate the Kaplan-Meier survival function S(t) = P(T > t).
    pub fn survival_function(&self, t: f64) -> f64 {
        self.km.survival_at(t)
    }

    /// Compute a pointwise Greenwood confidence interval for S(t).
    ///
    /// Uses the log-log transform for better small-sample coverage.
    ///
    /// # Arguments
    /// * `t`     – time at which to evaluate the CI.
    /// * `alpha` – significance level (e.g. 0.05 for a 95% CI).
    ///
    /// # Returns
    /// `(lower, upper)` – both in \[0, 1\].
    pub fn confidence_interval(&self, t: f64, alpha: f64) -> StatsResult<(f64, f64)> {
        if alpha <= 0.0 || alpha >= 1.0 {
            return Err(StatsError::InvalidArgument(format!(
                "alpha must be in (0, 1), got {alpha}"
            )));
        }
        let s = self.survival_function(t);
        if s <= 0.0 || s >= 1.0 {
            return Ok((s.clamp(0.0, 1.0), s.clamp(0.0, 1.0)));
        }

        // Greenwood cumulative variance Σ d_k / (n_k (n_k - d_k)) up to time t
        let greenwood: f64 = self
            .km
            .times
            .iter()
            .enumerate()
            .take_while(|(_, &tk)| tk <= t)
            .map(|(k, _)| {
                let n_k = self.km.n_at_risk[k] as f64;
                let d_k = self.km.n_events[k] as f64;
                if n_k > d_k {
                    d_k / (n_k * (n_k - d_k))
                } else {
                    0.0
                }
            })
            .sum();

        if greenwood == 0.0 {
            return Ok((s, s));
        }

        let z = norm_ppf(1.0 - alpha / 2.0);
        let ln_s = s.ln();
        let se_ll = (greenwood / (ln_s * ln_s)).sqrt();
        let log_log_s = (-ln_s).ln();

        let ll_lo = log_log_s - z * se_ll;
        let ll_hi = log_log_s + z * se_ll;

        // Back-transform: S = exp(-exp(θ)) is *decreasing* in θ
        let lower = (-ll_hi.exp()).exp().clamp(0.0, 1.0);
        let upper = (-ll_lo.exp()).exp().clamp(0.0, 1.0);

        Ok((lower.min(upper), lower.max(upper)))
    }

    /// Median survival time (smallest t with S(t) ≤ 0.5).
    pub fn median_survival(&self) -> Option<f64> {
        self.km.median_survival()
    }

    /// Mean survival time.
    pub fn mean_survival(&self) -> f64 {
        self.km.mean_survival()
    }
}

// ---------------------------------------------------------------------------
// NACurve
// ---------------------------------------------------------------------------

/// The result of fitting the Nelson-Aalen estimator.
///
/// Stores hazard increments and at-risk counts for variance computation.
pub struct NACurve {
    na: NelsonAalen,
    /// Hazard increments Δ H(t_k) = d_k / n_k at each event time.
    hazard_increments: Vec<f64>,
    /// Number at risk n_k at each event time.
    at_risk: Vec<usize>,
}

impl NACurve {
    /// Fit the Nelson-Aalen estimator.
    ///
    /// # Arguments
    /// * `times`  – observed times (must be ≥ 0, finite).
    /// * `events` – `true` if the event occurred (uncensored).
    pub fn fit(times: &[f64], events: &[bool]) -> StatsResult<Self> {
        if times.is_empty() {
            return Err(StatsError::InvalidArgument(
                "times array cannot be empty".to_string(),
            ));
        }
        if times.len() != events.len() {
            return Err(StatsError::InvalidArgument(format!(
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

        // Sort by time (ties: events before censored)
        let mut pairs: Vec<(f64, bool)> =
            times.iter().copied().zip(events.iter().copied()).collect();
        pairs.sort_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(b.1.cmp(&a.1))
        });

        let n = pairs.len();
        let mut at_risk_count = n;
        let mut hazard_increments = Vec::new();
        let mut at_risk_vec = Vec::new();
        let mut idx = 0;

        while idx < pairs.len() {
            let t = pairs[idx].0;
            let mut d = 0_usize;
            let mut c = 0_usize;
            while idx < pairs.len() && (pairs[idx].0 - t).abs() < 1e-12 {
                if pairs[idx].1 {
                    d += 1;
                } else {
                    c += 1;
                }
                idx += 1;
            }
            if d > 0 && at_risk_count > 0 {
                hazard_increments.push((d as f64) / (at_risk_count as f64));
                at_risk_vec.push(at_risk_count);
            }
            at_risk_count -= d + c;
        }

        let na = NelsonAalen::fit(times, events)?;
        Ok(Self {
            na,
            hazard_increments,
            at_risk: at_risk_vec,
        })
    }

    /// Evaluate the survival function S(t) = exp(−Ĥ(t)).
    pub fn survival_function(&self, t: f64) -> f64 {
        self.na.survival_at(t)
    }

    /// Evaluate the cumulative hazard Ĥ(t).
    pub fn cumulative_hazard(&self, t: f64) -> f64 {
        self.na.hazard_at(t)
    }

    /// Compute a pointwise confidence interval for S(t).
    ///
    /// Uses the log-transform CI for the cumulative hazard.
    ///
    /// # Arguments
    /// * `t`     – evaluation time.
    /// * `alpha` – significance level (e.g. 0.05 → 95% CI).
    ///
    /// # Returns
    /// `(lower, upper)`.
    pub fn confidence_interval(&self, t: f64, alpha: f64) -> StatsResult<(f64, f64)> {
        if alpha <= 0.0 || alpha >= 1.0 {
            return Err(StatsError::InvalidArgument(format!(
                "alpha must be in (0, 1), got {alpha}"
            )));
        }
        let s = self.survival_function(t);
        if s <= 0.0 || s >= 1.0 {
            return Ok((s.clamp(0.0, 1.0), s.clamp(0.0, 1.0)));
        }

        // Variance of Ĥ(t): Var[Ĥ(t)] ≈ Σ_{t_k ≤ t} d_k / n_k²
        // = Σ (increment_k) / n_k
        let var_h: f64 = self
            .na
            .times
            .iter()
            .enumerate()
            .take_while(|(_, &tk)| tk <= t)
            .map(|(k, _)| {
                if k < self.at_risk.len() && self.at_risk[k] > 0 {
                    self.hazard_increments[k] / self.at_risk[k] as f64
                } else {
                    0.0
                }
            })
            .sum();

        if var_h == 0.0 {
            return Ok((s, s));
        }

        let h = -s.ln();
        let z = norm_ppf(1.0 - alpha / 2.0);
        let se = var_h.sqrt();

        // Log-transform CI for H: (H / c, H * c) where c = exp(z se / H)
        let c = (z * se / h).exp();
        let h_lo = h / c;
        let h_hi = h * c;

        let upper = (-h_lo).exp().clamp(0.0, 1.0);
        let lower = (-h_hi).exp().clamp(0.0, 1.0);

        Ok((lower.min(upper), lower.max(upper)))
    }
}

// ---------------------------------------------------------------------------
// log_rank_test
// ---------------------------------------------------------------------------

/// Log-rank test comparing survival between two independent groups.
///
/// # Arguments
/// * `group1_times`  – observed times for group 1.
/// * `group1_events` – event indicators for group 1 (`true` = event).
/// * `group2_times`  – observed times for group 2.
/// * `group2_events` – event indicators for group 2.
///
/// # Returns
/// `(statistic, p_value)` – chi-squared test statistic and two-sided p-value.
pub fn log_rank_test(
    group1_times: &[f64],
    group1_events: &[bool],
    group2_times: &[f64],
    group2_events: &[bool],
) -> StatsResult<(f64, f64)> {
    let result =
        KaplanMeier::log_rank_test(group1_times, group1_events, group2_times, group2_events)?;
    Ok(result)
}

// ---------------------------------------------------------------------------
// CoxPHModel
// ---------------------------------------------------------------------------

/// Cox proportional hazards model.
///
/// Wraps [`CoxPH`] with a slice-based interface.
pub struct CoxPHModel {
    inner: CoxPH,
}

impl CoxPHModel {
    /// Fit the Cox PH model via Newton-Raphson partial likelihood optimisation.
    ///
    /// # Arguments
    /// * `times`      – observed event/censoring times.
    /// * `events`     – event indicators.
    /// * `covariates` – n_samples × n_features covariate matrix.
    pub fn fit(times: &[f64], events: &[bool], covariates: &Array2<f64>) -> StatsResult<Self> {
        let inner = CoxPH::fit(times, events, covariates)?;
        Ok(Self { inner })
    }

    /// Log hazard-ratio coefficients β.
    pub fn coefficients(&self) -> Vec<f64> {
        self.inner.coefficients.iter().copied().collect()
    }

    /// Standard errors of the coefficients.
    pub fn standard_errors(&self) -> Vec<f64> {
        self.inner.std_errors.iter().copied().collect()
    }

    /// Two-sided Wald test p-values.
    pub fn p_values(&self) -> Vec<f64> {
        self.inner.p_values.iter().copied().collect()
    }

    /// Hazard ratios exp(β).
    pub fn hazard_ratios(&self) -> Vec<f64> {
        self.inner.hazard_ratio().iter().copied().collect()
    }

    /// Predict risk score exp(Xβ) for a single observation given as a slice.
    pub fn predict_risk(&self, x: &[f64]) -> f64 {
        use scirs2_core::ndarray::Array1;
        let arr = Array1::from_vec(x.to_vec());
        self.inner.predict_risk(&arr)
    }

    /// Concordance index (C-statistic) evaluated on provided data.
    ///
    /// Requires the covariate matrix used at prediction time.
    pub fn concordance_index(
        &self,
        times: &[f64],
        events: &[bool],
        covariates: &Array2<f64>,
    ) -> f64 {
        self.inner.concordance_index(times, events, covariates)
    }

    /// Partial log-likelihood at convergence.
    pub fn log_likelihood(&self) -> f64 {
        self.inner.log_likelihood
    }

    /// Number of Newton-Raphson iterations performed.
    pub fn n_iterations(&self) -> usize {
        self.inner.n_iter
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn sample_data() -> (Vec<f64>, Vec<bool>) {
        (
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            vec![
                true, true, false, true, false, true, true, false, true, true,
            ],
        )
    }

    // ----- KMCurve -----

    #[test]
    fn test_kmcurve_survival_starts_at_one() {
        let (times, events) = sample_data();
        let km = KMCurve::fit(&times, &events).expect("fit failed");
        assert_eq!(km.survival_function(0.0), 1.0);
    }

    #[test]
    fn test_kmcurve_survival_bounded() {
        let (times, events) = sample_data();
        let km = KMCurve::fit(&times, &events).expect("fit failed");
        for t in [0.0, 1.5, 5.0, 10.0, 20.0] {
            let s = km.survival_function(t);
            assert!(s >= 0.0 && s <= 1.0, "S({t}) = {s} out of [0,1]");
        }
    }

    #[test]
    fn test_kmcurve_survival_non_increasing() {
        let (times, events) = sample_data();
        let km = KMCurve::fit(&times, &events).expect("fit failed");
        let ts = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20.0];
        let mut prev = 1.0_f64;
        for &t in &ts {
            let s = km.survival_function(t);
            assert!(s <= prev + 1e-12, "S({t}) = {s} > S_prev = {prev}");
            prev = s;
        }
    }

    #[test]
    fn test_kmcurve_confidence_interval_ordering() {
        let (times, events) = sample_data();
        let km = KMCurve::fit(&times, &events).expect("fit failed");
        for t in [2.0, 5.0, 8.0] {
            let (lo, hi) = km.confidence_interval(t, 0.05).expect("CI failed");
            assert!(lo <= hi + 1e-10, "lo > hi at t={t}: {lo} {hi}");
            assert!(lo >= 0.0 && hi <= 1.0);
        }
    }

    #[test]
    fn test_kmcurve_ci_invalid_alpha() {
        let (times, events) = sample_data();
        let km = KMCurve::fit(&times, &events).expect("fit failed");
        assert!(km.confidence_interval(3.0, 0.0).is_err());
        assert!(km.confidence_interval(3.0, 1.0).is_err());
    }

    // ----- NACurve -----

    #[test]
    fn test_nacurve_survival_starts_at_one() {
        let (times, events) = sample_data();
        let na = NACurve::fit(&times, &events).expect("fit failed");
        assert_eq!(na.survival_function(0.0), 1.0);
    }

    #[test]
    fn test_nacurve_survival_bounded() {
        let (times, events) = sample_data();
        let na = NACurve::fit(&times, &events).expect("fit failed");
        for t in [0.0, 2.5, 6.0, 12.0] {
            let s = na.survival_function(t);
            assert!(s >= 0.0 && s <= 1.0, "S({t}) = {s} out of [0,1]");
        }
    }

    #[test]
    fn test_nacurve_confidence_interval_ordering() {
        let (times, events) = sample_data();
        let na = NACurve::fit(&times, &events).expect("fit failed");
        let (lo, hi) = na.confidence_interval(5.0, 0.05).expect("CI failed");
        assert!(lo <= hi + 1e-10, "lo > hi: {lo} {hi}");
        assert!(lo >= 0.0 && hi <= 1.0);
    }

    #[test]
    fn test_nacurve_ci_invalid_alpha() {
        let (times, events) = sample_data();
        let na = NACurve::fit(&times, &events).expect("fit failed");
        assert!(na.confidence_interval(3.0, 0.0).is_err());
        assert!(na.confidence_interval(3.0, 1.5).is_err());
    }

    // ----- log_rank_test -----

    #[test]
    fn test_log_rank_different_groups() {
        let times1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let events1 = vec![true, true, true, true, true];
        let times2 = vec![6.0, 7.0, 8.0, 9.0, 10.0];
        let events2 = vec![true, true, true, true, true];
        let (stat, p) =
            log_rank_test(&times1, &events1, &times2, &events2).expect("log_rank_test failed");
        assert!(stat >= 0.0, "statistic should be non-negative");
        assert!(p >= 0.0 && p <= 1.0, "p-value out of range: {p}");
        assert!(p < 0.05, "expected significant difference, p = {p}");
    }

    #[test]
    fn test_log_rank_identical_groups() {
        let times = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let events = vec![true, true, false, true, false, true];
        let (stat, p) =
            log_rank_test(&times, &events, &times, &events).expect("log_rank_test failed");
        assert!(stat < 1e-10, "identical groups: stat={stat}");
        assert!(p > 0.5, "identical groups should have large p={p}");
    }

    // ----- CoxPHModel -----

    #[test]
    fn test_coxph_fit_and_coefficients() {
        let times = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let events = vec![true, true, false, true, false, true, true, false];
        let x = Array2::from_shape_vec((8, 1), vec![0.1, 0.5, 0.2, 0.8, 0.3, 0.9, 0.4, 0.7])
            .expect("array failed");
        let model = CoxPHModel::fit(&times, &events, &x).expect("fit failed");
        assert_eq!(model.coefficients().len(), 1);
        assert!(model.coefficients()[0].is_finite());
    }

    #[test]
    fn test_coxph_log_likelihood_finite() {
        let times = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let events = vec![true, true, false, true, false, true, true, false];
        let x = Array2::from_shape_vec((8, 1), vec![0.1, 0.5, 0.2, 0.8, 0.3, 0.9, 0.4, 0.7])
            .expect("array failed");
        let model = CoxPHModel::fit(&times, &events, &x).expect("fit failed");
        assert!(model.log_likelihood().is_finite());
    }

    #[test]
    fn test_coxph_hazard_ratios_positive() {
        let times = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let events = vec![true, true, false, true, false, true, true, false];
        let x = Array2::from_shape_vec((8, 1), vec![0.1, 0.5, 0.2, 0.8, 0.3, 0.9, 0.4, 0.7])
            .expect("array failed");
        let model = CoxPHModel::fit(&times, &events, &x).expect("fit failed");
        for &hr in model.hazard_ratios().iter() {
            assert!(hr > 0.0, "HR should be positive, got {hr}");
        }
    }

    #[test]
    fn test_coxph_predict_risk_positive() {
        let times = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let events = vec![true, true, false, true, false, true, true, false];
        let x = Array2::from_shape_vec((8, 1), vec![0.1, 0.5, 0.2, 0.8, 0.3, 0.9, 0.4, 0.7])
            .expect("array failed");
        let model = CoxPHModel::fit(&times, &events, &x).expect("fit failed");
        let risk = model.predict_risk(&[0.5]);
        assert!(risk > 0.0, "risk should be positive, got {risk}");
    }

    #[test]
    fn test_coxph_concordance_index_in_range() {
        let times = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let events = vec![true, true, false, true, false, true, true, false];
        let x_data = vec![0.1, 0.5, 0.2, 0.8, 0.3, 0.9, 0.4, 0.7];
        let x = Array2::from_shape_vec((8, 1), x_data.clone()).expect("array failed");
        let model = CoxPHModel::fit(&times, &events, &x).expect("fit failed");
        let ci = model.concordance_index(&times, &events, &x);
        assert!(ci >= 0.0 && ci <= 1.0, "C-index out of [0,1]: {ci}");
    }
}
