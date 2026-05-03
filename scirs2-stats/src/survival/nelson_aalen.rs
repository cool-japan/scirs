//! Nelson-Aalen non-parametric cumulative hazard estimator.
//!
//! Provides:
//! - [`NelsonAalenEstimator`] – cumulative hazard H(t) = Σ d_k/n_k
//! - `survival_at` – exp(-H(t)) as a survival-function estimate
//! - Breslow baseline hazard estimator compatible with the Cox model

use crate::error::{StatsError, StatsResult};

// ---------------------------------------------------------------------------
// Nelson-Aalen Estimator
// ---------------------------------------------------------------------------

/// Result of fitting the Nelson-Aalen cumulative hazard estimator.
///
/// The Nelson-Aalen estimator is H(t) = Σ_{t_k ≤ t} d_k / n_k, where
/// d_k is the number of events at time t_k and n_k is the number at risk.
#[derive(Debug, Clone)]
pub struct NelsonAalenEstimator {
    /// Distinct event times (sorted, ascending).
    pub times: Vec<f64>,
    /// Cumulative hazard H(t) at each event time.
    pub cumulative_hazard: Vec<f64>,
    /// Standard error of H(t) at each event time (Nelson's variance estimator).
    pub std_err: Vec<f64>,
    /// Number at risk immediately before each event time.
    pub n_at_risk: Vec<usize>,
    /// Number of events at each distinct event time.
    pub n_events: Vec<usize>,
}

impl NelsonAalenEstimator {
    /// Fit the Nelson-Aalen estimator.
    ///
    /// # Arguments
    /// * `times`  – observed survival / censoring times (finite, ≥ 0).
    /// * `events` – `true` if an event occurred, `false` if censored.
    ///
    /// # Errors
    /// Returns [`StatsError`] on empty input, mismatched lengths, or invalid times.
    pub fn fit(times: &[f64], events: &[bool]) -> StatsResult<Self> {
        if times.is_empty() {
            return Err(StatsError::InvalidArgument(
                "times must not be empty".to_string(),
            ));
        }
        if times.len() != events.len() {
            return Err(StatsError::DimensionMismatch(format!(
                "times length {} != events length {}",
                times.len(),
                events.len()
            )));
        }
        for &t in times {
            if !t.is_finite() || t < 0.0 {
                return Err(StatsError::InvalidArgument(format!(
                    "times must be finite and non-negative; got {t}"
                )));
            }
        }

        // Pair and sort by time
        let mut pairs: Vec<(f64, bool)> =
            times.iter().copied().zip(events.iter().copied()).collect();
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let n_total = pairs.len();

        // Collect distinct event times
        let mut event_times: Vec<f64> = Vec::new();
        let mut d_counts: Vec<usize> = Vec::new();
        let mut n_risk_vec: Vec<usize> = Vec::new();

        let mut i = 0usize;
        let mut n_remaining = n_total;

        while i < pairs.len() {
            let t_cur = pairs[i].0;
            let mut n_events_at_t = 0usize;
            let mut n_censored_at_t = 0usize;
            while i < pairs.len() && (pairs[i].0 - t_cur).abs() < 1e-14 {
                if pairs[i].1 {
                    n_events_at_t += 1;
                } else {
                    n_censored_at_t += 1;
                }
                i += 1;
            }
            if n_events_at_t > 0 {
                event_times.push(t_cur);
                d_counts.push(n_events_at_t);
                n_risk_vec.push(n_remaining);
            }
            n_remaining -= n_events_at_t + n_censored_at_t;
        }

        // Compute Nelson-Aalen cumulative hazard and variance
        let mut na_times = Vec::with_capacity(event_times.len());
        let mut na_hazard = Vec::with_capacity(event_times.len());
        let mut na_std_err = Vec::with_capacity(event_times.len());
        let mut na_n_risk = Vec::with_capacity(event_times.len());
        let mut na_n_events = Vec::with_capacity(event_times.len());

        let mut h = 0.0_f64;
        let mut var_h = 0.0_f64;

        for k in 0..event_times.len() {
            let n_k = n_risk_vec[k] as f64;
            let d_k = d_counts[k] as f64;

            // Nelson increment: d_k / n_k
            h += d_k / n_k;
            // Variance increment (Nelson's): d_k / n_k^2
            var_h += d_k / (n_k * n_k);

            na_times.push(event_times[k]);
            na_hazard.push(h);
            na_std_err.push(var_h.sqrt());
            na_n_risk.push(n_risk_vec[k]);
            na_n_events.push(d_counts[k]);
        }

        Ok(Self {
            times: na_times,
            cumulative_hazard: na_hazard,
            std_err: na_std_err,
            n_at_risk: na_n_risk,
            n_events: na_n_events,
        })
    }

    /// Evaluate the cumulative hazard H(t) as a step function.
    ///
    /// Returns 0 before the first event time.
    pub fn hazard_at(&self, t: f64) -> f64 {
        if self.times.is_empty() || t < self.times[0] {
            return 0.0;
        }
        let idx = self.times.partition_point(|&tk| tk <= t).saturating_sub(1);
        self.cumulative_hazard[idx]
    }

    /// Breslow survival estimate: S(t) = exp(-H(t)).
    pub fn survival_at(&self, t: f64) -> f64 {
        (-self.hazard_at(t)).exp()
    }

    /// Pointwise confidence interval for the cumulative hazard H(t).
    ///
    /// Uses log transform: CI = H(t) × exp(±z × σ(t) / H(t)).
    ///
    /// # Arguments
    /// * `t`     – time point.
    /// * `alpha` – significance level (e.g., 0.05 for 95% CI).
    ///
    /// # Returns
    /// `(lower, upper)` for H(t).
    pub fn confidence_interval(&self, t: f64, alpha: f64) -> (f64, f64) {
        let h = self.hazard_at(t);
        if h <= 0.0 {
            return (0.0, 0.0);
        }
        let z = norm_ppf(1.0 - alpha / 2.0);

        // Accumulated variance up to t
        let var_h: f64 = self
            .times
            .iter()
            .enumerate()
            .take_while(|(_, &tk)| tk <= t)
            .map(|(k, _)| {
                let n_k = self.n_at_risk[k] as f64;
                let d_k = self.n_events[k] as f64;
                d_k / (n_k * n_k)
            })
            .sum();
        let se_h = var_h.sqrt();

        // Log-transform CI on H
        let w = z * se_h / h;
        let lower = h * (-w).exp();
        let upper = h * w.exp();
        (lower.max(0.0), upper)
    }

    /// Breslow estimator of the baseline survival function given Cox risk scores.
    ///
    /// H₀(t) = Σ_{t_k ≤ t} d_k / Σ_{j in risk set at t_k} exp(x_j β)
    ///
    /// # Arguments
    /// * `risk_scores` – `exp(x_i β)` for each subject in the fitting dataset.
    /// * `pairs`       – `(time, is_event)` for each subject (same order as `risk_scores`).
    ///
    /// # Returns
    /// `(times, baseline_cumulative_hazard)` aligned vectors.
    pub fn breslow_baseline(
        risk_scores: &[f64],
        pairs: &[(f64, bool)],
    ) -> StatsResult<(Vec<f64>, Vec<f64>)> {
        if risk_scores.len() != pairs.len() {
            return Err(StatsError::DimensionMismatch(format!(
                "risk_scores length {} != pairs length {}",
                risk_scores.len(),
                pairs.len()
            )));
        }
        let n = pairs.len();
        // Sort indices by time
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by(|&a, &b| {
            pairs[a]
                .0
                .partial_cmp(&pairs[b].0)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut times_out = Vec::new();
        let mut hazard_out = Vec::new();
        let mut cum_h = 0.0_f64;

        let mut pos = 0usize;
        while pos < n {
            let t_cur = pairs[idx[pos]].0;
            // Collect all obs at t_cur
            let mut d_k = 0usize;
            let mut end = pos;
            while end < n && (pairs[idx[end]].0 - t_cur).abs() < 1e-14 {
                if pairs[idx[end]].1 {
                    d_k += 1;
                }
                end += 1;
            }
            if d_k > 0 {
                // Sum of risk scores in the risk set (all with time >= t_cur)
                let risk_set_sum: f64 = idx[pos..].iter().map(|&i| risk_scores[i]).sum();
                if risk_set_sum > 1e-300 {
                    cum_h += d_k as f64 / risk_set_sum;
                }
                times_out.push(t_cur);
                hazard_out.push(cum_h);
            }
            pos = end;
        }
        Ok((times_out, hazard_out))
    }
}

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
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_data() -> (Vec<f64>, Vec<bool>) {
        let times = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let events = vec![
            true, true, false, true, true, false, true, false, true, true,
        ];
        (times, events)
    }

    #[test]
    fn test_na_fit_basic() {
        let (times, events) = simple_data();
        let na = NelsonAalenEstimator::fit(&times, &events).expect("NA fit failed");
        assert!(!na.times.is_empty());
        assert_eq!(na.times.len(), na.cumulative_hazard.len());
        assert_eq!(na.times.len(), na.std_err.len());
    }

    #[test]
    fn test_na_hazard_monotone_increasing() {
        let (times, events) = simple_data();
        let na = NelsonAalenEstimator::fit(&times, &events).expect("NA fit");
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
        let na = NelsonAalenEstimator::fit(&times, &events).expect("NA fit");
        for &h in &na.cumulative_hazard {
            let s = (-h).exp();
            assert!(s >= 0.0 && s <= 1.0 + 1e-12, "S(t)={s} out of [0,1]");
        }
    }

    #[test]
    fn test_na_zero_before_first_event() {
        let (times, events) = simple_data();
        let na = NelsonAalenEstimator::fit(&times, &events).expect("NA fit");
        assert!((na.hazard_at(0.0) - 0.0).abs() < 1e-12);
        assert!((na.survival_at(0.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_na_confidence_interval() {
        let (times, events) = simple_data();
        let na = NelsonAalenEstimator::fit(&times, &events).expect("NA fit");
        let (lo, hi) = na.confidence_interval(5.0, 0.05);
        assert!(lo >= 0.0, "lower {lo} should be non-negative");
        assert!(hi >= lo, "upper should be >= lower");
    }

    #[test]
    fn test_na_std_err_non_negative() {
        let (times, events) = simple_data();
        let na = NelsonAalenEstimator::fit(&times, &events).expect("NA fit");
        for &se in &na.std_err {
            assert!(se >= 0.0, "std_err {se} should be non-negative");
        }
    }

    #[test]
    fn test_na_error_empty() {
        let result = NelsonAalenEstimator::fit(&[], &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_na_error_negative_time() {
        let result = NelsonAalenEstimator::fit(&[-1.0, 2.0], &[true, true]);
        assert!(result.is_err());
    }

    #[test]
    fn test_na_error_mismatch() {
        let result = NelsonAalenEstimator::fit(&[1.0, 2.0], &[true]);
        assert!(result.is_err());
    }

    #[test]
    fn test_breslow_baseline() {
        let pairs = vec![
            (1.0, true),
            (2.0, true),
            (3.0, false),
            (4.0, true),
            (5.0, true),
        ];
        let risk_scores = vec![1.0, 1.2, 0.8, 1.5, 0.9];
        let (bt, bh) =
            NelsonAalenEstimator::breslow_baseline(&risk_scores, &pairs).expect("breslow failed");
        assert_eq!(bt.len(), bh.len());
        // Cumulative hazard should be monotone increasing
        for i in 1..bh.len() {
            assert!(bh[i] >= bh[i - 1] - 1e-12);
        }
    }
}
