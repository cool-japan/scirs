//! Kaplan-Meier non-parametric survival estimator.
//!
//! Provides:
//! - [`KaplanMeierEstimator`] – the canonical KM estimator with Greenwood confidence intervals
//! - [`log_rank_test`]        – two-sample log-rank test (chi-square statistic + p-value)

use crate::error::{StatsError, StatsResult};

// ---------------------------------------------------------------------------
// Statistical helper functions
// ---------------------------------------------------------------------------

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

fn norm_cdf(z: f64) -> f64 {
    0.5 * (1.0 + erf_approx(z / std::f64::consts::SQRT_2))
}

/// Regularised incomplete gamma: Q(a, x) = 1 - P(a, x)
fn gamma_q(a: f64, x: f64) -> f64 {
    if x < 0.0 {
        return 1.0;
    }
    if x == 0.0 {
        return 1.0;
    }
    if x < a + 1.0 {
        // Series expansion for P(a, x)
        let mut ap = a;
        let mut sum = 1.0 / a;
        let mut del = sum;
        for _ in 0..200 {
            ap += 1.0;
            del *= x / ap;
            sum += del;
            if del.abs() < sum.abs() * 3e-15 {
                break;
            }
        }
        let p = sum * (-x + a * x.ln() - lgamma(a)).exp();
        1.0 - p
    } else {
        // Continued-fraction for Q(a, x)
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / 1e-300;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1_i64..200 {
            let an = -(i as f64) * (i as f64 - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < 1e-300 {
                d = 1e-300;
            }
            c = b + an / c;
            if c.abs() < 1e-300 {
                c = 1e-300;
            }
            d = 1.0 / d;
            let del = d * c;
            h *= del;
            if (del - 1.0).abs() < 3e-15 {
                break;
            }
        }
        (-x + a * x.ln() - lgamma(a)).exp() * h
    }
}

fn lgamma(x: f64) -> f64 {
    // Lanczos approximation
    let g = 7.0_f64;
    let c = [
        0.999_999_999_999_809_93,
        676.520_368_121_885_10,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_10,
        -176.615_029_162_140_60,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    let x = x - 1.0;
    let mut ser = c[0];
    for (i, &ci) in c[1..].iter().enumerate() {
        ser += ci / (x + i as f64 + 1.0);
    }
    let tmp = x + g + 0.5;
    0.5 * std::f64::consts::TAU.ln() + (x + 0.5) * tmp.ln() - tmp + ser.ln()
}

/// Chi-square survival function P(X > x) for df degrees of freedom.
fn chi2_sf(x: f64, df: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    gamma_q(df / 2.0, x / 2.0)
}

// ---------------------------------------------------------------------------
// Kaplan-Meier Estimator
// ---------------------------------------------------------------------------

/// Result of fitting the Kaplan-Meier non-parametric survival estimator.
///
/// The `times` vector holds the distinct event times (not censoring times).
/// All other vectors are aligned with `times`.
#[derive(Debug, Clone)]
pub struct KaplanMeierEstimator {
    /// Distinct event times (sorted, ascending).
    pub times: Vec<f64>,
    /// KM survival probability S(t) at each event time.
    pub survival: Vec<f64>,
    /// Greenwood standard error of S(t) at each event time.
    pub std_err: Vec<f64>,
    /// Number at risk immediately before each event time.
    pub n_at_risk: Vec<usize>,
    /// Number of events at each distinct event time.
    pub n_events: Vec<usize>,
}

impl KaplanMeierEstimator {
    /// Fit the Kaplan-Meier estimator.
    ///
    /// # Arguments
    /// * `times`  – observed survival / censoring times (must be finite and ≥ 0).
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
        let mut d_counts: Vec<usize> = Vec::new(); // events at each time
        let mut n_risk: Vec<usize> = Vec::new(); // n at risk before each time

        let mut i = 0usize;
        let mut n_remaining = n_total;

        while i < pairs.len() {
            let t_cur = pairs[i].0;
            let mut n_events_at_t = 0usize;
            let mut n_censored_at_t = 0usize;
            let j = i;
            // Collect all observations at t_cur
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
                n_risk.push(n_remaining);
            }
            n_remaining -= n_events_at_t + n_censored_at_t;
            let _ = j; // suppress unused variable warning
        }

        // Compute KM survival and Greenwood variance
        let mut km_times = Vec::with_capacity(event_times.len());
        let mut km_survival = Vec::with_capacity(event_times.len());
        let mut km_std_err = Vec::with_capacity(event_times.len());
        let mut km_n_risk = Vec::with_capacity(event_times.len());
        let mut km_n_events = Vec::with_capacity(event_times.len());

        let mut s = 1.0_f64;
        let mut greenwood_sum = 0.0_f64;

        for k in 0..event_times.len() {
            let n_k = n_risk[k] as f64;
            let d_k = d_counts[k] as f64;

            s *= (n_k - d_k) / n_k;

            // Greenwood term: d_k / (n_k * (n_k - d_k))
            if n_k > d_k {
                greenwood_sum += d_k / (n_k * (n_k - d_k));
            }
            let var_s = s * s * greenwood_sum;
            let se = var_s.sqrt();

            km_times.push(event_times[k]);
            km_survival.push(s);
            km_std_err.push(se);
            km_n_risk.push(n_risk[k]);
            km_n_events.push(d_counts[k]);
        }

        Ok(Self {
            times: km_times,
            survival: km_survival,
            std_err: km_std_err,
            n_at_risk: km_n_risk,
            n_events: km_n_events,
        })
    }

    /// Evaluate S(t) as a left-continuous step function.
    ///
    /// Returns 1.0 before the first event, and the last KM value after the last event.
    pub fn survival_at(&self, t: f64) -> f64 {
        if self.times.is_empty() || t < self.times[0] {
            return 1.0;
        }
        // Find last index where times[k] <= t
        let idx = self.times.partition_point(|&tk| tk <= t).saturating_sub(1);
        self.survival[idx]
    }

    /// Compute a pointwise Greenwood confidence interval for S(t).
    ///
    /// Uses the log-log transform (`log(-log S(t))`) for better small-sample
    /// coverage and guaranteed bounds in [0, 1].
    ///
    /// # Arguments
    /// * `t`     – time point.
    /// * `alpha` – significance level (e.g., 0.05 for 95% CI).
    ///
    /// # Returns
    /// `(lower, upper)` both clamped to [0, 1].
    pub fn confidence_interval(&self, t: f64, alpha: f64) -> (f64, f64) {
        let s = self.survival_at(t);
        if s <= 0.0 || s >= 1.0 {
            return (s.clamp(0.0, 1.0), s.clamp(0.0, 1.0));
        }

        // z_{1-alpha/2}
        let z = norm_ppf(1.0 - alpha / 2.0);

        // Greenwood cumulative variance at t
        let greenwood: f64 = self
            .times
            .iter()
            .enumerate()
            .take_while(|(_, &tk)| tk <= t)
            .map(|(k, _)| {
                let n_k = self.n_at_risk[k] as f64;
                let d_k = self.n_events[k] as f64;
                if n_k > d_k {
                    d_k / (n_k * (n_k - d_k))
                } else {
                    0.0
                }
            })
            .sum();

        // Log-log transform
        let log_s = s.ln();
        if log_s == 0.0 {
            return (s, s);
        }
        let w = (z * greenwood.sqrt()) / log_s.abs();
        // log-log CI: S(t)^exp(±w), where S(t) < 1
        // Since S < 1, S^(smaller exponent) > S^(larger exponent)
        // exp(w) > exp(-w) for w > 0, so S^exp(w) is the lower bound
        let lower = s.powf(w.exp());
        let upper = s.powf((-w).exp());
        (lower.clamp(0.0, 1.0), upper.clamp(0.0, 1.0))
    }

    /// Median survival time (smallest t with S(t) ≤ 0.5).
    ///
    /// Returns `None` if the survival never drops to or below 0.5.
    pub fn median_survival(&self) -> Option<f64> {
        for (k, &s) in self.survival.iter().enumerate() {
            if s <= 0.5 {
                return Some(self.times[k]);
            }
        }
        None
    }
}

/// Normal quantile function (inverse CDF) using Acklam's rational approximation.
///
/// Provides full double-precision accuracy across the entire (0, 1) range.
fn norm_ppf(p: f64) -> f64 {
    let p = p.clamp(1e-15, 1.0 - 1e-15);

    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_690e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239e0,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838e0,
        -2.549_732_539_343_734e0,
        4.374_664_141_464_968e0,
        2.938_163_982_698_783e0,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996e0,
        3.754_408_661_907_416e0,
    ];

    const P_LOW: f64 = 0.02425;
    const P_HIGH: f64 = 1.0 - P_LOW;

    if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= P_HIGH {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

// ---------------------------------------------------------------------------
// Log-rank test
// ---------------------------------------------------------------------------

/// Two-sample log-rank test for equality of survival distributions.
///
/// Returns `(statistic, p_value)` where `statistic` follows a chi-square
/// distribution with 1 degree of freedom under H₀.
///
/// # Arguments
/// * `group1_t` / `group1_e` – times and event indicators for group 1.
/// * `group2_t` / `group2_e` – times and event indicators for group 2.
///
/// # Errors
/// Returns an error if either group is empty or lengths are mismatched.
pub fn log_rank_test(
    group1_t: &[f64],
    group1_e: &[bool],
    group2_t: &[f64],
    group2_e: &[bool],
) -> StatsResult<(f64, f64)> {
    if group1_t.is_empty() || group2_t.is_empty() {
        return Err(StatsError::InvalidArgument(
            "Both groups must be non-empty".to_string(),
        ));
    }
    if group1_t.len() != group1_e.len() {
        return Err(StatsError::DimensionMismatch(format!(
            "group1: times length {} != events length {}",
            group1_t.len(),
            group1_e.len()
        )));
    }
    if group2_t.len() != group2_e.len() {
        return Err(StatsError::DimensionMismatch(format!(
            "group2: times length {} != events length {}",
            group2_t.len(),
            group2_e.len()
        )));
    }

    // Collect all distinct event times across both groups
    let mut all_event_times: Vec<f64> = group1_t
        .iter()
        .zip(group1_e.iter())
        .filter_map(|(&t, &e)| if e { Some(t) } else { None })
        .chain(
            group2_t
                .iter()
                .zip(group2_e.iter())
                .filter_map(|(&t, &e)| if e { Some(t) } else { None }),
        )
        .collect();
    all_event_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    all_event_times.dedup_by(|a, b| (*a - *b).abs() < 1e-14);

    if all_event_times.is_empty() {
        return Err(StatsError::InvalidArgument(
            "No events observed in either group".to_string(),
        ));
    }

    let mut o_minus_e_sum = 0.0_f64;
    let mut var_sum = 0.0_f64;

    for &t in &all_event_times {
        // Number at risk in each group at time t
        let n1 = group1_t.iter().filter(|&&ti| ti >= t).count() as f64;
        let n2 = group2_t.iter().filter(|&&ti| ti >= t).count() as f64;
        let n = n1 + n2;

        // Number of events in each group at time t
        let d1 = group1_t
            .iter()
            .zip(group1_e.iter())
            .filter(|(&ti, &ei)| (ti - t).abs() < 1e-14 && ei)
            .count() as f64;
        let d2 = group2_t
            .iter()
            .zip(group2_e.iter())
            .filter(|(&ti, &ei)| (ti - t).abs() < 1e-14 && ei)
            .count() as f64;
        let d = d1 + d2;

        if n <= 1.0 || d == 0.0 {
            continue;
        }

        let e1 = n1 * d / n;
        o_minus_e_sum += d1 - e1;

        // Hypergeometric variance
        let var = n1 * n2 * d * (n - d) / (n * n * (n - 1.0));
        var_sum += var;
    }

    if var_sum <= 0.0 {
        return Ok((0.0, 1.0));
    }

    let statistic = o_minus_e_sum * o_minus_e_sum / var_sum;
    let p_value = chi2_sf(statistic, 1.0);

    Ok((statistic, p_value))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn leukemia_data() -> (Vec<f64>, Vec<bool>) {
        // Classic leukemia remission data (Gehan 1965)
        let times = vec![
            6.0, 6.0, 6.0, 7.0, 10.0, 13.0, 16.0, 22.0, 23.0, 6.0, 9.0, 10.0, 11.0, 17.0, 19.0,
            20.0, 25.0, 32.0, 32.0, 34.0, 35.0,
        ];
        let events = vec![
            true, true, true, true, true, true, true, true, true, false, false, false, false,
            false, false, false, false, false, false, false, false,
        ];
        (times, events)
    }

    #[test]
    fn test_km_fit_basic() {
        let times = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let events = vec![true, true, false, true, false];
        let km = KaplanMeierEstimator::fit(&times, &events).expect("KM fit failed");
        assert!(!km.times.is_empty());
        // Should only have event times: 1, 2, 4
        assert_eq!(km.times, vec![1.0, 2.0, 4.0]);
        assert_eq!(km.n_at_risk, vec![5, 4, 2]);
        assert_eq!(km.n_events, vec![1, 1, 1]);
    }

    #[test]
    fn test_km_survival_decreasing() {
        let (times, events) = leukemia_data();
        let km = KaplanMeierEstimator::fit(&times, &events).expect("KM fit");
        for i in 1..km.survival.len() {
            assert!(km.survival[i] <= km.survival[i - 1] + 1e-12);
        }
    }

    #[test]
    fn test_km_survival_bounded() {
        let (times, events) = leukemia_data();
        let km = KaplanMeierEstimator::fit(&times, &events).expect("KM fit");
        for &s in &km.survival {
            assert!(s >= 0.0 && s <= 1.0 + 1e-12, "survival {s} out of [0,1]");
        }
    }

    #[test]
    fn test_km_survival_at_step_function() {
        let times = vec![1.0, 2.0, 3.0];
        let events = vec![true, true, true];
        let km = KaplanMeierEstimator::fit(&times, &events).expect("KM fit");
        assert!((km.survival_at(0.5) - 1.0).abs() < 1e-12);
        assert_eq!(km.survival_at(1.0), km.survival[0]);
        assert_eq!(km.survival_at(1.5), km.survival[0]);
        assert_eq!(km.survival_at(2.0), km.survival[1]);
    }

    #[test]
    fn test_km_confidence_interval() {
        let (times, events) = leukemia_data();
        let km = KaplanMeierEstimator::fit(&times, &events).expect("KM fit");
        let (lo, hi) = km.confidence_interval(6.0, 0.05);
        assert!(lo >= 0.0 && lo <= 1.0, "lower {lo} out of range");
        assert!(hi >= 0.0 && hi <= 1.0, "upper {hi} out of range");
        assert!(lo <= hi + 1e-12, "lower > upper");
        assert!(lo <= km.survival_at(6.0) + 1e-12);
        assert!(hi >= km.survival_at(6.0) - 1e-12);
    }

    #[test]
    fn test_km_median_survival() {
        let times = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let events = vec![true, true, true, true, true, true];
        let km = KaplanMeierEstimator::fit(&times, &events).expect("KM fit");
        let median = km.median_survival();
        assert!(median.is_some());
        let m = median.expect("median should be Some after assert");
        assert!(km.survival_at(m) <= 0.5 + 1e-12);
    }

    #[test]
    fn test_km_all_censored_no_events() {
        // All censored → no event times → empty KM
        let times = vec![1.0, 2.0, 3.0];
        let events = vec![false, false, false];
        let km = KaplanMeierEstimator::fit(&times, &events).expect("KM fit");
        assert!(km.times.is_empty());
        assert!((km.survival_at(10.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_km_error_empty() {
        let result = KaplanMeierEstimator::fit(&[], &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_km_error_mismatch() {
        let result = KaplanMeierEstimator::fit(&[1.0, 2.0], &[true]);
        assert!(result.is_err());
    }

    #[test]
    fn test_km_error_negative_time() {
        let result = KaplanMeierEstimator::fit(&[-1.0, 2.0], &[true, true]);
        assert!(result.is_err());
    }

    #[test]
    fn test_log_rank_different_groups() {
        // Treatment group (longer survival) vs control
        let t1 = vec![10.0, 12.0, 15.0, 18.0, 20.0, 25.0, 30.0];
        let e1 = vec![true, true, true, false, true, true, false];
        let t2 = vec![2.0, 3.0, 5.0, 7.0, 8.0, 9.0, 11.0];
        let e2 = vec![true, true, true, true, true, true, true];
        let (stat, pval) = log_rank_test(&t1, &e1, &t2, &e2).expect("log-rank failed");
        assert!(stat >= 0.0, "statistic must be non-negative");
        assert!(pval >= 0.0 && pval <= 1.0, "p-value must be in [0,1]");
        // Groups are quite different so p-value should be small
        assert!(pval < 0.10, "expected significant result, got p={pval}");
    }

    #[test]
    fn test_log_rank_identical_groups() {
        let t = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let e = vec![true, true, true, true, true];
        let (stat, pval) = log_rank_test(&t, &e, &t, &e).expect("log-rank failed");
        // Identical groups → statistic ≈ 0, p ≈ 1
        assert!(stat.abs() < 1e-10, "statistic should be ~0, got {stat}");
        assert!(
            pval > 0.5,
            "p-value should be large for identical groups, got {pval}"
        );
    }

    #[test]
    fn test_log_rank_error_empty() {
        let result = log_rank_test(&[], &[], &[1.0], &[true]);
        assert!(result.is_err());
    }
}
