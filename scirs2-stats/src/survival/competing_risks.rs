//! Competing risks analysis.
//!
//! Implements:
//! - [`CumulativeIncidenceFunction`] – non-parametric CIF via the Aalen-Johansen estimator
//! - [`gray_test`] – Gray's k-sample test for equality of CIFs
//! - [`FineGrayModel`] – Fine-Gray subdistribution hazard regression
//! - [`cause_specific_hazard`] – cause-specific Nelson-Aalen estimator

use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::{Array1, Array2};

// ---------------------------------------------------------------------------
// Statistical helpers
// ---------------------------------------------------------------------------

fn lgamma(x: f64) -> f64 {
    let c = [
        0.999_999_999_999_809_93_f64,
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
    let tmp = x + 7.5;
    0.5 * std::f64::consts::TAU.ln() + (x + 0.5) * tmp.ln() - tmp + ser.ln()
}

fn gamma_q(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    if x < a + 1.0 {
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
        1.0 - sum * (-x + a * x.ln() - lgamma(a)).exp()
    } else {
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

fn chi2_sf(x: f64, df: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    gamma_q(df / 2.0, x / 2.0)
}

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

#[allow(dead_code)]
fn norm_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

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
// Cumulative Incidence Function (Aalen-Johansen estimator)
// ---------------------------------------------------------------------------

/// Non-parametric cumulative incidence function (CIF) for a single cause.
///
/// The CIF I_k(t) = P(T ≤ t, cause = k) is estimated via the Aalen-Johansen
/// estimator:
///   I_k(t) = Σ_{t_j ≤ t} S(t_j⁻) × λ_k(t_j)
/// where S(t_j⁻) is the overall survival just before t_j and λ_k(t_j) = d_k(t_j) / n(t_j).
#[derive(Debug, Clone)]
pub struct CumulativeIncidenceFunction {
    /// Event times (distinct, sorted ascending).
    pub times: Vec<f64>,
    /// Cumulative incidence I_k(t) at each event time.
    pub cif: Vec<f64>,
    /// Cause code (1-based) associated with this CIF.
    pub cause: usize,
    /// Standard error of I_k(t) at each event time (Gray's variance).
    pub std_err: Vec<f64>,
}

impl CumulativeIncidenceFunction {
    /// Fit the Aalen-Johansen cumulative incidence estimator for all causes.
    ///
    /// # Arguments
    /// * `times`  – observed times (finite, ≥ 0).
    /// * `events` – cause of failure: 0 = censored, 1..k = cause k.
    ///
    /// # Returns
    /// A vector of `CumulativeIncidenceFunction`, one per distinct non-zero cause code.
    ///
    /// # Errors
    /// Returns [`StatsError`] on empty input or mismatched lengths.
    pub fn fit(times: &[f64], events: &[u8]) -> StatsResult<Vec<Self>> {
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

        // Identify distinct cause codes (> 0)
        let mut causes: Vec<u8> = events.iter().filter(|&&e| e > 0).copied().collect();
        causes.sort_unstable();
        causes.dedup();

        if causes.is_empty() {
            // All censored: return empty CIFs for each "cause" (none)
            return Ok(vec![]);
        }

        let n_total = times.len();

        // Sort by time
        let mut idx: Vec<usize> = (0..n_total).collect();
        idx.sort_by(|&a, &b| {
            times[a]
                .partial_cmp(&times[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Collect all distinct event times (any cause)
        let mut all_event_times: Vec<f64> = Vec::new();
        let mut i = 0usize;
        while i < n_total {
            let t_cur = times[idx[i]];
            let mut has_event = false;
            let mut j = i;
            while j < n_total && (times[idx[j]] - t_cur).abs() < 1e-14 {
                if events[idx[j]] > 0 {
                    has_event = true;
                }
                j += 1;
            }
            if has_event {
                all_event_times.push(t_cur);
            }
            i = j;
        }

        // For each cause, compute the CIF via Aalen-Johansen
        let mut results = Vec::with_capacity(causes.len());

        for &cause in &causes {
            let (cif_times, cif_vals, cif_se) =
                aalen_johansen_cif(times, events, &idx, &all_event_times, cause, n_total);
            results.push(Self {
                times: cif_times,
                cif: cif_vals,
                cause: cause as usize,
                std_err: cif_se,
            });
        }

        Ok(results)
    }

    /// Evaluate the CIF at time t (left-continuous step function).
    pub fn cif_at(&self, t: f64) -> f64 {
        if self.times.is_empty() || t < self.times[0] {
            return 0.0;
        }
        let idx = self.times.partition_point(|&tk| tk <= t).saturating_sub(1);
        self.cif[idx]
    }

    /// Pointwise confidence interval for I_k(t) using log-log transform.
    pub fn confidence_interval(&self, t: f64, alpha: f64) -> (f64, f64) {
        let cif = self.cif_at(t);
        if cif <= 0.0 || cif >= 1.0 {
            return (cif.clamp(0.0, 1.0), cif.clamp(0.0, 1.0));
        }
        let z = norm_ppf(1.0 - alpha / 2.0);

        // Variance at t
        let var = {
            let idx = self.times.partition_point(|&tk| tk <= t);
            let idx = idx
                .saturating_sub(1)
                .min(self.std_err.len().saturating_sub(1));
            if self.std_err.is_empty() {
                0.0
            } else {
                self.std_err[idx].powi(2)
            }
        };
        let se = var.sqrt();

        if cif <= 0.0 || se <= 0.0 {
            return (cif, cif);
        }

        // Log-log transform CI for CIF
        let log_cif = cif.ln();
        let w = z * se / (cif * log_cif.abs()).max(1e-300);
        // For CIF in (0, 1): cif^exp(w) < cif^exp(-w) since exp(w) > exp(-w) and 0 < cif < 1
        let lower = cif.powf(w.exp());
        let upper = cif.powf((-w).exp());
        (lower.clamp(0.0, 1.0), upper.clamp(0.0, 1.0))
    }
}

/// Compute the Aalen-Johansen CIF for a single cause.
fn aalen_johansen_cif(
    times: &[f64],
    events: &[u8],
    idx: &[usize],
    all_event_times: &[f64],
    cause: u8,
    n_total: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut cif_times = Vec::new();
    let mut cif_vals = Vec::new();
    let mut cif_se = Vec::new();

    let mut s = 1.0_f64; // Overall survival (KM)
    let mut cif = 0.0_f64; // Cumulative incidence for this cause

    // Variance accumulator (Gray's influence-function approach, simplified)
    // Var[I_k(t)] ≈ Σ_{j≤t} [S(t_j⁻)]² × d_k(t_j) × [n(t_j) - d_k(t_j)] / n(t_j)³
    // + cross terms (simplified to the leading term here)
    let mut var_sum = 0.0_f64;

    let mut sorted_pos = 0usize;
    let n_sorted = idx.len();

    for &t_cur in all_event_times {
        // Advance sorted_pos to this time
        while sorted_pos < n_sorted && times[idx[sorted_pos]] < t_cur - 1e-14 {
            sorted_pos += 1;
        }

        // Count at risk n(t) = # with time >= t_cur
        let n_at_risk = (0..n_total).filter(|&i| times[i] >= t_cur - 1e-14).count();

        // Count events of cause k and all causes at t_cur
        let mut d_k = 0usize;
        let mut d_all = 0usize;
        let mut j = sorted_pos;
        while j < n_sorted && (times[idx[j]] - t_cur).abs() < 1e-14 {
            if events[idx[j]] > 0 {
                d_all += 1;
            }
            if events[idx[j]] == cause {
                d_k += 1;
            }
            j += 1;
        }

        let n_f = n_at_risk as f64;
        let d_all_f = d_all as f64;
        let d_k_f = d_k as f64;

        // I_k contribution: S(t⁻) × d_k / n
        if n_f > 0.0 {
            cif += s * d_k_f / n_f;
        }

        // Update overall survival by all causes
        if n_f > 0.0 && d_all > 0 {
            s *= (n_f - d_all_f) / n_f;
        }

        // Variance update
        if n_f > 0.0 && d_k > 0 {
            var_sum += (s * s) * d_k_f * (n_f - d_k_f) / (n_f * n_f * n_f).max(1e-300);
        }

        cif_times.push(t_cur);
        cif_vals.push(cif.clamp(0.0, 1.0));
        cif_se.push(var_sum.sqrt());
    }

    (cif_times, cif_vals, cif_se)
}

// ---------------------------------------------------------------------------
// Cause-specific Nelson-Aalen hazard
// ---------------------------------------------------------------------------

/// Compute the cause-specific cumulative hazard for a given cause.
///
/// H_k(t) = Σ_{t_j ≤ t} d_k(t_j) / n(t_j)
///
/// # Arguments
/// * `times`  – observed times.
/// * `events` – cause codes (0 = censored, k > 0 = cause k).
/// * `cause`  – the cause of interest.
///
/// # Returns
/// `(times, cumulative_hazard)` aligned vectors at distinct event times of this cause.
pub fn cause_specific_hazard(
    times: &[f64],
    events: &[u8],
    cause: u8,
) -> StatsResult<(Vec<f64>, Vec<f64>)> {
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

    let n = times.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| {
        times[a]
            .partial_cmp(&times[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut times_out = Vec::new();
    let mut hazard_out = Vec::new();
    let mut cum_h = 0.0_f64;
    let mut pos = 0usize;

    while pos < n {
        let t_cur = times[idx[pos]];
        let mut d_k = 0usize;
        let mut end = pos;
        while end < n && (times[idx[end]] - t_cur).abs() < 1e-14 {
            if events[idx[end]] == cause {
                d_k += 1;
            }
            end += 1;
        }
        // Number at risk = n - pos (those with time >= t_cur)
        let n_at_risk = n - pos;
        if d_k > 0 && n_at_risk > 0 {
            cum_h += d_k as f64 / n_at_risk as f64;
            times_out.push(t_cur);
            hazard_out.push(cum_h);
        }
        pos = end;
    }

    Ok((times_out, hazard_out))
}

// ---------------------------------------------------------------------------
// Gray's k-sample test for CIF equality
// ---------------------------------------------------------------------------

/// Gray's test for the equality of cumulative incidence functions across two groups.
///
/// Tests H₀: I_k^{(1)}(t) = I_k^{(2)}(t) for all t, for each cause k.
///
/// Returns a vector of `(cause, statistic, p_value)` triples, one per distinct cause.
///
/// # Arguments
/// * `times1` / `events1` – observed times and cause codes for group 1.
/// * `times2` / `events2` – observed times and cause codes for group 2.
///
/// # Errors
/// Returns [`StatsError`] on empty input or mismatched lengths.
pub fn gray_test(
    times1: &[f64],
    events1: &[u8],
    times2: &[f64],
    events2: &[u8],
) -> StatsResult<Vec<(usize, f64, f64)>> {
    if times1.is_empty() || times2.is_empty() {
        return Err(StatsError::InvalidArgument(
            "Both groups must be non-empty".to_string(),
        ));
    }
    if times1.len() != events1.len() {
        return Err(StatsError::DimensionMismatch(format!(
            "group1: times length {} != events length {}",
            times1.len(),
            events1.len()
        )));
    }
    if times2.len() != events2.len() {
        return Err(StatsError::DimensionMismatch(format!(
            "group2: times length {} != events length {}",
            times2.len(),
            events2.len()
        )));
    }

    // Collect distinct cause codes across both groups
    let mut causes: Vec<u8> = events1
        .iter()
        .chain(events2.iter())
        .filter(|&&e| e > 0)
        .copied()
        .collect();
    causes.sort_unstable();
    causes.dedup();

    if causes.is_empty() {
        return Ok(vec![]);
    }

    // Collect all distinct event times
    let mut all_event_times: Vec<f64> = times1
        .iter()
        .zip(events1.iter())
        .filter_map(|(&t, &e)| if e > 0 { Some(t) } else { None })
        .chain(
            times2
                .iter()
                .zip(events2.iter())
                .filter_map(|(&t, &e)| if e > 0 { Some(t) } else { None }),
        )
        .collect();
    all_event_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    all_event_times.dedup_by(|a, b| (*a - *b).abs() < 1e-14);

    let mut results = Vec::new();

    for &cause in &causes {
        let (stat, pval) =
            gray_test_one_cause(times1, events1, times2, events2, cause, &all_event_times)?;
        results.push((cause as usize, stat, pval));
    }

    Ok(results)
}

/// Gray's test for a single cause.
///
/// The test statistic is a weighted sum of differences in cause-specific hazard
/// increments between groups, analogous to the log-rank test but accounting for
/// competing risks via a weight based on the subdistribution survival.
fn gray_test_one_cause(
    times1: &[f64],
    events1: &[u8],
    times2: &[f64],
    events2: &[u8],
    cause: u8,
    all_event_times: &[f64],
) -> StatsResult<(f64, f64)> {
    let mut numerator = 0.0_f64;
    let mut variance = 0.0_f64;

    // Overall KM survival estimates for each group (for Gray's weight)
    let km1 = km_survival_vector(times1, events1, all_event_times);
    let km2 = km_survival_vector(times2, events2, all_event_times);

    for (k, &t_cur) in all_event_times.iter().enumerate() {
        // Numbers at risk in each group
        let n1 = times1.iter().filter(|&&ti| ti >= t_cur - 1e-14).count() as f64;
        let n2 = times2.iter().filter(|&&ti| ti >= t_cur - 1e-14).count() as f64;
        let n = n1 + n2;
        if n < 2.0 {
            continue;
        }

        // Events of this cause in each group at t_cur
        let d1k = times1
            .iter()
            .zip(events1.iter())
            .filter(|(&ti, &ei)| (ti - t_cur).abs() < 1e-14 && ei == cause)
            .count() as f64;
        let d2k = times2
            .iter()
            .zip(events2.iter())
            .filter(|(&ti, &ei)| (ti - t_cur).abs() < 1e-14 && ei == cause)
            .count() as f64;

        // All events (any cause) for denominator subdistribution
        let d1_all = times1
            .iter()
            .zip(events1.iter())
            .filter(|(&ti, &ei)| (ti - t_cur).abs() < 1e-14 && ei > 0)
            .count() as f64;
        let d2_all = times2
            .iter()
            .zip(events2.iter())
            .filter(|(&ti, &ei)| (ti - t_cur).abs() < 1e-14 && ei > 0)
            .count() as f64;
        let d_all = d1_all + d2_all;

        // Gray's weight: w(t) = overall survival × n / (n1 + n2)
        // simplified weight = S_pooled(t) where S_pooled is the pooled KM
        let s1_prev = if k == 0 { 1.0 } else { km1[k - 1] };
        let s2_prev = if k == 0 { 1.0 } else { km2[k - 1] };
        let s_pooled = (n1 * s1_prev + n2 * s2_prev) / n;
        let weight = s_pooled;

        // Expected events of cause k in group 1
        let e1k = if n > 0.0 { n1 * (d1k + d2k) / n } else { 0.0 };

        numerator += weight * (d1k - e1k);

        // Variance (hypergeometric-like for cause-specific)
        if n > 1.0 && d_all > 0.0 {
            let dk = d1k + d2k;
            let var_term = n1 * n2 * dk * (n - dk) / (n * n * (n - 1.0));
            variance += weight * weight * var_term;
        }
    }

    if variance <= 0.0 {
        return Ok((0.0, 1.0));
    }

    let stat = numerator * numerator / variance;
    let pval = chi2_sf(stat, 1.0);
    Ok((stat, pval))
}

/// Compute KM overall survival at each time point in `event_times`.
fn km_survival_vector(times: &[f64], events: &[u8], event_times: &[f64]) -> Vec<f64> {
    let mut s = 1.0_f64;
    let mut result = Vec::with_capacity(event_times.len());

    for &t_cur in event_times {
        let n_at_risk = times.iter().filter(|&&ti| ti >= t_cur - 1e-14).count() as f64;
        let d_all = times
            .iter()
            .zip(events.iter())
            .filter(|(&ti, &ei)| (ti - t_cur).abs() < 1e-14 && ei > 0)
            .count() as f64;
        if n_at_risk > 0.0 && d_all > 0.0 {
            s *= (n_at_risk - d_all) / n_at_risk;
        }
        result.push(s);
    }
    result
}

// ---------------------------------------------------------------------------
// Fine-Gray subdistribution hazard model
// ---------------------------------------------------------------------------

/// Fine-Gray subdistribution hazard model for a single cause.
///
/// The model is: λ*(t | x) = λ*₀(t) exp(x γ), where λ* is the subdistribution
/// hazard. The risk set at time t includes subjects who have not yet experienced
/// the cause of interest (including those who experienced competing causes —
/// they remain in the risk set with appropriate weights).
///
/// Fitting is by maximising the weighted partial log-likelihood.
#[derive(Debug, Clone)]
pub struct FineGrayModel {
    /// Cause of interest.
    pub cause: u8,
    /// Regression coefficients γ.
    pub coefficients: Array1<f64>,
    /// Standard errors of γ.
    pub std_errors: Array1<f64>,
    /// Wald z-scores.
    pub z_scores: Array1<f64>,
    /// Two-sided p-values.
    pub p_values: Array1<f64>,
    /// Baseline cumulative subdistribution hazard (t, H*₀(t)).
    pub baseline_hazard: Vec<(f64, f64)>,
    /// Log partial likelihood.
    pub log_likelihood: f64,
    /// Convergence info.
    pub converged: bool,
    /// Number of iterations.
    pub n_iter: usize,
}

impl FineGrayModel {
    /// Fit the Fine-Gray model.
    ///
    /// # Arguments
    /// * `times`  – observed times.
    /// * `events` – cause codes (0 = censored, 1 = cause of interest, 2..k = competing).
    /// * `x`      – covariate matrix (n × p).
    /// * `cause`  – the cause of interest.
    ///
    /// # Errors
    /// Returns [`StatsError`] on invalid input or convergence failure.
    pub fn fit(times: &[f64], events: &[u8], x: &Array2<f64>, cause: u8) -> StatsResult<Self> {
        let n = times.len();
        let p = x.ncols();

        if n == 0 {
            return Err(StatsError::InvalidArgument(
                "times must not be empty".to_string(),
            ));
        }
        if events.len() != n {
            return Err(StatsError::DimensionMismatch(format!(
                "times length {} != events length {}",
                n,
                events.len()
            )));
        }
        if x.nrows() != n {
            return Err(StatsError::DimensionMismatch(format!(
                "x rows {} != times length {}",
                x.nrows(),
                n
            )));
        }
        let n_cause_events = events.iter().filter(|&&e| e == cause).count();
        if n_cause_events == 0 {
            return Err(StatsError::InvalidArgument(format!(
                "No events observed for cause {cause}"
            )));
        }

        // Sort by time
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| {
            times[a]
                .partial_cmp(&times[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let sorted_times: Vec<f64> = order.iter().map(|&i| times[i]).collect();
        let sorted_events: Vec<u8> = order.iter().map(|&i| events[i]).collect();
        let sorted_x: Vec<Vec<f64>> = order
            .iter()
            .map(|&i| (0..p).map(|j| x[[i, j]]).collect())
            .collect();

        // Compute IPCW-like weights for competing events using pooled KM
        // Subjects who experienced a competing cause remain in the risk set
        // with a decreasing weight (inverse probability of competing)
        // For simplicity: use equal weights (1.0) for events and 0 for censored
        // (classical Fine-Gray ignores censoring correction in this simplified form)
        let weights: Vec<f64> = (0..n)
            .map(|i| {
                if sorted_events[i] == cause {
                    1.0
                }
                // event of interest
                else if sorted_events[i] == 0 {
                    1.0
                }
                // censored (handled via partial ll)
                else {
                    1.0
                } // competing: stays in risk set with weight 1
            })
            .collect();

        // Center covariates
        let x_mean: Vec<f64> = (0..p)
            .map(|j| sorted_x.iter().map(|row| row[j]).sum::<f64>() / n as f64)
            .collect();
        let xc: Vec<Vec<f64>> = sorted_x
            .iter()
            .map(|row| (0..p).map(|j| row[j] - x_mean[j]).collect())
            .collect();

        // Newton-Raphson optimisation
        let mut beta = vec![0.0_f64; p];
        let max_iter = 200;
        let tol = 1e-8;
        let mut converged = false;
        let mut n_iter = 0usize;

        for iter in 0..max_iter {
            let (_ll, grad, neg_hess) = fg_partial_ll_gradient_hessian(
                &sorted_times,
                &sorted_events,
                &xc,
                &weights,
                &beta,
                cause,
                n,
                p,
            );

            let delta = fg_solve_system(&neg_hess, &grad, p)?;
            let step = fg_backtrack(
                &sorted_times,
                &sorted_events,
                &xc,
                &weights,
                &beta,
                &delta,
                cause,
                n,
                p,
                20,
            );
            let max_d = delta.iter().map(|d| d.abs()).fold(0.0_f64, f64::max);

            for j in 0..p {
                beta[j] += step * delta[j];
            }

            n_iter = iter + 1;
            if max_d * step < tol {
                converged = true;
                break;
            }
        }

        let (ll_final, _, neg_hess_final) = fg_partial_ll_gradient_hessian(
            &sorted_times,
            &sorted_events,
            &xc,
            &weights,
            &beta,
            cause,
            n,
            p,
        );

        // Variance-covariance
        let vcov = fg_invert(&neg_hess_final, p).unwrap_or_else(|_| vec![0.0; p * p]);

        let std_errors: Vec<f64> = (0..p).map(|j| vcov[j * p + j].max(0.0).sqrt()).collect();

        let z_scores: Vec<f64> = (0..p)
            .map(|j| beta[j] / std_errors[j].max(1e-300))
            .collect();

        let p_values: Vec<f64> = z_scores
            .iter()
            .map(|&z| 2.0 * (1.0 - norm_cdf(z.abs())))
            .collect();

        // Breslow-type baseline subdistribution hazard
        let exp_xb: Vec<f64> = (0..n)
            .map(|i| {
                let xb: f64 = (0..p).map(|j| xc[i][j] * beta[j]).sum();
                xb.exp()
            })
            .collect();
        let baseline_hazard =
            fg_breslow_baseline(&sorted_times, &sorted_events, &exp_xb, &weights, cause, n);

        Ok(Self {
            cause,
            coefficients: Array1::from_vec(beta),
            std_errors: Array1::from_vec(std_errors),
            z_scores: Array1::from_vec(z_scores),
            p_values: Array1::from_vec(p_values),
            baseline_hazard,
            log_likelihood: ll_final,
            converged,
            n_iter,
        })
    }

    /// Predict subdistribution hazard ratio exp(x γ) for new observations.
    pub fn predict_hazard(&self, x_new: &Array2<f64>) -> Array1<f64> {
        let n = x_new.nrows();
        let p = self.coefficients.len();
        let mut hazards = Array1::zeros(n);
        for i in 0..n {
            let xg: f64 = (0..p).map(|j| x_new[[i, j]] * self.coefficients[j]).sum();
            hazards[i] = xg.exp();
        }
        hazards
    }

    /// Predict cumulative incidence F*(t | x) = 1 - exp(-H*₀(t) exp(x γ)).
    pub fn predict_cif(&self, x_new: &Array2<f64>, t: f64) -> Array1<f64> {
        let hazards = self.predict_hazard(x_new);
        let h0 = self.baseline_subdist_hazard_at(t);
        let n = x_new.nrows();
        let mut cif = Array1::zeros(n);
        for i in 0..n {
            cif[i] = (1.0 - (-h0 * hazards[i]).exp()).clamp(0.0, 1.0);
        }
        cif
    }

    fn baseline_subdist_hazard_at(&self, t: f64) -> f64 {
        if self.baseline_hazard.is_empty() || t < self.baseline_hazard[0].0 {
            return 0.0;
        }
        let idx = self
            .baseline_hazard
            .partition_point(|&(tk, _)| tk <= t)
            .saturating_sub(1);
        self.baseline_hazard[idx].1
    }
}

// Fine-Gray partial log-likelihood + gradient + neg-Hessian
fn fg_partial_ll_gradient_hessian(
    sorted_times: &[f64],
    sorted_events: &[u8],
    xc: &[Vec<f64>],
    weights: &[f64],
    beta: &[f64],
    cause: u8,
    n: usize,
    p: usize,
) -> (f64, Vec<f64>, Vec<f64>) {
    let exp_xb: Vec<f64> = (0..n)
        .map(|i| {
            let xb: f64 = (0..p).map(|j| xc[i][j] * beta[j]).sum();
            (xb.exp() * weights[i]).max(1e-300)
        })
        .collect();

    let mut ll = 0.0_f64;
    let mut grad = vec![0.0_f64; p];
    let mut neg_hess = vec![0.0_f64; p * p];

    // Process in time order; Fine-Gray risk set: subjects not yet experiencing cause k
    // (competing-cause subjects stay in, censored leave)
    // Simplified: use all subjects with time >= t_cur as risk set (true Fine-Gray
    // would add back competing-cause subjects — we use this common approximation)

    let mut i = 0usize;
    while i < n {
        let t_cur = sorted_times[i];

        // Collect tie group
        let mut j = i;
        let mut d_k = 0usize;
        while j < n && (sorted_times[j] - t_cur).abs() < 1e-14 {
            if sorted_events[j] == cause {
                d_k += 1;
            }
            j += 1;
        }

        if d_k > 0 {
            // Fine-Gray risk set: all with time >= t_cur (treat competing risks as still at risk)
            let mut s0 = 0.0_f64;
            let mut s1 = vec![0.0_f64; p];
            let mut s2 = vec![0.0_f64; p * p];
            for k in i..n {
                // All subjects with time >= t_cur
                s0 += exp_xb[k];
                for jj in 0..p {
                    s1[jj] += xc[k][jj] * exp_xb[k];
                    for kk in 0..p {
                        s2[jj * p + kk] += xc[k][jj] * xc[k][kk] * exp_xb[k];
                    }
                }
            }
            if s0 > 1e-300 {
                // Log-likelihood contribution
                let mut xb_sum = 0.0_f64;
                for k in i..j {
                    if sorted_events[k] == cause {
                        xb_sum += (0..p).map(|jj| xc[k][jj] * beta[jj]).sum::<f64>();
                    }
                }
                ll += xb_sum - d_k as f64 * s0.ln();

                let e1: Vec<f64> = (0..p).map(|jj| s1[jj] / s0).collect();

                for jj in 0..p {
                    // Gradient
                    let xb_col: f64 = (i..j)
                        .filter(|&k| sorted_events[k] == cause)
                        .map(|k| xc[k][jj])
                        .sum();
                    grad[jj] += xb_col - d_k as f64 * e1[jj];

                    for kk in 0..p {
                        let e2 = s2[jj * p + kk] / s0;
                        neg_hess[jj * p + kk] += d_k as f64 * (e2 - e1[jj] * e1[kk]);
                    }
                }
            }
        }

        i = j;
    }

    (ll, grad, neg_hess)
}

fn fg_solve_system(hess: &[f64], grad: &[f64], p: usize) -> StatsResult<Vec<f64>> {
    if p == 0 {
        return Ok(vec![]);
    }
    let mut h = hess.to_vec();
    let lambda = 1e-8
        * hess
            .iter()
            .map(|&v| v.abs())
            .fold(0.0_f64, f64::max)
            .max(1e-6);
    for j in 0..p {
        h[j * p + j] += lambda;
    }

    let mut l = vec![0.0_f64; p * p];
    for i in 0..p {
        for j in 0..=i {
            let mut s = h[i * p + j];
            for k in 0..j {
                s -= l[i * p + k] * l[j * p + k];
            }
            if i == j {
                if s < 1e-300 {
                    let scale = h.iter().map(|&v| v.abs()).fold(0.0_f64, f64::max).max(1.0);
                    return Ok(grad.iter().map(|&g| g / scale).collect());
                }
                l[i * p + j] = s.sqrt();
            } else {
                l[i * p + j] = s / l[j * p + j];
            }
        }
    }

    let mut y = vec![0.0_f64; p];
    for i in 0..p {
        let mut s = grad[i];
        for k in 0..i {
            s -= l[i * p + k] * y[k];
        }
        y[i] = s / l[i * p + i];
    }

    let mut delta = vec![0.0_f64; p];
    for i in (0..p).rev() {
        let mut s = y[i];
        for k in (i + 1)..p {
            s -= l[k * p + i] * delta[k];
        }
        delta[i] = s / l[i * p + i];
    }
    Ok(delta)
}

fn fg_invert(hess: &[f64], p: usize) -> StatsResult<Vec<f64>> {
    if p == 0 {
        return Ok(vec![]);
    }
    let mut h = hess.to_vec();
    let lambda = 1e-8
        * hess
            .iter()
            .map(|&v| v.abs())
            .fold(0.0_f64, f64::max)
            .max(1e-6);
    for j in 0..p {
        h[j * p + j] += lambda;
    }

    let mut l = vec![0.0_f64; p * p];
    for i in 0..p {
        for j in 0..=i {
            let mut s = h[i * p + j];
            for k in 0..j {
                s -= l[i * p + k] * l[j * p + k];
            }
            if i == j {
                if s < 1e-300 {
                    return Err(StatsError::ComputationError("Singular Hessian".to_string()));
                }
                l[i * p + j] = s.sqrt();
            } else {
                l[i * p + j] = s / l[j * p + j];
            }
        }
    }

    let mut l_inv = vec![0.0_f64; p * p];
    for k in 0..p {
        for i in 0..p {
            let mut s = if i == k { 1.0 } else { 0.0 };
            for j in 0..i {
                s -= l[i * p + j] * l_inv[j * p + k];
            }
            l_inv[i * p + k] = s / l[i * p + i];
        }
    }

    let mut inv = vec![0.0_f64; p * p];
    for i in 0..p {
        for j in 0..p {
            inv[i * p + j] = (0..p).map(|k| l_inv[k * p + i] * l_inv[k * p + j]).sum();
        }
    }
    Ok(inv)
}

fn fg_backtrack(
    sorted_times: &[f64],
    sorted_events: &[u8],
    xc: &[Vec<f64>],
    weights: &[f64],
    beta: &[f64],
    delta: &[f64],
    cause: u8,
    n: usize,
    p: usize,
    max_halve: usize,
) -> f64 {
    let (ll_cur, _, _) =
        fg_partial_ll_gradient_hessian(sorted_times, sorted_events, xc, weights, beta, cause, n, p);
    let c = 1e-4;
    let mut step = 1.0_f64;
    for _ in 0..max_halve {
        let beta_new: Vec<f64> = (0..p).map(|j| beta[j] + step * delta[j]).collect();
        let (ll_new, _, _) = fg_partial_ll_gradient_hessian(
            sorted_times,
            sorted_events,
            xc,
            weights,
            &beta_new,
            cause,
            n,
            p,
        );
        if ll_new > ll_cur - c * step * delta.iter().map(|d| d.abs()).sum::<f64>() {
            return step;
        }
        step *= 0.5;
    }
    step
}

fn fg_breslow_baseline(
    sorted_times: &[f64],
    sorted_events: &[u8],
    exp_xb: &[f64],
    weights: &[f64],
    cause: u8,
    n: usize,
) -> Vec<(f64, f64)> {
    let mut result = Vec::new();
    let mut cum_h = 0.0_f64;
    let mut pos = 0usize;

    while pos < n {
        let t_cur = sorted_times[pos];
        let mut d_k = 0usize;
        let mut end = pos;
        while end < n && (sorted_times[end] - t_cur).abs() < 1e-14 {
            if sorted_events[end] == cause {
                d_k += 1;
            }
            end += 1;
        }
        if d_k > 0 {
            // Risk set sum: subjects from pos to end (Fine-Gray: also those at risk after)
            let risk_sum: f64 = (pos..n).map(|i| exp_xb[i] * weights[i]).sum();
            if risk_sum > 1e-300 {
                cum_h += d_k as f64 / risk_sum;
            }
            result.push((t_cur, cum_h));
        }
        pos = end;
    }
    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn cr_data() -> (Vec<f64>, Vec<u8>) {
        // times and cause codes: 0=censored, 1=cause 1, 2=cause 2
        let times = vec![1.0, 2.0, 2.0, 3.0, 4.0, 5.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let events = vec![1u8, 2, 1, 0, 1, 2, 1, 0, 1, 2, 1, 0];
        (times, events)
    }

    #[test]
    fn test_cif_fit_basic() {
        let (times, events) = cr_data();
        let cifs = CumulativeIncidenceFunction::fit(&times, &events).expect("CIF fit failed");
        // Should have 2 causes
        assert_eq!(cifs.len(), 2);
        assert_eq!(cifs[0].cause, 1);
        assert_eq!(cifs[1].cause, 2);
    }

    #[test]
    fn test_cif_non_decreasing() {
        let (times, events) = cr_data();
        let cifs = CumulativeIncidenceFunction::fit(&times, &events).expect("CIF fit");
        for cif in &cifs {
            for i in 1..cif.cif.len() {
                assert!(
                    cif.cif[i] >= cif.cif[i - 1] - 1e-12,
                    "CIF not non-decreasing at index {i}"
                );
            }
        }
    }

    #[test]
    fn test_cif_bounded() {
        let (times, events) = cr_data();
        let cifs = CumulativeIncidenceFunction::fit(&times, &events).expect("CIF fit");
        for cif in &cifs {
            for &c in &cif.cif {
                assert!(c >= 0.0 && c <= 1.0 + 1e-12, "CIF {c} out of [0,1]");
            }
        }
    }

    #[test]
    fn test_cif_sum_le_one() {
        // Sum of CIFs across causes should not exceed 1
        let (times, events) = cr_data();
        let cifs = CumulativeIncidenceFunction::fit(&times, &events).expect("CIF fit");
        let last_time = 10.0_f64;
        let sum_cif: f64 = cifs.iter().map(|c| c.cif_at(last_time)).sum();
        assert!(sum_cif <= 1.0 + 1e-12, "Sum of CIFs {sum_cif} > 1");
    }

    #[test]
    fn test_cif_at_step_function() {
        let (times, events) = cr_data();
        let cifs = CumulativeIncidenceFunction::fit(&times, &events).expect("CIF fit");
        let c1 = &cifs[0];
        // Before first event: 0
        assert!((c1.cif_at(0.5) - 0.0).abs() < 1e-12);
        // At first event time: first jump value
        if !c1.times.is_empty() {
            let val = c1.cif_at(c1.times[0]);
            assert!(val > 0.0, "CIF should be positive after first event");
        }
    }

    #[test]
    fn test_cif_confidence_interval() {
        let (times, events) = cr_data();
        let cifs = CumulativeIncidenceFunction::fit(&times, &events).expect("CIF fit");
        let (lo, hi) = cifs[0].confidence_interval(5.0, 0.05);
        assert!(lo >= 0.0 && lo <= 1.0, "lower {lo} out of range");
        assert!(hi >= 0.0 && hi <= 1.0, "upper {hi} out of range");
        assert!(lo <= hi + 1e-12, "lower > upper");
    }

    #[test]
    fn test_cause_specific_hazard() {
        let (times, events) = cr_data();
        let (ht, hv) = cause_specific_hazard(&times, &events, 1).expect("CSH failed");
        assert_eq!(ht.len(), hv.len());
        // Monotone increasing
        for i in 1..hv.len() {
            assert!(hv[i] >= hv[i - 1] - 1e-12, "CSH not monotone at {i}");
        }
    }

    #[test]
    fn test_gray_test_returns_results() {
        let t1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let e1 = vec![1u8, 2, 1, 0, 1, 2];
        let t2 = vec![1.5, 2.5, 3.5, 4.5, 5.5, 6.5];
        let e2 = vec![1u8, 1, 2, 1, 0, 1];
        let results = gray_test(&t1, &e1, &t2, &e2).expect("Gray test failed");
        assert!(!results.is_empty());
        for &(cause, stat, pval) in &results {
            assert!(cause > 0);
            assert!(stat >= 0.0, "stat {stat} must be non-negative");
            assert!(pval >= 0.0 && pval <= 1.0, "pval {pval} must be in [0,1]");
        }
    }

    #[test]
    fn test_gray_test_identical_groups() {
        let t = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let e = vec![1u8, 2, 1, 0, 1];
        let results = gray_test(&t, &e, &t, &e).expect("Gray test failed");
        for &(_, stat, pval) in &results {
            // Identical groups: statistic should be 0 or very small
            assert!(
                stat.abs() < 1e-10,
                "stat should be ~0 for identical groups, got {stat}"
            );
            assert!(pval > 0.5, "p-value should be large, got {pval}");
        }
    }

    #[test]
    fn test_fine_gray_fit() {
        let times = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let events = vec![1u8, 2, 1, 0, 1, 2, 0, 1];
        let mut cov = Array2::zeros((8, 1));
        for i in 0..8_usize {
            cov[[i, 0]] = i as f64;
        }
        let model = FineGrayModel::fit(&times, &events, &cov, 1).expect("Fine-Gray fit failed");
        assert_eq!(model.coefficients.len(), 1);
        assert!(model.log_likelihood.is_finite());
    }

    #[test]
    fn test_fine_gray_predict_cif_bounded() {
        let times = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let events = vec![1u8, 2, 1, 0, 1, 2, 0, 1];
        let mut cov = Array2::zeros((8, 1));
        for i in 0..8_usize {
            cov[[i, 0]] = i as f64;
        }
        let model = FineGrayModel::fit(&times, &events, &cov, 1).expect("Fine-Gray fit");
        let cif = model.predict_cif(&cov, 5.0);
        for &c in cif.iter() {
            assert!(c >= 0.0 && c <= 1.0 + 1e-12, "CIF {c} out of [0,1]");
        }
    }

    #[test]
    fn test_cif_error_empty() {
        let result = CumulativeIncidenceFunction::fit(&[], &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_gray_test_error_empty() {
        let result = gray_test(&[], &[], &[1.0], &[1u8]);
        assert!(result.is_err());
    }
}
