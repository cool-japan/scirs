//! Information-theoretic measures: entropy, divergence, and model selection criteria.
//!
//! This module provides a comprehensive set of information-theoretic functions for
//! measuring uncertainty in probability distributions, comparing distributions,
//! and selecting statistical models.
//!
//! ## Entropy Measures
//! - [`entropy`]: Shannon entropy of a probability vector
//! - [`joint_entropy`]: Joint entropy H(X, Y)
//! - [`conditional_entropy`]: Conditional entropy H(X|Y)
//! - [`mutual_information`]: Mutual information I(X; Y)
//! - [`normalized_mutual_information`]: NMI scaled to [0, 1]
//!
//! ## Divergence / Distance Measures
//! - [`kl_divergence`]: Kullback-Leibler divergence KL(P ∥ Q)
//! - [`js_divergence`]: Jensen-Shannon divergence (symmetric, bounded)
//! - [`total_variation`]: Total variation distance
//! - [`hellinger_distance`]: Hellinger distance
//!
//! ## Information Criteria for Model Selection
//! - [`aic`]: Akaike Information Criterion
//! - [`bic`]: Bayesian Information Criterion (Schwarz)
//! - [`aicc`]: Corrected AIC for small samples
//! - [`hqic`]: Hannan-Quinn Information Criterion
//!
//! # References
//! - Cover, T.M. & Thomas, J.A. (2006). *Elements of Information Theory* (2nd ed.). Wiley.
//! - Burnham, K.P. & Anderson, D.R. (2002). *Model Selection and Multimodel Inference*. Springer.

use crate::error::{StatsError, StatsResult};

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Validate that `pk` is a valid probability vector: all non-negative, sums to 1.
/// Returns `Ok(())` if valid, or an error describing the problem.
fn validate_probability_vector(pk: &[f64], name: &str) -> StatsResult<()> {
    if pk.is_empty() {
        return Err(StatsError::InsufficientData(format!(
            "{name}: probability vector must not be empty"
        )));
    }
    for &p in pk {
        if p < 0.0 || p.is_nan() {
            return Err(StatsError::InvalidArgument(format!(
                "{name}: all probabilities must be non-negative, found {p}"
            )));
        }
    }
    let s: f64 = pk.iter().sum();
    if (s - 1.0).abs() > 1e-6 {
        return Err(StatsError::InvalidArgument(format!(
            "{name}: probabilities must sum to 1, got {s}"
        )));
    }
    Ok(())
}

/// Normalise a non-negative slice to sum to 1.  Returns the normalised vector.
fn normalise(v: &[f64]) -> StatsResult<Vec<f64>> {
    let s: f64 = v.iter().sum();
    if s < f64::EPSILON {
        return Err(StatsError::InvalidArgument(
            "cannot normalise: sum is zero or negative".into(),
        ));
    }
    Ok(v.iter().map(|&x| x / s).collect())
}

/// Convert base: log(x) / log(base).  `None` → natural log.
#[inline]
fn log_base(x: f64, base: Option<f64>) -> f64 {
    match base {
        None => x.ln(),
        Some(b) => x.log2() / b.log2(), // = x.ln() / b.ln() (same thing)
    }
}

// ---------------------------------------------------------------------------
// 2-D histogram helpers for joint/conditional/mutual information
// ---------------------------------------------------------------------------

/// Compute a 2-D joint histogram of `(x, y)` with `bins` uniform bins
/// per axis.  Returns `(joint_hist, x_marginal, y_marginal)` all as
/// normalised probability tables.
fn joint_histogram(
    x: &[f64],
    y: &[f64],
    bins: usize,
) -> StatsResult<(Vec<Vec<f64>>, Vec<f64>, Vec<f64>)> {
    if x.is_empty() || y.is_empty() {
        return Err(StatsError::InsufficientData(
            "joint_histogram: data arrays must not be empty".into(),
        ));
    }
    if x.len() != y.len() {
        return Err(StatsError::DimensionMismatch(format!(
            "joint_histogram: x and y must have the same length, got {} vs {}",
            x.len(),
            y.len()
        )));
    }
    if bins == 0 {
        return Err(StatsError::InvalidArgument(
            "bins must be at least 1".into(),
        ));
    }

    let n = x.len();

    let x_min = x.iter().cloned().fold(f64::INFINITY, f64::min);
    let x_max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let y_min = y.iter().cloned().fold(f64::INFINITY, f64::min);
    let y_max = y.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    // If range is zero (all values equal), treat as single bin
    let x_range = if (x_max - x_min).abs() < f64::EPSILON {
        1.0
    } else {
        x_max - x_min
    };
    let y_range = if (y_max - y_min).abs() < f64::EPSILON {
        1.0
    } else {
        y_max - y_min
    };

    let mut counts = vec![vec![0usize; bins]; bins];

    for (&xi, &yi) in x.iter().zip(y.iter()) {
        let bx = ((((xi - x_min) / x_range) * bins as f64).floor() as usize).min(bins - 1);
        let by = ((((yi - y_min) / y_range) * bins as f64).floor() as usize).min(bins - 1);
        counts[bx][by] += 1;
    }

    let nf = n as f64;
    let joint: Vec<Vec<f64>> = counts
        .iter()
        .map(|row| row.iter().map(|&c| c as f64 / nf).collect())
        .collect();

    let px: Vec<f64> = joint.iter().map(|row| row.iter().sum::<f64>()).collect();
    let py: Vec<f64> = (0..bins)
        .map(|j| joint.iter().map(|row| row[j]).sum::<f64>())
        .collect();

    Ok((joint, px, py))
}

// ---------------------------------------------------------------------------
// Entropy measures
// ---------------------------------------------------------------------------

/// Compute the Shannon entropy of a discrete probability distribution.
///
/// `H(P) = −∑ pᵢ · log(pᵢ)` (convention: 0 · log 0 = 0).
///
/// # Arguments
///
/// * `pk` – Probability mass vector. Must be non-negative and sum to 1.
/// * `base` – Logarithm base (`None` → natural units "nats"; `Some(2.0)` →
///   bits; `Some(10.0)` → dits).
///
/// # Errors
///
/// Returns an error if `pk` is empty, contains negative values, or does not
/// sum to 1 (within 1e-6).
///
/// # Examples
///
/// ```
/// use scirs2_stats::information::entropy;
///
/// // Uniform distribution over 4 outcomes → 2 bits
/// let uniform = vec![0.25_f64; 4];
/// let h = entropy(&uniform, Some(2.0)).expect("ok");
/// assert!((h - 2.0).abs() < 1e-10);
///
/// // Deterministic outcome → 0 entropy
/// let certain = vec![1.0_f64, 0.0, 0.0];
/// let h0 = entropy(&certain, None).expect("ok");
/// assert_eq!(h0, 0.0);
/// ```
pub fn entropy(pk: &[f64], base: Option<f64>) -> StatsResult<f64> {
    validate_probability_vector(pk, "entropy")?;
    if let Some(b) = base {
        if b <= 0.0 || (b - 1.0).abs() < f64::EPSILON {
            return Err(StatsError::InvalidArgument(
                "logarithm base must be positive and ≠ 1".into(),
            ));
        }
    }

    let h = pk
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * log_base(p, base))
        .sum::<f64>();
    Ok(h)
}

/// Compute the joint entropy H(X, Y) from paired continuous observations
/// using a 2-D histogram estimator.
///
/// # Arguments
///
/// * `x` – Observations for variable X.
/// * `y` – Observations for variable Y.
/// * `bins` – Number of histogram bins per axis.
///
/// # Errors
///
/// Returns an error if arrays are empty, have different lengths, or `bins == 0`.
///
/// # Examples
///
/// ```
/// use scirs2_stats::information::joint_entropy;
///
/// let x = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
/// let y = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
/// let h = joint_entropy(&x, &y, 5).expect("ok");
/// assert!(h >= 0.0);
/// ```
pub fn joint_entropy(x: &[f64], y: &[f64], bins: usize) -> StatsResult<f64> {
    let (joint, _, _) = joint_histogram(x, y, bins)?;
    let h = joint
        .iter()
        .flat_map(|row| row.iter())
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.ln())
        .sum::<f64>();
    Ok(h)
}

/// Compute the conditional entropy H(X | Y) from paired continuous observations.
///
/// `H(X|Y) = H(X,Y) − H(Y)`
///
/// # Errors
///
/// Returns an error if arrays are empty, have different lengths, or `bins == 0`.
///
/// # Examples
///
/// ```
/// use scirs2_stats::information::conditional_entropy;
///
/// let x = vec![1.0_f64, 2.0, 3.0, 4.0];
/// let y = x.clone(); // Y = X → H(X|Y) ≈ 0
/// let h = conditional_entropy(&x, &y, 4).expect("ok");
/// assert!(h >= 0.0);
/// ```
pub fn conditional_entropy(x: &[f64], y: &[f64], bins: usize) -> StatsResult<f64> {
    let (joint, _, py) = joint_histogram(x, y, bins)?;

    // H(X|Y) = sum_{y} P(y) H(X|Y=y)
    //        = sum_{x,y} P(x,y) log[P(y)/P(x,y)]
    let mut h = 0.0_f64;
    for bx in 0..bins {
        for by in 0..bins {
            let pxy = joint[bx][by];
            let py_val = py[by];
            if pxy > 0.0 && py_val > 0.0 {
                h += pxy * (py_val / pxy).ln();
            }
        }
    }
    Ok(h.max(0.0))
}

/// Compute the mutual information I(X; Y) from paired continuous observations.
///
/// `I(X;Y) = H(X) + H(Y) − H(X,Y)`
///
/// Uses a 2-D histogram estimator.  The result is guaranteed non-negative.
///
/// # Errors
///
/// Returns an error if arrays are empty, have different lengths, or `bins == 0`.
///
/// # Examples
///
/// ```
/// use scirs2_stats::information::mutual_information;
///
/// // Completely correlated: I(X; X) = H(X)
/// let x = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
/// let mi = mutual_information(&x, &x, 5).expect("ok");
/// assert!(mi >= 0.0);
///
/// // Uniform noise independent of X
/// let noise: Vec<f64> = (0..5).map(|i| (i * 7 + 3) as f64 % 5.0).collect();
/// let mi_ind = mutual_information(&x, &noise, 5).expect("ok");
/// assert!(mi_ind >= 0.0);
/// ```
pub fn mutual_information(x: &[f64], y: &[f64], bins: usize) -> StatsResult<f64> {
    let (joint, px, py) = joint_histogram(x, y, bins)?;

    let mut mi = 0.0_f64;
    for bx in 0..bins {
        for by in 0..bins {
            let pxy = joint[bx][by];
            let px_val = px[bx];
            let py_val = py[by];
            if pxy > 0.0 && px_val > 0.0 && py_val > 0.0 {
                mi += pxy * (pxy / (px_val * py_val)).ln();
            }
        }
    }
    Ok(mi.max(0.0))
}

/// Compute the normalized mutual information (NMI) scaled to [0, 1].
///
/// `NMI(X, Y) = 2 · I(X;Y) / (H(X) + H(Y))`
///
/// Returns 1 when X and Y are perfectly correlated, and 0 when they are
/// independent (or one is constant).
///
/// # Errors
///
/// Returns an error if arrays are empty, have different lengths, or `bins == 0`.
///
/// # Examples
///
/// ```
/// use scirs2_stats::information::normalized_mutual_information;
///
/// let x = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
/// let nmi_self = normalized_mutual_information(&x, &x, 5).expect("ok");
/// assert!(nmi_self >= 0.0 && nmi_self <= 1.0 + 1e-10);
/// ```
pub fn normalized_mutual_information(x: &[f64], y: &[f64], bins: usize) -> StatsResult<f64> {
    let (joint, px, py) = joint_histogram(x, y, bins)?;

    // H(X)
    let hx: f64 = px.iter().filter(|&&p| p > 0.0).map(|&p| -p * p.ln()).sum();
    // H(Y)
    let hy: f64 = py.iter().filter(|&&p| p > 0.0).map(|&p| -p * p.ln()).sum();
    // I(X;Y)
    let mut mi = 0.0_f64;
    for bx in 0..bins {
        for by in 0..bins {
            let pxy = joint[bx][by];
            let px_val = px[bx];
            let py_val = py[by];
            if pxy > 0.0 && px_val > 0.0 && py_val > 0.0 {
                mi += pxy * (pxy / (px_val * py_val)).ln();
            }
        }
    }
    mi = mi.max(0.0);

    let denom = hx + hy;
    if denom < f64::EPSILON {
        return Ok(0.0); // both distributions are deterministic
    }
    Ok((2.0 * mi / denom).min(1.0))
}

// ---------------------------------------------------------------------------
// Divergence / distance measures
// ---------------------------------------------------------------------------

/// Compute the Kullback-Leibler divergence KL(P ∥ Q).
///
/// `KL(P∥Q) = ∑ₓ P(x) · ln[P(x)/Q(x)]`
///
/// Convention: terms where P(x) = 0 contribute 0. Terms where P(x) > 0 but
/// Q(x) = 0 yield +∞, and the function returns an error in that case.
///
/// # Arguments
///
/// * `p` – Reference distribution P (must be a valid probability vector).
/// * `q` – Approximating distribution Q (same length as P; must be a valid
///   probability vector).
///
/// # Errors
///
/// Returns [`StatsError::DimensionMismatch`] if lengths differ, or
/// [`StatsError::DomainError`] if Q(x) = 0 for some x where P(x) > 0.
///
/// # Examples
///
/// ```
/// use scirs2_stats::information::kl_divergence;
///
/// let p = vec![0.5_f64, 0.5];
/// let q = vec![0.5_f64, 0.5];
/// let kl = kl_divergence(&p, &q).expect("ok");
/// assert!(kl.abs() < 1e-10); // identical distributions → KL = 0
///
/// let p2 = vec![0.9_f64, 0.1];
/// let q2 = vec![0.5_f64, 0.5];
/// let kl2 = kl_divergence(&p2, &q2).expect("ok");
/// assert!(kl2 > 0.0);
/// ```
pub fn kl_divergence(p: &[f64], q: &[f64]) -> StatsResult<f64> {
    validate_probability_vector(p, "kl_divergence(p)")?;
    validate_probability_vector(q, "kl_divergence(q)")?;
    if p.len() != q.len() {
        return Err(StatsError::DimensionMismatch(format!(
            "kl_divergence: p and q must have the same length, got {} vs {}",
            p.len(),
            q.len()
        )));
    }

    let mut kl = 0.0_f64;
    for (&pi, &qi) in p.iter().zip(q.iter()) {
        if pi > 0.0 {
            if qi <= 0.0 {
                return Err(StatsError::DomainError(
                    "kl_divergence: Q(x) = 0 where P(x) > 0 → KL divergence is infinite".into(),
                ));
            }
            kl += pi * (pi / qi).ln();
        }
    }
    Ok(kl)
}

/// Compute the Jensen-Shannon divergence JS(P, Q).
///
/// `JS(P, Q) = ½ KL(P ∥ M) + ½ KL(Q ∥ M)` where M = (P + Q) / 2.
///
/// JS divergence is symmetric, always finite, and bounded by ln(2) (in nats).
///
/// # Errors
///
/// Returns an error if `p` and `q` are invalid or have different lengths.
///
/// # Examples
///
/// ```
/// use scirs2_stats::information::js_divergence;
///
/// let p = vec![1.0_f64, 0.0];
/// let q = vec![0.0_f64, 1.0];
/// let js = js_divergence(&p, &q).expect("ok");
/// // Max JS divergence is ln(2) ≈ 0.6931
/// assert!((js - 2.0_f64.ln()).abs() < 1e-10, "js={js}");
/// ```
pub fn js_divergence(p: &[f64], q: &[f64]) -> StatsResult<f64> {
    validate_probability_vector(p, "js_divergence(p)")?;
    validate_probability_vector(q, "js_divergence(q)")?;
    if p.len() != q.len() {
        return Err(StatsError::DimensionMismatch(format!(
            "js_divergence: p and q must have the same length, got {} vs {}",
            p.len(),
            q.len()
        )));
    }

    let m: Vec<f64> = p
        .iter()
        .zip(q.iter())
        .map(|(&pi, &qi)| (pi + qi) * 0.5)
        .collect();

    // KL(P ∥ M) — M is always > 0 where P or Q > 0
    let mut kl_pm = 0.0_f64;
    for (&pi, &mi) in p.iter().zip(m.iter()) {
        if pi > 0.0 {
            kl_pm += pi * (pi / mi).ln();
        }
    }

    // KL(Q ∥ M)
    let mut kl_qm = 0.0_f64;
    for (&qi, &mi) in q.iter().zip(m.iter()) {
        if qi > 0.0 {
            kl_qm += qi * (qi / mi).ln();
        }
    }

    Ok(0.5 * kl_pm + 0.5 * kl_qm)
}

/// Compute the total variation distance between two distributions.
///
/// `TV(P, Q) = ½ ∑|P(x) − Q(x)|`
///
/// The total variation distance is bounded in [0, 1].
///
/// # Errors
///
/// Returns an error if `p` or `q` are invalid or have different lengths.
///
/// # Examples
///
/// ```
/// use scirs2_stats::information::total_variation;
///
/// let p = vec![0.5_f64, 0.5];
/// let q = vec![1.0_f64, 0.0];
/// let tv = total_variation(&p, &q).expect("ok");
/// assert!((tv - 0.5).abs() < 1e-10);
/// ```
pub fn total_variation(p: &[f64], q: &[f64]) -> StatsResult<f64> {
    validate_probability_vector(p, "total_variation(p)")?;
    validate_probability_vector(q, "total_variation(q)")?;
    if p.len() != q.len() {
        return Err(StatsError::DimensionMismatch(format!(
            "total_variation: p and q must have the same length, got {} vs {}",
            p.len(),
            q.len()
        )));
    }

    let tv: f64 = p
        .iter()
        .zip(q.iter())
        .map(|(&pi, &qi)| (pi - qi).abs())
        .sum::<f64>()
        * 0.5;
    Ok(tv)
}

/// Compute the Hellinger distance between two distributions.
///
/// `H(P, Q) = (1/√2) · √(∑(√P(x) − √Q(x))²)`
///
/// Hellinger distance is bounded in [0, 1] and satisfies
/// `H²(P,Q) = 1 − BC(P,Q)` where BC is the Bhattacharyya coefficient.
///
/// # Errors
///
/// Returns an error if `p` or `q` are invalid or have different lengths.
///
/// # Examples
///
/// ```
/// use scirs2_stats::information::hellinger_distance;
///
/// let p = vec![0.5_f64, 0.5];
/// let q = vec![0.5_f64, 0.5]; // same distribution
/// let h = hellinger_distance(&p, &q).expect("ok");
/// assert!(h.abs() < 1e-10);
/// ```
pub fn hellinger_distance(p: &[f64], q: &[f64]) -> StatsResult<f64> {
    validate_probability_vector(p, "hellinger_distance(p)")?;
    validate_probability_vector(q, "hellinger_distance(q)")?;
    if p.len() != q.len() {
        return Err(StatsError::DimensionMismatch(format!(
            "hellinger_distance: p and q must have the same length, got {} vs {}",
            p.len(),
            q.len()
        )));
    }

    let sum_sq: f64 = p
        .iter()
        .zip(q.iter())
        .map(|(&pi, &qi)| (pi.sqrt() - qi.sqrt()).powi(2))
        .sum();
    Ok((sum_sq * 0.5).sqrt())
}

// ---------------------------------------------------------------------------
// Information criteria for model selection
// ---------------------------------------------------------------------------

/// Compute the Akaike Information Criterion (AIC).
///
/// `AIC = 2k − 2ℓ`
///
/// where `k` is the number of model parameters and `ℓ` is the maximised
/// log-likelihood.  Lower AIC indicates a better-fitting model.
///
/// # Arguments
///
/// * `log_likelihood` – Maximised log-likelihood ℓ.
/// * `k` – Number of free parameters (including intercept if applicable).
///
/// # Examples
///
/// ```
/// use scirs2_stats::information::aic;
///
/// let ll = -100.0_f64;
/// let k = 5;
/// let a = aic(ll, k);
/// assert_eq!(a, 2.0 * 5.0 - 2.0 * (-100.0));
/// ```
pub fn aic(log_likelihood: f64, k: usize) -> f64 {
    2.0 * k as f64 - 2.0 * log_likelihood
}

/// Compute the Bayesian Information Criterion (BIC) / Schwarz Criterion.
///
/// `BIC = k · ln(n) − 2ℓ`
///
/// BIC penalises model complexity more heavily than AIC for `n > 7`, making
/// it preferred for model selection when the true model is among the
/// candidates.
///
/// # Arguments
///
/// * `log_likelihood` – Maximised log-likelihood ℓ.
/// * `k` – Number of free parameters.
/// * `n` – Number of observations.
///
/// # Examples
///
/// ```
/// use scirs2_stats::information::bic;
///
/// let ll = -100.0_f64;
/// let b = bic(ll, 5, 100);
/// assert!((b - (5.0 * 100_f64.ln() - 2.0 * (-100.0))).abs() < 1e-10);
/// ```
pub fn bic(log_likelihood: f64, k: usize, n: usize) -> f64 {
    k as f64 * (n as f64).ln() - 2.0 * log_likelihood
}

/// Compute the corrected Akaike Information Criterion (AICc).
///
/// `AICc = AIC + 2k(k+1) / (n−k−1)`
///
/// AICc adds a second-order correction term that is important when `n/k` is
/// small (roughly `n/k < 40`).  As `n → ∞`, AICc → AIC.
///
/// # Arguments
///
/// * `log_likelihood` – Maximised log-likelihood ℓ.
/// * `k` – Number of free parameters.
/// * `n` – Number of observations (must satisfy `n > k + 1`).
///
/// # Examples
///
/// ```
/// use scirs2_stats::information::aicc;
///
/// let a = aicc(-50.0, 3, 20);
/// // AICc ≥ AIC
/// use scirs2_stats::information::aic;
/// assert!(a >= aic(-50.0, 3) - 1e-10);
/// ```
pub fn aicc(log_likelihood: f64, k: usize, n: usize) -> f64 {
    let base = aic(log_likelihood, k);
    let denom = n as f64 - k as f64 - 1.0;
    if denom <= 0.0 {
        // Undefined or infinite correction — return a large penalty
        return f64::INFINITY;
    }
    base + 2.0 * k as f64 * (k as f64 + 1.0) / denom
}

/// Compute the Hannan-Quinn Information Criterion (HQIC).
///
/// `HQIC = 2k · ln(ln(n)) − 2ℓ`
///
/// HQIC is a strongly consistent model selection criterion intermediate
/// between AIC and BIC.  It is particularly popular for time series
/// model order selection.
///
/// # Arguments
///
/// * `log_likelihood` – Maximised log-likelihood ℓ.
/// * `k` – Number of free parameters.
/// * `n` – Number of observations (must satisfy `n ≥ 3` for ln(ln(n)) > 0`).
///
/// # Examples
///
/// ```
/// use scirs2_stats::information::hqic;
///
/// let h = hqic(-100.0, 5, 100);
/// assert!(h > 0.0); // positive penalty for 100 observations
/// ```
pub fn hqic(log_likelihood: f64, k: usize, n: usize) -> f64 {
    2.0 * k as f64 * (n as f64).ln().ln() - 2.0 * log_likelihood
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- entropy ----

    #[test]
    fn test_entropy_uniform_bits() {
        let uniform = vec![0.25_f64; 4];
        let h = entropy(&uniform, Some(2.0)).expect("ok");
        assert!((h - 2.0).abs() < 1e-10, "h={h}");
    }

    #[test]
    fn test_entropy_deterministic() {
        let certain = vec![1.0_f64, 0.0, 0.0];
        let h = entropy(&certain, None).expect("ok");
        assert_eq!(h, 0.0);
    }

    #[test]
    fn test_entropy_nats_binary() {
        // Bernoulli(0.5) → ln(2) nats
        let p = vec![0.5_f64, 0.5];
        let h = entropy(&p, None).expect("ok");
        assert!((h - 2.0_f64.ln()).abs() < 1e-10, "h={h}");
    }

    #[test]
    fn test_entropy_invalid_negative() {
        assert!(entropy(&[-0.1, 1.1], None).is_err());
    }

    #[test]
    fn test_entropy_invalid_sum() {
        assert!(entropy(&[0.3, 0.3], None).is_err());
    }

    #[test]
    fn test_entropy_empty() {
        assert!(entropy(&[], None).is_err());
    }

    // ---- joint entropy ----

    #[test]
    fn test_joint_entropy_non_negative() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let h = joint_entropy(&x, &y, 5).expect("ok");
        assert!(h >= 0.0, "h={h}");
    }

    #[test]
    fn test_joint_entropy_identical_data() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        // H(X,X) ≤ H(X) + H(X) = 2 H(X)
        let h = joint_entropy(&x, &x, 4).expect("ok");
        assert!(h >= 0.0);
    }

    #[test]
    fn test_joint_entropy_length_mismatch() {
        assert!(joint_entropy(&[1.0, 2.0], &[1.0], 2).is_err());
    }

    // ---- conditional entropy ----

    #[test]
    fn test_conditional_entropy_non_negative() {
        let x = vec![1.0, 2.0, 1.0, 2.0, 3.0];
        let y = vec![1.0, 1.0, 2.0, 2.0, 3.0];
        let h = conditional_entropy(&x, &y, 3).expect("ok");
        assert!(h >= 0.0, "h={h}");
    }

    #[test]
    fn test_conditional_entropy_given_self() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        // H(X|X) should be ~0 (up to binning error)
        let h = conditional_entropy(&x, &x, 4).expect("ok");
        assert!(h >= 0.0);
        assert!(h < 0.5, "H(X|X) too large: {h}");
    }

    // ---- mutual information ----

    #[test]
    fn test_mutual_information_non_negative() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let mi = mutual_information(&x, &y, 5).expect("ok");
        assert!(mi >= 0.0, "mi={mi}");
    }

    #[test]
    fn test_mutual_information_self() {
        // I(X;X) = H(X) ≥ 0
        let x: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        let mi = mutual_information(&x, &x, 10).expect("ok");
        assert!(mi >= 0.0);
    }

    // ---- normalized mutual information ----

    #[test]
    fn test_nmi_range() {
        let x: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&v| v * 2.0).collect();
        let nmi = normalized_mutual_information(&x, &y, 10).expect("ok");
        assert!(nmi >= 0.0 && nmi <= 1.0 + 1e-9, "nmi={nmi}");
    }

    // ---- kl_divergence ----

    #[test]
    fn test_kl_divergence_identical() {
        let p = vec![0.2_f64, 0.3, 0.5];
        let kl = kl_divergence(&p, &p).expect("ok");
        assert!(kl.abs() < 1e-10, "kl={kl}");
    }

    #[test]
    fn test_kl_divergence_asymmetry() {
        let p = vec![0.9_f64, 0.1];
        let q = vec![0.5_f64, 0.5];
        let kl_pq = kl_divergence(&p, &q).expect("ok");
        let kl_qp = kl_divergence(&q, &p).expect("ok");
        assert!(kl_pq > 0.0);
        assert!((kl_pq - kl_qp).abs() > 1e-6, "KL should be asymmetric");
    }

    #[test]
    fn test_kl_divergence_q_zero_error() {
        let p = vec![0.5_f64, 0.5];
        let q = vec![1.0_f64, 0.0];
        // KL(p ∥ q) is infinite since p[1] > 0 but q[1] = 0
        assert!(kl_divergence(&p, &q).is_err());
    }

    #[test]
    fn test_kl_divergence_p_zero_ok() {
        // P(x) = 0 contributes 0 regardless of Q(x)
        let p = vec![1.0_f64, 0.0];
        let q = vec![0.5_f64, 0.5];
        let kl = kl_divergence(&p, &q).expect("ok");
        assert!(kl >= 0.0);
    }

    #[test]
    fn test_kl_divergence_length_mismatch() {
        assert!(kl_divergence(&[0.5, 0.5], &[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]).is_err());
    }

    // ---- js_divergence ----

    #[test]
    fn test_js_divergence_symmetric() {
        let p = vec![0.7_f64, 0.3];
        let q = vec![0.4_f64, 0.6];
        let js_pq = js_divergence(&p, &q).expect("ok");
        let js_qp = js_divergence(&q, &p).expect("ok");
        assert!((js_pq - js_qp).abs() < 1e-10, "JS should be symmetric");
    }

    #[test]
    fn test_js_divergence_identical_zero() {
        let p = vec![0.3_f64, 0.4, 0.3];
        let js = js_divergence(&p, &p).expect("ok");
        assert!(js.abs() < 1e-10, "js={js}");
    }

    #[test]
    fn test_js_divergence_max() {
        let p = vec![1.0_f64, 0.0];
        let q = vec![0.0_f64, 1.0];
        let js = js_divergence(&p, &q).expect("ok");
        assert!((js - 2.0_f64.ln()).abs() < 1e-10, "js={js}");
    }

    // ---- total_variation ----

    #[test]
    fn test_total_variation_identical() {
        let p = vec![0.5_f64, 0.5];
        let tv = total_variation(&p, &p).expect("ok");
        assert_eq!(tv, 0.0);
    }

    #[test]
    fn test_total_variation_disjoint() {
        let p = vec![1.0_f64, 0.0];
        let q = vec![0.0_f64, 1.0];
        let tv = total_variation(&p, &q).expect("ok");
        assert!((tv - 1.0).abs() < 1e-10, "tv={tv}");
    }

    #[test]
    fn test_total_variation_half() {
        let p = vec![0.5_f64, 0.5];
        let q = vec![1.0_f64, 0.0];
        let tv = total_variation(&p, &q).expect("ok");
        assert!((tv - 0.5).abs() < 1e-10, "tv={tv}");
    }

    // ---- hellinger_distance ----

    #[test]
    fn test_hellinger_identical() {
        let p = vec![0.25_f64; 4];
        let h = hellinger_distance(&p, &p).expect("ok");
        assert!(h.abs() < 1e-10, "h={h}");
    }

    #[test]
    fn test_hellinger_disjoint() {
        let p = vec![1.0_f64, 0.0];
        let q = vec![0.0_f64, 1.0];
        let h = hellinger_distance(&p, &q).expect("ok");
        assert!((h - 1.0).abs() < 1e-10, "h={h}");
    }

    #[test]
    fn test_hellinger_range() {
        let p = vec![0.7_f64, 0.3];
        let q = vec![0.2_f64, 0.8];
        let h = hellinger_distance(&p, &q).expect("ok");
        assert!(h >= 0.0 && h <= 1.0 + 1e-10, "h={h}");
    }

    // ---- information criteria ----

    #[test]
    fn test_aic_formula() {
        let ll = -50.0_f64;
        let k = 3;
        assert_eq!(aic(ll, k), 2.0 * 3.0 - 2.0 * (-50.0));
    }

    #[test]
    fn test_bic_formula() {
        let ll = -50.0_f64;
        let k = 3;
        let n = 100;
        let expected = 3.0 * (100_f64).ln() - 2.0 * (-50.0);
        assert!((bic(ll, k, n) - expected).abs() < 1e-10);
    }

    #[test]
    fn test_aicc_greater_than_aic() {
        let ll = -50.0_f64;
        let k = 3;
        let n = 20;
        assert!(aicc(ll, k, n) >= aic(ll, k) - 1e-10);
    }

    #[test]
    fn test_aicc_converges_to_aic_large_n() {
        let ll = -50.0_f64;
        let k = 3;
        let large_n = 1_000_000;
        let correction = (aicc(ll, k, large_n) - aic(ll, k)).abs();
        assert!(correction < 1e-3, "correction={correction}");
    }

    #[test]
    fn test_hqic_formula() {
        let ll = -100.0_f64;
        let k = 5;
        let n = 100;
        let expected = 2.0 * 5.0 * (100_f64).ln().ln() - 2.0 * (-100.0);
        assert!((hqic(ll, k, n) - expected).abs() < 1e-10);
    }

    #[test]
    fn test_aic_bic_ordering() {
        // For large n, BIC penalty > AIC penalty → BIC selects simpler models
        let ll = -100.0_f64;
        let k = 5;
        let n = 1000;
        let a = aic(ll, k);
        let b = bic(ll, k, n);
        // BIC > AIC when ln(n) > 2, i.e. n > e² ≈ 7.4
        assert!(b > a, "bic={b} should exceed aic={a} for n={n}");
    }
}
