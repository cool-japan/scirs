//! PELT (Pruned Exact Linear Time) change-point detection
//!
//! Implements Killick et al. (2012) "Optimal Detection of Changepoints With
//! a Linear Computational Cost". Uses Gaussian negative log-likelihood as the
//! cost function and maintains a pruning set to avoid exhaustive search.

use scirs2_core::ndarray::Array1;

use crate::error::{Result, TimeSeriesError};

/// PELT change-point detector
///
/// Detects change points in mean/variance of a time series using
/// the Pruned Exact Linear Time algorithm (Killick 2012).
#[derive(Debug, Clone)]
pub struct PELTDetector {
    /// Penalty for adding a change point (BIC-like: higher ⇒ fewer CPs)
    penalty: f64,
    /// Minimum segment length between change points
    min_size: usize,
}

impl PELTDetector {
    /// Create a new PELT detector
    ///
    /// # Arguments
    ///
    /// * `penalty` - Penalty for adding a change point (β in Killick 2012).
    ///   Typical values: `2 * ln(n)` (BIC) or a fixed constant.
    ///   Default `min_size` is 2.
    pub fn new(penalty: f64) -> Self {
        PELTDetector {
            penalty,
            min_size: 2,
        }
    }

    /// Create a PELT detector with explicit min_size
    ///
    /// # Arguments
    ///
    /// * `penalty` - Penalty parameter (β)
    /// * `min_size` - Minimum segment length between change points (≥ 1)
    pub fn with_min_size(penalty: f64, min_size: usize) -> Self {
        PELTDetector {
            penalty,
            min_size: min_size.max(1),
        }
    }

    /// Detect change points in a time series
    ///
    /// Returns the indices (exclusive right endpoints) of detected change points.
    /// For example, if change points are at positions 50 and 100, returns `[50, 100]`.
    pub fn detect(&self, series: &Array1<f64>) -> Result<Vec<usize>> {
        let n = series.len();
        if n < 2 {
            return Err(TimeSeriesError::InsufficientData {
                message: "PELT requires at least 2 data points".to_string(),
                required: 2,
                actual: n,
            });
        }

        // Pre-compute prefix sums for efficient segment cost computation
        let (prefix_sum, prefix_sum2) = compute_prefix_sums(series);

        // F[t] = minimum cost for segmenting series[0..t]
        // F[0] = -penalty (sentinel)
        let mut f = vec![f64::INFINITY; n + 1];
        f[0] = -self.penalty;

        // Store the last change-point for each position
        let mut last_cp = vec![0usize; n + 1];

        // Pruning set: candidate start positions for the current segment
        let mut candidates: Vec<usize> = vec![0];

        for t in 1..=n {
            let mut best_cost = f64::INFINITY;
            let mut best_tau = 0;

            // Evaluate cost for each candidate start position
            let mut new_candidates = Vec::new();

            for &tau in &candidates {
                // Segment is series[tau..t], which has length t - tau
                if t - tau < self.min_size {
                    new_candidates.push(tau);
                    continue;
                }

                let seg_cost = gaussian_cost(&prefix_sum, &prefix_sum2, tau, t);
                let total = f[tau] + seg_cost + self.penalty;

                if total < best_cost {
                    best_cost = total;
                    best_tau = tau;
                }

                // PELT pruning: keep tau in candidates for future t only if
                // F[tau] + C(tau+1:t) + β ≤ F[t]
                // After computing best_cost, we can prune below
                new_candidates.push(tau);
            }

            f[t] = best_cost;
            last_cp[t] = best_tau;

            // Apply pruning: remove candidates that can never be optimal
            // Condition to REMOVE: F[tau] + C(tau+1:t) + β > F[t]
            // But we need F[t] first, which we just computed.
            // Safe to prune for next round:
            candidates = new_candidates
                .into_iter()
                .filter(|&tau| {
                    if t - tau < self.min_size {
                        return true; // Keep; may become valid later
                    }
                    let seg_cost = gaussian_cost(&prefix_sum, &prefix_sum2, tau, t);
                    f[tau] + seg_cost + self.penalty <= f[t]
                })
                .collect();

            // Add current position as a candidate for next iteration
            candidates.push(t);
        }

        // Backtrack to find change points
        let mut change_points = Vec::new();
        let mut t = n;
        loop {
            let tau = last_cp[t];
            if tau == 0 {
                break;
            }
            change_points.push(tau);
            t = tau;
        }
        change_points.reverse();
        Ok(change_points)
    }
}

/// Pre-compute prefix sums for O(1) segment statistics
fn compute_prefix_sums(series: &Array1<f64>) -> (Vec<f64>, Vec<f64>) {
    let n = series.len();
    let mut prefix_sum = vec![0.0f64; n + 1];
    let mut prefix_sum2 = vec![0.0f64; n + 1];
    for i in 0..n {
        prefix_sum[i + 1] = prefix_sum[i] + series[i];
        prefix_sum2[i + 1] = prefix_sum2[i] + series[i] * series[i];
    }
    (prefix_sum, prefix_sum2)
}

/// Gaussian negative log-likelihood cost for segment series[start..end]
///
/// C(y_{s+1:t}) = -(t-s)/2 * ln(Var) + (t-s)/2
/// where Var = sample variance of the segment.
///
/// Uses the fact that for n points:
///   sum(y) and sum(y^2) can be computed in O(1) with prefix sums.
///
/// When variance is zero or segment is too small, returns a large cost.
fn gaussian_cost(prefix_sum: &[f64], prefix_sum2: &[f64], start: usize, end: usize) -> f64 {
    let len = end - start;
    if len == 0 {
        return 0.0;
    }
    if len == 1 {
        return 0.0; // Single point has zero variance; cost = 0 (log(0) → handle specially)
    }

    let sum = prefix_sum[end] - prefix_sum[start];
    let sum2 = prefix_sum2[end] - prefix_sum2[start];
    let n = len as f64;

    // Sample variance (population variance for cost)
    let variance = (sum2 - sum * sum / n) / n;

    if variance < 1e-300 {
        // Near-constant segment: cost is very low (perfect fit)
        return 0.0;
    }

    // Negative log-likelihood for Gaussian: n/2 * ln(2π*var) + n/2
    // Drop constant 2π term; use relative cost:
    // C = n/2 * ln(var) + n/2
    n / 2.0 * variance.ln() + n / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    #[test]
    fn test_pelt_detects_two_change_points() {
        // Piecewise constant: 50 zeros, 50 fives, 50 zeros
        let mut data = vec![0.0f64; 50];
        data.extend(vec![5.0f64; 50]);
        data.extend(vec![0.0f64; 50]);
        let series = Array1::from_vec(data);

        let detector = PELTDetector::with_min_size(10.0, 2);
        let cps = detector.detect(&series).expect("PELT detect failed");

        assert!(
            !cps.is_empty(),
            "Expected at least one change point, got none"
        );

        // Should detect change points around 50 and 100
        let has_cp_near_50 = cps.iter().any(|&cp| (cp as isize - 50).abs() <= 3);
        let has_cp_near_100 = cps.iter().any(|&cp| (cp as isize - 100).abs() <= 3);

        assert!(
            has_cp_near_50,
            "Expected change point near 50, got: {cps:?}"
        );
        assert!(
            has_cp_near_100,
            "Expected change point near 100, got: {cps:?}"
        );
    }

    #[test]
    fn test_pelt_no_change_points_iid() {
        // iid Gaussian should have 0 or very few spurious change points with high penalty
        let n = 100;
        // Deterministic pseudo-normal using wrapping arithmetic
        let series: Array1<f64> = Array1::from_iter((0..n).map(|i| {
            let h1 = (i as u64).wrapping_mul(1103515245).wrapping_add(12345) & 0x7fffffff;
            let h2 = ((i as u64).wrapping_add(7))
                .wrapping_mul(1103515245)
                .wrapping_add(12345)
                & 0x7fffffff;
            let u1 = (h1 as f64 / 0x7fffffff_u64 as f64).max(1e-12);
            let u2 = h2 as f64 / 0x7fffffff_u64 as f64;
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        }));

        // High penalty → fewer false positives
        let detector = PELTDetector::with_min_size(20.0, 5);
        let cps = detector.detect(&series).expect("PELT detect failed");

        // For truly iid data with a high penalty, we expect 0 or very few CPs
        assert!(
            cps.len() <= 2,
            "Expected ≤ 2 change points on iid data, got {}: {cps:?}",
            cps.len()
        );
    }
}
