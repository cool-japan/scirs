//! Fitness metrics for symbolic regression candidates.

/// Fitness measure (lower `combined` is better).
///
/// `mse` is mean squared error, `r_squared` is the coefficient of
/// determination, `combined` is `mse + complexity_penalty * node_count`
/// (parsimony-regularised).
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct Fitness {
    /// Mean squared error.
    pub mse: f64,
    /// R² (coefficient of determination).
    pub r_squared: f64,
    /// Combined fitness with parsimony penalty.
    pub combined: f64,
}

impl Fitness {
    /// New fitness from MSE, R², and combined score.
    pub fn new(mse: f64, r_squared: f64, combined: f64) -> Self {
        Self {
            mse,
            r_squared,
            combined,
        }
    }

    /// "Worst" fitness (used for invalid candidates).
    pub fn worst() -> Self {
        Self {
            mse: f64::INFINITY,
            r_squared: f64::NEG_INFINITY,
            combined: f64::INFINITY,
        }
    }

    /// Compute fitness from a prediction array vs a target array.
    ///
    /// Returns [`Self::worst`] if lengths differ, the slices are empty,
    /// or any prediction is non-finite.
    pub fn compute(predictions: &[f64], targets: &[f64], complexity: f64) -> Self {
        if predictions.len() != targets.len() || predictions.is_empty() {
            return Self::worst();
        }
        let n = predictions.len() as f64;
        let mean_target: f64 = targets.iter().sum::<f64>() / n;
        let mut ss_res = 0.0;
        let mut ss_tot = 0.0;
        for (p, t) in predictions.iter().zip(targets.iter()) {
            if !p.is_finite() {
                return Self::worst();
            }
            ss_res += (p - t).powi(2);
            ss_tot += (t - mean_target).powi(2);
        }
        let mse = ss_res / n;
        let r_squared = if ss_tot > 0.0 {
            1.0 - ss_res / ss_tot
        } else {
            0.0
        };
        let combined = mse + complexity;
        Self {
            mse,
            r_squared,
            combined,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn worst_combined_is_inf() {
        assert!(Fitness::worst().combined.is_infinite());
    }

    #[test]
    fn perfect_fit_zero_mse() {
        let p = [1.0, 2.0, 3.0];
        let t = [1.0, 2.0, 3.0];
        let f = Fitness::compute(&p, &t, 0.0);
        assert!(f.mse.abs() < 1e-15);
        assert!((f.r_squared - 1.0).abs() < 1e-12);
    }

    #[test]
    fn nonfinite_predictions_are_worst() {
        let p = [1.0, f64::NAN];
        let t = [1.0, 2.0];
        let f = Fitness::compute(&p, &t, 0.0);
        assert!(f.combined.is_infinite());
    }

    #[test]
    fn length_mismatch_is_worst() {
        let f = Fitness::compute(&[1.0, 2.0], &[1.0], 0.0);
        assert!(f.combined.is_infinite());
    }

    #[test]
    fn zero_variance_targets_yield_zero_r_squared() {
        // When all targets are equal, ss_tot = 0 and R² is defined as 0.
        let p = [1.0, 1.0, 1.0];
        let t = [1.0, 1.0, 1.0];
        let f = Fitness::compute(&p, &t, 0.0);
        assert_eq!(f.r_squared, 0.0);
        assert_eq!(f.mse, 0.0);
    }
}
