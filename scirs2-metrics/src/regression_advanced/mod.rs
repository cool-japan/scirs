//! Advanced regression and probabilistic forecasting metrics.
//!
//! Provides quantile loss (pinball loss), prediction interval metrics,
//! Winkler score, Continuous Ranked Probability Score (CRPS), and the
//! multivariate Energy Score.

use crate::error::{MetricsError, Result};

/// Quantile loss (pinball loss) for a given quantile level.
///
/// For a quantile `alpha` ∈ (0, 1):
///
/// L_alpha(y, ŷ) = alpha * (y - ŷ)_+ + (1 - alpha) * (ŷ - y)_+
///
/// where (·)_+ = max(·, 0).
///
/// # Arguments
/// * `y_true` — observed values
/// * `y_pred` — predicted quantile values
/// * `quantile` — quantile level in (0, 1)
///
/// # Returns
/// Mean quantile loss.
pub fn quantile_loss(y_true: &[f64], y_pred: &[f64], quantile: f64) -> Result<f64> {
    if y_true.len() != y_pred.len() {
        return Err(MetricsError::InvalidInput(format!(
            "y_true ({}) and y_pred ({}) have different lengths",
            y_true.len(),
            y_pred.len()
        )));
    }
    if y_true.is_empty() {
        return Err(MetricsError::InvalidInput("Empty input arrays".to_string()));
    }
    if quantile <= 0.0 || quantile >= 1.0 {
        return Err(MetricsError::InvalidInput(format!(
            "quantile must be in (0, 1), got {quantile}"
        )));
    }

    let loss: f64 = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(&y, &yp)| {
            let diff = y - yp;
            if diff >= 0.0 {
                quantile * diff
            } else {
                (1.0 - quantile) * (-diff)
            }
        })
        .sum();

    Ok(loss / y_true.len() as f64)
}

/// Coverage error: fraction of observations outside the prediction interval.
///
/// # Arguments
/// * `y_true` — observed values
/// * `y_lower` — lower bounds of prediction intervals
/// * `y_upper` — upper bounds of prediction intervals
///
/// # Returns
/// Fraction in [0, 1] of observations not covered by their interval.
pub fn coverage_error(y_true: &[f64], y_lower: &[f64], y_upper: &[f64]) -> Result<f64> {
    validate_triple_lengths(y_true, y_lower, y_upper)?;
    if y_true.is_empty() {
        return Err(MetricsError::InvalidInput("Empty input arrays".to_string()));
    }

    let not_covered = y_true
        .iter()
        .zip(y_lower.iter())
        .zip(y_upper.iter())
        .filter(|((&y, &lo), &hi)| y < lo || y > hi)
        .count();

    Ok(not_covered as f64 / y_true.len() as f64)
}

/// Mean width of prediction intervals.
///
/// # Arguments
/// * `y_lower` — lower bounds of prediction intervals
/// * `y_upper` — upper bounds of prediction intervals
pub fn interval_width(y_lower: &[f64], y_upper: &[f64]) -> Result<f64> {
    if y_lower.len() != y_upper.len() {
        return Err(MetricsError::InvalidInput(format!(
            "y_lower ({}) and y_upper ({}) have different lengths",
            y_lower.len(),
            y_upper.len()
        )));
    }
    if y_lower.is_empty() {
        return Err(MetricsError::InvalidInput("Empty input arrays".to_string()));
    }

    let total: f64 = y_lower
        .iter()
        .zip(y_upper.iter())
        .map(|(&lo, &hi)| {
            if hi < lo {
                0.0 // degenerate interval
            } else {
                hi - lo
            }
        })
        .sum();

    Ok(total / y_lower.len() as f64)
}

/// Winkler score for prediction intervals.
///
/// For each observation y with interval [L, U] at level alpha:
///
/// S = (U - L) + (2/alpha) * max(L - y, 0) + (2/alpha) * max(y - U, 0)
///
/// The Winkler score is the mean of these values; lower is better.
///
/// # Arguments
/// * `y_true` — observed values
/// * `y_lower` — lower bounds of prediction intervals
/// * `y_upper` — upper bounds of prediction intervals
/// * `alpha` — significance level of the prediction interval (e.g. 0.05 for 95%)
pub fn winkler_score(y_true: &[f64], y_lower: &[f64], y_upper: &[f64], alpha: f64) -> Result<f64> {
    validate_triple_lengths(y_true, y_lower, y_upper)?;
    if y_true.is_empty() {
        return Err(MetricsError::InvalidInput("Empty input arrays".to_string()));
    }
    if alpha <= 0.0 || alpha >= 1.0 {
        return Err(MetricsError::InvalidInput(format!(
            "alpha must be in (0, 1), got {alpha}"
        )));
    }

    let penalty_factor = 2.0 / alpha;
    let total: f64 = y_true
        .iter()
        .zip(y_lower.iter())
        .zip(y_upper.iter())
        .map(|((&y, &lo), &hi)| {
            let width = (hi - lo).max(0.0);
            let below = (lo - y).max(0.0);
            let above = (y - hi).max(0.0);
            width + penalty_factor * below + penalty_factor * above
        })
        .sum();

    Ok(total / y_true.len() as f64)
}

/// Continuous Ranked Probability Score (CRPS) for ensemble forecasts.
///
/// CRPS(F, y) = E|X - y| - 0.5 * E|X - X'|
///
/// where X and X' are independent draws from the forecast distribution F
/// (approximated by the ensemble members).
///
/// # Arguments
/// * `y_true` — observed scalar values, one per forecast case
/// * `ensemble` — ensemble members for each forecast case; inner Vec is members
///
/// # Returns
/// Mean CRPS across all forecast cases.
pub fn continuous_ranked_probability_score(y_true: &[f64], ensemble: &[Vec<f64>]) -> Result<f64> {
    if y_true.len() != ensemble.len() {
        return Err(MetricsError::InvalidInput(format!(
            "y_true ({}) and ensemble ({}) have different lengths",
            y_true.len(),
            ensemble.len()
        )));
    }
    if y_true.is_empty() {
        return Err(MetricsError::InvalidInput("Empty input arrays".to_string()));
    }

    let total: f64 = y_true
        .iter()
        .zip(ensemble.iter())
        .map(|(&y, members)| crps_single(y, members))
        .sum::<f64>();

    Ok(total / y_true.len() as f64)
}

/// CRPS for a single observation using the energy-score decomposition.
fn crps_single(y: f64, members: &[f64]) -> f64 {
    if members.is_empty() {
        return 0.0;
    }
    let m = members.len() as f64;

    // E|X - y|
    let e_xy: f64 = members.iter().map(|&x| (x - y).abs()).sum::<f64>() / m;

    // E|X - X'| via double sum (O(m^2) but exact)
    let mut e_xx = 0.0_f64;
    for &xi in members {
        for &xj in members {
            e_xx += (xi - xj).abs();
        }
    }
    e_xx /= m * m;

    e_xy - 0.5 * e_xx
}

/// Multivariate Energy Score.
///
/// ES = E||X - y|| - 0.5 * E||X - X'||
///
/// This is a proper multivariate scoring rule generalising CRPS.
///
/// # Arguments
/// * `y_true` — observed multivariate outcomes; outer index = case, inner = dimension
/// * `ensemble` — ensemble forecasts; outer = case, middle = member, inner = dimension
///
/// # Returns
/// Mean Energy Score across cases.
pub fn energy_score(y_true: &[Vec<f64>], ensemble: &[Vec<Vec<f64>>]) -> Result<f64> {
    if y_true.len() != ensemble.len() {
        return Err(MetricsError::InvalidInput(format!(
            "y_true ({}) and ensemble ({}) have different lengths",
            y_true.len(),
            ensemble.len()
        )));
    }
    if y_true.is_empty() {
        return Err(MetricsError::InvalidInput("Empty input arrays".to_string()));
    }

    let total: f64 = y_true
        .iter()
        .zip(ensemble.iter())
        .map(|(y, members)| energy_score_single(y, members))
        .sum::<f64>();

    Ok(total / y_true.len() as f64)
}

fn energy_score_single(y: &[f64], members: &[Vec<f64>]) -> f64 {
    if members.is_empty() {
        return 0.0;
    }
    let m = members.len() as f64;

    // E||X - y||
    let e_xy: f64 = members.iter().map(|x| l2_distance(x, y)).sum::<f64>() / m;

    // E||X - X'||
    let mut e_xx = 0.0_f64;
    for xi in members {
        for xj in members {
            e_xx += l2_distance(xi, xj);
        }
    }
    e_xx /= m * m;

    e_xy - 0.5 * e_xx
}

fn l2_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| (ai - bi).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn validate_triple_lengths(y_true: &[f64], y_lower: &[f64], y_upper: &[f64]) -> Result<()> {
    if y_true.len() != y_lower.len() || y_true.len() != y_upper.len() {
        return Err(MetricsError::InvalidInput(format!(
            "y_true ({}), y_lower ({}) and y_upper ({}) must all have the same length",
            y_true.len(),
            y_lower.len(),
            y_upper.len()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantile_loss_median_symmetric() {
        // For quantile=0.5 the loss is 0.5 * MAE
        let y_true = vec![1.0, 2.0, 3.0];
        let y_pred = vec![1.0, 2.0, 3.0];
        let loss = quantile_loss(&y_true, &y_pred, 0.5).expect("should succeed");
        assert!(
            (loss - 0.0).abs() < 1e-10,
            "Perfect predictions → 0 loss, got {loss}"
        );
    }

    #[test]
    fn test_quantile_loss_asymmetric() {
        // Underprediction: y=2, ŷ=1, quantile=0.9
        // loss = 0.9 * 1 = 0.9
        let y_true = vec![2.0];
        let y_pred = vec![1.0];
        let loss = quantile_loss(&y_true, &y_pred, 0.9).expect("should succeed");
        assert!((loss - 0.9).abs() < 1e-10);
    }

    #[test]
    fn test_quantile_loss_overprediction() {
        // Overprediction: y=1, ŷ=2, quantile=0.9
        // loss = (1 - 0.9) * 1 = 0.1
        let y_true = vec![1.0];
        let y_pred = vec![2.0];
        let loss = quantile_loss(&y_true, &y_pred, 0.9).expect("should succeed");
        assert!((loss - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_coverage_error_all_covered() {
        let y_true = vec![1.0, 2.0, 3.0];
        let y_lower = vec![0.5, 1.5, 2.5];
        let y_upper = vec![1.5, 2.5, 3.5];
        let err = coverage_error(&y_true, &y_lower, &y_upper).expect("should succeed");
        assert!((err - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_coverage_error_none_covered() {
        let y_true = vec![0.0, 0.0];
        let y_lower = vec![1.0, 1.0];
        let y_upper = vec![2.0, 2.0];
        let err = coverage_error(&y_true, &y_lower, &y_upper).expect("should succeed");
        assert!((err - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_interval_width() {
        let y_lower = vec![0.0, 1.0, 2.0];
        let y_upper = vec![1.0, 3.0, 4.0];
        let width = interval_width(&y_lower, &y_upper).expect("should succeed");
        // widths: 1, 2, 2 → mean = 5/3
        assert!((width - 5.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_winkler_score_perfect() {
        // All observations exactly on interval boundaries: no penalty
        let y_true = vec![1.0, 2.0];
        let y_lower = vec![1.0, 2.0];
        let y_upper = vec![1.0, 2.0];
        let score = winkler_score(&y_true, &y_lower, &y_upper, 0.05).expect("should succeed");
        // Width=0, no penalties → score=0
        assert!((score - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_winkler_score_penalty() {
        // y=3, interval=[1,2], alpha=0.1: width=1, above=(3-2)=1, penalty=2/0.1*1=20
        let y_true = vec![3.0];
        let y_lower = vec![1.0];
        let y_upper = vec![2.0];
        let score = winkler_score(&y_true, &y_lower, &y_upper, 0.1).expect("should succeed");
        assert!((score - 21.0).abs() < 1e-10, "Expected 21.0, got {score}");
    }

    #[test]
    fn test_crps_deterministic() {
        // Ensemble with single member equal to truth → CRPS = 0
        let y_true = vec![2.0];
        let ensemble = vec![vec![2.0]];
        let crps = continuous_ranked_probability_score(&y_true, &ensemble).expect("should succeed");
        assert!((crps - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_crps_known_value() {
        // Single member: CRPS = |X - y| - 0.5 * |X - X| = |X - y|
        let y_true = vec![1.0];
        let ensemble = vec![vec![3.0]];
        let crps = continuous_ranked_probability_score(&y_true, &ensemble).expect("should succeed");
        assert!((crps - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_energy_score_deterministic() {
        // Single ensemble member equal to truth → ES = 0
        let y_true = vec![vec![1.0, 2.0]];
        let ensemble = vec![vec![vec![1.0, 2.0]]];
        let es = energy_score(&y_true, &ensemble).expect("should succeed");
        assert!((es - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_energy_score_positive() {
        let y_true = vec![vec![0.0, 0.0]];
        let ensemble = vec![vec![vec![1.0, 0.0], vec![-1.0, 0.0]]];
        let es = energy_score(&y_true, &ensemble).expect("should succeed");
        assert!(es > 0.0, "Energy score should be positive, got {es}");
    }

    #[test]
    fn test_quantile_loss_invalid_quantile() {
        assert!(quantile_loss(&[1.0], &[1.0], 0.0).is_err());
        assert!(quantile_loss(&[1.0], &[1.0], 1.0).is_err());
    }
}
