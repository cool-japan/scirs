//! Cross-validation utilities for [`ProphetModel`].
//!
//! Implements a Prophet-style expanding-window cross-validation that places
//! cutoff dates at regular intervals and evaluates forecasting accuracy over
//! a fixed horizon.

use scirs2_core::ndarray::Array1;

use crate::error::{Result, TimeSeriesError};
use crate::prophet::{prophet_metrics, ProphetForecast, ProphetMetrics, ProphetModel};

// ─────────────────────────────────────────────────────────────── public types ─

/// Results from [`prophet_cv`].
#[derive(Debug, Clone)]
pub struct ProphetCvResult {
    /// Cutoff timestamp for each CV fold.
    pub cutoffs: Vec<f64>,
    /// RMSE per fold.
    pub rmse: Vec<f64>,
    /// MAE per fold.
    pub mae: Vec<f64>,
    /// Mean absolute percentage error (%) per fold.
    pub mape: Vec<f64>,
    /// Fraction of actuals within the 95 % prediction interval per fold.
    pub coverage: Vec<f64>,
}

impl ProphetCvResult {
    /// Compute mean RMSE across all folds.
    pub fn mean_rmse(&self) -> f64 {
        mean_slice(&self.rmse)
    }

    /// Compute mean MAE across all folds.
    pub fn mean_mae(&self) -> f64 {
        mean_slice(&self.mae)
    }

    /// Compute mean MAPE across all folds.
    pub fn mean_mape(&self) -> f64 {
        mean_slice(&self.mape)
    }

    /// Compute mean coverage across all folds.
    pub fn mean_coverage(&self) -> f64 {
        mean_slice(&self.coverage)
    }
}

// ─────────────────────────────────────────────────────────────────── main API ─

/// Run expanding-window cross-validation for a Prophet model.
///
/// # Arguments
/// * `timestamps` – observed timestamps in the original scale.
/// * `values`     – observed values.
/// * `initial_window` – size of the initial training window (in the same units
///   as `timestamps`).
/// * `horizon`    – forecast horizon (in the same units as `timestamps`).
/// * `period`     – spacing between successive cutoffs (in the same units as
///   `timestamps`).
///
/// # Returns
/// A [`ProphetCvResult`] containing per-cutoff metrics.
pub fn prophet_cv(
    timestamps: &Array1<f64>,
    values: &Array1<f64>,
    initial_window: f64,
    horizon: f64,
    period: f64,
) -> Result<ProphetCvResult> {
    let n = timestamps.len();
    if n < 4 {
        return Err(TimeSeriesError::InsufficientData {
            message: "prophet_cv".to_string(),
            required: 4,
            actual: n,
        });
    }
    if values.len() != n {
        return Err(TimeSeriesError::DimensionMismatch {
            expected: n,
            actual: values.len(),
        });
    }
    if horizon <= 0.0 {
        return Err(TimeSeriesError::InvalidParameter {
            name: "horizon".to_string(),
            message: "must be positive".to_string(),
        });
    }
    if period <= 0.0 {
        return Err(TimeSeriesError::InvalidParameter {
            name: "period".to_string(),
            message: "must be positive".to_string(),
        });
    }
    if initial_window <= 0.0 {
        return Err(TimeSeriesError::InvalidParameter {
            name: "initial_window".to_string(),
            message: "must be positive".to_string(),
        });
    }

    let t0 = timestamps[0];
    let t_end = timestamps[n - 1];

    // Build cutoff list: t0 + initial_window, t0 + initial_window + period, …
    // Each cutoff `c` requires data in `[c, c + horizon]`.
    let mut cutoffs: Vec<f64> = Vec::new();
    let mut c = t0 + initial_window;
    while c + horizon <= t_end + f64::EPSILON {
        cutoffs.push(c);
        c += period;
    }

    if cutoffs.is_empty() {
        return Err(TimeSeriesError::InsufficientData {
            message: "no valid CV cutoffs found — check initial_window, horizon, and period"
                .to_string(),
            required: 1,
            actual: 0,
        });
    }

    let mut cv_rmse = Vec::with_capacity(cutoffs.len());
    let mut cv_mae = Vec::with_capacity(cutoffs.len());
    let mut cv_mape = Vec::with_capacity(cutoffs.len());
    let mut cv_coverage = Vec::with_capacity(cutoffs.len());

    for &cutoff in &cutoffs {
        // Training slice: timestamps ≤ cutoff
        let train_idx: Vec<usize> = (0..n).filter(|&i| timestamps[i] <= cutoff).collect();
        // Test slice: cutoff < timestamps ≤ cutoff + horizon
        let test_idx: Vec<usize> = (0..n)
            .filter(|&i| timestamps[i] > cutoff && timestamps[i] <= cutoff + horizon)
            .collect();

        if train_idx.len() < 3 || test_idx.is_empty() {
            continue;
        }

        let t_train: Array1<f64> =
            Array1::from_vec(train_idx.iter().map(|&i| timestamps[i]).collect());
        let y_train: Array1<f64> = Array1::from_vec(train_idx.iter().map(|&i| values[i]).collect());
        let t_test: Array1<f64> =
            Array1::from_vec(test_idx.iter().map(|&i| timestamps[i]).collect());
        let y_test: Array1<f64> = Array1::from_vec(test_idx.iter().map(|&i| values[i]).collect());

        let mut model = build_cv_model();
        match model.fit(&t_train, &y_train) {
            Err(_) => continue,
            Ok(()) => {}
        }
        let fc = match model.predict(&t_test) {
            Err(_) => continue,
            Ok(f) => f,
        };

        let metrics = prophet_metrics(&fc, &y_test);
        let cov = compute_coverage(&fc, &y_test);

        cv_rmse.push(metrics.rmse);
        cv_mae.push(metrics.mae);
        cv_mape.push(metrics.mape);
        cv_coverage.push(cov);
    }

    if cv_rmse.is_empty() {
        return Err(TimeSeriesError::FittingError(
            "all CV folds failed to fit".to_string(),
        ));
    }

    // Trim cutoffs to match successful fold count
    let n_success = cv_rmse.len();
    let cutoffs_out: Vec<f64> = cutoffs.into_iter().take(n_success).collect();

    Ok(ProphetCvResult {
        cutoffs: cutoffs_out,
        rmse: cv_rmse,
        mae: cv_mae,
        mape: cv_mape,
        coverage: cv_coverage,
    })
}

// ──────────────────────────────────────────────────────── aggregate CV metrics ─

/// Aggregate CV metrics computed from a single [`ProphetCvResult`].
#[derive(Debug, Clone)]
pub struct ProphetCvSummary {
    /// Number of CV folds evaluated.
    pub n_folds: usize,
    /// Mean RMSE across folds.
    pub mean_rmse: f64,
    /// Mean MAE across folds.
    pub mean_mae: f64,
    /// Mean MAPE across folds.
    pub mean_mape: f64,
    /// Mean 95 % interval coverage across folds.
    pub mean_coverage: f64,
}

/// Summarise a [`ProphetCvResult`] into a single set of aggregate metrics.
pub fn summarise_cv(result: &ProphetCvResult) -> ProphetCvSummary {
    ProphetCvSummary {
        n_folds: result.rmse.len(),
        mean_rmse: result.mean_rmse(),
        mean_mae: result.mean_mae(),
        mean_mape: result.mean_mape(),
        mean_coverage: result.mean_coverage(),
    }
}

// ──────────────────────────────────────────────────────────────────── helpers ─

/// Construct a lightweight Prophet model for CV (fewer changepoints for speed).
fn build_cv_model() -> ProphetModel {
    ProphetModel::new()
        .with_n_changepoints(10)
        .with_yearly_seasonality(false)
        .with_weekly_seasonality(false)
}

/// Compute the fraction of actuals that fall within the prediction interval.
fn compute_coverage(fc: &ProphetForecast, actuals: &Array1<f64>) -> f64 {
    let n = fc.yhat.len().min(actuals.len());
    if n == 0 {
        return 0.0;
    }
    let hits: usize = (0..n)
        .filter(|&i| actuals[i] >= fc.yhat_lower[i] && actuals[i] <= fc.yhat_upper[i])
        .count();
    hits as f64 / n as f64
}

/// Compute the arithmetic mean of a slice (returns 0 for empty slices).
fn mean_slice(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.iter().sum::<f64>() / v.len() as f64
}

// ─────────────────────────────────────────────────────────────────────── tests ─

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    fn make_linear(n: usize, slope: f64, intercept: f64) -> (Array1<f64>, Array1<f64>) {
        let t: Array1<f64> = Array1::linspace(0.0, (n - 1) as f64, n);
        let y: Array1<f64> = t.mapv(|ti| slope * ti + intercept);
        (t, y)
    }

    #[test]
    fn test_cv_runs_on_linear_series() {
        let (t, y) = make_linear(100, 1.0, 5.0);
        let result = prophet_cv(&t, &y, 50.0, 10.0, 10.0).expect("CV should succeed");
        assert!(!result.rmse.is_empty());
        assert_eq!(result.rmse.len(), result.mae.len());
        assert_eq!(result.rmse.len(), result.coverage.len());
    }

    #[test]
    fn test_cv_metrics_finite() {
        let (t, y) = make_linear(120, 0.5, 2.0);
        let result = prophet_cv(&t, &y, 60.0, 15.0, 15.0).expect("CV should succeed");
        for (&rmse, (&mae, &mape)) in result
            .rmse
            .iter()
            .zip(result.mae.iter().zip(result.mape.iter()))
        {
            assert!(rmse.is_finite(), "RMSE must be finite");
            assert!(mae.is_finite(), "MAE must be finite");
            assert!(mape.is_finite(), "MAPE must be finite");
        }
    }

    #[test]
    fn test_cv_coverage_in_range() {
        let (t, y) = make_linear(150, 1.0, 0.0);
        let result = prophet_cv(&t, &y, 75.0, 20.0, 20.0).expect("CV should succeed");
        for &cov in &result.coverage {
            assert!(cov >= 0.0 && cov <= 1.0, "coverage={cov} must be in [0,1]");
        }
    }

    #[test]
    fn test_cv_insufficient_data() {
        let t: Array1<f64> = Array1::from_vec(vec![0.0, 1.0]);
        let y: Array1<f64> = Array1::from_vec(vec![1.0, 2.0]);
        assert!(prophet_cv(&t, &y, 1.0, 0.5, 0.5).is_err());
    }

    #[test]
    fn test_cv_invalid_horizon() {
        let (t, y) = make_linear(50, 1.0, 0.0);
        assert!(prophet_cv(&t, &y, 20.0, 0.0, 5.0).is_err());
    }

    #[test]
    fn test_cv_no_valid_cutoffs() {
        let (t, y) = make_linear(20, 1.0, 0.0);
        // initial_window > total range → no cutoffs
        assert!(prophet_cv(&t, &y, 30.0, 5.0, 5.0).is_err());
    }

    #[test]
    fn test_summarise_cv() {
        let (t, y) = make_linear(120, 1.0, 0.0);
        let result = prophet_cv(&t, &y, 60.0, 15.0, 15.0).expect("CV should succeed");
        let summary = summarise_cv(&result);
        assert!(summary.n_folds > 0);
        assert!(summary.mean_rmse >= 0.0);
        assert!(summary.mean_coverage >= 0.0 && summary.mean_coverage <= 1.0);
    }

    #[test]
    fn test_cv_result_means() {
        let result = ProphetCvResult {
            cutoffs: vec![10.0, 20.0],
            rmse: vec![2.0, 4.0],
            mae: vec![1.0, 3.0],
            mape: vec![5.0, 15.0],
            coverage: vec![0.9, 0.8],
        };
        assert!((result.mean_rmse() - 3.0).abs() < 1e-10);
        assert!((result.mean_mae() - 2.0).abs() < 1e-10);
        assert!((result.mean_mape() - 10.0).abs() < 1e-10);
        assert!((result.mean_coverage() - 0.85).abs() < 1e-10);
    }
}
