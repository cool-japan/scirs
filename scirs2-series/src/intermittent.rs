//! Intermittent Demand Forecasting
//!
//! Intermittent demand occurs when a product or service is demanded
//! only occasionally, with many zero-demand periods interspersed with
//! non-zero demand observations.  Conventional exponential smoothing
//! methods perform poorly in this setting; the methods in this module
//! are specifically designed for it.
//!
//! # Methods
//!
//! | Struct | Algorithm | Reference |
//! |---|---|---|
//! | [`Croston`] | Croston's method | Croston (1972) |
//! | [`Sba`] | Syntetos-Boylan Approximation | Syntetos & Boylan (2005) |
//! | [`Tsb`] | Teunter-Syntetos-Babai method | Teunter et al. (2011) |
//!
//! Additionally [`classify_demand`] returns a [`DemandType`] for a series
//! based on the ADI (Average Demand Interval) and CV² (squared coefficient of
//! variation of non-zero demands) classification scheme.
//!
//! # References
//!
//! - Croston, J.D. (1972). "Forecasting and stock control for intermittent
//!   demands." *Operational Research Quarterly*, 23(3), 289–303.
//! - Syntetos, A.A. & Boylan, J.E. (2001). "On the bias of intermittent demand
//!   estimates." *International Journal of Production Economics*, 71(1-3), 457–466.
//! - Syntetos, A.A. & Boylan, J.E. (2005). "The accuracy of intermittent demand
//!   estimates." *International Journal of Forecasting*, 21(2), 303–314.
//! - Teunter, R.H., Syntetos, A.A. & Babai, M.Z. (2011). "Intermittent demand:
//!   Linking forecasting to inventory obsolescence." *European Journal of
//!   Operational Research*, 214(3), 606–615.
//! - Boylan, J.E. & Syntetos, A.A. (2007). "The accuracy of a modified Croston
//!   procedure." *International Journal of Production Economics*, 107(2), 511–517.

use crate::error::{Result, TimeSeriesError};

// ─────────────────────────────────────────────────────────────────────────────
// Demand classification
// ─────────────────────────────────────────────────────────────────────────────

/// Classification of demand patterns following the Syntetos-Boylan (2005)
/// 2×2 matrix based on ADI and CV².
///
/// | ADI \ CV² | Low (< 0.49) | High (≥ 0.49) |
/// |---|---|---|
/// | **Low (< 1.32)** | Smooth | Erratic |
/// | **High (≥ 1.32)** | Intermittent | Lumpy |
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DemandType {
    /// Regular, non-zero demand with low variability.
    Smooth,
    /// Irregular size but frequent demand.
    Erratic,
    /// Regular size but infrequent (sporadic) demand.
    Intermittent,
    /// Both infrequent and highly variable demand.
    Lumpy,
}

impl std::fmt::Display for DemandType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DemandType::Smooth => write!(f, "Smooth"),
            DemandType::Erratic => write!(f, "Erratic"),
            DemandType::Intermittent => write!(f, "Intermittent"),
            DemandType::Lumpy => write!(f, "Lumpy"),
        }
    }
}

/// ADI threshold used by the Syntetos-Boylan classification.
pub const ADI_THRESHOLD: f64 = 1.32;
/// CV² threshold used by the Syntetos-Boylan classification.
pub const CV2_THRESHOLD: f64 = 0.49;

/// Compute the Average Demand Interval (ADI) of a demand series.
///
/// ADI = n / (number of non-zero periods).
///
/// Returns `f64::INFINITY` if there are no non-zero demands.
pub fn compute_adi(series: &[f64]) -> f64 {
    let n = series.len();
    if n == 0 {
        return f64::INFINITY;
    }
    let non_zero = series.iter().filter(|&&v| v > 0.0).count();
    if non_zero == 0 {
        return f64::INFINITY;
    }
    n as f64 / non_zero as f64
}

/// Compute the squared coefficient of variation (CV²) of the non-zero demands.
///
/// CV² = Var(non-zero demands) / Mean(non-zero demands)²
///
/// Returns `0.0` if there is fewer than two non-zero observations.
pub fn compute_cv2(series: &[f64]) -> f64 {
    let non_zero: Vec<f64> = series.iter().copied().filter(|&v| v > 0.0).collect();
    let k = non_zero.len();
    if k < 2 {
        return 0.0;
    }
    let mean = non_zero.iter().copied().sum::<f64>() / k as f64;
    if mean.abs() < f64::EPSILON {
        return 0.0;
    }
    let var = non_zero.iter().map(|&d| (d - mean).powi(2)).sum::<f64>() / (k - 1) as f64;
    var / (mean * mean)
}

/// Classify a demand series into one of the four [`DemandType`] categories.
///
/// # Arguments
/// * `series` — Non-negative demand observations (zeros indicate no demand).
///
/// # Returns
/// The [`DemandType`] corresponding to the ADI/CV² quadrant.
pub fn classify_demand(series: &[f64]) -> DemandType {
    let adi = compute_adi(series);
    let cv2 = compute_cv2(series);
    match (adi >= ADI_THRESHOLD, cv2 >= CV2_THRESHOLD) {
        (false, false) => DemandType::Smooth,
        (false, true) => DemandType::Erratic,
        (true, false) => DemandType::Intermittent,
        (true, true) => DemandType::Lumpy,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Croston's method
// ─────────────────────────────────────────────────────────────────────────────

/// Fitted model from Croston's method.
///
/// After calling [`Croston::fit`] the model stores the final smoothed demand
/// size (`z`) and the final smoothed inter-demand interval (`p`), which are
/// combined to produce a constant forecast `z / p`.
#[derive(Debug, Clone)]
pub struct CrostonModel {
    /// Final smoothed demand size (estimate of demand magnitude when non-zero).
    pub z: f64,
    /// Final smoothed inter-demand interval (average gap between non-zero obs).
    pub p: f64,
    /// Smoothing parameter used for both components (same alpha for both).
    pub alpha: f64,
    /// Number of periods since the last non-zero demand (used internally).
    pub q: f64,
    /// The correction factor applied to the ratio z/p (1.0 for Croston, < 1 for SBA).
    correction: f64,
}

impl CrostonModel {
    /// Produce h-step-ahead forecasts.
    ///
    /// Croston's forecast is constant over the horizon:
    /// `f = correction * z / p`
    ///
    /// # Arguments
    /// * `h` — Forecast horizon (number of periods ahead).
    ///
    /// # Returns
    /// A `Vec<f64>` of length h with the constant forecast.
    pub fn forecast(&self, h: usize) -> Vec<f64> {
        if h == 0 {
            return Vec::new();
        }
        let f = self.correction * self.z / self.p;
        vec![f; h]
    }

    /// The point forecast value (constant over all horizons).
    pub fn point_forecast(&self) -> f64 {
        self.correction * self.z / self.p
    }
}

/// Croston's method for intermittent demand forecasting.
///
/// The original algorithm (Croston 1972) separates a demand series into:
/// 1. The *demand size* sequence (non-zero values only).
/// 2. The *inter-demand interval* sequence (gaps between non-zero values).
///
/// Each component is smoothed independently using simple exponential smoothing
/// with the same smoothing parameter α.  The final forecast is `z / p` where z
/// is the smoothed demand and p the smoothed interval.
pub struct Croston;

impl Croston {
    /// Fit Croston's method to a demand series.
    ///
    /// # Arguments
    /// * `demand_series` — Historical demand observations (non-negative).
    /// * `alpha` — Smoothing parameter in (0, 1].  If `None`, a default of 0.1
    ///   is used (as recommended by Croston for intermittent demand).
    ///
    /// # Returns
    /// A fitted [`CrostonModel`].
    pub fn fit(demand_series: &[f64], alpha: Option<f64>) -> Result<CrostonModel> {
        validate_demand_series(demand_series)?;
        let alpha = resolve_alpha(alpha)?;

        let (z, p, q) = croston_recursion(demand_series, alpha, 1.0)?;
        Ok(CrostonModel {
            z,
            p,
            alpha,
            q,
            correction: 1.0,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SBA — Syntetos-Boylan Approximation
// ─────────────────────────────────────────────────────────────────────────────

/// SBA (Syntetos-Boylan Approximation) model, an improved variant of Croston's
/// method that corrects the upward bias by multiplying the demand forecast by
/// `(1 - alpha/2)`.
#[derive(Debug, Clone)]
pub struct SbaModel {
    /// Underlying Croston model state.
    pub inner: CrostonModel,
}

impl SbaModel {
    /// Produce h-step-ahead forecasts with the SBA bias correction.
    pub fn forecast(&self, h: usize) -> Vec<f64> {
        self.inner.forecast(h)
    }

    /// The bias-corrected point forecast.
    pub fn point_forecast(&self) -> f64 {
        self.inner.point_forecast()
    }
}

/// Syntetos-Boylan Approximation for intermittent demand.
///
/// Applies a multiplicative correction factor `(1 - alpha/2)` to the Croston
/// forecast to remove the systematic upward bias identified by Syntetos &
/// Boylan (2001, 2005).
pub struct Sba;

impl Sba {
    /// Fit the SBA method.
    ///
    /// # Arguments
    /// * `demand_series` — Historical demand observations.
    /// * `alpha` — Smoothing parameter in (0, 1].  Defaults to 0.1 if `None`.
    ///
    /// # Returns
    /// A fitted [`SbaModel`].
    pub fn fit(demand_series: &[f64], alpha: Option<f64>) -> Result<SbaModel> {
        validate_demand_series(demand_series)?;
        let alpha = resolve_alpha(alpha)?;

        let correction = 1.0 - alpha / 2.0;
        let (z, p, q) = croston_recursion(demand_series, alpha, correction)?;
        Ok(SbaModel {
            inner: CrostonModel {
                z,
                p,
                alpha,
                q,
                correction,
            },
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TSB — Teunter-Syntetos-Babai method
// ─────────────────────────────────────────────────────────────────────────────

/// Fitted TSB model.
///
/// The TSB method (Teunter et al., 2011) is designed to handle **demand
/// obsolescence**: products that are eventually discontinued and whose demand
/// probability decays over time.  Instead of smoothing the inter-demand
/// interval, TSB directly smoothes the *demand probability* p_t.
#[derive(Debug, Clone)]
pub struct TsbModel {
    /// Smoothed demand probability (probability of a non-zero demand period).
    pub demand_prob: f64,
    /// Smoothed demand size.
    pub demand_size: f64,
    /// Smoothing parameter for demand probability.
    pub alpha: f64,
    /// Smoothing parameter for demand size.
    pub beta: f64,
}

impl TsbModel {
    /// Produce h-step-ahead forecasts.
    ///
    /// The TSB forecast is `demand_prob * demand_size`, constant over the
    /// horizon.
    pub fn forecast(&self, h: usize) -> Vec<f64> {
        if h == 0 {
            return Vec::new();
        }
        let f = self.demand_prob * self.demand_size;
        vec![f; h]
    }

    /// The point forecast value.
    pub fn point_forecast(&self) -> f64 {
        self.demand_prob * self.demand_size
    }
}

/// TSB (Teunter-Syntetos-Babai) intermittent demand method.
///
/// The update equations are:
/// - When demand occurs (y_t > 0):
///   `p_t = (1 - alpha) * p_{t-1} + alpha`
///   `z_t = (1 - beta) * z_{t-1} + beta * y_t`
/// - When no demand (y_t = 0):
///   `p_t = (1 - alpha) * p_{t-1}`
///   `z_t = z_{t-1}` (unchanged)
pub struct Tsb;

impl Tsb {
    /// Fit the TSB method.
    ///
    /// # Arguments
    /// * `demand_series` — Historical demand observations.
    /// * `alpha` — Smoothing parameter for demand probability in (0, 1].
    ///   Defaults to 0.1 if `None`.
    /// * `beta` — Smoothing parameter for demand size in (0, 1].
    ///   Defaults to `alpha` if `None`.
    ///
    /// # Returns
    /// A fitted [`TsbModel`].
    pub fn fit(demand_series: &[f64], alpha: Option<f64>, beta: Option<f64>) -> Result<TsbModel> {
        validate_demand_series(demand_series)?;
        let alpha = resolve_alpha(alpha)?;
        let beta = match beta {
            Some(b) => {
                if !(0.0 < b && b <= 1.0) {
                    return Err(TimeSeriesError::InvalidParameter {
                        name: "beta".to_string(),
                        message: "beta must be in (0, 1]".to_string(),
                    });
                }
                b
            }
            None => alpha,
        };

        // Initialise: use first non-zero value for size, and proportion of
        // non-zero obs in first third of the series (floor 1) for probability.
        let first_third = (demand_series.len() / 3).max(1);
        let first_nonzero = demand_series[..first_third]
            .iter()
            .find(|&&v| v > 0.0)
            .copied()
            .unwrap_or_else(|| {
                demand_series
                    .iter()
                    .copied()
                    .filter(|&v| v > 0.0)
                    .next()
                    .unwrap_or(1.0)
            });
        let init_prob = demand_series[..first_third]
            .iter()
            .filter(|&&v| v > 0.0)
            .count() as f64
            / first_third as f64;
        let init_prob = init_prob.max(0.01); // avoid zero probability

        let mut p = init_prob;
        let mut z = first_nonzero;

        for &y in demand_series {
            if y > 0.0 {
                p = (1.0 - alpha) * p + alpha;
                z = (1.0 - beta) * z + beta * y;
            } else {
                p = (1.0 - alpha) * p;
                // z unchanged
            }
        }

        Ok(TsbModel {
            demand_prob: p,
            demand_size: z,
            alpha,
            beta,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Validate that a demand series is non-negative and non-empty.
fn validate_demand_series(series: &[f64]) -> Result<()> {
    if series.is_empty() {
        return Err(TimeSeriesError::InsufficientData {
            message: "Demand series must not be empty".to_string(),
            required: 1,
            actual: 0,
        });
    }
    for &v in series {
        if v < 0.0 {
            return Err(TimeSeriesError::InvalidInput(
                "Demand series must be non-negative".to_string(),
            ));
        }
    }
    // Must have at least one non-zero observation.
    let has_nonzero = series.iter().any(|&v| v > 0.0);
    if !has_nonzero {
        return Err(TimeSeriesError::InvalidInput(
            "Demand series must contain at least one non-zero observation".to_string(),
        ));
    }
    Ok(())
}

/// Return alpha or the default (0.1).
fn resolve_alpha(alpha: Option<f64>) -> Result<f64> {
    match alpha {
        Some(a) => {
            if !(0.0 < a && a <= 1.0) {
                Err(TimeSeriesError::InvalidParameter {
                    name: "alpha".to_string(),
                    message: "alpha must be in (0, 1]".to_string(),
                })
            } else {
                Ok(a)
            }
        }
        None => Ok(0.1),
    }
}

/// Core Croston recursion shared by both Croston and SBA.
///
/// Returns the final `(z, p, q)` where z = smoothed demand size,
/// p = smoothed interval, q = periods since last non-zero demand.
fn croston_recursion(series: &[f64], alpha: f64, _correction: f64) -> Result<(f64, f64, f64)> {
    // Initialise z to the first non-zero demand, p to the index + 1 of
    // the first non-zero observation (inter-demand interval to that point).
    let (first_idx, first_demand) = series
        .iter()
        .copied()
        .enumerate()
        .find(|&(_, v)| v > 0.0)
        .ok_or_else(|| {
            TimeSeriesError::InvalidInput(
                "No non-zero demand found in series; initialisation impossible".to_string(),
            )
        })?;

    let mut z = first_demand;
    let mut p = (first_idx + 1) as f64;
    let mut q = 1_f64; // periods since last non-zero

    // Start recursion from the period *after* the first non-zero observation.
    for &y in &series[(first_idx + 1)..] {
        q += 1.0;
        if y > 0.0 {
            z = (1.0 - alpha) * z + alpha * y;
            p = (1.0 - alpha) * p + alpha * q;
            q = 0.0;
        }
        // When y = 0, Croston's method does NOT update z or p.
    }
    // q holds periods since last non-zero (useful for some variants).
    Ok((z, p, q))
}

// ─────────────────────────────────────────────────────────────────────────────
// Convenience: summary statistics
// ─────────────────────────────────────────────────────────────────────────────

/// Summary statistics for an intermittent demand series.
#[derive(Debug, Clone)]
pub struct DemandSummary {
    /// Total number of periods in the series.
    pub n_periods: usize,
    /// Number of periods with non-zero demand.
    pub n_demand_periods: usize,
    /// Average Demand Interval.
    pub adi: f64,
    /// Squared coefficient of variation of non-zero demands.
    pub cv2: f64,
    /// Demand type classification.
    pub demand_type: DemandType,
    /// Mean of non-zero demand values.
    pub mean_demand: f64,
    /// Total cumulative demand.
    pub total_demand: f64,
}

impl DemandSummary {
    /// Compute summary statistics for a demand series.
    pub fn compute(series: &[f64]) -> Result<Self> {
        if series.is_empty() {
            return Err(TimeSeriesError::InsufficientData {
                message: "Cannot compute summary of empty series".to_string(),
                required: 1,
                actual: 0,
            });
        }

        let non_zero: Vec<f64> = series.iter().copied().filter(|&v| v > 0.0).collect();
        let n_demand_periods = non_zero.len();
        let total_demand: f64 = non_zero.iter().copied().sum();
        let mean_demand = if n_demand_periods > 0 {
            total_demand / n_demand_periods as f64
        } else {
            0.0
        };

        Ok(Self {
            n_periods: series.len(),
            n_demand_periods,
            adi: compute_adi(series),
            cv2: compute_cv2(series),
            demand_type: classify_demand(series),
            mean_demand,
            total_demand,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Classification ───────────────────────────────────────────────────────

    #[test]
    fn test_classify_smooth() {
        // Frequent demand, small variation → Smooth
        let series = vec![3.0, 4.0, 3.5, 4.0, 3.8, 4.2, 3.9, 4.1];
        assert_eq!(classify_demand(&series), DemandType::Smooth);
    }

    #[test]
    fn test_classify_intermittent() {
        // Infrequent demand, low variation of non-zeros → Intermittent
        let series = vec![5.0, 0.0, 0.0, 5.0, 0.0, 0.0, 5.0, 0.0, 0.0, 5.0];
        let dt = classify_demand(&series);
        assert!(
            dt == DemandType::Intermittent || dt == DemandType::Smooth,
            "got {:?}",
            dt
        );
    }

    #[test]
    fn test_classify_lumpy() {
        // Infrequent AND highly variable → Lumpy
        // ADI = 10/3 ≈ 3.33 ≥ 1.32, CV² will be large with values 1, 100, 2
        let mut series = vec![0.0_f64; 10];
        series[0] = 1.0;
        series[4] = 100.0;
        series[8] = 2.0;
        let dt = classify_demand(&series);
        assert_eq!(dt, DemandType::Lumpy);
    }

    #[test]
    fn test_compute_adi_all_nonzero() {
        let series = vec![1.0, 2.0, 3.0, 4.0];
        assert!((compute_adi(&series) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_compute_adi_half_nonzero() {
        let series = vec![1.0, 0.0, 1.0, 0.0];
        assert!((compute_adi(&series) - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_compute_cv2_constant() {
        let series = vec![5.0, 0.0, 5.0, 0.0, 5.0];
        // Non-zeros are all 5.0 → CV² = 0
        assert!(compute_cv2(&series) < 1e-9);
    }

    // ── Croston ──────────────────────────────────────────────────────────────

    #[test]
    fn test_croston_fit_basic() {
        let demand = vec![0.0, 5.0, 0.0, 0.0, 3.0, 0.0, 7.0, 0.0, 4.0];
        let model = Croston::fit(&demand, Some(0.1)).expect("failed to create model");
        // z/p should be a positive number
        assert!(model.point_forecast() > 0.0);
    }

    #[test]
    fn test_croston_forecast_constant() {
        let demand = vec![0.0, 4.0, 0.0, 4.0, 0.0, 4.0];
        let model = Croston::fit(&demand, Some(0.2)).expect("failed to create model");
        let fc = model.forecast(5);
        assert_eq!(fc.len(), 5);
        // All equal (constant forecast).
        for &f in &fc {
            assert!((f - fc[0]).abs() < 1e-9);
        }
    }

    #[test]
    fn test_croston_invalid_alpha() {
        let demand = vec![0.0, 1.0, 0.0];
        assert!(Croston::fit(&demand, Some(1.5)).is_err());
        assert!(Croston::fit(&demand, Some(0.0)).is_err());
    }

    #[test]
    fn test_croston_all_zero_error() {
        let demand = vec![0.0, 0.0, 0.0];
        assert!(Croston::fit(&demand, None).is_err());
    }

    #[test]
    fn test_croston_negative_demand_error() {
        let demand = vec![1.0, -1.0, 0.0];
        assert!(Croston::fit(&demand, None).is_err());
    }

    // ── SBA ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_sba_correction() {
        // SBA forecast should be (1 - alpha/2) × Croston forecast
        let demand = vec![0.0, 5.0, 0.0, 5.0, 0.0, 5.0, 0.0, 5.0];
        let alpha = 0.2;
        let croston = Croston::fit(&demand, Some(alpha)).expect("failed to create croston");
        let sba = Sba::fit(&demand, Some(alpha)).expect("failed to create sba");

        let expected = (1.0 - alpha / 2.0) * croston.point_forecast();
        assert!(
            (sba.point_forecast() - expected).abs() < 1e-9,
            "SBA = {}, expected = {}",
            sba.point_forecast(),
            expected
        );
    }

    #[test]
    fn test_sba_lower_than_croston() {
        // SBA should (by design) produce lower forecasts than Croston.
        let demand = vec![0.0, 3.0, 0.0, 0.0, 6.0, 0.0, 4.0];
        let alpha = 0.15;
        let c = Croston::fit(&demand, Some(alpha))
            .expect("failed to create c")
            .point_forecast();
        let s = Sba::fit(&demand, Some(alpha))
            .expect("failed to create s")
            .point_forecast();
        assert!(s <= c + 1e-9, "SBA ({s}) should be ≤ Croston ({c})");
    }

    // ── TSB ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_tsb_fit_basic() {
        let demand = vec![0.0, 5.0, 0.0, 0.0, 3.0, 0.0, 7.0, 0.0, 4.0];
        let model = Tsb::fit(&demand, Some(0.1), None).expect("failed to create model");
        assert!(model.demand_prob > 0.0);
        assert!(model.demand_prob <= 1.0);
        assert!(model.demand_size > 0.0);
        assert!(model.point_forecast() > 0.0);
    }

    #[test]
    fn test_tsb_probability_decays_on_zeros() {
        // A long run of zeros should drive probability toward 0.
        let mut demand = vec![5.0];
        demand.extend(vec![0.0_f64; 50]);
        let model = Tsb::fit(&demand, Some(0.3), Some(0.2)).expect("failed to create model");
        assert!(
            model.demand_prob < 0.1,
            "Probability should decay after long zero run, got {}",
            model.demand_prob
        );
    }

    #[test]
    fn test_tsb_forecast_length() {
        let demand = vec![0.0, 2.0, 0.0, 3.0, 0.0];
        let model = Tsb::fit(&demand, None, None).expect("failed to create model");
        let fc = model.forecast(10);
        assert_eq!(fc.len(), 10);
    }

    // ── DemandSummary ────────────────────────────────────────────────────────

    #[test]
    fn test_demand_summary() {
        let demand = vec![0.0, 5.0, 0.0, 0.0, 10.0, 0.0, 5.0];
        let summary = DemandSummary::compute(&demand).expect("failed to create summary");
        assert_eq!(summary.n_periods, 7);
        assert_eq!(summary.n_demand_periods, 3);
        assert!((summary.total_demand - 20.0).abs() < 1e-9);
        assert!((summary.mean_demand - 20.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_demand_summary_empty() {
        assert!(DemandSummary::compute(&[]).is_err());
    }
}
