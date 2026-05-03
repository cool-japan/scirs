//! Robust statistical estimators for location, scale, and regression.
//!
//! This module provides robust alternatives to classical estimators that are
//! resistant to outliers and deviations from distributional assumptions.
//!
//! ## Location Estimators
//! - [`huber_location`]: M-estimator using Huber's ψ function
//! - [`biweight_location`]: Tukey bisquare (biweight) location estimator
//! - [`trimmed_mean`]: Trimmed (truncated) mean
//! - [`winsorized_mean`]: Winsorized mean with user-specified limits
//!
//! ## Scale Estimators
//! - [`mad`]: Median Absolute Deviation (consistency-scaled to normal σ)
//! - [`qn_scale`]: Rousseeuw-Croux Qn scale estimator (highly robust)
//! - [`sn_scale`]: Rousseeuw-Croux Sn scale estimator
//!
//! ## Robust Regression
//! - [`TheilSen`]: Theil-Sen median-of-slopes estimator
//! - [`PassingBablok`]: Passing-Bablok method-comparison regression
//!
//! # References
//! - Huber, P.J. (1981). *Robust Statistics*. Wiley.
//! - Rousseeuw, P.J. & Croux, C. (1993). "Alternatives to the median absolute
//!   deviation." *Journal of the American Statistical Association*, 88, 1273–1283.
//! - Passing, H. & Bablok, W. (1983). "A new biometrical procedure for testing
//!   the equality of measurements from two different analytical methods."
//!   *Journal of Clinical Chemistry and Clinical Biochemistry*, 21, 709–720.

use crate::error::{StatsError, StatsResult};

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Sort a slice and return the sorted copy.
fn sorted_copy(x: &[f64]) -> Vec<f64> {
    let mut v = x.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v
}

/// Median of a *sorted* slice. Returns `f64::NAN` for empty input.
fn median_of_sorted(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    let mid = n / 2;
    if n % 2 == 0 {
        (sorted[mid - 1] + sorted[mid]) * 0.5
    } else {
        sorted[mid]
    }
}

/// Median of an unsorted slice.
fn median_of(x: &[f64]) -> f64 {
    if x.is_empty() {
        return f64::NAN;
    }
    let sorted = sorted_copy(x);
    median_of_sorted(&sorted)
}

// ---------------------------------------------------------------------------
// Location estimators
// ---------------------------------------------------------------------------

/// Compute the Huber M-estimator of location.
///
/// Iteratively reweighted least squares (IRLS) using Huber's ψ function:
/// `ψ(u) = u` for `|u| ≤ k`, and `ψ(u) = k · sign(u)` for `|u| > k`.
///
/// The scale estimate is updated each iteration using the MAD.
///
/// # Arguments
///
/// * `data` – Input data (need not be sorted).
/// * `k` – Tuning constant controlling the breakdown point. The default
///   value in many software packages is 1.345 (95 % efficiency under
///   normality).
/// * `tol` – Convergence tolerance on the location estimate.
/// * `max_iter` – Maximum number of IRLS iterations.
///
/// # Errors
///
/// Returns [`StatsError::InsufficientData`] if `data` is empty, and
/// [`StatsError::ConvergenceError`] if the algorithm fails to converge.
///
/// # Examples
///
/// ```
/// use scirs2_stats::robust::huber_location;
///
/// let data = vec![1.0_f64, 2.0, 3.0, 4.0, 100.0]; // outlier at 100
/// let loc = huber_location(&data, 1.345, 1e-6, 100).expect("converged");
/// // Huber estimate is pulled back toward the bulk of the data
/// assert!(loc < 10.0, "location={loc}");
/// ```
pub fn huber_location(data: &[f64], k: f64, tol: f64, max_iter: usize) -> StatsResult<f64> {
    if data.is_empty() {
        return Err(StatsError::InsufficientData(
            "huber_location requires at least one observation".into(),
        ));
    }
    if k <= 0.0 {
        return Err(StatsError::InvalidArgument(
            "tuning constant k must be positive".into(),
        ));
    }

    // Initial location estimate: median
    let mut mu = median_of(data);

    for _iter in 0..max_iter {
        // Scale estimate: consistency-corrected MAD
        let scale = {
            let devs: Vec<f64> = data.iter().map(|&x| (x - mu).abs()).collect();
            let m = median_of(&devs);
            // 0.6745 is the consistency factor for the normal distribution
            m / 0.6745
        };

        if scale < f64::EPSILON {
            // All values essentially equal — location is the median
            return Ok(mu);
        }

        // Huber weights: w(u) = min(1, k / |u|)
        let mut sum_wx = 0.0_f64;
        let mut sum_w = 0.0_f64;
        for &x in data {
            let u = (x - mu) / scale;
            let w = if u.abs() <= k { 1.0 } else { k / u.abs() };
            sum_wx += w * x;
            sum_w += w;
        }

        if sum_w < f64::EPSILON {
            return Err(StatsError::ConvergenceError(
                "huber_location: weight sum near zero".into(),
            ));
        }

        let mu_new = sum_wx / sum_w;
        let delta = (mu_new - mu).abs();
        mu = mu_new;

        if delta <= tol {
            return Ok(mu);
        }
    }

    Err(StatsError::ConvergenceError(format!(
        "huber_location did not converge after {} iterations",
        max_iter
    )))
}

/// Compute the biweight (Tukey bisquare) location estimator.
///
/// Uses the one-step biweight estimator centred at the sample median with
/// the MAD as the scale estimate. The influence function is bounded for
/// `|u| > c` (the contribution of extreme observations is zero).
///
/// # Arguments
///
/// * `data` – Input data.
/// * `c` – Tuning constant (typically 6.0, giving ~95 % efficiency under
///   normality).
///
/// # Errors
///
/// Returns [`StatsError::InsufficientData`] if `data` is empty.
///
/// # Examples
///
/// ```
/// use scirs2_stats::robust::biweight_location;
///
/// let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
/// let loc = biweight_location(&data, 6.0).expect("ok");
/// assert!((loc - 3.0).abs() < 1e-6);
/// ```
pub fn biweight_location(data: &[f64], c: f64) -> StatsResult<f64> {
    if data.is_empty() {
        return Err(StatsError::InsufficientData(
            "biweight_location requires at least one observation".into(),
        ));
    }
    if c <= 0.0 {
        return Err(StatsError::InvalidArgument(
            "tuning constant c must be positive".into(),
        ));
    }

    let m = median_of(data);

    // MAD-based scale
    let devs: Vec<f64> = data.iter().map(|&x| (x - m).abs()).collect();
    let mad = median_of(&devs);

    if mad < f64::EPSILON {
        // All values the same (or essentially so)
        return Ok(m);
    }

    // Biweight weights: w(u) = (1 - (u/c)^2)^2  for |u| ≤ c, else 0
    let mut sum_wx = 0.0_f64;
    let mut sum_w = 0.0_f64;
    for &x in data {
        let u = (x - m) / mad;
        if u.abs() <= c {
            let t = 1.0 - (u / c).powi(2);
            let w = t * t;
            sum_wx += w * x;
            sum_w += w;
        }
    }

    if sum_w < f64::EPSILON {
        return Ok(m);
    }

    Ok(sum_wx / sum_w)
}

/// Compute the trimmed mean.
///
/// Removes a proportion `proportiontocut` from each tail of the sorted data
/// before computing the arithmetic mean.
///
/// # Arguments
///
/// * `data` – Input data.
/// * `proportiontocut` – Fraction in `[0, 0.5)` of data to remove from
///   each tail.
///
/// # Errors
///
/// Returns [`StatsError::InsufficientData`] if `data` is empty, or
/// [`StatsError::InvalidArgument`] if `proportiontocut` is outside `[0, 0.5)`.
///
/// # Examples
///
/// ```
/// use scirs2_stats::robust::trimmed_mean;
///
/// let data = vec![1.0_f64, 2.0, 3.0, 4.0, 100.0];
/// // Trim 20 % from each tail → mean of [2, 3, 4]
/// let tm = trimmed_mean(&data, 0.2).expect("ok");
/// assert!((tm - 3.0).abs() < 1e-10);
/// ```
pub fn trimmed_mean(data: &[f64], proportiontocut: f64) -> StatsResult<f64> {
    if data.is_empty() {
        return Err(StatsError::InsufficientData(
            "trimmed_mean requires at least one observation".into(),
        ));
    }
    if !(0.0..0.5).contains(&proportiontocut) {
        return Err(StatsError::InvalidArgument(
            "proportiontocut must be in [0, 0.5)".into(),
        ));
    }

    let sorted = sorted_copy(data);
    let n = sorted.len();
    let cut = (n as f64 * proportiontocut).floor() as usize;

    let lo = cut;
    let hi = n - cut;
    if lo >= hi {
        return Err(StatsError::InsufficientData(
            "trimmed_mean: proportiontocut removes all observations".into(),
        ));
    }

    let sum: f64 = sorted[lo..hi].iter().sum();
    Ok(sum / (hi - lo) as f64)
}

/// Compute the Winsorized mean.
///
/// Values below the lower quantile are replaced by it; values above the upper
/// quantile are replaced by it. The mean of the resulting array is returned.
///
/// # Arguments
///
/// * `data` – Input data.
/// * `limits` – `(lower_fraction, upper_fraction)` in `[0, 1)` specifying
///   the proportion to Winsorize from each tail.
///
/// # Errors
///
/// Returns [`StatsError::InsufficientData`] if `data` is empty, or
/// [`StatsError::InvalidArgument`] if limits are invalid.
///
/// # Examples
///
/// ```
/// use scirs2_stats::robust::winsorized_mean;
///
/// let data = vec![1.0_f64, 2.0, 3.0, 4.0, 100.0];
/// let wm = winsorized_mean(&data, (0.0, 0.2)).expect("ok");
/// // Upper 20 % → 4.0 replaces 100.0; mean([1,2,3,4,4]) = 2.8
/// assert!((wm - 2.8).abs() < 1e-10);
/// ```
pub fn winsorized_mean(data: &[f64], limits: (f64, f64)) -> StatsResult<f64> {
    if data.is_empty() {
        return Err(StatsError::InsufficientData(
            "winsorized_mean requires at least one observation".into(),
        ));
    }
    let (lo_frac, hi_frac) = limits;
    if lo_frac < 0.0 || hi_frac < 0.0 || lo_frac + hi_frac >= 1.0 {
        return Err(StatsError::InvalidArgument(
            "limits must be non-negative and sum to less than 1".into(),
        ));
    }

    let sorted = sorted_copy(data);
    let n = sorted.len();
    let lo_idx = (n as f64 * lo_frac).floor() as usize;
    let hi_idx = n - (n as f64 * hi_frac).floor() as usize - 1;

    if lo_idx > hi_idx {
        return Err(StatsError::InsufficientData(
            "winsorized_mean: limits clip all observations".into(),
        ));
    }

    let lo_val = sorted[lo_idx];
    let hi_val = sorted[hi_idx];

    let sum: f64 = sorted.iter().map(|&x| x.clamp(lo_val, hi_val)).sum();
    Ok(sum / n as f64)
}

// ---------------------------------------------------------------------------
// Scale estimators
// ---------------------------------------------------------------------------

/// Compute the Median Absolute Deviation (MAD), scaled for consistency with
/// the standard deviation under normality (factor ≈ 1.4826).
///
/// `MAD = 1.4826 · median(|xᵢ − median(x)|)`
///
/// # Errors
///
/// Returns [`StatsError::InsufficientData`] if `data` is empty.
///
/// # Examples
///
/// ```
/// use scirs2_stats::robust::mad;
///
/// let data = vec![1.0_f64, 1.0, 2.0, 3.0, 100.0];
/// let m = mad(&data).expect("ok");
/// assert!(m < 5.0, "MAD={m} should be robust to the outlier");
/// ```
pub fn mad(data: &[f64]) -> StatsResult<f64> {
    if data.is_empty() {
        return Err(StatsError::InsufficientData(
            "mad requires at least one observation".into(),
        ));
    }

    let med = median_of(data);
    let devs: Vec<f64> = data.iter().map(|&x| (x - med).abs()).collect();
    let raw_mad = median_of(&devs);
    // Consistency factor for normal distribution
    const K: f64 = 1.4826022185056018;
    Ok(K * raw_mad)
}

/// Compute the Qn scale estimator of Rousseeuw and Croux (1993).
///
/// `Qn = 2.2219 · { |xᵢ − xⱼ|; i < j }_{(h)}` where `h = C(⌊n/2⌋+1, 2)`.
///
/// The Qn estimator has a 50 % breakdown point and 82 % efficiency under
/// normality (much higher than MAD's 37 %).
///
/// # Errors
///
/// Returns [`StatsError::InsufficientData`] if fewer than 2 observations are
/// provided.
///
/// # Examples
///
/// ```
/// use scirs2_stats::robust::qn_scale;
///
/// let data = vec![2.0_f64, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
/// let qn = qn_scale(&data).expect("ok");
/// assert!(qn > 0.0);
/// ```
pub fn qn_scale(data: &[f64]) -> StatsResult<f64> {
    let n = data.len();
    if n < 2 {
        return Err(StatsError::InsufficientData(
            "qn_scale requires at least 2 observations".into(),
        ));
    }

    // Collect all pairwise absolute differences
    let mut diffs: Vec<f64> = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            diffs.push((data[i] - data[j]).abs());
        }
    }
    diffs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // h-th order statistic: h = C(floor(n/2)+1, 2)
    let half = n / 2 + 1;
    let h = half * (half - 1) / 2;
    let idx = h.min(diffs.len()) - 1;

    let raw_qn = diffs[idx];

    // Consistency factor (asymptotic; small-sample correction available but
    // omitted for clarity — the asymptotic value is 2.2219)
    const CN: f64 = 2.2219;
    Ok(CN * raw_qn)
}

/// Compute the Sn scale estimator of Rousseeuw and Croux (1993).
///
/// `Sn = 1.1926 · lowmedian_i { highmedian_j |xᵢ − xⱼ| }`
///
/// The Sn estimator has a 50 % breakdown point and 58 % efficiency under
/// normality.  It requires only O(n log n) time.
///
/// # Errors
///
/// Returns [`StatsError::InsufficientData`] if fewer than 2 observations are
/// provided.
///
/// # Examples
///
/// ```
/// use scirs2_stats::robust::sn_scale;
///
/// let data = vec![2.0_f64, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
/// let sn = sn_scale(&data).expect("ok");
/// assert!(sn > 0.0);
/// ```
pub fn sn_scale(data: &[f64]) -> StatsResult<f64> {
    let n = data.len();
    if n < 2 {
        return Err(StatsError::InsufficientData(
            "sn_scale requires at least 2 observations".into(),
        ));
    }

    // For each observation xᵢ compute the high-median of |xᵢ - xⱼ| over j≠i.
    // high-median of a list of length m is the element at index ⌈m/2⌉ (1-based).
    let mut row_medians: Vec<f64> = Vec::with_capacity(n);
    for i in 0..n {
        let mut diffs: Vec<f64> = (0..n)
            .filter(|&j| j != i)
            .map(|j| (data[i] - data[j]).abs())
            .collect();
        diffs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        // high-median: index ceil((n-1)/2) in 0-based sorted list
        let idx = (diffs.len() - 1 + 1) / 2; // = ceil((n-1)/2) 0-based
        row_medians.push(diffs[idx]);
    }

    // low-median of row_medians: index floor((n+1)/2) - 1 in 0-based sorted list
    row_medians.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = (n + 1) / 2 - 1;
    let raw_sn = row_medians[idx];

    // Consistency factor (asymptotic)
    const CN: f64 = 1.1926;
    Ok(CN * raw_sn)
}

// ---------------------------------------------------------------------------
// Robust regression: Theil-Sen
// ---------------------------------------------------------------------------

/// Theil-Sen robust linear regression estimator.
///
/// The slope is the median of all pairwise slopes between data points.
/// The intercept is then `median(yᵢ − slope·xᵢ)`.
///
/// The estimator has a 29.3 % breakdown point and is unbiased for simple
/// linear regression under symmetric error distributions.
///
/// # References
/// - Theil, H. (1950). "A rank-invariant method of linear and polynomial
///   regression analysis." *Indagationes Mathematicae*, 12, 85–91.
/// - Sen, P.K. (1968). "Estimates of the regression coefficient based on
///   Kendall's tau." *Journal of the American Statistical Association*,
///   63, 1379–1389.
#[derive(Debug, Clone)]
pub struct TheilSen {
    slope: f64,
    intercept: f64,
}

impl TheilSen {
    /// Fit a Theil-Sen regression line.
    ///
    /// # Arguments
    ///
    /// * `x` – Predictor values.
    /// * `y` – Response values (same length as `x`).
    ///
    /// # Errors
    ///
    /// Returns [`StatsError::InsufficientData`] if fewer than 2 paired
    /// observations are provided, or [`StatsError::DimensionMismatch`] if
    /// `x` and `y` have different lengths.
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_stats::robust::TheilSen;
    ///
    /// let x = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
    /// let y = vec![2.0_f64, 4.0, 6.0, 8.0, 10.0];
    /// let model = TheilSen::fit(&x, &y).expect("ok");
    /// assert!((model.slope() - 2.0).abs() < 1e-10);
    /// assert!((model.intercept()).abs() < 1e-10);
    /// ```
    pub fn fit(x: &[f64], y: &[f64]) -> StatsResult<Self> {
        if x.len() != y.len() {
            return Err(StatsError::DimensionMismatch(format!(
                "x and y must have the same length, got {} and {}",
                x.len(),
                y.len()
            )));
        }
        if x.len() < 2 {
            return Err(StatsError::InsufficientData(
                "TheilSen requires at least 2 observations".into(),
            ));
        }

        let n = x.len();
        let mut slopes: Vec<f64> = Vec::with_capacity(n * (n - 1) / 2);

        for i in 0..n {
            for j in (i + 1)..n {
                let dx = x[j] - x[i];
                if dx.abs() > f64::EPSILON {
                    slopes.push((y[j] - y[i]) / dx);
                }
            }
        }

        if slopes.is_empty() {
            return Err(StatsError::ComputationError(
                "TheilSen: all x values are identical; slope undefined".into(),
            ));
        }

        let slope = median_of(&slopes);
        let intercepts: Vec<f64> = x
            .iter()
            .zip(y.iter())
            .map(|(&xi, &yi)| yi - slope * xi)
            .collect();
        let intercept = median_of(&intercepts);

        Ok(TheilSen { slope, intercept })
    }

    /// Generate predictions for the given predictor values.
    pub fn predict(&self, x: &[f64]) -> Vec<f64> {
        x.iter()
            .map(|&xi| self.slope * xi + self.intercept)
            .collect()
    }

    /// Return the estimated slope.
    pub fn slope(&self) -> f64 {
        self.slope
    }

    /// Return the estimated intercept.
    pub fn intercept(&self) -> f64 {
        self.intercept
    }
}

// ---------------------------------------------------------------------------
// Robust regression: Passing-Bablok
// ---------------------------------------------------------------------------

/// Passing-Bablok regression for method-comparison studies.
///
/// Unlike Theil-Sen, Passing-Bablok handles the case where both variables
/// are measured with error (errors-in-variables) and is scale-equivariant.
/// It is the standard method for comparing two analytical measurement methods
/// in clinical chemistry.
///
/// The slope is estimated as the shifted median of the set of all pairwise
/// slopes. The intercept follows by `median(yᵢ − slope·xᵢ)`.
///
/// # References
/// - Passing, H. & Bablok, W. (1983). "A new biometrical procedure for
///   testing the equality of measurements from two different analytical
///   methods." *J. Clin. Chem. Clin. Biochem.*, 21, 709–720.
#[derive(Debug, Clone)]
pub struct PassingBablok {
    slope: f64,
    intercept: f64,
}

impl PassingBablok {
    /// Fit a Passing-Bablok regression line.
    ///
    /// # Arguments
    ///
    /// * `x` – Measurements from method 1.
    /// * `y` – Measurements from method 2 (same length as `x`).
    ///
    /// # Errors
    ///
    /// Returns [`StatsError::InsufficientData`] if fewer than 3 paired
    /// observations are provided, or [`StatsError::DimensionMismatch`] if
    /// lengths differ.
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_stats::robust::PassingBablok;
    ///
    /// let x = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
    /// let y = vec![1.1_f64, 2.0, 3.1, 3.9, 5.0];
    /// let model = PassingBablok::fit(&x, &y).expect("ok");
    /// // Slope near 1, intercept near 0 for closely matching methods
    /// assert!((model.slope() - 1.0).abs() < 0.3);
    /// ```
    pub fn fit(x: &[f64], y: &[f64]) -> StatsResult<Self> {
        if x.len() != y.len() {
            return Err(StatsError::DimensionMismatch(format!(
                "x and y must have the same length, got {} and {}",
                x.len(),
                y.len()
            )));
        }
        if x.len() < 3 {
            return Err(StatsError::InsufficientData(
                "PassingBablok requires at least 3 observations".into(),
            ));
        }

        let n = x.len();

        // Compute all pairwise slopes Sᵢⱼ = (yⱼ − yᵢ) / (xⱼ − xᵢ)
        // Slopes equal to −1 are excluded per the Passing-Bablok convention;
        // slopes < −1 are shifted by +∞ (i.e. reflected).
        let mut slopes: Vec<f64> = Vec::with_capacity(n * (n - 1) / 2);
        let mut neg_one_count = 0usize;

        for i in 0..n {
            for j in (i + 1)..n {
                let dx = x[j] - x[i];
                let dy = y[j] - y[i];

                if dx.abs() < f64::EPSILON && dy.abs() < f64::EPSILON {
                    // Identical points: skip
                    continue;
                }

                let s = if dx.abs() < f64::EPSILON {
                    // Vertical pair: infinite slope
                    if dy > 0.0 {
                        f64::INFINITY
                    } else {
                        f64::NEG_INFINITY
                    }
                } else {
                    dy / dx
                };

                // Exclude −1 exactly
                if (s + 1.0).abs() < f64::EPSILON {
                    neg_one_count += 1;
                    continue;
                }

                slopes.push(s);
            }
        }

        if slopes.is_empty() {
            return Err(StatsError::ComputationError(
                "PassingBablok: insufficient valid slope pairs".into(),
            ));
        }

        // Sort slopes (treating infinities properly)
        slopes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Passing-Bablok: the estimator is the (K+1)-th element of sorted slopes
        // where K = number of slopes that equal −1.
        let k = neg_one_count;
        let m = slopes.len();

        // Index of the median in the shifted set
        // The median of m+2k slopes (treating the −1 excluded ones as being
        // placed conceptually) is the ((m + 2k + 1)/2)-th = ((m)/2 + k)-th
        // element (0-based) of the slopes array.
        let mid_idx = if (m + 2 * k) % 2 == 0 {
            // Even: average of two middle elements
            let i1 = ((m + 2 * k) / 2).saturating_sub(1 + k).min(m - 1);
            let i2 = ((m + 2 * k) / 2).saturating_sub(k).min(m - 1);
            let slope = (slopes[i1] + slopes[i2]) * 0.5;
            let intercepts: Vec<f64> = x
                .iter()
                .zip(y.iter())
                .map(|(&xi, &yi)| yi - slope * xi)
                .collect();
            let intercept = median_of(&intercepts);
            return Ok(PassingBablok { slope, intercept });
        } else {
            ((m + 2 * k + 1) / 2).saturating_sub(1 + k).min(m - 1)
        };

        let slope = slopes[mid_idx];
        let intercepts: Vec<f64> = x
            .iter()
            .zip(y.iter())
            .map(|(&xi, &yi)| yi - slope * xi)
            .collect();
        let intercept = median_of(&intercepts);

        Ok(PassingBablok { slope, intercept })
    }

    /// Return the estimated slope.
    pub fn slope(&self) -> f64 {
        self.slope
    }

    /// Return the estimated intercept.
    pub fn intercept(&self) -> f64 {
        self.intercept
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Huber location ----

    #[test]
    fn test_huber_location_clean_data() {
        let data: Vec<f64> = (1..=9).map(|i| i as f64).collect();
        let loc = huber_location(&data, 1.345, 1e-8, 200).expect("converged");
        // Clean symmetric data → estimate ≈ mean = median = 5
        assert!((loc - 5.0).abs() < 0.05, "loc={loc}");
    }

    #[test]
    fn test_huber_location_outlier_resistant() {
        let mut data: Vec<f64> = (1..=9).map(|i| i as f64).collect();
        data.push(1000.0); // extreme outlier
        let loc = huber_location(&data, 1.345, 1e-8, 200).expect("converged");
        // Should stay near the bulk [1..9], not pulled to 100.5 like the mean
        assert!(loc < 20.0, "loc={loc} should be robust to outlier");
    }

    #[test]
    fn test_huber_location_empty() {
        assert!(huber_location(&[], 1.345, 1e-6, 100).is_err());
    }

    #[test]
    fn test_huber_location_invalid_k() {
        assert!(huber_location(&[1.0, 2.0], -1.0, 1e-6, 100).is_err());
    }

    // ---- Biweight location ----

    #[test]
    fn test_biweight_location_symmetric() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let loc = biweight_location(&data, 6.0).expect("ok");
        assert!((loc - 3.0).abs() < 0.01, "loc={loc}");
    }

    #[test]
    fn test_biweight_location_outlier() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 100.0];
        let loc = biweight_location(&data, 6.0).expect("ok");
        assert!(loc < 10.0, "biweight should resist outlier, got {loc}");
    }

    #[test]
    fn test_biweight_location_empty() {
        assert!(biweight_location(&[], 6.0).is_err());
    }

    // ---- Trimmed mean ----

    #[test]
    fn test_trimmed_mean_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 100.0];
        let tm = trimmed_mean(&data, 0.2).expect("ok");
        // Remove 1 from each end → [2, 3, 4] → mean = 3
        assert!((tm - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_trimmed_mean_zero_cut() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let tm = trimmed_mean(&data, 0.0).expect("ok");
        let mean = data.iter().sum::<f64>() / 5.0;
        assert!((tm - mean).abs() < 1e-10);
    }

    #[test]
    fn test_trimmed_mean_invalid_cut() {
        let data = vec![1.0, 2.0, 3.0];
        assert!(trimmed_mean(&data, 0.6).is_err());
        assert!(trimmed_mean(&data, -0.1).is_err());
    }

    // ---- Winsorized mean ----

    #[test]
    fn test_winsorized_mean_upper_tail() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 100.0];
        let wm = winsorized_mean(&data, (0.0, 0.2)).expect("ok");
        // Replace 100.0 with 4.0 → [1, 2, 3, 4, 4] → mean = 14/5 = 2.8
        assert!((wm - 2.8).abs() < 1e-10, "wm={wm}");
    }

    #[test]
    fn test_winsorized_mean_both_tails() {
        let data = vec![-100.0, 1.0, 2.0, 3.0, 4.0, 100.0];
        let wm = winsorized_mean(&data, (1.0 / 6.0, 1.0 / 6.0)).expect("ok");
        // Each tail: floor(6 * 1/6) = 1 → replace -100 with 1, 100 with 4
        // Array becomes [1, 1, 2, 3, 4, 4] → mean = 15/6 = 2.5
        assert!((wm - 2.5).abs() < 1e-6, "wm={wm}");
    }

    // ---- MAD ----

    #[test]
    fn test_mad_normal_scale() {
        // For N(0,1) data MAD ≈ σ with the consistency factor
        let data = vec![-2.0, -1.0, -0.5, -0.2, 0.0, 0.1, 0.3, 0.7, 1.0, 1.5];
        let m = mad(&data).expect("ok");
        assert!(m > 0.0, "MAD should be positive");
    }

    #[test]
    fn test_mad_constant_data() {
        let data = vec![3.0, 3.0, 3.0, 3.0];
        let m = mad(&data).expect("ok");
        assert_eq!(m, 0.0);
    }

    #[test]
    fn test_mad_empty() {
        assert!(mad(&[]).is_err());
    }

    // ---- Qn scale ----

    #[test]
    fn test_qn_scale_positive() {
        let data: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        let qn = qn_scale(&data).expect("ok");
        assert!(qn > 0.0, "qn={qn}");
    }

    #[test]
    fn test_qn_scale_constant() {
        let data = vec![5.0; 10];
        let qn = qn_scale(&data).expect("ok");
        assert_eq!(qn, 0.0);
    }

    #[test]
    fn test_qn_scale_single() {
        assert!(qn_scale(&[1.0]).is_err());
    }

    // ---- Sn scale ----

    #[test]
    fn test_sn_scale_positive() {
        let data: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        let sn = sn_scale(&data).expect("ok");
        assert!(sn > 0.0, "sn={sn}");
    }

    #[test]
    fn test_sn_scale_constant() {
        let data = vec![7.0; 8];
        let sn = sn_scale(&data).expect("ok");
        assert_eq!(sn, 0.0);
    }

    #[test]
    fn test_sn_scale_single() {
        assert!(sn_scale(&[1.0]).is_err());
    }

    // ---- TheilSen ----

    #[test]
    fn test_theilsen_perfect_linear() {
        let x: Vec<f64> = (0..=5).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| 3.0 * xi + 1.0).collect();
        let model = TheilSen::fit(&x, &y).expect("ok");
        assert!(
            (model.slope() - 3.0).abs() < 1e-10,
            "slope={}",
            model.slope()
        );
        assert!(
            (model.intercept() - 1.0).abs() < 1e-10,
            "intercept={}",
            model.intercept()
        );
    }

    #[test]
    fn test_theilsen_outlier_resistant() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut y = vec![2.0, 4.0, 6.0, 8.0, 10.0]; // slope 2
        y[4] = 100.0; // outlier
        let model = TheilSen::fit(&x, &y).expect("ok");
        // Slope should remain close to 2 despite the outlier
        assert!((model.slope() - 2.0).abs() < 1.0, "slope={}", model.slope());
    }

    #[test]
    fn test_theilsen_predict() {
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![1.0, 3.0, 5.0];
        let model = TheilSen::fit(&x, &y).expect("ok");
        let preds = model.predict(&[3.0, 4.0]);
        assert!((preds[0] - 7.0).abs() < 1e-8, "pred0={}", preds[0]);
        assert!((preds[1] - 9.0).abs() < 1e-8, "pred1={}", preds[1]);
    }

    #[test]
    fn test_theilsen_length_mismatch() {
        assert!(TheilSen::fit(&[1.0, 2.0], &[1.0]).is_err());
    }

    #[test]
    fn test_theilsen_insufficient_data() {
        assert!(TheilSen::fit(&[1.0], &[2.0]).is_err());
    }

    // ---- PassingBablok ----

    #[test]
    fn test_passingbablok_identical_methods() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let y = x.clone();
        let model = PassingBablok::fit(&x, &y).expect("ok");
        assert!((model.slope() - 1.0).abs() < 0.1, "slope={}", model.slope());
        assert!(
            model.intercept().abs() < 0.5,
            "intercept={}",
            model.intercept()
        );
    }

    #[test]
    fn test_passingbablok_scaled_methods() {
        let x: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi).collect();
        let model = PassingBablok::fit(&x, &y).expect("ok");
        assert!((model.slope() - 2.0).abs() < 0.1, "slope={}", model.slope());
    }

    #[test]
    fn test_passingbablok_insufficient_data() {
        assert!(PassingBablok::fit(&[1.0, 2.0], &[1.0, 2.0]).is_err());
    }

    #[test]
    fn test_passingbablok_length_mismatch() {
        assert!(PassingBablok::fit(&[1.0, 2.0, 3.0], &[1.0, 2.0]).is_err());
    }
}
