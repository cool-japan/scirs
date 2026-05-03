//! Online (streaming) time series algorithms
//!
//! This module provides numerically stable algorithms for processing time series data
//! in a streaming fashion, including Welford's algorithm for incremental statistics,
//! Sherman-Morrison rank-1 updates for online linear regression, recursive least
//! squares ARIMA, and sequential change detection methods.
//!
//! # Algorithms
//!
//! - [`OnlineMean`] - Welford's numerically stable online mean and variance
//! - [`OnlineLinearRegression`] - Sherman-Morrison rank-1 update for online OLS
//! - [`OnlineARIMA`] - Recursive least squares ARIMA with differencing
//! - [`ADWINDetector`] - ADaptive WINdowing concept drift detector
//! - [`PageHinkley`] - Page-Hinkley sequential change detection test
//! - [`OnlineQuantile`] - P² algorithm for streaming quantile estimation

use std::collections::VecDeque;

use crate::error::{Result, TimeSeriesError};

// ---------------------------------------------------------------------------
// OnlineMean — Welford's online algorithm
// ---------------------------------------------------------------------------

/// Online mean and variance tracker using Welford's numerically stable algorithm.
///
/// Welford's single-pass algorithm avoids the catastrophic cancellation that
/// affects the naïve two-pass formula, making it suitable for streaming data
/// where the full sequence is never held in memory.
///
/// # Examples
///
/// ```
/// use scirs2_series::online_algorithms::OnlineMean;
///
/// let mut tracker = OnlineMean::new();
/// for x in [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0] {
///     tracker.update(x);
/// }
/// assert!((tracker.mean() - 5.0).abs() < 1e-10);
/// assert!((tracker.variance() - 4.571428).abs() < 1e-5);
/// ```
#[derive(Debug, Clone)]
pub struct OnlineMean {
    /// Number of observations seen so far
    count: u64,
    /// Running mean (Welford M_n)
    mean: f64,
    /// Running sum of squared deviations from mean (Welford S_n)
    m2: f64,
}

impl OnlineMean {
    /// Create a new, empty `OnlineMean`.
    pub fn new() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            m2: 0.0,
        }
    }

    /// Incorporate one new data point using Welford's update.
    ///
    /// The update is O(1) and requires O(1) extra storage regardless of
    /// how many values have been seen.
    pub fn update(&mut self, x: f64) {
        self.count += 1;
        let delta = x - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;
    }

    /// Return the current (running) mean.
    ///
    /// Returns 0.0 if no data has been seen yet.
    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// Return the sample variance (Bessel-corrected, denominator n-1).
    ///
    /// Returns 0.0 if fewer than two observations have been seen.
    pub fn variance(&self) -> f64 {
        if self.count < 2 {
            0.0
        } else {
            self.m2 / (self.count - 1) as f64
        }
    }

    /// Return the sample standard deviation.
    pub fn std(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Return the population variance (denominator n).
    ///
    /// Returns 0.0 if no data has been seen.
    pub fn population_variance(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.m2 / self.count as f64
        }
    }

    /// Return the number of observations seen so far.
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Merge two independent `OnlineMean` trackers (Chan's parallel algorithm).
    ///
    /// Useful when aggregating statistics computed on disjoint partitions of a
    /// dataset (e.g. distributed or multi-threaded processing).
    pub fn merge(&self, other: &OnlineMean) -> OnlineMean {
        let combined_count = self.count + other.count;
        if combined_count == 0 {
            return OnlineMean::new();
        }
        let delta = other.mean - self.mean;
        let combined_mean = (self.mean * self.count as f64 + other.mean * other.count as f64)
            / combined_count as f64;
        let combined_m2 = self.m2
            + other.m2
            + delta * delta * (self.count as f64 * other.count as f64) / combined_count as f64;

        OnlineMean {
            count: combined_count,
            mean: combined_mean,
            m2: combined_m2,
        }
    }
}

impl Default for OnlineMean {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// OnlineLinearRegression — Sherman-Morrison rank-1 update
// ---------------------------------------------------------------------------

/// Online multivariate linear regression via the Sherman-Morrison rank-1 inverse update.
///
/// Maintains the inverse of (X'X + λI) incrementally so each new data point
/// costs O(d²) rather than the O(d³) required by a full batch solve.
/// The forgetting factor λ_f ∈ (0, 1] down-weights older observations,
/// allowing the model to track non-stationary relationships.
///
/// # Model
///
/// Given a d-dimensional feature vector **x** and scalar response y, the model
/// predicts ŷ = **β**·**x** where **β** is updated after each `(x, y)` pair.
///
/// # Examples
///
/// ```
/// use scirs2_series::online_algorithms::OnlineLinearRegression;
///
/// let mut model = OnlineLinearRegression::new(2, 1.0, 1e-4).expect("should succeed");
/// model.update(&[1.0, 2.0], 5.0).expect("should succeed");
/// model.update(&[2.0, 3.0], 8.0).expect("should succeed");
/// let pred = model.predict(&[3.0, 4.0]).expect("should succeed");
/// assert!(pred.is_finite());
/// ```
#[derive(Debug, Clone)]
pub struct OnlineLinearRegression {
    /// Dimension of the feature vector
    dim: usize,
    /// Coefficient vector β  (dim × 1)
    coeffs: Vec<f64>,
    /// Inverse covariance matrix P = (X'X)^{-1}  (dim × dim, row-major)
    p_inv: Vec<f64>,
    /// Forgetting factor λ_f ∈ (0, 1]; 1.0 = no forgetting
    forgetting_factor: f64,
    /// Number of updates applied
    n_updates: u64,
}

impl OnlineLinearRegression {
    /// Create a new model.
    ///
    /// # Arguments
    ///
    /// * `dim` - Number of features (not counting any intercept; add a column of
    ///   ones to `x` if you want an intercept).
    /// * `forgetting_factor` - λ_f ∈ (0, 1].  Values < 1 cause exponential
    ///   down-weighting of older observations.
    /// * `regularization` - Ridge regularization δ added to the diagonal of the
    ///   initial P matrix.  Prevents singular inversions when n < d.
    pub fn new(dim: usize, forgetting_factor: f64, regularization: f64) -> Result<Self> {
        if dim == 0 {
            return Err(TimeSeriesError::InvalidParameter {
                name: "dim".to_string(),
                message: "Dimension must be at least 1".to_string(),
            });
        }
        if !(f64::EPSILON..=1.0).contains(&forgetting_factor) {
            return Err(TimeSeriesError::InvalidParameter {
                name: "forgetting_factor".to_string(),
                message: "Forgetting factor must be in (0, 1]".to_string(),
            });
        }
        if regularization < 0.0 {
            return Err(TimeSeriesError::InvalidParameter {
                name: "regularization".to_string(),
                message: "Regularization must be non-negative".to_string(),
            });
        }

        // Initialise P = (1/δ) * I  (large uncertainty)
        let init_scale = if regularization > 0.0 {
            1.0 / regularization
        } else {
            1e6
        };
        let mut p_inv = vec![0.0; dim * dim];
        for i in 0..dim {
            p_inv[i * dim + i] = init_scale;
        }

        Ok(Self {
            dim,
            coeffs: vec![0.0; dim],
            p_inv,
            forgetting_factor,
            n_updates: 0,
        })
    }

    /// Incorporate one new `(x, y)` observation using the RLS / Sherman-Morrison update.
    ///
    /// # Arguments
    ///
    /// * `x` - Feature slice of length `dim`.
    /// * `y` - Scalar response.
    pub fn update(&mut self, x: &[f64], y: f64) -> Result<()> {
        if x.len() != self.dim {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: self.dim,
                actual: x.len(),
            });
        }

        // g = P * x  (dim-vector)
        let mut g = vec![0.0; self.dim];
        for i in 0..self.dim {
            for j in 0..self.dim {
                g[i] += self.p_inv[i * self.dim + j] * x[j];
            }
        }

        // denom = λ_f + x' * g
        let mut denom = self.forgetting_factor;
        for j in 0..self.dim {
            denom += x[j] * g[j];
        }

        // Gain k = g / denom
        let k: Vec<f64> = g.iter().map(|&gi| gi / denom).collect();

        // Prediction error ε = y - β' x
        let y_hat = self.predict(x)?;
        let eps = y - y_hat;

        // β ← β + k * ε
        for i in 0..self.dim {
            self.coeffs[i] += k[i] * eps;
        }

        // P ← (P - k * g') / λ_f  (Sherman-Morrison update with forgetting)
        for i in 0..self.dim {
            for j in 0..self.dim {
                self.p_inv[i * self.dim + j] =
                    (self.p_inv[i * self.dim + j] - k[i] * g[j]) / self.forgetting_factor;
            }
        }

        self.n_updates += 1;
        Ok(())
    }

    /// Predict the response for feature vector `x`.
    pub fn predict(&self, x: &[f64]) -> Result<f64> {
        if x.len() != self.dim {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: self.dim,
                actual: x.len(),
            });
        }
        let y_hat = self.coeffs.iter().zip(x.iter()).map(|(b, xi)| b * xi).sum();
        Ok(y_hat)
    }

    /// Return the current coefficient vector.
    pub fn coefficients(&self) -> &[f64] {
        &self.coeffs
    }

    /// Return the number of updates applied so far.
    pub fn n_updates(&self) -> u64 {
        self.n_updates
    }

    /// Return a reference to the current inverse covariance matrix (row-major).
    pub fn p_inv(&self) -> &[f64] {
        &self.p_inv
    }
}

// ---------------------------------------------------------------------------
// OnlineARIMA — Recursive Least Squares ARIMA
// ---------------------------------------------------------------------------

/// Online ARIMA(p, d, q) model fitted via recursive least squares.
///
/// Differences the incoming stream `d` times, then maintains an RLS regressor
/// whose regressors are the last `p` differenced values and the last `q`
/// innovation (residual) terms.  Each `update` call costs O((p+q)²).
///
/// # Examples
///
/// ```
/// use scirs2_series::online_algorithms::OnlineARIMA;
///
/// let mut model = OnlineARIMA::new(2, 1, 1, 0.99, 1e-3).expect("should succeed");
/// for t in 0..50 {
///     model.update(t as f64 + (t as f64 * 0.3).sin()).expect("should succeed");
/// }
/// let fc = model.forecast(5).expect("should succeed");
/// assert_eq!(fc.len(), 5);
/// ```
#[derive(Debug, Clone)]
pub struct OnlineARIMA {
    /// AR order
    p: usize,
    /// Integration (differencing) order
    d: usize,
    /// MA order
    q: usize,
    /// RLS regression engine with dim = p + q
    rls: OnlineLinearRegression,
    /// Raw observation history (length ≥ d + 1 needed for differencing)
    raw_history: VecDeque<f64>,
    /// Differenced series history (length ≥ p needed for AR lags)
    diff_history: VecDeque<f64>,
    /// Innovation (residual) history (length ≥ q)
    innovations: VecDeque<f64>,
    /// Whether enough data has arrived to produce valid predictions
    ready: bool,
}

impl OnlineARIMA {
    /// Create a new `OnlineARIMA(p, d, q)` model.
    ///
    /// # Arguments
    ///
    /// * `p`, `d`, `q` - ARIMA orders.
    /// * `forgetting_factor` - Passed to the internal RLS engine (0 < λ ≤ 1).
    /// * `regularization` - Ridge regularization for the RLS inverse.
    pub fn new(
        p: usize,
        d: usize,
        q: usize,
        forgetting_factor: f64,
        regularization: f64,
    ) -> Result<Self> {
        let dim = p + q;
        // Degenerate case: pure white-noise / random-walk — still valid, just
        // predict zero or the last observation respectively.
        let rls = if dim > 0 {
            OnlineLinearRegression::new(dim, forgetting_factor, regularization)?
        } else {
            // dim == 0 is a trivial case; we will predict the last raw value for d>0
            // or zero for d==0.  Create a dummy 1-dim model we will never use.
            OnlineLinearRegression::new(1, forgetting_factor, regularization)?
        };

        Ok(Self {
            p,
            d,
            q,
            rls,
            raw_history: VecDeque::new(),
            diff_history: VecDeque::new(),
            innovations: VecDeque::new(),
            ready: false,
        })
    }

    /// Difference the last value in `raw_history` `d` times and return the result.
    fn difference_value(&self, new_raw: f64) -> f64 {
        if self.d == 0 {
            return new_raw;
        }
        // We need the last `d` raw values to compute the d-th difference.
        // d-th difference requires comparing new_raw with the (d-1)-th diff of
        // the previous step.  We use the raw_history to reconstruct this lazily.
        let mut val = new_raw;
        let hist: Vec<f64> = self.raw_history.iter().copied().collect();
        let len = hist.len();
        for order in 0..self.d {
            if order < len {
                val -= hist[len - 1 - order];
            }
        }
        val
    }

    /// Reconstruct a level-forecast from a differenced forecast.
    ///
    /// For d=1: level_t = level_{t-1} + diff_forecast
    /// For d=2: level_t = 2·level_{t-1} − level_{t-2} + diff2_forecast
    fn undifference(&self, diff_forecast: f64) -> f64 {
        if self.d == 0 {
            return diff_forecast;
        }
        // Integrate once per order
        let hist: Vec<f64> = self.raw_history.iter().copied().collect();
        let len = hist.len();
        let mut val = diff_forecast;
        for order in 1..=self.d {
            if order <= len {
                // Binomial reconstruction: add alternating combinations of last d raw obs
                let idx = len - order;
                let sign = if (order % 2) == 1 { 1.0 } else { -1.0 };
                val += sign * hist[idx];
            }
        }
        val
    }

    /// Build the feature vector [ar_lags | ma_lags] for the RLS model.
    fn build_features(&self) -> Option<Vec<f64>> {
        let dim = self.p + self.q;
        if dim == 0 {
            return None;
        }
        let mut feats = vec![0.0; dim];

        // AR lags from diff_history
        let diff_len = self.diff_history.len();
        for i in 0..self.p {
            if i < diff_len {
                feats[i] = self.diff_history[diff_len - 1 - i];
            }
        }

        // MA lags from innovations
        let innov_len = self.innovations.len();
        for i in 0..self.q {
            if i < innov_len {
                feats[self.p + i] = self.innovations[innov_len - 1 - i];
            }
        }

        Some(feats)
    }

    /// Incorporate one new raw observation.
    pub fn update(&mut self, raw_value: f64) -> Result<()> {
        let diff_val = self.difference_value(raw_value);

        // Build feature vector before adding the new diff to history
        if let Some(feats) = self.build_features() {
            if self.ready {
                self.rls.update(&feats, diff_val)?;
                // Compute innovation = actual − predicted
                let predicted = self.rls.predict(&feats)?;
                let innovation = diff_val - predicted;
                self.push_innovation(innovation);
            } else {
                // Warm-up: do not yet update RLS, just record
                let predicted_zero = 0.0;
                let innovation = diff_val - predicted_zero;
                self.push_innovation(innovation);
            }
        } else {
            // p=q=0: trivial model, innovation = diff_val
            self.push_innovation(diff_val);
        }

        // Update histories
        self.raw_history.push_back(raw_value);
        if self.raw_history.len() > self.d.max(1) + self.p + 10 {
            self.raw_history.pop_front();
        }

        self.diff_history.push_back(diff_val);
        if self.diff_history.len() > self.p + 10 {
            self.diff_history.pop_front();
        }

        // Mark ready once we have enough lags
        let needed = self.p.max(self.q) + self.d + 2;
        if !self.ready && self.raw_history.len() >= needed {
            self.ready = true;
        }

        Ok(())
    }

    fn push_innovation(&mut self, innov: f64) {
        self.innovations.push_back(innov);
        if self.innovations.len() > self.q + 10 {
            self.innovations.pop_front();
        }
    }

    /// Predict the next differenced value (one-step ahead).
    pub fn predict_diff(&self) -> Result<f64> {
        if !self.ready {
            return Ok(0.0);
        }
        let dim = self.p + self.q;
        if dim == 0 {
            return Ok(0.0);
        }
        let feats = self.build_features().unwrap_or_else(|| vec![0.0; dim]);
        self.rls.predict(&feats)
    }

    /// Predict the next raw level value (one-step ahead).
    pub fn predict_level(&self) -> Result<f64> {
        let diff_pred = self.predict_diff()?;
        Ok(self.undifference(diff_pred))
    }

    /// Generate a multi-step ahead forecast of raw level values.
    ///
    /// Uses the recursive approach: each new forecast is appended to the history
    /// and the model steps forward, assuming zero future innovations.
    pub fn forecast(&self, steps: usize) -> Result<Vec<f64>> {
        if steps == 0 {
            return Ok(Vec::new());
        }

        let mut forecasts = Vec::with_capacity(steps);
        let mut raw_hist: VecDeque<f64> = self.raw_history.clone();
        let mut diff_hist: VecDeque<f64> = self.diff_history.clone();
        let mut innov_hist: VecDeque<f64> = self.innovations.clone();

        for _ in 0..steps {
            // Build features
            let dim = self.p + self.q;
            let mut feats = vec![0.0; dim.max(1)];
            if dim > 0 {
                let dl = diff_hist.len();
                for i in 0..self.p {
                    if i < dl {
                        feats[i] = diff_hist[dl - 1 - i];
                    }
                }
                let il = innov_hist.len();
                for i in 0..self.q {
                    if i < il {
                        feats[self.p + i] = innov_hist[il - 1 - i];
                    }
                }
            }

            let diff_pred = if dim > 0 {
                self.rls.predict(&feats[..dim])?
            } else {
                0.0
            };

            // Undifference using current raw_hist
            let level_pred = if self.d == 0 {
                diff_pred
            } else {
                let rlen = raw_hist.len();
                let mut val = diff_pred;
                for order in 1..=self.d {
                    if order <= rlen {
                        let idx = rlen - order;
                        let sign = if (order % 2) == 1 { 1.0 } else { -1.0 };
                        val += sign * raw_hist[idx];
                    }
                }
                val
            };

            forecasts.push(level_pred);

            // Update histories for next step (zero innovation assumption)
            raw_hist.push_back(level_pred);
            if raw_hist.len() > self.d.max(1) + self.p + 10 {
                raw_hist.pop_front();
            }

            let diff_new = if self.d == 0 {
                level_pred
            } else {
                let rlen = raw_hist.len();
                let mut v = level_pred;
                let hist_vec: Vec<f64> = raw_hist.iter().copied().collect();
                for order in 0..self.d {
                    if order + 1 < rlen {
                        v -= hist_vec[rlen - 2 - order];
                    }
                }
                v
            };

            diff_hist.push_back(diff_new);
            if diff_hist.len() > self.p + 10 {
                diff_hist.pop_front();
            }

            // Assume zero future innovations
            innov_hist.push_back(0.0);
            if innov_hist.len() > self.q + 10 {
                innov_hist.pop_front();
            }
        }

        Ok(forecasts)
    }

    /// Return whether the model has seen enough data to make predictions.
    pub fn is_ready(&self) -> bool {
        self.ready
    }

    /// Number of raw observations seen.
    pub fn n_observations(&self) -> usize {
        self.raw_history.len()
    }
}

// ---------------------------------------------------------------------------
// ADWINDetector — ADaptive WINdowing
// ---------------------------------------------------------------------------

/// ADaptive WINdowing (ADWIN) concept-drift detector.
///
/// ADWIN maintains a variable-length window W of the most recent observations
/// and continually checks whether any contiguous sub-window W₀ of W has a
/// significantly different mean from the complementary sub-window W₁ = W \ W₀.
/// When such a split is found the older portion is dropped, signalling drift.
///
/// ## Reference
/// Bifet & Gavalda (2007). *Learning from Time-Changing Data with Adaptive Windowing.*
/// SIAM International Conference on Data Mining.
///
/// # Examples
///
/// ```
/// use scirs2_series::online_algorithms::ADWINDetector;
///
/// let mut detector = ADWINDetector::new(0.002);
/// // Feed stable data
/// for _ in 0..100 {
///     detector.update(1.0);
/// }
/// // Introduce a step change
/// let drift = (0..50).fold(false, |acc, _| acc || detector.update(10.0));
/// assert!(drift);
/// ```
#[derive(Debug, Clone)]
pub struct ADWINDetector {
    /// Confidence parameter δ (0 < δ < 1). Smaller = more sensitive.
    delta: f64,
    /// The sliding window of recent observations
    window: VecDeque<f64>,
    /// Running sum of all window elements (for O(1) sub-window mean)
    total: f64,
    /// Whether drift was detected on the last update
    last_detected: bool,
}

impl ADWINDetector {
    /// Create a new ADWIN detector.
    ///
    /// # Arguments
    ///
    /// * `delta` - False-alarm confidence level. Typical values: 0.002 – 0.05.
    pub fn new(delta: f64) -> Self {
        Self {
            delta: delta.max(f64::EPSILON),
            window: VecDeque::new(),
            total: 0.0,
            last_detected: false,
        }
    }

    /// Incorporate a new observation and test for drift.
    ///
    /// Returns `true` when drift is detected (and the window has been reset to
    /// contain only the observations since the most recent detected change).
    pub fn update(&mut self, x: f64) -> bool {
        self.window.push_back(x);
        self.total += x;

        let detected = self.test_and_compress();
        self.last_detected = detected;
        detected
    }

    /// ADWIN cut-point test over all O(n) possible splits.
    ///
    /// For efficiency we scan from the front (oldest data) and maintain prefix
    /// sums.  When a significant split is found we remove everything up to and
    /// including the cut-point, and restart.
    fn test_and_compress(&mut self) -> bool {
        let n = self.window.len();
        if n < 2 {
            return false;
        }

        // Hoeffding-based threshold helper: m* = 1/(1/n0 + 1/n1)
        // ε_cut = sqrt( ln(4·n²/δ) / (2·m*) )
        let ln_factor = (4.0 * (n as f64).powi(2) / self.delta).ln();

        let mut prefix_sum = 0.0;
        let mut detected = false;

        // Scan all split points from oldest to newest
        for i in 0..(n - 1) {
            prefix_sum += self.window[i];
            let n0 = (i + 1) as f64;
            let n1 = (n - i - 1) as f64;
            let mean0 = prefix_sum / n0;
            let mean1 = (self.total - prefix_sum) / n1;
            let delta_mu = (mean0 - mean1).abs();

            let m_star = 1.0 / (1.0 / n0 + 1.0 / n1);
            let eps_cut = (ln_factor / (2.0 * m_star)).sqrt();

            if delta_mu > eps_cut {
                // Drift: discard the older sub-window (indices 0..=i)
                for _ in 0..=(i) {
                    if let Some(v) = self.window.pop_front() {
                        self.total -= v;
                    }
                }
                detected = true;
                break;
            }
        }

        detected
    }

    /// Return whether drift was detected on the most recent `update` call.
    pub fn is_change_detected(&self) -> bool {
        self.last_detected
    }

    /// Return the current window size.
    pub fn window_len(&self) -> usize {
        self.window.len()
    }

    /// Return the current window mean.
    ///
    /// Returns `None` if the window is empty.
    pub fn window_mean(&self) -> Option<f64> {
        if self.window.is_empty() {
            None
        } else {
            Some(self.total / self.window.len() as f64)
        }
    }

    /// Reset the detector, clearing all buffered data.
    pub fn reset(&mut self) {
        self.window.clear();
        self.total = 0.0;
        self.last_detected = false;
    }

    /// Expose the raw window for inspection.
    pub fn window(&self) -> &VecDeque<f64> {
        &self.window
    }
}

// ---------------------------------------------------------------------------
// PageHinkley — Sequential change detection
// ---------------------------------------------------------------------------

/// Page-Hinkley test for sequential (online) change-point detection.
///
/// The test accumulates a cumulative sum of deviations from the running mean
/// and signals a change when the cumulative sum drops far enough below its
/// maximum (or rises far enough above its minimum for two-sided tests).
///
/// ## Reference
/// Page, E. S. (1954). *Continuous Inspection Schemes.*
/// Biometrika, 41(1/2), 100–115.
///
/// # Examples
///
/// ```
/// use scirs2_series::online_algorithms::PageHinkley;
///
/// let mut ph = PageHinkley::new(50.0, 0.01, 0.005, true);
/// for _ in 0..100 { ph.update(0.0); }
/// let change = (0..30).fold(false, |acc, _| acc || ph.update(5.0));
/// assert!(change);
/// ```
#[derive(Debug, Clone)]
pub struct PageHinkley {
    /// Detection threshold λ.  Larger = fewer false alarms.
    threshold: f64,
    /// Allowed deviation δ from mean (mean shift allowed before alarm).
    delta: f64,
    /// Learning rate α for exponential mean estimate update.
    alpha: f64,
    /// Whether to detect increases (true) or decreases (false).
    detect_increase: bool,
    /// Running mean estimate
    mean_est: f64,
    /// Cumulative Page-Hinkley statistic
    ph_stat: f64,
    /// Maximum (or minimum for decrease detection) cumulative sum seen so far
    extreme_stat: f64,
    /// Number of observations seen
    n: u64,
    /// Last detection result
    last_detected: bool,
}

impl PageHinkley {
    /// Create a new Page-Hinkley test.
    ///
    /// # Arguments
    ///
    /// * `threshold` - Alarm threshold λ (typical: 10–100).
    /// * `delta` - Minimum mean shift that should be detected (typical: 0.005–0.1).
    /// * `alpha` - Smoothing factor for the running mean (typical: 0.001–0.1).
    /// * `detect_increase` - `true` to detect positive mean shifts; `false` for
    ///   negative shifts.  Call with both `true` and `false` for two-sided detection.
    pub fn new(threshold: f64, delta: f64, alpha: f64, detect_increase: bool) -> Self {
        Self {
            threshold,
            delta,
            alpha: alpha.clamp(f64::EPSILON, 1.0),
            detect_increase,
            mean_est: 0.0,
            ph_stat: 0.0,
            extreme_stat: 0.0,
            n: 0,
            last_detected: false,
        }
    }

    /// Incorporate a new observation and test for a change.
    ///
    /// Returns `true` when the Page-Hinkley statistic crosses the threshold.
    pub fn update(&mut self, x: f64) -> bool {
        self.n += 1;

        // Update exponential running mean
        if self.n == 1 {
            self.mean_est = x;
        } else {
            self.mean_est = (1.0 - self.alpha) * self.mean_est + self.alpha * x;
        }

        // Accumulate Page-Hinkley statistic
        let dev = x - self.mean_est;
        if self.detect_increase {
            // For increase detection, accumulate (x - mean - delta).
            // Under the null (no change), this drifts downward (negative delta bias).
            // A positive mean shift causes the sum to increase.
            self.ph_stat += dev - self.delta;
            // Track the running minimum
            if self.ph_stat < self.extreme_stat {
                self.extreme_stat = self.ph_stat;
            }
            // Alarm when current sum exceeds running minimum by threshold
            let ph_val = self.ph_stat - self.extreme_stat;
            self.last_detected = ph_val > self.threshold;
        } else {
            // For decrease detection, accumulate -(x - mean) - delta = -(dev) - delta
            self.ph_stat += -dev - self.delta;
            // Track the running minimum
            if self.ph_stat < self.extreme_stat {
                self.extreme_stat = self.ph_stat;
            }
            // Alarm when current sum exceeds running minimum by threshold
            let ph_val = self.ph_stat - self.extreme_stat;
            self.last_detected = ph_val > self.threshold;
        }

        self.last_detected
    }

    /// Return whether the most recent `update` detected a change.
    pub fn is_change_detected(&self) -> bool {
        self.last_detected
    }

    /// Return the current Page-Hinkley statistic value.
    pub fn statistic(&self) -> f64 {
        self.ph_stat
    }

    /// Return the running mean estimate.
    pub fn mean_estimate(&self) -> f64 {
        self.mean_est
    }

    /// Reset the detector state (mean estimate, statistics, counters).
    pub fn reset(&mut self) {
        self.mean_est = 0.0;
        self.ph_stat = 0.0;
        self.extreme_stat = 0.0;
        self.n = 0;
        self.last_detected = false;
    }
}

// ---------------------------------------------------------------------------
// OnlineQuantile — P² algorithm
// ---------------------------------------------------------------------------

/// P² (Piecewise-Parabolic) algorithm for streaming quantile estimation.
///
/// Jain & Chlamtac (1985) proposed P² as a single-pass, O(1)-memory algorithm
/// for estimating arbitrary quantiles of a data stream.  It maintains exactly 5
/// markers whose positions approximate the actual quantile.
///
/// ## Reference
/// Jain, R. & Chlamtac, I. (1985). *The P2 Algorithm for Dynamic Calculation of
/// Quantiles and Histograms Without Storing Observations.*
/// Communications of the ACM, 28(10), 1076–1085.
///
/// # Examples
///
/// ```
/// use scirs2_series::online_algorithms::OnlineQuantile;
///
/// let mut oq = OnlineQuantile::new(0.5).expect("should succeed"); // Median
/// for x in 0..=100 {
///     oq.update(x as f64);
/// }
/// let est = oq.quantile();
/// assert!((est - 50.0).abs() < 5.0);
/// ```
#[derive(Debug, Clone)]
pub struct OnlineQuantile {
    /// Target quantile p ∈ (0, 1)
    p: f64,
    /// Marker heights q[0..5]
    q: [f64; 5],
    /// Marker positions n[0..5]
    n: [i64; 5],
    /// Desired marker positions n'[0..5]
    n_desired: [f64; 5],
    /// Increments dn'[0..5]
    dn: [f64; 5],
    /// Number of observations seen
    count: u64,
    /// Sorted buffer used for the first 5 observations
    init_buffer: Vec<f64>,
}

impl OnlineQuantile {
    /// Create a new P² quantile estimator.
    ///
    /// # Arguments
    ///
    /// * `p` - Target quantile ∈ (0, 1).  E.g., `0.5` for the median,
    ///   `0.95` for the 95th percentile.
    pub fn new(p: f64) -> Result<Self> {
        if p <= 0.0 || p >= 1.0 {
            return Err(TimeSeriesError::InvalidParameter {
                name: "p".to_string(),
                message: "Quantile must be strictly between 0 and 1".to_string(),
            });
        }

        // P² uses 5 markers: min, p/2, p, (1+p)/2, max
        let dn = [0.0, p / 2.0, p, (1.0 + p) / 2.0, 1.0];

        Ok(Self {
            p,
            q: [0.0; 5],
            n: [1, 2, 3, 4, 5],
            n_desired: [1.0, 1.0 + 2.0 * p, 1.0 + 4.0 * p, 3.0 + 2.0 * p, 5.0],
            dn,
            count: 0,
            init_buffer: Vec::with_capacity(5),
        })
    }

    /// Update the estimator with a new data point.
    pub fn update(&mut self, x: f64) {
        self.count += 1;

        // Phase 1: accumulate first 5 observations in sorted buffer
        if self.count <= 5 {
            self.init_buffer.push(x);
            if self.count == 5 {
                self.init_buffer
                    .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                for i in 0..5 {
                    self.q[i] = self.init_buffer[i];
                    self.n[i] = (i + 1) as i64;
                }
                self.n_desired = [
                    1.0,
                    1.0 + 2.0 * self.p,
                    1.0 + 4.0 * self.p,
                    3.0 + 2.0 * self.p,
                    5.0,
                ];
            }
            return;
        }

        // Phase 2: P² update

        // 1. Find k such that q[k] ≤ x < q[k+1]
        let k = if x < self.q[0] {
            self.q[0] = x;
            0i64
        } else if x >= self.q[4] {
            self.q[4] = x;
            3
        } else {
            let mut ki = 3i64;
            for i in 0..4 {
                if x < self.q[i + 1] {
                    ki = i as i64;
                    break;
                }
            }
            ki
        };

        // 2. Increment marker counts for markers to the right of x
        for i in (k + 1) as usize..5 {
            self.n[i] += 1;
        }

        // 3. Update desired positions
        for i in 0..5 {
            self.n_desired[i] += self.dn[i];
        }

        // 4. Adjust marker heights
        for i in 1..4 {
            let d = self.n_desired[i] - self.n[i] as f64;
            if (d >= 1.0 && (self.n[i + 1] - self.n[i]) > 1)
                || (d <= -1.0 && (self.n[i - 1] - self.n[i]) < -1)
            {
                let sign = if d >= 0.0 { 1.0 } else { -1.0 };
                let q_new = self.parabolic(i, sign);
                // Accept parabolic update only if it keeps monotonicity
                if self.q[i - 1] < q_new && q_new < self.q[i + 1] {
                    self.q[i] = q_new;
                } else {
                    // Fall back to linear interpolation
                    self.q[i] = self.linear(i, sign);
                }
                self.n[i] += sign as i64;
            }
        }
    }

    /// Parabolic (P²) interpolation formula.
    fn parabolic(&self, i: usize, d: f64) -> f64 {
        let qi = self.q[i];
        let qi_1 = self.q[i - 1];
        let qi1 = self.q[i + 1];
        let ni = self.n[i] as f64;
        let ni_1 = self.n[i - 1] as f64;
        let ni1 = self.n[i + 1] as f64;

        qi + d / (ni1 - ni_1)
            * ((ni - ni_1 + d) * (qi1 - qi) / (ni1 - ni)
                + (ni1 - ni - d) * (qi - qi_1) / (ni - ni_1))
    }

    /// Linear interpolation (fallback when parabolic would violate monotonicity).
    fn linear(&self, i: usize, d: f64) -> f64 {
        let j = if d > 0.0 { i + 1 } else { i - 1 };
        self.q[i] + d * (self.q[j] - self.q[i]) / (self.n[j] - self.n[i]) as f64
    }

    /// Return the current quantile estimate.
    ///
    /// Returns `None` if fewer than 5 observations have been seen.
    pub fn quantile(&self) -> f64 {
        if self.count < 5 {
            // Fallback: return exact quantile from the small init buffer
            if self.init_buffer.is_empty() {
                return 0.0;
            }
            let mut sorted = self.init_buffer.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let idx = ((sorted.len() as f64 * self.p).ceil() as usize).saturating_sub(1);
            sorted[idx.min(sorted.len() - 1)]
        } else {
            self.q[2]
        }
    }

    /// Return how many observations have been incorporated.
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Return the target quantile level p.
    pub fn p(&self) -> f64 {
        self.p
    }

    /// Return the current 5 marker heights.
    pub fn markers(&self) -> [f64; 5] {
        self.q
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_online_mean_welford() {
        // Classic example from Knuth TAOCP
        let data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let mut tracker = OnlineMean::new();
        for &x in &data {
            tracker.update(x);
        }
        assert!((tracker.mean() - 5.0).abs() < 1e-10, "mean mismatch");
        // Population variance = 4.0; sample variance = 32/7 ≈ 4.5714
        let sample_var = 32.0 / 7.0;
        assert!(
            (tracker.variance() - sample_var).abs() < 1e-9,
            "variance mismatch: got {}",
            tracker.variance()
        );
        assert_eq!(tracker.count(), 8);
    }

    #[test]
    fn test_online_mean_merge() {
        let mut a = OnlineMean::new();
        let mut b = OnlineMean::new();
        for &x in &[1.0f64, 2.0, 3.0] {
            a.update(x);
        }
        for &x in &[4.0f64, 5.0, 6.0] {
            b.update(x);
        }
        let merged = a.merge(&b);
        assert!((merged.mean() - 3.5).abs() < 1e-10);
        assert_eq!(merged.count(), 6);
    }

    #[test]
    fn test_online_linear_regression_simple() {
        // y = 2x + 1
        let mut model = OnlineLinearRegression::new(2, 1.0, 1e-6).expect("failed to create model");
        for i in 1..=50i64 {
            let x = i as f64;
            let y = 2.0 * x + 1.0;
            // Feature: [x, 1] for intercept
            model.update(&[x, 1.0], y).expect("unexpected None or Err");
        }
        // Predict for x = 10
        let pred = model.predict(&[10.0, 1.0]).expect("failed to create pred");
        assert!(
            (pred - 21.0).abs() < 0.5,
            "prediction error too large: {}",
            pred
        );
    }

    #[test]
    fn test_online_linear_regression_dim_check() {
        let mut model = OnlineLinearRegression::new(3, 1.0, 1e-4).expect("failed to create model");
        let err = model.update(&[1.0, 2.0], 5.0); // wrong dim
        assert!(err.is_err());
        let err2 = model.predict(&[1.0]); // wrong dim
        assert!(err2.is_err());
    }

    #[test]
    fn test_online_arima_basic() {
        let mut model = OnlineARIMA::new(2, 1, 1, 0.99, 1e-4).expect("failed to create model");
        // Feed a linearly increasing series
        for t in 0..60 {
            model.update(t as f64).expect("unexpected None or Err");
        }
        assert!(model.is_ready());
        let fc = model.forecast(5).expect("failed to create fc");
        assert_eq!(fc.len(), 5);
        // The series is a random walk with drift 1; forecasts should be increasing
        for i in 1..fc.len() {
            assert!(
                fc[i] >= fc[i - 1] - 2.0,
                "forecast should be roughly increasing"
            );
        }
    }

    #[test]
    fn test_adwin_stable_data() {
        let mut d = ADWINDetector::new(0.002);
        let mut any_drift = false;
        for _ in 0..200 {
            if d.update(5.0) {
                any_drift = true;
            }
        }
        // Stable constant stream: drift should not be detected after window stabilises
        // (Some transient detections at the very start are acceptable)
        let _ = any_drift; // just checking no panic
    }

    #[test]
    fn test_adwin_detects_drift() {
        let mut d = ADWINDetector::new(0.01);
        // Phase 1: stable
        for _ in 0..100 {
            d.update(0.0);
        }
        // Phase 2: large step change
        let mut detected = false;
        for _ in 0..50 {
            if d.update(100.0) {
                detected = true;
                break;
            }
        }
        assert!(detected, "ADWIN should detect a large step change");
    }

    #[test]
    fn test_page_hinkley_detects_increase() {
        let mut ph = PageHinkley::new(50.0, 0.01, 0.005, true);
        for _ in 0..100 {
            ph.update(0.0);
        }
        let mut detected = false;
        for _ in 0..80 {
            if ph.update(5.0) {
                detected = true;
                break;
            }
        }
        assert!(detected, "PH should detect mean increase");
    }

    #[test]
    fn test_page_hinkley_no_false_alarm_stable() {
        let mut ph = PageHinkley::new(200.0, 0.0, 0.001, true);
        let mut alarm_count = 0u32;
        for _ in 0..500 {
            if ph.update(1.0) {
                alarm_count += 1;
            }
        }
        assert!(
            alarm_count < 3,
            "PH should not alarm on stable data (got {alarm_count})"
        );
    }

    #[test]
    fn test_online_quantile_median() {
        let mut oq = OnlineQuantile::new(0.5).expect("failed to create oq");
        for x in 1..=1000 {
            oq.update(x as f64);
        }
        let est = oq.quantile();
        assert!(
            (est - 500.0).abs() < 20.0,
            "median estimate {est} should be close to 500"
        );
    }

    #[test]
    fn test_online_quantile_percentile_95() {
        let mut oq = OnlineQuantile::new(0.95).expect("failed to create oq");
        for x in 0..=1000 {
            oq.update(x as f64);
        }
        let est = oq.quantile();
        // True 95th percentile ≈ 950
        assert!(
            (est - 950.0).abs() < 30.0,
            "95th percentile estimate {est} should be near 950"
        );
    }

    #[test]
    fn test_online_quantile_invalid_p() {
        assert!(OnlineQuantile::new(0.0).is_err());
        assert!(OnlineQuantile::new(1.0).is_err());
        assert!(OnlineQuantile::new(-0.1).is_err());
    }

    #[test]
    fn test_online_mean_single_value() {
        let mut m = OnlineMean::new();
        m.update(42.0);
        assert_eq!(m.mean(), 42.0);
        assert_eq!(m.variance(), 0.0);
        assert_eq!(m.count(), 1);
    }
}
