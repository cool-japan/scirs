//! Online learning for time series with concept drift detection
//!
//! This module provides algorithms that process time series data in a single
//! pass, updating model state incrementally as each new observation arrives.
//! It also includes detectors for **concept drift** — sudden or gradual changes
//! in the data-generating distribution.
//!
//! # Algorithms
//!
//! | Struct | Purpose |
//! |--------|---------|
//! | [`PageHinkleyTest`] | Sequential mean-shift change detector (Page-Hinkley test) |
//! | [`Adwin`] | Adaptive windowing drift detector (ADWIN) |
//! | [`OnlineArima`] | Recursive-least-squares ARIMA with forgetting factor |
//! | [`AdaptiveExpSmoothing`] | Exponential smoothing with adaptive α |
//!
//! ## Design notes
//! - All structs are `no_std`-friendly apart from the use of `std::collections::VecDeque`.
//! - No `unwrap()` is used.
//! - Every public method returns a plain value rather than a `Result` where
//!   that value is well-defined even in degenerate cases.

use std::collections::VecDeque;

// ─── Page-Hinkley test ───────────────────────────────────────────────────────

/// Page-Hinkley (PH) sequential change-point test.
///
/// The test detects an *increase* in the mean of an observation stream.  After
/// `update` is called with successive observations the method returns `true`
/// when the cumulative sum (CUSUM) statistic exceeds `delta`.
///
/// # Parameters
/// * `lambda` – allowed per-step mean drift (sensitivity, small → more sensitive)
/// * `delta`  – detection threshold; alarm when `M_t - m_t > delta`
/// * `alpha`  – forgetting factor for the running mean (0 < α ≤ 1)
///
/// # Example
/// ```
/// use scirs2_series::online::PageHinkleyTest;
/// let mut ph = PageHinkleyTest::new(0.05, 50.0);
/// for _ in 0..100 { ph.update(0.0); }   // stable phase
/// // After a sudden shift the test should detect quickly
/// let mut detected = false;
/// for _ in 0..100 {
///     if ph.update(5.0) { detected = true; break; }
/// }
/// assert!(detected);
/// ```
#[derive(Debug, Clone)]
pub struct PageHinkleyTest {
    /// Per-step allowed mean increase
    pub lambda: f64,
    /// Detection threshold
    pub delta: f64,
    /// Forgetting factor for the running mean estimate (default 0.01)
    pub alpha: f64,
    cumsum: f64,
    min_cumsum: f64,
    running_mean: f64,
    n_obs: usize,
}

impl PageHinkleyTest {
    /// Create a new Page-Hinkley test.
    ///
    /// * `lambda` – permissible mean increase per observation (e.g. 0.05)
    /// * `delta`  – alarm threshold (e.g. 50.0 for low false-alarm rate)
    pub fn new(lambda: f64, delta: f64) -> Self {
        Self {
            lambda,
            delta,
            alpha: 0.01,
            cumsum: 0.0,
            min_cumsum: 0.0,
            running_mean: 0.0,
            n_obs: 0,
        }
    }

    /// Create with a custom forgetting factor `alpha` for the running mean.
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha.clamp(1e-6, 1.0);
        self
    }

    /// Incorporate a new observation.
    ///
    /// Returns `true` if a change has been detected.
    pub fn update(&mut self, value: f64) -> bool {
        self.n_obs += 1;
        // Warm-up: update running mean with a decaying alpha
        let effective_alpha = if self.n_obs == 1 { 1.0 } else { self.alpha };
        self.running_mean = (1.0 - effective_alpha) * self.running_mean + effective_alpha * value;

        // Page-Hinkley cumulative sum
        self.cumsum += value - self.running_mean - self.lambda;
        if self.cumsum < self.min_cumsum {
            self.min_cumsum = self.cumsum;
        }

        (self.cumsum - self.min_cumsum) > self.delta
    }

    /// Reset all internal state.
    pub fn reset(&mut self) {
        self.cumsum = 0.0;
        self.min_cumsum = 0.0;
        self.running_mean = 0.0;
        self.n_obs = 0;
    }

    /// Whether the most recent `update` raised an alarm.
    pub fn is_change_detected(&self) -> bool {
        (self.cumsum - self.min_cumsum) > self.delta
    }

    /// Current CUSUM statistic (cumsum - min_cumsum).
    pub fn statistic(&self) -> f64 {
        self.cumsum - self.min_cumsum
    }

    /// Number of observations processed so far.
    pub fn n_obs(&self) -> usize {
        self.n_obs
    }
}

// ─── ADWIN ───────────────────────────────────────────────────────────────────

/// ADWIN (ADaptive WINdowing) concept-drift detector.
///
/// ADWIN maintains a variable-length sliding window of recent observations.
/// After each update it tests whether any sub-window differs significantly in
/// mean from the rest.  When such a difference is found the older portion is
/// dropped, signalling drift.
///
/// This is a simplified but faithful implementation of the algorithm from
/// Bifet & Gavalda (2007).
///
/// # Example
/// ```
/// use scirs2_series::online::Adwin;
/// let mut adwin = Adwin::new(0.002);
/// for _ in 0..200 { adwin.update(0.0); }
/// let mut detected = false;
/// for _ in 0..200 {
///     if adwin.update(1.0) { detected = true; break; }
/// }
/// assert!(detected, "ADWIN should detect large step change");
/// ```
#[derive(Debug, Clone)]
pub struct Adwin {
    /// Confidence parameter δ (smaller → fewer false alarms, slower detection)
    pub delta: f64,
    window: VecDeque<f64>,
    /// Running sum of all elements in the window
    window_sum: f64,
}

impl Adwin {
    /// Construct a new ADWIN detector.
    ///
    /// * `delta` – confidence parameter (e.g. 0.002 for 0.2 % false alarm rate)
    pub fn new(delta: f64) -> Self {
        Self {
            delta: delta.max(1e-10),
            window: VecDeque::new(),
            window_sum: 0.0,
        }
    }

    /// Add a new observation.
    ///
    /// Returns `true` if drift was detected (the window was cut).
    pub fn update(&mut self, value: f64) -> bool {
        self.window.push_back(value);
        self.window_sum += value;

        // Attempt to detect: scan from the back (most recent split)
        let mut drift = false;
        let n = self.window.len();
        if n < 2 {
            return false;
        }

        // Check every possible split point for a significant mean difference
        let mut sum_b = 0.0_f64; // sum of window B (suffix)
        let mut cut_point = 0usize;

        // Collect into a vec for indexing (VecDeque doesn't support range slices)
        let data: Vec<f64> = self.window.iter().cloned().collect();

        for i in (1..n).rev() {
            let nb = (n - i) as f64;
            let na = i as f64;
            sum_b += data[i];
            let sum_a = self.window_sum - sum_b;
            let mean_a = sum_a / na;
            let mean_b = sum_b / nb;

            // Hoeffding bound
            let m_inv = 1.0 / na + 1.0 / nb;
            let epsilon_cut = ((m_inv / 2.0) * (1.0 / self.delta).ln()).sqrt();

            if (mean_a - mean_b).abs() >= epsilon_cut {
                cut_point = i;
                drift = true;
                break;
            }
        }

        if drift {
            // Drop all elements before the cut point
            let removed: f64 = data[..cut_point].iter().sum();
            for _ in 0..cut_point {
                self.window.pop_front();
            }
            self.window_sum -= removed;
        }

        drift
    }

    /// Current mean of the adaptive window.
    pub fn mean(&self) -> f64 {
        if self.window.is_empty() {
            0.0
        } else {
            self.window_sum / self.window.len() as f64
        }
    }

    /// Current number of elements in the adaptive window.
    pub fn width(&self) -> usize {
        self.window.len()
    }
}

// ─── Online ARIMA ────────────────────────────────────────────────────────────

/// Online ARIMA model updated via Recursive Least Squares (RLS) with a
/// forgetting factor λ ∈ (0, 1].
///
/// The model integrates the series `d` times and then fits an AR(p) model to
/// the differenced series.  MA terms are approximated by feeding back the
/// one-step prediction errors.
///
/// # Example
/// ```
/// use scirs2_series::online::OnlineArima;
/// let mut model = OnlineArima::new(2, 1, 1, 0.98);
/// for i in 0..50 {
///     let y = i as f64 + (i as f64 * 0.3).sin();
///     model.update(y);
/// }
/// let fc = model.predict(3);
/// assert_eq!(fc.len(), 3);
/// ```
#[derive(Debug, Clone)]
pub struct OnlineArima {
    /// AR order
    pub p: usize,
    /// Integration order
    pub d: usize,
    /// MA order (approximated via residuals)
    pub q: usize,
    /// Forgetting factor (0 < λ ≤ 1)
    pub forgetting_factor: f64,
    ar_params: Vec<f64>,
    ma_params: Vec<f64>,
    /// RLS gain matrix (p + q) × (p + q), stored row-major
    rls_p: Vec<f64>,
    history: VecDeque<f64>,      // raw level values
    diff_history: VecDeque<f64>, // differenced values (order d)
    errors: VecDeque<f64>,       // recent prediction errors
    n_obs: usize,
}

impl OnlineArima {
    /// Create a new online ARIMA(p, d, q) model.
    ///
    /// * `p`      – AR order  
    /// * `d`      – differencing order  
    /// * `q`      – MA order  
    /// * `lambda` – forgetting factor (0.9 – 1.0 typical; 1.0 = no forgetting)
    pub fn new(p: usize, d: usize, q: usize, lambda: f64) -> Self {
        let dim = p + q;
        let ar_params = vec![0.0; p];
        let ma_params = vec![0.0; q];
        // Initialise P to large diagonal (high prior uncertainty)
        let mut rls_p = vec![0.0_f64; dim * dim];
        for i in 0..dim {
            rls_p[i * dim + i] = 1000.0;
        }
        Self {
            p,
            d,
            q,
            forgetting_factor: lambda.clamp(0.5, 1.0),
            ar_params,
            ma_params,
            rls_p,
            history: VecDeque::new(),
            diff_history: VecDeque::new(),
            errors: VecDeque::new(),
            n_obs: 0,
        }
    }

    /// Difference a series `d` times.
    fn difference(series: &VecDeque<f64>, d: usize) -> Vec<f64> {
        let mut v: Vec<f64> = series.iter().cloned().collect();
        for _ in 0..d {
            let diff: Vec<f64> = v.windows(2).map(|w| w[1] - w[0]).collect();
            v = diff;
        }
        v
    }

    /// Integrate (cumulative sum) `d` times starting from `start_values`.
    fn integrate(diff_forecast: &[f64], last_levels: &[f64], d: usize) -> Vec<f64> {
        if d == 0 {
            return diff_forecast.to_vec();
        }
        let mut result = diff_forecast.to_vec();
        for _order in 0..d {
            let seed = *last_levels.last().unwrap_or(&0.0);
            let mut integrated = Vec::with_capacity(result.len());
            let mut prev = seed;
            for &v in &result {
                prev += v;
                integrated.push(prev);
            }
            result = integrated;
        }
        result
    }

    /// Build the regressor vector φ = [y_{t-1}, …, y_{t-p}, e_{t-1}, …, e_{t-q}].
    fn build_regressor(&self) -> Vec<f64> {
        let mut phi = Vec::with_capacity(self.p + self.q);
        let dh: Vec<f64> = self.diff_history.iter().cloned().collect();
        let len = dh.len();
        for i in 0..self.p {
            phi.push(if len > i { dh[len - 1 - i] } else { 0.0 });
        }
        let elen = self.errors.len();
        for i in 0..self.q {
            phi.push(if elen > i {
                *self.errors.iter().nth(elen - 1 - i).unwrap_or(&0.0)
            } else {
                0.0
            });
        }
        phi
    }

    /// Compute one-step-ahead prediction on the differenced series.
    fn predict_diff_one(&self) -> f64 {
        let phi = self.build_regressor();
        let ar = &self.ar_params;
        let ma = &self.ma_params;
        let mut pred = 0.0;
        for (i, &a) in ar.iter().enumerate() {
            pred += a * phi.get(i).copied().unwrap_or(0.0);
        }
        for (i, &m) in ma.iter().enumerate() {
            pred += m * phi.get(self.p + i).copied().unwrap_or(0.0);
        }
        pred
    }

    /// Update the model with a new raw observation.
    ///
    /// Returns the one-step-ahead prediction made *before* incorporating `value`.
    pub fn update(&mut self, value: f64) -> f64 {
        self.n_obs += 1;
        self.history.push_back(value);

        // Compute d-th order differences up to latest point
        let diffs = Self::difference(&self.history, self.d);
        if let Some(&last_diff) = diffs.last() {
            self.diff_history.push_back(last_diff);
        }

        let dim = self.p + self.q;
        if dim == 0 || self.diff_history.len() < 2 {
            return value; // not enough data yet
        }

        let phi = self.build_regressor();
        // One-step prediction
        let y_pred_diff = self.predict_diff_one();

        // Actual differenced value
        let y_act_diff = self.diff_history.back().copied().unwrap_or(0.0);
        let error = y_act_diff - y_pred_diff;

        // RLS update: K = P·φ / (λ + φᵀ·P·φ)
        let p_phi: Vec<f64> = (0..dim)
            .map(|i| {
                (0..dim)
                    .map(|j| self.rls_p[i * dim + j] * phi[j])
                    .sum::<f64>()
            })
            .collect();
        let phi_p_phi: f64 = (0..dim).map(|j| phi[j] * p_phi[j]).sum::<f64>();
        let denom = self.forgetting_factor + phi_p_phi;
        let k: Vec<f64> = p_phi.iter().map(|&v| v / denom).collect();

        // Update parameters: θ ← θ + K·error
        for (i, coef) in self.ar_params.iter_mut().enumerate() {
            *coef += k[i] * error;
        }
        for (i, coef) in self.ma_params.iter_mut().enumerate() {
            *coef += k[self.p + i] * error;
        }

        // Update P: P ← (P - K·φᵀ·P) / λ
        // Correct P update: P ← (I - K φᵀ) P / λ
        // Compute K·φᵀ first (dim×dim)
        let mut k_phi_t = vec![0.0_f64; dim * dim];
        for i in 0..dim {
            for j in 0..dim {
                k_phi_t[i * dim + j] = k[i] * phi[j];
            }
        }
        let mut new_p = vec![0.0_f64; dim * dim];
        for i in 0..dim {
            for j in 0..dim {
                let row_sum: f64 = (0..dim)
                    .map(|l| {
                        let ikp = if i == l { 1.0 } else { 0.0 } - k_phi_t[i * dim + l];
                        ikp * self.rls_p[l * dim + j]
                    })
                    .sum();
                new_p[i * dim + j] = row_sum / self.forgetting_factor;
            }
        }
        self.rls_p = new_p;

        // Store error for MA computation
        self.errors.push_back(error);
        if self.errors.len() > self.q.max(1) + 10 {
            self.errors.pop_front();
        }

        // Keep history bounded
        let keep = (self.p + self.d + 2).max(20);
        while self.history.len() > keep {
            self.history.pop_front();
        }
        let keep_diff = (self.p + 2).max(20);
        while self.diff_history.len() > keep_diff {
            self.diff_history.pop_front();
        }

        // Return prediction made before update (in level terms)
        let raw: Vec<f64> = self.history.iter().cloned().collect();
        // Approximate level prediction by integrating diff prediction
        let last_levels: Vec<f64> = raw.iter().cloned().collect();
        let diff_fc = vec![y_pred_diff];
        let level_fc = Self::integrate(&diff_fc, &last_levels, self.d);
        level_fc.into_iter().next().unwrap_or(value)
    }

    /// Generate an h-step-ahead forecast from the current model state.
    ///
    /// Uses the current AR/MA parameters and recursively extends the history.
    pub fn predict(&self, h: usize) -> Vec<f64> {
        if h == 0 {
            return Vec::new();
        }

        let mut dh: Vec<f64> = self.diff_history.iter().cloned().collect();
        let mut errors: Vec<f64> = self.errors.iter().cloned().collect();
        let mut forecasts = Vec::with_capacity(h);

        for _ in 0..h {
            let len = dh.len();
            let mut phi = Vec::with_capacity(self.p + self.q);
            for i in 0..self.p {
                phi.push(if len > i { dh[len - 1 - i] } else { 0.0 });
            }
            let elen = errors.len();
            for i in 0..self.q {
                phi.push(if elen > i { errors[elen - 1 - i] } else { 0.0 });
            }
            let mut pred = 0.0;
            for (i, &a) in self.ar_params.iter().enumerate() {
                pred += a * phi.get(i).copied().unwrap_or(0.0);
            }
            for (i, &m) in self.ma_params.iter().enumerate() {
                pred += m * phi.get(self.p + i).copied().unwrap_or(0.0);
            }
            dh.push(pred);
            errors.push(0.0); // future errors unknown → 0
            forecasts.push(pred);
        }

        // Integrate to get level forecasts
        let raw: Vec<f64> = self.history.iter().cloned().collect();
        Self::integrate(&forecasts, &raw, self.d)
    }

    /// Number of observations processed so far.
    pub fn n_obs(&self) -> usize {
        self.n_obs
    }
}

// ─── Adaptive exponential smoothing ──────────────────────────────────────────

/// Adaptive single exponential smoothing.
///
/// The smoothing parameter α is updated after each observation using a
/// gradient-descent step on the one-step squared error, keeping α ∈ [α_min,
/// α_max].
///
/// # Example
/// ```
/// use scirs2_series::online::AdaptiveExpSmoothing;
/// let mut aes = AdaptiveExpSmoothing::new(0.3);
/// for i in 0..20 {
///     let y = i as f64;
///     let pred = aes.update(y);
///     assert!(pred.is_finite());
/// }
/// ```
#[derive(Debug, Clone)]
pub struct AdaptiveExpSmoothing {
    /// Current smoothing parameter α ∈ (0, 1)
    pub alpha: f64,
    level: f64,
    n_obs: usize,
    /// Gradient of the level w.r.t. α (for gradient-based α update)
    d_level_d_alpha: f64,
    /// Learning rate for α adaptation
    lr: f64,
    /// Minimum allowed α
    alpha_min: f64,
    /// Maximum allowed α
    alpha_max: f64,
}

impl AdaptiveExpSmoothing {
    /// Create a new adaptive exponential smoother.
    ///
    /// * `initial_alpha` – starting smoothing parameter ∈ (0, 1)
    pub fn new(initial_alpha: f64) -> Self {
        let alpha = initial_alpha.clamp(0.01, 0.99);
        Self {
            alpha,
            level: 0.0,
            n_obs: 0,
            d_level_d_alpha: 0.0,
            lr: 0.01,
            alpha_min: 0.01,
            alpha_max: 0.99,
        }
    }

    /// Set the learning rate for adaptive α updates (default 0.01).
    pub fn with_lr(mut self, lr: f64) -> Self {
        self.lr = lr.max(0.0);
        self
    }

    /// Set the allowed range for α.
    pub fn with_alpha_bounds(mut self, alpha_min: f64, alpha_max: f64) -> Self {
        self.alpha_min = alpha_min.clamp(1e-6, 1.0 - 1e-6);
        self.alpha_max = alpha_max.clamp(1e-6, 1.0 - 1e-6);
        self
    }

    /// Process a new observation.
    ///
    /// Returns the one-step prediction made **before** incorporating `value`
    /// (i.e., the prediction for time t using information up to t-1).
    pub fn update(&mut self, value: f64) -> f64 {
        self.n_obs += 1;

        if self.n_obs == 1 {
            // Initialise level to first observation
            self.level = value;
            self.d_level_d_alpha = 0.0;
            return value;
        }

        let pred = self.level; // prediction before update

        let error = value - pred;

        // Gradient of prediction error w.r.t. α
        // ∂ŷ_t/∂α = ∂L_{t-1}/∂α = (1 - α)·∂L_{t-2}/∂α + (y_{t-1} - L_{t-2})
        let d_pred_d_alpha = self.d_level_d_alpha;

        // Gradient descent update on α to minimise MSE
        // ∂(error²/2)/∂α = -error · ∂ŷ/∂α
        let grad = -error * d_pred_d_alpha;
        self.alpha = (self.alpha - self.lr * grad).clamp(self.alpha_min, self.alpha_max);

        // Update derivative for next step
        self.d_level_d_alpha = (1.0 - self.alpha) * self.d_level_d_alpha + (value - self.level);

        // Update level
        self.level = self.alpha * value + (1.0 - self.alpha) * self.level;

        pred
    }

    /// Current smoothed level (the one-step-ahead forecast).
    pub fn current_level(&self) -> f64 {
        self.level
    }

    /// Number of observations processed.
    pub fn n_obs(&self) -> usize {
        self.n_obs
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── PageHinkleyTest ─────────────────────────────────────────────────────

    #[test]
    fn test_ph_detects_increase() {
        let mut ph = PageHinkleyTest::new(0.05, 50.0);
        // Stable phase
        for _ in 0..200 {
            ph.update(0.0);
        }
        assert!(!ph.is_change_detected(), "should not alarm on stable data");

        // Shift phase
        let mut detected = false;
        for _ in 0..200 {
            if ph.update(10.0) {
                detected = true;
                break;
            }
        }
        assert!(detected, "PH should detect a large mean increase");
    }

    #[test]
    fn test_ph_no_false_alarm_constant() {
        let mut ph = PageHinkleyTest::new(0.0, 500.0);
        let mut alarms = 0u32;
        for _ in 0..1000 {
            if ph.update(1.0) {
                alarms += 1;
            }
        }
        // With constant input and very high threshold there should be no alarm
        assert_eq!(alarms, 0, "no alarm on perfectly constant input");
    }

    #[test]
    fn test_ph_reset() {
        let mut ph = PageHinkleyTest::new(0.05, 10.0);
        for _ in 0..500 {
            ph.update(5.0);
        }
        ph.reset();
        assert_eq!(ph.n_obs(), 0);
        assert!((ph.statistic()).abs() < 1e-10);
    }

    #[test]
    fn test_ph_statistic_non_negative() {
        let mut ph = PageHinkleyTest::new(0.1, 100.0);
        for i in 0..50 {
            ph.update(i as f64 * 0.1);
            assert!(ph.statistic() >= 0.0);
        }
    }

    // ── ADWIN ───────────────────────────────────────────────────────────────

    #[test]
    fn test_adwin_detects_step_change() {
        let mut adwin = Adwin::new(0.002);
        for _ in 0..300 {
            adwin.update(0.0);
        }
        let mut detected = false;
        for _ in 0..300 {
            if adwin.update(1.0) {
                detected = true;
                break;
            }
        }
        assert!(detected, "ADWIN should detect a step change from 0 to 1");
    }

    #[test]
    fn test_adwin_stable_window_does_not_shrink_excessively() {
        let mut adwin = Adwin::new(0.002);
        for _ in 0..100 {
            adwin.update(0.5);
        }
        // Window should still be reasonably large on stable data
        assert!(adwin.width() >= 10, "window width = {}", adwin.width());
    }

    #[test]
    fn test_adwin_mean_tracks_value() {
        let mut adwin = Adwin::new(0.05);
        for _ in 0..50 {
            adwin.update(3.0);
        }
        let m = adwin.mean();
        assert!((m - 3.0).abs() < 0.5, "mean should be near 3.0, got {m}");
    }

    #[test]
    fn test_adwin_width_non_zero_after_updates() {
        let mut adwin = Adwin::new(0.002);
        for i in 0..20 {
            adwin.update(i as f64);
        }
        assert!(adwin.width() > 0);
    }

    // ── OnlineArima ─────────────────────────────────────────────────────────

    #[test]
    fn test_online_arima_runs_without_panic() {
        let mut model = OnlineArima::new(2, 1, 1, 0.98);
        for i in 0..50 {
            let y = i as f64 + (i as f64 * 0.3).sin();
            let pred = model.update(y);
            assert!(pred.is_finite(), "prediction should be finite at step {i}");
        }
    }

    #[test]
    fn test_online_arima_forecast_length() {
        let mut model = OnlineArima::new(2, 1, 0, 0.99);
        for i in 0..30 {
            model.update(i as f64);
        }
        let fc = model.predict(5);
        assert_eq!(fc.len(), 5);
    }

    #[test]
    fn test_online_arima_forecast_finite() {
        let mut model = OnlineArima::new(1, 0, 0, 0.95);
        for i in 0..40 {
            model.update(2.0 * i as f64 + 0.1);
        }
        for &f in &model.predict(3) {
            assert!(f.is_finite());
        }
    }

    #[test]
    fn test_online_arima_zero_h_forecast() {
        let mut model = OnlineArima::new(1, 1, 0, 0.98);
        for i in 0..10 {
            model.update(i as f64);
        }
        assert!(model.predict(0).is_empty());
    }

    #[test]
    fn test_online_arima_n_obs_increments() {
        let mut model = OnlineArima::new(1, 0, 0, 0.99);
        assert_eq!(model.n_obs(), 0);
        model.update(1.0);
        assert_eq!(model.n_obs(), 1);
        model.update(2.0);
        assert_eq!(model.n_obs(), 2);
    }

    // ── AdaptiveExpSmoothing ─────────────────────────────────────────────────

    #[test]
    fn test_aes_runs_without_panic() {
        let mut aes = AdaptiveExpSmoothing::new(0.3);
        for i in 0..50 {
            let pred = aes.update(i as f64);
            assert!(pred.is_finite());
        }
    }

    #[test]
    fn test_aes_alpha_stays_in_bounds() {
        let mut aes = AdaptiveExpSmoothing::new(0.5)
            .with_alpha_bounds(0.01, 0.99)
            .with_lr(0.1);
        for i in 0..100 {
            let y = if i < 50 { 0.0 } else { 10.0 };
            aes.update(y);
            assert!(
                aes.alpha >= 0.01 && aes.alpha <= 0.99,
                "alpha out of bounds: {}",
                aes.alpha
            );
        }
    }

    #[test]
    fn test_aes_level_tracks_constant() {
        let mut aes = AdaptiveExpSmoothing::new(0.5);
        for _ in 0..100 {
            aes.update(7.0);
        }
        assert!(
            (aes.current_level() - 7.0).abs() < 0.5,
            "level should converge near 7.0, got {}",
            aes.current_level()
        );
    }

    #[test]
    fn test_aes_n_obs_increments() {
        let mut aes = AdaptiveExpSmoothing::new(0.2);
        assert_eq!(aes.n_obs(), 0);
        aes.update(1.0);
        assert_eq!(aes.n_obs(), 1);
        aes.update(2.0);
        assert_eq!(aes.n_obs(), 2);
    }

    #[test]
    fn test_aes_first_prediction_equals_first_value() {
        let mut aes = AdaptiveExpSmoothing::new(0.4);
        let pred = aes.update(5.0);
        assert!(
            (pred - 5.0).abs() < 1e-10,
            "first pred should equal first value"
        );
    }
}
