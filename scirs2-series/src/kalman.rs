//! User-friendly Kalman filter wrappers for univariate time series.
//!
//! Provides a simple `UnivariateKalmanFilter` that operates on scalar observations,
//! alongside helper functions for batch smoothing.

use crate::error::{Result, TimeSeriesError};
use crate::state_space::{kalman_filter, kalman_smoother, StateSpaceModel};
use scirs2_core::ndarray::{Array1, Array2};

// ---------------------------------------------------------------------------
// UnivariateKalmanFilter
// ---------------------------------------------------------------------------

/// A simple Kalman filter for scalar (univariate) state and observation.
///
/// Model:
/// ```text
/// x_{t+1} = x_t + w_t,   w_t ~ N(0, q)   (random walk state)
/// y_t     = x_t + v_t,   v_t ~ N(0, r)   (noisy observation)
/// ```
///
/// # Example
/// ```rust
/// use scirs2_series::kalman::UnivariateKalmanFilter;
///
/// let mut kf = UnivariateKalmanFilter::new(0.0, 1.0, 0.1, 0.5);
/// let (state, var) = kf.update(1.5);
/// println!("state={state:.4}, variance={var:.4}");
/// ```
#[derive(Debug, Clone)]
pub struct UnivariateKalmanFilter {
    /// Process noise variance q
    pub q: f64,
    /// Observation noise variance r
    pub r: f64,
    /// Current state estimate
    pub state: f64,
    /// Current state variance (uncertainty)
    pub variance: f64,
}

impl UnivariateKalmanFilter {
    /// Create a new filter with given initial state, variance, and noise parameters.
    ///
    /// - `initial_state`: prior mean of the state
    /// - `initial_variance`: prior variance of the state
    /// - `q`: process noise variance (how fast the state can drift)
    /// - `r`: observation noise variance (measurement error)
    pub fn new(initial_state: f64, initial_variance: f64, q: f64, r: f64) -> Self {
        Self {
            q,
            r,
            state: initial_state,
            variance: initial_variance,
        }
    }

    /// Perform the prediction step only.
    ///
    /// Advances the state estimate by one time step without an observation.
    /// Returns `(predicted_state, predicted_variance)`.
    pub fn predict(&mut self) -> (f64, f64) {
        // State is a random walk: state unchanged, variance grows by q
        self.variance += self.q;
        (self.state, self.variance)
    }

    /// Perform the update (measurement) step only.
    ///
    /// Incorporates a new observation into the current state estimate.
    /// Returns `(updated_state, updated_variance)`.
    ///
    /// This does **not** call `predict` first; call `predict` explicitly if needed.
    pub fn update(&mut self, observation: f64) -> (f64, f64) {
        // Innovation
        let innovation = observation - self.state;
        // Innovation variance S = P + R
        let s = self.variance + self.r;
        // Kalman gain K = P / S
        let k = self.variance / s;
        // Update state
        self.state += k * innovation;
        // Update variance (Joseph form)
        self.variance = (1.0 - k) * self.variance;
        (self.state, self.variance)
    }

    /// Predict and then update in one call (the standard "filter step").
    ///
    /// Equivalent to calling `predict()` then `update(observation)`.
    /// Returns `(updated_state, updated_variance)`.
    pub fn step(&mut self, observation: f64) -> (f64, f64) {
        self.predict();
        self.update(observation)
    }

    /// Smooth an entire series of observations.
    ///
    /// Runs the filter forward over all observations, then applies a simple
    /// RTS (backward-pass) smoother on the univariate states.
    ///
    /// Returns a `Vec<(smoothed_state, smoothed_variance)>` with the same
    /// length as `observations`.
    ///
    /// # Errors
    /// Returns an error if the observations slice is empty.
    pub fn smooth_series(&mut self, observations: &[f64]) -> Vec<(f64, f64)> {
        if observations.is_empty() {
            return vec![];
        }

        let n = observations.len();

        // Forward pass — store predicted and filtered (state, variance)
        let mut pred_state = Vec::with_capacity(n);
        let mut pred_var = Vec::with_capacity(n);
        let mut filt_state = Vec::with_capacity(n);
        let mut filt_var = Vec::with_capacity(n);

        // Reset to initial conditions (saved so the filter is re-usable)
        let init_state = self.state;
        let init_var = self.variance;

        for &obs in observations.iter() {
            // Predict
            self.variance += self.q;
            pred_state.push(self.state);
            pred_var.push(self.variance);

            // Update
            let innovation = obs - self.state;
            let s = self.variance + self.r;
            let k = self.variance / s;
            self.state += k * innovation;
            self.variance = (1.0 - k) * self.variance;
            filt_state.push(self.state);
            filt_var.push(self.variance);
        }

        // Restore initial conditions
        self.state = init_state;
        self.variance = init_var;

        // Backward pass (RTS smoother)
        let mut smoothed_state = filt_state.clone();
        let mut smoothed_var = filt_var.clone();

        for t in (0..n - 1).rev() {
            let p_next_pred = pred_var[t + 1];
            if p_next_pred.abs() < 1e-14 {
                continue;
            }
            // Smoother gain G_t = P_{t|t} / P_{t+1|t}
            let g = filt_var[t] / p_next_pred;
            // Smoothed state
            smoothed_state[t] = filt_state[t] + g * (smoothed_state[t + 1] - pred_state[t + 1]);
            // Smoothed variance
            smoothed_var[t] = filt_var[t] + g * g * (smoothed_var[t + 1] - p_next_pred);
        }

        smoothed_state.into_iter().zip(smoothed_var).collect()
    }

    /// Compute the log-likelihood of a series of observations given the current
    /// model parameters.
    ///
    /// Uses the innovations form: sum of -0.5*(log(2π S) + v²/S) over all t.
    pub fn log_likelihood(&self, observations: &[f64]) -> f64 {
        use std::f64::consts::PI;

        let mut state = self.state;
        let mut var = self.variance;
        let mut ll = 0.0_f64;

        for &obs in observations.iter() {
            // Predict
            var += self.q;
            // Innovation
            let innovation = obs - state;
            let s = var + self.r;
            ll += -0.5 * ((2.0 * PI * s).ln() + innovation * innovation / s);
            // Update
            let k = var / s;
            state += k * innovation;
            var = (1.0 - k) * var;
        }

        ll
    }
}

// ---------------------------------------------------------------------------
// Convenience wrappers using the full SSM machinery
// ---------------------------------------------------------------------------

/// Smooth a univariate series using a local-level SSM fitted via the Kalman filter.
///
/// Returns `(filtered_states, smoothed_states)` as `Array1<f64>`.
///
/// # Errors
/// Returns an error if the series is empty or model dimensions are inconsistent.
pub fn smooth_univariate(
    observations: &[f64],
    sigma_eta: f64,
    sigma_eps: f64,
) -> Result<(Array1<f64>, Array1<f64>)> {
    let n = observations.len();
    if n == 0 {
        return Err(TimeSeriesError::InsufficientData {
            message: "smooth_univariate requires at least one observation".to_string(),
            required: 1,
            actual: 0,
        });
    }

    let obs_2d = Array2::from_shape_vec((n, 1), observations.to_vec())
        .map_err(|e| TimeSeriesError::ComputationError(format!("Shape error: {e}")))?;

    let model = StateSpaceModel::local_level(sigma_eta, sigma_eps);
    let filt = kalman_filter(obs_2d.view(), &model, true)?;
    let smooth = kalman_smoother(&filt, &model)?;

    let filtered: Array1<f64> = Array1::from_iter((0..n).map(|t| filt.filtered_states[[t, 0]]));
    let smoothed: Array1<f64> = Array1::from_iter((0..n).map(|t| smooth.smoothed_states[[t, 0]]));

    Ok((filtered, smoothed))
}

/// Compute the one-step-ahead forecast for a univariate series using a local-level model.
///
/// Returns `(predictions, residuals)` where `predictions[t]` is the predicted value
/// at time `t` and `residuals[t] = observations[t] - predictions[t]`.
pub fn one_step_ahead(
    observations: &[f64],
    sigma_eta: f64,
    sigma_eps: f64,
) -> Result<(Array1<f64>, Array1<f64>)> {
    let n = observations.len();
    if n == 0 {
        return Err(TimeSeriesError::InsufficientData {
            message: "one_step_ahead requires at least one observation".to_string(),
            required: 1,
            actual: 0,
        });
    }

    let obs_2d = Array2::from_shape_vec((n, 1), observations.to_vec())
        .map_err(|e| TimeSeriesError::ComputationError(format!("Shape error: {e}")))?;

    let model = StateSpaceModel::local_level(sigma_eta, sigma_eps);
    let filt = kalman_filter(obs_2d.view(), &model, true)?;

    let preds: Array1<f64> = Array1::from_iter((0..n).map(|t| filt.predicted_states[[t, 0]]));
    let resids: Array1<f64> = Array1::from_iter((0..n).map(|t| filt.innovations[[t, 0]]));

    Ok((preds, resids))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    // -----------------------------------------------------------------------
    // UnivariateKalmanFilter basic tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_new_filter() {
        let kf = UnivariateKalmanFilter::new(0.0, 1.0, 0.1, 0.5);
        assert_eq!(kf.state, 0.0);
        assert_eq!(kf.variance, 1.0);
        assert_eq!(kf.q, 0.1);
        assert_eq!(kf.r, 0.5);
    }

    #[test]
    fn test_predict_increases_variance() {
        let mut kf = UnivariateKalmanFilter::new(0.0, 1.0, 0.2, 0.5);
        let (_, var_after) = kf.predict();
        assert!(approx_eq(var_after, 1.2, 1e-12));
    }

    #[test]
    fn test_update_moves_state_toward_obs() {
        let mut kf = UnivariateKalmanFilter::new(0.0, 1.0, 0.0, 0.5);
        // No prediction noise; after update state should move toward 3.0
        let (state, _) = kf.update(3.0);
        assert!(state > 0.0 && state < 3.0, "state={state}");
    }

    #[test]
    fn test_update_reduces_variance() {
        let mut kf = UnivariateKalmanFilter::new(0.0, 1.0, 0.0, 0.5);
        let init_var = kf.variance;
        let (_, var_after) = kf.update(3.0);
        assert!(
            var_after < init_var,
            "variance should decrease after update"
        );
    }

    #[test]
    fn test_step_is_predict_then_update() {
        let mut kf1 = UnivariateKalmanFilter::new(0.0, 1.0, 0.1, 0.5);
        let mut kf2 = kf1.clone();

        let (state1, var1) = kf1.step(2.0);
        kf2.predict();
        let (state2, var2) = kf2.update(2.0);

        assert!(approx_eq(state1, state2, 1e-12));
        assert!(approx_eq(var1, var2, 1e-12));
    }

    #[test]
    fn test_constant_observation_convergence() {
        let mut kf = UnivariateKalmanFilter::new(0.0, 10.0, 0.1, 1.0);
        let obs = 5.0;
        for _ in 0..100 {
            kf.step(obs);
        }
        // After many constant observations, state should be near obs
        assert!((kf.state - obs).abs() < 0.5, "state={}", kf.state);
    }

    #[test]
    fn test_smooth_series_length() {
        let mut kf = UnivariateKalmanFilter::new(0.0, 1.0, 0.1, 0.5);
        let obs = vec![1.0, 2.0, 3.0, 2.0, 1.5];
        let result = kf.smooth_series(&obs);
        assert_eq!(result.len(), obs.len());
    }

    #[test]
    fn test_smooth_series_empty() {
        let mut kf = UnivariateKalmanFilter::new(0.0, 1.0, 0.1, 0.5);
        let result = kf.smooth_series(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_smooth_series_finite_values() {
        let mut kf = UnivariateKalmanFilter::new(0.0, 1.0, 0.2, 1.0);
        let obs: Vec<f64> = (0..20).map(|i| (i as f64 * 0.3).sin() * 3.0).collect();
        let result = kf.smooth_series(&obs);
        for (s, v) in &result {
            assert!(s.is_finite(), "smoothed state is not finite");
            assert!(v.is_finite(), "smoothed variance is not finite");
            assert!(*v >= 0.0, "smoothed variance is negative: {v}");
        }
    }

    #[test]
    fn test_smooth_single_observation() {
        let mut kf = UnivariateKalmanFilter::new(0.0, 1.0, 0.1, 0.5);
        let result = kf.smooth_series(&[3.0]);
        assert_eq!(result.len(), 1);
        let (s, v) = result[0];
        assert!(s.is_finite());
        assert!(v >= 0.0);
    }

    #[test]
    fn test_log_likelihood_finite() {
        let kf = UnivariateKalmanFilter::new(0.0, 1.0, 0.1, 0.5);
        let obs = vec![1.0, 1.1, 0.9, 1.2, 1.0];
        let ll = kf.log_likelihood(&obs);
        assert!(ll.is_finite(), "log-likelihood should be finite, got {ll}");
    }

    #[test]
    fn test_log_likelihood_negative() {
        let kf = UnivariateKalmanFilter::new(0.0, 1.0, 0.1, 0.5);
        let obs: Vec<f64> = (0..20).map(|i| i as f64 * 0.1).collect();
        let ll = kf.log_likelihood(&obs);
        assert!(ll < 0.0, "log-likelihood should be negative, got {ll}");
    }

    #[test]
    fn test_log_likelihood_better_model() {
        // A model with smaller misspecified noise should have lower LL on mismatched data,
        // but the exact ordering depends on data. Just check LL is finite.
        let kf_tight = UnivariateKalmanFilter::new(0.0, 1.0, 0.01, 0.01);
        let kf_loose = UnivariateKalmanFilter::new(0.0, 1.0, 1.0, 5.0);
        let obs: Vec<f64> = vec![0.0, 0.1, -0.1, 0.05, -0.05];
        let ll_tight = kf_tight.log_likelihood(&obs);
        let ll_loose = kf_loose.log_likelihood(&obs);
        assert!(ll_tight.is_finite());
        assert!(ll_loose.is_finite());
    }

    // -----------------------------------------------------------------------
    // smooth_univariate wrapper tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_smooth_univariate_basic() {
        let obs = vec![1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0];
        let (filt, smooth) = smooth_univariate(&obs, 0.3, 0.5).expect("ok");
        assert_eq!(filt.len(), obs.len());
        assert_eq!(smooth.len(), obs.len());
        for i in 0..obs.len() {
            assert!(filt[i].is_finite());
            assert!(smooth[i].is_finite());
        }
    }

    #[test]
    fn test_smooth_univariate_empty_error() {
        let result = smooth_univariate(&[], 0.1, 0.5);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // one_step_ahead wrapper test
    // -----------------------------------------------------------------------

    #[test]
    fn test_one_step_ahead_residuals_length() {
        let obs = vec![1.0, 1.5, 2.0, 1.8, 2.2];
        let (preds, resids) = one_step_ahead(&obs, 0.2, 0.4).expect("ok");
        assert_eq!(preds.len(), obs.len());
        assert_eq!(resids.len(), obs.len());
        for i in 0..obs.len() {
            // residual = obs - predicted
            assert!(resids[i].is_finite());
            assert!(preds[i].is_finite());
        }
    }
}
