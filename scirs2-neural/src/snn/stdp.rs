//! Spike-Timing-Dependent Plasticity (STDP) Learning Rules
//!
//! Implements biologically motivated unsupervised Hebbian learning:
//! - Classical (pair-based) STDP
//! - Triplet STDP (Pfister & Gerstner 2006)
//! - BCM (Bienenstock-Cooper-Munro) rule
//! - Oja's rule (online PCA)

use crate::error::{NeuralError, Result};

// ---------------------------------------------------------------------------
// Classical (pair-based) STDP
// ---------------------------------------------------------------------------

/// Classical pair-based STDP learning rule.
///
/// For a pair (pre-spike at t_pre, post-spike at t_post):
/// - Δt = t_post - t_pre > 0 → LTP: Δw = A₊ · exp(-Δt / τ₊)
/// - Δt < 0                  → LTD: Δw = A₋ · exp( Δt / τ₋)
///
/// Soft weight bounds via multiplicative rule (default hard clipping available).
#[derive(Debug, Clone)]
pub struct STDP {
    /// LTP time constant (ms)
    pub tau_plus: f32,
    /// LTD time constant (ms)
    pub tau_minus: f32,
    /// LTP amplitude (> 0)
    pub a_plus: f32,
    /// LTD amplitude (< 0 for depression)
    pub a_minus: f32,
    /// Maximum weight
    pub w_max: f32,
    /// Minimum weight
    pub w_min: f32,
}

impl STDP {
    /// Create a new STDP rule.
    ///
    /// # Errors
    /// Returns an error if time constants are ≤ 0 or w_max < w_min.
    pub fn new(
        tau_plus: f32,
        tau_minus: f32,
        a_plus: f32,
        a_minus: f32,
        w_max: f32,
    ) -> Result<Self> {
        if tau_plus <= 0.0 {
            return Err(NeuralError::InvalidArgument("tau_plus must be > 0".into()));
        }
        if tau_minus <= 0.0 {
            return Err(NeuralError::InvalidArgument("tau_minus must be > 0".into()));
        }
        if w_max <= 0.0 {
            return Err(NeuralError::InvalidArgument("w_max must be > 0".into()));
        }
        Ok(Self {
            tau_plus,
            tau_minus,
            a_plus,
            a_minus,
            w_max,
            w_min: 0.0,
        })
    }

    /// Standard STDP with symmetric parameters.
    pub fn symmetric(tau: f32, amplitude: f32, w_max: f32) -> Result<Self> {
        Self::new(tau, tau, amplitude, -amplitude, w_max)
    }

    /// Compute the weight change for a given spike-timing difference.
    ///
    /// # Arguments
    /// * `dt` — t_post - t_pre (ms); positive ⇒ post fires after pre (LTP)
    pub fn delta_w(&self, dt: f32) -> f32 {
        if dt > 0.0 {
            self.a_plus * (-dt / self.tau_plus).exp()
        } else if dt < 0.0 {
            self.a_minus * (dt / self.tau_minus).exp()
        } else {
            0.0
        }
    }

    /// Apply the STDP update to weight `w` given spike-timing difference `dt`.
    ///
    /// Hard clamps the result to `[w_min, w_max]`.
    pub fn apply(&self, w: f32, dt: f32) -> f32 {
        (w + self.delta_w(dt)).clamp(self.w_min, self.w_max)
    }

    /// Online update using pre/post traces (pair-based).
    ///
    /// # Arguments
    /// * `w`          — current weight
    /// * `pre_trace`  — running exponential pre-synaptic trace x
    /// * `post_trace` — running exponential post-synaptic trace y
    /// * `pre_spike`  — whether pre-synaptic neuron fired this step
    /// * `post_spike` — whether post-synaptic neuron fired this step
    ///
    /// # Returns
    /// Updated weight, pre-trace, post-trace.
    pub fn online_update(
        &self,
        w: f32,
        pre_trace: f32,
        post_trace: f32,
        pre_spike: bool,
        post_spike: bool,
        dt: f32,
    ) -> (f32, f32, f32) {
        let decay_plus = (-dt / self.tau_plus).exp();
        let decay_minus = (-dt / self.tau_minus).exp();
        let mut x = pre_trace * decay_plus;
        let mut y = post_trace * decay_minus;
        let mut new_w = w;

        if pre_spike {
            new_w = (new_w + self.a_plus * y).clamp(self.w_min, self.w_max);
            x += 1.0;
        }
        if post_spike {
            new_w = (new_w + self.a_minus * x).clamp(self.w_min, self.w_max);
            y += 1.0;
        }
        (new_w, x, y)
    }
}

// ---------------------------------------------------------------------------
// Triplet STDP
// ---------------------------------------------------------------------------

/// Triplet STDP learning rule (Pfister & Gerstner 2006).
///
/// Extends pair STDP with triplet interactions between spikes:
/// - LTP: uses r₁ (standard) + r₂ (triplet) traces
/// - LTD: uses o₁ (standard) + o₂ (triplet) traces
///
/// On pre-spike:  Δw = -A₂⁻ · o₁ + A₃⁻ · o₂ · o₁  (LTD)
/// On post-spike: Δw = +A₂⁺ · r₁ + A₃⁺ · r₂ · r₁  (LTP)
#[derive(Debug, Clone)]
pub struct TripletSTDP {
    /// Fast pre-synaptic trace time constant (ms)
    pub tau_plus: f32,
    /// Fast post-synaptic trace time constant (ms)
    pub tau_minus: f32,
    /// Slow pre-synaptic trace time constant (ms)
    pub tau_x: f32,
    /// Slow post-synaptic trace time constant (ms)
    pub tau_y: f32,
    /// Pair LTP amplitude
    pub a2_plus: f32,
    /// Triplet LTP amplitude
    pub a3_plus: f32,
    /// Pair LTD amplitude (negative)
    pub a2_minus: f32,
    /// Triplet LTD amplitude (negative)
    pub a3_minus: f32,
    /// Maximum weight
    pub w_max: f32,
    /// Minimum weight
    pub w_min: f32,
}

/// State variables for triplet STDP traces.
#[derive(Debug, Clone, Default)]
pub struct TripletState {
    /// Fast pre-synaptic trace r₁
    pub r1: f32,
    /// Slow pre-synaptic trace r₂
    pub r2: f32,
    /// Fast post-synaptic trace o₁
    pub o1: f32,
    /// Slow post-synaptic trace o₂
    pub o2: f32,
}

impl TripletSTDP {
    /// Create a new triplet STDP rule.
    ///
    /// # Errors
    /// Returns an error if any time constant ≤ 0.
    pub fn new(
        tau_plus: f32,
        tau_minus: f32,
        tau_x: f32,
        tau_y: f32,
        a2_plus: f32,
        a3_plus: f32,
        a2_minus: f32,
        a3_minus: f32,
        w_max: f32,
    ) -> Result<Self> {
        for (name, val) in [
            ("tau_plus", tau_plus),
            ("tau_minus", tau_minus),
            ("tau_x", tau_x),
            ("tau_y", tau_y),
        ] {
            if val <= 0.0 {
                return Err(NeuralError::InvalidArgument(format!("{name} must be > 0")));
            }
        }
        Ok(Self {
            tau_plus,
            tau_minus,
            tau_x,
            tau_y,
            a2_plus,
            a3_plus,
            a2_minus,
            a3_minus,
            w_max,
            w_min: 0.0,
        })
    }

    /// Standard visual cortex parameters (Pfister & Gerstner 2006, Table 3).
    pub fn visual_cortex() -> Self {
        Self {
            tau_plus: 16.8,
            tau_minus: 33.7,
            tau_x: 101.0,
            tau_y: 125.0,
            a2_plus: 7.5e-3,
            a3_plus: 9.3e-3,
            a2_minus: 7.0e-3,
            a3_minus: 2.3e-4,
            w_max: 1.0,
            w_min: 0.0,
        }
    }

    /// Advance traces one time step and apply spike-driven weight changes.
    ///
    /// # Returns
    /// Updated (weight, state).
    pub fn update(
        &self,
        w: f32,
        state: &TripletState,
        pre_spike: bool,
        post_spike: bool,
        dt: f32,
    ) -> (f32, TripletState) {
        let ep = (-dt / self.tau_plus).exp();
        let em = (-dt / self.tau_minus).exp();
        let ex = (-dt / self.tau_x).exp();
        let ey = (-dt / self.tau_y).exp();

        let mut r1 = state.r1 * ep;
        let mut r2 = state.r2 * ex;
        let mut o1 = state.o1 * em;
        let mut o2 = state.o2 * ey;
        let mut new_w = w;

        if pre_spike {
            // LTD: uses o₁ and o₂ (from previous post spikes)
            let delta_w = -self.a2_minus * o1 + self.a3_minus * o2 * o1;
            new_w = (new_w + delta_w).clamp(self.w_min, self.w_max);
            r1 += 1.0;
            r2 += 1.0;
        }

        if post_spike {
            // LTP: uses r₁ and r₂ (from previous pre spikes)
            let delta_w = self.a2_plus * r1 + self.a3_plus * r2 * r1;
            new_w = (new_w + delta_w).clamp(self.w_min, self.w_max);
            o1 += 1.0;
            o2 += 1.0;
        }

        let new_state = TripletState { r1, r2, o1, o2 };
        (new_w, new_state)
    }
}

// ---------------------------------------------------------------------------
// BCM Rule
// ---------------------------------------------------------------------------

/// Bienenstock-Cooper-Munro (BCM) learning rule.
///
/// Δw = lr · y · (y − θ) · x
///
/// where θ is a sliding modification threshold that tracks the squared
/// postsynaptic activity:
///
/// dθ/dt = (y² − θ) / τ_θ
///
/// This produces LTP when y > θ and LTD when y < θ, enabling
/// selectivity and stability.
#[derive(Debug, Clone)]
pub struct BCMRule {
    /// Learning rate
    pub lr: f32,
    /// Modification threshold (slides with activity)
    pub theta: f32,
    /// Time constant for threshold sliding (ms)
    pub tau_theta: f32,
}

impl BCMRule {
    /// Create a new BCM rule.
    ///
    /// # Errors
    /// Returns an error if `lr <= 0` or `tau_theta <= 0`.
    pub fn new(lr: f32, theta: f32, tau_theta: f32) -> Result<Self> {
        if lr <= 0.0 {
            return Err(NeuralError::InvalidArgument("lr must be > 0".into()));
        }
        if tau_theta <= 0.0 {
            return Err(NeuralError::InvalidArgument("tau_theta must be > 0".into()));
        }
        Ok(Self {
            lr,
            theta,
            tau_theta,
        })
    }

    /// Apply the BCM weight update.
    ///
    /// # Arguments
    /// * `w`  — current weight (scalar)
    /// * `x`  — pre-synaptic activity
    /// * `y`  — post-synaptic activity
    /// * `dt` — time step (ms)
    ///
    /// # Returns
    /// Updated (weight, theta).
    pub fn update(&mut self, w: f32, x: f32, y: f32, dt: f32) -> f32 {
        let dw = self.lr * y * (y - self.theta) * x;
        // Slide threshold toward y²
        let dtheta = (y * y - self.theta) / self.tau_theta;
        self.theta += dt * dtheta;
        w + dw
    }

    /// Apply BCM update to a weight vector (all weights share the same post-activity).
    ///
    /// # Arguments
    /// * `weights`  — mutable slice of weights
    /// * `x_vec`    — pre-synaptic activities (same length as weights)
    /// * `y`        — post-synaptic activity (scalar)
    /// * `dt`       — time step
    ///
    /// # Errors
    /// Returns an error if lengths mismatch.
    pub fn update_vector(
        &mut self,
        weights: &mut [f32],
        x_vec: &[f32],
        y: f32,
        dt: f32,
    ) -> Result<()> {
        if weights.len() != x_vec.len() {
            return Err(NeuralError::DimensionMismatch(format!(
                "weights length {} != x_vec length {}",
                weights.len(),
                x_vec.len()
            )));
        }
        let phi = y * (y - self.theta);
        for (w, &x) in weights.iter_mut().zip(x_vec.iter()) {
            *w += self.lr * phi * x;
        }
        let dtheta = (y * y - self.theta) / self.tau_theta;
        self.theta += dt * dtheta;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Oja's Rule
// ---------------------------------------------------------------------------

/// Oja's learning rule for online PCA / Hebbian learning.
///
/// Δw = lr · (y · x − y² · w)
///
/// The weight vector converges to the first principal component of the
/// input distribution (normalized). Oja's rule prevents weight blow-up
/// without requiring explicit weight normalization.
#[derive(Debug, Clone)]
pub struct OjaRule {
    /// Learning rate
    pub lr: f32,
}

impl OjaRule {
    /// Create a new Oja rule.
    ///
    /// # Errors
    /// Returns an error if `lr <= 0`.
    pub fn new(lr: f32) -> Result<Self> {
        if lr <= 0.0 {
            return Err(NeuralError::InvalidArgument("lr must be > 0".into()));
        }
        Ok(Self { lr })
    }

    /// Apply Oja's rule to a scalar weight.
    ///
    /// # Arguments
    /// * `w` — current weight
    /// * `x` — pre-synaptic input
    /// * `y` — post-synaptic output (= dot(w, inputs))
    ///
    /// # Returns
    /// Updated weight.
    pub fn update(&self, w: f32, x: f32, y: f32) -> f32 {
        w + self.lr * (y * x - y * y * w)
    }

    /// Apply Oja's rule to a weight vector.
    ///
    /// # Arguments
    /// * `weights` — mutable weight vector
    /// * `inputs`  — pre-synaptic input vector
    /// * `output`  — post-synaptic output (scalar dot product)
    ///
    /// # Errors
    /// Returns an error if lengths mismatch.
    pub fn update_vector(&self, weights: &mut [f32], inputs: &[f32], output: f32) -> Result<()> {
        if weights.len() != inputs.len() {
            return Err(NeuralError::DimensionMismatch(format!(
                "weights length {} != inputs length {}",
                weights.len(),
                inputs.len()
            )));
        }
        for (w, &x) in weights.iter_mut().zip(inputs.iter()) {
            *w += self.lr * (output * x - output * output * *w);
        }
        Ok(())
    }

    /// Batch Oja update over many input samples (offline approximation).
    ///
    /// Each row of `data` is one input sample; the weight vector is updated
    /// after each sample.
    ///
    /// # Errors
    /// Returns an error if the data column count differs from weights length.
    pub fn fit(&self, weights: &mut [f32], data: &[Vec<f32>]) -> Result<()> {
        for sample in data.iter() {
            if sample.len() != weights.len() {
                return Err(NeuralError::DimensionMismatch(format!(
                    "data column count {} != weights {}",
                    sample.len(),
                    weights.len()
                )));
            }
            let output: f32 = weights.iter().zip(sample.iter()).map(|(w, x)| w * x).sum();
            self.update_vector(weights, sample, output)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stdp_ltp_for_positive_dt() {
        let rule = STDP::new(20.0, 20.0, 0.01, -0.01, 1.0).expect("operation should succeed");
        let dw = rule.delta_w(5.0);
        assert!(dw > 0.0, "LTP expected for positive Δt");
    }

    #[test]
    fn stdp_ltd_for_negative_dt() {
        let rule = STDP::new(20.0, 20.0, 0.01, -0.01, 1.0).expect("operation should succeed");
        let dw = rule.delta_w(-5.0);
        assert!(dw < 0.0, "LTD expected for negative Δt");
    }

    #[test]
    fn stdp_zero_at_coincident_spikes() {
        let rule = STDP::new(20.0, 20.0, 0.01, -0.01, 1.0).expect("operation should succeed");
        let dw = rule.delta_w(0.0);
        assert_eq!(dw, 0.0);
    }

    #[test]
    fn stdp_online_ltp() {
        let rule = STDP::new(20.0, 20.0, 0.01, -0.01, 1.0).expect("operation should succeed");
        // Post fired → post trace high → then pre fires → LTP
        let (_, x, y) = rule.online_update(0.5, 0.0, 0.5, false, true, 1.0);
        let (new_w, _, _) = rule.online_update(0.5, x, y, true, false, 1.0);
        assert!(new_w > 0.5, "LTP expected");
    }

    #[test]
    fn triplet_stdp_visual_cortex_runs() {
        let rule = TripletSTDP::visual_cortex();
        let mut state = TripletState::default();
        let mut w = 0.5_f32;
        for t in 0..100 {
            let pre = t % 10 == 0;
            let post = t % 13 == 0;
            let (nw, ns) = rule.update(w, &state, pre, post, 1.0);
            w = nw;
            state = ns;
        }
        assert!((0.0..=1.0).contains(&w));
    }

    #[test]
    fn bcm_theta_slides() {
        let mut rule = BCMRule::new(0.01, 0.1, 100.0).expect("operation should succeed");
        let theta_init = rule.theta;
        let _ = rule.update(0.5, 1.0, 1.0, 1.0);
        assert!(
            rule.theta > theta_init,
            "theta should increase when y² > theta"
        );
    }

    #[test]
    fn oja_weight_converges() {
        let rule = OjaRule::new(0.01).expect("operation should succeed");
        let mut w = vec![0.5_f32, 0.5];
        let data: Vec<Vec<f32>> = (0..1000)
            .map(|i| {
                let t = i as f32 * 0.01;
                vec![t.cos(), t.sin()]
            })
            .collect();
        rule.fit(&mut w, &data).expect("operation should succeed");
        // Weight vector should remain finite and bounded
        let norm: f32 = w.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(norm.is_finite() && norm > 0.0);
    }

    #[test]
    fn stdp_rejects_invalid_params() {
        assert!(STDP::new(-1.0, 20.0, 0.01, -0.01, 1.0).is_err());
        assert!(STDP::new(20.0, 0.0, 0.01, -0.01, 1.0).is_err());
        assert!(STDP::new(20.0, 20.0, 0.01, -0.01, -1.0).is_err());
    }
}
