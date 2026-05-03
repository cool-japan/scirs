//! Synaptic Models for Spiking Neural Networks
//!
//! Provides:
//! - `ExponentialSynapse` — single-exponential conductance synapse
//! - `AlphaSynapse`       — double-exponential (alpha function) synapse
//! - `STDPSynapse`        — synapse with online STDP weight update
//! - `SynapticDelay`      — fixed axonal delay via ring buffer

use crate::error::{NeuralError, Result};
use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// ExponentialSynapse
// ---------------------------------------------------------------------------

/// Single-exponential conductance synapse.
///
/// The synaptic conductance decays as:
/// ```text
/// dg/dt = -g / τ
/// ```
/// and jumps by `weight` on each pre-synaptic spike.
#[derive(Debug, Clone)]
pub struct ExponentialSynapse {
    /// Decay time constant (ms)
    pub tau: f32,
    /// Synaptic weight (conductance increase per spike, nS)
    pub weight: f32,
    /// Current conductance (nS)
    pub g: f32,
    /// Reversal potential (mV)
    pub e_rev: f32,
}

impl ExponentialSynapse {
    /// Create a new exponential synapse.
    ///
    /// # Errors
    /// Returns an error if `tau <= 0`.
    pub fn new(tau: f32, weight: f32, e_rev: f32) -> Result<Self> {
        if tau <= 0.0 {
            return Err(NeuralError::InvalidArgument(format!(
                "tau must be > 0, got {tau}"
            )));
        }
        Ok(Self {
            tau,
            weight,
            g: 0.0,
            e_rev,
        })
    }

    /// Create an excitatory AMPA-like synapse (τ = 5 ms, E_rev = 0 mV).
    pub fn ampa(weight: f32) -> Self {
        Self {
            tau: 5.0,
            weight,
            g: 0.0,
            e_rev: 0.0,
        }
    }

    /// Create an inhibitory GABA-A-like synapse (τ = 10 ms, E_rev = -70 mV).
    pub fn gaba_a(weight: f32) -> Self {
        Self {
            tau: 10.0,
            weight,
            g: 0.0,
            e_rev: -70.0,
        }
    }

    /// Advance the synapse by `dt` ms and optionally receive a pre-synaptic spike.
    ///
    /// # Returns
    /// Synaptic current contribution I = g * (V_post - E_rev).
    /// The caller supplies the post-synaptic potential `v_post`.
    pub fn update(&mut self, spike: bool, dt: f32) -> f32 {
        self.g *= (-dt / self.tau).exp();
        if spike {
            self.g += self.weight;
        }
        self.g
    }

    /// Compute instantaneous synaptic current given post-synaptic potential.
    pub fn current(&self, v_post: f32) -> f32 {
        self.g * (v_post - self.e_rev)
    }
}

// ---------------------------------------------------------------------------
// AlphaSynapse
// ---------------------------------------------------------------------------

/// Double-exponential (alpha-function) synapse.
///
/// Models the rise and decay of synaptic conductance more realistically:
/// ```text
/// dx/dt = -x / τ_rise
/// dg/dt = -g / τ_decay + x
/// ```
/// On a pre-synaptic spike: x += weight
///
/// The alpha function peaks at t = τ_rise · τ_decay / (τ_decay − τ_rise) · ln(τ_decay/τ_rise).
#[derive(Debug, Clone)]
pub struct AlphaSynapse {
    /// Rise time constant (ms)
    pub tau_rise: f32,
    /// Decay time constant (ms)
    pub tau_decay: f32,
    /// Synaptic weight
    pub weight: f32,
    /// Reversal potential (mV)
    pub e_rev: f32,
    /// Rise component
    pub x: f32,
    /// Conductance
    pub g: f32,
}

impl AlphaSynapse {
    /// Create a new alpha synapse.
    ///
    /// # Errors
    /// Returns an error if either time constant is ≤ 0.
    pub fn new(tau_rise: f32, tau_decay: f32, weight: f32, e_rev: f32) -> Result<Self> {
        if tau_rise <= 0.0 {
            return Err(NeuralError::InvalidArgument(format!(
                "tau_rise must be > 0, got {tau_rise}"
            )));
        }
        if tau_decay <= 0.0 {
            return Err(NeuralError::InvalidArgument(format!(
                "tau_decay must be > 0, got {tau_decay}"
            )));
        }
        Ok(Self {
            tau_rise,
            tau_decay,
            weight,
            e_rev,
            x: 0.0,
            g: 0.0,
        })
    }

    /// Advance by `dt` ms; optionally receive a pre-synaptic spike.
    ///
    /// # Returns
    /// Current conductance `g`.
    pub fn update(&mut self, spike: bool, dt: f32) -> f32 {
        // Euler step for coupled ODE
        let dx = -self.x / self.tau_rise;
        let dg = -self.g / self.tau_decay + self.x;
        self.x += dt * dx;
        self.g += dt * dg;

        if spike {
            self.x += self.weight;
        }
        self.g.max(0.0)
    }

    /// Compute instantaneous synaptic current given post-synaptic potential.
    pub fn current(&self, v_post: f32) -> f32 {
        self.g * (v_post - self.e_rev)
    }
}

// ---------------------------------------------------------------------------
// STDPSynapse
// ---------------------------------------------------------------------------

/// Synapse with built-in Spike-Timing-Dependent Plasticity (STDP) weight update.
///
/// Maintains traces for online weight modification:
/// - Pre-synaptic trace `x` (potentiation): dx/dt = -x/τ_+
/// - Post-synaptic trace `y` (depression): dy/dt = -y/τ_-
///
/// Pre-spike:  w ← clip(w + A_+ · y,  0, w_max), x += 1
/// Post-spike: w ← clip(w + A_- · x,  0, w_max), y += 1
#[derive(Debug, Clone)]
pub struct STDPSynapse {
    /// Current synaptic weight
    pub w: f32,
    /// Pre-synaptic trace time constant (ms)
    pub tau_plus: f32,
    /// Post-synaptic trace time constant (ms)
    pub tau_minus: f32,
    /// LTP amplitude (> 0)
    pub a_plus: f32,
    /// LTD amplitude (< 0)
    pub a_minus: f32,
    /// Maximum weight bound
    pub w_max: f32,
    /// Pre-synaptic trace
    pub x: f32,
    /// Post-synaptic trace
    pub y: f32,
    /// Reversal potential (mV)
    pub e_rev: f32,
    /// Conductance for current delivery
    pub g: f32,
    /// Conductance decay time constant (ms)
    pub tau_g: f32,
}

impl STDPSynapse {
    /// Create a new STDP synapse.
    ///
    /// # Errors
    /// Returns an error if time constants are ≤ 0 or w_max ≤ 0.
    pub fn new(
        w: f32,
        tau_plus: f32,
        tau_minus: f32,
        a_plus: f32,
        a_minus: f32,
        w_max: f32,
    ) -> Result<Self> {
        if tau_plus <= 0.0 || tau_minus <= 0.0 {
            return Err(NeuralError::InvalidArgument(
                "STDP time constants must be > 0".into(),
            ));
        }
        if w_max <= 0.0 {
            return Err(NeuralError::InvalidArgument("w_max must be > 0".into()));
        }
        Ok(Self {
            w: w.clamp(0.0, w_max),
            tau_plus,
            tau_minus,
            a_plus,
            a_minus,
            w_max,
            x: 0.0,
            y: 0.0,
            e_rev: 0.0,
            g: 0.0,
            tau_g: 5.0,
        })
    }

    /// Advance traces and conductance; apply pre-synaptic spike.
    ///
    /// # Arguments
    /// * `pre_spike`  — whether the pre-synaptic neuron fired
    /// * `post_spike` — whether the post-synaptic neuron fired
    /// * `dt`         — time step (ms)
    ///
    /// # Returns
    /// Current conductance.
    pub fn update(&mut self, pre_spike: bool, post_spike: bool, dt: f32) -> f32 {
        // Decay traces and conductance
        self.x *= (-dt / self.tau_plus).exp();
        self.y *= (-dt / self.tau_minus).exp();
        self.g *= (-dt / self.tau_g).exp();

        if pre_spike {
            // Potentiation: pre after post
            self.w = (self.w + self.a_plus * self.y).clamp(0.0, self.w_max);
            self.x += 1.0;
            self.g += self.w;
        }

        if post_spike {
            // Depression: post after pre
            self.w = (self.w + self.a_minus * self.x).clamp(0.0, self.w_max);
            self.y += 1.0;
        }

        self.g
    }

    /// Compute instantaneous synaptic current.
    pub fn current(&self, v_post: f32) -> f32 {
        self.g * (v_post - self.e_rev)
    }
}

// ---------------------------------------------------------------------------
// SynapticDelay
// ---------------------------------------------------------------------------

/// Fixed axonal conduction delay implemented as a ring buffer.
///
/// Signals inserted via `push` are available via `pop` after exactly `delay`
/// time steps.
#[derive(Debug, Clone)]
pub struct SynapticDelay {
    buffer: VecDeque<f32>,
    delay: usize,
}

impl SynapticDelay {
    /// Create a new delay line with the given delay (in time steps).
    ///
    /// # Errors
    /// Returns an error if `delay == 0`.
    pub fn new(delay: usize) -> Result<Self> {
        if delay == 0 {
            return Err(NeuralError::InvalidArgument(
                "Synaptic delay must be at least 1 time step".into(),
            ));
        }
        let mut buffer = VecDeque::with_capacity(delay);
        for _ in 0..delay {
            buffer.push_back(0.0);
        }
        Ok(Self { buffer, delay })
    }

    /// Push a new value into the delay buffer and return the delayed output.
    ///
    /// The returned value is the signal that was pushed exactly `delay` steps ago.
    pub fn push_pop(&mut self, value: f32) -> f32 {
        self.buffer.push_back(value);
        self.buffer.pop_front().unwrap_or(0.0)
    }

    /// Returns the configured delay in time steps.
    pub fn delay(&self) -> usize {
        self.delay
    }

    /// Reset all buffered values to zero.
    pub fn reset(&mut self) {
        for v in self.buffer.iter_mut() {
            *v = 0.0;
        }
    }
}

// ---------------------------------------------------------------------------
// SpikeBoolDelay
// ---------------------------------------------------------------------------

/// Boolean version of `SynapticDelay` for spike trains.
#[derive(Debug, Clone)]
pub struct SpikeBoolDelay {
    buffer: VecDeque<bool>,
    delay: usize,
}

impl SpikeBoolDelay {
    /// Create a boolean spike delay buffer.
    ///
    /// # Errors
    /// Returns an error if `delay == 0`.
    pub fn new(delay: usize) -> Result<Self> {
        if delay == 0 {
            return Err(NeuralError::InvalidArgument(
                "Spike delay must be ≥ 1".into(),
            ));
        }
        let mut buffer = VecDeque::with_capacity(delay);
        for _ in 0..delay {
            buffer.push_back(false);
        }
        Ok(Self { buffer, delay })
    }

    /// Push a spike and return the delayed spike.
    pub fn push_pop(&mut self, spike: bool) -> bool {
        self.buffer.push_back(spike);
        self.buffer.pop_front().unwrap_or(false)
    }

    /// Returns the configured delay in time steps.
    pub fn delay(&self) -> usize {
        self.delay
    }

    /// Reset to silence.
    pub fn reset(&mut self) {
        for v in self.buffer.iter_mut() {
            *v = false;
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exponential_synapse_zero_without_spikes() {
        let mut s = ExponentialSynapse::ampa(1.0);
        for _ in 0..100 {
            s.update(false, 0.1);
        }
        assert!(s.g.abs() < 1e-6);
    }

    #[test]
    fn exponential_synapse_increases_on_spike() {
        let mut s = ExponentialSynapse::ampa(1.0);
        s.update(true, 0.1);
        assert!(s.g > 0.9, "g should be ~1.0 right after spike");
    }

    #[test]
    fn exponential_synapse_decays() {
        let mut s = ExponentialSynapse::ampa(1.0);
        s.update(true, 0.1); // inject spike
        let g_after_spike = s.g;
        for _ in 0..1000 {
            s.update(false, 0.1);
        }
        assert!(s.g < g_after_spike * 0.01, "conductance should decay");
    }

    #[test]
    fn alpha_synapse_rises_then_decays() {
        let mut s = AlphaSynapse::new(1.0, 5.0, 1.0, 0.0).expect("operation should succeed");
        s.update(true, 0.1);
        let mut peak = 0.0_f32;
        for _ in 0..500 {
            let g = s.update(false, 0.1);
            peak = peak.max(g);
        }
        assert!(
            peak > 0.01,
            "alpha synapse should produce non-zero response"
        );
        assert!(s.g < 1e-4, "conductance should decay to near zero");
    }

    #[test]
    fn stdp_potentiation_when_post_before_pre() {
        let mut s =
            STDPSynapse::new(0.5, 20.0, 20.0, 0.01, -0.01, 1.0).expect("operation should succeed");
        // Post-synaptic fires first (y trace builds up), then pre fires → LTP
        s.update(false, true, 1.0);
        let w_before = s.w;
        s.update(true, false, 1.0);
        assert!(s.w >= w_before, "LTP expected when pre fires after post");
    }

    #[test]
    fn synaptic_delay_delays_by_correct_steps() {
        let mut d = SynapticDelay::new(3).expect("operation should succeed");
        let out0 = d.push_pop(1.0);
        let out1 = d.push_pop(0.0);
        let out2 = d.push_pop(0.0);
        let out3 = d.push_pop(0.0);
        assert_eq!(out0, 0.0);
        assert_eq!(out1, 0.0);
        assert_eq!(out2, 0.0);
        assert!((out3 - 1.0).abs() < 1e-6, "signal should arrive at step 3");
    }

    #[test]
    fn synaptic_delay_rejects_zero() {
        assert!(SynapticDelay::new(0).is_err());
    }

    #[test]
    fn spike_bool_delay_delays_spikes() {
        let mut d = SpikeBoolDelay::new(2).expect("operation should succeed");
        let o0 = d.push_pop(true);
        let o1 = d.push_pop(false);
        let o2 = d.push_pop(false);
        assert!(!o0);
        assert!(!o1);
        assert!(o2, "spike should emerge after 2 steps");
    }
}
