//! Spiking Neuron Models
//!
//! Implements biologically-inspired spiking neuron models:
//! - Leaky Integrate-and-Fire (LIF)
//! - Adaptive Exponential Integrate-and-Fire (AdEx)
//! - Izhikevich model
//! - Hodgkin-Huxley conductance-based model

use crate::error::{NeuralError, Result};

// ---------------------------------------------------------------------------
// LIF Neuron
// ---------------------------------------------------------------------------

/// Configuration for Leaky Integrate-and-Fire neurons.
#[derive(Debug, Clone)]
pub struct LIFConfig {
    /// Resting membrane potential (mV)
    pub v_rest: f32,
    /// Spike threshold (mV)
    pub v_thresh: f32,
    /// Reset potential after spike (mV)
    pub v_reset: f32,
    /// Membrane time constant (ms)
    pub tau_m: f32,
    /// Membrane resistance (MΩ)
    pub r_m: f32,
    /// Absolute refractory period (ms)
    pub t_ref: f32,
}

impl Default for LIFConfig {
    fn default() -> Self {
        Self {
            v_rest: -65.0,
            v_thresh: -50.0,
            v_reset: -65.0,
            tau_m: 20.0,
            r_m: 10.0,
            t_ref: 2.0,
        }
    }
}

/// Leaky Integrate-and-Fire (LIF) neuron model.
///
/// dv/dt = -(v - v_rest) / tau_m + r_m * I(t) / tau_m
///
/// Euler update: v += dt * (-(v - v_rest) / tau_m + r_m * i / tau_m)
/// Spike when v >= v_thresh → reset v = v_reset
#[derive(Debug, Clone)]
pub struct LIFNeuron {
    /// Resting membrane potential (mV)
    pub v_rest: f32,
    /// Spike threshold (mV)
    pub v_thresh: f32,
    /// Reset potential (mV)
    pub v_reset: f32,
    /// Membrane time constant (ms)
    pub tau_m: f32,
    /// Membrane resistance (MΩ)
    pub r_m: f32,
    /// Absolute refractory period (ms)
    pub t_ref: f32,
    /// Current membrane potential (mV)
    pub v: f32,
    /// Remaining refractory time (ms)
    refractory_remaining: f32,
}

impl LIFNeuron {
    /// Create a new LIF neuron from configuration.
    pub fn new(config: &LIFConfig) -> Self {
        Self {
            v_rest: config.v_rest,
            v_thresh: config.v_thresh,
            v_reset: config.v_reset,
            tau_m: config.tau_m,
            r_m: config.r_m,
            t_ref: config.t_ref,
            v: config.v_rest,
            refractory_remaining: 0.0,
        }
    }

    /// Advance the neuron by one time step.
    ///
    /// # Arguments
    /// * `current` — injected current I(t) in nA
    /// * `dt`      — time step in ms
    ///
    /// # Returns
    /// `true` if the neuron fires a spike this step.
    pub fn step(&mut self, current: f32, dt: f32) -> bool {
        if self.refractory_remaining > 0.0 {
            self.refractory_remaining -= dt;
            self.v = self.v_reset;
            return false;
        }

        let dv = dt * (-(self.v - self.v_rest) / self.tau_m + self.r_m * current / self.tau_m);
        self.v += dv;

        if self.v >= self.v_thresh {
            self.v = self.v_reset;
            self.refractory_remaining = self.t_ref;
            return true;
        }
        false
    }

    /// Reset the neuron to its resting state.
    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.refractory_remaining = 0.0;
    }
}

// ---------------------------------------------------------------------------
// AdEx Neuron
// ---------------------------------------------------------------------------

/// Adaptive Exponential Integrate-and-Fire (AdEx) neuron.
///
/// C_m * dv/dt = -g_l*(v-E_l) + g_l*δ_T*exp((v-v_T)/δ_T) - w + I
/// τ_w  * dw/dt = a*(v - E_l) - w
///
/// Spike: v ≥ 0 mV → v = v_r, w += b
#[derive(Debug, Clone)]
pub struct AdExNeuron {
    /// Membrane capacitance (pF)
    pub c_m: f32,
    /// Leak conductance (nS)
    pub g_l: f32,
    /// Leak reversal potential (mV)
    pub e_l: f32,
    /// Threshold potential (mV)
    pub v_t: f32,
    /// Slope factor (mV)
    pub delta_t: f32,
    /// Sub-threshold adaptation coupling (nS)
    pub a: f32,
    /// Spike-triggered adaptation increment (pA)
    pub b: f32,
    /// Adaptation time constant (ms)
    pub tau_w: f32,
    /// Reset potential (mV)
    pub v_r: f32,
    /// Spike detection threshold
    pub v_peak: f32,
    /// Current membrane potential (mV)
    pub v: f32,
    /// Adaptation current (pA)
    pub w: f32,
}

impl AdExNeuron {
    /// Create an AdEx neuron with typical parameters (regular spiking).
    pub fn new_regular_spiking() -> Self {
        Self {
            c_m: 281.0,
            g_l: 30.0,
            e_l: -70.6,
            v_t: -50.4,
            delta_t: 2.0,
            a: 4.0,
            b: 80.5,
            tau_w: 144.0,
            v_r: -70.6,
            v_peak: 20.0,
            v: -70.6,
            w: 0.0,
        }
    }

    /// Create an AdEx neuron from explicit parameters.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        c_m: f32,
        g_l: f32,
        e_l: f32,
        v_t: f32,
        delta_t: f32,
        a: f32,
        b: f32,
        tau_w: f32,
        v_r: f32,
    ) -> Self {
        Self {
            c_m,
            g_l,
            e_l,
            v_t,
            delta_t,
            a,
            b,
            tau_w,
            v_r,
            v_peak: 20.0,
            v: e_l,
            w: 0.0,
        }
    }

    /// Advance the neuron by one time step (Euler method).
    ///
    /// # Arguments
    /// * `current` — injected current in pA
    /// * `dt`      — time step in ms
    ///
    /// # Returns
    /// `true` if the neuron fires a spike.
    pub fn step(&mut self, current: f32, dt: f32) -> bool {
        let exp_term =
            self.g_l * self.delta_t * ((self.v - self.v_t) / self.delta_t).exp().min(1e6_f32);

        let dv = dt / self.c_m * (-self.g_l * (self.v - self.e_l) + exp_term - self.w + current);
        let dw = dt / self.tau_w * (self.a * (self.v - self.e_l) - self.w);

        self.v += dv;
        self.w += dw;

        if self.v >= self.v_peak {
            self.v = self.v_r;
            self.w += self.b;
            return true;
        }
        false
    }

    /// Reset to resting state.
    pub fn reset(&mut self) {
        self.v = self.e_l;
        self.w = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Izhikevich Neuron
// ---------------------------------------------------------------------------

/// Firing patterns for Izhikevich neurons.
#[derive(Debug, Clone, Copy)]
pub enum IzhikevichPattern {
    /// Regular spiking cortical neurons
    RegularSpiking,
    /// Intrinsically bursting neurons
    IntrinsicallyBursting,
    /// Chattering neurons
    Chattering,
    /// Fast spiking interneurons
    FastSpiking,
    /// Low-threshold spiking
    LowThresholdSpiking,
    /// Resonator neurons
    Resonator,
    /// Thalamo-cortical relay mode
    ThalamoCorticalRelay,
    /// Custom parameters
    Custom { a: f32, b: f32, c: f32, d: f32 },
}

/// Izhikevich spiking neuron model.
///
/// v' = 0.04v² + 5v + 140 - u + I
/// u' = a(bv - u)
///
/// Spike condition: v ≥ 30 → v = c, u += d
#[derive(Debug, Clone)]
pub struct IzhikevichNeuron {
    /// Recovery variable time scale
    pub a: f32,
    /// Recovery variable sensitivity to sub-threshold oscillations
    pub b: f32,
    /// After-spike reset value of v (mV)
    pub c: f32,
    /// After-spike reset increment of u
    pub d: f32,
    /// Membrane potential (mV)
    pub v: f32,
    /// Recovery variable
    pub u: f32,
    /// Spike threshold
    pub v_peak: f32,
}

impl IzhikevichNeuron {
    /// Create a neuron with the specified firing pattern.
    pub fn new(pattern: IzhikevichPattern) -> Self {
        let (a, b, c, d) = match pattern {
            IzhikevichPattern::RegularSpiking => (0.02, 0.2, -65.0, 8.0),
            IzhikevichPattern::IntrinsicallyBursting => (0.02, 0.2, -55.0, 4.0),
            IzhikevichPattern::Chattering => (0.02, 0.2, -50.0, 2.0),
            IzhikevichPattern::FastSpiking => (0.1, 0.2, -65.0, 2.0),
            IzhikevichPattern::LowThresholdSpiking => (0.02, 0.25, -65.0, 2.0),
            IzhikevichPattern::Resonator => (0.1, 0.26, -65.0, 2.0),
            IzhikevichPattern::ThalamoCorticalRelay => (0.02, 0.25, -65.0, 0.05),
            IzhikevichPattern::Custom { a, b, c, d } => (a, b, c, d),
        };
        Self {
            a,
            b,
            c,
            d,
            v: -65.0,
            u: b * (-65.0),
            v_peak: 30.0,
        }
    }

    /// Advance the neuron by one time step (0.5 ms sub-step Euler).
    ///
    /// # Arguments
    /// * `current` — injected current
    /// * `dt`      — time step (ms); uses 0.5 ms sub-steps for stability
    ///
    /// # Returns
    /// `true` if a spike occurred.
    pub fn step(&mut self, current: f32, dt: f32) -> bool {
        let steps = ((dt / 0.5).ceil() as usize).max(1);
        let sub_dt = dt / steps as f32;
        let mut spiked = false;

        for _ in 0..steps {
            if self.v >= self.v_peak {
                self.v = self.c;
                self.u += self.d;
                spiked = true;
            }
            let dv = sub_dt * (0.04 * self.v * self.v + 5.0 * self.v + 140.0 - self.u + current);
            let du = sub_dt * self.a * (self.b * self.v - self.u);
            self.v += dv;
            self.u += du;
        }
        spiked
    }

    /// Reset to resting state.
    pub fn reset(&mut self) {
        self.v = -65.0;
        self.u = self.b * self.v;
    }
}

// ---------------------------------------------------------------------------
// Hodgkin-Huxley Neuron
// ---------------------------------------------------------------------------

/// Hodgkin-Huxley conductance-based neuron model.
///
/// Full biophysical model with voltage-gated Na⁺ and K⁺ channels:
///
/// C_m dV/dt = I - g_Na·m³·h·(V-E_Na) - g_K·n⁴·(V-E_K) - g_L·(V-E_L)
///
/// Gating variables m, h, n follow first-order kinetics.
#[derive(Debug, Clone)]
pub struct HodgkinHuxleyNeuron {
    /// Membrane capacitance (μF/cm²)
    pub c_m: f32,
    /// Maximum Na⁺ conductance (mS/cm²)
    pub g_na: f32,
    /// Maximum K⁺ conductance (mS/cm²)
    pub g_k: f32,
    /// Leak conductance (mS/cm²)
    pub g_l: f32,
    /// Na⁺ reversal potential (mV)
    pub e_na: f32,
    /// K⁺ reversal potential (mV)
    pub e_k: f32,
    /// Leak reversal potential (mV)
    pub e_l: f32,
    /// Spike detection threshold (mV)
    pub v_thresh: f32,

    // State variables
    /// Membrane potential (mV)
    pub v: f32,
    /// Na activation gate
    pub m: f32,
    /// Na inactivation gate
    pub h: f32,
    /// K activation gate
    pub n: f32,

    // Internal spike detection
    v_prev: f32,
}

impl Default for HodgkinHuxleyNeuron {
    fn default() -> Self {
        // Standard Hodgkin-Huxley parameters (squid giant axon, 6.3°C)
        let v0 = -65.0_f32;
        let m0 = Self::alpha_m(v0) / (Self::alpha_m(v0) + Self::beta_m(v0));
        let h0 = Self::alpha_h(v0) / (Self::alpha_h(v0) + Self::beta_h(v0));
        let n0 = Self::alpha_n(v0) / (Self::alpha_n(v0) + Self::beta_n(v0));
        Self {
            c_m: 1.0,
            g_na: 120.0,
            g_k: 36.0,
            g_l: 0.3,
            e_na: 50.0,
            e_k: -77.0,
            e_l: -54.387,
            v_thresh: 0.0,
            v: v0,
            m: m0,
            h: h0,
            n: n0,
            v_prev: v0,
        }
    }
}

impl HodgkinHuxleyNeuron {
    /// Create a new HH neuron with standard parameters.
    pub fn new() -> Self {
        Self::default()
    }

    // --- Rate functions ---

    fn alpha_m(v: f32) -> f32 {
        let dv = v + 40.0;
        if dv.abs() < 1e-7 {
            1.0
        } else {
            0.1 * dv / (1.0 - (-dv / 10.0).exp())
        }
    }

    fn beta_m(v: f32) -> f32 {
        4.0 * (-(v + 65.0) / 18.0).exp()
    }

    fn alpha_h(v: f32) -> f32 {
        0.07 * (-(v + 65.0) / 20.0).exp()
    }

    fn beta_h(v: f32) -> f32 {
        1.0 / (1.0 + (-(v + 35.0) / 10.0).exp())
    }

    fn alpha_n(v: f32) -> f32 {
        let dv = v + 55.0;
        if dv.abs() < 1e-7 {
            0.1
        } else {
            0.01 * dv / (1.0 - (-dv / 10.0).exp())
        }
    }

    fn beta_n(v: f32) -> f32 {
        0.125 * (-(v + 65.0) / 80.0).exp()
    }

    /// Advance by one time step using 4th-order Runge-Kutta.
    ///
    /// # Arguments
    /// * `current` — injected current (μA/cm²)
    /// * `dt`      — time step (ms)
    ///
    /// # Returns
    /// `true` if a spike threshold crossing (upward) occurred.
    pub fn step(&mut self, current: f32, dt: f32) -> bool {
        self.v_prev = self.v;
        self.rk4_step(current, dt);
        // Detect upward threshold crossing
        self.v_prev < self.v_thresh && self.v >= self.v_thresh
    }

    fn derivatives(&self, v: f32, m: f32, h: f32, n: f32, i_ext: f32) -> (f32, f32, f32, f32) {
        let i_na = self.g_na * m * m * m * h * (v - self.e_na);
        let i_k = self.g_k * n * n * n * n * (v - self.e_k);
        let i_l = self.g_l * (v - self.e_l);

        let dv = (i_ext - i_na - i_k - i_l) / self.c_m;
        let dm = Self::alpha_m(v) * (1.0 - m) - Self::beta_m(v) * m;
        let dh = Self::alpha_h(v) * (1.0 - h) - Self::beta_h(v) * h;
        let dn = Self::alpha_n(v) * (1.0 - n) - Self::beta_n(v) * n;
        (dv, dm, dh, dn)
    }

    fn rk4_step(&mut self, i_ext: f32, dt: f32) {
        let (v, m, h, n) = (self.v, self.m, self.h, self.n);

        let (k1v, k1m, k1h, k1n) = self.derivatives(v, m, h, n, i_ext);
        let (k2v, k2m, k2h, k2n) = self.derivatives(
            v + 0.5 * dt * k1v,
            m + 0.5 * dt * k1m,
            h + 0.5 * dt * k1h,
            n + 0.5 * dt * k1n,
            i_ext,
        );
        let (k3v, k3m, k3h, k3n) = self.derivatives(
            v + 0.5 * dt * k2v,
            m + 0.5 * dt * k2m,
            h + 0.5 * dt * k2h,
            n + 0.5 * dt * k2n,
            i_ext,
        );
        let (k4v, k4m, k4h, k4n) = self.derivatives(
            v + dt * k3v,
            m + dt * k3m,
            h + dt * k3h,
            n + dt * k3n,
            i_ext,
        );

        self.v += dt / 6.0 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v);
        self.m += dt / 6.0 * (k1m + 2.0 * k2m + 2.0 * k3m + k4m);
        self.h += dt / 6.0 * (k1h + 2.0 * k2h + 2.0 * k3h + k4h);
        self.n += dt / 6.0 * (k1n + 2.0 * k2n + 2.0 * k3n + k4n);

        // Clamp gating variables to [0,1]
        self.m = self.m.clamp(0.0, 1.0);
        self.h = self.h.clamp(0.0, 1.0);
        self.n = self.n.clamp(0.0, 1.0);
    }

    /// Reset to resting state.
    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

// ---------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------

/// Validate that a time constant is strictly positive.
pub(crate) fn validate_tau(tau: f32, name: &str) -> Result<()> {
    if tau <= 0.0 {
        return Err(NeuralError::InvalidArgument(format!(
            "{name} must be > 0, got {tau}"
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lif_no_input_stays_at_rest() {
        let mut n = LIFNeuron::new(&LIFConfig::default());
        for _ in 0..100 {
            let spike = n.step(0.0, 0.1);
            assert!(!spike);
        }
        assert!((n.v - n.v_rest).abs() < 1.0);
    }

    #[test]
    fn lif_strong_input_fires() {
        let mut n = LIFNeuron::new(&LIFConfig::default());
        let mut fired = false;
        for _ in 0..1000 {
            if n.step(10.0, 0.1) {
                fired = true;
                break;
            }
        }
        assert!(fired, "LIF should fire with strong input");
    }

    #[test]
    fn adex_fires_under_large_current() {
        let mut n = AdExNeuron::new_regular_spiking();
        let mut fired = false;
        for _ in 0..5000 {
            if n.step(1000.0, 0.1) {
                fired = true;
                break;
            }
        }
        assert!(fired, "AdEx should fire under strong current");
    }

    #[test]
    fn izhikevich_regular_spiking_fires() {
        let mut n = IzhikevichNeuron::new(IzhikevichPattern::RegularSpiking);
        let mut fired = false;
        for _ in 0..2000 {
            if n.step(10.0, 0.5) {
                fired = true;
                break;
            }
        }
        assert!(fired, "Izhikevich RS should fire");
    }

    #[test]
    fn hh_fires_under_current() {
        let mut n = HodgkinHuxleyNeuron::new();
        let mut fired = false;
        for _ in 0..5000 {
            if n.step(10.0, 0.01) {
                fired = true;
                break;
            }
        }
        assert!(fired, "HH neuron should fire under 10 μA/cm²");
    }

    #[test]
    fn validate_tau_rejects_zero() {
        assert!(validate_tau(0.0, "tau").is_err());
        assert!(validate_tau(-1.0, "tau").is_err());
        assert!(validate_tau(10.0, "tau").is_ok());
    }
}
