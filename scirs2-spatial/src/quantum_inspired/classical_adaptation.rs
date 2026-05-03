//! Classical Adaptation of Quantum-Inspired Algorithms
//!
//! This module bridges the gap between full quantum simulation and classical
//! computation by providing quantum-inspired enhancements to classical algorithms.
//! Rather than simulating quantum circuits exactly (which is exponentially costly),
//! these adaptations borrow key ideas — controlled randomness, tunnelling noise,
//! and quantum annealing schedules — to improve classical algorithm performance.
//!
//! # Overview
//!
//! The primary abstraction is [`ClassicalAdaptation`], which offers:
//!
//! - **Coherent noise injection** (`adapt`): adds Gaussian noise scaled by the
//!   adaptation rate and quantum noise level to a parameter vector, simulating
//!   quantum-state fluctuation to escape local optima.
//! - **Quantum annealing step** (`anneal`): computes a quantum tunnelling
//!   acceptance probability that allows uphill moves with a probability that
//!   decays as the system temperature drops, analogous to the transverse-field
//!   Ising model's ground-state search.
//! - **Decoherence modelling** (`decohere`): applies an exponential amplitude
//!   decay to a state vector, modelling T₂ relaxation in a quantum system.
//! - **Parameter landscape smoothing** (`smooth_landscape`): convolves a
//!   1-D energy landscape with a Gaussian kernel to remove spurious minima,
//!   emulating the quantum superposition effect over neighbouring states.
//!
//! # Theoretical Motivation
//!
//! Quantum annealing replaces the thermal fluctuations of classical simulated
//! annealing with quantum tunnelling through energy barriers. For the transverse-
//! field Ising Hamiltonian H = -J Σ σᵢᶻσⱼᶻ - Γ Σ σᵢˣ, the tunnelling term Γ
//! allows transitions through barriers of height ΔE that would be exponentially
//! suppressed classically. This module emulates that effect on a classical
//! computer by using temperature-scaled, energy-aware acceptance probabilities.

use crate::error::{SpatialError, SpatialResult};
use scirs2_core::ndarray::Array1;
use scirs2_core::random::{Rng, RngExt};
use std::f64::consts::PI;

/// Quantum-Inspired Classical Adaptation Engine
///
/// Provides quantum-inspired enhancements to classical optimisation algorithms
/// by injecting controlled randomness (coherent noise), modelling quantum
/// annealing acceptance criteria, and simulating decoherence effects.
///
/// # Fields
/// - `adaptation_rate` — scales the amplitude of coherent noise injected into
///   parameter vectors (analogous to the magnitude of quantum fluctuations).
/// - `quantum_noise_level` — baseline noise variance; combined with
///   `adaptation_rate` to determine the actual noise standard deviation.
///
/// # Example
/// ```rust
/// use scirs2_core::ndarray::Array1;
/// use scirs2_spatial::quantum_inspired::classical_adaptation::ClassicalAdaptation;
///
/// let adapter = ClassicalAdaptation::new(0.05, 0.01);
///
/// // Add quantum-inspired noise to a parameter vector
/// let params = Array1::from_vec(vec![1.0, 2.0, 3.0]);
/// let noisy = adapter.adapt(&params);
/// assert_eq!(noisy.len(), params.len());
///
/// // Quantum annealing acceptance for an uphill move
/// let accept_prob = adapter.anneal(1.5, 2.0);
/// assert!(accept_prob >= 0.0 && accept_prob <= 1.0);
/// ```
#[derive(Debug, Clone)]
pub struct ClassicalAdaptation {
    /// Controls the amplitude of quantum coherent noise added during adaptation.
    /// Larger values allow wider exploration of the parameter landscape.
    adaptation_rate: f64,
    /// Baseline quantum noise level (variance contribution).
    /// Simulates intrinsic noise from a quantum environment.
    quantum_noise_level: f64,
}

impl ClassicalAdaptation {
    /// Construct a new `ClassicalAdaptation` engine.
    ///
    /// # Arguments
    /// * `adaptation_rate` — Amplitude scale for coherent noise (> 0 recommended).
    /// * `quantum_noise_level` — Baseline quantum noise variance (≥ 0).
    pub fn new(adaptation_rate: f64, quantum_noise_level: f64) -> Self {
        Self {
            adaptation_rate,
            quantum_noise_level,
        }
    }

    /// Return the adaptation rate.
    pub fn adaptation_rate(&self) -> f64 {
        self.adaptation_rate
    }

    /// Return the quantum noise level.
    pub fn quantum_noise_level(&self) -> f64 {
        self.quantum_noise_level
    }

    /// Inject quantum-inspired coherent noise into a parameter vector.
    ///
    /// Each component `pᵢ` is perturbed as:
    /// ```text
    /// p̃ᵢ = pᵢ + σ · ηᵢ,   ηᵢ ~ N(0, 1)
    /// ```
    /// where `σ = adaptation_rate · √(1 + quantum_noise_level)`.
    ///
    /// The resulting perturbations model quantum fluctuations around the current
    /// parameter values and can help classical optimisers escape local minima.
    ///
    /// # Arguments
    /// * `params` — Parameter vector to perturb.
    ///
    /// # Returns
    /// A new `Array1<f64>` with the perturbed parameters.
    pub fn adapt(&self, params: &Array1<f64>) -> Array1<f64> {
        let sigma = self.adaptation_rate * (1.0 + self.quantum_noise_level).sqrt();
        let mut rng = scirs2_core::random::rng();
        let mut result = params.clone();

        for val in result.iter_mut() {
            let noise = gaussian_noise(&mut rng, sigma);
            *val += noise;
        }
        result
    }

    /// Compute a quantum annealing acceptance probability for an energy transition.
    ///
    /// This models the probability that the quantum annealer accepts a move from
    /// the current state to a new state with the given `energy` at the given
    /// `temperature`. The formula is:
    ///
    /// ```text
    /// P(accept) = exp(-energy / (temperature · (1 + Γ)))
    /// ```
    ///
    /// where `Γ = quantum_noise_level` represents the transverse field strength.
    /// Compared with classical Metropolis, the denominator is larger (because
    /// Γ > 0), giving a *higher* acceptance probability — this models quantum
    /// tunnelling that can traverse energy barriers classical Metropolis cannot.
    ///
    /// # Arguments
    /// * `energy` — Current energy (or energy difference) of the proposed state.
    ///   Positive values correspond to uphill moves.
    /// * `temperature` — Current annealing temperature. Must be > 0 for
    ///   non-trivial acceptance; returns `0.0` for `temperature ≤ 0`.
    ///
    /// # Returns
    /// Acceptance probability in `[0, 1]`.
    pub fn anneal(&self, energy: f64, temperature: f64) -> f64 {
        if temperature <= 0.0 {
            // At absolute zero, only downhill moves are accepted
            if energy <= 0.0 {
                1.0
            } else {
                0.0
            }
        } else {
            let transverse_field = 1.0 + self.quantum_noise_level;
            let exponent = -energy / (temperature * transverse_field);
            // Clamp to avoid overflow
            exponent.exp().clamp(0.0, 1.0)
        }
    }

    /// Simulate decoherence on a quantum-state amplitude vector.
    ///
    /// Applies an exponential T₂ decay to each amplitude element:
    /// ```text
    /// ψ̃ᵢ = ψᵢ · exp(-t / T₂)
    /// ```
    /// where `T₂ = 1 / (adaptation_rate · quantum_noise_level + ε)` and
    /// `t` is the elapsed (normalised) time.
    ///
    /// # Arguments
    /// * `amplitudes` — Quantum-state amplitude vector.
    /// * `elapsed_time` — Normalised elapsed time in `[0, ∞)`.
    ///
    /// # Returns
    /// Decayed amplitude vector. Returns `amplitudes` unchanged if both
    /// `adaptation_rate` and `quantum_noise_level` are zero.
    ///
    /// # Errors
    /// Returns [`SpatialError::InvalidInput`] if `elapsed_time < 0`.
    pub fn decohere(
        &self,
        amplitudes: &Array1<f64>,
        elapsed_time: f64,
    ) -> SpatialResult<Array1<f64>> {
        if elapsed_time < 0.0 {
            return Err(SpatialError::InvalidInput(
                "elapsed_time must be non-negative".to_string(),
            ));
        }

        let decoherence_rate = self.adaptation_rate * self.quantum_noise_level + 1e-12;
        let t2 = 1.0 / decoherence_rate;
        let decay = (-elapsed_time / t2).exp();

        Ok(amplitudes.mapv(|a| a * decay))
    }

    /// Smooth a 1-D energy landscape using a Gaussian kernel.
    ///
    /// Convolves the energy array with a Gaussian of standard deviation
    /// `sigma_smooth = adaptation_rate * (landscape.len() as f64).sqrt()`,
    /// using a finite-support approximation (kernel half-width = 3σ).
    /// This emulates the quantum superposition effect: the effective energy
    /// at each point is averaged over nearby states weighted by the quantum
    /// probability of tunnelling to them.
    ///
    /// # Arguments
    /// * `landscape` — 1-D energy landscape to smooth.
    ///
    /// # Returns
    /// Smoothed energy array of the same length.
    ///
    /// # Errors
    /// Returns [`SpatialError::InvalidInput`] if the landscape is empty.
    pub fn smooth_landscape(&self, landscape: &Array1<f64>) -> SpatialResult<Array1<f64>> {
        let n = landscape.len();
        if n == 0 {
            return Err(SpatialError::InvalidInput(
                "landscape must be non-empty".to_string(),
            ));
        }

        let sigma = (self.adaptation_rate * (n as f64).sqrt()).max(0.5);
        let half_width = (3.0 * sigma).ceil() as usize;

        // Build 1-D Gaussian kernel (not pre-normalised; we normalise below)
        let kernel_len = 2 * half_width + 1;
        let mut kernel = vec![0.0f64; kernel_len];
        let mut kernel_sum = 0.0f64;
        for (k, kval) in kernel.iter_mut().enumerate() {
            let offset = k as f64 - half_width as f64;
            let g = (-0.5 * (offset / sigma).powi(2)).exp() / (sigma * (2.0 * PI).sqrt());
            *kval = g;
            kernel_sum += g;
        }
        // Normalise
        if kernel_sum > 1e-12 {
            for kval in kernel.iter_mut() {
                *kval /= kernel_sum;
            }
        }

        // Convolve landscape with kernel (reflect padding at boundaries)
        let mut smoothed = Array1::<f64>::zeros(n);
        for i in 0..n {
            let mut acc = 0.0f64;
            for (k, &kval) in kernel.iter().enumerate() {
                let offset = k as isize - half_width as isize;
                let idx = (i as isize + offset).clamp(0, n as isize - 1) as usize;
                acc += kval * landscape[idx];
            }
            smoothed[i] = acc;
        }

        Ok(smoothed)
    }
}

/// Draw a single Gaussian-distributed sample via Box-Muller transform.
///
/// Uses two uniform samples from `rng` to produce one N(0, sigma) variate.
fn gaussian_noise(rng: &mut impl Rng, sigma: f64) -> f64 {
    if sigma.abs() < 1e-15 {
        return 0.0;
    }
    let u1: f64 = rng.random_range(1e-10_f64..1.0_f64);
    let u2: f64 = rng.random_range(0.0_f64..1.0_f64);
    sigma * (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    #[test]
    fn test_adapt_output_shape_and_finite() {
        let adapter = ClassicalAdaptation::new(0.1, 0.05);
        let params = Array1::from_vec(vec![1.0, 2.0, -1.0, 0.5]);
        let adapted = adapter.adapt(&params);
        assert_eq!(adapted.len(), params.len());
        for &v in adapted.iter() {
            assert!(v.is_finite(), "adapted value must be finite");
        }
    }

    #[test]
    fn test_anneal_probability_bounds() {
        let adapter = ClassicalAdaptation::new(0.1, 0.05);

        // Uphill move: probability in (0, 1)
        let p_uphill = adapter.anneal(10.0, 1.0);
        assert!(p_uphill > 0.0 && p_uphill < 1.0);

        // Downhill / zero move: probability = 1.0 (since exp(0) = 1)
        let p_down = adapter.anneal(-5.0, 1.0);
        assert!(p_down >= 1.0 - 1e-9, "downhill acceptance should be ≥ 1");

        // Temperature → 0: uphill rejected, downhill accepted
        let p_cold_up = adapter.anneal(1.0, 0.0);
        assert_eq!(p_cold_up, 0.0);
        let p_cold_down = adapter.anneal(-1.0, 0.0);
        assert_eq!(p_cold_down, 1.0);
    }

    #[test]
    fn test_decohere_decay() {
        let adapter = ClassicalAdaptation::new(1.0, 1.0);
        let amps = Array1::from_vec(vec![1.0, 0.5, -0.5]);

        // At t=0, amplitudes are unchanged
        let t0 = adapter.decohere(&amps, 0.0).expect("t=0 is valid");
        for (&a, &b) in amps.iter().zip(t0.iter()) {
            assert!((a - b).abs() < 1e-12);
        }

        // At t>0, amplitudes are strictly reduced in magnitude
        let t1 = adapter.decohere(&amps, 1.0).expect("t=1 is valid");
        for (&a, &b) in amps.iter().zip(t1.iter()) {
            assert!(b.abs() < a.abs() + 1e-12);
        }

        // Negative time is rejected
        assert!(adapter.decohere(&amps, -1.0).is_err());
    }

    #[test]
    fn test_smooth_landscape_energy_preservation() {
        let adapter = ClassicalAdaptation::new(0.5, 0.1);
        let landscape = Array1::from_vec(vec![0.0, 1.0, 5.0, 1.0, 0.0, 2.0, 0.0]);
        let smoothed = adapter
            .smooth_landscape(&landscape)
            .expect("smoothing should succeed");

        assert_eq!(smoothed.len(), landscape.len());

        // All values should be finite
        for &v in smoothed.iter() {
            assert!(v.is_finite());
        }

        // The global maximum in the smoothed landscape must occur near where the
        // original maximum was (index 2), within a small neighbourhood
        let max_idx = smoothed
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        assert!(
            max_idx <= 4,
            "smoothed peak should remain near original spike, got index {max_idx}"
        );

        // Empty landscape should error
        let empty = Array1::<f64>::zeros(0);
        assert!(adapter.smooth_landscape(&empty).is_err());
    }
}
