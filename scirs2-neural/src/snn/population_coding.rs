//! Population Coding and Spike Encoding
//!
//! Converts continuous-valued signals into spike trains using biologically
//! motivated encoding schemes:
//! - Rate coding (Poisson process)
//! - Temporal coding (time-to-first-spike)
//! - Population coding (Gaussian tuning curves)

use crate::error::{NeuralError, Result};

// ---------------------------------------------------------------------------
// Encoding Schemes
// ---------------------------------------------------------------------------

/// Spike encoding strategy.
#[derive(Debug, Clone)]
pub enum SpikeEncoding {
    /// Poisson rate coding: fires at rate proportional to input value.
    /// `period` is the reference period (ms) corresponding to rate = 1.
    RateCoding { period: f32 },
    /// Time-to-first-spike temporal coding.
    /// Stronger inputs produce earlier spikes within a `window` (ms).
    TemporalCoding { window: f32 },
    /// Population coding with Gaussian tuning curves.
    PopulationCoding {
        /// Number of neurons in the population
        n_neurons: usize,
        /// Width (σ) of each Gaussian tuning curve
        tuning_width: f32,
        /// Minimum preferred stimulus value
        x_min: f32,
        /// Maximum preferred stimulus value
        x_max: f32,
    },
}

// ---------------------------------------------------------------------------
// SpikeEncoder
// ---------------------------------------------------------------------------

/// Encodes continuous-valued inputs as spike trains.
#[derive(Debug, Clone)]
pub struct SpikeEncoder {
    /// Encoding scheme in use
    pub encoding: SpikeEncoding,
    /// Internal phase accumulators for rate coding (one per input dimension)
    phase: Vec<f32>,
}

impl SpikeEncoder {
    /// Create a new encoder with the given encoding scheme.
    pub fn new(encoding: SpikeEncoding) -> Self {
        Self {
            encoding,
            phase: Vec::new(),
        }
    }

    /// Initialize phase accumulators for a given input dimensionality.
    pub fn init(&mut self, n_inputs: usize) {
        self.phase = vec![0.0; n_inputs];
    }

    /// Encode a scalar input value as a single spike (true/false) for one time step.
    ///
    /// For `RateCoding` the phase accumulates at each step; a spike is emitted
    /// when `phase >= 1.0`.  Uses a deterministic integrate-and-fire approach
    /// rather than random sampling (preserves reproducibility).
    ///
    /// # Arguments
    /// * `value` — normalised input in [0, 1]
    /// * `dt`    — time step (ms)
    /// * `idx`   — encoder channel index (for phase tracking)
    ///
    /// # Errors
    /// Returns an error if `idx` is out of bounds.
    pub fn encode_scalar(&mut self, value: f32, dt: f32, idx: usize) -> Result<bool> {
        match &self.encoding {
            SpikeEncoding::RateCoding { period } => {
                if idx >= self.phase.len() {
                    return Err(NeuralError::InvalidArgument(format!(
                        "encoder index {idx} out of bounds ({})",
                        self.phase.len()
                    )));
                }
                let rate = value.clamp(0.0, 1.0) / period;
                self.phase[idx] += rate * dt;
                if self.phase[idx] >= 1.0 {
                    self.phase[idx] -= 1.0;
                    return Ok(true);
                }
                Ok(false)
            }
            SpikeEncoding::TemporalCoding { window: _ } => {
                // Temporal coding: single spike at time t = window * (1 - value)
                // At step 0 we emit iff value is above threshold for this dt slice.
                // Simplified: treat as rate encoding with inverse rate.
                if idx >= self.phase.len() {
                    return Err(NeuralError::InvalidArgument(format!(
                        "encoder index {idx} out of bounds"
                    )));
                }
                // Emit at most once; track via phase saturation
                if self.phase[idx] < 0.0 {
                    return Ok(false); // already fired
                }
                let threshold = 1.0 - value.clamp(0.0, 1.0);
                self.phase[idx] += dt;
                if self.phase[idx] >= threshold * 100.0 {
                    self.phase[idx] = -1.0; // mark as fired
                    return Ok(true);
                }
                Ok(false)
            }
            SpikeEncoding::PopulationCoding { .. } => Err(NeuralError::InvalidArgument(
                "Use encode_population() for PopulationCoding".into(),
            )),
        }
    }

    /// Encode a scalar value using population (Gaussian tuning curve) coding.
    ///
    /// Returns a boolean spike vector of length `n_neurons`.
    ///
    /// Each neuron `i` has preferred stimulus `μ_i = x_min + i*(x_max - x_min)/(n-1)`.
    /// Firing rate ∝ exp(-(x - μ_i)² / (2 σ²)). A spike is emitted if the
    /// accumulated phase for that neuron crosses 1.
    ///
    /// # Arguments
    /// * `value` — stimulus value
    /// * `dt`    — time step (ms)
    ///
    /// # Errors
    /// Returns an error if encoding is not `PopulationCoding`.
    pub fn encode_population(&mut self, value: f32, dt: f32) -> Result<Vec<bool>> {
        let (n_neurons, tuning_width, x_min, x_max) = match &self.encoding {
            SpikeEncoding::PopulationCoding {
                n_neurons,
                tuning_width,
                x_min,
                x_max,
            } => (*n_neurons, *tuning_width, *x_min, *x_max),
            _ => {
                return Err(NeuralError::InvalidArgument(
                    "SpikeEncoder is not configured for PopulationCoding".into(),
                ))
            }
        };

        if self.phase.len() != n_neurons {
            self.phase = vec![0.0; n_neurons];
        }

        let range = (x_max - x_min).max(1e-6);
        let mut spikes = vec![false; n_neurons];

        for (i, spike) in spikes.iter_mut().enumerate().take(n_neurons) {
            let mu = x_min + i as f32 * range / (n_neurons.saturating_sub(1).max(1)) as f32;
            let diff = (value - mu) / tuning_width;
            let rate = (-0.5 * diff * diff).exp(); // in [0, 1]
            self.phase[i] += rate * dt;
            if self.phase[i] >= 1.0 {
                self.phase[i] -= 1.0;
                *spike = true;
            }
        }
        Ok(spikes)
    }

    /// Encode a full input vector for one time step.
    ///
    /// # Arguments
    /// * `inputs` — slice of normalised values in [0, 1]
    /// * `dt`     — time step (ms)
    ///
    /// # Returns
    /// Spike vector (same length as `inputs` for rate/temporal coding).
    ///
    /// # Errors
    /// Returns an error for `PopulationCoding` — use `encode_population` instead.
    pub fn encode_step(&mut self, inputs: &[f32], dt: f32) -> Result<Vec<bool>> {
        match &self.encoding {
            SpikeEncoding::PopulationCoding { .. } => Err(NeuralError::InvalidArgument(
                "Use encode_population() for PopulationCoding".into(),
            )),
            _ => {
                if self.phase.len() != inputs.len() {
                    self.phase = vec![0.0; inputs.len()];
                }
                let mut spikes = vec![false; inputs.len()];
                for (i, &v) in inputs.iter().enumerate() {
                    spikes[i] = self.encode_scalar(v, dt, i)?;
                }
                Ok(spikes)
            }
        }
    }

    /// Reset all internal phase accumulators.
    pub fn reset(&mut self) {
        for p in self.phase.iter_mut() {
            *p = 0.0;
        }
    }
}

// ---------------------------------------------------------------------------
// Spike Decoder
// ---------------------------------------------------------------------------

/// Decodes spike train statistics back to continuous values.
#[derive(Debug, Clone)]
pub struct SpikeDecoder {
    /// Accumulator for counting spikes
    spike_counts: Vec<u32>,
    /// Total number of time steps accumulated
    total_steps: u32,
}

impl SpikeDecoder {
    /// Create a new decoder for `n` neurons.
    pub fn new(n: usize) -> Self {
        Self {
            spike_counts: vec![0; n],
            total_steps: 0,
        }
    }

    /// Accumulate one time step of spike observations.
    ///
    /// # Errors
    /// Returns an error if `spikes.len() != n`.
    pub fn accumulate(&mut self, spikes: &[bool]) -> Result<()> {
        if spikes.len() != self.spike_counts.len() {
            return Err(NeuralError::DimensionMismatch(format!(
                "expected {} spike channels, got {}",
                self.spike_counts.len(),
                spikes.len()
            )));
        }
        for (cnt, &s) in self.spike_counts.iter_mut().zip(spikes.iter()) {
            if s {
                *cnt += 1;
            }
        }
        self.total_steps += 1;
        Ok(())
    }

    /// Compute the mean firing rate (spikes / step) for each neuron.
    pub fn firing_rates(&self) -> Vec<f32> {
        if self.total_steps == 0 {
            return vec![0.0; self.spike_counts.len()];
        }
        let n = self.total_steps as f32;
        self.spike_counts.iter().map(|&c| c as f32 / n).collect()
    }

    /// Decode population spike counts to a scalar estimate via centre-of-mass.
    ///
    /// Useful for population coding: returns the weighted centroid of preferred
    /// stimuli values across neurons.
    ///
    /// # Arguments
    /// * `x_min` / `x_max` — stimulus range
    ///
    /// # Errors
    /// Returns an error if the decoder has zero neurons.
    pub fn decode_population(&self, x_min: f32, x_max: f32) -> Result<f32> {
        let n = self.spike_counts.len();
        if n == 0 {
            return Err(NeuralError::InvalidArgument(
                "decoder has zero neurons".into(),
            ));
        }
        let range = x_max - x_min;
        let mut num = 0.0_f32;
        let mut denom = 0.0_f32;
        for (i, &c) in self.spike_counts.iter().enumerate() {
            let mu = x_min + i as f32 * range / (n.saturating_sub(1).max(1)) as f32;
            let rate = c as f32;
            num += rate * mu;
            denom += rate;
        }
        if denom < 1e-9 {
            return Ok((x_min + x_max) * 0.5);
        }
        Ok(num / denom)
    }

    /// Reset all counters.
    pub fn reset(&mut self) {
        for c in self.spike_counts.iter_mut() {
            *c = 0;
        }
        self.total_steps = 0;
    }
}

// ---------------------------------------------------------------------------
// Spike Statistics
// ---------------------------------------------------------------------------

/// Compute the inter-spike interval (ISI) sequence from a boolean spike train.
///
/// # Arguments
/// * `train` — boolean spike train
/// * `dt`    — time step (ms)
///
/// # Returns
/// Vector of ISI values (ms). Empty if fewer than 2 spikes.
pub fn inter_spike_intervals(train: &[bool], dt: f32) -> Vec<f32> {
    let spike_times: Vec<f32> = train
        .iter()
        .enumerate()
        .filter(|(_, &s)| s)
        .map(|(i, _)| i as f32 * dt)
        .collect();

    spike_times.windows(2).map(|w| w[1] - w[0]).collect()
}

/// Compute the coefficient of variation (CV) of the ISI distribution.
///
/// CV = σ_ISI / μ_ISI. CV ≈ 1 for Poisson, < 1 for regular, > 1 for bursting.
pub fn isi_cv(train: &[bool], dt: f32) -> Option<f32> {
    let isis = inter_spike_intervals(train, dt);
    if isis.len() < 2 {
        return None;
    }
    let n = isis.len() as f32;
    let mean = isis.iter().sum::<f32>() / n;
    let var = isis.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n;
    let std = var.sqrt();
    if mean.abs() < 1e-9 {
        return None;
    }
    Some(std / mean)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rate_encoder_fires_at_correct_rate() {
        let mut enc = SpikeEncoder::new(SpikeEncoding::RateCoding { period: 1.0 });
        enc.init(1);
        // With value=0.5 and dt=1.0, rate = 0.5 spikes/ms
        let mut count = 0;
        for _ in 0..1000 {
            if enc
                .encode_scalar(0.5, 1.0, 0)
                .expect("operation should succeed")
            {
                count += 1;
            }
        }
        // Expect roughly 500 spikes (±50 tolerance)
        assert!(
            (count - 500_i32).abs() <= 50,
            "Expected ~500 spikes, got {count}"
        );
    }

    #[test]
    fn rate_encoder_zero_input_no_spike() {
        let mut enc = SpikeEncoder::new(SpikeEncoding::RateCoding { period: 10.0 });
        enc.init(1);
        for _ in 0..1000 {
            assert!(!enc
                .encode_scalar(0.0, 1.0, 0)
                .expect("operation should succeed"));
        }
    }

    #[test]
    fn population_encoder_fires_near_preferred() {
        let mut enc = SpikeEncoder::new(SpikeEncoding::PopulationCoding {
            n_neurons: 10,
            tuning_width: 0.2,
            x_min: 0.0,
            x_max: 1.0,
        });
        // Run for 1000 steps, stimulus near neuron 5's preferred value (0.5)
        let mut counts = [0usize; 10];
        for _ in 0..1000 {
            let spikes = enc
                .encode_population(0.5, 1.0)
                .expect("operation should succeed");
            for (i, &s) in spikes.iter().enumerate() {
                if s {
                    counts[i] += 1;
                }
            }
        }
        // Neuron index 4 or 5 (preferred ~0.44 or ~0.56) should fire most
        let max_idx = counts
            .iter()
            .enumerate()
            .max_by_key(|&(_, c)| c)
            .map(|(i, _)| i)
            .expect("operation should succeed");
        assert!(
            (max_idx as i32 - 4).abs() <= 2,
            "Max firing neuron {max_idx} should be near centre"
        );
    }

    #[test]
    fn spike_decoder_firing_rates() {
        let mut dec = SpikeDecoder::new(3);
        // 10 steps: neuron 0 fires all, neuron 1 fires half, neuron 2 never
        for i in 0..10 {
            dec.accumulate(&[true, i % 2 == 0, false])
                .expect("operation should succeed");
        }
        let rates = dec.firing_rates();
        assert!((rates[0] - 1.0).abs() < 1e-5);
        assert!((rates[1] - 0.5).abs() < 1e-5);
        assert_eq!(rates[2], 0.0);
    }

    #[test]
    fn isi_computation() {
        // Spikes at steps 0, 10, 20 with dt=1.0 → ISI = [10.0, 10.0]
        let mut train = vec![false; 25];
        train[0] = true;
        train[10] = true;
        train[20] = true;
        let isis = inter_spike_intervals(&train, 1.0);
        assert_eq!(isis.len(), 2);
        assert!((isis[0] - 10.0).abs() < 1e-5);
        assert!((isis[1] - 10.0).abs() < 1e-5);
        let cv = isi_cv(&train, 1.0).expect("operation should succeed");
        assert!(cv.abs() < 1e-5, "Regular train should have CV ≈ 0");
    }
}
