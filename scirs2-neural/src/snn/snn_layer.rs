//! Spiking Neural Network Layers and Networks
//!
//! Provides:
//! - `SpikingLayer` — a layer of LIF neurons with exponential synapses
//! - `SpikingNetwork` — multi-layer SNN with full simulation loop
//! - `SpikeEncoder` re-exported for convenience
//! - `SpikeTrain` statistics utilities

use crate::error::{NeuralError, Result};
use crate::snn::neuron_models::{LIFConfig, LIFNeuron};
use crate::snn::synapse::ExponentialSynapse;

// ---------------------------------------------------------------------------
// SpikingLayer
// ---------------------------------------------------------------------------

/// A single layer of LIF neurons, each receiving inputs through exponential synapses.
///
/// Connectivity: dense (n_in × n_out) with independent weights per synapse.
#[derive(Debug)]
pub struct SpikingLayer {
    /// Output neurons
    pub neurons: Vec<LIFNeuron>,
    /// Synapses: `synapses[j][i]` is the synapse from input `i` to output `j`
    pub synapses: Vec<Vec<ExponentialSynapse>>,
    /// Number of input channels
    pub n_in: usize,
    /// Number of output neurons
    pub n_out: usize,
}

impl SpikingLayer {
    /// Create a new spiking layer with initialised LIF neurons and AMPA synapses.
    ///
    /// All synaptic weights are initialised to `init_weight / n_in` to keep
    /// the total input approximately constant regardless of fan-in.
    ///
    /// # Arguments
    /// * `n_in`        — number of input spike channels
    /// * `n_out`       — number of output neurons
    /// * `config`      — LIF configuration
    /// * `init_weight` — baseline total excitatory drive (before fan-in scaling)
    ///
    /// # Errors
    /// Returns an error if `n_in == 0` or `n_out == 0`.
    pub fn new(n_in: usize, n_out: usize, config: &LIFConfig, init_weight: f32) -> Result<Self> {
        if n_in == 0 {
            return Err(NeuralError::InvalidArgument("n_in must be > 0".into()));
        }
        if n_out == 0 {
            return Err(NeuralError::InvalidArgument("n_out must be > 0".into()));
        }

        let w = init_weight / n_in as f32;
        let neurons: Vec<LIFNeuron> = (0..n_out).map(|_| LIFNeuron::new(config)).collect();
        let synapses: Vec<Vec<ExponentialSynapse>> = (0..n_out)
            .map(|_| (0..n_in).map(|_| ExponentialSynapse::ampa(w)).collect())
            .collect();

        Ok(Self {
            neurons,
            synapses,
            n_in,
            n_out,
        })
    }

    /// Create a layer with explicit synaptic weights.
    ///
    /// # Arguments
    /// * `weights` — 2-D weight matrix, shape [n_out][n_in]
    /// * `config`  — LIF neuron configuration
    ///
    /// # Errors
    /// Returns an error if the weight matrix is jagged.
    pub fn from_weights(weights: &[Vec<f32>], config: &LIFConfig) -> Result<Self> {
        let n_out = weights.len();
        if n_out == 0 {
            return Err(NeuralError::InvalidArgument(
                "weights must be non-empty".into(),
            ));
        }
        let n_in = weights[0].len();
        if n_in == 0 {
            return Err(NeuralError::InvalidArgument(
                "inner weight dimension must be > 0".into(),
            ));
        }
        for (j, row) in weights.iter().enumerate() {
            if row.len() != n_in {
                return Err(NeuralError::DimensionMismatch(format!(
                    "row {j} has {} weights, expected {n_in}",
                    row.len()
                )));
            }
        }

        let neurons: Vec<LIFNeuron> = (0..n_out).map(|_| LIFNeuron::new(config)).collect();
        let synapses: Vec<Vec<ExponentialSynapse>> = weights
            .iter()
            .map(|row| row.iter().map(|&w| ExponentialSynapse::ampa(w)).collect())
            .collect();

        Ok(Self {
            neurons,
            synapses,
            n_in,
            n_out,
        })
    }

    /// Run the layer forward for one time step.
    ///
    /// Each output neuron accumulates current from all active input synapses
    /// and integrates it via its LIF dynamics.
    ///
    /// # Arguments
    /// * `input_spikes` — boolean spike vector of length `n_in`
    /// * `dt`           — time step (ms)
    ///
    /// # Returns
    /// Boolean spike vector of length `n_out`.
    ///
    /// # Errors
    /// Returns an error if `input_spikes.len() != n_in`.
    pub fn forward(&mut self, input_spikes: &[bool], dt: f32) -> Result<Vec<bool>> {
        if input_spikes.len() != self.n_in {
            return Err(NeuralError::DimensionMismatch(format!(
                "input spike length {} != n_in {}",
                input_spikes.len(),
                self.n_in
            )));
        }

        let mut output_spikes = vec![false; self.n_out];

        for (j, (neuron, syn_row)) in self
            .neurons
            .iter_mut()
            .zip(self.synapses.iter_mut())
            .enumerate()
        {
            let mut total_current = 0.0_f32;
            for (syn, &spike) in syn_row.iter_mut().zip(input_spikes.iter()) {
                let g = syn.update(spike, dt);
                // Use fixed post-synaptic potential at rest for current calculation
                total_current += g * neuron.r_m;
            }
            output_spikes[j] = neuron.step(total_current, dt);
        }

        Ok(output_spikes)
    }

    /// Reset all neurons and synapses to their initial state.
    pub fn reset(&mut self) {
        for neuron in self.neurons.iter_mut() {
            neuron.reset();
        }
        for syn_row in self.synapses.iter_mut() {
            for syn in syn_row.iter_mut() {
                syn.g = 0.0;
            }
        }
    }

    /// Get the weight matrix as a 2D vector [n_out][n_in].
    pub fn weights(&self) -> Vec<Vec<f32>> {
        self.synapses
            .iter()
            .map(|row| row.iter().map(|s| s.weight).collect())
            .collect()
    }

    /// Set the weight of a specific synapse.
    ///
    /// # Errors
    /// Returns an error if indices are out of bounds.
    pub fn set_weight(&mut self, out_idx: usize, in_idx: usize, weight: f32) -> Result<()> {
        if out_idx >= self.n_out {
            return Err(NeuralError::InvalidArgument(format!(
                "out_idx {out_idx} >= n_out {}",
                self.n_out
            )));
        }
        if in_idx >= self.n_in {
            return Err(NeuralError::InvalidArgument(format!(
                "in_idx {in_idx} >= n_in {}",
                self.n_in
            )));
        }
        self.synapses[out_idx][in_idx].weight = weight;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// SpikingNetwork
// ---------------------------------------------------------------------------

/// Multi-layer spiking neural network.
///
/// Layers are stacked sequentially; the output spike train of layer k is
/// the input to layer k+1.
#[derive(Debug)]
pub struct SpikingNetwork {
    /// Ordered list of spiking layers
    pub layers: Vec<SpikingLayer>,
    /// Simulation time step (ms)
    pub dt: f32,
}

impl SpikingNetwork {
    /// Create a spiking network by stacking layers.
    ///
    /// # Arguments
    /// * `layer_sizes` — `[n0, n1, …, nL]` where n0 is input size and nL is output size
    /// * `config`      — LIF configuration applied to all layers
    /// * `init_weight` — initial total weight per output neuron
    /// * `dt`          — simulation time step (ms)
    ///
    /// # Errors
    /// Returns an error if fewer than 2 sizes are given.
    pub fn new(
        layer_sizes: &[usize],
        config: &LIFConfig,
        init_weight: f32,
        dt: f32,
    ) -> Result<Self> {
        if layer_sizes.len() < 2 {
            return Err(NeuralError::InvalidArchitecture(
                "At least 2 layer sizes required".into(),
            ));
        }
        let mut layers = Vec::with_capacity(layer_sizes.len() - 1);
        for window in layer_sizes.windows(2) {
            let n_in = window[0];
            let n_out = window[1];
            layers.push(SpikingLayer::new(n_in, n_out, config, init_weight)?);
        }
        Ok(Self { layers, dt })
    }

    /// Simulate the network for `T` time steps given input spike trains.
    ///
    /// # Arguments
    /// * `input_spikes` — spike trains for the input layer: `input_spikes[t]` is
    ///   the input spike vector at time step `t`
    /// * `t_steps`      — number of simulation steps (must equal `input_spikes.len()`)
    ///
    /// # Returns
    /// Spike trains for every layer at every time step:
    /// `result[t][layer][neuron]`
    ///
    /// # Errors
    /// Returns an error on dimension mismatches.
    pub fn simulate(
        &mut self,
        input_spikes: &[Vec<bool>],
        t_steps: usize,
    ) -> Result<Vec<Vec<Vec<bool>>>> {
        if input_spikes.len() != t_steps {
            return Err(NeuralError::DimensionMismatch(format!(
                "input_spikes has {} time steps, expected {t_steps}",
                input_spikes.len()
            )));
        }

        let n_layers = self.layers.len();
        // result[t][layer] = spike vector
        let mut result: Vec<Vec<Vec<bool>>> = Vec::with_capacity(t_steps);

        for input_t in input_spikes.iter().take(t_steps) {
            let mut layer_spikes: Vec<Vec<bool>> = Vec::with_capacity(n_layers);
            let mut current_input = input_t.clone();

            for layer in self.layers.iter_mut() {
                let out = layer.forward(&current_input, self.dt)?;
                layer_spikes.push(out.clone());
                current_input = out;
            }

            result.push(layer_spikes);
        }

        Ok(result)
    }

    /// Reset all layers to their initial state.
    pub fn reset(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.reset();
        }
    }

    /// Count the total number of spikes across all layers and time steps.
    pub fn count_spikes(spike_record: &[Vec<Vec<bool>>]) -> usize {
        spike_record
            .iter()
            .flat_map(|t| t.iter())
            .flat_map(|l| l.iter())
            .filter(|&&s| s)
            .count()
    }

    /// Compute mean firing rate (spikes / neuron / time step) for each layer.
    pub fn mean_firing_rates(spike_record: &[Vec<Vec<bool>>]) -> Vec<f32> {
        let t_steps = spike_record.len();
        if t_steps == 0 {
            return Vec::new();
        }
        let n_layers = spike_record[0].len();
        let mut rates = vec![0.0_f32; n_layers];

        for t in spike_record.iter() {
            for (l, layer_spikes) in t.iter().enumerate() {
                let n = layer_spikes.len() as f32;
                if n > 0.0 {
                    let fired: f32 = layer_spikes.iter().filter(|&&s| s).count() as f32;
                    rates[l] += fired / n;
                }
            }
        }
        for r in rates.iter_mut() {
            *r /= t_steps as f32;
        }
        rates
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> LIFConfig {
        LIFConfig {
            v_rest: -65.0,
            v_thresh: -50.0,
            v_reset: -65.0,
            tau_m: 20.0,
            r_m: 10.0,
            t_ref: 2.0,
        }
    }

    #[test]
    fn spiking_layer_silent_input_silent_output() {
        let mut layer =
            SpikingLayer::new(5, 3, &default_config(), 1.0).expect("operation should succeed");
        for _ in 0..100 {
            let out = layer
                .forward(&[false; 5], 0.1)
                .expect("operation should succeed");
            assert!(out.iter().all(|&s| !s), "no input → no output");
        }
    }

    #[test]
    fn spiking_layer_strong_input_fires() {
        let mut layer =
            SpikingLayer::new(4, 2, &default_config(), 100.0).expect("operation should succeed");
        let mut any_fired = false;
        for _ in 0..500 {
            let out = layer
                .forward(&[true; 4], 0.5)
                .expect("operation should succeed");
            if out.iter().any(|&s| s) {
                any_fired = true;
                break;
            }
        }
        assert!(
            any_fired,
            "Strong input should cause at least one output spike"
        );
    }

    #[test]
    fn spiking_layer_dimension_mismatch() {
        let mut layer =
            SpikingLayer::new(4, 2, &default_config(), 1.0).expect("operation should succeed");
        let result = layer.forward(&[false; 3], 0.1);
        assert!(result.is_err());
    }

    #[test]
    fn spiking_layer_set_weight() {
        let mut layer =
            SpikingLayer::new(3, 2, &default_config(), 1.0).expect("operation should succeed");
        layer
            .set_weight(1, 2, 5.0)
            .expect("operation should succeed");
        assert!((layer.synapses[1][2].weight - 5.0).abs() < 1e-6);
    }

    #[test]
    fn spiking_network_creates_and_simulates() {
        let config = default_config();
        let mut net =
            SpikingNetwork::new(&[4, 3, 2], &config, 5.0, 0.1).expect("operation should succeed");
        let input: Vec<Vec<bool>> = (0..50).map(|_| vec![true, false, true, false]).collect();
        let result = net.simulate(&input, 50).expect("operation should succeed");
        assert_eq!(result.len(), 50);
        assert_eq!(result[0].len(), 2); // 2 layers
        assert_eq!(result[0][0].len(), 3); // first hidden layer has 3 neurons
        assert_eq!(result[0][1].len(), 2); // output layer has 2 neurons
    }

    #[test]
    fn spiking_network_spike_count_statistics() {
        let config = default_config();
        let mut net =
            SpikingNetwork::new(&[2, 3], &config, 20.0, 1.0).expect("operation should succeed");
        let input: Vec<Vec<bool>> = (0..100).map(|_| vec![true, true]).collect();
        let record = net.simulate(&input, 100).expect("operation should succeed");
        let total = SpikingNetwork::count_spikes(&record);
        let rates = SpikingNetwork::mean_firing_rates(&record);
        assert!(total > 0, "Some spikes expected");
        assert_eq!(rates.len(), 1);
    }

    #[test]
    fn spiking_network_rejects_bad_input_length() {
        let config = default_config();
        let mut net =
            SpikingNetwork::new(&[2, 3], &config, 1.0, 0.1).expect("operation should succeed");
        // 5 steps provided but t_steps=3
        let input: Vec<Vec<bool>> = vec![vec![true, false]; 5];
        assert!(net.simulate(&input, 3).is_err());
    }

    #[test]
    fn from_weights_roundtrip() {
        let weights = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let layer = SpikingLayer::from_weights(&weights, &default_config())
            .expect("operation should succeed");
        let recovered = layer.weights();
        for (r, expected) in recovered.iter().zip(weights.iter()) {
            for (&got, &exp) in r.iter().zip(expected.iter()) {
                assert!((got - exp).abs() < 1e-6);
            }
        }
    }
}
