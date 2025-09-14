//! Spiking Neural Network Optimization
//!
//! This module implements optimization algorithms based on spiking neural networks,
//! which process information using discrete spike events rather than continuous signals.

use super::{NeuromorphicConfig, SpikeEvent};
use ndarray::{Array1, ArrayView1};
use rand::Rng;
use scirs2_core::error::CoreResult as Result;
use std::collections::VecDeque;

/// Spiking neural network for optimization
#[derive(Debug, Clone)]
pub struct SpikingNeuralNetwork {
    /// Network configuration
    pub config: NeuromorphicConfig,
    /// Neuron states
    pub neurons: Vec<SpikingNeuron>,
    /// Synaptic connections
    pub synapses: Vec<Vec<Synapse>>,
    /// Current simulation time
    pub current_time: f64,
    /// Spike history buffer
    pub spike_history: VecDeque<SpikeEvent>,
    /// Population activity monitor
    pub population_activity: Array1<f64>,
}

/// Spiking neuron model (Leaky Integrate-and-Fire)
#[derive(Debug, Clone)]
pub struct SpikingNeuron {
    /// Membrane potential
    pub membrane_potential: f64,
    /// Resting potential
    pub resting_potential: f64,
    /// Spike threshold
    pub threshold: f64,
    /// Membrane time constant
    pub tau_membrane: f64,
    /// Refractory period
    pub refractory_period: f64,
    /// Time of last spike
    pub last_spike_time: Option<f64>,
    /// Input current
    pub input_current: f64,
    /// Adaptation current
    pub adaptation_current: f64,
    /// Noise level
    pub noise_amplitude: f64,
}

/// Synaptic connection between neurons
#[derive(Debug, Clone)]
pub struct Synapse {
    /// Source neuron index
    pub source: usize,
    /// Target neuron index
    pub target: usize,
    /// Synaptic weight
    pub weight: f64,
    /// Synaptic delay
    pub delay: f64,
    /// Short-term plasticity variables
    pub facilitation: f64,
    pub depression: f64,
    /// STDP trace variables
    pub pre_trace: f64,
    pub post_trace: f64,
}

impl SpikingNeuron {
    /// Create a new LIF neuron
    pub fn new(config: &NeuromorphicConfig) -> Self {
        Self {
            membrane_potential: 0.0,
            resting_potential: 0.0,
            threshold: config.spike_threshold,
            tau_membrane: 0.020, // 20ms membrane time constant
            refractory_period: config.refractory_period,
            last_spike_time: None,
            input_current: 0.0,
            adaptation_current: 0.0,
            noise_amplitude: config.noise_level,
        }
    }

    /// Update neuron state for one time step
    pub fn update(&mut self, dt: f64, external_current: f64, current_time: f64) -> Option<f64> {
        // Check if in refractory period
        if let Some(last_spike) = self.last_spike_time {
            if (current_time - last_spike) < self.refractory_period {
                return None; // Still refractory
            }
        }

        // Add noise
        let noise = if self.noise_amplitude > 0.0 {
            let mut rng = rand::rng();
            (rng.random::<f64>() - 0.5) * 2.0 * self.noise_amplitude
        } else {
            0.0
        };

        // Leaky integrate-and-fire dynamics
        let total_current = external_current + self.input_current - self.adaptation_current + noise;
        let dv_dt = (-(self.membrane_potential - self.resting_potential) + total_current)
            / self.tau_membrane;

        self.membrane_potential += dv_dt * dt;

        // Check for spike
        if self.membrane_potential >= self.threshold {
            self.fire_spike();
            Some(0.0) // Return spike time (relative to current time)
        } else {
            None
        }
    }

    /// Fire a spike and reset membrane potential
    fn fire_spike(&mut self) {
        self.membrane_potential = self.resting_potential;
        self.last_spike_time = Some(0.0); // Will be updated by caller

        // Spike-triggered adaptation
        self.adaptation_current += 0.1; // Simple adaptation increment
    }

    /// Decay adaptation current
    pub fn decay_adaptation(&mut self, dt: f64) {
        let tau_adaptation = 0.1; // 100ms adaptation time constant
        self.adaptation_current *= (-dt / tau_adaptation).exp();
    }
}

impl Synapse {
    /// Create a new synapse
    pub fn new(source: usize, target: usize, weight: f64, delay: f64) -> Self {
        Self {
            source,
            target,
            weight,
            delay,
            facilitation: 1.0,
            depression: 1.0,
            pre_trace: 0.0,
            post_trace: 0.0,
        }
    }

    /// Compute synaptic current
    pub fn compute_current(&self, pre_spike: bool) -> f64 {
        if pre_spike {
            self.weight * self.facilitation * self.depression
        } else {
            0.0
        }
    }

    /// Update short-term plasticity
    pub fn update_stp(&mut self, dt: f64, pre_spike: bool) {
        let tau_facilitation = 0.050; // 50ms
        let tau_depression = 0.100; // 100ms

        // Decay
        self.facilitation += (1.0 - self.facilitation) * dt / tau_facilitation;
        self.depression += (1.0 - self.depression) * dt / tau_depression;

        if pre_spike {
            self.facilitation = (self.facilitation * 1.2).min(3.0); // Facilitate
            self.depression *= 0.8; // Depress
        }
    }

    /// Update STDP traces
    pub fn update_stdp_traces(&mut self, dt: f64, pre_spike: bool, post_spike: bool) {
        let tau_stdp = 0.020; // 20ms STDP time constant

        // Decay traces
        self.pre_trace *= (-dt / tau_stdp).exp();
        self.post_trace *= (-dt / tau_stdp).exp();

        // Update traces on spikes
        if pre_spike {
            self.pre_trace += 1.0;
        }
        if post_spike {
            self.post_trace += 1.0;
        }
    }

    /// Apply STDP weight update
    pub fn apply_stdp(&mut self, learning_rate: f64, pre_spike: bool, post_spike: bool) {
        let mut weight_change = 0.0;

        if pre_spike && self.post_trace > 0.0 {
            // Pre-before-post: potentiation
            weight_change += learning_rate * self.post_trace;
        }

        if post_spike && self.pre_trace > 0.0 {
            // Post-before-pre: depression
            weight_change -= learning_rate * 0.5 * self.pre_trace;
        }

        self.weight += weight_change;
        self.weight = self.weight.max(-1.0).min(1.0); // Bound weights
    }
}

impl SpikingNeuralNetwork {
    /// Create a new spiking neural network
    pub fn new(config: NeuromorphicConfig, num_parameters: usize) -> Self {
        let mut neurons = Vec::with_capacity(config.num_neurons);
        for _ in 0..config.num_neurons {
            neurons.push(SpikingNeuron::new(&config));
        }

        // Create random connectivity
        let mut synapses = vec![Vec::new(); config.num_neurons];
        let connection_probability = 0.1; // 10% connection probability
        let mut rng = rand::rng();

        for i in 0..config.num_neurons {
            for j in 0..config.num_neurons {
                if i != j && rng.random::<f64>() < connection_probability {
                    let weight = (rng.random::<f64>() - 0.5) * 0.2;
                    let delay = rng.random::<f64>() * 0.005; // 0-5ms delay
                    synapses[i].push(Synapse::new(i, j, weight, delay));
                }
            }
        }

        let num_neurons = config.num_neurons;
        Self {
            config,
            neurons,
            synapses,
            current_time: 0.0,
            spike_history: VecDeque::with_capacity(10000),
            population_activity: Array1::zeros(num_neurons),
        }
    }

    /// Encode parameters as spike trains
    pub fn encode_parameters(&mut self, parameters: &ArrayView1<f64>) {
        let neurons_per_param = self.config.num_neurons / parameters.len();

        for (param_idx, &param_val) in parameters.iter().enumerate() {
            let start_idx = param_idx * neurons_per_param;
            let end_idx = ((param_idx + 1) * neurons_per_param).min(self.config.num_neurons);

            // Rate coding: parameter value determines input current
            let input_current = (param_val + 1.0) * 5.0; // Scale to reasonable range

            for neuron_idx in start_idx..end_idx {
                self.neurons[neuron_idx].input_current = input_current;
            }
        }
    }

    /// Decode parameters from population activity
    pub fn decode_parameters(&self, num_parameters: usize) -> Array1<f64> {
        let mut decoded = Array1::zeros(num_parameters);
        let neurons_per_param = self.config.num_neurons / num_parameters;

        for param_idx in 0..num_parameters {
            let start_idx = param_idx * neurons_per_param;
            let end_idx = ((param_idx + 1) * neurons_per_param).min(self.config.num_neurons);

            // Average population activity
            let mut activity_sum = 0.0;
            for neuron_idx in start_idx..end_idx {
                activity_sum += self.population_activity[neuron_idx];
            }

            if end_idx > start_idx {
                decoded[param_idx] = (activity_sum / (end_idx - start_idx) as f64) - 1.0;
            }
        }

        decoded
    }

    /// Simulate one time step
    pub fn simulate_step(&mut self, objective_feedback: f64) -> Result<Vec<usize>> {
        let mut spiked_neurons = Vec::new();

        // Collect inputs for all neurons first to avoid borrow checker issues
        let inputs: Vec<(f64, f64)> = (0..self.neurons.len())
            .map(|neuron_idx| {
                let synaptic_input = self.compute_synaptic_input(neuron_idx);
                let feedback_input = self.compute_feedback_input(neuron_idx, objective_feedback);
                (synaptic_input, feedback_input)
            })
            .collect();

        // Update all neurons
        for (neuron_idx, neuron) in self.neurons.iter_mut().enumerate() {
            let (synaptic_input, feedback_input) = inputs[neuron_idx];
            let total_input = synaptic_input + feedback_input;

            // Update neuron
            if let Some(_spike_time) = neuron.update(self.config.dt, total_input, self.current_time)
            {
                spiked_neurons.push(neuron_idx);
                neuron.last_spike_time = Some(self.current_time);

                // Record spike
                self.spike_history.push_back(SpikeEvent {
                    time: self.current_time,
                    neuron_id: neuron_idx,
                    weight: 1.0,
                });

                // Update population activity
                self.population_activity[neuron_idx] = 1.0;
            } else {
                // Decay population activity
                self.population_activity[neuron_idx] *= 0.95;
            }

            // Decay adaptation
            neuron.decay_adaptation(self.config.dt);
        }

        // Update synapses
        self.update_synapses(&spiked_neurons)?;

        // Cleanup old spikes
        self.cleanup_spike_history();

        self.current_time += self.config.dt;

        Ok(spiked_neurons)
    }

    /// Compute synaptic input for a neuron
    fn compute_synaptic_input(&self, target_neuron: usize) -> f64 {
        let mut total_input = 0.0;

        // Check all neurons for connections to target
        for source_neuron in 0..self.config.num_neurons {
            for synapse in &self.synapses[source_neuron] {
                if synapse.target == target_neuron {
                    // Check if source neuron spiked recently (within delay)
                    if let Some(last_spike) = self.neurons[source_neuron].last_spike_time {
                        let time_since_spike = self.current_time - last_spike;
                        if time_since_spike >= synapse.delay
                            && time_since_spike < synapse.delay + self.config.dt
                        {
                            total_input += synapse.compute_current(true);
                        }
                    }
                }
            }
        }

        total_input
    }

    /// Compute objective-based feedback input
    fn compute_feedback_input(&self, neuron_idx: usize, objective_feedback: f64) -> f64 {
        // Simple feedback scheme: better objective values give positive input
        let feedback_strength = 1.0;
        let normalized_feedback = -objective_feedback; // Assume minimization

        // Different neurons get different phases of feedback
        let phase = neuron_idx as f64 / self.config.num_neurons as f64 * 2.0 * std::f64::consts::PI;
        feedback_strength * normalized_feedback * (phase.sin() + 1.0) * 0.5
    }

    /// Update synaptic plasticity
    fn update_synapses(&mut self, spiked_neurons: &[usize]) -> Result<()> {
        for source_neuron in 0..self.config.num_neurons {
            let source_spiked = spiked_neurons.contains(&source_neuron);

            for synapse in &mut self.synapses[source_neuron] {
                let target_spiked = spiked_neurons.contains(&synapse.target);

                // Update short-term plasticity
                synapse.update_stp(self.config.dt, source_spiked);

                // Update STDP traces
                synapse.update_stdp_traces(self.config.dt, source_spiked, target_spiked);

                // Apply STDP weight updates
                synapse.apply_stdp(self.config.learning_rate, source_spiked, target_spiked);
            }
        }

        Ok(())
    }

    /// Remove old spikes from history
    fn cleanup_spike_history(&mut self) {
        let cutoff_time = self.current_time - 0.1; // Keep 100ms of history
        while let Some(spike) = self.spike_history.front() {
            if spike.time < cutoff_time {
                self.spike_history.pop_front();
            } else {
                break;
            }
        }
    }

    /// Get firing rates over recent window
    pub fn get_firing_rates(&self, window_duration: f64) -> Array1<f64> {
        let mut rates = Array1::zeros(self.config.num_neurons);
        let start_time = self.current_time - window_duration;

        for spike in &self.spike_history {
            if spike.time >= start_time {
                rates[spike.neuron_id] += 1.0;
            }
        }

        // Convert to Hz
        rates /= window_duration;
        rates
    }

    /// Reset network state
    pub fn reset(&mut self) {
        self.current_time = 0.0;
        self.spike_history.clear();
        self.population_activity.fill(0.0);

        for neuron in &mut self.neurons {
            neuron.membrane_potential = neuron.resting_potential;
            neuron.last_spike_time = None;
            neuron.input_current = 0.0;
            neuron.adaptation_current = 0.0;
        }

        // Reset synaptic state
        for synapse_group in &mut self.synapses {
            for synapse in synapse_group {
                synapse.facilitation = 1.0;
                synapse.depression = 1.0;
                synapse.pre_trace = 0.0;
                synapse.post_trace = 0.0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spiking_neuron_creation() {
        let config = NeuromorphicConfig::default();
        let neuron = SpikingNeuron::new(&config);

        assert_eq!(neuron.membrane_potential, 0.0);
        assert_eq!(neuron.threshold, config.spike_threshold);
        assert!(neuron.last_spike_time.is_none());
    }

    #[test]
    fn test_neuron_spike() {
        let config = NeuromorphicConfig::default();
        let mut neuron = SpikingNeuron::new(&config);

        // Apply strong input to cause spike
        let spike_time = neuron.update(0.001, 50.0, 0.0);
        assert!(spike_time.is_some());
        assert_eq!(neuron.membrane_potential, neuron.resting_potential);
    }

    #[test]
    fn test_synapse_creation() {
        let synapse = Synapse::new(0, 1, 0.5, 0.002);

        assert_eq!(synapse.source, 0);
        assert_eq!(synapse.target, 1);
        assert_eq!(synapse.weight, 0.5);
        assert_eq!(synapse.delay, 0.002);
    }

    #[test]
    fn test_synapse_current() {
        let mut synapse = Synapse::new(0, 1, 0.5, 0.001);

        // No current without spike
        assert_eq!(synapse.compute_current(false), 0.0);

        // Current with spike
        let current = synapse.compute_current(true);
        assert!(current > 0.0);

        // Test short-term plasticity
        synapse.update_stp(0.001, true);
        let current_after_stp = synapse.compute_current(true);
        assert!(current_after_stp != current); // Should change due to plasticity
    }

    #[test]
    fn test_spiking_network_creation() {
        let config = NeuromorphicConfig::default();
        let network = SpikingNeuralNetwork::new(config, 3);

        assert_eq!(network.neurons.len(), 100); // Default num_neurons
        assert_eq!(network.synapses.len(), 100);
        assert_eq!(network.current_time, 0.0);
    }

    #[test]
    fn test_parameter_encoding() {
        let config = NeuromorphicConfig::default();
        let mut network = SpikingNeuralNetwork::new(config, 2);

        let params = Array1::from(vec![0.5, -0.3]);
        network.encode_parameters(&params.view());

        // Check that some neurons received input
        assert!(network.neurons.iter().any(|n| n.input_current != 0.0));
    }

    #[test]
    fn test_network_simulation() {
        let config = NeuromorphicConfig {
            num_neurons: 10,
            ..Default::default()
        };
        let mut network = SpikingNeuralNetwork::new(config, 2);

        // Simulate a few steps
        for _ in 0..10 {
            let _spiked = network.simulate_step(1.0).unwrap();
            // Should complete without error
        }

        assert!(network.current_time > 0.0);
    }

    #[test]
    fn test_firing_rates() {
        let config = NeuromorphicConfig {
            num_neurons: 5,
            ..Default::default()
        };
        let mut network = SpikingNeuralNetwork::new(config, 1);

        // Force some spikes by setting high input
        for neuron in &mut network.neurons {
            neuron.input_current = 20.0;
        }

        // Simulate to generate spikes
        for _ in 0..100 {
            network.simulate_step(0.0).unwrap();
        }

        let rates = network.get_firing_rates(0.1);
        assert!(rates.iter().any(|&r| r > 0.0)); // Should have some firing
    }

    #[test]
    fn test_network_reset() {
        let config = NeuromorphicConfig::default();
        let mut network = SpikingNeuralNetwork::new(config, 2);

        // Simulate to change state
        for _ in 0..10 {
            network.simulate_step(1.0).unwrap();
        }

        let _time_before_reset = network.current_time;
        network.reset();

        assert_eq!(network.current_time, 0.0);
        assert!(network.spike_history.is_empty());
        assert!(network.population_activity.iter().all(|&x| x == 0.0));
    }
}
