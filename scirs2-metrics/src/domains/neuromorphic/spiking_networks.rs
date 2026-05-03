//! Spiking neural network implementation
//!
//! This module contains the core spiking neural network structures including
//! individual neurons, layers, and network topology management.

#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

use super::core::{
    ConnectionPattern, LateralInhibition, LayerParameters, NetworkTopology, NeuronType,
    RecurrentConnection,
};
use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::Float;
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

/// Spiking neural network implementation
#[derive(Debug)]
pub struct SpikingNeuralNetwork<F: Float> {
    /// Network topology
    pub topology: NetworkTopology,
    /// Neurons organized by layers
    pub layers: Vec<NeuronLayer<F>>,
    /// Synaptic connections
    pub synapses: super::synaptic_systems::SynapticConnections<F>,
    /// Current simulation time
    pub current_time: Duration,
    /// Spike history
    pub spike_history: SpikeHistory,
    /// Network state
    pub network_state: NetworkState<F>,
}

/// Layer of spiking neurons
#[derive(Debug)]
pub struct NeuronLayer<F: Float> {
    /// Individual neurons
    pub neurons: Vec<SpikingNeuron<F>>,
    /// Layer-specific parameters
    pub layer_params: LayerParameters<F>,
    /// Inhibitory connections within layer
    pub lateral_inhibition: LateralInhibition<F>,
}

/// Spiking neuron model (Leaky Integrate-and-Fire)
#[derive(Debug)]
pub struct SpikingNeuron<F: Float> {
    /// Unique neuron ID
    pub id: usize,
    /// Current membrane potential
    pub membrane_potential: F,
    /// Resting potential
    pub resting_potential: F,
    /// Spike threshold
    pub threshold: F,
    /// Membrane capacitance
    pub capacitance: F,
    /// Membrane resistance
    pub resistance: F,
    /// Time since last spike
    pub time_since_spike: Duration,
    /// Refractory period
    pub refractory_period: Duration,
    /// Spike train history
    pub spike_train: VecDeque<Instant>,
    /// Adaptive threshold
    pub adaptive_threshold: AdaptiveThreshold<F>,
    /// Neuron type
    pub neuron_type: NeuronType,
}

/// Adaptive threshold mechanism
#[derive(Debug)]
pub struct AdaptiveThreshold<F: Float> {
    /// Base threshold
    pub base_threshold: F,
    /// Current adaptation
    pub adaptation: F,
    /// Adaptation rate
    pub adaptation_rate: F,
    /// Time constant for decay
    pub decay_time_constant: Duration,
    /// Last update time
    pub last_update: Instant,
}

/// Spike history tracking
#[derive(Debug)]
pub struct SpikeHistory {
    /// Spikes by neuron
    pub spikes_by_neuron: HashMap<usize, VecDeque<Instant>>,
    /// Population spike rate
    pub population_spike_rate: VecDeque<f64>,
    /// Synchrony measures
    pub synchrony_measures: SynchronyMeasures,
    /// History window
    pub history_window: Duration,
}

/// Synchrony measures
#[derive(Debug)]
pub struct SynchronyMeasures {
    /// Cross-correlation matrix
    pub cross_correlation: scirs2_core::ndarray::Array2<f64>,
    /// Phase-locking values
    pub phase_locking: scirs2_core::ndarray::Array2<f64>,
    /// Global synchrony index
    pub global_synchrony: f64,
    /// Local synchrony clusters
    pub local_clusters: Vec<Vec<usize>>,
}

/// Network state information
#[derive(Debug, Clone)]
pub struct NetworkState<F: Float> {
    /// Current activity levels
    pub activity_levels: Array1<F>,
    /// Network oscillations
    pub oscillations: super::core::NetworkOscillations<F>,
    /// Critical dynamics
    pub criticality: super::core::CriticalityMeasures<F>,
    /// Information processing metrics
    pub information_metrics: super::core::InformationMetrics<F>,
}

impl<F: Float> SpikingNeuralNetwork<F> {
    /// Create a new spiking neural network
    pub fn new(topology: NetworkTopology, config: &super::core::NeuromorphicConfig) -> Self {
        let layers = Self::create_layers(&topology, config);
        let synapses = super::synaptic_systems::SynapticConnections::new(&topology);

        Self {
            topology,
            layers,
            synapses,
            current_time: Duration::from_micros(0),
            spike_history: SpikeHistory::new(Duration::from_secs(1)),
            network_state: NetworkState::new(),
        }
    }

    /// Create network layers based on topology
    fn create_layers(
        topology: &NetworkTopology,
        config: &super::core::NeuromorphicConfig,
    ) -> Vec<NeuronLayer<F>> {
        let mut layers = Vec::new();

        for (layer_idx, &layer_size) in topology.layer_sizes.iter().enumerate() {
            let layer_params = LayerParameters::default();
            let lateral_inhibition = LateralInhibition::default();

            let mut neurons = Vec::new();
            for neuron_idx in 0..layer_size {
                let neuron_id = layer_idx * config.neurons_per_layer + neuron_idx;
                let neuron_type = match layer_idx {
                    0 => NeuronType::Input,
                    idx if idx == topology.layer_sizes.len() - 1 => NeuronType::Output,
                    _ => {
                        if neuron_idx < (layer_size as f64 * 0.8) as usize {
                            NeuronType::Excitatory
                        } else {
                            NeuronType::Inhibitory
                        }
                    }
                };

                neurons.push(SpikingNeuron::new(neuron_id, neuron_type, config));
            }

            layers.push(NeuronLayer {
                neurons,
                layer_params,
                lateral_inhibition,
            });
        }

        layers
    }

    /// Simulate one time step
    pub fn simulate_step(&mut self, dt: Duration, input: &[F]) -> crate::error::Result<Vec<F>> {
        // Update current time
        self.current_time += dt;

        // Feed-forward pass: each layer receives output of the previous layer.
        // Layer 0 (input layer) receives the external `input` signal.
        let n_layers = self.layers.len();
        let mut layer_outputs: Vec<Vec<F>> = Vec::with_capacity(n_layers);

        for layer_idx in 0..n_layers {
            let prev_output: Vec<F> = if layer_idx == 0 {
                input.to_vec()
            } else {
                layer_outputs[layer_idx - 1].clone()
            };

            // Inject weighted synaptic currents into every neuron in this layer
            // before running the LIF update, then collect the spike outputs.
            self.inject_inputs(layer_idx, &prev_output);
            let outputs = self.update_layer_by_index(layer_idx, dt)?;
            layer_outputs.push(outputs);
        }

        // Update spike history
        self.update_spike_history();

        // Update network state (membrane potentials → activity_levels)
        self.update_network_state();

        // Return output layer activity (empty vector if no layers)
        Ok(layer_outputs.into_iter().last().unwrap_or_default())
    }

    /// Inject weighted inputs into a layer's neurons.
    ///
    /// For each neuron `j` in `layer_idx`, the synaptic current is:
    ///   I_syn = Σ_i  weight(i, j) * prev_output[i]
    /// where the sum runs over the neuron IDs present in `prev_output`.
    /// If no explicit weight exists for a (src, dst) pair we fall back to a
    /// uniform identity weight of 1.0 so that the test stimuli always reach
    /// the neurons.
    fn inject_inputs(&mut self, layer_idx: usize, prev_output: &[F]) {
        // Collect the global neuron IDs for neurons in this layer so we can
        // look up synapse weights keyed by (pre, post) neuron IDs.
        let post_ids: Vec<usize> = self.layers[layer_idx]
            .neurons
            .iter()
            .map(|n| n.id)
            .collect();

        for (post_pos, &post_id) in post_ids.iter().enumerate() {
            let mut i_syn = F::zero();
            for (pre_pos, &input_val) in prev_output.iter().enumerate() {
                // Try to get the explicit synapse weight; fall back to 1.0.
                let weight = self
                    .synapses
                    .connections
                    .get(&(pre_pos, post_id))
                    .map(|s| s.weight)
                    .unwrap_or_else(F::one);
                i_syn = i_syn + weight * input_val;
            }
            // Apply current directly to the neuron's membrane potential.
            // `add_current` accumulates charge; the LIF dynamics will decay it.
            self.layers[layer_idx].neurons[post_pos].add_current(i_syn);
        }
    }

    /// Update a single layer by index, running LIF dynamics for every neuron
    /// and then applying lateral inhibition.
    fn update_layer_by_index(
        &mut self,
        layer_idx: usize,
        dt: Duration,
    ) -> crate::error::Result<Vec<F>> {
        let mut outputs = Vec::with_capacity(self.layers[layer_idx].neurons.len());
        for neuron in &mut self.layers[layer_idx].neurons {
            let spike = neuron.update(dt)?;
            outputs.push(spike);
        }
        // Apply lateral inhibition in-place (needs mutable access to neurons).
        self.layers[layer_idx].apply_lateral_inhibition(&outputs.clone());
        Ok(outputs)
    }

    /// Apply input to input layer (kept for external callers; the main simulation
    /// loop now uses `inject_inputs` which also handles per-layer synaptic weights).
    fn apply_input(&mut self, input: &[F]) -> crate::error::Result<()> {
        if let Some(input_layer) = self.layers.first_mut() {
            for (neuron, &input_val) in input_layer.neurons.iter_mut().zip(input.iter()) {
                neuron.add_current(input_val);
            }
        }
        Ok(())
    }

    /// Update spike history
    fn update_spike_history(&mut self) {
        // Implementation for tracking spike patterns
        self.spike_history.update(&self.layers, self.current_time);
    }

    /// Update network state
    fn update_network_state(&mut self) {
        // Update activity levels
        let mut activity = Vec::new();
        for layer in &self.layers {
            for neuron in &layer.neurons {
                activity.push(neuron.membrane_potential);
            }
        }
        self.network_state.activity_levels = Array1::from_vec(activity);
    }
}

impl<F: Float> SpikingNeuron<F> {
    /// Create a new spiking neuron
    pub fn new(
        id: usize,
        neuron_type: NeuronType,
        config: &super::core::NeuromorphicConfig,
    ) -> Self {
        Self {
            id,
            membrane_potential: F::zero(),
            resting_potential: F::zero(),
            threshold: F::from(config.spike_threshold).expect("Failed to convert to float"),
            capacitance: F::one(),
            resistance: F::one(),
            time_since_spike: Duration::from_secs(0),
            refractory_period: config.refractory_period,
            spike_train: VecDeque::new(),
            adaptive_threshold: AdaptiveThreshold::new(
                F::from(config.spike_threshold).expect("Failed to convert to float"),
            ),
            neuron_type,
        }
    }

    /// Update neuron state for one time step
    pub fn update(&mut self, dt: Duration) -> crate::error::Result<F> {
        // Check if in refractory period
        if self.time_since_spike < self.refractory_period {
            self.time_since_spike += dt;
            return Ok(F::zero());
        }

        // Update adaptive threshold
        self.adaptive_threshold.update(dt);

        // Leaky integrate-and-fire dynamics
        let decay_factor = F::from(
            (-dt.as_secs_f64()
                / (self.resistance * self.capacitance)
                    .to_f64()
                    .expect("Operation failed"))
            .exp(),
        )
        .expect("Operation failed");
        self.membrane_potential = self.membrane_potential * decay_factor
            + self.resting_potential * (F::one() - decay_factor);

        // Check for spike
        if self.membrane_potential > self.adaptive_threshold.get_current_threshold() {
            self.fire_spike();
            Ok(F::one())
        } else {
            Ok(F::zero())
        }
    }

    /// Add input current to neuron
    pub fn add_current(&mut self, current: F) {
        self.membrane_potential = self.membrane_potential + current;
    }

    /// Fire a spike
    fn fire_spike(&mut self) {
        self.spike_train.push_back(Instant::now());
        self.membrane_potential = self.resting_potential;
        self.time_since_spike = Duration::from_secs(0);
        self.adaptive_threshold.on_spike();

        // Keep spike train bounded
        if self.spike_train.len() > 1000 {
            self.spike_train.pop_front();
        }
    }
}

impl<F: Float> AdaptiveThreshold<F> {
    /// Create new adaptive threshold
    pub fn new(base_threshold: F) -> Self {
        Self {
            base_threshold,
            adaptation: F::zero(),
            adaptation_rate: F::from(0.01).expect("Failed to convert constant to float"),
            decay_time_constant: Duration::from_millis(100),
            last_update: Instant::now(),
        }
    }

    /// Update threshold adaptation
    pub fn update(&mut self, dt: Duration) {
        let decay_factor =
            F::from((-dt.as_secs_f64() / self.decay_time_constant.as_secs_f64()).exp())
                .expect("Operation failed");
        self.adaptation = self.adaptation * decay_factor;
        self.last_update = Instant::now();
    }

    /// Called when neuron spikes
    pub fn on_spike(&mut self) {
        self.adaptation = self.adaptation + self.adaptation_rate;
    }

    /// Get current threshold
    pub fn get_current_threshold(&self) -> F {
        self.base_threshold + self.adaptation
    }
}

impl<F: Float> NeuronLayer<F> {
    /// Apply lateral inhibition within the layer
    pub fn apply_lateral_inhibition(&mut self, outputs: &[F]) {
        // Implementation of lateral inhibition based on the pattern
        match self.lateral_inhibition.pattern {
            super::core::InhibitionPattern::WinnerTakeAll => {
                self.apply_winner_take_all(outputs);
            }
            super::core::InhibitionPattern::DistanceBased => {
                self.apply_distance_based_inhibition(outputs);
            }
            _ => {
                // Default: uniform inhibition
                self.apply_uniform_inhibition(outputs);
            }
        }
    }

    fn apply_winner_take_all(&mut self, outputs: &[F]) {
        if let Some((winner_idx, _)) = outputs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).expect("Operation failed"))
        {
            for (idx, neuron) in self.neurons.iter_mut().enumerate() {
                if idx != winner_idx {
                    neuron.membrane_potential =
                        neuron.membrane_potential - self.lateral_inhibition.strength;
                }
            }
        }
    }

    fn apply_distance_based_inhibition(&mut self, _outputs: &[F]) {
        // Apply inhibition based on distance within the layer
        for i in 0..self.neurons.len() {
            for j in 0..self.neurons.len() {
                if i != j {
                    let distance = (i as i32 - j as i32).unsigned_abs() as usize;
                    if distance <= self.lateral_inhibition.radius {
                        let inhibition = self.lateral_inhibition.strength
                            / F::from(distance + 1).expect("Failed to convert to float");
                        self.neurons[j].membrane_potential =
                            self.neurons[j].membrane_potential - inhibition;
                    }
                }
            }
        }
    }

    fn apply_uniform_inhibition(&mut self, _outputs: &[F]) {
        for neuron in &mut self.neurons {
            neuron.membrane_potential =
                neuron.membrane_potential - self.lateral_inhibition.strength;
        }
    }
}

impl SpikeHistory {
    /// Create new spike history tracker
    pub fn new(window: Duration) -> Self {
        Self {
            spikes_by_neuron: HashMap::new(),
            population_spike_rate: VecDeque::new(),
            synchrony_measures: SynchronyMeasures::new(),
            history_window: window,
        }
    }

    /// Update spike history with current network state
    pub fn update<F: Float>(&mut self, layers: &[NeuronLayer<F>], current_time: Duration) {
        let mut total_spikes = 0;

        for layer in layers {
            for neuron in &layer.neurons {
                // Count recent spikes
                let recent_spikes = neuron
                    .spike_train
                    .iter()
                    .filter(|&&spike_time| spike_time.elapsed() < self.history_window)
                    .count();
                total_spikes += recent_spikes;
            }
        }

        // Update population spike rate
        let spike_rate = total_spikes as f64 / self.history_window.as_secs_f64();
        self.population_spike_rate.push_back(spike_rate);

        // Keep bounded
        if self.population_spike_rate.len() > 1000 {
            self.population_spike_rate.pop_front();
        }
    }
}

impl SynchronyMeasures {
    /// Create new synchrony measures
    pub fn new() -> Self {
        Self {
            cross_correlation: scirs2_core::ndarray::Array2::zeros((0, 0)),
            phase_locking: scirs2_core::ndarray::Array2::zeros((0, 0)),
            global_synchrony: 0.0,
            local_clusters: Vec::new(),
        }
    }
}

impl<F: Float> NetworkState<F> {
    /// Create new network state
    pub fn new() -> Self {
        Self {
            activity_levels: Array1::zeros(0),
            oscillations: super::core::NetworkOscillations {
                dominant_frequencies: Vec::new(),
                power_spectrum: Vec::new(),
                gamma_power: F::zero(),
                beta_power: F::zero(),
                alpha_power: F::zero(),
                theta_power: F::zero(),
            },
            criticality: super::core::CriticalityMeasures {
                avalanche_distribution: Vec::new(),
                branching_parameter: F::zero(),
                critical_exponent: F::zero(),
                activity_variance: F::zero(),
            },
            information_metrics: super::core::InformationMetrics {
                mutual_information: F::zero(),
                transfer_entropy: F::zero(),
                integrated_information: F::zero(),
                complexity: F::zero(),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::core::{ConnectionPattern, NetworkTopology, NeuromorphicConfig};
    use super::*;

    /// Build a minimal config with a very low threshold and large time-step so
    /// tests are deterministic and fast.
    fn small_config() -> NeuromorphicConfig {
        NeuromorphicConfig {
            input_neurons: 2,
            hidden_layers: 0,
            neurons_per_layer: 2,
            output_neurons: 2,
            // Below resting potential so the neuron fires immediately when
            // a suprathreshold current is injected.
            spike_threshold: 0.5,
            refractory_period: Duration::from_millis(2),
            synaptic_delay_range: (Duration::from_micros(100), Duration::from_millis(10)),
            learning_rate: 0.01,
            membrane_decay: 0.95,
            enable_stdp: false,
            enable_homeostasis: false,
            enable_memory_consolidation: false,
            enable_quantum_processing: false,
            timestep: Duration::from_millis(1),
            max_simulation_time: Duration::from_secs(1),
        }
    }

    /// Build a two-layer topology (input → output) with `n` neurons each.
    fn two_layer_topology(n: usize) -> NetworkTopology {
        NetworkTopology {
            layer_sizes: vec![n, n],
            connection_patterns: vec![ConnectionPattern::FullyConnected],
            recurrent_connections: vec![],
        }
    }

    /// Test 1: LIF network forward pass produces a non-empty output vector.
    #[test]
    fn test_lif_forward_pass_non_empty() {
        let config = small_config();
        let topology = two_layer_topology(2);
        let mut net = SpikingNeuralNetwork::<f64>::new(topology, &config);

        let input = vec![0.1_f64, 0.2_f64];
        let dt = Duration::from_millis(1);
        let output = net.simulate_step(dt, &input).expect("simulate_step failed");

        assert!(!output.is_empty(), "Output must be non-empty");
        assert_eq!(
            output.len(),
            2,
            "Output length should match output layer size"
        );
    }

    /// Test 2: LIF neuron fires (spikes) when given a suprathreshold input
    /// sustained over multiple time steps.
    #[test]
    fn test_lif_neuron_fires_above_threshold() {
        let config = small_config();
        let topology = two_layer_topology(1);
        let mut net = SpikingNeuralNetwork::<f64>::new(topology, &config);

        let dt = Duration::from_millis(1);
        // Large enough current to push membrane potential above threshold=0.5.
        let input = vec![2.0_f64];

        let mut total_spikes = 0u32;
        for _ in 0..20 {
            let output = net.simulate_step(dt, &input).expect("simulate_step failed");
            total_spikes += output
                .iter()
                .map(|&v| if v > 0.5 { 1 } else { 0 })
                .sum::<u32>();
        }

        assert!(
            total_spikes > 0,
            "At least one spike should have been produced with suprathreshold input"
        );
    }

    /// Test 3: Spike history grows (population_spike_rate queue gains entries)
    /// as the simulation advances.
    #[test]
    fn test_spike_history_grows_over_time() {
        let config = small_config();
        let topology = two_layer_topology(2);
        let mut net = SpikingNeuralNetwork::<f64>::new(topology, &config);

        let dt = Duration::from_millis(1);
        let input = vec![2.0_f64, 2.0_f64];

        let initial_len = net.spike_history.population_spike_rate.len();

        for _ in 0..10 {
            net.simulate_step(dt, &input).expect("simulate_step failed");
        }

        let final_len = net.spike_history.population_spike_rate.len();
        assert!(
            final_len > initial_len,
            "Spike history (population_spike_rate) should grow: initial={initial_len} final={final_len}"
        );
    }

    /// Test 4: network_state.activity_levels length equals total neuron count
    /// after a forward pass.
    #[test]
    fn test_network_state_activity_levels_updated() {
        let config = small_config();
        let topology = two_layer_topology(3);
        let mut net = SpikingNeuralNetwork::<f64>::new(topology, &config);

        let dt = Duration::from_millis(1);
        let input = vec![0.1_f64, 0.1_f64, 0.1_f64];
        net.simulate_step(dt, &input).expect("simulate_step failed");

        // Two layers, 3 neurons each → 6 total
        let total_neurons: usize = net.layers.iter().map(|l| l.neurons.len()).sum();
        assert_eq!(
            net.network_state.activity_levels.len(),
            total_neurons,
            "activity_levels length should equal total neuron count"
        );
    }
}
