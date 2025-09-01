//! Neuromorphic Optimization Module
//!
//! This module implements optimization algorithms inspired by neuromorphic computing
//! and neural network architectures. These methods leverage principles from
//! biological neural networks and neuromorphic hardware for efficient optimization.
//!
//! # Key Features
//!
//! - **Spiking Neural Network Optimization**: Event-driven optimization using spike trains
//! - **Memristive Optimization**: Algorithms that mimic memristor behavior for adaptive optimization
//! - **Spike-Timing Dependent Plasticity (STDP)**: Learning rules based on spike timing
//! - **Neuromorphic Gradient Descent**: Event-driven gradient computation
//! - **Liquid State Machines**: Reservoir computing for optimization
//! - **Neural ODE Optimization**: Continuous-time neural network optimization
//!
//! # Applications
//!
//! - Low-power optimization for edge devices
//! - Real-time adaptive control systems
//! - Bio-inspired machine learning
//! - Neuromorphic hardware optimization
//! - Event-driven optimization problems

use crate::error::OptimizeResult;
use crate::result::OptimizeResults;
use ndarray::{Array1, Array2, ArrayView1};
use rand::Rng;
use scirs2_core::error::CoreResult as Result;

pub mod event_driven;
pub mod liquid_state_machines;
pub mod memristive_optimization;
pub mod neural_ode_optimization;
pub mod spiking_networks;
pub mod stdp_learning;

// Use glob re-exports with allow for ambiguous names
#[allow(ambiguous_glob_reexports)]
pub use event_driven::*;
#[allow(ambiguous_glob_reexports)]
pub use liquid_state_machines::*;
#[allow(ambiguous_glob_reexports)]
pub use memristive_optimization::*;
#[allow(ambiguous_glob_reexports)]
pub use neural_ode_optimization::*;
#[allow(ambiguous_glob_reexports)]
pub use spiking_networks::*;
#[allow(ambiguous_glob_reexports)]
pub use stdp_learning::*;

/// Configuration for neuromorphic optimization algorithms
#[derive(Debug, Clone)]
pub struct NeuromorphicConfig {
    /// Time step for discrete-time simulations
    pub dt: f64,
    /// Total simulation time
    pub total_time: f64,
    /// Number of neurons/nodes in the network
    pub num_neurons: usize,
    /// Spike threshold for spiking neurons
    pub spike_threshold: f64,
    /// Refractory period after spike
    pub refractory_period: f64,
    /// Learning rate for synaptic plasticity
    pub learning_rate: f64,
    /// Decay rate for membrane potential
    pub membrane_decay: f64,
    /// Noise level for stochastic processes
    pub noise_level: f64,
    /// Whether to use event-driven simulation
    pub event_driven: bool,
}

impl Default for NeuromorphicConfig {
    fn default() -> Self {
        Self {
            dt: 0.001,                // 1ms time step
            total_time: 1.0,          // 1 second simulation
            num_neurons: 100,         // 100 neurons
            spike_threshold: 1.0,     // Normalized threshold
            refractory_period: 0.002, // 2ms refractory period
            learning_rate: 0.01,
            membrane_decay: 0.95, // Exponential decay factor
            noise_level: 0.01,
            event_driven: true,
        }
    }
}

/// Spike event in neuromorphic simulation
#[derive(Debug, Clone, Copy)]
pub struct SpikeEvent {
    /// Time of the spike
    pub time: f64,
    /// ID of the neuron that spiked
    pub neuron_id: usize,
    /// Strength/weight of the spike
    pub weight: f64,
}

/// State of a neuromorphic neuron
#[derive(Debug, Clone)]
pub struct NeuronState {
    /// Membrane potential
    pub potential: f64,
    /// Last spike time
    pub last_spike_time: Option<f64>,
    /// Synaptic weights (connections to other neurons)
    pub weights: Array1<f64>,
    /// Current input current
    pub input_current: f64,
    /// Adaptation variable (for adaptive neurons)
    pub adaptation: f64,
}

impl NeuronState {
    /// Create a new neuron state
    pub fn new(num_connections: usize) -> Self {
        Self {
            potential: 0.0,
            last_spike_time: None,
            weights: Array1::zeros(num_connections),
            input_current: 0.0,
            adaptation: 0.0,
        }
    }

    /// Check if neuron is in refractory period
    pub fn is_refractory(&self, current_time: f64, refractory_period: f64) -> bool {
        if let Some(last_spike) = self.last_spike_time {
            current_time - last_spike < refractory_period
        } else {
            false
        }
    }

    /// Update membrane potential (integrate-and-fire dynamics)
    pub fn update_potential(&mut self, dt: f64, decay: f64, input: f64) {
        if !self.is_refractory(0.0, 0.0) {
            // Simplified check
            self.potential = self.potential * decay + input * dt;
        }
    }

    /// Check if neuron should spike
    pub fn should_spike(&self, threshold: f64) -> bool {
        self.potential >= threshold
    }

    /// Fire a spike (reset potential and record time)
    pub fn fire_spike(&mut self, time: f64) {
        self.potential = 0.0; // Reset potential
        self.last_spike_time = Some(time);
    }
}

/// Neuromorphic optimization network
#[derive(Debug, Clone)]
pub struct NeuromorphicNetwork {
    /// Network configuration
    config: NeuromorphicConfig,
    /// States of all neurons
    neurons: Vec<NeuronState>,
    /// Connectivity matrix (which neurons connect to which)
    connectivity: Array2<f64>,
    /// Current simulation time
    current_time: f64,
    /// Spike events queue for event-driven simulation
    spike_queue: Vec<SpikeEvent>,
    /// Objective function value history
    objective_history: Vec<f64>,
    /// Current parameter estimates
    parameters: Array1<f64>,
}

impl NeuromorphicNetwork {
    /// Create a new neuromorphic network
    pub fn new(config: NeuromorphicConfig, num_parameters: usize) -> Self {
        let mut neurons = Vec::with_capacity(config.num_neurons);
        for _ in 0..config.num_neurons {
            neurons.push(NeuronState::new(config.num_neurons));
        }

        // Initialize random connectivity
        let mut connectivity = Array2::zeros((config.num_neurons, config.num_neurons));
        for i in 0..config.num_neurons {
            for j in 0..config.num_neurons {
                if i != j {
                    // Random connection strength
                    connectivity[[i, j]] = rand::rng().random_range(-0.05..0.05);
                }
            }
        }

        Self {
            config,
            neurons,
            connectivity,
            current_time: 0.0,
            spike_queue: Vec::new(),
            objective_history: Vec::new(),
            parameters: Array1::zeros(num_parameters),
        }
    }

    /// Encode parameters as neural activity
    pub fn encode_parameters(&mut self, parameters: &ArrayView1<f64>) {
        let n_params = parameters.len();
        let neurons_per_param = self.config.num_neurons / n_params;

        for (i, &param) in parameters.iter().enumerate() {
            let start_idx = i * neurons_per_param;
            let end_idx = ((i + 1) * neurons_per_param).min(self.config.num_neurons);

            // Rate coding: parameter value determines firing rate
            let firing_rate = (param + 1.0) * 10.0; // Scale to reasonable firing rate

            for j in start_idx..end_idx {
                // Inject current proportional to desired firing rate
                self.neurons[j].input_current = firing_rate * 0.01;
            }
        }
    }

    /// Decode parameters from neural activity
    pub fn decode_parameters(&self) -> Array1<f64> {
        let n_params = self.parameters.len();
        let neurons_per_param = self.config.num_neurons / n_params;
        let mut decoded = Array1::zeros(n_params);

        for i in 0..n_params {
            let start_idx = i * neurons_per_param;
            let end_idx = ((i + 1) * neurons_per_param).min(self.config.num_neurons);

            // Average membrane potential as parameter estimate
            let mut sum = 0.0;
            for j in start_idx..end_idx {
                sum += self.neurons[j].potential;
            }

            if end_idx > start_idx {
                decoded[i] = sum / (end_idx - start_idx) as f64;
            }
        }

        decoded
    }

    /// Simulate one time step
    pub fn simulate_step(&mut self, objective_value: f64) -> Result<()> {
        // Process spike queue for event-driven simulation
        if self.config.event_driven {
            self.process_spike_events()?;
        }

        // Update all neurons
        for i in 0..self.config.num_neurons {
            // Calculate synaptic input from other neurons
            let mut synaptic_input = 0.0;
            for j in 0..self.config.num_neurons {
                if i != j {
                    synaptic_input += self.connectivity[[j, i]] * self.neurons[j].potential;
                }
            }

            // Add external input (objective-based feedback)
            let external_input = self.compute_external_input(i, objective_value);
            let total_input = synaptic_input + external_input + self.neurons[i].input_current;

            // Update membrane potential
            self.neurons[i].update_potential(
                self.config.dt,
                self.config.membrane_decay,
                total_input,
            );

            // Add noise
            if self.config.noise_level > 0.0 {
                let noise =
                    rand::rng().random_range(-self.config.noise_level..self.config.noise_level);
                self.neurons[i].potential += noise;
            }

            // Check for spike
            if self.neurons[i].should_spike(self.config.spike_threshold)
                && !self.neurons[i].is_refractory(self.current_time, self.config.refractory_period)
            {
                self.neurons[i].fire_spike(self.current_time);

                // Add spike event to queue
                self.spike_queue.push(SpikeEvent {
                    time: self.current_time,
                    neuron_id: i,
                    weight: 1.0,
                });
            }
        }

        // Update connectivity based on STDP-like rules
        self.update_connectivity(objective_value)?;

        // Update time
        self.current_time += self.config.dt;

        // Store objective value
        self.objective_history.push(objective_value);

        Ok(())
    }

    /// Process spike events for event-driven dynamics
    fn process_spike_events(&mut self) -> Result<()> {
        // Sort spike events by time
        self.spike_queue
            .sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap());

        // Process events that should have occurred by now
        while let Some(event) = self.spike_queue.first() {
            if event.time <= self.current_time {
                let event = self.spike_queue.remove(0);
                self.process_spike_event(event)?;
            } else {
                break;
            }
        }

        Ok(())
    }

    /// Process individual spike event
    fn process_spike_event(&mut self, event: SpikeEvent) -> Result<()> {
        // Propagate spike effects to connected neurons
        for i in 0..self.config.num_neurons {
            if i != event.neuron_id {
                let connection_strength = self.connectivity[[event.neuron_id, i]];
                self.neurons[i].potential += connection_strength * event.weight;
            }
        }

        Ok(())
    }

    /// Compute external input for objective-based feedback
    fn compute_external_input(&self, neuron_id: usize, objective_value: f64) -> f64 {
        // Simple feedback: better objective values lead to positive input
        let feedback_strength = 0.1;
        let normalized_objective = -objective_value; // Assume minimization

        // Different neurons can receive different feedback patterns
        let phase = neuron_id as f64 / self.config.num_neurons as f64 * 2.0 * std::f64::consts::PI;
        feedback_strength * normalized_objective * phase.sin()
    }

    /// Update connectivity using STDP-like plasticity rules
    fn update_connectivity(&mut self, objective_value: f64) -> Result<()> {
        // Simplified STDP: strengthen connections that led to better performance
        let performance_factor = if self.objective_history.len() > 1 {
            let prev_objective = self.objective_history[self.objective_history.len() - 2];
            if objective_value < prev_objective {
                1.0
            } else {
                -0.1
            }
        } else {
            0.0
        };

        // Update connections based on recent spike correlations
        for i in 0..self.config.num_neurons {
            for j in 0..self.config.num_neurons {
                if i != j {
                    // Simplified plasticity rule
                    let correlation = self.neurons[i].potential * self.neurons[j].potential;
                    let weight_change =
                        self.config.learning_rate * performance_factor * correlation;

                    self.connectivity[[i, j]] += weight_change;

                    // Bound weights
                    self.connectivity[[i, j]] = self.connectivity[[i, j]].max(-1.0).min(1.0);
                }
            }
        }

        Ok(())
    }
}

/// Trait for neuromorphic optimization algorithms
pub trait NeuromorphicOptimizer {
    /// Configuration
    fn config(&self) -> &NeuromorphicConfig;

    /// Run optimization for given objective function
    fn optimize<F>(
        &mut self,
        objective: F,
        initial_params: &ArrayView1<f64>,
    ) -> OptimizeResult<OptimizeResults<f64>>
    where
        F: Fn(&ArrayView1<f64>) -> f64;

    /// Get current network state
    fn network(&self) -> &NeuromorphicNetwork;

    /// Reset optimizer state
    fn reset(&mut self);
}

/// Basic neuromorphic optimizer implementation
#[derive(Debug, Clone)]
pub struct BasicNeuromorphicOptimizer {
    network: NeuromorphicNetwork,
    best_params: Array1<f64>,
    best_objective: f64,
    nit: usize,
}

impl BasicNeuromorphicOptimizer {
    /// Create a new basic neuromorphic optimizer
    pub fn new(config: NeuromorphicConfig, num_parameters: usize) -> Self {
        let network = NeuromorphicNetwork::new(config, num_parameters);

        Self {
            network,
            best_params: Array1::zeros(num_parameters),
            best_objective: f64::INFINITY,
            nit: 0,
        }
    }
}

impl NeuromorphicOptimizer for BasicNeuromorphicOptimizer {
    fn config(&self) -> &NeuromorphicConfig {
        &self.network.config
    }

    fn optimize<F>(
        &mut self,
        objective: F,
        initial_params: &ArrayView1<f64>,
    ) -> OptimizeResult<OptimizeResults<f64>>
    where
        F: Fn(&ArrayView1<f64>) -> f64,
    {
        self.network.parameters = initial_params.to_owned();
        self.best_params = initial_params.to_owned();
        self.best_objective = objective(initial_params);

        let max_nit = (self.network.config.total_time / self.network.config.dt) as usize;

        for iteration in 0..max_nit {
            // Encode current parameters into network
            let params = self.network.parameters.clone();
            self.network.encode_parameters(&params.view());

            // Evaluate objective
            let current_objective = objective(&self.network.parameters.view());

            // Simulate network step
            self.network.simulate_step(current_objective)?;

            // Decode new parameters
            self.network.parameters = self.network.decode_parameters();

            // Update best solution
            if current_objective < self.best_objective {
                self.best_objective = current_objective;
                self.best_params = self.network.parameters.clone();
            }

            self.nit = iteration + 1;

            // Check convergence
            if current_objective < 1e-6 {
                break;
            }
        }

        Ok(OptimizeResults::<f64> {
            x: self.best_params.clone(),
            fun: self.best_objective,
            success: self.best_objective < 1e-3,
            nit: self.nit,
            nfev: self.nit,
            njev: 0,
            nhev: 0,
            maxcv: 0,
            status: 0,
            jac: None,
            hess: None,
            constr: None,
            message: "Neuromorphic optimization completed".to_string(),
        })
    }

    fn network(&self) -> &NeuromorphicNetwork {
        &self.network
    }

    fn reset(&mut self) {
        self.network.current_time = 0.0;
        self.network.spike_queue.clear();
        self.network.objective_history.clear();
        self.best_objective = f64::INFINITY;
        self.nit = 0;

        // Reset neuron states
        for neuron in &mut self.network.neurons {
            neuron.potential = 0.0;
            neuron.last_spike_time = None;
            neuron.input_current = 0.0;
            neuron.adaptation = 0.0;
        }
    }
}

impl BasicNeuromorphicOptimizer {
    /// Get mutable reference to the network
    pub fn network_mut(&mut self) -> &mut NeuromorphicNetwork {
        &mut self.network
    }
}

/// Convenience function to create and run neuromorphic optimization
#[allow(dead_code)]
pub fn neuromorphic_optimize<F>(
    objective: F,
    initial_params: &ArrayView1<f64>,
    config: Option<NeuromorphicConfig>,
) -> OptimizeResult<OptimizeResults<f64>>
where
    F: Fn(&ArrayView1<f64>) -> f64,
{
    let config = config.unwrap_or_default();
    let mut optimizer = BasicNeuromorphicOptimizer::new(config, initial_params.len());
    optimizer.optimize(objective, initial_params)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neuron_state_creation() {
        let neuron = NeuronState::new(10);
        assert_eq!(neuron.weights.len(), 10);
        assert_eq!(neuron.potential, 0.0);
        assert!(neuron.last_spike_time.is_none());
    }

    #[test]
    fn test_neuron_spike_behavior() {
        let mut neuron = NeuronState::new(5);
        neuron.potential = 1.5;

        assert!(neuron.should_spike(1.0));
        neuron.fire_spike(0.1);

        assert_eq!(neuron.potential, 0.0);
        assert_eq!(neuron.last_spike_time, Some(0.1));
    }

    #[test]
    fn test_neuromorphic_network_creation() {
        let config = NeuromorphicConfig::default();
        let network = NeuromorphicNetwork::new(config, 3);

        assert_eq!(network.neurons.len(), 100); // Default num_neurons
        assert_eq!(network.parameters.len(), 3);
        assert_eq!(network.current_time, 0.0);
    }

    #[test]
    fn test_parameter_encoding_decoding() {
        let config = NeuromorphicConfig::default();
        let mut network = NeuromorphicNetwork::new(config, 2);

        let params = Array1::from(vec![0.5, -0.3]);
        network.encode_parameters(&params.view());

        // Check that neurons received input
        assert!(network.neurons.iter().any(|n| n.input_current != 0.0));

        // Decoding should give some result (though not exact due to neural dynamics)
        let decoded = network.decode_parameters();
        assert_eq!(decoded.len(), 2);
    }

    #[test]
    fn test_basic_optimizer() {
        let config = NeuromorphicConfig {
            total_time: 0.1, // Short simulation for test
            num_neurons: 20, // Fewer neurons for speed
            ..Default::default()
        };

        let mut optimizer = BasicNeuromorphicOptimizer::new(config, 2);

        // Simple quadratic objective
        let objective = |x: &ArrayView1<f64>| x[0].powi(2) + x[1].powi(2);
        let initial = Array1::from(vec![1.0, 1.0]);

        let result = optimizer.optimize(objective, &initial.view()).unwrap();

        assert!(result.nit > 0);
        assert!(result.fun < 2.0); // Should improve from initial value of 2.0
    }

    #[test]
    fn test_convenience_function() {
        let config = NeuromorphicConfig {
            total_time: 0.05,
            num_neurons: 10,
            ..Default::default()
        };

        let objective = |x: &ArrayView1<f64>| (x[0] - 1.0).powi(2);
        let initial = Array1::from(vec![0.0]);

        let result = neuromorphic_optimize(objective, &initial.view(), Some(config)).unwrap();

        assert!(result.nit > 0);
        assert!(result.x.len() == 1);
    }

    #[test]
    fn test_spike_queue_processing() {
        let config = NeuromorphicConfig::default();
        let mut network = NeuromorphicNetwork::new(config, 2);

        // Add some spike events
        network.spike_queue.push(SpikeEvent {
            time: 0.001,
            neuron_id: 0,
            weight: 1.0,
        });

        network.current_time = 0.002;
        network.process_spike_events().unwrap();

        // Spike queue should be processed
        assert!(network.spike_queue.is_empty());
    }
}
