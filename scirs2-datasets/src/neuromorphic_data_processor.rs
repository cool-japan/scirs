//! Neuromorphic Data Processing Engine
//!
//! This module provides bio-inspired neuromorphic computing capabilities for
//! advanced dataset processing, featuring spiking neural networks, synaptic
//! plasticity, and brain-inspired learning algorithms.

use crate::error::{DatasetsError, Result};
use crate::utils::Dataset;
use scirs2_core::ndarray::{s, Array1, Array2, Array3};
use scirs2_core::random::prelude::*;
use scirs2_core::random::rand_distributions::Uniform;
use statrs::statistics::Statistics;
use std::time::{Duration, Instant};

/// Neuromorphic data processor using spiking neural networks
#[derive(Debug, Clone)]
pub struct NeuromorphicProcessor {
    /// Network topology configuration
    network_config: NetworkTopology,
    /// Synaptic plasticity parameters
    plasticity_config: SynapticPlasticity,
    /// Spike timing dependent plasticity (STDP) enabled
    stdp_enabled: bool,
    /// Membrane potential decay rate
    membrane_decay: f64,
    /// Spike threshold voltage
    spike_threshold: f64,
    /// Learning rate for synaptic updates
    learning_rate: f64,
}

/// Network topology configuration for neuromorphic processing
#[derive(Debug, Clone)]
pub struct NetworkTopology {
    /// Number of input neurons
    pub input_neurons: usize,
    /// Number of hidden layer neurons
    pub hidden_neurons: usize,
    /// Number of output neurons
    pub output_neurons: usize,
    /// Connection probability between layers
    pub connection_probability: f64,
    /// Enable recurrent connections
    pub recurrent_connections: bool,
}

/// Synaptic plasticity configuration
#[derive(Debug, Clone)]
pub struct SynapticPlasticity {
    /// Hebbian learning strength
    pub hebbian_strength: f64,
    /// Anti-Hebbian learning strength  
    pub anti_hebbian_strength: f64,
    /// Synaptic weight decay rate
    pub weight_decay: f64,
    /// Maximum synaptic weight
    pub max_weight: f64,
    /// Minimum synaptic weight
    pub min_weight: f64,
}

/// Spiking neuron state
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct NeuronState {
    /// Current membrane potential
    membrane_potential: f64,
    /// Last spike time
    last_spike_time: Option<Instant>,
    /// Refractory period remaining
    refractory_time: Duration,
    /// Adaptive threshold
    adaptive_threshold: f64,
}

/// Synaptic connection with STDP
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct Synapse {
    /// Synaptic weight
    weight: f64,
    /// Pre-synaptic neuron index
    pre_neuron: usize,
    /// Post-synaptic neuron index
    post_neuron: usize,
    /// Synaptic delay
    delay: Duration,
    /// Spike trace for STDP
    spike_trace: f64,
}

/// Neuromorphic dataset transformation results
#[derive(Debug, Clone)]
pub struct NeuromorphicTransform {
    /// Transformed feature patterns
    pub spike_patterns: Array3<f64>, // (time, neurons, features)
    /// Synaptic connectivity matrix
    pub connectivity_matrix: Array2<f64>,
    /// Learning trajectory over time
    pub learning_trajectory: Vec<f64>,
    /// Emergent feature representations
    pub emergent_features: Array2<f64>,
}

impl Default for NetworkTopology {
    fn default() -> Self {
        Self {
            input_neurons: 100,
            hidden_neurons: 256,
            output_neurons: 10,
            connection_probability: 0.15,
            recurrent_connections: true,
        }
    }
}

impl Default for SynapticPlasticity {
    fn default() -> Self {
        Self {
            hebbian_strength: 0.01,
            anti_hebbian_strength: 0.005,
            weight_decay: 0.001,
            max_weight: 1.0,
            min_weight: -1.0,
        }
    }
}

impl Default for NeuromorphicProcessor {
    fn default() -> Self {
        Self {
            network_config: NetworkTopology::default(),
            plasticity_config: SynapticPlasticity::default(),
            stdp_enabled: true,
            membrane_decay: 0.95,
            spike_threshold: 1.0,
            learning_rate: 0.001,
        }
    }
}

impl NeuromorphicProcessor {
    /// Create a new neuromorphic processor
    pub fn new(network_config: NetworkTopology, plasticity_config: SynapticPlasticity) -> Self {
        Self {
            network_config,
            plasticity_config,
            stdp_enabled: true,
            membrane_decay: 0.95,
            spike_threshold: 1.0,
            learning_rate: 0.001,
        }
    }

    /// Configure spike timing dependent plasticity
    pub fn with_stdp(mut self, enabled: bool) -> Self {
        self.stdp_enabled = enabled;
        self
    }

    /// Set membrane dynamics parameters
    pub fn with_membrane_dynamics(mut self, decay: f64, threshold: f64) -> Self {
        self.membrane_decay = decay;
        self.spike_threshold = threshold;
        self
    }

    /// Transform dataset using neuromorphic processing
    pub fn transform_dataset(
        &self,
        dataset: &Dataset,
        simulation_time: Duration,
        random_seed: Option<u64>,
    ) -> Result<NeuromorphicTransform> {
        let data = &dataset.data;
        let n_samples = data.nrows();
        let n_features = data.ncols();

        if n_samples == 0 || n_features == 0 {
            return Err(DatasetsError::InvalidFormat(
                "Dataset must have samples and features".to_string(),
            ));
        }

        let mut rng = match random_seed {
            Some(_seed) => StdRng::seed_from_u64(_seed),
            None => StdRng::from_rng(&mut thread_rng()),
        };

        // Initialize neuromorphic network
        let mut network = self.initialize_network(&mut rng)?;

        // Process each sample through the spiking network
        let time_steps = (simulation_time.as_millis() as usize) / 10; // 10ms resolution
        let mut spike_patterns =
            Array3::zeros((time_steps, self.network_config.hidden_neurons, n_samples));
        let mut learning_trajectory = Vec::with_capacity(n_samples);

        for sample_idx in 0..n_samples {
            let sample = data.row(sample_idx);
            let (sample_spikes, learning_score) =
                self.process_sample_neuromorphic(&sample, &mut network, time_steps, &mut rng)?;

            // Store spike patterns for this sample
            for time_idx in 0..time_steps {
                for neuron_idx in 0..self.network_config.hidden_neurons {
                    spike_patterns[[time_idx, neuron_idx, sample_idx]] =
                        sample_spikes[[time_idx, neuron_idx]];
                }
            }

            learning_trajectory.push(learning_score);
        }

        // Extract connectivity matrix
        let connectivity_matrix = self.extract_connectivity_matrix(&network)?;

        // Generate emergent feature representations
        let emergent_features = self.extract_emergent_features(&spike_patterns)?;

        Ok(NeuromorphicTransform {
            spike_patterns,
            connectivity_matrix,
            learning_trajectory,
            emergent_features,
        })
    }

    /// Generate neuromorphic-enhanced dataset using bio-inspired processes
    pub fn generate_bioinspired_dataset(
        &self,
        n_samples: usize,
        n_features: usize,
        adaptation_cycles: usize,
        random_seed: Option<u64>,
    ) -> Result<Dataset> {
        let mut rng = match random_seed {
            Some(_seed) => StdRng::seed_from_u64(_seed),
            None => StdRng::from_rng(&mut thread_rng()),
        };

        // Initialize adaptive neural network
        let mut network = self.initialize_network(&mut rng)?;

        let mut data = Array2::zeros((n_samples, n_features));
        let mut targets = Array1::zeros(n_samples);

        // Generate _samples through neuromorphic adaptation
        for sample_idx in 0..n_samples {
            // Neural network driven feature generation
            let neural_features = self.generate_neural_features(n_features, &network, &mut rng)?;

            // Bio-inspired target assignment using competitive learning
            let target = self.competitive_learning_assignment(&neural_features, &mut rng)?;

            // Store generated sample
            for feature_idx in 0..n_features {
                data[[sample_idx, feature_idx]] = neural_features[feature_idx];
            }
            targets[sample_idx] = target;

            // Adapt network based on generated sample (Hebbian plasticity)
            if sample_idx % adaptation_cycles == 0 {
                self.adapt_network_hebbian(&mut network, &neural_features)?;
            }
        }

        Ok(Dataset::new(data, Some(targets)))
    }

    /// Process temporal sequences using spike timing
    pub fn process_temporal_sequence(
        &self,
        sequence_data: &Array3<f64>, // (time, samples, features)
        stdp_learning: bool,
        random_seed: Option<u64>,
    ) -> Result<NeuromorphicTransform> {
        let (time_steps, n_samples, n_features) = sequence_data.dim();

        if time_steps == 0 || n_samples == 0 || n_features == 0 {
            return Err(DatasetsError::InvalidFormat(
                "Sequence _data must have time, samples, and features".to_string(),
            ));
        }

        let mut rng = match random_seed {
            Some(_seed) => StdRng::seed_from_u64(_seed),
            None => StdRng::from_rng(&mut thread_rng()),
        };

        let mut network = self.initialize_network(&mut rng)?;
        let mut spike_patterns =
            Array3::zeros((time_steps, self.network_config.hidden_neurons, n_samples));
        let mut learning_trajectory = Vec::with_capacity(time_steps);

        // Process temporal sequence with spike timing dependent plasticity
        for time_idx in 0..time_steps {
            let mut time_step_learning = 0.0;

            for sample_idx in 0..n_samples {
                let current_input = sequence_data.slice(s![time_idx, sample_idx, ..]);
                let current_input_array = current_input.to_owned();

                // Convert to spike trains and process
                let spike_response = self.temporal_spike_processing(
                    &current_input_array,
                    &mut network,
                    time_idx,
                    &mut rng,
                )?;

                // Store spike responses
                for neuron_idx in 0..self.network_config.hidden_neurons {
                    spike_patterns[[time_idx, neuron_idx, sample_idx]] = spike_response[neuron_idx];
                }

                // Apply STDP _learning if enabled
                if stdp_learning && self.stdp_enabled {
                    let learning_change = self.apply_stdp_learning(&mut network, time_idx)?;
                    time_step_learning += learning_change;
                }
            }

            learning_trajectory.push(time_step_learning / n_samples as f64);
        }

        let connectivity_matrix = self.extract_connectivity_matrix(&network)?;
        let emergent_features = self.extract_emergent_features(&spike_patterns)?;

        Ok(NeuromorphicTransform {
            spike_patterns,
            connectivity_matrix,
            learning_trajectory,
            emergent_features,
        })
    }

    // Private helper methods for neuromorphic processing

    #[allow(clippy::needless_range_loop)]
    fn initialize_network(&self, rng: &mut StdRng) -> Result<Vec<Vec<Synapse>>> {
        let total_neurons = self.network_config.input_neurons
            + self.network_config.hidden_neurons
            + self.network_config.output_neurons;

        let mut network = vec![Vec::new(); total_neurons];

        // Create synaptic connections based on topology
        for pre_idx in 0..self.network_config.input_neurons {
            for post_idx in self.network_config.input_neurons
                ..(self.network_config.input_neurons + self.network_config.hidden_neurons)
            {
                if rng.random::<f64>() < self.network_config.connection_probability {
                    let weight =
                        (rng.random::<f64>() - 0.5) * 2.0 * self.plasticity_config.max_weight;
                    let delay = Duration::from_millis(rng.sample(Uniform::new(1, 5).unwrap()));

                    network[pre_idx].push(Synapse {
                        weight,
                        pre_neuron: pre_idx,
                        post_neuron: post_idx,
                        delay,
                        spike_trace: 0.0,
                    });
                }
            }
        }

        // Add recurrent connections if enabled
        if self.network_config.recurrent_connections {
            self.add_recurrent_connections(&mut network, rng)?;
        }

        Ok(network)
    }

    fn process_sample_neuromorphic(
        &self,
        sample: &scirs2_core::ndarray::ArrayView1<f64>,
        network: &mut [Vec<Synapse>],
        time_steps: usize,
        _rng: &mut StdRng,
    ) -> Result<(Array2<f64>, f64)> {
        let n_neurons = self.network_config.hidden_neurons;
        let mut spike_pattern = Array2::zeros((time_steps, n_neurons));
        let mut neuron_states = vec![
            NeuronState {
                membrane_potential: 0.0,
                last_spike_time: None,
                refractory_time: Duration::ZERO,
                adaptive_threshold: self.spike_threshold,
            };
            n_neurons
        ];

        let mut learning_score = 0.0;

        // Simulate network dynamics over time
        for time_idx in 0..time_steps {
            // Apply input stimulus
            self.apply_input_stimulus(sample, &mut neuron_states, time_idx)?;

            // Update neuron dynamics
            for neuron_idx in 0..n_neurons {
                // Membrane potential decay
                neuron_states[neuron_idx].membrane_potential *= self.membrane_decay;

                // Check for spike generation
                if neuron_states[neuron_idx].membrane_potential
                    > neuron_states[neuron_idx].adaptive_threshold
                {
                    spike_pattern[[time_idx, neuron_idx]] = 1.0;
                    neuron_states[neuron_idx].membrane_potential = 0.0; // Reset
                    neuron_states[neuron_idx].last_spike_time = Some(Instant::now());

                    // Adaptive threshold increase
                    neuron_states[neuron_idx].adaptive_threshold *= 1.05;

                    learning_score += 0.1; // Reward spiking activity
                }

                // Threshold decay
                neuron_states[neuron_idx].adaptive_threshold =
                    (neuron_states[neuron_idx].adaptive_threshold * 0.99).max(self.spike_threshold);
            }

            // Synaptic transmission
            self.propagate_spikes(network, &spike_pattern, &mut neuron_states, time_idx)?;
        }

        Ok((spike_pattern, learning_score / time_steps as f64))
    }

    fn apply_input_stimulus(
        &self,
        sample: &scirs2_core::ndarray::ArrayView1<f64>,
        neuron_states: &mut [NeuronState],
        _time_idx: usize,
    ) -> Result<()> {
        // Convert input features to spike trains using rate encoding
        for (feature_idx, &feature_value) in sample.iter().enumerate() {
            if feature_idx < self.network_config.input_neurons {
                // Rate encoding: higher values = higher spike probability
                let spike_probability = (feature_value.abs().tanh() + 1.0) / 2.0;
                let spike_current = if thread_rng().random::<f64>() < spike_probability {
                    0.5 * feature_value.signum()
                } else {
                    0.0
                };

                // Apply to corresponding hidden neurons
                if feature_idx < neuron_states.len() {
                    neuron_states[feature_idx].membrane_potential += spike_current;
                }
            }
        }

        Ok(())
    }

    fn propagate_spikes(
        &self,
        network: &mut [Vec<Synapse>],
        spike_pattern: &Array2<f64>,
        neuron_states: &mut [NeuronState],
        time_idx: usize,
    ) -> Result<()> {
        // Propagate spikes through synaptic connections
        for (pre_neuron_idx, synapses) in network.iter().enumerate() {
            if pre_neuron_idx < spike_pattern.ncols() {
                let spike_strength = spike_pattern[[time_idx, pre_neuron_idx]];

                if spike_strength > 0.0 {
                    for synapse in synapses {
                        let post_neuron_idx = synapse.post_neuron;
                        if post_neuron_idx < neuron_states.len() {
                            // Apply synaptic current with delay consideration
                            let synaptic_current = spike_strength * synapse.weight;
                            neuron_states[post_neuron_idx].membrane_potential += synaptic_current;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    fn extract_connectivity_matrix(&self, network: &[Vec<Synapse>]) -> Result<Array2<f64>> {
        let n_neurons = self.network_config.hidden_neurons;
        let mut connectivity = Array2::zeros((n_neurons, n_neurons));

        for (pre_idx, synapses) in network.iter().enumerate() {
            for synapse in synapses {
                if pre_idx < n_neurons && synapse.post_neuron < n_neurons {
                    connectivity[[pre_idx, synapse.post_neuron]] = synapse.weight;
                }
            }
        }

        Ok(connectivity)
    }

    fn extract_emergent_features(&self, spike_patterns: &Array3<f64>) -> Result<Array2<f64>> {
        let (time_steps, n_neurons, n_samples) = spike_patterns.dim();
        let mut features = Array2::zeros((n_samples, n_neurons));

        // Extract temporal spike statistics as emergent features
        for sample_idx in 0..n_samples {
            for neuron_idx in 0..n_neurons {
                let neuron_spikes = spike_patterns.slice(s![.., neuron_idx, sample_idx]);

                // Compute spike rate and temporal _patterns
                let spike_rate = neuron_spikes.sum() / time_steps as f64;
                let spike_variance = neuron_spikes.variance();

                // Combine metrics for emergent feature
                features[[sample_idx, neuron_idx]] = spike_rate + 0.1 * spike_variance;
            }
        }

        Ok(features)
    }

    fn generate_neural_features(
        &self,
        n_features: usize,
        network: &[Vec<Synapse>],
        rng: &mut StdRng,
    ) -> Result<Array1<f64>> {
        let mut features = Array1::zeros(n_features);

        // Use network weights to influence feature generation
        for feature_idx in 0..n_features {
            let mut feature_value = rng.random::<f64>() - 0.5;

            // Neural network influence
            if feature_idx < network.len() {
                let synaptic_influence: f64 = network[feature_idx]
                    .iter()
                    .map(|synapse| synapse.weight)
                    .sum::<f64>()
                    / network[feature_idx].len().max(1) as f64;

                feature_value += 0.3 * synaptic_influence;
            }

            features[feature_idx] = feature_value.tanh(); // Bounded activation
        }

        Ok(features)
    }

    fn competitive_learning_assignment(
        &self,
        features: &Array1<f64>,
        rng: &mut StdRng,
    ) -> Result<f64> {
        // Winner-take-all competitive learning for target assignment
        let max_feature_idx = features
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        // Add some noise for variability
        let noise = rng.random::<f64>() * 0.1 - 0.05;
        Ok(max_feature_idx as f64 + noise)
    }

    fn adapt_network_hebbian(
        &self,
        network: &mut [Vec<Synapse>],
        features: &Array1<f64>,
    ) -> Result<()> {
        // Apply Hebbian learning: "neurons that fire together, wire together"
        for (pre_idx, synapses) in network.iter_mut().enumerate() {
            if pre_idx < features.len() {
                let pre_activity = features[pre_idx];

                for synapse in synapses {
                    if synapse.post_neuron < features.len() {
                        let post_activity = features[synapse.post_neuron];

                        // Hebbian update: Δw = η * pre * post
                        let hebbian_change =
                            self.plasticity_config.hebbian_strength * pre_activity * post_activity;

                        // Weight decay
                        let decay_change = -self.plasticity_config.weight_decay * synapse.weight;

                        // Update weight with bounds checking
                        synapse.weight += hebbian_change + decay_change;
                        synapse.weight = synapse.weight.clamp(
                            self.plasticity_config.min_weight,
                            self.plasticity_config.max_weight,
                        );
                    }
                }
            }
        }

        Ok(())
    }

    fn temporal_spike_processing(
        &self,
        input: &Array1<f64>,
        _network: &mut [Vec<Synapse>],
        time_idx: usize,
        _rng: &mut StdRng,
    ) -> Result<Array1<f64>> {
        let n_neurons = self.network_config.hidden_neurons;
        let mut spike_response = Array1::zeros(n_neurons);

        // Convert temporal input to spike response
        for (neuron_idx, &input_val) in input.iter().enumerate().take(n_neurons) {
            // Temporal encoding with time dependency
            let temporal_factor = 1.0 - (time_idx as f64 / 100.0).min(0.9);
            let spike_probability = (input_val.abs() * temporal_factor).tanh();

            spike_response[neuron_idx] = if spike_probability > 0.5 { 1.0 } else { 0.0 };
        }

        Ok(spike_response)
    }

    fn apply_stdp_learning(&self, network: &mut [Vec<Synapse>], time_idx: usize) -> Result<f64> {
        let mut total_learning_change = 0.0;

        // Spike Timing Dependent Plasticity (STDP)
        for synapses in network.iter_mut() {
            for synapse in synapses {
                // Simplified STDP: recent activity strengthens connections
                let time_factor = 1.0 / (1.0 + time_idx as f64 * 0.01);
                let stdp_change = self.learning_rate * time_factor * synapse.spike_trace;

                synapse.weight += stdp_change;
                synapse.weight = synapse.weight.clamp(
                    self.plasticity_config.min_weight,
                    self.plasticity_config.max_weight,
                );

                // Decay spike trace
                synapse.spike_trace *= 0.95;

                total_learning_change += stdp_change.abs();
            }
        }

        Ok(total_learning_change)
    }

    #[allow(clippy::needless_range_loop)]
    fn add_recurrent_connections(
        &self,
        network: &mut [Vec<Synapse>],
        rng: &mut StdRng,
    ) -> Result<()> {
        let start_hidden = self.network_config.input_neurons;
        let end_hidden = start_hidden + self.network_config.hidden_neurons;

        // Add recurrent connections within hidden layer
        for pre_idx in start_hidden..end_hidden {
            for post_idx in start_hidden..end_hidden {
                if pre_idx != post_idx
                    && rng.random::<f64>() < self.network_config.connection_probability * 0.5
                {
                    let weight =
                        (rng.random::<f64>() - 0.5) * self.plasticity_config.max_weight * 0.5;
                    let delay = Duration::from_millis(rng.sample(Uniform::new(2, 10).unwrap()));

                    network[pre_idx].push(Synapse {
                        weight,
                        pre_neuron: pre_idx,
                        post_neuron: post_idx,
                        delay,
                        spike_trace: 0.0,
                    });
                }
            }
        }

        Ok(())
    }
}

/// Convenience function to create neuromorphic processor with default settings
#[allow(dead_code)]
pub fn create_neuromorphic_processor() -> NeuromorphicProcessor {
    NeuromorphicProcessor::default()
}

/// Convenience function to create neuromorphic processor with custom topology
#[allow(dead_code)]
pub fn create_neuromorphic_processor_with_topology(
    input_neurons: usize,
    hidden_neurons: usize,
    output_neurons: usize,
) -> NeuromorphicProcessor {
    let topology = NetworkTopology {
        input_neurons,
        hidden_neurons,
        output_neurons,
        connection_probability: 0.15,
        recurrent_connections: true,
    };

    NeuromorphicProcessor::new(topology, SynapticPlasticity::default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;
    use scirs2_core::random::Uniform;

    #[test]
    fn test_neuromorphic_dataset_transformation() {
        let data = Array2::from_shape_vec((10, 4), (0..40).map(|x| x as f64).collect()).unwrap();
        let targets = Array1::from((0..10).map(|x| (x % 2) as f64).collect::<Vec<_>>());
        let dataset = Dataset::new(data, Some(targets));

        let processor = NeuromorphicProcessor::default();
        let transform = processor
            .transform_dataset(&dataset, Duration::from_millis(100), Some(42))
            .unwrap();

        assert_eq!(transform.spike_patterns.dim().0, 10); // 100ms / 10ms = 10 time steps
        assert_eq!(transform.spike_patterns.dim().1, 256); // Default hidden neurons
        assert_eq!(transform.spike_patterns.dim().2, 10); // 10 samples
        assert_eq!(transform.connectivity_matrix.dim(), (256, 256));
        assert_eq!(transform.learning_trajectory.len(), 10);
        assert_eq!(transform.emergent_features.dim(), (10, 256));
    }

    #[test]
    fn test_bioinspired_dataset_generation() {
        let processor = NeuromorphicProcessor::default();
        let dataset = processor
            .generate_bioinspired_dataset(50, 5, 10, Some(42))
            .unwrap();

        assert_eq!(dataset.n_samples(), 50);
        assert_eq!(dataset.n_features(), 5);
        assert!(dataset.has_target());
    }

    #[test]
    fn test_temporal_sequence_processing() {
        let processor = NeuromorphicProcessor::default();
        let sequence = Array3::from_shape_fn((5, 10, 4), |(t, s, f)| {
            (t as f64 + s as f64 + f as f64) * 0.1
        });

        let result = processor
            .process_temporal_sequence(&sequence, true, Some(42))
            .unwrap();

        assert_eq!(result.spike_patterns.dim(), (5, 256, 10)); // time, neurons, samples
        assert_eq!(result.learning_trajectory.len(), 5);
    }

    #[test]
    fn test_network_topology_configuration() {
        let topology = NetworkTopology {
            input_neurons: 50,
            hidden_neurons: 128,
            output_neurons: 5,
            connection_probability: 0.2,
            recurrent_connections: false,
        };

        let plasticity = SynapticPlasticity {
            hebbian_strength: 0.02,
            anti_hebbian_strength: 0.01,
            weight_decay: 0.0005,
            max_weight: 2.0,
            min_weight: -2.0,
        };

        let processor = NeuromorphicProcessor::new(topology.clone(), plasticity.clone());
        assert_eq!(processor.network_config.input_neurons, 50);
        assert_eq!(processor.network_config.hidden_neurons, 128);
        assert_eq!(processor.plasticity_config.hebbian_strength, 0.02);
    }

    #[test]
    fn test_stdp_configuration() {
        let processor = NeuromorphicProcessor::default()
            .with_stdp(false)
            .with_membrane_dynamics(0.9, 1.5);

        assert!(!processor.stdp_enabled);
        assert_eq!(processor.membrane_decay, 0.9);
        assert_eq!(processor.spike_threshold, 1.5);
    }
}
