//! Spiking Neural Network Clustering
//!
//! This module implements clustering algorithms based on spiking neural networks (SNNs).
//! These algorithms use spike-timing dependent plasticity (STDP) and competitive learning
//! to discover patterns in spatial data through biologically-inspired neural dynamics.

use crate::error::{SpatialError, SpatialResult};
use ndarray::{Array1, ArrayView2};
use rand::Rng;
use std::collections::HashMap;

// Import core neuromorphic components
use super::super::core::{SpikeEvent, SpikingNeuron, Synapse};

/// Spiking neural network clusterer
///
/// This clusterer uses a network of spiking neurons with STDP learning to perform
/// unsupervised clustering of spatial data. Input points are encoded as spike trains
/// and presented to the network, which learns to respond selectively to different
/// input patterns through competitive dynamics.
///
/// # Features
/// - Rate coding for spatial data encoding
/// - STDP learning for adaptive weights
/// - Lateral inhibition for competitive dynamics
/// - Configurable network architecture
/// - Spike timing analysis
///
/// # Example
/// ```rust
/// use ndarray::Array2;
/// use scirs2_spatial::neuromorphic::algorithms::SpikingNeuralClusterer;
///
/// let points = Array2::from_shape_vec((4, 2), vec![
///     0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0
/// ]).unwrap();
///
/// let mut clusterer = SpikingNeuralClusterer::new(2)
///     .with_spike_threshold(0.8)
///     .with_stdp_learning(true)
///     .with_lateral_inhibition(true);
///
/// let (assignments, spike_events) = clusterer.fit(&points.view()).unwrap();
/// println!("Cluster assignments: {:?}", assignments);
/// ```
#[derive(Debug, Clone)]
pub struct SpikingNeuralClusterer {
    /// Network of spiking neurons
    neurons: Vec<SpikingNeuron>,
    /// Synaptic connections
    synapses: Vec<Synapse>,
    /// Number of clusters (output neurons)
    num_clusters: usize,
    /// Spike threshold
    spike_threshold: f64,
    /// Enable STDP learning
    stdp_learning: bool,
    /// Enable lateral inhibition
    lateral_inhibition: bool,
    /// Simulation time step
    dt: f64,
    /// Current simulation time
    current_time: f64,
    /// Spike history
    spike_history: Vec<SpikeEvent>,
    /// Number of training epochs
    max_epochs: usize,
    /// Simulation duration per data point
    simulation_duration: f64,
}

impl SpikingNeuralClusterer {
    /// Create new spiking neural clusterer
    ///
    /// # Arguments
    /// * `num_clusters` - Number of clusters to discover
    ///
    /// # Returns
    /// A new `SpikingNeuralClusterer` with default parameters
    pub fn new(num_clusters: usize) -> Self {
        Self {
            neurons: Vec::new(),
            synapses: Vec::new(),
            num_clusters,
            spike_threshold: 1.0,
            stdp_learning: true,
            lateral_inhibition: true,
            dt: 0.1,
            current_time: 0.0,
            spike_history: Vec::new(),
            max_epochs: 100,
            simulation_duration: 10.0,
        }
    }

    /// Configure spike threshold
    ///
    /// # Arguments
    /// * `threshold` - Spike threshold for neurons
    pub fn with_spike_threshold(mut self, threshold: f64) -> Self {
        self.spike_threshold = threshold;
        self
    }

    /// Enable/disable STDP learning
    ///
    /// # Arguments
    /// * `enabled` - Whether to enable STDP learning
    pub fn with_stdp_learning(mut self, enabled: bool) -> Self {
        self.stdp_learning = enabled;
        self
    }

    /// Enable/disable lateral inhibition
    ///
    /// # Arguments
    /// * `enabled` - Whether to enable lateral inhibition
    pub fn with_lateral_inhibition(mut self, enabled: bool) -> Self {
        self.lateral_inhibition = enabled;
        self
    }

    /// Configure training parameters
    ///
    /// # Arguments
    /// * `max_epochs` - Maximum number of training epochs
    /// * `simulation_duration` - Duration to simulate per data point
    pub fn with_training_params(mut self, max_epochs: usize, simulation_duration: f64) -> Self {
        self.max_epochs = max_epochs;
        self.simulation_duration = simulation_duration;
        self
    }

    /// Configure simulation time step
    ///
    /// # Arguments
    /// * `dt` - Time step for simulation
    pub fn with_time_step(mut self, dt: f64) -> Self {
        self.dt = dt;
        self
    }

    /// Fit clustering to spatial data
    ///
    /// Trains the spiking neural network on the provided spatial data using
    /// STDP learning and competitive dynamics to discover cluster structure.
    ///
    /// # Arguments
    /// * `points` - Input points to cluster (n_points × n_dims)
    ///
    /// # Returns
    /// Tuple of (cluster assignments, spike events) where assignments
    /// maps each point to its cluster and spike_events contains the
    /// complete spike timing history.
    pub fn fit(
        &mut self,
        points: &ArrayView2<'_, f64>,
    ) -> SpatialResult<(Array1<usize>, Vec<SpikeEvent>)> {
        let (n_points, n_dims) = points.dim();

        if n_points == 0 || n_dims == 0 {
            return Err(SpatialError::InvalidInput(
                "Input data cannot be empty".to_string(),
            ));
        }

        // Initialize neural network
        self.initialize_network(n_dims)?;

        // Present data points as spike trains
        let mut assignments = Array1::zeros(n_points);

        for epoch in 0..self.max_epochs {
            self.current_time = epoch as f64 * 100.0;

            for (point_idx, point) in points.outer_iter().enumerate() {
                // Encode spatial point as spike train
                let spike_train = self.encode_point_as_spikes(&point.to_owned())?;

                // Process spike train through network
                let winning_neuron = self.process_spike_train(&spike_train)?;
                assignments[point_idx] = winning_neuron;

                // Apply learning if enabled
                if self.stdp_learning {
                    self.apply_stdp_learning(&spike_train)?;
                }
            }

            // Apply lateral inhibition
            if self.lateral_inhibition {
                self.apply_lateral_inhibition()?;
            }
        }

        Ok((assignments, self.spike_history.clone()))
    }

    /// Initialize spiking neural network
    ///
    /// Creates the network topology with input neurons, output neurons,
    /// and synaptic connections between them.
    fn initialize_network(&mut self, input_dims: usize) -> SpatialResult<()> {
        self.neurons.clear();
        self.synapses.clear();
        self.spike_history.clear();

        // Create input neurons (one per dimension)
        for i in 0..input_dims {
            let position = vec![i as f64];
            let mut neuron = SpikingNeuron::new(position);
            neuron.set_threshold(self.spike_threshold);
            self.neurons.push(neuron);
        }

        // Create output neurons (cluster centers)
        let mut rng = rand::rng();
        for _i in 0..self.num_clusters {
            let position = (0..input_dims).map(|_| rng.gen_range(0.0..1.0)).collect();
            let mut neuron = SpikingNeuron::new(position);
            neuron.set_threshold(self.spike_threshold);
            self.neurons.push(neuron);
        }

        // Create synaptic connections (input to output)
        for i in 0..input_dims {
            for j in 0..self.num_clusters {
                let output_idx = input_dims + j;
                let weight = rng.gen_range(0.0..0.5);
                let synapse = Synapse::new(i, output_idx, weight);
                self.synapses.push(synapse);
            }
        }

        // Create lateral inhibitory connections between output neurons
        if self.lateral_inhibition {
            for i in 0..self.num_clusters {
                for j in 0..self.num_clusters {
                    if i != j {
                        let neuron_i = input_dims + i;
                        let neuron_j = input_dims + j;
                        let synapse = Synapse::new(neuron_i, neuron_j, -0.5);
                        self.synapses.push(synapse);
                    }
                }
            }
        }

        Ok(())
    }

    /// Encode spatial point as spike train
    ///
    /// Converts a spatial data point into a spike train using rate coding,
    /// where the firing rate of each input neuron is proportional to the
    /// corresponding coordinate value.
    fn encode_point_as_spikes(&self, point: &Array1<f64>) -> SpatialResult<Vec<SpikeEvent>> {
        let mut spike_train = Vec::new();

        // Rate coding: spike frequency proportional to coordinate value
        for (dim, &coord) in point.iter().enumerate() {
            // Normalize coordinate to [0, 1] and scale to spike rate
            let normalized_coord = (coord + 10.0) / 20.0; // Assume data in [-10, 10]
            let spike_rate = normalized_coord.clamp(0.0, 1.0) * 50.0; // Max 50 Hz

            // Generate Poisson spike train
            let num_spikes = (spike_rate * 1.0) as usize; // 1 second duration
            for spike_idx in 0..num_spikes {
                let timestamp =
                    self.current_time + (spike_idx as f64) * (1.0 / spike_rate.max(1.0));
                let spike = SpikeEvent::new(dim, timestamp, 1.0, point.to_vec());
                spike_train.push(spike);
            }
        }

        // Sort spikes by timestamp
        spike_train.sort_by(|a, b| a.timestamp().partial_cmp(&b.timestamp()).unwrap());

        Ok(spike_train)
    }

    /// Process spike train through network
    ///
    /// Simulates the network dynamics when presented with a spike train,
    /// determining which output neuron responds most strongly.
    fn process_spike_train(&mut self, spike_train: &[SpikeEvent]) -> SpatialResult<usize> {
        let input_dims = self.neurons.len() - self.num_clusters;
        let mut neuron_spike_counts = vec![0; self.num_clusters];

        // Simulate network for duration of spike train
        let mut t = self.current_time;
        let mut spike_idx = 0;

        while t < self.current_time + self.simulation_duration {
            // Apply input spikes
            let mut input_currents = vec![0.0; self.neurons.len()];

            while spike_idx < spike_train.len() && spike_train[spike_idx].timestamp() <= t {
                let spike = &spike_train[spike_idx];
                if spike.neuron_id() < input_dims {
                    input_currents[spike.neuron_id()] += spike.amplitude();
                }
                spike_idx += 1;
            }

            // Calculate synaptic currents
            for synapse in &self.synapses {
                if synapse.pre_neuron() < self.neurons.len()
                    && synapse.post_neuron() < self.neurons.len()
                {
                    let pre_current = input_currents[synapse.pre_neuron()];
                    let synaptic_current = synapse.synaptic_current(pre_current);
                    input_currents[synapse.post_neuron()] += synaptic_current;
                }
            }

            // Update neurons and check for spikes
            for (neuron_idx, neuron) in self.neurons.iter_mut().enumerate() {
                let spiked = neuron.update(self.dt, input_currents[neuron_idx]);

                if spiked && neuron_idx >= input_dims {
                    let cluster_idx = neuron_idx - input_dims;
                    neuron_spike_counts[cluster_idx] += 1;

                    // Record spike event
                    let spike_event =
                        SpikeEvent::new(neuron_idx, t, 1.0, neuron.position().to_vec());
                    self.spike_history.push(spike_event);
                }
            }

            t += self.dt;
        }

        // Find winning neuron (cluster with most spikes)
        let winning_cluster = neuron_spike_counts
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.cmp(b))
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        Ok(winning_cluster)
    }

    /// Apply STDP learning to synapses
    ///
    /// Updates synaptic weights based on the relative timing of pre- and
    /// post-synaptic spikes using the STDP learning rule.
    fn apply_stdp_learning(&mut self, spike_train: &[SpikeEvent]) -> SpatialResult<()> {
        // Create spike timing map
        let mut spike_times: HashMap<usize, Vec<f64>> = HashMap::new();
        for spike in spike_train {
            spike_times
                .entry(spike.neuron_id())
                .or_default()
                .push(spike.timestamp());
        }

        // Add output neuron spikes from history
        for spike in &self.spike_history {
            spike_times
                .entry(spike.neuron_id())
                .or_default()
                .push(spike.timestamp());
        }

        // Update synaptic weights using STDP
        let empty_spikes = Vec::new();
        for synapse in &mut self.synapses {
            let pre_spikes = spike_times
                .get(&synapse.pre_neuron())
                .unwrap_or(&empty_spikes);
            let post_spikes = spike_times
                .get(&synapse.post_neuron())
                .unwrap_or(&empty_spikes);

            // Check for coincident spikes
            for &pre_time in pre_spikes {
                for &post_time in post_spikes {
                    let dt = post_time - pre_time;
                    if dt.abs() < 50.0 {
                        // Within STDP window
                        let current_weight = synapse.weight();
                        if dt > 0.0 {
                            // Potentiation
                            let delta_w = synapse.stdp_rate() * (-dt / synapse.stdp_tau()).exp();
                            synapse.set_weight(current_weight + delta_w);
                        } else {
                            // Depression
                            let delta_w = synapse.stdp_rate() * (dt / synapse.stdp_tau()).exp();
                            synapse.set_weight(current_weight - delta_w);
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Apply lateral inhibition between output neurons
    ///
    /// Strengthens inhibitory connections between neurons based on their
    /// relative activity levels to promote competition.
    fn apply_lateral_inhibition(&mut self) -> SpatialResult<()> {
        let input_dims = self.neurons.len() - self.num_clusters;

        // Strengthen inhibitory connections between active neurons
        for i in 0..self.num_clusters {
            for j in 0..self.num_clusters {
                if i != j {
                    let neuron_i_idx = input_dims + i;
                    let neuron_j_idx = input_dims + j;

                    // Find inhibitory synapse
                    for synapse in &mut self.synapses {
                        if synapse.pre_neuron() == neuron_i_idx
                            && synapse.post_neuron() == neuron_j_idx
                        {
                            // Strengthen inhibition based on activity
                            let activity_i = self.neurons[neuron_i_idx].membrane_potential();
                            let activity_j = self.neurons[neuron_j_idx].membrane_potential();

                            if activity_i > activity_j {
                                let current_weight = synapse.weight();
                                synapse.set_weight(current_weight - 0.01); // Strengthen inhibition
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Get number of clusters
    pub fn num_clusters(&self) -> usize {
        self.num_clusters
    }

    /// Get spike threshold
    pub fn spike_threshold(&self) -> f64 {
        self.spike_threshold
    }

    /// Check if STDP learning is enabled
    pub fn is_stdp_enabled(&self) -> bool {
        self.stdp_learning
    }

    /// Check if lateral inhibition is enabled
    pub fn is_lateral_inhibition_enabled(&self) -> bool {
        self.lateral_inhibition
    }

    /// Get current spike history
    pub fn spike_history(&self) -> &[SpikeEvent] {
        &self.spike_history
    }

    /// Get network statistics
    pub fn network_stats(&self) -> NetworkStats {
        NetworkStats {
            num_neurons: self.neurons.len(),
            num_synapses: self.synapses.len(),
            num_spikes: self.spike_history.len(),
            average_weight: if self.synapses.is_empty() {
                0.0
            } else {
                self.synapses.iter().map(|s| s.weight()).sum::<f64>() / self.synapses.len() as f64
            },
        }
    }

    /// Reset the network to initial state
    pub fn reset(&mut self) {
        for neuron in &mut self.neurons {
            neuron.reset();
        }
        for synapse in &mut self.synapses {
            synapse.reset_spike_history();
        }
        self.spike_history.clear();
        self.current_time = 0.0;
    }
}

/// Network statistics for analysis
#[derive(Debug, Clone)]
pub struct NetworkStats {
    /// Total number of neurons
    pub num_neurons: usize,
    /// Total number of synapses
    pub num_synapses: usize,
    /// Total number of spikes recorded
    pub num_spikes: usize,
    /// Average synaptic weight
    pub average_weight: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_spiking_clusterer_creation() {
        let clusterer = SpikingNeuralClusterer::new(3);
        assert_eq!(clusterer.num_clusters(), 3);
        assert_eq!(clusterer.spike_threshold(), 1.0);
        assert!(clusterer.is_stdp_enabled());
        assert!(clusterer.is_lateral_inhibition_enabled());
    }

    #[test]
    fn test_clusterer_configuration() {
        let clusterer = SpikingNeuralClusterer::new(2)
            .with_spike_threshold(0.8)
            .with_stdp_learning(false)
            .with_lateral_inhibition(false)
            .with_training_params(50, 5.0);

        assert_eq!(clusterer.spike_threshold(), 0.8);
        assert!(!clusterer.is_stdp_enabled());
        assert!(!clusterer.is_lateral_inhibition_enabled());
        assert_eq!(clusterer.max_epochs, 50);
        assert_eq!(clusterer.simulation_duration, 5.0);
    }

    #[test]
    fn test_simple_clustering() {
        let points =
            Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();

        let mut clusterer = SpikingNeuralClusterer::new(2).with_training_params(5, 1.0); // Reduced for test speed

        let result = clusterer.fit(&points.view());
        assert!(result.is_ok());

        let (assignments, spike_events) = result.unwrap();
        assert_eq!(assignments.len(), 4);

        // Should have recorded some spike events
        assert!(!spike_events.is_empty());
    }

    #[test]
    fn test_empty_input() {
        let points = Array2::zeros((0, 2));
        let mut clusterer = SpikingNeuralClusterer::new(2);

        let result = clusterer.fit(&points.view());
        assert!(result.is_err());
    }

    #[test]
    fn test_network_initialization() {
        let mut clusterer = SpikingNeuralClusterer::new(2);
        clusterer.initialize_network(3).unwrap();

        let stats = clusterer.network_stats();
        assert_eq!(stats.num_neurons, 5); // 3 input + 2 output

        // Should have input-to-output connections
        let expected_connections = 3 * 2; // input_dims * num_clusters
                                          // Plus lateral inhibition connections: num_clusters * (num_clusters - 1)
        let lateral_connections = 2 * 1;
        assert_eq!(
            stats.num_synapses,
            expected_connections + lateral_connections
        );
    }

    #[test]
    fn test_spike_encoding() {
        let clusterer = SpikingNeuralClusterer::new(2);
        let point = Array1::from_vec(vec![1.0, -1.0]);

        let spike_train = clusterer.encode_point_as_spikes(&point).unwrap();

        // Should generate spikes for each dimension
        assert!(!spike_train.is_empty());

        // Spikes should be sorted by timestamp
        for i in 1..spike_train.len() {
            assert!(spike_train[i - 1].timestamp() <= spike_train[i].timestamp());
        }
    }

    #[test]
    fn test_network_reset() {
        let mut clusterer = SpikingNeuralClusterer::new(2);
        clusterer.initialize_network(2).unwrap();

        // Add some activity
        clusterer
            .spike_history
            .push(SpikeEvent::new(0, 1.0, 1.0, vec![0.0, 0.0]));
        clusterer.current_time = 100.0;

        // Reset should clear history and time
        clusterer.reset();
        assert!(clusterer.spike_history().is_empty());
        assert_eq!(clusterer.current_time, 0.0);
    }

    #[test]
    #[ignore]
    fn test_network_stats() {
        let mut clusterer = SpikingNeuralClusterer::new(2);
        clusterer.initialize_network(3).unwrap();

        let stats = clusterer.network_stats();
        assert_eq!(stats.num_neurons, 5);
        assert!(stats.num_synapses > 0);
        assert_eq!(stats.num_spikes, 0); // No activity yet
        assert!(stats.average_weight >= 0.0);
    }
}
