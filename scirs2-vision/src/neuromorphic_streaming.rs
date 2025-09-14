//! Neuromorphic streaming processing for brain-inspired computer vision
//!
//! This module implements neuromorphic computing principles for advanced-efficient
//! streaming processing, inspired by biological neural networks and spiking neurons.
//!
//! # Features
//!
//! - Spiking neural network processing stages
//! - Event-driven computation for sparse data
//! - Synaptic plasticity for adaptive learning

#![allow(dead_code)]
//! - Neuronal membrane dynamics modeling
//! - Energy-efficient processing inspired by biological neurons

use crate::error::Result;
#[cfg(test)]
use crate::streaming::FrameMetadata;
use crate::streaming::{Frame, ProcessingStage};
use ndarray::{Array1, Array2, ArrayView2};
use rand::Rng;
use scirs2_core::rng;
use statrs::statistics::Statistics;
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

/// Spiking neuron model for neuromorphic processing
#[derive(Debug, Clone)]
pub struct SpikingNeuron {
    /// Membrane potential
    pub membrane_potential: f64,
    /// Resting potential
    pub resting_potential: f64,
    /// Threshold for spiking
    pub spike_threshold: f64,
    /// Reset potential after spike
    pub reset_potential: f64,
    /// Membrane time constant
    pub tau_membrane: f64,
    /// Refractory period
    pub refractory_period: f64,
    /// Time since last spike
    pub time_since_spike: f64,
    /// Input current
    pub input_current: f64,
    /// Spike history
    pub spike_times: VecDeque<f64>,
    /// Neuron activity state
    pub is_refractory: bool,
}

impl SpikingNeuron {
    /// Create a new spiking neuron
    pub fn new() -> Self {
        Self {
            membrane_potential: -70.0, // mV
            resting_potential: -70.0,
            spike_threshold: -50.0,
            reset_potential: -80.0,
            tau_membrane: 10.0, // ms
            refractory_period: 2.0,
            time_since_spike: 0.0,
            input_current: 0.0,
            spike_times: VecDeque::with_capacity(100),
            is_refractory: false,
        }
    }

    /// Update neuron state using leaky integrate-and-fire model
    pub fn update(&mut self, dt: f64, inputcurrent: f64) -> bool {
        self.input_current = inputcurrent;
        self.time_since_spike += dt;

        // Check if in refractory period
        if self.is_refractory {
            if self.time_since_spike >= self.refractory_period {
                self.is_refractory = false;
                self.membrane_potential = self.resting_potential;
            }
            return false;
        }

        // Integrate membrane potential using Euler method
        let leak_current = (self.resting_potential - self.membrane_potential) / self.tau_membrane;
        let dvdt = leak_current + inputcurrent;
        self.membrane_potential += dvdt * dt;

        // Check for spike
        if self.membrane_potential >= self.spike_threshold {
            self.spike();
            return true;
        }

        false
    }

    /// Generate a spike
    fn spike(&mut self) {
        self.membrane_potential = self.reset_potential;
        self.is_refractory = true;
        self.time_since_spike = 0.0;

        // Record spike time
        self.spike_times.push_back(self.get_current_time());

        // Keep spike history bounded
        if self.spike_times.len() > 100 {
            self.spike_times.pop_front();
        }
    }

    /// Get current time (simplified)
    fn get_current_time(&self) -> f64 {
        // In a real implementation, this would return actual time
        rand::random::<f64>() * 1000.0
    }

    /// Calculate spike rate over recent history
    pub fn spike_rate(&self, timewindow: f64) -> f64 {
        let current_time = self.get_current_time();
        let cutoff_time = current_time - timewindow;

        let recent_spikes = self
            .spike_times
            .iter()
            .filter(|&&spike_time| spike_time >= cutoff_time)
            .count();

        recent_spikes as f64 / timewindow
    }
}

impl Default for SpikingNeuron {
    fn default() -> Self {
        Self::new()
    }
}

/// Synaptic connection with plasticity
#[derive(Debug, Clone)]
pub struct PlasticSynapse {
    /// Synaptic weight
    pub weight: f64,
    /// Pre-synaptic neuron ID
    pub pre_neuron_id: usize,
    /// Post-synaptic neuron ID
    pub post_neuron_id: usize,
    /// Time of last pre-synaptic spike
    pub last_pre_spike: Option<f64>,
    /// Time of last post-synaptic spike
    pub last_post_spike: Option<f64>,
    /// STDP parameters
    pub stdp_params: STDPParameters,
    /// Synaptic delay
    pub delay: f64,
}

/// Spike-timing dependent plasticity parameters
#[derive(Debug, Clone)]
pub struct STDPParameters {
    /// Learning rate for potentiation
    pub a_plus: f64,
    /// Learning rate for depression
    pub a_minus: f64,
    /// Time constant for potentiation
    pub tau_plus: f64,
    /// Time constant for depression
    pub tau_minus: f64,
    /// Maximum weight
    pub w_max: f64,
    /// Minimum weight
    pub w_min: f64,
}

impl Default for STDPParameters {
    fn default() -> Self {
        Self {
            a_plus: 0.01,
            a_minus: 0.012,
            tau_plus: 20.0,
            tau_minus: 20.0,
            w_max: 1.0,
            w_min: 0.0,
        }
    }
}

impl PlasticSynapse {
    /// Create a new plastic synapse
    pub fn new(pre_id: usize, post_id: usize, initialweight: f64) -> Self {
        Self {
            weight: initialweight,
            pre_neuron_id: pre_id,
            post_neuron_id: post_id,
            last_pre_spike: None,
            last_post_spike: None,
            stdp_params: STDPParameters::default(),
            delay: 1.0, // ms
        }
    }

    /// Update synaptic weight using STDP
    pub fn update_weight(&mut self, pre_spike_time: Option<f64>, post_spiketime: Option<f64>) {
        // Update spike times
        if let Some(pre_time) = pre_spike_time {
            self.last_pre_spike = Some(pre_time);
        }
        if let Some(post_time) = post_spiketime {
            self.last_post_spike = Some(post_time);
        }

        // Apply STDP if both neurons have spiked
        if let (Some(t_pre), Some(t_post)) = (self.last_pre_spike, self.last_post_spike) {
            let dt = t_post - t_pre - self.delay;

            let weight_change = if dt > 0.0 {
                // Potentiation (post after pre)
                self.stdp_params.a_plus * (-dt / self.stdp_params.tau_plus).exp()
            } else {
                // Depression (pre after post)
                -self.stdp_params.a_minus * (dt / self.stdp_params.tau_minus).exp()
            };

            self.weight += weight_change;
            self.weight = self
                .weight
                .clamp(self.stdp_params.w_min, self.stdp_params.w_max);
        }
    }

    /// Calculate synaptic current
    pub fn calculate_current(&self, prespike: bool) -> f64 {
        if prespike {
            self.weight * 10.0 // Scale factor for current injection
        } else {
            0.0
        }
    }
}

/// Neuromorphic spiking neural network
#[derive(Debug)]
pub struct SpikingNeuralNetwork {
    /// Network neurons
    neurons: Vec<SpikingNeuron>,
    /// Synaptic connections
    synapses: Vec<PlasticSynapse>,
    /// Network topology (adjacency list)
    connectivity: HashMap<usize, Vec<usize>>,
    /// Time step for simulation
    dt: f64,
    /// Current simulation time
    current_time: f64,
    /// Spike events queue
    spike_events: VecDeque<SpikeEvent>,
}

/// Spike event for event-driven processing
#[derive(Debug, Clone)]
pub struct SpikeEvent {
    /// Neuron ID that spiked
    pub neuron_id: usize,
    /// Time of spike
    pub spike_time: f64,
    /// Spike amplitude
    pub amplitude: f64,
}

impl SpikingNeuralNetwork {
    /// Create a new spiking neural network
    pub fn new(num_neurons: usize, connectivityprobability: f64) -> Self {
        let mut neurons = Vec::with_capacity(num_neurons);
        let mut synapses = Vec::new();
        let mut connectivity = HashMap::new();
        let mut rng = rng();

        // Initialize neurons
        for _ in 0..num_neurons {
            neurons.push(SpikingNeuron::new());
        }

        // Create random connectivity
        for i in 0..num_neurons {
            let mut connections = Vec::new();
            for j in 0..num_neurons {
                if i != j && rng.random::<f64>() < connectivityprobability {
                    connections.push(j);

                    // Create synapse
                    let weight = rng.gen_range(0.1..0.8);
                    synapses.push(PlasticSynapse::new(i, j, weight));
                }
            }
            connectivity.insert(i, connections);
        }

        Self {
            neurons,
            synapses,
            connectivity,
            dt: 0.1, // ms
            current_time: 0.0,
            spike_events: VecDeque::with_capacity(1000),
        }
    }

    /// Process input through the spiking network
    pub fn process_input(&mut self, input: &Array1<f64>) -> Array1<f64> {
        let num_neurons = self.neurons.len();
        let input_size = input.len();

        // Clear previous spike events
        self.spike_events.clear();

        // Inject input current to first layer neurons
        for (i, &input_val) in input.iter().enumerate() {
            if i < num_neurons {
                self.neurons[i].input_current = input_val * 50.0; // Scale input
            }
        }

        // Simulate network for one time step
        let mut spikes = vec![false; num_neurons];

        // First, collect neuron spike states to avoid borrow checker issues
        let neuron_spike_states: Vec<bool> = self
            .neurons
            .iter()
            .map(|neuron| neuron.time_since_spike < self.dt)
            .collect();

        for (i, neuron) in self.neurons.iter_mut().enumerate() {
            // Calculate total synaptic input
            let mut synaptic_input = 0.0;

            for synapse in &self.synapses {
                if synapse.post_neuron_id == i {
                    let prespike = neuron_spike_states[synapse.pre_neuron_id];
                    synaptic_input += synapse.calculate_current(prespike);
                }
            }

            // Add external input for input neurons
            if i < input_size {
                synaptic_input += neuron.input_current;
            }

            // Update neuron
            let spiked = neuron.update(self.dt, synaptic_input);
            spikes[i] = spiked;

            if spiked {
                self.spike_events.push_back(SpikeEvent {
                    neuron_id: i,
                    spike_time: self.current_time,
                    amplitude: 1.0,
                });
            }
        }

        // Update synaptic weights using STDP
        for synapse in &mut self.synapses {
            let pre_spike_time = if spikes[synapse.pre_neuron_id] {
                Some(self.current_time)
            } else {
                None
            };

            let post_spiketime = if spikes[synapse.post_neuron_id] {
                Some(self.current_time)
            } else {
                None
            };

            synapse.update_weight(pre_spike_time, post_spiketime);
        }

        self.current_time += self.dt;

        // Return spike rates as output
        let timewindow = 10.0; // ms
        let mut output = Array1::zeros(num_neurons);
        for (i, neuron) in self.neurons.iter().enumerate() {
            output[i] = neuron.spike_rate(timewindow);
        }

        output
    }

    /// Get network activity statistics
    pub fn get_activity_stats(&self) -> NetworkActivityStats {
        let total_spikes = self.spike_events.len();
        let active_neurons = self
            .neurons
            .iter()
            .filter(|neuron| neuron.spike_rate(10.0) > 0.0)
            .count();

        let avg_membrane_potential = self
            .neurons
            .iter()
            .map(|neuron| neuron.membrane_potential)
            .sum::<f64>()
            / self.neurons.len() as f64;

        let avg_weight = self
            .synapses
            .iter()
            .map(|synapse| synapse.weight)
            .sum::<f64>()
            / self.synapses.len() as f64;

        NetworkActivityStats {
            total_spikes,
            active_neurons,
            avg_membrane_potential,
            avg_synaptic_weight: avg_weight,
            network_sparsity: active_neurons as f64 / self.neurons.len() as f64,
        }
    }
}

/// Network activity statistics
#[derive(Debug, Clone)]
pub struct NetworkActivityStats {
    /// Total number of spikes in recent window
    pub total_spikes: usize,
    /// Number of active neurons
    pub active_neurons: usize,
    /// Average membrane potential
    pub avg_membrane_potential: f64,
    /// Average synaptic weight
    pub avg_synaptic_weight: f64,
    /// Network sparsity (fraction of active neurons)
    pub network_sparsity: f64,
}

/// Neuromorphic edge detection stage using spiking neurons
#[derive(Debug)]
pub struct NeuromorphicEdgeDetector {
    /// Spiking neural network for edge detection
    snn: SpikingNeuralNetwork,
    /// Input preprocessing parameters
    preprocessing_params: EdgePreprocessingParams,
    /// Adaptation parameters
    adaptation_rate: f64,
    /// Processing history for adaptation
    processing_history: VecDeque<f64>,
}

/// Parameters for edge detection preprocessing
#[derive(Debug, Clone)]
pub struct EdgePreprocessingParams {
    /// Contrast threshold
    pub contrast_threshold: f64,
    /// Temporal difference threshold
    pub temporal_threshold: f64,
    /// Spatial kernel size
    pub spatial_kernel_size: usize,
    /// Adaptation speed
    pub adaptation_speed: f64,
}

impl Default for EdgePreprocessingParams {
    fn default() -> Self {
        Self {
            contrast_threshold: 0.1,
            temporal_threshold: 0.05,
            spatial_kernel_size: 3,
            adaptation_speed: 0.01,
        }
    }
}

impl NeuromorphicEdgeDetector {
    /// Create a new neuromorphic edge detector
    pub fn new(_inputsize: usize) -> Self {
        let network_size = _inputsize * 2; // Hidden layer for processing
        let snn = SpikingNeuralNetwork::new(network_size, 0.3);

        Self {
            snn,
            preprocessing_params: EdgePreprocessingParams::default(),
            adaptation_rate: 0.001,
            processing_history: VecDeque::with_capacity(100),
        }
    }

    /// Convert image patch to spike train
    fn image_to_spikes(&self, imagepatch: &ArrayView2<f32>) -> Array1<f64> {
        let (height, width) = imagepatch.dim();
        let mut spike_input = Array1::zeros(height * width);

        // Convert pixel intensities to spike rates
        for (i, &pixel) in imagepatch.iter().enumerate() {
            // Higher intensity = higher spike rate
            let spike_rate = (pixel as f64 * 100.0).max(0.0); // Scale to reasonable spike rate
            spike_input[i] = spike_rate;
        }

        spike_input
    }

    /// Apply neuromorphic edge detection
    fn detect_edges_neuromorphic(&mut self, frame: &Frame) -> Result<Array2<f32>> {
        let (height, width) = frame.data.dim();
        let mut edge_map = Array2::zeros((height, width));

        let kernel_size = self.preprocessing_params.spatial_kernel_size;
        let half_kernel = kernel_size / 2;

        // Process image in overlapping patches
        for y in half_kernel..height.saturating_sub(half_kernel) {
            for x in half_kernel..width.saturating_sub(half_kernel) {
                // Extract local patch
                let patch = frame.data.slice(ndarray::s![
                    y.saturating_sub(half_kernel)..=(y + half_kernel).min(height - 1),
                    x.saturating_sub(half_kernel)..=(x + half_kernel).min(width - 1)
                ]);

                // Convert to spike input
                let spike_input = self.image_to_spikes(&patch);

                // Process through spiking network
                let network_output = self.snn.process_input(&spike_input);

                // Extract edge strength from network activity
                let edge_strength = network_output.mean() as f32;
                edge_map[[y, x]] = edge_strength;
            }
        }

        // Normalize edge map
        let max_edge = edge_map.iter().fold(0.0f32, |a, &b| a.max(b));
        if max_edge > 0.0 {
            edge_map.mapv_inplace(|x| x / max_edge);
        }

        Ok(edge_map)
    }

    /// Adapt preprocessing parameters based on performance
    fn adapt_parameters(&mut self, performancemetric: f64) {
        self.processing_history.push_back(performancemetric);

        if self.processing_history.len() > 10 {
            self.processing_history.pop_front();
        }

        // Calculate performance trend
        if self.processing_history.len() >= 2 {
            let recent_avg = self.processing_history.iter().rev().take(5).sum::<f64>()
                / 5.0_f64.min(self.processing_history.len() as f64);

            let older_avg = self.processing_history.iter().take(5).sum::<f64>()
                / 5.0_f64.min(self.processing_history.len() as f64);

            let trend = recent_avg - older_avg;

            // Adapt thresholds based on trend
            if trend < 0.0 {
                // Performance declining, adjust parameters
                self.preprocessing_params.contrast_threshold *=
                    1.0 - self.preprocessing_params.adaptation_speed;
                self.preprocessing_params.temporal_threshold *=
                    1.0 + self.preprocessing_params.adaptation_speed;
            } else if trend > 0.0 {
                // Performance improving, continue current direction
                self.preprocessing_params.contrast_threshold *=
                    1.0 + self.preprocessing_params.adaptation_speed * 0.5;
                self.preprocessing_params.temporal_threshold *=
                    1.0 - self.preprocessing_params.adaptation_speed * 0.5;
            }

            // Keep parameters in valid ranges
            self.preprocessing_params.contrast_threshold = self
                .preprocessing_params
                .contrast_threshold
                .clamp(0.01, 0.5);
            self.preprocessing_params.temporal_threshold = self
                .preprocessing_params
                .temporal_threshold
                .clamp(0.01, 0.2);
        }
    }
}

impl ProcessingStage for NeuromorphicEdgeDetector {
    fn process(&mut self, frame: Frame) -> Result<Frame> {
        // Apply neuromorphic edge detection
        let edge_map = self.detect_edges_neuromorphic(&frame)?;

        // Calculate performance metric (edge density)
        let edge_density =
            edge_map.iter().filter(|&&x| x > 0.1).count() as f64 / edge_map.len() as f64;

        // Adapt parameters
        self.adapt_parameters(edge_density);

        Ok(Frame {
            data: edge_map,
            timestamp: frame.timestamp,
            index: frame.index,
            metadata: frame.metadata,
        })
    }

    fn name(&self) -> &str {
        "NeuromorphicEdgeDetector"
    }
}

/// Event-driven sparse processing stage
#[derive(Debug)]
pub struct EventDrivenProcessor {
    /// Sparse event representation
    event_buffer: VecDeque<PixelEvent>,
    /// Event generation threshold
    _eventthreshold: f32,
    /// Previous frame for temporal differencing
    previous_frame: Option<Array2<f32>>,
    /// Spatial event clustering
    spatial_clusters: HashMap<(usize, usize), EventCluster>,
    /// Temporal integration window
    temporal_window: Duration,
    /// Processing efficiency metrics
    efficiency_metrics: EfficiencyMetrics,
}

/// Pixel change event
#[derive(Debug, Clone)]
pub struct PixelEvent {
    /// X pixel coordinate
    pub x: usize,
    /// Y pixel coordinate
    pub y: usize,
    /// Change magnitude
    pub magnitude: f32,
    /// Event timestamp
    pub timestamp: Instant,
    /// Event polarity (positive/negative change)
    pub polarity: EventPolarity,
}

/// Event polarity for sparse representation
#[derive(Debug, Clone, PartialEq)]
pub enum EventPolarity {
    /// Positive intensity change
    Positive,
    /// Negative intensity change
    Negative,
}

/// Spatial cluster of related events
#[derive(Debug, Clone)]
pub struct EventCluster {
    /// Cluster center
    pub center: (f32, f32),
    /// Events in cluster
    pub events: Vec<PixelEvent>,
    /// Cluster activity strength
    pub activity: f32,
    /// Last update time
    pub last_update: Instant,
}

/// Processing efficiency metrics
#[derive(Debug, Clone)]
pub struct EfficiencyMetrics {
    /// Percentage of pixels that generated events
    pub sparsity: f32,
    /// Energy consumption estimate
    pub energy_consumption: f32,
    /// Processing speedup from sparsity
    pub speedup_factor: f32,
    /// Data compression ratio
    pub compression_ratio: f32,
}

impl EventDrivenProcessor {
    /// Create a new event-driven processor
    pub fn new(_eventthreshold: f32) -> Self {
        Self {
            event_buffer: VecDeque::with_capacity(10000),
            _eventthreshold,
            previous_frame: None,
            spatial_clusters: HashMap::new(),
            temporal_window: Duration::from_millis(50),
            efficiency_metrics: EfficiencyMetrics {
                sparsity: 0.0,
                energy_consumption: 0.0,
                speedup_factor: 1.0,
                compression_ratio: 1.0,
            },
        }
    }

    /// Generate events from frame differences
    fn generate_events(&mut self, currentframe: &Array2<f32>) -> Vec<PixelEvent> {
        let mut events = Vec::new();
        let current_time = Instant::now();

        if let Some(ref prev_frame) = self.previous_frame {
            let (height, width) = currentframe.dim();

            for y in 0..height {
                for x in 0..width {
                    let current_val = currentframe[[y, x]];
                    let prev_val = prev_frame[[y, x]];
                    let diff = current_val - prev_val;

                    if diff.abs() > self._eventthreshold {
                        let polarity = if diff > 0.0 {
                            EventPolarity::Positive
                        } else {
                            EventPolarity::Negative
                        };

                        events.push(PixelEvent {
                            x,
                            y,
                            magnitude: diff.abs(),
                            timestamp: current_time,
                            polarity,
                        });
                    }
                }
            }
        }

        self.previous_frame = Some(currentframe.clone());
        events
    }

    /// Cluster events spatially for efficient processing
    fn cluster_events(&mut self, events: &[PixelEvent]) {
        const CLUSTER_RADIUS: f32 = 5.0;

        // Clear old clusters
        let current_time = Instant::now();
        self.spatial_clusters.retain(|_, cluster| {
            current_time.duration_since(cluster.last_update) < self.temporal_window
        });

        for event in events {
            let mut assigned_to_cluster = false;

            // Try to assign to existing cluster
            for cluster in self.spatial_clusters.values_mut() {
                let distance = ((event.x as f32 - cluster.center.0).powi(2)
                    + (event.y as f32 - cluster.center.1).powi(2))
                .sqrt();

                if distance <= CLUSTER_RADIUS {
                    cluster.events.push(event.clone());
                    cluster.activity += event.magnitude;
                    cluster.last_update = current_time;

                    // Update cluster center
                    let total_events = cluster.events.len() as f32;
                    cluster.center = (
                        (cluster.center.0 * (total_events - 1.0) + event.x as f32) / total_events,
                        (cluster.center.1 * (total_events - 1.0) + event.y as f32) / total_events,
                    );

                    assigned_to_cluster = true;
                    break;
                }
            }

            // Create new cluster if not assigned
            if !assigned_to_cluster {
                let cluster = EventCluster {
                    center: (event.x as f32, event.y as f32),
                    events: vec![event.clone()],
                    activity: event.magnitude,
                    last_update: current_time,
                };

                self.spatial_clusters.insert((event.x, event.y), cluster);
            }
        }
    }

    /// Process events efficiently using sparse representation
    fn process_events_sparse(&self, frameshape: (usize, usize)) -> Array2<f32> {
        let (height, width) = frameshape;
        let mut processed_frame = Array2::zeros((height, width));

        // Process only active clusters
        for cluster in self.spatial_clusters.values() {
            if cluster.activity > self._eventthreshold {
                // Apply processing to cluster region
                let cluster_x = cluster.center.0 as usize;
                let cluster_y = cluster.center.1 as usize;

                // Simple enhancement based on cluster activity
                let enhancement_radius = 2;
                for dy in -enhancement_radius..=enhancement_radius {
                    for dx in -enhancement_radius..=enhancement_radius {
                        let x = (cluster_x as i32 + dx).clamp(0, width as i32 - 1) as usize;
                        let y = (cluster_y as i32 + dy).clamp(0, height as i32 - 1) as usize;

                        let distance = ((dx as f32).powi(2) + (dy as f32).powi(2)).sqrt();
                        let weight = (1.0 - distance / enhancement_radius as f32).max(0.0);

                        processed_frame[[y, x]] += cluster.activity * weight;
                    }
                }
            }
        }

        // Normalize
        let max_val = processed_frame.iter().fold(0.0f32, |a, &b| a.max(b));
        if max_val > 0.0 {
            processed_frame.mapv_inplace(|x| x / max_val);
        }

        processed_frame
    }

    /// Update efficiency metrics
    fn update_efficiency_metrics(&mut self, events: &[PixelEvent], framesize: usize) {
        let event_count = events.len();

        // Calculate sparsity
        self.efficiency_metrics.sparsity = event_count as f32 / framesize as f32;

        // Estimate energy consumption (events require less energy than full processing)
        self.efficiency_metrics.energy_consumption = self.efficiency_metrics.sparsity * 0.1;

        // Calculate speedup from sparse processing
        self.efficiency_metrics.speedup_factor = 1.0 / self.efficiency_metrics.sparsity.max(0.01);

        // Calculate compression ratio
        self.efficiency_metrics.compression_ratio = framesize as f32 / event_count.max(1) as f32;
    }
}

impl ProcessingStage for EventDrivenProcessor {
    fn process(&mut self, frame: Frame) -> Result<Frame> {
        let framesize = frame.data.len();

        // Generate events from temporal differences
        let events = self.generate_events(&frame.data);

        // Cluster events spatially
        self.cluster_events(&events);

        // Process using sparse event representation
        let processed_data = self.process_events_sparse(frame.data.dim());

        // Update efficiency metrics
        self.update_efficiency_metrics(&events, framesize);

        // Store events in buffer
        for event in events {
            self.event_buffer.push_back(event);

            // Keep buffer bounded
            if self.event_buffer.len() > 10000 {
                self.event_buffer.pop_front();
            }
        }

        Ok(Frame {
            data: processed_data,
            timestamp: frame.timestamp,
            index: frame.index,
            metadata: frame.metadata,
        })
    }

    fn name(&self) -> &str {
        "EventDrivenProcessor"
    }
}

impl EventDrivenProcessor {
    /// Get current efficiency metrics
    pub fn get_efficiency_metrics(&self) -> &EfficiencyMetrics {
        &self.efficiency_metrics
    }

    /// Get current event statistics
    pub fn get_event_stats(&self) -> EventStats {
        let total_events = self.event_buffer.len();
        let active_clusters = self.spatial_clusters.len();

        let positive_events = self
            .event_buffer
            .iter()
            .filter(|event| event.polarity == EventPolarity::Positive)
            .count();

        let negative_events = total_events - positive_events;

        let avg_magnitude = if total_events > 0 {
            self.event_buffer
                .iter()
                .map(|event| event.magnitude)
                .sum::<f32>()
                / total_events as f32
        } else {
            0.0
        };

        EventStats {
            total_events,
            positive_events,
            negative_events,
            active_clusters,
            avg_event_magnitude: avg_magnitude,
            sparsity: self.efficiency_metrics.sparsity,
        }
    }
}

/// Event processing statistics
#[derive(Debug, Clone)]
pub struct EventStats {
    /// Total number of events
    pub total_events: usize,
    /// Number of positive polarity events
    pub positive_events: usize,
    /// Number of negative polarity events
    pub negative_events: usize,
    /// Number of active spatial clusters
    pub active_clusters: usize,
    /// Average event magnitude
    pub avg_event_magnitude: f32,
    /// Data sparsity ratio
    pub sparsity: f32,
}

/// Adaptive neuromorphic pipeline that combines multiple neuromorphic stages
#[derive(Debug)]
pub struct AdaptiveNeuromorphicPipeline {
    /// Neuromorphic edge detector
    edge_detector: NeuromorphicEdgeDetector,
    /// Event-driven processor
    event_processor: EventDrivenProcessor,
    /// Processing mode selection
    processing_mode: NeuromorphicMode,
    /// Adaptation parameters
    adaptation_params: AdaptationParams,
    /// Performance history
    performance_history: VecDeque<PerformanceSnapshot>,
}

/// Neuromorphic processing modes
#[derive(Debug, Clone, PartialEq)]
pub enum NeuromorphicMode {
    /// High accuracy mode with full processing
    HighAccuracy,
    /// Balanced mode with selective processing
    Balanced,
    /// Advanced-efficient mode with maximum sparsity
    AdvancedEfficient,
}

/// Adaptation parameters for neuromorphic processing
#[derive(Debug, Clone)]
pub struct AdaptationParams {
    /// Performance threshold for mode switching
    pub performance_threshold: f32,
    /// Energy budget constraint
    pub energy_budget: f32,
    /// Adaptation learning rate
    pub learning_rate: f32,
    /// Minimum accuracy requirement
    pub min_accuracy: f32,
}

/// Performance snapshot for adaptation
#[derive(Debug, Clone)]
pub struct PerformanceSnapshot {
    /// Processing accuracy estimate
    pub accuracy: f32,
    /// Energy consumption
    pub energy: f32,
    /// Processing speed (FPS)
    pub speed: f32,
    /// Data sparsity
    pub sparsity: f32,
    /// Timestamp
    pub timestamp: Instant,
}

impl AdaptiveNeuromorphicPipeline {
    /// Create a new adaptive neuromorphic pipeline
    pub fn new(_inputsize: usize) -> Self {
        let edge_detector = NeuromorphicEdgeDetector::new(_inputsize);
        let event_processor = EventDrivenProcessor::new(0.05);

        Self {
            edge_detector,
            event_processor,
            processing_mode: NeuromorphicMode::Balanced,
            adaptation_params: AdaptationParams {
                performance_threshold: 0.8,
                energy_budget: 1.0,
                learning_rate: 0.01,
                min_accuracy: 0.6,
            },
            performance_history: VecDeque::with_capacity(100),
        }
    }

    /// Process frame with adaptive neuromorphic processing
    pub fn process_adaptive(&mut self, frame: Frame) -> Result<Frame> {
        let start_time = Instant::now();

        // Select processing based on current mode
        let processed_frame = match self.processing_mode {
            NeuromorphicMode::HighAccuracy => {
                // Full neuromorphic processing
                let edge_frame = self.edge_detector.process(frame)?;
                self.event_processor.process(edge_frame)?
            }
            NeuromorphicMode::Balanced => {
                // Selective processing based on activity
                let event_stats = self.event_processor.get_event_stats();

                if event_stats.sparsity > 0.1 {
                    // High activity, use edge detection
                    let edge_frame = self.edge_detector.process(frame)?;
                    self.event_processor.process(edge_frame)?
                } else {
                    // Low activity, use event processing only
                    self.event_processor.process(frame)?
                }
            }
            NeuromorphicMode::AdvancedEfficient => {
                // Event-driven processing only
                self.event_processor.process(frame)?
            }
        };

        let processing_time = start_time.elapsed();

        // Record performance snapshot
        let efficiency_metrics = self.event_processor.get_efficiency_metrics();
        let snapshot = PerformanceSnapshot {
            accuracy: self.estimate_accuracy(&processed_frame),
            energy: efficiency_metrics.energy_consumption,
            speed: 1.0 / processing_time.as_secs_f32(),
            sparsity: efficiency_metrics.sparsity,
            timestamp: Instant::now(),
        };

        self.performance_history.push_back(snapshot);
        if self.performance_history.len() > 100 {
            self.performance_history.pop_front();
        }

        // Adapt processing mode if needed
        self.adapt_processing_mode();

        Ok(processed_frame)
    }

    /// Estimate processing accuracy (simplified)
    fn estimate_accuracy(&self, frame: &Frame) -> f32 {
        // Simple heuristic based on information content
        let mean = frame.data.mean().unwrap_or(0.0);
        let variance =
            frame.data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / frame.data.len() as f32;
        let edge_density =
            frame.data.iter().filter(|&&x| x > 0.1).count() as f32 / frame.data.len() as f32;

        (variance.sqrt() + edge_density).min(1.0)
    }

    /// Adapt processing mode based on performance history
    fn adapt_processing_mode(&mut self) {
        if self.performance_history.len() < 10 {
            return;
        }

        let recent_performance = &self
            .performance_history
            .iter()
            .rev()
            .take(10)
            .cloned()
            .collect::<Vec<_>>();

        let avg_accuracy = recent_performance.iter().map(|p| p.accuracy).sum::<f32>()
            / recent_performance.len() as f32;

        let avg_energy = recent_performance.iter().map(|p| p.energy).sum::<f32>()
            / recent_performance.len() as f32;

        let avg_speed = recent_performance.iter().map(|p| p.speed).sum::<f32>()
            / recent_performance.len() as f32;

        // Adaptation logic
        match self.processing_mode {
            NeuromorphicMode::HighAccuracy => {
                if avg_energy > self.adaptation_params.energy_budget && avg_speed < 30.0 {
                    self.processing_mode = NeuromorphicMode::Balanced;
                }
            }
            NeuromorphicMode::Balanced => {
                if avg_accuracy < self.adaptation_params.min_accuracy {
                    self.processing_mode = NeuromorphicMode::HighAccuracy;
                } else if avg_energy < self.adaptation_params.energy_budget * 0.5
                    && avg_speed > 60.0
                {
                    self.processing_mode = NeuromorphicMode::AdvancedEfficient;
                }
            }
            NeuromorphicMode::AdvancedEfficient => {
                if avg_accuracy < self.adaptation_params.min_accuracy * 0.8 {
                    self.processing_mode = NeuromorphicMode::Balanced;
                }
            }
        }
    }

    /// Get current processing statistics
    pub fn get_processing_stats(&self) -> NeuromorphicProcessingStats {
        let efficiency_metrics = self.event_processor.get_efficiency_metrics();
        let event_stats = self.event_processor.get_event_stats();

        let recent_performance = if !self.performance_history.is_empty() {
            self.performance_history
                .back()
                .expect("Performance history should not be empty after check")
                .clone()
        } else {
            PerformanceSnapshot {
                accuracy: 0.0,
                energy: 0.0,
                speed: 0.0,
                sparsity: 0.0,
                timestamp: Instant::now(),
            }
        };

        NeuromorphicProcessingStats {
            current_mode: self.processing_mode.clone(),
            accuracy: recent_performance.accuracy,
            energy_consumption: efficiency_metrics.energy_consumption,
            processing_speed: recent_performance.speed,
            sparsity: efficiency_metrics.sparsity,
            speedup_factor: efficiency_metrics.speedup_factor,
            total_events: event_stats.total_events,
            active_clusters: event_stats.active_clusters,
        }
    }

    /// Initialize adaptive learning capabilities
    pub async fn initialize_adaptive_learning(&mut self) -> Result<()> {
        // Reset performance history for fresh learning
        self.performance_history.clear();

        // Initialize optimal processing mode
        self.processing_mode = NeuromorphicMode::Balanced;

        // Reset adaptation parameters to defaults
        self.adaptation_params = AdaptationParams {
            performance_threshold: 0.8,
            energy_budget: 1.0,
            learning_rate: 0.01,
            min_accuracy: 0.6,
        };

        // Edge detector and event processor initialization handled in constructor

        Ok(())
    }
}

/// Comprehensive neuromorphic processing statistics
#[derive(Debug, Clone)]
pub struct NeuromorphicProcessingStats {
    /// Current processing mode
    pub current_mode: NeuromorphicMode,
    /// Processing accuracy estimate
    pub accuracy: f32,
    /// Energy consumption
    pub energy_consumption: f32,
    /// Processing speed (FPS)
    pub processing_speed: f32,
    /// Data sparsity ratio
    pub sparsity: f32,
    /// Speedup factor from neuromorphic processing
    pub speedup_factor: f32,
    /// Total number of events processed
    pub total_events: usize,
    /// Number of active spatial clusters
    pub active_clusters: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spiking_neuron() {
        let mut neuron = SpikingNeuron::new();

        // Test with moderate input current (should not spike)
        let spiked = neuron.update(1.0, 10.0);
        assert!(!spiked);
        assert!(neuron.membrane_potential > neuron.resting_potential);

        // Test spike generation with high current
        let mut spike_occurred = false;
        for _ in 0..100 {
            if neuron.update(1.0, 100.0) {
                spike_occurred = true;
                break;
            }
        }
        assert!(spike_occurred);
    }

    #[test]
    fn test_plastic_synapse() {
        let mut synapse = PlasticSynapse::new(0, 1, 0.5);

        // Test STDP with positive timing
        synapse.update_weight(Some(10.0), Some(15.0));
        assert!(synapse.weight >= 0.5); // Should increase

        // Test STDP with negative timing
        synapse.update_weight(Some(20.0), Some(18.0));
        // Weight might decrease depending on timing
    }

    #[test]
    fn test_spiking_neural_network() {
        let mut snn = SpikingNeuralNetwork::new(10, 0.2);
        let input = Array1::from_vec(vec![1.0, 0.5, 0.8, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        let output = snn.process_input(&input);
        assert_eq!(output.len(), 10);

        let stats = snn.get_activity_stats();
        assert!(stats.avg_membrane_potential < 0.0); // Should be negative
    }

    #[test]
    fn test_neuromorphic_edge_detector() {
        let mut detector = NeuromorphicEdgeDetector::new(64);

        let frame = Frame {
            data: Array2::from_shape_fn((8, 8), |(_y, x)| if x > 4 { 1.0 } else { 0.0 }),
            timestamp: Instant::now(),
            index: 0,
            metadata: Some(FrameMetadata {
                width: 8,
                height: 8,
                fps: 30.0,
                channels: 1,
            }),
        };

        let result = detector.process(frame);
        assert!(result.is_ok());

        let processed = result.expect("Result should be Ok after assertion");
        assert_eq!(processed.data.dim(), (8, 8));
    }

    #[test]
    fn test_event_driven_processor() {
        let mut processor = EventDrivenProcessor::new(0.1);

        // Create frame with some structure
        let frame1 = Frame {
            data: Array2::zeros((10, 10)),
            timestamp: Instant::now(),
            index: 0,
            metadata: None,
        };

        let frame2 = Frame {
            data: Array2::from_shape_fn((10, 10), |(_x, y)| if y == 5 { 1.0 } else { 0.0 }),
            timestamp: Instant::now(),
            index: 1,
            metadata: None,
        };

        // Process first frame
        let result1 = processor.process(frame1);
        assert!(result1.is_ok());

        // Process second frame (should generate events)
        let result2 = processor.process(frame2);
        assert!(result2.is_ok());

        let stats = processor.get_event_stats();
        println!("Event stats: {stats:?}");
    }

    #[test]
    fn test_adaptive_neuromorphic_pipeline() {
        let mut pipeline = AdaptiveNeuromorphicPipeline::new(64);

        let frame = Frame {
            data: Array2::from_shape_fn((8, 8), |(y, x)| (x + y) as f32 / 16.0),
            timestamp: Instant::now(),
            index: 0,
            metadata: None,
        };

        let result = pipeline.process_adaptive(frame);
        assert!(result.is_ok());

        let stats = pipeline.get_processing_stats();
        assert!(stats.sparsity >= 0.0 && stats.sparsity <= 1.0);
    }
}
