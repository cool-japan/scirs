//! Spike-Based Optimization Algorithms
//!
//! This module implements optimization algorithms that operate on spike trains
//! and temporal spike patterns, designed for neuromorphic computing platforms.

use super::{
    Spike, SpikeTrain, NeuromorphicMetrics, STDPConfig, MembraneDynamicsConfig,
    PlasticityModel, NeuromorphicEvent, EventPriority
};
use crate::error::Result;
use crate::optimizers::Optimizer;
use ndarray::{Array1, Array2, ArrayBase, Data, DataMut, Dimension};
use num_traits::Float;
use std::collections::{HashMap, VecDeque};
use std::time::Instant;

/// Spike-based optimization configuration
#[derive(Debug, Clone)]
pub struct SpikingConfig<T: Float> {
    /// Simulation time step (ms)
    pub time_step: T,
    
    /// Total simulation time (ms)
    pub simulation_time: T,
    
    /// Encoding method for input data
    pub encoding_method: SpikeEncodingMethod,
    
    /// Decoding method for output spikes
    pub decoding_method: SpikeDecodingMethod,
    
    /// Spike train learning rate
    pub spike_learning_rate: T,
    
    /// Temporal window for spike correlation (ms)
    pub temporal_window: T,
    
    /// Enable lateral inhibition
    pub lateral_inhibition: bool,
    
    /// Homeostatic scaling parameters
    pub homeostatic_config: HomeostaticConfig<T>,
    
    /// Noise parameters for spike generation
    pub noise_config: SpikeNoiseConfig<T>}

/// Spike encoding methods for converting continuous values to spike trains
#[derive(Debug, Clone, Copy)]
pub enum SpikeEncodingMethod {
    /// Rate coding (firing rate proportional to value)
    RateCoding,
    
    /// Temporal coding (spike time proportional to value)
    TemporalCoding,
    
    /// Population vector coding
    PopulationVectorCoding,
    
    /// Sparse coding
    SparseCoding,
    
    /// Phase coding
    PhaseCoding,
    
    /// Burst coding
    BurstCoding,
    
    /// Rank order coding
    RankOrderCoding}

/// Spike decoding methods for converting spike trains to continuous values
#[derive(Debug, Clone, Copy)]
pub enum SpikeDecodingMethod {
    /// Rate decoding (spike count in time window)
    RateDecoding,
    
    /// Temporal decoding (first spike time)
    TemporalDecoding,
    
    /// Population vector decoding
    PopulationVectorDecoding,
    
    /// Weighted spike count
    WeightedSpikeCount,
    
    /// Moving average filter
    MovingAverageFilter,
    
    /// Exponential decay filter
    ExponentialDecayFilter}

/// Homeostatic plasticity configuration
#[derive(Debug, Clone)]
pub struct HomeostaticConfig<T: Float> {
    /// Enable homeostatic scaling
    pub enable_homeostatic_scaling: bool,
    
    /// Target firing rate (Hz)
    pub target_firing_rate: T,
    
    /// Scaling time constant (ms)
    pub scaling_time_constant: T,
    
    /// Scaling factor
    pub scaling_factor: T,
    
    /// Enable intrinsic plasticity
    pub enable_intrinsic_plasticity: bool,
    
    /// Threshold adaptation rate
    pub threshold_adaptation_rate: T}

/// Spike noise configuration
#[derive(Debug, Clone)]
pub struct SpikeNoiseConfig<T: Float> {
    /// Background firing rate (Hz)
    pub background_rate: T,
    
    /// Jitter standard deviation (ms)
    pub jitter_std: T,
    
    /// Enable Poisson noise
    pub poisson_noise: bool,
    
    /// Noise amplitude
    pub noise_amplitude: T,
    
    /// Correlation noise
    pub correlation_noise: T}

impl<T: Float> Default for SpikingConfig<T> {
    fn default() -> Self {
        Self {
            time_step: num_traits::cast::cast(0.1).unwrap_or_else(|| T::zero()),
            simulation_time: num_traits::cast::cast(1000.0).unwrap_or_else(|| T::zero()),
            encoding_method: SpikeEncodingMethod::RateCoding,
            decoding_method: SpikeDecodingMethod::RateDecoding,
            spike_learning_rate: num_traits::cast::cast(0.01).unwrap_or_else(|| T::zero()),
            temporal_window: num_traits::cast::cast(20.0).unwrap_or_else(|| T::zero()),
            lateral_inhibition: false,
            homeostatic_config: HomeostaticConfig::default(),
            noise_config: SpikeNoiseConfig::default()}
    }
}

impl<T: Float> Default for HomeostaticConfig<T> {
    fn default() -> Self {
        Self {
            enable_homeostatic_scaling: false,
            target_firing_rate: num_traits::cast::cast(10.0).unwrap_or_else(|| T::zero()),
            scaling_time_constant: num_traits::cast::cast(1000.0).unwrap_or_else(|| T::zero()),
            scaling_factor: num_traits::cast::cast(0.01).unwrap_or_else(|| T::zero()),
            enable_intrinsic_plasticity: false,
            threshold_adaptation_rate: num_traits::cast::cast(0.001).unwrap_or_else(|| T::zero())}
    }
}

impl<T: Float> Default for SpikeNoiseConfig<T> {
    fn default() -> Self {
        Self {
            background_rate: num_traits::cast::cast(1.0).unwrap_or_else(|| T::zero()),
            jitter_std: num_traits::cast::cast(0.5).unwrap_or_else(|| T::zero()),
            poisson_noise: false,
            noise_amplitude: num_traits::cast::cast(0.1).unwrap_or_else(|| T::zero()),
            correlation_noise: T::zero()}
    }
}

/// Spike-based optimizer
pub struct SpikingOptimizer<T: Float + ndarray::ScalarOperand + std::fmt::Debug> {
    /// Configuration
    config: SpikingConfig<T>,
    
    /// STDP configuration
    stdp_config: STDPConfig<T>,
    
    /// Membrane dynamics configuration
    membrane_config: MembraneDynamicsConfig<T>,
    
    /// Current simulation time
    current_time: T,
    
    /// Spike trains for each neuron
    spike_trains: HashMap<usize, SpikeTrain<T>>,
    
    /// Current membrane potentials
    membrane_potentials: Array1<T>,
    
    /// Synaptic weights
    synaptic_weights: Array2<T>,
    
    /// Last spike times for each neuron
    last_spike_times: Array1<T>,
    
    /// Refractory state
    refractory_until: Array1<T>,
    
    /// Homeostatic scaling factors
    homeostatic_scales: Array1<T>,
    
    /// Spike buffer for temporal processing
    spike_buffer: VecDeque<Spike<T>>,
    
    /// Performance metrics
    metrics: NeuromorphicMetrics<T>,
    
    /// Plasticity model
    plasticity_model: PlasticityModel}

impl<T: Float + Send + Sync + ndarray::ScalarOperand + std::fmt::Debug> SpikingOptimizer<T> {
    /// Create a new spiking optimizer
    pub fn new(
        config: SpikingConfig<T>,
        stdp_config: STDPConfig<T>,
        membrane_config: MembraneDynamicsConfig<T>,
        num_neurons: usize,
    ) -> Self {
        Self {
            config,
            stdp_config,
            membrane_config,
            current_time: T::zero(),
            spike_trains: HashMap::new(),
            membrane_potentials: Array1::from_elem(num_neurons, membrane_config.resting_potential),
            synaptic_weights: Array2::ones((num_neurons, num_neurons)) * num_traits::cast::cast(0.1).unwrap_or_else(|| T::zero()),
            last_spike_times: Array1::from_elem(num_neurons, num_traits::cast::cast(-1000.0).unwrap_or_else(|| T::zero())),
            refractory_until: Array1::zeros(num_neurons),
            homeostatic_scales: Array1::ones(num_neurons),
            spike_buffer: VecDeque::new(),
            metrics: NeuromorphicMetrics::default(),
            plasticity_model: PlasticityModel::STDP}
    }
    
    /// Encode continuous input as spike trains
    pub fn encode_input(&self, input: &Array1<T>) -> Result<Vec<SpikeTrain<T>>> {
        let mut spike_trains = Vec::new();
        
        for (neuron_id, &value) in input.iter().enumerate() {
            let spike_train = match self.config.encoding_method {
                SpikeEncodingMethod::RateCoding => {
                    self.rate_encode(neuron_id, value)?
                }
                SpikeEncodingMethod::TemporalCoding => {
                    self.temporal_encode(neuron_id, value)?
                }
                SpikeEncodingMethod::PopulationVectorCoding => {
                    self.population_vector_encode(neuron_id, value)?
                }
                SpikeEncodingMethod::SparseCoding => {
                    self.sparse_encode(neuron_id, value)?
                }
                _ => {
                    // Fallback to rate coding
                    self.rate_encode(neuron_id, value)?
                }
            };
            
            spike_trains.push(spike_train);
        }
        
        Ok(spike_trains)
    }
    
    /// Rate encoding: firing rate proportional to input value
    fn rate_encode(&self, neuronid: usize, value: T) -> Result<SpikeTrain<T>> {
        let max_rate = num_traits::cast::cast(100.0).unwrap_or_else(|| T::zero()); // 100 Hz max
        let firing_rate = value.abs() * max_rate;
        
        let mut spike_times = Vec::new();
        let dt = self.config.time_step;
        let total_time = self.config.simulation_time;
        
        let mut time = T::zero();
        while time < total_time {
            // Poisson process: probability of spike in dt
            let spike_prob = firing_rate * dt / num_traits::cast::cast(1000.0).unwrap_or_else(|| T::zero());
            
            if fastrand::f64() < spike_prob.to_f64().unwrap_or(0.0) {
                spike_times.push(time);
            }
            
            time = time + dt;
        }
        
        Ok(SpikeTrain::new(neuron_id, spike_times))
    }
    
    /// Temporal encoding: spike time inversely proportional to input value
    fn temporal_encode(&self, neuronid: usize, value: T) -> Result<SpikeTrain<T>> {
        let max_delay = num_traits::cast::cast(20.0).unwrap_or_else(|| T::zero()); // 20 ms max delay
        let spike_time = if value > T::zero() {
            max_delay * (T::one() - value.min(T::one()))
        } else {
            max_delay // No spike for negative values
        };
        
        let spike_times = if spike_time < max_delay {
            vec![spike_time]
        } else {
            Vec::new()
        };
        
        Ok(SpikeTrain::new(neuron_id, spike_times))
    }
    
    /// Population vector encoding
    fn population_vector_encode(&self, neuronid: usize, value: T) -> Result<SpikeTrain<T>> {
        // Simplified population vector encoding
        self.rate_encode(neuron_id, value)
    }
    
    /// Sparse encoding: only strong inputs generate spikes
    fn sparse_encode(&self, neuronid: usize, value: T) -> Result<SpikeTrain<T>> {
        let threshold = num_traits::cast::cast(0.5).unwrap_or_else(|| T::zero());
        
        if value.abs() > threshold {
            self.rate_encode(neuron_id, value)
        } else {
            Ok(SpikeTrain::new(neuron_id, Vec::new()))
        }
    }
    
    /// Decode spike trains to continuous output
    pub fn decode_output(&self, spiketrains: &[SpikeTrain<T>]) -> Result<Array1<T>> {
        let mut output = Array1::zeros(spike_trains.len());
        
        for (i, spike_train) in spike_trains.iter().enumerate() {
            output[i] = match self.config.decoding_method {
                SpikeDecodingMethod::RateDecoding => {
                    self.rate_decode(spike_train)?
                }
                SpikeDecodingMethod::TemporalDecoding => {
                    self.temporal_decode(spike_train)?
                }
                SpikeDecodingMethod::WeightedSpikeCount => {
                    self.weighted_spike_count_decode(spike_train)?
                }
                _ => {
                    // Fallback to rate decoding
                    self.rate_decode(spike_train)?
                }
            };
        }
        
        Ok(output)
    }
    
    /// Rate decoding: spike count normalized by time window
    fn rate_decode(&self, spiketrain: &SpikeTrain<T>) -> Result<T> {
        let window_duration = self.config.temporal_window;
        let spike_count = num_traits::cast::cast(spike_train.spike_count).unwrap_or_else(|| T::zero());
        let rate = spike_count / (window_duration / num_traits::cast::cast(1000.0).unwrap_or_else(|| T::zero()));
        Ok(rate / num_traits::cast::cast(100.0).unwrap_or_else(|| T::zero())) // Normalize by max expected rate
    }
    
    /// Temporal decoding: use first spike time
    fn temporal_decode(&self, spiketrain: &SpikeTrain<T>) -> Result<T> {
        if spike_train.spike_times.is_empty() {
            Ok(T::zero())
        } else {
            let first_spike = spike_train.spike_times[0];
            let max_delay = num_traits::cast::cast(20.0).unwrap_or_else(|| T::zero());
            Ok(T::one() - (first_spike / max_delay).min(T::one()))
        }
    }
    
    /// Weighted spike count decoding
    fn weighted_spike_count_decode(&self, spiketrain: &SpikeTrain<T>) -> Result<T> {
        if spike_train.spike_times.is_empty() {
            return Ok(T::zero());
        }
        
        let mut weighted_sum = T::zero();
        let current_time = self.current_time;
        
        for &spike_time in &spike_train.spike_times {
            let time_diff = current_time - spike_time;
            let weight = (-time_diff / num_traits::cast::cast(10.0).unwrap_or_else(|| T::zero())).exp(); // Exponential decay
            weighted_sum = weighted_sum + weight;
        }
        
        Ok(weighted_sum)
    }
    
    /// Simulate membrane dynamics for one time step
    pub fn simulate_step(&mut self, inputspikes: &[Spike<T>]) -> Result<Vec<Spike<T>>> {
        let mut output_spikes = Vec::new();
        let dt = self.config.time_step;
        
        // Process input _spikes
        for spike in input_spikes {
            self.process_input_spike(spike)?;
        }
        
        // Update membrane potentials
        for neuron_id in 0..self.membrane_potentials.len() {
            if self.current_time >= self.refractory_until[neuron_id] {
                self.update_membrane_potential(neuron_id, dt)?;
                
                // Check for spike threshold
                if self.membrane_potentials[neuron_id] >= self.membrane_config.threshold_potential {
                    let spike = self.generate_spike(neuron_id)?;
                    output_spikes.push(spike);
                }
            }
        }
        
        // Apply plasticity updates
        self.update_plasticity(&output_spikes)?;
        
        // Update homeostatic mechanisms
        if self.config.homeostatic_config.enable_homeostatic_scaling {
            self.update_homeostatic_scaling()?;
        }
        
        self.current_time = self.current_time + dt;
        
        Ok(output_spikes)
    }
    
    /// Process an input spike
    fn process_input_spike(&mut self, spike: &Spike<T>) -> Result<()> {
        let target_neuron = spike.postsynaptic_id.unwrap_or(spike.neuron_id);
        
        if target_neuron < self.membrane_potentials.len() {
            // Add synaptic current
            let synaptic_current = spike.weight * spike.amplitude;
            self.membrane_potentials[target_neuron] = 
                self.membrane_potentials[target_neuron] + synaptic_current;
        }
        
        Ok(())
    }
    
    /// Update membrane potential using leaky integrate-and-fire model
    fn update_membrane_potential(&mut self, neuronid: usize, dt: T) -> Result<()> {
        let v = self.membrane_potentials[neuron_id];
        let v_rest = self.membrane_config.resting_potential;
        let tau = self.membrane_config.tau_membrane;
        
        // Leaky integration: dV/dt = (V_rest - V) / tau
        let dv_dt = (v_rest - v) / tau;
        let new_v = v + dv_dt * dt;
        
        self.membrane_potentials[neuron_id] = new_v;
        
        Ok(())
    }
    
    /// Generate a spike when threshold is reached
    fn generate_spike(&mut self, neuronid: usize) -> Result<Spike<T>> {
        // Reset membrane potential
        self.membrane_potentials[neuron_id] = self.membrane_config.reset_potential;
        
        // Set refractory period
        self.refractory_until[neuron_id] = 
            self.current_time + self.membrane_config.refractory_period;
        
        // Update last spike time
        self.last_spike_times[neuron_id] = self.current_time;
        
        // Create spike
        let spike = Spike {
            neuron_id,
            time: self.current_time,
            amplitude: num_traits::cast::cast(1.0).unwrap_or_else(|| T::zero()),
            width: Some(num_traits::cast::cast(1.0).unwrap_or_else(|| T::zero())),
            weight: T::one(),
            presynaptic_id: None,
            postsynaptic_id: None};
        
        // Update spike train
        if let Some(spike_train) = self.spike_trains.get_mut(&neuron_id) {
            spike_train.spike_times.push(self.current_time);
            spike_train.spike_count += 1;
        } else {
            let spike_train = SpikeTrain::new(neuron_id, vec![self.current_time]);
            self.spike_trains.insert(neuron_id, spike_train);
        }
        
        // Update metrics
        self.metrics.total_spikes += 1;
        
        Ok(spike)
    }
    
    /// Update synaptic plasticity
    fn update_plasticity(&mut self, outputspikes: &[Spike<T>]) -> Result<()> {
        match self.plasticity_model {
            PlasticityModel::STDP => {
                self.update_stdp(output_spikes)?;
            }
            PlasticityModel::Hebbian => {
                self.update_hebbian(output_spikes)?;
            }
            _ => {
                // Default to STDP
                self.update_stdp(output_spikes)?;
            }
        }
        
        Ok(())
    }
    
    /// Update STDP (Spike Timing Dependent Plasticity)
    fn update_stdp(&mut self, outputspikes: &[Spike<T>]) -> Result<()> {
        for spike in output_spikes {
            let post_id = spike.neuron_id;
            let post_time = spike.time;
            
            // Check all presynaptic connections
            for pre_id in 0..self.last_spike_times.len() {
                if pre_id != post_id {
                    let pre_time = self.last_spike_times[pre_id];
                    
                    if pre_time > num_traits::cast::cast(-1000.0).unwrap_or_else(|| T::zero()) { // Valid spike time
                        let dt = post_time - pre_time;
                        let weight_change = self.compute_stdp_update(dt);
                        
                        // Update synaptic weight
                        self.synaptic_weights[[pre_id, post_id]] = 
                            (self.synaptic_weights[[pre_id, post_id]] + weight_change)
                                .max(self.stdp_config.weight_min)
                                .min(self.stdp_config.weight_max);
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Compute STDP weight update
    fn compute_stdp_update(&self, dt: T) -> T {
        if dt > T::zero() {
            // Post-before-pre: LTP (potentiation)
            let exp_arg = -dt / self.stdp_config.tau_pot;
            self.stdp_config.learning_rate_pot * exp_arg.exp()
        } else {
            // Pre-before-post: LTD (depression)  
            let exp_arg = dt / self.stdp_config.tau_dep;
            -self.stdp_config.learning_rate_dep * exp_arg.exp()
        }
    }
    
    /// Update Hebbian plasticity
    fn update_hebbian(&mut self, outputspikes: &[Spike<T>]) -> Result<()> {
        // Simplified Hebbian learning
        for spike in output_spikes {
            let post_id = spike.neuron_id;
            
            for pre_id in 0..self.membrane_potentials.len() {
                if pre_id != post_id {
                    let pre_activity = self.membrane_potentials[pre_id] / 
                        self.membrane_config.threshold_potential;
                    
                    let weight_change = self.stdp_config.learning_rate_pot * pre_activity;
                    
                    self.synaptic_weights[[pre_id, post_id]] = 
                        (self.synaptic_weights[[pre_id, post_id]] + weight_change)
                            .max(self.stdp_config.weight_min)
                            .min(self.stdp_config.weight_max);
                }
            }
        }
        
        Ok(())
    }
    
    /// Update homeostatic scaling
    fn update_homeostatic_scaling(&mut self) -> Result<()> {
        let target_rate = self.config.homeostatic_config.target_firing_rate;
        let time_constant = self.config.homeostatic_config.scaling_time_constant;
        let dt = self.config.time_step;
        
        for neuron_id in 0..self.homeostatic_scales.len() {
            if let Some(spike_train) = self.spike_trains.get(&neuron_id) {
                let current_rate = spike_train.firing_rate;
                let rate_error = target_rate - current_rate;
                
                // Exponential approach to target
                let scale_change = rate_error * dt / time_constant;
                self.homeostatic_scales[neuron_id] = 
                    self.homeostatic_scales[neuron_id] + scale_change;
                
                // Apply scaling to synaptic weights
                for pre_id in 0..self.synapticweights.nrows() {
                    self.synaptic_weights[[pre_id, neuron_id]] = 
                        self.synaptic_weights[[pre_id, neuron_id]] * 
                        self.homeostatic_scales[neuron_id];
                }
            }
        }
        
        Ok(())
    }
    
    /// Get current neuromorphic metrics
    pub fn get_metrics(&self) -> &NeuromorphicMetrics<T> {
        &self.metrics
    }
    
    /// Reset the optimizer state
    pub fn reset(&mut self) {
        self.current_time = T::zero();
        self.membrane_potentials.fill(self.membrane_config.resting_potential);
        self.last_spike_times.fill(num_traits::cast::cast(-1000.0).unwrap_or_else(|| T::zero()));
        self.refractory_until.fill(T::zero());
        self.spike_trains.clear();
        self.spike_buffer.clear();
        self.metrics = NeuromorphicMetrics::default();
    }
}

/// Spike train optimizer for temporal pattern learning
pub struct SpikeTrainOptimizer<T: Float + ndarray::ScalarOperand + std::fmt::Debug> {
    /// Configuration
    config: SpikingConfig<T>,
    
    /// Spike pattern templates
    pattern_templates: Vec<SpikePattern<T>>,
    
    /// Pattern matching threshold
    matching_threshold: T,
    
    /// Learning rate for pattern adaptation
    pattern_learning_rate: T,
    
    /// Temporal kernel for pattern comparison
    temporal_kernel: TemporalKernel<T>}

/// Spike pattern template
#[derive(Debug, Clone)]
pub struct SpikePattern<T: Float> {
    /// Pattern ID
    pub pattern_id: usize,
    
    /// Spike times relative to pattern start
    pub relative_spike_times: Vec<T>,
    
    /// Pattern duration
    pub duration: T,
    
    /// Pattern weight/importance
    pub weight: T,
    
    /// Number of times pattern was observed
    pub observation_count: usize}

/// Temporal kernel for pattern matching
#[derive(Debug, Clone)]
pub struct TemporalKernel<T: Float> {
    /// Kernel type
    pub kernel_type: TemporalKernelType,
    
    /// Kernel width (ms)
    pub width: T,
    
    /// Kernel parameters
    pub parameters: Vec<T>}

/// Types of temporal kernels
#[derive(Debug, Clone, Copy)]
pub enum TemporalKernelType {
    /// Gaussian kernel
    Gaussian,
    
    /// Exponential kernel
    Exponential,
    
    /// Alpha function kernel
    Alpha,
    
    /// Rectangular kernel
    Rectangular}

impl<T: Float + Send + Sync + ndarray::ScalarOperand + std::fmt::Debug> SpikeTrainOptimizer<T> {
    /// Create a new spike train optimizer
    pub fn new(config: SpikingConfig<T>) -> Self {
        Self {
            config,
            pattern_templates: Vec::new(),
            matching_threshold: num_traits::cast::cast(0.8).unwrap_or_else(|| T::zero()),
            pattern_learning_rate: num_traits::cast::cast(0.1).unwrap_or_else(|| T::zero()),
            temporal_kernel: TemporalKernel {
                kernel_type: TemporalKernelType::Gaussian,
                width: num_traits::cast::cast(5.0).unwrap_or_else(|| T::zero()),
                parameters: vec![T::one()]}}
    }
    
    /// Learn spike patterns from training data
    pub fn learn_patterns(&mut self, spiketrains: &[SpikeTrain<T>]) -> Result<()> {
        for spike_train in spike_trains {
            self.extract_and_learn_patterns(spike_train)?;
        }
        
        Ok(())
    }
    
    /// Extract patterns from a spike train
    fn extract_and_learn_patterns(&mut self, spiketrain: &SpikeTrain<T>) -> Result<()> {
        let window_size = num_traits::cast::cast(50.0).unwrap_or_else(|| T::zero()); // 50 ms windows
        let step_size = num_traits::cast::cast(10.0).unwrap_or_else(|| T::zero());   // 10 ms steps
        
        let mut window_start = T::zero();
        
        while window_start < spike_train.duration {
            let window_end = window_start + window_size;
            
            // Extract spikes in current window
            let window_spikes: Vec<T> = spike_train.spike_times.iter()
                .filter(|&&t| t >= window_start && t < window_end)
                .map(|&t| t - window_start) // Make relative to window start
                .collect();
            
            if !window_spikes.is_empty() {
                let pattern = SpikePattern {
                    pattern_id: self.pattern_templates.len(),
                    relative_spike_times: window_spikes,
                    duration: window_size,
                    weight: T::one(),
                    observation_count: 1};
                
                // Check if similar pattern exists
                if let Some(similar_pattern_id) = self.find_similar_pattern(&pattern) {
                    self.update_pattern(similar_pattern_id, &pattern)?;
                } else {
                    self.pattern_templates.push(pattern);
                }
            }
            
            window_start = window_start + step_size;
        }
        
        Ok(())
    }
    
    /// Find similar existing pattern
    fn find_similar_pattern(&self, newpattern: &SpikePattern<T>) -> Option<usize> {
        for (i, existing_pattern) in self.pattern_templates.iter().enumerate() {
            let similarity = self.compute_pattern_similarity(new_pattern, existing_pattern);
            if similarity > self.matching_threshold {
                return Some(i);
            }
        }
        
        None
    }
    
    /// Compute similarity between two spike patterns
    fn compute_pattern_similarity(&self, pattern1: &SpikePattern<T>, pattern2: &SpikePattern<T>) -> T {
        // Use Victor-Purpura distance or similar metric
        let max_spikes = pattern1.relative_spike_times.len().max(pattern2.relative_spike_times.len());
        if max_spikes == 0 {
            return T::one();
        }
        
        // Simplified similarity based on spike count and timing
        let count_diff = (pattern1.relative_spike_times.len() as i32 - 
                         pattern2.relative_spike_times.len() as i32).abs() as f64;
        let count_similarity = T::one() - num_traits::cast::cast(count_diff / max_spikes as f64).unwrap_or_else(|| T::zero());
        
        // Add temporal similarity if both patterns have spikes
        if !pattern1.relative_spike_times.is_empty() && !pattern2.relative_spike_times.is_empty() {
            let temporal_similarity = self.compute_temporal_similarity(
                &pattern1.relative_spike_times,
                &pattern2.relative_spike_times,
            );
            (count_similarity + temporal_similarity) / num_traits::cast::cast(2.0).unwrap_or_else(|| T::zero())
        } else {
            count_similarity
        }
    }
    
    /// Compute temporal similarity between spike time sequences
    fn compute_temporal_similarity(&self, spikes1: &[T], spikes2: &[T]) -> T {
        // Use cross-correlation or DTW-like measure
        let mut max_correlation = T::zero();
        let max_shift = num_traits::cast::cast(10.0).unwrap_or_else(|| T::zero()); // 10 ms max shift
        let shift_step = num_traits::cast::cast(1.0).unwrap_or_else(|| T::zero());
        
        let mut shift = -max_shift;
        while shift <= max_shift {
            let correlation = self.compute_spike_correlation(spikes1, spikes2, shift);
            max_correlation = max_correlation.max(correlation);
            shift = shift + shift_step;
        }
        
        max_correlation
    }
    
    /// Compute spike correlation with time shift
    fn compute_spike_correlation(&self, spikes1: &[T], spikes2: &[T], shift: T) -> T {
        let mut correlation = T::zero();
        let kernel_width = self.temporal_kernel.width;
        
        for &t1 in spikes1 {
            for &t2 in spikes2 {
                let dt = (t1 - (t2 + shift)).abs();
                let kernel_value = (-dt * dt / (num_traits::cast::cast(2.0).unwrap_or_else(|| T::zero()) * kernel_width * kernel_width)).exp();
                correlation = correlation + kernel_value;
            }
        }
        
        // Normalize by number of spike pairs
        if !spikes1.is_empty() && !spikes2.is_empty() {
            correlation / T::from(spikes1.len() * spikes2.len()).unwrap()
        } else {
            T::zero()
        }
    }
    
    /// Update existing pattern with new observation
    fn update_pattern(&mut self, pattern_id: usize, newpattern: &SpikePattern<T>) -> Result<()> {
        if let Some(existing_pattern) = self.pattern_templates.get_mut(pattern_id) {
            // Update _pattern using exponential moving average
            let alpha = self.pattern_learning_rate;
            
            // Update spike times (simplified)
            if existing_pattern.relative_spike_times.len() == new_pattern.relative_spike_times.len() {
                for (existing_time, &new_time) in existing_pattern.relative_spike_times.iter_mut()
                    .zip(new_pattern.relative_spike_times.iter()) {
                    *existing_time = *existing_time * (T::one() - alpha) + new_time * alpha;
                }
            }
            
            existing_pattern.observation_count += 1;
            existing_pattern.weight = existing_pattern.weight * (T::one() - alpha) + 
                new_pattern.weight * alpha;
        }
        
        Ok(())
    }
    
    /// Recognize patterns in new spike train
    pub fn recognize_patterns(&self, spiketrain: &SpikeTrain<T>) -> Result<Vec> {
        let mut recognized_patterns = Vec::new();
        let window_size = num_traits::cast::cast(50.0).unwrap_or_else(|| T::zero());
        let step_size = num_traits::cast::cast(5.0).unwrap_or_else(|| T::zero());
        
        let mut window_start = T::zero();
        
        while window_start < spike_train.duration {
            let window_end = window_start + window_size;
            
            let window_spikes: Vec<T> = spike_train.spike_times.iter()
                .filter(|&&t| t >= window_start && t < window_end)
                .map(|&t| t - window_start)
                .collect();
            
            if !window_spikes.is_empty() {
                let test_pattern = SpikePattern {
                    pattern_id: 0,
                    relative_spike_times: window_spikes,
                    duration: window_size,
                    weight: T::one(),
                    observation_count: 1};
                
                // Find best matching pattern
                let mut best_match = (0, T::zero());
                for (i, template) in self.pattern_templates.iter().enumerate() {
                    let similarity = self.compute_pattern_similarity(&test_pattern, template);
                    if similarity > best_match.1 {
                        best_match = (i, similarity);
                    }
                }
                
                if best_match.1 > self.matching_threshold {
                    recognized_patterns.push((best_match.0, window_start, best_match.1));
                }
            }
            
            window_start = window_start + step_size;
        }
        
        Ok(recognized_patterns)
    }
    
    /// Get learned patterns
    pub fn get_patterns(&self) -> &[SpikePattern<T>] {
        &self.pattern_templates
    }
}
