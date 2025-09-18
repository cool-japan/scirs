//! Event-Driven Optimization Algorithms
//!
//! This module implements event-driven optimization algorithms that process
//! updates asynchronously based on neuromorphic events, designed for 
//! neuromorphic computing platforms with event-based architectures.

use super::{
    Spike, SpikeTrain, NeuromorphicMetrics, NeuromorphicEvent, EventPriority,
    STDPConfig, MembraneDynamicsConfig, PlasticityModel
};
use crate::error::{OptimError, Result};
use crate::optimizers::Optimizer;
use ndarray::{Array1, Array2, ArrayBase, Data, DataMut, Dimension};
use num_traits::Float;
use std::collections::{HashMap, VecDeque, BinaryHeap, HashSet};
use std::time::{Duration, Instant};
use std::sync::{Arc, Mutex, RwLock};
use std::cmp::Reverse;

/// Event types for neuromorphic computing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventType {
    /// Spike event from a neuron
    Spike,
    
    /// Synaptic weight update event
    WeightUpdate,
    
    /// Membrane potential threshold crossing
    ThresholdCrossing,
    
    /// Plasticity-triggered event
    PlasticityEvent,
    
    /// External stimulus event
    ExternalStimulus,
    
    /// Timer-based event
    TimerEvent,
    
    /// Error backpropagation event
    ErrorEvent,
    
    /// Homeostatic adaptation event
    HomeostaticEvent,
    
    /// Population synchronization event
    SynchronizationEvent,
    
    /// Energy budget event
    EnergyEvent,
}

/// Event-driven optimization configuration
#[derive(Debug, Clone)]
pub struct EventDrivenConfig<T: Float> {
    /// Maximum event queue size
    pub max_queue_size: usize,
    
    /// Event processing timeout (ms)
    pub processing_timeout: T,
    
    /// Enable event priority scheduling
    pub priority_scheduling: bool,
    
    /// Event filtering threshold
    pub event_threshold: T,
    
    /// Enable event batching
    pub event_batching: bool,
    
    /// Batch size for event processing
    pub batch_size: usize,
    
    /// Enable temporal event correlation
    pub temporal_correlation: bool,
    
    /// Temporal correlation window (ms)
    pub correlation_window: T,
    
    /// Enable adaptive event handling
    pub adaptive_handling: bool,
    
    /// Event rate limits (events/second)
    pub rate_limits: HashMap<EventType, T>,
    
    /// Enable event compression
    pub event_compression: bool,
    
    /// Compression algorithm
    pub compression_algorithm: EventCompressionAlgorithm,
    
    /// Enable distributed event processing
    pub distributed_processing: bool,
    
    /// Load balancing strategy
    pub load_balancing: LoadBalancingStrategy,
}

/// Event compression algorithms
#[derive(Debug, Clone, Copy)]
pub enum EventCompressionAlgorithm {
    /// No compression
    None,
    
    /// Delta encoding
    DeltaEncoding,
    
    /// Huffman encoding
    HuffmanEncoding,
    
    /// Run-length encoding
    RunLengthEncoding,
    
    /// Sparse encoding
    SparseEncoding,
    
    /// Predictive encoding
    PredictiveEncoding,
}

/// Load balancing strategies for distributed event processing
#[derive(Debug, Clone, Copy)]
pub enum LoadBalancingStrategy {
    /// Round-robin distribution
    RoundRobin,
    
    /// Event type-based partitioning
    TypeBased,
    
    /// Load-aware distribution
    LoadAware,
    
    /// Locality-aware distribution
    LocalityAware,
    
    /// Dynamic load balancing
    Dynamic,
}

impl<T: Float> Default for EventDrivenConfig<T> {
    fn default() -> Self {
        let mut rate_limits = HashMap::new();
        rate_limits.insert(EventType::Spike, num_traits::cast::cast(1000.0).unwrap_or_else(|| T::zero()));
        rate_limits.insert(EventType::WeightUpdate, num_traits::cast::cast(100.0).unwrap_or_else(|| T::zero()));
        rate_limits.insert(EventType::PlasticityEvent, num_traits::cast::cast(50.0).unwrap_or_else(|| T::zero()));
        
        Self {
            max_queue_size: 10000,
            processing_timeout: num_traits::cast::cast(1.0).unwrap_or_else(|| T::zero()),
            priority_scheduling: true,
            event_threshold: num_traits::cast::cast(0.001).unwrap_or_else(|| T::zero()),
            event_batching: true,
            batch_size: 32,
            temporal_correlation: true,
            correlation_window: num_traits::cast::cast(10.0).unwrap_or_else(|| T::zero()),
            adaptive_handling: true,
            rate_limits,
            event_compression: false,
            compression_algorithm: EventCompressionAlgorithm::None,
            distributed_processing: false,
            load_balancing: LoadBalancingStrategy::RoundRobin,
        }
    }
}

/// Priority queue entry for event scheduling
#[derive(Debug, Clone)]
struct PriorityEventEntry<T: Float> {
    event: NeuromorphicEvent<T>,
    insertion_time: Instant,
}

impl<T: Float> PartialEq for PriorityEventEntry<T> {
    fn eq(&self, other: &Self) -> bool {
        self.event.priority == other.event.priority
    }
}

impl<T: Float> Eq for PriorityEventEntry<T> {}

impl<T: Float> PartialOrd for PriorityEventEntry<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: Float> Ord for PriorityEventEntry<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Higher priority events come first (reverse order)
        other.event.priority.cmp(&self.event.priority)
            .then_with(|| self.insertion_time.cmp(&other.insertion_time))
    }
}

/// Event-driven optimizer
pub struct EventDrivenOptimizer<T: Float> {
    /// Configuration
    config: EventDrivenConfig<T>,
    
    /// STDP configuration
    stdp_config: STDPConfig<T>,
    
    /// Membrane dynamics configuration
    membrane_config: MembraneDynamicsConfig<T>,
    
    /// Event queue with priority scheduling
    event_queue: BinaryHeap<PriorityEventEntry<T>>,
    
    /// Event processing statistics
    event_stats: HashMap<EventType, EventStatistics<T>>,
    
    /// Current system state
    system_state: SystemState<T>,
    
    /// Event handlers
    event_handlers: HashMap<EventType, Box<dyn EventHandler<T>>>,
    
    /// Temporal correlation tracker
    correlation_tracker: TemporalCorrelationTracker<T>,
    
    /// Event rate limiter
    rate_limiter: EventRateLimiter<T>,
    
    /// Performance metrics
    metrics: NeuromorphicMetrics<T>,
    
    /// Distributed processing coordinator
    distributed_coordinator: Option<DistributedEventCoordinator<T>>,
    
    /// Event compression engine
    compression_engine: EventCompressionEngine<T>,
    
    /// Adaptive handler
    adaptive_handler: AdaptiveEventHandler<T>,
}

/// Event processing statistics
#[derive(Debug, Clone)]
struct EventStatistics<T: Float> {
    /// Total events processed
    total_processed: usize,
    
    /// Average processing time (ms)
    avg_processing_time: T,
    
    /// Event rate (events/second)
    event_rate: T,
    
    /// Queue wait time (ms)
    avg_queue_wait_time: T,
    
    /// Error count
    error_count: usize,
    
    /// Last update time
    last_update: Instant,
}

/// System state for event-driven optimization
#[derive(Debug, Clone)]
struct SystemState<T: Float> {
    /// Current membrane potentials
    membrane_potentials: Array1<T>,
    
    /// Synaptic weights
    synaptic_weights: Array2<T>,
    
    /// Last spike times
    last_spike_times: Array1<T>,
    
    /// Refractory states
    refractory_until: Array1<T>,
    
    /// Current simulation time
    current_time: T,
    
    /// Active neurons
    active_neurons: HashSet<usize>,
    
    /// Pending weight updates
    pending_updates: HashMap<(usize, usize), T>,
}

/// Event handler trait
trait EventHandler<T: Float>: Send + Sync {
    fn handle_event(&mut self, event: &NeuromorphicEvent<T>, state: &mut SystemState<T>) -> Result<()>;
    fn can_handle(&self, eventtype: EventType) -> bool;
}

/// Spike event handler
struct SpikeEventHandler<T: Float> {
    stdp_config: STDPConfig<T>,
    membrane_config: MembraneDynamicsConfig<T>,
}

impl<T: Float + Send + Sync> EventHandler<T> for SpikeEventHandler<T> {
    fn handle_event(&mut self, event: &NeuromorphicEvent<T>, state: &mut SystemState<T>) -> Result<()> {
        let neuron_id = event.source_neuron;
        
        // Generate spike
        if neuron_id < state.membrane_potentials.len() {
            // Reset membrane potential
            state.membrane_potentials[neuron_id] = self.membrane_config.reset_potential;
            
            // Set refractory period
            state.refractory_until[neuron_id] = state.current_time + self.membrane_config.refractory_period;
            
            // Update last spike time
            state.last_spike_times[neuron_id] = state.current_time;
            
            // Add to active neurons
            state.active_neurons.insert(neuron_id);
            
            // Trigger STDP updates for connected synapses
            self.trigger_stdp_updates(neuron_id, state)?;
        }
        
        Ok(())
    }
    
    fn can_handle(&self, eventtype: EventType) -> bool {
        event_type == EventType::Spike
    }
}

impl<T: Float + Send + Sync> SpikeEventHandler<T> {
    fn trigger_stdp_updates(&self, postneuron: usize, state: &mut SystemState<T>) -> Result<()> {
        // Check all presynaptic connections
        for pre_neuron in 0..state.last_spike_times.len() {
            if pre_neuron != post_neuron {
                let pre_spike_time = state.last_spike_times[pre_neuron];
                
                if pre_spike_time > num_traits::cast::cast(-1000.0).unwrap_or_else(|| T::zero()) {
                    let dt = state.current_time - pre_spike_time;
                    let weight_change = self.compute_stdp_weight_change(dt);
                    
                    // Add to pending updates
                    state.pending_updates.insert((pre_neuron, post_neuron), weight_change);
                }
            }
        }
        
        Ok(())
    }
    
    fn compute_stdp_weight_change(&self, dt: T) -> T {
        if dt > T::zero() {
            // Post-before-pre: LTP
            let exp_arg = -dt / self.stdp_config.tau_pot;
            self.stdp_config.learning_rate_pot * exp_arg.exp()
        } else {
            // Pre-before-post: LTD
            let exp_arg = dt / self.stdp_config.tau_dep;
            -self.stdp_config.learning_rate_dep * exp_arg.exp()
        }
    }
}

/// Weight update event handler
struct WeightUpdateEventHandler<T: Float> {
    stdp_config: STDPConfig<T>,
}

impl<T: Float + Send + Sync> EventHandler<T> for WeightUpdateEventHandler<T> {
    fn handle_event(&mut self, event: &NeuromorphicEvent<T>, state: &mut SystemState<T>) -> Result<()> {
        let source = event.source_neuron;
        
        if let Some(target) = event.target_neuron {
            if source < state.synapticweights.nrows() && target < state.synapticweights.ncols() {
                // Apply weight update
                let current_weight = state.synaptic_weights[[source, target]];
                let new_weight = (current_weight + event.value)
                    .max(self.stdp_config.weight_min)
                    .min(self.stdp_config.weight_max);
                
                state.synaptic_weights[[source, target]] = new_weight;
            }
        }
        
        Ok(())
    }
    
    fn can_handle(&self, eventtype: EventType) -> bool {
        event_type == EventType::WeightUpdate
    }
}

/// Temporal correlation tracker
struct TemporalCorrelationTracker<T: Float> {
    correlation_window: T,
    event_history: VecDeque<(T, EventType, usize)>,
    correlation_patterns: HashMap<(EventType, EventType), T>,
}

impl<T: Float + Send + Sync> TemporalCorrelationTracker<T> {
    fn new(_correlationwindow: T) -> Self {
        Self {
            correlation_window,
            event_history: VecDeque::new(),
            correlation_patterns: HashMap::new(),
        }
    }
    
    fn add_event(&mut self, time: T, event_type: EventType, neuronid: usize) {
        // Add new event
        self.event_history.push_back((time, event_type, neuron_id));
        
        // Remove old events outside correlation window
        while let Some(&(old_time_)) = self.event_history.front() {
            if time - old_time > self.correlation_window {
                self.event_history.pop_front();
            } else {
                break;
            }
        }
        
        // Update correlation patterns
        self.update_correlations(time, event_type);
    }
    
    fn update_correlations(&mut self, current_time: T, currentevent: EventType) {
        for &(event_time, event_type_) in &self.event_history {
            if current_time - event_time <= self.correlation_window {
                let correlation_key = (event_type, current_event);
                let time_diff = current_time - event_time;
                let correlation_strength = (-time_diff / self.correlation_window).exp();
                
                *self.correlation_patterns.entry(correlation_key).or_insert(T::zero()) += correlation_strength;
            }
        }
    }
    
    fn get_correlation(&self, event1: EventType, event2: EventType) -> T {
        self.correlation_patterns.get(&(event1, event2)).copied().unwrap_or(T::zero())
    }
}

/// Event rate limiter
struct EventRateLimiter<T: Float> {
    rate_limits: HashMap<EventType, T>,
    event_counts: HashMap<EventType, usize>,
    last_reset: Instant,
    reset_interval: Duration,
}

impl<T: Float + Send + Sync> EventRateLimiter<T> {
    fn new(_ratelimits: HashMap<EventType, T>) -> Self {
        Self {
            rate_limits,
            event_counts: HashMap::new(),
            last_reset: Instant::now(),
            reset_interval: Duration::from_secs(1),
        }
    }
    
    fn can_process(&mut self, eventtype: EventType) -> bool {
        // Reset counters if interval elapsed
        if self.last_reset.elapsed() >= self.reset_interval {
            self.event_counts.clear();
            self.last_reset = Instant::now();
        }
        
        if let Some(&limit) = self.rate_limits.get(&event_type) {
            let current_count = self.event_counts.get(&event_type).copied().unwrap_or(0);
            if num_traits::cast::cast(current_count).unwrap_or_else(|| T::zero()) < limit {
                *self.event_counts.entry(event_type).or_insert(0) += 1;
                true
            } else {
                false
            }
        } else {
            true
        }
    }
}

/// Event compression engine
struct EventCompressionEngine<T: Float> {
    algorithm: EventCompressionAlgorithm,
    compression_buffer: Vec<u8>,
    decompression_buffer: Vec<u8>,
}

impl<T: Float + Send + Sync> EventCompressionEngine<T> {
    fn new(algorithm: EventCompressionAlgorithm) -> Self {
        Self {
            algorithm,
            compression_buffer: Vec::new(),
            decompression_buffer: Vec::new(),
        }
    }
    
    fn compress_event(&mut self, event: &NeuromorphicEvent<T>) -> Result<Vec<u8>> {
        match self.algorithm {
            EventCompressionAlgorithm::None => {
                // No compression, serialize directly
                self.serialize_event(event)
            }
            EventCompressionAlgorithm::DeltaEncoding => {
                self.delta_encode_event(event)
            }
            EventCompressionAlgorithm::SparseEncoding => {
                self.sparse_encode_event(event)
            }
            _ => {
                // Fallback to no compression
                self.serialize_event(event)
            }
        }
    }
    
    fn serialize_event(&self, event: &NeuromorphicEvent<T>) -> Result<Vec<u8>> {
        // Simplified serialization
        let mut data = Vec::new();
        data.extend_from_slice(&(event.event_type as u8).to_le_bytes());
        data.extend_from_slice(&event.source_neuron.to_le_bytes());
        
        if let Some(target) = event.target_neuron {
            data.push(1);
            data.extend_from_slice(&target.to_le_bytes());
        } else {
            data.push(0);
        }
        
        Ok(data)
    }
    
    fn delta_encode_event(&mut self,
        event: &NeuromorphicEvent<T>) -> Result<Vec<u8>> {
        // Simplified delta encoding implementation
        Ok(vec![0u8; 16])
    }
    
    fn sparse_encode_event(&mut self,
        event: &NeuromorphicEvent<T>) -> Result<Vec<u8>> {
        // Simplified sparse encoding implementation
        Ok(vec![0u8; 8])
    }
}

/// Adaptive event handler
struct AdaptiveEventHandler<T: Float> {
    adaptation_rate: T,
    performance_history: VecDeque<T>,
    current_strategy: AdaptationStrategy,
}

#[derive(Debug, Clone, Copy)]
enum AdaptationStrategy {
    Conservative,
    Balanced,
    Aggressive,
}

impl<T: Float + Send + Sync> AdaptiveEventHandler<T> {
    fn new() -> Self {
        Self {
            adaptation_rate: num_traits::cast::cast(0.1).unwrap_or_else(|| T::zero()),
            performance_history: VecDeque::new(),
            current_strategy: AdaptationStrategy::Balanced,
        }
    }
    
    fn adapt_processing(&mut self, currentperformance: T) {
        self.performance_history.push_back(current_performance);
        
        if self.performance_history.len() > 100 {
            self.performance_history.pop_front();
        }
        
        if self.performance_history.len() >= 10 {
            let recent_avg = self.performance_history.iter().rev().take(10).cloned().sum::<T>() / num_traits::cast::cast(10).unwrap_or_else(|| T::zero());
            let older_avg = if self.performance_history.len() >= 20 {
                self.performance_history.iter().rev().skip(10).take(10).cloned().sum::<T>() / num_traits::cast::cast(10).unwrap_or_else(|| T::zero())
            } else {
                recent_avg
            };
            
            let performance_change = recent_avg - older_avg;
            
            self.current_strategy = if performance_change > num_traits::cast::cast(0.1).unwrap_or_else(|| T::zero()) {
                AdaptationStrategy::Aggressive
            } else if performance_change < num_traits::cast::cast(-0.1).unwrap_or_else(|| T::zero()) {
                AdaptationStrategy::Conservative
            } else {
                AdaptationStrategy::Balanced
            };
        }
    }
    
    fn get_adaptation_factor(&self) -> T {
        match self.current_strategy {
            AdaptationStrategy::Conservative => num_traits::cast::cast(0.5).unwrap_or_else(|| T::zero()),
            AdaptationStrategy::Balanced => T::one(),
            AdaptationStrategy::Aggressive => num_traits::cast::cast(1.5).unwrap_or_else(|| T::zero()),
        }
    }
}

/// Distributed event coordinator
struct DistributedEventCoordinator<T: Float> {
    load_balancing: LoadBalancingStrategy,
    worker_loads: HashMap<usize, T>,
    current_worker: usize,
    total_workers: usize,
}

impl<T: Float + Send + Sync> DistributedEventCoordinator<T> {
    fn new(_strategy: LoadBalancingStrategy, numworkers: usize) -> Self {
        Self {
            load_balancing: strategy,
            worker_loads: HashMap::new(),
            current_worker: 0,
            total_workers: num_workers,
        }
    }
    
    fn assign_worker(&mut self, event: &NeuromorphicEvent<T>) -> usize {
        match self.load_balancing {
            LoadBalancingStrategy::RoundRobin => {
                let worker = self.current_worker;
                self.current_worker = (self.current_worker + 1) % self.total_workers;
                worker
            }
            LoadBalancingStrategy::TypeBased => {
                // Hash event type to worker
                (event.event_type as usize) % self.total_workers
            }
            LoadBalancingStrategy::LoadAware => {
                // Find worker with minimum load
                self.worker_loads.iter()
                    .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(&worker_id_)| worker_id)
                    .unwrap_or(0)
            }
            _ => 0,
        }
    }
    
    fn update_worker_load(&mut self, workerid: usize, load: T) {
        self.worker_loads.insert(worker_id, load);
    }
}

impl<T: Float + Send + Sync> EventDrivenOptimizer<T> {
    /// Create a new event-driven optimizer
    pub fn new(
        config: EventDrivenConfig<T>,
        stdp_config: STDPConfig<T>,
        membrane_config: MembraneDynamicsConfig<T>,
        num_neurons: usize,
    ) -> Self {
        let mut optimizer = Self {
            _config: config.clone(),
            stdp_config: stdp_config.clone(),
            membrane_config: membrane_config.clone(),
            event_queue: BinaryHeap::new(),
            event_stats: HashMap::new(),
            system_state: SystemState {
                membrane_potentials: Array1::from_elem(num_neurons, membrane_config.resting_potential),
                synaptic_weights: Array2::ones((num_neurons, num_neurons)) * num_traits::cast::cast(0.1).unwrap_or_else(|| T::zero()),
                last_spike_times: Array1::from_elem(num_neurons, num_traits::cast::cast(-1000.0).unwrap_or_else(|| T::zero())),
                refractory_until: Array1::zeros(num_neurons),
                current_time: T::zero(),
                active_neurons: HashSet::new(),
                pending_updates: HashMap::new(),
            },
            event_handlers: HashMap::new(),
            correlation_tracker: TemporalCorrelationTracker::new(_config.correlation_window),
            rate_limiter: EventRateLimiter::new(_config.rate_limits.clone()),
            metrics: NeuromorphicMetrics::default(),
            distributed_coordinator: if config.distributed_processing {
                Some(DistributedEventCoordinator::new(_config.load_balancing, 4))
            } else {
                None
            },
            compression_engine: EventCompressionEngine::new(_config.compression_algorithm),
            adaptive_handler: AdaptiveEventHandler::new(),
        };
        
        // Register default event handlers
        optimizer.register_default_handlers();
        
        optimizer
    }
    
    /// Register default event handlers
    fn register_default_handlers(&mut self) {
        let spike_handler = Box::new(SpikeEventHandler {
            stdp_config: self.stdp_config.clone(),
            membrane_config: self.membrane_config.clone(),
        });
        
        let weight_handler = Box::new(WeightUpdateEventHandler {
            stdp_config: self.stdp_config.clone(),
        });
        
        self.event_handlers.insert(EventType::Spike, spike_handler);
        self.event_handlers.insert(EventType::WeightUpdate, weight_handler);
    }
    
    /// Add event to the processing queue
    pub fn enqueue_event(&mut self, event: NeuromorphicEvent<T>) -> Result<()> {
        // Check rate limits
        if !self.rate_limiter.can_process(event.event_type) {
            return Err(OptimError::InvalidConfig("Rate limit exceeded".to_string()));
        }
        
        // Check queue capacity
        if self.event_queue.len() >= self.config.max_queue_size {
            return Err(OptimError::InvalidConfig("Event queue full".to_string()));
        }
        
        let entry = PriorityEventEntry {
            event,
            insertion_time: Instant::now(),
        };
        
        self.event_queue.push(entry);
        
        // Update correlation tracking
        if self.config.temporal_correlation {
            self.correlation_tracker.add_event(
                event.time,
                event.event_type,
                event.source_neuron,
            );
        }
        
        Ok(())
    }
    
    /// Process events from the queue
    pub fn process_events(&mut self) -> Result<usize> {
        let mut processed_count = 0;
        let start_time = Instant::now();
        let timeout = Duration::from_millis(
            self.config.processing_timeout.to_u64().unwrap_or(1000)
        );
        
        while !self.event_queue.is_empty() && start_time.elapsed() < timeout {
            if self.config.event_batching {
                let batch_size = self.config.batch_size.min(self.event_queue.len());
                processed_count += self.process_event_batch(batch_size)?;
            } else {
                if let Some(entry) = self.event_queue.pop() {
                    self.process_single_event(&entry.event)?;
                    processed_count += 1;
                }
            }
        }
        
        // Apply pending weight updates
        self.apply_pending_updates()?;
        
        // Update adaptive processing
        let processing_rate = num_traits::cast::cast(processed_count).unwrap_or_else(|| T::zero()) / 
            T::from(start_time.elapsed().as_millis()).unwrap();
        self.adaptive_handler.adapt_processing(processing_rate);
        
        Ok(processed_count)
    }
    
    /// Process a batch of events
    fn process_event_batch(&mut self, batchsize: usize) -> Result<usize> {
        let mut batch_events = Vec::with_capacity(batch_size);
        
        // Collect batch events
        for _ in 0..batch_size {
            if let Some(entry) = self.event_queue.pop() {
                batch_events.push(entry.event);
            } else {
                break;
            }
        }
        
        // Process batch
        for event in &batch_events {
            self.process_single_event(event)?;
        }
        
        Ok(batch_events.len())
    }
    
    /// Process a single event
    fn process_single_event(&mut self, event: &NeuromorphicEvent<T>) -> Result<()> {
        let start_time = Instant::now();
        
        // Find appropriate handler
        if let Some(handler) = self.event_handlers.get_mut(&event.event_type) {
            handler.handle_event(event, &mut self.system_state)?;
        } else {
            // Default handling
            self.default_event_handling(event)?;
        }
        
        // Update statistics
        let processing_time = start_time.elapsed().as_nanos() as f64 / 1_000_000.0;
        self.update_event_statistics(event.event_type, num_traits::cast::cast(processing_time).unwrap_or_else(|| T::zero()));
        
        // Update energy consumption
        self.metrics.energy_consumption = self.metrics.energy_consumption + event.energy_cost;
        
        Ok(())
    }
    
    /// Default event handling
    fn default_event_handling(&mut self, event: &NeuromorphicEvent<T>) -> Result<()> {
        match event.event_type {
            EventType::ExternalStimulus => {
                // Apply external stimulus to neuron
                if event.source_neuron < self.system_state.membrane_potentials.len() {
                    self.system_state.membrane_potentials[event.source_neuron] = 
                        self.system_state.membrane_potentials[event.source_neuron] + event.value;
                }
            }
            EventType::TimerEvent => {
                // Update system time
                self.system_state.current_time = event.time;
            }
            _ => {
                // Ignore unknown events
            }
        }
        
        Ok(())
    }
    
    /// Apply pending weight updates
    fn apply_pending_updates(&mut self) -> Result<()> {
        for ((pre, post), weight_change) in self.system_state.pending_updates.drain() {
            if pre < self.system_state.synapticweights.nrows() && 
               post < self.system_state.synapticweights.ncols() {
                let current_weight = self.system_state.synaptic_weights[[pre, post]];
                let new_weight = (current_weight + weight_change)
                    .max(self.stdp_config.weight_min)
                    .min(self.stdp_config.weight_max);
                
                self.system_state.synaptic_weights[[pre, post]] = new_weight;
            }
        }
        
        Ok(())
    }
    
    /// Update event processing statistics
    fn update_event_statistics(&mut self, event_type: EventType, processingtime: T) {
        let stats = self.event_stats.entry(event_type).or_insert_with(|| EventStatistics {
            total_processed: 0,
            avg_processing_time: T::zero(),
            event_rate: T::zero(),
            avg_queue_wait_time: T::zero(),
            error_count: 0,
            last_update: Instant::now(),
        });
        
        stats.total_processed += 1;
        
        // Update average processing _time using exponential moving average
        let alpha = num_traits::cast::cast(0.1).unwrap_or_else(|| T::zero());
        stats.avg_processing_time = stats.avg_processing_time * (T::one() - alpha) + processing_time * alpha;
        
        // Update event rate
        let time_since_last = stats.last_update.elapsed().as_secs_f64();
        if time_since_last > 0.0 {
            let current_rate = T::one() / num_traits::cast::cast(time_since_last).unwrap_or_else(|| T::zero());
            stats.event_rate = stats.event_rate * (T::one() - alpha) + current_rate * alpha;
        }
        
        stats.last_update = Instant::now();
    }
    
    /// Get event processing statistics
    pub fn get_event_statistics(&self) -> &HashMap<EventType, EventStatistics<T>> {
        &self.event_stats
    }
    
    /// Get current system state
    pub fn get_system_state(&self) -> &SystemState<T> {
        &self.system_state
    }
    
    /// Get current metrics
    pub fn get_metrics(&self) -> &NeuromorphicMetrics<T> {
        &self.metrics
    }
    
    /// Clear event queue
    pub fn clear_event_queue(&mut self) {
        self.event_queue.clear();
    }
    
    /// Get queue size
    pub fn get_queue_size(&self) -> usize {
        self.event_queue.len()
    }
    
    /// Enable distributed processing
    pub fn enable_distributed_processing(&mut self, numworkers: usize) {
        self.distributed_coordinator = Some(DistributedEventCoordinator::new(
            self.config.load_balancing,
            num_workers,
        ));
        self.config.distributed_processing = true;
    }
    
    /// Disable distributed processing
    pub fn disable_distributed_processing(&mut self) {
        self.distributed_coordinator = None;
        self.config.distributed_processing = false;
    }
}

impl<T: Float> Default for EventStatistics<T> {
    fn default() -> Self {
        Self {
            total_processed: 0,
            avg_processing_time: T::zero(),
            event_rate: T::zero(),
            avg_queue_wait_time: T::zero(),
            error_count: 0,
            last_update: Instant::now(),
        }
    }
}
