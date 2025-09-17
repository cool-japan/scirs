//! Neuromorphic Computing for Spatial Data Processing
//!
//! This module implements brain-inspired computing paradigms for spatial algorithms,
//! leveraging spiking neural networks, memristive computing, and neuroplasticity
//! for energy-efficient adaptive spatial processing. These algorithms mimic biological
//! neural computation to achieve extreme energy efficiency and real-time adaptation.
//!
//! # Features
//!
//! - **Spiking Neural Networks (SNNs)** for spatial pattern recognition
//! - **Memristive crossbar arrays** for in-memory spatial computations
//! - **Spike-timing dependent plasticity (STDP)** for adaptive learning
//! - **Event-driven spatial processing** for real-time applications
//! - **Neuromorphic clustering** using competitive learning
//! - **Temporal coding** for multi-dimensional spatial data
//! - **Bio-inspired optimization** using neural adaptation mechanisms
//! - **Homeostatic plasticity** for stable learning
//! - **Neuromodulation** for context-dependent adaptation
//!
//! # Module Organization
//!
//! ## Core Components
//! - [`core::events`] - Spike event structures and utilities
//! - [`core::neurons`] - Spiking neuron models with various dynamics
//! - [`core::synapses`] - Synaptic models with STDP and metaplasticity
//!
//! ## Algorithm Implementations
//! - [`algorithms::spiking_clustering`] - SNN-based clustering
//! - [`algorithms::competitive_learning`] - Winner-take-all and homeostatic clustering
//! - [`algorithms::memristive_learning`] - Advanced memristive learning systems
//! - [`algorithms::processing`] - General neuromorphic processing pipeline
//!
//! # Examples
//!
//! ## Basic Spiking Neural Network Clustering
//! ```rust,ignore
//! use ndarray::Array2;
//! use scirs2_spatial::neuromorphic::SpikingNeuralClusterer;
//!
//! let points = Array2::from_shape_vec((4, 2), vec![
//!     0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0
//! ]).unwrap();
//!
//! let mut clusterer = SpikingNeuralClusterer::new(2)
//!     .with_spike_threshold(0.8)
//!     .with_stdp_learning(true)
//!     .with_lateral_inhibition(true);
//!
//! let (assignments, spike_events) = clusterer.fit(&points.view()).unwrap();
//! println!("Cluster assignments: {:?}", assignments);
//! println!("Recorded {} spike events", spike_events.len());
//! ```
//!
//! ## Competitive Learning with Homeostasis
//! ```rust,ignore
//! use ndarray::Array2;
//! use scirs2_spatial::neuromorphic::HomeostaticNeuralClusterer;
//!
//! let points = Array2::from_shape_vec((4, 2), vec![
//!     0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0
//! ]).unwrap();
//!
//! let mut clusterer = HomeostaticNeuralClusterer::new(2, 2)
//!     .with_homeostatic_params(0.1, 1000.0);
//!
//! let assignments = clusterer.fit(&points.view(), 50).unwrap();
//! println!("Homeostatic clustering results: {:?}", assignments);
//! ```
//!
//! ## Advanced Memristive Learning
//! ```rust,ignore
//! use ndarray::{Array1, Array2};
//! use scirs2_spatial::neuromorphic::{AdvancedMemristiveLearning, MemristiveDeviceType};
//!
//! let mut learning_system = AdvancedMemristiveLearning::new(
//!     4, 2, MemristiveDeviceType::TitaniumDioxide
//! ).with_forgetting_protection(true);
//!
//! let spatial_data = Array2::from_shape_vec((4, 4), vec![
//!     0.0, 0.0, 1.0, 1.0,
//!     1.0, 0.0, 0.0, 1.0,
//!     0.0, 1.0, 1.0, 0.0,
//!     1.0, 1.0, 0.0, 0.0
//! ]).unwrap();
//! let targets = Array1::from_vec(vec![0.0, 1.0, 1.0, 0.0]);
//!
//! # tokio_test::block_on(async {
//! let result = learning_system.train_spatial_data(
//!     &spatial_data.view(), &targets.view(), 50
//! ).await.unwrap();
//! println!("Training completed with final accuracy: {:.2}",
//!          result.training_metrics.last().unwrap().accuracy);
//! # });
//! ```
//!
//! ## Event-driven Neuromorphic Processing
//! ```rust,ignore
//! use ndarray::Array2;
//! use scirs2_spatial::neuromorphic::NeuromorphicProcessor;
//!
//! let points = Array2::from_shape_vec((3, 2), vec![
//!     0.0, 0.0, 1.0, 1.0, 2.0, 2.0
//! ]).unwrap();
//!
//! let mut processor = NeuromorphicProcessor::new()
//!     .with_memristive_crossbar(true)
//!     .with_temporal_coding(true)
//!     .with_crossbar_size(64, 64);
//!
//! // Encode spatial data as neuromorphic events
//! let events = processor.encode_spatial_events(&points.view()).unwrap();
//!
//! // Process events through neuromorphic pipeline
//! let processed_events = processor.process_events(&events).unwrap();
//! println!("Processed {} events", processed_events.len());
//! ```
//!
//! # Performance Considerations
//!
//! Neuromorphic algorithms are designed for:
//! - **Energy efficiency**: Event-driven processing reduces computation
//! - **Real-time adaptation**: Online learning without full retraining
//! - **Noise tolerance**: Biological inspiration provides robustness
//! - **Scalability**: Distributed processing capabilities
//!
//! # Biological Inspiration
//!
//! These algorithms draw inspiration from:
//! - **Synaptic plasticity**: Adaptive connection strengths
//! - **Homeostatic regulation**: Maintaining stable activity levels
//! - **Neuromodulation**: Context-dependent learning control
//! - **Memory consolidation**: Strengthening important patterns
//! - **Competitive dynamics**: Winner-take-all neural competition

pub mod algorithms;
pub mod core;

// Re-export core components for easier access
pub use core::events::{SpikeEvent, SpikeSequence};
pub use core::neurons::{AdaptiveSpikingNeuron, SpikingNeuron};
pub use core::synapses::{HomeostaticSynapse, MetaplasticSynapse, Synapse};

// Re-export main algorithm implementations
pub use algorithms::competitive_learning::{
    AdaptationScale, CompetitiveNeuralClusterer, HomeostaticNeuralClusterer, HomeostaticNeuron,
    LearningRateAdaptation, MetaplasticityController, MultiTimescaleAdaptation,
};
pub use algorithms::memristive_learning::{
    AdvancedMemristiveLearning, ConsolidationEvent, ConsolidationRules, ConsolidationType,
    ForgettingProtectionRules, HomeostaticMechanism, HomeostaticSystem, LearningHistory,
    LearningRateAdaptation as MemristiveLearningRateAdaptation, MemristiveCrossbar,
    MemristiveDeviceType, MetaplasticityRules, NeuromodulationEffects, NeuromodulationSystem,
    NeuromodulatorReleasePatterns, PerformanceMetrics, PlasticityEvent, PlasticityEventType,
    PlasticityLearningRates, PlasticityMechanism, PlasticityThresholds, PlasticityTimeConstants,
    PlasticityType, ThresholdAdaptation, TrainingResult,
};
pub use algorithms::processing::NeuromorphicProcessor;
pub use algorithms::spiking_clustering::{NetworkStats, SpikingNeuralClusterer};

/// Neuromorphic algorithm trait for unified interface
///
/// This trait provides a common interface for all neuromorphic algorithms,
/// enabling interchangeable use and consistent API across different approaches.
pub trait NeuromorphicAlgorithm<T> {
    /// Input data type
    type Input;
    /// Output data type  
    type Output;
    /// Error type
    type Error;

    /// Fit the algorithm to spatial data
    fn fit(&mut self, data: &Self::Input) -> Result<Self::Output, Self::Error>;

    /// Predict using the trained algorithm
    fn predict(&self, data: &Self::Input) -> Result<Self::Output, Self::Error>;

    /// Get algorithm parameters
    fn parameters(&self) -> T;

    /// Reset algorithm to initial state
    fn reset(&mut self);
}

/// Neuromorphic processing capabilities
///
/// Enumeration of different neuromorphic processing modes and capabilities
/// available in the system.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NeuromorphicCapability {
    /// Spike-timing dependent plasticity
    SpikePlasticity,
    /// Homeostatic regulation
    HomeostaticRegulation,
    /// Competitive learning dynamics
    CompetitiveDynamics,
    /// Memristive crossbar arrays
    MemristiveComputing,
    /// Event-driven processing
    EventDrivenProcessing,
    /// Temporal coding schemes
    TemporalCoding,
    /// Neuromodulation effects
    Neuromodulation,
    /// Memory consolidation
    MemoryConsolidation,
    /// Online learning
    OnlineLearning,
    /// Catastrophic forgetting protection
    ForgettingProtection,
}

/// Neuromorphic system configuration
///
/// Configuration structure for setting up neuromorphic systems with
/// specific capabilities and parameters.
#[derive(Debug, Clone)]
pub struct NeuromorphicConfig {
    /// Enabled capabilities
    pub capabilities: Vec<NeuromorphicCapability>,
    /// Number of neurons/clusters
    pub num_neurons: usize,
    /// Input dimensions
    pub input_dims: usize,
    /// Learning rate
    pub learning_rate: f64,
    /// Spike threshold
    pub spike_threshold: f64,
    /// Time step for simulation
    pub time_step: f64,
    /// Maximum simulation time
    pub max_time: f64,
    /// Enable debugging output
    pub debug_mode: bool,
}

impl Default for NeuromorphicConfig {
    fn default() -> Self {
        Self {
            capabilities: vec![
                NeuromorphicCapability::SpikePlasticity,
                NeuromorphicCapability::EventDrivenProcessing,
            ],
            num_neurons: 10,
            input_dims: 2,
            learning_rate: 0.01,
            spike_threshold: 1.0,
            time_step: 0.1,
            max_time: 100.0,
            debug_mode: false,
        }
    }
}

impl NeuromorphicConfig {
    /// Create new configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set number of neurons
    pub fn with_neurons(mut self, num_neurons: usize) -> Self {
        self.num_neurons = num_neurons;
        self
    }

    /// Set input dimensions
    pub fn with_input_dims(mut self, input_dims: usize) -> Self {
        self.input_dims = input_dims;
        self
    }

    /// Set learning rate
    pub fn with_learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Add capability
    pub fn with_capability(mut self, capability: NeuromorphicCapability) -> Self {
        if !self.capabilities.contains(&capability) {
            self.capabilities.push(capability);
        }
        self
    }

    /// Remove capability
    pub fn without_capability(mut self, capability: &NeuromorphicCapability) -> Self {
        self.capabilities.retain(|c| c != capability);
        self
    }

    /// Enable debug mode
    pub fn with_debug(mut self, debug: bool) -> Self {
        self.debug_mode = debug;
        self
    }

    /// Check if capability is enabled
    pub fn has_capability(&self, capability: &NeuromorphicCapability) -> bool {
        self.capabilities.contains(capability)
    }
}

/// Neuromorphic system factory
///
/// Factory for creating different types of neuromorphic systems based
/// on configuration and requirements.
pub struct NeuromorphicFactory;

impl NeuromorphicFactory {
    /// Create spiking neural network clusterer
    pub fn create_spiking_clusterer(config: &NeuromorphicConfig) -> SpikingNeuralClusterer {
        let mut clusterer = SpikingNeuralClusterer::new(config.num_neurons)
            .with_spike_threshold(config.spike_threshold)
            .with_time_step(config.time_step);

        if config.has_capability(&NeuromorphicCapability::SpikePlasticity) {
            clusterer = clusterer.with_stdp_learning(true);
        }

        if config.has_capability(&NeuromorphicCapability::CompetitiveDynamics) {
            clusterer = clusterer.with_lateral_inhibition(true);
        }

        clusterer
    }

    /// Create competitive neural clusterer
    pub fn create_competitive_clusterer(config: &NeuromorphicConfig) -> CompetitiveNeuralClusterer {
        CompetitiveNeuralClusterer::new(config.num_neurons, config.input_dims)
    }

    /// Create homeostatic neural clusterer
    pub fn create_homeostatic_clusterer(config: &NeuromorphicConfig) -> HomeostaticNeuralClusterer {
        let mut clusterer = HomeostaticNeuralClusterer::new(config.num_neurons, config.input_dims);

        if config.has_capability(&NeuromorphicCapability::HomeostaticRegulation) {
            clusterer = clusterer.with_homeostatic_params(0.1, 1000.0);
        }

        clusterer
    }

    /// Create advanced memristive learning system
    pub fn create_memristive_system(
        config: &NeuromorphicConfig,
        device_type: MemristiveDeviceType,
    ) -> AdvancedMemristiveLearning {
        let mut system =
            AdvancedMemristiveLearning::new(config.input_dims, config.num_neurons, device_type);

        if config.has_capability(&NeuromorphicCapability::ForgettingProtection) {
            system = system.with_forgetting_protection(true);
        }

        if config.has_capability(&NeuromorphicCapability::HomeostaticRegulation) {
            let target_rates = ndarray::Array1::from_elem(config.num_neurons, 0.1);
            system = system.with_homeostatic_regulation(target_rates);
        }

        system
    }

    /// Create neuromorphic processor
    pub fn create_processor(config: &NeuromorphicConfig) -> NeuromorphicProcessor {
        let mut processor = NeuromorphicProcessor::new();

        if config.has_capability(&NeuromorphicCapability::MemristiveComputing) {
            processor = processor.with_memristive_crossbar(true);
        }

        if config.has_capability(&NeuromorphicCapability::TemporalCoding) {
            processor = processor.with_temporal_coding(true);
        }

        processor
    }
}

/// Neuromorphic utilities
///
/// Utility functions for working with neuromorphic algorithms and data.
pub mod utils {
    use super::*;
    use crate::error::SpatialResult;
    use ndarray::ArrayView2;

    /// Convert spatial data to spike events
    ///
    /// Converts regular spatial data into spike events using rate coding,
    /// where higher values correspond to higher spike rates.
    pub fn spatial_to_spikes(
        data: &ArrayView2<f64>,
        time_window: f64,
        max_rate: f64,
    ) -> SpatialResult<Vec<SpikeEvent>> {
        let (n_points, n_dims) = data.dim();
        let mut events = Vec::new();

        for (point_idx, point) in data.outer_iter().enumerate() {
            for (dim, &value) in point.iter().enumerate() {
                // Normalize value to [0, 1] and scale to spike rate
                let normalized = (value + 10.0) / 20.0; // Assume data in [-10, 10]
                let spike_rate = normalized.clamp(0.0, 1.0) * max_rate;

                // Generate Poisson spike train
                let num_spikes = (spike_rate * time_window) as usize;
                for spike_idx in 0..num_spikes {
                    let timestamp = (spike_idx as f64) * (time_window / num_spikes as f64);
                    let event =
                        SpikeEvent::new(point_idx * n_dims + dim, timestamp, 1.0, point.to_vec());
                    events.push(event);
                }
            }
        }

        // Sort events by timestamp
        events.sort_by(|a, b| a.timestamp().partial_cmp(&b.timestamp()).unwrap());
        Ok(events)
    }

    /// Analyze spike patterns
    ///
    /// Analyzes spike timing patterns to extract information about
    /// spatial structure and temporal dynamics.
    pub fn analyze_spike_patterns(events: &[SpikeEvent]) -> SpikePatternAnalysis {
        if events.is_empty() {
            return SpikePatternAnalysis::default();
        }

        let total_events = events.len();
        let time_span = events.last().unwrap().timestamp() - events.first().unwrap().timestamp();
        let avg_rate = if time_span > 0.0 {
            total_events as f64 / time_span
        } else {
            0.0
        };

        // Calculate inter-spike intervals
        let mut intervals = Vec::new();
        for i in 1..events.len() {
            intervals.push(events[i].timestamp() - events[i - 1].timestamp());
        }

        let avg_interval = if !intervals.is_empty() {
            intervals.iter().sum::<f64>() / intervals.len() as f64
        } else {
            0.0
        };

        // Calculate coefficient of variation for regularity
        let interval_var = if intervals.len() > 1 {
            let mean = avg_interval;
            let variance =
                intervals.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / intervals.len() as f64;
            variance.sqrt() / mean.max(1e-10)
        } else {
            0.0
        };

        SpikePatternAnalysis {
            total_spikes: total_events,
            time_span,
            average_rate: avg_rate,
            average_interval: avg_interval,
            regularity: 1.0 / (1.0 + interval_var), // Higher = more regular
            unique_neurons: events
                .iter()
                .map(|e| e.neuron_id())
                .collect::<std::collections::HashSet<_>>()
                .len(),
        }
    }

    /// Spike pattern analysis results
    #[derive(Debug, Clone)]
    pub struct SpikePatternAnalysis {
        /// Total number of spikes
        pub total_spikes: usize,
        /// Total time span
        pub time_span: f64,
        /// Average firing rate
        pub average_rate: f64,
        /// Average inter-spike interval
        pub average_interval: f64,
        /// Regularity measure (0-1, higher = more regular)
        pub regularity: f64,
        /// Number of unique neurons
        pub unique_neurons: usize,
    }

    impl Default for SpikePatternAnalysis {
        fn default() -> Self {
            Self {
                total_spikes: 0,
                time_span: 0.0,
                average_rate: 0.0,
                average_interval: 0.0,
                regularity: 0.0,
                unique_neurons: 0,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_neuromorphic_config() {
        let config = NeuromorphicConfig::new()
            .with_neurons(5)
            .with_input_dims(3)
            .with_capability(NeuromorphicCapability::HomeostaticRegulation)
            .without_capability(&NeuromorphicCapability::SpikePlasticity);

        assert_eq!(config.num_neurons, 5);
        assert_eq!(config.input_dims, 3);
        assert!(config.has_capability(&NeuromorphicCapability::HomeostaticRegulation));
        assert!(!config.has_capability(&NeuromorphicCapability::SpikePlasticity));
    }

    #[test]
    fn test_neuromorphic_factory() {
        let config = NeuromorphicConfig::new()
            .with_neurons(3)
            .with_input_dims(2)
            .with_capability(NeuromorphicCapability::CompetitiveDynamics);

        let spiking_clusterer = NeuromorphicFactory::create_spiking_clusterer(&config);
        assert_eq!(spiking_clusterer.num_clusters(), 3);
        assert!(spiking_clusterer.is_lateral_inhibition_enabled());

        let competitive_clusterer = NeuromorphicFactory::create_competitive_clusterer(&config);
        assert_eq!(competitive_clusterer.num_clusters(), 3);

        let processor = NeuromorphicFactory::create_processor(&config);
        assert!(!processor.is_memristive_enabled()); // Not in capabilities
    }

    #[test]
    fn test_utils_spatial_to_spikes() {
        let data = Array2::from_shape_vec((2, 2), vec![0.0, 1.0, -1.0, 0.5]).unwrap();
        let events = utils::spatial_to_spikes(&data.view(), 1.0, 10.0).unwrap();

        // Should generate events for non-zero values
        assert!(!events.is_empty());

        // Events should be sorted by timestamp
        for i in 1..events.len() {
            assert!(events[i - 1].timestamp() <= events[i].timestamp());
        }
    }

    #[test]
    fn test_utils_spike_pattern_analysis() {
        let events = vec![
            SpikeEvent::new(0, 0.0, 1.0, vec![0.0, 0.0]),
            SpikeEvent::new(1, 1.0, 1.0, vec![1.0, 0.0]),
            SpikeEvent::new(0, 2.0, 1.0, vec![0.0, 1.0]),
            SpikeEvent::new(2, 3.0, 1.0, vec![1.0, 1.0]),
        ];

        let analysis = utils::analyze_spike_patterns(&events);
        assert_eq!(analysis.total_spikes, 4);
        assert_eq!(analysis.time_span, 3.0);
        assert!(analysis.average_rate > 0.0);
        assert_eq!(analysis.unique_neurons, 3);
    }

    #[test]
    fn test_empty_spike_analysis() {
        let events = Vec::new();
        let analysis = utils::analyze_spike_patterns(&events);
        assert_eq!(analysis.total_spikes, 0);
        assert_eq!(analysis.time_span, 0.0);
        assert_eq!(analysis.average_rate, 0.0);
    }
}
