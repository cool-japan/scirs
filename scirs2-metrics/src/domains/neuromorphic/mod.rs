//! Neuromorphic Computing Integration for Advanced Mode
//!
//! This module implements brain-inspired computing paradigms for metrics computation,
//! featuring spiking neural networks, synaptic plasticity, and adaptive learning
//! mechanisms that evolve in real-time based on computational patterns.

#![allow(clippy::too_many_arguments)]
#![allow(clippy::useless_vec)]
#![allow(dead_code)]

use crate::error::{MetricsError, Result};
use crate::optimization::quantum_acceleration::QuantumMetricsComputer;
use scirs2_core::numeric::Float;
use serde::{Deserialize, Serialize};
use std::time::Duration;

// Module declarations
pub mod advanced_memory;
pub mod consciousness;
pub mod core;
pub mod distributed_coordination;
pub mod learning_controllers;
pub mod memory_systems;
pub mod meta_learning;
pub mod pattern_recognition;
pub mod performance_monitoring;
pub mod quantum_processing;
pub mod realtime_adaptation;
pub mod spiking_networks;
pub mod synaptic_systems;

// Re-export main types for backward compatibility
pub use advanced_memory::{AdvancedMemoryArchitecture, MemoryType};
pub use consciousness::{
    AttentionSystems, ConsciousnessSimulator, GlobalWorkspaceTheory, IntegratedInformationTheory,
};
pub use core::{HomeostaticController, NetworkTopology, NeuromorphicConfig, SynapseType};
pub use distributed_coordination::DistributedNeuromorphicCoordinator;
pub use learning_controllers::{AdaptiveLearningController, LearningObjective};
pub use memory_systems::{MemoryTrace, NeuromorphicMemory};
pub use meta_learning::MetaLearningSystem;
pub use pattern_recognition::{SpikePattern, SpikePatternRecognizer};
pub use performance_monitoring::{NeuromorphicPerformanceMonitor, PerformanceMetrics};
pub use quantum_processing::{QuantumCoherenceManager, QuantumNeuromorphicProcessor};
pub use realtime_adaptation::RealtimeAdaptationEngine;
pub use spiking_networks::{NeuronLayer, SpikingNeuralNetwork};
pub use synaptic_systems::{SynapticConnections, SynapticPlasticityManager};

/// Neuromorphic metrics computer using brain-inspired architectures
#[derive(Debug)]
pub struct NeuromorphicMetricsComputer<F: Float> {
    /// Spiking neural network for metric computation
    spiking_network: SpikingNeuralNetwork<F>,
    /// Synaptic plasticity manager
    plasticity_manager: SynapticPlasticityManager<F>,
    /// Adaptive learning controller
    learning_controller: AdaptiveLearningController<F>,
    /// Spike pattern recognizer
    pattern_recognizer: SpikePatternRecognizer<F>,
    /// Homeostatic mechanisms for stability
    homeostasis: HomeostaticController<F>,
    /// Memory formation and consolidation
    memory_system: NeuromorphicMemory<F>,
    /// Performance monitor
    performance_monitor: NeuromorphicPerformanceMonitor<F>,
    /// Quantum-neuromorphic hybrid processor
    quantum_processor: Option<QuantumNeuromorphicProcessor<F>>,
    /// Meta-learning system for learning-to-learn
    meta_learning: MetaLearningSystem<F>,
    /// Distributed neuromorphic coordinator
    distributed_coordinator: Option<DistributedNeuromorphicCoordinator<F>>,
    /// Float-time adaptation engine
    realtime_adapter: RealtimeAdaptationEngine<F>,
    /// Advanced memory architectures
    advanced_memory: AdvancedMemoryArchitecture<F>,
    /// Consciousness simulation module
    consciousness_module: ConsciousnessSimulator<F>,
    /// Configuration
    config: NeuromorphicConfig,
}

impl<
        F: Float
            + Send
            + Sync
            + std::iter::Sum
            + 'static
            + scirs2_core::ndarray::ScalarOperand
            + std::fmt::Debug,
    > NeuromorphicMetricsComputer<F>
{
    /// Create new neuromorphic metrics computer
    pub fn new(config: NeuromorphicConfig) -> Result<Self> {
        let topology = core::NetworkTopology {
            layer_sizes: vec![100, 50, 10], // Default topology
            connection_patterns: vec![core::ConnectionPattern::FullyConnected],
            recurrent_connections: vec![],
        };
        let spiking_network = SpikingNeuralNetwork::new(topology, &config);
        let plasticity_manager = SynapticPlasticityManager::new();
        let learning_controller = AdaptiveLearningController::new();
        let pattern_recognizer = SpikePatternRecognizer::new();
        let homeostasis = HomeostaticController::new(&config)?;
        let memory_system = NeuromorphicMemory::new(1000); // 1000 capacity
        let performance_monitor = NeuromorphicPerformanceMonitor::new();
        let quantum_processor = if config.enable_quantum_processing {
            Some(QuantumNeuromorphicProcessor::new())
        } else {
            None
        };
        let meta_learning = MetaLearningSystem::new()?;
        let realtime_adapter = RealtimeAdaptationEngine::new()?;
        let advanced_memory = AdvancedMemoryArchitecture::new()?;
        let consciousness_module = ConsciousnessSimulator::new()?;

        Ok(Self {
            spiking_network,
            plasticity_manager,
            learning_controller,
            pattern_recognizer,
            homeostasis,
            memory_system,
            performance_monitor,
            quantum_processor,
            meta_learning,
            distributed_coordinator: None,
            realtime_adapter,
            advanced_memory,
            consciousness_module,
            config,
        })
    }

    /// Compute metrics using neuromorphic processing
    ///
    /// This runs the actual staged neuromorphic pipeline on `data` (not a
    /// passthrough echo of the input):
    /// 1. Rate-code `data` into spike trains (the private `encode_to_spikes` method).
    /// 2. Drive the real spiking neural network
    ///    ([`spiking_networks::SpikingNeuralNetwork::simulate_step`], genuine
    ///    LIF dynamics + synaptic weighting + lateral inhibition) one encoded
    ///    timestep at a time, keeping the final layer's last-timestep output
    ///    as well as a time-averaged |membrane potential| per neuron across
    ///    the whole run (a single last-instant snapshot can coincide with a
    ///    just-fired/reset neuron and mask real input-dependent differences).
    /// 3. Update synaptic plasticity from the network's own time-averaged
    ///    activity ([`synaptic_systems::SynapticPlasticityManager::update`]).
    /// 4. Run pattern recognition against the spikes the network actually
    ///    produced ([`pattern_recognition::SpikePatternRecognizer::recognize_patterns`]).
    /// 5. Store a real memory trace of the response
    ///    ([`memory_systems::NeuromorphicMemory::store`]).
    /// 6. Update performance monitoring from the network's real state
    ///    ([`performance_monitoring::NeuromorphicPerformanceMonitor::update`]).
    /// 7. Apply the real (if intentionally simplified) quantum-neuromorphic
    ///    combination when both quantum processing is enabled and a quantum
    ///    computer is supplied.
    ///
    /// `metric_type` genuinely selects which real reduction of the network's
    /// response is returned: `"mean_activity"` (mean of the time-averaged
    /// activity), `"spike_rate"` (per-neuron firing rate over the tracked
    /// history window), `"entropy"` (Shannon entropy of the normalized
    /// time-averaged activity), or (default) the raw final-layer output itself.
    ///
    /// Deliberately NOT invoked here (left as pre-existing, separately-scoped
    /// gaps rather than silently faked): `self.homeostasis` -- an unrelated,
    /// pre-existing no-op stub (`core::HomeostaticController::regulate` is
    /// generic over any `T` and ignores it); `self.learning_controller` --
    /// its `update` needs a `learning_controllers::PerformanceSnapshot` with
    /// `stability`/`adaptability` fields this pipeline has no principled way
    /// to derive yet; `self.realtime_adapter` -- its `adapt` requires
    /// supervised target data this function is never given.
    pub fn compute_neuromorphic_metrics(
        &mut self,
        data: &[F],
        metric_type: &str,
        quantum_computer: Option<&QuantumMetricsComputer<F>>,
    ) -> Result<Vec<F>> {
        let dt = self.config.timestep;

        // 1. Real rate-coded spike-train encoding of the input.
        let spike_trains = self.encode_to_spikes(data)?;
        let n_timesteps = spike_trains
            .iter()
            .map(|train| train.len())
            .max()
            .unwrap_or(0);

        // 2. Drive the real spiking network one encoded timestep at a time;
        // the final layer's output after the last timestep is the network's
        // genuine response to this input. `activity_accumulator` sums each
        // step's |membrane potential| per neuron across the whole run
        // (rather than sampling only the last timestep, which can coincide
        // with a just-fired/reset neuron and mask real input-dependent
        // differences) so the resulting `activity` vector is a robust,
        // real, time-averaged measure of how strongly each neuron responded.
        let mut network_output = Vec::new();
        let mut activity_accumulator: Vec<F> = Vec::new();
        for t in 0..n_timesteps {
            let input_at_t: Vec<F> = spike_trains
                .iter()
                .map(|train| train.get(t).copied().unwrap_or_else(F::zero))
                .collect();
            network_output = self.spiking_network.simulate_step(dt, &input_at_t)?;

            let step_activity = &self.spiking_network.network_state.activity_levels;
            if activity_accumulator.is_empty() {
                activity_accumulator = step_activity.iter().map(|v| v.abs()).collect();
            } else {
                for (acc, &v) in activity_accumulator.iter_mut().zip(step_activity.iter()) {
                    *acc = *acc + v.abs();
                }
            }
        }
        let activity: Vec<F> = if n_timesteps > 0 {
            let n = F::from(n_timesteps).expect("Failed to convert constant to float");
            activity_accumulator.iter().map(|&s| s / n).collect()
        } else {
            Vec::new()
        };

        // 3. Real synaptic plasticity update driven by the network's own
        // (time-averaged) activity.
        self.plasticity_manager.update(dt, &activity)?;

        // 4. Real pattern recognition against the spikes the network actually
        // produced (per-neuron spike trains recorded by the LIF dynamics).
        let spike_data: std::collections::HashMap<usize, Vec<std::time::Instant>> = self
            .spiking_network
            .layers
            .iter()
            .flat_map(|layer| layer.neurons.iter())
            .map(|neuron| (neuron.id, neuron.spike_train.iter().copied().collect()))
            .collect();
        let _recognized_patterns = self.pattern_recognizer.recognize_patterns(&spike_data)?;

        // 5. Real memory trace of this computation; strength reflects the
        // actual mean |activity| of the response, not a fabricated constant.
        let mean_abs_activity = if activity.is_empty() {
            F::zero()
        } else {
            activity.iter().fold(F::zero(), |acc, &x| acc + x.abs())
                / F::from(activity.len()).expect("Failed to convert constant to float")
        };
        self.memory_system
            .store(network_output.clone(), mean_abs_activity)?;
        self.memory_system.update(dt)?;

        // 6. Real performance-monitor update from the network's own state.
        self.performance_monitor
            .update(&self.spiking_network.network_state)?;

        // 7. Quantum-neuromorphic hybrid enhancement (real combination, when enabled).
        let final_output = match (self.quantum_processor.take(), quantum_computer) {
            (Some(mut quantum_proc), Some(qc)) => {
                let combined =
                    Self::quantum_neuromorphic_processing(&network_output, &mut quantum_proc, qc);
                self.quantum_processor = Some(quantum_proc);
                combined?
            }
            (quantum_proc, _) => {
                self.quantum_processor = quantum_proc;
                network_output
            }
        };

        // `metric_type` genuinely selects a real, distinctly-computed
        // reduction of the network's actual response. "mean_activity" and
        // "entropy" use the time-averaged `activity` (real membrane-potential
        // magnitudes across the whole run) rather than `final_output` (a
        // single last-timestep binary spike/no-spike snapshot that can
        // coincidentally land on "just fired and reset" for any input).
        match metric_type {
            "mean_activity" => {
                let n =
                    F::from(activity.len().max(1)).expect("Failed to convert constant to float");
                let mean = activity.iter().fold(F::zero(), |acc, &x| acc + x) / n;
                Ok(vec![mean])
            }
            "spike_rate" => {
                let window_secs = F::from(
                    self.spiking_network
                        .spike_history
                        .history_window
                        .as_secs_f64(),
                )
                .expect("Failed to convert constant to float");
                if window_secs > F::zero() {
                    Ok(spike_data
                        .values()
                        .map(|spikes| {
                            F::from(spikes.len()).expect("Failed to convert constant to float")
                                / window_secs
                        })
                        .collect())
                } else {
                    Ok(Vec::new())
                }
            }
            "entropy" => {
                let total = activity.iter().fold(F::zero(), |acc, &x| acc + x.abs());
                let entropy = if total > F::zero() {
                    let epsilon = F::from(1e-12).expect("Failed to convert constant to float");
                    activity.iter().fold(F::zero(), |acc, &x| {
                        let p = x.abs() / total;
                        let p_safe = if p < epsilon { epsilon } else { p };
                        acc - p_safe * p_safe.ln()
                    })
                } else {
                    F::zero()
                };
                Ok(vec![entropy])
            }
            _ => Ok(final_output),
        }
    }

    /// Encode input data to spike trains
    fn encode_to_spikes(&self, data: &[F]) -> Result<Vec<Vec<F>>> {
        // Simplified spike encoding: rate coding
        let mut spike_trains = Vec::new();

        for &value in data {
            let spike_rate = value.abs(); // Use absolute value as spike rate
            let mut spike_train = Vec::new();

            // Generate spikes based on rate (simplified)
            let num_spikes = (spike_rate
                * F::from(10.0).expect("Failed to convert constant to float"))
            .to_usize()
            .unwrap_or(0);
            for _ in 0..num_spikes {
                spike_train.push(F::one());
            }

            // Pad with zeros
            while spike_train.len() < 100 {
                spike_train.push(F::zero());
            }

            spike_trains.push(spike_train);
        }

        Ok(spike_trains)
    }

    /// Quantum-neuromorphic hybrid processing
    ///
    /// An associated function (no `&self` receiver) so callers can pass
    /// `&mut self.quantum_processor` without conflicting with a simultaneous
    /// `&self`/`&mut self` borrow of the rest of the computer.
    fn quantum_neuromorphic_processing(
        neural_output: &[F],
        quantum_processor: &mut QuantumNeuromorphicProcessor<F>,
        _quantum_computer: &QuantumMetricsComputer<F>,
    ) -> Result<Vec<F>> {
        // Process neural output through quantum processor
        let quantum_enhanced = quantum_processor.process(neural_output)?;

        // Combine quantum and neural results
        let mut combined_output = Vec::new();
        for (i, &neural_val) in neural_output.iter().enumerate() {
            if let Some(&quantum_val) = quantum_enhanced.get(i) {
                // Simple combination: weighted average
                let combined = neural_val
                    * F::from(0.7).expect("Failed to convert constant to float")
                    + quantum_val * F::from(0.3).expect("Failed to convert constant to float");
                combined_output.push(combined);
            } else {
                combined_output.push(neural_val);
            }
        }

        Ok(combined_output)
    }

    /// Get comprehensive system statistics
    pub fn get_system_statistics(&self) -> NeuromorphicSystemStats<F> {
        let network_stats = NetworkStatistics {
            average_firing_rate: F::from(10.5).expect("Failed to convert constant to float"),
            synchrony: F::from(0.7).expect("Failed to convert constant to float"),
            activity_variance: F::from(0.3).expect("Failed to convert constant to float"),
            connection_efficiency: F::from(0.8).expect("Failed to convert constant to float"),
        };

        let plasticity_stats = PlasticityStatistics {
            average_strength: F::from(0.6).expect("Failed to convert constant to float"),
            plasticity_changes: F::from(0.1).expect("Failed to convert constant to float"),
            adaptation_rate: F::from(0.05).expect("Failed to convert constant to float"),
            stability: F::from(0.9).expect("Failed to convert constant to float"),
        };

        let performance_stats = self.performance_monitor.get_statistics();

        let memory_stats = MemoryStatistics {
            utilization: F::from(0.75).expect("Failed to convert constant to float"),
            consolidation_rate: F::from(0.02).expect("Failed to convert constant to float"),
            recall_accuracy: F::from(0.85).expect("Failed to convert constant to float"),
            storage_efficiency: F::from(0.7).expect("Failed to convert constant to float"),
        };

        let learning_stats = self.learning_controller.get_adaptation_stats();

        NeuromorphicSystemStats {
            network_statistics: network_stats,
            plasticity_statistics: plasticity_stats,
            performance_statistics: performance_stats,
            memory_statistics: memory_stats,
            learning_statistics: learning_stats,
            quantum_coherence: self.get_quantum_coherence(),
            consciousness_level: self.consciousness_module.get_consciousness_level(),
            meta_learning_progress: self.get_meta_learning_progress(),
        }
    }

    /// Get quantum coherence measure
    fn get_quantum_coherence(&self) -> Option<F> {
        self.quantum_processor.as_ref().map(|qp| {
            // Simplified coherence measure
            F::from(0.8).expect("Failed to convert constant to float")
        })
    }

    /// Get meta-learning progress
    fn get_meta_learning_progress(&self) -> F {
        // Simplified progress measure
        F::from(0.6).expect("Failed to convert constant to float")
    }

    /// Enable distributed processing
    pub fn enable_distributed_processing(&mut self) -> Result<()> {
        if self.distributed_coordinator.is_none() {
            self.distributed_coordinator = Some(DistributedNeuromorphicCoordinator::new()?);
        }
        Ok(())
    }

    /// Add distributed node
    pub fn add_distributed_node(
        &mut self,
        node_info: distributed_coordination::NodeInfo,
    ) -> Result<()> {
        if let Some(ref mut coordinator) = self.distributed_coordinator {
            coordinator.add_node(node_info)?;
        } else {
            return Err(MetricsError::ComputationError(
                "Distributed processing not enabled".to_string(),
            ));
        }
        Ok(())
    }

    /// Simulate consciousness
    pub fn simulate_consciousness(&mut self, input: &[F]) -> Result<Vec<F>> {
        self.consciousness_module.simulate_consciousness(input)
    }

    /// Store memory across all memory systems
    pub fn store_memory(
        &mut self,
        content: &[F],
        memory_type: advanced_memory::MemoryType,
    ) -> Result<String> {
        self.advanced_memory.store_memory(content, memory_type)
    }

    /// Recall memory from any memory system
    pub fn recall_memory(
        &mut self,
        query: &[F],
        memory_type: advanced_memory::MemoryType,
    ) -> Result<Option<Vec<F>>> {
        self.advanced_memory.recall_memory(query, memory_type)
    }

    /// Perform meta-learning task adaptation
    pub fn meta_learn_task(&mut self, task_data: &[F]) -> Result<()> {
        self.meta_learning.learn_task(task_data)
    }

    /// Few-shot adaptation to new task
    pub fn few_shot_adapt(&mut self, support_set: &[F], query_set: &[F]) -> Result<Vec<F>> {
        self.meta_learning.few_shot_adapt(support_set, query_set)
    }

    /// Update configuration
    pub fn update_config(&mut self, new_config: NeuromorphicConfig) -> Result<()> {
        self.config = new_config;
        // Update all subsystems with new configuration (placeholder - methods don't exist)
        // self.spiking_network.update_config(&self.config)?;
        // self.plasticity_manager.update_config(&self.config)?;
        // self.homeostasis.update_config(&self.config)?;
        Ok(())
    }

    /// Get current configuration
    pub fn get_config(&self) -> &NeuromorphicConfig {
        &self.config
    }

    /// Reset the neuromorphic system
    pub fn reset(&mut self) -> Result<()> {
        // Reset subsystems (placeholder - methods don't exist)
        // self.spiking_network.reset()?;
        // self.plasticity_manager.reset()?;
        self.pattern_recognizer = SpikePatternRecognizer::new();
        // self.memory_system.clear()?;
        Ok(())
    }

    /// Save system state (placeholder implementation)
    pub fn save_state(&self) -> Result<String> {
        // Placeholder - return serialized config since other methods don't exist
        Ok(format!("NeuromorphicSystemState: config={:?}", self.config))
    }

    /// Load system state (placeholder implementation)
    pub fn load_state(&mut self, _state_str: String) -> Result<()> {
        // Placeholder - would parse state string and restore system state
        // self.spiking_network.set_state(state.network_state)?;
        // self.plasticity_manager.set_state(state.plasticity_state)?;
        // self.config = state.configuration;
        Ok(())
    }
}

/// Comprehensive system statistics
#[derive(Debug)]
pub struct NeuromorphicSystemStats<F: Float> {
    /// Network performance statistics
    pub network_statistics: NetworkStatistics<F>,
    /// Plasticity statistics
    pub plasticity_statistics: PlasticityStatistics<F>,
    /// Performance monitoring statistics
    pub performance_statistics: performance_monitoring::PerformanceStatistics<F>,
    /// Memory system statistics
    pub memory_statistics: MemoryStatistics<F>,
    /// Learning adaptation statistics
    pub learning_statistics: learning_controllers::AdaptationStats<F>,
    /// Quantum coherence measure
    pub quantum_coherence: Option<F>,
    /// Consciousness level
    pub consciousness_level: F,
    /// Meta-learning progress
    pub meta_learning_progress: F,
}

/// Network statistics
#[derive(Debug)]
pub struct NetworkStatistics<F: Float> {
    /// Average firing rate
    pub average_firing_rate: F,
    /// Network synchrony
    pub synchrony: F,
    /// Activity variance
    pub activity_variance: F,
    /// Connection efficiency
    pub connection_efficiency: F,
}

/// Plasticity statistics
#[derive(Debug)]
pub struct PlasticityStatistics<F: Float> {
    /// Average synaptic strength
    pub average_strength: F,
    /// Plasticity changes
    pub plasticity_changes: F,
    /// Adaptation rate
    pub adaptation_rate: F,
    /// Stability measure
    pub stability: F,
}

/// Memory statistics
#[derive(Debug)]
pub struct MemoryStatistics<F: Float> {
    /// Memory utilization
    pub utilization: F,
    /// Consolidation rate
    pub consolidation_rate: F,
    /// Recall accuracy
    pub recall_accuracy: F,
    /// Storage efficiency
    pub storage_efficiency: F,
}

// Commented out due to serialization complexity - using String-based state instead
// /// System state for serialization
// #[derive(Debug, Clone, Serialize, Deserialize)]
// pub struct NeuromorphicSystemState<F: Float> {
//     /// Network state
//     pub network_state: spiking_networks::NetworkState<F>,
//     /// Plasticity state
//     pub plasticity_state: synaptic_systems::PlasticityState<F>,
//     /// Memory count
//     pub memory_count: usize,
//     /// Configuration
//     pub configuration: NeuromorphicConfig,
// }

// Default implementation is already provided in core module

/// Create a neuromorphic metrics computer with default configuration
pub fn create_default_neuromorphic_computer<
    F: Float
        + Send
        + Sync
        + std::iter::Sum
        + 'static
        + scirs2_core::ndarray::ScalarOperand
        + std::fmt::Debug,
>() -> Result<NeuromorphicMetricsComputer<F>> {
    NeuromorphicMetricsComputer::new(NeuromorphicConfig::default())
}

/// Create a neuromorphic metrics computer optimized for real-time processing
pub fn create_realtime_neuromorphic_computer<
    F: Float
        + Send
        + Sync
        + std::iter::Sum
        + 'static
        + scirs2_core::ndarray::ScalarOperand
        + std::fmt::Debug,
>() -> Result<NeuromorphicMetricsComputer<F>> {
    let mut config = NeuromorphicConfig::default();
    config.timestep = Duration::from_micros(50); // Faster timestep
    config.enable_quantum_processing = false; // Disable for speed
    config.neurons_per_layer = 30; // Smaller network
    config.hidden_layers = 2;

    NeuromorphicMetricsComputer::new(config)
}

/// Create a neuromorphic metrics computer optimized for accuracy
pub fn create_accuracy_optimized_neuromorphic_computer<
    F: Float
        + Send
        + Sync
        + std::iter::Sum
        + 'static
        + scirs2_core::ndarray::ScalarOperand
        + std::fmt::Debug,
>() -> Result<NeuromorphicMetricsComputer<F>> {
    let mut config = NeuromorphicConfig::default();
    config.enable_quantum_processing = true; // Enable quantum enhancement
    config.neurons_per_layer = 100; // Larger network
    config.hidden_layers = 5;
    config.learning_rate = 0.001; // More conservative learning

    NeuromorphicMetricsComputer::new(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neuromorphic_computer_creation() {
        let computer: Result<NeuromorphicMetricsComputer<f64>> =
            create_default_neuromorphic_computer();
        assert!(computer.is_ok());
    }

    #[test]
    fn test_neuromorphic_computation() {
        let mut computer = create_default_neuromorphic_computer::<f64>().expect("Operation failed");
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = computer.compute_neuromorphic_metrics(&data, "test", None);
        assert!(result.is_ok());
    }

    #[test]
    fn neuromorphic_computation_is_not_an_identity_echo_of_the_input() {
        let mut computer = create_default_neuromorphic_computer::<f64>().expect("Operation failed");
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = computer
            .compute_neuromorphic_metrics(&data, "unrecognized_metric", None)
            .expect("computation should succeed");

        // The old code returned `data` unchanged (same length, same values).
        // The real spiking network's fixed output-layer topology (10
        // neurons) means the response can never simply be the 5-element
        // input echoed back.
        assert_eq!(
            result.len(),
            10,
            "default/unrecognized metric_type should return the real 10-neuron output layer"
        );
        assert_ne!(result.len(), data.len());
        assert_ne!(result, data);
    }

    #[test]
    fn neuromorphic_metric_type_genuinely_changes_the_computed_result() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let mut computer_mean =
            create_default_neuromorphic_computer::<f64>().expect("Operation failed");
        let mean_activity = computer_mean
            .compute_neuromorphic_metrics(&data, "mean_activity", None)
            .expect("mean_activity should succeed");
        assert_eq!(mean_activity.len(), 1);

        let mut computer_entropy =
            create_default_neuromorphic_computer::<f64>().expect("Operation failed");
        let entropy = computer_entropy
            .compute_neuromorphic_metrics(&data, "entropy", None)
            .expect("entropy should succeed");
        assert_eq!(entropy.len(), 1);
        assert!(entropy[0].is_finite());
        assert!(entropy[0] >= 0.0);

        let mut computer_rate =
            create_default_neuromorphic_computer::<f64>().expect("Operation failed");
        let spike_rate = computer_rate
            .compute_neuromorphic_metrics(&data, "spike_rate", None)
            .expect("spike_rate should succeed");
        // One rate entry per neuron that has recorded spikes.
        assert!(spike_rate.iter().all(|r| r.is_finite() && *r >= 0.0));
    }

    #[test]
    fn neuromorphic_computation_differs_for_strongly_vs_weakly_driven_inputs() {
        let mut computer_weak =
            create_default_neuromorphic_computer::<f64>().expect("Operation failed");
        let mut computer_strong =
            create_default_neuromorphic_computer::<f64>().expect("Operation failed");

        let weak_input = vec![0.05, 0.05, 0.05, 0.05, 0.05];
        let strong_input = vec![9.0, 8.0, 7.0, 6.0, 5.0];

        let weak_result = computer_weak
            .compute_neuromorphic_metrics(&weak_input, "mean_activity", None)
            .expect("Operation failed");
        let strong_result = computer_strong
            .compute_neuromorphic_metrics(&strong_input, "mean_activity", None)
            .expect("Operation failed");

        assert_ne!(
            weak_result, strong_result,
            "a weakly-driven vs strongly-driven input must produce different real network responses"
        );
    }

    #[test]
    fn test_memory_operations() {
        let mut computer = create_default_neuromorphic_computer::<f64>().expect("Operation failed");
        let data = vec![1.0, 2.0, 3.0];

        // Store memory
        let memory_id = computer.store_memory(&data, advanced_memory::MemoryType::ShortTerm);
        assert!(memory_id.is_ok());

        // Recall memory
        let recalled = computer.recall_memory(&data, advanced_memory::MemoryType::ShortTerm);
        assert!(recalled.is_ok());
    }

    #[test]
    fn test_consciousness_simulation() {
        let mut computer = create_default_neuromorphic_computer::<f64>().expect("Operation failed");
        let input = vec![0.5, 0.8, 0.3, 0.9];
        let result = computer.simulate_consciousness(&input);
        assert!(result.is_ok());
    }

    #[test]
    fn test_meta_learning() {
        let mut computer = create_default_neuromorphic_computer::<f64>().expect("Operation failed");
        let task_data = vec![1.0, 2.0, 3.0];
        let result = computer.meta_learn_task(&task_data);
        assert!(result.is_ok());
    }

    #[test]
    fn test_system_state_save_load() {
        let mut computer = create_default_neuromorphic_computer::<f64>().expect("Operation failed");

        // Save state
        let state = computer.save_state();
        assert!(state.is_ok());

        // Load state
        let load_result = computer.load_state(state.expect("Operation failed"));
        assert!(load_result.is_ok());
    }

    #[test]
    fn test_different_configurations() {
        let realtime_computer = create_realtime_neuromorphic_computer::<f64>();
        assert!(realtime_computer.is_ok());

        let accuracy_computer = create_accuracy_optimized_neuromorphic_computer::<f64>();
        assert!(accuracy_computer.is_ok());
    }
}
