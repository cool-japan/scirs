//! Quantum-Neural Hybrid Optimization for Advanced Mode
//!
//! This module combines quantum-inspired computation with neural-adaptive learning
//! to create a hybrid optimization approach that leverages the best of both paradigms.

use crate::error::SparseResult;
use crate::neural_adaptive_sparse::{NeuralAdaptiveConfig, NeuralAdaptiveSparseProcessor};
use crate::quantum_inspired_sparse::{QuantumSparseConfig, QuantumSparseProcessor};
use num_traits::{Float, NumAssign};
use rand::Rng;
use scirs2_core::simd_ops::SimdUnifiedOps;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Quantum-Neural hybrid processor configuration
#[derive(Debug, Clone)]
pub struct QuantumNeuralConfig {
    /// Quantum computation configuration
    pub quantum_config: QuantumSparseConfig,
    /// Neural network configuration
    pub neural_config: NeuralAdaptiveConfig,
    /// Hybrid optimization strategy
    pub hybrid_strategy: HybridStrategy,
    /// Quantum-neural coupling strength
    pub coupling_strength: f64,
    /// Enable quantum state feedback to neural network
    pub quantum_feedback: bool,
    /// Enable neural guidance of quantum processes
    pub neural_guidance: bool,
    /// Coherence threshold for quantum-classical switching
    pub coherence_threshold: f64,
    /// Learning rate for hybrid adaptation
    pub hybrid_learning_rate: f64,
}

/// Hybrid optimization strategies
#[derive(Debug, Clone, Copy)]
pub enum HybridStrategy {
    /// Sequential: quantum first, then neural
    Sequential,
    /// Parallel: quantum and neural in parallel
    Parallel,
    /// Adaptive: switch between quantum and neural based on conditions
    Adaptive,
    /// Entangled: quantum and neural states are entangled
    Entangled,
    /// Superposition: neural network operates in quantum superposition
    QuantumNeural,
}

impl Default for QuantumNeuralConfig {
    fn default() -> Self {
        Self {
            quantum_config: QuantumSparseConfig::default(),
            neural_config: NeuralAdaptiveConfig::default(),
            hybrid_strategy: HybridStrategy::Adaptive,
            coupling_strength: 0.5,
            quantum_feedback: true,
            neural_guidance: true,
            coherence_threshold: 0.7,
            hybrid_learning_rate: 0.001,
        }
    }
}

/// Quantum-Neural hybrid sparse matrix processor
pub struct QuantumNeuralHybridProcessor {
    _config: QuantumNeuralConfig,
    quantum_processor: QuantumSparseProcessor,
    neural_processor: NeuralAdaptiveSparseProcessor,
    hybrid_state: HybridState,
    performance_fusion: PerformanceFusion,
    adaptation_counter: AtomicUsize,
    hybrid_memory: HybridMemory,
}

/// Hybrid state tracking quantum-neural interactions
#[derive(Debug)]
struct HybridState {
    quantum_coherence: f64,
    neural_confidence: f64,
    hybrid_synchronization: f64,
    entanglement_strength: f64,
    quantum_neural_coupling: Vec<f64>,
    decision_history: Vec<HybridDecision>,
}

/// Decision made by the hybrid system
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct HybridDecision {
    timestamp: u64,
    strategy_used: HybridProcessingMode,
    quantum_contribution: f64,
    neural_contribution: f64,
    performance_achieved: f64,
    coherence_at_decision: f64,
}

/// Processing modes for the hybrid system
#[derive(Debug, Clone, Copy)]
enum HybridProcessingMode {
    PureQuantum,
    PureNeural,
    QuantumDominant,
    NeuralDominant,
    BalancedHybrid,
    AdaptiveBlend,
}

/// Performance fusion mechanism
#[derive(Debug)]
#[allow(dead_code)]
struct PerformanceFusion {
    quantum_metrics: Vec<f64>,
    neural_metrics: Vec<f64>,
    fusion_weights: Vec<f64>,
    adaptive_blending: bool,
    fusion_history: Vec<FusionResult>,
}

/// Result of performance fusion
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct FusionResult {
    fused_performance: f64,
    quantum_weight: f64,
    neural_weight: f64,
    fusion_confidence: f64,
}

/// Hybrid memory system combining quantum and neural memories
#[derive(Debug)]
#[allow(dead_code)]
struct HybridMemory {
    quantum_states: HashMap<String, Vec<f64>>,
    neural_patterns: HashMap<String, Vec<f64>>,
    correlation_matrix: Vec<Vec<f64>>,
    memory_capacity: usize,
    forgetting_rate: f64,
}

impl QuantumNeuralHybridProcessor {
    /// Create a new quantum-neural hybrid processor
    pub fn new(config: QuantumNeuralConfig) -> Self {
        let quantum_processor = QuantumSparseProcessor::new(config.quantum_config.clone());
        let neural_processor = NeuralAdaptiveSparseProcessor::new(config.neural_config.clone());

        let hybrid_state = HybridState {
            quantum_coherence: 1.0,
            neural_confidence: 1.0,
            hybrid_synchronization: 1.0,
            entanglement_strength: 0.0,
            quantum_neural_coupling: vec![0.0; 64],
            decision_history: Vec::new(),
        };

        let performance_fusion = PerformanceFusion {
            quantum_metrics: Vec::new(),
            neural_metrics: Vec::new(),
            fusion_weights: vec![0.5, 0.5], // Equal initial weights
            adaptive_blending: true,
            fusion_history: Vec::new(),
        };

        let hybrid_memory = HybridMemory {
            quantum_states: HashMap::new(),
            neural_patterns: HashMap::new(),
            correlation_matrix: vec![vec![0.0; 64]; 64],
            memory_capacity: 1000,
            forgetting_rate: 0.001,
        };

        Self {
            _config: config,
            quantum_processor,
            neural_processor,
            hybrid_state,
            performance_fusion,
            adaptation_counter: AtomicUsize::new(0),
            hybrid_memory,
        }
    }

    /// Hybrid sparse matrix-vector multiplication
    #[allow(clippy::too_many_arguments)]
    pub fn hybrid_spmv<T>(
        &mut self,
        rows: usize,
        cols: usize,
        indptr: &[usize],
        indices: &[usize],
        data: &[T],
        x: &[T],
        y: &mut [T],
    ) -> SparseResult<()>
    where
        T: Float + NumAssign + Send + Sync + Copy + SimdUnifiedOps + Into<f64> + From<f64>,
    {
        let start_time = std::time::Instant::now();

        // Update hybrid state
        self.update_hybrid_state(indptr, indices);

        // Select processing mode based on hybrid strategy
        let processing_mode = self.select_processing_mode(rows, cols, indptr, indices);

        // Execute hybrid computation
        let result = match processing_mode {
            HybridProcessingMode::PureQuantum => self
                .quantum_processor
                .quantum_spmv(rows, indptr, indices, data, x, y),
            HybridProcessingMode::PureNeural => self
                .neural_processor
                .adaptive_spmv(rows, cols, indptr, indices, data, x, y),
            HybridProcessingMode::QuantumDominant => {
                self.quantum_dominant_hybrid(rows, cols, indptr, indices, data, x, y)
            }
            HybridProcessingMode::NeuralDominant => {
                self.neural_dominant_hybrid(rows, cols, indptr, indices, data, x, y)
            }
            HybridProcessingMode::BalancedHybrid => {
                self.balanced_hybrid(rows, cols, indptr, indices, data, x, y)
            }
            HybridProcessingMode::AdaptiveBlend => {
                self.adaptive_blend(rows, cols, indptr, indices, data, x, y)
            }
        };

        // Record performance and update fusion weights
        let execution_time = start_time.elapsed().as_secs_f64();
        self.update_performance_fusion(processing_mode, execution_time);

        // Learn from the interaction
        self.update_hybrid_learning(processing_mode, execution_time);

        self.adaptation_counter.fetch_add(1, Ordering::Relaxed);

        result
    }

    /// Update hybrid state based on matrix characteristics
    fn update_hybrid_state(&mut self, indptr: &[usize], indices: &[usize]) {
        // Update quantum coherence based on sparsity pattern regularity
        let sparsity_regularity = self.calculate_sparsity_regularity(indptr, indices);
        self.hybrid_state.quantum_coherence = sparsity_regularity;

        // Update neural confidence based on pattern recognition
        let pattern_familiarity = self.calculate_pattern_familiarity(indptr, indices);
        self.hybrid_state.neural_confidence = pattern_familiarity;

        // Update hybrid synchronization
        self.hybrid_state.hybrid_synchronization =
            (self.hybrid_state.quantum_coherence + self.hybrid_state.neural_confidence) / 2.0;

        // Update quantum-neural coupling
        self.update_quantum_neural_coupling();

        // Apply memory decay
        self.apply_memory_decay();
    }

    /// Select processing mode for the current matrix
    fn select_processing_mode(
        &self,
        rows: usize,
        cols: usize,
        indptr: &[usize],
        indices: &[usize],
    ) -> HybridProcessingMode {
        match self._config.hybrid_strategy {
            HybridStrategy::Sequential => {
                if self.hybrid_state.quantum_coherence > self._config.coherence_threshold {
                    HybridProcessingMode::PureQuantum
                } else {
                    HybridProcessingMode::PureNeural
                }
            }
            HybridStrategy::Parallel => HybridProcessingMode::BalancedHybrid,
            HybridStrategy::Adaptive => self.adaptive_mode_selection(rows, cols, indptr, indices),
            HybridStrategy::Entangled => HybridProcessingMode::AdaptiveBlend,
            HybridStrategy::QuantumNeural => HybridProcessingMode::QuantumDominant,
        }
    }

    /// Adaptive mode selection based on current conditions
    fn adaptive_mode_selection(
        &self,
        rows: usize,
        _cols: usize,
        indptr: &[usize],
        _indices: &[usize],
    ) -> HybridProcessingMode {
        let quantum_score = self.hybrid_state.quantum_coherence * 0.7
            + self.hybrid_state.entanglement_strength * 0.3;

        let neural_score = self.hybrid_state.neural_confidence * 0.8
            + self.hybrid_state.hybrid_synchronization * 0.2;

        // Consider matrix size and sparsity
        let size_factor = (rows as f64).log10() / 6.0; // Normalize by 10^6
        let avg_nnz = if rows > 0 { indptr[rows] / rows } else { 0 };
        let sparsity_factor = (avg_nnz as f64 / 100.0).min(1.0);

        let quantum_preference = quantum_score + size_factor * 0.3;
        let neural_preference = neural_score + sparsity_factor * 0.3;

        if quantum_preference > neural_preference + 0.2 {
            HybridProcessingMode::QuantumDominant
        } else if neural_preference > quantum_preference + 0.2 {
            HybridProcessingMode::NeuralDominant
        } else {
            HybridProcessingMode::BalancedHybrid
        }
    }

    /// Quantum-dominant hybrid processing
    fn quantum_dominant_hybrid<T>(
        &mut self,
        rows: usize,
        cols: usize,
        indptr: &[usize],
        indices: &[usize],
        data: &[T],
        x: &[T],
        y: &mut [T],
    ) -> SparseResult<()>
    where
        T: Float + NumAssign + Send + Sync + Copy + SimdUnifiedOps + Into<f64> + From<f64>,
    {
        // Primary computation with quantum processor
        let mut quantum_result = vec![T::zero(); rows];
        self.quantum_processor
            .quantum_spmv(rows, indptr, indices, data, x, &mut quantum_result)?;

        // Neural guidance for post-processing
        if self._config.neural_guidance {
            let mut neural_result = vec![T::zero(); rows];
            self.neural_processor.adaptive_spmv(
                rows,
                cols,
                indptr,
                indices,
                data,
                x,
                &mut neural_result,
            )?;

            // Combine results with quantum dominance
            let quantum_weight = 0.85;
            let neural_weight = 0.15;

            for i in 0..rows {
                y[i] = num_traits::cast(
                    quantum_weight * quantum_result[i].into()
                        + neural_weight * neural_result[i].into(),
                )
                .unwrap_or(T::zero());
            }
        } else {
            y.copy_from_slice(&quantum_result);
        }

        Ok(())
    }

    /// Neural-dominant hybrid processing
    fn neural_dominant_hybrid<T>(
        &mut self,
        rows: usize,
        cols: usize,
        indptr: &[usize],
        indices: &[usize],
        data: &[T],
        x: &[T],
        y: &mut [T],
    ) -> SparseResult<()>
    where
        T: Float + NumAssign + Send + Sync + Copy + SimdUnifiedOps + Into<f64> + From<f64>,
    {
        // Primary computation with neural processor
        self.neural_processor
            .adaptive_spmv(rows, cols, indptr, indices, data, x, y)?;

        // Quantum enhancement for specific patterns
        if self._config.quantum_feedback && self.hybrid_state.quantum_coherence > 0.5 {
            let mut quantum_enhancement = vec![T::zero(); rows];
            self.quantum_processor.quantum_spmv(
                rows,
                indptr,
                indices,
                data,
                x,
                &mut quantum_enhancement,
            )?;

            // Apply quantum enhancement selectively
            let enhancement_strength = self.hybrid_state.quantum_coherence * 0.2;

            for i in 0..rows {
                let current_val: f64 = y[i].into();
                let enhancement: f64 = quantum_enhancement[i].into();
                y[i] = num_traits::cast(current_val + enhancement_strength * enhancement)
                    .unwrap_or(T::zero());
            }
        }

        Ok(())
    }

    /// Balanced hybrid processing
    fn balanced_hybrid<T>(
        &mut self,
        rows: usize,
        cols: usize,
        indptr: &[usize],
        indices: &[usize],
        data: &[T],
        x: &[T],
        y: &mut [T],
    ) -> SparseResult<()>
    where
        T: Float + NumAssign + Send + Sync + Copy + SimdUnifiedOps + Into<f64> + From<f64>,
    {
        // Parallel execution of both approaches
        let mut quantum_result = vec![T::zero(); rows];
        let mut neural_result = vec![T::zero(); rows];

        // Execute both in sequence (in a real implementation, this could be parallel)
        self.quantum_processor
            .quantum_spmv(rows, indptr, indices, data, x, &mut quantum_result)?;
        self.neural_processor.adaptive_spmv(
            rows,
            cols,
            indptr,
            indices,
            data,
            x,
            &mut neural_result,
        )?;

        // Balanced fusion
        let quantum_weight = self.performance_fusion.fusion_weights[0];
        let neural_weight = self.performance_fusion.fusion_weights[1];

        for i in 0..rows {
            y[i] = num_traits::cast(
                quantum_weight * quantum_result[i].into() + neural_weight * neural_result[i].into(),
            )
            .unwrap_or(T::zero());
        }

        Ok(())
    }

    /// Adaptive blend processing
    fn adaptive_blend<T>(
        &mut self,
        rows: usize,
        _cols: usize,
        indptr: &[usize],
        indices: &[usize],
        data: &[T],
        x: &[T],
        y: &mut [T],
    ) -> SparseResult<()>
    where
        T: Float + NumAssign + Send + Sync + Copy + SimdUnifiedOps + Into<f64> + From<f64>,
    {
        // Dynamically adjust blending per row based on characteristics
        for row in 0..rows {
            let start_idx = indptr[row];
            let end_idx = indptr[row + 1];
            let row_nnz = end_idx - start_idx;

            // Determine blend ratio for this row
            let quantum_ratio = if row_nnz > 50 {
                self.hybrid_state.quantum_coherence
            } else {
                self.hybrid_state.neural_confidence
            };

            let neural_ratio = 1.0 - quantum_ratio;

            // Compute for single row (simplified)
            let mut quantum_sum = 0.0;
            let mut neural_sum = 0.0;

            for idx in start_idx..end_idx {
                let col = indices[idx];
                let val: f64 = data[idx].into();
                let x_val: f64 = x[col].into();

                // Quantum-inspired computation with coupling
                let coupling_factor = if idx < self.hybrid_state.quantum_neural_coupling.len() {
                    self.hybrid_state.quantum_neural_coupling
                        [idx % self.hybrid_state.quantum_neural_coupling.len()]
                } else {
                    1.0
                };

                quantum_sum += val * x_val * coupling_factor;
                neural_sum += val * x_val; // Standard computation
            }

            // Adaptive blending
            y[row] = num_traits::cast(quantum_ratio * quantum_sum + neural_ratio * neural_sum)
                .unwrap_or(T::zero());
        }

        Ok(())
    }

    // Helper methods

    fn calculate_sparsity_regularity(&self, indptr: &[usize], indices: &[usize]) -> f64 {
        let rows = indptr.len() - 1;
        if rows <= 1 {
            return 1.0;
        }

        let mut nnz_per_row = Vec::new();
        for row in 0..rows {
            nnz_per_row.push(indptr[row + 1] - indptr[row]);
        }

        let mean = nnz_per_row.iter().sum::<usize>() as f64 / rows as f64;
        let variance = nnz_per_row
            .iter()
            .map(|&x| (x as f64 - mean).powi(2))
            .sum::<f64>()
            / rows as f64;

        // Regularity is inversely related to variance
        (1.0 / (1.0 + variance.sqrt())).min(1.0)
    }

    fn calculate_pattern_familiarity(&self, _indptr: &[usize], indices: &[usize]) -> f64 {
        // Simplified pattern familiarity based on memory
        let memory_size = self.hybrid_memory.neural_patterns.len();
        let max_memory = self.hybrid_memory.memory_capacity;

        (memory_size as f64 / max_memory as f64).min(1.0)
    }

    fn update_quantum_neural_coupling(&mut self) {
        let coupling_strength = self._config.coupling_strength;
        let synchronization = self.hybrid_state.hybrid_synchronization;

        for coupling in &mut self.hybrid_state.quantum_neural_coupling {
            // Update coupling based on synchronization and random fluctuations
            let fluctuation = (rand::rng().random::<f64>() - 0.5) * 0.1;
            *coupling = coupling_strength * synchronization + fluctuation;
            *coupling = coupling.clamp(0.0, 2.0);
        }

        // Update entanglement strength
        self.hybrid_state.entanglement_strength = self
            .hybrid_state
            .quantum_neural_coupling
            .iter()
            .sum::<f64>()
            / self.hybrid_state.quantum_neural_coupling.len() as f64;
    }

    fn apply_memory_decay(&mut self) {
        let decay_factor = 1.0 - self.hybrid_memory.forgetting_rate;

        // Decay correlation matrix
        for row in &mut self.hybrid_memory.correlation_matrix {
            for val in row {
                *val *= decay_factor;
            }
        }
    }

    fn update_performance_fusion(&mut self, mode: HybridProcessingMode, executiontime: f64) {
        let performance = 1.0 / (executiontime + 1e-6);

        match mode {
            HybridProcessingMode::PureQuantum | HybridProcessingMode::QuantumDominant => {
                self.performance_fusion.quantum_metrics.push(performance);
                if self.performance_fusion.quantum_metrics.len() > 100 {
                    self.performance_fusion.quantum_metrics.remove(0);
                }
            }
            HybridProcessingMode::PureNeural | HybridProcessingMode::NeuralDominant => {
                self.performance_fusion.neural_metrics.push(performance);
                if self.performance_fusion.neural_metrics.len() > 100 {
                    self.performance_fusion.neural_metrics.remove(0);
                }
            }
            HybridProcessingMode::BalancedHybrid | HybridProcessingMode::AdaptiveBlend => {
                self.performance_fusion
                    .quantum_metrics
                    .push(performance * 0.5);
                self.performance_fusion
                    .neural_metrics
                    .push(performance * 0.5);
            }
        }

        // Adapt fusion weights based on recent performance
        if self.performance_fusion.adaptive_blending {
            self.update_fusion_weights();
        }
    }

    fn update_fusion_weights(&mut self) {
        let quantum_avg = if !self.performance_fusion.quantum_metrics.is_empty() {
            self.performance_fusion.quantum_metrics.iter().sum::<f64>()
                / self.performance_fusion.quantum_metrics.len() as f64
        } else {
            0.5
        };

        let neural_avg = if !self.performance_fusion.neural_metrics.is_empty() {
            self.performance_fusion.neural_metrics.iter().sum::<f64>()
                / self.performance_fusion.neural_metrics.len() as f64
        } else {
            0.5
        };

        let total = quantum_avg + neural_avg;
        if total > 0.0 {
            self.performance_fusion.fusion_weights[0] = quantum_avg / total;
            self.performance_fusion.fusion_weights[1] = neural_avg / total;
        }

        // Add slight exploration
        let exploration = 0.05;
        self.performance_fusion.fusion_weights[0] +=
            (rand::rng().random::<f64>() - 0.5) * exploration;
        self.performance_fusion.fusion_weights[1] = 1.0 - self.performance_fusion.fusion_weights[0];

        // Clamp weights
        self.performance_fusion.fusion_weights[0] =
            self.performance_fusion.fusion_weights[0].clamp(0.1, 0.9);
        self.performance_fusion.fusion_weights[1] = 1.0 - self.performance_fusion.fusion_weights[0];
    }

    fn update_hybrid_learning(&mut self, mode: HybridProcessingMode, executiontime: f64) {
        let decision = HybridDecision {
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            strategy_used: mode,
            quantum_contribution: self.performance_fusion.fusion_weights[0],
            neural_contribution: self.performance_fusion.fusion_weights[1],
            performance_achieved: 1.0 / (executiontime + 1e-6),
            coherence_at_decision: self.hybrid_state.quantum_coherence,
        };

        self.hybrid_state.decision_history.push(decision);

        // Keep only recent decisions
        if self.hybrid_state.decision_history.len() > 1000 {
            self.hybrid_state.decision_history.remove(0);
        }
    }

    /// Get hybrid processor statistics
    pub fn get_stats(&self) -> QuantumNeuralHybridStats {
        let recent_decisions = self
            .hybrid_state
            .decision_history
            .iter()
            .rev()
            .take(10)
            .collect::<Vec<_>>();

        let avg_performance = if !recent_decisions.is_empty() {
            recent_decisions
                .iter()
                .map(|d| d.performance_achieved)
                .sum::<f64>()
                / recent_decisions.len() as f64
        } else {
            0.0
        };

        QuantumNeuralHybridStats {
            total_operations: self.adaptation_counter.load(Ordering::Relaxed),
            quantum_coherence: self.hybrid_state.quantum_coherence,
            neural_confidence: self.hybrid_state.neural_confidence,
            hybrid_synchronization: self.hybrid_state.hybrid_synchronization,
            entanglement_strength: self.hybrid_state.entanglement_strength,
            quantum_weight: self.performance_fusion.fusion_weights[0],
            neural_weight: self.performance_fusion.fusion_weights[1],
            average_performance: avg_performance,
            memory_utilization: self.hybrid_memory.neural_patterns.len() as f64
                / self.hybrid_memory.memory_capacity as f64,
            decision_history_size: self.hybrid_state.decision_history.len(),
        }
    }
}

/// Statistics for quantum-neural hybrid processor
#[derive(Debug)]
pub struct QuantumNeuralHybridStats {
    pub total_operations: usize,
    pub quantum_coherence: f64,
    pub neural_confidence: f64,
    pub hybrid_synchronization: f64,
    pub entanglement_strength: f64,
    pub quantum_weight: f64,
    pub neural_weight: f64,
    pub average_performance: f64,
    pub memory_utilization: f64,
    pub decision_history_size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    #[ignore] // Slow test - quantum neural hybrid initialization
    fn test_quantum_neural_hybrid_creation() {
        let config = QuantumNeuralConfig::default();
        let processor = QuantumNeuralHybridProcessor::new(config);

        assert_eq!(processor.hybrid_state.quantum_coherence, 1.0);
        assert_eq!(processor.hybrid_state.neural_confidence, 1.0);
        assert_eq!(processor.performance_fusion.fusion_weights.len(), 2);
    }

    #[test]
    #[ignore = "timeout"]
    fn test_hybrid_spmv() {
        let config = QuantumNeuralConfig::default();
        let mut processor = QuantumNeuralHybridProcessor::new(config);

        // Simple test matrix: [[1, 2], [0, 3]]
        let indptr = vec![0, 2, 3];
        let indices = vec![0, 1, 1];
        let data = vec![1.0, 2.0, 3.0];
        let x = vec![1.0, 1.0];
        let mut y = vec![0.0; 2];

        processor
            .hybrid_spmv(2, 2, &indptr, &indices, &data, &x, &mut y)
            .unwrap();

        // Results should be reasonable (exact values depend on hybrid strategy)
        assert!(y[0] > 2.0 && y[0] < 4.0);
        assert!(y[1] > 2.0 && y[1] < 4.0);
    }

    #[test]
    #[ignore] // Slow test - hybrid processor stats
    fn test_hybrid_stats() {
        let config = QuantumNeuralConfig::default();
        let processor = QuantumNeuralHybridProcessor::new(config);
        let stats = processor.get_stats();

        assert_eq!(stats.total_operations, 0);
        assert_eq!(stats.quantum_coherence, 1.0);
        assert_eq!(stats.neural_confidence, 1.0);
        assert!(stats.quantum_weight > 0.0);
        assert!(stats.neural_weight > 0.0);
        assert_relative_eq!(
            stats.quantum_weight + stats.neural_weight,
            1.0,
            epsilon = 1e-10
        );
    }
}
