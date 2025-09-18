//! Transformer-Based Meta-Learning for Optimization
//!
//! This module implements transformer architectures specifically designed for
//! meta-learning in optimization tasks. It includes attention mechanisms,
//! sequence modeling for optimization trajectories, and advanced transformer
//! architectures tailored for learning optimization strategies.

pub mod config;
pub mod architecture;
pub mod attention;
pub mod layers;
pub mod positional_encoding;
pub mod feedforward;
pub mod meta_learning;
pub mod sequence_processor;
pub mod memory_manager;
pub mod performance_tracker;
pub mod state;

// Re-export main types for backward compatibility
pub use config::{TransformerBasedOptimizerConfig, TransformerArchConfig};
pub use architecture::{TransformerArchitecture, TransformerLayer};
pub use attention::{MultiHeadAttention, AttentionMechanism};
pub use layers::{EmbeddingLayer, LayerNormalization, DropoutLayer, OutputProjection, ResidualConnections};
pub use positional_encoding::{PositionalEncoding, PositionalEncodingType};
pub use feedforward::{FeedForwardNetwork, ActivationFunction};
pub use meta_learning::{TransformerMetaLearning, MetaLearningStrategy};
pub use sequence_processor::{OptimizationSequenceProcessor, SequenceProcessingStrategy};
pub use memory_manager::{TransformerMemoryManager, MemoryManagementStrategy};
pub use performance_tracker::{TransformerPerformanceTracker, PerformanceMetrics};
pub use state::{TransformerOptimizerState, OptimizerStateSnapshot};

// Re-export for backward compatibility - create alias for the old name
pub use TransformerBasedOptimizerConfig as TransformerOptimizerConfig;

use ndarray::{Array1, Array2, Array3, ArrayBase, Data, Dimension, Axis};
use num_traits::{Float, ToPrimitive};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use serde::{Serialize, Deserialize};

use crate::adaptive_selection::OptimizerType;
use crate::error::{OptimError, Result};
use super::{
    LearnedOptimizerConfig, MetaOptimizationStrategy, NeuralOptimizerType,
    TaskContext, OptimizerState, NeuralOptimizerMetrics, TaskPerformance
};

// Import for external compatibility

/// Transformer-based meta-learning optimizer
pub struct TransformerOptimizer<T: Float> {
    /// Core transformer architecture
    transformer: TransformerArchitecture<T>,

    /// Positional encoding for sequence modeling
    positional_encoding: PositionalEncoding<T>,

    /// Attention mechanism for optimization history
    attention_mechanism: MultiHeadAttention<T>,

    /// Feed-forward networks for optimization steps
    feedforward_networks: Vec<FeedForwardNetwork<T>>,

    /// Meta-learning components
    meta_learning: TransformerMetaLearning<T>,

    /// Sequence processor for optimization trajectories
    sequence_processor: OptimizationSequenceProcessor<T>,

    /// Memory management for long sequences
    memory_manager: TransformerMemoryManager<T>,

    /// Configuration
    config: TransformerBasedOptimizerConfig<T>,

    /// Performance tracking
    performance_tracker: TransformerPerformanceTracker<T>,

    /// State management
    state: TransformerOptimizerState<T>,
}

impl<T: Float + ndarray::ScalarOperand + num_traits::FromPrimitive> TransformerOptimizer<T> {
    /// Create new transformer optimizer
    pub fn new(config: TransformerBasedOptimizerConfig<T>) -> Result<Self> {
        let transformer_config = TransformerArchConfig::from_optimizer_config(&config);
        let transformer = TransformerArchitecture::new(transformer_config)?;

        let positional_encoding = PositionalEncoding::new(
            config.sequence_length,
            config.model_dimension,
            config.positional_encoding_type,
        )?;

        let attention_mechanism = MultiHeadAttention::new(
            config.num_attention_heads,
            config.model_dimension,
            config.attention_head_dimension,
        )?;

        let mut feedforward_networks = Vec::new();
        for _ in 0..config.num_transformer_layers {
            feedforward_networks.push(FeedForwardNetwork::new(
                config.model_dimension,
                config.feedforward_dimension,
                config.activation_function,
            )?);
        }

        let meta_learning = TransformerMetaLearning::new(&config)?;
        let sequence_processor = OptimizationSequenceProcessor::new(&config)?;
        let memory_manager = TransformerMemoryManager::new(&config)?;
        let performance_tracker = TransformerPerformanceTracker::new();
        let state = TransformerOptimizerState::new(&config)?;

        Ok(Self {
            transformer,
            positional_encoding,
            attention_mechanism,
            feedforward_networks,
            meta_learning,
            sequence_processor,
            memory_manager,
            config,
            performance_tracker,
            state,
        })
    }

    /// Generate optimization step using transformer
    pub fn generate_optimization_step(
        &mut self,
        gradient_history: &Array2<T>,
        parameter_history: &Array2<T>,
        loss_history: &Array1<T>,
    ) -> Result<Array1<T>> {
        let start_time = Instant::now();

        // Process input sequences
        let processed_sequence = self.sequence_processor.process_optimization_sequence(
            gradient_history,
            parameter_history,
            loss_history,
        )?;

        // Apply positional encoding
        let encoded_sequence = self.positional_encoding.encode(&processed_sequence)?;

        // Forward pass through transformer
        let transformer_output = self.transformer.forward(&encoded_sequence)?;

        // Generate optimization step
        let optimization_step = self.meta_learning.generate_update(
            &transformer_output,
            &self.state.current_parameters,
        )?;

        // Update state
        self.state.update_with_step(&optimization_step, loss_history.last().copied())?;

        // Track performance
        let elapsed = start_time.elapsed();
        self.performance_tracker.record_optimization_step(elapsed, &optimization_step);

        Ok(optimization_step)
    }

    /// Train the transformer on optimization trajectories
    pub fn train_on_trajectories(
        &mut self,
        trajectories: &[OptimizationTrajectory<T>],
    ) -> Result<TrainingMetrics> {
        let start_time = Instant::now();
        let mut total_loss = T::zero();
        let mut batch_count = 0;

        for trajectory in trajectories {
            // Process trajectory into sequences
            let sequences = self.sequence_processor.trajectory_to_sequences(trajectory)?;

            for sequence in sequences {
                // Forward pass
                let prediction = self.forward_sequence(&sequence.input)?;

                // Calculate loss
                let loss = self.calculate_sequence_loss(&prediction, &sequence.target)?;
                total_loss = total_loss + loss;

                // Backward pass (simplified)
                self.backward_pass(&sequence.input, &sequence.target, loss)?;

                batch_count += 1;
            }
        }

        let avg_loss = if batch_count > 0 {
            total_loss / num_traits::cast::cast(batch_count).unwrap_or_else(|| T::zero())
        } else {
            T::zero()
        };

        let training_time = start_time.elapsed();

        let metrics = TrainingMetrics {
            loss: avg_loss.to_f64().unwrap_or(0.0),
            training_time,
            num_sequences: batch_count,
            convergence_rate: self.calculate_convergence_rate()?,
        };

        self.performance_tracker.record_training_epoch(metrics.clone());

        Ok(metrics)
    }

    /// Forward pass through the transformer for a sequence
    fn forward_sequence(&mut self, sequence: &Array2<T>) -> Result<Array2<T>> {
        // Apply positional encoding
        let encoded_sequence = self.positional_encoding.encode(sequence)?;

        // Forward through transformer
        self.transformer.forward(&encoded_sequence)
    }

    /// Calculate loss for sequence prediction
    fn calculate_sequence_loss(&self, prediction: &Array2<T>, target: &Array2<T>) -> Result<T> {
        if prediction.shape() != target.shape() {
            return Err(OptimError::Other("Shape mismatch in loss calculation".to_string()));
        }

        // Mean squared error
        let diff = prediction - target;
        let squared_diff = &diff * &diff;
        let sum = squared_diff.sum();
        let mse = sum / T::from(prediction.len()).unwrap();

        Ok(mse)
    }

    /// Simplified backward pass
    fn backward_pass(
        &mut self,
        input: &Array2<T>,
        target: &Array2<T>,
        loss: T,
    ) -> Result<()> {
        // In a full implementation, this would compute gradients and update parameters
        // For now, we'll just update the learning state
        self.meta_learning.update_from_loss(loss)?;
        Ok(())
    }

    /// Calculate convergence rate
    fn calculate_convergence_rate(&self) -> Result<f64> {
        let loss_history = self.performance_tracker.get_loss_history();
        if loss_history.len() < 2 {
            return Ok(0.0);
        }

        let recent_losses: Vec<_> = loss_history.iter().rev().take(10).collect();
        if recent_losses.len() < 2 {
            return Ok(0.0);
        }

        let initial_loss = *recent_losses.last().unwrap();
        let final_loss = *recent_losses.first().unwrap();

        let improvement = (initial_loss - final_loss) / initial_loss;
        let improvement_f64 = improvement.to_f64().unwrap_or(0.0);
        Ok(improvement_f64.max(0.0).min(1.0))
    }

    /// Get current state
    pub fn get_state(&self) -> &TransformerOptimizerState<T> {
        &self.state
    }

    /// Get performance metrics
    pub fn get_performance_metrics(&self) -> &TransformerPerformanceTracker<T> {
        &self.performance_tracker
    }

    /// Reset optimizer state
    pub fn reset_state(&mut self) -> Result<()> {
        self.state = TransformerOptimizerState::new(&self.config)?;
        self.performance_tracker.reset();
        Ok(())
    }
}

/// Optimization trajectory for training
#[derive(Debug, Clone)]
pub struct OptimizationTrajectory<T: Float> {
    pub gradient_sequence: Array2<T>,
    pub parameter_sequence: Array2<T>,
    pub loss_sequence: Array1<T>,
    pub metadata: TrajectoryMetadata,
}

/// Trajectory metadata
#[derive(Debug, Clone)]
pub struct TrajectoryMetadata {
    pub task_id: String,
    pub optimizer_type: String,
    pub convergence_achieved: bool,
    pub total_steps: usize,
}

/// Training sequence
#[derive(Debug, Clone)]
pub struct TrainingSequence<T: Float> {
    pub input: Array2<T>,
    pub target: Array2<T>,
    pub sequence_length: usize,
}

/// Training metrics
#[derive(Debug, Clone)]
pub struct TrainingMetrics {
    pub loss: f64,
    pub training_time: Duration,
    pub num_sequences: usize,
    pub convergence_rate: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transformer_optimizer_creation() {
        let config = TransformerBasedOptimizerConfig::default();
        let optimizer = TransformerOptimizer::<f32>::new(config);
        assert!(optimizer.is_ok());
    }

    #[test]
    fn test_trajectory_creation() {
        let trajectory = OptimizationTrajectory::<f32> {
            gradient_sequence: Array2::zeros((10, 5)),
            parameter_sequence: Array2::zeros((10, 5)),
            loss_sequence: Array1::zeros(10),
            metadata: TrajectoryMetadata {
                task_id: "test".to_string(),
                optimizer_type: "adam".to_string(),
                convergence_achieved: true,
                total_steps: 10,
            },
        };

        assert_eq!(trajectory.gradient_sequence.shape(), &[10, 5]);
        assert_eq!(trajectory.loss_sequence.len(), 10);
    }
}