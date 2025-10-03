//! Neural-Adaptive Sparse Matrix Operations for Advanced Mode
//!
//! This module implements neural network-inspired adaptive algorithms for sparse matrix
//! operations that learn and optimize based on matrix characteristics and usage patterns.
//!
//! ## Architecture
//!
//! The neural adaptive system consists of several interconnected components:
//!
//! - **Neural Networks**: Multi-layer perceptrons with attention mechanisms for pattern recognition
//! - **Transformer Models**: Advanced attention-based models for complex pattern understanding
//! - **Reinforcement Learning**: Agents that learn optimal strategies through trial and reward
//! - **Pattern Memory**: Efficient storage and retrieval of learned optimization patterns
//! - **Configuration**: Flexible configuration system for different use cases
//!
//! ## Usage
//!
//! ```rust,ignore
//! use scirs2_sparse::neural_adaptive_sparse::{
//!     NeuralAdaptiveSparseProcessor, NeuralAdaptiveConfig, OptimizationStrategy
//! };
//!
//! // Create a configuration
//! let config = NeuralAdaptiveConfig::default();
//!
//! // Create the processor
//! let mut processor = NeuralAdaptiveSparseProcessor::new(config);
//!
//! // Use the processor to optimize matrix operations
//! // (actual matrix features would be extracted from real sparse matrices)
//! let matrix_features = vec![1.0, 2.0, 3.0]; // Simplified example
//! let context = OperationContext {
//!     matrix_shape: (1000, 1000),
//!     nnz: 5000,
//!     operation_type: OperationType::MatVec,
//!     performance_target: PerformanceTarget::Speed,
//! };
//!
//! let strategy = processor.optimize_operation::<f64>(&matrix_features, &context)?;
//! ```
//!
//! ## Performance Learning
//!
//! The system learns from performance feedback to improve future optimizations:
//!
//! ```rust,ignore
//! use scirs2_sparse::neural_adaptive_sparse::{PerformanceMetrics, OptimizationStrategy};
//!
//! // After executing the operation, provide performance feedback
//! let performance = PerformanceMetrics::new(
//!     0.1,  // execution_time
//!     0.8,  // cache_efficiency
//!     0.9,  // simd_utilization
//!     0.7,  // parallel_efficiency
//!     0.85, // memory_bandwidth
//!     OptimizationStrategy::SIMDVectorized,
//! );
//!
//! processor.learn_from_performance(
//!     OptimizationStrategy::SIMDVectorized,
//!     performance,
//!     &matrix_features,
//!     &context,
//! )?;
//! ```

pub mod config;
pub mod neural_network;
pub mod pattern_memory;
pub mod processor;
pub mod reinforcement_learning;
pub mod transformer;

// Re-export main types for convenience
pub use config::NeuralAdaptiveConfig;
pub use pattern_memory::OptimizationStrategy;
pub use processor::{
    NeuralAdaptiveSparseProcessor, NeuralProcessorStats, OperationContext, OperationType,
    PerformanceTarget, ProcessorState,
};
pub use reinforcement_learning::{PerformanceMetrics, RLAlgorithm};

// Re-export key internal types that might be useful
pub use neural_network::ActivationFunction;
pub use transformer::{LayerGradients, TransformerGradients};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neural_adaptive_config_creation() {
        let config = NeuralAdaptiveConfig::new();
        assert_eq!(config.hidden_layers, 3);
        assert_eq!(config.neurons_per_layer, 64);
        assert!(config.reinforcement_learning);
    }

    #[test]
    fn test_config_builder_pattern() {
        let config = NeuralAdaptiveConfig::new()
            .with_hidden_layers(2)
            .with_neurons_per_layer(32)
            .with_learning_rate(0.01)
            .with_reinforcement_learning(false);

        assert_eq!(config.hidden_layers, 2);
        assert_eq!(config.neurons_per_layer, 32);
        assert_eq!(config.learningrate, 0.01);
        assert!(!config.reinforcement_learning);
    }

    #[test]
    fn test_config_validation() {
        let mut config = NeuralAdaptiveConfig::new();
        assert!(config.validate().is_ok());

        config.hidden_layers = 0;
        assert!(config.validate().is_err());

        config.hidden_layers = 3;
        config.learningrate = -0.1;
        assert!(config.validate().is_err());

        config.learningrate = 0.001;
        config.modeldim = 63; // Not divisible by attention_heads (8)
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_predefined_configurations() {
        let lightweight = NeuralAdaptiveConfig::lightweight();
        assert_eq!(lightweight.hidden_layers, 2);
        assert_eq!(lightweight.neurons_per_layer, 16);
        assert!(!lightweight.self_attention);

        let high_perf = NeuralAdaptiveConfig::high_performance();
        assert_eq!(high_perf.hidden_layers, 5);
        assert_eq!(high_perf.neurons_per_layer, 128);
        assert!(high_perf.self_attention);

        let mem_efficient = NeuralAdaptiveConfig::memory_efficient();
        assert_eq!(mem_efficient.hidden_layers, 2);
        assert_eq!(mem_efficient.memory_capacity, 1000);
        assert!(!mem_efficient.reinforcement_learning);
    }

    #[test]
    fn test_processor_creation() {
        let config = NeuralAdaptiveConfig::lightweight();
        let processor = NeuralAdaptiveSparseProcessor::new(config);

        let stats = processor.get_statistics();
        assert_eq!(stats.total_operations, 0);
        assert_eq!(stats.successful_adaptations, 0);
    }

    #[test]
    fn test_operation_context() {
        let context = OperationContext {
            matrix_shape: (1000, 1000),
            nnz: 5000,
            operation_type: OperationType::MatVec,
            performance_target: PerformanceTarget::Speed,
        };

        assert_eq!(context.matrix_shape, (1000, 1000));
        assert_eq!(context.nnz, 5000);
    }

    #[test]
    fn test_performance_metrics() {
        let performance = PerformanceMetrics::new(
            0.1,  // execution_time
            0.8,  // cache_efficiency
            0.9,  // simd_utilization
            0.7,  // parallel_efficiency
            0.85, // memory_bandwidth
            OptimizationStrategy::SIMDVectorized,
        );

        let reward = performance.compute_reward(0.2); // baseline_time
        assert!(reward > 0.0); // Should be positive since we improved time

        let score = performance.performance_score();
        assert!(score > 0.0 && score <= 1.0);
    }

    #[test]
    fn test_optimization_strategies() {
        // Test all optimization strategies are accessible
        let strategies = [
            OptimizationStrategy::RowWiseCache,
            OptimizationStrategy::ColumnWiseLocality,
            OptimizationStrategy::BlockStructured,
            OptimizationStrategy::DiagonalOptimized,
            OptimizationStrategy::Hierarchical,
            OptimizationStrategy::StreamingCompute,
            OptimizationStrategy::SIMDVectorized,
            OptimizationStrategy::ParallelWorkStealing,
            OptimizationStrategy::AdaptiveHybrid,
        ];

        assert_eq!(strategies.len(), 9);

        // Test that they can be used as hash keys
        use std::collections::HashMap;
        let mut strategy_map = HashMap::new();
        for strategy in &strategies {
            strategy_map.insert(*strategy, 1.0);
        }
        assert_eq!(strategy_map.len(), 9);
    }

    #[test]
    fn test_rl_algorithms() {
        // Test all RL algorithms are accessible
        let algorithms = [
            RLAlgorithm::DQN,
            RLAlgorithm::PolicyGradient,
            RLAlgorithm::ActorCritic,
            RLAlgorithm::PPO,
            RLAlgorithm::SAC,
        ];

        assert_eq!(algorithms.len(), 5);

        // Test configuration with different algorithms
        for algorithm in &algorithms {
            let config = NeuralAdaptiveConfig::new().with_rl_algorithm(*algorithm);
            assert!(config.validate().is_ok());
        }
    }

    #[test]
    fn test_processor_state_serialization() {
        let config = NeuralAdaptiveConfig::lightweight();
        let processor = NeuralAdaptiveSparseProcessor::new(config);

        let state = processor.save_state();
        assert_eq!(state.total_operations, 0);
        assert!(!state.neural_network_params.is_empty());

        // Test that state can be loaded (this would require a mutable processor in real usage)
        // Just test the state structure for now
        assert!(state.current_exploration_rate > 0.0);
        assert!(state.current_exploration_rate <= 1.0);
    }
}
