//! AI-based optimization for computer vision pipelines
//!
//! This module provides AI-driven optimization techniques for automatically
//! tuning computer vision processing parameters using reinforcement learning,
//! genetic algorithms, neural architecture search, and predictive scaling.
//!
//! **Note**: This module has been refactored into smaller, focused sub-modules
//! for better maintainability. The original API is preserved for backward compatibility.

// Import the modular implementation
#[path = "ai_optimization_modules/mod.rs"]
pub mod ai_optimization_modules;

// Re-export types for backward compatibility
pub use ai_optimization_modules::{
    // Neural Architecture Search
    AcquisitionFunction,
    // Reinforcement Learning
    ActionDiscrete,
    ActivationType,
    // Genetic Algorithms
    AdaptiveMutationStrategies,
    AdjustmentAction,
    ArchitecturePerformance,
    ArchitectureSearchSpace,
    AttentionType,
    ConnectionType,
    EliteArchives,
    Experience,
    GAParameters,
    GeneticPipelineOptimizer,
    LayerType,
    // Predictive Scaling
    ModelType,
    MutationStrategy,
    NeuralArchitectureSearch,
    NormalizationType,
    ParameterType,
    PerformanceMetric,
    PerformancePredictors,
    PipelineGenome,

    PoolingType,
    PredictionModel,
    PredictiveScaler,
    ProcessingArchitecture,
    RLLearningParams,
    RLParameterOptimizer,
    ResourceRequirement,
    ResourceType,
    ScalingAction,
    ScalingPrediction,
    ScalingRecommendation,
    ScalingState,
    SearchStrategy,

    StateDiscrete,

    WorkloadMeasurement,
};

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::time::{Duration, Instant};

    #[test]
    fn test_rl_parameter_optimizer() {
        let mut optimizer = RLParameterOptimizer::new();

        let state = StateDiscrete::default();
        let action_ = optimizer.select_action(&state);

        assert!(optimizer.action_space.contains(&action_));
    }

    #[test]
    fn test_genetic_optimizer() {
        let mut parameter_ranges = HashMap::new();
        parameter_ranges.insert("blur_sigma".to_string(), (0.1, 2.0));
        parameter_ranges.insert("threshold".to_string(), (0.01, 0.5));

        let mut optimizer = GeneticPipelineOptimizer::new(parameter_ranges);

        // Test fitness evaluation
        optimizer.evaluate_population(|genome| {
            // Simple fitness function
            genome.genes.get("blur_sigma").unwrap_or(&0.0)
                + genome.genes.get("threshold").unwrap_or(&0.0)
        });

        assert!(optimizer.population[0].fitness >= 0.0);
    }

    #[test]
    fn test_neural_architecture_search() {
        let _searchspace = ArchitectureSearchSpace {
            layer_types: vec![
                LayerType::Convolution {
                    kernel_size: 3,
                    stride: 1,
                },
                LayerType::Pooling {
                    pool_type: PoolingType::Max,
                    size: 2,
                },
            ],
            depth_range: (2, 5),
            width_range: (32, 128),
            activations: vec![ActivationType::ReLU],
            connections: vec![ConnectionType::Sequential],
        };

        let mut nas = NeuralArchitectureSearch::new(_searchspace, SearchStrategy::Random);

        let candidates = nas.generate_candidates(5);
        assert_eq!(candidates.len(), 5);

        for candidate in &candidates {
            assert!(candidate.layers.len() >= 2 && candidate.layers.len() <= 5);
        }
    }

    #[test]
    fn test_predictive_scaler() {
        let mut scaler = PredictiveScaler::new(300.0); // 5 minute prediction window

        // Record some workload measurements
        for i in 0..10 {
            let measurement = WorkloadMeasurement {
                timestamp: Instant::now(),
                processing_load: (i as f64) / 10.0,
                input_complexity: 0.5,
                required_resources: ResourceRequirement {
                    cpu_cores: 2.0,
                    memory_mb: 1024.0,
                    gpu_utilization: 0.5,
                },
            };
            scaler.record_workload(measurement);
        }

        // Generate predictions
        let horizons = vec![
            Duration::from_secs(60),
            Duration::from_secs(300),
            Duration::from_secs(600),
        ];

        let predictions = scaler.generate_predictions(horizons);
        assert_eq!(predictions.len(), 3);

        for prediction in &predictions {
            assert!(prediction.confidence >= 0.0 && prediction.confidence <= 1.0);
        }
    }
}
