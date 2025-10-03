//! Neural memory optimization module.
//!
//! This module implements cutting-edge memory optimization techniques using machine learning:
//! - Neural network-based cache miss prediction
//! - Reinforcement learning for prefetch optimization
//! - Adaptive memory compression with algorithm selection
//! - Pattern recognition for memory access optimization

// Module declarations
pub mod cache;
pub mod compression;
pub mod memory_intelligence;
pub mod numa;
pub mod patterns;
pub mod training;
pub mod types;

// Re-export commonly used types
pub use types::*;

// Re-export main components
pub use memory_intelligence::{
    AdvancedMemoryIntelligence, AdvancedMemoryOptimizationReport, ComplexityLevel,
    OptimizationCategory, OptimizationRecommendation,
};

// Re-export cache components
pub use cache::{
    BandwidthMeasurement, BandwidthMonitor, BandwidthSaturationPrediction, BottleneckType,
    CacheAccessPattern, CachePerformancePrediction, ConvolutionalLayer, DenseLayer, LstmLayer,
    NeuralCachePredictionModel, NeuralModelParameters, OptimizerType, PerformanceBottleneck,
    TrainingMetrics,
};

// Re-export compression components
pub use compression::{
    AdaptiveCompressionEngine, AttentionMechanism, BayesianNetwork, ClassificationNetwork,
    CompressionAlgorithm, CompressionConstraints, CompressionMetrics, CompressionQualityAssessor,
    CompressionSelectorNetwork, ConfidenceEstimator, FeatureExtractor, FeatureType,
    PerceptualFeatureType, PerceptualQualityModel, PriorDistribution, QualityMetric,
    QualityPredictionNetwork, SoftmaxLayer, UncertaintyQuantificationMethod, VariationalParameters,
};

// Re-export NUMA components
pub use numa::{
    MemoryAccessSample, MemoryAllocationStrategy, NumaNode, NumaOptimizationAction,
    NumaOptimizationPolicy, NumaOptimizationTrigger, NumaPerformanceMetrics,
    NumaPerformanceMonitor, NumaSuccessCriterion, NumaTopology, NumaTopologyOptimizer,
};

// Re-export pattern components
pub use patterns::{
    HashFunction, IndexParameters, LocalitySensitiveHashing, MemoryAccess, MemoryAccessPattern,
    PatternDatabase, PatternFeatures, PatternPerformance, PatternSimilarityIndex,
};

// Re-export training components
pub use training::{
    AdvancedMemoryLayout, AdvancedMemoryPatternLearning, BenchmarkResult, BenchmarkSuite,
    ClassificationHead, ConvolutionalPatternNetwork, EffortLevel, EmbeddingLayer, Experience,
    ExperienceReplayBuffer, FitnessEvaluator, FitnessMetric, GeneticAlgorithmParameters,
    GeneticLayoutOptimizer, MemoryBenchmark, OptimizationRecommendations, OptimizationType,
    PatternOptimization, PolicyNetwork, PoolingLayer, PoolingType, PrefetchStrategy, PrefetchType,
    QNetwork, RLLearningParameters, ReinforcementLearningAgent, SelectionMethod,
};

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_advanced_memory_intelligence_creation() {
        let memory_intelligence = AdvancedMemoryIntelligence::<f32>::new().unwrap();
        // Test successful creation - we can't easily test the internals due to Arc<Mutex<>>
        // but we can verify the struct was created without panicking
        drop(memory_intelligence);
    }

    #[test]
    fn test_cache_performance_prediction() {
        let memory_intelligence = AdvancedMemoryIntelligence::<f32>::new().unwrap();
        let workload = WorkloadCharacteristics {
            operation_types: vec![MemoryOperationType::MatrixMultiplication],
            datasizes: vec![TensorShape {
                dimensions: vec![100, 100],
                element_type: ElementType::F32,
                memory_layout: MemoryLayout::RowMajor,
            }],
            computation_intensity: 1.0,
            memory_intensity: 0.5,
        };
        let access_pattern = CacheAccessPattern::from_workload(&workload);

        let prediction = memory_intelligence.predict_cache_performance(&access_pattern);
        assert!(prediction.is_ok());

        let result = prediction.unwrap();
        assert!(result.hit_rate >= 0.0 && result.hit_rate <= 1.0);
        assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
    }

    #[test]
    fn test_compression_algorithm_selection() {
        let memory_intelligence = AdvancedMemoryIntelligence::<f32>::new().unwrap();
        let data = Array2::zeros((100, 100));
        let constraints = CompressionConstraints::default();

        let result = memory_intelligence.select_compression_algorithm(&data.view(), &constraints);
        assert!(result.is_ok());
    }

    #[test]
    fn test_numa_optimization() {
        let memory_intelligence = AdvancedMemoryIntelligence::<f32>::new().unwrap();
        let workload = WorkloadCharacteristics {
            operation_types: vec![MemoryOperationType::MatrixMultiplication],
            datasizes: vec![TensorShape {
                dimensions: vec![1000, 1000],
                element_type: ElementType::F32,
                memory_layout: MemoryLayout::RowMajor,
            }],
            computation_intensity: 2.0,
            memory_intensity: 1.0,
        };

        let result = memory_intelligence.optimize_numa_allocation(&workload);
        assert!(result.is_ok());
    }

    #[test]
    fn test_bandwidth_monitoring() {
        let memory_intelligence = AdvancedMemoryIntelligence::<f32>::new().unwrap();

        let result = memory_intelligence.monitor_bandwidth_saturation();
        assert!(result.is_ok());

        let prediction = result.unwrap();
        assert!(prediction.saturation_level >= 0.0 && prediction.saturation_level <= 1.0);
        assert!(prediction.confidence >= 0.0 && prediction.confidence <= 1.0);
    }

    #[test]
    fn test_comprehensive_analysis() {
        let memory_intelligence = AdvancedMemoryIntelligence::<f32>::new().unwrap();
        let workload = WorkloadCharacteristics {
            operation_types: vec![MemoryOperationType::MatrixMultiplication],
            datasizes: vec![TensorShape {
                dimensions: vec![500, 500],
                element_type: ElementType::F32,
                memory_layout: MemoryLayout::RowMajor,
            }],
            computation_intensity: 1.5,
            memory_intensity: 0.8,
        };
        let data = Array2::ones((500, 500));

        let result = memory_intelligence.comprehensive_analysis(&workload, &data.view());
        assert!(result.is_ok());

        let report = result.unwrap();
        assert!(report.optimization_score >= 0.0 && report.optimization_score <= 1.0);
        assert!(report.confidence >= 0.0 && report.confidence <= 1.0);
        assert!(!report.recommendations.is_empty());
    }

    #[test]
    fn test_neural_model_parameters() {
        let params = NeuralModelParameters::default();
        assert!(params.learning_rate > 0.0);
        assert!(params.batchsize > 0);
        assert!(params.validation_split > 0.0 && params.validation_split < 1.0);
    }

    #[test]
    fn test_compression_constraints() {
        let constraints = CompressionConstraints::default();
        assert!(constraints.min_compression_ratio >= 1.0);
        assert!(constraints.max_quality_loss >= 0.0 && constraints.max_quality_loss <= 1.0);
        assert!(constraints.memory_budget > 0);
    }

    #[test]
    fn test_genetic_algorithm_parameters() {
        let params = GeneticAlgorithmParameters::default();
        assert!(params.populationsize > 0);
        assert!(params.crossover_rate >= 0.0 && params.crossover_rate <= 1.0);
        assert!(params.mutation_rate >= 0.0 && params.mutation_rate <= 1.0);
        assert!(params.elitism_rate >= 0.0 && params.elitism_rate <= 1.0);
    }
}
