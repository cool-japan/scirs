//! AI optimization modules for computer vision pipelines
//!
//! This module has been refactored into focused sub-modules for better organization:
//! - `reinforcement_learning`: Q-learning optimization for parameter tuning
//! - `genetic_algorithms`: Advanced genetic algorithms with multi-objective optimization
//! - `neural_architecture_search`: Neural architecture search for processing pipelines
//! - `predictive_scaling`: Time series analysis for predictive resource scaling

pub mod genetic_algorithms;
pub mod neural_architecture_search;
pub mod predictive_scaling;
pub mod reinforcement_learning;

// Re-export main public API from reinforcement_learning module
pub use reinforcement_learning::{
    ActionDiscrete, AdjustmentAction, Experience, ParameterType, PerformanceMetric,
    RLLearningParams, RLParameterOptimizer, StateDiscrete,
};

// Re-export genetic algorithms functionality
pub use genetic_algorithms::{
    AdaptiveMutationStrategies, EliteArchives, GAParameters, GeneticPipelineOptimizer,
    MutationStrategy, PerformancePredictors, PipelineGenome,
};

// Re-export neural architecture search functionality
pub use neural_architecture_search::{
    AcquisitionFunction, ActivationType, ArchitecturePerformance, ArchitectureSearchSpace,
    AttentionType, ConnectionType, LayerType, NeuralArchitectureSearch, NormalizationType,
    PoolingType, ProcessingArchitecture, SearchStrategy,
};

// Re-export predictive scaling functionality
pub use predictive_scaling::{
    ModelType, PredictionModel, PredictiveScaler, ResourceRequirement, ResourceType, ScalingAction,
    ScalingPrediction, ScalingRecommendation, ScalingState, WorkloadMeasurement,
};
