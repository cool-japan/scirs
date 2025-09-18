//! Integration modules for advanced computer vision processing
//!
//! This module has been refactored into focused sub-modules for better organization:
//! - `neural_quantum_hybrid`: Neural-quantum hybrid processing core
//! - `cross_module_coordination`: Cross-module coordination interfaces
//! - `data_structures`: Data structures and result types
//! - `processing_core`: High-level processing functions

pub mod cross_module_coordination;
pub mod data_structures;
pub mod neural_quantum_hybrid;
pub mod processing_core;

// Re-export main public API from neural_quantum_hybrid module
pub use neural_quantum_hybrid::{
    AdvancedProcessingResult, AlignmentStrategy, BehaviorPattern, ComplexityMetrics,
    DomainAdaptationMethod, EfficiencyMetrics, EmergenceIndicator, EmergentBehaviorDetector,
    FeatureAlignmentConfig, FusionStrategy, HybridFusionParameters, ImpactMeasurement,
    ImpactTracker, MetaLearningAlgorithm, MetaLearningSystem, ModificationAction,
    ModificationEvent, ModificationRule, NeuralQuantumHybridProcessor, PerformanceTracker,
    RealtimeIndicators, SafetyConstraints, SafetyLevel, SelfModificationEngine,
    TaskAdaptationParams, TransferLearningConfig, TriggerCondition, VisionResult,
};

// Re-export cross-module coordination functionality
pub use cross_module_coordination::{
    AdvancedCrossModuleCoordinator, AdvancedInitializationReport, AdvancedResourceManager,
    AllocationDecision, ClusteringCoordinationInterface, ClusteringPerformanceFeedback,
    ComputationalResources, CrossModuleFewShotLearner, CrossModuleLearningEpisode,
    CrossModulePerformanceTracker, GlobalAdvancedOptimizer, GlobalResourceStrategy,
    MetaLearningPerformance, ModulePerformanceMetrics, MultiObjectiveTargets,
    NeuralNetworkInterface, NeuralPerformanceMetric, PerformanceBottleneck,
    PerformancePredictionModel, ReallocationTrigger, ResourceAllocation, SpatialPerformanceMetric,
    SpatialProcessingInterface, SystemPerformanceMetrics, UnifiedMetaLearningSystem,
};

// Re-export data structures functionality
pub use data_structures::{
    AdvancedAdvancedProcessingResult, AdvancedInputData, AdvancedModeStatus,
    AdvancedPerformanceMetrics, CrossModuleAdvancedProcessingResult, CrossModuleFusedResult,
    EmergentBehaviorDetection, FusionQualityIndicators, NeuromorphicProcessingMetrics,
    PerformanceMetrics, ProcessingTimingBreakdown, QualityAssuranceMetrics,
    QuantumProcessingMetrics, ResourceUsageStatistics, UncertaintyQuantification,
};

// Re-export processing core functionality
pub use processing_core::{
    batch_process_advanced, process_with_advanced_mode, realtime_advanced_stream,
};
