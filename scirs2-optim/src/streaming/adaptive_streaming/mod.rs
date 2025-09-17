//! Adaptive Streaming Optimization System
//!
//! This module provides a comprehensive adaptive streaming optimization system for machine learning
//! workflows. It includes drift detection, performance tracking, resource management, adaptive
//! buffering, meta-learning, and anomaly detection capabilities.
//!
//! # Key Components
//!
//! - **Configuration**: Centralized configuration management for all streaming components
//! - **Optimizer**: Core adaptive streaming optimizer with dynamic adaptation
//! - **Drift Detection**: Comprehensive concept drift detection using multiple methods
//! - **Performance Tracking**: Real-time performance monitoring and prediction
//! - **Resource Management**: Dynamic resource allocation and monitoring
//! - **Buffering**: Adaptive buffer management with quality-based retention
//! - **Meta-Learning**: Experience-based learning and strategy selection
//! - **Anomaly Detection**: Multi-method anomaly detection for streaming data
//!
//! # Example Usage
//!
//! ```rust
//! use scirs2_optim::streaming::adaptive_streaming::{
//!     AdaptiveStreamingOptimizer, StreamingConfig, DriftConfig,
//!     PerformanceConfig, ResourceConfig, BufferConfig
//! };
//! use ndarray::Array2;
//!
//! // Create configuration
//! let config = StreamingConfig::default();
//!
//! // Initialize optimizer
//! let mut optimizer = AdaptiveStreamingOptimizer::new(
//!     base_optimizer,
//!     config,
//! )?;
//!
//! // Process streaming data
//! for batch in data_stream {
//!     optimizer.update(&batch)?;
//! }
//! ```

// Core modules
pub mod config;
pub mod optimizer;
pub mod drift_detection;
pub mod performance;
pub mod resource_management;
pub mod buffering;
pub mod meta_learning;
pub mod anomaly_detection;

// Configuration re-exports
pub use config::{
    StreamingConfig, BufferConfig, DriftConfig, PerformanceConfig,
    ResourceConfig, MetaLearningConfig, AnomalyConfig, LearningRateConfig,
    // Buffer configuration
    BufferStrategy, EvictionStrategy, QualityMetric, QualityThreshold,
    // Drift configuration
    DriftDetectionMethod, StatisticalMethod, DistributionMethod,
    DriftSensitivity, AdaptationStrategy,
    // Performance configuration
    PerformanceMetric, TrendAnalysisMethod, PredictionMethod,
    PerformanceThreshold, OptimizationTarget,
    // Resource configuration
    ResourceType, AllocationStrategy, MonitoringLevel,
    ResourceBudget, ResourceConstraint,
    // Meta-learning configuration
    ExperienceReplayStrategy, StrategySelectionMethod,
    LearningStrategy, ExperienceType,
    // Anomaly configuration
    AnomalyDetectionMethod, EnsembleMethod, FalsePositiveHandling,
    AnomalyThreshold, AlertLevel,
    // Learning rate configuration
    LearningRateSchedule, AdaptationTrigger, RateAdjustmentMethod
};

// Core optimizer re-exports
pub use optimizer::{
    AdaptiveStreamingOptimizer, OptimizationState, AdaptationContext,
    StreamingDataPoint, OptimizationResult, AdaptationDecision,
    LearningPhase, OptimizationPhase
};

// Drift detection re-exports
pub use drift_detection::{
    EnhancedDriftDetector, DriftDetectionResult, DriftAlert,
    DriftStatistics, DriftContext, StatisticalTest, DistributionComparator,
    // Statistical tests
    KolmogorovSmirnovTest, ChiSquareTest, AndersonDarlingTest,
    MannWhitneyUTest, KruskalWallisTest, WilcoxonSignedRankTest,
    // Distribution comparators
    EarthMoverDistance, JensenShannonDivergence, KullbackLeiblerDivergence,
    WassersteinDistance, HellingerDistance, BhattacharyyaDistance,
    // Drift types
    DriftType, DriftIntensity, DriftPattern
};

// Performance tracking re-exports
pub use performance::{
    PerformanceTracker, PerformanceSnapshot, PerformanceTrend,
    PerformanceTrendAnalyzer, PerformancePredictor, PerformancePrediction,
    // Metrics
    PerformanceMetrics, ConvergenceMetrics, StabilityMetrics,
    EfficiencyMetrics, QualityMetrics,
    // Analysis
    TrendAnalysis, SeasonalityAnalysis, CyclicalAnalysis,
    NoiseAnalysis, OutlierAnalysis,
    // Prediction
    TrendPrediction, VariabilityPrediction, ConfidenceInterval,
    PredictionAccuracy, PredictionUncertainty
};

// Resource management re-exports
pub use resource_management::{
    ResourceManager, ResourceUsage, ResourceMonitor,
    ResourceAllocator, ResourceOptimizer, ResourcePredictor,
    // Usage types
    CPUUsage, MemoryUsage, DiskUsage, NetworkUsage,
    GPUUsage, BandwidthUsage, StorageUsage,
    // Allocation
    AllocationResult, AllocationRequest, AllocationPriority,
    ResourceAllocation, DynamicAllocation,
    // Monitoring
    ResourceAlert, UsagePattern, ResourceTrend,
    BottleneckDetection, CapacityPlanning,
    // Optimization
    ResourceOptimization, CostOptimization, PerformanceOptimization,
    EfficiencyOptimization, ScalingDecision
};

// Buffering re-exports
pub use buffering::{
    AdaptiveBuffer, BufferQualityMetrics, BufferStatistics,
    BufferOptimizer, QualityBasedRetention, PrioritizedDataPoint,
    // Buffer types
    PrimaryBuffer, SecondaryBuffer, EvictionBuffer,
    TemporaryBuffer, ArchiveBuffer,
    // Quality metrics
    DataQuality, InformationContent, NoveltyScore,
    RelevanceScore, DiversityScore, RecencyScore,
    // Strategies
    AdaptiveEviction, QualityFiltering, DiversityMaintenance,
    TemporalRelevance, InformationTheoreticSelection,
    // Statistics
    BufferEfficiency, RetentionStats, QualityDistribution,
    EvictionStats, BufferTurnover
};

// Meta-learning re-exports
pub use meta_learning::{
    MetaLearner, ExperienceBuffer, MetaModel, StrategySelector,
    PerformancePredictor as MetaPerformancePredictor,
    // Experience management
    Experience, ExperienceEntry, ExperienceReplay,
    ExperienceAnalysis, ExperienceDistillation,
    // Strategy selection
    StrategyPortfolio, StrategyPerformance, StrategyAdaptation,
    MultiArmedBandit, ContextualBandit, ThompsonSampling,
    EpsilonGreedy, UpperConfidenceBound,
    // Meta-model
    MetaFeatures, ContextualFeatures, PerformanceFeatures,
    MetaPrediction, TransferLearning, FewShotLearning,
    // Learning strategies
    OnlineLearning, IncrementalLearning, ContinualLearning,
    LifelongLearning, AdaptiveLearning
};

// Anomaly detection re-exports
pub use anomaly_detection::{
    AnomalyDetector, AnomalyResult, AnomalyAlert,
    EnsembleAnomalyDetector, StatisticalAnomalyDetector,
    MLAnomalyDetector, FalsePositiveTracker,
    // Statistical detectors
    ZScoreDetector, ModifiedZScoreDetector, IQRDetector,
    GrubbsTestDetector, DixonTestDetector, ChauvenetsDetector,
    // ML detectors
    IsolationForestDetector, OneClassSVMDetector, LocalOutlierFactorDetector,
    EllipticEnvelopeDetector, AutoencoderDetector, VAEDetector,
    // Ensemble methods
    VotingEnsemble, WeightedEnsemble, StackingEnsemble,
    BaggingEnsemble, BoostingEnsemble,
    // Anomaly types
    AnomalyType, AnomalySeverity, AnomalyConfidence,
    OutlierType, NoveltyType, DriftAnomaly,
    // False positive handling
    FalsePositiveFilter, ConfidenceCalibration,
    ThresholdAdaptation, ContextualFiltering
};

// Utility types and traits
pub use config::{Configurable, Validatable, Serializable};
pub use optimizer::{Optimizable, Adaptable, Streamable};
pub use drift_detection::{DriftDetectable, StatisticalTestable};
pub use performance::{PerformanceTrackable, Predictable, Analyzable};
pub use resource_management::{ResourceManageable, Monitorable, Allocatable};
pub use buffering::{Bufferable, QualityAssessable, Retainable};
pub use meta_learning::{MetaLearnable, ExperienceReplayable, StrategySelectable};
pub use anomaly_detection::{AnomalyDetectable, EnsembleDetectable, FalsePositiveHandleable};

// Common error types
pub use config::ConfigError;
pub use optimizer::OptimizerError;
pub use drift_detection::DriftDetectionError;
pub use performance::PerformanceError;
pub use resource_management::ResourceError;
pub use buffering::BufferError;
pub use meta_learning::MetaLearningError;
pub use anomaly_detection::AnomalyDetectionError;

// Type aliases for convenience
pub type StreamingOptimizer<O, A, D> = AdaptiveStreamingOptimizer<O, A, D>;
pub type StreamingResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;
pub type DriftResult<T> = Result<T, DriftDetectionError>;
pub type PerformanceResult<T> = Result<T, PerformanceError>;
pub type ResourceResult<T> = Result<T, ResourceError>;
pub type BufferResult<T> = Result<T, BufferError>;
pub type MetaResult<T> = Result<T, MetaLearningError>;
pub type AnomalyResult<T> = Result<T, AnomalyDetectionError>;

// Feature flags for conditional compilation
#[cfg(feature = "gpu")]
pub use optimizer::gpu_optimizer::GPUStreamingOptimizer;

#[cfg(feature = "distributed")]
pub use optimizer::distributed_optimizer::DistributedStreamingOptimizer;

#[cfg(feature = "visualization")]
pub use performance::visualization::{PerformanceVisualizer, DriftVisualizer};

#[cfg(feature = "benchmarking")]
pub use performance::benchmarking::{StreamingBenchmark, PerformanceBenchmark};

// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const AUTHORS: &str = env!("CARGO_PKG_AUTHORS");
pub const DESCRIPTION: &str = env!("CARGO_PKG_DESCRIPTION");

/// Creates a new adaptive streaming optimizer with default configuration
pub fn create_default_optimizer<O, A, D>() -> StreamingResult<AdaptiveStreamingOptimizer<O, A, D>>
where
    O: Clone + Send + Sync,
    A: ndarray::ScalarOperand + Clone + Default + Send + Sync + 'static,
    D: ndarray::Dimension,
{
    let config = StreamingConfig::default();
    AdaptiveStreamingOptimizer::new_with_config(config)
}

/// Creates a new adaptive streaming optimizer with custom configuration
pub fn create_optimizer_with_config<O, A, D>(
    config: StreamingConfig,
) -> StreamingResult<AdaptiveStreamingOptimizer<O, A, D>>
where
    O: Clone + Send + Sync,
    A: ndarray::ScalarOperand + Clone + Default + Send + Sync + 'static,
    D: ndarray::Dimension,
{
    AdaptiveStreamingOptimizer::new_with_config(config)
}

/// Validates a streaming configuration
pub fn validate_config(config: &StreamingConfig) -> Result<(), ConfigError> {
    config.validate()
}

/// Creates a comprehensive test configuration for development and testing
#[cfg(test)]
pub fn create_test_config() -> StreamingConfig {
    StreamingConfig {
        buffer_config: BufferConfig::test_config(),
        drift_config: DriftConfig::test_config(),
        performance_config: PerformanceConfig::test_config(),
        resource_config: ResourceConfig::test_config(),
        meta_learning_config: MetaLearningConfig::test_config(),
        anomaly_config: AnomalyConfig::test_config(),
        learning_rate_config: LearningRateConfig::test_config(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_exports() {
        // Test that all major types are accessible
        let _config = StreamingConfig::default();
        let _buffer_config = BufferConfig::default();
        let _drift_config = DriftConfig::default();
        let _performance_config = PerformanceConfig::default();
        let _resource_config = ResourceConfig::default();
        let _meta_config = MetaLearningConfig::default();
        let _anomaly_config = AnomalyConfig::default();
    }

    #[test]
    fn test_config_validation() {
        let config = create_test_config();
        assert!(validate_config(&config).is_ok());
    }

    #[test]
    fn test_version_info() {
        assert!(!VERSION.is_empty());
        assert!(!AUTHORS.is_empty());
        assert!(!DESCRIPTION.is_empty());
    }
}