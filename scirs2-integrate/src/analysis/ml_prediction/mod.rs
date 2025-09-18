//! Machine Learning Bifurcation Prediction Module
//!
//! This module provides advanced machine learning techniques for predicting
//! bifurcation points and classifying bifurcation types in dynamical systems.
//!
//! The module is organized into focused sub-modules for maintainability:
//! - `neural_network`: Core neural network components and implementations
//! - `training`: Training configuration and optimization algorithms
//! - `features`: Feature extraction and preprocessing configurations
//! - `uncertainty`: Uncertainty quantification and performance metrics
//! - `forecasting`: Time series forecasting for bifurcation prediction
//! - `ensemble`: Ensemble learning methods and meta-learners
//! - `monitoring`: Real-time monitoring and alerting systems
//! - `preprocessing`: Data preprocessing pipelines and quality checks

pub mod ensemble;
pub mod features;
pub mod forecasting;
pub mod monitoring;
pub mod neural_network;
pub mod preprocessing;
pub mod training;
pub mod uncertainty;

// Re-export all public types for backward compatibility
pub use neural_network::{
    ActivationFunction, BatchNormParams, BifurcationPrediction, BifurcationPredictionNetwork,
    ConnectionType, ModelParameters, NetworkArchitecture, SkipConnection, UncertaintyEstimate,
};

pub use training::{
    DataAugmentation, EarlyStoppingConfig, LearningRateSchedule, LossFunction, Optimizer,
    RegularizationConfig, TrainingConfiguration,
};

pub use features::{
    FeatureExtraction, FeatureNormalization, FrequencyFeatures, PhaseSpaceFeatures,
    StatisticalFeatures, TimeSeriesFeatures, TopologicalFeatures, WaveletType,
};

pub use uncertainty::{
    BayesianConfig, ConformalConfig, ConformityScore, DiversityMethod, EnsembleAggregation,
    EnsembleConfig, EpochMetrics, MCDropoutConfig, PerformanceMetrics, PriorParams, TestMetrics,
    UncertaintyQuantification, VariationalMethod,
};

pub use forecasting::{
    AnomalyDetectionConfig, AnomalyDetectionMethod, MultiStepStrategy,
    TimeSeriesBifurcationForecaster, TimeSeriesModel, TrendAnalysisConfig, TrendDetectionMethod,
};

pub use ensemble::{
    BaseClassifier, BifurcationEnsembleClassifier, CrossValidationConfig, DistanceMetric,
    EnsembleTrainingStrategy, FeatureSelectionConfig, FeatureSelectionMethod, KNNWeights,
    MetaLearner, SVMKernel, ScoreFunction,
};

pub use monitoring::{
    AccuracyMetrics, AdaptiveThresholdSystem, AlertAction, AlertMetrics, AlertSuppressionConfig,
    AlertSystemConfig, EscalationLevel, FeedbackMechanism, LatencyMetrics, LogFormat,
    MonitoringConfig, MonitoringEnsembleConfig, NotificationMethod, PerformanceTracker,
    RealTimeBifurcationMonitor, ResourceMetrics, ThresholdAdaptationMethod, VotingStrategy,
};

pub use preprocessing::{
    ConservationLaw, Constraint, FilterType, InterpolationMethod, OutlierDetectionMethod,
    PreprocessingPipeline, PreprocessingStep, QualityCheck, SmoothingMethod, StatisticalTest,
    ValidationRule,
};
