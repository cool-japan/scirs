//! Configuration management for adaptive streaming optimization
//!
//! This module provides comprehensive configuration types and settings for
//! streaming optimization scenarios, including drift detection parameters,
//! performance tracking settings, resource allocation strategies, and
//! adaptive buffer management configurations.

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Main configuration for adaptive streaming optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingConfig {
    /// Buffer management configuration
    pub buffer_config: BufferConfig,
    /// Drift detection configuration
    pub drift_config: DriftConfig,
    /// Performance tracking configuration
    pub performance_config: PerformanceConfig,
    /// Resource management configuration
    pub resource_config: ResourceConfig,
    /// Meta-learning configuration
    pub meta_learning_config: MetaLearningConfig,
    /// Anomaly detection configuration
    pub anomaly_config: AnomalyConfig,
    /// Learning rate adaptation configuration
    pub learning_rate_config: LearningRateConfig,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            buffer_config: BufferConfig::default(),
            drift_config: DriftConfig::default(),
            performance_config: PerformanceConfig::default(),
            resource_config: ResourceConfig::default(),
            meta_learning_config: MetaLearningConfig::default(),
            anomaly_config: AnomalyConfig::default(),
            learning_rate_config: LearningRateConfig::default(),
        }
    }
}

/// Buffer management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BufferConfig {
    /// Initial buffer size
    pub initial_size: usize,
    /// Maximum buffer size
    pub max_size: usize,
    /// Minimum buffer size
    pub min_size: usize,
    /// Buffer size adaptation strategy
    pub size_strategy: BufferSizeStrategy,
    /// Quality threshold for buffer processing
    pub quality_threshold: f64,
    /// Enable adaptive buffer sizing
    pub enable_adaptive_sizing: bool,
    /// Buffer processing timeout
    pub processing_timeout: Duration,
    /// Memory limit for buffer in MB
    pub memory_limit_mb: usize,
}

impl Default for BufferConfig {
    fn default() -> Self {
        Self {
            initial_size: 1000,
            max_size: 10000,
            min_size: 100,
            size_strategy: BufferSizeStrategy::Adaptive,
            quality_threshold: 0.8,
            enable_adaptive_sizing: true,
            processing_timeout: Duration::from_secs(30),
            memory_limit_mb: 512,
        }
    }
}

/// Buffer sizing strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BufferSizeStrategy {
    /// Fixed buffer size
    Fixed,
    /// Linear growth/shrinkage
    Linear { growth_rate: f64 },
    /// Exponential growth/shrinkage
    Exponential { base: f64 },
    /// Adaptive based on performance
    Adaptive,
    /// Based on resource availability
    ResourceBased,
}

/// Drift detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftConfig {
    /// Enable drift detection
    pub enable_detection: bool,
    /// Drift detection method
    pub detection_method: DriftDetectionMethod,
    /// Sensitivity for drift detection
    pub sensitivity: f64,
    /// Minimum samples before drift detection
    pub min_samples: usize,
    /// Warning threshold for gradual drift
    pub warning_threshold: f64,
    /// Drift threshold for concept drift
    pub drift_threshold: f64,
    /// Window size for statistical tests
    pub window_size: usize,
    /// Enable false positive tracking
    pub enable_false_positive_tracking: bool,
    /// Statistical significance level
    pub significance_level: f64,
    /// Adaptation speed after drift detection
    pub adaptation_speed: f64,
}

impl Default for DriftConfig {
    fn default() -> Self {
        Self {
            enable_detection: true,
            detection_method: DriftDetectionMethod::Statistical(StatisticalMethod::ADWIN),
            sensitivity: 0.05,
            min_samples: 30,
            warning_threshold: 0.8,
            drift_threshold: 1.2,
            window_size: 1000,
            enable_false_positive_tracking: true,
            significance_level: 0.05,
            adaptation_speed: 0.1,
        }
    }
}

/// Drift detection methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DriftDetectionMethod {
    /// Statistical methods for drift detection
    Statistical(StatisticalMethod),
    /// Distribution-based methods
    Distribution(DistributionMethod),
    /// Model-based drift detection
    ModelBased(ModelType),
    /// Ensemble of multiple methods
    Ensemble {
        methods: Vec<DriftDetectionMethod>,
        voting_strategy: VotingStrategy,
    },
}

/// Statistical methods for drift detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StatisticalMethod {
    /// Adaptive Windowing (ADWIN)
    ADWIN,
    /// Drift Detection Method (DDM)
    DDM,
    /// Early Drift Detection Method (EDDM)
    EDDM,
    /// Page Hinkley test
    PageHinkley,
    /// CUSUM test
    CUSUM,
    /// Kolmogorov-Smirnov test
    KolmogorovSmirnov,
    /// Mann-Whitney U test
    MannWhitneyU,
}

/// Distribution-based drift detection methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DistributionMethod {
    /// Kullback-Leibler divergence
    KLDivergence,
    /// Jensen-Shannon divergence
    JSDivergence,
    /// Hellinger distance
    HellingerDistance,
    /// Wasserstein distance
    WassersteinDistance,
    /// Earth Mover's Distance
    EarthMoverDistance,
}

/// Model types for drift detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelType {
    /// Linear model comparison
    Linear,
    /// Neural network based
    NeuralNetwork,
    /// Decision tree based
    DecisionTree,
    /// Ensemble methods
    Ensemble,
}

/// Voting strategies for ensemble drift detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VotingStrategy {
    /// Majority voting
    Majority,
    /// Weighted voting
    Weighted { weights: Vec<f64> },
    /// Unanimous decision
    Unanimous,
    /// Threshold-based (minimum number of votes)
    Threshold { min_votes: usize },
}

/// Performance tracking configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Enable performance tracking
    pub enable_tracking: bool,
    /// Performance metrics to track
    pub metrics: Vec<PerformanceMetricType>,
    /// Performance history size
    pub history_size: usize,
    /// Performance evaluation frequency
    pub evaluation_frequency: usize,
    /// Enable trend analysis
    pub enable_trend_analysis: bool,
    /// Trend window size
    pub trend_window_size: usize,
    /// Enable performance prediction
    pub enable_prediction: bool,
    /// Prediction horizon (steps ahead)
    pub prediction_horizon: usize,
    /// Performance baseline update frequency
    pub baseline_update_frequency: usize,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            enable_tracking: true,
            metrics: vec![
                PerformanceMetricType::Loss,
                PerformanceMetricType::Accuracy,
                PerformanceMetricType::Convergence,
                PerformanceMetricType::ResourceUsage,
            ],
            history_size: 1000,
            evaluation_frequency: 10,
            enable_trend_analysis: true,
            trend_window_size: 100,
            enable_prediction: true,
            prediction_horizon: 10,
            baseline_update_frequency: 100,
        }
    }
}

/// Types of performance metrics to track
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PerformanceMetricType {
    /// Loss function value
    Loss,
    /// Accuracy metric
    Accuracy,
    /// Convergence rate
    Convergence,
    /// Resource usage efficiency
    ResourceUsage,
    /// Gradient norm
    GradientNorm,
    /// Parameter updates magnitude
    ParameterUpdates,
    /// Learning rate effectiveness
    LearningRateEffectiveness,
    /// Custom metric
    Custom(String),
}

/// Resource management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceConfig {
    /// Maximum memory usage in MB
    pub max_memory_mb: usize,
    /// Maximum CPU usage percentage
    pub max_cpu_percent: f64,
    /// Resource allocation strategy
    pub allocation_strategy: ResourceAllocationStrategy,
    /// Enable dynamic resource allocation
    pub enable_dynamic_allocation: bool,
    /// Resource monitoring frequency
    pub monitoring_frequency: Duration,
    /// Resource budget constraints
    pub budget_constraints: ResourceBudgetConstraints,
    /// Enable resource prediction
    pub enable_resource_prediction: bool,
    /// Resource cleanup threshold
    pub cleanup_threshold: f64,
}

impl Default for ResourceConfig {
    fn default() -> Self {
        Self {
            max_memory_mb: 2048,
            max_cpu_percent: 80.0,
            allocation_strategy: ResourceAllocationStrategy::Adaptive,
            enable_dynamic_allocation: true,
            monitoring_frequency: Duration::from_secs(10),
            budget_constraints: ResourceBudgetConstraints::default(),
            enable_resource_prediction: true,
            cleanup_threshold: 0.9,
        }
    }
}

/// Resource allocation strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResourceAllocationStrategy {
    /// Static allocation
    Static,
    /// Dynamic allocation based on demand
    Dynamic,
    /// Adaptive allocation based on performance
    Adaptive,
    /// Proportional allocation
    Proportional,
    /// Priority-based allocation
    PriorityBased,
}

/// Resource budget constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceBudgetConstraints {
    /// Memory budget in MB
    pub memory_budget_mb: usize,
    /// CPU budget as percentage
    pub cpu_budget_percent: f64,
    /// Time budget per operation
    pub time_budget: Duration,
    /// Enable strict budget enforcement
    pub strict_enforcement: bool,
    /// Budget violation penalty
    pub violation_penalty: f64,
}

impl Default for ResourceBudgetConstraints {
    fn default() -> Self {
        Self {
            memory_budget_mb: 1024,
            cpu_budget_percent: 60.0,
            time_budget: Duration::from_secs(60),
            strict_enforcement: false,
            violation_penalty: 0.1,
        }
    }
}

/// Meta-learning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaLearningConfig {
    /// Enable meta-learning
    pub enable_meta_learning: bool,
    /// Meta-learning algorithm
    pub algorithm: MetaAlgorithm,
    /// Experience buffer size
    pub experience_buffer_size: usize,
    /// Meta-learning update frequency
    pub update_frequency: usize,
    /// Learning rate for meta-learning
    pub meta_learning_rate: f64,
    /// Exploration rate for meta-learning
    pub exploration_rate: f64,
    /// Meta-model complexity
    pub model_complexity: MetaModelComplexity,
    /// Enable transfer learning
    pub enable_transfer_learning: bool,
    /// Experience replay configuration
    pub replay_config: ExperienceReplayConfig,
}

impl Default for MetaLearningConfig {
    fn default() -> Self {
        Self {
            enable_meta_learning: true,
            algorithm: MetaAlgorithm::ModelAgnosticMetaLearning,
            experience_buffer_size: 10000,
            update_frequency: 100,
            meta_learning_rate: 0.001,
            exploration_rate: 0.1,
            model_complexity: MetaModelComplexity::Medium,
            enable_transfer_learning: true,
            replay_config: ExperienceReplayConfig::default(),
        }
    }
}

/// Meta-learning algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetaAlgorithm {
    /// Model-Agnostic Meta-Learning (MAML)
    ModelAgnosticMetaLearning,
    /// Learning to Learn by Gradient Descent by Gradient Descent
    LearningToLearn,
    /// Meta-SGD
    MetaSGD,
    /// Reptile
    Reptile,
    /// Online Meta-Learning
    OnlineMetaLearning,
    /// Continual Meta-Learning
    ContinualMetaLearning,
}

/// Meta-model complexity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetaModelComplexity {
    /// Low complexity (simple linear models)
    Low,
    /// Medium complexity (small neural networks)
    Medium,
    /// High complexity (large neural networks)
    High,
    /// Adaptive complexity based on performance
    Adaptive,
}

/// Experience replay configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperienceReplayConfig {
    /// Enable prioritized experience replay
    pub enable_prioritized_replay: bool,
    /// Priority calculation method
    pub priority_method: PriorityMethod,
    /// Replay batch size
    pub batch_size: usize,
    /// Replay frequency
    pub replay_frequency: usize,
    /// Experience importance sampling
    pub importance_sampling: bool,
}

impl Default for ExperienceReplayConfig {
    fn default() -> Self {
        Self {
            enable_prioritized_replay: true,
            priority_method: PriorityMethod::TDError,
            batch_size: 32,
            replay_frequency: 10,
            importance_sampling: true,
        }
    }
}

/// Priority calculation methods for experience replay
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PriorityMethod {
    /// Temporal Difference error
    TDError,
    /// Surprise-based priority
    Surprise,
    /// Gradient magnitude
    GradientMagnitude,
    /// Loss improvement
    LossImprovement,
    /// Random priority
    Random,
}

/// Anomaly detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyConfig {
    /// Enable anomaly detection
    pub enable_detection: bool,
    /// Anomaly detection method
    pub detection_method: AnomalyDetectionMethod,
    /// Anomaly threshold
    pub threshold: f64,
    /// Window size for anomaly detection
    pub window_size: usize,
    /// Enable adaptive thresholding
    pub enable_adaptive_threshold: bool,
    /// False positive rate tolerance
    pub false_positive_rate: f64,
    /// Contamination rate assumption
    pub contamination_rate: f64,
    /// Response strategy for detected anomalies
    pub response_strategy: AnomalyResponseStrategy,
}

impl Default for AnomalyConfig {
    fn default() -> Self {
        Self {
            enable_detection: true,
            detection_method: AnomalyDetectionMethod::StatisticalOutlier,
            threshold: 2.0,
            window_size: 100,
            enable_adaptive_threshold: true,
            false_positive_rate: 0.05,
            contamination_rate: 0.1,
            response_strategy: AnomalyResponseStrategy::Adaptive,
        }
    }
}

/// Anomaly detection methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnomalyDetectionMethod {
    /// Statistical outlier detection
    StatisticalOutlier,
    /// Isolation Forest
    IsolationForest,
    /// One-Class SVM
    OneClassSVM,
    /// Local Outlier Factor
    LocalOutlierFactor,
    /// Autoencoders
    Autoencoder,
    /// Ensemble methods
    Ensemble,
}

/// Response strategies for detected anomalies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnomalyResponseStrategy {
    /// Ignore anomalies
    Ignore,
    /// Filter out anomalous data
    Filter,
    /// Adapt model to handle anomalies
    Adaptive,
    /// Reset to safe state
    Reset,
    /// Custom response strategy
    Custom(String),
}

/// Learning rate adaptation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningRateConfig {
    /// Initial learning rate
    pub initial_rate: f64,
    /// Minimum learning rate
    pub min_rate: f64,
    /// Maximum learning rate
    pub max_rate: f64,
    /// Learning rate adaptation strategy
    pub adaptation_strategy: LearningRateAdaptationStrategy,
    /// Adaptation frequency
    pub adaptation_frequency: usize,
    /// Performance sensitivity for adaptation
    pub performance_sensitivity: f64,
    /// Enable cyclical learning rates
    pub enable_cyclical_rates: bool,
    /// Cycle configuration
    pub cycle_config: CyclicalRateConfig,
}

impl Default for LearningRateConfig {
    fn default() -> Self {
        Self {
            initial_rate: 0.001,
            min_rate: 1e-6,
            max_rate: 0.1,
            adaptation_strategy: LearningRateAdaptationStrategy::PerformanceBased,
            adaptation_frequency: 10,
            performance_sensitivity: 0.1,
            enable_cyclical_rates: false,
            cycle_config: CyclicalRateConfig::default(),
        }
    }
}

/// Learning rate adaptation strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LearningRateAdaptationStrategy {
    /// Fixed learning rate
    Fixed,
    /// Step decay
    StepDecay { decay_rate: f64, decay_steps: usize },
    /// Exponential decay
    ExponentialDecay { decay_rate: f64 },
    /// Performance-based adaptation
    PerformanceBased,
    /// Gradient-based adaptation
    GradientBased,
    /// Adaptive learning rate (AdaGrad-style)
    Adaptive,
    /// Cosine annealing
    CosineAnnealing { t_max: usize },
}

/// Cyclical learning rate configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CyclicalRateConfig {
    /// Base learning rate
    pub base_rate: f64,
    /// Maximum learning rate in cycle
    pub max_rate: f64,
    /// Cycle length (number of steps)
    pub cycle_length: usize,
    /// Cycle mode
    pub cycle_mode: CycleMode,
    /// Scale function for cycle
    pub scale_function: ScaleFunction,
}

impl Default for CyclicalRateConfig {
    fn default() -> Self {
        Self {
            base_rate: 0.0001,
            max_rate: 0.001,
            cycle_length: 1000,
            cycle_mode: CycleMode::Triangular,
            scale_function: ScaleFunction::Linear,
        }
    }
}

/// Cyclical learning rate cycle modes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CycleMode {
    /// Triangular cycle
    Triangular,
    /// Triangular2 (amplitude scales)
    Triangular2,
    /// Exponential range
    ExponentialRange,
    /// Custom cycle pattern
    Custom(String),
}

/// Scale functions for cyclical learning rates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScaleFunction {
    /// Linear scaling
    Linear,
    /// Exponential scaling
    Exponential { factor: f64 },
    /// Polynomial scaling
    Polynomial { power: f64 },
    /// Custom scaling function
    Custom(String),
}

/// Validation methods for configuration
impl StreamingConfig {
    /// Validates the configuration for consistency and feasibility
    pub fn validate(&self) -> Result<(), String> {
        // Validate buffer configuration
        if self.buffer_config.max_size < self.buffer_config.min_size {
            return Err("Buffer max_size must be >= min_size".to_string());
        }

        if self.buffer_config.initial_size < self.buffer_config.min_size
            || self.buffer_config.initial_size > self.buffer_config.max_size
        {
            return Err("Buffer initial_size must be between min_size and max_size".to_string());
        }

        // Validate drift configuration
        if self.drift_config.sensitivity <= 0.0 || self.drift_config.sensitivity > 1.0 {
            return Err("Drift sensitivity must be in (0, 1]".to_string());
        }

        if self.drift_config.warning_threshold >= self.drift_config.drift_threshold {
            return Err("Drift warning_threshold must be < drift_threshold".to_string());
        }

        // Validate learning rate configuration
        if self.learning_rate_config.min_rate >= self.learning_rate_config.max_rate {
            return Err("Learning rate min_rate must be < max_rate".to_string());
        }

        if self.learning_rate_config.initial_rate < self.learning_rate_config.min_rate
            || self.learning_rate_config.initial_rate > self.learning_rate_config.max_rate
        {
            return Err(
                "Learning rate initial_rate must be between min_rate and max_rate".to_string(),
            );
        }

        // Validate resource configuration
        if self.resource_config.max_cpu_percent <= 0.0
            || self.resource_config.max_cpu_percent > 100.0
        {
            return Err("Resource max_cpu_percent must be in (0, 100]".to_string());
        }

        // Validate meta-learning configuration
        if self.meta_learning_config.meta_learning_rate <= 0.0 {
            return Err("Meta-learning rate must be > 0".to_string());
        }

        if self.meta_learning_config.exploration_rate < 0.0
            || self.meta_learning_config.exploration_rate > 1.0
        {
            return Err("Meta-learning exploration_rate must be in [0, 1]".to_string());
        }

        Ok(())
    }

    /// Creates a configuration optimized for low-latency streaming
    pub fn low_latency() -> Self {
        let mut config = Self::default();
        config.buffer_config.initial_size = 100;
        config.buffer_config.max_size = 1000;
        config.buffer_config.processing_timeout = Duration::from_millis(100);
        config.performance_config.evaluation_frequency = 5;
        config.drift_config.min_samples = 10;
        config.resource_config.monitoring_frequency = Duration::from_secs(1);
        config
    }

    /// Creates a configuration optimized for high-throughput streaming
    pub fn high_throughput() -> Self {
        let mut config = Self::default();
        config.buffer_config.initial_size = 5000;
        config.buffer_config.max_size = 50000;
        config.buffer_config.memory_limit_mb = 2048;
        config.performance_config.evaluation_frequency = 100;
        config.drift_config.window_size = 5000;
        config.resource_config.max_memory_mb = 4096;
        config
    }

    /// Creates a configuration optimized for memory-constrained environments
    pub fn memory_efficient() -> Self {
        let mut config = Self::default();
        config.buffer_config.initial_size = 200;
        config.buffer_config.max_size = 2000;
        config.buffer_config.memory_limit_mb = 128;
        config.performance_config.history_size = 100;
        config.meta_learning_config.experience_buffer_size = 1000;
        config.resource_config.max_memory_mb = 256;
        config.drift_config.window_size = 500;
        config
    }
}
