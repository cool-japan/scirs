//! Neural Architecture Search Configuration
//!
//! This module contains all configuration types, enums, and parameter definitions
//! for the Neural Architecture Search system.

use std::collections::HashMap;
use std::time::Duration;
use num_traits::Float;
use crate::learned_optimizers::few_shot_optimizer::EvaluationMetric;

use crate::neural_architecture_search::{
    MultiObjectiveOptimizer, PerformanceEvaluator, PerformancePredictor,
    ParetoFront,
};

/// Neural Architecture Search configuration for optimizers
#[derive(Debug, Clone)]
pub struct NASConfig<T: Float> {
    /// Search strategy to use
    pub search_strategy: SearchStrategyType,

    /// Architecture search space
    pub search_space: SearchSpaceConfig,

    /// Performance evaluation configuration
    pub evaluation_config: EvaluationConfig<T>,

    /// Multi-objective optimization settings
    pub multi_objective_config: MultiObjectiveConfig<T>,

    /// Search budget (number of architectures to evaluate)
    pub search_budget: usize,

    /// Early stopping criteria
    pub early_stopping: EarlyStoppingConfig<T>,

    /// Enable progressive search
    pub progressive_search: bool,

    /// Population size for evolutionary/genetic algorithms
    pub population_size: usize,

    /// Enable architecture transfer learning
    pub enable_transfer_learning: bool,

    /// Architecture encoding strategy
    pub encoding_strategy: ArchitectureEncodingStrategy,

    /// Enable performance prediction
    pub enable_performance_prediction: bool,

    /// Search parallelization factor
    pub parallelization_factor: usize,

    /// Enable automated hyperparameter tuning
    pub auto_hyperparameter_tuning: bool,

    /// Resource constraints
    pub resource_constraints: ResourceConstraints<T>,
}

/// Search strategy types
#[derive(Debug, Clone, Copy)]
pub enum SearchStrategyType {
    /// Random search baseline
    Random,

    /// Evolutionary/genetic algorithm search
    Evolutionary,

    /// Reinforcement learning-based search
    ReinforcementLearning,

    /// Differentiable architecture search (DARTS)
    Differentiable,

    /// Bayesian optimization
    BayesianOptimization,

    /// Progressive search
    Progressive,

    /// Multi-objective evolutionary algorithm
    MultiObjectiveEvolutionary,

    /// Neural predictor-based search
    NeuralPredictorBased,
}

/// Search space configuration
#[derive(Debug, Clone)]
pub struct SearchSpaceConfig {
    /// Available optimizer components
    pub components: Vec<OptimizerComponentConfig>,

    /// Connection patterns between components
    pub connection_patterns: Vec<ConnectionPatternType>,

    /// Learning rate schedule search space
    pub learning_rate_schedules: LearningRateScheduleSpace,

    /// Regularization technique search space
    pub regularization_techniques: RegularizationSpace,

    /// Adaptive mechanism search space
    pub adaptive_mechanisms: AdaptiveMechanismSpace,

    /// Memory usage constraints
    pub memory_constraints: MemoryConstraints,

    /// Computation constraints
    pub computation_constraints: ComputationConstraints,
}

/// Optimizer component configuration for search
#[derive(Debug, Clone)]
pub struct OptimizerComponentConfig {
    /// Component type
    pub component_type: ComponentType,

    /// Hyperparameter search ranges
    pub hyperparameter_ranges: HashMap<String, ParameterRange>,

    /// Component complexity score
    pub complexity_score: f64,

    /// Memory requirement estimate
    pub memory_requirement: usize,

    /// Computational cost estimate
    pub computational_cost: f64,

    /// Compatibility constraints
    pub compatibility_constraints: Vec<CompatibilityConstraint>,
}

/// Component type configuration
#[derive(Debug, Clone)]
pub struct ComponentTypeConfig {
    /// Component type name
    pub name: String,
    /// Available parameters
    pub parameters: Vec<String>,
    /// Default configuration
    pub defaults: HashMap<String, String>,
}

/// Component types for optimizers
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ComponentType {
    /// Gradient computation methods
    GradientComputation,

    /// Momentum techniques
    Momentum,

    /// Adaptive learning rate methods
    AdaptiveLearningRate,

    /// Regularization techniques
    Regularization,

    /// Normalization methods
    Normalization,

    /// Second-order methods
    SecondOrder,

    /// Preconditioning techniques
    Preconditioning,

    /// Learning rate scheduling
    LearningRateScheduling,

    /// Gradient clipping methods
    GradientClipping,

    /// Memory management
    MemoryManagement,

    /// Convergence acceleration
    ConvergenceAcceleration,

    /// Custom components
    Custom(String),
}

/// Parameter range for hyperparameter search
#[derive(Debug, Clone)]
pub enum ParameterRange {
    /// Continuous range [min, max]
    Continuous(f64, f64),

    /// Discrete set of values
    Discrete(Vec<f64>),

    /// Integer range [min, max]
    Integer(i32, i32),

    /// Boolean choice
    Boolean,

    /// Categorical choice
    Categorical(Vec<String>),

    /// Log-uniform distribution
    LogUniform(f64, f64),
}

/// Connection pattern types
#[derive(Debug, Clone, Copy)]
pub enum ConnectionPatternType {
    /// Sequential connection
    Sequential,

    /// Parallel branches
    Parallel,

    /// Skip connections
    SkipConnection,

    /// Dense connections
    DenseConnection,

    /// Residual connections
    ResidualConnection,

    /// Attention-based connections
    AttentionConnection,

    /// Highway connections
    HighwayConnection,

    /// Squeeze-and-excitation connections
    SqueezeExcitationConnection,

    /// Custom connection patterns
    Custom(String),
}

/// Learning rate schedule search space
#[derive(Debug, Clone)]
pub struct LearningRateScheduleSpace {
    /// Available schedule types
    pub schedule_types: Vec<ScheduleType>,

    /// Initial learning rate range
    pub initial_lr_range: ParameterRange,

    /// Schedule-specific parameters
    pub schedule_parameters: HashMap<ScheduleType, HashMap<String, ParameterRange>>,
}

/// Schedule types for learning rate
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScheduleType {
    Constant,
    StepDecay,
    ExponentialDecay,
    CosineAnnealing,
    CyclicalLR,
    OneCycleLR,
    ReduceOnPlateau,
    WarmupLinear,
    WarmupCosine,
    Custom(String),
}

/// Regularization search space
#[derive(Debug, Clone)]
pub struct RegularizationSpace {
    /// Available regularization techniques
    pub techniques: Vec<RegularizationTechnique>,

    /// Regularization strength ranges
    pub strength_ranges: HashMap<RegularizationTechnique, ParameterRange>,

    /// Combination strategies
    pub combination_strategies: Vec<String>,
}

/// Regularization techniques
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RegularizationTechnique {
    L1,
    L2,
    Dropout,
    DropConnect,
    BatchNormalization,
    LayerNormalization,
    GroupNormalization,
    SpectralNormalization,
    WeightDecay,
    EarlyStopping,
    Custom(String),
}

/// Adaptive mechanism search space
#[derive(Debug, Clone)]
pub struct AdaptiveMechanismSpace {
    /// Available adaptation strategies
    pub adaptation_strategies: Vec<AdaptationStrategy>,

    /// Adaptation parameters
    pub adaptation_parameters: HashMap<AdaptationStrategy, HashMap<String, ParameterRange>>,

    /// Adaptation frequency options
    pub adaptation_frequencies: Vec<usize>,
}

/// Adaptation strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AdaptationStrategy {
    PerformanceBased,
    GradientBased,
    LossBased,
    TimeBased,
    HybridAdaptation,
    Custom(String),
}

/// Memory constraints for architecture search
#[derive(Debug, Clone)]
pub struct MemoryConstraints {
    /// Maximum memory usage (bytes)
    pub max_memory_bytes: usize,

    /// Memory usage per component limits
    pub component_memory_limits: HashMap<ComponentType, usize>,

    /// Enable memory optimization
    pub enable_memory_optimization: bool,

    /// Memory allocation strategy
    pub allocation_strategy: MemoryAllocationStrategy,
}

/// Memory allocation strategies
#[derive(Debug, Clone, Copy)]
pub enum MemoryAllocationStrategy {
    Static,
    Dynamic,
    Adaptive,
    Lazy,
}

/// Computation constraints
#[derive(Debug, Clone)]
pub struct ComputationConstraints {
    /// Maximum computational cost
    pub max_computational_cost: f64,

    /// Cost per component limits
    pub component_cost_limits: HashMap<ComponentType, f64>,

    /// Enable computation optimization
    pub enable_computation_optimization: bool,

    /// Parallelization constraints
    pub parallelization_constraints: ParallelizationConstraints,
}

/// Parallelization constraints
#[derive(Debug, Clone)]
pub struct ParallelizationConstraints {
    /// Maximum parallel workers
    pub max_workers: usize,

    /// Minimum batch size for parallelization
    pub min_batch_size: usize,

    /// Enable SIMD optimization
    pub enable_simd: bool,

    /// Enable GPU acceleration
    pub enable_gpu: bool,
}

/// Compatibility constraint between components
#[derive(Debug, Clone)]
pub struct CompatibilityConstraint {
    /// Constraint type
    pub constraint_type: CompatibilityType,

    /// Target components
    pub target_components: Vec<ComponentType>,

    /// Constraint condition
    pub condition: ConstraintCondition,
}

/// Compatibility constraint types
#[derive(Debug, Clone, Copy)]
pub enum CompatibilityType {
    /// Components must be used together
    Requires,

    /// Components cannot be used together
    Excludes,

    /// Components have conditional compatibility
    Conditional,

    /// Components have version requirements
    VersionRequirement,

    /// Components have parameter constraints
    ParameterConstraint,

    /// Custom compatibility rules
    Custom(String),
}

/// Constraint conditions
#[derive(Debug, Clone)]
pub enum ConstraintCondition {
    /// Always applies
    Always,

    /// Never applies
    Never,

    /// Applies under certain parameter conditions
    ParameterCondition(ParameterCondition),

    /// Applies based on architecture properties
    ArchitectureCondition(String),

    /// Custom condition logic
    Custom(String),
}

/// Parameter-based constraint conditions
#[derive(Debug, Clone)]
pub enum ParameterCondition {
    /// Parameter equals specific value
    Equals(String, f64),

    /// Parameter is greater than value
    GreaterThan(String, f64),

    /// Parameter is less than value
    LessThan(String, f64),

    /// Parameter is within range
    InRange(String, f64, f64),

    /// Parameter is in set of values
    InSet(String, Vec<f64>),

    /// Complex boolean conditions
    BooleanExpression(String),
}

/// Performance evaluation configuration
#[derive(Debug, Clone)]
pub struct EvaluationConfig<T: Float> {
    /// Evaluation metrics to use
    pub metrics: Vec<EvaluationMetric>,

    /// Benchmark datasets
    pub benchmark_datasets: Vec<BenchmarkDataset>,

    /// Evaluation budget (time/iterations)
    pub evaluation_budget: EvaluationBudget,

    /// Statistical testing configuration
    pub statistical_testing: StatisticalTestingConfig,

    /// Cross-validation settings
    pub cross_validation_folds: usize,

    /// Enable early stopping during evaluation
    pub enable_early_stopping: bool,

    /// Evaluation parallelization
    pub parallelization_factor: usize,

    /// Problem domain specification
    pub problem_domains: Vec<ProblemDomain>,

    /// Resource limits for evaluation
    pub resource_limits: HashMap<ResourceType, T>,
}

/// Problem domains for evaluation
#[derive(Debug, Clone, Copy)]
pub enum ProblemDomain {
    Classification,
    Regression,
    Reinforcement,
    GenerativeModeling,
    SequenceModeling,
    ComputerVision,
    NaturalLanguageProcessing,
    TimeSeriesForecasting,
    Custom(String),
}

/// Resource types for evaluation limits
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResourceType {
    Memory,
    CPUTime,
    GPUTime,
    NetworkBandwidth,
    StorageSpace,
}

/// Benchmark dataset configuration
#[derive(Debug, Clone)]
pub struct BenchmarkDataset {
    /// Dataset name
    pub name: String,

    /// Dataset path or URL
    pub path: String,

    /// Dataset characteristics
    pub characteristics: DatasetCharacteristics,

    /// Problem type
    pub problem_type: ProblemType,

    /// Evaluation weight
    pub weight: f64,

    /// Enable data augmentation
    pub enable_augmentation: bool,
}

/// Dataset characteristics
#[derive(Debug, Clone)]
pub struct DatasetCharacteristics {
    /// Number of samples
    pub num_samples: usize,

    /// Number of features
    pub num_features: usize,

    /// Number of classes (for classification)
    pub num_classes: Option<usize>,

    /// Dataset size category
    pub size_category: DatasetSizeCategory,

    /// Feature correlation structure
    pub correlation_structure: CorrelationStructure,

    /// Noise level estimate
    pub noise_level: f64,

    /// Imbalance ratio (for classification)
    pub imbalance_ratio: Option<f64>,
}

/// Problem types for evaluation
#[derive(Debug, Clone, Copy)]
pub enum ProblemType {
    BinaryClassification,
    MultiClassClassification,
    Regression,
    MultiTaskLearning,
    MetaLearning,
    TransferLearning,
    FewShotLearning,
    Custom(String),
}

/// Dataset size categories
#[derive(Debug, Clone, Copy)]
pub enum DatasetSizeCategory {
    Small,      // < 1K samples
    Medium,     // 1K - 100K samples
    Large,      // 100K - 1M samples
    VeryLarge,  // > 1M samples
}

/// Correlation structure in datasets
#[derive(Debug, Clone, Copy)]
pub enum CorrelationStructure {
    Independent,
    LowCorrelation,
    ModerateCorrelation,
    HighCorrelation,
    BlockStructure,
    Hierarchical,
}

/// Evaluation budget configuration
#[derive(Debug, Clone)]
pub struct EvaluationBudget {
    /// Maximum evaluation time per architecture
    pub max_time_per_architecture: Duration,

    /// Maximum number of training epochs
    pub max_epochs: usize,

    /// Maximum number of function evaluations
    pub max_function_evaluations: usize,

    /// Early stopping patience
    pub early_stopping_patience: usize,

    /// Resource allocation per evaluation
    pub resource_allocation: ResourceAllocation,
}

/// Resource allocation for evaluations
#[derive(Debug, Clone)]
pub struct ResourceAllocation {
    /// CPU cores allocated
    pub cpu_cores: usize,

    /// Memory allocated (GB)
    pub memory_gb: f64,

    /// GPU devices allocated
    pub gpu_devices: usize,

    /// Storage space allocated (GB)
    pub storage_gb: f64,
}

/// Statistical testing configuration
#[derive(Debug, Clone)]
pub struct StatisticalTestingConfig {
    /// Significance level (alpha)
    pub significance_level: f64,

    /// Statistical test type
    pub test_type: StatisticalTestType,

    /// Multiple comparison correction
    pub multiple_comparison_correction: MultipleComparisonCorrection,

    /// Minimum effect size
    pub min_effect_size: f64,

    /// Bootstrap samples for confidence intervals
    pub bootstrap_samples: usize,
}

/// Statistical test types
#[derive(Debug, Clone, Copy)]
pub enum StatisticalTestType {
    TTest,
    WilcoxonRankSum,
    KruskalWallis,
    FriedmanTest,
    Bootstrap,
}

/// Multiple comparison correction methods
#[derive(Debug, Clone, Copy)]
pub enum MultipleComparisonCorrection {
    None,
    Bonferroni,
    BenjaminiHochberg,
    BenjaminiYekutieli,
    Holm,
}

/// Multi-objective optimization configuration
#[derive(Debug, Clone)]
pub struct MultiObjectiveConfig<T: Float> {
    /// List of objectives to optimize
    pub objectives: Vec<ObjectiveConfig<T>>,

    /// Multi-objective algorithm to use
    pub algorithm: MultiObjectiveAlgorithm,

    /// User preferences for trade-offs
    pub user_preferences: Option<UserPreferences<T>>,

    /// Diversity promotion strategy
    pub diversity_strategy: DiversityStrategy,

    /// Constraint handling method
    pub constraint_handling: ConstraintHandlingMethod,
}

/// Objective configuration
#[derive(Debug, Clone)]
pub struct ObjectiveConfig<T: Float> {
    /// Objective name
    pub name: String,

    /// Objective type
    pub objective_type: ObjectiveType,

    /// Optimization direction
    pub direction: OptimizationDirection,

    /// Objective weight
    pub weight: T,

    /// Objective priority
    pub priority: ObjectivePriority,

    /// Normalization bounds
    pub normalization_bounds: Option<(T, T)>,
}

/// Objective types
#[derive(Debug, Clone, Copy)]
pub enum ObjectiveType {
    Accuracy,
    Loss,
    TrainingTime,
    InferenceTime,
    MemoryUsage,
    EnergyConsumption,
    ModelSize,
    Robustness,
    Fairness,
    Custom(String),
}

/// Optimization directions
#[derive(Debug, Clone, Copy)]
pub enum OptimizationDirection {
    Minimize,
    Maximize,
}

/// Objective priorities
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ObjectivePriority {
    Low,
    Medium,
    High,
    Critical,
}

/// Multi-objective algorithms
#[derive(Debug, Clone, Copy)]
pub enum MultiObjectiveAlgorithm {
    NSGA2,
    NSGA3,
    MOEAD,
    SPEA2,
    PAES,
    WeightedSum,
    EpsilonConstraint,
    GoalProgramming,
    Custom(String),
}

/// User preferences for multi-objective optimization
#[derive(Debug, Clone)]
pub struct UserPreferences<T: Float> {
    /// Preference type
    pub preference_type: PreferenceType<T>,

    /// Reference point (for reference point methods)
    pub reference_point: Option<Vec<T>>,

    /// Aspiration levels
    pub aspiration_levels: Option<Vec<T>>,

    /// Reservation levels
    pub reservation_levels: Option<Vec<T>>,
}

/// Preference specification types
#[derive(Debug, Clone)]
pub enum PreferenceType<T: Float> {
    WeightVector(Vec<T>),
    ReferencePoint(Vec<T>),
    GoalVector(Vec<T>),
    RankingOrder(Vec<usize>),
    PairwiseComparison(Vec<(usize, usize)>),
}

/// Diversity promotion strategies
#[derive(Debug, Clone, Copy)]
pub enum DiversityStrategy {
    Crowding,
    Sharing,
    Clearing,
    Clustering,
    Novelty,
    Custom(String),
}

/// Constraint handling methods
#[derive(Debug, Clone, Copy)]
pub enum ConstraintHandlingMethod {
    PenaltyFunction,
    DeathPenalty,
    Repair,
    Decoder,
    PreserveFeasibility,
    Custom(String),
}

/// Early stopping configuration
#[derive(Debug, Clone)]
pub struct EarlyStoppingConfig<T: Float> {
    /// Enable early stopping
    pub enabled: bool,

    /// Patience (number of generations without improvement)
    pub patience: usize,

    /// Minimum improvement threshold
    pub min_improvement: T,

    /// Convergence detection strategy
    pub convergence_strategy: ConvergenceDetectionStrategy,

    /// Minimum number of generations before stopping
    pub min_generations: usize,
}

/// Convergence detection strategies
#[derive(Debug, Clone, Copy)]
pub enum ConvergenceDetectionStrategy {
    BestScore,
    AverageScore,
    PopulationDiversity,
    ParetoFrontStability,
    Custom(String),
}

/// Architecture encoding strategies
#[derive(Debug, Clone, Copy)]
pub enum ArchitectureEncodingStrategy {
    /// Direct parameter encoding
    Direct,

    /// Graph-based encoding
    Graph,

    /// String-based encoding
    String,

    /// Binary encoding
    Binary,

    /// Real-valued encoding
    RealValued,

    /// Hierarchical encoding
    Hierarchical,

    /// Neural encoding (using neural networks)
    Neural,

    /// Hybrid encoding strategies
    Hybrid,

    /// Custom encoding
    Custom(String),
}

/// Resource constraints for NAS
#[derive(Debug, Clone)]
pub struct ResourceConstraints<T: Float> {
    /// Hardware resource limits
    pub hardware_resources: HardwareResources,

    /// Time constraints
    pub time_constraints: TimeConstraints,

    /// Energy constraints
    pub energy_constraints: Option<T>,

    /// Cost constraints
    pub cost_constraints: Option<T>,

    /// Resource violation handling
    pub violation_handling: ResourceViolationHandling,
}

/// Hardware resource specifications
#[derive(Debug, Clone)]
pub struct HardwareResources {
    /// Maximum memory usage (GB)
    pub max_memory_gb: f64,

    /// Maximum CPU cores
    pub max_cpu_cores: usize,

    /// Maximum GPU devices
    pub max_gpu_devices: usize,

    /// Maximum storage space (GB)
    pub max_storage_gb: f64,

    /// Network bandwidth limits (MB/s)
    pub max_network_bandwidth: f64,

    /// Enable cloud resource scaling
    pub enable_cloud_scaling: bool,

    /// Cloud resource budget
    pub cloud_budget: Option<f64>,
}

/// Time constraints for search
#[derive(Debug, Clone)]
pub struct TimeConstraints {
    /// Maximum total search time
    pub max_search_time: Duration,

    /// Maximum time per architecture evaluation
    pub max_evaluation_time: Duration,

    /// Search deadline
    pub search_deadline: Option<std::time::Instant>,

    /// Time budget allocation strategy
    pub budget_allocation: TimeBudgetAllocation,
}

/// Time budget allocation strategies
#[derive(Debug, Clone, Copy)]
pub enum TimeBudgetAllocation {
    Uniform,
    AdaptiveByComplexity,
    AdaptiveByPerformance,
    PriorityBased,
    Custom(String),
}

/// Resource violation handling strategies
#[derive(Debug, Clone, Copy)]
pub enum ResourceViolationHandling {
    /// Stop search immediately
    Abort,

    /// Skip violating architectures
    Skip,

    /// Scale down resources
    ScaleDown,

    /// Use approximation methods
    Approximate,

    /// Dynamic resource allocation
    Dynamic,

    /// Custom handling
    Custom(String),
}

// Default implementations
impl<T: Float> Default for NASConfig<T> {
    fn default() -> Self {
        Self {
            search_strategy: SearchStrategyType::Evolutionary,
            search_space: SearchSpaceConfig::default(),
            evaluation_config: EvaluationConfig::default(),
            multi_objective_config: MultiObjectiveConfig::default(),
            search_budget: 100,
            early_stopping: EarlyStoppingConfig::default(),
            progressive_search: false,
            population_size: 20,
            enable_transfer_learning: false,
            encoding_strategy: ArchitectureEncodingStrategy::Graph,
            enable_performance_prediction: false,
            parallelization_factor: 1,
            auto_hyperparameter_tuning: false,
            resource_constraints: ResourceConstraints::default(),
        }
    }
}

impl Default for SearchSpaceConfig {
    fn default() -> Self {
        Self {
            components: Vec::new(),
            connection_patterns: vec![
                ConnectionPatternType::Sequential,
                ConnectionPatternType::Parallel,
                ConnectionPatternType::SkipConnection,
            ],
            learning_rate_schedules: LearningRateScheduleSpace::default(),
            regularization_techniques: RegularizationSpace::default(),
            adaptive_mechanisms: AdaptiveMechanismSpace::default(),
            memory_constraints: MemoryConstraints::default(),
            computation_constraints: ComputationConstraints::default(),
        }
    }
}

impl Default for LearningRateScheduleSpace {
    fn default() -> Self {
        Self {
            schedule_types: vec![
                ScheduleType::Constant,
                ScheduleType::StepDecay,
                ScheduleType::ExponentialDecay,
                ScheduleType::CosineAnnealing,
            ],
            initial_lr_range: ParameterRange::LogUniform(1e-5, 1e-1),
            schedule_parameters: HashMap::new(),
        }
    }
}

impl Default for RegularizationSpace {
    fn default() -> Self {
        Self {
            techniques: vec![
                RegularizationTechnique::L1,
                RegularizationTechnique::L2,
                RegularizationTechnique::Dropout,
                RegularizationTechnique::WeightDecay,
            ],
            strength_ranges: HashMap::new(),
            combination_strategies: Vec::new(),
        }
    }
}

impl Default for AdaptiveMechanismSpace {
    fn default() -> Self {
        Self {
            adaptation_strategies: vec![
                AdaptationStrategy::PerformanceBased,
                AdaptationStrategy::GradientBased,
                AdaptationStrategy::LossBased,
            ],
            adaptation_parameters: HashMap::new(),
            adaptation_frequencies: vec![1, 5, 10, 25, 50, 100],
        }
    }
}

impl Default for MemoryConstraints {
    fn default() -> Self {
        Self {
            max_memory_bytes: 32 * 1024 * 1024 * 1024, // 32GB
            component_memory_limits: HashMap::new(),
            enable_memory_optimization: true,
            allocation_strategy: MemoryAllocationStrategy::Dynamic,
        }
    }
}

impl Default for ComputationConstraints {
    fn default() -> Self {
        Self {
            max_computational_cost: 1000.0,
            component_cost_limits: HashMap::new(),
            enable_computation_optimization: true,
            parallelization_constraints: ParallelizationConstraints::default(),
        }
    }
}

impl Default for ParallelizationConstraints {
    fn default() -> Self {
        Self {
            max_workers: num_cpus::get(),
            min_batch_size: 32,
            enable_simd: true,
            enable_gpu: true,
        }
    }
}

impl<T: Float> Default for EvaluationConfig<T> {
    fn default() -> Self {
        Self {
            metrics: vec![EvaluationMetric::Accuracy, EvaluationMetric::TrainingTime],
            benchmark_datasets: Vec::new(),
            evaluation_budget: EvaluationBudget::default(),
            statistical_testing: StatisticalTestingConfig::default(),
            cross_validation_folds: 5,
            enable_early_stopping: true,
            parallelization_factor: 1,
            problem_domains: vec![ProblemDomain::Classification],
            resource_limits: HashMap::new(),
        }
    }
}

impl Default for EvaluationBudget {
    fn default() -> Self {
        Self {
            max_time_per_architecture: Duration::from_secs(300),
            max_epochs: 100,
            max_function_evaluations: 1000,
            early_stopping_patience: 10,
            resource_allocation: ResourceAllocation::default(),
        }
    }
}

impl Default for ResourceAllocation {
    fn default() -> Self {
        Self {
            cpu_cores: 4,
            memory_gb: 8.0,
            gpu_devices: 1,
            storage_gb: 10.0,
        }
    }
}

impl Default for StatisticalTestingConfig {
    fn default() -> Self {
        Self {
            significance_level: 0.05,
            test_type: StatisticalTestType::TTest,
            multiple_comparison_correction: MultipleComparisonCorrection::BenjaminiHochberg,
            min_effect_size: 0.1,
            bootstrap_samples: 1000,
        }
    }
}

impl<T: Float> Default for MultiObjectiveConfig<T> {
    fn default() -> Self {
        Self {
            objectives: Vec::new(),
            algorithm: MultiObjectiveAlgorithm::NSGA2,
            user_preferences: None,
            diversity_strategy: DiversityStrategy::Crowding,
            constraint_handling: ConstraintHandlingMethod::PenaltyFunction,
        }
    }
}

impl<T: Float> Default for EarlyStoppingConfig<T> {
    fn default() -> Self {
        Self {
            enabled: true,
            patience: 20,
            min_improvement: T::from(0.001).unwrap(),
            convergence_strategy: ConvergenceDetectionStrategy::BestScore,
            min_generations: 10,
        }
    }
}

impl<T: Float> Default for ResourceConstraints<T> {
    fn default() -> Self {
        Self {
            hardware_resources: HardwareResources::default(),
            time_constraints: TimeConstraints::default(),
            energy_constraints: None,
            cost_constraints: None,
            violation_handling: ResourceViolationHandling::Skip,
        }
    }
}

impl Default for HardwareResources {
    fn default() -> Self {
        Self {
            max_memory_gb: 32.0,
            max_cpu_cores: num_cpus::get(),
            max_gpu_devices: 4,
            max_storage_gb: 100.0,
            max_network_bandwidth: 1000.0, // 1 GB/s
            enable_cloud_scaling: false,
            cloud_budget: None,
        }
    }
}

impl Default for TimeConstraints {
    fn default() -> Self {
        Self {
            max_search_time: Duration::from_secs(3600), // 1 hour
            max_evaluation_time: Duration::from_secs(300), // 5 minutes
            search_deadline: None,
            budget_allocation: TimeBudgetAllocation::Uniform,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nas_config_default() {
        let config = NASConfig::<f32>::default();
        assert_eq!(config.search_budget, 100);
        assert_eq!(config.population_size, 20);
        assert!(!config.progressive_search);
    }

    #[test]
    fn test_search_space_config_default() {
        let config = SearchSpaceConfig::default();
        assert!(!config.connection_patterns.is_empty());
        assert_eq!(config.connection_patterns.len(), 3);
    }

    #[test]
    fn test_parameter_range_variants() {
        let continuous = ParameterRange::Continuous(0.0, 1.0);
        let discrete = ParameterRange::Discrete(vec![0.1, 0.5, 0.9]);
        let integer = ParameterRange::Integer(1, 10);

        match continuous {
            ParameterRange::Continuous(min, max) => {
                assert_eq!(min, 0.0);
                assert_eq!(max, 1.0);
            }
            _ => panic!("Expected continuous range"),
        }

        match discrete {
            ParameterRange::Discrete(values) => {
                assert_eq!(values.len(), 3);
            }
            _ => panic!("Expected discrete range"),
        }

        match integer {
            ParameterRange::Integer(min, max) => {
                assert_eq!(min, 1);
                assert_eq!(max, 10);
            }
            _ => panic!("Expected integer range"),
        }
    }
}