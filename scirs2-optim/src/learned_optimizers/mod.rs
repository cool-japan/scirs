//! Learned optimizers using neural networks
//!
//! This module implements learned optimizers that use neural networks (particularly LSTMs)
//! to learn optimization strategies, enabling meta-learning for automated optimizer design.

use ndarray::{s, Array, Array1, Array2, ArrayBase, Data, Dimension};
use num_traits::Float;
use rand::Rng;
use std::collections::{HashMap, VecDeque};

pub mod adaptive_nas_system;
pub mod adaptive_transformer_enhancement;
pub mod few_shot_learning_enhancement;
pub mod few_shot_optimizer;
pub mod lstm_optimizer;
pub mod meta_learning;
pub mod neural_architecture_search;
pub mod transformer_optimizer;

// Advanced Mode - Advanced AI optimization coordination
pub mod optimization_coordinator;

use crate::error::{OptimError, Result};
use crate::optimizers::Optimizer;

/// Configuration for learned optimizers
#[derive(Debug, Clone)]
pub struct LearnedOptimizerConfig {
    /// Type of neural optimizer
    pub optimizer_type: NeuralOptimizerType,

    /// Hidden state size for LSTM-based optimizers
    pub hidden_size: usize,

    /// Number of layers in the neural network
    pub num_layers: usize,

    /// Input feature dimensions
    pub input_features: usize,

    /// Output dimensions (typically parameter updates)
    pub output_features: usize,

    /// Learning rate for meta-learning
    pub meta_learning_rate: f64,

    /// Window size for gradient history
    pub gradient_history_size: usize,

    /// Enable attention mechanism
    pub use_attention: bool,

    /// Attention head count
    pub attention_heads: usize,

    /// Enable recurrent connections
    pub use_recurrent: bool,

    /// Dropout rate for regularization
    pub dropout_rate: f64,

    /// Enable learned learning rate schedules
    pub learned_lr_schedule: bool,

    /// Meta-optimization strategy
    pub meta_strategy: MetaOptimizationStrategy,

    /// Pre-training dataset size
    pub pretraining_dataset_size: usize,

    /// Enable transfer learning
    pub enable_transfer_learning: bool,

    /// Advanced architecture features
    pub use_residual_connections: bool,

    /// Enable layer normalization
    pub use_layer_normalization: bool,

    /// Enable self-supervision
    pub enable_self_supervision: bool,

    /// Memory-efficient architecture
    pub memory_efficient: bool,

    /// Multi-scale processing
    pub enable_multiscale: bool,

    /// Adaptive architecture
    pub adaptive_architecture: bool,

    /// Hierarchical optimization
    pub hierarchical_optimization: bool,

    /// Dynamic architecture
    pub dynamic_architecture: bool,
}

impl Default for LearnedOptimizerConfig {
    fn default() -> Self {
        Self {
            optimizer_type: NeuralOptimizerType::LSTM,
            hidden_size: 256,
            num_layers: 2,
            input_features: 64,
            output_features: 1,
            meta_learning_rate: 0.001,
            gradient_history_size: 10,
            use_attention: true,
            attention_heads: 8,
            use_recurrent: true,
            dropout_rate: 0.1,
            learned_lr_schedule: true,
            meta_strategy: MetaOptimizationStrategy::MAML,
            pretraining_dataset_size: 10000,
            enable_transfer_learning: true,
            use_residual_connections: true,
            use_layer_normalization: true,
            enable_self_supervision: false,
            memory_efficient: true,
            enable_multiscale: true,
            adaptive_architecture: true,
            hierarchical_optimization: false,
            dynamic_architecture: false,
        }
    }
}

/// Types of neural optimizers
#[derive(Debug, Clone, Copy)]
pub enum NeuralOptimizerType {
    /// LSTM-based optimizer
    LSTM,
    /// Transformer-based optimizer
    Transformer,
    /// Convolutional optimizer
    Convolutional,
    /// Graph Neural Network optimizer
    GraphNN,
    /// Hybrid architecture
    Hybrid,
    /// Attention-based optimizer
    AttentionBased,
    /// Neural ODE optimizer
    NeuralODE,
    /// Neuroevolution optimizer
    Neuroevolution,
    /// Capsule network optimizer
    CapsuleNet,
    /// Memory-augmented optimizer
    MemoryAugmented,
}

/// Meta-optimization strategies
#[derive(Debug, Clone, Copy)]
pub enum MetaOptimizationStrategy {
    /// Model-Agnostic Meta-Learning
    MAML,
    /// Prototypical MAML
    ProtoMAML,
    /// Reptile meta-learning
    Reptile,
    /// Learning to Learn by Gradient Descent
    L2L,
    /// Meta-SGD
    MetaSGD,
    /// Learned optimizer (from scratch)
    LearnedOptimizer,
    /// Few-shot meta-learning
    FewShot,
    /// Continual meta-learning
    Continual,
    /// Task-agnostic meta-learning
    TaskAgnostic,
    /// Bayesian meta-learning
    Bayesian,
    /// Meta-learning with uncertainty
    UncertaintyAware,
    /// Hierarchical meta-learning
    Hierarchical,
}

/// LSTM-based neural optimizer
#[allow(dead_code)]
pub struct LSTMOptimizer<
    A: Float
        + std::fmt::Debug
        + Clone
        + Default
        + Send
        + Sync
        + ndarray::ScalarOperand
        + std::iter::Sum
        + std::iter::Sum<A>
        + for<'a> std::iter::Sum<&'a A>
        + 'static,
    D: Dimension = ndarray::Ix1,
> {
    /// Configuration
    config: LearnedOptimizerConfig,

    /// LSTM cell state
    cell_state: LSTMState<A>,

    /// Learned parameters
    parameters: LSTMParameters<A>,

    /// Gradient history for context
    gradient_history: VecDeque<Array1<A>>,

    /// Parameter history
    parameter_history: VecDeque<Array1<A>>,

    /// Loss history
    loss_history: VecDeque<A>,

    /// Meta-optimizer for learning the optimizer
    meta_optimizer: Box<dyn Optimizer<A, D> + Send + Sync>,

    /// Training state
    training_state: MetaTrainingState<A>,

    /// Performance metrics
    metrics: LearnedOptimizerMetrics,

    /// Step count
    step_count: usize,
}

impl<
        A: Float
            + std::fmt::Debug
            + Clone
            + Default
            + Send
            + Sync
            + ndarray::ScalarOperand
            + std::iter::Sum
            + std::iter::Sum<A>
            + for<'a> std::iter::Sum<&'a A>
            + 'static,
        D: Dimension,
    > std::fmt::Debug for LSTMOptimizer<A, D>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LSTMOptimizer")
            .field("config", &self.config)
            .field("cell_state", &self.cell_state)
            .field("parameters", &self.parameters)
            .field("gradient_history", &self.gradient_history)
            .field("parameter_history", &self.parameter_history)
            .field("loss_history", &self.loss_history)
            .field("meta_optimizer", &"<trait object>")
            .field("training_state", &self.training_state)
            .field("metrics", &self.metrics)
            .field("step_count", &self.step_count)
            .finish()
    }
}

impl<
        A: Float
            + std::fmt::Debug
            + Clone
            + Default
            + Send
            + Sync
            + ndarray::ScalarOperand
            + std::iter::Sum
            + std::iter::Sum<A>
            + for<'a> std::iter::Sum<&'a A>
            + 'static,
        D: Dimension,
    > Clone for LSTMOptimizer<A, D>
{
    fn clone(&self) -> Self {
        // For meta_optimizer, we'll create a new SGD instance as a fallback
        // In a real implementation, you'd want a proper cloning mechanism for optimizers
        let meta_optimizer: Box<dyn Optimizer<A, D> + Send + Sync> =
            Box::new(crate::optimizers::SGD::new(A::from(0.01).unwrap()));

        Self {
            config: self.config.clone(),
            cell_state: self.cell_state.clone(),
            parameters: self.parameters.clone(),
            gradient_history: self.gradient_history.clone(),
            parameter_history: self.parameter_history.clone(),
            loss_history: self.loss_history.clone(),
            meta_optimizer,
            training_state: self.training_state.clone(),
            metrics: self.metrics.clone(),
            step_count: self.step_count,
        }
    }
}

/// LSTM state for neural optimizer
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct LSTMState<A: Float> {
    /// Hidden states for each layer
    hidden_states: Vec<Array1<A>>,

    /// Cell states for each layer
    cell_states: Vec<Array1<A>>,

    /// Attention weights (if using attention)
    attention_weights: Option<Array2<A>>,

    /// Context vector
    context_vector: Option<Array1<A>>,
}

/// LSTM parameters (weights and biases)
#[derive(Debug, Clone)]
pub struct LSTMParameters<A: Float> {
    /// Input-to-hidden weights
    weight_ih: Vec<Array2<A>>,

    /// Hidden-to-hidden weights
    weight_hh: Vec<Array2<A>>,

    /// Input-to-hidden biases
    bias_ih: Vec<Array1<A>>,

    /// Hidden-to-hidden biases
    bias_hh: Vec<Array1<A>>,

    /// Output projection weights
    output_weights: Array2<A>,

    /// Output biases
    output_bias: Array1<A>,

    /// Attention parameters (if using attention)
    attention_params: Option<AttentionParameters<A>>,

    /// Learned learning rate parameters
    lr_params: Option<LearningRateParameters<A>>,
}

/// Attention mechanism parameters
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct AttentionParameters<A: Float> {
    /// Query projection weights
    query_weights: Array2<A>,

    /// Key projection weights
    key_weights: Array2<A>,

    /// Value projection weights
    value_weights: Array2<A>,

    /// Output projection weights
    output_weights: Array2<A>,

    /// Multi-head attention parameters
    head_weights: Vec<Array2<A>>,
}

/// Learned learning rate parameters
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct LearningRateParameters<A: Float> {
    /// Base learning rate
    base_lr: A,

    /// Adaptive factors
    adaptive_factors: Array1<A>,

    /// Schedule parameters
    schedule_params: Array1<A>,

    /// Decay parameters
    decay_params: Array1<A>,
}

/// Meta-training state
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct MetaTrainingState<A: Float> {
    /// Meta-training step count
    meta_step: usize,

    /// Meta-gradients accumulator
    meta_gradients: HashMap<String, Array1<A>>,

    /// Task performance history
    task_performance: VecDeque<TaskPerformance<A>>,

    /// Current meta-batch
    current_meta_batch: Vec<MetaTask<A>>,

    /// Meta-validation metrics
    meta_validation: MetaValidationMetrics<A>,

    /// Transfer learning state
    transfer_state: Option<TransferLearningState<A>>,
}

/// Task performance tracking
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct TaskPerformance<A: Float> {
    /// Task identifier
    task_id: String,

    /// Initial loss
    initial_loss: A,

    /// Final loss
    final_loss: A,

    /// Convergence steps
    convergence_steps: usize,

    /// Learning curve
    learning_curve: Vec<A>,

    /// Task metadata
    metadata: HashMap<String, String>,
}

/// Meta-learning task
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct MetaTask<A: Float> {
    /// Task identifier
    id: String,

    /// Training data
    train_data: Vec<(Array1<A>, A)>,

    /// Validation data
    val_data: Vec<(Array1<A>, A)>,

    /// Task-specific parameters
    task_params: HashMap<String, A>,

    /// Expected performance
    target_performance: Option<A>,
}

/// Meta-validation metrics
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct MetaValidationMetrics<A: Float> {
    /// Average task performance
    avg_task_performance: A,

    /// Performance variance
    performance_variance: A,

    /// Generalization error
    generalization_error: A,

    /// Adaptation speed
    adaptation_speed: A,

    /// Task diversity score
    task_diversity: A,
}

/// Transfer learning state
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct TransferLearningState<A: Float> {
    /// Source domain performance
    source_performance: A,

    /// Target domain performance
    target_performance: A,

    /// Transfer efficiency
    transfer_efficiency: A,

    /// Adapted parameters
    adapted_params: HashMap<String, Array1<A>>,

    /// Fine-tuning steps
    finetuning_steps: usize,
}

/// Performance metrics for learned optimizers
#[derive(Debug, Clone)]
pub struct LearnedOptimizerMetrics {
    /// Meta-training loss
    pub meta_training_loss: f64,

    /// Average task convergence speed
    pub avg_convergence_speed: f64,

    /// Generalization performance
    pub generalization_performance: f64,

    /// Parameter efficiency
    pub parameter_efficiency: f64,

    /// Transfer learning success rate
    pub transfer_success_rate: f64,

    /// Computational overhead
    pub computational_overhead: f64,

    /// Memory usage
    pub memory_usage_mb: f64,

    /// Meta-gradient norm
    pub meta_gradient_norm: f64,
}

/// Advanced Neural Optimizer Factory
#[allow(dead_code)]
pub struct AdvancedNeuralOptimizerFactory<A: Float> {
    /// Available optimizer types
    available_types: Vec<NeuralOptimizerType>,

    /// Optimizer registry
    optimizer_registry: HashMap<String, Box<dyn NeuralOptimizerBuilder<A>>>,

    /// Performance database
    performance_db: OptimizerPerformanceDatabase<A>,

    /// Auto-selection criteria
    auto_selection: AutoSelectionCriteria<A>,
}

/// Neural optimizer builder trait
pub trait NeuralOptimizerBuilder<A: Float> {
    /// Build the neural optimizer
    fn build(&self, config: &LearnedOptimizerConfig) -> Result<Box<dyn NeuralOptimizer<A>>>;

    /// Get optimizer metadata
    fn metadata(&self) -> NeuralOptimizerMetadata;

    /// Estimate resource requirements
    fn estimate_resources(&self, config: &LearnedOptimizerConfig) -> ResourceEstimate;
}

/// Neural optimizer trait
pub trait NeuralOptimizer<A: Float> {
    /// Perform optimization step
    fn step(&mut self, gradients: &Array1<A>) -> Result<Array1<A>>;

    /// Update meta-parameters
    fn meta_update(&mut self, metagradients: &Array1<A>) -> Result<()>;

    /// Adapt to new task
    fn adapt_to_task(&mut self, taskcontext: &TaskContext<A>) -> Result<()>;

    /// Get current state
    fn get_state(&self) -> OptimizerState<A>;

    /// Set state
    fn set_state(&mut self, state: OptimizerState<A>) -> Result<()>;

    /// Get performance metrics
    fn get_metrics(&self) -> NeuralOptimizerMetrics<A>;
}

/// Neural optimizer metadata
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct NeuralOptimizerMetadata {
    pub name: String,
    pub version: String,
    pub description: String,
    pub supported_domains: Vec<OptimizationDomain>,
    pub complexity_level: ComplexityLevel,
    pub memory_requirements: MemoryRequirements,
    pub computational_requirements: ComputationalRequirements,
}

/// Optimization domains for neural optimizers
#[derive(Debug, Clone)]
pub enum OptimizationDomain {
    /// Computer vision tasks
    ComputerVision,
    /// Natural language processing
    NaturalLanguageProcessing,
    /// Reinforcement learning
    ReinforcementLearning,
    /// General machine learning
    GeneralML,
    /// Scientific computing
    ScientificComputing,
    /// Recommendation systems
    RecommendationSystems,
    /// Time series analysis
    TimeSeries,
    /// Graph neural networks
    GraphNeuralNetworks,
}

/// Resource requirements for optimization tasks
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ResourceRequirements {
    /// Memory requirements in MB
    pub memory_mb: usize,
    /// CPU cores required
    pub cpu_cores: usize,
    /// GPU memory requirements in MB
    pub gpu_memory_mb: Option<usize>,
    /// Expected runtime in seconds
    pub expected_runtime_seconds: f64,
    /// Storage requirements in MB
    pub storage_mb: usize,
}

/// Complexity levels
#[derive(Debug, Clone)]
pub enum ComplexityLevel {
    Low,
    Medium,
    High,
    VeryHigh,
}

/// Memory requirements
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct MemoryRequirements {
    pub minimum_mb: usize,
    pub recommended_mb: usize,
    pub maximum_mb: Option<usize>,
    pub scales_with_input: bool,
}

/// Computational requirements
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ComputationalRequirements {
    pub minimum_ops_per_step: usize,
    pub gpu_acceleration: bool,
    pub parallel_execution: bool,
    pub real_time_capable: bool,
}

/// Resource estimate
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ResourceEstimate {
    pub memory_mb: usize,
    pub compute_ops: usize,
    pub training_time_hours: f64,
    pub inference_latency_ms: f64,
}

/// Task context for adaptation
#[derive(Debug)]
pub struct TaskContext<A: Float> {
    /// Task identifier
    pub task_id: String,

    /// Task type
    pub task_type: TaskType,

    /// Problem dimensions
    pub problem_dimensions: ProblemDimensions,

    /// Task-specific features
    pub features: HashMap<String, A>,

    /// Historical performance
    pub historical_performance: Vec<f64>,

    /// Task constraints
    pub constraints: TaskConstraints<A>,
}

/// Task types
#[derive(Debug, Clone)]
pub enum TaskType {
    Supervised,
    Unsupervised,
    Reinforcement,
    SelfSupervised,
    MetaLearning,
    Transfer,
    Multitask,
}

/// Problem dimensions
#[derive(Debug, Clone)]
pub struct ProblemDimensions {
    pub input_dim: usize,
    pub output_dim: usize,
    pub parameter_count: usize,
    pub dataset_size: usize,
    pub sequence_length: Option<usize>,
}

/// Task constraints
#[derive(Debug)]
pub struct TaskConstraints<A: Float> {
    /// Time budget
    pub time_budget: Option<std::time::Duration>,

    /// Memory budget
    pub memory_budget: Option<usize>,

    /// Accuracy threshold
    pub accuracy_threshold: Option<A>,

    /// Resource constraints
    pub resource_constraints: HashMap<String, A>,
}

/// Optimizer state
#[derive(Debug, Clone)]
pub struct OptimizerState<A: Float> {
    /// Internal parameters
    pub parameters: HashMap<String, Array1<A>>,

    /// Hidden states
    pub hidden_states: HashMap<String, Array2<A>>,

    /// Memory buffers
    pub memory_buffers: HashMap<String, Array2<A>>,

    /// Step count
    pub step_count: usize,

    /// State metadata
    pub metadata: StateMetadata,
}

/// State metadata
#[derive(Debug, Clone)]
pub struct StateMetadata {
    pub version: String,
    pub timestamp: std::time::SystemTime,
    pub checksum: u64,
    pub compression_level: u8,
}

/// Neural optimizer metrics
#[derive(Debug)]
pub struct NeuralOptimizerMetrics<A: Float> {
    /// Performance metrics
    pub performance: PerformanceMetrics<A>,

    /// Efficiency metrics
    pub efficiency: EfficiencyMetrics<A>,

    /// Robustness metrics
    pub robustness: RobustnessMetrics<A>,

    /// Interpretability metrics
    pub interpretability: InterpretabilityMetrics<A>,
}

/// Performance metrics
#[derive(Debug)]
pub struct PerformanceMetrics<A: Float> {
    pub convergence_rate: A,
    pub final_performance: A,
    pub sample_efficiency: A,
    pub generalization_gap: A,
    pub stability_measure: A,
}

/// Efficiency metrics
#[derive(Debug)]
pub struct EfficiencyMetrics<A: Float> {
    pub computation_per_step: A,
    pub memory_usage: A,
    pub communication_overhead: A,
    pub energy_consumption: A,
    pub parallelization_efficiency: A,
}

/// Robustness metrics
#[derive(Debug)]
pub struct RobustnessMetrics<A: Float> {
    pub noise_tolerance: A,
    pub hyperparameter_sensitivity: A,
    pub distribution_shift_robustness: A,
    pub adversarial_robustness: A,
    pub catastrophic_forgetting_resistance: A,
}

/// Interpretability metrics
#[derive(Debug)]
pub struct InterpretabilityMetrics<A: Float> {
    pub decision_transparency: A,
    pub feature_importance_clarity: A,
    pub uncertainty_quantification: A,
    pub causal_attribution: A,
    pub human_understandability: A,
}

/// Optimizer performance database
#[derive(Debug)]
#[allow(dead_code)]
pub struct OptimizerPerformanceDatabase<A: Float> {
    /// Performance records
    performance_records: HashMap<String, Vec<PerformanceRecord<A>>>,

    /// Benchmark results
    benchmark_results: HashMap<String, BenchmarkResults<A>>,

    /// Meta-analysis results
    meta_analysis: HashMap<String, MetaAnalysisResults<A>>,

    /// Database statistics
    statistics: DatabaseStatistics,
}

/// Performance record
#[derive(Debug)]
#[allow(dead_code)]
pub struct PerformanceRecord<A: Float> {
    /// Optimizer configuration
    pub config: LearnedOptimizerConfig,

    /// Task context
    pub task_context: TaskContext<A>,

    /// Performance metrics
    pub metrics: NeuralOptimizerMetrics<A>,

    /// Timestamp
    pub timestamp: std::time::SystemTime,

    /// Validation status
    pub validation_status: ValidationStatus,
}

/// Validation status
#[derive(Debug, Clone)]
pub enum ValidationStatus {
    Validated,
    PendingValidation,
    ValidationFailed,
    NotValidated,
}

/// Benchmark results
#[derive(Debug)]
#[allow(dead_code)]
pub struct BenchmarkResults<A: Float> {
    /// Benchmark suite
    pub benchmark_suite: String,

    /// Individual results
    pub individual_results: Vec<BenchmarkResult<A>>,

    /// Aggregate statistics
    pub aggregate_stats: AggregateStatistics<A>,

    /// Ranking information
    pub ranking: RankingInformation,
}

/// Benchmark result
#[derive(Debug)]
pub struct BenchmarkResult<A: Float> {
    /// Test name
    pub test_name: String,

    /// Score
    pub score: A,

    /// Relative performance
    pub relative_performance: A,

    /// Confidence interval
    pub confidence_interval: (A, A),

    /// Test metadata
    pub metadata: TestMetadata,
}

/// Test metadata
#[derive(Debug, Clone)]
pub struct TestMetadata {
    pub difficulty_level: DifficultyLevel,
    pub test_category: TestCategory,
    pub expected_runtime: std::time::Duration,
    pub resource_requirements: ResourceRequirements,
}

/// Difficulty levels
#[derive(Debug, Clone)]
pub enum DifficultyLevel {
    Easy,
    Medium,
    Hard,
    Expert,
}

/// Test categories
#[derive(Debug, Clone)]
pub enum TestCategory {
    Convergence,
    Generalization,
    Robustness,
    Efficiency,
    Scalability,
}

/// Aggregate statistics
#[derive(Debug)]
pub struct AggregateStatistics<A: Float> {
    pub mean_score: A,
    pub median_score: A,
    pub std_deviation: A,
    pub min_score: A,
    pub max_score: A,
    pub percentiles: HashMap<u8, A>,
}

/// Ranking information
#[derive(Debug, Clone)]
pub struct RankingInformation {
    pub overall_rank: usize,
    pub category_ranks: HashMap<TestCategory, usize>,
    pub elo_rating: f64,
    pub trend: RankingTrend,
}

/// Ranking trend
#[derive(Debug, Clone)]
pub enum RankingTrend {
    Improving,
    Stable,
    Declining,
    Volatile,
}

/// Meta-analysis results
#[derive(Debug)]
pub struct MetaAnalysisResults<A: Float> {
    /// Effect sizes
    pub effect_sizes: HashMap<String, A>,

    /// Statistical significance
    pub significance_tests: HashMap<String, StatisticalTest<A>>,

    /// Moderator analysis
    pub moderator_analysis: ModeratorAnalysis<A>,

    /// Publication bias assessment
    pub publication_bias: PublicationBiasAssessment<A>,
}

/// Statistical test
#[derive(Debug)]
pub struct StatisticalTest<A: Float> {
    pub test_statistic: A,
    pub p_value: A,
    pub confidence_interval: (A, A),
    pub effect_size: A,
    pub power: A,
}

/// Moderator analysis
#[derive(Debug)]
pub struct ModeratorAnalysis<A: Float> {
    /// Categorical moderators
    pub categorical_moderators: HashMap<String, Vec<A>>,

    /// Continuous moderators
    pub continuous_moderators: HashMap<String, A>,

    /// Interaction effects
    pub interaction_effects: HashMap<String, A>,

    /// Explained variance
    pub explained_variance: A,
}

/// Publication bias assessment
#[derive(Debug)]
pub struct PublicationBiasAssessment<A: Float> {
    /// Funnel plot asymmetry
    pub funnel_plot_asymmetry: A,

    /// Egger's test
    pub eggers_test: StatisticalTest<A>,

    /// Trim and fill analysis
    pub trim_and_fill: TrimAndFillResults<A>,

    /// File drawer number
    pub file_drawer_number: usize,
}

/// Trim and fill results
#[derive(Debug)]
pub struct TrimAndFillResults<A: Float> {
    pub trimmed_studies: usize,
    pub filled_studies: usize,
    pub adjusted_effect_size: A,
    pub adjusted_confidence_interval: (A, A),
}

/// Database statistics
#[derive(Debug, Clone)]
pub struct DatabaseStatistics {
    pub total_records: usize,
    pub unique_optimizers: usize,
    pub unique_tasks: usize,
    pub date_range: (std::time::SystemTime, std::time::SystemTime),
    pub update_frequency: std::time::Duration,
}

/// Auto-selection criteria
#[derive(Debug)]
#[allow(dead_code)]
pub struct AutoSelectionCriteria<A: Float> {
    /// Primary objectives
    pub primary_objectives: Vec<OptimizationObjective>,

    /// Secondary objectives
    pub secondary_objectives: Vec<OptimizationObjective>,

    /// Constraints
    pub constraints: Vec<OptimizationConstraint<A>>,

    /// Preferences
    pub preferences: UserPreferences<A>,

    /// Risk tolerance
    pub risk_tolerance: RiskTolerance,
}

/// Optimization objectives
#[derive(Debug, Clone)]
pub enum OptimizationObjective {
    MaximizePerformance,
    MinimizeTime,
    MinimizeMemory,
    MaximizeRobustness,
    MaximizeInterpretability,
    MinimizeEnergy,
    MaximizeGeneralization,
}

/// Optimization constraints
#[derive(Debug)]
pub struct OptimizationConstraint<A: Float> {
    pub constraint_type: ConstraintType,
    pub threshold: A,
    pub priority: ConstraintPriority,
    pub violation_penalty: A,
}

/// Constraint types
#[derive(Debug, Clone)]
pub enum ConstraintType {
    MaxMemory,
    MaxTime,
    MinAccuracy,
    MaxComplexity,
    MinRobustness,
    MaxEnergy,
}

/// Constraint priorities
#[derive(Debug, Clone)]
pub enum ConstraintPriority {
    Hard,     // Must be satisfied
    Soft,     // Preferred but not required
    Flexible, // Can be relaxed if necessary
}

/// User preferences
#[derive(Debug)]
pub struct UserPreferences<A: Float> {
    /// Preferred optimizer families
    pub preferred_families: Vec<NeuralOptimizerType>,

    /// Avoided optimizer families
    pub avoided_families: Vec<NeuralOptimizerType>,

    /// Complexity preference
    pub complexity_preference: ComplexityPreference,

    /// Performance vs efficiency tradeoff
    pub performance_efficiency_tradeoff: A,

    /// Novelty preference
    pub novelty_preference: NoveltyPreference,
}

/// Complexity preferences
#[derive(Debug, Clone)]
pub enum ComplexityPreference {
    Simple,
    Moderate,
    Complex,
    NoPreference,
}

/// Novelty preferences
#[derive(Debug, Clone)]
pub enum NoveltyPreference {
    Conservative, // Prefer well-tested optimizers
    Moderate,     // Balance between tested and novel
    Innovative,   // Prefer cutting-edge optimizers
    Experimental, // Willing to try experimental optimizers
}

/// Risk tolerance levels
#[derive(Debug, Clone)]
pub enum RiskTolerance {
    VeryLow,
    Low,
    Medium,
    High,
    VeryHigh,
}

impl<A: Float> AdvancedNeuralOptimizerFactory<A> {
    /// Create new factory
    pub fn new() -> Self {
        Self {
            available_types: vec![
                NeuralOptimizerType::LSTM,
                NeuralOptimizerType::Transformer,
                NeuralOptimizerType::AttentionBased,
                NeuralOptimizerType::MemoryAugmented,
                NeuralOptimizerType::Hybrid,
            ],
            optimizer_registry: HashMap::new(),
            performance_db: OptimizerPerformanceDatabase::new(),
            auto_selection: AutoSelectionCriteria::default(),
        }
    }

    /// Register optimizer builder
    pub fn register_builder(&mut self, name: String, builder: Box<dyn NeuralOptimizerBuilder<A>>) {
        self.optimizer_registry.insert(name, builder);
    }

    /// Auto-select optimal optimizer
    #[allow(dead_code)]
    #[allow(clippy::too_many_arguments)]
    pub fn auto_select_optimizer(
        &self,
        _task_context: &TaskContext<A>,
        _criteria: &AutoSelectionCriteria<A>,
    ) -> Result<String> {
        // Implementation would use ML-based selection
        // This is a simplified placeholder
        Ok("LSTM".to_string())
    }

    /// Create optimizer
    pub fn create_optimizer(
        &self,
        optimizer_name: &str,
        config: &LearnedOptimizerConfig,
    ) -> Result<Box<dyn NeuralOptimizer<A>>> {
        let builder = self.optimizer_registry.get(optimizer_name).ok_or_else(|| {
            OptimError::InvalidConfig(format!("Unknown optimizer: {}", optimizer_name))
        })?;

        builder.build(config)
    }
}

impl<A: Float> OptimizerPerformanceDatabase<A> {
    pub fn new() -> Self {
        Self {
            performance_records: HashMap::new(),
            benchmark_results: HashMap::new(),
            meta_analysis: HashMap::new(),
            statistics: DatabaseStatistics {
                total_records: 0,
                unique_optimizers: 0,
                unique_tasks: 0,
                date_range: (std::time::SystemTime::now(), std::time::SystemTime::now()),
                update_frequency: std::time::Duration::from_secs(24 * 60 * 60),
            },
        }
    }
}

impl<A: Float> Default for AutoSelectionCriteria<A> {
    fn default() -> Self {
        Self {
            primary_objectives: vec![OptimizationObjective::MaximizePerformance],
            secondary_objectives: vec![OptimizationObjective::MinimizeTime],
            constraints: Vec::new(),
            preferences: UserPreferences {
                preferred_families: Vec::new(),
                avoided_families: Vec::new(),
                complexity_preference: ComplexityPreference::Moderate,
                performance_efficiency_tradeoff: A::from(0.5).unwrap(),
                novelty_preference: NoveltyPreference::Moderate,
            },
            risk_tolerance: RiskTolerance::Medium,
        }
    }
}

impl<A, D> LSTMOptimizer<A, D>
where
    A: Float
        + Default
        + Clone
        + Send
        + Sync
        + std::fmt::Debug
        + 'static
        + ndarray::ScalarOperand
        + std::iter::Sum
        + std::iter::Sum<A>
        + for<'a> std::iter::Sum<&'a A>,
    D: Dimension,
{
    /// Create a new LSTM optimizer
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        config: LearnedOptimizerConfig,
        meta_optimizer: Box<dyn Optimizer<A, D> + Send + Sync>,
    ) -> Result<Self> {
        let cell_state = LSTMState::new(&config)?;
        let parameters = LSTMParameters::new(&config)?;

        let gradient_history = VecDeque::with_capacity(config.gradient_history_size);
        let parameter_history = VecDeque::with_capacity(config.gradient_history_size);
        let loss_history = VecDeque::with_capacity(1000);

        let training_state = MetaTrainingState::new(&config)?;
        let metrics = LearnedOptimizerMetrics::default();

        Ok(Self {
            config,
            cell_state,
            parameters,
            gradient_history,
            parameter_history,
            loss_history,
            meta_optimizer,
            training_state,
            metrics,
            step_count: 0,
        })
    }

    /// Perform learned optimization step
    pub fn learned_step<S, Dim>(
        &mut self,
        params: &ArrayBase<S, Dim>,
        gradients: &ArrayBase<S, Dim>,
        loss: Option<A>,
    ) -> Result<Array<A, Dim>>
    where
        S: Data<Elem = A>,
        Dim: Dimension + Clone,
    {
        // Convert inputs to 1D for processing
        let flat_params = self.flatten_array(params)?;
        let flat_gradients = self.flatten_array(gradients)?;

        // Update history
        self.update_history(&flat_params, &flat_gradients, loss);

        // Prepare input features for LSTM
        let input_features = self.prepare_input_features(&flat_gradients)?;

        // Forward pass through LSTM
        let update_direction = self.lstm_forward(&input_features)?;

        // Compute learned learning rate
        let learned_lr = self.compute_learned_lr(&flat_gradients)?;

        // Apply update
        let mut flat_updated = flat_params.clone();
        for i in 0..flat_updated.len() {
            flat_updated[i] = flat_updated[i] - learned_lr * update_direction[i];
        }

        // Reshape back to original dimensions
        let updated_params = self.reshape_array(&flat_updated, params.raw_dim())?;

        self.step_count += 1;

        // Update metrics
        self.update_metrics(&flat_gradients, &update_direction, learned_lr);

        Ok(updated_params)
    }

    /// Meta-training step
    pub fn meta_train_step(&mut self, metabatch: Vec<MetaTask<A>>) -> Result<A> {
        let mut total_meta_loss = A::zero();
        let batch_size = A::from(metabatch.len()).unwrap();

        for task in &metabatch {
            let task_loss = self.train_on_task(task)?;
            total_meta_loss = total_meta_loss + task_loss;
        }

        let avg_meta_loss = total_meta_loss / batch_size;

        // Compute meta-gradients
        let meta_gradients = self.compute_meta_gradients(&metabatch)?;

        // Update meta-parameters
        self.update_meta_parameters(&meta_gradients)?;

        self.training_state.meta_step += 1;
        self.metrics.meta_training_loss = avg_meta_loss.to_f64().unwrap_or(0.0);

        Ok(avg_meta_loss)
    }

    fn update_history(&mut self, params: &Array1<A>, gradients: &Array1<A>, loss: Option<A>) {
        self.parameter_history.push_back(params.clone());
        self.gradient_history.push_back(gradients.clone());

        if let Some(l) = loss {
            self.loss_history.push_back(l);
        }

        // Maintain window size
        if self.parameter_history.len() > self.config.gradient_history_size {
            self.parameter_history.pop_front();
        }
        if self.gradient_history.len() > self.config.gradient_history_size {
            self.gradient_history.pop_front();
        }
        if self.loss_history.len() > 1000 {
            self.loss_history.pop_front();
        }
    }

    fn prepare_input_features(&self, gradients: &Array1<A>) -> Result<Array1<A>> {
        let mut features = Vec::new();

        // Current gradient
        features.extend_from_slice(gradients.as_slice().unwrap());

        // Gradient history features
        if let Some(prev_grad) = self.gradient_history.back() {
            // Gradient difference
            let grad_diff: Vec<A> = gradients
                .iter()
                .zip(prev_grad.iter())
                .map(|(&g1, &g2)| g1 - g2)
                .collect();
            features.extend_from_slice(&grad_diff);

            // Gradient ratio
            let grad_ratio: Vec<A> = gradients
                .iter()
                .zip(prev_grad.iter())
                .map(|(&g1, &g2)| {
                    if g2.abs() > A::from(1e-8).unwrap() {
                        g1 / g2
                    } else {
                        A::one()
                    }
                })
                .collect();
            features.extend_from_slice(&grad_ratio);
        }

        // Gradient statistics
        let grad_norm = gradients.iter().map(|&g| g * g).sum::<A>().sqrt();
        let grad_mean = gradients.iter().cloned().sum::<A>() / A::from(gradients.len()).unwrap();
        let grad_std = {
            let variance = gradients
                .iter()
                .map(|&g| (g - grad_mean) * (g - grad_mean))
                .sum::<A>()
                / A::from(gradients.len()).unwrap();
            variance.sqrt()
        };

        features.push(grad_norm);
        features.push(grad_mean);
        features.push(grad_std);

        // Loss history features (if available)
        if self.loss_history.len() >= 2 {
            let current_loss = *self.loss_history.back().unwrap();
            let prev_loss = self.loss_history[self.loss_history.len() - 2];
            let loss_change = current_loss - prev_loss;
            let loss_ratio = if prev_loss.abs() > A::from(1e-8).unwrap() {
                current_loss / prev_loss
            } else {
                A::one()
            };

            features.push(loss_change);
            features.push(loss_ratio);
        }

        // Pad or truncate to input_features size
        features.resize(self.config.input_features, A::zero());

        Ok(Array1::from_vec(features))
    }

    fn lstm_forward(&mut self, input: &Array1<A>) -> Result<Array1<A>> {
        let mut current_input = input.clone();

        // Forward pass through LSTM layers
        for layer in 0..self.config.num_layers {
            let (hidden_output, cell_output) = self.lstm_cell_forward(
                &current_input,
                &self.cell_state.hidden_states[layer],
                &self.cell_state.cell_states[layer],
                layer,
            )?;

            self.cell_state.hidden_states[layer] = hidden_output.clone();
            self.cell_state.cell_states[layer] = cell_output;
            current_input = hidden_output;
        }

        // Apply attention if enabled
        if self.config.use_attention {
            current_input = self.apply_attention(&current_input)?;
        }

        // Final output projection
        let output =
            self.parameters.output_weights.dot(&current_input) + &self.parameters.output_bias;

        Ok(output)
    }

    fn lstm_cell_forward(
        &self,
        input: &Array1<A>,
        hidden: &Array1<A>,
        cell: &Array1<A>,
        layer: usize,
    ) -> Result<(Array1<A>, Array1<A>)> {
        // LSTM cell computation: i_t, f_t, g_t, o_t = σ(W_ih @ x_t + W_hh @ h_{t-1} + b)
        let ih_linear =
            self.parameters.weight_ih[layer].dot(input) + &self.parameters.bias_ih[layer];
        let hh_linear =
            self.parameters.weight_hh[layer].dot(hidden) + &self.parameters.bias_hh[layer];
        let combined = ih_linear + hh_linear;

        let hidden_size = self.config.hidden_size;

        // Split into gates
        let input_gate = self.sigmoid(&combined.slice(s![0..hidden_size]).to_owned());
        let forget_gate =
            self.sigmoid(&combined.slice(s![hidden_size..2 * hidden_size]).to_owned());
        let cell_gate = self.tanh(
            &combined
                .slice(s![2 * hidden_size..3 * hidden_size])
                .to_owned(),
        );
        let output_gate = self.sigmoid(
            &combined
                .slice(s![3 * hidden_size..4 * hidden_size])
                .to_owned(),
        );

        // Update cell state
        let new_cell = &forget_gate * cell + &input_gate * &cell_gate;

        // Update hidden state
        let new_hidden = &output_gate * &self.tanh(&new_cell);

        Ok((new_hidden, new_cell))
    }

    fn apply_attention(&mut self, hidden: &Array1<A>) -> Result<Array1<A>> {
        if let Some(ref attention_params) = self.parameters.attention_params {
            // Simplified attention mechanism
            // In practice, this would implement multi-head attention
            let query = attention_params.query_weights.dot(hidden);
            let key = attention_params.key_weights.dot(hidden);
            let value = attention_params.value_weights.dot(hidden);

            // Compute attention weights (simplified)
            let attention_score = query.dot(&key);
            let attention_weight = self.softmax(&Array1::from_vec(vec![attention_score]))[0];

            let attended_output = value * attention_weight;
            self.cell_state.context_vector = Some(attended_output.clone());

            Ok(attended_output)
        } else {
            Ok(hidden.clone())
        }
    }

    fn compute_learned_lr(&self, gradients: &Array1<A>) -> Result<A> {
        if let Some(ref lr_params) = self.parameters.lr_params {
            // Adaptive learning rate based on gradient statistics
            let grad_norm = gradients.iter().map(|&g| g * g).sum::<A>().sqrt();
            let adaptive_factor =
                lr_params.adaptive_factors[0] / (grad_norm + A::from(1e-8).unwrap());

            let learned_lr = lr_params.base_lr * adaptive_factor;
            Ok(learned_lr)
        } else {
            // Fallback to base learning rate
            Ok(A::from(self.config.meta_learning_rate).unwrap())
        }
    }

    fn train_on_task(&mut self, task: &MetaTask<A>) -> Result<A> {
        // Simplified task training
        let mut task_loss = A::zero();
        let data_size = A::from(task.train_data.len()).unwrap();

        for (features, target) in &task.train_data {
            // Compute prediction (simplified)
            let prediction = features.iter().sum::<A>() / A::from(features.len()).unwrap();
            let loss = (prediction - *target) * (prediction - *target);
            task_loss = task_loss + loss;
        }

        Ok(task_loss / data_size)
    }

    fn compute_meta_gradients(
        &self,
        _meta_batch: &[MetaTask<A>],
    ) -> Result<HashMap<String, Array1<A>>> {
        // Simplified meta-gradient computation
        // In practice, this would compute gradients of the meta-objective
        let mut meta_grads = HashMap::new();

        // Placeholder meta-gradients
        meta_grads.insert(
            "weight_ih_0".to_string(),
            Array1::zeros(self.config.hidden_size * 4),
        );
        meta_grads.insert(
            "weight_hh_0".to_string(),
            Array1::zeros(self.config.hidden_size * 4),
        );

        Ok(meta_grads)
    }

    fn update_meta_parameters(
        &mut self,
        _meta_gradients: &HashMap<String, Array1<A>>,
    ) -> Result<()> {
        // Update meta-parameters using meta-optimizer
        // This is simplified - in practice would update all LSTM parameters
        Ok(())
    }

    fn update_metrics(&mut self, gradients: &Array1<A>, updates: &Array1<A>, lr: A) {
        let grad_norm = gradients.iter().map(|&g| g * g).sum::<A>().sqrt();
        let update_norm = updates.iter().map(|&u| u * u).sum::<A>().sqrt();

        self.metrics.meta_gradient_norm = grad_norm.to_f64().unwrap_or(0.0);
        self.metrics.parameter_efficiency = (update_norm / grad_norm).to_f64().unwrap_or(1.0);

        // Update other metrics based on current performance
        if self.step_count > 0 {
            self.metrics.avg_convergence_speed = 1.0 / self.step_count as f64;
        }
    }

    // Utility functions
    fn flatten_array<S, Dim>(&self, array: &ArrayBase<S, Dim>) -> Result<Array1<A>>
    where
        S: Data<Elem = A>,
        Dim: Dimension,
    {
        Ok(Array1::from_vec(array.iter().cloned().collect()))
    }

    fn reshape_array<Dim>(&self, flatarray: &Array1<A>, shape: Dim) -> Result<Array<A, Dim>>
    where
        Dim: Dimension + Clone,
    {
        // Simplified reshape - in practice would handle arbitrary dimensions
        Array::from_shape_vec(shape, flatarray.to_vec())
            .map_err(|_| OptimError::InvalidConfig("Reshape error".to_string()))
    }

    fn sigmoid(&self, x: &Array1<A>) -> Array1<A> {
        x.mapv(|xi| A::one() / (A::one() + (-xi).exp()))
    }

    fn tanh(&self, x: &Array1<A>) -> Array1<A> {
        x.mapv(|xi| xi.tanh())
    }

    fn softmax(&self, x: &Array1<A>) -> Array1<A> {
        let exp_x = x.mapv(|xi| xi.exp());
        let sum_exp = exp_x.sum();
        exp_x / sum_exp
    }

    /// Get current performance metrics
    pub fn get_metrics(&self) -> &LearnedOptimizerMetrics {
        &self.metrics
    }

    /// Save learned optimizer state
    pub fn save_state(&self) -> LearnedOptimizerState<A> {
        LearnedOptimizerState {
            parameters: self.parameters.clone(),
            cell_state: self.cell_state.clone(),
            training_state: self.training_state.clone(),
            step_count: self.step_count,
        }
    }

    /// Load learned optimizer state
    pub fn load_state(&mut self, state: LearnedOptimizerState<A>) -> Result<()> {
        self.parameters = state.parameters;
        self.cell_state = state.cell_state;
        self.training_state = state.training_state;
        self.step_count = state.step_count;
        Ok(())
    }

    /// Transfer learning to new domain
    pub fn transfer_to_domain(
        &mut self,
        target_tasks: &[MetaTask<A>],
    ) -> Result<TransferResults<A>> {
        let initial_performance = self.evaluate_on_tasks(target_tasks)?;

        // Fine-tune on target domain
        let mut transfer_state = TransferLearningState {
            source_performance: A::from(self.metrics.generalization_performance).unwrap(),
            target_performance: initial_performance,
            transfer_efficiency: A::zero(),
            adapted_params: HashMap::new(),
            finetuning_steps: 0,
        };

        // Simplified fine-tuning process
        for _ in 0..100 {
            // 100 fine-tuning steps
            let batch: Vec<_> = target_tasks.iter().take(5).cloned().collect();
            self.meta_train_step(batch)?;
            transfer_state.finetuning_steps += 1;
        }

        let final_performance = self.evaluate_on_tasks(target_tasks)?;
        transfer_state.target_performance = final_performance;
        transfer_state.transfer_efficiency = final_performance / initial_performance;

        self.training_state.transfer_state = Some(transfer_state.clone());

        Ok(TransferResults {
            initial_performance,
            final_performance,
            transfer_state,
            adaptation_steps: 100,
        })
    }

    fn evaluate_on_tasks(&self, tasks: &[MetaTask<A>]) -> Result<A> {
        // Simplified evaluation
        let mut total_performance = A::zero();
        for task in tasks {
            let task_perf = self.evaluate_single_task(task)?;
            total_performance = total_performance + task_perf;
        }
        Ok(total_performance / A::from(tasks.len()).unwrap())
    }

    fn evaluate_single_task(&self, task: &MetaTask<A>) -> Result<A> {
        // Simplified single task evaluation
        let mut loss = A::zero();
        for (features, target) in &task.val_data {
            let prediction = features.iter().sum::<A>() / A::from(features.len()).unwrap();
            loss = loss + (prediction - *target) * (prediction - *target);
        }
        Ok(loss / A::from(task.val_data.len()).unwrap())
    }

    /// Perform learned step using LSTM
    pub fn lstm_step<S, Dim>(
        &mut self,
        parameters: &ArrayBase<S, Dim>,
        gradients: &ArrayBase<S, Dim>,
        _loss: Option<A>,
    ) -> Result<Array<A, Dim>>
    where
        S: Data<Elem = A>,
        Dim: Dimension + Clone,
    {
        // Simplified implementation - just apply simple gradient descent
        // In a full implementation, this would use the LSTM network
        let lr = A::from(0.01).unwrap(); // Simple learning rate
        let updates = gradients.mapv(|g| g * lr);
        let updated_params = parameters - &updates;
        Ok(updated_params.to_owned())
    }
}

/// Saved state for learned optimizers
#[derive(Debug, Clone)]
pub struct LearnedOptimizerState<A: Float> {
    pub parameters: LSTMParameters<A>,
    pub cell_state: LSTMState<A>,
    pub training_state: MetaTrainingState<A>,
    pub step_count: usize,
}

/// Transfer learning results
#[derive(Debug, Clone)]
pub struct TransferResults<A: Float> {
    pub initial_performance: A,
    pub final_performance: A,
    pub transfer_state: TransferLearningState<A>,
    pub adaptation_steps: usize,
}

// Implementation of initialization functions

impl<A: Float + Default + Clone> LSTMState<A> {
    fn new(config: &LearnedOptimizerConfig) -> Result<Self> {
        let mut hidden_states = Vec::new();
        let mut cell_states = Vec::new();

        for _ in 0..config.num_layers {
            hidden_states.push(Array1::zeros(config.hidden_size));
            cell_states.push(Array1::zeros(config.hidden_size));
        }

        Ok(Self {
            hidden_states,
            cell_states,
            attention_weights: if config.use_attention {
                Some(Array2::zeros((config.attention_heads, config.hidden_size)))
            } else {
                None
            },
            context_vector: None,
        })
    }
}

impl<A: Float + Default + Clone> LSTMParameters<A> {
    fn new(config: &LearnedOptimizerConfig) -> Result<Self> {
        let mut weight_ih = Vec::new();
        let mut weight_hh = Vec::new();
        let mut bias_ih = Vec::new();
        let mut bias_hh = Vec::new();

        for layer in 0..config.num_layers {
            let input_size = if layer == 0 {
                config.input_features
            } else {
                config.hidden_size
            };
            let hidden_size = config.hidden_size;

            // Xavier initialization
            let fan_in = input_size as f64;
            let fan_out = hidden_size as f64;
            let scale = (6.0 / (fan_in + fan_out)).sqrt();

            weight_ih.push(Self::random_array_2d(4 * hidden_size, input_size, scale));
            weight_hh.push(Self::random_array_2d(4 * hidden_size, hidden_size, scale));
            bias_ih.push(Array1::zeros(4 * hidden_size));
            bias_hh.push(Array1::zeros(4 * hidden_size));
        }

        let output_weights = Self::random_array_2d(config.output_features, config.hidden_size, 0.1);
        let output_bias = Array1::zeros(config.output_features);

        let attention_params = if config.use_attention {
            Some(AttentionParameters::new(config)?)
        } else {
            None
        };

        let lr_params = if config.learned_lr_schedule {
            Some(LearningRateParameters::new(config)?)
        } else {
            None
        };

        Ok(Self {
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
            output_weights,
            output_bias,
            attention_params,
            lr_params,
        })
    }

    fn random_array_2d(rows: usize, cols: usize, scale: f64) -> Array2<A> {
        // Simplified random initialization
        Array2::zeros((rows, cols)).mapv(|_: A| {
            A::from(scale * (scirs2_core::random::rng().gen_range(-0.5f64..0.5f64))).unwrap()
        })
    }
}

impl<A: Float + Default + Clone> AttentionParameters<A> {
    fn new(config: &LearnedOptimizerConfig) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let scale = 0.1;

        Ok(Self {
            query_weights: LSTMParameters::random_array_2d(hidden_size, hidden_size, scale),
            key_weights: LSTMParameters::random_array_2d(hidden_size, hidden_size, scale),
            value_weights: LSTMParameters::random_array_2d(hidden_size, hidden_size, scale),
            output_weights: LSTMParameters::random_array_2d(hidden_size, hidden_size, scale),
            head_weights: (0..config.attention_heads)
                .map(|_| LSTMParameters::random_array_2d(hidden_size, hidden_size, scale))
                .collect(),
        })
    }
}

impl<A: Float + Default + Clone> LearningRateParameters<A> {
    fn new(config: &LearnedOptimizerConfig) -> Result<Self> {
        Ok(Self {
            base_lr: A::from(config.meta_learning_rate).unwrap(),
            adaptive_factors: Array1::ones(config.output_features),
            schedule_params: Array1::zeros(4), // Parameters for schedule
            decay_params: Array1::zeros(2),    // Decay parameters
        })
    }
}

impl<A: Float + Default + Clone> MetaTrainingState<A> {
    fn new(config: &LearnedOptimizerConfig) -> Result<Self> {
        Ok(Self {
            meta_step: 0,
            meta_gradients: HashMap::new(),
            task_performance: VecDeque::with_capacity(1000),
            current_meta_batch: Vec::new(),
            meta_validation: MetaValidationMetrics {
                avg_task_performance: A::zero(),
                performance_variance: A::zero(),
                generalization_error: A::zero(),
                adaptation_speed: A::zero(),
                task_diversity: A::zero(),
            },
            transfer_state: None,
        })
    }
}

impl Default for LearnedOptimizerMetrics {
    fn default() -> Self {
        Self {
            meta_training_loss: 0.0,
            avg_convergence_speed: 0.0,
            generalization_performance: 0.0,
            parameter_efficiency: 1.0,
            transfer_success_rate: 0.0,
            computational_overhead: 1.0,
            memory_usage_mb: 0.0,
            meta_gradient_norm: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizers::SGD;

    #[test]
    fn test_learned_optimizer_config_default() {
        let config = LearnedOptimizerConfig::default();
        assert_eq!(config.hidden_size, 256);
        assert_eq!(config.num_layers, 2);
        assert!(config.use_attention);
        assert!(matches!(config.optimizer_type, NeuralOptimizerType::LSTM));
    }

    #[test]
    fn test_lstm_state_creation() {
        let config = LearnedOptimizerConfig::default();
        let state = LSTMState::<f64>::new(&config);
        assert!(state.is_ok());

        let state = state.unwrap();
        assert_eq!(state.hidden_states.len(), config.num_layers);
        assert_eq!(state.cell_states.len(), config.num_layers);
        assert!(state.attention_weights.is_some());
    }

    #[test]
    fn test_lstm_parameters_creation() {
        let config = LearnedOptimizerConfig::default();
        let params = LSTMParameters::<f64>::new(&config);
        assert!(params.is_ok());

        let params = params.unwrap();
        assert_eq!(params.weight_ih.len(), config.num_layers);
        assert_eq!(params.weight_hh.len(), config.num_layers);
        assert!(params.attention_params.is_some());
        assert!(params.lr_params.is_some());
    }

    #[test]
    fn test_lstm_optimizer_creation() {
        let config = LearnedOptimizerConfig::default();
        let meta_optimizer = Box::new(SGD::new(0.001));

        let optimizer = LSTMOptimizer::<f64>::new(config, meta_optimizer);
        assert!(optimizer.is_ok());

        let optimizer = optimizer.unwrap();
        assert_eq!(optimizer.step_count, 0);
        assert!(optimizer.gradient_history.is_empty());
    }

    #[test]
    fn test_metrics_default() {
        let metrics = LearnedOptimizerMetrics::default();
        assert_eq!(metrics.meta_training_loss, 0.0);
        assert_eq!(metrics.parameter_efficiency, 1.0);
        assert_eq!(metrics.transfer_success_rate, 0.0);
    }
}
