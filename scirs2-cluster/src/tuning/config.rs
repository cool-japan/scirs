//! Hyperparameter tuning configuration types and structures
//!
//! This module contains all configuration structures and enums used for
//! hyperparameter optimization in clustering algorithms.

use scirs2_core::ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Hyperparameter tuning configuration
#[derive(Debug, Clone)]
pub struct TuningConfig {
    /// Search strategy to use
    pub strategy: SearchStrategy,
    /// Evaluation metric for optimization
    pub metric: EvaluationMetric,
    /// Cross-validation configuration
    pub cv_config: CrossValidationConfig,
    /// Maximum number of evaluations
    pub max_evaluations: usize,
    /// Early stopping criteria
    pub early_stopping: Option<EarlyStoppingConfig>,
    /// Random seed for reproducible results
    pub random_seed: Option<u64>,
    /// Parallel evaluation configuration
    pub parallel_config: Option<ParallelConfig>,
    /// Resource constraints
    pub resource_constraints: ResourceConstraints,
}

impl Default for TuningConfig {
    fn default() -> Self {
        Self {
            strategy: SearchStrategy::RandomSearch { n_trials: 50 },
            metric: EvaluationMetric::SilhouetteScore,
            cv_config: CrossValidationConfig::default(),
            max_evaluations: 100,
            early_stopping: None,
            random_seed: Some(42),
            parallel_config: None,
            resource_constraints: ResourceConstraints::default(),
        }
    }
}

/// Hyperparameter search strategies
#[derive(Debug, Clone)]
pub enum SearchStrategy {
    /// Exhaustive grid search
    GridSearch,
    /// Random search with specified number of trials
    RandomSearch { n_trials: usize },
    /// Bayesian optimization using Gaussian processes
    BayesianOptimization {
        n_initial_points: usize,
        acquisition_function: AcquisitionFunction,
    },
    /// Adaptive search that adjusts based on results
    AdaptiveSearch {
        initial_strategy: Box<SearchStrategy>,
        adaptation_frequency: usize,
    },
    /// Multi-objective optimization
    MultiObjective {
        objectives: Vec<EvaluationMetric>,
        strategy: Box<SearchStrategy>,
    },
    /// Ensemble search combining multiple strategies
    EnsembleSearch {
        strategies: Vec<SearchStrategy>,
        weights: Vec<f64>,
    },
    /// Evolutionary search strategy
    EvolutionarySearch {
        population_size: usize,
        n_generations: usize,
        mutation_rate: f64,
        crossover_rate: f64,
    },
    /// Sequential model-based optimization
    SMBO {
        surrogate_model: SurrogateModel,
        acquisition_function: AcquisitionFunction,
    },
}

/// Acquisition functions for Bayesian optimization
#[derive(Debug, Clone)]
pub enum AcquisitionFunction {
    /// Expected Improvement
    ExpectedImprovement,
    /// Upper Confidence Bound
    UpperConfidenceBound { beta: f64 },
    /// Probability of Improvement
    ProbabilityOfImprovement,
    /// Entropy Search
    EntropySearch,
    /// Knowledge Gradient
    KnowledgeGradient,
    /// Thompson Sampling
    ThompsonSampling,
}

/// Surrogate models for SMBO
#[derive(Debug, Clone)]
pub enum SurrogateModel {
    /// Gaussian Process
    GaussianProcess { kernel: KernelType, noise: f64 },
    /// Random Forest
    RandomForest {
        n_trees: usize,
        max_depth: Option<usize>,
    },
    /// Gradient Boosting
    GradientBoosting {
        n_estimators: usize,
        learning_rate: f64,
    },
}

/// Kernel types for Gaussian Processes
#[derive(Debug, Clone)]
pub enum KernelType {
    /// Radial Basis Function (RBF)
    RBF { length_scale: f64 },
    /// Matérn kernel
    Matern { length_scale: f64, nu: f64 },
    /// Linear kernel
    Linear,
    /// Polynomial kernel
    Polynomial { degree: usize },
}

/// Evaluation metrics for hyperparameter optimization
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvaluationMetric {
    /// Silhouette coefficient (higher is better)
    SilhouetteScore,
    /// Davies-Bouldin index (lower is better)
    DaviesBouldinIndex,
    /// Calinski-Harabasz index (higher is better)
    CalinskiHarabaszIndex,
    /// Within-cluster sum of squares (lower is better)
    Inertia,
    /// Adjusted Rand Index (for labeled data)
    AdjustedRandIndex,
    /// Custom metric
    Custom(String),
    /// Ensemble consensus score
    EnsembleConsensus,
    /// Stability-based metrics
    Stability,
    /// Information-theoretic metrics
    MutualInformation,
}

/// Cross-validation configuration
#[derive(Debug, Clone)]
pub struct CrossValidationConfig {
    /// Number of folds
    pub n_folds: usize,
    /// Fraction of data to use for validation
    pub validation_ratio: f64,
    /// Strategy for cross-validation
    pub strategy: CVStrategy,
    /// Shuffle data before splitting
    pub shuffle: bool,
}

impl Default for CrossValidationConfig {
    fn default() -> Self {
        Self {
            n_folds: 5,
            validation_ratio: 0.2,
            strategy: CVStrategy::KFold,
            shuffle: true,
        }
    }
}

/// Cross-validation strategies
#[derive(Debug, Clone)]
pub enum CVStrategy {
    /// K-fold cross-validation
    KFold,
    /// Stratified K-fold (for labeled data)
    StratifiedKFold,
    /// Time series split (preserves temporal order)
    TimeSeriesSplit,
    /// Bootstrap cross-validation
    Bootstrap { n_bootstrap: usize },
    /// Ensemble cross-validation (multiple CV strategies)
    EnsembleCV { strategies: Vec<CVStrategy> },
    /// Monte Carlo cross-validation
    MonteCarlo { n_splits: usize, test_size: f64 },
    /// Nested cross-validation
    NestedCV {
        outer_folds: usize,
        inner_folds: usize,
    },
}

/// Early stopping configuration
#[derive(Debug, Clone)]
pub struct EarlyStoppingConfig {
    /// Patience (number of evaluations without improvement)
    pub patience: usize,
    /// Minimum improvement required
    pub min_improvement: f64,
    /// Evaluation frequency
    pub evaluation_frequency: usize,
}

/// Parallel evaluation configuration
#[derive(Debug, Clone)]
pub struct ParallelConfig {
    /// Number of parallel workers
    pub n_workers: usize,
    /// Batch size for parallel evaluation
    pub batch_size: usize,
    /// Load balancing strategy
    pub load_balancing: LoadBalancingStrategy,
}

/// Load balancing strategies for parallel evaluation
#[derive(Debug, Clone)]
pub enum LoadBalancingStrategy {
    /// Round-robin assignment
    RoundRobin,
    /// Work stealing
    WorkStealing,
    /// Dynamic load balancing
    Dynamic,
}

/// Resource constraints for hyperparameter tuning
#[derive(Debug, Clone, Default)]
pub struct ResourceConstraints {
    /// Maximum memory usage per evaluation (bytes)
    pub max_memory_per_evaluation: Option<usize>,
    /// Maximum time per evaluation (seconds)
    pub max_time_per_evaluation: Option<f64>,
    /// Maximum total tuning time (seconds)
    pub max_total_time: Option<f64>,
}

/// Hyperparameter specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HyperParameter {
    /// Integer parameter with range [min, max]
    Integer { min: i64, max: i64 },
    /// Float parameter with range [min, max]
    Float { min: f64, max: f64 },
    /// Categorical parameter with choices
    Categorical { choices: Vec<String> },
    /// Boolean parameter
    Boolean,
    /// Log-uniform distribution for float parameters
    LogUniform { min: f64, max: f64 },
    /// Discrete choices for integer parameters
    IntegerChoices { choices: Vec<i64> },
}

/// Hyperparameter search space for clustering algorithms
#[derive(Debug, Clone)]
pub struct SearchSpace {
    /// Parameters to optimize
    pub parameters: HashMap<String, HyperParameter>,
    /// Algorithm-specific constraints
    pub constraints: Vec<ParameterConstraint>,
}

/// Parameter constraints for interdependent hyperparameters
#[derive(Debug, Clone)]
pub enum ParameterConstraint {
    /// Conditional constraint: if condition then constraint
    Conditional {
        condition: String,
        constraint: Box<ParameterConstraint>,
    },
    /// Range constraint: parameter must be in range
    Range {
        parameter: String,
        min: f64,
        max: f64,
    },
    /// Dependency constraint: parameter A depends on parameter B
    Dependency {
        dependent: String,
        dependency: String,
        relationship: DependencyRelationship,
    },
}

/// Dependency relationships between parameters
#[derive(Debug, Clone)]
pub enum DependencyRelationship {
    /// Linear relationship: A = k * B + c
    Linear { k: f64, c: f64 },
    /// Proportional: A <= ratio * B
    Proportional { ratio: f64 },
    /// Custom function
    Custom(String),
}

/// Hyperparameter evaluation result
#[derive(Debug, Clone)]
pub struct EvaluationResult {
    /// Parameter values used
    pub parameters: HashMap<String, f64>,
    /// Primary metric score
    pub score: f64,
    /// Additional metrics
    pub additional_metrics: HashMap<String, f64>,
    /// Evaluation time (seconds)
    pub evaluation_time: f64,
    /// Memory usage (bytes)
    pub memory_usage: Option<usize>,
    /// Cross-validation scores
    pub cv_scores: Vec<f64>,
    /// Standard deviation of CV scores
    pub cv_std: f64,
    /// Algorithm-specific metadata
    pub metadata: HashMap<String, String>,
}

/// Tuning results
#[derive(Debug, Clone)]
pub struct TuningResult {
    /// Best parameter configuration found
    pub best_parameters: HashMap<String, f64>,
    /// Best score achieved
    pub best_score: f64,
    /// All evaluation results
    pub evaluation_history: Vec<EvaluationResult>,
    /// Convergence information
    pub convergence_info: ConvergenceInfo,
    /// Search space exploration statistics
    pub exploration_stats: ExplorationStats,
    /// Total tuning time
    pub total_time: f64,
    /// Ensemble results (if ensemble method was used)
    pub ensemble_results: Option<EnsembleResults>,
    /// Pareto front (for multi-objective optimization)
    pub pareto_front: Option<Vec<HashMap<String, f64>>>,
}

/// Results from ensemble tuning
#[derive(Debug, Clone)]
pub struct EnsembleResults {
    /// Results from each ensemble member
    pub member_results: Vec<TuningResult>,
    /// Consensus best parameters
    pub consensus_parameters: HashMap<String, f64>,
    /// Agreement score between ensemble members
    pub agreement_score: f64,
    /// Diversity metrics
    pub diversity_metrics: HashMap<String, f64>,
}

/// Bayesian optimization state
#[derive(Debug, Clone)]
pub struct BayesianState {
    /// Observed parameters and scores
    pub observations: Vec<(HashMap<String, f64>, f64)>,
    /// Gaussian process mean function
    pub gp_mean: Option<f64>,
    /// Gaussian process covariance matrix
    pub gp_covariance: Option<Array2<f64>>,
    /// Acquisition function values
    pub acquisition_values: Vec<f64>,
    /// Parameter names for consistent ordering
    pub parameter_names: Vec<String>,
    /// GP hyperparameters
    pub gp_hyperparameters: GpHyperparameters,
    /// Noise level
    pub noise_level: f64,
    /// Current best observed value
    pub currentbest: f64,
}

/// Gaussian Process hyperparameters
#[derive(Debug, Clone)]
pub struct GpHyperparameters {
    /// Length scales for each dimension
    pub length_scales: Vec<f64>,
    /// Signal variance
    pub signal_variance: f64,
    /// Noise variance
    pub noise_variance: f64,
    /// Kernel type
    pub kernel_type: KernelType,
}

/// Convergence information
#[derive(Debug, Clone)]
pub struct ConvergenceInfo {
    /// Whether tuning converged
    pub converged: bool,
    /// Iteration at which convergence was detected
    pub convergence_iteration: Option<usize>,
    /// Reason for stopping
    pub stopping_reason: StoppingReason,
}

/// Reasons for stopping hyperparameter tuning
#[derive(Debug, Clone)]
pub enum StoppingReason {
    /// Maximum evaluations reached
    MaxEvaluations,
    /// Early stopping triggered
    EarlyStopping,
    /// Time limit exceeded
    TimeLimit,
    /// Convergence achieved
    Convergence,
    /// User interruption
    UserInterruption,
    /// Resource constraints
    ResourceConstraints,
}

/// Search space exploration statistics
#[derive(Debug, Clone)]
pub struct ExplorationStats {
    /// Parameter space coverage
    pub coverage: f64,
    /// Distribution of parameter values explored
    pub parameter_distributions: HashMap<String, Vec<f64>>,
    /// Correlation between parameters and performance
    pub parameter_importance: HashMap<String, f64>,
}
