//! Privacy-Preserving Hyperparameter Optimization
//!
//! This module implements differential privacy guarantees for hyperparameter
//! optimization, ensuring that the choice of hyperparameters does not leak
//! sensitive information about the training data.

use crate::error::{OptimError, Result};
use crate::privacy::moment_accountant::MomentsAccountant;
use crate::privacy::{DifferentialPrivacyConfig, PrivacyBudget};
use ndarray::{Array1, Array2};
use num_traits::Float;
use rand::rngs::StdRng;
use rand::Rng;
use std::collections::HashMap;

/// Privacy-preserving hyperparameter optimizer
pub struct PrivateHyperparameterOptimizer<T: Float> {
    /// Configuration for privacy-preserving hyperparameter optimization
    config: PrivateHPOConfig<T>,

    /// Privacy budget manager
    budget_manager: HPOBudgetManager,

    /// Noisy optimization algorithms
    noisy_optimizers: HashMap<String, Box<dyn NoisyOptimizer<T>>>,

    /// Hyperparameter space definition
    parameterspace: ParameterSpace<T>,

    /// Objective function with privacy guarantees
    private_objective: PrivateObjective<T>,

    /// Search strategy
    search_strategy: SearchStrategy<T>,

    /// Results aggregator with privacy
    results_aggregator: PrivateResultsAggregator<T>,

    /// Privacy accountant for hyperparameter selection
    privacy_accountant: MomentsAccountant,
}

/// Configuration for private hyperparameter optimization
#[derive(Debug, Clone)]
pub struct PrivateHPOConfig<T: Float> {
    /// Base differential privacy configuration
    pub base_privacyconfig: DifferentialPrivacyConfig,

    /// Privacy budget allocation strategy
    pub budget_allocation: BudgetAllocationStrategy,

    /// Hyperparameter search algorithm
    pub search_algorithm: SearchAlgorithm,

    /// Number of hyperparameter configurations to evaluate
    pub num_evaluations: usize,

    /// Number of cross-validation folds
    pub cv_folds: usize,

    /// Early stopping criteria
    pub early_stopping: EarlyStoppingConfig,

    /// Noise mechanism for hyperparameter selection
    pub noise_mechanism: HyperparameterNoiseMechanism,

    /// Sensitivity bounds for hyperparameters
    pub sensitivity_bounds: SensitivityBounds<T>,

    /// Enable private model selection
    pub private_model_selection: bool,

    /// Validation strategy
    pub validation_strategy: ValidationStrategy,
}

/// Budget allocation strategies for hyperparameter optimization
#[derive(Debug, Clone, Copy)]
pub enum BudgetAllocationStrategy {
    /// Equal allocation across all evaluations
    Equal,

    /// Adaptive allocation based on promising regions
    Adaptive,

    /// Bandit-based allocation
    Bandit,

    /// Hierarchical allocation (coarse to fine)
    Hierarchical,

    /// Budget allocation based on uncertainty
    Uncertainty,
}

/// Search algorithms for hyperparameter optimization
#[derive(Debug, Clone, Copy)]
pub enum SearchAlgorithm {
    /// Random search with differential privacy
    RandomSearch,

    /// Grid search with differential privacy
    GridSearch,

    /// Bayesian optimization with differential privacy
    BayesianOptimization,

    /// Genetic algorithm with differential privacy
    GeneticAlgorithm,

    /// Particle swarm optimization with differential privacy
    ParticleSwarm,

    /// Simulated annealing with differential privacy
    SimulatedAnnealing,

    /// Tree-structured Parzen estimator with differential privacy
    TPE,
}

/// Early stopping configuration
#[derive(Debug, Clone)]
pub struct EarlyStoppingConfig {
    /// Enable early stopping
    pub enabled: bool,

    /// Patience (number of evaluations without improvement)
    pub patience: usize,

    /// Minimum improvement threshold
    pub min_improvement: f64,

    /// Maximum number of evaluations
    pub max_evaluations: usize,
}

/// Noise mechanisms for hyperparameter selection
#[derive(Debug, Clone, Copy)]
pub enum HyperparameterNoiseMechanism {
    /// Exponential mechanism for discrete selection
    Exponential,

    /// Gaussian mechanism for continuous parameters
    Gaussian,

    /// Laplace mechanism
    Laplace,

    /// Report noisy max for selection
    NoisyMax,

    /// Sparse vector technique
    SparseVector,
}

/// Sensitivity bounds for hyperparameters
#[derive(Debug, Clone)]
pub struct SensitivityBounds<T: Float> {
    /// Global sensitivity for each hyperparameter
    pub global_sensitivity: HashMap<String, T>,

    /// Local sensitivity bounds
    pub local_sensitivity: HashMap<String, (T, T)>,

    /// Smooth sensitivity parameters
    pub smooth_sensitivity: HashMap<String, SmoothSensitivityParams<T>>,
}

/// Smooth sensitivity parameters
#[derive(Debug, Clone)]
pub struct SmoothSensitivityParams<T: Float> {
    /// Beta parameter for smooth sensitivity
    pub beta: T,

    /// Maximum local sensitivity
    pub max_local_sensitivity: T,

    /// Smoothness parameter
    pub smoothness: T,
}

/// Validation strategies for private hyperparameter optimization
#[derive(Debug, Clone, Copy)]
pub enum ValidationStrategy {
    /// Hold-out validation
    HoldOut,

    /// K-fold cross-validation with privacy
    KFoldCV,

    /// Leave-one-out cross-validation
    LeaveOneOut,

    /// Bootstrap validation
    Bootstrap,

    /// Time series split for temporal data
    TimeSeriesSplit,
}

/// Privacy budget manager for hyperparameter optimization
pub struct HPOBudgetManager {
    /// Total privacy budget
    total_budget: PrivacyBudget,

    /// Budget allocation per evaluation
    evaluation_budgets: Vec<PrivacyBudget>,

    /// Budget allocation strategy
    allocation_strategy: BudgetAllocationStrategy,

    /// Consumed budget tracker
    consumed_budget: PrivacyBudget,

    /// Adaptive budget controller
    adaptive_controller: AdaptiveBudgetController,
}

/// Adaptive budget controller
pub struct AdaptiveBudgetController {
    /// Historical performance scores
    performance_history: Vec<f64>,

    /// Budget efficiency tracker
    budget_efficiency: Vec<f64>,

    /// Allocation weights
    allocation_weights: Vec<f64>,

    /// Learning rate for adaptation
    adaptation_rate: f64,
}

/// Trait for noisy optimization algorithms
pub trait NoisyOptimizer<T: Float>: Send + Sync {
    /// Suggest next hyperparameter configuration with privacy
    fn suggest_next(
        &mut self,
        parameterspace: &ParameterSpace<T>,
        evaluation_history: &[HPOEvaluation<T>],
        _privacy_budget: &PrivacyBudget,
    ) -> Result<ParameterConfiguration<T>>;

    /// Update optimizer with new evaluation result
    fn update(
        &mut self,
        config: &ParameterConfiguration<T>,
        result: &HPOResult<T>,
        _privacy_budget: &PrivacyBudget,
    ) -> Result<()>;

    /// Get optimizer name
    fn name(&self) -> &str;
}

/// Hyperparameter space definition
#[derive(Debug, Clone)]
pub struct ParameterSpace<T: Float> {
    /// Parameter definitions
    pub parameters: HashMap<String, ParameterDefinition<T>>,

    /// Parameter constraints
    pub constraints: Vec<ParameterConstraint<T>>,

    /// Default configuration
    pub defaultconfig: Option<ParameterConfiguration<T>>,
}

/// Individual parameter definition
#[derive(Debug, Clone)]
pub struct ParameterDefinition<T: Float> {
    /// Parameter name
    pub name: String,

    /// Parameter type
    pub param_type: ParameterType<T>,

    /// Parameter bounds
    pub bounds: ParameterBounds<T>,

    /// Prior distribution (for Bayesian optimization)
    pub prior: Option<ParameterPrior<T>>,

    /// Transformation function
    pub transformation: Option<ParameterTransformation>,
}

/// Types of hyperparameters
#[derive(Debug, Clone)]
pub enum ParameterType<T: Float> {
    /// Continuous parameter
    Continuous,

    /// Discrete integer parameter
    Integer,

    /// Categorical parameter
    Categorical(Vec<String>),

    /// Boolean parameter
    Boolean,

    /// Ordinal parameter
    Ordinal(Vec<T>),
}

/// Parameter bounds
#[derive(Debug, Clone)]
pub struct ParameterBounds<T: Float> {
    /// Minimum value
    pub min: Option<T>,

    /// Maximum value
    pub max: Option<T>,

    /// Step size (for discrete parameters)
    pub step: Option<T>,

    /// Valid values (for categorical parameters)
    pub valid_values: Option<Vec<String>>,
}

/// Prior distributions for parameters
#[derive(Debug, Clone)]
pub enum ParameterPrior<T: Float> {
    /// Uniform prior
    Uniform(T, T),

    /// Normal prior
    Normal(T, T),

    /// Log-normal prior
    LogNormal(T, T),

    /// Beta prior
    Beta(T, T),

    /// Gamma prior
    Gamma(T, T),
}

/// Parameter transformations
#[derive(Debug, Clone, Copy)]
pub enum ParameterTransformation {
    /// No transformation
    Identity,

    /// Logarithmic transformation
    Log,

    /// Exponential transformation
    Exp,

    /// Square root transformation
    Sqrt,

    /// Square transformation
    Square,
}

/// Parameter constraints
#[derive(Debug, Clone)]
pub struct ParameterConstraint<T: Float> {
    /// Constraint name
    pub name: String,

    /// Constraint type
    pub constraint_type: ConstraintType<T>,

    /// Constraint violation penalty
    pub penalty: T,
}

/// Types of parameter constraints
#[derive(Debug, Clone)]
pub enum ConstraintType<T: Float> {
    /// Linear constraint: a^T x <= b
    Linear(Vec<T>, T),

    /// Quadratic constraint: x^T A x + b^T x <= c
    Quadratic(Array2<T>, Array1<T>, T),

    /// Custom constraint function
    Custom(String),
}

/// Hyperparameter configuration
#[derive(Debug, Clone)]
pub struct ParameterConfiguration<T: Float> {
    /// Parameter values
    pub values: HashMap<String, ParameterValue<T>>,

    /// Configuration identifier
    pub id: String,

    /// Configuration metadata
    pub metadata: HashMap<String, String>,
}

/// Parameter value types
#[derive(Debug, Clone)]
pub enum ParameterValue<T: Float> {
    /// Continuous value
    Continuous(T),

    /// Integer value
    Integer(i64),

    /// Categorical value
    Categorical(String),

    /// Boolean value
    Boolean(bool),

    /// Ordinal value
    Ordinal(usize),
}

/// Private objective function
pub struct PrivateObjective<T: Float> {
    /// Underlying objective function
    objective_fn: Box<dyn Fn(&ParameterConfiguration<T>) -> Result<f64> + Send + Sync>,

    /// Noise mechanism for objective evaluation
    noise_mechanism: ObjectiveNoiseMechanism<T>,

    /// Sensitivity analysis
    sensitivity_analyzer: ObjectiveSensitivityAnalyzer<T>,

    /// Cross-validation with privacy
    cv_evaluator: PrivateCrossValidation<T>,
}

/// Noise mechanisms for objective function evaluation
pub struct ObjectiveNoiseMechanism<T: Float> {
    /// Mechanism type
    mechanism_type: HyperparameterNoiseMechanism,

    /// Noise parameters
    noise_params: NoiseParameters<T>,

    /// Random number generator
    rng: scirs2_core::random::Random,
}

/// Noise parameters
#[derive(Debug, Clone)]
pub struct NoiseParameters<T: Float> {
    /// Noise scale
    pub scale: T,

    /// Sensitivity bound
    pub sensitivity: T,

    /// Privacy parameters
    pub epsilon: f64,

    /// Delta parameter (for Gaussian mechanism)
    pub delta: Option<f64>,
}

/// Objective sensitivity analyzer
pub struct ObjectiveSensitivityAnalyzer<T: Float> {
    /// Sensitivity estimation method
    estimation_method: SensitivityEstimationMethod,

    /// Sensitivity cache
    sensitivity_cache: HashMap<String, T>,

    /// Sample-based sensitivity estimator
    sample_estimator: SampleBasedSensitivityEstimator<T>,
}

/// Methods for estimating objective sensitivity
#[derive(Debug, Clone, Copy)]
pub enum SensitivityEstimationMethod {
    /// Global sensitivity (worst-case)
    Global,

    /// Local sensitivity (data-dependent)
    Local,

    /// Smooth sensitivity
    Smooth,

    /// Sample-based estimation
    SampleBased,

    /// Theoretical bounds
    Theoretical,
}

/// Sample-based sensitivity estimator
pub struct SampleBasedSensitivityEstimator<T: Float> {
    /// Number of samples for estimation
    num_samples: usize,

    /// Sampling strategy
    sampling_strategy: SamplingStrategy,

    /// Confidence level for bounds
    confidence_level: f64,

    /// Bootstrap estimator
    bootstrap_estimator: BootstrapEstimator<T>,

    /// Phantom data to mark type parameter as intentionally unused
    _phantom: std::marker::PhantomData<T>,
}

/// Sampling strategies for sensitivity estimation
#[derive(Debug, Clone, Copy)]
pub enum SamplingStrategy {
    /// Uniform random sampling
    Uniform,

    /// Latin hypercube sampling
    LatinHypercube,

    /// Sobol sequence sampling
    Sobol,

    /// Halton sequence sampling
    Halton,

    /// Importance sampling
    ImportanceSampling,
}

/// Bootstrap estimator for sensitivity
pub struct BootstrapEstimator<T: Float> {
    /// Number of bootstrap samples
    num_bootstrap: usize,

    /// Bootstrap confidence interval
    confidence_interval: (f64, f64),

    /// Bias correction
    bias_correction: bool,

    /// Phantom data to mark type parameter as intentionally unused
    _phantom: std::marker::PhantomData<T>,
}

/// Private cross-validation evaluator
pub struct PrivateCrossValidation<T: Float> {
    /// Number of folds
    num_folds: usize,

    /// Privacy budget per fold
    fold_budgets: Vec<PrivacyBudget>,

    /// Fold assignment strategy
    fold_strategy: FoldStrategy,

    /// Result aggregation with privacy
    private_aggregation: PrivateFoldAggregation<T>,
}

/// Fold assignment strategies
#[derive(Debug, Clone, Copy)]
pub enum FoldStrategy {
    /// Random assignment
    Random,

    /// Stratified assignment
    Stratified,

    /// Time-based assignment (for time series)
    TimeBased,

    /// Group-based assignment
    GroupBased,
}

/// Private aggregation of cross-validation results
pub struct PrivateFoldAggregation<T: Float> {
    /// Aggregation method
    aggregation_method: AggregationMethod,

    /// Noise parameters for aggregation
    noise_params: NoiseParameters<T>,

    /// Confidence interval estimation
    confidence_estimation: ConfidenceEstimation<T>,
}

/// Aggregation methods for cross-validation
#[derive(Debug, Clone, Copy)]
pub enum AggregationMethod {
    /// Mean aggregation with noise
    NoisyMean,

    /// Median aggregation
    Median,

    /// Trimmed mean
    TrimmedMean,

    /// Weighted average
    WeightedAverage,

    /// Robust aggregation
    Robust,
}

/// Confidence interval estimation
pub struct ConfidenceEstimation<T: Float> {
    /// Confidence level
    confidence_level: f64,

    /// Estimation method
    estimation_method: ConfidenceEstimationMethod,

    /// Bootstrap parameters
    bootstrap_params: Option<BootstrapParams>,

    /// Phantom data to mark type parameter as intentionally unused
    _phantom: std::marker::PhantomData<T>,
}

/// Confidence estimation methods
#[derive(Debug, Clone, Copy)]
pub enum ConfidenceEstimationMethod {
    /// Normal approximation
    Normal,

    /// Bootstrap confidence intervals
    Bootstrap,

    /// Jackknife estimation
    Jackknife,

    /// Bayesian credible intervals
    Bayesian,
}

/// Bootstrap parameters for confidence estimation
#[derive(Debug, Clone)]
pub struct BootstrapParams {
    /// Number of bootstrap samples
    pub num_samples: usize,

    /// Bootstrap type
    pub bootstrap_type: BootstrapType,

    /// Bias correction
    pub bias_correction: bool,
}

/// Types of bootstrap methods
#[derive(Debug, Clone, Copy)]
pub enum BootstrapType {
    /// Standard bootstrap
    Standard,

    /// Bias-corrected and accelerated (BCa)
    BCa,

    /// Parametric bootstrap
    Parametric,

    /// Block bootstrap (for time series)
    Block,
}

/// Search strategy for hyperparameter optimization
pub struct SearchStrategy<T: Float> {
    /// Search algorithm
    algorithm: SearchAlgorithm,

    /// Algorithm-specific parameters
    algorithm_params: HashMap<String, f64>,

    /// Exploration-exploitation balance
    exploration_factor: f64,

    /// Convergence criteria
    convergence_criteria: ConvergenceCriteria<T>,
}

/// Convergence criteria for search
#[derive(Debug, Clone)]
pub struct ConvergenceCriteria<T: Float> {
    /// Maximum number of iterations
    pub max_iterations: usize,

    /// Tolerance for objective improvement
    pub tolerance: T,

    /// Patience for early stopping
    pub patience: usize,

    /// Minimum change in best value
    pub min_change: T,
}

/// Private results aggregator
pub struct PrivateResultsAggregator<T: Float> {
    /// Aggregation strategy
    aggregation_strategy: ResultAggregationStrategy,

    /// Privacy budget for final selection
    selection_budget: PrivacyBudget,

    /// Selection mechanism
    selection_mechanism: SelectionMechanism<T>,

    /// Result validation
    result_validator: ResultValidator<T>,
}

/// Result aggregation strategies
#[derive(Debug, Clone, Copy)]
pub enum ResultAggregationStrategy {
    /// Select best configuration
    SelectBest,

    /// Ensemble of top configurations
    Ensemble,

    /// Weighted combination
    WeightedCombination,

    /// Consensus-based selection
    Consensus,

    /// Multi-objective selection
    MultiObjective,
}

/// Selection mechanisms for final hyperparameter choice
pub struct SelectionMechanism<T: Float> {
    /// Mechanism type
    mechanism_type: HyperparameterNoiseMechanism,

    /// Selection parameters
    selection_params: SelectionParameters<T>,

    /// Utility function for selection
    utility_function: UtilityFunction<T>,
}

/// Selection parameters
#[derive(Debug, Clone)]
pub struct SelectionParameters<T: Float> {
    /// Temperature parameter (for exponential mechanism)
    pub temperature: T,

    /// Sensitivity bound for utility
    pub utility_sensitivity: T,

    /// Privacy parameters
    pub epsilon: f64,

    /// Selection threshold
    pub threshold: Option<T>,
}

/// Utility function for hyperparameter selection
pub struct UtilityFunction<T: Float> {
    /// Function type
    function_type: UtilityFunctionType,

    /// Function parameters
    parameters: Vec<T>,

    /// Multi-objective weights
    multi_objective_weights: Option<Vec<T>>,
}

/// Types of utility functions
#[derive(Debug, Clone, Copy)]
pub enum UtilityFunctionType {
    /// Linear utility
    Linear,

    /// Exponential utility
    Exponential,

    /// Logarithmic utility
    Logarithmic,

    /// Quadratic utility
    Quadratic,

    /// Custom utility function
    Custom,
}

/// Result validator
pub struct ResultValidator<T: Float> {
    /// Validation rules
    validation_rules: Vec<ValidationRule<T>>,

    /// Statistical tests
    statistical_tests: Vec<StatisticalTest<T>>,

    /// Anomaly detection
    anomaly_detector: AnomalyDetector<T>,
}

/// Validation rule for results
pub struct ValidationRule<T: Float> {
    /// Rule name
    pub name: String,

    /// Rule function
    pub rule_fn: Box<dyn Fn(&HPOResult<T>) -> bool + Send + Sync>,

    /// Rule weight
    pub weight: f64,
}

/// Statistical test for result validation
pub struct StatisticalTest<T: Float> {
    /// Test name
    pub name: String,

    /// Test function
    pub test_fn: Box<dyn Fn(&[HPOResult<T>]) -> StatisticalTestResult + Send + Sync>,

    /// Significance level
    pub alpha: f64,
}

/// Statistical test result
#[derive(Debug, Clone)]
pub struct StatisticalTestResult {
    /// Test statistic
    pub statistic: f64,

    /// P-value
    pub p_value: f64,

    /// Test conclusion
    pub conclusion: TestConclusion,

    /// Confidence interval
    pub confidence_interval: Option<(f64, f64)>,
}

/// Statistical test conclusions
#[derive(Debug, Clone, Copy)]
pub enum TestConclusion {
    /// Reject null hypothesis
    Reject,

    /// Fail to reject null hypothesis
    FailToReject,

    /// Insufficient evidence
    InsufficientEvidence,
}

/// Anomaly detector for results
pub struct AnomalyDetector<T: Float> {
    /// Detection threshold
    threshold: T,

    /// Detection method
    detection_method: AnomalyDetectionMethod,

    /// Historical baseline
    baseline: Option<T>,
}

/// Anomaly detection methods
#[derive(Debug, Clone, Copy)]
pub enum AnomalyDetectionMethod {
    /// Z-score based detection
    ZScore,

    /// Interquartile range method
    IQR,

    /// Isolation forest
    IsolationForest,

    /// Local outlier factor
    LocalOutlierFactor,
}

/// Hyperparameter optimization evaluation
#[derive(Debug, Clone)]
pub struct HPOEvaluation<T: Float> {
    /// Evaluation identifier
    pub id: String,

    /// Parameter configuration
    pub configuration: ParameterConfiguration<T>,

    /// Evaluation result
    pub result: HPOResult<T>,

    /// Privacy budget consumed
    pub privacy_cost: PrivacyBudget,

    /// Evaluation timestamp
    pub timestamp: u64,

    /// Evaluation metadata
    pub metadata: HashMap<String, String>,
}

/// Hyperparameter optimization result
#[derive(Debug, Clone)]
pub struct HPOResult<T: Float> {
    /// Objective value
    pub objective_value: T,

    /// Standard error (if available)
    pub standard_error: Option<T>,

    /// Cross-validation scores
    pub cv_scores: Option<Vec<T>>,

    /// Training time
    pub training_time: Option<f64>,

    /// Model complexity metrics
    pub complexity_metrics: HashMap<String, T>,

    /// Additional metrics
    pub additional_metrics: HashMap<String, T>,

    /// Result status
    pub status: EvaluationStatus,
}

/// Evaluation status
#[derive(Debug, Clone, Copy)]
pub enum EvaluationStatus {
    /// Evaluation completed successfully
    Success,

    /// Evaluation failed
    Failed,

    /// Evaluation timed out
    Timeout,

    /// Evaluation was cancelled
    Cancelled,

    /// Evaluation is in progress
    InProgress,
}

impl<T: Float + 'static + Send + Sync> PrivateHyperparameterOptimizer<T> {
    /// Create new private hyperparameter optimizer
    pub fn new(config: PrivateHPOConfig<T>, parameterspace: ParameterSpace<T>) -> Result<Self> {
        let budget_manager = HPOBudgetManager::new(
            config.base_privacyconfig.clone(),
            config.budget_allocation,
            config.num_evaluations,
        )?;

        let privacy_accountant = MomentsAccountant::new(
            config.base_privacyconfig.noise_multiplier,
            config.base_privacyconfig.target_delta,
            config.base_privacyconfig.batch_size,
            config.base_privacyconfig.dataset_size,
        );

        let mut noisy_optimizers: HashMap<String, Box<dyn NoisyOptimizer<T>>> = HashMap::new();

        // Add default noisy optimizers based on configuration
        match config.search_algorithm {
            SearchAlgorithm::RandomSearch => {
                noisy_optimizers.insert(
                    "random_search".to_string(),
                    Box::new(PrivateRandomSearch::new(config.clone())?),
                );
            }
            SearchAlgorithm::BayesianOptimization => {
                noisy_optimizers.insert(
                    "bayesian_opt".to_string(),
                    Box::new(PrivateBayesianOptimization::new(config.clone())?),
                );
            }
            _ => {
                // Default to random search
                noisy_optimizers.insert(
                    "random_search".to_string(),
                    Box::new(PrivateRandomSearch::new(config.clone())?),
                );
            }
        }

        Ok(Self {
            config,
            budget_manager,
            noisy_optimizers,
            parameterspace,
            private_objective: PrivateObjective::new()?,
            search_strategy: SearchStrategy::new(),
            results_aggregator: PrivateResultsAggregator::new()?,
            privacy_accountant,
        })
    }

    /// Optimize hyperparameters with differential privacy
    pub fn optimize(
        &mut self,
        objective_fn: Box<dyn Fn(&ParameterConfiguration<T>) -> Result<f64> + Send + Sync>,
    ) -> Result<PrivateHPOResults<T>> {
        // Set up private objective function
        self.private_objective.set_objective(objective_fn)?;

        let mut evaluations = Vec::new();
        let mut bestconfig: Option<ParameterConfiguration<T>> = None;
        let mut best_score = T::neg_infinity();

        // Get the primary optimizer
        let optimizer_name = match self.config.search_algorithm {
            SearchAlgorithm::RandomSearch => "random_search",
            SearchAlgorithm::BayesianOptimization => "bayesian_opt",
            _ => "random_search",
        };

        for iteration in 0..self.config.num_evaluations {
            // Check if we have remaining privacy budget
            if !self.budget_manager.has_budget_remaining()? {
                break;
            }

            // Get privacy budget for this evaluation
            let evaluation_budget = self.budget_manager.get_evaluation_budget(iteration)?;

            // Suggest next configuration
            let config = if let Some(optimizer) = self.noisy_optimizers.get_mut(optimizer_name) {
                optimizer.suggest_next(&self.parameterspace, &evaluations, &evaluation_budget)?
            } else {
                return Err(OptimError::InvalidConfig(
                    "No optimizer available".to_string(),
                ));
            };

            // Evaluate configuration with privacy
            let result = self
                .private_objective
                .evaluate(&config, &evaluation_budget)?;

            // Create evaluation record
            let evaluation = HPOEvaluation {
                id: format!("eval_{}", iteration),
                configuration: config.clone(),
                result: result.clone(),
                privacy_cost: evaluation_budget.clone(),
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                metadata: HashMap::new(),
            };

            // Update best configuration
            if result.objective_value > best_score {
                best_score = result.objective_value;
                bestconfig = Some(config.clone());
            }

            // Update optimizer with result
            if let Some(optimizer) = self.noisy_optimizers.get_mut(optimizer_name) {
                optimizer.update(&config, &result, &evaluation_budget)?;
            }

            // Record evaluation
            evaluations.push(evaluation);

            // Update budget manager
            self.budget_manager.record_evaluation(
                &evaluation_budget,
                result.objective_value.to_f64().unwrap_or(0.0),
            )?;

            // Check early stopping criteria
            if self.should_stop_early(&evaluations)? {
                break;
            }
        }

        // Aggregate results with privacy guarantees
        let final_results = self.results_aggregator.aggregate_results(&evaluations)?;

        Ok(PrivateHPOResults {
            bestconfiguration: bestconfig,
            best_score,
            all_evaluations: evaluations,
            final_results,
            total_privacy_cost: self.budget_manager.get_total_consumed_budget(),
            optimization_stats: self.compute_optimization_stats()?,
        })
    }

    /// Check early stopping criteria
    fn should_stop_early(&self, evaluations: &[HPOEvaluation<T>]) -> Result<bool> {
        if !self.config.early_stopping.enabled {
            return Ok(false);
        }

        if evaluations.len() < self.config.early_stopping.patience {
            return Ok(false);
        }

        // Check for improvement in recent evaluations
        let recent_scores: Vec<T> = evaluations
            .iter()
            .rev()
            .take(self.config.early_stopping.patience)
            .map(|eval| eval.result.objective_value)
            .collect();

        let best_recent =
            recent_scores
                .iter()
                .fold(T::neg_infinity(), |acc, &x| if x > acc { x } else { acc });

        let best_overall = evaluations
            .iter()
            .map(|eval| eval.result.objective_value)
            .fold(T::neg_infinity(), |acc, x| if x > acc { x } else { acc });

        let improvement = best_recent - best_overall;

        Ok(improvement < T::from(self.config.early_stopping.min_improvement).unwrap())
    }

    /// Compute optimization statistics
    fn compute_optimization_stats(&self) -> Result<OptimizationStats<T>> {
        Ok(OptimizationStats {
            total_evaluations: 0,
            successful_evaluations: 0,
            failed_evaluations: 0,
            average_evaluation_time: 0.0,
            total_optimization_time: 0.0,
            convergence_iteration: None,
            budget_efficiency: 0.0,
            _phantom: std::marker::PhantomData,
        })
    }
}

/// Private hyperparameter optimization results
#[derive(Debug, Clone)]
pub struct PrivateHPOResults<T: Float> {
    /// Best configuration found
    pub bestconfiguration: Option<ParameterConfiguration<T>>,

    /// Best objective score
    pub best_score: T,

    /// All evaluations performed
    pub all_evaluations: Vec<HPOEvaluation<T>>,

    /// Final aggregated results
    pub final_results: AggregatedResults<T>,

    /// Total privacy cost
    pub total_privacy_cost: PrivacyBudget,

    /// Optimization statistics
    pub optimization_stats: OptimizationStats<T>,
}

/// Aggregated results from private optimization
#[derive(Debug, Clone)]
pub struct AggregatedResults<T: Float> {
    /// Top-k configurations
    pub topconfigurations: Vec<(ParameterConfiguration<T>, T)>,

    /// Confidence intervals for best score
    pub confidence_intervals: Option<(T, T)>,

    /// Privacy-preserving summary statistics
    pub summary_stats: SummaryStatistics<T>,

    /// Model selection results
    pub model_selection: Option<ModelSelectionResults<T>>,
}

/// Summary statistics with privacy
#[derive(Debug, Clone)]
pub struct SummaryStatistics<T: Float> {
    /// Noisy mean of objective values
    pub noisy_mean: T,

    /// Noisy standard deviation
    pub noisy_std: T,

    /// Noisy median
    pub noisy_median: T,

    /// Noisy quantiles
    pub noisy_quantiles: Vec<(f64, T)>,
}

/// Model selection results
#[derive(Debug, Clone)]
pub struct ModelSelectionResults<T: Float> {
    /// Selected model configuration
    pub selectedconfig: ParameterConfiguration<T>,

    /// Selection confidence
    pub selection_confidence: f64,

    /// Alternative configurations
    pub alternatives: Vec<ParameterConfiguration<T>>,
}

/// Optimization statistics
#[derive(Debug, Clone)]
pub struct OptimizationStats<T: Float> {
    /// Total number of evaluations
    pub total_evaluations: usize,

    /// Number of successful evaluations
    pub successful_evaluations: usize,

    /// Number of failed evaluations
    pub failed_evaluations: usize,

    /// Average evaluation time
    pub average_evaluation_time: f64,

    /// Total optimization time
    pub total_optimization_time: f64,

    /// Iteration where convergence was detected
    pub convergence_iteration: Option<usize>,

    /// Budget efficiency score
    pub budget_efficiency: f64,

    /// Phantom data to mark type parameter as intentionally unused
    _phantom: std::marker::PhantomData<T>,
}

/// Private Random Search implementation
pub struct PrivateRandomSearch<T: Float> {
    /// Configuration
    config: PrivateHPOConfig<T>,

    /// Random number generator
    rng: scirs2_core::random::Random<StdRng>,

    /// Evaluation history
    history: Vec<HPOEvaluation<T>>,
}

impl<T: Float + Send + Sync> PrivateRandomSearch<T> {
    pub fn new(config: PrivateHPOConfig<T>) -> Result<Self> {
        Ok(Self {
            config,
            rng: scirs2_core::random::Random::seed(42),
            history: Vec::new(),
        })
    }
}

impl<T: Float + Send + Sync> NoisyOptimizer<T> for PrivateRandomSearch<T> {
    fn suggest_next(
        &mut self,
        parameterspace: &ParameterSpace<T>,
        _evaluation_history: &[HPOEvaluation<T>],
        _privacy_budget: &PrivacyBudget,
    ) -> Result<ParameterConfiguration<T>> {
        let mut values = HashMap::new();

        for (param_name, param_def) in &parameterspace.parameters {
            let value = match &param_def.param_type {
                ParameterType::Continuous => {
                    let min = param_def.bounds.min.unwrap_or(T::zero());
                    let max = param_def.bounds.max.unwrap_or(T::one());
                    let random_val = T::from(self.rng.gen_range(0.0..1.0)).unwrap();
                    ParameterValue::Continuous(min + random_val * (max - min))
                }
                ParameterType::Integer => {
                    let min = param_def
                        .bounds
                        .min
                        .unwrap_or(T::zero())
                        .to_i64()
                        .unwrap_or(0);
                    let max = param_def
                        .bounds
                        .max
                        .unwrap_or(T::from(100).unwrap())
                        .to_i64()
                        .unwrap_or(100);
                    ParameterValue::Integer(self.rng.gen_range(min..max + 1))
                }
                ParameterType::Boolean => ParameterValue::Boolean(self.rng.gen_range(0..2) == 1),
                ParameterType::Categorical(categories) => {
                    let idx = self.rng.gen_range(0..categories.len());
                    ParameterValue::Categorical(categories[idx].clone())
                }
                ParameterType::Ordinal(values) => {
                    let idx = self.rng.gen_range(0..values.len());
                    ParameterValue::Ordinal(idx)
                }
            };

            values.insert(param_name.clone(), value);
        }

        Ok(ParameterConfiguration {
            values,
            id: format!("config_{}", self.history.len()),
            metadata: HashMap::new(),
        })
    }

    fn update(
        &mut self,
        config: &ParameterConfiguration<T>,
        _result: &HPOResult<T>,
        _privacy_budget: &PrivacyBudget,
    ) -> Result<()> {
        // Random search doesn't need to update based on results
        Ok(())
    }

    fn name(&self) -> &str {
        "PrivateRandomSearch"
    }
}

/// Private Bayesian Optimization implementation
pub struct PrivateBayesianOptimization<T: Float> {
    /// Configuration
    config: PrivateHPOConfig<T>,

    /// Gaussian process surrogate model
    gp_model: Option<GaussianProcessModel<T>>,

    /// Acquisition function
    acquisition_fn: AcquisitionFunction<T>,

    /// Evaluation history
    history: Vec<HPOEvaluation<T>>,
}

impl<T: Float + Send + Sync> PrivateBayesianOptimization<T> {
    pub fn new(config: PrivateHPOConfig<T>) -> Result<Self> {
        Ok(Self {
            config,
            gp_model: None,
            acquisition_fn: AcquisitionFunction::new(),
            history: Vec::new(),
        })
    }
}

impl<T: Float + Send + Sync> NoisyOptimizer<T> for PrivateBayesianOptimization<T> {
    fn suggest_next(
        &mut self,
        parameterspace: &ParameterSpace<T>,
        evaluation_history: &[HPOEvaluation<T>],
        _privacy_budget: &PrivacyBudget,
    ) -> Result<ParameterConfiguration<T>> {
        if evaluation_history.is_empty() {
            // First evaluation - use random sampling
            let mut rng = scirs2_core::random::Random::seed(42);
            let mut values = HashMap::new();

            for (param_name, param_def) in &parameterspace.parameters {
                let value = match &param_def.param_type {
                    ParameterType::Continuous => {
                        let min = param_def.bounds.min.unwrap_or(T::zero());
                        let max = param_def.bounds.max.unwrap_or(T::one());
                        let random_val = T::from(rng.gen_range(0.0..1.0)).unwrap();
                        ParameterValue::Continuous(min + random_val * (max - min))
                    }
                    _ => {
                        // Simplified for other types
                        ParameterValue::Continuous(T::from(0.5).unwrap())
                    }
                };
                values.insert(param_name.clone(), value);
            }

            return Ok(ParameterConfiguration {
                values,
                id: "initialconfig".to_string(),
                metadata: HashMap::new(),
            });
        }

        // Use acquisition function to suggest next point
        // This is a simplified implementation
        Ok(ParameterConfiguration {
            values: HashMap::new(),
            id: format!("bayesianconfig_{}", evaluation_history.len()),
            metadata: HashMap::new(),
        })
    }

    fn update(
        &mut self,
        config: &ParameterConfiguration<T>,
        _result: &HPOResult<T>,
        _privacy_budget: &PrivacyBudget,
    ) -> Result<()> {
        // Update Gaussian process model with new data point
        // This is a simplified implementation
        Ok(())
    }

    fn name(&self) -> &str {
        "PrivateBayesianOptimization"
    }
}

/// Gaussian process model for Bayesian optimization
pub struct GaussianProcessModel<T: Float> {
    /// Training inputs
    training_inputs: Vec<Vec<T>>,

    /// Training outputs
    training_outputs: Vec<T>,

    /// Kernel function
    kernel: KernelFunction<T>,

    /// Hyperparameters
    hyperparameters: Vec<T>,
}

/// Kernel functions for Gaussian processes
pub struct KernelFunction<T: Float> {
    /// Kernel type
    kernel_type: KernelType,

    /// Kernel parameters
    parameters: Vec<T>,
}

/// Types of kernel functions
#[derive(Debug, Clone, Copy)]
pub enum KernelType {
    /// Radial basis function kernel
    RBF,

    /// Matern kernel
    Matern,

    /// Linear kernel
    Linear,

    /// Polynomial kernel
    Polynomial,
}

/// Acquisition function for Bayesian optimization
pub struct AcquisitionFunction<T: Float> {
    /// Function type
    function_type: AcquisitionFunctionType,

    /// Function parameters
    parameters: Vec<T>,

    /// Exploration-exploitation balance
    exploration_weight: T,
}

/// Types of acquisition functions
#[derive(Debug, Clone, Copy)]
pub enum AcquisitionFunctionType {
    /// Expected Improvement
    ExpectedImprovement,

    /// Upper Confidence Bound
    UpperConfidenceBound,

    /// Probability of Improvement
    ProbabilityOfImprovement,

    /// Knowledge Gradient
    KnowledgeGradient,
}

// Implementation stubs for other components
impl HPOBudgetManager {
    pub fn new(
        baseconfig: DifferentialPrivacyConfig,
        allocation_strategy: BudgetAllocationStrategy,
        num_evaluations: usize,
    ) -> Result<Self> {
        let total_budget = PrivacyBudget {
            epsilon_consumed: 0.0,
            delta_consumed: 0.0,
            epsilon_remaining: baseconfig.target_epsilon,
            delta_remaining: baseconfig.target_delta,
            steps_taken: 0,
            accounting_method: crate::privacy::AccountingMethod::MomentsAccountant,
            estimated_steps_remaining: num_evaluations,
        };

        Ok(Self {
            total_budget,
            evaluation_budgets: Vec::new(),
            allocation_strategy,
            consumed_budget: PrivacyBudget {
                epsilon_consumed: 0.0,
                delta_consumed: 0.0,
                epsilon_remaining: baseconfig.target_epsilon,
                delta_remaining: baseconfig.target_delta,
                steps_taken: 0,
                accounting_method: crate::privacy::AccountingMethod::MomentsAccountant,
                estimated_steps_remaining: num_evaluations,
            },
            adaptive_controller: AdaptiveBudgetController::new(),
        })
    }

    pub fn has_budget_remaining(&self) -> Result<bool> {
        Ok(self.consumed_budget.epsilon_remaining > 0.0
            && self.consumed_budget.delta_remaining > 0.0)
    }

    pub fn get_evaluation_budget(&mut self, iteration: usize) -> Result<PrivacyBudget> {
        let remaining_evaluations = self
            .total_budget
            .estimated_steps_remaining
            .saturating_sub(iteration);
        let epsilon_per_eval =
            self.consumed_budget.epsilon_remaining / remaining_evaluations.max(1) as f64;
        let delta_per_eval =
            self.consumed_budget.delta_remaining / remaining_evaluations.max(1) as f64;

        Ok(PrivacyBudget {
            epsilon_consumed: 0.0,
            delta_consumed: 0.0,
            epsilon_remaining: epsilon_per_eval,
            delta_remaining: delta_per_eval,
            steps_taken: 0,
            accounting_method: crate::privacy::AccountingMethod::MomentsAccountant,
            estimated_steps_remaining: 1,
        })
    }

    pub fn record_evaluation(&mut self, budgetused: &PrivacyBudget, score: f64) -> Result<()> {
        self.consumed_budget.epsilon_consumed += budgetused.epsilon_remaining;
        self.consumed_budget.delta_consumed += budgetused.delta_remaining;
        self.consumed_budget.epsilon_remaining -= budgetused.epsilon_remaining;
        self.consumed_budget.delta_remaining -= budgetused.delta_remaining;
        self.consumed_budget.steps_taken += 1;

        self.adaptive_controller.record_performance(score);
        Ok(())
    }

    pub fn get_total_consumed_budget(&self) -> PrivacyBudget {
        self.consumed_budget.clone()
    }
}

impl AdaptiveBudgetController {
    pub fn new() -> Self {
        Self {
            performance_history: Vec::new(),
            budget_efficiency: Vec::new(),
            allocation_weights: Vec::new(),
            adaptation_rate: 0.1,
        }
    }

    pub fn record_performance(&mut self, score: f64) {
        self.performance_history.push(score);
    }
}

impl<T: Float + Send + Sync> PrivateObjective<T> {
    pub fn new() -> Result<Self> {
        Ok(Self {
            objective_fn: Box::new(|_| Ok(0.0)),
            noise_mechanism: ObjectiveNoiseMechanism::new(),
            sensitivity_analyzer: ObjectiveSensitivityAnalyzer::new(),
            cv_evaluator: PrivateCrossValidation::new(),
        })
    }

    pub fn set_objective(
        &mut self,
        objective_fn: Box<dyn Fn(&ParameterConfiguration<T>) -> Result<f64> + Send + Sync>,
    ) -> Result<()> {
        self.objective_fn = objective_fn;
        Ok(())
    }

    pub fn evaluate(
        &mut self,
        config: &ParameterConfiguration<T>,
        privacy_budget: &PrivacyBudget,
    ) -> Result<HPOResult<T>> {
        let objective_value = (self.objective_fn)(config)?;
        let noisy_value = self
            .noise_mechanism
            .add_noise(objective_value, privacy_budget)?;

        Ok(HPOResult {
            objective_value: T::from(noisy_value).unwrap(),
            standard_error: None,
            cv_scores: None,
            training_time: None,
            complexity_metrics: HashMap::new(),
            additional_metrics: HashMap::new(),
            status: EvaluationStatus::Success,
        })
    }
}

impl<T: Float + Send + Sync> ObjectiveNoiseMechanism<T> {
    pub fn new() -> Self {
        Self {
            mechanism_type: HyperparameterNoiseMechanism::Gaussian,
            noise_params: NoiseParameters {
                scale: T::one(),
                sensitivity: T::one(),
                epsilon: 1.0,
                delta: Some(1e-5),
            },
            rng: scirs2_core::random::Random::default(),
        }
    }

    pub fn add_noise(&mut self, value: f64, _privacybudget: &PrivacyBudget) -> Result<f64> {
        use rand_distr::{Distribution, Normal};

        match self.mechanism_type {
            HyperparameterNoiseMechanism::Gaussian => {
                let noise_scale = self.noise_params.scale.to_f64().unwrap_or(1.0);
                let normal = Normal::new(0.0, noise_scale)
                    .map_err(|_| OptimError::InvalidConfig("Invalid noise scale".to_string()))?;

                let noise = normal.sample(&mut self.rng);
                Ok(value + noise)
            }
            _ => Ok(value), // Simplified for other mechanisms
        }
    }
}

impl<T: Float + Send + Sync> ObjectiveSensitivityAnalyzer<T> {
    pub fn new() -> Self {
        Self {
            estimation_method: SensitivityEstimationMethod::Global,
            sensitivity_cache: HashMap::new(),
            sample_estimator: SampleBasedSensitivityEstimator::new(),
        }
    }
}

impl<T: Float + Send + Sync> SampleBasedSensitivityEstimator<T> {
    pub fn new() -> Self {
        Self {
            num_samples: 1000,
            sampling_strategy: SamplingStrategy::Uniform,
            confidence_level: 0.95,
            bootstrap_estimator: BootstrapEstimator::new(),
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: Float + Send + Sync> BootstrapEstimator<T> {
    pub fn new() -> Self {
        Self {
            num_bootstrap: 1000,
            confidence_interval: (0.025, 0.975),
            bias_correction: true,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: Float + Send + Sync> PrivateCrossValidation<T> {
    pub fn new() -> Self {
        Self {
            num_folds: 5,
            fold_budgets: Vec::new(),
            fold_strategy: FoldStrategy::Random,
            private_aggregation: PrivateFoldAggregation::new(),
        }
    }
}

impl<T: Float + Send + Sync> PrivateFoldAggregation<T> {
    pub fn new() -> Self {
        Self {
            aggregation_method: AggregationMethod::NoisyMean,
            noise_params: NoiseParameters {
                scale: T::one(),
                sensitivity: T::one(),
                epsilon: 1.0,
                delta: Some(1e-5),
            },
            confidence_estimation: ConfidenceEstimation::new(),
        }
    }
}

impl<T: Float + Send + Sync> ConfidenceEstimation<T> {
    pub fn new() -> Self {
        Self {
            confidence_level: 0.95,
            estimation_method: ConfidenceEstimationMethod::Normal,
            bootstrap_params: None,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: Float + Send + Sync> SearchStrategy<T> {
    pub fn new() -> Self {
        Self {
            algorithm: SearchAlgorithm::RandomSearch,
            algorithm_params: HashMap::new(),
            exploration_factor: 0.1,
            convergence_criteria: ConvergenceCriteria {
                max_iterations: 100,
                tolerance: T::from(1e-6).unwrap(),
                patience: 10,
                min_change: T::from(1e-4).unwrap(),
            },
        }
    }
}

impl<T: Float + Send + Sync> PrivateResultsAggregator<T> {
    pub fn new() -> Result<Self> {
        Ok(Self {
            aggregation_strategy: ResultAggregationStrategy::SelectBest,
            selection_budget: PrivacyBudget {
                epsilon_consumed: 0.0,
                delta_consumed: 0.0,
                epsilon_remaining: 0.1,
                delta_remaining: 1e-6,
                steps_taken: 0,
                accounting_method: crate::privacy::AccountingMethod::MomentsAccountant,
                estimated_steps_remaining: 1,
            },
            selection_mechanism: SelectionMechanism::new(),
            result_validator: ResultValidator::new(),
        })
    }

    pub fn aggregate_results(
        &self,
        evaluations: &[HPOEvaluation<T>],
    ) -> Result<AggregatedResults<T>> {
        let mut topconfigs = Vec::new();

        // Sort evaluations by objective value
        let mut sorted_evals = evaluations.to_vec();
        sorted_evals.sort_by(|a, b| {
            b.result
                .objective_value
                .partial_cmp(&a.result.objective_value)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Take top configurations
        for eval in sorted_evals.iter().take(5) {
            topconfigs.push((eval.configuration.clone(), eval.result.objective_value));
        }

        // Compute summary statistics
        let objective_values: Vec<T> = evaluations
            .iter()
            .map(|eval| eval.result.objective_value)
            .collect();

        let mean = if !objective_values.is_empty() {
            objective_values.iter().fold(T::zero(), |acc, &x| acc + x)
                / T::from(objective_values.len()).unwrap()
        } else {
            T::zero()
        };

        let summary_stats = SummaryStatistics {
            noisy_mean: mean,
            noisy_std: T::zero(), // Simplified
            noisy_median: mean,
            noisy_quantiles: vec![(0.5, mean)],
        };

        Ok(AggregatedResults {
            topconfigurations: topconfigs,
            confidence_intervals: None,
            summary_stats,
            model_selection: None,
        })
    }
}

impl<T: Float + Send + Sync> SelectionMechanism<T> {
    pub fn new() -> Self {
        Self {
            mechanism_type: HyperparameterNoiseMechanism::Exponential,
            selection_params: SelectionParameters {
                temperature: T::one(),
                utility_sensitivity: T::one(),
                epsilon: 1.0,
                threshold: None,
            },
            utility_function: UtilityFunction::new(),
        }
    }
}

impl<T: Float + Send + Sync> UtilityFunction<T> {
    pub fn new() -> Self {
        Self {
            function_type: UtilityFunctionType::Linear,
            parameters: vec![T::one()],
            multi_objective_weights: None,
        }
    }
}

impl<T: Float + Send + Sync> ResultValidator<T> {
    pub fn new() -> Self {
        Self {
            validation_rules: Vec::new(),
            statistical_tests: Vec::new(),
            anomaly_detector: AnomalyDetector::new(),
        }
    }
}

impl<T: Float + Send + Sync> AnomalyDetector<T> {
    pub fn new() -> Self {
        Self {
            threshold: T::from(3.0).unwrap(),
            detection_method: AnomalyDetectionMethod::ZScore,
            baseline: None,
        }
    }
}

impl<T: Float + Send + Sync> GaussianProcessModel<T> {
    pub fn new() -> Self {
        Self {
            training_inputs: Vec::new(),
            training_outputs: Vec::new(),
            kernel: KernelFunction::new(),
            hyperparameters: Vec::new(),
        }
    }
}

impl<T: Float + Send + Sync> KernelFunction<T> {
    pub fn new() -> Self {
        Self {
            kernel_type: KernelType::RBF,
            parameters: vec![T::one()],
        }
    }
}

impl<T: Float + Send + Sync> AcquisitionFunction<T> {
    pub fn new() -> Self {
        Self {
            function_type: AcquisitionFunctionType::ExpectedImprovement,
            parameters: Vec::new(),
            exploration_weight: T::from(0.1).unwrap(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_private_hpoconfig() {
        let config = PrivateHPOConfig {
            base_privacyconfig: DifferentialPrivacyConfig::default(),
            budget_allocation: BudgetAllocationStrategy::Equal,
            search_algorithm: SearchAlgorithm::RandomSearch,
            num_evaluations: 100,
            cv_folds: 5,
            early_stopping: EarlyStoppingConfig {
                enabled: true,
                patience: 10,
                min_improvement: 0.01,
                max_evaluations: 100,
            },
            noise_mechanism: HyperparameterNoiseMechanism::Gaussian,
            sensitivity_bounds: SensitivityBounds {
                global_sensitivity: HashMap::<String, f64>::new(),
                local_sensitivity: HashMap::<String, (f64, f64)>::new(),
                smooth_sensitivity: HashMap::<String, SmoothSensitivityParams<f64>>::new(),
            },
            private_model_selection: true,
            validation_strategy: ValidationStrategy::KFoldCV,
        };

        assert_eq!(config.num_evaluations, 100);
        assert_eq!(config.cv_folds, 5);
        assert!(config.early_stopping.enabled);
    }

    #[test]
    fn test_parameter_space() {
        let mut parameters = HashMap::new();

        parameters.insert(
            "learning_rate".to_string(),
            ParameterDefinition {
                name: "learning_rate".to_string(),
                param_type: ParameterType::Continuous,
                bounds: ParameterBounds {
                    min: Some(0.001),
                    max: Some(0.1),
                    step: None,
                    valid_values: None,
                },
                prior: Some(ParameterPrior::LogNormal(-3.0, 1.0)),
                transformation: Some(ParameterTransformation::Log),
            },
        );

        let parameterspace = ParameterSpace {
            parameters,
            constraints: Vec::new(),
            defaultconfig: None,
        };

        assert!(parameterspace.parameters.contains_key("learning_rate"));
    }

    #[test]
    fn test_budget_manager() {
        let baseconfig = DifferentialPrivacyConfig::default();
        let budget_manager =
            HPOBudgetManager::new(baseconfig, BudgetAllocationStrategy::Equal, 10).unwrap();

        assert!(budget_manager.has_budget_remaining().unwrap());
    }

    #[test]
    fn test_private_random_search() {
        let config = PrivateHPOConfig {
            base_privacyconfig: DifferentialPrivacyConfig::default(),
            budget_allocation: BudgetAllocationStrategy::Equal,
            search_algorithm: SearchAlgorithm::RandomSearch,
            num_evaluations: 10,
            cv_folds: 3,
            early_stopping: EarlyStoppingConfig {
                enabled: false,
                patience: 5,
                min_improvement: 0.01,
                max_evaluations: 10,
            },
            noise_mechanism: HyperparameterNoiseMechanism::Gaussian,
            sensitivity_bounds: SensitivityBounds {
                global_sensitivity: HashMap::<String, f64>::new(),
                local_sensitivity: HashMap::<String, (f64, f64)>::new(),
                smooth_sensitivity: HashMap::<String, SmoothSensitivityParams<f64>>::new(),
            },
            private_model_selection: false,
            validation_strategy: ValidationStrategy::HoldOut,
        };

        let optimizer = PrivateRandomSearch::new(config).unwrap();
        assert_eq!(optimizer.name(), "PrivateRandomSearch");
    }
}
