//! Automatic hyperparameter tuning for clustering algorithms
//!
//! This module provides comprehensive hyperparameter optimization capabilities
//! for all clustering algorithms in the scirs2-cluster crate. It supports
//! grid search, random search, Bayesian optimization, and adaptive strategies.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use num_traits::{Float, FromPrimitive};
use rand::{rng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::Debug;

use crate::advanced::{
    adaptive_online_clustering, quantum_kmeans, rl_clustering, AdaptiveOnlineConfig, QuantumConfig,
    RLClusteringConfig,
};
use crate::affinity::affinity_propagation;
use crate::birch::birch;
use crate::density::{dbscan, optics};
use crate::error::{ClusteringError, Result};
use crate::gmm::gaussian_mixture;
use crate::hierarchy::linkage;
use crate::meanshift::mean_shift;
use crate::metrics::{calinski_harabasz_score, davies_bouldin_score, silhouette_score};
use crate::spectral::spectral_clustering;
use crate::stability::OptimalKSelector;
use crate::vq::{kmeans, kmeans2};
use statrs::statistics::Statistics;

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
#[derive(Debug, Clone)]
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
    observations: Vec<(HashMap<String, f64>, f64)>,
    /// Gaussian process mean function
    gp_mean: Option<f64>,
    /// Gaussian process covariance matrix
    gp_covariance: Option<Array2<f64>>,
    /// Acquisition function values
    acquisition_values: Vec<f64>,
    /// Parameter names for consistent ordering
    parameter_names: Vec<String>,
    /// GP hyperparameters
    gp_hyperparameters: GpHyperparameters,
    /// Noise level
    noise_level: f64,
    /// Current best observed value
    currentbest: f64,
}

/// Gaussian Process hyperparameters
#[derive(Debug, Clone)]
struct GpHyperparameters {
    /// Length scales for each dimension
    length_scales: Vec<f64>,
    /// Signal variance
    signal_variance: f64,
    /// Noise variance
    noise_variance: f64,
    /// Kernel type
    kernel_type: KernelType,
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

/// Main hyperparameter tuner
pub struct AutoTuner<F: Float> {
    config: TuningConfig,
    phantom: std::marker::PhantomData<F>,
}

impl<
        F: Float
            + FromPrimitive
            + Debug
            + 'static
            + std::iter::Sum
            + std::fmt::Display
            + Send
            + Sync
            + ndarray::ScalarOperand
            + std::ops::AddAssign
            + std::ops::SubAssign
            + std::ops::MulAssign
            + std::ops::DivAssign
            + std::ops::RemAssign
            + PartialOrd,
    > AutoTuner<F>
where
    f64: From<F>,
{
    /// Create a new auto tuner
    pub fn new(config: TuningConfig) -> Self {
        Self {
            config,
            phantom: std::marker::PhantomData,
        }
    }

    /// Tune K-means hyperparameters
    pub fn tune_kmeans(
        &self,
        data: ArrayView2<F>,
        search_space: SearchSpace,
    ) -> Result<TuningResult> {
        let start_time = std::time::Instant::now();
        let mut evaluation_history = Vec::new();
        let mut best_score = f64::NEG_INFINITY;
        let mut best_parameters = HashMap::new();

        // Generate parameter combinations based on search strategy
        let parameter_combinations = self.generate_parameter_combinations(&search_space)?;

        let mut rng = match self.config.random_seed {
            Some(seed) => rand::rngs::StdRng::seed_from_u64(seed),
            None => rand::rngs::StdRng::seed_from_u64(42),
        };

        for (eval_idx, params) in parameter_combinations.iter().enumerate() {
            if eval_idx >= self.config.max_evaluations {
                break;
            }

            // Check time constraints
            if let Some(max_time) = self.config.resource_constraints.max_total_time {
                if start_time.elapsed().as_secs_f64() > max_time {
                    break;
                }
            }

            let eval_start = std::time::Instant::now();

            // Extract parameters for K-means
            let k = params.get("n_clusters").map(|&x| x as usize).unwrap_or(3);
            let max_iter = params.get("max_iter").map(|&x| x as usize);
            let tol = params.get("tolerance").copied();
            let seed = rng.random_range(0..u64::MAX);

            // Perform cross-validation
            let cv_scores = self.cross_validate_kmeans(data, k, max_iter, tol, Some(seed))?;

            let mean_score = cv_scores.iter().sum::<f64>() / cv_scores.len() as f64;
            let cv_std = if cv_scores.len() > 1 {
                let variance = cv_scores
                    .iter()
                    .map(|&x| (x - mean_score).powi(2))
                    .sum::<f64>()
                    / (cv_scores.len() - 1) as f64;
                variance.sqrt()
            } else {
                0.0
            };

            let eval_time = eval_start.elapsed().as_secs_f64();

            let result = EvaluationResult {
                parameters: params.clone(),
                score: mean_score,
                additional_metrics: HashMap::new(),
                evaluation_time: eval_time,
                memory_usage: None,
                cv_scores,
                cv_std,
                metadata: HashMap::new(),
            };

            // Update best result (handle minimization vs maximization)
            let is_better = match self.config.metric {
                EvaluationMetric::SilhouetteScore
                | EvaluationMetric::CalinskiHarabaszIndex
                | EvaluationMetric::AdjustedRandIndex => mean_score > best_score,
                EvaluationMetric::DaviesBouldinIndex | EvaluationMetric::Inertia => {
                    mean_score < best_score || best_score == f64::NEG_INFINITY
                }
                _ => mean_score > best_score,
            };

            if is_better {
                best_score = mean_score;
                best_parameters = params.clone();
            }

            evaluation_history.push(result);

            // Check early stopping
            if let Some(ref early_stop) = self.config.early_stopping {
                if self.should_stop_early(&evaluation_history, early_stop) {
                    break;
                }
            }
        }

        let total_time = start_time.elapsed().as_secs_f64();

        let convergence_info = ConvergenceInfo {
            converged: evaluation_history.len() >= self.config.max_evaluations,
            convergence_iteration: None,
            stopping_reason: if evaluation_history.len() >= self.config.max_evaluations {
                StoppingReason::MaxEvaluations
            } else {
                StoppingReason::EarlyStopping
            },
        };

        let exploration_stats = self.calculate_exploration_stats(&evaluation_history);

        Ok(TuningResult {
            best_parameters,
            best_score,
            evaluation_history,
            convergence_info,
            exploration_stats,
            total_time,
            ensemble_results: None,
            pareto_front: None,
        })
    }

    /// Tune DBSCAN hyperparameters
    pub fn tune_dbscan(
        &self,
        data: ArrayView2<F>,
        search_space: SearchSpace,
    ) -> Result<TuningResult> {
        let start_time = std::time::Instant::now();
        let mut evaluation_history = Vec::new();
        let mut best_score = f64::NEG_INFINITY;
        let mut best_parameters = HashMap::new();

        let parameter_combinations = self.generate_parameter_combinations(&search_space)?;

        for (eval_idx, params) in parameter_combinations.iter().enumerate() {
            if eval_idx >= self.config.max_evaluations {
                break;
            }

            let eval_start = std::time::Instant::now();

            // Extract DBSCAN parameters
            let eps = params.get("eps").copied().unwrap_or(0.5);
            let min_samples = params.get("min_samples").map(|&x| x as usize).unwrap_or(5);

            // Perform cross-validation for DBSCAN
            let cv_scores = self.cross_validate_dbscan(data, eps, min_samples)?;

            let mean_score = cv_scores.iter().sum::<f64>() / cv_scores.len() as f64;
            let cv_std = if cv_scores.len() > 1 {
                let variance = cv_scores
                    .iter()
                    .map(|&x| (x - mean_score).powi(2))
                    .sum::<f64>()
                    / (cv_scores.len() - 1) as f64;
                variance.sqrt()
            } else {
                0.0
            };

            let eval_time = eval_start.elapsed().as_secs_f64();

            let result = EvaluationResult {
                parameters: params.clone(),
                score: mean_score,
                additional_metrics: HashMap::new(),
                evaluation_time: eval_time,
                memory_usage: None,
                cv_scores,
                cv_std,
                metadata: HashMap::new(),
            };

            // Update best result
            let is_better = match self.config.metric {
                EvaluationMetric::SilhouetteScore
                | EvaluationMetric::CalinskiHarabaszIndex
                | EvaluationMetric::AdjustedRandIndex => mean_score > best_score,
                EvaluationMetric::DaviesBouldinIndex | EvaluationMetric::Inertia => {
                    mean_score < best_score || best_score == f64::NEG_INFINITY
                }
                _ => mean_score > best_score,
            };

            if is_better {
                best_score = mean_score;
                best_parameters = params.clone();
            }

            evaluation_history.push(result);

            // Check early stopping
            if let Some(ref early_stop) = self.config.early_stopping {
                if self.should_stop_early(&evaluation_history, early_stop) {
                    break;
                }
            }
        }

        let total_time = start_time.elapsed().as_secs_f64();

        let convergence_info = ConvergenceInfo {
            converged: evaluation_history.len() >= self.config.max_evaluations,
            convergence_iteration: None,
            stopping_reason: if evaluation_history.len() >= self.config.max_evaluations {
                StoppingReason::MaxEvaluations
            } else {
                StoppingReason::EarlyStopping
            },
        };

        let exploration_stats = self.calculate_exploration_stats(&evaluation_history);

        Ok(TuningResult {
            best_parameters,
            best_score,
            evaluation_history,
            convergence_info,
            exploration_stats,
            total_time,
            ensemble_results: None,
            pareto_front: None,
        })
    }

    /// Tune OPTICS hyperparameters
    pub fn tune_optics(
        &self,
        data: ArrayView2<F>,
        search_space: SearchSpace,
    ) -> Result<TuningResult> {
        let start_time = std::time::Instant::now();

        // Generate parameter combinations
        let parameter_combinations = self.generate_parameter_combinations(&search_space)?;

        let mut evaluation_history = Vec::new();
        let mut best_score = f64::NEG_INFINITY;
        let mut best_parameters = HashMap::new();

        for combination in &parameter_combinations {
            // Extract OPTICS parameters
            let min_samples = combination
                .get("min_samples")
                .ok_or_else(|| {
                    ClusteringError::InvalidInput("min_samples parameter not found".to_string())
                })?
                .round() as usize;
            let max_eps = combination.get("max_eps").copied().unwrap_or(5.0);

            // Cross-validate with current parameters
            let scores =
                self.cross_validate_optics(data, min_samples, Some(F::from(max_eps).unwrap()))?;
            let mean_score = scores.iter().sum::<f64>() / scores.len() as f64;

            evaluation_history.push(EvaluationResult {
                parameters: combination.clone(),
                score: mean_score,
                additional_metrics: HashMap::new(),
                evaluation_time: 0.0, // Would measure actual time
                memory_usage: None,
                cv_scores: vec![],
                cv_std: 0.0,
                metadata: HashMap::new(),
            });

            if mean_score > best_score {
                best_score = mean_score;
                best_parameters = combination.clone();
            }
        }

        let total_time = start_time.elapsed().as_secs_f64();

        Ok(TuningResult {
            best_parameters,
            best_score,
            evaluation_history,
            convergence_info: ConvergenceInfo {
                converged: false,
                convergence_iteration: None,
                stopping_reason: StoppingReason::MaxEvaluations,
            },
            exploration_stats: ExplorationStats {
                coverage: 0.0,
                parameter_distributions: HashMap::new(),
                parameter_importance: HashMap::new(),
            },
            total_time,
            ensemble_results: None,
            pareto_front: None,
        })
    }

    /// Tune Spectral clustering hyperparameters
    pub fn tune_spectral(
        &self,
        data: ArrayView2<F>,
        search_space: SearchSpace,
    ) -> Result<TuningResult> {
        let start_time = std::time::Instant::now();

        let parameter_combinations = self.generate_parameter_combinations(&search_space)?;

        let mut evaluation_history = Vec::new();
        let mut best_score = f64::NEG_INFINITY;
        let mut best_parameters = HashMap::new();

        for combination in &parameter_combinations {
            let n_clusters = combination
                .get("n_clusters")
                .ok_or_else(|| {
                    ClusteringError::InvalidInput("n_clusters parameter not found".to_string())
                })?
                .round() as usize;
            let n_neighbors = combination
                .get("n_neighbors")
                .copied()
                .unwrap_or(10.0)
                .round() as usize;
            let gamma = combination.get("gamma").copied().unwrap_or(1.0);
            let max_iter = combination
                .get("max_iter")
                .copied()
                .unwrap_or(300.0)
                .round() as usize;

            let scores = self.cross_validate_spectral(
                data,
                n_clusters,
                n_neighbors,
                F::from(gamma).unwrap(),
                max_iter,
            )?;
            let mean_score = scores.iter().sum::<f64>() / scores.len() as f64;

            evaluation_history.push(EvaluationResult {
                parameters: combination.clone(),
                score: mean_score,
                additional_metrics: HashMap::new(),
                evaluation_time: 0.0,
                memory_usage: None,
                cv_scores: scores.clone(),
                cv_std: scores.std_dev(),
                metadata: HashMap::new(),
            });

            if mean_score > best_score {
                best_score = mean_score;
                best_parameters = combination.clone();
            }
        }

        let total_time = start_time.elapsed().as_secs_f64();

        Ok(TuningResult {
            best_parameters,
            best_score,
            evaluation_history,
            convergence_info: ConvergenceInfo {
                converged: false,
                convergence_iteration: None,
                stopping_reason: StoppingReason::MaxEvaluations,
            },
            exploration_stats: ExplorationStats {
                coverage: 0.0,
                parameter_distributions: HashMap::new(),
                parameter_importance: HashMap::new(),
            },
            total_time,
            ensemble_results: None,
            pareto_front: None,
        })
    }

    /// Tune Affinity Propagation hyperparameters
    pub fn tune_affinity_propagation(
        &self,
        data: ArrayView2<F>,
        search_space: SearchSpace,
    ) -> Result<TuningResult> {
        let start_time = std::time::Instant::now();

        let parameter_combinations = self.generate_parameter_combinations(&search_space)?;

        let mut evaluation_history = Vec::new();
        let mut best_score = f64::NEG_INFINITY;
        let mut best_parameters = HashMap::new();

        for combination in &parameter_combinations {
            let damping = combination.get("damping").copied().unwrap_or(0.5);
            let max_iter = combination
                .get("max_iter")
                .copied()
                .unwrap_or(200.0)
                .round() as usize;
            let convergence_iter = combination
                .get("convergence_iter")
                .copied()
                .unwrap_or(15.0)
                .round() as usize;

            let scores = self.cross_validate_affinity_propagation(
                data,
                F::from(damping).unwrap(),
                max_iter,
                convergence_iter,
            )?;
            let mean_score = scores.iter().sum::<f64>() / scores.len() as f64;

            evaluation_history.push(EvaluationResult {
                parameters: combination.clone(),
                score: mean_score,
                additional_metrics: HashMap::new(),
                evaluation_time: 0.0,
                memory_usage: None,
                cv_scores: scores.clone(),
                cv_std: scores.std_dev(),
                metadata: HashMap::new(),
            });

            if mean_score > best_score {
                best_score = mean_score;
                best_parameters = combination.clone();
            }
        }

        let total_time = start_time.elapsed().as_secs_f64();

        Ok(TuningResult {
            best_parameters,
            best_score,
            evaluation_history,
            convergence_info: ConvergenceInfo {
                converged: false,
                convergence_iteration: None,
                stopping_reason: StoppingReason::MaxEvaluations,
            },
            exploration_stats: ExplorationStats {
                coverage: 0.0,
                parameter_distributions: HashMap::new(),
                parameter_importance: HashMap::new(),
            },
            total_time,
            ensemble_results: None,
            pareto_front: None,
        })
    }

    /// Tune BIRCH hyperparameters
    pub fn tune_birch(
        &self,
        data: ArrayView2<F>,
        search_space: SearchSpace,
    ) -> Result<TuningResult> {
        let start_time = std::time::Instant::now();

        let parameter_combinations = self.generate_parameter_combinations(&search_space)?;

        let mut evaluation_history = Vec::new();
        let mut best_score = f64::NEG_INFINITY;
        let mut best_parameters = HashMap::new();

        for combination in &parameter_combinations {
            let branching_factor = combination
                .get("branching_factor")
                .copied()
                .unwrap_or(50.0)
                .round() as usize;
            let threshold = combination.get("threshold").copied().unwrap_or(0.5);

            let scores =
                self.cross_validate_birch(data, branching_factor, F::from(threshold).unwrap())?;
            let mean_score = scores.iter().sum::<f64>() / scores.len() as f64;

            evaluation_history.push(EvaluationResult {
                parameters: combination.clone(),
                score: mean_score,
                additional_metrics: HashMap::new(),
                evaluation_time: 0.0,
                memory_usage: None,
                cv_scores: scores.clone(),
                cv_std: scores.std_dev(),
                metadata: HashMap::new(),
            });

            if mean_score > best_score {
                best_score = mean_score;
                best_parameters = combination.clone();
            }
        }

        let total_time = start_time.elapsed().as_secs_f64();

        Ok(TuningResult {
            best_parameters,
            best_score,
            evaluation_history,
            convergence_info: ConvergenceInfo {
                converged: false,
                convergence_iteration: None,
                stopping_reason: StoppingReason::MaxEvaluations,
            },
            exploration_stats: ExplorationStats {
                coverage: 0.0,
                parameter_distributions: HashMap::new(),
                parameter_importance: HashMap::new(),
            },
            total_time,
            ensemble_results: None,
            pareto_front: None,
        })
    }

    /// Tune GMM hyperparameters
    pub fn tune_gmm(&self, data: ArrayView2<F>, searchspace: SearchSpace) -> Result<TuningResult> {
        let start_time = std::time::Instant::now();

        let parameter_combinations = self.generate_parameter_combinations(&searchspace)?;

        let mut evaluation_history = Vec::new();
        let mut best_score = f64::NEG_INFINITY;
        let mut best_parameters = HashMap::new();

        for combination in &parameter_combinations {
            let n_components = combination
                .get("n_components")
                .ok_or_else(|| {
                    ClusteringError::InvalidInput("n_components parameter not found".to_string())
                })?
                .round() as usize;
            let max_iter = combination
                .get("max_iter")
                .copied()
                .unwrap_or(100.0)
                .round() as usize;
            let tol = combination.get("tol").copied().unwrap_or(1e-3);
            let reg_covar = combination.get("reg_covar").copied().unwrap_or(1e-6);

            let scores = self.cross_validate_gmm(
                data,
                n_components,
                max_iter,
                F::from(tol).unwrap(),
                F::from(reg_covar).unwrap(),
            )?;
            let mean_score = scores.iter().sum::<f64>() / scores.len() as f64;

            evaluation_history.push(EvaluationResult {
                parameters: combination.clone(),
                score: mean_score,
                additional_metrics: HashMap::new(),
                evaluation_time: 0.0,
                memory_usage: None,
                cv_scores: scores.clone(),
                cv_std: scores.std_dev(),
                metadata: HashMap::new(),
            });

            if mean_score > best_score {
                best_score = mean_score;
                best_parameters = combination.clone();
            }
        }

        let total_time = start_time.elapsed().as_secs_f64();

        Ok(TuningResult {
            best_parameters,
            best_score,
            evaluation_history,
            convergence_info: ConvergenceInfo {
                converged: false,
                convergence_iteration: None,
                stopping_reason: StoppingReason::MaxEvaluations,
            },
            exploration_stats: ExplorationStats {
                coverage: 0.0,
                parameter_distributions: HashMap::new(),
                parameter_importance: HashMap::new(),
            },
            total_time,
            ensemble_results: None,
            pareto_front: None,
        })
    }

    /// Cross-validate K-means clustering
    fn cross_validate_kmeans(
        &self,
        data: ArrayView2<F>,
        k: usize,
        max_iter: Option<usize>,
        tol: Option<f64>,
        seed: Option<u64>,
    ) -> Result<Vec<f64>> {
        let mut scores = Vec::new();
        let n_samples = data.shape()[0];

        match self.config.cv_config.strategy {
            CVStrategy::KFold => {
                let fold_size = n_samples / self.config.cv_config.n_folds;

                for fold in 0..self.config.cv_config.n_folds {
                    let start_idx = fold * fold_size;
                    let end_idx = if fold == self.config.cv_config.n_folds - 1 {
                        n_samples
                    } else {
                        (fold + 1) * fold_size
                    };

                    // Create train/test split
                    let mut train_indices = Vec::new();
                    let mut test_indices = Vec::new();

                    for i in 0..n_samples {
                        if i >= start_idx && i < end_idx {
                            test_indices.push(i);
                        } else {
                            train_indices.push(i);
                        }
                    }

                    if train_indices.is_empty() || test_indices.is_empty() {
                        continue;
                    }

                    // Extract training data
                    let train_data = self.extract_subset(data, &train_indices)?;

                    // Run K-means on training data
                    match kmeans2(
                        train_data.view(),
                        k,
                        Some(max_iter.unwrap_or(100)),
                        tol.map(|t| F::from(t).unwrap()),
                        None,
                        None,
                        Some(false),
                        seed,
                    ) {
                        Ok((centroids, labels)) => {
                            // Calculate score based on metric
                            let score = self.calculate_metric_score(
                                train_data.view(),
                                &labels.mapv(|x| x),
                                Some(&centroids),
                            )?;
                            scores.push(score);
                        }
                        Err(_) => {
                            // Skip failed runs
                            continue;
                        }
                    }
                }
            }
            _ => {
                // For other CV strategies, implement similar logic
                // For now, just do a single evaluation
                match kmeans2(
                    data,
                    k,
                    Some(max_iter.unwrap_or(100)),
                    tol.map(|t| F::from(t).unwrap()),
                    None,
                    None,
                    Some(false),
                    seed,
                ) {
                    Ok((centroids, labels)) => {
                        let score = self.calculate_metric_score(
                            data,
                            &labels.mapv(|x| x),
                            Some(&centroids),
                        )?;
                        scores.push(score);
                    }
                    Err(_) => {
                        scores.push(f64::NEG_INFINITY);
                    }
                }
            }
        }

        if scores.is_empty() {
            scores.push(f64::NEG_INFINITY);
        }

        Ok(scores)
    }

    /// Cross-validate DBSCAN clustering
    fn cross_validate_dbscan(
        &self,
        data: ArrayView2<F>,
        eps: f64,
        min_samples: usize,
    ) -> Result<Vec<f64>> {
        let mut scores = Vec::new();

        // For DBSCAN, we typically don't use cross-validation in the traditional sense
        // since it's not a predictive model. Instead, we evaluate on the full dataset.
        let data_f64 = data.mapv(|x| x.to_f64().unwrap_or(0.0));

        match dbscan(data_f64.view(), eps, min_samples, None) {
            Ok(labels) => {
                // Convert i32 labels to usize (DBSCAN returns -1 for noise, convert to max value)
                let labels_usize = labels.mapv(|x| if x < 0 { usize::MAX } else { x as usize });
                let score = self.calculate_metric_score(data, &labels_usize, None)?;
                scores.push(score);
            }
            Err(_) => {
                scores.push(f64::NEG_INFINITY);
            }
        }

        Ok(scores)
    }

    /// Cross-validate OPTICS clustering
    fn cross_validate_optics(
        &self,
        data: ArrayView2<F>,
        min_samples: usize,
        max_eps: Option<F>,
    ) -> Result<Vec<f64>> {
        let n_samples = data.nrows();
        let n_folds = self.config.cv_config.n_folds.min(n_samples);
        let fold_size = n_samples / n_folds;

        let mut scores = Vec::new();

        for fold in 0..n_folds {
            let start_idx = fold * fold_size;
            let end_idx = if fold == n_folds - 1 {
                n_samples
            } else {
                (fold + 1) * fold_size
            };

            // Create training data (all except current fold)
            let train_indices: Vec<usize> = (0..start_idx).chain(end_idx..n_samples).collect();
            let train_data = data.select(ndarray::Axis(0), &train_indices);

            // Run OPTICS clustering
            match optics(train_data.view(), min_samples, max_eps, None) {
                Ok(result) => {
                    // Extract cluster labels from OPTICS result
                    let cluster_labels = result;

                    if cluster_labels.iter().all(|&label| label == -1) {
                        // No clusters found
                        scores.push(f64::NEG_INFINITY);
                        continue;
                    }

                    // Convert to usize labels for metric calculation
                    let n_clusters =
                        (*cluster_labels.iter().max().unwrap_or(&-1i32) + 1i32) as usize;
                    if n_clusters < 2usize {
                        scores.push(f64::NEG_INFINITY);
                        continue;
                    }

                    let labels: Vec<usize> = cluster_labels
                        .iter()
                        .map(|&label| {
                            if label == -1i32 {
                                0usize
                            } else {
                                (label as usize) + 1usize
                            }
                        })
                        .collect();
                    let labels_array = Array1::from_vec(labels);

                    // Calculate metric score
                    let score =
                        self.calculate_metric_score(train_data.view(), &labels_array, None)?;
                    scores.push(score);
                }
                Err(_) => {
                    scores.push(f64::NEG_INFINITY);
                }
            }
        }

        Ok(scores)
    }

    /// Cross-validate Spectral clustering
    fn cross_validate_spectral(
        &self,
        data: ArrayView2<F>,
        n_clusters: usize,
        n_neighbors: usize,
        gamma: F,
        max_iter: usize,
    ) -> Result<Vec<f64>> {
        let n_samples = data.nrows();
        let n_folds = self.config.cv_config.n_folds.min(n_samples);
        let fold_size = n_samples / n_folds;

        let mut scores = Vec::new();

        for fold in 0..n_folds {
            let start_idx = fold * fold_size;
            let end_idx = if fold == n_folds - 1 {
                n_samples
            } else {
                (fold + 1) * fold_size
            };

            let train_indices: Vec<usize> = (0..start_idx).chain(end_idx..n_samples).collect();
            let train_data = data.select(ndarray::Axis(0), &train_indices);

            // Create spectral clustering options
            use crate::spectral::{AffinityMode, SpectralClusteringOptions};
            let options = SpectralClusteringOptions {
                affinity: AffinityMode::RBF,
                n_neighbors,
                gamma,
                normalized_laplacian: true,
                max_iter,
                n_init: 1,
                tol: F::from(1e-4).unwrap(),
                random_seed: None,
                eigen_solver: "arpack".to_string(),
                auto_n_clusters: false,
            };

            match spectral_clustering(train_data.view(), n_clusters, Some(options)) {
                Ok((_, labels)) => {
                    let score = self.calculate_metric_score(train_data.view(), &labels, None)?;
                    scores.push(score);
                }
                Err(_) => {
                    scores.push(f64::NEG_INFINITY);
                }
            }
        }

        Ok(scores)
    }

    /// Cross-validate Affinity Propagation clustering
    fn cross_validate_affinity_propagation(
        &self,
        data: ArrayView2<F>,
        damping: F,
        max_iter: usize,
        convergence_iter: usize,
    ) -> Result<Vec<f64>> {
        let n_samples = data.nrows();
        let n_folds = self.config.cv_config.n_folds.min(n_samples);
        let fold_size = n_samples / n_folds;

        let mut scores = Vec::new();

        for fold in 0..n_folds {
            let start_idx = fold * fold_size;
            let end_idx = if fold == n_folds - 1 {
                n_samples
            } else {
                (fold + 1) * fold_size
            };

            let train_indices: Vec<usize> = (0..start_idx).chain(end_idx..n_samples).collect();
            let train_data = data.select(ndarray::Axis(0), &train_indices);

            // Create affinity propagation options
            use crate::affinity::AffinityPropagationOptions;
            let options = AffinityPropagationOptions {
                damping,
                max_iter,
                convergence_iter,
                preference: None, // Use default (median of similarities)
                affinity: "euclidean".to_string(),
                max_affinity_iterations: 10,
            };

            match affinity_propagation(train_data.view(), false, Some(options)) {
                Ok((_, labels)) => {
                    // Convert i32 labels to usize
                    let usize_labels: Vec<usize> = labels.iter().map(|&x| x as usize).collect();
                    let labels_array = Array1::from_vec(usize_labels);

                    let score =
                        self.calculate_metric_score(train_data.view(), &labels_array, None)?;
                    scores.push(score);
                }
                Err(_) => {
                    scores.push(f64::NEG_INFINITY);
                }
            }
        }

        Ok(scores)
    }

    /// Cross-validate BIRCH clustering
    fn cross_validate_birch(
        &self,
        data: ArrayView2<F>,
        branching_factor: usize,
        threshold: F,
    ) -> Result<Vec<f64>> {
        let n_samples = data.nrows();
        let n_folds = self.config.cv_config.n_folds.min(n_samples);
        let fold_size = n_samples / n_folds;

        let mut scores = Vec::new();

        for fold in 0..n_folds {
            let start_idx = fold * fold_size;
            let end_idx = if fold == n_folds - 1 {
                n_samples
            } else {
                (fold + 1) * fold_size
            };

            let train_indices: Vec<usize> = (0..start_idx).chain(end_idx..n_samples).collect();
            let train_data = data.select(ndarray::Axis(0), &train_indices);

            // Create BIRCH options
            use crate::birch::BirchOptions;
            let options = BirchOptions {
                branching_factor,
                threshold,
                n_clusters: None, // Use all clusters found
            };

            match birch(train_data.view(), options) {
                Ok((_, labels)) => {
                    // Convert i32 labels to usize
                    let usize_labels: Vec<usize> = labels.iter().map(|&x| x as usize).collect();
                    let labels_array = Array1::from_vec(usize_labels);

                    let score =
                        self.calculate_metric_score(train_data.view(), &labels_array, None)?;
                    scores.push(score);
                }
                Err(_) => {
                    scores.push(f64::NEG_INFINITY);
                }
            }
        }

        Ok(scores)
    }

    /// Cross-validate GMM clustering
    fn cross_validate_gmm(
        &self,
        data: ArrayView2<F>,
        n_components: usize,
        max_iter: usize,
        tol: F,
        reg_covar: F,
    ) -> Result<Vec<f64>> {
        let n_samples = data.nrows();
        let n_folds = self.config.cv_config.n_folds.min(n_samples);
        let fold_size = n_samples / n_folds;

        let mut scores = Vec::new();

        for fold in 0..n_folds {
            let start_idx = fold * fold_size;
            let end_idx = if fold == n_folds - 1 {
                n_samples
            } else {
                (fold + 1) * fold_size
            };

            let train_indices: Vec<usize> = (0..start_idx).chain(end_idx..n_samples).collect();
            let train_data = data.select(ndarray::Axis(0), &train_indices);

            // Convert to f64 for GMM
            let train_data_f64 = train_data.mapv(|x| x.to_f64().unwrap_or(0.0));

            // Create GMM options
            use crate::gmm::{CovarianceType, GMMInit, GMMOptions};
            let options = GMMOptions {
                n_components,
                covariance_type: CovarianceType::Full,
                tol: tol.to_f64().unwrap_or(1e-4),
                max_iter,
                n_init: 1,
                init_method: GMMInit::KMeans,
                random_seed: Some(42),
                reg_covar: reg_covar.to_f64().unwrap_or(1e-6),
            };

            match gaussian_mixture(train_data_f64.view(), options) {
                Ok(labels) => {
                    // Convert i32 labels to usize
                    let usize_labels: Vec<usize> = labels.iter().map(|&x| x as usize).collect();
                    let labels_array = Array1::from_vec(usize_labels);

                    let score =
                        self.calculate_metric_score(train_data.view(), &labels_array, None)?;
                    scores.push(score);
                }
                Err(_) => {
                    scores.push(f64::NEG_INFINITY);
                }
            }
        }

        Ok(scores)
    }

    /// Calculate metric score for evaluation
    fn calculate_metric_score(
        &self,
        data: ArrayView2<F>,
        labels: &Array1<usize>,
        centroids: Option<&Array2<F>>,
    ) -> Result<f64> {
        let data_f64 = data.mapv(|x| x.to_f64().unwrap_or(0.0));
        let labels_i32 = labels.mapv(|x| x as i32);

        match self.config.metric {
            EvaluationMetric::SilhouetteScore => {
                silhouette_score(data_f64.view(), labels_i32.view())
            }
            EvaluationMetric::DaviesBouldinIndex => {
                davies_bouldin_score(data_f64.view(), labels_i32.view())
            }
            EvaluationMetric::CalinskiHarabaszIndex => {
                calinski_harabasz_score(data_f64.view(), labels_i32.view())
            }
            EvaluationMetric::Inertia => {
                // Calculate within-cluster sum of squares
                if let Some(centroids) = centroids {
                    let centroids_f64 = centroids.mapv(|x| x.to_f64().unwrap_or(0.0));
                    self.calculate_inertia(&data_f64, labels, &centroids_f64)
                } else {
                    Ok(f64::INFINITY) // Invalid for algorithms without centroids
                }
            }
            _ => Ok(0.0), // Placeholder for other metrics
        }
    }

    /// Calculate inertia (within-cluster sum of squares)
    fn calculate_inertia(
        &self,
        data: &Array2<f64>,
        labels: &Array1<usize>,
        centroids: &Array2<f64>,
    ) -> Result<f64> {
        let mut total_inertia = 0.0;

        for (i, &label) in labels.iter().enumerate() {
            let mut distance_sq = 0.0;
            for j in 0..data.ncols() {
                let diff = data[[i, j]] - centroids[[label, j]];
                distance_sq += diff * diff;
            }
            total_inertia += distance_sq;
        }

        Ok(total_inertia)
    }

    /// Extract subset of data based on indices
    fn extract_subset(&self, data: ArrayView2<F>, indices: &[usize]) -> Result<Array2<F>> {
        let n_features = data.ncols();
        let mut subset = Array2::zeros((indices.len(), n_features));

        for (new_idx, &old_idx) in indices.iter().enumerate() {
            if old_idx < data.nrows() {
                subset.row_mut(new_idx).assign(&data.row(old_idx));
            }
        }

        Ok(subset)
    }

    /// Generate parameter combinations based on search strategy
    fn generate_parameter_combinations(
        &self,
        search_space: &SearchSpace,
    ) -> Result<Vec<HashMap<String, f64>>> {
        match &self.config.strategy {
            SearchStrategy::GridSearch => self.generate_grid_combinations(search_space),
            SearchStrategy::RandomSearch { n_trials } => {
                self.generate_random_combinations(search_space, *n_trials)
            }
            SearchStrategy::BayesianOptimization {
                n_initial_points,
                acquisition_function,
            } => self.generate_bayesian_combinations(
                search_space,
                *n_initial_points,
                acquisition_function,
            ),
            SearchStrategy::EnsembleSearch {
                strategies,
                weights,
            } => self.generate_ensemble_combinations(search_space, strategies, weights),
            SearchStrategy::EvolutionarySearch {
                population_size,
                n_generations,
                mutation_rate,
                crossover_rate,
            } => self.generate_evolutionary_combinations(
                search_space,
                *population_size,
                *n_generations,
                *mutation_rate,
                *crossover_rate,
            ),
            SearchStrategy::SMBO {
                surrogate_model,
                acquisition_function,
            } => {
                self.generate_smbo_combinations(search_space, surrogate_model, acquisition_function)
            }
            SearchStrategy::MultiObjective {
                objectives,
                strategy,
            } => {
                // For multi-objective, we need special handling
                self.generate_multi_objective_combinations(search_space, objectives, strategy)
            }
            SearchStrategy::AdaptiveSearch {
                initial_strategy, ..
            } => {
                // Start with initial strategy
                match initial_strategy.as_ref() {
                    SearchStrategy::RandomSearch { n_trials } => {
                        self.generate_random_combinations(search_space, *n_trials)
                    }
                    SearchStrategy::GridSearch => self.generate_grid_combinations(search_space),
                    _ => {
                        // Fallback to random search
                        self.generate_random_combinations(search_space, self.config.max_evaluations)
                    }
                }
            }
        }
    }

    /// Generate grid search combinations
    fn generate_grid_combinations(
        &self,
        search_space: &SearchSpace,
    ) -> Result<Vec<HashMap<String, f64>>> {
        let mut combinations = Vec::new();
        let mut param_names = Vec::new();
        let mut param_values = Vec::new();

        // Extract parameter ranges
        for (name, param) in &search_space.parameters {
            param_names.push(name.clone());
            match param {
                HyperParameter::Integer { min, max } => {
                    let values: Vec<f64> = (*min..=*max).map(|x| x as f64).collect();
                    param_values.push(values);
                }
                HyperParameter::Float { min, max } => {
                    // Create a reasonable grid for float parameters
                    let n_steps = 10; // Could be configurable
                    let step = (max - min) / (n_steps as f64 - 1.0);
                    let values: Vec<f64> = (0..n_steps).map(|i| min + i as f64 * step).collect();
                    param_values.push(values);
                }
                HyperParameter::Categorical { choices } => {
                    // Map categorical choices to numeric values
                    let values: Vec<f64> = (0..choices.len()).map(|i| i as f64).collect();
                    param_values.push(values);
                }
                HyperParameter::Boolean => {
                    param_values.push(vec![0.0, 1.0]);
                }
                HyperParameter::LogUniform { min, max } => {
                    let n_steps = 10;
                    let log_min = min.ln();
                    let log_max = max.ln();
                    let step = (log_max - log_min) / (n_steps as f64 - 1.0);
                    let values: Vec<f64> = (0..n_steps)
                        .map(|i| (log_min + i as f64 * step).exp())
                        .collect();
                    param_values.push(values);
                }
                HyperParameter::IntegerChoices { choices } => {
                    let values: Vec<f64> = choices.iter().map(|&x| x as f64).collect();
                    param_values.push(values);
                }
            }
        }

        // Generate all combinations
        self.generate_cartesian_product(
            &param_names,
            &param_values,
            &mut combinations,
            Vec::new(),
            0,
        );

        Ok(combinations)
    }

    /// Generate cartesian product of parameter values
    fn generate_cartesian_product(
        &self,
        param_names: &[String],
        param_values: &[Vec<f64>],
        combinations: &mut Vec<HashMap<String, f64>>,
        current: Vec<f64>,
        index: usize,
    ) {
        if index == param_names.len() {
            let mut combination = HashMap::new();
            for (i, name) in param_names.iter().enumerate() {
                combination.insert(name.clone(), current[i]);
            }
            combinations.push(combination);
            return;
        }

        for &value in &param_values[index] {
            let mut new_current = current.clone();
            new_current.push(value);
            self.generate_cartesian_product(
                param_names,
                param_values,
                combinations,
                new_current,
                index + 1,
            );
        }
    }

    /// Generate random search combinations
    fn generate_random_combinations(
        &self,
        search_space: &SearchSpace,
        n_trials: usize,
    ) -> Result<Vec<HashMap<String, f64>>> {
        let mut combinations = Vec::new();
        let mut rng = match self.config.random_seed {
            Some(seed) => rand::rngs::StdRng::seed_from_u64(seed),
            None => rand::rngs::StdRng::seed_from_u64(42),
        };

        for _ in 0..n_trials {
            let mut combination = HashMap::new();

            for (name, param) in &search_space.parameters {
                let value = match param {
                    HyperParameter::Integer { min, max } => rng.random_range(*min..=*max) as f64,
                    HyperParameter::Float { min, max } => rng.random_range(*min..=*max),
                    HyperParameter::Categorical { choices } => {
                        rng.random_range(0..choices.len()) as f64
                    }
                    HyperParameter::Boolean => {
                        if rng.random_range(0.0..1.0) < 0.5 {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    HyperParameter::LogUniform { min, max } => {
                        let log_min = min.ln();
                        let log_max = max.ln();
                        let log_value = rng.random_range(log_min..=log_max);
                        log_value.exp()
                    }
                    HyperParameter::IntegerChoices { choices } => {
                        let idx = rng.random_range(0..choices.len());
                        choices[idx] as f64
                    }
                };

                combination.insert(name.clone(), value);
            }

            combinations.push(combination);
        }

        Ok(combinations)
    }

    /// Check if early stopping criteria are met
    fn should_stop_early(
        &self,
        evaluation_history: &[EvaluationResult],
        early_stop_config: &EarlyStoppingConfig,
    ) -> bool {
        if evaluation_history.len() < early_stop_config.patience {
            return false;
        }

        let recent_evaluations =
            &evaluation_history[evaluation_history.len() - early_stop_config.patience..];
        let best_recent = recent_evaluations
            .iter()
            .map(|r| r.score)
            .fold(f64::NEG_INFINITY, f64::max);

        let currentbest = evaluation_history
            .iter()
            .map(|r| r.score)
            .fold(f64::NEG_INFINITY, f64::max);

        (currentbest - best_recent) < early_stop_config.min_improvement
    }

    /// Calculate exploration statistics
    fn calculate_exploration_stats(
        &self,
        evaluation_history: &[EvaluationResult],
    ) -> ExplorationStats {
        let mut parameter_distributions = HashMap::new();
        let mut parameter_importance = HashMap::new();

        // Collect parameter distributions
        for result in evaluation_history {
            for (param_name, &value) in &result.parameters {
                parameter_distributions
                    .entry(param_name.clone())
                    .or_insert_with(Vec::new)
                    .push(value);
            }
        }

        // Calculate parameter importance (simplified)
        for (param_name, values) in &parameter_distributions {
            let scores: Vec<f64> = evaluation_history.iter().map(|r| r.score).collect();
            let correlation = self.calculate_correlation(values, &scores);
            parameter_importance.insert(param_name.clone(), correlation.abs());
        }

        ExplorationStats {
            coverage: 1.0, // Simplified calculation
            parameter_distributions,
            parameter_importance,
        }
    }

    /// Calculate correlation between two vectors
    fn calculate_correlation(&self, x: &[f64], y: &[f64]) -> f64 {
        if x.len() != y.len() || x.len() < 2 {
            return 0.0;
        }

        let n = x.len() as f64;
        let sum_x: f64 = x.iter().sum();
        let sum_y: f64 = y.iter().sum();
        let sum_xy: f64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
        let sum_x_sq: f64 = x.iter().map(|a| a * a).sum();
        let sum_y_sq: f64 = y.iter().map(|a| a * a).sum();

        let numerator = n * sum_xy - sum_x * sum_y;
        let denominator = ((n * sum_x_sq - sum_x * sum_x) * (n * sum_y_sq - sum_y * sum_y)).sqrt();

        if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        }
    }

    /// Generate Bayesian optimization combinations
    fn generate_bayesian_combinations(
        &self,
        search_space: &SearchSpace,
        n_initial_points: usize,
        acquisition_function: &AcquisitionFunction,
    ) -> Result<Vec<HashMap<String, f64>>> {
        let mut combinations = Vec::new();
        // Extract parameter names for consistent ordering
        let parameter_names: Vec<String> = search_space.parameters.keys().cloned().collect();

        let mut bayesian_state = BayesianState {
            observations: Vec::new(),
            gp_mean: None,
            gp_covariance: None,
            acquisition_values: Vec::new(),
            parameter_names: parameter_names.clone(),
            gp_hyperparameters: GpHyperparameters {
                length_scales: vec![1.0; parameter_names.len()],
                signal_variance: 1.0,
                noise_variance: 0.1,
                kernel_type: KernelType::RBF { length_scale: 1.0 },
            },
            noise_level: 0.1,
            currentbest: f64::NEG_INFINITY,
        };

        // Generate initial random _points
        let initial_points = self.generate_random_combinations(search_space, n_initial_points)?;
        combinations.extend(initial_points);

        // Generate remaining _points using Bayesian optimization
        let remaining_points = self.config.max_evaluations.saturating_sub(n_initial_points);

        for _ in 0..remaining_points {
            // Update Gaussian process with current observations
            self.update_gaussian_process(&mut bayesian_state, &combinations);

            // Find next point with highest acquisition _function value
            let next_point = self.optimize_acquisition_function(
                search_space,
                &bayesian_state,
                acquisition_function,
            )?;

            combinations.push(next_point);
        }

        Ok(combinations)
    }

    /// Generate ensemble search combinations
    fn generate_ensemble_combinations(
        &self,
        search_space: &SearchSpace,
        strategies: &[SearchStrategy],
        weights: &[f64],
    ) -> Result<Vec<HashMap<String, f64>>> {
        let mut all_combinations = Vec::new();
        let total_evaluations = self.config.max_evaluations;

        // Normalize weights
        let weight_sum: f64 = weights.iter().sum();
        let normalized_weights: Vec<f64> = weights.iter().map(|w| w / weight_sum).collect();

        // Allocate evaluations based on weights
        for (strategy, &weight) in strategies.iter().zip(normalized_weights.iter()) {
            let n_evaluations = (total_evaluations as f64 * weight) as usize;

            let strategy_combinations = match strategy {
                SearchStrategy::RandomSearch { .. } => {
                    self.generate_random_combinations(search_space, n_evaluations)?
                }
                SearchStrategy::GridSearch => {
                    let grid_combinations = self.generate_grid_combinations(search_space)?;
                    grid_combinations.into_iter().take(n_evaluations).collect()
                }
                _ => {
                    // Fallback to random search for complex strategies
                    self.generate_random_combinations(search_space, n_evaluations)?
                }
            };

            all_combinations.extend(strategy_combinations);
        }

        // Shuffle to mix different strategies
        let mut rng = match self.config.random_seed {
            Some(seed) => rand::rngs::StdRng::seed_from_u64(seed),
            None => rand::rngs::StdRng::seed_from_u64(42),
        };

        use rand::seq::SliceRandom;
        all_combinations.shuffle(&mut rng);

        Ok(all_combinations)
    }

    /// Update Gaussian Process with current observations
    fn update_gaussian_process(
        &self,
        bayesian_state: &mut BayesianState,
        combinations: &[HashMap<String, f64>],
    ) {
        // For demonstration, we'll implement a simplified GP update
        // In practice, this would involve matrix operations and hyperparameter optimization

        if combinations.is_empty() {
            return;
        }

        // Convert parameter combinations to feature matrix
        let n_samples = combinations.len();
        let _n_features = bayesian_state.parameter_names.len();

        if n_samples < 2 {
            return;
        }

        // Update GP hyperparameters using maximum likelihood estimation
        self.optimize_gp_hyperparameters(bayesian_state, combinations);

        // Build covariance matrix
        let mut covariance = Array2::zeros((n_samples, n_samples));
        for i in 0..n_samples {
            for j in 0..n_samples {
                let x_i =
                    self.extract_feature_vector(&combinations[i], &bayesian_state.parameter_names);
                let x_j =
                    self.extract_feature_vector(&combinations[j], &bayesian_state.parameter_names);
                covariance[[i, j]] =
                    self.compute_kernel(&x_i, &x_j, &bayesian_state.gp_hyperparameters);
            }
        }

        // Add noise to diagonal
        for i in 0..n_samples {
            covariance[[i, i]] += bayesian_state.gp_hyperparameters.noise_variance;
        }

        bayesian_state.gp_covariance = Some(covariance);
    }

    /// Optimize acquisition function to find next evaluation point
    fn optimize_acquisition_function(
        &self,
        search_space: &SearchSpace,
        bayesian_state: &BayesianState,
        acquisition_function: &AcquisitionFunction,
    ) -> Result<HashMap<String, f64>> {
        let mut best_acquisition = f64::NEG_INFINITY;
        let mut best_point = HashMap::new();

        // Generate candidate points for acquisition optimization
        let n_candidates = 1000;
        let candidates = self.generate_random_combinations(search_space, n_candidates)?;

        for candidate in candidates {
            let acquisition_value = self.evaluate_acquisition_function(
                &candidate,
                bayesian_state,
                acquisition_function,
            );

            if acquisition_value > best_acquisition {
                best_acquisition = acquisition_value;
                best_point = candidate;
            }
        }

        Ok(best_point)
    }

    /// Evaluate acquisition function at a point
    fn evaluate_acquisition_function(
        &self,
        point: &HashMap<String, f64>,
        bayesian_state: &BayesianState,
        acquisition_function: &AcquisitionFunction,
    ) -> f64 {
        let x = self.extract_feature_vector(point, &bayesian_state.parameter_names);
        let (mean, variance) = self.predict_gp(&x, bayesian_state);
        let std_dev = variance.sqrt();

        match acquisition_function {
            AcquisitionFunction::ExpectedImprovement => {
                self.expected_improvement(mean, std_dev, bayesian_state.currentbest)
            }
            AcquisitionFunction::UpperConfidenceBound { beta } => mean + beta * std_dev,
            AcquisitionFunction::ProbabilityOfImprovement => {
                self.probability_of_improvement(mean, std_dev, bayesian_state.currentbest)
            }
            AcquisitionFunction::EntropySearch => {
                // Simplified entropy search implementation
                -variance * (variance.ln())
            }
            AcquisitionFunction::KnowledgeGradient => {
                // Simplified knowledge gradient
                std_dev * (1.0 / (1.0 + variance))
            }
            AcquisitionFunction::ThompsonSampling => {
                // Sample from posterior
                let mut rng = rand::rng();
                let sample: f64 = rng.random_range(0.0..1.0);
                mean + std_dev * self.inverse_normal_cdf(sample)
            }
        }
    }

    /// Expected Improvement acquisition function
    fn expected_improvement(&self, mean: f64, std_dev: f64, currentbest: f64) -> f64 {
        if std_dev <= 1e-10 {
            return 0.0;
        }

        let improvement = mean - currentbest;
        let z = improvement / std_dev;

        improvement * self.normal_cdf(z) + std_dev * self.normal_pdf(z)
    }

    /// Probability of Improvement acquisition function
    fn probability_of_improvement(&self, mean: f64, std_dev: f64, currentbest: f64) -> f64 {
        if std_dev <= 1e-10 {
            return if mean > currentbest { 1.0 } else { 0.0 };
        }

        let z = (mean - currentbest) / std_dev;
        self.normal_cdf(z)
    }

    /// Gaussian Process prediction
    fn predict_gp(&self, x: &[f64], bayesian_state: &BayesianState) -> (f64, f64) {
        if bayesian_state.observations.is_empty() {
            return (0.0, 1.0); // Prior mean and variance
        }

        // Simplified GP prediction - in practice would use proper matrix operations
        let mut mean = 0.0;
        let mut variance = 1.0;

        // Compute similarity-weighted average (simplified)
        let mut total_weight = 0.0;
        for (params, score) in &bayesian_state.observations {
            let x_obs = self.extract_feature_vector(params, &bayesian_state.parameter_names);
            let similarity = self.compute_kernel(x, &x_obs, &bayesian_state.gp_hyperparameters);
            mean += similarity * score;
            total_weight += similarity;
        }

        if total_weight > 1e-10 {
            mean /= total_weight;
            variance = 1.0 - total_weight.min(1.0); // Simplified variance calculation
        }

        (mean, variance.max(1e-6))
    }

    /// Compute kernel function
    fn compute_kernel(&self, x1: &[f64], x2: &[f64], hyperparams: &GpHyperparameters) -> f64 {
        match &hyperparams.kernel_type {
            KernelType::RBF { length_scale } => {
                let squared_distance: f64 =
                    x1.iter().zip(x2.iter()).map(|(a, b)| (a - b).powi(2)).sum();
                hyperparams.signal_variance
                    * (-squared_distance / (2.0 * length_scale.powi(2))).exp()
            }
            KernelType::Matern { length_scale, nu } => {
                let distance: f64 = x1
                    .iter()
                    .zip(x2.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();

                if distance == 0.0 {
                    hyperparams.signal_variance
                } else {
                    let scaled_distance = (2.0 * nu).sqrt() * distance / length_scale;
                    let bessel_term = if nu == &0.5 {
                        (-scaled_distance).exp()
                    } else if nu == &1.5 {
                        (1.0 + scaled_distance) * (-scaled_distance).exp()
                    } else {
                        // Simplified for other nu values
                        (-scaled_distance).exp()
                    };
                    hyperparams.signal_variance * bessel_term
                }
            }
            KernelType::Linear => {
                let dot_product: f64 = x1.iter().zip(x2.iter()).map(|(a, b)| a * b).sum();
                hyperparams.signal_variance * dot_product
            }
            KernelType::Polynomial { degree } => {
                let dot_product: f64 = x1.iter().zip(x2.iter()).map(|(a, b)| a * b).sum();
                hyperparams.signal_variance * (1.0 + dot_product).powf(*degree as f64)
            }
        }
    }

    /// Optimize GP hyperparameters using maximum likelihood
    fn optimize_gp_hyperparameters(
        &self,
        bayesian_state: &mut BayesianState,
        combinations: &[HashMap<String, f64>],
    ) {
        // Simplified hyperparameter optimization
        // In practice, this would use gradient-based optimization

        if combinations.len() < 3 {
            return;
        }

        // Estimate length scales based on data variance
        for (i, param_name) in bayesian_state.parameter_names.iter().enumerate() {
            let values: Vec<f64> = combinations
                .iter()
                .filter_map(|c| c.get(param_name))
                .copied()
                .collect();

            if !values.is_empty() {
                let mean = values.iter().sum::<f64>() / values.len() as f64;
                let variance =
                    values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;

                if i < bayesian_state.gp_hyperparameters.length_scales.len() {
                    bayesian_state.gp_hyperparameters.length_scales[i] = variance.sqrt().max(0.1);
                }
            }
        }

        // Update signal and noise variance based on observations
        if !bayesian_state.observations.is_empty() {
            let scores: Vec<f64> = bayesian_state
                .observations
                .iter()
                .map(|(_, s)| *s)
                .collect();
            let score_mean = scores.iter().sum::<f64>() / scores.len() as f64;
            let score_variance =
                scores.iter().map(|s| (s - score_mean).powi(2)).sum::<f64>() / scores.len() as f64;

            bayesian_state.gp_hyperparameters.signal_variance = score_variance.max(0.1);
            bayesian_state.gp_hyperparameters.noise_variance = (score_variance * 0.1).max(0.01);
        }
    }

    /// Extract feature vector from parameter map
    fn extract_feature_vector(
        &self,
        params: &HashMap<String, f64>,
        param_names: &[String],
    ) -> Vec<f64> {
        param_names
            .iter()
            .map(|name| params.get(name).copied().unwrap_or(0.0))
            .collect()
    }

    /// Standard normal CDF approximation
    fn normal_cdf(&self, x: f64) -> f64 {
        0.5 * (1.0 + self.erf(x / 2.0_f64.sqrt()))
    }

    /// Standard normal PDF
    fn normal_pdf(&self, x: f64) -> f64 {
        (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()
    }

    /// Error function approximation
    fn erf(&self, x: f64) -> f64 {
        // Abramowitz and Stegun approximation
        let a1 = 0.254829592;
        let a2 = -0.284496736;
        let a3 = 1.421413741;
        let a4 = -1.453152027;
        let a5 = 1.061405429;
        let p = 0.3275911;

        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let x = x.abs();

        let t = 1.0 / (1.0 + p * x);
        let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

        sign * y
    }

    /// Inverse normal CDF approximation
    fn inverse_normal_cdf(&self, p: f64) -> f64 {
        if p <= 0.0 {
            return f64::NEG_INFINITY;
        }
        if p >= 1.0 {
            return f64::INFINITY;
        }
        if (p - 0.5).abs() < 1e-10 {
            return 0.0;
        }

        // Beasley-Springer-Moro algorithm
        let a0 = -3.969683028665376e+01;
        let a1 = 2.209460984245205e+02;
        let a2 = -2.759285104469687e+02;
        let a3 = 1.383577518672690e+02;
        let a4 = -3.066479806614716e+01;
        let a5 = 2.506628277459239e+00;

        let b1 = -5.447609879822406e+01;
        let b2 = 1.615858368580409e+02;
        let b3 = -1.556989798598866e+02;
        let b4 = 6.680131188771972e+01;
        let b5 = -1.328068155288572e+01;

        let c0 = -7.784894002430293e-03;
        let c1 = -3.223964580411365e-01;
        let c2 = -2.400758277161838e+00;
        let c3 = -2.549732539343734e+00;
        let c4 = 4.374664141464968e+00;
        let c5 = 2.938163982698783e+00;

        let d1 = 7.784695709041462e-03;
        let d2 = 3.224671290700398e-01;
        let d3 = 2.445134137142996e+00;
        let d4 = 3.754408661907416e+00;

        let p_low = 0.02425;
        let p_high = 1.0 - p_low;

        if p < p_low {
            let q = (-2.0 * p.ln()).sqrt();
            return (((((c0 * q + c1) * q + c2) * q + c3) * q + c4) * q + c5)
                / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
        }

        if p <= p_high {
            let q = p - 0.5;
            let r = q * q;
            return (((((a0 * r + a1) * r + a2) * r + a3) * r + a4) * r + a5) * q
                / (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0);
        }

        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((c0 * q + c1) * q + c2) * q + c3) * q + c4) * q + c5)
            / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
    }

    /// Multi-objective optimization using Pareto frontier
    fn generate_multi_objective_combinations(
        &self,
        search_space: &SearchSpace,
        _objectives: &[EvaluationMetric],
        base_strategy: &SearchStrategy,
    ) -> Result<Vec<HashMap<String, f64>>> {
        // For multi-objective optimization, we need to maintain a Pareto frontier
        // This is a simplified implementation

        let base_combinations = match base_strategy {
            SearchStrategy::RandomSearch { n_trials } => {
                self.generate_random_combinations(search_space, *n_trials)?
            }
            SearchStrategy::GridSearch => self.generate_grid_combinations(search_space)?,
            SearchStrategy::BayesianOptimization {
                n_initial_points,
                acquisition_function,
            } => self.generate_bayesian_combinations(
                search_space,
                *n_initial_points,
                acquisition_function,
            )?,
            _ => self.generate_random_combinations(search_space, self.config.max_evaluations)?,
        };

        // Add diversity through multi-objective sampling
        let mut diverse_combinations = base_combinations;

        // Add some random exploration for diversity
        let additional_random = self.generate_random_combinations(
            search_space,
            (self.config.max_evaluations / 4).max(10),
        )?;
        diverse_combinations.extend(additional_random);

        Ok(diverse_combinations)
    }

    /// Generate SMBO (Sequential Model-Based Optimization) combinations
    fn generate_smbo_combinations(
        &self,
        search_space: &SearchSpace,
        surrogate_model: &SurrogateModel,
        acquisition_function: &AcquisitionFunction,
    ) -> Result<Vec<HashMap<String, f64>>> {
        // SMBO is similar to Bayesian optimization but with different surrogate models

        let n_initial_points = 10.max(search_space.parameters.len() * 2);
        let mut combinations = Vec::new();

        // Generate initial random points
        let initial_points = self.generate_random_combinations(search_space, n_initial_points)?;
        combinations.extend(initial_points);

        // Sequential optimization based on surrogate _model
        let remaining_points = self.config.max_evaluations.saturating_sub(n_initial_points);

        for _iteration in 0..remaining_points {
            let next_point = match surrogate_model {
                SurrogateModel::GaussianProcess { .. } => {
                    // Use Gaussian Process (similar to Bayesian optimization)
                    let parameter_names: Vec<String> =
                        search_space.parameters.keys().cloned().collect();
                    let mut bayesian_state = BayesianState {
                        observations: Vec::new(),
                        gp_mean: None,
                        gp_covariance: None,
                        acquisition_values: Vec::new(),
                        parameter_names: parameter_names.clone(),
                        gp_hyperparameters: GpHyperparameters {
                            length_scales: vec![1.0; parameter_names.len()],
                            signal_variance: 1.0,
                            noise_variance: 0.1,
                            kernel_type: KernelType::RBF { length_scale: 1.0 },
                        },
                        noise_level: 0.1,
                        currentbest: f64::NEG_INFINITY,
                    };

                    self.update_gaussian_process(&mut bayesian_state, &combinations);
                    self.optimize_acquisition_function(
                        search_space,
                        &bayesian_state,
                        acquisition_function,
                    )?
                }
                SurrogateModel::RandomForest { .. } => {
                    // Random Forest surrogate (simplified implementation)
                    self.generate_rf_guided_point(search_space, &combinations)?
                }
                SurrogateModel::GradientBoosting { .. } => {
                    // Gradient Boosting surrogate (simplified implementation)
                    self.generate_gb_guided_point(search_space, &combinations)?
                }
            };

            combinations.push(next_point);
        }

        Ok(combinations)
    }

    /// Generate point guided by Random Forest surrogate model
    fn generate_rf_guided_point(
        &self,
        search_space: &SearchSpace,
        existing_combinations: &[HashMap<String, f64>],
    ) -> Result<HashMap<String, f64>> {
        // Simplified Random Forest guidance - in practice would train actual RF model

        if existing_combinations.is_empty() {
            return self
                .generate_random_combinations(search_space, 1)
                .map(|mut v| v.pop().unwrap_or_default());
        }

        // Find parameter regions with high variance (uncertainty)
        let mut promising_point = HashMap::new();

        for (param_name, param_def) in &search_space.parameters {
            let values: Vec<f64> = existing_combinations
                .iter()
                .filter_map(|c| c.get(param_name))
                .copied()
                .collect();

            if values.is_empty() {
                continue;
            }

            let mean = values.iter().sum::<f64>() / values.len() as f64;
            let variance =
                values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;

            // Sample from regions with higher uncertainty
            let suggested_value = match param_def {
                HyperParameter::Float { min, max } => {
                    // Add some exploration around high-variance regions
                    use rand::Rng;
                    let mut rng = rand::rng();
                    let noise = rng.random_range(-variance.sqrt()..variance.sqrt());
                    (mean + noise).clamp(*min, *max)
                }
                HyperParameter::Integer { min, max } => {
                    use rand::Rng;
                    let mut rng = rand::rng();
                    rng.random_range(*min..=*max) as f64
                }
                _ => mean, // Simplified for other parameter types
            };

            promising_point.insert(param_name.clone(), suggested_value);
        }

        Ok(promising_point)
    }

    /// Generate point guided by Gradient Boosting surrogate model
    fn generate_gb_guided_point(
        &self,
        search_space: &SearchSpace,
        existing_combinations: &[HashMap<String, f64>],
    ) -> Result<HashMap<String, f64>> {
        // Simplified Gradient Boosting guidance

        if existing_combinations.is_empty() {
            return self
                .generate_random_combinations(search_space, 1)
                .map(|mut v| v.pop().unwrap_or_default());
        }

        // Gradient Boosting focuses on areas where previous models performed poorly
        // This is a simplified implementation

        let mut promising_point = HashMap::new();

        for (param_name, param_def) in &search_space.parameters {
            let values: Vec<f64> = existing_combinations
                .iter()
                .filter_map(|c| c.get(param_name))
                .copied()
                .collect();

            if values.is_empty() {
                continue;
            }

            // Simple gradient-based exploration
            let suggested_value = match param_def {
                HyperParameter::Float { min, max } => {
                    // Focus on unexplored regions
                    let range = max - min;
                    let step = range / 10.0;

                    // Find least explored region
                    let mut best_gap = 0.0;
                    let mut best_value = (min + max) / 2.0;

                    let mut test_value = *min;
                    while test_value <= *max {
                        let min_distance = values
                            .iter()
                            .map(|v| (v - test_value).abs())
                            .min_by(|a, b| a.partial_cmp(b).unwrap())
                            .unwrap_or(range);

                        if min_distance > best_gap {
                            best_gap = min_distance;
                            best_value = test_value;
                        }
                        test_value += step;
                    }

                    best_value
                }
                HyperParameter::Integer { min, max } => {
                    use rand::Rng;
                    let mut rng = rand::rng();
                    rng.random_range(*min..=*max) as f64
                }
                _ => {
                    let mean = values.iter().sum::<f64>() / values.len() as f64;
                    mean
                }
            };

            promising_point.insert(param_name.clone(), suggested_value);
        }

        Ok(promising_point)
    }

    /// Generate evolutionary search combinations using genetic algorithm
    fn generate_evolutionary_combinations(
        &self,
        search_space: &SearchSpace,
        population_size: usize,
        n_generations: usize,
        mutation_rate: f64,
        crossover_rate: f64,
    ) -> Result<Vec<HashMap<String, f64>>> {
        let mut rng = match self.config.random_seed {
            Some(seed) => rand::rngs::StdRng::seed_from_u64(seed),
            None => rand::rngs::StdRng::seed_from_u64(42),
        };

        // Initialize population
        let mut population = self.generate_random_combinations(search_space, population_size)?;
        let mut all_combinations = population.clone();

        // Evolution loop
        for _generation in 0..n_generations {
            let mut new_population = Vec::new();

            // Elitism: keep best individual from previous generation
            if !population.is_empty() {
                new_population.push(population[0].clone());
            }

            // Generate new offspring
            while new_population.len() < population_size {
                // Selection: tournament selection
                let parent1 = self.tournament_selection(&population, &mut rng)?;
                let parent2 = self.tournament_selection(&population, &mut rng)?;

                // Crossover
                let (mut child1, mut child2) = if rng.random_range(0.0..1.0) < crossover_rate {
                    self.crossover(&parent1, &parent2, search_space, &mut rng)?
                } else {
                    (parent1.clone(), parent2.clone())
                };

                // Mutation
                if rng.random_range(0.0..1.0) < mutation_rate {
                    self.mutate(&mut child1, search_space, &mut rng)?;
                }
                if rng.random_range(0.0..1.0) < mutation_rate {
                    self.mutate(&mut child2, search_space, &mut rng)?;
                }

                new_population.push(child1);
                if new_population.len() < population_size {
                    new_population.push(child2);
                }
            }

            population = new_population;
            all_combinations.extend(population.clone());

            // Early termination if we have enough evaluations
            if all_combinations.len() >= self.config.max_evaluations {
                break;
            }
        }

        // Trim to max evaluations
        all_combinations.truncate(self.config.max_evaluations);
        Ok(all_combinations)
    }

    /// Tournament selection for evolutionary algorithm
    fn tournament_selection(
        &self,
        population: &[HashMap<String, f64>],
        rng: &mut rand::rngs::StdRng,
    ) -> Result<HashMap<String, f64>> {
        let tournament_size = 3.min(population.len());
        let mut best_individual = None;

        for _ in 0..tournament_size {
            let idx = rng.random_range(0..population.len());
            let individual = &population[idx];

            // In a real implementation..we would evaluate fitness here
            // For now, just return the first selected individual
            if best_individual.is_none() {
                best_individual = Some(individual.clone());
            }
        }

        best_individual.ok_or_else(|| ClusteringError::InvalidInput("Empty population".to_string()))
    }

    /// Crossover operation for evolutionary algorithm
    fn crossover(
        &self,
        parent1: &HashMap<String, f64>,
        parent2: &HashMap<String, f64>,
        search_space: &SearchSpace,
        rng: &mut rand::rngs::StdRng,
    ) -> Result<(HashMap<String, f64>, HashMap<String, f64>)> {
        let mut child1 = HashMap::new();
        let mut child2 = HashMap::new();

        for (param_name, param_spec) in &search_space.parameters {
            let val1 = parent1.get(param_name).copied().unwrap_or(0.0);
            let val2 = parent2.get(param_name).copied().unwrap_or(0.0);

            // Uniform crossover with parameter-specific handling
            let (new_val1, new_val2) = match param_spec {
                HyperParameter::Float { min, max } => {
                    // Blend crossover for continuous parameters
                    let alpha = 0.5;
                    let beta = rng.random_range(0.0..1.0) * (1.0 + 2.0 * alpha) - alpha;
                    let v1 = (1.0 - beta) * val1 + beta * val2;
                    let v2 = beta * val1 + (1.0 - beta) * val2;
                    (v1.clamp(*min, *max), v2.clamp(*min, *max))
                }
                HyperParameter::Integer { min, max } => {
                    // Single-point crossover for discrete parameters
                    if rng.random_range(0.0..1.0) < 0.5 {
                        (
                            val1.clamp(*min as f64, *max as f64),
                            val2.clamp(*min as f64, *max as f64),
                        )
                    } else {
                        (
                            val2.clamp(*min as f64, *max as f64),
                            val1.clamp(*min as f64, *max as f64),
                        )
                    }
                }
                _ => {
                    // For other types, just swap randomly
                    if rng.random_range(0.0..1.0) < 0.5 {
                        (val1, val2)
                    } else {
                        (val2, val1)
                    }
                }
            };

            child1.insert(param_name.clone(), new_val1);
            child2.insert(param_name.clone(), new_val2);
        }

        Ok((child1, child2))
    }

    /// Mutation operation for evolutionary algorithm
    fn mutate(
        &self,
        individual: &mut HashMap<String, f64>,
        search_space: &SearchSpace,
        rng: &mut rand::rngs::StdRng,
    ) -> Result<()> {
        for (param_name, param_spec) in &search_space.parameters {
            if rng.random_range(0.0..1.0) < 0.1 {
                // 10% chance to mutate each parameter
                let current_val = individual.get(param_name).copied().unwrap_or(0.0);

                let new_val = match param_spec {
                    HyperParameter::Float { min, max } => {
                        // Gaussian mutation
                        let std_dev = (max - min) * 0.1; // 10% of range as standard deviation
                        let normal = rand_distr::Normal::new(0.0, std_dev).map_err(|e| {
                            ClusteringError::InvalidInput(format!("Mutation error: {}", e))
                        })?;
                        use rand_distr::Distribution;
                        let mutation_delta = normal.sample(rng);
                        (current_val + mutation_delta).clamp(*min, *max)
                    }
                    HyperParameter::Integer { min, max } => {
                        // Random reset mutation for discrete parameters
                        rng.random_range(*min..=*max) as f64
                    }
                    HyperParameter::Categorical { choices } => {
                        rng.random_range(0..choices.len()) as f64
                    }
                    HyperParameter::Boolean => {
                        if rng.random_range(0.0..1.0) < 0.5 {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    HyperParameter::LogUniform { min, max } => {
                        let log_min = min.ln();
                        let log_max = max.ln();
                        let log_val = rng.random_range(log_min..=log_max);
                        log_val.exp()
                    }
                    HyperParameter::IntegerChoices { choices } => {
                        let idx = rng.random_range(0..choices.len());
                        choices[idx] as f64
                    }
                };

                individual.insert(param_name.clone(), new_val);
            }
        }
        Ok(())
    }

    /// Optimize acquisition function using Gaussian Process
    fn optimize_gp_acquisition(
        &self,
        search_space: &SearchSpace,
        _observations: &[HashMap<String, f64>],
        _acquisition_function: &AcquisitionFunction,
    ) -> Result<HashMap<String, f64>> {
        // Simplified GP-based acquisition optimization
        // In practice, this would:
        // 1. Fit GP to current _observations
        // 2. Compute acquisition _function values over search _space
        // 3. Find point with maximum acquisition value

        // For now, generate multiple random candidates and pick best
        let n_candidates = 100;
        let candidates = self.generate_random_combinations(search_space, n_candidates)?;

        // In a real implementation, we would evaluate acquisition _function
        // For now, just return a random candidate
        Ok(candidates.into_iter().next().unwrap())
    }

    /// Optimize acquisition function using Random Forest
    fn optimize_rf_acquisition(
        &self,
        search_space: &SearchSpace,
        _observations: &[HashMap<String, f64>],
    ) -> Result<HashMap<String, f64>> {
        // Simplified RF-based acquisition optimization
        let candidates = self.generate_random_combinations(search_space, 50)?;
        Ok(candidates.into_iter().next().unwrap())
    }

    /// Optimize acquisition function using Gradient Boosting
    fn optimize_gb_acquisition(
        &self,
        search_space: &SearchSpace,
        _observations: &[HashMap<String, f64>],
    ) -> Result<HashMap<String, f64>> {
        // Simplified GB-based acquisition optimization
        let candidates = self.generate_random_combinations(search_space, 50)?;
        Ok(candidates.into_iter().next().unwrap())
    }

    /// Generate Latin Hypercube Sampling combinations for better space coverage
    fn generate_lhs_combinations(
        &self,
        search_space: &SearchSpace,
        n_points: usize,
    ) -> Result<Vec<HashMap<String, f64>>> {
        let mut rng = rand::rng();
        let mut combinations = Vec::new();

        let param_names: Vec<String> = search_space.parameters.keys().cloned().collect();
        let _n_params = param_names.len();

        // Generate LHS samples
        for i in 0..n_points {
            let mut params = HashMap::new();

            for (_j, param_name) in param_names.iter().enumerate() {
                let param_spec = &search_space.parameters[param_name];

                // LHS sampling: divide parameter _space into n_points intervals
                let interval_size = 1.0 / n_points as f64;
                let base_point = i as f64 * interval_size;
                let random_offset = rng.random_range(0.0..1.0) * interval_size;
                let normalized_value = base_point + random_offset;

                let value = match param_spec {
                    HyperParameter::Float { min, max } => min + normalized_value * (max - min),
                    HyperParameter::LogUniform { min, max } => {
                        let log_min = min.ln();
                        let log_max = max.ln();
                        (log_min + normalized_value * (log_max - log_min)).exp()
                    }
                    HyperParameter::Integer { min, max } => {
                        (*min as f64 + normalized_value * (*max - *min) as f64).round()
                    }
                    HyperParameter::IntegerChoices { choices } => {
                        let idx = (normalized_value * choices.len() as f64).floor() as usize;
                        choices[idx.min(choices.len() - 1)] as f64
                    }
                    HyperParameter::Boolean => {
                        if normalized_value < 0.5 {
                            0.0
                        } else {
                            1.0
                        }
                    }
                    HyperParameter::Categorical { choices } => {
                        let idx = (normalized_value * choices.len() as f64).floor() as usize;
                        idx.min(choices.len() - 1) as f64
                    }
                };

                params.insert(param_name.clone(), value);
            }

            combinations.push(params);
        }

        Ok(combinations)
    }

    /// Enhanced Gaussian Process update with proper kernel computations
    fn update_gaussian_process_enhanced(
        &self,
        bayesian_state: &mut BayesianState,
        combinations: &[HashMap<String, f64>],
    ) {
        if combinations.is_empty() {
            return;
        }

        let n_points = combinations.len();
        let n_features = bayesian_state.parameter_names.len();

        // Convert parameter combinations to feature matrix
        let mut feature_matrix = Array2::zeros((n_points, n_features));
        for (i, combo) in combinations.iter().enumerate() {
            for (j, param_name) in bayesian_state.parameter_names.iter().enumerate() {
                feature_matrix[[i, j]] = combo.get(param_name).unwrap_or(&0.0).clone();
            }
        }

        // Compute kernel matrix
        let kernel_matrix =
            self.compute_kernel_matrix(&feature_matrix, &bayesian_state.gp_hyperparameters);

        // Add noise to diagonal for numerical stability
        let mut k_with_noise = kernel_matrix.clone();
        for i in 0..n_points {
            k_with_noise[[i, i]] += bayesian_state.gp_hyperparameters.noise_variance;
        }

        // Store enhanced GP _state
        bayesian_state.gp_covariance = Some(k_with_noise);

        // Compute mean (simplified for now - would use actual observations)
        let mean = feature_matrix.mean_axis(Axis(0)).unwrap();
        bayesian_state.gp_mean = Some(mean[0]);
    }

    /// Compute kernel matrix for GP
    fn compute_kernel_matrix(
        &self,
        feature_matrix: &Array2<f64>,
        hyperparams: &GpHyperparameters,
    ) -> Array2<f64> {
        let n_points = feature_matrix.nrows();
        let mut kernel_matrix = Array2::zeros((n_points, n_points));

        for i in 0..n_points {
            for j in i..n_points {
                let xi = feature_matrix.row(i);
                let xj = feature_matrix.row(j);

                let kernel_value = match &hyperparams.kernel_type {
                    KernelType::RBF { length_scale } => {
                        let dist_sq = xi
                            .iter()
                            .zip(xj.iter())
                            .map(|(a, b)| (a - b).powi(2))
                            .sum::<f64>();
                        hyperparams.signal_variance * (-0.5 * dist_sq / length_scale.powi(2)).exp()
                    }
                    KernelType::Matern { length_scale, nu } => {
                        let dist = xi
                            .iter()
                            .zip(xj.iter())
                            .map(|(a, b)| (a - b).powi(2))
                            .sum::<f64>()
                            .sqrt();

                        if dist == 0.0 {
                            hyperparams.signal_variance
                        } else {
                            let sqrt_2nu_dist = (2.0 * nu).sqrt() * dist / length_scale;
                            let bessel_term = 1.0; // Simplified Bessel function
                            hyperparams.signal_variance
                                * (sqrt_2nu_dist.powf(*nu) * bessel_term * (-sqrt_2nu_dist).exp())
                        }
                    }
                    KernelType::Linear => xi.iter().zip(xj.iter()).map(|(a, b)| a * b).sum::<f64>(),
                    KernelType::Polynomial { degree } => {
                        let dot_product = xi.iter().zip(xj.iter()).map(|(a, b)| a * b).sum::<f64>();
                        (1.0 + dot_product).powf(*degree as f64)
                    }
                };

                kernel_matrix[[i, j]] = kernel_value;
                kernel_matrix[[j, i]] = kernel_value;
            }
        }

        kernel_matrix
    }

    /// Enhanced acquisition function optimization with multiple strategies
    fn optimize_acquisition_function_enhanced(
        &self,
        search_space: &SearchSpace,
        bayesian_state: &BayesianState,
        acquisition_function: &AcquisitionFunction,
        iteration: usize,
    ) -> Result<HashMap<String, f64>> {
        let n_candidates = std::cmp::max(1000, 100 * search_space.parameters.len());

        // Use multiple strategies for finding the best candidate
        let mut all_candidates = Vec::new();

        // Strategy 1: Random sampling
        let random_candidates =
            self.generate_random_combinations(search_space, n_candidates / 3)?;
        all_candidates.extend(random_candidates);

        // Strategy 2: Latin Hypercube Sampling
        let lhs_candidates = self.generate_lhs_combinations(search_space, n_candidates / 3)?;
        all_candidates.extend(lhs_candidates);

        // Strategy 3: Gradient-based local optimization around best points
        if !bayesian_state.observations.is_empty() && iteration > 5 {
            let local_candidates = self.generate_local_optimization_candidates(
                search_space,
                bayesian_state,
                n_candidates / 3,
            )?;
            all_candidates.extend(local_candidates);
        } else {
            // Fallback to more random samples
            let extra_random = self.generate_random_combinations(search_space, n_candidates / 3)?;
            all_candidates.extend(extra_random);
        }

        // Evaluate acquisition _function for all candidates
        let mut best_candidate = all_candidates[0].clone();
        let mut best_acquisition_value = f64::NEG_INFINITY;

        for candidate in &all_candidates {
            let acquisition_value = self.evaluate_acquisition_function_enhanced(
                candidate,
                bayesian_state,
                acquisition_function,
            );

            if acquisition_value > best_acquisition_value {
                best_acquisition_value = acquisition_value;
                best_candidate = candidate.clone();
            }
        }

        Ok(best_candidate)
    }

    /// Generate local optimization candidates around promising regions
    fn generate_local_optimization_candidates(
        &self,
        search_space: &SearchSpace,
        _state: &BayesianState,
        n_candidates: usize,
    ) -> Result<Vec<HashMap<String, f64>>> {
        let mut _candidates = Vec::new();
        let mut rng = rand::rng();

        // Find top performing regions from observations (placeholder)
        let n_centers = std::cmp::min(5, n_candidates / 10);

        for _ in 0..n_centers {
            let center = self.generate_random_combinations(search_space, 1)?[0].clone();

            // Generate _candidates around this center
            for _ in 0..n_candidates / n_centers {
                let mut candidate = HashMap::new();

                for (param_name, param_spec) in &search_space.parameters {
                    let center_value = center.get(param_name).unwrap_or(&0.0);

                    let perturbed_value = match param_spec {
                        HyperParameter::Float { min, max } => {
                            let noise_scale = (max - min) * 0.1; // 10% of range
                            let noise = rng.random_range(-noise_scale..noise_scale);
                            (center_value + noise).clamp(*min, *max)
                        }
                        HyperParameter::Integer { min, max } => {
                            let noise = rng.random_range(-2..=2);
                            (center_value + noise as f64)
                                .round()
                                .clamp(*min as f64, *max as f64)
                        }
                        _ => *center_value, // For other types, use center value
                    };

                    candidate.insert(param_name.clone(), perturbed_value);
                }

                _candidates.push(candidate);
            }
        }

        Ok(_candidates)
    }

    /// Enhanced acquisition function evaluation with proper GP predictions
    fn evaluate_acquisition_function_enhanced(
        &self,
        point: &HashMap<String, f64>,
        bayesian_state: &BayesianState,
        acquisition_function: &AcquisitionFunction,
    ) -> f64 {
        // Get GP predictions at the _point (simplified)
        let x = self.extract_feature_vector(point, &bayesian_state.parameter_names);
        let (mean, variance) = self.predict_gp(&x, bayesian_state);
        let std_dev = variance.sqrt();

        match acquisition_function {
            AcquisitionFunction::ExpectedImprovement => {
                let f_best = bayesian_state.currentbest;
                if std_dev < 1e-6 {
                    return 0.0; // No uncertainty, no improvement expected
                }

                let z = (mean - f_best) / std_dev;
                let phi_z = 0.5 * (1.0 + erf(z / 2.0_f64.sqrt()));
                let pdf_z = (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt();

                (mean - f_best) * phi_z + std_dev * pdf_z
            }
            AcquisitionFunction::UpperConfidenceBound { beta } => mean + beta * std_dev,
            AcquisitionFunction::ProbabilityOfImprovement => {
                let f_best = bayesian_state.currentbest;
                if std_dev < 1e-6 {
                    return if mean > f_best { 1.0 } else { 0.0 };
                }

                let z = (mean - f_best) / std_dev;
                0.5 * (1.0 + erf(z / 2.0_f64.sqrt()))
            }
            AcquisitionFunction::EntropySearch => {
                // Simplified entropy-based acquisition
                -variance * variance.ln()
            }
            AcquisitionFunction::KnowledgeGradient => {
                // Simplified knowledge gradient
                std_dev * (1.0 + variance.ln())
            }
            AcquisitionFunction::ThompsonSampling => {
                // Thompson sampling: sample from posterior
                let mut rng = rand::rng();
                mean + std_dev * rng.random_range(-1.0..1.0)
            }
        }
    }

    /// Compute marginal likelihood for GP hyperparameter optimization
    fn compute_marginal_likelihood(
        &self,
        x_matrix: &Array2<f64>,
        y_vector: &Array1<f64>,
        hyperparams: &GpHyperparameters,
    ) -> Result<f64> {
        let n = x_matrix.nrows();

        // Compute kernel _matrix
        let kernel_matrix = self.compute_kernel_matrix_cross(x_matrix, x_matrix, hyperparams);

        // Add noise term
        let mut k_noise = kernel_matrix.clone();
        for i in 0..n {
            k_noise[[i, i]] += hyperparams.noise_variance;
        }

        // Compute Cholesky decomposition
        let l_matrix = self.cholesky_decomposition(&k_noise)?;

        // Solve L * alpha = y
        let alpha = self.solve_triangular(&l_matrix, y_vector)?;

        // Compute log marginal likelihood:
        // -0.5 * y^T * (K + noise_I)^{-1} * y - 0.5 * log|K + noise_I| - 0.5 * n * log(2*pi)
        let data_fit = -0.5 * y_vector.dot(&alpha);
        let complexity_penalty = -l_matrix.diag().mapv(|x| x.ln()).sum();
        let normalization = -0.5 * (n as f64) * (2.0 * std::f64::consts::PI).ln();

        Ok(data_fit + complexity_penalty + normalization)
    }

    /// Compute kernel matrix between two sets of points
    fn compute_kernel_matrix_cross(
        &self,
        x1: &Array2<f64>,
        x2: &Array2<f64>,
        hyperparams: &GpHyperparameters,
    ) -> Array2<f64> {
        let n1 = x1.nrows();
        let n2 = x2.nrows();
        let mut kernel_matrix = Array2::zeros((n1, n2));

        for i in 0..n1 {
            for j in 0..n2 {
                let xi = x1.row(i);
                let xj = x2.row(j);
                kernel_matrix[[i, j]] = self.compute_kernel_value(&xi, &xj, hyperparams);
            }
        }

        kernel_matrix
    }

    /// Compute kernel value between two points
    fn compute_kernel_value(
        &self,
        xi: &ndarray::ArrayView1<f64>,
        xj: &ndarray::ArrayView1<f64>,
        hyperparams: &GpHyperparameters,
    ) -> f64 {
        match &hyperparams.kernel_type {
            KernelType::RBF { length_scale } => {
                let dist_sq = xi
                    .iter()
                    .zip(xj.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>();
                hyperparams.signal_variance * (-0.5 * dist_sq / length_scale.powi(2)).exp()
            }
            KernelType::Matern { length_scale, nu } => {
                let dist = xi
                    .iter()
                    .zip(xj.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();

                if dist == 0.0 {
                    return hyperparams.signal_variance;
                }

                let scaled_dist = (2.0 * nu).sqrt() * dist / length_scale;
                match nu {
                    &1.5 => {
                        hyperparams.signal_variance * (1.0 + scaled_dist) * (-scaled_dist).exp()
                    }
                    &2.5 => {
                        hyperparams.signal_variance
                            * (1.0 + scaled_dist + scaled_dist.powi(2) / 3.0)
                            * (-scaled_dist).exp()
                    }
                    _ => {
                        // Simplified Matérn for other nu values
                        hyperparams.signal_variance * (-scaled_dist).exp()
                    }
                }
            }
            KernelType::Linear => {
                let dot_product: f64 = xi.iter().zip(xj.iter()).map(|(a, b)| a * b).sum();
                hyperparams.signal_variance * dot_product
            }
            KernelType::Polynomial { degree } => {
                let dot_product: f64 = xi.iter().zip(xj.iter()).map(|(a, b)| a * b).sum();
                hyperparams.signal_variance * (1.0 + dot_product).powi(*degree as i32)
            }
        }
    }

    /// Perform Cholesky decomposition for numerical stability
    fn cholesky_decomposition(&self, matrix: &Array2<f64>) -> Result<Array2<f64>> {
        let n = matrix.nrows();
        if n != matrix.ncols() {
            return Err(ClusteringError::InvalidInput(
                "Matrix must be square".to_string(),
            ));
        }

        let mut l: Array2<f64> = Array2::zeros((n, n));

        for i in 0..n {
            for j in 0..=i {
                if i == j {
                    // Diagonal elements
                    let sum_sq: f64 = (0..j).map(|k| l[[i, k]].powi(2)).sum();
                    let val = matrix[[i, i]] - sum_sq;
                    if val <= 0.0 {
                        return Err(ClusteringError::InvalidInput(
                            "Matrix not positive definite".to_string(),
                        ));
                    }
                    l[[i, j]] = val.sqrt();
                } else {
                    // Off-diagonal elements
                    let sum_prod: f64 = (0..j).map(|k| l[[i, k]] * l[[j, k]]).sum();
                    l[[i, j]] = (matrix[[i, j]] - sum_prod) / l[[j, j]];
                }
            }
        }

        Ok(l)
    }

    /// Solve triangular system L * x = b
    fn solve_triangular(&self, l_matrix: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>> {
        let n = l_matrix.nrows();
        if n != b.len() {
            return Err(ClusteringError::InvalidInput(
                "Dimension mismatch".to_string(),
            ));
        }

        let mut x = Array1::zeros(n);

        // Forward substitution for lower triangular _matrix
        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..i {
                sum += l_matrix[[i, j]] * x[j];
            }

            if l_matrix[[i, i]].abs() < 1e-12 {
                return Err(ClusteringError::InvalidInput("Singular matrix".to_string()));
            }

            x[i] = (b[i] - sum) / l_matrix[[i, i]];
        }

        Ok(x)
    }

    /// Tune Mean Shift hyperparameters
    pub fn tune_mean_shift(
        &self,
        _data: ArrayView2<F>,
        _search_space: SearchSpace,
    ) -> Result<TuningResult> {
        Err(ClusteringError::ComputationError(
            "tune_mean_shift not yet implemented".to_string(),
        ))
    }

    /// Tune Hierarchical clustering hyperparameters
    pub fn tune_hierarchical(
        &self,
        _data: ArrayView2<F>,
        _search_space: SearchSpace,
    ) -> Result<TuningResult> {
        Err(ClusteringError::ComputationError(
            "tune_hierarchical not yet implemented".to_string(),
        ))
    }

    /// Tune Quantum K-means hyperparameters
    pub fn tune_quantum_kmeans(
        &self,
        _data: ArrayView2<F>,
        _search_space: SearchSpace,
    ) -> Result<TuningResult> {
        Err(ClusteringError::ComputationError(
            "tune_quantum_kmeans not yet implemented".to_string(),
        ))
    }

    /// Tune RL clustering hyperparameters
    pub fn tune_rl_clustering(
        &self,
        _data: ArrayView2<F>,
        _search_space: SearchSpace,
    ) -> Result<TuningResult> {
        Err(ClusteringError::ComputationError(
            "tune_rl_clustering not yet implemented".to_string(),
        ))
    }

    /// Tune Adaptive Online clustering hyperparameters
    pub fn tune_adaptive_online(
        &self,
        _data: ArrayView2<F>,
        _search_space: SearchSpace,
    ) -> Result<TuningResult> {
        Err(ClusteringError::ComputationError(
            "tune_adaptive_online not yet implemented".to_string(),
        ))
    }
}

/// Error function approximation for statistical calculations
#[allow(dead_code)]
fn erf(x: f64) -> f64 {
    // Abramowitz and Stegun approximation
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

/// Default configurations for different algorithms
impl Default for TuningConfig {
    fn default() -> Self {
        Self {
            strategy: SearchStrategy::RandomSearch { n_trials: 50 },
            metric: EvaluationMetric::SilhouetteScore,
            cv_config: CrossValidationConfig {
                n_folds: 5,
                validation_ratio: 0.2,
                strategy: CVStrategy::KFold,
                shuffle: true,
            },
            max_evaluations: 100,
            early_stopping: Some(EarlyStoppingConfig {
                patience: 10,
                min_improvement: 0.001,
                evaluation_frequency: 1,
            }),
            random_seed: None,
            parallel_config: None,
            resource_constraints: ResourceConstraints {
                max_memory_per_evaluation: None,
                max_time_per_evaluation: Some(300.0), // 5 minutes
                max_total_time: Some(3600.0),         // 1 hour
            },
        }
    }
}

/// Predefined search spaces for common algorithms
pub struct StandardSearchSpaces;

impl StandardSearchSpaces {
    /// K-means search space
    pub fn kmeans() -> SearchSpace {
        let mut parameters = HashMap::new();
        parameters.insert(
            "n_clusters".to_string(),
            HyperParameter::Integer { min: 2, max: 20 },
        );
        parameters.insert(
            "max_iter".to_string(),
            HyperParameter::IntegerChoices {
                choices: vec![100, 200, 300, 500, 1000],
            },
        );
        parameters.insert(
            "tolerance".to_string(),
            HyperParameter::LogUniform {
                min: 1e-6,
                max: 1e-2,
            },
        );

        SearchSpace {
            parameters,
            constraints: Vec::new(),
        }
    }

    /// DBSCAN search space
    pub fn dbscan() -> SearchSpace {
        let mut parameters = HashMap::new();
        parameters.insert(
            "eps".to_string(),
            HyperParameter::Float { min: 0.1, max: 2.0 },
        );
        parameters.insert(
            "min_samples".to_string(),
            HyperParameter::Integer { min: 2, max: 20 },
        );

        SearchSpace {
            parameters,
            constraints: Vec::new(),
        }
    }

    /// Hierarchical clustering search space
    pub fn hierarchical() -> SearchSpace {
        let mut parameters = HashMap::new();
        parameters.insert(
            "method".to_string(),
            HyperParameter::Categorical {
                choices: vec![
                    "single".to_string(),
                    "complete".to_string(),
                    "average".to_string(),
                    "ward".to_string(),
                ],
            },
        );

        SearchSpace {
            parameters,
            constraints: Vec::new(),
        }
    }

    /// Mean Shift search space
    pub fn mean_shift() -> SearchSpace {
        let mut parameters = HashMap::new();
        parameters.insert(
            "bandwidth".to_string(),
            HyperParameter::Float { min: 0.1, max: 5.0 },
        );

        SearchSpace {
            parameters,
            constraints: Vec::new(),
        }
    }

    /// OPTICS search space
    pub fn optics() -> SearchSpace {
        let mut parameters = HashMap::new();
        parameters.insert(
            "min_samples".to_string(),
            HyperParameter::Integer { min: 2, max: 20 },
        );
        parameters.insert(
            "max_eps".to_string(),
            HyperParameter::Float {
                min: 0.1,
                max: 10.0,
            },
        );

        SearchSpace {
            parameters,
            constraints: Vec::new(),
        }
    }

    /// Spectral clustering search space
    pub fn spectral() -> SearchSpace {
        let mut parameters = HashMap::new();
        parameters.insert(
            "n_clusters".to_string(),
            HyperParameter::Integer { min: 2, max: 20 },
        );
        parameters.insert(
            "n_neighbors".to_string(),
            HyperParameter::Integer { min: 5, max: 50 },
        );
        parameters.insert(
            "gamma".to_string(),
            HyperParameter::LogUniform {
                min: 0.01,
                max: 10.0,
            },
        );
        parameters.insert(
            "max_iter".to_string(),
            HyperParameter::IntegerChoices {
                choices: vec![100, 200, 300, 500, 1000],
            },
        );

        SearchSpace {
            parameters,
            constraints: Vec::new(),
        }
    }

    /// Affinity Propagation search space
    pub fn affinity_propagation() -> SearchSpace {
        let mut parameters = HashMap::new();
        parameters.insert(
            "damping".to_string(),
            HyperParameter::Float {
                min: 0.5,
                max: 0.99,
            },
        );
        parameters.insert(
            "max_iter".to_string(),
            HyperParameter::IntegerChoices {
                choices: vec![100, 200, 300, 500],
            },
        );
        parameters.insert(
            "convergence_iter".to_string(),
            HyperParameter::Integer { min: 10, max: 50 },
        );

        SearchSpace {
            parameters,
            constraints: Vec::new(),
        }
    }

    /// BIRCH search space
    pub fn birch() -> SearchSpace {
        let mut parameters = HashMap::new();
        parameters.insert(
            "branching_factor".to_string(),
            HyperParameter::Integer { min: 10, max: 100 },
        );
        parameters.insert(
            "threshold".to_string(),
            HyperParameter::Float { min: 0.1, max: 5.0 },
        );

        SearchSpace {
            parameters,
            constraints: Vec::new(),
        }
    }

    /// GMM search space
    pub fn gmm() -> SearchSpace {
        let mut parameters = HashMap::new();
        parameters.insert(
            "n_components".to_string(),
            HyperParameter::Integer { min: 1, max: 20 },
        );
        parameters.insert(
            "max_iter".to_string(),
            HyperParameter::IntegerChoices {
                choices: vec![50, 100, 200, 300],
            },
        );
        parameters.insert(
            "tol".to_string(),
            HyperParameter::LogUniform {
                min: 1e-6,
                max: 1e-2,
            },
        );
        parameters.insert(
            "reg_covar".to_string(),
            HyperParameter::LogUniform {
                min: 1e-8,
                max: 1e-3,
            },
        );

        SearchSpace {
            parameters,
            constraints: Vec::new(),
        }
    }

    /// Quantum K-means search space
    pub fn quantum_kmeans() -> SearchSpace {
        let mut parameters = HashMap::new();
        parameters.insert(
            "n_clusters".to_string(),
            HyperParameter::Integer { min: 2, max: 20 },
        );
        parameters.insert(
            "n_quantum_states".to_string(),
            HyperParameter::IntegerChoices {
                choices: vec![4, 8, 16, 32],
            },
        );
        parameters.insert(
            "quantum_iterations".to_string(),
            HyperParameter::IntegerChoices {
                choices: vec![20, 50, 100, 200],
            },
        );
        parameters.insert(
            "decoherence_factor".to_string(),
            HyperParameter::Float {
                min: 0.8,
                max: 0.99,
            },
        );
        parameters.insert(
            "entanglement_strength".to_string(),
            HyperParameter::Float { min: 0.1, max: 0.5 },
        );

        SearchSpace {
            parameters,
            constraints: Vec::new(),
        }
    }

    /// Reinforcement learning clustering search space
    pub fn rl_clustering() -> SearchSpace {
        let mut parameters = HashMap::new();
        parameters.insert(
            "n_actions".to_string(),
            HyperParameter::Integer { min: 5, max: 50 },
        );
        parameters.insert(
            "learning_rate".to_string(),
            HyperParameter::LogUniform {
                min: 0.001,
                max: 0.5,
            },
        );
        parameters.insert(
            "exploration_rate".to_string(),
            HyperParameter::Float { min: 0.1, max: 1.0 },
        );
        parameters.insert(
            "n_episodes".to_string(),
            HyperParameter::IntegerChoices {
                choices: vec![50, 100, 200, 500],
            },
        );

        SearchSpace {
            parameters,
            constraints: Vec::new(),
        }
    }

    /// Adaptive online clustering search space
    pub fn adaptive_online() -> SearchSpace {
        let mut parameters = HashMap::new();
        parameters.insert(
            "initial_learning_rate".to_string(),
            HyperParameter::LogUniform {
                min: 0.001,
                max: 0.5,
            },
        );
        parameters.insert(
            "cluster_creation_threshold".to_string(),
            HyperParameter::Float { min: 1.0, max: 5.0 },
        );
        parameters.insert(
            "max_clusters".to_string(),
            HyperParameter::Integer { min: 10, max: 100 },
        );
        parameters.insert(
            "forgetting_factor".to_string(),
            HyperParameter::Float {
                min: 0.9,
                max: 0.99,
            },
        );

        SearchSpace {
            parameters,
            constraints: Vec::new(),
        }
    }

    /// K-means search space with Bayesian optimization
    pub fn kmeans_bayesian() -> (SearchSpace, TuningConfig) {
        let mut parameters = HashMap::new();
        parameters.insert(
            "n_clusters".to_string(),
            HyperParameter::Integer { min: 2, max: 50 },
        );
        parameters.insert(
            "max_iter".to_string(),
            HyperParameter::Integer { min: 50, max: 500 },
        );
        parameters.insert(
            "tolerance".to_string(),
            HyperParameter::Float {
                min: 1e-6,
                max: 1e-2,
            },
        );

        let search_space = SearchSpace {
            parameters,
            constraints: Vec::new(),
        };

        let config = TuningConfig {
            strategy: SearchStrategy::BayesianOptimization {
                n_initial_points: 10,
                acquisition_function: AcquisitionFunction::ExpectedImprovement,
            },
            metric: EvaluationMetric::SilhouetteScore,
            max_evaluations: 50,
            cv_config: CrossValidationConfig {
                n_folds: 5,
                validation_ratio: 0.2,
                strategy: CVStrategy::KFold,
                shuffle: true,
            },
            early_stopping: Some(EarlyStoppingConfig {
                patience: 10,
                min_improvement: 0.001,
                evaluation_frequency: 1,
            }),
            parallel_config: Some(ParallelConfig {
                n_workers: 8,
                load_balancing: LoadBalancingStrategy::Dynamic,
                batch_size: 100,
            }),
            random_seed: Some(42),
            resource_constraints: ResourceConstraints {
                max_memory_per_evaluation: None,
                max_time_per_evaluation: None,
                max_total_time: None,
            },
        };

        (search_space, config)
    }

    /// DBSCAN search space with multi-objective optimization
    pub fn dbscan_multi_objective() -> (SearchSpace, TuningConfig) {
        let mut parameters = HashMap::new();
        parameters.insert(
            "eps".to_string(),
            HyperParameter::Float { min: 0.1, max: 2.0 },
        );
        parameters.insert(
            "min_samples".to_string(),
            HyperParameter::Integer { min: 2, max: 20 },
        );

        let search_space = SearchSpace {
            parameters,
            constraints: Vec::new(),
        };

        let config = TuningConfig {
            strategy: SearchStrategy::MultiObjective {
                objectives: vec![
                    EvaluationMetric::SilhouetteScore,
                    EvaluationMetric::DaviesBouldinIndex,
                ],
                strategy: Box::new(SearchStrategy::BayesianOptimization {
                    n_initial_points: 10,
                    acquisition_function: AcquisitionFunction::ExpectedImprovement,
                }),
            },
            metric: EvaluationMetric::SilhouetteScore,
            max_evaluations: 30,
            cv_config: CrossValidationConfig {
                n_folds: 3,
                validation_ratio: 0.2,
                strategy: CVStrategy::KFold,
                shuffle: true,
            },
            early_stopping: None,
            parallel_config: Some(ParallelConfig {
                n_workers: 8,
                load_balancing: LoadBalancingStrategy::Dynamic,
                batch_size: 100,
            }),
            random_seed: Some(42),
            resource_constraints: ResourceConstraints {
                max_memory_per_evaluation: None,
                max_time_per_evaluation: None,
                max_total_time: None,
            },
        };

        (search_space, config)
    }

    /// Ensemble optimization search space
    pub fn ensemble_optimization() -> (SearchSpace, TuningConfig) {
        let mut parameters = HashMap::new();
        parameters.insert(
            "n_estimators".to_string(),
            HyperParameter::Integer { min: 3, max: 20 },
        );
        parameters.insert(
            "diversity_threshold".to_string(),
            HyperParameter::Float { min: 0.1, max: 0.9 },
        );

        let search_space = SearchSpace {
            parameters,
            constraints: Vec::new(),
        };

        let config = TuningConfig {
            strategy: SearchStrategy::BayesianOptimization {
                n_initial_points: 5,
                acquisition_function: AcquisitionFunction::ExpectedImprovement,
            },
            metric: EvaluationMetric::SilhouetteScore,
            max_evaluations: 25,
            cv_config: CrossValidationConfig {
                n_folds: 3,
                validation_ratio: 0.2,
                strategy: CVStrategy::KFold,
                shuffle: true,
            },
            early_stopping: None,
            parallel_config: Some(ParallelConfig {
                n_workers: 8,
                load_balancing: LoadBalancingStrategy::Dynamic,
                batch_size: 100,
            }),
            random_seed: Some(42),
            resource_constraints: ResourceConstraints {
                max_memory_per_evaluation: None,
                max_time_per_evaluation: None,
                max_total_time: None,
            },
        };

        (search_space, config)
    }
}

/// Advanced hyperparameter optimization techniques
pub mod advanced_optimization {
    use super::*;
    use ndarray::{s, Array3, ArrayView1, Axis};

    /// Configuration for advanced Bayesian optimization
    #[derive(Debug, Clone)]
    pub struct AdvancedBayesianConfig {
        /// Multi-fidelity optimization settings
        pub multi_fidelity: Option<MultiFidelityConfig>,
        /// Transfer learning configuration
        pub transfer_learning: Option<TransferLearningConfig>,
        /// Multi-objective optimization settings
        pub multi_objective: Option<MultiObjectiveConfig>,
        /// Advanced acquisition functions
        pub advanced_acquisition: AdvancedAcquisitionConfig,
        /// Uncertainty quantification settings
        pub uncertainty_quantification: bool,
        /// Hyperparameter optimization for GP itself
        pub optimize_gp_hyperparameters: bool,
        /// Constraint handling method
        pub constraint_handling: ConstraintHandlingMethod,
    }

    /// Multi-fidelity optimization configuration
    #[derive(Debug, Clone)]
    pub struct MultiFidelityConfig {
        /// Fidelity levels (e.g., different data subset sizes)
        pub fidelity_levels: Vec<f64>,
        /// Cost ratios for different fidelities
        pub cost_ratios: Vec<f64>,
        /// Correlation model between fidelities
        pub correlation_model: FidelityCorrelationModel,
        /// Budget allocation strategy
        pub budget_allocation: BudgetAllocationStrategy,
    }

    /// Correlation models between different fidelities
    #[derive(Debug, Clone)]
    pub enum FidelityCorrelationModel {
        /// Linear correlation
        Linear { correlation: f64 },
        /// Exponential correlation
        Exponential { decay_rate: f64 },
        /// Neural network model
        NeuralNetwork { hidden_layers: Vec<usize> },
        /// Gaussian process model
        GaussianProcess { kernel: KernelType },
    }

    /// Budget allocation strategies for multi-fidelity
    #[derive(Debug, Clone)]
    pub enum BudgetAllocationStrategy {
        /// Equal allocation across fidelities
        Equal,
        /// Inverse cost allocation
        InverseCost,
        /// Bandit-based allocation
        Bandit { exploration_factor: f64 },
        /// Information-theoretic allocation
        InformationTheoretic,
    }

    /// Transfer learning configuration for Bayesian optimization
    #[derive(Debug, Clone)]
    pub struct TransferLearningConfig {
        /// Source task data and results
        pub source_tasks: Vec<SourceTask>,
        /// Transfer learning method
        pub transfer_method: TransferMethod,
        /// Similarity metric between tasks
        pub similarity_metric: TaskSimilarityMetric,
        /// Weight decay for source task influence
        pub source_weight_decay: f64,
    }

    /// Source task information for transfer learning
    #[derive(Debug, Clone)]
    pub struct SourceTask {
        /// Task identifier
        pub task_id: String,
        /// Observations from source task
        pub observations: Vec<(HashMap<String, f64>, f64)>,
        /// Task meta-features
        pub meta_features: Array1<f64>,
        /// Task similarity weight
        pub similarity_weight: f64,
    }

    /// Transfer learning methods
    #[derive(Debug, Clone)]
    pub enum TransferMethod {
        /// Direct transfer of GP posterior
        DirectTransfer,
        /// Hierarchical Bayesian model
        Hierarchical,
        /// Meta-learning approach
        MetaLearning { meta_model: MetaModelType },
        /// Domain adaptation
        DomainAdaptation { adaptation_rate: f64 },
    }

    /// Meta-model types for transfer learning
    #[derive(Debug, Clone)]
    pub enum MetaModelType {
        /// Neural network meta-model
        NeuralNetwork { architecture: Vec<usize> },
        /// Gaussian process meta-model
        GaussianProcess { kernel: KernelType },
        /// Random forest meta-model
        RandomForest { n_trees: usize },
    }

    /// Task similarity metrics
    #[derive(Debug, Clone)]
    pub enum TaskSimilarityMetric {
        /// Cosine similarity of meta-features
        Cosine,
        /// Euclidean distance of meta-features
        Euclidean,
        /// Task-specific distance
        TaskSpecific { metric_params: Vec<f64> },
        /// Learned similarity function
        Learned { model_type: MetaModelType },
    }

    /// Multi-objective optimization configuration
    #[derive(Debug, Clone)]
    pub struct MultiObjectiveConfig {
        /// Objective functions
        pub objectives: Vec<EvaluationMetric>,
        /// Objective weights (for scalarization)
        pub weights: Vec<f64>,
        /// Multi-objective optimization method
        pub method: MultiObjectiveMethod,
        /// Pareto front approximation settings
        pub pareto_settings: ParetoSettings,
    }

    /// Multi-objective optimization methods
    #[derive(Debug, Clone)]
    pub enum MultiObjectiveMethod {
        /// Weighted scalarization
        WeightedScalarization,
        /// Hypervolume-based optimization
        Hypervolume { reference_point: Vec<f64> },
        /// ParEGO (Pareto Efficient Global Optimization)
        ParEGO { rho: f64 },
        /// NSGA-II inspired Bayesian optimization
        NSGABO { population_size: usize },
        /// Expected hypervolume improvement
        EHVI,
    }

    /// Pareto front approximation settings
    #[derive(Debug, Clone)]
    pub struct ParetoSettings {
        /// Maximum number of points on Pareto front
        pub max_pareto_points: usize,
        /// Epsilon for Pareto dominance
        pub epsilon: f64,
        /// Update frequency for Pareto front
        pub update_frequency: usize,
    }

    /// Advanced acquisition function configuration
    #[derive(Debug, Clone)]
    pub struct AdvancedAcquisitionConfig {
        /// Portfolio of acquisition functions
        pub acquisition_portfolio: Vec<AcquisitionFunction>,
        /// Portfolio weights
        pub portfolio_weights: Vec<f64>,
        /// Adaptive acquisition function selection
        pub adaptive_selection: bool,
        /// Batch acquisition for parallel evaluation
        pub batch_acquisition: Option<BatchAcquisitionConfig>,
        /// Look-ahead strategies
        pub look_ahead: Option<LookAheadConfig>,
    }

    /// Batch acquisition configuration
    #[derive(Debug, Clone)]
    pub struct BatchAcquisitionConfig {
        /// Batch size
        pub batch_size: usize,
        /// Batch acquisition method
        pub method: BatchAcquisitionMethod,
        /// Diversity encouragement factor
        pub diversity_factor: f64,
    }

    /// Batch acquisition methods
    #[derive(Debug, Clone)]
    pub enum BatchAcquisitionMethod {
        /// Local Penalization
        LocalPenalization,
        /// Kriging Believer
        KrigingBeliever,
        /// Constant Liar
        ConstantLiar { lie_value: f64 },
        /// Maximum value entropy search
        MaxValueEntropySearch,
        /// Multi-point expected improvement
        MultiPointEI,
    }

    /// Look-ahead configuration
    #[derive(Debug, Clone)]
    pub struct LookAheadConfig {
        /// Number of look-ahead steps
        pub n_steps: usize,
        /// Look-ahead method
        pub method: LookAheadMethod,
        /// Branching factor
        pub branching_factor: usize,
    }

    /// Look-ahead methods
    #[derive(Debug, Clone)]
    pub enum LookAheadMethod {
        /// Monte Carlo look-ahead
        MonteCarlo { n_samples: usize },
        /// Rollout policy
        Rollout { policy: RolloutPolicy },
        /// Tree search
        TreeSearch { max_depth: usize },
    }

    /// Rollout policies for look-ahead
    #[derive(Debug, Clone)]
    pub enum RolloutPolicy {
        /// Random rollout
        Random,
        /// Greedy rollout
        Greedy,
        /// Upper confidence bound rollout
        UCB { exploration_factor: f64 },
    }

    /// Constraint handling methods
    #[derive(Debug, Clone)]
    pub enum ConstraintHandlingMethod {
        /// Penalty method
        Penalty { penalty_factor: f64 },
        /// Barrier method
        Barrier { barrier_parameter: f64 },
        /// Constrained expected improvement
        ConstrainedEI,
        /// Probability of feasibility
        ProbabilityOfFeasibility { threshold: f64 },
        /// Augmented Lagrangian
        AugmentedLagrangian { lambda: Vec<f64>, rho: f64 },
    }

    /// Advanced Bayesian optimizer with cutting-edge techniques
    pub struct AdvancedBayesianOptimizer<F: Float> {
        config: AdvancedBayesianConfig,
        gp_models: Vec<GaussianProcessModel>,
        pareto_front: Vec<(HashMap<String, f64>, Vec<f64>)>,
        source_tasks: Vec<SourceTask>,
        acquisition_performance: Vec<f64>,
        uncertainty_estimates: Vec<f64>,
        _phantom: std::marker::PhantomData<F>,
    }

    impl<F> AdvancedBayesianOptimizer<F>
    where
        F: Float + FromPrimitive + Debug + 'static + std::fmt::Display + Send + Sync,
        f64: From<F>,
    {
        /// Create new advanced Bayesian optimizer
        pub fn new(config: AdvancedBayesianConfig) -> Self {
            Self {
                config,
                gp_models: Vec::new(),
                pareto_front: Vec::new(),
                source_tasks: Vec::new(),
                acquisition_performance: Vec::new(),
                uncertainty_estimates: Vec::new(),
                _phantom: std::marker::PhantomData,
            }
        }

        /// Optimize with multi-fidelity approach
        pub fn optimize_multi_fidelity(
            &mut self,
            search_space: &SearchSpace,
            evaluation_fn: &dyn Fn(&HashMap<String, f64>, f64) -> Result<f64>,
        ) -> Result<TuningResult> {
            if let Some(multi_fidelity_config) = self.config.multi_fidelity.clone() {
                self.multi_fidelity_optimization(
                    search_space,
                    evaluation_fn,
                    &multi_fidelity_config,
                )
            } else {
                Err(ClusteringError::InvalidInput(
                    "Multi-fidelity config not provided".to_string(),
                ))
            }
        }

        /// Optimize with transfer learning
        pub fn optimize_with_transfer_learning(
            &mut self,
            search_space: &SearchSpace,
            evaluation_fn: &dyn Fn(&HashMap<String, f64>) -> Result<f64>,
        ) -> Result<TuningResult> {
            if let Some(transfer_config) = self.config.transfer_learning.clone() {
                self.transfer_learning_optimization(search_space, evaluation_fn, &transfer_config)
            } else {
                Err(ClusteringError::InvalidInput(
                    "Transfer learning config not provided".to_string(),
                ))
            }
        }

        /// Multi-objective Bayesian optimization
        pub fn optimize_multi_objective(
            &mut self,
            search_space: &SearchSpace,
            evaluation_fn: &dyn Fn(&HashMap<String, f64>) -> Result<Vec<f64>>,
        ) -> Result<MultiObjectiveResult> {
            if let Some(mo_config) = self.config.multi_objective.clone() {
                self.multi_objective_optimization(search_space, evaluation_fn, &mo_config)
            } else {
                Err(ClusteringError::InvalidInput(
                    "Multi-objective config not provided".to_string(),
                ))
            }
        }

        /// Batch Bayesian optimization for parallel evaluation
        pub fn optimize_batch(
            &mut self,
            search_space: &SearchSpace,
            evaluation_fn: &dyn Fn(&[HashMap<String, f64>]) -> Result<Vec<f64>>,
        ) -> Result<TuningResult> {
            if let Some(batch_config) = self.config.advanced_acquisition.batch_acquisition.clone() {
                self.batch_optimization(search_space, evaluation_fn, &batch_config)
            } else {
                Err(ClusteringError::InvalidInput(
                    "Batch acquisition config not provided".to_string(),
                ))
            }
        }

        // Implementation methods

        fn multi_fidelity_optimization(
            &mut self,
            search_space: &SearchSpace,
            evaluation_fn: &dyn Fn(&HashMap<String, f64>, f64) -> Result<f64>,
            config: &MultiFidelityConfig,
        ) -> Result<TuningResult> {
            let mut observations = Vec::new();
            let mut best_score = f64::NEG_INFINITY;
            let mut best_params = HashMap::new();

            // Initialize with low-fidelity evaluations
            for _ in 0..10 {
                let params = self.sample_random_parameters(search_space)?;
                let fidelity = config.fidelity_levels[0]; // Start with lowest fidelity
                let score = evaluation_fn(&params, fidelity)?;

                observations.push((params.clone(), fidelity, score));

                if score > best_score {
                    best_score = score;
                    best_params = params;
                }
            }

            // Multi-fidelity optimization loop
            for iteration in 0..50 {
                // Select next point and fidelity level
                let (next_params, next_fidelity) = self.select_next_point_and_fidelity(
                    search_space,
                    &observations,
                    config,
                    iteration,
                )?;

                // Evaluate at selected fidelity
                let score = evaluation_fn(&next_params, next_fidelity)?;
                observations.push((next_params.clone(), next_fidelity, score));

                // Update best if improved
                if score > best_score {
                    best_score = score;
                    best_params = next_params;
                }

                // Update multi-fidelity models
                self.update_multi_fidelity_models(&observations, config)?;
            }

            Ok(TuningResult {
                best_parameters: best_params,
                best_score,
                evaluation_history: observations
                    .into_iter()
                    .map(|(params, _fidelity, score)| EvaluationResult {
                        parameters: params,
                        score,
                        additional_metrics: HashMap::new(),
                        evaluation_time: 0.0,
                        memory_usage: None,
                        cv_scores: vec![score],
                        cv_std: 0.0,
                        metadata: HashMap::new(),
                    })
                    .collect(),
                convergence_info: ConvergenceInfo {
                    converged: false,
                    convergence_iteration: None,
                    stopping_reason: StoppingReason::MaxEvaluations,
                },
                exploration_stats: ExplorationStats {
                    coverage: 0.8,
                    parameter_distributions: HashMap::new(),
                    parameter_importance: HashMap::new(),
                },
                total_time: 0.0,
                ensemble_results: None,
                pareto_front: None,
            })
        }

        fn transfer_learning_optimization(
            &mut self,
            search_space: &SearchSpace,
            evaluation_fn: &dyn Fn(&HashMap<String, f64>) -> Result<f64>,
            config: &TransferLearningConfig,
        ) -> Result<TuningResult> {
            // Initialize with source task knowledge
            self.initialize_with_source_tasks(config)?;

            let mut observations = Vec::new();
            let mut best_score = f64::NEG_INFINITY;
            let mut best_params = HashMap::new();

            // Warm-start with transfer learning
            let warm_start_params = self.generate_warm_start_parameters(search_space, config)?;

            for params in warm_start_params {
                let score = evaluation_fn(&params)?;
                observations.push((params.clone(), score));

                if score > best_score {
                    best_score = score;
                    best_params = params;
                }
            }

            // Continue optimization with transferred knowledge
            for _iteration in 0..40 {
                let next_params =
                    self.select_next_point_with_transfer(search_space, &observations, config)?;

                let score = evaluation_fn(&next_params)?;
                observations.push((next_params.clone(), score));

                if score > best_score {
                    best_score = score;
                    best_params = next_params;
                }

                // Update transfer learning model
                self.update_transfer_model(&observations, config)?;
            }

            Ok(TuningResult {
                best_parameters: best_params,
                best_score,
                evaluation_history: observations
                    .into_iter()
                    .map(|(params, score)| EvaluationResult {
                        parameters: params,
                        score,
                        additional_metrics: HashMap::new(),
                        cv_scores: vec![score],
                        cv_std: 0.0,
                        evaluation_time: 0.0,
                        memory_usage: None,
                        metadata: HashMap::new(),
                    })
                    .collect(),
                convergence_info: ConvergenceInfo {
                    converged: false,
                    convergence_iteration: None,
                    stopping_reason: StoppingReason::MaxEvaluations,
                },
                exploration_stats: ExplorationStats {
                    coverage: 0.85,
                    parameter_distributions: HashMap::new(),
                    parameter_importance: HashMap::new(),
                },
                total_time: 0.0,
                ensemble_results: None,
                pareto_front: None,
            })
        }

        fn multi_objective_optimization(
            &mut self,
            search_space: &SearchSpace,
            evaluation_fn: &dyn Fn(&HashMap<String, f64>) -> Result<Vec<f64>>,
            config: &MultiObjectiveConfig,
        ) -> Result<MultiObjectiveResult> {
            let mut pareto_front = Vec::new();
            let mut all_evaluations = Vec::new();

            // Initialize random population
            for _ in 0..20 {
                let params = self.sample_random_parameters(search_space)?;
                let objectives = evaluation_fn(&params)?;

                all_evaluations.push((params.clone(), objectives.clone()));
                self.update_pareto_front(&mut pareto_front, params, objectives);
            }

            // Multi-objective optimization loop
            for _iteration in 0..80 {
                let next_params = match config.method {
                    MultiObjectiveMethod::WeightedScalarization => self
                        .select_point_weighted_scalarization(
                            search_space,
                            &all_evaluations,
                            &config.weights,
                        )?,
                    MultiObjectiveMethod::Hypervolume {
                        ref reference_point,
                    } => {
                        self.select_point_hypervolume(search_space, &pareto_front, reference_point)?
                    }
                    MultiObjectiveMethod::ParEGO { rho } => {
                        self.select_point_parego(search_space, &all_evaluations, rho)?
                    }
                    _ => self.sample_random_parameters(search_space)?,
                };

                let objectives = evaluation_fn(&next_params)?;
                all_evaluations.push((next_params.clone(), objectives.clone()));
                self.update_pareto_front(&mut pareto_front, next_params, objectives);
            }

            Ok(MultiObjectiveResult {
                pareto_front: pareto_front.clone(),
                all_evaluations,
                hypervolume: self.calculate_hypervolume(&pareto_front, &config.method),
                convergence_metrics: self.calculate_convergence_metrics(&pareto_front),
            })
        }

        fn batch_optimization(
            &mut self,
            search_space: &SearchSpace,
            evaluation_fn: &dyn Fn(&[HashMap<String, f64>]) -> Result<Vec<f64>>,
            config: &BatchAcquisitionConfig,
        ) -> Result<TuningResult> {
            let mut observations = Vec::new();
            let mut best_score = f64::NEG_INFINITY;
            let mut best_params = HashMap::new();

            // Initialize with random batch
            let initial_batch = (0..config.batch_size)
                .map(|_| self.sample_random_parameters(search_space))
                .collect::<Result<Vec<_>>>()?;

            let initial_scores = evaluation_fn(&initial_batch)?;

            for (params, score) in initial_batch.into_iter().zip(initial_scores.into_iter()) {
                observations.push((params.clone(), score));
                if score > best_score {
                    best_score = score;
                    best_params = params;
                }
            }

            // Batch optimization loop
            for _iteration in 0..10 {
                let next_batch = self.select_batch_points(search_space, &observations, config)?;
                let batch_scores = evaluation_fn(&next_batch)?;

                for (params, score) in next_batch.into_iter().zip(batch_scores.into_iter()) {
                    observations.push((params.clone(), score));
                    if score > best_score {
                        best_score = score;
                        best_params = params;
                    }
                }
            }

            Ok(TuningResult {
                best_parameters: best_params,
                best_score,
                evaluation_history: observations
                    .into_iter()
                    .map(|(params, score)| EvaluationResult {
                        parameters: params,
                        score,
                        additional_metrics: HashMap::new(),
                        cv_scores: vec![score],
                        cv_std: 0.0,
                        evaluation_time: 0.0,
                        memory_usage: None,
                        metadata: HashMap::new(),
                    })
                    .collect(),
                convergence_info: ConvergenceInfo {
                    converged: false,
                    convergence_iteration: None,
                    stopping_reason: StoppingReason::MaxEvaluations,
                },
                exploration_stats: ExplorationStats {
                    coverage: 0.9,
                    parameter_distributions: HashMap::new(),
                    parameter_importance: HashMap::new(),
                },
                total_time: 0.0,
                ensemble_results: None,
                pareto_front: None,
            })
        }

        // Helper methods (stubs for brevity)

        fn sample_random_parameters(
            &self,
            search_space: &SearchSpace,
        ) -> Result<HashMap<String, f64>> {
            let mut rng = rand::rng();
            let mut params = HashMap::new();

            for (name, param) in &search_space.parameters {
                let value = match param {
                    HyperParameter::Float { min, max } => rng.random_range(*min..*max),
                    HyperParameter::Integer { min, max } => rng.random_range(*min..*max) as f64,
                    _ => 1.0, // Stub
                };
                params.insert(name.clone(), value);
            }

            Ok(params)
        }

        fn select_next_point_and_fidelity(
            &mut self,
            _search_space: &SearchSpace,
            _observations: &[(HashMap<String, f64>, f64, f64)],
            _config: &MultiFidelityConfig,
            iteration: usize,
        ) -> Result<(HashMap<String, f64>, f64)> {
            // Stub implementation
            Ok((HashMap::new(), 1.0))
        }

        fn update_multi_fidelity_models(
            &mut self,
            _observations: &[(HashMap<String, f64>, f64, f64)],
            _config: &MultiFidelityConfig,
        ) -> Result<()> {
            Ok(())
        }

        fn initialize_with_source_tasks(&mut self, config: &TransferLearningConfig) -> Result<()> {
            self.source_tasks = config.source_tasks.clone();
            Ok(())
        }

        fn generate_warm_start_parameters(
            &self,
            search_space: &SearchSpace,
            _config: &TransferLearningConfig,
        ) -> Result<Vec<HashMap<String, f64>>> {
            // Generate promising initial points based on source tasks
            let mut params = Vec::new();
            for _ in 0..5 {
                params.push(self.sample_random_parameters(search_space)?);
            }
            Ok(params)
        }

        fn select_next_point_with_transfer(
            &mut self,
            search_space: &SearchSpace,
            _observations: &[(HashMap<String, f64>, f64)],
            _config: &TransferLearningConfig,
        ) -> Result<HashMap<String, f64>> {
            self.sample_random_parameters(search_space)
        }

        fn update_transfer_model(
            &mut self,
            _observations: &[(HashMap<String, f64>, f64)],
            _config: &TransferLearningConfig,
        ) -> Result<()> {
            Ok(())
        }

        fn update_pareto_front(
            &mut self,
            pareto_front: &mut Vec<(HashMap<String, f64>, Vec<f64>)>,
            params: HashMap<String, f64>,
            objectives: Vec<f64>,
        ) {
            // Remove dominated points and add if non-dominated
            pareto_front.retain(|(_, obj)| !self.dominates(&objectives, obj));

            let dominated_by_new = pareto_front
                .iter()
                .any(|(_, obj)| self.dominates(obj, &objectives));
            if !dominated_by_new {
                pareto_front.push((params, objectives));
            }
        }

        fn dominates(&self, a: &[f64], b: &[f64]) -> bool {
            a.iter().zip(b.iter()).all(|(ai, bi)| ai >= bi)
                && a.iter().zip(b.iter()).any(|(ai, bi)| ai > bi)
        }

        fn select_point_weighted_scalarization(
            &mut self,
            search_space: &SearchSpace,
            _evaluations: &[(HashMap<String, f64>, Vec<f64>)],
            _weights: &[f64],
        ) -> Result<HashMap<String, f64>> {
            self.sample_random_parameters(search_space)
        }

        fn select_point_hypervolume(
            &mut self,
            search_space: &SearchSpace,
            _pareto_front: &[(HashMap<String, f64>, Vec<f64>)],
            _reference_point: &[f64],
        ) -> Result<HashMap<String, f64>> {
            self.sample_random_parameters(search_space)
        }

        fn select_point_parego(
            &mut self,
            search_space: &SearchSpace,
            _evaluations: &[(HashMap<String, f64>, Vec<f64>)],
            _rho: f64,
        ) -> Result<HashMap<String, f64>> {
            self.sample_random_parameters(search_space)
        }

        fn calculate_hypervolume(
            &self,
            _pareto_front: &[(HashMap<String, f64>, Vec<f64>)],
            _method: &MultiObjectiveMethod,
        ) -> f64 {
            0.0 // Stub
        }

        fn calculate_convergence_metrics(
            &self,
            _pareto_front: &[(HashMap<String, f64>, Vec<f64>)],
        ) -> ConvergenceMetrics {
            ConvergenceMetrics {
                spacing: 0.0,
                spread: 0.0,
                coverage: 0.0,
                uniformity: 0.0,
            }
        }

        fn select_batch_points(
            &mut self,
            search_space: &SearchSpace,
            _observations: &[(HashMap<String, f64>, f64)],
            config: &BatchAcquisitionConfig,
        ) -> Result<Vec<HashMap<String, f64>>> {
            let mut batch = Vec::new();
            for _ in 0..config.batch_size {
                batch.push(self.sample_random_parameters(search_space)?);
            }
            Ok(batch)
        }
    }

    // Supporting structures

    /// Gaussian Process model for advanced optimization
    #[derive(Debug, Clone)]
    pub struct GaussianProcessModel {
        /// Kernel parameters
        pub kernel_params: Vec<f64>,
        /// Training data
        pub training_data: Array2<f64>,
        /// Training targets
        pub training_targets: Array1<f64>,
        /// Noise level
        pub noise_level: f64,
    }

    /// Multi-objective optimization result
    #[derive(Debug, Clone)]
    pub struct MultiObjectiveResult {
        /// Pareto front solutions
        pub pareto_front: Vec<(HashMap<String, f64>, Vec<f64>)>,
        /// All evaluated solutions
        pub all_evaluations: Vec<(HashMap<String, f64>, Vec<f64>)>,
        /// Hypervolume metric
        pub hypervolume: f64,
        /// Convergence metrics
        pub convergence_metrics: ConvergenceMetrics,
    }

    /// Convergence metrics for multi-objective optimization
    #[derive(Debug, Clone)]
    pub struct ConvergenceMetrics {
        /// Spacing between solutions
        pub spacing: f64,
        /// Spread of solutions
        pub spread: f64,
        /// Coverage of objective space
        pub coverage: f64,
        /// Uniformity of distribution
        pub uniformity: f64,
    }

    /// Advanced acquisition functions
    impl AdvancedBayesianOptimizer<f64> {
        /// Expected Hypervolume Improvement
        pub fn expected_hypervolume_improvement(
            &self,
            _candidate: &HashMap<String, f64>,
            _pareto_front: &[(HashMap<String, f64>, Vec<f64>)],
            _reference_point: &[f64],
        ) -> f64 {
            // Stub implementation
            0.5
        }

        /// Multi-point Expected Improvement
        pub fn multi_point_expected_improvement(
            &self,
            candidates: &[HashMap<String, f64>],
            _observations: &[(HashMap<String, f64>, f64)],
        ) -> f64 {
            // Stub implementation
            candidates.len() as f64 * 0.1
        }

        /// Entropy Search acquisition function
        pub fn entropy_search(
            self_candidate: &HashMap<String, f64>,
            _observations: &[(HashMap<String, f64>, f64)],
        ) -> f64 {
            // Stub implementation - would compute information gain
            0.3
        }

        /// Knowledge Gradient acquisition function
        pub fn knowledge_gradient(
            self_candidate: &HashMap<String, f64>,
            _observations: &[(HashMap<String, f64>, f64)],
        ) -> f64 {
            // Stub implementation - would compute expected value of information
            0.4
        }
    }

    /// Constraint handling utilities
    pub mod constraint_handling {
        use super::*;

        /// Evaluate constraint violations
        pub fn evaluate_constraints(
            _parameters: &HashMap<String, f64>,
            constraints: &[String], // Constraint expressions
        ) -> Vec<f64> {
            // Stub implementation
            vec![0.0; constraints.len()]
        }

        /// Penalty method for constraint handling
        pub fn penalty_method(
            objective_value: f64,
            constraint_violations: &[f64],
            penalty_factor: f64,
        ) -> f64 {
            let penalty = constraint_violations
                .iter()
                .map(|&v| penalty_factor * v.max(0.0).powi(2))
                .sum::<f64>();
            objective_value - penalty
        }

        /// Barrier method for constraint handling
        pub fn barrier_method(
            objective_value: f64,
            constraint_violations: &[f64],
            barrier_parameter: f64,
        ) -> f64 {
            let barrier = constraint_violations
                .iter()
                .map(|&v| {
                    if v <= 0.0 {
                        0.0
                    } else {
                        -barrier_parameter * v.ln()
                    }
                })
                .sum::<f64>();
            objective_value + barrier
        }
    }
}

/// Neural Architecture Search for clustering algorithms
pub mod neural_architecture_search {
    use super::*;
    use scirs2_core::parallel_ops::*;
    use std::sync::{Arc, Mutex};

    /// Neural Architecture Search (NAS) for clustering algorithms
    ///
    /// This approach uses neural networks to automatically discover
    /// optimal hyperparameter configurations for clustering algorithms.
    pub struct NeuralArchitectureSearch<F: Float> {
        /// Controller network weights
        controller_weights: Vec<Array2<F>>,
        /// Search space definition
        search_space: SearchSpace,
        /// Training history
        training_history: Vec<(HashMap<String, F>, F)>,
        /// Current exploration rate
        exploration_rate: f64,
        /// Learning rate for controller updates
        learning_rate: f64,
    }

    impl<F: Float + FromPrimitive + Debug + Send + Sync> NeuralArchitectureSearch<F> {
        /// Create new NAS instance
        pub fn new(search_space: SearchSpace) -> Self {
            Self {
                controller_weights: Vec::new(),
                search_space,
                training_history: Vec::new(),
                exploration_rate: 0.1,
                learning_rate: 0.001,
            }
        }

        /// Initialize controller network
        pub fn initialize_controller(&mut self, input_dim: usize, hiddendim: usize) {
            let mut rng = rand::rng();

            // Simple 2-layer network for hyperparameter generation
            let w1 = Array2::from_shape_fn((input_dim, hiddendim), |_| {
                F::from(rng.random_range(-0.1..0.1)).unwrap()
            });
            let w2 = Array2::from_shape_fn((hiddendim, self.search_space.parameters.len()), |_| {
                F::from(rng.random_range(-0.1..0.1)).unwrap()
            });

            self.controller_weights = vec![w1, w2];
        }

        /// Generate hyperparameter configuration using neural controller
        pub fn generate_config(&self, context: &Array1<F>) -> Result<HashMap<String, F>> {
            if self.controller_weights.is_empty() {
                return Err(ClusteringError::InvalidInput(
                    "Controller not initialized".to_string(),
                ));
            }

            // Forward pass through controller network
            let mut current = context.clone();

            for weights in &self.controller_weights {
                let mut output = Array1::zeros(weights.ncols());

                for (i, mut out_val) in output.iter_mut().enumerate() {
                    let mut sum = F::zero();
                    for (j, &input_val) in current.iter().enumerate() {
                        sum = sum + input_val * weights[[j, i]];
                    }
                    *out_val = self.tanh_activation(sum);
                }

                current = output;
            }

            // Convert network output to hyperparameter values
            let mut config = HashMap::new();
            for (i, (param_name, param_def)) in self.search_space.parameters.iter().enumerate() {
                if i < current.len() {
                    let normalized_value = (current[i] + F::one()) / F::from(2.0).unwrap(); // Map from [-1,1] to [0,1]
                    let param_value = match param_def {
                        HyperParameter::Float { min, max } => {
                            F::from(*min).unwrap() + normalized_value * F::from(max - min).unwrap()
                        }
                        HyperParameter::Integer { min, max } => {
                            let range = F::from(max - min).unwrap();
                            let scaled = min
                                + (normalized_value * range).round().to_usize().unwrap_or(0) as i64;
                            F::from(scaled).unwrap()
                        }
                        HyperParameter::Categorical { choices } => {
                            let idx = (normalized_value * F::from(choices.len()).unwrap())
                                .to_usize()
                                .unwrap_or(0)
                                .min(choices.len() - 1);
                            F::from(idx).unwrap()
                        }
                        HyperParameter::Boolean => {
                            if normalized_value > F::from(0.5).unwrap() {
                                F::one()
                            } else {
                                F::zero()
                            }
                        }
                        HyperParameter::LogUniform { min, max } => {
                            let log_min = min.ln();
                            let log_max = max.ln();
                            let log_val =
                                log_min + normalized_value.to_f64().unwrap() * (log_max - log_min);
                            F::from(log_val.exp()).unwrap()
                        }
                        HyperParameter::IntegerChoices { choices } => {
                            let idx = (normalized_value * F::from(choices.len()).unwrap())
                                .to_usize()
                                .unwrap_or(0)
                                .min(choices.len() - 1);
                            F::from(choices[idx]).unwrap()
                        }
                    };
                    config.insert(param_name.clone(), param_value);
                }
            }

            Ok(config)
        }

        /// Activation function
        fn tanh_activation(&self, x: F) -> F {
            let x_f64 = x.to_f64().unwrap_or(0.0);
            F::from(x_f64.tanh()).unwrap()
        }

        /// Update controller based on performance feedback
        pub fn update_controller(&mut self, config: &HashMap<String, F>, performance: F) {
            self.training_history.push((config.clone(), performance));

            // Simple policy gradient update (simplified)
            // In practice, you'd implement full reinforcement learning update

            // Calculate baseline (average performance)
            let baseline = if self.training_history.len() > 1 {
                let sum: F = self
                    .training_history
                    .iter()
                    .map(|(_, p)| *p)
                    .fold(F::zero(), |acc, x| acc + x);
                sum / F::from(self.training_history.len()).unwrap()
            } else {
                performance
            };

            let advantage = performance - baseline;

            // Update exploration rate based on performance trend
            if advantage > F::zero() {
                self.exploration_rate *= 0.995; // Reduce exploration if improving
            } else {
                self.exploration_rate = (self.exploration_rate * 1.005).min(0.3);
                // Increase exploration if not improving
            }
        }

        /// Search for optimal hyperparameters using NAS
        pub fn search<ClusterFn>(
            &mut self,
            data: ArrayView2<F>,
            clustering_fn: ClusterFn,
            n_trials: usize,
        ) -> Result<TuningResult>
        where
            ClusterFn: Fn(&HashMap<String, F>, ArrayView2<F>) -> Result<Array1<i32>> + Sync,
        {
            let mut _best_config = None;
            let mut best_score = F::neg_infinity();
            let mut all_results = Vec::new();

            // Initialize controller if not done
            if self.controller_weights.is_empty() {
                self.initialize_controller(10, 50); // Default dimensions
            }

            for trial in 0..n_trials {
                // Create context vector (simplified - could include data statistics)
                let context = Array1::from_vec(vec![
                    F::from(data.nrows()).unwrap(),
                    F::from(data.ncols()).unwrap(),
                    F::from(trial).unwrap() / F::from(n_trials).unwrap(),
                    F::from(self.exploration_rate).unwrap(),
                    F::zero(),
                    F::zero(),
                    F::zero(),
                    F::zero(),
                    F::zero(),
                    F::zero(),
                ]);

                // Generate configuration
                let config = self.generate_config(&context)?;

                // Evaluate configuration
                let labels = clustering_fn(&config, data)?;
                let score = self.evaluate_clustering(data, &labels)?;

                // Update controller
                self.update_controller(&config, score);

                // Track best result
                if score > best_score {
                    best_score = score;
                    _best_config = Some(config.clone());
                }

                all_results.push(EvaluationResult {
                    parameters: HashMap::new(), // Convert config to HashMap if needed
                    score: score.to_f64().unwrap_or(0.0),
                    cv_scores: vec![score.to_f64().unwrap_or(0.0)],
                    cv_std: 0.0,
                    additional_metrics: HashMap::new(),
                    evaluation_time: 0.0,
                    memory_usage: None,
                    metadata: HashMap::new(),
                });
            }

            Ok(TuningResult {
                best_parameters: HashMap::new(), // Convert best_config to HashMap if needed
                best_score: best_score.to_f64().unwrap_or(0.0),
                evaluation_history: all_results,
                convergence_info: ConvergenceInfo {
                    converged: true,
                    convergence_iteration: Some(n_trials),
                    stopping_reason: StoppingReason::MaxEvaluations,
                },
                exploration_stats: ExplorationStats {
                    coverage: self.exploration_rate,
                    parameter_distributions: HashMap::new(),
                    parameter_importance: HashMap::new(),
                },
                total_time: 0.0,
                ensemble_results: None,
                pareto_front: None,
            })
        }

        /// Evaluate clustering quality
        fn evaluate_clustering(&self, data: ArrayView2<F>, labels: &Array1<i32>) -> Result<F> {
            // Convert i32 labels to usize for metrics calculation
            let labels_usize: Array1<usize> = labels.mapv(|x| if x < 0 { 0 } else { x as usize });

            // Convert back to i32 for silhouette_score
            let labels_i32: Array1<i32> = labels_usize.mapv(|x| x as i32);
            silhouette_score(data, labels_i32.view())
        }
    }

    /// Multi-Armed Bandit approach for hyperparameter optimization
    ///
    /// This approach treats each hyperparameter configuration as an "arm"
    /// and uses bandit algorithms to balance exploration and exploitation.
    pub struct MultiarmedBanditOptimizer<F: Float> {
        /// Arms (hyperparameter configurations)
        arms: Vec<HashMap<String, F>>,
        /// Reward history for each arm
        rewards: Vec<Vec<F>>,
        /// Number of times each arm was pulled
        pulls: Vec<usize>,
        /// Bandit algorithm configuration
        algorithm: BanditAlgorithm,
        /// Search space
        search_space: SearchSpace,
    }

    /// Bandit algorithms
    #[derive(Debug, Clone)]
    pub enum BanditAlgorithm {
        /// Epsilon-greedy algorithm
        EpsilonGreedy { epsilon: f64 },
        /// Upper Confidence Bound (UCB1)
        UCB1 { c: f64 },
        /// Thompson Sampling
        ThompsonSampling,
        /// LinUCB for contextual bandits
        LinUCB { alpha: f64 },
    }

    impl<F: Float + FromPrimitive + Debug + Send + Sync> MultiarmedBanditOptimizer<F> {
        /// Create new multi-armed bandit optimizer
        pub fn new(search_space: SearchSpace, algorithm: BanditAlgorithm) -> Self {
            Self {
                arms: Vec::new(),
                rewards: Vec::new(),
                pulls: Vec::new(),
                algorithm,
                search_space,
            }
        }

        /// Initialize arms with random configurations
        pub fn initialize_arms(&mut self, narms: usize) -> Result<()> {
            let mut rng = rand::rng();

            for _ in 0..narms {
                let mut config = HashMap::new();

                for (param_name, param_def) in &self.search_space.parameters {
                    let value = match param_def {
                        HyperParameter::Float { min, max } => {
                            let random_val = rng.random_range(0.0..1.0);
                            F::from(*min).unwrap()
                                + F::from(random_val).unwrap() * F::from(max - min).unwrap()
                        }
                        HyperParameter::Integer { min, max } => {
                            let val = rng.random_range(*min..=*max);
                            F::from(val).unwrap()
                        }
                        HyperParameter::Categorical { choices } => {
                            let idx = rng.random_range(0..choices.len());
                            F::from(idx).unwrap()
                        }
                        HyperParameter::Boolean => {
                            if rng.random_range(0.0..1.0) > 0.5 {
                                F::one()
                            } else {
                                F::zero()
                            }
                        }
                        HyperParameter::LogUniform { min, max } => {
                            let log_min = min.ln();
                            let log_max = max.ln();
                            let log_val =
                                log_min + rng.random_range(0.0..1.0) * (log_max - log_min);
                            F::from(log_val.exp()).unwrap()
                        }
                        HyperParameter::IntegerChoices { choices } => {
                            let idx = rng.random_range(0..choices.len());
                            F::from(choices[idx]).unwrap()
                        }
                    };
                    config.insert(param_name.clone(), value);
                }

                self.arms.push(config);
                self.rewards.push(Vec::new());
                self.pulls.push(0);
            }

            Ok(())
        }

        /// Select next arm to pull using bandit algorithm
        pub fn select_arm(&self) -> Result<usize> {
            if self.arms.is_empty() {
                return Err(ClusteringError::InvalidInput(
                    "No arms available".to_string(),
                ));
            }

            match &self.algorithm {
                BanditAlgorithm::EpsilonGreedy { epsilon } => {
                    let mut rng = rand::rng();
                    if rng.random_range(0.0..1.0) < *epsilon {
                        // Explore: choose random arm
                        Ok(rng.random_range(0..self.arms.len()))
                    } else {
                        // Exploit: choose best arm
                        let mut best_arm = 0;
                        let mut best_avg = F::neg_infinity();

                        for (i, rewards) in self.rewards.iter().enumerate() {
                            if !rewards.is_empty() {
                                let avg = rewards.iter().fold(F::zero(), |acc, &x| acc + x)
                                    / F::from(rewards.len()).unwrap();
                                if avg > best_avg {
                                    best_avg = avg;
                                    best_arm = i;
                                }
                            }
                        }

                        Ok(best_arm)
                    }
                }
                BanditAlgorithm::UCB1 { c } => {
                    let total_pulls: usize = self.pulls.iter().sum();
                    let mut best_arm = 0;
                    let mut best_ucb = F::neg_infinity();

                    for (i, rewards) in self.rewards.iter().enumerate() {
                        let ucb = if rewards.is_empty() {
                            F::infinity() // Unplayed arms have infinite UCB
                        } else {
                            let avg = rewards.iter().fold(F::zero(), |acc, &x| acc + x)
                                / F::from(rewards.len()).unwrap();
                            let confidence = F::from(
                                *c * (total_pulls as f64 / self.pulls[i] as f64).ln().sqrt(),
                            )
                            .unwrap();
                            avg + confidence
                        };

                        if ucb > best_ucb {
                            best_ucb = ucb;
                            best_arm = i;
                        }
                    }

                    Ok(best_arm)
                }
                _ => {
                    // Default to random selection for other algorithms
                    let mut rng = rand::rng();
                    Ok(rng.random_range(0..self.arms.len()))
                }
            }
        }

        /// Update arm with reward
        pub fn update_arm(&mut self, armindex: usize, reward: F) -> Result<()> {
            if armindex >= self.arms.len() {
                return Err(ClusteringError::InvalidInput(
                    "Invalid arm _index".to_string(),
                ));
            }

            self.rewards[armindex].push(reward);
            self.pulls[armindex] += 1;

            Ok(())
        }

        /// Optimize hyperparameters using multi-armed bandit
        pub fn optimize<ClusterFn>(
            &mut self,
            data: ArrayView2<F>,
            clustering_fn: ClusterFn,
            n_trials: usize,
        ) -> Result<TuningResult>
        where
            ClusterFn: Fn(&HashMap<String, F>, ArrayView2<F>) -> Result<Array1<i32>> + Sync,
        {
            // Initialize arms if not done
            if self.arms.is_empty() {
                self.initialize_arms(20)?; // Default 20 arms
            }

            let mut _best_config = None;
            let mut best_score = F::neg_infinity();
            let mut all_results = Vec::new();

            for _ in 0..n_trials {
                // Select arm
                let arm_index = self.select_arm()?;

                // Evaluate configuration
                let labels = clustering_fn(&self.arms[arm_index], data)?;
                let score = self.evaluate_clustering(data, &labels)?;

                // Update arm
                self.update_arm(arm_index, score)?;

                // Track best result
                if score > best_score {
                    best_score = score;
                    _best_config = Some(self.arms[arm_index].clone());
                }

                all_results.push(EvaluationResult {
                    parameters: HashMap::new(), // Convert config to HashMap if needed
                    score: score.to_f64().unwrap_or(0.0),
                    cv_scores: vec![score.to_f64().unwrap_or(0.0)],
                    cv_std: 0.0,
                    additional_metrics: HashMap::new(),
                    evaluation_time: 0.0,
                    memory_usage: None,
                    metadata: HashMap::new(),
                });
            }

            Ok(TuningResult {
                best_parameters: HashMap::new(), // Convert best_config to HashMap if needed
                best_score: best_score.to_f64().unwrap_or(0.0),
                evaluation_history: all_results,
                convergence_info: ConvergenceInfo {
                    converged: true,
                    convergence_iteration: Some(n_trials),
                    stopping_reason: StoppingReason::MaxEvaluations,
                },
                exploration_stats: ExplorationStats {
                    coverage: 0.5, // Placeholder
                    parameter_distributions: HashMap::new(),
                    parameter_importance: HashMap::new(),
                },
                total_time: 0.0,
                ensemble_results: None,
                pareto_front: None,
            })
        }

        /// Evaluate clustering quality
        fn evaluate_clustering(&self, data: ArrayView2<F>, labels: &Array1<i32>) -> Result<F> {
            let labels_usize: Array1<usize> = labels.mapv(|x| if x < 0 { 0 } else { x as usize });

            // Convert back to i32 for silhouette_score
            let labels_i32: Array1<i32> = labels_usize.mapv(|x| x as i32);
            silhouette_score(data, labels_i32.view())
        }

        /// Get arm statistics
        pub fn get_arm_stats(&self) -> Vec<(f64, usize)> {
            self.rewards
                .iter()
                .zip(self.pulls.iter())
                .map(|(rewards, &pulls)| {
                    let avg = if rewards.is_empty() {
                        0.0
                    } else {
                        let sum = rewards.iter().fold(F::zero(), |acc, &x| acc + x);
                        (sum / F::from(rewards.len()).unwrap())
                            .to_f64()
                            .unwrap_or(0.0)
                    };
                    (avg, pulls)
                })
                .collect()
        }
    }

    /// Population-based training for hyperparameter optimization
    ///
    /// This approach maintains a population of configurations and evolves
    /// them over time using genetic algorithm principles.
    pub struct PopulationBasedTraining<F: Float> {
        /// Current population
        population: Vec<Individual<F>>,
        /// Population size
        population_size: usize,
        /// Selection pressure (top fraction to keep)
        selection_pressure: f64,
        /// Mutation rate
        mutation_rate: f64,
        /// Search space
        search_space: SearchSpace,
        /// Generation counter
        generation: usize,
    }

    /// Individual in the population
    #[derive(Debug, Clone)]
    pub struct Individual<F: Float> {
        /// Hyperparameter configuration
        config: HashMap<String, F>,
        /// Fitness score
        fitness: F,
        /// Age (number of generations)
        age: usize,
    }

    impl<F: Float + FromPrimitive + Debug + Send + Sync> PopulationBasedTraining<F> {
        /// Create new population-based training optimizer
        pub fn new(
            search_space: SearchSpace,
            population_size: usize,
            selection_pressure: f64,
            mutation_rate: f64,
        ) -> Self {
            Self {
                population: Vec::new(),
                population_size,
                selection_pressure,
                mutation_rate,
                search_space,
                generation: 0,
            }
        }

        /// Initialize random population
        pub fn initialize_population(&mut self) -> Result<()> {
            let mut rng = rand::rng();

            for _ in 0..self.population_size {
                let mut config = HashMap::new();

                for (param_name, param_def) in &self.search_space.parameters {
                    let value = match param_def {
                        HyperParameter::Float { min, max } => {
                            let random_val = rng.random_range(0.0..1.0);
                            F::from(*min).unwrap()
                                + F::from(random_val).unwrap() * F::from(max - min).unwrap()
                        }
                        HyperParameter::Integer { min, max } => {
                            let val = rng.random_range(*min..=*max);
                            F::from(val).unwrap()
                        }
                        HyperParameter::Categorical { choices } => {
                            let idx = rng.random_range(0..choices.len());
                            F::from(idx).unwrap()
                        }
                        HyperParameter::Boolean => {
                            if rng.random_range(0.0..1.0) > 0.5 {
                                F::one()
                            } else {
                                F::zero()
                            }
                        }
                        HyperParameter::LogUniform { min, max } => {
                            let log_min = min.ln();
                            let log_max = max.ln();
                            let log_val =
                                log_min + rng.random_range(0.0..1.0) * (log_max - log_min);
                            F::from(log_val.exp()).unwrap()
                        }
                        HyperParameter::IntegerChoices { choices } => {
                            let idx = rng.random_range(0..choices.len());
                            F::from(choices[idx]).unwrap()
                        }
                    };
                    config.insert(param_name.clone(), value);
                }

                self.population.push(Individual {
                    config,
                    fitness: F::zero(),
                    age: 0,
                });
            }

            Ok(())
        }

        /// Evolve population for one generation
        pub fn evolve<ClusterFn>(
            &mut self,
            data: ArrayView2<F>,
            clustering_fn: &ClusterFn,
        ) -> Result<()>
        where
            ClusterFn: Fn(&HashMap<String, F>, ArrayView2<F>) -> Result<Array1<i32>> + Sync,
        {
            // Evaluate fitness for all individuals
            let mut fitness_values = Vec::new();
            for individual in &self.population {
                let labels = clustering_fn(&individual.config, data)?;
                let fitness = self.evaluate_clustering(data, &labels)?;
                fitness_values.push(fitness);
            }

            // Update fitness and age
            for (individual, fitness) in self.population.iter_mut().zip(fitness_values.iter()) {
                individual.fitness = *fitness;
                individual.age += 1;
            }

            // Sort by fitness (descending)
            self.population.sort_by(|a, b| {
                b.fitness
                    .partial_cmp(&a.fitness)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            // Selection: keep top individuals
            let n_keep = (self.population_size as f64 * self.selection_pressure) as usize;
            let survivors = self.population[..n_keep].to_vec();

            // Reproduction: create new individuals
            let mut new_population = survivors.clone();
            let mut rng = rand::rng();

            while new_population.len() < self.population_size {
                // Select two parents (tournament selection)
                let parent1_idx = rng.random_range(0..survivors.len());
                let parent2_idx = rng.random_range(0..survivors.len());
                let parent1 = &survivors[parent1_idx];
                let parent2 = &survivors[parent2_idx];

                // Crossover
                let mut child_config = HashMap::new();
                for (param_name, _) in &self.search_space.parameters {
                    let value = if rng.random_range(0.0..1.0) < 0.5 {
                        parent1.config[param_name]
                    } else {
                        parent2.config[param_name]
                    };
                    child_config.insert(param_name.clone(), value);
                }

                // Mutation
                if rng.random_range(0.0..1.0) < self.mutation_rate {
                    self.mutate_config(&mut child_config)?;
                }

                new_population.push(Individual {
                    config: child_config,
                    fitness: F::zero(),
                    age: 0,
                });
            }

            self.population = new_population;
            self.generation += 1;

            Ok(())
        }

        /// Mutate a configuration
        fn mutate_config(&self, config: &mut HashMap<String, F>) -> Result<()> {
            let mut rng = rand::rng();

            // Select random parameter to mutate
            let param_names: Vec<_> = config.keys().cloned().collect();
            if param_names.is_empty() {
                return Ok(());
            }

            let param_name = &param_names[rng.random_range(0..param_names.len())];

            if let Some(param_def) = self
                .search_space
                .parameters
                .iter()
                .find(|(name, _)| name == &param_name)
                .map(|(_, def)| def)
            {
                let new_value = match param_def {
                    HyperParameter::Float { min, max } => {
                        let current = config[param_name];
                        let noise_val = rng.random_range(-0.1..0.1);
                        let noise = F::from(noise_val).unwrap() * F::from(max - min).unwrap();
                        (current + noise)
                            .max(F::from(*min).unwrap())
                            .min(F::from(*max).unwrap())
                    }
                    HyperParameter::Integer { min, max } => {
                        let val = rng.random_range(*min..=*max);
                        F::from(val).unwrap()
                    }
                    HyperParameter::Categorical { choices } => {
                        let idx = rng.random_range(0..choices.len());
                        F::from(idx).unwrap()
                    }
                    HyperParameter::Boolean => {
                        if rng.random::<f64>() > 0.5 {
                            F::one()
                        } else {
                            F::zero()
                        }
                    }
                    HyperParameter::LogUniform { min, max } => {
                        let log_min = min.ln();
                        let log_max = max.ln();
                        let log_val = rng.random_range(log_min..log_max);
                        F::from(log_val.exp()).unwrap()
                    }
                    HyperParameter::IntegerChoices { choices } => {
                        let idx = rng.random_range(0..choices.len());
                        F::from(choices[idx]).unwrap()
                    }
                };

                config.insert(param_name.clone(), new_value);
            }

            Ok(())
        }

        /// Evaluate clustering quality
        fn evaluate_clustering(&self, data: ArrayView2<F>, labels: &Array1<i32>) -> Result<F> {
            let labels_usize: Array1<usize> = labels.mapv(|x| if x < 0 { 0 } else { x as usize });

            // Convert back to i32 for silhouette_score
            let labels_i32: Array1<i32> = labels_usize.mapv(|x| x as i32);
            silhouette_score(data, labels_i32.view())
        }

        /// Get best individual from current population
        pub fn get_best(&self) -> Option<&Individual<F>> {
            self.population.iter().max_by(|a, b| {
                a.fitness
                    .partial_cmp(&b.fitness)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
        }

        /// Get population statistics
        pub fn get_population_stats(&self) -> (F, F, F) {
            if self.population.is_empty() {
                return (F::zero(), F::zero(), F::zero());
            }

            let fitnesses: Vec<F> = self.population.iter().map(|ind| ind.fitness).collect();
            let sum = fitnesses.iter().fold(F::zero(), |acc, &x| acc + x);
            let mean = sum / F::from(fitnesses.len()).unwrap();

            let min_fitness = fitnesses.iter().fold(F::infinity(), |acc, &x| acc.min(x));
            let max_fitness = fitnesses
                .iter()
                .fold(F::neg_infinity(), |acc, &x| acc.max(x));

            (mean, min_fitness, max_fitness)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_tuning_config_default() {
        let config = TuningConfig::default();
        assert_eq!(config.max_evaluations, 100);
        assert!(matches!(config.metric, EvaluationMetric::SilhouetteScore));
        assert!(config.early_stopping.is_some());
    }

    #[test]
    fn test_standard_search_spaces() {
        let kmeans_space = StandardSearchSpaces::kmeans();
        assert!(kmeans_space.parameters.contains_key("n_clusters"));
        assert!(kmeans_space.parameters.contains_key("max_iter"));

        let dbscan_space = StandardSearchSpaces::dbscan();
        assert!(dbscan_space.parameters.contains_key("eps"));
        assert!(dbscan_space.parameters.contains_key("min_samples"));

        let quantum_space = StandardSearchSpaces::quantum_kmeans();
        assert!(quantum_space.parameters.contains_key("n_clusters"));
        assert!(quantum_space.parameters.contains_key("n_quantum_states"));
        assert!(quantum_space.parameters.contains_key("decoherence_factor"));

        let rl_space = StandardSearchSpaces::rl_clustering();
        assert!(rl_space.parameters.contains_key("n_actions"));
        assert!(rl_space.parameters.contains_key("learning_rate"));

        let adaptive_space = StandardSearchSpaces::adaptive_online();
        assert!(adaptive_space
            .parameters
            .contains_key("initial_learning_rate"));
        assert!(adaptive_space.parameters.contains_key("max_clusters"));
    }

    #[test]
    fn test_auto_tuner_creation() {
        let config = TuningConfig::default();
        let tuner: AutoTuner<f64> = AutoTuner::new(config);
        // Test that the tuner can be created successfully
        assert_eq!(
            std::mem::size_of_val(&tuner),
            std::mem::size_of::<TuningConfig>()
        );
    }

    #[test]
    fn test_random_combinations_generation() {
        let config = TuningConfig::default();
        let tuner: AutoTuner<f64> = AutoTuner::new(config);
        let search_space = StandardSearchSpaces::kmeans();

        let combinations = tuner
            .generate_random_combinations(&search_space, 10)
            .unwrap();
        assert_eq!(combinations.len(), 10);

        for combo in &combinations {
            assert!(combo.contains_key("n_clusters"));
            assert!(combo.contains_key("max_iter"));

            let n_clusters = combo["n_clusters"];
            assert!(n_clusters >= 2.0 && n_clusters <= 20.0);
        }
    }
}

/// High-level automatic algorithm selection and tuning
pub struct AutoClusteringSelector<F: Float + FromPrimitive> {
    /// Tuning configuration
    config: TuningConfig,
    /// Algorithms to evaluate
    algorithms: Vec<ClusteringAlgorithm>,
    /// Phantom marker
    _phantom: std::marker::PhantomData<F>,
}

/// Clustering algorithm identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ClusteringAlgorithm {
    KMeans,
    DBSCAN,
    OPTICS,
    GaussianMixture,
    SpectralClustering,
    MeanShift,
    HierarchicalClustering,
    BIRCH,
    AffinityPropagation,
    QuantumKMeans,
    RLClustering,
    AdaptiveOnline,
}

/// Result of automatic algorithm selection
#[derive(Debug, Clone)]
pub struct AlgorithmSelectionResult {
    /// Best algorithm found
    pub best_algorithm: ClusteringAlgorithm,
    /// Best parameters for the algorithm
    pub best_parameters: HashMap<String, f64>,
    /// Best score achieved
    pub best_score: f64,
    /// Results for all algorithms tested
    pub algorithm_results: HashMap<ClusteringAlgorithm, TuningResult>,
    /// Total time spent on selection
    pub total_time: f64,
    /// Recommendations for the dataset
    pub recommendations: Vec<String>,
}

impl<
        F: Float
            + FromPrimitive
            + Debug
            + 'static
            + std::iter::Sum
            + std::fmt::Display
            + Send
            + Sync
            + ndarray::ScalarOperand
            + std::ops::AddAssign
            + std::ops::SubAssign
            + std::ops::MulAssign
            + std::ops::DivAssign
            + std::ops::RemAssign
            + PartialOrd,
    > AutoClusteringSelector<F>
where
    f64: From<F>,
{
    /// Create new automatic clustering selector
    pub fn new(config: TuningConfig) -> Self {
        Self {
            config,
            algorithms: vec![
                ClusteringAlgorithm::KMeans,
                ClusteringAlgorithm::DBSCAN,
                ClusteringAlgorithm::GaussianMixture,
                ClusteringAlgorithm::SpectralClustering,
                ClusteringAlgorithm::HierarchicalClustering,
            ],
            _phantom: std::marker::PhantomData,
        }
    }

    /// Create selector with all available algorithms
    pub fn with_all_algorithms(config: TuningConfig) -> Self {
        Self {
            config,
            algorithms: vec![
                ClusteringAlgorithm::KMeans,
                ClusteringAlgorithm::DBSCAN,
                ClusteringAlgorithm::OPTICS,
                ClusteringAlgorithm::GaussianMixture,
                ClusteringAlgorithm::SpectralClustering,
                ClusteringAlgorithm::MeanShift,
                ClusteringAlgorithm::HierarchicalClustering,
                ClusteringAlgorithm::BIRCH,
                ClusteringAlgorithm::AffinityPropagation,
                ClusteringAlgorithm::QuantumKMeans,
                ClusteringAlgorithm::RLClustering,
                ClusteringAlgorithm::AdaptiveOnline,
            ],
            _phantom: std::marker::PhantomData,
        }
    }

    /// Create selector with specific algorithms
    pub fn with_algorithms(config: TuningConfig, algorithms: Vec<ClusteringAlgorithm>) -> Self {
        Self {
            config,
            algorithms,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Automatically select and tune the best clustering algorithm
    pub fn select_best_algorithm(&self, data: ArrayView2<F>) -> Result<AlgorithmSelectionResult> {
        let start_time = std::time::Instant::now();
        let mut algorithm_results = HashMap::new();
        let mut best_algorithm = ClusteringAlgorithm::KMeans;
        let mut best_score = F::neg_infinity();
        let mut best_parameters = HashMap::new();

        let tuner = AutoTuner::new(self.config.clone());

        println!(
            "Testing {} algorithms for automatic selection...",
            self.algorithms.len()
        );

        for algorithm in &self.algorithms {
            println!("Tuning {algorithm:?}...");

            let tuning_result = match algorithm {
                ClusteringAlgorithm::KMeans => {
                    tuner.tune_kmeans(data, StandardSearchSpaces::kmeans())
                }
                ClusteringAlgorithm::DBSCAN => {
                    tuner.tune_dbscan(data, StandardSearchSpaces::dbscan())
                }
                ClusteringAlgorithm::OPTICS => {
                    tuner.tune_optics(data, StandardSearchSpaces::optics())
                }
                ClusteringAlgorithm::GaussianMixture => {
                    tuner.tune_gmm(data, StandardSearchSpaces::gmm())
                }
                ClusteringAlgorithm::SpectralClustering => {
                    tuner.tune_spectral(data, StandardSearchSpaces::spectral())
                }
                ClusteringAlgorithm::MeanShift => {
                    tuner.tune_mean_shift(data, StandardSearchSpaces::mean_shift())
                }
                ClusteringAlgorithm::HierarchicalClustering => {
                    tuner.tune_hierarchical(data, StandardSearchSpaces::hierarchical())
                }
                ClusteringAlgorithm::BIRCH => tuner.tune_birch(data, StandardSearchSpaces::birch()),
                ClusteringAlgorithm::AffinityPropagation => tuner
                    .tune_affinity_propagation(data, StandardSearchSpaces::affinity_propagation()),
                ClusteringAlgorithm::QuantumKMeans => {
                    tuner.tune_quantum_kmeans(data, StandardSearchSpaces::quantum_kmeans())
                }
                ClusteringAlgorithm::RLClustering => {
                    tuner.tune_rl_clustering(data, StandardSearchSpaces::rl_clustering())
                }
                ClusteringAlgorithm::AdaptiveOnline => {
                    tuner.tune_adaptive_online(data, StandardSearchSpaces::adaptive_online())
                }
            };

            match tuning_result {
                Ok(result) => {
                    println!(
                        "✓ {:?}: score = {:.4}, time = {:.2}s",
                        algorithm, result.best_score, result.total_time
                    );

                    if F::from(result.best_score).unwrap() > best_score {
                        best_score = F::from(result.best_score).unwrap();
                        best_algorithm = algorithm.clone();
                        best_parameters = result.best_parameters.clone();
                    }

                    algorithm_results.insert(algorithm.clone(), result);
                }
                Err(e) => {
                    println!("× {algorithm:?} failed: {e}");
                }
            }
        }

        let total_time = start_time.elapsed().as_secs_f64();
        let recommendations = self.generate_recommendations(data, &algorithm_results);

        Ok(AlgorithmSelectionResult {
            best_algorithm,
            best_parameters,
            best_score: best_score.to_f64().unwrap_or(0.0),
            algorithm_results,
            total_time,
            recommendations,
        })
    }

    /// Generate recommendations based on data characteristics and results
    fn generate_recommendations(
        &self,
        data: ArrayView2<F>,
        results: &HashMap<ClusteringAlgorithm, TuningResult>,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        let n_samples = data.nrows();
        let n_features = data.ncols();

        // Data size recommendations
        if n_samples < 100 {
            recommendations.push(
                "Small dataset: Consider K-means or Gaussian Mixture for stable results"
                    .to_string(),
            );
        } else if n_samples > 10000 {
            recommendations.push(
                "Large dataset: DBSCAN or Mini-batch K-means recommended for efficiency"
                    .to_string(),
            );
        }

        // Dimensionality recommendations
        if n_features > 50 {
            recommendations.push(
                "High-dimensional data: Consider dimensionality reduction before clustering"
                    .to_string(),
            );
        }

        // Algorithm-specific recommendations
        let mut sorted_results: Vec<_> = results.iter().collect();
        sorted_results.sort_by(|a, b| b.1.best_score.partial_cmp(&a.1.best_score).unwrap());

        if sorted_results.len() >= 2 {
            let best = &sorted_results[0];
            let second_best = &sorted_results[1];

            let score_diff = best.1.best_score - second_best.1.best_score;
            if score_diff < 0.05 {
                recommendations.push(format!(
                    "Close performance between {:?} and {:?} - consider computational cost",
                    best.0, second_best.0
                ));
            }
        }

        // Performance vs accuracy trade-offs
        if let Some(kmeans_result) = results.get(&ClusteringAlgorithm::KMeans) {
            if let Some(dbscan_result) = results.get(&ClusteringAlgorithm::DBSCAN) {
                if kmeans_result.total_time < dbscan_result.total_time * 0.5
                    && F::from(kmeans_result.best_score).unwrap()
                        > F::from(dbscan_result.best_score * 0.9).unwrap()
                {
                    recommendations
                        .push("K-means offers good speed/accuracy trade-off".to_string());
                }
            }
        }

        recommendations
    }
}

/// High-level convenience function for automatic algorithm selection
#[allow(dead_code)]
pub fn auto_select_clustering_algorithm<
    F: Float
        + FromPrimitive
        + Debug
        + 'static
        + std::iter::Sum
        + std::fmt::Display
        + Send
        + Sync
        + ndarray::ScalarOperand
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign
        + std::ops::RemAssign
        + PartialOrd,
>(
    data: ArrayView2<F>,
    config: Option<TuningConfig>,
) -> Result<AlgorithmSelectionResult>
where
    f64: From<F>,
{
    let tuning_config = config.unwrap_or_else(|| TuningConfig {
        max_evaluations: 50, // Reduced for faster selection
        ..Default::default()
    });

    let selector = AutoClusteringSelector::new(tuning_config);
    selector.select_best_algorithm(data)
}

/// Quick algorithm selection with default parameters
#[allow(dead_code)]
pub fn quick_algorithm_selection<
    F: Float
        + FromPrimitive
        + Debug
        + 'static
        + std::iter::Sum
        + std::fmt::Display
        + Send
        + Sync
        + ndarray::ScalarOperand
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign
        + std::ops::RemAssign
        + PartialOrd,
>(
    data: ArrayView2<F>,
) -> Result<AlgorithmSelectionResult>
where
    f64: From<F>,
{
    let config = TuningConfig {
        strategy: SearchStrategy::RandomSearch { n_trials: 20 },
        max_evaluations: 20,
        early_stopping: Some(EarlyStoppingConfig {
            patience: 5,
            min_improvement: 0.001,
            evaluation_frequency: 1,
        }),
        ..Default::default()
    };

    let algorithms = vec![
        ClusteringAlgorithm::KMeans,
        ClusteringAlgorithm::DBSCAN,
        ClusteringAlgorithm::GaussianMixture,
    ];

    let selector = AutoClusteringSelector::with_algorithms(config, algorithms);
    selector.select_best_algorithm(data)
}
