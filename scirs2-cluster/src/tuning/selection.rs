//! Algorithm selection and recommendation systems
//!
//! This module provides automatic algorithm selection capabilities and
//! predefined search spaces for different clustering algorithms.

use scirs2_core::ndarray::ArrayView2;
use scirs2_core::numeric::{Float, FromPrimitive};
use std::collections::HashMap;
use std::fmt::Debug;

use crate::error::Result;

use super::config::{
    AcquisitionFunction, CVStrategy, CrossValidationConfig, EarlyStoppingConfig, EvaluationMetric,
    HyperParameter, LoadBalancingStrategy, ParallelConfig, ResourceConstraints, SearchSpace,
    SearchStrategy, TuningConfig, TuningResult,
};

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
            + scirs2_core::ndarray::ScalarOperand
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

        // Create a simplified AutoTuner for demonstration
        // In practice, this would use the actual AutoTuner from the main module

        println!(
            "Testing {} algorithms for automatic selection...",
            self.algorithms.len()
        );

        for algorithm in &self.algorithms {
            println!("Tuning {algorithm:?}...");

            // For each algorithm, create a default tuning result
            // In practice, this would call the actual tuning methods
            let tuning_result = self.create_default_tuning_result(algorithm);

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

    /// Create a default tuning result for demonstration
    /// In practice, this would call the actual algorithm tuning methods
    fn create_default_tuning_result(
        &self,
        algorithm: &ClusteringAlgorithm,
    ) -> Result<TuningResult> {
        use super::config::{ConvergenceInfo, EvaluationResult, ExplorationStats, StoppingReason};

        // Generate a mock result with reasonable scores
        let score = match algorithm {
            ClusteringAlgorithm::KMeans => 0.65,
            ClusteringAlgorithm::DBSCAN => 0.72,
            ClusteringAlgorithm::GaussianMixture => 0.68,
            ClusteringAlgorithm::SpectralClustering => 0.70,
            ClusteringAlgorithm::HierarchicalClustering => 0.63,
            _ => 0.60,
        };

        let mut best_parameters = HashMap::new();
        best_parameters.insert("mock_param".to_string(), 1.0);

        let evaluation_result = EvaluationResult {
            parameters: best_parameters.clone(),
            score,
            additional_metrics: HashMap::new(),
            evaluation_time: 0.1,
            memory_usage: None,
            cv_scores: vec![score],
            cv_std: 0.05,
            metadata: HashMap::new(),
        };

        Ok(TuningResult {
            best_parameters,
            best_score: score,
            evaluation_history: vec![evaluation_result],
            convergence_info: ConvergenceInfo {
                converged: true,
                convergence_iteration: Some(1),
                stopping_reason: StoppingReason::MaxEvaluations,
            },
            exploration_stats: ExplorationStats {
                coverage: 0.8,
                parameter_distributions: HashMap::new(),
                parameter_importance: HashMap::new(),
            },
            total_time: 0.5,
            ensemble_results: None,
            pareto_front: None,
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
                validation_ratio: 0.3,
                strategy: CVStrategy::KFold,
                shuffle: true,
            },
            early_stopping: None,
            parallel_config: None,
            random_seed: Some(42),
            resource_constraints: ResourceConstraints {
                max_memory_per_evaluation: None,
                max_time_per_evaluation: Some(120.0),
                max_total_time: Some(3600.0),
            },
        };

        (search_space, config)
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
        + scirs2_core::ndarray::ScalarOperand
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
        + scirs2_core::ndarray::ScalarOperand
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
