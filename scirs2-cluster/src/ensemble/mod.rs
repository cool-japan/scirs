//! Ensemble clustering algorithms
//!
//! This module provides comprehensive ensemble clustering capabilities that combine
//! multiple base clustering algorithms to achieve more robust and stable results.
//!
//! # Examples
//!
//! ## Basic Ensemble Clustering
//!
//! ```rust
//! use scirs2_cluster::ensemble::{EnsembleClusterer, EnsembleConfig, SamplingStrategy};
//! use ndarray::Array2;
//!
//! // Create sample data
//! let data = Array2::from_shape_vec((100, 2), (0..200).map(|x| x as f64).collect()).unwrap();
//!
//! // Configure ensemble
//! let config = EnsembleConfig {
//!     n_estimators: 10,
//!     sampling_strategy: SamplingStrategy::Bootstrap { sample_ratio: 0.8 },
//!     ..Default::default()
//! };
//!
//! // Create and fit ensemble
//! let ensemble = EnsembleClusterer::new(config);
//! let result = ensemble.fit(data.view()).unwrap();
//! println!("Ensemble quality: {}", result.ensemble_quality);
//! ```
//!
//! ## Convenience Functions
//!
//! ```rust
//! use scirs2_cluster::ensemble::convenience::ensemble_clustering;
//! use ndarray::Array2;
//!
//! let data = Array2::from_shape_vec((50, 3), (0..150).map(|x| x as f64).collect()).unwrap();
//! let result = ensemble_clustering(data.view()).unwrap();
//! ```
//!
//! ## Advanced Ensemble Methods
//!
//! ```rust,no_run
//! use scirs2_cluster::ensemble::advanced::{AdvancedEnsembleClusterer, AdvancedEnsembleConfig};
//! use scirs2_cluster::ensemble::{
//!     EnsembleConfig,
//!     MetaLearningConfig, MetaLearningAlgorithm,
//!     BayesianAveragingConfig, PosteriorUpdateMethod,
//!     GeneticOptimizationConfig, SelectionMethod, FitnessFunction,
//!     BoostingConfig, ReweightingStrategy, ErrorFunction,
//!     StackingConfig, MetaClusteringAlgorithm, ClusteringAlgorithm,
//! };
//! use ndarray::Array2;
//!
//! // Advanced ensemble with meta-learning
//! let data = Array2::from_shape_vec((100, 4), (0..400).map(|x| x as f64).collect()).unwrap();
//! let base_config = EnsembleConfig::default();
//! let advanced_config = AdvancedEnsembleConfig {
//!     meta_learning: MetaLearningConfig {
//!         n_meta_features: 4,
//!         learning_rate: 0.01,
//!         n_iterations: 10,
//!         algorithm: MetaLearningAlgorithm::Linear { regularization: 0.1 },
//!         validation_split: 0.2,
//!     },
//!     bayesian_averaging: BayesianAveragingConfig {
//!         prior_alpha: 1.0,
//!         prior_beta: 1.0,
//!         n_samples: 10,
//!         burn_in: 2,
//!         update_method: PosteriorUpdateMethod::MetropolisHastings,
//!         adaptive_sampling: false,
//!     },
//!     genetic_optimization: GeneticOptimizationConfig {
//!         population_size: 5,
//!         n_generations: 2,
//!         crossover_prob: 0.8,
//!         mutation_prob: 0.1,
//!         selection_method: SelectionMethod::Tournament { tournament_size: 2 },
//!         elite_percentage: 0.1,
//!         fitness_function: FitnessFunction::Silhouette,
//!     },
//!     boostingconfig: BoostingConfig {
//!         n_rounds: 3,
//!         learning_rate: 1.0,
//!         reweighting_strategy: ReweightingStrategy::Exponential,
//!         error_function: ErrorFunction::DisagreementRate,
//!         adaptive_boosting: false,
//!     },
//!     stackingconfig: StackingConfig {
//!         base_algorithms: vec![ClusteringAlgorithm::KMeans { k_range: (2, 4) }],
//!         meta_algorithm: MetaClusteringAlgorithm::Hierarchical { linkage: "ward".into() },
//!         cv_folds: 2,
//!         blending_ratio: 0.5,
//!         feature_engineering: false,
//!     },
//!     uncertainty_quantification: false,
//! };
//! let mut advanced_ensemble = AdvancedEnsembleClusterer::new(advanced_config, base_config);
//! let result = advanced_ensemble.fit_with_meta_learning(data.view()).unwrap();
//! ```

pub mod advanced;
pub mod algorithms;
pub mod convenience;
pub mod core;

// Re-export main types for convenience
pub use algorithms::EnsembleClusterer;
pub use core::*;

// Re-export convenience functions at module level for backward compatibility
pub use convenience::{
    adaptive_ensemble, bootstrap_ensemble, ensemble_clustering, federated_ensemble,
    meta_clustering_ensemble, multi_algorithm_ensemble, AdaptationConfig, AdaptationStrategy,
    AggregationMethod, FederationConfig,
};

// Re-export advanced types
pub use advanced::{
    AdvancedEnsembleClusterer, AdvancedEnsembleConfig, BayesianAveragingConfig, BoostingConfig,
    ErrorFunction, FitnessFunction, GeneticOptimizationConfig, GeneticOptimizer,
    MetaClusteringAlgorithm, MetaLearner, MetaLearningAlgorithm, MetaLearningConfig,
    PosteriorUpdateMethod, ReweightingStrategy, SelectionMethod, StackingConfig,
};

// Maintain backward compatibility by re-exporting the convenience module
pub mod convenience_functions {
    pub use super::convenience::*;
}

/// Convenience function to create a default ensemble configuration
pub fn default_ensemble_config() -> EnsembleConfig {
    EnsembleConfig::default()
}

/// Convenience function to create a bootstrap ensemble configuration
pub fn bootstrap_ensemble_config(n_estimators: usize, sample_ratio: f64) -> EnsembleConfig {
    EnsembleConfig {
        n_estimators,
        sampling_strategy: SamplingStrategy::Bootstrap { sample_ratio },
        ..Default::default()
    }
}

/// Convenience function to create an algorithm diversity configuration
pub fn algorithm_diversity_config(algorithms: Vec<ClusteringAlgorithm>) -> EnsembleConfig {
    EnsembleConfig {
        diversity_strategy: Some(DiversityStrategy::AlgorithmDiversity { algorithms }),
        ..Default::default()
    }
}

/// Convenience function to create a weighted consensus configuration
pub fn weighted_consensus_config() -> EnsembleConfig {
    EnsembleConfig {
        consensus_method: ConsensusMethod::WeightedConsensus,
        ..Default::default()
    }
}

/// Convenience function to create a graph-based consensus configuration
pub fn graph_based_consensus_config(similarity_threshold: f64) -> EnsembleConfig {
    EnsembleConfig {
        consensus_method: ConsensusMethod::GraphBased {
            similarity_threshold,
        },
        ..Default::default()
    }
}

/// Convenience function for quick ensemble clustering with default parameters
pub fn quick_ensemble_clustering<F>(
    data: ndarray::ArrayView2<F>,
    n_estimators: Option<usize>,
) -> crate::error::Result<EnsembleResult>
where
    F: num_traits::Float
        + num_traits::FromPrimitive
        + std::fmt::Debug
        + 'static
        + std::iter::Sum
        + std::fmt::Display
        + Send
        + Sync,
    f64: From<F>,
{
    let config = EnsembleConfig {
        n_estimators: n_estimators.unwrap_or(10),
        ..Default::default()
    };
    let ensemble = EnsembleClusterer::new(config);
    ensemble.fit(data)
}

/// Convenience function for multi-algorithm ensemble with common algorithms
pub fn quick_multi_algorithm_ensemble<F>(
    data: ndarray::ArrayView2<F>,
) -> crate::error::Result<EnsembleResult>
where
    F: num_traits::Float
        + num_traits::FromPrimitive
        + std::fmt::Debug
        + 'static
        + std::iter::Sum
        + std::fmt::Display
        + Send
        + Sync,
    f64: From<F>,
{
    let algorithms = vec![
        ClusteringAlgorithm::KMeans { k_range: (2, 8) },
        ClusteringAlgorithm::DBSCAN {
            eps_range: (0.1, 1.0),
            min_samples_range: (3, 10),
        },
        ClusteringAlgorithm::AffinityPropagation {
            damping_range: (0.5, 0.9),
        },
    ];

    multi_algorithm_ensemble(data, algorithms)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_default_ensemble_config() {
        let config = default_ensemble_config();
        assert_eq!(config.n_estimators, 10);
        assert!(matches!(
            config.sampling_strategy,
            SamplingStrategy::Bootstrap { .. }
        ));
        assert!(matches!(
            config.consensus_method,
            ConsensusMethod::MajorityVoting
        ));
    }

    #[test]
    fn test_bootstrap_ensemble_config() {
        let config = bootstrap_ensemble_config(15, 0.7);
        assert_eq!(config.n_estimators, 15);
        if let SamplingStrategy::Bootstrap { sample_ratio } = config.sampling_strategy {
            assert!((sample_ratio - 0.7).abs() < 1e-10);
        } else {
            panic!("Expected Bootstrap sampling strategy");
        }
    }

    #[test]
    fn test_algorithm_diversity_config() {
        let algorithms = vec![
            ClusteringAlgorithm::KMeans { k_range: (2, 5) },
            ClusteringAlgorithm::DBSCAN {
                eps_range: (0.1, 1.0),
                min_samples_range: (3, 10),
            },
        ];
        let config = algorithm_diversity_config(algorithms.clone());

        if let Some(DiversityStrategy::AlgorithmDiversity { algorithms: algs }) =
            config.diversity_strategy
        {
            assert_eq!(algs.len(), 2);
        } else {
            panic!("Expected AlgorithmDiversity strategy");
        }
    }

    #[test]
    fn test_weighted_consensus_config() {
        let config = weighted_consensus_config();
        assert!(matches!(
            config.consensus_method,
            ConsensusMethod::WeightedConsensus
        ));
    }

    #[test]
    fn test_graph_based_consensus_config() {
        let config = graph_based_consensus_config(0.7);
        if let ConsensusMethod::GraphBased {
            similarity_threshold,
        } = config.consensus_method
        {
            assert!((similarity_threshold - 0.7).abs() < 1e-10);
        } else {
            panic!("Expected GraphBased consensus method");
        }
    }

    #[test]
    fn test_quick_ensemble_clustering() {
        let data = Array2::from_shape_vec((20, 2), (0..40).map(|x| x as f64).collect()).unwrap();
        let result = quick_ensemble_clustering(data.view(), Some(5));
        assert!(result.is_ok());

        let ensemble_result = result.unwrap();
        assert_eq!(ensemble_result.consensus_labels.len(), 20);
        assert_eq!(ensemble_result.individual_results.len(), 5);
    }

    #[test]
    fn test_quick_multi_algorithm_ensemble() {
        let data = Array2::from_shape_vec((30, 3), (0..90).map(|x| x as f64).collect()).unwrap();
        let result = quick_multi_algorithm_ensemble(data.view());
        assert!(result.is_ok());

        let ensemble_result = result.unwrap();
        assert_eq!(ensemble_result.consensus_labels.len(), 30);
    }

    #[test]
    fn test_ensemble_result_metrics() {
        let data = Array2::from_shape_vec((15, 2), (0..30).map(|x| x as f64).collect()).unwrap();
        let result = quick_ensemble_clustering(data.view(), Some(3));
        assert!(result.is_ok());

        let ensemble_result = result.unwrap();
        assert!(ensemble_result.ensemble_quality >= -1.0);
        assert!(ensemble_result.ensemble_quality <= 1.0);
        assert!(ensemble_result.stability_score >= 0.0);
        assert!(ensemble_result.stability_score <= 1.0);
    }

    #[test]
    fn test_consensus_statistics() {
        let data = Array2::from_shape_vec((10, 2), (0..20).map(|x| x as f64).collect()).unwrap();
        let result = quick_ensemble_clustering(data.view(), Some(3));
        assert!(result.is_ok());

        let ensemble_result = result.unwrap();
        let consensus_stats = &ensemble_result.consensus_stats;

        assert_eq!(consensus_stats.consensus_strength.len(), 10);
        assert_eq!(consensus_stats.agreement_counts.len(), 10);
        assert!(consensus_stats.average_consensus_strength() >= 0.0);
        assert!(consensus_stats.average_consensus_strength() <= 1.0);
    }

    #[test]
    fn test_diversity_metrics() {
        let data = Array2::from_shape_vec((12, 2), (0..24).map(|x| x as f64).collect()).unwrap();
        let result = quick_ensemble_clustering(data.view(), Some(4));
        assert!(result.is_ok());

        let ensemble_result = result.unwrap();
        let diversity_metrics = &ensemble_result.diversity_metrics;

        assert!(diversity_metrics.average_diversity >= 0.0);
        assert!(diversity_metrics.average_diversity <= 1.0);
        assert_eq!(diversity_metrics.diversity_matrix.nrows(), 4);
        assert_eq!(diversity_metrics.diversity_matrix.ncols(), 4);
    }

    #[test]
    fn test_ensemble_clusterer_creation() {
        let config = EnsembleConfig::default();
        let ensemble: EnsembleClusterer<f64> = EnsembleClusterer::new(config.clone());

        // Test that the ensemble can be created with different configurations
        let custom_config = EnsembleConfig {
            n_estimators: 20,
            sampling_strategy: SamplingStrategy::RandomSubspace { feature_ratio: 0.5 },
            consensus_method: ConsensusMethod::WeightedConsensus,
            random_seed: Some(42),
            diversity_strategy: Some(DiversityStrategy::AlgorithmDiversity {
                algorithms: vec![ClusteringAlgorithm::KMeans { k_range: (2, 10) }],
            }),
            quality_threshold: Some(0.1),
            max_clusters: Some(15),
        };

        let custom_ensemble: EnsembleClusterer<f64> = EnsembleClusterer::new(custom_config);

        // Both ensembles should be creatable without errors
        assert!(true); // If we get here, creation succeeded
    }
}
