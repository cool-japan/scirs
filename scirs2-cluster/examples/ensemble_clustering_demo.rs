//! Demonstration of comprehensive ensemble clustering capabilities
//!
//! This example showcases the full range of ensemble clustering methods
//! including majority voting, weighted consensus, graph-based consensus,
//! and advanced techniques like meta-learning and Bayesian averaging.

use scirs2_cluster::{
    ensemble::{
        advanced::*, convenience, ClusteringAlgorithm, ConsensusMethod, DiversityStrategy,
        EnsembleClusterer, EnsembleConfig, SamplingStrategy,
    },
    preprocess::standardize,
};
use scirs2_core::ndarray::Array2;
use std::collections::HashMap;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Comprehensive Ensemble Clustering Demonstration");
    println!("=============================================");

    // Generate test data with multiple clusters
    let data = generate_testdata();
    println!(
        "Generated {} data points with {} features",
        data.nrows(),
        data.ncols()
    );

    // Standardize the data
    let standardized = standardize(data.view(), true)?;

    // Demo 1: Basic ensemble clustering with default settings
    println!("\n🔬 Demo 1: Basic Ensemble Clustering");
    demonstrate_basic_ensemble(&standardized)?;

    // Demo 2: Bootstrap ensemble with multiple algorithms
    println!("\n🔬 Demo 2: Bootstrap Ensemble");
    demonstrate_bootstrap_ensemble(&standardized)?;

    // Demo 3: Advanced consensus methods
    println!("\n🔬 Demo 3: Advanced Consensus Methods");
    demonstrate_consensus_methods(&standardized)?;

    // Demo 4: Multi-algorithm ensemble
    println!("\n🔬 Demo 4: Multi-Algorithm Ensemble");
    demonstrate_multi_algorithm_ensemble(&standardized)?;

    // Demo 5: Meta-clustering ensemble
    println!("\n🔬 Demo 5: Meta-Clustering Ensemble");
    demonstrate_meta_clustering_ensemble(&standardized)?;

    // Demo 6: Advanced ensemble techniques
    println!("\n🔬 Demo 6: Advanced Ensemble Techniques");
    demonstrate_advanced_ensemble_techniques(&standardized)?;

    println!("\n✅ Ensemble clustering demonstration completed!");
    println!("   All ensemble methods successfully demonstrated diverse clustering approaches");
    println!("   and consensus strategies for improved robustness and stability.");

    Ok(())
}

/// Generate synthetic test data with multiple clusters
#[allow(dead_code)]
fn generate_testdata() -> Array2<f64> {
    let mut data = Vec::new();

    // Cluster 1: centered at (2, 2)
    for _ in 0..30 {
        data.push(2.0 + (scirs2_core::random::random::<f64>() - 0.5) * 1.0);
        data.push(2.0 + (scirs2_core::random::random::<f64>() - 0.5) * 1.0);
    }

    // Cluster 2: centered at (-2, 2)
    for _ in 0..30 {
        data.push(-2.0 + (scirs2_core::random::random::<f64>() - 0.5) * 1.0);
        data.push(2.0 + (scirs2_core::random::random::<f64>() - 0.5) * 1.0);
    }

    // Cluster 3: centered at (0, -2)
    for _ in 0..30 {
        data.push(0.0 + (scirs2_core::random::random::<f64>() - 0.5) * 1.0);
        data.push(-2.0 + (scirs2_core::random::random::<f64>() - 0.5) * 1.0);
    }

    Array2::from_shape_vec((90, 2), data).unwrap()
}

/// Demonstrate basic ensemble clustering
#[allow(dead_code)]
fn demonstrate_basic_ensemble(data: &Array2<f64>) -> Result<(), Box<dyn std::error::Error>> {
    let result = convenience::ensemble_clustering(data.view())?;

    println!("   📊 Basic ensemble results:");
    println!("      Ensemble quality: {:.4}", result.ensemble_quality);
    println!("      Stability score: {:.4}", result.stability_score);
    println!(
        "      Number of base clusterers: {}",
        result.individual_results.len()
    );
    println!(
        "      Average diversity: {:.4}",
        result.diversity_metrics.average_diversity
    );
    println!(
        "      Final clusters found: {}",
        count_unique_labels(&result.consensus_labels)
    );

    Ok(())
}

/// Demonstrate bootstrap ensemble clustering
#[allow(dead_code)]
fn demonstrate_bootstrap_ensemble(data: &Array2<f64>) -> Result<(), Box<dyn std::error::Error>> {
    let result = convenience::bootstrap_ensemble(data.view(), 8, 0.75)?;

    println!("   📊 Bootstrap ensemble results:");
    println!("      Ensemble quality: {:.4}", result.ensemble_quality);
    println!("      Stability score: {:.4}", result.stability_score);
    println!(
        "      Algorithm diversity: {:.4}",
        result.diversity_metrics.algorithm_distribution.len() as f64
            / result.individual_results.len() as f64
    );
    println!(
        "      Final clusters found: {}",
        count_unique_labels(&result.consensus_labels)
    );

    Ok(())
}

/// Demonstrate different consensus methods
#[allow(dead_code)]
fn demonstrate_consensus_methods(data: &Array2<f64>) -> Result<(), Box<dyn std::error::Error>> {
    let consensus_methods = vec![
        ("Majority Voting", ConsensusMethod::MajorityVoting),
        ("Weighted Consensus", ConsensusMethod::WeightedConsensus),
        (
            "Graph-Based",
            ConsensusMethod::GraphBased {
                similarity_threshold: 0.6,
            },
        ),
        (
            "Co-Association",
            ConsensusMethod::CoAssociation { threshold: 0.5 },
        ),
        (
            "Evidence Accumulation",
            ConsensusMethod::EvidenceAccumulation,
        ),
        (
            "Hierarchical",
            ConsensusMethod::Hierarchical {
                linkage_method: "ward".to_string(),
            },
        ),
    ];

    for (name, method) in consensus_methods {
        let config = EnsembleConfig {
            n_estimators: 6,
            consensus_method: method,
            sampling_strategy: SamplingStrategy::Bootstrap { sample_ratio: 0.8 },
            ..Default::default()
        };

        let ensemble = EnsembleClusterer::new(config);
        let result = ensemble.fit(data.view())?;

        println!("   📊 {} consensus:", name);
        println!(
            "      Quality: {:.4}, Stability: {:.4}, Clusters: {}",
            result.ensemble_quality,
            result.stability_score,
            count_unique_labels(&result.consensus_labels)
        );
    }

    Ok(())
}

/// Demonstrate multi-algorithm ensemble
#[allow(dead_code)]
fn demonstrate_multi_algorithm_ensemble(
    data: &Array2<f64>,
) -> Result<(), Box<dyn std::error::Error>> {
    let algorithms = vec![
        ClusteringAlgorithm::KMeans { k_range: (2, 5) },
        ClusteringAlgorithm::DBSCAN {
            eps_range: (0.3, 0.8),
            min_samples_range: (3, 7),
        },
        ClusteringAlgorithm::MeanShift {
            bandwidth_range: (0.5, 1.5),
        },
        ClusteringAlgorithm::Hierarchical {
            methods: vec!["ward".to_string(), "complete".to_string()],
        },
    ];

    let result = convenience::multi_algorithm_ensemble(data.view(), algorithms)?;

    println!("   📊 Multi-algorithm ensemble results:");
    println!("      Ensemble quality: {:.4}", result.ensemble_quality);
    println!("      Stability score: {:.4}", result.stability_score);
    println!(
        "      Algorithm types used: {}",
        result.diversity_metrics.algorithm_distribution.len()
    );
    println!(
        "      Final clusters found: {}",
        count_unique_labels(&result.consensus_labels)
    );

    // Show algorithm distribution
    println!("      Algorithm distribution:");
    for (alg, count) in &result.diversity_metrics.algorithm_distribution {
        println!("        {}: {} instances", alg, count);
    }

    Ok(())
}

/// Demonstrate meta-clustering ensemble
#[allow(dead_code)]
fn demonstrate_meta_clustering_ensemble(
    data: &Array2<f64>,
) -> Result<(), Box<dyn std::error::Error>> {
    let base_configs = vec![
        EnsembleConfig {
            n_estimators: 4,
            sampling_strategy: SamplingStrategy::Bootstrap { sample_ratio: 0.7 },
            diversity_strategy: Some(DiversityStrategy::AlgorithmDiversity {
                algorithms: vec![ClusteringAlgorithm::KMeans { k_range: (2, 4) }],
            }),
            ..Default::default()
        },
        EnsembleConfig {
            n_estimators: 4,
            sampling_strategy: SamplingStrategy::RandomSubspace { feature_ratio: 0.8 },
            diversity_strategy: Some(DiversityStrategy::AlgorithmDiversity {
                algorithms: vec![ClusteringAlgorithm::DBSCAN {
                    eps_range: (0.3, 0.7),
                    min_samples_range: (3, 6),
                }],
            }),
            ..Default::default()
        },
    ];

    let meta_config = EnsembleConfig {
        n_estimators: 3,
        consensus_method: ConsensusMethod::WeightedConsensus,
        ..Default::default()
    };

    let result = convenience::meta_clustering_ensemble(data.view(), base_configs, meta_config)?;

    println!("   📊 Meta-clustering ensemble results:");
    println!("      Ensemble quality: {:.4}", result.ensemble_quality);
    println!("      Stability score: {:.4}", result.stability_score);
    println!(
        "      Total base clusterers: {}",
        result.individual_results.len()
    );
    println!(
        "      Final clusters found: {}",
        count_unique_labels(&result.consensus_labels)
    );

    Ok(())
}

/// Demonstrate advanced ensemble techniques
#[allow(dead_code)]
fn demonstrate_advanced_ensemble_techniques(
    data: &Array2<f64>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Create advanced ensemble configuration
    let advanced_config = AdvancedEnsembleConfig {
        meta_learning: MetaLearningConfig {
            n_meta_features: 8,
            learning_rate: 0.01,
            n_iterations: 50,
            algorithm: MetaLearningAlgorithm::NeuralNetwork {
                hidden_layers: vec![16, 8],
            },
            validation_split: 0.2,
        },
        bayesian_averaging: BayesianAveragingConfig {
            prior_alpha: 1.0,
            prior_beta: 1.0,
            n_samples: 100,
            burn_in: 20,
            update_method: PosteriorUpdateMethod::MetropolisHastings,
            adaptive_sampling: true,
        },
        genetic_optimization: GeneticOptimizationConfig {
            population_size: 20,
            n_generations: 10,
            crossover_prob: 0.8,
            mutation_prob: 0.1,
            selection_method: SelectionMethod::Tournament { tournament_size: 3 },
            elite_percentage: 0.1,
            fitness_function: FitnessFunction::Silhouette,
        },
        boostingconfig: BoostingConfig {
            n_rounds: 5,
            learning_rate: 0.1,
            reweighting_strategy: ReweightingStrategy::Exponential,
            error_function: ErrorFunction::DisagreementRate,
            adaptive_boosting: true,
        },
        stackingconfig: StackingConfig {
            base_algorithms: vec![
                ClusteringAlgorithm::KMeans { k_range: (2, 4) },
                ClusteringAlgorithm::DBSCAN {
                    eps_range: (0.3, 0.7),
                    min_samples_range: (3, 6),
                },
            ],
            meta_algorithm: MetaClusteringAlgorithm::Hierarchical {
                linkage: "ward".to_string(),
            },
            cv_folds: 3,
            blending_ratio: 0.7,
            feature_engineering: true,
        },
        uncertainty_quantification: true,
    };

    let base_config = EnsembleConfig {
        n_estimators: 6,
        sampling_strategy: SamplingStrategy::BootstrapSubspace {
            sample_ratio: 0.8,
            feature_ratio: 0.9,
        },
        ..Default::default()
    };

    let mut advanced_ensemble = AdvancedEnsembleClusterer::new(advanced_config, base_config);

    // Test meta-learning ensemble
    match advanced_ensemble.fit_with_meta_learning(data.view()) {
        Ok(result) => {
            println!("   📊 Meta-learning ensemble results:");
            println!("      Ensemble quality: {:.4}", result.ensemble_quality);
            println!("      Stability score: {:.4}", result.stability_score);
            println!(
                "      Final clusters found: {}",
                count_unique_labels(&result.consensus_labels)
            );
        }
        Err(e) => println!("   ⚠️  Meta-learning ensemble: {}", e),
    }

    // Test Bayesian model averaging
    match advanced_ensemble.fit_with_bayesian_averaging(data.view()) {
        Ok(result) => {
            println!("   📊 Bayesian averaging results:");
            println!("      Ensemble quality: {:.4}", result.ensemble_quality);
            println!("      Stability score: {:.4}", result.stability_score);
            println!(
                "      Final clusters found: {}",
                count_unique_labels(&result.consensus_labels)
            );
        }
        Err(e) => println!("   ⚠️  Bayesian averaging: {}", e),
    }

    // Test boosting ensemble
    match advanced_ensemble.fit_with_boosting(data.view()) {
        Ok(result) => {
            println!("   📊 Boosting ensemble results:");
            println!("      Ensemble quality: {:.4}", result.ensemble_quality);
            println!("      Stability score: {:.4}", result.stability_score);
            println!(
                "      Final clusters found: {}",
                count_unique_labels(&result.consensus_labels)
            );
        }
        Err(e) => println!("   ⚠️  Boosting ensemble: {}", e),
    }

    Ok(())
}

/// Count unique labels in consensus result
#[allow(dead_code)]
fn count_unique_labels(labels: &scirs2_core::ndarray::Array1<i32>) -> usize {
    use std::collections::HashSet;
    let unique_labels: HashSet<i32> = labels.iter().cloned().collect();
    unique_labels.len()
}
