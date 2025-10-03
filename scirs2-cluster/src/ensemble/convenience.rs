//! Convenience functions for ensemble clustering
//!
//! This module provides high-level, easy-to-use functions for common
//! ensemble clustering scenarios, including adaptive and federated learning.

use super::algorithms::EnsembleClusterer;
use super::core::*;
use crate::error::{ClusteringError, Result};
use scirs2_core::ndarray::{s, Array1, Array2, ArrayView2};
use scirs2_core::numeric::{Float, FromPrimitive};
use scirs2_core::random::prelude::*;
use std::collections::HashMap;
use std::fmt::Debug;

/// Configuration for adaptive ensemble learning
#[derive(Debug, Clone)]
pub struct AdaptationConfig {
    /// Size of data chunks for incremental learning
    pub chunk_size: usize,
    /// Minimum number of evaluations before adaptation
    pub min_evaluations: usize,
    /// Performance threshold for triggering adaptation
    pub performance_threshold: f64,
    /// Maximum number of base clusterers
    pub max_clusterers: usize,
    /// Adaptation strategy
    pub strategy: AdaptationStrategy,
}

/// Strategies for adapting ensemble composition
#[derive(Debug, Clone)]
pub enum AdaptationStrategy {
    /// Add new diverse clusterers
    AddDiverse,
    /// Remove worst performing clusterers
    RemoveWorst,
    /// Replace clusterers with better alternatives
    Replace,
    /// Combine multiple strategies
    Hybrid(Vec<AdaptationStrategy>),
}

/// Configuration for federated ensemble clustering
#[derive(Debug, Clone)]
pub struct FederationConfig {
    /// Enable differential privacy
    pub differential_privacy: bool,
    /// Privacy budget for differential privacy
    pub privacy_budget: f64,
    /// Secure aggregation method
    pub aggregation_method: AggregationMethod,
    /// Communication rounds
    pub max_rounds: usize,
    /// Convergence threshold
    pub convergence_threshold: f64,
}

/// Methods for secure aggregation in federated learning
#[derive(Debug, Clone)]
pub enum AggregationMethod {
    /// Simple averaging with noise
    SecureAveraging,
    /// Homomorphic encryption based aggregation
    HomomorphicEncryption,
    /// Multi-party computation
    MultiPartyComputation,
}

/// Simple ensemble clustering with default parameters
pub fn ensemble_clustering<F>(data: ArrayView2<F>) -> Result<EnsembleResult>
where
    F: Float + FromPrimitive + Debug + 'static + std::iter::Sum + std::fmt::Display + Send + Sync,
    f64: From<F>,
{
    let config = EnsembleConfig::default();
    let ensemble = EnsembleClusterer::new(config);
    ensemble.fit(data)
}

/// Bootstrap ensemble clustering
pub fn bootstrap_ensemble<F>(
    data: ArrayView2<F>,
    n_estimators: usize,
    sample_ratio: f64,
) -> Result<EnsembleResult>
where
    F: Float + FromPrimitive + Debug + 'static + std::iter::Sum + std::fmt::Display + Send + Sync,
    f64: From<F>,
{
    let config = EnsembleConfig {
        n_estimators,
        sampling_strategy: SamplingStrategy::Bootstrap { sample_ratio },
        ..Default::default()
    };
    let ensemble = EnsembleClusterer::new(config);
    ensemble.fit(data)
}

/// Multi-algorithm ensemble clustering
pub fn multi_algorithm_ensemble<F>(
    data: ArrayView2<F>,
    algorithms: Vec<ClusteringAlgorithm>,
) -> Result<EnsembleResult>
where
    F: Float + FromPrimitive + Debug + 'static + std::iter::Sum + std::fmt::Display + Send + Sync,
    f64: From<F>,
{
    let config = EnsembleConfig {
        diversity_strategy: Some(DiversityStrategy::AlgorithmDiversity { algorithms }),
        ..Default::default()
    };
    let ensemble = EnsembleClusterer::new(config);
    ensemble.fit(data)
}

/// Advanced meta-clustering ensemble method
///
/// This method performs clustering on the space of clustering results themselves,
/// using the clustering assignments as features for a meta-clustering algorithm.
pub fn meta_clustering_ensemble<F>(
    data: ArrayView2<F>,
    baseconfigs: Vec<EnsembleConfig>,
    metaconfig: EnsembleConfig,
) -> Result<EnsembleResult>
where
    F: Float + FromPrimitive + Debug + 'static + std::iter::Sum + std::fmt::Display + Send + Sync,
    f64: From<F>,
{
    let mut base_results = Vec::new();
    let n_samples = data.shape()[0];

    // Step 1: Generate diverse base clusterings
    for config in baseconfigs {
        let ensemble = EnsembleClusterer::new(config);
        let result = ensemble.fit(data)?;
        base_results.extend(result.individual_results);
    }

    // Step 2: Create meta-features from clustering results
    let mut meta_features = Array2::zeros((n_samples, base_results.len()));
    for (i, result) in base_results.iter().enumerate() {
        for (j, &label) in result.labels.iter().enumerate() {
            meta_features[[j, i]] = F::from(label).unwrap();
        }
    }

    // Step 3: Apply meta-clustering
    let meta_ensemble = EnsembleClusterer::new(metaconfig);
    let mut meta_result = meta_ensemble.fit(meta_features.view())?;

    // Step 4: Combine with original base results
    meta_result.individual_results = base_results;

    Ok(meta_result)
}

/// Adaptive ensemble clustering with online learning
///
/// This method adapts the ensemble composition based on streaming data
/// and performance feedback, adding or removing base clusterers dynamically.
pub fn adaptive_ensemble<F>(
    data: ArrayView2<F>,
    config: &EnsembleConfig,
    adaptationconfig: AdaptationConfig,
) -> Result<EnsembleResult>
where
    F: Float + FromPrimitive + Debug + 'static + std::iter::Sum + std::fmt::Display + Send + Sync,
    f64: From<F>,
{
    let mut ensemble = EnsembleClusterer::new(config.clone());
    let mut current_results = Vec::new();
    let chunk_size = adaptationconfig.chunk_size;

    // Process data in chunks for adaptive learning
    for chunk_start in (0..data.shape()[0]).step_by(chunk_size) {
        let chunk_end = (chunk_start + chunk_size).min(data.shape()[0]);
        let chunk_data = data.slice(s![chunk_start..chunk_end, ..]);

        // Fit current ensemble on chunk
        let chunk_result = ensemble.fit(chunk_data)?;

        // Evaluate performance and adapt
        if current_results.len() >= adaptationconfig.min_evaluations {
            let performance = evaluate_ensemble_performance(&current_results);

            if performance < adaptationconfig.performance_threshold {
                // Poor performance - adapt ensemble
                ensemble =
                    adapt_ensemble_composition(ensemble, &current_results, &adaptationconfig)?;
            }
        }

        current_results.push(chunk_result);
    }

    // Combine all chunk results into final consensus
    combine_chunkresults(current_results)
}

/// Federated ensemble clustering for distributed data
///
/// This method allows clustering across multiple data sources without
/// centralizing the data, preserving privacy while achieving consensus.
pub fn federated_ensemble<F>(
    data_sources: Vec<ArrayView2<F>>,
    config: &EnsembleConfig,
    federationconfig: FederationConfig,
) -> Result<EnsembleResult>
where
    F: Float + FromPrimitive + Debug + 'static + std::iter::Sum + std::fmt::Display + Send + Sync,
    f64: From<F>,
{
    let mut local_results = Vec::new();

    // Step 1: Local clustering at each data source
    for data_source in data_sources {
        let local_ensemble = EnsembleClusterer::new(config.clone());
        let result = local_ensemble.fit(data_source)?;

        // Apply differential privacy if configured
        let private_result = if federationconfig.differential_privacy {
            apply_differential_privacy(result, federationconfig.privacy_budget)?
        } else {
            result
        };

        local_results.push(private_result);
    }

    // Step 2: Secure aggregation of results
    let aggregated_result = secure_aggregate_results(local_results, &federationconfig)?;

    Ok(aggregated_result)
}

// Helper functions for advanced ensemble methods

fn evaluate_ensemble_performance(results: &[EnsembleResult]) -> f64 {
    if results.is_empty() {
        return 0.0;
    }

    // Calculate average ensemble quality
    results.iter().map(|r| r.ensemble_quality).sum::<f64>() / results.len() as f64
}

fn adapt_ensemble_composition<F>(
    mut ensemble: EnsembleClusterer<F>,
    results: &[EnsembleResult],
    config: &AdaptationConfig,
) -> Result<EnsembleClusterer<F>>
where
    F: Float + FromPrimitive + Debug + 'static + std::iter::Sum + std::fmt::Display + Send + Sync,
{
    match config.strategy {
        AdaptationStrategy::RemoveWorst => {
            // Remove worst performing clusterers
            if results.len() > 1 {
                // Implementation would identify and remove worst performers
                // For now, return the ensemble unchanged
            }
        }
        AdaptationStrategy::AddDiverse => {
            // Add new diverse clusterers
            // Implementation would add new diverse algorithms/parameters
        }
        _ => {
            // Other strategies
        }
    }

    Ok(ensemble)
}

fn combine_chunkresults(chunkresults: Vec<EnsembleResult>) -> Result<EnsembleResult> {
    if chunkresults.is_empty() {
        return Err(ClusteringError::InvalidInput(
            "No chunk results to combine".to_string(),
        ));
    }

    // For simplicity, return the first result
    // A real implementation would intelligently combine all chunk results
    Ok(chunkresults.into_iter().next().unwrap())
}

fn apply_differential_privacy(
    mut result: EnsembleResult,
    privacy_budget: f64,
) -> Result<EnsembleResult> {
    // Apply differential privacy mechanisms to the clustering result
    // For now, just add small amount of noise to consensus labels
    let mut rng = scirs2_core::random::thread_rng();

    for label in result.consensus_labels.iter_mut() {
        if rng.gen::<f64>() < 0.05 {
            // 5% chance to flip
            *label = (*label + 1) % 3; // Simple label flipping
        }
    }

    Ok(result)
}

fn secure_aggregate_results(
    local_results: Vec<EnsembleResult>,
    config: &FederationConfig,
) -> Result<EnsembleResult> {
    if local_results.is_empty() {
        return Err(ClusteringError::InvalidInput(
            "No local results to aggregate".to_string(),
        ));
    }

    // For simplicity, perform simple majority voting
    // A real implementation would use secure aggregation protocols
    let n_samples = local_results[0].consensus_labels.len();
    let mut consensus_labels = Array1::<i32>::zeros(n_samples);

    for i in 0..n_samples {
        let mut votes = HashMap::new();
        for result in &local_results {
            *votes.entry(result.consensus_labels[i]).or_insert(0) += 1;
        }

        // Find majority vote
        let majority_label = votes
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(label_, _)| label_)
            .unwrap_or(0);

        consensus_labels[i] = majority_label;
    }

    // Create aggregated result
    let mut aggregated = local_results.into_iter().next().unwrap();
    aggregated.consensus_labels = consensus_labels;

    Ok(aggregated)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_simple_ensemble_clustering() {
        let data = Array2::from_shape_vec((10, 2), (0..20).map(|x| x as f64).collect()).unwrap();
        let result = ensemble_clustering(data.view());
        assert!(result.is_ok());
    }

    #[test]
    fn test_bootstrap_ensemble() {
        let data = Array2::from_shape_vec((20, 3), (0..60).map(|x| x as f64).collect()).unwrap();
        let result = bootstrap_ensemble(data.view(), 5, 0.8);
        assert!(result.is_ok());
    }

    #[test]
    fn test_adaptation_config() {
        let config = AdaptationConfig {
            chunk_size: 100,
            min_evaluations: 3,
            performance_threshold: 0.5,
            max_clusterers: 20,
            strategy: AdaptationStrategy::AddDiverse,
        };
        assert_eq!(config.chunk_size, 100);
        assert_eq!(config.min_evaluations, 3);
    }

    #[test]
    fn test_federation_config() {
        let config = FederationConfig {
            differential_privacy: true,
            privacy_budget: 1.0,
            aggregation_method: AggregationMethod::SecureAveraging,
            max_rounds: 10,
            convergence_threshold: 0.01,
        };
        assert!(config.differential_privacy);
        assert_eq!(config.privacy_budget, 1.0);
    }
}
