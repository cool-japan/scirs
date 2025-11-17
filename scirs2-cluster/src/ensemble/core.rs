//! Core types and configurations for ensemble clustering
//!
//! This module provides the fundamental data structures and configurations
//! used throughout the ensemble clustering system.

use scirs2_core::ndarray::ArrayStatCompat;
use scirs2_core::ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for ensemble clustering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsembleConfig {
    /// Number of base clustering algorithms to use
    pub n_estimators: usize,
    /// Sampling strategy for data subsets
    pub sampling_strategy: SamplingStrategy,
    /// Consensus method for combining results
    pub consensus_method: ConsensusMethod,
    /// Random seed for reproducible results
    pub random_seed: Option<u64>,
    /// Diversity enforcement strategy
    pub diversity_strategy: Option<DiversityStrategy>,
    /// Quality threshold for including results
    pub quality_threshold: Option<f64>,
    /// Maximum number of clusters to consider
    pub max_clusters: Option<usize>,
}

/// Sampling strategies for creating diverse datasets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SamplingStrategy {
    /// Bootstrap sampling with replacement
    Bootstrap { sample_ratio: f64 },
    /// Random subspace sampling (feature selection)
    RandomSubspace { feature_ratio: f64 },
    /// Combined bootstrap and subspace sampling
    BootstrapSubspace {
        sample_ratio: f64,
        feature_ratio: f64,
    },
    /// Random projection to lower dimensions
    RandomProjection { target_dimensions: usize },
    /// Noise injection for robustness testing
    NoiseInjection {
        noise_level: f64,
        noise_type: NoiseType,
    },
    /// No sampling (use full dataset)
    None,
}

/// Types of noise for injection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NoiseType {
    /// Gaussian noise
    Gaussian,
    /// Uniform noise
    Uniform,
    /// Outlier injection
    Outliers { outlier_ratio: f64 },
}

/// Methods for combining clustering results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsensusMethod {
    /// Simple majority voting
    MajorityVoting,
    /// Weighted consensus based on quality scores
    WeightedConsensus,
    /// Graph-based consensus clustering
    GraphBased { similarity_threshold: f64 },
    /// Hierarchical consensus
    Hierarchical { linkage_method: String },
    /// Co-association matrix approach
    CoAssociation { threshold: f64 },
    /// Evidence accumulation clustering
    EvidenceAccumulation,
}

/// Strategies for enforcing diversity among base clusterers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DiversityStrategy {
    /// Algorithm diversity (use different algorithms)
    AlgorithmDiversity {
        algorithms: Vec<ClusteringAlgorithm>,
    },
    /// Parameter diversity (same algorithm, different parameters)
    ParameterDiversity {
        algorithm: ClusteringAlgorithm,
        parameter_ranges: HashMap<String, ParameterRange>,
    },
    /// Data diversity (different data subsets)
    DataDiversity {
        sampling_strategies: Vec<SamplingStrategy>,
    },
    /// Combined diversity strategy
    Combined { strategies: Vec<DiversityStrategy> },
}

/// Supported clustering algorithms for ensemble
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ClusteringAlgorithm {
    /// K-means clustering
    KMeans { k_range: (usize, usize) },
    /// DBSCAN clustering
    DBSCAN {
        eps_range: (f64, f64),
        min_samples_range: (usize, usize),
    },
    /// Mean shift clustering
    MeanShift { bandwidth_range: (f64, f64) },
    /// Hierarchical clustering
    Hierarchical { methods: Vec<String> },
    /// Spectral clustering
    Spectral { k_range: (usize, usize) },
    /// Affinity propagation
    AffinityPropagation { damping_range: (f64, f64) },
}

/// Parameter ranges for diversity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterRange {
    /// Integer range
    Integer(i64, i64),
    /// Float range
    Float(f64, f64),
    /// Categorical choices
    Categorical(Vec<String>),
    /// Boolean choice
    Boolean,
}

/// Result of a single clustering run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusteringResult {
    /// Cluster labels
    pub labels: Array1<i32>,
    /// Algorithm used
    pub algorithm: String,
    /// Parameters used
    pub parameters: HashMap<String, String>,
    /// Quality score
    pub quality_score: f64,
    /// Stability score (if available)
    pub stability_score: Option<f64>,
    /// Number of clusters found
    pub n_clusters: usize,
    /// Runtime in seconds
    pub runtime: f64,
}

impl ClusteringResult {
    /// Create a new clustering result
    pub fn new(
        labels: Array1<i32>,
        algorithm: String,
        parameters: HashMap<String, String>,
        quality_score: f64,
        runtime: f64,
    ) -> Self {
        let n_clusters = labels
            .iter()
            .copied()
            .filter(|&x| x >= 0)
            .max()
            .map(|x| x as usize + 1)
            .unwrap_or(0);

        Self {
            labels,
            algorithm,
            parameters,
            quality_score,
            stability_score: None,
            n_clusters,
            runtime,
        }
    }

    /// Set stability score
    pub fn with_stability_score(mut self, score: f64) -> Self {
        self.stability_score = Some(score);
        self
    }

    /// Check if this result has noise points
    pub fn has_noise(&self) -> bool {
        self.labels.iter().any(|&x| x < 0)
    }

    /// Get number of noise points
    pub fn noise_count(&self) -> usize {
        self.labels.iter().filter(|&&x| x < 0).count()
    }

    /// Get cluster sizes
    pub fn cluster_sizes(&self) -> Vec<usize> {
        let mut sizes = vec![0; self.n_clusters];
        for &label in self.labels.iter() {
            if label >= 0 {
                let cluster_id = label as usize;
                if cluster_id < sizes.len() {
                    sizes[cluster_id] += 1;
                }
            }
        }
        sizes
    }
}

/// Ensemble clustering result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsembleResult {
    /// Final consensus labels
    pub consensus_labels: Array1<i32>,
    /// Individual clustering results
    pub individual_results: Vec<ClusteringResult>,
    /// Consensus statistics
    pub consensus_stats: ConsensusStatistics,
    /// Diversity metrics
    pub diversity_metrics: DiversityMetrics,
    /// Overall quality score
    pub ensemble_quality: f64,
    /// Stability score
    pub stability_score: f64,
}

impl EnsembleResult {
    /// Create a new ensemble result
    pub fn new(
        consensus_labels: Array1<i32>,
        individual_results: Vec<ClusteringResult>,
        consensus_stats: ConsensusStatistics,
        diversity_metrics: DiversityMetrics,
        ensemble_quality: f64,
        stability_score: f64,
    ) -> Self {
        Self {
            consensus_labels,
            individual_results,
            consensus_stats,
            diversity_metrics,
            ensemble_quality,
            stability_score,
        }
    }

    /// Get number of consensus clusters
    pub fn n_consensus_clusters(&self) -> usize {
        self.consensus_labels
            .iter()
            .copied()
            .filter(|&x| x >= 0)
            .max()
            .map(|x| x as usize + 1)
            .unwrap_or(0)
    }

    /// Get consensus cluster sizes
    pub fn consensus_cluster_sizes(&self) -> Vec<usize> {
        let n_clusters = self.n_consensus_clusters();
        let mut sizes = vec![0; n_clusters];
        for &label in self.consensus_labels.iter() {
            if label >= 0 {
                let cluster_id = label as usize;
                if cluster_id < sizes.len() {
                    sizes[cluster_id] += 1;
                }
            }
        }
        sizes
    }

    /// Get average quality of individual results
    pub fn average_individual_quality(&self) -> f64 {
        if self.individual_results.is_empty() {
            0.0
        } else {
            self.individual_results
                .iter()
                .map(|r| r.quality_score)
                .sum::<f64>()
                / self.individual_results.len() as f64
        }
    }

    /// Get best individual result
    pub fn best_individual_result(&self) -> Option<&ClusteringResult> {
        self.individual_results.iter().max_by(|a, b| {
            a.quality_score
                .partial_cmp(&b.quality_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Get algorithm distribution
    pub fn algorithm_distribution(&self) -> HashMap<String, usize> {
        let mut distribution = HashMap::new();
        for result in &self.individual_results {
            *distribution.entry(result.algorithm.clone()).or_insert(0) += 1;
        }
        distribution
    }
}

/// Statistics about the consensus process
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusStatistics {
    /// Agreement matrix between clusterers
    pub agreement_matrix: Array2<f64>,
    /// Per-sample consensus strength
    pub consensus_strength: Array1<f64>,
    /// Cluster stability scores
    pub cluster_stability: Vec<f64>,
    /// Number of clusterers agreeing on each sample
    pub agreement_counts: Array1<usize>,
}

impl ConsensusStatistics {
    /// Create new consensus statistics
    pub fn new(
        agreement_matrix: Array2<f64>,
        consensus_strength: Array1<f64>,
        cluster_stability: Vec<f64>,
        agreement_counts: Array1<usize>,
    ) -> Self {
        Self {
            agreement_matrix,
            consensus_strength,
            cluster_stability,
            agreement_counts,
        }
    }

    /// Get average consensus strength
    pub fn average_consensus_strength(&self) -> f64 {
        self.consensus_strength.mean_or(0.0)
    }

    /// Get minimum consensus strength
    pub fn min_consensus_strength(&self) -> f64 {
        self.consensus_strength
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min)
    }

    /// Get maximum consensus strength
    pub fn max_consensus_strength(&self) -> f64 {
        self.consensus_strength
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// Get average cluster stability
    pub fn average_cluster_stability(&self) -> f64 {
        if self.cluster_stability.is_empty() {
            0.0
        } else {
            self.cluster_stability.iter().sum::<f64>() / self.cluster_stability.len() as f64
        }
    }
}

/// Diversity metrics for the ensemble
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiversityMetrics {
    /// Average pairwise diversity (1 - ARI)
    pub average_diversity: f64,
    /// Diversity matrix between all pairs
    pub diversity_matrix: Array2<f64>,
    /// Algorithm distribution
    pub algorithm_distribution: HashMap<String, usize>,
    /// Parameter diversity statistics
    pub parameter_diversity: HashMap<String, f64>,
}

impl DiversityMetrics {
    /// Create new diversity metrics
    pub fn new(
        average_diversity: f64,
        diversity_matrix: Array2<f64>,
        algorithm_distribution: HashMap<String, usize>,
        parameter_diversity: HashMap<String, f64>,
    ) -> Self {
        Self {
            average_diversity,
            diversity_matrix,
            algorithm_distribution,
            parameter_diversity,
        }
    }

    /// Get maximum pairwise diversity
    pub fn max_diversity(&self) -> f64 {
        self.diversity_matrix
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// Get minimum pairwise diversity
    pub fn min_diversity(&self) -> f64 {
        self.diversity_matrix
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min)
    }

    /// Get diversity variance
    pub fn diversity_variance(&self) -> f64 {
        let mean = self.average_diversity;
        let variance = self
            .diversity_matrix
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>()
            / (self.diversity_matrix.len() as f64);
        variance
    }

    /// Check if ensemble has good diversity
    pub fn has_good_diversity(&self, threshold: f64) -> bool {
        self.average_diversity >= threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::arr1;

    #[test]
    fn test_ensemble_config_default() {
        let config = EnsembleConfig::default();
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
    fn test_clustering_result_creation() {
        let labels = arr1(&[0, 0, 1, 1, -1]);
        let mut params = HashMap::new();
        params.insert("k".to_string(), "2".to_string());

        let result = ClusteringResult::new(labels, "kmeans".to_string(), params, 0.8, 1.5);

        assert_eq!(result.n_clusters, 2);
        assert!(result.has_noise());
        assert_eq!(result.noise_count(), 1);
        assert_eq!(result.cluster_sizes(), vec![2, 2]);
    }

    #[test]
    fn test_ensemble_result_metrics() {
        let consensus_labels = arr1(&[0, 0, 1, 1]);
        let individual_results = vec![
            ClusteringResult::new(
                arr1(&[0, 0, 1, 1]),
                "kmeans".to_string(),
                HashMap::new(),
                0.8,
                1.0,
            ),
            ClusteringResult::new(
                arr1(&[1, 1, 0, 0]),
                "dbscan".to_string(),
                HashMap::new(),
                0.7,
                1.5,
            ),
        ];

        let consensus_stats = ConsensusStatistics::new(
            Array2::zeros((2, 2)),
            arr1(&[0.9, 0.9, 0.8, 0.8]),
            vec![0.9, 0.8],
            arr1(&[2, 2, 2, 2]),
        );

        let diversity_metrics =
            DiversityMetrics::new(0.5, Array2::zeros((2, 2)), HashMap::new(), HashMap::new());

        let result = EnsembleResult::new(
            consensus_labels,
            individual_results,
            consensus_stats,
            diversity_metrics,
            0.85,
            0.9,
        );

        assert_eq!(result.n_consensus_clusters(), 2);
        assert_eq!(result.average_individual_quality(), 0.75);
        assert!(result.best_individual_result().is_some());
    }

    #[test]
    fn test_consensus_statistics() {
        let stats = ConsensusStatistics::new(
            Array2::zeros((3, 3)),
            arr1(&[0.8, 0.9, 0.7]),
            vec![0.9, 0.8, 0.85],
            arr1(&[3, 2, 3]),
        );

        assert!((stats.average_consensus_strength() - 0.8).abs() < 1e-10);
        assert_eq!(stats.min_consensus_strength(), 0.7);
        assert_eq!(stats.max_consensus_strength(), 0.9);
        assert!((stats.average_cluster_stability() - 0.85).abs() < 1e-10);
    }

    #[test]
    fn test_diversity_metrics() {
        let metrics = DiversityMetrics::new(
            0.6,
            Array2::from_shape_vec((2, 2), vec![0.0, 0.8, 0.8, 0.0]).unwrap(),
            HashMap::new(),
            HashMap::new(),
        );

        assert_eq!(metrics.max_diversity(), 0.8);
        assert_eq!(metrics.min_diversity(), 0.0);
        assert!(metrics.has_good_diversity(0.5));
        assert!(!metrics.has_good_diversity(0.7));
    }
}
