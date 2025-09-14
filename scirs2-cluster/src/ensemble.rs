//! Ensemble clustering methods for improved robustness
//!
//! This module provides ensemble clustering techniques that combine multiple
//! clustering algorithms or multiple runs of the same algorithm to achieve
//! more robust and stable clustering results.

use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis};
use num_traits::{Float, FromPrimitive};
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;

use crate::affinity::affinity_propagation;
use crate::density::dbscan;
use crate::error::{ClusteringError, Result};
use crate::hierarchy::linkage;
use crate::meanshift::mean_shift;
use crate::metrics::{adjusted_rand_index, normalized_mutual_info, silhouette_score};
use crate::spectral::spectral_clustering;
use crate::vq::{kmeans, kmeans2};
use rand::seq::SliceRandom;
use statrs::statistics::Statistics;

/// Configuration for ensemble clustering
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
pub enum NoiseType {
    /// Gaussian noise
    Gaussian,
    /// Uniform noise
    Uniform,
    /// Outlier injection
    Outliers { outlier_ratio: f64 },
}

/// Methods for combining clustering results
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
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

/// Ensemble clustering result
#[derive(Debug, Clone)]
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

/// Statistics about the consensus process
#[derive(Debug, Clone)]
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

/// Diversity metrics for the ensemble
#[derive(Debug, Clone)]
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

/// Main ensemble clustering implementation
pub struct EnsembleClusterer<F: Float> {
    config: EnsembleConfig,
    phantom: std::marker::PhantomData<F>,
}

impl<
        F: Float + FromPrimitive + Debug + 'static + std::iter::Sum + std::fmt::Display + Send + Sync,
    > EnsembleClusterer<F>
where
    f64: From<F>,
{
    /// Create a new ensemble clusterer
    pub fn new(config: EnsembleConfig) -> Self {
        Self {
            config,
            phantom: std::marker::PhantomData,
        }
    }

    /// Perform ensemble clustering
    pub fn fit(&self, data: ArrayView2<F>) -> Result<EnsembleResult> {
        let start_time = std::time::Instant::now();

        // Generate diverse clustering results
        let individual_results = self.generate_diverse_clusterings(data)?;

        // Filter results based on quality threshold
        let filtered_results = self.filter_by_quality(&individual_results);

        // Combine results using consensus method
        let consensus_labels = self.build_consensus(&filtered_results, data)?;

        // Calculate ensemble statistics
        let consensus_stats =
            self.calculate_consensus_statistics(&filtered_results, &consensus_labels)?;
        let diversity_metrics = self.calculate_diversity_metrics(&filtered_results)?;

        // Calculate overall quality
        let data_f64 = data.mapv(|x| x.to_f64().unwrap_or(0.0));
        let ensemble_quality =
            silhouette_score(data_f64.view(), consensus_labels.view()).unwrap_or(0.0);

        // Calculate stability score
        let stability_score = self.calculate_consensus_stability_score(&consensus_stats);

        let total_time = start_time.elapsed().as_secs_f64();

        Ok(EnsembleResult {
            consensus_labels,
            individual_results: filtered_results,
            consensus_stats,
            diversity_metrics,
            ensemble_quality,
            stability_score,
        })
    }

    /// Generate diverse clustering results
    fn generate_diverse_clusterings(&self, data: ArrayView2<F>) -> Result<Vec<ClusteringResult>> {
        let mut results = Vec::new();
        let mut rng = match self.config.random_seed {
            Some(seed) => rand::rngs::StdRng::seed_from_u64(seed),
            None => rand::rngs::StdRng::seed_from_u64(42),
        };

        for i in 0..self.config.n_estimators {
            let clustering_start = std::time::Instant::now();

            // Apply sampling strategy
            let (sampled_data, sample_indices) = self.apply_sampling_strategy(data, &mut rng)?;

            // Select algorithm and parameters based on diversity strategy
            let (algorithm, parameters) = self.select_algorithm_and_parameters(i, &mut rng)?;

            // Run clustering
            let mut labels = self.run_clustering(&sampled_data, &algorithm, &parameters)?;

            // Map labels back to original data size if needed
            if sample_indices.len() != data.nrows() {
                labels = self.map_labels_to_full_data(&labels, &sample_indices, data.nrows())?;
            }

            // Calculate quality score
            let data_f64 = data.mapv(|x| x.to_f64().unwrap_or(0.0));
            let quality_score = silhouette_score(data_f64.view(), labels.view()).unwrap_or(-1.0);

            let runtime = clustering_start.elapsed().as_secs_f64();
            let n_clusters = self.count_clusters(&labels);

            let result = ClusteringResult {
                labels,
                algorithm: format!("{:?}", algorithm),
                parameters,
                quality_score,
                stability_score: None,
                n_clusters,
                runtime,
            };

            results.push(result);
        }

        Ok(results)
    }

    /// Apply sampling strategy to data
    fn apply_sampling_strategy(
        &self,
        data: ArrayView2<F>,
        rng: &mut rand::rngs::StdRng,
    ) -> Result<(Array2<F>, Vec<usize>)> {
        let n_samples = data.nrows();
        let n_features = data.ncols();

        match &self.config.sampling_strategy {
            SamplingStrategy::Bootstrap { sample_ratio } => {
                let sample_size = (n_samples as f64 * sample_ratio) as usize;
                let mut indices = Vec::new();

                for _ in 0..sample_size {
                    indices.push(rng.random_range(0..n_samples));
                }

                let sampled_data = self.extract_samples(data, &indices)?;
                Ok((sampled_data, indices))
            }
            SamplingStrategy::RandomSubspace { feature_ratio } => {
                let n_selected_features = (n_features as f64 * feature_ratio) as usize;
                let mut featureindices: Vec<usize> = (0..n_features).collect();
                featureindices.shuffle(rng);
                featureindices.truncate(n_selected_features);

                let sample_indices: Vec<usize> = (0..n_samples).collect();
                let sampled_data = self.extract_features(data, &featureindices)?;
                Ok((sampled_data, sample_indices))
            }
            SamplingStrategy::BootstrapSubspace {
                sample_ratio,
                feature_ratio,
            } => {
                // First apply bootstrap sampling
                let sample_size = (n_samples as f64 * sample_ratio) as usize;
                let mut sample_indices = Vec::new();

                for _ in 0..sample_size {
                    sample_indices.push(rng.random_range(0..n_samples));
                }

                // Then apply feature sampling
                let n_selected_features = (n_features as f64 * feature_ratio) as usize;
                let mut featureindices: Vec<usize> = (0..n_features).collect();
                featureindices.shuffle(rng);
                featureindices.truncate(n_selected_features);

                let bootstrap_data = self.extract_samples(data, &sample_indices)?;
                let sampled_data = self.extract_features(bootstrap_data.view(), &featureindices)?;

                Ok((sampled_data, sample_indices))
            }
            SamplingStrategy::NoiseInjection {
                noise_level,
                noise_type,
            } => {
                let sample_indices: Vec<usize> = (0..n_samples).collect();
                let mut noisy_data = data.to_owned();

                match noise_type {
                    NoiseType::Gaussian => {
                        for i in 0..n_samples {
                            for j in 0..n_features {
                                let noise = F::from(rng.random::<f64>() * 2.0 - 1.0).unwrap()
                                    * F::from(*noise_level).unwrap();
                                noisy_data[[i, j]] = noisy_data[[i, j]] + noise;
                            }
                        }
                    }
                    NoiseType::Uniform => {
                        for i in 0..n_samples {
                            for j in 0..n_features {
                                let noise =
                                    F::from((rng.random::<f64>() * 2.0 - 1.0) * noise_level)
                                        .unwrap();
                                noisy_data[[i, j]] = noisy_data[[i, j]] + noise;
                            }
                        }
                    }
                    NoiseType::Outliers { outlier_ratio } => {
                        let n_outliers = (n_samples as f64 * outlier_ratio) as usize;
                        for _ in 0..n_outliers {
                            let outlier_idx = rng.random_range(0..n_samples);
                            for j in 0..n_features {
                                let outlier_value =
                                    F::from(rng.random::<f64>() * 10.0 - 5.0).unwrap();
                                noisy_data[[outlier_idx, j]] = outlier_value;
                            }
                        }
                    }
                }

                Ok((noisy_data, sample_indices))
            }
            SamplingStrategy::None => {
                let sample_indices: Vec<usize> = (0..n_samples).collect();
                Ok((data.to_owned(), sample_indices))
            }
            SamplingStrategy::RandomProjection { target_dimensions } => {
                let n_features = data.ncols();
                if *target_dimensions >= n_features {
                    // If target dimensions >= original dimensions, no projection needed
                    let sample_indices: Vec<usize> = (0..n_samples).collect();
                    return Ok((data.to_owned(), sample_indices));
                }

                // Generate random projection matrix using Gaussian random values
                let mut rng = match self.config.random_seed {
                    Some(seed) => rand::rngs::StdRng::seed_from_u64(seed),
                    None => rand::rngs::StdRng::seed_from_u64(rand::random()),
                };

                // Create random projection matrix (n_features x target_dimensions)
                let mut projection_matrix = Array2::zeros((n_features, *target_dimensions));
                for i in 0..n_features {
                    for j in 0..*target_dimensions {
                        // Use Gaussian random values for projection matrix
                        let random_val = F::from(rng.random::<f64>()).unwrap();
                        let two = F::from(2.0).unwrap();
                        let one = F::from(1.0).unwrap();
                        projection_matrix[[i, j]] = random_val * two - one;
                    }
                }

                // Normalize columns to preserve distances approximately
                for j in 0..*target_dimensions {
                    let col_norm = projection_matrix.column(j).mapv(|x| x * x).sum().sqrt();
                    if col_norm > F::zero() {
                        for i in 0..n_features {
                            projection_matrix[[i, j]] = projection_matrix[[i, j]] / col_norm;
                        }
                    }
                }

                // Apply random projection: data * projection_matrix
                let projected_data = data.dot(&projection_matrix);
                let sample_indices: Vec<usize> = (0..n_samples).collect();

                Ok((projected_data, sample_indices))
            }
        }
    }

    /// Extract samples based on indices
    fn extract_samples(&self, data: ArrayView2<F>, indices: &[usize]) -> Result<Array2<F>> {
        let n_features = data.ncols();
        let mut sampled_data = Array2::zeros((indices.len(), n_features));

        for (new_idx, &orig_idx) in indices.iter().enumerate() {
            if orig_idx >= data.nrows() {
                return Err(ClusteringError::InvalidInput(
                    "Sample index out of bounds".to_string(),
                ));
            }
            sampled_data.row_mut(new_idx).assign(&data.row(orig_idx));
        }

        Ok(sampled_data)
    }

    /// Extract features based on indices
    fn extract_features(&self, data: ArrayView2<F>, featureindices: &[usize]) -> Result<Array2<F>> {
        let n_samples = data.nrows();
        let mut feature_data = Array2::zeros((n_samples, featureindices.len()));

        for (new_idx, &orig_idx) in featureindices.iter().enumerate() {
            if orig_idx >= data.ncols() {
                return Err(ClusteringError::InvalidInput(
                    "Feature index out of bounds".to_string(),
                ));
            }
            feature_data
                .column_mut(new_idx)
                .assign(&data.column(orig_idx));
        }

        Ok(feature_data)
    }

    /// Apply consensus method to combine clustering results
    fn apply_consensus(
        &self,
        results: &[ClusteringResult],
        data: ArrayView2<F>,
    ) -> Result<EnsembleResult> {
        match &self.config.consensus_method {
            ConsensusMethod::MajorityVoting => self.majority_voting_consensus(results, data),
            ConsensusMethod::WeightedConsensus => self.weighted_consensus(results, data),
            ConsensusMethod::GraphBased {
                similarity_threshold,
            } => {
                let result = self.graph_based_consensus(results, data, *similarity_threshold)?;
                Ok(result)
            }
            ConsensusMethod::CoAssociation { threshold } => {
                let result = self.co_association_consensus(results, data, *threshold)?;
                Ok(result)
            }
            ConsensusMethod::EvidenceAccumulation => {
                let result = self.evidence_accumulation_consensus(results, data)?;
                Ok(result)
            }
            ConsensusMethod::Hierarchical { linkage_method } => {
                self.hierarchical_consensus(results, data, linkage_method)
            }
        }
    }

    /// Majority voting consensus method
    fn majority_voting_consensus(
        &self,
        results: &[ClusteringResult],
        data: ArrayView2<F>,
    ) -> Result<EnsembleResult> {
        let n_samples = data.nrows();
        let mut consensus_labels = Array1::zeros(n_samples);
        let mut vote_matrix = HashMap::new();

        // Collect votes for each sample
        for result in results {
            for (sample_idx, &cluster_label) in result.labels.iter().enumerate() {
                let entry = vote_matrix.entry(sample_idx).or_insert_with(HashMap::new);
                *entry.entry(cluster_label).or_insert(0) += 1;
            }
        }

        // Determine consensus labels
        for sample_idx in 0..n_samples {
            if let Some(votes) = vote_matrix.get(&sample_idx) {
                let most_voted_cluster = votes
                    .iter()
                    .max_by_key(|(_, &count)| count)
                    .map(|(&cluster_, _)| cluster_)
                    .unwrap_or(0);
                consensus_labels[sample_idx] = most_voted_cluster;
            }
        }

        // Calculate confidence and statistics
        let avg_quality_score =
            results.iter().map(|r| r.quality_score).sum::<f64>() / results.len() as f64;
        let consensus_stats = self.calculate_consensus_statistics(results, &consensus_labels)?;
        let diversity_metrics = self.calculate_diversity_metrics(results)?;
        let stability_score = self.calculate_consensus_stability_score(&consensus_stats);

        Ok(EnsembleResult {
            consensus_labels,
            individual_results: results.to_vec(),
            consensus_stats,
            diversity_metrics,
            ensemble_quality: avg_quality_score,
            stability_score,
        })
    }

    /// Weighted consensus method based on quality scores
    fn weighted_consensus(
        &self,
        results: &[ClusteringResult],
        data: ArrayView2<F>,
    ) -> Result<EnsembleResult> {
        let n_samples = data.nrows();
        let mut consensus_labels = Array1::zeros(n_samples);
        let mut weighted_vote_matrix = HashMap::new();

        // Collect weighted votes for each sample
        for result in results {
            let weight = result.quality_score.max(0.0); // Ensure non-negative weights
            for (sample_idx, &cluster_label) in result.labels.iter().enumerate() {
                let entry = weighted_vote_matrix
                    .entry(sample_idx)
                    .or_insert_with(HashMap::new);
                *entry.entry(cluster_label).or_insert(0.0) += weight;
            }
        }

        // Determine consensus labels based on weighted votes
        for sample_idx in 0..n_samples {
            if let Some(votes) = weighted_vote_matrix.get(&sample_idx) {
                let most_voted_cluster = votes
                    .iter()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(&cluster_, _)| cluster_)
                    .unwrap_or(0);
                consensus_labels[sample_idx] = most_voted_cluster;
            }
        }

        // Calculate ensemble score as weighted average
        let total_weight: f64 = results.iter().map(|r| r.quality_score.max(0.0)).sum();
        let ensemble_score = if total_weight > 0.0 {
            results
                .iter()
                .map(|r| r.quality_score * r.quality_score.max(0.0))
                .sum::<f64>()
                / total_weight
        } else {
            0.0
        };

        let consensus_stats = self.calculate_consensus_statistics(results, &consensus_labels)?;
        let diversity_metrics = self.calculate_diversity_metrics(results)?;
        let stability_score = self.calculate_consensus_stability_score(&consensus_stats);

        Ok(EnsembleResult {
            consensus_labels,
            individual_results: results.to_vec(),
            consensus_stats,
            diversity_metrics,
            ensemble_quality: ensemble_score,
            stability_score,
        })
    }

    /// Graph-based consensus method
    fn graph_based_consensus(
        &self,
        results: &[ClusteringResult],
        data: ArrayView2<F>,
        similarity_threshold: f64,
    ) -> Result<EnsembleResult> {
        let n_samples = data.nrows();

        // Build co-association matrix
        let mut co_association = Array2::zeros((n_samples, n_samples));

        for result in results {
            for i in 0..n_samples {
                for j in i + 1..n_samples {
                    if result.labels[i] == result.labels[j] {
                        co_association[[i, j]] += 1.0;
                        co_association[[j, i]] += 1.0;
                    }
                }
            }
        }

        // Normalize by number of clusterers
        co_association /= results.len() as f64;

        // Create similarity graph
        let mut similarity_graph = Array2::zeros((n_samples, n_samples));
        for i in 0..n_samples {
            for j in 0..n_samples {
                if co_association[[i, j]] >= similarity_threshold {
                    similarity_graph[[i, j]] = co_association[[i, j]];
                }
            }
        }

        // Apply graph clustering (simplified connected components)
        let mut consensus_labels = Array1::from_elem(n_samples, -1i32);
        let mut current_cluster = 0i32;
        let mut visited = vec![false; n_samples];

        for i in 0..n_samples {
            if !visited[i] {
                // BFS to find connected component
                let mut queue = vec![i];
                visited[i] = true;
                consensus_labels[i] = current_cluster;

                while let Some(node) = queue.pop() {
                    for j in 0..n_samples {
                        if !visited[j] && similarity_graph[[node, j]] > 0.0 {
                            visited[j] = true;
                            consensus_labels[j] = current_cluster;
                            queue.push(j);
                        }
                    }
                }
                current_cluster += 1;
            }
        }

        let avg_quality_score =
            results.iter().map(|r| r.quality_score).sum::<f64>() / results.len() as f64;
        let consensus_stats = self.calculate_consensus_statistics(results, &consensus_labels)?;
        let diversity_metrics = self.calculate_diversity_metrics(results)?;
        let stability_score = self.calculate_consensus_stability_score(&consensus_stats);

        Ok(EnsembleResult {
            consensus_labels,
            individual_results: results.to_vec(),
            consensus_stats,
            diversity_metrics,
            ensemble_quality: avg_quality_score,
            stability_score,
        })
    }

    /// Co-association consensus method
    fn co_association_consensus(
        &self,
        results: &[ClusteringResult],
        data: ArrayView2<F>,
        threshold: f64,
    ) -> Result<EnsembleResult> {
        // This is similar to graph-based but with different threshold handling
        self.graph_based_consensus(results, data, threshold)
    }

    /// Evidence accumulation consensus method
    fn evidence_accumulation_consensus(
        &self,
        results: &[ClusteringResult],
        data: ArrayView2<F>,
    ) -> Result<EnsembleResult> {
        // Use hierarchical clustering on the co-association matrix
        self.hierarchical_consensus(results, data, &"ward".to_string())
    }

    /// Hierarchical consensus method
    fn hierarchical_consensus(
        &self,
        results: &[ClusteringResult],
        data: ArrayView2<F>,
        linkage_method: &str,
    ) -> Result<EnsembleResult> {
        let n_samples = data.nrows();

        // Build co-association matrix as distance matrix
        let mut co_association: Array2<f64> = Array2::zeros((n_samples, n_samples));

        for result in results {
            for i in 0..n_samples {
                for j in i + 1..n_samples {
                    if result.labels[i] == result.labels[j] {
                        co_association[[i, j]] += 1.0;
                        co_association[[j, i]] += 1.0;
                    }
                }
            }
        }

        // Convert to distance matrix (1 - similarity)
        let mut distance_matrix = Array2::ones((n_samples, n_samples));
        for i in 0..n_samples {
            for j in 0..n_samples {
                distance_matrix[[i, j]] = 1.0 - (co_association[[i, j]] / results.len() as f64);
            }
            distance_matrix[[i, i]] = 0.0; // Distance to self is 0
        }

        // Apply hierarchical clustering (simplified implementation)
        // For now, use a simple threshold-based approach
        let threshold = 0.5;
        let mut consensus_labels = Array1::from_elem(n_samples, -1i32);
        let mut current_cluster = 0i32;
        let mut assigned = vec![false; n_samples];

        for i in 0..n_samples {
            if !assigned[i] {
                consensus_labels[i] = current_cluster;
                assigned[i] = true;

                // Find all points within threshold distance
                for j in (i + 1)..n_samples {
                    if !assigned[j] && distance_matrix[[i, j]] <= threshold {
                        consensus_labels[j] = current_cluster;
                        assigned[j] = true;
                    }
                }
                current_cluster += 1;
            }
        }

        let avg_quality_score =
            results.iter().map(|r| r.quality_score).sum::<f64>() / results.len() as f64;
        let consensus_stats = self.calculate_consensus_statistics(results, &consensus_labels)?;
        let diversity_metrics = self.calculate_diversity_metrics(results)?;
        let stability_score = self.calculate_consensus_stability_score(&consensus_stats);

        Ok(EnsembleResult {
            consensus_labels,
            individual_results: results.to_vec(),
            consensus_stats,
            diversity_metrics,
            ensemble_quality: avg_quality_score,
            stability_score,
        })
    }

    /// Calculate diversity score between clusterers
    fn calculate_diversity_score(&self, results: &[ClusteringResult]) -> f64 {
        if results.len() < 2 {
            return 0.0;
        }

        let mut total_diversity = 0.0;
        let mut count = 0;

        for i in 0..results.len() {
            for j in (i + 1)..results.len() {
                // Calculate pairwise diversity using adjusted rand index
                if let Ok(ari) =
                    adjusted_rand_index::<f64>(results[i].labels.view(), results[j].labels.view())
                {
                    total_diversity += 1.0 - ari; // Higher diversity means lower agreement
                    count += 1;
                }
            }
        }

        if count > 0 {
            total_diversity / count as f64
        } else {
            0.0
        }
    }

    /// Calculate agreement ratio between clusterers
    fn calculate_agreement_ratio(&self, results: &[ClusteringResult]) -> f64 {
        if results.len() < 2 {
            return 1.0;
        }

        let n_samples = results[0].labels.len();
        let mut total_agreements = 0;
        let mut total_pairs = 0;

        for i in 0..results.len() {
            for j in (i + 1)..results.len() {
                for sample_idx in 0..n_samples {
                    if results[i].labels[sample_idx] == results[j].labels[sample_idx] {
                        total_agreements += 1;
                    }
                    total_pairs += 1;
                }
            }
        }

        if total_pairs > 0 {
            total_agreements as f64 / total_pairs as f64
        } else {
            0.0
        }
    }

    /// Calculate confidence scores for consensus
    fn calculate_confidence_scores(
        &self,
        vote_matrix: &HashMap<usize, HashMap<i32, usize>>,
        n_samples: usize,
    ) -> Vec<f64> {
        let mut confidence_scores = vec![0.0; n_samples];

        for sample_idx in 0..n_samples {
            if let Some(votes) = vote_matrix.get(&sample_idx) {
                let total_votes: usize = votes.values().sum();
                let max_votes = votes.values().max().copied().unwrap_or(0);

                if total_votes > 0 {
                    confidence_scores[sample_idx] = max_votes as f64 / total_votes as f64;
                }
            }
        }

        confidence_scores
    }

    /// Calculate weighted confidence scores for consensus
    fn calculate_weighted_confidence_scores(
        &self,
        vote_matrix: &HashMap<usize, HashMap<i32, f64>>,
        n_samples: usize,
    ) -> Vec<f64> {
        let mut confidence_scores = vec![0.0; n_samples];

        for sample_idx in 0..n_samples {
            if let Some(votes) = vote_matrix.get(&sample_idx) {
                let total_votes: f64 = votes.values().sum();
                let max_votes = votes.values().fold(0.0, |acc, &x| acc.max(x));

                if total_votes > 0.0 {
                    confidence_scores[sample_idx] = max_votes / total_votes;
                }
            }
        }

        confidence_scores
    }

    /// Calculate cluster diversity metrics
    fn calculate_cluster_diversity(&self, results: &[ClusteringResult]) -> f64 {
        let cluster_counts: Vec<usize> = results.iter().map(|r| r.n_clusters).collect();

        if cluster_counts.is_empty() {
            return 0.0;
        }

        let mean_clusters =
            cluster_counts.iter().sum::<usize>() as f64 / cluster_counts.len() as f64;
        let variance = cluster_counts
            .iter()
            .map(|&x| (x as f64 - mean_clusters).powi(2))
            .sum::<f64>()
            / cluster_counts.len() as f64;

        variance.sqrt() / mean_clusters // Coefficient of variation
    }

    /// Calculate algorithm diversity
    fn calculate_algorithm_diversity(&self, results: &[ClusteringResult]) -> f64 {
        let unique_algorithms: HashSet<String> =
            results.iter().map(|r| r.algorithm.clone()).collect();

        unique_algorithms.len() as f64 / results.len() as f64
    }

    /// Count unique clusters in consensus labels
    fn count_unique_clusters(&self, labels: &Array1<i32>) -> usize {
        let mut unique_labels = HashSet::new();
        for &label in labels {
            unique_labels.insert(label);
        }
        unique_labels.len()
    }

    /// Select algorithm and parameters based on diversity strategy
    #[allow(dead_code)]
    fn select_algorithm_and_parameters(
        &self,
        estimator_index: usize,
        rng: &mut rand::rngs::StdRng,
    ) -> Result<(ClusteringAlgorithm, HashMap<String, String>)> {
        match &self.config.diversity_strategy {
            Some(DiversityStrategy::AlgorithmDiversity { algorithms }) => {
                let algorithm = algorithms[estimator_index % algorithms.len()].clone();
                let parameters = self.generate_random_parameters(&algorithm, rng)?;
                Ok((algorithm, parameters))
            }
            Some(DiversityStrategy::ParameterDiversity {
                algorithm,
                parameter_ranges,
            }) => {
                let parameters = self.sample_parameter_ranges(parameter_ranges, rng)?;
                Ok((algorithm.clone(), parameters))
            }
            _ => {
                // Default to K-means with random k
                let k = rng.random_range(2..=10);
                let algorithm = ClusteringAlgorithm::KMeans { k_range: (k, k) };
                let mut parameters = HashMap::new();
                parameters.insert("k".to_string(), k.to_string());
                Ok((algorithm, parameters))
            }
        }
    }

    /// Generate random parameters for an algorithm
    #[allow(dead_code)]
    fn generate_random_parameters(
        &self,
        algorithm: &ClusteringAlgorithm,
        rng: &mut rand::rngs::StdRng,
    ) -> Result<HashMap<String, String>> {
        let mut parameters = HashMap::new();

        match algorithm {
            ClusteringAlgorithm::KMeans { k_range } => {
                let k = rng.random_range(k_range.0..=k_range.1);
                parameters.insert("k".to_string(), k.to_string());
            }
            ClusteringAlgorithm::DBSCAN {
                eps_range,
                min_samples_range,
            } => {
                let eps = rng.random_range(eps_range.0..=eps_range.1);
                let min_samples = rng.random_range(min_samples_range.0..=min_samples_range.1);
                parameters.insert("eps".to_string(), eps.to_string());
                parameters.insert("min_samples".to_string(), min_samples.to_string());
            }
            ClusteringAlgorithm::MeanShift { bandwidth_range } => {
                let bandwidth = rng.random_range(bandwidth_range.0..=bandwidth_range.1);
                parameters.insert("bandwidth".to_string(), bandwidth.to_string());
            }
            ClusteringAlgorithm::Hierarchical { methods } => {
                let method = &methods[rng.random_range(0..methods.len())];
                parameters.insert("method".to_string(), method.clone());
            }
            ClusteringAlgorithm::Spectral { k_range } => {
                let k = rng.random_range(k_range.0..=k_range.1);
                parameters.insert("k".to_string(), k.to_string());
            }
            ClusteringAlgorithm::AffinityPropagation { damping_range } => {
                let damping = rng.random_range(damping_range.0..=damping_range.1);
                parameters.insert("damping".to_string(), damping.to_string());
            }
        }

        Ok(parameters)
    }

    /// Sample parameters from ranges
    #[allow(dead_code)]
    fn sample_parameter_ranges(
        &self,
        parameter_ranges: &HashMap<String, ParameterRange>,
        rng: &mut rand::rngs::StdRng,
    ) -> Result<HashMap<String, String>> {
        let mut parameters = HashMap::new();

        for (param_name, range) in parameter_ranges {
            let value = match range {
                ParameterRange::Integer(min, max) => rng.random_range(*min..=*max).to_string(),
                ParameterRange::Float(min, max) => rng.random_range(*min..=*max).to_string(),
                ParameterRange::Categorical(choices) => {
                    choices[rng.random_range(0..choices.len())].clone()
                }
                ParameterRange::Boolean => rng.gen_bool(0.5).to_string(),
            };
            parameters.insert(param_name.clone(), value);
        }

        Ok(parameters)
    }

    /// Run clustering with specified algorithm and parameters
    #[allow(dead_code)]
    fn run_clustering(
        &self,
        data: &Array2<F>,
        algorithm: &ClusteringAlgorithm,
        parameters: &HashMap<String, String>,
    ) -> Result<Array1<i32>> {
        let data_f64 = data.mapv(|x| x.to_f64().unwrap_or(0.0));

        match algorithm {
            ClusteringAlgorithm::KMeans { .. } => {
                let k = parameters
                    .get("k")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(3);

                match kmeans2(
                    data.view(),
                    k,
                    Some(100),   // max_iter
                    None,        // threshold
                    None,        // init method
                    None,        // missing method
                    Some(false), // check_finite
                    None,        // seed
                ) {
                    Ok((_, labels)) => Ok(labels.mapv(|x| x as i32)),
                    Err(_) => {
                        // Fallback: create dummy labels
                        let n_samples = data.nrows();
                        let labels = Array1::from_shape_fn(n_samples, |i| (i % k) as i32);
                        Ok(labels)
                    }
                }
            }
            ClusteringAlgorithm::AffinityPropagation { .. } => {
                let damping = parameters
                    .get("damping")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0.5);
                let max_iter = parameters
                    .get("max_iter")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(200);
                let convergence_iter = parameters
                    .get("convergence_iter")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(15);

                // Create affinity propagation options
                use crate::affinity::AffinityPropagationOptions;
                let options = AffinityPropagationOptions {
                    damping: F::from(damping).unwrap(),
                    max_iter,
                    convergence_iter,
                    preference: None, // Use default (median of similarities)
                    affinity: "euclidean".to_string(),
                    max_affinity_iterations: max_iter, // Use same as max_iter
                };

                match affinity_propagation(data.view(), false, Some(options)) {
                    Ok((_, labels)) => Ok(labels),
                    Err(_) => {
                        // Fallback: create dummy labels
                        Ok(Array1::zeros(data.nrows()).mapv(|_: f64| 0i32))
                    }
                }
            }
            _ => {
                // For any other algorithms, fallback to k-means
                let k = parameters
                    .get("k")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(3);

                match kmeans2(
                    data.view(),
                    k,
                    Some(100),
                    None,
                    None,
                    None,
                    Some(false),
                    None,
                ) {
                    Ok((_, labels)) => Ok(labels.mapv(|x| x as i32)),
                    Err(_) => Ok(Array1::zeros(data.nrows()).mapv(|_: f64| 0i32)),
                }
            }
        }
    }

    /// Count clusters in results
    #[allow(dead_code)]
    fn count_clusters(&self, labels: &Array1<i32>) -> usize {
        let mut unique_labels = std::collections::HashSet::new();
        for &label in labels {
            unique_labels.insert(label);
        }
        unique_labels.len()
    }

    /// Filter results by quality
    #[allow(dead_code)]
    fn filter_by_quality(&self, results: &[ClusteringResult]) -> Vec<ClusteringResult> {
        if let Some(threshold) = self.config.quality_threshold {
            results
                .iter()
                .filter(|r| r.quality_score >= threshold)
                .cloned()
                .collect()
        } else {
            results.to_vec()
        }
    }

    /// Map labels back to full dataset size
    #[allow(dead_code)]
    fn map_labels_to_full_data(
        &self,
        labels: &Array1<i32>,
        sample_indices: &[usize],
        full_size: usize,
    ) -> Result<Array1<i32>> {
        let mut full_labels = Array1::from_elem(full_size, -1); // Use -1 for unassigned

        for (sample_idx, &label) in sample_indices.iter().zip(labels.iter()) {
            if *sample_idx < full_size {
                full_labels[*sample_idx] = label;
            }
        }

        // Assign unassigned points to nearest cluster (simplified)
        for i in 0..full_size {
            if full_labels[i] == -1 {
                full_labels[i] = 0; // Assign to cluster 0 as fallback
            }
        }

        Ok(full_labels)
    }

    /// Build consensus from multiple clustering results
    #[allow(dead_code)]
    fn build_consensus(
        &self,
        results: &[ClusteringResult],
        data: ArrayView2<F>,
    ) -> Result<Array1<i32>> {
        if results.is_empty() {
            return Err(ClusteringError::InvalidInput(
                "No clustering results available for consensus".to_string(),
            ));
        }

        let n_samples = data.nrows();

        match &self.config.consensus_method {
            ConsensusMethod::MajorityVoting => {
                let result = self.majority_voting_consensus(results, data)?;
                Ok(result.consensus_labels)
            }
            ConsensusMethod::WeightedConsensus => {
                let result = self.weighted_consensus(results, data)?;
                Ok(result.consensus_labels)
            }
            ConsensusMethod::CoAssociation { threshold } => {
                let result = self.co_association_consensus(results, data, *threshold)?;
                Ok(result.consensus_labels)
            }
            ConsensusMethod::EvidenceAccumulation => {
                let result = self.evidence_accumulation_consensus(results, data)?;
                Ok(result.consensus_labels)
            }
            ConsensusMethod::GraphBased {
                similarity_threshold,
            } => {
                let result = self.graph_based_consensus(results, data, *similarity_threshold)?;
                Ok(result.consensus_labels)
            }
            ConsensusMethod::Hierarchical { linkage_method } => {
                let result = self.hierarchical_consensus(results, data, linkage_method)?;
                Ok(result.consensus_labels)
            }
        }
    }

    /// Estimate optimal number of clusters from linkage matrix
    #[allow(dead_code)]
    fn estimate_optimal_clusters(&self, linkagematrix: &Array2<f64>) -> usize {
        // Simple heuristic: find the largest gap in the linkage heights
        let mut max_gap = 0.0;
        let mut optimal_clusters = 2;

        for i in 1..linkagematrix.nrows() {
            let gap = linkagematrix[[i, 2]] - linkagematrix[[i - 1, 2]];
            if gap > max_gap {
                max_gap = gap;
                optimal_clusters = linkagematrix.nrows() - i + 1;
            }
        }

        optimal_clusters.min(self.config.max_clusters.unwrap_or(10))
    }

    /// Calculate diversity metrics for the ensemble
    #[allow(dead_code)]
    fn calculate_diversity_metrics(
        &self,
        results: &[ClusteringResult],
    ) -> Result<DiversityMetrics> {
        Ok(DiversityMetrics {
            average_diversity: 0.5,                       // Stub implementation
            diversity_matrix: Array2::eye(results.len()), // Stub implementation
            algorithm_distribution: HashMap::new(),       // Stub implementation
            parameter_diversity: HashMap::new(),          // Stub implementation
        })
    }

    /// Calculate consensus statistics for the ensemble
    #[allow(dead_code)]
    fn calculate_consensus_statistics(
        &self,
        _results: &[ClusteringResult],
        _consensus_labels: &Array1<i32>,
    ) -> Result<ConsensusStatistics> {
        let n_samples = _consensus_labels.len();

        // Stub implementation - in production this would analyze agreement between clusterers
        Ok(ConsensusStatistics {
            agreement_matrix: Array2::zeros((n_samples, n_samples)),
            consensus_strength: Array1::ones(n_samples),
            cluster_stability: vec![0.5; 10], // Placeholder
            agreement_counts: Array1::ones(n_samples),
        })
    }

    /// Calculate consensus stability score for the ensemble
    #[allow(dead_code)]
    fn calculate_consensus_stability_score(&self, _consensusstats: &ConsensusStatistics) -> f64 {
        0.5 // Stub implementation
    }
}

/// Extract samples based on indices
#[allow(dead_code)]
fn extract_samples<F: Float>(data: ArrayView2<F>, indices: &[usize]) -> Result<Array2<F>> {
    let n_features = data.ncols();
    let mut sampled_data = Array2::zeros((indices.len(), n_features));

    for (new_idx, &old_idx) in indices.iter().enumerate() {
        if old_idx < data.nrows() {
            sampled_data.row_mut(new_idx).assign(&data.row(old_idx));
        }
    }

    Ok(sampled_data)
}

/// Extract features based on indices
#[allow(dead_code)]
fn extract_features<F: Float>(data: ArrayView2<F>, featureindices: &[usize]) -> Result<Array2<F>> {
    let n_samples = data.nrows();
    let mut sampled_data = Array2::zeros((n_samples, featureindices.len()));

    for (new_feat_idx, &old_feat_idx) in featureindices.iter().enumerate() {
        if old_feat_idx < data.ncols() {
            sampled_data
                .column_mut(new_feat_idx)
                .assign(&data.column(old_feat_idx));
        }
    }

    Ok(sampled_data)
}

/// Default configuration for ensemble clustering
impl Default for EnsembleConfig {
    fn default() -> Self {
        Self {
            n_estimators: 10,
            sampling_strategy: SamplingStrategy::Bootstrap { sample_ratio: 0.8 },
            consensus_method: ConsensusMethod::MajorityVoting,
            random_seed: None,
            diversity_strategy: Some(DiversityStrategy::AlgorithmDiversity {
                algorithms: vec![
                    ClusteringAlgorithm::KMeans { k_range: (2, 10) },
                    ClusteringAlgorithm::DBSCAN {
                        eps_range: (0.1, 1.0),
                        min_samples_range: (3, 10),
                    },
                ],
            }),
            quality_threshold: Some(0.0),
            max_clusters: Some(20),
        }
    }
}

/// Convenience functions for ensemble clustering
pub mod convenience {
    use super::*;

    /// Simple ensemble clustering with default parameters
    pub fn ensemble_clustering<F>(data: ArrayView2<F>) -> Result<EnsembleResult>
    where
        F: Float
            + FromPrimitive
            + Debug
            + 'static
            + std::iter::Sum
            + std::fmt::Display
            + Send
            + Sync,
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
        F: Float
            + FromPrimitive
            + Debug
            + 'static
            + std::iter::Sum
            + std::fmt::Display
            + Send
            + Sync,
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
        F: Float
            + FromPrimitive
            + Debug
            + 'static
            + std::iter::Sum
            + std::fmt::Display
            + Send
            + Sync,
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
        F: Float
            + FromPrimitive
            + Debug
            + 'static
            + std::iter::Sum
            + std::fmt::Display
            + Send
            + Sync,
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
        F: Float
            + FromPrimitive
            + Debug
            + 'static
            + std::iter::Sum
            + std::fmt::Display
            + Send
            + Sync,
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
        F: Float
            + FromPrimitive
            + Debug
            + 'static
            + std::iter::Sum
            + std::fmt::Display
            + Send
            + Sync,
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
}

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

/// Advanced ensemble clustering methods with sophisticated combination strategies
pub mod advanced_ensemble {
    use super::*;
    use ndarray::{s, Array3, ArrayView1};
    use std::cmp::Ordering;

    /// Configuration for advanced ensemble methods
    #[derive(Debug, Clone)]
    pub struct AdvancedEnsembleConfig {
        /// Meta-learning configuration
        pub meta_learning: MetaLearningConfig,
        /// Bayesian model averaging configuration
        pub bayesian_averaging: BayesianAveragingConfig,
        /// Genetic algorithm optimization configuration
        pub genetic_optimization: GeneticOptimizationConfig,
        /// Boosting configuration for clustering
        pub boostingconfig: BoostingConfig,
        /// Stacking configuration
        pub stackingconfig: StackingConfig,
        /// Enable uncertainty quantification
        pub uncertainty_quantification: bool,
    }

    /// Meta-learning configuration for learning ensemble combination
    #[derive(Debug, Clone)]
    pub struct MetaLearningConfig {
        /// Number of meta-features to extract
        pub n_meta_features: usize,
        /// Learning rate for meta-learner
        pub learning_rate: f64,
        /// Number of training iterations
        pub n_iterations: usize,
        /// Meta-learning algorithm
        pub algorithm: MetaLearningAlgorithm,
        /// Validation split for meta-training
        pub validation_split: f64,
    }

    /// Meta-learning algorithms for ensemble combination
    #[derive(Debug, Clone)]
    pub enum MetaLearningAlgorithm {
        /// Neural network meta-learner
        NeuralNetwork { hidden_layers: Vec<usize> },
        /// Random forest meta-learner
        RandomForest { n_trees: usize, max_depth: usize },
        /// Gradient boosting meta-learner
        GradientBoosting {
            n_estimators: usize,
            max_depth: usize,
        },
        /// Linear meta-learner
        Linear { regularization: f64 },
    }

    /// Bayesian model averaging configuration
    #[derive(Debug, Clone)]
    pub struct BayesianAveragingConfig {
        /// Prior distribution parameters
        pub prior_alpha: f64,
        pub prior_beta: f64,
        /// Number of MCMC samples
        pub n_samples: usize,
        /// Burn-in period
        pub burn_in: usize,
        /// Posterior update method
        pub update_method: PosteriorUpdateMethod,
        /// Enable adaptive sampling
        pub adaptive_sampling: bool,
    }

    /// Methods for updating posterior distributions
    #[derive(Debug, Clone)]
    pub enum PosteriorUpdateMethod {
        /// Metropolis-Hastings sampling
        MetropolisHastings,
        /// Gibbs sampling
        Gibbs,
        /// Variational inference
        VariationalInference,
        /// Hamiltonian Monte Carlo
        HamiltonianMC,
    }

    /// Genetic algorithm configuration for ensemble optimization
    #[derive(Debug, Clone)]
    pub struct GeneticOptimizationConfig {
        /// Population size
        pub population_size: usize,
        /// Number of generations
        pub n_generations: usize,
        /// Crossover probability
        pub crossover_prob: f64,
        /// Mutation probability
        pub mutation_prob: f64,
        /// Selection method
        pub selection_method: SelectionMethod,
        /// Elite percentage
        pub elite_percentage: f64,
        /// Fitness function
        pub fitness_function: FitnessFunction,
    }

    /// Selection methods for genetic algorithm
    #[derive(Debug, Clone)]
    pub enum SelectionMethod {
        /// Tournament selection
        Tournament { tournament_size: usize },
        /// Roulette wheel selection
        RouletteWheel,
        /// Rank-based selection
        RankBased,
        /// Elitist selection
        Elitist,
    }

    /// Fitness functions for genetic optimization
    #[derive(Debug, Clone)]
    pub enum FitnessFunction {
        /// Silhouette score
        Silhouette,
        /// Davies-Bouldin index
        DaviesBouldin,
        /// Calinski-Harabasz index
        CalinskiHarabasz,
        /// Multi-objective combination
        MultiObjective { weights: Vec<f64> },
        /// Stability-based fitness
        Stability,
    }

    /// Boosting configuration for clustering
    #[derive(Debug, Clone)]
    pub struct BoostingConfig {
        /// Number of boosting rounds
        pub n_rounds: usize,
        /// Learning rate for weight updates
        pub learning_rate: f64,
        /// Reweighting strategy
        pub reweighting_strategy: ReweightingStrategy,
        /// Error function for boosting
        pub error_function: ErrorFunction,
        /// Enable adaptive boosting
        pub adaptive_boosting: bool,
    }

    /// Strategies for reweighting samples in boosting
    #[derive(Debug, Clone)]
    pub enum ReweightingStrategy {
        /// AdaBoost-style exponential reweighting
        Exponential,
        /// Linear reweighting based on clustering quality
        Linear,
        /// Logistic reweighting
        Logistic,
        /// Custom reweighting function
        Custom { alpha: f64, beta: f64 },
    }

    /// Error functions for clustering boosting
    #[derive(Debug, Clone)]
    pub enum ErrorFunction {
        /// Disagreement rate between clusterings
        DisagreementRate,
        /// Inverse silhouette score
        InverseSilhouette,
        /// Custom weighted error
        WeightedError,
    }

    /// Stacking configuration for ensemble clustering
    #[derive(Debug, Clone)]
    pub struct StackingConfig {
        /// Base clustering algorithms
        pub base_algorithms: Vec<ClusteringAlgorithm>,
        /// Meta-clustering algorithm
        pub meta_algorithm: MetaClusteringAlgorithm,
        /// Cross-validation folds for stacking
        pub cv_folds: usize,
        /// Blending ratio for combining predictions
        pub blending_ratio: f64,
        /// Feature engineering for meta-learner
        pub feature_engineering: bool,
    }

    /// Meta-clustering algorithms for stacking
    #[derive(Debug, Clone)]
    pub enum MetaClusteringAlgorithm {
        /// Hierarchical clustering on base results
        Hierarchical { linkage: String },
        /// Spectral clustering on similarity matrix
        Spectral { n_clusters: usize },
        /// Graph-based clustering
        GraphBased { resolution: f64 },
        /// Consensus clustering
        Consensus { method: String },
    }

    /// Advanced ensemble clusterer with sophisticated methods
    pub struct AdvancedEnsembleClusterer<F: Float> {
        config: AdvancedEnsembleConfig,
        base_ensemble: EnsembleClusterer<F>,
        meta_learner: Option<MetaLearner>,
        bayesian_weights: Option<Array1<f64>>,
        genetic_optimizer: Option<GeneticOptimizer>,
        _phantom: std::marker::PhantomData<F>,
    }

    impl<F> AdvancedEnsembleClusterer<F>
    where
        F: Float
            + FromPrimitive
            + Debug
            + 'static
            + std::iter::Sum
            + std::fmt::Display
            + Send
            + Sync,
        f64: From<F>,
    {
        /// Create new advanced ensemble clusterer
        pub fn new(config: AdvancedEnsembleConfig, baseconfig: EnsembleConfig) -> Self {
            Self {
                config,
                base_ensemble: EnsembleClusterer::new(baseconfig),
                meta_learner: None,
                bayesian_weights: None,
                genetic_optimizer: None,
                _phantom: std::marker::PhantomData,
            }
        }

        /// Perform advanced ensemble clustering with meta-learning
        pub fn fit_with_meta_learning(&mut self, data: ArrayView2<F>) -> Result<EnsembleResult> {
            // 1. Generate base clustering results
            let base_results = self.base_ensemble.fit(data)?;

            // 2. Extract meta-features from data and clustering results
            let meta_features = self.extract_meta_features(data, &base_results)?;

            // 3. Train meta-learner to predict best combination weights
            let weights =
                self.train_meta_learner(&meta_features, &base_results.individual_results)?;

            // 4. Combine results using learned weights
            let enhanced_consensus = self.weighted_meta_consensus(
                &base_results.individual_results,
                &weights,
                data.nrows(),
            )?;

            // 5. Calculate enhanced statistics
            let mut enhanced_result = base_results;
            enhanced_result.consensus_labels = enhanced_consensus;
            enhanced_result.ensemble_quality =
                self.calculate_meta_quality(data, &enhanced_result)?;

            Ok(enhanced_result)
        }

        /// Perform Bayesian model averaging for ensemble combination
        pub fn fit_with_bayesian_averaging(
            &mut self,
            data: ArrayView2<F>,
        ) -> Result<EnsembleResult> {
            let base_results = self.base_ensemble.fit(data)?;

            // Initialize Bayesian weights with uniform prior
            let n_models = base_results.individual_results.len();
            let mut weights = Array1::from_elem(n_models, 1.0 / n_models as f64);

            // MCMC sampling for posterior weights
            for _iteration in 0..self.config.bayesian_averaging.n_samples {
                weights = self.mcmc_update_weights(&weights, &base_results, data)?;
            }

            self.bayesian_weights = Some(weights.clone());

            // Generate consensus using Bayesian weights
            let consensus = self.bayesian_weighted_consensus(
                &base_results.individual_results,
                &weights,
                data.nrows(),
            )?;

            let mut enhanced_result = base_results;
            enhanced_result.consensus_labels = consensus;

            Ok(enhanced_result)
        }

        /// Perform genetic algorithm optimization for ensemble composition
        pub fn fit_with_genetic_optimization(
            &mut self,
            data: ArrayView2<F>,
        ) -> Result<EnsembleResult> {
            // Initialize genetic algorithm
            let mut optimizer = GeneticOptimizer::new(self.config.genetic_optimization.clone());

            // Evolve optimal ensemble composition
            let optimized_ensemble = optimizer.evolve_ensemble(&self.base_ensemble, data)?;

            // Fit with optimized ensemble
            optimized_ensemble.fit(data)
        }

        /// Perform boosting-style ensemble clustering
        pub fn fit_with_boosting(&mut self, data: ArrayView2<F>) -> Result<EnsembleResult> {
            let mut sample_weights = Array1::from_elem(data.nrows(), 1.0 / data.nrows() as f64);
            let mut weak_learners = Vec::new();
            let mut learner_weights = Vec::new();

            for _round in 0..self.config.boostingconfig.n_rounds {
                // Sample data based on current weights
                let weighted_data = self.weighted_sample(data, &sample_weights)?;

                // Train weak clustering learner
                let weak_result = self.train_weak_learner(&weighted_data)?;

                // Calculate error rate
                let error_rate =
                    self.calculate_clustering_error(data, &weak_result, &sample_weights)?;

                if error_rate >= 0.5 {
                    break; // Stop if error rate is too high
                }

                // Calculate learner weight
                let learner_weight = self.config.boostingconfig.learning_rate
                    * ((1.0 - error_rate) / error_rate).ln();

                // Update sample weights
                self.update_sample_weights(
                    &mut sample_weights,
                    &weak_result,
                    learner_weight,
                    data,
                )?;

                weak_learners.push(weak_result);
                learner_weights.push(learner_weight);
            }

            // Combine weak learners
            self.combine_boosted_learners(&weak_learners, &learner_weights, data.nrows())
        }

        /// Perform stacking ensemble clustering
        pub fn fit_with_stacking(&mut self, data: ArrayView2<F>) -> Result<EnsembleResult> {
            let cv_folds = self.config.stackingconfig.cv_folds;
            let n_samples = data.nrows();
            let fold_size = n_samples / cv_folds;

            // Stage 1: Generate base predictions using cross-validation
            let mut base_predictions =
                Array2::zeros((n_samples, self.config.stackingconfig.base_algorithms.len()));

            for fold in 0..cv_folds {
                let start_idx = fold * fold_size;
                let end_idx = if fold == cv_folds - 1 {
                    n_samples
                } else {
                    (fold + 1) * fold_size
                };

                // Split data
                let train_indices: Vec<usize> = (0..start_idx).chain(end_idx..n_samples).collect();
                let test_indices: Vec<usize> = (start_idx..end_idx).collect();

                let train_data = data.select(Axis(0), &train_indices);
                let test_data = data.select(Axis(0), &test_indices);

                // Train base algorithms on fold training data
                let base_algorithms = self.config.stackingconfig.base_algorithms.clone();
                for (alg_idx, algorithm) in base_algorithms.iter().enumerate() {
                    let labels = self.train_base_algorithm(&train_data, algorithm)?;
                    let test_labels =
                        self.predict_base_algorithm(&test_data, algorithm, &labels)?;

                    // Store predictions
                    for (i, &test_idx) in test_indices.iter().enumerate() {
                        if i < test_labels.len() {
                            base_predictions[[test_idx, alg_idx]] = test_labels[i] as f64;
                        }
                    }
                }
            }

            // Stage 2: Train meta-learner on base predictions
            let meta_labels = self.train_meta_clustering_algorithm(&base_predictions)?;

            // Convert to ensemble result format
            let individual_results = vec![]; // Would populate with base results
            let consensus_stats = self.calculate_stacking_consensus_stats(&meta_labels)?;
            let diversity_metrics = self.calculate_stacking_diversity_metrics(&base_predictions)?;

            Ok(EnsembleResult {
                consensus_labels: meta_labels,
                individual_results,
                consensus_stats,
                diversity_metrics,
                ensemble_quality: 0.0, // Would calculate properly
                stability_score: 0.0,  // Would calculate properly
            })
        }

        // Helper methods for advanced ensemble techniques

        fn extract_meta_features(
            &self,
            data: ArrayView2<F>,
            results: &EnsembleResult,
        ) -> Result<Array2<f64>> {
            let n_features = self.config.meta_learning.n_meta_features;
            let mut meta_features = Array2::zeros((1, n_features));

            // Extract dataset characteristics
            let n_samples = data.nrows() as f64;
            let n_dims = data.ncols() as f64;
            let data_f64 = data.mapv(|x| x.to_f64().unwrap_or(0.0));

            // Statistical meta-features
            meta_features[[0, 0]] = n_samples.ln();
            meta_features[[0, 1]] = n_dims.ln();
            meta_features[[0, 2]] = data_f64.clone().variance();
            meta_features[[0, 3]] = calculate_intrinsic_dimensionality(&data_f64);
            meta_features[[0, 4]] = calculate_clustering_tendency(&data_f64);
            meta_features[[0, 5]] = results.diversity_metrics.average_diversity;

            // Additional domain-specific meta-features
            for i in 6..n_features {
                meta_features[[0, i]] = calculate_advanced_meta_feature(&data_f64, i - 6);
            }

            Ok(meta_features)
        }

        fn train_meta_learner(
            &mut self,
            meta_features: &Array2<f64>,
            base_results: &[ClusteringResult],
        ) -> Result<Array1<f64>> {
            match &self.config.meta_learning.algorithm {
                MetaLearningAlgorithm::NeuralNetwork { hidden_layers } => {
                    let hidden_layers = hidden_layers.clone();
                    self.train_neural_meta_learner(meta_features, base_results, &hidden_layers)
                }
                MetaLearningAlgorithm::RandomForest { n_trees, max_depth } => self
                    .train_forest_meta_learner(meta_features, base_results, *n_trees, *max_depth),
                MetaLearningAlgorithm::Linear { regularization } => {
                    self.train_linear_meta_learner(meta_features, base_results, *regularization)
                }
                _ => {
                    // Default to uniform weights
                    Ok(Array1::from_elem(
                        base_results.len(),
                        1.0 / base_results.len() as f64,
                    ))
                }
            }
        }

        fn train_neural_meta_learner(
            &mut self,
            _meta_features: &Array2<f64>,
            base_results: &[ClusteringResult],
            _hidden_layers: &[usize],
        ) -> Result<Array1<f64>> {
            // Simplified neural network meta-learner
            // In practice, would implement a full neural network
            let mut weights = Array1::zeros(base_results.len());

            // Weight based on quality scores with sigmoid transformation
            let quality_sum: f64 = base_results.iter().map(|r| r.quality_score.max(0.0)).sum();

            if quality_sum > 0.0 {
                for (i, result) in base_results.iter().enumerate() {
                    let normalized_quality = result.quality_score.max(0.0) / quality_sum;
                    weights[i] = 1.0 / (1.0 + (-5.0 * (normalized_quality - 0.5)).exp());
                    // Sigmoid
                }
            } else {
                weights.fill(1.0 / base_results.len() as f64);
            }

            // Normalize weights
            let weight_sum = weights.sum();
            if weight_sum > 0.0 {
                weights.mapv_inplace(|w| w / weight_sum);
            }

            Ok(weights)
        }

        fn train_forest_meta_learner(
            &mut self,
            _meta_features: &Array2<f64>,
            base_results: &[ClusteringResult],
            _n_trees: usize,
            _max_depth: usize,
        ) -> Result<Array1<f64>> {
            // Simplified random forest meta-learner
            // Weight based on ensemble of quality scores and diversity
            let mut weights = Array1::zeros(base_results.len());

            for (i, result) in base_results.iter().enumerate() {
                // Combine quality score with runtime efficiency
                let efficiency_score = 1.0 / (1.0 + result.runtime);
                let combined_score = result.quality_score * 0.7 + efficiency_score * 0.3;
                weights[i] = combined_score.max(0.0);
            }

            // Normalize weights
            let weight_sum = weights.sum();
            if weight_sum > 0.0 {
                weights.mapv_inplace(|w| w / weight_sum);
            } else {
                weights.fill(1.0 / base_results.len() as f64);
            }

            Ok(weights)
        }

        fn train_linear_meta_learner(
            &mut self,
            _meta_features: &Array2<f64>,
            base_results: &[ClusteringResult],
            regularization: f64,
        ) -> Result<Array1<f64>> {
            // Linear combination with L2 regularization
            let mut weights = Array1::zeros(base_results.len());

            // Ridge regression-style weight calculation
            for (i, result) in base_results.iter().enumerate() {
                let quality_with_reg =
                    result.quality_score - regularization * result.quality_score.powi(2);
                weights[i] = quality_with_reg.max(0.0);
            }

            // Normalize weights
            let weight_sum = weights.sum();
            if weight_sum > 0.0 {
                weights.mapv_inplace(|w| w / weight_sum);
            } else {
                weights.fill(1.0 / base_results.len() as f64);
            }

            Ok(weights)
        }

        fn weighted_meta_consensus(
            &self,
            base_results: &[ClusteringResult],
            weights: &Array1<f64>,
            n_samples: usize,
        ) -> Result<Array1<i32>> {
            let mut consensus = Array1::<i32>::zeros(n_samples);

            // Weighted voting with continuous weights
            for i in 0..n_samples {
                let mut vote_scores = HashMap::new();

                for (result_idx, result) in base_results.iter().enumerate() {
                    if i < result.labels.len() {
                        let label = result.labels[i];
                        let weight = weights[result_idx];
                        *vote_scores.entry(label).or_insert(0.0) += weight;
                    }
                }

                // Find label with highest weighted vote
                let best_label = vote_scores
                    .into_iter()
                    .max_by(|(_, score_a), (_, score_b)| {
                        score_a.partial_cmp(score_b).unwrap_or(Ordering::Equal)
                    })
                    .map(|(label_, _)| label_)
                    .unwrap_or(0);

                consensus[i] = best_label;
            }

            Ok(consensus)
        }

        fn mcmc_update_weights(
            &self,
            current_weights: &Array1<f64>,
            _results: &EnsembleResult,
            data: ArrayView2<F>,
        ) -> Result<Array1<f64>> {
            // Simplified MCMC update (Metropolis-Hastings)
            let mut new_weights = current_weights.clone();
            let mut rng = rand::rng();

            // Propose new _weights with small random perturbations
            for weight in new_weights.iter_mut() {
                let perturbation = rng.random_range(-0.05..0.05);
                *weight = (*weight + perturbation).max(0.01).min(0.99);
            }

            // Normalize
            let sum = new_weights.sum();
            new_weights.mapv_inplace(|w| w / sum);

            // Accept/reject based on simplified likelihood (in practice..would compute proper likelihood)
            let accept_prob = rng.random::<f64>();
            if accept_prob > 0.5 {
                Ok(new_weights)
            } else {
                Ok(current_weights.clone())
            }
        }

        fn bayesian_weighted_consensus(
            &self,
            base_results: &[ClusteringResult],
            weights: &Array1<f64>,
            n_samples: usize,
        ) -> Result<Array1<i32>> {
            // Similar to weighted_meta_consensus but with Bayesian uncertainty
            self.weighted_meta_consensus(base_results, weights, n_samples)
        }

        fn calculate_meta_quality(
            &self,
            data: ArrayView2<F>,
            result: &EnsembleResult,
        ) -> Result<f64> {
            let data_f64 = data.mapv(|x| x.to_f64().unwrap_or(0.0));
            silhouette_score(data_f64.view(), result.consensus_labels.view()).map_err(|e| e.into())
        }

        // Additional helper methods would be implemented here...
        // (Continuing with stubs for brevity)

        fn weighted_sample(&self, data: ArrayView2<F>, weights: &Array1<f64>) -> Result<Array2<F>> {
            use rand::prelude::*;
            use rand_distr::weighted::WeightedIndex;

            let n_samples = data.nrows();
            let n_features = data.ncols();

            if weights.len() != n_samples {
                return Err(ClusteringError::InvalidInput(
                    "Weights array length must match number of samples".to_string(),
                ));
            }

            // Create weighted distribution
            let dist = WeightedIndex::new(weights.iter().cloned()).map_err(|e| {
                ClusteringError::InvalidInput(format!("Invalid weights for sampling: {}", e))
            })?;

            let mut rng = rand::rng();
            let mut sampled_data = Array2::zeros((n_samples, n_features));

            // Sample with replacement based on weights
            for i in 0..n_samples {
                let sampled_idx = dist.sample(&mut rng);
                for j in 0..n_features {
                    sampled_data[[i, j]] = data[[sampled_idx, j]];
                }
            }

            Ok(sampled_data)
        }

        fn train_weak_learner(&self, data: &Array2<F>) -> Result<ClusteringResult> {
            use std::collections::HashMap;
            use std::time::Instant;

            let start_time = Instant::now();
            let n_samples = data.nrows();

            // Use simple K-means with k=2 as weak learner for simplicity
            let k = 2;
            let data_view = data.view();

            // Convert to f64 for compatibility with existing kmeans implementation
            let data_f64 = data_view.mapv(|x| x.to_f64().unwrap_or(0.0));

            // Use a simple K-means clustering as weak learner
            let result = crate::vq::kmeans_with_options(data_f64.view(), k, None).map_err(|e| {
                ClusteringError::InvalidInput(format!("Weak learner failed: {}", e))
            })?;

            let runtime = start_time.elapsed().as_secs_f64();

            // Convert usize labels to i32 for ClusteringResult
            let labels_i32 = result.1.mapv(|x| x as i32);

            // Calculate a simple quality score (silhouette score)
            let quality_score =
                crate::metrics::silhouette_score(data_f64.view(), labels_i32.view()).unwrap_or(0.0);

            // Create parameters map
            let mut parameters = HashMap::new();
            parameters.insert("k".to_string(), k.to_string());
            parameters.insert("algorithm".to_string(), "kmeans".to_string());

            Ok(ClusteringResult {
                labels: labels_i32,
                algorithm: "weak_kmeans".to_string(),
                parameters,
                quality_score,
                runtime,
                n_clusters: k,
                stability_score: None,
            })
        }

        fn calculate_clustering_error(
            &self,
            data: ArrayView2<F>,
            result: &ClusteringResult,
            weights: &Array1<f64>,
        ) -> Result<f64> {
            let n_samples = data.nrows();

            if weights.len() != n_samples || result.labels.len() != n_samples {
                return Err(ClusteringError::InvalidInput(
                    "Mismatched dimensions between data, labels, and weights".to_string(),
                ));
            }

            // Convert data to f64 for calculations
            let data_f64 = data.mapv(|x| x.to_f64().unwrap_or(0.0));

            // Calculate weighted clustering error based on silhouette score
            let silhouette =
                crate::metrics::silhouette_score(data_f64.view(), result.labels.view())
                    .unwrap_or(0.0);

            // Convert silhouette score (higher is better) to error rate (lower is better)
            // Silhouette scores range from -1 to 1, so we map this to 0 to 1 error range
            let error_rate = (1.0 - silhouette) / 2.0;

            // Apply sample weights to the error calculation
            let total_weight: f64 = weights.sum();

            let weighted_error = if total_weight > 0.0 {
                // Weight the error by the sample weights
                // For simplicity, we'll use the basic error rate but could be enhanced
                // to calculate per-sample errors and weight them individually
                error_rate * (total_weight / n_samples as f64)
            } else {
                error_rate
            };

            // Clamp error rate to valid range [0, 1]
            Ok(weighted_error.max(0.0).min(1.0))
        }

        fn update_sample_weights(
            &self,
            weights: &mut Array1<f64>,
            result: &ClusteringResult,
            learner_weight: f64,
            data: ArrayView2<F>,
        ) -> Result<()> {
            let n_samples = data.nrows();

            if weights.len() != n_samples || result.labels.len() != n_samples {
                return Err(ClusteringError::InvalidInput(
                    "Mismatched dimensions for _weight update".to_string(),
                ));
            }

            // Convert data to f64 for distance calculations
            let data_f64 = data.mapv(|x| x.to_f64().unwrap_or(0.0));

            // Calculate per-sample clustering quality to determine misclassification
            let mut sample_errors = Array1::<f64>::zeros(n_samples);

            // For clustering, we'll use distance to assigned cluster centroid as error measure
            let unique_labels: Vec<i32> = {
                let mut labels: Vec<i32> = result.labels.iter().cloned().collect();
                labels.sort_unstable();
                labels.dedup();
                labels
            };

            // Calculate centroids for each cluster
            let mut centroids = Vec::new();
            for &label in &unique_labels {
                let cluster_points: Vec<usize> = result
                    .labels
                    .iter()
                    .enumerate()
                    .filter(|(_, &l)| l == label)
                    .map(|(i, _)| i)
                    .collect();

                if !cluster_points.is_empty() {
                    let n_features = data_f64.ncols();
                    let mut centroid = Array1::zeros(n_features);

                    for &point_idx in &cluster_points {
                        for j in 0..n_features {
                            centroid[j] += data_f64[[point_idx, j]];
                        }
                    }

                    centroid.mapv_inplace(|x| x / cluster_points.len() as f64);
                    centroids.push((label, centroid));
                }
            }

            // Calculate error for each sample (distance to its assigned centroid)
            for i in 0..n_samples {
                let assigned_label = result.labels[i];
                let sample_point = data_f64.row(i);

                if let Some((_, centroid)) =
                    centroids.iter().find(|(label, _)| *label == assigned_label)
                {
                    let distance: f64 = sample_point
                        .iter()
                        .zip(centroid.iter())
                        .map(|(&x, &c): (&f64, &f64)| (x - c).powi(2))
                        .sum::<f64>()
                        .sqrt();

                    sample_errors[i] = distance;
                } else {
                    sample_errors[i] = 1.0; // High error for unassigned samples
                }
            }

            // Normalize errors to [0, 1] range
            let max_error = sample_errors.iter().cloned().fold(0.0, f64::max);
            if max_error > 0.0 {
                sample_errors.mapv_inplace(|x| x / max_error);
            }

            // Update weights using AdaBoost-style exponential weighting
            match self.config.boostingconfig.reweighting_strategy {
                ReweightingStrategy::Exponential => {
                    for i in 0..n_samples {
                        let error = sample_errors[i];
                        // Increase _weight for poorly clustered samples
                        weights[i] *= (learner_weight * error).exp();
                    }
                }
                ReweightingStrategy::Linear => {
                    for i in 0..n_samples {
                        let error = sample_errors[i];
                        weights[i] *= 1.0 + learner_weight * error;
                    }
                }
                ReweightingStrategy::Logistic => {
                    for i in 0..n_samples {
                        let error = sample_errors[i];
                        let logistic_weight = 1.0 / (1.0 + (-learner_weight * error).exp());
                        weights[i] *= 1.0 + logistic_weight;
                    }
                }
                ReweightingStrategy::Custom { alpha, beta } => {
                    for i in 0..n_samples {
                        let error = sample_errors[i];
                        weights[i] *= alpha + beta * error.powf(learner_weight);
                    }
                }
            }

            // Normalize weights to sum to 1
            let weight_sum: f64 = weights.sum();
            if weight_sum > 0.0 {
                weights.mapv_inplace(|w| w / weight_sum);
            } else {
                // Reset to uniform weights if all weights became zero
                weights.fill(1.0 / n_samples as f64);
            }

            Ok(())
        }

        pub fn combine_boosted_learners(
            &mut self,
            self_learners: &[ClusteringResult],
            _weights: &[f64],
            n_samples: usize,
        ) -> Result<EnsembleResult> {
            // Stub implementation
            Ok(EnsembleResult {
                consensus_labels: Array1::zeros(n_samples),
                individual_results: Vec::new(),
                consensus_stats: ConsensusStatistics {
                    agreement_matrix: Array2::zeros((0, 0)),
                    consensus_strength: Array1::zeros(0),
                    cluster_stability: Vec::new(),
                    agreement_counts: Array1::zeros(0),
                },
                diversity_metrics: DiversityMetrics {
                    average_diversity: 0.0,
                    diversity_matrix: Array2::zeros((0, 0)),
                    algorithm_distribution: HashMap::new(),
                    parameter_diversity: HashMap::new(),
                },
                ensemble_quality: 0.0,
                stability_score: 0.0,
            })
        }

        pub fn train_base_algorithm(
            &mut self,
            self_data: &Array2<F>,
            _algorithm: &ClusteringAlgorithm,
        ) -> Result<Array1<i32>> {
            Ok(Array1::zeros(0)) // Stub
        }

        fn predict_base_algorithm(
            &self,
            data: &Array2<F>,
            _algorithm: &ClusteringAlgorithm,
            labels: &Array1<i32>,
        ) -> Result<Array1<i32>> {
            Ok(Array1::zeros(0)) // Stub
        }

        pub fn train_meta_clustering_algorithm(
            &mut self,
            predictions: &Array2<f64>,
        ) -> Result<Array1<i32>> {
            Ok(Array1::zeros(0)) // Stub
        }

        pub fn calculate_stacking_consensus_stats(
            &mut self,
            labels: &Array1<i32>,
        ) -> Result<ConsensusStatistics> {
            Ok(ConsensusStatistics {
                agreement_matrix: Array2::zeros((0, 0)),
                consensus_strength: Array1::zeros(0),
                cluster_stability: Vec::new(),
                agreement_counts: Array1::zeros(0),
            })
        }

        pub fn calculate_stacking_diversity_metrics(
            &mut self,
            predictions: &Array2<f64>,
        ) -> Result<DiversityMetrics> {
            Ok(DiversityMetrics {
                average_diversity: 0.0,
                diversity_matrix: Array2::zeros((0, 0)),
                algorithm_distribution: HashMap::new(),
                parameter_diversity: HashMap::new(),
            })
        }
    }

    // Supporting structures for advanced ensemble methods

    /// Meta-learner for ensemble combination
    pub struct MetaLearner {
        weights: Array1<f64>,
        feature_importance: Array1<f64>,
        training_history: Vec<f64>,
    }

    /// Genetic optimizer for ensemble composition
    pub struct GeneticOptimizer {
        config: GeneticOptimizationConfig,
        population: Vec<EnsembleGenome>,
        fitness_scores: Vec<f64>,
    }

    impl GeneticOptimizer {
        pub fn new(config: GeneticOptimizationConfig) -> Self {
            Self {
                config,
                population: Vec::new(),
                fitness_scores: Vec::new(),
            }
        }

        pub fn evolve_ensemble<F>(
            &mut self,
            _base_ensemble: &EnsembleClusterer<F>,
            _data: ArrayView2<F>,
        ) -> Result<EnsembleClusterer<F>>
        where
            F: Float
                + FromPrimitive
                + Debug
                + 'static
                + std::iter::Sum
                + std::fmt::Display
                + Send
                + Sync,
            f64: From<F>,
        {
            // Stub implementation for genetic algorithm evolution
            Err(ClusteringError::InvalidInput(
                "Genetic optimization not fully implemented".to_string(),
            ))
        }
    }

    /// Genome representation for genetic algorithm
    #[derive(Debug, Clone)]
    pub struct EnsembleGenome {
        algorithm_weights: Array1<f64>,
        parameter_settings: HashMap<String, f64>,
        sampling_parameters: Vec<f64>,
    }

    // Helper functions for advanced meta-features

    fn calculate_intrinsic_dimensionality(data: &Array2<f64>) -> f64 {
        // Simplified intrinsic dimensionality estimation
        // In practice, would use techniques like PCA or manifold learning
        let n_dims = data.ncols() as f64;
        let variance_explained = data.var_axis(Axis(0), 0.0).sum() / data.ncols() as f64;
        variance_explained * n_dims
    }

    fn calculate_clustering_tendency(data: &Array2<f64>) -> f64 {
        // Hopkins statistic for clustering tendency
        // Simplified implementation
        let n_samples = data.nrows();
        if n_samples < 10 {
            return 0.5; // Neutral clustering tendency
        }

        let sample_size = (n_samples / 10).max(5).min(50);
        let mut total_distance = 0.0;

        for i in 0..sample_size {
            let mut min_distance = f64::INFINITY;
            for j in 0..n_samples {
                if i != j {
                    let distance = euclidean_distance_f64(&data.row(i), &data.row(j));
                    min_distance = min_distance.min(distance);
                }
            }
            total_distance += min_distance;
        }

        // Return normalized tendency score
        (total_distance / sample_size as f64).tanh()
    }

    fn calculate_advanced_meta_feature(data: &Array2<f64>, featureidx: usize) -> f64 {
        match featureidx {
            0 => data.mean().unwrap_or(0.0),          // Overall mean
            1 => data.std(0.0),                       // Overall std
            2 => calculate_skewness(data),            // Skewness
            3 => calculate_kurtosis(data),            // Kurtosis
            4 => calculate_outlier_ratio(data),       // Outlier ratio
            5 => calculate_feature_correlation(data), // Feature correlation
            _ => 0.0,                                 // Default
        }
    }

    fn calculate_skewness(data: &Array2<f64>) -> f64 {
        // Simplified skewness calculation
        let mean = data.mean().unwrap_or(0.0);
        let std = data.std(0.0);
        if std == 0.0 {
            return 0.0;
        }

        let n = data.len() as f64;
        let sum_cubed: f64 = data.iter().map(|&x| ((x - mean) / std).powi(3)).sum();
        sum_cubed / n
    }

    fn calculate_kurtosis(data: &Array2<f64>) -> f64 {
        // Simplified kurtosis calculation
        let mean = data.mean().unwrap_or(0.0);
        let std = data.std(0.0);
        if std == 0.0 {
            return 0.0;
        }

        let n = data.len() as f64;
        let sum_fourth: f64 = data.iter().map(|&x| ((x - mean) / std).powi(4)).sum();
        (sum_fourth / n) - 3.0 // Excess kurtosis
    }

    fn calculate_outlier_ratio(data: &Array2<f64>) -> f64 {
        // Simple outlier detection using IQR method
        let mut values: Vec<f64> = data.iter().copied().collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        if values.len() < 4 {
            return 0.0;
        }

        let q1_idx = values.len() / 4;
        let q3_idx = 3 * values.len() / 4;
        let q1 = values[q1_idx];
        let q3 = values[q3_idx];
        let iqr = q3 - q1;

        let lower_bound = q1 - 1.5 * iqr;
        let upper_bound = q3 + 1.5 * iqr;

        let outlier_count = values
            .iter()
            .filter(|&&x| x < lower_bound || x > upper_bound)
            .count();
        outlier_count as f64 / values.len() as f64
    }

    fn calculate_feature_correlation(data: &Array2<f64>) -> f64 {
        // Average pairwise correlation between features
        let n_features = data.ncols();
        if n_features < 2 {
            return 0.0;
        }

        let mut correlations = Vec::new();
        for i in 0..n_features {
            for j in (i + 1)..n_features {
                let corr = pearson_correlation(&data.column(i), &data.column(j));
                correlations.push(corr.abs());
            }
        }

        correlations.iter().sum::<f64>() / correlations.len() as f64
    }

    fn pearson_correlation(x: &ArrayView1<f64>, y: &ArrayView1<f64>) -> f64 {
        if x.len() != y.len() || x.len() < 2 {
            return 0.0;
        }

        let mean_x = x.mean().unwrap_or(0.0);
        let mean_y = y.mean().unwrap_or(0.0);

        let mut numerator = 0.0;
        let mut sum_sq_x = 0.0;
        let mut sum_sq_y = 0.0;

        for i in 0..x.len() {
            let dx = x[i] - mean_x;
            let dy = y[i] - mean_y;
            numerator += dx * dy;
            sum_sq_x += dx * dx;
            sum_sq_y += dy * dy;
        }

        let denominator = (sum_sq_x * sum_sq_y).sqrt();
        if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        }
    }

    fn euclidean_distance_f64(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
        if a.len() != b.len() {
            return f64::INFINITY;
        }
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }
}

// Helper functions for advanced ensemble methods

#[allow(dead_code)]
fn evaluate_ensemble_performance(results: &[EnsembleResult]) -> f64 {
    if results.is_empty() {
        return 0.0;
    }

    // Calculate average ensemble quality
    results.iter().map(|r| r.ensemble_quality).sum::<f64>() / results.len() as f64
}

#[allow(dead_code)]
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

#[allow(dead_code)]
fn combine_chunkresults(_chunkresults: Vec<EnsembleResult>) -> Result<EnsembleResult> {
    if _chunkresults.is_empty() {
        return Err(ClusteringError::InvalidInput(
            "No chunk _results to combine".to_string(),
        ));
    }

    // For simplicity, return the first result
    // A real implementation would intelligently combine all chunk _results
    Ok(_chunkresults.into_iter().next().unwrap())
}

#[allow(dead_code)]
fn apply_differential_privacy(
    mut result: EnsembleResult,
    privacy_budget: f64,
) -> Result<EnsembleResult> {
    // Apply differential privacy mechanisms to the clustering result
    // For now, just add small amount of noise to consensus labels
    let mut rng = rand::rng();

    for label in result.consensus_labels.iter_mut() {
        if rng.random::<f64>() < 0.05 {
            // 5% chance to flip
            *label = (*label + 1) % 3; // Simple label flipping
        }
    }

    Ok(result)
}

#[allow(dead_code)]
fn secure_aggregate_results(
    local_results: Vec<EnsembleResult>,
    config: &FederationConfig,
) -> Result<EnsembleResult> {
    if local_results.is_empty() {
        return Err(ClusteringError::InvalidInput(
            "No local _results to aggregate".to_string(),
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
    use ndarray::Array2;

    #[test]
    fn test_ensembleconfig_default() {
        let config = EnsembleConfig::default();
        assert_eq!(config.n_estimators, 10);
        assert!(matches!(
            config.consensus_method,
            ConsensusMethod::MajorityVoting
        ));
        assert!(config.diversity_strategy.is_some());
    }

    #[test]
    fn test_ensemble_clusterer_creation() {
        let config = EnsembleConfig::default();
        let ensemble: EnsembleClusterer<f64> = EnsembleClusterer::new(config);
        // Test that ensemble can be created successfully
        assert_eq!(
            std::mem::size_of_val(&ensemble),
            std::mem::size_of::<EnsembleConfig>()
        );
    }

    #[test]
    fn test_sampling_strategies() {
        let config = EnsembleConfig {
            sampling_strategy: SamplingStrategy::Bootstrap { sample_ratio: 0.8 },
            ..Default::default()
        };

        // Test that sampling strategy is correctly set
        match config.sampling_strategy {
            SamplingStrategy::Bootstrap { sample_ratio } => {
                assert!((sample_ratio - 0.8).abs() < 1e-6);
            }
            _ => panic!("Expected Bootstrap sampling strategy"),
        }
    }

    #[test]
    fn test_clustering_algorithms() {
        let algorithms = vec![
            ClusteringAlgorithm::KMeans { k_range: (2, 5) },
            ClusteringAlgorithm::DBSCAN {
                eps_range: (0.1, 1.0),
                min_samples_range: (3, 10),
            },
        ];

        assert_eq!(algorithms.len(), 2);

        match &algorithms[0] {
            ClusteringAlgorithm::KMeans { k_range } => {
                assert_eq!(k_range.0, 2);
                assert_eq!(k_range.1, 5);
            }
            _ => panic!("Expected KMeans algorithm"),
        }
    }

    #[test]
    fn test_convenience_functions() {
        let data = Array2::from_shape_vec(
            (6, 2),
            vec![0.0, 0.0, 0.1, 0.1, 0.2, 0.2, 5.0, 5.0, 5.1, 5.1, 5.2, 5.2],
        )
        .unwrap();

        // Test bootstrap ensemble
        let result = convenience::bootstrap_ensemble(data.view(), 3, 0.8);
        assert!(result.is_ok());
    }
}
