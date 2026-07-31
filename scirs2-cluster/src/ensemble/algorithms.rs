//! Main ensemble clustering implementation
//!
//! This module contains the core ensemble clustering algorithm that combines
//! multiple base clustering results using various consensus methods and
//! diversity strategies.

use super::core::*;
use crate::error::{ClusteringError, Result};
use crate::metrics::{adjusted_rand_index, silhouette_score};
use scirs2_core::ndarray::{Array1, Array2, ArrayView2};
use scirs2_core::numeric::{Float, FromPrimitive};
use scirs2_core::random::prelude::*;
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;

/// Main ensemble clustering implementation
pub struct EnsembleClusterer<F: Float> {
    config: EnsembleConfig,
    phantom: std::marker::PhantomData<F>,
}

impl<F: Float> EnsembleClusterer<F> {
    /// Borrow the ensemble's configuration.
    pub fn config(&self) -> &EnsembleConfig {
        &self.config
    }

    /// Mutably borrow the ensemble's configuration.
    ///
    /// Enables online adaptation of the ensemble composition (for example
    /// adjusting `n_estimators` based on observed performance) without rebuilding
    /// the clusterer.
    pub fn config_mut(&mut self) -> &mut EnsembleConfig {
        &mut self.config
    }
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
            Some(seed) => scirs2_core::random::rngs::StdRng::seed_from_u64(seed),
            None => scirs2_core::random::rngs::StdRng::seed_from_u64(42),
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
        rng: &mut scirs2_core::random::rngs::StdRng,
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
                                let noise = F::from(rng.random::<f64>() * 2.0 - 1.0)
                                    .expect("Operation failed")
                                    * F::from(*noise_level).expect("Failed to convert to float");
                                noisy_data[[i, j]] = noisy_data[[i, j]] + noise;
                            }
                        }
                    }
                    NoiseType::Uniform => {
                        for i in 0..n_samples {
                            for j in 0..n_features {
                                let noise =
                                    F::from((rng.random::<f64>() * 2.0 - 1.0) * noise_level)
                                        .expect("Operation failed");
                                noisy_data[[i, j]] = noisy_data[[i, j]] + noise;
                            }
                        }
                    }
                    NoiseType::Outliers { outlier_ratio } => {
                        let n_outliers = (n_samples as f64 * outlier_ratio) as usize;
                        for _ in 0..n_outliers {
                            let outlier_idx = rng.random_range(0..n_samples);
                            for j in 0..n_features {
                                let outlier_value = F::from(rng.random::<f64>() * 10.0 - 5.0)
                                    .expect("Operation failed");
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
                    Some(seed) => scirs2_core::random::rngs::StdRng::seed_from_u64(seed),
                    None => scirs2_core::random::rngs::StdRng::seed_from_u64(
                        scirs2_core::random::random(),
                    ),
                };

                // Create random projection matrix (n_features x target_dimensions)
                let mut projection_matrix = Array2::zeros((n_features, *target_dimensions));
                for i in 0..n_features {
                    for j in 0..*target_dimensions {
                        // Use Gaussian random values for projection matrix
                        let random_val = F::from(rng.random::<f64>()).expect("Operation failed");
                        let two = F::from(2.0).expect("Failed to convert constant to float");
                        let one = F::from(1.0).expect("Failed to convert constant to float");
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
        self.hierarchical_consensus(results, data, "ward")
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
    fn select_algorithm_and_parameters(
        &self,
        estimator_index: usize,
        rng: &mut scirs2_core::random::rngs::StdRng,
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
    fn generate_random_parameters(
        &self,
        algorithm: &ClusteringAlgorithm,
        rng: &mut scirs2_core::random::rngs::StdRng,
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
    fn sample_parameter_ranges(
        &self,
        parameter_ranges: &HashMap<String, ParameterRange>,
        rng: &mut scirs2_core::random::rngs::StdRng,
    ) -> Result<HashMap<String, String>> {
        let mut parameters = HashMap::new();

        for (param_name, range) in parameter_ranges {
            let value = match range {
                ParameterRange::Integer(min, max) => rng.random_range(*min..=*max).to_string(),
                ParameterRange::Float(min, max) => rng.random_range(*min..=*max).to_string(),
                ParameterRange::Categorical(choices) => {
                    choices[rng.random_range(0..choices.len())].clone()
                }
                ParameterRange::Boolean => rng.random_bool(0.5).to_string(),
            };
            parameters.insert(param_name.clone(), value);
        }

        Ok(parameters)
    }

    /// Run clustering with specified algorithm and parameters.
    ///
    /// A genuine failure from the underlying base clusterer is propagated to
    /// the caller rather than silently replaced with fabricated labels: a
    /// clustering that never actually converged must never be indistinguishable
    /// from a real one inside the ensemble's consensus. This mirrors how
    /// established ensemble implementations treat a failing base estimator
    /// (the whole `fit` fails loudly) instead of masking the failure.
    fn run_clustering(
        &self,
        data: &Array2<F>,
        algorithm: &ClusteringAlgorithm,
        parameters: &HashMap<String, String>,
    ) -> Result<Array1<i32>> {
        match algorithm {
            ClusteringAlgorithm::KMeans { .. } => {
                let requested_k = parameters
                    .get("k")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(3);
                // A bootstrap/subspace-sampled subset can legitimately be
                // smaller than the diversity strategy's k_range upper bound
                // (this is an expected, normal condition, not a genuine
                // clustering failure): never ask for more clusters than
                // there are samples to cluster. This is real parameter
                // adjustment before running the real algorithm, not the
                // silent-fabrication-of-results this function used to do.
                let k = requested_k.min(data.nrows().max(1)).max(1);

                // Use kmeans from crate
                use crate::vq::kmeans2;
                let (_, labels) = kmeans2(
                    data.view(),
                    k,
                    Some(100),   // max_iter
                    None,        // threshold
                    None,        // init method
                    None,        // missing method
                    Some(false), // check_finite
                    None,        // seed
                )?;
                Ok(labels.mapv(|x| x as i32))
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
                use crate::affinity::{affinity_propagation, AffinityPropagationOptions};
                let options = AffinityPropagationOptions {
                    damping: F::from(damping).expect("Failed to convert to float"),
                    max_iter,
                    convergence_iter,
                    preference: None, // Use default (median of similarities)
                    affinity: "euclidean".to_string(),
                    verbose: false,
                };

                let (_, labels) = affinity_propagation(data.view(), false, Some(options))?;
                Ok(labels)
            }
            _ => {
                // For any other algorithms, fallback to k-means
                let requested_k = parameters
                    .get("k")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(3);
                let k = requested_k.min(data.nrows().max(1)).max(1);

                use crate::vq::kmeans2;
                let (_, labels) = kmeans2(
                    data.view(),
                    k,
                    Some(100),
                    None,
                    None,
                    None,
                    Some(false),
                    None,
                )?;
                Ok(labels.mapv(|x| x as i32))
            }
        }
    }

    /// Count clusters in results
    fn count_clusters(&self, labels: &Array1<i32>) -> usize {
        let mut unique_labels = std::collections::HashSet::new();
        for &label in labels {
            unique_labels.insert(label);
        }
        unique_labels.len()
    }

    /// Filter results by quality
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

    /// Calculate diversity metrics for the ensemble.
    ///
    /// `diversity_matrix[i][j]` is the real pairwise disagreement between
    /// members `i` and `j`, computed as `1 - ARI(labels_i, labels_j)` (the
    /// Adjusted Rand Index the crate already provides in `metrics`), clamped
    /// to `[0, 1]` so the result stays a normalized diversity index (ARI can
    /// go negative for anti-correlated labelings, which would otherwise push
    /// `1 - ARI` above 1): two identical labelings have ARI = 1 so diversity
    /// = 0, while completely unrelated/anti-correlated labelings clamp to
    /// the maximum diversity of 1. This replaces the previous stub, which
    /// always reported a constant 0.5 average and an identity matrix
    /// regardless of what the ensemble members actually produced.
    fn calculate_diversity_metrics(
        &self,
        results: &[ClusteringResult],
    ) -> Result<DiversityMetrics> {
        let n = results.len();
        let mut diversity_matrix = Array2::<f64>::zeros((n, n));

        for i in 0..n {
            for j in (i + 1)..n {
                let ari: f64 =
                    adjusted_rand_index(results[i].labels.view(), results[j].labels.view())?;
                let diversity = (1.0 - ari).clamp(0.0, 1.0);
                diversity_matrix[[i, j]] = diversity;
                diversity_matrix[[j, i]] = diversity;
            }
        }

        let average_diversity = if n > 1 {
            let mut sum = 0.0;
            let mut count = 0usize;
            for i in 0..n {
                for j in (i + 1)..n {
                    sum += diversity_matrix[[i, j]];
                    count += 1;
                }
            }
            sum / count as f64
        } else {
            0.0
        };

        let mut algorithm_distribution = HashMap::new();
        for result in results {
            *algorithm_distribution
                .entry(result.algorithm.clone())
                .or_insert(0) += 1;
        }

        let parameter_diversity = self.calculate_parameter_diversity(results);

        Ok(DiversityMetrics {
            average_diversity,
            diversity_matrix,
            algorithm_distribution,
            parameter_diversity,
        })
    }

    /// Calculate, for each named parameter, how diverse the values used
    /// across ensemble members actually were: `0.0` when every member used
    /// the exact same value, approaching `1.0` as more members used mutually
    /// distinct values (normalized so a fully-diverse set of `total` members
    /// scores exactly `1.0`).
    fn calculate_parameter_diversity(&self, results: &[ClusteringResult]) -> HashMap<String, f64> {
        let mut values_seen: HashMap<String, HashSet<String>> = HashMap::new();
        let mut totals: HashMap<String, usize> = HashMap::new();

        for result in results {
            for (name, value) in &result.parameters {
                values_seen
                    .entry(name.clone())
                    .or_default()
                    .insert(value.clone());
                *totals.entry(name.clone()).or_insert(0) += 1;
            }
        }

        values_seen
            .into_iter()
            .map(|(name, distinct_values)| {
                let total = totals.get(&name).copied().unwrap_or(1);
                let diversity = if total > 1 {
                    (distinct_values.len() - 1) as f64 / (total - 1) as f64
                } else {
                    0.0
                };
                (name, diversity)
            })
            .collect()
    }

    /// Calculate consensus statistics for the ensemble
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
    fn calculate_consensus_stability_score(&self, _consensusstats: &ConsensusStatistics) -> f64 {
        0.5 // Stub implementation
    }
}

/// Extract samples based on indices
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
            quality_threshold: None,
            max_clusters: Some(20),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_data() -> Array2<f64> {
        Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
            .expect("Operation failed")
    }

    #[test]
    fn test_run_clustering_propagates_kmeans_failure_instead_of_fabricating_labels() {
        let clusterer = EnsembleClusterer::<f64>::new(EnsembleConfig::default());
        // Empty data cannot be clustered at all (there is no sample count to
        // clamp k down to): kmeans2 must genuinely fail, and that failure
        // must surface as a real error rather than a silently fabricated
        // round-robin labeling.
        let data = Array2::<f64>::zeros((0, 2));
        let algorithm = ClusteringAlgorithm::KMeans { k_range: (3, 3) };
        let mut parameters = HashMap::new();
        parameters.insert("k".to_string(), "3".to_string());

        let result = clusterer.run_clustering(&data, &algorithm, &parameters);
        assert!(
            result.is_err(),
            "a base clusterer that cannot possibly succeed must surface as an error, not fabricated labels"
        );
    }

    #[test]
    fn test_run_clustering_clamps_k_to_available_samples_instead_of_failing() {
        let clusterer = EnsembleClusterer::<f64>::new(EnsembleConfig::default());
        // Only 4 samples but request 50 clusters -- exactly what a
        // bootstrap-shrunk subset can legitimately look like. This must
        // gracefully clamp k down to what the data can support and run a
        // genuine clustering, rather than erroring out (asking for more
        // clusters than samples is a parameter mismatch, not a real
        // algorithmic failure) or fabricating labels.
        let data = tiny_data();
        let algorithm = ClusteringAlgorithm::KMeans { k_range: (50, 50) };
        let mut parameters = HashMap::new();
        parameters.insert("k".to_string(), "50".to_string());

        let labels = clusterer
            .run_clustering(&data, &algorithm, &parameters)
            .expect("clamped k-means should succeed");
        assert_eq!(labels.len(), 4);
        for &label in labels.iter() {
            assert!(label >= 0, "expected valid non-negative cluster labels");
        }
    }

    #[test]
    fn test_run_clustering_propagates_affinity_propagation_failure() {
        let clusterer = EnsembleClusterer::<f64>::new(EnsembleConfig::default());
        let empty_data = Array2::<f64>::zeros((0, 2));
        let algorithm = ClusteringAlgorithm::AffinityPropagation {
            damping_range: (0.5, 0.5),
        };
        let parameters = HashMap::new();

        let result = clusterer.run_clustering(&empty_data, &algorithm, &parameters);
        assert!(
            result.is_err(),
            "affinity propagation on empty data must surface as an error, not fabricated labels"
        );
    }

    #[test]
    fn test_fit_surfaces_base_clusterer_failure_as_error() {
        // Empty data cannot be clustered by any base clusterer -- every
        // ensemble member must genuinely fail, and that failure must
        // propagate out of `fit()` rather than being silently replaced with
        // a consensus built from fabricated labels.
        let config = EnsembleConfig {
            n_estimators: 3,
            sampling_strategy: SamplingStrategy::None,
            consensus_method: ConsensusMethod::MajorityVoting,
            random_seed: Some(1),
            diversity_strategy: Some(DiversityStrategy::AlgorithmDiversity {
                algorithms: vec![ClusteringAlgorithm::KMeans { k_range: (2, 2) }],
            }),
            quality_threshold: None,
            max_clusters: None,
        };
        let clusterer = EnsembleClusterer::<f64>::new(config);
        let empty_data = Array2::<f64>::zeros((0, 2));

        let result = clusterer.fit(empty_data.view());
        assert!(
            result.is_err(),
            "ensemble fit() must fail loudly when every base clusterer genuinely fails, \
             rather than silently returning a consensus built from fabricated labels"
        );
    }

    fn make_result(labels: Vec<i32>, algorithm: &str, k: &str) -> ClusteringResult {
        let mut parameters = HashMap::new();
        parameters.insert("k".to_string(), k.to_string());
        ClusteringResult::new(
            Array1::from_vec(labels),
            algorithm.to_string(),
            parameters,
            0.5,
            0.01,
        )
    }

    #[test]
    fn test_calculate_diversity_metrics_identical_members_have_zero_diversity() {
        let clusterer = EnsembleClusterer::<f64>::new(EnsembleConfig::default());
        let results = vec![
            make_result(vec![0, 0, 1, 1], "kmeans", "2"),
            make_result(vec![0, 0, 1, 1], "kmeans", "2"),
            make_result(vec![0, 0, 1, 1], "kmeans", "2"),
        ];

        let metrics = clusterer
            .calculate_diversity_metrics(&results)
            .expect("Operation failed");

        assert!(
            metrics.average_diversity.abs() < 1e-9,
            "identical members should have zero diversity, got {}",
            metrics.average_diversity
        );
        for i in 0..results.len() {
            for j in 0..results.len() {
                assert!(metrics.diversity_matrix[[i, j]].abs() < 1e-9);
            }
        }
    }

    #[test]
    fn test_calculate_diversity_metrics_orthogonal_partitions_have_high_diversity() {
        let clusterer = EnsembleClusterer::<f64>::new(EnsembleConfig::default());
        // Two maximally-disagreeing partitions of the same 6 points: one
        // groups them into 3 pairs by position, the other cuts straight
        // across those pairs into 2 halves.
        let results = vec![
            make_result(vec![0, 0, 1, 1, 2, 2], "kmeans", "3"),
            make_result(vec![0, 1, 0, 1, 0, 1], "kmeans", "3"),
        ];

        let metrics = clusterer
            .calculate_diversity_metrics(&results)
            .expect("Operation failed");

        assert!(
            metrics.average_diversity > 0.9,
            "orthogonal partitions should score high diversity, got {}",
            metrics.average_diversity
        );
        assert_ne!(
            metrics.average_diversity, 0.5,
            "must not be the old stub constant"
        );
    }

    #[test]
    fn test_calculate_parameter_diversity_reflects_real_variation() {
        let clusterer = EnsembleClusterer::<f64>::new(EnsembleConfig::default());

        let same_k = vec![
            make_result(vec![0, 1], "kmeans", "4"),
            make_result(vec![0, 1], "kmeans", "4"),
            make_result(vec![0, 1], "kmeans", "4"),
        ];
        let diverse_k = vec![
            make_result(vec![0, 1], "kmeans", "2"),
            make_result(vec![0, 1], "kmeans", "5"),
            make_result(vec![0, 1], "kmeans", "9"),
        ];

        let same_diversity = clusterer.calculate_parameter_diversity(&same_k);
        let diverse_diversity = clusterer.calculate_parameter_diversity(&diverse_k);

        assert!((same_diversity["k"] - 0.0).abs() < 1e-9);
        assert!((diverse_diversity["k"] - 1.0).abs() < 1e-9);
        assert!(diverse_diversity["k"] > same_diversity["k"]);
    }
}
