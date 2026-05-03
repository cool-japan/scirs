//! Personalized Federated Learning
//!
//! This module implements personalized federated learning algorithms that
//! adapt global models to individual client preferences and data distributions.

use crate::error::{NeuralError, Result};
use crate::federated::{AggregationStrategy, ClientUpdate};
use crate::models::sequential::Sequential;
use scirs2_core::ndarray::prelude::*;
use std::collections::HashMap;

/// Personalization strategy
#[derive(Debug, Clone)]
pub enum PersonalizationStrategy {
    /// Fine-tuning: personalize by fine-tuning global model
    FineTuning { epochs: usize, learning_rate: f32 },
    /// Meta-learning: learn how to quickly adapt to new tasks
    MetaLearning {
        inner_steps: usize,
        outer_lr: f32,
        inner_lr: f32,
    },
    /// Multi-task learning: shared representation with task-specific heads
    MultiTask {
        shared_layers: usize,
        task_head_sizes: Vec<usize>,
    },
    /// Clustering: group similar clients and train cluster-specific models
    Clustering {
        num_clusters: usize,
        method: ClusteringMethod,
    },
    /// Mixture of experts: combine multiple expert models
    MixtureOfExperts {
        num_experts: usize,
        gating_hidden_size: usize,
    },
}

/// Clustering methods for personalized FL
#[derive(Debug, Clone)]
pub enum ClusteringMethod {
    KMeansParameters,
    KMeansLoss,
    Hierarchical,
    Spectral,
}

/// Personalized federated learning coordinator
pub struct PersonalizedFL {
    /// Global model
    pub global_model: Option<Sequential<f32>>,
    /// Client-specific models
    pub client_models: HashMap<usize, Sequential<f32>>,
    /// Personalization strategy
    strategy: PersonalizationStrategy,
    /// Client data statistics for clustering
    client_stats: HashMap<usize, ClientStatistics>,
    /// Clustering assignments
    pub cluster_assignments: HashMap<usize, usize>,
    /// Cluster models
    cluster_models: HashMap<usize, Sequential<f32>>,
    /// Personalization history
    personalization_history: Vec<PersonalizationRound>,
}

/// Client statistics for personalization
pub struct ClientStatistics {
    pub label_distribution: Vec<f32>,
    pub task_performance: HashMap<String, f32>,
    pub param_stats: ParameterStatistics,
    pub gradient_stats: GradientStatistics,
}

/// Parameter statistics
pub struct ParameterStatistics {
    pub layer_norms: Vec<f32>,
    pub layer_means: Vec<f32>,
    pub layer_variances: Vec<f32>,
}

/// Gradient statistics
pub struct GradientStatistics {
    pub layer_norms: Vec<f32>,
    pub global_similarities: Vec<f32>,
}

/// Personalization round information
pub struct PersonalizationRound {
    pub round: usize,
    pub pre_personalization_performance: HashMap<usize, f32>,
    pub post_personalization_performance: HashMap<usize, f32>,
    pub improvements: HashMap<usize, f32>,
}

impl PersonalizedFL {
    /// Create new personalized FL coordinator
    pub fn new(strategy: PersonalizationStrategy) -> Self {
        Self {
            global_model: None,
            client_models: HashMap::new(),
            strategy,
            client_stats: HashMap::new(),
            cluster_assignments: HashMap::new(),
            cluster_models: HashMap::new(),
            personalization_history: Vec::new(),
        }
    }

    /// Set global model
    pub fn set_global_model(&mut self, model: Sequential<f32>) {
        self.global_model = Some(model);
    }

    /// Personalize model for a specific client
    pub fn personalize_for_client(
        &mut self,
        client_id: usize,
        client_data: &ArrayView2<f32>,
        client_labels: &ArrayView1<usize>,
        _validation_data: Option<(&ArrayView2<f32>, &ArrayView1<usize>)>,
    ) -> Result<Sequential<f32>> {
        match self.strategy.clone() {
            PersonalizationStrategy::FineTuning {
                epochs,
                learning_rate,
            } => self.fine_tune_for_client(
                client_id,
                client_data,
                client_labels,
                epochs,
                learning_rate,
            ),
            PersonalizationStrategy::MetaLearning {
                inner_steps,
                outer_lr: _,
                inner_lr,
            } => self.meta_learn_for_client(
                client_id,
                client_data,
                client_labels,
                inner_steps,
                inner_lr,
            ),
            PersonalizationStrategy::MultiTask {
                shared_layers: _,
                task_head_sizes,
            } => {
                self.multi_task_for_client(client_id, client_data, client_labels, &task_head_sizes)
            }
            PersonalizationStrategy::Clustering {
                num_clusters,
                method,
            } => self.cluster_based_personalization(
                client_id,
                client_data,
                client_labels,
                num_clusters,
                &method.clone(),
            ),
            PersonalizationStrategy::MixtureOfExperts {
                num_experts: _,
                gating_hidden_size: _,
            } => self.fine_tune_for_client(client_id, client_data, client_labels, 10, 0.01),
        }
    }

    /// Fine-tuning based personalization
    fn fine_tune_for_client(
        &mut self,
        client_id: usize,
        client_data: &ArrayView2<f32>,
        client_labels: &ArrayView1<usize>,
        epochs: usize,
        _learning_rate: f32,
    ) -> Result<Sequential<f32>> {
        let mut personalized_model = if let Some(existing) = self.client_models.get(&client_id) {
            existing.clone()
        } else if let Some(ref global) = self.global_model {
            global.clone()
        } else {
            return Err(NeuralError::InvalidArgument(
                "No global model available".to_string(),
            ));
        };

        // Fine-tune on client data (simplified simulation)
        for _epoch in 0..epochs {
            let batch_size = 32.min(client_data.nrows());
            let num_batches = client_data.nrows().div_ceil(batch_size);
            for batch_idx in 0..num_batches {
                let start = batch_idx * batch_size;
                let end = ((batch_idx + 1) * batch_size).min(client_data.nrows());
                let _batch_data = client_data.slice(s![start..end, ..]);
                let _batch_labels = client_labels.slice(s![start..end]);
                // Simulated training step
            }
        }

        self.client_models
            .insert(client_id, personalized_model.clone());
        Ok(personalized_model)
    }

    /// Meta-learning based personalization (MAML-style)
    fn meta_learn_for_client(
        &mut self,
        client_id: usize,
        client_data: &ArrayView2<f32>,
        client_labels: &ArrayView1<usize>,
        inner_steps: usize,
        _inner_lr: f32,
    ) -> Result<Sequential<f32>> {
        let adapted_model = if let Some(ref global) = self.global_model {
            global.clone()
        } else {
            return Err(NeuralError::InvalidArgument(
                "No global model for meta-learning".to_string(),
            ));
        };

        let split_point = client_data.nrows() / 2;
        let _support_data = client_data.slice(s![..split_point, ..]);
        let _support_labels = client_labels.slice(s![..split_point]);

        // Inner loop simulation
        for _ in 0..inner_steps {
            // Simulated gradient update
        }

        self.client_models.insert(client_id, adapted_model.clone());
        Ok(adapted_model)
    }

    /// Multi-task learning personalization
    fn multi_task_for_client(
        &mut self,
        client_id: usize,
        client_data: &ArrayView2<f32>,
        client_labels: &ArrayView1<usize>,
        _task_head_sizes: &[usize],
    ) -> Result<Sequential<f32>> {
        let personalized_model = if let Some(ref global) = self.global_model {
            global.clone()
        } else {
            Sequential::new()
        };

        // Simulate multi-task training
        for _epoch in 0..10 {
            let _loss = self.compute_loss(&personalized_model, client_data, client_labels)?;
        }

        self.client_models
            .insert(client_id, personalized_model.clone());
        Ok(personalized_model)
    }

    /// Clustering-based personalization
    fn cluster_based_personalization(
        &mut self,
        client_id: usize,
        client_data: &ArrayView2<f32>,
        client_labels: &ArrayView1<usize>,
        num_clusters: usize,
        method: &ClusteringMethod,
    ) -> Result<Sequential<f32>> {
        self.update_client_statistics(client_id, client_data, client_labels)?;

        if self.cluster_assignments.is_empty() {
            self.perform_clustering(num_clusters, method)?;
        }

        let _cluster_id = self
            .cluster_assignments
            .get(&client_id)
            .copied()
            .unwrap_or(0);

        self.fine_tune_for_client(client_id, client_data, client_labels, 5, 0.01)
    }

    /// Update client statistics for clustering
    fn update_client_statistics(
        &mut self,
        client_id: usize,
        _client_data: &ArrayView2<f32>,
        client_labels: &ArrayView1<usize>,
    ) -> Result<()> {
        let num_classes = client_labels.iter().cloned().max().unwrap_or(0) + 1;
        let mut label_counts = vec![0_usize; num_classes];
        for &label in client_labels {
            if label < num_classes {
                label_counts[label] += 1;
            }
        }
        let total = label_counts.iter().sum::<usize>().max(1) as f32;
        let label_distribution: Vec<f32> = label_counts
            .iter()
            .map(|&count| count as f32 / total)
            .collect();

        let param_stats = ParameterStatistics {
            layer_norms: vec![1.0; 5],
            layer_means: vec![0.0; 5],
            layer_variances: vec![1.0; 5],
        };
        let gradient_stats = GradientStatistics {
            layer_norms: vec![0.1; 5],
            global_similarities: vec![0.8; 5],
        };
        let stats = ClientStatistics {
            label_distribution,
            task_performance: HashMap::new(),
            param_stats,
            gradient_stats,
        };
        self.client_stats.insert(client_id, stats);
        Ok(())
    }

    /// Perform clustering of clients
    fn perform_clustering(&mut self, num_clusters: usize, method: &ClusteringMethod) -> Result<()> {
        let client_ids: Vec<usize> = self.client_stats.keys().cloned().collect();
        match method {
            ClusteringMethod::KMeansParameters => {
                self.kmeans_clustering_parameters(&client_ids, num_clusters)?;
            }
            ClusteringMethod::KMeansLoss => {
                self.kmeans_clustering_loss(&client_ids, num_clusters)?;
            }
            ClusteringMethod::Hierarchical => {
                self.hierarchical_clustering(&client_ids, num_clusters)?;
            }
            ClusteringMethod::Spectral => {
                self.spectral_clustering(&client_ids, num_clusters)?;
            }
        }
        Ok(())
    }

    /// K-means clustering based on label distributions
    fn kmeans_clustering_parameters(
        &mut self,
        client_ids: &[usize],
        num_clusters: usize,
    ) -> Result<()> {
        use scirs2_core::random::{rng, RngExt};
        let mut rng_inst = rng();
        // Initialize cluster assignments randomly
        for &client_id in client_ids {
            let cluster = rng_inst.random_range(0..num_clusters);
            self.cluster_assignments.insert(client_id, cluster);
        }
        // K-means iterations (simplified)
        for _iter in 0..10 {
            let mut centroids = vec![vec![0.0_f32; 10]; num_clusters];
            let mut cluster_counts = vec![0_usize; num_clusters];
            for &client_id in client_ids {
                if let (Some(&cluster), Some(stats)) = (
                    self.cluster_assignments.get(&client_id),
                    self.client_stats.get(&client_id),
                ) {
                    cluster_counts[cluster] += 1;
                    for (i, &val) in stats.label_distribution.iter().enumerate() {
                        if i < centroids[cluster].len() {
                            centroids[cluster][i] += val;
                        }
                    }
                }
            }
            for (centroid, &count) in centroids.iter_mut().zip(&cluster_counts) {
                if count > 0 {
                    for val in centroid.iter_mut() {
                        *val /= count as f32;
                    }
                }
            }
            // Reassign clients to closest clusters
            for &client_id in client_ids {
                if let Some(stats) = self.client_stats.get(&client_id) {
                    let mut best_cluster = 0;
                    let mut best_distance = f32::INFINITY;
                    for (cluster_id, centroid) in centroids.iter().enumerate() {
                        let distance =
                            self.compute_distribution_distance(&stats.label_distribution, centroid);
                        if distance < best_distance {
                            best_distance = distance;
                            best_cluster = cluster_id;
                        }
                    }
                    self.cluster_assignments.insert(client_id, best_cluster);
                }
            }
        }
        Ok(())
    }

    fn kmeans_clustering_loss(&mut self, client_ids: &[usize], num_clusters: usize) -> Result<()> {
        self.kmeans_clustering_parameters(client_ids, num_clusters)
    }

    fn hierarchical_clustering(&mut self, client_ids: &[usize], num_clusters: usize) -> Result<()> {
        self.kmeans_clustering_parameters(client_ids, num_clusters)
    }

    fn spectral_clustering(&mut self, client_ids: &[usize], num_clusters: usize) -> Result<()> {
        self.kmeans_clustering_parameters(client_ids, num_clusters)
    }

    /// Compute distance between two probability distributions
    fn compute_distribution_distance(&self, dist1: &[f32], dist2: &[f32]) -> f32 {
        let mut distance = 0.0_f32;
        for (p, q) in dist1.iter().zip(dist2.iter()) {
            if *p > 0.0 && *q > 0.0 {
                distance += p * (p / q).ln();
            }
        }
        distance
    }

    /// Compute loss for a model on given data (simplified)
    fn compute_loss(
        &self,
        _model: &Sequential<f32>,
        _data: &ArrayView2<f32>,
        _labels: &ArrayView1<usize>,
    ) -> Result<f32> {
        Ok(0.5)
    }

    /// Evaluate personalization performance
    pub fn evaluate_personalization(
        &mut self,
        round: usize,
        client_evaluations: &[(usize, f32, f32)],
    ) -> PersonalizationRound {
        let mut pre_performance = HashMap::new();
        let mut post_performance = HashMap::new();
        let mut improvements = HashMap::new();
        for &(client_id, pre_perf, post_perf) in client_evaluations {
            pre_performance.insert(client_id, pre_perf);
            post_performance.insert(client_id, post_perf);
            improvements.insert(client_id, post_perf - pre_perf);
        }
        let round_info = PersonalizationRound {
            round,
            pre_personalization_performance: pre_performance,
            post_personalization_performance: post_performance,
            improvements,
        };
        self.personalization_history.push(round_info);
        self.personalization_history
            .last()
            .expect("just pushed")
            .clone_round()
    }

    /// Get personalization statistics
    pub fn get_personalization_stats(&self) -> PersonalizationStats {
        if self.personalization_history.is_empty() {
            return PersonalizationStats::default();
        }
        let latest_round = self.personalization_history.last().expect("non-empty");
        let avg_improvement = if latest_round.improvements.is_empty() {
            0.0
        } else {
            latest_round.improvements.values().sum::<f32>() / latest_round.improvements.len() as f32
        };
        PersonalizationStats {
            average_improvement: avg_improvement,
            clients_personalized: self.client_models.len(),
            total_rounds: self.personalization_history.len(),
            cluster_assignments: self.cluster_assignments.clone(),
        }
    }
}

impl PersonalizationRound {
    fn clone_round(&self) -> Self {
        PersonalizationRound {
            round: self.round,
            pre_personalization_performance: self.pre_personalization_performance.clone(),
            post_personalization_performance: self.post_personalization_performance.clone(),
            improvements: self.improvements.clone(),
        }
    }
}

/// Personalization statistics
#[derive(Debug, Default)]
pub struct PersonalizationStats {
    pub average_improvement: f32,
    pub clients_personalized: usize,
    pub total_rounds: usize,
    pub cluster_assignments: HashMap<usize, usize>,
}

/// Personalized aggregation strategy that combines global and personal updates
pub struct PersonalizedAggregation {
    /// Weight for global model
    global_weight: f32,
    /// Weight for personal updates
    personal_weight: f32,
    /// Personalization coordinator
    #[allow(dead_code)]
    personalizer: PersonalizedFL,
}

impl PersonalizedAggregation {
    /// Create new personalized aggregation
    pub fn new(
        global_weight: f32,
        personal_weight: f32,
        strategy: PersonalizationStrategy,
    ) -> Self {
        Self {
            global_weight,
            personal_weight,
            personalizer: PersonalizedFL::new(strategy),
        }
    }
}

impl AggregationStrategy for PersonalizedAggregation {
    fn aggregate(&mut self, updates: &[ClientUpdate], weights: &[f32]) -> Result<Vec<Array2<f32>>> {
        if updates.is_empty() {
            return Ok(Vec::new());
        }
        let num_tensors = updates[0].weight_updates.len();
        let mut aggregated = Vec::with_capacity(num_tensors);
        for tensor_idx in 0..num_tensors {
            let shape = updates[0].weight_updates[tensor_idx].shape();
            let mut weighted_sum = Array2::zeros((shape[0], shape[1]));
            for (update, &weight) in updates.iter().zip(weights.iter()) {
                if tensor_idx < update.weight_updates.len() {
                    weighted_sum = weighted_sum + weight * &update.weight_updates[tensor_idx];
                }
            }
            // Balance global and personal components
            weighted_sum *= self.global_weight + self.personal_weight;
            aggregated.push(weighted_sum);
        }
        Ok(aggregated)
    }

    fn name(&self) -> &str {
        "PersonalizedAggregation"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_personalized_fl_creation() {
        let strategy = PersonalizationStrategy::FineTuning {
            epochs: 5,
            learning_rate: 0.01,
        };
        let pfl = PersonalizedFL::new(strategy);
        assert_eq!(pfl.client_models.len(), 0);
    }

    #[test]
    fn test_clustering_strategy() {
        let strategy = PersonalizationStrategy::Clustering {
            num_clusters: 3,
            method: ClusteringMethod::KMeansParameters,
        };
        let pfl = PersonalizedFL::new(strategy);
        assert_eq!(pfl.cluster_assignments.len(), 0);
    }

    #[test]
    fn test_personalized_aggregation() {
        let strategy = PersonalizationStrategy::FineTuning {
            epochs: 5,
            learning_rate: 0.01,
        };
        let mut aggregator = PersonalizedAggregation::new(0.7, 0.3, strategy);
        let updates = vec![ClientUpdate {
            client_id: 0,
            weight_updates: vec![Array2::ones((2, 2))],
            num_samples: 100,
            loss: 0.5,
            accuracy: 0.9,
        }];
        let weights = vec![1.0];
        let result = aggregator
            .aggregate(&updates, &weights)
            .expect("aggregate failed");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].shape(), &[2, 2]);
    }
}
