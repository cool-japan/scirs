//! Personalized Federated Learning
//!
//! This module implements personalized federated learning algorithms that
//! adapt global models to individual client preferences and data distributions.

use crate::error::{NeuralError, Result};
use crate::federated::{AggregationStrategy, ClientUpdate};
use crate::models::sequential::Sequential;
use ndarray::prelude::*;
use std::collections::HashMap;
/// Personalization strategy
#[derive(Debug, Clone)]
pub enum PersonalizationStrategy {
    /// Fine-tuning: personalize by fine-tuning global model
    FineTuning {
        /// Number of personalization epochs
        epochs: usize,
        /// Personalization learning rate
        learning_rate: f32,
    },
    /// Meta-learning: learn how to quickly adapt to new tasks
    MetaLearning {
        /// Inner loop steps
        inner_steps: usize,
        /// Outer loop learning rate
        outer_lr: f32,
        /// Inner loop learning rate
        inner_lr: f32,
    /// Multi-task learning: shared representation with task-specific heads
    MultiTask {
        /// Number of shared layers
        shared_layers: usize,
        /// Task-specific layer sizes
        task_head_sizes: Vec<usize>,
    /// Clustering: group similar clients and train cluster-specific models
    Clustering {
        /// Number of clusters
        num_clusters: usize,
        /// Clustering method
        method: ClusteringMethod,
    /// Mixture of experts: combine multiple expert models
    MixtureOfExperts {
        /// Number of expert models
        num_experts: usize,
        /// Gating network hidden size
        gating_hidden_size: usize,
}
/// Clustering methods for personalized FL
pub enum ClusteringMethod {
    /// K-means on model parameters
    KMeansParameters,
    /// K-means on loss landscapes
    KMeansLoss,
    /// Hierarchical clustering
    Hierarchical,
    /// Spectral clustering
    Spectral,
/// Personalized federated learning coordinator
pub struct PersonalizedFL {
    /// Global model
    global_model: Option<Sequential<f32>>,
    /// Client-specific models
    client_models: HashMap<usize, Sequential<f32>>,
    /// Personalization strategy
    strategy: PersonalizationStrategy,
    /// Client data statistics for clustering
    client_stats: HashMap<usize, ClientStatistics>,
    /// Clustering assignments
    cluster_assignments: HashMap<usize, usize>,
    /// Cluster models
    cluster_models: HashMap<usize, Sequential<f32>>,
    /// Personalization history
    personalization_history: Vec<PersonalizationRound>,
/// Client statistics for personalization
pub struct ClientStatistics {
    /// Data distribution (label frequencies)
    pub label_distribution: Vec<f32>,
    /// Average loss on different tasks
    pub task_performance: HashMap<String, f32>,
    /// Model parameter statistics
    pub param_stats: ParameterStatistics,
    /// Gradient statistics
    pub gradient_stats: GradientStatistics,
/// Parameter statistics
pub struct ParameterStatistics {
    /// Parameter norms per layer
    pub layer_norms: Vec<f32>,
    /// Parameter means per layer
    pub layer_means: Vec<f32>,
    /// Parameter variances per layer
    pub layer_variances: Vec<f32>,
/// Gradient statistics
pub struct GradientStatistics {
    /// Gradient norms per layer
    /// Gradient similarities with global
    pub global_similarities: Vec<f32>,
/// Personalization round information
pub struct PersonalizationRound {
    /// Round number
    pub round: usize,
    /// Client performance before personalization
    pub pre_personalization_performance: HashMap<usize, f32>,
    /// Client performance after personalization
    pub post_personalization_performance: HashMap<usize, f32>,
    /// Personalization improvements
    pub improvements: HashMap<usize, f32>,
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
    /// Personalize model for a specific client
    pub fn personalize_for_client(
        &mut self,
        client_id: usize,
        client_data: &ArrayView2<f32>,
        client_labels: &ArrayView1<usize>,
        validation_data: Option<(&ArrayView2<f32>, &ArrayView1<usize>)>,
    ) -> Result<Sequential<f32>> {
        match &self.strategy {
            PersonalizationStrategy::FineTuning {
                epochs,
                learning_rate,
            } => self.fine_tune_for_client(
                client_id,
                client_data,
                client_labels,
                *epochs,
                *learning_rate,
            ),
            PersonalizationStrategy::MetaLearning {
                inner_steps,
                outer_lr,
                inner_lr,
            } => self.meta_learn_for_client(
                *inner_steps,
                *outer_lr,
                *inner_lr,
            PersonalizationStrategy::MultiTask {
                shared_layers,
                task_head_sizes,
            } => self.multi_task_for_client(
                *shared_layers,
            PersonalizationStrategy::Clustering {
                num_clusters,
                method,
            } => self.cluster_based_personalization(
                *num_clusters,
            PersonalizationStrategy::MixtureOfExperts {
                num_experts,
                gating_hidden_size,
            } => self.mixture_of_experts_for_client(
                *num_experts,
                *gating_hidden_size,
    /// Fine-tuning based personalization
    fn fine_tune_for_client(
        // Start with global model or existing personalized model
        let mut personalized_model = if let Some(existing) = self.client_models.get(&client_id) {
            existing.clone()
        } else if let Some(ref global) = self.global_model {
            global.clone()
        } else {
            return Err(NeuralError::InvalidArgument(
                "No global model available".to_string(),
            ));
        };
        // Fine-tune on client data
        for epoch in 0..epochs {
            let batch_size = 32.min(client_data.nrows());
            let num_batches = (client_data.nrows() + batch_size - 1) / batch_size;
            for batch_idx in 0..num_batches {
                let start = batch_idx * batch_size;
                let end = ((batch_idx + 1) * batch_size).min(client_data.nrows());
                let batch_data = client_data.slice(s![start..end, ..]);
                let batch_labels = client_labels.slice(s![start..end]);
                // Simulate training step (simplified)
                // In practice, would do actual gradient computation and update
                let _loss = self.compute_loss(&personalized_model, &batch_data, &batch_labels)?;
                // personalized_model.update_weights(gradients, learning_rate);
            }
        // Store personalized model
        self.client_models
            .insert(client_id, personalized_model.clone());
        Ok(personalized_model)
    /// Meta-learning based personalization (MAML-style)
    fn meta_learn_for_client(
        if self.global_model.is_none() {
                "No global model for meta-learning".to_string(),
        let global_model = self.global_model.as_ref().unwrap();
        let mut adapted_model = global_model.clone();
        // Split data into support and query sets
        let split_point = client_data.nrows() / 2;
        let support_data = client_data.slice(s![..split_point, ..]);
        let support_labels = client_labels.slice(s![..split_point]);
        let query_data = client_data.slice(s![split_point.., ..]);
        let query_labels = client_labels.slice(s![split_point..]);
        // Inner loop: adapt to support set
        for _ in 0..inner_steps {
            let loss = self.compute_loss(&adapted_model, &support_data, &support_labels)?;
            // Compute gradients and update (simplified)
            // adapted_model.update_weights(gradients, inner_lr);
        // Evaluate on query set for meta-update
        let query_loss = self.compute_loss(&adapted_model, &query_data, &query_labels)?;
        // In practice, would compute meta-gradients and update global model
        // For now, just return the adapted model
        self.client_models.insert(client_id, adapted_model.clone());
        Ok(adapted_model)
    /// Multi-task learning personalization
    fn multi_task_for_client(
        task_head_sizes: &[usize],
        // Create model with shared representation and task-specific head
        let mut personalized_model = if let Some(ref global) = self.global_model {
            Sequential::new()
        // Add task-specific layers for this client
        // In practice, would modify the model architecture
        // Train with shared representation frozen (initially)
        let epochs = 10;
        for _epoch in 0..epochs {
            let _loss = self.compute_loss(&personalized_model, client_data, client_labels)?;
            // Update only task-specific parameters
    /// Clustering-based personalization
    fn cluster_based_personalization(
        method: &ClusteringMethod,
        // Update client statistics
        self.update_client_statistics(client_id, client_data, client_labels)?;
        // Perform clustering if not done yet
        if self.cluster_assignments.is_empty() {
            self.perform_clustering(num_clusters, method)?;
        // Get cluster assignment for this client
        let cluster_id = self
            .cluster_assignments
            .get(&client_id)
            .copied()
            .unwrap_or(0);
        // Get or create cluster model
        let cluster_model = if let Some(model) = self.cluster_models.get(&cluster_id) {
            model.clone()
            // Start with global model for new cluster
            let model = global.clone();
            self.cluster_models.insert(cluster_id, model.clone());
            model
                "No model available for clustering".to_string(),
        // Fine-tune cluster model on client data
        let personalized_model =
            self.fine_tune_for_client(client_id, client_data, client_labels, 5, 0.01)?;
    /// Mixture of experts personalization
    fn mixture_of_experts_for_client(
        // Create mixture of experts model
        // In practice, would have multiple expert networks and a gating network
        // For simplification, just return fine-tuned model
        self.fine_tune_for_client(client_id, client_data, client_labels, 10, 0.01)
    /// Update client statistics for clustering
    fn update_client_statistics(
    ) -> Result<()> {
        // Compute label distribution
        let num_classes = client_labels.iter().cloned().max().unwrap_or(0) + 1;
        let mut label_counts = vec![0; num_classes];
        for &label in client_labels {
            if label < num_classes {
                label_counts[label] += 1;
        let total = label_counts.iter().sum::<usize>() as f32;
        let label_distribution: Vec<f32> = label_counts
            .iter()
            .map(|&count| count as f32 / total)
            .collect();
        // Compute parameter statistics (simplified)
        let param_stats = ParameterStatistics {
            layer_norms: vec![1.0; 5],     // Placeholder
            layer_means: vec![0.0; 5],     // Placeholder
            layer_variances: vec![1.0; 5], // Placeholder
        // Compute gradient statistics (simplified)
        let gradient_stats = GradientStatistics {
            layer_norms: vec![0.1; 5],         // Placeholder
            global_similarities: vec![0.8; 5], // Placeholder
        let stats = ClientStatistics {
            label_distribution,
            task_performance: HashMap::new(),
            param_stats,
            gradient_stats,
        self.client_stats.insert(client_id, stats);
        Ok(())
    /// Perform clustering of clients
    fn perform_clustering(&mut self, numclusters: usize, method: &ClusteringMethod) -> Result<()> {
        let client_ids: Vec<usize> = self.client_stats.keys().cloned().collect();
        match method {
            ClusteringMethod::KMeansParameters => {
                self.kmeans_clustering_parameters(&client_ids, num_clusters)?;
            ClusteringMethod::KMeansLoss => {
                self.kmeans_clustering_loss(&client_ids, num_clusters)?;
            ClusteringMethod::Hierarchical => {
                self.hierarchical_clustering(&client_ids, num_clusters)?;
            ClusteringMethod::Spectral => {
                self.spectral_clustering(&client_ids, num_clusters)?;
    /// K-means clustering based on label distributions
    fn kmeans_clustering_parameters(
        client_ids: &[usize],
        // Simple k-means on label distributions
        use rand::prelude::*;
use ndarray::ArrayView1;
        let mut rng = rng();
        // Initialize cluster assignments randomly
        for &client_id in client_ids {
            let cluster = rng.gen_range(0..num_clusters);
            self.cluster_assignments.insert(client_id..cluster);
        // K-means iterations (simplified)
        for _iter in 0..10 {
            // Compute cluster centroids
            let mut centroids = vec![vec![0.0; 10]; num_clusters]; // Assuming 10 classes
            let mut cluster_counts = vec![0; num_clusters];
            for &client_id in client_ids {
                if let (Some(cluster), Some(stats)) = (
                    self.cluster_assignments.get(&client_id),
                    self.client_stats.get(&client_id),
                ) {
                    cluster_counts[*cluster] += 1;
                    for (i, &val) in stats.label_distribution.iter().enumerate() {
                        if i < centroids[*cluster].len() {
                            centroids[*cluster][i] += val;
                        }
                    }
                }
            // Normalize centroids
            for (centroid, &count) in centroids.iter_mut().zip(&cluster_counts) {
                if count > 0 {
                    for val in centroid.iter_mut() {
                        *val /= count as f32;
            // Reassign clients to closest clusters
                if let Some(stats) = self.client_stats.get(&client_id) {
                    let mut best_cluster = 0;
                    let mut best_distance = f32::INFINITY;
                    for (cluster_id, centroid) in centroids.iter().enumerate() {
                        let distance =
                            self.compute_distribution_distance(&stats.label_distribution, centroid);
                        if distance < best_distance {
                            best_distance = distance;
                            best_cluster = cluster_id;
                    self.cluster_assignments.insert(client_id, best_cluster);
    /// K-means clustering based on loss landscapes
    fn kmeans_clustering_loss(&mut self, client_ids: &[usize], numclusters: usize) -> Result<()> {
        // Simplified implementation - would compute actual loss landscapes
        self.kmeans_clustering_parameters(client_ids, num_clusters)
    fn hierarchical_clustering(&mut self, client_ids: &[usize], numclusters: usize) -> Result<()> {
        // Simplified implementation
    fn spectral_clustering(&mut self, client_ids: &[usize], numclusters: usize) -> Result<()> {
    /// Compute distance between two probability distributions
    fn compute_distribution_distance(&self, dist1: &[f32], dist2: &[f32]) -> f32 {
        // KL divergence (simplified)
        let mut distance = 0.0;
        for (p, q) in dist1.iter().zip(dist2.iter()) {
            if *p > 0.0 && *q > 0.0 {
                distance += p * (p / q).ln();
        distance
    /// Compute loss for a model on given data
    fn compute_loss(
        &self,
        model: &Sequential<f32>,
        data: &ArrayView2<f32>,
        labels: &ArrayView1<usize>,
    ) -> Result<f32> {
        // Simplified loss computation
        Ok(0.5) // Placeholder
    /// Evaluate personalization performance
    pub fn evaluate_personalization(
        round: usize,
        client_evaluations: &[(usize, f32, f32)], // (client_id, pre_perf, post_perf)
    ) -> PersonalizationRound {
        let mut pre_performance = HashMap::new();
        let mut post_performance = HashMap::new();
        let mut improvements = HashMap::new();
        for &(client_id, pre_perf, post_perf) in client_evaluations {
            pre_performance.insert(client_id, pre_perf);
            post_performance.insert(client_id, post_perf);
            improvements.insert(client_id, post_perf - pre_perf);
        let round_info = PersonalizationRound {
            round,
            pre_personalization_performance: pre_performance,
            post_personalization_performance: post_performance,
            improvements,
        self.personalization_history.push(round_info.clone());
        round_info
    /// Get personalization statistics
    pub fn get_personalization_stats(&self) -> PersonalizationStats {
        if self.personalization_history.is_empty() {
            return PersonalizationStats::default();
        let latest_round = self.personalization_history.last().unwrap();
        let avg_improvement = latest_round.improvements.values().sum::<f32>()
            / latest_round.improvements.len() as f32;
        let total_clients_personalized = self.client_models.len();
        PersonalizationStats {
            average_improvement: avg_improvement,
            clients_personalized: total_clients_personalized,
            total_rounds: self.personalization_history.len(),
            cluster_assignments: self.cluster_assignments.clone(),
/// Personalization statistics
#[derive(Debug, Default)]
pub struct PersonalizationStats {
    pub average_improvement: f32,
    pub clients_personalized: usize,
    pub total_rounds: usize,
    pub cluster_assignments: HashMap<usize, usize>,
/// Personalized aggregation strategy that combines global and personal updates
pub struct PersonalizedAggregation {
    /// Weight for global model
    global_weight: f32,
    /// Weight for personal updates
    personal_weight: f32,
    /// Personalization coordinator
    personalizer: PersonalizedFL,
impl PersonalizedAggregation {
    /// Create new personalized aggregation
    pub fn new(
        global_weight: f32,
        personal_weight: f32,
        strategy: PersonalizationStrategy,
    ) -> Self {
            global_weight,
            personal_weight,
            personalizer: PersonalizedFL::new(strategy),
impl AggregationStrategy for PersonalizedAggregation {
    fn aggregate(&mut self, updates: &[ClientUpdate], weights: &[f32]) -> Result<Vec<Array2<f32>>> {
        // Standard federated averaging for global model
        let num_tensors = updates[0].weight_updates.len();
        let mut aggregated = Vec::with_capacity(num_tensors);
        for tensor_idx in 0..num_tensors {
            let shape = updates[0].weight_updates[tensor_idx].shape();
            let mut weighted_sum = Array2::zeros((shape[0], shape[1]));
            for (update, &weight) in updates.iter().zip(weights.iter()) {
                if tensor_idx < update.weight_updates.len() {
                    weighted_sum = weighted_sum + weight * &update.weight_updates[tensor_idx];
            // Balance global and personal components
            weighted_sum = weighted_sum * (self.global_weight + self.personal_weight);
            aggregated.push(weighted_sum);
        Ok(aggregated)
    fn name(&self) -> &str {
        "PersonalizedAggregation"
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_personalized_fl_creation() {
        let strategy = PersonalizationStrategy::FineTuning {
            epochs: 5,
            learning_rate: 0.01,
        let pfl = PersonalizedFL::new(strategy);
        assert_eq!(pfl.client_models.len(), 0);
    fn test_clustering_strategy() {
        let strategy = PersonalizationStrategy::Clustering {
            num_clusters: 3,
            method: ClusteringMethod::KMeansParameters,
        assert_eq!(pfl.cluster_assignments.len(), 0);
    fn test_personalized_aggregation() {
        let mut aggregator = PersonalizedAggregation::new(0.7, 0.3, strategy);
        let updates = vec![ClientUpdate {
            client_id: 0,
            weight_updates: vec![Array2::ones((2, 2))],
            num_samples: 100,
            loss: 0.5,
            accuracy: 0.9,
        }];
        let weights = vec![1.0];
        let result = aggregator.aggregate(&updates, &weights).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].shape(), &[2, 2]);
