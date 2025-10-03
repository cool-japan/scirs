//! Federated Learning module
//!
//! This module provides federated learning capabilities for distributed
//! model training while preserving data privacy.

pub mod advanced_algorithms;
pub mod aggregation;
pub mod client;
pub mod communication;
pub mod fednova;
pub mod personalized;
pub mod privacy;
pub mod server;
pub mod strategies;
pub use num_algorithms::{AggregatorFactory, FedAdagrad, FedAdam, FedAvgM, FedLAG, SCAFFOLD};
pub use aggregation::{AggregationStrategy, FedAvg, FedProx, FedYogi, Krum, Median, TrimmedMean};
pub use client::{ClientConfig, FederatedClient};
pub use communication::{
    CommunicationProtocol, CompressedMessage, CompressionMethod, Message, MessageCompressor,
};
pub use fednova::{FedNova, FedNovaClient, FedNovaCoordinator, FedNovaUpdate};
pub use personalized::{
    PersonalizationStats, PersonalizationStrategy, PersonalizedAggregation, PersonalizedFL,
pub use privacy::{DifferentialPrivacy, SecureAggregation};
pub use server::{FederatedServer, ServerConfig};
pub use strategies::{ClientSelection, SamplingStrategy};
use crate::error::Result;
use crate::models::sequential::Sequential;
use scirs2_core::ndarray::prelude::*;
use std::collections::HashMap;
/// Configuration for federated learning
#[derive(Debug, Clone)]
pub struct FederatedConfig {
    /// Number of communication rounds
    pub num_rounds: usize,
    /// Clients per round
    pub clients_per_round: usize,
    /// Local epochs per client
    pub local_epochs: usize,
    /// Local batch size
    pub local_batch_size: usize,
    /// Learning rate
    pub learning_rate: f32,
    /// Aggregation strategy
    pub aggregation_strategy: String,
    /// Privacy budget (epsilon for differential privacy)
    pub privacy_budget: Option<f64>,
    /// Enable secure aggregation
    pub secure_aggregation: bool,
    /// Client selection strategy
    pub client_selection: String,
    /// Minimum clients required
    pub min_clients: usize,
    /// Communication compression
    pub enable_compression: bool,
    /// Compression ratio
    pub compression_ratio: f32,
}
impl Default for FederatedConfig {
    fn default() -> Self {
        Self {
            num_rounds: 100,
            clients_per_round: 10,
            local_epochs: 5,
            local_batch_size: 32,
            learning_rate: 0.01,
            aggregation_strategy: "fedavg".to_string(),
            privacy_budget: None,
            secure_aggregation: false,
            client_selection: "random".to_string(),
            min_clients: 2,
            enable_compression: false,
            compression_ratio: 0.1,
        }
    }
/// Federated learning coordinator
pub struct FederatedLearning {
    config: FederatedConfig,
    server: FederatedServer,
    clients: Vec<FederatedClient>,
    communication_rounds: Vec<RoundStatistics>,
/// Statistics for a communication round
pub struct RoundStatistics {
    /// Round number
    pub round: usize,
    /// Number of participating clients
    pub num_clients: usize,
    /// Average loss across clients
    pub avg_loss: f32,
    /// Average accuracy across clients
    pub avg_accuracy: f32,
    /// Communication cost in bytes
    pub communication_cost: usize,
    /// Round duration in seconds
    pub duration: f64,
    /// Privacy spent (epsilon)
    pub privacy_spent: Option<f64>,
impl FederatedLearning {
    /// Create a new federated learning instance
    pub fn new(_config: FederatedConfig, numclients: usize) -> Result<Self> {
        let server = FederatedServer::new(ServerConfig::from(&_config))?;
        let mut clients = Vec::with_capacity(num_clients);
        for i in 0..num_clients {
            let client_config = ClientConfig {
                client_id: i,
                local_epochs: config.local_epochs,
                batch_size: config.local_batch_size,
                learning_rate: config.learning_rate,
                enable_privacy: config.privacy_budget.is_some(),
                privacy_budget: config.privacy_budget,
            };
            clients.push(FederatedClient::new(client_config)?);
        Ok(Self {
            config,
            server,
            clients,
            communication_rounds: Vec::new(),
        })
    /// Run federated training
    pub fn train(
        &mut self,
        global_model: &mut Sequential<f32>,
        client_data: &HashMap<usize, (ArrayView2<f32>, ArrayView1<usize>)>,
    ) -> Result<()> {
        for round in 0..self.config.num_rounds {
            let round_start = std::time::Instant::now();
            // Select clients for this round
            let selected_clients = self.select_clients()?;
            // Send global model to selected clients
            let model_params = self.server.get_model_parameters(global_model)?;
            // Collect client updates
            let mut client_updates = Vec::new();
            let mut round_losses = Vec::new();
            let mut round_accuracies = Vec::new();
            let mut communication_bytes = 0;
            for &client_id in &selected_clients {
                if let Some((data, labels)) = client_data.get(&client_id) {
                    // Client trains on local data
                    let update =
                        self.clients[client_id].train_on_local_data(&model_params, data, labels)?;
                    communication_bytes += update.size_bytes();
                    round_losses.push(update.loss);
                    round_accuracies.push(update.accuracy);
                    client_updates.push(update);
                }
            }
            // Aggregate updates
            let aggregated_update = self.server.aggregate_updates(&client_updates)?;
            // Update global model
            self.server
                .update_global_model(global_model, &aggregated_update)?;
            // Record round statistics
            let round_stats = RoundStatistics {
                round,
                num_clients: selected_clients.len(),
                avg_loss: round_losses.iter().sum::<f32>() / round_losses.len() as f32,
                avg_accuracy: round_accuracies.iter().sum::<f32>() / round_accuracies.len() as f32,
                communication_cost: communication_bytes,
                duration: round_start.elapsed().as_secs_f64(),
                privacy_spent: self.calculate_privacy_spent(round),
            self.communication_rounds.push(round_stats);
            // Check for early stopping
            if self.should_stop_early() {
                break;
        Ok(())
    /// Select clients for a round
    fn select_clients(&self) -> Result<Vec<usize>> {
        use scirs2_core::random::prelude::*;
use scirs2_core::ndarray::ArrayView1;
use scirs2_core::random::seq::SliceRandom;
        let mut rng = rng();
        match self.config.client_selection.as_str() {
            "random" => {
                // Random selection
                let mut client_indices: Vec<usize> = (0..self.clients.len()).collect();
                client_indices.shuffle(&mut rng);
                Ok(client_indices
                    .into_iter()
                    .take(self.config.clients_per_round)
                    .collect())
            "importance" => {
                // Importance sampling based on data size
                // Simplified implementation
                let mut weighted_indices: Vec<(usize, f32)> = self
                    .clients
                    .iter()
                    .enumerate()
                    .map(|(i_)| (i, 1.0 + rng.random::<f32>()))
                    .collect();
                weighted_indices.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                Ok(weighted_indices
                    .map(|(i_)| i)
            _ => {
                // Default to random
                self.select_clients()
    /// Calculate privacy spent up to current round
    fn calculate_privacy_spent(&self, round: usize) -> Option<f64> {
        if let Some(budget_per_round) = self.config.privacy_budget {
            Some(budget_per_round * (round + 1) as f64)
        } else {
            None
    /// Check if training should stop early
    fn should_stop_early(&self) -> bool {
        if self.communication_rounds.len() < 10 {
            return false;
        // Check if loss has plateaued
        let recent_losses: Vec<f32> = self
            .communication_rounds
            .iter()
            .rev()
            .take(5)
            .map(|r| r.avg_loss)
            .collect();
        let loss_variance = self.calculate_variance(&recent_losses);
        loss_variance < 0.0001
    /// Calculate variance of values
    fn calculate_variance(&self, values: &[f32]) -> f32 {
        if values.is_empty() {
            return 0.0;
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / values.len() as f32;
        variance
    /// Get training history
    pub fn get_history(&self) -> &[RoundStatistics] {
        &self.communication_rounds
    /// Export training metrics
    pub fn export_metrics(&self, path: &str) -> Result<()> {
        use std::fs::File;
        use std::io::Write;
        let mut file = File::create(path)?;
        writeln!(
            file,
            "round,num_clients,avg_loss,avg_accuracy,communication_cost,duration,privacy_spent"
        )?;
        for round in &self.communication_rounds {
            writeln!(
                file,
                "{},{},{},{},{},{},{}",
                round.round,
                round.num_clients,
                round.avg_loss,
                round.avg_accuracy,
                round.communication_cost,
                round.duration,
                round.privacy_spent.unwrap_or(0.0)
            )?;
/// Client update information
pub struct ClientUpdate {
    /// Client ID
    pub client_id: usize,
    /// Model weight updates
    pub weight_updates: Vec<Array2<f32>>,
    /// Number of samples used
    pub num_samples: usize,
    /// Training loss
    pub loss: f32,
    /// Training accuracy
    pub accuracy: f32,
impl ClientUpdate {
    /// Calculate size in bytes
    pub fn size_bytes(&self) -> usize {
        self.weight_updates
            .map(|w| w.len() * std::mem::size_of::<f32>())
            .sum()
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_federated_config_default() {
        let config = FederatedConfig::default();
        assert_eq!(config.num_rounds, 100);
        assert_eq!(config.clients_per_round, 10);
        assert_eq!(config.aggregation_strategy, "fedavg");
    fn test_client_selection() {
        let fl = FederatedLearning::new(config, 20).unwrap();
        let selected = fl.select_clients().unwrap();
        assert_eq!(selected.len(), 10);
        assert!(selected.iter().all(|&id| id < 20));
