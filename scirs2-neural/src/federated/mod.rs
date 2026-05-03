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

pub use advanced_algorithms::{AggregatorFactory, FedAdagrad, FedAdam, FedAvgM, FedLAG, SCAFFOLD};
pub use aggregation::{AggregationStrategy, FedAvg, FedProx, FedYogi, Krum, Median, TrimmedMean};
pub use client::{ClientConfig, FederatedClient};
pub use communication::{
    CommunicationProtocol, CompressedMessage, CompressionMethod, Message, MessageCompressor,
};
pub use fednova::{FedNova, FedNovaClient, FedNovaCoordinator, FedNovaUpdate};
pub use personalized::{
    PersonalizationStats, PersonalizationStrategy, PersonalizedAggregation, PersonalizedFL,
};
pub use privacy::{DifferentialPrivacy, SecureAggregation};
pub use server::{FederatedServer, ServerConfig};
pub use strategies::{ClientSelection, SamplingStrategy};

use crate::error::{NeuralError, Result};
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
}

/// Statistics for a communication round
#[derive(Debug, Clone)]
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
}

/// Client update information
#[derive(Debug, Clone)]
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
}

impl ClientUpdate {
    /// Calculate size in bytes
    pub fn size_bytes(&self) -> usize {
        self.weight_updates
            .iter()
            .map(|w| w.len() * std::mem::size_of::<f32>())
            .sum()
    }
}

/// Federated learning coordinator
pub struct FederatedLearning {
    config: FederatedConfig,
    server: FederatedServer,
    clients: Vec<FederatedClient>,
    communication_rounds: Vec<RoundStatistics>,
}

impl FederatedLearning {
    /// Create a new federated learning instance
    pub fn new(config: FederatedConfig, num_clients: usize) -> Result<Self> {
        let server = FederatedServer::new(ServerConfig::from(&config))?;
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
        }
        Ok(Self {
            config,
            server,
            clients,
            communication_rounds: Vec::new(),
        })
    }

    /// Run federated training
    pub fn train(
        &mut self,
        global_model: &mut Sequential<f32>,
        client_data: &HashMap<usize, (Array2<f32>, Array1<usize>)>,
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
                    let update = self.clients[client_id].train_on_local_data(
                        &model_params,
                        &data.view(),
                        &labels.view(),
                    )?;
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
            let avg_loss = if round_losses.is_empty() {
                0.0
            } else {
                round_losses.iter().sum::<f32>() / round_losses.len() as f32
            };
            let avg_accuracy = if round_accuracies.is_empty() {
                0.0
            } else {
                round_accuracies.iter().sum::<f32>() / round_accuracies.len() as f32
            };
            // Record round statistics
            let round_stats = RoundStatistics {
                round,
                num_clients: selected_clients.len(),
                avg_loss,
                avg_accuracy,
                communication_cost: communication_bytes,
                duration: round_start.elapsed().as_secs_f64(),
                privacy_spent: self.calculate_privacy_spent(round),
            };
            self.communication_rounds.push(round_stats);
            // Check for early stopping
            if self.should_stop_early() {
                break;
            }
        }
        Ok(())
    }

    /// Select clients for a round
    fn select_clients(&self) -> Result<Vec<usize>> {
        use scirs2_core::random::rng;
        use scirs2_core::random::seq::SliceRandom;
        let mut rng_inst = rng();
        match self.config.client_selection.as_str() {
            "random" => {
                let mut client_indices: Vec<usize> = (0..self.clients.len()).collect();
                client_indices.shuffle(&mut rng_inst);
                Ok(client_indices
                    .into_iter()
                    .take(self.config.clients_per_round)
                    .collect())
            }
            "importance" => {
                use scirs2_core::random::Rng;
                use scirs2_core::RngExt;
                let mut weighted_indices: Vec<(usize, f32)> = self
                    .clients
                    .iter()
                    .enumerate()
                    .map(|(i, _)| (i, 0.0_f32))
                    .collect();
                for (_, w) in weighted_indices.iter_mut() {
                    *w = 1.0 + rng_inst.random::<f32>();
                }
                weighted_indices
                    .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                Ok(weighted_indices
                    .into_iter()
                    .take(self.config.clients_per_round)
                    .map(|(i, _)| i)
                    .collect())
            }
            _ => {
                // Default to random
                let mut client_indices: Vec<usize> = (0..self.clients.len()).collect();
                client_indices.shuffle(&mut rng_inst);
                Ok(client_indices
                    .into_iter()
                    .take(self.config.clients_per_round)
                    .collect())
            }
        }
    }

    /// Calculate privacy spent up to current round
    fn calculate_privacy_spent(&self, round: usize) -> Option<f64> {
        self.config
            .privacy_budget
            .map(|budget_per_round| budget_per_round * (round + 1) as f64)
    }

    /// Check if training should stop early
    fn should_stop_early(&self) -> bool {
        if self.communication_rounds.len() < 10 {
            return false;
        }
        let recent_losses: Vec<f32> = self
            .communication_rounds
            .iter()
            .rev()
            .take(5)
            .map(|r| r.avg_loss)
            .collect();
        let loss_variance = self.calculate_variance(&recent_losses);
        loss_variance < 0.0001
    }

    /// Calculate variance of values
    fn calculate_variance(&self, values: &[f32]) -> f32 {
        if values.is_empty() {
            return 0.0;
        }
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / values.len() as f32
    }

    /// Get training history
    pub fn get_history(&self) -> &[RoundStatistics] {
        &self.communication_rounds
    }

    /// Export training metrics
    pub fn export_metrics(&self, path: &str) -> Result<()> {
        use std::fs::File;
        use std::io::Write;
        let mut file = File::create(path)
            .map_err(|e| NeuralError::InferenceError(format!("Failed to create file: {e}")))?;
        writeln!(
            file,
            "round,num_clients,avg_loss,avg_accuracy,communication_cost,duration,privacy_spent"
        )
        .map_err(|e| NeuralError::InferenceError(format!("Write error: {e}")))?;
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
            )
            .map_err(|e| NeuralError::InferenceError(format!("Write error: {e}")))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_federated_config_default() {
        let config = FederatedConfig::default();
        assert_eq!(config.num_rounds, 100);
        assert_eq!(config.clients_per_round, 10);
        assert_eq!(config.aggregation_strategy, "fedavg");
    }

    #[test]
    fn test_client_update_size_bytes() {
        let update = ClientUpdate {
            client_id: 0,
            weight_updates: vec![Array2::zeros((10, 10)), Array2::zeros((5, 5))],
            num_samples: 100,
            loss: 0.5,
            accuracy: 0.9,
        };
        // 10*10 + 5*5 = 125 f32 values = 500 bytes
        assert_eq!(update.size_bytes(), 125 * 4);
    }

    #[test]
    fn test_federated_learning_new() {
        let config = FederatedConfig::default();
        let fl = FederatedLearning::new(config, 20).expect("FederatedLearning::new failed");
        assert_eq!(fl.clients.len(), 20);
        assert_eq!(fl.communication_rounds.len(), 0);
    }

    #[test]
    fn test_select_clients() {
        let config = FederatedConfig::default();
        let fl = FederatedLearning::new(config, 20).expect("FederatedLearning::new failed");
        let selected = fl.select_clients().expect("select_clients failed");
        assert_eq!(selected.len(), 10);
        assert!(selected.iter().all(|&id| id < 20));
    }
}
