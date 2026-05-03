//! Federated learning server implementation

use crate::error::{NeuralError, Result};
use crate::federated::{AggregationStrategy, ClientUpdate};
use crate::models::sequential::Sequential;
use scirs2_core::ndarray::prelude::*;
use std::sync::{Arc, RwLock};

/// Server configuration
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Minimum number of clients required per round
    pub min_clients: usize,
    /// Maximum time to wait for client updates (seconds)
    pub round_timeout: u64,
    /// Aggregation strategy name
    pub aggregation_strategy: String,
    /// Enable adaptive aggregation
    pub adaptive_aggregation: bool,
    /// Model staleness threshold
    pub staleness_threshold: usize,
    /// Enable asynchronous updates
    pub async_updates: bool,
}

impl From<&crate::federated::FederatedConfig> for ServerConfig {
    fn from(config: &crate::federated::FederatedConfig) -> Self {
        Self {
            min_clients: config.min_clients,
            round_timeout: 300,
            aggregation_strategy: config.aggregation_strategy.clone(),
            adaptive_aggregation: false,
            async_updates: false,
            staleness_threshold: 5,
        }
    }
}

/// Federated learning server
pub struct FederatedServer {
    config: ServerConfig,
    /// Global model state
    global_model_state: Arc<RwLock<ModelState>>,
    /// Aggregation strategy
    aggregator: Box<dyn AggregationStrategy>,
    /// Round counter
    pub current_round: usize,
    /// Client contributions tracker
    client_contributions: ClientContributions,
}

/// Model state information
#[derive(Clone)]
struct ModelState {
    /// Model parameters
    parameters: Vec<Array2<f32>>,
    /// Model version
    version: usize,
    /// Last update timestamp
    last_updated: std::time::Instant,
}

/// Track client contributions
struct ClientContributions {
    /// Total samples from each client
    samples_per_client: std::collections::HashMap<usize, usize>,
    /// Rounds participated by each client
    rounds_per_client: std::collections::HashMap<usize, usize>,
    /// Performance history per client
    performance_history: std::collections::HashMap<usize, Vec<f32>>,
}

impl FederatedServer {
    /// Create a new federated server
    pub fn new(config: ServerConfig) -> Result<Self> {
        let aggregator: Box<dyn AggregationStrategy> = match config.aggregation_strategy.as_str() {
            "fedavg" => Box::new(crate::federated::FedAvg::new()),
            "fedprox" => Box::new(crate::federated::FedProx::new(0.01)),
            "fedyogi" => Box::new(crate::federated::FedYogi::new()),
            _ => Box::new(crate::federated::FedAvg::new()),
        };
        Ok(Self {
            config,
            global_model_state: Arc::new(RwLock::new(ModelState {
                parameters: Vec::new(),
                version: 0,
                last_updated: std::time::Instant::now(),
            })),
            aggregator,
            current_round: 0,
            client_contributions: ClientContributions {
                samples_per_client: std::collections::HashMap::new(),
                rounds_per_client: std::collections::HashMap::new(),
                performance_history: std::collections::HashMap::new(),
            },
        })
    }

    /// Get current model parameters
    pub fn get_model_parameters(&self, _model: &Sequential<f32>) -> Result<Vec<Array2<f32>>> {
        let state = self
            .global_model_state
            .read()
            .map_err(|_| NeuralError::InferenceError("RwLock poisoned".to_string()))?;
        if state.parameters.is_empty() {
            Ok(vec![Array2::zeros((10, 10)); 5])
        } else {
            Ok(state.parameters.clone())
        }
    }

    /// Aggregate client updates
    pub fn aggregate_updates(&mut self, updates: &[ClientUpdate]) -> Result<AggregatedUpdate> {
        if updates.len() < self.config.min_clients {
            return Err(NeuralError::InvalidArgument(format!(
                "Not enough clients: {} < {}",
                updates.len(),
                self.config.min_clients
            )));
        }
        // Update client contributions
        for update in updates {
            *self
                .client_contributions
                .samples_per_client
                .entry(update.client_id)
                .or_insert(0) += update.num_samples;
            *self
                .client_contributions
                .rounds_per_client
                .entry(update.client_id)
                .or_insert(0) += 1;
            self.client_contributions
                .performance_history
                .entry(update.client_id)
                .or_default()
                .push(update.accuracy);
        }
        // Calculate weights for aggregation
        let weights = if self.config.adaptive_aggregation {
            self.calculate_adaptive_weights(updates)
        } else {
            self.calculate_sample_weights(updates)
        };
        // Aggregate updates
        let aggregated_weights = self.aggregator.aggregate(updates, &weights)?;
        self.current_round += 1;
        Ok(AggregatedUpdate {
            aggregated_weights,
            round: self.current_round,
            num_clients: updates.len(),
            total_samples: updates.iter().map(|u| u.num_samples).sum(),
        })
    }

    /// Calculate sample-based weights
    pub fn calculate_sample_weights(&self, updates: &[ClientUpdate]) -> Vec<f32> {
        let total_samples: usize = updates.iter().map(|u| u.num_samples).sum();
        if total_samples == 0 {
            return vec![1.0 / updates.len() as f32; updates.len()];
        }
        updates
            .iter()
            .map(|u| u.num_samples as f32 / total_samples as f32)
            .collect()
    }

    /// Calculate adaptive weights based on client performance
    fn calculate_adaptive_weights(&self, updates: &[ClientUpdate]) -> Vec<f32> {
        let mut weights: Vec<f32> = updates
            .iter()
            .map(|update| {
                let sample_weight = (update.num_samples as f32).max(1.0);
                let performance_factor = self
                    .client_contributions
                    .performance_history
                    .get(&update.client_id)
                    .map(|history| {
                        let recent: Vec<f32> = history.iter().rev().take(5).copied().collect();
                        if recent.is_empty() {
                            1.0
                        } else {
                            recent.iter().sum::<f32>() / recent.len() as f32
                        }
                    })
                    .unwrap_or(1.0);
                let rounds = self
                    .client_contributions
                    .rounds_per_client
                    .get(&update.client_id)
                    .copied()
                    .unwrap_or(1) as f32;
                let participation_factor = (rounds / self.current_round.max(1) as f32).sqrt();
                sample_weight * performance_factor * participation_factor
            })
            .collect();

        let sum: f32 = weights.iter().sum();
        if sum > 0.0 {
            for w in &mut weights {
                *w /= sum;
            }
        } else {
            let equal_weight = 1.0 / weights.len().max(1) as f32;
            weights.fill(equal_weight);
        }
        weights
    }

    /// Update global model with aggregated updates
    pub fn update_global_model(
        &mut self,
        _model: &mut Sequential<f32>,
        update: &AggregatedUpdate,
    ) -> Result<()> {
        let mut state = self
            .global_model_state
            .write()
            .map_err(|_| NeuralError::InferenceError("RwLock poisoned".to_string()))?;
        state.parameters = update.aggregated_weights.clone();
        state.version += 1;
        state.last_updated = std::time::Instant::now();
        Ok(())
    }

    /// Get server statistics
    pub fn get_statistics(&self) -> ServerStatistics {
        let total_clients = self.client_contributions.samples_per_client.len();
        let active_clients = self
            .client_contributions
            .rounds_per_client
            .values()
            .filter(|&&rounds| rounds > self.current_round.saturating_sub(5))
            .count();
        let total_samples: usize = self.client_contributions.samples_per_client.values().sum();
        let avg_rounds_per_client = if total_clients > 0 {
            self.client_contributions
                .rounds_per_client
                .values()
                .sum::<usize>() as f32
                / total_clients as f32
        } else {
            0.0
        };
        let model_version = self
            .global_model_state
            .read()
            .map(|s| s.version)
            .unwrap_or(0);
        ServerStatistics {
            current_round: self.current_round,
            total_clients,
            active_clients,
            total_samples_processed: total_samples,
            average_rounds_per_client: avg_rounds_per_client,
            model_version,
        }
    }

    /// Reset server state
    pub fn reset(&mut self) {
        self.current_round = 0;
        self.client_contributions = ClientContributions {
            samples_per_client: std::collections::HashMap::new(),
            rounds_per_client: std::collections::HashMap::new(),
            performance_history: std::collections::HashMap::new(),
        };
        if let Ok(mut state) = self.global_model_state.write() {
            state.parameters.clear();
            state.version = 0;
        }
    }
}

/// Aggregated update result
pub struct AggregatedUpdate {
    pub aggregated_weights: Vec<Array2<f32>>,
    pub round: usize,
    pub num_clients: usize,
    pub total_samples: usize,
}

/// Server statistics
pub struct ServerStatistics {
    pub current_round: usize,
    pub total_clients: usize,
    pub active_clients: usize,
    pub total_samples_processed: usize,
    pub average_rounds_per_client: f32,
    pub model_version: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::federated::FederatedConfig;

    #[test]
    fn test_server_creation() {
        let config = FederatedConfig::default();
        let server_config = ServerConfig::from(&config);
        let server = FederatedServer::new(server_config).expect("FederatedServer::new failed");
        assert_eq!(server.current_round, 0);
    }

    #[test]
    fn test_sample_weights() {
        let config = ServerConfig::from(&FederatedConfig::default());
        let server = FederatedServer::new(config).expect("FederatedServer::new failed");
        let updates = vec![
            ClientUpdate {
                client_id: 0,
                weight_updates: vec![],
                num_samples: 100,
                loss: 0.5,
                accuracy: 0.9,
            },
            ClientUpdate {
                client_id: 1,
                weight_updates: vec![],
                num_samples: 200,
                loss: 0.4,
                accuracy: 0.92,
            },
        ];
        let weights = server.calculate_sample_weights(&updates);
        assert_eq!(weights.len(), 2);
        assert!((weights[0] - 1.0 / 3.0).abs() < 0.001);
        assert!((weights[1] - 2.0 / 3.0).abs() < 0.001);
    }
}
