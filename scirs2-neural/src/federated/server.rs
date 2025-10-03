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
            round_timeout: 300, // 5 minutes default
            aggregation_strategy: config.aggregation_strategy.clone(),
            adaptive_aggregation: false,
            async_updates: false,
            staleness_threshold: 5,
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
    current_round: usize,
    /// Client contributions tracker
    client_contributions: ClientContributions,
/// Model state information
#[derive(Clone)]
struct ModelState {
    /// Model parameters
    parameters: Vec<Array2<f32>>,
    /// Model version
    version: usize,
    /// Last update timestamp
    last_updated: std::time::Instant,
/// Track client contributions
struct ClientContributions {
    /// Total samples from each client
    samples_per_client: std::collections::HashMap<usize, usize>,
    /// Rounds participated by each client
    rounds_per_client: std::collections::HashMap<usize, usize>,
    /// Performance history per client
    performance_history: std::collections::HashMap<usize, Vec<f32>>,
impl FederatedServer {
    /// Create a new federated server
    pub fn new(config: ServerConfig) -> Result<Self> {
        let aggregator: Box<dyn AggregationStrategy> = match config.aggregation_strategy.as_str() {
            "fedavg" => Box::new(crate::federated::FedAvg::new()),
            "fedprox" => Box::new(crate::federated::FedProx::new(0.01)),
            "fedyogi" => Box::new(crate::federated::FedYogi::new(), _ => Box::new(crate::federated::FedAvg::new()),
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
    /// Get current model parameters
    pub fn get_model_parameters(&self, model: &Sequential<f32>) -> Result<Vec<Array2<f32>>> {
        // Simplified implementation - extract model parameters
        // In practice, would extract actual weights from the model
        let state = self.global_model_state.read().unwrap();
        if state.parameters.is_empty() {
            // Initialize with dummy parameters
            Ok(vec![Array2::zeros((10, 10)); 5])
        } else {
            Ok(state.parameters.clone())
    /// Aggregate client updates
    pub fn aggregate_updates(&mut self, updates: &[ClientUpdate]) -> Result<AggregatedUpdate> {
        if updates.len() < self.config.min_clients {
            return Err(NeuralError::InvalidArgument(format!(
                "Not enough clients: {} < {}",
                updates.len(),
                self.config.min_clients
            )));
        // Update client contributions
        for update in updates {
            *self
                .client_contributions
                .samples_per_client
                .entry(update.client_id)
                .or_insert(0) += update.num_samples;
                .rounds_per_client
                .or_insert(0) += 1;
            self.client_contributions
                .performance_history
                .or_insert_with(Vec::new)
                .push(update.accuracy);
        // Calculate weights for aggregation
        let weights = if self.config.adaptive_aggregation {
            self.calculate_adaptive_weights(updates)?
            self.calculate_sample_weights(updates)
        // Aggregate updates using the selected strategy
        let aggregated_weights = self.aggregator.aggregate(updates, &weights)?;
        // Update round counter
        self.current_round += 1;
        Ok(AggregatedUpdate {
            aggregated_weights,
            round: self.current_round,
            num_clients: updates.len(),
            total_samples: updates.iter().map(|u| u.num_samples).sum(),
    /// Calculate sample-based weights
    fn calculate_sample_weights(&self, updates: &[ClientUpdate]) -> Vec<f32> {
        let total_samples: usize = updates.iter().map(|u| u.num_samples).sum();
        updates
            .iter()
            .map(|u| u.num_samples as f32 / total_samples as f32)
            .collect()
    /// Calculate adaptive weights based on client performance
    fn calculate_adaptive_weights(&self, updates: &[ClientUpdate]) -> Result<Vec<f32>> {
        let mut weights = Vec::with_capacity(updates.len());
            // Base weight from sample size
            let sample_weight = update.num_samples as f32;
            // Performance factor
            let performance_factor = if let Some(history) = self
                .get(&update.client_id)
            {
                // Use recent performance trend
                let recent_perf = history.iter().rev().take(5).copied().collect::<Vec<_>>();
                self.calculate_performance_score(&recent_perf)
            } else {
                1.0
            };
            // Participation factor
            let rounds = self
                .copied()
                .unwrap_or(1) as f32;
            let participation_factor = (rounds / self.current_round.max(1) as f32).sqrt();
            weights.push(sample_weight * performance_factor * participation_factor);
        // Normalize weights
        let sum: f32 = weights.iter().sum();
        if sum > 0.0 {
            for w in &mut weights {
                *w /= sum;
            }
            // Equal weights if sum is 0
            let equal_weight = 1.0 / weights.len() as f32;
            weights.fill(equal_weight);
        Ok(weights)
    /// Calculate performance score from history
    fn calculate_performance_score(&self, history: &[f32]) -> f32 {
        if history.is_empty() {
            return 1.0;
        // Consider both average performance and improvement trend
        let avg_performance = history.iter().sum::<f32>() / history.len() as f32;
        let trend = if history.len() >= 2 {
            let recent = history[history.len() - 1];
            let previous = history[0];
            (recent - previous).max(0.0)
            0.0
        // Combine average and trend
        (avg_performance + 0.1 * trend).min(1.0)
    /// Update global model with aggregated updates
    pub fn update_global_model(
        &mut self,
        model: &mut Sequential<f32>,
        update: &AggregatedUpdate,
    ) -> Result<()> {
        let mut state = self.global_model_state.write().unwrap();
        // Update model parameters
        state.parameters = update.aggregatedweights.clone();
        state.version += 1;
        state.last_updated = std::time::Instant::now();
        // In practice, would update the actual model weights
        // For now, we just store the aggregated weights
        Ok(())
    /// Handle asynchronous client update
    pub fn handle_async_update(&mut self, update: ClientUpdate) -> Result<Option<ModelState>> {
        if !self.config.async_updates {
            return Err(NeuralError::InvalidArgument(
                "Asynchronous updates not enabled".to_string(),
            ));
        // Check staleness
        let model_age = state.version;
        drop(state);
        if model_age > self.config.staleness_threshold {
            // Update is too stale, reject it
            return Ok(None);
        // Apply update with staleness-aware weighting
        let staleness_weight = 1.0 / (1.0 + (model_age as f32));
        // Simple asynchronous update (in practice, would be more sophisticated)
        // Apply weighted update to current parameters
        for (i, param_update) in update.weight_updates.iter().enumerate() {
            if i < state.parameters.len() {
                state.parameters[i] = &state.parameters[i] + staleness_weight * param_update;
        Ok(Some(state.clone()))
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
                .values()
                .sum::<usize>() as f32
                / total_clients as f32
        ServerStatistics {
            current_round: self.current_round,
            total_clients,
            active_clients,
            total_samples_processed: total_samples,
            average_rounds_per_client: avg_rounds_per_client,
            model_version: self.global_model_state.read().unwrap().version,
    /// Reset server state
    pub fn reset(&mut self) {
        self.current_round = 0;
        self.client_contributions = ClientContributions {
            samples_per_client: std::collections::HashMap::new(),
            rounds_per_client: std::collections::HashMap::new(),
            performance_history: std::collections::HashMap::new(),
        state.parameters.clear();
        state.version = 0;
/// Aggregated update result
pub struct AggregatedUpdate {
    pub aggregated_weights: Vec<Array2<f32>>,
    pub round: usize,
    pub num_clients: usize,
    pub total_samples: usize,
/// Server statistics
pub struct ServerStatistics {
    pub current_round: usize,
    pub total_clients: usize,
    pub active_clients: usize,
    pub total_samples_processed: usize,
    pub average_rounds_per_client: f32,
    pub model_version: usize,
#[cfg(test)]
mod tests {
    use super::*;
    use crate::federated::FederatedConfig;
    #[test]
    fn test_server_creation() {
        let config = FederatedConfig::default();
        let server_config = ServerConfig::from(&config);
        let server = FederatedServer::new(server_config).unwrap();
        assert_eq!(server.current_round, 0);
    fn test_sample_weights() {
        let config = ServerConfig::from(&FederatedConfig::default());
        let server = FederatedServer::new(config).unwrap();
        let updates = vec![
            ClientUpdate {
                client_id: 0,
                weight_updates: vec![],
                num_samples: 100,
                loss: 0.5,
                accuracy: 0.9,
                client_id: 1,
                num_samples: 200,
                loss: 0.4,
                accuracy: 0.92,
        ];
        let weights = server.calculate_sample_weights(&updates);
        assert_eq!(weights.len(), 2);
        assert!((weights[0] - 1.0 / 3.0).abs() < 0.001);
        assert!((weights[1] - 2.0 / 3.0).abs() < 0.001);
