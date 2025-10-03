//! FedNova: Tackling the Objective Inconsistency Problem in Heterogeneous Federated Learning
//!
//! FedNova normalizes and scales local updates based on the number of local steps
//! performed by each client, addressing the objective inconsistency problem that
//! arises when clients perform different numbers of local updates.

use crate::error::{NeuralError, Result};
use crate::federated::{AggregationStrategy, ClientUpdate};
use scirs2_core::ndarray::prelude::*;
use scirs2_core::ndarray::ArrayView1;
/// FedNova aggregation strategy
pub struct FedNova {
    /// Momentum parameter for server optimizer
    momentum: f32,
    /// Server learning rate
    server_lr: f32,
    /// Accumulated momentum for each parameter
    momentum_buffers: Option<Vec<Array2<f32>>>,
    /// Use momentum SGD on server
    use_momentum: bool,
}
impl FedNova {
    /// Create a new FedNova aggregator
    pub fn new(_server_lr: f32, momentum: f32, usemomentum: bool) -> Self {
        Self {
            momentum,
            server_lr,
            _momentum_buffers: None,
            use_momentum,
        }
    }
    /// Normalize updates based on number of local steps
    fn normalize_updates(
        &self,
        updates: &[ClientUpdate],
        local_steps: &[usize],
    ) -> Result<Vec<Vec<Array2<f32>>>> {
        let mut normalized_updates = Vec::new();
        for (update, &steps) in updates.iter().zip(local_steps.iter()) {
            let mut normalized_client_updates = Vec::new();
            for weight_update in &update.weight_updates {
                // Normalize by number of local steps
                let normalized = weight_update / steps as f32;
                normalized_client_updates.push(normalized);
            }
            normalized_updates.push(normalized_client_updates);
        Ok(normalized_updates)
    /// Compute effective number of steps for each client
    fn compute_effective_steps(&self, updates: &[ClientUpdate], taueff: f32) -> Vec<f32> {
        updates
            .iter()
            .map(|update| {
                // In practice, this would be based on actual local steps
                // For now, use a simple heuristic based on num_samples
                (update.num_samples as f32).min(tau_eff)
            })
            .collect()
impl AggregationStrategy for FedNova {
    fn aggregate(&mut self, updates: &[ClientUpdate], weights: &[f32]) -> Result<Vec<Array2<f32>>> {
        if updates.is_empty() {
            return Err(NeuralError::InvalidArgument(
                "No updates to aggregate".to_string(),
            ));
        let num_params = updates[0].weight_updates.len();
        // Get local steps for each client (simplified - would be tracked in practice)
        let local_steps: Vec<usize> = updates.iter()
            .map(|u| (u.num_samples / 32).max(1)) // Assuming batch size of 32
            .collect();
        // Normalize updates
        let normalized_updates = self.normalize_updates(updates, &local_steps)?;
        // Compute effective global steps
        let tau_eff = local_steps.iter().sum::<usize>() as f32 / updates.len() as f32;
        let effective_steps = self.compute_effective_steps(updates, tau_eff);
        // Initialize aggregated updates
        let mut aggregated_updates = vec![Array2::zeros((1, 1)); num_params];
        // Compute weighted average based on effective steps and data sizes
        let total_effective_data: f32 = updates
            .zip(&effective_steps)
            .map(|(u, &eff_steps)| u.num_samples as f32 * eff_steps)
            .sum();
        for param_idx in 0..num_params {
            // Get shape from first update
            let shape = normalized_updates[0][param_idx].shape();
            let mut aggregated = Array2::zeros((shape[0], shape[1]));
            for (client_idx, (update, &eff_steps)) in
                updates.iter().zip(&effective_steps).enumerate()
            {
                let weight = (update.num_samples as f32 * eff_steps) / total_effective_data;
                aggregated += &(&normalized_updates[client_idx][param_idx] * weight * tau_eff);
            aggregated_updates[param_idx] = aggregated;
        // Apply server momentum if enabled
        if self.use_momentum {
            if self.momentum_buffers.is_none() {
                // Initialize momentum buffers
                self.momentum_buffers = Some(
                    aggregated_updates
                        .iter()
                        .map(|u| Array2::zeros(u.shape()))
                        .collect(),
                );
            if let Some(ref mut buffers) = self.momentum_buffers {
                for (i, (update, buffer)) in aggregated_updates
                    .iter_mut()
                    .zip(buffers.iter_mut())
                    .enumerate()
                {
                    // Momentum SGD: v = momentum * v + lr * g
                    *buffer = &*buffer * self.momentum + &*update * self.server_lr;
                    *update = buffer.clone();
                }
        } else {
            // Just scale by server learning rate
            for update in &mut aggregated_updates {
                *update *= self.server_lr;
        Ok(aggregated_updates)
    fn name(&self) -> &str {
        "FedNova"
/// FedNova client with local step tracking
pub struct FedNovaClient {
    client_id: usize,
    local_steps: usize,
    batch_size: usize,
    local_lr: f32,
    /// Track gradient accumulation for proper normalization
    grad_accumulator: Option<Vec<Array2<f32>>>,
impl FedNovaClient {
    /// Create a new FedNova client
    pub fn new(_client_id: usize, batch_size: usize, locallr: f32) -> Self {
            client_id,
            local_steps: 0,
            batch_size,
            local_lr,
            grad_accumulator: None,
    /// Perform local training with proper gradient tracking
    pub fn local_train(
        &mut self,
        global_weights: &[Array2<f32>],
        data: &ArrayView2<f32>,
        labels: &ArrayView1<usize>,
        epochs: usize,
    ) -> Result<FedNovaUpdate> {
        let num_samples = data.shape()[0];
        let steps_per_epoch = (num_samples + self.batch_size - 1) / self.batch_size;
        self.local_steps = epochs * steps_per_epoch;
        // Initialize gradient accumulator
        if self.grad_accumulator.is_none() {
            self.grad_accumulator = Some(
                global_weights
                    .iter()
                    .map(|w| Array2::zeros(w.shape()))
                    .collect(),
            );
        // Reset accumulator
        if let Some(ref mut accumulator) = self.grad_accumulator {
            for acc in accumulator.iter_mut() {
                acc.fill(0.0);
        // Simulate training and gradient accumulation
        let mut total_loss = 0.0;
        let mut total_correct = 0;
        for epoch in 0..epochs {
            let (epoch_loss, epoch_correct) = self.train_epoch(global_weights, data, labels)?;
            total_loss += epoch_loss;
            total_correct += epoch_correct;
        // Compute normalized update (gradient * local_steps)
        let weight_updates = if let Some(ref accumulator) = self.grad_accumulator {
            accumulator.clone()
            vec![]
        };
        Ok(FedNovaUpdate {
            client_id: self.client_id,
            weight_updates,
            num_samples,
            local_steps: self.local_steps,
            loss: total_loss / epochs as f32,
            accuracy: total_correct as f32 / (num_samples * epochs) as f32,
        })
    /// Train for one epoch
    fn train_epoch(
        weights: &[Array2<f32>],
    ) -> Result<(f32, usize)> {
        let num_batches = (num_samples + self.batch_size - 1) / self.batch_size;
        let mut epoch_loss = 0.0;
        let mut correct = 0;
        for batch_idx in 0..num_batches {
            let start = batch_idx * self.batch_size;
            let end = ((batch_idx + 1) * self.batch_size).min(num_samples);
            let batch_size = end - start;
            // Simulate gradient computation
            if let Some(ref mut accumulator) = self.grad_accumulator {
                for (acc, weight) in accumulator.iter_mut().zip(weights.iter()) {
                    // Simulate gradient (simplified)
                    let grad = Array2::from_elem(weight.shape(), 0.01 / batch_size as f32);
                    *acc += &grad;
            // Simulate loss and accuracy
            epoch_loss += 0.5; // Placeholder
            correct += batch_size / 2; // Placeholder
        Ok((epoch_loss / num_batches as f32, correct))
/// FedNova-specific update structure
#[derive(Debug, Clone)]
pub struct FedNovaUpdate {
    pub client_id: usize,
    pub weight_updates: Vec<Array2<f32>>,
    pub num_samples: usize,
    pub local_steps: usize,
    pub loss: f32,
    pub accuracy: f32,
impl From<FedNovaUpdate> for ClientUpdate {
    fn from(update: FedNovaUpdate) -> Self {
        ClientUpdate {
            client_id: update.client_id,
            weight_updates: update.weight_updates,
            num_samples: update.num_samples,
            loss: update.loss,
            accuracy: update.accuracy,
/// FedNova coordinator with adaptive tau computation
pub struct FedNovaCoordinator {
    aggregator: FedNova,
    /// History of tau_eff values
    tau_history: Vec<f32>,
    /// Adaptive tau adjustment
    adaptive_tau: bool,
    /// Target tau_eff
    target_tau: f32,
impl FedNovaCoordinator {
    /// Create a new FedNova coordinator
    pub fn new(
        server_lr: f32,
        momentum: f32,
        use_momentum: bool,
        adaptive_tau: bool,
        target_tau: f32,
    ) -> Self {
            aggregator: FedNova::new(server_lr, momentum, use_momentum),
            tau_history: Vec::new(),
            adaptive_tau,
            target_tau,
    /// Coordinate a round of FedNova training
    pub fn coordinate_round(
        client_updates: Vec<FedNovaUpdate>,
    ) -> Result<Vec<Array2<f32>>> {
        // Compute current tau_eff
        let tau_eff = client_updates
            .map(|u| u.local_steps as f32)
            .sum::<f32>()
            / client_updates.len() as f32;
        self.tau_history.push(tau_eff);
        // Adjust client sampling if adaptive tau is enabled
        if self.adaptive_tau && self.tau_history.len() > 5 {
            // Analyze tau trend and adjust client selection strategy
            let recent_tau_avg = self.tau_history.iter().rev().take(5).sum::<f32>() / 5.0;
            if (recent_tau_avg - self.target_tau).abs() > 0.1 * self.target_tau {
                // Would adjust client selection probability here
        // Convert to standard ClientUpdate and aggregate
        let standard_updates: Vec<ClientUpdate> =
            client_updates.into_iter().map(|u| u.into()).collect();
        self.aggregator.aggregate(&standard_updates)
    /// Get tau statistics
    pub fn get_tau_stats(&self) -> TauStatistics {
        if self.tau_history.is_empty() {
            return TauStatistics::default();
        let mean = self.tau_history.iter().sum::<f32>() / self.tau_history.len() as f32;
        let variance = self
            .tau_history
            .map(|&tau| (tau - mean).powi(2))
            / self.tau_history.len() as f32;
        TauStatistics {
            current_tau: *self.tau_history.last().unwrap(),
            mean_tau: mean,
            std_tau: variance.sqrt(),
            min_tau: self
                .tau_history
                .iter()
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .cloned()
                .unwrap_or(0.0),
            max_tau: self
                .max_by(|a, b| a.partial_cmp(b).unwrap())
/// Statistics about tau_eff values
#[derive(Debug, Default)]
pub struct TauStatistics {
    pub current_tau: f32,
    pub mean_tau: f32,
    pub std_tau: f32,
    pub min_tau: f32,
    pub max_tau: f32,
#[cfg(test)]
mod tests {
    use super::*;
    fn create_test_update(_client_id: usize, numsamples: usize) -> FedNovaUpdate {
        let weight_updates = vec![
            Array2::from_elem((10, 10), 0.1),
            Array2::from_elem((10, 5), 0.2),
        ];
        FedNovaUpdate {
            local_steps: num_samples / 32,
            loss: 0.5,
            accuracy: 0.9,
    #[test]
    fn test_fednova_aggregation() {
        let mut aggregator = FedNova::new(0.1, 0.9, false);
        let updates = vec![
            create_test_update(0, 1000).into(),
            create_test_update(1, 500).into(),
            create_test_update(2, 2000).into(),
        let result = aggregator.aggregate(&updates).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].shape(), &[10, 10]);
        assert_eq!(result[1].shape(), &[10, 5]);
    fn test_fednova_client() {
        let mut client = FedNovaClient::new(0, 32, 0.01);
        let global_weights = vec![
            Array2::from_elem((10, 10), 0.5),
            Array2::from_elem((10, 5), 0.5),
        let data = Array2::from_elem((100, 10), 1.0);
        let labels = Array1::from_elem(100, 0);
        let update = client
            .local_train(&global_weights, &data.view(), &labels.view(), 5)
            .unwrap();
        assert_eq!(update.client_id, 0);
        assert_eq!(update.num_samples, 100);
        assert!(update.local_steps > 0);
    fn test_fednova_coordinator() {
        let mut coordinator = FedNovaCoordinator::new(0.1, 0.9, true, true, 10.0);
            create_test_update(0, 1000),
            create_test_update(1, 500),
            create_test_update(2, 2000),
        let result = coordinator.coordinate_round(updates).unwrap();
        assert!(!result.is_empty());
        let stats = coordinator.get_tau_stats();
        assert!(stats.current_tau > 0.0);
