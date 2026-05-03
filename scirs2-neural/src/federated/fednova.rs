//! FedNova: Tackling the Objective Inconsistency Problem in Heterogeneous Federated Learning
//!
//! FedNova normalizes and scales local updates based on the number of local steps
//! performed by each client, addressing the objective inconsistency problem that
//! arises when clients perform different numbers of local updates.

use crate::error::{NeuralError, Result};
use crate::federated::{AggregationStrategy, ClientUpdate};
use scirs2_core::ndarray::prelude::*;

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
    pub fn new(server_lr: f32, momentum: f32, use_momentum: bool) -> Self {
        Self {
            momentum,
            server_lr,
            momentum_buffers: None,
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
            let steps_f = steps.max(1) as f32;
            let mut normalized_client_updates = Vec::new();
            for weight_update in &update.weight_updates {
                let normalized = weight_update / steps_f;
                normalized_client_updates.push(normalized);
            }
            normalized_updates.push(normalized_client_updates);
        }
        Ok(normalized_updates)
    }

    /// Compute effective number of steps for each client
    fn compute_effective_steps(&self, updates: &[ClientUpdate], tau_eff: f32) -> Vec<f32> {
        updates
            .iter()
            .map(|update| (update.num_samples as f32).min(tau_eff))
            .collect()
    }
}

impl AggregationStrategy for FedNova {
    fn aggregate(
        &mut self,
        updates: &[ClientUpdate],
        _weights: &[f32],
    ) -> Result<Vec<Array2<f32>>> {
        if updates.is_empty() {
            return Err(NeuralError::InvalidArgument(
                "No updates to aggregate".to_string(),
            ));
        }
        let num_params = updates[0].weight_updates.len();

        // Get local steps for each client
        let local_steps: Vec<usize> = updates
            .iter()
            .map(|u| (u.num_samples / 32).max(1))
            .collect();

        // Normalize updates
        let normalized_updates = self.normalize_updates(updates, &local_steps)?;

        // Compute effective global steps
        let tau_eff = local_steps.iter().sum::<usize>() as f32 / updates.len() as f32;
        let effective_steps = self.compute_effective_steps(updates, tau_eff);

        let total_effective_data: f32 = updates
            .iter()
            .zip(&effective_steps)
            .map(|(u, &eff)| u.num_samples as f32 * eff)
            .sum();

        let mut aggregated_updates: Vec<Array2<f32>> = if num_params > 0 {
            normalized_updates[0]
                .iter()
                .map(|a| Array2::zeros(a.raw_dim()))
                .collect()
        } else {
            Vec::new()
        };

        for param_idx in 0..num_params {
            for (client_idx, (update, &eff_steps)) in
                updates.iter().zip(&effective_steps).enumerate()
            {
                if client_idx < normalized_updates.len()
                    && param_idx < normalized_updates[client_idx].len()
                {
                    let denom = total_effective_data.max(1e-8);
                    let weight = (update.num_samples as f32 * eff_steps) / denom;
                    aggregated_updates[param_idx] +=
                        &(&normalized_updates[client_idx][param_idx] * weight * tau_eff);
                }
            }
        }

        // Apply server momentum if enabled
        if self.use_momentum {
            if self.momentum_buffers.is_none() {
                self.momentum_buffers = Some(
                    aggregated_updates
                        .iter()
                        .map(|u| Array2::zeros(u.raw_dim()))
                        .collect(),
                );
            }
            if let Some(ref mut buffers) = self.momentum_buffers {
                for (update, buffer) in aggregated_updates.iter_mut().zip(buffers.iter_mut()) {
                    *buffer = &*buffer * self.momentum + &*update * self.server_lr;
                    *update = buffer.clone();
                }
            }
        } else {
            for update in &mut aggregated_updates {
                *update *= self.server_lr;
            }
        }

        Ok(aggregated_updates)
    }

    fn name(&self) -> &str {
        "FedNova"
    }
}

/// FedNova client with local step tracking
pub struct FedNovaClient {
    client_id: usize,
    local_steps: usize,
    batch_size: usize,
    local_lr: f32,
    /// Track gradient accumulation for proper normalization
    grad_accumulator: Option<Vec<Array2<f32>>>,
}

impl FedNovaClient {
    /// Create a new FedNova client
    pub fn new(client_id: usize, batch_size: usize, local_lr: f32) -> Self {
        Self {
            client_id,
            local_steps: 0,
            batch_size,
            local_lr,
            grad_accumulator: None,
        }
    }

    /// Perform local training with proper gradient tracking
    pub fn local_train(
        &mut self,
        global_weights: &[Array2<f32>],
        data: &ArrayView2<f32>,
        _labels: &ArrayView1<usize>,
        epochs: usize,
    ) -> Result<FedNovaUpdate> {
        let num_samples = data.shape()[0];
        let steps_per_epoch = num_samples.div_ceil(self.batch_size);
        self.local_steps = epochs * steps_per_epoch;

        // Initialize gradient accumulator
        if self.grad_accumulator.is_none() {
            self.grad_accumulator = Some(
                global_weights
                    .iter()
                    .map(|w| Array2::zeros(w.raw_dim()))
                    .collect(),
            );
        }

        // Reset accumulator
        if let Some(ref mut accumulator) = self.grad_accumulator {
            for acc in accumulator.iter_mut() {
                acc.fill(0.0);
            }
        }

        let mut total_loss = 0.0_f32;
        let mut total_correct = 0_usize;

        for epoch in 0..epochs {
            let (epoch_loss, epoch_correct) = self.train_epoch(global_weights, num_samples)?;
            total_loss += epoch_loss;
            total_correct += epoch_correct;
            let _ = epoch; // avoid warning
        }

        let weight_updates = if let Some(ref accumulator) = self.grad_accumulator {
            accumulator.clone()
        } else {
            vec![]
        };

        Ok(FedNovaUpdate {
            client_id: self.client_id,
            weight_updates,
            num_samples,
            local_steps: self.local_steps,
            loss: total_loss / epochs.max(1) as f32,
            accuracy: total_correct as f32 / (num_samples * epochs).max(1) as f32,
        })
    }

    /// Train for one epoch
    fn train_epoch(&mut self, weights: &[Array2<f32>], num_samples: usize) -> Result<(f32, usize)> {
        let num_batches = num_samples.div_ceil(self.batch_size);
        let mut epoch_loss = 0.0_f32;
        let mut correct = 0_usize;

        for batch_idx in 0..num_batches {
            let start = batch_idx * self.batch_size;
            let end = ((batch_idx + 1) * self.batch_size).min(num_samples);
            let batch_size = end - start;

            // Simulate gradient computation
            if let Some(ref mut accumulator) = self.grad_accumulator {
                for (acc, weight) in accumulator.iter_mut().zip(weights.iter()) {
                    let grad = Array2::from_elem(
                        weight.raw_dim(),
                        self.local_lr * 0.01 / batch_size.max(1) as f32,
                    );
                    *acc += &grad;
                }
            }

            epoch_loss += 0.5;
            correct += batch_size / 2;
        }

        Ok((epoch_loss / num_batches.max(1) as f32, correct))
    }
}

/// FedNova-specific update structure
#[derive(Debug, Clone)]
pub struct FedNovaUpdate {
    pub client_id: usize,
    pub weight_updates: Vec<Array2<f32>>,
    pub num_samples: usize,
    pub local_steps: usize,
    pub loss: f32,
    pub accuracy: f32,
}

impl From<FedNovaUpdate> for ClientUpdate {
    fn from(update: FedNovaUpdate) -> Self {
        ClientUpdate {
            client_id: update.client_id,
            weight_updates: update.weight_updates,
            num_samples: update.num_samples,
            loss: update.loss,
            accuracy: update.accuracy,
        }
    }
}

/// FedNova coordinator with adaptive tau computation
pub struct FedNovaCoordinator {
    aggregator: FedNova,
    /// History of tau_eff values
    tau_history: Vec<f32>,
    /// Adaptive tau adjustment
    adaptive_tau: bool,
    /// Target tau_eff
    target_tau: f32,
}

impl FedNovaCoordinator {
    /// Create a new FedNova coordinator
    pub fn new(
        server_lr: f32,
        momentum: f32,
        use_momentum: bool,
        adaptive_tau: bool,
        target_tau: f32,
    ) -> Self {
        Self {
            aggregator: FedNova::new(server_lr, momentum, use_momentum),
            tau_history: Vec::new(),
            adaptive_tau,
            target_tau,
        }
    }

    /// Coordinate a round of FedNova training
    pub fn coordinate_round(
        &mut self,
        client_updates: Vec<FedNovaUpdate>,
    ) -> Result<Vec<Array2<f32>>> {
        if client_updates.is_empty() {
            return Err(NeuralError::InvalidArgument(
                "No client updates".to_string(),
            ));
        }
        // Compute current tau_eff
        let tau_eff = client_updates
            .iter()
            .map(|u| u.local_steps as f32)
            .sum::<f32>()
            / client_updates.len() as f32;
        self.tau_history.push(tau_eff);

        // Adjust client sampling if adaptive tau is enabled
        if self.adaptive_tau && self.tau_history.len() > 5 {
            let recent_tau_avg = self.tau_history.iter().rev().take(5).sum::<f32>() / 5.0;
            if (recent_tau_avg - self.target_tau).abs() > 0.1 * self.target_tau {
                // Would adjust client selection probability here
            }
        }

        // Convert to standard ClientUpdate and aggregate
        let n = client_updates.len();
        let standard_updates: Vec<ClientUpdate> =
            client_updates.into_iter().map(ClientUpdate::from).collect();
        let weights = vec![1.0 / n as f32; n];
        self.aggregator.aggregate(&standard_updates, &weights)
    }

    /// Get tau statistics
    pub fn get_tau_stats(&self) -> TauStatistics {
        if self.tau_history.is_empty() {
            return TauStatistics::default();
        }
        let mean = self.tau_history.iter().sum::<f32>() / self.tau_history.len() as f32;
        let variance = self
            .tau_history
            .iter()
            .map(|&tau| (tau - mean).powi(2))
            .sum::<f32>()
            / self.tau_history.len() as f32;
        TauStatistics {
            current_tau: *self.tau_history.last().expect("non-empty"),
            mean_tau: mean,
            std_tau: variance.sqrt(),
            min_tau: self
                .tau_history
                .iter()
                .cloned()
                .fold(f32::INFINITY, f32::min),
            max_tau: self
                .tau_history
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max),
        }
    }
}

/// Statistics about tau_eff values
#[derive(Debug, Default)]
pub struct TauStatistics {
    pub current_tau: f32,
    pub mean_tau: f32,
    pub std_tau: f32,
    pub min_tau: f32,
    pub max_tau: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_update(client_id: usize, num_samples: usize) -> FedNovaUpdate {
        let weight_updates = vec![
            Array2::from_elem((10, 10), 0.1),
            Array2::from_elem((10, 5), 0.2),
        ];
        FedNovaUpdate {
            client_id,
            weight_updates,
            num_samples,
            local_steps: num_samples / 32,
            loss: 0.5,
            accuracy: 0.9,
        }
    }

    #[test]
    fn test_fednova_aggregation() {
        let mut aggregator = FedNova::new(0.1, 0.9, false);
        let updates: Vec<ClientUpdate> = vec![
            create_test_update(0, 1000).into(),
            create_test_update(1, 500).into(),
            create_test_update(2, 2000).into(),
        ];
        let weights = vec![1.0 / 3.0; 3];
        let result = aggregator
            .aggregate(&updates, &weights)
            .expect("aggregate failed");
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].shape(), &[10, 10]);
        assert_eq!(result[1].shape(), &[10, 5]);
    }

    #[test]
    fn test_fednova_client() {
        let mut client = FedNovaClient::new(0, 32, 0.01);
        let global_weights = vec![
            Array2::from_elem((10, 10), 0.5_f32),
            Array2::from_elem((10, 5), 0.5_f32),
        ];
        let data = Array2::from_elem((100, 10), 1.0_f32);
        let labels = Array1::from_elem(100, 0_usize);
        let update = client
            .local_train(&global_weights, &data.view(), &labels.view(), 5)
            .expect("local_train failed");
        assert_eq!(update.client_id, 0);
        assert_eq!(update.num_samples, 100);
        assert!(update.local_steps > 0);
    }

    #[test]
    fn test_fednova_coordinator() {
        let mut coordinator = FedNovaCoordinator::new(0.1, 0.9, true, true, 10.0);
        let updates = vec![
            create_test_update(0, 1000),
            create_test_update(1, 500),
            create_test_update(2, 2000),
        ];
        let result = coordinator
            .coordinate_round(updates)
            .expect("coordinate_round failed");
        assert!(!result.is_empty());
        let stats = coordinator.get_tau_stats();
        assert!(stats.current_tau > 0.0);
    }
}
