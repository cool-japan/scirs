//! Aggregation strategies for federated learning

use crate::error::Result;
use crate::federated::ClientUpdate;
use scirs2_core::ndarray::prelude::*;

/// Trait for aggregation strategies
pub trait AggregationStrategy: Send + Sync {
    /// Aggregate client updates
    fn aggregate(&mut self, updates: &[ClientUpdate], weights: &[f32]) -> Result<Vec<Array2<f32>>>;
    /// Get strategy name
    fn name(&self) -> &str;
}

/// Federated Averaging (FedAvg)
pub struct FedAvg {
    /// Momentum parameter (optional)
    momentum: Option<f32>,
    /// Previous aggregated state (for momentum)
    previous_state: Option<Vec<Array2<f32>>>,
}

impl FedAvg {
    /// Create new FedAvg aggregator
    pub fn new() -> Self {
        Self {
            momentum: None,
            previous_state: None,
        }
    }

    /// Create FedAvg with momentum
    pub fn with_momentum(momentum: f32) -> Self {
        Self {
            momentum: Some(momentum),
            previous_state: None,
        }
    }
}

impl Default for FedAvg {
    fn default() -> Self {
        Self::new()
    }
}

impl AggregationStrategy for FedAvg {
    fn aggregate(&mut self, updates: &[ClientUpdate], weights: &[f32]) -> Result<Vec<Array2<f32>>> {
        if updates.is_empty() {
            return Ok(Vec::new());
        }
        // Get number of weight tensors
        let num_tensors = updates[0].weight_updates.len();
        let mut aggregated = Vec::with_capacity(num_tensors);
        // Aggregate each tensor
        for tensor_idx in 0..num_tensors {
            // Get shape from first update
            let shape = updates[0].weight_updates[tensor_idx].shape();
            let mut weighted_sum = Array2::<f32>::zeros((shape[0], shape[1]));
            // Weighted sum of updates
            for (update, &weight) in updates.iter().zip(weights.iter()) {
                if tensor_idx < update.weight_updates.len() {
                    weighted_sum = weighted_sum + weight * &update.weight_updates[tensor_idx];
                }
            }
            // Apply momentum if configured
            if let (Some(momentum), Some(ref prev_state)) = (self.momentum, &self.previous_state) {
                if tensor_idx < prev_state.len() {
                    weighted_sum =
                        momentum * &prev_state[tensor_idx] + (1.0 - momentum) * &weighted_sum;
                }
            }
            aggregated.push(weighted_sum);
        }
        self.previous_state = Some(aggregated.clone());
        Ok(aggregated)
    }

    fn name(&self) -> &str {
        "FedAvg"
    }
}

/// FedProx - Federated optimization with proximal term
pub struct FedProx {
    /// Proximal parameter (mu)
    mu: f32,
}

impl FedProx {
    /// Create new FedProx aggregator
    pub fn new(mu: f32) -> Self {
        Self { mu }
    }
}

impl AggregationStrategy for FedProx {
    fn aggregate(&mut self, updates: &[ClientUpdate], weights: &[f32]) -> Result<Vec<Array2<f32>>> {
        // FedProx aggregation is similar to FedAvg but with proximal term in client optimization
        // The aggregation step itself is the same as FedAvg
        let mut fedavg = FedAvg::new();
        fedavg.aggregate(updates, weights)
    }

    fn name(&self) -> &str {
        "FedProx"
    }
}

/// FedYogi - Adaptive federated optimization
pub struct FedYogi {
    /// Learning rate
    lr: f32,
    /// First moment decay
    beta1: f32,
    /// Second moment decay
    beta2: f32,
    /// Epsilon for numerical stability
    epsilon: f32,
    /// First moment estimates
    m: Option<Vec<Array2<f32>>>,
    /// Second moment estimates
    v: Option<Vec<Array2<f32>>>,
    /// Step counter
    step: usize,
}

impl FedYogi {
    /// Create new FedYogi aggregator with default parameters
    pub fn new() -> Self {
        Self {
            lr: 0.01,
            beta1: 0.9,
            beta2: 0.99,
            epsilon: 1e-3,
            m: None,
            v: None,
            step: 0,
        }
    }

    /// Set learning rate
    pub fn with_lr(mut self, lr: f32) -> Self {
        self.lr = lr;
        self
    }
}

impl Default for FedYogi {
    fn default() -> Self {
        Self::new()
    }
}

impl AggregationStrategy for FedYogi {
    fn aggregate(&mut self, updates: &[ClientUpdate], weights: &[f32]) -> Result<Vec<Array2<f32>>> {
        if updates.is_empty() {
            return Ok(Vec::new());
        }
        self.step += 1;
        // First compute the weighted average (delta)
        let mut fedavg = FedAvg::new();
        let delta = fedavg.aggregate(updates, weights)?;

        // Initialize moment estimates if needed
        if self.m.is_none() {
            self.m = Some(delta.iter().map(|d| Array2::zeros(d.raw_dim())).collect());
            self.v = Some(delta.iter().map(|d| Array2::zeros(d.raw_dim())).collect());
        }

        let m_ref = self.m.as_mut().expect("m initialized above");
        let v_ref = self.v.as_mut().expect("v initialized above");

        let mut aggregated = Vec::with_capacity(delta.len());
        let step_f = self.step as f32;

        for (tensor_idx, delta_t) in delta.into_iter().enumerate() {
            // Ensure capacity
            if tensor_idx >= m_ref.len() {
                m_ref.push(Array2::zeros(delta_t.raw_dim()));
                v_ref.push(Array2::zeros(delta_t.raw_dim()));
            }

            // Update first moment
            let m_t = &m_ref[tensor_idx] * self.beta1 + &delta_t * (1.0 - self.beta1);

            // FedYogi update: v_t = v_{t-1} - (1-beta2) * sign(v_{t-1} - delta_t^2) * delta_t^2
            let delta_sq = &delta_t * &delta_t;
            let v_t = {
                let diff = &v_ref[tensor_idx] - &delta_sq;
                let sign = diff.mapv(|x| if x > 0.0 { 1.0_f32 } else { -1.0_f32 });
                &v_ref[tensor_idx] - (1.0 - self.beta2) * &sign * &delta_sq
            };

            // Bias correction
            let m_hat = &m_t / (1.0 - self.beta1.powf(step_f));
            let v_hat = &v_t / (1.0 - self.beta2.powf(step_f));

            // Compute update
            let update = self.lr * &m_hat / (v_hat.mapv(f32::sqrt) + self.epsilon);

            m_ref[tensor_idx] = m_t;
            v_ref[tensor_idx] = v_t;
            aggregated.push(update);
        }
        Ok(aggregated)
    }

    fn name(&self) -> &str {
        "FedYogi"
    }
}

/// Robust aggregation using trimmed mean
pub struct TrimmedMean {
    /// Fraction to trim from each end
    trim_ratio: f32,
}

impl TrimmedMean {
    /// Create new trimmed mean aggregator
    pub fn new(trim_ratio: f32) -> Self {
        Self { trim_ratio }
    }
}

impl AggregationStrategy for TrimmedMean {
    fn aggregate(
        &mut self,
        updates: &[ClientUpdate],
        _weights: &[f32],
    ) -> Result<Vec<Array2<f32>>> {
        if updates.is_empty() {
            return Ok(Vec::new());
        }
        let num_clients = updates.len();
        let trim_count = (num_clients as f32 * self.trim_ratio) as usize;
        let num_tensors = updates[0].weight_updates.len();
        let mut aggregated = Vec::with_capacity(num_tensors);

        for tensor_idx in 0..num_tensors {
            let shape = updates[0].weight_updates[tensor_idx].shape();
            let mut result = Array2::<f32>::zeros((shape[0], shape[1]));
            // For each element in the tensor
            for i in 0..shape[0] {
                for j in 0..shape[1] {
                    // Collect values from all clients
                    let mut values: Vec<f32> = updates
                        .iter()
                        .filter_map(|u| {
                            if tensor_idx < u.weight_updates.len() {
                                Some(u.weight_updates[tensor_idx][[i, j]])
                            } else {
                                None
                            }
                        })
                        .collect();
                    // Sort and trim
                    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    let end = values.len().saturating_sub(trim_count);
                    let trimmed = &values[trim_count.min(end)..end];
                    // Compute mean of trimmed values
                    if !trimmed.is_empty() {
                        result[[i, j]] = trimmed.iter().sum::<f32>() / trimmed.len() as f32;
                    }
                }
            }
            aggregated.push(result);
        }
        Ok(aggregated)
    }

    fn name(&self) -> &str {
        "TrimmedMean"
    }
}

/// Krum aggregation for Byzantine robustness
pub struct Krum {
    /// Number of Byzantine clients to tolerate
    num_byzantine: usize,
    /// Whether to use Multi-Krum
    multi_krum: bool,
}

impl Krum {
    /// Create new Krum aggregator
    pub fn new(num_byzantine: usize) -> Self {
        Self {
            num_byzantine,
            multi_krum: false,
        }
    }

    /// Enable Multi-Krum
    pub fn with_multi_krum(mut self) -> Self {
        self.multi_krum = true;
        self
    }

    /// Compute L2 distance between two updates
    fn compute_distance(&self, update1: &ClientUpdate, update2: &ClientUpdate) -> Result<f32> {
        let mut total_dist = 0.0;
        for (w1, w2) in update1
            .weight_updates
            .iter()
            .zip(update2.weight_updates.iter())
        {
            let diff = w1 - w2;
            total_dist += diff.iter().map(|x| x * x).sum::<f32>();
        }
        Ok(total_dist.sqrt())
    }
}

impl AggregationStrategy for Krum {
    fn aggregate(
        &mut self,
        updates: &[ClientUpdate],
        _weights: &[f32],
    ) -> Result<Vec<Array2<f32>>> {
        if updates.is_empty() {
            return Ok(Vec::new());
        }
        let num_clients = updates.len();
        let num_select = if self.multi_krum {
            num_clients.saturating_sub(self.num_byzantine)
        } else {
            1
        };

        // Compute pairwise distances
        let mut distances = vec![vec![0.0f32; num_clients]; num_clients];
        for i in 0..num_clients {
            for j in (i + 1)..num_clients {
                let dist = self.compute_distance(&updates[i], &updates[j])?;
                distances[i][j] = dist;
                distances[j][i] = dist;
            }
        }

        // Compute scores (sum of k nearest distances)
        let k = num_clients.saturating_sub(self.num_byzantine + 2);
        let mut scores = vec![0.0f32; num_clients];
        for i in 0..num_clients {
            let mut dists = distances[i].clone();
            dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            // Skip self (0 distance at index 0), sum k nearest
            scores[i] = dists[1..=k.min(dists.len().saturating_sub(1))].iter().sum();
        }

        // Select clients with lowest scores
        let mut indices: Vec<usize> = (0..num_clients).collect();
        indices.sort_by(|&i, &j| {
            scores[i]
                .partial_cmp(&scores[j])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let selected = &indices[..num_select.min(indices.len())];

        // Average selected updates
        let selected_updates: Vec<ClientUpdate> =
            selected.iter().map(|&i| updates[i].clone()).collect();
        let equal_weights = vec![1.0 / num_select as f32; num_select];
        let mut fedavg = FedAvg::new();
        fedavg.aggregate(&selected_updates, &equal_weights)
    }

    fn name(&self) -> &str {
        if self.multi_krum {
            "Multi-Krum"
        } else {
            "Krum"
        }
    }
}

/// Median aggregation
pub struct Median;

impl Median {
    /// Create new median aggregator
    pub fn new() -> Self {
        Self
    }
}

impl Default for Median {
    fn default() -> Self {
        Self::new()
    }
}

impl AggregationStrategy for Median {
    fn aggregate(
        &mut self,
        updates: &[ClientUpdate],
        _weights: &[f32],
    ) -> Result<Vec<Array2<f32>>> {
        if updates.is_empty() {
            return Ok(Vec::new());
        }
        let num_tensors = updates[0].weight_updates.len();
        let mut aggregated = Vec::with_capacity(num_tensors);

        for tensor_idx in 0..num_tensors {
            let shape = updates[0].weight_updates[tensor_idx].shape();
            let mut result = Array2::<f32>::zeros((shape[0], shape[1]));
            // For each element, compute median
            for i in 0..shape[0] {
                for j in 0..shape[1] {
                    let mut values: Vec<f32> = updates
                        .iter()
                        .filter_map(|u| {
                            if tensor_idx < u.weight_updates.len() {
                                Some(u.weight_updates[tensor_idx][[i, j]])
                            } else {
                                None
                            }
                        })
                        .collect();
                    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    let median = if values.len().is_multiple_of(2) {
                        (values[values.len() / 2 - 1] + values[values.len() / 2]) / 2.0
                    } else {
                        values[values.len() / 2]
                    };
                    result[[i, j]] = median;
                }
            }
            aggregated.push(result);
        }
        Ok(aggregated)
    }

    fn name(&self) -> &str {
        "Median"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_updates() -> Vec<ClientUpdate> {
        vec![
            ClientUpdate {
                client_id: 0,
                weight_updates: vec![Array2::ones((2, 2))],
                num_samples: 100,
                loss: 0.5,
                accuracy: 0.9,
            },
            ClientUpdate {
                client_id: 1,
                weight_updates: vec![Array2::ones((2, 2)) * 2.0],
                num_samples: 200,
                loss: 0.4,
                accuracy: 0.92,
            },
        ]
    }

    #[test]
    fn test_fedavg() {
        let mut aggregator = FedAvg::new();
        let updates = create_test_updates();
        let weights = vec![0.5, 0.5];
        let result = aggregator
            .aggregate(&updates, &weights)
            .expect("fedavg failed");
        assert_eq!(result.len(), 1);
        assert!((result[0][[0, 0]] - 1.5).abs() < 1e-5); // Average of 1 and 2
    }

    #[test]
    fn test_median() {
        let mut aggregator = Median::new();
        let updates = create_test_updates();
        let weights = vec![0.5, 0.5]; // Weights ignored for median
        let result = aggregator
            .aggregate(&updates, &weights)
            .expect("median failed");
        assert_eq!(result.len(), 1);
        assert!((result[0][[0, 0]] - 1.5).abs() < 1e-5); // Median of [1, 2]
    }

    #[test]
    fn test_trimmed_mean() {
        let mut aggregator = TrimmedMean::new(0.0);
        let updates = create_test_updates();
        let weights = vec![0.5, 0.5];
        let result = aggregator
            .aggregate(&updates, &weights)
            .expect("trimmed mean failed");
        assert_eq!(result.len(), 1);
    }
}
