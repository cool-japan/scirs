//! Advanced federated learning algorithms
//!
//! This module implements state-of-the-art federated learning algorithms
//! including SCAFFOLD, FedAvgM, FedAdam, FedAdagrad, FedLAG, and others.

use crate::error::{NeuralError, Result};
use crate::federated::{AggregationStrategy, ClientUpdate};
use scirs2_core::ndarray::prelude::*;
use std::collections::HashMap;

/// SCAFFOLD algorithm - stochastic controlled averaging for federated learning
pub struct SCAFFOLD {
    server_control: Option<Vec<Array2<f32>>>,
    client_controls: HashMap<usize, Vec<Array2<f32>>>,
    server_lr: f32,
    global_lr: f32,
}

impl SCAFFOLD {
    /// Create new SCAFFOLD aggregator
    pub fn new(server_lr: f32, global_lr: f32) -> Self {
        Self {
            server_control: None,
            client_controls: HashMap::new(),
            server_lr,
            global_lr,
        }
    }

    /// Update client control variate
    pub fn update_client_control(
        &mut self,
        client_id: usize,
        old_weights: &[Array2<f32>],
        new_weights: &[Array2<f32>],
        local_steps: usize,
        local_lr: f32,
    ) -> Result<()> {
        let mut new_control = Vec::new();
        let denom = (local_steps.max(1) as f32) * local_lr;

        if let Some(old_control) = self.client_controls.get(&client_id).cloned() {
            let server_control = self.server_control.clone().unwrap_or_default();
            for (idx, ((old_w, new_w), old_c)) in old_weights
                .iter()
                .zip(new_weights.iter())
                .zip(old_control.iter())
                .enumerate()
            {
                let server_c = server_control
                    .get(idx)
                    .cloned()
                    .unwrap_or_else(|| Array2::zeros(old_w.raw_dim()));
                let gradient_term = (old_w - new_w) / denom;
                let new_c = old_c - &server_c + &gradient_term;
                new_control.push(new_c);
            }
        } else {
            // Initialize control variate for new client
            for (old_w, new_w) in old_weights.iter().zip(new_weights.iter()) {
                let gradient_term = (old_w - new_w) / denom;
                new_control.push(gradient_term);
            }
        }
        self.client_controls.insert(client_id, new_control);
        Ok(())
    }

    /// Update server control variate
    fn update_server_control(&mut self, client_updates: &[ClientUpdate]) -> Result<()> {
        if client_updates.is_empty() {
            return Ok(());
        }
        let num_tensors = client_updates[0].weight_updates.len();
        let mut new_server_control = Vec::new();
        for tensor_idx in 0..num_tensors {
            let shape = client_updates[0].weight_updates[tensor_idx].shape();
            let mut control_sum = Array2::zeros((shape[0], shape[1]));
            let mut total_samples = 0_usize;
            for update in client_updates {
                if let Some(client_control) = self.client_controls.get(&update.client_id) {
                    if tensor_idx < client_control.len() {
                        control_sum =
                            control_sum + &client_control[tensor_idx] * update.num_samples as f32;
                        total_samples += update.num_samples;
                    }
                }
            }
            if total_samples > 0 {
                control_sum /= total_samples as f32;
            }
            new_server_control.push(control_sum);
        }
        self.server_control = Some(new_server_control);
        Ok(())
    }
}

impl AggregationStrategy for SCAFFOLD {
    fn aggregate(&mut self, updates: &[ClientUpdate], weights: &[f32]) -> Result<Vec<Array2<f32>>> {
        if updates.is_empty() {
            return Ok(Vec::new());
        }
        self.update_server_control(updates)?;
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
            // Apply control variate correction
            if let Some(ref server_control) = self.server_control {
                if tensor_idx < server_control.len() {
                    weighted_sum = weighted_sum + &server_control[tensor_idx] * self.global_lr;
                }
            }
            aggregated.push(weighted_sum * self.server_lr);
        }
        Ok(aggregated)
    }

    fn name(&self) -> &str {
        "SCAFFOLD"
    }
}

/// FedAvgM - FedAvg with server momentum
pub struct FedAvgM {
    server_lr: f32,
    momentum: f32,
    momentum_buffers: Option<Vec<Array2<f32>>>,
}

impl FedAvgM {
    /// Create new FedAvgM aggregator
    pub fn new(server_lr: f32, momentum: f32) -> Self {
        Self {
            server_lr,
            momentum,
            momentum_buffers: None,
        }
    }
}

impl AggregationStrategy for FedAvgM {
    fn aggregate(&mut self, updates: &[ClientUpdate], weights: &[f32]) -> Result<Vec<Array2<f32>>> {
        if updates.is_empty() {
            return Ok(Vec::new());
        }
        // First compute weighted average
        let num_tensors = updates[0].weight_updates.len();
        let mut aggregated = Vec::with_capacity(num_tensors);
        for tensor_idx in 0..num_tensors {
            let shape = updates[0].weight_updates[tensor_idx].shape();
            let mut weighted_sum = Array2::<f32>::zeros((shape[0], shape[1]));
            for (update, &weight) in updates.iter().zip(weights.iter()) {
                if tensor_idx < update.weight_updates.len() {
                    weighted_sum = weighted_sum + weight * &update.weight_updates[tensor_idx];
                }
            }
            aggregated.push(weighted_sum);
        }

        // Apply server momentum
        if self.momentum_buffers.is_none() {
            self.momentum_buffers = Some(
                aggregated
                    .iter()
                    .map(|a| Array2::zeros(a.raw_dim()))
                    .collect(),
            );
        }
        if let Some(ref mut buffers) = self.momentum_buffers {
            for (update, buffer) in aggregated.iter_mut().zip(buffers.iter_mut()) {
                *buffer = &*buffer * self.momentum + &*update * self.server_lr;
                *update = buffer.clone();
            }
        }
        Ok(aggregated)
    }

    fn name(&self) -> &str {
        "FedAvgM"
    }
}

/// FedAdam - Adaptive federated optimization
pub struct FedAdam {
    lr: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    m: Option<Vec<Array2<f32>>>,
    v: Option<Vec<Array2<f32>>>,
    step: usize,
}

impl FedAdam {
    /// Create new FedAdam aggregator
    pub fn new(lr: f32, beta1: f32, beta2: f32, epsilon: f32) -> Self {
        Self {
            lr,
            beta1,
            beta2,
            epsilon,
            m: None,
            v: None,
            step: 0,
        }
    }
}

impl AggregationStrategy for FedAdam {
    fn aggregate(&mut self, updates: &[ClientUpdate], weights: &[f32]) -> Result<Vec<Array2<f32>>> {
        if updates.is_empty() {
            return Ok(Vec::new());
        }
        self.step += 1;
        // First compute weighted average (gradient)
        let num_tensors = updates[0].weight_updates.len();
        let mut gradients = Vec::with_capacity(num_tensors);
        for tensor_idx in 0..num_tensors {
            let shape = updates[0].weight_updates[tensor_idx].shape();
            let mut weighted_sum = Array2::<f32>::zeros((shape[0], shape[1]));
            for (update, &weight) in updates.iter().zip(weights.iter()) {
                if tensor_idx < update.weight_updates.len() {
                    weighted_sum = weighted_sum + weight * &update.weight_updates[tensor_idx];
                }
            }
            gradients.push(weighted_sum);
        }

        // Initialize moment estimates if needed
        if self.m.is_none() {
            self.m = Some(
                gradients
                    .iter()
                    .map(|g| Array2::zeros(g.raw_dim()))
                    .collect(),
            );
            self.v = Some(
                gradients
                    .iter()
                    .map(|g| Array2::zeros(g.raw_dim()))
                    .collect(),
            );
        }

        let m = self.m.as_mut().expect("m initialized above");
        let v = self.v.as_mut().expect("v initialized above");
        let mut aggregated = Vec::with_capacity(gradients.len());

        for (i, grad) in gradients.into_iter().enumerate() {
            if i >= m.len() {
                m.push(Array2::zeros(grad.raw_dim()));
                v.push(Array2::zeros(grad.raw_dim()));
            }
            // Update biased first moment estimate
            m[i] = &m[i] * self.beta1 + &grad * (1.0 - self.beta1);
            // Update biased second moment estimate
            v[i] = &v[i] * self.beta2 + &grad * &grad * (1.0 - self.beta2);
            // Bias-corrected moment estimates
            let m_hat = &m[i] / (1.0 - self.beta1.powi(self.step as i32));
            let v_hat = &v[i] / (1.0 - self.beta2.powi(self.step as i32));
            let update = &m_hat * self.lr / (v_hat.mapv(f32::sqrt) + self.epsilon);
            aggregated.push(update);
        }
        Ok(aggregated)
    }

    fn name(&self) -> &str {
        "FedAdam"
    }
}

/// FedAdagrad - Adaptive gradient algorithm for federated learning
pub struct FedAdagrad {
    lr: f32,
    epsilon: f32,
    acc_grad: Option<Vec<Array2<f32>>>,
}

impl FedAdagrad {
    /// Create new FedAdagrad aggregator
    pub fn new(lr: f32, epsilon: f32) -> Self {
        Self {
            lr,
            epsilon,
            acc_grad: None,
        }
    }
}

impl AggregationStrategy for FedAdagrad {
    fn aggregate(&mut self, updates: &[ClientUpdate], weights: &[f32]) -> Result<Vec<Array2<f32>>> {
        if updates.is_empty() {
            return Ok(Vec::new());
        }
        let num_tensors = updates[0].weight_updates.len();
        let mut gradients = Vec::with_capacity(num_tensors);
        for tensor_idx in 0..num_tensors {
            let shape = updates[0].weight_updates[tensor_idx].shape();
            let mut weighted_sum = Array2::<f32>::zeros((shape[0], shape[1]));
            for (update, &weight) in updates.iter().zip(weights.iter()) {
                if tensor_idx < update.weight_updates.len() {
                    weighted_sum = weighted_sum + weight * &update.weight_updates[tensor_idx];
                }
            }
            gradients.push(weighted_sum);
        }

        if self.acc_grad.is_none() {
            self.acc_grad = Some(
                gradients
                    .iter()
                    .map(|g| Array2::zeros(g.raw_dim()))
                    .collect(),
            );
        }
        let acc_grad = self.acc_grad.as_mut().expect("acc_grad initialized above");
        let mut aggregated = Vec::with_capacity(gradients.len());

        for (i, grad) in gradients.into_iter().enumerate() {
            if i >= acc_grad.len() {
                acc_grad.push(Array2::zeros(grad.raw_dim()));
            }
            acc_grad[i] = &acc_grad[i] + &grad * &grad;
            let adaptive_lr = acc_grad[i].mapv(f32::sqrt) + self.epsilon;
            let update = &grad * self.lr / &adaptive_lr;
            aggregated.push(update);
        }
        Ok(aggregated)
    }

    fn name(&self) -> &str {
        "FedAdagrad"
    }
}

/// FedLAG - Federated learning with Lookahead and Gradient tracking
pub struct FedLAG {
    k: usize,
    alpha: f32,
    fast_weights: Option<Vec<Array2<f32>>>,
    slow_weights: Option<Vec<Array2<f32>>>,
    step_count: usize,
}

impl FedLAG {
    /// Create new FedLAG aggregator
    pub fn new(k: usize, alpha: f32) -> Self {
        Self {
            k,
            alpha,
            fast_weights: None,
            slow_weights: None,
            step_count: 0,
        }
    }
}

impl AggregationStrategy for FedLAG {
    fn aggregate(&mut self, updates: &[ClientUpdate], weights: &[f32]) -> Result<Vec<Array2<f32>>> {
        if updates.is_empty() {
            return Ok(Vec::new());
        }
        // Standard weighted aggregation
        let num_tensors = updates[0].weight_updates.len();
        let mut aggregated = Vec::with_capacity(num_tensors);
        for tensor_idx in 0..num_tensors {
            let shape = updates[0].weight_updates[tensor_idx].shape();
            let mut weighted_sum = Array2::<f32>::zeros((shape[0], shape[1]));
            for (update, &weight) in updates.iter().zip(weights.iter()) {
                if tensor_idx < update.weight_updates.len() {
                    weighted_sum = weighted_sum + weight * &update.weight_updates[tensor_idx];
                }
            }
            aggregated.push(weighted_sum);
        }

        // Initialize weights if needed
        if self.fast_weights.is_none() {
            self.fast_weights = Some(aggregated.clone());
            self.slow_weights = Some(aggregated.clone());
        }

        // Update fast weights
        if let Some(ref mut fast_weights) = self.fast_weights {
            for (fast_w, update) in fast_weights.iter_mut().zip(&aggregated) {
                *fast_w = &*fast_w + update;
            }
        }

        self.step_count += 1;

        // Update slow weights every k steps
        if self.step_count.is_multiple_of(self.k) {
            let fast = self.fast_weights.clone();
            if let (Some(ref mut slow_weights), Some(ref fast_weights)) =
                (&mut self.slow_weights, &fast)
            {
                for (slow_w, fast_w) in slow_weights.iter_mut().zip(fast_weights) {
                    *slow_w = &*slow_w + &(fast_w - &*slow_w) * self.alpha;
                }
                // Reset fast weights to slow weights
                if let Some(ref mut fw) = self.fast_weights {
                    for (fast_w, slow_w) in fw.iter_mut().zip(slow_weights.iter()) {
                        *fast_w = slow_w.clone();
                    }
                }
            }
        }

        Ok(aggregated)
    }

    fn name(&self) -> &str {
        "FedLAG"
    }
}

/// Information about an aggregator algorithm
#[derive(Debug, Clone)]
pub struct AggregatorInfo {
    pub name: String,
    pub description: String,
    pub key_features: Vec<String>,
    pub recommended_use: String,
}

/// Aggregator factory for creating different algorithms
pub struct AggregatorFactory;

impl AggregatorFactory {
    /// Create aggregator by name
    pub fn create(
        name: &str,
        config: &HashMap<String, f32>,
    ) -> Result<Box<dyn AggregationStrategy>> {
        match name.to_lowercase().as_str() {
            "scaffold" => {
                let server_lr = config.get("server_lr").copied().unwrap_or(1.0);
                let global_lr = config.get("global_lr").copied().unwrap_or(1.0);
                Ok(Box::new(SCAFFOLD::new(server_lr, global_lr)))
            }
            "fedavgm" => {
                let server_lr = config.get("server_lr").copied().unwrap_or(1.0);
                let momentum = config.get("momentum").copied().unwrap_or(0.9);
                Ok(Box::new(FedAvgM::new(server_lr, momentum)))
            }
            "fedadam" => {
                let lr = config.get("lr").copied().unwrap_or(0.001);
                let beta1 = config.get("beta1").copied().unwrap_or(0.9);
                let beta2 = config.get("beta2").copied().unwrap_or(0.999);
                let epsilon = config.get("epsilon").copied().unwrap_or(1e-8);
                Ok(Box::new(FedAdam::new(lr, beta1, beta2, epsilon)))
            }
            "fedadagrad" => {
                let lr = config.get("lr").copied().unwrap_or(0.01);
                let epsilon = config.get("epsilon").copied().unwrap_or(1e-8);
                Ok(Box::new(FedAdagrad::new(lr, epsilon)))
            }
            "fedlag" => {
                let k = config.get("k").copied().unwrap_or(5.0) as usize;
                let alpha = config.get("alpha").copied().unwrap_or(0.5);
                Ok(Box::new(FedLAG::new(k, alpha)))
            }
            _ => Err(NeuralError::InvalidArgument(format!(
                "Unknown aggregator: {}",
                name
            ))),
        }
    }

    /// Get list of available aggregators
    pub fn available_aggregators() -> Vec<&'static str> {
        vec!["scaffold", "fedavgm", "fedadam", "fedadagrad", "fedlag"]
    }

    /// Get default configuration for an aggregator
    pub fn default_config(name: &str) -> HashMap<String, f32> {
        let mut config = HashMap::new();
        match name.to_lowercase().as_str() {
            "scaffold" => {
                config.insert("server_lr".to_string(), 1.0);
                config.insert("global_lr".to_string(), 1.0);
            }
            "fedavgm" => {
                config.insert("server_lr".to_string(), 1.0);
                config.insert("momentum".to_string(), 0.9);
            }
            "fedadam" => {
                config.insert("lr".to_string(), 0.001);
                config.insert("beta1".to_string(), 0.9);
                config.insert("beta2".to_string(), 0.999);
                config.insert("epsilon".to_string(), 1e-8);
            }
            "fedadagrad" => {
                config.insert("lr".to_string(), 0.01);
                config.insert("epsilon".to_string(), 1e-8);
            }
            "fedlag" => {
                config.insert("k".to_string(), 5.0);
                config.insert("alpha".to_string(), 0.5);
            }
            _ => {}
        }
        config
    }

    /// Get detailed aggregator information
    pub fn get_aggregator_info(name: &str) -> Option<AggregatorInfo> {
        match name.to_lowercase().as_str() {
            "scaffold" => Some(AggregatorInfo {
                name: "SCAFFOLD".to_string(),
                description: "Stochastic Controlled Averaging for federated learning".to_string(),
                key_features: vec![
                    "Control variates".to_string(),
                    "Variance reduction".to_string(),
                ],
                recommended_use: "Heterogeneous data distributions".to_string(),
            }),
            "fedavgm" => Some(AggregatorInfo {
                name: "FedAvgM".to_string(),
                description: "FedAvg with server momentum".to_string(),
                key_features: vec![
                    "Server momentum".to_string(),
                    "Improved convergence".to_string(),
                ],
                recommended_use: "General federated learning".to_string(),
            }),
            "fedadam" => Some(AggregatorInfo {
                name: "FedAdam".to_string(),
                description: "Adaptive federated optimization with Adam".to_string(),
                key_features: vec![
                    "Adaptive learning rates".to_string(),
                    "Second-order moments".to_string(),
                ],
                recommended_use: "Tasks requiring adaptive optimization".to_string(),
            }),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::federated::ClientUpdate;

    fn create_test_updates() -> Vec<ClientUpdate> {
        vec![
            ClientUpdate {
                client_id: 0,
                weight_updates: vec![Array2::ones((3, 3))],
                num_samples: 100,
                loss: 0.5,
                accuracy: 0.9,
            },
            ClientUpdate {
                client_id: 1,
                weight_updates: vec![Array2::ones((3, 3)) * 2.0],
                num_samples: 200,
                loss: 0.4,
                accuracy: 0.92,
            },
        ]
    }

    #[test]
    fn test_scaffold() {
        let mut scaffold = SCAFFOLD::new(1.0, 1.0);
        let updates = create_test_updates();
        let weights = vec![0.5, 0.5];
        let result = scaffold
            .aggregate(&updates, &weights)
            .expect("scaffold aggregate failed");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].shape(), &[3, 3]);
    }

    #[test]
    fn test_fedavgm() {
        let mut fedavgm = FedAvgM::new(1.0, 0.9);
        let updates = create_test_updates();
        let weights = vec![0.5, 0.5];
        let result = fedavgm
            .aggregate(&updates, &weights)
            .expect("fedavgm aggregate failed");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].shape(), &[3, 3]);
    }

    #[test]
    fn test_fedadam() {
        let mut fedadam = FedAdam::new(0.001, 0.9, 0.999, 1e-8);
        let updates = create_test_updates();
        let weights = vec![0.5, 0.5];
        let result = fedadam
            .aggregate(&updates, &weights)
            .expect("fedadam aggregate failed");
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_fedadagrad() {
        let mut fedadagrad = FedAdagrad::new(0.01, 1e-8);
        let updates = create_test_updates();
        let weights = vec![0.5, 0.5];
        let result = fedadagrad
            .aggregate(&updates, &weights)
            .expect("fedadagrad aggregate failed");
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_aggregator_factory() {
        let config = AggregatorFactory::default_config("fedadam");
        let aggregator =
            AggregatorFactory::create("fedadam", &config).expect("factory create failed");
        assert_eq!(aggregator.name(), "FedAdam");
        let available = AggregatorFactory::available_aggregators();
        assert!(available.contains(&"scaffold"));
        assert!(available.contains(&"fedavgm"));
        assert!(available.contains(&"fedadam"));
    }
}
