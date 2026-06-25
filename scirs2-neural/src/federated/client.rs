//! Federated learning client implementation

use crate::error::Result;
use crate::federated::ClientUpdate;
use crate::models::sequential::Sequential;
use scirs2_core::ndarray::prelude::*;

/// Configuration for a federated client
#[derive(Debug, Clone)]
pub struct ClientConfig {
    /// Unique client identifier
    pub client_id: usize,
    /// Number of local training epochs
    pub local_epochs: usize,
    /// Local batch size
    pub batch_size: usize,
    /// Learning rate
    pub learning_rate: f32,
    /// Enable differential privacy
    pub enable_privacy: bool,
    /// Privacy budget (epsilon)
    pub privacy_budget: Option<f64>,
}

/// Federated learning client
pub struct FederatedClient {
    config: ClientConfig,
    /// Local model (optional, for stateful clients)
    #[allow(dead_code)]
    local_model: Option<Sequential<f32>>,
    /// Training history
    history: Vec<LocalTrainingRound>,
    /// Privacy accountant
    privacy_accountant: Option<PrivacyAccountant>,
}

/// Local training round information
#[allow(dead_code)]
struct LocalTrainingRound {
    round: usize,
    loss: f32,
    accuracy: f32,
    samples_processed: usize,
}

/// Privacy accountant for differential privacy
struct PrivacyAccountant {
    epsilon_spent: f64,
    delta: f64,
    max_epsilon: f64,
}

impl FederatedClient {
    /// Create a new federated client
    pub fn new(config: ClientConfig) -> Result<Self> {
        let privacy_accountant = if config.enable_privacy {
            Some(PrivacyAccountant {
                epsilon_spent: 0.0,
                delta: 1e-5,
                max_epsilon: config.privacy_budget.unwrap_or(10.0),
            })
        } else {
            None
        };
        Ok(Self {
            config,
            local_model: None,
            history: Vec::new(),
            privacy_accountant,
        })
    }

    /// Train on local data.
    ///
    /// Runs real local SGD: the global weights are copied, optimised on the
    /// client's data for `local_epochs`, and the difference (`local - global`)
    /// is returned as the client update (optionally privatised).
    ///
    /// The weight list is interpreted as a bias-free multi-layer perceptron with
    /// ReLU activations between hidden layers and a softmax + cross-entropy head.
    pub fn train_on_local_data(
        &mut self,
        global_weights: &[Array2<f32>],
        data: &ArrayView2<f32>,
        labels: &ArrayView1<usize>,
    ) -> Result<ClientUpdate> {
        let num_samples = data.shape()[0];
        if global_weights.is_empty() {
            return Err(crate::error::NeuralError::InvalidArgument(
                "global_weights must contain at least one layer".to_string(),
            ));
        }
        if data.shape()[1] != global_weights[0].shape()[0] {
            return Err(crate::error::NeuralError::DimensionMismatch(format!(
                "input feature dim {} does not match first layer input {}",
                data.shape()[1],
                global_weights[0].shape()[0]
            )));
        }

        // Local working copy that we actually optimise.
        let mut local_weights: Vec<Array2<f32>> = global_weights.to_vec();

        let mut total_loss = 0.0;
        let mut total_accuracy = 0.0;
        let epochs = self.config.local_epochs.max(1);
        for _epoch in 0..epochs {
            let (epoch_loss, epoch_acc) = self.train_epoch(&mut local_weights, data, labels)?;
            total_loss += epoch_loss;
            total_accuracy += epoch_acc;
        }
        let avg_loss = total_loss / epochs as f32;
        let avg_accuracy = total_accuracy / epochs as f32;

        // Client update = trained local weights - received global weights.
        let weight_updates = self.calculate_weight_updates(&local_weights, global_weights)?;

        // Apply differential privacy if enabled.
        let weight_updates = if self.config.enable_privacy {
            self.apply_differential_privacy(weight_updates)?
        } else {
            weight_updates
        };

        self.history.push(LocalTrainingRound {
            round: self.history.len(),
            loss: avg_loss,
            accuracy: avg_accuracy,
            samples_processed: num_samples,
        });

        Ok(ClientUpdate {
            client_id: self.config.client_id,
            weight_updates,
            num_samples,
            loss: avg_loss,
            accuracy: avg_accuracy,
        })
    }

    /// Train for one epoch with real forward/backward SGD over mini-batches.
    fn train_epoch(
        &self,
        weights: &mut [Array2<f32>],
        data: &ArrayView2<f32>,
        labels: &ArrayView1<usize>,
    ) -> Result<(f32, f32)> {
        let num_samples = data.shape()[0];
        if num_samples == 0 {
            return Ok((0.0, 0.0));
        }
        let batch_size = self.config.batch_size.max(1);
        let num_batches = num_samples.div_ceil(batch_size);
        let mut total_loss = 0.0;
        let mut correct = 0usize;

        // Shuffle sample order each epoch.
        let mut indices: Vec<usize> = (0..num_samples).collect();
        use scirs2_core::random::rng;
        use scirs2_core::random::seq::SliceRandom;
        let mut rng_inst = rng();
        indices.shuffle(&mut rng_inst);

        for batch_idx in 0..num_batches {
            let start = batch_idx * batch_size;
            let end = ((batch_idx + 1) * batch_size).min(num_samples);
            let batch_indices = &indices[start..end];
            let batch_data = self.get_batch_data(data, batch_indices);
            let batch_labels = self.get_batch_labels(labels, batch_indices);

            let (batch_loss, batch_correct) = Self::sgd_step(
                weights,
                &batch_data,
                &batch_labels,
                self.config.learning_rate,
            )?;
            total_loss += batch_loss * batch_indices.len() as f32;
            correct += batch_correct;
        }

        let avg_loss = total_loss / num_samples as f32;
        let accuracy = correct as f32 / num_samples as f32;
        Ok((avg_loss, accuracy))
    }

    /// One SGD step on a mini-batch. Returns `(mean_loss, num_correct)`.
    ///
    /// The network is a bias-free MLP: `z_l = a_l W_l`, ReLU on hidden layers,
    /// softmax + cross-entropy on the output. Gradients are standard
    /// back-propagation and weights are updated in place.
    fn sgd_step(
        weights: &mut [Array2<f32>],
        data: &Array2<f32>,
        labels: &Array1<usize>,
        learning_rate: f32,
    ) -> Result<(f32, usize)> {
        let num_layers = weights.len();
        let batch = data.shape()[0];
        if batch == 0 || num_layers == 0 {
            return Ok((0.0, 0));
        }

        // Forward pass, caching the input activation of each layer and the
        // pre-activation (z) of each layer.
        let mut activations: Vec<Array2<f32>> = Vec::with_capacity(num_layers + 1);
        let mut pre_activations: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
        activations.push(data.clone());
        for (l, w) in weights.iter().enumerate() {
            let a = &activations[l];
            if a.shape()[1] != w.shape()[0] {
                return Err(crate::error::NeuralError::DimensionMismatch(format!(
                    "layer {l}: activation dim {} != weight input dim {}",
                    a.shape()[1],
                    w.shape()[0]
                )));
            }
            let z = a.dot(w);
            pre_activations.push(z.clone());
            if l + 1 < num_layers {
                activations.push(z.mapv(|v| v.max(0.0)));
            } else {
                activations.push(z);
            }
        }

        let logits = &activations[num_layers];
        let num_classes = logits.shape()[1];

        // Row-wise softmax + cross-entropy loss + accuracy.
        let mut probs = Array2::<f32>::zeros((batch, num_classes));
        let mut loss = 0.0f32;
        let mut correct = 0usize;
        for i in 0..batch {
            let row = logits.row(i);
            let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for j in 0..num_classes {
                let e = (row[j] - max).exp();
                probs[[i, j]] = e;
                sum += e;
            }
            let label = labels[i];
            if label >= num_classes {
                return Err(crate::error::NeuralError::InvalidArgument(format!(
                    "label {label} out of range for {num_classes} output classes"
                )));
            }
            for j in 0..num_classes {
                probs[[i, j]] /= sum;
            }
            loss += -(probs[[i, label]].max(1e-12)).ln();
            let mut best = 0usize;
            let mut best_v = probs[[i, 0]];
            for j in 1..num_classes {
                if probs[[i, j]] > best_v {
                    best_v = probs[[i, j]];
                    best = j;
                }
            }
            if best == label {
                correct += 1;
            }
        }
        loss /= batch as f32;

        // dlogits = (probs - onehot) / batch
        let mut dz = probs;
        for i in 0..batch {
            dz[[i, labels[i]]] -= 1.0;
        }
        dz.mapv_inplace(|v| v / batch as f32);

        // Back-propagate, updating each weight matrix with SGD.
        for l in (0..num_layers).rev() {
            let a = &activations[l];
            let dw = a.t().dot(&dz); // [d_l, d_{l+1}]
            let da = dz.dot(&weights[l].t()); // [batch, d_l] (uses pre-update W)
            weights[l].scaled_add(-learning_rate, &dw);
            if l > 0 {
                // Back-prop through the ReLU at layer l-1.
                let z_prev = &pre_activations[l - 1];
                let mut dz_prev = da;
                for (g, &z) in dz_prev.iter_mut().zip(z_prev.iter()) {
                    if z <= 0.0 {
                        *g = 0.0;
                    }
                }
                dz = dz_prev;
            }
        }

        Ok((loss, correct))
    }

    /// Get batch data
    pub fn get_batch_data(&self, data: &ArrayView2<f32>, indices: &[usize]) -> Array2<f32> {
        let batch_size = indices.len();
        let feature_dim = data.shape()[1];
        let mut batch = Array2::zeros((batch_size, feature_dim));
        for (i, &idx) in indices.iter().enumerate() {
            batch.row_mut(i).assign(&data.row(idx));
        }
        batch
    }

    /// Get batch labels
    fn get_batch_labels(&self, labels: &ArrayView1<usize>, indices: &[usize]) -> Array1<usize> {
        let batch_size = indices.len();
        let mut batch = Array1::zeros(batch_size);
        for (i, &idx) in indices.iter().enumerate() {
            batch[i] = labels[idx];
        }
        batch
    }

    /// Compute the client update as the difference between trained local
    /// weights and the received global weights.
    fn calculate_weight_updates(
        &self,
        local_weights: &[Array2<f32>],
        global_weights: &[Array2<f32>],
    ) -> Result<Vec<Array2<f32>>> {
        if local_weights.len() != global_weights.len() {
            return Err(crate::error::NeuralError::DimensionMismatch(
                "local and global weight layer counts differ".to_string(),
            ));
        }
        let updates = local_weights
            .iter()
            .zip(global_weights.iter())
            .map(|(local_w, global_w)| local_w - global_w)
            .collect();
        Ok(updates)
    }

    /// Apply differential privacy to weight updates
    fn apply_differential_privacy(
        &mut self,
        mut updates: Vec<Array2<f32>>,
    ) -> Result<Vec<Array2<f32>>> {
        if let Some(ref mut accountant) = self.privacy_accountant {
            // Clip gradients
            let clip_threshold = 1.0_f32;
            for update in &mut updates {
                let norm = update.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > clip_threshold {
                    *update *= clip_threshold / norm;
                }
            }
            // Add Gaussian noise
            use scirs2_core::random::{Distribution, Normal};
            let noise_scale = (clip_threshold as f64 * (2.0 * (1.0 / accountant.delta).ln()).sqrt()
                / accountant.max_epsilon) as f32;
            let noise_dist = Normal::new(0.0_f32, noise_scale)
                .map_err(|e| crate::error::NeuralError::InferenceError(format!("{e}")))?;
            let mut rng_inst = scirs2_core::random::rng();
            for update in updates.iter_mut() {
                for elem in update.iter_mut() {
                    *elem += noise_dist.sample(&mut rng_inst);
                }
            }
            // Update privacy budget
            let epsilon_per_step = accountant.max_epsilon / 100.0;
            accountant.epsilon_spent += epsilon_per_step;
        }
        Ok(updates)
    }

    /// Get client statistics
    pub fn get_statistics(&self) -> ClientStatistics {
        let total_samples: usize = self.history.iter().map(|r| r.samples_processed).sum();
        let avg_loss = if self.history.is_empty() {
            0.0
        } else {
            self.history.iter().map(|r| r.loss).sum::<f32>() / self.history.len() as f32
        };
        let avg_accuracy = if self.history.is_empty() {
            0.0
        } else {
            self.history.iter().map(|r| r.accuracy).sum::<f32>() / self.history.len() as f32
        };
        ClientStatistics {
            rounds_participated: self.history.len(),
            total_samples_processed: total_samples,
            average_loss: avg_loss,
            average_accuracy: avg_accuracy,
            privacy_spent: self.privacy_accountant.as_ref().map(|a| a.epsilon_spent),
        }
    }
}

/// Client statistics
pub struct ClientStatistics {
    pub rounds_participated: usize,
    pub total_samples_processed: usize,
    pub average_loss: f32,
    pub average_accuracy: f32,
    pub privacy_spent: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_creation() {
        let config = ClientConfig {
            client_id: 0,
            local_epochs: 5,
            batch_size: 32,
            learning_rate: 0.01,
            enable_privacy: false,
            privacy_budget: None,
        };
        let client = FederatedClient::new(config).expect("FederatedClient::new failed");
        assert_eq!(client.config.client_id, 0);
    }

    #[test]
    fn test_batch_extraction() {
        let config = ClientConfig {
            client_id: 0,
            local_epochs: 1,
            batch_size: 2,
            learning_rate: 0.01,
            enable_privacy: false,
            privacy_budget: None,
        };
        let client = FederatedClient::new(config).expect("FederatedClient::new failed");
        let data = Array2::from_shape_vec(
            (4, 3),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .expect("from_shape_vec failed");
        let indices = vec![1, 3];
        let batch = client.get_batch_data(&data.view(), &indices);
        assert_eq!(batch.shape(), &[2, 3]);
        assert!((batch[[0, 0]] - 4.0).abs() < 1e-5);
        assert!((batch[[1, 0]] - 10.0).abs() < 1e-5);
    }

    #[test]
    fn test_local_training_produces_real_updates() {
        let config = ClientConfig {
            client_id: 0,
            local_epochs: 5,
            batch_size: 4,
            learning_rate: 0.1,
            enable_privacy: false,
            privacy_budget: None,
        };
        let mut client = FederatedClient::new(config).expect("client");
        // Single linear layer: 2 features -> 2 classes.
        let global = vec![Array2::<f32>::zeros((2, 2))];
        // Linearly separable: class 1 at x0>0, class 0 at x0<0.
        let data = Array2::from_shape_vec((4, 2), vec![1.0, 0.0, 1.0, 0.5, -1.0, 0.0, -1.0, 0.5])
            .expect("data");
        let labels = Array1::from_vec(vec![1usize, 1, 0, 0]);
        let update = client
            .train_on_local_data(&global, &data.view(), &labels.view())
            .expect("train");

        // Real SGD must produce data-dependent, varying updates -- the old stub
        // returned a constant 0.01 everywhere.
        assert_eq!(update.weight_updates.len(), 1);
        let w = &update.weight_updates[0];
        assert!(
            w.iter().any(|&v| v.abs() > 1e-6),
            "weight updates should be non-zero after real SGD"
        );
        let first = w[[0, 0]];
        assert!(
            !w.iter().all(|&v| (v - first).abs() < 1e-9),
            "updates must vary across entries, not be a constant fill"
        );
        assert!(update.loss.is_finite() && update.loss > 0.0);
        assert!((0.0..=1.0).contains(&update.accuracy));
    }
}
