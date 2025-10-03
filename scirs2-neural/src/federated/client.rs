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
    local_model: Option<Sequential<f32>>,
    /// Training history
    history: Vec<LocalTrainingRound>,
    /// Privacy accountant
    privacy_accountant: Option<PrivacyAccountant>,
/// Local training round information
struct LocalTrainingRound {
    round: usize,
    loss: f32,
    accuracy: f32,
    samples_processed: usize,
/// Privacy accountant for differential privacy
struct PrivacyAccountant {
    epsilon_spent: f64,
    delta: f64,
    max_epsilon: f64,
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
    /// Train on local data
    pub fn train_on_local_data(
        &mut self,
        global_weights: &[Array2<f32>],
        data: &ArrayView2<f32>,
        labels: &ArrayView1<usize>,
    ) -> Result<ClientUpdate> {
        // Initialize local model with global weights
        let mut local_model = self.create_model_from_weights(global_weights)?;
        let num_samples = data.shape()[0];
        let mut total_loss = 0.0;
        let mut correct_predictions = 0;
        // Local training epochs
        for epoch in 0..self.config.local_epochs {
            let (epoch_loss, epoch_acc) = self.train_epoch(&mut local_model, data, labels)?;
            total_loss += epoch_loss;
            correct_predictions += (epoch_acc * num_samples as f32) as usize;
        }
        // Calculate weight updates (difference from global weights)
        let weight_updates = self.calculate_weight_updates(&local_model, global_weights)?;
        // Apply differential privacy if enabled
        let weight_updates = if self.config.enable_privacy {
            self.apply_differential_privacy(weight_updates)?
            weight_updates
        let avg_loss = total_loss / self.config.local_epochs as f32;
        let avg_accuracy =
            correct_predictions as f32 / (num_samples * self.config.local_epochs) as f32;
        // Record training round
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
    /// Train for one epoch
    fn train_epoch(
        &self,
        model: &mut Sequential<f32>,
    ) -> Result<(f32, f32)> {
        let num_batches = (num_samples + self.config.batch_size - 1) / self.config.batch_size;
        let mut correct = 0;
        // Shuffle indices
        let mut indices: Vec<usize> = (0..num_samples).collect();
        use scirs2_core::random::prelude::*;
use scirs2_core::ndarray::ArrayView1;
use scirs2_core::random::seq::SliceRandom;
        indices.shuffle(&mut rng());
        for batch_idx in 0..num_batches {
            let start = batch_idx * self.config.batch_size;
            let end = ((batch_idx + 1) * self.config.batch_size).min(num_samples);
            // Get batch
            let batch_indices = &indices[start..end];
            let batch_data = self.get_batch_data(data, batch_indices);
            let batch_labels = self.get_batch_labels(labels, batch_indices);
            // Forward pass (simplified)
            let predictions = model.forward(&batch_data.into_dyn())?;
            // Calculate loss (simplified cross-entropy)
            let batch_loss = self.calculate_loss(&predictions, &batch_labels)?;
            total_loss += batch_loss * batch_indices.len() as f32;
            // Calculate accuracy
            let batch_correct = self.calculate_correct_predictions(&predictions, &batch_labels)?;
            correct += batch_correct;
            // Backward pass would go here in a real implementation
            // For now, we simulate training progress
        let avg_loss = total_loss / num_samples as f32;
        let accuracy = correct as f32 / num_samples as f32;
        Ok((avg_loss, accuracy))
    /// Get batch data
    fn get_batch_data(&self, data: &ArrayView2<f32>, indices: &[usize]) -> Array2<f32> {
        let batch_size = indices.len();
        let feature_dim = data.shape()[1];
        let mut batch = Array2::zeros((batch_size, feature_dim));
        for (i, &idx) in indices.iter().enumerate() {
            batch.row_mut(i).assign(&data.row(idx));
        batch
    /// Get batch labels
    fn get_batch_labels(&self, labels: &ArrayView1<usize>, indices: &[usize]) -> Array1<usize> {
        let mut batch = Array1::zeros(batch_size);
            batch[i] = labels[idx];
    /// Calculate loss (simplified)
    fn calculate_loss(&self, predictions: &ArrayD<f32>, labels: &Array1<usize>) -> Result<f32> {
        // Simplified cross-entropy loss
        let batch_size = labels.len();
        let mut loss = 0.0;
        for i in 0..batch_size {
            let label = labels[i];
            let pred_slice = predictions.slice(s![i, ..]);
            let max_val = pred_slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            // Stable softmax
            let exp_sum: f32 = pred_slice.iter().map(|&x| (x - max_val).exp()).sum();
            let log_prob = pred_slice[label] - max_val - exp_sum.ln();
            loss -= log_prob;
        Ok(loss / batch_size as f32)
    /// Calculate correct predictions
    fn calculate_correct_predictions(
        predictions: &ArrayD<f32>,
        labels: &Array1<usize>,
    ) -> Result<usize> {
            let predicted_class = pred_slice
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx_)| idx)
                .unwrap_or(0);
            if predicted_class == labels[i] {
                correct += 1;
            }
        Ok(correct)
    /// Create model from weights
    fn create_model_from_weights(&self, weights: &[Array2<f32>]) -> Result<Sequential<f32>> {
        // Simplified implementation
        // In practice, would reconstruct the model architecture and load weights
        Ok(Sequential::new())
    /// Calculate weight updates
    fn calculate_weight_updates(
        local_model: &Sequential<f32>,
    ) -> Result<Vec<Array2<f32>>> {
        // Return difference between local and global weights
        let mut updates = Vec::new();
        for (i, global_w) in globalweights.iter().enumerate() {
            // In practice, would get actual local weights
            let local_w = global_w + 0.01; // Simulated update
            updates.push(local_w - global_w);
        Ok(updates)
    /// Apply differential privacy to weight updates
    fn apply_differential_privacy(
        mut updates: Vec<Array2<f32>>,
        if let Some(ref mut accountant) = self.privacy_accountant {
            // Clip gradients
            let clip_threshold = 1.0;
            for update in &mut updates {
                let norm = update.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > clip_threshold {
                    *update *= clip_threshold / norm;
                }
            // Add Gaussian noise
            use scirs2_core::random::{Distribution, Normal};
            let noise_scale = clip_threshold * (2.0 * (1.0 / accountant.delta).ln()).sqrt()
                / accountant.max_epsilon;
            let noise_dist = Normal::new(0.0, noise_scale).unwrap();
            let mut rng = rng();
                for elem in update.iter_mut() {
                    *elem += noise_dist.sample(&mut rng);
            // Update privacy budget
            let epsilon_per_step = accountant.max_epsilon / 100.0; // Simplified
            accountant.epsilon_spent += epsilon_per_step;
    /// Get client statistics
    pub fn get_statistics(&self) -> ClientStatistics {
        let total_samples: usize = self.history.iter().map(|r| r.samples_processed).sum();
        let avg_loss = if self.history.is_empty() {
            0.0
            self.history.iter().map(|r| r.loss).sum::<f32>() / self.history.len() as f32
        let avg_accuracy = if self.history.is_empty() {
            self.history.iter().map(|r| r.accuracy).sum::<f32>() / self.history.len() as f32
        ClientStatistics {
            rounds_participated: self.history.len(),
            total_samples_processed: total_samples,
            average_loss: avg_loss,
            average_accuracy: avg_accuracy,
            privacy_spent: self.privacy_accountant.as_ref().map(|a| a.epsilon_spent),
/// Client statistics
pub struct ClientStatistics {
    pub rounds_participated: usize,
    pub total_samples_processed: usize,
    pub average_loss: f32,
    pub average_accuracy: f32,
    pub privacy_spent: Option<f64>,
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
        let client = FederatedClient::new(config).unwrap();
        assert_eq!(client.config.client_id, 0);
    fn test_batch_extraction() {
            local_epochs: 1,
            batch_size: 2,
        let data = Array2::from_shape_vec(
            (4, 3),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .unwrap();
        let indices = vec![1, 3];
        let batch = client.get_batch_data(&data.view(), &indices);
        assert_eq!(batch.shape(), &[2, 3]);
        assert_eq!(batch[[0, 0]], 4.0);
        assert_eq!(batch[[1, 0]], 10.0);
