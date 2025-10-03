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
            momentum: Some(_momentum),
impl AggregationStrategy for FedAvg {
    fn aggregate(&mut self, updates: &[ClientUpdate], weights: &[f32]) -> Result<Vec<Array2<f32>>> {
        if updates.is_empty() {
            return Ok(Vec::new());
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
                        momentum * &prev_state[tensor_idx] + (1.0 - momentum) * weighted_sum;
            aggregated.push(weighted_sum);
        Ok(aggregated)
    fn name(&self) -> &str {
        "FedAvg"
/// FedProx - Federated optimization with proximal term
pub struct FedProx {
    /// Proximal parameter (mu)
    mu: f32,
impl FedProx {
    /// Create new FedProx aggregator
    pub fn new(mu: f32) -> Self {
        Self { _mu }
impl AggregationStrategy for FedProx {
        // FedProx aggregation is similar to FedAvg but with proximal term in client optimization
        // The aggregation step itself is the same as FedAvg
        let mut fedavg = FedAvg::new();
        fedavg.aggregate(updates, weights)
        "FedProx"
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
impl FedYogi {
    /// Create new FedYogi aggregator
            lr: 0.01,
            beta1: 0.9,
            beta2: 0.99,
            epsilon: 1e-3,
            m: None,
            v: None,
            step: 0,
    /// Set learning rate
    pub fn with_lr(mut self, lr: f32) -> Self {
        self.lr = lr;
        self
impl AggregationStrategy for FedYogi {
        // First compute the weighted average (delta)
        let fedavg = FedAvg::new();
        let delta = fedavg.aggregate(updates, weights)?;
        // Apply adaptive optimization
        for (tensor_idx, delta_t) in delta.into_iter().enumerate() {
            let shape = delta_t.shape();
            // Initialize moment estimates if needed
            let m_t = if let Some(ref m) = self.m {
                if tensor_idx < m.len() {
                    self.beta1 * &m[tensor_idx] + (1.0 - self.beta1) * &delta_t
                } else {
                    (1.0 - self.beta1) * &delta_t
            } else {
                (1.0 - self.beta1) * &delta_t
            };
            let v_t = if let Some(ref v) = self.v {
                if tensor_idx < v.len() {
                    // FedYogi update: v_t = v_{t-1} - (1-beta2) * sign(v_{t-1} - delta_t^2) * delta_t^2
                    let delta_sq = &delta_t * &delta_t;
                    let sign =
                        (&v[tensor_idx] - &delta_sq).mapv(|x| if x > 0.0 { 1.0 } else { -1.0 });
                    &v[tensor_idx] - (1.0 - self.beta2) * sign * delta_sq
                    &delta_t * &delta_t
                &delta_t * &delta_t
            // Bias correction
            let step_f = (self.step + 1) as f32;
            let m_hat = &m_t / (1.0 - self.beta1.powf(step_f));
            let v_hat = &v_t / (1.0 - self.beta2.powf(step_f));
            // Compute update
            let update = self.lr * m_hat / (v_hat.mapv(f32::sqrt) + self.epsilon);
            aggregated.push(update);
        "FedYogi"
/// Robust aggregation using trimmed mean
pub struct TrimmedMean {
    /// Fraction to trim from each end
    trim_ratio: f32,
impl TrimmedMean {
    /// Create new trimmed mean aggregator
    pub fn new(_trimratio: f32) -> Self {
        Self { _trim_ratio }
impl AggregationStrategy for TrimmedMean {
    fn aggregate(&mut self, updates: &[ClientUpdate], weights: &[f32]) -> Result<Vec<Array2<f32>>> {
        let num_clients = updates.len();
        let trim_count = (num_clients as f32 * self.trim_ratio) as usize;
            let mut result = Array2::<f32>::zeros((shape[0], shape[1]));
            // For each element in the tensor
            for i in 0..shape[0] {
                for j in 0..shape[1] {
                    // Collect values from all clients
                    let mut values: Vec<f32> = updates
                        .iter()
                        .map(|u| u.weight_updates[tensor_idx][[i, j]])
                        .collect();
                    // Sort and trim
                    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let trimmed = &values[trim_count..num_clients - trim_count];
                    // Compute mean of trimmed values
                    result[[i, j]] = trimmed.iter().sum::<f32>() / trimmed.len() as f32;
            aggregated.push(result);
        "TrimmedMean"
/// Krum aggregation for Byzantine robustness
pub struct Krum {
    /// Number of Byzantine clients to tolerate
    num_byzantine: usize,
    /// Whether to use Multi-Krum
    multi_krum: bool,
impl Krum {
    /// Create new Krum aggregator
    pub fn new(_numbyzantine: usize) -> Self {
            num_byzantine,
            multi_krum: false,
    /// Enable Multi-Krum
    pub fn multi_krum(mut self) -> Self {
        self.multi_krum = true;
impl AggregationStrategy for Krum {
        let num_select = if self.multi_krum {
            num_clients - self._num_byzantine
        } else {
            1
        };
        // Compute pairwise distances
        let mut distances = vec![vec![0.0; num_clients]; num_clients];
        for i in 0..num_clients {
            for j in (i + 1)..num_clients {
                let dist = self.compute_distance(&updates[i], &updates[j])?;
                distances[i][j] = dist;
                distances[j][i] = dist;
        // Compute scores (sum of k nearest distances)
        let k = num_clients - self.num_byzantine - 2;
        let mut scores = vec![0.0; num_clients];
            let mut dists: Vec<f32> = distances[i].clone();
            dists.sort_by(|a, b| a.partial_cmp(b).unwrap());
            scores[i] = dists[1..=k].iter().sum(); // Skip self (0 distance)
        // Select clients with lowest scores
        let mut indices: Vec<usize> = (0..num_clients).collect();
        indices.sort_by(|&i, &j| scores[i].partial_cmp(&scores[j]).unwrap());
        let selected = &indices[..num_select];
        // Average selected updates
        let selected_updates: Vec<ClientUpdate> =
            selected.iter().map(|&i| updates[i].clone()).collect();
        let equal_weights = vec![1.0 / num_select as f32; num_select];
        fedavg.aggregate(&selected_updates, &equal_weights)
        if self.multi_krum {
            "Multi-Krum"
            "Krum"
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
        Ok(total_dist.sqrt())
/// Median aggregation
pub struct Median;
impl Median {
    /// Create new median aggregator
        Self
impl AggregationStrategy for Median {
            // For each element, compute median
                    let median = if values.len() % 2 == 0 {
                        (values[values.len() / 2 - 1] + values[values.len() / 2]) / 2.0
                    } else {
                        values[values.len() / 2]
                    };
                    result[[i, j]] = median;
        "Median"
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
                client_id: 1,
                weight_updates: vec![Array2::ones((2, 2)) * 2.0],
                num_samples: 200,
                loss: 0.4,
                accuracy: 0.92,
        ]
    #[test]
    fn test_fedavg() {
        let aggregator = FedAvg::new();
        let updates = create_test_updates();
        let weights = vec![0.5, 0.5];
        let result = aggregator.aggregate(&updates, &weights).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0][[0, 0]], 1.5); // Average of 1 and 2
    fn test_median() {
        let aggregator = Median::new();
        let weights = vec![0.5, 0.5]; // Weights ignored for median
        assert_eq!(result[0][[0, 0]], 1.5); // Median of [1, 2]
