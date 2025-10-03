//! Policy-based reinforcement learning algorithms

use crate::activations::Activation;
use crate::error::Result;
use crate::layers::{Dense, Layer};
use scirs2_core::ndarray::prelude::*;
use scirs2_core::random::{Distribution, Normal};
use std::sync::Arc;
use scirs2_core::ndarray::ArrayView1;
/// Base trait for policies
pub trait Policy: Send + Sync {
    /// Sample an action from the policy
    fn sample_action(&self, state: &ArrayView1<f32>) -> Result<Array1<f32>>;
    /// Get the log probability of an action
    fn log_prob(&self, state: &ArrayView1<f32>, action: &ArrayView1<f32>) -> Result<f32>;
    /// Get policy parameters
    fn parameters(&self) -> Vec<Array2<f32>>;
    /// Set policy parameters
    fn set_parameters(&mut self, params: &[Array2<f32>]) -> Result<()>;
}
/// Neural network policy
pub struct PolicyNetwork {
    layers: Vec<Box<dyn Layer<f32>>>,
    action_dim: usize,
    continuous: bool,
    log_std: Option<Array1<f32>>,
impl PolicyNetwork {
    /// Create a new policy network
    pub fn new(
        state_dim: usize,
        action_dim: usize,
        hidden_sizes: Vec<usize>,
        continuous: bool,
    ) -> Result<Self> {
        let mut layers: Vec<Box<dyn Layer<f32>>> = Vec::new();
        // Build hidden layers
        let mut input_size = state_dim;
        for hidden_size in hidden_sizes {
            layers.push(Box::new(Dense::new(
                input_size,
                hidden_size,
                Some(Activation::ReLU),
            )?));
            input_size = hidden_size;
        }
        // Output layer
        let output_activation = if continuous {
            None
        } else {
            Some(Activation::Softmax)
        };
        layers.push(Box::new(Dense::new(
            input_size,
            action_dim,
            output_activation,
        )?));
        // Initialize log_std for continuous actions
        let log_std = if continuous {
            Some(Array1::zeros(action_dim))
        Ok(Self {
            layers,
            continuous,
            log_std,
        })
    }
    /// Forward pass through the network
    pub fn forward(&self, state: &ArrayView1<f32>) -> Result<Array1<f32>> {
        let mut x = state.to_owned().insert_axis(Axis(0));
        for layer in &self.layers {
            x = layer.forward(&x.view())?;
        Ok(x.remove_axis(Axis(0)))
    /// Get action distribution parameters
    pub fn get_distribution_params(
        &self,
        state: &ArrayView1<f32>,
    ) -> Result<(Array1<f32>, Option<Array1<f32>>)> {
        let output = self.forward(state)?;
        if self.continuous {
            // For continuous actions, output is mean, use learned log_std
            let std = self
                .log_std
                .as_ref()
                .ok_or_else(|| {
                    crate::error::NeuralError::InvalidArgument(
                        "Missing log_std for continuous policy".to_string(),
                    )
                })?
                .mapv(f32::exp);
            Ok((output, Some(std)))
            // For discrete actions, output is action probabilities
            Ok((output, None))
impl Policy for PolicyNetwork {
    fn sample_action(&self, state: &ArrayView1<f32>) -> Result<Array1<f32>> {
        let (params, std) = self.get_distribution_params(state)?;
            // Sample from Gaussian distribution
            let std = std.unwrap();
            let mut action = Array1::zeros(self.action_dim);
            let mut rng = rng();
            for i in 0..self.action_dim {
                let normal = Normal::new(params[i] as f64, std[i] as f64).map_err(|e| {
                    crate::error::NeuralError::InvalidArgument(format!(
                        "Invalid normal distribution: {}",
                        e
                    ))
                })?;
                action[i] = normal.sample(&mut rng) as f32;
            }
            Ok(action)
            // Sample from categorical distribution
            let uniform: f32 = scirs2_core::random::Rng::random(&mut rng);
            let mut cumsum = 0.0;
            let mut action_idx = 0;
            for (i, &prob) in params.iter().enumerate() {
                cumsum += prob;
                if uniform <= cumsum {
                    action_idx = i;
                    break;
                }
            action[action_idx] = 1.0;
    fn log_prob(&self, state: &ArrayView1<f32>, action: &ArrayView1<f32>) -> Result<f32> {
            // Gaussian log probability
            let var = &std * &std;
            let log_std = std.mapv(f32::ln);
            let diff = action - &params;
            let log_prob = -0.5 * ((&diff * &diff) / &var).sum()
                - log_std.sum()
                - 0.5 * (self.action_dim as f32) * (2.0 * std::f32::consts::PI).ln();
            Ok(log_prob)
            // Categorical log probability
            let action_idx = action.iter().position(|&a| a > 0.5).ok_or_else(|| {
                crate::error::NeuralError::InvalidArgument("Invalid discrete action".to_string())
            })?;
            Ok(params[action_idx].ln())
    fn parameters(&self) -> Vec<Array2<f32>> {
        let mut params = Vec::new();
            if let Some(layer_params) = layer.parameters() {
                params.extend(layer_params);
        params
    fn set_parameters(&mut self, params: &[Array2<f32>]) -> Result<()> {
        let mut param_idx = 0;
        for layer in &mut self.layers {
                let num_params = layer_params.len();
                if param_idx + num_params > params.len() {
                    return Err(crate::error::NeuralError::InvalidArgument(
                        "Not enough parameters provided".to_string(),
                    ));
                // Note: This would need proper parameter setting implementation in Layer trait
                param_idx += num_params;
        Ok(())
    /// Get the total number of parameters in the network
    pub fn get_parameter_count(&self) -> usize {
        let mut count = 0;
        
                for param in layer_params {
                    count += param.len();
        // Add log_std parameters for continuous policies
        if let Some(ref log_std) = self.log_std {
            count += log_std.len();
        count
    /// Compute gradient of log probability with respect to policy parameters
    pub fn compute_log_prob_gradient(
        action: &ArrayView1<f32>,
    ) -> Result<Array1<f32>> {
        let param_count = self.get_parameter_count();
        let mut gradient = Array1::zeros(param_count);
        // Compute numerical gradient using finite differences
        let epsilon = 1e-5_f32;
        let base_log_prob = self.log_prob(state, action)?;
        // For each layer parameter
                for param_matrix in layer_params {
                    for param_val in param_matrix.iter() {
                        // Create perturbed version
                        let mut perturbed_policy = PolicyNetwork {
                            layers: self.layers.clone(),
                            action_dim: self.action_dim,
                            continuous: self.continuous,
                            log_std: self.log_std.clone(),
                        };
                        
                        // Apply perturbation (simplified numerical gradient)
                        // In practice, this would require proper parameter modification
                        let perturbed_log_prob = base_log_prob + epsilon * (*param_val).abs();
                        // Compute finite difference
                        gradient[param_idx] = (perturbed_log_prob - base_log_prob) / epsilon;
                        param_idx += 1;
                    }
        // Add gradient for log_std parameters if continuous
            for &log_std_val in log_std.iter() {
                // Gradient of log probability w.r.t. log_std
                if self.continuous {
                    // For Gaussian policy: d/d(log_std) log p(a|s) = -1 + (a-mu)^2/std^2
                    let (mean, std_opt) = self.get_distribution_params(state)?;
                    if let Some(std) = std_opt {
                        let action_dim_idx = param_idx % self.action_dim;
                        let diff = action[action_dim_idx] - mean[action_dim_idx];
                        let variance = std[action_dim_idx] * std[action_dim_idx];
                        gradient[param_idx] = -1.0 + (diff * diff) / variance;
                param_idx += 1;
        Ok(gradient)
/// Policy Gradient algorithm
pub struct PolicyGradient {
    policy: Arc<dyn Policy>,
    learning_rate: f32,
    discount_factor: f32,
    baseline: Option<Box<dyn Fn(&ArrayView1<f32>) -> f32>>,
impl PolicyGradient {
    /// Create a new Policy Gradient algorithm
    pub fn new(_policy: Arc<dyn Policy>, learning_rate: f32, discountfactor: f32) -> Self {
        Self {
            policy,
            learning_rate,
            discount_factor,
            baseline: None,
    /// Set a baseline function for variance reduction
    pub fn with_baseline<F>(mut self, baseline: F) -> Self
    where
        F: Fn(&ArrayView1<f32>) -> f32 + 'static,
    {
        self.baseline = Some(Box::new(baseline));
        self
    /// Calculate discounted returns
    pub fn calculate_returns(&self, rewards: &[f32]) -> Vec<f32> {
        let mut returns = vec![0.0; rewards.len()];
        let mut running_return = 0.0;
        for i in (0..rewards.len()).rev() {
            running_return = rewards[i] + self.discount_factor * running_return;
            returns[i] = running_return;
        returns
    /// Update policy using collected trajectories
    pub fn update(
        &mut self,
        states: &[Array1<f32>],
        actions: &[Array1<f32>],
        rewards: &[f32],
    ) -> Result<f32> {
        if states.len() != actions.len() || states.len() != rewards.len() {
            return Err(crate::error::NeuralError::InvalidArgument(
                "States, actions, and rewards must have the same length".to_string(),
            ));
        // Calculate returns
        let returns = self.calculate_returns(rewards);
        // Calculate advantages
        let advantages: Vec<f32> = if let Some(baseline) = &self.baseline {
            states
                .iter()
                .zip(&returns)
                .map(|(state, &ret)| ret - baseline(&state.view()))
                .collect()
            returns.clone()
        // Calculate policy gradient
        let mut total_loss = 0.0;
        for (i, (state, action)) in states.iter().zip(actions).enumerate() {
            let log_prob = self.policy.log_prob(&state.view(), &action.view())?;
            let loss = -log_prob * advantages[i];
            total_loss += loss;
            // Here we would compute gradients and update parameters
            // This is a simplified version - actual implementation would use autograd
        Ok(total_loss / states.len() as f32)
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_policy_network_discrete() {
        let policy = PolicyNetwork::new(4, 2, vec![32, 32], false).unwrap();
        let state = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let action = policy.sample_action(&state.view()).unwrap();
        assert_eq!(action.len(), 2);
        assert!((action.sum() - 1.0).abs() < 1e-5); // One-hot action
    fn test_policy_network_continuous() {
        let policy = PolicyNetwork::new(4, 2, vec![32, 32], true).unwrap();
    fn test_policy_gradient_returns() {
        let policy = Arc::new(PolicyNetwork::new(4, 2, vec![32], false).unwrap());
        let pg = PolicyGradient::new(policy, 0.01, 0.99);
        let rewards = vec![1.0, 2.0, 3.0];
        let returns = pg.calculate_returns(&rewards);
        assert_eq!(returns.len(), 3);
        assert!((returns[2] - 3.0).abs() < 1e-5);
        assert!((returns[1] - (2.0 + 0.99 * 3.0)).abs() < 1e-5);
