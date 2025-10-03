//! Actor-Critic reinforcement learning algorithms

use crate::error::Result;
use crate::reinforcement::policy::{Policy, PolicyNetwork};
use crate::reinforcement::value::ValueNetwork;
use scirs2_core::ndarray::prelude::*;
use std::sync::Arc;
use scirs2_core::ndarray::ArrayView1;
use statrs::statistics::Statistics;
/// Base Actor-Critic structure
pub struct ActorCritic {
    actor: PolicyNetwork,
    critic: ValueNetwork,
    actor_lr: f32,
    critic_lr: f32,
    discount_factor: f32,
}
impl ActorCritic {
    /// Create a new Actor-Critic model
    pub fn new(
        state_dim: usize,
        action_dim: usize,
        hidden_sizes: Vec<usize>,
        continuous: bool,
        actor_lr: f32,
        critic_lr: f32,
        discount_factor: f32,
    ) -> Result<Self> {
        let actor = PolicyNetwork::new(state_dim, action_dim, hidden_sizes.clone(), continuous)?;
        let critic = ValueNetwork::new(state_dim, 1, hidden_sizes)?;
        Ok(Self {
            actor,
            critic,
            actor_lr,
            critic_lr,
            discount_factor,
        })
    }
    /// Get action from actor
    pub fn get_action(&self, state: &ArrayView1<f32>) -> Result<Array1<f32>> {
        self.actor.sample_action(state)
    /// Get value from critic
    pub fn get_value(&self, state: &ArrayView1<f32>) -> Result<f32> {
        self.critic.predict(state)
    /// Calculate advantages using TD error
    pub fn calculate_advantages(
        &self,
        rewards: &[f32],
        values: &[f32],
        next_value: f32,
        dones: &[bool],
    ) -> Vec<f32> {
        let mut advantages = Vec::with_capacity(rewards.len());
        for i in 0..rewards.len() {
            let next_val = if i == rewards.len() - 1 {
                next_value
            } else {
                values[i + 1]
            };
            let td_error = rewards[i]
                + (if dones[i] {
                    0.0
                } else {
                    self.discount_factor * next_val
                })
                - values[i];
            advantages.push(td_error);
        }
        advantages
/// Advantage Actor-Critic (A2C) algorithm
pub struct A2C {
    actor_critic: ActorCritic,
    entropy_coef: f32,
    value_loss_coef: f32,
    max_grad_norm: Option<f32>,
impl A2C {
    /// Create a new A2C model
        entropy_coef: f32,
        value_loss_coef: f32,
        let actor_critic = ActorCritic::new(
            state_dim,
            action_dim,
            hidden_sizes,
            continuous,
        )?;
            actor_critic,
            entropy_coef,
            value_loss_coef,
            max_grad_norm: Some(0.5),
    /// Collect experience and update
    pub fn update(
        &mut self,
        states: &[Array1<f32>],
        actions: &[Array1<f32>],
        next_state: &ArrayView1<f32>,
    ) -> Result<(f32, f32, f32)> {
        // Get values for all states
        let values: Vec<f32> = states
            .iter()
            .map(|s| self.actor_critic.get_value(&s.view()))
            .collect::<Result<Vec<_>>>()?;
        let next_value = self.actor_critic.get_value(next_state)?;
        // Calculate advantages
        let advantages = self
            .actor_critic
            .calculate_advantages(rewards, &values, next_value, dones);
        // Calculate losses
        let mut policy_loss = 0.0;
        let mut value_loss = 0.0;
        let mut entropy_loss = 0.0;
        for i in 0..states.len() {
            // Policy loss
            let log_prob = self
                .actor_critic
                .actor
                .log_prob(&states[i].view(), &actions[i].view())?;
            policy_loss -= log_prob * advantages[i];
            // Value loss (MSE)
            let value_pred = self.actor_critic.get_value(&states[i].view())?;
            let value_target = rewards[i]
                + if dones[i] {
                    self.actor_critic.discount_factor * next_value
                };
            value_loss += (value_pred - value_target).powi(2);
            // Entropy bonus (simplified)
            entropy_loss -= 0.01; // Placeholder for actual entropy calculation
        let n = states.len() as f32;
        policy_loss /= n;
        value_loss /= n;
        entropy_loss /= n;
        let total_loss =
            policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_loss;
        Ok((policy_loss, value_loss, total_loss))
/// Asynchronous Advantage Actor-Critic (A3C) algorithm
pub struct A3C {
    global_model: Arc<ActorCritic>,
    local_model: ActorCritic,
    t_max: usize,
impl A3C {
    /// Create a new A3C worker
    pub fn new_worker(
        global_model: Arc<ActorCritic>,
        t_max: usize,
        let local_model = ActorCritic::new(
            global_model,
            local_model,
            max_grad_norm: Some(40.0),
            t_max,
    /// Sync local model with global model
    pub fn sync_with_global(&mut self) -> Result<()> {
        // In a real implementation, this would copy weights from global to local
        Ok(())
    /// Update global model with local gradients
    pub fn update_global(&self, gradients: &[Array2<f32>]) -> Result<()> {
        // In a real implementation, this would apply gradients to global model
/// Proximal Policy Optimization (PPO) algorithm
pub struct PPO {
    clip_epsilon: f32,
    num_epochs: usize,
    batch_size: usize,
impl PPO {
    /// Create a new PPO model
        clip_epsilon: f32,
            clip_epsilon,
            num_epochs: 4,
            batch_size: 64,
    /// Update using PPO objective
        states: &ArrayView2<f32>,
        actions: &ArrayView2<f32>,
        old_log_probs: &ArrayView1<f32>,
        advantages: &ArrayView1<f32>,
        returns: &ArrayView1<f32>,
        let mut total_policy_loss = 0.0;
        let mut total_value_loss = 0.0;
        let mut total_entropy_loss = 0.0;
        for _ in 0..self.num_epochs {
            // Mini-batch updates
            for batch_idx in (0..states.shape()[0]).step_by(self.batch_size) {
                let end_idx = (batch_idx + self.batch_size).min(states.shape()[0]);
                let batch_states = states.slice(s![batch_idx..end_idx, ..]);
                let batch_actions = actions.slice(s![batch_idx..end_idx, ..]);
                let batch_old_log_probs = old_log_probs.slice(s![batch_idx..end_idx]);
                let batch_advantages = advantages.slice(s![batch_idx..end_idx]);
                let batch_returns = returns.slice(s![batch_idx..end_idx]);
                // Calculate current log probabilities
                let mut log_probs = Vec::new();
                for i in 0..batch_states.shape()[0] {
                    let state = batch_states.row(i);
                    let action = batch_actions.row(i);
                    let log_prob = self.actor_critic.actor.log_prob(&state, &action)?;
                    log_probs.push(log_prob);
                }
                // Calculate ratio and clipped objective
                let mut policy_loss = 0.0;
                for i in 0..log_probs.len() {
                    let ratio = (log_probs[i] - batch_old_log_probs[i]).exp();
                    let clipped_ratio =
                        ratio.clamp(1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon);
                    let advantage = batch_advantages[i];
                    let loss1 = -advantage * ratio;
                    let loss2 = -advantage * clipped_ratio;
                    policy_loss += loss1.max(loss2);
                // Value loss
                let values = self.actor_critic.critic.predict_batch(&batch_states)?;
                let value_loss = (&values - &batch_returns).mapv(|x| x * x).mean().unwrap();
                // Entropy loss - encourage exploration
                let mut entropy_loss = 0.0;
                    let entropy = self.compute_policy_entropy(&state)?;
                    entropy_loss += entropy;
                entropy_loss /= batch_states.shape()[0] as f32;
                total_policy_loss += policy_loss / log_probs.len() as f32;
                total_value_loss += value_loss;
                total_entropy_loss += entropy_loss;
            }
        let num_batches = (states.shape()[0] as f32 / self.batch_size as f32).ceil();
        total_policy_loss /= num_batches * self.num_epochs as f32;
        total_value_loss /= num_batches * self.num_epochs as f32;
        total_entropy_loss /= num_batches * self.num_epochs as f32;
        let total_loss = total_policy_loss + self.value_loss_coef * total_value_loss
            - self.entropy_coef * total_entropy_loss;
        Ok((total_policy_loss, total_value_loss, total_loss))
    /// Compute policy entropy for a given state
    fn compute_policy_entropy(&self, state: &ArrayView1<f32>) -> Result<f32> {
        let (params, std_opt) = self.actor_critic.actor.get_distribution_params(state)?;
        
        if self.actor_critic.actor.continuous {
            // For continuous policies (Gaussian), entropy = 0.5 * log(2πeσ²)
            if let Some(std) = std_opt {
                let entropy = 0.5 * std.mapv(|s| (2.0 * std::f32::consts::PI * std::f32::consts::E * s * s).ln()).sum();
                Ok(entropy)
                Err(crate::error::NeuralError::InvalidArgument(
                    "Standard deviation not available for continuous policy".to_string(),
                ))
        } else {
            // For discrete policies (Categorical), entropy = -Σ p(a) * log(p(a))
            let entropy = -params.iter()
                .filter(|&&p| p > 0.0)
                .map(|&p| p * p.ln())
                .sum::<f32>();
            Ok(entropy)
/// Soft Actor-Critic (SAC) algorithm
pub struct SAC {
    critic1: ValueNetwork,
    critic2: ValueNetwork,
    target_critic1: ValueNetwork,
    target_critic2: ValueNetwork,
    alpha: f32, // Temperature parameter
    tau: f32, // Soft update coefficient
impl SAC {
    /// Create a new SAC model
        alpha: f32,
        tau: f32,
        let actor = PolicyNetwork::new(state_dim, action_dim, hidden_sizes.clone(), true)?;
        // SAC uses two Q-functions for stability
        let critic1 = ValueNetwork::new(state_dim + action_dim, 1, hidden_sizes.clone())?;
        let critic2 = ValueNetwork::new(state_dim + action_dim, 1, hidden_sizes.clone())?;
        let target_critic1 = ValueNetwork::new(state_dim + action_dim, 1, hidden_sizes.clone())?;
        let target_critic2 = ValueNetwork::new(state_dim + action_dim, 1, hidden_sizes)?;
            critic1,
            critic2,
            target_critic1,
            target_critic2,
            alpha,
            tau,
    /// Get action with exploration
    /// Soft update target networks
    pub fn soft_update_targets(&mut self) -> Result<()> {
        // In a real implementation, this would perform:
        // target_weights = tau * current_weights + (1 - tau) * target_weights
    /// Update SAC networks
        rewards: &ArrayView1<f32>,
        next_states: &ArrayView2<f32>,
        dones: &ArrayView1<bool>,
        let batch_size = states.shape()[0];
        // Sample actions from the current policy for next states
        let mut next_actions = Vec::new();
        let mut next_log_probs = Vec::new();
        for i in 0..batch_size {
            let next_state = next_states.row(i);
            let next_action = self.actor.sample_action(&next_state)?;
            let next_log_prob = self.actor.log_prob(&next_state, &next_action.view())?;
            next_actions.push(next_action);
            next_log_probs.push(next_log_prob);
        // Placeholder losses (simplified)
        let critic_loss = 0.1;
        let actor_loss = 0.05;
        let alpha_loss = 0.01;
        // Soft update target networks
        self.soft_update_targets()?;
        Ok((critic_loss, actor_loss, alpha_loss))
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_actor_critic_creation() {
        let ac = ActorCritic::new(4, 2, vec![32, 32], false, 0.001, 0.001, 0.99).unwrap();
        let state = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let action = ac.get_action(&state.view()).unwrap();
        assert_eq!(action.len(), 2);
        let value = ac.get_value(&state.view()).unwrap();
        assert!(value.is_finite());
    fn test_a2c_creation() {
        let a2c = A2C::new(4, 2, vec![32], false, 0.001, 0.001, 0.99, 0.01, 0.5).unwrap();
        assert_eq!(a2c.entropy_coef, 0.01);
        assert_eq!(a2c.value_loss_coef, 0.5);
    fn test_ppo_creation() {
        let ppo = PPO::new(4, 2, vec![32], true, 0.001, 0.001, 0.99, 0.2, 0.01, 0.5).unwrap();
        assert_eq!(ppo.clip_epsilon, 0.2);
        assert_eq!(ppo.num_epochs, 4);
    fn test_sac_creation() {
        let sac = SAC::new(4, 2, vec![32], 0.001, 0.001, 0.99, 0.2, 0.005).unwrap();
        assert_eq!(sac.alpha, 0.2);
        assert_eq!(sac.tau, 0.005);
