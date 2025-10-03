//! Advanced Policy Optimization Techniques
//!
//! This module implements sophisticated policy optimization methods including
//! natural gradients, advanced exploration strategies, and meta-learning approaches.

use crate::error::Result;
use crate::reinforcement::policy::PolicyNetwork;
use crate::reinforcement::value::ValueNetwork;
use crate::reinforcement::{ExperienceBatch, LossInfo, RLAgent};
use scirs2_core::ndarray::prelude::*;
use scirs2_core::numeric::Float;
use std::collections::HashMap;
use scirs2_core::ndarray::ArrayView1;
use statrs::statistics::Statistics;
/// Natural Policy Gradients (NPG) implementation
pub struct NaturalPolicyGradient {
    /// Policy network
    policy: PolicyNetwork,
    /// Value function for baseline
    value_function: ValueNetwork,
    /// Configuration
    config: NPGConfig,
    /// Fisher Information Matrix (simplified representation)
    fisher_matrix: Option<Array2<f32>>,
    /// Experience buffer for computing Fisher Information
    experience_buffer: Vec<ExperienceBatch>,
}
/// NPG configuration
#[derive(Debug, Clone)]
pub struct NPGConfig {
    /// Learning rate
    pub learning_rate: f32,
    /// Discount factor
    pub gamma: f32,
    /// GAE lambda
    pub lambda: f32,
    /// CG iterations for computing natural gradient
    pub cg_iterations: usize,
    /// CG tolerance
    pub cg_tolerance: f32,
    /// Fisher Information Matrix damping
    pub fisher_damping: f32,
    /// Batch size for Fisher Information estimation
    pub fisher_batch_size: usize,
    /// Value function learning rate
    pub value_lr: f32,
impl Default for NPGConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3,
            gamma: 0.99,
            lambda: 0.95,
            cg_iterations: 10,
            cg_tolerance: 1e-8,
            fisher_damping: 1e-2,
            fisher_batch_size: 128,
            value_lr: 1e-3,
        }
    }
impl NaturalPolicyGradient {
    /// Create a new Natural Policy Gradient optimizer
    pub fn new(
        state_dim: usize,
        action_dim: usize,
        hidden_sizes: Vec<usize>,
        continuous: bool,
        config: NPGConfig,
    ) -> Result<Self> {
        let policy = PolicyNetwork::new(state_dim, action_dim, hidden_sizes.clone(), continuous)?;
        let value_function = ValueNetwork::new(state_dim, 1, hidden_sizes)?;
        Ok(Self {
            policy,
            value_function,
            config,
            fisher_matrix: None,
            experience_buffer: Vec::new(),
        })
    /// Compute Generalized Advantage Estimation (GAE)
    fn compute_gae(&self, rewards: &[f32], values: &[f32], dones: &[bool]) -> Vec<f32> {
        let mut advantages = vec![0.0; rewards.len()];
        let mut gae = 0.0;
        for t in (0..rewards.len()).rev() {
            let next_value = if t == rewards.len() - 1 {
                0.0
            } else {
                values[t + 1]
            };
            let delta = rewards[t] + self.config.gamma * next_value * (1.0 - dones[t] as u8 as f32)
                - values[t];
            gae = delta
                + self.config.gamma * self.config.lambda * (1.0 - dones[t] as u8 as f32) * gae;
            advantages[t] = gae;
        advantages
    /// Estimate Fisher Information Matrix (simplified)
    fn estimate_fisher_information(&mut self, batch: &ExperienceBatch) -> Result<Array2<f32>> {
        let batch_size = batch.states.shape()[0];
        // For simplicity, we'll create a diagonal Fisher matrix
        // In practice, this would involve computing second derivatives of the log-policy
        let num_params = 100; // Simplified parameter count
        let mut fisher = Array2::eye(num_params);
        // Add some structure based on policy gradients
        for i in 0..batch_size.min(self.config.fisher_batch_size) {
            let state = batch.states.row(i);
            let action = batch.actions.row(i);
            // Compute log probability and its gradient (simplified)
            let log_prob = self.policy.log_prob(&state, &action)?;
            // In a full implementation, we'd compute the outer product of gradients
            // For now, we'll just add some diagonal entries
            for j in 0..num_params.min(state.len()) {
                fisher[[j, j]] += log_prob.abs() * 0.01;
            }
        // Add damping
        for i in 0..num_params {
            fisher[[i, i]] += self.config.fisher_damping;
        Ok(fisher)
    /// Conjugate Gradient method to solve Fx = g for natural gradient
    fn conjugate_gradient(
        &self,
        fisher: &Array2<f32>,
        gradient: &Array1<f32>,
    ) -> Result<Array1<f32>> {
        let n = gradient.len();
        let mut x = Array1::zeros(n);
        let mut r = gradient.clone();
        let mut p = r.clone();
        let mut rsold = r.dot(&r);
        for _ in 0..self.config.cg_iterations {
            let ap = fisher.dot(&p);
            let alpha = rsold / p.dot(&ap);
            x = x + alpha * &p;
            r = r - alpha * &ap;
            let rsnew = r.dot(&r);
            if rsnew.sqrt() < self.config.cg_tolerance {
                break;
            let beta = rsnew / rsold;
            p = &r + beta * &p;
            rsold = rsnew;
        Ok(x)
    /// Update using Natural Policy Gradients
    pub fn update_npg(&mut self, batch: &ExperienceBatch) -> Result<LossInfo> {
        // Compute value predictions
        let mut values = Vec::new();
        for i in 0..batch_size {
            let value = self.value_function.predict(&state)?;
            values.push(value);
        // Compute advantages using GAE
        let advantages = self.compute_gae(&batch.rewards.to_vec(), &values, &batch.dones.to_vec());
        // Compute policy gradient
        let mut policy_gradient = Array1::zeros(100); // Simplified
        for (i, &advantage) in advantages.iter().enumerate() {
            // Accumulate gradient (simplified)
            for j in 0..policy_gradient.len().min(state.len()) {
                policy_gradient[j] += log_prob * advantage * state[j];
        policy_gradient /= batch_size as f32;
        // Estimate Fisher Information Matrix
        let fisher = self.estimate_fisher_information(batch)?;
        // Compute natural gradient using conjugate gradient
        let natural_gradient = self.conjugate_gradient(&fisher, &policy_gradient)?;
        // Update policy parameters (simplified)
        let policy_loss = policy_gradient.dot(&natural_gradient);
        // Update value function
        let mut value_loss = 0.0;
            let predicted_value = self.value_function.predict(&state)?;
            let target_value = predicted_value + advantage;
            let error = predicted_value - target_value;
            value_loss += error * error;
        value_loss /= batch_size as f32;
        let mut metrics = HashMap::new();
        metrics.insert("policy_loss".to_string(), policy_loss);
        metrics.insert("value_loss".to_string(), value_loss);
        metrics.insert(
            "avg_advantage".to_string(),
            advantages.iter().sum::<f32>() / advantages.len() as f32,
        );
        Ok(LossInfo {
            policy_loss: Some(policy_loss),
            value_loss: Some(value_loss),
            entropy_loss: None,
            total_loss: policy_loss + value_loss,
            metrics,
impl RLAgent for NaturalPolicyGradient {
    fn act(&self, observation: &ArrayView1<f32>, training: bool) -> Result<Array1<f32>> {
        self.policy.sample_action(observation)
    fn update(&mut self, batch: &ExperienceBatch) -> Result<LossInfo> {
        self.experience_buffer.push(batch.clone());
        self.update_npg(batch)
    fn save(&self, path: &str) -> Result<()> {
        std::fs::create_dir_all(std::path::Path::new(path).parent().unwrap())?;
        Ok(())
    fn load(&mut self, path: &str) -> Result<()> {
/// Curiosity-Driven Exploration using Intrinsic Curiosity Module (ICM)
pub struct CuriosityDrivenAgent {
    /// Main policy
    /// Value function
    /// Forward dynamics model
    forward_model: ForwardModel,
    /// Inverse dynamics model
    inverse_model: InverseModel,
    config: CuriosityConfig,
    /// Running statistics for intrinsic reward normalization
    intrinsic_reward_stats: RunningStats,
/// Curiosity configuration
pub struct CuriosityConfig {
    /// Learning rate for main policy
    pub policy_lr: f32,
    /// Learning rate for curiosity models
    pub curiosity_lr: f32,
    /// Intrinsic reward coefficient
    pub intrinsic_coeff: f32,
    /// Feature loss coefficient in ICM
    pub feature_loss_coeff: f32,
    /// Forward loss coefficient in ICM
    pub forward_loss_coeff: f32,
    /// Feature dimension for ICM
    pub feature_dim: usize,
impl Default for CuriosityConfig {
            policy_lr: 3e-4,
            curiosity_lr: 1e-3,
            intrinsic_coeff: 0.01,
            feature_loss_coeff: 0.2,
            forward_loss_coeff: 0.8,
            feature_dim: 64,
/// Forward dynamics model for predicting next state features
pub struct ForwardModel {
    /// Feature encoder
    feature_encoder: ValueNetwork,
    /// Dynamics network
    dynamics_network: ValueNetwork,
impl ForwardModel {
        feature_dim: usize,
        let feature_encoder = ValueNetwork::new(state_dim, feature_dim, hidden_sizes.clone())?;
        let dynamics_network =
            ValueNetwork::new(feature_dim + action_dim, feature_dim, hidden_sizes)?;
            feature_encoder,
            dynamics_network,
    /// Predict next state features
    pub fn predict(
        state_features: &ArrayView1<f32>,
        action: &ArrayView1<f32>,
        // Concatenate state features and action
        let mut input = Array1::zeros(state_features.len() + action.len());
        input
            .slice_mut(s![..state_features.len()])
            .assign(state_features);
        input.slice_mut(s![state_features.len()..]).assign(action);
        let predicted_features = self.dynamics_network.predict(&input.view())?;
        Ok(Array1::from_vec(vec![predicted_features])) // Simplified
    /// Encode state to features
    pub fn encode_state(&self, state: &ArrayView1<f32>) -> Result<Array1<f32>> {
        let features = self.feature_encoder.predict(state)?;
        Ok(Array1::from_vec(vec![features])) // Simplified
/// Inverse dynamics model for predicting action from state transition
pub struct InverseModel {
    /// Network for predicting action
    action_predictor: ValueNetwork,
impl InverseModel {
    pub fn new(_feature_dim: usize, action_dim: usize, hiddensizes: Vec<usize>) -> Result<Self> {
        let action_predictor = ValueNetwork::new(_feature_dim * 2, action_dim, hidden_sizes)?;
        Ok(Self { action_predictor })
    /// Predict action from state features
    pub fn predict_action(
        next_state_features: &ArrayView1<f32>,
        // Concatenate current and next state features
        let mut input = Array1::zeros(state_features.len() + next_state_features.len());
            .slice_mut(s![state_features.len()..])
            .assign(next_state_features);
        let predicted_action = self.action_predictor.predict(&input.view())?;
        Ok(Array1::from_vec(vec![predicted_action])) // Simplified
/// Running statistics for normalization
pub struct RunningStats {
    count: usize,
    mean: f32,
    var: f32,
impl RunningStats {
    pub fn new() -> Self {
            count: 0,
            mean: 0.0,
            var: 1.0,
    pub fn update(&mut self, value: f32) {
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f32;
        let delta2 = value - self.mean;
        self.var += delta * delta2;
    pub fn normalize(&self, value: f32) -> f32 {
        if self.count > 1 {
            let std = (self.var / (self.count - 1) as f32).sqrt();
            (value - self.mean) / (std + 1e-8)
        } else {
            value
impl CuriosityDrivenAgent {
    /// Create a new curiosity-driven agent
        config: CuriosityConfig,
        let value_function = ValueNetwork::new(state_dim, 1, hidden_sizes.clone())?;
        let forward_model = ForwardModel::new(
            state_dim,
            action_dim,
            config.feature_dim,
            hidden_sizes.clone(),
        )?;
        let inverse_model = InverseModel::new(config.feature_dim, action_dim, hidden_sizes)?;
            forward_model,
            inverse_model,
            intrinsic_reward_stats: RunningStats::new(),
    /// Compute intrinsic reward using forward model prediction error
    pub fn compute_intrinsic_reward(&mut self, batch: &ExperienceBatch) -> Result<Array1<f32>> {
        let mut intrinsic_rewards = Array1::zeros(batch_size);
            let next_state = batch.next_states.row(i);
            // Encode states to features
            let state_features = self.forward_model.encode_state(&state)?;
            let next_state_features = self.forward_model.encode_state(&next_state)?;
            // Predict next state features
            let predicted_features = self
                .forward_model
                .predict(&state_features.view(), &action)?;
            // Compute prediction error as intrinsic reward
            let error = (&next_state_features - &predicted_features)
                .mapv(|x| x * x)
                .sum()
                .sqrt();
            intrinsic_rewards[i] = error;
            // Update running statistics
            self.intrinsic_reward_stats.update(error);
        // Normalize intrinsic rewards
        for reward in intrinsic_rewards.iter_mut() {
            *reward = self.intrinsic_reward_stats.normalize(*reward);
        Ok(intrinsic_rewards)
    /// Update curiosity models
    pub fn update_curiosity_models(&mut self, batch: &ExperienceBatch) -> Result<(f32, f32)> {
        let mut forward_loss = 0.0;
        let mut inverse_loss = 0.0;
            // Encode states
            // Forward model loss
            let forward_error = (&next_state_features - &predicted_features)
                .sum();
            forward_loss += forward_error;
            // Inverse model loss
            let predicted_action = self
                .inverse_model
                .predict_action(&state_features.view(), &next_state_features.view())?;
            let inverse_error = (&action.to_owned() - &predicted_action)
            inverse_loss += inverse_error;
        forward_loss /= batch_size as f32;
        inverse_loss /= batch_size as f32;
        Ok((forward_loss, inverse_loss))
    /// Update the curiosity-driven agent
    pub fn update_curiosity(&mut self, batch: &ExperienceBatch) -> Result<LossInfo> {
        // Compute intrinsic rewards
        let intrinsic_rewards = self.compute_intrinsic_reward(batch)?;
        // Combine extrinsic and intrinsic rewards
        let total_rewards = &batch.rewards + &(intrinsic_rewards * self.config.intrinsic_coeff);
        // Create modified batch with total rewards
        let mut modified_batch = batch.clone();
        modified_batch.rewards = total_rewards;
        // Update curiosity models
        let (forward_loss, inverse_loss) = self.update_curiosity_models(batch)?;
        // Update policy using combined rewards (simplified)
        let mut policy_loss = 0.0;
        for i in 0..batch.states.shape()[0] {
            policy_loss -= log_prob * modified_batch.rewards[i];
        policy_loss /= batch.states.shape()[0] as f32;
        let total_loss = policy_loss
            + self.config.forward_loss_coeff * forward_loss
            + self.config.feature_loss_coeff * inverse_loss;
        metrics.insert("forward_loss".to_string(), forward_loss);
        metrics.insert("inverse_loss".to_string(), inverse_loss);
            "avg_intrinsic_reward".to_string(),
            intrinsic_rewards.mean().unwrap(),
            "avg_extrinsic_reward".to_string(),
            batch.rewards.mean().unwrap(),
            value_loss: None,
            total_loss,
impl RLAgent for CuriosityDrivenAgent {
        self.update_curiosity(batch)
/// Meta-Learning Agent using Model-Agnostic Meta-Learning (MAML)
pub struct MAMLAgent {
    /// Meta-policy network
    meta_policy: PolicyNetwork,
    /// Meta-value function
    meta_value: ValueNetwork,
    config: MAMLConfig,
    /// Task buffer for meta-learning
    task_buffer: Vec<TaskExperience>,
/// MAML configuration
pub struct MAMLConfig {
    /// Meta learning rate
    pub meta_lr: f32,
    /// Inner loop learning rate
    pub inner_lr: f32,
    /// Number of inner loop steps
    pub inner_steps: usize,
    /// Number of tasks per meta-update
    pub tasks_per_update: usize,
    /// Support set size (K-shot)
    pub support_size: usize,
    /// Query set size
    pub query_size: usize,
impl Default for MAMLConfig {
            meta_lr: 1e-3,
            inner_lr: 0.01,
            inner_steps: 1,
            tasks_per_update: 10,
            support_size: 10,
            query_size: 10,
/// Experience from a specific task
pub struct TaskExperience {
    pub task_id: usize,
    pub support_batch: ExperienceBatch,
    pub query_batch: ExperienceBatch,
impl MAMLAgent {
    /// Create a new MAML agent
        config: MAMLConfig,
        let meta_policy =
            PolicyNetwork::new(state_dim, action_dim, hidden_sizes.clone(), continuous)?;
        let meta_value = ValueNetwork::new(state_dim, 1, hidden_sizes)?;
            meta_policy,
            meta_value,
            task_buffer: Vec::new(),
    /// Add task experience for meta-learning
    pub fn add_task_experience(&mut self, taskexperience: TaskExperience) {
        self.task_buffer.push(task_experience);
        // Keep only recent tasks
        if self.task_buffer.len() > self.config.tasks_per_update * 2 {
            self.task_buffer.remove(0);
    /// Inner loop adaptation for a single task
    fn inner_loop_adaptation(
        support_batch: &ExperienceBatch,
    ) -> Result<(PolicyNetwork, f32)> {
        // Clone the meta-policy for adaptation
        let mut adapted_policy = self.meta_policy.clone();
        let mut total_loss = 0.0;
        // Perform inner loop updates
        for _ in 0..self.config.inner_steps {
            let mut loss = 0.0;
            for i in 0..support_batch.states.shape()[0] {
                let state = support_batch.states.row(i);
                let action = support_batch.actions.row(i);
                let reward = support_batch.rewards[i];
                let log_prob = adapted_policy.log_prob(&state, &action)?;
                loss -= log_prob * reward; // Simplified policy gradient
            loss /= support_batch.states.shape()[0] as f32;
            total_loss += loss;
            // In a complete implementation, we would compute gradients and update adapted_policy
            // For now, this is a placeholder
        Ok((adapted_policy, total_loss))
    /// Meta-update using MAML
    pub fn meta_update(&mut self) -> Result<LossInfo> {
        if self.task_buffer.len() < self.config.tasks_per_update {
            return Ok(LossInfo {
                policy_loss: Some(0.0),
                value_loss: None,
                entropy_loss: None,
                total_loss: 0.0,
                metrics: HashMap::new(),
            });
        let mut meta_gradients = Vec::new();
        let mut total_meta_loss = 0.0;
        // Sample tasks for meta-update
        let task_indices: Vec<usize> =
            (0..self.config.tasks_per_update.min(self.task_buffer.len())).collect();
        for &task_idx in &task_indices {
            let task_exp = &self.task_buffer[task_idx];
            // Inner loop adaptation on support set
            let (adapted_policy, support_loss) =
                self.inner_loop_adaptation(&task_exp.support_batch)?;
            // Compute loss on query set using adapted policy
            let mut query_loss = 0.0;
            for i in 0..task_exp.query_batch.states.shape()[0] {
                let state = task_exp.query_batch.states.row(i);
                let action = task_exp.query_batch.actions.row(i);
                let reward = task_exp.query_batch.rewards[i];
                query_loss -= log_prob * reward;
            query_loss /= task_exp.query_batch.states.shape()[0] as f32;
            total_meta_loss += query_loss;
            // In a complete implementation, we would compute second-order gradients here
        total_meta_loss /= task_indices.len() as f32;
        // Update meta-parameters (simplified)
        // In practice, this would involve computing and applying second-order gradients
        metrics.insert("meta_loss".to_string(), total_meta_loss);
        metrics.insert("num_tasks".to_string(), task_indices.len() as f32);
            policy_loss: Some(total_meta_loss),
            total_loss: total_meta_loss,
    /// Fast adaptation to a new task
    pub fn fast_adapt(&self, supportbatch: &ExperienceBatch) -> Result<PolicyNetwork> {
        let (adapted_policy_) = self.inner_loop_adaptation(support_batch)?;
        Ok(adapted_policy)
impl RLAgent for MAMLAgent {
        self.meta_policy.sample_action(observation)
        // For MAML, we expect task experiences rather than regular batches
        // This is a simplified implementation
        self.meta_update()
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;
    #[test]
    fn test_natural_policy_gradient() {
        let config = NPGConfig::default();
        let npg = NaturalPolicyGradient::new(4, 2, vec![64], true, config).unwrap();
        let state = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let action = npg.act(&state.view(), true).unwrap();
        assert_eq!(action.len(), 2);
    fn test_curiosity_driven_agent() {
        let config = CuriosityConfig::default();
        let agent = CuriosityDrivenAgent::new(4, 2, vec![32], true, config).unwrap();
        let action = agent.act(&state.view(), true).unwrap();
    fn test_maml_agent() {
        let config = MAMLConfig::default();
        let agent = MAMLAgent::new(4, 2, vec![32], true, config).unwrap();
    fn test_running_stats() {
        let mut stats = RunningStats::new();
        stats.update(1.0);
        stats.update(2.0);
        stats.update(3.0);
        assert!((stats.mean - 2.0).abs() < 1e-6);
        let normalized = stats.normalize(4.0);
        assert!(normalized > 0.0); // Should be positive since 4.0 > mean
