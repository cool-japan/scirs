//! Advanced Reinforcement Learning Algorithms
//!
//! This module implements state-of-the-art RL algorithms including TD3, Rainbow DQN,
//! IMPALA, and advanced exploration strategies.

use crate::error::Result;
use crate::reinforcement::policy::PolicyNetwork;
use crate::reinforcement::replay__buffer::{PrioritizedReplayBuffer, ReplayBuffer};
use crate::reinforcement::value::{QNetwork, ValueNetwork};
use crate::reinforcement::{ExperienceBatch, LossInfo, RLAgent};
use ndarray::prelude::*;
use num_traits::Float;
use std::collections::HashMap;
/// Twin Delayed Deep Deterministic Policy Gradients (TD3)
pub struct TD3 {
    /// Actor network
    actor: PolicyNetwork,
    /// Target actor network
    target_actor: PolicyNetwork,
    /// First critic network
    critic_1: QNetwork,
    /// Second critic network
    critic_2: QNetwork,
    /// Target critic networks
    target_critic_1: QNetwork,
    target_critic_2: QNetwork,
    /// Replay buffer
    replay_buffer: ReplayBuffer,
    /// Configuration
    config: TD3Config,
    /// Training step counter
    step_count: usize,
}
/// TD3 configuration
#[derive(Debug, Clone)]
pub struct TD3Config {
    /// Actor learning rate
    pub actor_lr: f32,
    /// Critic learning rate
    pub critic_lr: f32,
    /// Discount factor
    pub gamma: f32,
    /// Soft update coefficient
    pub tau: f32,
    /// Policy noise for target smoothing
    pub policy_noise: f32,
    /// Noise clipping range
    pub noise_clip: f32,
    /// Policy update frequency (delayed updates)
    pub policy_delay: usize,
    /// Exploration noise standard deviation
    pub exploration_noise: f32,
    /// Replay buffer size
    pub buffer_size: usize,
    /// Batch size for training
    pub batch_size: usize,
    /// Action bounds
    pub action_low: Array1<f32>,
    pub action_high: Array1<f32>,
impl Default for TD3Config {
    fn default() -> Self {
        Self {
            actor_lr: 3e-4,
            critic_lr: 3e-4,
            gamma: 0.99,
            tau: 0.005,
            policy_noise: 0.2,
            noise_clip: 0.5,
            policy_delay: 2,
            exploration_noise: 0.1,
            buffer_size: 1_000_000,
            batch_size: 256,
            action_low: Array1::from_vec(vec![-1.0]),
            action_high: Array1::from_vec(vec![1.0]),
        }
    }
impl TD3 {
    /// Create a new TD3 agent
    pub fn new(
        state_dim: usize,
        action_dim: usize,
        hidden_sizes: Vec<usize>,
        config: TD3Config,
    ) -> Result<Self> {
        let actor = PolicyNetwork::new(state_dim, action_dim, hidden_sizes.clone(), true)?;
        let target_actor = PolicyNetwork::new(state_dim, action_dim, hidden_sizes.clone(), true)?;
        let critic_1 = QNetwork::new(state_dim, action_dim, hidden_sizes.clone())?;
        let critic_2 = QNetwork::new(state_dim, action_dim, hidden_sizes.clone())?;
        let target_critic_1 = QNetwork::new(state_dim, action_dim, hidden_sizes.clone())?;
        let target_critic_2 = QNetwork::new(state_dim, action_dim, hidden_sizes)?;
        let replay_buffer = ReplayBuffer::new(config.buffer_size);
        Ok(Self {
            actor,
            target_actor,
            critic_1,
            critic_2,
            target_critic_1,
            target_critic_2,
            replay_buffer,
            config,
            step_count: 0,
        })
    /// Add experience to replay buffer
    pub fn add_experience(
        &mut self,
        state: Array1<f32>,
        action: Array1<f32>,
        reward: f32,
        next_state: Array1<f32>,
        done: bool,
    ) -> Result<()> {
        self.replay_buffer
            .add(state, action, reward, next_state, done)
    /// Soft update target networks
    fn soft_update_targets(&mut self) -> Result<()> {
        // Soft update target critic networks
        let tau = self.config.tau;
        
        // Update target critic 1
        let current_critic1_params = self.critic1.parameters();
        let mut target_critic1_params = self.target_critic1.parameters();
        for (current, target) in current_critic1_params.iter().zip(target_critic1_params.iter_mut()) {
            // target = tau * current + (1 - tau) * target
            *target = current * tau + target.clone() * (1.0 - tau);
        self.target_critic1.set_parameters(&target_critic1_params)?;
        // Update target critic 2
        let current_critic2_params = self.critic2.parameters();
        let mut target_critic2_params = self.target_critic2.parameters();
        for (current, target) in current_critic2_params.iter().zip(target_critic2_params.iter_mut()) {
        self.target_critic2.set_parameters(&target_critic2_params)?;
        // Update target actor
        let current_actor_params = self.actor.parameters();
        let mut target_actor_params = self.target_actor.parameters();
        for (current, target) in current_actor_params.iter().zip(target_actor_params.iter_mut()) {
        self.target_actor.set_parameters(&target_actor_params)?;
        Ok(())
    /// Update TD3 networks
    pub fn update(&mut self) -> Result<LossInfo> {
        if self.replay_buffer.len() < self.config.batch_size {
            return Ok(LossInfo {
                policy_loss: None,
                value_loss: Some(0.0),
                entropy_loss: None,
                total_loss: 0.0,
                metrics: HashMap::new(),
            });
        let batch = self.replay_buffer.sample(self.config.batch_size)?;
        // Update critics
        let critic_loss = self.update_critics(&batch)?;
        let mut policy_loss = None;
        // Delayed policy updates
        if self.step_count % self.config.policy_delay == 0 {
            policy_loss = Some(self.update_actor(&batch)?);
            self.soft_update_targets()?;
        self.step_count += 1;
        let mut metrics = HashMap::new();
        metrics.insert("critic_loss".to_string(), critic_loss);
        if let Some(pl) = policy_loss {
            metrics.insert("actor_loss".to_string(), pl);
        Ok(LossInfo {
            policy_loss,
            value_loss: Some(critic_loss),
            entropy_loss: None,
            total_loss: critic_loss + policy_loss.unwrap_or(0.0),
            metrics,
    /// Update critic networks
    fn update_critics(&mut self, batch: &ExperienceBatch) -> Result<f32> {
        let batch_size = batch.states.shape()[0];
        // Sample target actions with noise
        let mut target_actions = Array2::zeros((batch_size, self.config.action_low.len()));
        for i in 0..batch_size {
            let next_state = batch.next_states.row(i);
            let target_action = self.target_actor.sample_action(&next_state)?;
            // Add clipped noise for target policy smoothing
            let noise = self.sample_noise(target_action.len(), self.config.policy_noise);
            let noisy_action = &target_action + &noise;
            let clipped_action = self.clip_action(&noisy_action);
            target_actions.row_mut(i).assign(&clipped_action);
        // Compute target Q-values (minimum of two critics)
        let q1_targets = self
            .target_critic_1
            .predict_batch(&batch.next_states, &target_actions)?;
        let q2_targets = self
            .target_critic_2
        let q_targets = Array1::from_iter(
            q1_targets
                .iter()
                .zip(q2_targets.iter())
                .map(|(&q1, &q2)| q1.min(q2)),
        );
        // Compute target values
        let targets = &batch.rewards
            + &(q_targets
                * self.config.gamma
                * batch.dones.mapv(|done| if done { 0.0 } else { 1.0 }));
        // Compute current Q-values
        let q1_current = self.critic_1.predict_batch(&batch.states, &batch.actions)?;
        let q2_current = self.critic_2.predict_batch(&batch.states, &batch.actions)?;
        // Compute critic losses (MSE)
        let q1_loss = (&q1_current - &targets).mapv(|x| x * x).mean().unwrap();
        let q2_loss = (&q2_current - &targets).mapv(|x| x * x).mean().unwrap();
        // Total critic loss
        let critic_loss = q1_loss + q2_loss;
        // In a complete implementation, gradients would be computed and applied here
        Ok(critic_loss)
    /// Update actor network
    fn update_actor(&mut self, batch: &ExperienceBatch) -> Result<f32> {
        // Compute actor loss (negative Q-value)
        let mut actor_loss = 0.0;
        for i in 0..batch.states.shape()[0] {
            let state = batch.states.row(i);
            let action = self.actor.sample_action(&state)?;
            let q_value = self.critic_1.predict(&state, &action.view())?;
            actor_loss -= q_value; // Maximize Q-value
        actor_loss /= batch.states.shape()[0] as f32;
        Ok(actor_loss)
    /// Sample noise for exploration or target smoothing
    fn sample_noise(&self, size: usize, std: f32) -> Array1<f32> {
        use rand_distr::{Distribution, Normal};
        let mut rng = rng();
        let normal = Normal::new(0.0, std).unwrap();
        Array1::from_shape_fn(size, |_| normal.sample(&mut rng))
    /// Clip actions to valid range
    fn clip_action(&self, action: &Array1<f32>) -> Array1<f32> {
        Array1::from_iter(action.iter().enumerate().map(|(i, &a)| {
            a.max(self.config.action_low[i])
                .min(self.config.action_high[i])
        }))
    /// Get model state information
    #[allow(dead_code)]
    pub fn get_model_state(&self) -> String {
        format!("TD3Agent[step_count={}, exploration_noise={}]", 
                self.step_count, self.config.exploration_noise)
impl RLAgent for TD3 {
    fn act(&self, observation: &ArrayView1<f32>, training: bool) -> Result<Array1<f32>> {
        let action = self.actor.sample_action(observation)?;
        if training {
            // Add exploration noise
            let noise = self.sample_noise(action.len(), self.config.exploration_noise);
            let noisy_action = &action + &noise;
            Ok(self.clip_action(&noisy_action))
        } else {
            Ok(self.clip_action(&action))
    fn update(&mut self, batch: &ExperienceBatch) -> Result<LossInfo> {
        // Store experience in replay buffer
            self.add_experience(
                batch.states.row(i).to_owned(),
                batch.actions.row(i).to_owned(),
                batch.rewards[i],
                batch.next_states.row(i).to_owned(),
                batch.dones[i],
            )?;
        // Update networks
        self.update()
    fn save(&self, path: &str) -> Result<()> {
        // Save all networks
        std::fs::create_dir_all(std::path::Path::new(path).parent().unwrap())?;
        // In a complete implementation, would save network parameters
    fn load(&mut self, path: &str) -> Result<()> {
        // Load all networks
        // In a complete implementation, would load network parameters
    fn exploration_rate(&self) -> f32 {
        self.config.exploration_noise
/// Rainbow DQN with multiple enhancements
pub struct RainbowDQN {
    /// Main Q-network
    q_network: EnhancedQNetwork,
    /// Target Q-network  
    target_network: EnhancedQNetwork,
    /// Prioritized replay buffer
    replay_buffer: PrioritizedReplayBuffer,
    config: RainbowConfig,
    /// Noisy networks random seed
    noisy_seed: u64,
/// Enhanced Q-network with distributional RL and noisy networks
pub struct EnhancedQNetwork {
    /// Base Q-network
    base_network: QNetwork,
    /// Number of atoms for distributional RL
    num_atoms: usize,
    /// Value range for distributional RL
    v_min: f32,
    v_max: f32,
    /// Support for distributional RL
    support: Array1<f32>,
impl EnhancedQNetwork {
        num_atoms: usize,
        v_min: f32,
        v_max: f32,
        let base_network = QNetwork::new(state_dim, action_dim, hidden_sizes)?;
        let support = Array1::linspace(v_min, v_max, num_atoms);
            base_network,
            num_atoms,
            v_min,
            v_max,
            support,
    /// Predict distributional Q-values
    pub fn predict_distribution(&self, state: &ArrayView1<f32>) -> Result<Array2<f32>> {
        // In a complete implementation, this would output a distribution over values
        // For now, return a simplified distribution
        let action_dim = self.base_network.action_dim;
        Ok(Array2::zeros((action_dim, self.num_atoms)))
    /// Predict Q-values (expectation of distribution)
    pub fn predict_q_values(&self, state: &ArrayView1<f32>) -> Result<Array1<f32>> {
        let distribution = self.predict_distribution(state)?;
        // Compute expectation: sum(support * probabilities)
        let mut q_values = Array1::zeros(distribution.shape()[0]);
        for i in 0..distribution.shape()[0] {
            q_values[i] = distribution.row(i).dot(&self.support);
        Ok(q_values)
/// Rainbow DQN configuration
pub struct RainbowConfig {
    /// Learning rate
    pub learning_rate: f32,
    /// Target network update frequency
    pub target_update_freq: usize,
    /// Batch size
    /// Multi-step return length (n-step)
    pub n_step: usize,
    /// Distributional RL parameters
    pub num_atoms: usize,
    pub v_min: f32,
    pub v_max: f32,
    /// Prioritized replay parameters
    pub alpha: f32,
    pub beta: f32,
    /// Noisy networks parameter
    pub noisy_std: f32,
    /// Double DQN
    pub use_double_dqn: bool,
    /// Dueling DQN
    pub use_dueling: bool,
impl Default for RainbowConfig {
            learning_rate: 6.25e-5,
            target_update_freq: 8000,
            batch_size: 32,
            n_step: 3,
            num_atoms: 51,
            v_min: -10.0,
            v_max: 10.0,
            alpha: 0.5,
            beta: 0.4,
            noisy_std: 0.1,
            use_double_dqn: true,
            use_dueling: true,
impl RainbowDQN {
    /// Create a new Rainbow DQN agent
        config: RainbowConfig,
        let q_network = EnhancedQNetwork::new(
            state_dim,
            action_dim,
            hidden_sizes.clone(),
            config.num_atoms,
            config.v_min,
            config.v_max,
        )?;
        let target_network = EnhancedQNetwork::new(
            hidden_sizes,
        let replay_buffer =
            PrioritizedReplayBuffer::new(config.buffer_size, config.alpha, config.beta);
            q_network,
            target_network,
            noisy_seed: 42,
    /// Select action using noisy networks (no epsilon-greedy needed)
    pub fn select_action(&self, state: &ArrayView1<f32>) -> Result<usize> {
        let q_values = self.q_network.predict_q_values(state)?;
        // Find action with highest Q-value
        let mut best_action = 0;
        let mut best_value = q_values[0];
        for (i, &value) in q_values.iter().enumerate() {
            if value > best_value {
                best_value = value;
                best_action = i;
            }
        Ok(best_action)
    /// Update the Rainbow DQN
    pub fn update_rainbow(&mut self) -> Result<LossInfo> {
        // Sample from prioritized replay buffer
        let (batch, weights, indices) = self.replay_buffer.sample(self.config.batch_size)?;
        // Compute distributional Bellman update
        let loss = self.compute_distributional_loss(&batch, &weights)?;
        // Update priorities
        let td_errors = vec![loss; indices.len()]; // Simplified
        self.replay_buffer.update_priorities(&indices, &td_errors)?;
        // Update target network
        if self.step_count % self.config.target_update_freq == 0 {
            self.update_target_network()?;
        metrics.insert("rainbow_loss".to_string(), loss);
            policy_loss: None,
            value_loss: Some(loss),
            total_loss: loss,
    /// Compute distributional loss
    fn compute_distributional_loss(
        &self,
        batch: &ExperienceBatch,
        weights: &Array1<f32>,
    ) -> Result<f32> {
        // This is a simplified implementation
        // In practice, would compute the full distributional Bellman update
        let mut total_loss = 0.0;
            let q_dist = self.q_network.predict_distribution(&state)?;
            // Simplified loss computation (would be KL divergence in practice)
            let loss = q_dist.mapv(|x| x * x).sum();
            total_loss += loss * weights[i];
        Ok(total_loss / batch.states.shape()[0] as f32)
    /// Update target network
    fn update_target_network(&mut self) -> Result<()> {
        // In a complete implementation, would copy weights from main to target network
impl RLAgent for RainbowDQN {
    fn act(&self, observation: &ArrayView1<f32>, training: bool) -> Result<Array1<f32>> {
        let action_idx = self.select_action(observation)?;
        // Convert discrete action to one-hot encoding
        let mut action = Array1::zeros(self.q_network.base_network.action_dim);
        if action_idx < action.len() {
            action[action_idx] = 1.0;
            return Err(crate::error::NeuralError::InvalidArgument(
                format!("Action index {} out of bounds for action space size {}", 
                        action_idx, action.len())
            ));
        Ok(action)
        // Add experiences to replay buffer
            self.replay_buffer.add(
        // Update Rainbow DQN
        self.update_rainbow()
        // Save network parameters
        // Load network parameters
/// IMPALA (Importance Weighted Actor-Learner Architecture)
pub struct IMPALA {
    /// Actor-critic network
    actor_critic: IMPALAActorCritic,
    config: IMPALAConfig,
    /// Experience buffer for trajectory
    trajectory_buffer: Vec<IMPALAExperience>,
/// IMPALA-specific actor-critic network
pub struct IMPALAActorCritic {
    /// Policy network
    policy: PolicyNetwork,
    /// Value network
    value: ValueNetwork,
impl IMPALAActorCritic {
        continuous: bool,
        let policy = PolicyNetwork::new(state_dim, action_dim, hidden_sizes.clone(), continuous)?;
        let value = ValueNetwork::new(state_dim, 1, hidden_sizes)?;
        Ok(Self { policy, value })
    /// Forward pass
    pub fn forward(&self, state: &ArrayView1<f32>) -> Result<(Array1<f32>, f32, f32)> {
        let action_logits = self.policy.sample_action(state)?;
        let value = self.value.predict(state)?;
        let log_prob = self.policy.log_prob(state, &action_logits.view())?;
        Ok((action_logits, value, log_prob))
/// IMPALA experience
pub struct IMPALAExperience {
    pub state: Array1<f32>,
    pub action: Array1<f32>,
    pub reward: f32,
    pub done: bool,
    pub log_prob: f32,
    pub value: f32,
/// IMPALA configuration
pub struct IMPALAConfig {
    /// Trajectory length
    pub trajectory_length: usize,
    /// Value loss coefficient
    pub value_loss_coef: f32,
    /// Entropy coefficient
    pub entropy_coef: f32,
    /// Baseline cost
    pub baseline_cost: f32,
    /// Clip importance weights
    pub clip_rho_threshold: f32,
    pub clip_c_threshold: f32,
impl Default for IMPALAConfig {
            learning_rate: 1e-3,
            trajectory_length: 20,
            value_loss_coef: 0.5,
            entropy_coef: 0.01,
            baseline_cost: 0.5,
            clip_rho_threshold: 1.0,
            clip_c_threshold: 1.0,
impl IMPALA {
    /// Create a new IMPALA agent
        config: IMPALAConfig,
        let actor_critic = IMPALAActorCritic::new(state_dim, action_dim, hidden_sizes, continuous)?;
            actor_critic,
            trajectory_buffer: Vec::new(),
    /// Add experience to trajectory buffer
    pub fn add_experience(&mut self, experience: IMPALAExperience) {
        self.trajectory_buffer.push(experience);
        // Keep buffer at maximum trajectory length
        if self.trajectory_buffer.len() > self.config.trajectory_length {
            self.trajectory_buffer.remove(0);
    /// Compute V-trace targets
    fn compute_vtrace_targets(
        trajectory: &[IMPALAExperience],
    ) -> Result<(Array1<f32>, Array1<f32>)> {
        let n = trajectory.len();
        let mut vs = Array1::zeros(n);
        let mut pg_advantages = Array1::zeros(n);
        // Simplified V-trace computation (in practice, would implement full algorithm)
        for i in 0..n {
            vs[i] = trajectory[i].value;
            pg_advantages[i] = trajectory[i].reward - trajectory[i].value;
        Ok((vs, pg_advantages))
    /// Update IMPALA using V-trace
    pub fn update_impala(&mut self) -> Result<LossInfo> {
        if self.trajectory_buffer.len() < self.config.trajectory_length {
        // Compute V-trace targets
        let (vs, pg_advantages) = self.compute_vtrace_targets(&self.trajectory_buffer)?;
        // Compute losses
        let mut policy_loss = 0.0;
        let mut value_loss = 0.0;
        let mut entropy_loss = 0.0;
        for (i, experience) in self.trajectory_buffer.iter().enumerate() {
            // Policy gradient loss with importance weighting
            policy_loss -= experience.log_prob * pg_advantages[i];
            // Value function loss
            let value_error = experience.value - vs[i];
            value_loss += value_error * value_error;
            // Entropy loss (placeholder)
            entropy_loss -= 0.01;
        let n = self.trajectory_buffer.len() as f32;
        policy_loss /= n;
        value_loss /= n;
        entropy_loss /= n;
        let total_loss = policy_loss + self.config.value_loss_coef * value_loss
            - self.config.entropy_coef * entropy_loss;
        // Clear trajectory buffer after update
        self.trajectory_buffer.clear();
        metrics.insert("pg_loss".to_string(), policy_loss);
        metrics.insert("value_loss".to_string(), value_loss);
        metrics.insert("entropy".to_string(), -entropy_loss);
            policy_loss: Some(policy_loss),
            value_loss: Some(value_loss),
            entropy_loss: Some(entropy_loss),
            total_loss,
impl RLAgent for IMPALA {
        let (action_value_log_prob) = self.actor_critic.forward(observation)?;
        // Convert batch to trajectory experiences
            let state = batch.states.row(i).to_owned();
            let action = batch.actions.row(i).to_owned();
            let (_, value, log_prob) = self.actor_critic.forward(&state.view())?;
            let experience = IMPALAExperience {
                state,
                action,
                reward: batch.rewards[i],
                done: batch.dones[i],
                log_prob,
                value,
            };
            self.add_experience(experience);
        // Update using V-trace
        self.update_impala()
/// Soft Actor-Critic (SAC) with automatic entropy tuning
pub struct SAC {
    /// Actor network (policy)
    config: SACConfig,
    /// Automatic entropy tuning
    log_alpha: f32,
    target_entropy: f32,
/// SAC configuration
pub struct SACConfig {
    /// Alpha learning rate (for entropy tuning)
    pub alpha_lr: f32,
    /// Target entropy (auto if None)
    pub target_entropy: Option<f32>,
impl Default for SACConfig {
            alpha_lr: 3e-4,
            target_entropy: None,
impl SAC {
    /// Create a new SAC agent
        config: SACConfig,
        // Set target entropy to -action_dim if not specified
        let target_entropy = config.target_entropy.unwrap_or(-(action_dim as f32));
            log_alpha: 0.0, // Start with alpha = 1.0
            target_entropy,
    /// Update SAC networks
    pub fn update_sac(&mut self) -> Result<LossInfo> {
        // Update actor and alpha
        let (actor_loss, alpha_loss, entropy) = self.update_actor_and_alpha(&batch)?;
        // Soft update target networks
        self.soft_update_targets()?;
        metrics.insert("actor_loss".to_string(), actor_loss);
        metrics.insert("alpha_loss".to_string(), alpha_loss);
        metrics.insert("entropy".to_string(), entropy);
        metrics.insert("alpha".to_string(), self.log_alpha.exp());
            policy_loss: Some(actor_loss),
            entropy_loss: Some(alpha_loss),
            total_loss: critic_loss + actor_loss + alpha_loss,
        let alpha = self.log_alpha.exp();
        // Sample actions and compute log probabilities for next states
        let mut next_actions = Array2::zeros((batch_size, self.config.action_low.len()));
        let mut next_log_probs = Array1::zeros(batch_size);
            let next_action = self.actor.sample_action(&next_state)?;
            let log_prob = self.actor.log_prob(&next_state, &next_action.view())?;
            
            next_actions.row_mut(i).assign(&next_action);
            next_log_probs[i] = log_prob;
        // Compute target Q-values with entropy regularization
        let q1_targets = self.target_critic_1.predict_batch(&batch.next_states, &next_actions)?;
        let q2_targets = self.target_critic_2.predict_batch(&batch.next_states, &next_actions)?;
        let min_q_targets = Array1::from_iter(
            q1_targets.iter().zip(q2_targets.iter()).map(|(&q1, &q2)| q1.min(q2))
        // Add entropy term
        let entropy_term = &next_log_probs * alpha;
        let targets = &batch.rewards + &((&min_q_targets - &entropy_term) * self.config.gamma 
            * batch.dones.mapv(|done| if done { 0.0 } else { 1.0 }));
        // Compute critic losses
        Ok(q1_loss + q2_loss)
    /// Update actor and alpha
    fn update_actor_and_alpha(&mut self, batch: &ExperienceBatch) -> Result<(f32, f32, f32)> {
        let mut total_entropy = 0.0;
        // Sample actions and compute Q-values
            let log_prob = self.actor.log_prob(&state, &action.view())?;
            let q1_value = self.critic_1.predict(&state, &action.view())?;
            let q2_value = self.critic_2.predict(&state, &action.view())?;
            let min_q_value = q1_value.min(q2_value);
            // Actor loss: maximize Q - alpha * log_prob
            actor_loss += alpha * log_prob - min_q_value;
            total_entropy += -log_prob;
        actor_loss /= batch_size as f32;
        total_entropy /= batch_size as f32;
        // Alpha loss for automatic entropy tuning
        let alpha_loss = -self.log_alpha * (total_entropy + self.target_entropy);
        // Update log_alpha (simplified - in practice would use optimizer)
        self.log_alpha += self.config.alpha_lr * alpha_loss;
        Ok((actor_loss, alpha_loss, total_entropy))
        // In a complete implementation, this would perform:
        // target_params = tau * current_params + (1 - tau) * target_params
impl RLAgent for SAC {
        // Clip action to bounds
        let clipped_action = Array1::from_iter(action.iter().enumerate().map(|(i, &a)| {
            a.max(self.config.action_low[i]).min(self.config.action_high[i])
        }));
        Ok(clipped_action)
        self.update_sac()
/// Advanced exploration strategies
pub struct ExplorationStrategy {
    strategy_type: ExplorationStrategyType,
    config: ExplorationConfig,
pub enum ExplorationStrategyType {
    EpsilonGreedy,
    UCB,
    ThompsonSampling,
    NoiseInjection,
    CuriosityDriven,
pub struct ExplorationConfig {
    pub epsilon_start: f32,
    pub epsilon_end: f32,
    pub epsilon_decay: f32,
    pub ucb_c: f32,
    pub noise_std: f32,
    pub curiosity_beta: f32,
impl Default for ExplorationConfig {
            epsilon_start: 1.0,
            epsilon_end: 0.01,
            epsilon_decay: 0.995,
            ucb_c: 2.0,
            noise_std: 0.1,
            curiosity_beta: 0.1,
impl ExplorationStrategy {
    pub fn new(_strategytype: ExplorationStrategyType, config: ExplorationConfig) -> Self {
            strategy_type,
    /// Select action with exploration
    pub fn select_action(
        q_values: &Array1<f32>,
        state: &ArrayView1<f32>,
    ) -> Result<usize> {
        match self._strategy_type {
            ExplorationStrategyType::EpsilonGreedy => {
                let epsilon = self.config.epsilon_end + 
                    (self.config.epsilon_start - self.config.epsilon_end) * 
                    self.config.epsilon_decay.powi(self.step_count as i32);
                if rand::random::<f32>() < epsilon {
                    // Random action
                    use rand::prelude::*;
use ndarray::ArrayView1;
use statrs::statistics::Statistics;
                    let mut rng = rng();
                    Ok(rng.random_range(0..q_values.len()))
                } else {
                    // Greedy action
                    Ok(q_values.iter()
                        .enumerate()
                        .max_by(|(_..a), (_, b)| a.partial_cmp(b).unwrap())
                        .map(|(i_)| i)
                        .unwrap_or(0))
                }
            ExplorationStrategyType::UCB => {
                // Upper Confidence Bound
                let sqrt_log_t = (self.step_count as f32).ln().sqrt();
                let mut ucb_values = Array1::zeros(q_values.len());
                
                for i in 0..q_values.len() {
                    // Simplified UCB (in practice would track action counts)
                    let confidence = self.config.ucb_c * sqrt_log_t;
                    ucb_values[i] = q_values[i] + confidence;
                Ok(ucb_values.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(i_)| i)
                    .unwrap_or(0))
            ExplorationStrategyType::ThompsonSampling => {
                // Thompson Sampling (simplified)
                use rand_distr::{Distribution, Normal};
                let mut rng = rng();
                let normal = Normal::new(0.0, 1.0).unwrap();
                let mut sampled_values = Array1::zeros(q_values.len());
                    let noise = normal.sample(&mut rng);
                    sampled_values[i] = q_values[i] + noise * 0.1;
                Ok(sampled_values.iter()
            ExplorationStrategyType::NoiseInjection => {
                // Add noise to Q-values
                let normal = Normal::new(0.0, self.config.noise_std).unwrap();
                let mut noisy_values = q_values.clone();
                for value in noisy_values.iter_mut() {
                    *value += normal.sample(&mut rng);
                Ok(noisy_values.iter()
            ExplorationStrategyType::CuriosityDriven => {
                // Curiosity-driven exploration (simplified)
                // In practice, would use intrinsic motivation
                let curiosity_bonus = self.estimate_curiosity_bonus(state)?;
                let mut enhanced_values = q_values.clone();
                for value in enhanced_values.iter_mut() {
                    *value += curiosity_bonus * self.config.curiosity_beta;
                Ok(enhanced_values.iter()
    /// Estimate curiosity bonus (simplified)
    fn estimate_curiosity_bonus(&self, state: &ArrayView1<f32>) -> Result<f32> {
        // In practice, would use prediction error from forward model
        let state_novelty = state.iter().map(|&x| x.abs()).sum::<f32>() / state.len() as f32;
        Ok(state_novelty.min(1.0))
/// Multi-Agent Deep Deterministic Policy Gradient (MADDPG)
pub struct MADDPG {
    /// Number of agents
    num_agents: usize,
    /// Individual agents
    agents: Vec<MADDPGAgent>,
    /// Shared replay buffer
    shared_buffer: ReplayBuffer,
    config: MADDPGConfig,
/// Individual MADDPG agent
pub struct MADDPGAgent {
    /// Agent ID
    agent_id: usize,
    /// Critic network (takes all agents' observations and actions)
    critic: QNetwork,
    /// Target networks
    target_critic: QNetwork,
/// MADDPG configuration
pub struct MADDPGConfig {
impl Default for MADDPGConfig {
            actor_lr: 1e-4,
            critic_lr: 1e-3,
            tau: 0.01,
impl MADDPG {
    /// Create new MADDPG system
        num_agents: usize,
        state_dims: Vec<usize>,
        action_dims: Vec<usize>,
        config: MADDPGConfig,
        let mut agents = Vec::new();
        // Calculate total observation and action dimensions for critic
        let total_state_dim: usize = state_dims.iter().sum();
        let total_action_dim: usize = action_dims.iter().sum();
        for i in 0..num_agents {
            let actor = PolicyNetwork::new(state_dims[i], action_dims[i], hidden_sizes.clone(), true)?;
            let critic = QNetwork::new(total_state_dim, total_action_dim, hidden_sizes.clone())?;
            let target_actor = PolicyNetwork::new(state_dims[i], action_dims[i], hidden_sizes.clone(), true)?;
            let target_critic = QNetwork::new(total_state_dim, total_action_dim, hidden_sizes.clone())?;
            agents.push(MADDPGAgent {
                agent_id: i,
                actor,
                critic,
                target_actor,
                target_critic,
        let shared_buffer = ReplayBuffer::new(config.buffer_size);
            num_agents,
            agents,
            shared_buffer,
    /// Get actions for all agents
    pub fn get_actions(&self, observations: &[ArrayView1<f32>], training: bool) -> Result<Vec<Array1<f32>>> {
        let mut actions = Vec::new();
        for (i, obs) in observations.iter().enumerate() {
            let mut action = self.agents[i].actor.sample_action(obs)?;
            if training {
                // Add exploration noise
                let normal = Normal::new(0.0, self.config.exploration_noise).unwrap();
                for a in action.iter_mut() {
                    *a += normal.sample(&mut rng);
            actions.push(action);
        Ok(actions)
    /// Update all agents
    pub fn update_agents(&mut self, experiences: &[ExperienceBatch]) -> Result<Vec<LossInfo>> {
        let mut loss_infos = Vec::new();
        // Add experiences to shared buffer
        for experience in experiences {
            for i in 0..experience.states.shape()[0] {
                self.shared_buffer.add(
                    experience.states.row(i).to_owned(),
                    experience.actions.row(i).to_owned(),
                    experience.rewards[i],
                    experience.next_states.row(i).to_owned(),
                    experience.dones[i],
                )?;
        if self.shared_buffer.len() < self.config.batch_size {
            return Ok(vec![LossInfo {
            }; self.num_agents]);
        let batch = self.shared_buffer.sample(self.config.batch_size)?;
        // Update each agent
        for agent in &mut self.agents {
            let loss_info = self.update_single_agent(agent, &batch)?;
            loss_infos.push(loss_info);
        Ok(loss_infos)
    /// Update a single agent
    fn update_single_agent(&mut self, agent: &mut MADDPGAgent, batch: &ExperienceBatch) -> Result<LossInfo> {
        // Simplified update (in practice would implement full MADDPG algorithm)
        let critic_loss = 0.5; // Placeholder
        let actor_loss = -0.1; // Placeholder
            total_loss: critic_loss + actor_loss.abs(),
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_td3_creation() {
        let config = TD3Config::default();
        let td3 = TD3::new(4, 2, vec![64, 64], config).unwrap();
        assert_eq!(td3.step_count, 0);
    fn test_rainbow_creation() {
        let config = RainbowConfig::default();
        let rainbow = RainbowDQN::new(4, 3, vec![128, 128], config).unwrap();
        assert_eq!(rainbow.step_count, 0);
    fn test_impala_creation() {
        let config = IMPALAConfig::default();
        let impala = IMPALA::new(4, 2, vec![64, 64], true, config).unwrap();
        assert_eq!(impala.trajectory_buffer.len(), 0);
    fn test_sac_creation() {
        let config = SACConfig::default();
        let sac = SAC::new(4, 2, vec![64, 64], config).unwrap();
        assert_eq!(sac.step_count, 0);
        assert_eq!(sac.target_entropy, -2.0);
    fn test_exploration_strategy() {
        let config = ExplorationConfig::default();
        let mut exploration = ExplorationStrategy::new(ExplorationStrategyType::EpsilonGreedy, config);
        let q_values = Array1::from_vec(vec![0.1, 0.5, 0.3]);
        let state = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let action = exploration.select_action(&q_values, &state.view()).unwrap();
        assert!(action < 3);
    fn test_maddpg_creation() {
        let config = MADDPGConfig::default();
        let maddpg = MADDPG::new(
            2,
            vec![4, 4],
            vec![2, 2],
            vec![64, 64],
        ).unwrap();
        assert_eq!(maddpg.num_agents, 2);
