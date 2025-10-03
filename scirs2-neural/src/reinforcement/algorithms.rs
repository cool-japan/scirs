//! High-level reinforcement learning algorithms

use crate::error::Result;
use crate::reinforcement::environments::Environment;
use crate::reinforcement::replay_buffer::{PrioritizedReplayBuffer, ReplayBuffer, ReplayBufferTrait};
use crate::reinforcement::{ExperienceBatch, LossInfo, RLAgent};
use scirs2_core::ndarray::prelude::*;
use scirs2_core::random::Rng;
use scirs2_core::ndarray::ArrayView1;
/// Training configuration for RL algorithms
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Total number of timesteps to train
    pub total_timesteps: usize,
    /// Number of steps between policy updates
    pub update_frequency: usize,
    /// Number of gradient steps per update
    pub gradient_steps: usize,
    /// Learning starts after this many steps
    pub learning_starts: usize,
    /// Batch size for training
    pub batch_size: usize,
    /// Buffer size for experience replay
    pub buffer_size: usize,
    /// Discount factor (gamma)
    pub gamma: f32,
    /// Learning rate
    pub learning_rate: f32,
    /// Target network update frequency (for value-based methods)
    pub target_update_freq: Option<usize>,
    /// Exploration parameters
    pub exploration_initial: f32,
    pub exploration_final: f32,
    pub exploration_fraction: f32,
    /// Logging frequency
    pub log_interval: usize,
    /// Evaluation frequency
    pub eval_freq: Option<usize>,
    /// Number of evaluation episodes
    pub n_eval_episodes: usize,
    /// Save frequency
    pub save_freq: Option<usize>,
    /// Save path
    pub save_path: Option<String>,
    /// Use prioritized replay
    pub use_prioritized_replay: bool,
    /// Prioritized replay alpha
    pub prioritized_replay_alpha: f32,
    /// Prioritized replay beta
    pub prioritized_replay_beta0: f32,
    pub prioritized_replay_beta_schedule: Option<String>,
}
impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            total_timesteps: 1_000_000,
            update_frequency: 4,
            gradient_steps: 1,
            learning_starts: 50_000,
            batch_size: 32,
            buffer_size: 1_000_000,
            gamma: 0.99,
            learning_rate: 1e-4,
            target_update_freq: Some(10_000),
            exploration_initial: 1.0,
            exploration_final: 0.05,
            exploration_fraction: 0.1,
            log_interval: 1000,
            eval_freq: Some(10_000),
            n_eval_episodes: 10,
            save_freq: Some(50_000),
            save_path: Some("checkpoints".to_string()),
            use_prioritized_replay: false,
            prioritized_replay_alpha: 0.6,
            prioritized_replay_beta0: 0.4,
            prioritized_replay_beta_schedule: Some("linear".to_string()),
        }
    }
/// Base trait for RL algorithms
pub trait RLAlgorithm: Send + Sync {
    /// Train the algorithm
    fn train(
        &mut self,
        env: &mut dyn Environment,
        config: &TrainingConfig,
    ) -> Result<TrainingResults>;
    /// Evaluate the algorithm
    fn evaluate(&self, env: &mut dyn Environment, nepisodes: usize) -> Result<EvaluationResults>;
    /// Save the algorithm
    fn save(&self, path: &str) -> Result<()>;
    /// Load the algorithm
    fn load(&mut self, path: &str) -> Result<()>;
    /// Get the underlying agent
    fn agent(&self) -> &dyn RLAgent;
    /// Get mutable reference to the agent
    fn agent_mut(&mut self) -> &mut dyn RLAgent;
/// Training results
pub struct TrainingResults {
    /// Episode rewards over training
    pub episode_rewards: Vec<f32>,
    /// Episode lengths
    pub episode_lengths: Vec<usize>,
    /// Loss values over training
    pub losses: Vec<LossInfo>,
    /// Evaluation results
    pub eval_results: Vec<EvaluationResults>,
    /// Total training time in seconds
    pub training_time: f64,
    /// Total number of steps
    pub total_steps: usize,
/// Evaluation results
pub struct EvaluationResults {
    /// Mean reward across episodes
    pub mean_reward: f32,
    /// Standard deviation of rewards
    pub std_reward: f32,
    /// Mean episode length
    pub mean_length: f32,
    /// Minimum reward
    pub min_reward: f32,
    /// Maximum reward
    pub max_reward: f32,
    /// Number of episodes evaluated
    pub n_episodes: usize,
/// Off-policy algorithm base implementation
pub struct OffPolicyAlgorithm<A: RLAgent> {
    agent: A,
    replay_buffer: Option<ReplayBuffer>,
    prioritized_buffer: Option<PrioritizedReplayBuffer>,
impl<A: RLAgent + 'static> OffPolicyAlgorithm<A> {
    /// Create a new off-policy algorithm
    pub fn new(agent: A, config: &TrainingConfig) -> Self {
        let replay_buffer = if !config.use_prioritized_replay {
            Some(ReplayBuffer::new(config.buffer_size))
        } else {
            None
        };
        let prioritized_buffer = if config.use_prioritized_replay {
            Some(PrioritizedReplayBuffer::new(
                config.buffer_size,
                config.prioritized_replay_alpha,
                config.prioritized_replay_beta0,
            ))
            agent,
            replay_buffer,
            prioritized_buffer,
    /// Add experience to replay buffer
    fn add_to_buffer(
        state: Array1<f32>,
        action: Array1<f32>,
        reward: f32,
        next_state: Array1<f32>,
        done: bool,
    ) -> Result<()> {
        if let Some(buffer) = &mut self.replay_buffer {
            buffer.add(state, action, reward, next_state, done)?;
        } else if let Some(buffer) = &mut self.prioritized_buffer {
        Ok(())
    /// Sample from replay buffer
    fn sample_batch(
        &self,
        batch_size: usize,
    ) -> Result<(ExperienceBatch, Option<Array1<f32>>, Option<Vec<usize>>)> {
        if let Some(buffer) = &self.replay_buffer {
            Ok((buffer.sample(batch_size)?, None, None))
        } else if let Some(buffer) = &self.prioritized_buffer {
            let (batch, weights, indices) = buffer.sample(batch_size)?;
            Ok((batch, Some(weights), Some(indices)))
            Err(crate::error::NeuralError::InvalidArgument(
                "No replay buffer configured".to_string(),
    /// Update prioritized replay buffer priorities
    fn update_priorities(&mut self, indices: &[usize], tderrors: &[f32]) -> Result<()> {
        if let Some(buffer) = &mut self.prioritized_buffer {
            buffer.update_priorities(indices, td_errors)?;
impl<A: RLAgent + 'static> RLAlgorithm for OffPolicyAlgorithm<A> {
    ) -> Result<TrainingResults> {
        let start_time = std::time::Instant::now();
        let mut total_steps = 0;
        let mut episode_rewards = Vec::new();
        let mut episode_lengths = Vec::new();
        let mut losses = Vec::new();
        let mut eval_results = Vec::new();
        let mut state = env.reset()?;
        let mut episode_reward = 0.0;
        let mut episode_length = 0;
        // Calculate exploration schedule
        let exploration_schedule = |t: usize| -> f32 {
            let fraction =
                (t as f32 / config.total_timesteps as f32).min(config.exploration_fraction);
            config.exploration_final
                + (config.exploration_initial - config.exploration_final)
                    * (1.0 - fraction / config.exploration_fraction)
        // Update beta schedule for prioritized replay
        let beta_schedule = |t: usize| -> f32 {
            let fraction = t as f32 / config.total_timesteps as f32;
            config.prioritized_replay_beta0 + (1.0 - config.prioritized_replay_beta0) * fraction
        while total_steps < config.total_timesteps {
            // Set exploration rate
            let exploration_rate = exploration_schedule(total_steps);
            // Select action
            let training = total_steps >= config.learning_starts;
            let action = self.agent.act(&state.view(), training)?;
            // Take action in environment
            let (next_state, reward, done_info) = env.step(&action)?;
            // Add to replay buffer
            self.add_to_buffer(
                state.clone(),
                action.clone(),
                reward,
                next_state.clone(),
                done,
            )?;
            // Update counters
            episode_reward += reward;
            episode_length += 1;
            total_steps += 1;
            // Training update
            if total_steps >= config.learning_starts && total_steps % config.update_frequency == 0 {
                for _ in 0..config.gradient_steps {
                    let (batch, weights, indices) = self.sample_batch(config.batch_size)?;
                    // Update agent
                    let loss_info = self.agent.update(&batch)?;
                    losses.push(loss_info.clone());
                    // Update priorities if using prioritized replay
                    if let Some(indices) = indices {
                        if let Some(td_errors) = loss_info.metrics.get("td_errors") {
                            // Note: This assumes td_errors is stored as a single value
                            // In practice, we'd need per-sample TD errors
                            let td_errors_vec = vec![*td_errors; indices.len()];
                            self.update_priorities(&indices, &td_errors_vec)?;
                        }
                    }
                }
                // Update beta for prioritized replay
                if let Some(buffer) = &mut self.prioritized_buffer {
                    buffer.update_beta(beta_schedule(total_steps));
            }
            // Episode finished
            if done || episode_length >= 1000 {
                episode_rewards.push(episode_reward);
                episode_lengths.push(episode_length);
                // Reset environment
                state = env.reset()?;
                episode_reward = 0.0;
                episode_length = 0;
                // Logging
                if episode_rewards.len() % config.log_interval == 0 {
                    let recent_rewards =
                        &episode_rewards[episode_rewards.len().saturating_sub(100)..];
                    let avg_reward =
                        recent_rewards.iter().sum::<f32>() / recent_rewards.len() as f32;
                    println!(
                        "Steps: {}, Episodes: {}, Avg Reward: {:.2}, Exploration: {:.3}",
                        total_steps,
                        episode_rewards.len(),
                        avg_reward,
                        exploration_rate
                    );
            } else {
                state = next_state;
            // Evaluation
            if let Some(eval_freq) = config.eval_freq {
                if total_steps % eval_freq == 0 {
                    let eval_result = self.evaluate(env, config.n_eval_episodes)?;
                        "Evaluation at step {}: mean_reward={:.2} +/- {:.2}",
                        total_steps, eval_result.mean_reward, eval_result.std_reward
                    eval_results.push(eval_result);
            // Save checkpoint
            if let Some(save_freq) = config.save_freq {
                if total_steps % save_freq == 0 {
                    if let Some(save_path) = &config.save_path {
                        let checkpoint_path =
                            format!("{}/checkpoint_{}.bin", save_path, total_steps);
                        self.save(&checkpoint_path)?;
        Ok(TrainingResults {
            episode_rewards,
            episode_lengths,
            losses,
            eval_results,
            training_time: start_time.elapsed().as_secs_f64(),
            total_steps,
        })
    fn evaluate(&self, env: &mut dyn Environment, nepisodes: usize) -> Result<EvaluationResults> {
        let mut rewards = Vec::new();
        let mut lengths = Vec::new();
        for _ in 0..n_episodes {
            let mut state = env.reset()?;
            let mut episode_reward = 0.0;
            let mut episode_length = 0;
            loop {
                let action = self.agent.act(&state.view(), false)?;
                let (next_state, reward, done_info) = env.step(&action)?;
                episode_reward += reward;
                episode_length += 1;
                if done || episode_length >= 1000 {
                    break;
            rewards.push(episode_reward);
            lengths.push(episode_length);
        let mean_reward = rewards.iter().sum::<f32>() / rewards.len() as f32;
        let variance = rewards
            .iter()
            .map(|r| (r - mean_reward).powi(2))
            .sum::<f32>()
            / rewards.len() as f32;
        let std_reward = variance.sqrt();
        let min_reward = rewards
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .copied()
            .unwrap_or(0.0);
        let max_reward = rewards
            .max_by(|a, b| a.partial_cmp(b).unwrap())
        let mean_length = lengths.iter().sum::<usize>() as f32 / lengths.len() as f32;
        Ok(EvaluationResults {
            mean_reward,
            std_reward,
            mean_length,
            min_reward,
            max_reward,
            n_episodes,
    fn save(&self, path: &str) -> Result<()> {
        // Create directory if it doesn't exist
        if let Some(parent) = std::path::Path::new(path).parent() {
            std::fs::create_dir_all(parent)?;
        // Save agent state
        self.agent.save(path)?;
        // Save replay buffer state for resuming training
            let buffer_path = format!("{}.replay_buffer", path);
            buffer.save(&buffer_path)?;
            let buffer_path = format!("{}.prioritized_buffer", path);
    fn load(&mut self, path: &str) -> Result<()> {
        self.agent.load(path)?;
        // Load replay buffer state if available
            if std::path::Path::new(&buffer_path).exists() {
                buffer.load(&buffer_path)?;
    fn agent(&self) -> &dyn RLAgent {
        &self.agent as &dyn RLAgent
    fn agent_mut(&mut self) -> &mut dyn RLAgent {
        &mut self.agent as &mut dyn RLAgent
/// Wrapper for DQN to implement RLAgent trait
pub struct DQNWrapper {
    dqn: crate::reinforcement::value::DQN,
    exploration_rate: f32,
    exploration_decay: f32,
    min_exploration: f32,
impl DQNWrapper {
    pub fn new(
        state_dim: usize,
        action_dim: usize,
        hidden_dims: Vec<usize>,
        learning_rate: f32,
        gamma: f32,
        exploration_initial: f32,
        exploration_final: f32,
        target_update_freq: usize,
    ) -> Result<Self> {
        let dqn = crate::reinforcement::value::DQN::new(
            state_dim,
            action_dim,
            hidden_dims,
            learning_rate,
            gamma,
            exploration_initial,
            target_update_freq,
        )?;
        Ok(Self {
            dqn,
            exploration_rate: exploration_initial,
            exploration_decay: (exploration_initial - exploration_final) / 1000000.0,
            min_exploration: exploration_final,
    fn update_exploration(&mut self) {
        self.exploration_rate =
            (self.exploration_rate - self.exploration_decay).max(self.min_exploration);
impl RLAgent for DQNWrapper {
    fn act(&self, observation: &ArrayView1<f32>, training: bool) -> Result<Array1<f32>> {
        // Use DQN's select_action method
        let action_idx = self.dqn.select_action(observation, training)?;
        // Convert to one-hot encoding (for compatibility with RLAgent interface)
        let action_dim = self.get_action_dim();
        let mut action = Array1::zeros(action_dim);
        if action_idx < action_dim {
            action[action_idx] = 1.0;
        Ok(action)
    fn update(&mut self, batch: &ExperienceBatch) -> Result<LossInfo> {
        // Update exploration rate
        self.update_exploration();
        // Use DQN's update method
        let loss = self.dqn.update(batch)?;
        Ok(LossInfo {
            policy_loss: None,
            value_loss: Some(loss),
            entropy_loss: None,
            total_loss: loss,
            metrics: std::collections::HashMap::new(),
        // For now, create a placeholder save implementation
        std::fs::create_dir_all(
            std::path::Path::new(path)
                .parent()
                .unwrap_or(std::path::Path::new(".")),
        )?
        .map_err(|e| {
            crate::error::NeuralError::IOError(format!("Failed to create directory: {}", e))
        })?;
        // In a real implementation, this would serialize the DQN weights
        let config = format!(
            "{{\"exploration_rate\": {}, \"exploration_decay\": {}, \"min_exploration\": {}}}",
            self.exploration_rate, self.exploration_decay, self.min_exploration
        );
        std::fs::write(format!("{}/dqn_config.json", path), config).map_err(|e| {
            crate::error::NeuralError::IOError(format!("Failed to save config: {}", e))
        // In a real implementation, this would load the DQN weights
        if let Ok(_config_str) = std::fs::read_to_string(format!("{}/dqn_config.json", path)) {
            // Simplified loading - in practice would parse JSON and load exploration params
            // For now just reset to initial values
            self.exploration_rate = 0.1;
    fn exploration_rate(&self) -> f32 {
        self.exploration_rate
    fn get_action_dim(&self) -> usize {
        // Access action dimension from DQN's q_network
        // For now, use a default value since we can't access private fields
        // In a real implementation, this would be exposed as a public method
        64 // placeholder
/// Wrapper for PPO to implement RLAgent trait
pub struct PPOWrapper {
    ppo: crate::reinforcement::actor_critic::PPO,
impl PPOWrapper {
        continuous: bool,
        actor_lr: f32,
        critic_lr: f32,
        clip_epsilon: f32,
        entropy_coef: f32,
        value_loss_coef: f32,
        let ppo = crate::reinforcement::actor_critic::PPO::new(
            continuous,
            actor_lr,
            critic_lr,
            clip_epsilon,
            entropy_coef,
            value_loss_coef,
        Ok(Self { ppo })
impl RLAgent for PPOWrapper {
    fn act(&self, observation: &ArrayView1<f32>, training: bool) -> Result<Array1<f32>> {
        // PPO acts directly on the observation
        self.ppo.act(observation)
        // PPO requires trajectory data, but we'll adapt the batch format
        let (policy_loss, value_loss, entropy_loss) = self.ppo.train_batch(
            &batch.states.view(),
            &batch.actions.view(),
            &batch.rewards.view(),
            &batch.next_states.view(),
            &batch.dones.view(),
        let total_loss = policy_loss + value_loss - entropy_loss; // Subtract entropy for exploration
        let mut metrics = std::collections::HashMap::new();
        metrics.insert("policy_loss".to_string(), policy_loss);
        metrics.insert("value_loss".to_string(), value_loss);
        metrics.insert("entropy_loss".to_string(), entropy_loss);
            policy_loss: Some(policy_loss),
            value_loss: Some(value_loss),
            entropy_loss: Some(entropy_loss),
            total_loss,
            metrics,
        self.ppo.save(path)
        self.ppo.load(path)
/// Generic RL algorithm implementation
pub struct RLAlgorithmImpl {
    pub agent: Box<dyn RLAgent>,
    pub replay_buffer: Option<Box<dyn ReplayBufferTrait>>,
impl RLAlgorithm for RLAlgorithmImpl {
        // Basic training loop - this is a simplified implementation
            let action = self.agent.act(&state.view(), true)?;
            
            // Step environment
            // Store experience in replay buffer if available
            if let Some(buffer) = &mut self.replay_buffer {
                buffer.add(state.clone(), action, reward, next_state.clone(), done)?;
            if done {
                
                // Reset for next episode
        let training_time = start_time.elapsed();
        
            total_timesteps: total_steps,
            losses: vec![], // Simplified - no loss tracking in this basic implementation
            eval_results: vec![],
            training_time,
            final_performance: episode_rewards.last().copied().unwrap_or(0.0),
                let action = self.agent.act(&state.view(), false)?; // Not training mode
                if done {
            episode_rewards.push(episode_reward);
            episode_lengths.push(episode_length);
            mean_reward: episode_rewards.iter().sum::<f32>() / episode_rewards.len() as f32,
            std_reward: {
                let mean = episode_rewards.iter().sum::<f32>() / episode_rewards.len() as f32;
                let variance = episode_rewards.iter()
                    .map(|&x| (x - mean).powi(2))
                    .sum::<f32>() / episode_rewards.len() as f32;
                variance.sqrt()
            },
            mean_length: episode_lengths.iter().sum::<usize>() as f32 / episode_lengths.len() as f32,
    fn save(selfpath: &str) -> Result<()> {
        // Simplified - no saving implementation
    fn load(&mut selfpath: &str) -> Result<()> {
        // Simplified - no loading implementation
/// Helper function to create common RL algorithms
#[allow(dead_code)]
pub fn create_algorithm(
    algorithm_name: &str,
    state_dim: usize,
    action_dim: usize,
    config: &TrainingConfig,
) -> Result<Box<dyn RLAlgorithm>> {
    match algorithm_name.to_lowercase().as_str() {
        "dqn" => {
            let agent = DQNWrapper::new(
                state_dim,
                action_dim,
                vec![64, 64],
                config.learning_rate,
                config.gamma,
                config.exploration_initial,
                config.exploration_final,
                config.target_update_freq.unwrap_or(10000),
            let buffer = if config.use_prioritized_replay {
                Box::new(PrioritizedReplayBuffer::new(
                    config.buffer_size,
                    config.prioritized_replay_alpha,
                    config.prioritized_replay_beta0,
                )) as Box<dyn ReplayBufferTrait>
                Box::new(
                    crate::reinforcement::replay_buffer::SimpleReplayBuffer::new(
                        config.buffer_size,
                    ),
                ) as Box<dyn ReplayBufferTrait>
            };
            Ok(Box::new(RLAlgorithmImpl {
                agent: Box::new(agent),
                replay_buffer: Some(buffer),
            }))
        "ppo" => {
            let agent = PPOWrapper::new(
                true, // continuous
                0.2,  // clip_epsilon
                0.01, // entropy_coef
                0.5,  // value_loss_coef
                replay_buffer: None, // PPO doesn't use replay buffer
        _ => Err(crate::error::NeuralError::InvalidArgument(format!(
            "Unknown algorithm: {}",
            algorithm_name
        ))),
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_training_config_default() {
        let config = TrainingConfig::default();
        assert_eq!(config.total_timesteps, 1_000_000);
        assert_eq!(config.batch_size, 32);
        assert_eq!(config.gamma, 0.99);
    fn test_exploration_schedule() {
        // At start
        let exploration_0 = config.exploration_initial;
        assert_eq!(exploration_0, 1.0);
        // At end of exploration
        let steps_end = (config.total_timesteps as f32 * config.exploration_fraction) as usize;
        let fraction = config.exploration_fraction;
        let exploration_end = config.exploration_final
            + (config.exploration_initial - config.exploration_final)
                * (1.0 - fraction / config.exploration_fraction);
        assert!((exploration_end - config.exploration_final).abs() < 1e-6);
