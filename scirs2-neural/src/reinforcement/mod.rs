//! Reinforcement Learning extensions module
//!
//! This module provides reinforcement learning capabilities including
//! policy gradient methods, value-based methods, and actor-critic architectures.

pub mod actor_critic;
pub mod advanced_algorithms;
pub mod advanced_environments;
pub mod algorithms;
pub mod curiosity;
pub mod environments;
pub mod model_based;
pub mod policy;
pub mod policy_optimization;
pub mod replay_buffer;
pub mod trpo;
pub mod value;
pub use actor_critic::{ActorCritic, A2C, A3C, PPO, SAC as ActorCriticSAC};
pub use num_algorithms::{
    IMPALAConfig, RainbowConfig, RainbowDQN, TD3Config, IMPALA, TD3,
    SAC, SACConfig, ExplorationStrategy, ExplorationStrategyType, ExplorationConfig,
    MADDPG, MADDPGConfig, EnhancedQNetwork,
};
pub use num_environments::{
    MultiAgentEnvironment, MultiAgentGridWorld, MultiAgentWrapper, PursuitEvasion,
pub use algorithms::{RLAlgorithm, TrainingConfig};
pub use curiosity::{EpisodicCuriosity, NoveltyExploration, ICM, RND};
pub use environments::{Action, Environment, Observation, Reward};
pub use model_based::{Dyna, DynamicsModel, WorldModel, MPC};
pub use policy::{Policy, PolicyGradient, PolicyNetwork};
pub use policy_optimization::{
    CuriosityConfig, CuriosityDrivenAgent, MAMLAgent, MAMLConfig, NPGConfig, NaturalPolicyGradient,
pub use replay_buffer::{PrioritizedReplayBuffer, ReplayBuffer, ReplayBufferTrait};
pub use trpo::{TRPOConfig, TRPO};
pub use value::{DoubleDQN, QNetwork, ValueNetwork, DQN};
use crate::error::Result;
use scirs2_core::ndarray::prelude::*;
use std::sync::Arc;
use scirs2_core::ndarray::ArrayView1;
/// Configuration for reinforcement learning
#[derive(Debug, Clone)]
pub struct RLConfig {
    /// Learning rate for policy
    pub policy_lr: f32,
    /// Learning rate for value function
    pub value_lr: f32,
    /// Discount factor (gamma)
    pub discount_factor: f32,
    /// Batch size for training
    pub batch_size: usize,
    /// Replay buffer size
    pub buffer_size: usize,
    /// Target network update frequency
    pub target_update_freq: usize,
    /// Entropy coefficient for exploration
    pub entropy_coef: f32,
    /// Value loss coefficient
    pub value_loss_coef: f32,
    /// Gradient clipping value
    pub grad_clip: Option<f32>,
    /// Use GAE (Generalized Advantage Estimation)
    pub use_gae: bool,
    /// GAE lambda parameter
    pub gae_lambda: f32,
    /// Number of parallel environments
    pub n_envs: usize,
    /// Number of steps before update
    pub n_steps: usize,
}
impl Default for RLConfig {
    fn default() -> Self {
        Self {
            policy_lr: 3e-4,
            value_lr: 1e-3,
            discount_factor: 0.99,
            batch_size: 32,
            buffer_size: 100000,
            target_update_freq: 1000,
            entropy_coef: 0.01,
            value_loss_coef: 0.5,
            grad_clip: Some(0.5),
            use_gae: true,
            gae_lambda: 0.95,
            n_envs: 1,
            n_steps: 128,
        }
    }
/// Base trait for RL agents
pub trait RLAgent: Send + Sync {
    /// Select an action given an observation
    fn act(&self, observation: &ArrayView1<f32>, training: bool) -> Result<Array1<f32>>;
    /// Update the agent with a batch of experiences
    fn update(&mut self, batch: &ExperienceBatch) -> Result<LossInfo>;
    /// Save the agent's state
    fn save(&self, path: &str) -> Result<()>;
    /// Load the agent's state
    fn load(&mut self, path: &str) -> Result<()>;
    /// Get current exploration rate (if applicable)
    fn exploration_rate(&self) -> f32 {
        0.0
/// Experience batch for training
pub struct ExperienceBatch {
    /// States (batch_size, state_dim)
    pub states: Array2<f32>,
    /// Actions (batch_size, action_dim)
    pub actions: Array2<f32>,
    /// Rewards (batch_size,)
    pub rewards: Array1<f32>,
    /// Next states (batch_size, state_dim)
    pub next_states: Array2<f32>,
    /// Done flags (batch_size,)
    pub dones: Array1<bool>,
    /// Additional info (e.g., log probabilities)
    pub info: Option<std::collections::HashMap<String, Array2<f32>>>,
/// Loss information from training
pub struct LossInfo {
    /// Policy loss
    pub policy_loss: Option<f32>,
    /// Value loss
    pub value_loss: Option<f32>,
    /// Entropy loss
    pub entropy_loss: Option<f32>,
    /// Total loss
    pub total_loss: f32,
    /// Additional metrics
    pub metrics: std::collections::HashMap<String, f32>,
/// Reinforcement learning trainer
pub struct RLTrainer<E: Environment> {
    agent: Arc<dyn RLAgent>,
    environment: E,
    config: RLConfig,
    replay_buffer: Option<ReplayBuffer>,
    episode_rewards: Vec<f32>,
    episode_lengths: Vec<usize>,
impl<E: Environment> RLTrainer<E> {
    /// Create a new RL trainer
    pub fn new(agent: Arc<dyn RLAgent>, environment: E, config: RLConfig) -> Self {
        let replay_buffer = if config.buffer_size > 0 {
            Some(ReplayBuffer::new(config.buffer_size))
        } else {
            None
        };
            agent,
            environment,
            config,
            replay_buffer,
            episode_rewards: Vec::new(),
            episode_lengths: Vec::new(),
    /// Train the agent for a number of episodes
    pub fn train(&mut self, numepisodes: usize) -> Result<TrainingStats> {
        let mut total_steps = 0;
        let mut episode_rewards = Vec::new();
        let mut episode_lengths = Vec::new();
        for episode in 0..num_episodes {
            let mut state = self.environment.reset()?;
            let mut episode_reward = 0.0;
            let mut episode_length = 0;
            let mut done = false;
            while !done {
                // Select action
                let action = self.agent.act(&state.view(), true)?;
                // Take action in environment
                let (next_state, reward, done_flag_info) = self.environment.step(&action)?;
                // Store experience if using replay buffer
                if let Some(buffer) = &mut self.replay_buffer {
                    buffer.add(
                        state.clone(),
                        action.clone(),
                        reward,
                        next_state.clone(),
                        done_flag,
                    )?;
                    // Update agent from replay buffer
                    if buffer.len() >= self.config.batch_size {
                        let batch = buffer.sample(self.config.batch_size)?;
                        let _loss_info = Arc::get_mut(&mut self.agent)
                            .ok_or_else(|| {
                                crate::error::NeuralError::InvalidArgument(
                                    "Cannot get mutable reference to agent".to_string(),
                                )
                            })?
                            .update(&batch)?;
                    }
                }
                state = next_state;
                episode_reward += reward;
                episode_length += 1;
                total_steps += 1;
                done = done_flag;
                // Max episode length
                if episode_length >= 1000 {
                    break;
            }
            episode_rewards.push(episode_reward);
            episode_lengths.push(episode_length);
            // Log progress
            if (episode + 1) % 100 == 0 {
                let avg_reward = episode_rewards[episode.saturating_sub(99)..=episode]
                    .iter()
                    .sum::<f32>()
                    / 100.0;
                println!("Episode {}: Avg Reward = {:.2}", episode + 1, avg_reward);
        self.episode_rewards.extend(&episode_rewards);
        self.episode_lengths.extend(&episode_lengths);
        Ok(TrainingStats {
            total_episodes: num_episodes,
            total_steps,
            episode_rewards,
            episode_lengths,
            final_avg_reward: self.episode_rewards.iter().rev().take(100).sum::<f32>()
                / 100.0.min(self.episode_rewards.len() as f32),
        })
    /// Evaluate the agent
    pub fn evaluate(&mut self, numepisodes: usize) -> Result<EvaluationStats> {
        for _ in 0..num_episodes {
            while !done && episode_length < 1000 {
                let action = self.agent.act(&state.view(), false)?;
        let mean_reward = episode_rewards.iter().sum::<f32>() / episode_rewards.len() as f32;
        let std_reward = {
            let variance = episode_rewards
                .iter()
                .map(|r| (r - mean_reward).powi(2))
                .sum::<f32>()
                / episode_rewards.len() as f32;
            variance.sqrt()
        Ok(EvaluationStats {
            num_episodes,
            mean_reward,
            std_reward,
            min_reward: episode_rewards
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .copied()
                .unwrap_or(0.0),
            max_reward: episode_rewards
                .max_by(|a, b| a.partial_cmp(b).unwrap())
            mean_length: episode_lengths.iter().sum::<usize>() as f32
                / episode_lengths.len() as f32,
/// Training statistics
pub struct TrainingStats {
    /// Total number of episodes
    pub total_episodes: usize,
    /// Total number of steps
    pub total_steps: usize,
    /// Rewards per episode
    pub episode_rewards: Vec<f32>,
    /// Episode lengths
    pub episode_lengths: Vec<usize>,
    /// Final average reward (last 100 episodes)
    pub final_avg_reward: f32,
/// Evaluation statistics
pub struct EvaluationStats {
    /// Number of evaluation episodes
    pub num_episodes: usize,
    /// Mean reward
    pub mean_reward: f32,
    /// Standard deviation of rewards
    pub std_reward: f32,
    /// Minimum reward
    pub min_reward: f32,
    /// Maximum reward
    pub max_reward: f32,
    /// Mean episode length
    pub mean_length: f32,
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_rl_config_default() {
        let config = RLConfig::default();
        assert_eq!(config.discount_factor, 0.99);
        assert_eq!(config.batch_size, 32);
        assert!(config.use_gae);
