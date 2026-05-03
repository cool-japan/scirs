//! Reinforcement Learning extensions module
//!
//! This module provides reinforcement learning capabilities including
//! policy gradient methods, value-based methods, and actor-critic architectures.
//!
//! ## Architecture
//!
//! The module is organised into the following sub-modules:
//!
//! | Module | Description |
//! |--------|-------------|
//! | [`environments`] | Core environment trait + CartPole, GridWorld |
//! | [`replay_buffer`] | Uniform and prioritized experience replay |
//! | [`policy`] | Policy trait + neural network actor |
//! | [`value`] | Value networks, Q-networks, DQN, DoubleDQN |
//! | [`actor_critic`] | A2C, A3C, PPO, SAC |
//! | [`algorithms`] | Training configs, `RLAlgorithm` trait, off-policy scaffold |
//! | [`policy_optimization`] | NPG, MAML, curiosity-driven agents |
//! | [`model_based`] | DynamicsModel, WorldModel, Dyna, MPC |
//! | [`curiosity`] | ICM, RND, EpisodicCuriosity, NoveltyExploration |
//! | [`trpo`] | Trust Region Policy Optimization |
//! | [`advanced_algorithms`] | TD3, Rainbow DQN, IMPALA, MADDPG |
//! | [`advanced_environments`] | Multi-agent environments, Pursuit-Evasion |

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

use crate::error::Result;
use scirs2_core::ndarray::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

// ── Core shared types ─────────────────────────────────────────────────────────

/// Configuration for reinforcement learning
#[derive(Debug, Clone)]
pub struct RLConfig {
    pub policy_lr: f32,
    pub value_lr: f32,
    pub discount_factor: f32,
    pub batch_size: usize,
    pub buffer_size: usize,
    pub target_update_freq: usize,
    pub entropy_coef: f32,
    pub value_loss_coef: f32,
    pub grad_clip: Option<f32>,
    pub use_gae: bool,
    pub gae_lambda: f32,
    pub n_envs: usize,
    pub n_steps: usize,
}

impl Default for RLConfig {
    fn default() -> Self {
        Self {
            policy_lr: 3e-4,
            value_lr: 1e-3,
            discount_factor: 0.99,
            batch_size: 32,
            buffer_size: 100_000,
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
}

/// Base trait for RL agents
pub trait RLAgent: Send + Sync {
    fn act(&self, observation: &ArrayView1<f32>, training: bool) -> Result<Array1<f32>>;
    fn update(&mut self, batch: &ExperienceBatch) -> Result<LossInfo>;
    fn save(&self, path: &str) -> Result<()>;
    fn load(&mut self, path: &str) -> Result<()>;
    fn exploration_rate(&self) -> f32 {
        0.0
    }
}

/// A batch of experiences sampled from a replay buffer
pub struct ExperienceBatch {
    pub states: Array2<f32>,
    pub actions: Array2<f32>,
    pub rewards: Array1<f32>,
    pub next_states: Array2<f32>,
    pub dones: Array1<bool>,
    pub info: Option<HashMap<String, Array2<f32>>>,
}

/// Loss information returned from an agent update
#[derive(Debug, Clone)]
pub struct LossInfo {
    pub policy_loss: Option<f32>,
    pub value_loss: Option<f32>,
    pub entropy_loss: Option<f32>,
    pub total_loss: f32,
    pub metrics: HashMap<String, f32>,
}

// ── Training / evaluation types ───────────────────────────────────────────────

/// Statistics from a training run
pub struct TrainingStats {
    pub total_episodes: usize,
    pub total_steps: usize,
    pub episode_rewards: Vec<f32>,
    pub episode_lengths: Vec<usize>,
    pub final_avg_reward: f32,
}

/// Statistics from an evaluation rollout
pub struct EvaluationStats {
    pub num_episodes: usize,
    pub mean_reward: f32,
    pub std_reward: f32,
    pub min_reward: f32,
    pub max_reward: f32,
    pub mean_length: f32,
}

// ── High-level trainer ────────────────────────────────────────────────────────

/// General-purpose RL trainer
pub struct RLTrainer<E: environments::Environment> {
    agent: Arc<dyn RLAgent>,
    environment: E,
    config: RLConfig,
    replay_buffer: Option<replay_buffer::ReplayBuffer>,
    episode_rewards: Vec<f32>,
    episode_lengths: Vec<usize>,
}

impl<E: environments::Environment> RLTrainer<E> {
    /// Create a new RL trainer
    pub fn new(agent: Arc<dyn RLAgent>, environment: E, config: RLConfig) -> Self {
        let replay_buffer = if config.buffer_size > 0 {
            Some(replay_buffer::ReplayBuffer::new(config.buffer_size))
        } else {
            None
        };
        Self {
            agent,
            environment,
            config,
            replay_buffer,
            episode_rewards: Vec::new(),
            episode_lengths: Vec::new(),
        }
    }

    /// Train the agent for `num_episodes` episodes
    pub fn train(&mut self, num_episodes: usize) -> Result<TrainingStats> {
        let mut total_steps = 0usize;
        let mut episode_rewards = Vec::new();
        let mut episode_lengths = Vec::new();

        for _episode in 0..num_episodes {
            let mut state = self.environment.reset()?;
            let mut episode_reward = 0.0f32;
            let mut episode_length = 0usize;
            let mut done = false;

            while !done {
                let action = self.agent.act(&state.view(), true)?;
                let (next_state, reward, done_flag, _info) = self.environment.step(&action)?;

                if let Some(buffer) = &mut self.replay_buffer {
                    let _ = buffer.add(
                        state.clone(),
                        action.clone(),
                        reward,
                        next_state.clone(),
                        done_flag,
                    );
                    if buffer.len() >= self.config.batch_size {
                        let batch = buffer.sample(self.config.batch_size)?;
                        let agent = Arc::get_mut(&mut self.agent).ok_or_else(|| {
                            crate::error::NeuralError::InvalidArgument(
                                "Cannot get mutable reference to agent".to_string(),
                            )
                        })?;
                        let _ = agent.update(&batch);
                    }
                }

                state = next_state;
                episode_reward += reward;
                episode_length += 1;
                total_steps += 1;
                done = done_flag;
                if episode_length >= 1000 {
                    break;
                }
            }
            episode_rewards.push(episode_reward);
            episode_lengths.push(episode_length);
        }
        self.episode_rewards.extend(&episode_rewards);
        self.episode_lengths.extend(&episode_lengths);

        let tail = self.episode_rewards.len().min(100);
        let final_avg_reward = self.episode_rewards[self.episode_rewards.len() - tail..]
            .iter()
            .sum::<f32>()
            / tail.max(1) as f32;

        Ok(TrainingStats {
            total_episodes: num_episodes,
            total_steps,
            episode_rewards,
            episode_lengths,
            final_avg_reward,
        })
    }

    /// Evaluate the agent for `num_episodes` episodes
    pub fn evaluate(&mut self, num_episodes: usize) -> Result<EvaluationStats> {
        let mut episode_rewards = Vec::new();
        let mut episode_lengths = Vec::new();

        for _ in 0..num_episodes {
            let mut state = self.environment.reset()?;
            let mut episode_reward = 0.0f32;
            let mut episode_length = 0usize;
            let mut done = false;
            while !done && episode_length < 1000 {
                let action = self.agent.act(&state.view(), false)?;
                let (ns, r, d, _) = self.environment.step(&action)?;
                episode_reward += r;
                episode_length += 1;
                state = ns;
                done = d;
            }
            episode_rewards.push(episode_reward);
            episode_lengths.push(episode_length);
        }

        let mean_reward = episode_rewards.iter().sum::<f32>() / episode_rewards.len().max(1) as f32;
        let variance = episode_rewards
            .iter()
            .map(|r| (r - mean_reward).powi(2))
            .sum::<f32>()
            / episode_rewards.len().max(1) as f32;
        let std_reward = variance.sqrt();
        let min_reward = episode_rewards
            .iter()
            .cloned()
            .fold(f32::INFINITY, f32::min);
        let max_reward = episode_rewards
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let mean_length =
            episode_lengths.iter().sum::<usize>() as f32 / episode_lengths.len().max(1) as f32;

        Ok(EvaluationStats {
            num_episodes,
            mean_reward,
            std_reward,
            min_reward: if min_reward.is_finite() {
                min_reward
            } else {
                0.0
            },
            max_reward: if max_reward.is_finite() {
                max_reward
            } else {
                0.0
            },
            mean_length,
        })
    }
}

// ── Top-level re-exports ──────────────────────────────────────────────────────

pub use actor_critic::{ActorCritic, SACConfig, A2C, A3C, PPO, SAC};
pub use advanced_algorithms::{
    EnhancedQNetwork, ExplorationConfig, ExplorationStrategy, ExplorationStrategyType,
    IMPALAConfig, MADDPGConfig, RainbowConfig, RainbowDQN, TD3Config, IMPALA, MADDPG, TD3,
};
pub use advanced_environments::{
    MultiAgentEnvironment, MultiAgentGridWorld, MultiAgentWrapper, PursuitEvasion,
};
pub use algorithms::{
    EvaluationResults, RLAlgorithm, RLAlgorithmImpl, TrainingConfig, TrainingResults,
};
pub use curiosity::{EpisodicCuriosity, NoveltyExploration, ICM, RND};
pub use environments::{Action, CartPole, Environment, GridWorld, Info, Observation, Reward};
pub use model_based::{Dyna, DynamicsModel, WorldModel, MPC};
pub use policy::{Policy, PolicyGradient, PolicyNetwork};
pub use policy_optimization::{
    CuriosityConfig, CuriosityDrivenAgent, MAMLAgent, MAMLConfig, NPGConfig, NaturalPolicyGradient,
};
pub use replay_buffer::{PrioritizedReplayBuffer, ReplayBuffer, ReplayBufferTrait};
pub use trpo::{TRPOConfig, TRPO};
pub use value::{DoubleDQN, QNetwork, ValueNetwork, DQN};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rl_config_default() {
        let config = RLConfig::default();
        assert_eq!(config.discount_factor, 0.99);
        assert_eq!(config.batch_size, 32);
        assert!(config.use_gae);
    }

    #[test]
    fn test_experience_batch_fields() {
        let batch = ExperienceBatch {
            states: Array2::zeros((4, 8)),
            actions: Array2::zeros((4, 2)),
            rewards: Array1::ones(4),
            next_states: Array2::zeros((4, 8)),
            dones: Array1::from_elem(4, false),
            info: None,
        };
        assert_eq!(batch.states.shape(), &[4, 8]);
        assert_eq!(batch.rewards.len(), 4);
    }

    #[test]
    fn test_loss_info_fields() {
        let info = LossInfo {
            policy_loss: Some(0.5),
            value_loss: Some(0.3),
            entropy_loss: Some(0.1),
            total_loss: 0.9,
            metrics: HashMap::new(),
        };
        assert!((info.total_loss - 0.9).abs() < 1e-6);
    }
}
