//! High-level reinforcement learning algorithm wrappers

use crate::error::{NeuralError, Result};
use crate::reinforcement::environments::Environment;
use crate::reinforcement::replay_buffer::{
    PrioritizedReplayBuffer, ReplayBuffer, ReplayBufferTrait,
};
use crate::reinforcement::{ExperienceBatch, LossInfo, RLAgent};
use scirs2_core::ndarray::prelude::*;

/// Sampled batch plus optional PER weights and indices
type SampledBatch = (ExperienceBatch, Option<Array1<f32>>, Option<Vec<usize>>);

/// Configuration shared across RL algorithms
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Total number of environment steps
    pub total_timesteps: usize,
    /// Steps between policy updates
    pub update_frequency: usize,
    /// Gradient steps per update
    pub gradient_steps: usize,
    /// Warm-up period before updates
    pub learning_starts: usize,
    /// Mini-batch size
    pub batch_size: usize,
    /// Replay-buffer capacity
    pub buffer_size: usize,
    /// Discount factor γ
    pub gamma: f32,
    /// Step size
    pub learning_rate: f32,
    /// Target network sync frequency (value-based only)
    pub target_update_freq: Option<usize>,
    /// Initial exploration rate
    pub exploration_initial: f32,
    /// Final exploration rate
    pub exploration_final: f32,
    /// Fraction of timesteps over which to anneal exploration
    pub exploration_fraction: f32,
    /// Episode logging frequency
    pub log_interval: usize,
    /// Optional evaluation frequency
    pub eval_freq: Option<usize>,
    /// Evaluation episodes
    pub n_eval_episodes: usize,
    /// Checkpoint save frequency
    pub save_freq: Option<usize>,
    /// Checkpoint save path
    pub save_path: Option<String>,
    /// Use prioritized experience replay
    pub use_prioritized_replay: bool,
    /// PER α exponent
    pub prioritized_replay_alpha: f32,
    /// PER β₀ initial weight
    pub prioritized_replay_beta0: f32,
    /// PER β schedule (e.g. "linear")
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
}

/// Results accumulated during training
#[derive(Debug)]
pub struct TrainingResults {
    pub episode_rewards: Vec<f32>,
    pub episode_lengths: Vec<usize>,
    pub losses: Vec<LossInfo>,
    pub eval_results: Vec<EvaluationResults>,
    pub training_time: f64,
    pub total_steps: usize,
}

/// Results from an evaluation rollout
#[derive(Debug, Clone)]
pub struct EvaluationResults {
    pub mean_reward: f32,
    pub std_reward: f32,
    pub mean_length: f32,
    pub min_reward: f32,
    pub max_reward: f32,
    pub n_episodes: usize,
}

/// Trait implemented by high-level RL algorithms
pub trait RLAlgorithm: Send + Sync {
    /// Run the training loop
    fn train(
        &mut self,
        env: &mut dyn Environment,
        config: &TrainingConfig,
    ) -> Result<TrainingResults>;

    /// Evaluate without exploration
    fn evaluate(&self, env: &mut dyn Environment, n_episodes: usize) -> Result<EvaluationResults>;

    /// Save model weights to `path`
    fn save(&self, path: &str) -> Result<()>;

    /// Load model weights from `path`
    fn load(&mut self, path: &str) -> Result<()>;

    /// Shared reference to the underlying RLAgent
    fn agent(&self) -> &dyn RLAgent;

    /// Mutable reference to the underlying RLAgent
    fn agent_mut(&mut self) -> &mut dyn RLAgent;
}

// ── Off-policy algorithm scaffold ────────────────────────────────────────────

/// Generic off-policy algorithm (DQN, SAC, TD3 …)
pub struct OffPolicyAlgorithm<A: RLAgent> {
    agent: A,
    replay_buffer: Option<ReplayBuffer>,
    prioritized_buffer: Option<PrioritizedReplayBuffer>,
}

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
        } else {
            None
        };
        Self {
            agent,
            replay_buffer,
            prioritized_buffer,
        }
    }

    fn add_to_buffer(
        &mut self,
        state: Array1<f32>,
        action: Array1<f32>,
        reward: f32,
        next_state: Array1<f32>,
        done: bool,
    ) -> Result<()> {
        if let Some(buffer) = &mut self.replay_buffer {
            buffer.add(state, action, reward, next_state, done)?;
        } else if let Some(buffer) = &mut self.prioritized_buffer {
            buffer.add(state, action, reward, next_state, done)?;
        }
        Ok(())
    }

    fn buffer_len(&self) -> usize {
        self.replay_buffer
            .as_ref()
            .map(|b| b.len())
            .or_else(|| self.prioritized_buffer.as_ref().map(|b| b.len()))
            .unwrap_or(0)
    }

    fn sample_batch(&mut self, batch_size: usize) -> Result<SampledBatch> {
        if let Some(buffer) = &mut self.replay_buffer {
            Ok((buffer.sample(batch_size)?, None, None))
        } else if let Some(buffer) = &mut self.prioritized_buffer {
            let (batch, weights, indices) = buffer.sample(batch_size)?;
            Ok((batch, Some(weights), Some(indices)))
        } else {
            Err(NeuralError::InvalidArgument(
                "No replay buffer configured".to_string(),
            ))
        }
    }

    fn update_priorities(&mut self, indices: &[usize], td_errors: &[f32]) -> Result<()> {
        if let Some(buffer) = &mut self.prioritized_buffer {
            buffer.update_priorities(indices, td_errors)?;
        }
        Ok(())
    }
}

impl<A: RLAgent + 'static> RLAlgorithm for OffPolicyAlgorithm<A> {
    fn train(
        &mut self,
        env: &mut dyn Environment,
        config: &TrainingConfig,
    ) -> Result<TrainingResults> {
        let start_time = std::time::Instant::now();
        let mut total_steps = 0usize;
        let mut episode_rewards: Vec<f32> = Vec::new();
        let mut episode_lengths: Vec<usize> = Vec::new();
        let mut losses: Vec<LossInfo> = Vec::new();
        let mut eval_results: Vec<EvaluationResults> = Vec::new();

        let mut state = env.reset()?;
        let mut episode_reward = 0.0f32;
        let mut episode_length = 0usize;

        // Exploration schedule
        let exploration = |t: usize| -> f32 {
            let fraction =
                (t as f32 / config.total_timesteps as f32).min(config.exploration_fraction);
            config.exploration_final
                + (config.exploration_initial - config.exploration_final)
                    * (1.0 - fraction / config.exploration_fraction.max(1e-8))
        };
        // Beta schedule for PER
        let beta_sched = |t: usize| -> f32 {
            config.prioritized_replay_beta0
                + (1.0 - config.prioritized_replay_beta0)
                    * (t as f32 / config.total_timesteps.max(1) as f32)
        };

        while total_steps < config.total_timesteps {
            let _exploration_rate = exploration(total_steps);
            let training = total_steps >= config.learning_starts;
            let action = self.agent.act(&state.view(), training)?;
            let (next_state, reward, done, _info) = env.step(&action)?;
            self.add_to_buffer(
                state.clone(),
                action.clone(),
                reward,
                next_state.clone(),
                done,
            )?;
            episode_reward += reward;
            episode_length += 1;
            total_steps += 1;

            if training
                && self.buffer_len() >= config.batch_size
                && total_steps.is_multiple_of(config.update_frequency)
            {
                for _ in 0..config.gradient_steps {
                    let (batch, _weights, indices) = self.sample_batch(config.batch_size)?;
                    let loss_info = self.agent.update(&batch)?;
                    if let Some(idxs) = indices {
                        let td_err = loss_info.total_loss;
                        let errs = vec![td_err; idxs.len()];
                        self.update_priorities(&idxs, &errs)?;
                    }
                    // Update PER beta
                    if let Some(buffer) = &mut self.prioritized_buffer {
                        buffer.update_beta(beta_sched(total_steps));
                    }
                    losses.push(loss_info);
                }
            }

            if done || episode_length >= 1000 {
                episode_rewards.push(episode_reward);
                episode_lengths.push(episode_length);
                state = env.reset()?;
                episode_reward = 0.0;
                episode_length = 0;
                if episode_rewards.len().is_multiple_of(config.log_interval) {
                    let recent = &episode_rewards[episode_rewards.len().saturating_sub(100)..];
                    let avg = recent.iter().sum::<f32>() / recent.len() as f32;
                    println!(
                        "Steps: {total_steps}, Episodes: {}, Avg Reward: {avg:.2}",
                        episode_rewards.len()
                    );
                }
            } else {
                state = next_state;
            }

            if let Some(eval_freq) = config.eval_freq {
                if total_steps.is_multiple_of(eval_freq) {
                    let res = self.evaluate(env, config.n_eval_episodes)?;
                    println!(
                        "Eval @ {total_steps}: mean_reward={:.2} ±{:.2}",
                        res.mean_reward, res.std_reward
                    );
                    eval_results.push(res);
                }
            }

            if let Some(save_freq) = config.save_freq {
                if total_steps.is_multiple_of(save_freq) {
                    if let Some(save_path) = &config.save_path {
                        let cp_path = format!("{save_path}/checkpoint_{total_steps}.bin");
                        let _ = self.save(&cp_path);
                    }
                }
            }
        }

        Ok(TrainingResults {
            episode_rewards,
            episode_lengths,
            losses,
            eval_results,
            training_time: start_time.elapsed().as_secs_f64(),
            total_steps,
        })
    }

    fn evaluate(&self, env: &mut dyn Environment, n_episodes: usize) -> Result<EvaluationResults> {
        let mut rewards = Vec::with_capacity(n_episodes);
        let mut lengths = Vec::with_capacity(n_episodes);
        for _ in 0..n_episodes {
            let mut state = env.reset()?;
            let mut episode_reward = 0.0f32;
            let mut episode_length = 0usize;
            loop {
                let action = self.agent.act(&state.view(), false)?;
                let (next_state, reward, done, _) = env.step(&action)?;
                episode_reward += reward;
                episode_length += 1;
                if done || episode_length >= 1000 {
                    break;
                }
                state = next_state;
            }
            rewards.push(episode_reward);
            lengths.push(episode_length);
        }
        let mean_reward = rewards.iter().sum::<f32>() / rewards.len().max(1) as f32;
        let variance = rewards
            .iter()
            .map(|r| (r - mean_reward).powi(2))
            .sum::<f32>()
            / rewards.len().max(1) as f32;
        let std_reward = variance.sqrt();
        let min_reward = rewards.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_reward = rewards.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mean_length = lengths.iter().sum::<usize>() as f32 / lengths.len().max(1) as f32;
        Ok(EvaluationResults {
            mean_reward,
            std_reward,
            mean_length,
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
            n_episodes,
        })
    }

    fn save(&self, path: &str) -> Result<()> {
        if let Some(parent) = std::path::Path::new(path).parent() {
            std::fs::create_dir_all(parent)?;
        }
        self.agent.save(path)
    }

    fn load(&mut self, path: &str) -> Result<()> {
        self.agent.load(path)
    }

    fn agent(&self) -> &dyn RLAgent {
        &self.agent as &dyn RLAgent
    }

    fn agent_mut(&mut self) -> &mut dyn RLAgent {
        &mut self.agent as &mut dyn RLAgent
    }
}

/// Minimal generic algorithm implementation (for testing / simple cases)
pub struct RLAlgorithmImpl {
    pub agent: Box<dyn RLAgent>,
    pub replay_buffer: Option<Box<dyn ReplayBufferTrait>>,
}

impl RLAlgorithm for RLAlgorithmImpl {
    fn train(
        &mut self,
        env: &mut dyn Environment,
        config: &TrainingConfig,
    ) -> Result<TrainingResults> {
        let start = std::time::Instant::now();
        let mut total_steps = 0usize;
        let mut episode_rewards = Vec::new();
        let mut episode_lengths = Vec::new();

        let mut state = env.reset()?;
        let mut episode_reward = 0.0f32;
        let mut episode_length = 0usize;

        while total_steps < config.total_timesteps {
            let action = self.agent.act(&state.view(), true)?;
            let (next_state, reward, done, _) = env.step(&action)?;
            if let Some(buf) = &mut self.replay_buffer {
                let _ = buf.add(state.clone(), action, reward, next_state.clone(), done);
            }
            episode_reward += reward;
            episode_length += 1;
            total_steps += 1;
            if done {
                episode_rewards.push(episode_reward);
                episode_lengths.push(episode_length);
                state = env.reset()?;
                episode_reward = 0.0;
                episode_length = 0;
            } else {
                state = next_state;
            }
        }

        Ok(TrainingResults {
            episode_rewards,
            episode_lengths,
            losses: Vec::new(),
            eval_results: Vec::new(),
            training_time: start.elapsed().as_secs_f64(),
            total_steps,
        })
    }

    fn evaluate(&self, env: &mut dyn Environment, n_episodes: usize) -> Result<EvaluationResults> {
        let mut rewards = Vec::new();
        let mut lengths = Vec::new();
        for _ in 0..n_episodes {
            let mut state = env.reset()?;
            let mut ep_reward = 0.0f32;
            let mut ep_len = 0usize;
            loop {
                let action = self.agent.act(&state.view(), false)?;
                let (next_state, reward, done, _) = env.step(&action)?;
                ep_reward += reward;
                ep_len += 1;
                if done || ep_len >= 1000 {
                    break;
                }
                state = next_state;
            }
            rewards.push(ep_reward);
            lengths.push(ep_len);
        }
        let mean = rewards.iter().sum::<f32>() / rewards.len().max(1) as f32;
        let var =
            rewards.iter().map(|r| (r - mean).powi(2)).sum::<f32>() / rewards.len().max(1) as f32;
        Ok(EvaluationResults {
            mean_reward: mean,
            std_reward: var.sqrt(),
            mean_length: lengths.iter().sum::<usize>() as f32 / lengths.len().max(1) as f32,
            min_reward: rewards
                .iter()
                .cloned()
                .fold(f32::INFINITY, f32::min)
                .max(f32::NEG_INFINITY),
            max_reward: rewards.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
            n_episodes,
        })
    }

    fn save(&self, path: &str) -> Result<()> {
        self.agent.save(path)
    }

    fn load(&mut self, path: &str) -> Result<()> {
        self.agent.load(path)
    }

    fn agent(&self) -> &dyn RLAgent {
        self.agent.as_ref()
    }

    fn agent_mut(&mut self) -> &mut dyn RLAgent {
        self.agent.as_mut()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_config_default() {
        let config = TrainingConfig::default();
        assert_eq!(config.total_timesteps, 1_000_000);
        assert_eq!(config.batch_size, 32);
        assert_eq!(config.gamma, 0.99);
    }

    #[test]
    fn test_exploration_schedule() {
        let config = TrainingConfig::default();
        let exploration_0 = config.exploration_initial;
        assert_eq!(exploration_0, 1.0);

        let frac = config.exploration_fraction;
        let steps_end = (config.total_timesteps as f32 * frac) as usize;
        let fraction = (steps_end as f32 / config.total_timesteps as f32).min(frac);
        let exploration_end = config.exploration_final
            + (config.exploration_initial - config.exploration_final)
                * (1.0 - fraction / frac.max(1e-8));
        assert!((exploration_end - config.exploration_final).abs() < 0.01);
    }

    #[test]
    fn test_training_results_fields() {
        let results = TrainingResults {
            episode_rewards: vec![1.0, 2.0],
            episode_lengths: vec![10, 20],
            losses: Vec::new(),
            eval_results: Vec::new(),
            training_time: 1.5,
            total_steps: 30,
        };
        assert_eq!(results.total_steps, 30);
        assert!((results.training_time - 1.5).abs() < 1e-6);
    }
}
