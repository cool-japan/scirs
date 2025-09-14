//! Actor-Critic Optimizers
//!
//! This module implements various actor-critic algorithms including A2C, A3C,
//! SAC (Soft Actor-Critic), and other modern actor-critic methods.

#![allow(dead_code)]

use super::{
    ActionDistribution, DistributionType, PolicyNetwork, RLOptimizationMetrics, RLOptimizerConfig,
    RLScheduler, TrajectoryBatch, ValueNetwork,
};
use crate::error::{OptimError, Result};
use ndarray::{Array1, Array2, ScalarOperand};
use num_traits::Float;
use rand::Rng;
use std::collections::HashMap;

/// Actor-Critic optimization methods
#[derive(Debug, Clone, Copy)]
pub enum ActorCriticMethod {
    /// Advantage Actor-Critic (A2C)
    A2C,

    /// Asynchronous Advantage Actor-Critic (A3C)
    A3C,

    /// Soft Actor-Critic (SAC)
    SAC,

    /// Twin Delayed Deep Deterministic Policy Gradients (TD3)
    TD3,

    /// Deep Deterministic Policy Gradients (DDPG)
    DDPG,

    /// Distributed Distributional Deterministic Policy Gradients (D4PG)
    D4PG,

    /// Maximum a Posteriori Policy Optimisation (MPO)
    MPO,
}

/// Actor-Critic configuration
#[derive(Debug, Clone)]
pub struct ActorCriticConfig<T: Float> {
    /// Base RL configuration
    pub base_config: RLOptimizerConfig<T>,

    /// Actor-Critic method
    pub method: ActorCriticMethod,

    /// SAC-specific configuration
    pub sac_config: SACConfig<T>,

    /// TD3-specific configuration
    pub td3_config: TD3Config<T>,

    /// DDPG-specific configuration
    pub ddpg_config: DDPGConfig<T>,

    /// Use target networks
    pub use_target_networks: bool,

    /// Target network soft update rate (tau)
    pub target_update_rate: T,

    /// Target network hard update frequency
    pub target_hard_update_freq: Option<usize>,

    /// Experience replay buffer size
    pub replay_buffer_size: usize,

    /// Enable prioritized experience replay
    pub prioritized_replay: bool,

    /// Prioritized replay alpha parameter
    pub per_alpha: T,

    /// Prioritized replay beta parameter
    pub per_beta: T,

    /// Number of critic networks (for twin critic methods)
    pub n_critics: usize,
}

/// SAC (Soft Actor-Critic) configuration
#[derive(Debug, Clone)]
pub struct SACConfig<T: Float> {
    /// Temperature parameter for entropy regularization
    pub temperature: T,

    /// Automatic entropy tuning
    pub auto_entropy_tuning: bool,

    /// Target entropy (for automatic tuning)
    pub target_entropy: Option<T>,

    /// Temperature learning rate
    pub temperature_lr: T,

    /// Use repameterization trick
    pub use_reparameterization: bool,

    /// Policy update frequency
    pub policy_update_freq: usize,

    /// Target network update frequency
    pub target_update_freq: usize,
}

/// TD3 (Twin Delayed DDPG) configuration
#[derive(Debug, Clone)]
pub struct TD3Config<T: Float> {
    /// Policy noise for target smoothing
    pub policy_noise: T,

    /// Noise clipping range
    pub noise_clip: T,

    /// Policy update delay
    pub policy_delay: usize,

    /// Exploration noise standard deviation
    pub exploration_noise: T,

    /// Action bounds for clipping
    pub action_bounds: Option<(T, T)>,
}

/// DDPG configuration
#[derive(Debug, Clone)]
pub struct DDPGConfig<T: Float> {
    /// Exploration noise standard deviation
    pub exploration_noise: T,

    /// Ornstein-Uhlenbeck noise parameters
    pub ou_noise_theta: T,
    pub ou_noise_sigma: T,

    /// Action bounds for clipping
    pub action_bounds: Option<(T, T)>,
}

impl<T: Float> Default for ActorCriticConfig<T> {
    fn default() -> Self {
        Self {
            base_config: RLOptimizerConfig::default(),
            method: ActorCriticMethod::A2C,
            sac_config: SACConfig::default(),
            td3_config: TD3Config::default(),
            ddpg_config: DDPGConfig::default(),
            use_target_networks: false,
            target_update_rate: T::from(0.005).unwrap(),
            target_hard_update_freq: None,
            replay_buffer_size: 100000,
            prioritized_replay: false,
            per_alpha: T::from(0.6).unwrap(),
            per_beta: T::from(0.4).unwrap(),
            n_critics: 1,
        }
    }
}

impl<T: Float> Default for SACConfig<T> {
    fn default() -> Self {
        Self {
            temperature: T::from(0.2).unwrap(),
            auto_entropy_tuning: true,
            target_entropy: None,
            temperature_lr: T::from(3e-4).unwrap(),
            use_reparameterization: true,
            policy_update_freq: 1,
            target_update_freq: 1,
        }
    }
}

impl<T: Float> Default for TD3Config<T> {
    fn default() -> Self {
        Self {
            policy_noise: T::from(0.2).unwrap(),
            noise_clip: T::from(0.5).unwrap(),
            policy_delay: 2,
            exploration_noise: T::from(0.1).unwrap(),
            action_bounds: Some((T::from(-1.0).unwrap(), T::from(1.0).unwrap())),
        }
    }
}

impl<T: Float> Default for DDPGConfig<T> {
    fn default() -> Self {
        Self {
            exploration_noise: T::from(0.1).unwrap(),
            ou_noise_theta: T::from(0.15).unwrap(),
            ou_noise_sigma: T::from(0.2).unwrap(),
            action_bounds: Some((T::from(-1.0).unwrap(), T::from(1.0).unwrap())),
        }
    }
}

/// Actor-Critic optimizer
pub struct ActorCriticOptimizer<T: Float, P: PolicyNetwork<T>, V: ValueNetwork<T>> {
    /// Configuration
    config: ActorCriticConfig<T>,

    /// Actor (policy) network
    actor: P,

    /// Critic (value) networks
    critics: Vec<V>,

    /// Target networks (if enabled)
    target_actor: Option<P>,
    target_critics: Option<Vec<V>>,

    /// Temperature parameter for SAC
    temperature: T,

    /// Learning rate schedulers
    actor_scheduler: Option<RLScheduler<T>>,
    critic_scheduler: Option<RLScheduler<T>>,
    temperature_scheduler: Option<RLScheduler<T>>,

    /// Optimization metrics
    metrics: ActorCriticMetrics<T>,

    /// Update counters
    update_count: usize,
    policy_update_count: usize,

    /// Experience replay buffer
    replay_buffer: ExperienceReplayBuffer<T>,

    /// Ornstein-Uhlenbeck noise state (for DDPG)
    ou_noise_state: Option<Array1<T>>,
}

/// Actor-Critic specific metrics
#[derive(Debug, Clone)]
pub struct ActorCriticMetrics<T: Float> {
    /// Base RL metrics
    pub base_metrics: RLOptimizationMetrics<T>,

    /// Actor loss
    pub actor_loss: T,

    /// Critic loss(es)
    pub critic_losses: Vec<T>,

    /// Temperature (for SAC)
    pub temperature: Option<T>,

    /// Temperature loss (for SAC)
    pub temperature_loss: Option<T>,

    /// Q-values statistics
    pub q_values_mean: T,
    pub q_values_std: T,

    /// Target Q-values statistics
    pub target_q_mean: T,
    pub target_q_std: T,

    /// Policy entropy
    pub policy_entropy: T,

    /// Critic gradient norms
    pub critic_grad_norms: Vec<T>,

    /// Experience replay metrics
    pub replay_buffer_size: usize,
    pub replay_sampling_time: Option<std::time::Duration>,
}

impl<T: Float> Default for ActorCriticMetrics<T> {
    fn default() -> Self {
        Self {
            base_metrics: RLOptimizationMetrics::default(),
            actor_loss: T::zero(),
            critic_losses: vec![T::zero()],
            temperature: None,
            temperature_loss: None,
            q_values_mean: T::zero(),
            q_values_std: T::zero(),
            target_q_mean: T::zero(),
            target_q_std: T::zero(),
            policy_entropy: T::zero(),
            critic_grad_norms: vec![T::zero()],
            replay_buffer_size: 0,
            replay_sampling_time: None,
        }
    }
}

/// Experience replay buffer entry
#[derive(Debug, Clone)]
pub struct Experience<T: Float> {
    /// State (observation)
    pub state: Array1<T>,

    /// Action taken
    pub action: Array1<T>,

    /// Reward received
    pub reward: T,

    /// Next state
    pub next_state: Array1<T>,

    /// Done flag
    pub done: bool,

    /// Priority (for prioritized replay)
    pub priority: T,

    /// Additional info
    pub info: HashMap<String, T>,
}

/// Experience replay buffer
pub struct ExperienceReplayBuffer<T: Float> {
    /// Buffer storage
    buffer: Vec<Experience<T>>,

    /// Maximum buffer size
    maxsize: usize,

    /// Current position in buffer
    position: usize,

    /// Whether the buffer is full
    is_full: bool,

    /// Prioritized replay parameters
    alpha: T,
    beta: T,

    /// Priority sum tree (for efficient sampling)
    priority_tree: Option<Vec<T>>,
}

impl<T: Float + Send + Sync> ExperienceReplayBuffer<T> {
    /// Create a new experience replay buffer
    pub fn new(maxsize: usize, alpha: T, beta: T) -> Self {
        Self {
            buffer: Vec::with_capacity(maxsize),
            maxsize,
            position: 0,
            is_full: false,
            alpha,
            beta,
            priority_tree: None,
        }
    }

    /// Add experience to buffer
    pub fn add(&mut self, experience: Experience<T>) {
        if self.buffer.len() < self.maxsize {
            self.buffer.push(experience);
        } else {
            self.buffer[self.position] = experience;
            self.is_full = true;
        }

        self.position = (self.position + 1) % self.maxsize;
    }

    /// Sample batch from buffer
    pub fn sample(&self, batchsize: usize) -> Vec<Experience<T>> {
        let available_size = if self.is_full {
            self.maxsize
        } else {
            self.buffer.len()
        };
        let sample_size = batchsize.min(available_size);

        let mut samples = Vec::new();
        for _ in 0..sample_size {
            let idx = scirs2_core::random::rng().gen_range(0..available_size);
            samples.push(self.buffer[idx].clone());
        }

        samples
    }

    /// Get buffer size
    pub fn len(&self) -> usize {
        if self.is_full {
            self.maxsize
        } else {
            self.buffer.len()
        }
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
}

impl<
        T: Float + num_traits::FromPrimitive + std::iter::Sum + Send + Sync + ScalarOperand,
        P: PolicyNetwork<T>,
        V: ValueNetwork<T>,
    > ActorCriticOptimizer<T, P, V>
{
    /// Create a new Actor-Critic optimizer
    pub fn new(config: ActorCriticConfig<T>, actor: P, critics: Vec<V>) -> Result<Self> {
        if critics.is_empty() {
            return Err(OptimError::InvalidConfig(
                "At least one critic required".to_string(),
            ));
        }

        let replay_buffer = ExperienceReplayBuffer::new(
            config.replay_buffer_size,
            config.per_alpha,
            config.per_beta,
        );

        let temperature = config.sac_config.temperature;

        Ok(Self {
            config,
            actor,
            critics,
            target_actor: None,
            target_critics: None,
            temperature,
            actor_scheduler: None,
            critic_scheduler: None,
            temperature_scheduler: None,
            metrics: ActorCriticMetrics::default(),
            update_count: 0,
            policy_update_count: 0,
            replay_buffer,
            ou_noise_state: None,
        })
    }

    /// Update using experience replay
    pub fn update_from_replay(&mut self, batchsize: usize) -> Result<ActorCriticMetrics<T>> {
        if self.replay_buffer.len() < batchsize {
            return Err(OptimError::InvalidConfig(
                "Not enough experiences in buffer".to_string(),
            ));
        }

        let experiences = self.replay_buffer.sample(batchsize);

        match self.config.method {
            ActorCriticMethod::SAC => self.update_sac(&experiences),
            ActorCriticMethod::TD3 => self.update_td3(&experiences),
            ActorCriticMethod::DDPG => self.update_ddpg(&experiences),
            ActorCriticMethod::A2C => Err(OptimError::InvalidConfig(
                "Method not implemented".to_string(),
            )),
            _ => Err(OptimError::InvalidConfig(
                "Method not implemented".to_string(),
            )),
        }
    }

    /// Update using trajectory (on-policy methods)
    pub fn update_from_trajectory(
        &mut self,
        trajectory: TrajectoryBatch<T>,
    ) -> Result<ActorCriticMetrics<T>> {
        match self.config.method {
            ActorCriticMethod::A2C => self.update_a2c(trajectory),
            ActorCriticMethod::A3C => Err(OptimError::InvalidConfig(
                "Method requires experience replay".to_string(),
            )),
            _ => Err(OptimError::InvalidConfig(
                "Method requires experience replay".to_string(),
            )),
        }
    }

    /// SAC update
    fn update_sac(&mut self, experiences: &[Experience<T>]) -> Result<ActorCriticMetrics<T>> {
        let _batch_size = experiences.len();

        // Extract batch data
        let states = self.extract_states(experiences)?;
        let actions = self.extract_actions(experiences)?;
        let rewards = self.extract_rewards(experiences)?;
        let next_states = self.extract_next_states(experiences)?;
        let dones = self.extract_dones(experiences)?;

        // Update critics
        let mut critic_losses = Vec::new();
        for (_i, critic) in self.critics.iter().enumerate() {
            let q_values = self.compute_q_values(critic, &states, &actions)?;
            let targetq = self.compute_target_q_sac(&next_states, &rewards, &dones)?;

            let critic_loss = self.compute_critic_loss(&q_values, &targetq)?;
            critic_losses.push(critic_loss);

            // Update critic (simplified)
            // In practice, compute gradients and update parameters
        }

        // Update actor (policy)
        let actor_loss = self.compute_actor_loss_sac(&states)?;

        // Update temperature (if auto-tuning)
        let temperature_loss = if self.config.sac_config.auto_entropy_tuning {
            Some(self.update_temperature_sac(&states)?)
        } else {
            None
        };

        // Update target networks
        if self.config.use_target_networks {
            self.soft_update_targets()?;
        }

        // Update metrics
        self.metrics.actor_loss = actor_loss;
        self.metrics.critic_losses = critic_losses;
        self.metrics.temperature = Some(self.temperature);
        self.metrics.temperature_loss = temperature_loss;
        self.metrics.replay_buffer_size = self.replay_buffer.len();

        self.update_count += 1;

        Ok(self.metrics.clone())
    }

    /// TD3 update
    fn update_td3(&mut self, experiences: &[Experience<T>]) -> Result<ActorCriticMetrics<T>> {
        let _batch_size = experiences.len();

        // Extract batch data
        let states = self.extract_states(experiences)?;
        let actions = self.extract_actions(experiences)?;
        let rewards = self.extract_rewards(experiences)?;
        let next_states = self.extract_next_states(experiences)?;
        let dones = self.extract_dones(experiences)?;

        // Update critics (always update both critics in TD3)
        let mut critic_losses = Vec::new();
        if self.critics.len() >= 2 {
            // Compute target actions with noise
            let target_actions = if let Some(ref target_actor) = self.target_actor {
                let target_action_dist = target_actor.get_action_distribution(&next_states)?;
                let mut target_actions =
                    self.sample_actions_from_distribution(&target_action_dist)?;

                // Add target policy smoothing noise (TD3 feature)
                for action in target_actions.iter_mut() {
                    let noise = T::from(scirs2_core::random::rng().random_f64() - 0.5).unwrap()
                        * T::from(2.0).unwrap()
                        * self.config.td3_config.policy_noise;
                    let clipped_noise = noise
                        .max(-self.config.td3_config.noise_clip)
                        .min(self.config.td3_config.noise_clip);
                    *action = *action + clipped_noise;

                    // Clip actions to bounds
                    if let Some((min_action, max_action)) = self.config.td3_config.action_bounds {
                        *action = action.max(min_action).min(max_action);
                    }
                }

                target_actions
            } else {
                // Fallback if no target actor
                actions.clone()
            };

            // Compute target Q-values using twin critics and minimum
            let target_q1 = if let Some(ref target_critics) = self.target_critics {
                if target_critics.len() >= 2 {
                    self.compute_q_values(&target_critics[0], &next_states, &target_actions)?
                } else {
                    self.compute_q_values(&self.critics[0], &next_states, &target_actions)?
                }
            } else {
                self.compute_q_values(&self.critics[0], &next_states, &target_actions)?
            };

            let target_q2 = if let Some(ref target_critics) = self.target_critics {
                if target_critics.len() >= 2 {
                    self.compute_q_values(&target_critics[1], &next_states, &target_actions)?
                } else {
                    self.compute_q_values(&self.critics[1], &next_states, &target_actions)?
                }
            } else {
                self.compute_q_values(&self.critics[1], &next_states, &target_actions)?
            };

            // Take minimum (TD3 feature for overestimation bias reduction)
            let mut min_target_q = Array1::zeros(target_q1.len());
            for i in 0..target_q1.len() {
                min_target_q[i] = target_q1[i].min(target_q2[i]);
            }

            // Compute TD targets
            let gamma = self.config.base_config.discount_factor;
            let mut td_targets = Array1::zeros(rewards.len());
            for i in 0..rewards.len() {
                td_targets[i] = rewards[i]
                    + gamma * min_target_q[i] * T::from(if dones[i] { 0.0 } else { 1.0 }).unwrap();
            }

            // Update both critics
            for (_i, critic) in self.critics.iter().enumerate().take(2) {
                let q_values = self.compute_q_values(critic, &states, &actions)?;
                let critic_loss = self.compute_critic_loss(&q_values, &td_targets)?;
                critic_losses.push(critic_loss);
            }
        }

        // Update actor with delayed policy updates (TD3 feature)
        let actor_loss = if self.update_count % self.config.td3_config.policy_delay == 0 {
            self.compute_actor_loss_td3(&states)?
        } else {
            T::zero()
        };

        // Update target networks
        if self.config.use_target_networks {
            self.soft_update_targets()?;
        }

        // Update metrics
        self.metrics.actor_loss = actor_loss;
        self.metrics.critic_losses = critic_losses;
        self.metrics.replay_buffer_size = self.replay_buffer.len();

        self.update_count += 1;

        Ok(self.metrics.clone())
    }

    /// DDPG update
    fn update_ddpg(&mut self, experiences: &[Experience<T>]) -> Result<ActorCriticMetrics<T>> {
        let _batch_size = experiences.len();

        // Extract batch data
        let states = self.extract_states(experiences)?;
        let actions = self.extract_actions(experiences)?;
        let rewards = self.extract_rewards(experiences)?;
        let next_states = self.extract_next_states(experiences)?;
        let dones = self.extract_dones(experiences)?;

        // Update critic
        let target_actions = if let Some(ref target_actor) = self.target_actor {
            let target_action_dist = target_actor.get_action_distribution(&next_states)?;
            self.sample_actions_from_distribution(&target_action_dist)?
        } else {
            // Use current actor if no target
            let action_dist = self.actor.get_action_distribution(&next_states)?;
            self.sample_actions_from_distribution(&action_dist)?
        };

        let targetq = if let Some(ref target_critics) = self.target_critics {
            self.compute_q_values(&target_critics[0], &next_states, &target_actions)?
        } else {
            self.compute_q_values(&self.critics[0], &next_states, &target_actions)?
        };

        // Compute TD targets
        let gamma = self.config.base_config.discount_factor;
        let mut td_targets = Array1::zeros(rewards.len());
        for i in 0..rewards.len() {
            td_targets[i] = rewards[i]
                + gamma * targetq[i] * T::from(if dones[i] { 0.0 } else { 1.0 }).unwrap();
        }

        let q_values = self.compute_q_values(&self.critics[0], &states, &actions)?;
        let critic_loss = self.compute_critic_loss(&q_values, &td_targets)?;

        // Update actor (DDPG deterministic policy gradient)
        let actor_loss = self.compute_actor_loss_ddpg(&states)?;

        // Update target networks
        if self.config.use_target_networks {
            self.soft_update_targets()?;
        }

        // Add Ornstein-Uhlenbeck noise for exploration (DDPG feature)
        self.update_ou_noise()?;

        // Update metrics
        self.metrics.actor_loss = actor_loss;
        self.metrics.critic_losses = vec![critic_loss];
        self.metrics.replay_buffer_size = self.replay_buffer.len();

        self.update_count += 1;

        Ok(self.metrics.clone())
    }

    /// Compute actor loss for TD3
    fn compute_actor_loss_td3(&self, states: &Array2<T>) -> Result<T> {
        // TD3 actor loss: maximize Q1(s, π(s))
        let action_dist = self.actor.get_action_distribution(states)?;
        let actions = self.sample_actions_from_distribution(&action_dist)?;

        // Use only the first critic for actor update (TD3 style)
        let q_values = self.compute_q_values(&self.critics[0], states, &actions)?;

        // Actor loss: negative Q-values (since we want to maximize Q)
        let actor_loss =
            -q_values.iter().copied().sum::<T>() / T::from(q_values.len()).unwrap_or(T::zero());

        Ok(actor_loss)
    }

    /// Compute actor loss for DDPG
    fn compute_actor_loss_ddpg(&self, states: &Array2<T>) -> Result<T> {
        // DDPG actor loss: maximize Q(s, π(s))
        let action_dist = self.actor.get_action_distribution(states)?;
        let actions = self.sample_actions_from_distribution(&action_dist)?;

        let q_values = self.compute_q_values(&self.critics[0], states, &actions)?;

        // Actor loss: negative Q-values
        let actor_loss =
            -q_values.iter().copied().sum::<T>() / T::from(q_values.len()).unwrap_or(T::zero());

        Ok(actor_loss)
    }

    /// Update Ornstein-Uhlenbeck noise for DDPG exploration
    fn update_ou_noise(&mut self) -> Result<()> {
        if let Some(ref mut ou_state) = self.ou_noise_state {
            let theta = self.config.ddpg_config.ou_noise_theta;
            let sigma = self.config.ddpg_config.ou_noise_sigma;

            // OU noise update: dx = theta * (0 - x) * dt + sigma * dW
            for noise in ou_state.iter_mut() {
                let dx = -theta * *noise
                    + sigma * T::from(scirs2_core::random::rng().random_f64() - 0.5).unwrap();
                *noise = *noise + dx;
            }
        }

        Ok(())
    }

    /// A2C update from trajectory
    fn update_a2c(&mut self, trajectory: TrajectoryBatch<T>) -> Result<ActorCriticMetrics<T>> {
        // Compute advantages
        let mut traj_copy = trajectory;
        let next_value = if let Some(critic) = self.critics.first() {
            let last_obs = traj_copy.observations.slice(s![-1.., ..]).to_owned();
            let mut last_obs_batch = Array2::zeros((1, last_obs.ncols()));
            last_obs_batch.row_mut(0).assign(&last_obs.row(0));
            critic.evaluate_value(&last_obs_batch)?[0]
        } else {
            T::zero()
        };

        traj_copy.compute_advantages(
            self.config.base_config.discount_factor,
            self.config.base_config.gae_lambda,
            next_value,
        )?;

        // Update critic
        let values = self.critics[0].evaluate_value(&traj_copy.observations)?;
        let critic_loss = (&values - &traj_copy.returns)
            .mapv(|x| x * x)
            .mean()
            .unwrap_or(T::zero());

        // Update actor
        let policy_eval = self
            .actor
            .evaluate_actions(&traj_copy.observations, &traj_copy.actions)?;
        let actor_loss = -(policy_eval.log_probs * traj_copy.advantages)
            .mean()
            .unwrap_or(T::zero());

        // Update metrics
        self.metrics.actor_loss = actor_loss;
        self.metrics.critic_losses = vec![critic_loss];
        self.metrics.policy_entropy = policy_eval.entropy.iter().copied().sum::<T>()
            / T::from(policy_eval.entropy.len()).unwrap_or(T::zero());

        self.update_count += 1;

        Ok(self.metrics.clone())
    }

    /// A3C update (asynchronous)
    fn update_a3c(&mut self, trajectory: TrajectoryBatch<T>) -> Result<ActorCriticMetrics<T>> {
        // A3C is similar to A2C but with asynchronous updates
        self.update_a2c(trajectory)
    }

    /// A2C update from experiences (for compatibility)
    fn update_a2c_from_experiences(
        &mut self,
        experiences: &[Experience<T>],
    ) -> Result<ActorCriticMetrics<T>> {
        // Convert experiences to trajectory format
        let trajectory = self.experiences_to_trajectory(experiences)?;
        self.update_a2c(trajectory)
    }

    /// Add experience to replay buffer
    pub fn add_experience(&mut self, experience: Experience<T>) {
        self.replay_buffer.add(experience);
    }

    /// Get current metrics
    pub fn get_metrics(&self) -> &ActorCriticMetrics<T> {
        &self.metrics
    }

    // Helper methods

    fn extract_states(&self, experiences: &[Experience<T>]) -> Result<Array2<T>> {
        if experiences.is_empty() {
            return Err(OptimError::InvalidConfig(
                "Empty experience batch".to_string(),
            ));
        }

        let batchsize = experiences.len();
        let state_dim = experiences[0].state.len();
        let mut states = Array2::zeros((batchsize, state_dim));

        for (i, exp) in experiences.iter().enumerate() {
            states.row_mut(i).assign(&exp.state);
        }

        Ok(states)
    }

    fn extract_actions(&self, experiences: &[Experience<T>]) -> Result<Array2<T>> {
        let batchsize = experiences.len();
        let action_dim = experiences[0].action.len();
        let mut actions = Array2::zeros((batchsize, action_dim));

        for (i, exp) in experiences.iter().enumerate() {
            actions.row_mut(i).assign(&exp.action);
        }

        Ok(actions)
    }

    fn extract_rewards(&self, experiences: &[Experience<T>]) -> Result<Array1<T>> {
        let rewards: Vec<T> = experiences.iter().map(|exp| exp.reward).collect();
        Ok(Array1::from_vec(rewards))
    }

    fn extract_next_states(&self, experiences: &[Experience<T>]) -> Result<Array2<T>> {
        let batchsize = experiences.len();
        let state_dim = experiences[0].next_state.len();
        let mut next_states = Array2::zeros((batchsize, state_dim));

        for (i, exp) in experiences.iter().enumerate() {
            next_states.row_mut(i).assign(&exp.next_state);
        }

        Ok(next_states)
    }

    fn extract_dones(&self, experiences: &[Experience<T>]) -> Result<Array1<bool>> {
        let dones: Vec<bool> = experiences.iter().map(|exp| exp.done).collect();
        Ok(Array1::from_vec(dones))
    }

    fn compute_q_values(
        &self,
        critic: &V,
        states: &Array2<T>,
        _actions: &Array2<T>,
    ) -> Result<Array1<T>> {
        // Simplified Q-value computation
        critic.evaluate_value(states)
    }

    fn compute_target_q_sac(
        &self,
        _next_states: &Array2<T>,
        rewards: &Array1<T>,
        _dones: &Array1<bool>,
    ) -> Result<Array1<T>> {
        // Simplified target Q computation for SAC
        Ok(rewards.clone())
    }

    fn compute_critic_loss(&self, q_values: &Array1<T>, targetq: &Array1<T>) -> Result<T> {
        Ok((q_values - targetq)
            .mapv(|x| x * x)
            .mean()
            .unwrap_or(T::zero()))
    }

    fn compute_actor_loss_sac(&self, states: &Array2<T>) -> Result<T> {
        // SAC actor loss: minimize Q(s,a) - alpha * H(π(·|s))
        let action_dist = self.actor.get_action_distribution(states)?;

        // Sample actions from current policy
        let sampled_actions = self.sample_actions_from_distribution(&action_dist)?;

        // Compute Q-values for sampled actions
        let q_values = if self.critics.len() >= 2 {
            // Use minimum of twin critics (TD3/SAC style)
            let q1 = self.compute_q_values(&self.critics[0], states, &sampled_actions)?;
            let q2 = self.compute_q_values(&self.critics[1], states, &sampled_actions)?;

            // Element-wise minimum
            let mut min_q = Array1::zeros(q1.len());
            for i in 0..q1.len() {
                min_q[i] = q1[i].min(q2[i]);
            }
            min_q
        } else {
            self.compute_q_values(&self.critics[0], states, &sampled_actions)?
        };

        // Compute log probabilities of sampled actions
        let log_probs = self.compute_log_probabilities(&action_dist, &sampled_actions)?;

        // SAC actor loss: -E[Q(s,a) - α log π(a|s)]
        let entropy_term = log_probs * self.temperature;
        let actor_objective = q_values - entropy_term;
        let actor_loss = -actor_objective.iter().copied().sum::<T>()
            / T::from(actor_objective.len()).unwrap_or(T::zero());

        Ok(actor_loss)
    }

    fn update_temperature_sac(&mut self, states: &Array2<T>) -> Result<T> {
        if !self.config.sac_config.auto_entropy_tuning {
            return Ok(T::zero());
        }

        // Compute current policy entropy
        let action_dist = self.actor.get_action_distribution(states)?;
        let sampled_actions = self.sample_actions_from_distribution(&action_dist)?;
        let log_probs = self.compute_log_probabilities(&action_dist, &sampled_actions)?;
        let current_entropy =
            -log_probs.iter().copied().sum::<T>() / T::from(log_probs.len()).unwrap_or(T::zero());

        // Target entropy (typically -action_dim for continuous actions)
        let target_entropy = self
            .config
            .sac_config
            .target_entropy
            .unwrap_or(-T::from(sampled_actions.ncols()).unwrap());

        // Temperature loss: α * (target_entropy - current_entropy)
        let temperature_loss = self.temperature * (target_entropy - current_entropy);

        // Update temperature (simplified gradient step)
        let temp_lr = self.config.sac_config.temperature_lr;
        let temp_gradient = target_entropy - current_entropy;
        self.temperature =
            (self.temperature - temp_lr * temp_gradient).max(T::from(0.001).unwrap());

        Ok(temperature_loss)
    }

    /// Sample actions from action distribution
    fn sample_actions_from_distribution(
        &self,
        action_dist: &ActionDistribution<T>,
    ) -> Result<Array2<T>> {
        match action_dist.distribution_type {
            DistributionType::Gaussian => {
                if let (Some(ref mean), Some(ref std)) = (&action_dist.mean, &action_dist.std) {
                    let mut actions = mean.clone();

                    // Add Gaussian noise: action = mean + std * noise
                    for ((action, &m), &s) in actions.iter_mut().zip(mean.iter()).zip(std.iter()) {
                        let noise = T::from(scirs2_core::random::rng().random_f64() - 0.5).unwrap()
                            * T::from(2.0).unwrap(); // Simplified noise
                        *action = m + s * noise;
                    }

                    Ok(actions)
                } else {
                    Err(OptimError::InvalidConfig(
                        "Invalid Gaussian distribution".to_string(),
                    ))
                }
            }
            DistributionType::Categorical => {
                if let Some(ref logits) = action_dist.logits {
                    // Sample from categorical distribution
                    let mut actions = Array2::zeros(logits.dim());

                    for i in 0..logits.nrows() {
                        // Convert logits to probabilities (simplified)
                        let row = logits.row(i);
                        let max_logit = row.iter().fold(T::neg_infinity(), |acc, &x| acc.max(x));
                        let exp_logits: Vec<T> =
                            row.iter().map(|&x| (x - max_logit).exp()).collect();
                        let sum_exp: T = exp_logits.iter().cloned().sum();

                        // Sample action (simplified - take argmax for now)
                        let mut max_idx = 0;
                        let mut max_prob = T::zero();
                        for (j, &prob) in exp_logits.iter().enumerate() {
                            let normalized_prob = prob / sum_exp;
                            if normalized_prob > max_prob {
                                max_prob = normalized_prob;
                                max_idx = j;
                            }
                        }

                        actions[[i, max_idx]] = T::one();
                    }

                    Ok(actions)
                } else {
                    Err(OptimError::InvalidConfig(
                        "Invalid categorical distribution".to_string(),
                    ))
                }
            }
            _ => Err(OptimError::InvalidConfig(
                "Unsupported distribution type".to_string(),
            )),
        }
    }

    /// Compute log probabilities of actions under distribution
    fn compute_log_probabilities(
        &self,
        action_dist: &ActionDistribution<T>,
        actions: &Array2<T>,
    ) -> Result<Array1<T>> {
        match action_dist.distribution_type {
            DistributionType::Gaussian => {
                if let (Some(ref mean), Some(ref std)) = (&action_dist.mean, &action_dist.std) {
                    let mut log_probs = Array1::zeros(actions.nrows());

                    for i in 0..actions.nrows() {
                        let mut log_prob = T::zero();

                        for j in 0..actions.ncols() {
                            let action = actions[[i, j]];
                            let mu = mean[[i, j]];
                            let sigma = std[[i, j]];

                            // Log probability of Gaussian: -0.5 * ((x - μ) / σ)² - log(σ) - 0.5 * log(2π)
                            let normalized_diff = (action - mu) / sigma;
                            let log_prob_term =
                                -T::from(0.5).unwrap() * normalized_diff * normalized_diff
                                    - sigma.ln()
                                    - T::from(0.5 * 2.0 * std::f64::consts::PI).unwrap().ln();
                            log_prob = log_prob + log_prob_term;
                        }

                        log_probs[i] = log_prob;
                    }

                    Ok(log_probs)
                } else {
                    Err(OptimError::InvalidConfig(
                        "Invalid Gaussian distribution".to_string(),
                    ))
                }
            }
            DistributionType::Categorical => {
                if let Some(ref logits) = action_dist.logits {
                    let mut log_probs = Array1::zeros(actions.nrows());

                    for i in 0..actions.nrows() {
                        // Find the action index (one-hot encoded)
                        let mut action_idx = 0;
                        for j in 0..actions.ncols() {
                            if actions[[i, j]] > T::from(0.5).unwrap() {
                                action_idx = j;
                                break;
                            }
                        }

                        // Compute log softmax
                        let row = logits.row(i);
                        let max_logit = row.iter().fold(T::neg_infinity(), |acc, &x| acc.max(x));
                        let log_sum_exp = (row.iter().map(|&x| (x - max_logit).exp()).sum::<T>())
                            .ln()
                            + max_logit;

                        log_probs[i] = logits[[i, action_idx]] - log_sum_exp;
                    }

                    Ok(log_probs)
                } else {
                    Err(OptimError::InvalidConfig(
                        "Invalid categorical distribution".to_string(),
                    ))
                }
            }
            _ => Err(OptimError::InvalidConfig(
                "Unsupported distribution type".to_string(),
            )),
        }
    }

    fn soft_update_targets(&mut self) -> Result<()> {
        let tau = self.config.target_update_rate;
        let one_minus_tau = T::one() - tau;

        // Update target critics
        if let Some(ref mut target_critics) = self.target_critics {
            for (target_critic, online_critic) in target_critics.iter_mut().zip(self.critics.iter())
            {
                let online_params = online_critic.get_parameters();
                let mut target_params = target_critic.get_parameters();

                // Soft update: target = tau * online + (1 - tau) * target
                for (param_name, online_param) in online_params {
                    if let Some(target_param) = target_params.get_mut(&param_name) {
                        *target_param =
                            &(target_param.clone() * one_minus_tau) + &(online_param * tau);
                    }
                }

                target_critic.update_parameters(&target_params)?;
            }
        }

        // Update target actor (for DDPG/TD3)
        if let Some(ref mut target_actor) = self.target_actor {
            let online_params = self.actor.get_parameters();
            let mut target_params = target_actor.get_parameters();

            for (param_name, online_param) in online_params {
                if let Some(target_param) = target_params.get_mut(&param_name) {
                    *target_param = &(target_param.clone() * one_minus_tau) + &(online_param * tau);
                }
            }

            target_actor.update_parameters(&target_params)?;
        }

        Ok(())
    }

    fn experiences_to_trajectory(
        &self,
        experiences: &[Experience<T>],
    ) -> Result<TrajectoryBatch<T>> {
        let states = self.extract_states(experiences)?;
        let actions = self.extract_actions(experiences)?;
        let rewards = self.extract_rewards(experiences)?;
        let dones = self.extract_dones(experiences)?;

        // Create dummy log_probs and values
        let log_probs = Array1::zeros(experiences.len());
        let values = Array1::zeros(experiences.len());

        TrajectoryBatch::new(states, actions, log_probs, rewards, values, dones)
    }
}

// Import slice syntax
use ndarray::s;
// use statrs::statistics::Statistics; // statrs not available
