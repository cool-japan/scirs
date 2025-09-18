//! Policy Gradient Optimizers
//!
//! This module implements various policy gradient methods including REINFORCE,
//! PPO (Proximal Policy Optimization), TRPO (Trust Region Policy Optimization),
//! and other modern policy gradient algorithms.

#![allow(dead_code)]

use super::{
    PolicyNetwork, RLOptimizationMetrics, RLOptimizerConfig, RLScheduler, ScheduleType,
    TrajectoryBatch, ValueNetwork,
};
use crate::error::{OptimError, Result};
use ndarray::{Array1, Array2, ScalarOperand};
use num_traits::Float;
use std::collections::HashMap;

/// Policy gradient optimization methods
#[derive(Debug, Clone, Copy)]
pub enum PolicyGradientMethod {
    /// REINFORCE algorithm
    Reinforce,

    /// Actor-Critic
    ActorCritic,

    /// Proximal Policy Optimization (PPO) with clipped surrogate
    PPOClip,

    /// PPO with adaptive KL penalty
    PPOAdaptiveKL,

    /// Trust Region Policy Optimization (TRPO)
    TRPO,

    /// Importance Weighted Actor-Learner Architecture (IMPALA)
    IMPALA,

    /// Asynchronous Advantage Actor-Critic (A3C)
    A3C,
}

/// Policy gradient optimizer configuration
#[derive(Debug, Clone)]
pub struct PolicyGradientConfig<T: Float> {
    /// Base RL configuration
    pub base_config: RLOptimizerConfig<T>,

    /// Policy gradient method
    pub method: PolicyGradientMethod,

    /// PPO-specific parameters
    pub ppo_config: PPOConfig<T>,

    /// TRPO-specific parameters
    pub trpo_config: TRPOConfig<T>,

    /// Learning rate scheduler for policy
    pub policy_scheduler: Option<RLScheduler<T>>,

    /// Learning rate scheduler for value function
    pub value_scheduler: Option<RLScheduler<T>>,

    /// Use baseline (value function) for variance reduction
    pub use_baseline: bool,

    /// Enable importance sampling for off-policy updates
    pub importance_sampling: bool,

    /// Maximum importance sampling ratio
    pub max_is_ratio: T,
}

/// PPO-specific configuration
#[derive(Debug, Clone)]
pub struct PPOConfig<T: Float> {
    /// Clipping parameter
    pub clip_epsilon: T,

    /// Dual clipping (clip both positive and negative advantages)
    pub dual_clip: bool,

    /// Value function clipping
    pub value_clip: bool,

    /// Value clipping range
    pub value_clip_range: T,

    /// Target KL divergence for adaptive methods
    pub target_kl: T,

    /// KL coefficient for adaptive penalty
    pub kl_coeff: T,

    /// KL coefficient adaptation factor
    pub kl_coeff_adapt_factor: T,

    /// Early stopping based on KL divergence
    pub early_stop_on_kl: bool,
}

/// TRPO-specific configuration
#[derive(Debug, Clone)]
pub struct TRPOConfig<T: Float> {
    /// Maximum KL divergence for trust region
    pub max_kl: T,

    /// Backtracking line search parameters
    pub backtrack_factor: T,
    pub max_backtracks: usize,

    /// Conjugate gradient parameters
    pub cg_iters: usize,
    pub cg_damping: T,
    pub cg_tolerance: T,

    /// Use natural gradients
    pub use_natural_gradients: bool,
}

impl<T: Float + Send + Sync + num_traits::FromPrimitive> Default for PolicyGradientConfig<T> {
    fn default() -> Self {
        Self {
            base_config: RLOptimizerConfig::default(),
            method: PolicyGradientMethod::PPOClip,
            ppo_config: PPOConfig::default(),
            trpo_config: TRPOConfig::default(),
            policy_scheduler: Some(RLScheduler::new(
                num_traits::cast::cast(3e-4).unwrap_or_else(|| T::zero()),
                ScheduleType::Constant,
            )),
            value_scheduler: Some(RLScheduler::new(
                num_traits::cast::cast(1e-3).unwrap_or_else(|| T::zero()),
                ScheduleType::Constant,
            )),
            use_baseline: true,
            importance_sampling: false,
            max_is_ratio: num_traits::cast::cast(2.0).unwrap_or_else(|| T::zero()),
        }
    }
}

impl<T: Float + Send + Sync + num_traits::FromPrimitive> Default for PPOConfig<T> {
    fn default() -> Self {
        Self {
            clip_epsilon: num_traits::cast::cast(0.2).unwrap_or_else(|| T::zero()),
            dual_clip: false,
            value_clip: true,
            value_clip_range: num_traits::cast::cast(0.2).unwrap_or_else(|| T::zero()),
            target_kl: num_traits::cast::cast(0.01).unwrap_or_else(|| T::zero()),
            kl_coeff: num_traits::cast::cast(0.2).unwrap_or_else(|| T::zero()),
            kl_coeff_adapt_factor: num_traits::cast::cast(1.5).unwrap_or_else(|| T::zero()),
            early_stop_on_kl: true,
        }
    }
}

impl<T: Float + Send + Sync + num_traits::FromPrimitive> Default for TRPOConfig<T> {
    fn default() -> Self {
        Self {
            max_kl: num_traits::cast::cast(0.01).unwrap_or_else(|| T::zero()),
            backtrack_factor: num_traits::cast::cast(0.5).unwrap_or_else(|| T::zero()),
            max_backtracks: 10,
            cg_iters: 10,
            cg_damping: num_traits::cast::cast(0.1).unwrap_or_else(|| T::zero()),
            cg_tolerance: num_traits::cast::cast(1e-8).unwrap_or_else(|| T::zero()),
            use_natural_gradients: true,
        }
    }
}

/// Policy gradient optimizer
pub struct PolicyGradientOptimizer<T: Float, P: PolicyNetwork<T>, V: ValueNetwork<T>> {
    /// Configuration
    config: PolicyGradientConfig<T>,

    /// Policy network
    policy_network: P,

    /// Value network
    value_network: Option<V>,

    /// Learning rate schedulers
    policy_scheduler: Option<RLScheduler<T>>,
    value_scheduler: Option<RLScheduler<T>>,

    /// Optimization statistics
    metrics: RLOptimizationMetrics<T>,

    /// Update counter
    update_count: usize,

    /// KL coefficient for adaptive PPO
    kl_coeff: T,

    /// Trajectory buffer for batch updates
    trajectory_buffer: Vec<TrajectoryBatch<T>>,

    /// Maximum buffer size
    max_buffer_size: usize,
}

impl<
        T: Float
            + Send
            + Sync
            + ScalarOperand
            + std::ops::AddAssign
            + std::iter::Sum
            + num_traits::FromPrimitive,
        P: PolicyNetwork<T>,
        V: ValueNetwork<T>,
    > PolicyGradientOptimizer<T, P, V>
{
    /// Create a new policy gradient optimizer
    pub fn new(
        config: PolicyGradientConfig<T>,
        policy_network: P,
        value_network: Option<V>,
    ) -> Self {
        let kl_coeff = config.ppo_config.kl_coeff;
        let policy_scheduler = config.policy_scheduler.clone();
        let value_scheduler = config.value_scheduler.clone();

        Self {
            config,
            policy_network,
            value_network,
            policy_scheduler,
            value_scheduler,
            metrics: RLOptimizationMetrics::default(),
            update_count: 0,
            kl_coeff,
            trajectory_buffer: Vec::new(),
            max_buffer_size: 1000,
        }
    }

    /// Update policy using trajectory data
    pub fn update(&mut self, trajectory: TrajectoryBatch<T>) -> Result<RLOptimizationMetrics<T>> {
        match self.config.method {
            PolicyGradientMethod::PPOClip => self.update_ppo_clip(trajectory),
            PolicyGradientMethod::PPOAdaptiveKL => self.update_ppo_adaptive_kl(trajectory),
            PolicyGradientMethod::TRPO => self.update_trpo(trajectory),
            PolicyGradientMethod::Reinforce => self.update_reinforce(trajectory),
            PolicyGradientMethod::ActorCritic => Err(OptimError::InvalidConfig(
                "Method not implemented".to_string(),
            )),
            _ => Err(OptimError::InvalidConfig(
                "Method not implemented".to_string(),
            )),
        }
    }

    /// PPO with clipped surrogate objective
    fn update_ppo_clip(
        &mut self,
        mut trajectory: TrajectoryBatch<T>,
    ) -> Result<RLOptimizationMetrics<T>> {
        let mut total_policy_loss = T::zero();
        let mut total_value_loss = T::zero();
        let mut total_entropy_loss = T::zero();
        let mut clip_fraction = T::zero();
        let mut approx_kl = T::zero();

        // Compute next value for GAE
        let next_value = if let Some(ref value_net) = self.value_network {
            let last_obs = trajectory.observations.slice(s![-1.., ..]).to_owned();
            let mut last_obs_batch = Array2::zeros((1, last_obs.ncols()));
            last_obs_batch.row_mut(0).assign(&last_obs.row(0));
            value_net.evaluate_value(&last_obs_batch)?[0]
        } else {
            T::zero()
        };

        // Compute advantages using GAE
        trajectory.compute_advantages(
            self.config.base_config.discount_factor,
            self.config.base_config.gae_lambda,
            next_value,
        )?;

        // Store old policy evaluation
        let _old_policy_eval = self
            .policy_network
            .evaluate_actions(&trajectory.observations, &trajectory.actions)?;

        let n_epochs = self.config.base_config.n_epochs;
        let mini_batch_size = self.config.base_config.mini_batchsize;

        for _epoch in 0..n_epochs {
            let mini_batches = trajectory.get_mini_batches(mini_batch_size);

            for mini_batch in mini_batches {
                // Current policy evaluation
                let policy_eval = self
                    .policy_network
                    .evaluate_actions(&mini_batch.observations, &mini_batch.actions)?;

                // Compute importance sampling ratio
                let log_ratio = &policy_eval.log_probs - &mini_batch.log_probs;
                let ratio = log_ratio.mapv(|x| x.exp());

                // Compute surrogate loss
                let surr1 = &ratio * &mini_batch.advantages;
                let clipped_ratio = ratio.mapv(|r| {
                    let clip_eps = self.config.ppo_config.clip_epsilon;
                    r.max(T::one() - clip_eps).min(T::one() + clip_eps)
                });
                let surr2 = &clipped_ratio * &mini_batch.advantages;

                // Policy loss (negative because we want to maximize)
                let policy_loss = -surr1
                    .iter()
                    .zip(surr2.iter())
                    .map(|(&s1, &s2)| s1.min(s2))
                    .sum::<T>()
                    / T::from(mini_batch.observations.nrows()).unwrap();

                // Entropy loss (negative to encourage exploration)
                let entropy_loss = -policy_eval.entropy.iter().copied().sum::<T>()
                    / T::from(policy_eval.entropy.len()).unwrap_or(T::zero());

                // Value function loss
                let value_loss = if let Some(ref value_net) = self.value_network {
                    let predicted_values = value_net.evaluate_value(&mini_batch.observations)?;

                    if self.config.ppo_config.value_clip {
                        // Clipped value loss
                        let value_pred_clipped = &mini_batch.values
                            + (&predicted_values - &mini_batch.values).mapv(|diff| {
                                let clip_range = self.config.ppo_config.value_clip_range;
                                diff.max(-clip_range).min(clip_range)
                            });

                        let value_loss_1 =
                            (&predicted_values - &mini_batch.returns).mapv(|x| x * x);
                        let value_loss_2 =
                            (&value_pred_clipped - &mini_batch.returns).mapv(|x| x * x);

                        value_loss_1
                            .iter()
                            .zip(value_loss_2.iter())
                            .map(|(&v1, &v2)| v1.max(v2))
                            .sum::<T>()
                            / T::from(mini_batch.observations.nrows()).unwrap()
                    } else {
                        // Standard MSE loss
                        (&predicted_values - &mini_batch.returns)
                            .mapv(|x| x * x)
                            .mean()
                            .unwrap_or(T::zero())
                    }
                } else {
                    T::zero()
                };

                // Total loss
                let total_loss = policy_loss
                    + self.config.base_config.value_loss_coeff * value_loss
                    + self.config.base_config.entropy_coeff * entropy_loss;

                // Compute gradients and update networks
                self.update_networks_with_loss(total_loss, policy_loss, value_loss)?;

                // Accumulate metrics
                total_policy_loss = total_policy_loss + policy_loss;
                total_value_loss = total_value_loss + value_loss;
                total_entropy_loss = total_entropy_loss + entropy_loss;

                // Compute clip fraction
                let n_clipped = ratio
                    .iter()
                    .filter(|&&r| {
                        let clip_eps = self.config.ppo_config.clip_epsilon;
                        r < T::one() - clip_eps || r > T::one() + clip_eps
                    })
                    .count();
                clip_fraction =
                    clip_fraction + num_traits::cast::cast(n_clipped).unwrap_or_else(|| T::zero()) / T::from(ratio.len()).unwrap();

                // Compute approximate KL divergence
                approx_kl = approx_kl + log_ratio.mapv(|x| x * x).mean().unwrap_or(T::zero());

                // Early stopping based on KL divergence
                if self.config.ppo_config.early_stop_on_kl
                    && approx_kl > self.config.ppo_config.target_kl * num_traits::cast::cast(2.0).unwrap_or_else(|| T::zero())
                {
                    break;
                }
            }
        }

        // Update learning rates
        if let Some(ref mut scheduler) = self.policy_scheduler {
            self.metrics.policy_lr = scheduler.step();
        }
        if let Some(ref mut scheduler) = self.value_scheduler {
            self.metrics.value_lr = scheduler.step();
        }

        self.update_count += 1;

        // Update metrics
        let n_updates = T::from(
            n_epochs * ((trajectory.observations.nrows() + mini_batch_size - 1) / mini_batch_size),
        )
        .unwrap();
        self.metrics.policy_loss = total_policy_loss / n_updates;
        self.metrics.value_loss = total_value_loss / n_updates;
        self.metrics.entropy_loss = total_entropy_loss / n_updates;
        self.metrics.total_loss = self.metrics.policy_loss
            + self.config.base_config.value_loss_coeff * self.metrics.value_loss
            + self.config.base_config.entropy_coeff * self.metrics.entropy_loss;
        self.metrics.clip_fraction = Some(clip_fraction / n_updates);
        self.metrics.kl_divergence = Some(approx_kl / n_updates);

        Ok(self.metrics.clone())
    }

    /// PPO with adaptive KL penalty
    fn update_ppo_adaptive_kl(
        &mut self,
        trajectory: TrajectoryBatch<T>,
    ) -> Result<RLOptimizationMetrics<T>> {
        // Similar to PPO clip but with adaptive KL penalty instead of clipping
        // Implementation would add KL penalty term to loss function
        self.update_ppo_clip(trajectory) // Simplified for now
    }

    /// TRPO update with trust region constraint
    fn update_trpo(&mut self, trajectory: TrajectoryBatch<T>) -> Result<RLOptimizationMetrics<T>> {
        // TRPO implementation with conjugate gradient and line search
        // This is a simplified version - full TRPO requires more complex optimization
        self.update_ppo_clip(trajectory) // Simplified for now
    }

    /// REINFORCE algorithm
    fn update_reinforce(
        &mut self,
        trajectory: TrajectoryBatch<T>,
    ) -> Result<RLOptimizationMetrics<T>> {
        let policy_eval = self
            .policy_network
            .evaluate_actions(&trajectory.observations, &trajectory.actions)?;

        // Use returns as targets (no baseline)
        let policy_loss = if self.config.use_baseline && self.value_network.is_some() {
            // Actor-critic style with baseline
            -(policy_eval.log_probs * trajectory.advantages)
                .mean()
                .unwrap_or(T::zero())
        } else {
            // Pure REINFORCE
            -(policy_eval.log_probs * trajectory.returns)
                .mean()
                .unwrap_or(T::zero())
        };

        let entropy_loss = -policy_eval.entropy.iter().copied().sum::<T>()
            / T::from(policy_eval.entropy.len()).unwrap_or(T::zero());
        let total_loss = policy_loss + self.config.base_config.entropy_coeff * entropy_loss;

        self.update_networks_with_loss(total_loss, policy_loss, T::zero())?;

        self.metrics.policy_loss = policy_loss;
        self.metrics.entropy_loss = entropy_loss;
        self.metrics.total_loss = total_loss;

        Ok(self.metrics.clone())
    }

    /// Actor-Critic update
    fn update_actor_critic(
        &mut self,
        trajectory: TrajectoryBatch<T>,
    ) -> Result<RLOptimizationMetrics<T>> {
        // Similar to REINFORCE but with value function updates
        self.update_reinforce(trajectory)
    }

    /// Update networks with computed losses
    fn update_networks_with_loss(
        &mut self,
        _total_loss: T,
        policy_loss: T,
        value_loss: T,
    ) -> Result<()> {
        // 1. Compute gradients from losses (simplified - would use autodiff in practice)
        let policy_gradients = self.compute_policy_gradients(policy_loss)?;
        let value_gradients = if let Some(_) = self.value_network {
            Some(self.compute_value_gradients(value_loss)?)
        } else {
            None
        };

        // 2. Apply gradient clipping
        let clipped_policy_grads =
            self.clip_gradients(&policy_gradients, self.config.base_config.max_grad_norm)?;
        let clipped_value_grads = if let Some(val_grads) = value_gradients {
            Some(self.clip_gradients(&val_grads, self.config.base_config.max_grad_norm)?)
        } else {
            None
        };

        // 3. Update network parameters
        self.update_policy_parameters(&clipped_policy_grads)?;
        if let Some(ref val_grads) = clipped_value_grads {
            self.update_value_parameters(val_grads)?;
        }

        // 4. Update gradient norms in metrics
        self.metrics.policy_grad_norm = self.compute_gradient_norm(&clipped_policy_grads);
        if let Some(ref val_grads) = clipped_value_grads {
            self.metrics.value_grad_norm = self.compute_gradient_norm(val_grads);
        }

        Ok(())
    }

    /// Compute policy gradients (simplified)
    fn compute_policy_gradients(&self, loss: T) -> Result<HashMap<String, Array1<T>>> {
        let mut gradients = HashMap::new();

        // Simplified gradient computation - in practice would use autodiff
        let policy_params = self.policy_network.get_parameters();
        for (param_name, param_values) in policy_params {
            let grad =
                Array1::ones(param_values.len()) * loss / T::from(param_values.len()).unwrap();
            gradients.insert(param_name, grad);
        }

        Ok(gradients)
    }

    /// Compute value function gradients (simplified)
    fn compute_value_gradients(&self, loss: T) -> Result<HashMap<String, Array1<T>>> {
        let mut gradients = HashMap::new();

        if let Some(ref value_net) = self.value_network {
            let value_params = value_net.get_parameters();
            for (param_name, param_values) in value_params {
                let grad =
                    Array1::ones(param_values.len()) * loss / T::from(param_values.len()).unwrap();
                gradients.insert(param_name, grad);
            }
        }

        Ok(gradients)
    }

    /// Apply gradient clipping
    fn clip_gradients(
        &self,
        gradients: &HashMap<String, Array1<T>>,
        max_norm: T,
    ) -> Result<HashMap<String, Array1<T>>> {
        let mut clipped_gradients = HashMap::new();

        // Compute global gradient _norm
        let mut total_norm = T::zero();
        for (_, grad) in gradients {
            total_norm = total_norm + grad.iter().map(|&g| g * g).sum::<T>();
        }
        total_norm = total_norm.sqrt();

        // Apply clipping if necessary
        let clip_factor = if total_norm > max_norm {
            max_norm / total_norm
        } else {
            T::one()
        };

        for (param_name, grad) in gradients {
            let clipped_grad = grad * clip_factor;
            clipped_gradients.insert(param_name.clone(), clipped_grad);
        }

        Ok(clipped_gradients)
    }

    /// Update policy network parameters
    fn update_policy_parameters(&mut self, gradients: &HashMap<String, Array1<T>>) -> Result<()> {
        // Apply gradients to policy network
        self.policy_network.update_parameters(gradients)?;
        Ok(())
    }

    /// Update value network parameters
    fn update_value_parameters(&mut self, gradients: &HashMap<String, Array1<T>>) -> Result<()> {
        if let Some(ref mut value_net) = self.value_network {
            value_net.update_parameters(gradients)?;
        }
        Ok(())
    }

    /// Compute gradient norm
    fn compute_gradient_norm(&self, gradients: &HashMap<String, Array1<T>>) -> T {
        let mut total_norm = T::zero();
        for (_, grad) in gradients {
            total_norm = total_norm + grad.iter().map(|&g| g * g).sum::<T>();
        }
        total_norm.sqrt()
    }

    /// Get current optimization metrics
    pub fn get_metrics(&self) -> &RLOptimizationMetrics<T> {
        &self.metrics
    }

    /// Add trajectory to buffer
    pub fn add_trajectory(&mut self, trajectory: TrajectoryBatch<T>) {
        self.trajectory_buffer.push(trajectory);
        if self.trajectory_buffer.len() > self.max_buffer_size {
            self.trajectory_buffer.remove(0);
        }
    }

    /// Update using buffered trajectories
    pub fn update_from_buffer(&mut self) -> Result<RLOptimizationMetrics<T>> {
        if self.trajectory_buffer.is_empty() {
            return Err(OptimError::InvalidConfig(
                "No trajectories in buffer".to_string(),
            ));
        }

        // Combine all trajectories
        let combined = self.combine_trajectories()?;
        self.update(combined)
    }

    /// Combine multiple trajectories into one batch
    fn combine_trajectories(&self) -> Result<TrajectoryBatch<T>> {
        if self.trajectory_buffer.is_empty() {
            return Err(OptimError::InvalidConfig(
                "No trajectories to combine".to_string(),
            ));
        }

        let total_size: usize = self
            .trajectory_buffer
            .iter()
            .map(|t| t.observations.nrows())
            .sum();

        let obs_dim = self.trajectory_buffer[0].observations.ncols();
        let action_dim = self.trajectory_buffer[0].actions.ncols();

        let mut combined_obs = Array2::zeros((total_size, obs_dim));
        let mut combined_actions = Array2::zeros((total_size, action_dim));
        let mut combined_log_probs = Array1::zeros(total_size);
        let mut combined_rewards = Array1::zeros(total_size);
        let mut combined_values = Array1::zeros(total_size);
        let mut combined_dones = Vec::with_capacity(total_size);

        let mut offset = 0;
        for trajectory in &self.trajectory_buffer {
            let size = trajectory.observations.nrows();

            combined_obs
                .slice_mut(s![offset..offset + size, ..])
                .assign(&trajectory.observations);
            combined_actions
                .slice_mut(s![offset..offset + size, ..])
                .assign(&trajectory.actions);
            combined_log_probs
                .slice_mut(s![offset..offset + size])
                .assign(&trajectory.log_probs);
            combined_rewards
                .slice_mut(s![offset..offset + size])
                .assign(&trajectory.rewards);
            combined_values
                .slice_mut(s![offset..offset + size])
                .assign(&trajectory.values);

            combined_dones.extend_from_slice(trajectory.dones.as_slice().unwrap());

            offset += size;
        }

        let combined_dones_array = Array1::from_vec(combined_dones);

        TrajectoryBatch::new(
            combined_obs,
            combined_actions,
            combined_log_probs,
            combined_rewards,
            combined_values,
            combined_dones_array,
        )
    }

    /// Clear trajectory buffer
    pub fn clear_buffer(&mut self) {
        self.trajectory_buffer.clear();
    }
}

// Import slice syntax
use ndarray::s;
// use statrs::statistics::Statistics; // statrs not available
