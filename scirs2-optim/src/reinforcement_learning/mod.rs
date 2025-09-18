//! Reinforcement Learning Optimizers
//!
//! This module provides specialized optimizers for reinforcement learning,
//! including policy gradient methods, actor-critic algorithms, and trust region methods.

use crate::error::{OptimError, Result};
use ndarray::{Array1, Array2};
use num_traits::Float;
use std::collections::HashMap;
use std::fmt::Debug;

pub mod actor_critic;
pub mod natural_gradients;
pub mod policy_gradient;
pub mod trust_region;

// Re-export key types
pub use actor_critic::{ActorCriticConfig, ActorCriticMethod, ActorCriticOptimizer};
pub use natural_gradients::{NaturalGradientConfig, NaturalPolicyGradient};
pub use policy_gradient::{PolicyGradientConfig, PolicyGradientMethod, PolicyGradientOptimizer};
pub use trust_region::{TrustRegionConfig, TrustRegionMethod, TrustRegionOptimizer};

/// Reinforcement Learning optimization configuration
#[derive(Debug, Clone)]
pub struct RLOptimizerConfig<T: Float + Debug + Send + Sync + 'static> {
    /// Policy learning rate
    pub policy_lr: T,

    /// Value function learning rate  
    pub value_lr: T,

    /// Discount factor (gamma)
    pub discount_factor: T,

    /// GAE lambda parameter
    pub gae_lambda: T,

    /// Clipping parameter for PPO
    pub clip_epsilon: T,

    /// Entropy regularization coefficient
    pub entropy_coeff: T,

    /// Value function loss coefficient
    pub value_loss_coeff: T,

    /// Maximum gradient norm for clipping
    pub max_grad_norm: T,

    /// Number of optimization epochs per update
    pub n_epochs: usize,

    /// Mini-batch size for optimization
    pub mini_batchsize: usize,

    /// Trust region methods configuration
    pub trust_region_config: Option<TrustRegionConfig<T>>,

    /// Enable natural policy gradients
    pub use_natural_gradients: bool,

    /// Fisher information matrix approximation method
    pub fisher_approximation: FisherApproximationMethod,
}

/// Methods for approximating the Fisher Information Matrix
#[derive(Debug, Clone, Copy)]
pub enum FisherApproximationMethod {
    /// Empirical Fisher Information Matrix
    Empirical,

    /// Kronecker-factored approximation
    KroneckerFactored,

    /// Diagonal approximation
    Diagonal,

    /// Block-diagonal approximation
    BlockDiagonal,

    /// Low-rank approximation
    LowRank,
}

impl<T: Float + Debug + Send + Sync + 'static> Default for RLOptimizerConfig<T> {
    fn default() -> Self {
        Self {
            policy_lr: num_traits::cast::cast(3e-4).unwrap_or_else(|| T::zero()),
            value_lr: num_traits::cast::cast(1e-3).unwrap_or_else(|| T::zero()),
            discount_factor: num_traits::cast::cast(0.99).unwrap_or_else(|| T::zero()),
            gae_lambda: num_traits::cast::cast(0.95).unwrap_or_else(|| T::zero()),
            clip_epsilon: num_traits::cast::cast(0.2).unwrap_or_else(|| T::zero()),
            entropy_coeff: num_traits::cast::cast(0.01).unwrap_or_else(|| T::zero()),
            value_loss_coeff: num_traits::cast::cast(0.5).unwrap_or_else(|| T::zero()),
            max_grad_norm: num_traits::cast::cast(0.5).unwrap_or_else(|| T::zero()),
            n_epochs: 4,
            mini_batchsize: 64,
            trust_region_config: None,
            use_natural_gradients: false,
            fisher_approximation: FisherApproximationMethod::Diagonal,
        }
    }
}

/// Trajectory data for RL optimization
#[derive(Debug, Clone)]
pub struct TrajectoryBatch<T: Float + Debug + Send + Sync + 'static> {
    /// Observations
    pub observations: Array2<T>,

    /// Actions taken
    pub actions: Array2<T>,

    /// Log probabilities of actions
    pub log_probs: Array1<T>,

    /// Rewards received
    pub rewards: Array1<T>,

    /// Value function estimates
    pub values: Array1<T>,

    /// Done flags (episode termination)
    pub dones: Array1<bool>,

    /// Advantage estimates
    pub advantages: Array1<T>,

    /// Target returns
    pub returns: Array1<T>,
}

impl<T: Float + Debug + Send + Sync + 'static + num_traits::FromPrimitive> TrajectoryBatch<T> {
    /// Create a new trajectory batch
    pub fn new(
        observations: Array2<T>,
        actions: Array2<T>,
        log_probs: Array1<T>,
        rewards: Array1<T>,
        values: Array1<T>,
        dones: Array1<bool>,
    ) -> Result<Self> {
        let batch_size = observations.nrows();

        // Validate dimensions
        if actions.nrows() != batch_size
            || log_probs.len() != batch_size
            || rewards.len() != batch_size
            || values.len() != batch_size
            || dones.len() != batch_size
        {
            return Err(OptimError::InvalidConfig(
                "Inconsistent batch dimensions".to_string(),
            ));
        }

        // Compute advantages and returns (will be updated by compute_advantages)
        let advantages = Array1::zeros(batch_size);
        let returns = Array1::zeros(batch_size);

        Ok(Self {
            observations,
            actions,
            log_probs,
            rewards,
            values,
            dones,
            advantages,
            returns,
        })
    }

    /// Compute Generalized Advantage Estimation (GAE)
    pub fn compute_advantages(&mut self, gamma: T, lambda: T, nextvalue: T) -> Result<()> {
        let batch_size = self.rewards.len();
        let mut gae = T::zero();

        // Compute advantages using GAE
        for t in (0..batch_size).rev() {
            let is_terminal = if t == batch_size - 1 {
                false // Assume not terminal for last step
            } else {
                self.dones[t]
            };

            let next_val = if t == batch_size - 1 {
                nextvalue
            } else {
                self.values[t + 1]
            };

            let delta = self.rewards[t] + gamma * next_val * num_traits::cast::cast(!is_terminal as u8).unwrap_or_else(|| T::zero())
                - self.values[t];
            gae = delta + gamma * lambda * num_traits::cast::cast(!is_terminal as u8).unwrap_or_else(|| T::zero()) * gae;

            self.advantages[t] = gae;
            self.returns[t] = gae + self.values[t];
        }

        // Normalize advantages
        let mean = self.advantages.mean().unwrap_or(T::zero());
        let std = self
            .advantages
            .mapv(|x| (x - mean) * (x - mean))
            .mean()
            .unwrap_or(T::one())
            .sqrt();

        if std > num_traits::cast::cast(1e-8).unwrap_or_else(|| T::zero()) {
            self.advantages.mapv_inplace(|x| (x - mean) / std);
        }

        Ok(())
    }

    /// Get mini-batches for optimization
    pub fn get_mini_batches(&self, mini_batchsize: usize) -> Vec<TrajectoryBatch<T>> {
        let batch_size = self.observations.nrows();
        let n_mini_batches = (batch_size + mini_batchsize - 1) / mini_batchsize;

        let mut mini_batches = Vec::new();

        for i in 0..n_mini_batches {
            let start = i * mini_batchsize;
            let end = ((i + 1) * mini_batchsize).min(batch_size);

            if start >= end {
                break;
            }

            let obs = self.observations.slice(s![start..end, ..]).to_owned();
            let acts = self.actions.slice(s![start..end, ..]).to_owned();
            let log_probs = self.log_probs.slice(s![start..end]).to_owned();
            let rewards = self.rewards.slice(s![start..end]).to_owned();
            let values = self.values.slice(s![start..end]).to_owned();
            let dones = self.dones.slice(s![start..end]).to_owned().to_vec();
            let advantages = self.advantages.slice(s![start..end]).to_owned();
            let returns = self.returns.slice(s![start..end]).to_owned();

            // Convert Vec<bool> back to Array1<bool>
            let dones_array = Array1::from_vec(dones);

            let mini_batch = TrajectoryBatch {
                observations: obs,
                actions: acts,
                log_probs,
                rewards,
                values,
                dones: dones_array,
                advantages,
                returns,
            };

            mini_batches.push(mini_batch);
        }

        mini_batches
    }
}

/// Policy network interface for RL optimizers
pub trait PolicyNetwork<T: Float> {
    /// Evaluate actions for given observations
    fn evaluate_actions(
        &self,
        observations: &Array2<T>,
        actions: &Array2<T>,
    ) -> Result<PolicyEvaluation<T>>;

    /// Get action distribution for given observations
    fn get_action_distribution(&self, observations: &Array2<T>) -> Result<ActionDistribution<T>>;

    /// Update policy parameters with gradients
    fn update_parameters(&mut self, gradients: &HashMap<String, Array1<T>>) -> Result<()>;

    /// Get current policy parameters
    fn get_parameters(&self) -> HashMap<String, Array1<T>>;
}

/// Value network interface for RL optimizers
pub trait ValueNetwork<T: Float> {
    /// Evaluate value function for given observations
    fn evaluate_value(&self, observations: &Array2<T>) -> Result<Array1<T>>;

    /// Update value function parameters with gradients
    fn update_parameters(&mut self, gradients: &HashMap<String, Array1<T>>) -> Result<()>;

    /// Get current value function parameters
    fn get_parameters(&self) -> HashMap<String, Array1<T>>;
}

/// Policy evaluation results
#[derive(Debug, Clone)]
pub struct PolicyEvaluation<T: Float + Debug + Send + Sync + 'static> {
    /// Log probabilities of actions
    pub log_probs: Array1<T>,

    /// Entropy of action distribution
    pub entropy: Array1<T>,

    /// Additional metrics
    pub metrics: HashMap<String, T>,
}

/// Action distribution representation
#[derive(Debug, Clone)]
pub struct ActionDistribution<T: Float + Debug + Send + Sync + 'static> {
    /// Mean of the distribution (for continuous actions)
    pub mean: Option<Array2<T>>,

    /// Standard deviation (for continuous actions)
    pub std: Option<Array2<T>>,

    /// Logits (for discrete actions)
    pub logits: Option<Array2<T>>,

    /// Distribution type
    pub distribution_type: DistributionType,
}

/// Types of action distributions
#[derive(Debug, Clone, Copy)]
pub enum DistributionType {
    /// Continuous Gaussian distribution
    Gaussian,

    /// Discrete categorical distribution
    Categorical,

    /// Beta distribution (for bounded continuous actions)
    Beta,

    /// Mixed discrete-continuous
    Mixed,
}

/// Learning rate scheduling for RL optimizers
#[derive(Debug, Clone)]
pub struct RLScheduler<T: Float + Debug + Send + Sync + 'static> {
    /// Initial learning rate
    pub initiallr: T,

    /// Current learning rate
    pub current_lr: T,

    /// Decay factor
    pub decay_factor: T,

    /// Decay schedule
    pub schedule: ScheduleType,

    /// Number of updates so far
    pub update_count: usize,

    /// Schedule parameters
    pub schedule_params: HashMap<String, T>,
}

/// Learning rate schedule types
#[derive(Debug, Clone, Copy)]
pub enum ScheduleType {
    /// Constant learning rate
    Constant,

    /// Linear decay
    Linear,

    /// Exponential decay
    Exponential,

    /// Cosine annealing
    Cosine,

    /// Step decay
    Step,

    /// Adaptive based on performance
    Adaptive,
}

impl<T: Float + Debug + Send + Sync + 'static> RLScheduler<T> {
    /// Create a new learning rate scheduler
    pub fn new(initiallr: T, schedule: ScheduleType) -> Self {
        Self {
            initiallr,
            current_lr: initiallr,
            decay_factor: num_traits::cast::cast(0.99).unwrap_or_else(|| T::zero()),
            schedule,
            update_count: 0,
            schedule_params: HashMap::new(),
        }
    }

    /// Update learning rate based on schedule
    pub fn step(&mut self) -> T {
        self.update_count += 1;

        match self.schedule {
            ScheduleType::Constant => {
                // No change
            }
            ScheduleType::Linear => {
                let decay_steps = self
                    .schedule_params
                    .get("decay_steps")
                    .copied()
                    .unwrap_or(num_traits::cast::cast(10000).unwrap_or_else(|| T::zero()));
                let progress = num_traits::cast::cast(self.update_count).unwrap_or_else(|| T::zero()) / decay_steps;
                self.current_lr = self.initiallr * (T::one() - progress).max(T::zero());
            }
            ScheduleType::Exponential => {
                self.current_lr = self.current_lr * self.decay_factor;
            }
            ScheduleType::Step => {
                let step_size = self
                    .schedule_params
                    .get("step_size")
                    .copied()
                    .unwrap_or(num_traits::cast::cast(1000).unwrap_or_else(|| T::zero()));
                if num_traits::cast::cast(self.update_count).unwrap_or_else(|| T::zero()) % step_size == T::zero() {
                    self.current_lr = self.current_lr * self.decay_factor;
                }
            }
            ScheduleType::Cosine => {
                let max_steps = self
                    .schedule_params
                    .get("max_steps")
                    .copied()
                    .unwrap_or(num_traits::cast::cast(10000).unwrap_or_else(|| T::zero()));
                let progress = num_traits::cast::cast(self.update_count).unwrap_or_else(|| T::zero()) / max_steps;
                let pi = num_traits::cast::cast(std::f64::consts::PI).unwrap_or_else(|| T::zero());
                self.current_lr =
                    self.initiallr * (T::one() + (pi * progress).cos()) / num_traits::cast::cast(2).unwrap_or_else(|| T::zero());
            }
            ScheduleType::Adaptive => {
                // Adaptive scheduling based on performance metrics
                // Implementation depends on specific performance indicators
            }
        }

        self.current_lr
    }

    /// Get current learning rate
    pub fn get_lr(&self) -> T {
        self.current_lr
    }

    /// Set schedule parameter
    pub fn set_param(&mut self, key: &str, value: T) {
        self.schedule_params.insert(key.to_string(), value);
    }
}

/// RL optimization metrics
#[derive(Debug, Clone)]
pub struct RLOptimizationMetrics<T: Float + Debug + Send + Sync + 'static> {
    /// Policy loss
    pub policy_loss: T,

    /// Value function loss
    pub value_loss: T,

    /// Entropy loss
    pub entropy_loss: T,

    /// Total loss
    pub total_loss: T,

    /// KL divergence (for trust region methods)
    pub kl_divergence: Option<T>,

    /// Explained variance
    pub explained_variance: T,

    /// Clip fraction (for PPO)
    pub clip_fraction: Option<T>,

    /// Learning rates
    pub policy_lr: T,
    pub value_lr: T,

    /// Gradient norms
    pub policy_grad_norm: T,
    pub value_grad_norm: T,

    /// Additional metrics
    pub custom_metrics: HashMap<String, T>,
}

impl<T: Float + Debug + Send + Sync + 'static> Default for RLOptimizationMetrics<T> {
    fn default() -> Self {
        Self {
            policy_loss: T::zero(),
            value_loss: T::zero(),
            entropy_loss: T::zero(),
            total_loss: T::zero(),
            kl_divergence: None,
            explained_variance: T::zero(),
            clip_fraction: None,
            policy_lr: num_traits::cast::cast(3e-4).unwrap_or_else(|| T::zero()),
            value_lr: num_traits::cast::cast(1e-3).unwrap_or_else(|| T::zero()),
            policy_grad_norm: T::zero(),
            value_grad_norm: T::zero(),
            custom_metrics: HashMap::new(),
        }
    }
}

// Import slice syntax
use ndarray::s;
// use statrs::statistics::Statistics; // statrs not available
