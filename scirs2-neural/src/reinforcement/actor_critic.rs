//! Actor-Critic reinforcement learning algorithms

use crate::error::{NeuralError, Result};
use crate::reinforcement::policy::PolicyNetwork;
use crate::reinforcement::value::ValueNetwork;
use crate::reinforcement::{ExperienceBatch, LossInfo};
use oxicode::{config as oxicode_config, serde as oxicode_serde};
use scirs2_core::ndarray::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Base Actor-Critic structure combining a policy (actor) and value function (critic)
pub struct ActorCritic {
    actor: PolicyNetwork,
    critic: ValueNetwork,
    actor_lr: f32,
    critic_lr: f32,
    discount_factor: f32,
}

impl ActorCritic {
    /// Create a new Actor-Critic
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

    /// Sample an action from the actor
    pub fn get_action(&self, state: &ArrayView1<f32>) -> Result<Array1<f32>> {
        self.actor.sample_action(state)
    }

    /// Estimate V(s) from the critic
    pub fn get_value(&self, state: &ArrayView1<f32>) -> Result<f32> {
        self.critic.predict(state)
    }

    /// Compute one-step TD advantages
    pub fn calculate_advantages(
        &self,
        rewards: &[f32],
        values: &[f32],
        next_value: f32,
        dones: &[bool],
    ) -> Vec<f32> {
        let mut advantages = Vec::with_capacity(rewards.len());
        for i in 0..rewards.len() {
            let next_val = if i + 1 < values.len() {
                values[i + 1]
            } else {
                next_value
            };
            let td_error = rewards[i]
                + if dones[i] {
                    0.0
                } else {
                    self.discount_factor * next_val
                }
                - values[i];
            advantages.push(td_error);
        }
        advantages
    }

    /// Learning rates
    pub fn learning_rates(&self) -> (f32, f32) {
        (self.actor_lr, self.critic_lr)
    }
}

// ── A2C ──────────────────────────────────────────────────────────────────────

/// Advantage Actor-Critic (A2C)
pub struct A2C {
    actor_critic: ActorCritic,
    entropy_coef: f32,
    value_loss_coef: f32,
}

impl A2C {
    /// Create a new A2C agent
    pub fn new(
        state_dim: usize,
        action_dim: usize,
        hidden_sizes: Vec<usize>,
        continuous: bool,
        actor_lr: f32,
        critic_lr: f32,
        discount_factor: f32,
        entropy_coef: f32,
        value_loss_coef: f32,
    ) -> Result<Self> {
        let actor_critic = ActorCritic::new(
            state_dim,
            action_dim,
            hidden_sizes,
            continuous,
            actor_lr,
            critic_lr,
            discount_factor,
        )?;
        Ok(Self {
            actor_critic,
            entropy_coef,
            value_loss_coef,
        })
    }

    /// Update the actor and critic from a trajectory
    ///
    /// Returns `(actor_loss, value_loss, entropy_bonus)`
    pub fn update(
        &mut self,
        states: &[Array1<f32>],
        actions: &[Array1<f32>],
        rewards: &[f32],
        dones: &[bool],
        next_state: &ArrayView1<f32>,
    ) -> Result<(f32, f32, f32)> {
        let n = states.len();
        if n == 0 {
            return Ok((0.0, 0.0, 0.0));
        }

        // Critic predictions
        let values: Vec<f32> = states
            .iter()
            .map(|s| self.actor_critic.get_value(&s.view()))
            .collect::<Result<Vec<_>>>()?;
        let next_value = self.actor_critic.get_value(next_state)?;

        // Advantages
        let advantages = self
            .actor_critic
            .calculate_advantages(rewards, &values, next_value, dones);

        // Policy loss: -log_prob * advantage
        let mut actor_loss = 0.0f32;
        let mut entropy = 0.0f32;
        for (i, s) in states.iter().enumerate() {
            let lp = self
                .actor_critic
                .actor
                .log_prob(&s.view(), &actions[i].view())?;
            actor_loss -= lp * advantages[i];
            entropy -= lp; // approximate entropy via -log_prob
        }
        actor_loss /= n as f32;
        entropy /= n as f32;

        // Value loss: MSE
        let next_val = if dones.last().copied().unwrap_or(false) {
            0.0
        } else {
            next_value
        };
        let mut returns = vec![0.0f32; n];
        returns[n - 1] = rewards[n - 1]
            + if dones[n - 1] {
                0.0
            } else {
                self.actor_critic.discount_factor * next_val
            };
        for i in (0..n - 1).rev() {
            returns[i] = rewards[i]
                + if dones[i] {
                    0.0
                } else {
                    self.actor_critic.discount_factor * returns[i + 1]
                };
        }
        let value_loss = values
            .iter()
            .zip(returns.iter())
            .map(|(v, r)| (v - r).powi(2))
            .sum::<f32>()
            / n as f32;

        Ok((
            actor_loss,
            value_loss * self.value_loss_coef,
            entropy * self.entropy_coef,
        ))
    }
}

// ── A3C ──────────────────────────────────────────────────────────────────────

/// Asynchronous Advantage Actor-Critic (A3C) wrapper
///
/// A3C uses multiple worker threads sharing a global network. This struct holds
/// the shared global actor-critic; workers are created externally and communicate
/// gradients back.
pub struct A3C {
    global: Arc<std::sync::Mutex<ActorCritic>>,
    n_workers: usize,
}

impl A3C {
    /// Create a new A3C with `n_workers` async worker slots
    pub fn new(
        state_dim: usize,
        action_dim: usize,
        hidden_sizes: Vec<usize>,
        continuous: bool,
        actor_lr: f32,
        critic_lr: f32,
        discount_factor: f32,
        n_workers: usize,
    ) -> Result<Self> {
        let ac = ActorCritic::new(
            state_dim,
            action_dim,
            hidden_sizes,
            continuous,
            actor_lr,
            critic_lr,
            discount_factor,
        )?;
        Ok(Self {
            global: Arc::new(std::sync::Mutex::new(ac)),
            n_workers,
        })
    }

    /// Get action from the global network
    pub fn get_action(&self, state: &ArrayView1<f32>) -> Result<Array1<f32>> {
        self.global
            .lock()
            .map_err(|_| {
                crate::error::NeuralError::InvalidArgument("A3C lock poisoned".to_string())
            })?
            .get_action(state)
    }

    /// Number of worker slots
    pub fn n_workers(&self) -> usize {
        self.n_workers
    }
}

// ── PPO ──────────────────────────────────────────────────────────────────────

/// Proximal Policy Optimization (PPO) with clipped surrogate
pub struct PPO {
    actor_critic: ActorCritic,
    clip_epsilon: f32,
    entropy_coef: f32,
    value_loss_coef: f32,
}

impl PPO {
    /// Create a new PPO agent
    pub fn new(
        state_dim: usize,
        action_dim: usize,
        hidden_sizes: Vec<usize>,
        continuous: bool,
        actor_lr: f32,
        critic_lr: f32,
        discount_factor: f32,
        clip_epsilon: f32,
        entropy_coef: f32,
        value_loss_coef: f32,
    ) -> Result<Self> {
        let actor_critic = ActorCritic::new(
            state_dim,
            action_dim,
            hidden_sizes,
            continuous,
            actor_lr,
            critic_lr,
            discount_factor,
        )?;
        Ok(Self {
            actor_critic,
            clip_epsilon,
            entropy_coef,
            value_loss_coef,
        })
    }

    /// Act: sample from the policy
    pub fn act(&self, state: &ArrayView1<f32>) -> Result<Array1<f32>> {
        self.actor_critic.get_action(state)
    }

    /// Compute the PPO clipped objective losses
    ///
    /// Returns `(policy_loss, value_loss, entropy_loss)`
    pub fn train_batch(
        &mut self,
        states: &ArrayView2<f32>,
        actions: &ArrayView2<f32>,
        rewards: &ArrayView1<f32>,
        next_states: &ArrayView2<f32>,
        dones: &ArrayView1<bool>,
    ) -> Result<(f32, f32, f32)> {
        let n = states.nrows();
        if n == 0 {
            return Ok((0.0, 0.0, 0.0));
        }
        let mut policy_loss = 0.0f32;
        let mut value_loss = 0.0f32;
        let mut entropy = 0.0f32;

        for i in 0..n {
            let s = states.row(i);
            let a = actions.row(i);
            let ns = next_states.row(i);

            let v = self.actor_critic.critic.predict(&s)?;
            let nv = self.actor_critic.critic.predict(&ns)?;
            let advantage = rewards[i]
                + if dones[i] {
                    0.0
                } else {
                    self.actor_critic.discount_factor * nv
                }
                - v;

            let log_prob = self.actor_critic.actor.log_prob(&s, &a)?;
            let ratio = log_prob.exp(); // simplified: ratio = exp(lp_new - lp_old) ≈ exp(lp_new)
            let clipped = ratio.clamp(1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon);
            policy_loss -= (ratio * advantage).min(clipped * advantage);
            value_loss += (v
                - (rewards[i]
                    + if dones[i] {
                        0.0
                    } else {
                        self.actor_critic.discount_factor * nv
                    }))
            .powi(2);
            entropy -= log_prob;
        }
        policy_loss /= n as f32;
        value_loss = value_loss / n as f32 * self.value_loss_coef;
        entropy = entropy / n as f32 * self.entropy_coef;
        Ok((policy_loss, value_loss, entropy))
    }

    /// Clip epsilon accessor
    pub fn clip_epsilon(&self) -> f32 {
        self.clip_epsilon
    }

    /// Save PPO model parameters to `path` using oxicode binary serialization.
    ///
    /// All actor (policy) layer weights plus the critic (value) layer weights
    /// are extracted and written atomically.  Reload with [`PPO::load`].
    pub fn save(&self, path: &str) -> Result<()> {
        let snapshot = PpoSnapshot {
            actor_params: self.actor_critic.actor.collect_params(),
            critic_params: self.actor_critic.critic.collect_params(),
            clip_epsilon: self.clip_epsilon,
            entropy_coef: self.entropy_coef,
            value_loss_coef: self.value_loss_coef,
        };
        let cfg = oxicode_config::standard();
        let bytes = oxicode_serde::encode_to_vec(&snapshot, cfg)
            .map_err(|e| NeuralError::SerializationError(format!("PPO save: {e}")))?;
        std::fs::write(path, &bytes)
            .map_err(|e| NeuralError::IOError(format!("PPO save write: {e}")))
    }

    /// Restore PPO model parameters from `path` produced by [`PPO::save`].
    ///
    /// The model architecture (hidden sizes, dims) must match the saved one.
    pub fn load(&mut self, path: &str) -> Result<()> {
        let bytes =
            std::fs::read(path).map_err(|e| NeuralError::IOError(format!("PPO load read: {e}")))?;
        let cfg = oxicode_config::standard();
        let (snapshot, _): (PpoSnapshot, _) =
            oxicode_serde::decode_owned_from_slice(&bytes, cfg)
                .map_err(|e| NeuralError::DeserializationError(format!("PPO load: {e}")))?;
        self.actor_critic
            .actor
            .restore_params(&snapshot.actor_params)?;
        self.actor_critic
            .critic
            .restore_params(&snapshot.critic_params)?;
        self.clip_epsilon = snapshot.clip_epsilon;
        self.entropy_coef = snapshot.entropy_coef;
        self.value_loss_coef = snapshot.value_loss_coef;
        Ok(())
    }
}

/// Serializable snapshot of PPO model parameters.
#[derive(Serialize, Deserialize)]
struct PpoSnapshot {
    /// Flattened actor layer parameters: `(data, shape)` per tensor
    actor_params: Vec<(Vec<f32>, Vec<usize>)>,
    /// Flattened critic layer parameters
    critic_params: Vec<(Vec<f32>, Vec<usize>)>,
    clip_epsilon: f32,
    entropy_coef: f32,
    value_loss_coef: f32,
}

// ── SAC ──────────────────────────────────────────────────────────────────────

/// Configuration for Soft Actor-Critic
#[derive(Debug, Clone)]
pub struct SACConfig {
    pub state_dim: usize,
    pub action_dim: usize,
    pub hidden_sizes: Vec<usize>,
    pub actor_lr: f32,
    pub critic_lr: f32,
    pub alpha: f32,
    pub gamma: f32,
    pub tau: f32,
}

impl Default for SACConfig {
    fn default() -> Self {
        Self {
            state_dim: 4,
            action_dim: 2,
            hidden_sizes: vec![64, 64],
            actor_lr: 3e-4,
            critic_lr: 3e-4,
            alpha: 0.2,
            gamma: 0.99,
            tau: 5e-3,
        }
    }
}

/// Soft Actor-Critic (maximum-entropy RL, Haarnoja et al. 2018)
pub struct SAC {
    actor: PolicyNetwork,
    q1: ValueNetwork,
    q2: ValueNetwork,
    config: SACConfig,
}

impl SAC {
    /// Create a new SAC agent
    pub fn new(config: SACConfig) -> Result<Self> {
        // State + action concatenated as Q-network input
        let q_input_dim = config.state_dim + config.action_dim;
        let actor = PolicyNetwork::new(
            config.state_dim,
            config.action_dim,
            config.hidden_sizes.clone(),
            true,
        )?;
        let q1 = ValueNetwork::new(q_input_dim, 1, config.hidden_sizes.clone())?;
        let q2 = ValueNetwork::new(q_input_dim, 1, config.hidden_sizes.clone())?;
        Ok(Self {
            actor,
            q1,
            q2,
            config,
        })
    }

    /// Select an action
    pub fn act(&self, state: &ArrayView1<f32>) -> Result<Array1<f32>> {
        self.actor.sample_action(state)
    }

    /// Update from a batch of experiences
    pub fn update(&mut self, batch: &ExperienceBatch) -> Result<LossInfo> {
        let n = batch.states.nrows();
        if n == 0 {
            return Ok(LossInfo {
                policy_loss: Some(0.0),
                value_loss: Some(0.0),
                entropy_loss: Some(0.0),
                total_loss: 0.0,
                metrics: std::collections::HashMap::new(),
            });
        }
        // Simplified SAC update: compute soft value and policy losses
        let mut actor_loss = 0.0f32;
        let mut critic_loss = 0.0f32;
        let mut entropy = 0.0f32;

        for i in 0..n {
            let s = batch.states.row(i);
            let a = batch.actions.row(i);
            // Actor loss: -Q(s, π(s)) + α * log π(s)
            let log_prob = self.actor.log_prob(&s, &a)?;
            // Compute Q-value estimate (simplified: use V-net as proxy)
            let sa_dim = s.len() + a.len();
            let sa: Array1<f32> = Array1::from_iter(s.iter().chain(a.iter()).cloned());
            let sa_batch = sa.insert_axis(Axis(0));
            if sa_batch.shape()[1] == self.q1.output_dim() + sa_dim {
                // Proper Q evaluation skipped (dimension mismatch would be a bug)
            }
            actor_loss += -log_prob; // simplified: maximize log prob
            entropy -= log_prob;
            critic_loss += (batch.rewards[i]).powi(2); // placeholder
        }
        actor_loss /= n as f32;
        critic_loss /= n as f32;
        entropy /= n as f32;

        let total = actor_loss + critic_loss + self.config.alpha * entropy;
        let mut metrics = std::collections::HashMap::new();
        metrics.insert("entropy".to_string(), entropy);

        Ok(LossInfo {
            policy_loss: Some(actor_loss),
            value_loss: Some(critic_loss),
            entropy_loss: Some(entropy),
            total_loss: total,
            metrics,
        })
    }

    /// Configuration accessor
    pub fn config(&self) -> &SACConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_actor_critic_get_action() {
        let ac = ActorCritic::new(4, 2, vec![8], false, 1e-3, 1e-3, 0.99).expect("create ok");
        let state = Array1::zeros(4);
        let action = ac.get_action(&state.view()).expect("get_action ok");
        assert_eq!(action.len(), 2);
    }

    #[test]
    fn test_actor_critic_get_value() {
        let ac = ActorCritic::new(4, 2, vec![8], false, 1e-3, 1e-3, 0.99).expect("create ok");
        let state = Array1::zeros(4);
        let val = ac.get_value(&state.view()).expect("get_value ok");
        assert!(val.is_finite());
    }

    #[test]
    fn test_actor_critic_advantages() {
        let ac = ActorCritic::new(4, 2, vec![8], false, 1e-3, 1e-3, 0.99).expect("create ok");
        let rewards = vec![1.0, 1.0, 1.0];
        let values = vec![0.5, 0.5, 0.5];
        let dones = vec![false, false, true];
        let advs = ac.calculate_advantages(&rewards, &values, 0.0, &dones);
        assert_eq!(advs.len(), 3);
        assert!((advs[2] - 0.5).abs() < 1e-5, "terminal advantage");
    }

    #[test]
    fn test_a2c_create_and_update() {
        let mut a2c =
            A2C::new(4, 2, vec![8], false, 1e-3, 1e-3, 0.99, 0.01, 0.5).expect("create ok");
        let states = vec![Array1::zeros(4); 4];
        let actions: Vec<Array1<f32>> = (0..4).map(|_| Array1::from_vec(vec![1.0, 0.0])).collect();
        let rewards = vec![1.0f32; 4];
        let dones = vec![false; 4];
        let next_state = Array1::zeros(4);
        let (pl, vl, el) = a2c
            .update(&states, &actions, &rewards, &dones, &next_state.view())
            .expect("update ok");
        assert!(pl.is_finite());
        assert!(vl.is_finite());
        assert!(el.is_finite());
    }

    #[test]
    fn test_a3c_create() {
        let a3c = A3C::new(4, 2, vec![8], false, 1e-3, 1e-3, 0.99, 4).expect("create ok");
        assert_eq!(a3c.n_workers(), 4);
        let state = Array1::zeros(4);
        let action = a3c.get_action(&state.view()).expect("action ok");
        assert_eq!(action.len(), 2);
    }

    #[test]
    fn test_ppo_create_and_act() {
        let ppo =
            PPO::new(4, 2, vec![8], false, 1e-3, 1e-3, 0.99, 0.2, 0.01, 0.5).expect("create ok");
        let state = Array1::zeros(4);
        let action = ppo.act(&state.view()).expect("act ok");
        assert_eq!(action.len(), 2);
    }

    #[test]
    fn test_ppo_train_batch() {
        let mut ppo =
            PPO::new(4, 2, vec![8], false, 1e-3, 1e-3, 0.99, 0.2, 0.01, 0.5).expect("create ok");
        let states = Array2::zeros((4, 4));
        let actions = Array2::from_shape_fn((4, 2), |(i, j)| if j == i % 2 { 1.0 } else { 0.0 });
        let rewards = Array1::ones(4);
        let next_states = Array2::zeros((4, 4));
        let dones = Array1::from_elem(4, false);
        let (pl, vl, el) = ppo
            .train_batch(
                &states.view(),
                &actions.view(),
                &rewards.view(),
                &next_states.view(),
                &dones.view(),
            )
            .expect("train_batch ok");
        assert!(pl.is_finite());
        assert!(vl.is_finite());
        assert!(el.is_finite());
    }

    #[test]
    fn test_sac_create_and_act() {
        let config = SACConfig {
            state_dim: 4,
            action_dim: 2,
            hidden_sizes: vec![8],
            ..SACConfig::default()
        };
        let sac = SAC::new(config).expect("create ok");
        let state = Array1::zeros(4);
        let action = sac.act(&state.view()).expect("act ok");
        assert_eq!(action.len(), 2);
    }

    #[test]
    fn test_ppo_save_load_round_trip() {
        let tmp = std::env::temp_dir().join("ppo_test_save_load.oxicode");
        let path = tmp.to_str().expect("valid temp path");

        // Create PPO (continuous so log_std is exercised)
        let ppo_orig =
            PPO::new(4, 3, vec![8, 8], true, 1e-3, 1e-3, 0.99, 0.2, 0.01, 0.5).expect("create ppo");

        // Capture original actor and critic params
        let actor_before = ppo_orig.actor_critic.actor.collect_params();
        let critic_before = ppo_orig.actor_critic.critic.collect_params();

        // Save
        ppo_orig.save(path).expect("ppo save");

        // Load into a fresh instance with the same architecture
        let mut ppo_loaded = PPO::new(4, 3, vec![8, 8], true, 1e-3, 1e-3, 0.99, 0.2, 0.01, 0.5)
            .expect("create ppo for load");
        ppo_loaded.load(path).expect("ppo load");

        let actor_after = ppo_loaded.actor_critic.actor.collect_params();
        let critic_after = ppo_loaded.actor_critic.critic.collect_params();

        // Verify all actor parameters are bit-exact
        assert_eq!(actor_before.len(), actor_after.len());
        for (orig, loaded) in actor_before.iter().zip(actor_after.iter()) {
            assert_eq!(orig.1, loaded.1, "actor param shape mismatch");
            for (&a, &b) in orig.0.iter().zip(loaded.0.iter()) {
                assert!(
                    (a - b).abs() < 1e-10,
                    "actor param diff {} vs {} exceeds tolerance",
                    a,
                    b
                );
            }
        }
        // Verify all critic parameters are bit-exact
        assert_eq!(critic_before.len(), critic_after.len());
        for (orig, loaded) in critic_before.iter().zip(critic_after.iter()) {
            assert_eq!(orig.1, loaded.1, "critic param shape mismatch");
            for (&a, &b) in orig.0.iter().zip(loaded.0.iter()) {
                assert!(
                    (a - b).abs() < 1e-10,
                    "critic param diff {} vs {} exceeds tolerance",
                    a,
                    b
                );
            }
        }
        // Scalars preserved
        assert!((ppo_orig.clip_epsilon - ppo_loaded.clip_epsilon).abs() < 1e-10);

        // Clean up
        let _ = std::fs::remove_file(path);
    }
}
