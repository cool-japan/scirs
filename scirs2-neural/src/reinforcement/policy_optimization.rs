//! Advanced policy optimisation — NPG, MAML, curiosity-driven agents

use crate::error::Result;
use crate::reinforcement::curiosity::ICM;
use crate::reinforcement::policy::PolicyNetwork;
use crate::reinforcement::value::ValueNetwork;
use crate::reinforcement::{ExperienceBatch, LossInfo};
use scirs2_core::ndarray::prelude::*;
use std::collections::HashMap;

// ── Natural Policy Gradient ───────────────────────────────────────────────────

/// Configuration for Natural Policy Gradient
#[derive(Debug, Clone)]
pub struct NPGConfig {
    /// Policy learning rate
    pub learning_rate: f32,
    /// Discount factor
    pub gamma: f32,
    /// GAE lambda
    pub lambda: f32,
    /// Conjugate-gradient iterations
    pub cg_iterations: usize,
    /// Conjugate-gradient tolerance
    pub cg_tolerance: f32,
    /// Fisher matrix damping
    pub fisher_damping: f32,
    /// Batch size for Fisher estimation
    pub fisher_batch_size: usize,
    /// Value function learning rate
    pub value_lr: f32,
}

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
}

/// Natural Policy Gradient agent
pub struct NaturalPolicyGradient {
    policy: PolicyNetwork,
    value_function: ValueNetwork,
    config: NPGConfig,
}

impl NaturalPolicyGradient {
    /// Create a new NPG agent
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
        })
    }

    /// Compute GAE advantages
    pub fn compute_gae(&self, rewards: &[f32], values: &[f32], dones: &[bool]) -> Vec<f32> {
        let n = rewards.len();
        let mut advantages = vec![0.0f32; n];
        let mut gae = 0.0f32;
        for i in (0..n).rev() {
            let next_v = if i + 1 < n { values[i + 1] } else { 0.0 };
            let delta = rewards[i]
                + if dones[i] {
                    0.0
                } else {
                    self.config.gamma * next_v
                }
                - values[i];
            gae = delta
                + if dones[i] {
                    0.0
                } else {
                    self.config.gamma * self.config.lambda * gae
                };
            advantages[i] = gae;
        }
        advantages
    }

    /// Sample an action
    pub fn act(&self, state: &ArrayView1<f32>) -> Result<Array1<f32>> {
        self.policy.sample_action(state)
    }

    /// Compute policy loss from a batch
    pub fn compute_policy_loss(&self, batch: &ExperienceBatch) -> Result<f32> {
        let n = batch.states.nrows();
        let mut loss = 0.0f32;
        for i in 0..n {
            let s = batch.states.row(i);
            let a = batch.actions.row(i);
            let lp = self.policy.log_prob(&s, &a)?;
            loss -= lp * batch.rewards[i];
        }
        Ok(loss / n.max(1) as f32)
    }
}

// ── MAML ─────────────────────────────────────────────────────────────────────

/// MAML configuration
#[derive(Debug, Clone)]
pub struct MAMLConfig {
    /// Inner-loop learning rate (per task)
    pub inner_lr: f32,
    /// Outer-loop learning rate
    pub outer_lr: f32,
    /// Number of inner gradient steps per task
    pub n_inner_steps: usize,
    /// Number of tasks per outer update
    pub n_tasks: usize,
    /// Inner-loop batch size
    pub inner_batch_size: usize,
}

impl Default for MAMLConfig {
    fn default() -> Self {
        Self {
            inner_lr: 0.01,
            outer_lr: 3e-4,
            n_inner_steps: 5,
            n_tasks: 8,
            inner_batch_size: 32,
        }
    }
}

/// Model-Agnostic Meta-Learning (MAML) agent
pub struct MAMLAgent {
    meta_policy: PolicyNetwork,
    config: MAMLConfig,
}

impl MAMLAgent {
    /// Create a new MAML agent
    pub fn new(
        state_dim: usize,
        action_dim: usize,
        hidden_sizes: Vec<usize>,
        continuous: bool,
        config: MAMLConfig,
    ) -> Result<Self> {
        let meta_policy = PolicyNetwork::new(state_dim, action_dim, hidden_sizes, continuous)?;
        Ok(Self {
            meta_policy,
            config,
        })
    }

    /// Act using the meta-policy (before task adaptation)
    pub fn act(&self, state: &ArrayView1<f32>) -> Result<Array1<f32>> {
        self.meta_policy.sample_action(state)
    }

    /// Simulate inner-loop adaptation and return the adapted policy loss
    pub fn adapt_and_evaluate(&self, support_batch: &ExperienceBatch) -> Result<f32> {
        // Simplified: compute the policy gradient loss on support data
        let n = support_batch.states.nrows();
        let mut loss = 0.0f32;
        for i in 0..n {
            let s = support_batch.states.row(i);
            let a = support_batch.actions.row(i);
            let lp = self.meta_policy.log_prob(&s, &a)?;
            loss -= lp * support_batch.rewards[i];
        }
        Ok(loss / n.max(1) as f32)
    }

    /// Configuration accessor
    pub fn config(&self) -> &MAMLConfig {
        &self.config
    }
}

// ── Curiosity-driven agent ────────────────────────────────────────────────────

/// Configuration for the curiosity-driven agent
#[derive(Debug, Clone)]
pub struct CuriosityConfig {
    /// Scaling factor for intrinsic reward
    pub eta: f32,
    /// Forward/inverse loss mixing (β)
    pub beta: f32,
    /// Feature dimension
    pub feature_dim: usize,
    /// Hidden sizes for curiosity networks
    pub hidden_sizes: Vec<usize>,
}

impl Default for CuriosityConfig {
    fn default() -> Self {
        Self {
            eta: 0.01,
            beta: 0.2,
            feature_dim: 32,
            hidden_sizes: vec![64, 64],
        }
    }
}

/// Curiosity-driven agent combining intrinsic and extrinsic rewards
pub struct CuriosityDrivenAgent {
    policy: PolicyNetwork,
    value_fn: ValueNetwork,
    icm: ICM,
    curiosity_config: CuriosityConfig,
    curiosity_weight: f32,
}

impl CuriosityDrivenAgent {
    /// Create a new curiosity-driven agent
    pub fn new(
        state_dim: usize,
        action_dim: usize,
        hidden_sizes: Vec<usize>,
        continuous: bool,
        curiosity_config: CuriosityConfig,
        curiosity_weight: f32,
    ) -> Result<Self> {
        let policy = PolicyNetwork::new(state_dim, action_dim, hidden_sizes.clone(), continuous)?;
        let value_fn = ValueNetwork::new(state_dim, 1, hidden_sizes)?;
        let icm = ICM::new(
            state_dim,
            action_dim,
            curiosity_config.feature_dim,
            curiosity_config.hidden_sizes.clone(),
            curiosity_config.eta,
            curiosity_config.beta,
        )?;
        Ok(Self {
            policy,
            value_fn,
            icm,
            curiosity_config,
            curiosity_weight,
        })
    }

    /// Select action
    pub fn act(&self, state: &ArrayView1<f32>) -> Result<Array1<f32>> {
        self.policy.sample_action(state)
    }

    /// Compute augmented reward = extrinsic + curiosity_weight × intrinsic
    pub fn augment_reward(
        &self,
        state: &ArrayView1<f32>,
        action: &ArrayView1<f32>,
        next_state: &ArrayView1<f32>,
        extrinsic_reward: f32,
    ) -> Result<f32> {
        let intrinsic = self
            .icm
            .compute_intrinsic_reward(state, action, next_state)?;
        Ok(extrinsic_reward + self.curiosity_weight * intrinsic)
    }

    /// Compute the combined training loss (policy + curiosity)
    pub fn compute_loss(
        &self,
        batch: &ExperienceBatch,
        next_states: &ArrayView2<f32>,
    ) -> Result<LossInfo> {
        let n = batch.states.nrows();
        let mut policy_loss = 0.0f32;
        let mut curiosity_loss = 0.0f32;

        for i in 0..n {
            let s = batch.states.row(i);
            let a = batch.actions.row(i);
            let ns = next_states.row(i);
            let lp = self.policy.log_prob(&s, &a)?;
            policy_loss -= lp * batch.rewards[i];
            curiosity_loss += self.icm.compute_loss(&s, &a, &ns)?;
        }
        policy_loss /= n.max(1) as f32;
        curiosity_loss /= n.max(1) as f32;
        let total = policy_loss + self.curiosity_weight * curiosity_loss;

        let mut metrics = HashMap::new();
        metrics.insert("curiosity_loss".to_string(), curiosity_loss);

        Ok(LossInfo {
            policy_loss: Some(policy_loss),
            value_loss: None,
            entropy_loss: None,
            total_loss: total,
            metrics,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reinforcement::ExperienceBatch;

    #[test]
    fn test_npg_config_default() {
        let config = NPGConfig::default();
        assert_eq!(config.cg_iterations, 10);
        assert!((config.gamma - 0.99).abs() < 1e-6);
    }

    #[test]
    fn test_npg_create_and_act() {
        let npg = NaturalPolicyGradient::new(4, 2, vec![8], false, NPGConfig::default())
            .expect("create ok");
        let state = Array1::zeros(4);
        let action = npg.act(&state.view()).expect("act ok");
        assert_eq!(action.len(), 2);
    }

    #[test]
    fn test_npg_compute_gae() {
        let npg = NaturalPolicyGradient::new(4, 2, vec![8], false, NPGConfig::default())
            .expect("create ok");
        let rewards = vec![1.0f32; 5];
        let values = vec![0.5f32; 5];
        let dones = vec![false; 5];
        let advs = npg.compute_gae(&rewards, &values, &dones);
        assert_eq!(advs.len(), 5);
        for a in &advs {
            assert!(a.is_finite());
        }
    }

    #[test]
    fn test_maml_create_and_act() {
        let maml = MAMLAgent::new(4, 2, vec![8], false, MAMLConfig::default()).expect("create ok");
        let state = Array1::zeros(4);
        let action = maml.act(&state.view()).expect("act ok");
        assert_eq!(action.len(), 2);
    }

    #[test]
    fn test_maml_adapt_evaluate() {
        let maml = MAMLAgent::new(4, 2, vec![8], false, MAMLConfig::default()).expect("create ok");
        let batch = ExperienceBatch {
            states: Array2::zeros((4, 4)),
            actions: Array2::from_shape_fn((4, 2), |(i, j)| if j == i % 2 { 1.0 } else { 0.0 }),
            rewards: Array1::ones(4),
            next_states: Array2::zeros((4, 4)),
            dones: Array1::from_elem(4, false),
            info: None,
        };
        let loss = maml.adapt_and_evaluate(&batch).expect("adapt ok");
        assert!(loss.is_finite());
    }

    #[test]
    fn test_curiosity_driven_agent_act() {
        let agent =
            CuriosityDrivenAgent::new(4, 2, vec![8], false, CuriosityConfig::default(), 0.1)
                .expect("create ok");
        let state = Array1::zeros(4);
        let action = agent.act(&state.view()).expect("act ok");
        assert_eq!(action.len(), 2);
    }

    #[test]
    fn test_curiosity_augment_reward() {
        let agent =
            CuriosityDrivenAgent::new(4, 2, vec![8], false, CuriosityConfig::default(), 0.5)
                .expect("create ok");
        let state = Array1::zeros(4);
        let action = Array1::from_vec(vec![1.0, 0.0]);
        let next_state = Array1::ones(4);
        let augmented = agent
            .augment_reward(&state.view(), &action.view(), &next_state.view(), 1.0)
            .expect("augment ok");
        assert!(augmented.is_finite());
        assert!(augmented >= 1.0, "augmented reward must be ≥ extrinsic");
    }
}
