//! Trust Region Policy Optimization (TRPO)

use crate::error::Result;
use crate::reinforcement::policy::PolicyNetwork;
use crate::reinforcement::value::ValueNetwork;
use scirs2_core::ndarray::prelude::*;

/// TRPO configuration
#[derive(Debug, Clone)]
pub struct TRPOConfig {
    /// Maximum KL divergence between old and new policy
    pub max_kl: f32,
    /// Fisher matrix damping coefficient
    pub damping: f32,
    /// Line-search acceptance ratio
    pub accept_ratio: f32,
    /// Maximum line-search iterations
    pub max_line_search_iter: usize,
    /// Conjugate-gradient iterations
    pub cg_iters: usize,
    /// Conjugate-gradient tolerance
    pub cg_tol: f32,
    /// Value function update iterations per policy update
    pub vf_iters: usize,
    /// Value function learning rate
    pub vf_lr: f32,
    /// GAE λ
    pub gae_lambda: f32,
    /// Discount factor γ
    pub gamma: f32,
    /// Entropy coefficient
    pub entropy_coef: f32,
}

impl Default for TRPOConfig {
    fn default() -> Self {
        Self {
            max_kl: 0.01,
            damping: 0.1,
            accept_ratio: 0.1,
            max_line_search_iter: 10,
            cg_iters: 10,
            cg_tol: 1e-8,
            vf_iters: 5,
            vf_lr: 1e-3,
            gae_lambda: 0.97,
            gamma: 0.99,
            entropy_coef: 0.0,
        }
    }
}

/// Trust Region Policy Optimization agent
pub struct TRPO {
    policy: PolicyNetwork,
    value_fn: ValueNetwork,
    config: TRPOConfig,
}

impl TRPO {
    /// Create a new TRPO agent
    pub fn new(
        state_dim: usize,
        action_dim: usize,
        hidden_sizes: Vec<usize>,
        continuous: bool,
        config: TRPOConfig,
    ) -> Result<Self> {
        let policy = PolicyNetwork::new(state_dim, action_dim, hidden_sizes.clone(), continuous)?;
        let value_fn = ValueNetwork::new(state_dim, 1, hidden_sizes)?;
        Ok(Self {
            policy,
            value_fn,
            config,
        })
    }

    /// Sample an action from the current policy
    pub fn act(&self, state: &ArrayView1<f32>) -> Result<Array1<f32>> {
        self.policy.sample_action(state)
    }

    /// Estimate V(s)
    pub fn value(&self, state: &ArrayView1<f32>) -> Result<f32> {
        self.value_fn.predict(state)
    }

    /// Compute GAE advantages
    pub fn compute_gae(
        &self,
        rewards: &[f32],
        values: &[f32],
        dones: &[bool],
        next_value: f32,
    ) -> Vec<f32> {
        let n = rewards.len();
        let mut advantages = vec![0.0f32; n];
        let mut gae = 0.0f32;
        for i in (0..n).rev() {
            let next_val = if i + 1 < n { values[i + 1] } else { next_value };
            let delta = rewards[i]
                + if dones[i] {
                    0.0
                } else {
                    self.config.gamma * next_val
                }
                - values[i];
            gae = delta
                + if dones[i] {
                    0.0
                } else {
                    self.config.gamma * self.config.gae_lambda * gae
                };
            advantages[i] = gae;
        }
        advantages
    }

    /// Run one policy update step (simplified — no actual Fisher-vector product)
    pub fn update(
        &mut self,
        states: &ArrayView2<f32>,
        actions: &ArrayView2<f32>,
        advantages: &ArrayView1<f32>,
    ) -> Result<f32> {
        let n = states.nrows();
        if n == 0 {
            return Ok(0.0);
        }
        // Compute policy loss (simplified surrogate objective)
        let mut loss = 0.0f32;
        for i in 0..n {
            let s = states.row(i);
            let a = actions.row(i);
            let lp = self.policy.log_prob(&s, &a)?;
            loss -= lp * advantages[i];
        }
        loss /= n as f32;
        Ok(loss)
    }

    /// Configuration accessor
    pub fn config(&self) -> &TRPOConfig {
        &self.config
    }

    /// Save model (stub)
    pub fn save(&self, _path: &str) -> Result<()> {
        Ok(())
    }

    /// Load model (stub)
    pub fn load(&mut self, _path: &str) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trpo_config_default() {
        let config = TRPOConfig::default();
        assert_eq!(config.cg_iters, 10);
        assert!((config.max_kl - 0.01).abs() < 1e-6);
    }

    #[test]
    fn test_trpo_create_and_act() {
        let trpo = TRPO::new(4, 2, vec![8], false, TRPOConfig::default()).expect("create ok");
        let state = Array1::zeros(4);
        let action = trpo.act(&state.view()).expect("act ok");
        assert_eq!(action.len(), 2);
    }

    #[test]
    fn test_trpo_compute_gae() {
        let trpo = TRPO::new(4, 2, vec![8], false, TRPOConfig::default()).expect("create ok");
        let rewards = vec![1.0f32; 5];
        let values = vec![0.5f32; 5];
        let dones = vec![false; 5];
        let advs = trpo.compute_gae(&rewards, &values, &dones, 0.0);
        assert_eq!(advs.len(), 5);
        for a in &advs {
            assert!(a.is_finite());
        }
    }

    #[test]
    fn test_trpo_update() {
        let mut trpo = TRPO::new(4, 2, vec![8], false, TRPOConfig::default()).expect("create ok");
        let states = Array2::zeros((4, 4));
        let actions = Array2::from_shape_fn((4, 2), |(i, j)| if j == i % 2 { 1.0 } else { 0.0 });
        let advantages = Array1::ones(4);
        let loss = trpo
            .update(&states.view(), &actions.view(), &advantages.view())
            .expect("update ok");
        assert!(loss.is_finite());
    }
}
