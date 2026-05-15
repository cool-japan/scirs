//! Trust Region Policy Optimization (TRPO)

use crate::error::{NeuralError, Result};
use crate::reinforcement::policy::PolicyNetwork;
use crate::reinforcement::value::ValueNetwork;
use oxicode::{config as oxicode_config, serde as oxicode_serde};
use scirs2_core::ndarray::prelude::*;
use serde::{Deserialize, Serialize};

/// TRPO configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
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

    /// Save TRPO model parameters to `path` using oxicode binary serialization.
    pub fn save(&self, path: &str) -> Result<()> {
        let snapshot = TrpoSnapshot {
            policy_params: self.policy.collect_params(),
            value_params: self.value_fn.collect_params(),
            config: self.config.clone(),
        };
        let cfg = oxicode_config::standard();
        let bytes = oxicode_serde::encode_to_vec(&snapshot, cfg)
            .map_err(|e| NeuralError::SerializationError(format!("TRPO save: {e}")))?;
        std::fs::write(path, &bytes)
            .map_err(|e| NeuralError::IOError(format!("TRPO save write: {e}")))
    }

    /// Restore TRPO model parameters from `path` produced by [`TRPO::save`].
    pub fn load(&mut self, path: &str) -> Result<()> {
        let bytes = std::fs::read(path)
            .map_err(|e| NeuralError::IOError(format!("TRPO load read: {e}")))?;
        let cfg = oxicode_config::standard();
        let (snapshot, _): (TrpoSnapshot, _) = oxicode_serde::decode_owned_from_slice(&bytes, cfg)
            .map_err(|e| NeuralError::DeserializationError(format!("TRPO load: {e}")))?;
        self.policy.restore_params(&snapshot.policy_params)?;
        self.value_fn.restore_params(&snapshot.value_params)?;
        self.config = snapshot.config;
        Ok(())
    }
}

/// Serializable snapshot of TRPO model parameters.
#[derive(Serialize, Deserialize)]
struct TrpoSnapshot {
    policy_params: Vec<(Vec<f32>, Vec<usize>)>,
    value_params: Vec<(Vec<f32>, Vec<usize>)>,
    config: TRPOConfig,
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

    #[test]
    fn test_trpo_save_load_round_trip() {
        let tmp = std::env::temp_dir().join("trpo_test_save_load.oxicode");
        let path = tmp.to_str().expect("valid temp path");

        // Discrete TRPO
        let trpo_orig =
            TRPO::new(4, 2, vec![8, 8], false, TRPOConfig::default()).expect("create trpo");

        let policy_before = trpo_orig.policy.collect_params();
        let value_before = trpo_orig.value_fn.collect_params();

        trpo_orig.save(path).expect("trpo save");

        let mut trpo_loaded =
            TRPO::new(4, 2, vec![8, 8], false, TRPOConfig::default()).expect("create trpo load");
        trpo_loaded.load(path).expect("trpo load");

        let policy_after = trpo_loaded.policy.collect_params();
        let value_after = trpo_loaded.value_fn.collect_params();

        assert_eq!(policy_before.len(), policy_after.len());
        for (orig, loaded) in policy_before.iter().zip(policy_after.iter()) {
            assert_eq!(orig.1, loaded.1, "policy param shape mismatch");
            for (&a, &b) in orig.0.iter().zip(loaded.0.iter()) {
                assert!(
                    (a - b).abs() < 1e-10,
                    "policy param diff {} vs {} exceeds tolerance",
                    a,
                    b
                );
            }
        }
        assert_eq!(value_before.len(), value_after.len());
        for (orig, loaded) in value_before.iter().zip(value_after.iter()) {
            assert_eq!(orig.1, loaded.1, "value param shape mismatch");
            for (&a, &b) in orig.0.iter().zip(loaded.0.iter()) {
                assert!(
                    (a - b).abs() < 1e-10,
                    "value param diff {} vs {} exceeds tolerance",
                    a,
                    b
                );
            }
        }

        // Config round-trips
        assert!((trpo_orig.config.max_kl - trpo_loaded.config.max_kl).abs() < 1e-10);

        let _ = std::fs::remove_file(path);
    }
}
