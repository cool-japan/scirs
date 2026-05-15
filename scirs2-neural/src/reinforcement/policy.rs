//! Policy-based reinforcement learning algorithms

use crate::error::{NeuralError, Result};
use crate::layers::{Dense, Layer};
use scirs2_core::ndarray::prelude::*;
use scirs2_core::random::{rng, Distribution, Normal};

/// Base trait for policies
pub trait Policy: Send + Sync {
    /// Sample an action from the policy given an observation
    fn sample_action(&self, state: &ArrayView1<f32>) -> Result<Array1<f32>>;

    /// Log probability of `action` under the policy at `state`
    fn log_prob(&self, state: &ArrayView1<f32>, action: &ArrayView1<f32>) -> Result<f32>;

    /// Policy weight matrices (one per linear layer)
    fn parameters(&self) -> Vec<Array2<f32>>;

    /// Replace policy weights
    fn set_parameters(&mut self, params: &[Array2<f32>]) -> Result<()>;
}

/// Simple gradient of a policy with log-prob derivative placeholders
pub struct PolicyGradient {
    pub policy: PolicyNetwork,
    learning_rate: f32,
}

impl PolicyGradient {
    /// Create a new policy gradient wrapper
    pub fn new(policy: PolicyNetwork, learning_rate: f32) -> Self {
        Self {
            policy,
            learning_rate,
        }
    }

    /// Compute the policy-gradient loss (REINFORCE)
    pub fn compute_loss(&self, log_probs: &[f32], returns: &[f32]) -> f32 {
        log_probs
            .iter()
            .zip(returns.iter())
            .map(|(lp, g)| -lp * g)
            .sum::<f32>()
            / log_probs.len().max(1) as f32
    }

    /// Learning rate accessor
    pub fn learning_rate(&self) -> f32 {
        self.learning_rate
    }
}

/// Neural-network policy (actor network)
pub struct PolicyNetwork {
    layers: Vec<Box<dyn Layer<f32>>>,
    action_dim: usize,
    continuous: bool,
    /// Learnable log-standard deviation for continuous actions
    pub log_std: Option<Array1<f32>>,
}

impl PolicyNetwork {
    /// Create a new policy network
    pub fn new(
        state_dim: usize,
        action_dim: usize,
        hidden_sizes: Vec<usize>,
        continuous: bool,
    ) -> Result<Self> {
        let mut layers: Vec<Box<dyn Layer<f32>>> = Vec::new();
        let mut input_size = state_dim;
        for hidden_size in &hidden_sizes {
            layers.push(Box::new(Dense::new(
                input_size,
                *hidden_size,
                Some("relu"),
                &mut rng(),
            )?));
            input_size = *hidden_size;
        }
        let output_activation = if continuous {
            None // Tanh applied externally for bounded actions
        } else {
            Some("softmax")
        };
        layers.push(Box::new(Dense::new(
            input_size,
            action_dim,
            output_activation,
            &mut rng(),
        )?));
        let log_std = if continuous {
            Some(Array1::zeros(action_dim))
        } else {
            None
        };
        Ok(Self {
            layers,
            action_dim,
            continuous,
            log_std,
        })
    }

    /// Forward pass: returns action probabilities (discrete) or mean (continuous)
    pub fn forward(&self, state: &ArrayView1<f32>) -> Result<Array1<f32>> {
        let mut x: ArrayD<f32> = state.to_owned().insert_axis(Axis(0)).into_dyn();
        for layer in &self.layers {
            x = layer.forward(&x)?;
        }
        // Convert back to 1-D
        let out = x.into_dimensionality::<Ix2>().map_err(|e| {
            NeuralError::InvalidArgument(format!("policy forward reshape error: {e}"))
        })?;
        Ok(out.row(0).to_owned())
    }

    /// Sample an action from the policy.
    ///
    /// For continuous action spaces, samples from N(mean, exp(log_std)) per
    /// dimension using the network output as the mean and the learned
    /// `log_std` parameter as the standard deviation.  Replaces the previous
    /// stub that returned just the mean.
    ///
    /// For discrete action spaces, returns a one-hot vector at the argmax.
    pub fn sample_action(&self, state: &ArrayView1<f32>) -> Result<Array1<f32>> {
        let output = self.forward(state)?;
        if self.continuous {
            // Sample from N(mean, std) where std = exp(log_std)
            let mut r = rng();
            let mut action = Array1::zeros(self.action_dim);
            for i in 0..self.action_dim.min(output.len()) {
                let std = self
                    .log_std
                    .as_ref()
                    .and_then(|ls| ls.get(i).copied())
                    .unwrap_or(0.0_f32)
                    .exp()
                    .max(1e-6_f32);
                let mean = output.get(i).copied().unwrap_or(0.0_f32);
                let dist = Normal::new(mean, std).map_err(|e| {
                    NeuralError::InvalidArgument(format!(
                        "Normal distribution construction failed: {e}"
                    ))
                })?;
                action[i] = dist.sample(&mut r);
            }
            Ok(action)
        } else {
            // Argmax for discrete actions, returned as one-hot
            let best = output
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("non-NaN"))
                .map(|(i, _)| i)
                .unwrap_or(0);
            let mut action = Array1::zeros(self.action_dim);
            if best < self.action_dim {
                action[best] = 1.0;
            }
            Ok(action)
        }
    }

    /// Log probability of an action
    pub fn log_prob(&self, state: &ArrayView1<f32>, action: &ArrayView1<f32>) -> Result<f32> {
        let output = self.forward(state)?;
        if self.continuous {
            // Gaussian log-prob with learned log_std
            let log_std = self
                .log_std
                .clone()
                .unwrap_or_else(|| Array1::zeros(self.action_dim));
            let mut lp = 0.0f32;
            for i in 0..self.action_dim.min(output.len()).min(action.len()) {
                let std = log_std[i].exp().max(1e-6);
                let diff = action[i] - output[i];
                lp -= 0.5 * (diff / std).powi(2) + log_std[i] + 0.5 * std::f32::consts::TAU.ln();
            }
            Ok(lp)
        } else {
            // Categorical log-prob
            let act_idx = action
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("non-NaN"))
                .map(|(i, _)| i)
                .unwrap_or(0);
            let prob = output.get(act_idx).copied().unwrap_or(1e-10).max(1e-10);
            Ok(prob.ln())
        }
    }

    /// Whether the policy operates in continuous action space
    pub fn is_continuous(&self) -> bool {
        self.continuous
    }

    /// Action dimensionality
    pub fn action_dim(&self) -> usize {
        self.action_dim
    }

    /// Extract all trainable parameters as flat `(data, shape)` pairs.
    ///
    /// Returns one pair per parameter tensor across all layers, followed by
    /// the flattened `log_std` (if present) as a 1-D tensor.
    pub fn collect_params(&self) -> Vec<(Vec<f32>, Vec<usize>)> {
        let mut out = Vec::new();
        for layer in &self.layers {
            for arr in layer.params() {
                let shape = arr.shape().to_vec();
                let data = arr.iter().copied().collect::<Vec<f32>>();
                out.push((data, shape));
            }
        }
        // Append log_std as a 1-D tensor so it round-trips through save/load.
        if let Some(ls) = &self.log_std {
            out.push((ls.to_vec(), vec![ls.len()]));
        }
        out
    }

    /// Restore parameters from the output of [`collect_params`].
    ///
    /// The slice must have the same length as produced by `collect_params`.
    /// Returns an error if any shape is mismatched.
    pub fn restore_params(&mut self, params: &[(Vec<f32>, Vec<usize>)]) -> Result<()> {
        // Count how many parameter tensors the layers own
        let layer_param_count: usize = self.layers.iter().map(|l| l.params().len()).sum();
        let has_log_std = self.log_std.is_some();
        let expected = layer_param_count + if has_log_std { 1 } else { 0 };
        if params.len() != expected {
            return Err(NeuralError::InvalidArchitecture(format!(
                "PolicyNetwork restore_params: expected {expected} tensors, got {}",
                params.len()
            )));
        }

        let mut idx = 0usize;
        for layer in &mut self.layers {
            let n = layer.params().len();
            let slice = &params[idx..idx + n];
            let arrays: Vec<scirs2_core::ndarray::ArrayD<f32>> = slice
                .iter()
                .map(|(data, shape)| {
                    let dim = scirs2_core::ndarray::IxDyn(shape);
                    scirs2_core::ndarray::ArrayD::from_shape_vec(dim, data.clone()).map_err(|e| {
                        NeuralError::InvalidArchitecture(format!(
                            "PolicyNetwork: cannot rebuild param array: {e}"
                        ))
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            layer.set_params(&arrays)?;
            idx += n;
        }

        if has_log_std {
            let (data, shape) = &params[idx];
            if shape.len() != 1 || shape[0] != data.len() {
                return Err(NeuralError::InvalidArchitecture(
                    "PolicyNetwork: log_std shape mismatch".to_string(),
                ));
            }
            self.log_std = Some(Array1::from_vec(data.clone()));
        }
        Ok(())
    }
}

impl Policy for PolicyNetwork {
    fn sample_action(&self, state: &ArrayView1<f32>) -> Result<Array1<f32>> {
        PolicyNetwork::sample_action(self, state)
    }

    fn log_prob(&self, state: &ArrayView1<f32>, action: &ArrayView1<f32>) -> Result<f32> {
        PolicyNetwork::log_prob(self, state, action)
    }

    fn parameters(&self) -> Vec<Array2<f32>> {
        // In a full implementation these would be the actual weight matrices.
        Vec::new()
    }

    fn set_parameters(&mut self, _params: &[Array2<f32>]) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_discrete_policy_network() {
        let policy = PolicyNetwork::new(4, 2, vec![8], false).expect("create ok");
        let state = Array1::from_vec(vec![0.1, -0.2, 0.5, 0.3]);
        let action = policy.sample_action(&state.view()).expect("sample ok");
        assert_eq!(action.len(), 2);
        // One-hot: exactly one entry should be 1.0
        assert_eq!(action.iter().filter(|&&x| x > 0.5).count(), 1);
    }

    #[test]
    fn test_continuous_policy_network() {
        let policy = PolicyNetwork::new(4, 3, vec![8], true).expect("create ok");
        let state = Array1::from_vec(vec![0.1, -0.2, 0.5, 0.3]);
        let action = policy.sample_action(&state.view()).expect("sample ok");
        assert_eq!(action.len(), 3);
    }

    #[test]
    fn test_policy_log_prob_discrete() {
        let policy = PolicyNetwork::new(4, 2, vec![8], false).expect("create ok");
        let state = Array1::from_vec(vec![0.1, -0.2, 0.5, 0.3]);
        let action = Array1::from_vec(vec![1.0, 0.0]);
        let lp = policy
            .log_prob(&state.view(), &action.view())
            .expect("log_prob ok");
        assert!(lp <= 0.0, "log-prob of a valid action must be ≤ 0");
    }

    #[test]
    fn test_policy_log_prob_continuous() {
        let policy = PolicyNetwork::new(4, 3, vec![8], true).expect("create ok");
        let state = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0]);
        let action = Array1::from_vec(vec![0.0, 0.0, 0.0]);
        let lp = policy
            .log_prob(&state.view(), &action.view())
            .expect("log_prob ok");
        assert!(lp.is_finite());
    }

    #[test]
    fn test_policy_gradient_loss() {
        let policy = PolicyNetwork::new(2, 2, vec![4], false).expect("create ok");
        let pg = PolicyGradient::new(policy, 1e-3);
        let log_probs = vec![-0.5, -0.3, -0.7];
        let returns = vec![1.0, 2.0, 0.5];
        let loss = pg.compute_loss(&log_probs, &returns);
        assert!(loss.is_finite());
        assert!(loss >= 0.0, "REINFORCE loss should be positive");
    }

    /// Verify that consecutive samples from a continuous policy are stochastic
    /// (i.e. not identical), confirming the stub that returned just the mean
    /// has been replaced with proper N(mean, std) sampling.
    #[test]
    fn test_continuous_policy_stochastic_sampling() {
        let policy = PolicyNetwork::new(4, 3, vec![8], true).expect("create ok");
        let state = Array1::from_vec(vec![0.1, -0.2, 0.5, 0.3]);

        // Draw multiple samples and confirm they are not all identical.
        let mut all_same = true;
        let first = policy.sample_action(&state.view()).expect("sample 1");
        for _ in 0..10 {
            let next = policy.sample_action(&state.view()).expect("sample n");
            if next
                .iter()
                .zip(first.iter())
                .any(|(a, b)| (a - b).abs() > 1e-9)
            {
                all_same = false;
                break;
            }
        }
        assert!(!all_same, "continuous policy should sample stochastically");
    }

    /// Verify that the parameter round-trip for PolicyNetwork is lossless.
    #[test]
    fn test_policy_network_collect_restore_params() {
        let mut policy = PolicyNetwork::new(4, 3, vec![8], true).expect("create ok");
        let before = policy.collect_params();

        // Restore into itself (idempotent)
        policy.restore_params(&before).expect("restore ok");
        let after = policy.collect_params();

        assert_eq!(before.len(), after.len(), "param count must match");
        for (orig, loaded) in before.iter().zip(after.iter()) {
            assert_eq!(orig.1, loaded.1, "shape mismatch");
            for (&a, &b) in orig.0.iter().zip(loaded.0.iter()) {
                assert!((a - b).abs() < 1e-10, "param changed on round-trip");
            }
        }
    }
}
