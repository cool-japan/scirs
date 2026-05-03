//! Model-based reinforcement learning algorithms

use crate::error::{NeuralError, Result};
use crate::layers::{Dense, Layer};
use crate::reinforcement::environments::Environment;
use crate::reinforcement::policy::PolicyNetwork;
use scirs2_core::ndarray::prelude::*;
use scirs2_core::random::rng;

/// Dynamics model predicting next state and reward from (state, action)
pub struct DynamicsModel {
    state_dim: usize,
    action_dim: usize,
    layers: Vec<Box<dyn Layer<f32>>>,
    reward_head: Box<dyn Layer<f32>>,
    uncertainty_estimation: bool,
}

impl DynamicsModel {
    /// Create a new dynamics model
    pub fn new(
        state_dim: usize,
        action_dim: usize,
        hidden_sizes: Vec<usize>,
        uncertainty_estimation: bool,
    ) -> Result<Self> {
        let input_dim = state_dim + action_dim;
        let mut layers: Vec<Box<dyn Layer<f32>>> = Vec::new();
        let mut current_dim = input_dim;
        for &hidden_size in &hidden_sizes {
            layers.push(Box::new(Dense::new(
                current_dim,
                hidden_size,
                Some("relu"),
                &mut rng(),
            )?));
            current_dim = hidden_size;
        }
        let output_dim = if uncertainty_estimation {
            state_dim * 2 // mean and log-variance
        } else {
            state_dim
        };
        layers.push(Box::new(Dense::new(
            current_dim,
            output_dim,
            None,
            &mut rng(),
        )?));
        let reward_head = Box::new(Dense::new(current_dim, 1, None, &mut rng())?);
        Ok(Self {
            state_dim,
            action_dim,
            layers,
            reward_head,
            uncertainty_estimation,
        })
    }

    /// Predict `(next_state, reward, optional_uncertainty)`
    pub fn predict(
        &self,
        state: &ArrayView1<f32>,
        action: &ArrayView1<f32>,
    ) -> Result<(Array1<f32>, f32, Option<Array1<f32>>)> {
        let mut input_vec = Vec::with_capacity(self.state_dim + self.action_dim);
        input_vec.extend_from_slice(state.as_slice().unwrap_or_default());
        input_vec.extend_from_slice(action.as_slice().unwrap_or_default());
        let input: ArrayD<f32> = Array2::from_shape_vec((1, input_vec.len()), input_vec)
            .map_err(|e| NeuralError::InvalidArgument(format!("dynamics input shape: {e}")))?
            .into_dyn();

        let mut x = input.clone();
        // Pass through all hidden layers (all but last)
        for layer in &self.layers[..self.layers.len().saturating_sub(1)] {
            x = layer.forward(&x)?;
        }

        // Reward head
        let reward_out = self.reward_head.forward(&x)?;
        let reward = reward_out.iter().next().copied().unwrap_or(0.0);

        // State head (last layer)
        let state_out = if let Some(last) = self.layers.last() {
            last.forward(&x)?
        } else {
            x
        };
        let state_1d: Array1<f32> = state_out
            .into_dimensionality::<Ix2>()
            .map_err(|e| NeuralError::InvalidArgument(format!("dynamics state reshape: {e}")))?
            .row(0)
            .to_owned();

        let (next_state, uncertainty) =
            if self.uncertainty_estimation && state_1d.len() >= self.state_dim * 2 {
                let mean = state_1d.slice(s![..self.state_dim]).to_owned();
                let log_var = state_1d
                    .slice(s![self.state_dim..self.state_dim * 2])
                    .to_owned();
                let std = log_var.mapv(|v| (v / 2.0).exp());
                (mean, Some(std))
            } else {
                let ns = if state_1d.len() >= self.state_dim {
                    state_1d.slice(s![..self.state_dim]).to_owned()
                } else {
                    state_1d
                };
                (ns, None)
            };
        Ok((next_state, reward, uncertainty))
    }

    /// State and action dimensionalities
    pub fn dims(&self) -> (usize, usize) {
        (self.state_dim, self.action_dim)
    }
}

/// World model wrapping one or more dynamics models for ensemble planning
pub struct WorldModel {
    models: Vec<DynamicsModel>,
    n_models: usize,
}

impl WorldModel {
    /// Create an ensemble of `n_models` dynamics models
    pub fn new(
        state_dim: usize,
        action_dim: usize,
        hidden_sizes: Vec<usize>,
        n_models: usize,
    ) -> Result<Self> {
        let mut models = Vec::with_capacity(n_models);
        for _ in 0..n_models {
            models.push(DynamicsModel::new(
                state_dim,
                action_dim,
                hidden_sizes.clone(),
                false,
            )?);
        }
        Ok(Self { models, n_models })
    }

    /// Ensemble-average prediction
    pub fn predict(
        &self,
        state: &ArrayView1<f32>,
        action: &ArrayView1<f32>,
    ) -> Result<(Array1<f32>, f32)> {
        let mut next_sum = Array1::zeros(state.len());
        let mut reward_sum = 0.0f32;
        for model in &self.models {
            let (ns, r, _) = model.predict(state, action)?;
            let len = ns.len().min(next_sum.len());
            for i in 0..len {
                next_sum[i] += ns[i];
            }
            reward_sum += r;
        }
        next_sum /= self.n_models.max(1) as f32;
        reward_sum /= self.n_models.max(1) as f32;
        Ok((next_sum, reward_sum))
    }
}

/// Dyna-Q: model-based RL with simulated rollouts
pub struct Dyna {
    world_model: WorldModel,
    policy: PolicyNetwork,
    planning_horizon: usize,
    n_simulations: usize,
}

impl Dyna {
    /// Create a new Dyna agent
    pub fn new(
        state_dim: usize,
        action_dim: usize,
        hidden_sizes: Vec<usize>,
        continuous: bool,
        planning_horizon: usize,
        n_simulations: usize,
    ) -> Result<Self> {
        let world_model = WorldModel::new(state_dim, action_dim, hidden_sizes.clone(), 1)?;
        let policy = PolicyNetwork::new(state_dim, action_dim, hidden_sizes, continuous)?;
        Ok(Self {
            world_model,
            policy,
            planning_horizon,
            n_simulations,
        })
    }

    /// Select action from the policy
    pub fn act(&self, state: &ArrayView1<f32>) -> Result<Array1<f32>> {
        self.policy.sample_action(state)
    }

    /// Run model-based rollouts and return accumulated simulated reward
    pub fn plan(&self, start_state: &ArrayView1<f32>) -> Result<f32> {
        let mut total_reward = 0.0f32;
        for _ in 0..self.n_simulations {
            let mut state = start_state.to_owned();
            for _ in 0..self.planning_horizon {
                let action = self.policy.sample_action(&state.view())?;
                let (next_state, reward) =
                    self.world_model.predict(&state.view(), &action.view())?;
                total_reward += reward;
                state = next_state;
            }
        }
        Ok(total_reward / self.n_simulations.max(1) as f32)
    }
}

/// Model Predictive Control agent
pub struct MPC {
    world_model: WorldModel,
    horizon: usize,
    n_samples: usize,
    action_dim: usize,
    rng_state: u64,
}

impl MPC {
    /// Create a new MPC agent
    pub fn new(
        state_dim: usize,
        action_dim: usize,
        hidden_sizes: Vec<usize>,
        horizon: usize,
        n_samples: usize,
    ) -> Result<Self> {
        let world_model = WorldModel::new(state_dim, action_dim, hidden_sizes, 1)?;
        Ok(Self {
            world_model,
            horizon,
            n_samples,
            action_dim,
            rng_state: 0x1a2b3c4d_5e6f7a8b,
        })
    }

    /// Select the best action via random shooting
    pub fn act(&mut self, state: &ArrayView1<f32>) -> Result<Array1<f32>> {
        let mut best_action = Array1::zeros(self.action_dim);
        let mut best_reward = f32::NEG_INFINITY;

        for _ in 0..self.n_samples {
            let mut sim_state = state.to_owned();
            let mut action_seq = Array1::zeros(self.action_dim);
            let mut total_reward = 0.0f32;

            // Generate random action sequence and simulate
            for h in 0..self.horizon {
                // Random action in [-1, 1]
                self.rng_state ^= self.rng_state << 13;
                self.rng_state ^= self.rng_state >> 7;
                self.rng_state ^= self.rng_state << 17;
                let action: Array1<f32> = Array1::from_iter((0..self.action_dim).map(|_| {
                    self.rng_state ^= self.rng_state << 13;
                    self.rng_state ^= self.rng_state >> 7;
                    self.rng_state ^= self.rng_state << 17;
                    (self.rng_state >> 33) as f32 / u32::MAX as f32 * 2.0 - 1.0
                }));
                if h == 0 {
                    action_seq = action.clone();
                }
                let (next_state, reward) = self
                    .world_model
                    .predict(&sim_state.view(), &action.view())?;
                total_reward += reward;
                sim_state = next_state;
            }
            if total_reward > best_reward {
                best_reward = total_reward;
                best_action = action_seq;
            }
        }
        Ok(best_action)
    }

    /// Planning horizon
    pub fn horizon(&self) -> usize {
        self.horizon
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dynamics_model_predict() {
        let dm = DynamicsModel::new(4, 2, vec![8], false).expect("create ok");
        let state = Array1::zeros(4);
        let action = Array1::from_vec(vec![0.5, -0.5]);
        let (next_state, reward, unc) = dm
            .predict(&state.view(), &action.view())
            .expect("predict ok");
        assert_eq!(next_state.len(), 4);
        assert!(reward.is_finite());
        assert!(unc.is_none());
    }

    #[test]
    fn test_dynamics_model_with_uncertainty() {
        let dm = DynamicsModel::new(4, 2, vec![8], true).expect("create ok");
        let state = Array1::zeros(4);
        let action = Array1::zeros(2);
        let (ns, r, unc) = dm
            .predict(&state.view(), &action.view())
            .expect("predict ok");
        assert_eq!(ns.len(), 4);
        assert!(r.is_finite());
        // Uncertainty is present when output is 2*state_dim
        if let Some(u) = unc {
            assert_eq!(u.len(), 4);
        }
    }

    #[test]
    fn test_world_model_ensemble_predict() {
        let wm = WorldModel::new(4, 2, vec![8], 3).expect("create ok");
        let state = Array1::zeros(4);
        let action = Array1::zeros(2);
        let (ns, r) = wm
            .predict(&state.view(), &action.view())
            .expect("predict ok");
        assert_eq!(ns.len(), 4);
        assert!(r.is_finite());
    }

    #[test]
    fn test_dyna_act() {
        let dyna = Dyna::new(4, 2, vec![8], false, 5, 3).expect("create ok");
        let state = Array1::zeros(4);
        let action = dyna.act(&state.view()).expect("act ok");
        assert_eq!(action.len(), 2);
    }

    #[test]
    fn test_mpc_act() {
        let mut mpc = MPC::new(4, 2, vec![8], 3, 5).expect("create ok");
        let state = Array1::zeros(4);
        let action = mpc.act(&state.view()).expect("act ok");
        assert_eq!(action.len(), 2);
    }
}
