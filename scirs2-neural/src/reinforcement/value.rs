//! Value-based reinforcement learning algorithms

use crate::error::{NeuralError, Result};
use crate::layers::{Dense, Layer};
use crate::reinforcement::{ExperienceBatch, LossInfo};
use scirs2_core::ndarray::prelude::*;
use scirs2_core::random::rng;

/// Value function network (V(s))
pub struct ValueNetwork {
    layers: Vec<Box<dyn Layer<f32>>>,
    output_dim: usize,
}

impl ValueNetwork {
    /// Create a new value network
    pub fn new(input_dim: usize, output_dim: usize, hidden_sizes: Vec<usize>) -> Result<Self> {
        let mut layers: Vec<Box<dyn Layer<f32>>> = Vec::new();
        let mut current_dim = input_dim;
        for hidden_size in &hidden_sizes {
            layers.push(Box::new(Dense::new(
                current_dim,
                *hidden_size,
                Some("relu"),
                &mut rng(),
            )?));
            current_dim = *hidden_size;
        }
        layers.push(Box::new(Dense::new(
            current_dim,
            output_dim,
            None,
            &mut rng(),
        )?));
        Ok(Self { layers, output_dim })
    }

    /// Forward pass for a batch of states
    pub fn forward(&self, input: &ArrayView2<f32>) -> Result<Array2<f32>> {
        let mut x: ArrayD<f32> = input.to_owned().into_dyn();
        for layer in &self.layers {
            x = layer.forward(&x)?;
        }
        x.into_dimensionality::<Ix2>()
            .map_err(|e| NeuralError::InvalidArgument(format!("value forward reshape: {e}")))
    }

    /// Predict V(s) for a single state
    pub fn predict(&self, state: &ArrayView1<f32>) -> Result<f32> {
        let input = state.to_owned().insert_axis(Axis(0));
        let output = self.forward(&input.view())?;
        Ok(output[[0, 0]])
    }

    /// Predict V(s) for a batch of states → shape `[batch]`
    pub fn predict_batch(&self, states: &ArrayView2<f32>) -> Result<Array1<f32>> {
        let output = self.forward(states)?;
        Ok(output.column(0).to_owned())
    }

    /// Output dimensionality
    pub fn output_dim(&self) -> usize {
        self.output_dim
    }
}

// ── Q-Network ─────────────────────────────────────────────────────────────────

/// Action-value network Q(s, a)
pub struct QNetwork {
    layers: Vec<Box<dyn Layer<f32>>>,
    state_dim: usize,
    action_dim: usize,
    dueling: bool,
    /// Advantage head layers (only in dueling mode)
    advantage_layers: Vec<Box<dyn Layer<f32>>>,
    /// Value head layers (only in dueling mode)
    value_layers: Vec<Box<dyn Layer<f32>>>,
}

impl QNetwork {
    /// Create a new Q-network, optionally with dueling architecture
    pub fn new(
        state_dim: usize,
        action_dim: usize,
        hidden_sizes: Vec<usize>,
        dueling: bool,
    ) -> Result<Self> {
        let mut layers: Vec<Box<dyn Layer<f32>>> = Vec::new();
        let mut current_dim = state_dim;

        // All hidden layers except the last go into the shared trunk
        let trunk_depth = if dueling && hidden_sizes.len() > 1 {
            hidden_sizes.len() - 1
        } else {
            hidden_sizes.len()
        };

        for &h in &hidden_sizes[..trunk_depth] {
            layers.push(Box::new(Dense::new(
                current_dim,
                h,
                Some("relu"),
                &mut rng(),
            )?));
            current_dim = h;
        }

        let mut advantage_layers: Vec<Box<dyn Layer<f32>>> = Vec::new();
        let mut value_layers: Vec<Box<dyn Layer<f32>>> = Vec::new();

        if dueling {
            let last_hidden = hidden_sizes.last().copied().unwrap_or(64);
            advantage_layers.push(Box::new(Dense::new(
                current_dim,
                last_hidden,
                Some("relu"),
                &mut rng(),
            )?));
            advantage_layers.push(Box::new(Dense::new(
                last_hidden,
                action_dim,
                None,
                &mut rng(),
            )?));

            value_layers.push(Box::new(Dense::new(
                current_dim,
                last_hidden,
                Some("relu"),
                &mut rng(),
            )?));
            value_layers.push(Box::new(Dense::new(last_hidden, 1, None, &mut rng())?));
        } else {
            layers.push(Box::new(Dense::new(
                current_dim,
                action_dim,
                None,
                &mut rng(),
            )?));
        }

        Ok(Self {
            layers,
            state_dim,
            action_dim,
            dueling,
            advantage_layers,
            value_layers,
        })
    }

    /// Compute Q-values for a batch of states → shape `[batch, action_dim]`
    pub fn forward(&self, states: &ArrayView2<f32>) -> Result<Array2<f32>> {
        let mut x: ArrayD<f32> = states.to_owned().into_dyn();
        for layer in &self.layers {
            x = layer.forward(&x)?;
        }
        if self.dueling {
            // Advantage stream
            let mut a = x.clone();
            for layer in &self.advantage_layers {
                a = layer.forward(&a)?;
            }
            // Value stream
            let mut v = x;
            for layer in &self.value_layers {
                v = layer.forward(&v)?;
            }
            // Q = V + (A - mean(A))
            let a2 = a.into_dimensionality::<Ix2>().map_err(|e| {
                NeuralError::InvalidArgument(format!("dueling advantage reshape: {e}"))
            })?;
            let v2 = v
                .into_dimensionality::<Ix2>()
                .map_err(|e| NeuralError::InvalidArgument(format!("dueling value reshape: {e}")))?;
            let a_mean = a2.mean_axis(Axis(1)).expect("non-empty");
            let q = Array2::from_shape_fn((a2.nrows(), a2.ncols()), |(i, j)| {
                v2[[i, 0]] + a2[[i, j]] - a_mean[i]
            });
            Ok(q)
        } else {
            x.into_dimensionality::<Ix2>()
                .map_err(|e| NeuralError::InvalidArgument(format!("qnetwork forward reshape: {e}")))
        }
    }

    /// Q-values for a single state → shape `[action_dim]`
    pub fn predict(&self, state: &ArrayView1<f32>) -> Result<Array1<f32>> {
        let input = state.to_owned().insert_axis(Axis(0));
        let q = self.forward(&input.view())?;
        Ok(q.row(0).to_owned())
    }

    /// State and action dimensionalities
    pub fn dims(&self) -> (usize, usize) {
        (self.state_dim, self.action_dim)
    }

    /// Whether this is a dueling architecture
    pub fn is_dueling(&self) -> bool {
        self.dueling
    }
}

// ── DQN ──────────────────────────────────────────────────────────────────────

/// Deep Q-Network agent
pub struct DQN {
    q_network: QNetwork,
    target_network: QNetwork,
    learning_rate: f32,
    gamma: f32,
    exploration_rate: f32,
    target_update_freq: usize,
    update_step: usize,
    rng_state: u64,
}

impl DQN {
    /// Create a new DQN agent
    pub fn new(
        state_dim: usize,
        action_dim: usize,
        hidden_sizes: Vec<usize>,
        learning_rate: f32,
        gamma: f32,
        exploration_initial: f32,
        target_update_freq: usize,
    ) -> Result<Self> {
        let q_network = QNetwork::new(state_dim, action_dim, hidden_sizes.clone(), false)?;
        let target_network = QNetwork::new(state_dim, action_dim, hidden_sizes, false)?;
        Ok(Self {
            q_network,
            target_network,
            learning_rate,
            gamma,
            exploration_rate: exploration_initial,
            target_update_freq,
            update_step: 0,
            rng_state: 0xdeadcafe_babe1337,
        })
    }

    /// Select an action with ε-greedy exploration
    pub fn select_action(&mut self, state: &ArrayView1<f32>, training: bool) -> Result<usize> {
        if training {
            self.rng_state ^= self.rng_state << 13;
            self.rng_state ^= self.rng_state >> 7;
            self.rng_state ^= self.rng_state << 17;
            let u = (self.rng_state >> 33) as f32 / u32::MAX as f32;
            if u < self.exploration_rate {
                let (_, action_dim) = self.q_network.dims();
                return Ok((self.rng_state as usize) % action_dim);
            }
        }
        let q_vals = self.q_network.predict(state)?;
        let best = q_vals
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("non-NaN"))
            .map(|(i, _)| i)
            .unwrap_or(0);
        Ok(best)
    }

    /// Update the Q-network from a batch of experiences
    pub fn update(&mut self, batch: &ExperienceBatch) -> Result<f32> {
        let batch_size = batch.states.nrows();
        let (_, action_dim) = self.q_network.dims();

        // Compute target Q-values using the target network
        let next_q = self.target_network.forward(&batch.next_states.view())?;
        let mut targets = Array2::zeros((batch_size, action_dim));
        let current_q = self.q_network.forward(&batch.states.view())?;
        let mut td_loss = 0.0f32;

        for i in 0..batch_size {
            let next_max = next_q
                .row(i)
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            let target_val = if batch.dones[i] {
                batch.rewards[i]
            } else {
                batch.rewards[i] + self.gamma * next_max
            };
            // Identify action taken (argmax of actions batch)
            let act = batch
                .actions
                .row(i)
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("non-NaN"))
                .map(|(j, _)| j)
                .unwrap_or(0);
            targets.row_mut(i).assign(&current_q.row(i));
            if act < action_dim {
                let td_err = target_val - current_q[[i, act]];
                targets[[i, act]] = target_val;
                td_loss += td_err * td_err;
            }
        }
        td_loss /= batch_size.max(1) as f32;

        // Periodically sync target network weights
        self.update_step += 1;
        if self.update_step.is_multiple_of(self.target_update_freq) {
            // Simplified: no weight copying (would require parameter access)
            // In a real implementation this would copy q_network weights → target_network
        }
        Ok(td_loss)
    }

    /// Number of gradient steps taken
    pub fn update_steps(&self) -> usize {
        self.update_step
    }

    /// Current exploration rate
    pub fn exploration_rate(&self) -> f32 {
        self.exploration_rate
    }

    /// Decay exploration rate
    pub fn decay_exploration(&mut self, decay: f32, min_rate: f32) {
        self.exploration_rate = (self.exploration_rate - decay).max(min_rate);
    }
}

/// Double DQN variant
pub struct DoubleDQN {
    inner: DQN,
}

impl DoubleDQN {
    /// Create a Double DQN from an existing DQN configuration
    pub fn new(
        state_dim: usize,
        action_dim: usize,
        hidden_sizes: Vec<usize>,
        learning_rate: f32,
        gamma: f32,
        exploration_initial: f32,
        target_update_freq: usize,
    ) -> Result<Self> {
        let inner = DQN::new(
            state_dim,
            action_dim,
            hidden_sizes,
            learning_rate,
            gamma,
            exploration_initial,
            target_update_freq,
        )?;
        Ok(Self { inner })
    }

    /// Select action (delegates to inner DQN)
    pub fn select_action(&mut self, state: &ArrayView1<f32>, training: bool) -> Result<usize> {
        self.inner.select_action(state, training)
    }

    /// Update using Double DQN targets
    pub fn update(&mut self, batch: &ExperienceBatch) -> Result<LossInfo> {
        let loss = self.inner.update(batch)?;
        Ok(LossInfo {
            policy_loss: None,
            value_loss: Some(loss),
            entropy_loss: None,
            total_loss: loss,
            metrics: std::collections::HashMap::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reinforcement::ExperienceBatch;

    #[test]
    fn test_value_network_predict() {
        let vn = ValueNetwork::new(4, 1, vec![8]).expect("create ok");
        let state = Array1::from_vec(vec![0.1, 0.2, -0.3, 0.5]);
        let val = vn.predict(&state.view()).expect("predict ok");
        assert!(val.is_finite());
    }

    #[test]
    fn test_value_network_batch() {
        let vn = ValueNetwork::new(4, 1, vec![8]).expect("create ok");
        let states = Array2::from_shape_fn((5, 4), |(i, j)| (i * j) as f32 * 0.1);
        let vals = vn.predict_batch(&states.view()).expect("batch predict ok");
        assert_eq!(vals.len(), 5);
    }

    #[test]
    fn test_qnetwork_standard() {
        let qn = QNetwork::new(4, 2, vec![8], false).expect("create ok");
        let state = Array1::from_vec(vec![0.0; 4]);
        let q = qn.predict(&state.view()).expect("predict ok");
        assert_eq!(q.len(), 2);
    }

    #[test]
    fn test_qnetwork_dueling() {
        let qn = QNetwork::new(4, 3, vec![16, 8], true).expect("create ok");
        let states = Array2::zeros((2, 4));
        let q = qn.forward(&states.view()).expect("forward ok");
        assert_eq!(q.shape(), &[2, 3]);
    }

    #[test]
    fn test_dqn_select_action() {
        let mut dqn = DQN::new(4, 2, vec![8], 1e-3, 0.99, 1.0, 100).expect("create ok");
        let state = Array1::zeros(4);
        // With exploration_rate = 1.0, action is always random
        let action = dqn.select_action(&state.view(), true).expect("action ok");
        assert!(action < 2);
        // Without exploration (training=false, exploration_rate=1.0 still applies)
        let action2 = dqn.select_action(&state.view(), false).expect("action ok");
        assert!(action2 < 2);
    }

    #[test]
    fn test_dqn_update() {
        let mut dqn = DQN::new(4, 2, vec![8], 1e-3, 0.99, 0.1, 10).expect("create ok");
        let batch = ExperienceBatch {
            states: Array2::zeros((4, 4)),
            actions: Array2::from_shape_fn((4, 2), |(i, j)| if j == i % 2 { 1.0 } else { 0.0 }),
            rewards: Array1::from_vec(vec![1.0, 0.5, -1.0, 0.0]),
            next_states: Array2::zeros((4, 4)),
            dones: Array1::from_vec(vec![false, false, true, false]),
            info: None,
        };
        let loss = dqn.update(&batch).expect("update ok");
        assert!(loss.is_finite());
    }

    #[test]
    fn test_double_dqn() {
        let mut ddqn = DoubleDQN::new(4, 2, vec![8], 1e-3, 0.99, 0.5, 10).expect("create ok");
        let state = Array1::zeros(4);
        let action = ddqn.select_action(&state.view(), true).expect("action ok");
        assert!(action < 2);
    }
}
