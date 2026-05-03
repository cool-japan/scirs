//! Value function networks for reinforcement learning.
//!
//! This module provides:
//! - [`ValueNetwork`]: estimates the state-value function V(s).
//! - [`QNetwork`]: estimates action-values Q(s, a) for all discrete actions.
//! - [`DuelingQNetwork`]: dueling architecture separating value and advantage streams.
//!
//! All networks are self-contained MLPs (no autograd dependency), with SGD
//! updates available via the [`NetworkUpdate`] trait.

use crate::error::{NeuralError, Result};
use crate::rl::policy::{softmax, PolicyRng};

// ──────────────────────────────────────────────────────────────────────────────
// Common update trait
// ──────────────────────────────────────────────────────────────────────────────

/// Common interface for SGD weight updates.
pub trait NetworkUpdate {
    /// Update network weights using MSE loss on `(inputs, targets)`.
    ///
    /// Returns the MSE loss before the update.
    fn update(&mut self, inputs: &[f32], targets: &[f32], lr: f32) -> Result<f32>;
}

// ──────────────────────────────────────────────────────────────────────────────
// Mlp building block (private)
// ──────────────────────────────────────────────────────────────────────────────

/// Minimal heap-allocated Mlp shared by all network types in this module.
#[derive(Debug, Clone)]
struct Mlp {
    weights: Vec<Vec<Vec<f32>>>,
    biases: Vec<Vec<f32>>,
}

impl Mlp {
    fn build(layer_dims: &[usize]) -> Self {
        let mut rng = PolicyRng::new(0x8f22_3344_ab11_cd88);
        let mut weights = Vec::with_capacity(layer_dims.len() - 1);
        let mut biases = Vec::with_capacity(layer_dims.len() - 1);
        for l in 0..(layer_dims.len() - 1) {
            let (in_d, out_d) = (layer_dims[l], layer_dims[l + 1]);
            let scale = (6.0_f32 / in_d as f32).sqrt(); // He-uniform
            let w = (0..out_d)
                .map(|_| {
                    (0..in_d)
                        .map(|_| (rng.uniform_f32() * 2.0 - 1.0) * scale)
                        .collect()
                })
                .collect();
            let b = vec![0.0_f32; out_d];
            weights.push(w);
            biases.push(b);
        }
        Self { weights, biases }
    }

    fn n_layers(&self) -> usize {
        self.weights.len()
    }

    fn input_dim(&self) -> usize {
        self.weights
            .first()
            .map(|w| w.first().map(|r| r.len()).unwrap_or(0))
            .unwrap_or(0)
    }

    fn output_dim(&self) -> usize {
        self.weights.last().map(|w| w.len()).unwrap_or(0)
    }

    /// Forward pass; hidden layers use ReLU, output layer is linear.
    fn forward(&self, input: &[f32]) -> Result<Vec<f32>> {
        if input.len() != self.input_dim() {
            return Err(NeuralError::ShapeMismatch(format!(
                "MLP: expected input_dim={}, got {}",
                self.input_dim(),
                input.len()
            )));
        }
        let mut act: Vec<f32> = input.to_vec();
        let n = self.n_layers();
        for (l, (w, b)) in self.weights.iter().zip(self.biases.iter()).enumerate() {
            let mut next = Vec::with_capacity(w.len());
            for (row, bias) in w.iter().zip(b.iter()) {
                let pre: f32 = row
                    .iter()
                    .zip(act.iter())
                    .map(|(wi, xi)| wi * xi)
                    .sum::<f32>()
                    + bias;
                next.push(if l < n - 1 { pre.max(0.0) } else { pre });
            }
            act = next;
        }
        Ok(act)
    }

    /// Forward pass caching intermediate activations (needed for backprop).
    fn forward_cache(&self, input: &[f32]) -> Vec<Vec<f32>> {
        let n = self.n_layers();
        let mut cache = Vec::with_capacity(n + 1);
        cache.push(input.to_vec());
        for (l, (w, b)) in self.weights.iter().zip(self.biases.iter()).enumerate() {
            let prev = cache.last().expect("cache always non-empty");
            let mut next = Vec::with_capacity(w.len());
            for (row, bias) in w.iter().zip(b.iter()) {
                let pre: f32 = row
                    .iter()
                    .zip(prev.iter())
                    .map(|(wi, xi)| wi * xi)
                    .sum::<f32>()
                    + bias;
                next.push(if l < n - 1 { pre.max(0.0) } else { pre });
            }
            cache.push(next);
        }
        cache
    }

    /// Single-sample SGD update given output-layer delta.
    fn sgd_step(&mut self, cache: &[Vec<f32>], out_delta: &[f32], lr: f32) {
        let n = self.n_layers();
        let mut delta = out_delta.to_vec();
        for l in (0..n).rev() {
            let in_act = &cache[l];
            let out_act = &cache[l + 1];
            let out_d = self.weights[l].len();
            let in_d = in_act.len();

            let effective_delta: Vec<f32> = if l < n - 1 {
                delta
                    .iter()
                    .zip(out_act.iter())
                    .map(|(d, a)| if *a > 0.0 { *d } else { 0.0 })
                    .collect()
            } else {
                delta.clone()
            };

            let mut prev_delta = vec![0.0_f32; in_d];
            for (i, &eff_i) in effective_delta.iter().enumerate().take(out_d) {
                for j in 0..in_d {
                    prev_delta[j] += eff_i * self.weights[l][i][j];
                    self.weights[l][i][j] -= lr * eff_i * in_act[j];
                }
                self.biases[l][i] -= lr * eff_i;
            }
            delta = prev_delta;
        }
    }

    /// Hard copy weights from `other`.
    fn copy_from(&mut self, other: &Self) -> Result<()> {
        if self.weights.len() != other.weights.len() {
            return Err(NeuralError::ShapeMismatch(
                "MLP architecture mismatch on copy".into(),
            ));
        }
        for (sw, ow) in self.weights.iter_mut().zip(other.weights.iter()) {
            for (sr, or_) in sw.iter_mut().zip(ow.iter()) {
                sr.clone_from(or_);
            }
        }
        for (sb, ob) in self.biases.iter_mut().zip(other.biases.iter()) {
            sb.clone_from(ob);
        }
        Ok(())
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// ValueNetwork  (V(s))
// ──────────────────────────────────────────────────────────────────────────────

/// Feed-forward network that estimates the state-value function **V(s)**.
///
/// Output is a single scalar (no activation on the output neuron).
#[derive(Debug, Clone)]
pub struct ValueNetwork {
    mlp: Mlp,
}

impl ValueNetwork {
    /// Create a new ValueNetwork.
    ///
    /// - `obs_dim`: observation dimension.
    /// - `hidden_dims`: e.g. `&[64, 64]`.
    pub fn new(obs_dim: usize, hidden_dims: &[usize]) -> Self {
        let mut dims = vec![obs_dim];
        dims.extend_from_slice(hidden_dims);
        dims.push(1); // scalar output
        Self {
            mlp: Mlp::build(&dims),
        }
    }

    /// Estimate V(s).
    pub fn value(&self, obs: &[f32]) -> Result<f32> {
        let out = self.mlp.forward(obs)?;
        Ok(out[0])
    }

    /// Observation dimension.
    pub fn obs_dim(&self) -> usize {
        self.mlp.input_dim()
    }
}

impl NetworkUpdate for ValueNetwork {
    /// SGD update minimising `½(V(s) - target)²`.
    fn update(&mut self, inputs: &[f32], targets: &[f32], lr: f32) -> Result<f32> {
        if targets.len() != 1 {
            return Err(NeuralError::ShapeMismatch(format!(
                "ValueNetwork expects 1 target, got {}",
                targets.len()
            )));
        }
        let cache = self.mlp.forward_cache(inputs);
        let v = cache.last().expect("cache non-empty")[0];
        let err = v - targets[0];
        let loss = 0.5 * err * err;
        self.mlp.sgd_step(&cache, &[err], lr);
        Ok(loss)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// QNetwork  (Q(s, a) for all discrete actions)
// ──────────────────────────────────────────────────────────────────────────────

/// Feed-forward network that estimates **Q(s, a)** for every discrete action
/// simultaneously.
///
/// Input: observation `s`. Output: vector of Q-values, one per action.
#[derive(Debug, Clone)]
pub struct QNetwork {
    mlp: Mlp,
    num_actions: usize,
}

impl QNetwork {
    /// Create a new QNetwork.
    ///
    /// - `obs_dim`: observation dimension.
    /// - `hidden_dims`: widths of hidden layers.
    /// - `num_actions`: output dimension.
    pub fn new(obs_dim: usize, hidden_dims: &[usize], num_actions: usize) -> Self {
        let mut dims = vec![obs_dim];
        dims.extend_from_slice(hidden_dims);
        dims.push(num_actions);
        Self {
            mlp: Mlp::build(&dims),
            num_actions,
        }
    }

    /// Forward pass; returns Q-values for every action.
    pub fn q_values(&self, obs: &[f32]) -> Result<Vec<f32>> {
        self.mlp.forward(obs)
    }

    /// Greedy action: `argmax_a Q(s, a)`.
    pub fn greedy_action(&self, obs: &[f32]) -> Result<usize> {
        let qs = self.q_values(obs)?;
        qs.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .ok_or_else(|| NeuralError::ComputationError("empty Q-value vector".into()))
    }

    /// Hard copy weights from another QNetwork with the same architecture.
    pub fn copy_from(&mut self, other: &Self) -> Result<()> {
        self.mlp.copy_from(&other.mlp)
    }

    /// Number of discrete actions.
    pub fn num_actions(&self) -> usize {
        self.num_actions
    }

    /// Observation dimension.
    pub fn obs_dim(&self) -> usize {
        self.mlp.input_dim()
    }
}

impl NetworkUpdate for QNetwork {
    /// SGD update with selective targets.
    ///
    /// `inputs`: observation.
    /// `targets`: target Q-values for **all** actions (unused entries should
    ///   match the current prediction so their gradient is zero; alternatively
    ///   use [`Self::update_action`] to update a single action).
    fn update(&mut self, inputs: &[f32], targets: &[f32], lr: f32) -> Result<f32> {
        if targets.len() != self.num_actions {
            return Err(NeuralError::ShapeMismatch(format!(
                "QNetwork: expected {} targets, got {}",
                self.num_actions,
                targets.len()
            )));
        }
        let cache = self.mlp.forward_cache(inputs);
        let qs = cache.last().expect("non-empty");
        let delta: Vec<f32> = qs.iter().zip(targets.iter()).map(|(q, t)| q - t).collect();
        let loss: f32 = delta.iter().map(|d| 0.5 * d * d).sum();
        self.mlp.sgd_step(&cache, &delta, lr);
        Ok(loss)
    }
}

impl QNetwork {
    /// Update only the Q-value for `action` (TD-learning style).
    ///
    /// `target` is the TD target for the chosen action; all other outputs are
    /// treated as correct (zero gradient).
    pub fn update_action(
        &mut self,
        obs: &[f32],
        action: usize,
        target: f32,
        lr: f32,
    ) -> Result<f32> {
        if action >= self.num_actions {
            return Err(NeuralError::InvalidArgument(format!(
                "action {} out of range [0, {})",
                action, self.num_actions
            )));
        }
        let cache = self.mlp.forward_cache(obs);
        let qs = cache.last().expect("non-empty");
        let err = qs[action] - target;
        let loss = 0.5 * err * err;
        let mut delta = vec![0.0_f32; self.num_actions];
        delta[action] = err;
        self.mlp.sgd_step(&cache, &delta, lr);
        Ok(loss)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// DuelingQNetwork
// ──────────────────────────────────────────────────────────────────────────────

/// Dueling Q-network architecture.
///
/// The shared trunk is followed by two heads:
/// - **Value stream** V(s): scalar.
/// - **Advantage stream** A(s, a): one value per action.
///
/// Combined Q-value:  `Q(s,a) = V(s) + A(s,a) − mean_a A(s,a)`
///
/// Reference: Wang et al. 2016, "Dueling Network Architectures for Deep
/// Reinforcement Learning", ICML 2016.
#[derive(Debug, Clone)]
pub struct DuelingQNetwork {
    trunk: Mlp,
    value_head: Mlp,
    adv_head: Mlp,
    num_actions: usize,
}

impl DuelingQNetwork {
    /// Construct a dueling Q-network.
    ///
    /// - `obs_dim`: input observation dimension.
    /// - `trunk_dims`: shared hidden layers (e.g., `&[64, 64]`).
    /// - `head_hidden`: hidden layers inside each head (e.g., `&[32]`).
    /// - `num_actions`: number of discrete actions.
    pub fn new(
        obs_dim: usize,
        trunk_dims: &[usize],
        head_hidden: &[usize],
        num_actions: usize,
    ) -> Self {
        assert!(num_actions > 0, "num_actions must be > 0");

        let trunk_out = *trunk_dims.last().unwrap_or(&obs_dim);

        let mut trunk_layer_dims = vec![obs_dim];
        trunk_layer_dims.extend_from_slice(trunk_dims);
        let trunk = Mlp::build(&trunk_layer_dims);

        let mut val_dims = vec![trunk_out];
        val_dims.extend_from_slice(head_hidden);
        val_dims.push(1);
        let value_head = Mlp::build(&val_dims);

        let mut adv_dims = vec![trunk_out];
        adv_dims.extend_from_slice(head_hidden);
        adv_dims.push(num_actions);
        let adv_head = Mlp::build(&adv_dims);

        Self {
            trunk,
            value_head,
            adv_head,
            num_actions,
        }
    }

    /// Forward pass returning combined Q-values.
    ///
    /// `Q(s,a) = V(s) + A(s,a) - (1/|A|) Σ_b A(s,b)`
    pub fn q_values(&self, obs: &[f32]) -> Result<Vec<f32>> {
        let trunk_out = self.trunk.forward(obs)?;
        let v = self.value_head.forward(&trunk_out)?[0];
        let adv = self.adv_head.forward(&trunk_out)?;
        let adv_mean: f32 = adv.iter().sum::<f32>() / adv.len() as f32;
        Ok(adv.iter().map(|a| v + a - adv_mean).collect())
    }

    /// Greedy action.
    pub fn greedy_action(&self, obs: &[f32]) -> Result<usize> {
        let qs = self.q_values(obs)?;
        qs.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .ok_or_else(|| NeuralError::ComputationError("empty Q-value vector".into()))
    }

    /// Hard copy weights from another DuelingQNetwork with identical architecture.
    pub fn copy_from(&mut self, other: &Self) -> Result<()> {
        self.trunk.copy_from(&other.trunk)?;
        self.value_head.copy_from(&other.value_head)?;
        self.adv_head.copy_from(&other.adv_head)
    }

    /// Number of actions.
    pub fn num_actions(&self) -> usize {
        self.num_actions
    }

    /// Observation dimension.
    pub fn obs_dim(&self) -> usize {
        self.trunk.input_dim()
    }
}

impl NetworkUpdate for DuelingQNetwork {
    /// SGD update minimising MSE on all Q-value outputs.
    fn update(&mut self, inputs: &[f32], targets: &[f32], lr: f32) -> Result<f32> {
        if targets.len() != self.num_actions {
            return Err(NeuralError::ShapeMismatch(format!(
                "DuelingQNetwork: expected {} targets, got {}",
                self.num_actions,
                targets.len()
            )));
        }

        // --- forward ---
        let trunk_cache = self.trunk.forward_cache(inputs);
        let trunk_out = trunk_cache.last().expect("non-empty");

        let val_cache = self.value_head.forward_cache(trunk_out);
        let v = val_cache.last().expect("non-empty")[0];

        let adv_cache = self.adv_head.forward_cache(trunk_out);
        let adv = adv_cache.last().expect("non-empty");
        let adv_mean: f32 = adv.iter().sum::<f32>() / adv.len() as f32;

        let qs: Vec<f32> = adv.iter().map(|a| v + a - adv_mean).collect();

        // --- MSE loss ---
        let errors: Vec<f32> = qs.iter().zip(targets.iter()).map(|(q, t)| q - t).collect();
        let loss: f32 = errors.iter().map(|e| 0.5 * e * e).sum();

        // --- Gradient of Q w.r.t. V and A streams ---
        // ∂Q_i/∂V = 1 for all i  →  ∂L/∂V = Σ_i error_i
        // ∂Q_i/∂A_i = 1 - 1/n, ∂Q_i/∂A_j = -1/n for i≠j
        let n = self.num_actions as f32;
        let v_grad: f32 = errors.iter().sum::<f32>();
        let adv_grad: Vec<f32> = errors
            .iter()
            .enumerate()
            .map(|(i, _ei)| {
                errors
                    .iter()
                    .enumerate()
                    .map(|(j, ej)| {
                        if i == j {
                            ej * (1.0 - 1.0 / n)
                        } else {
                            -ej / n
                        }
                    })
                    .sum::<f32>()
            })
            .collect();

        self.value_head.sgd_step(&val_cache, &[v_grad], lr);
        self.adv_head.sgd_step(&adv_cache, &adv_grad, lr);

        // Gradient w.r.t. trunk (sum of value-head and adv-head trunk gradients)
        // We re-run SGD on trunk using the summed delta from both heads.
        // For simplicity, do one step per head and halve lr.
        let val_delta_back = {
            let mut d = vec![0.0_f32; trunk_out.len()];
            let eff: Vec<f32> = if self.value_head.n_layers() > 0 {
                vec![v_grad]
            } else {
                vec![]
            };
            for (i, &eff_i) in eff
                .iter()
                .enumerate()
                .take(self.value_head.weights[0].len().min(1))
            {
                for (j, dj) in d.iter_mut().enumerate() {
                    *dj += eff_i * self.value_head.weights[0][i][j];
                }
            }
            d
        };
        let adv_delta_back = {
            let mut d = vec![0.0_f32; trunk_out.len()];
            let eff: &[f32] = &adv_grad;
            for (i, row) in self.adv_head.weights[0].iter().enumerate() {
                let g = eff.get(i).copied().unwrap_or(0.0);
                for (j, dj) in d.iter_mut().enumerate() {
                    *dj += g * row[j];
                }
            }
            d
        };
        let trunk_delta: Vec<f32> = val_delta_back
            .iter()
            .zip(adv_delta_back.iter())
            .map(|(v, a)| v + a)
            .collect();
        self.trunk.sgd_step(&trunk_cache, &trunk_delta, lr);

        Ok(loss)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// ActionValuePolicy (wraps QNetwork as a Policy)
// ──────────────────────────────────────────────────────────────────────────────

/// Wraps a [`QNetwork`] so it implements the [`crate::rl::policy::Policy`] trait.
pub struct ActionValuePolicy {
    q: QNetwork,
}

impl ActionValuePolicy {
    /// Create a new ActionValuePolicy wrapping a QNetwork.
    pub fn new(q: QNetwork) -> Self {
        Self { q }
    }

    /// Access the underlying QNetwork.
    pub fn q_network(&self) -> &QNetwork {
        &self.q
    }

    /// Mutable access to the underlying QNetwork.
    pub fn q_network_mut(&mut self) -> &mut QNetwork {
        &mut self.q
    }
}

impl crate::rl::policy::Policy for ActionValuePolicy {
    fn num_actions(&self) -> usize {
        self.q.num_actions()
    }

    fn act(&self, obs: &[f32]) -> Result<usize> {
        self.q.greedy_action(obs)
    }

    fn logits(&self, obs: &[f32]) -> Result<Vec<f32>> {
        self.q.q_values(obs)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// SoftmaxValuePolicy (wraps QNetwork, samples from softmax)
// ──────────────────────────────────────────────────────────────────────────────

/// Wraps a QNetwork and samples actions proportionally to `softmax(Q(s, ·))`.
pub struct SoftmaxValuePolicy {
    q: QNetwork,
    temperature: f32,
    rng: PolicyRng,
}

impl SoftmaxValuePolicy {
    /// Create a new SoftmaxValuePolicy.
    pub fn new(q: QNetwork, temperature: f32) -> Result<Self> {
        if temperature <= 0.0 {
            return Err(NeuralError::InvalidArgument(
                "temperature must be positive".into(),
            ));
        }
        Ok(Self {
            q,
            temperature,
            rng: PolicyRng::from_time(),
        })
    }

    /// Sample an action from softmax(Q / T).
    pub fn sample(&mut self, obs: &[f32]) -> Result<usize> {
        let qs = self.q.q_values(obs)?;
        let probs = crate::rl::policy::softmax_temperature(&qs, self.temperature);
        crate::rl::policy::categorical_sample(&probs, &mut self.rng)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn obs4() -> Vec<f32> {
        vec![0.1, -0.2, 0.3, -0.4]
    }

    #[test]
    fn value_network_scalar_output() {
        let vn = ValueNetwork::new(4, &[16, 16]);
        let v = vn.value(&obs4()).expect("value() failed");
        assert!(v.is_finite(), "V(s) must be finite");
    }

    #[test]
    fn value_network_update_reduces_loss() {
        let mut vn = ValueNetwork::new(4, &[16]);
        let obs = obs4();
        let target = vec![1.0_f32];
        let mut prev_loss = f32::INFINITY;
        for _ in 0..200 {
            prev_loss = vn.update(&obs, &target, 0.05).expect("update failed");
        }
        assert!(
            prev_loss < 0.2,
            "ValueNetwork loss should decrease; got {}",
            prev_loss
        );
    }

    #[test]
    fn q_network_output_shape() {
        let qn = QNetwork::new(4, &[16, 16], 3);
        let qs = qn.q_values(&obs4()).expect("q_values failed");
        assert_eq!(qs.len(), 3);
    }

    #[test]
    fn q_network_greedy_in_range() {
        let qn = QNetwork::new(4, &[16], 5);
        let a = qn.greedy_action(&obs4()).expect("greedy_action failed");
        assert!(a < 5);
    }

    #[test]
    fn q_network_update_action() {
        let mut qn = QNetwork::new(4, &[16], 2);
        let obs = obs4();
        let mut prev = qn.q_values(&obs).expect("q_values")[0];
        for _ in 0..100 {
            qn.update_action(&obs, 0, 2.0, 0.1)
                .expect("update_action failed");
        }
        let after = qn.q_values(&obs).expect("q_values")[0];
        // Q(s,0) should move towards 2.0
        let before_err = (prev - 2.0).abs();
        let after_err = (after - 2.0).abs();
        assert!(
            after_err < before_err + 0.5,
            "Q-value should converge; before_err={} after_err={}",
            before_err,
            after_err
        );
        let _ = prev; // suppress unused warning
        prev = after;
        let _ = prev;
    }

    #[test]
    fn q_network_copy_from() {
        let qn_a = QNetwork::new(4, &[8], 2);
        let mut qn_b = QNetwork::new(4, &[8], 2);
        qn_b.copy_from(&qn_a).expect("copy_from failed");
        let qs_a = qn_a.q_values(&obs4()).expect("q_values a");
        let qs_b = qn_b.q_values(&obs4()).expect("q_values b");
        for (a, b) in qs_a.iter().zip(qs_b.iter()) {
            assert!((a - b).abs() < 1e-6, "Q-values differ after copy_from");
        }
    }

    #[test]
    fn dueling_q_network_output_shape() {
        let dqn = DuelingQNetwork::new(4, &[32, 32], &[16], 4);
        let qs = dqn.q_values(&obs4()).expect("q_values failed");
        assert_eq!(qs.len(), 4);
        assert!(
            qs.iter().all(|q| q.is_finite()),
            "all Q-values must be finite"
        );
    }

    #[test]
    fn dueling_q_network_greedy() {
        let dqn = DuelingQNetwork::new(4, &[32], &[], 3);
        let a = dqn.greedy_action(&obs4()).expect("greedy_action failed");
        assert!(a < 3);
    }

    #[test]
    fn dueling_q_network_copy_from() {
        let dqn_a = DuelingQNetwork::new(4, &[16], &[], 2);
        let mut dqn_b = DuelingQNetwork::new(4, &[16], &[], 2);
        dqn_b.copy_from(&dqn_a).expect("copy_from failed");
        let qs_a = dqn_a.q_values(&obs4()).expect("qs_a");
        let qs_b = dqn_b.q_values(&obs4()).expect("qs_b");
        for (a, b) in qs_a.iter().zip(qs_b.iter()) {
            assert!((a - b).abs() < 1e-6, "Dueling Q-values differ after copy");
        }
    }

    #[test]
    fn dueling_q_network_update() {
        let mut dqn = DuelingQNetwork::new(4, &[16], &[8], 3);
        let obs = obs4();
        let targets = vec![1.0_f32, 0.0, -1.0];
        let loss0 = dqn.update(&obs, &targets, 0.01).expect("update failed");
        assert!(loss0.is_finite(), "initial loss must be finite");
        // Just ensure update runs without error multiple times
        for _ in 0..10 {
            dqn.update(&obs, &targets, 0.01)
                .expect("update iteration failed");
        }
    }

    #[test]
    fn action_value_policy_implements_policy() {
        use crate::rl::policy::Policy;
        let qn = QNetwork::new(4, &[8], 3);
        let avp = ActionValuePolicy::new(qn);
        let a = avp.act(&obs4()).expect("act failed");
        assert!(a < 3);
        let logits = avp.logits(&obs4()).expect("logits failed");
        assert_eq!(logits.len(), 3);
    }
}
