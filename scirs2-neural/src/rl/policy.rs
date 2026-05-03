//! Policy representations for reinforcement learning.
//!
//! This module provides:
//! - [`Policy`] trait — selects a discrete action given an observation.
//! - [`SimpleNetwork`] — a compact, heap-allocated MLP (no autograd dependency).
//! - [`EpsilonGreedy`] — ε-greedy wrapper with linear epsilon decay.
//! - [`BoltzmannPolicy`] — softmax-temperature exploration policy.

use crate::error::{NeuralError, Result};

// ──────────────────────────────────────────────────────────────────────────────
// Minimal inline RNG (avoid heavy external crate)
// ──────────────────────────────────────────────────────────────────────────────

/// XorShift64 — fast, dependency-free pseudo-random number generator.
#[derive(Debug, Clone)]
pub struct PolicyRng {
    state: u64,
}

impl PolicyRng {
    /// Create a new RNG from a seed (must not be 0; uses fallback if 0).
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0xcafe_babe_dead_beef
            } else {
                seed
            },
        }
    }

    /// Time-based seed using subsecond nanoseconds.
    pub fn from_time() -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};
        let ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.subsec_nanos() as u64 ^ d.as_secs().wrapping_mul(6364136223846793005))
            .unwrap_or(0xcafe_babe_dead_beef);
        Self::new(ns)
    }

    /// Next raw u64.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    /// Uniform float in [0, 1).
    #[inline]
    pub fn uniform_f32(&mut self) -> f32 {
        (self.next_u64() >> 11) as f32 / (1u64 << 53) as f32
    }

    /// Uniform integer in `[0, n)`.
    #[inline]
    pub fn usize_below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Policy trait
// ──────────────────────────────────────────────────────────────────────────────

/// Trait for discrete-action policies.
///
/// Implementors wrap a neural network (or any other decision function) and
/// expose a single method that selects an action index for a given observation.
pub trait Policy: Send + Sync {
    /// Return the number of available actions.
    fn num_actions(&self) -> usize;

    /// Select an action for the given observation vector.
    ///
    /// The observation must have length equal to the network's input dimension.
    fn act(&self, obs: &[f32]) -> Result<usize>;

    /// Raw action-value logits (Q-values or log-probabilities) for the
    /// given observation.  Used internally by wrapper policies.
    fn logits(&self, obs: &[f32]) -> Result<Vec<f32>>;
}

// ──────────────────────────────────────────────────────────────────────────────
// SimpleNetwork — lightweight MLP, no autograd
// ──────────────────────────────────────────────────────────────────────────────

/// A fully-connected feed-forward network stored as `Vec<Vec<Vec<f32>>>`.
///
/// Layers are indexed `[layer][out_neuron][in_neuron]` for weights and
/// `[layer][out_neuron]` for biases.  Activation is ReLU for hidden layers
/// and linear (identity) for the output layer.
///
/// The network doubles as a Q-network (output = Q(s,a) for each action).
#[derive(Debug, Clone)]
pub struct SimpleNetwork {
    /// `weights[l]` has shape `[out_dim_l, in_dim_l]`.
    weights: Vec<Vec<Vec<f32>>>,
    /// `biases[l]` has length `out_dim_l`.
    biases: Vec<Vec<f32>>,
    /// Number of actions (= output dimension).
    num_actions: usize,
    /// Learning rate for manual SGD.
    lr: f32,
}

impl SimpleNetwork {
    /// Construct a new network.
    ///
    /// - `obs_dim`: input (observation) dimension.
    /// - `hidden_dims`: widths of hidden layers (e.g., `&[64, 64]`).
    /// - `num_actions`: output dimension (one logit per action).
    /// - `lr`: learning-rate used in [`Self::sgd_update`].
    ///
    /// Weights are initialised with He-uniform initialisation.
    pub fn new(obs_dim: usize, hidden_dims: &[usize], num_actions: usize, lr: f32) -> Self {
        assert!(obs_dim > 0, "obs_dim must be > 0");
        assert!(num_actions > 0, "num_actions must be > 0");
        assert!(lr > 0.0, "lr must be positive");

        let mut rng = PolicyRng::new(42);
        let mut dims = vec![obs_dim];
        dims.extend_from_slice(hidden_dims);
        dims.push(num_actions);

        let mut weights = Vec::with_capacity(dims.len() - 1);
        let mut biases = Vec::with_capacity(dims.len() - 1);

        for layer in 0..(dims.len() - 1) {
            let in_d = dims[layer];
            let out_d = dims[layer + 1];
            // He-uniform: scale = sqrt(6 / in_d)
            let scale = (6.0_f32 / in_d as f32).sqrt();
            let w: Vec<Vec<f32>> = (0..out_d)
                .map(|_| {
                    (0..in_d)
                        .map(|_| (rng.uniform_f32() * 2.0 - 1.0) * scale)
                        .collect()
                })
                .collect();
            let b: Vec<f32> = vec![0.0; out_d];
            weights.push(w);
            biases.push(b);
        }

        Self {
            weights,
            biases,
            num_actions,
            lr,
        }
    }

    /// Forward pass through the network.
    ///
    /// Returns the output activations (logits for each action).
    pub fn forward(&self, obs: &[f32]) -> Result<Vec<f32>> {
        if obs.len() != self.input_dim() {
            return Err(NeuralError::ShapeMismatch(format!(
                "SimpleNetwork: expected obs_dim={}, got {}",
                self.input_dim(),
                obs.len()
            )));
        }
        let mut activation: Vec<f32> = obs.to_vec();
        let n_layers = self.weights.len();

        for (layer_idx, (w, b)) in self.weights.iter().zip(self.biases.iter()).enumerate() {
            let mut next = Vec::with_capacity(w.len());
            for (out_i, (row, bias)) in w.iter().zip(b.iter()).enumerate() {
                let _ = out_i;
                let dot: f32 = row
                    .iter()
                    .zip(activation.iter())
                    .map(|(wi, xi)| wi * xi)
                    .sum();
                let pre = dot + bias;
                // ReLU on all but the last layer
                let post = if layer_idx < n_layers - 1 {
                    pre.max(0.0)
                } else {
                    pre
                };
                next.push(post);
            }
            activation = next;
        }
        Ok(activation)
    }

    /// Intermediate activations for every layer (including input; used for
    /// backpropagation in [`Self::sgd_update`]).
    fn forward_with_cache(&self, obs: &[f32]) -> Vec<Vec<f32>> {
        let mut cache = Vec::with_capacity(self.weights.len() + 1);
        cache.push(obs.to_vec());

        let n_layers = self.weights.len();
        for (layer_idx, (w, b)) in self.weights.iter().zip(self.biases.iter()).enumerate() {
            let prev = cache.last().expect("cache is never empty");
            let mut next = Vec::with_capacity(w.len());
            for (row, bias) in w.iter().zip(b.iter()) {
                let dot: f32 = row.iter().zip(prev.iter()).map(|(wi, xi)| wi * xi).sum();
                let pre = dot + bias;
                let post = if layer_idx < n_layers - 1 {
                    pre.max(0.0)
                } else {
                    pre
                };
                next.push(post);
            }
            cache.push(next);
        }
        cache
    }

    /// Single-sample stochastic gradient descent update.
    ///
    /// - `obs`: input observation.
    /// - `targets`: target values aligned with the output neurons.
    ///   Only the positions where `update_mask[i]` is `true` are updated;
    ///   pass `&vec![true; num_actions]` to update all outputs.
    /// - `update_mask`: which outputs to train on (e.g., only the taken action
    ///   for Q-learning).
    ///
    /// Uses MSE loss: `½ Σ (output_i - target_i)²`.
    pub fn sgd_update(
        &mut self,
        obs: &[f32],
        targets: &[f32],
        update_mask: &[bool],
    ) -> Result<f32> {
        if obs.len() != self.input_dim() {
            return Err(NeuralError::ShapeMismatch(format!(
                "obs_dim mismatch: expected {}, got {}",
                self.input_dim(),
                obs.len()
            )));
        }
        if targets.len() != self.num_actions {
            return Err(NeuralError::ShapeMismatch(format!(
                "targets len mismatch: expected {}, got {}",
                self.num_actions,
                targets.len()
            )));
        }
        if update_mask.len() != self.num_actions {
            return Err(NeuralError::ShapeMismatch(format!(
                "update_mask len mismatch: expected {}, got {}",
                self.num_actions,
                update_mask.len()
            )));
        }

        let cache = self.forward_with_cache(obs);
        let n_layers = self.weights.len();
        let output = cache.last().expect("cache is never empty");

        // Compute output-layer delta (MSE gradient masked by update_mask)
        let mut delta: Vec<f32> = output
            .iter()
            .zip(targets.iter())
            .zip(update_mask.iter())
            .map(|((o, t), &mask)| if mask { o - t } else { 0.0 })
            .collect();

        let loss: f32 = output
            .iter()
            .zip(targets.iter())
            .zip(update_mask.iter())
            .filter(|(_, &m)| m)
            .map(|((o, t), _)| (o - t).powi(2))
            .sum::<f32>()
            * 0.5;

        // Backprop through each layer (reversed)
        for layer_idx in (0..n_layers).rev() {
            let in_act = &cache[layer_idx];
            let out_act = &cache[layer_idx + 1];
            let out_d = self.weights[layer_idx].len();

            // Apply ReLU gradient to delta (all but output layer)
            let delta_with_relu: Vec<f32> = if layer_idx < n_layers - 1 {
                delta
                    .iter()
                    .zip(out_act.iter())
                    .map(|(d, a)| if *a > 0.0 { *d } else { 0.0 })
                    .collect()
            } else {
                delta.clone()
            };

            // Propagate delta to previous layer
            let in_d = in_act.len();
            let mut prev_delta = vec![0.0_f32; in_d];
            for (i, &dwr_i) in delta_with_relu.iter().enumerate().take(out_d) {
                for (j, pd_j) in prev_delta.iter_mut().enumerate().take(in_d) {
                    *pd_j += dwr_i * self.weights[layer_idx][i][j];
                }
            }

            // Update weights and biases
            for (i, &dwr_i) in delta_with_relu.iter().enumerate().take(out_d) {
                for (j, &ia_j) in in_act.iter().enumerate().take(in_d) {
                    self.weights[layer_idx][i][j] -= self.lr * dwr_i * ia_j;
                }
                self.biases[layer_idx][i] -= self.lr * dwr_i;
            }

            delta = prev_delta;
        }

        Ok(loss)
    }

    /// Copies weights from `other` into `self` (hard target-network update).
    pub fn copy_from(&mut self, other: &Self) -> Result<()> {
        if self.weights.len() != other.weights.len() {
            return Err(NeuralError::ShapeMismatch(
                "copy_from: networks have different architectures".to_string(),
            ));
        }
        for (l, (sw, ow)) in self
            .weights
            .iter_mut()
            .zip(other.weights.iter())
            .enumerate()
        {
            if sw.len() != ow.len() {
                return Err(NeuralError::ShapeMismatch(format!(
                    "copy_from: layer {} weight row count mismatch",
                    l
                )));
            }
            for (srow, orow) in sw.iter_mut().zip(ow.iter()) {
                srow.clone_from(orow);
            }
        }
        for (sb, ob) in self.biases.iter_mut().zip(other.biases.iter()) {
            sb.clone_from(ob);
        }
        Ok(())
    }

    /// Polyak-average the weights towards `other`: `self ← τ·other + (1−τ)·self`.
    pub fn polyak_update(&mut self, other: &Self, tau: f32) -> Result<()> {
        if self.weights.len() != other.weights.len() {
            return Err(NeuralError::ShapeMismatch(
                "polyak_update: architecture mismatch".to_string(),
            ));
        }
        for (sw, ow) in self.weights.iter_mut().zip(other.weights.iter()) {
            for (srow, orow) in sw.iter_mut().zip(ow.iter()) {
                for (sv, ov) in srow.iter_mut().zip(orow.iter()) {
                    *sv = tau * ov + (1.0 - tau) * (*sv);
                }
            }
        }
        for (sb, ob) in self.biases.iter_mut().zip(other.biases.iter()) {
            for (sv, ov) in sb.iter_mut().zip(ob.iter()) {
                *sv = tau * ov + (1.0 - tau) * (*sv);
            }
        }
        Ok(())
    }

    /// Input (observation) dimension.
    pub fn input_dim(&self) -> usize {
        self.weights
            .first()
            .map(|w| w.first().map(|row| row.len()).unwrap_or(0))
            .unwrap_or(0)
    }

    /// Output dimension.
    pub fn output_dim(&self) -> usize {
        self.weights.last().map(|w| w.len()).unwrap_or(0)
    }
}

impl Policy for SimpleNetwork {
    fn num_actions(&self) -> usize {
        self.num_actions
    }

    fn act(&self, obs: &[f32]) -> Result<usize> {
        let logits = self.logits(obs)?;
        // Greedy: argmax
        let best = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .ok_or_else(|| NeuralError::ComputationError("empty logits".to_string()))?;
        Ok(best)
    }

    fn logits(&self, obs: &[f32]) -> Result<Vec<f32>> {
        self.forward(obs)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// EpsilonGreedy
// ──────────────────────────────────────────────────────────────────────────────

/// ε-greedy exploration wrapper.
///
/// With probability ε the agent picks a **uniformly random** action; with
/// probability 1−ε it picks the **greedy** action from the inner policy.
/// ε decays linearly from `eps_start` to `eps_end` over `decay_steps` calls
/// to [`Self::act_train`].
pub struct EpsilonGreedy<P: Policy> {
    inner: P,
    eps_start: f32,
    eps_end: f32,
    decay_steps: usize,
    step: usize,
    rng: PolicyRng,
}

impl<P: Policy> EpsilonGreedy<P> {
    /// Wrap a policy in an ε-greedy shell.
    ///
    /// - `inner`: the base policy (typically [`SimpleNetwork`]).
    /// - `eps_start`: initial exploration probability (e.g., 1.0).
    /// - `eps_end`: final exploration probability (e.g., 0.05).
    /// - `decay_steps`: number of `act_train` calls over which ε is annealed.
    pub fn new(inner: P, eps_start: f32, eps_end: f32, decay_steps: usize) -> Self {
        assert!(eps_start >= eps_end, "eps_start must be >= eps_end");
        assert!(
            eps_end >= 0.0 && eps_start <= 1.0,
            "epsilon must be in [0, 1]"
        );
        Self {
            inner,
            eps_start,
            eps_end,
            decay_steps,
            step: 0,
            rng: PolicyRng::from_time(),
        }
    }

    /// Current ε value (linearly interpolated).
    pub fn epsilon(&self) -> f32 {
        let frac = (self.step as f32 / self.decay_steps.max(1) as f32).min(1.0);
        self.eps_start + frac * (self.eps_end - self.eps_start)
    }

    /// Select an action during **training** (increments the step counter and
    /// may explore randomly).
    pub fn act_train(&mut self, obs: &[f32]) -> Result<usize> {
        let eps = self.epsilon();
        self.step += 1;
        if self.rng.uniform_f32() < eps {
            Ok(self.rng.usize_below(self.inner.num_actions()))
        } else {
            self.inner.act(obs)
        }
    }

    /// Select an action during **evaluation** (greedy, no exploration).
    pub fn act_eval(&self, obs: &[f32]) -> Result<usize> {
        self.inner.act(obs)
    }

    /// Immutable borrow of the inner policy.
    pub fn inner(&self) -> &P {
        &self.inner
    }

    /// Mutable borrow of the inner policy (needed to update weights).
    pub fn inner_mut(&mut self) -> &mut P {
        &mut self.inner
    }

    /// Total number of training steps taken so far.
    pub fn steps(&self) -> usize {
        self.step
    }
}

impl<P: Policy> Policy for EpsilonGreedy<P> {
    fn num_actions(&self) -> usize {
        self.inner.num_actions()
    }

    /// In `act`, we use the greedy action (no exploration).
    /// To explore during training, call [`Self::act_train`] instead.
    fn act(&self, obs: &[f32]) -> Result<usize> {
        self.inner.act(obs)
    }

    fn logits(&self, obs: &[f32]) -> Result<Vec<f32>> {
        self.inner.logits(obs)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// BoltzmannPolicy (softmax temperature exploration)
// ──────────────────────────────────────────────────────────────────────────────

/// Boltzmann (softmax temperature) exploration policy.
///
/// Actions are sampled from the probability distribution:
/// `π(a|s) = exp(Q(s,a)/T) / Σ_b exp(Q(s,b)/T)`
///
/// - Higher `temperature` → more uniform (more exploration).
/// - `temperature → 0` → approaches greedy.
pub struct BoltzmannPolicy<P: Policy> {
    inner: P,
    temperature: f32,
    rng: PolicyRng,
}

impl<P: Policy> BoltzmannPolicy<P> {
    /// Wrap a policy with Boltzmann (softmax) exploration.
    ///
    /// `temperature` must be positive.
    pub fn new(inner: P, temperature: f32) -> Result<Self> {
        if temperature <= 0.0 {
            return Err(NeuralError::InvalidArgument(
                "BoltzmannPolicy: temperature must be positive".to_string(),
            ));
        }
        Ok(Self {
            inner,
            temperature,
            rng: PolicyRng::from_time(),
        })
    }

    /// Current temperature.
    pub fn temperature(&self) -> f32 {
        self.temperature
    }

    /// Adjust the temperature (e.g., for annealing schedules).
    pub fn set_temperature(&mut self, t: f32) -> Result<()> {
        if t <= 0.0 {
            return Err(NeuralError::InvalidArgument(
                "temperature must be positive".to_string(),
            ));
        }
        self.temperature = t;
        Ok(())
    }

    /// Sample an action from the Boltzmann distribution.
    pub fn sample_action(&mut self, obs: &[f32]) -> Result<usize> {
        let logits = self.inner.logits(obs)?;
        let probs = softmax_temperature(&logits, self.temperature);
        categorical_sample(&probs, &mut self.rng)
    }

    /// Immutable borrow of the inner policy.
    pub fn inner(&self) -> &P {
        &self.inner
    }

    /// Mutable borrow of the inner policy.
    pub fn inner_mut(&mut self) -> &mut P {
        &mut self.inner
    }
}

impl<P: Policy> Policy for BoltzmannPolicy<P> {
    fn num_actions(&self) -> usize {
        self.inner.num_actions()
    }

    /// Uses Boltzmann sampling during both training and evaluation.
    fn act(&self, _obs: &[f32]) -> Result<usize> {
        // Note: act() takes &self so cannot mutate rng; use sample_action(&mut self, ...) instead.
        Err(NeuralError::InvalidState(
            "BoltzmannPolicy::act requires mutable self; use sample_action() instead".to_string(),
        ))
    }

    fn logits(&self, obs: &[f32]) -> Result<Vec<f32>> {
        self.inner.logits(obs)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Helper functions
// ──────────────────────────────────────────────────────────────────────────────

/// Numerically stable softmax with temperature scaling.
pub fn softmax_temperature(logits: &[f32], temperature: f32) -> Vec<f32> {
    let scaled: Vec<f32> = logits.iter().map(|x| x / temperature).collect();
    let max = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scaled.iter().map(|x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum == 0.0 {
        vec![1.0 / logits.len() as f32; logits.len()]
    } else {
        exps.iter().map(|x| x / sum).collect()
    }
}

/// Standard softmax (temperature = 1).
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    softmax_temperature(logits, 1.0)
}

/// Sample a categorical index from `probs` (must sum to 1).
pub fn categorical_sample(probs: &[f32], rng: &mut PolicyRng) -> Result<usize> {
    if probs.is_empty() {
        return Err(NeuralError::InvalidArgument(
            "empty probability vector".to_string(),
        ));
    }
    let u = rng.uniform_f32();
    let mut cumsum = 0.0_f32;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if u <= cumsum {
            return Ok(i);
        }
    }
    // Fallback to last index due to floating-point rounding
    Ok(probs.len() - 1)
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn obs4() -> Vec<f32> {
        vec![0.1, -0.3, 0.5, 0.0]
    }

    #[test]
    fn simple_network_forward_shape() {
        let net = SimpleNetwork::new(4, &[16, 16], 2, 1e-3);
        let out = net.forward(&obs4()).expect("forward pass failed");
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn simple_network_greedy_act() {
        let net = SimpleNetwork::new(4, &[16, 16], 2, 1e-3);
        let a = net.act(&obs4()).expect("act failed");
        assert!(a < 2);
    }

    #[test]
    fn sgd_update_reduces_loss() {
        let mut net = SimpleNetwork::new(4, &[16], 2, 0.1);
        let obs = obs4();
        let targets = vec![1.0_f32, 0.0];
        let mask = vec![true, true];

        let mut prev_loss = f32::INFINITY;
        for _ in 0..100 {
            let loss = net.sgd_update(&obs, &targets, &mask).expect("sgd failed");
            prev_loss = loss;
        }
        assert!(prev_loss < 0.5, "loss should decrease; got {}", prev_loss);
    }

    #[test]
    fn copy_from_makes_identical_outputs() {
        let net_a = SimpleNetwork::new(4, &[8], 3, 1e-3);
        let mut net_b = SimpleNetwork::new(4, &[8], 3, 1e-3);
        net_b.copy_from(&net_a).expect("copy_from failed");
        let obs = obs4();
        let out_a = net_a.forward(&obs).expect("forward a");
        let out_b = net_b.forward(&obs).expect("forward b");
        for (a, b) in out_a.iter().zip(out_b.iter()) {
            assert!((a - b).abs() < 1e-6, "outputs differ after copy_from");
        }
    }

    #[test]
    fn epsilon_greedy_decays() {
        let net = SimpleNetwork::new(4, &[8], 2, 1e-3);
        let mut eg = EpsilonGreedy::new(net, 1.0, 0.1, 100);
        assert!((eg.epsilon() - 1.0).abs() < 1e-5);
        for _ in 0..100 {
            let _ = eg.act_train(&obs4()).expect("act_train failed");
        }
        assert!(
            (eg.epsilon() - 0.1).abs() < 1e-4,
            "epsilon should reach eps_end"
        );
    }

    #[test]
    fn epsilon_greedy_eval_is_greedy() {
        let net = SimpleNetwork::new(4, &[8], 2, 1e-3);
        let eg = EpsilonGreedy::new(net, 1.0, 0.0, 10);
        // eval always uses inner greedy policy
        let a = eg.act_eval(&obs4()).expect("eval failed");
        assert!(a < 2);
    }

    #[test]
    fn softmax_sums_to_one() {
        let logits = vec![1.0_f32, 2.0, 3.0, 0.5];
        let probs = softmax(&logits);
        let sum: f32 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "softmax probs should sum to 1, got {}",
            sum
        );
    }

    #[test]
    fn boltzmann_sample_valid_action() {
        let net = SimpleNetwork::new(4, &[8], 3, 1e-3);
        let mut bp = BoltzmannPolicy::new(net, 1.0).expect("construction failed");
        let a = bp.sample_action(&obs4()).expect("sample_action failed");
        assert!(a < 3);
    }

    #[test]
    fn polyak_update_blends_weights() {
        let net_a = SimpleNetwork::new(4, &[8], 2, 1e-3);
        let mut net_b = SimpleNetwork::new(4, &[8], 2, 1e-3);
        net_b
            .polyak_update(&net_a, 1.0)
            .expect("polyak update failed");
        // After τ=1 update, net_b should equal net_a
        let obs = obs4();
        let out_a = net_a.forward(&obs).expect("forward a");
        let out_b = net_b.forward(&obs).expect("forward b");
        for (a, b) in out_a.iter().zip(out_b.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "polyak τ=1 should make nets identical"
            );
        }
    }
}
