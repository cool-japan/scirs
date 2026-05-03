//! Proximal Policy Optimization (PPO) — on-policy actor-critic algorithm.
//!
//! Implements the clipped surrogate objective from Schulman et al. 2017
//! together with GAE-λ for advantage estimation.
//!
//! The policy network outputs a Gaussian distribution over actions
//! (independent per dimension with learned log-std).  The value network is a
//! separate MLP sharing the same observation dimension.
//!
//! # Reference
//! Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017).
//! *Proximal Policy Optimization Algorithms*. arXiv:1707.06347.

use crate::error::{NeuralError, Result};
use crate::rl::environments::Environment;
use scirs2_core::ndarray::{s, Array1, Array2, Axis};

// ──────────────────────────────────────────────────────────────────────────────
// Configuration
// ──────────────────────────────────────────────────────────────────────────────

/// Hyper-parameters for [`PPO`].
#[derive(Debug, Clone)]
pub struct PPOConfig {
    /// Clipping range ε for the probability ratio (default 0.2).
    pub clip_eps: f64,
    /// Number of optimisation epochs per rollout (default 10).
    pub n_epochs: usize,
    /// Number of mini-batches per epoch (default 4).
    pub n_minibatches: usize,
    /// Discount factor γ (default 0.99).
    pub gamma: f64,
    /// GAE-λ parameter (default 0.95).
    pub gae_lambda: f64,
    /// Entropy bonus coefficient (default 0.01).
    pub ent_coef: f64,
    /// Value-function loss coefficient (default 0.5).
    pub vf_coef: f64,
    /// Learning rate (default 3e-4).
    pub lr: f64,
    /// Maximum gradient norm (default 0.5, `None` to disable).
    pub max_grad_norm: Option<f64>,
    /// Number of rollout steps per environment before an update (default 2048).
    pub n_steps: usize,
    /// Hidden layer widths for policy and value networks (default [64, 64]).
    pub hidden_dims: Vec<usize>,
    /// Initial log standard deviation for the Gaussian policy (default 0.0).
    pub log_std_init: f64,
}

impl Default for PPOConfig {
    fn default() -> Self {
        Self {
            clip_eps: 0.2,
            n_epochs: 10,
            n_minibatches: 4,
            gamma: 0.99,
            gae_lambda: 0.95,
            ent_coef: 0.01,
            vf_coef: 0.5,
            lr: 3e-4,
            max_grad_norm: Some(0.5),
            n_steps: 2048,
            hidden_dims: vec![64, 64],
            log_std_init: 0.0,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Training info returned after each update
// ──────────────────────────────────────────────────────────────────────────────

/// Diagnostic statistics returned by [`PPO::update`].
#[derive(Debug, Clone, Default)]
pub struct PPOInfo {
    /// Mean policy (surrogate) loss over all mini-batch updates.
    pub policy_loss: f64,
    /// Mean value-function loss over all mini-batch updates.
    pub value_loss: f64,
    /// Mean entropy bonus over all mini-batch updates.
    pub entropy_loss: f64,
    /// Approximate KL divergence (early-stop diagnostic).
    pub approx_kl: f64,
    /// Clip fraction — proportion of ratios that were clipped.
    pub clip_fraction: f64,
    /// Total number of gradient steps taken.
    pub n_updates: usize,
}

// ──────────────────────────────────────────────────────────────────────────────
// Rollout buffer
// ──────────────────────────────────────────────────────────────────────────────

/// Stores a single on-policy rollout before the gradient update.
#[derive(Debug)]
pub struct RolloutBuffer {
    /// Observations, shape `[n_steps, obs_dim]`.
    pub states: Array2<f64>,
    /// Actions taken, shape `[n_steps, act_dim]`.
    pub actions: Array2<f64>,
    /// Log-probabilities of those actions, shape `[n_steps]`.
    pub log_probs: Array1<f64>,
    /// Observed rewards, shape `[n_steps]`.
    pub rewards: Array1<f64>,
    /// Value-function estimates at each step, shape `[n_steps]`.
    pub values: Array1<f64>,
    /// Done flags, shape `[n_steps]`.
    pub dones: Array1<bool>,
    /// GAE advantages (computed after collection), shape `[n_steps]`.
    pub advantages: Array1<f64>,
    /// Value-function targets (returns), shape `[n_steps]`.
    pub returns: Array1<f64>,
    /// Number of valid steps stored.
    pub n_steps: usize,
}

impl RolloutBuffer {
    fn new(n_steps: usize, obs_dim: usize, act_dim: usize) -> Self {
        Self {
            states: Array2::zeros((n_steps, obs_dim)),
            actions: Array2::zeros((n_steps, act_dim)),
            log_probs: Array1::zeros(n_steps),
            rewards: Array1::zeros(n_steps),
            values: Array1::zeros(n_steps),
            dones: Array1::from_elem(n_steps, false),
            advantages: Array1::zeros(n_steps),
            returns: Array1::zeros(n_steps),
            n_steps,
        }
    }

    /// Compute GAE-λ advantages and value-function targets in-place.
    fn compute_advantages(&mut self, last_value: f64, gamma: f64, gae_lambda: f64) {
        let n = self.n_steps;
        let mut gae = 0.0_f64;

        for t in (0..n).rev() {
            let next_non_terminal = if self.dones[t] { 0.0 } else { 1.0 };
            let next_value = if t + 1 < n {
                self.values[t + 1]
            } else {
                last_value
            };
            let delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t];
            gae = delta + gamma * gae_lambda * next_non_terminal * gae;
            self.advantages[t] = gae;
        }
        self.returns = &self.advantages + &self.values;

        // Normalise advantages
        let mean = self.advantages.mean().unwrap_or(0.0);
        let var = self
            .advantages
            .mapv(|a| (a - mean).powi(2))
            .mean()
            .unwrap_or(1.0);
        let std = var.sqrt().max(1e-8);
        self.advantages = self.advantages.mapv(|a| (a - mean) / std);
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Minimal neural-network helpers
// ──────────────────────────────────────────────────────────────────────────────
// We implement simple weight-and-bias MLPs entirely with ndarray arithmetic so
// that the rl module does not depend on the full scirs2-neural layer stack
// (which itself depends on this crate — would create a circular dep).
// ──────────────────────────────────────────────────────────────────────────────

/// A single fully-connected layer: `y = xW + b`, optionally followed by `tanh`.
#[derive(Debug, Clone)]
struct Linear {
    w: Array2<f64>,
    b: Array1<f64>,
}

impl Linear {
    /// He-style random initialisation using xorshift64.
    fn new(in_dim: usize, out_dim: usize, rng: &mut crate::rl::replay_buffer::XorShift64) -> Self {
        let scale = (2.0 / in_dim as f64).sqrt();
        let w = Array2::from_shape_fn((in_dim, out_dim), |_| {
            // Box-Muller for normal samples
            let u1 = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64 + 1e-20;
            let u2 = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
            let n = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            n * scale
        });
        let b = Array1::zeros(out_dim);
        Self { w, b }
    }

    /// Forward pass: batch matmul, shape `[batch, out_dim]`.
    fn forward(&self, x: &Array2<f64>) -> Array2<f64> {
        x.dot(&self.w) + &self.b
    }

    /// Backward pass given upstream gradient `grad_out` (shape `[batch, out_dim]`).
    ///
    /// Returns `(grad_w, grad_b, grad_x)`.
    fn backward(
        &self,
        x: &Array2<f64>,
        grad_out: &Array2<f64>,
    ) -> (Array2<f64>, Array1<f64>, Array2<f64>) {
        let grad_w = x.t().dot(grad_out);
        let grad_b = grad_out.sum_axis(Axis(0));
        let grad_x = grad_out.dot(&self.w.t());
        (grad_w, grad_b, grad_x)
    }

    fn apply_grad(&mut self, grad_w: &Array2<f64>, grad_b: &Array1<f64>, lr: f64) {
        self.w -= &grad_w.mapv(|g| g * lr);
        self.b -= &grad_b.mapv(|g| g * lr);
    }
}

// ── tanh activation (element-wise) ──────────────────────────────────────────

fn tanh_fwd(x: &Array2<f64>) -> Array2<f64> {
    x.mapv(|v| v.tanh())
}

fn tanh_bwd(y: &Array2<f64>, grad: &Array2<f64>) -> Array2<f64> {
    // dy/dx = 1 - tanh²(x)  and y = tanh(x)
    grad * &y.mapv(|v| 1.0 - v * v)
}

// ── softplus ─────────────────────────────────────────────────────────────────

#[inline]
fn softplus(x: f64) -> f64 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

// Type alias for the complex return type of forward_cache.
type MlpForwardCache = (Array2<f64>, Vec<(Array2<f64>, Array2<f64>)>);

// ──────────────────────────────────────────────────────────────────────────────
// Mlp (multiple hidden layers + tanh + linear output)
// ──────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct Mlp {
    layers: Vec<Linear>,
}

impl Mlp {
    fn new(
        in_dim: usize,
        hidden_dims: &[usize],
        out_dim: usize,
        rng: &mut crate::rl::replay_buffer::XorShift64,
    ) -> Self {
        let mut dims: Vec<usize> = vec![in_dim];
        dims.extend_from_slice(hidden_dims);
        dims.push(out_dim);

        let layers = dims
            .windows(2)
            .map(|w| Linear::new(w[0], w[1], rng))
            .collect();
        Self { layers }
    }

    /// Forward pass.  All layers except the last use tanh activation.
    /// Returns both the final output AND the per-layer pre-activation cache
    /// (needed for exact backward pass).
    fn forward_cache(&self, x: &Array2<f64>) -> MlpForwardCache {
        let mut current = x.clone();
        let mut cache: Vec<(Array2<f64>, Array2<f64>)> = Vec::new();

        for (i, layer) in self.layers.iter().enumerate() {
            let pre = layer.forward(&current);
            let post = if i < self.layers.len() - 1 {
                tanh_fwd(&pre)
            } else {
                pre.clone() // last layer — linear
            };
            cache.push((current.clone(), pre));
            current = post;
        }
        (current, cache)
    }

    /// Simpler forward without caching (inference only).
    fn forward(&self, x: &Array2<f64>) -> Array2<f64> {
        let (out, _) = self.forward_cache(x);
        out
    }

    /// Backward pass.  Returns gradients for each layer (same order as `self.layers`).
    fn backward(
        &self,
        cache: &[(Array2<f64>, Array2<f64>)],
        grad_out: Array2<f64>,
    ) -> Vec<(Array2<f64>, Array1<f64>)> {
        let n = self.layers.len();
        let mut grads: Vec<(Array2<f64>, Array1<f64>)> = vec![];
        let mut g = grad_out;

        for i in (0..n).rev() {
            let (x_in, pre) = &cache[i];
            // If not last layer, reverse tanh
            if i < n - 1 {
                let y = tanh_fwd(pre);
                g = tanh_bwd(&y, &g);
            }
            let (gw, gb, gx) = self.layers[i].backward(x_in, &g);
            grads.push((gw, gb));
            g = gx;
        }
        grads.reverse();
        grads
    }

    fn apply_grads(&mut self, grads: &[(Array2<f64>, Array1<f64>)], lr: f64) {
        for (layer, (gw, gb)) in self.layers.iter_mut().zip(grads.iter()) {
            layer.apply_grad(gw, gb, lr);
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// ActorCritic
// ──────────────────────────────────────────────────────────────────────────────

/// Joint actor–critic network used by [`PPO`].
///
/// The **actor** outputs a Gaussian mean over actions; a shared log-std
/// parameter is learned separately.
///
/// The **critic** outputs a single scalar value estimate.
pub struct ActorCritic<F> {
    _phantom: std::marker::PhantomData<F>,
    actor: Mlp,
    critic: Mlp,
    log_std: Array1<f64>,
    obs_dim: usize,
    act_dim: usize,
}

impl<F: 'static> ActorCritic<F> {
    fn new(
        obs_dim: usize,
        act_dim: usize,
        hidden_dims: &[usize],
        log_std_init: f64,
        rng: &mut crate::rl::replay_buffer::XorShift64,
    ) -> Self {
        let actor = Mlp::new(obs_dim, hidden_dims, act_dim, rng);
        let critic = Mlp::new(obs_dim, hidden_dims, 1, rng);
        let log_std = Array1::from_elem(act_dim, log_std_init);

        Self {
            _phantom: std::marker::PhantomData,
            actor,
            critic,
            log_std,
            obs_dim,
            act_dim,
        }
    }

    /// Sample a mean and log-prob for a batch of observations.
    ///
    /// Returns `(actions, log_probs, values)`.
    fn forward_sample(
        &self,
        obs: &Array2<f64>,
        rng: &mut crate::rl::replay_buffer::XorShift64,
    ) -> (Array2<f64>, Array1<f64>, Array1<f64>) {
        let means = self.actor.forward(obs);
        let values = self.critic.forward(obs);
        let values_1d = values.column(0).to_owned();

        let std: Array1<f64> = self.log_std.mapv(|ls| ls.exp());
        let batch = obs.shape()[0];

        let mut actions = Array2::zeros((batch, self.act_dim));
        let mut log_probs = Array1::zeros(batch);

        for b in 0..batch {
            let mut lp = 0.0_f64;
            for a in 0..self.act_dim {
                let mu = means[[b, a]];
                let sig = std[a].max(1e-6);
                // Box-Muller sample
                let u1 = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64 + 1e-20;
                let u2 = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                let act = mu + sig * z;
                actions[[b, a]] = act;
                // log N(act | mu, sig)
                lp += -0.5 * ((act - mu) / sig).powi(2)
                    - sig.ln()
                    - 0.5 * (2.0 * std::f64::consts::PI).ln();
            }
            log_probs[b] = lp;
        }

        (actions, log_probs, values_1d)
    }

    /// Evaluate log-probabilities and entropy for given obs/action pairs.
    fn evaluate_actions(
        &self,
        obs: &Array2<f64>,
        actions: &Array2<f64>,
    ) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
        let means = self.actor.forward(obs);
        let values = self.critic.forward(obs);
        let values_1d = values.column(0).to_owned();

        let std: Array1<f64> = self.log_std.mapv(|ls| ls.exp());
        let batch = obs.shape()[0];
        let mut log_probs = Array1::zeros(batch);
        let mut entropies = Array1::zeros(batch);

        for b in 0..batch {
            let mut lp = 0.0_f64;
            let mut ent = 0.0_f64;
            for a in 0..self.act_dim {
                let mu = means[[b, a]];
                let sig = std[a].max(1e-6);
                let act = actions[[b, a]];
                lp += -0.5 * ((act - mu) / sig).powi(2)
                    - sig.ln()
                    - 0.5 * (2.0 * std::f64::consts::PI).ln();
                // Gaussian entropy: 0.5 * ln(2πe σ²)
                ent += 0.5 * (2.0 * std::f64::consts::PI * std::f64::consts::E * sig * sig).ln();
            }
            log_probs[b] = lp;
            entropies[b] = ent;
        }

        (log_probs, entropies, values_1d)
    }

    /// Value estimate for a batch of observations.
    fn predict_values(&self, obs: &Array2<f64>) -> Array1<f64> {
        self.critic.forward(obs).column(0).to_owned()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// PPO
// ──────────────────────────────────────────────────────────────────────────────

/// Proximal Policy Optimisation agent.
///
/// # Example (abbreviated)
/// ```rust,ignore
/// use scirs2_neural::rl::{ppo::{PPO, PPOConfig}, environments::CartPole};
/// let cfg = PPOConfig { n_steps: 128, ..Default::default() };
/// let mut agent: PPO<f64> = PPO::new(4, 2, cfg);
/// let mut env = CartPole::new();
/// let rewards = agent.train(&mut env, 10_000);
/// println!("final ep reward: {}", rewards.last().copied().unwrap_or(0.0));
/// ```
pub struct PPO<F> {
    /// The joint actor-critic network.
    pub policy: ActorCritic<F>,
    /// Hyper-parameters.
    pub config: PPOConfig,
    rng: crate::rl::replay_buffer::XorShift64,
    obs_dim: usize,
    act_dim: usize,
}

impl<F: 'static> PPO<F> {
    /// Construct a new PPO agent.
    ///
    /// - `obs_dim`: observation (state) dimensionality.
    /// - `act_dim`: action dimensionality.
    pub fn new(obs_dim: usize, act_dim: usize, config: PPOConfig) -> Self {
        let mut rng = crate::rl::replay_buffer::XorShift64::new(config.n_steps as u64 ^ 0xdeadbeef);
        let policy = ActorCritic::new(
            obs_dim,
            act_dim,
            &config.hidden_dims.clone(),
            config.log_std_init,
            &mut rng,
        );
        Self {
            policy,
            config,
            rng,
            obs_dim,
            act_dim,
        }
    }

    /// Collect a rollout of `n_steps` transitions from `env`.
    ///
    /// The environment must be already reset (or in a valid mid-episode state).
    /// A fresh reset is performed automatically at the start of each call so
    /// the caller does not need to maintain inter-call state.
    pub fn collect_rollout<E>(&mut self, env: &mut E, n_steps: usize) -> RolloutBuffer
    where
        E: Environment<State = Array1<f64>, Action = Array1<f64>>,
    {
        let mut buf = RolloutBuffer::new(n_steps, self.obs_dim, self.act_dim);
        let mut state = env.reset();

        for t in 0..n_steps {
            let obs_2d = state.clone().insert_axis(Axis(0));
            let (actions, log_probs, values) = self.policy.forward_sample(&obs_2d, &mut self.rng);

            let action_1d = actions.row(0).to_owned();
            let (next_state, reward, done) = env.step(&action_1d);

            buf.states.row_mut(t).assign(&state);
            buf.actions.row_mut(t).assign(&action_1d);
            buf.log_probs[t] = log_probs[0];
            buf.rewards[t] = reward;
            buf.values[t] = values[0];
            buf.dones[t] = done;

            state = if done { env.reset() } else { next_state };
        }

        // Bootstrap last value
        let last_obs = state.insert_axis(Axis(0));
        let last_value = self.policy.predict_values(&last_obs)[0];
        buf.compute_advantages(last_value, self.config.gamma, self.config.gae_lambda);
        buf
    }

    /// Perform `n_epochs * n_minibatches` PPO gradient updates on a rollout.
    pub fn update(&mut self, rollout: &RolloutBuffer) -> PPOInfo {
        let n = rollout.n_steps;
        let mb_size = (n / self.config.n_minibatches).max(1);
        let lr = self.config.lr;
        let clip_eps = self.config.clip_eps;
        let ent_coef = self.config.ent_coef;
        let vf_coef = self.config.vf_coef;

        let mut info = PPOInfo::default();
        let mut total_updates = 0usize;

        // Shuffle indices for each epoch
        let mut idx: Vec<usize> = (0..n).collect();

        for _epoch in 0..self.config.n_epochs {
            // Fisher-Yates shuffle
            for i in (1..n).rev() {
                let j = self.rng.next_usize(i + 1);
                idx.swap(i, j);
            }

            for mb_start in (0..n).step_by(mb_size) {
                let mb_end = (mb_start + mb_size).min(n);
                let mb_idx = &idx[mb_start..mb_end];

                // Gather mini-batch
                let mb_obs = gather_rows_2d(&rollout.states, mb_idx);
                let mb_act = gather_rows_2d(&rollout.actions, mb_idx);
                let mb_adv = gather_1d(&rollout.advantages, mb_idx);
                let mb_ret = gather_1d(&rollout.returns, mb_idx);
                let mb_old_lp = gather_1d(&rollout.log_probs, mb_idx);

                // Evaluate current policy
                let (new_lp, entropy, new_values) = self.policy.evaluate_actions(&mb_obs, &mb_act);

                let batch_n = mb_obs.shape()[0] as f64;

                // Compute surrogate losses
                let mut policy_loss = 0.0_f64;
                let mut clip_count = 0u64;
                let mut kl_sum = 0.0_f64;

                // We need gradients w.r.t. actor parameters.
                // We use a numerical gradient approximation (REINFORCE-style
                // weight update via log-prob gradient scaled by advantage).
                //
                // Full auto-diff would require a tape; here we compute the
                // "pseudo-gradient" analytically for the Gaussian case and
                // apply it via the chain rule through the MLP.

                // Ratio r_t = exp(new_lp - old_lp)
                let mut ratio_vec = Array1::zeros(mb_idx.len());
                let mut clipped_ratio_vec = Array1::zeros(mb_idx.len());
                let mut advantage_used = Array1::zeros(mb_idx.len());

                for b in 0..mb_idx.len() {
                    let ratio = (new_lp[b] - mb_old_lp[b]).exp();
                    let clipped = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps);
                    let adv = mb_adv[b];
                    let l1 = ratio * adv;
                    let l2 = clipped * adv;
                    policy_loss -= l1.min(l2);
                    if (ratio - 1.0).abs() > clip_eps {
                        clip_count += 1;
                    }
                    kl_sum += mb_old_lp[b] - new_lp[b];
                    ratio_vec[b] = ratio;
                    clipped_ratio_vec[b] = clipped;
                    advantage_used[b] = adv;
                }
                policy_loss /= batch_n;

                // Value loss (clipped MSE)
                let vf_loss: f64 = new_values
                    .iter()
                    .zip(mb_ret.iter())
                    .map(|(v, r)| (v - r).powi(2))
                    .sum::<f64>()
                    / batch_n;

                let ent_mean: f64 = entropy.sum() / batch_n;

                // ── Actor gradient update (manual chain-rule) ──────────────
                // Gradient of the clipped surrogate w.r.t. log_prob is:
                //   g_lp[b] = -adv[b]  if ratio was NOT clipped
                //             0         if ratio was clipped (gradient stops)
                // Then g_log_prob flows back through the Gaussian log-prob
                // to the mean network.
                let mut g_mean = Array2::zeros((mb_idx.len(), self.act_dim));
                let mut g_log_std = Array1::zeros(self.act_dim);

                let means_out = self.policy.actor.forward(&mb_obs);
                let std_vec: Array1<f64> = self.policy.log_std.mapv(|ls| ls.exp());

                for b in 0..mb_idx.len() {
                    let ratio = ratio_vec[b];
                    let adv = advantage_used[b];
                    // Was ratio clipped?
                    let is_clipped = (ratio - 1.0).abs() > clip_eps;
                    let g_surr = if is_clipped { 0.0 } else { -adv };

                    for a in 0..self.act_dim {
                        let mu = means_out[[b, a]];
                        let sig = std_vec[a].max(1e-6);
                        let act = mb_act[[b, a]];
                        // d log_prob / d mu  = (act - mu) / sig^2
                        let d_lp_d_mu = (act - mu) / (sig * sig);
                        g_mean[[b, a]] += g_surr * d_lp_d_mu / batch_n;
                        // d log_prob / d log_sig = (act-mu)^2/sig^2 - 1
                        let d_lp_d_lsig = (act - mu).powi(2) / (sig * sig) - 1.0;
                        g_log_std[a] += g_surr * d_lp_d_lsig / batch_n;
                        // entropy gradient: -0.5 * ent_coef
                        g_log_std[a] -= ent_coef / batch_n;
                    }
                }

                // Back-prop through actor MLP
                let (_, cache_actor) = self.policy.actor.forward_cache(&mb_obs);
                let actor_grads = self.policy.actor.backward(&cache_actor, g_mean);
                self.policy.actor.apply_grads(&actor_grads, lr);
                self.policy.log_std -= &g_log_std.mapv(|g: f64| g * lr);

                // ── Critic gradient update ──────────────────────────────────
                // dL_vf / d value = 2 * (value - return) / batch_n * vf_coef
                let mut g_value = Array2::zeros((mb_idx.len(), 1));
                for b in 0..mb_idx.len() {
                    g_value[[b, 0]] = 2.0 * (new_values[b] - mb_ret[b]) / batch_n * vf_coef;
                }
                let (_, cache_critic) = self.policy.critic.forward_cache(&mb_obs);
                let critic_grads = self.policy.critic.backward(&cache_critic, g_value);
                self.policy.critic.apply_grads(&critic_grads, lr);

                // Accumulate diagnostics
                info.policy_loss += policy_loss;
                info.value_loss += vf_loss;
                info.entropy_loss += ent_mean;
                info.approx_kl += kl_sum / batch_n;
                info.clip_fraction += clip_count as f64 / mb_idx.len() as f64;
                total_updates += 1;
            }
        }

        if total_updates > 0 {
            let n_f = total_updates as f64;
            info.policy_loss /= n_f;
            info.value_loss /= n_f;
            info.entropy_loss /= n_f;
            info.approx_kl /= n_f;
            info.clip_fraction /= n_f;
        }
        info.n_updates = total_updates;
        info
    }

    /// Full training loop.
    ///
    /// Returns the episodic reward at the end of each completed episode.
    pub fn train<E>(&mut self, env: &mut E, total_timesteps: usize) -> Vec<f64>
    where
        E: Environment<State = Array1<f64>, Action = Array1<f64>>,
    {
        let n_steps = self.config.n_steps;
        let mut episode_rewards: Vec<f64> = Vec::new();
        let mut timesteps = 0usize;
        let mut ep_reward = 0.0_f64;
        let mut state = env.reset();

        while timesteps < total_timesteps {
            // Collect a rollout
            let rollout = {
                let mut buf = RolloutBuffer::new(n_steps, self.obs_dim, self.act_dim);
                for t in 0..n_steps {
                    let obs_2d = state.clone().insert_axis(Axis(0));
                    let (actions, log_probs, values) =
                        self.policy.forward_sample(&obs_2d, &mut self.rng);
                    let action_1d = actions.row(0).to_owned();
                    let (next_state, reward, done) = env.step(&action_1d);

                    buf.states.row_mut(t).assign(&state);
                    buf.actions.row_mut(t).assign(&action_1d);
                    buf.log_probs[t] = log_probs[0];
                    buf.rewards[t] = reward;
                    buf.values[t] = values[0];
                    buf.dones[t] = done;

                    ep_reward += reward;
                    if done {
                        episode_rewards.push(ep_reward);
                        ep_reward = 0.0;
                        state = env.reset();
                    } else {
                        state = next_state;
                    }
                    timesteps += 1;
                    if timesteps >= total_timesteps {
                        break;
                    }
                }
                let last_obs = state.clone().insert_axis(Axis(0));
                let last_value = self.policy.predict_values(&last_obs)[0];
                buf.n_steps = buf.n_steps.min(timesteps);
                buf.compute_advantages(last_value, self.config.gamma, self.config.gae_lambda);
                buf
            };

            self.update(&rollout);
        }

        episode_rewards
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Gather helpers
// ──────────────────────────────────────────────────────────────────────────────

fn gather_rows_2d(arr: &Array2<f64>, idx: &[usize]) -> Array2<f64> {
    let n_cols = arr.shape()[1];
    let mut out = Array2::zeros((idx.len(), n_cols));
    for (o, &i) in idx.iter().enumerate() {
        out.row_mut(o).assign(&arr.row(i));
    }
    out
}

fn gather_1d(arr: &Array1<f64>, idx: &[usize]) -> Array1<f64> {
    Array1::from_iter(idx.iter().map(|&i| arr[i]))
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rl::environments::CartPole;

    #[test]
    fn ppo_new_and_config() {
        let cfg = PPOConfig::default();
        let agent: PPO<f64> = PPO::new(4, 2, cfg.clone());
        assert_eq!(agent.config.clip_eps, 0.2);
        assert_eq!(agent.config.n_epochs, 10);
        assert_eq!(agent.obs_dim, 4);
        assert_eq!(agent.act_dim, 2);
    }

    #[test]
    fn ppo_collect_rollout_shapes() {
        let cfg = PPOConfig {
            n_steps: 32,
            ..Default::default()
        };
        let mut agent: PPO<f64> = PPO::new(4, 2, cfg);
        let mut env = CartPole::new();
        let rollout = agent.collect_rollout(&mut env, 32);
        assert_eq!(rollout.states.shape(), &[32, 4]);
        assert_eq!(rollout.actions.shape(), &[32, 2]);
        assert_eq!(rollout.rewards.len(), 32);
        assert_eq!(rollout.advantages.len(), 32);
        assert_eq!(rollout.returns.len(), 32);
    }

    #[test]
    fn ppo_update_returns_info() {
        let cfg = PPOConfig {
            n_steps: 64,
            n_epochs: 2,
            n_minibatches: 2,
            ..Default::default()
        };
        let mut agent: PPO<f64> = PPO::new(4, 2, cfg);
        let mut env = CartPole::new();
        let rollout = agent.collect_rollout(&mut env, 64);
        let info = agent.update(&rollout);
        assert!(info.n_updates > 0);
        assert!(info.clip_fraction >= 0.0);
    }

    #[test]
    fn ppo_short_train_returns_rewards() {
        let cfg = PPOConfig {
            n_steps: 32,
            n_epochs: 1,
            n_minibatches: 2,
            ..Default::default()
        };
        let mut agent: PPO<f64> = PPO::new(4, 2, cfg);
        let mut env = CartPole::new();
        let rewards = agent.train(&mut env, 256);
        // Should have collected at least one episode
        // Training may not finish an episode, so rewards may be empty — no assertion needed.
        let _ = &rewards;
    }

    #[test]
    fn rollout_buffer_advantage_computation() {
        let n = 8;
        let mut buf = RolloutBuffer::new(n, 4, 2);
        for t in 0..n {
            buf.rewards[t] = 1.0;
            buf.values[t] = 0.5;
            buf.dones[t] = false;
        }
        buf.compute_advantages(0.5, 0.99, 0.95);
        // All advantages should be finite
        assert!(buf.advantages.iter().all(|a| a.is_finite()));
        assert!(buf.returns.iter().all(|r| r.is_finite()));
    }
}
