//! Soft Actor-Critic (SAC) — off-policy maximum-entropy RL algorithm.
//!
//! Implements the variant from Haarnoja et al. 2018 (v2) with:
//! - Dual soft Q-functions for pessimistic value estimation
//! - Target Q-networks updated via Polyak averaging
//! - Entropy temperature `α` (fixed or auto-tuned)
//! - Squashed Gaussian policy (tanh transform of a Gaussian)
//!
//! # References
//! Haarnoja, T., Zhou, A., Hartikainen, K., Tucker, G., Ha, S., Tan, J., ... & Levine, S. (2018).
//! *Soft Actor-Critic Algorithms and Applications*. arXiv:1812.05905.

use crate::rl::environments::Environment;
use crate::rl::replay_buffer::{ReplayBuffer, Transition, XorShift64};
use scirs2_core::ndarray::{Array1, Array2, Axis};

// ──────────────────────────────────────────────────────────────────────────────
// Configuration
// ──────────────────────────────────────────────────────────────────────────────

/// Hyper-parameters for [`SAC`].
#[derive(Debug, Clone)]
pub struct SACConfig {
    /// Replay buffer capacity (default 1_000_000).
    pub buffer_size: usize,
    /// Mini-batch size for each gradient step (default 256).
    pub batch_size: usize,
    /// Number of gradient steps per environment step (default 1).
    pub gradient_steps: usize,
    /// Minimum number of transitions before training begins (default 1_000).
    pub learning_starts: usize,
    /// Discount factor γ (default 0.99).
    pub gamma: f64,
    /// Polyak averaging coefficient τ for target-network updates (default 0.005).
    pub tau: f64,
    /// Initial entropy temperature α (ignored when `auto_alpha = true`, default 0.2).
    pub alpha: f64,
    /// Whether to automatically tune α (default `true`).
    pub auto_alpha: bool,
    /// Target entropy for auto-α (default `-act_dim`).
    pub target_entropy: Option<f64>,
    /// Actor learning rate (default 3e-4).
    pub actor_lr: f64,
    /// Critic learning rate (default 3e-4).
    pub critic_lr: f64,
    /// Alpha (temperature) learning rate (default 3e-4).
    pub alpha_lr: f64,
    /// Hidden layer widths (default [256, 256]).
    pub hidden_dims: Vec<usize>,
    /// Log-std bounds for the squashed Gaussian policy.
    pub log_std_min: f64,
    pub log_std_max: f64,
}

impl Default for SACConfig {
    fn default() -> Self {
        Self {
            buffer_size: 1_000_000,
            batch_size: 256,
            gradient_steps: 1,
            learning_starts: 1_000,
            gamma: 0.99,
            tau: 0.005,
            alpha: 0.2,
            auto_alpha: true,
            target_entropy: None, // set to -act_dim at construction time
            actor_lr: 3e-4,
            critic_lr: 3e-4,
            alpha_lr: 3e-4,
            hidden_dims: vec![256, 256],
            log_std_min: -20.0,
            log_std_max: 2.0,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Training info
// ──────────────────────────────────────────────────────────────────────────────

/// Diagnostic statistics returned by [`SAC::train_step`].
#[derive(Debug, Clone, Default)]
pub struct SACInfo {
    /// Mean critic loss (average of critic1 and critic2).
    pub critic_loss: f64,
    /// Actor loss.
    pub actor_loss: f64,
    /// Current entropy temperature α.
    pub alpha: f64,
    /// Alpha loss (0 when `auto_alpha = false`).
    pub alpha_loss: f64,
}

// Type alias for the complex return type of Mlp::forward_cache.
type MlpForwardCache = (Array2<f64>, Vec<(Array2<f64>, Array2<f64>)>);

// ──────────────────────────────────────────────────────────────────────────────
// Minimal Mlp (re-uses same construction as ppo.rs but in this file for independence)
// ──────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct Linear {
    w: Array2<f64>,
    b: Array1<f64>,
}

impl Linear {
    fn new(in_d: usize, out_d: usize, rng: &mut XorShift64) -> Self {
        let scale = (2.0 / in_d as f64).sqrt();
        let w = Array2::from_shape_fn((in_d, out_d), |_| {
            let u1 = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64 + 1e-20;
            let u2 = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
            let n = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            n * scale
        });
        let b = Array1::zeros(out_d);
        Self { w, b }
    }

    fn forward(&self, x: &Array2<f64>) -> Array2<f64> {
        x.dot(&self.w) + &self.b
    }

    fn backward(
        &self,
        x: &Array2<f64>,
        g: &Array2<f64>,
    ) -> (Array2<f64>, Array1<f64>, Array2<f64>) {
        (x.t().dot(g), g.sum_axis(Axis(0)), g.dot(&self.w.t()))
    }

    fn apply(&mut self, gw: &Array2<f64>, gb: &Array1<f64>, lr: f64) {
        self.w -= &gw.mapv(|v| v * lr);
        self.b -= &gb.mapv(|v| v * lr);
    }

    /// Polyak average: self ← τ·other + (1-τ)·self.
    fn polyak_update(&mut self, other: &Linear, tau: f64) {
        self.w = self.w.mapv(|v| v * (1.0 - tau)) + other.w.mapv(|v| v * tau);
        self.b = self.b.mapv(|v| v * (1.0 - tau)) + other.b.mapv(|v| v * tau);
    }
}

fn relu(x: &Array2<f64>) -> Array2<f64> {
    x.mapv(|v| v.max(0.0))
}

fn relu_bwd(pre: &Array2<f64>, g: &Array2<f64>) -> Array2<f64> {
    g * &pre.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 })
}

#[derive(Debug, Clone)]
struct Mlp {
    layers: Vec<Linear>,
}

impl Mlp {
    fn new(in_d: usize, hidden: &[usize], out_d: usize, rng: &mut XorShift64) -> Self {
        let mut dims = vec![in_d];
        dims.extend_from_slice(hidden);
        dims.push(out_d);
        let layers = dims
            .windows(2)
            .map(|w| Linear::new(w[0], w[1], rng))
            .collect();
        Self { layers }
    }

    fn forward_cache(&self, x: &Array2<f64>) -> MlpForwardCache {
        let mut cur = x.clone();
        let mut cache = Vec::new();
        for (i, l) in self.layers.iter().enumerate() {
            let pre = l.forward(&cur);
            let post = if i < self.layers.len() - 1 {
                relu(&pre)
            } else {
                pre.clone()
            };
            cache.push((cur.clone(), pre));
            cur = post;
        }
        (cur, cache)
    }

    fn forward(&self, x: &Array2<f64>) -> Array2<f64> {
        self.forward_cache(x).0
    }

    fn backward(
        &self,
        cache: &[(Array2<f64>, Array2<f64>)],
        g: Array2<f64>,
    ) -> Vec<(Array2<f64>, Array1<f64>)> {
        let n = self.layers.len();
        let mut grads = Vec::new();
        let mut g = g;
        for i in (0..n).rev() {
            let (x_in, pre) = &cache[i];
            if i < n - 1 {
                g = relu_bwd(pre, &g);
            }
            let (gw, gb, gx) = self.layers[i].backward(x_in, &g);
            grads.push((gw, gb));
            g = gx;
        }
        grads.reverse();
        grads
    }

    fn apply_grads(&mut self, grads: &[(Array2<f64>, Array1<f64>)], lr: f64) {
        for (l, (gw, gb)) in self.layers.iter_mut().zip(grads) {
            l.apply(gw, gb, lr);
        }
    }

    fn polyak_update(&mut self, other: &Mlp, tau: f64) {
        for (a, b) in self.layers.iter_mut().zip(other.layers.iter()) {
            a.polyak_update(b, tau);
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Stochastic actor (squashed Gaussian)
// ──────────────────────────────────────────────────────────────────────────────

/// Stochastic actor outputting a squashed Gaussian policy.
///
/// The network maps observations to `(mean, log_std)`, samples actions from
/// the Gaussian and squashes them through tanh.
pub struct StochasticActor {
    net: Mlp,
    act_dim: usize,
    log_std_min: f64,
    log_std_max: f64,
}

impl StochasticActor {
    fn new(
        obs_dim: usize,
        act_dim: usize,
        hidden: &[usize],
        log_std_min: f64,
        log_std_max: f64,
        rng: &mut XorShift64,
    ) -> Self {
        // Network outputs 2 * act_dim: first half = mean, second half = log_std
        let net = Mlp::new(obs_dim, hidden, act_dim * 2, rng);
        Self {
            net,
            act_dim,
            log_std_min,
            log_std_max,
        }
    }

    /// Sample a squashed action and its log-probability.
    ///
    /// Returns `(action, log_prob)` both of length `act_dim`.
    fn sample(&self, obs: &Array2<f64>, rng: &mut XorShift64) -> (Array2<f64>, Array1<f64>) {
        let out = self.net.forward(obs);
        let batch = obs.shape()[0];
        let mut actions = Array2::zeros((batch, self.act_dim));
        let mut log_probs = Array1::zeros(batch);

        for b in 0..batch {
            let mut lp = 0.0_f64;
            for a in 0..self.act_dim {
                let mu = out[[b, a]];
                let ls = out[[b, self.act_dim + a]].clamp(self.log_std_min, self.log_std_max);
                let sig = ls.exp().max(1e-6);

                // Box-Muller
                let u1 = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64 + 1e-20;
                let u2 = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                let raw = mu + sig * z;
                let act = raw.tanh();

                // log π(a|s) = log N(raw|mu,sig) - log(1 - tanh(raw)^2 + ε)
                let log_n = -0.5 * (z * z) - sig.ln() - 0.5 * (2.0 * std::f64::consts::PI).ln();
                let log_jac = (1.0 - act * act + 1e-6).ln();
                lp += log_n - log_jac;

                actions[[b, a]] = act;
            }
            log_probs[b] = lp;
        }
        (actions, log_probs)
    }

    /// Deterministic action (tanh of mean) for evaluation.
    pub fn deterministic_action(&self, obs: &Array1<f64>) -> Array1<f64> {
        let obs_2d = obs.clone().insert_axis(Axis(0));
        let out = self.net.forward(&obs_2d);
        Array1::from_iter((0..self.act_dim).map(|a| out[[0, a]].tanh()))
    }

    /// Actor gradient update given upstream gradient of the objective
    /// w.r.t. the sampled action (via the reparameterisation trick).
    ///
    /// Returns actor loss (scalar, for logging).
    fn update_actor(
        &mut self,
        obs: &Array2<f64>,
        actions: &Array2<f64>,
        log_probs: &Array1<f64>,
        q_min: &Array1<f64>,
        alpha: f64,
        lr: f64,
    ) -> f64 {
        let batch = obs.shape()[0] as f64;
        // Actor loss: E[ α * log_prob - Q(s,a) ]
        // d_loss / d_log_prob = α / batch  (constant w.r.t. network output)
        // We approximate by flowing gradient through log_prob → net output.

        // Gradient of actor objective w.r.t. the raw network output (mean part):
        // We use the reparameterisation approximation:
        //   g[b, a] ≈ -dQ/da * dtanh/draw / batch
        // Here Q is treated as constant (stop-grad), so we use a finite-diff proxy.
        // For simplicity we use the policy gradient: g = (α * log_prob - q) / batch.
        // This is the "score function" direction for the mean — an approximation
        // sufficient for pedagogical purposes.

        let loss = log_probs
            .iter()
            .zip(q_min.iter())
            .map(|(lp, q)| alpha * lp - q)
            .sum::<f64>()
            / batch;

        // Gradient of loss w.r.t. net output (2*act_dim columns):
        // For the mean column: d(alpha*log_prob - q) / d_mu ≈ 0 (gradient only via log_prob)
        // We use a simplified gradient: push means towards actions that have higher Q.
        let mut g_out = Array2::zeros((obs.shape()[0], self.act_dim * 2));
        let out = self.net.forward(obs);
        for b in 0..obs.shape()[0] {
            for a in 0..self.act_dim {
                let mu = out[[b, a]];
                let ls = out[[b, self.act_dim + a]].clamp(self.log_std_min, self.log_std_max);
                let sig = ls.exp().max(1e-6);
                let act = actions[[b, a]]; // squashed action = tanh(raw)
                let raw = act.clamp(-0.9999, 0.9999).atanh(); // inverse tanh

                // Gradient of alpha * log_prob w.r.t. mu:
                //   d/d_mu [log N(raw|mu,sig)] = (raw - mu) / sig^2
                let g_mu_lp = (raw - mu) / (sig * sig);
                // Gradient of alpha * log_prob w.r.t. log_sig:
                //   d/d_ls [log N] = (raw-mu)^2/sig^2 - 1  (chain rule through sig)
                let g_ls_lp = (raw - mu).powi(2) / (sig * sig) - 1.0;

                g_out[[b, a]] = alpha * g_mu_lp / batch;
                g_out[[b, self.act_dim + a]] = alpha * g_ls_lp / batch;
                // Subtract Q gradient (Q is monotone w.r.t. action direction)
                // Approximate: push action toward higher Q via tanh'·sign(q_advantage)
                let dtanh = 1.0 - act * act + 1e-6;
                g_out[[b, a]] -= q_min[b] * dtanh / batch;
            }
        }

        let (_, cache) = self.net.forward_cache(obs);
        let grads = self.net.backward(&cache, g_out);
        self.net.apply_grads(&grads, lr);
        loss
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Critic (soft Q-function)
// ──────────────────────────────────────────────────────────────────────────────

/// Soft Q-function: maps `(observation, action)` → scalar Q-value.
pub struct Critic {
    net: Mlp,
}

impl Critic {
    fn new(obs_dim: usize, act_dim: usize, hidden: &[usize], rng: &mut XorShift64) -> Self {
        let net = Mlp::new(obs_dim + act_dim, hidden, 1, rng);
        Self { net }
    }

    fn forward(&self, obs: &Array2<f64>, actions: &Array2<f64>) -> Array1<f64> {
        let sa = ndarray_hstack(obs, actions);
        self.net.forward(&sa).column(0).to_owned()
    }

    /// MSE update toward `targets`.  Returns (loss, cache_sa).
    fn update(
        &mut self,
        obs: &Array2<f64>,
        actions: &Array2<f64>,
        targets: &Array1<f64>,
        lr: f64,
    ) -> f64 {
        let sa = ndarray_hstack(obs, actions);
        let (pred, cache) = self.net.forward_cache(&sa);
        let pred_1d = pred.column(0).to_owned();
        let batch = obs.shape()[0] as f64;

        let mut g_out = Array2::zeros((obs.shape()[0], 1));
        let mut loss = 0.0_f64;
        for b in 0..obs.shape()[0] {
            let residual = pred_1d[b] - targets[b];
            loss += residual * residual;
            g_out[[b, 0]] = 2.0 * residual / batch;
        }
        loss /= batch;

        let grads = self.net.backward(&cache, g_out);
        self.net.apply_grads(&grads, lr);
        loss
    }

    fn polyak_update(&mut self, other: &Critic, tau: f64) {
        self.net.polyak_update(&other.net, tau);
    }
}

fn ndarray_hstack(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let rows = a.shape()[0];
    let ca = a.shape()[1];
    let cb = b.shape()[1];
    let mut out = Array2::zeros((rows, ca + cb));
    out.slice_mut(scirs2_core::ndarray::s![.., ..ca]).assign(a);
    out.slice_mut(scirs2_core::ndarray::s![.., ca..]).assign(b);
    out
}

// ──────────────────────────────────────────────────────────────────────────────
// SAC
// ──────────────────────────────────────────────────────────────────────────────

/// Soft Actor-Critic agent.
///
/// # Example (abbreviated)
/// ```rust,ignore
/// use scirs2_neural::rl::{sac::{SAC, SACConfig}, environments::CartPole};
/// let cfg = SACConfig {
///     learning_starts: 100,
///     batch_size: 64,
///     ..Default::default()
/// };
/// let mut agent: SAC<f64> = SAC::new(4, 1, cfg);
/// let mut env = CartPole::new();
/// let rewards = agent.train(&mut env, 5_000);
/// ```
pub struct SAC<F> {
    _phantom: std::marker::PhantomData<F>,
    /// Stochastic Gaussian policy.
    pub actor: StochasticActor,
    /// First soft Q-function.
    pub critic1: Critic,
    /// Second soft Q-function.
    pub critic2: Critic,
    /// Target of critic1 (Polyak-averaged copy).
    pub target_critic1: Critic,
    /// Target of critic2 (Polyak-averaged copy).
    pub target_critic2: Critic,
    /// Experience replay buffer.
    pub replay_buffer: ReplayBuffer<f64>,
    /// Hyper-parameters.
    pub config: SACConfig,
    /// Current entropy temperature.
    log_alpha: f64,
    target_entropy: f64,
    rng: XorShift64,
    obs_dim: usize,
    act_dim: usize,
}

impl<F: 'static> SAC<F> {
    /// Construct a new SAC agent.
    pub fn new(obs_dim: usize, act_dim: usize, config: SACConfig) -> Self {
        let mut rng = XorShift64::new(obs_dim as u64 ^ act_dim as u64 ^ 0xcafef00d);
        let actor = StochasticActor::new(
            obs_dim,
            act_dim,
            &config.hidden_dims.clone(),
            config.log_std_min,
            config.log_std_max,
            &mut rng,
        );
        let critic1 = Critic::new(obs_dim, act_dim, &config.hidden_dims.clone(), &mut rng);
        let critic2 = Critic::new(obs_dim, act_dim, &config.hidden_dims.clone(), &mut rng);
        let target_critic1 = Critic::new(obs_dim, act_dim, &config.hidden_dims.clone(), &mut rng);
        let target_critic2 = Critic::new(obs_dim, act_dim, &config.hidden_dims.clone(), &mut rng);

        let replay_buffer = ReplayBuffer::new(config.buffer_size, obs_dim, act_dim);
        let target_entropy = config.target_entropy.unwrap_or(-(act_dim as f64));
        let log_alpha = config.alpha.max(1e-6).ln();

        Self {
            _phantom: std::marker::PhantomData,
            actor,
            critic1,
            critic2,
            target_critic1,
            target_critic2,
            replay_buffer,
            log_alpha,
            target_entropy,
            config,
            rng,
            obs_dim,
            act_dim,
        }
    }

    /// Select an action for a single observation (with exploration noise).
    pub fn select_action(&mut self, obs: &Array1<f64>) -> Array1<f64> {
        let obs_2d = obs.clone().insert_axis(Axis(0));
        let (actions, _) = self.actor.sample(&obs_2d, &mut self.rng);
        actions.row(0).to_owned()
    }

    /// Deterministic action (for evaluation without exploration).
    pub fn select_action_deterministic(&self, obs: &Array1<f64>) -> Array1<f64> {
        self.actor.deterministic_action(obs)
    }

    /// Perform one gradient step using a mini-batch from the replay buffer.
    ///
    /// Returns diagnostic info.
    pub fn train_step(&mut self) -> SACInfo {
        let alpha = self.log_alpha.exp();
        let batch_size = self.config.batch_size;
        let gamma = self.config.gamma;
        let tau = self.config.tau;
        let critic_lr = self.config.critic_lr;
        let actor_lr = self.config.actor_lr;

        let tr = self.replay_buffer.sample(batch_size, &mut self.rng);
        let Transition {
            states,
            actions,
            rewards,
            next_states,
            dones,
        } = tr;

        // ── Critic targets ───────────────────────────────────────────────
        let (next_actions, next_log_probs) = self.actor.sample(&next_states, &mut self.rng);
        let q1_next = self.target_critic1.forward(&next_states, &next_actions);
        let q2_next = self.target_critic2.forward(&next_states, &next_actions);
        let q_next_min: Array1<f64> = q1_next
            .iter()
            .zip(q2_next.iter())
            .map(|(a, b)| a.min(*b))
            .collect();

        let targets: Array1<f64> = rewards
            .iter()
            .zip(dones.iter())
            .zip(q_next_min.iter())
            .zip(next_log_probs.iter())
            .map(|(((r, d), q_n), lp)| {
                let mask = if *d { 0.0 } else { 1.0 };
                r + gamma * mask * (q_n - alpha * lp)
            })
            .collect();

        // ── Critic updates ───────────────────────────────────────────────
        let c1_loss = self.critic1.update(&states, &actions, &targets, critic_lr);
        let c2_loss = self.critic2.update(&states, &actions, &targets, critic_lr);
        let critic_loss = (c1_loss + c2_loss) * 0.5;

        // ── Actor update ─────────────────────────────────────────────────
        let (new_actions, new_log_probs) = self.actor.sample(&states, &mut self.rng);
        let q1_new = self.critic1.forward(&states, &new_actions);
        let q2_new = self.critic2.forward(&states, &new_actions);
        let q_min: Array1<f64> = q1_new
            .iter()
            .zip(q2_new.iter())
            .map(|(a, b)| a.min(*b))
            .collect();

        let actor_loss = self.actor.update_actor(
            &states,
            &new_actions,
            &new_log_probs,
            &q_min,
            alpha,
            actor_lr,
        );

        // ── Alpha update ─────────────────────────────────────────────────
        let alpha_loss = if self.config.auto_alpha {
            let mean_lp = new_log_probs.mean().unwrap_or(0.0);
            let g = -(mean_lp + self.target_entropy);
            self.log_alpha += self.config.alpha_lr * g;
            self.log_alpha = self.log_alpha.clamp(-20.0, 2.0);
            g.abs()
        } else {
            0.0
        };

        // ── Polyak target updates ────────────────────────────────────────
        self.target_critic1.polyak_update(&self.critic1, tau);
        self.target_critic2.polyak_update(&self.critic2, tau);

        SACInfo {
            critic_loss,
            actor_loss,
            alpha: self.log_alpha.exp(),
            alpha_loss,
        }
    }

    /// Full training loop.
    ///
    /// Returns episodic rewards (one entry per completed episode).
    pub fn train<E>(&mut self, env: &mut E, total_timesteps: usize) -> Vec<f64>
    where
        E: Environment<State = Array1<f64>, Action = Array1<f64>>,
    {
        let mut episode_rewards: Vec<f64> = Vec::new();
        let mut ep_reward = 0.0_f64;
        let mut state = env.reset();
        let learning_starts = self.config.learning_starts;
        let gradient_steps = self.config.gradient_steps;
        let batch_size = self.config.batch_size;

        for t in 0..total_timesteps {
            // Random actions during warm-up
            let action = if t < learning_starts {
                // Random action (uniform in [-1, 1] for squashed Gaussian)
                Array1::from_shape_fn(self.act_dim, |_| {
                    let u = (self.rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
                    u * 2.0 - 1.0
                })
            } else {
                self.select_action(&state)
            };

            let (next_state, reward, done) = env.step(&action);
            ep_reward += reward;

            self.replay_buffer
                .push(state.clone(), action, reward, next_state.clone(), done);

            if done {
                episode_rewards.push(ep_reward);
                ep_reward = 0.0;
                state = env.reset();
            } else {
                state = next_state;
            }

            // Gradient updates
            if t >= learning_starts && self.replay_buffer.len() >= batch_size {
                for _ in 0..gradient_steps {
                    let _ = self.train_step();
                }
            }
        }

        episode_rewards
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rl::environments::ContinuousCartPole;

    fn small_config() -> SACConfig {
        SACConfig {
            buffer_size: 1_000,
            batch_size: 32,
            gradient_steps: 1,
            learning_starts: 50,
            hidden_dims: vec![32, 32],
            ..Default::default()
        }
    }

    #[test]
    fn sac_new_and_config() {
        let cfg = small_config();
        let agent: SAC<f64> = SAC::new(4, 1, cfg.clone());
        assert_eq!(agent.obs_dim, 4);
        assert_eq!(agent.act_dim, 1);
        assert_eq!(agent.replay_buffer.capacity(), cfg.buffer_size);
    }

    #[test]
    fn sac_select_action_shape() {
        let mut agent: SAC<f64> = SAC::new(4, 2, small_config());
        let obs = Array1::zeros(4);
        let act = agent.select_action(&obs);
        assert_eq!(act.len(), 2);
        // Squashed actions must lie in (-1, 1)
        assert!(act.iter().all(|&a| a > -1.0 && a < 1.0));
    }

    #[test]
    fn sac_replay_buffer_fills() {
        let mut agent: SAC<f64> = SAC::new(4, 1, small_config());
        let mut env = ContinuousCartPole::new();
        let mut state = env.reset();
        for _ in 0..60 {
            let act = agent.select_action(&state);
            let (ns, r, done) = env.step(&act);
            agent
                .replay_buffer
                .push(state.clone(), act, r, ns.clone(), done);
            state = if done { env.reset() } else { ns };
        }
        assert!(agent.replay_buffer.len() > 0);
    }

    #[test]
    fn sac_train_step_returns_finite_losses() {
        let cfg = small_config();
        let mut agent: SAC<f64> = SAC::new(4, 1, cfg.clone());
        let mut env = ContinuousCartPole::new();

        // Fill the buffer
        let mut state = env.reset();
        for _ in 0..cfg.batch_size * 2 {
            let act = agent.select_action(&state);
            let (ns, r, done) = env.step(&act);
            agent
                .replay_buffer
                .push(state.clone(), act, r, ns.clone(), done);
            state = if done { env.reset() } else { ns };
        }

        let info = agent.train_step();
        assert!(info.critic_loss.is_finite());
        assert!(info.actor_loss.is_finite());
        assert!(info.alpha > 0.0);
    }

    #[test]
    fn sac_short_train_loop() {
        let cfg = SACConfig {
            buffer_size: 500,
            batch_size: 16,
            learning_starts: 32,
            hidden_dims: vec![16, 16],
            ..Default::default()
        };
        let mut agent: SAC<f64> = SAC::new(4, 1, cfg);
        let mut env = ContinuousCartPole::new();
        let _rewards = agent.train(&mut env, 200);
        // Just ensure it runs without panic
    }
}
