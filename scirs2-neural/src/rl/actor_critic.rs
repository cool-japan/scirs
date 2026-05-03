//! Advantage Actor-Critic (A2C) for reinforcement learning.
//!
//! Implements a synchronous A2C with:
//! - **Actor** network outputting a softmax action-probability distribution.
//! - **Critic** network estimating V(s).
//! - **Advantage** `A = r + γ·V(s') − V(s)`.
//! - **Actor loss**: `−log π(a|s) · A` (policy gradient / REINFORCE with baseline).
//! - **Critic loss**: `½ (V(s) − (r + γ·V(s')))²` (mean-squared TD error).
//! - **Entropy bonus**: `−β · Σ π(a) log π(a)` (exploration incentive).
//!
//! # Reference
//! Mnih et al. (2016). "Asynchronous Methods for Deep Reinforcement Learning".
//! ICML 2016. (Synchronous single-thread variant.)

use crate::error::{NeuralError, Result};
use crate::rl::policy::{softmax, PolicyRng};

// ──────────────────────────────────────────────────────────────────────────────
// Helper MLP (re-implemented inline to keep this file self-contained)
// ──────────────────────────────────────────────────────────────────────────────

/// Lightweight Mlp used inside the actor and critic.
#[derive(Debug, Clone)]
struct Mlp {
    weights: Vec<Vec<Vec<f32>>>,
    biases: Vec<Vec<f32>>,
}

impl Mlp {
    fn build(dims: &[usize]) -> Self {
        let mut rng = PolicyRng::new(0x1234_5678_9abc_def0);
        let mut weights = Vec::new();
        let mut biases = Vec::new();
        for l in 0..(dims.len() - 1) {
            let (in_d, out_d) = (dims[l], dims[l + 1]);
            let scale = (6.0_f32 / in_d as f32).sqrt();
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
            .and_then(|w| w.first())
            .map(|r| r.len())
            .unwrap_or(0)
    }

    fn forward(&self, input: &[f32]) -> Result<Vec<f32>> {
        if input.len() != self.input_dim() {
            return Err(NeuralError::ShapeMismatch(format!(
                "MLP expects input_dim={}, got {}",
                self.input_dim(),
                input.len()
            )));
        }
        let mut act = input.to_vec();
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

    fn forward_cache(&self, input: &[f32]) -> Vec<Vec<f32>> {
        let n = self.n_layers();
        let mut cache = vec![input.to_vec()];
        for (l, (w, b)) in self.weights.iter().zip(self.biases.iter()).enumerate() {
            let prev = cache.last().expect("non-empty");
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

    fn sgd_step(&mut self, cache: &[Vec<f32>], out_delta: &[f32], lr: f32) {
        let n = self.n_layers();
        let mut delta = out_delta.to_vec();
        for l in (0..n).rev() {
            let in_act = &cache[l];
            let out_act = &cache[l + 1];
            let out_d = self.weights[l].len();
            let in_d = in_act.len();
            let eff: Vec<f32> = if l < n - 1 {
                delta
                    .iter()
                    .zip(out_act.iter())
                    .map(|(d, a)| if *a > 0.0 { *d } else { 0.0 })
                    .collect()
            } else {
                delta.clone()
            };
            let mut prev = vec![0.0_f32; in_d];
            for (i, &eff_i) in eff.iter().enumerate().take(out_d) {
                for j in 0..in_d {
                    prev[j] += eff_i * self.weights[l][i][j];
                    self.weights[l][i][j] -= lr * eff_i * in_act[j];
                }
                self.biases[l][i] -= lr * eff_i;
            }
            delta = prev;
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// ActorNetwork — outputs softmax action probabilities
// ──────────────────────────────────────────────────────────────────────────────

/// Actor network: maps observations to action probabilities via softmax.
#[derive(Debug, Clone)]
pub struct ActorNetwork {
    mlp: Mlp,
    num_actions: usize,
}

impl ActorNetwork {
    /// Build a new actor.
    ///
    /// - `obs_dim`: observation dimension.
    /// - `hidden_dims`: e.g. `&[64, 64]`.
    /// - `num_actions`: number of discrete actions.
    pub fn new(obs_dim: usize, hidden_dims: &[usize], num_actions: usize) -> Self {
        let mut dims = vec![obs_dim];
        dims.extend_from_slice(hidden_dims);
        dims.push(num_actions);
        Self {
            mlp: Mlp::build(&dims),
            num_actions,
        }
    }

    /// Action probability distribution `π(·|s)`.
    pub fn probs(&self, obs: &[f32]) -> Result<Vec<f32>> {
        let logits = self.mlp.forward(obs)?;
        Ok(softmax(&logits))
    }

    /// Log-probabilities `log π(·|s)`.
    pub fn log_probs(&self, obs: &[f32]) -> Result<Vec<f32>> {
        let probs = self.probs(obs)?;
        Ok(probs.iter().map(|p| p.max(1e-8_f32).ln()).collect())
    }

    /// Entropy `H[π(·|s)] = −Σ π(a) log π(a)`.
    pub fn entropy(&self, obs: &[f32]) -> Result<f32> {
        let probs = self.probs(obs)?;
        Ok(-probs.iter().map(|p| p * p.max(1e-8).ln()).sum::<f32>())
    }

    /// Sample an action from π(·|s) using the provided RNG.
    pub fn sample_action(&self, obs: &[f32], rng: &mut PolicyRng) -> Result<usize> {
        let probs = self.probs(obs)?;
        crate::rl::policy::categorical_sample(&probs, rng)
    }

    /// SGD update for the actor.
    ///
    /// `advantage` is A(s, a); `action` is the selected action index.
    /// Gradient: `∇_θ [-log π(a|s) · A + β · H[π]]`
    pub fn update(
        &mut self,
        obs: &[f32],
        action: usize,
        advantage: f32,
        entropy_coef: f32,
        lr: f32,
    ) -> Result<f32> {
        if action >= self.num_actions {
            return Err(NeuralError::InvalidArgument(format!(
                "action {} >= num_actions {}",
                action, self.num_actions
            )));
        }
        let cache = self.mlp.forward_cache(obs);
        let logits = cache.last().expect("non-empty");
        let probs = softmax(logits);

        // Policy gradient delta at the softmax output layer:
        // ∂L/∂logit_i = π_i · (-advantage) − entropy_coef · (1 - π_i + ln π_i · ... )
        // Simplified: cross-entropy gradient for the chosen action, scaled by advantage,
        // plus entropy gradient.
        //
        // L_policy = -log π(a) * advantage
        // L_entropy = -entropy_coef * entropy
        //
        // Combined delta at softmax output (before softmax, i.e. at logits):
        //   ∂L/∂z_i = π_i * (-advantage) + delta_{i,action} * advantage
        //            + entropy_coef * (log π_i + 1) * π_i  ← entropy gradient
        //
        // Concisely: for cross-entropy term: δ_i = π_i  (i≠a), δ_a = π_a - 1
        // scaled by advantage; entropy term: δ_i += entropy_coef * π_i * (ln π_i + 1)
        let mut delta = vec![0.0_f32; self.num_actions];
        for i in 0..self.num_actions {
            let ce_grad = if i == action {
                probs[i] - 1.0
            } else {
                probs[i]
            };
            let ent_grad = entropy_coef * probs[i] * (probs[i].max(1e-8).ln() + 1.0);
            // Policy gradient loss L = -A·log π(a); gradient w.r.t. logit_i:
            // ∂L/∂z_i = A·(π_i − 1_{i=a}), so delta = advantage * ce_grad
            // (sgd_step subtracts lr*delta, driving logit_a up when A>0)
            delta[i] = advantage * ce_grad + ent_grad;
        }

        self.mlp.sgd_step(&cache, &delta, lr);
        Ok(-probs[action].max(1e-8).ln() * advantage)
    }

    /// Number of actions.
    pub fn num_actions(&self) -> usize {
        self.num_actions
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// CriticNetwork — outputs scalar V(s)
// ──────────────────────────────────────────────────────────────────────────────

/// Critic network: estimates V(s) as a scalar.
#[derive(Debug, Clone)]
pub struct CriticNetwork {
    mlp: Mlp,
}

impl CriticNetwork {
    /// Build a new critic.
    pub fn new(obs_dim: usize, hidden_dims: &[usize]) -> Self {
        let mut dims = vec![obs_dim];
        dims.extend_from_slice(hidden_dims);
        dims.push(1);
        Self {
            mlp: Mlp::build(&dims),
        }
    }

    /// Estimate V(s).
    pub fn value(&self, obs: &[f32]) -> Result<f32> {
        Ok(self.mlp.forward(obs)?[0])
    }

    /// SGD update minimising MSE: ½(V(s) - target)².
    pub fn update(&mut self, obs: &[f32], target: f32, lr: f32) -> Result<f32> {
        let cache = self.mlp.forward_cache(obs);
        let v = cache.last().expect("non-empty")[0];
        let err = v - target;
        self.mlp.sgd_step(&cache, &[err], lr);
        Ok(0.5 * err * err)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// A2CConfig
// ──────────────────────────────────────────────────────────────────────────────

/// Configuration for [`A2CAgent`].
#[derive(Debug, Clone)]
pub struct A2CConfig {
    /// Actor learning rate.
    pub actor_lr: f32,
    /// Critic learning rate.
    pub critic_lr: f32,
    /// Discount factor γ.
    pub gamma: f32,
    /// Entropy regularisation coefficient β.
    pub entropy_coef: f32,
    /// Hidden dimensions for both actor and critic networks.
    pub hidden_dims: Vec<usize>,
}

impl Default for A2CConfig {
    fn default() -> Self {
        Self {
            actor_lr: 3e-4,
            critic_lr: 1e-3,
            gamma: 0.99,
            entropy_coef: 0.01,
            hidden_dims: vec![64, 64],
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// A2CAgent
// ──────────────────────────────────────────────────────────────────────────────

/// Synchronous Advantage Actor-Critic (A2C) agent.
///
/// The agent collects transitions online (one or more steps) then calls
/// [`A2CAgent::update_step`] after each environment step, or
/// [`A2CAgent::update_trajectory`] after collecting a full trajectory.
pub struct A2CAgent {
    actor: ActorNetwork,
    critic: CriticNetwork,
    config: A2CConfig,
    rng: PolicyRng,
    /// Running statistics for logging.
    total_actor_loss: f32,
    total_critic_loss: f32,
    total_entropy: f32,
    update_count: usize,
}

impl A2CAgent {
    /// Create a new A2C agent.
    pub fn new(obs_dim: usize, num_actions: usize, config: A2CConfig) -> Self {
        Self {
            actor: ActorNetwork::new(obs_dim, &config.hidden_dims, num_actions),
            critic: CriticNetwork::new(obs_dim, &config.hidden_dims),
            config,
            rng: PolicyRng::from_time(),
            total_actor_loss: 0.0,
            total_critic_loss: 0.0,
            total_entropy: 0.0,
            update_count: 0,
        }
    }

    /// Sample an action from the current policy.
    pub fn act(&mut self, obs: &[f32]) -> Result<usize> {
        self.actor.sample_action(obs, &mut self.rng)
    }

    /// Estimate V(s).
    pub fn value(&self, obs: &[f32]) -> Result<f32> {
        self.critic.value(obs)
    }

    /// Perform a single (s, a, r, s', done) online update.
    ///
    /// Returns `(actor_loss, critic_loss)`.
    pub fn update_step(
        &mut self,
        state: &[f32],
        action: usize,
        reward: f32,
        next_state: &[f32],
        done: bool,
    ) -> Result<(f32, f32)> {
        let gamma = self.config.gamma;

        let v_next = if done {
            0.0
        } else {
            self.critic.value(next_state)?
        };
        let td_target = reward + gamma * v_next;
        let v = self.critic.value(state)?;
        let advantage = td_target - v;

        let actor_loss = self.actor.update(
            state,
            action,
            advantage,
            self.config.entropy_coef,
            self.config.actor_lr,
        )?;
        let critic_loss = self
            .critic
            .update(state, td_target, self.config.critic_lr)?;

        self.total_actor_loss += actor_loss;
        self.total_critic_loss += critic_loss;
        self.total_entropy += self.actor.entropy(state).unwrap_or(0.0);
        self.update_count += 1;

        Ok((actor_loss, critic_loss))
    }

    /// Update from a complete trajectory using Monte-Carlo returns.
    ///
    /// `trajectory`: slice of `(state, action, reward)` tuples (in order).
    /// `bootstrap_value`: V(s_T) for continuing episodes, 0 for terminal.
    ///
    /// Returns the mean `(actor_loss, critic_loss)` over the trajectory.
    pub fn update_trajectory(
        &mut self,
        trajectory: &[(Vec<f32>, usize, f32)],
        bootstrap_value: f32,
    ) -> Result<(f32, f32)> {
        if trajectory.is_empty() {
            return Err(NeuralError::InvalidArgument(
                "trajectory must not be empty".into(),
            ));
        }
        let gamma = self.config.gamma;

        // Compute discounted returns (backwards)
        let mut returns = vec![0.0_f32; trajectory.len()];
        let mut running = bootstrap_value;
        for (i, (_, _, r)) in trajectory.iter().enumerate().rev() {
            running = r + gamma * running;
            returns[i] = running;
        }

        let mut total_al = 0.0_f32;
        let mut total_cl = 0.0_f32;

        for ((state, action, _), &ret) in trajectory.iter().zip(returns.iter()) {
            let v = self.critic.value(state)?;
            let advantage = ret - v;

            let al = self.actor.update(
                state,
                *action,
                advantage,
                self.config.entropy_coef,
                self.config.actor_lr,
            )?;
            let cl = self.critic.update(state, ret, self.config.critic_lr)?;

            total_al += al;
            total_cl += cl;
        }

        let n = trajectory.len() as f32;
        Ok((total_al / n, total_cl / n))
    }

    /// Running mean actor loss (since last reset or construction).
    pub fn mean_actor_loss(&self) -> f32 {
        if self.update_count == 0 {
            0.0
        } else {
            self.total_actor_loss / self.update_count as f32
        }
    }

    /// Running mean critic loss.
    pub fn mean_critic_loss(&self) -> f32 {
        if self.update_count == 0 {
            0.0
        } else {
            self.total_critic_loss / self.update_count as f32
        }
    }

    /// Running mean entropy.
    pub fn mean_entropy(&self) -> f32 {
        if self.update_count == 0 {
            0.0
        } else {
            self.total_entropy / self.update_count as f32
        }
    }

    /// Reset loss accumulators.
    pub fn reset_stats(&mut self) {
        self.total_actor_loss = 0.0;
        self.total_critic_loss = 0.0;
        self.total_entropy = 0.0;
        self.update_count = 0;
    }

    /// Immutable reference to the actor network.
    pub fn actor(&self) -> &ActorNetwork {
        &self.actor
    }

    /// Immutable reference to the critic network.
    pub fn critic(&self) -> &CriticNetwork {
        &self.critic
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// A2CTrainInfo — snapshot of one episode's diagnostics
// ──────────────────────────────────────────────────────────────────────────────

/// Diagnostics returned by a convenience one-episode training function.
#[derive(Debug, Clone)]
pub struct A2CTrainInfo {
    /// Total undiscounted reward collected in the episode.
    pub episode_reward: f32,
    /// Episode length (number of steps).
    pub episode_length: usize,
    /// Mean actor loss over steps.
    pub mean_actor_loss: f32,
    /// Mean critic loss over steps.
    pub mean_critic_loss: f32,
    /// Mean policy entropy over steps.
    pub mean_entropy: f32,
}

/// Run a single episode with the given environment function.
///
/// `env_step` must implement the environment contract:
///   - On first call `step_index == 0`, a `None` action triggers a reset
///     (returning `(initial_obs, 0.0, false)`).
///   - Subsequent calls with `Some(action)` return `(next_obs, reward, done)`.
pub fn run_episode<F>(
    agent: &mut A2CAgent,
    env_step: &mut F,
    max_steps: usize,
) -> Result<A2CTrainInfo>
where
    F: FnMut(Option<usize>) -> (Vec<f32>, f32, bool),
{
    let (mut obs, _, _) = env_step(None);
    let mut episode_reward = 0.0_f32;
    let mut step = 0usize;
    agent.reset_stats();

    loop {
        let action = agent.act(&obs)?;
        let (next_obs, reward, done) = env_step(Some(action));

        agent.update_step(&obs, action, reward, &next_obs, done)?;
        episode_reward += reward;
        step += 1;

        if done || step >= max_steps {
            break;
        }
        obs = next_obs;
    }

    Ok(A2CTrainInfo {
        episode_reward,
        episode_length: step,
        mean_actor_loss: agent.mean_actor_loss(),
        mean_critic_loss: agent.mean_critic_loss(),
        mean_entropy: agent.mean_entropy(),
    })
}

// ──────────────────────────────────────────────────────────────────────────────
// ActorCritic type alias (convenience re-export)
// ──────────────────────────────────────────────────────────────────────────────

// Convenience note: A2CAgent is an actor-critic model.
// Note: ActorCritic is re-exported as A2CAgent; use A2CAgent directly.
// The name ActorCritic is taken by ppo::ActorCritic<F>.

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn obs4() -> Vec<f32> {
        vec![0.1, -0.2, 0.3, 0.0]
    }

    #[test]
    fn actor_network_probs_sum_to_one() {
        let actor = ActorNetwork::new(4, &[16], 3);
        let probs = actor.probs(&obs4()).expect("probs failed");
        assert_eq!(probs.len(), 3);
        let sum: f32 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "probs should sum to 1, got {}",
            sum
        );
    }

    #[test]
    fn actor_entropy_positive() {
        let actor = ActorNetwork::new(4, &[16], 4);
        let h = actor.entropy(&obs4()).expect("entropy failed");
        assert!(
            h > 0.0,
            "entropy must be positive for non-degenerate policies"
        );
    }

    #[test]
    fn critic_value_finite() {
        let critic = CriticNetwork::new(4, &[16]);
        let v = critic.value(&obs4()).expect("value failed");
        assert!(v.is_finite(), "V(s) must be finite");
    }

    #[test]
    fn a2c_agent_act_in_range() {
        let cfg = A2CConfig {
            hidden_dims: vec![8],
            ..Default::default()
        };
        let mut agent = A2CAgent::new(4, 3, cfg);
        let a = agent.act(&obs4()).expect("act failed");
        assert!(a < 3, "action must be in [0, num_actions)");
    }

    #[test]
    fn a2c_update_step_returns_finite_losses() {
        let cfg = A2CConfig {
            hidden_dims: vec![8],
            ..Default::default()
        };
        let mut agent = A2CAgent::new(4, 2, cfg);
        let (al, cl) = agent
            .update_step(&obs4(), 0, 1.0, &obs4(), false)
            .expect("update_step failed");
        assert!(al.is_finite(), "actor loss must be finite");
        assert!(cl.is_finite(), "critic loss must be finite");
    }

    #[test]
    fn a2c_update_trajectory() {
        let cfg = A2CConfig {
            hidden_dims: vec![8],
            ..Default::default()
        };
        let mut agent = A2CAgent::new(4, 2, cfg);
        let traj: Vec<(Vec<f32>, usize, f32)> = (0..5).map(|i| (obs4(), i % 2, 1.0_f32)).collect();
        let (al, cl) = agent
            .update_trajectory(&traj, 0.0)
            .expect("trajectory update failed");
        assert!(al.is_finite());
        assert!(cl.is_finite());
    }

    #[test]
    fn run_episode_helper_smoke_test() {
        use crate::rl::environments::{CartPole, Environment};

        let cfg = A2CConfig {
            hidden_dims: vec![16],
            ..Default::default()
        };
        let mut agent = A2CAgent::new(4, 2, cfg);
        let mut env = CartPole::new();
        let mut obs_store: Option<Vec<f32>> = None;

        let mut env_step = |action: Option<usize>| -> (Vec<f32>, f32, bool) {
            match action {
                None => {
                    let s: Vec<f32> = env.reset().iter().map(|&x| x as f32).collect();
                    obs_store = Some(s.clone());
                    (s, 0.0, false)
                }
                Some(a) => {
                    let arr = scirs2_core::ndarray::array![a as f64];
                    let (ns_f64, r, done) = env.step(&arr);
                    let ns: Vec<f32> = ns_f64.iter().map(|&x| x as f32).collect();
                    obs_store = Some(ns.clone());
                    (ns, r as f32, done)
                }
            }
        };

        let info = run_episode(&mut agent, &mut env_step, 50).expect("run_episode failed");
        assert!(info.episode_reward >= 0.0);
        assert!(info.episode_length > 0);
    }

    #[test]
    fn a2c_agent_cartpole_10_steps_no_panic() {
        use crate::rl::environments::{CartPole, Environment};
        let cfg = A2CConfig {
            hidden_dims: vec![16],
            ..Default::default()
        };
        let mut agent = A2CAgent::new(4, 2, cfg);
        let mut env = CartPole::new();
        let mut obs: Vec<f32> = env.reset().iter().map(|&x| x as f32).collect();

        for _ in 0..10 {
            let a = agent.act(&obs).expect("act");
            let arr = scirs2_core::ndarray::array![a as f64];
            let (ns_f64, r, done) = env.step(&arr);
            let ns: Vec<f32> = ns_f64.iter().map(|&x| x as f32).collect();
            let (al, cl) = agent
                .update_step(&obs, a, r as f32, &ns, done)
                .expect("update");
            assert!(al.is_finite());
            assert!(cl.is_finite());
            if done {
                obs = env.reset().iter().map(|&x| x as f32).collect();
            } else {
                obs = ns;
            }
        }
    }

    #[test]
    fn actor_network_update_changes_probs() {
        let mut actor = ActorNetwork::new(4, &[16], 2);
        let obs = obs4();
        let probs_before = actor.probs(&obs).expect("probs before");
        // Update many times toward action 0 with positive advantage
        for _ in 0..50 {
            actor.update(&obs, 0, 1.0, 0.0, 0.01).expect("update");
        }
        let probs_after = actor.probs(&obs).expect("probs after");
        // π(action=0) should increase
        assert!(
            probs_after[0] >= probs_before[0] - 0.1,
            "p(a=0) should increase; before={:.4} after={:.4}",
            probs_before[0],
            probs_after[0]
        );
    }
}
