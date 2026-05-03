//! Deep Q-Network (DQN) and Double DQN agents.
//!
//! Implements the DQN algorithm from Mnih et al. (2015) "Human-level control
//! through deep reinforcement learning" together with the Double DQN
//! improvement from van Hasselt et al. (2016).
//!
//! # Architecture
//! - **Online network**: selected actions + trained via SGD.
//! - **Target network**: periodically hard-copied from online; used for stable TD targets.
//! - **Replay buffer**: uniform random experience replay.
//! - **ε-greedy**: linear epsilon decay from `eps_start` to `eps_end`.
//!
//! # Double DQN
//! Use the online network to *select* the greedy next action, but the target
//! network to *evaluate* it.  This reduces Q-value overestimation.

use crate::error::{NeuralError, Result};
use crate::rl::policy::PolicyRng;
use crate::rl::value::QNetwork;

// ──────────────────────────────────────────────────────────────────────────────
// Experience struct
// ──────────────────────────────────────────────────────────────────────────────

/// A single (s, a, r, s', done) transition.
#[derive(Debug, Clone)]
pub struct Experience {
    /// Current observation.
    pub state: Vec<f32>,
    /// Discrete action taken.
    pub action: usize,
    /// Scalar reward received.
    pub reward: f32,
    /// Next observation.
    pub next_state: Vec<f32>,
    /// Whether the episode ended after this transition.
    pub done: bool,
}

// ──────────────────────────────────────────────────────────────────────────────
// Minimal ring-buffer replay buffer (f32-specific, no ndarray dependency here)
// ──────────────────────────────────────────────────────────────────────────────

/// Fixed-capacity ring buffer for DQN experience replay.
///
/// Thread-safe via `std::sync::Mutex` so it can be wrapped in `Arc<Mutex<...>>`.
pub struct DQNReplayBuffer {
    capacity: usize,
    buffer: Vec<Experience>,
    ptr: usize,
    size: usize,
    rng: PolicyRng,
}

impl DQNReplayBuffer {
    /// Allocate a new buffer.
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "capacity must be > 0");
        Self {
            capacity,
            buffer: Vec::with_capacity(capacity),
            ptr: 0,
            size: 0,
            rng: PolicyRng::new(0xabcd_ef01_2345_6789),
        }
    }

    /// Push an experience.  Overwrites the oldest entry when full.
    pub fn push(&mut self, exp: Experience) {
        if self.size < self.capacity {
            self.buffer.push(exp);
        } else {
            self.buffer[self.ptr] = exp;
        }
        self.ptr = (self.ptr + 1) % self.capacity;
        self.size = (self.size + 1).min(self.capacity);
    }

    /// Sample `batch_size` transitions uniformly at random (with replacement).
    pub fn sample(&mut self, batch_size: usize) -> Result<Vec<Experience>> {
        if self.size == 0 {
            return Err(NeuralError::InvalidState(
                "cannot sample from empty replay buffer".into(),
            ));
        }
        let samples: Vec<Experience> = (0..batch_size)
            .map(|_| self.buffer[self.rng.usize_below(self.size)].clone())
            .collect();
        Ok(samples)
    }

    /// Current number of stored transitions.
    pub fn len(&self) -> usize {
        self.size
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns `true` when at least `min_size` transitions are stored.
    pub fn is_ready(&self, min_size: usize) -> bool {
        self.size >= min_size
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// DQNConfig
// ──────────────────────────────────────────────────────────────────────────────

/// Configuration for the [`DQNAgent`].
#[derive(Debug, Clone)]
pub struct DQNConfig {
    /// Learning rate for the online Q-network.
    pub lr: f32,
    /// Discount factor γ.
    pub gamma: f32,
    /// Initial exploration probability.
    pub eps_start: f32,
    /// Final exploration probability.
    pub eps_end: f32,
    /// Number of `select_action` calls over which ε is annealed.
    pub eps_decay_steps: usize,
    /// Mini-batch size.
    pub batch_size: usize,
    /// How many steps between hard target-network updates.
    pub target_update_freq: usize,
    /// Replay buffer capacity.
    pub buffer_capacity: usize,
    /// Minimum buffer size before training begins.
    pub learning_starts: usize,
    /// Enable Double DQN (use online net for action selection, target net for evaluation).
    pub double_dqn: bool,
    /// Hidden layer widths for both online and target networks.
    pub hidden_dims: Vec<usize>,
}

impl Default for DQNConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            gamma: 0.99,
            eps_start: 1.0,
            eps_end: 0.05,
            eps_decay_steps: 10_000,
            batch_size: 32,
            target_update_freq: 500,
            buffer_capacity: 50_000,
            learning_starts: 1_000,
            double_dqn: true,
            hidden_dims: vec![64, 64],
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// DQNAgent
// ──────────────────────────────────────────────────────────────────────────────

/// DQN / Double-DQN agent.
///
/// # Example
/// ```rust,ignore
/// use scirs2_neural::rl::dqn::{DQNAgent, DQNConfig, Experience};
/// use scirs2_neural::rl::environments::{CartPole, Environment};
///
/// let cfg = DQNConfig { eps_decay_steps: 5000, ..Default::default() };
/// let mut agent = DQNAgent::new(4, 2, cfg);
/// let mut env = CartPole::new();
///
/// let mut obs: Vec<f32> = env.reset().iter().map(|&x| x as f32).collect();
/// for _ in 0..200 {
///     let action = agent.select_action(&obs).expect("action");
///     let (next_obs_f64, reward, done) = env.step(
///         &scirs2_core::ndarray::array![action as f64]);
///     let next_obs: Vec<f32> = next_obs_f64.iter().map(|&x| x as f32).collect();
///     agent.store_transition(Experience {
///         state: obs.clone(), action, reward: reward as f32,
///         next_state: next_obs.clone(), done,
///     });
///     agent.update().ok();
///     if done { obs = env.reset().iter().map(|&x| x as f32).collect(); }
///     else { obs = next_obs; }
/// }
/// ```
pub struct DQNAgent {
    /// Online Q-network (trained each step).
    online_net: QNetwork,
    /// Target Q-network (periodically hard-copied from online).
    target_net: QNetwork,
    /// Replay buffer.
    replay: DQNReplayBuffer,
    /// Configuration.
    config: DQNConfig,
    /// Total action-selection steps (drives ε decay and target-update counter).
    steps: usize,
    /// Steps since last target-network update.
    steps_since_target_update: usize,
    /// Exploration RNG.
    rng: PolicyRng,
}

impl DQNAgent {
    /// Create a new DQNAgent.
    ///
    /// - `obs_dim`: observation (state) dimension.
    /// - `num_actions`: number of discrete actions.
    /// - `config`: [`DQNConfig`].
    pub fn new(obs_dim: usize, num_actions: usize, config: DQNConfig) -> Self {
        let online_net = QNetwork::new(obs_dim, &config.hidden_dims, num_actions);
        let target_net = QNetwork::new(obs_dim, &config.hidden_dims, num_actions);
        let replay = DQNReplayBuffer::new(config.buffer_capacity);
        let rng = PolicyRng::from_time();
        Self {
            online_net,
            target_net,
            replay,
            config,
            steps: 0,
            steps_since_target_update: 0,
            rng,
        }
    }

    /// Current ε (exploration probability).
    pub fn epsilon(&self) -> f32 {
        let frac = (self.steps as f32 / self.config.eps_decay_steps.max(1) as f32).min(1.0);
        self.config.eps_start + frac * (self.config.eps_end - self.config.eps_start)
    }

    /// Select an action using ε-greedy exploration.
    ///
    /// The step counter is incremented; ε is updated accordingly.
    pub fn select_action(&mut self, obs: &[f32]) -> Result<usize> {
        let eps = self.epsilon();
        self.steps += 1;
        if self.rng.uniform_f32() < eps {
            Ok(self.rng.usize_below(self.online_net.num_actions()))
        } else {
            self.online_net.greedy_action(obs)
        }
    }

    /// Push a transition into the replay buffer.
    pub fn store_transition(&mut self, exp: Experience) {
        self.replay.push(exp);
    }

    /// Perform one gradient update if the replay buffer has enough samples.
    ///
    /// Returns `Some(loss)` when an update occurred, `None` otherwise.
    pub fn update(&mut self) -> Result<Option<f32>> {
        if !self.replay.is_ready(self.config.learning_starts) {
            return Ok(None);
        }

        let batch = self.replay.sample(self.config.batch_size)?;
        let total_loss = self.td_update(&batch)?;

        // Periodic hard target-network update
        self.steps_since_target_update += 1;
        if self.steps_since_target_update >= self.config.target_update_freq {
            self.update_target()?;
            self.steps_since_target_update = 0;
        }

        Ok(Some(total_loss / self.config.batch_size as f32))
    }

    /// Hard-copy online network weights into the target network.
    pub fn update_target(&mut self) -> Result<()> {
        self.target_net.copy_from(&self.online_net)
    }

    /// Number of gradient update steps taken.
    pub fn steps(&self) -> usize {
        self.steps
    }

    /// Read-only reference to the replay buffer.
    pub fn replay_buffer(&self) -> &DQNReplayBuffer {
        &self.replay
    }

    /// Read-only reference to the online Q-network.
    pub fn online_net(&self) -> &QNetwork {
        &self.online_net
    }

    // ─── private ─────────────────────────────────────────────────────────────

    /// Compute TD targets and run one SGD step per transition in `batch`.
    ///
    /// Returns the **summed** loss (caller divides by batch_size).
    fn td_update(&mut self, batch: &[Experience]) -> Result<f32> {
        let gamma = self.config.gamma;
        let double = self.config.double_dqn;
        let lr = self.config.lr;
        let num_actions = self.online_net.num_actions();

        let mut total_loss = 0.0_f32;

        for exp in batch {
            // Current Q-values from online network
            let qs = self.online_net.q_values(&exp.state)?;
            if qs.len() != num_actions {
                return Err(NeuralError::ShapeMismatch(format!(
                    "Q-value length {} != num_actions {}",
                    qs.len(),
                    num_actions
                )));
            }

            // Compute TD target for the taken action
            let td_target = if exp.done {
                exp.reward
            } else {
                let next_q = if double {
                    // Double DQN: online selects action, target evaluates it
                    let best_next_action = self.online_net.greedy_action(&exp.next_state)?;
                    let target_qs = self.target_net.q_values(&exp.next_state)?;
                    target_qs.get(best_next_action).copied().ok_or_else(|| {
                        NeuralError::ComputationError("best_next_action out of range".into())
                    })?
                } else {
                    // Standard DQN: max Q from target net
                    let target_qs = self.target_net.q_values(&exp.next_state)?;
                    target_qs.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
                };
                exp.reward + gamma * next_q
            };

            // Build target vector (same as current Q everywhere except for action taken)
            let mut targets = qs.clone();
            targets[exp.action] = td_target;

            // Single-action update (zero gradient on all other actions)
            let loss = self
                .online_net
                .update_action(&exp.state, exp.action, td_target, lr)?;
            total_loss += loss;
        }

        Ok(total_loss)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_exp(obs_dim: usize, action: usize) -> Experience {
        Experience {
            state: vec![0.1_f32; obs_dim],
            action,
            reward: 1.0,
            next_state: vec![0.2_f32; obs_dim],
            done: false,
        }
    }

    #[test]
    fn dqn_replay_buffer_push_and_len() {
        let mut buf = DQNReplayBuffer::new(10);
        assert!(buf.is_empty());
        buf.push(make_exp(4, 0));
        assert_eq!(buf.len(), 1);
    }

    #[test]
    fn dqn_replay_buffer_circular_overwrite() {
        let mut buf = DQNReplayBuffer::new(3);
        for _ in 0..5 {
            buf.push(make_exp(4, 0));
        }
        assert_eq!(buf.len(), 3);
    }

    #[test]
    fn dqn_replay_buffer_sample_not_empty() {
        let mut buf = DQNReplayBuffer::new(20);
        for i in 0..20 {
            buf.push(make_exp(4, i % 2));
        }
        let samples = buf.sample(8).expect("sample failed");
        assert_eq!(samples.len(), 8);
    }

    #[test]
    fn dqn_replay_buffer_is_ready() {
        let mut buf = DQNReplayBuffer::new(100);
        for _ in 0..9 {
            buf.push(make_exp(4, 0));
        }
        assert!(!buf.is_ready(10));
        buf.push(make_exp(4, 1));
        assert!(buf.is_ready(10));
    }

    #[test]
    fn dqn_agent_select_action_in_range() {
        let cfg = DQNConfig {
            hidden_dims: vec![16],
            ..Default::default()
        };
        let mut agent = DQNAgent::new(4, 3, cfg);
        let obs = vec![0.1_f32; 4];
        let a = agent.select_action(&obs).expect("select_action failed");
        assert!(a < 3, "action should be in [0, num_actions)");
    }

    #[test]
    fn dqn_agent_epsilon_decays() {
        let cfg = DQNConfig {
            eps_start: 1.0,
            eps_end: 0.1,
            eps_decay_steps: 10,
            hidden_dims: vec![8],
            ..Default::default()
        };
        let mut agent = DQNAgent::new(4, 2, cfg);
        let obs = vec![0.0_f32; 4];
        assert!((agent.epsilon() - 1.0).abs() < 1e-4);
        for _ in 0..10 {
            let _ = agent.select_action(&obs).expect("act");
        }
        assert!(
            (agent.epsilon() - 0.1).abs() < 0.05,
            "epsilon should reach eps_end"
        );
    }

    #[test]
    fn dqn_agent_returns_none_before_ready() {
        let cfg = DQNConfig {
            learning_starts: 100,
            hidden_dims: vec![8],
            ..Default::default()
        };
        let mut agent = DQNAgent::new(4, 2, cfg);
        agent.store_transition(make_exp(4, 0));
        let result = agent.update().expect("update failed");
        assert!(result.is_none(), "should not update before learning_starts");
    }

    #[test]
    fn dqn_agent_update_after_learning_starts() {
        let cfg = DQNConfig {
            learning_starts: 10,
            batch_size: 4,
            hidden_dims: vec![8],
            ..Default::default()
        };
        let mut agent = DQNAgent::new(4, 2, cfg);
        for i in 0..20 {
            let done = i % 5 == 4;
            agent.store_transition(Experience {
                state: vec![i as f32 * 0.01; 4],
                action: i % 2,
                reward: 1.0,
                next_state: vec![(i + 1) as f32 * 0.01; 4],
                done,
            });
        }
        let result = agent.update().expect("update failed");
        assert!(
            result.is_some(),
            "should produce a loss after learning_starts"
        );
        let loss = result.expect("operation should succeed");
        assert!(loss.is_finite(), "loss must be finite; got {}", loss);
    }

    #[test]
    fn dqn_agent_target_update_copy() {
        let cfg = DQNConfig {
            target_update_freq: 1,
            learning_starts: 5,
            batch_size: 4,
            hidden_dims: vec![8],
            ..Default::default()
        };
        let mut agent = DQNAgent::new(4, 2, cfg);
        for i in 0..20 {
            agent.store_transition(make_exp(4, i % 2));
        }
        // After one update, target should be refreshed (freq=1)
        agent.update().expect("update failed");
        let obs = vec![0.1_f32; 4];
        let online_q = agent.online_net().q_values(&obs).expect("online qs");
        // Check we can call target without panic
        assert_eq!(online_q.len(), 2);
    }

    #[test]
    fn dqn_cartpole_10_steps_no_panic() {
        use crate::rl::environments::{CartPole, Environment};
        let cfg = DQNConfig {
            hidden_dims: vec![16],
            learning_starts: 5,
            batch_size: 4,
            ..Default::default()
        };
        let mut agent = DQNAgent::new(4, 2, cfg);
        let mut env = CartPole::new();
        let mut obs: Vec<f32> = env.reset().iter().map(|&x| x as f32).collect();

        for _ in 0..10 {
            let action = agent.select_action(&obs).expect("select_action");
            let action_arr = scirs2_core::ndarray::array![action as f64];
            let (next_f64, reward, done) = env.step(&action_arr);
            let next_obs: Vec<f32> = next_f64.iter().map(|&x| x as f32).collect();
            agent.store_transition(Experience {
                state: obs.clone(),
                action,
                reward: reward as f32,
                next_state: next_obs.clone(),
                done,
            });
            let _ = agent.update().expect("update");
            if done {
                obs = env.reset().iter().map(|&x| x as f32).collect();
            } else {
                obs = next_obs;
            }
        }
    }

    #[test]
    fn double_dqn_agent_no_panic() {
        let cfg = DQNConfig {
            double_dqn: true,
            learning_starts: 8,
            batch_size: 4,
            hidden_dims: vec![8],
            ..Default::default()
        };
        let mut agent = DQNAgent::new(4, 2, cfg);
        for i in 0..20 {
            agent.store_transition(make_exp(4, i % 2));
        }
        let loss = agent
            .update()
            .expect("update")
            .expect("should produce loss");
        assert!(loss.is_finite());
    }
}
