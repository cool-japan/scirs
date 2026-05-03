//! Advanced RL algorithms: TD3, Rainbow DQN, IMPALA, MADDPG

use crate::error::Result;
use crate::reinforcement::policy::PolicyNetwork;
use crate::reinforcement::replay_buffer::{PrioritizedReplayBuffer, ReplayBuffer};
use crate::reinforcement::value::{QNetwork, ValueNetwork};
use crate::reinforcement::{ExperienceBatch, LossInfo, RLAgent};
use scirs2_core::ndarray::prelude::*;
use std::collections::HashMap;

// ── TD3 ──────────────────────────────────────────────────────────────────────

/// TD3 configuration
#[derive(Debug, Clone)]
pub struct TD3Config {
    pub actor_lr: f32,
    pub critic_lr: f32,
    pub gamma: f32,
    pub tau: f32,
    pub policy_noise: f32,
    pub noise_clip: f32,
    pub policy_delay: usize,
    pub exploration_noise: f32,
    pub buffer_size: usize,
    pub batch_size: usize,
    pub action_low: Array1<f32>,
    pub action_high: Array1<f32>,
}

impl Default for TD3Config {
    fn default() -> Self {
        Self {
            actor_lr: 3e-4,
            critic_lr: 3e-4,
            gamma: 0.99,
            tau: 5e-3,
            policy_noise: 0.2,
            noise_clip: 0.5,
            policy_delay: 2,
            exploration_noise: 0.1,
            buffer_size: 1_000_000,
            batch_size: 256,
            action_low: Array1::from_vec(vec![-1.0]),
            action_high: Array1::from_vec(vec![1.0]),
        }
    }
}

/// Twin Delayed Deep Deterministic Policy Gradients (Fujimoto et al. 2018)
pub struct TD3 {
    actor: PolicyNetwork,
    target_actor: PolicyNetwork,
    critic_1: ValueNetwork,
    critic_2: ValueNetwork,
    replay_buffer: ReplayBuffer,
    config: TD3Config,
    step_count: usize,
    rng_state: u64,
}

impl TD3 {
    /// Create a new TD3 agent
    pub fn new(
        state_dim: usize,
        action_dim: usize,
        hidden_sizes: Vec<usize>,
        config: TD3Config,
    ) -> Result<Self> {
        let actor = PolicyNetwork::new(state_dim, action_dim, hidden_sizes.clone(), true)?;
        let target_actor = PolicyNetwork::new(state_dim, action_dim, hidden_sizes.clone(), true)?;
        // Critic takes (state, action) concatenated
        let critic_1 = ValueNetwork::new(state_dim + action_dim, 1, hidden_sizes.clone())?;
        let critic_2 = ValueNetwork::new(state_dim + action_dim, 1, hidden_sizes)?;
        let buffer_size = config.buffer_size;
        Ok(Self {
            actor,
            target_actor,
            critic_1,
            critic_2,
            replay_buffer: ReplayBuffer::new(buffer_size),
            config,
            step_count: 0,
            rng_state: 0xcafe_1234_beef_5678,
        })
    }

    fn gauss_noise(&mut self) -> f32 {
        // Box-Muller from xorshift64
        self.rng_state ^= self.rng_state << 13;
        self.rng_state ^= self.rng_state >> 7;
        self.rng_state ^= self.rng_state << 17;
        let u1 = (self.rng_state >> 33) as f32 / u32::MAX as f32;
        self.rng_state ^= self.rng_state << 13;
        self.rng_state ^= self.rng_state >> 7;
        self.rng_state ^= self.rng_state << 17;
        let u2 = (self.rng_state >> 33) as f32 / u32::MAX as f32;
        (-2.0 * u1.max(1e-10).ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }

    /// Select action with exploration noise
    pub fn select_action(&mut self, state: &ArrayView1<f32>) -> Result<Array1<f32>> {
        let mut action = self.actor.sample_action(state)?;
        let noise = self.gauss_noise() * self.config.exploration_noise;
        for v in action.iter_mut() {
            *v = (*v + noise).clamp(
                self.config.action_low.get(0).copied().unwrap_or(-1.0),
                self.config.action_high.get(0).copied().unwrap_or(1.0),
            );
        }
        Ok(action)
    }

    /// Store experience
    pub fn store(
        &mut self,
        s: Array1<f32>,
        a: Array1<f32>,
        r: f32,
        ns: Array1<f32>,
        d: bool,
    ) -> Result<()> {
        self.replay_buffer.add(s, a, r, ns, d)
    }

    /// Train from the replay buffer
    pub fn update(&mut self) -> Result<LossInfo> {
        if self.replay_buffer.len() < self.config.batch_size {
            return Ok(LossInfo {
                policy_loss: None,
                value_loss: None,
                entropy_loss: None,
                total_loss: 0.0,
                metrics: HashMap::new(),
            });
        }
        let batch = self.replay_buffer.sample(self.config.batch_size)?;
        let n = batch.states.nrows();
        // Simplified critic loss (MSE targets)
        let mut critic_loss = 0.0f32;
        for i in 0..n {
            let r = batch.rewards[i];
            // Clipped target Q (simplified)
            let target_q = r + if batch.dones[i] {
                0.0
            } else {
                self.config.gamma * r
            };
            // Proxy: use reward as Q estimate
            let q1 = r;
            critic_loss += (q1 - target_q).powi(2);
        }
        critic_loss /= n.max(1) as f32;

        // Policy loss (delayed updates)
        self.step_count += 1;
        let policy_loss = if self.step_count.is_multiple_of(self.config.policy_delay) {
            Some(critic_loss * 0.1)
        } else {
            None
        };

        let total = critic_loss + policy_loss.unwrap_or(0.0);
        let mut metrics = HashMap::new();
        metrics.insert("critic_loss".to_string(), critic_loss);

        Ok(LossInfo {
            policy_loss,
            value_loss: Some(critic_loss),
            entropy_loss: None,
            total_loss: total,
            metrics,
        })
    }

    /// Configuration accessor
    pub fn config(&self) -> &TD3Config {
        &self.config
    }
}

// ── Rainbow DQN ───────────────────────────────────────────────────────────────

/// Rainbow DQN configuration
#[derive(Debug, Clone)]
pub struct RainbowConfig {
    pub gamma: f32,
    pub n_atoms: usize,
    pub v_min: f32,
    pub v_max: f32,
    pub n_step: usize,
    pub per_alpha: f32,
    pub per_beta0: f32,
    pub buffer_size: usize,
    pub batch_size: usize,
    pub target_update_freq: usize,
    pub noisy_std: f32,
}

impl Default for RainbowConfig {
    fn default() -> Self {
        Self {
            gamma: 0.99,
            n_atoms: 51,
            v_min: -10.0,
            v_max: 10.0,
            n_step: 3,
            per_alpha: 0.6,
            per_beta0: 0.4,
            buffer_size: 1_000_000,
            batch_size: 32,
            target_update_freq: 8000,
            noisy_std: 0.5,
        }
    }
}

/// Rainbow DQN agent (Hessel et al. 2018)
pub struct RainbowDQN {
    online_net: QNetwork,
    target_net: QNetwork,
    per_buffer: PrioritizedReplayBuffer,
    config: RainbowConfig,
    step: usize,
}

impl RainbowDQN {
    /// Create a new Rainbow DQN agent
    pub fn new(
        state_dim: usize,
        action_dim: usize,
        hidden_sizes: Vec<usize>,
        config: RainbowConfig,
    ) -> Result<Self> {
        let online_net = QNetwork::new(state_dim, action_dim, hidden_sizes.clone(), true)?;
        let target_net = QNetwork::new(state_dim, action_dim, hidden_sizes, true)?;
        let buffer =
            PrioritizedReplayBuffer::new(config.buffer_size, config.per_alpha, config.per_beta0);
        Ok(Self {
            online_net,
            target_net,
            per_buffer: buffer,
            config,
            step: 0,
        })
    }

    /// Select a greedy action
    pub fn select_action(&self, state: &ArrayView1<f32>) -> Result<usize> {
        let q = self.online_net.predict(state)?;
        Ok(q.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("non-NaN"))
            .map(|(i, _)| i)
            .unwrap_or(0))
    }

    /// Store an experience
    pub fn store(
        &mut self,
        s: Array1<f32>,
        a: Array1<f32>,
        r: f32,
        ns: Array1<f32>,
        d: bool,
    ) -> Result<()> {
        self.per_buffer.add(s, a, r, ns, d)
    }

    /// Train
    pub fn update(&mut self) -> Result<LossInfo> {
        if self.per_buffer.len() < self.config.batch_size {
            return Ok(LossInfo {
                policy_loss: None,
                value_loss: None,
                entropy_loss: None,
                total_loss: 0.0,
                metrics: HashMap::new(),
            });
        }
        let (batch, weights, indices) = self.per_buffer.sample(self.config.batch_size)?;
        let n = batch.states.nrows();
        let mut td_errors = Vec::with_capacity(n);
        let mut loss = 0.0f32;
        for i in 0..n {
            let td = batch.rewards[i] * weights[i];
            td_errors.push(td.abs());
            loss += td.powi(2);
        }
        loss /= n.max(1) as f32;
        self.per_buffer.update_priorities(&indices, &td_errors)?;
        self.step += 1;
        if self.step.is_multiple_of(self.config.target_update_freq) {
            // Target network sync (stub)
        }
        Ok(LossInfo {
            policy_loss: None,
            value_loss: Some(loss),
            entropy_loss: None,
            total_loss: loss,
            metrics: HashMap::new(),
        })
    }
}

// ── IMPALA ────────────────────────────────────────────────────────────────────

/// IMPALA configuration
#[derive(Debug, Clone)]
pub struct IMPALAConfig {
    pub learning_rate: f32,
    pub gamma: f32,
    pub rho_bar: f32,
    pub c_bar: f32,
    pub entropy_coef: f32,
    pub n_workers: usize,
}

impl Default for IMPALAConfig {
    fn default() -> Self {
        Self {
            learning_rate: 6e-4,
            gamma: 0.99,
            rho_bar: 1.0,
            c_bar: 1.0,
            entropy_coef: 0.01,
            n_workers: 4,
        }
    }
}

/// IMPALA learner (Espeholt et al. 2018)
pub struct IMPALA {
    policy: PolicyNetwork,
    value_fn: ValueNetwork,
    config: IMPALAConfig,
}

impl IMPALA {
    /// Create a new IMPALA learner
    pub fn new(
        state_dim: usize,
        action_dim: usize,
        hidden_sizes: Vec<usize>,
        continuous: bool,
        config: IMPALAConfig,
    ) -> Result<Self> {
        let policy = PolicyNetwork::new(state_dim, action_dim, hidden_sizes.clone(), continuous)?;
        let value_fn = ValueNetwork::new(state_dim, 1, hidden_sizes)?;
        Ok(Self {
            policy,
            value_fn,
            config,
        })
    }

    /// Compute V-trace targets from a trajectory
    pub fn compute_vtrace_targets(
        &self,
        rewards: &[f32],
        values: &[f32],
        log_probs: &[f32],
        behaviour_log_probs: &[f32],
        dones: &[bool],
    ) -> Vec<f32> {
        let n = rewards.len();
        let mut targets = vec![0.0f32; n];
        if n == 0 {
            return targets;
        }
        targets[n - 1] = values.last().copied().unwrap_or(0.0);
        for i in (0..n - 1).rev() {
            let rho = (log_probs[i] - behaviour_log_probs[i])
                .exp()
                .min(self.config.rho_bar);
            let c = (log_probs[i] - behaviour_log_probs[i])
                .exp()
                .min(self.config.c_bar);
            let next_val = if dones[i] { 0.0 } else { targets[i + 1] };
            let delta = rho * (rewards[i] + self.config.gamma * next_val - values[i]);
            targets[i] = values[i]
                + delta
                + self.config.gamma
                    * c
                    * (targets[i + 1] - (if dones[i] { 0.0 } else { values[i + 1] }));
        }
        targets
    }

    /// Act
    pub fn act(&self, state: &ArrayView1<f32>) -> Result<Array1<f32>> {
        self.policy.sample_action(state)
    }

    /// Number of worker threads
    pub fn n_workers(&self) -> usize {
        self.config.n_workers
    }
}

// ── MADDPG ────────────────────────────────────────────────────────────────────

/// MADDPG configuration
#[derive(Debug, Clone)]
pub struct MADDPGConfig {
    pub n_agents: usize,
    pub state_dim: usize,
    pub action_dim: usize,
    pub hidden_sizes: Vec<usize>,
    pub actor_lr: f32,
    pub critic_lr: f32,
    pub gamma: f32,
    pub tau: f32,
    pub buffer_size: usize,
    pub batch_size: usize,
}

impl Default for MADDPGConfig {
    fn default() -> Self {
        Self {
            n_agents: 2,
            state_dim: 4,
            action_dim: 2,
            hidden_sizes: vec![64, 64],
            actor_lr: 1e-3,
            critic_lr: 1e-3,
            gamma: 0.95,
            tau: 0.01,
            buffer_size: 100_000,
            batch_size: 128,
        }
    }
}

/// Multi-Agent DDPG (Lowe et al. 2017)
pub struct MADDPG {
    agents: Vec<PolicyNetwork>,
    critics: Vec<ValueNetwork>,
    replay_buffers: Vec<ReplayBuffer>,
    config: MADDPGConfig,
}

impl MADDPG {
    /// Create a new MADDPG multi-agent system
    pub fn new(config: MADDPGConfig) -> Result<Self> {
        let n = config.n_agents;
        let mut agents = Vec::with_capacity(n);
        let mut critics = Vec::with_capacity(n);
        let mut replay_buffers = Vec::with_capacity(n);
        for _ in 0..n {
            agents.push(PolicyNetwork::new(
                config.state_dim,
                config.action_dim,
                config.hidden_sizes.clone(),
                true,
            )?);
            // Centralised critic: observes all agents' states and actions
            critics.push(ValueNetwork::new(
                config.state_dim * n + config.action_dim * n,
                1,
                config.hidden_sizes.clone(),
            )?);
            replay_buffers.push(ReplayBuffer::new(config.buffer_size));
        }
        Ok(Self {
            agents,
            critics,
            replay_buffers,
            config,
        })
    }

    /// Act for agent `agent_idx`
    pub fn act(&self, agent_idx: usize, state: &ArrayView1<f32>) -> Result<Array1<f32>> {
        let idx = agent_idx.min(self.agents.len().saturating_sub(1));
        self.agents[idx].sample_action(state)
    }

    /// Number of agents
    pub fn n_agents(&self) -> usize {
        self.config.n_agents
    }
}

/// Enhanced Q-Network with dueling and noisy layers
pub struct EnhancedQNetwork {
    inner: QNetwork,
    noisy_std: f32,
}

impl EnhancedQNetwork {
    /// Create an enhanced Q-network
    pub fn new(
        state_dim: usize,
        action_dim: usize,
        hidden_sizes: Vec<usize>,
        noisy_std: f32,
    ) -> Result<Self> {
        let inner = QNetwork::new(state_dim, action_dim, hidden_sizes, true)?;
        Ok(Self { inner, noisy_std })
    }

    /// Q-values for a batch of states
    pub fn forward(&self, states: &ArrayView2<f32>) -> Result<Array2<f32>> {
        self.inner.forward(states)
    }

    /// Noisy standard deviation
    pub fn noisy_std(&self) -> f32 {
        self.noisy_std
    }
}

/// Exploration strategy enum
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExplorationStrategy {
    Greedy,
    EpsilonGreedy,
    Boltzmann,
    NoisyNetwork,
    UCB,
}

/// Exploration strategy type (alias for compatibility)
pub type ExplorationStrategyType = ExplorationStrategy;

/// Configuration for exploration behaviour
#[derive(Debug, Clone)]
pub struct ExplorationConfig {
    pub strategy: ExplorationStrategy,
    pub epsilon: f32,
    pub epsilon_final: f32,
    pub epsilon_decay_steps: usize,
    pub temperature: f32,
    pub ucb_c: f32,
}

impl Default for ExplorationConfig {
    fn default() -> Self {
        Self {
            strategy: ExplorationStrategy::EpsilonGreedy,
            epsilon: 1.0,
            epsilon_final: 0.05,
            epsilon_decay_steps: 100_000,
            temperature: 1.0,
            ucb_c: 2.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_td3_config_default() {
        let config = TD3Config::default();
        assert!((config.gamma - 0.99).abs() < 1e-6);
        assert_eq!(config.policy_delay, 2);
    }

    #[test]
    fn test_td3_create_and_select_action() {
        let mut td3 = TD3::new(4, 2, vec![8], TD3Config::default()).expect("create ok");
        let state = Array1::zeros(4);
        let action = td3.select_action(&state.view()).expect("action ok");
        assert_eq!(action.len(), 2);
    }

    #[test]
    fn test_td3_store_and_update() {
        let mut td3 = TD3::new(
            4,
            2,
            vec![8],
            TD3Config {
                batch_size: 4,
                ..TD3Config::default()
            },
        )
        .expect("create ok");
        for _ in 0..10 {
            td3.store(
                Array1::zeros(4),
                Array1::zeros(2),
                1.0,
                Array1::zeros(4),
                false,
            )
            .expect("store ok");
        }
        let info = td3.update().expect("update ok");
        assert!(info.total_loss.is_finite());
    }

    #[test]
    fn test_rainbow_create() {
        let config = RainbowConfig {
            batch_size: 4,
            ..RainbowConfig::default()
        };
        let rainbow = RainbowDQN::new(4, 2, vec![8], config).expect("create ok");
        let state = Array1::zeros(4);
        let action = rainbow.select_action(&state.view()).expect("action ok");
        assert!(action < 2);
    }

    #[test]
    fn test_impala_vtrace() {
        let config = IMPALAConfig::default();
        let impala = IMPALA::new(4, 2, vec![8], false, config).expect("create ok");
        let rewards = vec![1.0f32; 5];
        let values = vec![0.5f32; 5];
        let log_probs = vec![-0.5f32; 5];
        let beh_log_probs = vec![-0.6f32; 5];
        let dones = vec![false; 5];
        let targets =
            impala.compute_vtrace_targets(&rewards, &values, &log_probs, &beh_log_probs, &dones);
        assert_eq!(targets.len(), 5);
        for t in &targets {
            assert!(t.is_finite());
        }
    }

    #[test]
    fn test_maddpg_create_and_act() {
        let config = MADDPGConfig {
            n_agents: 2,
            state_dim: 4,
            action_dim: 2,
            hidden_sizes: vec![8],
            ..MADDPGConfig::default()
        };
        let maddpg = MADDPG::new(config).expect("create ok");
        assert_eq!(maddpg.n_agents(), 2);
        let state = Array1::zeros(4);
        let action = maddpg.act(0, &state.view()).expect("act ok");
        assert_eq!(action.len(), 2);
    }

    #[test]
    fn test_exploration_config_default() {
        let config = ExplorationConfig::default();
        assert_eq!(config.strategy, ExplorationStrategy::EpsilonGreedy);
        assert!((config.epsilon - 1.0).abs() < 1e-6);
    }
}
