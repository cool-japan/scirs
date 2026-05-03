//! # Reinforcement Learning (`rl`)
//!
//! Self-contained deep RL implementations for both discrete and continuous action spaces.
//!
//! ## Sub-modules
//!
//! | Module | Description |
//! |--------|-------------|
//! | [`environments`] | [`Environment`] trait + `CartPole`, `PendulumEnv`, `GridWorld` |
//! | [`replay_buffer`] | Fixed-capacity circular replay buffer |
//! | [`policy`] | [`Policy`] trait, [`SimpleNetwork`], [`EpsilonGreedy`], [`BoltzmannPolicy`] |
//! | [`value`] | [`ValueNetwork`], [`QNetwork`], [`DuelingQNetwork`] |
//! | [`dqn`] | DQN / Double-DQN agent with ε-greedy exploration |
//! | [`actor_critic`] | A2C: synchronous Advantage Actor-Critic |
//! | [`ppo`] | Proximal Policy Optimisation (clipped surrogate + GAE) |
//! | [`sac`] | Soft Actor-Critic (maximum-entropy, off-policy) |

pub mod actor_critic;
pub mod dqn;
pub mod environments;
pub mod policy;
pub mod ppo;
pub mod replay_buffer;
pub mod sac;
pub mod value;

// ── Environments ─────────────────────────────────────────────────────────────

pub use environments::{
    ActionSpace, CartPole, ContinuousCartPole, Environment, GridCell, GridWorld, ObservationSpace,
    PendulumEnv,
};

// ── Policy ───────────────────────────────────────────────────────────────────

pub use policy::{
    categorical_sample, softmax, softmax_temperature, BoltzmannPolicy, EpsilonGreedy, Policy,
    PolicyRng, SimpleNetwork,
};

// ── Value networks ───────────────────────────────────────────────────────────

pub use value::{
    ActionValuePolicy, DuelingQNetwork, NetworkUpdate, QNetwork, SoftmaxValuePolicy, ValueNetwork,
};

// ── DQN ──────────────────────────────────────────────────────────────────────

pub use dqn::{DQNAgent, DQNConfig, DQNReplayBuffer, Experience};

// ── A2C ──────────────────────────────────────────────────────────────────────

pub use actor_critic::{
    run_episode, A2CAgent, A2CConfig, A2CTrainInfo, ActorNetwork, CriticNetwork,
};

// ── PPO ──────────────────────────────────────────────────────────────────────

pub use ppo::{ActorCritic, PPOConfig, PPOInfo, RolloutBuffer, PPO};

// ── SAC ──────────────────────────────────────────────────────────────────────

pub use sac::{Critic, SACConfig, SACInfo, StochasticActor, SAC};

// ── Replay buffer ────────────────────────────────────────────────────────────

pub use replay_buffer::{ReplayBuffer, Rng, Transition, XorShift64};
