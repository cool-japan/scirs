//! Reinforcement learning environment interfaces and built-in environments.
//!
//! This module defines the [`Environment`] trait and several classic environments:
//! - [`CartPole`]: discrete-action pole-balancing task
//! - [`GridWorld`]: simple grid navigation with configurable rewards
//! - [`ContinuousCartPole`]: continuous-action variant of CartPole

use crate::error::{NeuralError, Result};
use scirs2_core::ndarray::Array1;

// ──────────────────────────────────────────────────────────────────────────────
// Action / Observation spaces
// ──────────────────────────────────────────────────────────────────────────────

/// Description of the action space of an environment.
#[derive(Debug, Clone, PartialEq)]
pub enum ActionSpace {
    /// Finite set of `n` discrete actions (integers 0 … n-1).
    Discrete { n: usize },
    /// Box of continuous actions with per-dimension bounds.
    Continuous { low: Vec<f64>, high: Vec<f64> },
}

impl ActionSpace {
    /// Number of dimensions.  For `Discrete` this is 1 (the action index).
    #[inline]
    pub fn ndim(&self) -> usize {
        match self {
            ActionSpace::Discrete { .. } => 1,
            ActionSpace::Continuous { low, .. } => low.len(),
        }
    }

    /// Clip a continuous action to the allowed box.  No-op for discrete.
    pub fn clip(&self, action: &Array1<f64>) -> Array1<f64> {
        match self {
            ActionSpace::Discrete { .. } => action.clone(),
            ActionSpace::Continuous { low, high } => {
                let mut out = action.clone();
                for (i, v) in out.iter_mut().enumerate() {
                    *v = v.clamp(low[i], high[i]);
                }
                out
            }
        }
    }
}

/// Description of the observation space of an environment.
#[derive(Debug, Clone, PartialEq)]
pub struct ObservationSpace {
    /// Dimensionality of the observation vector.
    pub ndim: usize,
    /// Optional per-dimension lower bounds.
    pub low: Option<Vec<f64>>,
    /// Optional per-dimension upper bounds.
    pub high: Option<Vec<f64>>,
}

impl ObservationSpace {
    /// Unbounded observation space of a given dimensionality.
    pub fn new(ndim: usize) -> Self {
        Self {
            ndim,
            low: None,
            high: None,
        }
    }

    /// Bounded observation space.
    pub fn bounded(low: Vec<f64>, high: Vec<f64>) -> Self {
        assert_eq!(
            low.len(),
            high.len(),
            "low and high must have the same length"
        );
        let ndim = low.len();
        Self {
            ndim,
            low: Some(low),
            high: Some(high),
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Environment trait
// ──────────────────────────────────────────────────────────────────────────────

/// Core trait every RL environment must implement.
///
/// `State` and `Action` are generic associated types so that discrete,
/// continuous, or structured observations/actions are all representable.
///
/// # Example
/// ```rust,ignore
/// use scirs2_neural::rl::environments::{CartPole, Environment};
/// let mut env = CartPole::new();
/// let state = env.reset();
/// let (next, reward, done) = env.step(&scirs2_core::ndarray::array![1.0_f64]);
/// ```
pub trait Environment {
    /// The state / observation type returned by the environment.
    type State;
    /// The action type accepted by the environment.
    type Action;

    /// Reset the environment and return the initial state.
    fn reset(&mut self) -> Self::State;

    /// Apply `action` and return `(next_state, reward, done)`.
    ///
    /// When `done` is `true` the caller should invoke [`reset`](Environment::reset)
    /// before the next episode.
    fn step(&mut self, action: &Self::Action) -> (Self::State, f64, bool);

    /// Description of the action space.
    fn action_space(&self) -> ActionSpace;

    /// Description of the observation space.
    fn observation_space(&self) -> ObservationSpace;

    /// Optional render / debug output.  Implementations may leave this as a no-op.
    fn render(&self) {}
}

// ──────────────────────────────────────────────────────────────────────────────
// Random helpers (tiny RNG not depending on the heavy rand crate)
// ──────────────────────────────────────────────────────────────────────────────

/// Minimal xorshift64 state for environment stochasticity.
struct XorShift64(u64);

impl XorShift64 {
    fn new(seed: u64) -> Self {
        let s = if seed == 0 { 6364136223846793005 } else { seed };
        Self(s)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// Sample from Uniform(low, high).
    fn uniform(&mut self, low: f64, high: f64) -> f64 {
        let bits = self.next_u64();
        let f = (bits >> 11) as f64 / (1u64 << 53) as f64;
        low + f * (high - low)
    }

    /// Sample from Uniform(-mag, mag).
    fn sym(&mut self, mag: f64) -> f64 {
        self.uniform(-mag, mag)
    }
}

/// Create a deterministic-but-environment-unique seed from the system nano clock.
fn make_seed() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.subsec_nanos() as u64 ^ (d.as_secs().wrapping_mul(6364136223846793005)))
        .unwrap_or(123456789)
}

// ──────────────────────────────────────────────────────────────────────────────
// CartPole (discrete actions)
// ──────────────────────────────────────────────────────────────────────────────

/// Classic pole-balancing task (OpenAI Gym–compatible semantics).
///
/// **State** (4-dim): `[cart_position, cart_velocity, pole_angle, pole_angular_velocity]`
///
/// **Actions** (discrete, 2): push left (0) or push right (1)
///
/// The episode ends when the pole falls beyond ±12° or the cart leaves ±2.4 m,
/// or after `max_steps` steps.
pub struct CartPole {
    state: Array1<f64>,
    steps: usize,
    max_steps: usize,
    gravity: f64,
    mass_cart: f64,
    mass_pole: f64,
    half_length: f64,
    force_mag: f64,
    tau: f64,
    rng: XorShift64,
}

impl Default for CartPole {
    fn default() -> Self {
        Self::new()
    }
}

impl CartPole {
    /// Create a new `CartPole` environment with standard parameters.
    pub fn new() -> Self {
        Self {
            state: Array1::zeros(4),
            steps: 0,
            max_steps: 500,
            gravity: 9.8,
            mass_cart: 1.0,
            mass_pole: 0.1,
            half_length: 0.5,
            force_mag: 10.0,
            tau: 0.02,
            rng: XorShift64::new(make_seed()),
        }
    }

    /// Set the maximum number of steps per episode.
    pub fn with_max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = max_steps;
        self
    }

    fn is_terminal(&self) -> bool {
        let x = self.state[0];
        let theta = self.state[2];
        x.abs() > 2.4 || theta.abs() > 12.0_f64.to_radians() || self.steps >= self.max_steps
    }

    /// Euler-integrate the physics for one timestep given a force.
    fn physics_step(&mut self, force: f64) {
        let x_dot = self.state[1];
        let theta = self.state[2];
        let theta_dot = self.state[3];

        let cos_t = theta.cos();
        let sin_t = theta.sin();
        let total_mass = self.mass_cart + self.mass_pole;
        let pole_ml = self.mass_pole * self.half_length;

        let temp = (force + pole_ml * theta_dot * theta_dot * sin_t) / total_mass;
        let theta_acc = (self.gravity * sin_t - cos_t * temp)
            / (self.half_length * (4.0 / 3.0 - self.mass_pole * cos_t * cos_t / total_mass));
        let x_acc = temp - pole_ml * theta_acc * cos_t / total_mass;

        self.state[0] += self.tau * x_dot;
        self.state[1] += self.tau * x_acc;
        self.state[2] += self.tau * theta_dot;
        self.state[3] += self.tau * theta_acc;
    }
}

impl Environment for CartPole {
    type State = Array1<f64>;
    type Action = Array1<f64>;

    fn reset(&mut self) -> Array1<f64> {
        self.state = Array1::from_vec(vec![
            self.rng.sym(0.05),
            self.rng.sym(0.05),
            self.rng.sym(0.05),
            self.rng.sym(0.05),
        ]);
        self.steps = 0;
        self.state.clone()
    }

    fn step(&mut self, action: &Array1<f64>) -> (Array1<f64>, f64, bool) {
        let force = if action[0] >= 0.5 {
            self.force_mag
        } else {
            -self.force_mag
        };
        self.physics_step(force);
        self.steps += 1;

        let done = self.is_terminal();
        let reward = if done { 0.0 } else { 1.0 };
        (self.state.clone(), reward, done)
    }

    fn action_space(&self) -> ActionSpace {
        ActionSpace::Discrete { n: 2 }
    }

    fn observation_space(&self) -> ObservationSpace {
        ObservationSpace::bounded(
            vec![
                -4.8,
                f64::NEG_INFINITY,
                -24.0_f64.to_radians(),
                f64::NEG_INFINITY,
            ],
            vec![4.8, f64::INFINITY, 24.0_f64.to_radians(), f64::INFINITY],
        )
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// ContinuousCartPole
// ──────────────────────────────────────────────────────────────────────────────

/// Continuous-action variant of `CartPole`.
///
/// The action is a single f64 in `[-force_mag, +force_mag]` applied directly
/// as a horizontal force on the cart.
pub struct ContinuousCartPole {
    inner: CartPole,
}

impl Default for ContinuousCartPole {
    fn default() -> Self {
        Self::new()
    }
}

impl ContinuousCartPole {
    /// Create a new continuous CartPole environment.
    pub fn new() -> Self {
        Self {
            inner: CartPole::new(),
        }
    }
}

impl Environment for ContinuousCartPole {
    type State = Array1<f64>;
    type Action = Array1<f64>;

    fn reset(&mut self) -> Array1<f64> {
        self.inner.reset()
    }

    fn step(&mut self, action: &Array1<f64>) -> (Array1<f64>, f64, bool) {
        let force = action[0].clamp(-self.inner.force_mag, self.inner.force_mag);
        self.inner.physics_step(force);
        self.inner.steps += 1;

        let done = self.inner.is_terminal();
        let reward = if done { 0.0 } else { 1.0 };
        (self.inner.state.clone(), reward, done)
    }

    fn action_space(&self) -> ActionSpace {
        let mag = self.inner.force_mag;
        ActionSpace::Continuous {
            low: vec![-mag],
            high: vec![mag],
        }
    }

    fn observation_space(&self) -> ObservationSpace {
        self.inner.observation_space()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// GridWorld
// ──────────────────────────────────────────────────────────────────────────────

/// Cell type for `GridWorld`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GridCell {
    /// Open passable cell.
    Empty,
    /// Impassable wall.
    Wall,
    /// Goal cell (episode ends with positive reward).
    Goal,
    /// Hazard cell (episode ends with negative reward).
    Hazard,
}

/// A simple rectangular grid navigation environment.
///
/// **State** (2-dim): `[row / (rows-1), col / (cols-1)]` (normalised to [0, 1]).
///
/// **Actions** (discrete, 4): 0 = up, 1 = right, 2 = down, 3 = left.
///
/// The agent starts at `start` and must reach `goal`.  Walls are impassable.
pub struct GridWorld {
    rows: usize,
    cols: usize,
    grid: Vec<Vec<GridCell>>,
    start: (usize, usize),
    agent: (usize, usize),
    goal_reward: f64,
    hazard_reward: f64,
    step_reward: f64,
    max_steps: usize,
    steps: usize,
    rng: XorShift64,
}

impl GridWorld {
    /// Create a fully-open `rows × cols` grid with the agent at `(0,0)` and
    /// the goal at `(rows-1, cols-1)`.
    pub fn new(rows: usize, cols: usize) -> Self {
        assert!(rows >= 2 && cols >= 2, "grid must be at least 2×2");
        let grid = vec![vec![GridCell::Empty; cols]; rows];
        Self {
            rows,
            cols,
            grid,
            start: (0, 0),
            agent: (0, 0),
            goal_reward: 1.0,
            hazard_reward: -1.0,
            step_reward: -0.01,
            max_steps: rows * cols * 4,
            steps: 0,
            rng: XorShift64::new(make_seed()),
        }
    }

    /// Place a wall at `(row, col)`.
    pub fn set_wall(&mut self, row: usize, col: usize) -> Result<()> {
        self.check_bounds(row, col)?;
        self.grid[row][col] = GridCell::Wall;
        Ok(())
    }

    /// Place the goal at `(row, col)`.
    pub fn set_goal(&mut self, row: usize, col: usize) -> Result<()> {
        self.check_bounds(row, col)?;
        self.grid[row][col] = GridCell::Goal;
        Ok(())
    }

    /// Place a hazard at `(row, col)`.
    pub fn set_hazard(&mut self, row: usize, col: usize) -> Result<()> {
        self.check_bounds(row, col)?;
        self.grid[row][col] = GridCell::Hazard;
        Ok(())
    }

    /// Set the starting cell.
    pub fn set_start(&mut self, row: usize, col: usize) -> Result<()> {
        self.check_bounds(row, col)?;
        self.start = (row, col);
        Ok(())
    }

    fn check_bounds(&self, row: usize, col: usize) -> Result<()> {
        if row >= self.rows || col >= self.cols {
            return Err(NeuralError::InvalidArgument(format!(
                "GridWorld: cell ({}, {}) out of bounds ({}×{})",
                row, col, self.rows, self.cols
            )));
        }
        Ok(())
    }

    fn state_as_array(&self) -> Array1<f64> {
        let r = self.agent.0 as f64 / (self.rows - 1).max(1) as f64;
        let c = self.agent.1 as f64 / (self.cols - 1).max(1) as f64;
        Array1::from_vec(vec![r, c])
    }
}

impl Environment for GridWorld {
    type State = Array1<f64>;
    type Action = Array1<f64>;

    fn reset(&mut self) -> Array1<f64> {
        self.agent = self.start;
        self.steps = 0;
        self.state_as_array()
    }

    fn step(&mut self, action: &Array1<f64>) -> (Array1<f64>, f64, bool) {
        let act = action[0].round() as usize % 4;
        let (r, c) = self.agent;

        let (nr, nc) = match act {
            0 => (r.saturating_sub(1), c),        // up
            1 => (r, (c + 1).min(self.cols - 1)), // right
            2 => ((r + 1).min(self.rows - 1), c), // down
            _ => (r, c.saturating_sub(1)),        // left
        };

        // Wall check — stay in place if blocked
        if self.grid[nr][nc] != GridCell::Wall {
            self.agent = (nr, nc);
        }
        self.steps += 1;

        let cell = self.grid[self.agent.0][self.agent.1];
        let (reward, done) = match cell {
            GridCell::Goal => (self.goal_reward, true),
            GridCell::Hazard => (self.hazard_reward, true),
            _ => (self.step_reward, self.steps >= self.max_steps),
        };

        (self.state_as_array(), reward, done)
    }

    fn action_space(&self) -> ActionSpace {
        ActionSpace::Discrete { n: 4 }
    }

    fn observation_space(&self) -> ObservationSpace {
        ObservationSpace::bounded(vec![0.0; 2], vec![1.0; 2])
    }

    fn render(&self) {
        for r in 0..self.rows {
            let row: String = (0..self.cols)
                .map(|c| {
                    if (r, c) == self.agent {
                        'A'
                    } else {
                        match self.grid[r][c] {
                            GridCell::Empty => '.',
                            GridCell::Wall => '#',
                            GridCell::Goal => 'G',
                            GridCell::Hazard => 'H',
                        }
                    }
                })
                .collect();
            println!("{}", row);
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

// ──────────────────────────────────────────────────────────────────────────────
// PendulumEnv (continuous action, swing-up task)
// ──────────────────────────────────────────────────────────────────────────────

/// Continuous-action pendulum swing-up environment.
///
/// **State** (3-dim): `[cos(θ), sin(θ), θ̇]`
///
/// **Action** (continuous, 1-dim): torque in `[−2, 2]`
///
/// **Reward**: `−(θ² + 0.1·θ̇² + 0.001·τ²)` — penalises deviation from upright.
///
/// The pendulum starts at a random angle and angular velocity.
/// Episodes are fixed-length (default 200 steps).
///
/// Reference: OpenAI Gym `Pendulum-v1` specification.
pub struct PendulumEnv {
    /// Pole angle in radians (0 = upright, π = hanging down).
    theta: f64,
    /// Angular velocity.
    theta_dot: f64,
    /// Current timestep within the episode.
    steps: usize,
    /// Maximum steps per episode.
    max_steps: usize,
    /// Gravity constant.
    gravity: f64,
    /// Pole mass.
    mass: f64,
    /// Pole length.
    length: f64,
    /// Integration timestep.
    dt: f64,
    /// Maximum torque.
    max_torque: f64,
    /// Maximum angular velocity (for clipping).
    max_speed: f64,
    rng: XorShift64,
}

impl Default for PendulumEnv {
    fn default() -> Self {
        Self::new()
    }
}

impl PendulumEnv {
    /// Create a new PendulumEnv with standard parameters.
    pub fn new() -> Self {
        Self {
            theta: 0.0,
            theta_dot: 0.0,
            steps: 0,
            max_steps: 200,
            gravity: 10.0,
            mass: 1.0,
            length: 1.0,
            dt: 0.05,
            max_torque: 2.0,
            max_speed: 8.0,
            rng: XorShift64::new(make_seed()),
        }
    }

    /// Set the maximum episode length.
    pub fn with_max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = max_steps;
        self
    }

    /// Convert angle to the canonical `[−π, π]` range.
    fn angle_normalize(theta: f64) -> f64 {
        let pi = std::f64::consts::PI;
        let modded = theta % (2.0 * pi);
        if modded > pi {
            modded - 2.0 * pi
        } else if modded < -pi {
            modded + 2.0 * pi
        } else {
            modded
        }
    }

    fn current_state(&self) -> Array1<f64> {
        Array1::from_vec(vec![self.theta.cos(), self.theta.sin(), self.theta_dot])
    }
}

impl Environment for PendulumEnv {
    type State = Array1<f64>;
    type Action = Array1<f64>;

    fn reset(&mut self) -> Array1<f64> {
        // Random initial angle in [−π, π] and angular velocity in [−1, 1]
        let pi = std::f64::consts::PI;
        self.theta = self.rng.uniform(-pi, pi);
        self.theta_dot = self.rng.uniform(-1.0, 1.0);
        self.steps = 0;
        self.current_state()
    }

    fn step(&mut self, action: &Array1<f64>) -> (Array1<f64>, f64, bool) {
        let torque = action[0].clamp(-self.max_torque, self.max_torque);
        let g = self.gravity;
        let m = self.mass;
        let l = self.length;
        let dt = self.dt;

        // Euler integration of `θ̈ = (3g/2l)·sin(θ) + (3/ml²)·τ`
        let theta_acc = (3.0 * g / (2.0 * l)) * self.theta.sin() + (3.0 / (m * l * l)) * torque;

        self.theta_dot = (self.theta_dot + dt * theta_acc).clamp(-self.max_speed, self.max_speed);
        self.theta = Self::angle_normalize(self.theta + dt * self.theta_dot);
        self.steps += 1;

        // Reward: −(θ_norm² + 0.1·θ̇² + 0.001·τ²)
        let theta_norm = Self::angle_normalize(self.theta);
        let reward = -(theta_norm * theta_norm
            + 0.1 * self.theta_dot * self.theta_dot
            + 0.001 * torque * torque);

        let done = self.steps >= self.max_steps;
        (self.current_state(), reward, done)
    }

    fn action_space(&self) -> ActionSpace {
        ActionSpace::Continuous {
            low: vec![-self.max_torque],
            high: vec![self.max_torque],
        }
    }

    fn observation_space(&self) -> ObservationSpace {
        ObservationSpace::bounded(
            vec![-1.0, -1.0, -self.max_speed],
            vec![1.0, 1.0, self.max_speed],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cartpole_reset_shape() {
        let mut env = CartPole::new();
        let s = env.reset();
        assert_eq!(s.len(), 4);
    }

    #[test]
    fn cartpole_step_produces_valid_output() {
        let mut env = CartPole::new();
        env.reset();
        let action = Array1::from_vec(vec![1.0_f64]);
        let (s, r, _done) = env.step(&action);
        assert_eq!(s.len(), 4);
        assert!(r >= 0.0);
    }

    #[test]
    fn cartpole_observation_space() {
        let env = CartPole::new();
        assert_eq!(env.observation_space().ndim, 4);
    }

    #[test]
    fn cartpole_discrete_actions() {
        let env = CartPole::new();
        assert!(matches!(env.action_space(), ActionSpace::Discrete { n: 2 }));
    }

    #[test]
    fn continuous_cartpole_actions_are_continuous() {
        let env = ContinuousCartPole::new();
        assert!(matches!(env.action_space(), ActionSpace::Continuous { .. }));
    }

    #[test]
    fn gridworld_basic_episode() {
        let mut env = GridWorld::new(4, 4);
        env.set_goal(3, 3).expect("set goal");
        let s = env.reset();
        assert_eq!(s.len(), 2);

        // Walk right then down repeatedly
        for _ in 0..20 {
            let act = Array1::from_vec(vec![1.0_f64]);
            let (_s, _r, done) = env.step(&act);
            if done {
                break;
            }
        }
    }

    #[test]
    fn gridworld_wall_blocks() {
        let mut env = GridWorld::new(3, 3);
        env.set_goal(2, 2).expect("set goal");
        env.set_wall(0, 1).expect("set wall");
        env.reset();
        // Try to move right (col 0 → col 1, but wall is there)
        let act = Array1::from_vec(vec![1.0_f64]);
        let (s, _, _) = env.step(&act);
        // Agent stays at col 0
        assert!((s[1] - 0.0).abs() < 1e-9, "agent should be blocked by wall");
    }

    #[test]
    fn action_space_clip() {
        let space = ActionSpace::Continuous {
            low: vec![-1.0],
            high: vec![1.0],
        };
        let action = Array1::from_vec(vec![5.0_f64]);
        let clipped = space.clip(&action);
        assert!((clipped[0] - 1.0).abs() < 1e-12);
    }
}
