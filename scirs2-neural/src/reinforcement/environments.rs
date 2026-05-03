//! Reinforcement learning environments

use crate::error::Result;
use scirs2_core::ndarray::prelude::*;
use std::collections::HashMap;

/// Observation from environment
pub type Observation = Array1<f32>;
/// Action to take in environment
pub type Action = Array1<f32>;
/// Reward from environment
pub type Reward = f32;
/// Environment info map
pub type Info = HashMap<String, f32>;

/// Base trait for reinforcement learning environments
pub trait Environment: Send + Sync {
    /// Reset the environment and return initial observation
    fn reset(&mut self) -> Result<Observation>;

    /// Take a step: returns `(next_obs, reward, done, info)`
    fn step(&mut self, action: &Action) -> Result<(Observation, Reward, bool, Info)>;

    /// Observation space dimensionality
    fn observation_space(&self) -> usize;

    /// Action space dimensionality
    fn action_space(&self) -> usize;

    /// Whether actions are continuous
    fn continuous_actions(&self) -> bool;

    /// Action bounds for continuous actions
    fn action_bounds(&self) -> Option<(Array1<f32>, Array1<f32>)> {
        None
    }

    /// Render the environment (no-op by default)
    fn render(&self) -> Result<()> {
        Ok(())
    }

    /// Close the environment and release resources
    fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

// ── CartPole ──────────────────────────────────────────────────────────────────

/// Classic CartPole environment (Barto, Sutton, Anderson, 1983)
pub struct CartPole {
    state: Array1<f32>,
    steps: usize,
    max_steps: usize,
    gravity: f32,
    mass_cart: f32,
    mass_pole: f32,
    length: f32,
    force_mag: f32,
    tau: f32,
    rng_seed: u64,
}

impl Default for CartPole {
    fn default() -> Self {
        Self {
            state: Array1::zeros(4),
            steps: 0,
            max_steps: 200,
            gravity: 9.8,
            mass_cart: 1.0,
            mass_pole: 0.1,
            length: 0.5,
            force_mag: 10.0,
            tau: 0.02,
            rng_seed: 42,
        }
    }
}

impl CartPole {
    /// Create a new CartPole environment
    pub fn new() -> Self {
        Self::default()
    }

    fn is_done(&self) -> bool {
        let x = self.state[0];
        let theta = self.state[2];
        !(-2.4_f32..=2.4).contains(&x)
            || !(-0.2095_f32..=0.2095).contains(&theta)
            || self.steps >= self.max_steps
    }

    // Simple deterministic small noise using LCG
    fn next_f32(&mut self) -> f32 {
        self.rng_seed = self
            .rng_seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let hi = (self.rng_seed >> 33) as u32;
        (hi as f32 / u32::MAX as f32) * 0.1 - 0.05
    }
}

impl Environment for CartPole {
    fn reset(&mut self) -> Result<Observation> {
        self.state = Array1::from_vec(vec![
            self.next_f32(),
            self.next_f32(),
            self.next_f32(),
            self.next_f32(),
        ]);
        self.steps = 0;
        Ok(self.state.clone())
    }

    fn step(&mut self, action: &Action) -> Result<(Observation, Reward, bool, Info)> {
        let x = self.state[0];
        let x_dot = self.state[1];
        let theta = self.state[2];
        let theta_dot = self.state[3];

        let force = if action.is_empty() || action[0] > 0.0 {
            self.force_mag
        } else {
            -self.force_mag
        };

        let total_mass = self.mass_cart + self.mass_pole;
        let polemass_len = self.mass_pole * self.length;
        let cos_theta = theta.cos();
        let sin_theta = theta.sin();
        let tmp = (force + polemass_len * theta_dot * theta_dot * sin_theta) / total_mass;
        let theta_acc = (self.gravity * sin_theta - cos_theta * tmp)
            / (self.length * (4.0 / 3.0 - self.mass_pole * cos_theta * cos_theta / total_mass));
        let x_acc = tmp - polemass_len * theta_acc * cos_theta / total_mass;

        self.state = Array1::from_vec(vec![
            x + self.tau * x_dot,
            x_dot + self.tau * x_acc,
            theta + self.tau * theta_dot,
            theta_dot + self.tau * theta_acc,
        ]);
        self.steps += 1;
        let done = self.is_done();
        let reward = if done && self.steps < self.max_steps {
            0.0
        } else {
            1.0
        };
        Ok((self.state.clone(), reward, done, Info::new()))
    }

    fn observation_space(&self) -> usize {
        4
    }

    fn action_space(&self) -> usize {
        2
    }

    fn continuous_actions(&self) -> bool {
        false
    }
}

// ── GridWorld ─────────────────────────────────────────────────────────────────

/// Simple grid-world navigation environment
pub struct GridWorld {
    size: usize,
    agent_pos: (usize, usize),
    goal_pos: (usize, usize),
    steps: usize,
    max_steps: usize,
}

impl GridWorld {
    /// Create a new GridWorld of the given size
    pub fn new(size: usize) -> Self {
        Self {
            size,
            agent_pos: (0, 0),
            goal_pos: (size - 1, size - 1),
            steps: 0,
            max_steps: size * size * 4,
        }
    }

    fn pos_to_obs(&self) -> Array1<f32> {
        let mut obs = Array1::zeros(self.size * self.size);
        let idx = self.agent_pos.0 * self.size + self.agent_pos.1;
        if idx < obs.len() {
            obs[idx] = 1.0;
        }
        obs
    }
}

impl Environment for GridWorld {
    fn reset(&mut self) -> Result<Observation> {
        self.agent_pos = (0, 0);
        self.steps = 0;
        Ok(self.pos_to_obs())
    }

    fn step(&mut self, action: &Action) -> Result<(Observation, Reward, bool, Info)> {
        let act = if action.is_empty() {
            0
        } else {
            action[0] as usize % 4
        };
        let (r, c) = self.agent_pos;
        let new_pos = match act {
            0 => (r.saturating_sub(1), c),        // up
            1 => ((r + 1).min(self.size - 1), c), // down
            2 => (r, c.saturating_sub(1)),        // left
            _ => (r, (c + 1).min(self.size - 1)), // right
        };
        self.agent_pos = new_pos;
        self.steps += 1;
        let done = self.agent_pos == self.goal_pos || self.steps >= self.max_steps;
        let reward = if self.agent_pos == self.goal_pos {
            10.0
        } else {
            -0.01
        };
        Ok((self.pos_to_obs(), reward, done, Info::new()))
    }

    fn observation_space(&self) -> usize {
        self.size * self.size
    }

    fn action_space(&self) -> usize {
        4
    }

    fn continuous_actions(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cartpole_reset() {
        let mut env = CartPole::new();
        let obs = env.reset().expect("reset ok");
        assert_eq!(obs.len(), 4);
    }

    #[test]
    fn test_cartpole_step() {
        let mut env = CartPole::new();
        env.reset().expect("reset ok");
        let action = Array1::from_vec(vec![1.0]);
        let (obs, _reward, _done, _info) = env.step(&action).expect("step ok");
        assert_eq!(obs.len(), 4);
    }

    #[test]
    fn test_cartpole_spaces() {
        let env = CartPole::new();
        assert_eq!(env.observation_space(), 4);
        assert_eq!(env.action_space(), 2);
        assert!(!env.continuous_actions());
    }

    #[test]
    fn test_gridworld_basic() {
        let mut env = GridWorld::new(4);
        let obs = env.reset().expect("reset ok");
        assert_eq!(obs.len(), 16);
        let action = Array1::from_vec(vec![1.0]); // down
        let (obs2, _r, _done, _info) = env.step(&action).expect("step ok");
        assert_eq!(obs2.len(), 16);
    }

    #[test]
    fn test_environment_render_close() {
        let mut env = CartPole::new();
        env.reset().expect("reset ok");
        env.render().expect("render ok");
        env.close().expect("close ok");
    }
}
