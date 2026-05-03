//! Advanced RL environments: multi-agent, continuous control, pursuit-evasion

use crate::error::{NeuralError, Result};
use crate::reinforcement::environments::{Action, Environment, Info, Observation, Reward};
use scirs2_core::ndarray::prelude::*;
use std::collections::HashMap;

// ── Type aliases ──────────────────────────────────────────────────────────────

/// Multi-agent step result: (observations, rewards, dones, infos)
pub type MultiAgentStepResult = (Vec<Observation>, Vec<Reward>, Vec<bool>, Vec<Info>);

/// Pursuit-evasion joint step result: (pursuer_obs, evader_obs, pursuer_rewards, evader_rewards, done)
pub type JointStepResult = (Vec<Observation>, Vec<Observation>, Vec<f32>, Vec<f32>, bool);

// ── Multi-agent trait ─────────────────────────────────────────────────────────

/// Trait for multi-agent environments
pub trait MultiAgentEnvironment: Send + Sync {
    /// Number of agents
    fn num_agents(&self) -> usize;

    /// Reset environment → initial observations for all agents
    fn reset(&mut self) -> Result<Vec<Observation>>;

    /// Step with all agents' actions → (next_obs, rewards, dones, infos)
    fn step(&mut self, actions: &[Action]) -> Result<MultiAgentStepResult>;

    /// Observation space dimensions per agent
    fn observation_spaces(&self) -> Vec<usize>;

    /// Action space dimensions per agent
    fn action_spaces(&self) -> Vec<usize>;

    /// Whether actions are continuous per agent
    fn continuous_actions(&self) -> Vec<bool>;
}

// ── Multi-agent grid world ─────────────────────────────────────────────────────

/// Multi-agent cooperative grid world
pub struct MultiAgentGridWorld {
    width: usize,
    height: usize,
    agent_positions: Vec<(usize, usize)>,
    goal_positions: Vec<(usize, usize)>,
    obstacles: Vec<(usize, usize)>,
    observation_radius: usize,
    step_count: usize,
    max_steps: usize,
    communication_enabled: bool,
    rng_state: u64,
}

impl MultiAgentGridWorld {
    /// Create a new multi-agent grid world
    pub fn new(
        width: usize,
        height: usize,
        num_agents: usize,
        observation_radius: usize,
        communication_enabled: bool,
    ) -> Self {
        let mut rng_state: u64 = 0xdeadbeef_00000042;
        let mut agent_positions = Vec::with_capacity(num_agents);
        let mut goal_positions = Vec::with_capacity(num_agents);
        for i in 0..num_agents {
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            let ax = (rng_state as usize) % width;
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            let ay = (rng_state as usize) % height;
            agent_positions.push((ax, ay));
            // Goals on opposite half
            let gx = (i + 1) * width / (num_agents + 1);
            let gy = height.saturating_sub(1);
            goal_positions.push((gx, gy));
        }
        let max_steps = width * height * 4;
        Self {
            width,
            height,
            agent_positions,
            goal_positions,
            obstacles: Vec::new(),
            observation_radius,
            step_count: 0,
            max_steps,
            communication_enabled,
            rng_state,
        }
    }

    /// Add obstacle at (x, y)
    pub fn add_obstacle(&mut self, x: usize, y: usize) {
        self.obstacles.push((x, y));
    }

    fn local_obs(&self, agent_idx: usize) -> Array1<f32> {
        let (ax, ay) = self.agent_positions[agent_idx];
        let r = self.observation_radius;
        let diam = 2 * r + 1;
        let mut obs = Array1::zeros(diam * diam);
        for dy in 0..diam {
            for dx in 0..diam {
                let wx = ax as isize + dx as isize - r as isize;
                let wy = ay as isize + dy as isize - r as isize;
                if wx < 0 || wy < 0 || wx >= self.width as isize || wy >= self.height as isize {
                    obs[dy * diam + dx] = -1.0; // wall
                } else {
                    let (wx, wy) = (wx as usize, wy as usize);
                    if self.obstacles.contains(&(wx, wy)) {
                        obs[dy * diam + dx] = -1.0;
                    } else if self.goal_positions.get(agent_idx) == Some(&(wx, wy)) {
                        obs[dy * diam + dx] = 1.0; // goal
                    }
                }
            }
        }
        obs
    }

    fn obs_dim(&self) -> usize {
        let diam = 2 * self.observation_radius + 1;
        diam * diam
    }
}

impl MultiAgentEnvironment for MultiAgentGridWorld {
    fn num_agents(&self) -> usize {
        self.agent_positions.len()
    }

    fn reset(&mut self) -> Result<Vec<Observation>> {
        self.step_count = 0;
        // Re-randomize agent positions
        for i in 0..self.agent_positions.len() {
            self.rng_state ^= self.rng_state << 13;
            self.rng_state ^= self.rng_state >> 7;
            self.rng_state ^= self.rng_state << 17;
            let ax = (self.rng_state as usize) % self.width;
            self.rng_state ^= self.rng_state << 13;
            self.rng_state ^= self.rng_state >> 7;
            self.rng_state ^= self.rng_state << 17;
            let ay = (self.rng_state as usize) % self.height;
            self.agent_positions[i] = (ax, ay);
        }
        Ok((0..self.agent_positions.len())
            .map(|i| self.local_obs(i))
            .collect())
    }

    fn step(&mut self, actions: &[Action]) -> Result<MultiAgentStepResult> {
        let n = self.agent_positions.len();
        let mut next_obs = Vec::with_capacity(n);
        let mut rewards = vec![0.0f32; n];
        let mut dones = vec![false; n];
        let infos = vec![Info::new(); n];

        for (i, action) in actions.iter().enumerate().take(n) {
            let act = if action.is_empty() {
                0
            } else {
                action[0] as usize % 4
            };
            let (r, c) = self.agent_positions[i];
            let new_pos = match act {
                0 => (r.saturating_sub(1), c),
                1 => ((r + 1).min(self.height - 1), c),
                2 => (r, c.saturating_sub(1)),
                _ => (r, (c + 1).min(self.width - 1)),
            };
            if !self.obstacles.contains(&new_pos) {
                self.agent_positions[i] = new_pos;
            }
            if self.agent_positions[i] == self.goal_positions[i] {
                rewards[i] = 10.0;
                dones[i] = true;
            } else {
                rewards[i] = -0.01;
            }
        }
        self.step_count += 1;
        let timeout = self.step_count >= self.max_steps;
        if timeout {
            dones.iter_mut().for_each(|d| *d = true);
        }
        for i in 0..n {
            next_obs.push(self.local_obs(i));
        }
        Ok((next_obs, rewards, dones, infos))
    }

    fn observation_spaces(&self) -> Vec<usize> {
        vec![self.obs_dim(); self.agent_positions.len()]
    }

    fn action_spaces(&self) -> Vec<usize> {
        vec![4; self.agent_positions.len()]
    }

    fn continuous_actions(&self) -> Vec<bool> {
        vec![false; self.agent_positions.len()]
    }
}

// ── Multi-agent wrapper ───────────────────────────────────────────────────────

/// Wraps a single-agent environment for each of multiple independent agents
pub struct MultiAgentWrapper<E: Environment> {
    envs: Vec<E>,
}

impl<E: Environment> MultiAgentWrapper<E> {
    /// Create a wrapper over multiple environment instances
    pub fn new(envs: Vec<E>) -> Self {
        Self { envs }
    }

    /// Number of agents
    pub fn n_agents(&self) -> usize {
        self.envs.len()
    }

    /// Reset all environments
    pub fn reset_all(&mut self) -> Result<Vec<Observation>> {
        self.envs.iter_mut().map(|e| e.reset()).collect()
    }

    /// Step all environments independently
    pub fn step_all(
        &mut self,
        actions: &[Action],
    ) -> Result<Vec<(Observation, Reward, bool, Info)>> {
        self.envs
            .iter_mut()
            .zip(actions.iter())
            .map(|(e, a)| e.step(a))
            .collect()
    }
}

// ── Pursuit-Evasion ───────────────────────────────────────────────────────────

/// Pursuit-evasion game: k pursuers try to catch m evaders
pub struct PursuitEvasion {
    width: usize,
    height: usize,
    pursuer_positions: Vec<(usize, usize)>,
    evader_positions: Vec<(usize, usize)>,
    capture_radius: usize,
    step_count: usize,
    max_steps: usize,
    rng_state: u64,
}

impl PursuitEvasion {
    /// Create a new pursuit-evasion game
    pub fn new(
        width: usize,
        height: usize,
        n_pursuers: usize,
        n_evaders: usize,
        capture_radius: usize,
    ) -> Self {
        Self {
            width,
            height,
            pursuer_positions: vec![(0, 0); n_pursuers],
            evader_positions: vec![(width - 1, height - 1); n_evaders],
            capture_radius,
            step_count: 0,
            max_steps: width * height * 2,
            rng_state: 0xabcd1234_5678ef90,
        }
    }

    fn obs_for(&self, pos: (usize, usize)) -> Observation {
        let (x, y) = pos;
        // Observation: normalised (x, y) of agent + nearest pursuer/evader
        Array1::from_vec(vec![
            x as f32 / self.width.max(1) as f32,
            y as f32 / self.height.max(1) as f32,
        ])
    }

    fn pursuer_obs(&self) -> Vec<Observation> {
        let mut obs: Vec<Observation> = self
            .pursuer_positions
            .iter()
            .map(|&p| self.obs_for(p))
            .collect();
        // Append nearest evader info
        for (i, &pp) in self.pursuer_positions.iter().enumerate() {
            if let Some(&ep) = self.evader_positions.first() {
                let dx = ep.0 as f32 - pp.0 as f32;
                let dy = ep.1 as f32 - pp.1 as f32;
                let mut extended = obs[i].to_vec();
                extended.push(dx / self.width.max(1) as f32);
                extended.push(dy / self.height.max(1) as f32);
                obs[i] = Array1::from_vec(extended);
            }
        }
        obs
    }

    fn evader_obs(&self) -> Vec<Observation> {
        self.evader_positions
            .iter()
            .map(|&p| self.obs_for(p))
            .collect()
    }

    fn move_pos(&self, pos: (usize, usize), act: usize) -> (usize, usize) {
        let (r, c) = pos;
        match act {
            0 => (r.saturating_sub(1), c),
            1 => ((r + 1).min(self.height - 1), c),
            2 => (r, c.saturating_sub(1)),
            _ => (r, (c + 1).min(self.width - 1)),
        }
    }

    fn is_captured(&self, evader: (usize, usize)) -> bool {
        self.pursuer_positions.iter().any(|&p| {
            let dx = (p.0 as isize - evader.0 as isize).unsigned_abs();
            let dy = (p.1 as isize - evader.1 as isize).unsigned_abs();
            dx + dy <= self.capture_radius
        })
    }

    /// Take a joint step; returns (pursuer_obs, evader_obs, pursuer_rewards, evader_rewards, done)
    pub fn joint_step(
        &mut self,
        pursuer_actions: &[Action],
        evader_actions: &[Action],
    ) -> Result<JointStepResult> {
        for (i, a) in pursuer_actions.iter().enumerate() {
            if i < self.pursuer_positions.len() {
                let act = if a.is_empty() { 0 } else { a[0] as usize % 4 };
                self.pursuer_positions[i] = self.move_pos(self.pursuer_positions[i], act);
            }
        }
        for (i, a) in evader_actions.iter().enumerate() {
            if i < self.evader_positions.len() {
                let act = if a.is_empty() { 0 } else { a[0] as usize % 4 };
                self.evader_positions[i] = self.move_pos(self.evader_positions[i], act);
            }
        }
        self.step_count += 1;

        let evader_captured: Vec<bool> = self
            .evader_positions
            .iter()
            .map(|&e| self.is_captured(e))
            .collect();
        let n_captured = evader_captured.iter().filter(|&&c| c).count();
        let pursuer_rewards = vec![n_captured as f32; self.pursuer_positions.len()];
        let evader_rewards: Vec<f32> = evader_captured
            .iter()
            .map(|&c| if c { -1.0 } else { 0.1 })
            .collect();
        let done = evader_captured.iter().all(|&c| c) || self.step_count >= self.max_steps;

        Ok((
            self.pursuer_obs(),
            self.evader_obs(),
            pursuer_rewards,
            evader_rewards,
            done,
        ))
    }

    /// Pursuer count
    pub fn n_pursuers(&self) -> usize {
        self.pursuer_positions.len()
    }

    /// Evader count
    pub fn n_evaders(&self) -> usize {
        self.evader_positions.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_agent_grid_world_reset() {
        let mut env = MultiAgentGridWorld::new(5, 5, 2, 1, false);
        let obs = env.reset().expect("reset ok");
        assert_eq!(obs.len(), 2);
        for o in &obs {
            assert_eq!(o.len(), 9); // (2*1+1)^2
        }
    }

    #[test]
    fn test_multi_agent_grid_world_step() {
        let mut env = MultiAgentGridWorld::new(5, 5, 2, 1, false);
        env.reset().expect("reset ok");
        let actions = vec![Array1::from_vec(vec![1.0]), Array1::from_vec(vec![0.0])];
        let (obs, rewards, dones, _infos) = env.step(&actions).expect("step ok");
        assert_eq!(obs.len(), 2);
        assert_eq!(rewards.len(), 2);
        assert_eq!(dones.len(), 2);
    }

    #[test]
    fn test_multi_agent_spaces() {
        let env = MultiAgentGridWorld::new(4, 4, 3, 2, true);
        assert_eq!(env.num_agents(), 3);
        let obs_spaces = env.observation_spaces();
        assert_eq!(obs_spaces.len(), 3);
        let act_spaces = env.action_spaces();
        assert!(act_spaces.iter().all(|&a| a == 4));
    }

    #[test]
    fn test_pursuit_evasion_joint_step() {
        let mut pe = PursuitEvasion::new(6, 6, 2, 1, 1);
        let p_actions = vec![Array1::from_vec(vec![1.0]); 2];
        let e_actions = vec![Array1::from_vec(vec![0.0])];
        let (pobs, eobs, pr, er, _done) = pe.joint_step(&p_actions, &e_actions).expect("step ok");
        assert_eq!(pobs.len(), 2);
        assert_eq!(eobs.len(), 1);
        assert_eq!(pr.len(), 2);
        assert_eq!(er.len(), 1);
    }

    #[test]
    fn test_pursuit_evasion_counts() {
        let pe = PursuitEvasion::new(8, 8, 3, 2, 1);
        assert_eq!(pe.n_pursuers(), 3);
        assert_eq!(pe.n_evaders(), 2);
    }
}
