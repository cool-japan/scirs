//! Advanced Reinforcement Learning Environments
//!
//! This module implements sophisticated RL environments including multi-agent
//! scenarios, partially observable environments, and complex control tasks.

use crate::error::{NeuralError, Result};
use crate::reinforcement::environments::{Action, Environment, Info, Observation, Reward};
use ndarray::prelude::*;
use rand::Rng;
use std::collections::HashMap;
/// Multi-agent environment trait
pub trait MultiAgentEnvironment: Send + Sync {
    /// Number of agents
    fn num_agents(&self) -> usize;
    /// Reset environment and return initial observations for all agents
    fn reset(&mut self) -> Result<Vec<Observation>>;
    /// Take a step with actions from all agents
    fn step(
        &mut self,
        actions: &[Action],
    ) -> Result<(Vec<Observation>, Vec<Reward>, Vec<bool>, Vec<Info>)>;
    /// Get observation space for each agent
    fn observation_spaces(&self) -> Vec<usize>;
    /// Get action space for each agent
    fn action_spaces(&self) -> Vec<usize>;
    /// Check if actions are continuous for each agent
    fn continuous_actions(&self) -> Vec<bool>;
}
/// Partially Observable Stochastic Game (POSG)
pub struct MultiAgentGridWorld {
    /// Grid dimensions
    width: usize,
    height: usize,
    /// Agent positions
    agent_positions: Vec<(usize, usize)>,
    /// Goal positions
    goal_positions: Vec<(usize, usize)>,
    /// Obstacle positions
    obstacles: Vec<(usize, usize)>,
    /// Agents' local observation radius
    observation_radius: usize,
    /// Current step
    step_count: usize,
    /// Maximum steps per episode
    max_steps: usize,
    /// Whether agents can share observations
    communication_enabled: bool,
impl MultiAgentGridWorld {
    /// Create a new multi-agent grid world
    pub fn new(
        width: usize,
        height: usize,
        num_agents: usize,
        observation_radius: usize,
        communication_enabled: bool,
    ) -> Self {
        let mut agent_positions = Vec::new();
        let mut goal_positions = Vec::new();
        // Place agents randomly
        let mut rng = rng();
        for _ in 0..num_agents {
            let x = rng.random_range(0..width);
            let y = rng.random_range(0..height);
            agent_positions.push((x..y));
            // Place goals randomly (different from agent positions)
            loop {
                let gx = rng.random_range(0..width);
                let gy = rng.random_range(0..height);
                if !agent_positions.contains(&(gx..gy)) {
                    goal_positions.push((gx, gy));
                    break;
                }
            }
        }
        // Add some random obstacles
        let mut obstacles = Vec::new();
        let num_obstacles = (width * height) / 10;
        for _ in 0..num_obstacles {
                let ox = rng.random_range(0..width);
                let oy = rng.random_range(0..height);
                if !agent_positions.contains(&(ox..oy)) && !goal_positions.contains(&(ox, oy)) {
                    obstacles.push((ox, oy));
        Self {
            width,
            height,
            agent_positions,
            goal_positions,
            obstacles,
            observation_radius,
            step_count: 0,
            max_steps: 100,
            communication_enabled,
    }
    /// Get local observation for an agent
    fn get_local_observation(&self, agentid: usize) -> Array1<f32> {
        let (ax, ay) = self.agent_positions[agent_id];
        let r = self.observation_radius as i32;
        // Local grid observation
        let obs_size = (2 * self.observation_radius + 1).pow(2);
        let mut observation = Array1::zeros(obs_size * 4); // 4 channels: empty, obstacles, agents, goals
        let mut idx = 0;
        for dy in -r..=r {
            for dx in -r..=r {
                let x = ax as i32 + dx;
                let y = ay as i32 + dy;
                // Check bounds
                if x >= 0 && x < self.width as i32 && y >= 0 && y < self.height as i32 {
                    let pos = (x as usize, y as usize);
                    // Empty space (default 1)
                    observation[idx] = 1.0;
                    // Obstacles
                    if self.obstacles.contains(&pos) {
                        observation[idx] = 0.0;
                        observation[idx + obs_size] = 1.0;
                    }
                    // Other agents
                    for (i, &agent_pos) in self.agent_positions.iter().enumerate() {
                        if i != agent_id && agent_pos == pos {
                            observation[idx] = 0.0;
                            observation[idx + 2 * obs_size] = 1.0;
                        }
                    // Goals
                    if self.goal_positions.contains(&pos) {
                        observation[idx + 3 * obs_size] = 1.0;
                } else {
                    // Out of bounds (treat as obstacle)
                    observation[idx] = 0.0;
                    observation[idx + obs_size] = 1.0;
                idx += 1;
        // Add communication channel if enabled
        if self.communication_enabled {
            let comm_size = self.agent_positions.len() * 2; // x, y coordinates of all agents
            let mut comm_data = Array1::zeros(comm_size);
            for (i, &(x, y)) in self.agent_positions.iter().enumerate() {
                comm_data[i * 2] = x as f32 / self.width as f32;
                comm_data[i * 2 + 1] = y as f32 / self.height as f32;
            // Concatenate observation with communication data
            let mut full_obs = Array1::zeros(observation.len() + comm_data.len());
            full_obs
                .slice_mut(s![..observation.len()])
                .assign(&observation);
                .slice_mut(s![observation.len()..])
                .assign(&comm_data);
            return full_obs;
        observation
    /// Check if position is valid (not occupied by obstacle or other agent)
    fn is_valid_position(&self, pos: (usize, usize), exclude_agent: Option<usize>) -> bool {
        if self.obstacles.contains(&pos) {
            return false;
        for (i, &agent_pos) in self.agent_positions.iter().enumerate() {
            if Some(i) != exclude_agent && agent_pos == pos {
                return false;
        true
impl MultiAgentEnvironment for MultiAgentGridWorld {
    fn num_agents(&self) -> usize {
        self.agent_positions.len()
    fn reset(&mut self) -> Result<Vec<Observation>> {
        // Reset agent and goal positions
        for i in 0..self.agent_positions.len() {
            // Reset agent position
                let x = rng.random_range(0..self.width);
                let y = rng.random_range(0..self.height);
                if self.is_valid_position((x..y), Some(i)) {
                    self.agent_positions[i] = (x, y);
            // Reset goal position
                let gx = rng.random_range(0..self.width);
                let gy = rng.random_range(0..self.height);
                if self.is_valid_position((gx..gy), None)
                    && !self.goal_positions.contains(&(gx, gy))
                {
                    self.goal_positions[i] = (gx, gy);
        self.step_count = 0;
        // Return observations for all agents
        let mut observations = Vec::new();
        for i in 0..self.num_agents() {
            observations.push(self.get_local_observation(i));
        Ok(observations)
    ) -> Result<(Vec<Observation>, Vec<Reward>, Vec<bool>, Vec<Info>)> {
        if actions.len() != self.num_agents() {
            return Err(NeuralError::InvalidArgument(format!(
                "Expected {} actions, got {}",
                self.num_agents(),
                actions.len()
            )));
        let mut rewards = vec![0.0; self.num_agents()];
        let mut dones = vec![false; self.num_agents()];
        let mut infos = vec![HashMap::new(); self.num_agents()];
        // Process actions for each agent
        for (i, action) in actions.iter().enumerate() {
            let (x, y) = self.agent_positions[i];
            // Decode action (0: up, 1: down, 2: left, 3: right, 4: stay)
            let action_idx = if action[0] < 0.2 {
                0 // up
            } else if action[0] < 0.4 {
                1 // down
            } else if action[0] < 0.6 {
                2 // left
            } else if action[0] < 0.8 {
                3 // right
            } else {
                4 // stay
            };
            let new_pos = match action_idx {
                0 => (x, y.saturating_sub(1)),          // up
                1 => (x, (y + 1).min(self.height - 1)), // down
                2 => (x.saturating_sub(1), y),          // left
                3 => ((x + 1).min(self.width - 1), y),  // right
                _ => (x, y),                            // stay
            // Check if new position is valid
            if self.is_valid_position(new_pos, Some(i)) {
                self.agent_positions[i] = new_pos;
            // Check if agent reached its goal
            if self.agent_positions[i] == self.goal_positions[i] {
                rewards[i] = 10.0;
                dones[i] = true;
                // Small negative reward for each step
                rewards[i] = -0.01;
                // Bonus for getting closer to goal
                let old_dist = ((x as f32 - self.goal_positions[i].0 as f32).powi(2)
                    + (y as f32 - self.goal_positions[i].1 as f32).powi(2))
                .sqrt();
                let new_dist = ((self.agent_positions[i].0 as f32
                    - self.goal_positions[i].0 as f32)
                    .powi(2)
                    + (self.agent_positions[i].1 as f32 - self.goal_positions[i].1 as f32).powi(2))
                if new_dist < old_dist {
                    rewards[i] += 0.1;
            infos[i].insert("position_x".to_string(), self.agent_positions[i].0 as f32);
            infos[i].insert("position_y".to_string(), self.agent_positions[i].1 as f32);
            infos[i].insert("goal_x".to_string(), self.goal_positions[i].0 as f32);
            infos[i].insert("goal_y".to_string(), self.goal_positions[i].1 as f32);
        self.step_count += 1;
        // Episode ends if max steps reached or all agents done
        let episode_done = self.step_count >= self.max_steps || dones.iter().all(|&d| d);
        if episode_done {
            for done in &mut dones {
                *done = true;
        // Get new observations
        Ok((observations, rewards, dones, infos))
    fn observation_spaces(&self) -> Vec<usize> {
        let obs_size = (2 * self.observation_radius + 1).pow(2) * 4; // 4 channels
        let comm_size = if self.communication_enabled {
            self.agent_positions.len() * 2
        } else {
            0
        };
        vec![obs_size + comm_size; self.num_agents()]
    fn action_spaces(&self) -> Vec<usize> {
        vec![1; self.num_agents()] // Single continuous action per agent
    fn continuous_actions(&self) -> Vec<bool> {
        vec![false; self.num_agents()] // Discrete actions encoded as continuous
/// Pursuit-Evasion environment
pub struct PursuitEvasion {
    /// Environment dimensions
    width: f32,
    height: f32,
    /// Pursuer positions and velocities
    pursuers: Vec<Agent>,
    /// Evader positions and velocities
    evaders: Vec<Agent>,
    /// Maximum speed for agents
    max_speed: f32,
    /// Capture radius
    capture_radius: f32,
    /// Maximum steps
/// Agent in pursuit-evasion game
#[derive(Debug, Clone)]
struct Agent {
    position: (f32, f32),
    velocity: (f32, f32),
    captured: bool,
impl Agent {
    fn new(x: f32, y: f32) -> Self {
            position: (x, y),
            velocity: (0.0, 0.0),
            captured: false,
    fn distance_to(&self, other: &Agent) -> f32 {
        let dx = self.position.0 - other.position.0;
        let dy = self.position.1 - other.position.1;
        (dx * dx + dy * dy).sqrt()
impl PursuitEvasion {
    /// Create a new pursuit-evasion environment
        width: f32,
        height: f32,
        num_pursuers: usize,
        num_evaders: usize,
        max_speed: f32,
        capture_radius: f32,
        let mut pursuers = Vec::new();
        for _ in 0..num_pursuers {
            let x = rng.random_range(0.0..width);
            let y = rng.random_range(0.0..height);
            pursuers.push(Agent::new(x..y));
        let mut evaders = Vec::new();
        for _ in 0..num_evaders {
            evaders.push(Agent::new(x, y));
            pursuers,
            evaders,
            max_speed,
            capture_radius,
            max_steps: 500,
    /// Get observation for a pursuer
    fn get_pursuer_observation(&self, pursuerid: usize) -> Array1<f32> {
        let pursuer = &self.pursuers[pursuer_id];
        let mut obs = Vec::new();
        // Own position and velocity
        obs.push(pursuer.position.0 / self.width);
        obs.push(pursuer.position.1 / self.height);
        obs.push(pursuer.velocity.0 / self.max_speed);
        obs.push(pursuer.velocity.1 / self.max_speed);
        // Relative positions of other pursuers
        for (i, other) in self.pursuers.iter().enumerate() {
            if i != pursuer_id {
                let dx = (other.position.0 - pursuer.position.0) / self.width;
                let dy = (other.position.1 - pursuer.position.1) / self.height;
                obs.push(dx);
                obs.push(dy);
        // Relative positions of evaders (only if not captured)
        for evader in &self.evaders {
            if !evader.captured {
                let dx = (evader.position.0 - pursuer.position.0) / self.width;
                let dy = (evader.position.1 - pursuer.position.1) / self.height;
                obs.push(if evader.captured { 0.0 } else { 1.0 });
                obs.push(0.0);
        Array1::from_vec(obs)
    /// Get observation for an evader
    fn get_evader_observation(&self, evaderid: usize) -> Array1<f32> {
        let evader = &self.evaders[evader_id];
        obs.push(evader.position.0 / self.width);
        obs.push(evader.position.1 / self.height);
        obs.push(evader.velocity.0 / self.max_speed);
        obs.push(evader.velocity.1 / self.max_speed);
        // Relative positions of pursuers
        for pursuer in &self.pursuers {
            let dx = (pursuer.position.0 - evader.position.0) / self.width;
            let dy = (pursuer.position.1 - evader.position.1) / self.height;
            obs.push(dx);
            obs.push(dy);
        // Relative positions of other evaders
        for (i, other) in self.evaders.iter().enumerate() {
            if i != evader_id && !other.captured {
                let dx = (other.position.0 - evader.position.0) / self.width;
                let dy = (other.position.1 - evader.position.1) / self.height;
impl MultiAgentEnvironment for PursuitEvasion {
        self.pursuers.len() + self.evaders.len()
        // Reset pursuers
        for pursuer in &mut self.pursuers {
            pursuer.position.0 = rng.random_range(0.0..self.width);
            pursuer.position.1 = rng.random_range(0.0..self.height);
            pursuer.velocity = (0.0..0.0);
            pursuer.captured = false;
        // Reset evaders
        for evader in &mut self.evaders {
            evader.position.0 = rng.random_range(0.0..self.width);
            evader.position.1 = rng.random_range(0.0..self.height);
            evader.velocity = (0.0..0.0);
            evader.captured = false;
        // Pursuer observations
        for i in 0..self.pursuers.len() {
            observations.push(self.get_pursuer_observation(i));
        // Evader observations
        for i in 0..self.evaders.len() {
            observations.push(self.get_evader_observation(i));
        let dt = 0.1; // Time step
        // Update pursuers
        for (i, action) in actions.iter().take(self.pursuers.len()).enumerate() {
            let pursuer = &mut self.pursuers[i];
            // Action is acceleration in x and y
            let ax = action[0] * self.max_speed;
            let ay = action[1] * self.max_speed;
            // Update velocity with damping
            pursuer.velocity.0 = (pursuer.velocity.0 + ax * dt) * 0.9;
            pursuer.velocity.1 = (pursuer.velocity.1 + ay * dt) * 0.9;
            // Clip velocity to max speed
            let speed = (pursuer.velocity.0 * pursuer.velocity.0
                + pursuer.velocity.1 * pursuer.velocity.1)
            if speed > self.max_speed {
                pursuer.velocity.0 *= self.max_speed / speed;
                pursuer.velocity.1 *= self.max_speed / speed;
            // Update position
            pursuer.position.0 += pursuer.velocity.0 * dt;
            pursuer.position.1 += pursuer.velocity.1 * dt;
            // Keep within bounds
            pursuer.position.0 = pursuer.position.0.max(0.0).min(self.width);
            pursuer.position.1 = pursuer.position.1.max(0.0).min(self.height);
        // Update evaders
        for (i, action) in actions.iter().skip(self.pursuers.len()).enumerate() {
            if self.evaders[i].captured {
                continue;
            let evader = &mut self.evaders[i];
            evader.velocity.0 = (evader.velocity.0 + ax * dt) * 0.9;
            evader.velocity.1 = (evader.velocity.1 + ay * dt) * 0.9;
            let speed = (evader.velocity.0 * evader.velocity.0
                + evader.velocity.1 * evader.velocity.1)
                evader.velocity.0 *= self.max_speed / speed;
                evader.velocity.1 *= self.max_speed / speed;
            evader.position.0 += evader.velocity.0 * dt;
            evader.position.1 += evader.velocity.1 * dt;
            evader.position.0 = evader.position.0.max(0.0).min(self.width);
            evader.position.1 = evader.position.1.max(0.0).min(self.height);
        // Check for captures
            if evader.captured {
            for pursuer in &self.pursuers {
                if pursuer.distance_to(evader) < self.capture_radius {
                    evader.captured = true;
        // Calculate rewards
        let captured_count = self.evaders.iter().filter(|e| e.captured).count();
        // Pursuer rewards: positive for captures, small negative for time
            rewards[i] = captured_count as f32 * 10.0 - 0.01;
        // Evader rewards: negative for being captured, positive for survival
                rewards[self.pursuers.len() + i] = -10.0;
                rewards[self.pursuers.len() + i] = 0.1;
        // Episode ends if all evaders captured or max steps reached
        let all_captured = self.evaders.iter().all(|e| e.captured);
        let episode_done = all_captured || self.step_count >= self.max_steps;
        let dones = vec![episode_done; self.num_agents()];
        // Create info
        for (i, info) in infos.iter_mut().enumerate() {
            info.insert("step".to_string(), self.step_count as f32);
            info.insert("captured_count".to_string(), captured_count as f32);
        let mut spaces = Vec::new();
        // Pursuer observation space
        for _ in 0..self.pursuers.len() {
            let obs_size = 4 + // own state
                (self.pursuers.len() - 1) * 2 + // other pursuers
                self.evaders.len() * 3; // evaders
            spaces.push(obs_size);
        // Evader observation space
        for _ in 0..self.evaders.len() {
                self.pursuers.len() * 2 + // pursuers
                (self.evaders.len() - 1) * 2; // other evaders
        spaces
        vec![2; self.num_agents()] // 2D acceleration for all agents
        vec![true; self.num_agents()] // Continuous actions for all agents
/// Wrapper to convert multi-agent environment to single-agent
pub struct MultiAgentWrapper<E: MultiAgentEnvironment> {
    env: E,
    agent_id: usize,
impl<E: MultiAgentEnvironment> MultiAgentWrapper<E> {
    /// Create a new multi-agent wrapper for a specific agent
    pub fn new(_env: E, agentid: usize) -> Result<Self> {
        if agent_id >= env.num_agents() {
                "Agent ID {} out of range (0-{})",
                agent_id,
                env.num_agents() - 1
        Ok(Self { env, agent_id })
    /// Get the underlying multi-agent environment
    pub fn get_env(&self) -> &E {
        &self.env
    /// Get mutable reference to the underlying environment
    pub fn get_env_mut(&mut self) -> &mut E {
        &mut self.env
impl<E: MultiAgentEnvironment> Environment for MultiAgentWrapper<E> {
    fn reset(&mut self) -> Result<Observation> {
        let observations = self.env.reset()?;
        Ok(observations[self.agent_id].clone())
    fn step(&mut self, action: &Action) -> Result<(Observation, Reward, bool, Info)> {
        // Create dummy actions for other agents (zeros)
        let mut actions = Vec::new();
        for i in 0..self.env.num_agents() {
            if i == self.agent_id {
                actions.push(action.clone());
                let action_size = self.env.action_spaces()[i];
                actions.push(Array1::zeros(action_size));
        let (observations, rewards, dones, infos) = self.env.step(&actions)?;
        Ok((
            observations[self.agent_id].clone(),
            rewards[self.agent_id],
            dones[self.agent_id],
            infos[self.agent_id].clone(),
        ))
    fn observation_space(&self) -> usize {
        self.env.observation_spaces()[self.agent_id]
    fn action_space(&self) -> usize {
        self.env.action_spaces()[self.agent_id]
    fn continuous_actions(&self) -> bool {
        self.env.continuous_actions()[self.agent_id]
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_multi_agent_grid_world() {
        let mut env = MultiAgentGridWorld::new(5, 5, 2, 1, false);
        assert_eq!(env.num_agents(), 2);
        let observations = env.reset().unwrap();
        assert_eq!(observations.len(), 2);
        let actions = vec![
            Array1::from_vec(vec![0.1]), // up
            Array1::from_vec(vec![0.9]), // stay
        ];
        let (next_obs, rewards, dones, infos) = env.step(&actions).unwrap();
        assert_eq!(next_obs.len(), 2);
        assert_eq!(rewards.len(), 2);
        assert_eq!(dones.len(), 2);
        assert_eq!(infos.len(), 2);
    fn test_pursuit_evasion() {
        let mut env = PursuitEvasion::new(10.0, 10.0, 2, 1, 1.0, 0.5);
        assert_eq!(env.num_agents(), 3); // 2 pursuers + 1 evader
        assert_eq!(observations.len(), 3);
            Array1::from_vec(vec![0.5, 0.5]),  // pursuer 1
            Array1::from_vec(vec![-0.5, 0.0]), // pursuer 2
            Array1::from_vec(vec![0.0, -0.5]), // evader
        assert_eq!(next_obs.len(), 3);
        assert_eq!(rewards.len(), 3);
    fn test_multi_agent_wrapper() {
        let env = MultiAgentGridWorld::new(3, 3, 2, 1, false);
        let mut wrapper = MultiAgentWrapper::new(env, 0).unwrap();
        let obs = wrapper.reset().unwrap();
        assert!(obs.len() > 0);
        let action = Array1::from_vec(vec![0.5]);
        let (next_obs, reward, done, info) = wrapper.step(&action).unwrap();
        assert!(next_obs.len() > 0);
        assert!(reward.is_finite());
        assert!(info.contains_key("position_x"));
