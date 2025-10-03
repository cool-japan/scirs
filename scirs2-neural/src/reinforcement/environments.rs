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
/// Environment information
pub type Info = HashMap<String, f32>;
/// Base trait for reinforcement learning environments
pub trait Environment: Send + Sync {
    /// Reset the environment and return initial observation
    fn reset(&mut self) -> Result<Observation>;
    /// Take a step in the environment
    fn step(&mut self, action: &Action) -> Result<(Observation, Reward, bool, Info)>;
    /// Get observation space dimensions
    fn observation_space(&self) -> usize;
    /// Get action space dimensions
    fn action_space(&self) -> usize;
    /// Check if actions are continuous
    fn continuous_actions(&self) -> bool;
    /// Get action bounds for continuous actions
    fn action_bounds(&self) -> Option<(Array1<f32>, Array1<f32>)> {
        None
    }
    /// Render the environment (optional)
    fn render(&self) -> Result<()> {
        Ok(())
    /// Close the environment
    fn close(&mut self) -> Result<()> {
}
/// Classic CartPole environment
pub struct CartPole {
    state: Array1<f32>,
    steps: usize,
    max_steps: usize,
    gravity: f32,
    mass_cart: f32,
    mass_pole: f32,
    length: f32,
    force_mag: f32,
    tau: f32, // Time step
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
        }
impl CartPole {
    /// Create a new CartPole environment
    pub fn new() -> Self {
        Self::default()
    /// Check if episode is done
    fn is_done(&self) -> bool {
        let x = self.state[0];
        let theta = self.state[2];
        x < -2.4 || x > 2.4 || theta < -0.2095 || theta > 0.2095 || self.steps >= self.max_steps
impl Environment for CartPole {
    fn reset(&mut self) -> Result<Observation> {
        use scirs2_core::random::{Distribution, Uniform};
        let mut rng = rng();
        let uniform = Uniform::new(-0.05, 0.05);
        self.state = Array1::from_vec(vec![
            uniform.sample(&mut rng),
        ]);
        self.steps = 0;
        Ok(self.state.clone())
    fn step(&mut self, action: &Action) -> Result<(Observation, Reward, bool, Info)> {
        // Extract state variables
        let x_dot = self.state[1];
        let theta_dot = self.state[3];
        // Apply action (0 = left, 1 = right)
        let force = if action[0] > 0.5 {
            self.force_mag
        } else {
            -self.force_mag
        };
        // Physics calculations
        let cos_theta = theta.cos();
        let sin_theta = theta.sin();
        let total_mass = self.mass_cart + self.mass_pole;
        let pole_mass_length = self.mass_pole * self.length;
        let temp = (force + pole_mass_length * theta_dot * theta_dot * sin_theta) / total_mass;
        let theta_acc = (self.gravity * sin_theta - cos_theta * temp)
            / (self.length * (4.0 / 3.0 - self.mass_pole * cos_theta * cos_theta / total_mass));
        let x_acc = temp - pole_mass_length * theta_acc * cos_theta / total_mass;
        // Update state
        self.state[0] += self.tau * x_dot;
        self.state[1] += self.tau * x_acc;
        self.state[2] += self.tau * theta_dot;
        self.state[3] += self.tau * theta_acc;
        self.steps += 1;
        let done = self.is_done();
        let reward = if done { 0.0 } else { 1.0 };
        let mut info = HashMap::new();
        info.insert("steps".to_string(), self.steps as f32);
        Ok((self.state.clone(), reward, done, info))
    fn observation_space(&self) -> usize {
        4
    fn action_space(&self) -> usize {
        1 // Binary action encoded as continuous
    fn continuous_actions(&self) -> bool {
        false
/// Mountain Car environment
pub struct MountainCar {
    position: f32,
    velocity: f32,
impl Default for MountainCar {
            position: -0.5,
            velocity: 0.0,
impl MountainCar {
    /// Create a new MountainCar environment
impl Environment for MountainCar {
        self.position = Uniform::new(-0.6, -0.4).sample(&mut rng);
        self.velocity = 0.0;
        Ok(Array1::from_vec(vec![self.position, self.velocity]))
        // Action: 0 = left, 1 = nothing, 2 = right
        let action_value = if action[0] < 0.33 {
            -1.0
        } else if action[0] < 0.67 {
            0.0
            1.0
        // Update velocity and position
        self.velocity += 0.001 * action_value - 0.0025 * (3.0 * self.position).cos();
        self.velocity = self.velocity.clamp(-0.07, 0.07);
        self.position += self.velocity;
        self.position = self.position.clamp(-1.2, 0.6);
        // Reset velocity if hit the left boundary
        if self.position <= -1.2 {
            self.velocity = 0.0;
        // Check if reached the goal
        let done = self.position >= 0.5 || self.steps >= self.max_steps;
        let reward = if self.position >= 0.5 { 0.0 } else { -1.0 };
        info.insert("position".to_string(), self.position);
        info.insert("velocity".to_string(), self.velocity);
        Ok((
            Array1::from_vec(vec![self.position, self.velocity]),
            reward,
            done,
            info,
        ))
        2
        1 // Three discrete actions encoded as continuous
/// Continuous control pendulum environment
pub struct Pendulum {
    theta: f32,
    theta_dot: f32,
    max_torque: f32,
    dt: f32,
    g: f32,
    m: f32,
    l: f32,
impl Default for Pendulum {
            theta: 0.0,
            theta_dot: 0.0,
            max_torque: 2.0,
            dt: 0.05,
            g: 10.0,
            m: 1.0,
            l: 1.0,
impl Pendulum {
    /// Create a new Pendulum environment
    /// Normalize angle to [-pi, pi]
    fn angle_normalize(&self, x: f32) -> f32 {
        ((x + std::f32::consts::PI) % (2.0 * std::f32::consts::PI)) - std::f32::consts::PI
impl Environment for Pendulum {
        self.theta = Uniform::new(-std::f32::consts::PI, std::f32::consts::PI).sample(&mut rng);
        self.theta_dot = Uniform::new(-1.0, 1.0).sample(&mut rng);
        Ok(Array1::from_vec(vec![
            self.theta.cos(),
            self.theta.sin(),
            self.theta_dot,
        ]))
        let torque = action[0].clamp(-self.max_torque, self.max_torque);
        // Physics simulation
        let theta_acc = -3.0 * self.g / (2.0 * self.l) * self.theta.sin()
            + 3.0 / (self.m * self.l * self.l) * torque;
        self.theta_dot += theta_acc * self.dt;
        self.theta_dot = self.theta_dot.clamp(-8.0, 8.0);
        self.theta += self.theta_dot * self.dt;
        self.theta = self.angle_normalize(self.theta);
        // Reward is negative cost
        let cost = self.angle_normalize(self.theta).powi(2)
            + 0.1 * self.theta_dot.powi(2)
            + 0.001 * torque.powi(2);
        let reward = -cost;
        let done = self.steps >= self.max_steps;
        info.insert("theta".to_string(), self.theta);
        info.insert("theta_dot".to_string(), self.theta_dot);
        info.insert("torque".to_string(), torque);
            Array1::from_vec(vec![self.theta.cos(), self.theta.sin(), self.theta_dot]),
        3
        1
        true
        Some((
            Array1::from_vec(vec![-self.max_torque]),
            Array1::from_vec(vec![self.max_torque]),
/// Multi-environment wrapper for parallel execution
pub struct VectorizedEnvironment<E: Environment + Clone> {
    envs: Vec<E>,
    num_envs: usize,
impl<E: Environment + Clone> VectorizedEnvironment<E> {
    /// Create a new vectorized environment
    pub fn new(_envfn: impl Fn() -> E, num_envs: usize) -> Self {
        let envs = (0..num_envs).map(|_| _env_fn()).collect();
        Self { envs, num_envs }
    /// Reset all environments
    pub fn reset_all(&mut self) -> Result<Array2<f32>> {
        let obs_dim = self.envs[0].observation_space();
        let mut observations = Array2::zeros((self.num_envs, obs_dim));
        for (i, env) in self.envs.iter_mut().enumerate() {
            let obs = env.reset()?;
            observations.row_mut(i).assign(&obs);
        Ok(observations)
    /// Step all environments
    pub fn step_all(
        &mut self,
        actions: &ArrayView2<f32>,
    ) -> Result<(Array2<f32>, Array1<f32>, Array1<bool>, Vec<Info>)> {
        if actions.shape()[0] != self.num_envs {
            return Err(crate::error::NeuralError::InvalidArgument(format!(
                "Expected {} actions, got {}",
                self.num_envs,
                actions.shape()[0]
            )));
        let mut rewards = Array1::zeros(self.num_envs);
        let mut dones = Array1::from_elem(self.num_envs, false);
        let mut infos = Vec::with_capacity(self.num_envs);
            let action = actions.row(i).to_owned();
            let (obs, reward, done, info) = env.step(&action)?;
            rewards[i] = reward;
            dones[i] = done;
            infos.push(info);
            // Auto-reset if done
            if done {
                let new_obs = env.reset()?;
                observations.row_mut(i).assign(&new_obs);
            }
        Ok((observations, rewards, dones, infos))
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_cartpole() {
        let mut env = CartPole::new();
        let obs = env.reset().unwrap();
        assert_eq!(obs.len(), 4);
        let action = Array1::from_vec(vec![1.0]);
        let (next_obs, reward, done, info) = env.step(&action).unwrap();
        assert_eq!(next_obs.len(), 4);
        assert!(reward >= 0.0);
        assert!(info.contains_key("steps"));
    fn test_mountain_car() {
        let mut env = MountainCar::new();
        assert_eq!(obs.len(), 2);
        assert!(obs[0] >= -0.6 && obs[0] <= -0.4);
        let action = Array1::from_vec(vec![0.8]); // Go right
        let (_, reward__) = env.step(&action).unwrap();
        assert_eq!(reward, -1.0); // Not at goal yet
    fn test_pendulum() {
        let mut env = Pendulum::new();
        assert_eq!(obs.len(), 3);
        assert_eq!(env.observation_space(), 3);
        assert_eq!(env.action_space(), 1);
        assert!(env.continuous_actions());
        let bounds = env.action_bounds().unwrap();
        assert_eq!(bounds.0[0], -2.0);
        assert_eq!(bounds.1[0], 2.0);
    fn test_vectorized_env() {
        let mut vec_env = VectorizedEnvironment::new(CartPole::new, 4);
        let observations = vec_env.reset_all().unwrap();
        assert_eq!(observations.shape(), &[4, 4]);
        let actions = Array2::ones((4, 1));
        let (next_obs, rewards, dones, infos) = vec_env.step_all(&actions.view()).unwrap();
        assert_eq!(next_obs.shape(), &[4, 4]);
        assert_eq!(rewards.len(), 4);
        assert_eq!(dones.len(), 4);
        assert_eq!(infos.len(), 4);
