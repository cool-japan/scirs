//! Model-based reinforcement learning algorithms
//!
//! This module implements algorithms that learn a model of the environment
//! and use it for planning and decision making.

use crate::activations::Activation;
use crate::error::{NeuralError, Result};
use crate::layers::{Dense, Layer};
use crate::reinforcement::environments::Environment;
use scirs2_core::ndarray::concatenate;
use scirs2_core::ndarray::prelude::*;
use scirs2_core::random::seq::SliceRandom;
use scirs2_core::ndarray::ArrayView1;
/// Dynamics model for predicting environment transitions
pub struct DynamicsModel {
    state_dim: usize,
    action_dim: usize,
    layers: Vec<Box<dyn Layer<f32>>>,
    reward_head: Box<dyn Layer<f32>>,
    uncertainty_estimation: bool,
}
impl DynamicsModel {
    /// Create a new dynamics model
    pub fn new(
        state_dim: usize,
        action_dim: usize,
        hidden_sizes: Vec<usize>,
        uncertainty_estimation: bool,
    ) -> Result<Self> {
        let mut layers: Vec<Box<dyn Layer<f32>>> = Vec::new();
        let input_dim = state_dim + action_dim;
        let mut current_dim = input_dim;
        // Hidden layers
        for hidden_size in hidden_sizes {
            layers.push(Box::new(Dense::new(
                current_dim,
                hidden_size,
                Some(Activation::ReLU),
            )?));
            current_dim = hidden_size;
        }
        // Output next state (and uncertainty if enabled)
        let output_dim = if uncertainty_estimation {
            state_dim * 2 // Mean and variance
        } else {
            state_dim
        };
        layers.push(Box::new(Dense::new(current_dim, output_dim, None)?));
        // Separate head for reward prediction
        let reward_head = Box::new(Dense::new(current_dim, 1, None)?);
        Ok(Self {
            state_dim,
            action_dim,
            layers,
            reward_head,
            uncertainty_estimation,
        })
    }
    /// Predict next state and reward
    pub fn predict(
        &self,
        state: &ArrayView1<f32>,
        action: &ArrayView1<f32>,
    ) -> Result<(Array1<f32>, f32, Option<Array1<f32>>)> {
        // Concatenate state and action
        let input = concatenate![Axis(0), *state, *action];
        let mut x = input.insert_axis(Axis(0));
        // Forward through main layers (except last)
        for (i, layer) in self.layers[..self.layers.len() - 1].iter().enumerate() {
            x = layer.forward(&x.view())?;
        // Get features for reward prediction
        let features = x.clone();
        // Predict next state
        let state_output = self.layers.last().unwrap().forward(&x.view())?;
        let state_output = state_output.remove_axis(Axis(0));
        let (next_state, uncertainty) = if self.uncertainty_estimation {
            let mean = state_output.slice(s![..self.state_dim]).to_owned();
            let log_var = state_output.slice(s![self.state_dim..]).to_owned();
            let std = log_var.mapv(|x| (x / 2.0).exp());
            (mean, Some(std))
            (state_output, None)
        // Predict reward
        let reward_output = self.reward_head.forward(&features.view())?;
        let reward = reward_output[[0, 0]];
        Ok((next_state, reward, uncertainty))
    /// Train the model on a batch of transitions
    pub fn update(
        &mut self,
        states: &ArrayView2<f32>,
        actions: &ArrayView2<f32>,
        next_states: &ArrayView2<f32>,
        rewards: &ArrayView1<f32>,
    ) -> Result<(f32, f32)> {
        let batch_size = states.shape()[0];
        let mut state_loss = 0.0;
        let mut reward_loss = 0.0;
        for i in 0..batch_size {
            let (pred_next_state, pred_reward_) =
                self.predict(&states.row(i), &actions.row(i))?;
            // State prediction loss
            let state_error = &pred_next_state - &next_states.row(i);
            state_loss += state_error.mapv(|x| x.powi(2)).sum();
            // Reward prediction loss
            reward_loss += (pred_reward - rewards[i]).powi(2);
        state_loss /= batch_size as f32;
        reward_loss /= batch_size as f32;
        Ok((state_loss, reward_loss))
/// Model Predictive Control (MPC) planner
pub struct MPC {
    dynamics_model: DynamicsModel,
    horizon: usize,
    num_simulations: usize,
    action_bounds: Option<(Array1<f32>, Array1<f32>)>,
impl MPC {
    /// Create a new MPC planner
        dynamics_model: DynamicsModel,
        horizon: usize,
        num_simulations: usize,
        action_bounds: Option<(Array1<f32>, Array1<f32>)>,
    ) -> Self {
        Self {
            dynamics_model,
            horizon,
            num_simulations,
            action_bounds,
    /// Plan actions using random shooting
    pub fn plan(&self, initialstate: &ArrayView1<f32>) -> Result<Array1<f32>> {
        let mut best_action_sequence = vec![Array1::zeros(self.action_dim); self.horizon];
        let mut best_reward = f32::NEG_INFINITY;
        for _ in 0..self.num_simulations {
            // Sample random action sequence
            let action_sequence = self.sample_action_sequence()?;
            // Simulate trajectory
            let total_reward = self.simulate_trajectory(initial_state, &action_sequence)?;
            // Update best sequence
            if total_reward > best_reward {
                best_reward = total_reward;
                best_action_sequence = action_sequence;
            }
        // Return first action of best sequence
        Ok(best_action_sequence[0].clone())
    /// Sample a random action sequence
    fn sample_action_sequence(&self) -> Result<Vec<Array1<f32>>> {
        use scirs2_core::random::{Distribution, Uniform};
        let mut rng = rng();
        let mut sequence = Vec::with_capacity(self.horizon);
        for _ in 0..self.horizon {
            let mut action = Array1::zeros(self.action_dim);
            if let Some((lower, upper)) = &self.action_bounds {
                for i in 0..self.action_dim {
                    let dist = Uniform::new(lower[i], upper[i]);
                    action[i] = dist.sample(&mut rng);
                }
            } else {
                // Sample from standard normal
                    let dist = scirs2_core::random::Normal::new(0.0, 1.0).map_err(|e| {
                        NeuralError::InvalidArgument(format!("Invalid normal distribution: {}", e))
                    })?;
            sequence.push(action);
        Ok(sequence)
    /// Simulate a trajectory with given action sequence
    fn simulate_trajectory(
        initial_state: &ArrayView1<f32>,
        action_sequence: &[Array1<f32>],
    ) -> Result<f32> {
        let mut state = initial_state.to_owned();
        let mut total_reward = 0.0;
        let mut discount = 1.0;
        for action in action_sequence {
            let (next_state, reward_) =
                self.dynamics_model.predict(&state.view(), &action.view())?;
            total_reward += discount * reward;
            discount *= 0.99; // Discount factor
            state = next_state;
        Ok(total_reward)
/// Dyna-style model-based RL algorithm
pub struct Dyna<P: Policy> {
    policy: P,
    planning_steps: usize,
    model_buffer: ModelBuffer,
impl<P: Policy> Dyna<P> {
    /// Create a new Dyna agent
        policy: P,
        planning_steps: usize,
        buffer_size: usize,
            policy,
            planning_steps,
            model_buffer: ModelBuffer::new(buffer_size),
    /// Update from real experience
    pub fn update_from_experience(
        reward: f32,
        next_state: &ArrayView1<f32>,
        done: bool,
    ) -> Result<()> {
        // Add to model buffer
        self.model_buffer.add(
            state.to_owned(),
            action.to_owned(),
            reward,
            next_state.to_owned(),
        )?;
        // Update dynamics model
        if self.model_buffer.len() >= 32 {
            let batch = self.model_buffer.sample(32)?;
            self.dynamics_model.update(
                &batch.states.view(),
                &batch.actions.view(),
                &batch.next_states.view(),
                &batch.rewards.view(),
            )?;
        // Planning: simulate experiences using the model
        for _ in 0..self.planning_steps {
            self.planning_step()?;
        Ok(())
    /// Perform one planning step
    fn planning_step(&mut self) -> Result<()> {
        if self.model_buffer.is_empty() {
            return Ok(());
        // Sample a state from buffer
        let (state___) = self.model_buffer.sample_single()?;
        // Sample action from policy
        let action = self.policy.sample_action(&state.view())?;
        // Predict next state and reward using model
        let (next_state, reward_) = self.dynamics_model.predict(&state.view(), &action.view())?;
        // Update policy using simulated experience
        // (This would integrate with the policy's update mechanism)
/// Buffer for storing model training data
struct ModelBuffer {
    states: Vec<Array1<f32>>,
    actions: Vec<Array1<f32>>,
    rewards: Vec<f32>,
    next_states: Vec<Array1<f32>>,
    capacity: usize,
    ptr: usize,
impl ModelBuffer {
    fn new(capacity: usize) -> Self {
            states: Vec::with_capacity(_capacity),
            actions: Vec::with_capacity(_capacity),
            rewards: Vec::with_capacity(_capacity),
            next_states: Vec::with_capacity(_capacity),
            capacity,
            ptr: 0,
    fn add(
        state: Array1<f32>,
        action: Array1<f32>,
        next_state: Array1<f32>,
        if self.states.len() < self._capacity {
            self.states.push(state);
            self.actions.push(action);
            self.rewards.push(reward);
            self.next_states.push(next_state);
            self.states[self.ptr] = state;
            self.actions[self.ptr] = action;
            self.rewards[self.ptr] = reward;
            self.next_states[self.ptr] = next_state;
        self.ptr = (self.ptr + 1) % self._capacity;
    fn sample(&self, batchsize: usize) -> Result<ModelBatch> {
        let indices: Vec<usize> = (0..self.len())
            .collect::<Vec<_>>()
            .choose_multiple(&mut rng, batch_size)
            .cloned()
            .collect();
        let state_dim = self.states[0].len();
        let action_dim = self.actions[0].len();
        let mut states = Array2::zeros((batch_size, state_dim));
        let mut actions = Array2::zeros((batch_size, action_dim));
        let mut rewards = Array1::zeros(batch_size);
        let mut next_states = Array2::zeros((batch_size, state_dim));
        for (i, &idx) in indices.iter().enumerate() {
            states.row_mut(i).assign(&self.states[idx]);
            actions.row_mut(i).assign(&self.actions[idx]);
            rewards[i] = self.rewards[idx];
            next_states.row_mut(i).assign(&self.next_states[idx]);
        Ok(ModelBatch {
            states,
            actions,
            rewards,
            next_states,
    fn sample_single(&self) -> Result<(Array1<f32>, Array1<f32>, f32, Array1<f32>)> {
        if self.is_empty() {
            return Err(NeuralError::InvalidArgument("Buffer is empty".to_string()));
        let idx = scirs2_core::random::random::<usize>() % self.len();
        Ok((
            self.states[idx].clone(),
            self.actions[idx].clone(),
            self.rewards[idx],
            self.next_states[idx].clone(),
        ))
    fn len(&self) -> usize {
        self.states.len()
    fn is_empty(&self) -> bool {
        self.states.is_empty()
/// Batch of model training data
struct ModelBatch {
    states: Array2<f32>,
    actions: Array2<f32>,
    rewards: Array1<f32>,
    next_states: Array2<f32>,
/// Trait for policies used in model-based RL
pub trait Policy: Send + Sync {
    fn sample_action(&self, state: &ArrayView1<f32>) -> Result<Array1<f32>>;
/// World Models for planning in latent space
pub struct WorldModel {
    encoder: Encoder,
    dynamics: LatentDynamics,
    decoder: Decoder,
    latent_dim: usize,
impl WorldModel {
    /// Create a new world model
        latent_dim: usize,
        let encoder = Encoder::new(state_dim, latent_dim, hidden_sizes.clone())?;
        let dynamics = LatentDynamics::new(latent_dim, action_dim, hidden_sizes.clone())?;
        let decoder = Decoder::new(latent_dim, state_dim, hidden_sizes)?;
            encoder,
            dynamics,
            decoder,
            latent_dim,
    /// Encode state to latent representation
    pub fn encode(&self, state: &ArrayView1<f32>) -> Result<Array1<f32>> {
        self.encoder.encode(state)
    /// Decode latent representation to state
    pub fn decode(&self, latent: &ArrayView1<f32>) -> Result<Array1<f32>> {
        self.decoder.decode(latent)
    /// Predict next latent state
    pub fn predict_latent(
        latent: &ArrayView1<f32>,
    ) -> Result<(Array1<f32>, f32)> {
        self.dynamics.predict(latent, action)
    /// Full prediction: state -> latent -> next latent -> next state
        let latent = self.encode(state)?;
        let (next_latent, reward) = self.predict_latent(&latent.view(), action)?;
        let next_state = self.decode(&next_latent.view())?;
        Ok((next_state, reward))
/// Encoder for world model
struct Encoder {
impl Encoder {
    fn new(_input_dim: usize, latent_dim: usize, hiddensizes: Vec<usize>) -> Result<Self> {
        layers.push(Box::new(Dense::new(current_dim, latent_dim, None)?));
        Ok(Self { layers })
    fn encode(&self, state: &ArrayView1<f32>) -> Result<Array1<f32>> {
        let mut x = state.to_owned().insert_axis(Axis(0));
        for layer in &self.layers {
        Ok(x.remove_axis(Axis(0)))
/// Latent dynamics model
struct LatentDynamics {
impl LatentDynamics {
    fn new(_latent_dim: usize, action_dim: usize, hiddensizes: Vec<usize>) -> Result<Self> {
        let input_dim = latent_dim + action_dim;
    fn predict(
        let input = concatenate![Axis(0), *latent, *action];
        for layer in &self.layers[..self.layers.len() - 1] {
        // Predict next latent state
        let next_latent = self.layers.last().unwrap().forward(&x.view())?;
        let next_latent = next_latent.remove_axis(Axis(0));
        let reward = self.reward_head.forward(&features.view())?;
        let reward = reward[[0, 0]];
        Ok((next_latent, reward))
/// Decoder for world model
struct Decoder {
impl Decoder {
    fn new(_latent_dim: usize, output_dim: usize, hiddensizes: Vec<usize>) -> Result<Self> {
        let mut current_dim = latent_dim;
    fn decode(&self, latent: &ArrayView1<f32>) -> Result<Array1<f32>> {
        let mut x = latent.to_owned().insert_axis(Axis(0));
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_dynamics_model() {
        let model = DynamicsModel::new(4, 2, vec![32, 32], false).unwrap();
        let state = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let action = Array1::from_vec(vec![0.5, -0.5]);
        let (next_state, reward, uncertainty) =
            model.predict(&state.view(), &action.view()).unwrap();
        assert_eq!(next_state.len(), 4);
        assert!(reward.is_finite());
        assert!(uncertainty.is_none());
    fn test_mpc_planner() {
        let dynamics_model = DynamicsModel::new(4, 2, vec![32], false).unwrap();
        let mpc = MPC::new(dynamics_model, 10, 100, 2, None);
        let action = mpc.plan(&state.view()).unwrap();
        assert_eq!(action.len(), 2);
    fn test_world_model() {
        let world_model = WorldModel::new(4, 2, 8, vec![32]).unwrap();
        // Test encoding/decoding
        let latent = world_model.encode(&state.view()).unwrap();
        assert_eq!(latent.len(), 8);
        let reconstructed = world_model.decode(&latent.view()).unwrap();
        assert_eq!(reconstructed.len(), 4);
        // Test prediction
        let (next_state, reward) = world_model.predict(&state.view(), &action.view()).unwrap();
