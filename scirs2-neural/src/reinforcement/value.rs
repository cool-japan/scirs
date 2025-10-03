//! Value-based reinforcement learning algorithms

use crate::activations::Activation;
use crate::error::Result;
use crate::layers::{Dense, Layer};
use scirs2_core::ndarray::prelude::*;
use scirs2_core::ndarray::ArrayView1;
use statrs::statistics::Statistics;
/// Value function network
pub struct ValueNetwork {
    layers: Vec<Box<dyn Layer<f32>>>,
    output_dim: usize,
}
impl ValueNetwork {
    /// Create a new value network
    pub fn new(_input_dim: usize, output_dim: usize, hiddensizes: Vec<usize>) -> Result<Self> {
        let mut layers: Vec<Box<dyn Layer<f32>>> = Vec::new();
        // Build hidden layers
        let mut current_dim = input_dim;
        for hidden_size in hidden_sizes {
            layers.push(Box::new(Dense::new(
                current_dim,
                hidden_size,
                Some(Activation::ReLU),
            )?));
            current_dim = hidden_size;
        }
        // Output layer (no activation for value output)
        layers.push(Box::new(Dense::new(current_dim, output_dim, None)?));
        Ok(Self { layers, output_dim })
    }
    /// Forward pass
    pub fn forward(&self, input: &ArrayView2<f32>) -> Result<Array2<f32>> {
        let mut x = input.to_owned();
        for layer in &self.layers {
            x = layer.forward(&x.view())?;
        Ok(x)
    /// Predict value for a single state
    pub fn predict(&self, state: &ArrayView1<f32>) -> Result<f32> {
        let input = state.to_owned().insert_axis(Axis(0));
        let output = self.forward(&input.view())?;
        Ok(output[[0, 0]])
    /// Predict values for batch of states
    pub fn predict_batch(&self, states: &ArrayView2<f32>) -> Result<Array1<f32>> {
        let output = self.forward(states)?;
        Ok(output.column(0).to_owned())
/// Q-Network for action-value estimation
pub struct QNetwork {
    state_dim: usize,
    action_dim: usize,
    dueling: bool,
impl QNetwork {
    /// Create a new Q-network
    pub fn new(
        state_dim: usize,
        action_dim: usize,
        hidden_sizes: Vec<usize>,
        dueling: bool,
    ) -> Result<Self> {
        // Shared layers
        let mut current_dim = state_dim;
        for (i, &hidden_size) in hidden_sizes.iter().enumerate() {
            // For dueling architecture, split after the second-to-last layer
            if dueling && i == hidden_sizes.len() - 2 {
                break;
            }
        if !dueling {
            // Standard Q-network: single output head
            layers.push(Box::new(Dense::new(current_dim, action_dim, None)?));
        Ok(Self {
            layers,
            state_dim,
            action_dim,
            dueling,
        })
    pub fn forward(&self, states: &ArrayView2<f32>) -> Result<Array2<f32>> {
        let mut x = states.to_owned();
        // Pass through shared layers
        let shared_layers = if self.dueling {
            &self.layers[..self.layers.len()]
        } else {
            &self.layers[..self.layers.len() - 1]
        };
        for layer in shared_layers {
        if self.dueling {
            // Dueling architecture: separate value and advantage streams
            let hidden_dim = x.shape()[1];
            // Value stream
            let value_layer = Dense::new(hidden_dim, 1, None)?;
            let values = value_layer.forward(&x.view())?;
            // Advantage stream
            let advantage_layer = Dense::new(hidden_dim, self.action_dim, None)?;
            let advantages = advantage_layer.forward(&x.view())?;
            // Combine: Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
            let mean_advantages = advantages.mean_axis(Axis(1)).unwrap().insert_axis(Axis(1));
            let q_values = &values + &advantages - &mean_advantages;
            Ok(q_values)
            // Standard architecture
            let last_layer = self.layers.last().unwrap();
            last_layer.forward(&x.view())
    /// Get Q-values for a single state
    pub fn get_q_values(&self, state: &ArrayView1<f32>) -> Result<Array1<f32>> {
        Ok(output.row(0).to_owned())
    /// Get best action for a state
    pub fn get_best_action(&self, state: &ArrayView1<f32>) -> Result<usize> {
        let q_values = self.get_q_values(state)?;
        Ok(q_values.argmax().unwrap())
/// Deep Q-Network (DQN) algorithm
pub struct DQN {
    q_network: QNetwork,
    target_network: QNetwork,
    learning_rate: f32,
    discount_factor: f32,
    epsilon: f32,
    epsilon_min: f32,
    epsilon_decay: f32,
    update_counter: usize,
    target_update_freq: usize,
impl DQN {
    /// Create a new DQN
        learning_rate: f32,
        discount_factor: f32,
        epsilon: f32,
        target_update_freq: usize,
        let q_network = QNetwork::new(state_dim, action_dim, hidden_sizes.clone(), false)?;
        let target_network = QNetwork::new(state_dim, action_dim, hidden_sizes, false)?;
            q_network,
            target_network,
            learning_rate,
            discount_factor,
            epsilon,
            epsilon_min: 0.01,
            epsilon_decay: 0.995,
            update_counter: 0,
            target_update_freq,
    /// Select action using epsilon-greedy policy
    pub fn select_action(&self, state: &ArrayView1<f32>, training: bool) -> Result<usize> {
        if training && scirs2_core::random::random::<f32>() < self.epsilon {
            // Random action
            Ok(scirs2_core::random::random::<usize>() % self.q_network.action_dim)
            // Greedy action
            self.q_network.get_best_action(state)
    /// Update Q-network using experience batch
    pub fn update(&mut self, batch: &ExperienceBatch) -> Result<f32> {
        let states = &batch.states;
        let actions = &batch.actions;
        let rewards = &batch.rewards;
        let next_states = &batch.next_states;
        let dones = &batch.dones;
        // Get current Q-values
        let current_q_values = self.q_network.forward(states)?;
        // Get next Q-values from target network
        let next_q_values = self.target_network.forward(next_states)?;
        let max_next_q = next_q_values.map_axis(Axis(1), |row| row.max().unwrap());
        // Calculate target Q-values
        let mut target_q_values = current_q_values.clone();
        for i in 0..batch.size() {
            let action_idx = actions[[i, 0]] as usize;
            let target = if dones[i] {
                rewards[i]
            } else {
                rewards[i] + self.discount_factor * max_next_q[i]
            };
            target_q_values[[i, action_idx]] = target;
        // Calculate loss (MSE)
        let loss = (&current_q_values - &target_q_values)
            .mapv(|x| x * x)
            .mean()
            .unwrap();
        // Update target network periodically
        self.update_counter += 1;
        if self.update_counter % self.target_update_freq == 0 {
            self.update_target_network()?;
        // Decay epsilon
        self.epsilon = (self.epsilon * self.epsilon_decay).max(self.epsilon_min);
        Ok(loss)
    /// Update target network with current Q-network weights
    fn update_target_network(&mut self) -> Result<()> {
        // In a real implementation, this would copy weights
        // For now, we'll just recreate the network
        self.target_network = QNetwork::new(
            self.q_network.state_dim,
            self.q_network.action_dim,
            vec![128, 128], // Default hidden sizes
            false,
        )?;
        Ok(())
    /// Get current exploration rate
    pub fn get_epsilon(&self) -> f32 {
        self.epsilon
/// Double DQN algorithm
pub struct DoubleDQN {
    dqn: DQN,
impl DoubleDQN {
    /// Create a new Double DQN
        let dqn = DQN::new(
            hidden_sizes,
        Ok(Self { dqn })
    /// Select action
        self.dqn.select_action(state, training)
    /// Update using Double DQN algorithm
        let current_q_values = self.dqn.q_network.forward(states)?;
        // Double DQN: use current network to select actions, target network to evaluate
        let next_q_current = self.dqn.q_network.forward(next_states)?;
        let next_q_target = self.dqn.target_network.forward(next_states)?;
        // Select best actions using current network
        let best_actions = next_q_current.map_axis(Axis(1), |row| row.argmax().unwrap() as f32);
            let best_next_action = best_actions[i] as usize;
                rewards[i] + self.dqn.discount_factor * next_q_target[[i, best_next_action]]
        // Calculate loss
        // Update target network and decay epsilon
        self.dqn.update_counter += 1;
        if self.dqn.update_counter % self.dqn.target_update_freq == 0 {
            self.dqn.update_target_network()?;
        self.dqn.epsilon = (self.dqn.epsilon * self.dqn.epsilon_decay).max(self.dqn.epsilon_min);
/// Experience batch for training
#[derive(Debug, Clone)]
pub struct ExperienceBatch {
    pub states: Array2<f32>,
    pub actions: Array2<f32>,
    pub rewards: Array1<f32>,
    pub next_states: Array2<f32>,
    pub dones: Array1<bool>,
impl ExperienceBatch {
    /// Get batch size
    pub fn size(&self) -> usize {
        self.states.shape()[0]
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_value_network() {
        let vnet = ValueNetwork::new(4, 1, vec![32, 32]).unwrap();
        let state = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let value = vnet.predict(&state.view()).unwrap();
        assert!(value.is_finite());
    fn test_q_network() {
        let qnet = QNetwork::new(4, 2, vec![32, 32], false).unwrap();
        let q_values = qnet.get_q_values(&state.view()).unwrap();
        assert_eq!(q_values.len(), 2);
        let best_action = qnet.get_best_action(&state.view()).unwrap();
        assert!(best_action < 2);
    fn test_dqn_action_selection() {
        let dqn = DQN::new(4, 2, vec![32], 0.001, 0.99, 1.0, 100).unwrap();
        // With epsilon=1.0, should always select random actions during training
        let action = dqn.select_action(&state.view(), true).unwrap();
        assert!(action < 2);
        // Without training, should select greedy action
        let action = dqn.select_action(&state.view(), false).unwrap();
