//! Experience replay buffers for reinforcement learning

use crate::error::Result;
use crate::reinforcement::ExperienceBatch;
use scirs2_core::ndarray::prelude::*;
use scirs2_core::random::seq::SliceRandom;
use std::collections::VecDeque;
/// Trait for experience replay buffers
pub trait ReplayBufferTrait: Send + Sync {
    /// Add an experience to the buffer
    fn add(
        &mut self,
        state: Array1<f32>,
        action: Array1<f32>,
        reward: f32,
        next_state: Array1<f32>,
        done: bool,
    ) -> Result<()>;
    /// Sample a batch of experiences (returns basic batch for common interface)
    fn sample_batch(&self, batchsize: usize) -> Result<ExperienceBatch>;
    /// Get the current size of the buffer
    fn len(&self) -> usize;
    /// Check if the buffer is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Get the capacity of the buffer
    fn capacity(&self) -> usize;
}
/// Experience tuple for storage
#[derive(Clone, Debug)]
pub struct Experience {
    pub state: Array1<f32>,
    pub action: Array1<f32>,
    pub reward: f32,
    pub next_state: Array1<f32>,
    pub done: bool,
    pub info: Option<std::collections::HashMap<String, f32>>,
/// Simple replay buffer (alias for ReplayBuffer for compatibility)
pub type SimpleReplayBuffer = ReplayBuffer;
/// Standard replay buffer
pub struct ReplayBuffer {
    buffer: VecDeque<Experience>,
    capacity: usize,
impl ReplayBuffer {
    /// Create a new replay buffer
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(_capacity),
            capacity,
        }
    pub fn add(
    ) -> Result<()> {
        let experience = Experience {
            state,
            action,
            reward,
            next_state,
            done,
            info: None,
        };
        if self.buffer.len() >= self.capacity {
            self.buffer.pop_front();
        self.buffer.push_back(experience);
        Ok(())
    /// Add an experience with additional info
    pub fn add_with_info(
        info: std::collections::HashMap<String, f32>,
            info: Some(info),
    /// Sample a batch of experiences
    pub fn sample(&self, batchsize: usize) -> Result<ExperienceBatch> {
        if self.buffer.len() < batch_size {
            return Err(crate::error::NeuralError::InvalidArgument(format!(
                "Not enough experiences in buffer: {} < {}",
                self.buffer.len(),
                batch_size
            )));
        let mut rng = rng();
        let indices: Vec<usize> = (0..self.buffer.len())
            .collect::<Vec<_>>()
            .choose_multiple(&mut rng, batch_size)
            .cloned()
            .collect();
        self.create_batch(&indices)
    /// Get the last n experiences
    pub fn get_last_n(&self, n: usize) -> Result<ExperienceBatch> {
        let n = n.min(self.buffer.len());
        let indices: Vec<usize> = (self.buffer.len() - n..self.buffer.len()).collect();
    /// Create a batch from indices
    fn create_batch(&self, indices: &[usize]) -> Result<ExperienceBatch> {
        if indices.is_empty() {
            return Err(crate::error::NeuralError::InvalidArgument(
                "Cannot create batch from empty indices".to_string(),
            ));
        let state_dim = self.buffer[indices[0]].state.len();
        let action_dim = self.buffer[indices[0]].action.len();
        let batch_size = indices.len();
        let mut states = Array2::zeros((batch_size, state_dim));
        let mut actions = Array2::zeros((batch_size, action_dim));
        let mut rewards = Array1::zeros(batch_size);
        let mut next_states = Array2::zeros((batch_size, state_dim));
        let mut dones = Array1::from_elem(batch_size, false);
        for (i, &idx) in indices.iter().enumerate() {
            let exp = &self.buffer[idx];
            states.row_mut(i).assign(&exp.state);
            actions.row_mut(i).assign(&exp.action);
            rewards[i] = exp.reward;
            next_states.row_mut(i).assign(&exp.next_state);
            dones[i] = exp.done;
        Ok(ExperienceBatch {
            states,
            actions,
            rewards,
            next_states,
            dones,
        })
    /// Get current buffer size
    pub fn len(&self) -> usize {
        self.buffer.len()
    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    /// Clear the buffer
    pub fn clear(&mut self) {
        self.buffer.clear();
impl ReplayBufferTrait for ReplayBuffer {
        self.add(state, action, reward, next_state, done)
    fn sample_batch(&self, batchsize: usize) -> Result<ExperienceBatch> {
        self.sample(batch_size)
    fn len(&self) -> usize {
    fn capacity(&self) -> usize {
        self.capacity
/// Prioritized experience replay buffer
pub struct PrioritizedReplayBuffer {
    buffer: Vec<Experience>,
    priorities: Vec<f32>,
    alpha: f32,   // Priority exponent
    beta: f32,    // Importance sampling exponent
    epsilon: f32, // Small constant to avoid zero priorities
    max_priority: f32,
    ptr: usize,
impl PrioritizedReplayBuffer {
    /// Create a new prioritized replay buffer
    pub fn new(capacity: usize, alpha: f32, beta: f32) -> Self {
            buffer: Vec::with_capacity(_capacity),
            priorities: vec![1.0; _capacity],
            alpha,
            beta,
            epsilon: 1e-6,
            max_priority: 1.0,
            ptr: 0,
    /// Add an experience with maximum priority
        if self.buffer.len() < self._capacity {
            self.buffer.push(experience);
        } else {
            self.buffer[self.ptr] = experience;
        // Assign maximum priority to new experience
        self.priorities[self.ptr] = self.max_priority.powf(self.alpha);
        self.ptr = (self.ptr + 1) % self.capacity;
    /// Sample a batch with importance sampling weights
    pub fn sample(&self, batchsize: usize) -> Result<(ExperienceBatch, Array1<f32>, Vec<usize>)> {
        // Calculate sampling probabilities
        let priorities = &self.priorities[..self.buffer.len()];
        let probs = self.calculate_probabilities(priorities);
        // Sample indices based on priorities
        let indices = self.sample_indices(&probs, batch_size)?;
        // Calculate importance sampling weights
        let weights = self.calculate_is_weights(&probs, &indices, batch_size);
        // Create batch
        let batch = self.create_batch(&indices)?;
        Ok((batch, weights, indices))
    /// Update priorities for sampled experiences
    pub fn update_priorities(&mut self, indices: &[usize], tderrors: &[f32]) -> Result<()> {
        if indices.len() != td_errors.len() {
                "Indices and TD errors must have the same length".to_string(),
        for (&idx, &td_error) in indices.iter().zip(td_errors.iter()) {
            let priority = (td_error.abs() + self.epsilon).powf(self.alpha);
            self.priorities[idx] = priority;
            self.max_priority = self.max_priority.max(priority);
    /// Calculate sampling probabilities
    fn calculate_probabilities(&self, priorities: &[f32]) -> Vec<f32> {
        let sum: f32 = priorities.iter().sum();
        priorities.iter().map(|&p| p / sum).collect()
    /// Sample indices based on probabilities
    fn sample_indices(&self, probs: &[f32], batchsize: usize) -> Result<Vec<usize>> {
        use scirs2_core::random::prelude::*;
        use scirs2_core::random::weighted::WeightedIndex;
        let dist = WeightedIndex::new(probs).map_err(|e| {
            crate::error::NeuralError::InvalidArgument(format!("Invalid weights: {}", e))
        })?;
        let indices: Vec<usize> = (0..batch_size).map(|_| dist.sample(&mut rng)).collect();
        Ok(indices)
    /// Calculate importance sampling weights
    fn calculate_is_weights(
        &self,
        probs: &[f32],
        indices: &[usize],
        batch_size: usize,
    ) -> Array1<f32> {
        let n = self.buffer.len() as f32;
        let min_prob = probs
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(&1.0);
        let max_weight = (min_prob * n).powf(-self.beta);
        let mut weights = Array1::zeros(batch_size);
            let weight = (probs[idx] * n).powf(-self.beta) / max_weight;
            weights[i] = weight;
        weights
    /// Increase beta for importance sampling
    pub fn update_beta(&mut self, beta: f32) {
        self.beta = beta.min(1.0);
impl ReplayBufferTrait for PrioritizedReplayBuffer {
        // For common interface, just return the batch part of the prioritized sample
        let (batch_weights_indices) = self.sample(batch_size)?;
        Ok(batch)
/// N-step replay buffer for multi-step returns
pub struct NStepReplayBuffer {
    buffer: ReplayBuffer,
    n_step_buffer: VecDeque<Experience>,
    n: usize,
    gamma: f32,
impl NStepReplayBuffer {
    /// Create a new n-step replay buffer
    pub fn new(capacity: usize, n: usize, gamma: f32) -> Self {
            buffer: ReplayBuffer::new(_capacity),
            n_step_buffer: VecDeque::with_capacity(n),
            n,
            gamma,
    /// Add an experience
        self.n_step_buffer.push_back(experience);
        // If we have enough experiences, compute n-step return and add to main buffer
        if self.n_step_buffer.len() >= self.n || done {
            self.add_n_step_experience()?;
        // Clear n-step buffer if episode ended
        if done {
            self.n_step_buffer.clear();
    /// Compute n-step return and add to main buffer
    fn add_n_step_experience(&mut self) -> Result<()> {
        if self.n_step_buffer.is_empty() {
            return Ok(());
        let first_exp = self.n_step_buffer.front().unwrap().clone();
        let mut n_step_reward = 0.0;
        let mut gamma_power = 1.0;
        let mut final_next_state = first_exp.next_state.clone();
        let mut final_done = first_exp.done;
        // Calculate n-step return
        for (i, exp) in self.n_step_buffer.iter().enumerate() {
            n_step_reward += gamma_power * exp.reward;
            gamma_power *= self.gamma;
            if i == self.n_step_buffer.len() - 1 || exp.done {
                final_next_state = exp.next_state.clone();
                final_done = exp.done;
                break;
            }
        // Add n-step experience to main buffer
        self.buffer.add(
            first_exp.state,
            first_exp.action,
            n_step_reward,
            final_next_state,
            final_done,
        )?;
        // Remove first experience from n-step buffer
        self.n_step_buffer.pop_front();
    /// Sample from the buffer
        self.buffer.sample(batch_size)
    /// Get buffer size
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_replay_buffer() {
        let mut buffer = ReplayBuffer::new(100);
        // Add experiences
        for i in 0..10 {
            let state = Array1::from_vec(vec![i as f32; 4]);
            let action = Array1::from_vec(vec![i as f32]);
            let next_state = Array1::from_vec(vec![(i + 1) as f32; 4]);
            buffer
                .add(state, action, i as f32, next_state, false)
                .unwrap();
        assert_eq!(buffer.len(), 10);
        // Sample batch
        let batch = buffer.sample(5).unwrap();
        assert_eq!(batch.states.shape(), &[5, 4]);
        assert_eq!(batch.actions.shape(), &[5, 1]);
        assert_eq!(batch.rewards.len(), 5);
    fn test_prioritized_replay_buffer() {
        let mut buffer = PrioritizedReplayBuffer::new(100, 0.6, 0.4);
        // Sample with importance weights
        let (batch, weights, indices) = buffer.sample(5).unwrap();
        assert_eq!(weights.len(), 5);
        assert_eq!(indices.len(), 5);
        // Update priorities
        let td_errors = vec![0.1, 0.5, 0.2, 0.8, 0.3];
        buffer.update_priorities(&indices, &td_errors).unwrap();
    fn test_n_step_replay_buffer() {
        let mut buffer = NStepReplayBuffer::new(100, 3, 0.99);
            buffer.add(state, action, 1.0, next_state, i == 9).unwrap();
        // Should have computed n-step returns
        assert!(buffer.len() > 0);
        assert!(buffer.len() <= 8); // Some experiences are still in n-step buffer
