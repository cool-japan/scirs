//! Curiosity-driven exploration for reinforcement learning
//!
//! This module implements intrinsic curiosity modules (ICM) and other
//! curiosity-based exploration strategies to improve exploration in RL.

use crate::activations::Activation;
use crate::error::Result;
use crate::layers::{Dense, Layer};
use scirs2_core::ndarray::concatenate;
use scirs2_core::ndarray::prelude::*;
use std::collections::VecDeque;
use scirs2_core::ndarray::ArrayView1;
/// Intrinsic Curiosity Module (ICM)
///
/// ICM consists of two models:
/// - Forward model: predicts next state given current state and action
/// - Inverse model: predicts action given current and next state
pub struct ICM {
    forward_model: ForwardModel,
    inverse_model: InverseModel,
    feature_encoder: FeatureEncoder,
    eta: f32,  // Scaling factor for intrinsic reward
    beta: f32, // Weight between forward and inverse loss
}
impl ICM {
    /// Create a new ICM
    pub fn new(
        state_dim: usize,
        action_dim: usize,
        feature_dim: usize,
        hidden_sizes: Vec<usize>,
        eta: f32,
        beta: f32,
    ) -> Result<Self> {
        let feature_encoder = FeatureEncoder::new(state_dim, feature_dim, hidden_sizes.clone())?;
        let forward_model =
            ForwardModel::new(feature_dim, action_dim, feature_dim, hidden_sizes.clone())?;
        let inverse_model = InverseModel::new(feature_dim, feature_dim, action_dim, hidden_sizes)?;
        Ok(Self {
            forward_model,
            inverse_model,
            feature_encoder,
            eta,
            beta,
        })
    }
    /// Compute intrinsic reward
    pub fn compute_intrinsic_reward(
        &self,
        state: &ArrayView1<f32>,
        action: &ArrayView1<f32>,
        next_state: &ArrayView1<f32>,
    ) -> Result<f32> {
        // Encode states to feature space
        let phi_s = self.feature_encoder.encode(state)?;
        let phi_s_next = self.feature_encoder.encode(next_state)?;
        // Predict next feature using forward model
        let phi_s_next_pred = self.forward_model.predict(&phi_s.view(), action)?;
        // Compute prediction error as intrinsic reward
        let error = (&phi_s_next - &phi_s_next_pred).mapv(|x| x.powi(2)).sum();
        Ok(self.eta * error)
    /// Update ICM models
    pub fn update(
        &mut self,
        states: &ArrayView2<f32>,
        actions: &ArrayView2<f32>,
        next_states: &ArrayView2<f32>,
    ) -> Result<(f32, f32)> {
        let batch_size = states.shape()[0];
        let mut forward_loss = 0.0;
        let mut inverse_loss = 0.0;
        for i in 0..batch_size {
            let state = states.row(i);
            let action = actions.row(i);
            let next_state = next_states.row(i);
            // Encode states
            let phi_s = self.feature_encoder.encode(&state)?;
            let phi_s_next = self.feature_encoder.encode(&next_state)?;
            // Forward model loss
            let phi_s_next_pred = self.forward_model.predict(&phi_s.view(), &action)?;
            forward_loss += (&phi_s_next - &phi_s_next_pred).mapv(|x| x.powi(2)).sum();
            // Inverse model loss
            let action_pred = self
                .inverse_model
                .predict(&phi_s.view(), &phi_s_next.view())?;
            inverse_loss += (&action - &action_pred).mapv(|x| x.powi(2)).sum();
        }
        forward_loss /= batch_size as f32;
        inverse_loss /= batch_size as f32;
        // Combined loss
        let total_loss = self.beta * forward_loss + (1.0 - self.beta) * inverse_loss;
        Ok((forward_loss, inverse_loss))
/// Feature encoder for state representation
struct FeatureEncoder {
    layers: Vec<Box<dyn Layer<f32>>>,
impl FeatureEncoder {
    fn new(_input_dim: usize, output_dim: usize, hiddensizes: Vec<usize>) -> Result<Self> {
        let mut layers: Vec<Box<dyn Layer<f32>>> = Vec::new();
        let mut current_dim = input_dim;
        for hidden_size in hidden_sizes {
            layers.push(Box::new(Dense::new(
                current_dim,
                hidden_size,
                Some(Activation::ReLU),
            )?));
            current_dim = hidden_size;
        layers.push(Box::new(Dense::new(current_dim, output_dim, None)?));
        Ok(Self { layers })
    fn encode(&self, state: &ArrayView1<f32>) -> Result<Array1<f32>> {
        let mut x = state.to_owned().insert_axis(Axis(0));
        for layer in &self.layers {
            x = layer.forward(&x.view())?;
        Ok(x.remove_axis(Axis(0)))
/// Forward dynamics model
struct ForwardModel {
impl ForwardModel {
    fn new(
        state_feat_dim: usize,
        output_dim: usize,
        let input_dim = state_feat_dim + action_dim;
    fn predict(
        state_feat: &ArrayView1<f32>,
    ) -> Result<Array1<f32>> {
        // Concatenate state features and action
        let input = concatenate![Axis(0), *state_feat, *action];
        let mut x = input.insert_axis(Axis(0));
/// Inverse dynamics model
struct InverseModel {
impl InverseModel {
        next_state_feat_dim: usize,
        let input_dim = state_feat_dim + next_state_feat_dim;
        layers.push(Box::new(Dense::new(current_dim, action_dim, None)?));
        next_state_feat: &ArrayView1<f32>,
        // Concatenate state features
        let input = concatenate![Axis(0), *state_feat, *next_state_feat];
/// Random Network Distillation (RND) for exploration
/// RND uses a fixed random network as a target and trains a predictor network.
/// The prediction error serves as an exploration bonus.
pub struct RND {
    target_network: RandomNetwork,
    predictor_network: PredictorNetwork,
    learning_rate: f32,
    intrinsic_reward_scale: f32,
impl RND {
    /// Create a new RND module
        learning_rate: f32,
        intrinsic_reward_scale: f32,
        let target_network = RandomNetwork::new(state_dim, feature_dim)?;
        let predictor_network = PredictorNetwork::new(state_dim, feature_dim, hidden_sizes)?;
            target_network,
            predictor_network,
            learning_rate,
            intrinsic_reward_scale,
    pub fn compute_intrinsic_reward(&self, state: &ArrayView1<f32>) -> Result<f32> {
        let target_features = self.target_network.forward(state)?;
        let predicted_features = self.predictor_network.forward(state)?;
        let error = (&target_features - &predicted_features)
            .mapv(|x| x.powi(2))
            .sum();
        Ok(self.intrinsic_reward_scale * error)
    /// Update predictor network
    pub fn update(&mut self, states: &ArrayView2<f32>) -> Result<f32> {
        let mut total_loss = 0.0;
            let target_features = self.target_network.forward(&state)?;
            let predicted_features = self.predictor_network.forward(&state)?;
            let loss = (&target_features - &predicted_features)
                .mapv(|x| x.powi(2))
                .sum();
            total_loss += loss;
        Ok(total_loss / batch_size as f32)
/// Fixed random network for RND
struct RandomNetwork {
impl RandomNetwork {
    fn new(_input_dim: usize, outputdim: usize) -> Result<Self> {
        // Simple 2-layer network with fixed random weights
        layers.push(Box::new(Dense::new(
            input_dim,
            256,
            Some(Activation::ReLU),
        )?));
        layers.push(Box::new(Dense::new(256, output_dim, None)?));
        // Weights are initialized randomly and kept fixed
    fn forward(&self, state: &ArrayView1<f32>) -> Result<Array1<f32>> {
/// Predictor network for RND
struct PredictorNetwork {
impl PredictorNetwork {
/// Novelty-based exploration
/// Maintains a buffer of visited states and rewards novel states
pub struct NoveltyExploration {
    state_buffer: VecDeque<Array1<f32>>,
    buffer_size: usize,
    k_nearest: usize,
    novelty_scale: f32,
impl NoveltyExploration {
    /// Create a new novelty exploration module
    pub fn new(_buffer_size: usize, k_nearest: usize, noveltyscale: f32) -> Self {
        Self {
            state_buffer: VecDeque::with_capacity(buffer_size),
            buffer_size,
            k_nearest,
            novelty_scale,
    /// Compute novelty reward
    pub fn compute_novelty_reward(&self, state: &ArrayView1<f32>) -> f32 {
        if self.state_buffer.len() < self.k_nearest {
            return self.novelty_scale;
        // Compute distances to all states in buffer
        let mut distances: Vec<f32> = self
            .state_buffer
            .iter()
            .map(|s| (s - state).mapv(|x| x.powi(2)).sum().sqrt())
            .collect();
        // Sort and take k nearest
        distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let k_nearest_distances: Vec<f32> = distances.into_iter().take(self.k_nearest).collect();
        // Average distance to k nearest neighbors
        let avg_distance = k_nearest_distances.iter().sum::<f32>() / self.k_nearest as f32;
        self.novelty_scale * avg_distance
    /// Add state to buffer
    pub fn add_state(&mut self, state: Array1<f32>) {
        if self.state_buffer.len() >= self.buffer_size {
            self.state_buffer.pop_front();
        self.state_buffer.push_back(state);
/// Episodic curiosity module
/// Provides bonus based on episodic memory of state visitations
pub struct EpisodicCuriosity {
    episodic_memory: Vec<Array1<f32>>,
    visit_counts: Vec<usize>,
    embedding_dim: usize,
    bonus_scale: f32,
    similarity_threshold: f32,
impl EpisodicCuriosity {
    /// Create a new episodic curiosity module
    pub fn new(_embedding_dim: usize, bonus_scale: f32, similaritythreshold: f32) -> Self {
            episodic_memory: Vec::new(),
            visit_counts: Vec::new(),
            embedding_dim,
            bonus_scale,
            similarity_threshold,
    /// Compute episodic bonus
    pub fn compute_bonus(&mut self, stateembedding: &ArrayView1<f32>) -> f32 {
        // Find similar states in memory
        let mut min_distance = f32::MAX;
        let mut closest_idx = None;
        for (idx, memory_state) in self.episodic_memory.iter().enumerate() {
            let distance = (memory_state - state_embedding)
                .sum()
                .sqrt();
            if distance < min_distance {
                min_distance = distance;
                closest_idx = Some(idx);
            }
        if let Some(idx) = closest_idx {
            if min_distance < self.similarity_threshold {
                // State is similar to one in memory
                self.visit_counts[idx] += 1;
                let count = self.visit_counts[idx] as f32;
                return self.bonus_scale / count.sqrt();
        // New state
        self.episodic_memory.push(state_embedding.to_owned());
        self.visit_counts.push(1);
        self.bonus_scale
    /// Clear episodic memory (e.g., at episode boundaries)
    pub fn clear(&mut self) {
        self.episodic_memory.clear();
        self.visit_counts.clear();
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_icm_creation() {
        let icm = ICM::new(4, 2, 32, vec![64, 64], 0.01, 0.5).unwrap();
        let state = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let action = Array1::from_vec(vec![0.5, -0.5]);
        let next_state = Array1::from_vec(vec![1.1, 2.1, 3.1, 4.1]);
        let reward = icm
            .compute_intrinsic_reward(&state.view(), &action.view(), &next_state.view())
            .unwrap();
        assert!(reward >= 0.0);
    fn test_rnd_creation() {
        let rnd = RND::new(4, 32, vec![64, 64], 0.001, 0.1).unwrap();
        let reward = rnd.compute_intrinsic_reward(&state.view()).unwrap();
    fn test_novelty_exploration() {
        let mut novelty = NoveltyExploration::new(100, 5, 1.0);
        // Add some states
        for i in 0..10 {
            let state = Array1::from_vec(vec![i as f32; 4]);
            novelty.add_state(state);
        // Test novelty reward
        let novel_state = Array1::from_vec(vec![100.0; 4]);
        let familiar_state = Array1::from_vec(vec![5.0; 4]);
        let novel_reward = novelty.compute_novelty_reward(&novel_state.view());
        let familiar_reward = novelty.compute_novelty_reward(&familiar_state.view());
        assert!(novel_reward > familiar_reward);
    fn test_episodic_curiosity() {
        let mut episodic = EpisodicCuriosity::new(4, 1.0, 0.1);
        let state1 = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let state2 = Array1::from_vec(vec![1.1, 2.1, 3.1, 4.1]); // Similar to state1
        let state3 = Array1::from_vec(vec![10.0, 20.0, 30.0, 40.0]); // Different
        let bonus1 = episodic.compute_bonus(&state1.view());
        let bonus2 = episodic.compute_bonus(&state2.view());
        let bonus3 = episodic.compute_bonus(&state3.view());
        assert!(bonus1 > 0.0);
        assert!(bonus2 < bonus1); // Similar state gets lower bonus
        assert_eq!(bonus3, bonus1); // New state gets full bonus
