//! Efficient Neural Architecture Search (ENAS) implementation
//!
//! ENAS speeds up NAS by sharing weights among child models and using
//! a controller to sample architectures.

use crate::activations::Activation;
use crate::error::{NeuralError, Result};
use crate::layers::{Dense, Layer};
use crate::nas::search_space::{Architecture, LayerType};
use crate::nas::SearchSpace;
use scirs2_core::ndarray::concatenate;
use scirs2_core::ndarray::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use scirs2_core::ndarray::ArrayView1;
/// ENAS Controller that generates architectures
pub struct ENASController {
    hidden_size: usize,
    num_layers: usize,
    search_space: SearchSpace,
    lstm_cell: LSTMCell,
    embedding_dim: usize,
    temperature: f32,
}
impl ENASController {
    /// Create a new ENAS controller
    pub fn new(
        hidden_size: usize,
        num_layers: usize,
        search_space: SearchSpace,
        embedding_dim: usize,
        temperature: f32,
    ) -> Result<Self> {
        let lstm_cell = LSTMCell::new(embedding_dim, hidden_size)?;
        Ok(Self {
            hidden_size,
            num_layers,
            search_space,
            lstm_cell,
            embedding_dim,
            temperature,
        })
    }
    /// Sample an architecture from the controller
    pub fn sample_architecture(&self, batchsize: usize) -> Result<Vec<Architecture>> {
        let mut architectures = Vec::with_capacity(batch_size);
        let mut log_probs = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            let (arch, log_prob) = self.sample_single()?;
            architectures.push(arch);
            log_probs.push(log_prob);
        }
        Ok(architectures)
    /// Sample a single architecture
    fn sample_single(&self) -> Result<(Architecture, f32)> {
        let mut layers = Vec::new();
        let mut connections = Vec::new();
        let mut total_log_prob = 0.0;
        // Initialize LSTM hidden state
        let mut hidden = Array1::zeros(self.hidden_size);
        let mut cell = Array1::zeros(self.hidden_size);
        // Sample layers
        let num_layers = self.sample_num_layers(&mut hidden, &mut cell)?;
        for i in 0..num_layers {
            // Sample layer type
            let (layer_type, log_prob) = self.sample_layer_type(&hidden, i)?;
            layers.push(layer_type);
            total_log_prob += log_prob;
            // Update LSTM state
            let input_embedding = self.get_layer_embedding(i)?;
            let (new_hidden, new_cell) =
                self.lstm_cell
                    .forward(&input_embedding.view(), &hidden.view(), &cell.view())?;
            hidden = new_hidden;
            cell = new_cell;
            // Sample connections (skip connections)
            if i > 0 && self.search_space.config.allow_branches {
                let (skip_connections, skip_log_probs) =
                    self.sample_skip_connections(&hidden, i)?;
                for (j, &skip) in skip_connections.iter().enumerate() {
                    if skip {
                        connections.push((j, i));
                        total_log_prob += skip_log_probs[j];
                    }
                }
            }
        // Sample multipliers
        let width_mult = self.sample_width_multiplier(&hidden)?;
        let depth_mult = self.sample_depth_multiplier(&hidden)?;
        Ok((
            Architecture {
                layers,
                connections,
                width_multiplier: width_mult,
                depth_multiplier: depth_mult,
            },
            total_log_prob,
        ))
    /// Sample number of layers
    fn sample_num_layers(&self, hidden: &mut Array1<f32>, cell: &mut Array1<f32>) -> Result<usize> {
        let min_layers = self.search_space.config.min_layers;
        let max_layers = self.search_space.config.max_layers;
        let num_choices = max_layers - min_layers + 1;
        // Use a linear layer to predict logits
        let logits = Array1::from_vec(vec![1.0 / num_choices as f32; num_choices]);
        let probs = softmax(&logits, self.temperature);
        // Sample from categorical distribution
        let choice = sample_categorical(&probs)?;
        Ok(min_layers + choice)
    /// Sample layer type
    fn sample_layer_type(
        &self,
        hidden: &ArrayView1<f32>,
        position: usize,
    ) -> Result<(LayerType, f32)> {
        let layer_choices = &self.search_space.layer_choices[position].choices;
        let num_choices = layer_choices.len();
        // Simple uniform distribution for now
        let log_prob = probs[choice].ln();
        Ok((layer_choices[choice].clone(), log_prob))
    /// Sample skip connections
    fn sample_skip_connections(
        current_layer: usize,
    ) -> Result<(Vec<bool>, Vec<f32>)> {
        let mut skip_connections = vec![false; current_layer];
        let mut log_probs = vec![0.0; current_layer];
        for i in 0..current_layer {
            // Predict skip connection probability
            let skip_prob = self.search_space.config.skip_connection_prob;
            let probs = Array1::from_vec(vec![1.0 - skip_prob, skip_prob]);
            let choice = sample_categorical(&probs)?;
            skip_connections[i] = choice == 1;
            log_probs[i] = probs[choice].ln();
        Ok((skip_connections, log_probs))
    /// Sample width multiplier
    fn sample_width_multiplier(&self, hidden: &ArrayView1<f32>) -> Result<f32> {
        let choices = &self.search_space.config.width_multipliers;
        let idx = scirs2_core::random::random::<usize>() % choices.len();
        Ok(choices[idx])
    /// Sample depth multiplier
    fn sample_depth_multiplier(&self, hidden: &ArrayView1<f32>) -> Result<f32> {
        let choices = &self.search_space.config.depth_multipliers;
    /// Get embedding for a layer position
    fn get_layer_embedding(&self, position: usize) -> Result<Array1<f32>> {
        // Simple positional embedding
        let mut embedding = Array1::zeros(self.embedding_dim);
        for i in 0..self.embedding_dim {
            if i % 2 == 0 {
                embedding[i] = (position as f32
                    / 10000.0_f32.powf(i as f32 / self.embedding_dim as f32))
                .sin();
            } else {
                    / 10000.0_f32.powf((i - 1) as f32 / self.embedding_dim as f32))
                .cos();
        Ok(embedding)
    /// Train the controller with REINFORCE
    pub fn train_step(&mut self, rewards: &[f32], logprobs: &[f32], baseline: f32) -> Result<f32> {
        let advantages: Vec<f32> = rewards.iter().map(|&r| r - baseline).collect();
        // Compute policy gradient loss
        let mut loss = 0.0;
        for (log_prob, advantage) in log_probs.iter().zip(advantages.iter()) {
            loss -= log_prob * advantage;
        loss /= rewards.len() as f32;
        // Add entropy regularization
        let entropy_bonus = 0.0; // Simplified - would compute actual entropy
        loss -= self.temperature * entropy_bonus;
        Ok(loss)
/// Super network with shared weights
pub struct SuperNetwork {
    shared_weights: Arc<RwLock<HashMap<String, Array2<f32>>>>,
    max_layers: usize,
    layer_configs: Vec<Vec<LayerConfig>>,
#[derive(Clone)]
struct LayerConfig {
    layer_type: LayerType,
    input_dim: usize,
    output_dim: usize,
    weight_key: String,
impl SuperNetwork {
    /// Create a new super network
    pub fn new(_searchspace: &SearchSpace) -> Result<Self> {
        let shared_weights = Arc::new(RwLock::new(HashMap::new()));
        let max_layers = search_space.config.max_layers;
        let mut layer_configs = vec![Vec::new(); max_layers];
        // Initialize shared weights for all possible layers
        for (pos, layer_choice) in search_space.layer_choices.iter().enumerate() {
            for layer_type in &layer_choice.choices {
                let config = Self::create_layer_config(layer_type, pos)?;
                layer_configs[pos].push(config);
            shared_weights,
            max_layers,
            layer_configs,
    /// Create layer configuration
    fn create_layer_config(_layertype: &LayerType, position: usize) -> Result<LayerConfig> {
        let (input_dim, output_dim) = match layer_type {
            LayerType::Dense(units) => (512, *units), // Placeholder dimensions
            LayerType::Conv2D { filters, .. } => (64, *filters, _ => (512, 512),
        };
        let weight_key = format!("{:?}_pos_{}", layer_type, position);
        Ok(LayerConfig {
            layer_type: layer_type.clone(),
            input_dim,
            output_dim,
            weight_key,
    /// Execute a child model with given architecture
    pub fn execute_child(
        architecture: &Architecture,
        input: &ArrayView2<f32>,
    ) -> Result<Array2<f32>> {
        let mut activations: HashMap<usize, Array2<f32>> = HashMap::new();
        activations.insert(0, input.to_owned());
        for (i, layer_type) in architecture.layers.iter().enumerate() {
            // Get input to this layer
            let layer_input = if i == 0 {
                input.to_owned()
                // Check for skip connections
                let mut sum = activations.get(&(i - 1)).unwrap().clone();
                for (j, k) in &architecture.connections {
                    if *k == i {
                        if let Some(skip_input) = activations.get(j) {
                            // Simple addition for skip connections
                            sum = sum + skip_input;
                        }
                sum
            };
            // Execute layer with shared weights
            let output = self.execute_layer(layer_type, &layer_input.view(), i)?;
            activations.insert(i + 1, output);
        // Return final layer output
        activations
            .get(&architecture.layers.len())
            .ok_or_else(|| NeuralError::InvalidArgument("No output computed".to_string()))
            .map(|a| a.clone())
    /// Execute a single layer
    fn execute_layer(
        layer_type: &LayerType,
        // Find matching layer config
        let config = self.layer_configs[position]
            .iter()
            .find(|c| &c.layer_type == layer_type)
            .ok_or_else(|| NeuralError::InvalidArgument("Layer config not found".to_string()))?;
        // Get shared weights
        let weights = self.sharedweights.read().unwrap();
        let weight = weights
            .get(&config.weight_key)
            .ok_or_else(|| NeuralError::InvalidArgument("Shared weights not found".to_string()))?;
        // Simple matrix multiplication (simplified)
        let output = input.dot(weight);
        // Apply activation
        match layer_type {
            LayerType::Activation(name) => {
                let activation = match name.as_str() {
                    "relu" => Activation::ReLU,
                    "swish" => Activation::Swish_ => Activation::ReLU,
                };
                Ok(activation.apply(&output))
            _ => Ok(output),
    /// Update shared weights
    pub fn update_weights(&mut self, gradients: &HashMap<String, Array2<f32>>) -> Result<()> {
        let mut weights = self.sharedweights.write().unwrap();
        for (key, grad) in gradients {
            if let Some(weight) = weights.get_mut(key) {
                // Simple SGD update
                *weight = weight - 0.01 * grad;
        Ok(())
/// LSTM cell for the controller
struct LSTMCell {
    hidden_dim: usize,
    w_i: Dense<f32>,
    w_f: Dense<f32>,
    w_o: Dense<f32>,
    w_g: Dense<f32>,
impl LSTMCell {
    fn new(_input_dim: usize, hiddendim: usize) -> Result<Self> {
        let combined_dim = _input_dim + hidden_dim;
            hidden_dim,
            w_i: Dense::new(combined_dim, hidden_dim, Some(Activation::Sigmoid))?,
            w_f: Dense::new(combined_dim, hidden_dim, Some(Activation::Sigmoid))?,
            w_o: Dense::new(combined_dim, hidden_dim, Some(Activation::Sigmoid))?,
            w_g: Dense::new(combined_dim, hidden_dim, Some(Activation::Tanh))?,
    fn forward(
        input: &ArrayView1<f32>,
        cell: &ArrayView1<f32>,
    ) -> Result<(Array1<f32>, Array1<f32>)> {
        // Concatenate input and hidden state
        let combined = concatenate![Axis(0), *input, *hidden];
        let combined = combined.insert_axis(Axis(0));
        // Compute gates
        let i = self.w_i.forward(&combined.view())?.remove_axis(Axis(0));
        let f = self.w_f.forward(&combined.view())?.remove_axis(Axis(0));
        let o = self.w_o.forward(&combined.view())?.remove_axis(Axis(0));
        let g = self.w_g.forward(&combined.view())?.remove_axis(Axis(0));
        // Update cell state
        let new_cell = &f * cell + &i * &g;
        // Update hidden state
        let new_hidden = &o * new_cell.mapv(f32::tanh);
        Ok((new_hidden, new_cell))
/// ENAS trainer
pub struct ENASTrainer {
    controller: ENASController,
    super_network: SuperNetwork,
    controller_lr: f32,
    child_lr: f32,
    entropy_weight: f32,
    baseline_decay: f32,
    baseline: Option<f32>,
impl ENASTrainer {
    /// Create a new ENAS trainer
        controller_hidden_size: usize,
        controller_lr: f32,
        child_lr: f32,
        entropy_weight: f32,
        let controller = ENASController::new(
            controller_hidden_size,
            search_space.config.max_layers,
            search_space.clone(),
            32,
            1.0,
        )?;
        let super_network = SuperNetwork::new(&search_space)?;
            controller,
            super_network,
            controller_lr,
            child_lr,
            entropy_weight,
            baseline_decay: 0.99,
            baseline: None,
    /// Train one epoch
    pub fn train_epoch(
        &mut self,
        train_data: &ArrayView2<f32>,
        train_labels: &ArrayView1<usize>,
        val_data: &ArrayView2<f32>,
        val_labels: &ArrayView1<usize>,
        controller_steps: usize,
        child_steps: usize,
    ) -> Result<(f32, f32)> {
        // Train child model (super network)
        let mut child_loss = 0.0;
        for _ in 0..child_steps {
            let architectures = self.controller.sample_architecture(1)?;
            let arch = &architectures[0];
            // Forward pass through super network
            let output = self.super_network.execute_child(arch, train_data)?;
            // Compute loss (simplified)
            child_loss += 0.1; // Placeholder
            // Update shared weights
            let gradients = HashMap::new(); // Would compute actual gradients
            self.super_network.update_weights(&gradients)?;
        // Train controller
        let mut controller_loss = 0.0;
        let mut rewards = Vec::new();
        for _ in 0..controller_steps {
            // Evaluate architecture on validation set
            let output = self.super_network.execute_child(arch, val_data)?;
            let reward = self.compute_reward(&output, val_labels)?;
            rewards.push(reward);
            // Update baseline
            self.update_baseline(reward);
            // Train controller
            let log_probs = vec![0.0]; // Would track actual log probs
            let loss =
                self.controller
                    .train_step(&[reward], &log_probs, self.baseline.unwrap_or(0.0))?;
            controller_loss += loss;
            child_loss / child_steps as f32,
            controller_loss / controller_steps as f32,
    /// Compute reward for an architecture
    fn compute_reward(&self, predictions: &Array2<f32>, labels: &ArrayView1<usize>) -> Result<f32> {
        // Simplified accuracy computation
        Ok(0.9) // Placeholder
    /// Update baseline with exponential moving average
    fn update_baseline(&mut self, reward: f32) {
        self.baseline = Some(match self.baseline {
            Some(b) => self.baseline_decay * b + (1.0 - self.baseline_decay) * reward,
            None => reward,
        });
    /// Get the best architecture found
    pub fn get_best_architecture(&self) -> Result<Architecture> {
        // Sample with low temperature for best architecture
        let mut controller = self.controller.clone();
        controller.temperature = 0.1;
        let architectures = controller.sample_architecture(1)?;
        Ok(architectures[0].clone())
/// Helper function to apply softmax with temperature
#[allow(dead_code)]
fn softmax(logits: &Array1<f32>, temperature: f32) -> Array1<f32> {
    let scaled_logits = logits / temperature;
    let max_logit = scaled_logits
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    let exp_logits = (scaled_logits - max_logit).mapv(f32::exp);
    let sum = exp_logits.sum();
    exp_logits / sum
/// Helper function to sample from categorical distribution
#[allow(dead_code)]
fn sample_categorical(probs: &Array1<f32>) -> Result<usize> {
    let mut rng = rng();
    let uniform: f32 = scirs2_core::random::Rng::random(&mut rng);
    let mut cumsum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if uniform <= cumsum {
            return Ok(i);
    Ok(probs.len() - 1)
// Make ENASController cloneable for the trainer
impl Clone for ENASController {
    fn clone(&self) -> Self {
        Self {
            hidden_size: self.hidden_size,
            num_layers: self.num_layers,
            search_space: self.search_space.clone(),
            lstm_cell: self.lstm_cell.clone(),
            embedding_dim: self.embedding_dim,
            temperature: self.temperature,
impl Clone for LSTMCell {
            input_dim: self.input_dim,
            hidden_dim: self.hidden_dim,
            w_i: self.w_i.clone(),
            w_f: self.w_f.clone(),
            w_o: self.w_o.clone(),
            w_g: self.w_g.clone(),
#[cfg(test)]
mod tests {
    use super::*;
    use crate::nas::SearchSpaceConfig;
    #[test]
    fn test_enas_controller() {
        let config = SearchSpaceConfig::default();
        let search_space = SearchSpace::new(config).unwrap();
        let controller = ENASController::new(100, 10, search_space, 32, 1.0).unwrap();
        let architectures = controller.sample_architecture(5).unwrap();
        assert_eq!(architectures.len(), 5);
        for arch in &architectures {
            assert!(!arch.layers.is_empty());
    fn test_super_network() {
        let super_net = SuperNetwork::new(&search_space).unwrap();
        // Test with dummy architecture
        let arch = search_space.sample().unwrap();
        let input = Array2::ones((32, 512)); // Batch size 32, feature dim 512
        // Would test execution but needs proper weight initialization
    fn test_softmax() {
        let logits = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let probs = softmax(&logits, 1.0);
        assert!((probs.sum() - 1.0).abs() < 1e-6);
        assert!(probs.iter().all(|&p| p >= 0.0 && p <= 1.0));
