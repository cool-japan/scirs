//! Configuration for neural-adaptive sparse matrix processing
//!
//! This module contains configuration structures and enums for the neural
//! adaptive sparse matrix processor system.

use super::reinforcement_learning::RLAlgorithm;

/// Neural-adaptive sparse matrix processor configuration
#[derive(Debug, Clone)]
pub struct NeuralAdaptiveConfig {
    /// Number of hidden layers in the neural network
    pub hidden_layers: usize,
    /// Neurons per hidden layer
    pub neurons_per_layer: usize,
    /// Learning rate for adaptive optimization
    pub learningrate: f64,
    /// Memory capacity for pattern learning
    pub memory_capacity: usize,
    /// Enable reinforcement learning
    pub reinforcement_learning: bool,
    /// Attention mechanism configuration
    pub attention_heads: usize,
    /// Enable transformer-style self-attention
    pub self_attention: bool,
    /// Reinforcement learning algorithm
    pub rl_algorithm: RLAlgorithm,
    /// Exploration rate for RL
    pub exploration_rate: f64,
    /// Discount factor for future rewards
    pub discountfactor: f64,
    /// Experience replay buffer size
    pub replay_buffer_size: usize,
    /// Transformer model dimension
    pub modeldim: usize,
    /// Feed-forward network dimension in transformer
    pub ff_dim: usize,
    /// Number of transformer layers
    pub transformer_layers: usize,
}

impl Default for NeuralAdaptiveConfig {
    fn default() -> Self {
        Self {
            hidden_layers: 3,
            neurons_per_layer: 64,
            learningrate: 0.001,
            memory_capacity: 10000,
            reinforcement_learning: true,
            attention_heads: 8,
            self_attention: true,
            rl_algorithm: RLAlgorithm::DQN,
            exploration_rate: 0.1,
            discountfactor: 0.99,
            replay_buffer_size: 10000,
            modeldim: 512,
            ff_dim: 2048,
            transformer_layers: 6,
        }
    }
}

impl NeuralAdaptiveConfig {
    /// Create a new configuration with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of hidden layers
    pub fn with_hidden_layers(mut self, layers: usize) -> Self {
        self.hidden_layers = layers;
        self
    }

    /// Set neurons per layer
    pub fn with_neurons_per_layer(mut self, neurons: usize) -> Self {
        self.neurons_per_layer = neurons;
        self
    }

    /// Set learning rate
    pub fn with_learning_rate(mut self, rate: f64) -> Self {
        self.learningrate = rate;
        self
    }

    /// Set memory capacity
    pub fn with_memory_capacity(mut self, capacity: usize) -> Self {
        self.memory_capacity = capacity;
        self
    }

    /// Enable or disable reinforcement learning
    pub fn with_reinforcement_learning(mut self, enabled: bool) -> Self {
        self.reinforcement_learning = enabled;
        self
    }

    /// Set number of attention heads
    pub fn with_attention_heads(mut self, heads: usize) -> Self {
        self.attention_heads = heads;
        self
    }

    /// Enable or disable self-attention
    pub fn with_self_attention(mut self, enabled: bool) -> Self {
        self.self_attention = enabled;
        self
    }

    /// Set RL algorithm
    pub fn with_rl_algorithm(mut self, algorithm: RLAlgorithm) -> Self {
        self.rl_algorithm = algorithm;
        self
    }

    /// Set exploration rate
    pub fn with_exploration_rate(mut self, rate: f64) -> Self {
        self.exploration_rate = rate;
        self
    }

    /// Set discount factor
    pub fn with_discount_factor(mut self, factor: f64) -> Self {
        self.discountfactor = factor;
        self
    }

    /// Set replay buffer size
    pub fn with_replay_buffer_size(mut self, size: usize) -> Self {
        self.replay_buffer_size = size;
        self
    }

    /// Set transformer model dimension
    pub fn with_model_dim(mut self, dim: usize) -> Self {
        self.modeldim = dim;
        self
    }

    /// Set feed-forward dimension
    pub fn with_ff_dim(mut self, dim: usize) -> Self {
        self.ff_dim = dim;
        self
    }

    /// Set number of transformer layers
    pub fn with_transformer_layers(mut self, layers: usize) -> Self {
        self.transformer_layers = layers;
        self
    }

    /// Validate configuration parameters
    pub fn validate(&self) -> Result<(), String> {
        if self.hidden_layers == 0 {
            return Err("Hidden layers must be greater than 0".to_string());
        }

        if self.neurons_per_layer == 0 {
            return Err("Neurons per layer must be greater than 0".to_string());
        }

        if self.learningrate <= 0.0 || self.learningrate > 1.0 {
            return Err("Learning rate must be between 0 and 1".to_string());
        }

        if self.memory_capacity == 0 {
            return Err("Memory capacity must be greater than 0".to_string());
        }

        if self.attention_heads == 0 {
            return Err("Attention heads must be greater than 0".to_string());
        }

        if self.exploration_rate < 0.0 || self.exploration_rate > 1.0 {
            return Err("Exploration rate must be between 0 and 1".to_string());
        }

        if self.discountfactor < 0.0 || self.discountfactor > 1.0 {
            return Err("Discount factor must be between 0 and 1".to_string());
        }

        if self.replay_buffer_size == 0 {
            return Err("Replay buffer size must be greater than 0".to_string());
        }

        if self.modeldim == 0 {
            return Err("Model dimension must be greater than 0".to_string());
        }

        if self.ff_dim == 0 {
            return Err("Feed-forward dimension must be greater than 0".to_string());
        }

        if self.transformer_layers == 0 {
            return Err("Transformer layers must be greater than 0".to_string());
        }

        if !self.modeldim.is_multiple_of(self.attention_heads) {
            return Err(
                "Model dimension must be divisible by number of attention heads".to_string(),
            );
        }

        Ok(())
    }

    /// Create a lightweight configuration for testing
    pub fn lightweight() -> Self {
        Self {
            hidden_layers: 2,
            neurons_per_layer: 16,
            learningrate: 0.01,
            memory_capacity: 100,
            reinforcement_learning: true,
            attention_heads: 2,
            self_attention: false,
            rl_algorithm: RLAlgorithm::DQN,
            exploration_rate: 0.3,
            discountfactor: 0.9,
            replay_buffer_size: 100,
            modeldim: 64,
            ff_dim: 128,
            transformer_layers: 2,
        }
    }

    /// Create a high-performance configuration
    pub fn high_performance() -> Self {
        Self {
            hidden_layers: 5,
            neurons_per_layer: 128,
            learningrate: 0.0001,
            memory_capacity: 50000,
            reinforcement_learning: true,
            attention_heads: 16,
            self_attention: true,
            rl_algorithm: RLAlgorithm::PPO,
            exploration_rate: 0.05,
            discountfactor: 0.995,
            replay_buffer_size: 50000,
            modeldim: 1024,
            ff_dim: 4096,
            transformer_layers: 12,
        }
    }

    /// Create a memory-efficient configuration
    pub fn memory_efficient() -> Self {
        Self {
            hidden_layers: 2,
            neurons_per_layer: 32,
            learningrate: 0.005,
            memory_capacity: 1000,
            reinforcement_learning: false,
            attention_heads: 4,
            self_attention: false,
            rl_algorithm: RLAlgorithm::DQN,
            exploration_rate: 0.1,
            discountfactor: 0.99,
            replay_buffer_size: 1000,
            modeldim: 256,
            ff_dim: 512,
            transformer_layers: 3,
        }
    }
}
