//! Configuration structures for transformer-based optimizer

use num_traits::Float;
use serde::{Serialize, Deserialize};
use super::positional_encoding::PositionalEncodingType;

/// Configuration for transformer-based optimizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformerBasedOptimizerConfig<T: Float> {
    /// Model dimension (embedding size)
    pub model_dimension: usize,

    /// Number of transformer layers
    pub num_transformer_layers: usize,

    /// Number of attention heads
    pub num_attention_heads: usize,

    /// Dimension of each attention head
    pub attention_head_dimension: usize,

    /// Feed-forward network dimension
    pub feedforward_dimension: usize,

    /// Maximum sequence length
    pub sequence_length: usize,

    /// Dropout rate
    pub dropout_rate: f64,

    /// Learning rate
    pub learning_rate: T,

    /// Batch size for training
    pub batch_size: usize,

    /// Number of training epochs
    pub num_epochs: usize,

    /// Activation function type
    pub activation_function: ActivationFunction,

    /// Positional encoding type
    pub positional_encoding_type: PositionalEncodingType,

    /// Memory management configuration
    pub memory_config: MemoryConfig,

    /// Meta-learning configuration
    pub meta_learning_config: MetaLearningConfig<T>,

    /// Performance tracking configuration
    pub performance_config: PerformanceConfig,

    /// Enable gradient clipping
    pub enable_gradient_clipping: bool,

    /// Gradient clipping threshold
    pub gradient_clip_value: T,

    /// Weight decay factor
    pub weight_decay: T,

    /// Warmup steps for learning rate schedule
    pub warmup_steps: usize,

    /// Enable layer normalization
    pub enable_layer_norm: bool,

    /// Pre-norm vs post-norm
    pub use_pre_norm: bool,

    /// Enable residual connections
    pub enable_residual_connections: bool,
}

impl<T: Float> Default for TransformerBasedOptimizerConfig<T> {
    fn default() -> Self {
        Self {
            model_dimension: 512,
            num_transformer_layers: 6,
            num_attention_heads: 8,
            attention_head_dimension: 64,
            feedforward_dimension: 2048,
            sequence_length: 128,
            dropout_rate: 0.1,
            learning_rate: T::from(1e-4).unwrap(),
            batch_size: 32,
            num_epochs: 100,
            activation_function: ActivationFunction::ReLU,
            positional_encoding_type: PositionalEncodingType::Sinusoidal,
            memory_config: MemoryConfig::default(),
            meta_learning_config: MetaLearningConfig::default(),
            performance_config: PerformanceConfig::default(),
            enable_gradient_clipping: true,
            gradient_clip_value: T::from(1.0).unwrap(),
            weight_decay: T::from(1e-5).unwrap(),
            warmup_steps: 1000,
            enable_layer_norm: true,
            use_pre_norm: true,
            enable_residual_connections: true,
        }
    }
}

impl<T: Float> TransformerBasedOptimizerConfig<T> {
    /// Create configuration for small models
    pub fn small() -> Self {
        Self {
            model_dimension: 256,
            num_transformer_layers: 4,
            num_attention_heads: 4,
            attention_head_dimension: 64,
            feedforward_dimension: 1024,
            sequence_length: 64,
            ..Self::default()
        }
    }

    /// Create configuration for large models
    pub fn large() -> Self {
        Self {
            model_dimension: 1024,
            num_transformer_layers: 12,
            num_attention_heads: 16,
            attention_head_dimension: 64,
            feedforward_dimension: 4096,
            sequence_length: 256,
            ..Self::default()
        }
    }

    /// Create configuration optimized for training
    pub fn for_training() -> Self {
        Self {
            batch_size: 64,
            num_epochs: 200,
            learning_rate: T::from(2e-4).unwrap(),
            warmup_steps: 2000,
            enable_gradient_clipping: true,
            weight_decay: T::from(1e-4).unwrap(),
            ..Self::default()
        }
    }

    /// Create configuration optimized for inference
    pub fn for_inference() -> Self {
        Self {
            batch_size: 1,
            dropout_rate: 0.0,
            enable_gradient_clipping: false,
            ..Self::default()
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.model_dimension == 0 {
            return Err("model_dimension must be greater than 0".to_string());
        }

        if self.num_transformer_layers == 0 {
            return Err("num_transformer_layers must be greater than 0".to_string());
        }

        if self.num_attention_heads == 0 {
            return Err("num_attention_heads must be greater than 0".to_string());
        }

        if self.model_dimension % self.num_attention_heads != 0 {
            return Err("model_dimension must be divisible by num_attention_heads".to_string());
        }

        if self.attention_head_dimension * self.num_attention_heads != self.model_dimension {
            return Err("attention_head_dimension * num_attention_heads must equal model_dimension".to_string());
        }

        if self.sequence_length == 0 {
            return Err("sequence_length must be greater than 0".to_string());
        }

        if self.dropout_rate < 0.0 || self.dropout_rate > 1.0 {
            return Err("dropout_rate must be between 0.0 and 1.0".to_string());
        }

        if self.learning_rate <= T::zero() {
            return Err("learning_rate must be positive".to_string());
        }

        if self.batch_size == 0 {
            return Err("batch_size must be greater than 0".to_string());
        }

        self.memory_config.validate()?;
        self.meta_learning_config.validate()?;
        self.performance_config.validate()?;

        Ok(())
    }

    /// Calculate total parameters estimate
    pub fn estimate_parameter_count(&self) -> usize {
        let embedding_params = self.model_dimension * self.model_dimension; // Input embedding
        let positional_params = self.sequence_length * self.model_dimension;

        let attention_params_per_layer = 4 * self.model_dimension * self.model_dimension; // Q, K, V, O projections
        let ffn_params_per_layer = 2 * self.model_dimension * self.feedforward_dimension; // Up and down projections
        let norm_params_per_layer = 2 * self.model_dimension; // Layer norm parameters

        let layer_params = attention_params_per_layer + ffn_params_per_layer + norm_params_per_layer;
        let total_layer_params = layer_params * self.num_transformer_layers;

        let output_params = self.model_dimension * self.model_dimension; // Output projection

        embedding_params + positional_params + total_layer_params + output_params
    }

    /// Calculate memory requirements (in MB)
    pub fn estimate_memory_usage(&self) -> f64 {
        let param_count = self.estimate_parameter_count();
        let bytes_per_param = if std::mem::size_of::<T>() == 4 { 4.0 } else { 8.0 };

        let model_memory = param_count as f64 * bytes_per_param;
        let activation_memory = self.batch_size as f64 * self.sequence_length as f64 * self.model_dimension as f64 * bytes_per_param;
        let gradient_memory = model_memory; // Assume same as model for gradients

        let total_bytes = model_memory + activation_memory + gradient_memory;
        total_bytes / (1024.0 * 1024.0) // Convert to MB
    }
}

/// Transformer architecture configuration
#[derive(Debug, Clone)]
pub struct TransformerArchConfig {
    pub model_dimension: usize,
    pub num_layers: usize,
    pub num_attention_heads: usize,
    pub feedforward_dimension: usize,
    pub dropout_rate: f64,
    pub use_pre_norm: bool,
    pub enable_residual_connections: bool,
}

impl TransformerArchConfig {
    pub fn from_optimizer_config<T: Float>(config: &TransformerBasedOptimizerConfig<T>) -> Self {
        Self {
            model_dimension: config.model_dimension,
            num_layers: config.num_transformer_layers,
            num_attention_heads: config.num_attention_heads,
            feedforward_dimension: config.feedforward_dimension,
            dropout_rate: config.dropout_rate,
            use_pre_norm: config.use_pre_norm,
            enable_residual_connections: config.enable_residual_connections,
        }
    }
}

/// Activation function types
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ActivationFunction {
    ReLU,
    GELU,
    Swish,
    Tanh,
    Sigmoid,
    LeakyReLU,
}


/// Memory management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Maximum memory cache size
    pub max_cache_size: usize,
    /// Enable memory compression
    pub enable_compression: bool,
    /// Cache eviction strategy
    pub eviction_strategy: CacheEvictionStrategy,
    /// Memory allocation block size
    pub allocation_block_size: usize,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            max_cache_size: 1024 * 1024 * 1024, // 1GB
            enable_compression: false,
            eviction_strategy: CacheEvictionStrategy::LRU,
            allocation_block_size: 4096,
        }
    }
}

impl MemoryConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.max_cache_size == 0 {
            return Err("max_cache_size must be greater than 0".to_string());
        }

        if self.allocation_block_size == 0 {
            return Err("allocation_block_size must be greater than 0".to_string());
        }

        Ok(())
    }
}

/// Cache eviction strategies
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum CacheEvictionStrategy {
    LRU,
    LFU,
    FIFO,
    Random,
}

/// Meta-learning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaLearningConfig<T: Float> {
    /// Meta-learning rate
    pub meta_learning_rate: T,
    /// Number of inner optimization steps
    pub inner_steps: usize,
    /// Inner learning rate
    pub inner_learning_rate: T,
    /// Enable first-order approximation
    pub first_order: bool,
    /// Number of support examples
    pub num_support: usize,
    /// Number of query examples
    pub num_query: usize,
}

impl<T: Float> Default for MetaLearningConfig<T> {
    fn default() -> Self {
        Self {
            meta_learning_rate: T::from(1e-3).unwrap(),
            inner_steps: 5,
            inner_learning_rate: T::from(1e-2).unwrap(),
            first_order: false,
            num_support: 5,
            num_query: 15,
        }
    }
}

impl<T: Float> MetaLearningConfig<T> {
    pub fn validate(&self) -> Result<(), String> {
        if self.meta_learning_rate <= T::zero() {
            return Err("meta_learning_rate must be positive".to_string());
        }

        if self.inner_learning_rate <= T::zero() {
            return Err("inner_learning_rate must be positive".to_string());
        }

        if self.inner_steps == 0 {
            return Err("inner_steps must be greater than 0".to_string());
        }

        if self.num_support == 0 {
            return Err("num_support must be greater than 0".to_string());
        }

        if self.num_query == 0 {
            return Err("num_query must be greater than 0".to_string());
        }

        Ok(())
    }
}

/// Performance tracking configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Enable detailed performance tracking
    pub enable_detailed_tracking: bool,
    /// Performance metrics collection interval
    pub metrics_interval: usize,
    /// Maximum history size for metrics
    pub max_history_size: usize,
    /// Enable memory usage tracking
    pub track_memory_usage: bool,
    /// Enable timing analysis
    pub enable_timing_analysis: bool,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            enable_detailed_tracking: true,
            metrics_interval: 10,
            max_history_size: 10000,
            track_memory_usage: true,
            enable_timing_analysis: true,
        }
    }
}

impl PerformanceConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.metrics_interval == 0 {
            return Err("metrics_interval must be greater than 0".to_string());
        }

        if self.max_history_size == 0 {
            return Err("max_history_size must be greater than 0".to_string());
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = TransformerBasedOptimizerConfig::<f32>::default();
        assert!(config.validate().is_ok());
        assert_eq!(config.model_dimension, 512);
        assert_eq!(config.num_transformer_layers, 6);
    }

    #[test]
    fn test_config_validation() {
        let mut config = TransformerBasedOptimizerConfig::<f32>::default();

        // Test invalid model dimension
        config.model_dimension = 0;
        assert!(config.validate().is_err());

        // Test mismatched attention dimensions
        config.model_dimension = 512;
        config.num_attention_heads = 7; // 512 is not divisible by 7
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_parameter_estimation() {
        let config = TransformerBasedOptimizerConfig::<f32>::small();
        let param_count = config.estimate_parameter_count();
        assert!(param_count > 0);

        let memory_usage = config.estimate_memory_usage();
        assert!(memory_usage > 0.0);
    }

    #[test]
    fn test_preset_configs() {
        let small_config = TransformerBasedOptimizerConfig::<f32>::small();
        assert!(small_config.validate().is_ok());
        assert_eq!(small_config.model_dimension, 256);

        let large_config = TransformerBasedOptimizerConfig::<f32>::large();
        assert!(large_config.validate().is_ok());
        assert_eq!(large_config.model_dimension, 1024);

        let training_config = TransformerBasedOptimizerConfig::<f32>::for_training();
        assert!(training_config.validate().is_ok());
        assert_eq!(training_config.batch_size, 64);
    }
}