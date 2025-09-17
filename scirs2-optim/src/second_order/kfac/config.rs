//! Configuration structures for K-FAC optimizer
//!
//! This module contains configuration types and data structures for the
//! K-FAC (Kronecker-Factored Approximate Curvature) second-order optimizer.

use num_traits::Float;

/// K-FAC optimizer configuration
#[derive(Debug, Clone)]
pub struct KFACConfig<T: Float> {
    /// Learning rate
    pub learning_rate: T,

    /// Damping parameter for numerical stability
    pub damping: T,

    /// Weight decay (L2 regularization)
    pub weight_decay: T,

    /// Update frequency for covariance matrices
    pub cov_update_freq: usize,

    /// Update frequency for inverse covariance matrices
    pub inv_update_freq: usize,

    /// Exponential moving average decay for statistics
    pub stat_decay: T,

    /// Minimum eigenvalue for regularization
    pub min_eigenvalue: T,

    /// Maximum number of iterations for iterative inversion
    pub max_inv_iterations: usize,

    /// Tolerance for iterative inversion
    pub inv_tolerance: T,

    /// Use Tikhonov regularization
    pub use_tikhonov: bool,

    /// Enable automatic damping adjustment
    pub auto_damping: bool,

    /// Target acceptance ratio for damping adjustment
    pub target_acceptance_ratio: T,
}

impl<T: Float> Default for KFACConfig<T> {
    fn default() -> Self {
        Self {
            learning_rate: T::from(0.001).unwrap(),
            damping: T::from(0.001).unwrap(),
            weight_decay: T::from(0.0).unwrap(),
            cov_update_freq: 10,
            inv_update_freq: 100,
            stat_decay: T::from(0.95).unwrap(),
            min_eigenvalue: T::from(1e-7).unwrap(),
            max_inv_iterations: 50,
            inv_tolerance: T::from(1e-6).unwrap(),
            use_tikhonov: true,
            auto_damping: true,
            target_acceptance_ratio: T::from(0.75).unwrap(),
        }
    }
}

/// Layer information for K-FAC
#[derive(Debug, Clone)]
pub struct LayerInfo {
    /// Layer name/identifier
    pub name: String,

    /// Input dimension
    pub input_dim: usize,

    /// Output dimension
    pub output_dim: usize,

    /// Layer type
    pub layer_type: LayerType,

    /// Whether to include bias
    pub has_bias: bool,
}

/// Types of layers supported by K-FAC
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LayerType {
    /// Dense/Fully connected layer
    Dense,

    /// Convolutional layer
    Convolution,

    /// Convolutional layer with grouped/depthwise convolution
    GroupedConvolution { groups: usize },

    /// Embedding layer
    Embedding,

    /// Batch normalization layer
    BatchNorm,
}

/// K-FAC performance statistics
#[derive(Debug, Clone, Default)]
pub struct KFACStats<T: Float> {
    /// Total number of optimization steps
    pub total_steps: usize,

    /// Number of covariance updates
    pub cov_updates: usize,

    /// Number of inverse updates
    pub inv_updates: usize,

    /// Average condition number of covariance matrices
    pub avg_condition_number: T,

    /// Time spent in different operations (in microseconds)
    pub time_cov_update: u64,
    pub time_inv_update: u64,
    pub time_gradient_update: u64,

    /// Memory usage estimate (in bytes)
    pub memory_usage: usize,
}

impl<T: Float> KFACConfig<T> {
    /// Create configuration optimized for large models
    pub fn for_large_model() -> Self {
        let mut config = Self::default();
        config.cov_update_freq = 20;
        config.inv_update_freq = 200;
        config.stat_decay = T::from(0.99).unwrap();
        config.damping = T::from(0.01).unwrap();
        config
    }

    /// Create configuration optimized for small models with frequent updates
    pub fn for_small_model() -> Self {
        let mut config = Self::default();
        config.cov_update_freq = 5;
        config.inv_update_freq = 50;
        config.stat_decay = T::from(0.9).unwrap();
        config.damping = T::from(0.001).unwrap();
        config
    }

    /// Create configuration with conservative damping for stability
    pub fn for_stability() -> Self {
        let mut config = Self::default();
        config.damping = T::from(0.1).unwrap();
        config.min_eigenvalue = T::from(1e-5).unwrap();
        config.auto_damping = false;
        config.use_tikhonov = true;
        config
    }

    /// Validate configuration parameters
    pub fn validate(&self) -> Result<(), String> {
        if self.learning_rate <= T::zero() {
            return Err("Learning rate must be positive".to_string());
        }
        if self.damping < T::zero() {
            return Err("Damping must be non-negative".to_string());
        }
        if self.weight_decay < T::zero() {
            return Err("Weight decay must be non-negative".to_string());
        }
        if self.cov_update_freq == 0 {
            return Err("Covariance update frequency must be positive".to_string());
        }
        if self.inv_update_freq == 0 {
            return Err("Inverse update frequency must be positive".to_string());
        }
        if self.stat_decay < T::zero() || self.stat_decay > T::one() {
            return Err("Statistics decay must be between 0 and 1".to_string());
        }
        if self.min_eigenvalue <= T::zero() {
            return Err("Minimum eigenvalue must be positive".to_string());
        }
        if self.inv_tolerance <= T::zero() {
            return Err("Inverse tolerance must be positive".to_string());
        }
        if self.target_acceptance_ratio <= T::zero() || self.target_acceptance_ratio >= T::one() {
            return Err("Target acceptance ratio must be between 0 and 1".to_string());
        }
        Ok(())
    }
}

impl LayerInfo {
    /// Create layer info for a dense layer
    pub fn dense(name: String, input_dim: usize, output_dim: usize, has_bias: bool) -> Self {
        Self {
            name,
            input_dim,
            output_dim,
            layer_type: LayerType::Dense,
            has_bias,
        }
    }

    /// Create layer info for a convolutional layer
    pub fn convolution(name: String, input_dim: usize, output_dim: usize, has_bias: bool) -> Self {
        Self {
            name,
            input_dim,
            output_dim,
            layer_type: LayerType::Convolution,
            has_bias,
        }
    }

    /// Get the expected size of the input covariance matrix
    pub fn input_cov_size(&self) -> usize {
        match self.layer_type {
            LayerType::Dense => {
                if self.has_bias {
                    self.input_dim + 1
                } else {
                    self.input_dim
                }
            }
            LayerType::Convolution => {
                // For convolutional layers, this depends on the specific implementation
                // This is a simplified calculation
                self.input_dim
            }
            LayerType::GroupedConvolution { .. } => self.input_dim,
            LayerType::Embedding => self.input_dim,
            LayerType::BatchNorm => self.input_dim,
        }
    }

    /// Get the expected size of the output gradient covariance matrix
    pub fn output_cov_size(&self) -> usize {
        self.output_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = KFACConfig::<f32>::default();
        assert!(config.learning_rate > 0.0);
        assert!(config.damping >= 0.0);
        assert!(config.cov_update_freq > 0);
        assert!(config.inv_update_freq > 0);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_presets() {
        let large_config = KFACConfig::<f64>::for_large_model();
        assert!(large_config.validate().is_ok());
        assert_eq!(large_config.cov_update_freq, 20);

        let small_config = KFACConfig::<f64>::for_small_model();
        assert!(small_config.validate().is_ok());
        assert_eq!(small_config.cov_update_freq, 5);

        let stable_config = KFACConfig::<f64>::for_stability();
        assert!(stable_config.validate().is_ok());
        assert!(!stable_config.auto_damping);
    }

    #[test]
    fn test_layer_info_creation() {
        let dense_layer = LayerInfo::dense("layer1".to_string(), 128, 64, true);
        assert_eq!(dense_layer.layer_type, LayerType::Dense);
        assert_eq!(dense_layer.input_cov_size(), 129); // +1 for bias

        let conv_layer = LayerInfo::convolution("conv1".to_string(), 64, 32, false);
        assert_eq!(conv_layer.layer_type, LayerType::Convolution);
        assert_eq!(conv_layer.input_cov_size(), 64);
    }

    #[test]
    fn test_config_validation() {
        let mut config = KFACConfig::<f32>::default();

        config.learning_rate = -0.1;
        assert!(config.validate().is_err());

        config.learning_rate = 0.001;
        config.damping = -0.1;
        assert!(config.validate().is_err());

        config.damping = 0.001;
        config.stat_decay = 1.5;
        assert!(config.validate().is_err());

        config.stat_decay = 0.95;
        assert!(config.validate().is_ok());
    }
}