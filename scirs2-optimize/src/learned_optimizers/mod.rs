//! Learned Optimizers Module
//!
//! This module implements optimization algorithms that learn to optimize, including:
//! - Meta-learning based optimizers that adapt to different problem types
//! - Neural Architecture Search (NAS) systems
//! - Transformer-based optimization enhancements
//! - Few-shot learning for optimization
//! - Adaptive neural optimizers
//! - Learned hyperparameter tuning systems

use crate::error::OptimizeError;
use crate::error::OptimizeResult;
use crate::result::OptimizeResults;
use ndarray::{Array1, Array2, ArrayView1};
use rand::{rng, Rng};
use statrs::statistics::Statistics;
use std::collections::HashMap;

type Result<T> = std::result::Result<T, OptimizeError>;

pub mod adaptive_nas_system;
pub mod adaptive_transformer_enhancement;
pub mod few_shot_learning_enhancement;
pub mod learned_hyperparameter_tuner;
pub mod meta_learning_optimizer;
pub mod neural_adaptive_optimizer;

// Use glob re-exports with allow for ambiguous names
#[allow(ambiguous_glob_reexports)]
pub use adaptive_nas_system::*;
#[allow(ambiguous_glob_reexports)]
pub use adaptive_transformer_enhancement::*;
#[allow(ambiguous_glob_reexports)]
pub use few_shot_learning_enhancement::*;
#[allow(ambiguous_glob_reexports)]
pub use learned_hyperparameter_tuner::*;
#[allow(ambiguous_glob_reexports)]
pub use meta_learning_optimizer::*;
#[allow(ambiguous_glob_reexports)]
pub use neural_adaptive_optimizer::*;

/// Configuration for learned optimizers
#[derive(Debug, Clone)]
pub struct LearnedOptimizationConfig {
    /// Number of meta-training episodes
    pub meta_training_episodes: usize,
    /// Learning rate for meta-optimizer
    pub meta_learning_rate: f64,
    /// Number of inner optimization steps
    pub inner_steps: usize,
    /// Inner learning rate
    pub inner_learning_rate: f64,
    /// Batch size for meta-learning
    pub batch_size: usize,
    /// Maximum number of parameters to optimize
    pub max_parameters: usize,
    /// Whether to use transformer architecture
    pub use_transformer: bool,
    /// Hidden size for neural networks
    pub hidden_size: usize,
    /// Number of attention heads (for transformer)
    pub num_heads: usize,
    /// Whether to enable few-shot adaptation
    pub few_shot_adaptation: bool,
    /// Temperature for exploration
    pub exploration_temperature: f64,
}

impl Default for LearnedOptimizationConfig {
    fn default() -> Self {
        Self {
            meta_training_episodes: 10000,
            meta_learning_rate: 0.001,
            inner_steps: 10,
            inner_learning_rate: 0.01,
            batch_size: 32,
            max_parameters: 1000,
            use_transformer: true,
            hidden_size: 256,
            num_heads: 8,
            few_shot_adaptation: true,
            exploration_temperature: 1.0,
        }
    }
}

/// Meta-learning problem specification
#[derive(Debug, Clone)]
pub struct OptimizationProblem {
    /// Problem identifier
    pub name: String,
    /// Problem dimensionality
    pub dimension: usize,
    /// Problem class (e.g., "quadratic", "neural_network", "sparse")
    pub problem_class: String,
    /// Additional metadata
    pub metadata: HashMap<String, f64>,
    /// Function evaluations budget
    pub max_evaluations: usize,
    /// Target accuracy
    pub target_accuracy: f64,
}

/// Training task for meta-learning
#[derive(Debug, Clone)]
pub struct TrainingTask {
    /// Problem specification
    pub problem: OptimizationProblem,
    /// Initial parameter distribution
    pub initial_distribution: ParameterDistribution,
    /// Ground truth optimum (if known)
    pub true_optimum: Option<Array1<f64>>,
    /// Task difficulty weight
    pub difficulty_weight: f64,
}

/// Parameter distribution for initialization
#[derive(Debug, Clone)]
pub enum ParameterDistribution {
    /// Uniform distribution in range
    Uniform { low: f64, high: f64 },
    /// Normal distribution
    Normal { mean: f64, std: f64 },
    /// Custom distribution from samples
    Custom { samples: Vec<Array1<f64>> },
}

/// Meta-optimizer state
#[derive(Debug, Clone)]
pub struct MetaOptimizerState {
    /// Current meta-parameters
    pub meta_params: Array1<f64>,
    /// Optimizer network weights
    pub network_weights: Array2<f64>,
    /// Performance history
    pub performance_history: Vec<f64>,
    /// Adaptation statistics
    pub adaptation_stats: AdaptationStatistics,
    /// Current episode
    pub episode: usize,
}

/// Statistics for tracking adaptation
#[derive(Debug, Clone)]
pub struct AdaptationStatistics {
    /// Average convergence rate
    pub avg_convergence_rate: f64,
    /// Success rate across problems
    pub success_rate: f64,
    /// Average function evaluations used
    pub avg_evaluations: f64,
    /// Transfer learning efficiency
    pub transfer_efficiency: f64,
    /// Exploration-exploitation balance
    pub exploration_ratio: f64,
}

impl Default for AdaptationStatistics {
    fn default() -> Self {
        Self {
            avg_convergence_rate: 0.0,
            success_rate: 0.0,
            avg_evaluations: 0.0,
            transfer_efficiency: 0.0,
            exploration_ratio: 0.5,
        }
    }
}

/// Trait for learned optimizers
pub trait LearnedOptimizer {
    /// Meta-train the optimizer on a distribution of problems
    fn meta_train(&mut self, training_tasks: &[TrainingTask]) -> Result<()>;

    /// Adapt to a new problem with few-shot learning
    fn adapt_to_problem(
        &mut self,
        problem: &OptimizationProblem,
        initial_params: &ArrayView1<f64>,
    ) -> Result<()>;

    /// Optimize a function using learned knowledge
    fn optimize<F>(
        &mut self,
        objective: F,
        initial_params: &ArrayView1<f64>,
    ) -> OptimizeResult<OptimizeResults<f64>>
    where
        F: Fn(&ArrayView1<f64>) -> f64;

    /// Get current meta-optimizer state
    fn get_state(&self) -> &MetaOptimizerState;

    /// Reset the optimizer
    fn reset(&mut self);
}

/// Neural network for learned optimization
#[derive(Debug, Clone)]
pub struct OptimizationNetwork {
    /// Input embedding layer
    pub input_embedding: Array2<f64>,
    /// Hidden layers
    pub hidden_layers: Vec<Array2<f64>>,
    /// Output layer
    pub output_layer: Array2<f64>,
    /// Attention weights (if using transformer)
    pub attention_weights: Option<Vec<Array2<f64>>>,
    /// Layer normalization parameters
    pub layer_norms: Vec<LayerNorm>,
    /// Activation function type
    pub activation: ActivationType,
}

/// Layer normalization parameters
#[derive(Debug, Clone)]
pub struct LayerNorm {
    /// Scale parameter
    pub gamma: Array1<f64>,
    /// Shift parameter
    pub beta: Array1<f64>,
    /// Small constant for numerical stability
    pub epsilon: f64,
}

/// Types of activation functions
#[derive(Debug, Clone, Copy)]
pub enum ActivationType {
    ReLU,
    GELU,
    Swish,
    Tanh,
    LeakyReLU,
}

impl ActivationType {
    pub fn apply(&self, x: f64) -> f64 {
        match self {
            ActivationType::ReLU => x.max(0.0),
            ActivationType::GELU => {
                x * 0.5 * (1.0 + (x * 0.7978845608 * (1.0 + 0.044715 * x * x)).tanh())
            }
            ActivationType::Swish => x / (1.0 + (-x).exp()),
            ActivationType::Tanh => x.tanh(),
            ActivationType::LeakyReLU => {
                if x > 0.0 {
                    x
                } else {
                    0.01 * x
                }
            }
        }
    }

    pub fn derivative(&self, x: f64) -> f64 {
        match self {
            ActivationType::ReLU => {
                if x > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            ActivationType::GELU => {
                let tanh_arg = x * 0.7978845608 * (1.0 + 0.044715 * x * x);
                let tanh_val = tanh_arg.tanh();
                0.5 * (1.0 + tanh_val)
                    + x * 0.5
                        * (1.0 - tanh_val * tanh_val)
                        * 0.7978845608
                        * (1.0 + 0.134145 * x * x)
            }
            ActivationType::Swish => {
                let sigmoid = 1.0 / (1.0 + (-x).exp());
                sigmoid * (1.0 + x * (1.0 - sigmoid))
            }
            ActivationType::Tanh => {
                let t = x.tanh();
                1.0 - t * t
            }
            ActivationType::LeakyReLU => {
                if x > 0.0 {
                    1.0
                } else {
                    0.01
                }
            }
        }
    }
}

impl OptimizationNetwork {
    /// Create new optimization network
    pub fn new(
        input_size: usize,
        hidden_sizes: Vec<usize>,
        output_size: usize,
        use_attention: bool,
        activation: ActivationType,
    ) -> Self {
        let mut hidden_layers = Vec::new();
        let mut layer_norms = Vec::new();

        // Create layers
        let mut prev_size = input_size;
        for &hidden_size in &hidden_sizes {
            let weights = Array2::from_shape_fn((hidden_size, prev_size), |_| {
                rand::rng().random_range(-0.5..0.5) * (2.0 / prev_size as f64).sqrt()
            });
            hidden_layers.push(weights);

            // Layer normalization
            layer_norms.push(LayerNorm {
                gamma: Array1::ones(hidden_size),
                beta: Array1::zeros(hidden_size),
                epsilon: 1e-6,
            });

            prev_size = hidden_size;
        }

        // Input embedding
        let input_embedding = Array2::from_shape_fn((hidden_sizes[0], input_size), |_| {
            rand::rng().random_range(-0.5..0.5) * (2.0 / input_size as f64).sqrt()
        });

        // Output layer
        let output_layer = Array2::from_shape_fn((output_size, prev_size), |_| {
            rand::rng().random_range(-0.5..0.5) * (2.0 / prev_size as f64).sqrt()
        });

        // Attention weights (simplified)
        let attention_weights = if use_attention {
            Some(vec![Array2::from_shape_fn((prev_size, prev_size), |_| {
                rand::rng().random_range(-0.5..0.5) * (2.0 / prev_size as f64).sqrt()
            })])
        } else {
            None
        };

        Self {
            input_embedding,
            hidden_layers,
            output_layer,
            attention_weights,
            layer_norms,
            activation,
        }
    }

    /// Forward pass through the network
    pub fn forward(&self, input: &ArrayView1<f64>) -> Array1<f64> {
        // Input embedding
        let mut current = Array1::zeros(self.input_embedding.nrows());
        for i in 0..current.len() {
            for j in 0..input.len().min(self.input_embedding.ncols()) {
                current[i] += self.input_embedding[[i, j]] * input[j];
            }
        }

        // Apply activation
        current.mapv_inplace(|x| self.activation.apply(x));

        // Hidden layers with layer normalization
        for (layer_idx, layer) in self.hidden_layers.iter().enumerate() {
            let mut next = Array1::zeros(layer.nrows());

            // Linear transformation
            for i in 0..next.len() {
                for j in 0..current.len().min(layer.ncols()) {
                    next[i] += layer[[i, j]] * current[j];
                }
            }

            // Layer normalization
            if layer_idx < self.layer_norms.len() {
                let layer_norm = &self.layer_norms[layer_idx];
                let mean = next.view().mean();
                let var = next.view().variance();
                let std = (var + layer_norm.epsilon).sqrt();

                for i in 0..next.len() {
                    if i < layer_norm.gamma.len() && i < layer_norm.beta.len() {
                        next[i] = layer_norm.gamma[i] * (next[i] - mean) / std + layer_norm.beta[i];
                    }
                }
            }

            // Apply attention (simplified)
            if let Some(ref attention) = self.attention_weights {
                if !attention.is_empty() {
                    let attn_weights = &attention[0];
                    let mut attended: Array1<f64> = Array1::zeros(next.len());

                    for i in 0..attended.len() {
                        for j in 0..next.len().min(attn_weights.ncols()) {
                            attended[i] += attn_weights[[i, j]] * next[j];
                        }
                    }

                    // Residual connection
                    next = &next + &attended;
                }
            }

            // Activation
            next.mapv_inplace(|x| self.activation.apply(x));
            current = next;
        }

        // Output layer
        let mut output = Array1::zeros(self.output_layer.nrows());
        for i in 0..output.len() {
            for j in 0..current.len().min(self.output_layer.ncols()) {
                output[i] += self.output_layer[[i, j]] * current[j];
            }
        }

        output
    }
}

/// Problem encoder for creating embeddings
#[derive(Debug, Clone)]
pub struct ProblemEncoder {
    /// Dimensionality features
    pub dim_encoder: Array2<f64>,
    /// Gradient features encoder
    pub gradient_encoder: Array2<f64>,
    /// Hessian features encoder
    pub hessian_encoder: Array2<f64>,
    /// Output embedding size
    pub embedding_size: usize,
}

impl ProblemEncoder {
    /// Create new problem encoder
    pub fn new(embedding_size: usize) -> Self {
        let dim = 10; // Feature dimensions for different aspects

        Self {
            dim_encoder: Array2::from_shape_fn((embedding_size, dim), |_| {
                rand::rng().random_range(-0.5..0.5) * 0.1
            }),
            gradient_encoder: Array2::from_shape_fn((embedding_size, dim), |_| {
                rand::rng().random_range(-0.5..0.5) * 0.1
            }),
            hessian_encoder: Array2::from_shape_fn((embedding_size, dim), |_| {
                rand::rng().random_range(-0.5..0.5) * 0.1
            }),
            embedding_size,
        }
    }

    /// Encode a problem into an embedding
    pub fn encode_problem<F>(
        &self,
        objective: &F,
        current_params: &ArrayView1<f64>,
        problem: &OptimizationProblem,
    ) -> Array1<f64>
    where
        F: Fn(&ArrayView1<f64>) -> f64,
    {
        let mut embedding = Array1::zeros(self.embedding_size);

        // Compute basic features
        let dim_features = self.compute_dimensionality_features(current_params, problem);
        let grad_features = self.compute_gradient_features(objective, current_params);
        let hessian_features = self.compute_hessian_features(objective, current_params);

        // Combine features
        for i in 0..self.embedding_size {
            for j in 0..dim_features.len().min(self.dim_encoder.ncols()) {
                embedding[i] += self.dim_encoder[[i, j]] * dim_features[j];
            }
            for j in 0..grad_features.len().min(self.gradient_encoder.ncols()) {
                embedding[i] += self.gradient_encoder[[i, j]] * grad_features[j];
            }
            for j in 0..hessian_features.len().min(self.hessian_encoder.ncols()) {
                embedding[i] += self.hessian_encoder[[i, j]] * hessian_features[j];
            }
        }

        embedding
    }

    fn compute_dimensionality_features(
        &self,
        params: &ArrayView1<f64>,
        problem: &OptimizationProblem,
    ) -> Array1<f64> {
        let mut features = Array1::zeros(10);

        features[0] = (params.len() as f64).ln(); // Log dimensionality
        features[1] = params.view().variance(); // Parameter variance
        features[2] = params.view().mean(); // Parameter mean
        features[3] = params.iter().map(|&x| x.abs()).sum::<f64>() / params.len() as f64; // L1 norm
        features[4] = (params.iter().map(|&x| x * x).sum::<f64>()).sqrt(); // L2 norm
        features[5] = problem.dimension as f64 / 1000.0; // Normalized dimension
        features[6] = problem.max_evaluations as f64 / 10000.0; // Normalized budget
        features[7] = problem.target_accuracy.ln().abs(); // Log target accuracy

        // Add problem-specific metadata
        if let Some(&complexity) = problem.metadata.get("complexity") {
            features[8] = complexity.tanh();
        }
        if let Some(&sparsity) = problem.metadata.get("sparsity") {
            features[9] = sparsity;
        }

        features
    }

    fn compute_gradient_features<F>(&self, objective: &F, params: &ArrayView1<f64>) -> Array1<f64>
    where
        F: Fn(&ArrayView1<f64>) -> f64,
    {
        let mut features = Array1::zeros(10);
        let h = 1e-6;
        let f0 = objective(params);

        let mut gradient_norm = 0.0;
        let mut gradient_components = Vec::new();

        // Compute finite difference gradient
        for i in 0..params.len().min(20) {
            // Limit for efficiency
            let mut params_plus = params.to_owned();
            params_plus[i] += h;
            let f_plus = objective(&params_plus.view());
            let grad_i = (f_plus - f0) / h;
            gradient_components.push(grad_i);
            gradient_norm += grad_i * grad_i;
        }

        gradient_norm = gradient_norm.sqrt();

        features[0] = gradient_norm.ln().tanh(); // Log gradient norm
        features[1] = f0.abs().ln().tanh(); // Log objective value

        if !gradient_components.is_empty() {
            let grad_mean =
                gradient_components.iter().sum::<f64>() / gradient_components.len() as f64;
            let grad_var = gradient_components
                .iter()
                .map(|&g| (g - grad_mean).powi(2))
                .sum::<f64>()
                / gradient_components.len() as f64;

            features[2] = grad_mean.tanh();
            features[3] = grad_var.sqrt().tanh();
            features[4] = gradient_components
                .iter()
                .map(|&g| g.abs())
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(0.0)
                .tanh();
            features[5] = gradient_components
                .iter()
                .map(|&g| g.abs())
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(0.0)
                .tanh();
        }

        features
    }

    fn compute_hessian_features<F>(&self, objective: &F, params: &ArrayView1<f64>) -> Array1<f64>
    where
        F: Fn(&ArrayView1<f64>) -> f64,
    {
        let mut features = Array1::zeros(10);
        let h = 1e-4;
        let f0 = objective(params);

        // Compute a few diagonal Hessian elements for efficiency
        for i in 0..params.len().min(5) {
            let mut params_plus = params.to_owned();
            let mut params_minus = params.to_owned();

            params_plus[i] += h;
            params_minus[i] -= h;

            let f_plus = objective(&params_plus.view());
            let f_minus = objective(&params_minus.view());

            let hessian_ii = (f_plus - 2.0 * f0 + f_minus) / (h * h);

            if i < features.len() {
                features[i] = hessian_ii.tanh();
            }
        }

        features
    }
}

/// Convenience function for learned optimization
#[allow(dead_code)]
pub fn learned_optimize<F>(
    objective: F,
    initial_params: &ArrayView1<f64>,
    config: Option<LearnedOptimizationConfig>,
) -> OptimizeResult<OptimizeResults<f64>>
where
    F: Fn(&ArrayView1<f64>) -> f64,
{
    let config = config.unwrap_or_default();

    // Create meta-learning optimizer
    let mut optimizer = MetaLearningOptimizer::new(config);

    // Simple optimization without extensive meta-training
    optimizer.optimize(objective, initial_params)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_learned_optimization_config() {
        let config = LearnedOptimizationConfig::default();
        assert_eq!(config.meta_training_episodes, 10000);
        assert_eq!(config.hidden_size, 256);
        assert!(config.use_transformer);
    }

    #[test]
    fn test_optimization_network_creation() {
        let network = OptimizationNetwork::new(10, vec![32, 32], 5, true, ActivationType::GELU);

        assert_eq!(network.hidden_layers.len(), 2);
        assert_eq!(network.layer_norms.len(), 2);
        assert!(network.attention_weights.is_some());
    }

    #[test]
    fn test_network_forward_pass() {
        let network = OptimizationNetwork::new(5, vec![10], 3, false, ActivationType::ReLU);

        let input = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let output = network.forward(&input.view());

        assert_eq!(output.len(), 3);
        assert!(output.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_activation_functions() {
        assert_eq!(ActivationType::ReLU.apply(-1.0), 0.0);
        assert_eq!(ActivationType::ReLU.apply(1.0), 1.0);
        assert!(ActivationType::GELU.apply(0.0).abs() < 0.1);
        assert!(ActivationType::Swish.apply(0.0).abs() < 0.1);
    }

    #[test]
    fn test_problem_encoder() {
        let encoder = ProblemEncoder::new(32);
        let params = Array1::from(vec![1.0, 2.0]);
        let objective = |x: &ArrayView1<f64>| x[0].powi(2) + x[1].powi(2);

        let problem = OptimizationProblem {
            name: "test".to_string(),
            dimension: 2,
            problem_class: "quadratic".to_string(),
            metadata: HashMap::new(),
            max_evaluations: 1000,
            target_accuracy: 1e-6,
        };

        let embedding = encoder.encode_problem(&objective, &params.view(), &problem);
        assert_eq!(embedding.len(), 32);
        assert!(embedding.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_basic_learned_optimization() {
        let objective = |x: &ArrayView1<f64>| x[0].powi(2) + x[1].powi(2);
        let initial = Array1::from(vec![2.0, 2.0]);

        let config = LearnedOptimizationConfig {
            meta_training_episodes: 10,
            inner_steps: 5,
            ..Default::default()
        };

        let result = learned_optimize(objective, &initial.view(), Some(config)).unwrap();

        assert!(result.fun >= 0.0);
        assert_eq!(result.x.len(), 2);
    }
}

#[allow(dead_code)]
pub fn placeholder() {
    // Placeholder function to prevent unused module warnings
}
