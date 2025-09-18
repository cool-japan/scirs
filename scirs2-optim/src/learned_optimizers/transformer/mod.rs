//! Transformer-based learned optimizer
//!
//! This module provides a modular implementation of a transformer-based neural optimizer
//! that uses self-attention mechanisms to adaptively update optimization parameters.
//! The implementation is organized into architectural components, optimization strategies,
//! and training infrastructure for improved maintainability and extensibility.

pub mod architecture;
pub mod strategies;
pub mod training;

// Re-export key types for convenience
pub use architecture::{
    MultiHeadAttention, TransformerLayer, FeedForwardNetwork, LayerNorm,
    PositionalEncoder, PositionalEncodingType, AttentionOptimization,
    OutputProjectionLayer, InputEmbedding, ActivationFunction
};

pub use strategies::{
    GradientProcessor, GradientProcessingStrategy, 
    LearningRateAdapter, LearningRateAdaptationStrategy,
    MomentumIntegrator, MomentumStrategy,
    TransformerRegularizer, RegularizationStrategy
};

pub use training::{
    TransformerMetaLearner, MetaLearningStrategy,
    CurriculumLearner, CurriculumStrategy,
    TransformerEvaluator, EvaluationStrategy
};

use ndarray::{Array1, Array2};
use num_traits::Float;
use std::collections::{HashMap, VecDeque};

use crate::error::{OptimError, Result};
use super::{LearnedOptimizerConfig, MetaOptimizationStrategy};

/// Configuration specific to Transformer optimizer
#[derive(Debug, Clone)]
pub struct TransformerOptimizerConfig {
    /// Base learned optimizer config
    pub base_config: LearnedOptimizerConfig,
    
    /// Model dimension (d_model)
    pub modeldim: usize,
    
    /// Number of attention heads
    pub numheads: usize,
    
    /// Feed-forward network dimension
    pub ff_dim: usize,
    
    /// Number of transformer layers
    pub num_layers: usize,
    
    /// Maximum sequence length
    pub max_sequence_length: usize,
    
    /// Attention dropout rate
    pub attention_dropout: f64,
    
    /// Feed-forward dropout rate
    pub ff_dropout: f64,
    
    /// Layer normalization epsilon
    pub layer_norm_eps: f64,
    
    /// Use pre-layer normalization
    pub pre_layer_norm: bool,
    
    /// Positional encoding type
    pub pos_encoding_type: PositionalEncodingType,
    
    /// Enable relative position bias
    pub relative_position_bias: bool,
    
    /// Use rotary position embedding
    pub use_rope: bool,
    
    /// Enable gradient checkpointing
    pub gradient_checkpointing: bool,
    
    /// Attention pattern optimization
    pub attention_optimization: AttentionOptimization,
    
    /// Multi-scale attention
    pub multi_scale_attention: bool,
    
    /// Cross-attention for multi-task learning
    pub cross_attention: bool,
    
    /// Memory efficiency mode
    pub memory_efficient: bool,
}

/// Transformer network architecture
#[derive(Debug, Clone)]
pub struct TransformerNetwork<T: Float> {
    /// Input embedding layer
    input_embedding: InputEmbedding<T>,
    
    /// Transformer layers
    layers: Vec<TransformerLayer<T>>,
    
    /// Output projection
    output_projection: OutputProjectionLayer<T>,
    
    /// Layer normalization for output
    output_layer_norm: LayerNorm<T>,
    
    /// Position encoder
    position_encoder: PositionalEncoder<T>,
    
    /// Configuration
    config: TransformerOptimizerConfig,
}

/// Transformer-based neural optimizer with self-attention mechanisms
#[derive(Debug, Clone)]
pub struct TransformerOptimizer<T: Float> {
    /// Configuration for the Transformer optimizer
    config: TransformerOptimizerConfig,
    
    /// Transformer network architecture
    transformer_network: TransformerNetwork<T>,
    
    /// Gradient processing strategies
    gradient_processor: GradientProcessor<T>,
    
    /// Learning rate adaptation
    lr_adapter: LearningRateAdapter<T>,
    
    /// Momentum integration
    momentum_integrator: MomentumIntegrator<T>,
    
    /// Regularization strategies
    regularizer: TransformerRegularizer<T>,
    
    /// Meta-learning components
    meta_learner: TransformerMetaLearner<T>,
    
    /// Curriculum learning
    curriculum_learner: CurriculumLearner<T>,
    
    /// Evaluation framework
    evaluator: TransformerEvaluator<T>,
    
    /// Sequence buffer for maintaining optimization history
    sequence_buffer: SequenceBuffer<T>,
    
    /// Performance metrics
    metrics: TransformerOptimizerMetrics,
    
    /// Current optimization step
    step_count: usize,
    
    /// Random number generator
    rng: scirs2_core::random::Random,
}

/// Sequence buffer for optimization history
#[derive(Debug, Clone)]
pub struct SequenceBuffer<T: Float> {
    /// Gradient sequences
    gradient_sequences: VecDeque<Array1<T>>,
    
    /// Parameter sequences
    parameter_sequences: VecDeque<Array1<T>>,
    
    /// Loss sequences
    loss_sequences: VecDeque<T>,
    
    /// Learning rate sequences
    lr_sequences: VecDeque<T>,
    
    /// Buffer capacity
    capacity: usize,
}

/// Performance metrics for transformer optimizer
#[derive(Debug, Clone)]
pub struct TransformerOptimizerMetrics {
    /// Total optimization steps
    total_steps: usize,
    
    /// Convergence history
    convergence_history: Vec<f64>,
    
    /// Attention pattern statistics
    attention_stats: HashMap<String, f64>,
    
    /// Strategy usage statistics
    strategy_stats: HashMap<String, f64>,
    
    /// Performance comparisons
    performance_comparisons: HashMap<String, f64>,
}

impl<T: Float + Default + Clone> TransformerNetwork<T> {
    /// Create new transformer network
    pub fn new(config: &TransformerOptimizerConfig) -> Result<Self> {
        let input_embedding = InputEmbedding::new(config.modeldim, config.modeldim)?;
        
        let mut layers = Vec::new();
        for _ in 0..config.num_layers {
            let mut rng = scirs2_core::random::rng();
            layers.push(TransformerLayer::new(config, &mut rng)?);
        }
        
        let output_projection = OutputProjectionLayer::new(config.modeldim, config.modeldim)?;
        let output_layer_norm = LayerNorm::new(config.modeldim);
        let position_encoder = PositionalEncoder::new(config)?;
        
        Ok(Self {
            input_embedding,
            layers,
            output_projection,
            output_layer_norm,
            position_encoder,
            config: config.clone(),
        })
    }
    
    /// Forward pass through transformer network
    pub fn forward(&mut self, input: &Array2<T>) -> Result<Array2<T>> {
        // Input embedding
        let mut x = self.input_embedding.forward(input)?;
        
        // Add positional encoding
        x = self.position_encoder.encode(&x)?;
        
        // Pass through transformer layers
        for layer in &mut self.layers {
            x = layer.forward(&x)?;
        }
        
        // Output layer normalization
        x = self.output_layer_norm.forward(&x)?;
        
        // Output projection
        let output = self.output_projection.forward(&x)?;
        
        Ok(output)
    }
    
    /// Get attention patterns from all layers
    pub fn get_attention_patterns(&self) -> Vec<Option<&ndarray::Array3<T>>> {
        self.layers.iter()
            .map(|layer| layer.get_attention_patterns())
            .collect()
    }
}

impl<T: Float + Default + Clone> TransformerOptimizer<T> {
    /// Create new transformer optimizer
    pub fn new(config: TransformerOptimizerConfig) -> Result<Self> {
        let transformer_network = TransformerNetwork::new(&config)?;
        let gradient_processor = GradientProcessor::new(GradientProcessingStrategy::Adaptive);
        let lr_adapter = LearningRateAdapter::new(
            LearningRateAdaptationStrategy::TransformerPredicted, 
            num_traits::cast::cast(0.001).unwrap_or_else(|| T::zero())
        );
        let momentum_integrator = MomentumIntegrator::new(MomentumStrategy::TransformerPredicted);
        let regularizer = TransformerRegularizer::new(RegularizationStrategy::Adaptive);
        let meta_learner = TransformerMetaLearner::new(MetaLearningStrategy::GradientBased)?;
        let curriculum_learner = CurriculumLearner::new(CurriculumStrategy::Adaptive)?;
        let evaluator = TransformerEvaluator::new(EvaluationStrategy::Comprehensive)?;
        let sequence_buffer = SequenceBuffer::new(1000);
        let metrics = TransformerOptimizerMetrics::new();
        
        Ok(Self {
            config,
            transformer_network,
            gradient_processor,
            lr_adapter,
            momentum_integrator,
            regularizer,
            meta_learner,
            curriculum_learner,
            evaluator,
            sequence_buffer,
            metrics,
            step_count: 0,
            rng: scirs2_core::random::rng(),
        })
    }
    
    /// Perform optimization step
    pub fn step(
        &mut self,
        parameters: &mut HashMap<String, Array2<T>>,
        gradients: &mut HashMap<String, Array2<T>>,
        loss: T
    ) -> Result<T> {
        self.step_count += 1;
        
        // Process gradients for each parameter
        for (param_name, gradient) in gradients.iter_mut() {
            // Flatten gradient for processing
            let flat_gradient = gradient.iter().cloned().collect::<Vec<_>>();
            let gradient_array = Array1::from_vec(flat_gradient);
            
            // Apply gradient processing
            let processed_gradient = self.gradient_processor.process_gradients(&gradient_array)?;
            
            // Update learning rate
            let current_lr = self.lr_adapter.update_learning_rate(
                Some(loss), 
                Some(&processed_gradient)
            )?;
            
            // Apply momentum
            let momentum_gradient = self.momentum_integrator.integrate_momentum(
                &processed_gradient,
                None // Would pass attention patterns in full implementation
            )?;
            
            // Apply regularization
            let mut param_map = HashMap::new();
            if let Some(param_values) = parameters.get(param_name) {
                param_map.insert(param_name.clone(), param_values.clone());
            }
            
            let mut grad_map = HashMap::new();
            grad_map.insert(param_name.clone(), gradient.clone());
            
            let _reg_loss = self.regularizer.apply_regularization(
                &param_map,
                &mut grad_map,
                None // Would pass attention patterns in full implementation
            )?;
            
            // Store processed gradients in sequence buffer
            self.sequence_buffer.add_gradient(momentum_gradient);
        }
        
        // Update sequence buffer with loss and learning rate
        self.sequence_buffer.add_loss(loss);
        self.sequence_buffer.add_learning_rate(self.lr_adapter.current_learning_rate());
        
        // Update curriculum learning
        let task_id = "current_task"; // In practice, this would be provided
        self.curriculum_learner.update_curriculum(task_id, loss, self.step_count)?;
        
        // Update metrics
        self.metrics.update_step(loss.to_f64().unwrap_or(0.0), self.step_count);
        
        Ok(loss)
    }
    
    /// Get current optimization statistics
    pub fn get_statistics(&self) -> HashMap<String, f64> {
        let mut stats = HashMap::new();
        
        stats.insert("step_count".to_string(), self.step_count as f64);
        stats.insert("current_lr".to_string(), self.lr_adapter.current_learning_rate().to_f64().unwrap_or(0.0));
        
        // Add gradient processor statistics
        let grad_stats = self.gradient_processor.statistics();
        stats.insert("mean_gradient_magnitude".to_string(), grad_stats.mean_magnitude().to_f64().unwrap_or(0.0));
        stats.insert("gradient_sparsity".to_string(), grad_stats.sparsity().to_f64().unwrap_or(0.0));
        
        // Add momentum statistics
        let momentum_stats = self.momentum_integrator.statistics();
        stats.insert("momentum_magnitude".to_string(), momentum_stats.avg_momentum_magnitude.to_f64().unwrap_or(0.0));
        
        // Add curriculum statistics
        let curriculum_stats = self.curriculum_learner.get_curriculum_statistics();
        for (key, value) in curriculum_stats {
            stats.insert(format!("curriculum_{}", key), value.to_f64().unwrap_or(0.0));
        }
        
        stats
    }
    
    /// Reset optimizer state
    pub fn reset(&mut self) -> Result<()> {
        self.step_count = 0;
        self.gradient_processor.reset();
        self.lr_adapter.reset();
        self.momentum_integrator.reset();
        self.regularizer.reset();
        self.meta_learner.reset();
        self.curriculum_learner.reset();
        self.evaluator.reset();
        self.sequence_buffer.clear();
        self.metrics = TransformerOptimizerMetrics::new();
        
        Ok(())
    }
}

impl<T: Float + Default + Clone> SequenceBuffer<T> {
    /// Create new sequence buffer
    pub fn new(capacity: usize) -> Self {
        Self {
            gradient_sequences: VecDeque::new(),
            parameter_sequences: VecDeque::new(),
            loss_sequences: VecDeque::new(),
            lr_sequences: VecDeque::new(),
            capacity,
        }
    }
    
    /// Add gradient to buffer
    pub fn add_gradient(&mut self, gradient: Array1<T>) {
        self.gradient_sequences.push_back(gradient);
        if self.gradient_sequences.len() > self.capacity {
            self.gradient_sequences.pop_front();
        }
    }
    
    /// Add loss to buffer
    pub fn add_loss(&mut self, loss: T) {
        self.loss_sequences.push_back(loss);
        if self.loss_sequences.len() > self.capacity {
            self.loss_sequences.pop_front();
        }
    }
    
    /// Add learning rate to buffer
    pub fn add_learning_rate(&mut self, lr: T) {
        self.lr_sequences.push_back(lr);
        if self.lr_sequences.len() > self.capacity {
            self.lr_sequences.pop_front();
        }
    }
    
    /// Clear buffer
    pub fn clear(&mut self) {
        self.gradient_sequences.clear();
        self.parameter_sequences.clear();
        self.loss_sequences.clear();
        self.lr_sequences.clear();
    }
    
    /// Get recent gradient history
    pub fn get_recent_gradients(&self, count: usize) -> Vec<&Array1<T>> {
        self.gradient_sequences.iter()
            .rev()
            .take(count)
            .collect()
    }
}

impl TransformerOptimizerMetrics {
    /// Create new metrics tracker
    pub fn new() -> Self {
        Self {
            total_steps: 0,
            convergence_history: Vec::new(),
            attention_stats: HashMap::new(),
            strategy_stats: HashMap::new(),
            performance_comparisons: HashMap::new(),
        }
    }
    
    /// Update metrics after optimization step
    pub fn update_step(&mut self, loss: f64, step: usize) {
        self.total_steps = step;
        self.convergence_history.push(loss);
        
        // Keep only recent history
        if self.convergence_history.len() > 10000 {
            self.convergence_history.remove(0);
        }
    }
}