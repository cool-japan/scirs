//! Core transformer architecture implementation

use ndarray::{Array1, Array2, Array3, Axis};
use num_traits::Float;
use crate::error::Result;
use super::config::TransformerArchConfig;
use super::layers::{EmbeddingLayer, LayerNormalization, DropoutLayer, OutputProjection, ResidualConnections};
use super::attention::MultiHeadAttention;
use super::feedforward::FeedForwardNetwork;

/// Core transformer architecture
pub struct TransformerArchitecture<T: Float> {
    /// Transformer layers
    layers: Vec<TransformerLayer<T>>,

    /// Input embedding
    input_embedding: EmbeddingLayer<T>,

    /// Output projection
    output_projection: OutputProjection<T>,

    /// Layer normalization
    layer_norm: LayerNormalization<T>,

    /// Dropout for regularization
    dropout: DropoutLayer,

    /// Architecture configuration
    config: TransformerArchConfig,
}

impl<T: Float + 'static> TransformerArchitecture<T> {
    /// Create new transformer architecture
    pub fn new(config: TransformerArchConfig) -> Result<Self> {
        let mut layers = Vec::new();

        for layer_idx in 0..config.num_layers {
            let layer = TransformerLayer::new(
                config.model_dimension,
                config.num_attention_heads,
                config.feedforward_dimension,
                config.dropout_rate,
                config.use_pre_norm,
                layer_idx,
            )?;
            layers.push(layer);
        }

        let input_embedding = EmbeddingLayer::new(
            config.model_dimension,
            config.model_dimension, // Vocab size equals model dimension for continuous inputs
        )?;

        let output_projection = OutputProjection::new(
            config.model_dimension,
            config.model_dimension,
        )?;

        let layer_norm = LayerNormalization::new(config.model_dimension)?;
        let dropout = DropoutLayer::new(config.dropout_rate);

        Ok(Self {
            layers,
            input_embedding,
            output_projection,
            layer_norm,
            dropout,
            config,
        })
    }

    /// Forward pass through the transformer
    pub fn forward(&mut self, input: &Array2<T>) -> Result<Array2<T>> {
        // Input embedding
        let mut hidden_states = self.input_embedding.forward(input)?;

        // Apply input dropout
        hidden_states = self.dropout.forward(&hidden_states);

        // Forward through transformer layers
        for layer in &mut self.layers {
            hidden_states = layer.forward(&hidden_states)?;
        }

        // Final layer normalization
        hidden_states = self.layer_norm.forward(&hidden_states)?;

        // Output projection
        let output = self.output_projection.forward(&hidden_states)?;

        Ok(output)
    }

    /// Get number of parameters
    pub fn parameter_count(&self) -> usize {
        let mut count = 0;

        count += self.input_embedding.parameter_count();
        count += self.output_projection.parameter_count();
        count += self.layer_norm.parameter_count();

        for layer in &self.layers {
            count += layer.parameter_count();
        }

        count
    }

    /// Reset all layers
    pub fn reset(&mut self) -> Result<()> {
        self.input_embedding.reset()?;
        self.output_projection.reset()?;
        self.layer_norm.reset()?;

        for layer in &mut self.layers {
            layer.reset()?;
        }

        Ok(())
    }

    /// Get configuration
    pub fn get_config(&self) -> &TransformerArchConfig {
        &self.config
    }
}

/// Individual transformer layer
pub struct TransformerLayer<T: Float> {
    /// Multi-head self-attention
    self_attention: MultiHeadAttention<T>,

    /// Feed-forward network
    feed_forward: FeedForwardNetwork<T>,

    /// Layer normalization (pre-norm style)
    pre_norm1: LayerNormalization<T>,
    pre_norm2: LayerNormalization<T>,

    /// Post-norm layers (if not using pre-norm)
    post_norm1: Option<LayerNormalization<T>>,
    post_norm2: Option<LayerNormalization<T>>,

    /// Residual connections
    residual_connections: ResidualConnections<T>,

    /// Layer-specific dropout
    dropout: DropoutLayer,

    /// Layer configuration
    use_pre_norm: bool,
    layer_index: usize,
}

impl<T: Float + 'static> TransformerLayer<T> {
    /// Create new transformer layer
    pub fn new(
        model_dimension: usize,
        num_attention_heads: usize,
        feedforward_dimension: usize,
        dropout_rate: f64,
        use_pre_norm: bool,
        layer_index: usize,
    ) -> Result<Self> {
        let self_attention = MultiHeadAttention::new(
            num_attention_heads,
            model_dimension,
            model_dimension / num_attention_heads,
        )?;

        let feed_forward = FeedForwardNetwork::new(
            model_dimension,
            feedforward_dimension,
            super::config::ActivationFunction::ReLU,
        )?;

        let pre_norm1 = LayerNormalization::new(model_dimension)?;
        let pre_norm2 = LayerNormalization::new(model_dimension)?;

        let (post_norm1, post_norm2) = if !use_pre_norm {
            (
                Some(LayerNormalization::new(model_dimension)?),
                Some(LayerNormalization::new(model_dimension)?),
            )
        } else {
            (None, None)
        };

        let residual_connections = ResidualConnections::new(model_dimension);
        let dropout = DropoutLayer::new(dropout_rate);

        Ok(Self {
            self_attention,
            feed_forward,
            pre_norm1,
            pre_norm2,
            post_norm1,
            post_norm2,
            residual_connections,
            dropout,
            use_pre_norm,
            layer_index,
        })
    }

    /// Forward pass through the layer
    pub fn forward(&mut self, input: &Array2<T>) -> Result<Array2<T>> {
        if self.use_pre_norm {
            self.forward_pre_norm(input)
        } else {
            self.forward_post_norm(input)
        }
    }

    /// Forward pass with pre-normalization
    fn forward_pre_norm(&mut self, input: &Array2<T>) -> Result<Array2<T>> {
        // Self-attention sub-layer with pre-norm
        let normed_input = self.pre_norm1.forward(input)?;
        let attention_output = self.self_attention.forward(&normed_input, &normed_input, &normed_input)?;
        let attention_output = self.dropout.forward(&attention_output);
        let attention_residual = self.residual_connections.add(input, &attention_output)?;

        // Feed-forward sub-layer with pre-norm
        let normed_attention = self.pre_norm2.forward(&attention_residual)?;
        let ff_output = self.feed_forward.forward(&normed_attention)?;
        let ff_output = self.dropout.forward(&ff_output);
        let ff_residual = self.residual_connections.add(&attention_residual, &ff_output)?;

        Ok(ff_residual)
    }

    /// Forward pass with post-normalization
    fn forward_post_norm(&mut self, input: &Array2<T>) -> Result<Array2<T>> {
        // Self-attention sub-layer with post-norm
        let attention_output = self.self_attention.forward(input, input, input)?;
        let attention_output = self.dropout.forward(&attention_output);
        let attention_residual = self.residual_connections.add(input, &attention_output)?;
        let attention_normed = if let Some(ref mut post_norm1) = self.post_norm1 {
            post_norm1.forward(&attention_residual)?
        } else {
            attention_residual
        };

        // Feed-forward sub-layer with post-norm
        let ff_output = self.feed_forward.forward(&attention_normed)?;
        let ff_output = self.dropout.forward(&ff_output);
        let ff_residual = self.residual_connections.add(&attention_normed, &ff_output)?;
        let ff_normed = if let Some(ref mut post_norm2) = self.post_norm2 {
            post_norm2.forward(&ff_residual)?
        } else {
            ff_residual
        };

        Ok(ff_normed)
    }

    /// Get number of parameters in this layer
    pub fn parameter_count(&self) -> usize {
        let mut count = 0;

        count += self.self_attention.parameter_count();
        count += self.feed_forward.parameter_count();
        count += self.pre_norm1.parameter_count();
        count += self.pre_norm2.parameter_count();

        if let Some(ref post_norm1) = self.post_norm1 {
            count += post_norm1.parameter_count();
        }

        if let Some(ref post_norm2) = self.post_norm2 {
            count += post_norm2.parameter_count();
        }

        count
    }

    /// Reset layer parameters
    pub fn reset(&mut self) -> Result<()> {
        self.self_attention.reset()?;
        self.feed_forward.reset()?;
        self.pre_norm1.reset()?;
        self.pre_norm2.reset()?;

        if let Some(ref mut post_norm1) = self.post_norm1 {
            post_norm1.reset()?;
        }

        if let Some(ref mut post_norm2) = self.post_norm2 {
            post_norm2.reset()?;
        }

        Ok(())
    }

    /// Get layer index
    pub fn get_layer_index(&self) -> usize {
        self.layer_index
    }

    /// Get attention weights (for visualization/analysis)
    pub fn get_attention_weights(&self) -> Option<Array3<T>> {
        self.self_attention.get_attention_weights()
    }
}

/// Architecture statistics
#[derive(Debug, Clone)]
pub struct ArchitectureStats {
    pub total_parameters: usize,
    pub total_layers: usize,
    pub model_dimension: usize,
    pub attention_heads: usize,
    pub memory_usage_mb: f64,
}

impl<T: Float + 'static> TransformerArchitecture<T> {
    /// Get architecture statistics
    pub fn get_stats(&self) -> ArchitectureStats {
        let total_parameters = self.parameter_count();
        let memory_usage_mb = (total_parameters * std::mem::size_of::<T>()) as f64 / (1024.0 * 1024.0);

        ArchitectureStats {
            total_parameters,
            total_layers: self.config.num_layers,
            model_dimension: self.config.model_dimension,
            attention_heads: self.config.num_attention_heads,
            memory_usage_mb,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learned_optimizers::transformer_based_optimizer::config::TransformerArchConfig;

    fn create_test_config() -> TransformerArchConfig {
        TransformerArchConfig {
            model_dimension: 128,
            num_layers: 2,
            num_attention_heads: 4,
            feedforward_dimension: 256,
            dropout_rate: 0.1,
            use_pre_norm: true,
            enable_residual_connections: true,
        }
    }

    #[test]
    fn test_transformer_architecture_creation() {
        let config = create_test_config();
        let architecture = TransformerArchitecture::<f32>::new(config);
        assert!(architecture.is_ok());

        let arch = architecture.unwrap();
        assert_eq!(arch.layers.len(), 2);
        assert_eq!(arch.config.model_dimension, 128);
    }

    #[test]
    fn test_transformer_layer_creation() {
        let layer = TransformerLayer::<f32>::new(128, 4, 256, 0.1, true, 0);
        assert!(layer.is_ok());

        let l = layer.unwrap();
        assert_eq!(l.layer_index, 0);
        assert!(l.use_pre_norm);
    }

    #[test]
    fn test_forward_pass() {
        let config = create_test_config();
        let mut architecture = TransformerArchitecture::<f32>::new(config).unwrap();

        let input = Array2::<f32>::zeros((4, 128)); // batch_size=4, seq_len=128
        let result = architecture.forward(&input);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.shape(), &[4, 128]);
    }

    #[test]
    fn test_parameter_count() {
        let config = create_test_config();
        let architecture = TransformerArchitecture::<f32>::new(config).unwrap();

        let param_count = architecture.parameter_count();
        assert!(param_count > 0);

        let stats = architecture.get_stats();
        assert_eq!(stats.total_parameters, param_count);
        assert_eq!(stats.total_layers, 2);
    }
}