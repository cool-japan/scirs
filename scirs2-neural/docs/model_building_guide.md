# Model Building Guide

This comprehensive guide covers how to build various types of neural network models using scirs2-neural, from simple feedforward networks to complex architectures like Transformers and ResNets.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Basic Model Building](#basic-model-building)
3. [Computer Vision Models](#computer-vision-models)
4. [Natural Language Processing Models](#natural-language-processing-models)
5. [Advanced Architectures](#advanced-architectures)
6. [Model Composition Patterns](#model-composition-patterns)
7. [Performance Optimization](#performance-optimization)
8. [Debugging and Troubleshooting](#debugging-and-troubleshooting)

## Getting Started

### Essential Imports

```rust
use scirs2_neural::prelude::*;
use scirs2_neural::layers::{Sequential, Dense, Conv2D, LSTM, MultiHeadAttention};
use scirs2_neural::training::{Trainer, TrainingConfig};
use scirs2_neural::losses::{CrossEntropyLoss, MeanSquaredError};
use scirs2_neural::optimizers::{Adam, SGD};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use ndarray::Array;
```

### Model Building Philosophy

1. **Start Simple**: Begin with simple architectures and gradually add complexity
2. **Understand Your Data**: Match model architecture to data characteristics
3. **Consider Constraints**: Balance accuracy, speed, and memory requirements
4. **Iterate and Experiment**: Neural architecture is often an empirical process

## Basic Model Building

### Simple Feedforward Network

```rust
fn build_feedforward_classifier(
    input_size: usize,
    num_classes: usize,
    hidden_sizes: &[usize],
    rng: &mut SmallRng,
) -> Result<Sequential<f32>> {
    let mut model = Sequential::new();
    
    // Input layer
    let mut prev_size = input_size;
    
    // Hidden layers
    for &hidden_size in hidden_sizes {
        model.add(Dense::new(prev_size, hidden_size, Some("relu"), rng)?);
        model.add(Dropout::new(0.3));
        prev_size = hidden_size;
    }
    
    // Output layer
    model.add(Dense::new(prev_size, num_classes, Some("softmax"), rng)?);
    
    Ok(model)
}

// Usage example
let mut rng = SmallRng::seed_from_u64(42);
let model = build_feedforward_classifier(
    784,  // MNIST input size (28*28)
    10,   // 10 classes
    &[512, 256, 128],  // Hidden layer sizes
    &mut rng,
)?;
```

### Regression Model

```rust
fn build_regression_model(
    input_size: usize,
    output_size: usize,
    rng: &mut SmallRng,
) -> Result<Sequential<f32>> {
    let mut model = Sequential::new();
    
    // Feature extraction layers
    model.add(Dense::new(input_size, 256, Some("relu"), rng)?);
    model.add(Dropout::new(0.2));
    
    model.add(Dense::new(256, 128, Some("relu"), rng)?);
    model.add(Dropout::new(0.2));
    
    model.add(Dense::new(128, 64, Some("relu"), rng)?);
    model.add(Dropout::new(0.1));
    
    // Output layer (no activation for regression)
    model.add(Dense::new(64, output_size, None, rng)?);
    
    Ok(model)
}
```

### Binary Classification

```rust
fn build_binary_classifier(
    input_size: usize,
    rng: &mut SmallRng,
) -> Result<Sequential<f32>> {
    let mut model = Sequential::new();
    
    model.add(Dense::new(input_size, 128, Some("relu"), rng)?);
    model.add(Dropout::new(0.4));
    
    model.add(Dense::new(128, 64, Some("relu"), rng)?);
    model.add(Dropout::new(0.4));
    
    model.add(Dense::new(64, 32, Some("relu"), rng)?);
    model.add(Dropout::new(0.3));
    
    // Single output with sigmoid for binary classification
    model.add(Dense::new(32, 1, Some("sigmoid"), rng)?);
    
    Ok(model)
}
```

## Computer Vision Models

### Simple CNN for Image Classification

```rust
fn build_simple_cnn(
    input_channels: usize,
    num_classes: usize,
    rng: &mut SmallRng,
) -> Result<Sequential<f32>> {
    let mut model = Sequential::new();
    
    // First convolutional block
    model.add(Conv2D::new(input_channels, 32, (3, 3), (1, 1), PaddingMode::Same, rng)?);
    model.add(BatchNorm::new(32, 1e-5, 0.1)?);
    // Note: Add ReLU activation in practice
    model.add(MaxPool2D::new(2, 2)?);
    
    // Second convolutional block
    model.add(Conv2D::new(32, 64, (3, 3), (1, 1), PaddingMode::Same, rng)?);
    model.add(BatchNorm::new(64, 1e-5, 0.1)?);
    // Note: Add ReLU activation in practice
    model.add(MaxPool2D::new(2, 2)?);
    
    // Third convolutional block
    model.add(Conv2D::new(64, 128, (3, 3), (1, 1), PaddingMode::Same, rng)?);
    model.add(BatchNorm::new(128, 1e-5, 0.1)?);
    // Note: Add ReLU activation in practice
    model.add(AdaptiveMaxPool2D::new(1, 1)?);
    
    // Classifier head
    model.add(Dense::new(128, 256, Some("relu"), rng)?);
    model.add(Dropout::new(0.5));
    model.add(Dense::new(256, num_classes, Some("softmax"), rng)?);
    
    Ok(model)
}
```

### Advanced CNN with Residual Connections

```rust
fn build_residual_block(
    channels: usize,
    stride: usize,
    rng: &mut SmallRng,
) -> Result<Sequential<f32>> {
    let mut block = Sequential::new();
    
    // First convolution
    block.add(Conv2D::new(channels, channels, (3, 3), (stride, stride), PaddingMode::Same, rng)?);
    block.add(BatchNorm::new(channels, 1e-5, 0.1)?);
    // Note: Add ReLU activation in practice
    
    // Second convolution
    block.add(Conv2D::new(channels, channels, (3, 3), (1, 1), PaddingMode::Same, rng)?);
    block.add(BatchNorm::new(channels, 1e-5, 0.1)?);
    
    // Note: In a complete implementation, you'd add the residual connection here
    
    Ok(block)
}

fn build_resnet_like(
    input_channels: usize,
    num_classes: usize,
    rng: &mut SmallRng,
) -> Result<Sequential<f32>> {
    let mut model = Sequential::new();
    
    // Initial convolution
    model.add(Conv2D::new(input_channels, 64, (7, 7), (2, 2), PaddingMode::Same, rng)?);
    model.add(BatchNorm::new(64, 1e-5, 0.1)?);
    // Note: Add ReLU activation in practice
    model.add(MaxPool2D::new(3, 2)?);
    
    // Residual blocks (simplified - in practice you'd implement proper residual connections)
    let block1 = build_residual_block(64, 1, rng)?;
    let block2 = build_residual_block(128, 2, rng)?;
    let block3 = build_residual_block(256, 2, rng)?;
    let block4 = build_residual_block(512, 2, rng)?;
    
    // Global average pooling
    model.add(AdaptiveMaxPool2D::new(1, 1)?);
    
    // Final classifier
    model.add(Dense::new(512, num_classes, Some("softmax"), rng)?);
    
    Ok(model)
}
```

### Object Detection Feature Extractor

```rust
fn build_feature_pyramid_network(
    input_channels: usize,
    rng: &mut SmallRng,
) -> Result<Sequential<f32>> {
    let mut model = Sequential::new();
    
    // Bottom-up pathway
    model.add(Conv2D::new(input_channels, 64, (3, 3), (1, 1), PaddingMode::Same, rng)?);
    model.add(BatchNorm::new(64, 1e-5, 0.1)?);
    
    model.add(Conv2D::new(64, 128, (3, 3), (2, 2), PaddingMode::Same, rng)?);
    model.add(BatchNorm::new(128, 1e-5, 0.1)?);
    
    model.add(Conv2D::new(128, 256, (3, 3), (2, 2), PaddingMode::Same, rng)?);
    model.add(BatchNorm::new(256, 1e-5, 0.1)?);
    
    model.add(Conv2D::new(256, 512, (3, 3), (2, 2), PaddingMode::Same, rng)?);
    model.add(BatchNorm::new(512, 1e-5, 0.1)?);
    
    // Note: In practice, you'd implement lateral connections and top-down pathway
    
    Ok(model)
}
```

## Natural Language Processing Models

### Basic Text Classifier with Embeddings

```rust
fn build_text_classifier(
    vocab_size: usize,
    embed_dim: usize,
    num_classes: usize,
    max_seq_len: usize,
    rng: &mut SmallRng,
) -> Result<Sequential<f32>> {
    let mut model = Sequential::new();
    
    // Note: This is a simplified version since our Sequential model
    // expects specific layer types. In practice, you'd use proper embedding layers.
    
    // Simulate embedding with dense layer
    model.add(Dense::new(max_seq_len, embed_dim, Some("relu"), rng)?);
    model.add(Dropout::new(0.1));
    
    // Text processing layers
    model.add(Dense::new(embed_dim, 256, Some("relu"), rng)?);
    model.add(Dropout::new(0.2));
    
    model.add(Dense::new(256, 128, Some("relu"), rng)?);
    model.add(Dropout::new(0.2));
    
    // Classification head
    model.add(Dense::new(128, num_classes, Some("softmax"), rng)?);
    
    Ok(model)
}
```

### Sequence-to-Sequence Model

```rust
fn build_seq2seq_encoder(
    vocab_size: usize,
    embed_dim: usize,
    hidden_dim: usize,
    rng: &mut SmallRng,
) -> Result<Sequential<f32>> {
    let mut encoder = Sequential::new();
    
    // Simplified encoder using dense layers
    encoder.add(Dense::new(vocab_size, embed_dim, Some("relu"), rng)?);
    encoder.add(Dropout::new(0.1));
    
    // Simulate LSTM processing
    encoder.add(Dense::new(embed_dim, hidden_dim, Some("tanh"), rng)?);
    encoder.add(Dropout::new(0.2));
    
    encoder.add(Dense::new(hidden_dim, hidden_dim, Some("tanh"), rng)?);
    
    Ok(encoder)
}

fn build_seq2seq_decoder(
    vocab_size: usize,
    embed_dim: usize,
    hidden_dim: usize,
    rng: &mut SmallRng,
) -> Result<Sequential<f32>> {
    let mut decoder = Sequential::new();
    
    // Simplified decoder
    decoder.add(Dense::new(embed_dim + hidden_dim, hidden_dim, Some("tanh"), rng)?);
    decoder.add(Dropout::new(0.2));
    
    decoder.add(Dense::new(hidden_dim, hidden_dim, Some("tanh"), rng)?);
    decoder.add(Dropout::new(0.2));
    
    // Output projection
    decoder.add(Dense::new(hidden_dim, vocab_size, Some("softmax"), rng)?);
    
    Ok(decoder)
}
```

### Language Model

```rust
fn build_language_model(
    vocab_size: usize,
    embed_dim: usize,
    hidden_dim: usize,
    num_layers: usize,
    rng: &mut SmallRng,
) -> Result<Sequential<f32>> {
    let mut model = Sequential::new();
    
    // Input embedding simulation
    model.add(Dense::new(vocab_size, embed_dim, Some("relu"), rng)?);
    model.add(Dropout::new(0.1));
    
    // Stacked layers
    let mut current_dim = embed_dim;
    for _ in 0..num_layers {
        model.add(Dense::new(current_dim, hidden_dim, Some("tanh"), rng)?);
        model.add(LayerNorm::new(hidden_dim, 1e-5)?);
        model.add(Dropout::new(0.2));
        current_dim = hidden_dim;
    }
    
    // Output projection back to vocabulary
    model.add(Dense::new(hidden_dim, vocab_size, Some("softmax"), rng)?);
    
    Ok(model)
}
```

## Advanced Architectures

### Transformer Block

```rust
fn build_transformer_block(
    embed_dim: usize,
    num_heads: usize,
    ff_dim: usize,
    dropout: f32,
    rng: &mut SmallRng,
) -> Result<Sequential<f32>> {
    let mut block = Sequential::new();
    
    // Note: This is simplified. In practice, you'd implement proper attention
    // and residual connections
    
    // Simulate multi-head attention
    block.add(Dense::new(embed_dim, embed_dim, Some("relu"), rng)?);
    block.add(Dropout::new(dropout));
    block.add(LayerNorm::new(embed_dim, 1e-5)?);
    
    // Feed-forward network
    block.add(Dense::new(embed_dim, ff_dim, Some("relu"), rng)?);
    block.add(Dropout::new(dropout));
    block.add(Dense::new(ff_dim, embed_dim, None, rng)?);
    block.add(Dropout::new(dropout));
    block.add(LayerNorm::new(embed_dim, 1e-5)?);
    
    Ok(block)
}

fn build_transformer_encoder(
    vocab_size: usize,
    embed_dim: usize,
    num_heads: usize,
    num_layers: usize,
    ff_dim: usize,
    max_seq_len: usize,
    dropout: f32,
    rng: &mut SmallRng,
) -> Result<Sequential<f32>> {
    let mut model = Sequential::new();
    
    // Input embeddings
    model.add(Dense::new(vocab_size, embed_dim, Some("relu"), rng)?);
    model.add(Dropout::new(dropout));
    
    // Transformer blocks
    for _ in 0..num_layers {
        let block = build_transformer_block(embed_dim, num_heads, ff_dim, dropout, rng)?;
        // Note: In practice, you'd add each layer of the block separately
    }
    
    Ok(model)
}
```

### Vision Transformer (ViT)

```rust
fn build_patch_embedding(
    img_channels: usize,
    patch_size: usize,
    embed_dim: usize,
    rng: &mut SmallRng,
) -> Result<Sequential<f32>> {
    let mut patch_embed = Sequential::new();
    
    // Simulate patch embedding with convolution
    let kernel_size = patch_size;
    let stride = patch_size;
    
    patch_embed.add(Conv2D::new(
        img_channels,
        embed_dim,
        (kernel_size, kernel_size),
        (stride, stride),
        PaddingMode::Valid,
        rng,
    )?);
    
    Ok(patch_embed)
}

fn build_vision_transformer(
    img_size: usize,
    patch_size: usize,
    img_channels: usize,
    embed_dim: usize,
    num_heads: usize,
    num_layers: usize,
    num_classes: usize,
    dropout: f32,
    rng: &mut SmallRng,
) -> Result<Sequential<f32>> {
    let mut model = Sequential::new();
    
    // Patch embedding
    let patch_embed = build_patch_embedding(img_channels, patch_size, embed_dim, rng)?;
    
    // Calculate number of patches
    let num_patches = (img_size / patch_size).pow(2);
    
    // Flatten patches and add position embedding
    model.add(Dense::new(embed_dim * num_patches, embed_dim, Some("relu"), rng)?);
    model.add(Dropout::new(dropout));
    
    // Transformer encoder blocks
    for _ in 0..num_layers {
        let block = build_transformer_block(embed_dim, num_heads, embed_dim * 4, dropout, rng)?;
        // Note: Add transformer block layers here
    }
    
    // Classification head
    model.add(LayerNorm::new(embed_dim, 1e-5)?);
    model.add(Dense::new(embed_dim, num_classes, Some("softmax"), rng)?);
    
    Ok(model)
}
```

### Autoencoder

```rust
fn build_autoencoder(
    input_dim: usize,
    encoding_dims: &[usize],
    rng: &mut SmallRng,
) -> Result<(Sequential<f32>, Sequential<f32>)> {
    let mut encoder = Sequential::new();
    let mut decoder = Sequential::new();
    
    // Build encoder
    let mut current_dim = input_dim;
    for &dim in encoding_dims {
        encoder.add(Dense::new(current_dim, dim, Some("relu"), rng)?);
        encoder.add(Dropout::new(0.2));
        current_dim = dim;
    }
    
    // Build decoder (reverse of encoder)
    let mut decode_dims = encoding_dims.to_vec();
    decode_dims.reverse();
    decode_dims.push(input_dim);
    
    current_dim = encoding_dims[encoding_dims.len() - 1];
    for &dim in &decode_dims {
        let activation = if dim == input_dim { Some("sigmoid") } else { Some("relu") };
        decoder.add(Dense::new(current_dim, dim, activation, rng)?);
        if dim != input_dim {
            decoder.add(Dropout::new(0.2));
        }
        current_dim = dim;
    }
    
    Ok((encoder, decoder))
}
```

## Model Composition Patterns

### Ensemble Models

```rust
struct EnsembleModel<F: Float + Debug + ScalarOperand> {
    models: Vec<Sequential<F>>,
    weights: Vec<F>,
}

impl<F: Float + Debug + ScalarOperand> EnsembleModel<F> {
    fn new(models: Vec<Sequential<F>>, weights: Option<Vec<F>>) -> Self {
        let weights = weights.unwrap_or_else(|| {
            let weight = F::from(1.0 / models.len() as f64).unwrap();
            vec![weight; models.len()]
        });
        
        Self { models, weights }
    }
    
    fn predict(&self, input: &ArrayD<F>) -> Result<ArrayD<F>> {
        let mut ensemble_output = None;
        
        for (model, &weight) in self.models.iter().zip(&self.weights) {
            let output = model.forward(input)?;
            
            if let Some(ref mut ensemble) = ensemble_output {
                *ensemble = ensemble + &(output * weight);
            } else {
                ensemble_output = Some(output * weight);
            }
        }
        
        Ok(ensemble_output.unwrap())
    }
}
```

### Multi-Task Learning

```rust
fn build_multitask_model(
    shared_dim: usize,
    task_dims: &[usize],
    rng: &mut SmallRng,
) -> Result<(Sequential<f32>, Vec<Sequential<f32>>)> {
    // Shared feature extractor
    let mut shared_model = Sequential::new();
    shared_model.add(Dense::new(shared_dim, 512, Some("relu"), rng)?);
    shared_model.add(Dropout::new(0.3));
    shared_model.add(Dense::new(512, 256, Some("relu"), rng)?);
    shared_model.add(Dropout::new(0.3));
    
    // Task-specific heads
    let mut task_heads = Vec::new();
    for &task_dim in task_dims {
        let mut head = Sequential::new();
        head.add(Dense::new(256, 128, Some("relu"), rng)?);
        head.add(Dropout::new(0.2));
        head.add(Dense::new(128, task_dim, Some("softmax"), rng)?);
        task_heads.push(head);
    }
    
    Ok((shared_model, task_heads))
}
```

### Hierarchical Models

```rust
fn build_hierarchical_classifier(
    input_dim: usize,
    coarse_classes: usize,
    fine_classes_per_coarse: &[usize],
    rng: &mut SmallRng,
) -> Result<(Sequential<f32>, Vec<Sequential<f32>>)> {
    // Shared feature extractor
    let mut feature_extractor = Sequential::new();
    feature_extractor.add(Dense::new(input_dim, 512, Some("relu"), rng)?);
    feature_extractor.add(Dropout::new(0.3));
    feature_extractor.add(Dense::new(512, 256, Some("relu"), rng)?);
    feature_extractor.add(Dropout::new(0.3));
    
    // Coarse classifier
    let mut coarse_classifier = Sequential::new();
    coarse_classifier.add(Dense::new(256, 128, Some("relu"), rng)?);
    coarse_classifier.add(Dense::new(128, coarse_classes, Some("softmax"), rng)?);
    
    // Fine classifiers for each coarse class
    let mut fine_classifiers = Vec::new();
    for &num_fine in fine_classes_per_coarse {
        let mut fine_classifier = Sequential::new();
        fine_classifier.add(Dense::new(256, 128, Some("relu"), rng)?);
        fine_classifier.add(Dense::new(128, num_fine, Some("softmax"), rng)?);
        fine_classifiers.push(fine_classifier);
    }
    
    Ok((coarse_classifier, fine_classifiers))
}
```

## Performance Optimization

### Memory-Efficient Training

```rust
fn build_memory_efficient_model(
    input_dim: usize,
    output_dim: usize,
    rng: &mut SmallRng,
) -> Result<Sequential<f32>> {
    let mut model = Sequential::new();
    
    // Use smaller hidden dimensions to save memory
    model.add(Dense::new(input_dim, 256, Some("relu"), rng)?);
    model.add(Dropout::new(0.2));
    
    // Use multiple smaller layers instead of one large layer
    model.add(Dense::new(256, 128, Some("relu"), rng)?);
    model.add(Dropout::new(0.2));
    
    model.add(Dense::new(128, 64, Some("relu"), rng)?);
    model.add(Dropout::new(0.1));
    
    model.add(Dense::new(64, output_dim, Some("softmax"), rng)?);
    
    Ok(model)
}
```

### Inference-Optimized Model

```rust
fn build_inference_optimized_model(
    input_dim: usize,
    output_dim: usize,
    rng: &mut SmallRng,
) -> Result<Sequential<f32>> {
    let mut model = Sequential::new();
    
    // Fewer layers for faster inference
    model.add(Dense::new(input_dim, 128, Some("relu"), rng)?);
    // No dropout during inference
    
    model.add(Dense::new(128, 64, Some("relu"), rng)?);
    
    model.add(Dense::new(64, output_dim, Some("softmax"), rng)?);
    
    Ok(model)
}
```

### Model Compression

```rust
fn build_compressed_model(
    input_dim: usize,
    output_dim: usize,
    compression_ratio: f32,
    rng: &mut SmallRng,
) -> Result<Sequential<f32>> {
    let mut model = Sequential::new();
    
    // Reduce layer sizes based on compression ratio
    let hidden1 = ((512.0 * compression_ratio) as usize).max(32);
    let hidden2 = ((256.0 * compression_ratio) as usize).max(16);
    let hidden3 = ((128.0 * compression_ratio) as usize).max(8);
    
    model.add(Dense::new(input_dim, hidden1, Some("relu"), rng)?);
    model.add(Dense::new(hidden1, hidden2, Some("relu"), rng)?);
    model.add(Dense::new(hidden2, hidden3, Some("relu"), rng)?);
    model.add(Dense::new(hidden3, output_dim, Some("softmax"), rng)?);
    
    Ok(model)
}
```

## Debugging and Troubleshooting

### Common Issues and Solutions

#### 1. Gradient Vanishing/Exploding

```rust
fn build_gradient_stable_model(
    input_dim: usize,
    output_dim: usize,
    rng: &mut SmallRng,
) -> Result<Sequential<f32>> {
    let mut model = Sequential::new();
    
    // Use batch normalization to stabilize gradients
    model.add(Dense::new(input_dim, 256, None, rng)?);
    model.add(BatchNorm::new(256, 1e-5, 0.1)?);
    // Add activation after batch norm
    
    model.add(Dense::new(256, 128, None, rng)?);
    model.add(BatchNorm::new(128, 1e-5, 0.1)?);
    // Add activation after batch norm
    
    model.add(Dense::new(128, output_dim, Some("softmax"), rng)?);
    
    Ok(model)
}
```

#### 2. Overfitting

```rust
fn build_regularized_model(
    input_dim: usize,
    output_dim: usize,
    rng: &mut SmallRng,
) -> Result<Sequential<f32>> {
    let mut model = Sequential::new();
    
    // Increase dropout rates
    model.add(Dense::new(input_dim, 256, Some("relu"), rng)?);
    model.add(Dropout::new(0.5));  // Higher dropout
    
    model.add(Dense::new(256, 128, Some("relu"), rng)?);
    model.add(Dropout::new(0.4));
    
    // Add activity regularization
    model.add(L2ActivityRegularization::new(0.01));
    
    model.add(Dense::new(128, output_dim, Some("softmax"), rng)?);
    
    Ok(model)
}
```

#### 3. Slow Convergence

```rust
fn build_fast_converging_model(
    input_dim: usize,
    output_dim: usize,
    rng: &mut SmallRng,
) -> Result<Sequential<f32>> {
    let mut model = Sequential::new();
    
    // Use layer normalization for faster convergence
    model.add(Dense::new(input_dim, 512, Some("relu"), rng)?);
    model.add(LayerNorm::new(512, 1e-5)?);
    
    model.add(Dense::new(512, 256, Some("relu"), rng)?);
    model.add(LayerNorm::new(256, 1e-5)?);
    
    model.add(Dense::new(256, output_dim, Some("softmax"), rng)?);
    
    Ok(model)
}
```

### Model Validation Checklist

1. **Architecture Validation**
   - Verify input/output dimensions match your data
   - Check that activation functions are appropriate
   - Ensure model complexity matches dataset size

2. **Training Validation**
   - Monitor both training and validation loss
   - Check for proper convergence patterns
   - Validate gradient magnitudes are reasonable

3. **Performance Validation**
   - Test inference speed meets requirements
   - Monitor memory usage during training
   - Verify model generalizes to test data

### Example Training Setup

```rust
fn setup_training(
    model: Sequential<f32>,
    learning_rate: f32,
) -> Result<Trainer<f32>> {
    // Configure training
    let config = TrainingConfig {
        batch_size: 32,
        epochs: 100,
        learning_rate: learning_rate as f64,
        shuffle: true,
        verbose: 1,
        validation: Some(ValidationSettings {
            enabled: true,
            validation_split: 0.2,
            batch_size: 64,
            num_workers: 0,
        }),
        gradient_accumulation: None,
        mixed_precision: None,
    };
    
    // Create components
    let loss_fn = CrossEntropyLoss::new();
    let optimizer = Adam::new(learning_rate);
    
    // Create trainer
    let trainer = Trainer::new(model, optimizer, loss_fn, config);
    
    Ok(trainer)
}
```

This guide provides a comprehensive foundation for building various types of neural network models with scirs2-neural. Remember to start simple, iterate based on results, and always validate your models thoroughly before deployment.