# Layer Configuration Guide

This guide provides comprehensive documentation for configuring neural network layers in scirs2-neural. Each layer type has specific parameters that control its behavior, performance, and output characteristics.

## Table of Contents

1. [Dense Layers](#dense-layers)
2. [Convolutional Layers](#convolutional-layers)
3. [Pooling Layers](#pooling-layers)
4. [Normalization Layers](#normalization-layers)
5. [Regularization Layers](#regularization-layers)
6. [Recurrent Layers](#recurrent-layers)
7. [Attention Layers](#attention-layers)
8. [Embedding Layers](#embedding-layers)
9. [Common Configuration Patterns](#common-configuration-patterns)
10. [Performance Considerations](#performance-considerations)

## Dense Layers

Dense (fully connected) layers are the most fundamental building blocks of neural networks.

### Basic Configuration

```rust
use scirs2_neural::layers::{Dense, Layer};
use rand::rngs::SmallRng;
use rand::SeedableRng;

let mut rng = SmallRng::seed_from_u64(42);

// Basic dense layer: 128 inputs -> 64 outputs with ReLU activation
let dense = Dense::new(128, 64, Some("relu"), &mut rng)?;
```

### Parameters

| Parameter | Type | Description | Default | Common Values |
|-----------|------|-------------|---------|---------------|
| `input_dim` | `usize` | Number of input features | Required | 784 (MNIST), 512, 1024 |
| `output_dim` | `usize` | Number of output neurons | Required | 10 (classification), 1 (regression) |
| `activation` | `Option<&str>` | Activation function | `None` | "relu", "sigmoid", "tanh", "softmax" |
| `rng` | `&mut SmallRng` | Random number generator for weight initialization | Required | - |

### Activation Functions

- **"relu"**: Rectified Linear Unit, most common for hidden layers
- **"sigmoid"**: For binary classification output layers
- **"tanh"**: Alternative to ReLU, outputs in [-1, 1]
- **"softmax"**: For multi-class classification output layers
- **"linear"** or `None`: No activation, useful for regression outputs

### Configuration Examples

```rust
// Hidden layer with dropout-friendly ReLU
let hidden = Dense::new(512, 256, Some("relu"), &mut rng)?;

// Binary classification output
let binary_output = Dense::new(256, 1, Some("sigmoid"), &mut rng)?;

// Multi-class classification output
let multiclass_output = Dense::new(256, 10, Some("softmax"), &mut rng)?;

// Regression output (no activation)
let regression_output = Dense::new(256, 1, None, &mut rng)?;

// Feature extraction layer
let features = Dense::new(784, 512, Some("relu"), &mut rng)?;
```

### Best Practices

1. **Layer Size Progression**: Gradually reduce layer sizes (512 → 256 → 128 → output)
2. **Activation Choice**: Use ReLU for hidden layers, appropriate activation for output
3. **Weight Initialization**: The library uses He initialization for ReLU, Xavier for tanh/sigmoid
4. **Avoid Very Large Layers**: Prefer multiple smaller layers over one huge layer

## Convolutional Layers

Convolutional layers are essential for processing spatial data like images.

### 2D Convolution (Conv2D)

```rust
use scirs2_neural::layers::{Conv2D, PaddingMode};

let conv = Conv2D::new(
    3,                    // input_channels
    32,                   // output_channels  
    (3, 3),              // kernel_size
    (1, 1),              // stride
    PaddingMode::Same,   // padding
    &mut rng
)?;
```

### Parameters

| Parameter | Type | Description | Common Values |
|-----------|------|-------------|---------------|
| `input_channels` | `usize` | Number of input channels | 3 (RGB), 1 (grayscale), 64, 128 |
| `output_channels` | `usize` | Number of output feature maps | 32, 64, 128, 256, 512 |
| `kernel_size` | `(usize, usize)` | Size of convolution kernel | (3,3), (5,5), (7,7), (1,1) |
| `stride` | `(usize, usize)` | Step size for convolution | (1,1), (2,2) |
| `padding` | `PaddingMode` | How to handle borders | Same, Valid, Custom(n) |

### Padding Modes

```rust
// Preserve spatial dimensions
PaddingMode::Same      // Output size = Input size / Stride

// Reduce spatial dimensions  
PaddingMode::Valid     // Output size = (Input - Kernel + 1) / Stride

// Custom padding
PaddingMode::Custom(1) // Add 1 pixel padding on all sides
```

### Common CNN Architectures

```rust
// Feature extraction block
let conv1 = Conv2D::new(3, 64, (3, 3), (1, 1), PaddingMode::Same, &mut rng)?;
let conv2 = Conv2D::new(64, 64, (3, 3), (1, 1), PaddingMode::Same, &mut rng)?;
// Add pooling layer here

// Downsampling block  
let conv3 = Conv2D::new(64, 128, (3, 3), (2, 2), PaddingMode::Same, &mut rng)?;

// 1x1 convolution for channel adjustment
let conv_1x1 = Conv2D::new(128, 64, (1, 1), (1, 1), PaddingMode::Valid, &mut rng)?;

// Large receptive field
let conv_large = Conv2D::new(64, 128, (7, 7), (1, 1), PaddingMode::Same, &mut rng)?;
```

## Pooling Layers

Pooling layers reduce spatial dimensions and computational load.

### Max Pooling

```rust
use scirs2_neural::layers::{MaxPool2D, AdaptiveMaxPool2D};

// Standard max pooling
let maxpool = MaxPool2D::new(2, 2)?; // 2x2 pool, stride 2

// Adaptive pooling to fixed output size
let adaptive_pool = AdaptiveMaxPool2D::new(7, 7)?; // Always outputs 7x7
```

### Configuration Guide

| Pool Type | When to Use | Parameters | Output Size |
|-----------|-------------|------------|-------------|
| MaxPool2D | After conv layers | pool_size, stride | (H-pool+1)/stride |
| AdaptiveMaxPool2D | Before classifier | output_height, output_width | Fixed size |
| GlobalAvgPool2D | Replace flatten | None | (batch, channels, 1, 1) |

```rust
// Common pooling patterns
let pool_2x2 = MaxPool2D::new(2, 2)?;           // Halve spatial dimensions
let pool_3x3 = MaxPool2D::new(3, 3)?;           // Aggressive downsampling
let global_pool = GlobalAvgPool2D::new()?;      // Spatial dimensions → 1x1
let adaptive = AdaptiveMaxPool2D::new(1, 1)?;   // Force to 1x1 output
```

## Normalization Layers

Normalization improves training stability and convergence speed.

### Batch Normalization

```rust
use scirs2_neural::layers::BatchNorm;

// For dense layers (after linear transformation, before activation)
let bn_dense = BatchNorm::new(256, 1e-5, 0.1)?;

// For conv layers (normalize across spatial dimensions)
let bn_conv = BatchNorm::new(64, 1e-5, 0.1)?;
```

### Layer Normalization

```rust
use scirs2_neural::layers::LayerNorm;

// Normalize across feature dimension
let layer_norm = LayerNorm::new(512, 1e-5)?;
```

### Parameters

| Parameter | Description | Typical Values | Notes |
|-----------|-------------|----------------|-------|
| `num_features` | Size of normalized dimension | Layer output size | |
| `eps` | Numerical stability constant | 1e-5, 1e-6 | Prevent division by zero |
| `momentum` | Moving average momentum | 0.1, 0.01 | Only for BatchNorm |

### Usage Patterns

```rust
// Standard CNN block with batch norm
model.add(Conv2D::new(32, 64, (3,3), (1,1), PaddingMode::Same, &mut rng)?);
model.add(BatchNorm::new(64, 1e-5, 0.1)?);
// Add activation function here

// Dense layer with layer norm  
model.add(Dense::new(512, 256, None, &mut rng)?);
model.add(LayerNorm::new(256, 1e-5)?);
// Add activation function here
```

## Regularization Layers

Regularization prevents overfitting and improves generalization.

### Dropout

```rust
use scirs2_neural::layers::Dropout;

// Standard dropout for dense layers
let dropout = Dropout::new(0.5);  // Drop 50% of neurons

// Light dropout for early layers
let light_dropout = Dropout::new(0.1);  // Drop 10% of neurons

// Heavy dropout before output
let heavy_dropout = Dropout::new(0.7);  // Drop 70% of neurons
```

### Activity Regularization

```rust
use scirs2_neural::layers::{L1ActivityRegularization, L2ActivityRegularization};

// L1 regularization (promotes sparsity)
let l1_reg = L1ActivityRegularization::new(0.01);

// L2 regularization (prevents large activations) 
let l2_reg = L2ActivityRegularization::new(0.001);
```

### Dropout Guidelines

| Layer Type | Dropout Rate | Reasoning |
|------------|-------------|-----------|
| Input | 0.1-0.2 | Light regularization |
| Hidden Dense | 0.3-0.5 | Standard regularization |
| Before Output | 0.5-0.7 | Heavy regularization |
| Conv Layers | 0.1-0.25 | Spatial correlation |

```rust
// Typical dropout pattern
model.add(Dense::new(1024, 512, Some("relu"), &mut rng)?);
model.add(Dropout::new(0.3));

model.add(Dense::new(512, 256, Some("relu"), &mut rng)?);
model.add(Dropout::new(0.4));

model.add(Dense::new(256, 128, Some("relu"), &mut rng)?);
model.add(Dropout::new(0.5));

model.add(Dense::new(128, 10, Some("softmax"), &mut rng)?);
// No dropout on output layer
```

## Recurrent Layers

RNNs process sequential data with memory.

### LSTM Configuration

```rust
use scirs2_neural::layers::{LSTM, LSTMConfig};

let lstm_config = LSTMConfig {
    input_size: 128,      // Input feature size
    hidden_size: 256,     // Hidden state size  
    num_layers: 2,        // Stacked LSTM layers
    dropout: 0.2,         // Dropout between layers
    bidirectional: false, // Forward only
};

let lstm = LSTM::new(lstm_config, &mut rng)?;
```

### GRU Configuration

```rust
use scirs2_neural::layers::{GRU, GRUConfig};

let gru_config = GRUConfig {
    input_size: 128,
    hidden_size: 256,
    num_layers: 1,
    dropout: 0.1,
    bidirectional: true,  // Process in both directions
};

let gru = GRU::new(gru_config, &mut rng)?;
```

### Bidirectional Wrapper

```rust
use scirs2_neural::layers::Bidirectional;

// Wrap any recurrent layer for bidirectional processing
let bi_lstm = Bidirectional::new(lstm);
// Output size will be 2 * hidden_size
```

### RNN Configuration Guidelines

| Parameter | Small Model | Medium Model | Large Model |
|-----------|-------------|--------------|-------------|
| hidden_size | 64-128 | 256-512 | 512-1024 |
| num_layers | 1-2 | 2-3 | 3-4 |
| dropout | 0.1-0.2 | 0.2-0.3 | 0.3-0.5 |

## Attention Layers

Attention mechanisms help models focus on relevant parts of the input.

### Multi-Head Attention

```rust
use scirs2_neural::layers::{MultiHeadAttention, AttentionConfig};

let attention_config = AttentionConfig {
    embed_dim: 512,       // Embedding dimension
    num_heads: 8,         // Number of attention heads
    dropout: 0.1,         // Attention dropout
    bias: true,           // Use bias in projections
};

let attention = MultiHeadAttention::new(attention_config, &mut rng)?;
```

### Self-Attention

```rust
use scirs2_neural::layers::SelfAttention;

let self_attention = SelfAttention::new(
    512,    // embed_dim
    8,      // num_heads  
    0.1,    // dropout
    &mut rng
)?;
```

### Attention Guidelines

| Use Case | embed_dim | num_heads | dropout |
|----------|-----------|-----------|---------|
| Small Text | 128-256 | 4-8 | 0.1 |
| Medium Text | 512 | 8-12 | 0.1-0.2 |
| Large Text | 768-1024 | 12-16 | 0.1-0.3 |

## Embedding Layers

Embeddings convert discrete inputs to dense vectors.

### Word Embeddings

```rust
use scirs2_neural::layers::{Embedding, EmbeddingConfig};

let embedding_config = EmbeddingConfig {
    vocab_size: 10000,    // Size of vocabulary
    embed_dim: 300,       // Embedding dimension
    padding_idx: Some(0), // Index for padding token
    max_norm: None,       // Gradient clipping
    scale_grad: false,    // Scale gradients by frequency
};

let embedding = Embedding::new(embedding_config, &mut rng)?;
```

### Positional Embeddings

```rust
use scirs2_neural::layers::PositionalEmbedding;

let pos_embedding = PositionalEmbedding::new(
    512,    // max_sequence_length
    256,    // embed_dim
    &mut rng
)?;
```

### Patch Embeddings (for Vision Transformers)

```rust
use scirs2_neural::layers::PatchEmbedding;

let patch_embedding = PatchEmbedding::new(
    3,      // input_channels (RGB)
    (16, 16), // patch_size
    768,    // embed_dim
    &mut rng
)?;
```

### Embedding Guidelines

| Domain | vocab_size | embed_dim | Notes |
|--------|------------|-----------|-------|
| Small Text | 5K-10K | 128-256 | Simple tasks |
| Medium Text | 20K-50K | 256-512 | General NLP |
| Large Text | 50K+ | 512-768 | Complex NLP |
| Vision | N/A | 768-1024 | Patch embeddings |

## Common Configuration Patterns

### Image Classification CNN

```rust
fn build_image_classifier(num_classes: usize, rng: &mut SmallRng) -> Result<Sequential<f32>> {
    let mut model = Sequential::new();
    
    // Feature extraction
    model.add(Conv2D::new(3, 32, (3,3), (1,1), PaddingMode::Same, rng)?);
    model.add(BatchNorm::new(32, 1e-5, 0.1)?);
    // Add ReLU activation
    model.add(MaxPool2D::new(2, 2)?);
    
    model.add(Conv2D::new(32, 64, (3,3), (1,1), PaddingMode::Same, rng)?);
    model.add(BatchNorm::new(64, 1e-5, 0.1)?);
    // Add ReLU activation  
    model.add(MaxPool2D::new(2, 2)?);
    
    model.add(Conv2D::new(64, 128, (3,3), (1,1), PaddingMode::Same, rng)?);
    model.add(BatchNorm::new(128, 1e-5, 0.1)?);
    // Add ReLU activation
    model.add(AdaptiveMaxPool2D::new(1, 1)?);
    
    // Classifier
    model.add(Dense::new(128, 256, Some("relu"), rng)?);
    model.add(Dropout::new(0.5));
    model.add(Dense::new(256, num_classes, Some("softmax"), rng)?);
    
    Ok(model)
}
```

### Text Classification with LSTM

```rust
fn build_text_classifier(vocab_size: usize, num_classes: usize, rng: &mut SmallRng) -> Result<Sequential<f32>> {
    let mut model = Sequential::new();
    
    // Embeddings
    let embed_config = EmbeddingConfig {
        vocab_size,
        embed_dim: 300,
        padding_idx: Some(0),
        max_norm: None,
        scale_grad: false,
    };
    model.add(Embedding::new(embed_config, rng)?);
    
    // Sequence processing
    let lstm_config = LSTMConfig {
        input_size: 300,
        hidden_size: 128,
        num_layers: 2,
        dropout: 0.2,
        bidirectional: true,
    };
    model.add(LSTM::new(lstm_config, rng)?);
    
    // Classification head
    model.add(Dense::new(256, 128, Some("relu"), rng)?); // 2 * hidden_size for bidirectional
    model.add(Dropout::new(0.3));
    model.add(Dense::new(128, num_classes, Some("softmax"), rng)?);
    
    Ok(model)
}
```

### Transformer Block

```rust
fn build_transformer_block(embed_dim: usize, rng: &mut SmallRng) -> Result<Sequential<f32>> {
    let mut block = Sequential::new();
    
    // Self-attention
    let attention_config = AttentionConfig {
        embed_dim,
        num_heads: 8,
        dropout: 0.1,
        bias: true,
    };
    block.add(MultiHeadAttention::new(attention_config, rng)?);
    block.add(LayerNorm::new(embed_dim, 1e-5)?);
    
    // Feed-forward
    block.add(Dense::new(embed_dim, embed_dim * 4, Some("relu"), rng)?);
    block.add(Dropout::new(0.1));
    block.add(Dense::new(embed_dim * 4, embed_dim, None, rng)?);
    block.add(LayerNorm::new(embed_dim, 1e-5)?);
    
    Ok(block)
}
```

## Performance Considerations

### Memory Optimization

1. **Batch Size**: Larger batches use more memory but may train faster
2. **Model Width**: Wider layers use more memory and computation
3. **Sequence Length**: Longer sequences in RNNs increase memory usage
4. **Attention**: Quadratic memory growth with sequence length

### Computation Optimization

1. **Kernel Sizes**: Smaller kernels (3x3) are more efficient than large ones
2. **Channels**: Powers of 2 (32, 64, 128) may be more efficient
3. **Depth vs Width**: Deeper networks often perform better than wider ones
4. **Skip Connections**: Help with gradient flow in deep networks

### Training Stability

1. **Normalization**: Use batch norm or layer norm for stable training
2. **Gradient Clipping**: Prevent exploding gradients in RNNs
3. **Learning Rate**: Start with 1e-3, adjust based on convergence
4. **Warmup**: Gradually increase learning rate for transformers

### Layer-Specific Tips

| Layer Type | Memory | Computation | Stability |
|------------|--------|-------------|-----------|
| Dense | O(input×output) | O(batch×input×output) | Very stable |
| Conv2D | O(kernel×channels) | O(batch×H×W×kernel×channels) | Stable |
| LSTM | O(4×hidden²) | O(seq×batch×hidden²) | Can be unstable |
| Attention | O(seq²×embed) | O(batch×seq²×embed) | Generally stable |

This guide provides a comprehensive overview of layer configuration in scirs2-neural. For specific use cases, refer to the examples in the repository and experiment with different configurations to find what works best for your data and task.