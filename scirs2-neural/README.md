# scirs2-neural

[![crates.io](https://img.shields.io/crates/v/scirs2-neural.svg)](https://crates.io/crates/scirs2-neural)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](../LICENSE)
[![Documentation](https://img.shields.io/docsrs/scirs2-neural)](https://docs.rs/scirs2-neural)

A comprehensive, production-ready neural network library for Rust, part of the [SciRS2](https://github.com/cool-japan/scirs) scientific computing ecosystem.

## Overview

`scirs2-neural` provides PyTorch-style neural network building blocks with state-of-the-art architectures, training utilities, and advanced capabilities including Mixture of Experts, Spiking Neural Networks, Graph Neural Networks, Reinforcement Learning, and generative models. The library is designed for both research and production use, with a focus on correctness, performance, and idiomatic Rust.

## Features

### Attention Mechanisms
- Multi-head attention, self-attention, cross-attention
- Rotary Position Embeddings (RoPE)
- Grouped Query Attention (GQA) for memory-efficient inference
- Linear attention and efficient attention variants
- Sparse attention patterns
- Multi-head latent attention

### Mixture of Experts (MoE)
- Top-k routing with load balancing
- Configurable expert capacity and auxiliary loss
- Integration with transformer blocks

### Capsule Networks
- Dynamic routing between capsules (Sabour et al.)
- Squash activation and routing agreement

### Spiking Neural Networks (SNN)
- Leaky Integrate-and-Fire (LIF) neurons
- Spike-Timing Dependent Plasticity (STDP)
- Temporal coding and rate coding

### Graph Neural Networks (GNN)
- Graph Convolutional Networks (GCN)
- Graph Attention Networks (GAT)
- GraphSAGE and GraphSAGE-Mean
- Graph Isomorphism Network (GIN)
- Message Passing Neural Networks (MPNN)
- Graph pooling: DiffPool, SAGPool, global pooling

### Vision Architectures
- SWIN Transformer (shifted window attention)
- Vision Transformer (ViT) with patch embeddings
- UNet for dense prediction / segmentation
- CLIP dual-encoder for vision-language alignment
- ConvNeXt (Tiny to XLarge variants)
- PatchEmbedding layers

### NLP / Sequence Architectures
- GPT-2 architecture (decoder-only transformer)
- T5 encoder-decoder architecture
- Full transformer (encoder + decoder)
- Positional encodings: sinusoidal, learned, RoPE, relative

### Generative Models
- Generative Adversarial Networks (GAN)
- Variational Autoencoders (VAE / autoencoder)
- Diffusion models (DDPM-style)
- Normalizing flow models
- Energy-based models (EBM)

### Training Infrastructure
- Knowledge distillation (response-based and feature-based)
- Continual learning (EWC, progressive networks)
- Meta-learning (MAML-style)
- Contrastive learning (SimCLR, MoCo-style)
- Multitask learning
- Self-supervised pretraining utilities
- Magnitude-based and structured pruning
- Post-training and quantization-aware training
- DPO (Direct Preference Optimization)
- PPO for reinforcement learning from human feedback
- Reward modeling and preference data handling

### Reinforcement Learning
- Proximal Policy Optimization (PPO)
- Actor-critic architectures
- Policy and value networks
- Replay buffers (uniform and prioritized)

### Serialization and Deployment
- Model graph serialization
- Weight format (portable, versioned)
- Gradient checkpointing for memory-efficient training
- Half-precision (FP16) support

### Core Layers
- Dense / Linear with configurable activations
- Conv1D, Conv2D, Conv3D and transposed variants
- Depthwise separable convolutions
- MaxPool, AvgPool, GlobalPool, AdaptivePool
- LSTM, GRU, bidirectional RNNs
- BatchNorm, LayerNorm, GroupNorm, InstanceNorm, RMSNorm
- Dropout, SpatialDropout, AlphaDropout
- Embedding layers
- Mamba / Selective State Space (S4)
- MLP-Mixer blocks

### Activation Functions
- ReLU, LeakyReLU, PReLU, ELU, SELU
- GELU, Swish/SiLU, Mish, Snake
- Sigmoid, Tanh, Softmax, LogSoftmax

### Loss Functions
- MSE, MAE, Huber / Smooth-L1
- Cross-entropy, Binary cross-entropy, Sparse categorical cross-entropy
- Focal loss for class-imbalanced datasets
- Contrastive loss, Triplet loss
- KL divergence, CTC loss

### Optimizers
- SGD (with momentum and Nesterov)
- Adam, AdamW, RAdam
- AdaGrad, RMSprop
- Learning rate schedulers: step decay, cosine annealing, warm restarts

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
scirs2-neural = "0.5.0"
```

With optional features:

```toml
[dependencies]
scirs2-neural = { version = "0.5.0", features = ["parallel"] }
```

### Building a Sequential MLP

```rust
use scirs2_neural::prelude::*;
use scirs2_core::random::rng;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = rng();

    let mut model = Sequential::<f32>::new();
    model.add(Dense::new(784, 256, Some("relu"), &mut rng)?);
    model.add(Dense::new(256, 128, Some("relu"), &mut rng)?);
    model.add(Dense::new(128, 10, None, &mut rng)?);

    println!("Model: {} layers", model.len());
    Ok(())
}
```

### Using Transformer Attention

```rust
use scirs2_neural::layers::attention::MultiHeadAttention;
use scirs2_core::random::rng;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = rng();
    // 8 heads, 64-dim model, 16-dim key/value
    let attn = MultiHeadAttention::<f32>::new(8, 64, 16, &mut rng)?;
    Ok(())
}
```

### Knowledge Distillation

```rust
use scirs2_neural::training::knowledge_distillation::{
    DistillationConfig, KnowledgeDistillationTrainer,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = DistillationConfig {
        temperature: 4.0,
        alpha: 0.7,   // weight for soft targets
        ..Default::default()
    };
    // Pair teacher and student models, then call trainer.train()
    Ok(())
}
```

### Graph Neural Network

```rust
use scirs2_neural::layers::gnn::{GCNLayer, GATLayer};
use scirs2_core::random::rng;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = rng();
    let gcn = GCNLayer::<f32>::new(32, 64, &mut rng)?;
    let gat = GATLayer::<f32>::new(64, 32, 4, &mut rng)?; // 4 attention heads
    Ok(())
}
```

## Examples

See `examples/image_classification.rs` for a complete image classification pipeline with convolutional networks.

## Feature Flags

| Flag | Description |
|------|-------------|
| `parallel` | Enable Rayon-based multi-threaded operations |
| `simd` | SIMD-accelerated activation functions and matrix ops |

## Related Crates

- [`scirs2-autograd`](../scirs2-autograd) - Automatic differentiation engine
- [`scirs2-linalg`](../scirs2-linalg) - Linear algebra primitives
- [`scirs2-optimize`](../scirs2-optimize) - Optimization algorithms
- [SciRS2 project](https://github.com/cool-japan/scirs)

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](../LICENSE) for details.
