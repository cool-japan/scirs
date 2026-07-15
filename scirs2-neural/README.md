# scirs2-neural

[![crates.io](https://img.shields.io/crates/v/scirs2-neural.svg)](https://crates.io/crates/scirs2-neural)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](../LICENSE)
[![Documentation](https://img.shields.io/docsrs/scirs2-neural)](https://docs.rs/scirs2-neural)
[![Status](https://img.shields.io/badge/status-stable-brightgreen.svg)](TODO.md)

A comprehensive, production-ready neural network library for Rust, part of the [SciRS2](https://github.com/cool-japan/scirs) scientific computing ecosystem.

## Overview

`scirs2-neural` provides PyTorch-style neural network building blocks with state-of-the-art architectures, training utilities, and advanced capabilities including Mixture of Experts, Spiking Neural Networks, Graph Neural Networks, Reinforcement Learning, and generative models. The library is designed for both research and production use, with a focus on correctness, performance, and idiomatic Rust.

**Status (2026-07-15, v0.6.1):** Stable. `cargo nextest run -p scirs2-neural` passes 1814/1814 tests with default features and 1863/1863 with `--all-features`; 0 `todo!()`/`unimplemented!()` stubs remain in `src/`. A small number of peripheral export/packaging formats intentionally return "not yet implemented" errors rather than panicking — see [TODO.md](TODO.md) for details.

## Features

### Attention Mechanisms
- Multi-head attention, self-attention, cross-attention
- Rotary Position Embeddings (RoPE)
- Grouped Query Attention (GQA) for memory-efficient inference
- Multi-query attention (MQA)
- Flash Attention v2 (tiled, memory-efficient)
- Linear attention and efficient attention variants (including Performer)
- Sparse attention and sliding-window attention patterns

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
- Real host-target code generation: compiles packaged models to native binaries / C-shared-libraries via `cargo build --release`

### Core Layers
- Dense / Linear with configurable activations
- Conv1D, Conv2D, Conv3D and transposed variants
- Depthwise separable convolutions
- MaxPool, AvgPool, GlobalPool, AdaptivePool
- LSTM, GRU, bidirectional RNNs
- BatchNorm, LayerNorm, GroupNorm, InstanceNorm, RMSNorm
- Dropout
- Embedding layers
- Mamba / Selective State Space (S4)
- MLP-Mixer blocks

### Activation Functions
- ReLU, LeakyReLU, ELU, SELU
- GELU, Swish/SiLU, Mish
- Sigmoid, Tanh, Softmax, LogSoftmax

### Loss Functions
- MSE, MAE, Huber / Smooth-L1
- Cross-entropy
- Focal loss for class-imbalanced datasets
- Dice loss, Tversky loss (for segmentation)
- Contrastive loss, Triplet loss

### Optimizers
- SGD (with momentum and Nesterov)
- Adam, AdamW, RAdam
- AdaGrad, RMSprop
- Learning rate schedulers: step decay, cosine annealing, warm restarts

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
scirs2-neural = "0.6.1"
```

With optional features:

```toml
[dependencies]
scirs2-neural = { version = "0.6.1", features = ["gpu"] }
```

### Building a Sequential MLP

```rust
use scirs2_neural::prelude::*;
use scirs2_core::random::rng;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = rng();

    let mut model = Sequential::<f32>::new();
    model.add_layer(Dense::new(784, 256, Some("relu"), &mut rng)?);
    model.add_layer(Dense::new(256, 128, Some("relu"), &mut rng)?);
    model.add_layer(Dense::new(128, 10, None, &mut rng)?);

    println!("Model: {} layers", model.num_layers());
    Ok(())
}
```

### Using Transformer Attention

```rust
use scirs2_neural::layers::{AttentionConfig, MultiHeadAttention};
use scirs2_core::random::rng;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = rng();
    // 8 heads, 64-dim model (head_dim = d_model / num_heads = 8)
    let config = AttentionConfig {
        num_heads: 8,
        head_dim: 8,
        dropout_prob: 0.0,
        causal: false,
        scale: None,
    };
    let attn = MultiHeadAttention::<f32>::new(64, config, &mut rng)?;
    Ok(())
}
```

### Knowledge Distillation

```rust
use scirs2_neural::training::knowledge_distillation::{distillation_loss, DistillationConfig};
use scirs2_core::ndarray::array;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = DistillationConfig {
        temperature: 4.0,
        alpha: 0.7, // weight for soft-target (teacher) loss
    };
    let student_logits = array![[2.0_f64, 1.0, 0.5], [0.1, 3.0, 1.2]];
    let teacher_logits = array![[1.8_f64, 1.1, 0.6], [0.0, 2.8, 1.0]];
    let true_labels = vec![0usize, 1];
    let loss = distillation_loss(
        student_logits.view(),
        teacher_logits.view(),
        &true_labels,
        &config,
    )?;
    println!("distillation loss: {loss}");
    Ok(())
}
```

### Graph Neural Network

```rust
use scirs2_neural::gnn::{GATLayer, GCNLayer};
use scirs2_neural::gnn::gcn::Activation;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let gcn = GCNLayer::new(32, 64, Activation::ReLU);
    let gat = GATLayer::new(64, 32, 4); // 4 attention heads
    Ok(())
}
```

## Examples

See `examples/image_classification.rs` for a complete image classification pipeline with convolutional networks.

## Feature Flags

| Flag | Description |
|------|-------------|
| `gpu` | GPU device detection and acceleration via `scirs2-core::gpu` |
| `metrics_integration` | `ScirsMetricsCallback` adapter for computing training/eval metrics via `scirs2-metrics` |
| `symbolic` | Symbolic weight initialization, formula extraction, and closed-form RoPE attention via `scirs2-symbolic` |
| `legacy_serialization` | Legacy model save/load format and runtime-binary packaging path |

Rayon-based parallelism and SIMD-accelerated ops are always enabled via the `scirs2-core` dependency (not separate toggles on this crate).

## Related Crates

- [`scirs2-autograd`](../scirs2-autograd) - Automatic differentiation engine
- [`scirs2-linalg`](../scirs2-linalg) - Linear algebra primitives
- [`scirs2-optimize`](../scirs2-optimize) - Optimization algorithms
- [SciRS2 project](https://github.com/cool-japan/scirs)

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](../LICENSE) for details.
