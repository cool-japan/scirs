# SciRS2 Autograd

[![crates.io](https://img.shields.io/crates/v/scirs2-autograd.svg)](https://crates.io/crates/scirs2-autograd)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](../LICENSE)
[![Documentation](https://img.shields.io/docsrs/scirs2-autograd)](https://docs.rs/scirs2-autograd)

Automatic differentiation module for SciRS2, providing functionality comparable to PyTorch/TensorFlow's autograd systems and extending NumPy/SciPy's capabilities.

## Features

- Reverse-mode automatic differentiation
- Tensor-based computation with graph tracking
- Optimizers for machine learning tasks (SGD, Adam, etc.)
- Neural network operations with numerical stability enhancements:
  - Activation functions
  - Cross-entropy loss functions
  - Convolution operations
  - Pooling operations
  - Batch normalization
- Gradient computation and propagation with improved numerical stability
- Lazy tensor evaluation
- Higher-order derivatives
- Memory optimization with gradient checkpointing
- Enhanced linear algebra operations
- GPU computation support (planned)
- BLAS acceleration for linear algebra operations
- Numerically stable SVD gradient computation

## Installation

Add the following to your `Cargo.toml`:

```toml
[dependencies]
scirs2-autograd = "0.1.0-alpha.4"
```

To enable optimizations and GPU support:

```toml
[dependencies]
scirs2-autograd = { version = "0.1.0-alpha.4", features = ["blas", "cuda"] }
```

## Usage

```rust
use ndarray::array;
use scirs2_autograd::{run, tensor_ops as T};

// Basic gradient computation
run(|ctx| {
    let x = ctx.placeholder("x", &[]);
    let y = ctx.placeholder("y", &[]);
    let z = 2.0 * x * x + 3.0 * y + 1.0;

    // dz/dy
    let gy = &T::grad(&[z], &[y])[0];
    println!("{:?}", gy.eval(ctx));  // => 3.0

    // dz/dx with a value for x
    let gx = &T::grad(&[z], &[x])[0];
    let feed = ndarray::arr0(2.0);
    println!("{:?}", ctx.evaluator().push(gx).feed(x, feed.view()).run()[0]);  // => 8.0

    // Second derivative (ddz/dx)
    let ggx = &T::grad(&[gx], &[x])[0];
    println!("{:?}", ggx.eval(ctx));  // => 4.0
});
```

## Neural Network Example

```rust
use scirs2_autograd::{tensor_ops::*, VariableEnvironment};
use scirs2_autograd::optimizers::adam::Adam;

// Create a simple neural network for MNIST classification
let mut env = VariableEnvironment::new();
let rng = scirs2_autograd::ndarray_ext::ArrayRng::<f32>::default();

// Register variables
env.name("w1").set(rng.glorot_uniform(&[784, 128]));
env.name("b1").set(zeros(&[1, 128]));
env.name("w2").set(rng.glorot_uniform(&[128, 10]));
env.name("b2").set(zeros(&[1, 10]));

// Create optimizer
let adam = Adam::default(
    "adam", 
    env.default_namespace().current_var_ids(), 
    &mut env
);

// Training loop
for epoch in 0..10 {
    env.run(|ctx| {
        let x = ctx.placeholder("x", &[-1, 784]);
        let y = ctx.placeholder("y", &[-1]);
        
        // Forward pass
        let w1 = ctx.variable("w1");
        let b1 = ctx.variable("b1");
        let w2 = ctx.variable("w2");
        let b2 = ctx.variable("b2");
        
        let h = relu(matmul(x, w1) + b1);
        let logits = matmul(h, w2) + b2;
        
        // Loss
        let loss = reduce_mean(
            sparse_softmax_cross_entropy(logits, &y), 
            &[0], 
            false
        );
        
        // Compute gradients
        let grads = &grad(&[loss], &[w1, b1, w2, b2]);
        
        // Update parameters with batched data
        // adam.update(&[w1, b1, w2, b2], grads, ctx, feeder);
    });
}
```

## Advanced Features

- Higher-order derivatives for complex optimization problems
- Support for custom operations and gradients
- Memory-efficient computation graph management
- Numerically stable matrix decompositions including SVD
- Enhanced gradient precision for large matrices
- Integration with the broader SciRS2 ecosystem
- Multi-dimensional tensor operations
- Broadcasting operations like NumPy
- Support for both eager and graph execution models

## Gradient Checkpointing

Gradient checkpointing is a memory optimization technique that trades additional computation time for reduced memory usage during backpropagation. This is especially useful for training large models under memory constraints.

### How It Works

During standard backpropagation, all intermediate activations must be stored to compute gradients, which can lead to high memory usage in deep networks. Gradient checkpointing selectively discards intermediate activations during the forward pass and recomputes them during the backward pass as needed.

### Benefits

- Significantly reduced memory usage (typically 50-80% reduction)
- Enables training of deeper/larger models that would otherwise not fit in memory
- Flexible strategies to balance memory usage vs. computation time

### Usage Options

```rust
use scirs2_autograd::{run, tensor_ops as T};

run(|ctx| {
    // 1. Basic checkpointing of individual tensors
    let input = T::ones(&[128, 128], ctx);
    let w = T::ones(&[128, 128], ctx);
    let hidden = T::matmul(&input, &w);
    let hidden_checkpoint = T::checkpoint(&hidden);  // This tensor will be recomputed during backprop
    
    // 2. Adaptive checkpointing based on memory threshold (in bytes)
    let large_tensor = T::matmul(&input, &w);
    let adaptive_checkpoint = T::adaptive_checkpoint(&large_tensor, 1_000_000);  // 1MB threshold
    
    // 3. Checkpoint groups for multi-output operations
    let checkpoint_group = T::CheckpointGroup::new(ctx);
    let (output1, output2) = checkpoint_group.checkpoint_fn_flex2(&[&input, &w], |inputs| {
        let a = T::matmul(&inputs[0], &inputs[1]);
        let b = T::transpose(&a, &[1, 0]);
        (a, b)  // Both tensors will be checkpointed together
    });
});
```

### Profiling Checkpoint Performance

You can measure the memory savings and performance impact of your checkpointing strategy:

```rust
// Start tracking memory usage
T::CheckpointProfiler::start_tracking();

// Your model with checkpointing
// ... (model code with checkpoint operations)

// Evaluate performance
println!("Memory saved: {} KB", T::CheckpointProfiler::memory_saved() / 1024);
println!("Checkpoint operations: {}", T::CheckpointProfiler::checkpoint_count());

// Reset for next test
T::CheckpointProfiler::reset_statistics();
```

### Optimization Strategies

- **Basic Strategy**: Checkpoint every N layers (e.g., every other layer)
- **Adaptive Strategy**: Use automatic thresholds based on tensor size
- **Targeted Strategy**: Manually checkpoint only the largest tensors
- **Segment Strategy**: Checkpoint entire computation segments together

## License

This project is dual-licensed under:

- [MIT License](../LICENSE-MIT)
- [Apache License Version 2.0](../LICENSE-APACHE)

You can choose to use either license. See the [LICENSE](../LICENSE) file for details.
