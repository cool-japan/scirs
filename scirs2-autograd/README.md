# scirs2-autograd

[![crates.io](https://img.shields.io/crates/v/scirs2-autograd.svg)](https://crates.io/crates/scirs2-autograd)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](../LICENSE)
[![Documentation](https://img.shields.io/docsrs/scirs2-autograd)](https://docs.rs/scirs2-autograd)
[![Version](https://img.shields.io/badge/version-0.6.3-green)]()
[![Status](https://img.shields.io/badge/status-stable-brightgreen)]()

Automatic differentiation engine for Rust, part of the [SciRS2](https://github.com/cool-japan/scirs) scientific computing ecosystem.

## Overview

`scirs2-autograd` provides PyTorch-style automatic differentiation with lazy tensor evaluation, enabling efficient gradient computation for scientific computing and machine learning. It supports reverse-mode AD (backpropagation), forward-mode AD (JVP), higher-order derivatives, gradient checkpointing, and a rich set of differentiable mathematical operations.

**Tests:** 1260/1260 passing (default features), 1345/1345 passing (`--all-features`) — as of 2026-07-22.

## Features

### Core Differentiation
- Reverse-mode AD (VJP / backpropagation) via tape-based gradient accumulation
- Forward-mode AD (JVP / Jacobian-vector products)
- Higher-order derivatives: Hessian, Hessian-vector products
- Second-order optimization support
- Dynamic computation graphs (eager-friendly construction)
- Lazy evaluation: build the graph, evaluate only when needed

### Gradient Utilities
- Finite difference numerical differentiation (forward, central, backward)
- Richardson extrapolation for improved accuracy
- Gradient checking / verification utilities
- Numerical differentiation as a fallback

### Memory and Performance
- Gradient checkpointing (recompute-based; reduces memory by 50-80%)
- Adaptive checkpointing based on tensor size threshold
- Checkpoint groups for multi-output operations
- Memory pooling and in-place operation support
- SIMD-accelerated element-wise operations
- Parallel processing with work-stealing thread pool

### Functional Transforms
- `grad` - gradient of a scalar output w.r.t. inputs
- `jacobian` - full Jacobian matrix computation
- `hessian` - second-order partial derivatives
- `vmap`-like vectorized map over batch dimensions
- Functional API for composable transforms

### Implicit Differentiation
- Implicit function theorem-based differentiation
- Fixed-point iteration gradients
- Bi-level optimization support

### Mixed Precision
- FP16 / FP32 mixed precision gradient computation
- Loss scaling for numeric stability

### Lazy Evaluation and JIT
- Computation graph construction without immediate execution
- Graph-level optimizations: constant folding, CSE, loop fusion
- JIT-like fusion of element-wise operations

### Optimizers (with State Management)
- SGD (with momentum and Nesterov)
- Adam, AdamW
- AdaGrad, RMSprop
- Learning rate schedulers: step, exponential, cosine annealing
- Gradient clipping (norm-based and value-based)
- Namespace-based variable management for multi-model setups

### Differentiable Mathematical Operations
- Arithmetic: add, sub, mul, div, pow with broadcasting
- Linear algebra: matmul, batch matmul, matrix inverse, determinant
- Decompositions with gradients: QR, SVD, Cholesky, LU
- Matrix functions: exp, log, sqrt, power, matrix exponential
- Matrix norms: Frobenius, spectral, nuclear
- Reductions: sum, mean, max, min, variance
- Activation functions: ReLU, Sigmoid, Tanh, Softmax, GELU, Swish, Mish
- Loss functions: MSE, cross-entropy, sparse categorical cross-entropy
- Convolution: Conv2D, transposed convolution, max/avg pooling
- Tensor manipulation: reshape, slice, concat, pad, advanced indexing

### Debugging and Visualization
- Computation graph visualization (DOT / Graphviz output)
- Gradient tape inspection
- NaN/Inf detection hooks
- Step-by-step execution tracing

### Distributed Gradient Computation
- Gradient aggregation across workers
- All-reduce primitives for distributed training

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
scirs2-autograd = "0.6.3"
```

For OxiBLAS-accelerated matrix operations (recommended):

```toml
[dependencies]
scirs2-autograd = { version = "0.6.3", features = ["blas"] }
```

### Basic Differentiation

```rust
use scirs2_autograd as ag;
use ag::tensor_ops as T;

ag::run(|ctx: &mut ag::Context<f64>| {
    let x = ctx.placeholder("x", &[]);
    let y = ctx.placeholder("y", &[]);

    // z = 2x^2 + 3y + 1
    let z = 2.0 * x * x + 3.0 * y + 1.0;

    // dz/dy = 3 (constant)
    let dz_dy = &T::grad(&[z], &[y])[0];
    println!("dz/dy = {:?}", dz_dy.eval(ctx)); // => 3.0

    // dz/dx at x=2 => 4*2 = 8
    let dz_dx = &T::grad(&[z], &[x])[0];
    let x_val = scirs2_core::ndarray::arr0(2.0_f64);
    let result = ctx.evaluator()
        .push(dz_dx)
        .feed(x, x_val.view().into_dyn())
        .run()[0].clone();
    println!("dz/dx at x=2: {:?}", result); // => 8.0

    // Second-order: d^2z/dx^2 = 4
    let d2z_dx2 = &T::grad(&[dz_dx], &[x])[0];
    println!("d2z/dx2 = {:?}", d2z_dx2.eval(ctx)); // => 4.0
});
```

### Neural Network Training

```rust
use scirs2_autograd as ag;
use ag::tensor_ops::*;
use ag::optimizers::adam::Adam;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut env = ag::VariableEnvironment::new();
    let mut rng = ag::ndarray_ext::ArrayRng::<f32>::default();

    // Initialize weights
    env.name("w1").set(rng.glorot_uniform(&[784, 256]));
    env.name("b1").set(ag::ndarray_ext::zeros(&[1, 256]));
    env.name("w2").set(rng.glorot_uniform(&[256, 10]));
    env.name("b2").set(ag::ndarray_ext::zeros(&[1, 10]));

    let var_ids = env.default_namespace().current_var_ids();
    let adam = Adam::default("adam", var_ids, &mut env);

    env.run(|ctx| {
        let x = ctx.placeholder("x", &[-1, 784]);
        let y = ctx.placeholder("y", &[-1]);

        let w1 = ctx.variable("w1");
        let b1 = ctx.variable("b1");
        let w2 = ctx.variable("w2");
        let b2 = ctx.variable("b2");

        let h = relu(matmul(x, w1) + b1);
        let logits = matmul(h, w2) + b2;
        let loss = reduce_mean(
            sparse_softmax_cross_entropy(logits, &y),
            &[0], false
        );

        let params = [w1, b1, w2, b2];
        let grads = &grad(&[loss], &params);
        // adam.update(&params, grads, ctx, &feeder);
    });

    Ok(())
}
```

### Gradient Checkpointing

```rust
use scirs2_autograd as ag;
use ag::tensor_ops as T;

ag::run(|ctx| {
    let input = T::ones(&[128, 128], ctx);
    let w = T::ones(&[128, 128], ctx);

    // Mark intermediate tensor for recomputation during backward
    let hidden = T::matmul(&input, &w);
    let hidden_ckpt = T::checkpoint(&hidden);

    // Adaptive: only checkpoint tensors larger than 1 MB
    let large = T::matmul(&input, &w);
    let large_ckpt = T::adaptive_checkpoint(&large, 1_000_000);
});
```

### JVP and VJP

```rust
use scirs2_autograd::jvp_vjp::{jvp, vjp};

// Jacobian-vector product (forward mode)
// jvp(f, inputs, tangents) -> (output, output_tangent)

// Vector-Jacobian product (reverse mode)
// vjp(f, inputs, cotangents) -> (output, input_cotangents)
```

## Feature Flags

| Flag | Description |
|------|-------------|
| `blas` | OxiBLAS-accelerated matrix operations (pure Rust BLAS) |
| `simd` | SIMD-accelerated element-wise operations |
| `gpu` | GPU acceleration support (delegates to `scirs2-core/gpu`) |
| `symbolic` | `scirs2-symbolic` EML tape backend (`symbolic_backend::EmlOp`, `tape::eml_tape`) — a `Tensor` constructed from a `LoweredOp` flows gradients through the EML symbolic kernel by provenance dispatch instead of the float tape |

## Related Crates

- [`scirs2-neural`](../scirs2-neural) - Neural network building blocks
- [`scirs2-optimize`](../scirs2-optimize) - Optimization algorithms
- [SciRS2 project](https://github.com/cool-japan/scirs)

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](../LICENSE) for details.
