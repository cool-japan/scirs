# SciRS2 Autograd

[![crates.io](https://img.shields.io/crates/v/scirs2-autograd.svg)](https://crates.io/crates/scirs2-autograd)
[![License](https://img.shields.io/crates/l/scirs2-autograd.svg)](../LICENSE)
[![Documentation](https://img.shields.io/docsrs/scirs2-autograd)](https://docs.rs/scirs2-autograd)

Automatic differentiation module for SciRS2, providing functionality comparable to PyTorch/TensorFlow's autograd systems and extending NumPy/SciPy's capabilities.

## Features

- Reverse-mode automatic differentiation
- Tensor-based computation with graph tracking
- Optimizers for machine learning tasks (SGD, Adam, etc.)
- Neural network operations:
  - Activation functions
  - Cross-entropy loss functions
  - Convolution operations
  - Pooling operations
  - Batch normalization
- Gradient computation and propagation
- Lazy tensor evaluation
- Higher-order derivatives
- GPU computation support (planned)
- BLAS acceleration for linear algebra operations

## Installation

Add the following to your `Cargo.toml`:

```toml
[dependencies]
scirs2-autograd = "0.1.0-alpha.1"
```

To enable optimizations and GPU support:

```toml
[dependencies]
scirs2-autograd = { version = "0.1.0-alpha.1", features = ["blas", "cuda"] }
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
- Integration with the broader SciRS2 ecosystem
- Multi-dimensional tensor operations
- Broadcasting operations like NumPy
- Support for both eager and graph execution models

## License

This project is licensed under the Apache License, Version 2.0 - see the [LICENSE](../LICENSE) file for details.