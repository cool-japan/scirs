# SciRS2 Neural

[![crates.io](https://img.shields.io/crates/v/scirs2-neural.svg)](https://crates.io/crates/scirs2-neural)
[![License](https://img.shields.io/crates/l/scirs2-neural.svg)](../LICENSE)
[![Documentation](https://img.shields.io/docsrs/scirs2-neural)](https://docs.rs/scirs2-neural)

Neural network module for the SciRS2 scientific computing library. This module provides tools for building, training, and evaluating neural networks.

## Features

- **Core Neural Network Components**: Layers, activations, loss functions
- **Sequential Model API**: Simple API for creating feed-forward neural networks
- **Automatic Differentiation**: Efficient gradient computation with autograd
- **Optimizers**: Various optimization algorithms (SGD, Adam, etc.)
- **Utilities**: Initializers, metrics, and dataset handling

## Installation

Add the following to your `Cargo.toml`:

```toml
[dependencies]
scirs2-neural = "0.1.0-alpha.1"
```

To enable optimizations and optional features:

```toml
[dependencies]
scirs2-neural = { version = "0.1.0-alpha.1", features = ["cuda", "blas"] }
```

## Usage

Basic usage examples:

```rust
use scirs2_neural::{models, layers, activations, losses, optimizers};
use scirs2_core::error::CoreResult;
use ndarray::array;

// Create a simple neural network
fn create_neural_network() -> CoreResult<()> {
    // Create a sequential model
    let mut model = models::sequential::Sequential::new();
    
    // Add layers
    model.add_layer(layers::dense::Dense::new(2, 32, None, None)?);
    model.add_layer(activations::relu::ReLU::new());
    model.add_layer(layers::dense::Dense::new(32, 16, None, None)?);
    model.add_layer(activations::relu::ReLU::new());
    model.add_layer(layers::dense::Dense::new(16, 1, None, None)?);
    model.add_layer(activations::sigmoid::Sigmoid::new());
    
    // Set loss function and optimizer
    let loss = losses::mse::MeanSquaredError::new();
    let optimizer = optimizers::adam::Adam::new(0.001, 0.9, 0.999, 1e-8);
    
    // Compile the model
    model.compile(loss, optimizer);
    
    // Sample training data (XOR problem)
    let x_train = array![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
    let y_train = array![[0.0], [1.0], [1.0], [0.0]];
    
    // Train the model
    model.fit(&x_train, &y_train, 1000, 4, None, None)?;
    
    // Make predictions
    let predictions = model.predict(&x_train)?;
    println!("Predictions: {:?}", predictions);
    
    Ok(())
}

// Using autograd for manual gradient computation
fn autograd_example() -> CoreResult<()> {
    use scirs2_neural::autograd::{Variable, Graph};
    
    // Create computation graph
    let mut graph = Graph::new();
    
    // Create input variables
    let x = graph.variable(2.0);
    let y = graph.variable(3.0);
    
    // Build computation
    let z = graph.add(&x, &y);  // z = x + y
    let w = graph.multiply(&z, &x);  // w = z * x = (x + y) * x
    
    // Forward pass
    graph.forward()?;
    println!("Result: {}", w.value()?);  // Should be (2 + 3) * 2 = 10
    
    // Backward pass to compute gradients
    graph.backward(&w)?;
    
    // Get gradients
    println!("dw/dx: {}", x.gradient()?);  // Should be d((x+y)*x)/dx = (x+y) + x*1 = 2+3 + 2*1 = 7
    println!("dw/dy: {}", y.gradient()?);  // Should be d((x+y)*x)/dy = x*1 = 2
    
    Ok(())
}
```

## Components

### Layers

Neural network layer implementations:

```rust
use scirs2_neural::layers::{
    Layer,                  // Layer trait
    dense::Dense,           // Fully connected layer
    
    // Other layer types (examples)
    // dropout::Dropout,       // Dropout layer
    // conv2d::Conv2D,         // 2D convolutional layer
    // maxpool2d::MaxPool2D,   // 2D max pooling layer
    // batchnorm::BatchNorm,   // Batch normalization layer
};
```

### Activations

Activation functions:

```rust
use scirs2_neural::activations::{
    Activation,             // Activation trait
    relu::ReLU,             // Rectified Linear Unit
    sigmoid::Sigmoid,       // Sigmoid activation
    tanh::Tanh,             // Hyperbolic tangent
    softmax::Softmax,       // Softmax activation
};
```

### Loss Functions

Loss function implementations:

```rust
use scirs2_neural::losses::{
    Loss,                   // Loss trait
    mse::MeanSquaredError,  // Mean Squared Error
    crossentropy::CrossEntropy, // Cross Entropy Loss
};
```

### Models

Neural network model implementations:

```rust
use scirs2_neural::models::{
    sequential::Sequential,  // Sequential model
    trainer::Trainer,        // Training utilities
};
```

### Optimizers

Optimization algorithms:

```rust
use scirs2_neural::optimizers_temp::{
    // Optimizer,            // Optimizer trait
    sgd::SGD,               // Stochastic Gradient Descent
    adagrad::AdaGrad,       // Adaptive Gradient Algorithm
    rmsprop::RMSprop,       // Root Mean Square Propagation
    adam::Adam,             // Adaptive Moment Estimation
};
```

### Autograd

Automatic differentiation functionality:

```rust
use scirs2_neural::autograd::{
    Variable,               // Variable holding value and gradient
    Graph,                  // Computation graph
    Tape,                   // Gradient tape
    Function,               // Function trait
    ops,                    // Basic operations
};
```

### Utilities

Helper utilities:

```rust
use scirs2_neural::utils::{
    initializers,           // Weight initialization functions
    metrics,                // Evaluation metrics
    datasets,               // Dataset utilities
};
```

## Integration with Other SciRS2 Modules

This module integrates with other SciRS2 modules:

- **scirs2-linalg**: For efficient matrix operations
- **scirs2-optim**: For advanced optimization algorithms
- **scirs2-autograd**: For automatic differentiation (if used separately)

Example of using linear algebra functions:

```rust
use scirs2_neural::linalg::batch_operations;
use ndarray::Array2;

// Batch matrix multiplication
let a = Array2::<f64>::zeros((32, 10, 20));
let b = Array2::<f64>::zeros((32, 20, 15));
let result = batch_operations::batch_matmul(&a, &b);
```

## Contributing

See the [CONTRIBUTING.md](../CONTRIBUTING.md) file for contribution guidelines.

## License

This project is licensed under the Apache License, Version 2.0 - see the [LICENSE](../LICENSE) file for details.