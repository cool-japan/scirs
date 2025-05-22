# SciRS2 Neural

[![crates.io](https://img.shields.io/crates/v/scirs2-neural.svg)](https://crates.io/crates/scirs2-neural)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](../LICENSE)
[![Documentation](https://img.shields.io/docsrs/scirs2-neural)](https://docs.rs/scirs2-neural)

Neural network module for the SciRS2 scientific computing library. This module provides tools for building, training, and evaluating neural networks.

## Features

- **Core Neural Network Components**: Layers, activations, loss functions
- **Advanced Layer Types**: Convolutional, pooling, recurrent, normalization, and attention layers
- **Transformer Architecture**: Full transformer implementation with multi-head attention
- **Sequential Model API**: Simple API for creating feed-forward neural networks
- **Advanced Activations**: GELU, Swish/SiLU, Mish, and more
- **Automatic Differentiation**: Efficient gradient computation with autograd
- **Optimizers**: Various optimization algorithms (SGD, Adam, AdamW, RAdam, etc.)
- **Model Serialization**: Save and load trained models
- **Utilities**: Initializers, metrics, and dataset handling

## Installation

Add the following to your `Cargo.toml`:

```toml
[dependencies]
scirs2-neural = "0.1.0-alpha.4"
```

To enable optimizations and optional features:

```toml
[dependencies]
scirs2-neural = { version = "0.1.0-alpha.4", features = ["cuda", "blas"] }

# For integration with scirs2-metrics
scirs2-neural = { version = "0.1.0-alpha.4", features = ["metrics_integration"] }
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
    dropout::Dropout,       // Dropout layer
    conv::Conv2D,           // 2D convolutional layer
    conv::Conv2DTranspose,  // 2D transposed convolutional layer
    pooling::MaxPool2D,     // 2D max pooling layer
    pooling::AvgPool2D,     // 2D average pooling layer
    pooling::GlobalPooling, // Global pooling layer
    norm::BatchNorm,        // Batch normalization layer
    norm::LayerNorm,        // Layer normalization layer
    recurrent::LSTM,        // Long Short-Term Memory layer
    recurrent::GRU,         // Gated Recurrent Unit layer
    recurrent::RNN,         // Simple RNN layer
    attention::MultiHeadAttention, // Multi-head attention mechanism
    attention::SelfAttention,      // Self-attention mechanism
    transformer::TransformerEncoder, // Transformer encoder block
    transformer::TransformerDecoder, // Transformer decoder block
    transformer::Transformer,        // Full transformer architecture
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
    gelu::GELU,             // Gaussian Error Linear Unit
    swish::Swish,           // Swish/SiLU activation
    mish::Mish,             // Mish activation
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
use scirs2_neural::optimizers::{
    Optimizer,              // Optimizer trait
    sgd::SGD,               // Stochastic Gradient Descent
    adagrad::AdaGrad,       // Adaptive Gradient Algorithm
    rmsprop::RMSprop,       // Root Mean Square Propagation
    adam::Adam,             // Adaptive Moment Estimation
    adamw::AdamW,           // Adam with decoupled weight decay
    radam::RAdam,           // Rectified Adam
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

// Model serialization
use scirs2_neural::serialization::{
    SaveLoad,               // Save/load trait for models
    ModelConfig,            // Configuration for model serialization
    load_model,             // Load model from file
};
```

## Integration with Other SciRS2 Modules

This module integrates with other SciRS2 modules:

- **scirs2-linalg**: For efficient matrix operations
- **scirs2-optim**: For advanced optimization algorithms
- **scirs2-autograd**: For automatic differentiation (if used separately)
- **scirs2-metrics**: For advanced evaluation metrics and visualizations

Example of using linear algebra functions:

```rust
use scirs2_neural::linalg::batch_operations;
use ndarray::Array3;

// Batch matrix multiplication
let a = Array3::<f64>::zeros((32, 10, 20));
let b = Array3::<f64>::zeros((32, 20, 15));
let result = batch_operations::batch_matmul(&a, &b);
```

### Metrics Integration

With the `metrics_integration` feature, you can use scirs2-metrics for advanced evaluation:

```rust
use scirs2_metrics::integration::neural::{NeuralMetricAdapter, MetricsCallback};
use scirs2_neural::callbacks::ScirsMetricsCallback;
use scirs2_neural::evaluation::MetricType;

// Create metric adapters
let metrics = vec![
    NeuralMetricAdapter::<f32>::accuracy(),
    NeuralMetricAdapter::<f32>::precision(),
    NeuralMetricAdapter::<f32>::f1_score(),
    NeuralMetricAdapter::<f32>::mse(),
    NeuralMetricAdapter::<f32>::r2(),
];

// Create callback for tracking metrics during training
let metrics_callback = ScirsMetricsCallback::new(metrics);

// Train model with metrics tracking
model.fit(&x_train, &y_train, 
    epochs, 
    batch_size, 
    Some(&[&metrics_callback]), 
    None
)?;

// Get evaluation metrics
let eval_results = model.evaluate(
    &x_test, 
    &y_test, 
    Some(batch_size),
    Some(vec![
        MetricType::Accuracy,
        MetricType::Precision,
        MetricType::F1Score,
    ])
)?;

// Visualize results
let roc_viz = neural_roc_curve_visualization(&y_true, &y_pred, Some(auc))?;
```

## Contributing

See the [CONTRIBUTING.md](../CONTRIBUTING.md) file for contribution guidelines.

## License

This project is dual-licensed under:

- [MIT License](../LICENSE-MIT)
- [Apache License Version 2.0](../LICENSE-APACHE)

You can choose to use either license. See the [LICENSE](../LICENSE) file for details.
