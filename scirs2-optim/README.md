# SciRS2 Optim

[![crates.io](https://img.shields.io/crates/v/scirs2-optim.svg)](https://crates.io/crates/scirs2-optim)
[[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)]](../LICENSE)
[![Documentation](https://img.shields.io/docsrs/scirs2-optim)](https://docs.rs/scirs2-optim)

Optimization algorithms for the SciRS2 scientific computing library. This module provides various optimizers, regularizers, and learning rate schedulers for machine learning and numerical optimization tasks.

## Features

- **Optimizers**: Various first-order optimization algorithms (SGD, Adam, RMSProp, etc.)
- **Regularizers**: Regularization techniques to prevent overfitting (L1, L2, Elastic Net, Dropout)
- **Learning Rate Schedulers**: Techniques for adjusting learning rates during training
- **Utility Functions**: Additional utilities for optimization tasks

## Installation

Add the following to your `Cargo.toml`:

```toml
[dependencies]
scirs2-optim = "0.1.0-alpha.4"
```

To enable optimizations or integration with other modules:

```toml
[dependencies]
scirs2-optim = { version = "0.1.0-alpha.4", features = ["parallel"] }

# For integration with scirs2-metrics
scirs2-optim = { version = "0.1.0-alpha.4", features = ["metrics_integration"] }
```

## Usage

Basic usage examples:

```rust
use scirs2_optim::{optimizers, regularizers, schedulers};
use scirs2_core::error::CoreResult;
use ndarray::array;

// Optimizer example: Stochastic Gradient Descent
fn sgd_optimizer_example() -> CoreResult<()> {
    // Create parameters
    let mut params = array![1.0, 2.0, 3.0];
    
    // Create gradients (computed elsewhere)
    let grads = array![0.1, 0.2, 0.3];
    
    // Create SGD optimizer with learning rate 0.01
    let mut optimizer = optimizers::sgd::SGD::new(0.01, 0.9, false);
    
    // Update parameters
    optimizer.step(&mut params, &grads)?;
    
    println!("Updated parameters: {:?}", params);
    
    Ok(())
}

// Adam optimizer with a learning rate scheduler
fn adam_with_scheduler_example() -> CoreResult<()> {
    // Create parameters
    let mut params = array![1.0, 2.0, 3.0];
    
    // Create Adam optimizer with default parameters
    let mut optimizer = optimizers::adam::Adam::new(0.001, 0.9, 0.999, 1e-8);
    
    // Create a learning rate scheduler (exponential decay)
    let mut scheduler = schedulers::exponential_decay::ExponentialDecay::new(
        0.001,  // initial learning rate
        0.95,   // decay rate
        100     // decay steps
    )?;
    
    // Training loop (simplified)
    for epoch in 0..1000 {
        // Compute gradients (would normally be from a model)
        let grads = array![0.1, 0.2, 0.3];
        
        // Update learning rate based on epoch
        let lr = scheduler.get_learning_rate(epoch)?;
        optimizer.set_learning_rate(lr);
        
        // Update parameters
        optimizer.step(&mut params, &grads)?;
        
        if epoch % 100 == 0 {
            println!("Epoch {}, LR: {}, Params: {:?}", epoch, lr, params);
        }
    }
    
    Ok(())
}

// Regularization example
fn regularization_example() -> CoreResult<()> {
    // Parameters
    let params = array![1.0, 2.0, 3.0];
    
    // L1 regularization (Lasso)
    let l1_reg = regularizers::l1::L1::new(0.01);
    let l1_penalty = l1_reg.regularization_term(&params)?;
    let l1_grad = l1_reg.gradient(&params)?;
    
    println!("L1 penalty: {}", l1_penalty);
    println!("L1 gradient contribution: {:?}", l1_grad);
    
    // L2 regularization (Ridge)
    let l2_reg = regularizers::l2::L2::new(0.01);
    let l2_penalty = l2_reg.regularization_term(&params)?;
    let l2_grad = l2_reg.gradient(&params)?;
    
    println!("L2 penalty: {}", l2_penalty);
    println!("L2 gradient contribution: {:?}", l2_grad);
    
    // Elastic Net (combination of L1 and L2)
    let elastic_net = regularizers::elastic_net::ElasticNet::new(0.01, 0.5)?;
    let elastic_penalty = elastic_net.regularization_term(&params)?;
    
    println!("Elastic Net penalty: {}", elastic_penalty);
    
    Ok(())
}
```

## Components

### Optimizers

Optimization algorithms for machine learning:

```rust
use scirs2_optim::optimizers::{
    Optimizer,              // Optimizer trait
    sgd::SGD,               // Stochastic Gradient Descent
    adagrad::AdaGrad,       // Adaptive Gradient Algorithm
    rmsprop::RMSprop,       // Root Mean Square Propagation
    adam::Adam,             // Adaptive Moment Estimation
};
```

### Regularizers

Regularization techniques for preventing overfitting:

```rust
use scirs2_optim::regularizers::{
    Regularizer,            // Regularizer trait
    l1::L1,                 // L1 regularization (Lasso)
    l2::L2,                 // L2 regularization (Ridge)
    elastic_net::ElasticNet, // Elastic Net regularization
    dropout::Dropout,       // Dropout regularization
};
```

### Learning Rate Schedulers

Learning rate adjustment strategies:

```rust
use scirs2_optim::schedulers::{
    Scheduler,              // Scheduler trait
    exponential_decay::ExponentialDecay, // Exponential decay scheduler
    linear_decay::LinearDecay, // Linear decay scheduler
    step_decay::StepDecay,  // Step decay scheduler
    cosine_annealing::CosineAnnealing, // Cosine annealing scheduler
    reduce_on_plateau::ReduceOnPlateau, // Reduce learning rate when metric plateaus
};
```

## Advanced Features

### Integration with Metrics

The `metrics_integration` feature provides integration with `scirs2-metrics` for metric-based optimization:

```rust
use scirs2_optim::metrics::{MetricOptimizer, MetricScheduler, MetricBasedReduceOnPlateau};
use scirs2_optim::optimizers::{SGD, Optimizer};

// Create an SGD optimizer guided by metrics
let mut optimizer = MetricOptimizer::new(
    SGD::new(0.01), 
    "accuracy",  // Metric to optimize
    true        // Maximize
);

// Create a metric-guided learning rate scheduler
let mut scheduler = MetricBasedReduceOnPlateau::new(
    0.1,        // Initial learning rate
    0.5,        // Factor to reduce learning rate (0.5 = halve it)
    3,          // Patience - number of epochs with no improvement
    0.001,      // Minimum learning rate
    "val_loss", // Metric name to monitor
    false,      // Maximize? No, we want to minimize loss
);

// During training loop:
for epoch in 0..num_epochs {
    // Train model for one epoch...
    let train_metrics = train_epoch(&model, &train_data);
    
    // Evaluate on validation set
    let val_metrics = evaluate(&model, &val_data);
    
    // Update optimizer with metric value
    optimizer.update_metric(train_metrics.accuracy);
    
    // Update scheduler with validation loss
    let new_lr = scheduler.step_with_metric(val_metrics.loss);
    
    // Apply scheduler to optimizer
    scheduler.apply_to(&mut optimizer);
    
    // Print current learning rate
    println!("Epoch {}: LR = {}", epoch, new_lr);
}
```

For more advanced hyperparameter tuning, the integration also supports random search:

```rust
use scirs2_metrics::integration::optim::{HyperParameterTuner, HyperParameter};

// Define hyperparameters to tune
let params = vec![
    HyperParameter::new("learning_rate", 0.01, 0.001, 0.1),
    HyperParameter::new("weight_decay", 0.0001, 0.0, 0.001),
    HyperParameter::discrete("batch_size", 32.0, 16.0, 128.0, 16.0),
];

// Create hyperparameter tuner
let mut tuner = HyperParameterTuner::new(
    params,
    "validation_accuracy",  // Metric to optimize
    true,                   // Maximize
    30                      // Number of trials
);

// Run random search
let result = tuner.random_search(|params| {
    // Train model with these parameters
    let model_result = train_model_with_params(params)?;
    Ok(model_result.val_accuracy)
})?;

// Get best parameters
println!("Best hyperparameters:");
for (name, value) in result.best_params() {
    println!("  {}: {}", name, value);
}
```

### Combining Optimizers and Regularizers

Example of how to use optimizers with regularizers:

```rust
use scirs2_optim::{optimizers::adam::Adam, regularizers::l2::L2};
use ndarray::Array1;

// Create parameters
let mut params = Array1::from_vec(vec![1.0, 2.0, 3.0]);

// Create gradients (computed elsewhere)
let mut grads = Array1::from_vec(vec![0.1, 0.2, 0.3]);

// Create optimizer
let mut optimizer = Adam::new(0.001, 0.9, 0.999, 1e-8);

// Create regularizer
let regularizer = L2::new(0.01);

// Add regularization gradient
let reg_grads = regularizer.gradient(&params).unwrap();
grads += &reg_grads;

// Update parameters
optimizer.step(&mut params, &grads).unwrap();
```

### Custom Learning Rate Schedulers

Creating a custom learning rate scheduler:

```rust
use scirs2_optim::schedulers::Scheduler;
use scirs2_core::error::{CoreError, CoreResult};

struct CustomScheduler {
    initial_lr: f64,
}

impl CustomScheduler {
    fn new(initial_lr: f64) -> Self {
        Self { initial_lr }
    }
}

impl Scheduler for CustomScheduler {
    fn get_learning_rate(&mut self, epoch: usize) -> CoreResult<f64> {
        // Custom learning rate schedule
        // Example: square root decay
        Ok(self.initial_lr / (1.0 + epoch as f64).sqrt())
    }
}
```

## Examples

The module includes several example applications:

- SGD optimization example
- Adam optimizer with learning rate scheduling
- Regularization techniques showcase
- Custom optimization workflows

## Contributing

See the [CONTRIBUTING.md](../CONTRIBUTING.md) file for contribution guidelines.

## License

This project is dual-licensed under:

- [MIT License](../LICENSE-MIT)
- [Apache License Version 2.0](../LICENSE-APACHE)

You can choose to use either license. See the [LICENSE](../LICENSE) file for details.
