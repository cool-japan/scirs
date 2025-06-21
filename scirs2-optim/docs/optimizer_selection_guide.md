# SciRS2 Optimizer Selection Guide

This guide provides comprehensive recommendations for selecting the appropriate optimizer for different machine learning tasks and scenarios.

## Table of Contents

1. [Quick Selection Guide](#quick-selection-guide)
2. [Optimizer Categories](#optimizer-categories)
3. [Task-Specific Recommendations](#task-specific-recommendations)
4. [Performance Characteristics](#performance-characteristics)
5. [Hyperparameter Tuning Guidelines](#hyperparameter-tuning-guidelines)
6. [Common Pitfalls and Solutions](#common-pitfalls-and-solutions)
7. [Advanced Usage Patterns](#advanced-usage-patterns)

## Quick Selection Guide

| Scenario | Recommended Optimizer | Alternative Options |
|----------|----------------------|-------------------|
| **Computer Vision (Large datasets)** | Adam, AdamW | SGD with momentum, RMSprop |
| **Natural Language Processing** | AdamW, LAMB | Adam, RAdam |
| **Small datasets** | SGD with momentum | L-BFGS, AdaGrad |
| **Sparse features** | AdaGrad, SparseAdam | RMSprop |
| **Large batch training** | LAMB, LARS | SGD with momentum |
| **Fine-tuning pre-trained models** | AdamW with low LR | Adam with cosine annealing |
| **Research/Experimentation** | Adam | AdamW, RAdam |
| **Production deployment** | SGD with momentum | AdamW |

## Optimizer Categories

### First-Order Optimizers

#### Gradient Descent Variants
- **SGD**: Simple, reliable, memory-efficient
- **SGD with Momentum**: Better convergence, handles noisy gradients
- **Nesterov Momentum**: Improved momentum with look-ahead

#### Adaptive Learning Rate Optimizers
- **AdaGrad**: Good for sparse features, decreasing learning rates
- **RMSprop**: Addresses AdaGrad's aggressive learning rate decay
- **Adam**: Combines momentum and adaptive learning rates
- **AdamW**: Adam with decoupled weight decay

#### Advanced Adaptive Optimizers
- **RAdam**: Rectified Adam with variance rectification
- **Lookahead**: Meta-optimizer that can wrap other optimizers
- **Lion**: EvoLved Sign Momentum optimizer
- **SAM**: Sharpness-Aware Minimization for better generalization

### Second-Order Optimizers
- **L-BFGS**: Quasi-Newton method for small-scale problems
- **Natural Gradient Methods**: Use Fisher information matrix

### Specialized Optimizers
- **LAMB**: Layer-wise Adaptive Moments for large batch training
- **LARS**: Layer-wise Adaptive Rate Scaling
- **SparseAdam**: Optimized for sparse gradients

## Task-Specific Recommendations

### Computer Vision

#### Image Classification
```rust
use scirs2_optim::optimizers::{Adam, AdamConfig};
use scirs2_optim::schedulers::{CosineAnnealingScheduler, SchedulerConfig};

// For large datasets (ImageNet-scale)
let adam_config = AdamConfig {
    learning_rate: 0.001,
    beta1: 0.9,
    beta2: 0.999,
    epsilon: 1e-8,
    weight_decay: 0.0001,
    ..Default::default()
};

// With cosine annealing
let scheduler_config = SchedulerConfig::CosineAnnealing {
    t_max: 100,
    eta_min: 1e-6,
};
```

#### Object Detection
```rust
// SGD with momentum is often preferred for object detection
let sgd_config = SGDConfig {
    learning_rate: 0.01,
    momentum: 0.9,
    weight_decay: 0.0001,
    nesterov: true,
    ..Default::default()
};

// With step decay
let scheduler_config = SchedulerConfig::StepDecay {
    step_size: 30,
    gamma: 0.1,
};
```

### Natural Language Processing

#### Transformer Models
```rust
// AdamW is the gold standard for transformer training
let adamw_config = AdamWConfig {
    learning_rate: 2e-5,  // Typical for fine-tuning
    beta1: 0.9,
    beta2: 0.999,
    epsilon: 1e-8,
    weight_decay: 0.01,   // Crucial for AdamW
    ..Default::default()
};

// With linear warmup
let scheduler_config = SchedulerConfig::LinearWarmup {
    warmup_steps: 1000,
    total_steps: 10000,
    final_lr_ratio: 0.0,
};
```

#### RNN/LSTM Models
```rust
// RMSprop or Adam work well for RNNs
let rmsprop_config = RMSpropConfig {
    learning_rate: 0.001,
    alpha: 0.9,
    epsilon: 1e-8,
    weight_decay: 0.0,
    momentum: 0.0,
    centered: false,
    ..Default::default()
};

// Important: gradient clipping for RNNs
let gradient_config = GradientConfig {
    clip_method: ClipMethod::Norm(1.0),
    ..Default::default()
};
```

### Recommendation Systems

#### Collaborative Filtering
```rust
// AdaGrad works well with sparse features
let adagrad_config = AdaGradConfig {
    learning_rate: 0.01,
    epsilon: 1e-8,
    weight_decay: 0.0,
    lr_decay: 0.0,
    ..Default::default()
};
```

#### Deep Learning Based
```rust
// Adam for deep recommendation models
let adam_config = AdamConfig {
    learning_rate: 0.001,
    beta1: 0.9,
    beta2: 0.999,
    epsilon: 1e-8,
    weight_decay: 1e-5,
    ..Default::default()
};
```

### Time Series Forecasting

```rust
// L-BFGS for small datasets, Adam for large datasets
let lbfgs_config = LBFGSConfig {
    history_size: 10,
    max_iter: 20,
    tolerance_grad: 1e-5,
    tolerance_change: 1e-9,
    line_search_fn: LineSearchFunction::StrongWolfe,
    ..Default::default()
};
```

## Performance Characteristics

### Convergence Speed
- **Fastest**: L-BFGS (small problems), Adam/AdamW (large problems)
- **Moderate**: RMSprop, RAdam
- **Slower but Stable**: SGD with momentum

### Memory Usage
- **Lowest**: SGD (only gradients)
- **Low**: SGD with momentum
- **Moderate**: AdaGrad, RMSprop
- **High**: Adam, AdamW, L-BFGS

### Generalization Performance
- **Best**: SGD with momentum, SAM
- **Good**: AdamW with proper weight decay
- **Variable**: Adam (can overfit)

### Robustness to Hyperparameters
- **Most Robust**: Adam, AdamW
- **Moderately Robust**: RMSprop
- **Less Robust**: SGD, L-BFGS

## Hyperparameter Tuning Guidelines

### Learning Rate Selection

```rust
// Learning rate ranges for different optimizers
let lr_ranges = HashMap::from([
    ("SGD", (0.01, 0.1)),
    ("Adam", (1e-4, 1e-2)),
    ("AdamW", (1e-5, 1e-2)),
    ("RMSprop", (1e-4, 1e-2)),
    ("AdaGrad", (0.01, 0.1)),
]);
```

### Batch Size Considerations

```rust
// Learning rate scaling with batch size
fn scale_learning_rate(base_lr: f64, batch_size: usize, base_batch_size: usize) -> f64 {
    base_lr * (batch_size as f64 / base_batch_size as f64).sqrt()
}

// For linear scaling (large batch training)
fn linear_scale_learning_rate(base_lr: f64, batch_size: usize, base_batch_size: usize) -> f64 {
    base_lr * (batch_size as f64 / base_batch_size as f64)
}
```

### Weight Decay Guidelines

| Optimizer | Recommended Weight Decay | Notes |
|-----------|-------------------------|-------|
| SGD | 1e-4 to 5e-4 | Applied to all parameters |
| Adam | 0 | Use AdamW instead for weight decay |
| AdamW | 0.01 to 0.1 | Decoupled weight decay |
| RMSprop | 1e-5 to 1e-4 | Usually small values |

## Common Pitfalls and Solutions

### Problem: Training Loss Oscillates

**Symptoms**: Loss jumps up and down, unstable training
**Causes**: Learning rate too high, batch size too small
**Solutions**:
```rust
// Reduce learning rate
let config = config.with_learning_rate(config.learning_rate * 0.1);

// Use learning rate scheduling
let scheduler = ExponentialDecayScheduler::new(0.95);

// Increase batch size or use gradient accumulation
let gradient_accumulation_steps = 4;
```

### Problem: Slow Convergence

**Symptoms**: Loss decreases very slowly
**Causes**: Learning rate too low, poor initialization
**Solutions**:
```rust
// Increase learning rate gradually
let warmup_scheduler = LinearWarmupScheduler::new(1000);

// Use adaptive optimizers
let adam_config = AdamConfig::default();

// Check initialization
let init_std = (2.0 / fan_in as f64).sqrt();
```

### Problem: Overfitting

**Symptoms**: Training loss much lower than validation loss
**Causes**: Insufficient regularization
**Solutions**:
```rust
// Add weight decay
let adamw_config = AdamWConfig {
    weight_decay: 0.01,
    ..Default::default()
};

// Use dropout
let dropout_rate = 0.1;

// Reduce model capacity or add early stopping
```

### Problem: Gradient Explosion (RNNs)

**Symptoms**: Loss becomes NaN or very large
**Causes**: Gradients become too large in recurrent networks
**Solutions**:
```rust
// Gradient clipping
let gradient_processor = GradientProcessor::new(
    GradientConfig {
        clip_method: ClipMethod::Norm(1.0),
        ..Default::default()
    }
);

// Lower learning rate
let reduced_lr = 0.001;
```

## Advanced Usage Patterns

### Multi-Optimizer Workflows

```rust
use scirs2_optim::composition::OptimizerComposition;

// Pre-training with Adam, fine-tuning with SGD
let pretraining_optimizer = Adam::new(adam_config);
let finetuning_optimizer = SGD::new(sgd_config);

let composition = OptimizerComposition::new()
    .stage(pretraining_optimizer, 1000)  // 1000 steps
    .stage(finetuning_optimizer, 500);   // 500 steps
```

### Parameter-Specific Optimization

```rust
// Different optimizers for different parameter groups
let weight_optimizer = SGD::new(sgd_config);
let bias_optimizer = Adam::new(adam_config);

let group_optimizer = ParameterGroupOptimizer::new()
    .group("weights", weight_optimizer)
    .group("biases", bias_optimizer);
```

### Learning Rate Scheduling

```rust
// Combine multiple scheduling strategies
let combined_scheduler = CombinedScheduler::new()
    .add(LinearWarmupScheduler::new(1000))
    .add(CosineAnnealingScheduler::new(10000));

// Adaptive scheduling based on validation loss
let adaptive_scheduler = ReduceOnPlateauScheduler::new(
    ReduceOnPlateauConfig {
        patience: 10,
        factor: 0.5,
        min_lr: 1e-6,
        ..Default::default()
    }
);
```

### Meta-Learning Integration

```rust
use scirs2_optim::meta_learning::{MetaOptimizer, HyperparameterStrategy};

// Automatic hyperparameter optimization
let meta_config = MetaOptimizerConfig {
    strategy: HyperparameterStrategy::BayesianOptimization {
        num_trials: 50,
        bounds: HashMap::from([
            ("learning_rate".to_string(), (1e-5, 1e-1)),
            ("weight_decay".to_string(), (0.0, 0.1)),
        ]),
        acquisition: AcquisitionFunction::ExpectedImprovement,
    },
    ..Default::default()
};

let meta_optimizer = MetaOptimizer::new(meta_config);
```

## Best Practices Summary

1. **Start Simple**: Begin with Adam or SGD with momentum
2. **Monitor Closely**: Watch training/validation curves
3. **Use Weight Decay**: Essential for generalization
4. **Learning Rate Scheduling**: Almost always beneficial
5. **Gradient Clipping**: Crucial for RNNs and transformers
6. **Warmup**: Important for large batch training
7. **Early Stopping**: Prevent overfitting
8. **Reproducibility**: Set random seeds and use deterministic operations

## References

- Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization.
- Loshchilov, I., & Hutter, F. (2017). Decoupled weight decay regularization.
- You, Y., et al. (2017). Large batch training of convolutional networks.
- Zhang, M., et al. (2019). Lookahead optimizer: k steps forward, 1 step back.
- Foret, P., et al. (2020). Sharpness-aware minimization for efficiently improving generalization.