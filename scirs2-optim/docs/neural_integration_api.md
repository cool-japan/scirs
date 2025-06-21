# Neural Network Integration API Reference

The `neural_integration` module provides comprehensive interfaces for integrating optimizers with neural networks, including parameter management, architecture-aware optimizations, and forward/backward pass hooks.

## Table of Contents

1. [Core Traits](#core-traits)
2. [Parameter Management](#parameter-management)
3. [Architecture-Aware Optimization](#architecture-aware-optimization)
4. [Forward/Backward Integration](#forwardbackward-integration)
5. [Usage Examples](#usage-examples)
6. [Best Practices](#best-practices)

## Core Traits

### ParameterOptimizer

The main trait for neural network parameter optimization.

```rust
pub trait ParameterOptimizer<A: Float, D: Dimension> {
    /// Register a parameter for optimization
    fn register_parameter(
        &mut self,
        param_id: ParamId,
        parameter: &Array<A, D>,
        metadata: ParameterMetadata,
    ) -> Result<()>;

    /// Update registered parameters with gradients
    fn step(
        &mut self,
        gradients: HashMap<ParamId, Array<A, D>>,
        parameters: &mut HashMap<ParamId, Array<A, D>>,
    ) -> Result<()>;

    /// Get parameter-specific learning rate
    fn get_learning_rate(&self, param_id: &ParamId) -> Option<A>;

    /// Set parameter-specific learning rate
    fn set_learning_rate(&mut self, param_id: &ParamId, lr: A) -> Result<()>;

    /// Get optimizer state for a parameter
    fn get_parameter_state(&self, param_id: &ParamId) -> Option<&HashMap<String, Array<A, D>>>;

    /// Reset optimizer state
    fn reset_state(&mut self);

    /// Get all registered parameter IDs
    fn registered_parameters(&self) -> Vec<ParamId>;
}
```

**Key Features:**
- Generic over number types (`Float`) and tensor dimensions (`Dimension`)
- Lazy parameter registration support
- Parameter-specific learning rates
- State management for advanced optimizers
- Metadata-driven optimization decisions

## Parameter Management

### ParameterManager

Central component for managing neural network parameters.

```rust
pub struct ParameterManager<A: Float, D: Dimension> {
    // Internal fields...
}

impl<A: Float + ScalarOperand + Debug, D: Dimension> ParameterManager<A, D> {
    /// Create a new parameter manager
    pub fn new(config: OptimizationConfig<A>) -> Self;

    /// Enable lazy registration mode
    pub fn enable_lazy_mode(&mut self);

    /// Disable lazy registration mode and process pending registrations
    pub fn disable_lazy_mode(&mut self) -> Result<()>;

    /// Register a parameter
    pub fn register_parameter(
        &mut self,
        param_id: ParamId,
        metadata: ParameterMetadata,
    ) -> Result<()>;

    /// Register a layer architecture
    pub fn register_layer(&mut self, layer_id: LayerId, architecture: LayerArchitecture);

    /// Set layer-specific optimization rule
    pub fn set_layer_rule(&mut self, layer_id: LayerId, rule: LayerOptimizationRule<A>);

    /// Get effective learning rate for a parameter
    pub fn get_effective_learning_rate(&self, param_id: &ParamId) -> A;

    /// Get effective weight decay for a parameter
    pub fn get_effective_weight_decay(&self, param_id: &ParamId) -> A;

    /// Check if parameter is frozen
    pub fn is_parameter_frozen(&self, param_id: &ParamId) -> bool;
}
```

### ParameterMetadata

Comprehensive metadata for neural network parameters.

```rust
pub struct ParameterMetadata {
    /// Layer name this parameter belongs to
    pub layer_name: LayerId,
    /// Parameter name within the layer
    pub param_name: ParamId,
    /// Parameter shape
    pub shape: Vec<usize>,
    /// Whether parameter requires gradients
    pub requires_grad: bool,
    /// Parameter type (weights, bias, etc.)
    pub param_type: ParameterType,
    /// Sharing group for parameter sharing
    pub sharing_group: Option<String>,
    /// Custom tags for architecture-specific optimizations
    pub tags: Vec<String>,
}

pub enum ParameterType {
    Weight,
    Bias,
    Normalization,
    Embedding,
    Attention,
    Custom,
}
```

### OptimizationConfig

Global optimization configuration.

```rust
pub struct OptimizationConfig<A: Float> {
    /// Base learning rate
    pub base_learning_rate: A,
    /// Global weight decay
    pub weight_decay: A,
    /// Gradient clipping threshold
    pub gradient_clip: Option<A>,
    /// Whether to use mixed precision
    pub mixed_precision: bool,
    /// Architecture-specific optimizations
    pub architecture_optimizations: HashMap<String, bool>,
}
```

## Architecture-Aware Optimization

### ArchitectureStrategy

Strategies for different neural network architectures.

```rust
pub enum ArchitectureStrategy {
    /// Transformer-specific optimizations
    Transformer {
        /// Use different learning rates for different components
        component_specific_lr: bool,
        /// Apply layer-wise learning rate decay
        layer_wise_decay: bool,
        /// Warmup steps for attention parameters
        attention_warmup: usize,
    },
    /// CNN-specific optimizations
    ConvolutionalNet {
        /// Use different learning rates for conv vs fc layers
        layer_type_lr: bool,
        /// Apply depth-wise learning rate scaling
        depth_scaling: bool,
        /// Batch norm parameter handling
        bn_special_handling: bool,
    },
    /// RNN-specific optimizations
    RecurrentNet {
        /// Gradient clipping specifically for RNNs
        rnn_gradient_clip: Option<f64>,
        /// Different learning rates for recurrent vs linear weights
        weight_type_lr: bool,
    },
    /// Custom architecture strategy
    Custom {
        /// Custom optimization rules
        rules: HashMap<String, LayerConfig>,
    },
}
```

### ArchitectureAwareOptimizer

Optimizer that applies architecture-specific optimizations.

```rust
pub struct ArchitectureAwareOptimizer<A: Float, D: Dimension> {
    // Internal fields...
}

impl<A: Float + ScalarOperand + Debug, D: Dimension> ArchitectureAwareOptimizer<A, D> {
    /// Create a new architecture-aware optimizer
    pub fn new(config: OptimizationConfig<A>, strategy: ArchitectureStrategy) -> Self;

    /// Apply architecture-specific optimizations
    pub fn apply_architecture_optimizations(&mut self) -> Result<()>;

    /// Step the optimizer
    pub fn step(&mut self) -> Result<()>;

    /// Get parameter manager
    pub fn parameter_manager(&self) -> &ParameterManager<A, D>;
}
```

## Forward/Backward Integration

### Hook Traits

```rust
/// Forward pass hook for parameter tracking
pub trait ForwardHook<A: Float, D: Dimension> {
    /// Called before layer forward pass
    fn pre_forward(&mut self, layer_id: &LayerId, inputs: &[Array<A, D>]) -> Result<()>;

    /// Called after layer forward pass
    fn post_forward(&mut self, layer_id: &LayerId, outputs: &[Array<A, D>]) -> Result<()>;
}

/// Backward pass hook for gradient processing
pub trait BackwardHook<A: Float, D: Dimension> {
    /// Called before layer backward pass
    fn pre_backward(&mut self, layer_id: &LayerId, grad_outputs: &[Array<A, D>]) -> Result<()>;

    /// Called after layer backward pass
    fn post_backward(&mut self, layer_id: &LayerId, grad_inputs: &[Array<A, D>]) -> Result<()>;
}
```

### NeuralIntegration

Main integration manager for forward/backward passes.

```rust
pub struct NeuralIntegration<A: Float, D: Dimension> {
    // Internal fields...
}

impl<A: Float + ScalarOperand + Debug + 'static, D: Dimension + 'static> NeuralIntegration<A, D> {
    /// Create a new neural integration manager
    pub fn new(config: OptimizationConfig<A>) -> Self;

    /// Register a forward hook for a layer
    pub fn register_forward_hook<H>(&mut self, layer_id: LayerId, hook: H)
    where
        H: ForwardHook<A, D> + 'static;

    /// Register a backward hook for a layer
    pub fn register_backward_hook<H>(&mut self, layer_id: LayerId, hook: H)
    where
        H: BackwardHook<A, D> + 'static;

    /// Enable gradient accumulation
    pub fn enable_gradient_accumulation(&mut self);

    /// Disable gradient accumulation and return accumulated gradients
    pub fn disable_gradient_accumulation(&mut self) -> HashMap<ParamId, Array<A, D>>;

    /// Execute forward pass with hooks
    pub fn forward_pass(
        &mut self,
        layer_id: &LayerId,
        inputs: &[Array<A, D>],
    ) -> Result<Vec<Array<A, D>>>;

    /// Execute backward pass with hooks
    pub fn backward_pass(
        &mut self,
        layer_id: &LayerId,
        grad_outputs: &[Array<A, D>],
    ) -> Result<Vec<Array<A, D>>>;

    /// Accumulate gradients for parameters
    pub fn accumulate_gradients(
        &mut self,
        gradients: HashMap<ParamId, Array<A, D>>,
    ) -> Result<()>;
}
```

## Usage Examples

### Basic Parameter Management

```rust
use scirs2_optim::neural_integration::*;
use ndarray::Array1;

// Create optimization configuration
let config = OptimizationConfig {
    base_learning_rate: 0.001,
    weight_decay: 0.01,
    gradient_clip: Some(1.0),
    mixed_precision: false,
    architecture_optimizations: HashMap::new(),
};

// Create parameter manager
let mut manager = ParameterManager::<f64, ndarray::Ix1>::new(config);

// Register parameters
let weight_metadata = ParameterMetadata {
    layer_name: "dense1".to_string(),
    param_name: "weight".to_string(),
    shape: vec![128, 64],
    requires_grad: true,
    param_type: ParameterType::Weight,
    sharing_group: None,
    tags: vec!["dense".to_string()],
};

manager.register_parameter("dense1_weight".to_string(), weight_metadata)?;

// Set layer-specific rules
let layer_rule = LayerOptimizationRule {
    lr_multiplier: 1.5,
    weight_decay_multiplier: 0.8,
    frozen: false,
    custom_settings: HashMap::new(),
};

manager.set_layer_rule("dense1".to_string(), layer_rule);

// Get effective learning rate
let effective_lr = manager.get_effective_learning_rate(&"dense1_weight".to_string());
println!("Effective learning rate: {}", effective_lr);
```

### Architecture-Aware Optimization

```rust
use scirs2_optim::neural_integration::architecture_aware::*;

// Configure Transformer-specific optimizations
let strategy = ArchitectureStrategy::Transformer {
    component_specific_lr: true,
    layer_wise_decay: true,
    attention_warmup: 1000,
};

// Create architecture-aware optimizer
let mut optimizer = ArchitectureAwareOptimizer::<f64, ndarray::Ix1>::new(config, strategy);

// Register layer architecture
let layer_arch = LayerArchitecture {
    layer_type: "transformer_block".to_string(),
    input_dims: vec![512],
    output_dims: vec![512],
    config: HashMap::new(),
    trainable: true,
};

optimizer
    .parameter_manager_mut()
    .register_layer("layer_0".to_string(), layer_arch);

// Register parameters with appropriate tags
let attention_metadata = ParameterMetadata {
    layer_name: "layer_0".to_string(),
    param_name: "attention_weight".to_string(),
    shape: vec![512, 512],
    requires_grad: true,
    param_type: ParameterType::Attention,
    sharing_group: None,
    tags: vec!["attention".to_string()],
};

optimizer
    .parameter_manager_mut()
    .register_parameter("attn_param".to_string(), attention_metadata)?;

// Apply architecture optimizations
optimizer.apply_architecture_optimizations()?;
```

### Forward/Backward Hook Integration

```rust
use scirs2_optim::neural_integration::forward_backward::*;

// Custom forward hook implementation
struct ActivationLogger<A: Float, D: Dimension> {
    activations: HashMap<LayerId, Vec<Array<A, D>>>,
}

impl<A: Float, D: Dimension> ForwardHook<A, D> for ActivationLogger<A, D> {
    fn pre_forward(&mut self, layer_id: &LayerId, inputs: &[Array<A, D>]) -> Result<()> {
        println!("Forward pass starting for layer: {}", layer_id);
        Ok(())
    }

    fn post_forward(&mut self, layer_id: &LayerId, outputs: &[Array<A, D>]) -> Result<()> {
        self.activations.insert(layer_id.clone(), outputs.to_vec());
        println!("Forward pass completed for layer: {}", layer_id);
        Ok(())
    }
}

// Custom backward hook implementation
struct GradientLogger<A: Float, D: Dimension> {
    gradients: HashMap<LayerId, Vec<Array<A, D>>>,
}

impl<A: Float, D: Dimension> BackwardHook<A, D> for GradientLogger<A, D> {
    fn pre_backward(&mut self, layer_id: &LayerId, grad_outputs: &[Array<A, D>]) -> Result<()> {
        println!("Backward pass starting for layer: {}", layer_id);
        Ok(())
    }

    fn post_backward(&mut self, layer_id: &LayerId, grad_inputs: &[Array<A, D>]) -> Result<()> {
        self.gradients.insert(layer_id.clone(), grad_inputs.to_vec());
        println!("Backward pass completed for layer: {}", layer_id);
        Ok(())
    }
}

// Setup neural integration with hooks
let mut integration = NeuralIntegration::<f64, ndarray::Ix1>::new(config);

let activation_logger = ActivationLogger {
    activations: HashMap::new(),
};
let gradient_logger = GradientLogger {
    gradients: HashMap::new(),
};

integration.register_forward_hook("layer1".to_string(), activation_logger);
integration.register_backward_hook("layer1".to_string(), gradient_logger);

// Enable gradient accumulation
integration.enable_gradient_accumulation();

// Execute forward and backward passes
let inputs = vec![Array1::from_vec(vec![1.0, 2.0, 3.0])];
let outputs = integration.forward_pass(&"layer1".to_string(), &inputs)?;

let grad_outputs = vec![Array1::from_vec(vec![0.1, 0.2, 0.3])];
let grad_inputs = integration.backward_pass(&"layer1".to_string(), &grad_outputs)?;

// Accumulate gradients
let parameter_gradients = HashMap::from([
    ("layer1_weight".to_string(), Array1::from_vec(vec![0.05, 0.1, 0.15])),
    ("layer1_bias".to_string(), Array1::from_vec(vec![0.01, 0.02, 0.03])),
]);

integration.accumulate_gradients(parameter_gradients)?;

// Get accumulated gradients
let accumulated = integration.disable_gradient_accumulation();
```

### Lazy Registration

```rust
// Enable lazy mode for batch parameter registration
let mut manager = ParameterManager::<f64, ndarray::Ix1>::new(config);
manager.enable_lazy_mode();

// Register multiple parameters (stored in pending queue)
for i in 0..1000 {
    let metadata = ParameterMetadata {
        layer_name: format!("layer_{}", i / 10),
        param_name: format!("param_{}", i),
        shape: vec![64, 32],
        requires_grad: true,
        param_type: ParameterType::Weight,
        sharing_group: None,
        tags: vec![],
    };
    manager.register_parameter(format!("param_{}", i), metadata)?;
}

// Process all pending registrations at once
manager.disable_lazy_mode()?;

// Now all parameters are registered and available
let all_params = manager.get_all_parameters();
println!("Registered {} parameters", all_params.len());
```

## Best Practices

### Parameter Organization

1. **Use Descriptive Names**: Parameter IDs should clearly indicate layer and parameter type
2. **Consistent Tagging**: Use consistent tags for architecture-specific optimizations
3. **Layer Grouping**: Group related parameters by layer for easier management
4. **Metadata Completeness**: Provide complete metadata for optimal optimization

### Performance Optimization

1. **Lazy Registration**: Use lazy mode when registering many parameters
2. **Efficient Hooks**: Keep forward/backward hooks lightweight
3. **Memory Management**: Be mindful of gradient accumulation memory usage
4. **Architecture Awareness**: Leverage architecture-specific optimizations

### Integration Patterns

1. **Gradual Integration**: Start with basic parameter management, add hooks as needed
2. **Modular Design**: Separate optimization logic from neural network implementation
3. **Error Handling**: Always handle potential errors in parameter operations
4. **Testing**: Test with different architectures and parameter configurations

### Common Patterns

```rust
// Pattern 1: Basic integration
let manager = ParameterManager::new(config);
// Register parameters, set rules, optimize

// Pattern 2: Architecture-aware integration
let optimizer = ArchitectureAwareOptimizer::new(config, strategy);
// Register architecture, apply optimizations

// Pattern 3: Full integration with hooks
let integration = NeuralIntegration::new(config);
// Register hooks, execute passes, accumulate gradients

// Pattern 4: Custom optimizer implementation
struct MyOptimizer<A: Float, D: Dimension> {
    manager: ParameterManager<A, D>,
    // Custom state...
}

impl<A: Float, D: Dimension> ParameterOptimizer<A, D> for MyOptimizer<A, D> {
    // Implement required methods...
}
```

This API provides a comprehensive foundation for integrating optimizers with neural networks, supporting various architectures and optimization strategies while maintaining flexibility and performance.