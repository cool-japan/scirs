# SCIRS Array Protocol Guide

This guide explains how to use the SCIRS Array Protocol for interoperability between different array implementations. The protocol is inspired by NumPy's `__array_function__` protocol from NEP-18 and allows third-party array libraries to integrate seamlessly with SCIRS.

## Overview

The SCIRS Array Protocol provides a mechanism for dispatching operations to the appropriate implementation based on the array types involved. This enables:

- Support for specialized array types (GPU arrays, distributed arrays, etc.)
- Integration with Just-In-Time (JIT) compilation systems
- Zero-copy operations between different array implementations
- Extensibility through third-party array libraries

## Using Built-in Array Types

### Initialization

Before using the array protocol, initialize the system:

```rust
use scirs2_core::array_protocol;

// Initialize array protocol for the current thread
array_protocol::init();
```

### Creating Specialized Arrays

#### GPU Arrays

```rust
use scirs2_core::array_protocol::{GPUNdarray, GPUConfig, GPUBackend};
use ndarray::Array2;

// Create a regular array
let array = Array2::<f64>::ones((10, 5));

// Create a GPU array configuration
let config = GPUConfig {
    backend: GPUBackend::CUDA,  // Or OpenCL, ROCm, Metal, etc.
    device_id: 0,
    async_ops: true,
    mixed_precision: false,
    memory_fraction: 0.9,
};

// Create a GPU array
let gpu_array = GPUNdarray::new(array.clone(), config);

// Transfer back to CPU when needed
let cpu_array = gpu_array.to_cpu().unwrap();
```

#### Distributed Arrays

```rust
use scirs2_core::array_protocol::{DistributedNdarray, DistributedConfig, DistributionStrategy, DistributedBackend};
use ndarray::Array2;

// Create a regular array
let array = Array2::<f64>::ones((10, 5));

// Create a distributed array configuration
let config = DistributedConfig {
    chunks: 3,  // Number of chunks to split the array into
    balance: true,
    strategy: DistributionStrategy::RowWise,  // Or ColumnWise, Blocks
    backend: DistributedBackend::Threaded,  // Or MPI, Ray, Dask, etc.
};

// Create a distributed array
let dist_array = DistributedNdarray::from_array(array.clone(), config);

// Convert back to a regular array when needed
let result = dist_array.to_array().unwrap();
```

#### JIT-Enabled Arrays

```rust
use scirs2_core::array_protocol::{JITEnabledArray, NdarrayWrapper};
use ndarray::Array2;

// Create a regular array
let array = Array2::<f64>::ones((10, 5));
let wrapped = NdarrayWrapper::new(array);

// Create a JIT-enabled array
let jit_array = JITEnabledArray::new(wrapped);

// Compile a function using JIT
let expression = "x + y";
let jit_function = jit_array.compile(expression).unwrap();

// Execute the compiled function
let result = jit_function.execute(&[&jit_array, &jit_array]).unwrap();
```

## Defining Array Functions

You can define functions that work with any array implementation using the `array_function!` macro:

```rust
use scirs2_core::array_protocol;
use ndarray::Array2;

// Define a function using the macro
array_protocol::array_function!(
    fn sum_array(array: &Array2<f64>) -> f64 {
        array.sum()
    },
    "my_library::sum_array"
);

// Register the function
let registered_sum = sum_array.register();

// Use the function
let array = Array2::<f64>::ones((3, 3));
let sum = registered_sum(&array);
assert_eq!(sum, 9.0);
```

## Implementing the Array Protocol for Third-Party Types

To make your custom array type work with the SCIRS Array Protocol, implement the `ArrayProtocol` trait:

```rust
use scirs2_core::array_protocol::{ArrayProtocol, ArrayFunction, NotImplemented};
use std::any::{Any, TypeId};
use std::collections::HashMap;

// Define your custom array type
struct MyCustomArray<T> {
    data: Vec<T>,
    shape: Vec<usize>,
}

impl<T: Clone + 'static> MyCustomArray<T> {
    fn new(data: Vec<T>, shape: Vec<usize>) -> Self {
        Self { data, shape }
    }
    
    fn shape(&self) -> &[usize] {
        &self.shape
    }
}

// Implement ArrayProtocol for your type
impl<T: Clone + Send + Sync + 'static> ArrayProtocol for MyCustomArray<T> {
    fn array_function(
        &self,
        func: &ArrayFunction,
        types: &[TypeId],
        args: &[Box<dyn Any>],
        kwargs: &HashMap<String, Box<dyn Any>>,
    ) -> Result<Box<dyn Any>, NotImplemented> {
        // Implement handlers for specific functions
        match func.name.as_str() {
            "my_library::sum" => {
                // Implement sum for this array type
                // ...
                Ok(Box::new(42.0f64))  // Example return value
            },
            _ => Err(NotImplemented),  // Function not implemented for this type
        }
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// Register your implementation with SCIRS
fn register_my_implementation() {
    use scirs2_core::array_protocol;
    
    // Create and register your implementation
    let implementation = MyCustomArray::<f64>::new(vec![1.0, 2.0, 3.0], vec![3]);
    array_protocol::register_array_implementation(implementation);
}
```

## Advanced Features

### Special Traits

The array protocol provides additional traits for specialized functionality:

- `StridedArray`: For arrays with strided memory layout
- `ZeroCopyArray`: For arrays that support zero-copy operations
- `DifferentiableArray`: For arrays that support automatic differentiation
- `AsyncArray`: For arrays that support asynchronous operations
- `MixedPrecisionSupport`: For arrays that support mixed-precision operations

### Automatic Device Placement

The `auto_device` module provides automatic device placement for arrays based on their size and the operations being performed:

```rust
use scirs2_core::array_protocol::auto_device::{AutoDevice, DeviceType, AutoDeviceConfig, set_auto_device_config};
use ndarray::Array2;

// Configure auto device placement
let config = AutoDeviceConfig {
    gpu_threshold: 1_000_000,            // 1M elements threshold for GPU
    distributed_threshold: 100_000_000,  // 100M elements threshold for distributed
    enable_mixed_precision: true,
    preferred_gpu_backend: GPUBackend::CUDA,
    // ...other configuration options
};
set_auto_device_config(config);

// Create arrays
let a = Array2::<f64>::ones((1000, 1000));  // 1M elements
let b = Array2::<f64>::ones((1000, 1000));  // 1M elements

// Wrap in AutoDevice
let mut auto_a = AutoDevice::new(a);
let mut auto_b = AutoDevice::new(b);

// Perform operation with automatic device selection
let result = array_protocol::auto_device::ops::matmul(&mut auto_a, &mut auto_b).unwrap();
```

### Mixed-Precision Operations

The `mixed_precision` module enables arrays to use different precision levels for storage and computation:

```rust
use scirs2_core::array_protocol::mixed_precision::{MixedPrecisionArray, Precision, MixedPrecisionConfig, set_mixed_precision_config};
use ndarray::Array2;

// Configure mixed precision
let config = MixedPrecisionConfig {
    storage_precision: Precision::Single,
    compute_precision: Precision::Double,
    auto_precision: true,
    // ...other configuration options
};
set_mixed_precision_config(config);

// Create arrays
let a = Array2::<f64>::ones((100, 100));
let b = Array2::<f32>::ones((100, 100)).mapv(|x| x as f32);

// Wrap in MixedPrecisionArray
let mixed_a = MixedPrecisionArray::new(a);
let mixed_b = MixedPrecisionArray::new(b);

// Perform operation with specific precision
let result = array_protocol::mixed_precision::ops::matmul(
    &mixed_a, &mixed_b, Precision::Double).unwrap();
```

### JIT Compilation

The JIT system supports multiple backends:

```rust
use scirs2_core::array_protocol::{JITManager, JITConfig, JITBackend};

// Configure JIT
let config = JITConfig {
    backend: JITBackend::LLVM,  // Or Cranelift, WebAssembly
    optimize_level: 2,
    enable_avx2: true,
    cache_compiled: true,
};

// Get the global JIT manager
let jit_manager = JITManager::global();
let mut jit_manager = jit_manager.write().unwrap();
jit_manager.set_config(config);

// Register custom JIT function factories
// jit_manager.register_factory(Box::new(MyCustomJITFactory));
```

## Machine Learning Support

The array protocol includes support for common machine learning operations:

### Activation Functions

```rust
use scirs2_core::array_protocol::ml_ops::{activation, ActivationFunc};
use ndarray::Array2;

// Create an array
let x = Array2::<f64>::ones((10, 10));
let wrapped_x = NdarrayWrapper::new(x);

// Apply ReLU activation
let result = activation(&wrapped_x, ActivationFunc::ReLU).unwrap();

// Apply Sigmoid activation
let result = activation(&wrapped_x, ActivationFunc::Sigmoid).unwrap();

// Apply Leaky ReLU activation
let result = activation(&wrapped_x, ActivationFunc::LeakyReLU(0.1)).unwrap();
```

### Convolution and Pooling

```rust
use scirs2_core::array_protocol::ml_ops::{conv2d, max_pool2d};
use ndarray::Array4;

// Create a 4D input tensor [batch, height, width, channels]
let input = Array4::<f64>::ones((1, 28, 28, 3));
let wrapped_input = NdarrayWrapper::new(input);

// Create a 4D filters tensor [height, width, in_channels, out_channels]
let filters = Array4::<f64>::ones((3, 3, 3, 16));
let wrapped_filters = NdarrayWrapper::new(filters);

// Perform 2D convolution
let result = conv2d(
    &wrapped_input,
    &wrapped_filters,
    (1, 1),  // stride
    (0, 0),  // padding
).unwrap();

// Perform max pooling
let result = max_pool2d(
    &wrapped_input,
    (2, 2),  // kernel size
    (2, 2),  // stride
    (0, 0),  // padding
).unwrap();
```

## Neural Network Models

The array protocol includes neural network layer and model implementations:

### Creating Layers

```rust
use scirs2_core::array_protocol::neural::{Linear, Conv2D};
use scirs2_core::array_protocol::ml_ops::ActivationFunc;

// Create a linear layer
let linear = Linear::with_shape(
    "fc1",
    128,             // Input features
    64,              // Output features
    true,            // With bias
    Some(ActivationFunc::ReLU),
);

// Create a convolutional layer
let conv = Conv2D::with_shape(
    "conv1",
    3, 3,            // Filter size
    3, 16,           // In/out channels
    (1, 1),          // Stride
    (1, 1),          // Padding
    true,            // With bias
    Some(ActivationFunc::ReLU),
);
```

### Building Models

```rust
use scirs2_core::array_protocol::neural::{Sequential, Linear, Conv2D, MaxPool2D, create_simple_cnn};

// Create a sequential model manually
let mut model = Sequential::new("MyModel", Vec::new());

// Add layers
model.add_layer(Box::new(Conv2D::with_shape(
    "conv1", 3, 3, 3, 16, (1, 1), (1, 1), true, Some(ActivationFunc::ReLU)
)));

model.add_layer(Box::new(MaxPool2D::new(
    "pool1", (2, 2), None, (0, 0)
)));

// Or use a model builder function
let model = create_simple_cnn((28, 28, 3), 10);  // Input shape, num_classes
```

### Running Inference

```rust
use scirs2_core::array_protocol::neural::Sequential;

// Create a model
let model = create_simple_cnn((28, 28, 1), 10);

// Set to evaluation mode
model.eval();

// Run inference
let input = /* ... */;
let output = model.forward(&input).unwrap();
```

## Gradient Computation and Automatic Differentiation

The array protocol includes support for gradient computation and automatic differentiation:

### Creating Gradient Tensors

```rust
use scirs2_core::array_protocol::grad::{GradientTensor, grad_add, grad_multiply, grad_matmul};
use ndarray::Array2;

// Create gradient tensors
let a_array = Array2::<f64>::ones((2, 3));
let b_array = Array2::<f64>::ones((3, 4));

let a = GradientTensor::from_array(a_array, true);  // requires_grad = true
let b = GradientTensor::from_array(b_array, true);

// Perform operations with gradient tracking
let c = grad_matmul(&a, &b).unwrap();
let d = grad_add(&c, &c).unwrap();

// Compute gradients
d.backward().unwrap();

// Access gradients
let a_grad = a.grad().unwrap();
let b_grad = b.grad().unwrap();
```

### Creating Variables for Optimization

```rust
use scirs2_core::array_protocol::grad::{Variable, SGD, Adam, Optimizer};
use ndarray::Array2;

// Create variables
let weight_array = Array2::<f64>::ones((2, 2));
let weight = Variable::new("weight", weight_array);

let bias_array = Array2::<f64>::zeros((2, 2));
let bias = Variable::new("bias", bias_array);

// Create optimizer
let mut optimizer = SGD::new(0.01, Some(0.9));  // lr, momentum
optimizer.add_variable(weight);
optimizer.add_variable(bias);

// Alternatively, use Adam optimizer
let mut optimizer = Adam::new(0.001, Some(0.9), Some(0.999), Some(1e-8));
optimizer.add_variable(weight);
optimizer.add_variable(bias);

// Perform optimization step
optimizer.step().unwrap();

// Zero gradients
optimizer.zero_grad();
```

## Model Training

The array protocol includes utilities for training neural networks:

### Creating Datasets and DataLoaders

```rust
use scirs2_core::array_protocol::training::{InMemoryDataset, DataLoader};
use ndarray::Array2;

// Create input and target arrays
let inputs = Array2::<f64>::ones((100, 10));
let targets = Array2::<f64>::zeros((100, 3));

// Create dataset
let dataset = InMemoryDataset::from_arrays(inputs, targets);

// Create data loader
let train_loader = DataLoader::new(
    Box::new(dataset),
    32,           // batch size
    true,         // shuffle
    Some(42),     // random seed
);
```

### Setting Up Loss Functions

```rust
use scirs2_core::array_protocol::training::{MSELoss, CrossEntropyLoss};

// Create MSE loss
let mse_loss = MSELoss::new(Some("mean"));

// Create cross-entropy loss
let ce_loss = CrossEntropyLoss::new(Some("mean"));
```

### Training a Model

```rust
use scirs2_core::array_protocol::neural::Sequential;
use scirs2_core::array_protocol::grad::{Adam, Optimizer};
use scirs2_core::array_protocol::training::{Trainer, ProgressCallback, DataLoader, CrossEntropyLoss};

// Create model, optimizer, and loss function
let model = /* ... */;
let optimizer = Box::new(Adam::new(0.001, Some(0.9), Some(0.999), Some(1e-8)));
let loss_fn = Box::new(CrossEntropyLoss::new(Some("mean")));

// Create trainer
let mut trainer = Trainer::new(model, optimizer, loss_fn);

// Add progress callback
trainer.add_callback(Box::new(ProgressCallback::new(true)));

// Train the model
let num_epochs = 10;
trainer.train(train_loader, num_epochs, Some(val_loader)).unwrap();
```

## Best Practices

1. **Initialization**: Always call `array_protocol::init()` before using the array protocol.

2. **Function Registration**: Register your array functions with unique identifiers to avoid conflicts.

3. **Fallbacks**: Provide fallback implementations for operations that aren't optimized.

4. **Error Handling**: Handle `NotImplemented` errors gracefully by falling back to more general implementations.

5. **Performance**: Consider providing specialized implementations for common operations to optimize performance.

6. **Memory Management**: Be careful with zero-copy operations to avoid unexpected behavior.

7. **Device Placement**: Use the `AutoDevice` wrapper for automatic device selection based on array size and operation.

8. **Mixed Precision**: Use the `MixedPrecisionArray` wrapper for mixed-precision operations.

9. **Distributed Training**: Use the distributed training utilities for multi-node training.

10. **Model Serialization**: Use the serialization utilities to save and load models.

## Examples

See the `examples` module in the `array_protocol.rs` file for complete usage examples, including:

- Basic usage of different array types
- Interoperability between different array implementations
- Creating custom array types that implement the array protocol
- Defining and using array functions with the protocol

## Implementation Status

The SCIRS Array Protocol is currently in alpha development with the following components implemented:

### Implemented Features

1. **Core Protocol Infrastructure**
   - `ArrayProtocol` trait for interoperability
   - `ArrayFunction` type for function representation
   - Dynamic type-based dispatch mechanism
   - Basic registry system

2. **Specialized Array Types**
   - `NdarrayWrapper`: Adapter for ndarray arrays
   - `GPUNdarray`: Basic GPU array implementation with multiple backend support
   - `DistributedNdarray`: Distributed array implementation with different distribution strategies
   - `JITEnabledArray`: Array implementation with preliminary JIT support
   - `MixedPrecisionArray`: Array wrapper for mixed-precision operations

3. **Operations**
   - Basic array operations (add, subtract, multiply)
   - Matrix multiplication
   - Transpose
   - Reshape
   - Sum and other reductions
   - Element-wise operations

4. **Device Support**
   - CUDA backend for GPU arrays
   - Feature flags for ROCm, Metal, WebGPU, and OpenCL
   - Threaded backend for distributed arrays
   - Feature flags for MPI and other communication backends

5. **Mixed Precision Support**
   - Different storage and compute precision levels
   - Automatic precision selection
   - Support for mixed-precision operations

### Features Under Development

The following features are partially implemented or under active development:

1. **GPU Operations**
   - Full CUDA kernel library
   - Optimized GPU operations
   - Multiple GPU support

2. **Distributed Computing**
   - Advanced distribution strategies
   - Efficient communication patterns
   - Node-level optimizations

3. **JIT Compilation**
   - LLVM backend integration
   - Runtime code generation
   - Operation fusion

4. **Machine Learning Support**
   - Neural network layers
   - Gradient computation
   - Training utilities

### Current Limitations

- Performance optimization is ongoing
- Some operations may fall back to CPU implementation
- Documentation is still being developed
- Test coverage is being expanded
- API stability is not guaranteed