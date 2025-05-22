# Array Protocol and Interoperability

## Overview

The Array Protocol module provides a mechanism for interoperability between different array implementations in SCIRS, inspired by NumPy's `__array_function__` protocol from NEP-18. It enables:

- Seamless integration of specialized array types (GPU, distributed, etc.)
- Just-In-Time compilation for optimized array operations
- Extensibility through third-party array implementations
- Type-based dispatch for array operations

## Key Components

### Core Protocol

- `ArrayProtocol`: Trait that must be implemented by array types for interoperability
- `ArrayFunction`: Represents a function that can be dispatched to different implementations
- `NotImplemented`: Error type returned when an operation is not supported

### Specialized Array Implementations

- `GPUNdarray`: Array implementation for GPU-accelerated computing
- `DistributedNdarray`: Array implementation for distributed computing
- `JITEnabledArray`: Array implementation with Just-In-Time compilation support
- `AutoDevice`: Wrapper for automatic device placement based on array size and operation
- `MixedPrecisionArray`: Wrapper for mixed-precision operations

### Advanced Features

- Automatic device placement based on array size and operation complexity
- Mixed-precision operations with configurable storage and compute precision
- Common array operations (matmul, add, multiply, transpose, etc.)
- CUDA-optimized operations for GPU arrays
- Support for third-party array implementations
- Machine learning operations (activations, convolutions, pooling, normalization)
- Neural network layer implementations (Linear, Conv2D, MaxPool2D, BatchNorm, etc.)
- Neural network model implementation with automatic device selection
- Gradient computation and automatic differentiation
- Optimization algorithms (SGD, Adam)
- Training pipeline with datasets, dataloaders, and metrics tracking
- Distributed training with data parallelism, model parallelism, and pipeline parallelism
- Model serialization and deserialization for checkpointing and deployment
- ONNX export for interoperability with other frameworks

### Extended Functionality

- Support for array function registration and discovery
- Dynamic dispatch based on array types
- Additional traits for specialized functionality (strided access, zero-copy, etc.)
- JIT compilation with multiple backends (LLVM, Cranelift, WebAssembly)

## Usage

```rust
use scirs2_core::array_protocol;
use scirs2_core::array_protocol::{GPUNdarray, GPUConfig, GPUBackend};
use ndarray::Array2;

// Initialize the array protocol system
array_protocol::init();

// Create a regular array
let array = Array2::<f64>::ones((10, 5));

// Create a GPU array
let config = GPUConfig {
    backend: GPUBackend::CUDA,
    device_id: 0,
    async_ops: true,
    mixed_precision: false,
    memory_fraction: 0.9,
};
let gpu_array = GPUNdarray::new(array.clone(), config);

// Define an array function
array_protocol::array_function!(
    fn matrix_sum(matrix: &Array2<f64>) -> f64 {
        matrix.sum()
    },
    "my_library::matrix_sum"
);

// Register and use the function
let sum_func = matrix_sum.register();
let result = sum_func(&array);
```

## Documentation

For more detailed information, see:

- [Array Protocol Guide](../../docs/array_protocol_guide.md): Comprehensive guide to using the array protocol
- Examples in the `examples` submodule in `mod.rs`
- Module documentation in the source code

## Implementation Status

This module is partially implemented with active development on the following components:

- GPU arrays with multiple backends (CUDA, ROCm, Metal, WebGPU, OpenCL)
  - CUDA backend is fully implemented
  - ROCm, Metal, WebGPU, and OpenCL support is implemented as feature flags
  - Basic operations (add, matmul, transpose, reshape) are implemented
  
- Distributed arrays with different distribution strategies
  - Basic infrastructure is implemented
  - Row-wise, column-wise, and block distribution strategies are supported
  - Map-reduce operations are implemented
  - Operations (add, subtract, multiply, matmul, transpose, reshape) are implemented
  
- Mixed-precision operations
  - Support for different storage and compute precisions
  - Automatic precision selection based on array size and operation
  - Precision conversion between f32 and f64
  - Mixed-precision operations for matmul, element-wise operations, and reductions
  
- JIT compilation with multiple backends
  - Basic infrastructure is implemented
  - Function expression compilation is supported
  - Support for LLVM backend is in development
  
- Core array protocol infrastructure
  - ArrayProtocol trait for interoperability
  - ArrayFunction for dispatch
  - Dynamic type-based dispatch

The module is currently in alpha status, with ongoing development to improve performance, robustness, and feature completeness. The core functionality is usable for experimental code, but not yet production-ready. The primary focus is on expanding the operation set and optimizing performance for different backend implementations.