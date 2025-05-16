# SciRS2 Core

[![crates.io](https://img.shields.io/crates/v/scirs2-core.svg)](https://crates.io/crates/scirs2-core)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](../LICENSE)
[![Documentation](https://img.shields.io/docsrs/scirs2-core)](https://docs.rs/scirs2-core)

Core utilities and common functionality for the SciRS2 library. This crate provides the foundation for the entire SciRS2 ecosystem. All modules in the SciRS2 project should leverage this core module to ensure consistency and reduce duplication.

## Features

### Core Features

- **Error Handling**: Comprehensive error system with context, location tracking, and error chaining
- **Configuration System**: Global and thread-local configuration with environment variable integration
- **Numeric Traits**: Generic numeric traits for unified handling of different numeric types
- **Validation**: Utilities for validating numerical operations and data (via `validation` feature)
- **I/O Utilities**: Common I/O operations with proper error handling
- **Constants**: Mathematical and physical constants
- **Utility Functions**: Comprehensive set of utility functions for common operations

### Performance Optimizations

- **Caching**: Memoization with TTL (Time-To-Live) support (via `cache` feature)
- **SIMD Acceleration**: CPU vector instructions for faster array operations (via `simd` feature)
- **Parallel Processing**: Multi-core support for improved performance (via `parallel` feature)
- **GPU Acceleration**: Support for GPU computation via CUDA, WebGPU, Metal (via `gpu` feature)
- **Memory Management**: Efficient memory usage for large-scale computations (via `memory_management` feature)
- **Memory-Efficient Operations**: Chunked processing, lazy evaluation, and out-of-core arrays (via `memory_efficient` feature)
- **Scientific Arrays**: Masked arrays and record arrays for scientific computing (via `array` feature)

### Development Support

- **Logging**: Structured logging for scientific computing (via `logging` feature)
- **Profiling**: Function-level timing instrumentation and memory tracking (via `profiling` feature)
- **Random Numbers**: Consistent interface for random sampling (via `random` feature)
- **Type Conversions**: Safe numeric and complex number conversions (via `types` feature)

## Documentation

- [Core Module Usage Guidelines](../docs/core_module_usage.md): Complete guide for using scirs2-core across modules
- [Error Handling Best Practices](docs/error_handling.md): Best practices for error handling

## Usage

Add the following to your `Cargo.toml`, including only the features you need:

```toml
[dependencies]
scirs2-core = { version = "0.1.0-alpha.3", features = ["validation", "simd", "parallel", "cache"] }
```

Basic usage examples:

```rust
// Array operations
use scirs2_core::utils::{linspace, arange, normalize, pad_array, maximum, minimum};
// Validation functions
use scirs2_core::validation::{check_positive, check_probability, check_shape};
// Configuration 
use scirs2_core::config::{Config, set_global_config};
// SIMD operations
use scirs2_core::simd::{simd_add, simd_multiply};
// Import ndarray for examples
use ndarray::array;

// Set global configuration
let mut config = Config::default();
config.set_precision(1e-10);
set_global_config(config);

// Create arrays
let x = linspace(0.0, 1.0, 100);
let y = arange(0.0, 5.0, 1.0);

// Normalize a vector to unit energy
let signal = vec![1.0, 2.0, 3.0, 4.0];
let normalized = normalize(&signal, "energy").unwrap();

// Pad an array with zeros
let arr = array![1.0, 2.0, 3.0];
let padded = pad_array(&arr, &[(1, 2)], "constant", Some(0.0)).unwrap();

// Validate inputs
let result = check_positive(0.5, "alpha").unwrap(); // Returns 0.5
let probability = check_probability(0.3, "p").unwrap(); // Returns 0.3

// Element-wise operations
let a = array![[1.0, 2.0], [3.0, 4.0]];
let b = array![[4.0, 3.0], [2.0, 1.0]];
let max_ab = maximum(&a, &b);  // [[4.0, 3.0], [3.0, 4.0]]
```

## Feature Flags

The core module uses feature flags to enable optional functionality:

- `validation`: Enable validation utilities (recommended for all modules)
- `simd`: Enable SIMD acceleration (requires the `wide` crate)
- `parallel`: Enable parallel processing (requires `rayon` and `ndarray/rayon`)
- `cache`: Enable caching and memoization functionality (requires `cached` crate)
- `logging`: Enable structured logging and diagnostics
- `gpu`: Enable GPU acceleration abstractions
- `cuda`: Enable CUDA-specific GPU acceleration (requires `gpu` feature)
- `memory_management`: Enable advanced memory management tools
- `memory_efficient`: Enable memory-efficient operations (chunking, lazy evaluation, out-of-core processing)
- `array`: Enable scientific array types (MaskedArray, RecordArray)
- `memory_metrics`: Enable detailed memory usage tracking and analysis
- `memory_visualization`: Enable memory usage visualization capabilities
- `memory_call_stack`: Enable call stack tracking for memory operations
- `profiling`: Enable performance profiling tools
- `random`: Enable random number generation utilities
- `types`: Enable type conversion utilities
- `ufuncs`: Enable universal functions for array operations
- `linalg`: Enable linear algebra with BLAS/LAPACK bindings
- `all`: Enable all features except backend-specific ones

Each module should enable only the features it requires:

```toml
# For modules performing numerical computations
scirs2-core = { version = "0.1.0-alpha.3", features = ["validation", "simd"] }

# For modules with parallel operations and caching
scirs2-core = { version = "0.1.0-alpha.3", features = ["validation", "parallel", "cache"] }

# For AI/ML modules that need GPU acceleration
scirs2-core = { version = "0.1.0-alpha.3", features = ["validation", "gpu", "memory_management", "random"] }

# For development and testing
scirs2-core = { version = "0.1.0-alpha.3", features = ["validation", "logging", "profiling"] }
```

## Core Module Components

### New Components

#### GPU Acceleration

```rust
use scirs2_core::gpu::{GpuContext, GpuBackend, GpuBuffer};

// Create a GPU context with the default backend
let ctx = GpuContext::new(GpuBackend::default())?;

// Allocate memory on the GPU
let mut buffer = ctx.create_buffer::<f32>(1024);

// Copy data to GPU
let host_data = vec![1.0f32; 1024];
buffer.copy_from_host(&host_data);

// Execute a computation
ctx.execute(|compiler| {
    let kernel = compiler.compile(kernel_code)?;
    kernel.set_buffer(0, &mut buffer);
    kernel.dispatch([1024, 1, 1]);
    Ok(())
})?;
```

#### Memory Management

```rust
use scirs2_core::memory::{ChunkProcessor2D, BufferPool, ZeroCopyView};

// Process large arrays in chunks
let mut processor = ChunkProcessor2D::new(&large_array, (1000, 1000));
processor.process_chunks(|chunk, coords| {
    // Process each chunk...
});

// Reuse memory with buffer pools
let mut pool = BufferPool::<f64>::new();
let mut buffer = pool.acquire_vec(1000);
// Use buffer...
pool.release_vec(buffer);

// Efficient transformations with zero-copy views
let view = ZeroCopyView::new(&array);
let transformed = view.transform(|&x| x * 2.0);
```

#### Memory-Efficient Operations

```rust
use scirs2_core::memory_efficient::{
    chunk_wise_op, chunk_wise_binary_op, chunk_wise_reduce, ChunkingStrategy,
    LazyArray, evaluate, create_disk_array, transpose_view, diagonal_view
};

// Process large arrays in chunks to reduce memory usage
let result = chunk_wise_op(
    &large_array,
    |chunk| chunk.map(|&x| x * x),
    ChunkingStrategy::Auto,
)?;

// Create a lazy array that defers computation until needed
let lazy_array = LazyArray::new(data.clone());
let lazy_result = lazy_array.map(|&x| x * 2.0);
let result = evaluate(&lazy_result)?;

// Store large arrays on disk when they don't fit in RAM
let disk_array = create_disk_array(
    &data,
    path,
    ChunkingStrategy::Fixed(1000),
    false,  // read-only
)?;

// Create memory-efficient views without copying data
let transposed = transpose_view(&data)?;
let diagonal = diagonal_view(&data)?;
```

#### Scientific Arrays

```rust
use scirs2_core::array::{
    MaskedArray, mask_array, masked_equal, masked_invalid,
    RecordArray, Record, FieldValue, record_array_from_arrays
};

// Create a masked array to handle missing/invalid data
let data = Array1::from_vec(vec![1.0, 2.0, f64::NAN, 4.0, 5.0]);
let masked = masked_invalid(&data);  // Automatically masks NaN values

// Apply operations while preserving the mask
let result = &masked * 2.0;  // Masked values remain masked

// Create a record array for heterogeneous data
let names = vec![FieldValue::String("Alice".to_string()), FieldValue::String("Bob".to_string())];
let ages = vec![FieldValue::Int(30), FieldValue::Int(25)];
let record_array = record_array_from_arrays(&["name", "age"], &[names, ages])?;

// Access records and fields
let record = record_array.get_record(0)?;
let name = record.get_field_as_string("name")?;
let age = record.get_field_as_int("age")?;
```

#### Enhanced Memory Metrics, Snapshots, and GPU Memory Tracking

```rust
use scirs2_core::memory::metrics::{
    track_allocation, track_deallocation, generate_memory_report, 
    format_memory_report, MemoryMetricsCollector, TrackedBufferPool
};

// Track memory allocations manually
track_allocation("MyComponent", 1024, 0x1000);
// Do work with the memory
track_deallocation("MyComponent", 1024, 0x1000);

// Automatically track memory with a buffer pool
let mut pool = TrackedBufferPool::<f64>::new("NumericalComputation");
let vec = pool.acquire_vec(1000);
// Use the vector...
pool.release_vec(vec);

// Generate a memory report
let report = generate_memory_report();
println!("Total current memory usage: {}", report.total_current_usage);
println!("Peak memory usage: {}", report.total_peak_usage);

// Print a formatted report
println!("{}", format_memory_report());

// Track memory usage during chunk processing
let mut processor = TrackedChunkProcessor2D::new(
    &large_array,
    (1000, 1000),
    "ArrayProcessing"
);

processor.process_chunks(|chunk, coords| {
    // Process each chunk...
    println!("Processing chunk at {:?}", coords);
    
    // Get memory usage after processing this chunk
    let report = generate_memory_report();
    println!("Current memory: {}", format_bytes(report.total_current_usage));
});

// Track GPU memory allocations
use scirs2_core::gpu::GpuBackend;
use scirs2_core::memory::metrics::{TrackedGpuContext, setup_gpu_memory_tracking};

// Set up GPU memory tracking hooks
setup_gpu_memory_tracking();

// Create a tracked GPU context
let context = TrackedGpuContext::with_backend(GpuBackend::Cpu, "GpuOperations").unwrap();

// Create buffers that are automatically tracked
let buffer = context.create_buffer::<f32>(1000);
let data_buffer = context.create_buffer_from_slice(&[1.0f32, 2.0, 3.0]);

// All allocations and deallocations are automatically tracked
let report = generate_memory_report();
println!("GPU memory usage: {}", format_bytes(report.total_current_usage));

// Memory snapshots and leak detection
use scirs2_core::memory::metrics::{
    take_snapshot, compare_snapshots, save_snapshots, load_snapshots
};

// Take snapshots at different points in time
let snapshot1 = take_snapshot("baseline", "Initial memory state");

// ... perform operations that might leak memory ...

// Take another snapshot
let snapshot2 = take_snapshot("after_operations", "After memory-intensive operations");

// Compare snapshots to detect memory leaks
let diff = compare_snapshots("baseline", "after_operations").unwrap();
println!("{}", diff.format());

// Check if there are potential memory leaks
if diff.has_potential_leaks() {
    println!("Potential memory leaks detected in components:");
    for component in diff.get_potential_leak_components() {
        println!("  - {}", component);
    }
}

// Save snapshots to disk for later analysis
save_snapshots("/path/to/snapshot/directory").unwrap();
```

#### Logging and Progress Tracking

```rust
use scirs2_core::logging::{Logger, LogLevel, ProgressTracker};

// Create a logger for a module
let logger = Logger::new("matrix_ops")
    .with_field("precision", "double");

// Log at different levels
logger.info("Starting matrix multiplication");
logger.debug("Using algorithm: Standard");

// Track progress for long operations
let mut progress = ProgressTracker::new("Processing", 1000);
for i in 0..1000 {
    // Do work...
    progress.update(i + 1);
}
progress.complete();
```

#### Profiling

```rust
use scirs2_core::profiling::{Profiler, Timer};

// Start the global profiler
Profiler::global().lock().unwrap().start();

// Time a block of code
let timer = Timer::start("operation");
// Do work...
timer.stop();

// Time a function with result
let result = Timer::time_function("calculate", || {
    // Calculate...
    42
});

// Print profiling report
Profiler::global().lock().unwrap().print_report();
```

#### Random Number Generation

```rust
use scirs2_core::random::{Random, DistributionExt};
use rand_distr::Normal;

// Create a random number generator
let mut rng = Random::default();

// Generate values and arrays
let value = rng.random_range(0.0, 1.0);
let normal = Normal::new(0.0, 1.0).unwrap();
let samples = rng.sample_vec(normal, 100);
let random_array = normal.random_array(&mut rng, [10, 10]);
```

#### Type Conversions

```rust
use scirs2_core::types::{NumericConversion, ComplexExt};
use num_complex::Complex64;

// Convert with error handling
let float_value: f64 = 123.45;
let int_result: Result<i32, _> = float_value.to_numeric();

// Safe conversions for out-of-range values
let large_value: f64 = 1e20;
let clamped: i32 = large_value.to_numeric_clamped();

// Complex number operations
let z1 = Complex64::new(3.0, 4.0);
let mag = z1.magnitude();
let z_norm = z1.normalize();
```

### Existing Components

### Validation Utilities

For validating various types of inputs:

```rust
use scirs2_core::validation::{
    check_probability,        // Check if value is in [0,1]
    check_probabilities,      // Check if all values in array are in [0,1]  
    check_probabilities_sum_to_one,  // Check if probabilities sum to 1
    check_positive,           // Check if value is positive
    check_non_negative,       // Check if value is non-negative
    check_in_bounds,          // Check if value is in a range
    check_finite,             // Check if value is finite
    check_array_finite,       // Check if all array values are finite
    check_same_shape,         // Check if arrays have same shape
    check_shape,              // Check if array has expected shape
    check_square,             // Check if matrix is square
    check_1d,                 // Check if array is 1D
    check_2d,                 // Check if array is 2D
};
```

### Utility Functions

Common utility functions for various operations:

```rust
use scirs2_core::utils::{
    // Array comparison
    is_close,                 // Compare floats with tolerance
    points_equal,             // Compare points (slices) with tolerance
    arrays_equal,             // Compare arrays with tolerance
    
    // Array generation and manipulation
    linspace,                 // Create linearly spaced array
    logspace,                 // Create logarithmically spaced array
    arange,                   // Create range with step size
    fill_diagonal,            // Fill diagonal of matrix
    pad_array,                // Pad array with various modes
    get_window,               // Generate window functions
    
    // Element-wise operations
    maximum,                  // Element-wise maximum
    minimum,                  // Element-wise minimum
    
    // Vector operations
    normalize,                // Normalize vector (energy, peak, sum, max)
    
    // Numerical calculus
    differentiate,            // Differentiate function
    integrate,                // Integrate function
    
    // General utilities
    prod,                     // Product of elements
    all,                      // Check if all elements satisfy predicate
    any,                      // Check if any elements satisfy predicate
};
```

### SIMD Operations

Vectorized operations for improved performance:

```rust
use scirs2_core::simd::{
    simd_add,                 // Add arrays using SIMD
    simd_subtract,            // Subtract arrays using SIMD
    simd_multiply,            // Multiply arrays using SIMD
    simd_divide,              // Divide arrays using SIMD
    simd_min,                 // Element-wise minimum using SIMD
    simd_max,                 // Element-wise maximum using SIMD
};
```

### Caching and Memoization

Utilities for caching computation results:

```rust
use scirs2_core::cache::{
    CacheBuilder,             // Builder for cache configuration
    TTLSizedCache,            // Time-to-live cache with size limit
};
```

## Error Handling

All modules should properly propagate core errors:

```rust
use thiserror::Error;
use scirs2_core::error::CoreError;

#[derive(Debug, Error)]
pub enum ModuleError {
    // Module-specific errors
    #[error("IO error: {0}")]
    IOError(String),
    
    // Propagate core errors
    #[error("{0}")]
    CoreError(#[from] CoreError),
}
```

## Advanced Usage Examples

### Error Handling with Context

```rust
use scirs2_core::{CoreError, ErrorContext, CoreResult, value_err_loc};

fn calculate_value(x: f64) -> CoreResult<f64> {
    if x < 0.0 {
        return Err(value_err_loc!("Input must be non-negative, got {}", x));
    }
    
    Ok(x.sqrt())
}
```

### Caching Expensive Operations

```rust
use scirs2_core::cache::{CacheBuilder, TTLSizedCache};
use std::cell::RefCell;

struct DataLoader {
    cache: RefCell<TTLSizedCache<String, Vec<f64>>>,
}

impl DataLoader {
    pub fn new() -> Self {
        let cache = RefCell::new(
            CacheBuilder::new()
                .with_size(100)
                .with_ttl(3600) // 1 hour TTL
                .build_sized_cache()
        );
        
        Self { cache }
    }
    
    pub fn load_data(&self, key: &str) -> Vec<f64> {
        // Check cache first
        if let Some(data) = self.cache.borrow().get(&key.to_string()) {
            return data.clone();
        }
        
        // Expensive data loading operation
        let data = vec![1.0, 2.0, 3.0]; // Placeholder
        
        // Cache the result
        self.cache.borrow_mut().insert(key.to_string(), data.clone());
        
        data
    }
}
```

## Current Status

The core module now provides a comprehensive foundation for the entire SciRS2 ecosystem with:

- **Complete ndarray Extensions**: Advanced indexing, statistical operations, and array manipulation similar to NumPy
- **Array Protocol Implementation**: Extensible protocol for interoperability between different array implementations
- **GPU Acceleration**: Backend abstraction layer supporting CUDA, WebGPU, Metal, and OpenCL
- **Memory Management**: Advanced memory management including memory mapping, metrics, and adaptive chunking
- **Memory Efficiency**: Zero-copy transformations, buffer pools, and chunk-based processing
- **Profiling and Diagnostics**: Function-level timing, memory tracking, and performance reporting
- **Robust Testing**: Comprehensive test suite with all basic functionality passing

Future work will focus on:
- Enhancing parallel processing with better load balancing and nested parallelism
- Adding support for distributed computing across multiple nodes
- Improving GPU acceleration with more specialized kernels and tensor core support
- Extending memory management with cross-device support and out-of-core processing

## Known Issues

The array protocol implementation is currently in active development and has known test failures in the following areas:

- Distributed arrays: `test_distributed_ndarray_creation`, `test_distributed_ndarray_to_array`
- Custom array types: `example_custom_array_type`, `example_distributed_array`
- Gradient computation: `test_gradient_computation_add`, `test_gradient_computation_multiply`, `test_sgd_optimizer`
- Mixed precision: `test_mixed_precision_array`
- Array operations: `test_operations_with_ndarray`
- Serialization: `test_model_serializer`, `test_save_load_checkpoint`
- Training: `test_mse_loss`

These failures are expected as part of the ongoing implementation work and will be addressed in future releases.

## Contributing

See the [CONTRIBUTING.md](../CONTRIBUTING.md) file for contribution guidelines.

## License

This project is dual-licensed under:

- [MIT License](../LICENSE-MIT)
- [Apache License Version 2.0](../LICENSE-APACHE)

You can choose to use either license. See the [LICENSE](../LICENSE) file for details.
