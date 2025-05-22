# FFT Planning Guide

This guide explains the FFT planning system in `scirs2-fft`, which provides advanced planning capabilities for optimizing Fast Fourier Transform (FFT) operations.

## Overview

The planning system in `scirs2-fft` is designed to optimize performance of FFT operations by caching, reusing, and adaptively selecting the best algorithms for specific situations. It's modeled after FFTW's planning approach but with a more Rust-focused design.

Proper use of planning can lead to significant performance improvements, especially when:
- Performing repeated transforms of the same size
- Processing large datasets
- Working in memory-constrained environments
- Utilizing specific hardware features (SIMD, GPU)

## Basic Concepts

### Plans

An FFT plan is a pre-computed configuration for performing an FFT of a specific size and direction. Plans can encode:
- The best algorithm for the specific transform
- Memory access patterns
- Low-level optimizations
- Hardware-specific features

Creating a plan usually involves some upfront cost but saves time when executing the same transform multiple times.

### Planning Strategies

`scirs2-fft` offers several planning strategies:

1. **AlwaysNew**:
   - Creates a new plan each time
   - Useful when memory is constrained or plans are rarely reused

2. **CacheFirst**:
   - Checks an in-memory cache for an existing plan before creating a new one
   - Good default choice for most applications

3. **SerializedFirst**:
   - Checks for serialized plans saved from previous runs
   - Falls back to in-memory cache, then creates new plans
   - Best for applications that repeatedly use the same sizes across runs

4. **AutoTuned**:
   - Performs benchmarks to find the optimal algorithm for your hardware
   - Best performance but has higher upfront cost

## Using the Planning System

### Basic Usage

The simplest way to use the planning system is through the standard FFT functions, which handle planning automatically:

```rust
use scirs2_fft::{fft, init_global_planner, PlanningConfig, PlanningStrategy};

// Optional: Configure the global planner (otherwise uses defaults)
init_global_planner(PlanningConfig {
    strategy: PlanningStrategy::CacheFirst,
    ..Default::default()
}).unwrap();

// Perform FFT - planning happens automatically
let input = vec![1.0, 2.0, 3.0, 4.0];
let result = fft(&input, None).unwrap();
```

### Manual Planning

For more control, you can create and use plans directly:

```rust
use scirs2_fft::{FftPlanExecutor, PlanBuilder, PlanningStrategy};
use num_complex::Complex64;

// Create a plan
let plan = PlanBuilder::new()
    .shape(&[1024])
    .forward(true)
    .strategy(PlanningStrategy::AutoTuned)
    .build()
    .unwrap();

// Create an executor for the plan
let executor = FftPlanExecutor::new(plan);

// Prepare input and output
let input = vec![Complex64::new(1.0, 0.0); 1024];
let mut output = vec![Complex64::default(); 1024];

// Execute the plan
executor.execute(&input, &mut output).unwrap();
```

### Ahead-of-Time Planning

For applications with predictable FFT sizes, you can pre-compute plans:

```rust
use scirs2_fft::plan_ahead_of_time;

// Pre-compute plans for common sizes
let sizes = [128, 256, 512, 1024, 2048];
plan_ahead_of_time(&sizes, Some("./fft_plans.json")).unwrap();
```

This saves the plans to disk for faster initialization in subsequent runs.

## Planning Configuration

The `PlanningConfig` struct offers fine-grained control over planning:

```rust
use scirs2_fft::{PlanningConfig, PlanningStrategy};
use std::time::Duration;

let config = PlanningConfig {
    // Basic strategy
    strategy: PlanningStrategy::CacheFirst,
    
    // Performance measurement for optimization
    measure_performance: true,
    
    // Path for serialized plans
    serialized_db_path: Some("./fft_plans.json".to_string()),
    
    // Cache size limits
    max_cached_plans: 128,
    
    // How long to keep plans in cache
    max_plan_age: Duration::from_secs(3600), // 1 hour
    
    // Whether to use parallel planning
    parallel_planning: true,
    
    // Auto-tuning configuration is optional
    auto_tune_config: None,
};
```

## Advanced Features

### Plan Builders

The `PlanBuilder` provides a fluent interface for creating customized plans:

```rust
use scirs2_fft::{PlanBuilder, PlanningStrategy};

let plan = PlanBuilder::new()
    .shape(&[1024, 1024])  // 2D transform
    .forward(false)        // inverse FFT
    .strategy(PlanningStrategy::SerializedFirst)
    .measure_performance(true)
    .serialized_db_path("./plans.json")
    .max_cached_plans(256)
    .build()
    .unwrap();
```

### Global Planner Management

The library maintains a global planner that can be accessed and configured:

```rust
use scirs2_fft::{get_global_planner, init_global_planner, PlanningConfig};

// Initialize with custom configuration
init_global_planner(PlanningConfig::default()).unwrap();

// Access the global planner
let planner = get_global_planner();
let mut planner_guard = planner.lock().unwrap();
```

### Different Backends

The planning system supports different FFT backends:

```rust
use scirs2_fft::{PlanBuilder, PlannerBackend};

let plan = PlanBuilder::new()
    .shape(&[1024])
    .backend(PlannerBackend::CUDA)  // Use GPU acceleration 
    .build()
    .unwrap();
```

Available backends include:
- `RustFFT` (default): Pure Rust implementation
- `FFTW`: Compatible with the FFTW C library
- `CUDA`: GPU-accelerated FFT
- `Custom`: For specialized implementations

## Performance Tips

1. **Choose the Right Strategy**:
   - `CacheFirst` is a good default
   - `SerializedFirst` is best for applications run repeatedly
   - `AutoTuned` gives best performance at the cost of initial overhead

2. **Plan Ahead**:
   - If you know which FFT sizes you'll use, create plans ahead of time
   - Use `plan_ahead_of_time` or precompute common sizes at startup

3. **Tune Cache Settings**:
   - Increase `max_cached_plans` for applications using many different sizes
   - Decrease plan age for memory-sensitive applications

4. **Use Appropriate Hardware**:
   - Select specialized backends when appropriate
   - Use the CUDA backend for large transforms when GPU is available

5. **Benchmark Different Approaches**:
   - Use the advanced_planning_strategies example as a template
   - Measure performance with your specific workload

## Common Pitfalls

1. **Excessive Planning**:
   - Creating too many plans wastes memory
   - Creating and discarding plans repeatedly negates performance benefits

2. **Inappropriate Strategy**:
   - Using AutoTuned for one-off transforms has high overhead
   - Using AlwaysNew for repeated transforms misses caching benefits

3. **Serialization Without Verification**:
   - Serialized plans from different hardware may perform poorly
   - The library checks architecture compatibility by default

4. **Large Cache Size**:
   - Very large caches can impact memory usage
   - Consider smaller cache sizes with more frequent eviction for memory-constrained environments

## Advanced Customization

For specialized needs, you can directly work with the `AdvancedFftPlanner` and implement custom planning logic:

```rust
use scirs2_fft::{AdvancedFftPlanner, PlanningConfig};

let mut planner = AdvancedFftPlanner::with_config(PlanningConfig::default());

// Custom planning logic
// ...

// Save any plans
planner.save_plans().unwrap();
```

## Integration with Other Modules

The planning system integrates with:

1. **Auto-tuning**: For adaptive algorithm selection
2. **Backend system**: For different FFT implementations
3. **Worker pools**: For parallel execution
4. **Memory management**: For efficient allocation strategies

## Conclusion

The FFT planning system provides powerful tools for optimizing FFT performance across a wide range of use cases. By selecting appropriate planning strategies and configurations, you can significantly improve the performance of your FFT operations.

For even more advanced usage scenarios, refer to the API documentation and the example files included with the library.