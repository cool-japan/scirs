# Parallel Optimization Module

The parallel module provides utilities for parallel computation in optimization algorithms.

## Features

- **Parallel Function Evaluation**: Batch evaluation of objective functions across multiple threads
- **Parallel Gradient Computation**: Concurrent finite difference approximations of gradients
- **Parallel Hessian Computation**: Efficient parallel computation of Hessian matrices
- **Multi-Start Optimization**: Run multiple optimization instances from different starting points
- **Parallel Line Search**: Concurrent evaluation of multiple line search points

## Usage

### Basic Parallel Evaluation

```rust
use scirs2_optimize::parallel::{parallel_evaluate_batch, ParallelOptions};
use ndarray::{array, ArrayView1};

fn objective(x: &ArrayView1<f64>) -> f64 {
    x.iter().map(|&xi| xi.powi(2)).sum()
}

let points = vec![
    array![1.0, 2.0],
    array![3.0, 4.0],
    array![5.0, 6.0],
];

let options = ParallelOptions::default();
let results = parallel_evaluate_batch(&objective, &points, &options);
```

### Parallel Gradient Computation

```rust
use scirs2_optimize::parallel::{parallel_finite_diff_gradient, ParallelOptions};
use ndarray::{array, ArrayView1};

fn objective(x: &ArrayView1<f64>) -> f64 {
    x[0].powi(2) + x[1].powi(2)
}

let x = array![1.0, 2.0];
let options = ParallelOptions::default();
let gradient = parallel_finite_diff_gradient(&objective, x.view(), &options);
```

### Integration with Optimization Algorithms

Many optimization algorithms in scirs2-optimize support parallel computation through the `ParallelOptions` field:

```rust
use scirs2_optimize::{differential_evolution, DifferentialEvolutionOptions};
use scirs2_optimize::parallel::ParallelOptions;
use ndarray::ArrayView1;

fn expensive_function(x: &ArrayView1<f64>) -> f64 {
    // Simulate expensive computation
    std::thread::sleep(std::time::Duration::from_millis(10));
    x[0].powi(2) + x[1].powi(2)
}

let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];

let mut options = DifferentialEvolutionOptions::default();
options.parallel = Some(ParallelOptions {
    num_workers: None, // Use all available cores
    min_parallel_size: 4,
    chunk_size: 1,
    parallel_evaluations: true,
    parallel_gradient: true,
});

let result = differential_evolution(
    expensive_function,
    bounds,
    Some(options),
    None,
).unwrap();
```

## Configuration

The `ParallelOptions` struct provides fine-grained control over parallel execution:

- `num_workers`: Number of worker threads (None = use rayon default)
- `min_parallel_size`: Minimum batch size for parallel execution
- `chunk_size`: Number of items per parallel chunk
- `parallel_evaluations`: Enable parallel function evaluations
- `parallel_gradient`: Enable parallel gradient computations

## Performance Considerations

- Parallel execution is beneficial for expensive objective functions
- For simple functions, the overhead may outweigh the benefits
- The `min_parallel_size` parameter helps avoid excessive parallelization for small problems
- Consider the trade-off between parallelization overhead and computation time

## Thread Safety

All functions in this module require the objective function to implement the `Sync` trait, ensuring thread-safe execution. The objective function must not have mutable state or use proper synchronization if needed.

## Future Enhancements

- GPU acceleration for suitable algorithms
- Asynchronous parallel optimization for varying evaluation times
- Better load balancing for heterogeneous workloads
- Support for distributed computing across multiple machines