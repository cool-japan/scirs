# Bayesian Optimization Implementation Plan

## Overview
This document outlines the plan for implementing a Bayesian optimization module for scirs2-optimize. Bayesian optimization is a global optimization method for expensive black-box functions that uses a surrogate model and acquisition functions to efficiently find global optima.

## Components

### 1. Parameter Space Definition
- Implement a `Space` struct that manages different parameter types
- Support for three parameter types:
  - `Real`: Continuous parameters
  - `Integer`: Integer parameters
  - `Categorical`: Discrete parameters
- Each parameter type will handle transformations to the internal representation
- Implement sampling methods for initial points

### 2. Surrogate Model
- Use existing Gaussian Process implementation from a library (eg. Friedrich or egobox-gp)
- Define a trait `SurrogateModel` with the following methods:
  - `fit(&mut self, x: &[Array1<f64>], y: &[f64])`: Fit model to observed data
  - `predict(&self, x: &Array1<f64>, return_std: bool) -> (f64, Option<f64>)`: Predict mean and std
  - `update(&mut self, x: &Array1<f64>, y: f64)`: Update model with new data
- Implement the trait for Gaussian Process models
- Support for different kernel functions

### 3. Acquisition Functions
- Define a trait `AcquisitionFunction` with methods:
  - `evaluate(&self, x: &Array1<f64>) -> f64`: Compute acquisition value
  - `gradient(&self, x: &Array1<f64>) -> Array1<f64>`: Compute gradient (when available)
- Implement common acquisition functions:
  - `ExpectedImprovement`: Balances exploration and exploitation
  - `LowerConfidenceBound`: Configurable exploration parameter (kappa)
  - `ProbabilityOfImprovement`: More aggressive exploitation
- Allow for custom acquisition functions

### 4. Acquisition Optimization
- Implement methods to optimize acquisition functions:
  - Use L-BFGS-B for continuous spaces from our existing implementation
  - Use sampling-based optimization for mixed/categorical spaces
- Support for multiple restarts to avoid local optima
- Parallel optimization via the existing parallel module

### 5. Optimizer Implementation
- Create a `BayesianOptimization` struct containing:
  - Search space
  - Surrogate model
  - Acquisition function
  - Observations (x, y pairs)
  - Optimization history
- Implement an ask/tell interface:
  - `ask(&self) -> Array1<f64>`: Get next point to evaluate
  - `tell(&mut self, x: Array1<f64>, y: f64)`: Update with observation
- Implement convenience methods:
  - `optimize(&mut self, f: F, n_calls: usize) -> OptimizeResult<f64>`
- Support for callbacks for monitoring and early stopping

## Integration

### Public API
```rust
// Import necessary modules
use scirs2_optimize::bayesian::{BayesianOptimization, AcquisitionFunction, Space, Parameter};
use scirs2_optimize::parallel::ParallelOptions;

// Define objective function
fn objective(x: &ArrayView1<f64>) -> f64 {
    x[0].powi(2) + x[1].powi(2) - 0.5 * x[0] * x[1]
}

// Define search space
let space = Space::new()
    .add("x1", Parameter::Real(-5.0, 5.0))
    .add("x2", Parameter::Real(-5.0, 5.0));

// Create optimizer
let mut optimizer = BayesianOptimization::new(
    space,
    None, // Use default model
    None, // Use default acquisition function (EI)
    None, // Use default options
);

// Run optimization
let result = optimizer.optimize(objective, 25);

// Access results
println!("Best parameters: {:?}", result.x);
println!("Best value: {}", result.fun);
```

### Integration with Existing Module
- Create a new module `src/global/bayesian.rs`
- Update `src/global/mod.rs` to include the new module
- Implement the public function:
```rust
pub fn bayesian_optimization<F>(
    func: F,
    bounds: Bounds,
    n_calls: usize,
    options: Option<BayesianOptimizationOptions>,
) -> Result<OptimizeResult<f64>, OptimizeError>
where
    F: Fn(&ArrayView1<f64>) -> f64
```
- Connect with the `parallel` module for parallel acquisition function optimization

## Dependencies
- **Friedrich** or **egobox-gp** for Gaussian Process implementation
- **ndarray** for array manipulations
- **rayon** for parallel computation
- Our existing optimization code for acquisition function optimization

## Roadmap

1. **Phase 1: Core Implementation**
   - Implement parameter space definitions
   - Integrate Gaussian Process implementation
   - Implement basic acquisition functions
   - Create optimizer structure with ask/tell interface

2. **Phase 2: Advanced Features**
   - Parallel acquisition optimization
   - Multiple acquisition function strategies
   - Support for categorical parameters
   - Implement callbacks

3. **Phase 3: Documentation and Integration**
   - Comprehensive documentation with examples
   - Integration with the rest of the crate
   - Benchmarks and comparison with SciPy implementations
   - Write unit and integration tests