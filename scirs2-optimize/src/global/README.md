# Global Optimization Module

This module provides various global optimization algorithms for finding the global minimum of multivariate functions, especially those with multiple local minima.

## Implemented Algorithms

### 1. Differential Evolution
- A stochastic, population-based optimization algorithm
- Does not require gradient information
- Good for functions with many local minima
- Supports various mutation strategies (best1bin, rand1bin, etc.)

### 2. Basin-hopping
- Combines random perturbations with local minimization
- Uses a Metropolis criterion for accepting new points
- Effective for complex landscapes with multiple basins

### 3. Dual Annealing
- Combines classical simulated annealing with fast simulated annealing
- Uses a visiting distribution for efficient exploration
- Includes both global and local search phases

### 4. Particle Swarm Optimization
- Population-based algorithm inspired by swarm behavior
- Each particle moves influenced by personal and global best positions
- Good for continuous optimization problems
- Supports adaptive inertia weight

### 5. Simulated Annealing
- Probabilistic optimization inspired by metallurgy
- Accepts worse solutions with decreasing probability
- Multiple cooling schedules available
- Effective for discrete and combinatorial problems

## Features

- All algorithms support bounds constraints
- Customizable parameters for each algorithm
- Random seed support for reproducibility
- Integration with local optimization methods for refinement
- Comprehensive test coverage

## Usage Example

```rust
use scirs2_optimize::global::{differential_evolution, DifferentialEvolutionOptions};
use ndarray::ArrayView1;

// Define the objective function
let rosenbrock = |x: &ArrayView1<f64>| {
    let x0 = x[0];
    let x1 = x[1];
    (1.0 - x0).powi(2) + 100.0 * (x1 - x0.powi(2)).powi(2)
};

// Define bounds
let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];

// Configure options
let options = DifferentialEvolutionOptions {
    maxiter: 100,
    popsize: 15,
    seed: Some(42),
    ..Default::default()
};

// Run optimization
let result = differential_evolution(rosenbrock, bounds, Some(options), None).unwrap();
println!("Solution: {:?}", result.x);
println!("Function value: {}", result.fun);
```

## Algorithm Selection Guide

- **Differential Evolution**: Best for problems with many dimensions or when derivatives are expensive/unavailable
- **Basin-hopping**: Good for problems with clear basin structure, especially when combined with efficient local optimizers
- **Dual Annealing**: Effective for problems with very rough landscapes or when high precision is needed
- **Particle Swarm**: Suitable for smooth continuous problems with good convergence properties
- **Simulated Annealing**: Works well for discrete, combinatorial, or highly multimodal problems

## References

1. Storn, R., Price, K. (1997). "Differential Evolution – A Simple and Efficient Heuristic for Global Optimization over Continuous Spaces"
2. Wales, D. J., Doye, J. P. K. (1997). "Global Optimization by Basin-Hopping and the Lowest Energy Structures of Lennard-Jones Clusters"
3. Xiang, Y., Gubian, S., Suomela, B., Hoeng, J. (2013). "Generalized Simulated Annealing for Global Optimization: The GenSA Package"