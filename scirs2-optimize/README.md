# SciRS2 Optimization Module

[![crates.io](https://img.shields.io/crates/v/scirs2-optimize.svg)](https://crates.io/crates/scirs2-optimize)
[[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)]](../LICENSE)
[![Documentation](https://img.shields.io/docsrs/scirs2-optimize)](https://docs.rs/scirs2-optimize)

`scirs2-optimize` is a production-ready optimization library providing comprehensive algorithms for unconstrained and constrained optimization, least-squares problems, root finding, and global optimization. It provides a high-performance Rust implementation of SciPy's optimization functionality with an ergonomic API, advanced features, and excellent performance.

## Features

### 🚀 Core Optimization Methods

**Unconstrained Optimization**
- Nelder-Mead simplex algorithm with adaptive parameters
- BFGS and L-BFGS quasi-Newton methods
- Powell's direction set method with line search
- Conjugate Gradient with Polak-Ribière and Fletcher-Reeves variants
- Full bounds support for all methods

**Constrained Optimization**
- SLSQP (Sequential Least Squares Programming)
- Trust Region Constrained algorithm
- Augmented Lagrangian methods
- Advanced constraint handling

**Least Squares Optimization**
- Levenberg-Marquardt with adaptive damping
- Trust Region Reflective algorithm
- Robust variants: Huber, Bisquare, Cauchy loss functions
- Weighted, bounded, separable, and total least squares

**Root Finding**
- Hybrid methods (modified Powell)
- Broyden's methods (Good and Bad variants)
- Anderson acceleration for iterative methods
- Krylov subspace methods (GMRES)

### 🌍 Global Optimization

**Metaheuristic Algorithms**
- Differential Evolution with adaptive strategies
- Particle Swarm Optimization
- Simulated Annealing with adaptive cooling
- Basin-hopping with local search
- Dual Annealing combining fast and classical annealing

**Bayesian Optimization**
- Gaussian Process surrogate models
- Multiple acquisition functions (EI, LCB, PI)
- Automatic hyperparameter tuning
- Multi-start and clustering strategies

**Multi-objective Optimization**
- NSGA-II for bi-objective problems
- NSGA-III for many-objective problems
- Scalarization methods (weighted sum, Tchebycheff, ε-constraint)

### ⚡ Performance & Advanced Features

**High Performance Computing**
- Parallel evaluation with configurable worker threads
- SIMD-accelerated operations
- Memory-efficient algorithms for large-scale problems
- JIT compilation for performance-critical functions

**Automatic Differentiation**
- Forward-mode AD for gradient computation
- Reverse-mode AD for high-dimensional problems
- Sparse numerical differentiation

**Stochastic Optimization**
- SGD variants with momentum and Nesterov acceleration
- Adam, AdamW, RMSprop optimizers
- Mini-batch processing for large datasets
- Adaptive learning rate schedules

**Specialized Methods**
- Async optimization for slow function evaluations
- Sparse matrix optimization
- Multi-start strategies with clustering

## Installation

Add the following to your `Cargo.toml`:

```toml
[dependencies]
scirs2-optimize = "0.1.0-alpha.6"
```

For advanced features, enable optional feature flags:

```toml
[dependencies]
scirs2-optimize = { version = "0.1.0-alpha.6", features = ["async"] }
```

## Quick Start

### Basic Unconstrained Optimization

```rust
use scirs2_optimize::prelude::*;

// Minimize the Rosenbrock function
fn rosenbrock(x: &[f64]) -> f64 {
    let (a, b) = (1.0, 100.0);
    (a - x[0]).powi(2) + b * (x[1] - x[0].powi(2)).powi(2)
}

fn main() -> Result<(), OptimizeError> {
    let result = minimize(rosenbrock, &[0.0, 0.0], UnconstrainedMethod::BFGS, None)?;
    println!("Minimum at: {:?} with value: {:.6}", result.x, result.fun);
    Ok(())
}
```

### Global Optimization

```rust
use scirs2_optimize::prelude::*;

fn main() -> Result<(), OptimizeError> {
    // Find global minimum using Differential Evolution
    let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
    let result = differential_evolution(rosenbrock, &bounds, None)?;
    println!("Global minimum: {:?}", result.x);
    Ok(())
}
```

### Robust Least Squares

```rust
use scirs2_optimize::prelude::*;
use ndarray::Array1;

fn residual(params: &[f64], data: &[f64]) -> Array1<f64> {
    // Linear regression residuals
    let n = data.len() / 2;
    let (x_vals, y_vals) = data.split_at(n);
    
    Array1::from_iter((0..n).map(|i| 
        y_vals[i] - (params[0] + params[1] * x_vals[i])
    ))
}

fn main() -> Result<(), OptimizeError> {
    let data = vec![0., 1., 2., 3., 4., 0.1, 0.9, 2.1, 2.9, 10.0]; // with outlier
    let result = robust_least_squares(
        residual, &[0.0, 0.0], HuberLoss::new(1.0), None, &data, None
    )?;
    println!("Robust fit: intercept={:.3}, slope={:.3}", result.x[0], result.x[1]);
    Ok(())
}
```

### Bayesian Optimization

```rust
use scirs2_optimize::prelude::*;

fn main() -> Result<(), OptimizeError> {
    let space = Space::new(vec![
        Parameter::Real { name: "x".to_string(), low: -5.0, high: 5.0 },
        Parameter::Real { name: "y".to_string(), low: -5.0, high: 5.0 },
    ]);
    
    let result = bayesian_optimization(rosenbrock, &space, None)?;
    println!("Bayesian optimum: {:?}", result.x);
    Ok(())
}
```

## Why Choose scirs2-optimize?

**🔒 Production Ready**
- Stable API with comprehensive error handling
- Extensive test coverage and numerical validation
- Memory-safe implementation with zero-cost abstractions

**⚡ High Performance** 
- SIMD-accelerated operations where applicable
- Parallel evaluation support
- Memory-efficient algorithms for large-scale problems
- JIT compilation for critical performance paths

**🧠 Intelligent Defaults**
- Robust numerical stability safeguards
- Adaptive parameters that work across problem types
- Automatic algorithm selection helpers

**🔧 Comprehensive Toolkit**
- Complete SciPy optimize API coverage
- Advanced methods beyond SciPy (Bayesian optimization, multi-objective)
- Seamless integration with ndarray ecosystem

**📊 Scientific Computing Focus**
- IEEE 754 compliance and careful numerical handling
- Extensive documentation with mathematical background
- Benchmarked against reference implementations

## Algorithm Selection Guide

| Problem Type | Recommended Method | Use Case |
|--------------|-------------------|----------|
| **Smooth unconstrained** | `BFGS`, `L-BFGS` | Fast convergence with gradients |
| **Noisy/non-smooth** | `Nelder-Mead`, `Powell` | Derivative-free robust optimization |
| **Large-scale** | `L-BFGS`, `CG` | Memory-efficient for high dimensions |
| **Global minimum** | `DifferentialEvolution`, `BayesianOptimization` | Avoid local minima |
| **With constraints** | `SLSQP`, `TrustConstr` | Handle complex constraint sets |
| **Least squares** | `LevenbergMarquardt` | Nonlinear curve fitting |
| **With outliers** | `HuberLoss`, `BisquareLoss` | Robust regression |

## Integration & Ecosystem

- **Zero-copy integration** with ndarray and nalgebra
- **Feature flags** for optional dependencies (async, BLAS backends)
- **Workspace compatibility** with other scirs2 modules
- **C API bindings** available for integration with existing codebases

## License

This project is dual-licensed under:

- [MIT License](../LICENSE-MIT)
- [Apache License Version 2.0](../LICENSE-APACHE)

You can choose to use either license. See the [LICENSE](../LICENSE) file for details.
