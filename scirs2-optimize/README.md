# SciRS2 Optimization Module

[![crates.io](https://img.shields.io/crates/v/scirs2-optimize.svg)](https://crates.io/crates/scirs2-optimize)
[[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)]](../LICENSE)
[![Documentation](https://img.shields.io/docsrs/scirs2-optimize)](https://docs.rs/scirs2-optimize)

`scirs2-optimize` is a comprehensive optimization library providing algorithms for unconstrained and constrained optimization, least-squares problems, and root finding. It aims to provide a Rust implementation of SciPy's optimization functionality with a similar API.

## Features

The module is divided into several key components:

### Unconstrained Optimization

Algorithms for minimizing scalar functions of one or more variables without constraints:

- Nelder-Mead simplex algorithm
- BFGS (Broyden-Fletcher-Goldfarb-Shanno) algorithm
- Powell's method
- Conjugate Gradient method

### Constrained Optimization

Algorithms for minimizing scalar functions with constraints:

- SLSQP (Sequential Least Squares Programming)
- Trust Region Constrained algorithm

### Least Squares Optimization

Algorithms for solving nonlinear least squares problems:

- Levenberg-Marquardt algorithm
- Trust Region Reflective algorithm

### Root Finding

Algorithms for finding roots of nonlinear functions:

- Hybrid method (modified Powell algorithm)
- Broyden's method (Good and Bad variants)
- Anderson acceleration
- Krylov subspace methods (GMRES)

## Installation

Add the following to your `Cargo.toml`:

```toml
[dependencies]
scirs2-optimize = "0.1.0-alpha.2"
```

To enable optimizations through the core module, add feature flags:

```toml
[dependencies]
scirs2-optimize = { version = "0.1.0-alpha.2", features = ["parallel"] }
```

## Usage Examples

### Unconstrained Optimization

```rust
use ndarray::array;
use scirs2_optimize::unconstrained::{minimize, Method};

// Define a function to minimize (e.g., Rosenbrock function)
fn rosenbrock(x: &[f64]) -> f64 {
    let a = 1.0;
    let b = 100.0;
    let x0 = x[0];
    let x1 = x[1];
    (a - x0).powi(2) + b * (x1 - x0.powi(2)).powi(2)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initial guess
    let x0 = array![0.0, 0.0];
    
    // Minimize using BFGS algorithm
    let result = minimize(rosenbrock, &x0, Method::BFGS, None)?;
    
    println!("Solution: {:?}", result.x);
    println!("Function value at solution: {}", result.fun);
    println!("Number of iterations: {}", result.nit);
    println!("Success: {}", result.success);
    
    Ok(())
}
```

### Constrained Optimization

```rust
use ndarray::array;
use scirs2_optimize::constrained::{minimize_constrained, Method, Constraint};

// Define an objective function
fn objective(x: &[f64]) -> f64 {
    (x[0] - 1.0).powi(2) + (x[1] - 2.5).powi(2)
}

// Define a constraint: x[0] + x[1] <= 3
fn constraint(x: &[f64]) -> f64 {
    3.0 - x[0] - x[1]  // Should be >= 0
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initial point
    let initial_point = array![0.0, 0.0];
    let constraints = vec![Constraint::new(constraint, Constraint::INEQUALITY)];
    
    let result = minimize_constrained(
        objective, 
        &initial_point, 
        &constraints, 
        Method::SLSQP, 
        None
    )?;
    
    println!("Solution: {:?}", result.x);
    println!("Function value at solution: {}", result.fun);
    
    Ok(())
}
```

### Least Squares Optimization

```rust
use ndarray::{array, Array1, Array2};
use scirs2_optimize::least_squares::{least_squares, Method};

// Define residual function
fn residual(x: &[f64], _: &[f64]) -> Array1<f64> {
    array![
        x[0] + 2.0 * x[1] - 2.0,
        x[0] + x[1] - 1.0
    ]
}

// Define Jacobian (optional)
fn jacobian(_: &[f64], _: &[f64]) -> Array2<f64> {
    array![[1.0, 2.0], [1.0, 1.0]]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initial guess
    let x0 = array![0.0, 0.0];
    let data = array![];  // No data needed for this example
    
    // Solve using Levenberg-Marquardt
    let result = least_squares(
        residual,
        &x0,
        Method::LevenbergMarquardt,
        Some(jacobian),
        &data,
        None
    )?;
    
    println!("Solution: {:?}", result.x);
    println!("Function value at solution: {}", result.fun);
    
    Ok(())
}
```

### Root Finding

```rust
use ndarray::{array, Array1, Array2};
use scirs2_optimize::roots::{root, Method};

// Define a function for which we want to find the root
fn f(x: &[f64]) -> Array1<f64> {
    let x0 = x[0];
    let x1 = x[1];
    array![
        x0.powi(2) + x1.powi(2) - 1.0,  // x^2 + y^2 - 1 = 0 (circle equation)
        x0 - x1                         // x = y (line equation)
    ]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initial guess
    let x0 = array![2.0, 2.0];
    
    // Find the root using Hybrid method
    let result = root(f, &x0, Method::Hybr, None::<fn(&[f64]) -> Array2<f64>>, None)?;
    
    println!("Root: {:?}", result.x);
    println!("Function value at root: {:?}", result.fun);
    
    Ok(())
}
```

## Numerical Stability

The optimization algorithms are designed with numerical stability in mind:

- Gradient calculations include checks for small values to avoid division by zero
- Trust region methods handle degenerate cases robustly
- Line search strategies have safeguards against infinite loops and numerical issues
- Appropriate defaults are chosen to ensure algorithms work across a wide range of problems

## Algorithm Selection

Choose the appropriate algorithm based on your problem:

- **Unconstrained optimization**:
  - `Nelder-Mead`: Robust, doesn't require derivatives, but can be slow for high-dimensional problems
  - `BFGS`: Fast convergence for smooth functions, requires only function values and gradients
  - `Powell`: Good for functions where derivatives are unavailable or unreliable
  - `CG` (Conjugate Gradient): Efficient for large-scale problems

- **Constrained optimization**:
  - `SLSQP`: Efficient for problems with equality and inequality constraints
  - `Trust-Constr`: Trust-region algorithm that handles nonlinear constraints well

- **Least squares**:
  - `LevenbergMarquardt`: Robust for most nonlinear least squares problems
  - `TrustRegionReflective`: Good for bound-constrained problems

- **Root finding**:
  - `Hybr`: Robust hybrid method (modified Powell algorithm)
  - `Broyden1`/`Broyden2`: Good for systems where Jacobian evaluation is expensive
  - `Anderson`: Accelerates convergence for iterative methods
  - `Krylov`: Efficient for large-scale systems

## Error Handling

All functions return `OptimizeResult<OptimizeResults<T>>` where:
- `OptimizeResult` is a Result type that can contain errors like convergence failures
- `OptimizeResults<T>` contains optimization results including the solution, function value, and convergence information

## Performance Considerations

- Most algorithms have been optimized for numerical stability and efficiency
- The code leverages Rust's strong type system and memory safety features
- Performance is comparable to other Rust optimization libraries

## License

This project is dual-licensed under:

- [MIT License](../LICENSE-MIT)
- [Apache License Version 2.0](../LICENSE-APACHE)

You can choose to use either license. See the [LICENSE](../LICENSE) file for details.
