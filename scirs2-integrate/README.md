# SciRS2 Integrate

[![crates.io](https://img.shields.io/crates/v/scirs2-integrate.svg)](https://crates.io/crates/scirs2-integrate)
[![License](https://img.shields.io/crates/l/scirs2-integrate.svg)](../LICENSE)
[![Documentation](https://img.shields.io/docsrs/scirs2-integrate)](https://docs.rs/scirs2-integrate)

Numerical integration module for the SciRS2 scientific computing library. This module provides methods for numerical integration of functions and ordinary differential equations (ODEs).

## Features

- **Quadrature Methods**: Various numerical integration methods for definite integrals
  - Basic methods (trapezoid rule, Simpson's rule)
  - Gaussian quadrature for high accuracy with fewer evaluations
  - Romberg integration using Richardson extrapolation
  - Monte Carlo methods for high-dimensional integrals
- **ODE Solvers**: Solvers for ordinary differential equations
  - Euler method
  - Runge-Kutta methods (RK4)
- **Adaptive Integration**: Methods with adaptive step size for improved accuracy
- **Multi-dimensional Integration**: Support for integrating functions of several variables
- **Vector ODE Support**: Support for systems of ODEs

## Installation

Add the following to your `Cargo.toml`:

```toml
[dependencies]
scirs2-integrate = "0.1.0-alpha.1"
ndarray = "0.16.1"
```

Basic usage examples:

```rust
use scirs2_integrate::{quad, ode, gaussian, romberg, monte_carlo};
use scirs2_core::error::CoreResult;
use ndarray::ArrayView1;

// Numerical integration using simpson's rule
fn integrate_example() -> CoreResult<f64> {
    // Define a function to integrate
    let f = |x| x.sin();
    
    // Integrate sin(x) from 0 to pi
    let result = quad::simpson(f, 0.0, std::f64::consts::PI, None)?;
    
    // The exact result should be 2.0
    println!("Integral of sin(x) from 0 to pi: {}", result);
    Ok(result)
}

// Using Gaussian quadrature for high accuracy
fn gaussian_example() -> CoreResult<f64> {
    // Integrate sin(x) from 0 to pi with Gauss-Legendre quadrature
    let result = gaussian::gauss_legendre(|x| x.sin(), 0.0, std::f64::consts::PI, 5)?;
    println!("Gauss-Legendre result: {}", result);
    
    // The error should be very small with just 5 points
    Ok(result)
}

// Using Romberg integration for high accuracy
fn romberg_example() -> CoreResult<f64> {
    let result = romberg::romberg(|x| x.sin(), 0.0, std::f64::consts::PI, None)?;
    println!("Romberg result: {}, Error: {}", result.value, result.abs_error);
    
    // Romberg integration converges very rapidly
    Ok(result.value)
}

// Monte Carlo integration for high-dimensional problems
fn monte_carlo_example() -> CoreResult<f64> {
    // Define options for Monte Carlo integration
    let options = monte_carlo::MonteCarloOptions {
        n_samples: 100000,
        seed: Some(42), // For reproducibility
        ..Default::default()
    };
    
    // Integrate a 3D function: f(x,y,z) = sin(x+y+z) over [0,1]³
    let result = monte_carlo::monte_carlo(
        |point: ArrayView1<f64>| (point[0] + point[1] + point[2]).sin(),
        &[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
        Some(options)
    )?;
    
    println!("Monte Carlo result: {}, Std Error: {}", result.value, result.std_error);
    Ok(result.value)
}

// Solving an ODE: dy/dx = -y, y(0) = 1
fn ode_example() -> CoreResult<()> {
    // Define the ODE: dy/dx = -y
    let f = |_x, y: &[f64]| vec![-y[0]];
    
    // Initial condition
    let y0 = vec![1.0];
    
    // Time points at which we want the solution
    let t_span = (0.0, 5.0);
    let t_eval = Some(vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]);
    
    // Solve the ODE
    let result = ode::solve_ivp(f, t_span, y0, None, t_eval, None)?;
    
    // Print the solution
    println!("Times: {:?}", result.t);
    println!("Values: {:?}", result.y);
    
    // The exact solution is y = e^(-x)
    println!("Exact solution at x=5: {}", (-5.0f64).exp());
    println!("Numerical solution at x=5: {}", result.y.last().unwrap()[0]);
    
    Ok(())
}
```

## Components

### Quadrature Methods

Functions for numerical integration:

```rust
// Basic quadrature methods
use scirs2_integrate::quad::{
    trapezoid,              // Trapezoidal rule integration
    simpson,                // Simpson's rule integration
    adaptive_quad,          // Adaptive quadrature with error estimation
    quad,                   // General-purpose integration
};

// Gaussian quadrature methods
use scirs2_integrate::gaussian::{
    gauss_legendre,         // Gauss-Legendre quadrature
    multi_gauss_legendre,   // Multi-dimensional Gauss-Legendre quadrature
    GaussLegendreQuadrature, // Object-oriented interface for Gauss-Legendre
};

// Romberg integration methods
use scirs2_integrate::romberg::{
    romberg,                // Romberg integration with Richardson extrapolation
    multi_romberg,          // Multi-dimensional Romberg integration
    RombergOptions,         // Options for controlling Romberg integration
    RombergResult,          // Results including error estimates
};

// Monte Carlo integration methods
use scirs2_integrate::monte_carlo::{
    monte_carlo,            // Basic Monte Carlo integration
    importance_sampling,    // Monte Carlo with importance sampling
    MonteCarloOptions,      // Options for controlling Monte Carlo integration
    MonteCarloResult,       // Results including statistical error estimates
    ErrorEstimationMethod,  // Methods for estimating error in Monte Carlo
};
```

### ODE Solvers

Solvers for ordinary differential equations:

```rust
use scirs2_integrate::ode::{
    // ODE Solver Types
    ODESolver,              // Trait for ODE solvers
    RK45,                   // Explicit Runge-Kutta method of order 5(4)
    RK23,                   // Explicit Runge-Kutta method of order 3(2)
    Dopri5,                 // Dormand-Prince method of order 5
    DOP853,                 // Dormand-Prince method of order 8
    Radau,                  // Implicit Runge-Kutta method of Radau IIA family
    BDF,                    // Backward differentiation formula
    LSODA,                  // Adams/BDF method with automatic stiffness detection
    
    // Solve Initial Value Problems
    solve_ivp,              // Solve initial value problem for a system of ODEs
    
    // ODE System Types
    ODESystem,              // Trait for ODE systems
    ODEResult,              // Result of ODE integration
    IVPOptions,             // Options for solve_ivp
};
```

## Advanced Features

### Monte Carlo Integration

For high-dimensional problems, Monte Carlo integration is often the most practical approach:

```rust
use scirs2_integrate::monte_carlo::{monte_carlo, MonteCarloOptions};
use std::marker::PhantomData;
use ndarray::ArrayView1;

// Integrate a function over a 5D hypercube
let f = |x: ArrayView1<f64>| {
    // Sum of squared components: ∫∫∫∫∫(x² + y² + z² + w² + v²) dx dy dz dw dv
    x.iter().map(|&xi| xi * xi).sum()
};

let options = MonteCarloOptions {
    n_samples: 100000,
    seed: Some(42),  // For reproducibility
    _phantom: PhantomData,
    ..Default::default()
};

// Integrate over [0,1]⁵
let result = monte_carlo(
    f,
    &[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
    Some(options)
).unwrap();

println!("Result: {}, Standard error: {}", result.value, result.std_error);
```

### Romberg Integration

Romberg integration uses Richardson extrapolation to accelerate convergence:

```rust
use scirs2_integrate::romberg::{romberg, RombergOptions};

// Function to integrate
let f = |x: f64| x.sin();

// Options
let options = RombergOptions {
    max_iters: 10,
    abs_tol: 1e-12,
    rel_tol: 1e-12,
};

// Integrate sin(x) from 0 to pi
let result = romberg(f, 0.0, std::f64::consts::PI, Some(options)).unwrap();

println!("Result: {}, Error: {}, Iterations: {}", 
         result.value, result.abs_error, result.n_iters);
// Romberg table gives the sequence of approximations
println!("Convergence history: {:?}", result.table);
```

### Adaptive Integration

The module includes adaptive integration methods that adjust step size based on error estimation:

```rust
// Example of adaptive quadrature
use scirs2_integrate::quad::adaptive_quad;

let f = |x| x.sin();
let a = 0.0;
let b = std::f64::consts::PI;
let atol = 1e-8;  // Absolute tolerance
let rtol = 1e-8;  // Relative tolerance

let result = adaptive_quad(&f, a, b, atol, rtol, None).unwrap();
println!("Integral: {}, Error estimate: {}", result.0, result.1);
```

### Vector ODE Support

Support for systems of ODEs:

```rust
// Lotka-Volterra predator-prey model
use scirs2_integrate::ode::{solve_ivp, IVPOptions};

// Define the system: dx/dt = alpha*x - beta*x*y, dy/dt = delta*x*y - gamma*y
let lotka_volterra = |_t, state: &[f64]| {
    let (x, y) = (state[0], state[1]);
    let alpha = 1.0;
    let beta = 0.1;
    let delta = 0.1;
    let gamma = 1.0;
    
    vec![
        alpha * x - beta * x * y,  // dx/dt
        delta * x * y - gamma * y   // dy/dt
    ]
};

// Initial conditions
let initial_state = vec![10.0, 5.0];  // Initial populations of prey and predator

// Time span
let t_span = (0.0, 20.0);

// Solve the system
let result = solve_ivp(lotka_volterra, t_span, initial_state, None, None, None).unwrap();

// Plot or analyze the results
```

## Contributing

See the [CONTRIBUTING.md](../CONTRIBUTING.md) file for contribution guidelines.

## License

This project is licensed under the Apache License, Version 2.0 - see the [LICENSE](../LICENSE) file for details.