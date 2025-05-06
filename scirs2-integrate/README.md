# SciRS2 Integrate

[![crates.io](https://img.shields.io/crates/v/scirs2-integrate.svg)](https://crates.io/crates/scirs2-integrate)
[[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)]](../LICENSE)
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
  - Variable step-size methods (RK45, RK23)
  - Implicit methods for stiff problems (BDF)
- **Boundary Value Problem Solvers**: Methods for two-point boundary value problems
  - Collocation methods with adjustable mesh
  - Support for Dirichlet and Neumann boundary conditions
- **Adaptive Methods**: Algorithms with adaptive step size for improved accuracy and efficiency
- **Multi-dimensional Integration**: Support for integrating functions of several variables
- **Vector ODE Support**: Support for systems of ODEs
- **Numerical Utilities**: Common numerical methods for solving mathematical problems
  - Jacobian calculation
  - Newton iteration methods
  - Linear system solvers

## Installation

Add the following to your `Cargo.toml`:

```toml
[dependencies]
scirs2-integrate = "0.1.0-alpha.2"
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
    // ODE Methods
    ODEMethod,              // Enum of available ODE methods
    ODEOptions,             // Options for ODE solvers
    ODEResult,              // Result of ODE integration
    
    // Solve Initial Value Problems
    solve_ivp,              // Solve initial value problem for a system of ODEs
};

// Available methods include:
// - ODEMethod::Euler       // First-order Euler method
// - ODEMethod::RK4         // Fourth-order Runge-Kutta method
// - ODEMethod::RK45        // Dormand-Prince method (variable step)
// - ODEMethod::RK23        // Bogacki-Shampine method (variable step)
// - ODEMethod::BDF         // Backward differentiation formula (for stiff problems)
```

### Boundary Value Problem Solvers

Solvers for two-point boundary value problems:

```rust
use scirs2_integrate::bvp::{
    // BVP solver functions
    solve_bvp,              // Solve a two-point boundary value problem
    solve_bvp_auto,         // Automatically set up and solve common BVP types
    
    // BVP Types
    BVPOptions,             // Options for BVP solvers
    BVPResult,              // Result of BVP solution
};
```

### Numerical Utilities

Common numerical methods used across integration algorithms:

```rust
use scirs2_integrate::utils::{
    // Numerical differentiation
    numerical_jacobian,          // Compute numerical Jacobian of a vector function
    numerical_jacobian_with_param, // Compute Jacobian with scalar parameter
    
    // Linear algebra
    solve_linear_system,         // Solve linear system using Gaussian elimination
    
    // Nonlinear solvers
    newton_method,               // Newton's method for nonlinear systems
    newton_method_with_param,    // Newton's method with scalar parameter
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
use ndarray::array;
use scirs2_integrate::ode::{solve_ivp, ODEOptions, ODEMethod};

// Define the system: dx/dt = alpha*x - beta*x*y, dy/dt = delta*x*y - gamma*y
let lotka_volterra = |_t, y| {
    let (x, y) = (y[0], y[1]);
    let alpha = 1.0;
    let beta = 0.1;
    let delta = 0.1;
    let gamma = 1.0;
    
    array![
        alpha * x - beta * x * y,  // dx/dt
        delta * x * y - gamma * y   // dy/dt
    ]
};

// Initial conditions
let initial_state = array![10.0, 5.0];  // Initial populations of prey and predator

// Options for adaptive solver
let options = ODEOptions {
    method: ODEMethod::RK45,  // Use adaptive Runge-Kutta
    rtol: 1e-6,               // Relative tolerance  
    atol: 1e-8,               // Absolute tolerance
    ..Default::default()
};

// Solve the system
let result = solve_ivp(lotka_volterra, [0.0, 20.0], initial_state, Some(options)).unwrap();

// Plot or analyze the results
println!("Time points: {:?}", result.t);
println!("Prey population at t=20: {}", result.y.last().unwrap()[0]);
println!("Predator population at t=20: {}", result.y.last().unwrap()[1]);
```

### Boundary Value Problem Example

Solving a two-point boundary value problem:

```rust
use ndarray::{array, ArrayView1};
use scirs2_integrate::bvp::{solve_bvp, BVPOptions};
use std::f64::consts::PI;

// Solve the harmonic oscillator ODE: y'' + y = 0
// as a first-order system: y0' = y1, y1' = -y0
// with boundary conditions y0(0) = 0, y0(π) = 0
// Exact solution: y0(x) = sin(x), y1(x) = cos(x)

// Define the ODE system
let fun = |_x: f64, y: ArrayView1<f64>| array![y[1], -y[0]];

// Define the boundary conditions
let bc = |ya: ArrayView1<f64>, yb: ArrayView1<f64>| {
    // Boundary conditions: y0(0) = 0, y0(π) = 0
    array![ya[0], yb[0]]
};

// Initial mesh: 5 points from 0 to π
let x = vec![0.0, PI/4.0, PI/2.0, 3.0*PI/4.0, PI];

// Initial guess: zeros
let y_init = vec![
    array![0.0, 0.0],
    array![0.0, 0.0],
    array![0.0, 0.0],
    array![0.0, 0.0],
    array![0.0, 0.0],
];

// Set options
let options = BVPOptions {
    tol: 1e-6,
    max_iter: 50,
    ..Default::default()
};

// Solve the BVP
let result = solve_bvp(fun, bc, Some(x), y_init, Some(options)).unwrap();

// The solution should be proportional to sin(x)
println!("BVP solution successfully computed with {} iterations", result.n_iter);
println!("Final residual norm: {:.2e}", result.residual_norm);
```

## Implementation Notes

### Boundary Value Problem Solver

The boundary value problem (BVP) solver implements a collocation method that discretizes the differential equation on a mesh and uses a residual-based approach to find the solution. It supports:

- Two-point boundary value problems
- Multiple boundary condition types: Dirichlet, Neumann, and mixed
- Automatic mesh refinement based on solution gradient
- Newton's method for solving the resulting nonlinear systems

### Improved ODE Solvers

The ODE solvers have been enhanced with:

- Improved Runge-Kutta methods with adaptive step size (RK23, RK45)
- Better BDF implementation for stiff equations with numerical Jacobian calculation
- Comprehensive error estimation and step size control

### Numerical Utilities

The module includes several numerical utilities that are useful for solving differential equations:

- Numerical Jacobian calculation for vector functions
- Linear system solver using Gaussian elimination with partial pivoting
- Newton's method for solving nonlinear systems of equations

## Contributing

See the [CONTRIBUTING.md](../CONTRIBUTING.md) file for contribution guidelines.

## License

This project is dual-licensed under:

- [MIT License](../LICENSE-MIT)
- [Apache License Version 2.0](../LICENSE-APACHE)

You can choose to use either license. See the [LICENSE](../LICENSE) file for details.
