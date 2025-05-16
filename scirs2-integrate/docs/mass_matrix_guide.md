# Mass Matrices in ODE Solvers

This guide explains how to use mass matrices with the ODE solvers in scirs2-integrate.

## What are Mass Matrices?

In standard ODE form, equations are written as:

```
y' = f(t, y)
```

However, many physical systems naturally appear in the form:

```
M(t, y) · y' = f(t, y)
```

where M is a square matrix called the mass matrix (or mass-inertia matrix in mechanical systems).

Common examples include:
- Mechanical systems where M represents mass or inertia
- Circuit equations with inductors and capacitors
- Constrained systems in physics
- Semi-discretized PDEs

## Types of Mass Matrices

The scirs2-integrate library supports different types of mass matrices:

1. **Identity Mass Matrix**: Standard ODE form (y' = f(t, y))
2. **Constant Mass Matrix**: M is a constant matrix
3. **Time-Dependent Mass Matrix**: M(t) depends on time
4. **State-Dependent Mass Matrix**: M(t, y) depends on both time and state (limited support)

## Using Mass Matrices

### Basic Usage

To use a mass matrix with an ODE solver, you need to:

1. Create a mass matrix specification using the `MassMatrix` struct
2. Set the mass matrix in the solver options
3. Pass the options to `solve_ivp`

```rust
use ndarray::{array, Array2, ArrayView1};
use scirs2_integrate::ode::{solve_ivp, ODEMethod, ODEOptions, MassMatrix};

// Create a constant mass matrix
let mut m = Array2::<f64>::eye(2);
m[[0, 0]] = 2.0;  // Modified mass element

// Create mass matrix specification
let mass = MassMatrix::constant(m);

// Define ODE function: f(t, y)
let f = |_t: f64, y: ArrayView1<f64>| array![y[1], -y[0]];

// Set up solver options with mass matrix
let opts = ODEOptions {
    method: ODEMethod::RK45,
    rtol: 1e-6,
    atol: 1e-8,
    mass_matrix: Some(mass),
    ..Default::default()
};

// Solve the ODE
let result = solve_ivp(
    f,
    [0.0, 10.0],
    array![1.0, 0.0],
    Some(opts)
)?;
```

### Types of Mass Matrix Specifications

#### 1. Identity Mass Matrix (Standard ODE)

```rust
let mass = MassMatrix::identity();
```

This is equivalent to the standard ODE form and doesn't require special handling.

#### 2. Constant Mass Matrix

```rust
let mut m = Array2::<f64>::eye(3);
m[[0, 1]] = 1.0;
m[[1, 0]] = 2.0;

let mass = MassMatrix::constant(m);
```

This is useful for systems with fixed coefficients.

#### 3. Time-Dependent Mass Matrix

```rust
let mass_func = |t: f64| {
    let mut m = Array2::<f64>::eye(2);
    m[[0, 0]] = 1.0 + 0.5 * t.sin();
    m
};

let mass = MassMatrix::time_dependent(mass_func);
```

This is useful for systems with time-varying coefficients.

#### 4. State-Dependent Mass Matrix

```rust
// Note: Currently, implementation is limited
let mass_func = |t: f64, y: ArrayView1<f64>| {
    let mut m = Array2::<f64>::eye(2);
    m[[0, 0]] = 1.0 + y[0] * y[0];
    m
};

let mass = MassMatrix::state_dependent(mass_func);
```

### Banded Matrix Support

For large systems with banded matrices (e.g., from discretized PDEs), you can specify the bandwidth:

```rust
let mass = MassMatrix::constant(m)
    .with_bandwidth(lower_bandwidth, upper_bandwidth);
```

This can significantly improve performance for large systems.

## Solver Selection

Different solvers handle mass matrices differently:

1. **Explicit Methods (RK45, RK23, etc.)**:
   - Work well with identity, constant, and time-dependent mass matrices
   - Will use matrix inversion to transform the system
   - Not recommended for state-dependent mass matrices

2. **Implicit Methods (BDF, Radau, etc.)**:
   - Required for state-dependent mass matrices
   - More efficient for stiff problems with mass matrices
   - Can handle near-singular mass matrices

## Mathematical Background

### Transforming to Standard Form

For constant and time-dependent mass matrices, the system:

```
M · y' = f(t, y)
```

can be transformed to standard form:

```
y' = M⁻¹ · f(t, y)
```

This transformation is done automatically when you use explicit solvers.

### Direct Handling in Implicit Methods

For state-dependent mass matrices or when M is nearly singular, the solver needs to directly handle the mass matrix in its formulation.

For implicit methods, this involves solving systems of the form:

```
(M - h·J) · Δy = -R
```

where J is the Jacobian of f, h is the step size, and R is the residual.

## Example: Mechanical System

A simple mechanical system with a time-dependent mass:

```rust
// Spring-mass system with time-varying mass
// m(t)·x'' + k·x = 0
// where m(t) = 1 + 0.5·sin(t)

// Define as first-order system
// [m(t) 0] [x'] = [ v ]
// [ 0   1] [v']   [-kx]

// Time-dependent mass matrix
let mass_func = |t: f64| {
    let mut m = Array2::<f64>::eye(2);
    m[[0, 0]] = 1.0 + 0.5 * t.sin();
    m
};

let mass = MassMatrix::time_dependent(mass_func);

// Spring constant
let k = 1.0;

// ODE function
let f = |_t: f64, y: ArrayView1<f64>| {
    array![y[1], -k * y[0]]
};

// Solve with RK45
let opts = ODEOptions {
    method: ODEMethod::RK45,
    mass_matrix: Some(mass),
    ..Default::default()
};

let result = solve_ivp(f, [0.0, 20.0], array![1.0, 0.0], Some(opts))?;
```

## Limitations and Future Work

Current limitations:
- State-dependent mass matrices have limited support
- Direct handling in implicit methods is not yet fully implemented
- Singular mass matrices (DAE systems) are not yet supported

Future improvements:
- Full support for state-dependent mass matrices in all solvers
- Specialized handling for DAE systems
- Improved performance for large-scale systems
- Better diagnostics for mass matrix issues