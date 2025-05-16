# Implicit Methods for PDEs

This guide explains how to use the implicit methods for solving partial differential equations (PDEs) in the `scirs2-integrate` crate.

## Overview

Implicit methods offer superior stability compared to explicit methods when solving PDEs, especially for stiff problems. This makes them particularly valuable for:

- Problems with multiple time scales
- Problems with diffusion terms where stability constraints would require very small time steps with explicit methods
- Problems that require unconditional stability

The `scirs2-integrate` crate provides several implicit methods:

- **Crank-Nicolson**: A second-order accurate, A-stable method
- **Backward Euler**: A first-order, L-stable method that excels with very stiff problems
- **Alternating Direction Implicit (ADI)**: An efficient operator splitting method for multi-dimensional problems

## One-Dimensional Solvers

### Crank-Nicolson Method

The Crank-Nicolson method is a popular choice for parabolic PDEs due to its second-order accuracy and stability properties.

```rust
use scirs2_integrate::pde::{
    Domain, BoundaryCondition, BoundaryConditionType, BoundaryLocation
};
use scirs2_integrate::pde::implicit::{CrankNicolson1D, ImplicitOptions};

// Define domain: x ∈ [0, 1]
let domain = Domain::new(vec![0.0..1.0], vec![101])?;
let time_range = [0.0, 1.0];

// Define boundary conditions (e.g., Dirichlet)
let boundary_conditions = vec![
    BoundaryCondition {
        bc_type: BoundaryConditionType::Dirichlet,
        location: BoundaryLocation::Lower,
        dimension: 0,
        value: 0.0,
        coefficients: None,
    },
    BoundaryCondition {
        bc_type: BoundaryConditionType::Dirichlet,
        location: BoundaryLocation::Upper,
        dimension: 0,
        value: 0.0,
        coefficients: None,
    },
];

// Define diffusion coefficient function
let diffusion_coeff = |x: f64, t: f64, u: f64| 1.0;

// Define initial condition
let initial_condition = |x: f64| x * (1.0 - x);

// Create solver options
let options = ImplicitOptions {
    dt: Some(0.01),
    verbose: true,
    ..Default::default()
};

// Create and use the solver
let solver = CrankNicolson1D::new(
    domain,
    time_range,
    diffusion_coeff,
    initial_condition,
    boundary_conditions,
    Some(options),
)?;

// Add optional advection and reaction terms if needed
let solver = solver
    .with_advection(|x, t, u| x) // Optional advection term
    .with_reaction(|x, t, u| -u); // Optional reaction term

// Solve the PDE
let result = solver.solve()?;
```

### Backward Euler Method

The Backward Euler method is a robust first-order method that provides excellent stability for stiff problems.

```rust
use scirs2_integrate::pde::implicit::{BackwardEuler1D, ImplicitOptions, ImplicitMethod};

// Create solver options for Backward Euler
let options = ImplicitOptions {
    method: ImplicitMethod::BackwardEuler,
    dt: Some(0.01),
    verbose: true,
    ..Default::default()
};

// Create and use the solver (similar to Crank-Nicolson)
let solver = BackwardEuler1D::new(
    domain,
    time_range,
    diffusion_coeff,
    initial_condition,
    boundary_conditions,
    Some(options),
)?;

// Solve the PDE
let result = solver.solve()?;
```

## Multi-Dimensional Solvers

### Alternating Direction Implicit (ADI) Method

The ADI method is an efficient approach for solving multi-dimensional PDEs. It splits the problem into a sequence of one-dimensional problems, making it computationally efficient.

```rust
use scirs2_integrate::pde::implicit::{ADI2D, ImplicitOptions};

// Define 2D domain: (x,y) ∈ [0,1]×[0,1]
let domain = Domain::new(vec![0.0..1.0, 0.0..1.0], vec![51, 51])?;
let time_range = [0.0, 0.5];

// Create boundary conditions for all four sides
let boundary_conditions = vec![
    // x-direction boundaries
    BoundaryCondition {
        bc_type: BoundaryConditionType::Dirichlet,
        location: BoundaryLocation::Lower,
        dimension: 0,
        value: 0.0,
        coefficients: None,
    },
    BoundaryCondition {
        bc_type: BoundaryConditionType::Dirichlet,
        location: BoundaryLocation::Upper,
        dimension: 0,
        value: 0.0,
        coefficients: None,
    },
    // y-direction boundaries
    BoundaryCondition {
        bc_type: BoundaryConditionType::Dirichlet,
        location: BoundaryLocation::Lower,
        dimension: 1,
        value: 0.0,
        coefficients: None,
    },
    BoundaryCondition {
        bc_type: BoundaryConditionType::Dirichlet,
        location: BoundaryLocation::Upper,
        dimension: 1,
        value: 0.0,
        coefficients: None,
    },
];

// Define diffusion coefficients for both directions
let diffusion_x = |_x: f64, _y: f64, _t: f64, _u: f64| 1.0;
let diffusion_y = |_x: f64, _y: f64, _t: f64, _u: f64| 1.0;

// Define initial condition
let initial_condition = |x: f64, y: f64| (std::f64::consts::PI * x).sin() * (std::f64::consts::PI * y).sin();

// Create solver options
let options = ImplicitOptions {
    dt: Some(0.01),
    verbose: true,
    ..Default::default()
};

// Create and use the ADI solver
let adi_solver = ADI2D::new(
    domain,
    time_range,
    diffusion_x,
    diffusion_y,
    initial_condition,
    boundary_conditions,
    Some(options),
)?;

// Add optional advection and reaction terms if needed
let adi_solver = adi_solver
    .with_advection(
        |x, y, t, u| 0.5, // x-direction advection
        |x, y, t, u| 0.5  // y-direction advection
    )
    .with_reaction(|x, y, t, u| -u); // Optional reaction term

// Solve the PDE
let result = adi_solver.solve()?;
```

## Solver Options

All implicit solvers use the `ImplicitOptions` struct for configuration:

```rust
let options = ImplicitOptions {
    method: ImplicitMethod::CrankNicolson, // Method type
    dt: Some(0.01),                        // Fixed time step
    tolerance: 1e-6,                       // Tolerance for iterative solvers
    max_iterations: 100,                   // Maximum iterations for linear solvers
    min_dt: Some(1e-6),                    // Minimum time step (for adaptive stepping)
    max_dt: Some(0.1),                     // Maximum time step (for adaptive stepping)
    save_every: Some(10),                  // Save solution every N steps
    verbose: true,                         // Print progress information
};
```

## Boundary Conditions

All solvers support four types of boundary conditions:

1. **Dirichlet**: Fixed values at the boundary
2. **Neumann**: Fixed derivatives at the boundary
3. **Robin**: Linear combination of value and derivative
4. **Periodic**: Values at opposite boundaries are equal

Example of Robin boundary conditions:

```rust
// Robin boundary condition: a*u + b*du/dx = c
let robin_bc = BoundaryCondition {
    bc_type: BoundaryConditionType::Robin,
    location: BoundaryLocation::Lower,
    dimension: 0,
    value: 0.0, // Not used for Robin conditions
    coefficients: Some([1.0, 2.0, 3.0]), // [a, b, c] for a*u + b*du/dx = c
};
```

## Result Processing

The solver results can be converted to a standard `PDESolution` for visualization and analysis:

```rust
// Convert ImplicitResult to PDESolution
let pde_solution: PDESolution<f64> = result.into();

// Access solution values
let final_time_index = pde_solution.values.len() - 1;
let final_solution = &pde_solution.values[final_time_index];

// Access solver information
println!("Computation time: {} seconds", pde_solution.info.computation_time);
println!("Number of iterations: {}", pde_solution.info.num_iterations);
```

## Use Cases and Examples

### Heat Equation

```rust
// Heat equation: ∂u/∂t = ∂²u/∂x²
let diffusion_coeff = |_x: f64, _t: f64, _u: f64| 1.0;
let initial_condition = |x: f64| (std::f64::consts::PI * x).sin();

let solver = CrankNicolson1D::new(
    domain,
    time_range,
    diffusion_coeff,
    initial_condition,
    boundary_conditions,
    Some(options),
)?;
```

### Advection-Diffusion Equation

```rust
// Advection-diffusion equation: ∂u/∂t + v*∂u/∂x = D*∂²u/∂x²
let diffusion_coeff = |_x: f64, _t: f64, _u: f64| 0.01; // D
let advection_coeff = |_x: f64, _t: f64, _u: f64| 1.0;  // v

let solver = CrankNicolson1D::new(
    domain,
    time_range,
    diffusion_coeff,
    initial_condition,
    boundary_conditions,
    Some(options),
)?.with_advection(advection_coeff);
```

### Reaction-Diffusion Equation

```rust
// Fisher-KPP equation: ∂u/∂t = D*∂²u/∂x² + r*u*(1-u)
let diffusion_coeff = |_x: f64, _t: f64, _u: f64| 0.5; // D
let reaction_term = |_x: f64, _t: f64, u: f64| 1.0 * (1.0 - 2.0 * u); // linearized r*u*(1-u)

let solver = CrankNicolson1D::new(
    domain,
    time_range,
    diffusion_coeff,
    initial_condition,
    boundary_conditions,
    Some(options),
)?.with_reaction(reaction_term);
```

## Performance Considerations

1. **Solver Choice**:
   - For highest accuracy: Crank-Nicolson
   - For maximum stability: Backward Euler
   - For multi-dimensional problems: ADI

2. **Time Step Selection**:
   - Larger time steps are possible with implicit methods compared to explicit methods
   - Backward Euler allows the largest time steps for very stiff problems
   - For Crank-Nicolson, very large time steps can lead to oscillations with certain problems

3. **Grid Resolution**:
   - Finer grids provide more accurate spatial discretization but increase computational cost
   - Implicit methods handle fine grids better than explicit methods due to their stability properties

4. **Linear System Solving**:
   - For 1D problems, the tridiagonal solver is very efficient
   - For multi-dimensional problems, the ADI method significantly reduces the computational cost