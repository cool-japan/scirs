# PDE Solver Examples

This document provides examples of how to solve various types of partial differential equations (PDEs) using the `scirs2-integrate` crate. It covers both explicit and implicit methods and demonstrates when to use each.

## 1. Heat Equation (Diffusion)

The heat equation is a classic parabolic PDE that models how temperature changes over time:

$\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}$

Where:
- $u(x,t)$ is temperature at position $x$ and time $t$
- $\alpha$ is the thermal diffusivity coefficient

### 1.1 Method of Lines (Explicit)

```rust
use scirs2_integrate::pde::{Domain, BoundaryCondition, BoundaryConditionType, BoundaryLocation};
use scirs2_integrate::pde::method_of_lines::{MOLParabolicSolver1D, MOLOptions};

// Create domain: x ∈ [0, 1]
let domain = Domain::new(vec![0.0..1.0], vec![101])?;

// Define boundary conditions: u(0,t) = 0, u(1,t) = 0
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

// Define diffusion coefficient: α = 0.01
let diffusion_coeff = |_x: f64, _t: f64, _u: f64| 0.01;

// Initial condition: u(x,0) = sin(πx)
let initial_condition = |x: f64| (std::f64::consts::PI * x).sin();

// Solver options
let options = MOLOptions {
    dt: 0.0001, // Small time step for stability
    t_final: 0.5,
    ..Default::default()
};

// Create and use solver
let solver = MOLParabolicSolver1D::new(
    domain,
    initial_condition,
    diffusion_coeff,
    boundary_conditions,
    Some(options),
)?;

let result = solver.solve()?;
```

### 1.2 Crank-Nicolson (Implicit)

```rust
use scirs2_integrate::pde::implicit::{CrankNicolson1D, ImplicitOptions};

// Create Crank-Nicolson solver
let cn_solver = CrankNicolson1D::new(
    domain,
    [0.0, 0.5], // time range
    diffusion_coeff,
    initial_condition,
    boundary_conditions,
    Some(ImplicitOptions {
        dt: Some(0.01), // Can use much larger time step!
        verbose: true,
        ..Default::default()
    }),
)?;

let result = cn_solver.solve()?;
```

### 1.3 Backward Euler (Implicit)

```rust
use scirs2_integrate::pde::implicit::{BackwardEuler1D, ImplicitMethod, ImplicitOptions};

// Create Backward Euler solver
let be_solver = BackwardEuler1D::new(
    domain,
    [0.0, 0.5], // time range
    diffusion_coeff,
    initial_condition,
    boundary_conditions,
    Some(ImplicitOptions {
        method: ImplicitMethod::BackwardEuler,
        dt: Some(0.01),
        verbose: true,
        ..Default::default()
    }),
)?;

let result = be_solver.solve()?;
```

## 2. Advection-Diffusion Equation

The advection-diffusion equation models transport phenomena with both advection and diffusion:

$\frac{\partial u}{\partial t} + v \frac{\partial u}{\partial x} = D \frac{\partial^2 u}{\partial x^2}$

Where:
- $u(x,t)$ is the concentration at position $x$ and time $t$
- $v$ is the advection velocity 
- $D$ is the diffusion coefficient

### 2.1 Method of Lines (Explicit)

```rust
// Define advection coefficient
let velocity = 1.0;
let advection_coeff = |_x: f64, _t: f64, _u: f64| velocity;

// Create solver with advection term
let solver = MOLParabolicSolver1D::new(
    domain,
    initial_condition,
    diffusion_coeff,
    boundary_conditions,
    Some(options),
)?
.with_advection(advection_coeff);

let result = solver.solve()?;
```

### 2.2 Crank-Nicolson (Implicit)

```rust
// Define coefficients
let diffusion_coeff = |_x: f64, _t: f64, _u: f64| 0.01;
let advection_coeff = |_x: f64, _t: f64, _u: f64| 1.0;

// Create solver with advection term
let cn_solver = CrankNicolson1D::new(
    domain,
    [0.0, 0.5],
    diffusion_coeff,
    initial_condition,
    boundary_conditions,
    Some(ImplicitOptions { dt: Some(0.01), ..Default::default() }),
)?
.with_advection(advection_coeff);

let result = cn_solver.solve()?;
```

## 3. Reaction-Diffusion Equation

Reaction-diffusion equations model systems where chemical species react and diffuse:

$\frac{\partial u}{\partial t} = D \frac{\partial^2 u}{\partial x^2} + R(u)$

Where:
- $u(x,t)$ is the concentration
- $D$ is the diffusion coefficient
- $R(u)$ is the reaction term

### 3.1 Fisher-KPP Equation

The Fisher-KPP equation, a classic reaction-diffusion equation with logistic growth:

$\frac{\partial u}{\partial t} = D \frac{\partial^2 u}{\partial x^2} + ru(1-u)$

```rust
// Define diffusion coefficient
let diffusion_coeff = |_x: f64, _t: f64, _u: f64| 0.1;

// Define reaction term (logistic growth)
let growth_rate = 1.0;
let reaction_term = |_x: f64, _t: f64, u: f64| growth_rate * u * (1.0 - u);

// Create solver with reaction term
let solver = BackwardEuler1D::new(
    domain,
    [0.0, 10.0],
    diffusion_coeff,
    initial_condition,
    boundary_conditions,
    Some(ImplicitOptions { dt: Some(0.1), ..Default::default() }),
)?
.with_reaction(reaction_term);

let result = solver.solve()?;
```

## 4. Full Advection-Diffusion-Reaction Equation

The complete equation combines all three phenomena:

$\frac{\partial u}{\partial t} + v \frac{\partial u}{\partial x} = D \frac{\partial^2 u}{\partial x^2} + R(u)$

```rust
// Define all coefficients
let diffusion_coeff = |_x: f64, _t: f64, _u: f64| 0.01;
let advection_coeff = |_x: f64, _t: f64, _u: f64| 1.0;
let reaction_term = |_x: f64, _t: f64, u: f64| u * (1.0 - u);

// Create solver with all terms
let solver = CrankNicolson1D::new(
    domain,
    [0.0, 5.0],
    diffusion_coeff,
    initial_condition,
    boundary_conditions,
    Some(ImplicitOptions { dt: Some(0.05), ..Default::default() }),
)?
.with_advection(advection_coeff)
.with_reaction(reaction_term);

let result = solver.solve()?;
```

## 5. 2D Heat Equation

For 2D problems, we can use the ADI (Alternating Direction Implicit) method:

$\frac{\partial u}{\partial t} = D_x \frac{\partial^2 u}{\partial x^2} + D_y \frac{\partial^2 u}{\partial y^2}$

```rust
use scirs2_integrate::pde::implicit::{ADI2D, ImplicitOptions};

// Define 2D domain: (x,y) ∈ [0,1]×[0,1]
let domain = Domain::new(vec![0.0..1.0, 0.0..1.0], vec![51, 51])?;

// Define boundary conditions for all four sides
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
let diffusion_x = |_x: f64, _y: f64, _t: f64, _u: f64| 0.01;
let diffusion_y = |_x: f64, _y: f64, _t: f64, _u: f64| 0.01;

// Initial condition: u(x,y,0) = sin(πx)sin(πy)
let initial_condition = |x: f64, y: f64| {
    (std::f64::consts::PI * x).sin() * (std::f64::consts::PI * y).sin()
};

// Create and use ADI solver
let adi_solver = ADI2D::new(
    domain,
    [0.0, 1.0],
    diffusion_x,
    diffusion_y,
    initial_condition,
    boundary_conditions,
    Some(ImplicitOptions { dt: Some(0.01), ..Default::default() }),
)?;

let result = adi_solver.solve()?;
```

## 6. Boundary Condition Types

### 6.1 Dirichlet Boundary Conditions

Fix the value at the boundary.

```rust
// u(0, t) = 0, u(1, t) = 1
let bc_dirichlet = vec![
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
        value: 1.0,
        coefficients: None,
    },
];
```

### 6.2 Neumann Boundary Conditions

Fix the derivative at the boundary.

```rust
// du/dx(0, t) = 0, du/dx(1, t) = 0 (no-flux)
let bc_neumann = vec![
    BoundaryCondition {
        bc_type: BoundaryConditionType::Neumann,
        location: BoundaryLocation::Lower,
        dimension: 0,
        value: 0.0,
        coefficients: None,
    },
    BoundaryCondition {
        bc_type: BoundaryConditionType::Neumann,
        location: BoundaryLocation::Upper,
        dimension: 0,
        value: 0.0,
        coefficients: None,
    },
];
```

### 6.3 Robin Boundary Conditions

Define a linear combination of value and derivative.

```rust
// a*u + b*du/dx = c
// Example: u + du/dx = 0 at x=0, 2*u - du/dx = 1 at x=1
let bc_robin = vec![
    BoundaryCondition {
        bc_type: BoundaryConditionType::Robin,
        location: BoundaryLocation::Lower,
        dimension: 0,
        value: 0.0, // Not used for Robin
        coefficients: Some([1.0, 1.0, 0.0]), // a, b, c coefficients
    },
    BoundaryCondition {
        bc_type: BoundaryConditionType::Robin,
        location: BoundaryLocation::Upper,
        dimension: 0,
        value: 0.0, // Not used for Robin
        coefficients: Some([2.0, -1.0, 1.0]), // a, b, c coefficients
    },
];
```

### 6.4 Periodic Boundary Conditions

Make the domain wrap around.

```rust
// u(0, t) = u(1, t) and du/dx(0, t) = du/dx(1, t)
let bc_periodic = vec![
    BoundaryCondition {
        bc_type: BoundaryConditionType::Periodic,
        location: BoundaryLocation::Lower,
        dimension: 0,
        value: 0.0, // Not used for periodic
        coefficients: None,
    },
    BoundaryCondition {
        bc_type: BoundaryConditionType::Periodic,
        location: BoundaryLocation::Upper,
        dimension: 0,
        value: 0.0, // Not used for periodic
        coefficients: None,
    },
];
```

## 7. Advanced PDE Examples

### 7.1 Burgers' Equation (Nonlinear Advection)

The Burgers' equation is a nonlinear hyperbolic PDE:

$\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2}$

```rust
// Define the nonlinear advection coefficient as the solution itself
let advection_coeff = |_x: f64, _t: f64, u: f64| u;

// Define diffusion coefficient (viscosity term)
let diffusion_coeff = |_x: f64, _t: f64, _u: f64| 0.01; // viscosity

// Create implicit solver for Burgers' equation
let solver = BackwardEuler1D::new(
    domain,
    [0.0, 1.0],
    diffusion_coeff,
    initial_condition,
    boundary_conditions,
    Some(ImplicitOptions { dt: Some(0.005), ..Default::default() }),
)?
.with_advection(advection_coeff);

let result = solver.solve()?;
```

### 7.2 Allen-Cahn Equation

The Allen-Cahn equation is a reaction-diffusion equation modeling phase separation:

$\frac{\partial u}{\partial t} = \epsilon \frac{\partial^2 u}{\partial x^2} + u - u^3$

```rust
// Define diffusion coefficient
let epsilon = 0.01;
let diffusion_coeff = |_x: f64, _t: f64, _u: f64| epsilon;

// Define Allen-Cahn reaction term: u - u³
let reaction_term = |_x: f64, _t: f64, u: f64| u - u.powi(3);

// Create solver for Allen-Cahn equation
let solver = CrankNicolson1D::new(
    domain,
    [0.0, 10.0],
    diffusion_coeff,
    initial_condition,
    boundary_conditions,
    Some(ImplicitOptions { dt: Some(0.1), ..Default::default() }),
)?
.with_reaction(reaction_term);

let result = solver.solve()?;
```

## 8. Working with Solutions

### 8.1 Converting to Standard PDESolution

```rust
use scirs2_integrate::pde::PDESolution;

// Convert ImplicitResult to PDESolution
let pde_solution: PDESolution<f64> = result.into();

// Access solution values
let final_time_index = pde_solution.values.len() - 1;
let final_solution = &pde_solution.values[final_time_index];

// Get computation info
println!("Solver: {}", pde_solution.info.method);
println!("Computation time: {} seconds", pde_solution.info.computation_time);
```

### 8.2 Analyzing Results

```rust
// Calculate the maximum value in the solution
let max_value = final_solution.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

// Calculate the L2 norm of the solution
let l2_norm = (final_solution.iter().map(|&x| x * x).sum::<f64>() / 
              final_solution.len() as f64).sqrt();

// Find the position of the maximum value
let (max_i, max_j) = final_solution.indexed_iter()
    .max_by(|(_, &a), (_, &b)| a.partial_cmp(&b).unwrap())
    .map(|((i, j), _)| (i, j))
    .unwrap();
let max_x = domain.grid(0)?[max_i];
let max_y = domain.grid(1)?[max_j];
```

## 9. Method Selection Guidelines

* **Explicit methods** (Method of Lines):
  * Simplest to implement and understand
  * Good for non-stiff problems
  * Require small time steps for stability (especially with diffusion)
  * Limited by CFL condition for advection terms

* **Crank-Nicolson**:
  * Second-order accurate in time
  * Unconditionally stable for the heat equation
  * Can handle moderate stiffness
  * May have oscillations for very stiff problems
  * Good balance of accuracy and stability

* **Backward Euler**:
  * First-order accurate in time (lower accuracy than Crank-Nicolson)
  * Unconditionally stable for all parabolic problems (L-stable)
  * Excellent for very stiff problems
  * Strong damping of high-frequency components
  * May introduce numerical diffusion

* **ADI method**:
  * Efficient for multi-dimensional problems
  * Reduces multi-dimensional problems to a sequence of 1D problems
  * Second-order accurate when using Crank-Nicolson for splitting
  * Good for problems without strong coupling between dimensions

## 10. Performance Tips

1. **Time Step Selection**:
   * For diffusion-dominated problems, dt ~ dx²
   * For advection-dominated problems, dt ~ dx
   * Implicit methods allow much larger time steps

2. **Grid Resolution**:
   * For advection terms, ensure dx is small enough to resolve important features
   * For diffusion terms, ensure the grid can resolve the smallest relevant length scales

3. **Method Selection**:
   * For non-stiff problems, MOL might be fastest due to simpler iterations
   * For stiff problems, Backward Euler often allows the largest time steps
   * For accuracy-critical applications, Crank-Nicolson often gives the best results