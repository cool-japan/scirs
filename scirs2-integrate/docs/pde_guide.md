# PDE Solver Guide

This guide provides an overview of the Partial Differential Equation (PDE) solvers in the `scirs2-integrate` crate.

## Introduction

The PDE module implements several numerical methods for solving partial differential equations, with a focus on the Method of Lines (MOL) approach. The Method of Lines transforms PDEs into systems of ODEs by discretizing the spatial derivatives, which can then be solved using the existing ODE solvers.

## Supported PDE Types

Currently, the following PDE types are supported:

- **Parabolic PDEs**: Such as the heat equation and advection-diffusion equations
  - 1D Example: `∂u/∂t = α ∂²u/∂x² - v ∂u/∂x`
  - 2D Example: `∂u/∂t = α(∂²u/∂x² + ∂²u/∂y²) - vx∂u/∂x - vy∂u/∂y`

- **Hyperbolic PDEs**: Such as the wave equation
  - 1D Example: `∂²u/∂t² = c² ∂²u/∂x²`

- **Elliptic PDEs**: Such as Poisson's equation and Laplace's equation
  - Poisson Example: `∇²u = f(x,y)` or expanded as `∂²u/∂x² + ∂²u/∂y² = f(x,y)`
  - Laplace Example: `∇²u = 0` or expanded as `∂²u/∂x² + ∂²u/∂y² = 0`

## Domain and Boundary Conditions

To solve a PDE, you need to define:

1. The spatial domain
2. Boundary conditions
3. Initial conditions

```rust
use scirs2_integrate::{
    Domain, BoundaryCondition, BoundaryConditionType, BoundaryLocation
};

// Define spatial domain [0, 1] with 101 grid points
let domain = Domain::new(
    vec![0.0..1.0],   // Spatial range [0, 1]
    vec![101],        // Number of grid points
)?;

// Define boundary conditions
let bcs = vec![
    // Dirichlet boundary condition at x=0 (left)
    BoundaryCondition {
        bc_type: BoundaryConditionType::Dirichlet,
        location: BoundaryLocation::Lower,
        dimension: 0,
        value: 0.0,
        coefficients: None,
    },
    // Neumann boundary condition at x=1 (right)
    BoundaryCondition {
        bc_type: BoundaryConditionType::Neumann,
        location: BoundaryLocation::Upper,
        dimension: 0,
        value: 0.0,  // ∂u/∂x = 0 at x=1
        coefficients: None,
    },
];
```

## Method of Lines

The Method of Lines (MOL) is a technique for solving PDEs by discretizing all but one of the independent variables, transforming the PDE into a system of ODEs. Typically, space is discretized while time remains continuous.

### Parabolic PDEs (1D)

For 1D parabolic PDEs like the heat equation or advection-diffusion equation:

```rust
use scirs2_integrate::{
    MOLParabolicSolver1D, MOLOptions, ODEMethod
};

// Define physical parameters
let alpha = 0.01;  // Diffusion coefficient
let velocity = 0.5; // Advection velocity (optional)

// Time range
let time_range = [0.0, 0.5];

// Diffusion coefficient function
let diffusion_coeff = |_x: f64, _t: f64, _u: f64| -> f64 {
    alpha  // Could be position, time, or solution-dependent
};

// Initial condition
let initial_condition = |x: f64| -> f64 {
    // Example: sinusoidal initial condition
    (std::f64::consts::PI * x).sin()
};

// Configure the solver
let options = MOLOptions {
    ode_method: ODEMethod::RK45,
    atol: 1e-6,
    rtol: 1e-3,
    max_steps: Some(10000),
    verbose: true,
};

// Create the solver
let mol_solver = MOLParabolicSolver1D::new(
    domain,
    time_range,
    diffusion_coeff,
    initial_condition,
    bcs,
    Some(options),
)?;

// Add optional advection term
let solver_with_advection = mol_solver.with_advection(
    |_x: f64, _t: f64, _u: f64| -> f64 {
        velocity  // Positive velocity: flow from left to right
    }
);

// Add optional reaction term
let complete_solver = solver_with_advection.with_reaction(
    |x: f64, t: f64, u: f64| -> f64 {
        -0.1 * u  // Example: first-order decay term
    }
);

// Solve the PDE
let result = complete_solver.solve()?;

// Access the solution
let t = &result.t;  // Time points
let u = &result.u[0];  // Solution values [time, space]
```

## Finite Difference Methods

The PDE module provides various finite difference schemes for spatial discretization:

```rust
use scirs2_integrate::{
    FiniteDifferenceScheme, first_derivative, second_derivative,
    first_derivative_matrix, second_derivative_matrix
};

// Available schemes
let central_scheme = FiniteDifferenceScheme::CentralDifference;
let forward_scheme = FiniteDifferenceScheme::ForwardDifference;
let backward_scheme = FiniteDifferenceScheme::BackwardDifference;
let high_order_scheme = FiniteDifferenceScheme::FourthOrderCentral;

// Generate differentiation matrices
let dx = 0.01;
let n = 101;
let d1_matrix = first_derivative_matrix(n, dx, central_scheme)?;
let d2_matrix = second_derivative_matrix(n, dx, central_scheme)?;

// Apply to a function
let u = // ... array of function values
let du_dx = apply_diff_matrix(&d1_matrix, &u.view())?;
let d2u_dx2 = apply_diff_matrix(&d2_matrix, &u.view())?;
```

## Boundary Condition Types

The following boundary condition types are supported:

1. **Dirichlet**: Fixed value at the boundary
   ```rust
   BoundaryCondition {
       bc_type: BoundaryConditionType::Dirichlet,
       location: BoundaryLocation::Lower,
       dimension: 0,
       value: 1.0,  // u = 1.0 at boundary
       coefficients: None,
   }
   ```

2. **Neumann**: Fixed gradient at the boundary
   ```rust
   BoundaryCondition {
       bc_type: BoundaryConditionType::Neumann,
       location: BoundaryLocation::Upper,
       dimension: 0,
       value: 0.0,  // ∂u/∂x = 0 at boundary
       coefficients: None,
   }
   ```

3. **Robin (Mixed)**: Linear combination of value and derivative
   ```rust
   BoundaryCondition {
       bc_type: BoundaryConditionType::Robin,
       location: BoundaryLocation::Lower,
       dimension: 0,
       value: 0.0,  // Not used for Robin
       coefficients: Some([1.0, 2.0, 3.0]),  // a*u + b*∂u/∂x = c
   }
   ```

4. **Periodic**: Value at one boundary equals value at the opposite boundary
   ```rust
   BoundaryCondition {
       bc_type: BoundaryConditionType::Periodic,
       location: BoundaryLocation::Lower,  // Only need to specify at one end
       dimension: 0,
       value: 0.0,  // Not used for periodic
       coefficients: None,
   }
   ```

## Example Problems

### Heat Equation

```rust
// Solving: ∂u/∂t = α ∂²u/∂x²
// with u(0,t) = u(1,t) = 0
// and u(x,0) = sin(πx)

// Diffusion coefficient
let alpha = 0.01;
let diffusion_coeff = |_x, _t, _u| alpha;

// Initial condition
let initial_condition = |x| (std::f64::consts::PI * x).sin();

// Dirichlet boundary conditions
let bcs = vec![
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

// Create and solve
let solver = MOLParabolicSolver1D::new(
    domain, [0.0, 0.5], diffusion_coeff, initial_condition, bcs, None
)?;
let result = solver.solve()?;
```

### Advection-Diffusion Equation

```rust
// Solving: ∂u/∂t = α ∂²u/∂x² - v ∂u/∂x
// with u(0,t) = 1, ∂u/∂x|_{x=1} = 0
// and u(x,0) = step function

// Parameters
let alpha = 0.01;
let velocity = 0.5;

// Initial step function
let initial_condition = |x| if x <= 0.1 { 1.0 } else { 0.0 };

// Mixed boundary conditions
let bcs = vec![
    BoundaryCondition {
        bc_type: BoundaryConditionType::Dirichlet,
        location: BoundaryLocation::Lower,
        dimension: 0,
        value: 1.0,
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

// Create and solve
let solver = MOLParabolicSolver1D::new(
    domain, [0.0, 0.5], |_,_,_| alpha, initial_condition, bcs, None
)?
.with_advection(|_,_,_| velocity);

let result = solver.solve()?;
```

## Wave Equation 

```rust
// Solving: ∂²u/∂t² = c² ∂²u/∂x²
// with u(0,t) = u(1,t) = 0
// and u(x,0) = sin(πx), ∂u/∂t(x,0) = 0

// Wave speed squared function (constant)
let wave_speed = 1.0;
let wave_speed_squared = |_x: f64, _t: f64, _u: f64| wave_speed * wave_speed;

// Initial condition and velocity
let initial_condition = |x: f64| (PI * x).sin();
let initial_velocity = |_x: f64| 0.0;

// Boundary conditions
let bcs = vec![
    // Dirichlet boundary conditions at both ends
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

// Create and solve
let solver = MOLWaveEquation1D::new(
    domain, [0.0, 2.0], wave_speed_squared, initial_condition, 
    initial_velocity, bcs, None
)?;
let result = solver.solve()?;

// Access the solution
let t = &result.t;
let u = &result.u;      // Displacement values [time, space]
let u_t = &result.u_t;  // Velocity values [time, space]
```

## 2D Heat Equation

```rust
// Solving: ∂u/∂t = α(∂²u/∂x² + ∂²u/∂y²)
// with u = 0 on all boundaries
// and u(x, y, 0) = sin(πx) * sin(πy)

// Diffusion coefficients (constant in both directions)
let alpha = 0.01;
let diffusion_x = |_x: f64, _y: f64, _t: f64, _u: f64| alpha;
let diffusion_y = |_x: f64, _y: f64, _t: f64, _u: f64| alpha;

// Initial condition
let initial_condition = |x: f64, y: f64| (PI * x).sin() * (PI * y).sin();

// Boundary conditions (Dirichlet on all sides)
let bcs = vec![
    // X-direction boundaries
    BoundaryCondition {
        bc_type: BoundaryConditionType::Dirichlet,
        location: BoundaryLocation::Lower,
        dimension: 0, // x dimension
        value: 0.0,
        coefficients: None,
    },
    BoundaryCondition {
        bc_type: BoundaryConditionType::Dirichlet,
        location: BoundaryLocation::Upper,
        dimension: 0, // x dimension
        value: 0.0,
        coefficients: None,
    },
    // Y-direction boundaries
    BoundaryCondition {
        bc_type: BoundaryConditionType::Dirichlet,
        location: BoundaryLocation::Lower,
        dimension: 1, // y dimension
        value: 0.0,
        coefficients: None,
    },
    BoundaryCondition {
        bc_type: BoundaryConditionType::Dirichlet,
        location: BoundaryLocation::Upper,
        dimension: 1, // y dimension
        value: 0.0,
        coefficients: None,
    },
];

// Create and solve
let solver = MOLParabolicSolver2D::new(
    domain, [0.0, 0.1], diffusion_x, diffusion_y, 
    initial_condition, bcs, None
)?;
let result = solver.solve()?;

// Access the solution
let t = &result.t;
let u = &result.u; // Solution values [time, y, x]
```

## Elliptic PDE Solvers

Elliptic PDEs represent steady-state problems that don't involve time derivatives. The module provides solvers for Poisson's equation and Laplace's equation using both direct and iterative methods.

### Poisson's Equation

```rust
// Solving: ∇²u = f(x,y)
// with Dirichlet boundary conditions (u = 0) on all edges

// Set up domain
let domain = Domain::new(
    vec![0.0..1.0, 0.0..1.0],   // Spatial ranges [0, 1] × [0, 1]
    vec![65, 65],               // Number of grid points
)?;

// Source term: -2π² sin(πx) sin(πy)
let source_term = |x: f64, y: f64| -> f64 {
    -2.0 * PI * PI * (PI * x).sin() * (PI * y).sin()
};

// Boundary conditions
let bcs = vec![
    // Left, right, bottom, top boundaries (all Dirichlet with u = 0)
    BoundaryCondition {
        bc_type: BoundaryConditionType::Dirichlet,
        location: BoundaryLocation::Lower,
        dimension: 0, // x dimension
        value: 0.0,
        coefficients: None,
    },
    // ... add other boundaries similarly
];

// Solver options
let options = EllipticOptions {
    max_iterations: 10000,
    tolerance: 1e-6,
    save_convergence_history: true,
    omega: 1.5, // Relaxation parameter for SOR
    verbose: true,
    fd_scheme: FiniteDifferenceScheme::CentralDifference,
};

// Create the solver
let poisson_solver = PoissonSolver2D::new(
    domain, source_term, bcs, Some(options)
)?;

// Solve using SOR (iterative method)
let sor_result = poisson_solver.solve_sor()?;

// Or solve using direct method
let direct_result = poisson_solver.solve_direct()?;

// Access the solution
let u = &sor_result.u;  // Solution values [y, x]
```

### Laplace's Equation

```rust
// Solving: ∇²u = 0
// with mixed boundary conditions

// Create the Laplace solver
let laplace_solver = LaplaceSolver2D::new(
    domain, boundary_conditions, Some(options)
)?;

// Solve (iterative method)
let result = laplace_solver.solve_sor()?;

// Or solve using direct method
let direct_result = laplace_solver.solve_direct()?;

// Access the solution
let u = &result.u;  // Solution values [y, x]
```

## Finite Element Method (FEM)

The Finite Element Method is a powerful numerical technique for solving PDEs, especially on complex geometries or irregular domains. It approximates the solution using piecewise polynomial functions on a mesh of simple elements.

### Triangular Mesh Generation

```rust
// Create a simple rectangular mesh divided into triangles
let mesh = TriangularMesh::generate_rectangular(
    (0.0, 1.0),  // x range
    (0.0, 1.0),  // y range
    20, 20       // divisions in x and y directions
);

// Alternatively, create a custom mesh for irregular domains
let mesh = create_custom_mesh();
```

### Solving Poisson's Equation with FEM

```rust
// Source term
let source_term = |x: f64, y: f64| -> f64 {
    -2.0 * PI * PI * (PI * x).sin() * (PI * y).sin()
};

// Boundary conditions (Dirichlet u = 0 on all boundaries)
let bcs = vec![
    BoundaryCondition {
        bc_type: BoundaryConditionType::Dirichlet,
        location: BoundaryLocation::Lower,
        dimension: 1, // y dimension (bottom)
        value: 0.0,
        coefficients: None,
    },
    // ... add other boundaries similarly
];

// FEM solver options
let options = FEMOptions {
    element_type: ElementType::Linear,
    max_iterations: 1000,
    tolerance: 1e-6,
    save_convergence_history: false,
    verbose: true,
};

// Create the FEM solver
let mut fem_solver = FEMPoissonSolver::new(
    mesh, source_term, bcs, Some(options)
)?;

// Solve the problem
let result = fem_solver.solve()?;

// Access the solution (values at mesh nodes)
let u = &result.u;         // Solution values
let mesh = &result.mesh;   // Mesh with node coordinates
```

### Handling Irregular Domains

The FEM approach naturally handles irregular domains by using a mesh that conforms to the domain boundary. For example, to solve a PDE on an L-shaped domain:

```rust
// Create an L-shaped domain mesh
let mut mesh = create_l_shaped_mesh(20);

// Rest of the setup is the same as for regular domains
let mut fem_solver = FEMPoissonSolver::new(
    mesh, source_term, bcs, Some(options)
)?;

let result = fem_solver.solve()?;
```

## Spectral Methods

Spectral methods are high-accuracy numerical techniques for solving PDEs, particularly effective for problems with smooth solutions. They approximate the solution using global basis functions, which provide exponential convergence for smooth problems.

### Fourier Spectral Method (Periodic Problems)

```rust
// Set up domain
let domain = Domain::new(
    vec![0.0..2.0*PI],  // Domain [0, 2π]
    vec![128],          // Number of grid points
)?;

// Define source term
let source_term = |x: f64| -> f64 {
    -x.sin() - 0.5 * (3.0 * x).sin()
};

// Define periodic boundary conditions
let bcs = vec![
    BoundaryCondition {
        bc_type: BoundaryConditionType::Periodic,
        location: BoundaryLocation::Lower,
        dimension: 0,
        value: 0.0, // Ignored for periodic conditions
        coefficients: None,
    },
];

// Solver options
let options = SpectralOptions {
    basis: SpectralBasis::Fourier,
    num_modes: 128,
    tolerance: 1e-10,
    use_real_transform: true,
    use_dealiasing: false,
    verbose: true,
    ..Default::default()
};

// Create and solve
let solver = FourierSpectralSolver1D::new(
    domain, source_term, bcs, Some(options)
)?;

let result = solver.solve()?;

// Access solution
let u = &result.u;
let grid = &result.grid;
let coeffs = &result.coefficients;
```

### Chebyshev Spectral Method (Non-Periodic Problems)

```rust
// Set up domain
let domain = Domain::new(
    vec![-1.0..1.0],  // Domain [-1, 1]
    vec![65],         // Number of Chebyshev points
)?;

// Define source term
let source_term = |x: f64| -> f64 {
    -PI * PI * (PI * x).sin()
};

// Define boundary conditions
let bcs = vec![
    // Left boundary (x = -1)
    BoundaryCondition {
        bc_type: BoundaryConditionType::Dirichlet,
        location: BoundaryLocation::Lower,
        dimension: 0,
        value: 0.0,
        coefficients: None,
    },
    // Right boundary (x = 1)
    BoundaryCondition {
        bc_type: BoundaryConditionType::Dirichlet,
        location: BoundaryLocation::Upper,
        dimension: 0,
        value: 0.0,
        coefficients: None,
    },
];

// Solver options
let options = SpectralOptions {
    basis: SpectralBasis::Chebyshev,
    num_modes: 65,
    tolerance: 1e-10,
    verbose: true,
    ..Default::default()
};

// Create and solve
let solver = ChebyshevSpectralSolver1D::new(
    domain, source_term, bcs, Some(options)
)?;

let result = solver.solve()?;

// Access solution
let u = &result.u;
let grid = &result.grid;  // Note: These are Chebyshev points, clustered near boundaries
let coeffs = &result.coefficients;
```

## Future Extensions

The PDE module will be extended to support:

1. 3D problems
2. Higher-order finite elements (quadratic and cubic)
3. Legendre and other spectral methods
4. Systems of coupled PDEs
5. Implicit time-stepping for stiff PDEs
6. More advanced boundary conditions
7. Automatic mesh generation for complex domains