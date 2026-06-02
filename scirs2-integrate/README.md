# scirs2-integrate

[![crates.io](https://img.shields.io/crates/v/scirs2-integrate.svg)](https://crates.io/crates/scirs2-integrate)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](../LICENSE)
[![Documentation](https://img.shields.io/docsrs/scirs2-integrate)](https://docs.rs/scirs2-integrate)

**Numerical integration, ODE/PDE/SDE solvers, and physics simulation for the SciRS2 scientific computing library (v0.5.0).**

`scirs2-integrate` provides a comprehensive suite of numerical integration methods modeled after SciPy's `integrate` module, extended with advanced capabilities including Stochastic Differential Equation (SDE) solvers, Lattice Boltzmann Method (LBM) fluid simulation, Discontinuous Galerkin (DG) finite elements, phase field models, Boundary Element Methods, Isogeometric Analysis, Port-Hamiltonian discretization, and Monte Carlo / Quasi-Monte Carlo integration — all as pure Rust.

## Features (v0.5.0)

### Quadrature (Definite Integrals)
- **Adaptive quadrature**: Automatic error control; `quad`, `dblquad`, `tplquad`, `nquad`
- **Gaussian quadrature**: Gauss-Legendre, Gauss-Hermite, Gauss-Laguerre, Gauss-Chebyshev
- **Romberg integration**: Richardson extrapolation with configurable depth
- **Tanh-sinh quadrature**: High accuracy near endpoint singularities
- **Lebedev rules**: Spherical integration with angular quadrature
- **Newton-Cotes rules**: Coefficient generation for arbitrary orders
- **`quad_vec`**: Vectorized quadrature for array-valued integrands
- **Cubature**: Adaptive multidimensional cubature rules

### Monte Carlo and Quasi-Monte Carlo Integration
- **Standard Monte Carlo**: Importance sampling, stratified sampling, control variates
- **Quasi-Monte Carlo (QMC)**: Sobol sequences, Halton sequences, lattice rules
- **`qmc_quad`**: High-dimensional integration with low-discrepancy sequences
- **Parallel Monte Carlo**: Work-stealing parallel evaluation for throughput

### ODE Solvers (Initial Value Problems)
- **Explicit methods**: Euler, RK4 (fixed step), RK23 (Bogacki-Shampine), RK45 (Dormand-Prince)
- **High-order explicit**: DOP853 (Dormand-Prince 8(5,3)), high-precision adaptive
- **Implicit / stiff methods**: BDF (orders 1-5), Radau IIA (L-stable), LSODA (auto-switching)
- **`solve_ivp`**: Unified solver interface supporting all methods
- **Event detection**: Zero-crossing with direction control, terminal events, dense output
- **Mass matrix support**: Constant, time-dependent, and state-dependent M(t,y)·y' = f(t,y)
- **IMEX methods**: Implicit-Explicit splitting for stiff + non-stiff additive systems

### Boundary Value Problem Solvers
- **Collocation BVP**: `solve_bvp` with adaptive mesh refinement
- **Shooting methods**: Single and multiple shooting for two-point BVPs
- **Continuation methods**: Parameter-dependent BVP families, arc-length continuation

### Differential-Algebraic Equations (DAE)
- **Index-1 DAE**: BDF-based solver for semi-explicit index-1 systems
- **Higher-index DAE**: Pantelides algorithm for automatic index reduction
- **Block preconditioners**: Scalable Krylov methods for large DAE systems

### Partial Differential Equations (PDE)
- **Finite Difference**: 1D/2D/3D spatial schemes; central, upwind, WENO
- **Finite Element (FEM)**: Linear/quadratic triangular and tetrahedral elements
- **Spectral methods**: Fourier, Chebyshev, Legendre, spectral element
- **Finite Volume**: Conservative schemes; upwind flux, Godunov, Roe
- **Time-stepping FEM**: Space-time Galerkin for parabolic/hyperbolic PDEs
- **Adaptive Mesh Refinement**: Automatic grid refinement and coarsening

### Stochastic Differential Equations (SDE)
- **Euler-Maruyama**: First-order explicit SDE solver
- **Milstein scheme**: Strong order 1.0 SDE solver
- **Strong order 1.5**: Iterated stochastic integral methods
- **Multi-dimensional SDEs**: Correlated noise, vector Wiener processes
- **Stochastic PDE (SPDE)**: Space-time white and colored noise PDEs

### Lattice Boltzmann Method (LBM)
- **D2Q9 and D3Q19 lattices**: Standard 2D and 3D fluid simulation
- **BGK and MRT collision operators**: Single-relaxation and multi-relaxation time
- **Boundary conditions**: Bounce-back, Zou-He, periodic
- **Turbulence models**: Smagorinsky subgrid-scale model
- **Multiphase LBM**: Shan-Chen interaction potential

### Discontinuous Galerkin (DG)
- **Modal DG**: Polynomial basis functions on reference elements
- **Nodal DG**: Interpolation-based formulation
- **Numerical fluxes**: Upwind, Lax-Friedrichs, Roe, HLLC
- **hp-adaptivity**: Simultaneous mesh and polynomial degree refinement

### Phase Field Models
- **Cahn-Hilliard equation**: Phase separation with free energy functional
- **Allen-Cahn equation**: Interface dynamics and crystal growth
- **Phase field crystal**: Periodic density functional models
- **Coupled mechanics**: Chemo-mechanical coupling for battery electrode models

### Boundary Element Method (BEM)
- **Laplace BEM**: Potential flow and heat conduction
- **Helmholtz BEM**: Acoustic and electromagnetic scattering
- **Fast multipole BEM**: O(N log N) matrix-vector products
- **Galerkin and collocation**: Both formulations supported

### Isogeometric Analysis (IGA)
- **B-spline and NURBS basis**: Exact geometry representation
- **k-refinement**: Simultaneous h- and p-refinement of NURBS patches
- **Structural IGA**: Shells, beams, and solid mechanics

### Port-Hamiltonian Discretization
- **Structure-preserving**: Discrete Dirac structures on staggered grids
- **Interconnection**: Energy-routing between subsystems
- **Passivity**: Guaranteed energy dissipation bounds

### Symplectic Integrators
- **Stormer-Verlet**: 2nd-order symplectic for separable Hamiltonians
- **Ruth 4th-order**: Higher-order symplectic Runge-Kutta
- **Leapfrog / velocity Verlet**: Molecular dynamics and N-body
- **Gauss-Legendre collocation**: Implicit symplectic for non-separable H

### Specialized Domain Solvers
- **Quantum mechanics**: Schrödinger equation (split-operator, Crank-Nicolson)
- **Fluid dynamics**: Navier-Stokes (projection, incompressible)
- **Financial PDEs**: Black-Scholes, Heston, Monte Carlo for exotic derivatives
- **Integral equations**: Fredholm and Volterra equations of the 1st and 2nd kind

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
scirs2-integrate = "0.5.0"
```

With optional performance features:

```toml
[dependencies]
scirs2-integrate = { version = "0.5.0", features = ["parallel", "simd"] }
```

### Adaptive 1D quadrature

```rust
use scirs2_integrate::quad::quad;

// Integrate sin(x) from 0 to pi; exact result = 2.0
let result = quad(|x: f64| x.sin(), 0.0, std::f64::consts::PI, None)?;
assert!((result.value - 2.0).abs() < 1e-10);
println!("integral = {}, error = {}", result.value, result.abs_error);
```

### Solving an ODE with adaptive step size

```rust
use scirs2_integrate::ode::{solve_ivp, ODEOptions, ODEMethod};
use scirs2_core::ndarray::array;

// dy/dt = -y, y(0) = 1 -> exact: y = exp(-t)
let opts = ODEOptions { method: ODEMethod::RK45, rtol: 1e-8, atol: 1e-10, ..Default::default() };
let result = solve_ivp(
    |_t, y| array![-y[0]],
    [0.0, 5.0],
    array![1.0],
    Some(opts),
)?;
println!("y(5) = {} (exact {})", result.y.last().unwrap()[0], (-5.0f64).exp());
```

### Quasi-Monte Carlo integration

```rust
use scirs2_integrate::monte_carlo::MonteCarloOptions;
use scirs2_integrate::quasi_monte_carlo::qmc_quad;

// Integrate f(x,y) = sin(x+y) over [0,1]^2
let result = qmc_quad(
    |pt| (pt[0] + pt[1]).sin(),
    &[(0.0, 1.0), (0.0, 1.0)],
    MonteCarloOptions { n_samples: 100_000, ..Default::default() },
)?;
println!("QMC result = {}, stderr = {}", result.value, result.std_error);
```

### Stochastic Differential Equation (Euler-Maruyama)

```rust
use scirs2_integrate::sde_simple::{sde_euler_maruyama, SdeOptions};

// dX = -X dt + 0.5 dW, X(0) = 1.0
let opts = SdeOptions { dt: 1e-3, t_end: 1.0, n_paths: 1000, ..Default::default() };
let paths = sde_euler_maruyama(|_t, x| -x, |_t, _x| 0.5, 1.0, opts)?;
println!("E[X(1)] ≈ {}", paths.mean_at_end());
```

### Lattice Boltzmann (2D lid-driven cavity)

```rust
use scirs2_integrate::lbm::{LBMSolver, LBMConfig, D2Q9};

let cfg = LBMConfig {
    nx: 64, ny: 64,
    viscosity: 0.02,
    lid_velocity: 0.1,
    ..Default::default()
};
let mut solver = LBMSolver::<D2Q9>::new(cfg);
solver.run(10_000)?;
let velocity_field = solver.velocity_field();
```

### Cahn-Hilliard phase field

```rust
use scirs2_integrate::phase_field::{CahnHilliard, CahnHilliardConfig};

let cfg = CahnHilliardConfig { nx: 128, ny: 128, epsilon: 0.05, dt: 0.01, ..Default::default() };
let mut sim = CahnHilliard::random_initial(cfg)?;
sim.advance(500)?; // 500 time steps
let order_param = sim.phi(); // phase field array
```

## API Overview

| Module | Description |
|--------|-------------|
| `quad` | Adaptive 1D quadrature (`quad`, `dblquad`, `nquad`) |
| `gaussian` | Gauss-Legendre, Gauss-Hermite, etc. |
| `romberg` | Romberg / Richardson extrapolation |
| `tanhsinh` | Tanh-sinh quadrature for singular integrands |
| `monte_carlo` | Monte Carlo integration with importance sampling |
| `quasi_monte_carlo` | QMC with Sobol/Halton sequences |
| `ode` | ODE initial value problems (`solve_ivp`, all methods) |
| `bvp` | Boundary value problems (`solve_bvp`) |
| `dae` | Differential-algebraic equations |
| `pde` | Finite difference, FEM, spectral, finite volume |
| `sde` / `sde_simple` | Stochastic ODE and SPDE solvers |
| `lbm` | Lattice Boltzmann Method |
| `dg` | Discontinuous Galerkin |
| `phase_field` | Cahn-Hilliard, Allen-Cahn, phase field crystal |
| `bem` | Boundary Element Method |
| `iga` | Isogeometric Analysis |
| `port_hamiltonian` | Port-Hamiltonian structure-preserving methods |
| `shooting` | Single and multiple shooting for BVPs |
| `continuation` | Parameter continuation methods |
| `symplectic` | Symplectic integrators (Verlet, Ruth, GL) |
| `integral_equations` | Fredholm and Volterra integral equations |
| `specialized` | Domain-specific solvers (quantum, fluids, finance) |
| `adaptive` | Adaptive quadrature primitives |
| `quadrature` | Quadrature rule coefficient tables |
| `acceleration` | Anderson acceleration for iterative solvers |
| `autotuning` | Hardware-aware parameter tuning |

## Feature Flags

| Feature | Description |
|---------|-------------|
| `default` | Core quadrature and ODE solvers |
| `simd` | SIMD-accelerated numerical operations |
| `parallel` | Multi-threaded parallel execution |
| `symplectic` | Symplectic integrators for Hamiltonian systems |
| `parallel_jacobian` | Parallel Jacobian computation for ODE solvers |

## Documentation

Full API documentation is available at [docs.rs/scirs2-integrate](https://docs.rs/scirs2-integrate).

Additional guides are in the `docs/` directory:
- `docs/event_detection_guide.md`: Zero-crossing event detection for ODEs
- `docs/mass_matrix_guide.md`: M(t,y)·y' = f(t,y) formulations
- `docs/method_selection_guide.md`: Choosing the right solver
- `docs/performance_optimization_guide.md`: Tuning for throughput

## License

Licensed under the Apache License 2.0. See [LICENSE](../LICENSE) for details.
