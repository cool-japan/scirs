# scirs2-integrate

[![crates.io](https://img.shields.io/crates/v/scirs2-integrate.svg)](https://crates.io/crates/scirs2-integrate)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](../LICENSE)
[![Documentation](https://img.shields.io/docsrs/scirs2-integrate)](https://docs.rs/scirs2-integrate)
[![Status](https://img.shields.io/badge/status-partial-yellow)]()

**Numerical integration, ODE/PDE/SDE solvers, and physics simulation for the SciRS2 scientific computing library (v0.6.3).**

`scirs2-integrate` provides a comprehensive suite of numerical integration methods modeled after SciPy's `integrate` module, extended with advanced capabilities including Stochastic Differential Equation (SDE) solvers, Lattice Boltzmann Method (LBM) fluid simulation, Discontinuous Galerkin (DG) finite elements, phase field models, Boundary Element Methods, Isogeometric Analysis, Port-Hamiltonian discretization, and Monte Carlo / Quasi-Monte Carlo integration — all as pure Rust.

## Features (v0.6.3)

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
- **Finite Element (FEM)**: Linear/quadratic triangular elements (`TriangularMesh`, `ElementType::{Linear, Quadratic}`)
- **Spectral methods**: Fourier, Chebyshev, Legendre, spectral element
- **Finite Volume**: Conservative schemes; upwind and Lax-Wendroff flux, MUSCL reconstruction with minmod/superbee/van Leer limiters
- **Time-stepping FEM**: θ-method for parabolic (heat) problems and Newmark-β for hyperbolic (wave) problems (`time_fem`)
- **Adaptive Mesh Refinement**: Automatic grid refinement and coarsening

### Stochastic Differential Equations (SDE)
- **Euler-Maruyama**: First-order explicit SDE solver
- **Milstein scheme**: Strong order 1.0 SDE solver
- **Strong order 1.5**: Iterated stochastic integral methods
- **Multi-dimensional SDEs**: Correlated noise, vector Wiener processes
- **Stochastic PDE (SPDE)**: Space-time white and colored noise PDEs

### Lattice Boltzmann Method (LBM)
- **D2Q9 and D3Q19 lattices**: Standard 2D and 3D fluid simulation (`D2Q9Lbm`, `D3Q19Lbm`)
- **BGK collision operator**: Single-relaxation-time (multi-relaxation-time is not yet implemented)
- **Boundary conditions**: Full bounce-back walls, prescribed-velocity inlets, zero-gradient outlets, and periodic streaming
- **GPU-accelerated variant**: Separate `gpu_lbm` module (D2Q9 BGK, periodic/no-slip/free-slip boundaries)

### Discontinuous Galerkin (DG)
- **Nodal DG**: 1D solver on Gauss-Legendre-Lobatto nodes (`Dg1dSolver`) with RK4 time-stepping
- **Modal basis utilities**: Legendre modal basis, nodal/modal conversion, troubled-cell indicator, p-refinement (`dg_advanced`)
- **Numerical fluxes**: Upwind, Lax-Friedrichs, Roe, HLLC (Euler systems via `pde::dg_systems`)

### Phase Field Models
- **Cahn-Hilliard equation**: Phase separation with free energy functional
- **Allen-Cahn equation**: Interface dynamics and crystal growth
- **Phase field crystal**: Periodic density functional models
- **Coupled mechanics**: Chemo-mechanical coupling for battery electrode models

### Boundary Element Method (BEM)
- **Laplace BEM**: Potential flow and heat conduction (`LaplaceKernel`)
- **Helmholtz BEM**: Acoustic scattering (`HelmholtzKernel`)
- **Collocation formulation**: Boundary integral equation solved via element-midpoint collocation (`BEMSolver`)
- **Panel method**: Potential-flow panel solver for external aerodynamics (`PanelMethod`)

### Isogeometric Analysis (IGA)
- **B-spline and NURBS basis**: Exact geometry representation (`BSplineBasis`, `BSplineCurve`/`BSplineSurface`, `NurbsCurve`/`NurbsSurface`)
- **1D/2D IGA solvers**: Boundary-value problems over B-spline/NURBS geometry (`IGASolver1D`, `IGASolver2D`)

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
- **Financial PDEs**: Black-Scholes, Heston, Monte Carlo for exotic derivatives; higher-level facades for `AdvancedMonteCarloEngine` (Sobol variance reduction), `ExoticOptionPricer` (barrier/Asian/lookback/digital), `RealTimeRiskMonitor`/`RiskDashboard`, and `RiskAnalyzer`/`PortfolioRiskMetrics` (historical/Monte Carlo VaR, Greeks)
- **Integral equations**: Fredholm and Volterra equations of the 1st and 2nd kind

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
scirs2-integrate = "0.6.4"
```

With optional performance features:

```toml
[dependencies]
scirs2-integrate = { version = "0.6.4", features = ["parallel", "simd"] }
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
if let Some(y_final) = result.y.last() {
    println!("y(5) = {} (exact {})", y_final[0], (-5.0f64).exp());
}
```

### Quasi-Monte Carlo integration

```rust
use scirs2_core::ndarray::{Array1, ArrayView1};
use scirs2_integrate::qmc::qmc_quad;

// Integrate f(x,y) = sin(x+y) over [0,1]^2
let a = Array1::from_vec(vec![0.0, 0.0]);
let b = Array1::from_vec(vec![1.0, 1.0]);
let result = qmc_quad(
    |pt: ArrayView1<f64>| (pt[0] + pt[1]).sin(),
    &a,
    &b,
    Some(8),       // n_estimates
    Some(100_000), // n_points per estimate
    None,          // qrng: defaults to a Halton sequence
    false,         // log
)?;
println!("QMC result = {}, stderr = {}", result.integral, result.standard_error);
```

### Stochastic Differential Equation (Euler-Maruyama)

```rust
use scirs2_integrate::sde_simple::{solve_sde, EulerMaruyama};

// dX = -X dt + 0.5 dW, X(0) = 1.0
let solver = EulerMaruyama::new(
    |x: &[f64], _t: f64| vec![-x[0]],
    |_x: &[f64], _t: f64| vec![vec![0.5]],
    1, // state dimension
    1, // Wiener process dimension
);
let sol = solve_sde(&solver, &[1.0], 0.0, 1.0, 1e-3, 1000, 42);
let final_mean = sol.mean_trajectory().last().map(|v| v[0]).unwrap_or(f64::NAN);
println!("E[X(1)] ≈ {}", final_mean);
```

### Lattice Boltzmann (2D lid-driven cavity)

```rust
use scirs2_integrate::lbm::lid_driven_cavity;

// nx=64, ny=64, kinematic viscosity=0.02, lid velocity=0.1
let mut lbm = lid_driven_cavity(64, 64, 0.02, 0.1);
lbm.run(10_000);
let (velocity_x, velocity_y) = (&lbm.velocity_x, &lbm.velocity_y);
```

### Cahn-Hilliard phase field

```rust
use scirs2_integrate::phase_field::CahnHilliardSolver;

// nx=128, ny=128, dx=1/128, epsilon=0.05, mobility=1.0
let mut solver = CahnHilliardSolver::new(128, 128, 1.0 / 128.0, 0.05, 1.0);
solver.random_init(0.05, 42); // noise amplitude, seed
solver.run(500, 0.01); // 500 time steps, dt = 0.01
let order_param = &solver.phi; // phase field array (Vec<Vec<f64>>)
```

## API Overview

| Module | Description |
|--------|-------------|
| `quad` | Adaptive 1D quadrature (`quad`, `dblquad`, `nquad`) |
| `gaussian` | Gauss-Legendre, Gauss-Hermite, etc. |
| `romberg` | Romberg / Richardson extrapolation |
| `tanhsinh` | Tanh-sinh quadrature for singular integrands |
| `monte_carlo` | Monte Carlo integration with importance sampling |
| `qmc` | Quasi-Monte Carlo quadrature (`qmc_quad`) |
| `quasi_monte_carlo` | Low-discrepancy sequence generators (Sobol, Halton, lattice rules) |
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
| `default` | Core quadrature, ODE, PDE, and other solvers not behind an optional dependency |
| `simd` | SIMD-accelerated numerical operations |
| `parallel` | Multi-threaded parallel execution (Monte Carlo, etc.) |
| `parallel_jacobian` | Parallel Jacobian computation for ODE solvers |
| `async` | Async ODE solving via `tokio` (`async_ode` module, e.g. cached RK4 graphs for repeated solves) |
| `autodiff` | Automatic-differentiation-based Jacobians via `scirs2-autograd` |
| `symbolic` | Symbolic-first ODE/quadrature dispatch via `scirs2-symbolic` (`eml`, `symbolic_first` modules) |
| `gpu_fem` | GPU-accelerated (wgpu) FEM stiffness-matrix assembly, with automatic CPU fallback |
| `symplectic` | Reserved; currently has no compile-time effect — the `symplectic` module (Verlet, Ruth, Gauss-Legendre) is always available |
| `new_ode` | Reserved; currently unused (no code is gated behind it) |

## Test Coverage

Freshly measured via `cargo nextest run -p scirs2-integrate` (2026-07-15):

| Mode | Result |
|------|--------|
| Default features | 1815 tests run: 1815 passed, 4 skipped, 0 failed |
| `--all-features` | 1873 tests run: 1873 passed, 4 skipped, 0 failed |

The 4 skipped tests are `#[ignore]`d in both modes (PINN training tests in `src/pinn/high_level.rs`; reason: "slow: PINN training exceeds test timeout"). The slowest passing tests in both modes are the finance Monte Carlo Greeks/parity tests (`specialized::finance::monte_carlo`), each taking 30-90 seconds — expected given their sample sizes, not a hang.

## Documentation

Full API documentation is available at [docs.rs/scirs2-integrate](https://docs.rs/scirs2-integrate).

Additional guides are in the `docs/` directory:
- `docs/getting_started_scipy_users.md`: Migrating from `scipy.integrate`
- `docs/dae_solver_theory.md`: Theory behind the DAE solvers
- `docs/pde_examples.md`: Worked PDE examples
- `docs/implementation/ODE_MODULE_ORGANIZATION.md`: ODE module internals

## License

Licensed under the Apache License 2.0. See [LICENSE](../LICENSE) for details.
