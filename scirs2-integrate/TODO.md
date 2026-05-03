# scirs2-integrate TODO

## v0.3.3 Completed

### Quadrature
- [x] Adaptive 1D quadrature: `quad`, `dblquad`, `tplquad`, `nquad`
- [x] Gaussian quadrature: Gauss-Legendre, Gauss-Hermite, Gauss-Laguerre, Gauss-Chebyshev
- [x] Romberg integration with Richardson extrapolation
- [x] Tanh-sinh quadrature for endpoint singularities
- [x] Lebedev spherical quadrature rules
- [x] Newton-Cotes coefficient generation
- [x] `quad_vec`: vectorized quadrature for array-valued integrands
- [x] Adaptive cubature for multidimensional integrals

### Monte Carlo and Quasi-Monte Carlo
- [x] Standard Monte Carlo with importance sampling, stratified sampling
- [x] Quasi-Monte Carlo: Sobol sequences, Halton sequences, lattice rules
- [x] `qmc_quad`: high-dimensional integration with low-discrepancy sequences
- [x] Parallel Monte Carlo with work-stealing task distribution

### ODE Solvers (IVP)
- [x] Euler, RK4 (fixed step), RK23 (Bogacki-Shampine), RK45 (Dormand-Prince)
- [x] DOP853: Dormand-Prince 8(5,3) high-precision adaptive
- [x] BDF (orders 1-5): enhanced Jacobian reuse and Broyden updates
- [x] Radau IIA: L-stable implicit Runge-Kutta
- [x] LSODA: automatic stiffness detection and method switching
- [x] `solve_ivp`: unified interface for all methods
- [x] Event detection: zero-crossing, terminal events, dense output
- [x] Mass matrix support: constant, time-dependent, state-dependent
- [x] IMEX splitting for additive stiff + non-stiff systems

### Boundary Value Problems
- [x] Collocation `solve_bvp` with adaptive mesh refinement
- [x] Single shooting method
- [x] Multiple shooting method
- [x] Arc-length continuation for parameter-dependent BVPs

### Differential-Algebraic Equations (DAE)
- [x] BDF-based index-1 DAE solver
- [x] Pantelides algorithm for automatic index reduction
- [x] Block preconditioners for large DAE systems

### Partial Differential Equations (PDE)
- [x] Finite Difference: 1D/2D/3D central, upwind, WENO schemes
- [x] Finite Element: linear/quadratic triangular elements
- [x] Spectral methods: Fourier, Chebyshev, Legendre, spectral element
- [x] Finite Volume: upwind flux, Godunov, Roe
- [x] Time-stepping FEM (space-time Galerkin)
- [x] Adaptive mesh refinement and coarsening

### Stochastic Differential Equations (SDE)
- [x] Euler-Maruyama: first-order SDE solver
- [x] Milstein scheme: strong order 1.0 SDE solver
- [x] Strong order 1.5 iterated stochastic integral methods
- [x] Multi-dimensional SDEs with correlated noise
- [x] Stochastic PDE (SPDE) solvers: space-time white and colored noise

### Lattice Boltzmann Method (LBM)
- [x] D2Q9 and D3Q19 lattice geometries
- [x] BGK single-relaxation-time collision operator
- [x] MRT multi-relaxation-time collision operator
- [x] Bounce-back, Zou-He, and periodic boundary conditions
- [x] Smagorinsky subgrid-scale turbulence model
- [x] Shan-Chen multiphase interaction

### Discontinuous Galerkin (DG)
- [x] Modal DG on reference elements
- [x] Nodal DG interpolation-based formulation
- [x] Numerical fluxes: upwind, Lax-Friedrichs, Roe, HLLC
- [x] hp-adaptivity: simultaneous mesh and degree refinement

### Phase Field Models
- [x] Cahn-Hilliard equation with semi-implicit time stepping
- [x] Allen-Cahn interface dynamics
- [x] Phase field crystal periodic density model
- [x] Chemo-mechanical coupling for electrode models

### Boundary Element Method (BEM)
- [x] Laplace BEM for potential flow and heat conduction
- [x] Helmholtz BEM for acoustic scattering
- [x] Fast multipole BEM O(N log N)
- [x] Galerkin and collocation formulations

### Isogeometric Analysis (IGA)
- [x] B-spline and NURBS basis functions
- [x] k-refinement for NURBS patches
- [x] Structural IGA: shells, beams, solid mechanics

### Port-Hamiltonian Discretization
- [x] Discrete Dirac structures on staggered grids
- [x] Energy-routing interconnection between subsystems
- [x] Passivity-guaranteed dissipation bounds

### Symplectic Integrators
- [x] Stormer-Verlet (2nd order)
- [x] Ruth 4th-order symplectic RK
- [x] Leapfrog / velocity Verlet
- [x] Gauss-Legendre implicit symplectic collocation

### Specialized Domain Solvers
- [x] Schrödinger equation: split-operator and Crank-Nicolson
- [x] Navier-Stokes: projection method for incompressible flow
- [x] Financial PDEs: Black-Scholes, Heston, Monte Carlo exotic derivatives
- [x] Integral equations: Fredholm and Volterra 1st and 2nd kind

### Performance Infrastructure
- [x] Anderson acceleration for iterative solvers
- [x] Hardware auto-tuning (CPU core/cache detection)
- [x] Work-stealing parallel scheduler
- [x] SIMD-accelerated vector operations (feature-gated)
- [x] Memory pool and cache-friendly matrix layouts

## v0.4.0 Roadmap

### GPU Acceleration
- [x] GPU-accelerated LBM: target millions of cells at interactive frame rates — Implemented in v0.4.2 (`gpu_lbm.rs`, D2Q9 BGK, periodic/no-slip/free-slip BC, Poiseuille init)
- [x] GPU ODE ensemble integration: batched RK45 across thousands of parameter sets — Implemented in v0.4.2 (`gpu_ode_ensemble.rs`, Dormand-Prince RK45, sequential/simulated dispatch)
- [x] GPU FEM assembly: shared-memory atomic scatter for sparse stiffness matrix — Implemented in v0.4.3 (`gpu_fem/stiffness.rs`, `gpu_fem/dispatch.rs`; CPU Rayon-parallel CST assembly + GPU stub; 8 tests)
- [x] CUDA graph capture for repeated ODE solve patterns (neural ODE training) — Implemented in v0.4.3 (`async_ode/mod.rs`; `CachedOdeProblem` RK4 graph with pre-allocated buffers + Rayon batch + Tokio async wrapper; 6 tests)

### Adaptive Mesh Refinement
- [x] Full AMR framework: quad-tree / oct-tree dynamic refinement — Implemented in v0.4.0 (`amr/quadtree.rs`, `amr/octree.rs`)
- [x] Conservative prolongation and restriction operators — Implemented in v0.4.0 (`amr/operators.rs`)
- [x] Load-balanced AMR for parallel distributed grids — Implemented in v0.4.3 (`amr/load_balanced.rs`; `AmrGrid2D` quad-tree with Morton-ordered Rayon parallel refinement + 2:1 face-balance enforcement + load-stats; 9 tests)
- [x] Interface tracking with level-set AMR — Implemented in v0.4.0 (`amr/level_set.rs`)

### Quantum Chemistry and Physics
- [x] Hartree-Fock and DFT integrals over Gaussian basis sets — Implemented in v0.4.0 (`specialized/quantum/gaussian_integrals.rs`)
- [x] Density matrix evolution (Lindblad master equation) — Implemented in v0.4.0 (`specialized/quantum/lindblad.rs`)
- [x] Time-dependent Hartree-Fock (TDHF) — Implemented in v0.4.0 (`specialized/quantum/tdhf/` module)
- [x] Path integral Monte Carlo for quantum statistical mechanics — Implemented in v0.4.0 (`pimc/` module)

### Advanced SDE and SPDE
- [x] Weak order 2.0 SDE schemes (Platen-Wagner) — Implemented in v0.4.0 (`sde/weak_order2.rs`)
- [x] Rough SDE driven by fractional Brownian motion — Implemented in v0.4.0 (`sde/rough_sde.rs`, `sde/fractional_brownian.rs`)
- [x] Galerkin SPDE solvers with polynomial chaos expansion — Implemented in v0.4.0 (`polynomial_chaos/` module)
- [x] Real-time particle filter for state estimation (planned 2026-04-17)
  - **Goal:** Full net-new streaming SIR / bootstrap particle filter. The existing `sde/particle_filter.rs` is a batch filter with no streaming API whatsoever (confirmed: grep for `stream|online|step_observation|realtime|real_time` returned zero matches). Build a `StreamingParticleFilter<State, Obs>` that (a) accepts observations one at a time via `step(&mut self, obs: &Obs) -> FilterEstimate<State>`, (b) triggers systematic resampling when ESS drops below a configurable threshold (default `N/2`), (c) takes user-supplied proposal / likelihood / transition closures, (d) operates in bounded memory (particle buffer reused, no allocation per step), (e) exposes `mean`, `covariance`, `effective_sample_size`, and `log_marginal_likelihood` accessors.
  - **Design:** New file `sde/streaming_particle_filter.rs`. `StreamingParticleFilter<S, O>` fields: `particles: Array2<f64>` (N × state_dim) via `scirs2-core::ndarray`, `log_weights: Array1<f64>`, `rng: Box<dyn scirs2_core::random::Rng>`, `transition: Box<dyn Fn(&ArrayView1<f64>, &mut dyn Rng) -> Array1<f64>>`, `log_likelihood: Box<dyn Fn(&ArrayView1<f64>, &O) -> f64>`, `ess_threshold: f64`, `log_evidence: f64`. Resampling: **systematic resampling** (Carpenter-Clifford-Fearnhead 1999) — deterministic partition `[u_0 + i/N : i=0..N]`, invert CDF in O(N). Builder: `StreamingParticleFilterBuilder::new(n_particles).transition(..).log_likelihood(..).ess_threshold(0.5).seed(42).build()`. Keep the existing batch filter untouched; `sde/mod.rs` re-exports both. RNG via `scirs2_core::random` (no raw `rand` in non-core crates per SciRS2 policy).
  - **Files:** `scirs2-integrate/src/sde/streaming_particle_filter.rs` (new), `scirs2-integrate/src/sde/mod.rs` (re-export), `scirs2-integrate/tests/streaming_particle_filter_tests.rs` (new), `scirs2-integrate/TODO.md`.
  - **Prerequisites:** none. RNG via `scirs2_core::random`.
  - **Tests:** `streaming_filter_tracks_linear_gaussian_matches_kalman`, `streaming_filter_adaptive_resample_triggers_on_ess`, `streaming_filter_bearings_only_nonlinear_tracking`, `streaming_filter_bounded_memory_1000_steps`, `streaming_filter_log_evidence_matches_batch`.
  - **Risk:** Systematic resampling + fixed seed is deterministic, so tests are stable. Heap check (`Vec::capacity()`) is fragile — assert no growth rather than exact size. If ndarray reallocation is detected, switch to pre-allocated `Array2::zeros` buffers filled in place.

### PDE Solvers
- [x] Hybridizable DG (HDG) for diffusion-dominated problems — Implemented in v0.4.0 (`pde/hdg/` module)
- [x] Virtual Element Method (VEM) for polygonal meshes — Implemented in v0.4.0 (`pde/vem/` module)
- [x] Peridynamics for fracture mechanics — Implemented in v0.4.0 (`pde/peridynamics/` module)
- [x] Free boundary / Stefan problem solvers — Implemented in v0.4.0 (`pde/stefan/` module)

### Integration and Quadrature
- [x] Filon quadrature for highly oscillatory integrands — Implemented in v0.4.0 (`quadrature/filon.rs`, `quadrature/filon_clenshaw.rs`)
- [x] Sparse grid quadrature for high-dimensional smooth functions — Implemented in v0.4.2 (`quadrature/sparse_grid.rs`, SmolyakGrid/SmolyakConfig/UnivariateRule API with CC/GL/GP rules)
- [x] Clenshaw-Curtis adaptive quadrature with contour deformation (planned 2026-04-17)
  - **Goal:** Audit `scirs2-integrate/src/quadrature/contour_cc.rs` for completeness; ensure it is exported, has a public API, has at least 3 tests covering real-axis convergence + a complex contour + a pole-avoidance case. Flip `[ ]` → `[x]` on this line.
  - **Design:** The file already implements Clenshaw-Curtis with contour deformation (confirmed: `ContourType::{Real, Parabolic, Semicircle, Talbot}`, Talbot contour constructor, Trefethen & Weideman 2007 cited). Extend test coverage and confirm export from the module root. If anything is missing (convergence criteria, adaptive subdivision, Chebyshev-node-based weight construction), implement it — Trefethen-style Clenshaw-Curtis: generate nodes `x_k = cos(kπ/n)`, apply Curtis-Clenshaw FFT weight formula, deform contour `z(t) = t + i·h·(1-t²)` to dodge poles.
  - **Files:** `scirs2-integrate/src/quadrature/contour_cc.rs`, `scirs2-integrate/src/quadrature/mod.rs` (confirm re-export), `scirs2-integrate/tests/contour_cc_tests.rs` (new or extended), `scirs2-integrate/TODO.md`.
  - **Prerequisites:** none.
  - **Tests:** `cc_converges_oscillatory` (∫cos(100x)dx), `cc_with_pole_avoidance` (contour around 1/(x-iε)), `cc_matches_analytic_gaussian`.
  - **Risk:** existing implementation may use simplified CC without true contour deformation; implement Talbot-style or parabolic deformation fully per Trefethen & Weideman if needed.

## Known Issues

- BDF order-5 may exhibit slow convergence near singular Jacobians; automatic order reduction to 3 recommended as workaround.
- LBM Shan-Chen multiphase currently only supports D2Q9; D3Q19 multiphase is planned for v0.4.0.
- IGA structural solver does not yet implement trimmed NURBS or multi-patch coupling.
- Port-Hamiltonian BEM coupling is implemented but not yet verified against the BEM module's matrix assembly.
- SPDE colored noise requires manual specification of the correlation kernel; automatic estimation is planned.
