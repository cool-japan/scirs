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
- [x] Finite Element: linear/quadratic triangular elements (no tetrahedral elements found — README previously claimed tetrahedral too; corrected 2026-07-15)
- [x] Spectral methods: Fourier, Chebyshev, Legendre, spectral element
- [x] Finite Volume: upwind and Lax-Wendroff flux, MUSCL reconstruction (minmod/superbee/van Leer limiters) — **corrected 2026-07-15**: "Godunov" is not found anywhere in `src/pde/finite_volume.rs`; the only "Roe" reference there is a citation for the Superbee *limiter* ("Roe 1986"), not a Roe approximate-Riemann-solver flux. Wording corrected; underlying finite-volume capability is real.
- [x] Time-stepping FEM for parabolic/hyperbolic PDEs — **corrected 2026-07-15**: `src/pde/time_fem.rs` implements the θ-method (heat equation) and Newmark-β method (wave equation), not "space-time Galerkin" in the strict technical sense (time as an additional FE dimension within one variational form). The underlying time-stepping-for-FEM capability is real, just was mislabeled.
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
- [ ] MRT multi-relaxation-time collision operator — **discrepancy found 2026-07-15**: no MRT code in `src/lbm.rs` or `src/gpu_lbm.rs`; both solvers (`D2Q9Lbm`, `D3Q19Lbm`, `GpuLbm2D`) implement BGK only. Unmarked from `[x]`.
- [x] Full bounce-back walls and periodic streaming boundary conditions — **corrected 2026-07-15**: verified real (`BoundaryType::Wall` full bounce-back, modulo-indexed periodic streaming in `stream_and_bc`). The velocity inlet uses direct equilibrium-distribution assignment, not the true Zou-He method the text previously claimed; there is no Zou-He implementation anywhere in `src/`.
- [ ] Smagorinsky subgrid-scale turbulence model — **discrepancy found 2026-07-15**: not found anywhere tied to the LBM module. A general (non-LBM) LES/RANS turbulence suite exists under `specialized/fluid_dynamics/turbulence/`, unrelated to the lattice-Boltzmann solvers. Unmarked from `[x]`.
- [ ] Shan-Chen multiphase interaction — **discrepancy found 2026-07-15**: no "Shan", "shan_chen", or multiphase LBM code found anywhere in `src/`. Unmarked from `[x]`.

### Discontinuous Galerkin (DG)
- [x] Modal DG on reference elements
- [x] Nodal DG interpolation-based formulation
- [x] Numerical fluxes: upwind, Lax-Friedrichs, Roe, HLLC (spread across `src/dg.rs` [upwind, Lax-Friedrichs], `src/dg_advanced/types.rs` [Roe, HLLC enum variants], and `src/pde/dg_systems/hllc_euler.rs` [HLLC for Euler systems])
- [ ] hp-adaptivity: simultaneous mesh and degree refinement — **discrepancy found 2026-07-15**: only p-refinement is evidenced (`p_refine_step` + `troubled_cell_indicator` in `src/dg_advanced/high_order_dg.rs`); no h-refinement (element/mesh refinement) for DG was found anywhere, so *combined* hp-adaptivity is not substantiated. Unmarked from `[x]`.

### Phase Field Models
- [x] Cahn-Hilliard equation with semi-implicit time stepping
- [x] Allen-Cahn interface dynamics
- [x] Phase field crystal periodic density model
- [x] Chemo-mechanical coupling for electrode models

### Boundary Element Method (BEM)
- [x] Laplace BEM for potential flow and heat conduction
- [x] Helmholtz BEM for acoustic scattering
- [ ] Fast multipole BEM O(N log N) — **discrepancy found 2026-07-15**: no multipole/FMM code found anywhere in `src/`. Unmarked from `[x]`.
- [ ] Galerkin and collocation formulations — **discrepancy found 2026-07-15**: `src/bem/solver.rs` explicitly documents itself as collocation-only ("The solver discretises the BIE using the collocation method: one equation per element..."); no separate Galerkin assembly path was found despite `src/bem/mod.rs`'s doc-comment claiming "full Galerkin/collocation solver". Unmarked from `[x]`.

### Isogeometric Analysis (IGA)
- [x] B-spline and NURBS basis functions
- [ ] k-refinement for NURBS patches — **discrepancy found 2026-07-15**: no k-refinement code found in `src/iga/`. Unmarked from `[x]`.
- [ ] Structural IGA: shells, beams, solid mechanics — **discrepancy found 2026-07-15**: no shell/beam/solid-mechanics code found; `src/iga/` provides `IGASolver1D`/`IGASolver2D`, generic 1D/2D boundary-value-problem solvers over B-spline/NURBS geometry, not a structural-mechanics layer. Unmarked from `[x]`.

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
- [x] **Streaming particle filter (`StreamingParticleFilter`)** — Implemented (sde/streaming_particle_filter.rs, 701 LoC; complete API: step / systematic_resample / log-weights / ESS / FilterEstimate / SimpleRng helper)

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

## Wave 72 — Symbolic-first ODE solver (2026-05-06)

- [x] **Closed-form ODE branches when `cas::solve_ode` succeeds** (completed 2026-05-07)
  - **Goal:** `scirs2_integrate::symbolic_first::solve_ode_symbolic_or_numerical` — attempt `cas::solve_ode` first; on success return LoweredOp-typed x(t); on failure fall back to existing numerical RK45/BDF interface.
  - **Design:** New file `scirs2-integrate/src/symbolic_first.rs` behind `symbolic` feature (already gates eml.rs). Public enum `SymbolicOrNumericalResult { Symbolic { x_of_t: LoweredOp, kind: OdeKind, integration_constants: Vec<usize> }, Numerical { trajectory: Array2<f64>, time: Array1<f64> } }`. Function `solve_ode_symbolic_or_numerical<F>(rhs_symbolic: Option<&LoweredOp>, rhs_numeric: F, x_var: usize, t_var: usize, ic: (f64,f64), t_end: f64, opts: &OdeOpts) -> Result<SymbolicOrNumericalResult, IntegrateError>`. On SolveOdeError, fall back to solve_ivp. OdeOpts includes `preferred: SolvePreference` with `ForceNumerical` variant. Provide `rhs_from_symbolic_only` constructor that JIT-compiles rhs_symbolic so rhs_numeric is derived from it.
  - **Files:** `scirs2-integrate/src/symbolic_first.rs` (new); `scirs2-integrate/src/lib.rs` (export under `#[cfg(feature = "symbolic")]`); `scirs2-integrate/Cargo.toml` (scirs2-symbolic already optional under symbolic feature — verify only).
  - **Prerequisites:** `scirs2_symbolic::cas::solve_ode` (scirs2-symbolic Wave 72).
  - **Tests:** ≥6 in `tests/symbolic_first_tests.rs`: dx/dt=x → Symbolic, JIT-eval matches exp(t) to 1e-12; Lorenz system → Numerical fallback matches solve_ivp to 1e-10 at t=10; Painlevé I → Numerical fallback; ForceNumerical bypasses symbolic attempt; dimension mismatch clear error; rhs_from_symbolic_only works correctly.
  - **Risk:** Caller misuse if rhs_symbolic and rhs_numeric disagree. Mitigation: rhs_from_symbolic_only constructor; doctest emphasis.

## Wave 73 — Pantelides graph algorithms + SDE Lévy area (2026-05-07)

- [x] **Hopcroft-Karp matching + Tarjan SCC for DAE index reduction** (completed 2026-05-07)
  - Replaced heuristic `find_singular_subsets` at `index_reduction.rs:468` with Pantelides algorithm using Hopcroft-Karp O(E√V) matching and Tarjan iterative SCC.
  - Files: `src/dae/bipartite_matching.rs`, `src/dae/tarjan_scc.rs`, `src/dae/index_reduction.rs`
  - Tests: 13 in `tests/dae_pantelides_tests.rs`

- [x] **SDE Lévy-area via Wiktorsson approximation** (completed 2026-05-07)
  - Lifts strong-order-1.5 SDE solver from scalar/diagonal-only to general noise via `srk_strong_general`.
  - Files: `src/sde/levy_area.rs`, `src/sde/runge_kutta_sde.rs`
  - Tests: 10 in `tests/sde_levy_area_tests.rs`

## Known Issues

- BDF order-5 may exhibit slow convergence near singular Jacobians; automatic order reduction to 3 recommended as workaround.
- LBM multiphase (Shan-Chen) is not implemented yet, for either D2Q9 or D3Q19 — **corrected 2026-07-15**; the previous text claimed D2Q9 support already existed, which could not be verified anywhere in `src/`.
- IGA does not yet include a structural-mechanics (shell/beam/solid) solver at all — **corrected 2026-07-15**; the previous text assumed a structural solver existed and was only missing trimmed NURBS / multi-patch support, but no structural-mechanics code was found in `src/iga/` to begin with.
- No Port-Hamiltonian/BEM coupling code was found anywhere in `src/port_hamiltonian/` or `src/bem/` — **corrected 2026-07-15**; the previous text claimed such coupling was "implemented but not yet verified," but no cross-references between the two modules exist in source.
- SPDE colored noise requires manual specification of the correlation length (`correlation_length` field in `spde::pathwise`); automatic estimation is planned.

## Finance facade build-out (2026-07-07)

- [x] **`specialized::finance` facade layer** — added `AdvancedMonteCarloEngine` (Sobol-sequence variance reduction via `VarianceReductionSuite`, wraps `mc_european_option` / `mc_asian_option` / `mc_barrier_option` / `mc_lookback_option` / `mc_greeks`), `RealTimeRiskMonitor` / `RiskDashboard` (`RiskAlert`, `RiskAlertType`, `AlertSeverity`, `RiskSnapshot`), `ExoticOptionPricer` (`ExoticOptionType`, `PricingMethod`, `PricingResult` — barrier/Asian/lookback/digital), and `RiskAnalyzer` / `PortfolioRiskMetrics` (historical VaR, Monte Carlo VaR, Greeks). These are facades wiring existing Monte Carlo / risk primitives behind a friendlier API, not new numerical methods; exported from `specialized::mod` and the crate root (`lib.rs`).
  - Files: `src/specialized/finance/{advanced_monte_carlo_engine,realtime_risk_engine,exotic_options,risk_management}.rs`, `src/specialized/finance/mod.rs`, `src/specialized/mod.rs`, `src/lib.rs`.
  - See "Proposed follow-ups" below for the three items (`QuantumInspiredRNG`, `RainbowPayoffType`, `StressScenario`) intentionally deferred from this build-out pending a design decision — those remain not-done.
  - scirs2-integrate: 1,873 / 1,815 tests pass, 0 failed (all-features / default-features); re-verified 2026-07-15 via `cargo nextest run -p scirs2-integrate [--all-features]` — identical figures to the 2026-07-07 measurement. Both modes skip the same 4 tests (`#[ignore]`d PINN training tests in `src/pinn/high_level.rs`, reason "slow: PINN training exceeds test timeout"). The slowest passing tests in both modes are the finance Monte Carlo Greeks/parity tests (`specialized::finance::monte_carlo::tests::test_greeks_{delta_call_positive,delta_put_negative,vega_positive}`, `test_put_call_parity`), each 30-90s — expected, not hangs.

## Proposed follow-ups

Deferred from the `specialized::finance` facade build-out
(`advanced_monte_carlo_engine`, `realtime_risk_engine`, `exotic_options`,
`risk_management`) because each needs a concrete numerical-design decision
before implementation, rather than being pure wiring over existing
primitives:

- **`QuantumInspiredRNG`** — the name has no clear existing reference to
  port: there is no established, concrete "quantum-inspired" low-discrepancy
  or pseudo-random generator algorithm specified anywhere in this codebase
  or its design docs to wire up. Implementing it would mean inventing a new
  generator from scratch (as opposed to `Sobol`, which already has a
  working, tested implementation in `finance::monte_carlo::sobol_sequence`
  that the new `advanced_monte_carlo_engine::VarianceReductionSuite` facade
  does wire up). Needs: a decision on which concrete algorithm
  "quantum-inspired RNG" refers to (e.g. quantum-walk-based low-discrepancy
  sequences, QRNG-seeded PRNGs, or something else) before any implementation
  can begin.
- **`RainbowPayoffType`** — multi-asset, correlated-path Monte Carlo pricing
  (rainbow/basket options such as best-of/worst-of/spread payoffs on
  multiple correlated underlyings). This is genuinely new numerical work,
  not just wiring: it requires simulating jointly correlated GBM paths
  across N underlyings (Cholesky-correlated multi-asset path generation,
  analogous to but distinct from the existing single-asset
  `generate_gbm_paths` and the covariance-based `mc_portfolio_var` scenario
  generator), plus a payoff-type enum covering the various rainbow
  conventions (best-of-N call, worst-of-N put, spread, etc.) and their
  respective discounting/averaging conventions. Needs: a decision on which
  rainbow payoff conventions to support and the correlation/calibration
  model (constant correlation matrix vs. time-varying) before
  implementation.
- **`StressScenario`** — portfolio-shock revaluation (e.g. "shock all
  volatilities +20% and spot -10%, reprice the book"). This is genuinely new
  numerical work: it requires a scenario-definition schema (which risk
  factors can be shocked, additive vs. multiplicative shocks, correlated vs.
  independent multi-factor shocks) and a full portfolio revaluation pipeline
  that reprices every position type (vanilla, barrier, Asian, lookback,
  digital, …) under the shocked market state and diffs against the base
  case. `PortfolioRiskMetrics`/`RiskAnalyzer` (implemented in this
  build-out) aggregate VaR/Greeks under the *current* market state only, and
  intentionally do not attempt scenario revaluation. Needs: a decision on
  the scenario schema and which position types must be revaluable under a
  shock before implementation.
