# scirs2-optimize

[![crates.io](https://img.shields.io/crates/v/scirs2-optimize.svg)](https://crates.io/crates/scirs2-optimize)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](../LICENSE)
[![Documentation](https://img.shields.io/docsrs/scirs2-optimize)](https://docs.rs/scirs2-optimize)

**Comprehensive optimization algorithms for Rust** — part of the [SciRS2](https://github.com/cool-japan/scirs) scientific computing ecosystem.

`scirs2-optimize` is a production-ready, pure-Rust optimization library providing classical numerical methods through state-of-the-art algorithms: mixed-integer programming, semidefinite and conic programming, NSGA-III multi-objective optimization, stochastic gradient methods with variance reduction, Bayesian optimization (constrained, multi-fidelity, transfer, warm-start), game-theoretic formulations, bilevel optimization, and combinatorial solvers.

---

## Overview

Optimization problems appear across all of scientific computing: fitting models to data, engineering design, portfolio construction, neural network training, logistics scheduling, and mechanism design. `scirs2-optimize` covers:

- **Continuous optimization**: unconstrained, constrained (equality, inequality, bounds), conic
- **Discrete & combinatorial**: mixed-integer programming, branch-and-bound, dynamic programming
- **Global optimization**: Bayesian optimization, DIRECT, differential evolution, metaheuristics
- **Multi-objective**: NSGA-II, NSGA-III, scalarization, epsilon-constraint
- **Stochastic & online**: SGD variants, Adam, variance reduction (SVRG, SARAH), schedules
- **Structured problems**: game theory, bilevel, minimax, robust optimization, decomposition

---

## Feature List (v0.5.1)

### Unconstrained Optimization
- Nelder-Mead simplex with adaptive parameter selection
- BFGS and L-BFGS (limited-memory) quasi-Newton methods
- L-BFGS-B: L-BFGS extended with bound constraints
- Newton-CG (Hessian-free Newton with conjugate gradient inner loop)
- Powell's direction set method with Brent line search
- Conjugate Gradient (Polak-Ribiere, Fletcher-Reeves, Hestenes-Stiefel)
- SR1 and DFP quasi-Newton updates
- Hager-Zhang (CG_DESCENT) line search

### Constrained Optimization
- SLSQP (Sequential Least Squares Programming) with active-set QP solver
- SQP (Sequential Quadratic Programming) — enhanced with second-order corrections
- Advanced SQP with exact second-order information and Hessian regularisation
- Trust Region Constrained (TRCON) algorithm
- Augmented Lagrangian methods with adaptive penalty
- Penalty methods (quadratic penalty, barrier, log-barrier)
- Epsilon-constraint method for generating Pareto fronts

### Conic & Convex Optimization
- **Semidefinite Programming (SDP)**: ADMM and interior-point solver for SDP in standard and dual form; linear matrix inequalities (LMIs)
- **Second-Order Cone Programming (SOCP)**: cone constraints via interior-point methods
- **LP / QP interior point**: primal-dual path-following for linear and quadratic programs
- Proximal gradient methods: gradient descent with proximal operator, ADMM, Douglas-Rachford splitting
- Frank-Wolfe (conditional gradient) method for constrained convex problems

### Mixed Integer Programming (MIP)
- Branch and bound framework with LP relaxation at each node
- Cutting plane methods: Gomory cuts, mixed-integer cuts
- Branch and cut with presolve and integrality tightening
- Heuristics: rounding, random rounding, feasibility pump
- MILP formulations for standard combinatorial problems (knapsack, set cover, assignment)

### Multi-Objective Optimization
- NSGA-II: non-dominated sorting + crowding distance selection
- NSGA-III: reference point-based selection for many-objective problems (4+ objectives)
- MOEA/D: decomposition-based multi-objective EA
- Weighted sum, Tchebycheff, and augmented Tchebycheff scalarisation
- Epsilon-constraint with exact Pareto front enumeration
- Pareto front approximation quality metrics (hypervolume, IGD, epsilon indicator)

### Global Optimization
- DIRECT (Dividing RECTangles) deterministic global optimizer
- DIRECT-L (locally biased DIRECT variant)
- Multistart with clustering (systematic basin identification)
- Simulated Annealing with adaptive cooling schedules (geometric, Cauchy, fast)
- Basin-hopping with configurable local search
- Dual Annealing (hybrid fast SA + classical SA)

### Metaheuristics
- Differential Evolution (DE) with strategies: rand/1/bin, best/1/exp, current-to-best, JADE self-adaptation
- Particle Swarm Optimization (PSO) with inertia weight and constriction factor variants
- Ant Colony Optimization (ACO): AS, MMAS, ACS for combinatorial problems
- Harmony Search (HS) with dynamic memory consideration and pitch adjustment rates
- Simulated Annealing variants (fast SA, generalized SA)

### Bayesian Optimization
- Gaussian Process surrogate model with SE, Matern 5/2, and ARD kernels
- Acquisition functions: Expected Improvement (EI), Lower Confidence Bound (LCB), Probability of Improvement (PI), Thompson sampling
- **Constrained Bayesian optimization**: handles unknown feasibility constraints via separate GP models for each constraint
- **Multi-fidelity Bayesian optimization**: BOCA / MF-GP-UCB with fidelity-cost trade-off
- **Transfer Bayesian optimization**: warm-starting from related tasks via task-adaptive priors (RGPE, TAF)
- **Warm-start BO**: reuse of previous evaluations from prior runs
- Hyperparameter optimization via marginal likelihood maximization
- Parallel / batch acquisition (qEI, kriging believer, constant liar)

### Stochastic Optimization
- SGD with momentum (Polyak heavy ball), Nesterov Accelerated Gradient (NAG)
- Adam, AdamW (decoupled weight decay), AMSGrad
- RMSprop and Adadelta
- **SVRG** (Stochastic Variance Reduced Gradient) for finite-sum problems
- **SARAH / SPIDER** variance reduction with near-optimal convergence
- **SARAH+** with automatic restart
- Learning rate schedules: step decay, cosine annealing, cosine-with-warm-restarts (SGDR), cyclic LR, one-cycle, polynomial decay, warm-up linear
- Mini-batch processing and gradient clipping (global norm, value)

### Derivative-Free Optimization
- Nelder-Mead and Powell as derivative-free fallbacks
- COBYLA (Constrained Optimization BY Linear Approximation)
- BOBYQA (Bound Optimization BY Quadratic Approximation)
- NOMAD / MADS (Mesh Adaptive Direct Search) framework
- Pattern search (coordinate search, Hooke-Jeeves)

### Root Finding
- Hybrid methods (modified Powell / hybrd)
- Broyden's good and bad methods
- Anderson acceleration for fixed-point iterations
- Krylov subspace methods (GMRES-based)
- Scalar root finding: Brent's method, Illinois algorithm, ridder's method, secant method

### Least Squares Optimization
- Levenberg-Marquardt with adaptive damping and Jacobian scaling
- Trust Region Reflective for bounded least squares
- Robust variants: Huber, Bisquare (Tukey), Cauchy, Arctan loss functions
- Weighted least squares, total least squares
- Separable least squares (variable projection / VARPRO)
- Bounded nonlinear least squares

### Game Theory & Equilibrium
- Nash equilibrium computation: support enumeration (2-player zero-sum), linear complementarity (LCP), support enumeration (general sum)
- Stackelberg equilibrium (bilevel leader-follower) via MPEC reformulation
- Coarse correlated equilibrium (CCE) via linear programming
- Regret minimisation (Hedge / multiplicative weights, CFR for extensive form)
- Mechanism design utilities

### Bilevel Optimization
- KKT-based reformulation of bilevel to single-level (MPEC/MPCC)
- Penalty-based bilevel method for nonconvex followers
- Value function approach for bilevel with convex lower level
- Iterative best response dynamics

### Decomposition Methods
- Benders decomposition for structured MIPs
- Lagrangian relaxation with subgradient and bundle methods
- Dantzig-Wolfe decomposition (column generation)
- ADMM for distributed optimization
- Alternating Direction Method of Multipliers with operator splitting

### Minimax & Robust Optimization
- Minimax problems: alternating gradient descent-ascent, extragradient, optimistic gradient
- Distributionally robust optimization (DRO): Wasserstein ball, moment-based ambiguity sets
- Robust linear programming with uncertain right-hand side and constraint matrix
- Worst-case analysis via second-order cone reformulations

### Combinatorial Optimization
- Branch and bound with upper bounding heuristics
- Dynamic programming (tabulation and memoization framework)
- Knapsack (0-1, bounded, unbounded) via DP and LP relaxation
- Traveling salesman problem (TSP): nearest-neighbor heuristic, 2-opt, 3-opt, Lin-Kernighan
- Assignment problem (Hungarian algorithm)
- Shortest path: Dijkstra, Bellman-Ford, Floyd-Warshall
- Maximum matching (bipartite: Hungarian; general: Edmond's blossom)

### Convex Optimization (Proximal Methods)
- Proximal gradient descent (ISTA, FISTA)
- Accelerated proximal gradient (APG) with restart
- Proximal operators: L1, L2, Linf, nuclear norm, indicator functions
- Primal-dual methods: Chambolle-Pock, split Bregman
- Frank-Wolfe with linear minimisation oracle

### Automatic & Numerical Differentiation
- Forward-mode (dual numbers) for low-dimensional gradient computation
- Reverse-mode AD via `scirs2-autograd` integration
- Sparse numerical differentiation (Jacobian and Hessian with coloring)
- Richardson extrapolation for high-accuracy finite differences
- Complex-step differentiation for near-machine-precision gradients

### Surrogate Modelling
- Radial Basis Function (RBF) surrogate model (multiquadric, inverse-multiquadric, Gaussian, linear, cubic)
- Polynomial surrogate (full factorial and sparse grid)
- Kriging / GP surrogate with nugget estimation
- Trust-region surrogate management (DYCORS, SRBF)

---

## Quick Start

```toml
[dependencies]
scirs2-optimize = "0.5.1"
```

### Unconstrained Minimisation (BFGS)

```rust
use scirs2_optimize::minimize;
use scirs2_optimize::unconstrained::UnconstrainedMethod;

fn rosenbrock(x: &[f64]) -> f64 {
    let (a, b) = (1.0, 100.0);
    (a - x[0]).powi(2) + b * (x[1] - x[0].powi(2)).powi(2)
}

let result = minimize(rosenbrock, &[0.0, 0.0], UnconstrainedMethod::BFGS, None).unwrap();
println!("Minimum at {:?}, f = {:.2e}", result.x, result.fun);
```

### Mixed Integer Programming

```rust
use scirs2_optimize::mip::{MIP, Variable, VariableKind};

let mut problem = MIP::new();
let x = problem.add_variable(Variable::new(VariableKind::Binary));
let y = problem.add_variable(Variable::new(VariableKind::Continuous { lo: 0.0, hi: 10.0 }));

// Minimize -x - 2y subject to x + y <= 5
problem.set_objective(vec![-1.0, -2.0]);
problem.add_constraint(vec![1.0, 1.0], "<=", 5.0);

let result = problem.solve().unwrap();
println!("MIP optimum: x={}, y={:.2}", result.x[x], result.x[y]);
```

### NSGA-III Multi-Objective Optimisation

```rust
use scirs2_optimize::multiobjective::{nsga3, NSGA3Config};

// Minimise two conflicting objectives
let objectives: Vec<Box<dyn Fn(&[f64]) -> f64>> = vec![
    Box::new(|x| x[0]),
    Box::new(|x| (1.0 - x[0].sqrt()) * x[0] + x[1]),
];

let config = NSGA3Config {
    population_size: 100,
    n_generations: 200,
    bounds: vec![(0.0, 1.0), (0.0, 1.0)],
    ..Default::default()
};

let pareto_front = nsga3(&objectives, config).unwrap();
println!("Pareto front has {} points", pareto_front.len());
```

### Constrained Bayesian Optimisation

```rust
use scirs2_optimize::bayesian::constrained_bo::{ConstrainedBO, ConstrainedBOConfig};

let config = ConstrainedBOConfig {
    n_initial: 10,
    n_iterations: 50,
    bounds: vec![(-5.0, 5.0), (-5.0, 5.0)],
    ..Default::default()
};

let mut bo = ConstrainedBO::new(config);

// Objective and constraint (must be <= 0 for feasibility)
let result = bo
    .minimize(|x| x[0].powi(2) + x[1].powi(2))
    .with_constraint(|x| x[0] + x[1] - 1.0)  // x + y <= 1
    .run()
    .unwrap();

println!("Best feasible: {:?}", result.x);
```

### Stochastic Gradient Descent with Variance Reduction (SVRG)

```rust
use scirs2_optimize::stochastic::new_variance_reduction::{SVRG, SVRGConfig};

let config = SVRGConfig {
    learning_rate: 0.01,
    inner_loop_size: 100,
    max_epochs: 50,
    ..Default::default()
};

let mut optimizer = SVRG::new(config);
optimizer.minimize(&finite_sum_gradient_fn, &mut params, n_samples).unwrap();
```

### Semidefinite Programming

```rust
use scirs2_optimize::conic::{SDP, SDPConstraint};
use scirs2_core::ndarray::{array, Array2};

// Maximise trace(C * X) subject to X >= 0 (PSD), trace(A_i * X) = b_i
let c = array![[2.0, 0.5], [0.5, 1.0]];
let mut sdp = SDP::new(c);

sdp.add_equality_constraint(
    array![[1.0, 0.0], [0.0, 0.0]],
    1.0,
);
sdp.add_equality_constraint(
    array![[0.0, 0.0], [0.0, 1.0]],
    1.0,
);

let result = sdp.solve().unwrap();
println!("SDP optimal value: {:.4}", result.objective);
```

### Nash Equilibrium

```rust
use scirs2_optimize::game_theory::{TwoPlayerGame, find_nash_equilibrium};
use scirs2_core::ndarray::array;

// Prisoner's Dilemma payoff matrix (row player)
let payoffs_row = array![[-1.0, -3.0], [0.0, -2.0]];
let payoffs_col = array![[-1.0, 0.0], [-3.0, -2.0]];

let game = TwoPlayerGame::new(payoffs_row, payoffs_col);
let nash = find_nash_equilibrium(&game).unwrap();
println!("Nash equilibrium: row={:?}, col={:?}", nash.strategy_row, nash.strategy_col);
```

---

## API Overview

| Module | Description |
|---|---|
| `unconstrained` | Nelder-Mead, BFGS, L-BFGS, L-BFGS-B, Newton-CG, Powell, CG, SR1, DFP |
| `constrained` | SLSQP, SQP, trust-region constrained, augmented Lagrangian, penalty |
| `constrained::sqp_advanced` | SQP with second-order corrections |
| `constrained::trust_constr_advanced` | Advanced trust-region constrained |
| `constrained::epsilon_constraint` | Epsilon-constraint multi-objective |
| `constrained::lp_qp_interior` | LP and QP interior-point |
| `conic` | SDP and SOCP interior-point solvers |
| `mip` | Mixed integer programming (branch and cut) |
| `multiobjective` | NSGA-II, NSGA-III, MOEA/D, scalarisation |
| `multi_objective::advanced` | Hypervolume computation, IGD, Pareto pruning |
| `global` | DIRECT, DIRECT-L, dual annealing, basin-hopping |
| `global::direct` | DIRECT algorithm implementation |
| `global::multistart` | Clustering-based multistart |
| `bayesian` | Gaussian Process BO with EI/LCB/PI/Thompson |
| `bayesian::constrained_bo` | BO with unknown feasibility constraints |
| `bayesian::multi_fidelity` | Multi-fidelity BO (BOCA/MF-GP-UCB) |
| `bayesian::transfer_bo` | Transfer BO across related tasks |
| `bayesian::warm_start` | Warm-start BO from prior evaluations |
| `metaheuristics` | DE, PSO, SA |
| `metaheuristics::aco` | Ant Colony Optimization |
| `metaheuristics::de` | Differential Evolution (JADE) |
| `metaheuristics::sa` | Simulated Annealing variants |
| `metaheuristics::harmony` | Harmony Search |
| `evolution` | Evolutionary algorithms framework |
| `stochastic` | SGD, Adam, AdamW, RMSprop, Adadelta |
| `stochastic::new_variance_reduction` | SVRG, SARAH, SPIDER |
| `stochastic::schedules` | LR schedules (cosine, cyclic, one-cycle) |
| `proximal` | ISTA, FISTA, ADMM, proximal operators |
| `convex` | Frank-Wolfe, projected gradient, Chambolle-Pock |
| `decomposition` | Benders, Lagrangian relaxation, Dantzig-Wolfe |
| `bilevel` | KKT reformulation, penalty, value function approaches |
| `minimax` | Alternating GDA, extragradient, optimistic GD |
| `robust` | DRO (Wasserstein, moment), robust LP/QP |
| `game_theory` | Nash, Stackelberg, CCE, regret minimisation |
| `combinatorial` | Branch and bound, DP, TSP, knapsack, assignment |
| `derivative_free` | COBYLA, BOBYQA, MADS, pattern search |
| `surrogate` | RBF, polynomial, kriging surrogate models |
| `hessian` | Hessian approximation and finite-difference utilities |
| `line_search` | Wolfe, strong-Wolfe, Armijo, Hager-Zhang |
| `least_squares` | Levenberg-Marquardt, TRR, robust variants |
| `root_finding` | Hybrid, Broyden, Anderson acceleration, Krylov |
| `scalar` | Brent, golden section, bounded scalar optimisation |
| `symbolic` | Symbolic Newton (exact gradient/Hessian via `scirs2-symbolic`), Lagrangian/KKT, L-BFGS symbolic, trust-region symbolic |
| `wgpu` | GPU-accelerated optimizers (CG, Newton, L-BFGS) via `GpuNdarray` and wgpu dispatch |

### Symbolic and GPU Optimizers (v0.5.1)

| Module | Description |
|---|---|
| `symbolic::newton` | Newton optimizer with exact symbolic Hessian + gradient via `eml::grad`; Gaussian-elimination linear solve |
| `symbolic::lagrangian` | `build_kkt` + `solve_lagrangian_symbolic`; Newton on full N×N KKT system |
| `symbolic::lbfgs` | L-BFGS with symbolic gradient; trust-region symbolic (dogleg) |
| `gpu::cg_gpu` | CG solver with GpuNdarray dot/direction-update; `GPU_CG_THRESHOLD=4096` |
| `gpu::newton_gpu` | Newton with Hessian-vector matmul via GpuNdarray |
| `gpu::lbfgs_gpu` | L-BFGS two-loop recursion via GpuNdarray; `GPU_LBFGS_THRESHOLD=4096`; f64↔f32 boundary |

---

## Feature Flags

| Flag | Description |
|---|---|
| `parallel` | Rayon parallel function evaluation |
| `simd` | SIMD-accelerated linear algebra via `scirs2-core` |
| `async` | Async function evaluation for expensive oracles |
| `serde` | Serialization of results and configurations |
| `symbolic` | Symbolic Newton/Lagrangian/KKT/L-BFGS via `scirs2-symbolic` |
| `wgpu` | GPU-accelerated CG/Newton/L-BFGS via wgpu (`scirs2-core` GpuNdarray) |

Default features: none (pure Rust, no C/Fortran dependencies).

---

## Links

- [SciRS2 project](https://github.com/cool-japan/scirs)
- [docs.rs](https://docs.rs/scirs2-optimize)
- [crates.io](https://crates.io/crates/scirs2-optimize)
- [TODO.md](./TODO.md)

## License

Apache License 2.0. See [LICENSE](../LICENSE) for details.
