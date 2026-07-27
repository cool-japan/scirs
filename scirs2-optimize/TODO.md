# scirs2-optimize TODO

## Status: v0.6.3 (last updated 2026-07-27)

Untouched by this release's fix work (no optimize-specific changes shipped in 0.6.3); the
test-suite results and verification notes below (from the 0.6.2 release) remain accurate for
0.6.3 since the crate source is unchanged.

`scirs2-optimize` test suite: 2003/2003 tests pass with default features; 2037/2037 tests pass with `--all-features` (both measured 2026-07-15). 0 `todo!()`/`unimplemented!()` stubs in `src/`. Most items below are verified directly against source; a small number of previously-claimed items were not found and are now marked `[ ]` with a citation of where they were checked (see Metaheuristics, MIP, Proximal & Convex Methods, Game Theory & Equilibrium, Minimax & Robust Optimization, Least Squares, Numerical Differentiation, Combinatorial Optimization, and Multi-Objective Optimization sections).

---

## v0.3.3 Completed

### Unconstrained Optimization
- [x] Nelder-Mead simplex with adaptive parameters (Gao-Han scaling)
- [x] BFGS quasi-Newton with Wolfe line search
- [x] L-BFGS with two-loop recursion, configurable memory size
- [x] L-BFGS-B: L-BFGS extended to bound constraints via projected gradient
- [x] Newton-CG: exact Newton with CG inner loop (Hessian-free via finite differences)
- [x] Powell's direction set method with Brent line search
- [x] Conjugate gradient: Polak-Ribiere+, Fletcher-Reeves, Hestenes-Stiefel
- [x] SR1 (Symmetric Rank-1) and DFP (Davidon-Fletcher-Powell) quasi-Newton updates
- [x] Hager-Zhang (CG_DESCENT) line search algorithm

### Constrained Optimization
- [x] SLSQP: sequential QP with active-set QP solver and KKT conditions
- [x] Advanced SQP with second-order corrections and merit function
- [x] Trust Region Constrained (TRCON): dogleg and 2D trust-region subproblem
- [x] Augmented Lagrangian: exact and modified AL methods with adaptive penalty
- [x] Quadratic, linear, and log-barrier penalty methods
- [x] Epsilon-constraint method with systematic constraint relaxation for Pareto front generation

### Mixed Integer Programming (MIP)
- [x] Branch and bound framework with LP relaxation (LP-BB)
- [x] Gomory mixed-integer cuts
- [x] MILP formulations: knapsack, set cover, assignment
- [ ] Feasibility pump heuristic (not found in `src/integer/` or `src/mip.rs`; not yet implemented)
- [ ] Rounding / random-rounding heuristics (not found; not yet implemented)
- [ ] Facility location MILP formulation (not found; not yet implemented)

### Semidefinite & Conic Programming
- [x] SDP solver via ADMM (primal-dual with augmented Lagrangian)
- [x] SDP via interior-point method (primal-dual path-following)
- [x] Linear matrix inequality (LMI) constraint formulation
- [x] SOCP (Second-Order Cone Programming) via interior-point
- [x] LP and QP interior-point (primal-dual path-following)

### Multi-Objective Optimization
- [x] NSGA-II: non-dominated sorting, crowding distance, tournament selection
- [x] NSGA-III: reference point generation (Das-Dennis), reference-point-based selection for many objectives (4+)
- [x] MOEA/D: decomposition via weighted Tchebycheff with neighbourhood mating restriction
- [x] Weighted sum, Tchebycheff, augmented Tchebycheff scalarisation
- [x] Epsilon-constraint with lexicographic optimisation
- [x] Pareto front quality metrics: hypervolume indicator (WFG algorithm, 3-7 objectives; Monte Carlo fallback beyond), IGD, GD
- [ ] Epsilon indicator (additive/multiplicative) quality metric (not found in `multi_objective/indicators/` or `advanced.rs`; not yet implemented)

### Global Optimization
- [x] DIRECT (Dividing RECTangles): Jones et al. deterministic global optimizer
- [x] DIRECT-L: locally biased variant with balance parameter
- [x] Multistart with k-means clustering (basin identification)
- [x] Simulated Annealing: geometric, Cauchy (fast), Boltzmann cooling
- [x] Basin-hopping with configurable local search and step function
- [x] Dual Annealing: hybrid fast SA + classical SA with restart

### Metaheuristics
- [x] Differential Evolution (DE): rand/1/bin, best/1/exp, rand-to-best/1/bin (`DeStrategy::RandToBest1`); jDE self-adaptive variant (`JdeOptions`)
- [x] Particle Swarm Optimization (PSO): inertia weight and constriction factor
- [x] Ant Colony Optimization (ACO): Ant System (AS), Max-Min Ant System (MMAS) for combinatorial instances
- [ ] Ant Colony System (ACS) variant (not found in `src/metaheuristics/aco.rs`; only AS and MMAS implemented)
- [x] Harmony Search (HS): dynamic memory consideration rate, dynamic pitch adjustment
- [x] Simulated Annealing variants (fast SA, generalized SA with visiting distribution)

### Bayesian Optimization
- [x] GP surrogate with SE, Matern 5/2, and ARD kernels; marginal likelihood optimization
- [x] Acquisition functions: EI, LCB, PI, Thompson sampling
- [x] Parallel/batch acquisition: qEI, kriging believer, constant liar
- [x] Constrained BO: unknown feasibility via separate GP per constraint; probability-of-feasibility and expected-feasible-improvement acquisition strategies
- [x] Multi-fidelity BO: AR(1) coregionalization model (Kennedy & O'Hagan 2000) with cost-normalized Expected Improvement across fidelity levels
- [x] Transfer BO: task-similarity-weighted GP ensemble (`exp(-distance/temperature)` task weights) with adaptive target/source acquisition blending and warm-start injection
- [x] Warm-start BO: reuse of evaluations from prior runs via prior data injection

### Stochastic Optimization
- [x] SGD with momentum (Polyak heavy ball) and Nesterov Accelerated Gradient (NAG)
- [x] Adam (Kingma-Ba), AdamW (decoupled weight decay), AMSGrad
- [x] RMSprop (per-parameter adaptive learning rates), Adadelta
- [x] SVRG: full gradient snapshot with variance-reduced stochastic gradient
- [x] SARAH: recursive stochastic gradient with near-optimal convergence
- [x] SPIDER: SARAH with spider-boost momentum updates
- [x] Learning rate schedules: step decay, exponential decay, cosine annealing (SGDR), cyclic LR, one-cycle, polynomial, linear warm-up + cosine decay
- [x] Gradient clipping: global L2-norm clipping, per-parameter value clipping

### Derivative-Free Optimization
- [x] COBYLA: linear approximation-based constrained derivative-free
- [x] BOBYQA: quadratic model-based bound-constrained
- [x] Pattern search: coordinate (compass) search, Hooke-Jeeves
- [x] Mesh Adaptive Direct Search (MADS) framework
- [x] CMA-ES (Covariance Matrix Adaptation Evolution Strategy) — `global/cmaes.rs` (v0.4.2, Wave 44)

### Proximal & Convex Methods
- [x] ISTA (Iterative Soft-Thresholding Algorithm) and FISTA (accelerated, adaptive restart)
- [x] ADMM; Douglas-Rachford and Peaceman-Rachford operator splitting
- [x] Chambolle-Pock primal-dual algorithm
- [x] Proximal operators: L1 (soft-threshold), L2, Linf (projection), nuclear norm, box projection, simplex projection, indicator functions
- [ ] Split Bregman iteration (not found in `src/proximal/`; not yet implemented)
- [ ] Frank-Wolfe (conditional gradient) with linear minimisation oracle (not found anywhere in the crate; not yet implemented)

### Decomposition Methods
- [x] Benders decomposition with cut aggregation and pareto-optimal cuts
- [x] Lagrangian relaxation with subgradient method and bundle method
- [x] Dantzig-Wolfe decomposition (column generation) for block-angular structure
- [x] ADMM-based distributed optimization with variable splitting

### Game Theory & Equilibrium
- [x] Two-player zero-sum Nash equilibrium via LP (`zero_sum::linear_program_minimax`)
- [x] Two-player general-sum Nash equilibrium via support enumeration (`normal_form::find_all_nash_equilibria`)
- [x] Bilevel leader-follower (Stackelberg-style) problems via the generic bilevel KKT reformulation (see Bilevel Optimization section)
- [x] Cooperative game theory: Shapley value, Banzhaf index, core membership, nucleolus, tau-value (`game_theory::cooperative`) — previously undocumented
- [x] Fictitious play, best-response dynamics, replicator dynamics, evolutionarily stable strategies (ESS) — previously undocumented
- [ ] Lemke-Howson pivoting algorithm for general-sum Nash equilibrium (not found; not yet implemented)
- [ ] Coarse correlated equilibrium (CCE) via LP (not found; not yet implemented)
- [ ] Hedge / multiplicative weights for online learning and equilibrium computation (not found; not yet implemented)
- [ ] Counterfactual Regret minimisation (CFR) for extensive-form games (not found; not yet implemented)
- [ ] Mechanism design utilities (not found; not yet implemented)

### Bilevel Optimization
- [x] KKT-based single-level reformulation (MPCC) for convex lower level
- [x] Penalty-based bilevel for nonconvex lower level
- [x] Value function (implicit function) approach for bilevel with convex follower
- [x] Iterative best response dynamics

### Minimax & Robust Optimization
- [x] Alternating gradient descent-ascent (GDA) for min-max problems
- [x] Extragradient method (Korpelevich) for saddle-point problems
- [x] Optimistic gradient descent-ascent (OGDA)
- [x] Distributionally robust optimization: Wasserstein-ball ambiguity set; CVaR (Conditional Value-at-Risk) risk-measure formulation
- [ ] Moment-based (mean-covariance) ambiguity set (not found in `src/dro/`; not yet implemented)
- [x] Robust LP/QP via second-order cone reformulations

### Combinatorial Optimization
- [x] Branch and bound with upper bounding heuristics (greedy, LP relaxation)
- [x] Dynamic programming framework (tabulation and memoization)
- [x] 0-1 knapsack, bounded and unbounded knapsack (DP and LP relaxation)
- [x] TSP: nearest-neighbor heuristic, 2-opt local search, 3-opt, Or-opt segment relocation, Christofides-style MST lower bound
- [ ] Lin-Kernighan moves (not found in `src/combinatorial/tsp.rs`; not yet implemented)
- [x] Assignment problem: Hungarian algorithm (O(n³))
- [x] Bipartite matching: augmenting paths
- [x] Shortest paths: Dijkstra, Bellman-Ford, Floyd-Warshall

### Root Finding
- [x] Hybrid method (modified Powell / hybrd) for systems of equations
- [x] Broyden's good and bad methods for secant-type iteration
- [x] Anderson acceleration for fixed-point iteration
- [x] Krylov-based (GMRES) Newton-Krylov for large systems
- [x] Scalar: Brent, Illinois, ridder's, secant, bisection

### Least Squares
- [x] Levenberg-Marquardt with adaptive damping, Jacobian scaling, trust-region strategy
- [x] Trust Region Reflective for bound-constrained nonlinear LS
- [x] Huber, Bisquare (Tukey biweight), Cauchy robust loss functions
- [x] Weighted, total, separable (VARPRO) least squares
- [x] Scalar/linear least squares with regularisation
- [ ] Arctan robust loss function (not found in `src/least_squares/robust.rs`; only Huber/Bisquare/Cauchy implemented)

### Numerical Differentiation
- [x] Forward, backward, and central finite differences
- [x] Complex-step differentiation (machine-precision gradients)
- [x] Sparse Jacobian computation via graph colouring
- [x] Self-contained reverse-mode AD (computation-graph/tape in `automatic_differentiation::reverse_mode`)
- [ ] Richardson extrapolation for improved accuracy (not found in `src/`; not yet implemented)
- [ ] `scirs2-autograd` integration for reverse-mode AD (crate has no `scirs2-autograd` dependency; reverse-mode AD is self-contained, not integrated with `scirs2-autograd`)

---

## v0.4.0 Roadmap

### Differentiable Optimization
- [x] Differentiable convex optimization layers (OptNet / CVXPY-layers style) — Implemented in v0.4.0 (`differentiable_optimization/layer.rs`, `qp_layer.rs`)
- [x] Implicit differentiation through optimization solutions (KKT sensitivity) — Implemented in v0.4.0 (`differentiable_optimization/kkt_sensitivity.rs`, `implicit_diff.rs`)
- [x] Differentiable LP and QP solvers for end-to-end training — Implemented in v0.4.0 (`differentiable_optimization/diff_lp.rs`, `diff_qp.rs`, `lp_layer.rs`)
- [x] Differentiable combinatorial optimization (perturbed optimizers, SparseMAP) — Implemented in v0.4.0 (`differentiable_optimization/combinatorial.rs`, `perturbed_optimizer.rs`)

### Quantum-Classical Hybrid
- [x] QAOA (Quantum Approximate Optimization Algorithm) interface for combinatorial problems — Implemented in v0.4.0 (`quantum_classical/qaoa.rs`)
- [x] VQE (Variational Quantum Eigensolver) adapter for ground-state problems — Implemented in v0.4.0 (`quantum_classical/vqe.rs`)
- [x] Quantum-inspired tensor network optimization — Implemented in v0.4.0 (`quantum_classical/tensor_network.rs`)
- [x] Classical simulation of small QAOA circuits for benchmarking — Implemented in v0.4.0 (`quantum_classical/statevector.rs`)

### Neural Architecture Search (NAS) Improvements
- [x] DARTS (Differentiable Architecture Search) implementation — darts/mod.rs
- [x] GDAS and SNAS for efficient one-shot NAS — darts/gdas.rs, darts/snas.rs (v0.4.2)
- [x] Predictor-based NAS (surrogate model over architecture space) — darts/predictor_nas.rs (v0.4.2)
- [x] Hardware-aware NAS with latency constraints — hardware_nas/mod.rs

### High-Dimensional Optimization
- [x] Coordinate descent with random and greedy selection rules — Implemented in v0.4.0 (`coordinate_descent/` module)
- [x] Randomized Kaczmarz and block Kaczmarz for large linear systems — Implemented in v0.4.0 (`kaczmarz/` module)
- [x] Subspace embedding methods for dimensionality-reduced optimization — Implemented in v0.4.2 (`subspace_embedding.rs`)
- [x] Sketched gradient descent for massive least-squares — Implemented in v0.4.0 (`sketched/` module)

### Advanced Integer Programming
- [x] Conflict-driven clause learning (CDCL)-style MIP branching — Implemented in v0.4.0 (`integer/cdcl_branching.rs`)
- [x] Lift-and-project cuts — Implemented in v0.4.0 (`integer/lift_project.rs`, `lift_project_mip.rs`)
- [x] Lattice-reduction preprocessing for integer programs — Implemented in v0.4.0 (`integer/lattice/` module)
- [x] Column generation with pricing subproblem interface — Implemented in v0.4.0 (`integer/column_generation.rs`)

---

## Known Issues

- SDP ADMM convergence may be slow for ill-conditioned problems; interior-point is preferred for high-accuracy requirements
- DIRECT becomes computationally expensive beyond ~15 dimensions; switch to Bayesian optimization or differential evolution for high-dimensional global problems
- TSP 3-opt and Or-opt are heuristic and do not guarantee optimality for large instances (n > 200); use exact branch-and-cut for guaranteed solutions
