# Algorithm Reference Guide

This document provides detailed information about the optimization algorithms implemented in `scirs2-optimize`, including their mathematical foundations, use cases, and implementation details.

## Table of Contents

1. [Unconstrained Optimization](#unconstrained-optimization)
2. [Constrained Optimization](#constrained-optimization)
3. [Stochastic Optimization](#stochastic-optimization)
4. [Least Squares](#least-squares)
5. [Multi-Objective Optimization](#multi-objective-optimization)
6. [Global Optimization](#global-optimization)
7. [Root Finding](#root-finding)
8. [Algorithm Comparison](#algorithm-comparison)

## Unconstrained Optimization

### BFGS (Broyden-Fletcher-Goldfarb-Shanno)

**Mathematical Foundation:**
BFGS is a quasi-Newton method that approximates the Hessian matrix using gradient information from successive iterations.

**Update Rules:**
```
x_{k+1} = x_k + α_k * p_k
p_k = -H_k * ∇f(x_k)
H_{k+1} = H_k + (s_k s_k^T)/(s_k^T y_k) - (H_k y_k y_k^T H_k)/(y_k^T H_k y_k)
```
where:
- `s_k = x_{k+1} - x_k` (step)
- `y_k = ∇f(x_{k+1}) - ∇f(x_k)` (gradient difference)
- `α_k` is the step size from line search

**Characteristics:**
- **Convergence Rate:** Superlinear
- **Memory:** O(n²) for Hessian approximation
- **Gradient Required:** Yes
- **Hessian Required:** No (approximated)

**Best For:**
- Smooth, well-conditioned functions
- Medium-sized problems (< 1000 variables)
- When gradients are available

**Implementation Details:**
```rust
pub struct BfgsState {
    pub hessian_inv: Array2<f64>,  // Inverse Hessian approximation
    pub gradient_prev: Option<Array1<f64>>,
    pub x_prev: Option<Array1<f64>>,
    pub curvature_condition: f64,  // For enforcing positive definiteness
}
```

### L-BFGS (Limited-memory BFGS)

**Mathematical Foundation:**
L-BFGS maintains only the last `m` vectors pairs `(s_k, y_k)` to implicitly represent the Hessian approximation.

**Two-Loop Recursion:**
```rust
// Simplified two-loop recursion for L-BFGS
fn compute_search_direction(gradient: &Array1<f64>, history: &LbfgsHistory) -> Array1<f64> {
    let mut q = gradient.clone();
    let mut alpha = vec![0.0; history.s_vectors.len()];
    
    // First loop (backward)
    for i in (0..history.s_vectors.len()).rev() {
        let rho = 1.0 / history.s_vectors[i].dot(&history.y_vectors[i]);
        alpha[i] = rho * history.s_vectors[i].dot(&q);
        q = &q - &history.y_vectors[i] * alpha[i];
    }
    
    // Scale by initial Hessian approximation
    let gamma = if !history.s_vectors.is_empty() {
        let last_idx = history.s_vectors.len() - 1;
        history.s_vectors[last_idx].dot(&history.y_vectors[last_idx]) /
        history.y_vectors[last_idx].dot(&history.y_vectors[last_idx])
    } else { 1.0 };
    
    let mut r = &q * gamma;
    
    // Second loop (forward)
    for i in 0..history.s_vectors.len() {
        let rho = 1.0 / history.s_vectors[i].dot(&history.y_vectors[i]);
        let beta = rho * history.y_vectors[i].dot(&r);
        r = &r + &history.s_vectors[i] * (alpha[i] - beta);
    }
    
    -r // Return search direction
}
```

**Characteristics:**
- **Memory:** O(mn) where m is history size (typically 5-20)
- **Convergence:** Superlinear (asymptotically)
- **Scalability:** Excellent for large problems

**Best For:**
- Large-scale problems (> 1000 variables)
- Limited memory environments
- When full BFGS Hessian storage is prohibitive

### Newton's Method

**Mathematical Foundation:**
Uses the exact Hessian for second-order convergence.

**Update Rule:**
```
x_{k+1} = x_k - [∇²f(x_k)]^{-1} * ∇f(x_k)
```

**Characteristics:**
- **Convergence Rate:** Quadratic (near minimum)
- **Memory:** O(n²) for Hessian
- **Requirements:** Hessian must be provided or computed

**Best For:**
- Problems where Hessian is cheaply available
- High accuracy requirements
- Well-conditioned problems

### Conjugate Gradient (CG)

**Mathematical Foundation:**
Generates conjugate directions for quadratic functions, generalizes to nonlinear problems.

**Update Rules:**
```
p_0 = -∇f(x_0)
x_{k+1} = x_k + α_k * p_k
β_k = ||∇f(x_{k+1})||² / ||∇f(x_k)||²  (Fletcher-Reeves)
p_{k+1} = -∇f(x_{k+1}) + β_k * p_k
```

**Variants Implemented:**
- **Fletcher-Reeves:** `β_k = ||g_{k+1}||² / ||g_k||²`
- **Polak-Ribière:** `β_k = g_{k+1}^T(g_{k+1} - g_k) / ||g_k||²`
- **Hestenes-Stiefel:** `β_k = g_{k+1}^T(g_{k+1} - g_k) / p_k^T(g_{k+1} - g_k)`

**Characteristics:**
- **Memory:** O(n) - very memory efficient
- **Convergence:** Linear to superlinear
- **Restart:** Automatically restarts every n iterations

**Best For:**
- Large-scale problems with limited memory
- Smooth functions
- When Hessian approximation is too expensive

### Powell's Method

**Mathematical Foundation:**
Derivative-free method using successive line searches along conjugate directions.

**Algorithm:**
1. Start with coordinate directions
2. Perform line searches along each direction
3. Replace one direction with the overall displacement
4. Repeat until convergence

**Characteristics:**
- **Derivatives:** Not required
- **Convergence:** Superlinear for quadratic functions
- **Robustness:** Good for noisy or discontinuous functions

**Best For:**
- Functions where gradients are unavailable
- Noisy or discontinuous objectives
- Expensive function evaluations

### Nelder-Mead Simplex

**Mathematical Foundation:**
Maintains a simplex (n+1 points in n dimensions) and transforms it using reflection, expansion, contraction, and shrinkage.

**Operations:**
- **Reflection:** `x_r = x_c + α(x_c - x_h)`
- **Expansion:** `x_e = x_c + γ(x_r - x_c)`
- **Contraction:** `x_{cc} = x_c + ρ(x_h - x_c)`
- **Shrinkage:** `x_i = x_l + σ(x_i - x_l)`

where `x_l`, `x_h`, `x_c` are lowest, highest, and centroid points.

**Characteristics:**
- **Derivatives:** Not required
- **Robustness:** Very robust to noise
- **Scalability:** Poor for high dimensions (> 10-20)

**Best For:**
- Small-dimensional problems
- Noisy or discontinuous functions
- Initial exploration of unknown functions

## Constrained Optimization

### SLSQP (Sequential Least Squares Programming)

**Mathematical Foundation:**
Solves a sequence of quadratic programming subproblems.

**QP Subproblem:**
```
minimize: ½p^T H_k p + ∇f(x_k)^T p
subject to: ∇c_i(x_k)^T p + c_i(x_k) = 0  (equality)
           ∇c_j(x_k)^T p + c_j(x_k) ≥ 0  (inequality)
```

**Characteristics:**
- **Constraints:** Handles equality and inequality constraints
- **Convergence:** Superlinear under LICQ
- **Memory:** O(n²) for Hessian approximation

**Best For:**
- Problems with smooth constraints
- Medium-sized problems
- When constraint gradients are available

### Trust-Constr (Trust Region Constrained)

**Mathematical Foundation:**
Solves constrained subproblems within a trust region.

**Subproblem:**
```
minimize: f(x_k) + ∇f(x_k)^T p + ½p^T H_k p
subject to: ||p|| ≤ Δ_k
           c_i(x_k) + ∇c_i(x_k)^T p = 0
           c_j(x_k) + ∇c_j(x_k)^T p ≥ 0
```

**Characteristics:**
- **Global Convergence:** Guaranteed under mild conditions
- **Robustness:** Handles ill-conditioned problems well
- **Flexibility:** Adapts step size automatically

**Best For:**
- Nonlinear constraints
- Ill-conditioned problems
- When robustness is important

### Interior Point Method

**Mathematical Foundation:**
Converts inequality constraints to barrier functions.

**Barrier Problem:**
```
minimize: f(x) - μ Σ log(c_i(x))
subject to: h_j(x) = 0
```

**Characteristics:**
- **Path Following:** Follows central path as μ → 0
- **Convergence:** Polynomial complexity
- **Large Scale:** Efficient for large problems

**Best For:**
- Large-scale problems
- Many inequality constraints
- Linear programming relaxations

## Stochastic Optimization

### Stochastic Gradient Descent (SGD)

**Mathematical Foundation:**
Uses noisy gradient estimates to update parameters.

**Update Rule:**
```
x_{k+1} = x_k - α_k * ∇f_i(x_k)
```
where `∇f_i(x_k)` is the gradient for sample/batch `i`.

**Variants:**
- **Mini-batch SGD:** Uses batches of samples
- **SGD with Momentum:** Adds momentum term
- **Nesterov SGD:** Uses "look-ahead" gradients

**Characteristics:**
- **Memory:** O(n) - very efficient
- **Convergence:** Sublinear in general, linear under strong convexity
- **Noise Tolerance:** Naturally handles stochastic gradients

**Best For:**
- Machine learning problems
- Large datasets
- Online learning

### Adam (Adaptive Moment Estimation)

**Mathematical Foundation:**
Maintains running averages of gradients and their second moments.

**Update Rules:**
```
m_k = β₁ * m_{k-1} + (1 - β₁) * g_k
v_k = β₂ * v_{k-1} + (1 - β₂) * g_k²
m̂_k = m_k / (1 - β₁^k)
v̂_k = v_k / (1 - β₂^k)
x_{k+1} = x_k - α * m̂_k / (√v̂_k + ε)
```

**Hyperparameters:**
- `β₁ = 0.9` (first moment decay)
- `β₂ = 0.999` (second moment decay)
- `α = 0.001` (learning rate)
- `ε = 1e-8` (numerical stability)

**Characteristics:**
- **Adaptive:** Per-parameter learning rates
- **Bias Correction:** Corrects for initialization bias
- **Robustness:** Works well across many problems

**Best For:**
- Deep learning
- Sparse gradients
- Non-stationary objectives

### AdamW (Adam with Weight Decay)

**Mathematical Foundation:**
Decouples weight decay from gradient-based updates.

**Key Difference from Adam:**
```
// Standard Adam with L2 regularization
g_k = ∇f(x_k) + λ * x_k
x_{k+1} = x_k - α * m̂_k / (√v̂_k + ε)

// AdamW
x_{k+1} = (1 - α * λ) * x_k - α * m̂_k / (√v̂_k + ε)
```

**Benefits:**
- Better generalization than standard Adam
- Decoupled regularization strength
- More stable training

### RMSProp

**Mathematical Foundation:**
Adapts learning rate using moving average of squared gradients.

**Update Rules:**
```
v_k = β * v_{k-1} + (1 - β) * g_k²
x_{k+1} = x_k - α * g_k / (√v_k + ε)
```

**Characteristics:**
- **Adaptive:** Automatically scales learning rates
- **Memory Efficient:** Only stores second moment
- **Stability:** Good for non-convex problems

**Best For:**
- Recurrent neural networks
- Non-stationary problems
- When Adam is too complex

## Least Squares

### Levenberg-Marquardt

**Mathematical Foundation:**
Interpolates between Gauss-Newton and gradient descent.

**Update Rule:**
```
(J^T J + λI) p = -J^T r
x_{k+1} = x_k + p
```

**Damping Parameter:**
- Large `λ`: Behaves like gradient descent
- Small `λ`: Behaves like Gauss-Newton
- Adaptive: Increases `λ` on bad steps, decreases on good steps

**Characteristics:**
- **Robustness:** Handles ill-conditioned problems
- **Efficiency:** Fast convergence near minimum
- **Automatic:** Self-tuning damping parameter

**Best For:**
- Nonlinear least squares
- Curve fitting
- Parameter estimation

### Trust Region Reflective

**Mathematical Foundation:**
Solves trust region subproblems with bound constraints.

**Subproblem:**
```
minimize: ½||J_k p + r_k||²
subject to: ||p|| ≤ Δ_k
           l ≤ x_k + p ≤ u
```

**Key Features:**
- **Reflective:** Handles bounds by reflection
- **Trust Region:** Adaptive step size control
- **Robustness:** Global convergence guarantees

**Best For:**
- Bounded least squares problems
- Large residual problems
- When Levenberg-Marquardt fails

### Robust Least Squares

**Mathematical Foundation:**
Uses robust loss functions to reduce outlier influence.

**M-Estimators:**
```
minimize: Σ ρ(r_i / σ)
```

**Loss Functions:**
- **Huber:** `ρ(z) = z²/2` if `|z| ≤ c`, else `c|z| - c²/2`
- **Bisquare:** `ρ(z) = (c²/6)[1 - (1 - (z/c)²)³]` if `|z| ≤ c`, else `c²/6`
- **Cauchy:** `ρ(z) = (c²/2) log(1 + (z/c)²)`

**Characteristics:**
- **Outlier Resistance:** Reduces influence of bad data
- **Breakdown Point:** High for bisquare and Cauchy
- **Efficiency:** Good performance on clean data

**Best For:**
- Data with outliers
- Robust regression
- Sensor fusion problems

## Multi-Objective Optimization

### NSGA-II (Non-dominated Sorting Genetic Algorithm II)

**Mathematical Foundation:**
Uses non-dominated sorting and crowding distance for selection.

**Algorithm Steps:**
1. **Non-dominated Sorting:** Rank solutions by dominance
2. **Crowding Distance:** Maintain diversity
3. **Selection:** Choose based on rank and crowding distance
4. **Crossover/Mutation:** Generate offspring
5. **Elitism:** Combine parent and offspring populations

**Characteristics:**
- **Elitism:** Preserves best solutions
- **Diversity:** Maintains spread along Pareto front
- **Complexity:** O(MN²) where M is objectives, N is population

**Best For:**
- 2-3 objectives
- Moderate problem sizes
- When diversity is important

### NSGA-III

**Mathematical Foundation:**
Uses reference points for selection in many-objective problems.

**Key Differences from NSGA-II:**
- **Reference Points:** Structured or supplied
- **Niche Preservation:** Association with reference points
- **Selection:** Emphasizes convergence and diversity

**Characteristics:**
- **Scalability:** Handles many objectives (> 3)
- **Reference Points:** Can be uniform or problem-specific
- **Balance:** Better convergence-diversity trade-off

**Best For:**
- Many-objective problems (> 3 objectives)
- Engineering design
- When reference preferences exist

### Scalarization Methods

**Weighted Sum:**
```
minimize: Σ w_i * f_i(x)
subject to: constraints
```

**Chebyshev:**
```
minimize: max_i{w_i * (f_i(x) - z_i*)}
subject to: constraints
```

**Characteristics:**
- **Simplicity:** Easy to implement
- **Limitations:** Cannot find non-convex parts (weighted sum)
- **Preferences:** Requires weight specification

## Global Optimization

### Differential Evolution

**Mathematical Foundation:**
Population-based method using differential mutation.

**Mutation:**
```
v_{i,g+1} = x_{r1,g} + F * (x_{r2,g} - x_{r3,g})
```

**Crossover:**
```
u_{j,i,g+1} = v_{j,i,g+1} if rand ≤ CR or j = jrand
             x_{j,i,g}    otherwise
```

**Selection:**
```
x_{i,g+1} = u_{i,g+1} if f(u_{i,g+1}) ≤ f(x_{i,g})
           x_{i,g}    otherwise
```

**Characteristics:**
- **Parallel:** Population-based
- **Robust:** Works on many problem types
- **Parameters:** F (differential weight), CR (crossover rate)

**Best For:**
- Global optimization
- Multimodal problems
- Parallel computation

### Bayesian Optimization

**Mathematical Foundation:**
Uses Gaussian process surrogate model with acquisition functions.

**Components:**
1. **Surrogate Model:** Gaussian Process
2. **Acquisition Function:** Expected Improvement, UCB, etc.
3. **Optimization:** Maximize acquisition function

**Acquisition Functions:**
- **Expected Improvement:** `EI(x) = E[max(f_min - f(x), 0)]`
- **Upper Confidence Bound:** `UCB(x) = μ(x) + κσ(x)`
- **Probability of Improvement:** `PI(x) = P(f(x) < f_min)`

**Characteristics:**
- **Sample Efficiency:** Few function evaluations
- **Uncertainty:** Models uncertainty explicitly
- **Expensive Functions:** Optimal for costly evaluations

**Best For:**
- Expensive function evaluations
- Hyperparameter optimization
- Small to medium dimensions

### Basin-hopping

**Mathematical Foundation:**
Combines random perturbations with local minimization.

**Algorithm:**
1. **Local Minimization:** Find local minimum
2. **Random Perturbation:** Jump to new point
3. **Accept/Reject:** Metropolis-like criterion
4. **Repeat:** Until convergence

**Acceptance Criterion:**
```
accept if f_new < f_current
accept with probability exp(-(f_new - f_current)/T) otherwise
```

**Characteristics:**
- **Global Search:** Escapes local minima
- **Local Refinement:** Uses local optimization
- **Temperature:** Controls acceptance probability

**Best For:**
- Rugged energy landscapes
- Many local minima
- Physics-inspired problems

## Algorithm Comparison

### Convergence Rates

| Algorithm | Rate | Conditions |
|-----------|------|------------|
| Newton | Quadratic | Hessian available, near minimum |
| BFGS | Superlinear | Smooth function, good conditioning |
| L-BFGS | Superlinear | Large scale, limited memory |
| CG | Linear to Superlinear | Smooth function |
| SGD | Sublinear | General, Linear under strong convexity |
| Adam | Sublinear | General, often faster in practice |

### Memory Requirements

| Algorithm | Memory | Notes |
|-----------|--------|-------|
| Newton | O(n²) | Hessian storage |
| BFGS | O(n²) | Hessian approximation |
| L-BFGS | O(mn) | m vectors, typically m=5-20 |
| CG | O(n) | Only vectors |
| SGD | O(n) | Only parameters |
| Adam | O(n) | Moment estimates |

### Function Evaluation Requirements

| Algorithm | Function | Gradient | Hessian |
|-----------|----------|----------|---------|
| Newton | Yes | Yes | Yes |
| BFGS | Yes | Yes | No |
| L-BFGS | Yes | Yes | No |
| CG | Yes | Yes | No |
| Powell | Yes | No | No |
| Nelder-Mead | Yes | No | No |
| SGD | Yes | Yes | No |

### Problem Size Recommendations

| Problem Size | Recommended Algorithms |
|--------------|----------------------|
| Small (< 100) | BFGS, Newton, Nelder-Mead |
| Medium (100-1000) | BFGS, L-BFGS, CG |
| Large (1000-10⁶) | L-BFGS, CG, Adam |
| Very Large (> 10⁶) | SGD, Adam, mini-batch methods |

### Robustness Comparison

| Algorithm | Noise Tolerance | Ill-conditioning | Discontinuities |
|-----------|----------------|------------------|-----------------|
| BFGS | Low | Medium | Low |
| L-BFGS | Low | Medium | Low |
| CG | Low | High | Low |
| Powell | Medium | Medium | Medium |
| Nelder-Mead | High | Medium | High |
| SGD | High | Low | Medium |
| Adam | High | Medium | Medium |

This algorithm reference provides the mathematical foundations and practical guidance for choosing the appropriate optimization method for your specific problem. Each algorithm has been carefully implemented with numerical stability and efficiency in mind.