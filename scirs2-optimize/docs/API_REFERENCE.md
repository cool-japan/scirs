# scirs2-optimize API Reference

This document provides comprehensive API documentation for the `scirs2-optimize` library, a Rust implementation of SciPy's optimization functionality.

## Table of Contents

1. [Overview](#overview)
2. [Module Structure](#module-structure)
3. [Unconstrained Optimization](#unconstrained-optimization)
4. [Constrained Optimization](#constrained-optimization)
5. [Stochastic Optimization](#stochastic-optimization)
6. [Least Squares](#least-squares)
7. [Multi-Objective Optimization](#multi-objective-optimization)
8. [Global Optimization](#global-optimization)
9. [Automatic Differentiation](#automatic-differentiation)
10. [Root Finding](#root-finding)
11. [Scalar Optimization](#scalar-optimization)
12. [Error Handling](#error-handling)
13. [Performance Considerations](#performance-considerations)
14. [Examples and Use Cases](#examples-and-use-cases)

## Overview

`scirs2-optimize` provides a comprehensive suite of optimization algorithms for Rust applications. The library is designed to be:

- **Ergonomic**: Easy-to-use APIs with sensible defaults
- **Fast**: Optimized implementations with optional parallelization
- **Robust**: Handles edge cases and numerical stability issues
- **Comprehensive**: Covers most optimization use cases

### Core Features

- **Unconstrained optimization** with BFGS, L-BFGS, Nelder-Mead, Powell, and Conjugate Gradient
- **Constrained optimization** with SLSQP, Trust-Constr, COBYLA, and Interior Point methods
- **Stochastic optimization** with SGD, Adam, RMSProp, AdamW, and momentum variants
- **Least squares** with Levenberg-Marquardt, robust methods, and bounded constraints
- **Multi-objective optimization** with NSGA-II, NSGA-III, and scalarization
- **Global optimization** with Differential Evolution, Basin-hopping, and Bayesian optimization
- **Automatic differentiation** for gradient and Hessian computation
- **Root finding** with hybrid methods, Anderson acceleration, and Krylov methods

## Module Structure

```rust
scirs2_optimize/
├── unconstrained/       # Unconstrained optimization algorithms
├── constrained/         # Constrained optimization algorithms  
├── stochastic/         # Stochastic optimization for ML/large-scale
├── least_squares/      # Nonlinear least squares solvers
├── multi_objective/    # Multi-objective optimization
├── global/            # Global optimization algorithms
├── automatic_differentiation/  # AD for gradients/Hessians
├── roots/             # Root finding algorithms
├── scalar/           # Scalar (1D) optimization
├── sparse_numdiff/   # Sparse numerical differentiation
├── parallel/         # Parallel evaluation utilities
└── error/           # Error types and handling
```

## Unconstrained Optimization

### Core Functions

#### `minimize`
The primary entry point for unconstrained optimization.

```rust
pub fn minimize<F, S>(
    func: F,
    x0: &ArrayView1<f64>,
    method: Method,
    options: Option<Options>,
) -> Result<OptimizeResult, OptimizeError>
where
    F: Fn(&ArrayView1<f64>) -> f64,
    S: AsRef<str>,
```

**Parameters:**
- `func`: Objective function to minimize
- `x0`: Initial guess (starting point)
- `method`: Optimization algorithm to use
- `options`: Optional configuration parameters

**Returns:** `Result<OptimizeResult, OptimizeError>`

### Available Methods

```rust
pub enum Method {
    BFGS,           // Quasi-Newton method with BFGS updates
    LBFGS,          // Limited-memory BFGS for large problems
    Newton,         // Newton's method (requires Hessian)
    CG,             // Nonlinear Conjugate Gradient
    Powell,         // Powell's derivative-free method
    NelderMead,     // Nelder-Mead simplex algorithm
    TrustRegion,    // Trust region method
}
```

### Options Configuration

```rust
pub struct Options {
    pub max_iter: usize,           // Maximum iterations (default: 1000)
    pub gtol: f64,                 // Gradient tolerance (default: 1e-5)
    pub ftol: f64,                 // Function tolerance (default: 2.22e-9)
    pub xtol: f64,                 // Parameter tolerance (default: 1e-8)
    pub bounds: Option<Bounds>,    // Variable bounds
    pub callback: Option<Box<dyn Fn(&ArrayView1<f64>, f64) -> bool>>,
    pub jac: Option<Box<dyn Fn(&ArrayView1<f64>) -> Array1<f64>>>,
    pub hess: Option<Box<dyn Fn(&ArrayView1<f64>) -> Array2<f64>>>,
}
```

### Bounds Handling

```rust
// Create bounds for variables
let bounds = Bounds::new(&[
    (Some(0.0), Some(1.0)),  // 0 <= x[0] <= 1
    (Some(-1.0), None),      // x[1] >= -1
    (None, Some(10.0)),      // x[2] <= 10
    (None, None)             // x[3] unbounded
]);

let mut options = Options::default();
options.bounds = Some(bounds);
```

### Example Usage

```rust
use scirs2_optimize::unconstrained::{minimize, Method, Options};
use ndarray::Array1;

// Rosenbrock function
let rosenbrock = |x: &ArrayView1<f64>| -> f64 {
    let (a, b) = (1.0, 100.0);
    (a - x[0]).powi(2) + b * (x[1] - x[0].powi(2)).powi(2)
};

let x0 = Array1::from_vec(vec![-1.2, 1.0]);
let result = minimize(rosenbrock, &x0, Method::BFGS, None)?;

println!("Solution: {:?}", result.x);      // Optimal point
println!("Value: {:.6}", result.fun);      // Function value
println!("Iterations: {}", result.nit);    // Number of iterations
println!("Success: {}", result.success);   // Convergence flag
```

## Constrained Optimization

### Core Functions

#### `minimize_constrained`
Main entry point for constrained optimization problems.

```rust
pub fn minimize_constrained<F>(
    func: F,
    x0: &ArrayView1<f64>,
    constraints: &[Constraint],
    method: Method,
    options: Option<Options>,
) -> Result<OptimizeResult, OptimizeError>
```

### Constraint Types

```rust
pub struct Constraint {
    pub fun: Box<dyn Fn(&ArrayView1<f64>) -> f64>,
    pub constraint_type: ConstraintType,
    pub jac: Option<Box<dyn Fn(&ArrayView1<f64>) -> Array1<f64>>>,
}

pub enum ConstraintType {
    Equality,     // c(x) = 0
    Inequality,   // c(x) >= 0
}
```

### Available Methods

```rust
pub enum Method {
    SLSQP,        // Sequential Least Squares Programming
    TrustConstr,  // Trust-region constrained algorithm
    COBYLA,       // Constrained Optimization BY Linear Approximations
    InteriorPoint, // Interior point method
}
```

### Example Usage

```rust
use scirs2_optimize::constrained::{minimize_constrained, Constraint, ConstraintType, Method};

// Objective: minimize (x-1)² + (y-2.5)²
let objective = |x: &ArrayView1<f64>| -> f64 {
    (x[0] - 1.0).powi(2) + (x[1] - 2.5).powi(2)
};

// Constraint: x + y <= 3 (rewritten as 3 - x - y >= 0)
let constraint = |x: &ArrayView1<f64>| -> f64 {
    3.0 - x[0] - x[1]
};

let constraints = vec![Constraint {
    fun: Box::new(constraint),
    constraint_type: ConstraintType::Inequality,
    jac: None, // Auto-computed if not provided
}];

let x0 = Array1::from_vec(vec![0.0, 0.0]);
let result = minimize_constrained(objective, &x0, &constraints, Method::SLSQP, None)?;
```

## Stochastic Optimization

Stochastic optimization is essential for machine learning and large-scale problems where exact gradients are expensive or noisy.

### Core Functions

All stochastic optimizers follow this pattern:

```rust
pub fn minimize_<algorithm><F>(
    grad_func: F,
    x0: Array1<f64>,
    data_provider: Box<dyn DataProvider>,
    options: <Algorithm>Options,
) -> Result<OptimizeResult, OptimizeError>
where
    F: StochasticGradientFunction,
```

### Available Algorithms

```rust
// Basic SGD
minimize_sgd(grad_func, x0, data_provider, SGDOptions::default())?;

// SGD with Momentum
minimize_sgd_momentum(grad_func, x0, data_provider, MomentumOptions::default())?;

// Adam optimizer
minimize_adam(grad_func, x0, data_provider, AdamOptions::default())?;

// AdamW with weight decay
minimize_adamw(grad_func, x0, data_provider, AdamWOptions::default())?;

// RMSProp
minimize_rmsprop(grad_func, x0, data_provider, RMSPropOptions::default())?;
```

### Gradient Function Trait

```rust
pub trait StochasticGradientFunction {
    fn compute_gradient(&mut self, x: &ArrayView1<f64>, batch_data: &[f64]) -> Array1<f64>;
    fn compute_value(&mut self, x: &ArrayView1<f64>, batch_data: &[f64]) -> f64;
}
```

### Data Provider Trait

```rust
pub trait DataProvider {
    fn num_samples(&self) -> usize;
    fn get_batch(&self, indices: &[usize]) -> Vec<f64>;
    fn get_full_data(&self) -> Vec<f64>;
}

// Simple in-memory implementation
let data_provider = Box::new(InMemoryDataProvider::new(training_data));
```

### Algorithm Options

#### SGD Options
```rust
pub struct SGDOptions {
    pub learning_rate: f64,        // Step size (default: 0.01)
    pub max_iter: usize,           // Maximum epochs (default: 1000)
    pub batch_size: Option<usize>, // Mini-batch size
    pub tol: f64,                  // Convergence tolerance
    pub lr_schedule: LearningRateSchedule, // Learning rate decay
    pub gradient_clip: Option<f64>, // Gradient clipping threshold
}
```

#### Adam Options
```rust
pub struct AdamOptions {
    pub learning_rate: f64,   // Step size (default: 0.001)
    pub beta1: f64,          // First moment decay (default: 0.9)
    pub beta2: f64,          // Second moment decay (default: 0.999)
    pub epsilon: f64,        // Numerical stability (default: 1e-8)
    pub max_iter: usize,     // Maximum epochs
    pub amsgrad: bool,       // Use AMSGrad variant
    // ... other options
}
```

### Learning Rate Schedules

```rust
pub enum LearningRateSchedule {
    Constant,
    ExponentialDecay { decay_rate: f64 },
    StepDecay { decay_factor: f64, decay_steps: usize },
    LinearDecay,
    CosineAnnealing,
    InverseTimeDecay { decay_rate: f64 },
}
```

### Example Usage

```rust
use scirs2_optimize::stochastic::*;

// Define a quadratic function for testing
struct QuadraticFunction;

impl StochasticGradientFunction for QuadraticFunction {
    fn compute_gradient(&mut self, x: &ArrayView1<f64>, _batch_data: &[f64]) -> Array1<f64> {
        x.mapv(|xi| 2.0 * xi) // Gradient of sum(x_i^2)
    }
    
    fn compute_value(&mut self, x: &ArrayView1<f64>, _batch_data: &[f64]) -> f64 {
        x.mapv(|xi| xi * xi).sum() // sum(x_i^2)
    }
}

// Setup
let grad_func = QuadraticFunction;
let x0 = Array1::from_vec(vec![1.0, -1.0]);
let data_provider = Box::new(InMemoryDataProvider::new(vec![1.0; 100]));

// Adam optimization
let options = AdamOptions {
    learning_rate: 0.1,
    max_iter: 1000,
    tol: 1e-6,
    ..Default::default()
};

let result = minimize_adam(grad_func, x0, data_provider, options)?;
```

## Least Squares

Specialized algorithms for nonlinear least squares problems.

### Core Functions

```rust
// Basic least squares
pub fn least_squares<F, J>(
    residual: F,
    x0: &ArrayView1<f64>,
    method: Method,
    jacobian: Option<J>,
    data: &ArrayView1<f64>,
    options: Option<Options>,
) -> Result<OptimizeResult, OptimizeError>

// Robust least squares (outlier-resistant)
pub fn robust_least_squares<F, J, L>(
    residual: F,
    x0: &ArrayView1<f64>,
    loss: L,
    jacobian: Option<J>,
    data: &ArrayView1<f64>,
    options: Option<RobustOptions>,
) -> Result<OptimizeResult, OptimizeError>

// Bounded least squares
pub fn bounded_least_squares<F, J>(
    residual: F,
    x0: &ArrayView1<f64>,
    bounds: &Bounds,
    jacobian: Option<J>,
    data: &ArrayView1<f64>,
    options: Option<BoundedOptions>,
) -> Result<OptimizeResult, OptimizeError>
```

### Available Methods

```rust
pub enum Method {
    LevenbergMarquardt,  // Levenberg-Marquardt algorithm
    TrustRegionReflective, // Trust-region reflective
    Dogbox,              // Dogleg with box constraints
}
```

### Robust Loss Functions

```rust
// Huber loss (reduces influence of moderate outliers)
let huber_loss = HuberLoss::new(1.345); // threshold parameter

// Bisquare loss (completely rejects extreme outliers)
let bisquare_loss = BisquareLoss::new(4.685);

// Cauchy loss (very strong outlier resistance)
let cauchy_loss = CauchyLoss::new(2.385);
```

### Example Usage

```rust
use scirs2_optimize::least_squares::*;

// Define residual function for curve fitting
fn residual(params: &[f64], data: &[f64]) -> Array1<f64> {
    let n = data.len() / 2;
    let x_vals = &data[0..n];
    let y_vals = &data[n..];
    
    let mut residuals = Array1::zeros(n);
    for i in 0..n {
        let predicted = params[0] * x_vals[i].exp() + params[1];
        residuals[i] = y_vals[i] - predicted;
    }
    residuals
}

// Fit exponential model: y = a*exp(x) + b
let x0 = Array1::from_vec(vec![1.0, 0.0]); // Initial guess for [a, b]
let data = Array1::from_vec(vec![0.0, 1.0, 2.0, 0.1, 2.7, 7.4]); // [x_vals, y_vals]

let result = least_squares(
    residual,
    &x0,
    Method::LevenbergMarquardt,
    None, // Auto-compute Jacobian
    &data,
    None
)?;

println!("Fitted parameters: a={:.3}, b={:.3}", result.x[0], result.x[1]);
```

## Multi-Objective Optimization

For problems with multiple conflicting objectives.

### Core Functions

```rust
// NSGA-II algorithm
pub fn nsga_ii<F>(
    objectives: Vec<F>,
    bounds: &[(f64, f64)],
    config: MultiObjectiveConfig,
) -> Result<MultiObjectiveResult, OptimizeError>

// NSGA-III algorithm  
pub fn nsga_iii<F>(
    objectives: Vec<F>,
    bounds: &[(f64, f64)],
    config: MultiObjectiveConfig,
) -> Result<MultiObjectiveResult, OptimizeError>

// Scalarization approach
pub fn scalarization<F>(
    objectives: Vec<F>,
    weights: &[f64],
    bounds: &[(f64, f64)],
    method: ScalarizationMethod,
) -> Result<OptimizeResult, OptimizeError>
```

### Configuration

```rust
pub struct MultiObjectiveConfig {
    pub population_size: usize,     // Population size (default: 100)
    pub max_generations: usize,     // Maximum generations (default: 1000)
    pub crossover_rate: f64,        // Crossover probability (default: 0.9)
    pub mutation_rate: f64,         // Mutation probability (default: 0.1)
    pub seed: Option<u64>,          // Random seed for reproducibility
}

pub enum ScalarizationMethod {
    WeightedSum,        // Linear combination of objectives
    Chebyshev,          // Chebyshev scalarization
    AugmentedChebyshev, // Augmented Chebyshev method
}
```

### Example Usage

```rust
use scirs2_optimize::multi_objective::*;

// Define two conflicting objectives
let obj1 = |x: &[f64]| x[0].powi(2) + x[1].powi(2);           // Minimize distance from origin
let obj2 = |x: &[f64]| (x[0] - 1.0).powi(2) + (x[1] - 1.0).powi(2); // Minimize distance from (1,1)

let objectives = vec![
    Box::new(obj1) as Box<dyn Fn(&[f64]) -> f64>,
    Box::new(obj2) as Box<dyn Fn(&[f64]) -> f64>,
];

let bounds = vec![(-2.0, 2.0), (-2.0, 2.0)]; // Variable bounds

let config = MultiObjectiveConfig {
    population_size: 100,
    max_generations: 200,
    ..Default::default()
};

let result = nsga_ii(objectives, &bounds, config)?;

// Access Pareto front
for solution in &result.pareto_front {
    println!("Solution: {:?}, Objectives: {:?}", 
             solution.variables, solution.objectives);
}
```

## Global Optimization

For finding global optima in multimodal problems.

### Available Algorithms

```rust
// Differential Evolution
pub fn differential_evolution<F>(
    func: F,
    bounds: &[(f64, f64)],
    options: Option<DifferentialEvolutionOptions>,
) -> Result<OptimizeResult, OptimizeError>

// Basin-hopping
pub fn basinhopping<F>(
    func: F,
    x0: &ArrayView1<f64>,
    options: Option<BasinHoppingOptions>,
) -> Result<OptimizeResult, OptimizeError>

// Simulated Annealing
pub fn simulated_annealing<F>(
    func: F,
    bounds: &[(f64, f64)],
    options: Option<SimulatedAnnealingOptions>,
) -> Result<OptimizeResult, OptimizeError>

// Bayesian Optimization
pub fn bayesian_optimization<F>(
    func: F,
    bounds: &[(f64, f64)],
    options: Option<BayesianOptimizationOptions>,
) -> Result<OptimizeResult, OptimizeError>
```

### Example Usage

```rust
use scirs2_optimize::global::*;

// Rastrigin function (multimodal test function)
let rastrigin = |x: &ArrayView1<f64>| -> f64 {
    let a = 10.0;
    let n = x.len() as f64;
    a * n + x.iter().map(|&xi| xi.powi(2) - a * (2.0 * std::f64::consts::PI * xi).cos()).sum::<f64>()
};

let bounds = vec![(-5.12, 5.12); 2]; // 2D problem with bounds

// Differential Evolution
let de_options = DifferentialEvolutionOptions {
    population_size: 45,
    max_iter: 1000,
    f: 0.8,      // Differential weight
    cr: 0.9,     // Crossover probability
    ..Default::default()
};

let result = differential_evolution(rastrigin, &bounds, Some(de_options))?;
println!("Global minimum: {:?} at {:?}", result.fun, result.x);
```

## Automatic Differentiation

Compute gradients and Hessians automatically.

### Core Functions

```rust
// Compute gradient using automatic differentiation
pub fn create_ad_gradient<F>(func: F) -> impl Fn(&ArrayView1<f64>) -> Array1<f64>
where F: Fn(&ArrayView1<DualNumber>) -> DualNumber

// Compute Hessian using automatic differentiation
pub fn create_ad_hessian<F>(func: F) -> impl Fn(&ArrayView1<f64>) -> Array2<f64>
where F: Fn(&ArrayView1<DualNumber>) -> DualNumber

// Full automatic differentiation
pub fn autodiff<F>(
    func: F,
    x: &ArrayView1<f64>,
    mode: ADMode,
    options: AutoDiffOptions,
) -> ADResult
```

### AD Modes

```rust
pub enum ADMode {
    Forward,  // Forward-mode AD (efficient for few inputs)
    Reverse,  // Reverse-mode AD (efficient for few outputs)
    Auto,     // Automatically choose best mode
}
```

### Example Usage

```rust
use scirs2_optimize::automatic_differentiation::*;

// Define function using dual numbers
let func = |x: &ArrayView1<DualNumber>| -> DualNumber {
    x[0] * x[0] + x[1] * x[1] + x[0] * x[1]
};

// Create gradient function
let grad_func = create_ad_gradient(func);

// Evaluate gradient at point
let x = Array1::from_vec(vec![1.0, 2.0]);
let gradient = grad_func(&x);
println!("Gradient: {:?}", gradient); // [4.0, 5.0]

// Use with optimization
let objective = |x: &ArrayView1<f64>| -> f64 {
    x[0] * x[0] + x[1] * x[1] + x[0] * x[1]
};

let mut options = Options::default();
options.jac = Some(Box::new(grad_func));

let result = minimize(objective, &x, Method::BFGS, Some(options))?;
```

## Root Finding

Find zeros of nonlinear functions.

### Core Functions

```rust
pub fn root<F, J>(
    func: F,
    x0: &ArrayView1<f64>,
    method: Method,
    jacobian: Option<J>,
    options: Option<Options>,
) -> Result<OptimizeResult, OptimizeError>
```

### Available Methods

```rust
pub enum Method {
    Hybr,        // Hybrid method (modified Powell)
    Broyden1,    // Broyden's "good" method  
    Broyden2,    // Broyden's "bad" method
    Anderson,    // Anderson acceleration
    Krylov,      // Krylov subspace methods (GMRES)
}
```

### Example Usage

```rust
use scirs2_optimize::roots::*;

// System of equations: [x² + y² - 1, x - y] = 0
let system = |x: &ArrayView1<f64>| -> Array1<f64> {
    Array1::from_vec(vec![
        x[0].powi(2) + x[1].powi(2) - 1.0,  // Circle: x² + y² = 1
        x[0] - x[1]                          // Line: x = y
    ])
};

let x0 = Array1::from_vec(vec![1.0, 1.0]); // Initial guess
let result = root(system, &x0, Method::Hybr, None, None)?;

println!("Root found: {:?}", result.x);      // Should be close to [√2/2, √2/2]
println!("Function value: {:?}", result.fun); // Should be close to [0, 0]
```

## Error Handling

### Error Types

```rust
#[derive(Debug, thiserror::Error)]
pub enum OptimizeError {
    #[error("Convergence failed: {0}")]
    ConvergenceError(String),
    
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    
    #[error("Numerical error: {0}")]
    NumericalError(String),
    
    #[error("Function evaluation failed: {0}")]
    FunctionEvaluationError(String),
    
    #[error("Linear algebra error: {0}")]
    LinalgError(String),
}
```

### Result Types

```rust
pub struct OptimizeResult {
    pub x: Array1<f64>,        // Solution point
    pub fun: f64,              // Function value at solution
    pub success: bool,         // Whether optimization succeeded
    pub iterations: usize,     // Number of iterations
    pub nit: usize,           // Alias for iterations
    pub func_evals: usize,    // Number of function evaluations
    pub nfev: usize,          // Alias for func_evals
    pub jacobian: Option<Array1<f64>>, // Final gradient
    pub hessian: Option<Array2<f64>>,  // Final Hessian
    pub message: String,      // Status message
}
```

### Error Handling Patterns

```rust
use scirs2_optimize::{OptimizeError, OptimizeResult};

match minimize(func, &x0, Method::BFGS, None) {
    Ok(result) => {
        if result.success {
            println!("Optimization succeeded: {:?}", result.x);
        } else {
            println!("Optimization failed: {}", result.message);
        }
    }
    Err(OptimizeError::ConvergenceError(msg)) => {
        println!("Convergence failed: {}", msg);
    }
    Err(OptimizeError::InvalidInput(msg)) => {
        println!("Invalid input: {}", msg);
    }
    Err(e) => {
        println!("Other error: {}", e);
    }
}
```

## Performance Considerations

### Choosing the Right Algorithm

**Problem Size:**
- Small problems (< 100 variables): Any method works well
- Medium problems (100-1000 variables): BFGS, L-BFGS, CG
- Large problems (> 1000 variables): L-BFGS, CG, stochastic methods

**Function Properties:**
- Smooth, well-conditioned: Newton, BFGS
- Non-smooth or noisy: Nelder-Mead, Powell
- Expensive function evaluations: BFGS (fewer evaluations)
- Cheap function evaluations: Nelder-Mead (more robust)

**Constraints:**
- No constraints: Unconstrained methods
- Simple bounds: Bounded versions or projection methods  
- General constraints: SLSQP, Trust-Constr, Interior Point

### Parallelization

Enable parallel features for better performance:

```toml
[dependencies]
scirs2_optimize = { version = "0.1.0-alpha.4", features = ["parallel"] }
```

```rust
use scirs2_optimize::parallel::*;

// Parallel function evaluation
let results = parallel_evaluate_batch(&functions, &points, ParallelOptions::default())?;

// Parallel finite difference gradients
let gradient = parallel_finite_diff_gradient(func, &x, ParallelOptions::default())?;
```

### Memory Efficiency

For large-scale problems:
- Use L-BFGS instead of BFGS (limited memory)
- Enable sparse numerical differentiation
- Use stochastic methods for very large datasets

```rust
// Sparse gradient computation
use scirs2_optimize::sparse_numdiff::*;

let options = SparseFiniteDiffOptions {
    rel_step: None,
    abs_step: 1e-8,
    bounds: None,
    sparsity: None, // Auto-detect sparsity pattern
};

let sparse_jac = sparse_jacobian(func, &x, &options)?;
```

### JIT Optimization

For performance-critical code with repeated function evaluations:

```rust
use scirs2_optimize::jit_optimization::*;

// JIT compile optimization patterns
let jit_options = JitOptions {
    enable_simd: true,
    enable_parallel: true,
    cache_size: 1000,
    optimization_level: 3,
};

let optimized_func = optimize_function(func, FunctionPattern::Quadratic, jit_options)?;
```

### SIMD Acceleration

For numerical operations on large arrays:

```rust
use scirs2_optimize::simd_ops::*;

// SIMD-accelerated vector operations
let result = simd_dot_product(&vec1, &vec2);
let norm = simd_l2_norm(&vector);
let scaled = simd_vector_scale(&vector, scale_factor);
```

## Examples and Use Cases

### Machine Learning: Logistic Regression

```rust
use scirs2_optimize::stochastic::*;

struct LogisticRegression {
    features: Array2<f64>,
    labels: Array1<f64>,
}

impl StochasticGradientFunction for LogisticRegression {
    fn compute_gradient(&mut self, params: &ArrayView1<f64>, batch_indices: &[f64]) -> Array1<f64> {
        let mut grad = Array1::zeros(params.len());
        
        for &idx in batch_indices {
            let i = idx as usize;
            let x = self.features.row(i);
            let y = self.labels[i];
            
            let z = x.dot(params);
            let pred = 1.0 / (1.0 + (-z).exp());
            let error = pred - y;
            
            for j in 0..params.len() {
                grad[j] += error * x[j];
            }
        }
        
        grad / batch_indices.len() as f64
    }
    
    fn compute_value(&mut self, params: &ArrayView1<f64>, batch_indices: &[f64]) -> f64 {
        let mut loss = 0.0;
        
        for &idx in batch_indices {
            let i = idx as usize;
            let x = self.features.row(i);
            let y = self.labels[i];
            
            let z = x.dot(params);
            let pred = 1.0 / (1.0 + (-z).exp());
            
            loss += -y * pred.ln() - (1.0 - y) * (1.0 - pred).ln();
        }
        
        loss / batch_indices.len() as f64
    }
}

// Training
let logreg = LogisticRegression { features, labels };
let x0 = Array1::zeros(num_features);
let data_provider = Box::new(InMemoryDataProvider::new((0..num_samples).map(|i| i as f64).collect()));

let options = AdamOptions {
    learning_rate: 0.01,
    max_iter: 1000,
    batch_size: Some(32),
    ..Default::default()
};

let result = minimize_adam(logreg, x0, data_provider, options)?;
println!("Trained weights: {:?}", result.x);
```

### Engineering: Parameter Estimation

```rust
use scirs2_optimize::least_squares::*;

// Fit exponential decay model: y = A * exp(-k * t) + C
fn exponential_model(params: &[f64], data: &[f64]) -> Array1<f64> {
    let n = data.len() / 2;
    let t_vals = &data[0..n];
    let y_vals = &data[n..];
    
    let (a, k, c) = (params[0], params[1], params[2]);
    
    let mut residuals = Array1::zeros(n);
    for i in 0..n {
        let predicted = a * (-k * t_vals[i]).exp() + c;
        residuals[i] = y_vals[i] - predicted;
    }
    residuals
}

// Experimental data
let time_data = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
let measurements = Array1::from_vec(vec![10.0, 6.7, 4.5, 3.0, 2.0, 1.4]);
let data = Array1::from_iter(time_data.iter().chain(measurements.iter()).cloned());

// Initial guess: A=10, k=0.5, C=1
let x0 = Array1::from_vec(vec![10.0, 0.5, 1.0]);

let result = least_squares(
    exponential_model,
    &x0,
    Method::LevenbergMarquardt,
    None,
    &data,
    None
)?;

println!("Fitted parameters: A={:.3}, k={:.3}, C={:.3}", 
         result.x[0], result.x[1], result.x[2]);
```

### Finance: Portfolio Optimization

```rust
use scirs2_optimize::constrained::*;

// Mean-variance portfolio optimization
fn portfolio_objective(weights: &ArrayView1<f64>) -> f64 {
    // Minimize portfolio variance (maximize return via constraints)
    let covariance = get_covariance_matrix(); // Asset covariance matrix
    weights.dot(&covariance.dot(weights))
}

// Constraints: weights sum to 1, all weights >= 0
let sum_constraint = |w: &ArrayView1<f64>| -> f64 {
    w.sum() - 1.0 // w.sum() = 1
};

let constraints = vec![
    Constraint {
        fun: Box::new(sum_constraint),
        constraint_type: ConstraintType::Equality,
        jac: None,
    }
];

// Bounds: 0 <= weight <= 0.4 (max 40% in any asset)
let bounds = Bounds::new(&vec![(Some(0.0), Some(0.4)); num_assets]);

let x0 = Array1::from_elem(num_assets, 1.0 / num_assets as f64); // Equal weights

let mut options = Options::default();
options.bounds = Some(bounds);

let result = minimize_constrained(
    portfolio_objective,
    &x0,
    &constraints,
    Method::SLSQP,
    Some(options)
)?;

println!("Optimal portfolio weights: {:?}", result.x);
```

This comprehensive API reference covers all major functionality of the `scirs2-optimize` library. For more specific examples and advanced usage patterns, refer to the individual module documentation and the examples directory in the repository.