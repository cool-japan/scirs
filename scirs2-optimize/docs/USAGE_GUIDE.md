# scirs2-optimize Usage Guide

This guide provides practical examples and best practices for using the `scirs2-optimize` library effectively.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Choosing the Right Algorithm](#choosing-the-right-algorithm)
3. [Common Use Cases](#common-use-cases)
4. [Performance Optimization](#performance-optimization)
5. [Troubleshooting](#troubleshooting)
6. [Advanced Features](#advanced-features)
7. [Best Practices](#best-practices)

## Getting Started

### Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
scirs2_optimize = "0.1.0-alpha.6"
ndarray = "0.16"

# Optional: Enable parallelization
scirs2_optimize = { version = "0.1.0-alpha.6", features = ["parallel"] }
```

### Basic Example

```rust
use scirs2_optimize::prelude::*;
use ndarray::Array1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Define objective function: f(x, y) = x² + y²
    let objective = |x: &ArrayView1<f64>| -> f64 {
        x[0].powi(2) + x[1].powi(2)
    };
    
    // Starting point
    let x0 = Array1::from_vec(vec![1.0, 1.0]);
    
    // Minimize using BFGS
    let result = minimize(objective, &x0, Method::BFGS, None)?;
    
    println!("Minimum found at: {:?}", result.x);
    println!("Function value: {:.6}", result.fun);
    println!("Converged: {}", result.success);
    
    Ok(())
}
```

## Choosing the Right Algorithm

### Decision Tree

```
Is your problem...?

├─ Unconstrained?
│  ├─ Smooth function with cheap gradients?
│  │  ├─ Small problem (< 100 vars): BFGS
│  │  └─ Large problem (> 100 vars): L-BFGS
│  ├─ No gradients available?
│  │  ├─ Deterministic: Powell
│  │  └─ Noisy/discontinuous: Nelder-Mead
│  └─ Very large/stochastic?
│     └─ Use stochastic methods: Adam, SGD
│
├─ Has constraints?
│  ├─ Only bounds: Bounded methods or L-BFGS-B
│  ├─ Linear constraints: SLSQP
│  └─ Nonlinear constraints: Trust-Constr
│
├─ Multiple objectives?
│  ├─ Find Pareto front: NSGA-II/III
│  └─ Scalarize: Weighted sum
│
├─ Global optimum needed?
│  ├─ Continuous: Differential Evolution
│  ├─ Expensive function: Bayesian Optimization
│  └─ Many local minima: Basin-hopping
│
└─ Least squares problem?
   ├─ No outliers: Levenberg-Marquardt
   ├─ With outliers: Robust methods
   └─ Large residuals: Trust Region
```

### Algorithm Comparison

| Algorithm | Problem Type | Pros | Cons | Best For |
|-----------|--------------|------|------|----------|
| **BFGS** | Unconstrained | Fast convergence, robust | Needs gradients, O(n²) memory | Smooth functions, medium size |
| **L-BFGS** | Unconstrained | Low memory, scalable | Needs gradients | Large-scale problems |
| **Nelder-Mead** | Unconstrained | No gradients needed | Slow, not scalable | Noisy/discontinuous functions |
| **Powell** | Unconstrained | No gradients, deterministic | Can be slow | Expensive function evaluations |
| **Adam** | Unconstrained | Good for ML, handles noise | Hyperparameter sensitive | Machine learning, stochastic |
| **SLSQP** | Constrained | Handles mixed constraints | Needs gradients | General constrained problems |
| **Differential Evolution** | Global | Robust, parallel | Slow convergence | Multimodal problems |

## Common Use Cases

### 1. Function Fitting

Fit a parametric model to experimental data.

```rust
use scirs2_optimize::least_squares::*;
use ndarray::{Array1, Array2};

// Fit y = a*exp(b*x) + c to data
fn exponential_model(params: &[f64], data: &[f64]) -> Array1<f64> {
    let n = data.len() / 2;
    let x_data = &data[0..n];
    let y_data = &data[n..];
    
    let (a, b, c) = (params[0], params[1], params[2]);
    
    x_data.iter().zip(y_data.iter()).enumerate()
        .map(|(i, (&x, &y))| {
            let predicted = a * (b * x).exp() + c;
            y - predicted
        })
        .collect()
}

fn fit_exponential(x_data: &[f64], y_data: &[f64]) -> Result<Array1<f64>, OptimizeError> {
    // Combine data for residual function
    let data: Vec<f64> = x_data.iter().chain(y_data.iter()).cloned().collect();
    let data_array = Array1::from_vec(data);
    
    // Initial guess: a=1, b=1, c=0
    let x0 = Array1::from_vec(vec![1.0, 1.0, 0.0]);
    
    // Use robust fitting to handle outliers
    let huber_loss = HuberLoss::new(1.345);
    let result = robust_least_squares(
        exponential_model,
        &x0,
        huber_loss,
        None, // Auto-compute Jacobian
        &data_array,
        None
    )?;
    
    Ok(result.x)
}

// Usage
let x_data = vec![0.0, 1.0, 2.0, 3.0, 4.0];
let y_data = vec![1.2, 3.1, 7.8, 19.5, 48.2]; // With some noise
let params = fit_exponential(&x_data, &y_data)?;
println!("Fitted: y = {:.2}*exp({:.2}*x) + {:.2}", params[0], params[1], params[2]);
```

### 2. Neural Network Training

Train a simple neural network using stochastic optimization.

```rust
use scirs2_optimize::stochastic::*;
use ndarray::{Array1, Array2, ArrayView1};

struct SimpleNN {
    features: Array2<f64>,
    targets: Array1<f64>,
    architecture: Vec<usize>, // [input_size, hidden_size, output_size]
}

impl SimpleNN {
    fn predict(&self, weights: &ArrayView1<f64>, features: &ArrayView1<f64>) -> f64 {
        // Simple feedforward: input -> hidden -> output
        let (input_size, hidden_size) = (self.architecture[0], self.architecture[1]);
        
        // Extract weights
        let w1_end = input_size * hidden_size;
        let w1 = weights.slice(s![0..w1_end]);
        let b1 = weights.slice(s![w1_end..w1_end + hidden_size]);
        let w2 = weights.slice(s![w1_end + hidden_size..w1_end + hidden_size * 2]);
        let b2 = weights[weights.len() - 1];
        
        // Forward pass
        let mut hidden = Array1::zeros(hidden_size);
        for i in 0..hidden_size {
            for j in 0..input_size {
                hidden[i] += w1[i * input_size + j] * features[j];
            }
            hidden[i] += b1[i];
            hidden[i] = hidden[i].tanh(); // Activation
        }
        
        let mut output = b2;
        for i in 0..hidden_size {
            output += w2[i] * hidden[i];
        }
        
        output
    }
}

impl StochasticGradientFunction for SimpleNN {
    fn compute_gradient(&mut self, weights: &ArrayView1<f64>, batch_indices: &[f64]) -> Array1<f64> {
        let mut grad = Array1::zeros(weights.len());
        
        // Numerical gradient (for simplicity - use autodiff in practice)
        let h = 1e-7;
        for i in 0..weights.len() {
            let mut weights_plus = weights.to_owned();
            let mut weights_minus = weights.to_owned();
            weights_plus[i] += h;
            weights_minus[i] -= h;
            
            let loss_plus = self.compute_value(&weights_plus.view(), batch_indices);
            let loss_minus = self.compute_value(&weights_minus.view(), batch_indices);
            
            grad[i] = (loss_plus - loss_minus) / (2.0 * h);
        }
        
        grad
    }
    
    fn compute_value(&mut self, weights: &ArrayView1<f64>, batch_indices: &[f64]) -> f64 {
        let mut loss = 0.0;
        
        for &idx in batch_indices {
            let i = idx as usize % self.features.nrows();
            let features = self.features.row(i);
            let target = self.targets[i];
            
            let prediction = self.predict(weights, &features);
            loss += 0.5 * (prediction - target).powi(2); // MSE
        }
        
        loss / batch_indices.len() as f64
    }
}

fn train_neural_network(
    features: Array2<f64>, 
    targets: Array1<f64>
) -> Result<Array1<f64>, OptimizeError> {
    let architecture = vec![features.ncols(), 10, 1]; // Input -> 10 hidden -> 1 output
    let total_weights = architecture[0] * architecture[1] + // W1
                       architecture[1] +                  // b1
                       architecture[1] +                  // W2
                       1;                                 // b2
    
    let mut nn = SimpleNN { features, targets, architecture };
    
    // Random initialization
    let x0 = Array1::from_shape_simple_fn(total_weights, || {
        use rand::Rng;
        rand::rng().random_range(-0.1..0.1)
    });
    
    let data_provider = Box::new(InMemoryDataProvider::new(
        (0..nn.features.nrows()).map(|i| i as f64).collect()
    ));
    
    let options = AdamOptions {
        learning_rate: 0.001,
        max_iter: 1000,
        batch_size: Some(32),
        tol: 1e-6,
        ..Default::default()
    };
    
    let result = minimize_adam(nn, x0, data_provider, options)?;
    Ok(result.x)
}
```

### 3. Portfolio Optimization

Optimize asset allocation for a financial portfolio.

```rust
use scirs2_optimize::constrained::*;
use ndarray::{Array1, Array2};

struct Portfolio {
    expected_returns: Array1<f64>,
    covariance_matrix: Array2<f64>,
    risk_aversion: f64,
}

impl Portfolio {
    // Mean-variance utility: E[r] - (γ/2) * Var[r]
    fn utility(&self, weights: &ArrayView1<f64>) -> f64 {
        let expected_return = self.expected_returns.dot(weights);
        let variance = weights.dot(&self.covariance_matrix.dot(weights));
        expected_return - 0.5 * self.risk_aversion * variance
    }
    
    // Maximize utility = minimize negative utility
    fn negative_utility(&self, weights: &ArrayView1<f64>) -> f64 {
        -self.utility(weights)
    }
}

fn optimize_portfolio(
    expected_returns: Array1<f64>,
    covariance_matrix: Array2<f64>,
    risk_aversion: f64,
    min_weight: f64,
    max_weight: f64,
) -> Result<Array1<f64>, OptimizeError> {
    let n_assets = expected_returns.len();
    let portfolio = Portfolio { expected_returns, covariance_matrix, risk_aversion };
    
    // Objective: maximize utility = minimize negative utility
    let objective = move |w: &ArrayView1<f64>| portfolio.negative_utility(w);
    
    // Constraint: weights sum to 1
    let sum_constraint = |w: &ArrayView1<f64>| -> f64 {
        w.sum() - 1.0
    };
    
    let constraints = vec![
        Constraint {
            fun: Box::new(sum_constraint),
            constraint_type: ConstraintType::Equality,
            jac: None, // Auto-computed
        }
    ];
    
    // Bounds: min_weight <= w_i <= max_weight
    let bounds = Bounds::new(&vec![(Some(min_weight), Some(max_weight)); n_assets]);
    
    // Initial guess: equal weights
    let x0 = Array1::from_elem(n_assets, 1.0 / n_assets as f64);
    
    let mut options = Options::default();
    options.bounds = Some(bounds);
    
    let result = minimize_constrained(objective, &x0, &constraints, Method::SLSQP, Some(options))?;
    
    Ok(result.x)
}

// Usage
let returns = Array1::from_vec(vec![0.08, 0.12, 0.15, 0.10]); // Expected returns
let cov = Array2::from_shape_vec((4, 4), vec![
    0.04, 0.02, 0.01, 0.02,
    0.02, 0.09, 0.03, 0.01,
    0.01, 0.03, 0.16, 0.02,
    0.02, 0.01, 0.02, 0.06,
])?; // Covariance matrix

let optimal_weights = optimize_portfolio(returns, cov, 2.0, 0.0, 0.5)?;
println!("Optimal portfolio weights: {:?}", optimal_weights);
```

### 4. Hyperparameter Optimization

Optimize machine learning hyperparameters using Bayesian optimization.

```rust
use scirs2_optimize::global::*;

// Cross-validation objective for hyperparameter tuning
fn cross_validation_score(hyperparams: &ArrayView1<f64>) -> f64 {
    let learning_rate = hyperparams[0];
    let regularization = hyperparams[1];
    let hidden_size = hyperparams[2].round() as usize;
    
    // Simulate training and validation (replace with actual ML pipeline)
    let validation_error = simulate_training(learning_rate, regularization, hidden_size);
    
    validation_error // Minimize validation error
}

fn simulate_training(lr: f64, reg: f64, hidden: usize) -> f64 {
    // Simplified model: error decreases with better hyperparameters
    // In practice, this would train your actual model
    let base_error = 0.1;
    let lr_penalty = (lr - 0.001).abs() * 10.0;
    let reg_penalty = (reg - 0.01).abs() * 5.0;
    let size_penalty = ((hidden as f64 - 64.0) / 64.0).abs() * 0.02;
    
    base_error + lr_penalty + reg_penalty + size_penalty
}

fn optimize_hyperparameters() -> Result<Array1<f64>, OptimizeError> {
    // Define search space
    let bounds = vec![
        (1e-5, 1e-1),  // Learning rate
        (1e-5, 1e-1),  // Regularization
        (8.0, 256.0),  // Hidden size (will be rounded)
    ];
    
    let options = BayesianOptimizationOptions {
        n_initial: 10,     // Initial random samples
        n_iterations: 50,  // Total optimization iterations
        acquisition: AcquisitionFunctionType::ExpectedImprovement,
        kernel: KernelType::RBF { length_scale: 1.0 },
        noise_level: 1e-6,
        ..Default::default()
    };
    
    let result = bayesian_optimization(cross_validation_score, &bounds, Some(options))?;
    
    println!("Best hyperparameters found:");
    println!("  Learning rate: {:.2e}", result.x[0]);
    println!("  Regularization: {:.2e}", result.x[1]);
    println!("  Hidden size: {}", result.x[2].round() as usize);
    println!("  Validation error: {:.6}", result.fun);
    
    Ok(result.x)
}
```

## Performance Optimization

### 1. Providing Gradients

Always provide analytical gradients when possible:

```rust
use scirs2_optimize::unconstrained::*;

// Instead of relying on numerical gradients
let objective = |x: &ArrayView1<f64>| x[0].powi(2) + x[1].powi(2);

// Provide analytical gradient
let gradient = |x: &ArrayView1<f64>| Array1::from_vec(vec![2.0 * x[0], 2.0 * x[1]]);

let mut options = Options::default();
options.jac = Some(Box::new(gradient));

let result = minimize(objective, &x0, Method::BFGS, Some(options))?;
```

### 2. Using Automatic Differentiation

For complex functions, use automatic differentiation:

```rust
use scirs2_optimize::automatic_differentiation::*;

// Define function with dual numbers
let dual_func = |x: &ArrayView1<DualNumber>| -> DualNumber {
    x[0] * x[0] + x[1] * x[1] + (x[0] * x[1]).sin()
};

// Create gradient function automatically
let grad_func = create_ad_gradient(dual_func);

let mut options = Options::default();
options.jac = Some(Box::new(grad_func));
```

### 3. Parallelization

Enable parallel evaluation for expensive functions:

```rust
use scirs2_optimize::parallel::*;

// Enable parallel finite differences
let parallel_options = ParallelOptions {
    num_threads: Some(4),
    chunk_size: None, // Auto-determine
};

let gradient = parallel_finite_diff_gradient(expensive_function, &x, parallel_options)?;
```

### 4. Sparse Problems

For large sparse problems, use sparse numerical differentiation:

```rust
use scirs2_optimize::sparse_numdiff::*;

let sparse_options = SparseFiniteDiffOptions {
    rel_step: None,
    abs_step: 1e-8,
    bounds: None,
    sparsity: None, // Auto-detect sparsity pattern
};

let sparse_jacobian = sparse_jacobian(residual_func, &x, &sparse_options)?;
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Convergence Problems

**Symptoms:** Algorithm doesn't converge or converges to wrong solution.

**Solutions:**
```rust
// Try different starting points
let starting_points = vec![
    Array1::from_vec(vec![0.0, 0.0]),
    Array1::from_vec(vec![1.0, 1.0]),
    Array1::from_vec(vec![-1.0, 1.0]),
];

for x0 in starting_points {
    if let Ok(result) = minimize(func, &x0, Method::BFGS, None) {
        if result.success {
            println!("Converged from starting point {:?}", x0);
            break;
        }
    }
}

// Relax tolerances
let mut options = Options::default();
options.gtol = 1e-3;  // Less strict gradient tolerance
options.ftol = 1e-6;  // Less strict function tolerance
options.max_iter = 5000; // More iterations
```

#### 2. Numerical Instability

**Symptoms:** NaN values, infinite values, or erratic behavior.

**Solutions:**
```rust
// Add bounds to prevent extreme values
let bounds = Bounds::new(&[
    (Some(-10.0), Some(10.0)),
    (Some(-10.0), Some(10.0)),
]);

// Use more robust algorithm
let result = minimize(func, &x0, Method::TrustRegion, Some(options))?;

// Scale your problem
let scaled_func = |x: &ArrayView1<f64>| -> f64 {
    let scaled_x = x.mapv(|xi| xi * 0.1); // Scale inputs
    func(&scaled_x) * 1000.0 // Scale output
};
```

#### 3. Slow Convergence

**Symptoms:** Algorithm takes too many iterations.

**Solutions:**
```rust
// Use faster algorithm for your problem type
let result = match problem_characteristics {
    "smooth_unconstrained" => minimize(func, &x0, Method::BFGS, options),
    "large_scale" => minimize(func, &x0, Method::LBFGS, options),
    "no_gradients" => minimize(func, &x0, Method::Powell, options),
    _ => minimize(func, &x0, Method::BFGS, options),
}?;

// Provide better initial guess
let better_x0 = find_good_starting_point(func);
```

#### 4. Memory Issues

**Symptoms:** Out of memory errors for large problems.

**Solutions:**
```rust
// Use memory-efficient algorithms
let result = minimize(func, &x0, Method::LBFGS, options)?; // Instead of BFGS

// Use stochastic methods for very large problems
let result = minimize_adam(grad_func, x0, data_provider, adam_options)?;

// Enable sparse differentiation
use scirs2_optimize::sparse_numdiff::*;
```

### Debugging Tips

#### 1. Monitor Optimization Progress

```rust
let callback = |x: &ArrayView1<f64>, f_val: f64| -> bool {
    println!("Current point: {:?}, value: {:.6}", x, f_val);
    false // Continue optimization
};

let mut options = Options::default();
options.callback = Some(Box::new(callback));
```

#### 2. Validate Your Function

```rust
// Check for common issues
fn validate_function<F>(func: F, x: &ArrayView1<f64>) 
where F: Fn(&ArrayView1<f64>) -> f64 {
    let f_val = func(x);
    
    assert!(!f_val.is_nan(), "Function returns NaN at {:?}", x);
    assert!(!f_val.is_infinite(), "Function returns Inf at {:?}", x);
    
    // Check that small perturbations don't cause huge changes
    let eps = 1e-8;
    for i in 0..x.len() {
        let mut x_pert = x.to_owned();
        x_pert[i] += eps;
        let f_pert = func(&x_pert.view());
        let relative_change = (f_pert - f_val).abs() / f_val.abs().max(1e-10);
        
        if relative_change > 1e3 {
            println!("Warning: Function is very sensitive to parameter {}", i);
        }
    }
}
```

## Advanced Features

### 1. Custom Optimization Algorithms

Implement your own optimization algorithm:

```rust
use scirs2_optimize::unconstrained::*;

struct CustomOptimizer {
    max_iter: usize,
    tolerance: f64,
}

impl CustomOptimizer {
    fn minimize<F>(&self, func: F, x0: &ArrayView1<f64>) -> Result<OptimizeResult, OptimizeError>
    where F: Fn(&ArrayView1<f64>) -> f64 {
        let mut x = x0.to_owned();
        let mut f_val = func(&x.view());
        
        for iter in 0..self.max_iter {
            // Implement your algorithm here
            // This is a simple random search example
            use rand::Rng;
            let mut rng = rand::rng();
            
            let mut x_new = x.clone();
            for i in 0..x.len() {
                x_new[i] += rng.random_range(-0.1..0.1);
            }
            
            let f_new = func(&x_new.view());
            if f_new < f_val {
                x = x_new;
                f_val = f_new;
            }
            
            if f_val < self.tolerance {
                return Ok(OptimizeResult {
                    x,
                    fun: f_val,
                    success: true,
                    iterations: iter,
                    nit: iter,
                    func_evals: iter + 1,
                    nfev: iter + 1,
                    jacobian: None,
                    hessian: None,
                    message: "Custom algorithm converged".to_string(),
                });
            }
        }
        
        Ok(OptimizeResult {
            x,
            fun: f_val,
            success: false,
            iterations: self.max_iter,
            nit: self.max_iter,
            func_evals: self.max_iter,
            nfev: self.max_iter,
            jacobian: None,
            hessian: None,
            message: "Maximum iterations reached".to_string(),
        })
    }
}
```

### 2. Multi-Start Optimization

Find global optimum using multiple starting points:

```rust
use scirs2_optimize::global::*;

fn multi_start_optimization<F>(
    func: F,
    bounds: &[(f64, f64)],
    n_starts: usize,
) -> Result<OptimizeResult, OptimizeError>
where F: Fn(&ArrayView1<f64>) -> f64 + Clone {
    let mut best_result = None;
    let mut best_value = f64::INFINITY;
    
    for _ in 0..n_starts {
        // Generate random starting point
        use rand::Rng;
        let mut rng = rand::rng();
        let x0: Array1<f64> = bounds.iter()
            .map(|(min, max)| rng.random_range(*min..*max))
            .collect();
        
        // Local optimization from this starting point
        if let Ok(result) = minimize(func.clone(), &x0, Method::BFGS, None) {
            if result.success && result.fun < best_value {
                best_value = result.fun;
                best_result = Some(result);
            }
        }
    }
    
    best_result.ok_or_else(|| OptimizeError::ConvergenceError("No successful optimization".to_string()))
}
```

### 3. Constrained Global Optimization

Combine global search with constraint handling:

```rust
use scirs2_optimize::{global::*, constrained::*};

fn constrained_global_optimization<F, C>(
    objective: F,
    constraints: Vec<C>,
    bounds: &[(f64, f64)],
) -> Result<OptimizeResult, OptimizeError>
where 
    F: Fn(&ArrayView1<f64>) -> f64 + Clone,
    C: Fn(&ArrayView1<f64>) -> f64 + Clone,
{
    // Penalty method: add constraint violations to objective
    let penalty_func = move |x: &ArrayView1<f64>| -> f64 {
        let obj_val = objective(x);
        let penalty: f64 = constraints.iter()
            .map(|c| {
                let violation = c(x);
                if violation < 0.0 { // Constraint violated
                    1000.0 * violation.powi(2)
                } else {
                    0.0
                }
            })
            .sum();
        obj_val + penalty
    };
    
    // Use global optimization on penalized function
    differential_evolution(penalty_func, bounds, None)
}
```

## Best Practices

### 1. Problem Formulation

- **Scale your variables** to similar magnitudes (e.g., all between 0 and 1)
- **Choose appropriate bounds** to prevent numerical issues
- **Provide good initial guesses** when possible
- **Validate your objective function** for NaN/Inf values

### 2. Algorithm Selection

- **Start simple**: Try BFGS for smooth unconstrained problems
- **Provide gradients**: Analytical > automatic differentiation > numerical
- **Use appropriate method**: Match algorithm to problem characteristics
- **Consider problem size**: L-BFGS for large problems, stochastic for very large

### 3. Convergence Tuning

- **Adjust tolerances** based on your accuracy requirements
- **Set reasonable iteration limits**
- **Use callbacks** to monitor progress
- **Try multiple starting points** for global problems

### 4. Error Handling

```rust
match minimize(func, &x0, Method::BFGS, None) {
    Ok(result) => {
        if result.success {
            println!("Success: {:?}", result.x);
        } else {
            println!("Failed: {}", result.message);
            // Try different algorithm or parameters
        }
    }
    Err(e) => {
        eprintln!("Error: {}", e);
        // Handle specific error types
    }
}
```

### 5. Testing and Validation

- **Test on known problems** with analytical solutions
- **Validate gradients** using finite differences
- **Check convergence** from multiple starting points
- **Monitor objective function** behavior during optimization

This usage guide should help you effectively use the `scirs2-optimize` library for various optimization problems. For more specific questions, refer to the API documentation or the examples in the repository.