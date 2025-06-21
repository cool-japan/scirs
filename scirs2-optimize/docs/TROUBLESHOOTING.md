# Troubleshooting Guide

This guide helps you diagnose and resolve common issues when using the `scirs2-optimize` library.

## Table of Contents

1. [Common Issues](#common-issues)
2. [Convergence Problems](#convergence-problems)
3. [Performance Issues](#performance-issues)
4. [Numerical Stability](#numerical-stability)
5. [Memory Issues](#memory-issues)
6. [Algorithm Selection](#algorithm-selection)
7. [Debugging Tools](#debugging-tools)
8. [Error Reference](#error-reference)

## Common Issues

### Function Evaluation Errors

**Symptoms:** Function returns NaN, infinity, or panics during evaluation.

**Causes:**
- Division by zero
- Logarithm of negative numbers
- Square root of negative numbers
- Overflow/underflow in calculations

**Solutions:**
```rust
// Add input validation
fn safe_function(x: &ArrayView1<f64>) -> f64 {
    // Check for valid inputs
    if x.iter().any(|&xi| !xi.is_finite()) {
        return f64::INFINITY; // Return large value for invalid inputs
    }
    
    // Clamp inputs to safe ranges
    let safe_x = x.mapv(|xi| xi.max(-100.0).min(100.0));
    
    // Add small epsilon to avoid exact zeros
    let eps = 1e-12;
    let result = (safe_x[0] + eps).ln() + safe_x[1].powi(2);
    
    // Check result validity
    if result.is_finite() {
        result
    } else {
        f64::INFINITY
    }
}
```

### Gradient Computation Issues

**Symptoms:** Optimization fails with gradient-related errors.

**Causes:**
- Incorrect gradient implementation
- Numerical gradient computation issues
- Inconsistent function/gradient evaluations

**Solutions:**
```rust
use scirs2_optimize::test_utilities::*;

// Check gradient correctness
fn validate_gradient() -> Result<(), OptimizeError> {
    let func = |x: &ArrayView1<f64>| x[0].powi(2) + x[1].powi(2);
    let grad = |x: &ArrayView1<f64>| Array1::from_vec(vec![2.0 * x[0], 2.0 * x[1]]);
    
    let x_test = Array1::from_vec(vec![1.0, 2.0]);
    let error = check_gradient(func, grad, &x_test.view(), 1e-8);
    
    if error > 1e-6 {
        return Err(OptimizeError::InvalidInput(
            format!("Gradient error too large: {:.2e}", error)
        ));
    }
    
    Ok(())
}

// Use automatic differentiation as backup
use scirs2_optimize::automatic_differentiation::*;

let func_ad = |x: &ArrayView1<DualNumber>| -> DualNumber {
    x[0] * x[0] + x[1] * x[1]
};

let grad_func = create_ad_gradient(func_ad);
```

### Bounds Constraint Issues

**Symptoms:** Algorithm violates bounds or gets stuck at boundaries.

**Causes:**
- Improper bounds specification
- Algorithm doesn't support bounds
- Conflicting constraints

**Solutions:**
```rust
// Properly specify bounds
let bounds = Bounds::new(&[
    (Some(0.0), Some(1.0)),    // 0 <= x[0] <= 1
    (Some(-10.0), None),       // x[1] >= -10, no upper bound
    (None, Some(5.0)),         // x[2] <= 5, no lower bound
]);

// Check for bound violations
fn check_bounds(x: &ArrayView1<f64>, bounds: &Bounds) -> bool {
    for (i, &xi) in x.iter().enumerate() {
        if let Some(lower) = bounds.lower()[i] {
            if xi < lower - 1e-10 {
                return false;
            }
        }
        if let Some(upper) = bounds.upper()[i] {
            if xi > upper + 1e-10 {
                return false;
            }
        }
    }
    true
}

// Use projection for bound enforcement
fn project_to_bounds(x: &mut Array1<f64>, bounds: &Bounds) {
    for (i, xi) in x.iter_mut().enumerate() {
        if let Some(lower) = bounds.lower()[i] {
            *xi = xi.max(lower);
        }
        if let Some(upper) = bounds.upper()[i] {
            *xi = xi.min(upper);
        }
    }
}
```

## Convergence Problems

### Algorithm Doesn't Converge

**Symptoms:** Optimization reaches maximum iterations without convergence.

**Diagnosis:**
```rust
// Monitor convergence progress
let callback = |x: &ArrayView1<f64>, f_val: f64| -> bool {
    println!("Current: x={:?}, f={:.6}", x, f_val);
    
    // Check for stagnation
    static mut PREV_F: f64 = f64::INFINITY;
    static mut STAGNATION_COUNT: usize = 0;
    
    unsafe {
        if (f_val - PREV_F).abs() < 1e-12 {
            STAGNATION_COUNT += 1;
            if STAGNATION_COUNT > 10 {
                println!("Warning: Function value stagnating");
            }
        } else {
            STAGNATION_COUNT = 0;
        }
        PREV_F = f_val;
    }
    
    false // Continue optimization
};

let mut options = Options::default();
options.callback = Some(Box::new(callback));
```

**Solutions:**

1. **Adjust Tolerances:**
```rust
let mut options = Options::default();
options.gtol = 1e-3;  // Relax gradient tolerance
options.ftol = 1e-4;  // Relax function tolerance
options.xtol = 1e-4;  // Relax parameter tolerance
options.max_iter = 5000; // Increase iteration limit
```

2. **Try Different Starting Points:**
```rust
let starting_points = vec![
    Array1::from_vec(vec![0.0, 0.0]),
    Array1::from_vec(vec![1.0, 1.0]),
    Array1::from_vec(vec![-1.0, 1.0]),
    Array1::from_vec(vec![0.5, -0.5]),
];

let mut best_result = None;
let mut best_value = f64::INFINITY;

for x0 in starting_points {
    if let Ok(result) = minimize(func, &x0, Method::BFGS, Some(options.clone())) {
        if result.success && result.fun < best_value {
            best_value = result.fun;
            best_result = Some(result);
        }
    }
}
```

3. **Multi-Algorithm Approach:**
```rust
let algorithms = vec![
    Method::BFGS,
    Method::LBFGS,
    Method::Powell,
    Method::NelderMead,
];

for method in algorithms {
    match minimize(func, &x0, method, Some(options.clone())) {
        Ok(result) if result.success => {
            println!("Converged with {:?}", method);
            return Ok(result);
        }
        Ok(result) => {
            println!("{:?} failed: {}", method, result.message);
        }
        Err(e) => {
            println!("{:?} error: {}", method, e);
        }
    }
}
```

### Convergence to Wrong Solution

**Symptoms:** Algorithm converges but to incorrect or suboptimal solution.

**Causes:**
- Local minimum instead of global minimum
- Poor initial guess
- Ill-conditioned problem

**Solutions:**

1. **Global Optimization:**
```rust
use scirs2_optimize::global::*;

// Use differential evolution for global search
let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
let de_options = DifferentialEvolutionOptions {
    population_size: 50,
    max_iter: 1000,
    f: 0.8,
    cr: 0.9,
    ..Default::default()
};

let global_result = differential_evolution(func, &bounds, Some(de_options))?;

// Use global result as starting point for local optimization
let local_result = minimize(func, &global_result.x, Method::BFGS, None)?;
```

2. **Multi-Start Optimization:**
```rust
use scirs2_optimize::global::multi_start_with_clustering;

let bounds = vec![(-10.0, 10.0), (-10.0, 10.0)];
let clustering_options = ClusteringOptions {
    n_starts: 20,
    cluster_radius: 0.5,
    max_clusters: 5,
};

let result = multi_start_with_clustering(func, &bounds, clustering_options)?;
```

3. **Problem Conditioning:**
```rust
// Scale variables to similar magnitudes
fn scale_problem(func: F) -> impl Fn(&ArrayView1<f64>) -> f64 
where F: Fn(&ArrayView1<f64>) -> f64 {
    move |x_scaled: &ArrayView1<f64>| -> f64 {
        // Transform scaled variables back to original scale
        let x_original = x_scaled.mapv(|xi| xi * 100.0); // Example scaling
        func(&x_original) / 1e6 // Scale function value
    }
}
```

## Performance Issues

### Slow Convergence

**Symptoms:** Algorithm takes many iterations or long time to converge.

**Diagnosis:**
```rust
use std::time::Instant;

let start_time = Instant::now();
let result = minimize(func, &x0, Method::BFGS, None)?;
let elapsed = start_time.elapsed();

println!("Optimization took: {:?}", elapsed);
println!("Function evaluations: {}", result.nfev);
println!("Iterations: {}", result.nit);
println!("Time per iteration: {:?}", elapsed / result.nit as u32);
```

**Solutions:**

1. **Algorithm Selection:**
```rust
// For large problems, use L-BFGS
if problem_size > 1000 {
    minimize(func, &x0, Method::LBFGS, options)
} else {
    minimize(func, &x0, Method::BFGS, options)
}
```

2. **Provide Analytical Gradients:**
```rust
let gradient = |x: &ArrayView1<f64>| -> Array1<f64> {
    // Analytical gradient is much faster than numerical
    Array1::from_vec(vec![
        2.0 * x[0],              // ∂f/∂x₀
        2.0 * x[1] + x[0],       // ∂f/∂x₁
    ])
};

let mut options = Options::default();
options.jac = Some(Box::new(gradient));
```

3. **Enable Parallelization:**
```rust
// Enable parallel features
use scirs2_optimize::parallel::*;

let parallel_options = ParallelOptions {
    num_threads: Some(std::thread::available_parallelism()?.get()),
    chunk_size: Some(64),
};

// Use parallel gradient computation
let grad_func = |x: &ArrayView1<f64>| -> Array1<f64> {
    parallel_finite_diff_gradient(func, x, parallel_options.clone())
        .unwrap_or_else(|_| finite_diff_gradient(func, x, 1e-8))
};
```

### Memory Usage Issues

**Symptoms:** High memory consumption or out-of-memory errors.

**Solutions:**

1. **Use Memory-Efficient Algorithms:**
```rust
// L-BFGS instead of BFGS for large problems
let result = minimize(func, &x0, Method::LBFGS, options)?;

// Configure L-BFGS memory
let mut lbfgs_options = Options::default();
lbfgs_options.memory_limit = Some(20); // Limit BFGS history
```

2. **Sparse Computations:**
```rust
use scirs2_optimize::sparse_numdiff::*;

let sparse_options = SparseFiniteDiffOptions {
    sparsity: None, // Auto-detect sparsity
    rel_step: None,
    abs_step: 1e-8,
    bounds: None,
};

// Use sparse Jacobian computation
let jac_func = |x: &ArrayView1<f64>| -> Array1<f64> {
    let sparse_jac = sparse_jacobian(residual_func, x, &sparse_options)?;
    sparse_jac.to_dense() // Convert if needed
};
```

3. **Streaming/Chunked Processing:**
```rust
// For very large datasets, process in chunks
fn chunked_gradient(
    grad_func: &mut dyn StochasticGradientFunction,
    x: &ArrayView1<f64>,
    data_provider: &dyn DataProvider,
    chunk_size: usize
) -> Array1<f64> {
    let mut total_grad = Array1::zeros(x.len());
    let n_samples = data_provider.num_samples();
    
    for chunk_start in (0..n_samples).step_by(chunk_size) {
        let chunk_end = (chunk_start + chunk_size).min(n_samples);
        let chunk_indices: Vec<f64> = (chunk_start..chunk_end)
            .map(|i| i as f64)
            .collect();
        
        let chunk_grad = grad_func.compute_gradient(x, &chunk_indices);
        total_grad = total_grad + chunk_grad;
    }
    
    total_grad / (n_samples as f64 / chunk_size as f64)
}
```

## Numerical Stability

### Ill-Conditioned Problems

**Symptoms:** Erratic convergence, sensitivity to small changes, poor accuracy.

**Diagnosis:**
```rust
// Check condition number of Hessian (if available)
fn check_conditioning(hess: &Array2<f64>) -> f64 {
    use ndarray_linalg::SolveH;
    let eigenvals = hess.eigvalsh(UPLO::Upper).unwrap();
    let max_eigval = eigenvals.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let min_eigval = eigenvals.iter().fold(f64::INFINITY, |a, &b| a.min(b.abs()));
    max_eigval / min_eigval
}

// Monitor gradient norms
let callback = |x: &ArrayView1<f64>, f_val: f64| -> bool {
    let grad = numerical_gradient(func, x, 1e-8);
    let grad_norm = grad.dot(&grad).sqrt();
    println!("Gradient norm: {:.2e}", grad_norm);
    
    if grad_norm < 1e-15 {
        println!("Warning: Very small gradient, possible numerical issues");
    }
    false
};
```

**Solutions:**

1. **Problem Scaling:**
```rust
// Scale variables to unit order of magnitude
fn scale_variables(x: &ArrayView1<f64>, scales: &ArrayView1<f64>) -> Array1<f64> {
    x / scales
}

fn unscale_variables(x_scaled: &ArrayView1<f64>, scales: &ArrayView1<f64>) -> Array1<f64> {
    x_scaled * scales
}

// Estimate appropriate scaling
fn estimate_scales(func: F, x0: &ArrayView1<f64>) -> Array1<f64> 
where F: Fn(&ArrayView1<f64>) -> f64 {
    let grad = numerical_gradient(func, x0, 1e-8);
    grad.mapv(|gi| 1.0 / gi.abs().max(1e-8))
}
```

2. **Regularization:**
```rust
// Add regularization term to objective
fn regularized_objective(x: &ArrayView1<f64>, lambda: f64) -> f64 {
    let original_value = original_function(x);
    let regularization = lambda * x.dot(x); // L2 regularization
    original_value + regularization
}
```

3. **Robust Algorithms:**
```rust
// Use trust region methods for better stability
let result = minimize(func, &x0, Method::TrustRegion, options)?;

// Or use robust least squares for data fitting
let huber_loss = HuberLoss::new(1.345);
let result = robust_least_squares(residual, &x0, huber_loss, None, &data, None)?;
```

### Numerical Precision Issues

**Symptoms:** Results change significantly with small tolerance changes.

**Solutions:**

1. **Adaptive Precision:**
```rust
use scirs2_optimize::unconstrained::robust_convergence::*;

let robust_options = RobustConvergenceOptions {
    adaptive_tolerance: AdaptiveToleranceOptions {
        enable: true,
        initial_gtol: 1e-6,
        min_gtol: 1e-12,
        max_gtol: 1e-3,
        adaptation_factor: 0.1,
        noise_level_estimate: 1e-10,
    },
    enable_noise_robust: true,
    statistical_tests: true,
    ..Default::default()
};
```

2. **Higher Precision Types:**
```rust
// For critical applications, consider using higher precision
// (Note: This would require additional dependencies)
use rug::{Float, Assign};

fn high_precision_function(x: &[Float]) -> Float {
    let mut result = Float::with_val(128, 0); // 128-bit precision
    // Implement function with high precision arithmetic
    result
}
```

## Algorithm Selection

### Choosing the Right Algorithm

**Decision Matrix:**

| Problem Type | Size | Constraints | Noise | Recommended Algorithm |
|--------------|------|-------------|-------|----------------------|
| Smooth | Small | None | Low | BFGS |
| Smooth | Large | None | Low | L-BFGS |
| Smooth | Any | Linear | Low | SLSQP |
| Smooth | Any | Nonlinear | Low | Trust-Constr |
| Non-smooth | Small | None | High | Nelder-Mead |
| Non-smooth | Any | None | High | Powell |
| Multimodal | Any | None | Any | Differential Evolution |
| Stochastic | Large | None | High | Adam/SGD |

**Algorithm Switching:**
```rust
fn adaptive_algorithm_selection(
    func: F,
    x0: &ArrayView1<f64>,
    problem_size: usize,
    has_constraints: bool,
    is_noisy: bool,
) -> Result<OptimizeResult, OptimizeError>
where F: Fn(&ArrayView1<f64>) -> f64 + Clone {
    
    let algorithms = if has_constraints {
        vec![Method::SLSQP, Method::TrustConstr]
    } else if is_noisy {
        vec![Method::NelderMead, Method::Powell]
    } else if problem_size > 1000 {
        vec![Method::LBFGS, Method::CG]
    } else {
        vec![Method::BFGS, Method::Newton]
    };
    
    for method in algorithms {
        match minimize(func.clone(), x0, method, None) {
            Ok(result) if result.success => return Ok(result),
            Ok(result) => println!("{:?} failed: {}", method, result.message),
            Err(e) => println!("{:?} error: {}", method, e),
        }
    }
    
    // Fallback to global optimization
    let bounds = estimate_bounds(x0);
    differential_evolution(func, &bounds, None)
}
```

## Debugging Tools

### Function Analysis

```rust
use scirs2_optimize::debug_tools::*;

// Analyze function behavior
fn analyze_function<F>(func: F, x0: &ArrayView1<f64>) 
where F: Fn(&ArrayView1<f64>) -> f64 + Clone {
    
    // Check for discontinuities
    let discontinuity_check = check_continuity(func.clone(), x0, 1e-6);
    println!("Discontinuity measure: {:.2e}", discontinuity_check);
    
    // Estimate condition number
    let condition_number = estimate_condition_number(func.clone(), x0);
    println!("Estimated condition number: {:.2e}", condition_number);
    
    // Check gradient consistency
    let grad_error = verify_gradient_consistency(func.clone(), x0);
    println!("Gradient consistency error: {:.2e}", grad_error);
    
    // Analyze local landscape
    let landscape = analyze_local_landscape(func, x0, 0.1);
    println!("Local minima detected: {}", landscape.local_minima.len());
}
```

### Convergence Analysis

```rust
use scirs2_optimize::analysis::*;

// Track convergence history
struct ConvergenceTracker {
    function_values: Vec<f64>,
    gradient_norms: Vec<f64>,
    step_sizes: Vec<f64>,
    iteration_times: Vec<Duration>,
}

impl ConvergenceTracker {
    fn callback(&mut self) -> impl FnMut(&ArrayView1<f64>, f64) -> bool + '_ {
        |x: &ArrayView1<f64>, f_val: f64| -> bool {
            let grad = numerical_gradient(func, x, 1e-8);
            let grad_norm = grad.dot(&grad).sqrt();
            
            self.function_values.push(f_val);
            self.gradient_norms.push(grad_norm);
            
            // Detect convergence issues
            if self.function_values.len() > 10 {
                let recent_change = (self.function_values.last().unwrap() - 
                                   self.function_values[self.function_values.len()-10]).abs();
                if recent_change < 1e-15 {
                    println!("Warning: Function value plateaued");
                }
            }
            
            false
        }
    }
    
    fn plot_convergence(&self) {
        // Integration with plotting libraries
        use plotters::prelude::*;
        
        let root = BitMapBackend::new("convergence.png", (800, 600))
            .into_drawing_area();
        root.fill(&WHITE)?;
        
        let mut chart = ChartBuilder::on(&root)
            .caption("Convergence History", ("Arial", 50))
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(40)
            .build_cartesian_2d(
                0..self.function_values.len(),
                *self.function_values.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()
                ..*self.function_values.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()
            )?;
        
        chart.draw_series(LineSeries::new(
            self.function_values.iter().enumerate().map(|(i, &v)| (i, v)),
            &RED,
        ))?;
        
        root.present()?;
    }
}
```

## Error Reference

### Common Error Messages and Solutions

#### "Convergence failed: Maximum iterations reached"
- **Cause**: Algorithm couldn't converge within iteration limit
- **Solution**: Increase `max_iter`, relax tolerances, or try different algorithm

#### "Invalid input: NaN or infinite value in function"
- **Cause**: Function evaluation returned invalid value
- **Solution**: Add input validation and bounds to your function

#### "Numerical error: Singular matrix in linear solve"
- **Cause**: Hessian or Jacobian is singular (not invertible)
- **Solution**: Add regularization, use different algorithm, or check problem formulation

#### "Function evaluation failed: Gradient check failed"
- **Cause**: Analytical gradient doesn't match numerical gradient
- **Solution**: Fix gradient implementation or use automatic differentiation

#### "Linear algebra error: Matrix decomposition failed"
- **Cause**: Numerical issues in matrix operations
- **Solution**: Use more robust algorithms, add regularization, or scale problem

### Error Recovery Strategies

```rust
use scirs2_optimize::error_recovery::*;

fn robust_minimize<F>(
    func: F,
    x0: &ArrayView1<f64>,
    max_attempts: usize,
) -> Result<OptimizeResult, OptimizeError>
where F: Fn(&ArrayView1<f64>) -> f64 + Clone {
    
    let strategies = vec![
        ErrorRecoveryStrategy::RelaxTolerances,
        ErrorRecoveryStrategy::ChangeAlgorithm,
        ErrorRecoveryStrategy::AddRegularization,
        ErrorRecoveryStrategy::ScaleProblem,
        ErrorRecoveryStrategy::GlobalOptimization,
    ];
    
    let mut last_error = None;
    
    for (attempt, strategy) in strategies.iter().enumerate() {
        if attempt >= max_attempts {
            break;
        }
        
        match strategy.apply(func.clone(), x0) {
            Ok(result) => return Ok(result),
            Err(e) => {
                println!("Attempt {} failed with strategy {:?}: {}", attempt + 1, strategy, e);
                last_error = Some(e);
            }
        }
    }
    
    Err(last_error.unwrap_or_else(|| 
        OptimizeError::ConvergenceError("All recovery strategies failed".to_string())
    ))
}
```

This troubleshooting guide provides comprehensive solutions to common optimization problems. When encountering issues, start with the most likely causes and work through the suggested solutions systematically.