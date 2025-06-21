# Getting Started with scirs2-optimize

This guide will help you get up and running with the `scirs2-optimize` library, from installation to solving your first optimization problems.

## Table of Contents

1. [Installation](#installation)
2. [Basic Concepts](#basic-concepts)
3. [Your First Optimization](#your-first-optimization)
4. [Step-by-Step Tutorials](#step-by-step-tutorials)
5. [Common Patterns](#common-patterns)
6. [Next Steps](#next-steps)

## Installation

### Prerequisites

Ensure you have Rust installed. If not, install it from [rustup.rs](https://rustup.rs/).

### Adding the Dependency

Add `scirs2-optimize` to your `Cargo.toml`:

```toml
[dependencies]
scirs2-optimize = "0.1.0-alpha.4"
ndarray = "0.16"

# Optional: Enable additional features
scirs2-optimize = { version = "0.1.0-alpha.4", features = ["parallel", "simd"] }
```

### Verify Installation

Create a simple test to verify everything works:

```rust
// src/main.rs
use scirs2_optimize::prelude::*;
use ndarray::Array1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("scirs2-optimize is working!");
    
    // Simple quadratic function: f(x) = x²
    let func = |x: &ArrayView1<f64>| x[0].powi(2);
    let x0 = Array1::from_vec(vec![2.0]);
    
    let result = minimize(func, &x0, Method::BFGS, None)?;
    println!("Minimum found at: {:.6}", result.x[0]);
    println!("Function value: {:.6}", result.fun);
    
    Ok(())
}
```

Run with `cargo run` - you should see the minimum found near 0.

## Basic Concepts

### Optimization Problem Structure

Most optimization problems follow this pattern:

```
minimize f(x)
subject to constraints (optional)
where x ∈ ℝⁿ
```

- **f(x)**: Objective function to minimize
- **x**: Decision variables (parameters to optimize)
- **constraints**: Optional restrictions on x

### Key Components

1. **Objective Function**: What you want to minimize
2. **Initial Guess**: Starting point for optimization
3. **Algorithm**: Method to find the minimum
4. **Options**: Configuration parameters
5. **Result**: Solution and convergence information

## Your First Optimization

Let's solve a simple but non-trivial problem: the Rosenbrock function.

### Problem Definition

The Rosenbrock function is a classic test case:
```
f(x, y) = (a - x)² + b(y - x²)²
```

With standard parameters a=1, b=100, the global minimum is at (1, 1) with value 0.

### Complete Example

```rust
use scirs2_optimize::prelude::*;
use ndarray::Array1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Define the Rosenbrock function
    let rosenbrock = |x: &ArrayView1<f64>| -> f64 {
        let (a, b) = (1.0, 100.0);
        (a - x[0]).powi(2) + b * (x[1] - x[0].powi(2)).powi(2)
    };
    
    // Define the analytical gradient (optional but recommended)
    let rosenbrock_grad = |x: &ArrayView1<f64>| -> Array1<f64> {
        let (a, b) = (1.0, 100.0);
        Array1::from_vec(vec![
            -2.0 * (a - x[0]) - 4.0 * b * x[0] * (x[1] - x[0].powi(2)),
            2.0 * b * (x[1] - x[0].powi(2))
        ])
    };
    
    // Starting point (standard initial guess)
    let x0 = Array1::from_vec(vec![-1.2, 1.0]);
    
    // Configure optimization
    let mut options = Options::default();
    options.jac = Some(Box::new(rosenbrock_grad));
    options.max_iter = 1000;
    options.gtol = 1e-6;
    
    // Solve the optimization problem
    let result = minimize(rosenbrock, &x0, Method::BFGS, Some(options))?;
    
    // Display results
    println!("Optimization Results:");
    println!("  Solution: [{:.6}, {:.6}]", result.x[0], result.x[1]);
    println!("  Function value: {:.2e}", result.fun);
    println!("  Iterations: {}", result.nit);
    println!("  Function evaluations: {}", result.nfev);
    println!("  Success: {}", result.success);
    println!("  Message: {}", result.message);
    
    // Verify the result
    let expected = Array1::from_vec(vec![1.0, 1.0]);
    let error = (&result.x - &expected).mapv(|x| x.abs()).sum();
    println!("  Error from true minimum: {:.2e}", error);
    
    Ok(())
}
```

### Understanding the Output

- **Solution**: The optimal values of x and y
- **Function value**: f(x, y) at the solution (should be close to 0)
- **Iterations**: Number of optimization steps
- **Function evaluations**: Total function calls (includes line searches)
- **Success**: Whether the algorithm converged
- **Message**: Status information

## Step-by-Step Tutorials

### Tutorial 1: Unconstrained Optimization

**Goal**: Minimize a simple multivariate function.

```rust
use scirs2_optimize::prelude::*;
use ndarray::Array1;

// Problem: Minimize f(x,y) = (x-2)² + (y-3)² + 5
// Expected solution: x=2, y=3, f=5

fn tutorial_1() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Tutorial 1: Basic Unconstrained Optimization ===");
    
    // Step 1: Define your objective function
    let objective = |x: &ArrayView1<f64>| -> f64 {
        (x[0] - 2.0).powi(2) + (x[1] - 3.0).powi(2) + 5.0
    };
    
    // Step 2: Choose a starting point
    let x0 = Array1::from_vec(vec![0.0, 0.0]);
    
    // Step 3: Select an algorithm
    let method = Method::BFGS;
    
    // Step 4: Run optimization
    let result = minimize(objective, &x0, method, None)?;
    
    // Step 5: Analyze results
    println!("Starting point: [{:.1}, {:.1}]", x0[0], x0[1]);
    println!("Solution: [{:.6}, {:.6}]", result.x[0], result.x[1]);
    println!("Function value: {:.6}", result.fun);
    println!("Expected: [2.0, 3.0], value: 5.0");
    
    // Step 6: Verify success
    assert!((result.x[0] - 2.0).abs() < 1e-5);
    assert!((result.x[1] - 3.0).abs() < 1e-5);
    assert!((result.fun - 5.0).abs() < 1e-10);
    
    println!("✓ Tutorial 1 completed successfully!\n");
    Ok(())
}
```

### Tutorial 2: Providing Gradients

**Goal**: Improve performance by providing analytical gradients.

```rust
fn tutorial_2() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Tutorial 2: Using Analytical Gradients ===");
    
    // Same objective function
    let objective = |x: &ArrayView1<f64>| -> f64 {
        (x[0] - 2.0).powi(2) + (x[1] - 3.0).powi(2) + 5.0
    };
    
    // Analytical gradient: ∇f = [2(x-2), 2(y-3)]
    let gradient = |x: &ArrayView1<f64>| -> Array1<f64> {
        Array1::from_vec(vec![
            2.0 * (x[0] - 2.0),
            2.0 * (x[1] - 3.0)
        ])
    };
    
    let x0 = Array1::from_vec(vec![0.0, 0.0]);
    
    // Compare with and without gradients
    let start_time = std::time::Instant::now();
    let result_no_grad = minimize(objective, &x0, Method::BFGS, None)?;
    let time_no_grad = start_time.elapsed();
    
    let mut options = Options::default();
    options.jac = Some(Box::new(gradient));
    
    let start_time = std::time::Instant::now();
    let result_with_grad = minimize(objective, &x0, Method::BFGS, Some(options))?;
    let time_with_grad = start_time.elapsed();
    
    println!("Without gradient:");
    println!("  Function evaluations: {}", result_no_grad.nfev);
    println!("  Time: {:?}", time_no_grad);
    
    println!("With gradient:");
    println!("  Function evaluations: {}", result_with_grad.nfev);
    println!("  Time: {:?}", time_with_grad);
    
    println!("Speedup: {:.2}x", time_no_grad.as_secs_f64() / time_with_grad.as_secs_f64());
    println!("✓ Tutorial 2 completed successfully!\n");
    
    Ok(())
}
```

### Tutorial 3: Bounds Constraints

**Goal**: Optimize with simple bounds on variables.

```rust
fn tutorial_3() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Tutorial 3: Bounds-Constrained Optimization ===");
    
    // Problem: minimize (x+1)² + (y+1)² with constraints 0 ≤ x,y ≤ 2
    // Unconstrained minimum: (-1, -1)
    // Constrained minimum: (0, 0)
    
    let objective = |x: &ArrayView1<f64>| -> f64 {
        (x[0] + 1.0).powi(2) + (x[1] + 1.0).powi(2)
    };
    
    let x0 = Array1::from_vec(vec![1.0, 1.0]);
    
    // First, solve without constraints
    let unconstrained_result = minimize(objective, &x0, Method::BFGS, None)?;
    
    // Now add bounds: 0 ≤ x,y ≤ 2
    let bounds = Bounds::new(&[
        (Some(0.0), Some(2.0)),  // 0 ≤ x ≤ 2
        (Some(0.0), Some(2.0)),  // 0 ≤ y ≤ 2
    ]);
    
    let mut options = Options::default();
    options.bounds = Some(bounds);
    
    let constrained_result = minimize(objective, &x0, Method::BFGS, Some(options))?;
    
    println!("Unconstrained solution: [{:.6}, {:.6}]", 
             unconstrained_result.x[0], unconstrained_result.x[1]);
    println!("Unconstrained value: {:.6}", unconstrained_result.fun);
    
    println!("Constrained solution: [{:.6}, {:.6}]", 
             constrained_result.x[0], constrained_result.x[1]);
    println!("Constrained value: {:.6}", constrained_result.fun);
    
    // Verify bounds are satisfied
    assert!(constrained_result.x[0] >= -1e-10);
    assert!(constrained_result.x[1] >= -1e-10);
    assert!(constrained_result.x[0] <= 2.0 + 1e-10);
    assert!(constrained_result.x[1] <= 2.0 + 1e-10);
    
    println!("✓ Tutorial 3 completed successfully!\n");
    Ok(())
}
```

### Tutorial 4: Constrained Optimization

**Goal**: Handle equality and inequality constraints.

```rust
use scirs2_optimize::constrained::*;

fn tutorial_4() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Tutorial 4: General Constrained Optimization ===");
    
    // Problem: minimize (x-1)² + (y-2)²
    // Subject to: x + y ≤ 3 (inequality)
    //            x² + y² = 5 (equality)
    
    let objective = |x: &ArrayView1<f64>| -> f64 {
        (x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2)
    };
    
    // Inequality constraint: 3 - x - y ≥ 0
    let ineq_constraint = |x: &ArrayView1<f64>| -> f64 {
        3.0 - x[0] - x[1]
    };
    
    // Equality constraint: x² + y² - 5 = 0
    let eq_constraint = |x: &ArrayView1<f64>| -> f64 {
        x[0].powi(2) + x[1].powi(2) - 5.0
    };
    
    let constraints = vec![
        Constraint {
            fun: Box::new(ineq_constraint),
            constraint_type: ConstraintType::Inequality,
            jac: None, // Auto-computed
        },
        Constraint {
            fun: Box::new(eq_constraint),
            constraint_type: ConstraintType::Equality,
            jac: None,
        },
    ];
    
    // Starting point (on the circle x² + y² = 5)
    let x0 = Array1::from_vec(vec![2.0, 1.0]);
    
    let result = minimize_constrained(
        objective, 
        &x0, 
        &constraints, 
        Method::SLSQP, 
        None
    )?;
    
    println!("Solution: [{:.6}, {:.6}]", result.x[0], result.x[1]);
    println!("Objective value: {:.6}", result.fun);
    
    // Verify constraints
    let ineq_val = ineq_constraint(&result.x.view());
    let eq_val = eq_constraint(&result.x.view());
    
    println!("Inequality constraint (≥0): {:.6}", ineq_val);
    println!("Equality constraint (=0): {:.6}", eq_val);
    
    assert!(ineq_val >= -1e-6, "Inequality constraint violated");
    assert!(eq_val.abs() < 1e-6, "Equality constraint violated");
    
    println!("✓ Tutorial 4 completed successfully!\n");
    Ok(())
}
```

### Tutorial 5: Least Squares Fitting

**Goal**: Fit a model to experimental data.

```rust
use scirs2_optimize::least_squares::*;

fn tutorial_5() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Tutorial 5: Curve Fitting with Least Squares ===");
    
    // Generate synthetic data: y = 2*exp(-0.5*x) + noise
    let true_params = [2.0, -0.5];
    let x_data = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let y_data = vec![2.1, 1.2, 0.7, 0.4, 0.2]; // With some noise
    
    // Model: y = a * exp(b * x)
    let residual = |params: &[f64], _: &[f64]| -> Array1<f64> {
        let (a, b) = (params[0], params[1]);
        let mut residuals = Array1::zeros(x_data.len());
        
        for (i, (&x, &y)) in x_data.iter().zip(y_data.iter()).enumerate() {
            let predicted = a * (b * x).exp();
            residuals[i] = y - predicted;
        }
        
        residuals
    };
    
    // Initial guess
    let x0 = Array1::from_vec(vec![1.0, -0.1]);
    let data = Array1::zeros(0); // No additional data needed
    
    let result = least_squares(
        residual,
        &x0,
        Method::LevenbergMarquardt,
        None, // Auto-compute Jacobian
        &data,
        None
    )?;
    
    println!("True parameters: a={:.1}, b={:.1}", true_params[0], true_params[1]);
    println!("Fitted parameters: a={:.3}, b={:.3}", result.x[0], result.x[1]);
    println!("Residual norm: {:.2e}", result.fun);
    
    // Calculate R²
    let mean_y: f64 = y_data.iter().sum::<f64>() / y_data.len() as f64;
    let ss_tot: f64 = y_data.iter().map(|y| (y - mean_y).powi(2)).sum();
    let ss_res: f64 = result.fun.powi(2);
    let r_squared = 1.0 - ss_res / ss_tot;
    
    println!("R² = {:.4}", r_squared);
    
    println!("✓ Tutorial 5 completed successfully!\n");
    Ok(())
}
```

### Running All Tutorials

```rust
fn main() -> Result<(), Box<dyn std::error::Error>> {
    tutorial_1()?;
    tutorial_2()?;
    tutorial_3()?;
    tutorial_4()?;
    tutorial_5()?;
    
    println!("🎉 All tutorials completed successfully!");
    println!("You're ready to tackle real optimization problems!");
    
    Ok(())
}
```

## Common Patterns

### Pattern 1: Algorithm Selection Strategy

```rust
fn solve_with_fallback<F>(
    func: F,
    x0: &ArrayView1<f64>,
) -> Result<OptimizeResult, OptimizeError>
where F: Fn(&ArrayView1<f64>) -> f64 + Clone {
    
    // Try algorithms in order of preference
    let algorithms = vec![
        Method::BFGS,        // Fast for smooth functions
        Method::LBFGS,       // Memory-efficient alternative
        Method::Powell,      // Derivative-free
        Method::NelderMead,  // Robust but slow
    ];
    
    for method in algorithms {
        match minimize(func.clone(), x0, method, None) {
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
    
    Err(OptimizeError::ConvergenceError("All methods failed".to_string()))
}
```

### Pattern 2: Progress Monitoring

```rust
fn optimize_with_progress<F>(
    func: F,
    x0: &ArrayView1<f64>,
) -> Result<OptimizeResult, OptimizeError>
where F: Fn(&ArrayView1<f64>) -> f64 {
    
    let start_time = std::time::Instant::now();
    let mut iteration_count = 0;
    
    let callback = |x: &ArrayView1<f64>, f_val: f64| -> bool {
        iteration_count += 1;
        
        if iteration_count % 10 == 0 {
            let elapsed = start_time.elapsed();
            println!("Iter {}: f={:.6e}, time={:.1}s", 
                     iteration_count, f_val, elapsed.as_secs_f64());
        }
        
        false // Continue optimization
    };
    
    let mut options = Options::default();
    options.callback = Some(Box::new(callback));
    
    minimize(func, x0, Method::BFGS, Some(options))
}
```

### Pattern 3: Problem Scaling

```rust
fn optimize_with_scaling<F>(
    func: F,
    x0: &ArrayView1<f64>,
    scales: &ArrayView1<f64>,
) -> Result<OptimizeResult, OptimizeError>
where F: Fn(&ArrayView1<f64>) -> f64 {
    
    // Create scaled version of the function
    let scaled_func = |x_scaled: &ArrayView1<f64>| -> f64 {
        let x_original = x_scaled * scales;
        func(&x_original)
    };
    
    // Scale initial point
    let x0_scaled = x0 / scales;
    
    // Optimize in scaled space
    let result = minimize(scaled_func, &x0_scaled, Method::BFGS, None)?;
    
    // Transform result back to original space
    let x_original = &result.x * scales;
    
    Ok(OptimizeResult {
        x: x_original,
        fun: result.fun,
        success: result.success,
        nit: result.nit,
        nfev: result.nfev,
        jacobian: result.jacobian,
        hessian: result.hessian,
        message: result.message,
    })
}
```

## Next Steps

### Advanced Topics to Explore

1. **Stochastic Optimization**: For machine learning and large-scale problems
   - Read: [Stochastic Optimization Guide](STOCHASTIC_GUIDE.md)
   - Example: Neural network training

2. **Global Optimization**: For multimodal problems
   - Read: [Global Optimization Guide](GLOBAL_GUIDE.md)
   - Example: Parameter estimation with multiple local minima

3. **Multi-Objective Optimization**: For conflicting objectives
   - Read: [Multi-Objective Guide](MULTI_OBJECTIVE_GUIDE.md)
   - Example: Engineering design trade-offs

4. **Automatic Differentiation**: For exact gradients
   - Read: [Automatic Differentiation Guide](AD_GUIDE.md)
   - Example: Complex function optimization

### Performance Optimization

1. **Enable Parallelization**:
   ```toml
   scirs2_optimize = { version = "0.1.0-alpha.4", features = ["parallel"] }
   ```

2. **Use SIMD Operations**:
   ```toml
   scirs2_optimize = { version = "0.1.0-alpha.4", features = ["simd"] }
   ```

3. **Profile Your Code**:
   ```bash
   cargo build --release
   cargo flamegraph --bin your_binary
   ```

### Real-World Applications

- **Machine Learning**: Parameter tuning, neural network training
- **Engineering**: Design optimization, control systems
- **Finance**: Portfolio optimization, risk management
- **Science**: Parameter estimation, model fitting
- **Operations Research**: Resource allocation, scheduling

### Getting Help

1. **Documentation**: Comprehensive API reference and guides
2. **Examples**: Real-world examples in the repository
3. **Community**: GitHub issues for questions and bug reports
4. **Contributing**: Help improve the library

You're now ready to tackle complex optimization problems with `scirs2-optimize`! Start with the patterns that match your use case and gradually explore more advanced features as needed.