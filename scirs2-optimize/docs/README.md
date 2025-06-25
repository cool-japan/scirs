# scirs2-optimize Documentation

Welcome to the comprehensive documentation for `scirs2-optimize`, a high-performance optimization library for Rust inspired by SciPy's optimization module.

## Documentation Overview

This documentation is organized to help you quickly find the information you need, whether you're a beginner getting started or an expert looking for specific implementation details.

### 📚 Core Documentation

| Document | Description | Audience |
|----------|-------------|----------|
| [Getting Started](GETTING_STARTED.md) | Complete tutorial from installation to first optimization | Beginners |
| [Usage Guide](USAGE_GUIDE.md) | Practical examples and best practices | All users |
| [API Reference](API_REFERENCE.md) | Comprehensive API documentation | All users |
| [Features](FEATURES.md) | Complete feature reference and capabilities | All users |
| [Algorithm Reference](ALGORITHMS.md) | Mathematical foundations and algorithm details | Advanced users |

### 🎯 Specialized Guides

| Guide | Focus Area | Use Cases |
|-------|-----------|-----------|
| [Examples](EXAMPLES.md) | Real-world application examples | Learning by example |
| [Troubleshooting](TROUBLESHOOTING.md) | Common issues and solutions | Problem solving |

### 🚀 Quick Navigation

**New to optimization?** → Start with [Getting Started](GETTING_STARTED.md)

**Need specific examples?** → Check [Examples](EXAMPLES.md)

**Looking for an API?** → Browse [API Reference](API_REFERENCE.md)

**Having problems?** → Consult [Troubleshooting](TROUBLESHOOTING.md)

**Want to understand algorithms?** → Read [Algorithm Reference](ALGORITHMS.md)

## What is scirs2-optimize?

`scirs2-optimize` is a comprehensive optimization library for Rust that provides:

- **Unconstrained optimization** with BFGS, L-BFGS, Nelder-Mead, Powell, and more
- **Constrained optimization** with SLSQP, Trust-Constr, and Interior Point methods
- **Stochastic optimization** with SGD, Adam, RMSProp for machine learning
- **Least squares** with robust methods and outlier handling
- **Global optimization** with Differential Evolution, Basin-hopping, and Bayesian optimization
- **Multi-objective optimization** with NSGA-II/III and scalarization
- **Automatic differentiation** for exact gradients and Hessians
- **Parallel processing** and SIMD acceleration for performance

## Key Features

### 🔧 **Ergonomic API**
```rust
use scirs2_optimize::prelude::*;

let result = minimize(objective_function, &initial_guess, Method::BFGS, None)?;
println!("Solution: {:?}", result.x);
```

### ⚡ **High Performance**
- Optimized algorithms with numerical stability
- Optional SIMD acceleration
- Parallel function evaluation
- Memory-efficient implementations for large-scale problems

### 🎯 **Comprehensive**
- Covers most optimization use cases
- Compatible with SciPy's optimization API
- Extensive algorithm selection
- Robust error handling

### 🧪 **Reliable**
- Extensive testing against known solutions
- Numerical stability safeguards
- Comprehensive documentation
- Real-world validation

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
scirs2_optimize = "0.1.0-alpha.6"
ndarray = "0.16"

# Optional features
scirs2_optimize = { 
    version = "0.1.0-alpha.6", 
    features = ["parallel", "simd", "async"] 
}
```

## Quick Start Example

```rust
use scirs2_optimize::prelude::*;
use ndarray::Array1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Define the Rosenbrock function
    let rosenbrock = |x: &ArrayView1<f64>| -> f64 {
        let (a, b) = (1.0, 100.0);
        (a - x[0]).powi(2) + b * (x[1] - x[0].powi(2)).powi(2)
    };
    
    // Starting point
    let x0 = Array1::from_vec(vec![-1.2, 1.0]);
    
    // Optimize
    let result = minimize(rosenbrock, &x0, Method::BFGS, None)?;
    
    println!("Solution: [{:.6}, {:.6}]", result.x[0], result.x[1]);
    println!("Function value: {:.2e}", result.fun);
    println!("Converged: {}", result.success);
    
    Ok(())
}
```

## Algorithm Categories

### Unconstrained Optimization
For problems without constraints on variables.

| Algorithm | Best For | Characteristics |
|-----------|----------|----------------|
| **BFGS** | Smooth functions, medium size | Fast convergence, needs gradients |
| **L-BFGS** | Large-scale problems | Memory efficient, scalable |
| **Newton** | High accuracy needs | Quadratic convergence, needs Hessian |
| **Powell** | Derivative-free | Robust, no gradients needed |
| **Nelder-Mead** | Noisy/discontinuous | Very robust, slower convergence |

### Constrained Optimization
For problems with equality/inequality constraints.

| Algorithm | Best For | Constraint Types |
|-----------|----------|-----------------|
| **SLSQP** | General constraints | Equality + inequality |
| **Trust-Constr** | Nonlinear constraints | All types, robust |
| **Interior Point** | Many inequalities | Inequality constraints |

### Stochastic Optimization
For machine learning and large-scale problems.

| Algorithm | Best For | Characteristics |
|-----------|----------|----------------|
| **SGD** | Large datasets | Simple, requires tuning |
| **Adam** | Deep learning | Adaptive, robust |
| **AdamW** | Better generalization | Decoupled weight decay |
| **RMSProp** | RNNs | Good for non-stationary objectives |

### Specialized Methods

| Category | Algorithms | Applications |
|----------|------------|-------------|
| **Least Squares** | Levenberg-Marquardt, Robust methods | Curve fitting, parameter estimation |
| **Global Optimization** | Differential Evolution, Bayesian | Multimodal problems, global search |
| **Multi-Objective** | NSGA-II/III | Trade-off optimization |
| **Root Finding** | Hybrid, Anderson, Krylov | Solving nonlinear equations |

## Performance Features

### Parallelization
```rust
use scirs2_optimize::parallel::*;

let gradient = parallel_finite_diff_gradient(func, &x, ParallelOptions::default())?;
```

### SIMD Acceleration
```rust
use scirs2_optimize::simd_ops::*;

let result = simd_dot_product(&vec1, &vec2);
```

### Automatic Differentiation
```rust
use scirs2_optimize::automatic_differentiation::*;

let grad_func = create_ad_gradient(|x: &ArrayView1<DualNumber>| x[0] * x[0] + x[1] * x[1]);
```

### Memory Efficiency
```rust
// L-BFGS for large problems
let result = minimize(func, &x0, Method::LBFGS, None)?;

// Sparse differentiation
use scirs2_optimize::sparse_numdiff::*;
let sparse_jac = sparse_jacobian(func, &x, &SparseFiniteDiffOptions::default())?;
```

## Documentation Structure

### For Beginners
1. [Getting Started](GETTING_STARTED.md) - Installation and first steps
2. [Examples](EXAMPLES.md) - Learn by example
3. [Usage Guide](USAGE_GUIDE.md) - Common patterns and best practices

### For Practitioners
1. [API Reference](API_REFERENCE.md) - Complete function documentation
2. [Features](FEATURES.md) - All available capabilities
3. [Troubleshooting](TROUBLESHOOTING.md) - Solve common problems

### For Experts
1. [Algorithm Reference](ALGORITHMS.md) - Mathematical foundations
2. Source code - Implementation details
3. Benchmarks - Performance comparisons

## Examples by Domain

### Machine Learning
- [Logistic Regression Training](EXAMPLES.md#machine-learning-logistic-regression-from-scratch)
- [Neural Network Optimization](EXAMPLES.md#neural-network-training)
- [Hyperparameter Tuning](USAGE_GUIDE.md#hyperparameter-optimization)

### Engineering
- [Parameter Identification](EXAMPLES.md#parameter-identification-in-dynamical-systems)
- [Optimal Control](EXAMPLES.md#optimal-control-problem)
- [Design Optimization](FEATURES.md#engineering-applications)

### Finance
- [Portfolio Optimization](EXAMPLES.md#portfolio-optimization-with-risk-management)
- [Risk Management](USAGE_GUIDE.md#portfolio-optimization)
- [Asset Allocation](API_REFERENCE.md#finance-portfolio-optimization)

### Science
- [Protein Folding](EXAMPLES.md#protein-folding-energy-minimization)
- [Curve Fitting](GETTING_STARTED.md#tutorial-5-least-squares-fitting)
- [Parameter Estimation](TROUBLESHOOTING.md#engineering-parameter-estimation)

## Contributing

We welcome contributions! Areas where help is appreciated:

- **Examples**: More real-world applications
- **Documentation**: Improvements and clarifications
- **Algorithms**: New optimization methods
- **Performance**: Optimization and benchmarking
- **Testing**: Edge cases and validation

## Community and Support

- **GitHub Issues**: Bug reports and feature requests
- **Discussions**: Questions and community help
- **Documentation**: Comprehensive guides and examples
- **Examples**: Real-world use cases

## License

This project is dual-licensed under MIT and Apache 2.0 licenses. See [LICENSE](../LICENSE) for details.

---

**Ready to get started?** Head over to [Getting Started](GETTING_STARTED.md) for your first optimization problem!

**Need help?** Check the [Troubleshooting Guide](TROUBLESHOOTING.md) or open an issue on GitHub.

**Want to see examples?** Browse our [comprehensive examples](EXAMPLES.md) covering various domains and applications.