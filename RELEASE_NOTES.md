# SciRS2 0.1.0-alpha.4 Release Notes

## 0.1.0-alpha.4 (June 2025)

### Major Improvements

#### Enhanced Autograd Module
- **Refactored tensor operations**: Improved gradient computation and Jacobian calculation
- **Added conv2d_transpose operation**: Support for transposed convolution in neural networks
- **Fixed gradient propagation issues**: More accurate backpropagation through complex operations
- **Improved error handling**: Better error messages and context in gradient computations

#### Linear Algebra Enhancements
- **Fixed matrix_exp accuracy**: Corrected Padé coefficients for accurate matrix exponential computation
- **Added Hermitian eigenvalue decomposition tests**: Ensure accuracy for Hermitian matrices
- **Enhanced complex matrix operations**: Better support for complex decompositions

#### Array Protocol Improvements (scirs2-core)
- **Enhanced gradient support**: Better integration with autograd module
- **Improved neural network operations**: More efficient backpropagation support
- **Extended training utilities**: Better support for model training workflows

### Code Quality Improvements
- **Cleaned up repository**: Removed numerous temporary documentation files and work summaries
- **Reorganized documentation**: Moved documentation files to appropriate `docs/` directories
- **Improved code organization**: Better module structure and file organization
- **Test file cleanup**: Moved test files from root directories to appropriate tests/ or examples/ directories
- **Removed temporary files**: Cleaned up .bak, .old, and temporary output files across all modules

### Bug Fixes
- Fixed gradient computation issues in autograd module
- Corrected matrix exponential implementation in linalg
- Resolved various test failures and warnings
- Fixed Jacobian computation for neural network operations

### Testing Improvements
- Added comprehensive test examples for conv2d_transpose
- Added Jacobian computation tests
- Added matrix multiplication gradient tests
- Enhanced test coverage for complex operations

## 0.1.0-alpha.3 (May 2025)

## 0.1.0-alpha.2 (May 2025)

## 0.1.0-alpha.1 (April 2025)

# SciRS2 0.1.0 Release Notes

We're excited to announce the initial release of SciRS2 (Scientific Computing in Rust), a comprehensive scientific computing library designed to provide SciPy-compatible APIs while leveraging Rust's performance, safety, and concurrency features.

## Overview

SciRS2 is an ambitious project that aims to bring scientific computing capabilities to the Rust ecosystem with a modular design, comprehensive error handling, and a focus on performance. This initial 0.1.0 release includes a robust set of core modules covering various scientific computing domains, with additional modules available as previews.

## Core Features

### Modular Architecture
- **Independent Crates**: Each functional area is implemented as a separate crate
- **Flexible Dependencies**: Users can select only the features they need
- **Consistent Design**: Common patterns and abstractions across all modules
- **Comprehensive Error Handling**: Detailed error information and context

### Performance Optimization
- **SIMD Acceleration**: Vectorized operations via the `simd` feature
- **Parallel Processing**: Multi-threaded algorithms via the `parallel` feature
- **Caching Mechanisms**: Performance optimizations for repeated calculations
- **Memory Efficiency**: Algorithms designed for efficient memory usage

### Rust-First Approach
- **Type Safety**: Leveraging Rust's type system to prevent common errors
- **Generic Programming**: Flexible implementations that work with multiple numeric types
- **Trait-Based Design**: Well-defined traits for algorithm abstractions
- **Zero-Cost Abstractions**: High-level interfaces without compromising performance

## Module Status

### Stable Modules
- **scirs2-core**: Core utilities and common functionality
- **scirs2-linalg**: Linear algebra operations, decompositions, and solvers
- **scirs2-stats**: Statistical distributions, tests, and functions
- **scirs2-optimize**: Optimization algorithms and root finding
- **scirs2-interpolate**: Interpolation methods for 1D and ND data
- **scirs2-special**: Special mathematical functions
- **scirs2-fft**: Fast Fourier Transform operations
- **scirs2-signal**: Signal processing capabilities
- **scirs2-sparse**: Sparse matrix formats and operations
- **scirs2-spatial**: Spatial algorithms and data structures
- **scirs2-cluster**: Clustering algorithms (K-means, hierarchical)
- **scirs2-transform**: Data transformation utilities
- **scirs2-metrics**: Evaluation metrics for ML models

### Preview Modules
The following modules are included as previews and may undergo significant changes in future releases:
- **scirs2-ndimage**: N-dimensional image processing
- **scirs2-neural**: Neural network building blocks
- **scirs2-optim**: ML-specific optimization algorithms
- **scirs2-series**: Time series analysis
- **scirs2-text**: Text processing utilities
- **scirs2-io**: Input/output utilities
- **scirs2-datasets**: Dataset utilities
- **scirs2-graph**: Graph processing algorithms
- **scirs2-vision**: Computer vision operations
- **scirs2-autograd**: Automatic differentiation engine

## Key Capabilities

### Linear Algebra (scirs2-linalg)
- Matrix operations: determinants, inverses, etc.
- Matrix decompositions: LU, QR, SVD, Cholesky
- Eigenvalue/eigenvector computations
- Linear equation solvers
- BLAS and LAPACK bindings

### Statistics (scirs2-stats)
- Comprehensive distribution library (Normal, t, Chi-square, F, and more)
- Multivariate distributions (Multivariate Normal, Wishart, Dirichlet)
- Statistical tests (t-tests, normality tests, etc.)
- Random number generation with modern rand 0.9.0 API
- Sampling utilities (bootstrap, stratified sampling)

### Optimization (scirs2-optimize)
- Unconstrained minimization (Nelder-Mead, BFGS, Powell, Conjugate Gradient)
- Constrained minimization (SLSQP, Trust-region)
- Least squares minimization (Levenberg-Marquardt, Trust Region Reflective)
- Root finding algorithms (Broyden, Anderson, Krylov)

### Additional Functionality
- **Interpolation**: Linear, cubic, spline interpolation in 1D and ND
- **FFT**: Fast Fourier Transform with real and complex variants
- **Signal**: Filtering, convolution, and spectral analysis
- **Special**: Mathematical special functions (Bessel, Gamma, etc.)
- **Spatial**: K-D trees, distance calculations, spatial algorithms
- **Sparse**: Efficient sparse matrix formats and operations

## Installation

Add the following to your `Cargo.toml`:

```toml
[dependencies]
scirs2 = "0.1.0-alpha.4"  # Import the whole library
```

Or select only the modules you need:

```toml
[dependencies]
scirs2-linalg = "0.1.0-alpha.4"     # Linear algebra only
scirs2-stats = "0.1.0-alpha.4"      # Statistics only
scirs2-optimize = "0.1.0-alpha.4"   # Optimization only
```

You can also enable specific features:

```toml
[dependencies]
scirs2-core = { version = "0.1.0-alpha.4", features = ["simd", "parallel"] }
```

## Usage Examples

### Linear Algebra Operations
```rust
use scirs2_linalg::{basic, decomposition};
use ndarray::array;

// Create a matrix
let a = array![[1.0, 2.0], [3.0, 4.0]];

// Compute determinant and inverse
let det = basic::det(&a).unwrap();
let inv = basic::inv(&a).unwrap();

// Perform matrix decomposition
let svd = decomposition::svd(&a, true, true).unwrap();
println!("U: {:?}, S: {:?}, Vt: {:?}", svd.u, svd.s, svd.vt);
```

### Statistical Distributions
```rust
use scirs2_stats::distributions::normal::Normal;

// Create a normal distribution
let normal = Normal::new(0.0, 1.0).unwrap();

// Calculate PDF, CDF, and quantiles
let pdf = normal.pdf(1.0)?;
let cdf = normal.cdf(1.0)?;
let ppf = normal.ppf(0.975)?;

// Generate random samples
let samples = normal.random_sample(1000, None)?;
```

### Optimization
```rust
use scirs2_optimize::unconstrained;

// Define objective function and gradient
let f = |x: &[f64]| x[0].powi(2) + x[1].powi(2);
let df = |x: &[f64], grad: &mut [f64]| {
    grad[0] = 2.0 * x[0];
    grad[1] = 2.0 * x[1];
};

// Minimize using BFGS
let result = unconstrained::minimize(
    f, df, &[1.0, 1.0], "BFGS", None, None
).unwrap();

println!("Minimum at: {:?}, value: {}", result.x, result.fun);
```

## Roadmap

This is just the beginning for SciRS2. Our future plans include:

- **API Refinement**: Fine-tuning APIs based on community feedback
- **Additional Modules**: Completing implementation of IO, datasets, vision modules
- **Performance Optimization**: Continuous benchmarking and optimization
- **Extended Functionality**: Adding more algorithms and capabilities
- **Ecosystem Integration**: Better integration with the broader Rust ecosystem

## Acknowledgments

SciRS2 is inspired by SciPy and other scientific computing libraries. We thank the Rust community for the excellent ecosystem of crates that make this project possible.

## License

SciRS2 is dual-licensed under MIT License and Apache License, Version 2.0. You can choose to use either license. See the LICENSE file for details.

## Contributing

Contributions are welcome! See the CONTRIBUTING.md file for guidelines on how to contribute to SciRS2.

---

This release represents a significant milestone in bringing scientific computing capabilities to Rust. We invite you to try SciRS2, provide feedback, and join us in building a comprehensive scientific computing ecosystem for Rust!