# SciRS2: Scientific Computing in Rust

[![crates.io](https://img.shields.io/crates/v/scirs2.svg)](https://crates.io/crates/scirs2)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](../LICENSE)
[![Documentation](https://img.shields.io/docsrs/scirs2)](https://docs.rs/scirs2)

SciRS2 is a comprehensive scientific computing library for Rust, inspired by SciPy and designed to provide a complete ecosystem for numerical computation, statistical analysis, and scientific algorithms.

## Overview

This is the main SciRS2 crate, which provides a convenient facade over the ecosystem of specialized sub-crates. Each sub-crate focuses on a specific domain of scientific computing, while this crate re-exports their functionality in a unified interface.

## Features

SciRS2 brings together a large collection of scientific computing tools:

- **Core Utilities**: Common functionality through `scirs2-core`
- **Linear Algebra**: Matrix operations, decompositions, and solvers via `scirs2-linalg`
- **Statistics**: Distributions, hypothesis testing, and statistical functions in `scirs2-stats`
- **Optimization**: Minimization, root finding, and curve fitting with `scirs2-optimize`
- **Integration**: Numerical integration and ODE solvers through `scirs2-integrate`
- **Interpolation**: Various interpolation methods in `scirs2-interpolate`
- **Special Functions**: Mathematical special functions via `scirs2-special`
- **Fourier Analysis**: FFT and related transforms in `scirs2-fft`
- **Signal Processing**: Filtering, convolution, and spectral analysis in `scirs2-signal`
- **Sparse Matrices**: Efficient sparse matrix formats and operations via `scirs2-sparse`
- **Spatial Algorithms**: Spatial data structures and algorithms in `scirs2-spatial`
- **Image Processing**: Multidimensional image processing in `scirs2-ndimage`
- **Machine Learning**: Clustering, metrics, and neural networks via multiple sub-crates
- **Data I/O**: Reading and writing various scientific data formats via `scirs2-io`
- **And more**: Various additional modules for specific applications

## Installation

Add the following to your `Cargo.toml`:

```toml
[dependencies]
scirs2 = "0.1.0-alpha.4"  # Main package with default features
```

You can enable only the features you need:

```toml
[dependencies]
scirs2 = { version = "0.1.0-alpha.4", features = ["linalg", "stats", "optimize"] }
```

Or use specific modules directly:

```toml
[dependencies]
scirs2-core = "0.1.0-alpha.4"
scirs2-linalg = "0.1.0-alpha.4"
scirs2-stats = "0.1.0-alpha.4"
```

Basic usage examples:

```rust
use scirs2::prelude::*;
use ndarray::array;

fn main() -> CoreResult<()> {
    // Linear algebra operations
    let a = array![[1., 2.], [3., 4.]];
    let eig = linalg::eigen::eig(&a)?;
    println!("Eigenvalues: {:?}", eig.eigenvalues);
    println!("Eigenvectors: {:?}", eig.eigenvectors);
    
    // Statistical distributions
    let normal = stats::distributions::normal::Normal::new(0.0, 1.0)?;
    let samples = normal.random_sample(1000, None)?;
    let mean = stats::descriptive::mean(&samples)?;
    let std_dev = stats::descriptive::std_dev(&samples, None)?;
    println!("Sample mean: {}, std dev: {}", mean, std_dev);
    
    // Optimization
    let f = |x: &[f64]| x[0].powi(2) + x[1].powi(2);
    let df = |x: &[f64], grad: &mut [f64]| {
        grad[0] = 2.0 * x[0];
        grad[1] = 2.0 * x[1];
    };
    let result = optimize::unconstrained::minimize(
        f, df, &[1.0, 1.0], "L-BFGS-B", None, None)?;
    println!("Optimization result: {:?}", result);
    
    // Special functions
    let gamma = special::gamma::gamma(5.0)?;
    println!("Gamma(5) = {}", gamma);
    
    // FFT
    let signal = array![1.0, 2.0, 3.0, 4.0];
    let fft_result = fft::fft(&signal)?;
    println!("FFT result: {:?}", fft_result);
    
    Ok(())
}
```

## Feature Flags

This crate uses feature flags to control which sub-crates are included:

- `core`: Core utilities (always enabled)
- `linalg`: Linear algebra operations
- `stats`: Statistical functions and distributions
- `optimize`: Optimization algorithms
- `integrate`: Numerical integration and ODEs
- `interpolate`: Interpolation methods
- `fft`: Fast Fourier Transform
- `special`: Special functions
- `signal`: Signal processing
- `sparse`: Sparse matrices
- `spatial`: Spatial algorithms
- `ndimage`: N-dimensional image processing
- `cluster`: Clustering algorithms
- `datasets`: Dataset utilities
- `io`: I/O utilities
- `neural`: Neural networks
- `optim`: Optimization for machine learning
- `graph`: Graph algorithms
- `transform`: Data transformation
- `metrics`: Evaluation metrics
- `text`: Text processing
- `vision`: Computer vision
- `series`: Time series analysis
- `autograd`: Automatic differentiation

## Architecture

SciRS2 follows a modular architecture where each domain of scientific computing is implemented in a separate crate. This main crate provides a unified interface by re-exporting their functionality.

### Architecture Benefits

- **Modular Development**: Each domain can be developed and tested independently
- **Reduced Compilation Time**: Users can include only the features they need
- **Flexible Dependencies**: Sub-crates can have different dependency requirements
- **Focused Documentation**: Each domain has its own focused documentation

### Module Structure

```
scirs2
├── core       // Core utilities
├── linalg     // Linear algebra
├── stats      // Statistics
├── optimize   // Optimization
├── integrate  // Integration
├── interpolate // Interpolation
├── fft        // Fourier transforms
├── special    // Special functions
├── signal     // Signal processing
├── sparse     // Sparse matrices
├── spatial    // Spatial algorithms
├── ndimage    // Image processing
└── ...        // Other modules
```

## Performance

SciRS2 is designed with performance in mind:

- Uses optimized BLAS and LAPACK implementations where appropriate
- Leverages SIMD operations when available
- Provides parallel processing capabilities
- Uses memory-efficient algorithms
- Employs caching for expensive computations

## Contributing

See the [CONTRIBUTING.md](../CONTRIBUTING.md) file for contribution guidelines.

## License

This project is dual-licensed under:

- [MIT License](../LICENSE-MIT)
- [Apache License Version 2.0](../LICENSE-APACHE)

You can choose to use either license. See the [LICENSE](../LICENSE) file for details.
