# SciRS2 - Scientific Computing and AI in Rust

[![crates.io](https://img.shields.io/crates/v/scirs2.svg)](https://crates.io/crates/scirs2)
[![License](https://img.shields.io/crates/l/scirs2.svg)](LICENSE)
![SciRS2 CI](https://github.com/cool-japan/scirs/workflows/SciRS2%20CI/badge.svg)
![Advanced & AI/ML Modules](https://github.com/cool-japan/scirs/workflows/SciRS2%20Advanced%20%26%20AI%2FML%20Modules/badge.svg)
![Documentation](https://github.com/cool-japan/scirs/workflows/SciRS2%20Documentation/badge.svg)

SciRS2 is an ambitious project to provide a comprehensive scientific computing and AI/ML infrastructure in Rust, providing SciPy-compatible APIs while leveraging Rust's performance and safety features.

## Project Goals

- Create a comprehensive scientific computing and machine learning library in Rust
- Maintain API compatibility with SciPy where reasonable
- Provide specialized tools for AI and machine learning development
- Leverage Rust's performance, safety, and concurrency features
- Build a sustainable open-source ecosystem for scientific and AI computing in Rust

## Project Structure

SciRS2 adopts a modular architecture with separate crates for different functional areas, using Rust's workspace feature to manage them:

```
/
# Core Scientific Computing Modules
├── Cargo.toml                # Workspace configuration
├── scirs2-core/              # Core utilities and common functionality
├── scirs2-linalg/            # Linear algebra module
├── scirs2-integrate/         # Numerical integration
├── scirs2-interpolate/       # Interpolation algorithms
├── scirs2-optimize/          # Optimization algorithms
├── scirs2-fft/               # Fast Fourier Transform
├── scirs2-stats/             # Statistical functions
├── scirs2-special/           # Special mathematical functions
├── scirs2-signal/            # Signal processing
├── scirs2-sparse/            # Sparse matrix operations
├── scirs2-spatial/           # Spatial algorithms

# Future Advanced Modules
├── scirs2-cluster/           # Clustering algorithms
├── scirs2-ndimage/           # N-dimensional image processing
├── scirs2-io/                # Input/output utilities
├── scirs2-datasets/          # Sample datasets and loaders

# AI/ML Modules
├── scirs2-autograd/          # Automatic differentiation engine
├── scirs2-neural/            # Neural network building blocks
├── scirs2-optim/             # ML-specific optimization algorithms
├── scirs2-graph/             # Graph processing algorithms
├── scirs2-transform/         # Data transformation utilities
├── scirs2-metrics/           # ML evaluation metrics
├── scirs2-text/              # Text processing utilities
├── scirs2-vision/            # Computer vision operations
├── scirs2-series/            # Time series analysis

# Main Integration Crate
└── scirs2/                   # Main integration crate
    ├── Cargo.toml
    └── src/
        └── lib.rs            # Re-exports from all other crates
```

This modular architecture offers several advantages:
- **Flexible Dependencies**: Users can select only the features they need
- **Independent Development**: Each module can be developed and tested separately
- **Clear Separation**: Each module focuses on a specific functional area
- **No Circular Dependencies**: Clear hierarchy prevents circular dependencies
- **AI/ML Focus**: Specialized modules for machine learning and AI workloads

## Module Documentation

Each module has its own README with detailed documentation and is available on crates.io:

### Main Integration Crate
- [**scirs2**](scirs2/README.md): Main integration crate [![crates.io](https://img.shields.io/crates/v/scirs2.svg)](https://crates.io/crates/scirs2)

### Core Modules
- [**scirs2-core**](scirs2-core/README.md): Core utilities and common functionality [![crates.io](https://img.shields.io/crates/v/scirs2-core.svg)](https://crates.io/crates/scirs2-core)
- [**scirs2-linalg**](scirs2-linalg/README.md): Linear algebra module [![crates.io](https://img.shields.io/crates/v/scirs2-linalg.svg)](https://crates.io/crates/scirs2-linalg)
- [**scirs2-integrate**](scirs2-integrate/README.md): Numerical integration [![crates.io](https://img.shields.io/crates/v/scirs2-integrate.svg)](https://crates.io/crates/scirs2-integrate)
- [**scirs2-interpolate**](scirs2-interpolate/README.md): Interpolation algorithms [![crates.io](https://img.shields.io/crates/v/scirs2-interpolate.svg)](https://crates.io/crates/scirs2-interpolate)
- [**scirs2-optimize**](scirs2-optimize/README.md): Optimization algorithms [![crates.io](https://img.shields.io/crates/v/scirs2-optimize.svg)](https://crates.io/crates/scirs2-optimize)
- [**scirs2-fft**](scirs2-fft/README.md): Fast Fourier Transform [![crates.io](https://img.shields.io/crates/v/scirs2-fft.svg)](https://crates.io/crates/scirs2-fft)
- [**scirs2-stats**](scirs2-stats/README.md): Statistical functions [![crates.io](https://img.shields.io/crates/v/scirs2-stats.svg)](https://crates.io/crates/scirs2-stats)
- [**scirs2-special**](scirs2-special/README.md): Special mathematical functions [![crates.io](https://img.shields.io/crates/v/scirs2-special.svg)](https://crates.io/crates/scirs2-special)
- [**scirs2-signal**](scirs2-signal/README.md): Signal processing [![crates.io](https://img.shields.io/crates/v/scirs2-signal.svg)](https://crates.io/crates/scirs2-signal)
- [**scirs2-sparse**](scirs2-sparse/README.md): Sparse matrix operations [![crates.io](https://img.shields.io/crates/v/scirs2-sparse.svg)](https://crates.io/crates/scirs2-sparse)
- [**scirs2-spatial**](scirs2-spatial/README.md): Spatial algorithms [![crates.io](https://img.shields.io/crates/v/scirs2-spatial.svg)](https://crates.io/crates/scirs2-spatial)

### Advanced Modules
- [**scirs2-cluster**](scirs2-cluster/README.md): Clustering algorithms [![crates.io](https://img.shields.io/crates/v/scirs2-cluster.svg)](https://crates.io/crates/scirs2-cluster)
- [**scirs2-ndimage**](scirs2-ndimage/README.md): N-dimensional image processing [![crates.io](https://img.shields.io/crates/v/scirs2-ndimage.svg)](https://crates.io/crates/scirs2-ndimage)
- [**scirs2-io**](scirs2-io/README.md): Input/output utilities [![crates.io](https://img.shields.io/crates/v/scirs2-io.svg)](https://crates.io/crates/scirs2-io)
- [**scirs2-datasets**](scirs2-datasets/README.md): Sample datasets and loaders [![crates.io](https://img.shields.io/crates/v/scirs2-datasets.svg)](https://crates.io/crates/scirs2-datasets)

### AI/ML Modules
- [**scirs2-autograd**](scirs2-autograd/README.md): Automatic differentiation engine [![crates.io](https://img.shields.io/crates/v/scirs2-autograd.svg)](https://crates.io/crates/scirs2-autograd)
- [**scirs2-neural**](scirs2-neural/README.md): Neural network building blocks [![crates.io](https://img.shields.io/crates/v/scirs2-neural.svg)](https://crates.io/crates/scirs2-neural)
- [**scirs2-optim**](scirs2-optim/README.md): ML-specific optimization algorithms [![crates.io](https://img.shields.io/crates/v/scirs2-optim.svg)](https://crates.io/crates/scirs2-optim)
- [**scirs2-graph**](scirs2-graph/README.md): Graph processing algorithms [![crates.io](https://img.shields.io/crates/v/scirs2-graph.svg)](https://crates.io/crates/scirs2-graph)
- [**scirs2-transform**](scirs2-transform/README.md): Data transformation utilities [![crates.io](https://img.shields.io/crates/v/scirs2-transform.svg)](https://crates.io/crates/scirs2-transform)
- [**scirs2-metrics**](scirs2-metrics/README.md): ML evaluation metrics [![crates.io](https://img.shields.io/crates/v/scirs2-metrics.svg)](https://crates.io/crates/scirs2-metrics)
- [**scirs2-text**](scirs2-text/README.md): Text processing utilities [![crates.io](https://img.shields.io/crates/v/scirs2-text.svg)](https://crates.io/crates/scirs2-text)
- [**scirs2-vision**](scirs2-vision/README.md): Computer vision operations [![crates.io](https://img.shields.io/crates/v/scirs2-vision.svg)](https://crates.io/crates/scirs2-vision)
- [**scirs2-series**](scirs2-series/README.md): Time series analysis [![crates.io](https://img.shields.io/crates/v/scirs2-series.svg)](https://crates.io/crates/scirs2-series)

## Implementation Strategy

We follow a phased approach:

1. **Core functionality analysis**: Identify key features and APIs of each SciPy module
2. **Prioritization**: Begin with highest-demand modules (linalg, stats, optimize)
3. **Interface design**: Balance Rust idioms with SciPy compatibility
4. **Scientific computing foundation**: Implement core scientific computing modules first
5. **Advanced modules**: Implement specialized modules for advanced scientific computing
6. **AI/ML infrastructure**: Develop specialized tools for AI and machine learning
7. **Integration and optimization**: Ensure all modules work together efficiently

## Dependencies

SciRS2 leverages the Rust ecosystem:

### Core Dependencies
- `ndarray` or `nalgebra`: Multidimensional array operations
- `num`: Numeric abstractions
- `rayon`: Parallel processing
- `rustfft`: Fast Fourier transforms
- `ndarray-linalg` or `linxal`: Linear algebra computations
- `argmin`: Optimization algorithms
- `rand` and `rand_distr`: Random number generation and distributions

### Future AI/ML Dependencies
- `tch-rs`: Bindings to the PyTorch C++ API
- `burn`: Pure Rust neural network framework
- `pandrs`: Fast DataFrame library for data manipulation (upcoming crate)
- `numrs`: NumPy-like array operations in Rust (upcoming crate)
- `tokenizers`: Fast tokenization utilities
- `image`: Image processing utilities
- `petgraph`: Graph algorithms and data structures

## Contributing

Contributions are welcome! Please see our [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under [Apache License 2.0](LICENSE).

## Installation and Usage

SciRS2 and all its modules are available on [crates.io](https://crates.io/crates/scirs2). You can add them to your project using Cargo:

```toml
# Add the main integration crate for all functionality
[dependencies]
scirs2 = "0.1.0-alpha.2"
```

Or include only the specific modules you need:

```toml
[dependencies]
# Core utilities
scirs2-core = "0.1.0-alpha.1"

# Scientific computing modules
scirs2-linalg = "0.1.0-alpha.1"
scirs2-stats = "0.1.0-alpha.1"
scirs2-optimize = "0.1.0-alpha.1"

# AI/ML modules
scirs2-neural = "0.1.0-alpha.1"
scirs2-autograd = "0.1.0-alpha.1"
```

### Example Usage

```rust
// Using the main integration crate
use scirs2::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Example with linear algebra (when feature is enabled)
    #[cfg(feature = "linalg")]
    {
        use ndarray::Array2;
        
        // Create two matrices
        let a = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0])?;
        let b = Array2::from_shape_vec((2, 2), vec![5.0, 6.0, 7.0, 8.0])?;
        
        // Use the linalg module
        if let Ok(result) = scirs2::linalg::basic::matrix_multiply(&a, &b) {
            println!("Matrix multiplication result: {:?}", result);
        }
    }
    
    Ok(())
}
```

Or import specific modules directly:

```rust
use scirs2_linalg::decomposition::svd;
use scirs2_stats::distributions::normal::Normal;
use ndarray::Array2;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a matrix for SVD
    let a = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
    
    // Perform SVD
    let (u, s, vt) = svd(&a)?;
    println!("SVD result: U={:?}, S={:?}, Vt={:?}", u, s, vt);
    
    // Create and sample from a normal distribution
    let normal = Normal::new(0.0, 1.0)?;
    let samples = normal.random_sample(5);
    println!("Samples from normal distribution: {:?}", samples);
    
    Ok(())
}
```

## Current Status

### Completed Modules

The following core scientific computing modules have been implemented:

- **Linear Algebra Module** (`scirs2-linalg`): Basic matrix operations, decompositions (LU, QR, SVD), eigenvalue problems, BLAS/LAPACK interfaces
- **Statistics Module** (`scirs2-stats`): Descriptive statistics, distributions, statistical tests, regression models
- **Optimization Module** (`scirs2-optimize`): Unconstrained & constrained optimization, least squares, root finding
- **Integration Module** (`scirs2-integrate`): Numerical integration, ODE solvers
- **Interpolation Module** (`scirs2-interpolate`): 1D & ND interpolation, splines
- **Signal Processing** (`scirs2-signal`): Filtering, convolution, spectral analysis
- **FFT Module** (`scirs2-fft`): FFT, inverse FFT, DCT, DST
- **Sparse Matrix** (`scirs2-sparse`): Various sparse matrix formats and operations
- **Special Functions** (`scirs2-special`): Mathematical special functions
- **Spatial Algorithms** (`scirs2-spatial`): KD-trees, distance calculations

### Current Phase: AI and Machine Learning Modules

We have completed the implementation of core scientific computing modules, advanced modules, and now we're focusing on AI/ML infrastructure:

#### Completed Advanced Modules:
- **Clustering** (`scirs2-cluster`): Vector quantization, hierarchical clustering, density-based clustering
- **N-dimensional Image Processing** (`scirs2-ndimage`): Filtering, morphology, measurements
- **I/O utilities** (`scirs2-io`): MATLAB, WAV, and ARFF file formats, image handling
- **Datasets** (`scirs2-datasets`): Sample datasets and loaders

#### Completed AI/ML Modules:
- **Automatic Differentiation** (`scirs2-autograd`): Reverse-mode automatic differentiation, tensor operations, neural network primitives
- **Neural Networks** (`scirs2-neural`): Neural network layers, activations, loss functions
- **ML Optimization** (`scirs2-optim`): Optimizers, learning rate schedulers, regularization techniques
- **Graph Processing** (`scirs2-graph`): Graph algorithms and data structures
- **Data Transformation** (`scirs2-transform`): Feature engineering, normalization
- **Evaluation Metrics** (`scirs2-metrics`): Classification, regression metrics
- **Text Processing** (`scirs2-text`): Tokenization, vectorization
- **Computer Vision** (`scirs2-vision`): Image processing, feature detection
- **Time Series Analysis** (`scirs2-series`): Decomposition, forecasting

### Current Focus: Refinement and Optimization

We are now focusing on refining the existing modules, improving performance, and ensuring comprehensive test coverage and documentation:
- Performance optimization
- API refinement
- Comprehensive documentation
- Integration testing
- User feedback incorporation
- Benchmark comparison with other libraries

### Publication Status

All SciRS2 modules have been published to crates.io as alpha releases (0.1.0-alpha.1). This marks an important milestone in the project, making all components available to the Rust community while we continue to refine and improve them.

- **Alpha Release (0.1.0-alpha.1)**: All modules released on crates.io (April 2025)
- **Future Beta Release (0.1.0-beta.1)**: Planned after incorporating initial feedback
- **Future First Stable Release (0.1.0)**: Targeted after API stabilization and comprehensive testing

For more detailed information on development status and roadmap, check the [TODO.md](TODO.md) file.