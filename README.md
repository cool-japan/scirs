# SciRS2 - Scientific Computing and AI in Rust

[![crates.io](https://img.shields.io/crates/v/scirs2.svg)](https://crates.io/crates/scirs2)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)
![SciRS2 CI](https://github.com/cool-japan/scirs/workflows/SciRS2%20CI/badge.svg)

SciRS2 is a comprehensive scientific computing and AI/ML infrastructure in Rust, providing SciPy-compatible APIs while leveraging Rust's performance, safety, and concurrency features. The project aims to provide a complete ecosystem for scientific computing, data analysis, and machine learning in Rust.

## Features

### Scientific Computing
- **Linear Algebra**: Matrix operations, decompositions, eigensolvers, and specialized matrix types
- **Statistics**: Distributions, descriptive statistics, tests, and regression models
- **Optimization**: Unconstrained and constrained optimization, root finding, and least squares
- **Integration**: Numerical integration, ODE solvers, and boundary value problems
- **Interpolation**: Linear, spline, and multi-dimensional interpolation
- **Special Functions**: Mathematical special functions including Bessel, gamma, and elliptic functions
- **Signal Processing**: FFT, wavelet transforms, filtering, and spectral analysis
- **Sparse Matrices**: Multiple sparse matrix formats and operations
- **Spatial Algorithms**: Distance calculations, KD-trees, and spatial data structures

### Advanced Features
- **N-dimensional Image Processing**: Filtering, feature detection, and segmentation
- **Clustering**: K-means, hierarchical, and density-based clustering
- **I/O Utilities**: Scientific data format reading and writing
- **Sample Datasets**: Data generation and loading tools

### AI and Machine Learning
- **Automatic Differentiation**: Reverse-mode and forward-mode autodiff engine
- **Neural Networks**: Layers, optimizers, and model architectures
- **Graph Processing**: Graph algorithms and data structures
- **Data Transformation**: Feature engineering and normalization
- **Metrics**: Evaluation metrics for ML models
- **Text Processing**: Tokenization and text analysis tools
- **Computer Vision**: Image processing and feature detection
- **Time Series**: Analysis and forecasting tools

### Performance and Safety
- **Memory Management**: Efficient handling of large datasets
- **GPU Acceleration**: CUDA and hardware-agnostic backends for computation
- **Parallelization**: Multi-core processing for compute-intensive operations
- **Safety**: Memory safety and thread safety through Rust's ownership model
- **Type Safety**: Strong typing and compile-time checks
- **Error Handling**: Comprehensive error system with context

## Project Scale

This project now contains **over 1.5 million lines of code** and runs **over 6,500 tests** across all modules, demonstrating the comprehensive nature of the SciRS2 ecosystem.

## Project Goals

- Create a comprehensive scientific computing and machine learning library in Rust
- Maintain API compatibility with SciPy where reasonable
- Provide specialized tools for AI and machine learning development
- Leverage Rust's performance, safety, and concurrency features
- Build a sustainable open-source ecosystem for scientific and AI computing in Rust
- Offer performance similar to or better than Python-based solutions
- Provide a smooth migration path for SciPy users

## Project Structure

SciRS2 adopts a modular architecture with separate crates for different functional areas, using Rust's workspace feature to manage them:

```
/
# Core Scientific Computing Modules
├── Cargo.toml                # Workspace configuration
├── scirs2-core/              # Core utilities and common functionality
├── scirs2-autograd/          # Automatic differentiation engine
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

# Advanced Modules
├── scirs2-cluster/           # Clustering algorithms
├── scirs2-ndimage/           # N-dimensional image processing
├── scirs2-io/                # Input/output utilities
├── scirs2-datasets/          # Sample datasets and loaders

# AI/ML Modules
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

### Architectural Benefits

This modular architecture offers several advantages:
- **Flexible Dependencies**: Users can select only the features they need
- **Independent Development**: Each module can be developed and tested separately
- **Clear Separation**: Each module focuses on a specific functional area
- **No Circular Dependencies**: Clear hierarchy prevents circular dependencies
- **AI/ML Focus**: Specialized modules for machine learning and AI workloads
- **Feature Flags**: Granular control over enabled functionality
- **Memory Efficiency**: Import only what you need to reduce overhead

## Advanced Core Features

The core module (scirs2-core) provides several advanced features that are leveraged across the ecosystem:

### GPU Acceleration

```rust
use scirs2_core::gpu::{GpuContext, GpuBackend, GpuBuffer};

// Create a GPU context with the default backend
let ctx = GpuContext::new(GpuBackend::default())?;

// Allocate memory on the GPU
let mut buffer = ctx.create_buffer::<f32>(1024);

// Execute a computation
ctx.execute(|compiler| {
    let kernel = compiler.compile(kernel_code)?;
    kernel.set_buffer(0, &mut buffer);
    kernel.dispatch([1024, 1, 1]);
    Ok(())
})?;
```

### Memory Management

```rust
use scirs2_core::memory::{ChunkProcessor2D, BufferPool, ZeroCopyView};

// Process large arrays in chunks
let mut processor = ChunkProcessor2D::new(&large_array, (1000, 1000));
processor.process_chunks(|chunk, coords| {
    // Process each chunk...
});

// Reuse memory with buffer pools
let mut pool = BufferPool::<f64>::new();
let mut buffer = pool.acquire_vec(1000);
// Use buffer...
pool.release_vec(buffer);
```

### Memory Metrics and Profiling

```rust
use scirs2_core::memory::metrics::{track_allocation, generate_memory_report};
use scirs2_core::profiling::{Profiler, Timer};

// Track memory allocations
track_allocation("MyComponent", 1024, 0x1000);

// Time a block of code
let timer = Timer::start("matrix_multiply");
// Do work...
timer.stop();

// Print profiling report
Profiler::global().lock().unwrap().print_report();
```

## Module Documentation

Each module has its own README with detailed documentation and is available on crates.io:

### Main Integration Crate
- [**scirs2**](scirs2/README.md): Main integration crate [![crates.io](https://img.shields.io/crates/v/scirs2.svg)](https://crates.io/crates/scirs2)

### Core Modules
- [**scirs2-core**](scirs2-core/README.md): Core utilities and common functionality [![crates.io](https://img.shields.io/crates/v/scirs2-core.svg)](https://crates.io/crates/scirs2-core)
- [**scirs2-linalg**](scirs2-linalg/README.md): Linear algebra module [![crates.io](https://img.shields.io/crates/v/scirs2-linalg.svg)](https://crates.io/crates/scirs2-linalg)
- [**scirs2-autograd**](scirs2-autograd/README.md): Automatic differentiation engine [![crates.io](https://img.shields.io/crates/v/scirs2-autograd.svg)](https://crates.io/crates/scirs2-autograd)
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
8. **Ecosystem development**: Create tooling, documentation, and community resources

## Core Module Usage Policy

All modules in the SciRS2 ecosystem are expected to leverage functionality from scirs2-core:

- **Validation**: Use `scirs2-core::validation` for parameter checking
- **Error Handling**: Base module-specific errors on `scirs2-core::error::CoreError`
- **Numeric Operations**: Use `scirs2-core::numeric` for generic numeric functions
- **Optimization**: Use core-provided performance optimizations:
  - SIMD operations via `scirs2-core::simd`
  - Parallelism via `scirs2-core::parallel`
  - Memory management via `scirs2-core::memory`
  - Caching via `scirs2-core::cache`

## Dependency Management

SciRS2 uses workspace inheritance for consistent dependency versioning:

- All shared dependencies are defined in the root `Cargo.toml`
- Module crates reference dependencies with `workspace = true`
- Feature-gated dependencies use `workspace = true` with `optional = true`

```toml
# In workspace root Cargo.toml
[workspace.dependencies]
ndarray = { version = "0.16.1", features = ["serde", "rayon"] }
num-complex = "0.4.3"
rayon = "1.7.0"

# In module Cargo.toml
[dependencies]
ndarray = { workspace = true }
num-complex = { workspace = true }
rayon = { workspace = true, optional = true }

[features]
parallel = ["rayon"]
```

## Core Dependencies

SciRS2 leverages the Rust ecosystem:

### Core Dependencies
- `ndarray`: Multidimensional array operations
- `num`: Numeric abstractions
- `rayon`: Parallel processing
- `rustfft`: Fast Fourier transforms
- `ndarray-linalg`: Linear algebra computations
- `argmin`: Optimization algorithms
- `rand` and `rand_distr`: Random number generation and distributions

### AI/ML Dependencies
- `tch-rs`: Bindings to the PyTorch C++ API
- `burn`: Pure Rust neural network framework
- `tokenizers`: Fast tokenization utilities
- `image`: Image processing utilities
- `petgraph`: Graph algorithms and data structures

## Installation and Usage

SciRS2 and all its modules are available on [crates.io](https://crates.io/crates/scirs2). You can add them to your project using Cargo:

```toml
# Add the main integration crate for all functionality
[dependencies]
scirs2 = "0.1.0-alpha.6"
```

Or include only the specific modules you need:

```toml
[dependencies]
# Core utilities
scirs2-core = "0.1.0-alpha.6"

# Scientific computing modules
scirs2-linalg = "0.1.0-alpha.6"
scirs2-stats = "0.1.0-alpha.6"
scirs2-optimize = "0.1.0-alpha.6"

# AI/ML modules
scirs2-neural = "0.1.0-alpha.6"
scirs2-autograd = "0.1.0-alpha.6"
```

### Example Usage

#### Basic Scientific Computing

```rust
// Using the main integration crate
use scirs2::prelude::*;
use ndarray::Array2;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a matrix
    let a = Array2::from_shape_vec((3, 3), vec![
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0
    ])?;
    
    // Perform matrix operations
    let (u, s, vt) = scirs2::linalg::decomposition::svd(&a)?;
    
    println!("Singular values: {:.4?}", s);
    
    // Compute the condition number
    let cond = scirs2::linalg::basic::condition(&a, None)?;
    println!("Condition number: {:.4}", cond);
    
    // Generate random samples from a distribution
    let normal = scirs2::stats::distributions::normal::Normal::new(0.0, 1.0)?;
    let samples = normal.random_sample(5, None)?;
    println!("Random samples: {:.4?}", samples);
    
    Ok(())
}
```

#### Neural Network Example

```rust
use scirs2_neural::layers::{Dense, Layer};
use scirs2_neural::activations::{ReLU, Sigmoid};
use scirs2_neural::models::sequential::Sequential;
use scirs2_neural::losses::mse::MSE;
use scirs2_neural::optimizers::sgd::SGD;
use ndarray::{Array, Array2};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a simple feedforward neural network
    let mut model = Sequential::new();
    
    // Add layers
    model.add(Dense::new(2, 8)?);
    model.add(ReLU::new());
    model.add(Dense::new(8, 4)?);
    model.add(ReLU::new());
    model.add(Dense::new(4, 1)?);
    model.add(Sigmoid::new());
    
    // Compile the model
    let loss = MSE::new();
    let optimizer = SGD::new(0.01);
    model.compile(loss, optimizer);
    
    // Create dummy data
    let x = Array2::from_shape_vec((4, 2), vec![
        0.0, 0.0,
        0.0, 1.0,
        1.0, 0.0,
        1.0, 1.0
    ])?;
    
    let y = Array2::from_shape_vec((4, 1), vec![
        0.0,
        1.0,
        1.0,
        0.0
    ])?;
    
    // Train the model
    model.fit(&x, &y, 1000, Some(32), Some(true));
    
    // Make predictions
    let predictions = model.predict(&x);
    println!("Predictions: {:.4?}", predictions);
    
    Ok(())
}
```

#### GPU-Accelerated Example

```rust
use scirs2_core::gpu::{GpuContext, GpuBackend};
use scirs2_linalg::batch::matrix_multiply_gpu;
use ndarray::Array3;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create GPU context
    let ctx = GpuContext::new(GpuBackend::default())?;
    
    // Create batch of matrices (batch_size x m x n)
    let a_batch = Array3::<f32>::ones((64, 128, 256));
    let b_batch = Array3::<f32>::ones((64, 256, 64));
    
    // Perform batch matrix multiplication on GPU
    let result = matrix_multiply_gpu(&ctx, &a_batch, &b_batch)?;
    
    println!("Batch matrix multiply result shape: {:?}", result.shape());
    
    Ok(())
}
```

## Current Status

### Stable Modules

The following SciRS2 modules are considered stable with well-tested core functionality:

#### Core Scientific Computing Modules
- **Linear Algebra Module** (`scirs2-linalg`): Basic matrix operations, decompositions, eigenvalue problems
- **Statistics Module** (`scirs2-stats`): Descriptive statistics, distributions, statistical tests, regression
- **Optimization Module** (`scirs2-optimize`): Unconstrained & constrained optimization, least squares, root finding
- **Integration Module** (`scirs2-integrate`): Numerical integration, ODE solvers
- **Interpolation Module** (`scirs2-interpolate`): 1D & ND interpolation, splines
- **Signal Processing** (`scirs2-signal`): Filtering, convolution, spectral analysis, wavelets
- **FFT Module** (`scirs2-fft`): FFT, inverse FFT, real FFT, DCT, DST, Hermitian FFT
- **Sparse Matrix** (`scirs2-sparse`): CSR, CSC, COO, BSR, DIA, DOK, LIL formats and operations
- **Special Functions** (`scirs2-special`): Gamma, Bessel, elliptic, orthogonal polynomials
- **Spatial Algorithms** (`scirs2-spatial`): KD-trees, distance calculations, convex hull, Voronoi diagrams
- **Clustering** (`scirs2-cluster`): K-means, hierarchical clustering, DBSCAN
- **Data Transformation** (`scirs2-transform`): Feature engineering, normalization
- **Evaluation Metrics** (`scirs2-metrics`): Classification, regression metrics

### Preview Modules

The following modules are in preview state and may undergo API changes:

#### Advanced Modules
- **N-dimensional Image Processing** (`scirs2-ndimage`): Filtering, morphology, measurements
- **I/O utilities** (`scirs2-io`): MATLAB, WAV, ARFF file formats, CSV
- **Datasets** (`scirs2-datasets`): Sample datasets and loaders

#### AI/ML Modules
- **Automatic Differentiation** (`scirs2-autograd`): Tensor ops, neural network primitives
- **Neural Networks** (`scirs2-neural`): Layers, activations, loss functions
- **ML Optimization** (`scirs2-optim`): Optimizers, schedulers, regularization
- **Graph Processing** (`scirs2-graph`): Graph algorithms and data structures
- **Text Processing** (`scirs2-text`): Tokenization, vectorization, word embeddings
- **Computer Vision** (`scirs2-vision`): Image processing, feature detection
- **Time Series Analysis** (`scirs2-series`): Decomposition, forecasting

### Advanced Core Features Implemented

- **GPU Acceleration** with backend abstraction layer (CUDA, WebGPU, Metal)
- **Memory Management** for large-scale computations
- **Logging and Diagnostics** with progress tracking
- **Profiling** with timing and memory tracking
- **Memory Metrics** for detailed memory usage analysis
- **Optimized SIMD Operations** for performance-critical code

### Key Capabilities

SciRS2 provides:
- **Advanced Error Handling**: Comprehensive error framework with recovery strategies, async support, and diagnostics engine
- **Computer Vision Registration**: Rigid, affine, homography, and non-rigid registration algorithms with RANSAC robustness
- **Performance Benchmarking**: Automated benchmarking framework with SciPy comparison and optimization tools
- **Numerical Precision**: High-precision eigenvalue solvers and optimized numerical algorithms

### Installation

All SciRS2 modules are available on crates.io. Add the modules you need to your `Cargo.toml`:

```toml
[dependencies]
scirs2 = "0.1.0-alpha.6"  # Core library with all modules
# Or individual modules:
scirs2-linalg = "0.1.0-alpha.6"  # Linear algebra
scirs2-stats = "0.1.0-alpha.6"   # Statistics
# ... and more
```

For development roadmap and contribution guidelines, see [TODO.md](TODO.md) and [CONTRIBUTING.md](CONTRIBUTING.md).

## Performance Characteristics

SciRS2 prioritizes performance through several strategies:

- **SIMD Vectorization**: CPU vector instructions for numerical operations
- **Cache Efficiency**: Algorithms designed for modern CPU cache hierarchies
- **GPU Acceleration**: Hardware acceleration for compute-intensive operations
- **Memory Management**: Efficient allocation strategies for large datasets
- **Parallelism**: Multi-core utilization via Rayon
- **Zero-cost Abstractions**: Rust's compiler optimizations eliminate runtime overhead

Initial benchmarks on core operations show performance comparable to or exceeding NumPy/SciPy:

| Operation | SciRS2 (ms) | NumPy/SciPy (ms) | Speedup |
|-----------|-------------|------------------|---------|
| Matrix multiplication (1000×1000) | 18.5 | 23.2 | 1.25× |
| SVD decomposition (500×500) | 112.3 | 128.7 | 1.15× |
| FFT (1M points) | 8.7 | 11.5 | 1.32× |
| Normal distribution sampling (10M) | 42.1 | 67.9 | 1.61× |
| K-means clustering (100K points) | 321.5 | 378.2 | 1.18× |

*Note: Performance may vary based on hardware, compiler optimization, and specific workloads.*

## Contributing

Contributions are welcome! Please see our [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Areas Where We Need Help

- **Core Algorithm Implementation**: Implementing remaining algorithms from SciPy
- **Performance Optimization**: Improving performance of existing implementations
- **Documentation**: Writing examples, tutorials, and API documentation
- **Testing**: Expanding test coverage and creating property-based tests
- **Integration with Other Ecosystems**: Python bindings, WebAssembly support
- **Domain-Specific Extensions**: Financial algorithms, geospatial tools, etc.

See our [TODO.md](TODO.md) for specific tasks and project roadmap.

## License

This project is dual-licensed under:

- [MIT License](LICENSE-MIT)
- [Apache License Version 2.0](LICENSE-APACHE)

You can choose to use either license. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

SciRS2 builds on the shoulders of giants:
- The SciPy and NumPy communities for their pioneering work
- The Rust ecosystem and its contributors
- The numerous mathematical and scientific libraries that inspired this project

## Future Directions

- **Extended Hardware Support**: ARM, RISC-V, mobile, embedded
- **Cloud Deployment**: Container optimization, serverless function support
- **Domain-Specific Extensions**: Finance, bioinformatics, physics
- **Ecosystem Integration**: Python and Julia interoperability
- **Performance Monitoring**: Runtime analyzers, configuration optimizers
- **Automated Architecture Selection**: Hardware-aware algorithm choices

For more detailed information on development status and roadmap, check the [TODO.md](TODO.md) file.