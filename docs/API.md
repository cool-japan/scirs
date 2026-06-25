# SciRS2 v0.3.0 API Documentation

**Comprehensive API Reference for SciRS2 Scientific Computing Library**

Version: 0.3.0 | Updated: February 26, 2026

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start Examples](#quick-start-examples)
3. [Core Modules](#core-modules)
4. [Python Bindings](#python-bindings)
5. [WebAssembly (WASM)](#webassembly-wasm)
6. [SIMD Acceleration](#simd-acceleration)
7. [GPU Acceleration](#gpu-acceleration)
8. [Model Serialization](#model-serialization)
9. [Bayesian Methods](#bayesian-methods)
10. [Advanced Features](#advanced-features)
11. [Performance Tuning](#performance-tuning)

---

## Overview

SciRS2 is a comprehensive scientific computing and AI/ML infrastructure in **Pure Rust**, providing SciPy-compatible APIs with performance improvements of 10-100x through SIMD acceleration, GPU backends, and advanced optimization techniques.

### Key Features (v0.3.0)

- **Pure Rust by Default**: Zero C/C++/Fortran dependencies (OxiBLAS, OxiFFT)
- **SIMD Acceleration**: 3-12x speedups across all operations
- **GPU Support**: CUDA, Metal, OpenCL, WebGPU backends
- **Python Ecosystem**: Full NumPy/SciPy-compatible bindings
- **WASM Support**: Browser and Node.js compatible modules
- **Bayesian Methods**: NUTS sampler, MCMC diagnostics
- **Model Serialization**: SafeTensors format support
- **Memory Management**: Advanced allocators (Buddy, Slab, Compaction, Hybrid)

### Installation

```bash
# Rust
cargo add scirs2

# Python
pip install scirs2

# WASM
npm install scirs2-wasm
```

---

## Quick Start Examples

### Rust: Basic Linear Algebra

```rust
use scirs2::linalg;
use ndarray::arr2;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a matrix
    let a = arr2(&[[4.0, 2.0], [2.0, 3.0]]);

    // Compute determinant
    let det = linalg::det(&a)?;
    println!("Determinant: {}", det);

    // Compute inverse
    let inv = linalg::inv(&a)?;
    println!("Inverse:\n{:?}", inv);

    // QR decomposition
    let (q, r) = linalg::qr(&a)?;
    println!("Q:\n{:?}\nR:\n{:?}", q, r);

    Ok(())
}
```

### Rust: SIMD-Accelerated Operations

```rust
use scirs2_core::simd;
use ndarray::Array1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x = Array1::linspace(0.0, 10.0, 1000);
    let y = Array1::linspace(0.0, 10.0, 1000);

    // SIMD-accelerated element-wise operations (3-12x faster)
    let sum = simd::add_simd(&x, &y)?;
    let prod = simd::mul_simd(&x, &y)?;

    // SIMD reductions
    let total = simd::sum_simd(&x)?;
    let mean = simd::mean_simd(&x)?;

    println!("Sum: {}, Mean: {}", total, mean);

    Ok(())
}
```

### Rust: GPU Acceleration

```rust
use scirs2_core::gpu::{GpuContext, GpuBackend, GpuBuffer};

#[cfg(feature = "gpu")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize GPU context (auto-detects best backend)
    let ctx = GpuContext::new(GpuBackend::default())?;
    println!("Using backend: {}", ctx.backend());

    // Create GPU buffers
    let a = vec![1.0f32, 2.0, 3.0, 4.0];
    let b = vec![5.0f32, 6.0, 7.0, 8.0];

    let buf_a = ctx.create_buffer(&a)?;
    let buf_b = ctx.create_buffer(&b)?;

    // Perform GPU computation
    let result = ctx.add(&buf_a, &buf_b)?;

    // Copy result back to CPU
    let output: Vec<f32> = result.to_vec()?;
    println!("Result: {:?}", output);

    Ok(())
}
```

### Rust: Automatic Differentiation

```rust
use scirs2_autograd::{run, tensor_ops as T};

fn main() {
    // Compute gradients of z = 2x² + 3y + 1
    run(|ctx| {
        let x = ctx.placeholder("x", &[]);
        let y = ctx.placeholder("y", &[]);
        let z = 2.0 * x * x + 3.0 * y + 1.0;

        // First-order gradients
        let dz_dy = &T::grad(&[z], &[y])[0];
        println!("dz/dy = {:?}", dz_dy.eval(ctx));  // => 3.0

        let dz_dx = &T::grad(&[z], &[x])[0];
        let x_val = scirs2_autograd::ndarray::arr0(2.0);
        let result = ctx.evaluator()
            .push(dz_dx)
            .feed(x, x_val.view())
            .run()[0];
        println!("dz/dx at x=2 = {:?}", result);  // => 8.0

        // Higher-order derivatives
        let d2z_dx2 = &T::grad(&[dz_dx], &[x])[0];
        println!("d²z/dx² = {:?}", d2z_dx2.eval(ctx));  // => 4.0
    });
}
```

---

## Core Modules

### scirs2-core: Foundation

Core numerical computing infrastructure with SIMD acceleration and GPU support.

**Features:**
- SIMD operations (AVX2, AVX-512, NEON)
- GPU backends (CUDA, Metal, OpenCL, WebGPU)
- Memory management (Buddy, Slab, Compaction allocators)
- Array protocol and interoperability
- Random number generation

**Example: SIMD Operations**

```rust
use scirs2_core::simd;
use ndarray::Array1;

// Element-wise operations (SIMD accelerated)
let x = Array1::linspace(0.0, 1.0, 1000);
let y = Array1::linspace(0.0, 1.0, 1000);

// Basic arithmetic (3-5x faster)
let sum = simd::add_simd(&x, &y)?;
let diff = simd::sub_simd(&x, &y)?;
let prod = simd::mul_simd(&x, &y)?;
let quot = simd::div_simd(&x, &y)?;

// Transcendental functions (4-10x faster)
let exp_x = simd::exp_simd(&x)?;
let log_x = simd::log_simd(&x)?;
let sin_x = simd::sin_simd(&x)?;
let cos_x = simd::cos_simd(&x)?;

// Reductions (2-6x faster)
let total = simd::sum_simd(&x)?;
let mean = simd::mean_simd(&x)?;
let max_val = simd::max_simd(&x)?;
let min_val = simd::min_simd(&x)?;
```

**Example: GPU Memory Management**

```rust
use scirs2_core::gpu::memory_management::{
    BuddyAllocator, SlabAllocator, HybridAllocator
};

// Hybrid allocator (automatic strategy selection)
let allocator = HybridAllocator::new()?;

// Small allocation (uses slab - O(1))
let small = allocator.allocate(256)?;

// Medium allocation (uses buddy - O(log n))
let medium = allocator.allocate(256 * 1024)?;

// Large allocation (direct - O(1))
let large = allocator.allocate(32 * 1024 * 1024)?;

// Cleanup
allocator.deallocate(small, 256)?;
allocator.deallocate(medium, 256 * 1024)?;
allocator.deallocate(large, 32 * 1024 * 1024)?;
```

### scirs2-linalg: Linear Algebra

Comprehensive linear algebra operations with OxiBLAS backend.

**Features:**
- Matrix operations (multiply, transpose, inverse)
- Decompositions (LU, QR, SVD, Cholesky, Eigenvalue)
- Linear solvers (direct and iterative)
- Specialized matrices (Toeplitz, Hankel, Circulant)
- SIMD-accelerated BLAS operations

**Example: Matrix Decompositions**

```rust
use scirs2::linalg;
use ndarray::arr2;

let a = arr2(&[[4.0, 2.0], [2.0, 3.0]]);

// LU decomposition
let (l, u, p) = linalg::lu(&a)?;

// QR decomposition
let (q, r) = linalg::qr(&a)?;

// SVD decomposition
let (u, s, vt) = linalg::svd(&a)?;

// Cholesky decomposition (for positive definite matrices)
let chol = linalg::cholesky(&a)?;

// Eigenvalue decomposition
let (eigenvalues, eigenvectors) = linalg::eig(&a)?;
```

**Example: Linear System Solving**

```rust
use scirs2::linalg;
use ndarray::{arr2, arr1};

let a = arr2(&[[3.0, 2.0], [1.0, 2.0]]);
let b = arr1(&[5.0, 3.0]);

// Direct solve: Ax = b
let x = linalg::solve(&a, &b)?;
println!("Solution: {:?}", x);

// Least squares: minimize ||Ax - b||²
let x_ls = linalg::lstsq(&a, &b)?;
```

### scirs2-stats: Statistics

Statistical distributions, tests, and descriptive statistics.

**Features:**
- Distributions (Normal, Uniform, Exponential, etc.)
- Descriptive statistics (mean, variance, skewness, kurtosis)
- Hypothesis tests (t-test, chi-square, ANOVA)
- Bayesian methods (NUTS sampler, MCMC diagnostics)

**Example: Descriptive Statistics**

```rust
use scirs2::stats;
use ndarray::Array1;

let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

// Basic statistics
let mean = stats::mean(&data)?;
let median = stats::median(&data)?;
let std = stats::std(&data, 1)?;  // ddof=1
let var = stats::variance(&data, 1)?;

// Higher-order moments (52x faster than SciPy!)
let skewness = stats::skew(&data)?;
let kurtosis = stats::kurtosis(&data)?;

println!("Mean: {}, Std: {}, Skewness: {}", mean, std, skewness);
```

**Example: Statistical Tests**

```rust
use scirs2::stats;
use ndarray::Array1;

let sample1 = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
let sample2 = Array1::from_vec(vec![2.0, 3.0, 4.0, 5.0, 6.0]);

// Two-sample t-test
let (t_stat, p_value) = stats::ttest_ind(&sample1, &sample2)?;
println!("t-statistic: {}, p-value: {}", t_stat, p_value);

// Chi-square test
let observed = Array1::from_vec(vec![16.0, 18.0, 16.0, 14.0, 12.0, 12.0]);
let expected = Array1::from_vec(vec![16.0, 16.0, 16.0, 16.0, 16.0, 8.0]);
let (chi2, p) = stats::chisquare(&observed, &expected)?;
```

### scirs2-fft: Fast Fourier Transform

High-performance FFT operations using OxiFFT (Pure Rust).

**Features:**
- 1D/2D/nD FFT and inverse FFT
- Real FFT (rfft/irfft)
- FFT planning and optimization
- SIMD acceleration

**Example: FFT Operations**

```rust
use scirs2::fft;
use ndarray::Array1;
use num_complex::Complex;

// Real FFT
let signal = Array1::linspace(0.0, 1.0, 512);
let spectrum = fft::rfft(&signal)?;
println!("Spectrum size: {}", spectrum.len());

// Complex FFT
let complex_signal: Array1<Complex<f64>> = signal
    .mapv(|x| Complex::new(x, 0.0));
let fft_result = fft::fft(&complex_signal)?;

// Inverse FFT
let reconstructed = fft::ifft(&fft_result)?;

// 2D FFT
let image = ndarray::Array2::zeros((256, 256));
let fft_2d = fft::fft2(&image)?;
```

### scirs2-signal: Signal Processing

Digital signal processing operations.

**Features:**
- Filtering (FIR, IIR, Butterworth, Chebyshev)
- Convolution and correlation
- Window functions (Hamming, Hann, Blackman)
- Spectral analysis
- Resampling

**Example: Filtering**

```rust
use scirs2::signal;
use ndarray::Array1;

let signal = Array1::linspace(0.0, 1.0, 1000);

// Design Butterworth filter
let (b, a) = signal::butter(4, 0.1, "lowpass")?;

// Apply filter
let filtered = signal::filtfilt(&b, &a, &signal)?;

// Convolution
let kernel = Array1::from_vec(vec![1.0, 2.0, 1.0]) / 4.0;
let convolved = signal::convolve(&signal, &kernel, "same")?;
```

### scirs2-integrate: Numerical Integration

Numerical integration and ODE solvers.

**Features:**
- Quadrature (quad, romberg, simpson)
- ODE solvers (RK45, DOP853, BDF)
- Boundary value problems (BVP)
- DAE solvers

**Example: Integration**

```rust
use scirs2::integrate;

// Integrate f(x) = x² from 0 to 1
let result = integrate::quad(|x| x * x, 0.0, 1.0)?;
println!("Integral: {} (exact: 0.333...)", result.value);

// Romberg integration
let result_romberg = integrate::romberg(|x| x * x, 0.0, 1.0)?;

// Simpson's rule
let x = ndarray::Array1::linspace(0.0, 1.0, 100);
let y = x.mapv(|xi| xi * xi);
let result_simps = integrate::simps(&y, Some(&x))?;
```

**Example: ODE Solving**

```rust
use scirs2::integrate;
use ndarray::Array1;

// Solve dy/dt = -2y, y(0) = 1
fn dydt(_t: f64, y: &Array1<f64>) -> Array1<f64> {
    -2.0 * y
}

let y0 = Array1::from_vec(vec![1.0]);
let t_span = (0.0, 5.0);
let t_eval = Array1::linspace(0.0, 5.0, 100);

let solution = integrate::solve_ivp(
    dydt,
    t_span,
    &y0,
    Some(&t_eval),
    "RK45"
)?;

println!("Solution at t=5: {:?}", solution.y.last());
```

### scirs2-optimize: Optimization

Optimization algorithms for unconstrained and constrained problems.

**Features:**
- Scalar minimization (Brent, Golden section)
- Multivariate minimization (Nelder-Mead, BFGS, L-BFGS-B)
- Root finding (bisect, Newton, secant)
- Curve fitting
- Constrained optimization

**Example: Minimization**

```rust
use scirs2::optimize;
use ndarray::Array1;

// Minimize f(x) = (x - 2)²
let result = optimize::minimize_scalar(
    |x| (x - 2.0).powi(2),
    (0.0, 5.0),
    "brent"
)?;
println!("Minimum at x = {}", result.x);

// Multivariate minimization
fn rosenbrock(x: &Array1<f64>) -> f64 {
    let a = 1.0 - x[0];
    let b = x[1] - x[0] * x[0];
    a * a + 100.0 * b * b
}

let x0 = Array1::from_vec(vec![0.0, 0.0]);
let result = optimize::minimize(rosenbrock, &x0, "BFGS")?;
println!("Minimum: {:?}", result.x);
```

### scirs2-interpolate: Interpolation

Interpolation methods for 1D and multi-dimensional data.

**Features:**
- 1D interpolation (linear, cubic, PCHIP)
- Spline interpolation (CubicSpline, UnivariateSpline)
- 2D interpolation (bilinear, bicubic)
- Radial basis functions

**Example: 1D Interpolation**

```rust
use scirs2::interpolate;
use ndarray::Array1;

let x = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0]);
let y = Array1::from_vec(vec![0.0, 1.0, 4.0, 9.0]);

// Cubic spline interpolation
let spline = interpolate::CubicSpline::new(&x, &y)?;
let x_new = Array1::linspace(0.0, 3.0, 100);
let y_new = spline.evaluate(&x_new)?;

// PCHIP (Piecewise Cubic Hermite Interpolating Polynomial)
let pchip = interpolate::PchipInterpolator::new(&x, &y)?;
let y_pchip = pchip.evaluate(&x_new)?;
```

### scirs2-sparse: Sparse Matrices

Sparse matrix operations and solvers.

**Features:**
- Sparse formats (CSR, CSC, COO, DOK)
- Sparse linear algebra
- Iterative solvers (CG, BiCGSTAB, GMRES)
- Graph algorithms

**Example: Sparse Matrices**

```rust
use scirs2::sparse;
use ndarray::Array1;

// Create sparse matrix (CSR format)
let row_indices = vec![0, 0, 1, 2, 2];
let col_indices = vec![0, 2, 1, 0, 2];
let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

let sparse_matrix = sparse::CsrMatrix::new(
    (3, 3),
    row_indices,
    col_indices,
    data
)?;

// Sparse matrix-vector multiplication
let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);
let y = sparse_matrix.dot(&x)?;

// Solve sparse linear system using CG
let b = Array1::from_vec(vec![1.0, 2.0, 3.0]);
let x_solution = sparse::linalg::cg(&sparse_matrix, &b)?;
```

### scirs2-cluster: Clustering

Clustering algorithms with SIMD acceleration.

**Features:**
- K-means (2-8x faster with SIMD)
- Hierarchical clustering (SIMD distance computation)
- DBSCAN (2-6x faster with SIMD neighborhood queries)
- Gaussian Mixture Models (GMM with SIMD E-step)
- Spectral clustering

**Example: K-Means Clustering**

```rust
use scirs2::cluster;
use ndarray::Array2;

// Generate sample data
let data = Array2::from_shape_fn((1000, 2), |(i, j)| {
    (i as f64 + j as f64) / 10.0
});

// K-means clustering (SIMD accelerated)
let kmeans = cluster::KMeans::new(3)
    .max_iter(100)
    .tolerance(1e-4);

let result = kmeans.fit(&data)?;
println!("Cluster centers:\n{:?}", result.centers);
println!("Labels: {:?}", result.labels);
println!("Inertia: {}", result.inertia);
```

**Example: DBSCAN**

```rust
use scirs2::cluster;
use ndarray::Array2;

let data = Array2::from_shape_fn((500, 2), |(i, j)| {
    (i as f64, j as f64)
});

// DBSCAN clustering (SIMD accelerated)
let dbscan = cluster::DBSCAN::new(0.5, 5);
let labels = dbscan.fit_predict(&data)?;

println!("Number of clusters: {}",
    labels.iter().filter(|&&x| x >= 0).max().unwrap_or(&0) + 1);
```

### scirs2-series: Time Series Analysis

Time series analysis and forecasting with SIMD acceleration.

**Features:**
- ARIMA models (2-6x faster with SIMD)
- SARIMA (seasonal ARIMA)
- Autocorrelation (ACF/PACF)
- STL decomposition (SIMD accelerated)
- Exponential smoothing

**Example: ARIMA Model**

```rust
use scirs2::series;
use ndarray::Array1;

// Time series data
let data = Array1::from_vec(vec![
    10.0, 12.0, 13.0, 15.0, 18.0, 20.0, 22.0, 25.0
]);

// Fit ARIMA(1,1,1) model (SIMD accelerated)
let arima = series::ARIMA::new(1, 1, 1)?;
let fitted = arima.fit(&data)?;

// Forecast
let forecast = fitted.forecast(10)?;
println!("Forecast: {:?}", forecast);

// Get model parameters
println!("AR coefficients: {:?}", fitted.ar_params());
println!("MA coefficients: {:?}", fitted.ma_params());
```

### scirs2-autograd: Automatic Differentiation

Automatic differentiation with operation fusion and memory optimization.

**Features:**
- Reverse-mode AD with dynamic graphs
- Higher-order derivatives
- Operation fusion (1.3-2.0x speedup)
- Memory pooling for tensors
- SIMD-accelerated operations

**Example: Neural Network Training**

```rust
use scirs2_autograd::{tensor_ops as T, VariableEnvironment};
use scirs2_autograd::optimizers::adam::Adam;

fn main() {
    let mut env = VariableEnvironment::new();

    // Define model parameters
    let w1 = env.slot();
    let w2 = env.slot();

    // Initialize with random values
    env.set(w1, ndarray::Array2::zeros((784, 128)));
    env.set(w2, ndarray::Array2::zeros((128, 10)));

    // Training loop
    let mut optimizer = Adam::default();

    for epoch in 0..10 {
        // Forward pass
        let x = T::placeholder(&[None, 784]);
        let y_true = T::placeholder(&[None, 10]);

        let h = T::matmul(x, w1);
        let h_relu = T::relu(h);
        let logits = T::matmul(h_relu, w2);

        // Compute loss
        let loss = T::sparse_softmax_cross_entropy(logits, y_true);

        // Backward pass
        let grads = T::grad(&[loss], &[w1, w2]);

        // Update parameters
        optimizer.update(&mut env, &[w1, w2], &grads)?;

        println!("Epoch {}: loss = {:?}", epoch, loss.eval());
    }
}
```

### scirs2-neural: Neural Networks

Neural network layers and architectures with model serialization.

**Features:**
- Layers (Dense, Conv2D, LSTM, Transformer)
- Activations (ReLU, GELU, Swish, Mish)
- Optimizers (SGD, Adam, AdamW)
- Model serialization (SafeTensors format)
- Pre-trained architectures (ResNet, BERT, GPT, ViT)

**Example: Build and Train Neural Network**

```rust
use scirs2_neural::{Sequential, layers::*, optimizers::Adam};
use ndarray::Array2;

// Build model
let mut model = Sequential::new()
    .add(Dense::new(784, 128)?)
    .add(ReLU::new())
    .add(Dense::new(128, 10)?)
    .add(Softmax::new());

// Training data
let x_train = Array2::zeros((1000, 784));
let y_train = Array2::zeros((1000, 10));

// Train
let optimizer = Adam::default();
for epoch in 0..10 {
    let loss = model.fit(&x_train, &y_train, &optimizer)?;
    println!("Epoch {}: loss = {}", epoch, loss);
}

// Predict
let x_test = Array2::zeros((100, 784));
let predictions = model.predict(&x_test)?;
```

### scirs2-graph: Graph Algorithms

Graph data structures and algorithms with SIMD acceleration.

**Features:**
- Graph construction and manipulation
- Shortest paths (Dijkstra, Bellman-Ford)
- PageRank (2-5x faster with SIMD)
- Community detection
- Spectral methods (SIMD accelerated)

**Example: Graph Analysis**

```rust
use scirs2::graph;

// Create graph
let mut g = graph::Graph::new();
g.add_edge(0, 1, 1.0)?;
g.add_edge(1, 2, 2.0)?;
g.add_edge(2, 3, 1.0)?;
g.add_edge(3, 0, 3.0)?;

// Compute PageRank (SIMD accelerated)
let pagerank = graph::pagerank(&g, 0.85, 100)?;
println!("PageRank: {:?}", pagerank);

// Shortest path
let path = graph::shortest_path(&g, 0, 3)?;
println!("Shortest path: {:?}", path);

// Connected components
let components = graph::connected_components(&g)?;
```

### scirs2-special: Special Functions

Mathematical special functions with SIMD acceleration.

**Features:**
- Bessel functions (J0, J1, Y0, Y1, I0, I1, K0, K1)
- Gamma functions
- Error functions (erf, erfc)
- Beta functions
- Elliptic integrals

**Example: Special Functions**

```rust
use scirs2::special;
use ndarray::Array1;

let x = Array1::linspace(0.0, 10.0, 100);

// Bessel functions (SIMD accelerated)
let j0 = special::j0_simd(&x)?;
let j1 = special::j1_simd(&x)?;

// Gamma function
let gamma_vals = x.mapv(|xi| special::gamma(xi));

// Error function
let erf_vals = special::erf_simd(&x)?;

println!("J0(5.0) = {}", special::j0(5.0)?);
println!("Gamma(5) = {}", special::gamma(5.0));
```

### scirs2-spatial: Spatial Algorithms

Spatial data structures and algorithms.

**Features:**
- KD-trees
- Distance computations (SIMD accelerated)
- Nearest neighbor search
- Delaunay triangulation
- Voronoi diagrams

**Example: KD-Tree**

```rust
use scirs2::spatial;
use ndarray::Array2;

// Build KD-tree
let data = Array2::from_shape_fn((1000, 3), |(i, j)| {
    (i as f64 + j as f64) / 10.0
});

let kdtree = spatial::KDTree::new(&data)?;

// Query nearest neighbors
let query_point = ndarray::Array1::from_vec(vec![5.0, 5.0, 5.0]);
let (distances, indices) = kdtree.query(&query_point, 10)?;

println!("Nearest neighbors: {:?}", indices);
println!("Distances: {:?}", distances);
```

---

## Python Bindings

Full Python API with NumPy/SciPy compatibility.

### Installation

```bash
pip install scirs2
```

### Linear Algebra

```python
import numpy as np
import scirs2

# Matrix operations
A = np.array([[4.0, 2.0], [2.0, 3.0]])

# Determinant
det = scirs2.det_py(A)
print(f"Determinant: {det}")

# Inverse
inv = scirs2.inv_py(A)

# Decompositions
lu_result = scirs2.lu_py(A)  # Returns {'L', 'U', 'P'}
qr_result = scirs2.qr_py(A)  # Returns {'Q', 'R'}
svd_result = scirs2.svd_py(A)  # Returns {'U', 'S', 'Vt'}

# Solve linear system
b = np.array([1.0, 2.0])
x = scirs2.solve_py(A, b)
```

### Statistics (Up to 410x Faster!)

```python
import numpy as np
import scirs2

data = np.random.randn(1000)

# Basic statistics (8-14x faster than NumPy)
mean = scirs2.mean_py(data)
std = scirs2.std_py(data, 0)  # ddof=0

# Higher-order moments (52x faster than SciPy!)
skewness = scirs2.skew_py(data)
kurtosis = scirs2.kurtosis_py(data)

print(f"Mean: {mean:.4f}, Std: {std:.4f}")
print(f"Skewness: {skewness:.4f}, Kurtosis: {kurtosis:.4f}")
```

### FFT (3x Faster on Small Data)

```python
import numpy as np
import scirs2

# Real FFT
signal = np.random.randn(512)
spectrum = scirs2.rfft_py(signal)

# Inverse FFT
reconstructed = scirs2.irfft_py(spectrum)

# 2D FFT
image = np.random.randn(256, 256)
fft_2d = scirs2.fft2_py(image)
```

### Signal Processing

```python
import numpy as np
import scirs2

# Design Butterworth filter
b, a = scirs2.butter_py(4, 0.1, 'lowpass')

# Filter signal
signal = np.random.randn(1000)
filtered = scirs2.filtfilt_py(b, a, signal)

# Convolution
kernel = np.array([1, 2, 1]) / 4.0
convolved = scirs2.convolve_py(signal, kernel, mode='same')
```

### Integration

```python
import numpy as np
import scirs2

# Integrate function
def f(x):
    return x**2

result = scirs2.quad_py(f, 0.0, 1.0)
print(f"Integral: {result['value']}")

# Simpson's rule
x = np.linspace(0, 1, 100)
y = x**2
integral = scirs2.simps_py(y, x)
```

### Optimization

```python
import numpy as np
import scirs2

# Minimize scalar function
def f(x):
    return (x - 2)**2

result = scirs2.minimize_scalar_py(f, bounds=(0, 5), method='brent')
print(f"Minimum at x = {result['x']}")

# Multivariate minimization
def rosenbrock(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

x0 = np.array([0.0, 0.0])
result = scirs2.minimize_py(rosenbrock, x0, method='BFGS')
```

### Clustering

```python
import numpy as np
import scirs2

# Generate data
data = np.random.randn(1000, 2)

# K-means clustering (SIMD accelerated)
centers, labels, inertia = scirs2.kmeans_py(data, n_clusters=3)

# DBSCAN
labels = scirs2.dbscan_py(data, eps=0.5, min_samples=5)

# Hierarchical clustering
linkage = scirs2.linkage_py(data, method='ward')
```

### Time Series

```python
import numpy as np
import scirs2

# Time series data
data = np.array([10, 12, 13, 15, 18, 20, 22, 25])

# ARIMA model (SIMD accelerated)
model = scirs2.ARIMA_py(data, order=(1, 1, 1))
forecast = model.forecast(steps=10)

print(f"Forecast: {forecast}")
```

---

## WebAssembly (WASM)

Browser and Node.js compatible scientific computing.

### Installation

```bash
npm install scirs2-wasm
```

### Browser Usage

```javascript
import init, * as scirs2 from 'scirs2-wasm';

async function main() {
  // Initialize WASM module
  await init();

  // Create arrays
  const a = new scirs2.WasmArray([1, 2, 3, 4]);
  const b = new scirs2.WasmArray([5, 6, 7, 8]);

  // Operations
  const sum = scirs2.add(a, b);
  const mean_a = scirs2.mean(a);
  const std_a = scirs2.std(a);

  console.log('Sum:', sum.to_array());
  console.log('Mean:', mean_a);
  console.log('Std:', std_a);

  // FFT
  const signal = new scirs2.WasmArray(Array.from({length: 512}, () => Math.random()));
  const spectrum = scirs2.rfft(signal);

  // Signal processing
  const filtered = scirs2.convolve(signal, kernel, 'same');
}

main();
```

### Node.js Usage

```javascript
const scirs2 = require('scirs2-wasm');

async function main() {
  // Create matrix
  const matrix = scirs2.WasmArray.from_shape(
    [2, 2],
    [1, 2, 3, 4]
  );

  // Linear algebra
  const det = scirs2.det(matrix);
  const trace = scirs2.trace(matrix);

  console.log('Determinant:', det);
  console.log('Trace:', trace);

  // Integration
  const result = scirs2.quad(x => x * x, 0, 1);
  console.log('Integral:', result.value);
}

main();
```

### TypeScript Support

```typescript
import init, * as scirs2 from 'scirs2-wasm';

async function main(): Promise<void> {
  await init();

  // Type-safe array operations
  const arr: scirs2.WasmArray = scirs2.WasmArray.linspace(0, 10, 100);

  const mean: number = scirs2.mean(arr);
  const std: number = scirs2.std(arr);

  console.log(`Mean: ${mean}, Std: ${std}`);

  // Optimization
  const result = scirs2.minimize_scalar(
    (x: number) => (x - 2) ** 2,
    { bounds: [0, 5], method: 'brent' }
  );

  console.log(`Minimum at x = ${result.x}`);
}
```

### Available WASM Modules

**FFT Operations:**
- `fft()`, `ifft()`, `rfft()`, `irfft()`
- `fft2()`, `ifft2()`

**Signal Processing:**
- `convolve()`, `correlate()`
- `filter()`, `resample()`
- Window functions: `hamming()`, `hann()`, `blackman()`

**Integration:**
- `quad()`, `romberg()`, `simps()`
- ODE solvers: `odeint()`, `solve_ivp()`

**Optimization:**
- `minimize_scalar()`: Brent, Golden section
- `brent()`, `golden()`, `newton()`, `bisect()`

**Interpolation:**
- `interp1d()`: Linear, cubic, nearest
- `CubicSpline()`, `PchipInterpolator()`

---

## SIMD Acceleration

SciRS2 provides comprehensive SIMD acceleration across all operations.

### Performance Improvements

| Operation | Speedup | Architecture |
|-----------|---------|--------------|
| Element-wise operations | 3-5x | AVX2/NEON |
| Transcendental functions | 4-10x | AVX2/AVX-512 |
| Reductions (sum, mean) | 2-6x | AVX2/NEON |
| Matrix operations | 2-4x | AVX2/AVX-512 |
| Distance computations | 5-8x | AVX2/NEON |
| Clustering (K-means) | 2-8x | AVX2/AVX-512 |
| Time series (ARIMA) | 2-6x | AVX2/NEON |
| Graph (PageRank) | 2-5x | AVX2/AVX-512 |
| Special functions | 4-10x | AVX2/AVX-512 |

### Enabling SIMD

SIMD acceleration is **enabled by default** and auto-detects CPU capabilities.

```rust
// SIMD is automatically used when available
use scirs2_core::simd;

let x = ndarray::Array1::linspace(0.0, 10.0, 1000);
let y = ndarray::Array1::linspace(0.0, 10.0, 1000);

// Automatically uses SIMD if available
let sum = simd::add_simd(&x, &y)?;
```

### Manual SIMD Control

```rust
use scirs2_core::simd::detect;

// Check SIMD capabilities
if detect::has_avx2() {
    println!("AVX2 available");
}

if detect::has_avx512f() {
    println!("AVX-512 available");
}

if detect::has_neon() {
    println!("ARM NEON available");
}

// Force scalar fallback (for testing)
std::env::set_var("SCIRS2_DISABLE_SIMD", "1");
```

### SIMD Operations Reference

**Element-wise Operations:**
```rust
use scirs2_core::simd;

// Arithmetic
let sum = simd::add_simd(&x, &y)?;
let diff = simd::sub_simd(&x, &y)?;
let prod = simd::mul_simd(&x, &y)?;
let quot = simd::div_simd(&x, &y)?;

// Transcendental
let exp_x = simd::exp_simd(&x)?;
let log_x = simd::log_simd(&x)?;
let sin_x = simd::sin_simd(&x)?;
let cos_x = simd::cos_simd(&x)?;
let sqrt_x = simd::sqrt_simd(&x)?;
```

**Reductions:**
```rust
let total = simd::sum_simd(&x)?;
let mean = simd::mean_simd(&x)?;
let max_val = simd::max_simd(&x)?;
let min_val = simd::min_simd(&x)?;
let variance = simd::variance_simd(&x)?;
```

**Distance Computations:**
```rust
let euclidean = simd::euclidean_distance_simd(&x, &y)?;
let manhattan = simd::manhattan_distance_simd(&x, &y)?;
let cosine_sim = simd::cosine_similarity_simd(&x, &y)?;
```

---

## GPU Acceleration

Multi-backend GPU support for high-performance computing.

### Supported Backends

| Backend | Platform | Features |
|---------|----------|----------|
| CUDA | NVIDIA GPUs | Full support, tensor cores |
| Metal | Apple Silicon | MPS integration, unified memory |
| OpenCL | Cross-platform | Broad device support |
| WebGPU | Web/Cross-platform | Portable compute shaders |
| CPU | Fallback | Optimized CPU implementations |

### Basic GPU Usage

```rust
use scirs2_core::gpu::{GpuContext, GpuBackend};

#[cfg(feature = "gpu")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Auto-detect best backend
    let ctx = GpuContext::new(GpuBackend::default())?;
    println!("Using: {}", ctx.backend());

    // Create buffers
    let data = vec![1.0f32; 1_000_000];
    let buffer = ctx.create_buffer(&data)?;

    // GPU operations
    let result = ctx.square(&buffer)?;
    let sum = ctx.sum(&result)?;

    println!("Sum: {}", sum);

    Ok(())
}
```

### Metal Backend (Apple Silicon)

```rust
use scirs2_core::gpu::{GpuContext, GpuBackend};

#[cfg(all(target_os = "macos", feature = "metal"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = GpuContext::new(GpuBackend::Metal)?;

    // Metal Performance Shaders (MPS) integration
    // 2-5x speedup on M1/M2/M3 chips

    let a = vec![1.0f32; 1024 * 1024];
    let b = vec![2.0f32; 1024 * 1024];

    let buf_a = ctx.create_buffer(&a)?;
    let buf_b = ctx.create_buffer(&b)?;

    // Uses Metal Shading Language (MSL) kernels
    let result = ctx.add(&buf_a, &buf_b)?;

    // Unified memory (zero-copy when possible)
    let output = result.to_vec_zerocopy()?;

    Ok(())
}
```

### CUDA Backend

```rust
use scirs2_core::gpu::{GpuContext, GpuBackend};

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = GpuContext::new(GpuBackend::Cuda)?;

    // Check device properties
    let props = ctx.device_properties()?;
    println!("GPU: {}", props.name);
    println!("Compute capability: {}.{}",
        props.major, props.minor);

    // Matrix multiplication with tensor cores
    let a = vec![1.0f32; 4096 * 4096];
    let b = vec![2.0f32; 4096 * 4096];

    let buf_a = ctx.create_buffer(&a)?;
    let buf_b = ctx.create_buffer(&b)?;

    // Uses cuBLAS with tensor core acceleration
    let result = ctx.matmul(&buf_a, &buf_b, (4096, 4096, 4096))?;

    Ok(())
}
```

### Async Execution

```rust
use scirs2_core::gpu::async_execution::GpuTaskQueue;
use tokio;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let queue = GpuTaskQueue::new()?;

    // Submit tasks asynchronously
    let task1 = queue.submit_async(|ctx| {
        // GPU operation 1
        ctx.matmul(&a, &b, dims)
    })?;

    let task2 = queue.submit_async(|ctx| {
        // GPU operation 2
        ctx.convolution(&input, &kernel)
    })?;

    // Wait for results
    let (result1, result2) = tokio::join!(task1, task2);

    Ok(())
}
```

### Memory Management

```rust
use scirs2_core::gpu::memory_management::*;

// Hybrid allocator (automatic strategy selection)
let allocator = HybridAllocator::new()?;

// Small allocation (uses slab - O(1))
let small = allocator.allocate(256)?;

// Medium allocation (uses buddy - O(log n))
let medium = allocator.allocate(256 * 1024)?;

// Large allocation (direct - O(1))
let large = allocator.allocate(32 * 1024 * 1024)?;

// Cleanup
allocator.deallocate(small, 256)?;
allocator.deallocate(medium, 256 * 1024)?;
allocator.deallocate(large, 32 * 1024 * 1024)?;
```

**Allocator Strategies:**

1. **BuddyAllocator**: Binary buddy system for power-of-2 allocations
   - O(log n) allocation/deallocation
   - Low fragmentation
   - Best for medium allocations (64KB - 16MB)

2. **SlabAllocator**: Fixed-size block allocator
   - O(1) allocation/deallocation
   - Zero internal fragmentation
   - Best for small allocations (< 64KB)

3. **CompactionAllocator**: Defragmentation support
   - Relocates buffers during idle periods
   - Near-zero fragmentation with periodic compaction
   - Best for long-running applications

4. **HybridAllocator**: Combined strategy
   - Automatically selects best allocator based on size
   - Optimal performance across all allocation sizes

---

## Model Serialization

SafeTensors format support for neural network models.

### SafeTensors Format

SafeTensors is a safe, fast serialization format compatible with HuggingFace and PyTorch.

**Features:**
- Safe: No arbitrary code execution
- Fast: Memory-mapped loading
- Cross-framework: Works with PyTorch, TensorFlow, JAX
- Metadata: Preserves model information

### Save Model

```rust
use scirs2_neural::serialization::{SafeTensorsWriter, NamedParameters};
use std::collections::HashMap;
use ndarray::Array2;

// Create model parameters
let mut params = HashMap::new();
params.insert("layer1.weight".to_string(),
    Array2::zeros((128, 784)).into_dyn());
params.insert("layer1.bias".to_string(),
    Array1::zeros(128).into_dyn());
params.insert("layer2.weight".to_string(),
    Array2::zeros((10, 128)).into_dyn());
params.insert("layer2.bias".to_string(),
    Array1::zeros(10).into_dyn());

// Create writer
let mut writer = SafeTensorsWriter::new();

// Add metadata
writer.add_metadata("model_type", "mlp");
writer.add_metadata("num_layers", "2");
writer.add_metadata("version", "0.3.0");

// Add tensors
for (name, tensor) in params {
    writer.add_tensor::<f32>(name, &tensor)?;
}

// Write to file
writer.write_to_file("model.safetensors")?;
```

### Load Model

```rust
use scirs2_neural::serialization::SafeTensorsReader;

// Load from file
let reader = SafeTensorsReader::from_file("model.safetensors")?;

// Read metadata
let model_type = reader.metadata().get("model_type");
println!("Model type: {:?}", model_type);

// Load specific tensor
let weights = reader.load_tensor::<f32>("layer1.weight")?;
println!("Layer 1 weights shape: {:?}", weights.shape());

// Load all tensors
let all_tensors = reader.load_all::<f32>()?;
for (name, tensor) in all_tensors {
    println!("{}: {:?}", name, tensor.shape());
}
```

### Serialize Pre-trained Models

```rust
use scirs2_neural::serialization::architecture::*;

// Serialize ResNet-50
let resnet = ResNetSerializer::new(ResNetVariant::ResNet50)?;
resnet.save("resnet50.safetensors")?;

// Serialize BERT
let bert = BertSerializer::new(BertConfig {
    hidden_size: 768,
    num_hidden_layers: 12,
    num_attention_heads: 12,
    intermediate_size: 3072,
    vocab_size: 30522,
})?;
bert.save("bert_base.safetensors")?;

// Serialize GPT-2
let gpt = GptSerializer::new(GptConfig {
    n_layer: 12,
    n_head: 12,
    n_embd: 768,
    vocab_size: 50257,
})?;
gpt.save("gpt2.safetensors")?;
```

### Convert from PyTorch

```rust
use scirs2_io::ml_framework::converters::safetensors::*;

// Convert PyTorch checkpoint
let converter = PyTorchConverter::new();
converter.convert_to_safetensors(
    "pytorch_model.bin",
    "scirs2_model.safetensors"
)?;

// Load in SciRS2
let reader = SafeTensorsReader::from_file("scirs2_model.safetensors")?;
let weights = reader.load_all::<f32>()?;
```

---

## Bayesian Methods

Advanced Bayesian inference with NUTS sampler and MCMC diagnostics.

### NUTS Sampler

No-U-Turn Sampler (NUTS) is an advanced Hamiltonian Monte Carlo method.

**Features:**
- Automatic step size adaptation
- Dynamic tree building
- Dual averaging for tuning
- Comprehensive diagnostics

```rust
use scirs2_stats::bayesian::nuts::*;
use ndarray::{Array1, Array2};

// Define log probability function
fn log_prob(theta: &Array1<f64>) -> f64 {
    // Example: Normal distribution log-probability
    let mu = 0.0;
    let sigma = 1.0;
    let sum: f64 = theta.iter()
        .map(|&x| -0.5 * ((x - mu) / sigma).powi(2))
        .sum();
    sum - theta.len() as f64 * (sigma * (2.0 * std::f64::consts::PI).sqrt()).ln()
}

// Define gradient
fn grad_log_prob(theta: &Array1<f64>) -> Array1<f64> {
    // Gradient of log probability
    -theta.clone()
}

// Initialize NUTS sampler
let mut nuts = NutsSampler::new(
    log_prob,
    grad_log_prob,
    2  // number of parameters
)?;

// Sample
let n_samples = 1000;
let n_warmup = 500;

let samples = nuts.sample(n_samples, n_warmup)?;

// Get diagnostics
let diagnostics = nuts.diagnostics();
println!("Acceptance rate: {:.3}", diagnostics.acceptance_rate);
println!("Step size: {:.6}", diagnostics.step_size);
println!("Tree depth: {:.2}", diagnostics.mean_tree_depth);
```

### MCMC Diagnostics

```rust
use scirs2_stats::bayesian::diagnostics::*;
use ndarray::Array2;

// Samples from multiple chains
let chain1 = Array2::zeros((1000, 2));
let chain2 = Array2::zeros((1000, 2));
let chain3 = Array2::zeros((1000, 2));

let chains = vec![chain1, chain2, chain3];

// Gelman-Rubin diagnostic (R-hat)
let rhat = gelman_rubin(&chains)?;
println!("R-hat: {:?}", rhat);
// R-hat < 1.1 indicates convergence

// Effective sample size
let ess = effective_sample_size(&chains)?;
println!("ESS: {:?}", ess);

// Geweke diagnostic
let geweke = geweke_diagnostic(&chains[0])?;
println!("Geweke z-score: {:?}", geweke);
// |z| < 2 indicates convergence
```

### Bayesian Inference Example

```rust
use scirs2_stats::bayesian::*;
use ndarray::Array1;

// Observed data
let data = Array1::from_vec(vec![
    2.1, 1.9, 2.3, 2.0, 1.8, 2.2, 2.1
]);

// Bayesian linear regression
fn log_posterior(params: &Array1<f64>, data: &Array1<f64>) -> f64 {
    let intercept = params[0];
    let slope = params[1];
    let sigma = params[2].exp();  // log-scale

    // Likelihood
    let x = Array1::linspace(0.0, 6.0, data.len());
    let predictions = intercept + slope * &x;
    let residuals = data - &predictions;
    let log_likelihood = -0.5 * (residuals.mapv(|r| r * r).sum() / (sigma * sigma))
        - data.len() as f64 * sigma.ln();

    // Prior (weak)
    let log_prior = -0.5 * (params[0].powi(2) + params[1].powi(2));

    log_likelihood + log_prior
}

// Sample from posterior
let mut sampler = NutsSampler::new(
    |p| log_posterior(p, &data),
    |p| grad_log_posterior(p, &data),
    3
)?;

let posterior_samples = sampler.sample(2000, 1000)?;

// Analyze posterior
let mean_params = posterior_samples.mean_axis(ndarray::Axis(0))?;
println!("Posterior mean: {:?}", mean_params);

let std_params = posterior_samples.std_axis(ndarray::Axis(0), 0.0);
println!("Posterior std: {:?}", std_params);
```

---

## Advanced Features

### Operation Fusion

Automatic fusion of operations to reduce memory bandwidth.

```rust
use scirs2_autograd::{tensor_ops as T, run};

run(|ctx| {
    let x = ctx.placeholder("x", &[None, 784]);

    // These operations are automatically fused:
    // relu(x + bias) => fused_relu_add(x, bias)
    let bias = ctx.constant(Array1::zeros(784));
    let h = T::relu(T::add(x, bias));

    // 1.3-2.0x speedup from fusion
    // Eliminates intermediate tensor allocations

    let result = h.eval(ctx);
});
```

### Memory Pooling

Tensor memory pool for reduced allocations.

```rust
use scirs2_autograd::memory::TensorPool;

// Create pool
let pool = TensorPool::new(1024 * 1024 * 1024)?;  // 1GB

// Allocate tensors from pool
let tensor1 = pool.allocate(&[1024, 1024])?;
let tensor2 = pool.allocate(&[512, 512])?;

// Tensors are automatically returned to pool when dropped
drop(tensor1);

// Reuse memory
let tensor3 = pool.allocate(&[1024, 1024])?;  // Reuses tensor1's memory
```

### GPU Async Execution

Non-blocking GPU operations.

```rust
use scirs2_core::gpu::async_execution::*;
use tokio;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let queue = GpuTaskQueue::new()?;

    // Submit multiple tasks
    let futures: Vec<_> = (0..10).map(|i| {
        queue.submit_async(move |ctx| {
            // GPU computation
            ctx.compute_task(i)
        })
    }).collect();

    // Wait for all
    let results = futures::future::join_all(futures).await;

    Ok(())
}
```

### Heterogeneous Computing

Automatic workload distribution across CPU and GPU.

```rust
use scirs2_core::gpu::heterogeneous::*;

let hetero = HeterogeneousExecutor::new()?;

// Automatically splits work between CPU and GPU
let result = hetero.execute_balanced(|device| {
    match device {
        Device::Cpu => {
            // CPU computation
            cpu_intensive_task()
        }
        Device::Gpu => {
            // GPU computation
            gpu_intensive_task()
        }
    }
})?;
```

---

## Performance Tuning

### Benchmark Your Code

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use scirs2::linalg;
use ndarray::Array2;

fn benchmark_matrix_multiply(c: &mut Criterion) {
    let n = 1000;
    let a = Array2::<f64>::zeros((n, n));
    let b = Array2::<f64>::zeros((n, n));

    c.bench_function("matmul 1000x1000", |bencher| {
        bencher.iter(|| {
            let result = linalg::matmul(black_box(&a), black_box(&b));
            black_box(result)
        });
    });
}

criterion_group!(benches, benchmark_matrix_multiply);
criterion_main!(benches);
```

### Enable Optimizations

**Cargo.toml:**
```toml
[profile.release]
opt-level = 3
lto = "thin"
codegen-units = 1

[dependencies]
scirs2 = { version = "0.3.0", features = ["simd", "gpu", "oxifft"] }
```

### SIMD Optimization Tips

1. **Align data**: Use aligned allocations for best SIMD performance
2. **Batch operations**: Process multiple elements together
3. **Minimize branching**: Branches can break SIMD pipelines
4. **Use appropriate batch sizes**: See scirs2-core SIMD constants

```rust
use scirs2_core::simd::constants::*;

// Optimal batch sizes for different operations
const OPTIMAL_BATCH_EUCLIDEAN: usize = 64;  // Euclidean distance
const OPTIMAL_BATCH_DOT: usize = 128;       // Dot product
const OPTIMAL_BATCH_GEMM_M: usize = 64;     // Matrix multiply (M dimension)
```

### GPU Optimization Tips

1. **Minimize data transfers**: Keep data on GPU when possible
2. **Use async execution**: Overlap computation and transfers
3. **Batch operations**: Amortize kernel launch overhead
4. **Use unified memory**: Eliminate copies on supported platforms
5. **Profile with tools**: Use nvprof (CUDA) or Instruments (Metal)

```rust
// Good: Minimize transfers
let result = ctx.chain()
    .add(&a, &b)
    .multiply(&c)
    .sum()
    .execute()?;

// Bad: Multiple transfers
let temp1 = ctx.add(&a, &b)?.to_cpu()?;
let temp2 = ctx.multiply(&temp1.to_gpu()?, &c)?.to_cpu()?;
let result = ctx.sum(&temp2.to_gpu()?)?;
```

### Memory Optimization Tips

1. **Use memory pools**: Reduce allocation overhead
2. **Reuse buffers**: Avoid repeated allocations
3. **Choose appropriate allocator**: Match workload characteristics
4. **Monitor fragmentation**: Run compaction when needed

```rust
// Use hybrid allocator for mixed workloads
let allocator = HybridAllocator::new()?;

// Monitor fragmentation
let stats = allocator.statistics();
if stats.fragmentation_ratio > 0.3 {
    allocator.compact()?;
}
```

---

## Additional Resources

### Documentation

- **API Reference**: `cargo doc --open`
- **Python Docs**: `python -c "import scirs2; help(scirs2)"`
- **Examples**: `<repo-root>/*/examples/`
- **Benchmarks**: `<repo-root>/benches/`

### Community

- **Repository**: https://github.com/cool-japan/scirs
- **Issues**: https://github.com/cool-japan/scirs/issues
- **Discussions**: https://github.com/cool-japan/scirs/discussions

### Related Projects

- **OxiBLAS**: https://github.com/cool-japan/oxiblas
- **OxiFFT**: https://github.com/cool-japan/oxifft
- **SplitRS**: https://github.com/cool-japan/splitrs

---

## License

SciRS2 is licensed under Apache-2.0.

Copyright (c) 2026 COOLJAPAN OU (Team KitaSan)

---

*Generated on February 26, 2026 for SciRS2 v0.3.0*
