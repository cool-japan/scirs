# SciRS2: Scientific Computing in Rust — All-in-One Meta-Crate

[![crates.io](https://img.shields.io/crates/v/scirs2.svg)](https://crates.io/crates/scirs2)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](../LICENSE)
[![Documentation](https://img.shields.io/docsrs/scirs2)](https://docs.rs/scirs2)
[![Version](https://img.shields.io/badge/version-0.6.2-green.svg)]()

`scirs2` is the **all-in-one convenience meta-crate** for the SciRS2 scientific computing ecosystem. It re-exports the complete set of SciRS2 sub-crates through a unified interface, so you can depend on a single crate and enable only the domains you need via Cargo feature flags.

If you prefer minimal compile times and finer dependency control, use the individual sub-crates directly (e.g., `scirs2-linalg`, `scirs2-stats`). If you want the full ecosystem available in one dependency, use this crate.

## Installation

Add the meta-crate to your `Cargo.toml`:

```toml
[dependencies]
scirs2 = "0.6.2"
```

With only the sub-crates you need (recommended for production):

```toml
[dependencies]
scirs2 = { version = "0.6.2", default-features = false, features = ["linalg", "stats", "optimize"] }
```

For the complete ecosystem:

```toml
[dependencies]
scirs2 = { version = "0.6.2", features = ["full"] }
```

Or depend on individual sub-crates directly for fastest compile times:

```toml
[dependencies]
scirs2-core     = "0.6.2"
scirs2-linalg   = "0.6.2"
scirs2-stats    = "0.6.2"
```

## Feature Flags

Each sub-crate is gated behind a feature flag of the same name. `default` enables the `standard` feature group.

### Feature Groups

| Feature       | Includes                                                                                    |
|---------------|---------------------------------------------------------------------------------------------|
| `standard`    | `linalg`, `stats`, `integrate`, `interpolate`, `optimize`, `fft`, `special`, `signal`, `sparse`, `spatial`, `cluster`, `transform`, `metrics` |
| `ai`          | `neural`, `autograd`                                                                        |
| `experimental`| `ndimage`, `neural`, `series`, `text`, `io`, `datasets`, `graph`, `vision`, `autograd`     |
| `full`        | `standard` + `experimental`                                                                 |

### Individual Feature Flags

| Feature       | Sub-crate              | Description                                        |
|---------------|------------------------|----------------------------------------------------|
| `linalg`      | `scirs2-linalg`        | Linear algebra: decompositions, solvers, matrix functions |
| `stats`       | `scirs2-stats`         | Distributions, hypothesis testing, Bayesian methods |
| `integrate`   | `scirs2-integrate`     | ODE/PDE solvers, quadrature, Monte Carlo integration |
| `interpolate` | `scirs2-interpolate`   | Splines, RBF, MLS, kriging, barycentric interpolation |
| `optimize`    | `scirs2-optimize`      | Unconstrained/constrained/global optimization, metaheuristics |
| `fft`         | `scirs2-fft`           | FFT, DCT/DST, NUFFT, wavelet packets, spectral analysis |
| `oxifft`      | `scirs2-fft` (OxiFFT) | High-performance pure-Rust FFT via OxiFFT backend  |
| `special`     | `scirs2-special`       | Special functions: Bessel, gamma, elliptic, hypergeometric |
| `signal`      | `scirs2-signal`        | Signal processing: filters, STFT, Kalman, source separation |
| `sparse`      | `scirs2-sparse`        | Sparse matrix formats (CSR/CSC/COO), sparse solvers |
| `spatial`     | `scirs2-spatial`       | KD-tree, R*-tree, Voronoi, convex hull, geodata    |
| `cluster`     | `scirs2-cluster`       | K-means, DBSCAN, GMM, hierarchical, spectral clustering |
| `transform`   | `scirs2-transform`     | PCA, ICA, UMAP, t-SNE, NMF, metric learning        |
| `metrics`     | `scirs2-metrics`       | Classification, regression, ranking, segmentation metrics |
| `ndimage`     | `scirs2-ndimage`       | N-dimensional image processing: morphology, segmentation |
| `neural`      | `scirs2-neural`        | Neural networks, transformers, GNNs, training utilities |
| `autograd`    | `scirs2-autograd`      | Automatic differentiation, higher-order gradients, JVP/VJP |
| `series`      | `scirs2-series`        | Time series: ARIMA, Prophet, state-space, forecasting |
| `text`        | `scirs2-text`          | NLP: tokenization, NER, topic models, embeddings   |
| `io`          | `scirs2-io`            | Data I/O: CSV, JSON, HDF5-lite, Parquet-lite, Arrow |
| `datasets`    | `scirs2-datasets`      | Benchmark datasets and synthetic data generators   |
| `graph`       | `scirs2-graph`         | Graph algorithms, GNNs, community detection        |
| `vision`      | `scirs2-vision`        | Computer vision: feature detection, stereo, depth  |

### Advanced Feature Flags (v0.5.0+)

| Feature         | Requires          | Description                                                          |
|-----------------|-------------------|----------------------------------------------------------------------|
| `oxifft`        | `fft`             | High-performance pure-Rust FFT via OxiFFT backend                    |
| `cuda`          | —                 | CUDA GPU-accelerated paths (placeholder; future release)             |
| `rocm`          | —                 | AMD ROCm GPU acceleration (placeholder; future release)              |
| `distributed`   | `scirs2-core`     | Cluster/distributed computing abstractions via `scirs2-core`         |
| `jit`           | `autograd`        | JAX-style JIT functional transformation framework                    |
| `mobile`        | —                 | iOS Metal + Android NNAPI gated paths (placeholder; future release)  |
| `nn`            | `neural`          | Neural architecture re-exports under `scirs2::nn` namespace          |
| `symbolic`      | —                 | Planned `scirs2-symbolic` crate integration (placeholder)            |
| `benchmarks`    | `datasets`        | Expose benchmark helpers from `scirs2-datasets`                      |
| `wasm`          | `scirs2-wasm`     | WebAssembly bindings — re-exports `scirs2-wasm` as `scirs2::wasm`    |

## Re-exported Sub-crates

All sub-crates are accessible as top-level modules when their feature is enabled:

| Module path           | Feature flag   | Domain                             |
|-----------------------|----------------|------------------------------------|
| `scirs2::core`        | always         | Core utilities, SIMD, GPU, memory  |
| `scirs2::linalg`      | `linalg`       | Linear algebra                     |
| `scirs2::stats`       | `stats`        | Statistics and probability         |
| `scirs2::integrate`   | `integrate`    | Numerical integration and ODEs     |
| `scirs2::interpolate` | `interpolate`  | Interpolation                      |
| `scirs2::optimize`    | `optimize`     | Optimization                       |
| `scirs2::fft`         | `fft`          | Fourier transforms                 |
| `scirs2::special`     | `special`      | Special functions                  |
| `scirs2::signal`      | `signal`       | Signal processing                  |
| `scirs2::sparse`      | `sparse`       | Sparse matrices                    |
| `scirs2::spatial`     | `spatial`      | Spatial algorithms                 |
| `scirs2::cluster`     | `cluster`      | Clustering                         |
| `scirs2::transform`   | `transform`    | Dimensionality reduction           |
| `scirs2::metrics`     | `metrics`      | Evaluation metrics                 |
| `scirs2::ndimage`     | `ndimage`      | Image processing                   |
| `scirs2::neural`      | `neural`       | Neural networks                    |
| `scirs2::autograd`    | `autograd`     | Automatic differentiation          |
| `scirs2::series`      | `series`       | Time series analysis               |
| `scirs2::text`        | `text`         | Natural language processing        |
| `scirs2::io`          | `io`           | Data input/output                  |
| `scirs2::datasets`    | `datasets`     | Datasets and benchmarks            |
| `scirs2::graph`       | `graph`        | Graph algorithms                   |
| `scirs2::vision`      | `vision`       | Computer vision                    |

## Quick Start Examples

### Linear Algebra + Statistics

```rust
use scirs2::prelude::*;
use ndarray::array;

fn main() -> CoreResult<()> {
    // Matrix eigendecomposition (scirs2-linalg)
    let a = array![[4.0_f64, 2.0], [1.0, 3.0]];
    let eig = linalg::eigen::eig(&a)?;
    println!("Eigenvalues: {:?}", eig.eigenvalues);

    // Normal distribution and sampling (scirs2-stats)
    let normal = stats::distributions::Normal::new(0.0_f64, 1.0)?;
    let samples = normal.rvs(1000)?;
    let mean = stats::descriptive::mean(&samples.view())?;
    println!("Sample mean: {:.4}", mean);

    Ok(())
}
```

### Optimization + Special Functions

```rust
use scirs2::{optimize, special};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Minimize the Rosenbrock function (scirs2-optimize)
    let result = optimize::unconstrained::minimize(
        |x: &[f64]| (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2),
        &[0.0, 0.0],
        "L-BFGS-B",
        None,
    )?;
    println!("Minimum at: {:?}", result.x);

    // Gamma function (scirs2-special)
    let g5 = special::gamma::gamma(5.0_f64)?;
    println!("Gamma(5) = {}", g5);  // 24.0

    Ok(())
}
```

### Signal Processing + FFT

```rust
use scirs2::{signal, fft};
use ndarray::Array1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let t = Array1::linspace(0.0_f64, 1.0, 1024);
    let sig = t.mapv(|ti| (2.0 * std::f64::consts::PI * 50.0 * ti).sin());

    // Butterworth lowpass filter (scirs2-signal)
    let sos = signal::filter::iirfilter(4, &[100.0], None, None, "butter", "low", 1024.0)?;
    let filtered = signal::filter::sosfilt(&sos, sig.view())?;

    // FFT of filtered signal (scirs2-fft)
    let spectrum = fft::rfft(filtered.view())?;
    println!("FFT output length: {}", spectrum.len());

    Ok(())
}
```

### Neural Network + Autograd

```rust
use scirs2::{neural, autograd};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Build a feed-forward network (scirs2-neural)
    let model = neural::Sequential::new()
        .dense(784, 256)?
        .relu()
        .dense(256, 10)?
        .softmax()?;

    // Compute gradients via automatic differentiation (scirs2-autograd)
    autograd::run(|ctx| {
        let x = ctx.placeholder("x", &[1, 784]);
        let logits = model.forward_autograd(ctx, x)?;
        let loss = autograd::losses::cross_entropy(logits, &[3])?;
        let grads = autograd::grad(&[loss], model.parameters())?;
        println!("Gradient norms computed: {}", grads.len());
        Ok(())
    })?;

    Ok(())
}
```

## Architecture

SciRS2 follows a strict layered architecture per the [SciRS2 Ecosystem Policy](../scirs2-core/SCIRS2_POLICY.md):

```
scirs2 (meta-crate, re-exports all)
├── scirs2-core          ← Only crate allowed external dependencies (OxiBLAS, OxiFFT, etc.)
├── Scientific Computing Layer
│   ├── scirs2-linalg    ← Linear algebra
│   ├── scirs2-stats     ← Statistics
│   ├── scirs2-optimize  ← Optimization
│   ├── scirs2-integrate ← Integration / ODEs
│   ├── scirs2-interpolate ← Interpolation
│   ├── scirs2-fft       ← Fourier analysis
│   ├── scirs2-special   ← Special functions
│   ├── scirs2-signal    ← Signal processing
│   ├── scirs2-sparse    ← Sparse matrices
│   └── scirs2-spatial   ← Spatial algorithms
├── Machine Learning Layer
│   ├── scirs2-cluster   ← Clustering
│   ├── scirs2-transform ← Dimensionality reduction
│   ├── scirs2-metrics   ← Evaluation metrics
│   ├── scirs2-neural    ← Neural networks
│   └── scirs2-autograd  ← Automatic differentiation
└── Application Layer
    ├── scirs2-ndimage   ← Image processing
    ├── scirs2-series    ← Time series
    ├── scirs2-graph     ← Graph algorithms
    ├── scirs2-vision    ← Computer vision
    ├── scirs2-text      ← NLP
    ├── scirs2-io        ← Data I/O
    └── scirs2-datasets  ← Datasets
```

### Key Design Principles

- **Pure Rust by Default**: Zero C/Fortran dependencies. OxiBLAS replaces OpenBLAS; OxiFFT replaces FFTW. All optional C-backed features are feature-gated.
- **Core-Only External Dependencies**: Only `scirs2-core` links to external libraries. All other crates use `scirs2-core` abstractions.
- **No `unwrap()`**: The entire ecosystem enforces proper error propagation through `Result`.
- **Workspace-Unified**: All crates share the same version, edition, and lint configuration from the workspace root `Cargo.toml`.

## Performance

SciRS2 v0.6.2 delivers production-grade performance:

- **SIMD Acceleration**: AVX2/AVX-512/NEON paths for 3-12x speedups on element-wise operations
- **GPU Backends**: Metal (Apple Silicon), with CUDA and ROCm in development
- **Pure Rust FFT**: OxiFFT backend for competitive FFT throughput without C dependencies
- **Parallel Algorithms**: Rayon-based parallel iterators in compute-intensive paths
- **Memory-Efficient**: Buddy/Slab/Compaction allocators, arena allocators for hot paths

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for contribution guidelines.

## License

Apache License 2.0. See [LICENSE](../LICENSE) for details.

Copyright COOLJAPAN OU (Team Kitasan)
