# SciRS2 - Python Bindings

**SciRS2**: High-performance scientific computing in Rust with Python bindings. A comprehensive, type-safe alternative to SciPy with exceptional performance for statistical analysis, linear algebra, FFT, signal processing, clustering, and more.

[![Partial](https://img.shields.io/badge/status-partial-yellow)]()
[![PyPI](https://img.shields.io/pypi/v/scirs2)](https://pypi.org/project/scirs2/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Python](https://img.shields.io/pypi/pyversions/scirs2)](https://pypi.org/project/scirs2/)
[![Version](https://img.shields.io/badge/version-0.6.5-green)]()

**Testing status**: the Rust-side test suite is green (39/39 `cargo nextest` tests passing as of v0.6.5). A broader Python-facing `pytest` sweep across the full test suite currently shows approximately 1,146 passed / 404 failed, concentrated in the vision, neural, sparse, pandas-compat, async, io, and text binding modules — see [TODO.md](TODO.md#known-issues) for details. This is a pre-existing binding-surface gap, not a regression; treat module-level completeness claims below as "bindings exist and are registered," not as "fully verified end-to-end."

## Overview

`scirs2-python` provides Python bindings for the entire SciRS2 scientific computing ecosystem using [PyO3](https://pyo3.rs/) and [Maturin](https://github.com/PyO3/maturin). It leverages `scirs2-numpy` — a SciRS2-maintained fork of `rust-numpy` with native ndarray 0.17 support — for zero-copy NumPy array interoperability.

### Key Characteristics

- **Exceptional Statistics Performance**: Up to 410x faster than SciPy for higher-order moments (skewness, kurtosis) on small/medium datasets
- **Pure Rust Stack**: OxiBLAS for linear algebra, OxiFFT for transforms — no system OpenBLAS or FFTW required
- **Zero-Copy NumPy Interop**: Direct memory sharing between Rust and NumPy arrays via `scirs2-numpy`
- **SciPy-Compatible API**: Familiar naming conventions for Python scientists migrating from SciPy
- **Maturin Build System**: Single `pip install scirs2` — no C/Fortran compiler needed
- **Type Stubs**: `.pyi` files for full IDE autocompletion and type checking
- **Feature-Gated Modules**: Enable only the SciRS2 crates you need

## Architecture

```
scirs2-python/
├── src/
│   ├── lib.rs          - PyO3 module registration
│   ├── linalg.rs       - Linear algebra (OxiBLAS-powered)
│   ├── stats.rs        - Statistics and distributions
│   ├── fft.rs          - FFT (OxiFFT-powered)
│   ├── cluster.rs      - Clustering algorithms
│   ├── series.rs       - Time series analysis
│   ├── signal.rs       - Signal processing
│   ├── optimize.rs     - Optimization algorithms
│   ├── spatial.rs      - Spatial algorithms
│   ├── sparse.rs       - Sparse matrix operations
│   ├── ndimage.rs      - N-dimensional image processing
│   ├── graph.rs        - Graph algorithms
│   ├── metrics.rs      - ML evaluation metrics
│   ├── io.rs           - File I/O (CSV, HDF5, Parquet, etc.)
│   ├── datasets.rs     - Dataset loading and generation
│   ├── transform.rs    - Dimensionality reduction
│   ├── text.rs         - NLP and text processing
│   ├── vision.rs       - Computer vision
│   ├── linalg_ext.rs   - Extended linear algebra (v0.3.4+)
│   ├── signal_ext.rs   - Extended signal processing (v0.3.4+)
│   ├── optimize_ext.rs - Extended optimization (v0.3.4+)
│   └── stats/
│       └── mcmc_gp.rs  - MCMC and Gaussian process bindings (v0.3.4+)
├── tests/
│   └── test_module_structure.py
├── scirs2.pyi          - Type stubs
└── pyproject.toml
```

## Installation

```bash
pip install scirs2
```

For development (builds Rust from source):
```bash
pip install maturin
git clone https://github.com/cool-japan/scirs
cd scirs/scirs2-python
maturin develop --release
```

No system BLAS, OpenBLAS, or FFTW installation required — SciRS2 uses pure Rust OxiBLAS and OxiFFT.

## Quick Start

### Statistics (Fastest Module)

SciRS2 delivers exceptional speed for statistical analysis:

```python
import numpy as np
import scirs2

data = np.random.randn(1000)

# Basic statistics (5-25x faster than NumPy on small-medium data)
mean     = scirs2.mean_py(data)           # 8x faster
std      = scirs2.std_py(data, 0)         # 14x faster
var      = scirs2.var_py(data, 1)
median   = scirs2.median_py(data)
iqr      = scirs2.iqr_py(data)

# Higher-order moments (50-410x faster than SciPy!)
skewness = scirs2.skew_py(data)           # 52x faster
kurtosis = scirs2.kurtosis_py(data)       # 52x faster

# Correlation (17-127x faster)
x = np.random.randn(100)
y = np.random.randn(100)
corr = scirs2.correlation_py(x, y)
cov  = scirs2.covariance_py(x, y, 1)

# Full descriptive summary
summary = scirs2.describe_py(data)
# Returns: {'mean', 'std', 'min', 'max', 'skewness', 'kurtosis', ...}
```

### Linear Algebra

```python
import numpy as np
import scirs2

A = np.array([[4.0, 2.0], [2.0, 3.0]])
b = np.array([1.0, 2.0])

# Basic operations
det   = scirs2.det_py(A)
inv   = scirs2.inv_py(A)
trace = scirs2.trace_py(A)

# Decompositions
lu   = scirs2.lu_py(A)       # {'L', 'U', 'P'}
qr   = scirs2.qr_py(A)       # {'Q', 'R'}
svd  = scirs2.svd_py(A)      # {'U', 'S', 'Vt'}
chol = scirs2.cholesky_py(A)

# Eigenvalues/vectors
eig  = scirs2.eig_py(A)      # {'eigenvalues_real', 'eigenvalues_imag', 'eigenvectors'}
eigh = scirs2.eigh_py(A)     # For symmetric matrices

# Solvers
x     = scirs2.solve_py(A, b)
lstsq = scirs2.lstsq_py(A, b)

# Norms and properties
norm_fro = scirs2.matrix_norm_py(A, "fro")
norm_vec = scirs2.vector_norm_py(b, 2)
cond     = scirs2.cond_py(A)
rank     = scirs2.matrix_rank_py(A)
```

### FFT (OxiFFT Backend)

```python
import numpy as np
import scirs2

data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

# FFT (2-5x faster than NumPy on small data < 2K samples)
fft_result = scirs2.fft_py(data)           # {'real', 'imag'}
real, imag = fft_result['real'], fft_result['imag']

ifft_result = scirs2.ifft_py(
    np.array(real), np.array(imag)
)

rfft  = scirs2.rfft_py(data)
irfft = scirs2.irfft_py(
    np.array(rfft['real']), np.array(rfft['imag']), len(data)
)

# DCT
dct  = scirs2.dct_py(data, 2)             # Type-II DCT
idct = scirs2.idct_py(np.array(dct), 2)

# Helpers
freqs    = scirs2.fftfreq_py(len(data), 1.0)
rfreqs   = scirs2.rfftfreq_py(len(data), 1.0)
shifted  = scirs2.fftshift_py(data)
fast_len = scirs2.next_fast_len_py(100, False)
```

### Clustering

```python
import numpy as np
import scirs2

X = np.vstack([
    np.random.randn(100, 2) + [0, 0],
    np.random.randn(100, 2) + [5, 5],
])

# K-Means
kmeans = scirs2.KMeans(n_clusters=2)
kmeans.fit(X)
labels   = kmeans.labels
inertia  = kmeans.inertia_

# Cluster quality metrics
silhouette = scirs2.silhouette_score_py(X, labels)
db_score   = scirs2.davies_bouldin_score_py(X, labels)
ch_score   = scirs2.calinski_harabasz_score_py(X, labels)

# Preprocessing
X_std  = scirs2.standardize_py(X, True)
X_norm = scirs2.normalize_py(X, "l2")
```

### Time Series

```python
import numpy as np
import scirs2

data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
ts   = scirs2.PyTimeSeries(data, None)

# Transformations
diff1    = scirs2.apply_differencing(ts, 1)
seasonal = scirs2.apply_seasonal_differencing(ts, 4)

# ARIMA modelling
arima = scirs2.PyARIMA(1, 1, 0)
arima.fit(ts)
forecast = arima.forecast(5)
print(arima.summary())

# Box-Cox transform
result      = scirs2.boxcox_transform(ts, None)   # auto lambda
transformed = result['transformed']
lambda_val  = result['lambda']
recovered   = scirs2.boxcox_inverse(np.array(transformed), lambda_val)

# Stationarity test
adf = scirs2.adf_test(ts, None)
print(f"ADF statistic: {adf['statistic']}, p-value: {adf['p_value']}")

# STL decomposition
decomp   = scirs2.stl_decomposition(ts, 4)
trend    = decomp['trend']
seasonal = decomp['seasonal']
residual = decomp['residual']
```

### MCMC and Gaussian Processes (v0.3.4+)

```python
import numpy as np
import scirs2

# Gaussian Process regression
X_train = np.linspace(0, 2 * np.pi, 20).reshape(-1, 1)
y_train = np.sin(X_train.ravel()) + 0.1 * np.random.randn(20)

gp = scirs2.GaussianProcessRegressor(kernel='rbf', length_scale=1.0, noise=0.01)
gp.fit(X_train, y_train)

X_test = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)
mean, std = gp.predict(X_test, return_std=True)

# MCMC sampling
sampler = scirs2.MetropolisHastings(log_prob_fn=my_log_prob, step_size=0.1)
samples = sampler.sample(initial=np.zeros(3), n_samples=10000, burnin=1000)
```

### Extended Optimization (v0.3.4+)

```python
import numpy as np
import scirs2

# Gradient-free optimization
result = scirs2.minimize_nelder_mead(
    lambda x: (x[0] - 1)**2 + (x[1] + 2)**2,
    x0=np.array([0.0, 0.0]),
    tol=1e-8
)

# Constrained optimization
result = scirs2.minimize_slsqp(
    fun=objective,
    x0=x0,
    constraints=[{'type': 'eq', 'fun': equality_constraint}],
    bounds=[(0, None), (0, None)]
)
```

## Performance Guide

### Where SciRS2 Excels

| Operation | Data Size | Speedup vs SciPy | Notes |
|-----------|-----------|------------------|-------|
| Skewness | 100 | **410x** | Higher-order moments |
| Kurtosis | 100 | **408x** | Higher-order moments |
| Pearson correlation | 100 | **127x** | Small dataset |
| IQR | 100 | **95x** | Quartile calculations |
| Percentile | 100 | **49x** | Distribution analysis |
| Skewness | 1,000 | **52x** | Medium dataset |
| Std | 100 | **25x** | Small data variability |
| Mean | 100 | **11x** | Small data average |
| FFT (rfft) | 128 | **5.2x** | Small signal |
| Linear solve | 10x10 | **9.4x** | Small systems |

### Where NumPy/SciPy May Win

| Operation | Size | Notes |
|-----------|------|-------|
| Linear algebra (SVD, QR) | 200x200+ | SciPy LAPACK highly optimized |
| FFT | 32K+ samples | NumPy FFT optimized for large N |
| Basic stats | 100K+ elements | NumPy C SIMD optimizations |

### Recommended Hybrid Approach

```python
import numpy as np
import scirs2

data = np.random.randn(1000)

# Use scirs2 for statistics on small-medium data
skewness = scirs2.skew_py(data)      # 52x faster than SciPy
kurtosis = scirs2.kurtosis_py(data)  # 52x faster than SciPy
mean     = scirs2.mean_py(data)      # 8x faster
std      = scirs2.std_py(data, 0)    # 14x faster

# Use scirs2 for FFT on small signals
signal = np.random.randn(512)
rfft   = scirs2.rfft_py(signal)      # 3x faster than NumPy

# Fall back to NumPy/SciPy for large transforms
large_signal = np.random.randn(65536)
spectrum     = np.fft.rfft(large_signal)  # NumPy wins here
```

## Feature Flags

Enable specific SciRS2 crates in `pyproject.toml`:

```toml
[features]
default = ["linalg", "stats", "fft", "cluster", "series"]
linalg      = ["scirs2-linalg"]
stats       = ["scirs2-stats"]
fft         = ["scirs2-fft"]
cluster     = ["scirs2-cluster"]
series      = ["scirs2-series"]
signal      = ["scirs2-signal"]
optimize    = ["scirs2-optimize"]
spatial     = ["scirs2-spatial"]
sparse      = ["scirs2-sparse"]
ndimage     = ["scirs2-ndimage"]
graph       = ["scirs2-graph"]
metrics     = ["scirs2-metrics"]
io          = ["scirs2-io"]
datasets    = ["scirs2-datasets"]
transform   = ["scirs2-transform"]
text        = ["scirs2-text"]
vision      = ["scirs2-vision"]
```

## Type Hints

```python
import scirs2

# IDE provides full autocompletion from .pyi stubs
result: float    = scirs2.det_py(matrix)
svd_res: dict    = scirs2.svd_py(matrix)
fft_res: dict    = scirs2.fft_py(data)
labels: np.ndarray = scirs2.KMeans(n_clusters=3).fit_predict(X)
```

## Building from Source

```bash
# 1. Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 2. Install maturin
pip install maturin

# 3. Build and install in development mode
cd scirs2-python
maturin develop --release

# 4. Run tests
pip install pytest numpy
pytest tests/
```

## Dependencies

- [PyO3](https://pyo3.rs/) - Rust/Python FFI
- [scirs2-numpy](../scirs2-numpy/) - SciRS2 fork of rust-numpy with ndarray 0.17 support
- [Maturin](https://github.com/PyO3/maturin) - Build system for Python/Rust extensions
- [OxiBLAS](https://crates.io/crates/oxiblas) - Pure Rust BLAS/LAPACK (no system dependencies)
- [OxiFFT](https://crates.io/crates/oxifft) - Pure Rust FFT

## Related Projects

- [SciRS2 Core](https://github.com/cool-japan/scirs) - Rust scientific computing library
- [scirs2-numpy](../scirs2-numpy/) - NumPy/ndarray 0.17 bridge
- [NumPy](https://numpy.org/) - Array operations
- [SciPy](https://scipy.org/) - Python scientific computing (API inspiration)

## License

Licensed under the Apache License 2.0. See [LICENSE](../LICENSE) for details.

## Authors

COOLJAPAN OU (Team KitaSan)
