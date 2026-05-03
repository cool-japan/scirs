# scirs2-python TODO

## Status: v0.4.3 Released (2026-05-03)

## v0.3.3 Completed

### Infrastructure
- [x] PyO3-based Python/Rust interop layer
- [x] Maturin build system (`pyproject.toml`)
- [x] `scirs2-numpy` integration for native ndarray 0.17 support
- [x] Zero-copy NumPy array sharing via `scirs2-numpy` bridge
- [x] Type stubs (`scirs2.pyi`) for IDE autocompletion
- [x] Feature-gated modules for selective SciRS2 crate inclusion
- [x] `async_ops.rs` - Async Python-compatible operations
- [x] `error.rs` - Unified error type translating Rust errors to Python exceptions
- [x] `pandas_compat.rs` - pandas DataFrame interop utilities

### Linear Algebra (`linalg.rs`, `linalg_ext.rs`)
- [x] `det_py`, `inv_py`, `trace_py` - Basic matrix properties
- [x] `lu_py`, `qr_py`, `svd_py`, `cholesky_py` - Decompositions
- [x] `eig_py`, `eigh_py` - Eigenvalues and eigenvectors
- [x] `solve_py`, `lstsq_py` - Linear solvers
- [x] `matrix_norm_py`, `vector_norm_py`, `cond_py`, `matrix_rank_py`
- [x] Extended: `expm_py` (matrix exponential), `logm_py`, `sqrtm_py`
- [x] Extended: `schur_py`, `kron_py`, `block_diag_py`
- [x] OxiBLAS backend — no system OpenBLAS needed

### Statistics (`stats.rs`, `stats/mcmc_gp.rs`)
- [x] Descriptive: `mean_py`, `std_py`, `var_py`, `median_py`, `percentile_py`, `iqr_py`
- [x] Higher-order moments: `skew_py`, `kurtosis_py`
- [x] Correlation: `correlation_py`, `covariance_py`
- [x] Full summary: `describe_py`
- [x] MCMC: `MetropolisHastings`, `HamiltonianMC`, `NUTS` samplers
- [x] Gaussian Process: `GaussianProcessRegressor` with RBF, Matern, periodic kernels
- [x] Survival analysis: `KaplanMeier`, `CoxPH`, `NelsonAalen`

### FFT (`fft.rs`)
- [x] `fft_py`, `ifft_py` - Complex FFT
- [x] `rfft_py`, `irfft_py` - Real FFT
- [x] `dct_py`, `idct_py` - Discrete cosine transform (types I-IV)
- [x] `fftfreq_py`, `rfftfreq_py`, `fftshift_py`, `ifftshift_py`
- [x] `next_fast_len_py`
- [x] OxiFFT backend — no system FFTW needed

### Clustering (`cluster.rs`)
- [x] `KMeans` - K-Means clustering
- [x] `silhouette_score_py`, `davies_bouldin_score_py`, `calinski_harabasz_score_py`
- [x] `standardize_py`, `normalize_py`
- [x] DBSCAN, Hierarchical clustering bindings

### Time Series (`series.rs`)
- [x] `PyTimeSeries` - Time series container
- [x] `PyARIMA` - ARIMA modelling (fit, forecast, summary)
- [x] `apply_differencing`, `apply_seasonal_differencing`
- [x] `boxcox_transform`, `boxcox_inverse`
- [x] `adf_test` - Augmented Dickey-Fuller stationarity test
- [x] `stl_decomposition` - STL trend/seasonal/residual decomposition

### Signal Processing (`signal.rs`, `signal_ext.rs`)
- [x] Filter design: `butter_py`, `cheby1_py`, `cheby2_py`
- [x] Filter application: `lfilter_py`, `sosfilt_py`
- [x] Spectrogram, STFT, periodogram
- [x] Extended: Kalman filter, EKF, UKF Python bindings
- [x] Extended: adaptive filter (LMS, RLS) bindings

### Optimization (`optimize.rs`, `optimize_ext.rs`)
- [x] Unconstrained: Nelder-Mead, BFGS, L-BFGS-B, CG
- [x] Constrained: SLSQP with equality and inequality constraints
- [x] Global: differential evolution, basin-hopping
- [x] Curve fitting: `curve_fit_py`
- [x] Extended: SQP, interior-point LP/QP bindings

### Other Modules
- [x] `spatial.rs` - KD-tree, ball tree, distance metrics
- [x] `sparse.rs` - CSR/CSC sparse matrix ops
- [x] `ndimage.rs` - Gaussian blur, morphology, labeling
- [x] `graph.rs` - Graph construction, shortest paths, community detection
- [x] `metrics.rs` - Classification, regression, clustering metrics
- [x] `io.rs` - CSV, HDF5, Parquet, Arrow read/write
- [x] `datasets.rs` - `load_iris_py`, `load_boston_py`, `make_classification_py`, etc.
- [x] `transform.rs` - PCA, ICA, t-SNE, UMAP bindings
- [x] `text.rs` - Tokenization, TF-IDF, Word2Vec, NER bindings
- [x] `vision.rs` - Image processing, feature detection bindings

### Testing
- [x] `tests/test_module_structure.py` - Module import and structure tests
- [x] Basic numerical correctness tests for each module
- [x] SciPy comparison tests for statistics and FFT

## v0.4.2 Changes (Wave 43, 2026-04-12)

### Policy Compliance Fixes
- [x] Eliminated all `expect()` / `unwrap()` violations in production code
  - `integrate.rs`: Python callback closures now return NaN on error (no panic)
  - `interpolate.rs`: Array ops use `ok_or_else` / `map_err` instead of `expect`
  - `stats/functions.rs`: `as_slice()` calls use `ok_or_else`
  - `stats/functions_2.rs`: sort comparator uses `unwrap_or`
  - `stats/mcmc_gp.rs`: HMC gradient callback returns NaN-filled vector on error
  - `optimize.rs`: All five objective-function callback closures use `unwrap_or`
  - `signal.rs`: `as_slice()` calls use `ok_or_else`
  - `series.rs`: sort comparator uses `unwrap_or`
  - `cluster.rs`: `cluster_centers_` access uses `ok_or_else`

### scirs2.special Module — Extended Coverage
- [x] `scirs2.special` module: Bessel, Gamma, hypergeometric functions
  - New: `polygamma(n, x)` — n-th derivative of digamma
  - New: `zeta(s)`, `hurwitz_zeta(s, q)`, `zetac(s)` — Riemann/Hurwitz zeta
  - New: `hyp0f1(v, z)`, `hyp1f1(a, b, z)`, `hyp2f1(a, b, c, z)`, `hyperu(a, b, x)` — hypergeometric functions
  - New: `airy_ai(x)`, `airy_aip(x)`, `airy_bi(x)`, `airy_bip(x)` — Airy functions
  - New: `sici_si(x)`, `sici_ci(x)`, `shichi_shi(x)`, `shichi_chi(x)` — trig/exp integrals
  - New: `betainc(a, b, x)`, `betaincinv(a, b, p)` — regularized incomplete beta
  - New vectorized wrappers: `lgamma_array`, `erfc_array`, `digamma_array`, `j1_array`

### scirs2.interpolate Module — Extended Coverage
- [x] `scirs2.interpolate` module: spline, RBF, PCHIP
  - New: `RBFInterpolator` class — Gaussian, Multiquadric, InverseMultiquadric,
    ThinPlateSpline, Linear, Cubic kernels with configurable epsilon

### scirs2.integrate Module — Already Complete
- [x] `scirs2.integrate` module: ODE solvers (RK45, BDF, LSODA), quadrature
  - `solve_ivp`: RK45/RK23/DOP853/BDF/Radau/LSODA/RK4/Euler methods
  - `quad`: adaptive Gauss-Kronrod quadrature
  - `trapezoid`, `simpson`, `cumulative_trapezoid`, `romberg` array-based integration
  - Note: Python-side tests require a Python/maturin environment to run

### scirs2.stats — Distribution Parity
- [x] Complete parity with all scirs2-stats distributions
  - New: `bernoulli(p)` — Bernoulli distribution
  - New: `nbinom(n, p)` — Negative Binomial distribution
  - New: `hypergeom(m, n, k)` — Hypergeometric distribution
  - Previously bound: norm, binom, poisson, expon, uniform, beta, gamma, chi2, t, cauchy, f,
    lognorm, weibull_min, laplace, logistic, pareto, geom

## v0.4.0 Roadmap

### Full API Coverage
- [x] Complete parity with all scirs2-linalg functions
- [x] Complete parity with all scirs2-stats distributions
- [x] `scirs2.special` module: Bessel, Gamma, hypergeometric functions
- [x] `scirs2.interpolate` module: spline, RBF, PCHIP
- [x] `scirs2.integrate` module: ODE solvers (RK45, BDF, LSODA), quadrature

### Async Python Support
- [x] Native `async/await` for long-running computations
- [x] `asyncio`-compatible interface using `pyo3-asyncio`
- [x] Parallel batch processing with Python threads

### GPU Tensor Bridge
- [x] Optional CUDA tensor bridge via `cudarc` or `candle` — CPU dispatch layer with `cuda_bridge` feature gate in `gpu_ops.rs`; full cudarc integration deferred until GPU CI available
- [x] PyTorch tensor interop (zero-copy via DLPack)
- [x] GPU-accelerated matrix operations exposed to Python — `gpu_ops.rs`: `gpu_matmul`, `gpu_elementwise`, `gpu_matrix_add`, `gpu_matrix_scale`, `gpu_frobenius_norm`, `cuda_tensor_matmul` (18 tests)

### Type System Improvements
- [x] `Protocol`-based type stubs for duck-typed APIs
- [x] Full `mypy`-compatible stubs for all modules
- [x] Auto-generated stubs from PyO3 introspection — `examples/generate_stubs.rs`: parses `#[pyfunction]` attrs, infers Python return types, emits valid `.pyi` output (10 tests)

### Packaging and Distribution
- [x] Pre-built wheels for `manylinux2014`, `musllinux`, `macOS-arm64`, `macOS-x86_64`, `win-amd64`
- [x] GitHub Actions release pipeline via Maturin's `zig` cross-compilation
- [x] PyPI publishing automation

### Documentation
- [x] Sphinx API documentation with `maturin-sphinx` plugin
- [x] SciPy migration guide with side-by-side examples
- [x] Performance comparison notebooks (Jupyter)

## Known Issues

- ndarray version boundary: `scirs2-numpy` resolves the ndarray 0.16/0.17 mismatch that blocked earlier versions; this is fully resolved in v0.3.1.
- Large matrix operations (>200x200) may be slower than SciPy with a well-tuned system LAPACK; use NumPy/SciPy for those cases.
- `scirs2-python` is excluded from the default workspace build (`--exclude scirs2-python`) because it requires Python dev headers.
- Graph module suppresses `#[allow(deprecated)]` for `PyAnyMethods::downcast`; will be updated when PyO3 stabilizes the replacement.
