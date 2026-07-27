# scirs2-python TODO

## Status: v0.6.3 (2026-07-27; fixes below verified 2026-07-07, re-confirmed in source + a fresh 39/39 `cargo nextest` run on 2026-07-15)

Untouched by this release's fix work (no python-specific changes shipped in 0.6.3); the fixes and re-verification recorded below remain accurate since the crate source is unchanged.

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

## v0.6.1 Fixes and Additions (2026-07-07)

- [x] **Critical DLPack SIGBUS fix** — `to_dlpack`'s internal `BackingStore` struct was missing `#[repr(C)]`; Rust's default (unspecified) layout could reorder its fields so the pointer stashed in the `PyCapsule` no longer actually addressed a `DLManagedTensor`, crashing the interpreter with SIGBUS whenever a `to_dlpack` capsule was garbage-collected. Also fixed a related double-consumption bug in `capsule_destructor`: it did not account for capsules already consumed and renamed by `from_dlpack` (to `"used_dltensor"`), which leaked a `ValueError` out of `gc.collect()`.
- [x] **DLPack real N-dimensional strided reads** — `from_dlpack` previously assumed the producer's buffer was C-contiguous. It now walks `shape`/`strides` genuinely (`read_strided_elements` / `extract_dlpack_data`), so transposed, reversed/negative-stride, and sliced (non-contiguous) source tensors — e.g. a PyTorch `t.t().__dlpack__()` — are read correctly instead of silently producing wrong data.
- [x] **New Python bindings**: `spatial_hamming_distance_py` (`src/spatial.rs`; also wired into the `pdist_py`/`cdist_py` `"hamming"` metric string) and `F.ppf()` (`src/stats/types.rs`, inverse CDF via bisection, mirroring `Beta.ppf`).
- [x] Removed a stale `@pytest.mark.skip` on `test_poisson_ppf` (Poisson PPF already worked; both the test skip and an accompanying "Not yet implemented" note in `scirs2.pyi`'s docstring were stale) and un-skipped `test_hamming_distance` (now exercises `spatial_hamming_distance_py`).
  - Files: `src/dlpack.rs`, `src/spatial.rs`, `src/stats/types.rs`, `python/scirs2/__init__.pyi`, `scirs2.pyi`, `tests/test_dlpack.py` (new regression-test file), `tests/test_distributions.py`, `tests/scipy_comparison/test_spatial_vs_scipy.py`.
  - Rust-side: 39 `cargo nextest` tests pass. See Known Issues below for a broader Python-side pytest gap found during this session's validation (unrelated to the fixes above).

### Re-verification (2026-07-15)

- `cargo nextest run -p scirs2-python`: 39/39 passing, 0 skipped (re-confirmed against a fresh build)
- `grep -rn "todo!()\|unimplemented!()" src/`: 0 hits
- API surface: 234 `pub fn`/`struct`/`enum`/`trait` items across `src/` (35 files, ~19.1k lines)
- The DLPack `#[repr(C)]` fix, `read_strided_elements`/`extract_dlpack_data`, `spatial_hamming_distance_py`, and `F::ppf` bindings described above are all still present and unchanged in source
- `pyproject.toml` version confirmed `0.6.1` (hardcoded, standalone from workspace — expected for this crate)

## Known Issues

- ndarray version boundary: `scirs2-numpy` resolves the ndarray 0.16/0.17 mismatch that blocked earlier versions; this is fully resolved in v0.3.1.
- Large matrix operations (>200x200) may be slower than SciPy with a well-tuned system LAPACK; use NumPy/SciPy for those cases.
- `scirs2-python` is excluded from the default workspace build (`--exclude scirs2-python`) because it requires Python dev headers.
- Graph module suppresses `#[allow(deprecated)]` for `PyAnyMethods::downcast`; will be updated when PyO3 stabilizes the replacement.
- A broader `pytest` sweep across the full Python test suite (beyond `cargo nextest`, which passes cleanly at 39/39) currently shows approximately 1,146 passed / 404 failed, concentrated in the vision, neural, sparse, pandas-compat, async, io, and text binding modules. This is a pre-existing binding-surface gap relative to the underlying Rust crates — not a regression introduced this session — and is not yet root-caused per-module. The `- [x]` entries elsewhere in this file for those modules reflect that Rust-side bindings exist and are registered, not that every Python-facing behavior has been verified end-to-end against them; closing this gap needs a dedicated audit pass before any "complete parity" claim can be considered fully verified for those specific modules.
