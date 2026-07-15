# SciRS2 - Scientific Computing and AI in Rust

[![crates.io](https://img.shields.io/crates/v/scirs2.svg)](https://crates.io/crates/scirs2)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Lines of Code](https://img.shields.io/badge/Rust_SLoC-3.9M-blue)](https://github.com/cool-japan/scirs)
[![Tests](https://img.shields.io/badge/tests-37.4k-green)](https://github.com/cool-japan/scirs)

**Production-Ready Pure Rust Scientific Computing** • **No System Dependencies** • **10-100x Performance Gains**

SciRS2 is a comprehensive scientific computing and AI/ML infrastructure in **Pure Rust**, providing SciPy-compatible APIs while leveraging Rust's performance, safety, and concurrency features. Unlike traditional scientific libraries, SciRS2 is **100% Pure Rust by default** with no C/C++/Fortran dependencies required, making installation effortless and ensuring cross-platform compatibility.

## Quick Start

```bash
# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Add SciRS2 to your project
cargo add scirs2

# Build your project - no system libraries needed!
cargo build --release
```

## Key Highlights

✨ **Pure Rust**: Zero C/C++/Fortran dependencies (OxiBLAS for BLAS/LAPACK, OxiFFT for FFT)
⚡ **Ultra-Fast**: 10-100x performance improvements through SIMD optimization
🔒 **Memory Safe**: Rust's ownership system prevents memory leaks and data races
🌍 **Cross-Platform**: Linux, macOS, Windows, WebAssembly - identical behavior
🧪 **Battle-Tested**: 37,402 tests (all-features) / 35,632 (default-features) + 5,001 doc-tests — workspace-wide via `cargo nextest` — plus 3.9M+ lines of Rust code, 29 workspace crates
📊 **Comprehensive**: Linear algebra, statistics, ML, FFT, signal processing, computer vision, and more

## Project Overview

SciRS2 provides a complete ecosystem for scientific computing, data analysis, and machine learning in Rust, with production-grade quality and performance that rivals or exceeds traditional C/Fortran-based libraries.

## 🎉 Release Status: v0.6.1 (2026-07-15)

**Latest Stable Release** - v0.6.1 (July 15, 2026) 🚀

- ✅ **37,402 Tests (all-features) / 35,632 (default-features) + 5,001 Doc-Tests**: Full workspace-wide test suite via `cargo nextest`, across 29 workspace crates
- ✅ **3.9M+ Lines of Rust Code**: Comprehensive coverage of scientific computing and AI/ML
- ✅ **29 Workspace Crates**: Specialized modules for every scientific computing domain
- ✅ **80,800+ Public API Items**: Extensive, well-documented API surface
- ✅ **Near-complete implementation**: Minimal stubs remaining across all modules
- ✅ **Pure Rust by Default**: OxiBLAS, OxiFFT, oxiarc-* - zero C/Fortran dependencies
- ✅ **Zero Warnings Policy**: Clean build with 0 compilation errors, 0 clippy warnings, 0 rustdoc warnings
- 📅 **Release Date**: July 15, 2026

**What's New in 0.6.1** — Real Codegen, DLPack Safety & GPU-Dispatch Honesty:

- **Real model-serving codegen**: `scirs2-neural`'s `ModelPackager::package()` now genuinely compiles a Rust project via `cargo`/`rustc` for native binaries (`PackageFormat::Native`) and C-shared-library (`PackageFormat::CSharedLibrary`) targets, replacing the previous placeholder byte-writing stubs; new `serving` module split into `mod.rs`/`types.rs`/`packager.rs`/`codegen.rs`/`tests.rs`.
- **Critical DLPack fix**: a missing `#[repr(C)]` on `to_dlpack`'s `BackingStore` in `scirs2-python` caused a process-crashing SIGBUS whenever the DLPack capsule was garbage-collected; also fixed a double-consumption bug in the capsule destructor and a stride-handling bug in `from_dlpack` that silently produced wrong data for non-contiguous/transposed tensors.
- **GPU-dispatch and hardware-detection honesty**: `scirs2-datasets`'s `AdvancedGpuOptimizer` now performs real wgpu/`GpuNdarray` GPU dispatch instead of a simulated mock (minor breaking change: `BenchmarkResult.gpu_time_ms`/`.speedup` are now `Option<f64>`, `None` when no real GPU dispatch ran); `scirs2-core`'s `PlatformCapabilities::detect()` now runtime-probes for real NVIDIA/Metal GPU hardware instead of hardcoding `cuda_available` to `false`; `scirs2-special`'s permanently-failing `execute_kernel` GPU path now has a real CPU fallback for `gamma`/`bessel_j0`/`erf`; wgpu adapter/device discovery across 7 crates now restricts to `Backends::PRIMARY` (Vulkan/Metal/DX12/BrowserWebGPU) instead of `Backends::all()`, so a broken secondary GL backend can no longer report false availability and then die on first use.
- **New API surface**: `F` distribution `ppf()` (inverse CDF) in `scirs2-stats`, plus opt-in real memory tracking in its SciPy benchmark framework; `hamming()` distance in `scirs2-spatial`; finance facades (`AdvancedMonteCarloEngine`, `RealTimeRiskMonitor`, `ExoticOptionPricer`, `RiskAnalyzer`) in `scirs2-integrate`; `training_history()` on `scirs2-core`'s generative-sampling models (`NormalizingFlow` and others) and `export_data()` on `ProductionProfiler`; `simd_distance_squared_euclidean` on `SimdUnifiedOps`.
- **Correctness fixes**: `scirs2-autograd`'s gradient-dispatch engine now actually invokes `CondOp`/`LogDetOp` backward rules (previously dead code behind op-name string dispatch), plus a proper Wielandt-deflation SVD fix and a condition-number computation fix; `scirs2-signal`'s `waverec` now reconstructs to the correct original length (was `2x` for non-Haar filters) with real perfect-reconstruction verification; restored SIMD paths for `scirs2-interpolate`'s `weighted_sum`/`squared_euclidean_distance` (previously always fell back to scalar); fixed a GEMM operand-layout bug in the `scirs2-datasets`/`scirs2-interpolate`/`scirs2-optimize` CUDA paths (mistagged `ColMajor`, but `oxicuda-blas` requires `RowMajor`); two Miri-detected undefined-behavior fixes (unaligned-reference UB in `cast_bytes_to_slice`, a missing-`else` branch in a SIMD kernel).
- **Housekeeping**: removed 14 dead/orphaned stub types from `scirs2-stats`; split 3 oversized (>2000-line) files into module directories via SplitRS (`scirs2-io`'s `advanced_coordinator.rs`/`hdf5/mod.rs`, `scirs2-series`'s `utils.rs`); dependency bumps — `oxiarc-archive`/`-lz4`/`-bzip2`/`-zstd`/`-deflate`/`-snappy`/`-brotli` `0.3.3` → `0.3.6` (the `0.3.4` regression this was previously pinned around is fixed upstream as of `0.3.5`), `oxisql-core`/`-sqlite-compat` `0.3.1` → `0.3.2`, `oxicuda-*` `0.4.0` → `0.5.0`; `cargo-scirs2-policy`'s `--format` flag is now a validated enum instead of silently falling back on typos.

See [CHANGELOG.md](CHANGELOG.md) `[0.6.1]` for the complete list.

**What's New in 0.6.0** — Pure-Rust CUDA Decentralization:

- **`oxicuda-*` CUDA backends**: introduced a direct, per-crate, off-by-default, runtime-probed `cuda` feature across 10 crates — scirs2-fft, scirs2-symbolic, scirs2-interpolate, scirs2-special, scirs2-stats, scirs2-graph, scirs2-linalg, scirs2-optimize, scirs2-datasets, scirs2-vision — decentralizing GPU acceleration out of `scirs2-core`. Every path is f64-native, real CUDA PTX→driver JIT (not a CPU/wgpu simulation), and additive alongside the existing wgpu portability story.
- **wgpu feature standardization** (breaking): every real wgpu/WebGPU portability feature across the ecosystem renamed to a single `wgpu` name (was `wgpu_backend`/`gpu_wgpu`/`wgpu_fft`/`wgpu_kernels`/`wgpu_rbf`).
- **Pure Rust win**: retired `scirs2-core`'s cudarc-based CUDA backend entirely, dropping the `cudarc` dependency; `GpuBackend::Cuda` now directs users to the per-crate `oxicuda-*` features.

See [CHANGELOG.md](CHANGELOG.md) `[0.6.0]` for the complete list.

**What's New in 0.5.1** — Correctness, API-Stability & Pure Rust Hardening:

- **Exact autograd gradients**: new `TraceBackwardOp` (reverse-mode gradient for `trace`) and a `tensor_ops/decomposition_backward.rs` module with exact spectral gradients — matrix-sqrt (Sylvester equation), matrix-log/matrix-power (Daleckii–Krein divided differences), and reduced-SVD VJP (degenerate-singular-value detection). Previously these returned all-zeros or wrong shapes.
- **API stability**: restored the `#[non_exhaustive]` attribute on `scirs2-core`'s `CoreError` (lost previously); a new `core_error_non_exhaustive` compile-fail test in `scirs2-stability-tests` now guards the contract.
- **GPU/CUDA honesty**: CUDA/GPU paths across scirs2-core, scirs2-linalg, and scirs2-fft now report `BackendNotAvailable` / `NotImplemented` / `Option<f64>` metrics when no real device or measurement exists, instead of returning fabricated contexts or invented throughput numbers.
- **Zarr v2/v3**: chunked-array support is now publicly exported (`pub mod zarr`) — directory stores, codec pipeline, and chunk-boundary slice I/O.
- **Pure Rust hardening**: removed the C/MPFR `rug`, C-backed `rusqlite`, `tokenizers`, and external `either`/`hex`/`urlencoding`/`data-encoding` crates — arbitrary precision now on `oxinum-*`, SQLite on `oxisql-sqlite-compat`, `blake3` built with its no-ASM `pure` feature. Ecosystem dep bumps: OxiArc 0.3.3, OxiSQL 0.3.1, OxiNum 0.1.2, OxiH5 0.1.3, OxiZ 0.2.3, OxiCode 0.2.4, plus OxiEML 0.1.2.
- **Numerical bug fixes**: scirs2-series ADF test now uses an SVD-based Moore–Penrose pseudo-inverse for rank-deficient designs; plus 5 newly-fixed numerical bugs — SSPRK(5,4) integrator (now true 4th-order), Haar perfect-reconstruction check, cheby2/bessel bandpass transfer functions, and TimeSeriesReservoir exponential recency weighting.

**What's New in 0.5.0** — CAS & Symbolic Mathematics + GPU Acceleration (Waves 53–77):

**Core EML/CAS substrate (Waves 53–57)**:
- **scirs2-symbolic** — Complete EML-IR-native CAS: `eml::tree`, `LoweredOp` flat IR, stack-safe evaluator, symbolic differentiation (`grad`/`hessian`/`jacobian`), algebraic simplification with fixed-point rewriting
- **`cas::canonicalize`** — EML-native canonical form with 7 algebraic rewrite rules; e-graph equality saturation; `cas::pattern` engine; `cas::identity_db` (11 trig/log identities)
- **`cas::smt`** — OxiZ QF_NRA SMT integration: `EmlSmtSolver`, structural-hash fast path, Ackermann transcendental encoding for 16 ops, certified rewriting with RAII safety
- **`compile::to_jit`** — Cranelift CPU JIT for `LoweredOp`; `to_jit_auto` auto-dispatches CPU/GPU by batch size threshold
- **`compile::to_gpu`** — WGSL compute shader JIT from `LoweredOp` tape; real wgpu submission via `eval_batch` (Wave 75)
- **LaTeX export** — `eml::to_latex`: `\frac`, `\cdot`, `a^{b}`, `\sqrt`, `\pi`/`e` exact constants, `\operatorname{arcsinh}` — stack-safe on deeply nested trees
- **Symbolic regression** — beam-search SR engine with `with_constraints` (SMT-pruned `BoundedOutput` + `Monotonic`); `discover_multi`, `discover_ode` (SINDy-style); NUMA-aware parallel predict
- **Python bindings** — `scirs2.symbolic` sub-namespace: `PyEmlTree`, `PyCanonical`, `lower`/`simplify`/`grad`/`eval_real`/`discover`; GIL-releasing `discover`
- **scirs2-autograd `symbolic` feature** — `eml_scalar_op` creates differentiable scalar tensors; float-tape ↔ EML gradient parity; criterion benchmark suite
- **scirs2-optimize symbolic** — `symbolic::newton` (exact Hessian/gradient), `symbolic::lagrangian`/`build_kkt` (Lagrangian+KKT Newton), L-BFGS symbolic, trust-region symbolic

**Advanced CAS (Waves 59–70)**:
- `cas::solve`/`solve_system`/`solve_ode` — algebraic solver, 3-tier system solver (Bareiss/Gröbner/transcendental), 5 ODE family solvers
- `cas::integrate_rational` — Risch-LITE rational integration
- `cas::moments_catalog`/`expected_fisher_catalog`/`noether_conservation` — closed-form statistical moments, Fisher information, Noether conservation laws
- `cas::series` — Taylor + Padé approximants in EML form
- `cas::cse_dag::CseDag` — structural-hash CSE DAG with topological-order evaluation
- `diffgeom` module — `Tensor`/`Metric`/`christoffel`/`ricci`/`einstein`/`covariant_derivative`/`Riemann tensor`/`Weyl tensor`
- `scirs2-linalg::symbolic` — `det_symbolic`, `eigenvalues_symbolic_2x2`, `expm_symbolic`, `StructureKind` recognition, spectral dispatch
- `scirs2-stats::mle_symbolic` — gradient-descent MLE from symbolic PDF
- `scirs2-neural::symbolic::rope_attention` — closed-form RoPE attention logit

**Wave 73–77 GPU + Algorithms**:
- **scirs2-core** — NUMA-aware `par_map_chunks` (Linux pthread affinity pin, rayon fallback Darwin/WASM); `GpuNdarray<f32>` (7 WGSL kernels: elementwise, matmul, sum, transpose); `concat_axis.wgsl` (axis>0) + `reduce_sum_axis.wgsl` (rank≥3)
- **scirs2-interpolate** — Real wgpu RBF kernel-matrix + evaluation (WGSL, kernel_id uniform, OnceLock GPU probe cache)
- **scirs2-graph** — Real wgpu BFS (level-sync, atomicCompareExchange), Bellman-Ford SSSP (edge-parallel atomicMin), true delta-stepping WGSL (light/heavy phase + changed_flag convergence); CPU-parallel fallback via rayon+AtomicU32
- **scirs2-optimize** — CG GPU (GpuNdarray dot/direction-update), Newton GPU (Hessian-vector matmul), L-BFGS GPU (two-loop recursion via GpuNdarray)
- **scirs2-symbolic** — ALiBi positional bias (`alibi_slope`, `alibi_bias_expr`, `verify_symbolic_vs_numerical`); SR-as-prior for time series (`discover_series_prior`, `series_prior_regularization`); `SymbolicPriorLoss` in scirs2-neural
- **scirs2-integrate** — Pantelides index reduction (Hopcroft-Karp O(E√V) + Tarjan SCC); Wiktorsson 2001 Lévy-area for SDE strong order 1.5
- **scirs2-spatial** — 2D+3D Hilbert curve sort (24-state Butz/Hamilton for 3D)

<details>
<summary><strong>What was in 0.4.3</strong></summary>

- **36,446 Tests** across 29 workspace crates — integration test pipelines (ML/signal/NLP/vision/graph/scientific)
- **Dependency upgrades**: oxifft 0.1.4, sha2 0.11, egui/eframe 0.34 — latest COOLJAPAN ecosystem
- **scirs2-core**: Async GPU transfer, unified memory, RRB-tree, Tracy profiler, stream allocator, memory defrag, NUMA bandwidth optimization
- **scirs2-linalg**: Auto-precision solver dispatch, GPU eigensolvers, H-matrix compression (hierarchical matrices)
- **scirs2-special**: Spheroidal wave functions, Mathieu-Hill, f16 mixed-precision, GPU auto-dispatch, Clebsch-Gordan SU(2)/SU(3)/SO(5), Hall polynomials, Hecke L-functions, elliptic L-functions
- **scirs2-fft**: Ring-buffer STFT, cache-oblivious FFT, streaming FFT (out-of-core)
- **scirs2-signal**: GPU spectrograms, matched filter bank, batched Welch PSD + EFDD modal analysis
- **scirs2-sparse**: ILU(0) mixed CPU/GPU preconditioning
- **scirs2-optimize**: GDAS/SNAS/predictor-based NAS, CMA-ES optimizer, subspace embedding (JL/Gaussian/sparse, sketched LS)
- **scirs2-integrate**: GPU LBM, ODE ensemble, sparse grid quadrature, particle filter
- **scirs2-interpolate**: Physics-informed RBF, random RBF, GPU deep kriging + active learning
- **scirs2-io**: Iceberg tables, DataFusion provider, object-store abstraction, S3 multipart, GCS + Azure SAS, exactly-once semantics
- **scirs2-datasets**: ndarray generators + sharding, HuggingFace integration, dataset sharding
- **scirs2-text**: USE/SimCSE/HDP topic modeling, Unicode tokenizer, enhanced BPE + chat templates, sentence embeddings, multilingual
- **scirs2-series**: PC causality analysis, all v0.4.0 items verified
- **scirs2-neural**: NAS repair (74 tests), Mamba SSM, QAT, model tracing
- **scirs2-numpy**: DLPack protocol, masked arrays, structured dtypes, array protocol
- **scirs2-metrics**: Rotated IoU, 17 new tests
- **scirs2-python**: special/interpolate/integrate bindings
- **Zero warnings**: cargo clippy + rustdoc + fmt all clean; Apache-2.0 license compliance enforced
- **scirs2-wasm TypeScript bindings**: Full TypeScript type declarations (19 API sections, ~96 exported symbols) covering stats, signal, linear algebra, FFT, ML models, streaming ops, React hooks, Web Worker utilities, FinalizationRegistry memory management
- **scirs2-core SIMD examples**: 5 new SIMD benchmark and demo programs
- **Build infrastructure**: Vendored `bitflags` 0.6.0 patch; refined ARM NEON `pathfinder_simd` build patch

</details>

<details>
<summary><strong>What was in 0.4.0</strong></summary>

- **60+ clippy warnings fixed** — zero-warning workspace (clippy, rustdoc, compilation)
- **Flash Attention 2, QAT, ONNX export, LoRA/DoRA/GPTQ** — production neural network training and deployment
- **GPU PDE solvers, GPU FFT pipeline, GPU SpMV** — hardware-accelerated scientific computing
- **Temporal GNN (TGAT/TGN), Graph Transformers (GraphGPS/Graphormer)** — state-of-the-art graph ML
- **NeRF/instant-NGP, 3D detection, depth completion** — advanced computer vision
- **NUMA-aware scheduler, lock-free data structures** — high-performance core infrastructure
- **WebGPU/WASM backend with 76 tests** — browser-side GPU compute
- **Conformal prediction, Bayesian NNs, INLA** — uncertainty quantification and Bayesian ML
- **Cache-oblivious FFT, adaptive sparse FFT** — advanced spectral methods
- **mdBook documentation (26 EN + 12 JP pages)** — comprehensive bilingual docs
- **Distribution validation (78 tests, 15+ distributions)** — numerical accuracy verified
- **Deep Kriging, GP Surrogate, online streaming interpolation** — advanced interpolation
- **Neural Audio (Conv-TasNet/vocoder/enhancement)** — audio ML pipeline
- **Delta Lake, TileDB, Arrow Flight, Kafka** — enterprise data I/O
- **PCMCI causality, physics-informed time series** — advanced time series analysis

</details>

<details>
<summary><strong>What was in 0.3.4</strong></summary>

- **OxiARC Compression Upgrades**: Upgraded all OxiARC compression libraries (oxiarc-archive, oxiarc-lz4, oxiarc-bzip2, oxiarc-zstd, oxiarc-core, oxiarc-deflate) from 0.2.4 to 0.2.5
- **Crates.io Migration**: Migrated oxiarc-snappy and oxiarc-brotli from local path dependencies to crates.io version 0.2.5
- **Clippy Cleanup**: Fixed ~50 clippy warnings across the workspace (sort_by to sort_by_key, manual checked division, loop counters, redundant closures)
- **Dependency Cleanup**: Removed 10+ unused dependencies (ndarray-npy, x509-parser, itertools, num-rational, gmp-mpfr-sys, opentelemetry-prometheus, opentelemetry-semantic-conventions, mongodb, redis, prost) — eliminated `zip` crate from dependency tree
</details>

<details>
<summary><strong>What was in 0.3.3</strong></summary>

- **Pure Rust Compression**: Replaced C-based compression (flate2, lz4, zstd, bzip2) with oxiarc pure Rust alternatives across core, cluster, and io
- **Pure Rust Memory Profiling**: Replaced tikv-jemallocator with OS-native APIs (Mach task_info/procfs)
- **WASM Target Support**: Added getrandom WASM backend and wasm32 configuration
- **Pure Rust Directory Detection**: Replaced `dirs` crate with custom `platform_dirs` module
- **Parquet Pure Rust**: Configured parquet with pure Rust feature flags, switched Zstd to Brotli codec
</details>

<details>
<summary><strong>What was in 0.3.2</strong></summary>

- **pyo3 0.28.2 Upgrade**: Migrated Python bindings to pyo3 0.28.2 (`Python::with_gil` -> `Python::attach`)
- **#[pyclass] Deprecation Fixes**: Updated `from_py_object` attribute usage to resolve deprecation warnings
- **Benchmark Modernization**: Replaced deprecated `criterion::black_box` with `std::hint::black_box` across all benchmarks
</details>

<details>
<summary><strong>What was in 0.3.1</strong></summary>

- **Neural Networks**: Transformer architectures (GPT-2, T5, Swin), GNNs (GCN/GAT/GIN), diffusion models, capsule networks, spiking neural networks (SNN)
- **Advanced Statistics**: Gaussian process regression, survival analysis (Cox/Kaplan-Meier/Nelson-Aalen), Bayesian networks, copulas, nonparametric Bayes
- **Signal Processing**: OMP/ISTA compressed sensing, LMS/RLS adaptive filtering, MFCC/EMD, source separation (ICA/NMF)
- **Graph Algorithms**: Louvain/Girvan-Newman community detection, VF2 isomorphism, Node2Vec embeddings, network flow
- **Sparse Linear Algebra**: LOBPCG, IRAM, AMG, BCSR/ELLPACK formats, block preconditioners
- **Time Series**: TFT, N-BEATS, DeepAR, VECM, DFM, EGARCH/FIGARCH, online ARIMA
- **Optimization**: SQP, LP/QP interior point, SGD/Adam/NSGA-III, MIP/SDP/SOCP solvers
- **FFT Extensions**: Sparse FFT, Prony method, MUSIC, Lomb-Scargle, Burg method, NTT
- **Interpolation**: RBF, MLS, Floater-Hormann, spherical harmonics, kriging
- **Special Functions**: Mathieu, Coulomb wave functions, Wigner 3j/6j, Jacobi theta
- **Computer Vision**: Stereo matching, depth estimation, ICP point cloud registration, SLAM
- **Julia Bindings**: New Julia interface for seamless interoperability
</details>

See [SCIRS2_POLICY.md](SCIRS2_POLICY.md) for architectural details and [CHANGELOG.md](CHANGELOG.md) for complete release history.

## 🦀 Pure Rust by Default

**SciRS2 is 100% Pure Rust by default** - no C, C++, or Fortran dependencies required!

Unlike traditional scientific computing libraries that rely on external system libraries (OpenBLAS, LAPACK), SciRS2 provides a completely self-contained Pure Rust implementation:

- ✅ **BLAS/LAPACK**: Pure Rust [OxiBLAS](https://github.com/cool-japan/oxiblas) implementation (no OpenBLAS/MKL/Accelerate required)
- ✅ **FFT**: Pure Rust [OxiFFT](https://github.com/cool-japan/oxifft) with FFTW-comparable performance (no C libraries required)
- ✅ **Random Number Generation**: Pure Rust implementations of all statistical distributions
- ✅ **All Core Modules**: Every scientific computing module works out-of-the-box without external dependencies

**Benefits**:
- 🚀 **Easy Installation**: `cargo add scirs2` - no system library setup required
- 🔒 **Memory Safety**: Rust's ownership system prevents memory leaks and data races
- 🌍 **Cross-Platform**: Same code works on Linux, macOS, Windows, and WebAssembly
- 📦 **Reproducible Builds**: No external library version conflicts
- ⚡ **Performance**: High performance Pure Rust FFT via OxiFFT (FFTW-compatible algorithms)

**Optional Performance Enhancements** (not required for functionality):
- `oxifft` feature: High-performance Pure Rust FFT with FFTW-compatible algorithms
- `mpsgraph` feature: Apple Metal GPU acceleration (macOS only, Objective-C)
- `cuda` feature: NVIDIA CUDA GPU acceleration
- `arbitrary-precision` feature: GMP/MPFR for arbitrary precision arithmetic (C library)

Enable with: `cargo add scirs2 --features oxifft,cuda`

By default, SciRS2 provides a **fully functional, Pure Rust scientific computing stack** that rivals the performance of traditional C/Fortran-based libraries while offering superior safety, portability, and ease of use.

## Features

### Scientific Computing Core
- **Linear Algebra** (`scirs2-linalg`): Matrix operations, GMRES/PCG/BiCGStab iterative solvers, Lanczos/Arnoldi factorizations, CP-ALS/Tucker tensor decompositions, matrix functions (expm/logm/sqrtm), control theory (Riccati/Lyapunov)
- **Statistics** (`scirs2-stats`): Distributions (stable, GPD, von Mises-Fisher, Tweedie), Bayesian methods (NUTS/HMC/SMC), Gaussian processes, survival analysis (Cox/KM/AFT), copulas, Bayesian networks, causal inference
- **Optimization** (`scirs2-optimize`): MIP/SDP/SOCP solvers, Bayesian optimization, NSGA-III multi-objective, stochastic (SGD/Adam/SVRG), metaheuristics (ACO/SA/DE/Harmony), convex (ADMM/proximal), combinatorial
- **Integration** (`scirs2-integrate`): ODE/BVP/DAE solvers, PDE (FEM/LBM/DG), SDE/SPDE solvers, BEM, phase-field (Cahn-Hilliard/Allen-Cahn), port-Hamiltonian systems, QMC/IMEX methods
- **Interpolation** (`scirs2-interpolate`): RBF, MLS, PCHIP, spherical harmonics, kriging, B-spline surfaces, tensor product, natural neighbor, barycentric
- **Special Functions** (`scirs2-special`): Mathieu, Coulomb wave functions, spherical harmonics, Wigner 3j/6j/9j, Jacobi theta, Fox H-function, Heun, Appell, q-analogs, Weierstrass, polylogarithm
- **Signal Processing** (`scirs2-signal`): Matched filter, CFAR radar detection, Kalman/EKF/UKF state estimation, compressed sensing (OMP/ISTA/CoSaMP), MFCC, EMD/HHT, source separation (ICA/NMF), adaptive filtering (LMS/RLS)
- **Sparse Matrices** (`scirs2-sparse`): LOBPCG/IRAM eigensolvers, AMG, BCSR/ELLPACK formats, block preconditioners (Jacobi/SPAI/Schwarz), GCRO-DR recycled Krylov, domain decomposition
- **Spatial Algorithms** (`scirs2-spatial`): R*-Tree, Fortune's Voronoi, WGS84/UTM geodata projections, spatial statistics, trajectory analysis, sweep-line algorithms, 3D convex hull

### Advanced Modules
- **N-Dimensional Image Processing** (`scirs2-ndimage`): Gabor/SIFT/HOG feature detection, watershed/SLIC/GrabCut segmentation, optical flow (Farneback/LK), 3D morphology, medical imaging, texture analysis (GLCM/LBP)
- **Clustering** (`scirs2-cluster`): GMM, SOM, HDBSCAN, Dirichlet process, kernel k-means, biclustering (Cheng-Church/FABIA), topological (Mapper/TDA), deep clustering (DEC), stream/online (CluStream/DenStream)
- **FFT and Spectral Methods** (`scirs2-fft`): Sparse FFT, Prony method, MUSIC/ESPRIT, Lomb-Scargle, NTT, CZT/FRFT, polyphase filterbank, all DCT/DST variants, wavelet packets, reassigned spectrogram
- **I/O Utilities** (`scirs2-io`): Protobuf/msgpack/CBOR/BSON/Avro serialization, Parquet/Feather/ORC columnar formats, streaming JSON/CSV/Arrow, cloud storage abstraction, HDF5-lite, schema management, ETL pipeline
- **Sample Datasets** (`scirs2-datasets`): Text/NER/QA, medical imaging, graph benchmarks, recommendation, anomaly detection, time series (UCR-compatible), synthetic generators
- **Distribution Validation** (`scirs2-validation`): Distribution validation utilities and reference values for statistical testing, PDF/CDF/PPF testing across 15+ distributions

### AI and Machine Learning
- **Automatic Differentiation** (`scirs2-autograd`): Custom gradient rules, gradient checkpointing, JVP/VJP (forward/reverse mode), implicit differentiation, mixed precision (FP16/BF16), distributed gradient, Hessian computation
- **Neural Networks** (`scirs2-neural`): Transformers (GPT-2/T5/SWIN), GNNs (GCN/GAT/GraphSAGE/GIN), diffusion models (DDPM/DDIM), VAE, GAN, capsule networks, SNN, PPO/DPO RL, MoE, ViT/CLIP/ConvNeXt, knowledge distillation, quantization, pruning, meta-learning (MAML)
- **Graph Processing** (`scirs2-graph`): Louvain/Girvan-Newman community detection, VF2 isomorphism, Node2Vec embeddings, maximum flow (Dinic/push-relabel), temporal graphs, hypergraphs, force-directed layout, SVG visualization
- **Data Transformation** (`scirs2-transform`): UMAP, Barnes-Hut t-SNE, sparse PCA, persistent homology (TDA), optimal transport (Wasserstein/Sinkhorn), archetypal analysis, metric learning (LMNN/ITML), multiview (CCA/deep CCA)
- **Metrics** (`scirs2-metrics`): Detection (IoU/AP/mAP/NMS), ranking (NDCG/MAP/MRR), generative (FID/IS/LPIPS), fairness (demographic parity/equalized odds), segmentation, streaming metrics
- **Text Processing** (`scirs2-text`): BPE/WordPiece tokenizers, CRF/HMM sequence labeling, FastText, NER, topic modeling (LDA/NMF), coreference resolution, knowledge graph extraction, RST discourse analysis
- **Computer Vision** (`scirs2-vision`): Stereo depth estimation, ICP point cloud registration, PnP camera pose, dense optical flow, video processing, SLAM framework, panoptic/semantic/instance segmentation, 3D reconstruction (SfM)
- **Time Series** (`scirs2-series`): TFT/N-BEATS/DeepAR deep learning forecasting, VAR/VECM/DFM models, EGARCH/FIGARCH volatility, FDA (functional data analysis), conformal prediction, online ARIMA, Granger causality, hierarchical reconciliation

### Performance and Safety
- **Pure Rust by Default**: 100% Rust with no C/C++/Fortran dependencies (OxiBLAS for BLAS/LAPACK, OxiFFT for FFT)
- **Ultra-Optimized SIMD**: Ecosystem-wide vectorization achieving 10-30x performance improvements
- **Work-Stealing Scheduler**: Adaptive parallel task execution with NUMA-aware allocation
- **Multi-Backend GPU Acceleration**: CUDA, ROCm, Metal, WGPU, OpenCL support
- **Memory Efficiency**: Smart allocators, buffer pools, zero-copy operations, cache-oblivious algorithms
- **Safety**: Memory safety and thread safety through Rust's ownership model; zero `unwrap()` in production code
- **Error Handling**: Comprehensive error system with context, recovery strategies, and circuit-breaker patterns

## Project Scale

SciRS2 is a large-scale scientific computing ecosystem with comprehensive coverage:

- **📊 Total Lines**: ~4.2M lines across all files (Rust, Python, Julia, TOML, Markdown, etc.)
- **🦀 Rust Code**: ~3.89M SLoC across 8,149 files (tokei measured)
- **📝 Documentation**: Comprehensive comment lines + embedded Markdown in Rust docs
- **🧪 Testing**: 37,402 tests (all-features) / 35,632 (default-features) + 5,001 doc-tests passing — workspace-wide via `cargo nextest`
- **📦 Modules**: 29 workspace crates covering scientific computing, machine learning, and AI
- **🔌 Public API**: 80,800+ public API items across all crates
- **🏗️ Development Effort**: Estimated 89+ months with 136 developers (COCOMO model)
- **💰 Estimated Value**: $137M+ development cost equivalent (COCOMO model)

This demonstrates the comprehensive nature and production-ready maturity of the SciRS2 ecosystem.

## Project Goals

- Create a comprehensive scientific computing and machine learning library in Rust
- **Provide a Pure Rust implementation by default** - eliminating external C/Fortran dependencies for easier installation and better portability
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
├── scirs2-validation/        # Distribution validation utilities and reference values

# AI/ML Modules
├── scirs2-neural/            # Neural network building blocks
# Note: scirs2-optim separated into independent OptiRS project
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

Each module has its own README with detailed documentation and is available on crates.io.

### Complete Crate Reference (v0.6.1)

| Crate | Description | docs.rs |
|-------|-------------|---------|
| [**scirs2**](scirs2/README.md) | Main integration crate — re-exports from all subcrates | [![docs.rs](https://img.shields.io/docsrs/scirs2)](https://docs.rs/scirs2) |
| [**scirs2-core**](scirs2-core/README.md) | Foundational infrastructure: work-stealing scheduler, NUMA `par_map_chunks`, HAMT, cache-oblivious algorithms, GPU backends, `GpuNdarray<f32>` (wgpu), distributed ops | [![docs.rs](https://img.shields.io/docsrs/scirs2-core)](https://docs.rs/scirs2-core) |
| [**scirs2-linalg**](scirs2-linalg/README.md) | Linear algebra: iterative solvers (GMRES/PCG/BiCGStab), tensor decompositions (CP-ALS/Tucker), matrix functions (expm/logm), control theory | [![docs.rs](https://img.shields.io/docsrs/scirs2-linalg)](https://docs.rs/scirs2-linalg) |
| [**scirs2-stats**](scirs2-stats/README.md) | Statistics: 40+ distributions, NUTS/HMC/SMC Bayesian inference, Gaussian processes, survival analysis (Cox/KM/AFT), Bayesian networks, copulas | [![docs.rs](https://img.shields.io/docsrs/scirs2-stats)](https://docs.rs/scirs2-stats) |
| [**scirs2-optimize**](scirs2-optimize/README.md) | Optimization: MIP/SDP/SOCP, Bayesian BO, NSGA-III multi-objective, stochastic (SGD/Adam/SVRG), ACO/SA/DE metaheuristics, ADMM/proximal, GPU CG/Newton/L-BFGS (wgpu) | [![docs.rs](https://img.shields.io/docsrs/scirs2-optimize)](https://docs.rs/scirs2-optimize) |
| [**scirs2-integrate**](scirs2-integrate/README.md) | Numerical integration: ODE/BVP/DAE, PDE (FEM/LBM/DG), SDE/SPDE, BEM, phase-field, port-Hamiltonian, IGA, QMC | [![docs.rs](https://img.shields.io/docsrs/scirs2-integrate)](https://docs.rs/scirs2-integrate) |
| [**scirs2-interpolate**](scirs2-interpolate/README.md) | Interpolation: RBF (with wgpu GPU acceleration), PCHIP, MLS, kriging, spherical harmonics, B-spline surfaces, tensor product, natural neighbor, barycentric | [![docs.rs](https://img.shields.io/docsrs/scirs2-interpolate)](https://docs.rs/scirs2-interpolate) |
| [**scirs2-fft**](scirs2-fft/README.md) | FFT and spectral: sparse FFT, Prony, MUSIC, Lomb-Scargle, NTT, CZT, FRFT, DCT/DST all variants, wavelet packets, polyphase filterbank | [![docs.rs](https://img.shields.io/docsrs/scirs2-fft)](https://docs.rs/scirs2-fft) |
| [**scirs2-signal**](scirs2-signal/README.md) | Signal processing: matched filter, CFAR radar, Kalman/EKF/UKF, OMP/ISTA compressed sensing, MFCC, EMD/HHT, ICA/NMF source separation | [![docs.rs](https://img.shields.io/docsrs/scirs2-signal)](https://docs.rs/scirs2-signal) |
| [**scirs2-sparse**](scirs2-sparse/README.md) | Sparse matrices: LOBPCG/IRAM eigensolvers, AMG, BCSR/ELLPACK formats, block preconditioners (Jacobi/SPAI/Schwarz), recycled Krylov (GCRO-DR) | [![docs.rs](https://img.shields.io/docsrs/scirs2-sparse)](https://docs.rs/scirs2-sparse) |
| [**scirs2-special**](scirs2-special/README.md) | Special functions: Mathieu, Coulomb wave, spherical harmonics, Wigner 3j/6j/9j, Jacobi theta, Fox H-function, Heun, Appell, q-analogs | [![docs.rs](https://img.shields.io/docsrs/scirs2-special)](https://docs.rs/scirs2-special) |
| [**scirs2-spatial**](scirs2-spatial/README.md) | Spatial: R*-Tree, Fortune's Voronoi, WGS84/UTM geodata, spatial statistics, trajectory analysis, sweep-line algorithms, 3D convex hull | [![docs.rs](https://img.shields.io/docsrs/scirs2-spatial)](https://docs.rs/scirs2-spatial) |
| [**scirs2-cluster**](scirs2-cluster/README.md) | Clustering: GMM, SOM, HDBSCAN, Dirichlet process, kernel k-means, biclustering (FABIA), topological (Mapper/TDA), deep clustering (DEC) | [![docs.rs](https://img.shields.io/docsrs/scirs2-cluster)](https://docs.rs/scirs2-cluster) |
| [**scirs2-ndimage**](scirs2-ndimage/README.md) | N-dim image processing: Gabor/SIFT/HOG, watershed/SLIC/GrabCut, optical flow, 3D morphology, medical imaging, GLCM/LBP texture | [![docs.rs](https://img.shields.io/docsrs/scirs2-ndimage)](https://docs.rs/scirs2-ndimage) |
| [**scirs2-io**](scirs2-io/README.md) | Data I/O: Protobuf/msgpack/CBOR/BSON/Avro, Parquet/Feather/ORC, streaming JSON/CSV/Arrow, cloud storage abstraction, HDF5-lite, ETL pipeline | [![docs.rs](https://img.shields.io/docsrs/scirs2-io)](https://docs.rs/scirs2-io) |
| [**scirs2-datasets**](scirs2-datasets/README.md) | Datasets: text/NER/QA, medical imaging, graph benchmarks, recommendation, anomaly detection, time series (UCR-compatible), synthetic generators | [![docs.rs](https://img.shields.io/docsrs/scirs2-datasets)](https://docs.rs/scirs2-datasets) |
| [**scirs2-autograd**](scirs2-autograd/README.md) | Automatic differentiation: JVP/VJP, custom gradients, checkpointing, mixed precision (FP16/BF16), distributed gradient, Hessian, tape-based AD | [![docs.rs](https://img.shields.io/docsrs/scirs2-autograd)](https://docs.rs/scirs2-autograd) |
| `scirs2-symbolic` | Symbolic mathematics: EML-IR CAS, symbolic diff/hessian/jacobian, canonicalize, e-graphs, SMT (OxiZ), Cranelift JIT, GPU WGSL eval, diffgeom (Riemann/Weyl), ALiBi, SR engine, LaTeX export | — |
| [**scirs2-neural**](scirs2-neural/README.md) | Neural networks: GPT-2/T5/SWIN/ViT/CLIP/ConvNeXt transformers, GCN/GAT/GIN GNNs, DDPM diffusion models, SNN, capsule, PPO/DPO RL, MoE | [![docs.rs](https://img.shields.io/docsrs/scirs2-neural)](https://docs.rs/scirs2-neural) |
| [**scirs2-graph**](scirs2-graph/README.md) | Graph algorithms: Louvain/Leiden/Girvan-Newman community detection, VF2 isomorphism, Node2Vec, Dinic max-flow, temporal graphs, SVG visualization; GPU BFS/SSSP/delta-stepping (wgpu) | [![docs.rs](https://img.shields.io/docsrs/scirs2-graph)](https://docs.rs/scirs2-graph) |
| [**scirs2-transform**](scirs2-transform/README.md) | Dimensionality reduction: UMAP, Barnes-Hut t-SNE, sparse PCA, persistent homology (TDA), optimal transport (Wasserstein/Sinkhorn), metric learning | [![docs.rs](https://img.shields.io/docsrs/scirs2-transform)](https://docs.rs/scirs2-transform) |
| [**scirs2-metrics**](scirs2-metrics/README.md) | ML metrics: IoU/AP/mAP detection, NDCG/MAP/MRR ranking, FID/IS/LPIPS generative, fairness (equalized odds), segmentation, streaming metrics | [![docs.rs](https://img.shields.io/docsrs/scirs2-metrics)](https://docs.rs/scirs2-metrics) |
| [**scirs2-text**](scirs2-text/README.md) | NLP: BPE/WordPiece tokenizers, CRF/HMM sequence labeling, FastText, NER, LDA topic modeling, coreference resolution, RST discourse analysis | [![docs.rs](https://img.shields.io/docsrs/scirs2-text)](https://docs.rs/scirs2-text) |
| [**scirs2-vision**](scirs2-vision/README.md) | Computer vision: stereo depth, ICP point cloud, PnP camera pose, dense optical flow, SLAM, panoptic/semantic/instance segmentation, SfM | [![docs.rs](https://img.shields.io/docsrs/scirs2-vision)](https://docs.rs/scirs2-vision) |
| [**scirs2-series**](scirs2-series/README.md) | Time series: TFT/N-BEATS/DeepAR forecasting, VAR/VECM/DFM, EGARCH/FIGARCH volatility, FDA, conformal prediction, online ARIMA, Granger causality | [![docs.rs](https://img.shields.io/docsrs/scirs2-series)](https://docs.rs/scirs2-series) |
| [**scirs2-wasm**](scirs2-wasm/README.md) | WebAssembly bindings: WasmMatrix JS/TS API, TypeScript type definitions, WASM SIMD (128-bit), Web Worker parallel computation, streaming | [![docs.rs](https://img.shields.io/docsrs/scirs2-wasm)](https://docs.rs/scirs2-wasm) |
| [**scirs2-python**](scirs2-python/README.md) | Python bindings via PyO3: 15+ modules including linalg, stats, neural, autograd with NumPy interoperability (optional, feature-gated) | — |

Note: `scirs2-optim` has been separated into the independent [OptiRS](https://github.com/cool-japan/optirs) project.

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

SciRS2 follows the COOLJAPAN Pure Rust Policy. All default dependencies are 100% Pure Rust.

### Core Pure Rust Dependencies
- `oxiblas`: Pure Rust BLAS/LAPACK implementation (no C/Fortran, no OpenBLAS/MKL required)
- `oxifft`: Pure Rust FFT with FFTW-comparable performance (no FFTW/CLFFT C library required)
- `oxiarc-archive`/`oxiarc-*`: Pure Rust archive/compression (replaces zip/zlib C bindings)
- `oxicode`: Pure Rust serialization (replaces bincode)
- `ndarray`: Multidimensional array operations (via `scirs2-core` abstraction)
- `num`: Numeric abstractions
- `rayon`: Data-parallel processing

### Infrastructure Dependencies
- `serde`/`serde_json`: Serialization/deserialization
- `thiserror`/`anyhow`: Error handling
- `tokio`: Async runtime (for async IO utilities)
- `petgraph`: Graph data structures
- `image`: Image encoding/decoding utilities

### Optional Feature-Gated C Dependencies (not enabled by default)
- `cuda` feature: NVIDIA CUDA GPU acceleration
- `mpsgraph` feature: Apple Metal GPU acceleration (macOS only)
- `arbitrary-precision` feature: GMP/MPFR arbitrary precision arithmetic

## Recent Development History

### v0.3.1 (Released March 9, 2026) - Massive Ecosystem Expansion

**Major Feature Release**
- 🚀 **29 Workspace Crates**: Comprehensive modular ecosystem for scientific computing and AI
- 🚀 **27,632 Tests**: Full test suite with comprehensive coverage
- 🚀 **Advanced Neural Networks**: Transformers, GNNs, diffusion models, SNN, capsule networks
- 🚀 **Statistics & Probabilistic ML**: Gaussian processes, Bayesian networks, survival analysis, copulas
- 🚀 **Graph Algorithms**: Community detection, GNN embeddings, isomorphism, flow algorithms
- 🚀 **Signal Processing**: Compressed sensing, adaptive filtering, source separation, synchrosqueezing
- 🚀 **Optimization**: SQP, MIP, SDP, SOCP, Bayesian optimization, metaheuristics (ACO/SA/DE)
- 🚀 **Time Series**: TFT, N-BEATS, DeepAR, VECM, DFM, EGARCH/FIGARCH models
- 🚀 **Julia Bindings**: New Julia interface for ecosystem interoperability
- 🚀 **FFT Extensions**: Sparse FFT, Prony, MUSIC, Lomb-Scargle, Burg, NTT
- 🚀 **Sparse Linear Algebra**: LOBPCG, IRAM, AMG, BCSR/ELLPACK, block preconditioners

### v0.2.0 (Released February 12, 2026) - SIMD Expansion & Spatial Enhancement

**Major Feature Release**
- 🚀 **SIMD Phase 60-69**: 8 new advanced SIMD operation modules (beta functions, interpolation, geometry, probability, array ops)
- 🚀 **Spatial Algorithms**: Complete Delaunay triangulation refactoring with modular Bowyer-Watson 2D/3D/ND implementation
- 🚀 **FFT Enhancements**: Advanced coordinator architecture for complex FFT pipelines
- 🚀 **Special Functions**: Interactive learning modules and advanced derivation studio
- 🐛 **Fixed**: Optimizer::update() now correctly updates variables (Issue #100)
- 🐛 **Fixed**: Eliminated "Index out of bounds in ComputeContext::input" warning spam
- ✅ **Enhanced**: Python bindings expanded to 11 additional modules
- ✅ **Enhanced**: PCHIP interpolation with linear extrapolation
- ✅ **Improved**: Build system for better manylinux compatibility

### v0.1.3 (Released January 25, 2026) - Maintenance & Enhancement

**Interpolation & Python Bindings**
- ✅ **Added**: Python bindings for autograd, datasets, graph, io, metrics, ndimage, neural, sparse, text, transform, vision modules
- ✅ **Enhanced**: PCHIP extrapolation improvements with configurable modes
- ✅ **Fixed**: Adam optimizer scalar/1×1 parameter handling (Issue #98)
- ✅ **Improved**: PyO3 configuration for cross-platform builds

### v0.1.2 (Released January 15, 2026) - Performance & Pure Rust Enhancement

**FFT Migration & SIMD Performance**
- ✅ **Migration**: Complete switch to Pure Rust OxiFFT (no C dependencies)
- ✅ **Performance**: Zero-allocation SIMD operations with in-place computation
- ✅ **ML Infrastructure**: Production-ready functional optimizers and training loops
- ✅ **Code Quality**: All clippy warnings resolved, enhanced API compatibility

## Installation and Usage

### System Dependencies

**v0.6.1 uses Pure Rust dependencies only - No system libraries required!** 🎉

SciRS2 is **100% Pure Rust** with OxiBLAS (Pure Rust BLAS/LAPACK implementation). You don't need to install:
- ❌ OpenBLAS
- ❌ Intel MKL
- ❌ Apple Accelerate Framework bindings
- ❌ LAPACK
- ❌ Any C/Fortran compilers

**Just install Rust and build:**
```bash
# That's it! No system dependencies needed.
cargo build --release
```

### Cargo Installation

SciRS2 and all its modules are available on [crates.io](https://crates.io/crates/scirs2). You can add them to your project using Cargo:

```toml
# Add the main integration crate for all functionality
[dependencies]
scirs2 = "0.6.1"
```

Or include only the specific modules you need:

```toml
[dependencies]
# Core utilities
scirs2-core = "0.6.1"

# Scientific computing modules
scirs2-linalg = "0.6.1"
scirs2-stats = "0.6.1"
scirs2-optimize = "0.6.1"

# AI/ML modules
scirs2-neural = "0.6.1"
scirs2-autograd = "0.6.1"
# Note: For ML optimization algorithms, use the independent OptiRS project
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

## Platform Compatibility

SciRS2 v0.6.1 (July 15, 2026) has been tested on the following platforms:

### ✅ Fully Supported Platforms

| Platform | Architecture | Test Status | Notes |
|----------|-------------|-------------|-------|
| **macOS** | Apple M3 (ARM64) | ✅ All tests passing (37,402 tests, all-features) | macOS 15.6.1, 24GB RAM |
| **Linux** | x86_64 | ✅ All tests passing (37,402 tests, all-features) | With required dependencies |
| **Linux + CUDA** | x86_64 + NVIDIA GPU | ✅ All tests passing (37,402 tests, all-features) | CUDA support enabled |

### ⚠️ Partially Supported Platforms

| Platform | Architecture | Test Status | Notes |
|----------|-------------|-------------|-------|
| **Windows** | x86_64 | ⚠️ Build succeeds, some tests fail | Windows 11 Pro - see known issues below |

### Platform-Specific Requirements

#### macOS / Linux
To run the full test suite with all features:
```bash
# No system dependencies required - Pure Rust!
cargo nextest run --nff --all-features  # 37,402 tests
```

#### Windows
```bash
# Build works successfully
cargo build

# Note: Some crates have test failures on Windows
# Full test compatibility is planned for v0.5.x
cargo test  # Some tests may fail
```

### Running Tests

**Recommended test runner**: Use `cargo nextest` instead of `cargo test` for better performance and output:

```bash
# Install nextest
cargo install cargo-nextest

# Run all tests
cargo nextest run --nff --all-features
```

## Current Status (v0.6.1 - Released July 15, 2026)

### 🎉 Production-Ready Features

#### Pure Rust Scientific Computing Stack
- **100% Pure Rust by Default**: No C/C++/Fortran dependencies required (OxiBLAS for BLAS/LAPACK, OxiFFT for FFT)
- **Zero System Dependencies**: Works out-of-the-box with just `cargo build`
- **Cross-Platform**: Identical behavior on Linux, macOS, Windows, and WebAssembly
- **Memory Safety**: Rust's ownership system prevents memory leaks and data races

#### High-Performance Computing
- **Ultra-Optimized SIMD**: 10-100x performance improvements through bandwidth-saturated operations
  - **SIMD Phase 60-69 **: Advanced operations including beta functions, interpolation kernels, geometric operations, probability distributions, and array operations
  - 14.17x speedup for element-wise operations (AVX2/NEON)
  - 15-25x speedup for signal convolution
  - 20-30x speedup for bootstrap sampling
  - TLB-optimized algorithms with cache-line aware processing
- **Multi-Backend GPU Acceleration**: CUDA, ROCm, Metal, WGPU, OpenCL support
- **Advanced Parallel Processing**: Work-stealing scheduler, NUMA-aware allocation, tree reduction algorithms
- **Memory Efficiency**: Smart allocators, buffer pools, zero-copy operations, memory-mapped arrays

#### Comprehensive Module Coverage
- **Core Scientific Computing**: Linear algebra, statistics, optimization, integration, interpolation, FFT, signal processing
- **Advanced Algorithms**:
  - Sparse matrices (CSR, CSC, COO, BSR, DIA, DOK, LIL formats)
  - **Spatial algorithms**: Enhanced modular Delaunay triangulation (2D/3D/ND), constrained triangulation, KD-trees, convex hull, Voronoi diagrams
  - Clustering (K-means, hierarchical, DBSCAN)
- **AI/ML Infrastructure**: Automatic differentiation (with fixed optimizers), neural networks, graph processing, computer vision, time series
- **Data I/O**: MATLAB, HDF5, NetCDF, Parquet, Arrow, CSV, image formats
- **Production Quality**: 37,402 tests (all-features) / 35,632 (default-features) + 5,001 doc-tests, zero warnings policy, comprehensive error handling

#### New in v0.4.0
- ✨ **Massive Feature Expansion**: 39 waves of development adding 200+ major features
- ✨ **Neural Networks**: Flash Attention 2, QAT, ONNX export, LoRA/DoRA/GPTQ, KAN networks
- ✨ **GPU Computing**: PDE solvers, FFT pipeline, SpMV, tiled GEMM, adaptive dispatch
- ✨ **Graph ML**: Temporal GNN (TGAT/TGN), GraphGPS, Graphormer, E(n)-GNN, SPONGE
- ✨ **Computer Vision**: NeRF/instant-NGP, 3D detection, depth completion, video object segmentation
- ✨ **Core Infrastructure**: NUMA-aware scheduler, lock-free skiplist/B-tree, WebGPU/WASM backend
- ✨ **Statistics**: Conformal prediction (CQR/RAPS/Mondrian), Bayesian NNs, INLA, ADVI/Laplace/SWAG
- ✨ **Zero Warnings**: 60+ clippy warnings fixed, 0 errors, 0 warnings, 0 rustdoc warnings

### Stable Modules (Production Ready — v0.6.1)

All 29 workspace crates are production-ready with comprehensive test coverage (37,402 tests all-features, 35,632 default-features, + 5,001 doc-tests).

#### Core Scientific Computing Modules
- **Linear Algebra** (`scirs2-linalg`): Full decompositions, iterative solvers (GMRES/PCG/BiCGStab/MINRES), tensor decompositions, matrix functions, control theory
- **Statistics** (`scirs2-stats`): 40+ distributions, Bayesian inference (NUTS/HMC/SMC), Gaussian processes, survival analysis, Bayesian networks, copulas, causal inference
- **Optimization** (`scirs2-optimize`): MIP/SDP/SOCP, Bayesian optimization, NSGA-III, stochastic (SGD/Adam/SVRG), metaheuristics, convex (ADMM/proximal), combinatorial
- **Integration** (`scirs2-integrate`): ODE/PDE/SDE/SPDE solvers, LBM, DG, phase-field, BEM, port-Hamiltonian, IGA, QMC
- **Interpolation** (`scirs2-interpolate`): RBF, PCHIP, MLS, kriging, spherical harmonics, B-spline surfaces, tensor product, natural neighbor
- **Signal Processing** (`scirs2-signal`): CFAR radar, Kalman/EKF/UKF, compressed sensing, MFCC, EMD/HHT, source separation, adaptive filtering, system identification
- **FFT** (`scirs2-fft`): Standard/sparse/fractional FFT, NTT, Lomb-Scargle, MUSIC, Prony, DCT/DST all variants, wavelet packets
- **Sparse Matrices** (`scirs2-sparse`): LOBPCG/IRAM eigensolvers, AMG, BCSR/ELLPACK/DIA/SELL-C-sigma, recycled Krylov, domain decomposition
- **Special Functions** (`scirs2-special`): Mathieu, Coulomb, spherical harmonics, Wigner 3j/6j/9j, Jacobi theta, Fox H-function, Heun, Appell, q-analogs, Weierstrass
- **Spatial Algorithms** (`scirs2-spatial`): R*-Tree, Fortune's Voronoi, geodata projections, trajectory analysis, spatial statistics, 3D convex hull
- **Clustering** (`scirs2-cluster`): GMM, SOM, HDBSCAN, Dirichlet process, biclustering, topological (Mapper), deep clustering, stream/online
- **Data Transformation** (`scirs2-transform`): UMAP, Barnes-Hut t-SNE, sparse PCA, persistent homology, optimal transport, metric learning, multiview learning
- **Evaluation Metrics** (`scirs2-metrics`): IoU/AP/mAP detection, NDCG ranking, FID/IS generative, fairness, segmentation, streaming metrics

#### Advanced Modules
- **N-dimensional Image Processing** (`scirs2-ndimage`): Gabor/SIFT/HOG, watershed/SLIC segmentation, optical flow, 3D morphology, medical imaging, texture analysis
- **I/O Utilities** (`scirs2-io`): Protobuf/msgpack/CBOR/BSON/Avro, Parquet/Feather/ORC, streaming JSON/CSV/Arrow, cloud storage, HDF5-lite, ETL pipeline
- **Datasets** (`scirs2-datasets`): Text/NER/QA, medical imaging, graph benchmarks, recommendation, anomaly, time series (UCR-compatible), synthetic generators

#### AI/ML Modules
- **Automatic Differentiation** (`scirs2-autograd`): JVP/VJP, custom gradients, checkpointing, mixed precision, distributed gradient, Hessian
- **Neural Networks** (`scirs2-neural`): Transformers (GPT-2/T5/SWIN/ViT/CLIP/ConvNeXt), GNNs, diffusion models, SNN, capsule networks, PPO/DPO, MoE, knowledge distillation, quantization
- **Graph Processing** (`scirs2-graph`): Louvain/Leiden community detection, VF2 isomorphism, Node2Vec, max-flow (Dinic), temporal graphs, hypergraphs, SVG visualization
- **Text Processing** (`scirs2-text`): BPE/WordPiece tokenizers, CRF/HMM labeling, FastText, NER, topic modeling (LDA), coreference, knowledge graph extraction
- **Computer Vision** (`scirs2-vision`): Stereo depth, ICP, PnP, dense optical flow, SLAM, panoptic/semantic/instance segmentation, SfM reconstruction
- **Time Series Analysis** (`scirs2-series`): TFT/N-BEATS/DeepAR forecasting, VAR/VECM/DFM, EGARCH/FIGARCH, FDA, conformal prediction, online ARIMA
- **WebAssembly** (`scirs2-wasm`): WasmMatrix operations, TypeScript type bindings, WASM SIMD, Web Worker parallel computation

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
- **Parallel Processing**: Enhanced work-stealing scheduler, custom partitioning strategies, and nested parallelism
- **Arbitrary Precision**: Complete arbitrary precision arithmetic with GMP/MPFR backend
- **Numerical Stability**: Comprehensive algorithms for stable computation including Kahan summation and log-sum-exp

### Installation

All SciRS2 modules are available on crates.io. Add the modules you need to your `Cargo.toml`:

```toml
[dependencies]
scirs2 = "0.6.1"  # Core library with all modules
# Or individual modules:
scirs2-linalg = "0.6.1"  # Linear algebra
scirs2-stats = "0.6.1"   # Statistics
# ... and more
```

For development roadmap and contribution guidelines, see [TODO.md](TODO.md) and [CONTRIBUTING.md](CONTRIBUTING.md).

## Performance Characteristics

SciRS2 prioritizes performance through several strategies:

- **Ultra-Optimized SIMD**: Advanced vectorization achieving up to 14.17x faster than scalar operations through cache-line aware processing, software pipelining, and TLB optimization
- **Multi-Backend GPU Acceleration**: Hardware acceleration across CUDA, ROCm, Metal, WGPU, and OpenCL for compute-intensive operations
- **Advanced Memory Management**: Smart allocators, bandwidth optimization, and NUMA-aware allocation strategies for large datasets
- **Work-Stealing Parallelism**: Advanced parallel algorithms with load balancing and NUMA topology awareness
- **Cache-Optimized Algorithms**: Data structures and algorithms designed for modern CPU cache hierarchies
- **Zero-cost Abstractions**: Rust's compiler optimizations eliminate runtime overhead while maintaining safety

Performance benchmarks on core operations demonstrate significant improvements:

| Operation Category | Operation | SciRS2 | Baseline | Speedup |
|-------------------|-----------|---------|-----------|---------|
| **SIMD Operations** | Element-wise (1M elements) | 0.71 ms | 10.05 ms | **14.17×** |
| **Signal Processing** | Convolution (bandwidth-saturated) | 2.1 ms | 52.5 ms | **25.0×** |
| **Statistics** | Statistical Moments | 1.8 ms | 45.3 ms | **25.2×** |
| **Monte Carlo** | Bootstrap Sampling | 8.9 ms | 267.0 ms | **30.0×** |
| **Quasi-Random** | Sobol Sequence Generation | 3.2 ms | 48.7 ms | **15.2×** |
| **FFT** | Fractional Fourier Transform | 4.5 ms | 112.3 ms | **24.9×** |
| **Linear Algebra** | Matrix Multiply (1000×1000) | 18.5 ms | 23.2 ms | 1.25× |
| **Decomposition** | SVD (500×500) | 112.3 ms | 128.7 ms | 1.15× |
| **FFT** | Standard FFT (1M points) | 8.7 ms | 11.5 ms | 1.32× |
| **Random** | Normal Distribution (10M samples) | 42.1 ms | 67.9 ms | 1.61× |
| **Clustering** | K-means (100K points, 5 clusters) | 321.5 ms | 378.2 ms | 1.18× |

**Key Takeaways**:
- 🚀 Ultra-optimized SIMD operations achieve **10-30x speedups**
- ⚡ Traditional operations match or exceed NumPy/SciPy performance
- 🎯 Pure Rust implementation with no runtime overhead
- 📊 Benchmarks run on Apple M3 (ARM64) with 24GB RAM

*Performance may vary based on hardware, compiler optimization, and workload characteristics.*

## Core Module Usage Policy

Following the [SciRS2 Ecosystem Policy](SCIRS2_POLICY.md), all SciRS2 modules now follow a strict layered architecture:

- **Only `scirs2-core` uses external dependencies directly**
- **All other modules must use SciRS2-Core abstractions**
- **Benefits**: Consistent APIs, centralized version control, type safety, maintainability

### Required Usage Patterns

```rust
// ❌ FORBIDDEN in non-core crates
use rand::*;
use ndarray::Array2;
use num_complex::Complex;

// ✅ REQUIRED in non-core crates
use scirs2_core::random::*;
use scirs2_core::array::*;
use scirs2_core::complex::*;
```

This policy ensures ecosystem consistency and enables better optimization across the entire SciRS2 framework.

## Development Roadmap

For detailed development plans, upcoming features, and contribution opportunities, see:
- [TODO.md](TODO.md) - Development roadmap and task tracking
- [CHANGELOG.md](CHANGELOG.md) - Complete version history and detailed release notes
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines and development workflow
- [SCIRS2_POLICY.md](SCIRS2_POLICY.md) - Architectural policies and best practices

## Development Branch Status

**Current Branch**: `0.6.1` (July 15, 2026)

**Release Status**: All major features through v0.6.1 have been implemented and tested (Waves 53–78, plus the 0.6.0 CUDA-decentralization and 0.6.1 hardening cycles):
- ✅ 29 workspace crates fully implemented
- ✅ 78 waves of development completed
- ✅ Flash Attention 2, QAT, ONNX export, LoRA/DoRA/GPTQ in neural
- ✅ GPU PDE solvers, GPU FFT pipeline, GPU SpMV, real wgpu dispatch (BFS/SSSP/delta-stepping/RBF/CG/Newton/L-BFGS)
- ✅ Temporal GNN, Graph Transformers, NeRF/instant-NGP
- ✅ NUMA-aware `par_map_chunks`, lock-free data structures, GpuNdarray<f32>
- ✅ WebGPU/WASM backend, conformal prediction, Bayesian NNs
- ✅ Complete EML-IR CAS: canonicalize, e-graphs, SMT (OxiZ), JIT, GPU eval, diffgeom, ALiBi
- ✅ Pure-Rust `oxicuda-*` CUDA backend across 10 crates (0.6.0), real model-serving codegen + DLPack safety fixes (0.6.1)
- ✅ 37,402 tests (all-features) / 35,632 (default-features) + 5,001 doc-tests passing
- ✅ Zero warnings policy maintained (clippy, rustdoc, compilation)
- ✅ 80,800+ public API items documented

**Next Steps**:
- Ready for git commit and version tagging
- mdBook documentation (26 EN + 12 JP pages) completed
- Preparing for crates.io publication

## Known Limitations

### Python Bindings

**Status**: ✅ **Functional** - scirs2-python provides Python integration via PyO3

- Python bindings available for 15+ modules (core, linalg, stats, autograd, neural, etc.)
- scirs2-numpy compatibility layer handles ndarray 0.17+ integration
- Python features are **optional** and disabled by default
- Enable with: `cargo build --features python` (requires PyO3 setup)

### Platform Support

#### Fully Supported Platforms
- ✅ **Linux (x86_64)**: Full support with CUDA acceleration available
- ✅ **macOS (Apple Silicon / Intel)**: Full support with Metal acceleration
- ✅ **Windows (x86_64)**: Full support with Pure Rust OxiBLAS

All platforms benefit from:
- Pure Rust BLAS/LAPACK (OxiBLAS) - no system library installation required
- Pure Rust FFT (OxiFFT) - FFTW-comparable performance without C dependencies
- Zero-allocation SIMD operations for high performance
- Comprehensive test coverage (37,402 tests all-features, 35,632 default-features, + 5,001 doc-tests passing)

### Module-Specific Notes

#### scirs2-autograd
- ✅ **Fixed in v0.2.0**: Optimizer::update() now correctly updates variables
- ✅ **Fixed in v0.2.0**: Eliminated warning spam during gradient computation
- ✅ **Enhanced in v0.4.0**: Custom gradient, checkpointing, FD/Richardson differentiation, JVP/VJP, implicit diff

#### scirs2-spatial
- ✅ **New in v0.2.0**: Enhanced Delaunay triangulation with modular Bowyer-Watson architecture (2D/3D/ND)
- ✅ **New in v0.2.0**: Constrained Delaunay triangulation support
- ✅ **New in v0.4.0**: R*-Tree, geodata handling, Voronoi Fortune algorithm, trajectory analysis
- ✅ **Stable**: KD-trees, distance calculations, convex hull, Voronoi diagrams

#### scirs2-optimize / scirs2-stats / scirs2-special
- 🚧 **Active Development**: These modules have ongoing compilation fixes and enhancements
- ℹ️ Some features may be incomplete or in testing phase

### Delivered in v0.4.0 (Previously Planned)
All items from the v0.4.0 roadmap have been implemented:
- ✅ Flash Attention v2 and quantization-aware training (INT4/INT8) in scirs2-neural
- ✅ GPU-accelerated matrix operations (tiled GEMM, batched, adaptive dispatch) in scirs2-linalg
- ✅ Variational inference (ADVI/Laplace/SWAG) and causal inference in scirs2-stats
- ✅ GPU-accelerated PDE solvers (FDM+FEM) and adaptive mesh refinement in scirs2-integrate
- ✅ WebGPU/WASM backend with 76 tests for browser-side GPU compute
- ✅ Temporal GNN (TGAT/TGN) and graph transformers (GraphGPS/Graphormer) in scirs2-graph
- ✅ Distributed optimization (ADMM/PDMM/EXTRA) and hardware-aware NAS (DARTS) in scirs2-optimize
- ✅ Conformal prediction (CQR/RAPS/Mondrian/ACI) and multivariate deep learning in scirs2-series
- ✅ ONNX export support for neural network models
- ✅ mdBook documentation website (26 EN + 12 JP pages)
- ✅ Python PyPI wheel distribution via maturin

### Future Enhancements (v0.5.0 Roadmap)
Planned for upcoming releases:
- Extended hardware support: ARM NEON optimization, RISC-V support, embedded systems
- Cloud-native features: container optimization, serverless support, distributed computing
- Domain extensions: quantitative finance, bioinformatics, computational physics
- Enhanced ecosystem integration: improved Python/Julia interoperability, R bindings
- Automated optimization: hardware-aware algorithm selection, auto-tuning frameworks

See [TODO.md](TODO.md) for the complete development roadmap.

### Performance Tests
- Benchmark and performance tests are excluded from regular CI runs (404 tests marked as ignored) to optimize build times. Run with `cargo test -- --ignored` to execute full test suite including benchmarks.

### Hardware-Dependent Features
- GPU acceleration features require compatible hardware and drivers
- Tests automatically fall back to CPU implementations when GPU is unavailable
- Specialized hardware support (FPGA, ASIC) uses mock implementations when hardware is not present

### Test Coverage
- Total tests: 37,402 passing across all modules with `--all-features` (35,632 with default features; plus 5,001 doc-tests)
- Regular CI tests: All passing ✅
- Performance tests: Included in full test suite (run with `--all-features`)

For the most up-to-date information on limitations and ongoing development, please check our [GitHub Issues](https://github.com/cool-japan/scirs/issues).

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

## Sponsorship

SciRS2 is developed and maintained by **COOLJAPAN OU (Team Kitasan)**.

The COOLJAPAN Ecosystem represents one of the largest Pure Rust scientific computing efforts in existence — spanning 40+ projects, 500+ crates, and millions of lines of Rust code across scientific computing, machine learning, quantum computing, geospatial analysis, legal technology, multimedia processing, and more. Every line is written and maintained by a small dedicated team committed to a C/Fortran-free future for scientific software.

If you find SciRS2 or any COOLJAPAN project useful, please consider sponsoring to support continued development.

[![Sponsor](https://img.shields.io/badge/Sponsor-%E2%9D%A4-red?logo=github)](https://github.com/sponsors/cool-japan)

**[https://github.com/sponsors/cool-japan](https://github.com/sponsors/cool-japan)**

Your sponsorship helps us:
- Maintain and expand the COOLJAPAN ecosystem (40+ projects, 500+ crates)
- Keep the entire stack 100% Pure Rust — no C/Fortran/system library dependencies
- Develop production-grade alternatives to OpenCV, FFmpeg, SciPy, NumPy, scikit-learn, PyTorch, TensorFlow, GDAL, and more
- Provide long-term support, security updates, and documentation
- Fund research into novel Rust-native algorithms and optimizations

## License

Licensed under the [Apache License Version 2.0](LICENSE).

## Acknowledgments

SciRS2 builds on the shoulders of giants:
- The SciPy and NumPy communities for their pioneering work
- The Rust ecosystem and its contributors
- The numerous mathematical and scientific libraries that inspired this project

## 🌐 COOLJAPAN Ecosystem

SciRS2 is part of the **COOLJAPAN Ecosystem** - a comprehensive collection of production-grade Rust libraries for scientific computing, machine learning, and data science. All ecosystem projects follow the [SciRS2 POLICY](SCIRS2_POLICY.md) for consistent architecture, leveraging scirs2-core abstractions for optimal performance and maintainability.

### 📊 Scientific Computing & Data Processing

#### [NumRS2](https://github.com/cool-japan/numrs)
**NumPy-compatible N-dimensional arrays in pure Rust**
- Pure Rust implementation of NumPy with 95%+ API coverage
- Zero-copy views, advanced broadcasting, and memory-efficient operations
- SIMD vectorization achieving 2-10x performance over Python NumPy

#### [PandRS](https://github.com/cool-japan/pandrs)
**Pandas-compatible DataFrames for high-performance data manipulation**
- Full Pandas API compatibility with Rust's safety guarantees
- Advanced indexing, groupby operations, and time series functionality
- 10-50x faster than Python pandas for large datasets

#### [QuantRS2](https://github.com/cool-japan/quantrs)
**Quantum computing library in pure Rust**
- Quantum circuit simulation and execution
- Quantum algorithm implementations
- Integration with quantum hardware backends

### 🤖 Machine Learning & Deep Learning

#### [OptiRS](https://github.com/cool-japan/optirs)
**Advanced ML optimization algorithms extending SciRS2**
- GPU-accelerated training (CUDA, ROCm, Metal) with 10-50x speedups
- 19 production-ready optimizers: Adam, RAdam, Lookahead, LAMB, learned optimizers
- Neural Architecture Search (NAS), pruning, and quantization
- Distributed training with data/model parallelism and TPU coordination

#### [ToRSh](https://github.com/cool-japan/torsh)
**PyTorch-compatible deep learning framework in pure Rust**
- 100% SciRS2 integration across all 18 crates
- Dynamic computation graphs with eager execution
- Graph neural networks, transformers, time series, and computer vision
- Distributed training and ONNX export for production deployment

#### [TenfloweRS](https://github.com/cool-japan/tenflowers)
**TensorFlow-compatible ML framework with dual execution modes**
- Eager execution (PyTorch-style) and static graphs (TensorFlow-style)
- Cross-platform GPU acceleration via WGPU (Metal, Vulkan, DirectX)
- Built on NumRS2 and SciRS2 for numerical computing foundation
- Python bindings via PyO3 and ONNX support for model exchange

#### [SkleaRS](https://github.com/cool-japan/sklears)
**scikit-learn compatible machine learning library**
- 3-100x performance improvements over Python implementations
- Classification, regression, clustering, preprocessing, and model selection
- GPU acceleration, ONNX export, and AutoML capabilities

#### [TrustformeRS](https://github.com/cool-japan/trustformers)
**Hugging Face Transformers in pure Rust for production deployment**
- BERT, GPT-2/3/4, T5, BART, RoBERTa, DistilBERT, and more
- Full training infrastructure with mixed precision and gradient accumulation
- Optimized inference (up to 1.67x faster than PyTorch) with quantization support

### 🎙️ Speech & Audio Processing

#### [VoiRS](https://github.com/cool-japan/voirs)
**Pure-Rust neural speech synthesis (Text-to-Speech)**
- State-of-the-art quality with VITS and DiffWave models (MOS 4.4+)
- Real-time performance: ≤0.3× RTF on CPUs, ≤0.05× RTF on GPUs
- Multi-platform support (x86_64, aarch64, WASM) with streaming synthesis
- SSML support and 20+ languages with pluggable G2P backends

### 🕸️ Semantic Web & Knowledge Graphs

#### [OxiRS](https://github.com/cool-japan/oxirs)
**Semantic Web platform with SPARQL 1.2, GraphQL, and AI reasoning**
- Rust-first alternative to Apache Jena + Fuseki with memory safety
- Advanced SPARQL 1.2 features: property paths, aggregation, federation
- GraphQL API with real-time subscriptions and schema stitching
- AI-augmented reasoning: embedding-based semantic search, LLM integration
- Vision transformers for image understanding and vector database integration

### 🧮 Formal Verification & Constraint Solving

#### [OxiZ](https://github.com/cool-japan/oxiz)
**Pure Rust SMT solver - Z3-compatible constraint solving engine**
- Drop-in replacement for Z3 with no C/C++ dependencies
- Satisfiability Modulo Theories (SMT) for formal verification and program analysis
- Support for propositional logic, linear arithmetic, bitvectors, and arrays
- Integration with COOLJAPAN ecosystem for mathematical proof and optimization

#### [OxiLean](https://github.com/cool-japan/oxilean)
**Pure Rust Interactive Theorem Prover — Calculus of Inductive Constructions**
- Zero-dependency kernel (115k SLOC TCB), WASM-first design
- Universe hierarchy, dependent types, inductive types, proof irrelevance, universe polymorphism
- Cargo integration for proof libraries as crates; 11 workspace crates

### 🌍 Geospatial & Data Processing

#### [OxiGDAL](https://github.com/cool-japan/oxigdal)
**Pure Rust Geospatial Data Abstraction Library — Production-Grade GDAL Alternative**
- 76 workspace crates, ~540k SLoC with 15 format drivers (GeoTIFF/COG, GeoJSON, GeoParquet, Zarr, FlatGeobuf, Shapefile, NetCDF, HDF5, GRIB, JPEG2000, VRT, COPC/LAS, GeoPackage, MBTiles, PMTiles)
- Full CRS transformations (20+ projections, 211+ EPSG codes), cloud-native I/O (S3/GCS/Azure), GPU acceleration
- Cross-platform: WASM, iOS, Android, embedded (no_std); zero C/C++ dependencies in default features

### ⚖️ Legal Technology

#### [Legalis-RS](https://github.com/cool-japan/legalis)
**Rust Framework for Parsing, Analyzing, and Simulating Legal Statutes — "Governance as Code, Justice as Narrative"**
- 23 operational jurisdictions (JP, US, EU, UK, DE, FR, SG, CN, IN, BR, etc.), 46 workspace crates, ~897k SLoC, 14,705 tests
- Deterministic legal logic separated from judicial discretion via `LegalResult<T>` type
- LLM integration, formal verification (SMT via OxiZ), statute diffing, smart contract export (Solidity/WASM/Ink!), Linked Open Data (RDF)

### 🤖 AI Infrastructure

#### [OxiRAG](https://github.com/cool-japan/oxirag)
**Four-Layer RAG Engine with SMT-Based Logic Verification and Knowledge Graph Support**
- 4 layers: Echo (vector search), Speculator (draft verification with SLM), Judge (SMT verification via OxiZ), Graph (GraphRAG)
- Speculative RAG, context-aware prefix caching, on-the-fly distillation, hidden states manipulation
- Native + WASM cross-platform; Candle-based SLM integration

#### [OxiFY](https://github.com/cool-japan/oxify)
**Graph-Based LLM Workflow Orchestration Platform in Pure Rust**
- DAG-based workflow engine with type-safe execution; node types: LLM, Retriever, Vision/OCR, Code, IfElse, Tool
- Multi-provider LLM support (OpenAI, Anthropic), vector DB integration (Qdrant, pgvector), MCP support
- 16 workspace crates; ReBAC authorization (Zanzibar-style), JWT/OAuth2, REST API (Axum)

#### [OxiGAF](https://github.com/cool-japan/oxigaf)
**Pure Rust Gaussian Avatar Reconstruction from Monocular Videos**
- 512×512 multi-view generation with IP-Adapter, classifier-free guidance, and latent upsampling
- Differentiable 3D Gaussian Splatting rasterizer (wgpu), FLAME parametric head model
- 7 workspace crates, 796 tests, zero C/Fortran dependencies; PyTorch weight conversion bridge

### 🎬 Multimedia & Computer Vision

#### [OxiMedia](https://github.com/cool-japan/oximedia)
**Pure Rust reconstruction of OpenCV + FFmpeg — Patent-free multimedia and computer vision framework**
- 106 workspace crates, ~2.16M SLoC; codec encode/decode (AV1, VP9, Opus, FLAC), container mux/demux (MP4, MKV, MPEG-TS, OGG)
- Streaming protocols (HLS, DASH, RTMP, SRT, WebRTC), transcoding pipelines, filter graphs (DAG-based)
- Computer vision: object detection, motion tracking, video stabilization, scene analysis, shot detection, denoising
- Professional image I/O (DPX, OpenEXR, TIFF), color management (ICC, ACES, HDR), quality metrics (PSNR/SSIM/VMAF)
- Zero C/Fortran dependencies, async-first (Tokio), WASM-ready, patent-free codecs only

### 🔗 Ecosystem Integration

All Cool Japan Ecosystem projects share:
- **Unified Architecture**: SciRS2 POLICY compliance for consistent APIs
- **Performance First**: SIMD optimization, GPU acceleration, zero-cost abstractions
- **Production Ready**: Memory safety, comprehensive testing, battle-tested in production
- **Cross-Platform**: Linux, macOS, Windows, WebAssembly, mobile, and edge devices
- **Python Interop**: PyO3 bindings for seamless Python integration
- **Enterprise Support**: Professional documentation, active maintenance, community support

**Getting Started**: Each project includes comprehensive documentation, examples, and migration guides. Visit individual project repositories for detailed installation instructions and tutorials.

## Future Directions

SciRS2 continues to evolve with ambitious goals:

### Near-Term (v0.4.x - v0.5.0)
- **SIMD Phase 70-80**: Additional advanced mathematical operations and optimizations
- **Enhanced GPU Support**: Improved multi-backend GPU acceleration and auto-tuning
- **Python Ecosystem**: Enhanced PyPI distribution, improved NumPy compatibility
- **Documentation**: Expanded tutorials, cookbook-style examples, migration guides
- **Performance Tuning**: Further optimization of hot paths

### Medium-Term (v0.5.x - v0.6.0)
- **Extended Hardware Support**: ARM NEON optimization, RISC-V support, embedded systems
- **Cloud Native**: Container optimization, serverless function support, distributed computing
- **Domain Extensions**: Quantitative finance, bioinformatics, computational physics
- **Ecosystem Integration**: Enhanced Python/Julia interoperability, R bindings
- **WebAssembly**: Optimized WASM builds for browser-based scientific computing

### Long-Term Vision
- **Automated Optimization**: Hardware-aware algorithm selection, auto-tuning frameworks
- **Advanced Accelerators**: TPU support, custom ASIC integration
- **Enterprise Features**: High-availability clusters, fault tolerance, monitoring dashboards
- **Educational Platform**: Interactive notebooks, online learning resources, certification programs

For detailed development status and contribution opportunities, see [TODO.md](TODO.md).

## Community and Support

### Get Involved

We welcome contributions from the community! Whether you're:
- 🐛 Reporting bugs or suggesting features
- 📝 Improving documentation or writing tutorials
- 🔬 Implementing new algorithms or optimizations
- 🎓 Using SciRS2 in research or education
- 💼 Deploying SciRS2 in production environments

Your participation helps make SciRS2 better for everyone.

### Resources

- **📖 Documentation**: Comprehensive API docs on [docs.rs/scirs2](https://docs.rs/scirs2)
- **💬 Discussions**: [GitHub Discussions](https://github.com/cool-japan/scirs/discussions)
- **🐛 Issue Tracker**: [GitHub Issues](https://github.com/cool-japan/scirs/issues)
- **📧 Contact**: [COOLJAPAN OU Team](https://github.com/cool-japan)
- **🌟 Star us**: Show your support on [GitHub](https://github.com/cool-japan/scirs)

### Citation

If you use SciRS2 in your research, please cite:

```bibtex
@software{scirs2_2026,
  title = {SciRS2: Scientific Computing and AI in Pure Rust},
  author = {{COOLJAPAN OU (Team KitaSan)}},
  year = {2026},
  url = {https://github.com/cool-japan/scirs},
  version = {0.6.1}
}
```

## Acknowledgments

SciRS2 builds on the shoulders of giants:
- **NumPy & SciPy**: Pioneering scientific computing in Python
- **Rust Community**: Creating a safe, fast, and productive language
- **ndarray**: High-quality array computing foundation
- **OxiBLAS & OxiFFT**: Pure Rust performance libraries (COOLJAPAN ecosystem)
- **Contributors**: Everyone who has contributed code, documentation, or feedback

Special thanks to the scientific computing and machine learning communities for their continuous innovation and open collaboration.

---

**Built with ❤️ by [COOLJAPAN OU (Team KitaSan)](https://github.com/cool-japan)**

**Part of the [Cool Japan Ecosystem](https://github.com/cool-japan) - Production-Grade Rust Libraries for Scientific Computing and AI**