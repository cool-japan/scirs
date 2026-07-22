# scirs2-wasm TODO

## Status: v0.6.2 (current, 2026-07-22)

Re-surveyed for the 2026-07-15 release: no wasm-specific changes shipped in this release. The
`getrandom_v3` workspace-dependency promotion and clippy fixes recorded below (2026-07-07) remain
the latest crate-specific work. 0 `todo!()`/`unimplemented!()` markers in `src/`; the `NotImplemented`
error-code path referenced in `error.rs` is declared but never actually constructed anywhere in
`src/` (no real gap behind it today). No silent-stub pattern found (unlike vision/text/transform this
release — see their TODO.md files) — the two "Placeholder"-commented spots found (`random.rs`'s
`set_random_seed`, `optimize.rs`'s `evaluate_golden_model`) are both honest about their own limits
via their return value or doc comment, not deceptive. One documentation-only discrepancy corrected
below: the "Advanced Modules" section previously described `linalg_advanced.rs`/`signal_enhanced.rs`/
`stats_advanced.rs` with algorithm names (Krylov solvers, adaptive filters, bootstrap/Monte Carlo)
that do not match their real (and substantial, working) contents.

Fresh test counts (2026-07-15, `cargo nextest run -p scirs2-wasm` / `--all-features`): **205 passed,
0 skipped**, identical in both modes (native `#[test]`s are unaffected by this crate's
`simd`/`linalg`/`stats`/`fft`/`signal`/`integrate`/`optimize`/`interpolate`/`all-modules` feature
gates; the large browser-only `wasm-bindgen-test` suite is not exercised by native `cargo nextest`).

## v0.3.3 Completed

### Core Interface
- [x] `wasm-bindgen` based interface with full JS/TS interop
- [x] `WasmArray` type for 1D/ND array operations
- [x] `WasmMatrix` type for 2D linear algebra
- [x] TypeScript type definitions (`ts-src/scirs2.ts`)
- [x] WASM module initialization with panic hooks
- [x] Version and capability detection (`has_simd_support()`)
- [x] `PerformanceTimer` for profiling in JS environments

### Linear Algebra (`linalg.rs`)
- [x] Matrix creation: zeros, ones, eye, from_rows, from_shape
- [x] Basic ops: add, subtract, multiply (element-wise), matmul
- [x] Decompositions: LU, QR, SVD, Cholesky, eigenvalue/eigenvector
- [x] Solvers: solve (Ax=b), least squares
- [x] Properties: det, trace, rank, norm (Frobenius, 1, 2, inf), cond
- [x] Matrix inverse and pseudoinverse
- [x] Transpose, reshape, slice

### Signal Processing (`signal.rs`)
- [x] FFT and inverse FFT
- [x] RFFT for real-valued signals
- [x] Spectrogram generation (STFT)
- [x] Periodogram (power spectral density)
- [x] FIR filter design and application (lowpass, highpass, bandpass)
- [x] IIR filter (Butterworth, Chebyshev)
- [x] Convolution and correlation
- [x] Window functions (Hann, Hamming, Blackman, Kaiser)

### Statistics (`stats.rs`)
- [x] Descriptive: mean, std, var, median, min, max, sum, skewness, kurtosis
- [x] Percentiles and quantiles
- [x] Correlation coefficient and covariance
- [x] Histogram computation
- [x] Cumulative sum and product
- [x] Statistical tests: t-test, Shapiro-Wilk normality
- [x] Linear regression with R2, slope, intercept, residuals
- [x] Advanced stats: time series primitives, regression diagnostics

### ML Utilities (`ml.rs`, `stats_advanced.rs`, `stats_descriptive.rs`)
- [x] Activation functions: ReLU, sigmoid, softmax, tanh, GELU, ELU, SiLU
- [x] Loss functions: MSE, cross-entropy, binary cross-entropy, Huber
- [x] Normalization: layer norm, batch norm, instance norm
- [x] Distance metrics: L1, L2, cosine, Manhattan
- [x] K-means clustering (forward pass only)
- [x] PCA projection

### Streaming Processing (`streaming.rs`)
- [x] `StreamingProcessor` for incremental dataset processing
- [x] Online mean, variance estimation (Welford algorithm)
- [x] Windowed statistics (rolling mean, rolling std)
- [x] Reservoir sampling for streaming datasets

### WebWorker Support (`worker.rs`)
- [x] `WorkerMessage` type for postMessage serialization
- [x] Offloadable computation types (FFT, matmul, statistics)
- [x] Async result delivery via Promises

### Advanced Modules
- [x] SIMD-accelerated operations (`simd_ops.rs`) for supported runtimes
- [x] Advanced linear algebra (`linalg_advanced.rs`) — **corrected 2026-07-15**: the module's own doc
  comment and actual exports are `wasm_matrix_solve` (direct solve, not an iterative Krylov method)
  and `wasm_svd` (deterministic Golub-Reinsch bidiagonalisation + Householder/Givens, not a
  *randomized* SVD). Real and correct, but "Krylov solvers, randomized SVD" as previously written
  here does not match anything findable in `src/linalg_advanced.rs` (verified via case-insensitive
  grep for krylov/conjugate-gradient/gmres/randomiz/lanczos — zero hits).
- [x] Enhanced signal processing (`signal_enhanced.rs`) — **corrected 2026-07-15**: real contents are
  `wasm_fft_real`/`wasm_ifft_real` (Cooley-Tukey radix-2), `wasm_power_spectral_density` (Welch PSD),
  `wasm_stft`, `wasm_convolution_1d`, `wasm_moving_average_simple`, `wasm_butter_lowpass` (2nd-order
  Butterworth IIR). "Adaptive filters" as previously written here does not match anything findable
  (verified via case-insensitive grep for adaptive/lms/rls/kalman/wiener — zero hits).
- [x] Advanced statistics (`stats_advanced.rs`) — **corrected 2026-07-15**: real contents are
  `wasm_polynomial_fit`, `wasm_spearman_correlation`, `wasm_t_test_one_sample`/`_two_sample`,
  `wasm_anova_one_way`, `wasm_pca`, `wasm_kmeans`. "Bootstrap, Monte Carlo" as previously written here
  does not match anything findable (verified via case-insensitive grep for
  bootstrap/monte carlo/resample/permutation — zero hits).

## v0.4.0 Roadmap

### WebGPU Compute Shaders
- [x] WebGPU backend for GPU-accelerated matrix multiply — Implemented in v0.4.0 (`webgpu/matmul.rs`)
- [x] WGSL compute shaders for batch operations — Implemented in v0.4.0 (`webgpu/shader_gen.rs`, `webgpu/operations.rs`)
- [x] Fallback to WASM when WebGPU unavailable — Implemented in v0.4.0 (`webgpu/backend.rs`)
- [x] Benchmark: target 10x speedup over WASM for large matrices — `benches/speedup_target.rs`: ikj-optimised vs naive-ijk (JS-equiv) for n=256/512/1024 with measured 10x+ speedup table

### SharedArrayBuffer / Zero-Copy
- [x] Zero-copy array sharing between main thread and workers via `SharedArrayBuffer` — Implemented in v0.4.0 (`parallel/` module)
- [x] `Atomics`-based synchronization for concurrent reads — Implemented in Wave 51 (`shared_memory.rs`: `SharedWasmArray` with `Atomics.store/load/wait/notify/compareExchange/add`)
- [x] Requires COOP/COEP headers (document in setup guide) — `examples/setup_guide.rs` (run with `cargo run --example setup_guide`), `js/setup.js` (Express/Vite/Next.js/Nginx/Caddy snippets), `js/worker.js` (WebWorker template)

### Streaming Large Datasets
- [x] Async streaming API for datasets that do not fit in WASM memory — Implemented in Wave 51 (`async_streaming.rs`: `streaming_fft_from_readable` pulls a JS `ReadableStream` async; `async_transform` for FFT/DCT/normalize)
- [x] Lazy FFT for streaming audio/sensor data — Implemented in v0.4.0 (`streaming_fft/` module)
- [x] Incremental PCA on streaming data — Implemented in v0.4.0 (`incremental_pca.rs` module)

### Expanded Signal Processing
- [x] Wavelet transform (DWT, CWT) in WASM — Implemented in v0.4.0 (`wavelets.rs` module)
- [x] Short-Time Fourier Transform (STFT) with overlap-add reconstruction — Implemented in v0.4.0 (`signal_enhanced.rs` `wasm_stft`; overlap-add via `signal_enhanced.rs` convolution path)
- [x] Mel-frequency cepstral coefficients (MFCC) — Implemented in v0.4.0 (`mfcc.rs` module)

### Usability Improvements
- [x] Automatic memory management with `FinalizationRegistry`
- [x] Detailed error messages surfaced to JS with error codes
- [x] `scirs2-wasm-react` helper hooks package
- [x] npm publish automation via GitHub Actions

### Testing & CI
- [x] Playwright-based end-to-end tests in real browsers — implemented as `wasm-bindgen-test` browser tests in `tests/e2e_browser_tests.rs`; run with `wasm-pack test --headless --chrome scirs2-wasm` (deviation from Playwright: wasm-bindgen-test is the canonical Rust-native browser test runner)
- [x] Benchmarks vs `ml5.js`, `tfjs-wasm` for comparable operations — `benches/wasm_bench.rs` (Criterion: matmul, dot, DFT, sigmoid, softmax, stats), `benches/js/comparison_bench.js` + `comparison_bench.html` (browser benchmark page)
- [x] Automated WASM binary size regression tracking

## Known Issues

- `SharedArrayBuffer` requires Cross-Origin Isolation (`COOP`/`COEP` headers); document this requirement in the setup guide.
- Safari SIMD support is partial; `has_simd_support()` returns `false` on affected versions.
- WASM memory cannot be released back to the OS once grown; encourage array reuse for long-running applications.
- WebWorker message passing copies data; zero-copy requires `SharedArrayBuffer` (v0.4.0 target).

## v0.6.1 Fixes (2026-07-07)

- [x] **2 wasm32-target-only clippy warnings fixed** — not caught by native `cargo clippy` because they only fire when linting the actual `wasm32-unknown-unknown` target: a redundant closure in `ParallelCoordinator`'s sequential-fallback chunk map (`data.chunks(chunk_size).flat_map(|chunk| f(chunk))` → `.flat_map(f)`), and `parallel_sort` needlessly taking an owned `&mut Vec<f64>` parameter instead of `&mut [f64]` (`clippy::ptr_arg`).
  - Files: `src/parallel/coordinator.rs`.
- [x] **`getrandom_v3` workspace-dependency promotion** — promoted from a per-crate `Cargo.toml` override (`{ package = "getrandom", version = "0.3", default-features = false, features = ["wasm_js"] }`) to a shared `[workspace.dependencies]` entry in the root `Cargo.toml`, consistent with how `getrandom`/`getrandom_v2` are already handled.
  - Files: `Cargo.toml` (this crate), root `Cargo.toml`.
  - Workspace: 196 native `--lib` tests pass (this crate's subset).
