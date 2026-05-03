# scirs2-wasm TODO

## Status: v0.4.3 Released (2026-05-03)

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
- [x] Advanced linear algebra (`linalg_advanced.rs`): Krylov solvers, randomized SVD
- [x] Enhanced signal processing (`signal_enhanced.rs`): adaptive filters
- [x] Advanced statistics (`stats_advanced.rs`): bootstrap, Monte Carlo

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
