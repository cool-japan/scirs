# scirs2-core

[![crates.io](https://img.shields.io/crates/v/scirs2-core)](https://crates.io/crates/scirs2-core)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](../LICENSE)
[![Documentation](https://img.shields.io/docsrs/scirs2-core)](https://docs.rs/scirs2-core)
[![Status](https://img.shields.io/badge/status-stable-brightgreen)]()

**Foundation crate for the SciRS2 scientific computing ecosystem.**

`scirs2-core` provides the essential utilities, abstractions, and optimizations shared by every SciRS2 module. It enforces the SciRS2 POLICY: only `scirs2-core` uses external dependencies directly; all other crates consume re-exports and abstractions from this crate.

**Tests:** 3024/3025 passing (default features), 4540/4540 passing (`--all-features`) — as of 2026-07-15.

## Installation

```toml
[dependencies]
scirs2-core = "0.6.5"
```

With optional feature flags:

```toml
[dependencies]
scirs2-core = { version = "0.6.5", features = ["validation", "simd", "parallel", "gpu"] }
```

## Features (v0.6.5)

### Performance

- SIMD-accelerated array operations (SSE, AVX, AVX2, AVX-512, NEON) — up to 14x speedup over scalar
- Ultra-optimized SIMD with multiple accumulators, FMA, 8-way loop unrolling, software pipelining
- Work-stealing scheduler with NUMA-aware thread placement
- Parallel iterators (parallel map, reduce, scan, map-reduce)
- Async utilities: semaphore, channel, timeout, rate limiter
- Cache-oblivious B-tree and matrix multiply algorithms
- GPU memory management: pool allocator, slab allocator, buddy allocator, best-fit allocator

### Data Structures

- Lock-free queue, stack, and hash map (using CAS, epoch-based reclamation)
- HAMT (Hash Array Mapped Trie) persistent functional data structure
- Persistent red-black tree (immutable update)
- Interval tree, segment tree, van Emde Boas tree
- Skip list, finger tree, B-tree variants
- String interning (global thread-safe interner)
- Task graph with topological scheduling

### Memory Management

- Arena allocator (bump allocation)
- Slab allocator (fixed-size object pools)
- NUMA-aware allocator with topology detection
- Object pool with configurable capacity
- Zero-copy buffer management
- Memory-mapped array support (`MemoryMappedArray`)
- Chunked out-of-core array processing

### Distributed Computing

- Ring allreduce (parameter averaging across nodes)
- Parameter server (key-value store with async push/pull)
- Collective operations: broadcast, scatter, gather, allgather, reduce-scatter
- Lock-free distributed data structures

### Validation

- Schema-based data validation with constraints
- Config file validation (JSON/TOML/YAML compatible schemas)
- Assertion helpers for scalars and arrays (`check_finite`, `check_positive`, `checkarray_finite`, `checkshape`)
- Type coercion utilities

### Scientific Infrastructure

- 30+ mathematical constants, 40+ physical constants
- Generic numeric traits (`Float`, `ScalarElem`, `LinalgScalar`, etc.)
- Complex number support via `num-complex` re-exports
- Arbitrary precision arithmetic (multi-precision floats and integers)
- Interval arithmetic (verified computing)
- Extended precision accumulators (Kahan, pairwise)

### ML Pipeline

- `Transformer` trait for data preprocessing steps
- `Predictor` trait for model inference
- `Evaluator` trait for scoring and metrics
- `Pipeline` struct for chaining transformers and a final predictor
- Batch inference utilities

### Observability

- Structured logging (tracing-compatible)
- Metrics collector (counters, histograms, gauges)
- GPU profiler and perf-event profiler stubs
- Distributed tracing integration

### Other Utilities

- Bioinformatics: sequence alignment extensions, motif finding, sequence type utilities
- Geospatial: geodesic calculations, projections, spatial indexing
- Quantum computing primitives: qubit representation, gate operations, measurement simulation
- Reactive programming primitives: observable, subject, operators
- Combinatorics utilities: permutations, combinations, partitions
- Concurrent collections: concurrent hash map, priority queue

## Usage Examples

### Basic validation

```rust
use scirs2_core::validation::{check_positive, checkarray_finite};
use scirs2_core::ndarray::array;

// Array-level finiteness check
let data = array![[1.0_f64, 2.0], [3.0, 4.0]];
checkarray_finite(&data, "input")?;

// Scalar positivity check
let weight = 0.5_f64;
check_positive(weight, "weight")?;
```

### SIMD operations

```rust
use scirs2_core::simd::{simd_add_f64, simd_dot_f64};
use scirs2_core::ndarray::Array1;

let a: Array1<f64> = Array1::from_elem(1024, 1.0);
let b: Array1<f64> = Array1::from_elem(1024, 2.0);

let sum = simd_add_f64(&a.view(), &b.view());
let dot = simd_dot_f64(&a.view(), &b.view());
```

### Parallel processing

```rust
use scirs2_core::concurrent::parallel_iter::{parallel_map, parallel_reduce};

let data: Vec<f64> = (0..1_000_000).map(|i| i as f64).collect();

// Trailing arg is a worker-count hint (0 = auto-detect)
let squares: Vec<f64> = parallel_map(&data, |&x| x * x, 0)?;
let total: f64 = parallel_reduce(&data, 0.0, |acc, &x| acc + x, |a, b| a + b, 0)?;
```

### Lock-free queue

```rust
use scirs2_core::concurrent::LockFreeQueue;

// Capacity is rounded up to the next power of two (minimum 2)
let queue: LockFreeQueue<i32> = LockFreeQueue::new(4);
assert!(queue.push(42));
let val = queue.pop(); // Some(42)
```

### ML pipeline

```rust
use scirs2_core::ml_pipeline::pipeline::{linear_regression_pipeline, RegressionPipeline};
use scirs2_core::ndarray::{Array1, Array2};

// Convenience constructor: StandardScaler -> LinearRegressor
let mut pipeline: RegressionPipeline = linear_regression_pipeline();

let x = Array2::from_shape_vec((4, 1), vec![1.0_f64, 2.0, 3.0, 4.0])?;
let y = Array1::from_vec(vec![3.0_f64, 5.0, 7.0, 9.0]); // y = 2x + 1

pipeline.fit(x.view(), y.view())?;
let predictions = pipeline.predict(x.view())?;
```

## v0.5.0 Additions

### NUMA-Aware Parallel Mapping

`par_map_chunks` provides typed-result chunk-parallel mapping with NUMA locality (Linux `pthread` affinity pin; rayon fallback on Darwin/WASM):

```rust
use scirs2_core::par_map_chunks; // re-exported at the crate root

let data = vec![1.0_f64; 4096];
// Returns Vec<f64> directly (not a Result) — chunk order is preserved.
let results: Vec<f64> = par_map_chunks(&data, 64, |chunk| {
    chunk.iter().map(|x| x * x).collect::<Vec<_>>()
});
```

### GpuNdarray — Native f32 Array on WebGPU

`GpuNdarray<f32>` implements `ArrayProtocol` with real wgpu dispatch for elementwise add/subtract/multiply,
scalar multiply, sum reduction, dot product, and tiled matmul:

```rust
use scirs2_core::array_protocol::gpu_ndarray::GpuNdarray;

// GPU array operations (construction uploads to the process-wide wgpu device;
// falls back to an error if no adapter is available rather than silently using the CPU)
let a = GpuNdarray::from_data(&[1.0_f32, 2.0, 3.0, 4.0], vec![2, 2])?;
let b = GpuNdarray::from_data(&[5.0_f32, 6.0, 7.0, 8.0], vec![2, 2])?;
let c = a.add(&b)?;         // WGSL elementwise add
let m = a.matmul(&b)?;      // WGSL tiled matmul (16×16 shared mem)
let s = a.sum_all()?;       // WGSL two-pass reduce
let d = a.dot_gpu(&b)?;     // dot product (elementwise multiply + sum_all)
```

Other array-protocol operations (transpose, axis-reductions, SVD, inverse) are provided
generically through the `array_protocol::operations` dispatch layer rather than as
`GpuNdarray` inherent methods, and fall back to CPU implementations.

### WGSL Kernel Registry (v0.5.0)

All 13 previously-empty WGSL kernel slots are now filled in `gpu/kernels/mod.rs`:
Adam, SGD, RMSprop, Adagrad, LAMB optimizers; memcpy, fill, reduce_sum, reduce_max; RK4 stages (rk4_1/2/3/4 + combine) and error estimate.

## v0.6.1 Additions

### SIMD Squared Euclidean Distance

`SimdUnifiedOps` gained `simd_distance_squared_euclidean` (squared Euclidean distance, no `sqrt`), implemented for `f32`/`f64` in `src/simd/distances.rs`:

```rust
use scirs2_core::simd_ops::SimdUnifiedOps;
use scirs2_core::ndarray::array;

let a = array![1.0_f64, 2.0, 3.0];
let b = array![4.0_f64, 5.0, 6.0];
let d2 = f64::simd_distance_squared_euclidean(&a.view(), &b.view());
```

### Other Additions

- `training_history(&self) -> &[f64]` on `NormalizingFlow`, `ScoreBasedDiffusion`, `EnergyBasedModel`, and `NeuralPosteriorEstimation` (`src/random/neural_sampling.rs`) — per-epoch average loss recorded during `train()`.
- Real GPU runtime detection feeding `PlatformCapabilities` (`src/simd_ops/gpu_detection.rs`): CUDA is probed via dynamic `libcuda`/`nvcuda` loading plus `cuInit`/`cuDeviceGetCount`; Metal is probed via the `metal` feature or a documented platform heuristic. Replaces the previous stub.
- `ProductionProfiler::export_data()` (`src/profiling/production.rs`) now returns a real JSON snapshot (config, resource utilization, active workload IDs) instead of a placeholder.

## v0.6.3 — Windows Compatibility Fix

Two Windows-only bugs were fixed in this release:

- **Real memory-profiling backend for Windows** — `profiling::memory_profiling` (`src/profiling/memory_profiling.rs`) gained a `K32GetProcessMemoryInfo` (kernel32)-backed backend reporting actual process memory use; it previously fell back to an atomic-tracking estimate on Windows.
- **`CrossPlatformValidator::validate_windows_path` drive-letter fix** (`src/validation/cross_platform.rs`) — the validator incorrectly flagged the drive-letter colon in any absolute Windows path (e.g. `C:\Users\...`) as an invalid character; it now strips the leading `X:` drive specifier before scanning for `<>:"|?*`.

Verified by code review and the existing macOS/Linux test suite (unchanged pass counts above); not exercised under a Windows CI run.

## v0.6.2 — Pure Rust Dependency Removals

Three C/C++ dependencies were removed from `scirs2-core`'s default and optional feature builds:

- **`libnuma` (C) removed** — the `extern "C"` FFI block backing `NumaTopology::discover()` is gone; Linux NUMA topology discovery is now handled unconditionally by the existing pure-Rust `sysfs` (`/sys`) parser, which also filters out CPU-less/memory-only nodes to match `libnuma`'s prior behavior. The `libnuma` Cargo feature name is kept as an inert no-op alias so existing `features = ["libnuma"]` declarations keep resolving.
- **`tracy-client` (C++) removed** — the `tracy` feature is now implemented entirely in pure Rust. `TracyClient` gained `export_chrome_trace(path)`, writing recorded spans/frame-marks/messages out as a Chrome Trace Event Format (Perfetto-compatible) JSON document viewable at `ui.perfetto.dev` or `chrome://tracing`; the `TracyClient`/`TracySpan`/`tracy_span!` API is otherwise unchanged:

```rust
use scirs2_core::profiling::tracy::TracyClient;

let client = TracyClient::new();
if client.is_active() {
    client.message("profiling enabled");
}
{
    let _span = client.span("my_operation");
    // work here
} // span ends on drop

// Valid even when the `tracy` feature is disabled -- writes an empty-but-valid trace document.
let _ = client.export_chrome_trace(std::env::temp_dir().join("trace.json"));
```

- **`opencl3` removed** — OpenCL support now resolves `libOpenCL.so.1`/`libOpenCL.so`/`OpenCL` via a runtime `dlopen` loader instead of linking at build time; the public `OpenCLContext` API is unchanged, and code paths still fall back gracefully to wgpu/CPU when no OpenCL ICD is present. `gpu/backends/opencl.rs` was split into `opencl/{mod.rs, ffi.rs, memory_pool.rs}`.

## Feature Flags

| Feature | Description |
|---------|-------------|
| `validation` | Data validation helpers (`check_finite`, schema validation) |
| `simd` | SIMD-accelerated array operations |
| `parallel` | Multi-threaded parallel processing via Rayon |
| `gpu` | GPU memory management and kernel abstractions (backend-agnostic) |
| `opencl` / `metal` / `wgpu` / `rocm` | Backend-specific GPU acceleration (each requires `gpu`) |
| `memory_management` | Advanced memory utilities (arena, slab, pool) |
| `array_protocol` | Extensible unified array interface |
| `array_protocol_wgpu` | `GpuNdarray<f32>` with real wgpu dispatch (requires `array_protocol`) |
| `logging` | Structured logging integration |
| `profiling` | Performance profiling tools (metrics, dashboards, flame graphs, OpenTelemetry/Prometheus export); GPU- and OS-hardware-counter profiling remain partially stubbed on non-Linux platforms — see Observability note above |
| `std` | Standard library support (enabled by default; disable for `no_std`) |
| `all` | All stable features |

Note: as of 0.6.x, NVIDIA CUDA support is no longer a `scirs2-core` feature — it was decentralized into
per-crate `oxicuda-*` backend dependencies (direct CUDA integration lives in the consuming crate, not in
`scirs2-core`). `scirs2-core`'s own `gpu` feature stays backend-agnostic (`opencl`, `metal`, `wgpu`, `rocm`).
`scirs2-core` exposes 70+ Cargo features in total; the table above lists the most commonly used ones — see
`scirs2-core/Cargo.toml` `[features]` for the complete, authoritative list.

## Links

- [API Documentation](https://docs.rs/scirs2-core)
- [SciRS2 Repository](https://github.com/cool-japan/scirs)
- [SciRS2 POLICY](../SCIRS2_POLICY.md)

## License

Licensed under the Apache License 2.0. See [LICENSE](../LICENSE) for details.
