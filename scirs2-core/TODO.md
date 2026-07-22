# scirs2-core Development TODO

## Status: v0.6.2 (released 2026-07-22)

**0.6.2:** removed three C/C++ dependencies — `libnuma` (the `NumaTopology::discover()` FFI block; NUMA topology now read from the existing pure-Rust `/sys` sysfs parser unconditionally, `libnuma` Cargo feature kept as an inert no-op alias), `tracy-client` (the `tracy` feature is now pure Rust; `TracyClient` gained `export_chrome_trace(path)`, a Chrome Trace Event Format / Perfetto-compatible JSON exporter), and `opencl3` (OpenCL now loads via runtime `dlopen` instead of build-time linking; `gpu/backends/opencl.rs` split into `opencl/{mod.rs, ffi.rs, memory_pool.rs}`). See `CHANGELOG.md` `[0.6.2]` for full detail.

scirs2-core's own test suite (freshly run 2026-07-15, predates the 0.6.2 changes above): 3025 tests run, 3024 passed, 1 failed, 2 skipped
(default features); 4540 tests run, 4540 passed (2 flagged leaky by nextest's leak detector, not
failures), 14 skipped (`--all-features`). The 1 default-features failure is
`simd_matmul_tests::test_performance_benchmark`, a hardcoded wall-clock threshold assertion
(`elapsed.as_millis() < 500` for a 256×256 f32 matmul) — it failed under heavy CPU contention from
concurrent builds sharing this machine (100+ concurrent cargo/rustc processes from other agents at the
time), not a functional regression; the same test passed cleanly in the `--all-features` run, and the
underlying SIMD matmul correctness assertions in the same test passed in both runs. This session also
fixed a broken rustdoc intra-doc link in `profiling/production.rs` (`Dashboard::export_config` →
`PerformanceDashboard::export_config`) — a docs-only fix.

## v0.3.3 — COMPLETED

### Work-Stealing Scheduler and Parallel Iterators
- Work-stealing deque with Chase-Lev algorithm
- Parallel map, reduce, scan, map-reduce primitives
- Parallel iterator adapters (ParallelIterator trait)
- NUMA-aware thread placement and affinity

### Async Utilities
- Async semaphore (tokio-compatible)
- Bounded async channel
- Async timeout wrapper
- Async rate limiter (token bucket)

### Cache-Oblivious Algorithms
- Cache-oblivious B-tree (van Emde Boas layout)
- Cache-oblivious matrix multiply (recursive tiling)
- Cache-oblivious merge sort

### Lock-Free Data Structures
- Lock-free queue (Michael-Scott queue with epoch GC)
- Lock-free stack (Treiber stack)
- Lock-free hash map (split-ordered lists)
- Fixed `LockFreeQueue` CAS-before-read race condition (Feb 26, 2026)

### HAMT Persistent Data Structure
- Hash array mapped trie with structural sharing
- Persistent insert, delete, lookup
- Iterator over key-value pairs

### GPU Memory Management
- Pool allocator (fixed-size blocks)
- Slab allocator (typed object pools)
- Buddy allocator (power-of-two splitting/merging)
- Best-fit allocator with free-list coalescing
- GPU buffer abstraction over multiple backends

### Memory Utilities
- Arena allocator (bump pointer)
- NUMA allocator with topology detection
- Object pool with reuse tracking
- Zero-copy buffer management
- `MemoryMappedArray` for out-of-core data

### Validation System
- Schema-based validation (`ValidationSchema`, `Constraint`)
- Config validation with JSON/TOML-compatible schemas
- Assertion helpers: `check_finite`, `check_positive`, `check_shape`, `check_range`
- Type coercion utilities

### Distributed Computing
- Ring allreduce (bandwidth-optimal gradient averaging)
- Parameter server with async push/pull
- Collective ops: broadcast, scatter, gather, allgather, reduce-scatter

### ML Pipeline Abstractions
- `Transformer` trait (fit/transform)
- `Predictor` trait (predict/predict_proba)
- `Evaluator` trait (score with configurable metrics)
- `Pipeline` struct for chaining steps
- Batch and streaming inference modes

### Metrics Collector
- Counters, gauges, histograms
- Label sets for multi-dimensional metrics
- Export hooks (text format compatible with Prometheus)

### Other Additions
- Bioinformatics: alignment extensions, motif detection, sequence types
- Geospatial: geodesic distance, coordinate projections, spatial stats
- Quantum computing primitives: qubit, gate, measurement
- Reactive programming: Observable, Subject, filter/map/merge operators
- Combinatorics: permutations, combinations, partitions, multinomials
- String interning: global interner with `InternedStr` type
- Arbitrary precision: multi-precision floats and integers
- Interval arithmetic: directed rounding, verified inclusion

---

## v0.4.0 — Planned

### GPU Memory Pooling Enhancements
- [x] Unified memory (CPU+GPU shared pages) allocator — implemented in v0.4.2 (`gpu/memory_management/unified_memory.rs`, `UnifiedAllocator`/`UnifiedBuffer`/`SyncState`)
- [x] Async GPU buffer transfer pipeline — implemented in v0.4.2 (`gpu/async_transfer.rs`)
- [x] Per-stream allocation for CUDA streams — implemented in v0.4.2 (`gpu/stream_allocator.rs`, `StreamAllocator`/`StreamId`)
- [x] Memory defragmentation for long-running workloads — implemented in v0.4.2 (`memory/defrag.rs`, `DefragPlanner`/`OnlineDefragmenter`/`DefragStats`)

### NUMA-Aware Allocation
- [x] NUMA-local allocator backed by Pure Rust sysfs topology discovery — implemented in v0.4.2 (`memory/numa_allocator.rs` `discover_linux`); `libnuma` C dependency removed, `libnuma` feature is now an inert no-op alias
- [x] Automatic NUMA-aware placement for parallel work items — implemented in v0.4.2 (`memory/numa_bandwidth.rs` `optimal_placement_node`)
- [x] Cross-NUMA bandwidth measurement and routing — implemented in v0.4.2 (`memory/numa_bandwidth.rs`, `NumaBandwidthMatrix`/`probe_bandwidth_matrix`/`measure_copy_bandwidth`)

### WebGPU Backend Preparation
- [x] `wgpu`-based GPU buffer abstraction — implemented in `gpu/backends/wgpu.rs`
- [x] Compute shader dispatch via WebGPU (implemented 2026-04-17)
  - **Goal:** `scirs2-core` runs a real WGSL compute shader end-to-end through the `wgpu` backend — WGSL source → `wgpu::ShaderModule` → `wgpu::ComputePipeline` → bind group → `compute_pass.dispatch_workgroups(...)` → GPU buffer read-back matching the expected result of a deterministic kernel (vector-add). The current mock (line 31 `type WgpuComputePipeline = *mut std::ffi::c_void`; line 455 `Ok(0x3 as WgpuComputePipeline)`) is replaced with a `WgpuComputePipeline` struct wrapping the real `wgpu::ComputePipeline` + `wgpu::BindGroupLayout`. `scirs2-core/TODO.md` flips `[ ]` → `[x]` only after the real path runs.
  - **Design:** In `scirs2-core/src/gpu/backends/wgpu.rs`, replace the `*mut c_void` alias with a proper `WgpuComputePipeline { pipeline: wgpu::ComputePipeline, bind_group_layout: wgpu::BindGroupLayout, workgroup_size: [u32; 3] }`. Implement `compile_wgsl(source: &str) -> Result<WgpuComputePipeline>` using `device.create_shader_module(ShaderModuleDescriptor { source: ShaderSource::Wgsl(...) })` + `device.create_compute_pipeline(...)`. Implement `dispatch(pipeline, bindings, workgroups)` using the real `CommandEncoder` → `ComputePass` path. Retain a **headless-adapter fallback** (request `PowerPreference::LowPower` + `backends: PRIMARY`) so tests can run on CI runners with llvmpipe / Vulkan SW. If no adapter is available on the host, skip at runtime with `#[ignore]`.
  - **Files:** `scirs2-core/src/gpu/backends/wgpu.rs` (replace mock, add real pipeline struct + compile + dispatch), `scirs2-core/tests/webgpu_compute_smoke.rs` (new), `scirs2-core/TODO.md`.
  - **Prerequisites:** `wgpu = "29"` is already a workspace dep. Confirm `wgpu::Instance::new(...)` compiles.
  - **Tests:** `webgpu_compute_smoke::vector_add_runs_on_real_gpu_when_available` (acquires adapter via `wgpu::Instance`; if `None`, emit `#[ignore]` message and pass); `webgpu_compute_smoke::wgsl_compile_rejects_invalid_source`; `webgpu_compute_smoke::pipeline_struct_is_send_sync` (static assertion).
  - **Risk:** Real `wgpu` device creation is async on some backends; use `pollster::block_on` at the FFI boundary. Large risk: adapter unavailable in headless CI → mitigated by runtime-detect + skip. If file exceeds 2000 lines after edits, split via `splitrs` into `wgpu/pipeline.rs` + `wgpu/dispatch.rs`.
- [x] Browser-compatible feature flag (`target_arch = "wasm32"`) — WASM backend in `gpu/backends/wasm.rs`

### Distributed Computing Enhancements
- [x] Gossip protocol for peer discovery — implemented in `distributed/param_server/gossip.rs`
- [x] Fault-tolerant parameter server (leader election) — implemented in `distributed/param_server/fault_tolerance.rs`
- [x] Gradient compression (top-k sparsification, quantization) — Implemented in v0.4.0 (`distributed/compression.rs` top-k sparsification; `distributed/parameter_server.rs` error-feedback compressor)

### Profiling Improvements
- [x] perf-event integration for Linux hardware counters — implemented in `profiling/hardware_counters.rs`
- [x] Tracy profiler integration (feature-gated) — implemented in v0.4.2 (`profiling/tracy.rs`, gated by `tracy` feature)
- [x] Flame graph export from profiling data — implemented in `profiling/flame_graph_svg.rs`

### Additional Data Structures
- [x] Persistent vector (RRB-tree) — implemented in v0.4.2 (`data_structures/rrb_tree.rs`)
- [x] Concurrent skip list — Implemented in v0.4.0
- [x] Compressed trie for string keys — Implemented in v0.4.0
- [x] Bloom filter and counting Bloom filter — Implemented in v0.4.0 (includes count-min sketch, HyperLogLog)

---

## v0.4.1 — COMPLETED

### JIT Compilation Improvements
- [x] Added two targeted enhancements to `jit.rs` (branch 0.4.1, March 2026)
- [x] All v0.4.0 items carried forward as complete

### v0.4.0 Items Status
All items listed under v0.4.0 Planned were implemented during Waves 1-39 and are complete as of v0.4.1.

---

## Known Issues / Technical Debt

- Verified 2026-07-15: no source file in `scirs2-core/src` currently exceeds the 2000-line refactoring threshold (closest: `ecosystem/validation.rs` at 1924, `performance_optimization.rs` at 1865, `profiling/coverage.rs` at 1834). Several files are close to the limit — re-check with `rslines 50` before adding substantial code to any of them.
- `#![allow(dead_code)]` is blanket-applied; should be narrowed to specific items
- GPU allocator tests are `#[ignore]`d on CI due to hardware availability; need mock backend
- NUMA allocator falls back silently when sysfs topology is unavailable; add explicit warning log
- `no_std` support is declared but not regularly tested; add CI job without `std` feature
- Lock-free structures use Rust `std::sync::atomic`; `loom` model checking not yet integrated
- **Audit-log disk-space check is Unix-only** (`observability/audit/storage.rs`, `get_available_disk_space`). It uses `libc::statvfs`; on Windows (and any non-Unix target, or with the `libc` feature off) it now returns `Err("not determinable")`. Consequence: `AuditStorage`'s emergency cleanup — which deletes the oldest audit logs when free space drops below `retention_policy.min_free_space` — is **silently skipped** on those platforms, so a full disk surfaces only as a write failure. Fix by implementing it portably via `sysinfo::Disks::new_with_refreshed_list()` (match the longest mount point that prefixes the target path, read `available_space()`); `sysinfo` is already an optional dep with a matching cargo feature, so this adds no new dependency. Keep the `Err` fallback for builds with that feature off. **Do not** restore the previous `Ok(10 GB)` stub — besides disabling cleanup under the 1 GB default, a `min_free_space` above the stub made the caller's `new_available >= min_free_space` loop guard permanently false, deleting every collected audit log. While fixing, also reconsider the two `if let Ok(..)` call sites (~`storage.rs:464`, `:499`) that swallow the error without logging.

## v0.4.2 Additions

- [x] Metal GPU backend: `.expect()` calls replaced with proper error propagation (no-unwrap policy enforced)
- [x] Metal GPU batch dispatch: `begin_batch` / `end_batch` / `try_batch_dispatch` for grouped kernel submission
- [x] Metal GPU async dispatch: `dispatch_no_wait` + `gpu_sync` for non-blocking GPU work
- [x] Tracy profiler integration (feature-gated) — `profiling/tracy.rs`, enable with `tracy` cargo feature
- [x] NUMA-local allocator Pure Rust sysfs discovery — `memory/numa_allocator.rs` `discover_linux`; `libnuma` cargo feature is now an inert no-op alias
- [x] wgpu-based GPU buffer abstraction — `gpu/backends/wgpu.rs`
- [x] Browser-compatible feature flag (`target_arch = "wasm32"`) — WASM backend in `gpu/backends/wasm.rs`
- [x] Async GPU buffer transfer pipeline — `gpu/async_transfer.rs`
- [x] Unified memory allocator (CPU+GPU shared pages) — `gpu/memory_management/unified_memory.rs` (`UnifiedAllocator`, `UnifiedBuffer`, `SyncState`)
- [x] Persistent vector (RRB-tree) — `data_structures/rrb_tree.rs`
- [x] Tracy profiler integration (feature-gated) in `profiling/tracy.rs`
- [x] NUMA-local allocator Pure Rust sysfs discovery (`libnuma` feature is now an inert no-op alias)
- [x] Per-stream GPU memory allocator (Wave 43) — `gpu/stream_allocator.rs` (`StreamAllocator`, `StreamId`); 9 tests
- [x] Memory defragmentation for long-running workloads (Wave 43) — `memory/defrag.rs` (`DefragPlanner`, `OnlineDefragmenter`, `DefragStats`, `FreeBlock`); 8 tests
- [x] Cross-NUMA bandwidth measurement and routing (Wave 43) — `memory/numa_bandwidth.rs` (`NumaBandwidthMatrix`, `BandwidthMeasurement`, `probe_bandwidth_matrix`, `measure_copy_bandwidth`, `optimal_placement_node`); 11 tests

## Wave 73 — NUMA par_map_chunks (2026-05-07)

- [x] **Typed-result NUMA-locality chunk map** (completed 2026-05-07)
  - `par_map_chunks<T, U, F>(input, chunk_size, f)` — rayon-backed with NUMA-locality thread pool on Linux; plain rayon fallback on Darwin/WASM.
  - Files: `src/parallel/numa/mod.rs`, `src/parallel/numa/par_map_chunks.rs`
  - Tests: 8 in `tests/parallel_numa_par_map_chunks_tests.rs`

## v0.6.2 — Pure Rust Dependency Removals (2026-07-22)

- [x] **`libnuma` C dependency removed** — the `#[link(name = "numa")]` `extern "C"` block (8 declared functions) backing `NumaTopology::discover()` is deleted; Linux NUMA topology discovery is now unconditionally handled by the existing pure-Rust `sysfs` (`/sys`) parser (a separate code path from `memory/numa_allocator.rs`'s `discover_linux`, already pure-Rust since v0.4.2 — see above). The sysfs parser now also filters out CPU-less/memory-only nodes to match `libnuma`'s prior behavior. The `libnuma` Cargo feature name is kept as an inert no-op alias. The now-orphaned `[patch.crates-io]` entry for `bitflags` 0.6.0 (pulled in solely by `libnuma`'s transitive deps) was also dropped.
- [x] **`tracy-client` C++ dependency removed** — the `tracy` feature is now implemented entirely in pure Rust (`profiling/tracy.rs`). `TracyClient` gained `export_chrome_trace(path)`, exporting recorded spans/frame-marks/messages as Chrome Trace Event Format (Perfetto-compatible) JSON; `TracyClient`/`TracySpan`/`tracy_span!` API otherwise unchanged.
- [x] **`opencl3` dependency removed** — OpenCL support now resolves `libOpenCL.so.1`/`libOpenCL.so`/`OpenCL` via a runtime `dlopen` loader (`libloading`) instead of linking `-lOpenCL` at build time; public `OpenCLContext` API unchanged; falls back gracefully to wgpu/CPU when no ICD is present. `gpu/backends/opencl.rs` split into `opencl/{mod.rs, ffi.rs, memory_pool.rs}`.
- Net effect: `libnuma`, `tracy-client`, and `opencl3` no longer appear anywhere in the dependency graph. One deliberate exception remains: the `mpsgraph` feature (Apple MPSGraph GPU acceleration, macOS-only, off by default) still compiles a small Objective-C wrapper via the `cc` crate — unchanged by this release — since there is no pure-Rust binding path to that framework.
