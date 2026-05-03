# scirs2-core Development TODO

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
- [x] NUMA-local allocator backed by `libnuma` (feature-gated) — implemented in v0.4.2 (`memory/numa_allocator.rs` `discover_libnuma`, gated by `libnuma` feature)
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

- Several source files exceed 2000 lines (refactoring policy); track with `rslines 50` and split
- `#![allow(dead_code)]` is blanket-applied; should be narrowed to specific items
- GPU allocator tests are `#[ignore]`d on CI due to hardware availability; need mock backend
- NUMA allocator falls back silently when `libnuma` is absent; add explicit warning log
- `no_std` support is declared but not regularly tested; add CI job without `std` feature
- Lock-free structures use Rust `std::sync::atomic`; `loom` model checking not yet integrated

## v0.4.2 Additions

- [x] Metal GPU backend: `.expect()` calls replaced with proper error propagation (no-unwrap policy enforced)
- [x] Metal GPU batch dispatch: `begin_batch` / `end_batch` / `try_batch_dispatch` for grouped kernel submission
- [x] Metal GPU async dispatch: `dispatch_no_wait` + `gpu_sync` for non-blocking GPU work
- [x] Tracy profiler integration (feature-gated) — `profiling/tracy.rs`, enable with `tracy` cargo feature
- [x] NUMA-local allocator libnuma feature gate — `memory/numa_allocator.rs` `discover_libnuma`, enable with `libnuma` cargo feature
- [x] wgpu-based GPU buffer abstraction — `gpu/backends/wgpu.rs`
- [x] Browser-compatible feature flag (`target_arch = "wasm32"`) — WASM backend in `gpu/backends/wasm.rs`
- [x] Async GPU buffer transfer pipeline — `gpu/async_transfer.rs`
- [x] Unified memory allocator (CPU+GPU shared pages) — `gpu/memory_management/unified_memory.rs` (`UnifiedAllocator`, `UnifiedBuffer`, `SyncState`)
- [x] Persistent vector (RRB-tree) — `data_structures/rrb_tree.rs`
- [x] Tracy profiler integration (feature-gated) in `profiling/tracy.rs`
- [x] NUMA-local allocator `libnuma` feature gate added
- [x] Per-stream GPU memory allocator (Wave 43) — `gpu/stream_allocator.rs` (`StreamAllocator`, `StreamId`); 9 tests
- [x] Memory defragmentation for long-running workloads (Wave 43) — `memory/defrag.rs` (`DefragPlanner`, `OnlineDefragmenter`, `DefragStats`, `FreeBlock`); 8 tests
- [x] Cross-NUMA bandwidth measurement and routing (Wave 43) — `memory/numa_bandwidth.rs` (`NumaBandwidthMatrix`, `BandwidthMeasurement`, `probe_bandwidth_matrix`, `measure_copy_bandwidth`, `optimal_placement_node`); 11 tests
