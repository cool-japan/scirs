# scirs2-core

[![crates.io](https://img.shields.io/crates/v/scirs2-core)](https://crates.io/crates/scirs2-core)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](../LICENSE)
[![Documentation](https://img.shields.io/docsrs/scirs2-core)](https://docs.rs/scirs2-core)

**Foundation crate for the SciRS2 scientific computing ecosystem.**

`scirs2-core` provides the essential utilities, abstractions, and optimizations shared by every SciRS2 module. It enforces the SciRS2 POLICY: only `scirs2-core` uses external dependencies directly; all other crates consume re-exports and abstractions from this crate.

## Installation

```toml
[dependencies]
scirs2-core = "0.4.4"
```

With optional feature flags:

```toml
[dependencies]
scirs2-core = { version = "0.4.4", features = ["validation", "simd", "parallel", "gpu"] }
```

## Features (v0.4.4)

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
- Assertion helpers for numeric arrays (check_finite, check_positive, check_shape)
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
use scirs2_core::validation::{check_finite, check_positive};
use scirs2_core::ndarray::array;

let data = array![[1.0_f64, 2.0], [3.0, 4.0]];
check_finite(&data.view(), "input")?;
check_positive(&data.view(), "weights")?;
```

### SIMD operations

```rust
use scirs2_core::simd_ops::{simd_add_f64, simd_dot_f64};

let a = vec![1.0_f64; 1024];
let b = vec![2.0_f64; 1024];

let sum = simd_add_f64(&a, &b);
let dot = simd_dot_f64(&a, &b);
```

### Parallel processing

```rust
use scirs2_core::parallel_ops::{parallel_map, parallel_reduce};

let data: Vec<f64> = (0..1_000_000).map(|i| i as f64).collect();

let squares: Vec<f64> = parallel_map(&data, |&x| x * x)?;
let total: f64 = parallel_reduce(&data, 0.0, |acc, &x| acc + x)?;
```

### Lock-free queue

```rust
use scirs2_core::concurrent::LockFreeQueue;

let queue: LockFreeQueue<i32> = LockFreeQueue::new();
queue.push(42);
let val = queue.pop(); // Some(42)
```

### ML pipeline

```rust
use scirs2_core::ml_pipeline::{Pipeline, Transformer, Predictor};

// Build a pipeline: StandardScaler -> LinearModel
let pipeline = Pipeline::builder()
    .add_transformer(StandardScaler::new())
    .set_predictor(LinearModel::load("model.bin")?)
    .build();

let predictions = pipeline.predict(&features)?;
```

## Feature Flags

| Feature | Description |
|---------|-------------|
| `validation` | Data validation helpers (`check_finite`, schema validation) |
| `simd` | SIMD-accelerated array operations |
| `parallel` | Multi-threaded parallel processing via Rayon |
| `gpu` | GPU memory management and kernel abstractions |
| `cuda` | NVIDIA CUDA backend (requires `gpu`) |
| `memory_management` | Advanced memory utilities (arena, slab, pool) |
| `array_protocol` | Extensible unified array interface |
| `logging` | Structured logging integration |
| `profiling` | Performance profiling stubs |
| `std` | Standard library support (enabled by default; disable for `no_std`) |
| `all` | All stable features |

## Links

- [API Documentation](https://docs.rs/scirs2-core)
- [SciRS2 Repository](https://github.com/cool-japan/scirs)
- [SciRS2 POLICY](../SCIRS2_POLICY.md)

## License

Licensed under the Apache License 2.0. See [LICENSE](../LICENSE) for details.
