# SciRS2 Ecosystem Policy

## Core Architectural Principles

This document establishes the foundational policies for the SciRS2 scientific computing ecosystem to ensure consistency, maintainability, and architectural integrity across all crates.

## Table of Contents

### Part I: Ecosystem Architecture
1. [Overview](#overview)
2. [Pure Rust Migration (v0.1)](#pure-rust-migration-v010)
3. [Dependency Abstraction Policy](#dependency-abstraction-policy)
4. [Core Architectural Principles](#core-architectural-principles-1)
5. [Implementation Guidelines](#implementation-guidelines)
6. [Migration Strategy](#migration-strategy)

### Part II: Technical Policies
7. [SIMD Operations Policy](#simd-operations-policy)
8. [GPU Operations Policy](#gpu-operations-policy)
9. [Parallel Processing Policy](#parallel-processing-policy)
10. [BLAS Operations Policy](#blas-operations-policy)
11. [Platform Detection Policy](#platform-detection-policy)
12. [Performance Optimization Policy](#performance-optimization-policy)
13. [Error Handling Policy](#error-handling-policy)
14. [Memory Management Policy](#memory-management-policy)

### Part III: Implementation
15. [Refactoring Guidelines](#refactoring-guidelines)
16. [Examples](#examples)
17. [Enforcement](#enforcement)
18. [Benefits](#benefits)

---

## Part I: Ecosystem Architecture

## Overview

The scirs2-core crate serves as the central hub for all common functionality, optimizations, and abstractions used across SciRS2 modules. This centralized approach ensures:

- **Consistency**: All modules use the same optimized implementations
- **Maintainability**: Updates and improvements are made in one place
- **Performance**: Optimizations are available to all modules
- **Portability**: Platform-specific code is isolated in core
- **Version Control**: Only core manages external dependency versions
- **Type Safety**: Prevents mixing external types with SciRS2 types

## Pure Rust Migration (v0.3.1)

**Major architectural changes in v0.3.1 (December 2025):**

### OxiBLAS Migration - Pure Rust BLAS/LAPACK

**REMOVED Dependencies:**
- ❌ `ndarray-linalg` - Replaced with scirs2-linalg independent implementation
- ❌ `openblas-src` / `blas-src` / `lapack-src` - System BLAS libraries
- ❌ `accelerate-src` - macOS Accelerate Framework bindings
- ❌ `intel-mkl-src` - Intel MKL bindings
- ❌ `netlib-src` - Netlib reference implementation

**ADDED Dependencies:**
- ✅ `oxiblas-ndarray` v0.3.1+ - Pure Rust ndarray integration
- ✅ `oxiblas-blas` v0.3.1+ - Pure Rust BLAS implementation
- ✅ `oxiblas-lapack` v0.3.1+ - Pure Rust LAPACK implementation (supports Complex<f64>)

**Benefits:**
- 🚀 **Zero System Dependencies** - No need to install OpenBLAS, MKL, or system BLAS
- 🔧 **Easy Cross-Compilation** - Pure Rust works on all platforms
- 📦 **Simplified Builds** - No C/Fortran compiler required
- 🔒 **Complete Control** - Full Rust ecosystem integration
- ⚡ **SIMD Optimized** - Performance competitive with native BLAS

### Oxicode Migration - SIMD-Optimized Serialization

**REMOVED Dependencies:**
- ❌ `bincode` - Generic binary serialization

**ADDED Dependencies (COOLJAPAN Policy):**
- ✅ `oxicode` v0.3.1+ - SIMD-optimized binary serialization
- ✅ `oxicode_derive` - Derive macros for custom types

**Benefits:**
- ⚡ **SIMD Acceleration** - Up to 4x faster than bincode
- 🎯 **Scientific Data Optimized** - Specialized for numeric arrays
- 🔒 **Type Safe** - Compile-time serialization verification

## Dependency Abstraction Policy

### Core Principle: Layered Abstraction Architecture

The SciRS2 ecosystem follows a strict layered architecture where only the core crate can use external dependencies directly, while all other crates must use SciRS2-Core abstractions.

> **GPU exception (0.6.x):** The `oxicuda-*` crate family is a sanctioned per-crate
> **direct** dependency for the NVIDIA-only CUDA performance path. GPU acceleration is
> decentralized in 0.6.x, so a non-core crate MAY depend on `oxicuda-*` directly in its
> own `Cargo.toml` (behind an OFF-by-default, runtime-probed `cuda` feature) rather than
> routing CUDA through `scirs2-core`. See the *GPU Operations Policy* for the full rules.
> This carve-out applies only to `oxicuda-*`; all other external deps (`rand`, `ndarray`,
> num-traits, BLAS, …) remain core-only.

### Policy: No Direct External Dependencies in Non-Core Crates

**Applies to:** All SciRS2 crates except `scirs2-core`
- `scirs2-linalg`, `scirs2-stats`, `scirs2-optimize`, etc.
- All tests, examples, benchmarks in all crates (including scirs2-core)
- All integration tests and documentation examples

#### Prohibited Direct Dependencies in Cargo.toml:
```toml
# ❌ FORBIDDEN in non-core crates (scirs2-linalg, scirs2-stats, etc.)
[dependencies]
rand = { workspace = true }              # ❌ Use scirs2-core instead
rand_distr = { workspace = true }        # ❌ Use scirs2-core instead
rand_core = { workspace = true }         # ❌ Use scirs2-core instead
rand_chacha = { workspace = true }       # ❌ Use scirs2-core instead
rand_pcg = { workspace = true }          # ❌ Use scirs2-core instead
ndarray = { workspace = true }           # ❌ Use scirs2-core instead
ndarray-rand = { workspace = true }      # ❌ Use scirs2-core instead
ndarray-stats = { workspace = true }     # ❌ Use scirs2-core instead
ndarray-npy = { workspace = true }       # ❌ Use scirs2-core instead
ndarray-linalg = { workspace = true }    # ❌ REMOVED v0.3.1 - scirs2-linalg independent implementation
num-traits = { workspace = true }        # ❌ Use scirs2-core instead
num-complex = { workspace = true }       # ❌ Use scirs2-core instead
num-integer = { workspace = true }       # ❌ Use scirs2-core instead
nalgebra = { workspace = true }          # ❌ Use scirs2-core instead
bincode = { workspace = true }           # ❌ REMOVED v0.3.1 - Use oxicode instead (COOLJAPAN Policy)
openblas-src = { workspace = true }      # ❌ REMOVED v0.3.1 - Use OxiBLAS instead
blas-src = { workspace = true }          # ❌ REMOVED v0.3.1 - Use OxiBLAS instead
lapack-src = { workspace = true }        # ❌ REMOVED v0.3.1 - Use OxiBLAS instead
```

#### Required Core Dependency in Cargo.toml:
```toml
# ✅ REQUIRED in all non-core crates
[dependencies]
scirs2-core = { workspace = true, features = ["array", "random"] }
# All external dependencies accessed through scirs2-core
```

#### Prohibited Direct Imports in Code:
```rust
// ❌ FORBIDDEN in non-core crates
use rand::*;
use rand::Rng;
use rand::seq::SliceRandom;
use rand_distr::{Beta, Normal, StudentT};  // Use scirs2_core::random instead
use ndarray::*;
use ndarray::{Array, Array1, Array2};
use ndarray::{array, s};  // Macros now available through scirs2_core
use num_complex::Complex;
use num_traits::*;
// etc.
```

#### Required SciRS2-Core Abstractions:
```rust
// ✅ REQUIRED in non-core crates and all tests / examples including core crate

// === Random Number Generation ===
use scirs2_core::random::*;           // Complete rand + rand_distr functionality
// Includes: thread_rng, Rng, SliceRandom, etc.
// All distributions: Beta, Cauchy, ChiSquared, Normal, StudentT, Weibull, etc.

// === Array Operations ===
use scirs2_core::ndarray::*;          // Complete ndarray ecosystem
// Includes: Array, Array1, Array2, ArrayView, array!, s!, azip! macros
// Includes: ndarray-rand, ndarray-stats, ndarray-npy when array feature enabled
// NOTE: ndarray-linalg removed v0.3.1 - scirs2-linalg provides independent implementation

// === Numerical Traits ===
use scirs2_core::numeric::*;          // num-traits, num-complex, num-integer
// Includes: Float, Zero, One, Num, Complex, etc.

// === Advanced Types ===
use scirs2_core::array::*;            // Scientific array types (MaskedArray, RecordArray)
use scirs2_core::linalg::*;           // Linear algebra (nalgebra when needed)
```

### Complete Dependency Mapping

| External Crate | SciRS2-Core Module | Note |
|----------------|-------------------|------|
| `rand` | `scirs2_core::random` | Full functionality |
| `rand_distr` | `scirs2_core::random` | All distributions |
| `rand_core` | `scirs2_core::random` | Core traits |
| `rand_chacha` | `scirs2_core::random` | ChaCha RNG |
| `rand_pcg` | `scirs2_core::random` | PCG RNG |
| `ndarray` | `scirs2_core::ndarray` | Full functionality |
| `ndarray-rand` | `scirs2_core::ndarray` | Via `array` feature |
| `ndarray-stats` | `scirs2_core::ndarray` | Via `array` feature |
| `ndarray-npy` | `scirs2_core::ndarray` | Via `array` feature |
| ~~`ndarray-linalg`~~ | N/A | **REMOVED** - scirs2-linalg provides independent implementation |
| `num-traits` | `scirs2_core::numeric` | All traits |
| `num-complex` | `scirs2_core::numeric` | Complex numbers |
| `num-integer` | `scirs2_core::numeric` | Integer traits |
| `nalgebra` | `scirs2_core::linalg` | When needed |
| `oxiblas-*` | `scirs2_core::linalg` | Pure Rust BLAS/LAPACK (v0.3.1+) |
| ~~`bincode`~~ | N/A | **REPLACED** by `oxicode` (SIMD-optimized) |
| `oxicode` | Direct usage | COOLJAPAN Policy - Pure Rust serialization |
```

### Exception: SciRS2-Core Foundation Layer

**Only `scirs2-core` may use external dependencies directly:**
- ✅ `rand`, `ndarray`, `num_complex`, `nalgebra`, etc.
- ✅ Direct integration with external scientific computing libraries
- ✅ Platform-specific optimizations and SIMD operations

**GPU carve-out (0.6.x):** the `oxicuda-*` family is the one external dependency that
non-core crates MAY depend on directly (per-crate, OFF-by-default `cuda` feature,
runtime-probed, NVIDIA-only). See the *GPU Operations Policy*.

### Benefits of This Architecture

1. **Consistent APIs**: All SciRS2 crates use the same interfaces
2. **Version Control**: Only core manages external dependency versions
3. **Type Safety**: Prevents mixing external types with SciRS2 types
4. **Maintainability**: Changes to external APIs only affect core
5. **Performance**: Core can optimize all external library usage
6. **Documentation**: Single source of truth for API documentation

## Implementation Guidelines

### For Developers

When writing code in non-core SciRS2 crates:

1. **Never import external crates directly**
2. **Always use SciRS2-Core re-exports**
3. **Use CoreRandom instead of rand::Rng**
4. **Use SciRS2 array types instead of ndarray directly**
5. **Follow existing patterns in other SciRS2 crates**

### For Tests and Examples

```rust
// ❌ Wrong - direct external usage
use rand::thread_rng;
use rand_distr::{Beta, Normal};
use ndarray::{Array2, array, s};
let mut rng = thread_rng();
let arr = array![[1, 2], [3, 4]];
let slice = arr.slice(s![.., 0]);

// ✅ Correct - SciRS2-Core unified abstractions (v0.3.1+)
use scirs2_core::random::*;
use scirs2_core::ndarray::*;

let mut rng = thread_rng();  // Now available through scirs2_core
let beta = RandBeta::new(2.0, 5.0)?;  // All distributions available
let arr = array![[1, 2], [3, 4]];  // array! macro works
let slice = arr.slice(s![.., 0]);  // s! macro works
```

## Migration Strategy

### For v0.3.1 and Beyond

1. **Phase 1**: Document policy (✅ Completed - This document)
2. **Phase 2**: Systematic refactoring of all non-core code (✅ Completed - All 23 crates)
3. **Phase 3**: Update CLAUDE.md and documentation (⏳ In Progress)
4. **Phase 4**: Establish CI checks to enforce policy (⏳ Planned)
5. **Phase 5**: Monitor and maintain compliance (⏳ Ongoing)

---

## Part II: Technical Policies

## SIMD Operations Policy

### Mandatory Rules

1. **ALWAYS use `scirs2-core::simd_ops::SimdUnifiedOps` trait** for all SIMD operations
2. **NEVER implement custom SIMD** code in individual modules
3. **NEVER use direct SIMD libraries** (wide, packed_simd, std::arch) in modules
4. **ALWAYS provide scalar fallbacks** through the unified trait

### Required Usage Pattern

```rust
use scirs2_core::simd_ops::SimdUnifiedOps;

// CORRECT - Uses unified SIMD operations
let result = f32::simd_add(&a.view(), &b.view());
let dot_product = f64::simd_dot(&x.view(), &y.view());

// INCORRECT - Direct SIMD implementation
// use wide::f32x8;  // FORBIDDEN in modules
// let vec = f32x8::new(...);  // FORBIDDEN
```

### Available SIMD Operations

All operations are available through the `SimdUnifiedOps` trait:

- `simd_add`, `simd_sub`, `simd_mul`, `simd_div` - Element-wise operations
- `simd_dot` - Dot product
- `simd_gemv` - Matrix-vector multiplication
- `simd_gemm` - Matrix-matrix multiplication
- `simd_norm` - L2 norm
- `simd_max`, `simd_min` - Element-wise min/max
- `simd_scalar_mul` - Scalar multiplication
- `simd_sum`, `simd_mean` - Reductions
- `simd_fma` - Fused multiply-add
- `simd_transpose` - Matrix transpose
- `simd_abs`, `simd_sqrt` - Mathematical operations

## GPU Operations Policy

> **Revised for the 0.6.x series (decentralized GPU).** Prior to 0.6.0 this policy
> mandated that *all* GPU work be routed through `scirs2-core::gpu`. That single-hub
> rule is **superseded**. GPU acceleration is now **decentralized**: each crate owns
> its GPU story, the CPU path remains the per-crate source of truth, and `scirs2-core`
> provides an **optional** portable backend rather than a mandatory gateway.
>
> As of 0.6.x, **`scirs2-core` ships no CUDA backend at all.** The `cudarc` dependency
> and `gpu/backends/cuda.rs` were removed (a Pure-Rust win); `GpuBackend::Cuda` survives
> only as an enum tag whose context constructor returns an honest error pointing at the
> per-crate `oxicuda-*` `cuda` features. The portable wgpu backend
> (`GpuNdarray`/`WebGPUContext`) **remains** in core — only CUDA was retired — and
> gradually phasing out that wgpu hub is still future work.

### Two complementary GPU stories

SciRS2 supports two GPU paths that are **complementary, not alternatives**. A crate may
ship either, both, or neither — but always atop a CPU implementation.

| Story | Backend | Crates / feature | Devices | Precision | Routing |
|-------|---------|------------------|---------|-----------|---------|
| **Portability** | wgpu / WebGPU | `scirs2-core::gpu` (`GpuNdarray`) **or** a crate's own wgpu | macOS, browser, AMD, Intel, NVIDIA | f32 only | core (optional) or per-crate |
| **Performance** | `oxicuda-*` | per-crate **direct** path dep | NVIDIA only | f32 + f64 | **never** through core |

`oxicuda-*` does **not** replace wgpu; it is the high-performance NVIDIA-only path that
complements the portable wgpu backend. Pick wgpu when portability/browser/non-NVIDIA
matters; reach for `oxicuda-*` when you need NVIDIA throughput and/or f64.

### Mandatory Rules (0.6.x)

1. **CPU is always the source of truth.** Every GPU-accelerated function MUST have a
   CPU implementation in the same crate that is correct and complete on its own.
2. **CUDA = `oxicuda-*`, as a per-crate DIRECT dependency.** A crate that wants NVIDIA
   acceleration depends on the relevant `oxicuda-*` crates directly in its own
   `Cargo.toml`. This GPU path is **NOT** routed through `scirs2-core`. (This is the one
   sanctioned exception to the "only core uses external deps" rule — see
   *Dependency Abstraction Policy*.)
3. **`scirs2-core::gpu` / `GpuNdarray` is the OPTIONAL portable wgpu backend.** It is no
   longer the mandatory hub. Crates MAY use it for portable f32 wgpu work, and crates
   that already maintain their own wgpu kernels are explicitly sanctioned (see below).
4. **All GPU features are OFF by default and runtime-probed.** No GPU feature may be in a
   crate's `default` feature set. At runtime, code MUST probe for an actual device/adapter
   before dispatching and fall back to CPU when none is present.
5. **macOS stays green.** Because `oxicuda-*` is NVIDIA-only and macOS has no NVIDIA path,
   `cuda` features are OFF by default and MUST remain compile-only on macOS — a default
   `cargo build`/`cargo test` on macOS MUST NOT require CUDA. Portable GPU work on macOS
   goes through wgpu (Metal backend).
6. **Honesty is mandatory — never fabricate GPU results.** When no device/adapter is
   available, or a path is unimplemented, return an explicit error
   (`BackendNotAvailable` / `NotImplemented`) or fall back to the real CPU path. NEVER
   return fabricated, simulated, or hard-coded "GPU" numbers, and NEVER report a GPU
   dispatch that did not occur.

### Standard feature naming

As of the 0.6.x series, the earlier feature-name sprawl (`gpu`, `wgpu`, `wgpu_rbf`,
`gpu_wgpu`, `wgpu_fft`, `wgpu_kernels`, `gpu_kdtree`, `gpu_fem`, `cuda`, …) has been
**standardized** — the portability backends are unified under `wgpu`, parallel to the
per-crate `cuda`:

| Feature | Meaning |
|---------|---------|
| `cuda`  | NVIDIA-only acceleration via direct `oxicuda-*` deps (f32 + f64). |
| `wgpu`  | Portable f32 acceleration via wgpu/WebGPU (own kernels or core's `GpuNdarray`). |
| `gpu`   | Umbrella convenience feature; enables whichever of `cuda`/`wgpu` the crate offers. |

All three are OFF by default. The 0.6.x phase-out below migrated every real (`dep:wgpu`)
portability feature in the ecosystem to this standard: `scirs2-core`'s `wgpu_backend`, the
per-crate `gpu`/`gpu_wgpu`/`wgpu_fft` features (vision, graph, optimize, datasets, stats, fft),
special's `wgpu_kernels`, and interpolate's `wgpu_rbf` are now all `wgpu`, so each exposes a
consistent `cuda` + `wgpu` pair. The only non-`wgpu` GPU flags left are the two empty bare-flag
placeholders `scirs2-integrate/gpu_fem` and `scirs2-interpolate/gpu_kdtree`, which pull no
`dep:wgpu` acceleration and gate orthogonal paths.

### Reference implementation

**`scirs2-fft` is the pilot / reference implementation** for the decentralized model
(Phase 0). New GPU work in other crates should mirror its structure: CPU source of truth,
`cuda` feature wiring direct `oxicuda-*` deps with a runtime probe + CPU fallback, optional
portable `wgpu` path, macOS-green defaults, and honest no-device behavior.

### GPU Decentralization — 0.6.x phases

The single-hub model has been dismantled across four phases, all landed in the 0.6.x
series. Core's portable wgpu `GpuNdarray` deliberately **stays in place** — only the CUDA
hub was removed — and gradually phasing out the wgpu hub remains future work.

- **Phase 0 — Pilot (`scirs2-fft`) ✅.** Stood up the reference: per-crate direct
  `oxicuda-*` `cuda` feature (oxicuda-fft), runtime probe, CPU fallback, standardized
  features, macOS-green.
- **Phase 1 — Sanction + extend ✅.** Added off-by-default direct `oxicuda-*` `cuda` paths
  to six crates: `scirs2-fft` (oxicuda-fft), `scirs2-symbolic` (oxicuda-ptx custom
  kernels), `scirs2-interpolate` (oxicuda-blas + oxicuda-solver), `scirs2-special` and
  `scirs2-stats` (oxicuda-ptx custom kernels), and `scirs2-graph` (oxicuda-sparse). No
  requirement to route through core.
- **Phase 2 — Migrate core-coupled crates ✅.** Gave the four crates that consumed core's
  `GpuNdarray` their own per-crate `cuda` path plus a `gpu_cuda` module: `scirs2-linalg`
  (oxicuda-blas + oxicuda-solver), `scirs2-optimize` (oxicuda-blas GEMV), `scirs2-datasets`
  (oxicuda-blas GEMV), and `scirs2-vision` (oxicuda-dnn conv2d). Added `oxicuda-dnn` to
  `[workspace.dependencies]`.
- **Phase 3 — Retire core's CUDA backend ✅.** Removed `scirs2-core`'s own cudarc-based
  CUDA backend: deleted `gpu/backends/cuda.rs`, dropped the `cudarc` dependency (a
  Pure-Rust win), and removed core's `cuda` / `array_protocol_cuda` features.
  `GpuBackend::Cuda` remains only as an enum tag whose context constructor returns an
  honest error pointing at the per-crate `oxicuda-*` `cuda` features; core's portable wgpu
  `GpuNdarray`/`WebGPUContext` is retained. Tidy: made `scirs2-spatial`'s GPU docs honest
  (its `gpu_accel` path is a CPU-SIMD fallback, not GPU) and removed dead GPU feature
  aliases (`opencl`/`metal`/`oneapi`) from `scirs2-cluster`.

Net: ten crates now ship direct per-crate `oxicuda-*` CUDA paths — `scirs2-fft`,
`scirs2-symbolic`, `scirs2-interpolate`, `scirs2-special`, `scirs2-stats`, `scirs2-graph`,
`scirs2-linalg`, `scirs2-optimize`, `scirs2-datasets`, and `scirs2-vision` — while
`scirs2-core` no longer ships any CUDA backend.

### Architecture boundary — what decentralization did *not* change

Decentralization removed **CUDA (the `cudarc` backend)** from `scirs2-core`; it did
**not** remove the portable wgpu layer. Stated plainly:

- **`scirs2-core` owns the wgpu/WebGPU portability layer** — `GpuNdarray`,
  `GpuContext`, `GpuBackend`, `WebGPUContext` (f32, cross-platform, CPU fallback). This
  is a **shared foundation** that leaf crates (`scirs2-optimize`, `scirs2-datasets`,
  `scirs2-vision`, `scirs2-linalg`, …) MAY and DO depend on. That dependency is **by
  design and is NOT "core aggregating GPU"** — it is leaf crates building on a common
  portability primitive, exactly as they build on core's `ndarray`/`numeric` re-exports.
- **`oxicuda-*` owns the per-crate CUDA performance path** — NVIDIA-only, f64, real
  CUDA, no fallback, behind each crate's OFF-by-default `cuda` feature, wired as a direct
  path dep in that crate's `gpu_cuda.rs`. This path is **never** routed through core.
- Therefore "GPU decentralization" means **CUDA left core**, not wgpu. Leaf crates
  keeping their dependency on core's wgpu portability layer is expected and correct.

### Usage Pattern

```rust
// PERFORMANCE path — NVIDIA-only, direct oxicuda-* dep, f32 + f64.
// In the crate's own Cargo.toml (NOT via scirs2-core):
//   [dependencies]
//   oxicuda = { workspace = true, optional = true }
//   [features]
//   cuda = ["dep:oxicuda"]   # OFF by default; NVIDIA-only; runtime-probed
#[cfg(feature = "cuda")]
fn run_cuda(data: &[f64]) -> Result<Vec<f64>, Error> {
    // Probe for a real device first; fall back to CPU when absent.
    if !oxicuda::device_available() {
        return cpu_impl(data); // honest fallback — never fabricate
    }
    oxicuda_impl(data)
}

// PORTABILITY path — optional wgpu via core's GpuNdarray (f32 only).
#[cfg(feature = "wgpu")]
use scirs2_core::gpu::GpuNdarray;

// CPU remains the source of truth and is always present.
fn cpu_impl(data: &[f64]) -> Result<Vec<f64>, Error> { /* ... */ }
```

## Parallel Processing Policy

### Mandatory Rules

1. **ALWAYS use `scirs2-core::parallel_ops`** for all parallel operations
2. **NEVER add direct `rayon` dependency** to module Cargo.toml files
3. **ALWAYS import via `use scirs2_core::parallel_ops::*`**
4. **NEVER use `rayon::prelude::*` directly** in modules

### Required Usage Pattern

```rust
// CORRECT - Uses core parallel abstractions
use scirs2_core::parallel_ops::*;

let results: Vec<i32> = (0..1000)
    .into_par_iter()
    .map(|x| x * x)
    .collect();

// INCORRECT - Direct Rayon usage
// use rayon::prelude::*;  // FORBIDDEN in modules
```

### Features Provided

The `parallel_ops` module provides:

- **Full Rayon functionality** when `parallel` feature is enabled
- **Sequential fallbacks** when `parallel` feature is disabled
- **Helper functions**:
  - `par_range(start, end)` - Create parallel iterator from range
  - `par_chunks(slice, size)` - Process slices in parallel chunks
  - `par_scope(closure)` - Execute in parallel scope
  - `par_join(a, b)` - Execute two closures in parallel
- **Runtime detection**:
  - `is_parallel_enabled()` - Check if parallel processing is available
  - `num_threads()` - Get number of threads for parallel operations

### Module Dependencies

```toml
# CORRECT - Module Cargo.toml
[dependencies]
scirs2-core = { workspace = true, features = ["parallel"] }

# INCORRECT - Direct Rayon dependency
# rayon = { workspace = true }  # FORBIDDEN
```

## BLAS Operations Policy

### Mandatory Rules

1. **ALL BLAS operations go through `scirs2-core`**
2. **NEVER add direct BLAS dependencies** to individual modules
3. **Backend selection is handled by core's platform configuration**
4. **Use feature flags through core** for BLAS backend selection

### Supported BLAS Backends

**v0.3.1+ (Current):**
- **All Platforms: OxiBLAS (Pure Rust BLAS/LAPACK)** - Default and recommended
  - No system dependencies required
  - Cross-compilation friendly
  - Complete Rust ecosystem integration

**Legacy (Removed in v0.3.1):**
- ~~macOS: Accelerate Framework~~ **REMOVED**
- ~~Linux/Windows: OpenBLAS~~ **REMOVED**
- ~~Intel MKL~~ **REMOVED**
- ~~Netlib~~ **REMOVED**

### Module Dependencies

```toml
# CORRECT - Module Cargo.toml (v0.3.1+)
[dependencies]
scirs2-core = { workspace = true, features = ["linalg"] }
# OxiBLAS accessed through scirs2-core/linalg feature

# INCORRECT - Direct BLAS dependency (FORBIDDEN)
# openblas-src = "0.10"  # FORBIDDEN
# blas-src = "0.10"      # FORBIDDEN
# lapack-src = "0.10"    # FORBIDDEN
# accelerate-src = "0.4" # FORBIDDEN - Use OxiBLAS instead
```

## Platform Detection Policy

### Mandatory Rules

1. **ALWAYS use `scirs2-core::simd_ops::PlatformCapabilities`** for capability detection
2. **NEVER implement custom CPU feature detection**
3. **NEVER duplicate platform detection code**

### Usage Pattern

```rust
use scirs2_core::simd_ops::PlatformCapabilities;

// CORRECT - Uses core platform detection
let caps = PlatformCapabilities::detect();
if caps.simd_available {
    // Use SIMD path
}

// INCORRECT - Custom detection
// if is_x86_feature_detected!("avx2") {  // FORBIDDEN
```

### Available Capabilities

- `simd_available` - SIMD support
- `gpu_available` - GPU support
- `cuda_available` - CUDA support
- `opencl_available` - OpenCL support
- `metal_available` - Metal support (macOS)
- `avx2_available` - AVX2 instructions
- `avx512_available` - AVX512 instructions
- `neon_available` - ARM NEON instructions

## Performance Optimization Policy

### Automatic Optimization Selection

Use `scirs2-core::simd_ops::AutoOptimizer` for automatic selection:

```rust
use scirs2_core::simd_ops::AutoOptimizer;

let optimizer = AutoOptimizer::new();

// Automatically selects best implementation based on problem size
if optimizer.should_use_gpu(problem_size) {
    // Use a GPU implementation — core's portable wgpu backend, or a per-crate `cuda`/`wgpu` path
} else if optimizer.should_use_simd(problem_size) {
    // Use SIMD implementation from core
} else {
    // Use scalar implementation
}
```

### Required Core Features

Each module should enable relevant core features:

```toml
[dependencies]
scirs2-core = { workspace = true, features = ["simd", "parallel", "gpu", "blas"] }
```

## Error Handling Policy

### Mandatory Rules

1. **Base all module errors on `scirs2-core::error`**
2. **Provide proper error conversions** to/from core errors
3. **Use core validation functions** for parameter checking

### Usage Pattern

```rust
use scirs2_core::error::CoreError;
use scirs2_core::validation::{check_positive, check_finite};

// Module-specific error should derive from core
#[derive(Debug, thiserror::Error)]
pub enum ModuleError {
    #[error(transparent)]
    Core(#[from] CoreError),
    // Module-specific variants...
}

// Use core validation
check_positive(value, "parameter_name")?;
check_finite(&array)?;
```

## Memory Management Policy

### Mandatory Rules

1. **Use `scirs2-core::memory_efficient` algorithms** for large data
2. **Use `scirs2-core::cache` for caching** instead of custom solutions
3. **Follow core memory pooling strategies** when available

### Available Memory-Efficient Operations

- `chunk_wise_op` - Process large arrays in chunks
- `streaming_op` - Stream processing for very large data
- Memory pools for temporary allocations

### Caching

```rust
use scirs2_core::cache::{CacheBuilder, TTLSizedCache};

// CORRECT - Uses core caching
let cache = CacheBuilder::new()
    .max_size(100)
    .ttl(Duration::from_secs(60))
    .build();

// INCORRECT - Custom caching
// let mut cache = HashMap::new();  // Don't implement custom caching
```

---

## Part III: Implementation

## Refactoring Guidelines

When encountering code that violates these policies, follow this priority order:

1. **SIMD implementations** - Replace all custom SIMD with `scirs2-core::simd_ops`
2. **GPU implementations** - Decentralized (0.6.x): keep a CPU source of truth per crate; use direct per-crate `oxicuda-*` for the NVIDIA `cuda` path and the optional portable wgpu backend (`scirs2-core::gpu`/`GpuNdarray` or own kernels) for `wgpu` — do NOT force everything through core. See *GPU Operations Policy*.
3. **Parallel operations** - Replace direct Rayon usage with `scirs2-core::parallel_ops`
4. **Platform detection** - Replace with `PlatformCapabilities::detect()`
5. **BLAS operations** - Ensure all go through core
6. **Caching mechanisms** - Replace custom caching with core implementations
7. **Error types** - Base on core error types
8. **Validation** - Use core validation functions

## Examples

### Example 1: Matrix Operations

```rust
use scirs2_core::simd_ops::SimdUnifiedOps;
use scirs2_core::ndarray::{Array2, ArrayView2};

pub fn matrix_multiply(a: &ArrayView2<f32>, b: &ArrayView2<f32>) -> Array2<f32> {
    let mut result = Array2::zeros((a.nrows(), b.ncols()));

    // Use unified SIMD operations - no direct SIMD code
    f32::simd_gemm(1.0, a, b, 0.0, &mut result);

    result
}
```

### Example 2: Adaptive Implementation

```rust
use scirs2_core::simd_ops::{SimdUnifiedOps, AutoOptimizer};
use scirs2_core::ndarray::ArrayView1;

pub fn process_data(data: &ArrayView1<f64>) -> f64 {
    let optimizer = AutoOptimizer::new();
    let size = data.len();

    if optimizer.should_use_simd(size) {
        // Automatically uses SIMD if available
        f64::simd_sum(data) / size as f64
    } else {
        // Falls back to scalar
        data.sum() / size as f64
    }
}
```

### Example 3: Platform-Aware Code

```rust
use scirs2_core::simd_ops::PlatformCapabilities;

pub fn get_optimization_info() -> String {
    let caps = PlatformCapabilities::detect();

    format!(
        "Available optimizations: {}",
        caps.summary()
    )
}
```

### Example 4: Parallel Processing

```rust
use scirs2_core::parallel_ops::*;
use scirs2_core::ndarray::{Array1, ArrayView1};

pub fn parallel_distance_matrix(points: &ArrayView1<f64>) -> Array1<f64> {
    // Works with or without parallel feature
    let distances: Vec<f64> = (0..points.len())
        .into_par_iter()
        .map(|i| {
            // Complex computation for each point
            compute_distance(points[i])
        })
        .collect();

    Array1::from_vec(distances)
}

pub fn adaptive_processing(data: &[f64]) -> f64 {
    if is_parallel_enabled() && data.len() > 1000 {
        // Use parallel processing for large datasets
        data.into_par_iter()
            .map(|&x| x * x)
            .sum::<f64>()
    } else {
        // Use sequential for small datasets
        data.iter()
            .map(|&x| x * x)
            .sum()
    }
}
```

### Example 5: Random Number Generation

```rust
use scirs2_core::random::*;

pub fn generate_samples(n: usize) -> Vec<f64> {
    let mut rng = thread_rng();
    let dist = Normal::new(0.0, 1.0).unwrap();

    (0..n).map(|_| dist.sample(&mut rng)).collect()
}

pub fn bootstrap_sample<T: Clone>(data: &[T]) -> Vec<T> {
    let mut rng = thread_rng();
    data.choose_multiple(&mut rng, data.len())
        .cloned()
        .collect()
}
```

## Enforcement

### Automated Checks (Future)
- CI pipeline checks for prohibited imports
- `cargo deny` configuration for dependency restrictions
- Custom linting rules for SciRS2 ecosystem

### Manual Review
- All PRs must follow this policy
- Code reviews must verify SciRS2-Core usage
- Examples and tests must demonstrate proper patterns

### Current Enforcement
- Code reviews MUST check for policy compliance
- Regular audits should identify and refactor non-compliant code
- New modules MUST follow these policies from the start

## Benefits

By following these policies, we achieve:

1. **Unified Performance**: All modules benefit from optimizations
2. **Easier Maintenance**: Updates in one place benefit all modules
3. **Consistent Behavior**: Same optimizations across the ecosystem
4. **Better Testing**: Centralized testing of critical operations
5. **Improved Portability**: Platform-specific code is isolated
6. **Reduced Duplication**: No repeated implementation of common operations
7. **Version Control**: Simplified dependency management
8. **Type Safety**: Consistent types across the ecosystem

## Recent Enhancements (v0.3.1)

### Stable Core Abstractions
As of v0.3.1, `scirs2_core::random` provides:
- ✅ All `rand_distr` distributions (Beta, Cauchy, ChiSquared, FisherF, LogNormal, StudentT, Weibull, etc.)
- ✅ Unified distribution interface with enhanced sampling
- ✅ Full compatibility with ToRSh and other ecosystem projects
- ✅ Production-ready stability and performance

### Unified NDArray Module
As of v0.3.1, `scirs2_core::ndarray` provides:
- ✅ Complete ndarray functionality including all macros (`array!`, `s!`, `azip!`)
- ✅ All array types, views, and operations
- ✅ Single unified import point for all array operations
- ✅ Backward compatibility with existing `ndarray_ext`
- ✅ Enhanced documentation and examples

## Inspiration

This policy is inspired by successful production systems:
- **OxiRS**: Similar abstraction layers for graph processing
- **Other SciRS2 Projects**: Consistent architectural patterns
- **Enterprise Software**: Layered dependency management

## Questions or Clarifications

If you have questions about these policies or need clarification on specific use cases, please:

1. Check the `scirs2-core` documentation
2. Review existing implementations in other modules
3. Open an issue for discussion
4. Consult with the core team

Remember: When in doubt, use the core abstractions!

## Policy Version
- **Version**: 3.0.0 (Enhanced - Dependency Management)
- **Effective Date**: SciRS2 v0.3.1
- **Last Updated**: 2025-12-29
- **Status**: Active - Migration Complete

## Current Status (v0.3.1)

### Policy Compliance Audit

**Investigation Results:**
- Total Non-Core Crates: 23
- **Policy Violations: 23/23 (100%)** 

### Migration Roadmap

#### Phase 1: Core Infrastructure (v0.3.1) ✅
1. ✅ Enhanced `scirs2-core::ndarray` with full ecosystem (array feature)
2. ✅ Policy documentation updated with Cargo.toml guidelines
3. ✅ Dependency mapping table completed

#### Phase 2: High Priority Crates (v0.3.1) ✅
1. ✅ scirs2-linalg - Linear algebra foundation
2. ✅ scirs2-stats - Statistical computing foundation
3. ✅ scirs2-ndimage - Image processing foundation
4. ✅ scirs2-optimize - Optimization algorithms
5. ✅ scirs2-integrate - Integration and ODEs
6. ✅ scirs2-interpolate - Interpolation methods

#### Phase 3: Core Numerical Modules (v0.3.1) ✅
7. ✅ scirs2-special - Special functions
8. ✅ scirs2-fft - Fast Fourier Transform
9. ✅ scirs2-signal - Signal processing
10. ✅ scirs2-sparse - Sparse matrices
11. ✅ scirs2-spatial - Spatial algorithms

#### Phase 4: Advanced & ML Modules (v0.3.1) ✅
12. ✅ scirs2-cluster - Clustering algorithms
13. ✅ scirs2-io - Input/output utilities
14. ✅ scirs2-datasets - Sample datasets
15. ✅ scirs2-autograd - Automatic differentiation
16. ✅ scirs2-neural - Neural networks
17. ✅ scirs2-graph - Graph processing
18. ✅ scirs2-transform - Data transformation
19. ✅ scirs2-metrics - ML metrics
20. ✅ scirs2-text - Text processing
21. ✅ scirs2-vision - Computer vision
22. ✅ scirs2-series - Time series analysis
23. ✅ scirs2 - Main integration crate

**Status**: All 23 crates are now POLICY-compliant (100% complete as of v0.3.1)

### Enforcement Strategy

Starting from v0.3.1:
1. **New Code**: Must follow policy from day one
2. **Existing Code**: Gradual migration with priority order
3. **CI Checks**: Automated policy compliance checks (planned)
4. **Documentation**: All examples updated to show correct patterns

---

*This policy ensures the SciRS2 ecosystem remains maintainable, consistent, and high-performance as it scales to support the broader scientific computing community.*