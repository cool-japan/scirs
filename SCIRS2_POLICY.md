# SciRS2 Ecosystem Policy

## Core Architectural Principles

This document establishes the foundational policies for the SciRS2 scientific computing ecosystem to ensure consistency, maintainability, and architectural integrity across all crates.

## Table of Contents

### Part I: Ecosystem Architecture
1. [Overview](#overview)
2. [Dependency Abstraction Policy](#dependency-abstraction-policy)
3. [Core Architectural Principles](#core-architectural-principles-1)
4. [Implementation Guidelines](#implementation-guidelines)
5. [Migration Strategy](#migration-strategy)

### Part II: Technical Policies
6. [SIMD Operations Policy](#simd-operations-policy)
7. [GPU Operations Policy](#gpu-operations-policy)
8. [Parallel Processing Policy](#parallel-processing-policy)
9. [BLAS Operations Policy](#blas-operations-policy)
10. [Platform Detection Policy](#platform-detection-policy)
11. [Performance Optimization Policy](#performance-optimization-policy)
12. [Error Handling Policy](#error-handling-policy)
13. [Memory Management Policy](#memory-management-policy)

### Part III: Implementation
14. [Refactoring Guidelines](#refactoring-guidelines)
15. [Examples](#examples)
16. [Enforcement](#enforcement)
17. [Benefits](#benefits)

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

## Dependency Abstraction Policy

### Core Principle: Layered Abstraction Architecture

The SciRS2 ecosystem follows a strict layered architecture where only the core crate can use external dependencies directly, while all other crates must use SciRS2-Core abstractions.

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
ndarray-linalg = { workspace = true }    # ❌ Use scirs2-core instead
num-traits = { workspace = true }        # ❌ Use scirs2-core instead
num-complex = { workspace = true }       # ❌ Use scirs2-core instead
num-integer = { workspace = true }       # ❌ Use scirs2-core instead
nalgebra = { workspace = true }          # ❌ Use scirs2-core instead
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
// Includes: ndarray-linalg, ndarray-stats, ndarray-npy when array feature enabled

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
| `ndarray-linalg` | `scirs2_core::ndarray` | Via `array` feature |
| `num-traits` | `scirs2_core::numeric` | All traits |
| `num-complex` | `scirs2_core::numeric` | Complex numbers |
| `num-integer` | `scirs2_core::numeric` | Integer traits |
| `nalgebra` | `scirs2_core::linalg` | When needed |
```

### Exception: SciRS2-Core Foundation Layer

**Only `scirs2-core` may use external dependencies directly:**
- ✅ `rand`, `ndarray`, `num_complex`, `nalgebra`, etc.
- ✅ Direct integration with external scientific computing libraries
- ✅ Platform-specific optimizations and SIMD operations

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

// ✅ Correct - SciRS2-Core unified abstractions (v0.1.0-beta.4+)
use scirs2_core::random::*;
use scirs2_core::ndarray::*;

let mut rng = thread_rng();  // Now available through scirs2_core
let beta = RandBeta::new(2.0, 5.0)?;  // All distributions available
let arr = array![[1, 2], [3, 4]];  // array! macro works
let slice = arr.slice(s![.., 0]);  // s! macro works
```

## Migration Strategy

### For Beta.3 and Beyond

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

### Mandatory Rules

1. **ALWAYS use `scirs2-core::gpu` module** for all GPU operations
2. **NEVER implement direct CUDA/OpenCL/Metal kernels** in modules
3. **NEVER make direct GPU API calls** outside of core
4. **ALWAYS register GPU kernels** in the core GPU kernel registry

### GPU Backend Support

The core GPU module provides unified abstractions for:
- CUDA
- ROCm
- WebGPU
- Metal
- OpenCL

### Usage Pattern

```rust
use scirs2_core::gpu::{GpuDevice, GpuKernel};

// CORRECT - Uses core GPU abstractions
let device = GpuDevice::default()?;
let kernel = device.compile_kernel(KERNEL_SOURCE)?;

// INCORRECT - Direct CUDA usage
// use cuda_sys::*;  // FORBIDDEN in modules
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

- macOS: Accelerate Framework (default)
- Linux/Windows: OpenBLAS (default)
- Intel MKL (optional)
- Netlib (fallback)

### Module Dependencies

```toml
# CORRECT - Module Cargo.toml
[dependencies]
scirs2-core = { workspace = true, features = ["blas"] }

# INCORRECT - Direct BLAS dependency
# openblas-src = "0.10"  # FORBIDDEN
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
    // Use GPU implementation from core
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
2. **GPU implementations** - Centralize all GPU kernels in `scirs2-core::gpu`
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

## Recent Enhancements (v0.1.0-beta.4)

### Stable Core Abstractions
As of beta.4, `scirs2_core::random` provides:
- ✅ All `rand_distr` distributions (Beta, Cauchy, ChiSquared, FisherF, LogNormal, StudentT, Weibull, etc.)
- ✅ Unified distribution interface with enhanced sampling
- ✅ Full compatibility with ToRSh and other ecosystem projects
- ✅ Production-ready stability and performance

### Unified NDArray Module
As of beta.4, `scirs2_core::ndarray` provides:
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
- **Effective Date**: SciRS2 v0.1.0-RC.1
- **Last Updated**: 2025-10-03
- **Status**: Active - Migration in Progress

## Current Status (v0.1.0-RC.1)

### Policy Compliance Audit (2025-10-03)

**Investigation Results:**
- Total Non-Core Crates: 23
- **Policy Violations: 23/23 (100%)** ❌

### Violation Breakdown

| Violation Type | Affected Crates | Severity |
|---------------|----------------|----------|
| Direct `ndarray` import | 23/23 (100%) | 🔴 Critical |
| Direct `num-traits`, `num-complex` | 22/23 (96%) | 🔴 Critical |
| Direct `rand` import | 21/23 (91%) | 🔴 Critical |
| Direct `rand_distr` import | 15/23 (65%) | 🟡 High |
| Direct `ndarray-rand` import | 5/23 (22%) | 🟡 Medium |

### Migration Roadmap

#### Phase 1: Core Infrastructure (v0.1.0-RC.1) ✅
1. ✅ Enhanced `scirs2-core::ndarray` with full ecosystem (array feature)
2. ✅ Policy documentation updated with Cargo.toml guidelines
3. ✅ Dependency mapping table completed

#### Phase 2: High Priority Crates (v0.1.0-RC.1) ✅
1. ✅ scirs2-linalg - Linear algebra foundation
2. ✅ scirs2-stats - Statistical computing foundation
3. ✅ scirs2-ndimage - Image processing foundation
4. ✅ scirs2-optimize - Optimization algorithms
5. ✅ scirs2-integrate - Integration and ODEs
6. ✅ scirs2-interpolate - Interpolation methods

#### Phase 3: Core Numerical Modules (v0.1.0-RC.1) ✅
7. ✅ scirs2-special - Special functions
8. ✅ scirs2-fft - Fast Fourier Transform
9. ✅ scirs2-signal - Signal processing
10. ✅ scirs2-sparse - Sparse matrices
11. ✅ scirs2-spatial - Spatial algorithms

#### Phase 4: Advanced & ML Modules (v0.1.0-RC.1) ✅
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

**Status**: All 23 crates are now POLICY-compliant (100% complete as of v0.1.0-RC.1)

### Enforcement Strategy

Starting from v0.1.0-RC.2:
1. **New Code**: Must follow policy from day one
2. **Existing Code**: Gradual migration with priority order
3. **CI Checks**: Automated policy compliance checks (planned)
4. **Documentation**: All examples updated to show correct patterns

---

*This policy ensures the SciRS2 ecosystem remains maintainable, consistent, and high-performance as it scales to support the broader scientific computing community.*