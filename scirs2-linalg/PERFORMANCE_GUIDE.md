# Performance Optimization Guide for scirs2-linalg

This guide provides comprehensive recommendations for optimizing performance when using the scirs2-linalg linear algebra library. By following these guidelines, you can achieve maximum computational efficiency for your applications.

## Table of Contents

1. [Quick Performance Checklist](#quick-performance-checklist)
2. [Matrix Size Considerations](#matrix-size-considerations)
3. [Algorithm Selection](#algorithm-selection)
4. [Parallel Processing](#parallel-processing)
5. [Memory Management](#memory-management)
6. [SIMD Acceleration](#simd-acceleration)
7. [Numerical Precision vs Performance](#numerical-precision-vs-performance)
8. [Benchmarking and Profiling](#benchmarking-and-profiling)
9. [Platform-Specific Optimizations](#platform-specific-optimizations)
10. [Common Performance Pitfalls](#common-performance-pitfalls)

## Quick Performance Checklist

✅ **Essential optimizations for immediate performance gains:**

- Enable `simd` feature for SIMD acceleration
- Use parallel algorithms for matrices > 1000x1000 elements
- Choose the right algorithm for your matrix properties (SPD, symmetric, sparse, etc.)
- Minimize memory allocations in hot loops
- Use in-place operations when possible
- Enable release mode compilation (`cargo build --release`)

## Matrix Size Considerations

### Small Matrices (< 100x100)

**Recommendations:**
- Use serial algorithms to avoid parallel overhead
- Disable parallel processing: `workers = None` or set `parallel_threshold` high
- Consider stack allocation for very small matrices

```rust
use scirs2_linalg::parallel::WorkerConfig;

// For small matrices, avoid parallelization overhead
let config = WorkerConfig::new()
    .with_threshold(10000); // Only use parallel for large matrices

// Use simple algorithms for small matrices
let result = matrix_norm(&small_matrix.view(), "fro", None)?;
```

### Medium Matrices (100x100 to 1000x1000)

**Recommendations:**
- Selective use of parallel algorithms based on operation
- Monitor memory usage patterns
- Consider cache-friendly blocked algorithms

```rust
// Adaptive strategy based on matrix size
let size = matrix.nrows() * matrix.ncols();
let use_parallel = size > 50_000; // Tune this threshold

if use_parallel {
    let config = WorkerConfig::new().with_workers(4);
    parallel_matvec(&matrix.view(), &vector.view(), &config)?
} else {
    matrix.dot(&vector)
}
```

### Large Matrices (> 1000x1000)

**Recommendations:**
- Always use parallel algorithms
- Enable all performance features
- Consider out-of-core algorithms for very large matrices
- Use blocked operations to improve cache performance

```rust
// Optimal configuration for large matrices
let config = WorkerConfig::new()
    .with_workers(num_cpus::get())
    .with_threshold(1000)
    .with_chunk_size(64);

// Use memory-efficient algorithms
let (q, r) = parallel_qr(&large_matrix.view(), &config)?;
```

## Algorithm Selection

### Matrix Decompositions

| Matrix Type | Recommended Decomposition | Performance Notes |
|-------------|---------------------------|-------------------|
| **Symmetric Positive Definite** | Cholesky | ~2x faster than LU, guaranteed stability |
| **Symmetric** | Eigendecomposition (eigh) | Specialized algorithms, better numerical properties |
| **General Square** | LU with pivoting | Good general-purpose choice |
| **Overdetermined** | QR decomposition | More stable than normal equations |
| **Sparse** | Sparse-specific algorithms | Orders of magnitude faster for sparse matrices |

```rust
// Example: Choose optimal decomposition
if is_symmetric_positive_definite(&matrix) {
    // Use Cholesky - fastest for SPD matrices
    let l = cholesky(&matrix.view(), None)?;
} else if is_symmetric(&matrix) {
    // Use symmetric eigendecomposition
    let (eigenvals, eigenvecs) = eigh(&matrix.view(), None)?;
} else {
    // Use general LU decomposition
    let (p, l, u) = lu(&matrix.view(), None)?;
}
```

### Linear System Solving

| System Type | Recommended Solver | Performance Characteristics |
|-------------|-------------------|----------------------------|
| **SPD Systems** | Cholesky solve | Fastest, most stable |
| **Symmetric** | LDLT or iterative | Good for large systems |
| **General Dense** | LU solve | Standard choice |
| **Large Sparse** | Iterative (CG, GMRES) | Scalable to very large systems |
| **Ill-conditioned** | Iterative refinement | Better numerical stability |

```rust
// Adaptive solver selection
match analyze_matrix_properties(&a) {
    MatrixType::SymmetricPositiveDefinite => {
        // Fastest path for SPD matrices
        let l = cholesky(&a.view(), None)?;
        solve_triangular(&l, &b)?
    },
    MatrixType::Sparse => {
        // Use iterative solver for sparse matrices
        conjugate_gradient(&a.view(), &b.view(), 1000, 1e-10, None)?
    },
    _ => {
        // General direct solver
        solve(&a.view(), &b.view(), None)?
    }
}
```

## Parallel Processing

### Worker Configuration

```rust
use scirs2_linalg::parallel::{WorkerConfig, set_global_workers};

// Set global worker configuration
set_global_workers(Some(num_cpus::get()));

// Or use operation-specific configuration
let config = WorkerConfig::new()
    .with_workers(4)                    // Number of threads
    .with_threshold(10_000)             // Minimum size for parallelization
    .with_chunk_size(64);               // Block size for cache efficiency
```

### When to Use Parallel Algorithms

| Operation | Parallel Threshold | Expected Speedup |
|-----------|-------------------|------------------|
| **Matrix Multiplication** | > 200x200 | 2-4x with 4 cores |
| **Matrix-Vector Product** | > 10,000 elements | 2-3x with 4 cores |
| **QR Decomposition** | > 500x500 | 1.5-3x with 4 cores |
| **SVD** | > 300x300 | 2-4x with 4 cores |
| **Eigenvalue Problems** | > 400x400 | 1.5-2.5x with 4 cores |

### Thread Pool Management

```rust
use scirs2_linalg::parallel::ScopedWorkers;

// Temporarily change thread count for specific operations
{
    let _guard = ScopedWorkers::new(Some(2)); // Use 2 threads temporarily
    
    // Operations here use 2 threads
    let result = parallel_gemm(&a.view(), &b.view(), &config)?;
} // Thread count restored to previous setting
```

## Memory Management

### Minimize Allocations

```rust
// ❌ Inefficient: Multiple allocations
let mut result = Vec::new();
for i in 0..n {
    let temp = matrix.dot(&vectors[i]);
    result.push(temp);
}

// ✅ Efficient: Pre-allocate and reuse
let mut temp = Array1::zeros(n);
let mut results = Array2::zeros((n, m));
for i in 0..n {
    // Reuse pre-allocated memory
    matrix.dot_into(&vectors[i], &mut temp);
    results.row_mut(i).assign(&temp);
}
```

### In-Place Operations

```rust
// ❌ Creates new array
let result = &matrix1 + &matrix2;

// ✅ In-place operation (when possible)
let mut result = matrix1.clone();
result += &matrix2;

// ✅ Even better: mutate existing matrix
matrix1 += &matrix2;
```

### Memory Layout Optimization

```rust
use ndarray::Array2;

// Prefer standard (row-major) layout for better cache performance
let matrix = Array2::<f64>::zeros((n, m)); // Row-major by default

// For column-major algorithms, use appropriate order
let matrix_col_major = Array2::<f64>::zeros((n, m).f()); // Column-major
```

## SIMD Acceleration

### Enable SIMD Features

```toml
# In Cargo.toml
[features]
default = ["simd"]
simd = ["wide"]

[dependencies]
scirs2-linalg = { version = "0.1", features = ["simd"] }
```

### SIMD-Optimized Operations

```rust
#[cfg(feature = "simd")]
use scirs2_linalg::simd_ops::{
    gemm::{simd_gemm_f64, simd_matmul_optimized_f64},
    norms::{simd_frobenius_norm_f64, simd_vector_norm_f64},
    transpose::{simd_transpose_f64},
};

// Use SIMD operations for performance-critical code
#[cfg(feature = "simd")]
let norm = simd_frobenius_norm_f64(&matrix.view());

#[cfg(not(feature = "simd"))]
let norm = matrix_norm(&matrix.view(), "fro", None)?;
```

### Matrix Size for SIMD Efficiency

- **Minimum effective size**: 64x64 matrices
- **Optimal sizes**: Multiples of SIMD width (4 for f64, 8 for f32)
- **Alignment considerations**: Use aligned allocations when possible

## Numerical Precision vs Performance

### Precision Trade-offs

| Precision Level | Performance | Use Cases |
|----------------|-------------|-----------|
| **Single (f32)** | ~2x faster | Graphics, ML inference, approximate solutions |
| **Double (f64)** | Standard | Scientific computing, precision-critical applications |
| **Extended** | Slower | Ill-conditioned problems, high-precision requirements |

```rust
// Choose precision based on requirements
match precision_requirements {
    PrecisionLevel::Fast => {
        // Use f32 for ~2x performance improvement
        let matrix_f32: Array2<f32> = matrix.mapv(|x| x as f32);
        let result = simd_gemm_f32(&matrix_f32.view(), &matrix_f32.view())?;
    },
    PrecisionLevel::Balanced => {
        // Use f64 (standard)
        let result = parallel_gemm(&matrix.view(), &matrix.view(), &config)?;
    },
    PrecisionLevel::HighPrecision => {
        // Use iterative refinement or extended precision
        let result = solve_with_refinement(&matrix.view(), &b.view())?;
    }
}
```

### Tolerance Settings

```rust
// Adaptive tolerance based on matrix conditioning
let condition_number = cond(&matrix.view(), None, None)?;
let tolerance = if condition_number > 1e12 {
    1e-12  // Tight tolerance for ill-conditioned matrices
} else {
    1e-8   // Relaxed tolerance for well-conditioned matrices
};

let rank = matrix_rank(&matrix.view(), Some(tolerance), None)?;
```

## Benchmarking and Profiling

### Performance Measurement

```rust
use std::time::Instant;

// Benchmark matrix operations
fn benchmark_operation<F, T>(name: &str, operation: F) -> T
where
    F: FnOnce() -> T,
{
    let start = Instant::now();
    let result = operation();
    let duration = start.elapsed();
    println!("{}: {:?}", name, duration);
    result
}

// Usage
let result = benchmark_operation("Matrix multiplication", || {
    a.dot(&b)
});
```

### Memory Profiling

```rust
// Monitor memory usage patterns
fn memory_efficient_operation(matrices: &[Array2<f64>]) -> Array2<f64> {
    let mut result = Array2::zeros((n, n));
    
    // Process in chunks to control memory usage
    for chunk in matrices.chunks(chunk_size) {
        let chunk_result = process_chunk(chunk);
        result += &chunk_result;
    }
    
    result
}
```

### CPU Profiling

```rust
// Use criterion for detailed benchmarking
use criterion::{criterion_group, criterion_main, Criterion};

fn benchmark_matrix_multiply(c: &mut Criterion) {
    let a = Array2::<f64>::ones((1000, 1000));
    let b = Array2::<f64>::ones((1000, 1000));
    
    c.bench_function("matrix_multiply_1000x1000", |bench| {
        bench.iter(|| a.dot(&b))
    });
}

criterion_group!(benches, benchmark_matrix_multiply);
criterion_main!(benches);
```

## Platform-Specific Optimizations

### CPU Architecture

```rust
// Detect and optimize for specific CPU features
#[cfg(target_arch = "x86_64")]
fn optimized_operation_x86() {
    // Use AVX/AVX2 optimizations
    #[cfg(target_feature = "avx2")]
    {
        // AVX2-optimized code path
    }
}

#[cfg(target_arch = "aarch64")]
fn optimized_operation_arm() {
    // Use NEON optimizations
    #[cfg(target_feature = "neon")]
    {
        // NEON-optimized code path
    }
}
```

### Operating System

```rust
// OS-specific optimizations
#[cfg(target_os = "linux")]
fn set_cpu_affinity() {
    // Set CPU affinity for consistent performance
}

#[cfg(target_os = "windows")]
fn set_process_priority() {
    // Adjust process priority for better scheduling
}
```

## Common Performance Pitfalls

### ❌ Pitfall 1: Unnecessary Copying

```rust
// Avoid unnecessary cloning
let result = expensive_operation(matrix.clone()); // ❌ Unnecessary copy

// Use views instead
let result = expensive_operation(&matrix.view()); // ✅ No copy
```

### ❌ Pitfall 2: Wrong Algorithm Choice

```rust
// Don't use general algorithms for special matrices
let result = lu(&spd_matrix.view(), None)?; // ❌ Slow for SPD matrices

// Use specialized algorithms
let result = cholesky(&spd_matrix.view(), None)?; // ✅ 2x faster
```

### ❌ Pitfall 3: Serial Processing of Large Data

```rust
// Avoid serial processing for large matrices
for i in 0..large_n {
    result[i] = expensive_computation(&data[i]); // ❌ Serial
}

// Use parallel processing
result.par_iter_mut()
    .zip(data.par_iter())
    .for_each(|(r, d)| *r = expensive_computation(d)); // ✅ Parallel
```

### ❌ Pitfall 4: Ignoring Cache Effects

```rust
// Column-major access on row-major matrix (cache-unfriendly)
for j in 0..n {
    for i in 0..m {
        result += matrix[[i, j]]; // ❌ Poor cache locality
    }
}

// Row-major access (cache-friendly)
for i in 0..m {
    for j in 0..n {
        result += matrix[[i, j]]; // ✅ Good cache locality
    }
}
```

### ❌ Pitfall 5: Not Using Problem Structure

```rust
// Ignoring sparsity
let result = solve(&sparse_matrix.to_dense(), &b.view(), None)?; // ❌ Slow

// Exploit sparsity
let result = sparse_solve(&sparse_matrix, &b.view())?; // ✅ Much faster
```

## Performance Validation

### Verification Checklist

Before deploying performance-critical code:

1. **Profile with realistic data sizes**
2. **Test on target hardware**
3. **Validate numerical accuracy**
4. **Check memory usage patterns**
5. **Benchmark against alternatives**
6. **Monitor for performance regressions**

### Sample Performance Test

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn performance_regression_test(c: &mut Criterion) {
    let sizes = vec![100, 500, 1000, 2000];
    
    for size in sizes {
        let matrix = Array2::<f64>::ones((size, size));
        
        c.bench_function(&format!("qr_decomposition_{}", size), |b| {
            b.iter(|| {
                let (q, r) = qr(&black_box(matrix.view()), None).unwrap();
                (q, r)
            })
        });
    }
}

criterion_group!(performance_tests, performance_regression_test);
criterion_main!(performance_tests);
```

## Conclusion

Optimal performance in scirs2-linalg requires understanding your problem characteristics and choosing appropriate algorithms, data structures, and computational strategies. The key principles are:

1. **Match algorithms to problem structure**
2. **Leverage parallelism for large problems**  
3. **Minimize memory allocations and copies**
4. **Use SIMD acceleration when available**
5. **Profile and benchmark regularly**

For additional performance questions or optimization suggestions, please refer to the [scirs2-linalg documentation](https://docs.rs/scirs2-linalg) or open an issue on [GitHub](https://github.com/cool-japan/scirs).