# Performance Optimization Guidelines for SciRS2 Linear Algebra

This document provides practical, actionable guidelines for optimizing performance when using the SciRS2 linear algebra library. Use this as a quick reference guide to make performance-critical decisions in your code.

## Table of Contents

1. [Quick Performance Checklist](#quick-performance-checklist)
2. [Algorithm Selection Guidelines](#algorithm-selection-guidelines)
3. [Memory Optimization Strategies](#memory-optimization-strategies)
4. [Data Type and Precision Decisions](#data-type-and-precision-decisions)
5. [Parallelization Guidelines](#parallelization-guidelines)
6. [Matrix Structure Exploitation](#matrix-structure-exploitation)
7. [Common Performance Anti-patterns](#common-performance-anti-patterns)
8. [Performance Debugging Tools](#performance-debugging-tools)
9. [Environment-Specific Optimizations](#environment-specific-optimizations)

## Quick Performance Checklist

Before optimizing, ask these questions:

### 📊 **Matrix Size Analysis**
- [ ] **Small matrices (< 100×100)**: Avoid overhead from parallelization and BLAS
- [ ] **Medium matrices (100×100 - 1000×1000)**: Use BLAS acceleration and moderate parallelization  
- [ ] **Large matrices (> 1000×1000)**: Prioritize memory efficiency and chunked processing

### 🔍 **Matrix Properties**
- [ ] **Symmetric?** Use `_sym` variants and Cholesky for positive definite cases
- [ ] **Sparse?** Switch to sparse algorithms when >70% zeros
- [ ] **Structured?** Exploit Toeplitz, circulant, banded patterns
- [ ] **Well-conditioned?** Use direct methods; otherwise use iterative + preconditioning

### 💾 **Memory Constraints**
- [ ] **Limited memory?** Use iterative solvers and out-of-core algorithms
- [ ] **Repeated operations?** Pre-factorize matrices and reuse decompositions
- [ ] **Batch processing?** Group operations to amortize setup costs

### ⚡ **Precision Requirements**
- [ ] **High accuracy needed?** Use `f64` and stable algorithms
- [ ] **Memory/speed critical?** Consider `f32` or quantization
- [ ] **Intermediate results?** Use mixed precision approaches

## Algorithm Selection Guidelines

### Linear System Solving

```rust
use scirs2_linalg::*;

fn choose_solver(
    matrix: &ArrayView2<f64>, 
    rhs: &ArrayView1<f64>
) -> Result<Array1<f64>, LinalgError> {
    
    let n = matrix.nrows();
    let properties = analyze_matrix(matrix);
    
    match (n, properties) {
        // Small systems: direct methods
        (n, _) if n < 50 => {
            solve(matrix, rhs) // Uses LU with partial pivoting
        },
        
        // Symmetric positive definite: use Cholesky (2x faster than LU)
        (_, props) if props.is_symmetric && props.is_positive_definite == Some(true) => {
            cholesky_solve(matrix, rhs)
        },
        
        // Diagonally dominant: iterative methods work well
        (_, props) if props.is_diagonally_dominant == Some(true) => {
            conjugate_gradient(matrix, rhs, 1000, 1e-10)
        },
        
        // Large sparse systems
        (n, props) if n > 1000 && props.sparsity_ratio > 0.7 => {
            sparse_conjugate_gradient(matrix, rhs, 10000, 1e-10)
        },
        
        // Large dense systems: use blocked LU
        (n, _) if n > 1000 => {
            blocked_lu_solve(matrix, rhs)
        },
        
        // Default: general LU solver
        _ => solve(matrix, rhs)
    }
}
```

### Matrix Decomposition Selection

```rust
fn choose_decomposition(matrix: &ArrayView2<f64>) -> DecompositionStrategy {
    let n = matrix.nrows();
    let cond_estimate = estimate_condition_number(matrix);
    
    match (n, cond_estimate) {
        // Small well-conditioned: LU is fastest
        (n, cond) if n < 200 && cond < 1e6 => {
            DecompositionStrategy::LU
        },
        
        // Numerically challenging: QR is more stable
        (_, cond) if cond > 1e12 => {
            DecompositionStrategy::QR  // Better numerical stability
        },
        
        // Large matrices: consider memory
        (n, _) if n > 1000 => {
            DecompositionStrategy::IterativeRefinement  // Memory efficient
        },
        
        // Least squares problems: always QR or SVD
        (_, _) if matrix.nrows() != matrix.ncols() => {
            if matrix.nrows() > matrix.ncols() * 2 {
                DecompositionStrategy::SVD  // More robust for rank-deficient
            } else {
                DecompositionStrategy::QR   // Faster for well-conditioned
            }
        },
        
        _ => DecompositionStrategy::LU
    }
}
```

### Eigenvalue Solver Selection

```rust
fn choose_eigen_solver(
    matrix: &ArrayView2<f64>, 
    num_eigenvalues: Option<usize>
) -> EigenStrategy {
    
    let n = matrix.nrows();
    let num_requested = num_eigenvalues.unwrap_or(n);
    
    match (n, num_requested) {
        // Small matrices: compute all eigenvalues
        (n, _) if n < 100 => EigenStrategy::Dense,
        
        // Large matrices, few eigenvalues: use sparse methods
        (n, k) if n > 500 && k < n / 10 => EigenStrategy::Sparse,
        
        // Large matrices, many eigenvalues: blocked dense methods
        (n, k) if n > 500 && k > n / 2 => EigenStrategy::BlockedDense,
        
        // Power iteration for dominant eigenvalue
        (_, 1) => EigenStrategy::PowerIteration,
        
        _ => EigenStrategy::Dense
    }
}
```

## Memory Optimization Strategies

### 1. Pre-allocate Output Buffers

```rust
// ❌ Avoid: Creates temporary allocations
fn inefficient_batch_processing(matrices: &[Array2<f64>]) -> Vec<Array2<f64>> {
    matrices.iter()
        .map(|m| m.dot(m))  // Each dot() allocates result
        .collect()
}

// ✅ Better: Reuse output buffer
fn efficient_batch_processing(matrices: &[Array2<f64>]) -> Vec<Array2<f64>> {
    let mut result = Array2::zeros((matrices[0].nrows(), matrices[0].ncols()));
    let mut results = Vec::with_capacity(matrices.len());
    
    for matrix in matrices {
        // Reuse the same output buffer
        matmul_into(matrix, matrix, &mut result);
        results.push(result.clone());
    }
    results
}
```

### 2. Use Views Instead of Copies

```rust
// ❌ Avoid: Unnecessary copying
fn process_submatrices_bad(matrix: &Array2<f64>) -> f64 {
    let mut sum = 0.0;
    for i in 0..matrix.nrows()-10 {
        let submatrix = matrix.slice(s![i..i+10, i..i+10]).to_owned(); // Copies data
        sum += determinant(&submatrix.view());
    }
    sum
}

// ✅ Better: Use views
fn process_submatrices_good(matrix: &Array2<f64>) -> f64 {
    let mut sum = 0.0;
    for i in 0..matrix.nrows()-10 {
        let submatrix = matrix.slice(s![i..i+10, i..i+10]); // Zero-copy view
        sum += determinant(&submatrix);
    }
    sum
}
```

### 3. Memory-Aware Algorithm Selection

```rust
fn memory_aware_solve(
    matrix: &ArrayView2<f64>, 
    rhs: &ArrayView1<f64>
) -> Result<Array1<f64>, LinalgError> {
    
    let matrix_memory = matrix.len() * std::mem::size_of::<f64>();
    let available_memory = get_available_memory(); // Custom function
    
    // Rule of thumb: direct methods need ~3x matrix memory for workspace
    if matrix_memory * 3 > available_memory {
        // Use iterative methods (minimal extra memory)
        println!("Using memory-efficient iterative solver");
        conjugate_gradient(matrix, rhs, 10000, 1e-12)
    } else if matrix.nrows() > 2000 {
        // Large but fits in memory: use blocked algorithms
        println!("Using blocked direct solver");
        blocked_lu_solve(matrix, rhs)
    } else {
        // Normal case: standard direct solver
        solve(matrix, rhs)
    }
}
```

## Data Type and Precision Decisions

### Precision Selection Framework

```rust
#[derive(Debug, Clone)]
pub struct PrecisionRequirements {
    pub accuracy_tolerance: f64,
    pub memory_constraints: usize,
    pub performance_priority: bool,
    pub numerical_stability_critical: bool,
}

fn select_precision(requirements: &PrecisionRequirements) -> PrecisionStrategy {
    match requirements {
        // High accuracy, stability critical: always f64
        req if req.numerical_stability_critical && req.accuracy_tolerance < 1e-12 => {
            PrecisionStrategy::F64
        },
        
        // Memory constrained, moderate accuracy: f32
        req if req.memory_constraints > 0 && req.accuracy_tolerance > 1e-6 => {
            PrecisionStrategy::F32
        },
        
        // Performance critical with acceptable accuracy: mixed precision
        req if req.performance_priority && req.accuracy_tolerance > 1e-10 => {
            PrecisionStrategy::Mixed { compute: F32, accumulate: F64 }
        },
        
        // Extreme memory constraints: quantization
        req if req.memory_constraints > 0 && req.accuracy_tolerance > 1e-3 => {
            PrecisionStrategy::Quantized { bits: 8 }
        },
        
        // Default: f64 for safety
        _ => PrecisionStrategy::F64
    }
}

// Example usage for different scenarios
fn ml_inference_precision() -> PrecisionStrategy {
    select_precision(&PrecisionRequirements {
        accuracy_tolerance: 1e-4,      // ML typically tolerates some error
        memory_constraints: 1_000_000, // Mobile/edge constraints
        performance_priority: true,     // Inference speed critical
        numerical_stability_critical: false,
    })
}

fn scientific_simulation_precision() -> PrecisionStrategy {
    select_precision(&PrecisionRequirements {
        accuracy_tolerance: 1e-15,     // High accuracy needed
        memory_constraints: 0,         // Assume sufficient memory
        performance_priority: false,   // Accuracy over speed
        numerical_stability_critical: true,
    })
}
```

### Mixed Precision Patterns

```rust
// Pattern 1: F32 computation with F64 accumulation
fn mixed_precision_matmul(
    a: &Array2<f32>, 
    b: &Array2<f32>
) -> Array2<f64> {
    let mut result = Array2::<f64>::zeros((a.nrows(), b.ncols()));
    
    for i in 0..a.nrows() {
        for j in 0..b.ncols() {
            let mut sum = 0.0_f64;  // Accumulate in f64
            for k in 0..a.ncols() {
                sum += (a[[i, k]] as f64) * (b[[k, j]] as f64);
            }
            result[[i, j]] = sum;
        }
    }
    result
}

// Pattern 2: Adaptive precision based on conditioning
fn adaptive_precision_solve(
    matrix: &Array2<f64>, 
    rhs: &Array1<f64>
) -> Result<Array1<f64>, LinalgError> {
    
    let cond_estimate = estimate_condition_number(matrix);
    
    if cond_estimate < 1e7 && matrix.len() < 1_000_000 {
        // Well-conditioned and not too large: try f32 first
        let matrix_f32 = matrix.mapv(|x| x as f32);
        let rhs_f32 = rhs.mapv(|x| x as f32);
        
        match solve(&matrix_f32.view(), &rhs_f32.view()) {
            Ok(solution_f32) => {
                // Verify accuracy and promote to f64 if needed
                let solution_f64 = solution_f32.mapv(|x| x as f64);
                let residual = matrix.dot(&solution_f64) - rhs;
                let relative_error = vector_norm(&residual.view(), 2)?;
                
                if relative_error < 1e-10 {
                    Ok(solution_f64)
                } else {
                    // F32 wasn't accurate enough, use f64
                    solve(&matrix.view(), &rhs.view())
                }
            },
            Err(_) => {
                // F32 failed, fall back to f64
                solve(&matrix.view(), &rhs.view())
            }
        }
    } else {
        // Use f64 for challenging problems
        solve(&matrix.view(), &rhs.view())
    }
}
```

## Parallelization Guidelines

### When to Use Parallelization

```rust
fn should_parallelize(operation: &str, data_size: usize, num_cores: usize) -> bool {
    match operation {
        "matmul" => {
            // Matrix multiplication: parallelize for medium to large matrices
            let elements = data_size;
            let parallel_overhead = 1000; // Approximate overhead in elements
            elements > parallel_overhead * num_cores
        },
        
        "elementwise" => {
            // Element-wise operations: need large arrays to benefit
            data_size > 100_000
        },
        
        "solve" => {
            // Linear solve: only for large systems
            let n = (data_size as f64).sqrt() as usize;
            n > 500 && num_cores > 2
        },
        
        "decomposition" => {
            // Decompositions: parallelization often built into BLAS
            let n = (data_size as f64).sqrt() as usize;
            n > 1000  // Let BLAS handle smaller matrices
        },
        
        _ => false
    }
}

// Example: Batch processing with optimal parallelization
fn process_batch_optimal<T, F, R>(
    items: &[T], 
    operation: F,
    num_cores: usize
) -> Vec<R> 
where 
    T: Sync,
    F: Fn(&T) -> R + Sync,
    R: Send
{
    use rayon::prelude::*;
    
    let parallel_threshold = num_cores * 4; // Minimum items per core
    
    if items.len() < parallel_threshold {
        // Sequential processing for small batches
        items.iter().map(|item| operation(item)).collect()
    } else {
        // Parallel processing for large batches
        items.par_iter()
             .with_max_len(items.len() / num_cores) // Chunk size
             .map(|item| operation(item))
             .collect()
    }
}
```

## Matrix Structure Exploitation

### Automatic Structure Detection

```rust
use scirs2_linalg::analysis::*;

fn exploit_structure_automatically(matrix: &Array2<f64>) -> StructureStrategy {
    let analysis = analyze_matrix_structure(matrix);
    
    match analysis {
        // Diagonal matrices: O(1) operations
        s if s.is_diagonal => StructureStrategy::Diagonal,
        
        // Symmetric positive definite: use Cholesky
        s if s.is_symmetric && s.is_positive_definite == Some(true) => {
            StructureStrategy::CholeskyDecomposition
        },
        
        // Banded matrices: use banded solvers
        s if s.bandwidth < matrix.nrows() / 4 => {
            StructureStrategy::BandedSolver { bandwidth: s.bandwidth }
        },
        
        // Toeplitz matrices: use FFT-based algorithms
        s if s.is_toeplitz => StructureStrategy::ToeplitzSolver,
        
        // Circulant matrices: even faster FFT algorithms
        s if s.is_circulant => StructureStrategy::CirculantSolver,
        
        // Sparse matrices: switch to sparse algorithms
        s if s.sparsity_ratio > 0.7 => {
            StructureStrategy::SparseDirect { 
                fill_reducing_ordering: true 
            }
        },
        
        // Low-rank matrices: use rank-revealing decompositions
        s if s.effective_rank < matrix.nrows() / 2 => {
            StructureStrategy::LowRank { rank: s.effective_rank }
        },
        
        // General dense matrices
        _ => StructureStrategy::DenseGeneral
    }
}

// Example: Automatic solver selection based on structure
fn solve_with_structure_detection(
    matrix: &Array2<f64>, 
    rhs: &Array1<f64>
) -> Result<Array1<f64>, LinalgError> {
    
    match exploit_structure_automatically(matrix) {
        StructureStrategy::Diagonal => {
            // O(n) operation for diagonal matrices
            diagonal_solve(matrix, rhs)
        },
        
        StructureStrategy::CholeskyDecomposition => {
            // ~2x faster than LU for SPD matrices
            cholesky_solve(matrix, rhs)
        },
        
        StructureStrategy::BandedSolver { bandwidth } => {
            // O(n*b^2) instead of O(n^3) for bandwidth b
            banded_solve(matrix, rhs, bandwidth)
        },
        
        StructureStrategy::ToeplitzSolver => {
            // O(n log n) using FFT
            toeplitz_solve(matrix, rhs)
        },
        
        StructureStrategy::CirculantSolver => {
            // O(n log n) with better constants than Toeplitz
            circulant_solve(matrix, rhs)
        },
        
        StructureStrategy::SparseDirect { fill_reducing_ordering } => {
            sparse_direct_solve(matrix, rhs, fill_reducing_ordering)
        },
        
        StructureStrategy::LowRank { rank } => {
            low_rank_solve(matrix, rhs, rank)
        },
        
        StructureStrategy::DenseGeneral => {
            // Standard dense solver
            solve(matrix, rhs)
        }
    }
}
```

## Common Performance Anti-patterns

### ❌ Anti-pattern 1: Unnecessary Matrix Copies

```rust
// BAD: Creates multiple unnecessary copies
fn bad_matrix_chain_multiply(matrices: &[Array2<f64>]) -> Array2<f64> {
    let mut result = matrices[0].clone(); // Copy 1
    for i in 1..matrices.len() {
        let temp = result.dot(&matrices[i]); // Copy 2 (result allocation)
        result = temp; // Copy 3 (assignment)
    }
    result
}

// GOOD: Minimize copies with careful memory management
fn good_matrix_chain_multiply(matrices: &[Array2<f64>]) -> Array2<f64> {
    if matrices.is_empty() {
        return Array2::zeros((0, 0));
    }
    
    let mut result = matrices[0].clone(); // Only necessary copy
    let mut temp = Array2::zeros((result.nrows(), matrices[1].ncols()));
    
    for i in 1..matrices.len() {
        matmul_into(&result.view(), &matrices[i].view(), &mut temp);
        std::mem::swap(&mut result, &mut temp); // Zero-copy swap
    }
    result
}
```

### ❌ Anti-pattern 2: Wrong Loop Order for Memory Access

```rust
// BAD: Column-wise access (poor cache locality)
fn bad_matrix_access(matrix: &mut Array2<f64>) {
    for j in 0..matrix.ncols() {
        for i in 0..matrix.nrows() {
            matrix[[i, j]] *= 2.0; // Non-contiguous memory access
        }
    }
}

// GOOD: Row-wise access (good cache locality)
fn good_matrix_access(matrix: &mut Array2<f64>) {
    for i in 0..matrix.nrows() {
        for j in 0..matrix.ncols() {
            matrix[[i, j]] *= 2.0; // Contiguous memory access
        }
    }
}

// EVEN BETTER: Use ndarray's optimized methods
fn best_matrix_access(matrix: &mut Array2<f64>) {
    matrix.mapv_inplace(|x| x * 2.0); // Optimized by ndarray
}
```

### ❌ Anti-pattern 3: Inefficient Repeated Decompositions

```rust
// BAD: Recomputes decomposition for each solve
fn bad_multiple_solves(matrix: &Array2<f64>, rhs_vectors: &[Array1<f64>]) -> Vec<Array1<f64>> {
    rhs_vectors.iter()
        .map(|rhs| solve(&matrix.view(), &rhs.view()).unwrap())
        .collect()
}

// GOOD: Factorize once, solve multiple times
fn good_multiple_solves(matrix: &Array2<f64>, rhs_vectors: &[Array1<f64>]) -> Result<Vec<Array1<f64>>, LinalgError> {
    // Factorize the matrix once
    let (p, l, u) = lu(&matrix.view())?;
    
    // Solve for each RHS using the factorization
    let mut solutions = Vec::with_capacity(rhs_vectors.len());
    for rhs in rhs_vectors {
        let solution = lu_solve_factored(&p, &l, &u, &rhs.view())?;
        solutions.push(solution);
    }
    
    Ok(solutions)
}
```

### ❌ Anti-pattern 4: Ignoring Numerical Stability

```rust
// BAD: Always uses fastest algorithm without considering conditioning
fn bad_solver_choice(matrix: &Array2<f64>, rhs: &Array1<f64>) -> Array1<f64> {
    // LU is fastest but may be unstable for ill-conditioned matrices
    solve(&matrix.view(), &rhs.view()).unwrap()
}

// GOOD: Considers matrix conditioning for algorithm choice
fn good_solver_choice(matrix: &Array2<f64>, rhs: &Array1<f64>) -> Result<Array1<f64>, LinalgError> {
    let cond_estimate = estimate_condition_number(matrix);
    
    if cond_estimate > 1e12 {
        // Use SVD for severely ill-conditioned matrices
        println!("Warning: Matrix is ill-conditioned (cond ≈ {:.2e}), using SVD", cond_estimate);
        svd_solve(matrix, rhs)
    } else if cond_estimate > 1e8 {
        // Use QR for moderately ill-conditioned matrices
        qr_solve(matrix, rhs)
    } else {
        // Use LU for well-conditioned matrices
        solve(&matrix.view(), &rhs.view())
    }
}
```

## Performance Debugging Tools

### Built-in Performance Profiling

```rust
use scirs2_linalg::diagnostics::*;

fn debug_performance_issue(matrix: &Array2<f64>) {
    // Enable detailed performance tracking
    let profiler = PerformanceProfiler::new();
    
    // Analyze matrix properties that affect performance
    let analysis = profiler.analyze_matrix_performance(matrix);
    
    println!("Matrix Performance Analysis:");
    println!("  Size: {}x{}", matrix.nrows(), matrix.ncols());
    println!("  Condition number estimate: {:.2e}", analysis.condition_estimate);
    println!("  Memory footprint: {:.2} MB", analysis.memory_mb);
    println!("  Cache efficiency score: {:.1}/10", analysis.cache_efficiency);
    println!("  Sparsity ratio: {:.2}%", analysis.sparsity_ratio * 100.0);
    
    // Performance recommendations
    for recommendation in analysis.recommendations {
        println!("💡 {}", recommendation);
    }
    
    // Benchmark different algorithms
    let solvers = ["lu", "qr", "svd", "iterative"];
    for solver in &solvers {
        let timing = profiler.benchmark_solver(matrix, solver);
        println!("  {}: {:.2}ms (±{:.2}ms)", solver, timing.mean_ms, timing.std_ms);
    }
}

// Memory usage tracking
fn track_memory_usage<F, R>(operation_name: &str, operation: F) -> R 
where F: FnOnce() -> R 
{
    let start_memory = get_memory_usage();
    let start_time = std::time::Instant::now();
    
    let result = operation();
    
    let end_time = start_time.elapsed();
    let peak_memory = get_peak_memory_usage();
    let end_memory = get_memory_usage();
    
    println!("Operation: {}", operation_name);
    println!("  Duration: {:.2}ms", end_time.as_millis());
    println!("  Memory allocated: {:.2} MB", (peak_memory - start_memory) as f64 / 1_048_576.0);
    println!("  Memory retained: {:.2} MB", (end_memory - start_memory) as f64 / 1_048_576.0);
    
    result
}
```

### Custom Benchmarking Framework

```rust
use criterion::{Criterion, BenchmarkId};

fn create_comprehensive_benchmark() {
    let mut criterion = Criterion::default()
        .measurement_time(std::time::Duration::from_secs(30))
        .sample_size(100);
    
    // Matrix sizes to test
    let sizes = [50, 100, 200, 500, 1000];
    
    // Matrix types to test
    let matrix_types = [
        ("well_conditioned", create_well_conditioned_matrix),
        ("ill_conditioned", create_ill_conditioned_matrix),
        ("sparse", create_sparse_matrix),
        ("structured", create_structured_matrix),
    ];
    
    // Algorithms to benchmark
    let algorithms = [
        ("lu_solve", |m: &Array2<f64>, r: &Array1<f64>| solve(&m.view(), &r.view())),
        ("qr_solve", |m: &Array2<f64>, r: &Array1<f64>| qr_solve(m, r)),
        ("iterative", |m: &Array2<f64>, r: &Array1<f64>| conjugate_gradient(&m.view(), &r.view(), 1000, 1e-10)),
    ];
    
    for size in &sizes {
        for (matrix_type, matrix_fn) in &matrix_types {
            let matrix = matrix_fn(*size);
            let rhs = Array1::ones(*size);
            
            for (alg_name, alg_fn) in &algorithms {
                criterion.bench_with_input(
                    BenchmarkId::new(
                        format!("{}_{}", alg_name, matrix_type), 
                        size
                    ),
                    &(&matrix, &rhs),
                    |b, (m, r)| {
                        b.iter(|| alg_fn(m, r))
                    },
                );
            }
        }
    }
}
```

## Environment-Specific Optimizations

### CPU Architecture Optimization

```rust
fn optimize_for_cpu_architecture() -> OptimizationSettings {
    let cpu_info = detect_cpu_features();
    
    OptimizationSettings {
        use_avx2: cpu_info.has_avx2,
        use_fma: cpu_info.has_fma,
        cache_line_size: cpu_info.cache_line_size,
        l1_cache_size: cpu_info.l1_cache_size,
        l2_cache_size: cpu_info.l2_cache_size,
        l3_cache_size: cpu_info.l3_cache_size,
        preferred_block_size: calculate_optimal_block_size(&cpu_info),
        numa_aware: cpu_info.numa_nodes > 1,
    }
}

fn calculate_optimal_block_size(cpu_info: &CpuInfo) -> usize {
    // Rule of thumb: block should fit in L2 cache
    // For matrix multiplication: 3 blocks of size^2 * sizeof(f64)
    let available_l2 = cpu_info.l2_cache_size * 8 / 10; // Use 80% of L2
    let block_memory = available_l2 / (3 * std::mem::size_of::<f64>());
    (block_memory as f64).sqrt() as usize
}
```

### Memory Hierarchy Optimization

```rust
fn cache_efficient_matrix_multiply(
    a: &Array2<f64>, 
    b: &Array2<f64>,
    block_size: Option<usize>
) -> Array2<f64> {
    
    let n = a.nrows();
    let block_size = block_size.unwrap_or_else(|| {
        // Dynamic block size based on available cache
        let l2_cache = get_l2_cache_size();
        ((l2_cache / (3 * std::mem::size_of::<f64>())) as f64).sqrt() as usize
    });
    
    let mut result = Array2::zeros((n, n));
    
    // Blocked matrix multiplication for cache efficiency
    for i in (0..n).step_by(block_size) {
        for j in (0..n).step_by(block_size) {
            for k in (0..n).step_by(block_size) {
                let i_end = (i + block_size).min(n);
                let j_end = (j + block_size).min(n);
                let k_end = (k + block_size).min(n);
                
                // Multiply blocks
                for ii in i..i_end {
                    for jj in j..j_end {
                        let mut sum = result[[ii, jj]];
                        for kk in k..k_end {
                            sum += a[[ii, kk]] * b[[kk, jj]];
                        }
                        result[[ii, jj]] = sum;
                    }
                }
            }
        }
    }
    
    result
}
```

### Platform-Specific Code Paths

```rust
#[cfg(target_arch = "x86_64")]
fn platform_optimized_operation(data: &[f64]) -> f64 {
    if std::arch::is_x86_feature_detected!("avx2") {
        unsafe { avx2_optimized_operation(data) }
    } else if std::arch::is_x86_feature_detected!("sse4.1") {
        unsafe { sse_optimized_operation(data) }
    } else {
        fallback_operation(data)
    }
}

#[cfg(target_arch = "aarch64")]
fn platform_optimized_operation(data: &[f64]) -> f64 {
    if std::arch::is_aarch64_feature_detected!("neon") {
        unsafe { neon_optimized_operation(data) }
    } else {
        fallback_operation(data)
    }
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn platform_optimized_operation(data: &[f64]) -> f64 {
    fallback_operation(data)
}
```

## Quick Reference Summary

### Algorithm Selection Cheat Sheet

| Matrix Size | Condition | Best Algorithm | Complexity | Memory |
|------------|-----------|----------------|------------|---------|
| < 100×100 | Any | Direct LU | O(n³) | Low |
| < 100×100 | SPD | Cholesky | O(n³/3) | Low |
| 100-1000×100 | Well-cond | BLAS LU | O(n³) | Medium |
| 100-1000×100 | Ill-cond | QR | O(n³) | Medium |
| > 1000×1000 | Any | Iterative | O(kn²) | Low |
| > 1000×1000 | Sparse | Sparse CG | O(knnz) | Low |

### Memory Optimization Priorities

1. **🔄 Reuse allocations** - Pre-allocate buffers, use `_into` functions
2. **👁️ Use views** - Avoid `.to_owned()` when possible
3. **🧮 Choose precision** - `f32` vs `f64` based on requirements
4. **📦 Exploit structure** - Sparse, banded, symmetric matrices
5. **⚡ Batch operations** - Amortize setup costs

### Performance Red Flags

- ⚠️ **Matrix copying in loops** - Use views or pre-allocated buffers
- ⚠️ **Wrong loop order** - Row-major access for better cache locality  
- ⚠️ **Repeated decompositions** - Factorize once, solve multiple times
- ⚠️ **Ignoring structure** - Check for symmetry, sparsity, special patterns
- ⚠️ **Wrong algorithm choice** - Consider condition number and matrix size

This optimization guide provides practical, actionable advice for achieving optimal performance with the SciRS2 linear algebra library. Always profile your specific use case to validate optimization decisions.