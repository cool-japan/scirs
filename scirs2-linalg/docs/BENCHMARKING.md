# Benchmarking Guide for SciRS2 Linear Algebra

This guide provides detailed instructions for running benchmarks, interpreting results, and conducting performance analysis with SciRS2.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Available Benchmarks](#available-benchmarks)
3. [Running Benchmarks](#running-benchmarks)
4. [Interpreting Results](#interpreting-results)
5. [Custom Benchmarks](#custom-benchmarks)
6. [Performance Regression Testing](#performance-regression-testing)
7. [Comparative Analysis](#comparative-analysis)

## Quick Start

To run all benchmarks:

```bash
# Install required dependencies
cargo install criterion

# Run all benchmarks with default settings
cargo bench

# Run specific benchmark suite
cargo bench --bench scipy_compat_benchmarks

# Generate detailed reports
cargo bench -- --output-format html
```

## Available Benchmarks

SciRS2 includes several comprehensive benchmark suites:

### 1. SciPy Compatibility Benchmarks (`scipy_compat_benchmarks.rs`)

Tests the performance of SciPy-compatible API functions:

- **Basic Operations**: `det`, `inv`, matrix operations
- **Matrix Norms**: Frobenius, 1-norm, infinity norm, vector norms
- **Decompositions**: LU, QR, SVD, Cholesky, Schur
- **Eigenvalue Problems**: `eigh`, `eigvals`, symmetric solvers
- **Linear Solvers**: General solve, least squares, triangular systems
- **Matrix Functions**: `expm`, `logm`, `sqrtm`, trigonometric functions
- **Advanced Operations**: Condition numbers, matrix rank, pseudoinverse
- **Utility Functions**: Block diagonal construction, memory allocation

### 2. Core Linear Algebra Benchmarks (`linalg_bench.rs`)

Fundamental BLAS operations:

- Dot products for various vector sizes
- Vector norms (L1, L2, infinity)
- Basic BLAS level 1, 2, and 3 operations

### 3. SIMD Acceleration Benchmarks (`simd_bench.rs`)

Compares SIMD-optimized implementations:

- **Matrix-Vector Multiplication**: Regular vs SIMD vs BLAS
- **Matrix Multiplication**: Performance across different implementations
- **Dot Products**: Vector operations with different acceleration methods

### 4. Performance Optimization Benchmarks (`perf_opt_bench.rs`)

Tests advanced optimization techniques:

- Blocked matrix algorithms
- Cache-aware implementations
- Memory-efficient operations
- Parallel processing scenarios

### 5. Quantization Benchmarks (`quantization_bench.rs`)

Reduced precision arithmetic:

- Quantized matrix operations
- Memory efficiency measurements
- Accuracy vs performance trade-offs

### 6. Attention Mechanisms (`attention_bench.rs`)

Neural network-specific operations:

- Multi-head attention
- Flash attention implementations
- Batch processing performance

## Running Benchmarks

### Basic Usage

```bash
# Run all benchmarks
cargo bench

# Run with specific target
cargo bench --target x86_64-unknown-linux-gnu

# Run in release mode (recommended)
cargo bench --release
```

### Benchmark-Specific Options

```bash
# Run only SciPy compatibility benchmarks
cargo bench --bench scipy_compat_benchmarks

# Run specific test group
cargo bench --bench scipy_compat_benchmarks basic_operations

# Run with custom measurement time
cargo bench -- --measurement-time 30

# Generate detailed statistical reports
cargo bench -- --verbose
```

### Matrix Size Customization

Modify benchmark parameters in the source files:

```rust
// In scipy_compat_benchmarks.rs
for &size in &[10, 50, 100, 200] { // Add custom sizes
    let matrix = create_test_matrix(size);
    // ... benchmark code
}
```

### Platform-Specific Benchmarks

```bash
# Enable SIMD features
RUSTFLAGS="-C target-cpu=native" cargo bench

# Enable specific CPU features
RUSTFLAGS="-C target-feature=+avx2,+fma" cargo bench

# Run with different optimization levels
RUSTFLAGS="-C opt-level=3" cargo bench
```

## Interpreting Results

### Standard Criterion Output

Criterion provides detailed statistical analysis:

```
matrix_operations/det_compat/100
                        time:   [15.234 μs 15.456 μs 15.678 μs]
                        change: [-2.3% +0.1% +2.8%] (p = 0.92 > 0.05)
                        No change in performance detected.
Found 12 outliers among 100 measurements (12.00%)
  8 (8.00%) high mild
  4 (4.00%) high severe
```

### Key Metrics Explained

- **Time Range**: [lower_bound mean upper_bound] with 95% confidence interval
- **Change**: Performance change from previous run
- **P-value**: Statistical significance of the change
- **Outliers**: Measurements significantly different from the mean

### Performance Categories

Based on our benchmark results:

| Performance Category | Time Range | Use Case |
|---------------------|------------|----------|
| Excellent | < 1 μs | Real-time applications |
| Good | 1-100 μs | Interactive applications |
| Acceptable | 100 μs - 10 ms | Batch processing |
| Slow | > 10 ms | Research/prototyping |

### Comparing Implementation Variants

Look for patterns in the results:

```
Operation         SciPy API    Direct API   BLAS Accelerated
det_100x100       15.4 μs      14.2 μs      12.8 μs
inv_100x100       87.3 μs      82.1 μs      45.6 μs
qr_100x100        234.7 μs     228.4 μs     156.3 μs
```

## Custom Benchmarks

### Creating Your Own Benchmark

1. Create a new file in the `benches/` directory:

```rust
// benches/my_custom_bench.rs
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use ndarray::Array2;
use scirs2_linalg::*;

fn my_operation_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("my_operation");
    
    for &size in &[50, 100, 200] {
        let matrix = Array2::from_shape_fn((size, size), |(i, j)| {
            (i as f64 + j as f64) / (size as f64)
        });
        
        group.bench_with_input(
            BenchmarkId::new("my_algorithm", size),
            &matrix,
            |b, m| b.iter(|| my_algorithm(&m.view()))
        );
    }
    
    group.finish();
}

fn my_algorithm(matrix: &ndarray::ArrayView2<f64>) -> f64 {
    // Your algorithm implementation
    matrix.sum()
}

criterion_group!(benches, my_operation_benchmark);
criterion_main!(benches);
```

2. Run your custom benchmark:

```bash
cargo bench --bench my_custom_bench
```

### Advanced Benchmark Features

#### Throughput Measurement

```rust
use criterion::Throughput;

group.throughput(Throughput::Elements(size as u64 * size as u64));
group.bench_with_input(
    BenchmarkId::new("matrix_operation", size),
    &matrix,
    |b, m| b.iter(|| expensive_operation(m))
);
```

#### Parameter Sweeps

```rust
fn benchmark_parameter_sweep(c: &mut Criterion) {
    let mut group = c.benchmark_group("parameter_sweep");
    
    for &matrix_size in &[50, 100, 200] {
        for &algorithm_param in &[1, 5, 10] {
            group.bench_with_input(
                BenchmarkId::from_parameter(format!("{}x{}_param{}", 
                                                   matrix_size, matrix_size, algorithm_param)),
                &(matrix_size, algorithm_param),
                |b, &(size, param)| {
                    let matrix = create_test_matrix(size);
                    b.iter(|| algorithm_with_param(&matrix, param))
                }
            );
        }
    }
}
```

#### Memory Usage Profiling

```rust
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

struct TrackingAllocator {
    allocated: AtomicUsize,
}

impl TrackingAllocator {
    fn allocated(&self) -> usize {
        self.allocated.load(Ordering::Relaxed)
    }
}

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let size = layout.size();
        self.allocated.fetch_add(size, Ordering::Relaxed);
        System.alloc(layout)
    }
    
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        let size = layout.size();
        self.allocated.fetch_sub(size, Ordering::Relaxed);
        System.dealloc(ptr, layout)
    }
}

fn benchmark_with_memory_tracking(c: &mut Criterion) {
    let allocator = TrackingAllocator { allocated: AtomicUsize::new(0) };
    
    c.bench_function("memory_tracked_operation", |b| {
        b.iter_custom(|iters| {
            let start_memory = allocator.allocated();
            let start_time = std::time::Instant::now();
            
            for _ in 0..iters {
                expensive_operation();
            }
            
            let duration = start_time.elapsed();
            let peak_memory = allocator.allocated() - start_memory;
            
            println!("Peak memory usage: {} bytes", peak_memory);
            duration
        })
    });
}
```

## Performance Regression Testing

### Setting Up Baseline

1. Create a baseline measurement:

```bash
# Run benchmarks and save results
cargo bench -- --save-baseline main

# Make your changes...

# Compare against baseline
cargo bench -- --baseline main
```

### Automated Regression Detection

Create a script to detect performance regressions:

```bash
#!/bin/bash
# scripts/performance_check.sh

# Run benchmarks and capture output
cargo bench --bench scipy_compat_benchmarks -- --baseline main > bench_results.txt

# Check for significant regressions (>5% slowdown)
if grep -q "change:.*+[5-9]\.[0-9]*%" bench_results.txt; then
    echo "Performance regression detected!"
    exit 1
else
    echo "No significant performance regression"
    exit 0
fi
```

### Continuous Integration Integration

Add to your CI pipeline:

```yaml
# .github/workflows/performance.yml
name: Performance Testing
on: [push, pull_request]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Install Rust
      uses: actions-rs/toolchain@v1
      with:
        toolchain: stable
    - name: Run benchmarks
      run: cargo bench --bench scipy_compat_benchmarks
    - name: Check for regressions
      run: ./scripts/performance_check.sh
```

## Comparative Analysis

### Comparing with Reference Implementations

Create benchmarks that compare SciRS2 with other libraries:

```rust
// benches/comparative_bench.rs
fn compare_with_nalgebra(c: &mut Criterion) {
    let mut group = c.benchmark_group("nalgebra_comparison");
    
    for &size in &[50, 100, 200] {
        // SciRS2 implementation
        let scirs_matrix = scirs2_linalg::Array2::zeros((size, size));
        group.bench_with_input(
            BenchmarkId::new("scirs2", size),
            &scirs_matrix,
            |b, m| b.iter(|| scirs2_linalg::det(&m.view()))
        );
        
        // nalgebra implementation
        let nalgebra_matrix = nalgebra::DMatrix::<f64>::zeros(size, size);
        group.bench_with_input(
            BenchmarkId::new("nalgebra", size),
            &nalgebra_matrix,
            |b, m| b.iter(|| m.determinant())
        );
    }
}
```

### Cross-Platform Performance

Test performance across different platforms:

```bash
# Linux
cargo bench --target x86_64-unknown-linux-gnu

# macOS
cargo bench --target x86_64-apple-darwin

# Windows
cargo bench --target x86_64-pc-windows-msvc

# ARM (if available)
cargo bench --target aarch64-unknown-linux-gnu
```

### Generating Performance Reports

Create comprehensive performance reports:

```bash
# Generate HTML reports
cargo bench -- --output-format html

# Generate JSON data for custom analysis
cargo bench -- --output-format json > benchmark_results.json

# Create flame graphs (requires perf and flamegraph)
cargo bench --bench scipy_compat_benchmarks -- --profile-time=5
```

### Performance Visualization

Use the generated data to create performance charts:

```python
# scripts/plot_performance.py
import json
import matplotlib.pyplot as plt

with open('benchmark_results.json') as f:
    data = json.load(f)

# Extract and plot performance data
sizes = []
times = []

for benchmark in data['benchmarks']:
    if 'matrix_size' in benchmark['id']:
        size = int(benchmark['id'].split('_')[-1])
        time = benchmark['mean']['estimate']
        sizes.append(size)
        times.append(time)

plt.loglog(sizes, times, 'o-')
plt.xlabel('Matrix Size')
plt.ylabel('Time (ns)')
plt.title('Performance Scaling')
plt.grid(True)
plt.savefig('performance_scaling.png')
```

## Troubleshooting Benchmark Issues

### Common Problems

1. **Inconsistent Results**
   ```bash
   # Increase measurement time
   cargo bench -- --measurement-time 30
   
   # Reduce system load
   sudo systemctl stop unnecessary-services
   ```

2. **Outliers**
   ```bash
   # Run with more samples
   cargo bench -- --sample-size 1000
   
   # Check for thermal throttling
   watch sensors  # Monitor CPU temperature
   ```

3. **Memory Issues**
   ```bash
   # Monitor memory usage during benchmarks
   cargo bench &
   watch -n 1 free -h
   ```

### Best Practices for Reliable Benchmarks

1. **System Configuration**
   - Disable CPU frequency scaling
   - Close unnecessary applications
   - Use dedicated benchmark machine

2. **Benchmark Design**
   - Use appropriate matrix sizes
   - Include warm-up iterations
   - Test multiple algorithms

3. **Statistical Validity**
   - Run multiple times
   - Check for statistical significance
   - Account for measurement noise

This comprehensive benchmarking infrastructure allows you to accurately measure and optimize the performance of your linear algebra operations in SciRS2.