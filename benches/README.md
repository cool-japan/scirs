# SciRS2 Performance Benchmarking Suite

This directory contains a comprehensive benchmarking suite for SciRS2 that measures performance against SciPy, analyzes numerical stability, evaluates memory efficiency, and provides detailed performance characterization.

## Overview

The benchmarking suite consists of four main components:

1. **Linear Algebra Benchmarks** (`linalg_benchmarks.rs`) - Core mathematical operation performance
2. **SciPy Comparison** (`scipy_comparison.rs` + `scipy_benchmark.py`) - Direct performance comparison
3. **Memory Efficiency** (`memory_efficiency.rs`) - Memory usage and optimization analysis
4. **Numerical Stability** (`numerical_stability.rs`) - Accuracy and robustness testing

## Quick Start

### Run All Benchmarks

```bash
# Run the complete benchmark suite
./run_benchmarks.sh
```

### Run Individual Benchmarks

```bash
# Run specific benchmark categories
cargo bench --bench linalg_benchmarks
cargo bench --bench memory_efficiency  
cargo bench --bench numerical_stability
cargo bench --bench scipy_comparison

# Run Python SciPy comparison
python3 scipy_benchmark.py
```

### Skip Python Comparison

```bash
# If SciPy is not available or you want to skip comparison
./run_benchmarks.sh --skip-python
```

## Benchmark Categories

### 1. Linear Algebra Performance (`linalg_benchmarks.rs`)

Tests the performance of core linear algebra operations across different matrix sizes:

**Basic Operations:**
- Matrix determinant computation
- Matrix inverse (for smaller matrices)
- Matrix norms (Frobenius, spectral, etc.)

**Matrix Decompositions:**
- LU decomposition with partial pivoting
- QR decomposition (Householder reflections)
- SVD decomposition (for smaller matrices)
- Cholesky decomposition (for positive definite matrices)

**Linear System Solvers:**
- General linear system solving
- Least squares problems
- Triangular system solving

**Eigenvalue Problems:**
- Symmetric eigenvalue computation
- Eigenvector computation

**Performance Analysis:**
- Algorithmic complexity verification (O(n²), O(n³))
- Memory usage patterns
- Numerical stability with ill-conditioned matrices

### 2. SciPy Comparison (`scipy_comparison.rs` + `scipy_benchmark.py`)

Direct performance comparison with SciPy equivalents:

**Comparison Metrics:**
- Execution time (nanosecond precision)
- Memory usage during computation
- Numerical accuracy verification
- Cross-platform performance analysis

**Operations Compared:**
- All basic linear algebra operations
- Matrix decompositions
- Linear system solvers
- Performance across different data types (f32 vs f64)

**Output:**
- Speedup ratios (Rust/Python)
- Performance regression analysis
- Detailed comparison reports in JSON format

### 3. Memory Efficiency (`memory_efficiency.rs`)

Analyzes memory usage patterns and optimization effectiveness:

**Buffer Management:**
- Buffer pool allocation efficiency
- Memory reuse patterns
- Fragmentation resistance

**Chunked Operations:**
- Large matrix processing with memory constraints
- Memory-efficient algorithms
- Zero-copy operation performance

**Memory Usage Patterns:**
- In-place vs. copying operations
- Peak memory usage tracking
- Memory allocation strategies

**Large Matrix Handling:**
- Memory-constrained computations
- Out-of-core algorithm preparation
- Memory usage scaling analysis

### 4. Numerical Stability (`numerical_stability.rs`)

Tests numerical accuracy and robustness under challenging conditions:

**Condition Number Analysis:**
- Well-conditioned matrices (condition numbers 10² to 10¹²)
- Performance degradation vs. numerical accuracy trade-offs

**Pathological Matrices:**
- Hilbert matrices (notoriously ill-conditioned)
- Vandermonde matrices
- Near-singular matrices
- Rank-deficient matrices

**Decomposition Accuracy:**
- Reconstruction error analysis for LU, QR, SVD
- Numerical stability of different algorithms
- Error propagation in iterative methods

**Edge Cases:**
- Very small numerical values (near machine epsilon)
- Very large numerical values
- Mixed precision scenarios

## Understanding the Results

### Criterion HTML Reports

After running benchmarks, detailed HTML reports are generated in `target/benchmark-results/*_criterion_reports/`. These provide:

- Interactive performance graphs
- Statistical analysis of timing data
- Regression analysis
- Comparison between different implementations

### JSON Data Files

Several JSON files contain raw benchmark data for programmatic analysis:

- `rust_benchmark_results.json` - Raw Rust timing data
- `python_benchmark_results.json` - Raw SciPy timing data  
- `benchmark_comparison.json` - Comprehensive comparison analysis
- `stability_test_results.json` - Numerical stability test results
- `complexity_analysis.json` - Algorithmic complexity measurements

### Performance Visualization

If Python with matplotlib is available, the benchmark suite generates visualization plots:

- `rust_vs_scipy_comparison.png` - Performance comparison bar chart
- `numerical_stability.png` - Stability vs. condition number analysis
- `complexity_analysis.png` - Algorithmic complexity verification

## Interpreting Performance Metrics

### Throughput vs. Latency

- **Throughput**: Operations per second (higher is better)
- **Latency**: Time per operation (lower is better)
- Both metrics are provided for different use cases

### Memory Metrics

- **Peak Memory Usage**: Maximum memory allocated during operation
- **Memory Efficiency**: Operations per MB of memory used
- **Allocation Rate**: Memory allocations per second

### Numerical Accuracy

- **Relative Error**: `||computed - exact|| / ||exact||`
- **Success Rate**: Percentage of tests passing accuracy threshold
- **Condition Number Threshold**: Maximum condition number for stable computation

### Speedup Analysis

- **Speedup > 1.0**: Rust implementation is faster than SciPy
- **Speedup < 1.0**: SciPy implementation is faster
- **Speedup ≈ 1.0**: Comparable performance

## Configuration and Customization

### Matrix Sizes

Default test sizes are defined in each benchmark file:

```rust
const MATRIX_SIZES: &[usize] = &[10, 50, 100, 200, 500, 1000];
const COMPARISON_SIZES: &[usize] = &[50, 100, 200, 500];
```

Modify these arrays to test different size ranges.

### Condition Numbers

For numerical stability tests:

```rust
let condition_numbers = vec![1e3, 1e6, 1e12];
```

### Random Seed

All benchmarks use a fixed seed for reproducibility:

```rust
const SEED: u64 = 42;
```

### Performance Thresholds

Adjust accuracy thresholds in `numerical_stability.rs`:

```rust
let success = relative_error < 1e-10; // Tolerance for success
```

## Dependencies

### Rust Dependencies

All required dependencies are specified in `Cargo.toml`:

- `criterion` - Benchmarking framework
- `ndarray` - N-dimensional arrays
- `scirs2-*` - SciRS2 modules being benchmarked

### Python Dependencies

For SciPy comparison (optional):

```bash
pip install numpy scipy matplotlib psutil
```

## Troubleshooting

### Common Issues

**"Failed to compile benchmarks"**
- Check that all SciRS2 modules compile successfully
- Ensure all features are properly enabled in dependencies

**"SciPy benchmarks failed"**
- Verify Python dependencies: `python3 -c "import numpy, scipy, matplotlib"`
- Check that Python script has execution permissions

**"Memory usage too high"**
- Reduce `MATRIX_SIZES` for memory-constrained environments
- Monitor system memory during benchmark execution

**"Numerical stability tests failing"**
- This may indicate implementation issues with ill-conditioned matrices
- Review the condition number thresholds and test matrix generation

### Performance Considerations

**Benchmark Execution Time:**
- Complete benchmark suite: 10-30 minutes depending on hardware
- Individual benchmarks: 2-5 minutes each
- Large matrix tests may require significant time and memory

**System Resources:**
- Ensure sufficient RAM (recommend 8GB+ for large matrix tests)
- Close other applications during benchmarking for accurate results
- Use consistent system state between benchmark runs

## Contributing

### Adding New Benchmarks

1. Create new benchmark file in `benches/`
2. Add benchmark entry to `Cargo.toml`
3. Update `run_benchmarks.sh` to include new benchmark
4. Add documentation to this README

### Benchmark Design Guidelines

- Use consistent random seeds for reproducibility
- Include both performance and accuracy verification
- Provide meaningful progress output
- Save results in JSON format for analysis
- Include error handling for edge cases

## Continuous Integration

The benchmark suite is designed for integration with CI/CD systems:

```yaml
# Example GitHub Actions configuration
- name: Run Performance Benchmarks
  run: |
    cd benches
    ./run_benchmarks.sh --skip-python
    # Upload results as artifacts
```

## License

This benchmarking suite is part of SciRS2 and is licensed under the same terms (MIT OR Apache-2.0).