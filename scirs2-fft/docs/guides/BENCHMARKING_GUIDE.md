# FFT Benchmarking Guide

This guide explains how to run and interpret the comprehensive benchmark suite for scirs2-fft.

## Overview

The benchmark suite includes:

1. **Performance Benchmarks**: Measure execution time for various FFT operations
2. **Memory Profiling**: Track memory usage and allocation patterns
3. **Accuracy Testing**: Compare numerical accuracy with analytical results
4. **SciPy Comparison**: Direct comparison with SciPy's FFT implementation

## Quick Start

Run all benchmarks with a single command:

```bash
./run_all_benchmarks.sh
```

This will:
- Run Rust performance benchmarks
- Profile memory usage
- Test numerical accuracy
- Generate comparison reports

## Individual Benchmarks

### Performance Benchmarks

```bash
cargo bench --bench fft_benchmarks
```

Measures execution time for:
- 1D FFT/IFFT operations
- Real FFT (RFFT/IRFFT)
- 2D and N-dimensional FFT
- Fractional FFT
- Memory-efficient variants

### Memory Profiling

```bash
cargo run --release --bin memory_profiling
```

Tracks:
- Peak memory usage
- Allocation patterns
- Memory efficiency of different algorithms

### Accuracy Testing

```bash
cargo run --release --bin accuracy_comparison
```

Tests:
- Analytical accuracy for known signals
- Energy conservation (Parseval's theorem)
- Inverse transform accuracy
- Transform additivity properties

### SciPy Comparison

```bash
cargo bench --bench scipy_comparison
python benches/run_scipy_benchmarks.py
```

Compares:
- Performance with SciPy FFT
- Memory usage differences
- Accuracy comparison

## Visualization and Analysis

Generate visual reports:

```bash
cargo run --example benchmark_analysis
```

This creates:
- Performance comparison plots
- Memory usage graphs
- Accuracy comparison charts
- HTML comparison tables

## Understanding Results

### Performance Metrics

- **Throughput**: Operations per second
- **Latency**: Time per operation
- **Scaling**: Performance vs problem size
- **Parallelization**: Effect of worker threads

### Memory Metrics

- **Peak Usage**: Maximum memory allocated
- **Allocation Count**: Number of memory allocations
- **Efficiency**: Memory per data point

### Accuracy Metrics

- **Max Error**: Worst-case error
- **Mean Error**: Average error
- **RMS Error**: Root mean square error
- **Relative Error**: Error relative to signal magnitude

## Optimization Tips

Based on benchmark results:

1. **Use RFFT for real data**: ~2x performance improvement
2. **Enable parallelization**: For arrays > 4096 elements
3. **Choose appropriate precision**: f32 vs f64 trade-offs
4. **Use in-place operations**: When memory is constrained

## Benchmark Configuration

Modify benchmark parameters in:
- `benches/fft_benchmarks.rs`
- `benches/scipy_comparison.rs`

Key parameters:
- Input sizes
- Number of iterations
- Worker thread counts
- Transform types

## Troubleshooting

Common issues:

1. **Out of memory**: Reduce array sizes or use memory-efficient variants
2. **Inconsistent results**: Increase iteration count or disable CPU frequency scaling
3. **Missing Python dependencies**: Install scipy and numpy

## Contributing

To add new benchmarks:

1. Create a new benchmark file in `benches/`
2. Add to `Cargo.toml` under `[[bench]]`
3. Update `run_all_benchmarks.sh`
4. Document in this guide

## Performance Targets

Current performance goals:
- 1D FFT: < 1ms for 4096 points
- 2D FFT: < 10ms for 256x256
- Memory overhead: < 2x input size
- Accuracy: < 1e-10 relative error