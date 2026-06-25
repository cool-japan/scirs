# SciRS2 v0.3.0 Benchmark Suite

## Quick Start

```bash
cd <path-to-scirs-repo>

# Run all v0.3.0 benchmarks
./benches/v030_run_all_benchmarks.sh

# Aggregate and analyze results
python3 benches/aggregate_v030_results.py
```

## Benchmark Files

### Core Benchmarks

1. **v030_comprehensive_suite.rs** (613 lines)
   - Linear algebra: GEMM, SVD, Eigenvalues, LU/QR/Cholesky
   - FFT operations (64-8192 points)
   - Statistical operations (1k-1M elements)
   - Optimization algorithms
   - Special functions (Bessel, Gamma, Error)
   - Clustering (K-means)
   - Signal processing (convolution, correlation)

2. **v030_autograd_benchmarks.rs** (528 lines)
   - Forward pass: simple ops, matrix ops, complex graphs
   - Backward pass: gradients and backpropagation
   - Gradient accumulation
   - SIMD vs non-SIMD comparison
   - Memory overhead profiling

3. **v030_neural_benchmarks.rs** (514 lines)
   - MNIST training (MLP and CNN)
   - CIFAR-10 training (CNN)
   - Inference latency (batch sizes: 1, 8, 32, 128)
   - Layer-wise performance (Dense, Conv2D)
   - Optimizer performance (SGD, Adam)
   - Memory usage profiling

4. **v030_series_benchmarks.rs** (628 lines)
   - ARIMA fitting and forecasting
   - SARIMA fitting and forecasting
   - STL and classical decomposition
   - ACF/PACF computation
   - Differencing operations
   - Rolling window operations
   - Forecasting accuracy metrics

## Running Specific Benchmarks

### Run Individual Suites

```bash
# Comprehensive suite only
cargo bench --package scirs2-benchmarks --bench v030_comprehensive_suite

# Autograd only
cargo bench --package scirs2-benchmarks --bench v030_autograd_benchmarks

# Neural networks only
cargo bench --package scirs2-benchmarks --bench v030_neural_benchmarks

# Time series only
cargo bench --package scirs2-benchmarks --bench v030_series_benchmarks
```

### Run Specific Categories

```bash
# Linear algebra benchmarks only
cargo bench --package scirs2-benchmarks --bench v030_comprehensive_suite -- linalg

# Forward pass benchmarks only
cargo bench --package scirs2-benchmarks --bench v030_autograd_benchmarks -- forward

# MNIST benchmarks only
cargo bench --package scirs2-benchmarks --bench v030_neural_benchmarks -- mnist

# ARIMA benchmarks only
cargo bench --package scirs2-benchmarks --bench v030_series_benchmarks -- arima
```

## Performance Optimization

### Enable Maximum Performance

```bash
# Set CPU-specific optimizations
export RUSTFLAGS="-C target-cpu=native -C opt-level=3"

# Disable CPU frequency scaling (Linux)
sudo cpupower frequency-set --governor performance

# Disable turbo boost for consistency
echo "1" | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo
```

### SIMD Optimizations

```bash
# Enable SIMD features
cargo bench --package scirs2-benchmarks --features simd

# Check SIMD usage
RUSTFLAGS="-C target-cpu=native -C opt-level=3 -C llvm-args=-print-after-all" \
  cargo bench --package scirs2-benchmarks 2>&1 | grep -i simd
```

## Benchmark Results

### Output Locations

JSON results are saved to `/tmp/`:
- `/tmp/scirs2_v030_comprehensive_results.json`
- `/tmp/scirs2_v030_autograd_results.json`
- `/tmp/scirs2_v030_neural_results.json`
- `/tmp/scirs2_v030_series_results.json`

HTML reports are in:
- `<repo-root>/target/criterion/`
- Open `target/criterion/report/index.html` in a browser

Aggregated reports in `/tmp/scirs2_v030_reports/`:
- `v030_benchmark_report.md` - Human-readable markdown
- `v030_aggregate.json` - Machine-readable JSON

### Result Format

JSON results contain:
```json
{
  "category": "linalg_gemm",
  "operation": "gemm",
  "size": 512,
  "mean_time_ns": 1234567.89,
  "throughput_ops_per_sec": 810000.0,
  "std_dev_ns": 12345.67
}
```

## Performance Targets (v0.3.0)

### Comprehensive Suite
- SIMD operations: 5-20x speedup vs naive
- Matrix operations: competitive with BLAS
- FFT: competitive with rustfft/OxiFFT
- Parallel: near-linear scaling up to 16 cores
- Memory: <10% overhead vs optimal

### Autograd
- Forward pass overhead: <10% vs manual
- Backward pass: <2x forward pass time
- SIMD optimization: 3-8x speedup
- Memory overhead: <30% for tape

### Neural Networks
- MNIST training: >1000 images/sec
- CIFAR-10 training: >500 images/sec
- Single image inference: <10ms
- Batch inference (32): <50ms
- Memory: <500MB for typical models

### Time Series
- ARIMA fitting: <1s for 1000 points
- SARIMA fitting: <5s for 1000 points
- STL decomposition: <500ms for 1000 points
- Forecasting: <100ms for 100 steps
- Autocorrelation: <10ms for 1000 lags

## Comparison with v0.2.0

### Run Comparison

```bash
# Baseline (v0.2.0)
git checkout 0.2.0
cargo bench --package scirs2-benchmarks -- --save-baseline v020

# Current (v0.3.0)
git checkout 0.3.0
cargo bench --package scirs2-benchmarks -- --baseline v020
```

### Expected Improvements

- **Autograd**: New in v0.3.0 (no baseline)
- **Neural Networks**: 2-3x faster than manual backprop
- **Time Series**: 20-30% faster with SIMD
- **GEMM**: 10-15% faster with optimized layouts
- **FFT**: 5-10% faster with OxiFFT improvements

## Troubleshooting

### Compilation Errors

```bash
# Check dependencies
cargo tree --package scirs2-benchmarks

# Clean rebuild
cargo clean
cargo build --package scirs2-benchmarks --benches --release
```

### Performance Variations

High standard deviation usually indicates:
- CPU frequency scaling enabled
- Background processes consuming CPU
- Thermal throttling
- Memory pressure

Solutions:
1. Close background applications
2. Disable CPU frequency scaling
3. Ensure adequate cooling
4. Increase measurement time in benchmark code

### Memory Issues

If benchmarks crash with OOM:
1. Reduce batch sizes in neural benchmarks
2. Reduce matrix sizes in linalg benchmarks
3. Close other applications
4. Monitor with `top` or `htop`

## Advanced Usage

### Custom Measurement Time

Edit benchmark files and modify:
```rust
group.measurement_time(Duration::from_secs(10)); // Increase for stability
```

### Warm-up Iterations

Add warm-up before benchmarking:
```rust
group.warm_up_time(Duration::from_secs(3));
```

### Sample Size

Adjust sample size for faster/slower runs:
```rust
group.sample_size(100); // Default is 100
```

### Profiling with perf (Linux)

```bash
# Record performance data
cargo bench --package scirs2-benchmarks --bench v030_comprehensive_suite \
  --profile-time 30 -- --profile-time 30

# Analyze with perf
perf record -g cargo bench --bench v030_comprehensive_suite
perf report
```

### Flame Graphs

```bash
# Install cargo-flamegraph
cargo install flamegraph

# Generate flame graph
cargo flamegraph --bench v030_comprehensive_suite
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: v0.3.0 Benchmarks
on: [push, pull_request]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
          override: true

      - name: Run benchmarks
        run: ./benches/v030_run_all_benchmarks.sh

      - name: Aggregate results
        run: python3 benches/aggregate_v030_results.py

      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: benchmark-results
          path: |
            /tmp/scirs2_v030_*.json
            /tmp/scirs2_v030_reports/
```

## Documentation

Full documentation: `/tmp/V030_BENCHMARKS.md`

## Support

For issues or questions:
- GitHub Issues: https://github.com/cool-japan/scirs
- Documentation: See `/tmp/V030_BENCHMARKS.md`

## License

Apache-2.0 - COOLJAPAN OU (Team Kitasan)
