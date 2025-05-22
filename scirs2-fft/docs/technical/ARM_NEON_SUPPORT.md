# ARM NEON Support for FFT Operations

This document describes the SIMD acceleration for Fast Fourier Transform (FFT) operations on ARM platforms using the NEON SIMD instruction set.

## Overview

The `scirs2-fft` crate provides optimized FFT implementations for ARM processors with NEON SIMD support, offering significant performance improvements for various FFT operations, including:

- 1D FFT and inverse FFT
- 2D FFT and inverse FFT
- N-dimensional FFT and inverse FFT
- Memory-efficient FFT operations
- Adaptive planning strategies optimized for ARM architecture

## Implementation Details

### Automatic Detection and Dispatch

On ARM platforms (aarch64), the library automatically uses NEON-accelerated implementations:

- NEON is a standard feature in all ARMv8-A processors (aarch64 architecture)
- No runtime feature detection is needed, unlike x86 platforms where AVX2 is optional
- The adaptive dispatchers automatically select the NEON implementation
- Planning strategies are optimized for ARM's cache hierarchy and pipeline

### Optimized Operations

The following operations are optimized:

1. **SIMD-accelerated complex number normalization**
   - Uses NEON vector instructions to perform normalization in parallel
   - Significant speedup for post-FFT scaling operations
   - Integrated with memory-efficient operations for improved cache utilization

2. **Optimized f64 to complex conversion**
   - Fast conversion of real inputs to complex format
   - Uses vector loads/stores for improved memory bandwidth
   - Aligns memory for optimal NEON performance

3. **Memory-efficient FFT operations**
   - In-place FFT computations to minimize memory allocations
   - SIMD-accelerated normalization integrated with memory-efficient operations
   - Adaptive memory usage based on available system resources

4. **Real-valued FFT optimizations (RFFT)**
   - Specialized SIMD-accelerated implementation for real-valued input signals
   - More efficient than general complex FFT for real inputs
   - Automatically uses NEON instructions on ARM platforms
   - Provides both forward (RFFT) and inverse (IRFFT) operations

5. **Advanced planning strategies**
   - Optimized plan caching for ARM architecture
   - Multi-threading support through parallel planning
   - Adaptive algorithm selection based on input size and system capabilities

## Performance Considerations

For best performance on ARM platforms:

1. **Use adaptive dispatchers**
   - `fft::adaptive`, `ifft::adaptive`, `fft2::adaptive`, `fftn::adaptive`, etc.
   - These will automatically select the best implementation for your hardware

2. **Optimized for larger FFTs**
   - Performance improvements become more significant with larger datasets
   - For small FFTs (e.g., size < 32), overhead may outweigh benefits
   - Parallel planning benefits increase with input size

3. **Memory layout matters**
   - Contiguous memory with proper alignment gives the best performance
   - Row-major layout is assumed for multi-dimensional operations
   - Using memory-efficient operations further improves performance

4. **Use planning optimization**
   - Enable plan caching for repeated transformations of same size
   - For best results with large datasets, use parallel planning strategies

## Benchmarking

The crate includes ARM-specific benchmarks in the `benches/arm_fft_bench.rs` file:

- Comparison of standard vs. NEON implementations
- Benchmarks for 1D, 2D, and 3D FFT operations
- Tests with different input sizes and planning strategies

To run the benchmarks on an ARM platform:

```bash
cargo bench --bench arm_fft_bench
```

For planning-specific benchmarks:

```bash
cargo bench --bench planning_benchmarks
```

## Testing

ARM-specific tests are included in `src/arm_fft_test.rs`. These tests verify:

1. Correctness of 1D FFT and inverse FFT
2. Correctness of 2D FFT operations
3. Correctness of N-dimensional FFT
4. Proper frequency peak detection for known input signals
5. Verification of planning optimization strategies

To run just the ARM-specific tests:

```bash
cargo test arm_fft_test
```

## Current Implementations

The crate currently provides the following optimized operations for ARM:

1. Basic 1D, 2D, and ND FFT operations with NEON acceleration
2. Memory-efficient variants with optimized cache usage
3. Real-to-complex and complex-to-real transformations (RFFT/IRFFT)
4. Adaptive planning strategies with parallel execution
5. SIMD-optimized normalization for post-processing

## Future Improvements

Planned enhancements for ARM NEON support:

1. Further optimization of specialized operations (e.g., RFFT, HFFT)
2. Better cache utilization for multi-dimensional transforms
3. Support for half-precision (f16) operations on ARM processors that support it
4. Enhanced thread pool integration for multi-core ARM processors
5. SVE (Scalable Vector Extension) support for newer ARM processors