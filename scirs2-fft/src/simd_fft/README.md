# SIMD-Optimized FFT Module

This module provides SIMD-accelerated implementations of Fast Fourier Transform (FFT) operations. These implementations can significantly improve performance on modern CPUs that support SIMD instructions.

## Overview

The module includes SIMD-optimized versions of:

- 1D FFT and IFFT (`fft_simd`, `ifft_simd`)
- 2D FFT and IFFT (`fft2_simd`, `ifft2_simd`)
- N-dimensional FFT and IFFT (`fftn_simd`, `ifftn_simd`)

It also provides adaptive dispatchers (`fft_adaptive`, `ifft_adaptive`, etc.) that automatically select between SIMD-optimized and standard implementations based on hardware support.

## Usage

### Basic Usage

```rust
use scirs2_fft::{fft_adaptive, ifft_adaptive, simd_support_available};

// Check if SIMD acceleration is available
let simd_available = simd_support_available();
println!("SIMD support: {}", simd_available);

// Create a signal
let signal: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];

// Compute FFT using adaptive implementation
// This will automatically use SIMD if available
let spectrum = fft_adaptive(&signal, None, None).unwrap();

// Compute IFFT to recover the signal
let recovered_signal = ifft_adaptive(&spectrum, None, None).unwrap();

// Extract real part of the recovered signal
let real_part: Vec<f64> = recovered_signal.iter().map(|c| c.re).collect();
```

### Direct Access to SIMD Functions

If you want to explicitly use the SIMD implementations:

```rust
use scirs2_fft::{fft_simd, ifft_simd};

// Only run this if SIMD is available
if scirs2_fft::simd_support_available() {
    let signal = vec![1.0, 2.0, 3.0, 4.0];
    
    // SIMD version of FFT
    let spectrum = fft_simd(&signal, None, None).unwrap();
    
    // SIMD version of IFFT
    let recovered = ifft_simd(&spectrum, None, None).unwrap();
}
```

### 2D FFT Example

```rust
use scirs2_fft::{fft2_adaptive, ifft2_adaptive};

// Create a 2D signal
let width = 8;
let height = 8;
let signal = vec![1.0; width * height];

// Compute 2D FFT
let spectrum = fft2_adaptive(&signal, [height, width], None, None).unwrap();

// Compute inverse 2D FFT
let recovered = ifft2_adaptive(&spectrum, [height, width], None, None).unwrap();
```

### N-dimensional FFT Example

```rust
use scirs2_fft::{fftn_adaptive, ifftn_adaptive};

// Create a 3D signal
let shape = [4, 4, 4];
let signal = vec![1.0; shape.iter().product()];

// Compute ND FFT
let spectrum = fftn_adaptive(&signal, &shape, None, None).unwrap();

// Compute inverse ND FFT
let recovered = ifftn_adaptive(&spectrum, &shape, None, None).unwrap();
```

## Performance Considerations

1. **Data Size**: SIMD acceleration is most beneficial for larger data sizes (typically 1024 elements or more).

2. **Memory Alignment**: For best performance, ensure your data is properly aligned in memory. The implementation handles unaligned data, but performance will be better with aligned data.

3. **Complex Input**: If your input is already complex, SIMD acceleration still provides benefits but may not be as dramatic as for real-to-complex transforms.

4. **Repeated Transforms**: If you need to perform many transforms of the same size, the SIMD implementations can provide significant speedups.

## Supported CPU Features

The implementation detects and uses the following SIMD instruction sets:

### x86_64
- AVX2 (preferred)
- SSE4.1 (fallback)

### aarch64
- NEON (when available)

## Implementation Details

The SIMD optimizations include:
- Vectorized data conversion and normalization
- SIMD-accelerated complex arithmetic operations
- Optimized memory access patterns for better cache utilization
- Efficient striding for multi-dimensional transforms

## Examples

Check the examples directory for comprehensive usage examples:
- `benchmark_simd_vs_standard.rs`: Performance comparison between SIMD and standard implementations
- `simd_fft_example.rs`: Basic usage examples
- `simd_fft2_image_processing.rs`: Image processing with 2D FFT
- `simd_fftn_volumetric_data.rs`: Processing volumetric data with N-dimensional FFT
- `spectral_analysis_simd.rs`: Scientific application for spectral analysis