# SciRS2 FFT

[![crates.io](https://img.shields.io/crates/v/scirs2-fft.svg)](https://crates.io/crates/scirs2-fft)
[![License](https://img.shields.io/crates/l/scirs2-fft.svg)](../LICENSE)
[![Documentation](https://img.shields.io/docsrs/scirs2-fft)](https://docs.rs/scirs2-fft)

Fast Fourier Transform implementation and related functionality for the SciRS2 scientific computing library. This module provides efficient FFT implementations, DCT/DST transforms, and window functions.

## Features

- **FFT Implementation**: Efficient implementations of Fast Fourier Transform 
- **Real FFT**: Specialized implementation for real input
- **DCT/DST**: Discrete Cosine Transform and Discrete Sine Transform
- **Window Functions**: Variety of window functions (Hann, Hamming, Blackman, etc.)
- **Helper Functions**: Utilities for working with frequency domain data
- **Parallel Processing**: Optimized parallel implementations for large arrays
- **Memory-Efficient Operations**: Specialized functions for processing large arrays with minimal memory usage
- **Signal Analysis**: Hilbert transform for analytical signal computation
- **Non-Uniform FFT**: Support for data sampled at non-uniform intervals
- **Fractional Fourier Transform**: Generalization of the FFT for arbitrary angles in the time-frequency plane
- **Time-Frequency Analysis**: STFT, spectrogram, and waterfall plots for visualization
- **Visualization Tools**: Colormaps and 3D data formatting for signal visualization
- **Spectral Analysis**: Comprehensive tools for frequency domain analysis

## Installation

Add the following to your `Cargo.toml`:

```toml
[dependencies]
scirs2-fft = "0.1.0-alpha.1"
# Optional: Enable parallel processing
scirs2-fft = { version = "0.1.0-alpha.1", features = ["parallel"] }
```

Basic usage examples:

```rust
use scirs2_fft::{fft, rfft, window, hilbert, nufft, frft, frft_complex, 
                stft, spectrogram, spectrogram_normalized,
                waterfall_3d, waterfall_mesh, waterfall_lines, apply_colormap,
                memory_efficient::{fft_inplace, fft2_efficient, fft_streaming, process_in_chunks, FftMode}};
use ndarray::{Array1, array};
use num_complex::Complex64;

// Compute FFT
let data = array![1.0, 2.0, 3.0, 4.0];
let result = fft::fft(&data).unwrap();
println!("FFT result: {:?}", result);

// Compute real FFT (more efficient for real input)
let real_data = array![1.0, 2.0, 3.0, 4.0];
let real_result = rfft::rfft(&real_data).unwrap();
println!("Real FFT result: {:?}", real_result);

// Use a window function
let window_func = window::hann(64);
println!("Hann window: {:?}", window_func);

// Compute DCT (Discrete Cosine Transform)
let dct_data = array![1.0, 2.0, 3.0, 4.0];
let dct_result = dct::dct(&dct_data, Some(DCTType::Type2), None).unwrap();
println!("DCT result: {:?}", dct_result);

// Use parallel FFT for large arrays (with "parallel" feature enabled)
use ndarray::Array2;
let large_data = Array2::<f64>::zeros((256, 256));
let parallel_result = fft2_parallel(&large_data.view(), None).unwrap();
println!("Parallel 2D FFT completed");

// Compute Hilbert transform (analytic signal)
let time_signal = vec![1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0];
let analytic_signal = hilbert(&time_signal).unwrap();
println!("Analytic signal magnitude: {}", 
         (analytic_signal[0].re.powi(2) + analytic_signal[0].im.powi(2)).sqrt());

// Non-uniform FFT (Type 1: non-uniform samples to uniform frequencies)
use std::f64::consts::PI;
use scirs2_fft::nufft::InterpolationType;

// Create non-uniform sample points
let n = 50;
let sample_points: Vec<f64> = (0..n).map(|i| -PI + 1.8*PI*i as f64/(n as f64)).collect();
let sample_values: Vec<Complex64> = sample_points.iter()
    .map(|&x| Complex64::new(x.cos(), 0.0))
    .collect();

// Compute NUFFT (Type 1)
let m = 64; // Output grid size
let nufft_result = nufft::nufft_type1(
    &sample_points, &sample_values, m, 
    InterpolationType::Gaussian, 1e-6
).unwrap();

// Fractional Fourier Transform
// For real input (alpha=0.5 is halfway between time and frequency domain)
let signal: Vec<f64> = (0..128).map(|i| (2.0 * PI * 10.0 * i as f64 / 128.0).sin()).collect();
let frft_result = frft(&signal, 0.5, None).unwrap();

// For complex input, use frft_complex directly
let complex_signal: Vec<Complex64> = (0..64).map(|i| {
    let t = i as f64 / 64.0;
    Complex64::new((2.0 * PI * 5.0 * t).cos(), 0.0)
}).collect();
let frft_complex_result = frft_complex(&complex_signal, 0.5, None).unwrap();

// Time-Frequency Analysis with STFT and Spectrogram
let fs = 1000.0; // 1 kHz sampling rate
let t = (0..1000).map(|i| i as f64 / fs).collect::<Vec<_>>();
let chirp = t.iter().map(|&ti| (2.0 * PI * (10.0 + 50.0 * ti) * ti).sin()).collect::<Vec<_>>();

// Compute Short-Time Fourier Transform
let (frequencies, times, stft_result) = stft(
    &chirp,
    Window::Hann,
    256,        // Segment length
    Some(128),  // Overlap
    None,       // Default FFT length
    Some(fs),   // Sampling rate
    None,       // Default detrending
    None,       // Default boundary handling
).unwrap();

// Generate a spectrogram (power spectral density)
let (_, _, psd) = spectrogram(
    &chirp,
    Some(fs),
    Some(Window::Hann),
    Some(256),
    Some(128),
    None,
    None,
    Some("density"),
    Some("psd"),
).unwrap();

// Generate a normalized spectrogram suitable for visualization
let (_, _, normalized) = spectrogram_normalized(
    &chirp,
    Some(fs),
    Some(256),
    Some(128),
    Some(80.0),  // 80 dB dynamic range
).unwrap();

// Waterfall plots (3D visualization of spectrograms)
// Generate 3D coordinates (t, f, amplitude) suitable for 3D plotting
let (t, f, coords) = waterfall_3d(
    &chirp,
    Some(fs),    // Sampling rate
    Some(256),   // Segment length
    Some(128),   // Overlap
    Some(true),  // Use log scale
    Some(80.0),  // 80 dB dynamic range
).unwrap();

// Generate mesh format data for surface plotting
let (time_mesh, freq_mesh, amplitude_mesh) = waterfall_mesh(
    &chirp,
    Some(fs),
    Some(256),
    Some(128),
    Some(true),
    Some(80.0),
).unwrap();

// Generate stacked lines format (traditional waterfall plot view)
let (times, freqs, line_data) = waterfall_lines(
    &chirp,
    Some(fs),
    Some(256),    // Segment length
    Some(128),    // Overlap
    Some(20),     // Number of lines to include
    Some(0.1),    // Vertical offset between lines
    Some(true),   // Use log scale
    Some(80.0),   // Dynamic range in dB
).unwrap();

// Apply a colormap to amplitude values
let amplitudes = Array1::from_vec(vec![0.0, 0.25, 0.5, 0.75, 1.0]);
let colors = apply_colormap(&amplitudes, "jet").unwrap();  // Options: jet, viridis, plasma, grayscale, hot
```

## Components

### FFT Implementation

Core FFT functionality:

```rust
use scirs2_fft::fft::{
    fft,                // Forward FFT
    ifft,               // Inverse FFT
    fft2,               // 2D FFT
    ifft2,              // 2D inverse FFT
    fft2_parallel,      // Parallel implementation of 2D FFT (with "parallel" feature)
    fftn,               // n-dimensional FFT
    ifftn,              // n-dimensional inverse FFT
    fftfreq,            // Return the Discrete Fourier Transform sample frequencies
    fftshift,           // Shift the zero-frequency component to the center
    ifftshift,          // Inverse of fftshift
};

// Memory-efficient operations for large arrays
use scirs2_fft::memory_efficient::{
    fft_inplace,         // In-place FFT that minimizes allocations
    fft2_efficient,      // Memory-efficient 2D FFT
    fft_streaming,       // Process large arrays in streaming fashion
    process_in_chunks,   // Apply custom operation to chunks of large array
    FftMode,             // Forward or Inverse FFT mode enum
};
```

### Real FFT

Specialized functions for real input:

```rust
use scirs2_fft::rfft::{
    rfft,               // Real input FFT (more efficient)
    irfft,              // Inverse of rfft
    rfft2,              // 2D real FFT
    irfft2,             // 2D inverse real FFT
    rfftn,              // n-dimensional real FFT
    irfftn,             // n-dimensional inverse real FFT
};
```

### DCT/DST

Discrete Cosine Transform and Discrete Sine Transform:

```rust
use scirs2_fft::dct::{
    dct,                // Discrete Cosine Transform
    idct,               // Inverse Discrete Cosine Transform
    Type,               // Enum for DCT types (DCT1, DCT2, DCT3, DCT4)
};

use scirs2_fft::dst::{
    dst,                // Discrete Sine Transform
    idst,               // Inverse Discrete Sine Transform
    Type,               // Enum for DST types (DST1, DST2, DST3, DST4)
};
```

### Window Functions

Various window functions for signal processing:

```rust
use scirs2_fft::window::{
    hann,               // Hann window
    hamming,            // Hamming window
    blackman,           // Blackman window
    bartlett,           // Bartlett window
    flattop,            // Flat top window
    kaiser,             // Kaiser window
    gaussian,           // Gaussian window
    general_cosine,     // General cosine window
    general_hamming,    // General Hamming window
    nuttall,            // Nuttall window
    blackman_harris,    // Blackman-Harris window
};
```

### Helper Functions

Utilities for working with frequency domain data:

```rust
use scirs2_fft::helper::{
    next_fast_len,      // Find the next fast size for FFT
    fftfreq,            // Get FFT sample frequencies
    rfftfreq,           // Get real FFT sample frequencies
    fftshift,           // Shift zero frequency to center
    ifftshift,          // Inverse of fftshift
};
```

## Performance

The FFT implementation in this module is optimized for performance:

- Uses the `rustfft` crate for the core FFT algorithm
- Provides SIMD-accelerated implementations when available
- Includes specialized implementations for common cases 
- Parallel implementations for large arrays using Rayon
- Offers automatic selection of the most efficient algorithm
- Smart thresholds to choose between sequential and parallel implementations

## Contributing

See the [CONTRIBUTING.md](../CONTRIBUTING.md) file for contribution guidelines.

## License

This project is licensed under the Apache License, Version 2.0 - see the [LICENSE](../LICENSE) file for details.