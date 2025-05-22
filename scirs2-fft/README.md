# SciRS2 FFT

[![crates.io](https://img.shields.io/crates/v/scirs2-fft.svg)](https://crates.io/crates/scirs2-fft)
[[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)]](../LICENSE)
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
- **Sparse FFT**: Algorithms for efficiently computing FFT of sparse signals
  - Sublinear-time sparse FFT
  - Compressed sensing-based approach
  - Iterative and deterministic variants
  - Frequency pruning and spectral flatness methods
  - Advanced batch processing for multiple signals
    - Parallel CPU implementation for high throughput
    - Memory-efficient processing for large batches
    - Optimized GPU batch processing with CUDA
- **GPU Acceleration**: CUDA-accelerated implementations for high-performance computing
  - GPU-optimized batch processing with stream management
  - Multiple algorithm variants and specialized kernels
  - Memory management optimizations and efficient buffer allocation
  - Spectral flatness analysis with GPU acceleration
  - Memory pooling for iterative operations

## Installation

Add the following to your `Cargo.toml`:

```toml
[dependencies]
scirs2-fft = "0.1.0-alpha.4"
# Optional: Enable parallel processing
scirs2-fft = { version = "0.1.0-alpha.4", features = ["parallel"] }
# Optional: Enable CUDA GPU acceleration
scirs2-fft = { version = "0.1.0-alpha.4", features = ["cuda"] }
# Optional: Enable both parallel processing and CUDA
scirs2-fft = { version = "0.1.0-alpha.4", features = ["parallel", "cuda"] }
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

// Advanced parallel planning and execution
use scirs2_fft::{
    ParallelPlanner,       // Create FFT plans in parallel
    ParallelExecutor,      // Execute FFT plans in parallel
    ParallelPlanningConfig // Configure parallel planning behavior
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

### Sparse FFT

Efficient algorithms for signals with few significant frequency components:

```rust
use scirs2_fft::sparse_fft::{
    sparse_fft,                   // Compute sparse FFT
    sparse_fft2,                  // 2D sparse FFT
    sparse_fftn,                  // N-dimensional sparse FFT
    adaptive_sparse_fft,          // Adaptively adjust sparsity parameter
    frequency_pruning_sparse_fft, // Using frequency pruning algorithm
    spectral_flatness_sparse_fft, // Using spectral flatness algorithm
    reconstruct_spectrum,         // Reconstruct full spectrum from sparse result
    reconstruct_time_domain,      // Reconstruct time domain signal
    reconstruct_high_resolution,  // High-resolution reconstruction
    SparseFFTAlgorithm,           // Algorithm variants
    WindowFunction,               // Window functions for sparse FFT
};
```

### GPU Acceleration

CUDA-accelerated implementations for high-performance computing:

```rust
use scirs2_fft::{
    // GPU-accelerated sparse FFT
    cuda_sparse_fft,
    cuda_batch_sparse_fft,
    is_cuda_available,
    get_cuda_devices,
    
    // GPU memory management
    init_global_memory_manager,
    get_global_memory_manager,
    BufferLocation,
    AllocationStrategy,
    
    // GPU backend management
    GPUBackend,
    
    // CUDA kernel management
    execute_cuda_sublinear_sparse_fft,
    execute_cuda_compressed_sensing_sparse_fft,
    execute_cuda_iterative_sparse_fft,
    KernelStats,
    KernelConfig,
};

// Check if CUDA is available
if is_cuda_available() {
    // Get available CUDA devices
    let devices = get_cuda_devices().unwrap();
    println!("Found {} CUDA device(s)", devices.len());
    
    // Initialize memory manager
    init_global_memory_manager(
        GPUBackend::CUDA,
        0,  // Use first device
        AllocationStrategy::CacheBySize,
        1024 * 1024 * 1024  // 1 GB limit
    ).unwrap();
    
    // Create a signal
    let signal = vec![1.0, 2.0, 3.0, 4.0];
    
    // Compute sparse FFT on GPU with different algorithms
    
    // 1. Sublinear algorithm (fastest for most cases)
    let result_sublinear = cuda_sparse_fft(
        &signal,
        2,  // Expected sparsity
        0,  // Device ID
        Some(SparseFFTAlgorithm::Sublinear),
        Some(WindowFunction::Hann)
    ).unwrap();
    
    // 2. CompressedSensing algorithm (best accuracy)
    let result_cs = cuda_sparse_fft(
        &signal,
        2,
        0,
        Some(SparseFFTAlgorithm::CompressedSensing),
        Some(WindowFunction::Hann)
    ).unwrap();
    
    // 3. Iterative algorithm (best for noisy signals)
    let result_iterative = cuda_sparse_fft(
        &signal,
        2,
        0,
        Some(SparseFFTAlgorithm::Iterative),
        Some(WindowFunction::Hann)
    ).unwrap();
    
    // 4. Frequency Pruning algorithm (best for large signals)
    let result_frequency_pruning = cuda_sparse_fft(
        &signal,
        2,
        0,
        Some(SparseFFTAlgorithm::FrequencyPruning),
        Some(WindowFunction::Hann)
    ).unwrap();
    
    // Batch processing for multiple signals
    let signals = vec![
        vec![1.0, 2.0, 3.0, 4.0],
        vec![4.0, 3.0, 2.0, 1.0],
    ];
    
    let batch_results = cuda_batch_sparse_fft(
        &signals,
        2,  // Expected sparsity
        0,  // Device ID
        Some(SparseFFTAlgorithm::Sublinear),
        Some(WindowFunction::Hann)
    ).unwrap();
    
    println!("CUDA-accelerated sparse FFT completed!");
    println!("Found {} significant frequencies", result_sublinear.values.len());
    println!("Computation time: {:?}", result_sublinear.computation_time);
}
```

The GPU acceleration module provides:

1. **Multiple Algorithm Support**:
   - `Sublinear`: Fastest algorithm for most cases
   - `CompressedSensing`: Highest accuracy for clean signals
   - `Iterative`: Best performance on noisy signals
   - `FrequencyPruning`: Excellent for very large signals with clustered frequency components

2. **Memory Management**:
   - Efficient buffer allocation and caching strategies
   - Automatic cleanup and resource management
   - Support for pinned, device, and unified memory

3. **Performance Features**:
   - Batch processing for multiple signals
   - Automatic performance tuning based on signal characteristics
   - Hardware-specific optimizations

4. **Platform Support**:
   - CUDA for NVIDIA GPUs
   - Extensible architecture for other GPU platforms
   - Automatic CPU fallback when GPU is unavailable

## Performance

The FFT implementation in this module is optimized for performance:

- Uses the `rustfft` crate for the core FFT algorithm
- Provides SIMD-accelerated implementations when available
- Includes specialized implementations for common cases 
- Parallel implementations for large arrays using Rayon
- GPU acceleration for even greater performance on supported hardware
- Advanced parallel planning system for creating and executing multiple FFT plans concurrently
- Offers automatic selection of the most efficient algorithm
- Smart thresholds to choose between sequential and parallel implementations

### Parallel Planning

The parallel planning system allows for concurrent creation and execution of FFT plans:

```rust
use scirs2_fft::{ParallelPlanner, ParallelExecutor, ParallelPlanningConfig};
use num_complex::Complex64;

// Configure parallel planning
let config = ParallelPlanningConfig {
    parallel_threshold: 1024,  // Only use parallelism for FFTs >= 1024 elements
    max_threads: None,         // Use all available threads
    parallel_execution: true,  // Enable parallel execution
    ..Default::default()
};

// Create a parallel planner
let planner = ParallelPlanner::new(Some(config.clone()));

// Create multiple plans in parallel
let plan_specs = vec![
    (vec![1024], true, Default::default()),       // 1D FFT of size 1024
    (vec![512, 512], true, Default::default()),   // 2D FFT of size 512x512
    (vec![128, 128, 128], true, Default::default()), // 3D FFT of size 128x128x128
];

let results = planner.plan_multiple(&plan_specs).unwrap();

// Use the plans for execution
let plan = &results[0].plan;
let executor = ParallelExecutor::new(plan.clone(), Some(config));

// Create input data
let size = plan.shape().iter().product::<usize>();
let input = vec![Complex64::new(1.0, 0.0); size];
let mut output = vec![Complex64::default(); size];

// Execute the FFT plan in parallel
executor.execute(&input, &mut output).unwrap();

// Batch execution of multiple FFTs
let batch_size = 4;
let mut inputs = Vec::with_capacity(batch_size);
let mut outputs = Vec::with_capacity(batch_size);

// Create batch data
for _ in 0..batch_size {
    inputs.push(vec![Complex64::new(1.0, 0.0); size]);
    outputs.push(vec![Complex64::default(); size]);
}

// Get mutable references to outputs
let mut output_refs: Vec<&mut [Complex64]> = outputs.iter_mut()
    .map(|v| v.as_mut_slice())
    .collect();

// Execute batch of FFTs in parallel
executor.execute_batch(
    &inputs.iter().map(|v| v.as_slice()).collect::<Vec<_>>(),
    &mut output_refs
).unwrap();
```

Benefits of using the parallel planning system:
- Create multiple FFT plans concurrently, reducing initialization time
- Execute FFTs in parallel for better hardware utilization
- Batch processing for multiple input signals
- Configurable thresholds to control when parallelism is used
- Worker pool management for optimal thread usage

## Testing

To run the tests for this crate:

```bash
# Run only library tests (recommended to avoid timeouts with large-scale tests)
cargo test --lib

# Or use the Makefile.toml task (if cargo-make is installed)
cargo make test

# Run all tests including benchmarks (may timeout on slower systems)
cargo test
```

Some of the extensive benchmark tests with large FFT sizes may timeout during testing. We recommend using the `--lib` flag to run only the core library tests.

## Contributing

See the [CONTRIBUTING.md](../CONTRIBUTING.md) file for contribution guidelines.

## License

This project is dual-licensed under:

- [MIT License](../LICENSE-MIT)
- [Apache License Version 2.0](../LICENSE-APACHE)

You can choose to use either license. See the [LICENSE](../LICENSE) file for details.
