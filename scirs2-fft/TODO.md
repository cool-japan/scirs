# scirs2-fft TODO

This module provides Fast Fourier Transform functionality similar to SciPy's fft module.

## Current Status

- [x] Set up module structure
- [x] Error handling
- [x] FFT and inverse FFT (1D, 2D, and N-dimensional)
- [x] Real FFT and inverse Real FFT (optimized for real input)
- [x] Discrete Cosine Transform (DCT) types I-IV
- [x] Discrete Sine Transform (DST) types I-IV
- [x] Helper functions (fftshift, ifftshift, fftfreq, rfftfreq)
- [x] Window functions (Hann, Hamming, Blackman, etc.)
- [x] Integration with ndarray for multi-dimensional arrays

## Future Tasks

- [x] Fix remaining Clippy warnings
  - [x] Address needless_range_loop warnings
  - [x] Fix comparison_chain warnings
  - [x] Address only_used_in_recursion warnings
  - [x] Fix doc_overindented_list_items warning
- [x] Performance optimizations
  - [x] Parallelization of larger transforms
  - [x] More efficient memory usage for large arrays
    - [x] Implemented memory-efficient 2D FFT
    - [x] Added streaming FFT for processing large arrays in chunks
    - [x] Created in-place FFT operations to minimize allocations
- [x] Add more examples and documentation
  - [x] Tutorial for common FFT operations (fft_tutorial.rs)
  - [x] Examples for spectral analysis (spectral_analysis.rs)
  - [x] Memory-efficient FFT examples (memory_efficient_fft.rs)
- [x] Additional functionality
  - [x] Short-time Fourier transform (STFT) interface
  - [x] Non-uniform FFT
  - [x] Hilbert transform
  - [x] Fractional Fourier transform (implementation with complex number handling needs improvement)
- [x] Add visualization utilities
  - [x] Spectrograms
  - [x] Waterfall plots

## Enhanced FFT API and Interoperability

- [x] Implement array interoperability features
  - [x] Support for various array-like objects
  - [x] Backend system similar to SciPy's backend model
  - [x] Pluggable FFT implementations
- [x] Enhance worker management for parallelization
  - [x] Thread pool configuration
  - [x] Worker count control similar to SciPy's `set_workers`/`get_workers`
  - [x] Thread safety guarantees for all operations
- [x] Add context managers for FFT settings
  - [x] Backend selection context
  - [x] Worker count context
  - [x] Plan caching control

## Fast Hankel Transform

- [x] Implement Fast Hankel Transform (FHT)
  - [x] Forward transform (fht)
  - [x] Inverse transform (ifht)
  - [x] Optimal offset calculation (fhtoffset)
  - [x] Support for biased transforms
  - [x] Comprehensive examples with visualizations

## Multidimensional Transform Enhancements

- [x] Improve N-dimensional transforms
  - [x] Optimized memory access patterns
  - [x] Advanced chunking strategies for large arrays
  - [x] Axis-specific operations with optional normalization
  - [ ] Advanced striding support

## Plan Caching and Optimization

- [x] Implement advanced planning strategies
  - [x] Plan caching mechanism for repeated transforms
  - [ ] Auto-tuning for hardware-specific optimizations
  - [ ] Plan serialization for reuse across runs
  - [x] Plan sharing across threads
- [x] Add `next_fast_len` and `prev_fast_len` helpers
  - [x] Optimal sizing for FFT speed
  - [x] Support for SIMD-friendly sizes
  - [x] Automatic padding strategies

## Extended Transform Types

- [x] Implement additional transform variants
  - [x] Higher-order DCT types (V-VIII)
  - [x] Higher-order DST types (V-VIII)
  - [x] Hartley transform
  - [x] Modified DCT/DST (MDCT/MDST)
  - [x] Z-transform for non-uniform frequency spacing (CZT - Chirp Z-Transform)

## Custom Window Functions

- [x] Extend window function support
  - [x] Comprehensive window catalog matching SciPy
  - [x] Window design tools and generators
  - [x] Window visualization utilities
  - [x] Window properties analysis (energy, bandwidth)

## Long-term Goals

- [ ] Performance comparable to or better than FFTW
  - [ ] Benchmark suite for comparison
  - [ ] Performance optimization database
  - [ ] Auto-tuning for specific hardware
- [ ] GPU-accelerated implementations
  - [ ] CUDA/HIP/SYCL support
  - [ ] Memory management for large transforms
  - [ ] Hybrid CPU/GPU execution strategies
- [ ] Support for distributed FFT computations
  - [ ] MPI-based distributed transforms
  - [ ] Domain decomposition strategies
  - [ ] Network efficiency optimizations
- [ ] Integration with signal processing and filtering
  - [ ] Filter design and application in frequency domain
  - [ ] Convolution optimizations
  - [ ] Signal analysis toolkit
- [ ] Advanced time-frequency analysis tools
  - [ ] Enhanced spectrogram tools
  - [ ] Wavelet transform integration
  - [ ] Reassignment methods
  - [ ] Synchrosqueezing transforms
- [ ] Support for specialized hardware (FPGA, custom accelerators)
  - [ ] Hardware-specific optimizations
  - [ ] Offloading strategies
  - [ ] Custom kernels for different architectures