# scirs2-fft

[![crates.io](https://img.shields.io/crates/v/scirs2-fft)](https://crates.io/crates/scirs2-fft)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](../LICENSE)
[![Documentation](https://img.shields.io/docsrs/scirs2-fft)](https://docs.rs/scirs2-fft)

**Fast Fourier Transform library for Rust, modeled after SciPy's fft module.**

`scirs2-fft` provides a comprehensive suite of frequency-domain algorithms — FFT, RFFT, DCT/DST, STFT, NUFFT, sparse FFT, fractional FFT, wavelet packets, polyphase filter banks, and more — all in pure Rust using OxiFFT as the core FFT engine.

## Installation

```toml
[dependencies]
scirs2-fft = "0.5.0"
```

With parallel processing:

```toml
[dependencies]
scirs2-fft = { version = "0.5.0", features = ["parallel"] }
```

## Features (v0.5.0)

### Core Transforms

| Function | SciPy Equivalent | Description |
|----------|-----------------|-------------|
| `fft` / `ifft` | `scipy.fft.fft` | Complex-to-complex 1D FFT |
| `rfft` / `irfft` | `scipy.fft.rfft` | Real-input optimized FFT |
| `fft2` / `ifft2` | `scipy.fft.fft2` | 2D FFT |
| `fftn` / `ifftn` | `scipy.fft.fftn` | N-dimensional FFT |
| `rfft2` / `irfft2` | `scipy.fft.rfft2` | 2D real FFT |
| `rfftn` / `irfftn` | `scipy.fft.rfftn` | N-dimensional real FFT |
| `fftfreq` / `rfftfreq` | `scipy.fft.fftfreq` | Sample frequency computation |
| `fftshift` / `ifftshift` | `scipy.fft.fftshift` | Zero-frequency centering |
| `next_fast_len` | `scipy.fft.next_fast_len` | Optimal FFT size selection |

### DCT and DST Variants

- DCT types I, II, III, IV, V, VI, VII, VIII
- DST types I, II, III, IV, V, VI, VII, VIII
- Inverse transforms for all types
- MDCT (Modified DCT) and MDST
- N-dimensional DCT/DST

### Window Functions Library

100+ window functions including:
- Standard: Hann, Hamming, Blackman, Bartlett, flat-top
- Kaiser, Gaussian, Tukey, Parzen, Bohman
- Nuttall, Blackman-Harris, Blackman-Nuttall
- DPSS (discrete prolate spheroidal sequences) for multi-taper
- Exponential, triangular, rectangular (boxcar)
- General cosine and general Hamming families
- Kaiser-Bessel derived (KBD)

### Short-Time Fourier Transform

- `stft` / `istft` with configurable window, overlap, and boundary handling
- `spectrogram` (power spectral density)
- Normalized spectrogram for visualization
- Waterfall plot data generation (3D mesh, line stacks)
- Coherence and cross-spectral density

### Sparse FFT (Sub-Linear Time)

- Sublinear sparse FFT (O(k log n) for k-sparse signals)
- Compressed sensing-based sparse FFT
- Iterative sparse FFT (robust to noise)
- Frequency pruning variant
- Prony method for damped sinusoid recovery
- MUSIC algorithm for super-resolution spectral estimation
- Batch sparse FFT (parallel CPU processing)
- GPU-accelerated sparse FFT (CUDA/ROCm/SYCL via feature flag)

### Specialized Transforms

- Fractional Fourier Transform (FrFT) — 3 algorithm variants (Ozaktas-Arikan, Candan, sampling)
- Chirp-Z Transform (CZT) — FFT on arbitrary contours in the z-plane
- Non-Uniform FFT (NUFFT) types 1, 2, and 3 (Gaussian / sinc interpolation)
- Fast Hankel Transform (FHT)
- Hilbert transform (analytic signal)
- Hartley transform
- Number-theoretic Transform (NTT) over prime fields
- Bluestein's algorithm for prime-length FFT
- Mixed-radix FFT (arbitrary composite lengths)

### Spectral Estimation

- Lomb-Scargle periodogram for irregularly sampled data
- MUSIC (Multiple Signal Classification) algorithm
- ESPRIT algorithm for frequency estimation
- Burg method (AR model-based PSD)
- Welch's method for PSD estimation
- Multitaper spectral analysis (DPSS windows)

### Wavelet Packets

- Wavelet packet transform tree
- Best-basis selection (entropy criterion)
- Reconstruction from wavelet packet tree
- Available wavelet families: Daubechies, Symlets, Coiflets, Biorthogonal

### Polyphase Filter Bank

- Analysis and synthesis polyphase filter banks
- Critically sampled and oversampled configurations
- Perfect reconstruction condition verification
- DFT modulated filter banks

### Hilbert-Huang Transform (HHT)

- Empirical Mode Decomposition (EMD) into Intrinsic Mode Functions (IMFs)
- Instantaneous frequency and amplitude via Hilbert transform
- Ensemble EMD (EEMD) and Complete EEMD (CEEMDAN) for noise-assisted decomposition
- Hilbert spectrum visualization data

### Multidimensional FFT Utilities

- Row-column decomposition for 2D FFT
- Separable N-dimensional FFT plans
- Mixed-radix N-dimensional FFT
- In-place 2D FFT with memory-efficient tiling

### Convolution and Correlation

- Overlap-save and overlap-add convolution
- Circular convolution via FFT
- Cross-correlation and autocorrelation
- FFT-based polynomial multiplication

### Plan Caching and Execution

- Automatic plan reuse for repeated same-size transforms (thread-safe cache)
- Parallel planner: create multiple plans concurrently
- Parallel executor: execute batch FFTs across worker threads
- Memory-efficient streaming FFT for large arrays

## Usage Examples

### Basic 1D FFT

```rust
use scirs2_fft::{fft, ifft, rfft};

let signal = vec![1.0_f64, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 0.0];

// Complex FFT
let spectrum = fft(&signal, None)?;

// Inverse FFT
let recovered = ifft(&spectrum, None)?;

// Real FFT (more efficient for real inputs)
let real_spectrum = rfft(&signal, None)?;  // length = n/2 + 1
```

### DCT

```rust
use scirs2_fft::dct::{dct, idct, DctType};

let data = vec![1.0_f64, 2.0, 3.0, 4.0];
let coeffs = dct(&data, DctType::II, None)?;   // DCT-II (most common)
let back = idct(&coeffs, DctType::II, None)?;
```

### STFT and Spectrogram

```rust
use scirs2_fft::{stft, spectrogram, Window};

let fs = 44100.0_f64;
let signal: Vec<f64> = (0..44100).map(|i| (2.0 * std::f64::consts::PI * 440.0 * i as f64 / fs).sin()).collect();

// Short-Time Fourier Transform
let (freqs, times, stft_mat) = stft(&signal, Window::Hann, 1024, Some(512), None, Some(fs), None, None)?;

// Power spectral density spectrogram
let (freqs, times, psd) = spectrogram(&signal, Some(fs), Some(Window::Hann), Some(1024), Some(512), None, None, Some("density"), Some("psd"))?;
```

### Fractional Fourier Transform

```rust
use scirs2_fft::fractional_ft::frft;

let signal: Vec<f64> = (0..256).map(|i| (0.1 * i as f64).sin()).collect();
// alpha = 0.5: halfway between time and frequency domain
let result = frft(&signal, 0.5, None)?;
```

### Sparse FFT

```rust
use scirs2_fft::sparse_fft::{sparse_fft, SparseFFTAlgorithm};

// Signal with only a few significant frequency components
let result = sparse_fft(&signal, 5, SparseFFTAlgorithm::Sublinear, None)?;
println!("Dominant frequencies: {:?}", result.indices);
println!("Their amplitudes: {:?}", result.values);
```

### Lomb-Scargle Periodogram

```rust
use scirs2_fft::spectral_analysis::lomb_scargle;

// Irregularly sampled time series
let times: Vec<f64> = vec![0.0, 0.3, 0.7, 1.1, 2.0, 3.5, 5.1];
let values: Vec<f64> = vec![1.0, 0.5, -0.3, -0.9, -0.2, 0.8, 0.4];
let frequencies: Vec<f64> = (1..=50).map(|i| i as f64 * 0.1).collect();

let power = lomb_scargle(&times, &values, &frequencies, true)?;
```

### NTT (Number-Theoretic Transform)

```rust
use scirs2_fft::ntt::ntt;

let data: Vec<u64> = vec![1, 2, 3, 4, 5, 6, 7, 8];
let prime = 998_244_353_u64;  // NTT-friendly prime
let result = ntt(&data, prime)?;
```

### Wavelet Packet Transform

```rust
use scirs2_fft::wavelet_packets::{WaveletPacketTree, WaveletFamily};

let signal: Vec<f64> = (0..256).map(|i| (i as f64 * 0.1).sin()).collect();
let tree = WaveletPacketTree::new(&signal, WaveletFamily::Daubechies(4), 4)?;
let best_basis = tree.best_basis()?;
```

## Feature Flags

| Feature | Description |
|---------|-------------|
| `parallel` | Multi-threaded FFT execution and batch processing via Rayon |
| `simd` | SIMD-accelerated butterfly operations and window application |
| `gpu` | GPU-accelerated sparse FFT (CUDA/ROCm/SYCL, requires `scirs2-core` gpu) |
| `cuda` | NVIDIA CUDA sparse FFT backend |
| `hip` | AMD ROCm/HIP sparse FFT backend |
| `sycl` | Cross-platform SYCL sparse FFT backend |

## Links

- [API Documentation](https://docs.rs/scirs2-fft)
- [SciRS2 Repository](https://github.com/cool-japan/scirs)

## License

Licensed under the Apache License 2.0. See [LICENSE](../LICENSE) for details.
