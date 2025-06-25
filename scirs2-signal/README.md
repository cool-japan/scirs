# SciRS2 Signal

[![crates.io](https://img.shields.io/crates/v/scirs2-signal.svg)](https://crates.io/crates/scirs2-signal)
[[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)]](../LICENSE)
[![Documentation](https://img.shields.io/docsrs/scirs2-signal)](https://docs.rs/scirs2-signal)

Production-ready signal processing module for the SciRS2 scientific computing library. This module provides core signal processing tools including filtering, convolution, spectral analysis, and wavelet transforms.

## Core Features (Production-Ready)

- **Signal Generation**: Essential waveform generation functions
- **Digital Filtering**: Comprehensive IIR and FIR filter design and application
- **Convolution & Correlation**: Efficient signal convolution and correlation operations
- **Spectral Analysis**: Fundamental frequency domain analysis (FFT, PSD, spectrograms)
- **Wavelet Transforms**: Core wavelet analysis (DWT, CWT) with multiple families
- **Peak Detection**: Robust algorithms for finding and analyzing peaks
- **Signal Measurements**: Standard signal quality metrics (RMS, SNR, THD)
- **Basic Resampling**: Up/down sampling and rate conversion

## Installation

Add the following to your `Cargo.toml`:

```toml
[dependencies]
scirs2-signal = "0.1.0-alpha.6"
scirs2-core = "0.1.0-alpha.6"
ndarray = "0.16.1"
```

> **Note**: This is an alpha release. The API may change before the stable 1.0 release.

Basic usage examples:

```rust
use scirs2_signal::{waveforms, filter, convolve, spectral, peak};
use scirs2_core::error::CoreResult;
use ndarray::{Array1, array};
use std::f64::consts::PI;

// Signal generation
fn waveform_example() -> CoreResult<()> {
    // Create time array
    let t = Array1::linspace(0.0, 1.0, 1000); // 1000 points from 0 to 1 second
    let freq = 5.0; // 5 Hz
    
    // Generate basic waveforms
    let sine = t.mapv(|x| (2.0 * PI * freq * x).sin());
    let square = waveforms::square(&t, freq, 0.0, 1.0, 0.5)?;
    let chirp = waveforms::chirp(&t, 1.0, 1.0, 10.0)?;
    
    println!("Generated sine, square, and chirp signals");
    
    Ok(())
}

// Filtering example
fn filter_example() -> CoreResult<()> {
    // Create a noisy signal
    let t = Array1::linspace(0.0, 1.0, 1000);
    let signal = t.mapv(|x| (2.0 * PI * 5.0 * x).sin()); // 5 Hz sine wave
    
    // Add some noise (simplified)
    let noisy_signal = signal.mapv(|x| x + 0.1 * rand::random::<f64>());
    
    // Design a low-pass Butterworth filter
    let fs = 1000.0; // Sample rate: 1000 Hz
    let cutoff = 10.0; // Cutoff frequency: 10 Hz
    let order = 4; // Filter order
    let (b, a) = filter::butter(order, &[cutoff / (fs / 2.0)], "lowpass", None, None)?;
    
    // Apply the filter (zero-phase)
    let filtered = filter::filtfilt(&b, &a, &noisy_signal)?;
    
    println!("Applied Butterworth low-pass filter");
    
    Ok(())
}

// Convolution example
fn convolution_example() -> CoreResult<()> {
    // Create a signal
    let signal = array![1.0, 2.0, 3.0, 4.0, 5.0];
    
    // Create a kernel
    let kernel = array![0.1, 0.2, 0.4, 0.2, 0.1];
    
    // Perform convolution
    let result = convolve::convolve(&signal, &kernel, "same")?;
    
    println!("Convolution result: {:?}", result);
    
    Ok(())
}

// Spectral analysis example
fn spectral_example() -> CoreResult<()> {
    // Create a signal with multiple frequency components
    let fs = 1000.0; // Sample rate: 1000 Hz
    let t = Array1::linspace(0.0, 1.0, 1000);
    
    // Create signal with 5 Hz and 20 Hz components
    let signal = t.mapv(|x| {
        (2.0 * PI * 5.0 * x).sin() + 0.5 * (2.0 * PI * 20.0 * x).sin()
    });
    
    // Compute the power spectral density using Welch's method
    let (f, psd) = spectral::welch(&signal, None, None, None, None, Some(fs))?;
    
    // Find peaks in the PSD
    let peaks = peak::find_peaks(&psd, None, None, None, None)?;
    
    println!("Found {} peaks in the power spectrum", peaks.len());
    for (i, &idx) in peaks.iter().enumerate() {
        if idx < f.len() && idx < psd.len() {
            println!("Peak {}: frequency = {:.1} Hz, power = {:.2}", 
                     i+1, f[idx], psd[idx]);
        }
    }
    
    Ok(())
}

// Resampling example (basic)
fn resampling_example() -> CoreResult<()> {
    // Create a signal
    let t = Array1::linspace(0.0, 1.0, 1000);
    let signal = t.mapv(|x| (2.0 * std::f64::consts::PI * 5.0 * x).sin());
    
    // Basic resampling (Note: Advanced resampling in future releases)
    // let resampled = resample::resample(&signal, 1000, 2000)?;
    
    println!("Original signal: {} points", signal.len());
    
    Ok(())
}
```

## Components

### Waveforms

Functions for generating signal waveforms:

```rust
use scirs2_signal::waveforms::{
    time_array,             // Create a time array
    sine,                   // Sine wave
    cosine,                 // Cosine wave
    square,                 // Square wave
    sawtooth,               // Sawtooth wave
    triangle,               // Triangle wave
    chirp,                  // Frequency sweep (chirp)
    sweep_poly,             // Polynomial frequency sweep
    noise,                  // Noise generator
    impulse,                // Impulse signal
    step,                   // Step signal
    gaussian,               // Gaussian pulse
    gabor,                  // Gabor wavelet
};
```

### Digital Filtering

Comprehensive filtering capabilities:

```rust
use scirs2_signal::filter::{
    // IIR Filter Design
    butter,                 // Butterworth filter design
    cheby1,                 // Chebyshev Type I filter design
    cheby2,                 // Chebyshev Type II filter design
    ellip,                  // Elliptic filter design
    bessel,                 // Bessel filter design
    
    // FIR Filter Design
    firwin,                 // Window-based FIR design
    remez,                  // Parks-McClellan optimal FIR design
    
    // Filter Application
    lfilter,                // Apply filter to data
    filtfilt,               // Zero-phase filtering
    
    // Specialized Filters
    notch_filter,           // Notch filter design
    comb_filter,            // Comb filter design
    allpass_filter,         // Allpass filter design
    
    // Filter Analysis
    analyze_filter,         // Analyze filter properties
    check_filter_stability, // Check filter stability
};
use scirs2_signal::savgol::savgol_filter; // Savitzky-Golay filter
```

### Convolution

Functions for signal convolution:

```rust
use scirs2_signal::convolve::{
    convolve,               // 1D convolution
    convolve2d,             // 2D convolution
    fftconvolve,            // FFT-based convolution
    correlate,              // 1D correlation
    correlate2d,            // 2D correlation
};
```

### Spectral Analysis

Fundamental frequency domain analysis:

```rust
use scirs2_signal::spectral::{
    periodogram,            // Periodogram power spectral density estimate
    welch,                  // Welch's method for PSD estimation
    spectrogram,            // Time-frequency representation
};
use scirs2_signal::stft::ShortTimeFft; // Short-time Fourier transform class

// Note: For direct FFT operations, use scirs2-fft crate
use scirs2_fft::fft::{
    fft,                    // Fast Fourier Transform
    ifft,                   // Inverse FFT
    rfft,                   // Real FFT
    irfft,                  // Inverse real FFT
};
```

### Peak Detection

Functions for finding peaks in signals:

```rust
use scirs2_signal::peak::{
    find_peaks,             // Find peaks in data
    find_peaks_cwt,         // Find peaks using continuous wavelet transform
    peak_prominences,       // Calculate peak prominences
    peak_widths,            // Calculate peak widths
};
```

### Resampling

Basic resampling operations:

```rust
use scirs2_signal::resample::{
    resample,               // Resample signal to new sampling rate
    // Note: Advanced resampling features are planned for future releases
};
```

### Signal Measurements

Standard signal quality metrics:

```rust
use scirs2_signal::measurements::{
    rms,                    // Root mean square
    snr,                    // Signal-to-noise ratio
    thd,                    // Total harmonic distortion
    peak_to_peak,           // Peak-to-peak measurement
    peak_to_rms,            // Peak-to-RMS ratio
};
```

### Wavelet Transforms

Core wavelet analysis with proven algorithms:

```rust
use scirs2_signal::{
    dwt::{Wavelet, dwt_decompose, dwt_reconstruct, wavedec, waverec},
    wavelets::{cwt, morlet, ricker},
    denoise::{denoise_wavelet, ThresholdMethod, ThresholdSelect},
};

// Supported wavelet families (stable)
let wavelets = [
    Wavelet::Haar,          // Haar wavelet
    Wavelet::DB(4),         // Daubechies wavelet (4 vanishing moments)
    Wavelet::Sym(4),        // Symlet wavelet (4 vanishing moments)
    Wavelet::Coif(3),       // Coiflet wavelet (3 vanishing moments)
    Wavelet::Meyer,         // Meyer wavelet
];

// Discrete Wavelet Transform (DWT)
let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
let (approx, detail) = dwt_decompose(&signal, Wavelet::DB(4), None)?;
let reconstructed = dwt_reconstruct(&approx, &detail, Wavelet::DB(4))?;

// Multi-level decomposition
let coeffs = wavedec(&signal, Wavelet::DB(4), 3)?; // 3 levels
let reconstructed = waverec(&coeffs, Wavelet::DB(4))?;

// Continuous Wavelet Transform (CWT)
let scales = vec![1.0, 2.0, 4.0, 8.0, 16.0];
let cwt_result = cwt(&signal, morlet, &scales)?;

// Wavelet denoising
let denoised = denoise_wavelet(
    &noisy_signal,
    Wavelet::DB(4),
    3,                      // Decomposition levels
    ThresholdMethod::Soft,  // Soft thresholding
    ThresholdSelect::Universal, // Universal threshold
    None,
)?;
```

## Integration with Other SciRS2 Modules

Seamless integration with the SciRS2 ecosystem:

```rust
use scirs2_signal::spectral;
use scirs2_fft::fft;
use scirs2_core::error::CoreResult;
use ndarray::Array1;

// Combined usage with scirs2-fft
let data: Array1<f64> = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

// Direct FFT computation
let fft_result = fft::fft(&data)?;

// Power spectral density estimation
let (freq, psd) = spectral::welch(&data, None, None, None, None, Some(1000.0))?;

// Filter design and application
let (b, a) = filter::butter(4, &[0.1], "lowpass", None, None)?;
let filtered = filter::filtfilt(&b, &a, &data)?;
```

## Development Status

**Current Release**: 0.1.0-alpha.6 (Final Alpha)

### Production-Ready Features ✅
- Digital filtering (IIR/FIR design and application)
- Basic spectral analysis (periodogram, Welch's method, STFT)
- Core wavelet transforms (DWT, CWT)
- Signal convolution and correlation
- Peak detection and signal measurements
- Waveform generation and basic resampling

### Experimental Features ⚠️
- Advanced time-frequency analysis
- 2D wavelet transforms
- Advanced denoising and restoration
- Real-time processing capabilities

*Note: Experimental features may have API changes or require additional validation.*

## Contributing

See the [CONTRIBUTING.md](../CONTRIBUTING.md) file for contribution guidelines.

## License

This project is dual-licensed under:

- [MIT License](../LICENSE-MIT)
- [Apache License Version 2.0](../LICENSE-APACHE)

You can choose to use either license. See the [LICENSE](../LICENSE) file for details.
