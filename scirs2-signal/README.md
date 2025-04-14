# SciRS2 Signal

[![crates.io](https://img.shields.io/crates/v/scirs2-signal.svg)](https://crates.io/crates/scirs2-signal)
[![License](https://img.shields.io/crates/l/scirs2-signal.svg)](../LICENSE)
[![Documentation](https://img.shields.io/docsrs/scirs2-signal)](https://docs.rs/scirs2-signal)

Signal processing module for the SciRS2 scientific computing library. This module provides tools for signal creation, filtering, convolution, peak detection, spectral analysis, and more.

## Features

- **Signal Generation**: Functions for creating various waveforms
- **Filtering**: Various filter designs and implementations
- **Convolution**: Efficient convolution operations
- **Spectral Analysis**: Tools for frequency domain analysis
- **Peak Detection**: Algorithms for finding peaks in signals
- **Resampling**: Methods for changing sampling rates
- **Measurements**: Signal quality and statistical measurements

## Installation

Add the following to your `Cargo.toml`:

```toml
[dependencies]
scirs2-signal = "0.1.0-alpha.1"
ndarray = "0.16.1"
```

Basic usage examples:

```rust
use scirs2_signal::{waveforms, filter, convolve, spectral, peak, resample};
use scirs2_core::error::CoreResult;
use ndarray::{Array1, array};

// Signal generation
fn waveform_example() -> CoreResult<()> {
    // Generate a sine wave
    let t = waveforms::time_array(0.0, 1.0, 1000)?; // 1000 points from 0 to 1 second
    let freq = 5.0; // 5 Hz
    let sine = waveforms::sine(&t, freq, 0.0, 1.0)?;
    
    // Generate a square wave
    let square = waveforms::square(&t, freq, 0.0, 1.0, 0.5)?;
    
    // Generate a chirp signal (frequency sweep)
    let chirp = waveforms::chirp(&t, 1.0, 1.0, 10.0, None)?;
    
    println!("Generated sine, square, and chirp signals");
    
    Ok(())
}

// Filtering example
fn filter_example() -> CoreResult<()> {
    // Create a noisy signal
    let t = waveforms::time_array(0.0, 1.0, 1000)?;
    let signal = waveforms::sine(&t, 5.0, 0.0, 1.0)?; // 5 Hz sine wave
    let noise = waveforms::noise(&signal.shape(), 0.0, 0.2, None)?; // Gaussian noise
    let noisy_signal = &signal + &noise;
    
    // Design a low-pass Butterworth filter
    let fs = 1000.0; // Sample rate: 1000 Hz
    let cutoff = 10.0; // Cutoff frequency: 10 Hz
    let order = 4; // Filter order
    let (b, a) = filter::butter(order, &[cutoff], "lowpass", None, Some(fs))?;
    
    // Apply the filter
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
    let t = waveforms::time_array(0.0, 1.0, 1000)?;
    
    // 5 Hz and 20 Hz components
    let signal_5hz = waveforms::sine(&t, 5.0, 0.0, 1.0)?;
    let signal_20hz = waveforms::sine(&t, 20.0, 0.0, 0.5)?;
    let signal = &signal_5hz + &signal_20hz;
    
    // Compute the power spectral density
    let (f, psd) = spectral::welch(&signal, None, None, None, None, Some(fs))?;
    
    // Find peaks in the PSD
    let peaks = peak::find_peaks(&psd, None, None, None, None)?;
    
    println!("Found {} peaks in the power spectrum", peaks.len());
    for (i, &idx) in peaks.iter().enumerate() {
        println!("Peak {}: frequency = {} Hz, power = {}", 
                 i+1, f[idx], psd[idx]);
    }
    
    Ok(())
}

// Resampling example
fn resampling_example() -> CoreResult<()> {
    // Create a signal
    let t = waveforms::time_array(0.0, 1.0, 1000)?;
    let signal = waveforms::sine(&t, 5.0, 0.0, 1.0)?;
    
    // Resample to a higher rate (upsampling)
    let upsampled = resample::resample(&signal, 1000, 4000, None, None)?;
    
    // Resample to a lower rate (downsampling)
    let downsampled = resample::resample(&signal, 1000, 500, None, None)?;
    
    println!("Original signal: {} points", signal.len());
    println!("Upsampled signal: {} points", upsampled.len());
    println!("Downsampled signal: {} points", downsampled.len());
    
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

### Filtering

Signal filtering functions:

```rust
use scirs2_signal::filter::{
    // Filter Design
    butter,                 // Butterworth filter design
    cheby1,                 // Chebyshev Type I filter design
    cheby2,                 // Chebyshev Type II filter design
    ellip,                  // Elliptic filter design
    bessel,                 // Bessel filter design
    
    // Filtering Functions
    lfilter,                // Filter data along one dimension
    filtfilt,               // Zero-phase filtering
    sosfilt,                // Filter data using second-order sections
    
    // Specific Filters
    medfilt,                // Median filter
    wiener,                 // Wiener filter
    savgol_filter,          // Savitzky-Golay filter
    
    // Frequency Transformations
    bilinear,               // Bilinear transform
    lp2lp,                  // Transform lowpass to lowpass
    lp2hp,                  // Transform lowpass to highpass
    lp2bp,                  // Transform lowpass to bandpass
    lp2bs,                  // Transform lowpass to bandstop
};
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

Functions for frequency domain analysis:

```rust
use scirs2_signal::spectral::{
    fft,                    // Fast Fourier Transform
    ifft,                   // Inverse FFT
    rfft,                   // Real FFT
    irfft,                  // Inverse real FFT
    fftfreq,                // FFT frequency bins
    fftshift,               // Shift zero-frequency component to center
    spectrogram,            // Time-frequency representation
    stft,                   // Short-time Fourier transform
    istft,                  // Inverse short-time Fourier transform
    periodogram,            // Periodogram power spectral density estimate
    welch,                  // Welch's power spectral density estimate
    csd,                    // Cross spectral density
    coherence,              // Magnitude squared coherence
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

Functions for signal resampling:

```rust
use scirs2_signal::resample::{
    resample,               // Resample signal to new sampling rate
    resample_poly,          // Resample using polyphase filtering
    decimate,               // Downsample by an integer factor
    upfirdn,                // Upsample, apply FIR filter, then downsample
};
```

### Measurements

Functions for signal measurements:

```rust
use scirs2_signal::measurements::{
    snr,                    // Signal-to-noise ratio
    psnr,                   // Peak signal-to-noise ratio
    thd,                    // Total harmonic distortion
    enob,                   // Effective number of bits
    sinad,                  // Signal-to-noise and distortion ratio
    sfdr,                   // Spurious-free dynamic range
};
```

## Integration with FFT Module

This module integrates with the `scirs2-fft` module for spectral analysis:

```rust
use scirs2_signal::spectral;
use scirs2_fft::fft;

// Direct FFT using scirs2-fft
let data = array![1.0, 2.0, 3.0, 4.0];
let fft_result = fft::fft(&data).unwrap();

// Spectral analysis using scirs2-signal
let (freq, psd) = spectral::periodogram(&data, None, None, None, Some(1000.0)).unwrap();
```

## Contributing

See the [CONTRIBUTING.md](../CONTRIBUTING.md) file for contribution guidelines.

## License

This project is licensed under the Apache License, Version 2.0 - see the [LICENSE](../LICENSE) file for details.