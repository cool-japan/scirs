# Spectral Utilities Guide

The spectral utilities module in SciRS2 provides a collection of functions for analyzing and processing spectral representations of signals. These utilities extend beyond basic spectral analysis to provide deeper insights into frequency domain characteristics.

## Table of Contents

1. [Introduction](#introduction)
2. [Spectral Descriptors](#spectral-descriptors)
3. [Spectral Representations](#spectral-representations)
4. [Comparison Metrics](#comparison-metrics)
5. [Usage Examples](#usage-examples)

## Introduction

Frequency domain analysis is an essential part of signal processing. While the basic transforms like FFT and DFT provide the frequency content of a signal, spectral descriptors help in understanding the characteristics and patterns in the frequency domain.

The spectral utilities module offers functions for:
- Computing spectral shape descriptors (centroid, spread, skewness, kurtosis)
- Measuring spectral tonal characteristics (flatness, crest factor, contrast)
- Analyzing frequency content (dominant frequencies, bandwidth, slope, decrease)
- Analyzing spectral changes over time (flux)
- Representing spectral energy in different forms (ESD, normalized PSD)
- Finding spectral thresholds and cutoffs (rolloff)

These utilities are particularly useful for:
- Audio signal analysis and music information retrieval
- Speech processing and recognition
- Vibration analysis in mechanical systems
- Biomedical signal processing
- Feature extraction for machine learning on time-series data

## Spectral Descriptors

### Spectral Centroid

The spectral centroid represents the "center of mass" of the spectrum. It indicates the frequency around which most of the energy is concentrated.

```rust
let centroid = spectral_centroid(&psd, &freqs)?;
```

### Spectral Spread

The spectral spread measures the standard deviation of the spectrum around its centroid. It indicates how spread out the spectral energy is from the centroid.

```rust
let spread = spectral_spread(&psd, &freqs, None)?;
```

The optional third parameter allows you to provide a pre-calculated centroid to avoid redundant computation.

### Spectral Skewness

Spectral skewness measures the asymmetry of the spectral distribution. Positive skewness indicates more energy in frequencies above the centroid, while negative skewness indicates more energy below the centroid.

```rust
let skewness = spectral_skewness(&psd, &freqs, None, None)?;
```

### Spectral Kurtosis

Spectral kurtosis quantifies the "tailedness" or "peakedness" of the spectral distribution. It helps identify whether the energy is concentrated around a few frequencies or distributed more evenly.

```rust
let kurtosis = spectral_kurtosis(&psd, &freqs, None, None)?;
```

### Spectral Crest Factor

The spectral crest factor is the ratio of the maximum value to the arithmetic mean of the power spectrum. Higher values indicate more tone-like sounds (a clear peak), while lower values indicate more noise-like sounds (a flatter spectrum).

```rust
let crest = spectral_crest(&psd)?;
```

### Spectral Decrease

The spectral decrease measures the amount of decreasing in the spectral amplitude with frequency. It's often used in audio analysis for timbral characterization.

```rust
let decrease = spectral_decrease(&psd, &freqs)?;
```

### Spectral Slope

The spectral slope measures how quickly the spectrum falls off with frequency. It's calculated as the linear regression slope of the spectrum magnitude.

```rust
let slope = spectral_slope(&psd, &freqs)?;
```

### Spectral Contrast

Spectral contrast measures the difference between peaks and valleys in the spectrum. It's computed for multiple sub-bands and helps characterize the distribution of spectral peaks.

```rust
let contrast = spectral_contrast(&psd, &freqs, 4)?; // 4 bands
```

### Spectral Bandwidth

The spectral bandwidth is the width of the frequency band where the spectral magnitudes are above a specific threshold relative to the peak magnitude.

```rust
let bandwidth = spectral_bandwidth(&psd, &freqs, -3.0)?; // -3dB threshold
```

### Dominant Frequency

Find the dominant frequency (with highest magnitude) in a spectrum.

```rust
let (dominant_freq, magnitude) = dominant_frequency(&psd, &freqs)?;
```

### Multiple Dominant Frequencies

Find multiple dominant frequencies (local maxima) in a spectrum with minimum separation.

```rust
let peaks = dominant_frequencies(&psd, &freqs, 3, 10.0)?; // Top 3 peaks with min separation of 10 Hz
```

## Spectral Representations

### Energy Spectral Density (ESD)

The energy spectral density describes how the energy of a signal is distributed across frequency components, scaled by the time interval.

```rust
let esd = energy_spectral_density(&psd, fs)?;
```

### Normalized Power Spectral Density

This representation normalizes the PSD to have unit area, making it easier to compare spectra with different energy levels.

```rust
let normalized_psd = normalized_psd(&psd)?;
```

## Comparison Metrics

### Spectral Flatness

Spectral flatness measures how noise-like versus tone-like a signal is. Values close to 1 indicate a noise-like signal, while values close to 0 indicate a more tonal signal.

```rust
let flatness = spectral_flatness(&psd)?;
```

### Spectral Flux

Spectral flux measures the rate of change of the spectral content between consecutive frames. It is useful for detecting transients and changes in a signal.

```rust
let flux = spectral_flux(&psd1, &psd2, "l2")?;
```

The third parameter specifies the norm to use for comparison: "l1" (Manhattan), "l2" (Euclidean), or "max" (Chebyshev).

### Spectral Rolloff

Spectral rolloff finds the frequency below which a specified percentage of the total spectral energy is contained.

```rust
let rolloff = spectral_rolloff(&psd, &freqs, 0.85)?;
```

## Usage Examples

### Basic Spectral Analysis

```rust
use scirs2_signal::spectral::periodogram;
use scirs2_signal::utilities::spectral::*;

// Compute power spectral density
let (psd, freqs) = periodogram(&signal, Some(fs), Some("hann"), Some(n_samples), None, None)?;

// Calculate spectral descriptors
let centroid = spectral_centroid(&psd, &freqs)?;
let spread = spectral_spread(&psd, &freqs, None)?;
let flatness = spectral_flatness(&psd)?;
let rolloff_85 = spectral_rolloff(&psd, &freqs, 0.85)?;

println!("Centroid: {:.2} Hz", centroid);
println!("Spread: {:.2} Hz", spread);
println!("Flatness: {:.4}", flatness);
println!("85% Rolloff: {:.2} Hz", rolloff_85);
```

### Comparing Spectral Changes

```rust
// Compute spectral flux between two frames
let flux_l1 = spectral_flux(&psd1, &psd2, "l1")?;
let flux_l2 = spectral_flux(&psd1, &psd2, "l2")?;
let flux_max = spectral_flux(&psd1, &psd2, "max")?;

// Higher flux values indicate more change between frames
```

### Analyzing Different Signals

Spectral descriptors can help categorize different types of signals:

- **Tonal signals** (like musical notes) typically have:
  - Low spectral flatness
  - High kurtosis
  - Centroids near the fundamental frequency

- **Noisy signals** typically have:
  - High spectral flatness
  - Low kurtosis
  - More uniform spectral spread

- **Transient signals** (like percussion) typically have:
  - Rapidly changing spectral flux
  - Wider spectral spread
  - Higher rolloff frequencies

For more detailed examples, see the `spectral_descriptors.rs` example in the repository.