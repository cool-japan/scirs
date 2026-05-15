# scirs2-signal

[![crates.io](https://img.shields.io/crates/v/scirs2-signal.svg)](https://crates.io/crates/scirs2-signal)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](../LICENSE)
[![Documentation](https://img.shields.io/docsrs/scirs2-signal)](https://docs.rs/scirs2-signal)
[![Version](https://img.shields.io/badge/version-0.4.4-green)]()

**Production-ready signal processing for Rust** — part of the [SciRS2](https://github.com/cool-japan/scirs) scientific computing ecosystem.

`scirs2-signal` provides a comprehensive signal processing toolkit modelled after SciPy's `signal` module while going considerably further in v0.4.4: matched filtering, CFAR detection, Kalman/EKF/UKF state estimation, MFCC and cepstral analysis, EMD/HHT, compressed sensing (OMP/ISTA), blind source separation (ICA, NMF audio), system identification (ARX, N4SID), radar processing, and music information retrieval. The Savitzky-Golay polynomial smoother (`savgol`) was re-enabled and validated in the v0.4.4 stub-check sweep.

---

## Overview

Signal processing tasks range from basic filtering and spectral analysis through advanced topics such as sparse recovery, time-frequency representations, source separation, and system identification. `scirs2-signal` covers the full spectrum in a unified, type-safe Rust API with no C or Fortran dependencies.

---

## Feature List (v0.4.4)

### Filter Design & Application
- **IIR filters**: Butterworth, Chebyshev I/II, Elliptic, Bessel — analog prototype design and digital transformation
- **FIR filters**: window method (Hamming, Hanning, Blackman, Kaiser, flat-top), Parks-McClellan / Remez exchange algorithm
- **Zero-phase filtering**: `filtfilt` (forward-backward) with edge-padding
- **Specialized filters**: notch, comb, allpass, peaking EQ, shelving EQ
- **Savitzky-Golay filter**: polynomial smoothing with arbitrary derivative order
- **Filter analysis**: frequency response (`freqz`, `freqs`), group delay, stability (pole-zero analysis), impulse and step response
- **Filter transformations**: lowpass-to-bandpass, lowpass-to-highstop, analog-to-digital (bilinear transform, impulse invariance, matched-z)
- **Second-order sections (SOS)**: numerically stable cascaded biquad representation

### Convolution & Correlation
- 1-D convolution with `full`, `same`, `valid` modes and `direct`, `fft`, `auto` methods
- Cross-correlation and autocorrelation
- Basic deconvolution (Wiener)
- FFT-based fast convolution via OxiFFT

### Spectral Analysis
- Periodogram (rectangular window)
- Welch's method for power spectral density (PSD) estimation
- Bartlett's method
- Short-time Fourier transform (STFT) and inverse STFT
- Spectrogram with configurable window, overlap, and FFT size
- Lomb-Scargle periodogram for unevenly sampled data
- **Multitaper spectral estimation** (DPSS / Slepian sequences): minimises spectral leakage; adaptive weighting
- **Parametric spectral estimation**:
  - AR model (Yule-Walker, Burg, Covariance, Modified Covariance)
  - ARMA model spectral estimation
  - MUSIC (MUltiple SIgnal Classification) for superresolution frequency estimation
  - ESPRIT (Estimation of Signal Parameters via Rotational Invariance Techniques)
- Coherence and cross-power spectral density
- Signal detrending (constant, linear, polynomial)

### Time-Frequency Representations
- **Synchrosqueezing transform (SST)**: time-frequency reassignment for sharp IF ridges; ridge extraction
- **Reassigned spectrogram**: locally improved time-frequency localisation via phase derivatives
- **Wigner-Ville distribution (WVD)** and Pseudo-WVD (PWVD)
- Cohen's class of time-frequency distributions (Choi-Williams, Born-Jordan)
- Zoom FFT (chirp-z transform) for high-resolution analysis in a sub-band
- Hilbert transform and analytic signal, instantaneous frequency and amplitude

### Wavelet Transforms
- Discrete Wavelet Transform (DWT): Haar, Daubechies (2–20), Symlets, Coiflets, Biorthogonal
- Continuous Wavelet Transform (CWT): Morlet, Paul, DOG, Mexican Hat
- Stationary / undecimated DWT (SWT) for shift-invariant decomposition
- Dual-tree complex wavelet transform (DTCWT)
- Wavelet packets (full binary tree decomposition)
- **Wavelet denoising**: VisuShrink, BayesShrink, SUREshrink threshold selection; hard and soft thresholding

### Empirical Mode Decomposition & HHT
- **EMD (Empirical Mode Decomposition)**: intrinsic mode function (IMF) extraction via sifting algorithm; stopping criterion (Cauchy, fixed iterations, S-number)
- **EEMD** (Ensemble EMD) for mode mixing reduction
- **CEEMDAN** (Complete EEMD with Adaptive Noise)
- **Hilbert-Huang Transform (HHT)**: instantaneous frequency and amplitude of each IMF
- **HHT spectrum** (Hilbert spectrum) for time-frequency-energy representation

### Adaptive Filters
- **LMS (Least Mean Squares)**: normalized LMS (NLMS), leaky LMS, sign-error LMS
- **RLS (Recursive Least Squares)**: standard RLS, QR-decomposition RLS (lattice form)
- **Kalman adaptive filter**: state-space formulation for tracking non-stationary signals
- Applications: echo cancellation, noise cancellation, channel equalization, system identification

### State Estimation (Kalman Family)
- **Kalman filter** and Rauch-Tung-Striebel (RTS) smoother
- **Extended Kalman Filter (EKF)**: linearisation via Jacobians (analytical or numerical)
- **Unscented Kalman Filter (UKF)**: sigma-point propagation (Van der Merwe parametrisation)
- Square-root formulations of EKF and UKF for numerical stability

### Compressed Sensing & Sparse Recovery
- **OMP (Orthogonal Matching Pursuit)**: greedy sparse recovery with sparsity or residual stopping
- **Basis Pursuit / LASSO**: L1-minimisation via ADMM and ISTA/FISTA
- **ISTA / FISTA** (Iterative Soft Thresholding): convergence-guaranteed sparse recovery
- **CoSaMP** (Compressive Sampling Matching Pursuit)
- Measurement matrix construction: Gaussian, Bernoulli, subsampled DFT
- Signal recovery from compressive measurements with noise

### Independent Component Analysis (ICA) & Blind Source Separation (BSS)
- **FastICA**: fixed-point algorithm with logcosh and kurtosis contrast functions
- **JADE (Joint Approximate Diagonalisation of Eigenmatrices)**: fourth-order cumulant-based ICA
- **SOBI (Second Order Blind Identification)**: based on non-stationarity and temporal structure
- **Convolutive BSS**: frequency-domain approach for reverberant mixtures
- **NMF audio source separation**: non-negative matrix factorisation with Itakura-Saito divergence (for magnitude spectrograms)

### Cepstral Analysis & MFCCs
- Complex and real cepstrum computation and inverse cepstrum
- Liftering (quefrency-domain windowing)
- **MFCC (Mel-Frequency Cepstral Coefficients)**:
  - Mel filterbank design (HTK and Slaney parametrisations)
  - Log mel spectrogram
  - DCT for coefficient extraction
  - Delta and delta-delta (velocity and acceleration) coefficients
- Pitch (F0) estimation: autocorrelation, AMDF, YIN algorithm
- Spectral flatness, spectral roll-off, spectral centroid

### System Identification
- **ARX** (Autoregressive with Exogenous input): least-squares estimation, order selection
- **ARMAX**: iterative least-squares for MA noise modelling
- **N4SID** (Numerical Algorithms for Subspace State Space System Identification): subspace-based state-space identification
- **Eigensystem Realisation Algorithm (ERA)**: impulse-response-based realisation
- Transfer function and state-space model estimation
- Validation: residual analysis, one-step-ahead prediction, cross-validation

### Matched Filter & Detection
- **Matched filter**: correlate received signal with known template; SNR-optimal detection
- **CFAR (Constant False Alarm Rate)** detector:
  - Cell-Averaging CFAR (CA-CFAR)
  - Order Statistics CFAR (OS-CFAR)
  - Greatest Of / Smallest Of CFAR (GO/SO-CFAR)
- **Pulse compression**: linear frequency modulation (LFM/chirp), polyphase codes (Frank, P4)
- Radar range-Doppler processing (2D FFT with Doppler windowing)

### Resampling
- Upsampling, downsampling, and arbitrary rational resampling
- Polyphase filterbank-based efficient resampling
- Anti-aliasing filter design for downsampling
- Asynchronous sample rate conversion (ASRC)

### Waveform Generation
- Sine, cosine, square (duty-cycle configurable), sawtooth, triangle waveforms
- Chirp (linear, quadratic, logarithmic, hyperbolic frequency sweep)
- Gaussian pulse and Gaussian modulated sinusoid
- Unit impulse, step, ramp
- Noise: white, pink (1/f), brown (1/f²)

### Linear System Analysis
- Transfer function and state-space representation
- Frequency response (`bode`, `freqz`), pole-zero maps, root locus
- Step response, impulse response, initial condition response
- Stability analysis: Routh-Hurwitz, Nyquist criterion, gain/phase margins
- System interconnection: series, parallel, feedback loops
- Continuous-to-discrete conversion (ZOH, Tustin/bilinear, matched pole-zero)

### Peak Detection & Signal Measurements
- Peak finding with prominence, width, height, and distance constraints
- Peak properties: FWHM, area, asymmetry
- RMS, peak, peak-to-peak, crest factor, PAR
- SNR (signal-to-noise ratio), THD (total harmonic distortion), SFDR (spurious-free dynamic range)
- EVM (error vector magnitude)

### Music Information Retrieval (MIR)
- Beat tracking and tempo estimation (onset-strength-based)
- Chroma features (PCP, CQT-based chroma)
- Tonal centroid (Harmonic Network) and key detection
- Onset detection (spectral flux, HFC, complex domain)
- Structural segmentation via self-similarity matrices

### Radar Signal Processing
- Linear and non-linear frequency-modulated chirp waveforms
- Pulse Doppler processing: coherent integration, range-Doppler maps
- CFAR detection in range-Doppler domain
- Sidelobe suppression (weighting windows in range and Doppler)
- Ambiguity function computation

### Super-Advanced Denoising
- Deep-learning-inspired shrinkage functions (learnable threshold parameters)
- Empirical Wiener filter from multiple signal estimates
- Non-local means denoising adapted for 1-D signals

---

## Quick Start

```toml
[dependencies]
scirs2-signal = "0.4.4"
```

### Butterworth Low-Pass Filter

```rust
use scirs2_signal::filter::{butter, lfilter, filtfilt};
use scirs2_core::ndarray::Array1;
use std::f64::consts::PI;

let fs = 1000.0_f64;
let n_samples = 1000_usize;
let t = Array1::linspace(0.0, 1.0, n_samples);

// 5 Hz + 150 Hz mixed signal
let signal = t.mapv(|x| (2.0 * PI * 5.0 * x).sin() + 0.3 * (2.0 * PI * 150.0 * x).sin());

// Design 4th-order Butterworth low-pass at 20 Hz
let (b, a) = butter(4, &[20.0 / (fs / 2.0)], "low").unwrap();

// Zero-phase filtering
let filtered = filtfilt(&b, &a, &signal.view()).unwrap();
println!("Filtered {} samples", filtered.len());
```

### STFT and Spectrogram

```rust
use scirs2_signal::spectral::{stft, spectrogram};
use scirs2_core::ndarray::Array1;

let fs = 8000.0_f64;
let signal: Array1<f64> = /* ... your audio signal ... */ Array1::zeros(8000);

let (freqs, times, stft_matrix) = stft(&signal.view(), fs, 256, 128, "hann").unwrap();
println!("STFT shape: {} freqs x {} frames", freqs.len(), times.len());

let spec = spectrogram(&signal.view(), fs, 512, 256, "hamming").unwrap();
```

### MFCC Extraction

```rust
use scirs2_signal::cepstrum::mfcc;

// 1 second of 16 kHz audio
let audio: Vec<f64> = vec![0.0_f64; 16000];

let features = mfcc(&audio, 16000.0, 13, Some(512), Some(40), None).unwrap();
// features: shape [n_frames x 13]
println!("MFCC frames: {}", features.nrows());
```

### OMP Sparse Recovery

```rust
use scirs2_signal::compressed_sensing::omp;
use scirs2_core::ndarray::{Array1, Array2};

// y = Phi * x_sparse + noise
let (x_recovered, support) = omp(&phi.view(), &y.view(), 10, None).unwrap();
println!("Recovered {} non-zero coefficients", support.len());
```

### Kalman Filter Tracking

```rust
use scirs2_signal::kalman::{KalmanFilter, KalmanConfig};
use scirs2_core::ndarray::{array, Array1, Array2};

// Constant-velocity 1D model
let config = KalmanConfig {
    state_dim: 2,
    obs_dim: 1,
    transition: array![[1.0, 1.0], [0.0, 1.0]],
    observation: array![[1.0, 0.0]],
    process_noise: Array2::eye(2) * 0.01,
    observation_noise: Array2::eye(1) * 1.0,
    initial_state: Array1::zeros(2),
    initial_covariance: Array2::eye(2),
};

let mut kf = KalmanFilter::new(config);

for obs in &measurements {
    let (state, cov) = kf.update(&array![*obs]).unwrap();
    println!("Position estimate: {:.3}", state[0]);
}
```

### Matched Filter Detection

```rust
use scirs2_signal::radar::{matched_filter, ca_cfar};

let detected = matched_filter(&received.view(), &template.view()).unwrap();

// CA-CFAR detection
let detections = ca_cfar(&detected.view(), 16, 4, 1e-4).unwrap();
println!("Detected {} targets", detections.iter().filter(|&&d| d).count());
```

### Granger / AR Spectral Estimation

```rust
use scirs2_signal::parametric_spectral::{burg_ar, ar_spectrum};

let signal: Vec<f64> = /* ... */ vec![0.0_f64; 1024];

// Fit AR(16) via Burg's method
let (ar_coeffs, variance) = burg_ar(&signal, 16).unwrap();

// Evaluate spectrum at 1024 frequency bins
let (freqs, psd) = ar_spectrum(&ar_coeffs, variance, 1.0, 1024).unwrap();
```

---

## API Overview

| Module | Description |
|---|---|
| `filter` | IIR/FIR design, filtfilt, SOS, notch, comb, Savitzky-Golay |
| `filter::iir` | Butterworth, Chebyshev, Elliptic, Bessel prototypes |
| `filter::application` | `lfilter`, `filtfilt`, `sosfilt`, `sosfiltfilt` |
| `spectral` | Periodogram, Welch, STFT, spectrogram, Lomb-Scargle |
| `spectral_estimation` | Multitaper (DPSS), parametric AR/ARMA, MUSIC, ESPRIT |
| `parametric_spectral` | AR via Yule-Walker, Burg, covariance; ARMA |
| `reassigned` | Reassigned spectrogram, synchrosqueezing transform |
| `wigner_ville` | Wigner-Ville and Pseudo-WVD, Cohen's class |
| `synchrosqueezing` | SST, ridge extraction, inverse SST |
| `zoom_fft` | Chirp-z transform and zoom FFT sub-band analysis |
| `wavelet` | DWT, CWT, SWT, DTCWT, packets, wavelet denoising |
| `wavelet_denoise` | VisuShrink, BayesShrink, SUREshrink |
| `cepstrum` | Complex/real cepstrum, MFCC, mel filterbank, pitch estimation |
| `kalman` | KF, EKF, UKF, RTS smoother |
| `adaptive_filter` | LMS, NLMS, RLS, lattice RLS |
| `compressed_sensing` | OMP, CoSaMP, ISTA/FISTA, basis pursuit |
| `compressive_sensing` | Measurement matrix construction, recovery diagnostics |
| `sparse_recovery` | Unified sparse recovery interface |
| `bss` | FastICA, JADE, SOBI, convolutive BSS, NMF audio |
| `source_separation` | High-level source separation API |
| `sysid_enhanced` | ARX, ARMAX, N4SID, ERA, validation |
| `hht` | EMD, EEMD, CEEMDAN, HHT spectrum |
| `multitaper_mod` | Multitaper PSD and coherence |
| `radar` | Matched filter, CA/OS/GO/SO-CFAR, pulse compression, ambiguity function |
| `mir` | Beat tracking, chroma, onset detection, key detection |
| `multiscale` | Multiscale signal decomposition utilities |
| `lti` | Transfer function, state-space, Bode, root locus |
| `waveforms` | Sine, chirp, Gaussian pulse, noise waveforms |
| `peak` | Peak detection, prominence, width, FWHM |
| `convolve` | 1-D convolution and correlation with mode control |
| `resampling` | Upsampling, downsampling, polyphase rational resampling |
| `denoise_super_advanced` | Advanced multi-method denoising pipeline |
| `spectral_scipy_validation_v2` | SciPy-compatible spectral output validation |

---

## Feature Flags

| Flag | Description |
|---|---|
| `parallel` | Rayon parallel computation |
| `simd` | SIMD-accelerated operations via `scirs2-core` |
| `serde` | Serialization support |

Default features: none (pure Rust, no C/Fortran dependencies).

---

## Links

- [SciRS2 project](https://github.com/cool-japan/scirs)
- [docs.rs](https://docs.rs/scirs2-signal)
- [crates.io](https://crates.io/crates/scirs2-signal)
- [TODO.md](./TODO.md)

## License

Apache License 2.0. See [LICENSE](../LICENSE) for details.
