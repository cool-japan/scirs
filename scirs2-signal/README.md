# scirs2-signal

[![crates.io](https://img.shields.io/crates/v/scirs2-signal.svg)](https://crates.io/crates/scirs2-signal)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](../LICENSE)
[![Documentation](https://img.shields.io/docsrs/scirs2-signal)](https://docs.rs/scirs2-signal)
[![Status](https://img.shields.io/badge/status-stable-brightgreen)]()
[![Version](https://img.shields.io/badge/version-0.6.6-green)]()

**Production-ready signal processing for Rust** — part of the [SciRS2](https://github.com/cool-japan/scirs) scientific computing ecosystem.

`scirs2-signal` provides a comprehensive signal processing toolkit modelled after SciPy's `signal` module, going considerably further: matched filtering, CFAR detection, a full Kalman filter family (standard/EKF/UKF/ensemble/information/particle), MFCC and cepstral analysis, EMD/HHT, compressed sensing (OMP/basis pursuit/LASSO/CoSaMP), blind source separation (ICA/BSS, NMF audio), and system identification (ARX, N4SID). The Savitzky-Golay polynomial smoother (`savgol`) was re-enabled and validated in the v0.5.0 stub-check sweep. Note: the Kalman, compressed-sensing, and BSS modules, plus a class-based `ShortTimeFFT`-style STFT port, existed as source since earlier releases but had no `mod` declaration anywhere in `lib.rs` — they were only wired into the compiled crate as of v0.6.5 (see below); matched filtering/CFAR (`radar`) was in the same position.

Tested: 1489/1489 tests passing with default features, 1489/1489 with `--all-features` (verified 2026-07-15, predates the v0.6.5 file wiring/deletion below — the exact count has shifted and has not been re-measured for this docs pass).

**v0.6.3:** fixed two correctness bugs — `eigenvalues_francis_qr` (`src/phase_estimation/esprit.rs`, used by ESPRIT phase estimation) had a wrong bulge-term index and a right-multiply loop one row short in its double-shift QR, letting eigenvalues silently drift from the input matrix's; and `n4sid_estimate` (`src/system_identification/n4sid.rs`) now solves its least-squares steps through an SVD-based minimum-norm solve (new `pseudoinverse_product`) instead of normal equations, which previously returned silently-bogus huge-norm solutions for rank-deficient (non-persistently-exciting) inputs instead of erroring.

**v0.6.4:** `oxifft`'s `threading` (rayon) feature is no longer enabled by default on `wasm32-unknown-unknown` — `oxifft` is now declared via target-gated `[target.'cfg(...)'.dependencies]` tables, resolving a hard `compile_error!` that `oxifft` 0.4.1 added for that combination.

**v0.6.5:** wired 83 previously-orphaned-but-real files back into the crate (see `wire_files.txt` at the crate root) — a full Kalman filter family (standard/extended/unscented/ensemble/information/particle, `src/kalman/`), a BSS/ICA toolkit (`src/bss/`), compressed-sensing sparse recovery (OMP/basis-pursuit/LASSO/CoSaMP, `src/compressed_sensing/`), time-series feature extraction (`src/features/`), and a SciPy-`ShortTimeFFT`-class STFT port (`src/stft/`) — none of which had any wired equivalent anywhere in the crate before this release. Also deleted ~167 unreachable legacy/duplicate files (541 total `.rs` files in `src/`, 255 unreachable — 47%, not the ~82 previously estimated): six mutually-redundant Lomb-Scargle validation suites, five competing WPT validation suites, four independent copies of the sparse-recovery algorithm family, three redundant synchrosqueezed-transform implementations, and several half-finished `splitrs`-style refactors never wired into `lib.rs`. **One casualty of the cleanup**: `mir.rs` (the crate's only Music Information Retrieval implementation — beat tracking, tempo estimation, key detection, tonal centroid, self-similarity structural segmentation) was deleted with no surviving equivalent wired anywhere. CQT-based chroma (`cqt::chromagram`) and spectral-flux onset detection (`streaming::spectral_analysis::SpectralFlux`) remain available and unaffected, but the rest of the former MIR feature set is not currently implemented (see the MIR section below and `TODO.md`).

---

## Overview

Signal processing tasks range from basic filtering and spectral analysis through advanced topics such as sparse recovery, time-frequency representations, source separation, and system identification. `scirs2-signal` covers the full spectrum in a unified, type-safe Rust API with no C or Fortran dependencies.

---

## Feature List (v0.6.6)

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
- **Kalman filter** (`kalman::KalmanFilter`) and Rauch-Tung-Striebel (RTS) smoother (`kalman::legacy`)
- **Extended Kalman Filter (EKF)**: linearisation via Jacobians, analytical or numerical (`kalman::extended`)
- **Unscented Kalman Filter (UKF)**: sigma-point propagation (`kalman::unscented`)
- **Ensemble Kalman Filter (EnKF)**: Monte Carlo ensemble propagation for high-dimensional state (`kalman::ensemble`)
- **Information Filter**: dual/information-form linear Kalman filter (`kalman::information`)
- **Particle filter**: sequential Monte Carlo for nonlinear/non-Gaussian systems, with Gaussian and Student-t likelihoods and configurable resampling (`kalman::particle`)

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
- Chroma features: CQT-based chroma (`cqt::chromagram`)
- Onset detection: spectral flux (`streaming::spectral_analysis::SpectralFlux`)
- **Not currently implemented** (the crate's only implementation, `mir.rs`, was removed with no
  replacement wired during v0.6.5's dead-code cleanup — see "What's New" above): beat tracking and
  tempo estimation, PCP chroma, HFC/complex-domain onset detection, tonal centroid (Harmonic
  Network), key detection, and structural segmentation via self-similarity matrices

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
scirs2-signal = "0.6.6"
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
use scirs2_signal::cepstral::mfcc;

// 1 second of 16 kHz audio
let audio: Vec<f64> = vec![0.0_f64; 16000];

let features = mfcc(&audio, 16000.0, 13, Some(512), Some(40), None).unwrap();
// features: shape [n_frames x 13]
println!("MFCC frames: {}", features.nrows());
```

### OMP Sparse Recovery

```rust
use scirs2_signal::compressed_sensing::{omp, OmpConfig};

// y = Phi * x_sparse + noise (`phi`: Array2<f64> sensing matrix, `y`: Array1<f64> measurements)
let cfg = OmpConfig { sparsity: 10, ..Default::default() };
let x_recovered = omp(&phi, &y, &cfg).unwrap();
let nnz = x_recovered.iter().filter(|v| v.abs() > 1e-8).count();
println!("Recovered {nnz} non-zero coefficients");
```

### Kalman Filter Tracking

```rust
use scirs2_signal::kalman::KalmanFilter;

// Constant-velocity 1D model: state = [position, velocity]
let mut kf = KalmanFilter::new(2, 1); // (state_dim, obs_dim)
kf.set_F(vec![vec![1.0, 1.0], vec![0.0, 1.0]]).unwrap();
kf.set_H(vec![vec![1.0, 0.0]]).unwrap();
kf.set_Q(vec![vec![0.01, 0.0], vec![0.0, 0.01]]).unwrap();
kf.set_R(vec![vec![0.1]]).unwrap();
kf.set_initial_state(&[0.0, 1.0]).unwrap();

for obs in &measurements {
    kf.predict().unwrap();
    kf.update(&[*obs]).unwrap();
    println!("Position estimate: {:.3}", kf.state()[0]);
}
```

### Matched Filter Detection

```rust
use scirs2_signal::radar::{matched_filter, cfar_detector, CfarConfig, CfarVariant};

let detected = matched_filter(&received, &template).unwrap();

// CA-CFAR detection: 16 reference cells, 4 guard cells, Pfa = 1e-4
let power: Vec<f64> = detected.iter().map(|v| v * v).collect();
let cfg = CfarConfig::new(16, 4, 1e-4, CfarVariant::CellAveraging).unwrap();
let detections = cfar_detector(&power, &cfg).unwrap();
println!("Detected {} targets", detections.len());
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
| `time_frequency` | Wigner-Ville/Pseudo-WVD, Cohen's class (Choi-Williams, Born-Jordan), reassigned spectrogram — the standalone `wvd`/`reassigned`/`reassignment` modules were deleted as duplicates in v0.6.5; this already-wired module is the surviving implementation |
| `synchrosqueezing` | SST, ridge extraction, inverse SST |
| `czt` | Chirp-Z transform, Zoom FFT, Goertzel, Sliding DFT — merged in from the former `zoom_fft` module (renamed in v0.6.5) |
| `wavelet` | DWT, CWT, SWT, DTCWT, packets, wavelet denoising |
| `wavelet_denoise` | VisuShrink, BayesShrink, SUREshrink |
| `cepstral` | Complex/real cepstrum, MFCC, mel filterbank, pitch estimation, inverse cepstrum, liftering — merged in from the former `cepstrum` module (renamed in v0.6.5) |
| `kalman` | Standard/EKF/UKF/ensemble/information/particle Kalman filters, RTS smoother — newly wired in v0.6.5, previously present as unreachable source only |
| `adaptive` | LMS, NLMS, RLS, lattice RLS (module renamed from `adaptive_filter`) |
| `compressed_sensing` | OMP, CoSaMP, ISTA/FISTA, basis pursuit, LASSO, measurement matrices — newly wired in v0.6.5; the duplicate `compressive_sensing`/`sparse_recovery`/`source_separation` modules were deleted |
| `bss` | FastICA, JADE, SOBI, convolutive BSS, NMF audio — newly wired in v0.6.5, previously present as unreachable source only |
| `sysid_enhanced` | ARX, ARMAX, N4SID, ERA, validation |
| `emd` | EMD, EEMD, Hilbert-Huang spectrum (`hilbert_huang_spectrum`) |
| `multiscale` | HHT (wraps `emd`'s EMD/EEMD with the Hilbert transform) and Variational Mode Decomposition (VMD) |
| `multitaper_mod` | Multitaper PSD and coherence |
| `radar` | Matched filter, CA/OS/GO/SO-CFAR, pulse compression, ambiguity function — newly wired in v0.6.5, previously present as unreachable source only |
| `cqt` | Constant-Q transform; CQT-based chromagram (`chromagram`) — the only MIR-adjacent feature still wired after v0.6.5's `mir.rs` cleanup |
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
| `parallel` | Rayon parallel computation (`scirs2-core/parallel`) — in `default` |
| `unstable_avx512` | Experimental AVX-512 support |

Corrected 2026-07-15: `default = ["parallel"]` (not "none" as previously stated). SIMD acceleration (via `scirs2-core`'s `simd` feature) and Serde support are unconditional dependencies of this crate — always compiled in, not optional feature flags that can be toggled off. Pure Rust throughout, no C/Fortran dependencies.

---

## Links

- [SciRS2 project](https://github.com/cool-japan/scirs)
- [docs.rs](https://docs.rs/scirs2-signal)
- [crates.io](https://crates.io/crates/scirs2-signal)
- [TODO.md](./TODO.md)

## License

Apache License 2.0. See [LICENSE](../LICENSE) for details.
