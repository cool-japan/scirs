# scirs2-signal TODO

## Status: v0.6.5 (released, 2026-07-31)

**0.6.5:** the biggest single-crate change in this release. Wired 83 previously-orphaned-but-real
files back into the crate (`wire_files.txt` at the crate root lists them, and `src/lib.rs` gained
the corresponding `pub mod` declarations) — a full Kalman filter family (standard/extended/
unscented/ensemble/information/particle, `src/kalman/`), a BSS/ICA toolkit (`src/bss/`),
compressed-sensing sparse recovery (OMP/basis-pursuit/LASSO/CoSaMP, `src/compressed_sensing/`),
time-series feature extraction (`src/features/`), a SciPy-`ShortTimeFFT`-class STFT port
(`src/stft/`), and matched-filter/CFAR radar processing (`src/radar.rs`) — **none of these had any
wired equivalent anywhere in the compiled crate before this release**, despite `README.md` and this
file having described several of them (Kalman, BSS, compressed sensing, matched filtering/CFAR) as
complete since as far back as "v0.3.3 Completed" below; those checkmarks were aspirational until
now. Also deleted ~167 unreachable legacy/duplicate files (541 total `.rs` files in `src/`, 255
unreachable — 47%, not the ~82 previously estimated): six mutually-redundant Lomb-Scargle validation
suites, five competing WPT validation suites, four independent copies of the sparse-recovery
algorithm family (`sparse.rs`, `sparse_recovery.rs`, `compressive_sensing/`, `source_separation/`),
three redundant synchrosqueezed-transform implementations (`synchrosqueeze.rs`, `sswt.rs`, plus the
surviving `synchrosqueezing.rs`), and several half-finished `splitrs`-style refactors never wired
into `lib.rs`. Renamed-and-merged in the same pass: `zoom_fft.rs` → `czt.rs` (now also hosts
Goertzel/Sliding-DFT), `cepstrum.rs` → merged into `cepstral.rs`, `resample.rs`/`multirate.rs` →
`resampling/`. **One real regression found while verifying this for docs**: `mir.rs` (the crate's
only Music Information Retrieval implementation — beat tracking, tempo estimation, key detection,
tonal centroid, self-similarity structural segmentation) was deleted with no replacement wired
anywhere; only CQT-based chroma (`cqt::chromagram`) and spectral-flux onset detection
(`streaming::spectral_analysis::SpectralFlux`) survive of the former MIR feature set — see the
correction in "Music Information Retrieval (MIR)" under "v0.3.3 Completed" below. `wvd.rs` and
`reassigned.rs`/`reassignment.rs` were also deleted but are **not** regressions: their Wigner-Ville/
Cohen's-class/reassigned-spectrogram functionality already had a genuine, already-wired equivalent
in `time_frequency.rs`. See root `CHANGELOG.md` `[0.6.5]` for full detail.

**0.6.4:** `oxifft`'s `threading` (rayon) feature is no longer enabled by default on
`wasm32-unknown-unknown`; no other signal-specific source changes. See `CHANGELOG.md` `[0.6.4]`.

**0.6.3:** fixed two correctness bugs in `eigenvalues_francis_qr` (`src/phase_estimation/esprit.rs`, used by ESPRIT phase estimation)'s double-shift QR that let eigenvalues silently drift from the input matrix's — a wrong bulge-term index and a right-multiply loop one row short. Also fixed `n4sid_estimate` (`src/system_identification/n4sid.rs`)'s least-squares solve, which used normal equations and silently returned bogus huge-norm solutions for rank-deficient (non-persistently-exciting) inputs instead of erroring — both solves now go through an SVD-based minimum-norm solve (new `pseudoinverse_product`). Verified by code review and the (unchanged) test suite below; not exercised under Windows CI. See `CHANGELOG.md` `[0.6.3]` for full detail.

scirs2-signal's own test suite (freshly re-run 2026-07-15, **predates the v0.6.5 wiring/deletion
above — the 83-file addition and 167-file deletion has since changed the underlying test
population and this count has not been re-measured**): 1489 tests pass, 2 skipped, 0 failed with
default features; 1489 tests pass, 2 skipped, 0 failed with `--all-features`. Also: as of this docs
pass, `src/` and `tests/` carry zero `#[ignore]`d tests (`grep -rn '#\[ignore'` returns nothing),
consistent with the workspace-wide ignore-legitimacy audit's disposition of this crate's share of
the 132 → 59 count. Fixed a real `waverec` bug (`dwt/multiscale.rs`): it was reconstructing to 2x the correct length for wavelet filters with more than 2 taps; the DWT round-trip test now asserts genuine perfect reconstruction instead of just checking non-empty output.

## Status: v0.4.3 Released (May 3, 2026)

34,275+ workspace tests pass (100% pass rate). All v0.4.3 features are complete and production-ready. The Savitzky-Golay filter module (`savgol`) was uncommented and validated in the Wave 5 stub-check (+5 new tests).

## Status: v0.3.4 Released (March 18, 2026)

19,685 workspace tests pass (100% pass rate). All v0.3.4 features are complete and production-ready.

---

## v0.3.3 Completed

### Core Filtering
- [x] IIR filter design: Butterworth, Chebyshev I/II, Elliptic, Bessel (analog prototypes + bilinear/impulse-invariance transformation)
- [x] FIR filter design: window method (Hamming, Hanning, Blackman, Kaiser, flat-top), Parks-McClellan / Remez exchange
- [x] Zero-phase filtering: `filtfilt` with edge-padding strategies
- [x] Specialized filters: notch, comb, allpass, peaking EQ, shelving EQ
- [x] Savitzky-Golay filter with arbitrary polynomial order and derivative
- [x] Second-order sections (SOS) cascaded representation for numerical stability
- [x] Filter analysis: `freqz`, `freqs`, group delay, pole-zero maps, stability check
- [x] Filter transformations: LP-to-BP, LP-to-BS, analog-to-digital (bilinear, impulse invariance, matched-z)

### Spectral Analysis
- [x] Periodogram and Bartlett's method
- [x] Welch's method for PSD estimation with overlapping segments
- [x] Short-time Fourier transform (STFT) and inverse STFT
- [x] Spectrogram with configurable window, overlap, FFT size
- [x] Lomb-Scargle periodogram for non-uniform sampling
- [x] Coherence and cross-power spectral density
- [x] Signal detrending (constant, linear, polynomial)

### Multitaper Spectral Estimation
- [x] DPSS (Slepian) window sequences for arbitrary bandwidth-time product
- [x] Adaptive multitaper PSD (eigenspectrum weighting by expected bias/variance trade-off)
- [x] Jackknife confidence intervals for multitaper estimates
- [x] Multitaper coherence estimation

### Parametric Spectral Estimation
- [x] AR model via Yule-Walker equations
- [x] AR model via Burg's method (recursive lattice, exact maximum entropy)
- [x] AR model via covariance and modified covariance methods
- [x] ARMA spectral estimation
- [x] MUSIC (MUltiple SIgnal Classification) pseudo-spectrum
- [x] ESPRIT for superresolution frequency estimation

### Time-Frequency Representations
- [x] Synchrosqueezing transform (SST) with phase-based reassignment
- [x] Ridge extraction from SST and reassigned spectrogram
- [x] Reassigned spectrogram (partial derivatives of phase)
- [x] Wigner-Ville distribution (WVD) and Pseudo-WVD
- [x] Cohen's class: Choi-Williams, Born-Jordan distributions
- [x] Zoom FFT (chirp-z transform) for high-resolution sub-band analysis
- [x] Hilbert transform, analytic signal, instantaneous frequency/amplitude

### Wavelet Transforms
- [x] DWT: Haar, Daubechies (db2-db20), Symlets (sym2-sym20), Coiflets (coif1-coif5), Biorthogonal
- [x] CWT: Morlet, Paul, DOG, Mexican Hat wavelets
- [x] Stationary / undecimated DWT (SWT)
- [x] Dual-tree complex wavelet transform (DTCWT)
- [x] Wavelet packets (full binary tree decomposition with best-basis selection)
- [x] Wavelet denoising: VisuShrink, BayesShrink, SUREshrink; hard and soft thresholding

### EMD / HHT
- [x] EMD: sifting algorithm with Cauchy and S-number stopping criteria; cubic spline envelopes
- [x] EEMD: ensemble EMD with configurable noise amplitude and ensemble size
- [x] CEEMDAN (Complete EEMD with Adaptive Noise)
- [x] HHT: Hilbert transform of each IMF for instantaneous frequency and amplitude
- [x] Hilbert spectrum (time-frequency-energy representation) and marginal spectrum

### Adaptive Filters
- [x] LMS: standard, normalized (NLMS), leaky LMS, sign-error LMS
- [x] RLS: standard RLS with exponential forgetting, QR-based RLS (lattice form)
- [x] Adaptive Kalman filter for time-varying gain

### State Estimation
- [x] Kalman filter with Rauch-Tung-Striebel (RTS) smoother
- [x] Extended Kalman Filter (EKF) with analytical and numerical Jacobians
- [x] Unscented Kalman Filter (UKF) with Van der Merwe sigma-point parametrisation
- [x] Square-root EKF and UKF for improved numerical stability

### Compressed Sensing & Sparse Recovery
- [x] OMP (Orthogonal Matching Pursuit): sparsity and residual tolerance stopping
- [x] CoSaMP (Compressive Sampling Matching Pursuit)
- [x] ISTA and FISTA (Iterative Soft Thresholding Algorithm): convergence-guaranteed L1 minimisation
- [x] Basis Pursuit via ADMM
- [x] Measurement matrix construction: Gaussian, Bernoulli, subsampled DFT
- [x] Recovery quality metrics: relative error, support recovery rate

### Blind Source Separation (BSS) & ICA
- [x] FastICA: fixed-point algorithm with logcosh and kurtosis contrast
- [x] JADE: fourth-order cumulant tensor diagonalisation
- [x] SOBI: second-order blind identification using temporal structure
- [x] Convolutive BSS: frequency-domain approach with permutation alignment
- [x] NMF audio source separation with Itakura-Saito divergence and beta divergence

### Cepstral Analysis & MFCCs
- [x] Complex cepstrum, real cepstrum, inverse cepstrum
- [x] Liftering (quefrency-domain smoothing)
- [x] MFCC: mel filterbank design (HTK and Slaney), log mel spectrogram, DCT-II, delta and delta-delta coefficients
- [x] Pitch (F0) estimation: autocorrelation, YIN algorithm
- [x] Spectral features: centroid, bandwidth, roll-off, flatness, contrast

### System Identification
- [x] ARX model: least-squares estimation, order selection via AIC/MDL
- [x] ARMAX model: iterative least-squares for MA noise component
- [x] N4SID: subspace-based state-space system identification (PI-MOESP, CVA)
- [x] ERA (Eigensystem Realisation Algorithm): Hankel-matrix-based impulse response realisation
- [x] Validation: one-step-ahead prediction, residual whiteness test, fit percentage

### Matched Filter & Radar Detection
- [x] Matched filter: template correlation with SNR-optimal detection
- [x] CA-CFAR (Cell-Averaging CFAR)
- [x] OS-CFAR (Order Statistics CFAR)
- [x] GO-CFAR / SO-CFAR (Greatest Of / Smallest Of)
- [x] Linear FM (LFM/chirp) pulse compression
- [x] Range-Doppler processing: 2D FFT with Doppler windowing
- [x] Ambiguity function computation for waveform analysis

### Music Information Retrieval (MIR)
**Regression found 2026-07-31**: this section's checkmarks describe the crate's former `mir.rs`,
which was deleted (as part of the ~167-file dead-code cleanup) in the same v0.6.5 pass that wired
in Kalman/BSS/compressed-sensing/STFT — but unlike those, `mir.rs` had no replacement wired in
anywhere. Corrected below; see "Status: v0.6.5" above for the full story.
- [x] Chroma features: CQT-based chroma only, via `cqt::chromagram` (the short-time-Fourier/PCP
      chroma variant was only in the now-deleted `mir.rs`)
- [x] Onset detection: spectral flux only, via `streaming::spectral_analysis::SpectralFlux` (the
      high-frequency-content and complex-domain variants were only in the now-deleted `mir.rs`)
- [ ] ~~Beat tracking and tempo estimation via onset strength envelope~~ — not currently
      implemented; no surviving equivalent after `mir.rs`'s deletion
- [ ] ~~Key detection via chroma profiles~~ — not currently implemented; no surviving equivalent
- [ ] ~~Tonal centroid (Harmonic Network features)~~ — not currently implemented; no surviving equivalent
- [ ] ~~Structural segmentation via self-similarity matrices~~ — not currently implemented; no surviving equivalent

### Resampling
- [x] Upsampling and downsampling with anti-aliasing filters
- [x] Arbitrary rational resampling (polyphase filterbank)
- [x] Polyphase decomposition for efficient multi-rate processing

### Waveform Generation
- [x] Sine, cosine, square (configurable duty cycle), sawtooth, triangle
- [x] Chirp: linear, quadratic, logarithmic, hyperbolic FM sweep
- [x] Gaussian pulse and Gaussian-modulated sinusoid
- [x] Unit impulse, step, ramp
- [x] Noise: white Gaussian, pink (1/f), brown/red (1/f²)

### Linear System Analysis
- [x] Transfer function and state-space representations (continuous and discrete)
- [x] Bode plot (magnitude and phase), Nyquist diagram, root locus
- [x] Step, impulse, and initial condition responses
- [x] Stability: Routh-Hurwitz (continuous), Jury (discrete), Lyapunov
- [x] Gain margin, phase margin, delay margin
- [x] System interconnection: series, parallel, feedback
- [x] Continuous-to-discrete: ZOH, Tustin / bilinear, matched pole-zero

### Peak Detection & Signal Measurements
- [x] Peak finding with distance, prominence, width, height thresholds
- [x] Peak width at fractional height (FWHM), peak area, peak asymmetry
- [x] RMS, peak, peak-to-peak, crest factor, PAR
- [x] SNR, THD (with harmonic order), SFDR

### Super-Advanced Denoising
- [x] Empirical Wiener filter via multi-estimate combination
- [x] Learnable soft-thresholding with data-driven threshold selection
- [x] Non-local means 1-D denoising

---

## v0.4.0 Roadmap

### Real-Time Streaming Processing
- [x] Block-based filter processing with state preservation between blocks — Implemented in v0.4.0 (`streaming/block_filter.rs`)
- [x] Ring-buffer abstraction for streaming convolution and correlation — Implemented in v0.4.0 (`streaming/ring_buffer.rs`)
- [x] Online STFT with overlap-save/overlap-add block updating — Implemented in v0.4.0 (`streaming/online_stft.rs`)
- [x] Streaming OMP for adaptive sparse coding — Implemented in v0.4.0 (`streaming/streaming_omp.rs`)

### GPU-Accelerated FFT Pipeline
- [x] OxiFFT GPU backend integration for large-batch spectrograms — implemented in v0.4.2 (`gpu_spectrograms.rs`)
- [x] GPU-accelerated matched filter bank (multiple templates simultaneously) — implemented in v0.4.2 (`gpu_matched_filter.rs`)
- [x] Batched Welch PSD for parallel channel processing — Implemented in v0.4.2 (`welch_batch.rs`)
- [x] GPU wavelet transform for high-throughput applications — Implemented in v0.4.3 (`gpu_wavelet.rs`: `GpuWaveletConfig`, `GpuWaveletFamily`, `GpuWaveletBackend`, `dwt_dispatch`, `dwt_dispatch_batch`; wraps `gpu/fast_wavelet.rs` with Auto/Cpu/WebGpu backend selection)

Status (verified 2026-07-15): `gpu_spectrograms.rs` and `gpu_matched_filter.rs` provide GPU-ready APIs backed by "a correct CPU reference implementation that can be swapped for a GPU kernel without changing calling code" (their own doc comments); `gpu_wavelet.rs`'s `wgpu` code-path "is reserved for future WGSL compute shader integration" and currently returns `GpuWaveletError::GpuNotAvailable`, auto-falling back to the rayon-parallel CPU path. All three are real, tested, useful implementations, but no actual GPU dispatch occurs yet on any backend — checked off for the API/CPU-path completeness, not for GPU hardware execution. `welch_batch.rs` (third bullet) is unaffected by this caveat — it is a genuine CPU-parallel multi-channel Welch/CPSD estimator with no GPU claim of its own.

### Deep Learning-Based Denoising
- [x] Learned speech enhancement model (Conv-TasNet architecture) in pure Rust — Implemented in v0.4.0 (`neural_audio/conv_tasnet.rs`)
- [x] Deep filtering via scirs2-neural integration — Implemented in v0.4.3 (`deep_filter.rs`)
- [x] Denoising diffusion probabilistic model for audio restoration — Implemented in v0.4.0 (`dl_denoising/diffusion.rs`, `dl_denoising/audio_diffusion.rs`)
- [x] Pre-trained model weight loading from oxicode format — Implemented in v0.4.3 (`model_weights.rs`: `SignalWeightStore`, `SignalWeightFormat`, save/load via `oxicode` binary or JSON with path-based convenience API)

### Modal Analysis (Structural Dynamics)
- [x] Frequency Domain Decomposition (FDD) for operational modal analysis — Implemented in v0.4.0 (`modal_analysis/fdd.rs`)
- [x] Enhanced FDD (EFDD) with damping estimation — Implemented in v0.4.2 (`oma_efdd.rs`)
- [x] Stochastic Subspace Identification (SSI-COV, SSI-DATA) — Implemented in v0.4.0 (`modal_analysis/ssi.rs`)
- [x] Modal Assurance Criterion (MAC) for mode shape comparison — Implemented in v0.4.0 (`modal_analysis/mac.rs`)

### Advanced Array Processing — Implemented in v0.4.0
- [x] Delay-and-sum beamforming for microphone / sensor arrays
- [x] MVDR (Capon) beamformer
- [x] MUSIC / ESPRIT for direction-of-arrival (DOA) estimation
- [x] Adaptive beamforming with interference cancellation

---

## Known Issues

- CEEMDAN with very short signals (<256 samples) may produce spurious IMFs; EEMD is more stable in this regime
- N4SID identification with high model orders (>20) requires well-conditioned data; use regularized variant
- NMF audio separation is sensitive to initialization; multiple random restarts recommended for reliable separation
- **Music Information Retrieval is largely unimplemented as of v0.6.5** — `mir.rs` (the crate's only implementation of beat tracking, tempo estimation, key detection, tonal centroid, and self-similarity structural segmentation) was deleted in this release's dead-code cleanup with no replacement wired anywhere. Only CQT-based chroma (`cqt::chromagram`) and spectral-flux onset detection (`streaming::spectral_analysis::SpectralFlux`) remain available. Restoring the rest would mean re-implementing from scratch, not re-wiring — the source is gone, not just unreachable.
