# scirs2-signal TODO (v0.1.0-rc.1)

This module provides signal processing functionality similar to SciPy's signal module. Following the [SciRS2 POLICY](../SCIRS2_POLICY.md), this module is part of the RC.1 release with ecosystem consistency through scirs2-core abstractions.

## 🚨 URGENT: SIMD DISABLING PROJECT (Current Priority)

**Status**: Active compilation issue resolution
**Target**: Fully functional scirs2-signal module with SIMD disabled  
**Estimated Time**: 5-7 hours
**Current Errors**: 1000+ due to missing SIMD implementations

### IMMEDIATE TASKS (Current Session)
- [x] **Phase 0**: Project documentation setup (TODO.md, PROGRESS.md, README.md)
- [ ] **Phase 1**: Error categorization and analysis (30 mins)
- [ ] **Phase 2**: Investigate scalar implementation fallbacks (45 mins)  
- [ ] **Phase 3**: Create automated fix scripts (1-1.5 hours)
- [ ] **Phase 4**: Implement missing scalar functions (2-3 hours)
- [ ] **Phase 5**: Fix non-SIMD issues (rand::chacha, ndarray::s) (1 hour)
- [ ] **Phase 6**: Full compilation testing (30 mins)
- [ ] **Phase 7**: Documentation and handover prep (45 mins)

### IMPLEMENTATION STRATEGY
- **Disable SIMD optimizations** via scalar fallbacks
- **Preserve API surface** for future SIMD re-enabling
- **Focus on correctness** over performance for now
- **Systematic pattern-based** batch fixing approach

**📋 Detailed planning**: See `/tmp/simd_disabling_project_plan.md`
**📊 Progress tracking**: See `PROGRESS.md`

---

## Production-Ready Features (v0.1.0-rc.1)

### Core Signal Processing ✅
- [x] Module structure and error handling
- [x] Comprehensive filtering system
  - [x] IIR filters (Butterworth, Chebyshev I/II, Elliptic, Bessel)
  - [x] FIR filters (Window method, Parks-McClellan/Remez)
  - [x] Zero-phase filtering (filtfilt)
  - [x] Specialized filters (notch, comb, allpass, peak)
  - [x] Filter analysis and stability checking
  - [x] Savitzky-Golay filters
- [x] Signal convolution and correlation
  - [x] 1D convolution with different modes
  - [x] Cross-correlation and autocorrelation
  - [x] Basic deconvolution
- [x] Spectral analysis fundamentals
  - [x] Periodogram
  - [x] Welch's method for PSD estimation
  - [x] Short-time Fourier transform (STFT)
  - [x] Spectrogram computation
  - [x] Signal detrending (constant, linear, polynomial)
- [x] Core wavelet transforms
  - [x] Discrete Wavelet Transform (DWT)
  - [x] Continuous Wavelet Transform (CWT)
  - [x] Multiple wavelet families (Haar, Daubechies, Morlet, Meyer, etc.)
  - [x] Wavelet-based denoising (basic methods)
- [x] Linear system analysis (basic)
  - [x] Transfer function representation
  - [x] Frequency response calculation
  - [x] Basic stability analysis
- [x] Peak detection and analysis
  - [x] Peak finding with various criteria
  - [x] Peak properties (prominence, width)
- [x] Waveform generation
  - [x] Basic waveforms (sine, cosine, square, sawtooth, triangle)
  - [x] Specialized signals (chirp, Gaussian pulse, noise)
- [x] Signal measurements
  - [x] RMS, SNR, THD calculations
  - [x] Peak-to-peak and peak-to-RMS ratios
- [x] Basic resampling
  - [x] Up/down sampling
  - [x] Arbitrary rate resampling
- [x] Code quality improvements
  - [x] Comprehensive test coverage for core features
  - [x] Well-documented APIs with examples

## Planned for Future Releases

### Next Priority (v0.1.0-rc.1)
- [ ] Enhanced spectral analysis
  - [ ] Multitaper spectral estimation (refine and validate)
  - [ ] Lomb-Scargle periodogram (add more validation)
  - [ ] Parametric spectral estimation (AR, ARMA models)
- [ ] Advanced wavelet features
  - [ ] 2D wavelet transforms (refine implementation)
  - [ ] Wavelet packet transforms (add more validation)
  - [ ] Advanced denoising methods
- [ ] Improved LTI system analysis
  - [ ] Enhanced system identification
  - [ ] More robust controllability/observability analysis
- [ ] Performance optimization
  - [ ] Parallel processing for filtering operations
  - [ ] SIMD vectorization for compute-intensive operations
  - [ ] Memory optimization for large signals
- [ ] Comprehensive test suite
  - [ ] Numerical validation against SciPy
  - [ ] Integration tests for complex workflows
  - [ ] Performance benchmarks

### Medium-term Goals (v0.1.0)
- [ ] Advanced time-frequency analysis
  - [ ] Wigner-Ville distribution (stabilize)
  - [ ] Reassigned spectrograms (refine)
  - [ ] Synchrosqueezed wavelet transforms (validate)
- [ ] Signal enhancement and restoration
  - [ ] Advanced denoising algorithms
  - [ ] Deconvolution techniques
  - [ ] Missing data interpolation
- [ ] Specialized processing
  - [ ] Blind source separation methods
  - [ ] Sparse signal recovery
  - [ ] Robust filtering for outliers
- [ ] Real-time processing capabilities
  - [ ] Streaming STFT (validate and optimize)
  - [ ] Low-latency filtering
  - [ ] Memory-efficient large signal processing

### Long-term Vision (v0.2.0+)

- [ ] Complete SciPy signal module parity
  - [ ] Numerical accuracy validation against SciPy
  - [ ] Performance benchmarking and optimization
  - [ ] API compatibility layer

- [ ] Advanced integration with scirs ecosystem
  - [ ] Seamless integration with scirs2-interpolate
  - [ ] Matrix-based processing with scirs2-linalg
  - [ ] Parameter estimation with scirs2-optimize
  - [ ] Sparse representations with scirs2-sparse

- [ ] Domain-specific extensions
  - [ ] Audio processing toolkit
  - [ ] Biomedical signal analysis
  - [ ] Communications signal processing
  - [ ] Radar and sonar processing

- [ ] High-performance computing
  - [ ] GPU acceleration for large datasets
  - [ ] Real-time processing with bounded latency
  - [ ] SIMD optimization for critical paths
  - [ ] Distributed processing capabilities

- [ ] Advanced ecosystem features
  - [ ] Visualization tools for signal analysis
  - [ ] Interactive notebooks and tutorials
  - [ ] Machine learning integration
  - [ ] Comprehensive benchmarking suite

## Development Notes

### Code Quality Standards
- All code must pass `cargo clippy` without warnings
- Comprehensive test coverage for all public APIs
- Documentation with examples for all public functions
- Numerical validation against reference implementations

### Performance Requirements
- Memory-efficient algorithms for large signals
- Parallel processing where applicable
- Benchmarking against established libraries
- Zero-copy operations where possible

### API Design Principles
- Consistent error handling patterns
- Clear parameter validation
- Intuitive function naming following SciPy conventions
- Comprehensive documentation with usage examples