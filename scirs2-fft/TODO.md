# scirs2-fft Development TODO

## Status: v0.6.2 (released, 2026-07-22)

scirs2-fft's own test suite (freshly re-run 2026-07-22): 674 tests pass, 0 skipped, 0 failed with default features; 718 tests pass, 0 skipped, 0 failed with `--all-features`.

## v0.3.3 — COMPLETED

### Sparse FFT Algorithms
- Sublinear sparse FFT (O(k log n) for k-sparse signals, randomized hashing)
- Compressed sensing-based sparse FFT (LASSO-style recovery)
- Iterative sparse FFT (robust to noise via ISTA/FISTA)
- Frequency pruning variant
- Spectral flatness-based sparse FFT
- Prony method for damped sinusoid recovery
- MUSIC algorithm (Multiple Signal Classification) for super-resolution
- Batch sparse FFT with parallel CPU execution

### Chirp-Z Transform (CZT)
- CZT on arbitrary contours in the z-plane
- Zoom FFT as a special case of CZT
- Bluestein's algorithm for prime-length FFT

### Fractional Fourier Transform (FrFT)
- Ozaktas-Arikan algorithm (decomposition into chirp multiply-FFT steps)
- Candan sampling-based algorithm
- Complex signal FrFT (`frft_complex`)
- Batch FrFT for multiple rotation angles

### Number-Theoretic Transform (NTT)
- NTT over arbitrary NTT-friendly primes
- Inverse NTT
- Negacyclic NTT (used in lattice cryptography)
- Polynomial multiplication via NTT

### Lomb-Scargle Periodogram
- Fast Lomb-Scargle (extirpolation + FFT, O(n log n + N log N))
- Generalized Lomb-Scargle with floating mean
- FAP (false alarm probability) estimation

### Mixed-Radix FFT
- Arbitrary composite-length FFT (Cooley-Tukey + Rader + Bluestein)
- Mixed-radix 2D and N-dimensional FFT
- Split-radix FFT for 2^n lengths

### DCT/DST Variants (complete set)
- DCT types I–VIII
- DST types I–VIII
- Inverse transforms for all types
- MDCT and MDST (Modified DCT/DST)
- N-dimensional DCT/DST via separable application

### Wavelet Packet Transform
- Full wavelet packet decomposition tree
- Best-basis selection via Shannon entropy criterion
- Reconstruction from any subtree
- Wavelet families: Daubechies (up to 20), Symlets, Coiflets, Biorthogonal
- Continuous wavelet transform (CWT) via FFT convolution

### Polyphase Filter Bank
- Analysis and synthesis polyphase decomposition
- DFT-modulated (cosine-modulated) filter bank
- Critically sampled and oversampled modes
- Perfect reconstruction condition check

### Hilbert-Huang Transform (HHT)
- EMD (Empirical Mode Decomposition) via cubic spline envelope
- EEMD (Ensemble EMD) with white noise injection
- CEEMDAN (Complete EEMD with Adaptive Noise)
- Hilbert spectrum from IMFs
- Instantaneous frequency via Teager energy operator

### Spectral Analysis Enhancements
- Burg AR model spectral estimation
- Welch's method with configurable averaging
- Multitaper spectral estimation (DPSS)
- ESPRIT frequency estimator
- Capon beamformer spectral estimator

### Multidimensional FFT Utilities
- `multidim.rs` / `multidim_utils.rs` — separable N-dimensional plans
- In-place tiled 2D FFT for large arrays
- Row-column FFT with configurable tile size

### Convolution and Correlation
- Overlap-save (OLS) convolution
- Overlap-add (OLA) convolution
- FFT-based cross-correlation
- FFT-based polynomial multiplication
- Correlation-based delay estimation

### Window Functions Library
- 100+ windows including Kaiser-Bessel derived (KBD), DPSS, Parzen, Bohman
- `window_functions.rs` module with parameterized window generation
- Window coherent gain and noise bandwidth computation

### Spectrogram Enhancements
- Enhanced normalized spectrogram with configurable dynamic range
- Waterfall 3D data generation (mesh, line stacks)
- Reassigned spectrogram (improved time-frequency localization)
- Synchrosqueezed STFT

---

## v0.4.0 — Planned

### GPU FFT via OxiFFT GPU Backend
- [x] GPU-accelerated FFT via dispatch layer — Implemented in v0.4.3 (`gpu_fft/dispatch.rs` `fft_auto_dispatch`, `gpu_fft/wgpu_backend.rs` wgpu stub with WGSL shader, `wgpu_fft` feature gate; CPU path always functional via `GpuFftPipeline`)
- [x] Automatic CPU/GPU dispatch based on input size and available hardware — Implemented in v0.4.3 (`gpu_fft/dispatch.rs` `AutoDispatchConfig` with `gpu_threshold=4096`; routes to wgpu when feature+adapter available, falls back to CPU)
- [x] GPU batch FFT for many same-size transforms in parallel — Implemented in v0.4.3 (`gpu_fft/dispatch.rs` `fft_batch_gpu`; delegates to `GpuFftPipeline::execute_batch` with zero-pad alignment)
- [x] GPU overlap-save convolution for real-time filtering — Implemented in v0.4.3 (`gpu_fft/overlap_save.rs` `overlap_save_gpu`; pure-Rust FFT-based OLS, CPU path always available)

### Streaming FFT for Large Data
- [x] Streaming FFT processor with configurable buffer and overlap — Implemented in v0.4.2 (`streaming.rs` `StreamingFft`, overlap-add/overlap-save, Hann/Hamming/Blackman/Rectangular windows)
- [x] Out-of-core 2D FFT for images too large for RAM — Implemented in v0.4.2 (`outofcore.rs` `OutOfCoreFft2D`, row/column decomposition, disk-based transpose via `tempfile`)
- [x] Streaming spectrogram with rolling window output — Implemented in v0.4.2 (`ring_buffer_stft.rs` `StreamingSpectrogram`)
- [x] Ring-buffer STFT for online/real-time applications — Implemented in v0.4.2 (`ring_buffer_stft.rs` `RingBufferStft` with overlap-add reconstruction)

### Quantum FFT
- [x] Quantum Fourier Transform circuit simulation (via `scirs2-core` quantum primitives) — Implemented in v0.4.0 (`quantum/qft.rs`)
- [x] Phase estimation circuit using QFT — Implemented in v0.4.0 (`quantum/phase_estimation.rs`)
- [x] Shor's algorithm building blocks — Implemented in v0.4.0 (`shor/mod.rs`)

### Additional Algorithms
- [x] Short-time fractional Fourier transform (STFRFT) — Implemented in v0.4.0 (`fractional/stfrft.rs`)
- [x] Wigner-Ville distribution (full, smoothed) — Implemented in v0.4.0 (`wigner_ville/` module)
- [x] Ambiguity function computation — Implemented in v0.4.0 (`ambiguity/mod.rs`)
- [x] Cyclostationary spectral analysis — Implemented in v0.4.0 (`cyclostationary/` module)
- [x] Ramanujan periodic transform — Implemented in v0.4.0 (`ramanujan/mod.rs`)

### Performance
- [x] AVX-512 butterfly kernels for radix-4 and radix-8 FFT stages (planned 2026-04-17)
  - **Goal:** Pure-Rust AVX-512 intrinsics path in `butterfly.rs` / `simd_fft/` that delivers measurably faster radix-4 and radix-8 butterflies than the scalar baseline for f32 and f64. Feature-gated on `target_feature = "avx512f"` with runtime dispatch.
  - **Design:** Use `std::arch::x86_64` AVX-512 intrinsics (`_mm512_fmadd_pd`, `_mm512_mul_pd`, `_mm512_permutex_pd`, `_mm512_shuffle_f64x2`, etc.). Radix-4 butterfly on 8 complex f64 lanes (2 radix-4 at once, 8 complex = 16 f64 lanes). Radix-8 butterfly on 4 complex f64 via twiddle pre-fetch. Runtime dispatch: `is_x86_feature_detected!("avx512f")` at the plan boundary; scalar fallback otherwise. Criterion benchmark `butterfly_bench.rs` to demonstrate speedup.
  - **Files:** `scirs2-fft/src/butterfly.rs` (extend), `scirs2-fft/src/simd_fft/avx512.rs` (new), `scirs2-fft/src/simd_fft/mod.rs` (dispatch), `scirs2-fft/benches/butterfly_bench.rs` (new), `scirs2-fft/tests/avx512_correctness_tests.rs` (new), `scirs2-fft/TODO.md`.
  - **Prerequisites:** none.
  - **Tests:** `avx512_radix4_matches_scalar`, `avx512_radix8_matches_scalar`, `avx512_matches_full_fft_on_random_input`, plus gated criterion bench.
  - **Risk:** AVX-512 not available on CI runners → tests gated on `is_x86_feature_detected!("avx512f")` with fallthrough to "skipped on this host" marker. CI compile-checks the path only; document.
- [x] NEON/SVE butterfly kernels for ARM (planned 2026-04-17)
  - **Goal:** ARM64 NEON (AArch64 baseline) and optional SVE (scalable vector length) butterfly kernels, same radix-4 / radix-8 coverage. Feature-gated, runtime-dispatched.
  - **Design:** NEON intrinsics via `std::arch::aarch64` (`vfmaq_f64`, `vmulq_f64`, `vtrnq_f64`, etc.). Radix-4 on 2 complex f64 lanes = 4 f64. Radix-8 via two radix-4 + twiddle. SVE path gated on `target_feature = "sve"`, scalable-length via `svfloat64_t`. Runtime dispatch at plan boundary via `is_aarch64_feature_detected!("neon")` / `is_aarch64_feature_detected!("sve")`.
  - **Files:** `scirs2-fft/src/simd_fft/neon.rs` (new), `scirs2-fft/src/simd_fft/sve.rs` (new), `scirs2-fft/src/simd_fft/mod.rs` (extend dispatch), `scirs2-fft/tests/neon_correctness_tests.rs` (new), `scirs2-fft/TODO.md`.
  - **Prerequisites:** none.
  - **Tests:** `neon_radix4_matches_scalar`, `neon_radix8_matches_scalar`, `sve_radix4_matches_scalar`, plus cross-check against scalar following the avx512 test pattern.
  - **Risk:** SVE availability is sparse on CI; gate its tests behind `is_aarch64_feature_detected!("sve")` and accept compile-only coverage when feature is absent.
- [x] Cache-oblivious recursive FFT (Frigo-Johnson style) — Implemented in v0.4.0 (`cache_oblivious.rs`); real-input variant `cache_oblivious_rfft` added in v0.4.2
- [x] FFT plan serialization for ahead-of-time compilation — Implemented in v0.4.0 (`fft_plan.rs`, `plan_serialization.rs`)

---

## Known Issues / Technical Debt

- `spectral.rs` was deleted and replaced by the `spectral/` submodule; verify no broken re-exports remain
- `nufft_legacy.rs` (359 lines) was marked for removal in v0.4.1 but is still present at v0.6.1; verified 2026-07-15 that it is now fully orphaned — not declared as a module anywhere (`grep` for `mod nufft_legacy` and for any reference to `nufft_legacy` across `src/` returns nothing outside the file itself), so it is dead code not compiled into the crate at all. Safe to delete in a future source-code pass (out of scope for this docs-only update).
- EMD cubic spline envelope may not converge for highly non-stationary signals; add iteration cap with warning
- NTT works only for inputs whose length divides `p - 1`; document this constraint clearly
- Lomb-Scargle FAP estimation is approximate (chi-squared); implement bootstrap alternative
- [x] No source file currently exceeds 2000 lines (verified 2026-07-15: largest is `src/polynomial/legacy.rs` at 1989 lines); re-check with `rslines 50` if new files grow past the threshold
- GPU sparse FFT feature flags (`cuda`, `hip`, `sycl`) depend on external hardware; CI uses mock backend
- STFT `istft` reconstruction requires correct `noverlap`; add assertion for perfect reconstruction condition
- Wavelet packet tree reconstruction is not yet invertible for all wavelet families; test suite should cover round-trip error
