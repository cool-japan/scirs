#![allow(clippy::all)]
#![allow(dead_code)]
#![allow(unreachable_patterns)]
#![allow(unused_assignments)]
#![allow(unused_variables)]
#![allow(private_interfaces)]
//! # SciRS2 Signal - Digital Signal Processing
//!
//! **scirs2-signal** provides comprehensive signal processing capabilities modeled after SciPy's
//! `signal` module, offering filtering, spectral analysis, wavelet transforms, system identification,
//! and time-frequency analysis with SIMD acceleration and parallel processing.
//!
//! ## 🎯 Key Features
//!
//! - **SciPy Compatibility**: Drop-in replacement for `scipy.signal` functions
//! - **Digital Filters**: FIR, IIR, Butterworth, Chebyshev, elliptic, Bessel
//! - **Spectral Analysis**: FFT-based PSD, spectrograms, Lomb-Scargle periodograms
//! - **Wavelet Transforms**: DWT, CWT, dual-tree complex wavelets, 2D transforms
//! - **Convolution**: Fast 1D/2D convolution with SIMD and parallel support
//! - **LTI Systems**: Transfer functions, state-space, frequency response
//! - **Advanced Methods**: EMD, Hilbert transform, system identification
//!
//! ## 📦 Module Overview
//!
//! | SciRS2 Module | SciPy Equivalent | Description |
//! |---------------|------------------|-------------|
//! | `filter` | `scipy.signal.butter`, `cheby1` | Digital filter design (FIR/IIR) |
//! | `convolve` | `scipy.signal.convolve` | 1D/2D convolution and correlation |
//! | `spectral` | `scipy.signal.periodogram` | Power spectral density, spectrograms |
//! | `dwt` | `pywt.dwt` | Discrete wavelet transform |
//! | `wavelets` | `pywt.cwt` | Continuous wavelet transform |
//! | `window` | `scipy.signal.get_window` | Window functions (Hann, Hamming, etc.) |
//! | `lti` | `scipy.signal.TransferFunction` | LTI system representation |
//! | `lombscargle` | `scipy.signal.lombscargle` | Lomb-Scargle periodogram |
//!
//! ## 🚀 Quick Start
//!
//! ```toml
//! [dependencies]
//! scirs2-signal = "0.6.3"
//! ```
//!
//! ```rust
//! use scirs2_signal::{convolve, filter, spectral};
//!
//! // Convolution
//! let signal = vec![1.0, 2.0, 3.0];
//! let kernel = vec![0.25, 0.5, 0.25];
//! let filtered = convolve(&signal, &kernel, "same").expect("operation should succeed");
//! ```
//!
//! ## 🔒 Version: 0.6.3 (July 22, 2026)

// Core error handling - ESSENTIAL
pub mod error;
pub use error::{SignalError, SignalResult};

// Core modules
pub mod convolve;
pub mod convolve_parallel;
pub mod measurements;
pub mod utils;

// Window functions module
pub mod window;

// LTI (Linear Time-Invariant) systems module - required by filter
pub mod lti;

// Digital filter module
pub mod filter;

// Spectral analysis module
pub mod spectral;

// Discrete Wavelet Transform module
pub mod dwt;

// Enhanced 2D Discrete Wavelet Transform module
pub mod dwt2d_enhanced;

// Advanced-refined 2D Discrete Wavelet Transform module with memory efficiency
pub mod dwt2d_super_refined;

// Comprehensive wavelets module (CWT, dual-tree complex, etc.)
pub mod wavelets;

// Advanced wavelet features for v0.2.0
pub mod dwt2d_advanced;
pub mod wavelet_advanced;
pub mod wpt;
pub mod wpt2d;
pub mod wpt_enhanced;

// Additional signal processing modules
pub mod denoise;
pub mod denoise_advanced;
pub mod denoise_enhanced;
pub mod emd;
pub mod hilbert;
pub mod median;
pub mod parametric;
pub mod parametric_advanced;
pub mod parametric_advanced_enhanced;
pub mod spline;
pub mod swt;
pub mod sysid;
pub mod sysid_advanced_enhanced;
pub mod sysid_enhanced;
pub mod tv;
pub mod waveforms;

// Lomb-Scargle periodogram module (refactored)
pub mod lombscargle;
pub mod lombscargle_enhanced;
pub mod lombscargle_scipy_validation;
// pub mod utilities;
pub mod cqt;
pub mod simd_advanced;
// Chirp Z-Transform (generalisation of the DFT)
pub mod czt;
// pub mod wvd;
// pub mod nlm;
// pub mod wiener;
// pub mod dwt2d;
// pub mod swt2d;
// pub mod wavelet_vis;
// pub mod reassigned;
// pub mod deconvolution;
pub mod savgol;

// Signal processing submodules (temporarily disabled)
// pub mod bss;
// pub mod features;
pub mod multitaper;

// v0.3.0 Enhanced Spectral Analysis (multitaper, Lomb-Scargle, parametric)
pub mod spectral_advanced;

// v0.2.0 Advanced Spectral Analysis Modules
pub mod advanced_spectral_v2;
pub mod memory_optimized;
pub mod parallel_filtering_v2;
pub mod parallel_spectral;
pub mod spectral_scipy_validation_v2;

// v0.3.0 Real-time / streaming signal processing
pub mod streaming;

// v0.3.0 Adaptive filters (LMS, NLMS, RLS, VS-LMS, APA, FDLMS, LMF, SM-LMS)
pub mod adaptive;

// v0.3.0 Cepstral analysis (real/complex cepstrum, MFCC, Mel filter banks)
pub mod cepstral;

// v0.3.0 Modulation / demodulation (AM, FM, QAM)
pub mod modulation;

// v0.3.0 Beamforming (delay-and-sum, MVDR/Capon, steering vectors)
pub mod beamforming;

// v0.3.0 System Identification (ARX, ARMAX, OE, N4SID, RLS, PEM)
pub mod system_identification;

// v0.3.0 Enhanced Transfer Function Analysis (pole-zero, root locus, Nyquist, Nichols, margins)
pub mod tf_analysis;

// v0.3.0 State Space Operations (Gramians, balanced realization, model reduction, conversions)
pub mod state_space_ops;

// v0.3.0 Multi-channel signal processing (mixing, ICA, CSP, cross-correlation)
pub mod multichannel;

// v0.3.0 Time-Frequency Analysis (WVD, Choi-Williams, Cohen's class, reassignment)
pub mod time_frequency;

// v0.3.0 Signal Quality Metrics (SNR, SDR, PESQ-like, spectral flatness, crest factor)
pub mod signal_quality;

// v0.3.0 Resampling (polyphase, sinc interpolation, fractional delay, anti-aliasing)
pub mod resampling;

// Deep learning denoising
pub mod dl_denoising;
// Echo cancellation (multi-delay AEC)
pub mod echo_cancellation;
// GPU-accelerated signal processing
pub mod gpu;
// GPU-accelerated spectrogram computation
pub mod gpu_spectrograms;
// GPU-accelerated matched filter bank
pub mod gpu_matched_filter;
// GPU wavelet transform dispatch layer
pub mod gpu_wavelet;
pub use gpu_wavelet::{
    dwt_dispatch, dwt_dispatch_batch, GpuWaveletBackend, GpuWaveletConfig, GpuWaveletFamily,
};
// Operational modal analysis
pub mod modal_analysis;
// Batched Welch PSD for parallel multi-channel processing
pub mod welch_batch;
// Enhanced FDD (EFDD) with damping estimation
pub mod oma_efdd;
// Neural audio processing
pub mod neural_audio;
// Deep filtering via neural-predicted FIR coefficients
pub mod deep_filter;
// Pre-trained model weight loading / saving in oxicode format
pub mod model_weights;
pub use model_weights::{SignalWeightFormat, SignalWeightStore};
// Phase estimation (ESPRIT, MUSIC)
pub mod phase_estimation;
// Real-time DSP pipeline
pub mod realtime_dsp;

// Re-export core functionality
pub use convolve::{convolve, convolve_simd_ultra, correlate};
pub use convolve_parallel::{parallel_convolve1d, parallel_convolve_simd_ultra};
pub use measurements::{peak_to_peak, peak_to_rms, rms, snr, thd};

// Re-export key filter functionality
pub use filter::{analyze_filter, butter, filtfilt, firwin, FilterType};

// Re-export key LTI functionality
pub use lti::{design_tf, impulse_response, lsim, step_response, TransferFunction};

// Re-export key spectral analysis functionality
pub use spectral::{get_window_simd_ultra, periodogram, spectrogram, stft, welch};

// Re-export key DWT functionality
pub use dwt::{
    dwt_decompose, dwt_reconstruct, wavedec, waverec, DecompositionResult, Wavelet, WaveletFilters,
};

// Re-export key wavelets functionality
pub use wavelets::{complex_morlet, cwt, morlet, ricker, scalogram};

// Re-export key additional modules functionality
pub use parametric::{ar_spectrum, burg_method, yule_walker};
pub use parametric_advanced_enhanced::{
    adaptive_ar_spectral_estimation, advanced_enhanced_arma, high_resolution_spectral_estimation,
    multitaper_parametric_estimation, robust_parametric_spectral_estimation, AdaptiveARConfig,
    AdvancedEnhancedConfig, HighResolutionConfig, MultitaperParametricConfig,
    RobustParametricConfig,
};
pub use savgol::{savgol_coeffs, savgol_filter};
pub use swt::{iswt, swt, swt_decompose_simd_pipelined};
pub use tv::{tv_denoise_1d, tv_denoise_2d};
pub use waveforms::{chirp, sawtooth, square};

// Re-export advanced wavelet features for v0.2.0
pub use dwt2d_advanced::{
    denoise_2d, dwt2d_decompose, dwt2d_reconstruct, wavedec2, waverec2, Dwt2DCoeffs, EdgeMode2D,
    MultilevelDwt2D,
};
pub use wavelet_advanced::{
    advanced_denoise_1d, block_denoise_1d, select_best_basis, BestBasisResult,
    CostFunction as WaveletCostFunction, DenoisingConfig, ThresholdMode, ThresholdRule,
};
pub use wpt_enhanced::{
    best_basis_analysis, wpt_denoise, CostFunction as WptCostFunction, WaveletPacketTree, WptNode,
    WptValidationResult,
};

// Re-export v0.2.0 advanced spectral analysis functionality
pub use advanced_spectral_v2::{
    ar_spectral_estimation, arma_spectral_estimation, memory_optimized_ar_spectral,
    ARMASpectralConfig, ARMASpectralMethod, ARMASpectralResult, ARSpectralConfig, ARSpectralMethod,
    ARSpectralResult, MemoryOptimizedSpectralConfig, ParallelSpectralConfigV2,
    StreamingSpectralEstimator,
};
pub use parallel_filtering_v2::{
    batch_fir_filter, batch_iir_filter, parallel_fir_filter, parallel_iir_filter,
    parallel_median_filter, parallel_moving_average, parallel_savgol_filter, BatchFilterConfig,
    FIRFilterMethod, PaddingMode, ParallelFIRConfig, ParallelIIRConfig, StreamingFIRFilter,
    StreamingIIRFilter,
};
pub use spectral_scipy_validation_v2::{
    generate_validation_report, run_comprehensive_validation, ValidationResult, ValidationSuite,
};

// Re-export v0.3.0 adaptive filter functionality
pub use adaptive::{
    AdaptiveFilter, AdaptiveFilterConfig, AdaptiveMethod, ApaFilter, FdlmsFilter, LmfFilter,
    LmsFilter, NlmsFilter, RlsFilter, SmLmsFilter, VsLmsFilter,
};

// Re-export v0.3.0 cepstral analysis functionality
pub use cepstral::{
    complex_cepstrum, compute_deltas, mel_filter_bank, mfcc, mfcc_extract, mfcc_frame,
    real_cepstrum, MelFilterBankConfig, MfccConfig,
};

// Re-export v0.3.0 modulation/demodulation functionality
pub use modulation::{
    am_demodulate, am_modulate, demodulate, fm_demodulate, fm_modulate, modulate,
    qam_constellation, qam_demodulate_bits, qam_modulate_bits, qam_modulate_passband, AmMode,
    ModulationMethod, QamOrder, QamSymbol,
};

// Re-export v0.3.0 beamforming functionality
pub use beamforming::{
    beamform, delay_and_sum_filter, delay_and_sum_power, estimate_covariance,
    estimate_covariance_real, mvdr_power, mvdr_weights, scan_angles_degrees, steering_vector_ula,
    steering_vectors_ula, BeamformMethod,
};

// Re-export v0.3.0 system identification functionality
pub use system_identification::{
    armax_estimate, arx_estimate, n4sid_estimate, oe_estimate, pem_estimate, rls_batch,
    ArmaxConfig, ArxConfig, N4sidConfig, OeConfig, PemConfig, RlsConfig, RlsEstimator,
    SubspaceIdResult, SysIdResult,
};

// Re-export v0.3.0 transfer function analysis functionality
pub use tf_analysis::{
    nichols_chart, nyquist_diagram, pole_zero_analysis, root_locus, sensitivity_functions,
    stability_margins, NicholsResult, NyquistResult, PoleZeroResult, RootLocusResult,
    SensitivityResult, StabilityMargins,
};

// Re-export v0.3.0 state space operations functionality
pub use state_space_ops::{
    balanced_realization, balanced_truncation, compute_gramians, hankel_norm_reduction,
    minimal_realization, ss_feedback, ss_parallel, ss_series, ss_to_tf, tf_to_ss_controllable,
    tf_to_ss_observable, BalancedRealization, GramianResult, MinimalRealization, ReducedModel,
};

// Re-export v0.3.0 enhanced spectral analysis functionality
pub use spectral_advanced::{
    // Parametric methods
    burg_spectral,
    esprit_spectral,
    // Lomb-Scargle
    false_alarm_level,
    false_alarm_probability,
    lomb_scargle_periodogram,
    // Multitaper
    multitaper_ftest_line_detection,
    multitaper_psd,
    music_spectral,
    yule_walker_spectral,
    BurgConfig,
    BurgResult,
    EspritConfig,
    EspritResult,
    FTestResult as MultitaperFTestResult,
    FalseAlarmResult,
    FapMethod,
    LombScargleConfig,
    LombScargleNormalization,
    LombScargleResult,
    MultitaperConfig,
    MultitaperResult,
    MusicConfig,
    MusicResult,
    YuleWalkerConfig,
    YuleWalkerResult,
};

// Re-export v0.3.0 multi-channel processing functionality
pub use multichannel::{
    apply_mixing_matrix, cross_channel_correlation, cross_correlation_lag, csp, csp_apply, fastica,
    mix_to_mono, mono_to_multichannel, reorder_channels, select_channels, CspConfig, CspResult,
    FastIcaConfig, FastIcaResult, MixMode, MultiChannelSignal,
};

// Re-export v0.3.0 time-frequency analysis functionality
pub use time_frequency::{
    choi_williams, cohens_class, gaussian_window as tf_gaussian_window,
    hann_window as tf_hann_window, instantaneous_amplitude, instantaneous_frequency,
    kernel_born_jordan, kernel_wigner_ville, pseudo_wigner_ville, reassigned_spectrogram,
    smoothed_pseudo_wigner_ville, wigner_ville, CohenKernelFn, ReassignedTfDistribution,
    TfDistribution,
};

// Re-export v0.3.0 signal quality metrics functionality
pub use signal_quality::{
    crest_factor as signal_crest_factor, crest_factor_db, dynamic_range, enob, perceptual_quality,
    segmental_snr, si_sdr, sinad, snr_blind, snr_from_noise_floor, snr_reference,
    spectral_flatness, spectral_flatness_frames, zero_crossing_rate, zero_crossing_rate_frames,
    BlindSnrConfig, DynamicRangeResult, PerceptualQualityResult,
};

// Re-export v0.3.0 resampling functionality
pub use resampling::{
    decimate, design_anti_alias_filter, downsample, fractional_delay, interpolate,
    lagrange_delay_filter, resample, resample_poly, resample_to_length, sinc_delay_filter,
    upsample, ResamplingConfig, ResamplingQuality, WindowType as ResamplingWindowType,
};

// Re-export batched Welch PSD
pub use welch_batch::{BatchedWelch, WelchConfig, WelchResult, WelchWindow};

// Re-export EFDD
pub use oma_efdd::{efdd, EfddConfig, EfddMode, EfddResult};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dwt::utils::check_perfect_reconstruction;
    use crate::dwt::{wavedec, waverec, Wavelet};

    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }

    /// Real numerical perfect-reconstruction (PR) test: `waverec(wavedec(x))`
    /// must match `x` to a tight (1e-10) tolerance, for a parametrized set of
    /// genuinely PR-capable (orthogonal wavelet family, extension mode,
    /// decomposition level, signal length) configurations.
    ///
    /// This replaces the previous version of this test, which only asserted
    /// that the reconstructed signal was non-empty. That was not just weak
    /// but actively misleading: `wavedec(&[1.0..8.0], Wavelet::DB(4),
    /// Some(1), None)` silently decomposes to a single (identity) level,
    /// because `wavedec`'s `max_level` computation clamps to 0 whenever the
    /// signal length does not exceed the filter length (8 samples vs. DB(4)'s
    /// 8-tap filter) -- so the old test never actually exercised the
    /// transform at all.
    ///
    /// ## Confirmed PR requirements (empirically determined)
    ///
    /// * **Filter validity is necessary but not sufficient.** The raw filter
    ///   coefficients must satisfy the QMF/orthonormality conditions checked
    ///   by [`check_perfect_reconstruction`] (lowpass DC gain = sqrt(2),
    ///   highpass DC gain = 0, unit energy, double-shift orthogonality).
    ///   `Haar`, `DB(n)`, and `Coif(n)` all satisfy this to 1e-10.
    ///   **`Sym(4)`, `Sym(5)`, and `Sym(8)` do not** (e.g. `Sym(4)`'s
    ///   highpass DC gain is ~4e-2, not ~0, and its filter energy is off by
    ///   ~8e-4) -- their hardcoded coefficients in `dwt/filters/mod.rs` are
    ///   not valid conjugate-mirror filters. That is a distinct, pre-existing
    ///   data bug (already hinted at by the deliberately loose bounds in
    ///   `test_symlet_finite_energy` in
    ///   `tests/dwt_advanced_wavelet_test.rs`) and is out of scope for this
    ///   test; `Coif(2)` is used as the third orthogonal family instead of
    ///   `Sym(4)`.
    /// * **Any signal length works once decomposition actually happens**,
    ///   not just lengths that are exact powers of two equal to the filter
    ///   length -- both the default ("symmetric") and "periodic" extension
    ///   modes give tight PR.
    /// * **`wavedec`/`waverec` had a real, fixed length bug** for every
    ///   filter longer than Haar's 2 taps: `waverec` only cropped
    ///   intermediate reconstruction levels to their correct length (by
    ///   cross-referencing the next stored detail array's length); the
    ///   *final* (finest) level had no further stored array to crop against,
    ///   so it came out at `dwt_reconstruct`'s raw, uncropped
    ///   `2 * input_len` samples instead of the original signal length
    ///   (e.g. `DB(4)` on a 16-sample signal reconstructed to 22 samples).
    ///   Fixed in `dwt::multiscale::waverec` by cropping *every* level
    ///   (including the last) to the canonical single-level reconstruction
    ///   length `2 * input_len - filter_len + 2`, which mirrors the
    ///   encode-side `output_len = (n + filter_len - 1) / 2` formula in
    ///   `dwt::transform::dwt_decompose` and is exact whenever the
    ///   pre-decomposition length and filter length share the same parity
    ///   (true for every even-tap filter bank exercised here). This fix is
    ///   confined to `multiscale.rs`; `dwt_reconstruct` itself (used
    ///   directly by `wpt.rs`, `dwt2d/*`, `wavelet_advanced.rs`, and
    ///   `denoise_adaptive_advanced.rs`) is untouched, so those call sites
    ///   keep their existing (self-managed) length handling.
    #[test]
    fn test_dwt_phase3_verification() {
        // Comfortably below the worst-case error observed across every case
        // below (~3e-11, for Coif(2) at level 2 on a 64-sample signal).
        const PR_TOLERANCE: f64 = 1e-10;

        // A smooth, bounded, non-symmetric signal. Amplitude matters for an
        // *absolute*-error tolerance: a large-magnitude ramp (e.g. 1..=64)
        // still reconstructs correctly, but pushes the level-3 `Coif(2)`
        // error toward ~1e-9 purely from floating-point accumulation over
        // larger numbers -- an amplitude effect, not a PR failure. Keeping
        // the signal in a small, bounded range avoids conflating the two.
        fn test_signal(n: usize) -> Vec<f64> {
            (0..n)
                .map(|i| {
                    let x = i as f64;
                    1.0 + 0.5 * (x * 0.37).sin() + 0.25 * (x * 0.11).cos()
                })
                .collect()
        }

        // (family label, wavelet, signal length, decomposition level, mode)
        let cases: Vec<(&str, Wavelet, usize, usize, Option<&str>)> = vec![
            // Haar: trivially exact PR at any length/level (2-tap filter).
            ("Haar", Wavelet::Haar, 16, 1, None),
            ("Haar", Wavelet::Haar, 16, 2, None),
            ("Haar", Wavelet::Haar, 64, 3, Some("periodic")),
            // DB(4): 8-tap orthogonal filter.
            ("DB(4)", Wavelet::DB(4), 16, 1, None),
            ("DB(4)", Wavelet::DB(4), 32, 2, None),
            ("DB(4)", Wavelet::DB(4), 64, 2, Some("periodic")),
            ("DB(4)", Wavelet::DB(4), 64, 3, None),
            // Coif(2): 12-tap orthogonal filter.
            ("Coif(2)", Wavelet::Coif(2), 32, 1, None),
            ("Coif(2)", Wavelet::Coif(2), 64, 2, None),
            ("Coif(2)", Wavelet::Coif(2), 64, 2, Some("periodic")),
        ];

        for (name, wavelet, len, level, mode) in cases {
            // The filters themselves must satisfy the QMF/orthonormality
            // conditions -- necessary for PR, and independent of
            // wavedec/waverec's length bookkeeping.
            let filters = wavelet
                .filters()
                .unwrap_or_else(|e| panic!("{name}: filters() should succeed: {e:?}"));
            let filters_are_pr = check_perfect_reconstruction(&filters, Some(PR_TOLERANCE))
                .unwrap_or_else(|e| panic!("{name}: check_perfect_reconstruction errored: {e:?}"));
            assert!(
                filters_are_pr,
                "{name}: filter coefficients do not satisfy perfect-reconstruction conditions"
            );

            let signal = test_signal(len);
            let coeffs = wavedec(&signal, wavelet, Some(level), mode).unwrap_or_else(|e| {
                panic!("{name} len={len} level={level} mode={mode:?}: wavedec failed: {e:?}")
            });
            let reconstructed = waverec(&coeffs, wavelet).unwrap_or_else(|e| {
                panic!("{name} len={len} level={level} mode={mode:?}: waverec failed: {e:?}")
            });

            assert_eq!(
                reconstructed.len(),
                signal.len(),
                "{name} len={len} level={level} mode={mode:?}: reconstructed length {} != original length {}",
                reconstructed.len(),
                signal.len(),
            );

            let max_err = signal
                .iter()
                .zip(reconstructed.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);

            assert!(
                max_err < PR_TOLERANCE,
                "{name} len={len} level={level} mode={mode:?}: perfect reconstruction failed, \
                 max error {max_err:e} >= tolerance {PR_TOLERANCE:e}"
            );
        }
    }
}
