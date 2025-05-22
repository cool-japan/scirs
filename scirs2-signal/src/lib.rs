//! Signal processing module
//!
//! This module provides implementations of various signal processing algorithms
//! for filtering, convolution, and spectral analysis.
//!
//! Refactored modules:
//! - features: Comprehensive time series feature extraction in modular structure
//! - wavelets: Wavelet transforms and related functionality in modular structure
//! - multitaper: Multitaper spectral estimation methods in modular structure
//!
//! ## Overview
//!
//! * Filtering: FIR and IIR filters, filter design, Savitzky-Golay filter
//! * Convolution and correlation
//! * Spectral analysis and periodograms
//! * Short-time Fourier transform (STFT) and spectrograms
//! * Wavelet transforms (1D and 2D)
//! * Peak finding and signal measurements
//! * Waveform generation and processing
//! * Resampling and interpolation
//! * Linear Time-Invariant (LTI) systems analysis
//! * Chirp Z-Transform (CZT) for non-uniform frequency sampling
//! * Signal detrending and trend analysis
//! * Hilbert transform and analytic signal analysis
//! * Multi-resolution analysis with wavelets (CWT, DWT, DWT2D, SWT, SWT2D, WPT)
//! * Time-frequency analysis with Wigner-Ville distribution, reassigned spectrograms, and synchrosqueezed wavelets
//! * Parametric spectral estimation (AR, ARMA models)
//! * Higher-order spectral analysis (bispectrum, bicoherence, trispectrum)
//! * Signal denoising techniques (Wiener, Non-Local Means, Total Variation, Median, Kalman)
//! * Signal deconvolution (Wiener, Richardson-Lucy, Tikhonov, Total Variation, Blind)
//! * Blind source separation (ICA, PCA, NMF, Sparse Component Analysis)
//! * Missing data interpolation (Linear, Cubic Spline, Gaussian Process, Kriging, RBF, Spectral)
//! * Sparse signal recovery (OMP, MP, CoSaMP, ISTA, FISTA, Basis Pursuit)
//! * Compressed sensing and missing data reconstruction
//! * Advanced filtering (median filtering for impulse noise removal)
//! * State estimation (Kalman filtering, Extended Kalman, Unscented Kalman)
//! * Synchrosqueezed Wavelet Transform (SSWT) for improved time-frequency analysis
//! * Window functions (Hamming, Hann, Blackman, etc.) for spectral analysis
//!
//! ## Examples
//!
//! ```
//! use scirs2_signal::filter::butter;
//! use scirs2_signal::filter::filtfilt;
//!
//! // Generate a simple signal and apply a Butterworth filter
//! // let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
//! // let fs = 100.0;  // Sample rate in Hz
//! // let cutoff = 10.0;  // Cutoff frequency in Hz
//! //
//! // let (b, a) = butter(4, cutoff / (fs / 2.0), "lowpass").unwrap();
//! // let filtered = filtfilt(&b, &a, &signal).unwrap();
//! ```

extern crate openblas_src;

// Export error types
pub mod error;
pub use error::{SignalError, SignalResult};

// Signal processing module structure
pub mod bss;
pub mod convolve;
pub mod cqt;
pub mod czt;
pub mod deconvolution;
pub mod denoise;
pub mod detrend;
pub mod dwt;
pub mod dwt2d;
pub mod dwt2d_image;
pub mod emd;
pub mod features;
pub mod image_features;
pub mod utilities;

// Re-export the main public DWT functionality
pub use dwt::{
    dwt_decompose, dwt_reconstruct, extend_signal, wavedec, waverec, Wavelet, WaveletFilters,
};
pub mod filter;
pub mod higher_order;
pub mod interpolate;
pub mod kalman;
pub mod lombscargle;
pub mod lti;
pub mod lti_response;
pub mod median;
pub mod multitaper;
pub mod nlm;
pub mod parametric;
pub mod peak;
pub mod phase_vocoder;
pub mod reassigned;
pub mod resample;
pub mod savgol;
pub mod sparse;
pub mod spectral;
pub mod spline;
pub mod sswt;
pub mod stft;
pub mod swt;
pub mod swt2d;
pub mod tv;
pub mod waveforms;
pub mod wavelet_vis;
pub mod wavelets;
pub mod wiener;
pub mod window;
pub mod wpt;
pub mod wpt2d;
pub mod wvd;

// Re-export commonly used functions
pub use bss::{
    calculate_correlation_matrix, calculate_mutual_information, estimate_source_count, ica,
    joint_bss, joint_diagonalization, kernel_ica, multivariate_emd, nmf, pca, sort_components,
    sparse_component_analysis, BssConfig, IcaMethod, NonlinearityFunction,
};
pub use convolve::{convolve, convolve2d, correlate, deconvolve};
pub use cqt::{
    chromagram, constant_q_transform, cqt_magnitude, inverse_constant_q_transform, CqtConfig,
};
pub use deconvolution::{
    blind_deconvolution_1d, blind_deconvolution_2d, clean_deconvolution_1d, mem_deconvolution_1d,
    optimal_deconvolution_1d, richardson_lucy_deconvolution_1d, richardson_lucy_deconvolution_2d,
    richardson_lucy_deconvolution_color, tikhonov_deconvolution_1d, tv_deconvolution_2d,
    wiener_deconvolution_1d, wiener_deconvolution_2d, wiener_deconvolution_color,
    DeconvolutionConfig, DeconvolutionMethod,
};
pub use filter::{bessel, butter, cheby1, cheby2, ellip, filtfilt, firwin, lfilter};
pub use higher_order::{
    biamplitude, bicoherence, bispectrum, cumulative_bispectrum, detect_phase_coupling,
    skewness_spectrum, trispectrum, BispecEstimator, HigherOrderConfig,
};
pub use interpolate::{
    auto_interpolate, cubic_hermite_interpolate, cubic_spline_interpolate,
    gaussian_process_interpolate, interpolate, interpolate_2d, kriging_interpolate,
    linear_interpolate, minimum_energy_interpolate, nearest_neighbor_interpolate, rbf_functions,
    rbf_interpolate, sinc_interpolate, spectral_interpolate, variogram_models, InterpolationConfig,
    InterpolationMethod,
};
pub use kalman::{
    adaptive_kalman_filter, ensemble_kalman_filter, extended_kalman_filter, kalman_denoise_1d,
    kalman_denoise_2d, kalman_denoise_color, kalman_filter, kalman_smooth, robust_kalman_filter,
    unscented_kalman_filter, KalmanConfig,
};
pub use lombscargle::{
    find_peaks as find_ls_peaks, lombscargle, significance_levels, AutoFreqMethod,
};
pub use median::{
    hybrid_median_filter_2d, median_filter_1d, median_filter_2d, median_filter_color,
    rank_filter_1d, EdgeMode, MedianConfig,
};
pub use nlm::{
    nlm_block_matching_2d, nlm_color_image, nlm_denoise_1d, nlm_denoise_2d, nlm_multiscale_2d,
    NlmConfig,
};
pub use parametric::{
    ar_spectrum, arma_spectrum, estimate_ar, estimate_arma, select_ar_order, ARMethod,
    OrderSelection,
};
pub use peak::{find_peaks, peak_prominences, peak_widths};
pub use reassigned::{reassigned_spectrogram, smoothed_reassigned_spectrogram, ReassignedConfig};
pub use sparse::{
    basis_pursuit, compressed_sensing_recover, cosamp, estimate_rip_constant, fista, iht,
    image_inpainting, ista, lasso, matrix_coherence, measure_sparsity, mp, omp,
    random_sensing_matrix, recover_missing_samples, smooth_l0, sparse_denoise, subspace_pursuit,
    SparseRecoveryConfig, SparseRecoveryMethod, SparseTransform,
};
pub use spectral::{periodogram, spectrogram, stft as spectral_stft, welch};
pub use stft::{closest_stft_dual_window, create_cola_window, ShortTimeFft};
pub use tv::{
    tv_bregman_1d, tv_bregman_2d, tv_denoise_1d, tv_denoise_2d, tv_denoise_color, tv_inpaint,
    TvConfig, TvVariant,
};
pub use waveforms::{chirp, gausspulse, sawtooth, square};
pub use wiener::{
    iterative_wiener_filter, kalman_wiener_filter, psd_wiener_filter, spectral_subtraction,
    wiener_filter, wiener_filter_2d, wiener_filter_freq, wiener_filter_time, WienerConfig,
};
pub use wvd::{cross_wigner_ville, smoothed_pseudo_wigner_ville, wigner_ville, WvdConfig};

// Savitzky-Golay filtering
pub use savgol::{savgol_coeffs, savgol_filter};

// Multitaper spectral analysis
pub use multitaper::{
    adaptive_psd, coherence, dpss, multitaper_filtfilt, multitaper_spectrogram, pmtm,
};

// Wavelet transform functions already re-exported above
pub use dwt2d::{dwt2d_decompose, dwt2d_reconstruct, wavedec2, waverec2, Dwt2dResult};
pub use swt::{iswt, swt, swt_decompose, swt_reconstruct};
pub use swt2d::{iswt2d, swt2d, swt2d_decompose, swt2d_reconstruct, Swt2dResult};
pub use wavelets::{
    complex_gaussian, complex_morlet, cwt, cwt_magnitude, cwt_phase, fbsp, morlet, paul, ricker,
    scale_to_frequency, scalogram, shannon,
};
pub use wpt::{
    get_level_coefficients, reconstruct_from_nodes, wp_decompose, WaveletPacket, WaveletPacketTree,
};
pub use wpt2d::{wpt2d_full, wpt2d_selective, WaveletPacket2D, WaveletPacketTree2D};

// LTI systems functions
pub use lti::system::{c2d, ss, tf, zpk};
pub use lti::{bode, LtiSystem, StateSpace, TransferFunction, ZerosPoleGain};
pub use lti_response::{impulse_response, lsim, step_response};

// Chirp Z-Transform functions
pub use czt::{czt, czt_points};

// Hilbert transform and related functions
pub mod hilbert;
pub use hilbert::{envelope, hilbert, instantaneous_frequency, instantaneous_phase};

// Detrending functions
pub use detrend::{detrend, detrend_axis, detrend_poly};

// Signal denoising functions
pub use denoise::{denoise_wavelet, ThresholdMethod, ThresholdSelect};

// 2D Wavelet image processing functions
pub use dwt2d_image::{compress_image, denoise_image, detect_edges, DenoisingMethod};

// Wavelet visualization utilities
pub use wavelet_vis::{
    arrange_coefficients_2d, arrange_multilevel_coefficients_2d, calculate_energy_1d,
    calculate_energy_2d, calculate_energy_swt2d, colormaps, count_nonzero_coefficients,
    create_coefficient_heatmap, normalize_coefficients, NormalizationStrategy, WaveletCoeffCount,
    WaveletEnergy,
};

// Utilities module re-exports
pub use utilities::spectral::{
    energy_spectral_density, normalized_psd, spectral_centroid, spectral_flatness, spectral_flux,
    spectral_kurtosis, spectral_rolloff, spectral_skewness, spectral_spread,
};

// Signal measurement functions
pub mod measurements;
pub use measurements::{peak_to_peak, peak_to_rms, rms, snr, thd};

// Phase vocoder for time stretching and pitch shifting
pub use phase_vocoder::{phase_vocoder, PhaseVocoderConfig};

// Empirical Mode Decomposition (EMD) for nonlinear and non-stationary signals
pub use emd::{eemd, emd, hilbert_huang_spectrum, EmdConfig, EmdResult};

// Feature extraction for time series analysis
pub use features::{
    activity_recognition_features, extract_features, extract_features_batch, FeatureOptions,
};

// Feature extraction for image analysis
pub use image_features::{
    extract_color_image_features, extract_image_features, ImageFeatureOptions,
};

// Utility functions for signal processing
pub mod utils;

// Re-export spectral utility functions
pub use utilities::spectral::{
    dominant_frequencies, dominant_frequency, spectral_bandwidth, spectral_contrast,
    spectral_crest, spectral_decrease, spectral_slope,
};

// Window functions
pub use window::{
    barthann, bartlett, blackman, blackmanharris, bohman, boxcar, cosine, exponential, flattop,
    get_window, hamming, hann, nuttall, parzen, triang, tukey,
};

// B-spline functions
pub use spline::{
    bspline_basis, bspline_coefficients, bspline_derivative, bspline_evaluate, bspline_filter,
    bspline_smooth, SplineOrder,
};

// Synchrosqueezed wavelet transform functions
pub use sswt::{
    extract_ridges, frequency_bins, log_scales, reconstruct_from_ridge, synchrosqueezed_cwt,
    SynchroCwtConfig, SynchroCwtResult,
};

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
