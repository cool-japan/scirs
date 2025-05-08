//! Signal processing module
//!
//! This module provides implementations of various signal processing algorithms
//! for filtering, convolution, and spectral analysis.
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
pub mod convolve;
pub mod czt;
pub mod denoise;
pub mod detrend;
pub mod dwt;
pub mod dwt2d;
pub mod dwt2d_image;
pub mod filter;
pub mod lti;
pub mod lti_response;
pub mod peak;
pub mod resample;
pub mod savgol;
pub mod spectral;
pub mod swt;
pub mod swt2d;
pub mod waveforms;
pub mod wavelet_vis;
pub mod wavelets;
pub mod wpt;
pub mod wpt2d;

// Re-export commonly used functions
pub use convolve::{convolve, correlate, deconvolve};
pub use filter::{bessel, butter, cheby1, cheby2, ellip, filtfilt, firwin, lfilter};
pub use peak::{find_peaks, peak_prominences, peak_widths};
pub use spectral::{periodogram, spectrogram, stft, welch};
pub use waveforms::{chirp, gausspulse, sawtooth, square};

// Savitzky-Golay filtering
pub use savgol::{savgol_coeffs, savgol_filter};

// Wavelet transform functions
pub use dwt::{dwt_decompose, dwt_reconstruct, wavedec, waverec, Wavelet, WaveletFilters};
pub use dwt2d::{dwt2d_decompose, dwt2d_reconstruct, wavedec2, waverec2, Dwt2dResult};
pub use swt::{iswt, swt, swt_decompose, swt_reconstruct};
pub use swt2d::{iswt2d, swt2d, swt2d_decompose, swt2d_reconstruct, Swt2dResult};
pub use wavelets::{complex_gaussian, complex_morlet, cwt, fbsp, morlet, paul, ricker, shannon};
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

// Signal measurement functions
pub mod measurements;
pub use measurements::{peak_to_peak, peak_to_rms, rms, snr, thd};

// Utility functions for signal processing
pub mod utils;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
