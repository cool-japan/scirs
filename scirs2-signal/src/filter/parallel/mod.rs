//! Parallel filtering operations module
//!
//! This module provides parallel implementations of various digital signal processing
//! filtering operations optimized for multi-core systems. The module is organized
//! into several sub-modules for different categories of filters:
//!
//! - [`types`] - Configuration types and enums
//! - [`core`] - Core filtering operations (filtfilt, lfilter, batch processing)
//! - [`convolution`] - Convolution operations (1D and 2D)
//! - [`filter_banks`] - Filter bank implementations (FIR, IIR, wavelet, polyphase)
//! - [`adaptive`] - Adaptive filtering (LMS, NLMS, block LMS)
//! - [`fft`] - FFT-based filtering operations
//! - [`morphological`] - Morphological filtering operations
//! - [`statistical`] - Statistical filters (median, rank-order, bilateral)
//! - [`specialized`] - Specialized filters (CIC, Wiener, matched filter)
//! - [`utils`] - Utility functions and signal processing helpers

// Module declarations
pub mod adaptive;
pub mod convolution;
pub mod core;
pub mod fft;
pub mod filter_banks;
pub mod morphological;
pub mod specialized;
pub mod statistical;
pub mod types;
pub mod utils;

// Re-export commonly used types for convenience
pub use types::{MorphologicalOperation, ParallelFilterConfig, ParallelFilterType};

// Re-export core filtering functions for backward compatibility
pub use core::{
    filter_direct, parallel_batch_filter, parallel_decimate_filter, parallel_filter_overlap_save,
    parallel_filtfilt, parallel_lfilter,
};

// Re-export convolution functions
pub use convolution::{parallel_convolve, parallel_convolve2d};

// Re-export filter bank functions
pub use filter_banks::{
    parallel_fir_filter_bank, parallel_iir_filter_bank, parallel_polyphase_filter,
    parallel_savgol_filter, parallel_wavelet_filter_bank,
};

// Re-export adaptive filtering functions
pub use adaptive::{
    parallel_adaptive_lms_filter, parallel_adaptive_nlms_filter, parallel_block_lms_filter,
    parallel_fda_lms_filter,
};

// Re-export FFT-based filtering functions
pub use fft::{
    parallel_fft_filter, parallel_fft_filter_design, parallel_overlap_add_convolution,
    parallel_overlap_save_convolution,
};

// Re-export morphological filtering functions
pub use morphological::{
    create_structuring_element, parallel_morphological_filter, parallel_morphological_gradient,
    parallel_morphological_reconstruction, parallel_top_hat_transform,
};

// Re-export statistical filtering functions
pub use statistical::{
    parallel_bilateral_filter, parallel_median_filter, parallel_percentile_filter,
    parallel_rank_order_filter, parallel_trimmed_mean_filter,
};

// Re-export specialized filtering functions
pub use specialized::{
    parallel_cic_filter, parallel_group_delay, parallel_matched_filter, parallel_minimum_phase,
    parallel_wiener_filter,
};

// Re-export utility functions
pub use utils::{
    compute_mse, compute_snr_db, cross_correlation, find_peaks, generate_test_signal,
    normalize_signal_amplitude, normalize_signal_energy, welch_psd_estimate, zero_pad_signal,
};

/// Advanced parallel filtering with configuration options
///
/// Provides a unified interface for various parallel filtering operations
/// with configuration options for optimization.
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `filter_type` - Type of filter to apply
/// * `config` - Configuration for parallel processing
///
/// # Returns
///
/// * Filtered signal
pub fn parallel_filter_advanced(
    signal: &[f64],
    filter_type: &ParallelFilterType,
    config: &ParallelFilterConfig,
) -> crate::error::SignalResult<Vec<f64>> {
    match filter_type {
        ParallelFilterType::FIR { coeffs } => {
            let dummy_denom = vec![1.0];
            let signal_array = scirs2_core::ndarray::Array1::from(signal.to_vec());
            parallel_filter_overlap_save(coeffs, &dummy_denom, &signal_array, config.chunk_size)
                .map(|result| result.to_vec())
        }

        ParallelFilterType::IIR {
            numerator,
            denominator,
        } => {
            let signal_array = scirs2_core::ndarray::Array1::from(signal.to_vec());
            parallel_filter_overlap_save(numerator, denominator, &signal_array, config.chunk_size)
                .map(|result| result.to_vec())
        }

        ParallelFilterType::Adaptive {
            desired,
            filter_length,
            step_size,
        } => {
            let (output, _, _) = parallel_adaptive_lms_filter(
                signal,
                desired,
                *filter_length,
                *step_size,
                config.chunk_size,
            )?;
            Ok(output)
        }

        ParallelFilterType::FFT { impulse_response } => {
            parallel_fft_filter(signal, impulse_response, config.chunk_size)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_parallel_filter_advanced() {
        let signal: Vec<f64> = (0..100)
            .map(|i| (2.0 * PI * i as f64 / 10.0).sin())
            .collect();

        let filter_type = ParallelFilterType::FIR {
            coeffs: vec![0.25, 0.5, 0.25], // Simple smoothing filter
        };

        let config = ParallelFilterConfig::default();
        let result = parallel_filter_advanced(&signal, &filter_type, &config).unwrap();

        assert_eq!(result.len(), signal.len());
    }

    #[test]
    fn test_module_integration() {
        // Test that all major functions are accessible through the module interface
        let signal = vec![1.0, 2.0, 3.0, 2.0, 1.0];

        // Test core filtering
        let b = vec![0.25, 0.5, 0.25];
        let a = vec![1.0];
        let filtered = parallel_filtfilt(&b, &a, &signal, None).unwrap();
        assert_eq!(filtered.len(), signal.len());

        // Test convolution
        let kernel = vec![0.5, 0.5];
        let convolved = parallel_convolve(&signal, &kernel, "same", None).unwrap();
        assert_eq!(convolved.len(), signal.len());

        // Test statistical filtering
        let median_filtered = parallel_median_filter(&signal, 3, None).unwrap();
        assert_eq!(median_filtered.len(), signal.len());

        // Test configuration
        let config = ParallelFilterConfig::default();
        assert!(config.use_simd);
        assert!(config.load_balancing);
        assert_eq!(config.prefetch_factor, 2);
    }

    #[test]
    fn test_filter_banks() {
        let signal: Vec<f64> = (0..200)
            .map(|i| (2.0 * PI * i as f64 / 20.0).sin())
            .collect();

        // Test FIR filter bank
        let filter_bank = vec![vec![0.5, 0.5], vec![1.0, -1.0]];

        let results = parallel_fir_filter_bank(&signal, &filter_bank, None).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].len(), signal.len());
        assert_eq!(results[1].len(), signal.len());
    }

    #[test]
    fn test_adaptive_filtering() {
        let n = 50;
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * i as f64 / 8.0).sin()).collect();
        let desired: Vec<f64> = signal.iter().map(|&x| x * 0.7).collect();

        let (output, coeffs, _error) =
            parallel_adaptive_lms_filter(&signal, &desired, 8, 0.05, None).unwrap();

        assert_eq!(output.len(), n);
        assert_eq!(coeffs.len(), 8);

        // Check that filter adapted
        let coeff_energy: f64 = coeffs.iter().map(|&x| x * x).sum();
        assert!(coeff_energy > 0.0);
    }

    #[test]
    fn test_morphological_operations() {
        let signal = vec![0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0];
        let structuring_element = vec![1.0, 1.0, 1.0];

        let eroded = parallel_morphological_filter(
            &signal,
            &structuring_element,
            MorphologicalOperation::Erosion,
            None,
        )
        .unwrap();

        assert_eq!(eroded.len(), signal.len());

        let dilated = parallel_morphological_filter(
            &signal,
            &structuring_element,
            MorphologicalOperation::Dilation,
            None,
        )
        .unwrap();

        assert_eq!(dilated.len(), signal.len());
    }

    #[test]
    fn test_utility_functions() {
        // Test signal generation
        let frequencies = vec![10.0, 20.0];
        let amplitudes = vec![1.0, 0.5];
        let test_signal =
            generate_test_signal(100, &frequencies, &amplitudes, 0.1, 1000.0).unwrap();
        assert_eq!(test_signal.len(), 100);

        // Test normalization
        let signal = vec![2.0, 4.0, 6.0];
        let normalized = normalize_signal_amplitude(&signal).unwrap();
        let max_val = normalized.iter().map(|&x| x.abs()).fold(0.0f64, f64::max);
        assert!((max_val - 1.0).abs() < 1e-10);

        // Test SNR computation
        let clean = vec![1.0, 2.0, 3.0];
        let noisy = vec![1.1, 2.1, 3.1];
        let snr = compute_snr_db(&clean, &noisy).unwrap();
        assert!(snr > 0.0);
    }
}
