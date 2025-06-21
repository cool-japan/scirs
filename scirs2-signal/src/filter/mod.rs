//! Digital filter design and analysis
//!
//! This module provides comprehensive digital filter design capabilities including
//! IIR and FIR filter design, filter analysis, specialized filters, and filter
//! transformation functions. The module is organized into focused submodules:
//!
//! - [`common`] - Common types, enums, and utilities shared across all filter modules
//! - [`iir`] - IIR (Infinite Impulse Response) filter designs (Butterworth, Chebyshev, etc.)
//! - [`fir`] - FIR (Finite Impulse Response) filter designs (window method, Parks-McClellan)
//! - [`application`] - Filter application functions (filtfilt, lfilter, matched filtering)
//! - [`analysis`] - Filter analysis and characterization functions
//! - [`transform`] - Filter transformation functions (bilinear transform, zpk conversions)
//! - [`specialized`] - Specialized filter designs (notch, comb, allpass, etc.)
//!
//! # Quick Start
//!
//! ## IIR Filter Design
//!
//! ```rust
//! use scirs2_signal::filter::{butter, FilterType};
//!
//! // Design a 4th order Butterworth lowpass filter
//! let (b, a) = butter(4, 0.3, FilterType::Lowpass).unwrap();
//!
//! // Or using string parameter
//! let (b, a) = butter(4, 0.3, "lowpass").unwrap();
//! ```
//!
//! ## FIR Filter Design
//!
//! ```rust
//! use scirs2_signal::filter::firwin;
//!
//! // Design a 65-tap FIR lowpass filter with Hamming window
//! let h = firwin(65, 0.3, "hamming", true).unwrap();
//! ```
//!
//! ## Filter Application
//!
//! ```rust
//! use scirs2_signal::filter::{butter, filtfilt};
//!
//! // Design filter and apply with zero phase delay
//! let (b, a) = butter(4, 0.2, "lowpass").unwrap();
//! let signal = vec![1.0, 2.0, 3.0, 2.0, 1.0];
//! let filtered = filtfilt(&b, &a, &signal).unwrap();
//! ```
//!
//! ## Filter Analysis
//!
//! ```rust
//! use scirs2_signal::filter::{butter, analyze_filter};
//!
//! // Analyze filter characteristics
//! let (b, a) = butter(4, 0.2, "lowpass").unwrap();
//! let analysis = analyze_filter(&b, &a, Some(512)).unwrap();
//! println!("3dB cutoff: {:.3}", analysis.cutoff_3db);
//! ```

// Re-export all public modules
pub mod analysis;
pub mod application;
pub mod common;
pub mod fir;
pub mod iir;
pub mod specialized;
pub mod transform;

// Re-export commonly used types for convenience
pub use common::{
    math::{add_digital_zeros, bilinear_pole_transform, butterworth_poles, prewarp_frequency},
    validation::{
        convert_filter_type, validate_band_frequencies, validate_cutoff_frequency, validate_order,
    },
    FilterAnalysis, FilterCoefficients, FilterStability, FilterType, FilterTypeParam,
};

// Re-export all IIR filter design functions
pub use iir::{bessel, butter, butter_bandpass_bandstop, cheby1, cheby2, ellip};

// Re-export all FIR filter design functions
pub use fir::{firwin, remez};

// Re-export filter application functions
pub use application::{
    filtfilt, group_delay, lfilter, matched_filter, matched_filter_detect, minimum_phase,
};

// Re-export filter analysis functions
pub use analysis::{
    analyze_filter, check_filter_stability, compute_q_factor, find_poles_zeros, frequency_response,
};

// Re-export filter transformation functions
pub use transform::{
    bilinear_transform, lp_to_bp_transform, lp_to_bs_transform, lp_to_hp_transform,
    lp_to_lp_transform, normalize_coefficients, tf_to_zpk, zpk_to_tf,
};

// Re-export specialized filter functions
pub use specialized::{
    allpass_filter, allpass_second_order, comb_filter, dc_blocker, differentiator_filter,
    fractional_delay_filter, hilbert_filter, integrator_filter, notch_filter, peak_filter,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_butter_filter_basic() {
        // Test basic Butterworth filter design
        let result = butter(4, 0.3, FilterType::Lowpass);
        assert!(result.is_ok());

        let (_b, a) = result.unwrap();
        assert_eq!(a.len(), 5); // 4th order = 5 coefficients
        assert_eq!(a[0], 1.0); // Normalized denominator
    }

    #[test]
    fn test_firwin_filter_basic() {
        // Test basic FIR window method design
        let result = firwin(21, 0.3, "hamming", true);
        assert!(result.is_ok());

        let h = result.unwrap();
        assert_eq!(h.len(), 21);

        // Check symmetry (linear phase)
        for i in 0..h.len() / 2 {
            assert!((h[i] - h[h.len() - 1 - i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_filter_analysis() {
        // Test filter analysis functionality
        let (b, a) = butter(4, 0.2, "lowpass").unwrap();
        let result = analyze_filter(&b, &a, Some(256));
        assert!(result.is_ok());

        let analysis = result.unwrap();
        assert_eq!(analysis.frequencies.len(), 256);
        assert_eq!(analysis.magnitude.len(), 256);
        assert!(analysis.cutoff_3db > 0.0 && analysis.cutoff_3db < 1.0);
    }

    #[test]
    fn test_filtfilt_application() {
        // Test zero-phase filtering
        let (b, a) = butter(2, 0.3, "lowpass").unwrap();
        let signal = vec![1.0, 2.0, 3.0, 2.0, 1.0];
        let result = filtfilt(&b, &a, &signal);
        assert!(result.is_ok());

        let filtered = result.unwrap();
        assert_eq!(filtered.len(), signal.len());
    }

    #[test]
    fn test_notch_filter() {
        // Test specialized notch filter
        let result = notch_filter(0.25, 35.0);
        assert!(result.is_ok());

        let (b, a) = result.unwrap();
        assert!(b.len() >= 2);
        assert!(a.len() >= 2);
        assert_eq!(a[0], 1.0); // Normalized
    }

    #[test]
    fn test_stability_check() {
        // Test filter stability analysis
        let (_b, a) = butter(4, 0.2, "lowpass").unwrap();
        let result = check_filter_stability(&a);
        assert!(result.is_ok());

        let stability = result.unwrap();
        assert!(stability.is_stable);
        assert!(stability.stability_margin > 0.0);
    }

    #[test]
    fn test_zpk_transform() {
        // Test zeros-poles-gain transformation
        use num_complex::Complex64;

        let zeros = vec![Complex64::new(-1.0, 0.0)];
        let poles = vec![Complex64::new(0.5, 0.0)];
        let gain = 1.0;

        let result = zpk_to_tf(&zeros, &poles, gain);
        assert!(result.is_ok());

        let (_b, a) = result.unwrap();
        assert_eq!(a[0], 1.0); // Normalized
    }

    #[test]
    fn test_filter_type_conversion() {
        // Test filter type parameter conversion
        let filter_type = convert_filter_type(FilterTypeParam::String("lowpass".to_string()));
        assert!(filter_type.is_ok());
        assert_eq!(filter_type.unwrap(), FilterType::Lowpass);

        let filter_type = convert_filter_type(FilterTypeParam::Type(FilterType::Highpass));
        assert!(filter_type.is_ok());
        assert_eq!(filter_type.unwrap(), FilterType::Highpass);
    }

    #[test]
    fn test_validation_functions() {
        // Test parameter validation
        assert!(validate_order(0).is_err());
        assert!(validate_order(4).is_ok());

        assert!(validate_cutoff_frequency(0.5).is_ok());
        assert!(validate_cutoff_frequency(1.5).is_err());
        assert!(validate_cutoff_frequency(-0.1).is_err());

        assert!(validate_band_frequencies(0.1, 0.4).is_ok());
        assert!(validate_band_frequencies(0.4, 0.1).is_err());
        assert!(validate_band_frequencies(-0.1, 0.4).is_err());
    }
}
