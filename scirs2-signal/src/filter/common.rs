// Common types and utilities for digital filter design and analysis
//
// This module provides shared types, enums, and utility functions used across
// all filter design and application modules including IIR, FIR, and specialized filters.

use crate::error::{SignalError, SignalResult};
use scirs2_core::numeric::Complex64;
use scirs2_core::numeric::{Float, NumCast};
use std::f64::consts::PI;
use std::fmt::Debug;

#[allow(unused_imports)]
/// Filter type for digital filter design
///
/// Specifies the frequency response characteristics of the filter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FilterType {
    /// Lowpass filter - passes low frequencies, attenuates high frequencies
    Lowpass,
    /// Highpass filter - passes high frequencies, attenuates low frequencies
    Highpass,
    /// Bandpass filter - passes frequencies within a specified band
    Bandpass,
    /// Bandstop filter - attenuates frequencies within a specified band
    Bandstop,
}

impl FilterType {
    /// Parse a string into a filter type (deprecated - use FromStr trait instead)
    #[deprecated(since = "0.1.0", note = "use `parse()` from the FromStr trait instead")]
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> SignalResult<Self> {
        s.parse()
    }
}

impl std::str::FromStr for FilterType {
    type Err = SignalError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "lowpass" | "low" => Ok(FilterType::Lowpass),
            "highpass" | "high" => Ok(FilterType::Highpass),
            "bandpass" | "band" => Ok(FilterType::Bandpass),
            "bandstop" | "stop" => Ok(FilterType::Bandstop),
            _ => Err(SignalError::ValueError(format!(
                "Unknown filter type: {}",
                s
            ))),
        }
    }
}

/// Helper enum to handle different filter type parameter types
///
/// This enum allows filter design functions to accept either a FilterType enum
/// or a string representation of the filter type, providing API flexibility.
#[derive(Debug)]
pub enum FilterTypeParam {
    /// Filter type enum
    Type(FilterType),
    /// Filter type as string
    String(String),
}

impl From<FilterType> for FilterTypeParam {
    fn from(_filtertype: FilterType) -> Self {
        FilterTypeParam::Type(_filtertype)
    }
}

impl From<&str> for FilterTypeParam {
    fn from(s: &str) -> Self {
        FilterTypeParam::String(s.to_string())
    }
}

impl From<String> for FilterTypeParam {
    fn from(s: String) -> Self {
        FilterTypeParam::String(s)
    }
}

/// Comprehensive filter analysis results
///
/// Contains frequency response characteristics, stability information,
/// and other properties useful for filter evaluation and verification.
#[derive(Debug, Clone)]
pub struct FilterAnalysis {
    /// Frequency points (normalized from 0 to 1, where 1 is Nyquist frequency)
    pub frequencies: Vec<f64>,
    /// Magnitude response (linear scale)
    pub magnitude: Vec<f64>,
    /// Magnitude response in dB
    pub magnitude_db: Vec<f64>,
    /// Phase response (radians)
    pub phase: Vec<f64>,
    /// Group delay (samples)
    pub group_delay: Vec<f64>,
    /// Whether the filter is stable
    pub is_stable: bool,
    /// Passband ripple (dB)
    pub passband_ripple: f64,
    /// Stopband attenuation (dB)
    pub stopband_attenuation: f64,
    /// Transition width (normalized frequency)
    pub transition_width: f64,
    /// Filter order
    pub order: usize,
}

impl Default for FilterAnalysis {
    fn default() -> Self {
        Self {
            frequencies: Vec::new(),
            magnitude: Vec::new(),
            magnitude_db: Vec::new(),
            phase: Vec::new(),
            group_delay: Vec::new(),
            is_stable: false,
            passband_ripple: 0.0,
            stopband_attenuation: 0.0,
            transition_width: 0.0,
            order: 0,
        }
    }
}

/// Filter stability information
#[derive(Debug, Clone, PartialEq)]
pub enum FilterStability {
    /// Filter is stable (all poles inside unit circle)
    Stable,
    /// Filter is unstable (poles outside unit circle)
    Unstable { unstable_poles: Vec<Complex64> },
    /// Filter is marginally stable (poles on unit circle)
    MarginallyStable { marginal_poles: Vec<Complex64> },
}

/// Common validation functions for filter parameters
pub mod validation {
    use crate::error::{SignalError, SignalResult};
    use crate::filter::{FilterType, FilterTypeParam};
    use scirs2_core::numeric::Complex64;
    use scirs2_core::numeric::{Float, NumCast};
    use std::fmt::Debug;

    /// Validate filter order
    pub fn validate_order(order: usize) -> SignalResult<()> {
        if order == 0 {
            return Err(SignalError::ValueError(
                "Filter order must be greater than 0".to_string(),
            ));
        }
        Ok(())
    }

    /// Validate normalized cutoff frequency
    pub fn validate_cutoff_frequency<T>(cutoff: T) -> SignalResult<f64>
    where
        T: Float + NumCast + Debug,
    {
        let wn = NumCast::from(cutoff).ok_or_else(|| {
            SignalError::ValueError(format!("Could not convert {:?} to f64", cutoff))
        })?;

        if wn <= 0.0 || wn >= 1.0 {
            return Err(SignalError::ValueError(format!(
                "Cutoff frequency must be between 0 and 1, got {}",
                wn
            )));
        }
        Ok(wn)
    }

    /// Validate bandpass/bandstop frequency pairs
    pub fn validate_band_frequencies(low: f64, high: f64) -> SignalResult<()> {
        if low <= 0.0 || high >= 1.0 || low >= high {
            return Err(SignalError::ValueError(
                "Invalid band frequencies: low must be positive, high must be less than 1, and low < high".to_string(),
            ));
        }
        Ok(())
    }

    /// Convert filter type parameter to FilterType enum
    pub fn convert_filter_type(_filter_typeparam: FilterTypeParam) -> SignalResult<FilterType> {
        match _filter_typeparam {
            FilterTypeParam::Type(t) => Ok(t),
            FilterTypeParam::String(s) => s.parse(),
        }
    }
}

/// Common mathematical operations for filter design
pub mod math {
    use crate::filter::FilterType;
    use scirs2_core::numeric::Complex64;
    use std::f64::consts::PI;

    /// Pre-warp frequency for bilinear transform
    ///
    /// Pre-warps the digital frequency to compensate for the frequency warping
    /// effect of the bilinear transform.
    pub fn prewarp_frequency(_digitalfreq: f64) -> f64 {
        (PI * _digitalfreq / 2.0).tan()
    }

    /// Apply bilinear transform to convert analog pole to digital
    ///
    /// Transforms analog domain pole using bilinear transform: z = (2 + s) / (2 - s)
    pub fn bilinear_pole_transform(_analogpole: Complex64) -> Complex64 {
        (2.0 + _analogpole) / (2.0 - _analogpole)
    }

    /// Apply bilinear transform to convert analog zero to digital
    ///
    /// Transforms analog domain zero using bilinear transform: z = (2 + s) / (2 - s)
    pub fn bilinear_zero_transform(_analogzero: Complex64) -> Complex64 {
        (2.0 + _analogzero) / (2.0 - _analogzero)
    }

    /// Calculate poles for analog Butterworth prototype
    ///
    /// Generates the poles for an analog Butterworth lowpass prototype filter
    /// of the specified order.
    pub fn butterworth_poles(order: usize) -> Vec<Complex64> {
        let mut poles = Vec::with_capacity(order);
        for k in 0..order {
            let angle = PI * (2.0 * k as f64 + order as f64 + 1.0) / (2.0 * order as f64);
            let real = angle.cos();
            let imag = angle.sin();
            poles.push(Complex64::new(real, imag));
        }
        poles
    }

    /// Add zeros for digital filter based on filter type
    ///
    /// Adds appropriate zeros in the digital domain based on the filter type
    /// to complete the bilinear transform process.
    pub fn add_digital_zeros(_filtertype: FilterType, order: usize) -> Vec<Complex64> {
        let mut digital_zeros = Vec::new();

        match _filtertype {
            FilterType::Lowpass => {
                // Lowpass: zeros at z = -1 (Nyquist frequency)
                for _ in 0..order {
                    digital_zeros.push(Complex64::new(-1.0, 0.0));
                }
            }
            FilterType::Highpass => {
                // Highpass: zeros at z = 1 (DC)
                for _ in 0..order {
                    digital_zeros.push(Complex64::new(1.0, 0.0));
                }
            }
            FilterType::Bandpass | FilterType::Bandstop => {
                // Bandpass/Bandstop zeros are added during frequency transformation
                // This is handled in the specific transformation functions
            }
        }

        digital_zeros
    }
}

/// Type aliases for common filter coefficient representations
pub type FilterCoefficients = (Vec<f64>, Vec<f64>);
pub type ZerosPolesGain = (Vec<Complex64>, Vec<Complex64>, f64);
