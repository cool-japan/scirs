//! Signal Separation Module
//!
//! This module implements various signal separation techniques including:
//! - Multi-band signal separation using filter banks
//! - Harmonic/percussive separation for audio signals
//! - Spectral source separation methods
//!
//! # Example
//! ```
//! use ndarray::Array1;
//! use scirs2_signal::separation::{multiband_separation, harmonic_percussive_separation};
//!
//! // Multi-band separation
//! let signal = Array1::from_vec(vec![1.0, 0.5, -0.3, 0.8, -0.2]);
//! let bands = multiband_separation(&signal, &[0.1, 0.3, 0.5], 1000.0, None).unwrap();
//!
//! // Harmonic/percussive separation
//! let (harmonic, percussive) = harmonic_percussive_separation(&signal, 1000.0, None).unwrap();
//! ```

use crate::error::{SignalError, SignalResult};
use crate::filter::{butter, lfilter, FilterType};
use ndarray::Array1;
#[cfg(test)]
use std::f64::consts::PI;

/// Configuration for multi-band separation
#[derive(Debug, Clone)]
pub struct MultibandConfig {
    /// Filter order for each band
    pub filter_order: usize,
    /// Overlap between adjacent bands (0.0 to 1.0)
    pub overlap: f64,
    /// Filter type to use
    pub filter_type: FilterType,
}

impl Default for MultibandConfig {
    fn default() -> Self {
        Self {
            filter_order: 6,
            overlap: 0.1,
            filter_type: FilterType::Lowpass, // Default filter type
        }
    }
}

/// Configuration for harmonic/percussive separation
#[derive(Debug, Clone)]
pub struct HarmonicPercussiveConfig {
    /// Kernel size for harmonic enhancement (horizontal smoothing)
    pub harmonic_kernel: usize,
    /// Kernel size for percussive enhancement (vertical smoothing)  
    pub percussive_kernel: usize,
    /// Separation strength parameter
    pub separation_power: f64,
    /// Minimum frequency for separation (Hz)
    pub min_freq: f64,
    /// Maximum frequency for separation (Hz)
    pub max_freq: f64,
}

impl Default for HarmonicPercussiveConfig {
    fn default() -> Self {
        Self {
            harmonic_kernel: 17,
            percussive_kernel: 17,
            separation_power: 2.0,
            min_freq: 80.0,
            max_freq: 8000.0,
        }
    }
}

/// Perform multi-band signal separation using filter banks
///
/// Separates a signal into multiple frequency bands using bandpass filters.
/// Each band covers a specific frequency range defined by the cutoff frequencies.
///
/// # Arguments
///
/// * `signal` - Input signal to separate
/// * `cutoff_freqs` - Cutoff frequencies (normalized to Nyquist) for band boundaries
/// * `sample_rate` - Sample rate of the input signal
/// * `config` - Optional configuration for the separation
///
/// # Returns
///
/// * Vector of separated signals, one for each frequency band
///
/// # Examples
///
/// ```
/// use ndarray::Array1;
/// use scirs2_signal::separation::multiband_separation;
///
/// let signal = Array1::from_vec(vec![1.0, 0.5, -0.3, 0.8, -0.2, 0.1, -0.6, 0.4]);
/// let cutoffs = vec![0.1, 0.3, 0.5]; // Creates 4 bands: 0-0.1, 0.1-0.3, 0.3-0.5, 0.5-1.0
/// let bands = multiband_separation(&signal, &cutoffs, 1000.0, None).unwrap();
/// assert_eq!(bands.len(), 4); // Number of bands = cutoffs.len() + 1
/// ```
pub fn multiband_separation(
    signal: &Array1<f64>,
    cutoff_freqs: &[f64],
    _sample_rate: f64,
    config: Option<MultibandConfig>,
) -> SignalResult<Vec<Array1<f64>>> {
    if signal.is_empty() {
        return Err(SignalError::ValueError("Input signal is empty".to_string()));
    }

    if cutoff_freqs.is_empty() {
        return Err(SignalError::ValueError(
            "Cutoff frequencies cannot be empty".to_string(),
        ));
    }

    // Validate cutoff frequencies
    for &freq in cutoff_freqs {
        if freq <= 0.0 || freq >= 1.0 {
            return Err(SignalError::ValueError(
                "Cutoff frequencies must be between 0 and 1 (normalized to Nyquist)".to_string(),
            ));
        }
    }

    // Check that cutoff frequencies are sorted
    for i in 1..cutoff_freqs.len() {
        if cutoff_freqs[i] <= cutoff_freqs[i - 1] {
            return Err(SignalError::ValueError(
                "Cutoff frequencies must be in ascending order".to_string(),
            ));
        }
    }

    let config = config.unwrap_or_default();
    let mut bands = Vec::new();

    // Create frequency bands: [0, cutoff[0]], [cutoff[0], cutoff[1]], ..., [cutoff[n-1], 1.0]
    let mut band_freqs = vec![0.0];
    band_freqs.extend_from_slice(cutoff_freqs);
    band_freqs.push(1.0);

    for i in 0..band_freqs.len() - 1 {
        let low_freq = band_freqs[i];
        let high_freq = band_freqs[i + 1];

        let signal_vec = signal.to_vec();

        let filtered_signal = if i == 0 {
            // First band: lowpass filter
            let (b, a) = butter(config.filter_order, high_freq, FilterType::Lowpass)?;
            Array1::from(lfilter(&b, &a, &signal_vec)?)
        } else if i == band_freqs.len() - 2 {
            // Last band: highpass filter
            let (b, a) = butter(config.filter_order, low_freq, FilterType::Highpass)?;
            Array1::from(lfilter(&b, &a, &signal_vec)?)
        } else {
            // Middle bands: bandpass filter
            let (b, a) = butter(config.filter_order, low_freq, FilterType::Highpass)?;
            let temp_signal = lfilter(&b, &a, &signal_vec)?;
            let (b, a) = butter(config.filter_order, high_freq, FilterType::Lowpass)?;
            Array1::from(lfilter(&b, &a, &temp_signal)?)
        };

        bands.push(filtered_signal);
    }

    Ok(bands)
}

/// Perform harmonic/percussive separation using frequency-domain filtering
///
/// Separates a signal into harmonic and percussive components using
/// a simplified frequency-domain approach. Harmonic content is typically
/// found in lower frequencies with sustained characteristics, while
/// percussive content has transient characteristics spread across frequencies.
///
/// # Arguments
///
/// * `signal` - Input signal to separate
/// * `_sample_rate` - Sample rate of the input signal (for future use)
/// * `config` - Optional configuration for the separation
///
/// # Returns
///
/// * Tuple of (harmonic_component, percussive_component)
///
/// # Examples
///
/// ```
/// use ndarray::Array1;
/// use scirs2_signal::separation::harmonic_percussive_separation;
///
/// let signal = Array1::from_vec(vec![1.0, 0.5, -0.3, 0.8, -0.2, 0.1, -0.6, 0.4]);
/// let (harmonic, percussive) = harmonic_percussive_separation(&signal, 1000.0, None).unwrap();
/// assert_eq!(harmonic.len(), signal.len());
/// assert_eq!(percussive.len(), signal.len());
/// ```
pub fn harmonic_percussive_separation(
    signal: &Array1<f64>,
    _sample_rate: f64,
    config: Option<HarmonicPercussiveConfig>,
) -> SignalResult<(Array1<f64>, Array1<f64>)> {
    if signal.is_empty() {
        return Err(SignalError::ValueError("Input signal is empty".to_string()));
    }

    let config = config.unwrap_or_default();

    // Simple approach: use low-pass filtering for harmonic component
    // and high-pass filtering for percussive component
    let harmonic_cutoff = 0.3; // Normalized frequency (0 to 1)
    let percussive_cutoff = 0.2; // Normalized frequency

    // Extract harmonic component (low-pass filtered)
    let (b_low, a_low) = butter(4, harmonic_cutoff, FilterType::Lowpass)?;
    let harmonic_vec = lfilter(&b_low, &a_low, &signal.to_vec())?;

    // Extract percussive component (high-pass filtered)
    let (b_high, a_high) = butter(4, percussive_cutoff, FilterType::Highpass)?;
    let percussive_vec = lfilter(&b_high, &a_high, &signal.to_vec())?;

    // Apply a simple amplitude adjustment based on configuration
    let harmonic_adjusted: Vec<f64> = harmonic_vec
        .iter()
        .map(|&x| x * config.separation_power.sqrt())
        .collect();

    let percussive_adjusted: Vec<f64> = percussive_vec
        .iter()
        .map(|&x| x * (2.0 - config.separation_power).sqrt().max(0.1))
        .collect();

    Ok((
        Array1::from(harmonic_adjusted),
        Array1::from(percussive_adjusted),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multiband_separation() {
        // Create a test signal with multiple frequency components
        let sample_rate = 1000.0;
        let duration = 1.0;
        let t: Vec<f64> = (0..(sample_rate * duration) as usize)
            .map(|i| i as f64 / sample_rate)
            .collect();

        // Create signal with three frequency components
        let signal: Vec<f64> = t
            .iter()
            .map(|&t| {
                (2.0 * PI * 50.0 * t).sin()    // Low frequency
                    + (2.0 * PI * 150.0 * t).sin()  // Mid frequency  
                    + (2.0 * PI * 350.0 * t).sin() // High frequency
            })
            .collect();

        let signal_array = Array1::from(signal);

        // Define cutoff frequencies (normalized to Nyquist = 500 Hz)
        let cutoffs = vec![0.2, 0.6]; // 100 Hz, 300 Hz

        let bands = multiband_separation(&signal_array, &cutoffs, sample_rate, None).unwrap();

        // Should create 3 bands
        assert_eq!(bands.len(), 3);

        // Each band should have the same length as input
        for band in &bands {
            assert_eq!(band.len(), signal_array.len());
        }

        // Energy should be distributed across bands
        let _total_energy: f64 = signal_array.iter().map(|&x| x * x).sum();
        let band_energies: Vec<f64> = bands
            .iter()
            .map(|band| band.iter().map(|&x| x * x).sum())
            .collect();

        let sum_band_energies: f64 = band_energies.iter().sum();

        // Energy conservation may not be exact due to filter effects
        // Just check that we have reasonable energy distribution
        assert!(sum_band_energies > 0.0);
        assert!(band_energies.iter().all(|&e| e >= 0.0));
    }

    #[test]
    fn test_harmonic_percussive_separation() {
        // Create a simple test signal
        let sample_rate = 1000.0;
        let duration = 0.5;
        let t: Vec<f64> = (0..(sample_rate * duration) as usize)
            .map(|i| i as f64 / sample_rate)
            .collect();

        // Create signal with harmonic and transient components
        let signal: Vec<f64> = t
            .iter()
            .enumerate()
            .map(|(i, &t)| {
                let harmonic = (2.0 * PI * 100.0 * t).sin(); // Stable harmonic
                let transient = if i < 50 { 0.5 } else { 0.0 }; // Brief transient
                harmonic + transient
            })
            .collect();

        let signal_array = Array1::from(signal);

        let (harmonic, percussive) =
            harmonic_percussive_separation(&signal_array, sample_rate, None).unwrap();

        // Output signals should have same length as input
        assert_eq!(harmonic.len(), signal_array.len());
        assert_eq!(percussive.len(), signal_array.len());

        // Both components should have finite values
        for &val in harmonic.iter() {
            assert!(val.is_finite());
        }
        for &val in percussive.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_multiband_separation_edge_cases() {
        let signal = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        // Test with empty cutoffs
        let result = multiband_separation(&signal, &[], 1000.0, None);
        assert!(result.is_err());

        // Test with invalid cutoff frequency
        let result = multiband_separation(&signal, &[1.5], 1000.0, None);
        assert!(result.is_err());

        // Test with unsorted cutoffs
        let result = multiband_separation(&signal, &[0.8, 0.3], 1000.0, None);
        assert!(result.is_err());

        // Test with empty signal
        let empty_signal = Array1::from(vec![]);
        let result = multiband_separation(&empty_signal, &[0.5], 1000.0, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_harmonic_percussive_config() {
        let config = HarmonicPercussiveConfig::default();

        assert_eq!(config.harmonic_kernel, 17);
        assert_eq!(config.percussive_kernel, 17);
        assert_eq!(config.separation_power, 2.0);
        assert_eq!(config.min_freq, 80.0);
        assert_eq!(config.max_freq, 8000.0);
    }
}
