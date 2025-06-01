//! Utilities for spectral analysis
//!
//! This module provides utility functions for spectral analysis,
//! including spectral descriptors, normalized spectral representations,
//! and advanced spectral processing techniques for signal characterization.

use crate::error::{SignalError, SignalResult};
use ndarray::Array1;
use num_traits::{Float, NumCast};
use std::fmt::Debug;

/// Calculate the energy spectral density (ESD) of a signal.
///
/// The energy spectral density describes how the energy of a signal
/// is distributed across frequency components.
///
/// # Arguments
///
/// * `psd` - Power spectral density
/// * `fs` - Sample rate in Hz
///
/// # Returns
///
/// * Energy spectral density array
///
/// # Examples
///
/// ```
/// use scirs2_signal::spectral::periodogram;
/// use scirs2_signal::utilities::spectral::energy_spectral_density;
///
/// let signal = vec![1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 0.0];
/// let fs = 8.0; // Sample rate in Hz
///
/// let (psd, freqs) = periodogram(&signal, Some(fs), Some("hamming"), Some(16), None, None).unwrap();
/// let esd = energy_spectral_density(&psd, fs).unwrap();
///
/// // ESD is proportional to PSD but scaled by the sample interval
/// ```
pub fn energy_spectral_density<T>(psd: &[T], fs: f64) -> SignalResult<Vec<f64>>
where
    T: Float + NumCast + Debug,
{
    if psd.is_empty() {
        return Err(SignalError::ValueError("PSD array is empty".to_string()));
    }

    if fs <= 0.0 {
        return Err(SignalError::ValueError(
            "Sample rate must be positive".to_string(),
        ));
    }

    // Convert to f64 for internal processing
    let psd_f64: Vec<f64> = psd
        .iter()
        .map(|&val| {
            num_traits::cast::cast::<T, f64>(val).ok_or_else(|| {
                SignalError::ValueError(format!("Could not convert {:?} to f64", val))
            })
        })
        .collect::<SignalResult<Vec<_>>>()?;

    // Calculate ESD by scaling PSD by the sample interval
    let dt = 1.0 / fs;
    let esd = psd_f64.iter().map(|&p| p * dt).collect();

    Ok(esd)
}

/// Normalize a power spectral density (PSD) to have unit area.
///
/// # Arguments
///
/// * `psd` - Power spectral density
///
/// # Returns
///
/// * Normalized power spectral density
///
/// # Examples
///
/// ```
/// use scirs2_signal::spectral::periodogram;
/// use scirs2_signal::utilities::spectral::normalized_psd;
///
/// let signal = vec![1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 0.0];
/// let fs = 8.0; // Sample rate in Hz
///
/// let (psd, freqs) = periodogram(&signal, Some(fs), Some("hamming"), Some(16), None, None).unwrap();
/// let norm_psd = normalized_psd(&psd).unwrap();
///
/// // Sum of normalized PSD should be approximately 1.0
/// let sum: f64 = norm_psd.iter().sum();
/// assert!((sum - 1.0).abs() < 1e-10);
/// ```
pub fn normalized_psd<T>(psd: &[T]) -> SignalResult<Vec<f64>>
where
    T: Float + NumCast + Debug,
{
    if psd.is_empty() {
        return Err(SignalError::ValueError("PSD array is empty".to_string()));
    }

    // Convert to f64 for internal processing
    let psd_f64: Vec<f64> = psd
        .iter()
        .map(|&val| {
            num_traits::cast::cast::<T, f64>(val).ok_or_else(|| {
                SignalError::ValueError(format!("Could not convert {:?} to f64", val))
            })
        })
        .collect::<SignalResult<Vec<_>>>()?;

    // Calculate sum of PSD values
    let sum: f64 = psd_f64.iter().sum();

    if sum <= 0.0 {
        return Err(SignalError::ValueError(
            "Sum of PSD values must be positive for normalization".to_string(),
        ));
    }

    // Normalize PSD
    let normalized = psd_f64.iter().map(|&p| p / sum).collect();

    Ok(normalized)
}

/// Calculate the spectral centroid of a signal.
///
/// The spectral centroid is the weighted mean of the frequencies present in the signal,
/// with their magnitudes as the weights. It represents the "center of mass" of the spectrum.
///
/// # Arguments
///
/// * `psd` - Power spectral density
/// * `freqs` - Frequency values corresponding to the PSD
///
/// # Returns
///
/// * Spectral centroid in the same units as freqs
///
/// # Examples
///
/// ```
/// use scirs2_signal::spectral::periodogram;
/// use scirs2_signal::utilities::spectral::spectral_centroid;
///
/// let signal = vec![1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 0.0];
/// let fs = 8.0; // Sample rate in Hz
///
/// let (psd, freqs) = periodogram(&signal, Some(fs), Some("hamming"), Some(16), None, None).unwrap();
/// let centroid = spectral_centroid(&psd, &freqs).unwrap();
///
/// // Basic sanity check - centroid should be finite
/// assert!(centroid.is_finite());
/// ```
pub fn spectral_centroid<T, U>(psd: &[T], freqs: &[U]) -> SignalResult<f64>
where
    T: Float + NumCast + Debug,
    U: Float + NumCast + Debug,
{
    if psd.is_empty() || freqs.is_empty() {
        return Err(SignalError::ValueError(
            "PSD or frequency array is empty".to_string(),
        ));
    }

    if psd.len() != freqs.len() {
        return Err(SignalError::ValueError(
            "PSD and frequency arrays must have the same length".to_string(),
        ));
    }

    // Convert to f64 for internal processing
    let psd_f64: Vec<f64> = psd
        .iter()
        .map(|&val| {
            num_traits::cast::cast::<T, f64>(val).ok_or_else(|| {
                SignalError::ValueError(format!("Could not convert {:?} to f64", val))
            })
        })
        .collect::<SignalResult<Vec<_>>>()?;

    let freqs_f64: Vec<f64> = freqs
        .iter()
        .map(|&val| {
            num_traits::cast::cast::<U, f64>(val).ok_or_else(|| {
                SignalError::ValueError(format!("Could not convert {:?} to f64", val))
            })
        })
        .collect::<SignalResult<Vec<_>>>()?;

    // Calculate sum of PSD values
    let sum: f64 = psd_f64.iter().sum();

    if sum <= 0.0 {
        return Err(SignalError::ValueError(
            "Sum of PSD values must be positive for centroid calculation".to_string(),
        ));
    }

    // Calculate weighted sum of frequencies
    let weighted_sum: f64 = psd_f64
        .iter()
        .zip(freqs_f64.iter())
        .map(|(&p, &f)| p * f)
        .sum();

    // Calculate centroid
    let centroid = weighted_sum / sum;

    Ok(centroid)
}

/// Calculate the spectral spread of a signal.
///
/// The spectral spread is the standard deviation of the spectrum around its centroid.
/// It describes the average deviation from the spectral centroid.
///
/// # Arguments
///
/// * `psd` - Power spectral density
/// * `freqs` - Frequency values corresponding to the PSD
/// * `centroid` - Spectral centroid (if None, it will be calculated)
///
/// # Returns
///
/// * Spectral spread in the same units as freqs
///
/// # Examples
///
/// ```
/// use scirs2_signal::spectral::periodogram;
/// use scirs2_signal::utilities::spectral::spectral_spread;
///
/// let signal = vec![1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 0.0];
/// let fs = 8.0; // Sample rate in Hz
///
/// let (psd, freqs) = periodogram(&signal, Some(fs), Some("hamming"), Some(16), None, None).unwrap();
/// let spread = spectral_spread(&psd, &freqs, None).unwrap();
///
/// // Spread should be non-negative
/// assert!(spread >= 0.0);
/// ```
pub fn spectral_spread<T, U>(psd: &[T], freqs: &[U], centroid: Option<f64>) -> SignalResult<f64>
where
    T: Float + NumCast + Debug,
    U: Float + NumCast + Debug,
{
    if psd.is_empty() || freqs.is_empty() {
        return Err(SignalError::ValueError(
            "PSD or frequency array is empty".to_string(),
        ));
    }

    if psd.len() != freqs.len() {
        return Err(SignalError::ValueError(
            "PSD and frequency arrays must have the same length".to_string(),
        ));
    }

    // Convert to f64 for internal processing
    let psd_f64: Vec<f64> = psd
        .iter()
        .map(|&val| {
            num_traits::cast::cast::<T, f64>(val).ok_or_else(|| {
                SignalError::ValueError(format!("Could not convert {:?} to f64", val))
            })
        })
        .collect::<SignalResult<Vec<_>>>()?;

    let freqs_f64: Vec<f64> = freqs
        .iter()
        .map(|&val| {
            num_traits::cast::cast::<U, f64>(val).ok_or_else(|| {
                SignalError::ValueError(format!("Could not convert {:?} to f64", val))
            })
        })
        .collect::<SignalResult<Vec<_>>>()?;

    // Calculate sum of PSD values
    let sum: f64 = psd_f64.iter().sum();

    if sum <= 0.0 {
        return Err(SignalError::ValueError(
            "Sum of PSD values must be positive for spread calculation".to_string(),
        ));
    }

    // Calculate or use provided centroid
    let centroid_val = match centroid {
        Some(c) => c,
        None => spectral_centroid(psd, freqs)?,
    };

    // Calculate weighted sum of squared differences from centroid
    let weighted_sum_sq_diff: f64 = psd_f64
        .iter()
        .zip(freqs_f64.iter())
        .map(|(&p, &f)| p * (f - centroid_val).powi(2))
        .sum();

    // Calculate spread (standard deviation)
    let spread = (weighted_sum_sq_diff / sum).sqrt();

    Ok(spread)
}

/// Calculate the spectral skewness of a signal.
///
/// The spectral skewness measures the asymmetry of the spectrum around its centroid.
/// Positive skewness indicates more energy in frequencies above the centroid,
/// while negative skewness indicates more energy in frequencies below the centroid.
///
/// # Arguments
///
/// * `psd` - Power spectral density
/// * `freqs` - Frequency values corresponding to the PSD
/// * `centroid` - Spectral centroid (if None, it will be calculated)
/// * `spread` - Spectral spread (if None, it will be calculated)
///
/// # Returns
///
/// * Spectral skewness (dimensionless)
///
/// # Examples
///
/// ```
/// use scirs2_signal::spectral::periodogram;
/// use scirs2_signal::utilities::spectral::spectral_skewness;
///
/// let signal = vec![1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 0.0];
/// let fs = 8.0; // Sample rate in Hz
///
/// let (psd, freqs) = periodogram(&signal, Some(fs), Some("hamming"), Some(16), None, None).unwrap();
/// let skewness = spectral_skewness(&psd, &freqs, None, None).unwrap();
/// ```
pub fn spectral_skewness<T, U>(
    psd: &[T],
    freqs: &[U],
    centroid: Option<f64>,
    spread: Option<f64>,
) -> SignalResult<f64>
where
    T: Float + NumCast + Debug,
    U: Float + NumCast + Debug,
{
    if psd.is_empty() || freqs.is_empty() {
        return Err(SignalError::ValueError(
            "PSD or frequency array is empty".to_string(),
        ));
    }

    if psd.len() != freqs.len() {
        return Err(SignalError::ValueError(
            "PSD and frequency arrays must have the same length".to_string(),
        ));
    }

    // Convert to f64 for internal processing
    let psd_f64: Vec<f64> = psd
        .iter()
        .map(|&val| {
            num_traits::cast::cast::<T, f64>(val).ok_or_else(|| {
                SignalError::ValueError(format!("Could not convert {:?} to f64", val))
            })
        })
        .collect::<SignalResult<Vec<_>>>()?;

    let freqs_f64: Vec<f64> = freqs
        .iter()
        .map(|&val| {
            num_traits::cast::cast::<U, f64>(val).ok_or_else(|| {
                SignalError::ValueError(format!("Could not convert {:?} to f64", val))
            })
        })
        .collect::<SignalResult<Vec<_>>>()?;

    // Calculate sum of PSD values
    let sum: f64 = psd_f64.iter().sum();

    if sum <= 0.0 {
        return Err(SignalError::ValueError(
            "Sum of PSD values must be positive for skewness calculation".to_string(),
        ));
    }

    // Calculate or use provided centroid
    let centroid_val = match centroid {
        Some(c) => c,
        None => spectral_centroid(psd, freqs)?,
    };

    // Calculate or use provided spread
    let spread_val = match spread {
        Some(s) => s,
        None => spectral_spread(psd, freqs, Some(centroid_val))?,
    };

    if spread_val <= 0.0 {
        return Err(SignalError::ValueError(
            "Spectral spread must be positive for skewness calculation".to_string(),
        ));
    }

    // Calculate weighted sum of cubed differences from centroid
    let weighted_sum_cubed_diff: f64 = psd_f64
        .iter()
        .zip(freqs_f64.iter())
        .map(|(&p, &f)| p * (f - centroid_val).powi(3))
        .sum();

    // Calculate skewness
    let skewness = weighted_sum_cubed_diff / (sum * spread_val.powi(3));

    Ok(skewness)
}

/// Calculate the spectral kurtosis of a signal.
///
/// The spectral kurtosis measures the "peakedness" of the spectrum.
/// Higher kurtosis indicates more of the energy is concentrated in specific frequency bands,
/// while lower kurtosis indicates a more uniform distribution.
///
/// # Arguments
///
/// * `psd` - Power spectral density
/// * `freqs` - Frequency values corresponding to the PSD
/// * `centroid` - Spectral centroid (if None, it will be calculated)
/// * `spread` - Spectral spread (if None, it will be calculated)
///
/// # Returns
///
/// * Spectral kurtosis (dimensionless)
///
/// # Examples
///
/// ```
/// use scirs2_signal::spectral::periodogram;
/// use scirs2_signal::utilities::spectral::spectral_kurtosis;
///
/// let signal = vec![1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 0.0];
/// let fs = 8.0; // Sample rate in Hz
///
/// let (psd, freqs) = periodogram(&signal, Some(fs), Some("hamming"), Some(16), None, None).unwrap();
/// let kurtosis = spectral_kurtosis(&psd, &freqs, None, None).unwrap();
///
/// // Kurtosis should be greater than or equal to -2.0 (theoretical lower bound)
/// assert!(kurtosis >= -2.0);
/// ```
pub fn spectral_kurtosis<T, U>(
    psd: &[T],
    freqs: &[U],
    centroid: Option<f64>,
    spread: Option<f64>,
) -> SignalResult<f64>
where
    T: Float + NumCast + Debug,
    U: Float + NumCast + Debug,
{
    if psd.is_empty() || freqs.is_empty() {
        return Err(SignalError::ValueError(
            "PSD or frequency array is empty".to_string(),
        ));
    }

    if psd.len() != freqs.len() {
        return Err(SignalError::ValueError(
            "PSD and frequency arrays must have the same length".to_string(),
        ));
    }

    // Convert to f64 for internal processing
    let psd_f64: Vec<f64> = psd
        .iter()
        .map(|&val| {
            num_traits::cast::cast::<T, f64>(val).ok_or_else(|| {
                SignalError::ValueError(format!("Could not convert {:?} to f64", val))
            })
        })
        .collect::<SignalResult<Vec<_>>>()?;

    let freqs_f64: Vec<f64> = freqs
        .iter()
        .map(|&val| {
            num_traits::cast::cast::<U, f64>(val).ok_or_else(|| {
                SignalError::ValueError(format!("Could not convert {:?} to f64", val))
            })
        })
        .collect::<SignalResult<Vec<_>>>()?;

    // Calculate sum of PSD values
    let sum: f64 = psd_f64.iter().sum();

    if sum <= 0.0 {
        return Err(SignalError::ValueError(
            "Sum of PSD values must be positive for kurtosis calculation".to_string(),
        ));
    }

    // Calculate or use provided centroid
    let centroid_val = match centroid {
        Some(c) => c,
        None => spectral_centroid(psd, freqs)?,
    };

    // Calculate or use provided spread
    let spread_val = match spread {
        Some(s) => s,
        None => spectral_spread(psd, freqs, Some(centroid_val))?,
    };

    if spread_val <= 0.0 {
        return Err(SignalError::ValueError(
            "Spectral spread must be positive for kurtosis calculation".to_string(),
        ));
    }

    // Calculate weighted sum of fourth-power differences from centroid
    let weighted_sum_fourth_power_diff: f64 = psd_f64
        .iter()
        .zip(freqs_f64.iter())
        .map(|(&p, &f)| p * (f - centroid_val).powi(4))
        .sum();

    // Calculate kurtosis (excess kurtosis: normal distribution has kurtosis = 0)
    let kurtosis = weighted_sum_fourth_power_diff / (sum * spread_val.powi(4)) - 3.0;

    Ok(kurtosis)
}

/// Calculate the spectral flatness of a signal.
///
/// Spectral flatness is the ratio of the geometric mean to the arithmetic mean
/// of the power spectrum. It quantifies how noise-like or tone-like a signal is.
///
/// - Values close to 0 indicate a more tonal sound
/// - Values close to 1 indicate a more noise-like sound
///
/// # Arguments
///
/// * `psd` - Power spectral density
///
/// # Returns
///
/// * Spectral flatness (dimensionless)
///
/// # Examples
///
/// ```
/// use scirs2_signal::spectral::periodogram;
/// use scirs2_signal::utilities::spectral::spectral_flatness;
///
/// let signal = vec![1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 0.0];
/// let fs = 8.0; // Sample rate in Hz
///
/// let (psd, _) = periodogram(&signal, Some(fs), Some("hamming"), Some(16), None, None).unwrap();
/// let flatness = spectral_flatness(&psd).unwrap();
///
/// // Flatness should be between 0 and 1
/// assert!(flatness >= 0.0 && flatness <= 1.0);
/// ```
pub fn spectral_flatness<T>(psd: &[T]) -> SignalResult<f64>
where
    T: Float + NumCast + Debug,
{
    if psd.is_empty() {
        return Err(SignalError::ValueError("PSD array is empty".to_string()));
    }

    // Convert to f64 for internal processing
    let psd_f64: Vec<f64> = psd
        .iter()
        .map(|&val| {
            num_traits::cast::cast::<T, f64>(val).ok_or_else(|| {
                SignalError::ValueError(format!("Could not convert {:?} to f64", val))
            })
        })
        .collect::<SignalResult<Vec<_>>>()?;

    // Filter out zero and negative values for geometric mean calculation
    let positive_psd: Vec<f64> = psd_f64.iter().filter(|&&p| p > 0.0).copied().collect();

    if positive_psd.is_empty() {
        return Err(SignalError::ValueError(
            "PSD contains only zero or negative values".to_string(),
        ));
    }

    // Calculate arithmetic mean
    let arithmetic_mean = positive_psd.iter().sum::<f64>() / positive_psd.len() as f64;

    if arithmetic_mean <= 0.0 {
        return Err(SignalError::ValueError(
            "Arithmetic mean of PSD is zero or negative".to_string(),
        ));
    }

    // Calculate geometric mean
    let log_sum: f64 = positive_psd.iter().map(|&p| p.ln()).sum();
    let geometric_mean = (log_sum / positive_psd.len() as f64).exp();

    // Calculate spectral flatness
    let flatness = geometric_mean / arithmetic_mean;

    Ok(flatness)
}

/// Calculate the spectral flux between two power spectral densities.
///
/// Spectral flux measures how quickly the power spectrum changes between consecutive
/// frames of a signal. It's often used to detect transients in audio signals.
///
/// # Arguments
///
/// * `psd1` - First power spectral density
/// * `psd2` - Second power spectral density
/// * `norm` - Normalization type: "l1" (Manhattan), "l2" (Euclidean), or "max" (Chebyshev)
///
/// # Returns
///
/// * Spectral flux (dimensionless)
///
/// # Examples
///
/// ```
/// use scirs2_signal::spectral::periodogram;
/// use scirs2_signal::utilities::spectral::spectral_flux;
///
/// let signal1 = vec![1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 0.0];
/// let signal2 = vec![0.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0];
/// let fs = 8.0; // Sample rate in Hz
///
/// let (psd1, _) = periodogram(&signal1, Some(fs), Some("hamming"), Some(16), None, None).unwrap();
/// let (psd2, _) = periodogram(&signal2, Some(fs), Some("hamming"), Some(16), None, None).unwrap();
///
/// let flux = spectral_flux(&psd1, &psd2, "l2").unwrap();
///
/// // Flux should be non-negative
/// assert!(flux >= 0.0);
/// ```
pub fn spectral_flux<T, U>(psd1: &[T], psd2: &[U], norm: &str) -> SignalResult<f64>
where
    T: Float + NumCast + Debug,
    U: Float + NumCast + Debug,
{
    if psd1.is_empty() || psd2.is_empty() {
        return Err(SignalError::ValueError(
            "PSD arrays must not be empty".to_string(),
        ));
    }

    if psd1.len() != psd2.len() {
        return Err(SignalError::ValueError(
            "PSD arrays must have the same length".to_string(),
        ));
    }

    // Convert to f64 for internal processing
    let psd1_f64: Vec<f64> = psd1
        .iter()
        .map(|&val| {
            num_traits::cast::cast::<T, f64>(val).ok_or_else(|| {
                SignalError::ValueError(format!("Could not convert {:?} to f64", val))
            })
        })
        .collect::<SignalResult<Vec<_>>>()?;

    let psd2_f64: Vec<f64> = psd2
        .iter()
        .map(|&val| {
            num_traits::cast::cast::<U, f64>(val).ok_or_else(|| {
                SignalError::ValueError(format!("Could not convert {:?} to f64", val))
            })
        })
        .collect::<SignalResult<Vec<_>>>()?;

    // Convert to ndarray for easier operations
    let psd1_array = Array1::from(psd1_f64);
    let psd2_array = Array1::from(psd2_f64);

    // Calculate difference
    let diff = &psd2_array - &psd1_array;

    // Calculate flux based on norm type
    match norm.to_lowercase().as_str() {
        "l1" => {
            // Manhattan distance (L1 norm)
            let flux = diff.iter().map(|&d| d.abs()).sum();
            Ok(flux)
        }
        "l2" => {
            // Euclidean distance (L2 norm)
            let flux = (diff.iter().map(|&d| d * d).sum::<f64>()).sqrt();
            Ok(flux)
        }
        "max" => {
            // Chebyshev distance (max norm)
            let flux = diff.iter().fold(0.0, |max_val, &d| max_val.max(d.abs()));
            Ok(flux)
        }
        _ => Err(SignalError::ValueError(format!(
            "Unknown norm type: {}",
            norm
        ))),
    }
}

/// Calculate the spectral rolloff frequency of a signal.
///
/// The spectral rolloff is the frequency below which a specified percentage
/// of the total spectral energy is contained.
///
/// # Arguments
///
/// * `psd` - Power spectral density
/// * `freqs` - Frequency values corresponding to the PSD
/// * `percentage` - Percentage of energy (0.0 to 1.0)
///
/// # Returns
///
/// * Spectral rolloff frequency
///
/// # Examples
///
/// ```
/// use scirs2_signal::spectral::periodogram;
/// use scirs2_signal::utilities::spectral::spectral_rolloff;
///
/// let signal = vec![1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 0.0];
/// let fs = 8.0; // Sample rate in Hz
///
/// let (psd, freqs) = periodogram(&signal, Some(fs), Some("hamming"), Some(16), None, None).unwrap();
/// let rolloff = spectral_rolloff(&psd, &freqs, 0.85).unwrap();
///
/// // Basic sanity check - rolloff should be finite
/// assert!(rolloff.is_finite());
/// ```
pub fn spectral_rolloff<T, U>(psd: &[T], freqs: &[U], percentage: f64) -> SignalResult<f64>
where
    T: Float + NumCast + Debug,
    U: Float + NumCast + Debug,
{
    if psd.is_empty() || freqs.is_empty() {
        return Err(SignalError::ValueError(
            "PSD or frequency array is empty".to_string(),
        ));
    }

    if psd.len() != freqs.len() {
        return Err(SignalError::ValueError(
            "PSD and frequency arrays must have the same length".to_string(),
        ));
    }

    if !(0.0..=1.0).contains(&percentage) {
        return Err(SignalError::ValueError(
            "Percentage must be between 0.0 and 1.0".to_string(),
        ));
    }

    // Convert to f64 for internal processing
    let psd_f64: Vec<f64> = psd
        .iter()
        .map(|&val| {
            num_traits::cast::cast::<T, f64>(val).ok_or_else(|| {
                SignalError::ValueError(format!("Could not convert {:?} to f64", val))
            })
        })
        .collect::<SignalResult<Vec<_>>>()?;

    let freqs_f64: Vec<f64> = freqs
        .iter()
        .map(|&val| {
            num_traits::cast::cast::<U, f64>(val).ok_or_else(|| {
                SignalError::ValueError(format!("Could not convert {:?} to f64", val))
            })
        })
        .collect::<SignalResult<Vec<_>>>()?;

    // Calculate total energy
    let total_energy: f64 = psd_f64.iter().sum();

    if total_energy <= 0.0 {
        return Err(SignalError::ValueError(
            "Total energy must be positive for rolloff calculation".to_string(),
        ));
    }

    // Calculate target energy
    let target_energy = total_energy * percentage;

    // Find the frequency where cumulative energy exceeds target
    let mut cumulative_energy = 0.0;
    for (&psd_val, &freq_val) in psd_f64.iter().zip(freqs_f64.iter()) {
        cumulative_energy += psd_val;
        if cumulative_energy >= target_energy {
            return Ok(freq_val);
        }
    }

    // If we reach here, return the highest frequency
    Ok(freqs_f64[freqs_f64.len() - 1])
}

/// Calculate the spectral crest factor of a signal.
///
/// The spectral crest factor is the ratio of the maximum value to the arithmetic mean
/// of the power spectrum. It is a measure of the "peakiness" of the spectrum.
/// Higher values indicate more tone-like sounds, while lower values indicate more noise-like sounds.
///
/// # Arguments
///
/// * `psd` - Power spectral density
///
/// # Returns
///
/// * Spectral crest factor (dimensionless)
///
/// # Examples
///
/// ```
/// use scirs2_signal::spectral::periodogram;
/// use scirs2_signal::utilities::spectral::spectral_crest;
///
/// let signal = vec![1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 0.0];
/// let fs = 8.0; // Sample rate in Hz
///
/// let (psd, _) = periodogram(&signal, Some(fs), Some("hamming"), Some(16), None, None).unwrap();
/// let crest = spectral_crest(&psd).unwrap();
///
/// // Crest factor should be greater than or equal to 1.0
/// assert!(crest >= 1.0);
/// ```
pub fn spectral_crest<T>(psd: &[T]) -> SignalResult<f64>
where
    T: Float + NumCast + Debug,
{
    if psd.is_empty() {
        return Err(SignalError::ValueError("PSD array is empty".to_string()));
    }

    // Convert to f64 for internal processing
    let psd_f64: Vec<f64> = psd
        .iter()
        .map(|&val| {
            num_traits::cast::cast::<T, f64>(val).ok_or_else(|| {
                SignalError::ValueError(format!("Could not convert {:?} to f64", val))
            })
        })
        .collect::<SignalResult<Vec<_>>>()?;

    // Calculate maximum value
    let max_val = psd_f64.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    // Calculate arithmetic mean
    let mean_val = psd_f64.iter().sum::<f64>() / psd_f64.len() as f64;

    if mean_val <= 0.0 {
        return Err(SignalError::ValueError(
            "Mean of PSD values must be positive for crest factor calculation".to_string(),
        ));
    }

    // Calculate spectral crest factor
    let crest = max_val / mean_val;

    Ok(crest)
}

/// Calculate the spectral decrease of a signal.
///
/// The spectral decrease is a perceptual measure that represents the amount of
/// decreasing in the spectral amplitude. It is often used in audio analysis.
///
/// # Arguments
///
/// * `psd` - Power spectral density
/// * `freqs` - Frequency values corresponding to the PSD
///
/// # Returns
///
/// * Spectral decrease (dimensionless)
///
/// # Examples
///
/// ```
/// use scirs2_signal::spectral::periodogram;
/// use scirs2_signal::utilities::spectral::spectral_decrease;
///
/// let signal = vec![1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 0.0];
/// let fs = 8.0; // Sample rate in Hz
///
/// let (psd, freqs) = periodogram(&signal, Some(fs), Some("hamming"), Some(16), None, None).unwrap();
/// let decrease = spectral_decrease(&psd, &freqs).unwrap();
/// ```
pub fn spectral_decrease<T, U>(psd: &[T], freqs: &[U]) -> SignalResult<f64>
where
    T: Float + NumCast + Debug,
    U: Float + NumCast + Debug,
{
    if psd.is_empty() || freqs.is_empty() {
        return Err(SignalError::ValueError(
            "PSD or frequency array is empty".to_string(),
        ));
    }

    if psd.len() != freqs.len() {
        return Err(SignalError::ValueError(
            "PSD and frequency arrays must have the same length".to_string(),
        ));
    }

    // Convert to f64 for internal processing
    let psd_f64: Vec<f64> = psd
        .iter()
        .map(|&val| {
            num_traits::cast::cast::<T, f64>(val).ok_or_else(|| {
                SignalError::ValueError(format!("Could not convert {:?} to f64", val))
            })
        })
        .collect::<SignalResult<Vec<_>>>()?;

    let freqs_f64: Vec<f64> = freqs
        .iter()
        .map(|&val| {
            num_traits::cast::cast::<U, f64>(val).ok_or_else(|| {
                SignalError::ValueError(format!("Could not convert {:?} to f64", val))
            })
        })
        .collect::<SignalResult<Vec<_>>>()?;

    if psd_f64.len() < 2 {
        return Err(SignalError::ValueError(
            "Need at least 2 points to calculate spectral decrease".to_string(),
        ));
    }

    let first_value = psd_f64[0];
    if first_value == 0.0 {
        // Handle the case where first value is zero
        return Ok(0.0);
    }

    // Calculate weighted sum of amplitude differences
    let mut weighted_sum = 0.0;
    let mut amplitude_sum = 0.0;

    for i in 1..psd_f64.len() {
        weighted_sum += (psd_f64[i] - psd_f64[0]) / freqs_f64[i];
        amplitude_sum += psd_f64[i];
    }

    // Calculate spectral decrease
    let decrease = weighted_sum / amplitude_sum;

    Ok(decrease)
}

/// Calculate the spectral slope of a signal.
///
/// The spectral slope is a measure of how quickly the spectrum falls off with frequency.
/// It is calculated as the linear regression slope of the spectrum magnitude.
///
/// # Arguments
///
/// * `psd` - Power spectral density
/// * `freqs` - Frequency values corresponding to the PSD
///
/// # Returns
///
/// * Spectral slope
///
/// # Examples
///
/// ```
/// use scirs2_signal::spectral::periodogram;
/// use scirs2_signal::utilities::spectral::spectral_slope;
///
/// let signal = vec![1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 0.0];
/// let fs = 8.0; // Sample rate in Hz
///
/// let (psd, freqs) = periodogram(&signal, Some(fs), Some("hamming"), Some(16), None, None).unwrap();
/// let slope = spectral_slope(&psd, &freqs).unwrap();
/// ```
pub fn spectral_slope<T, U>(psd: &[T], freqs: &[U]) -> SignalResult<f64>
where
    T: Float + NumCast + Debug,
    U: Float + NumCast + Debug,
{
    if psd.is_empty() || freqs.is_empty() {
        return Err(SignalError::ValueError(
            "PSD or frequency array is empty".to_string(),
        ));
    }

    if psd.len() != freqs.len() {
        return Err(SignalError::ValueError(
            "PSD and frequency arrays must have the same length".to_string(),
        ));
    }

    // Convert to f64 for internal processing
    let psd_f64: Vec<f64> = psd
        .iter()
        .map(|&val| {
            num_traits::cast::cast::<T, f64>(val).ok_or_else(|| {
                SignalError::ValueError(format!("Could not convert {:?} to f64", val))
            })
        })
        .collect::<SignalResult<Vec<_>>>()?;

    let freqs_f64: Vec<f64> = freqs
        .iter()
        .map(|&val| {
            num_traits::cast::cast::<U, f64>(val).ok_or_else(|| {
                SignalError::ValueError(format!("Could not convert {:?} to f64", val))
            })
        })
        .collect::<SignalResult<Vec<_>>>()?;

    // Calculate means
    let freq_mean = freqs_f64.iter().sum::<f64>() / freqs_f64.len() as f64;
    let psd_mean = psd_f64.iter().sum::<f64>() / psd_f64.len() as f64;

    // Calculate covariance and variance
    let mut covariance = 0.0;
    let mut variance = 0.0;

    for i in 0..psd_f64.len() {
        let freq_diff = freqs_f64[i] - freq_mean;
        let psd_diff = psd_f64[i] - psd_mean;

        covariance += freq_diff * psd_diff;
        variance += freq_diff * freq_diff;
    }

    if variance.abs() < f64::EPSILON {
        return Err(SignalError::ValueError(
            "Frequency variance is too small for slope calculation".to_string(),
        ));
    }

    // Calculate linear regression slope
    let slope = covariance / variance;

    Ok(slope)
}

/// Calculate the Spectral Contrast of a signal.
///
/// Spectral contrast measures the difference between peaks and valleys in the spectrum.
/// It is computed by calculating the difference between the peak and valley
/// for sub-bands of the spectrum.
///
/// # Arguments
///
/// * `psd` - Power spectral density
/// * `freqs` - Frequency values corresponding to the PSD
/// * `n_bands` - Number of sub-bands to divide the spectrum into
///
/// # Returns
///
/// * Vector of spectral contrast values, one per band
///
/// # Examples
///
/// ```
/// use scirs2_signal::spectral::periodogram;
/// use scirs2_signal::utilities::spectral::spectral_contrast;
///
/// let signal = vec![1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 0.0];
/// let fs = 8.0; // Sample rate in Hz
///
/// let (psd, freqs) = periodogram(&signal, Some(fs), Some("hamming"), Some(16), None, None).unwrap();
/// let contrast = spectral_contrast(&psd, &freqs, 4).unwrap();
///
/// assert_eq!(contrast.len(), 4);
/// ```
pub fn spectral_contrast<T, U>(psd: &[T], freqs: &[U], n_bands: usize) -> SignalResult<Vec<f64>>
where
    T: Float + NumCast + Debug,
    U: Float + NumCast + Debug,
{
    if psd.is_empty() || freqs.is_empty() {
        return Err(SignalError::ValueError(
            "PSD or frequency array is empty".to_string(),
        ));
    }

    if psd.len() != freqs.len() {
        return Err(SignalError::ValueError(
            "PSD and frequency arrays must have the same length".to_string(),
        ));
    }

    if n_bands < 1 {
        return Err(SignalError::ValueError(
            "Number of bands must be positive".to_string(),
        ));
    }

    if psd.len() < n_bands * 2 {
        return Err(SignalError::ValueError(
            "Not enough PSD points for requested number of bands".to_string(),
        ));
    }

    // Convert to f64 for internal processing
    let psd_f64: Vec<f64> = psd
        .iter()
        .map(|&val| {
            num_traits::cast::cast::<T, f64>(val).ok_or_else(|| {
                SignalError::ValueError(format!("Could not convert {:?} to f64", val))
            })
        })
        .collect::<SignalResult<Vec<_>>>()?;

    let freqs_f64: Vec<f64> = freqs
        .iter()
        .map(|&val| {
            num_traits::cast::cast::<U, f64>(val).ok_or_else(|| {
                SignalError::ValueError(format!("Could not convert {:?} to f64", val))
            })
        })
        .collect::<SignalResult<Vec<_>>>()?;

    // Find frequency range
    let min_freq = freqs_f64[0];
    let max_freq = freqs_f64[freqs_f64.len() - 1];
    let freq_range = max_freq - min_freq;

    // Calculate band boundaries
    let band_width = freq_range / n_bands as f64;
    let mut contrasts = Vec::with_capacity(n_bands);

    // Calculate contrast for each band
    for band in 0..n_bands {
        let band_start = min_freq + band as f64 * band_width;
        let band_end = band_start + band_width;

        // Find indices in this band
        let band_indices: Vec<usize> = freqs_f64
            .iter()
            .enumerate()
            .filter(|(_, &freq)| freq >= band_start && freq < band_end)
            .map(|(i, _)| i)
            .collect();

        if band_indices.is_empty() {
            contrasts.push(0.0); // No data in this band
            continue;
        }

        // Get and sort PSD values in this band
        let mut band_psd: Vec<f64> = band_indices.iter().map(|&idx| psd_f64[idx]).collect();
        band_psd.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Calculate percentiles
        let n = band_psd.len();
        let valley_idx = (n as f64 * 0.15) as usize;
        let peak_idx = (n as f64 * 0.85) as usize;

        let valley = if valley_idx < n {
            band_psd[valley_idx]
        } else {
            band_psd[0]
        };
        let peak = if peak_idx < n {
            band_psd[peak_idx]
        } else {
            band_psd[n - 1]
        };

        // Handle log computation carefully to avoid numerical issues
        if peak <= 0.0 || valley <= 0.0 {
            contrasts.push(0.0);
        } else {
            let contrast = (peak / valley).log10();
            contrasts.push(contrast);
        }
    }

    Ok(contrasts)
}

/// Calculate the spectral bandwidth of a signal at a specific threshold.
///
/// The spectral bandwidth is the width of the frequency band where the
/// spectral magnitudes are above a specific threshold relative to the peak magnitude.
///
/// # Arguments
///
/// * `psd` - Power spectral density
/// * `freqs` - Frequency values corresponding to the PSD
/// * `threshold_db` - Threshold in decibels below the peak (e.g., -3 for half-power bandwidth)
///
/// # Returns
///
/// * Spectral bandwidth in the same units as freqs
///
/// # Examples
///
/// ```
/// use scirs2_signal::spectral::periodogram;
/// use scirs2_signal::utilities::spectral::spectral_bandwidth;
///
/// let signal = vec![1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 0.0];
/// let fs = 8.0; // Sample rate in Hz
///
/// let (psd, freqs) = periodogram(&signal, Some(fs), Some("hamming"), Some(16), None, None).unwrap();
/// let bandwidth = spectral_bandwidth(&psd, &freqs, -3.0).unwrap();
///
/// // Basic sanity check - bandwidth should be finite
/// assert!(bandwidth.is_finite());
/// ```
pub fn spectral_bandwidth<T, U>(psd: &[T], freqs: &[U], threshold_db: f64) -> SignalResult<f64>
where
    T: Float + NumCast + Debug,
    U: Float + NumCast + Debug,
{
    if psd.is_empty() || freqs.is_empty() {
        return Err(SignalError::ValueError(
            "PSD or frequency array is empty".to_string(),
        ));
    }

    if psd.len() != freqs.len() {
        return Err(SignalError::ValueError(
            "PSD and frequency arrays must have the same length".to_string(),
        ));
    }

    // Convert to f64 for internal processing
    let psd_f64: Vec<f64> = psd
        .iter()
        .map(|&val| {
            num_traits::cast::cast::<T, f64>(val).ok_or_else(|| {
                SignalError::ValueError(format!("Could not convert {:?} to f64", val))
            })
        })
        .collect::<SignalResult<Vec<_>>>()?;

    let freqs_f64: Vec<f64> = freqs
        .iter()
        .map(|&val| {
            num_traits::cast::cast::<U, f64>(val).ok_or_else(|| {
                SignalError::ValueError(format!("Could not convert {:?} to f64", val))
            })
        })
        .collect::<SignalResult<Vec<_>>>()?;

    // Find peak magnitude
    let peak_magnitude = psd_f64.iter().fold(0.0, |max_val, &val| max_val.max(val));

    if peak_magnitude <= 0.0 {
        return Err(SignalError::ValueError(
            "Peak magnitude must be positive for bandwidth calculation".to_string(),
        ));
    }

    // Calculate threshold in linear scale
    let threshold_linear = peak_magnitude * 10.0_f64.powf(threshold_db / 10.0);

    // Find the frequencies where the PSD crosses the threshold
    let mut crossings = Vec::new();

    for i in 0..psd_f64.len() - 1 {
        if (psd_f64[i] <= threshold_linear && psd_f64[i + 1] > threshold_linear)
            || (psd_f64[i] > threshold_linear && psd_f64[i + 1] <= threshold_linear)
        {
            // Linear interpolation to find the exact crossing
            let t = (threshold_linear - psd_f64[i]) / (psd_f64[i + 1] - psd_f64[i]);
            let crossing_freq = freqs_f64[i] + t * (freqs_f64[i + 1] - freqs_f64[i]);
            crossings.push(crossing_freq);
        }
    }

    // Need at least two crossings to calculate bandwidth
    if crossings.len() < 2 {
        // If we can't find two crossings, return 0.0 as a reasonable default
        return Ok(0.0);
    }

    // Sort crossings to find the outermost ones
    crossings.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Calculate bandwidth as the difference between the outermost crossings
    let bandwidth = crossings[crossings.len() - 1] - crossings[0];

    Ok(bandwidth)
}

/// Find the dominant frequency in a power spectrum.
///
/// The dominant frequency is the frequency with the highest magnitude in the spectrum.
///
/// # Arguments
///
/// * `psd` - Power spectral density
/// * `freqs` - Frequency values corresponding to the PSD
///
/// # Returns
///
/// * Tuple containing (dominant frequency, magnitude at that frequency)
///
/// # Examples
///
/// ```
/// use scirs2_signal::spectral::periodogram;
/// use scirs2_signal::utilities::spectral::dominant_frequency;
///
/// let signal = vec![1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 0.0];
/// let fs = 8.0; // Sample rate in Hz
///
/// let (psd, freqs) = periodogram(&signal, Some(fs), Some("hamming"), Some(16), None, None).unwrap();
/// let (dominant_freq, magnitude) = dominant_frequency(&psd, &freqs).unwrap();
///
/// // Basic sanity check - just verify function succeeds
/// assert!(dominant_freq.is_finite());
/// assert!(magnitude >= 0.0);
/// ```
pub fn dominant_frequency<T, U>(psd: &[T], freqs: &[U]) -> SignalResult<(f64, f64)>
where
    T: Float + NumCast + Debug,
    U: Float + NumCast + Debug,
{
    if psd.is_empty() || freqs.is_empty() {
        return Err(SignalError::ValueError(
            "PSD or frequency array is empty".to_string(),
        ));
    }

    if psd.len() != freqs.len() {
        return Err(SignalError::ValueError(
            "PSD and frequency arrays must have the same length".to_string(),
        ));
    }

    // Convert to f64 for internal processing
    let psd_f64: Vec<f64> = psd
        .iter()
        .map(|&val| {
            num_traits::cast::cast::<T, f64>(val).ok_or_else(|| {
                SignalError::ValueError(format!("Could not convert {:?} to f64", val))
            })
        })
        .collect::<SignalResult<Vec<_>>>()?;

    let freqs_f64: Vec<f64> = freqs
        .iter()
        .map(|&val| {
            num_traits::cast::cast::<U, f64>(val).ok_or_else(|| {
                SignalError::ValueError(format!("Could not convert {:?} to f64", val))
            })
        })
        .collect::<SignalResult<Vec<_>>>()?;

    // Find the index of the maximum magnitude
    let (max_idx, &max_val) = psd_f64
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap_or((0, &0.0));

    let dominant_freq = freqs_f64[max_idx];

    Ok((dominant_freq, max_val))
}

/// Find the n most dominant frequencies in a power spectrum.
///
/// # Arguments
///
/// * `psd` - Power spectral density
/// * `freqs` - Frequency values corresponding to the PSD
/// * `n` - Number of dominant frequencies to find
/// * `min_separation` - Minimum frequency separation between peaks (in same units as freqs)
///
/// # Returns
///
/// * Vector of tuples, each containing (frequency, magnitude)
///
/// # Examples
///
/// ```
/// use scirs2_signal::spectral::periodogram;
/// use scirs2_signal::utilities::spectral::dominant_frequencies;
///
/// let signal = vec![1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 0.0];
/// let fs = 8.0; // Sample rate in Hz
///
/// let (psd, freqs) = periodogram(&signal, Some(fs), Some("hamming"), Some(16), None, None).unwrap();
/// let peaks = dominant_frequencies(&psd, &freqs, 3, 0.5).unwrap();
///
/// // There might be fewer than n peaks if not enough can be found
/// assert!(peaks.len() <= 3);
/// ```
pub fn dominant_frequencies<T, U>(
    psd: &[T],
    freqs: &[U],
    n: usize,
    min_separation: f64,
) -> SignalResult<Vec<(f64, f64)>>
where
    T: Float + NumCast + Debug,
    U: Float + NumCast + Debug,
{
    if psd.is_empty() || freqs.is_empty() {
        return Err(SignalError::ValueError(
            "PSD or frequency array is empty".to_string(),
        ));
    }

    if psd.len() != freqs.len() {
        return Err(SignalError::ValueError(
            "PSD and frequency arrays must have the same length".to_string(),
        ));
    }

    if n == 0 {
        return Err(SignalError::ValueError(
            "Number of peaks to find must be positive".to_string(),
        ));
    }

    if min_separation < 0.0 {
        return Err(SignalError::ValueError(
            "Minimum separation must be non-negative".to_string(),
        ));
    }

    // Convert to f64 for internal processing
    let psd_f64: Vec<f64> = psd
        .iter()
        .map(|&val| {
            num_traits::cast::cast::<T, f64>(val).ok_or_else(|| {
                SignalError::ValueError(format!("Could not convert {:?} to f64", val))
            })
        })
        .collect::<SignalResult<Vec<_>>>()?;

    let freqs_f64: Vec<f64> = freqs
        .iter()
        .map(|&val| {
            num_traits::cast::cast::<U, f64>(val).ok_or_else(|| {
                SignalError::ValueError(format!("Could not convert {:?} to f64", val))
            })
        })
        .collect::<SignalResult<Vec<_>>>()?;

    // Create a vector of (index, magnitude) pairs
    let mut peaks = Vec::new();

    // Find local maxima in the spectrum
    for i in 1..psd_f64.len() - 1 {
        if psd_f64[i] > psd_f64[i - 1] && psd_f64[i] > psd_f64[i + 1] {
            peaks.push((i, psd_f64[i]));
        }
    }

    // Also consider endpoints if they're the highest point
    if psd_f64.len() > 1 {
        if psd_f64[0] > psd_f64[1] {
            peaks.push((0, psd_f64[0]));
        }

        let last = psd_f64.len() - 1;
        if psd_f64[last] > psd_f64[last - 1] {
            peaks.push((last, psd_f64[last]));
        }
    }

    // Sort by magnitude (descending)
    peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Select the top n peaks with the given minimum separation
    let mut selected_peaks = Vec::new();
    let mut selected_indices = Vec::new();

    for (idx, mag) in peaks {
        // Check if this peak is far enough from all selected peaks
        let freq: f64 = freqs_f64[idx];
        let mut too_close = false;

        for &selected_idx in &selected_indices {
            let selected_freq: f64 = freqs_f64[selected_idx];
            let distance: f64 = (freq - selected_freq).abs();
            if distance < min_separation {
                too_close = true;
                break;
            }
        }

        if !too_close {
            selected_peaks.push((freq, mag));
            selected_indices.push(idx);

            if selected_peaks.len() >= n {
                break;
            }
        }
    }

    Ok(selected_peaks)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_energy_spectral_density() {
        // Create a simple PSD
        let psd = vec![1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0];
        let fs = 100.0; // Sample rate in Hz

        let esd = energy_spectral_density(&psd, fs).unwrap();

        // Check scaling by sample interval (1/fs)
        for (i, &p) in psd.iter().enumerate() {
            assert_relative_eq!(esd[i], p / fs, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_normalized_psd() {
        // Create a simple PSD
        let psd = vec![1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0];

        let norm_psd = normalized_psd(&psd).unwrap();

        // Check sum is 1.0
        let sum: f64 = norm_psd.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-10);

        // Check shape is preserved
        for (i, &p) in psd.iter().enumerate() {
            if i > 0 {
                assert_relative_eq!(norm_psd[i] / norm_psd[0], p / psd[0], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_spectral_centroid() {
        // Create a symmetric PSD with peak in the middle
        let psd = vec![1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0];
        let freqs = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let centroid = spectral_centroid(&psd, &freqs).unwrap();

        // For symmetric PSD with peak in the middle, centroid should be at the middle frequency
        assert_relative_eq!(centroid, 3.0, epsilon = 1e-10);

        // Test non-symmetric PSD
        let psd = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let freqs = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let centroid = spectral_centroid(&psd, &freqs).unwrap();

        // Centroid should be biased toward higher frequencies
        assert!(centroid > 3.0);
    }

    #[test]
    fn test_spectral_spread() {
        // Create a symmetric PSD with peak in the middle
        let psd = vec![1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0];
        let freqs = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let spread = spectral_spread(&psd, &freqs, None).unwrap();

        // Spread should be positive
        assert!(spread > 0.0);

        // Create a very narrow PSD
        let psd = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        let freqs = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let spread = spectral_spread(&psd, &freqs, None).unwrap();

        // Spread should be very small for narrow PSD
        assert!(spread < 0.1);
    }

    #[test]
    fn test_spectral_flatness() {
        // Create a flat PSD (white noise-like)
        let psd = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

        let flatness = spectral_flatness(&psd).unwrap();

        // Flatness should be close to 1.0 for flat PSD
        assert_relative_eq!(flatness, 1.0, epsilon = 1e-10);

        // Create a PSD with a single peak (tone-like)
        let psd = vec![0.01, 0.01, 0.01, 1.0, 0.01, 0.01, 0.01];

        let flatness = spectral_flatness(&psd).unwrap();

        // Flatness should be close to 0.0 for peak PSD
        assert!(flatness < 0.3);
    }

    #[test]
    fn test_spectral_flux() {
        // Create two identical PSDs
        let psd1 = vec![1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0];
        let psd2 = vec![1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0];

        let flux_l1 = spectral_flux(&psd1, &psd2, "l1").unwrap();
        let flux_l2 = spectral_flux(&psd1, &psd2, "l2").unwrap();
        let flux_max = spectral_flux(&psd1, &psd2, "max").unwrap();

        // Flux should be 0.0 for identical PSDs
        assert_relative_eq!(flux_l1, 0.0, epsilon = 1e-10);
        assert_relative_eq!(flux_l2, 0.0, epsilon = 1e-10);
        assert_relative_eq!(flux_max, 0.0, epsilon = 1e-10);

        // Create two different PSDs
        let psd1 = vec![1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0];
        let psd2 = vec![0.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0];

        let flux_l1 = spectral_flux(&psd1, &psd2, "l1").unwrap();
        let flux_l2 = spectral_flux(&psd1, &psd2, "l2").unwrap();
        let flux_max = spectral_flux(&psd1, &psd2, "max").unwrap();

        // Flux should be positive for different PSDs
        assert!(flux_l1 > 0.0);
        assert!(flux_l2 > 0.0);
        assert!(flux_max > 0.0);
    }

    #[test]
    fn test_spectral_rolloff() {
        // Create a PSD with energy concentrated in first half
        let psd = vec![1.0, 2.0, 3.0, 4.0, 0.1, 0.1, 0.1];
        let freqs = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let rolloff = spectral_rolloff(&psd, &freqs, 0.95).unwrap();

        // Rolloff should be in the lower frequency range
        assert!(rolloff <= 4.0);

        // Create a PSD with energy concentrated in second half
        let psd = vec![0.1, 0.1, 0.1, 0.1, 3.0, 4.0, 5.0];
        let freqs = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let rolloff = spectral_rolloff(&psd, &freqs, 0.95).unwrap();

        // Rolloff should be in the higher frequency range
        assert!(rolloff >= 5.0);
    }
}
