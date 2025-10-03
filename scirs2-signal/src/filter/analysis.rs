// Filter analysis and characterization functions
//
// This module provides comprehensive analysis capabilities for digital filters
// including frequency response analysis, stability checking, and filter
// characterization for design validation and performance evaluation.

use super::application::{evaluate_transfer_function, find_polynomial_roots, group_delay};
use crate::error::{SignalError, SignalResult};
use scirs2_core::numeric::Complex64;

#[allow(unused_imports)]
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
    /// Phase response in radians
    pub phase: Vec<f64>,
    /// Group delay in samples
    pub group_delay: Vec<f64>,
    /// Passband ripple in dB
    pub passband_ripple: f64,
    /// Stopband attenuation in dB
    pub stopband_attenuation: f64,
    /// 3dB cutoff frequency (normalized)
    pub cutoff_3db: f64,
    /// 6dB cutoff frequency (normalized)
    pub cutoff_6db: f64,
    /// Transition bandwidth (normalized)
    pub transition_bandwidth: f64,
}

impl Default for FilterAnalysis {
    fn default() -> Self {
        Self {
            frequencies: Vec::new(),
            magnitude: Vec::new(),
            magnitude_db: Vec::new(),
            phase: Vec::new(),
            group_delay: Vec::new(),
            passband_ripple: 0.0,
            stopband_attenuation: 0.0,
            cutoff_3db: 0.0,
            cutoff_6db: 0.0,
            transition_bandwidth: 0.0,
        }
    }
}

/// Filter stability analysis results
///
/// Contains information about filter stability including pole locations,
/// stability margins, and overall stability assessment.
#[derive(Debug, Clone)]
pub struct FilterStability {
    /// Whether the filter is stable
    pub is_stable: bool,
    /// Pole locations in the z-plane
    pub poles: Vec<Complex64>,
    /// Stability margin (minimum distance from unit circle)
    pub stability_margin: f64,
    /// Maximum pole magnitude
    pub max_pole_magnitude: f64,
}

impl Default for FilterStability {
    fn default() -> Self {
        Self {
            is_stable: false,
            poles: Vec::new(),
            stability_margin: 0.0,
            max_pole_magnitude: 0.0,
        }
    }
}

/// Perform comprehensive filter analysis
///
/// This function provides detailed frequency response analysis including
/// magnitude, phase, group delay, and filter characteristics such as
/// cutoff frequencies, ripple, and transition bandwidth.
///
/// # Arguments
///
/// * `b` - Numerator coefficients
/// * `a` - Denominator coefficients
/// * `num_points` - Number of frequency points for analysis (default: 512)
///
/// # Returns
///
/// * FilterAnalysis struct containing comprehensive filter characteristics
///
/// # Examples
///
/// ```
/// use scirs2_signal::filter::analysis::analyze_filter;
/// use scirs2_signal::filter::iir::butter;
///
/// // Analyze a Butterworth filter
/// let (b, a) = butter(4, 0.2, "lowpass").unwrap();
/// let analysis = analyze_filter(&b, &a, Some(256)).unwrap();
///
/// println!("3dB cutoff: {:.3}", analysis.cutoff_3db);
/// println!("Stopband attenuation: {:.1} dB", analysis.stopband_attenuation);
/// ```
#[allow(dead_code)]
pub fn analyze_filter(
    b: &[f64],
    a: &[f64],
    num_points: Option<usize>,
) -> SignalResult<FilterAnalysis> {
    let n_points = num_points.unwrap_or(512);

    // Generate frequency _points from 0 to π (normalized 0 to 1)
    let frequencies: Vec<f64> = (0..n_points)
        .map(|i| i as f64 / (n_points - 1) as f64)
        .collect();

    let w_radians: Vec<f64> = frequencies
        .iter()
        .map(|&f| f * std::f64::consts::PI)
        .collect();

    // Calculate frequency response
    let mut magnitude = Vec::with_capacity(n_points);
    let mut phase = Vec::with_capacity(n_points);

    for &w in &w_radians {
        let h = evaluate_transfer_function(b, a, w);
        magnitude.push(h.norm());
        phase.push(h.arg());
    }

    // Convert magnitude to dB
    let magnitude_db: Vec<f64> = magnitude
        .iter()
        .map(|&mag| 20.0 * mag.log10().max(-100.0)) // Limit minimum to -100 dB
        .collect();

    // Calculate group delay
    let group_delay_result = group_delay(b, a, &w_radians)?;

    // Analyze filter characteristics
    let max_magnitude_db = magnitude_db
        .iter()
        .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    // Find 3dB and 6dB cutoff frequencies
    let cutoff_3db = find_cutoff_frequency(&frequencies, &magnitude_db, max_magnitude_db - 3.0);
    let cutoff_6db = find_cutoff_frequency(&frequencies, &magnitude_db, max_magnitude_db - 6.0);

    // Estimate passband and stopband characteristics
    let passband_end = cutoff_3db.min(0.3); // Assume passband ends around 3dB point or 0.3
    let stopband_start = cutoff_3db.max(0.7); // Assume stopband starts after 3dB point or 0.7

    let passband_indices: Vec<usize> = frequencies
        .iter()
        .enumerate()
        .filter(|(_, &f)| f <= passband_end)
        .map(|(i_, _)| i_)
        .collect();

    let stopband_indices: Vec<usize> = frequencies
        .iter()
        .enumerate()
        .filter(|(_, &f)| f >= stopband_start)
        .map(|(i_, _)| i_)
        .collect();

    let passband_ripple = if !passband_indices.is_empty() {
        let passband_max = passband_indices
            .iter()
            .map(|&i| magnitude_db[i])
            .fold(f64::NEG_INFINITY, f64::max);
        let passband_min = passband_indices
            .iter()
            .map(|&i| magnitude_db[i])
            .fold(f64::INFINITY, f64::min);
        passband_max - passband_min
    } else {
        0.0
    };

    let stopband_attenuation = if !stopband_indices.is_empty() {
        max_magnitude_db
            - stopband_indices
                .iter()
                .map(|&i| magnitude_db[i])
                .fold(f64::NEG_INFINITY, f64::max)
    } else {
        0.0
    };

    let transition_bandwidth = (cutoff_6db - cutoff_3db).abs();

    Ok(FilterAnalysis {
        frequencies,
        magnitude,
        magnitude_db,
        phase,
        group_delay: group_delay_result,
        passband_ripple,
        stopband_attenuation,
        cutoff_3db,
        cutoff_6db,
        transition_bandwidth,
    })
}

/// Check filter stability by analyzing pole locations
///
/// A digital filter is stable if all poles are inside the unit circle.
/// This function analyzes the poles and provides stability information.
///
/// # Arguments
///
/// * `a` - Denominator coefficients
///
/// # Returns
///
/// * FilterStability struct with stability analysis
///
/// # Examples
///
/// ```
/// use scirs2_signal::filter::analysis::check_filter_stability;
/// use scirs2_signal::filter::iir::butter;
///
/// // Check stability of a Butterworth filter
/// let (b, a) = butter(4, 0.2, "lowpass").unwrap();
/// let stability = check_filter_stability(&a).unwrap();
///
/// println!("Filter is stable: {}", stability.is_stable);
/// println!("Stability margin: {:.6}", stability.stability_margin);
/// ```
#[allow(dead_code)]
pub fn check_filter_stability(a: &[f64]) -> SignalResult<FilterStability> {
    if a.is_empty() || a[0].abs() < 1e-15 {
        return Err(SignalError::ValueError(
            "Invalid denominator coefficients".to_string(),
        ));
    }

    // Find the poles (roots of denominator polynomial)
    let poles = find_polynomial_roots(a)?;

    // Check if all poles are inside unit circle
    let mut is_stable = true;
    let mut max_magnitude: f64 = 0.0;
    let mut min_distance_to_unit_circle = f64::INFINITY;

    for &pole in &poles {
        let magnitude = pole.norm();
        max_magnitude = max_magnitude.max(magnitude);

        if magnitude >= 1.0 {
            is_stable = false;
        }

        let distance_to_unit_circle = 1.0 - magnitude;
        min_distance_to_unit_circle = min_distance_to_unit_circle.min(distance_to_unit_circle);
    }

    Ok(FilterStability {
        is_stable,
        poles,
        stability_margin: min_distance_to_unit_circle,
        max_pole_magnitude: max_magnitude,
    })
}

/// Evaluate magnitude and phase response at specific frequencies
///
/// Computes the magnitude and phase response of a digital filter at
/// specified frequency points.
///
/// # Arguments
///
/// * `b` - Numerator coefficients
/// * `a` - Denominator coefficients
/// * `frequencies` - Frequency points (normalized from 0 to 1)
///
/// # Returns
///
/// * Tuple of (magnitude, phase) vectors
///
/// # Examples
///
/// ```
/// use scirs2_signal::filter::analysis::frequency_response;
/// use scirs2_signal::filter::iir::butter;
///
/// let (b, a) = butter(4, 0.2, "lowpass").unwrap();
/// let freqs = vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5];
/// let (mag, phase) = frequency_response(&b, &a, &freqs).unwrap();
/// ```
#[allow(dead_code)]
pub fn frequency_response(
    b: &[f64],
    a: &[f64],
    frequencies: &[f64],
) -> SignalResult<(Vec<f64>, Vec<f64>)> {
    if a.is_empty() || a[0].abs() < 1e-15 {
        return Err(SignalError::ValueError(
            "Invalid denominator coefficients".to_string(),
        ));
    }

    let mut magnitude = Vec::with_capacity(frequencies.len());
    let mut phase = Vec::with_capacity(frequencies.len());

    for &freq in frequencies {
        if !(0.0..=1.0).contains(&freq) {
            return Err(SignalError::ValueError(
                "Frequencies must be normalized between 0 and 1".to_string(),
            ));
        }

        let w = freq * std::f64::consts::PI;
        let h = evaluate_transfer_function(b, a, w);
        magnitude.push(h.norm());
        phase.push(h.arg());
    }

    Ok((magnitude, phase))
}

/// Find poles and zeros of a digital filter
///
/// Extracts the poles and zeros from the transfer function coefficients.
/// This is useful for understanding filter behavior and stability.
///
/// # Arguments
///
/// * `b` - Numerator coefficients
/// * `a` - Denominator coefficients
///
/// # Returns
///
/// * Tuple of (zeros, poles) as complex numbers
///
/// # Examples
///
/// ```
/// use scirs2_signal::filter::analysis::find_poles_zeros;
/// use scirs2_signal::filter::iir::butter;
///
/// let (b, a) = butter(2, 0.3, "lowpass").unwrap();
/// let (zeros, poles) = find_poles_zeros(&b, &a).unwrap();
///
/// println!("Number of zeros: {}", zeros.len());
/// println!("Number of poles: {}", poles.len());
/// ```
#[allow(dead_code)]
pub fn find_poles_zeros(b: &[f64], a: &[f64]) -> SignalResult<(Vec<Complex64>, Vec<Complex64>)> {
    if a.is_empty() || a[0].abs() < 1e-15 {
        return Err(SignalError::ValueError(
            "Invalid denominator coefficients".to_string(),
        ));
    }

    let zeros = if b.is_empty() {
        Vec::new()
    } else {
        find_polynomial_roots(b)?
    };

    let poles = find_polynomial_roots(a)?;

    Ok((zeros, poles))
}

/// Compute filter quality factor (Q factor)
///
/// The Q factor describes the sharpness of the filter response.
/// For bandpass filters, Q = center_frequency / bandwidth.
///
/// # Arguments
///
/// * `b` - Numerator coefficients
/// * `a` - Denominator coefficients
/// * `num_points` - Number of points for frequency response analysis
///
/// # Returns
///
/// * Q factor value
///
/// # Examples
///
/// ```
/// use scirs2_signal::filter::analysis::compute_q_factor;
/// use scirs2_signal::filter::iir::butter;
///
/// let (b, a) = butter(4, 0.2, "lowpass").unwrap();
/// let q = compute_q_factor(&b, &a, 512).unwrap();
/// println!("Q factor: {:.2}", q);
/// ```
#[allow(dead_code)]
pub fn compute_q_factor(b: &[f64], a: &[f64], numpoints: usize) -> SignalResult<f64> {
    let analysis = analyze_filter(b, a, Some(numpoints))?;

    // Find the peak frequency
    let max_mag = analysis.magnitude.iter().fold(0.0f64, |a, &b| a.max(b));
    let peak_idx = analysis
        .magnitude
        .iter()
        .position(|&mag| mag == max_mag)
        .unwrap_or(0);

    let peak_freq = analysis.frequencies[peak_idx];

    // Find -3dB _points around the peak
    let target_mag = max_mag / std::f64::consts::SQRT_2; // -3dB point

    let mut lower_freq = 0.0;
    let mut upper_freq = 1.0;

    // Find lower -3dB point
    for i in (0..peak_idx).rev() {
        if analysis.magnitude[i] <= target_mag {
            lower_freq = analysis.frequencies[i];
            break;
        }
    }

    // Find upper -3dB point
    for i in peak_idx..analysis.magnitude.len() {
        if analysis.magnitude[i] <= target_mag {
            upper_freq = analysis.frequencies[i];
            break;
        }
    }

    let bandwidth = upper_freq - lower_freq;

    if bandwidth > 1e-10 {
        Ok(peak_freq / bandwidth)
    } else {
        Ok(f64::INFINITY)
    }
}

// Helper functions

/// Find the frequency where magnitude drops to a specific dB level
#[allow(dead_code)]
fn find_cutoff_frequency(frequencies: &[f64], magnitude_db: &[f64], targetdb: f64) -> f64 {
    // Find the index where magnitude first drops below target
    for (i, &mag_db) in magnitude_db.iter().enumerate() {
        if mag_db <= targetdb {
            if i == 0 {
                return frequencies[0];
            }
            // Linear interpolation between points
            let f1 = frequencies[i - 1];
            let f2 = frequencies[i];
            let m1 = magnitude_db[i - 1];
            let m2 = magnitude_db[i];

            if (m1 - m2).abs() < 1e-10 {
                return f1;
            }

            let t = (targetdb - m1) / (m2 - m1);
            return f1 + t * (f2 - f1);
        }
    }
    frequencies[frequencies.len() - 1]
}
