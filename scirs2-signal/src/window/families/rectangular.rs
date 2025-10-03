//! Rectangular Window Family
//!
//! This module implements rectangular and uniform window functions including:
//! - Boxcar window (rectangular/uniform window)
//! - Dirichlet window (alias for boxcar)

use super::super::{_extend, _len_guards, _truncate};
use crate::error::{SignalError, SignalResult};
use std::f64::consts::PI;

/// Boxcar window (rectangular/uniform window)
///
/// The boxcar window is a simple rectangular window with all samples equal to 1.
/// It provides the best frequency resolution but has high spectral leakage.
///
/// # Arguments
/// * `m` - Number of points in the output window
/// * `sym` - If true, generates a symmetric window, otherwise a periodic window
///
/// # Returns
/// A Vec<f64> of window values (all ones)
///
/// # Examples
/// ```
/// use scirs2_signal::window::families::rectangular::boxcar;
/// let window = boxcar(10, true).unwrap();
/// assert_eq!(window.len(), 10);
/// assert!(window.iter().all(|&x| (x - 1.0).abs() < 1e-10));
/// ```
pub fn boxcar(m: usize, sym: bool) -> SignalResult<Vec<f64>> {
    if _len_guards(m) {
        return Ok(vec![1.0; m]);
    }

    let (n, needs_trunc) = _extend(m, sym);
    let w = vec![1.0; n];
    Ok(_truncate(w, needs_trunc))
}

/// Dirichlet window (alias for boxcar window)
///
/// The Dirichlet window is another name for the rectangular/boxcar window.
/// It corresponds to the Dirichlet kernel in frequency domain.
///
/// # Arguments
/// * `m` - Number of points in the output window
/// * `sym` - If true, generates a symmetric window, otherwise a periodic window
///
/// # Returns
/// A Vec<f64> of window values (all ones)
pub fn dirichlet(m: usize, sym: bool) -> SignalResult<Vec<f64>> {
    boxcar(m, sym)
}

/// Rectangular window with specified amplitude
///
/// A generalized rectangular window with user-specified constant amplitude.
///
/// # Arguments
/// * `m` - Number of points in the output window
/// * `amplitude` - Constant amplitude value for all samples
/// * `sym` - If true, generates a symmetric window, otherwise a periodic window
///
/// # Returns
/// A Vec<f64> of window values (all equal to amplitude)
///
/// # Examples
/// ```
/// use scirs2_signal::window::families::rectangular::rectangular_with_amplitude;
/// let window = rectangular_with_amplitude(10, 0.5, true).unwrap();
/// assert_eq!(window.len(), 10);
/// assert!(window.iter().all(|&x| (x - 0.5).abs() < 1e-10));
/// ```
pub fn rectangular_with_amplitude(m: usize, amplitude: f64, sym: bool) -> SignalResult<Vec<f64>> {
    if _len_guards(m) {
        return Ok(vec![amplitude; m]);
    }

    let (n, needs_trunc) = _extend(m, sym);
    let w = vec![amplitude; n];
    Ok(_truncate(w, needs_trunc))
}

/// Analyze rectangular window properties
///
/// Computes important properties of rectangular windows for analysis
pub fn analyze_rectangular_window(window: &[f64]) -> RectangularWindowAnalysis {
    let n = window.len();

    // For rectangular windows, all samples should be equal
    let amplitude = window.first().copied().unwrap_or(1.0);

    // Coherent gain
    let coherent_gain = amplitude;

    // Processing gain (for rectangular window = 1.0)
    let processing_gain = 1.0;

    // Scalloping loss (worst for rectangular window)
    let scalloping_loss = 3.92; // ~3.92 dB for rectangular window

    // Main lobe width (narrowest of all windows)
    let main_lobe_width = 2.0; // 2 bins for rectangular window

    // Maximum sidelobe level (worst of all windows)
    let max_sidelobe_level = -13.3; // First sidelobe at -13.3 dB

    RectangularWindowAnalysis {
        amplitude,
        coherent_gain,
        processing_gain,
        scalloping_loss,
        main_lobe_width,
        max_sidelobe_level,
    }
}

/// Rectangular window analysis results
#[derive(Debug, Clone)]
pub struct RectangularWindowAnalysis {
    /// Window amplitude (constant value)
    pub amplitude: f64,
    /// Coherent gain (same as amplitude)
    pub coherent_gain: f64,
    /// Processing gain (always 1.0 for rectangular)
    pub processing_gain: f64,
    /// Scalloping loss in dB
    pub scalloping_loss: f64,
    /// Main lobe width in bins
    pub main_lobe_width: f64,
    /// Maximum sidelobe level in dB
    pub max_sidelobe_level: f64,
}

/// Check if a window is rectangular (constant amplitude)
///
/// Determines if a given window is rectangular by checking if all values are equal.
///
/// # Arguments
/// * `window` - Window samples to analyze
/// * `tolerance` - Tolerance for equality comparison
///
/// # Returns
/// true if the window is rectangular (constant), false otherwise
pub fn is_rectangular_window(window: &[f64], tolerance: f64) -> bool {
    if window.is_empty() {
        return false;
    }

    let first_value = window[0];
    window.iter().all(|&x| (x - first_value).abs() <= tolerance)
}

/// Generate rectangular pulse train window
///
/// Creates a window consisting of rectangular pulses with specified duty cycle.
///
/// # Arguments
/// * `m` - Number of points in the output window
/// * `duty_cycle` - Fraction of time the pulse is high (0.0 to 1.0)
/// * `n_pulses` - Number of pulses in the window
/// * `sym` - If true, generates a symmetric window, otherwise a periodic window
///
/// # Returns
/// A Vec<f64> of window values representing rectangular pulse train
pub fn rectangular_pulse_train(
    m: usize,
    duty_cycle: f64,
    n_pulses: usize,
    sym: bool,
) -> SignalResult<Vec<f64>> {
    if !(0.0..=1.0).contains(&duty_cycle) {
        return Err(SignalError::ValueError(
            "Duty cycle must be between 0.0 and 1.0".to_string(),
        ));
    }

    if n_pulses == 0 {
        return Err(SignalError::ValueError(
            "Number of pulses must be positive".to_string(),
        ));
    }

    if _len_guards(m) {
        return Ok(vec![1.0; m]);
    }

    let (n, needs_trunc) = _extend(m, sym);
    let mut w = vec![0.0; n];

    let pulse_period = n as f64 / n_pulses as f64;
    let pulse_width = pulse_period * duty_cycle;

    for pulse in 0..n_pulses {
        let pulse_start = (pulse as f64 * pulse_period) as usize;
        let pulse_end = (pulse_start as f64 + pulse_width) as usize;
        let pulse_end = pulse_end.min(n);

        for i in pulse_start..pulse_end {
            w[i] = 1.0;
        }
    }

    Ok(_truncate(w, needs_trunc))
}

/// Compute rectangular window spectrum (sinc function)
///
/// Computes the theoretical frequency response of a rectangular window,
/// which is a sinc function in the frequency domain.
///
/// # Arguments
/// * `frequencies` - Normalized frequencies at which to evaluate (-0.5 to 0.5)
/// * `window_length` - Length of the rectangular window
///
/// # Returns
/// Complex frequency response values
pub fn rectangular_window_spectrum(
    frequencies: &[f64],
    window_length: usize,
) -> Vec<scirs2_core::numeric::Complex64> {
    use scirs2_core::numeric::Complex64;

    frequencies
        .iter()
        .map(|&f| {
            if f.abs() < 1e-10 {
                // Handle DC case
                Complex64::new(window_length as f64, 0.0)
            } else {
                let omega = 2.0 * PI * f;
                let numerator = (omega * window_length as f64 / 2.0).sin();
                let denominator = (omega / 2.0).sin();
                let magnitude = numerator / denominator;

                // Phase is 0 or π depending on sign
                if magnitude >= 0.0 {
                    Complex64::new(magnitude, 0.0)
                } else {
                    Complex64::new(-magnitude, 0.0) * Complex64::i()
                }
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_boxcar_basic() {
        let window = boxcar(5, true).unwrap();
        assert_eq!(window.len(), 5);
        assert!(window.iter().all(|&x| (x - 1.0).abs() < 1e-10));
    }

    #[test]
    fn test_rectangular_with_amplitude() {
        let window = rectangular_with_amplitude(5, 0.5, true).unwrap();
        assert_eq!(window.len(), 5);
        assert!(window.iter().all(|&x| (x - 0.5).abs() < 1e-10));
    }

    #[test]
    fn test_is_rectangular_window() {
        let rect_window = vec![1.0, 1.0, 1.0, 1.0];
        let non_rect_window = vec![0.0, 0.5, 1.0, 0.5];

        assert!(is_rectangular_window(&rect_window, 1e-10));
        assert!(!is_rectangular_window(&non_rect_window, 1e-10));
    }

    #[test]
    fn test_pulse_train() {
        let window = rectangular_pulse_train(10, 0.5, 2, true).unwrap();
        assert_eq!(window.len(), 10);

        // Should have roughly half the samples as 1.0
        let ones_count = window.iter().filter(|&&x| x > 0.5).count();
        assert!(ones_count >= 4 && ones_count <= 6); // Approximately half
    }
}
