//! Exponential Window Family
//!
//! This module implements exponential and Gaussian window functions including:
//! - Exponential window (exponential decay)
//! - Gaussian window (Gaussian bell curve)
//! - Kaiser window (modified Bessel function based)
//! - Tukey window (tapered cosine)

use super::super::{_extend, _len_guards, _truncate};
use crate::error::{SignalError, SignalResult};
use std::f64::consts::PI;

/// Exponential window with exponential decay
///
/// The exponential window provides exponential weighting from center to edges.
/// Useful for systems with exponential decay characteristics.
///
/// # Arguments
/// * `m` - Number of points in the output window
/// * `tau` - Decay time constant (larger = slower decay)
/// * `sym` - If true, generates a symmetric window, otherwise a periodic window
///
/// # Returns
/// A Vec<f64> of window values
///
/// # Examples
/// ```
/// use scirs2_signal::window::families::exponential::exponential;
/// let window = exponential(10, 2.0, true).unwrap();
/// assert_eq!(window.len(), 10);
/// ```
pub fn exponential(m: usize, tau: f64, sym: bool) -> SignalResult<Vec<f64>> {
    if tau <= 0.0 {
        return Err(SignalError::ValueError(
            "Decay time constant tau must be positive".to_string(),
        ));
    }

    if _len_guards(m) {
        return Ok(vec![1.0; m]);
    }

    let (n, needs_trunc) = _extend(m, sym);
    let center = (n - 1) as f64 / 2.0;

    let mut w = Vec::with_capacity(n);
    for i in 0..n {
        let distance = (i as f64 - center).abs();
        let w_val = (-distance / tau).exp();
        w.push(w_val);
    }

    Ok(_truncate(w, needs_trunc))
}

/// Gaussian window (bell curve)
///
/// The Gaussian window follows a Gaussian (normal) distribution shape.
/// It provides smooth transitions and good frequency characteristics.
///
/// # Arguments
/// * `m` - Number of points in the output window
/// * `std` - Standard deviation (larger = wider bell curve)
/// * `sym` - If true, generates a symmetric window, otherwise a periodic window
///
/// # Returns
/// A Vec<f64> of window values
///
/// # Examples
/// ```
/// use scirs2_signal::window::families::exponential::gaussian;
/// let window = gaussian(10, 1.0, true).unwrap();
/// assert_eq!(window.len(), 10);
/// ```
pub fn gaussian(m: usize, std: f64, sym: bool) -> SignalResult<Vec<f64>> {
    if std <= 0.0 {
        return Err(SignalError::ValueError(
            "Standard deviation must be positive".to_string(),
        ));
    }

    if _len_guards(m) {
        return Ok(vec![1.0; m]);
    }

    let (n, needs_trunc) = _extend(m, sym);
    let center = (n - 1) as f64 / 2.0;

    let mut w = Vec::with_capacity(n);
    for i in 0..n {
        let distance = i as f64 - center;
        let w_val = (-(distance * distance) / (2.0 * std * std)).exp();
        w.push(w_val);
    }

    Ok(_truncate(w, needs_trunc))
}

/// Kaiser window (modified Bessel function based)
///
/// The Kaiser window uses modified Bessel functions and provides adjustable
/// trade-off between main lobe width and side lobe level.
///
/// # Arguments
/// * `m` - Number of points in the output window
/// * `beta` - Shape parameter (0 = rectangular, larger = more tapered)
/// * `sym` - If true, generates a symmetric window, otherwise a periodic window
///
/// # Returns
/// A Vec<f64> of window values
pub fn kaiser(m: usize, beta: f64, sym: bool) -> SignalResult<Vec<f64>> {
    if beta < 0.0 {
        return Err(SignalError::ValueError(
            "Beta parameter must be non-negative".to_string(),
        ));
    }

    if _len_guards(m) {
        return Ok(vec![1.0; m]);
    }

    let (n, needs_trunc) = _extend(m, sym);

    let mut w = Vec::with_capacity(n);
    let alpha = (n - 1) as f64 / 2.0;
    let i0_beta = modified_bessel_i0(beta);

    for i in 0..n {
        let x = (i as f64 - alpha) / alpha;
        let arg = beta * (1.0 - x * x).sqrt();
        let w_val = modified_bessel_i0(arg) / i0_beta;
        w.push(w_val);
    }

    Ok(_truncate(w, needs_trunc))
}

/// Tukey window (tapered cosine)
///
/// The Tukey window is a rectangular window with cosine tapers at both ends.
/// Also known as the tapered cosine window.
///
/// # Arguments
/// * `m` - Number of points in the output window
/// * `alpha` - Taper fraction (0.0 = rectangular, 1.0 = full cosine taper)
/// * `sym` - If true, generates a symmetric window, otherwise a periodic window
///
/// # Returns
/// A Vec<f64> of window values
pub fn tukey(m: usize, alpha: f64, sym: bool) -> SignalResult<Vec<f64>> {
    if !(0.0..=1.0).contains(&alpha) {
        return Err(SignalError::ValueError(
            "Alpha must be between 0.0 and 1.0".to_string(),
        ));
    }

    if _len_guards(m) {
        return Ok(vec![1.0; m]);
    }

    let (n, needs_trunc) = _extend(m, sym);

    let mut w = Vec::with_capacity(n);
    let taper_length = (alpha * (n - 1) as f64 / 2.0) as usize;

    for i in 0..n {
        let w_val = if i < taper_length {
            // Left taper
            0.5 * (1.0 + (PI * i as f64 / taper_length as f64 - PI).cos())
        } else if i >= n - taper_length {
            // Right taper
            let idx = n - 1 - i;
            0.5 * (1.0 + (PI * idx as f64 / taper_length as f64 - PI).cos())
        } else {
            // Rectangular middle section
            1.0
        };
        w.push(w_val);
    }

    Ok(_truncate(w, needs_trunc))
}

/// Lanczos window (sinc-based)
///
/// The Lanczos window is based on the sinc function and provides
/// good characteristics for interpolation applications.
///
/// # Arguments
/// * `m` - Number of points in the output window
/// * `a` - Support parameter (typically 2 or 3)
/// * `sym` - If true, generates a symmetric window, otherwise a periodic window
///
/// # Returns
/// A Vec<f64> of window values
pub fn lanczos(m: usize, a: f64, sym: bool) -> SignalResult<Vec<f64>> {
    if a <= 0.0 {
        return Err(SignalError::ValueError(
            "Support parameter 'a' must be positive".to_string(),
        ));
    }

    if _len_guards(m) {
        return Ok(vec![1.0; m]);
    }

    let (n, needs_trunc) = _extend(m, sym);
    let center = (n - 1) as f64 / 2.0;

    let mut w = Vec::with_capacity(n);
    for i in 0..n {
        let x = (i as f64 - center) / center * a;

        let w_val = if x.abs() < 1e-10 {
            1.0 // sinc(0) = 1
        } else if x.abs() >= a {
            0.0 // Outside support
        } else {
            let sinc_val = (PI * x).sin() / (PI * x);
            let sinc_scaled = (PI * x / a).sin() / (PI * x / a);
            sinc_val * sinc_scaled
        };
        w.push(w_val);
    }

    Ok(_truncate(w, needs_trunc))
}

/// Generalized exponential window
///
/// Creates an exponential window with customizable parameters for asymmetric
/// or specialized exponential characteristics.
///
/// # Arguments
/// * `m` - Number of points in the output window
/// * `decay_left` - Left side decay rate
/// * `decay_right` - Right side decay rate  
/// * `peak_position` - Peak position (0.0 = left, 0.5 = center, 1.0 = right)
/// * `sym` - If true, generates a symmetric window, otherwise a periodic window
///
/// # Returns
/// A Vec<f64> of window values
pub fn generalized_exponential(
    m: usize,
    decay_left: f64,
    decay_right: f64,
    peak_position: f64,
    sym: bool,
) -> SignalResult<Vec<f64>> {
    if decay_left <= 0.0 || decay_right <= 0.0 {
        return Err(SignalError::ValueError(
            "Decay rates must be positive".to_string(),
        ));
    }

    if !(0.0..=1.0).contains(&peak_position) {
        return Err(SignalError::ValueError(
            "Peak position must be between 0.0 and 1.0".to_string(),
        ));
    }

    if _len_guards(m) {
        return Ok(vec![1.0; m]);
    }

    let (n, needs_trunc) = _extend(m, sym);
    let peak_idx = peak_position * (n - 1) as f64;

    let mut w = Vec::with_capacity(n);
    for i in 0..n {
        let w_val = if (i as f64) <= peak_idx {
            // Left side
            let distance = peak_idx - i as f64;
            (-distance / decay_left).exp()
        } else {
            // Right side
            let distance = i as f64 - peak_idx;
            (-distance / decay_right).exp()
        };
        w.push(w_val);
    }

    Ok(_truncate(w, needs_trunc))
}

/// Analyze exponential window properties
///
/// Computes important properties of exponential windows for analysis
pub fn analyze_exponential_window(window: &[f64]) -> ExponentialWindowAnalysis {
    let n = window.len();

    // Find peak value and position
    let (peak_idx, peak_value) = window
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap_or((n / 2, &1.0));

    // Estimate decay characteristics
    let decay_estimate = estimate_decay_constant(window);

    // Coherent gain
    let coherent_gain = window.iter().sum::<f64>() / n as f64;

    // Processing gain
    let sum_squares = window.iter().map(|&w| w * w).sum::<f64>();
    let processing_gain = window.iter().sum::<f64>().powi(2) / (n as f64 * sum_squares);

    // Effective width (where window drops to 1/e of peak)
    let threshold = peak_value / std::f64::consts::E;
    let effective_width = estimate_effective_width(window, threshold);

    // Classify window type
    let window_type = classify_exponential_window(window);

    ExponentialWindowAnalysis {
        window_type,
        peak_value: *peak_value,
        peak_position: peak_idx,
        decay_constant: decay_estimate,
        coherent_gain,
        processing_gain,
        effective_width,
    }
}

/// Exponential window analysis results
#[derive(Debug, Clone)]
pub struct ExponentialWindowAnalysis {
    /// Detected window type
    pub window_type: ExponentialWindowType,
    /// Peak value of the window
    pub peak_value: f64,
    /// Position of the peak (sample index)
    pub peak_position: usize,
    /// Estimated decay constant
    pub decay_constant: f64,
    /// Coherent gain
    pub coherent_gain: f64,
    /// Processing gain
    pub processing_gain: f64,
    /// Effective width at 1/e of peak
    pub effective_width: f64,
}

/// Types of exponential windows
#[derive(Debug, Clone, PartialEq)]
pub enum ExponentialWindowType {
    /// Pure exponential decay
    Exponential,
    /// Gaussian bell curve
    Gaussian,
    /// Kaiser window (Bessel-based)
    Kaiser,
    /// Tukey window (tapered cosine)
    Tukey,
    /// Lanczos window (sinc-based)
    Lanczos,
    /// Other/unknown exponential variant
    Other,
}

// Helper functions

/// Modified Bessel function of the first kind, order 0
fn modified_bessel_i0(x: f64) -> f64 {
    let t = x / 3.75;
    if x.abs() < 3.75 {
        let y = t * t;
        1.0 + y
            * (3.515623
                + y * (3.089943 + y * (1.20675 + y * (0.265973 + y * (0.0360768 + y * 0.0045813)))))
    } else {
        let y = 1.0 / t;
        (x.abs().exp() / x.abs().sqrt())
            * (0.39894228
                + y * (0.01328592
                    + y * (0.00225319
                        + y * (-0.00157565
                            + y * (0.00916281
                                + y * (-0.02057706
                                    + y * (0.02635537 + y * (-0.01647633 + y * 0.00392377))))))))
    }
}

/// Estimate decay constant from window samples
fn estimate_decay_constant(window: &[f64]) -> f64 {
    let n = window.len();
    if n < 3 {
        return 1.0;
    }

    let peak_idx = window
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(n / 2);

    let peak_value = window[peak_idx];
    let target_value = peak_value / std::f64::consts::E;

    // Find where window drops to 1/e of peak
    for i in (peak_idx + 1)..n {
        if window[i] <= target_value {
            return (i - peak_idx) as f64;
        }
    }

    // If not found, estimate from available samples
    n as f64 / 4.0
}

/// Estimate effective width at specified threshold
fn estimate_effective_width(window: &[f64], threshold: f64) -> f64 {
    let n = window.len();
    let mut left_edge = 0;
    let mut right_edge = n - 1;

    // Find left edge
    for i in 0..n {
        if window[i] >= threshold {
            left_edge = i;
            break;
        }
    }

    // Find right edge
    for i in (0..n).rev() {
        if window[i] >= threshold {
            right_edge = i;
            break;
        }
    }

    (right_edge - left_edge + 1) as f64
}

/// Classify the type of exponential window
fn classify_exponential_window(window: &[f64]) -> ExponentialWindowType {
    let n = window.len();
    if n < 5 {
        return ExponentialWindowType::Other;
    }

    // Check if it's symmetric (Gaussian-like)
    let is_symmetric = is_approximately_symmetric(window);

    if is_symmetric {
        // Could be Gaussian or Kaiser
        if is_gaussian_like(window) {
            ExponentialWindowType::Gaussian
        } else if is_kaiser_like(window) {
            ExponentialWindowType::Kaiser
        } else {
            ExponentialWindowType::Other
        }
    } else {
        // Asymmetric - likely pure exponential or Tukey
        if is_tukey_like(window) {
            ExponentialWindowType::Tukey
        } else if is_lanczos_like(window) {
            ExponentialWindowType::Lanczos
        } else {
            ExponentialWindowType::Exponential
        }
    }
}

/// Check if window is approximately symmetric
fn is_approximately_symmetric(window: &[f64]) -> bool {
    let n = window.len();
    let tolerance = 0.05;

    for i in 0..(n / 2) {
        let left = window[i];
        let right = window[n - 1 - i];
        let relative_error = (left - right).abs() / left.max(right);
        if relative_error > tolerance {
            return false;
        }
    }
    true
}

/// Check if window has Gaussian characteristics
fn is_gaussian_like(window: &[f64]) -> bool {
    let n = window.len();
    let center = n / 2;
    let peak = window[center];

    // Check if it follows Gaussian decay pattern
    let quarter = n / 4;
    if quarter < center {
        let ratio = window[center - quarter] / peak;
        // Gaussian should have specific ratio at quarter distance
        ratio > 0.6 && ratio < 0.9
    } else {
        false
    }
}

/// Check if window has Kaiser characteristics
fn is_kaiser_like(window: &[f64]) -> bool {
    let n = window.len();
    let center = n / 2;
    let peak = window[center];

    // Kaiser windows have characteristic Bessel function shape
    let quarter = n / 4;
    if quarter < center {
        let ratio = window[center - quarter] / peak;
        // Kaiser has different ratio pattern than Gaussian
        ratio > 0.4 && ratio < 0.8
    } else {
        false
    }
}

/// Check if window has Tukey characteristics
fn is_tukey_like(window: &[f64]) -> bool {
    let n = window.len();

    // Tukey should have flat middle section
    let middle_start = n / 3;
    let middle_end = 2 * n / 3;

    if middle_end > middle_start {
        let middle_values: Vec<_> = window[middle_start..middle_end].to_vec();
        let is_flat = middle_values.windows(2).all(|w| (w[0] - w[1]).abs() < 0.01);
        is_flat
    } else {
        false
    }
}

/// Check if window has Lanczos characteristics
fn is_lanczos_like(window: &[f64]) -> bool {
    // Lanczos has zero crossings and sinc-like oscillations
    let zero_crossings = window.windows(2).filter(|w| w[0] * w[1] < 0.0).count();
    zero_crossings > 0 // Has oscillations
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_window() {
        let window = gaussian(11, 1.0, true).unwrap();
        assert_eq!(window.len(), 11);

        // Peak should be at center
        let center = window.len() / 2;
        let peak_idx = window
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;
        assert_eq!(peak_idx, center);

        // Should be symmetric
        assert!((window[0] - window[10]).abs() < 1e-10);
        assert!((window[1] - window[9]).abs() < 1e-10);
    }

    #[test]
    fn test_kaiser_window() {
        let window = kaiser(10, 5.0, true).unwrap();
        assert_eq!(window.len(), 10);

        // Should have peak near center
        let max_val = window.iter().fold(0.0f64, |a, &b| a.max(b));
        assert!(max_val > 0.9);
    }

    #[test]
    fn test_tukey_window() {
        let window = tukey(20, 0.5, true).unwrap();
        assert_eq!(window.len(), 20);

        // Should have flat middle section
        let middle = &window[8..12];
        let is_flat = middle.windows(2).all(|w| (w[0] - w[1]).abs() < 0.1);
        assert!(is_flat);
    }

    #[test]
    fn test_exponential_window() {
        let window = exponential(10, 2.0, true).unwrap();
        assert_eq!(window.len(), 10);

        // Peak should be at center for symmetric version
        let center = window.len() / 2;
        let peak_idx = window
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;
        assert_eq!(peak_idx, center);
    }

    #[test]
    fn test_invalid_parameters() {
        assert!(gaussian(10, -1.0, true).is_err());
        assert!(kaiser(10, -1.0, true).is_err());
        assert!(tukey(10, -0.5, true).is_err());
        assert!(tukey(10, 1.5, true).is_err());
    }
}
