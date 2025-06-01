//! Window functions for signal processing.
//!
//! This module provides various window functions commonly used in signal processing,
//! including Hamming, Hann, Blackman, and others. These windows are useful for
//! reducing spectral leakage in Fourier transforms and filter design.

use crate::error::{SignalError, SignalResult};
use std::f64::consts::PI;

// Import specialized window implementations
mod kaiser;
pub use kaiser::{kaiser, kaiser_bessel_derived};

/// Create a window function of a specified type and length.
///
/// # Arguments
///
/// * `window_type` - Type of window function to create
/// * `length` - Length of the window
/// * `periodic` - If true, the window is periodic, otherwise symmetric
///
/// # Returns
///
/// * Window function of specified type and length
///
/// # Examples
///
/// ```
/// use scirs2_signal::window::get_window;
///
/// // Create a Hamming window of length 10
/// let window = get_window("hamming", 10, false).unwrap();
///
/// assert_eq!(window.len(), 10);
/// assert!(window[0] > 0.0 && window[0] < 1.0);
/// assert!(window[window.len() / 2] > 0.9);
/// ```
pub fn get_window(window_type: &str, length: usize, periodic: bool) -> SignalResult<Vec<f64>> {
    if length == 0 {
        return Err(SignalError::ValueError(
            "Window length must be positive".to_string(),
        ));
    }

    // Dispatch to specific window function
    match window_type.to_lowercase().as_str() {
        "hamming" => hamming(length, !periodic),
        "hanning" | "hann" => hann(length, !periodic),
        "blackman" => blackman(length, !periodic),
        "bartlett" => bartlett(length, !periodic),
        "flattop" => flattop(length, !periodic),
        "boxcar" | "rectangular" => boxcar(length, !periodic),
        "triang" => triang(length, !periodic),
        "bohman" => bohman(length, !periodic),
        "parzen" => parzen(length, !periodic),
        "nuttall" => nuttall(length, !periodic),
        "blackmanharris" => blackmanharris(length, !periodic),
        "cosine" => cosine(length, !periodic),
        "exponential" => exponential(length, None, 1.0, !periodic),
        "tukey" => tukey(length, 0.5, !periodic),
        "barthann" => barthann(length, !periodic),
        "kaiser" => {
            // Default beta value of 8.6 gives sidelobe attenuation of about 60dB
            kaiser(length, 8.6, !periodic)
        }
        "kaiser_bessel_derived" => {
            // Default beta value of 8.6
            kaiser_bessel_derived(length, 8.6, !periodic)
        }
        _ => Err(SignalError::ValueError(format!(
            "Unknown window type: {}",
            window_type
        ))),
    }
}

/// Helper function to handle small or incorrect window lengths
pub(crate) fn _len_guards(m: usize) -> bool {
    // Return true for trivial windows with length 0 or 1
    m <= 1
}

/// Helper function to extend window by 1 sample if needed for DFT-even symmetry
pub(crate) fn _extend(m: usize, sym: bool) -> (usize, bool) {
    if !sym {
        (m + 1, true)
    } else {
        (m, false)
    }
}

/// Helper function to truncate window by 1 sample if needed
pub(crate) fn _truncate(w: Vec<f64>, needed: bool) -> Vec<f64> {
    if needed {
        w[..w.len() - 1].to_vec()
    } else {
        w
    }
}

/// Hamming window.
///
/// The Hamming window is a taper formed by using a raised cosine with
/// non-zero endpoints.
///
/// # Arguments
///
/// * `m` - Number of points in the output window
/// * `sym` - If true, generates a symmetric window, otherwise a periodic window
///
/// # Returns
///
/// A Vec<f64> of window values
///
/// # Examples
///
/// ```
/// use scirs2_signal::window::hamming;
///
/// let window = hamming(10, true).unwrap();
/// assert_eq!(window.len(), 10);
/// ```
pub fn hamming(m: usize, sym: bool) -> SignalResult<Vec<f64>> {
    if _len_guards(m) {
        return Ok(vec![1.0; m]);
    }

    let (n, needs_trunc) = _extend(m, sym);

    let mut w = Vec::with_capacity(n);
    for i in 0..n {
        let w_val = 0.54 - 0.46 * (2.0 * PI * i as f64 / (n - 1) as f64).cos();
        w.push(w_val);
    }

    Ok(_truncate(w, needs_trunc))
}

/// Hann window.
///
/// The Hann window is a taper formed by using a raised cosine.
///
/// # Arguments
///
/// * `m` - Number of points in the output window
/// * `sym` - If true, generates a symmetric window, otherwise a periodic window
///
/// # Returns
///
/// A Vec<f64> of window values
///
/// # Examples
///
/// ```
/// use scirs2_signal::window::hann;
///
/// let window = hann(10, true).unwrap();
/// assert_eq!(window.len(), 10);
/// ```
pub fn hann(m: usize, sym: bool) -> SignalResult<Vec<f64>> {
    if _len_guards(m) {
        return Ok(vec![1.0; m]);
    }

    let (n, needs_trunc) = _extend(m, sym);

    let mut w = Vec::with_capacity(n);
    for i in 0..n {
        let w_val = 0.5 * (1.0 - (2.0 * PI * i as f64 / (n - 1) as f64).cos());
        w.push(w_val);
    }

    Ok(_truncate(w, needs_trunc))
}

/// Blackman window.
///
/// The Blackman window is a taper formed by using the first three terms of
/// a summation of cosines.
///
/// # Arguments
///
/// * `m` - Number of points in the output window
/// * `sym` - If true, generates a symmetric window, otherwise a periodic window
///
/// # Returns
///
/// A Vec<f64> of window values
///
/// # Examples
///
/// ```
/// use scirs2_signal::window::blackman;
///
/// let window = blackman(10, true).unwrap();
/// assert_eq!(window.len(), 10);
/// ```
pub fn blackman(m: usize, sym: bool) -> SignalResult<Vec<f64>> {
    if _len_guards(m) {
        return Ok(vec![1.0; m]);
    }

    let (n, needs_trunc) = _extend(m, sym);

    let mut w = Vec::with_capacity(n);
    for i in 0..n {
        let w_val = 0.42 - 0.5 * (2.0 * PI * i as f64 / (n - 1) as f64).cos()
            + 0.08 * (4.0 * PI * i as f64 / (n - 1) as f64).cos();
        w.push(w_val);
    }

    Ok(_truncate(w, needs_trunc))
}

/// Bartlett window.
///
/// The Bartlett window is a triangular window that is the convolution of two rectangular windows.
///
/// # Arguments
///
/// * `m` - Number of points in the output window
/// * `sym` - If true, generates a symmetric window, otherwise a periodic window
///
/// # Returns
///
/// A Vec<f64> of window values
///
/// # Examples
///
/// ```
/// use scirs2_signal::window::bartlett;
///
/// let window = bartlett(10, true).unwrap();
/// assert_eq!(window.len(), 10);
/// ```
pub fn bartlett(m: usize, sym: bool) -> SignalResult<Vec<f64>> {
    if _len_guards(m) {
        return Ok(vec![1.0; m]);
    }

    let (n, needs_trunc) = _extend(m, sym);

    let mut w = Vec::with_capacity(n);
    let m2 = (n - 1) as f64 / 2.0;
    for i in 0..n {
        let w_val = 1.0 - ((i as f64 - m2) / m2).abs();
        w.push(w_val);
    }

    Ok(_truncate(w, needs_trunc))
}

/// Triangular window (slightly different from Bartlett).
///
/// # Arguments
///
/// * `m` - Number of points in the output window
/// * `sym` - If true, generates a symmetric window, otherwise a periodic window
///
/// # Returns
///
/// A Vec<f64> of window values
///
/// # Examples
///
/// ```
/// use scirs2_signal::window::triang;
///
/// let window = triang(10, true).unwrap();
/// assert_eq!(window.len(), 10);
/// ```
pub fn triang(m: usize, sym: bool) -> SignalResult<Vec<f64>> {
    if _len_guards(m) {
        return Ok(vec![1.0; m]);
    }

    let (n, needs_trunc) = _extend(m, sym);

    let mut w = Vec::with_capacity(n);
    let m2 = (n as f64 - 1.0) / 2.0;
    for i in 0..n {
        let w_val = 1.0 - ((i as f64 - m2) / (m2 + 1.0)).abs();
        w.push(w_val);
    }

    Ok(_truncate(w, needs_trunc))
}

/// Flat top window.
///
/// The flat top window is a taper formed by using a weighted sum of cosine functions.
/// This window has the best amplitude flatness in the frequency domain.
///
/// # Arguments
///
/// * `m` - Number of points in the output window
/// * `sym` - If true, generates a symmetric window, otherwise a periodic window
///
/// # Returns
///
/// A Vec<f64> of window values
///
/// # Examples
///
/// ```
/// use scirs2_signal::window::flattop;
///
/// let window = flattop(10, true).unwrap();
/// assert_eq!(window.len(), 10);
/// ```
pub fn flattop(m: usize, sym: bool) -> SignalResult<Vec<f64>> {
    if _len_guards(m) {
        return Ok(vec![1.0; m]);
    }

    let (n, needs_trunc) = _extend(m, sym);

    let mut w = Vec::with_capacity(n);
    let a0 = 0.21557895;
    let a1 = 0.41663158;
    let a2 = 0.277263158;
    let a3 = 0.083578947;
    let a4 = 0.006947368;

    for i in 0..n {
        let w_val = a0 - a1 * (2.0 * PI * i as f64 / (n - 1) as f64).cos()
            + a2 * (4.0 * PI * i as f64 / (n - 1) as f64).cos()
            - a3 * (6.0 * PI * i as f64 / (n - 1) as f64).cos()
            + a4 * (8.0 * PI * i as f64 / (n - 1) as f64).cos();
        w.push(w_val);
    }

    Ok(_truncate(w, needs_trunc))
}

/// Rectangular window.
///
/// The rectangular window is the simplest window, equivalent to replacing all frame samples by a constant.
///
/// # Arguments
///
/// * `m` - Number of points in the output window
/// * `sym` - If true, generates a symmetric window, otherwise a periodic window
///
/// # Returns
///
/// A Vec<f64> of window values
///
/// # Examples
///
/// ```
/// use scirs2_signal::window::boxcar;
///
/// let window = boxcar(10, true).unwrap();
/// assert_eq!(window.len(), 10);
/// assert_eq!(window[0], 1.0);
/// ```
pub fn boxcar(m: usize, sym: bool) -> SignalResult<Vec<f64>> {
    if _len_guards(m) {
        return Ok(vec![1.0; m]);
    }

    let (n, needs_trunc) = _extend(m, sym);
    let w = vec![1.0; n];
    Ok(_truncate(w, needs_trunc))
}

/// Bohman window.
///
/// The Bohman window is the product of a cosine and a sinc function.
///
/// # Arguments
///
/// * `m` - Number of points in the output window
/// * `sym` - If true, generates a symmetric window, otherwise a periodic window
///
/// # Returns
///
/// A Vec<f64> of window values
///
/// # Examples
///
/// ```
/// use scirs2_signal::window::bohman;
///
/// let window = bohman(10, true).unwrap();
/// assert_eq!(window.len(), 10);
/// ```
pub fn bohman(m: usize, sym: bool) -> SignalResult<Vec<f64>> {
    if _len_guards(m) {
        return Ok(vec![1.0; m]);
    }

    let (n, needs_trunc) = _extend(m, sym);

    let mut w = Vec::with_capacity(n);
    for i in 0..n {
        let x = 2.0 * i as f64 / (n - 1) as f64 - 1.0;
        let x_abs = x.abs();
        let w_val = if x_abs <= 1.0 {
            (1.0 - x_abs) * (PI * x_abs).cos() + PI.recip() * (PI * x_abs).sin()
        } else {
            0.0
        };
        w.push(w_val);
    }

    Ok(_truncate(w, needs_trunc))
}

/// Parzen window.
///
/// The Parzen window is a piecewise cubic approximation of the Gaussian window.
///
/// # Arguments
///
/// * `m` - Number of points in the output window
/// * `sym` - If true, generates a symmetric window, otherwise a periodic window
///
/// # Returns
///
/// A Vec<f64> of window values
///
/// # Examples
///
/// ```
/// use scirs2_signal::window::parzen;
///
/// let window = parzen(10, true).unwrap();
/// assert_eq!(window.len(), 10);
/// ```
pub fn parzen(m: usize, sym: bool) -> SignalResult<Vec<f64>> {
    if _len_guards(m) {
        return Ok(vec![1.0; m]);
    }

    let (n, needs_trunc) = _extend(m, sym);

    let mut w = Vec::with_capacity(n);
    let n1 = (n - 1) as f64;

    for i in 0..n {
        let x = 2.0 * i as f64 / n1 - 1.0;
        let x_abs = x.abs();

        let w_val = if x_abs <= 0.5 {
            1.0 - 6.0 * x_abs.powi(2) + 6.0 * x_abs.powi(3)
        } else if x_abs <= 1.0 {
            2.0 * (1.0 - x_abs).powi(3)
        } else {
            0.0
        };

        w.push(w_val);
    }

    Ok(_truncate(w, needs_trunc))
}

/// Nuttall window.
///
/// The Nuttall window is a minimal 4-term Blackman-Harris window.
///
/// # Arguments
///
/// * `m` - Number of points in the output window
/// * `sym` - If true, generates a symmetric window, otherwise a periodic window
///
/// # Returns
///
/// A Vec<f64> of window values
///
/// # Examples
///
/// ```
/// use scirs2_signal::window::nuttall;
///
/// let window = nuttall(10, true).unwrap();
/// assert_eq!(window.len(), 10);
/// ```
pub fn nuttall(m: usize, sym: bool) -> SignalResult<Vec<f64>> {
    if _len_guards(m) {
        return Ok(vec![1.0; m]);
    }

    let (n, needs_trunc) = _extend(m, sym);

    let mut w = Vec::with_capacity(n);
    let a0 = 0.3635819;
    let a1 = 0.4891775;
    let a2 = 0.1365995;
    let a3 = 0.0106411;

    for i in 0..n {
        let w_val = a0 - a1 * (2.0 * PI * i as f64 / (n - 1) as f64).cos()
            + a2 * (4.0 * PI * i as f64 / (n - 1) as f64).cos()
            - a3 * (6.0 * PI * i as f64 / (n - 1) as f64).cos();
        w.push(w_val);
    }

    Ok(_truncate(w, needs_trunc))
}

/// Blackman-Harris window.
///
/// The Blackman-Harris window is a taper formed by using the first four terms of a
/// summation of cosines.
///
/// # Arguments
///
/// * `m` - Number of points in the output window
/// * `sym` - If true, generates a symmetric window, otherwise a periodic window
///
/// # Returns
///
/// A Vec<f64> of window values
///
/// # Examples
///
/// ```
/// use scirs2_signal::window::blackmanharris;
///
/// let window = blackmanharris(10, true).unwrap();
/// assert_eq!(window.len(), 10);
/// ```
pub fn blackmanharris(m: usize, sym: bool) -> SignalResult<Vec<f64>> {
    if _len_guards(m) {
        return Ok(vec![1.0; m]);
    }

    let (n, needs_trunc) = _extend(m, sym);

    let mut w = Vec::with_capacity(n);
    let a0 = 0.35875;
    let a1 = 0.48829;
    let a2 = 0.14128;
    let a3 = 0.01168;

    for i in 0..n {
        let w_val = a0 - a1 * (2.0 * PI * i as f64 / (n - 1) as f64).cos()
            + a2 * (4.0 * PI * i as f64 / (n - 1) as f64).cos()
            - a3 * (6.0 * PI * i as f64 / (n - 1) as f64).cos();
        w.push(w_val);
    }

    Ok(_truncate(w, needs_trunc))
}

/// Cosine window.
///
/// Also known as the sine window, half-cosine, or half-sine window.
///
/// # Arguments
///
/// * `m` - Number of points in the output window
/// * `sym` - If true, generates a symmetric window, otherwise a periodic window
///
/// # Returns
///
/// A Vec<f64> of window values
///
/// # Examples
///
/// ```
/// use scirs2_signal::window::cosine;
///
/// let window = cosine(10, true).unwrap();
/// assert_eq!(window.len(), 10);
/// ```
pub fn cosine(m: usize, sym: bool) -> SignalResult<Vec<f64>> {
    if _len_guards(m) {
        return Ok(vec![1.0; m]);
    }

    let (n, needs_trunc) = _extend(m, sym);

    let mut w = Vec::with_capacity(n);
    for i in 0..n {
        let w_val = (PI * i as f64 / (n - 1) as f64).sin();
        w.push(w_val);
    }

    Ok(_truncate(w, needs_trunc))
}

/// Exponential window.
///
/// # Arguments
///
/// * `m` - Number of points in the output window
/// * `center` - Optional parameter defining the center point of the window, default is None (m/2)
/// * `tau` - Parameter defining the decay rate
/// * `sym` - If true, generates a symmetric window, otherwise a periodic window
///
/// # Returns
///
/// A Vec<f64> of window values
///
/// # Examples
///
/// ```
/// use scirs2_signal::window::exponential;
///
/// let window = exponential(10, None, 1.0, true).unwrap();
/// assert_eq!(window.len(), 10);
/// ```
pub fn exponential(m: usize, center: Option<f64>, tau: f64, sym: bool) -> SignalResult<Vec<f64>> {
    if _len_guards(m) {
        return Ok(vec![1.0; m]);
    }

    let (n, needs_trunc) = _extend(m, sym);

    let center_val = center.unwrap_or(((n - 1) as f64) / 2.0);

    let mut w = Vec::with_capacity(n);
    for i in 0..n {
        let w_val = (-((i as f64 - center_val).abs() / tau)).exp();
        w.push(w_val);
    }

    Ok(_truncate(w, needs_trunc))
}

/// Tukey window.
///
/// The Tukey window, also known as the cosine-tapered window, is a window
/// with a flat middle section and cosine tapered ends.
///
/// # Arguments
///
/// * `m` - Number of points in the output window
/// * `alpha` - Shape parameter of the Tukey window, representing the ratio of
///   cosine-tapered section length to the total window length
/// * `sym` - If true, generates a symmetric window, otherwise a periodic window
///
/// # Returns
///
/// A Vec<f64> of window values
///
/// # Examples
///
/// ```
/// use scirs2_signal::window::tukey;
///
/// let window = tukey(10, 0.5, true).unwrap();
/// assert_eq!(window.len(), 10);
/// ```
pub fn tukey(m: usize, alpha: f64, sym: bool) -> SignalResult<Vec<f64>> {
    if _len_guards(m) {
        return Ok(vec![1.0; m]);
    }

    let alpha = alpha.clamp(0.0, 1.0);

    if alpha == 0.0 {
        return boxcar(m, sym);
    }

    if alpha == 1.0 {
        return hann(m, sym);
    }

    let (n, needs_trunc) = _extend(m, sym);

    let mut w = Vec::with_capacity(n);
    let width = (alpha * (n - 1) as f64 / 2.0).floor() as usize;
    let width = width.max(1); // Ensure width is at least 1

    for i in 0..n {
        let w_val = if i < width {
            0.5 * (1.0 + (PI * (-1.0 + 2.0 * i as f64 / (alpha * (n - 1) as f64))).cos())
        } else if i >= n - width {
            0.5 * (1.0
                + (PI * (-2.0 / alpha + 1.0 + 2.0 * i as f64 / (alpha * (n - 1) as f64))).cos())
        } else {
            1.0
        };
        w.push(w_val);
    }

    Ok(_truncate(w, needs_trunc))
}

/// Barthann window.
///
/// The Barthann window is the product of a Bartlett window and a Hann window.
///
/// # Arguments
///
/// * `m` - Number of points in the output window
/// * `sym` - If true, generates a symmetric window, otherwise a periodic window
///
/// # Returns
///
/// A Vec<f64> of window values
///
/// # Examples
///
/// ```
/// use scirs2_signal::window::barthann;
///
/// let window = barthann(10, true).unwrap();
/// assert_eq!(window.len(), 10);
/// ```
pub fn barthann(m: usize, sym: bool) -> SignalResult<Vec<f64>> {
    if _len_guards(m) {
        return Ok(vec![1.0; m]);
    }

    let (n, needs_trunc) = _extend(m, sym);

    let mut w = Vec::with_capacity(n);
    for i in 0..n {
        let fac = (i as f64) / (n - 1) as f64;
        let w_val = 0.62 - 0.48 * (fac * 2.0 - 1.0).abs() - 0.38 * (2.0 * PI * fac).cos();
        w.push(w_val);
    }

    Ok(_truncate(w, needs_trunc))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_hamming_window() {
        let window = hamming(10, true).unwrap();
        assert_eq!(window.len(), 10);

        // Test symmetry
        for i in 0..5 {
            assert_relative_eq!(window[i], window[9 - i], epsilon = 1e-10);
        }

        // Test specific values
        assert_relative_eq!(window[0], 0.08, epsilon = 0.01);
        // The peak is at indices 4 and 5 for a 10-point symmetric window
        assert!(window[4] > 0.95);
        assert!(window[5] > 0.95);
    }

    #[test]
    fn test_hann_window() {
        let window = hann(10, true).unwrap();
        assert_eq!(window.len(), 10);

        // Test symmetry
        for i in 0..5 {
            assert_relative_eq!(window[i], window[9 - i], epsilon = 1e-10);
        }

        // Test specific values
        assert_relative_eq!(window[0], 0.0, epsilon = 0.01);
        // The peak is at indices 4 and 5 for a 10-point symmetric window
        assert!(window[4] > 0.95);
        assert!(window[5] > 0.95);
    }

    #[test]
    fn test_blackman_window() {
        let window = blackman(10, true).unwrap();
        assert_eq!(window.len(), 10);

        // Test symmetry
        for i in 0..5 {
            assert_relative_eq!(window[i], window[9 - i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_bartlett_window() {
        let window = bartlett(10, true).unwrap();
        assert_eq!(window.len(), 10);

        // Test symmetry
        for i in 0..5 {
            assert_relative_eq!(window[i], window[9 - i], epsilon = 1e-10);
        }

        // Test endpoints - Bartlett window has zero at endpoints
        assert_relative_eq!(window[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(window[9], 0.0, epsilon = 1e-10);

        // Test that it increases from start to middle
        assert!(window[1] > window[0]);
        assert!(window[2] > window[1]);

        // Test middle values are close to 1
        let mid_val = window[4];
        assert!(mid_val > 0.8 && mid_val <= 1.0);
    }

    #[test]
    fn test_flattop_window() {
        let window = flattop(10, true).unwrap();
        assert_eq!(window.len(), 10);

        // Test symmetry
        for i in 0..5 {
            assert_relative_eq!(window[i], window[9 - i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_boxcar_window() {
        let window = boxcar(10, true).unwrap();
        assert_eq!(window.len(), 10);

        // Test all values are 1.0
        for val in window {
            assert_relative_eq!(val, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_bohman_window() {
        let window = bohman(10, true).unwrap();
        assert_eq!(window.len(), 10);

        // Test symmetry
        for i in 0..5 {
            assert_relative_eq!(window[i], window[9 - i], epsilon = 1e-10);
        }

        // Test endpoints
        assert_relative_eq!(window[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(window[9], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_triang_window() {
        let window = triang(10, true).unwrap();
        assert_eq!(window.len(), 10);

        // Test symmetry
        for i in 0..5 {
            assert_relative_eq!(window[i], window[9 - i], epsilon = 1e-10);
        }
    }
}
