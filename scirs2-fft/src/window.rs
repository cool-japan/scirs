//! Window functions for signal processing
//!
//! This module provides various window functions commonly used in signal processing
//! to reduce spectral leakage when performing Fourier transforms on finite data.
//!
//! Window functions are used to taper the samples at the beginning and end of the
//! data to reduce the spectral leakage effect in FFT processing.

use crate::error::{FFTError, FFTResult};
use scirs2_core::ndarray::{Array1, ArrayBase, Data, Ix1};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::f64::consts::PI;
use std::fmt::Debug;
use std::str::FromStr;

/// Window function types
#[derive(Debug, Clone, PartialEq)]
pub enum Window {
    /// Rectangular window (no windowing)
    Rectangular,
    /// Hann window (raised cosine)
    Hann,
    /// Hamming window (raised cosine with offset)
    Hamming,
    /// Blackman window
    Blackman,
    /// Bartlett window (triangular)
    Bartlett,
    /// Flat top window
    FlatTop,
    /// Parzen window
    Parzen,
    /// Bohman window
    Bohman,
    /// Blackman-Harris window
    BlackmanHarris,
    /// Nuttall window
    Nuttall,
    /// Barthann window
    Barthann,
    /// Cosine window
    Cosine,
    /// Exponential window
    Exponential,
    /// Tukey window (tapered cosine)
    Tukey(f64),
    /// Kaiser window
    Kaiser(f64),
    /// Gaussian window
    Gaussian(f64),
    /// General cosine window with custom coefficients
    GeneralCosine(Vec<f64>),
}

impl FromStr for Window {
    type Err = FFTError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "rectangular" | "boxcar" | "rect" => Ok(Window::Rectangular),
            "hann" | "hanning" => Ok(Window::Hann),
            "hamming" => Ok(Window::Hamming),
            "blackman" => Ok(Window::Blackman),
            "bartlett" | "triangular" | "triangle" => Ok(Window::Bartlett),
            "flattop" | "flat" => Ok(Window::FlatTop),
            "parzen" => Ok(Window::Parzen),
            "bohman" => Ok(Window::Bohman),
            "blackmanharris" | "blackman-harris" => Ok(Window::BlackmanHarris),
            "nuttall" => Ok(Window::Nuttall),
            "barthann" => Ok(Window::Barthann),
            "cosine" | "cos" => Ok(Window::Cosine),
            "exponential" | "exp" => Ok(Window::Exponential),
            _ => Err(FFTError::ValueError(format!("Unknown window type: {s}"))),
        }
    }
}

/// Get a window of the specified type and length.
///
/// # Arguments
///
/// * `window` - Window type or name
/// * `n` - Window length
/// * `sym` - Whether the window is symmetric (default: true)
///
/// # Returns
///
/// A vector containing the window values.
///
/// # Examples
///
/// ```
/// use scirs2_fft::window::{Window, get_window};
///
/// // Create a Hann window
/// let win = get_window(Window::Hann, 10, true).unwrap();
/// assert_eq!(win.len(), 10);
///
/// // Create a Kaiser window with beta=8.6
/// let win = get_window(Window::Kaiser(8.6), 10, true).unwrap();
/// assert_eq!(win.len(), 10);
///
/// // Create a window by name
/// let win = get_window("hamming", 10, true).unwrap();
/// assert_eq!(win.len(), 10);
/// ```
#[allow(dead_code)]
pub fn get_window<T>(window: T, n: usize, sym: bool) -> FFTResult<Array1<f64>>
where
    T: Into<WindowParam>,
{
    if n == 0 {
        return Err(FFTError::ValueError(
            "Window length must be positive".to_string(),
        ));
    }

    let window_param = window.into();
    let window_type = match window_param {
        WindowParam::Type(wt) => wt,
        WindowParam::Name(s) => Window::from_str(&s)?,
    };

    match window_type {
        Window::Rectangular => rectangular(n),
        Window::Hann => hann(n, sym),
        Window::Hamming => hamming(n, sym),
        Window::Blackman => blackman(n, sym),
        Window::Bartlett => bartlett(n, sym),
        Window::FlatTop => flattop(n, sym),
        Window::Parzen => parzen(n, sym),
        Window::Bohman => bohman(n),
        Window::BlackmanHarris => blackmanharris(n, sym),
        Window::Nuttall => nuttall(n, sym),
        Window::Barthann => barthann(n, sym),
        Window::Cosine => cosine(n, sym),
        Window::Exponential => exponential(n, sym, 1.0),
        Window::Tukey(alpha) => tukey(n, sym, alpha),
        Window::Kaiser(beta) => kaiser(n, sym, beta),
        Window::Gaussian(std) => gaussian(n, sym, std),
        Window::GeneralCosine(coeffs) => general_cosine(n, sym, &coeffs),
    }
}

/// Helper enum to handle different window parameter types
#[derive(Debug)]
pub enum WindowParam {
    /// Window type enum
    Type(Window),
    /// Window name as string
    Name(String),
}

impl From<Window> for WindowParam {
    fn from(window: Window) -> Self {
        WindowParam::Type(window)
    }
}

impl From<&str> for WindowParam {
    fn from(s: &str) -> Self {
        WindowParam::Name(s.to_string())
    }
}

impl From<String> for WindowParam {
    fn from(s: String) -> Self {
        WindowParam::Name(s)
    }
}

/// Rectangular window
///
/// A rectangular window is a window with constant value 1.
#[allow(dead_code)]
fn rectangular(n: usize) -> FFTResult<Array1<f64>> {
    Ok(Array1::ones(n))
}

/// Hann window
///
/// The Hann window is a taper formed by using a raised cosine with ends that
/// touch zero.
#[allow(dead_code)]
fn hann(n: usize, sym: bool) -> FFTResult<Array1<f64>> {
    general_cosine(n, sym, &[0.5, 0.5])
}

/// Hamming window
///
/// The Hamming window is a taper formed by using a raised cosine with non-zero
/// endpoints.
#[allow(dead_code)]
fn hamming(n: usize, sym: bool) -> FFTResult<Array1<f64>> {
    general_cosine(n, sym, &[0.54, 0.46])
}

/// Blackman window
///
/// The Blackman window is a taper formed by using the first three terms of
/// a summation of cosines.
#[allow(dead_code)]
fn blackman(n: usize, sym: bool) -> FFTResult<Array1<f64>> {
    general_cosine(n, sym, &[0.42, 0.5, 0.08])
}

/// Bartlett window
///
/// The Bartlett window is a triangular window that touches zero at both ends.
#[allow(dead_code)]
fn bartlett(n: usize, sym: bool) -> FFTResult<Array1<f64>> {
    if n == 1 {
        return Ok(Array1::ones(1));
    }

    let mut n = n;
    if !sym {
        n += 1;
    }

    let mut w = Array1::zeros(n);
    let range: Vec<f64> = (0..n).map(|i| i as f64).collect();

    for (i, &x) in range.iter().enumerate() {
        if x < (n as f64) / 2.0 {
            w[i] = 2.0 * x / (n as f64 - 1.0);
        } else {
            w[i] = 2.0 - 2.0 * x / (n as f64 - 1.0);
        }
    }

    if !sym {
        // Remove last element for asymmetric case
        let w_slice = w.slice(scirs2_core::ndarray::s![0..n - 1]).to_owned();
        Ok(w_slice)
    } else {
        Ok(w)
    }
}

/// Flat top window
///
/// The flat top window is a window with the main lobe widened and flattened
/// at the expense of higher sidelobes compared to other windows.
#[allow(dead_code)]
fn flattop(n: usize, sym: bool) -> FFTResult<Array1<f64>> {
    general_cosine(
        n,
        sym,
        &[
            0.215_578_95,
            0.416_631_58,
            0.277_263_158,
            0.083_578_947,
            0.006_947_368,
        ],
    )
}

/// Parzen window
///
/// The Parzen window is a piecewise cubic approximation of the Gaussian window.
#[allow(dead_code)]
fn parzen(n: usize, sym: bool) -> FFTResult<Array1<f64>> {
    if n == 1 {
        return Ok(Array1::ones(1));
    }

    let mut n = n;
    if !sym {
        n += 1;
    }

    let mut w = Array1::zeros(n);
    let half_n = (n as f64) / 2.0;

    for i in 0..n {
        let x = (i as f64 - half_n + 0.5).abs() / half_n;

        if x <= 0.5 {
            w[i] = 1.0 - 6.0 * x.powi(2) * (1.0 - x);
        } else if x <= 1.0 {
            w[i] = 2.0 * (1.0 - x).powi(3);
        }
    }

    if !sym {
        // Remove last element for asymmetric case
        let w_slice = w.slice(scirs2_core::ndarray::s![0..n - 1]).to_owned();
        Ok(w_slice)
    } else {
        Ok(w)
    }
}

/// Bohman window
///
/// The Bohman window is the convolution of two half-duration cosine lobes.
#[allow(dead_code)]
fn bohman(n: usize) -> FFTResult<Array1<f64>> {
    if n == 1 {
        return Ok(Array1::ones(1));
    }

    let mut w = Array1::zeros(n);
    let half_n = (n as f64 - 1.0) / 2.0;

    for i in 0..n {
        let x = ((i as f64) - half_n).abs() / half_n;
        if x <= 1.0 {
            w[i] = (1.0 - x) * (PI * x).cos() + (PI * x).sin() / PI;
        }
    }

    Ok(w)
}

/// Blackman-Harris window
///
/// The Blackman-Harris window is a variation of the Blackman window with
/// better sidelobe suppression.
#[allow(dead_code)]
fn blackmanharris(n: usize, sym: bool) -> FFTResult<Array1<f64>> {
    general_cosine(n, sym, &[0.35875, 0.48829, 0.14128, 0.01168])
}

/// Nuttall window
///
/// The Nuttall window is a Blackman-Harris window with slightly different
/// coefficients for improved sidelobe behavior.
#[allow(dead_code)]
fn nuttall(n: usize, sym: bool) -> FFTResult<Array1<f64>> {
    general_cosine(
        n,
        sym,
        &[0.363_581_9, 0.489_177_5, 0.136_599_5, 0.010_641_1],
    )
}

/// Barthann window
///
/// The Barthann window is the convolution of the Bartlett and Hann windows.
#[allow(dead_code)]
fn barthann(n: usize, sym: bool) -> FFTResult<Array1<f64>> {
    if n == 1 {
        return Ok(Array1::ones(1));
    }

    let mut n = n;
    if !sym {
        n += 1;
    }

    let mut w = Array1::zeros(n);
    let fac = 1.0 / (n as f64 - 1.0);

    for i in 0..n {
        let x = i as f64 * fac;
        w[i] = 0.62 - 0.48 * (2.0 * x - 1.0).abs() + 0.38 * (2.0 * PI * (2.0 * x - 1.0)).cos();
    }

    if !sym {
        // Remove last element for asymmetric case
        let w_slice = w.slice(scirs2_core::ndarray::s![0..n - 1]).to_owned();
        Ok(w_slice)
    } else {
        Ok(w)
    }
}

/// Cosine window
///
/// Simple cosine (half-cycle) window.
#[allow(dead_code)]
fn cosine(n: usize, sym: bool) -> FFTResult<Array1<f64>> {
    if n == 1 {
        return Ok(Array1::ones(1));
    }

    let mut w = Array1::zeros(n);

    let range: Vec<f64> = if sym {
        (0..n).map(|i| i as f64 / (n as f64 - 1.0)).collect()
    } else {
        (0..n).map(|i| i as f64 / n as f64).collect()
    };

    for (i, &x) in range.iter().enumerate() {
        w[i] = (PI * x).sin();
    }

    Ok(w)
}

/// Exponential window
///
/// The exponential window has the form: w(n) = exp(-|n - center| / tau)
///
/// # Arguments
///
/// * `n` - Window length
/// * `sym` - Whether the window is symmetric
/// * `tau` - Decay constant (tau > 0)
#[allow(dead_code)]
fn exponential(n: usize, sym: bool, tau: f64) -> FFTResult<Array1<f64>> {
    if tau <= 0.0 {
        return Err(FFTError::ValueError("tau must be positive".to_string()));
    }

    if n == 1 {
        return Ok(Array1::ones(1));
    }

    let center = if sym { (n as f64 - 1.0) / 2.0 } else { 0.0 };

    let mut w = Array1::zeros(n);

    for i in 0..n {
        let x = (i as f64 - center).abs() / (tau * (n as f64));
        w[i] = (-x).exp();
    }

    Ok(w)
}

/// Tukey window
///
/// The Tukey window is a cosine lobe of width alpha * N/2 with tails that are
/// rectangular windows.
///
/// # Arguments
///
/// * `n` - Window length
/// * `sym` - Whether the window is symmetric
/// * `alpha` - Shape parameter of the Tukey window, representing the ratio of cosine-tapered
///   section length to the total window length (0 <= alpha <= 1)
#[allow(dead_code)]
fn tukey(n: usize, sym: bool, alpha: f64) -> FFTResult<Array1<f64>> {
    if !(0.0..=1.0).contains(&alpha) {
        return Err(FFTError::ValueError(
            "alpha must be between 0 and 1".to_string(),
        ));
    }

    if n == 1 {
        return Ok(Array1::ones(1));
    }

    if alpha == 0.0 {
        return rectangular(n);
    }

    if alpha == 1.0 {
        return hann(n, sym);
    }

    let mut w = Array1::ones(n);
    let width = (alpha * (n as f64 - 1.0) / 2.0).floor() as usize;

    // Left taper
    for i in 0..width {
        let x = 0.5 * (1.0 + ((PI * i as f64) / width as f64).cos());
        w[i] = x;
    }

    // Right taper
    for i in 0..width {
        let idx = n - 1 - i;
        let x = 0.5 * (1.0 + ((PI * i as f64) / width as f64).cos());
        w[idx] = x;
    }

    Ok(w)
}

/// Kaiser window
///
/// The Kaiser window is a taper formed by using a Bessel function of the first kind.
///
/// # Arguments
///
/// * `n` - Window length
/// * `sym` - Whether the window is symmetric
/// * `beta` - Shape parameter for Kaiser window
#[allow(dead_code)]
fn kaiser(n: usize, sym: bool, beta: f64) -> FFTResult<Array1<f64>> {
    if n == 1 {
        return Ok(Array1::ones(1));
    }

    if beta < 0.0 {
        return Err(FFTError::ValueError(
            "beta must be non-negative".to_string(),
        ));
    }

    let mut n = n;
    if !sym {
        n += 1;
    }

    let mut w = Array1::zeros(n);
    let alpha = 0.5 * (n as f64 - 1.0);
    let i0_beta = bessel_i0(beta);

    for i in 0..n {
        let x = beta * (1.0 - ((i as f64 - alpha) / alpha).powi(2)).sqrt();
        w[i] = bessel_i0(x) / i0_beta;
    }

    if !sym {
        // Remove last element for asymmetric case
        let w_slice = w.slice(scirs2_core::ndarray::s![0..n - 1]).to_owned();
        Ok(w_slice)
    } else {
        Ok(w)
    }
}

/// Gaussian window
///
/// The Gaussian window is defined as a Gaussian function with specified standard deviation.
///
/// # Arguments
///
/// * `n` - Window length
/// * `sym` - Whether the window is symmetric
/// * `std` - Standard deviation of the Gaussian window
#[allow(dead_code)]
fn gaussian(n: usize, sym: bool, std: f64) -> FFTResult<Array1<f64>> {
    if n == 1 {
        return Ok(Array1::ones(1));
    }

    if std <= 0.0 {
        return Err(FFTError::ValueError("std must be positive".to_string()));
    }

    let mut w = Array1::zeros(n);
    let center = if sym { (n as f64 - 1.0) / 2.0 } else { 0.0 };

    for i in 0..n {
        let x = (i as f64 - center) / (std * (n as f64 - 1.0) / 2.0);
        w[i] = (-0.5 * x.powi(2)).exp();
    }

    Ok(w)
}

/// General cosine window
///
/// General cosine window with custom coefficients.
///
/// # Arguments
///
/// * `n` - Window length
/// * `sym` - Whether the window is symmetric
/// * `a` - Window coefficients in the form: w(n) = a₀ - a₁*cos(2πn/N) + a₂*cos(4πn/N) - ...
#[allow(dead_code)]
fn general_cosine(n: usize, sym: bool, a: &[f64]) -> FFTResult<Array1<f64>> {
    if n == 1 {
        return Ok(Array1::ones(1));
    }

    let mut w = Array1::zeros(n);

    // Calculate window values
    let fac = if sym {
        2.0 * PI / (n as f64 - 1.0)
    } else {
        2.0 * PI / n as f64
    };

    for i in 0..n {
        let mut win_val = a[0];

        for (k, &coef) in a.iter().enumerate().skip(1) {
            let sign = if k % 2 == 1 { -1.0 } else { 1.0 };
            win_val += sign * coef * ((k as f64) * fac * (i as f64)).cos();
        }

        w[i] = win_val;
    }

    Ok(w)
}

/// Modified Bessel function of the first kind, order 0
///
/// Approximation of the modified Bessel function I₀(x) using polynomial
/// approximation for numerical stability.
#[allow(dead_code)]
fn bessel_i0(x: f64) -> f64 {
    let ax = x.abs();

    // For small arguments, use polynomial approximation
    if ax < 3.75 {
        let y = (x / 3.75).powi(2);
        return y.mul_add(
            3.515_622_9
                + y * (3.089_942_4
                    + y * (1.206_749_2 + y * (0.265_973_2 + y * (0.036_076_8 + y * 0.004_581_3)))),
            1.0,
        );
    }

    // For large arguments, use asymptotic expansion
    let y = 3.75 / ax;
    let exp_term = (ax).exp() / (ax).sqrt();

    exp_term
        * y.mul_add(
            0.013_285_92
                + y * (0.002_253_19
                    + y * (-0.001_575_65
                        + y * (0.009_162_81
                            + y * (-0.020_577_06
                                + y * (0.026_355_37 + y * (-0.016_476_33 + y * 0.003_923_77)))))),
            0.398_942_28,
        )
}

/// Apply window function to a signal
///
/// # Arguments
///
/// * `x` - Input signal
/// * `window` - Window function to apply
///
/// # Returns
///
/// The windowed signal.
///
/// # Examples
///
/// ```
/// use scirs2_fft::window::{Window, apply_window};
/// use scirs2_core::ndarray::Array1;
///
/// let signal = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
/// let window = Window::Hann;
/// let windowed = apply_window(&signal, window).unwrap();
/// ```
/// # Errors
///
/// Returns an error if the window calculation fails or if the value cannot be
/// converted to the target floating point type.
///
/// # Panics
///
/// Panics if the conversion from f64 to type F fails.
#[allow(dead_code)]
pub fn apply_window<F, S>(x: &ArrayBase<S, Ix1>, window: Window) -> FFTResult<Array1<F>>
where
    S: Data<Elem = F>,
    F: Float + FromPrimitive + Debug,
{
    let n = x.len();
    let win = get_window(window, n, true)?;

    let mut result = Array1::zeros(n);
    for i in 0..n {
        result[i] = x[i] * F::from_f64(win[i]).unwrap();
    }

    Ok(result)
}

/// Compute the equivalent noise bandwidth of a window
///
/// # Arguments
///
/// * `window` - Window function
/// * `n` - Window length
///
/// # Returns
///
/// The equivalent noise bandwidth of the window.
///
/// # Examples
///
/// ```
/// use scirs2_fft::window::{Window, enbw};
/// use approx::assert_relative_eq;
///
/// let bandwidth = enbw(Window::Hann, 1024).unwrap();
/// assert_relative_eq!(bandwidth, 1.5, epsilon = 0.01);
/// ```
/// # Errors
///
/// Returns an error if the window calculation fails.
#[allow(dead_code)]
pub fn enbw(window: Window, n: usize) -> FFTResult<f64> {
    let w = get_window(window, n, true)?;

    let sum_squared = w.iter().map(|&x| x.powi(2)).sum::<f64>();
    let square_sum = w.iter().sum::<f64>().powi(2);

    // Safe conversion as n is unlikely to exceed f64 precision in practice
    let n_f64 = n as f64;
    Ok(n_f64 * sum_squared / square_sum)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_rectangular() {
        let win = rectangular(5).unwrap();
        let expected = [1.0, 1.0, 1.0, 1.0, 1.0];
        for (a, &b) in win.iter().zip(expected.iter()) {
            assert_relative_eq!(a, &b, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_hann() {
        let win = hann(5, true).unwrap();
        let expected = [0.0, 0.5, 1.0, 0.5, 0.0];
        for (a, &b) in win.iter().zip(expected.iter()) {
            assert_relative_eq!(a, &b, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_hamming() {
        let win = hamming(5, true).unwrap();
        let expected = [0.08, 0.54, 1.0, 0.54, 0.08];
        for (a, &b) in win.iter().zip(expected.iter()) {
            assert_relative_eq!(a, &b, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_blackman() {
        let win = blackman(5, true).unwrap();
        let expected = [0.0, 0.34, 1.0, 0.34, 0.0];
        for (a, &b) in win.iter().zip(expected.iter()) {
            assert_relative_eq!(a, &b, epsilon = 0.01);
        }
    }

    #[test]
    fn test_from_str() {
        assert_eq!(Window::from_str("hann").unwrap(), Window::Hann);
        assert_eq!(Window::from_str("hamming").unwrap(), Window::Hamming);
        assert_eq!(Window::from_str("blackman").unwrap(), Window::Blackman);
        assert_eq!(
            Window::from_str("rectangular").unwrap(),
            Window::Rectangular
        );
        assert!(Window::from_str("invalid").is_err());
    }

    #[test]
    fn test_get_window() {
        let win1 = get_window(Window::Hann, 5, true).unwrap();
        let win2 = get_window("hann", 5, true).unwrap();

        for (a, b) in win1.iter().zip(win2.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_apply_window() {
        let signal = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let win = apply_window(&signal.view(), Window::Hann).unwrap();

        let expected = Array1::from_vec(vec![0.0, 1.0, 3.0, 2.0, 0.0]);
        for (a, b) in win.iter().zip(expected.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_enbw() {
        // Known ENBW values for common windows
        let rect_enbw = enbw(Window::Rectangular, 1024).unwrap();
        assert_relative_eq!(rect_enbw, 1.0, epsilon = 1e-10);

        let hann_enbw = enbw(Window::Hann, 1024).unwrap();
        assert_relative_eq!(hann_enbw, 1.5, epsilon = 0.01);

        let hamming_enbw = enbw(Window::Hamming, 1024).unwrap();
        assert_relative_eq!(hamming_enbw, 1.36, epsilon = 0.01);
    }
}
