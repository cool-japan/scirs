//! Window functions for FFT operations
//!
//! This module provides windowing functions that can be applied to signals before
//! applying FFT to reduce spectral leakage.
//!
//! Common window functions include:
//! - Hann window
//! - Hamming window
//! - Blackman window
//! - Kaiser window
//! - Tukey window

use crate::error::{FFTError, FFTResult};
use std::f64::consts::PI;

/// Different types of window functions
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WindowType {
    /// Rectangular window (no windowing)
    Rectangular,
    /// Hann window (raised cosine)
    Hann,
    /// Hamming window (raised cosine with non-zero endpoints)
    Hamming,
    /// Blackman window (three-term cosine)
    Blackman,
    /// Blackman-Harris window (four-term cosine)
    BlackmanHarris,
    /// Flat-top window (optimized for amplitude accuracy)
    FlatTop,
    /// Bartlett window (triangular)
    Bartlett,
    /// Bartlett-Hann window (combination of Bartlett and Hann)
    BartlettHann,
    /// Tukey window (tapered cosine)
    Tukey(f64), // parameter is alpha
    /// Kaiser window (based on Bessel function)
    Kaiser(f64), // parameter is beta
    /// Gaussian window
    Gaussian(f64), // parameter is sigma
}

/// Creates a window function of the specified type and length
///
/// # Arguments
///
/// * `window_type` - The type of window to create
/// * `length` - The length of the window
///
/// # Returns
///
/// A vector containing the window values
///
/// # Examples
///
/// ```
/// use scirs2_fft::fft::windowing::{create_window, WindowType};
///
/// // Create a Hann window of length 10
/// let window = create_window(WindowType::Hann, 10).unwrap();
/// assert_eq!(window.len(), 10);
/// assert!(window[0] < 0.01); // Near zero at the edges
/// assert!(window[5] > 0.9); // Near one in the middle
/// ```
pub fn create_window(window_type: WindowType, length: usize) -> FFTResult<Vec<f64>> {
    if length == 0 {
        return Err(FFTError::ValueError("Window length cannot be zero".into()));
    }

    let mut window = vec![0.0; length];
    let n = length as f64;

    match window_type {
        WindowType::Rectangular => {
            // Rectangular window (all ones)
            window.iter_mut().for_each(|w| *w = 1.0);
        }
        WindowType::Hann => {
            // Hann window (raised cosine)
            for (i, w) in window.iter_mut().enumerate() {
                let x = i as f64 / (n - 1.0);
                *w = 0.5 * (1.0 - (2.0 * PI * x).cos());
            }
        }
        WindowType::Hamming => {
            // Hamming window (raised cosine with non-zero endpoints)
            for (i, w) in window.iter_mut().enumerate() {
                let x = i as f64 / (n - 1.0);
                *w = 0.54 - 0.46 * (2.0 * PI * x).cos();
            }
        }
        WindowType::Blackman => {
            // Blackman window (three-term cosine)
            for (i, w) in window.iter_mut().enumerate() {
                let x = i as f64 / (n - 1.0);
                *w = 0.42 - 0.5 * (2.0 * PI * x).cos() + 0.08 * (4.0 * PI * x).cos();
            }
        }
        WindowType::BlackmanHarris => {
            // Blackman-Harris window (four-term cosine)
            for (i, w) in window.iter_mut().enumerate() {
                let x = i as f64 / (n - 1.0);
                *w = 0.35875 - 0.48829 * (2.0 * PI * x).cos() + 0.14128 * (4.0 * PI * x).cos()
                    - 0.01168 * (6.0 * PI * x).cos();
            }
        }
        WindowType::FlatTop => {
            // Flat-top window (good amplitude accuracy)
            for (i, w) in window.iter_mut().enumerate() {
                let x = i as f64 / (n - 1.0);
                *w = 0.21557895 - 0.41663158 * (2.0 * PI * x).cos()
                    + 0.277263158 * (4.0 * PI * x).cos()
                    - 0.083578947 * (6.0 * PI * x).cos()
                    + 0.006947368 * (8.0 * PI * x).cos();
            }
        }
        WindowType::Bartlett => {
            // Bartlett window (triangular)
            for (i, w) in window.iter_mut().enumerate() {
                let x = i as f64 / (n - 1.0);
                *w = 1.0 - (2.0 * x - 1.0).abs();
            }
        }
        WindowType::BartlettHann => {
            // Bartlett-Hann window
            for (i, w) in window.iter_mut().enumerate() {
                let x = i as f64 / (n - 1.0);
                *w = 0.62 - 0.48 * (x - 0.5).abs() - 0.38 * (2.0 * PI * x).cos();
            }
        }
        WindowType::Tukey(alpha) => {
            // Tukey window (tapered cosine)
            if alpha <= 0.0 {
                // If alpha <= 0, it's a rectangular window
                window.iter_mut().for_each(|w| *w = 1.0);
            } else if alpha >= 1.0 {
                // If alpha >= 1, it's a Hann window
                for (i, w) in window.iter_mut().enumerate() {
                    let x = i as f64 / (n - 1.0);
                    *w = 0.5 * (1.0 - (2.0 * PI * x).cos());
                }
            } else {
                // Otherwise, it's a true Tukey window
                let alpha_n = alpha * (n - 1.0) / 2.0;
                for (i, w) in window.iter_mut().enumerate() {
                    let x = i as f64;
                    if x < alpha_n {
                        *w = 0.5 * (1.0 - (PI * x / alpha_n).cos());
                    } else if x <= (n - 1.0) - alpha_n {
                        *w = 1.0;
                    } else {
                        *w = 0.5 * (1.0 - (PI * (x - (n - 1.0) + alpha_n) / alpha_n).cos());
                    }
                }
            }
        }
        WindowType::Kaiser(beta) => {
            // Kaiser window
            let bessel_i0 = |x: f64| -> f64 {
                // Approximate the modified Bessel function of order 0
                let mut sum = 1.0;
                let mut term = 1.0;
                for k in 1..20 {
                    let k_f = k as f64;
                    term *= (x / 2.0).powi(2) / (k_f * k_f);
                    sum += term;
                    if term < 1e-12 * sum {
                        break;
                    }
                }
                sum
            };

            let beta_i0 = bessel_i0(beta);
            for (i, w) in window.iter_mut().enumerate() {
                let x = 2.0 * i as f64 / (n - 1.0) - 1.0;
                let arg = beta * (1.0 - x * x).sqrt();
                *w = bessel_i0(arg) / beta_i0;
            }
        }
        WindowType::Gaussian(sigma) => {
            // Gaussian window
            let center = (n - 1.0) / 2.0;
            for (i, w) in window.iter_mut().enumerate() {
                let x = i as f64 - center;
                *w = (-0.5 * (x / (sigma * center)).powi(2)).exp();
            }
        }
    }

    Ok(window)
}

/// Apply a window function to a signal
///
/// # Arguments
///
/// * `signal` - The signal to apply the window to
/// * `window` - The window function to apply
///
/// # Returns
///
/// A vector containing the windowed signal
///
/// # Examples
///
/// ```
/// use scirs2_fft::fft::windowing::{apply_window, create_window, WindowType};
///
/// // Create a simple signal
/// let signal = vec![1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0];
///
/// // Create a Hann window
/// let window = create_window(WindowType::Hann, signal.len()).unwrap();
///
/// // Apply the window to the signal
/// let windowed_signal = apply_window(&signal, &window).unwrap();
///
/// // Check that the window was applied correctly
/// assert_eq!(windowed_signal.len(), signal.len());
/// assert!(windowed_signal[0] < signal[0]); // Edges are attenuated
/// assert!(windowed_signal[3] <= signal[3]); // Middle is preserved or slightly attenuated
/// ```
pub fn apply_window(signal: &[f64], window: &[f64]) -> FFTResult<Vec<f64>> {
    if signal.len() != window.len() {
        return Err(FFTError::ValueError(
            "Signal and window lengths must match".into(),
        ));
    }

    let mut windowed_signal = vec![0.0; signal.len()];
    for (i, w) in windowed_signal.iter_mut().enumerate() {
        *w = signal[i] * window[i];
    }

    Ok(windowed_signal)
}

/// Calculate window properties like the equivalent noise bandwidth
///
/// # Arguments
///
/// * `window` - The window function
///
/// # Returns
///
/// A struct containing window properties
pub fn window_properties(window: &[f64]) -> WindowProperties {
    let n = window.len();
    let mut sum = 0.0;
    let mut sum_squared = 0.0;
    let mut coherent_gain = 0.0;

    for &w in window {
        sum += w;
        sum_squared += w * w;
        coherent_gain += w;
    }

    coherent_gain /= n as f64;
    let processing_gain = coherent_gain.powi(2) / (sum_squared / n as f64);
    let equivalent_noise_bandwidth = n as f64 * sum_squared / (sum * sum);

    WindowProperties {
        coherent_gain,
        processing_gain,
        equivalent_noise_bandwidth,
    }
}

/// Properties of a window function
#[derive(Debug, Clone, Copy)]
pub struct WindowProperties {
    /// Coherent gain (mean value of the window)
    pub coherent_gain: f64,
    /// Processing gain (improvement in SNR)
    pub processing_gain: f64,
    /// Equivalent noise bandwidth
    pub equivalent_noise_bandwidth: f64,
}
