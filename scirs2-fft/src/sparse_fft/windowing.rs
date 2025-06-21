//! Window function implementations for Sparse FFT
//!
//! This module provides various window functions that can be applied to signals
//! before performing sparse FFT operations to reduce spectral leakage.

use crate::error::{FFTError, FFTResult};
use num_complex::Complex64;
use num_traits::NumCast;
use std::f64::consts::PI;
use std::fmt::Debug;

use super::config::WindowFunction;

/// Apply a window function to the signal
pub fn apply_window<T>(
    signal: &[T],
    window_function: WindowFunction,
    kaiser_beta: f64,
) -> FFTResult<Vec<Complex64>>
where
    T: NumCast + Copy + Debug + 'static,
{
    // Convert input to complex
    let signal_complex: Vec<Complex64> = signal
        .iter()
        .map(|&val| {
            let val_f64 = NumCast::from(val).ok_or_else(|| {
                FFTError::ValueError(format!("Could not convert {:?} to f64", val))
            })?;
            Ok(Complex64::new(val_f64, 0.0))
        })
        .collect::<FFTResult<Vec<_>>>()?;

    let n = signal_complex.len();

    // If no windowing is required, return the original signal
    if window_function == WindowFunction::None {
        return Ok(signal_complex);
    }

    // Apply the selected window function
    let windowed_signal = match window_function {
        WindowFunction::None => signal_complex, // Already handled above, but included for completeness

        WindowFunction::Hann => {
            let mut windowed = signal_complex;
            for (i, sample) in windowed.iter_mut().enumerate() {
                let window_val = 0.5 * (1.0 - (2.0 * PI * i as f64 / (n - 1) as f64).cos());
                *sample *= window_val;
            }
            windowed
        }

        WindowFunction::Hamming => {
            let mut windowed = signal_complex;
            for (i, sample) in windowed.iter_mut().enumerate() {
                let window_val = 0.54 - 0.46 * (2.0 * PI * i as f64 / (n - 1) as f64).cos();
                *sample *= window_val;
            }
            windowed
        }

        WindowFunction::Blackman => {
            let mut windowed = signal_complex;
            for (i, sample) in windowed.iter_mut().enumerate() {
                let angle = 2.0 * PI * i as f64 / (n - 1) as f64;
                let window_val = 0.42 - 0.5 * angle.cos() + 0.08 * (2.0 * angle).cos();
                *sample *= window_val;
            }
            windowed
        }

        WindowFunction::FlatTop => {
            let mut windowed = signal_complex;
            for (i, sample) in windowed.iter_mut().enumerate() {
                let angle = 2.0 * PI * i as f64 / (n - 1) as f64;
                let window_val = 0.21557895 - 0.41663158 * angle.cos()
                    + 0.277263158 * (2.0 * angle).cos()
                    - 0.083578947 * (3.0 * angle).cos()
                    + 0.006947368 * (4.0 * angle).cos();
                *sample *= window_val;
            }
            windowed
        }

        WindowFunction::Kaiser => {
            let mut windowed = signal_complex;
            let beta = kaiser_beta;
            let alpha = (n - 1) as f64 / 2.0;
            let i0_beta = modified_bessel_i0(beta);

            for (i, sample) in windowed.iter_mut().enumerate() {
                let x = beta * (1.0 - ((i as f64 - alpha) / alpha).powi(2)).sqrt();
                let window_val = modified_bessel_i0(x) / i0_beta;
                *sample *= window_val;
            }
            windowed
        }
    };

    Ok(windowed_signal)
}

/// Modified Bessel function of the first kind, order 0
/// Used for Kaiser window computation
fn modified_bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    let half_x = x / 2.0;

    for k in 1..=50 {
        term *= (half_x / k as f64).powi(2);
        sum += term;
        if term < 1e-12 * sum {
            break;
        }
    }

    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_apply_window_none() {
        let signal = vec![1.0, 2.0, 3.0, 4.0];
        let result = apply_window(&signal, WindowFunction::None, 14.0).unwrap();

        assert_eq!(result.len(), 4);
        assert_eq!(result[0], Complex64::new(1.0, 0.0));
        assert_eq!(result[1], Complex64::new(2.0, 0.0));
    }

    #[test]
    fn test_apply_window_hann() {
        let signal = vec![1.0; 4];
        let result = apply_window(&signal, WindowFunction::Hann, 14.0).unwrap();

        assert_eq!(result.len(), 4);
        // First and last samples should be zero for Hann window
        assert!((result[0].re).abs() < 1e-10);
        assert!((result[3].re).abs() < 1e-10);
    }

    #[test]
    fn test_apply_window_hamming() {
        let signal = vec![1.0; 4];
        let result = apply_window(&signal, WindowFunction::Hamming, 14.0).unwrap();

        assert_eq!(result.len(), 4);
        // Hamming window should not be zero at endpoints
        assert!(result[0].re > 0.0);
        assert!(result[3].re > 0.0);
    }

    #[test]
    fn test_modified_bessel_i0() {
        // Test known values
        let result = modified_bessel_i0(0.0);
        assert!((result - 1.0).abs() < 1e-10);

        let result = modified_bessel_i0(1.0);
        assert!((result - 1.2660658777520084).abs() < 1e-10);
    }
}
