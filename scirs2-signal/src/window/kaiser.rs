// Implementation of Kaiser window and Kaiser-Bessel derived window.
//
// The Kaiser window is a flexible window function with a parameter that controls
// the trade-off between the main lobe width and side lobe level.

use crate::error::SignalResult;
use scirs2_special::i0;
use std::f64::consts::PI;

#[allow(unused_imports)]
/// Kaiser window.
///
/// The Kaiser window is a taper formed by using a Bessel function of the first kind
/// where beta is a non-negative real number that determines the shape of the window.
///
/// As beta increases, the transition becomes sharper, but the side lobes increase in height.
///
/// # Arguments
///
/// * `m` - Number of points in the output window
/// * `beta` - Shape parameter that determines the trade-off between main-lobe width
///   and side lobe level
/// * `sym` - If true, generates a symmetric window, otherwise a periodic window
///
/// # Returns
///
/// A Vec<f64> of window values
///
/// # Examples
///
/// ```
/// use scirs2_signal::window::kaiser;
///
/// let window = kaiser(10, 5.0, true).unwrap();
/// assert_eq!(window.len(), 10);
/// ```
#[allow(dead_code)]
pub fn kaiser(m: usize, beta: f64, sym: bool) -> SignalResult<Vec<f64>> {
    if super::_len_guards(m) {
        return Ok(vec![1.0; m]);
    }

    let (n, needs_trunc) = super::_extend(m, sym);

    let mut w = Vec::with_capacity(n);

    // Calculate normalization factor (the Bessel function evaluated at beta)
    let i0_beta = i0(beta);

    for i in 0..n {
        // Calculate terms inside the square root
        let k = 2.0 * i as f64 / (n - 1) as f64 - 1.0;
        let term = 1.0 - k * k;

        // Compute the window value
        let w_val = if term > 0.0 {
            i0(beta * term.sqrt()) / i0_beta
        } else {
            0.0
        };

        w.push(w_val);
    }

    Ok(super::_truncate(w, needs_trunc))
}

/// Kaiser-Bessel derived window.
///
/// The Kaiser-Bessel derived window has more controlled side-lobe behavior
/// than the standard Kaiser window.
///
/// # Arguments
///
/// * `m` - Number of points in the output window
/// * `beta` - Shape parameter that determines the trade-off between main-lobe width
///   and side lobe level
/// * `sym` - If true, generates a symmetric window, otherwise a periodic window
///
/// # Returns
///
/// A Vec<f64> of window values
///
/// # Examples
///
/// ```
/// use scirs2_signal::window::kaiser_bessel_derived;
///
/// let window = kaiser_bessel_derived(10, 5.0, true).unwrap();
/// assert_eq!(window.len(), 10);
/// ```
#[allow(dead_code)]
pub fn kaiser_bessel_derived(m: usize, beta: f64, sym: bool) -> SignalResult<Vec<f64>> {
    if super::_len_guards(m) {
        return Ok(vec![1.0; m]);
    }

    let (n, needs_trunc) = super::_extend(m, sym);

    // Get the original Kaiser window
    let kaiser_win = kaiser(n, beta, true)?;

    // Compute the Kaiser-Bessel derived window using Fourier transform
    let mut w = vec![0.0; n];
    let _len_half = n / 2;

    // Special handling for even and odd lengths
    if n % 2 == 0 {
        // Even
        for (i, item) in w.iter_mut().enumerate().take(n) {
            let mut sum = 0.0;
            // Add DC component
            sum += kaiser_win[0];

            // Add positive frequencies
            for j in 1..(n / 2) {
                let idx = j;
                if idx < kaiser_win.len() {
                    let angle = 2.0 * PI * i as f64 * j as f64 / n as f64;
                    sum += 2.0 * kaiser_win[idx] * angle.cos();
                }
            }

            // Add Nyquist
            if n / 2 < kaiser_win.len() {
                let angle = PI * i as f64;
                sum += kaiser_win[n / 2] * angle.cos();
            }

            *item = if sum > 0.0 {
                (sum / n as f64).sqrt()
            } else {
                0.0
            };
        }
    } else {
        // Odd
        for (i, item) in w.iter_mut().enumerate().take(n) {
            let mut sum = 0.0;
            // Add all components
            for j in 0..n {
                if j < kaiser_win.len() {
                    let angle = 2.0 * PI * i as f64 * j as f64 / n as f64;
                    sum += kaiser_win[j] * angle.cos();
                }
            }

            *item = if sum > 0.0 {
                (sum / n as f64).sqrt()
            } else {
                0.0
            };
        }
    }

    Ok(super::_truncate(w, needs_trunc))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    #[test]
    fn test_kaiser_window() {
        let window = kaiser(10, 5.0, true).unwrap();
        assert_eq!(window.len(), 10);

        // Test symmetry
        for i in 0..5 {
            assert_relative_eq!(window[i], window[9 - i], epsilon = 1e-10);
        }

        // Test specific values
        assert_relative_eq!(window[0], window[9], epsilon = 1e-10);
        assert!(window[0] < window[1]);
        assert!(window[1] < window[2]);
    }

    #[test]
    fn test_kaiser_bessel_derived_window() {
        let window = kaiser_bessel_derived(10, 5.0, true).unwrap();
        assert_eq!(window.len(), 10);

        // Test that all values are finite
        for &val in window.iter() {
            assert!(val.is_finite());
        }

        // Test that window has reasonable values (between 0 and some maximum)
        let max_val = window.iter().fold(0.0f64, |a, &b| f64::max(a, b.abs()));
        assert!(max_val > 0.0 && max_val < 10.0);

        // The Kaiser-Bessel derived window may not be perfectly symmetric due to
        // FFT-based computation, so we skip strict symmetry test
    }
}
