// Filter application and signal processing functions
//
// This module provides functions for applying filters to signals including
// forward-backward filtering (filtfilt), direct filtering (lfilter), minimum
// phase conversion, and matched filtering for signal detection.

use crate::error::{SignalError, SignalResult};
use scirs2_core::numeric::Complex64;
use scirs2_core::numeric::{Float, NumCast, Zero};
use std::fmt::Debug;

#[allow(unused_imports)]
/// Apply a digital filter forward and backward to a signal (zero-phase filtering)
///
/// This function applies the filter forwards, then backwards to achieve zero-phase
/// distortion. The result has zero phase delay but twice the filter order.
/// This is equivalent to MATLAB's filtfilt function.
///
/// # Arguments
///
/// * `b` - Numerator coefficients
/// * `a` - Denominator coefficients  
/// * `x` - Input signal
///
/// # Returns
///
/// * Filtered signal with zero phase delay
///
/// # Examples
///
/// ```
/// use scirs2_signal::filter::application::filtfilt;
/// use scirs2_signal::filter::iir::butter;
///
/// // Design a filter and apply it with zero phase delay
/// let (b, a) = butter(4, 0.2, "lowpass").unwrap();
/// let signal = vec![1.0, 2.0, 3.0, 2.0, 1.0];
/// let filtered = filtfilt(&b, &a, &signal).unwrap();
/// ```
#[allow(dead_code)]
pub fn filtfilt<T>(b: &[f64], a: &[f64], x: &[T]) -> SignalResult<Vec<f64>>
where
    T: Float + NumCast + Debug,
{
    if a.is_empty() || a[0] == 0.0 {
        return Err(SignalError::ValueError(
            "First denominator coefficient cannot be zero".to_string(),
        ));
    }

    // Convert input to f64
    let x_f64: Vec<f64> = x
        .iter()
        .map(|&val| {
            NumCast::from(val).ok_or_else(|| {
                SignalError::ValueError(format!("Could not convert {:?} to f64", val))
            })
        })
        .collect::<SignalResult<Vec<_>>>()?;

    // 1. Apply filter forward
    let y1 = lfilter(b, a, &x_f64)?;

    // 2. Reverse the result
    let mut y1_rev = y1.clone();
    y1_rev.reverse();

    // 3. Apply filter backward
    let y2 = lfilter(b, a, &y1_rev)?;

    // 4. Reverse again to get the final result
    let mut result = y2;
    result.reverse();

    Ok(result)
}

/// Apply a digital filter to a signal (direct form II transposed)
///
/// This function implements the standard direct form II transposed structure
/// for applying IIR and FIR filters. It performs causal filtering with the
/// inherent group delay of the filter.
///
/// # Arguments
///
/// * `b` - Numerator coefficients
/// * `a` - Denominator coefficients
/// * `x` - Input signal
///
/// # Returns
///
/// * Filtered signal
///
/// # Examples
///
/// ```
/// use scirs2_signal::filter::application::lfilter;
/// use scirs2_signal::filter::iir::butter;
///
/// // Design a filter and apply it to a signal
/// let (b, a) = butter(4, 0.2, "lowpass").unwrap();
/// let signal = vec![1.0, 2.0, 3.0, 2.0, 1.0];
/// let filtered = lfilter(&b, &a, &signal).unwrap();
/// ```
#[allow(dead_code)]
pub fn lfilter<T>(b: &[f64], a: &[f64], x: &[T]) -> SignalResult<Vec<f64>>
where
    T: Float + NumCast + Debug,
{
    if a.is_empty() || a[0] == 0.0 {
        return Err(SignalError::ValueError(
            "First denominator coefficient cannot be zero".to_string(),
        ));
    }

    // Normalize coefficients by a[0]
    let a0 = a[0];
    let b_norm: Vec<f64> = b.iter().map(|&val| val / a0).collect();
    let a_norm: Vec<f64> = a.iter().map(|&val| val / a0).collect();

    // Convert input to f64
    let x_f64: Vec<f64> = x
        .iter()
        .map(|&val| {
            NumCast::from(val).ok_or_else(|| {
                SignalError::ValueError(format!("Could not convert {:?} to f64", val))
            })
        })
        .collect::<SignalResult<Vec<_>>>()?;

    // Apply filter using direct form II transposed
    let n = x_f64.len();
    let mut y = vec![0.0; n];
    let mut z = vec![0.0; a_norm.len().max(b_norm.len()) - 1]; // State variables

    for i in 0..n {
        // Compute output
        y[i] = if !b_norm.is_empty() {
            b_norm[0] * x_f64[i]
        } else {
            0.0
        } + if !z.is_empty() { z[0] } else { 0.0 };

        // Update state variables
        for j in 1..z.len() {
            let b_term = if j < b_norm.len() {
                b_norm[j] * x_f64[i]
            } else {
                0.0
            };
            let a_term = if j < a_norm.len() {
                a_norm[j] * y[i]
            } else {
                0.0
            };
            let next_z = if j + 1 < z.len() { z[j] } else { 0.0 };

            z[j - 1] = b_term + next_z - a_term;
        }

        // Update last state variable if it exists
        if !z.is_empty() {
            let last = z.len() - 1;
            let b_term = if last + 1 < b_norm.len() {
                b_norm[last + 1] * x_f64[i]
            } else {
                0.0
            };
            let a_term = if last + 1 < a_norm.len() {
                a_norm[last + 1] * y[i]
            } else {
                0.0
            };
            z[last] = b_term - a_term;
        }
    }

    Ok(y)
}

/// Convert a filter to minimum phase
///
/// A minimum phase filter has all its zeros inside the unit circle (discrete-time)
/// or with negative real parts (continuous-time). This function converts any filter
/// to its minimum phase equivalent while preserving the magnitude response.
///
/// # Arguments
///
/// * `b` - Numerator coefficients of the filter
/// * `discrete_time` - True for discrete-time systems, false for continuous-time
///
/// # Returns
///
/// * Minimum phase filter coefficients
///
/// # Examples
///
/// ```
/// use scirs2_signal::filter::application::minimum_phase;
///
/// // Convert a filter to minimum phase
/// let b = vec![1.0, -2.0, 1.0]; // (z-1)^2, has zeros at z=1 (outside unit circle)
/// let b_min = minimum_phase(&b, true).unwrap();
/// ```
#[allow(dead_code)]
pub fn minimum_phase(b: &[f64], discretetime: bool) -> SignalResult<Vec<f64>> {
    if b.is_empty() {
        return Err(SignalError::ValueError(
            "Filter coefficients cannot be empty".to_string(),
        ));
    }

    // For constant filters, return as-is
    if b.len() == 1 {
        return Ok(b.to_vec());
    }

    // Find the roots (zeros) of the polynomial
    let zeros = find_polynomial_roots(b)?;

    // Convert non-minimum phase zeros to minimum phase
    let mut min_phase_zeros = Vec::new();
    let mut gain_adjustment = 1.0;

    for zero in zeros {
        if discretetime {
            // For discrete-_time: zeros inside unit circle are minimum phase
            if zero.norm() > 1.0 {
                // Reflect zero to its conjugate reciprocal: 1/conj(zero)
                let min_zero = 1.0 / zero.conj();
                min_phase_zeros.push(min_zero);
                // Adjust gain to preserve magnitude response
                gain_adjustment *= zero.norm();
            } else {
                min_phase_zeros.push(zero);
            }
        } else {
            // For continuous-_time: zeros with negative real parts are minimum phase
            if zero.re > 0.0 {
                // Reflect zero to negative real part: -conj(zero)
                let min_zero = -zero.conj();
                min_phase_zeros.push(min_zero);
                // Adjust gain to preserve magnitude response at s=0
                gain_adjustment *= -zero.re / min_zero.re;
            } else {
                min_phase_zeros.push(zero);
            }
        }
    }

    // Reconstruct polynomial from minimum phase zeros
    let mut min_phase_b = polynomial_from_roots(&min_phase_zeros);

    // Apply gain adjustment
    for coeff in &mut min_phase_b {
        *coeff *= gain_adjustment;
    }

    // Normalize to match original leading coefficient if needed
    if !min_phase_b.is_empty() && min_phase_b[0].abs() > 1e-10 {
        let scale = b[0] / min_phase_b[0];
        for coeff in &mut min_phase_b {
            *coeff *= scale;
        }
    }

    Ok(min_phase_b)
}

/// Compute group delay of a digital filter
///
/// Group delay is the negative derivative of the phase response with respect to frequency.
/// It represents the time delay experienced by different frequency components.
///
/// # Arguments
///
/// * `b` - Numerator coefficients
/// * `a` - Denominator coefficients
/// * `w` - Frequency points (normalized from 0 to π)
///
/// # Returns
///
/// * Group delay values at the specified frequencies
///
/// # Examples
///
/// ```
/// use scirs2_signal::filter::application::group_delay;
/// use scirs2_signal::filter::iir::butter;
///
/// // Compute group delay of a Butterworth filter
/// let (b, a) = butter(4, 0.2, "lowpass").unwrap();
/// let frequencies = (0..128).map(|i| std::f64::consts::PI * i as f64 / 127.0).collect::<Vec<_>>();
/// let gd = group_delay(&b, &a, &frequencies).unwrap();
/// ```
#[allow(dead_code)]
pub fn group_delay(b: &[f64], a: &[f64], w: &[f64]) -> SignalResult<Vec<f64>> {
    if a.is_empty() || a[0].abs() < 1e-10 {
        return Err(SignalError::ValueError(
            "Invalid denominator coefficients".to_string(),
        ));
    }

    let mut gd = Vec::with_capacity(w.len());

    for &freq in w {
        // Compute the group delay using the derivative method
        // gd = -d(phase)/dw = -d(arg(H(e^jw)))/dw

        // For numerical computation, use a small frequency step
        let eps = 1e-6;
        let freq_minus = (freq - eps).max(0.0);
        let freq_plus = (freq + eps).min(std::f64::consts::PI);

        // Evaluate transfer function at freq-eps and freq+eps
        let h_minus = evaluate_transfer_function(b, a, freq_minus);
        let h_plus = evaluate_transfer_function(b, a, freq_plus);

        // Compute phase difference and normalize by frequency difference
        let phase_diff = h_plus.arg() - h_minus.arg();
        let freq_diff = freq_plus - freq_minus;

        if freq_diff > 0.0 {
            gd.push(-phase_diff / freq_diff);
        } else {
            gd.push(0.0);
        }
    }

    Ok(gd)
}

/// Design a matched filter for detecting a known signal in noise
///
/// A matched filter is optimal for detecting a known signal in the presence of
/// additive white Gaussian noise. It maximizes the signal-to-noise ratio at the
/// output and is widely used in radar, communications, and correlation applications.
///
/// # Arguments
///
/// * `template` - The known signal template to match against
/// * `normalize` - If true, normalize the filter to unit energy
///
/// # Returns
///
/// * Matched filter coefficients (time-reversed and conjugated template)
///
/// # Examples
///
/// ```
/// use scirs2_signal::filter::application::matched_filter;
///
/// // Design a matched filter for a simple pulse
/// let template = vec![1.0, 1.0, 1.0, 0.0, 0.0];
/// let mf = matched_filter(&template, true).unwrap();
/// ```
#[allow(dead_code)]
pub fn matched_filter(template: &[f64], normalize: bool) -> SignalResult<Vec<f64>> {
    if template.is_empty() {
        return Err(SignalError::ValueError(
            "Template cannot be empty".to_string(),
        ));
    }

    // Matched filter is the time-reversed (and conjugated for complex signals) _template
    let mut mf: Vec<f64> = template.iter().rev().copied().collect();

    if normalize {
        // Normalize to unit energy
        let energy: f64 = mf.iter().map(|&x| x * x).sum();
        if energy > 1e-10 {
            let norm_factor = 1.0 / energy.sqrt();
            for coeff in &mut mf {
                *coeff *= norm_factor;
            }
        }
    }

    Ok(mf)
}

/// Apply matched filter to detect template in signal
///
/// Applies the matched filter to a signal and returns the correlation output.
/// Peak values in the output indicate potential locations of the template.
///
/// # Arguments
///
/// * `signal` - Input signal to search
/// * `template` - Template to detect
/// * `normalize` - If true, normalize the matched filter
///
/// # Returns
///
/// * Correlation output (same length as input signal)
///
/// # Examples
///
/// ```
/// use scirs2_signal::filter::application::matched_filter_detect;
///
/// let signal = vec![0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0];
/// let template = vec![1.0, 1.0, 1.0];
/// let output = matched_filter_detect(&signal, &template, true).unwrap();
/// ```
#[allow(dead_code)]
pub fn matched_filter_detect(
    signal: &[f64],
    template: &[f64],
    normalize: bool,
) -> SignalResult<Vec<f64>> {
    let mf = matched_filter(template, normalize)?;

    // Apply the matched filter using convolution
    let mut output = vec![0.0; signal.len()];

    for i in 0..signal.len() {
        for (j, &coeff) in mf.iter().enumerate() {
            if i >= j {
                output[i] += signal[i - j] * coeff;
            }
        }
    }

    Ok(output)
}

// Helper functions for internal use

/// Evaluate transfer function H(z) = B(z)/A(z) at a frequency
#[allow(dead_code)]
pub fn evaluate_transfer_function(b: &[f64], a: &[f64], w: f64) -> Complex64 {
    let z = Complex64::new(w.cos(), w.sin());

    // Evaluate numerator
    let mut num_val = Complex64::zero();
    for (i, &coeff) in b.iter().enumerate() {
        let power = b.len() - 1 - i;
        num_val += Complex64::new(coeff, 0.0) * z.powi(power as i32);
    }

    // Evaluate denominator
    let mut den_val = Complex64::zero();
    for (i, &coeff) in a.iter().enumerate() {
        let power = a.len() - 1 - i;
        den_val += Complex64::new(coeff, 0.0) * z.powi(power as i32);
    }

    if den_val.norm() < 1e-10 {
        Complex64::new(f64::INFINITY, 0.0)
    } else {
        num_val / den_val
    }
}

/// Find polynomial roots using a simplified iterative method
///
/// This is a basic implementation for demonstration purposes.
/// Production code would use more robust algorithms like Jenkins-Traub or eigenvalue methods.
#[allow(dead_code)]
pub fn find_polynomial_roots(coeffs: &[f64]) -> SignalResult<Vec<Complex64>> {
    if coeffs.is_empty() {
        return Ok(Vec::new());
    }

    // Remove leading zeros
    let mut trimmed_coeffs = coeffs.to_vec();
    while trimmed_coeffs.len() > 1 && trimmed_coeffs[0].abs() < 1e-10 {
        trimmed_coeffs.remove(0);
    }

    let n = trimmed_coeffs.len() - 1;
    if n == 0 {
        return Ok(Vec::new());
    }

    let mut roots = Vec::new();

    // Handle linear case
    if n == 1 {
        if trimmed_coeffs[0].abs() > 1e-10 {
            roots.push(Complex64::new(-trimmed_coeffs[1] / trimmed_coeffs[0], 0.0));
        }
        return Ok(roots);
    }

    // Handle quadratic case
    if n == 2 {
        let a = trimmed_coeffs[0];
        let b = trimmed_coeffs[1];
        let c = trimmed_coeffs[2];

        if a.abs() > 1e-10 {
            let discriminant = b * b - 4.0 * a * c;
            if discriminant >= 0.0 {
                let sqrt_disc = discriminant.sqrt();
                roots.push(Complex64::new((-b + sqrt_disc) / (2.0 * a), 0.0));
                roots.push(Complex64::new((-b - sqrt_disc) / (2.0 * a), 0.0));
            } else {
                let sqrt_disc = (-discriminant).sqrt();
                roots.push(Complex64::new(-b / (2.0 * a), sqrt_disc / (2.0 * a)));
                roots.push(Complex64::new(-b / (2.0 * a), -sqrt_disc / (2.0 * a)));
            }
        }
        return Ok(roots);
    }

    // For higher-order polynomials, use a simplified iterative method
    let max_iterations = 100;
    let tolerance = 1e-10;

    // Use initial guesses on a circle
    let mut estimates = Vec::with_capacity(n);
    for k in 0..n {
        let angle = 2.0 * std::f64::consts::PI * k as f64 / n as f64;
        estimates.push(Complex64::new(angle.cos(), angle.sin()));
    }

    for _iter in 0..max_iterations {
        let mut converged = true;

        for estimate in estimates.iter_mut().take(n) {
            // Evaluate polynomial at current estimate
            let z = *estimate;
            let (p_val, p_prime) = evaluate_polynomial_and_derivative(&trimmed_coeffs, z);

            // Simple Newton's method step
            if p_prime.norm() > tolerance {
                let correction = p_val / p_prime;
                *estimate = z - correction;

                if correction.norm() > tolerance {
                    converged = false;
                }
            }
        }

        if converged {
            break;
        }
    }

    // Filter out potential spurious roots
    for estimate in estimates {
        let (p_val, _) = evaluate_polynomial_and_derivative(&trimmed_coeffs, estimate);
        if p_val.norm() < 1e-6 {
            roots.push(estimate);
        }
    }

    Ok(roots)
}

/// Evaluate polynomial and its derivative at a complex point
#[allow(dead_code)]
fn evaluate_polynomial_and_derivative(coeffs: &[f64], z: Complex64) -> (Complex64, Complex64) {
    if coeffs.is_empty() {
        return (Complex64::zero(), Complex64::zero());
    }

    let n = coeffs.len() - 1;
    let mut p_val = Complex64::new(coeffs[0], 0.0);
    let mut p_prime = Complex64::zero();

    for (i, &coeff) in coeffs.iter().enumerate().skip(1) {
        let power = (n - i) as i32;
        p_prime = p_prime * z + p_val * Complex64::new(power as f64, 0.0);
        p_val = p_val * z + Complex64::new(coeff, 0.0);
    }

    (p_val, p_prime)
}

/// Reconstruct polynomial coefficients from roots
#[allow(dead_code)]
fn polynomial_from_roots(roots: &[Complex64]) -> Vec<f64> {
    if roots.is_empty() {
        return vec![1.0];
    }

    // Start with polynomial: 1
    let mut poly = vec![Complex64::new(1.0, 0.0)];

    // Multiply by (z - root) for each root
    for &root in roots {
        let mut new_poly = vec![Complex64::zero(); poly.len() + 1];

        // Multiply existing polynomial by z
        for (i, &coeff) in poly.iter().enumerate() {
            new_poly[i] += coeff;
        }

        // Subtract root times existing polynomial
        for (i, &coeff) in poly.iter().enumerate() {
            new_poly[i + 1] -= coeff * root;
        }

        poly = new_poly;
    }

    // Convert to real coefficients (imaginary parts should be small for conjugate pairs)
    poly.iter().map(|c| c.re).collect()
}
