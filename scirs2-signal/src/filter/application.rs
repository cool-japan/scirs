// Filter application and signal processing functions
//
// This module provides functions for applying filters to signals including
// forward-backward filtering (filtfilt), direct filtering (lfilter), minimum
// phase conversion, matched filtering for signal detection, and steady-state
// initial-condition computation for second-order-section (SOS) filters
// (sosfilt_zi).

use crate::error::{SignalError, SignalResult};
use scirs2_core::ndarray::Array2;
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
/// let (b, a) = butter(4, 0.2, "lowpass").expect("operation should succeed");
/// let signal = vec![1.0, 2.0, 3.0, 2.0, 1.0];
/// let filtered = filtfilt(&b, &a, &signal).expect("operation should succeed");
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
/// let (b, a) = butter(4, 0.2, "lowpass").expect("operation should succeed");
/// let signal = vec![1.0, 2.0, 3.0, 2.0, 1.0];
/// let filtered = lfilter(&b, &a, &signal).expect("operation should succeed");
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

        // Update state variables (Direct Form II Transposed)
        //
        //   z[j-1] = b[j]*x[i] - a[j]*y[i] + z[j]    for j = 1 .. N-2
        //   z[N-1] = b[N]*x[i] - a[N]*y[i]             (last state, no next_z)
        //
        // where N = z.len() and indices j run over the delay line.  Because j
        // increases, z[j] is always the *old* value when we read it.
        let n_state = z.len();
        for j in 0..n_state {
            let b_idx = j + 1;
            let a_idx = j + 1;
            let b_term = if b_idx < b_norm.len() {
                b_norm[b_idx] * x_f64[i]
            } else {
                0.0
            };
            let a_term = if a_idx < a_norm.len() {
                a_norm[a_idx] * y[i]
            } else {
                0.0
            };
            let next_z = if j + 1 < n_state { z[j + 1] } else { 0.0 };
            z[j] = b_term + next_z - a_term;
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
/// let b_min = minimum_phase(&b, true).expect("operation should succeed");
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
/// let (b, a) = butter(4, 0.2, "lowpass").expect("operation should succeed");
/// let frequencies = (0..128).map(|i| std::f64::consts::PI * i as f64 / 127.0).collect::<Vec<_>>();
/// let gd = group_delay(&b, &a, &frequencies).expect("operation should succeed");
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
/// let mf = matched_filter(&template, true).expect("operation should succeed");
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
/// let output = matched_filter_detect(&signal, &template, true).expect("operation should succeed");
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

/// Compute steady-state initial conditions for an SOS (cascaded biquad) IIR filter.
///
/// Given a Second-Order-Section representation of an IIR filter, returns the
/// internal Direct-Form-II-Transposed state values that, applied as initial
/// conditions, make the filter behave as if a constant input has been applied
/// for an infinite time prior to the first sample.  This is exactly the SciPy
/// `scipy.signal.sosfilt_zi` semantics.
///
/// # Algorithm
///
/// Each biquad section is normalised so that `a0 = 1`.  For section `i` with
/// coefficients `[b0, b1, b2, 1, a1, a2]` the Direct-Form-II-Transposed
/// state-space matrices are
///
/// ```text
/// A_sys = [[ -a1, 1 ],     B_sys = [ b1 - a1*b0,
///          [ -a2, 0 ]]              b2 - a2*b0 ]ᵀ
/// ```
///
/// Under a unit step the steady-state state vector is
///
/// ```text
/// zi_section = (I - A_sys)⁻¹ · B_sys
/// ```
///
/// `I - A_sys = [[1+a1, -1], [a2, 1]]` has determinant `1 + a1 + a2`, which is
/// also the DC denominator of the section.  The 2×2 system is solved
/// analytically (Cramer's rule) — no allocation of a linear-algebra solver is
/// needed.
///
/// For an SOS *cascade* the input to section `i+1` is the steady-state output
/// of section `i`.  Under a unit step the running gain `g` accumulates as the
/// product of per-section DC gains
/// `(b0 + b1 + b2) / (1 + a1 + a2)` so that `zi[i] = g_i · zi_section_i`,
/// where `g_0 = 1` and
/// `g_{i+1} = g_i · (b0_i + b1_i + b2_i) / (1 + a1_i + a2_i)`.
///
/// # Arguments
///
/// * `sos` - Slice of biquad rows, each `[b0, b1, b2, a0, a1, a2]`.  `a0` may
///   be any non-zero value and is internally normalised to 1.
///
/// # Returns
///
/// `Array2<f64>` of shape `(n_sections, 2)`, where row `i` is the
/// steady-state state `[w1, w2]` for section `i` of the cascade.
///
/// # Errors
///
/// Returns [`SignalError::ValueError`] if
///
/// * `sos` is empty,
/// * any section has `a0 ≈ 0` (cannot be normalised), or
/// * any section has `1 + a1 + a2 ≈ 0` (filter is unstable at DC, so the
///   steady-state response is undefined).
///
/// # Example
///
/// ```
/// use scirs2_signal::filter::application::sosfilt_zi;
///
/// // Allpass / pass-through cascade — initial state must be all zeros.
/// let sos = [[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]];
/// let zi = sosfilt_zi(&sos).expect("valid SOS");
/// assert_eq!(zi.shape(), &[1, 2]);
/// assert!(zi[[0, 0]].abs() < 1e-15);
/// assert!(zi[[0, 1]].abs() < 1e-15);
/// ```
#[allow(dead_code)]
pub fn sosfilt_zi(sos: &[[f64; 6]]) -> SignalResult<Array2<f64>> {
    if sos.is_empty() {
        return Err(SignalError::ValueError(
            "sosfilt_zi: SOS array must contain at least one section".to_string(),
        ));
    }

    let n_sections = sos.len();
    let mut zi = Array2::<f64>::zeros((n_sections, 2));

    // Cumulative DC gain through preceding sections — the steady-state input
    // to section `i` under a unit step at the cascade input.
    let mut scale = 1.0_f64;

    for (i, row) in sos.iter().enumerate() {
        // Normalise the section so that a0 = 1.
        let a0 = row[3];
        if a0.abs() < 1e-30 {
            return Err(SignalError::ValueError(format!(
                "sosfilt_zi: section {i}: a0 ({a0}) must be non-zero"
            )));
        }
        let inv_a0 = 1.0 / a0;
        let b0 = row[0] * inv_a0;
        let b1 = row[1] * inv_a0;
        let b2 = row[2] * inv_a0;
        let a1 = row[4] * inv_a0;
        let a2 = row[5] * inv_a0;

        // Determinant of (I - A_sys) for this biquad section.
        // det = (1 + a1) * 1 - (-1) * a2 = 1 + a1 + a2.
        let det = 1.0 + a1 + a2;
        if det.abs() < 1e-30 {
            return Err(SignalError::ValueError(format!(
                "sosfilt_zi: section {i}: 1 + a1 + a2 ({det}) is zero — \
                 filter has a pole at z = 1, steady-state is undefined"
            )));
        }

        // B_sys for the DF-II-Transposed realisation.
        let bs0 = b1 - a1 * b0;
        let bs1 = b2 - a2 * b0;

        // Solve (I - A_sys) · z = B_sys analytically (Cramer's rule).
        //   I - A_sys = [[1 + a1, -1], [a2, 1]]
        //   z[0] = ( bs0 *  1   - bs1 * (-1) ) / det = (bs0 + bs1) / det
        //   z[1] = ( (1+a1) * bs1 - a2 * bs0 ) / det
        let z0 = (bs0 + bs1) / det;
        let z1 = ((1.0 + a1) * bs1 - a2 * bs0) / det;

        zi[[i, 0]] = scale * z0;
        zi[[i, 1]] = scale * z1;

        // DC gain of this section, used to scale the next section's zi.
        // Numerator at z=1: b0 + b1 + b2.  Denominator at z=1: 1 + a1 + a2.
        let num_dc = b0 + b1 + b2;
        let den_dc = det;
        scale *= num_dc / den_dc;
    }

    Ok(zi)
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Apply a Direct-Form-II-Transposed SOS cascade to `input`, starting from
    /// the supplied per-section initial state `zi` (shape `(n_sections, 2)`).
    /// Returns the output sequence.  Used purely to validate `sosfilt_zi`
    /// against simulation; mirrors the kernel inside
    /// `streaming::ws78_block_filter::StatefulIirFilter`.
    fn simulate_sosfilt(sos: &[[f64; 6]], zi: &Array2<f64>, input: &[f64]) -> Vec<f64> {
        let n = input.len();
        let mut buf = input.to_vec();
        let mut state: Vec<[f64; 2]> = (0..sos.len()).map(|i| [zi[[i, 0]], zi[[i, 1]]]).collect();

        for (sec_idx, row) in sos.iter().enumerate() {
            let inv_a0 = 1.0 / row[3];
            let b0 = row[0] * inv_a0;
            let b1 = row[1] * inv_a0;
            let b2 = row[2] * inv_a0;
            let a1 = row[4] * inv_a0;
            let a2 = row[5] * inv_a0;

            let st = &mut state[sec_idx];
            for k in 0..n {
                let xv = buf[k];
                let w1 = st[0];
                let w2 = st[1];
                let y = b0 * xv + w1;
                st[0] = b1 * xv - a1 * y + w2;
                st[1] = b2 * xv - a2 * y;
                buf[k] = y;
            }
        }
        buf
    }

    /// Section DC gain helper.
    fn section_dc_gain(row: &[f64; 6]) -> f64 {
        let inv_a0 = 1.0 / row[3];
        let num = (row[0] + row[1] + row[2]) * inv_a0;
        let den = 1.0 + row[4] * inv_a0 + row[5] * inv_a0;
        num / den
    }

    #[test]
    fn test_sosfilt_zi_allpass_identity() {
        // Pure pass-through biquad: y[n] = x[n].  Steady-state state must be
        // exactly zero in both delays.
        let sos = [[1.0_f64, 0.0, 0.0, 1.0, 0.0, 0.0]];
        let zi = sosfilt_zi(&sos).expect("allpass should succeed");
        assert_eq!(zi.shape(), &[1, 2]);
        assert!(zi[[0, 0]].abs() < 1e-15, "zi[0,0] = {}", zi[[0, 0]]);
        assert!(zi[[0, 1]].abs() < 1e-15, "zi[0,1] = {}", zi[[0, 1]]);
    }

    #[test]
    fn test_sosfilt_zi_constant_gain() {
        // y[n] = 2 * x[n].  Delays carry no information, so zi is zero, but
        // the cascade gain through this section is 2 (relevant for cascades).
        let sos = [[2.0_f64, 0.0, 0.0, 1.0, 0.0, 0.0]];
        let zi = sosfilt_zi(&sos).expect("gain section should succeed");
        assert!(zi[[0, 0]].abs() < 1e-15);
        assert!(zi[[0, 1]].abs() < 1e-15);

        // Steady-state check via simulation: with x = 1 forever, y = 2.
        let x = vec![1.0_f64; 16];
        let y = simulate_sosfilt(&sos, &zi, &x);
        for (k, &yi) in y.iter().enumerate() {
            assert!((yi - 2.0).abs() < 1e-12, "y[{k}] = {yi}, expected 2.0");
        }
    }

    #[test]
    fn test_sosfilt_zi_first_order_lowpass_padded() {
        // First-order lowpass padded as a biquad:
        //   b = [0.5, 0.5, 0],  a = [1, -0.5, 0]
        // Hand-computed steady-state: zi = [1.5, 0.0].
        let sos = [[0.5_f64, 0.5, 0.0, 1.0, -0.5, 0.0]];
        let zi = sosfilt_zi(&sos).expect("lowpass should succeed");
        assert!(
            (zi[[0, 0]] - 1.5).abs() < 1e-12,
            "zi[0,0] = {}, expected 1.5",
            zi[[0, 0]]
        );
        assert!(
            zi[[0, 1]].abs() < 1e-12,
            "zi[0,1] = {}, expected 0",
            zi[[0, 1]]
        );

        // Empirical: feeding x = 1 forever should give a constant y = DC gain
        // with no transient.  DC gain = (0.5+0.5)/(1-0.5) = 2.0.
        let x = vec![1.0_f64; 32];
        let y = simulate_sosfilt(&sos, &zi, &x);
        for (k, &yi) in y.iter().enumerate() {
            assert!((yi - 2.0).abs() < 1e-12, "y[{k}] = {yi}, expected 2.0");
        }
    }

    #[test]
    fn test_sosfilt_zi_steady_state_no_transient() {
        // A non-trivial second-order section with both b and a coefficients
        // populated.  This is the canonical SciPy validation: feeding a
        // step input pre-loaded with sosfilt_zi must produce a step response
        // equal to the section's DC gain at every sample (no transient).
        let sos = [[0.0675_f64, 0.135, 0.0675, 1.0, -1.143, 0.413]];
        let zi = sosfilt_zi(&sos).expect("Butterworth-like section");

        let dc_gain = section_dc_gain(&sos[0]);

        let x = vec![1.0_f64; 64];
        let y = simulate_sosfilt(&sos, &zi, &x);
        for (k, &yi) in y.iter().enumerate() {
            assert!(
                (yi - dc_gain).abs() < 1e-10,
                "y[{k}] = {yi}, expected {dc_gain} (no-transient property violated)"
            );
        }
    }

    #[test]
    fn test_sosfilt_zi_multi_section_cascade() {
        // Two-biquad cascade.  Verifies (a) the per-section state is correct,
        // and (b) the gain-propagation `scale *= dc_gain_i` rule is right by
        // checking the cascade output equals the product of section DC gains
        // when fed a unit step pre-loaded with zi.
        let sos: [[f64; 6]; 2] = [
            [0.5, 0.5, 0.0, 1.0, -0.5, 0.0],    // DC gain = 2.0
            [0.25, 0.0, 0.25, 1.0, -0.5, 0.25], // arbitrary stable 2nd-order
        ];
        let zi = sosfilt_zi(&sos).expect("multi-section should succeed");
        assert_eq!(zi.shape(), &[2, 2]);

        // Cascade DC gain = product of per-section DC gains.
        let g0 = section_dc_gain(&sos[0]);
        let g1 = section_dc_gain(&sos[1]);
        let cascade_dc = g0 * g1;

        // Sanity: section 0's zi matches a single-section computation.
        let zi_solo0 = sosfilt_zi(&[sos[0]]).expect("single section");
        assert!((zi[[0, 0]] - zi_solo0[[0, 0]]).abs() < 1e-12);
        assert!((zi[[0, 1]] - zi_solo0[[0, 1]]).abs() < 1e-12);

        // Section 1's zi is g0 times the standalone result for section 1
        // (because the steady-state input to section 1 is g0).
        let zi_solo1 = sosfilt_zi(&[sos[1]]).expect("single section");
        assert!(
            (zi[[1, 0]] - g0 * zi_solo1[[0, 0]]).abs() < 1e-12,
            "cascade scaling broken at zi[1,0]: got {}, expected {}",
            zi[[1, 0]],
            g0 * zi_solo1[[0, 0]]
        );
        assert!(
            (zi[[1, 1]] - g0 * zi_solo1[[0, 1]]).abs() < 1e-12,
            "cascade scaling broken at zi[1,1]: got {}, expected {}",
            zi[[1, 1]],
            g0 * zi_solo1[[0, 1]]
        );

        // Steady-state: feeding x = 1 with zi pre-loaded must produce
        // y = cascade_dc at every sample.
        let x = vec![1.0_f64; 48];
        let y = simulate_sosfilt(&sos, &zi, &x);
        for (k, &yi) in y.iter().enumerate() {
            assert!(
                (yi - cascade_dc).abs() < 1e-10,
                "y[{k}] = {yi}, expected {cascade_dc}"
            );
        }
    }

    #[test]
    fn test_sosfilt_zi_a0_normalization() {
        // The section is the same lowpass as `test_sosfilt_zi_first_order_lowpass_padded`
        // but every coefficient is scaled by 3.0 (so a0 = 3.0).  After
        // normalisation the result must match the unscaled case.
        let sos_scaled = [[1.5_f64, 1.5, 0.0, 3.0, -1.5, 0.0]];
        let zi = sosfilt_zi(&sos_scaled).expect("scaled section should succeed");
        assert!((zi[[0, 0]] - 1.5).abs() < 1e-12);
        assert!(zi[[0, 1]].abs() < 1e-12);
    }

    #[test]
    fn test_sosfilt_zi_empty_error() {
        let sos: [[f64; 6]; 0] = [];
        let r = sosfilt_zi(&sos);
        assert!(r.is_err(), "empty SOS must return Err");
    }

    #[test]
    fn test_sosfilt_zi_zero_a0_error() {
        let sos = [[1.0_f64, 0.0, 0.0, 0.0, 0.0, 0.0]];
        let r = sosfilt_zi(&sos);
        assert!(r.is_err(), "a0 == 0 must return Err");
    }

    #[test]
    fn test_sosfilt_zi_dc_singular_error() {
        // 1 + a1 + a2 = 0 — the filter has a pole at z = 1, so DC is
        // unbounded.  sosfilt_zi must reject this.
        let sos = [[0.5_f64, 0.0, 0.0, 1.0, -1.5, 0.5]];
        // Confirm the singularity: 1 + (-1.5) + 0.5 = 0.
        let r = sosfilt_zi(&sos);
        assert!(r.is_err(), "DC-singular section must return Err");
    }

    #[test]
    fn test_sosfilt_zi_scipy_no_transient_property() {
        // Definitive scipy-equivalence test (matches the `_lfilter_zi` /
        // `sosfilt_zi` validation in scipy.signal): under a unit step input
        // pre-loaded with the state returned by `sosfilt_zi`, the cascade
        // output must equal the cascade DC gain at every sample with no
        // transient at the start.
        //
        // The two biquads below are stable second-order sections with the
        // shape produced by typical IIR lowpass designs (Butterworth-style:
        // numerator [1, 2, 1] before scaling, complex-conjugate pole pairs
        // inside the unit circle, paired so each section has a positive
        // real DC denominator).  The numerical values are *not* a verbatim
        // SciPy butter() output — what's tested is the *property* SciPy uses
        // to validate sosfilt_zi.
        let sos: [[f64; 6]; 2] = [
            [
                0.010_232_952_984_399_8,
                0.020_465_905_968_799_6,
                0.010_232_952_984_399_8,
                1.0,
                -0.532_075_368_338_206,
                0.141_421_356_237_309_5,
            ],
            [
                1.0,
                2.0,
                1.0,
                1.0,
                -0.717_654_792_862_181_7,
                0.533_314_732_287_267_8,
            ],
        ];

        let zi = sosfilt_zi(&sos).expect("4th-order biquad cascade");
        assert_eq!(zi.shape(), &[2, 2]);

        // Compute the cascade's true DC gain analytically from the
        // coefficients; this is the value the no-transient step response
        // must equal at every sample.
        let g0 = section_dc_gain(&sos[0]);
        let g1 = section_dc_gain(&sos[1]);
        let cascade_dc = g0 * g1;
        assert!(
            cascade_dc.is_finite(),
            "cascade DC gain must be finite, got {cascade_dc}"
        );

        // Steady-state property: the simulated step response, primed with
        // sosfilt_zi, must equal cascade_dc at every sample (no transient).
        let x = vec![1.0_f64; 128];
        let y = simulate_sosfilt(&sos, &zi, &x);
        for (k, &yi) in y.iter().enumerate() {
            assert!(
                (yi - cascade_dc).abs() < 1e-10,
                "biquad cascade primed with sosfilt_zi must produce a \
                 transient-free step response: y[{k}] = {yi}, expected {cascade_dc}"
            );
        }

        // Also: zi values should be finite and non-zero (the filter has
        // genuine internal state).
        for (i, j) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
            assert!(
                zi[[i, j]].is_finite(),
                "zi[{i},{j}] is non-finite: {}",
                zi[[i, j]]
            );
        }
        // Section 0 has non-trivial state (zi[0,*] should not all be zero).
        assert!(zi[[0, 0]].abs() > 1e-6 || zi[[0, 1]].abs() > 1e-6);
    }
}
