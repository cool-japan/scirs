// Filter transformation functions
//
// This module provides comprehensive transformation functions for converting between
// different filter representations including analog-to-digital transforms, frequency
// transformations, and conversions between zeros-poles-gain and transfer function forms.

use super::common::FilterCoefficients;
use crate::error::{SignalError, SignalResult};
use crate::lti::design::tf;
use crate::lti::TransferFunction;
use scirs2_core::numeric::Complex64;
use scirs2_core::numeric::Zero;

#[allow(unused_imports)]
/// Apply bilinear transform to convert analog filter to digital
///
/// The bilinear transform is a method for converting analog filter designs to digital
/// filter designs. It maps the s-plane to the z-plane using the transformation:
/// s = 2 * (z - 1) / (z + 1)
///
/// # Arguments
///
/// * `zeros` - Analog zeros in the s-domain
/// * `poles` - Analog poles in the s-domain
/// * `gain` - Analog gain
/// * `sample_rate` - Sampling rate (Hz)
///
/// # Returns
///
/// * Tuple of (digital_zeros, digital_poles, digital_gain)
///
/// # Examples
///
/// ```
/// use scirs2_signal::filter::transform::bilinear_transform;
/// use scirs2_core::numeric::Complex64;
///
/// // Convert simple analog integrator to digital
/// let analog_poles = vec![Complex64::new(0.0, 0.0)];
/// let analog_zeros = vec![];
/// let (z, p, k) = bilinear_transform(&analog_zeros, &analog_poles, 1.0, 1000.0).unwrap();
/// ```
#[allow(dead_code)]
pub fn bilinear_transform(
    zeros: &[Complex64],
    poles: &[Complex64],
    gain: f64,
    sample_rate: f64,
) -> SignalResult<(Vec<Complex64>, Vec<Complex64>, f64)> {
    if sample_rate <= 0.0 {
        return Err(SignalError::ValueError(
            "Sample _rate must be positive".to_string(),
        ));
    }

    let fs_2 = sample_rate / 2.0;
    let mut digital_zeros = Vec::new();
    let mut digital_poles = Vec::new();

    // Transform zeros: z = (fs/2 + s) / (fs/2 - s)
    for &zero in zeros {
        let digital_zero = (fs_2 + zero) / (fs_2 - zero);
        digital_zeros.push(digital_zero);
    }

    // Transform poles: z = (fs/2 + s) / (fs/2 - s)
    for &pole in poles {
        let digital_pole = (fs_2 + pole) / (fs_2 - pole);
        digital_poles.push(digital_pole);
    }

    // Adjust gain for the bilinear transform
    // The gain adjustment accounts for the frequency scaling
    let degree_diff = poles.len() as i32 - zeros.len() as i32;
    let digital_gain = gain * fs_2.powi(degree_diff);

    Ok((digital_zeros, digital_poles, digital_gain))
}

/// Convert zeros, poles, and gain to transfer function coefficients
///
/// Converts a filter representation in zeros-poles-gain form to
/// transfer function coefficients (numerator and denominator polynomials).
/// This is a more comprehensive version than the internal zpk_to_tf function.
///
/// # Arguments
///
/// * `zeros` - Filter zeros
/// * `poles` - Filter poles
/// * `gain` - Filter gain
///
/// # Returns
///
/// * Tuple of (numerator_coeffs, denominator_coeffs)
///
/// # Examples
///
/// ```
/// use scirs2_signal::filter::transform::zpk_to_tf;
/// use scirs2_core::numeric::Complex64;
///
/// // Convert simple first-order system
/// let zeros = vec![];
/// let poles = vec![Complex64::new(-1.0, 0.0)];
/// let (b, a) = zpk_to_tf(&zeros, &poles, 1.0).unwrap();
/// ```
#[allow(dead_code)]
pub fn zpk_to_tf(
    zeros: &[Complex64],
    poles: &[Complex64],
    gain: f64,
) -> SignalResult<FilterCoefficients> {
    // Build numerator polynomial from zeros
    let mut num_poly = vec![Complex64::new(1.0, 0.0)];
    for &zero in zeros {
        // Multiply polynomial by (z - zero)
        let mut new_poly = vec![Complex64::new(0.0, 0.0); num_poly.len() + 1];

        // Multiply by z (shift coefficients)
        for (i, &coeff) in num_poly.iter().enumerate() {
            new_poly[i] += coeff;
        }

        // Subtract zero times polynomial
        for (i, &coeff) in num_poly.iter().enumerate() {
            new_poly[i + 1] -= zero * coeff;
        }

        num_poly = new_poly;
    }

    // Build denominator polynomial from poles
    let mut den_poly = vec![Complex64::new(1.0, 0.0)];
    for &pole in poles {
        // Multiply polynomial by (z - pole)
        let mut new_poly = vec![Complex64::new(0.0, 0.0); den_poly.len() + 1];

        // Multiply by z (shift coefficients)
        for (i, &coeff) in den_poly.iter().enumerate() {
            new_poly[i] += coeff;
        }

        // Subtract pole times polynomial
        for (i, &coeff) in den_poly.iter().enumerate() {
            new_poly[i + 1] -= pole * coeff;
        }

        den_poly = new_poly;
    }

    // Apply gain to numerator
    for coeff in &mut num_poly {
        *coeff *= gain;
    }

    // Convert complex coefficients to real (should be real for proper filter design)
    let b: Vec<f64> = num_poly
        .iter()
        .map(|c| {
            if c.im.abs() > 1e-10 {
                eprintln!(
                    "Warning: Numerator coefficient has significant imaginary part: {}",
                    c.im
                );
            }
            c.re
        })
        .collect();

    let a: Vec<f64> = den_poly
        .iter()
        .map(|c| {
            if c.im.abs() > 1e-10 {
                eprintln!(
                    "Warning: Denominator coefficient has significant imaginary part: {}",
                    c.im
                );
            }
            c.re
        })
        .collect();

    // Ensure denominator is monic (leading coefficient = 1)
    if a.is_empty() || a[0].abs() < 1e-15 {
        return Err(SignalError::ValueError(
            "Invalid denominator polynomial".to_string(),
        ));
    }

    let a0 = a[0];
    let b_normalized: Vec<f64> = b.iter().map(|&coeff| coeff / a0).collect();
    let a_normalized: Vec<f64> = a.iter().map(|&coeff| coeff / a0).collect();

    Ok((b_normalized, a_normalized))
}

/// Convert transfer function coefficients to zeros, poles, and gain
///
/// Converts transfer function coefficients (numerator and denominator polynomials)
/// to zeros-poles-gain representation. This is the inverse of zpk_to_tf.
///
/// # Arguments
///
/// * `b` - Numerator coefficients
/// * `a` - Denominator coefficients
///
/// # Returns
///
/// * Tuple of (zeros, poles, gain)
///
/// # Examples
///
/// ```
/// use scirs2_signal::filter::transform::tf_to_zpk;
///
/// // Convert simple first-order transfer function
/// let b = vec![1.0];
/// let a = vec![1.0, 1.0]; // H(z) = 1 / (z + 1)
/// let (zeros, poles, gain) = tf_to_zpk(&b, &a).unwrap();
/// ```
#[allow(dead_code)]
pub fn tf_to_zpk(b: &[f64], a: &[f64]) -> SignalResult<(Vec<Complex64>, Vec<Complex64>, f64)> {
    if a.is_empty() || a[0].abs() < 1e-15 {
        return Err(SignalError::ValueError(
            "Invalid denominator coefficients".to_string(),
        ));
    }

    // Find zeros (roots of numerator)
    let zeros = if b.is_empty() || b.iter().all(|&x: &f64| x.abs() < 1e-15) {
        Vec::new()
    } else {
        find_polynomial_roots(b)?
    };

    // Find poles (roots of denominator)
    let poles = find_polynomial_roots(a)?;

    // Calculate gain from leading coefficients
    let gain = if b.is_empty() { 0.0 } else { b[0] / a[0] };

    Ok((zeros, poles, gain))
}

/// Apply lowpass to lowpass frequency transformation
///
/// Transforms a lowpass prototype filter to another lowpass filter with
/// different cutoff frequency.
///
/// # Arguments
///
/// * `zeros` - Prototype zeros
/// * `poles` - Prototype poles
/// * `gain` - Prototype gain
/// * `new_cutoff` - New cutoff frequency (normalized, 0 to 1)
///
/// # Returns
///
/// * Transformed (zeros, poles, gain)
#[allow(dead_code)]
pub fn lp_to_lp_transform(
    zeros: &[Complex64],
    poles: &[Complex64],
    gain: f64,
    new_cutoff: f64,
) -> SignalResult<(Vec<Complex64>, Vec<Complex64>, f64)> {
    if new_cutoff <= 0.0 || new_cutoff >= 1.0 {
        return Err(SignalError::ValueError(
            "Cutoff frequency must be between 0 and 1".to_string(),
        ));
    }

    // Scale factor for frequency transformation
    let scale = (std::f64::consts::PI * new_cutoff).tan();

    // Transform zeros and poles by scaling
    let transformed_zeros: Vec<_> = zeros.iter().map(|&z| z * scale).collect();
    let transformed_poles: Vec<_> = poles.iter().map(|&p| p * scale).collect();

    // Adjust gain
    let order_diff = poles.len() as i32 - zeros.len() as i32;
    let transformed_gain = gain * scale.powi(order_diff);

    Ok((transformed_zeros, transformed_poles, transformed_gain))
}

/// Apply lowpass to highpass frequency transformation
///
/// Transforms a lowpass prototype filter to a highpass filter using
/// the transformation s -> cutoff_freq / s.
///
/// # Arguments
///
/// * `zeros` - Prototype zeros
/// * `poles` - Prototype poles  
/// * `gain` - Prototype gain
/// * `cutoff` - Cutoff frequency (normalized, 0 to 1)
///
/// # Returns
///
/// * Transformed (zeros, poles, gain)
#[allow(dead_code)]
pub fn lp_to_hp_transform(
    zeros: &[Complex64],
    poles: &[Complex64],
    gain: f64,
    cutoff: f64,
) -> SignalResult<(Vec<Complex64>, Vec<Complex64>, f64)> {
    if cutoff <= 0.0 || cutoff >= 1.0 {
        return Err(SignalError::ValueError(
            "Cutoff frequency must be between 0 and 1".to_string(),
        ));
    }

    let wc = (std::f64::consts::PI * cutoff).tan();

    // Transform zeros: s -> wc / s
    let mut transformed_zeros = Vec::new();
    for &zero in zeros {
        if zero.norm() > 1e-10 {
            transformed_zeros.push(wc / zero);
        } else {
            // Zero at origin becomes zero at infinity (not represented)
        }
    }

    // Add zeros at origin for each pole that doesn't have a corresponding zero
    let num_added_zeros = poles.len() - zeros.len();
    for _ in 0..num_added_zeros {
        transformed_zeros.push(Complex64::new(0.0, 0.0));
    }

    // Transform poles: s -> wc / s
    let transformed_poles: Vec<_> = poles.iter().map(|&p| wc / p).collect();

    // Adjust gain: multiply by (-1)^n * wc^(n-m) where n=poles, m=zeros
    let n = poles.len() as i32;
    let m = zeros.len() as i32;
    let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
    let transformed_gain = gain * sign * wc.powi(n - m);

    Ok((transformed_zeros, transformed_poles, transformed_gain))
}

/// Apply lowpass to bandpass frequency transformation
///
/// Transforms a lowpass prototype filter to a bandpass filter using
/// the transformation s -> (s^2 + wc^2) / (s * BW) where wc is center frequency
/// and BW is bandwidth.
///
/// # Arguments
///
/// * `zeros` - Prototype zeros
/// * `poles` - Prototype poles
/// * `gain` - Prototype gain
/// * `low_freq` - Low cutoff frequency (normalized, 0 to 1)
/// * `high_freq` - High cutoff frequency (normalized, 0 to 1)
///
/// # Returns
///
/// * Transformed (zeros, poles, gain)
#[allow(dead_code)]
pub fn lp_to_bp_transform(
    zeros: &[Complex64],
    poles: &[Complex64],
    gain: f64,
    low_freq: f64,
    high_freq: f64,
) -> SignalResult<(Vec<Complex64>, Vec<Complex64>, f64)> {
    if low_freq <= 0.0 || high_freq >= 1.0 || low_freq >= high_freq {
        return Err(SignalError::ValueError(
            "Invalid frequency band: low must be positive, high must be less than 1, and low < high".to_string(),
        ));
    }

    let wl = (std::f64::consts::PI * low_freq).tan();
    let wh = (std::f64::consts::PI * high_freq).tan();
    let wc = (wl * wh).sqrt(); // Center frequency
    let bw = wh - wl; // Bandwidth

    let mut transformed_zeros = Vec::new();
    let mut transformed_poles = Vec::new();

    // Transform each zero
    for &zero in zeros {
        // Apply bandpass transformation: s -> (s^2 + wc^2) / (s * BW)
        // This creates two zeros for each original zero
        let discriminant = (bw * zero / 2.0).powi(2) + wc.powi(2);
        if discriminant.re >= 0.0 {
            let sqrt_disc = discriminant.sqrt();
            let z1 = bw * zero / 2.0 + sqrt_disc;
            let z2 = bw * zero / 2.0 - sqrt_disc;
            transformed_zeros.push(z1);
            transformed_zeros.push(z2);
        } else {
            let sqrt_disc = (-discriminant).sqrt();
            let z1 = Complex64::new((bw * zero / 2.0).re, sqrt_disc.re);
            let z2 = Complex64::new((bw * zero / 2.0).re, -sqrt_disc.re);
            transformed_zeros.push(z1);
            transformed_zeros.push(z2);
        }
    }

    // Transform each pole
    for &pole in poles {
        // Apply bandpass transformation
        let discriminant = (bw * pole / 2.0).powi(2) + wc.powi(2);
        if discriminant.re >= 0.0 {
            let sqrt_disc = discriminant.sqrt();
            let p1 = bw * pole / 2.0 + sqrt_disc;
            let p2 = bw * pole / 2.0 - sqrt_disc;
            transformed_poles.push(p1);
            transformed_poles.push(p2);
        } else {
            let sqrt_disc = (-discriminant).sqrt();
            let p1 = Complex64::new((bw * pole / 2.0).re, sqrt_disc.re);
            let p2 = Complex64::new((bw * pole / 2.0).re, -sqrt_disc.re);
            transformed_poles.push(p1);
            transformed_poles.push(p2);
        }
    }

    // Add zeros at origin for bandpass transformation
    let num_added_zeros = poles.len() - zeros.len();
    for _ in 0..num_added_zeros {
        transformed_zeros.push(Complex64::new(0.0, 0.0));
    }

    // Adjust gain
    let transformed_gain = gain * bw.powi(poles.len() as i32 - zeros.len() as i32);

    Ok((transformed_zeros, transformed_poles, transformed_gain))
}

/// Apply lowpass to bandstop frequency transformation
///
/// Transforms a lowpass prototype filter to a bandstop filter using
/// the transformation s -> (s * BW) / (s^2 + wc^2).
///
/// # Arguments
///
/// * `zeros` - Prototype zeros
/// * `poles` - Prototype poles
/// * `gain` - Prototype gain
/// * `low_freq` - Low cutoff frequency (normalized, 0 to 1)
/// * `high_freq` - High cutoff frequency (normalized, 0 to 1)
///
/// # Returns
///
/// * Transformed (zeros, poles, gain)
#[allow(dead_code)]
pub fn lp_to_bs_transform(
    zeros: &[Complex64],
    poles: &[Complex64],
    gain: f64,
    low_freq: f64,
    high_freq: f64,
) -> SignalResult<(Vec<Complex64>, Vec<Complex64>, f64)> {
    if low_freq <= 0.0 || high_freq >= 1.0 || low_freq >= high_freq {
        return Err(SignalError::ValueError(
            "Invalid frequency band: low must be positive, high must be less than 1, and low < high".to_string(),
        ));
    }

    let wl = (std::f64::consts::PI * low_freq).tan();
    let wh = (std::f64::consts::PI * high_freq).tan();
    let wc = (wl * wh).sqrt(); // Center frequency
    let bw = wh - wl; // Bandwidth

    let mut transformed_zeros = Vec::new();
    let mut transformed_poles = Vec::new();

    // Transform each zero
    for &zero in zeros {
        // Apply bandstop transformation: s -> (s * BW) / (s^2 + wc^2)
        if zero.norm() > 1e-10 {
            let discriminant = (bw / (2.0 * zero)).powi(2) + wc.powi(2);
            if discriminant.re >= 0.0 {
                let sqrt_disc = discriminant.sqrt();
                let z1 = bw / (2.0 * zero) + sqrt_disc;
                let z2 = bw / (2.0 * zero) - sqrt_disc;
                transformed_zeros.push(z1);
                transformed_zeros.push(z2);
            } else {
                let sqrt_disc = (-discriminant).sqrt();
                let z1 = Complex64::new((bw / (2.0 * zero)).re, sqrt_disc.re);
                let z2 = Complex64::new((bw / (2.0 * zero)).re, -sqrt_disc.re);
                transformed_zeros.push(z1);
                transformed_zeros.push(z2);
            }
        }
    }

    // Transform each pole
    for &pole in poles {
        let discriminant = (bw / (2.0 * pole)).powi(2) + wc.powi(2);
        if discriminant.re >= 0.0 {
            let sqrt_disc = discriminant.sqrt();
            let p1 = bw / (2.0 * pole) + sqrt_disc;
            let p2 = bw / (2.0 * pole) - sqrt_disc;
            transformed_poles.push(p1);
            transformed_poles.push(p2);
        } else {
            let sqrt_disc = (-discriminant).sqrt();
            let p1 = Complex64::new((bw / (2.0 * pole)).re, sqrt_disc.re);
            let p2 = Complex64::new((bw / (2.0 * pole)).re, -sqrt_disc.re);
            transformed_poles.push(p1);
            transformed_poles.push(p2);
        }
    }

    // Add notch zeros at ±j*wc for bandstop
    let num_added_zeros = 2 * poles.len() - transformed_zeros.len();
    for _ in 0..num_added_zeros / 2 {
        transformed_zeros.push(Complex64::new(0.0, wc));
        transformed_zeros.push(Complex64::new(0.0, -wc));
    }

    // Adjust gain
    let transformed_gain = gain;

    Ok((transformed_zeros, transformed_poles, transformed_gain))
}

/// Normalize filter coefficients
///
/// Normalizes filter coefficients to ensure the denominator has a leading
/// coefficient of 1.0, which is the standard form for digital filters.
///
/// # Arguments
///
/// * `b` - Numerator coefficients
/// * `a` - Denominator coefficients
///
/// # Returns
///
/// * Normalized (numerator, denominator) coefficients
#[allow(dead_code)]
pub fn normalize_coefficients(b: &[f64], a: &[f64]) -> SignalResult<FilterCoefficients> {
    if a.is_empty() || a[0].abs() < 1e-15 {
        return Err(SignalError::ValueError(
            "Invalid denominator coefficients".to_string(),
        ));
    }

    let a0 = a[0];
    let b_norm: Vec<f64> = b.iter().map(|&coeff| coeff / a0).collect();
    let a_norm: Vec<f64> = a.iter().map(|&coeff| coeff / a0).collect();

    Ok((b_norm, a_norm))
}

// Helper function for polynomial root finding
#[allow(dead_code)]
fn find_polynomial_roots(coeffs: &[f64]) -> SignalResult<Vec<Complex64>> {
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

    // For higher-order polynomials, use a simplified method
    // In production, would use more robust algorithms like Jenkins-Traub
    let max_iterations = 100;
    let tolerance = 1e-10;

    let mut estimates = Vec::with_capacity(n);
    for k in 0..n {
        let angle = 2.0 * std::f64::consts::PI * k as f64 / n as f64;
        estimates.push(Complex64::new(angle.cos(), angle.sin()));
    }

    for _iter in 0..max_iterations {
        let mut converged = true;

        for estimate in estimates.iter_mut().take(n) {
            let z = *estimate;
            let (p_val, p_prime) = evaluate_polynomial_and_derivative(&trimmed_coeffs, z);

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

    for estimate in estimates {
        let (p_val, p_deriv) = evaluate_polynomial_and_derivative(&trimmed_coeffs, estimate);
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
