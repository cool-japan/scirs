// IIR (Infinite Impulse Response) filter design functions
//
// This module provides comprehensive IIR filter design capabilities including
// classic analog filter prototypes (Butterworth, Chebyshev, Elliptic, Bessel)
// and specialized IIR design methods. All filters use the bilinear transform
// for analog-to-digital conversion.
//
// Split by filter family (each family's lowpass/highpass design and its
// bandpass/bandstop companion are kept together in the same submodule) to
// stay under the workspace's per-file line-count policy:
// - `butterworth`: Butterworth filter design.
// - `chebyshev`: Chebyshev Type I and Type II filter design.
// - `elliptic`: Elliptic (Cauer) filter design, including the genuine
//   Jacobi-elliptic-function-based analog prototype.
// - `bessel`: Bessel (maximally-flat-group-delay) filter design.

use crate::error::{SignalError, SignalResult};
use crate::filter::common::FilterCoefficients;
#[allow(unused_imports)]
use crate::lti::design::tf as design_tf;
use crate::lti::TransferFunction;
use scirs2_core::numeric::Complex64;

#[allow(unused_imports)]
// Helper enum for handling either single values or slices
#[derive(Debug, Clone)]
pub enum Either<A, B> {
    Left(A),
    Right(B),
}

pub mod bessel;
pub mod butterworth;
pub mod chebyshev;
pub mod elliptic;

pub use bessel::{bessel, bessel_bandpass_bandstop};
pub use butterworth::{butter, butter_bandpass_bandstop};
pub use chebyshev::{cheby1, cheby1_bandpass_bandstop, cheby2, cheby2_bandpass_bandstop};
pub use elliptic::{ellip, ellip_bandpass_bandstop};

/// Convert zeros, poles, and gain to transfer function coefficients
///
/// Converts a filter representation in zeros-poles-gain form to
/// transfer function coefficients (numerator and denominator polynomials).
///
/// # Arguments
///
/// * `zeros` - Filter zeros in the z-domain
/// * `poles` - Filter poles in the z-domain  
/// * `gain` - Filter gain
///
/// # Returns
///
/// * Tuple of (numerator_coeffs, denominator_coeffs)
#[allow(dead_code)]
fn zpk_to_tf(
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

#[allow(dead_code)]
fn tf(num: Vec<f64>, den: Vec<f64>) -> TransferFunction {
    TransferFunction::new(num, den, None).expect("Operation failed")
}
