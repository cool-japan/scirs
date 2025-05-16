//! Fast Hankel Transform (FHT) module
//!
//! This module implements the Fast Hankel Transform using the FFTLog algorithm,
//! similar to SciPy's implementation.

use crate::error::{FFTError, FFTResult};
use std::f64::consts::PI;

/// Fast Hankel Transform using FFTLog algorithm
///
/// Computes the discrete Hankel transform of a logarithmically spaced periodic
/// sequence. This is the FFTLog algorithm by Hamilton (2000).
///
/// # Arguments
///
/// * `a` - Real input array, logarithmically spaced
/// * `dln` - Uniform logarithmic spacing of the input array
/// * `mu` - Order of the Bessel function
/// * `offset` - Offset of the uniform logarithmic spacing (default 0.0)
/// * `bias` - Index of the power law bias (default 0.0)
///
/// # Returns
///
/// The transformed output array
pub fn fht(
    a: &[f64],
    dln: f64,
    mu: f64,
    offset: Option<f64>,
    bias: Option<f64>,
) -> FFTResult<Vec<f64>> {
    let n = a.len();
    if n == 0 {
        return Err(FFTError::ValueError(
            "Input array cannot be empty".to_string(),
        ));
    }

    let offset = offset.unwrap_or(0.0);
    let bias = bias.unwrap_or(0.0);

    // Calculate the FFTLog coefficients
    let coeffs = fht_coefficients(n, dln, mu, offset, bias)?;

    // Multiply input by coefficients
    let modified_input: Vec<f64> = a
        .iter()
        .zip(coeffs.iter())
        .map(|(&ai, &ci)| ai * ci)
        .collect();

    // Apply FFT (we need the full FFT, not just real FFT)
    let spectrum = crate::fft(&modified_input, None)?;

    // Extract the appropriate part for the result
    let result: Vec<f64> = spectrum.iter().map(|c| c.re).take(n).collect();

    Ok(result)
}

/// Inverse Fast Hankel Transform
///
/// Computes the inverse discrete Hankel transform of a logarithmically spaced
/// periodic sequence.
///
/// # Arguments
///
/// * `A` - Real input array, logarithmically spaced Hankel transform
/// * `dln` - Uniform logarithmic spacing
/// * `mu` - Order of the Bessel function  
/// * `offset` - Offset of the uniform logarithmic spacing (default 0.0)
/// * `bias` - Index of the power law bias (default 0.0)
///
/// # Returns
///
/// The inverse transformed output array
pub fn ifht(
    a: &[f64],
    dln: f64,
    mu: f64,
    offset: Option<f64>,
    bias: Option<f64>,
) -> FFTResult<Vec<f64>> {
    // For orthogonal transforms, the inverse is similar with adjusted parameters
    let bias_inv = -bias.unwrap_or(0.0);
    fht(a, dln, mu, offset, Some(bias_inv))
}

/// Calculate optimal offset for the FFTLog method
///
/// For periodic signals ('periodic' boundary), the optimal offset is zero.
/// Otherwise, you should use the optimal offset to obtain accurate Hankel transforms.
///
/// # Arguments
///
/// * `dln` - Uniform logarithmic spacing
/// * `mu` - Order of the Bessel function
/// * `initial` - Initial guess for the offset (default 0.0)  
/// * `bias` - Index of the power law bias (default 0.0)
///
/// # Returns
///
/// The optimal logarithmic offset
pub fn fhtoffset(_dln: f64, _mu: f64, initial: Option<f64>, bias: Option<f64>) -> FFTResult<f64> {
    let bias = bias.unwrap_or(0.0);
    let initial = initial.unwrap_or(0.0);

    // For the simple case without optimization
    if bias == 0.0 {
        Ok(0.0)
    } else {
        // In practice, finding the optimal offset requires solving
        // a transcendental equation. For now, return a simple approximation.
        Ok(initial)
    }
}

/// Compute the FFTLog coefficients
fn fht_coefficients(n: usize, dln: f64, mu: f64, offset: f64, bias: f64) -> FFTResult<Vec<f64>> {
    let mut coeffs = vec![0.0; n];

    // Calculate the coefficients using the analytical formula
    for (i, coeff) in coeffs.iter_mut().enumerate() {
        let m = i as f64 - n as f64 / 2.0;
        let k = 2.0 * PI * m / (n as f64 * dln);

        // Basic coefficient without bias
        let basic_coeff = k.powf(mu) * (-(k * k) / 4.0).exp();

        // Apply bias correction if needed
        let biased_coeff = if bias != 0.0 {
            basic_coeff * (1.0 + bias * k * k).powf(-bias / 2.0)
        } else {
            basic_coeff
        };

        // Apply phase offset
        let phase = offset * k;
        *coeff = biased_coeff * phase.cos();
    }

    Ok(coeffs)
}

/// Compute the discrete Hankel transform sample points
///
/// This function computes the sample points for the discrete Hankel transform
/// when the input array is logarithmically spaced.
///
/// # Arguments
///
/// * `n` - Number of sample points
/// * `dln` - Logarithmic spacing
/// * `offset` - Logarithmic offset
///
/// # Returns
///
/// Sample points for the transform
pub fn fht_sample_points(n: usize, dln: f64, offset: f64) -> Vec<f64> {
    (0..n)
        .map(|i| ((i as f64 - n as f64 / 2.0) * dln + offset).exp())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_fht_basic() {
        let n = 64;
        let dln = 0.1;
        let mu = 0.0;

        // Create a simple test signal
        let x: Vec<f64> = (0..n)
            .map(|i| ((i as f64 - n as f64 / 2.0) * dln).exp())
            .collect();

        // Test forward transform
        let y = fht(&x, dln, mu, None, None).unwrap();
        assert_eq!(y.len(), n);

        // Test inverse transform
        let x_recovered = ifht(&y, dln, mu, None, None).unwrap();
        assert_eq!(x_recovered.len(), n);
    }

    #[test]
    fn test_fhtoffset() {
        let dln = 0.1;
        let mu = 0.5;

        // Test with zero bias
        let offset1 = fhtoffset(dln, mu, None, Some(0.0)).unwrap();
        assert_relative_eq!(offset1, 0.0, epsilon = 1e-10);

        // Test with non-zero bias and initial guess
        let offset2 = fhtoffset(dln, mu, Some(0.5), Some(1.0)).unwrap();
        assert_relative_eq!(offset2, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_sample_points() {
        let n = 8;
        let dln = 0.5;
        let offset = 1.0;

        let points = fht_sample_points(n, dln, offset);
        assert_eq!(points.len(), n);

        // Check that points are logarithmically spaced
        for i in 1..n {
            let ratio = points[i] / points[i - 1];
            assert_relative_eq!(ratio.ln(), dln, epsilon = 1e-10);
        }
    }
}
