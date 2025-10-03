//! Voigt profile functions
//!
//! The Voigt profile is the convolution of a Gaussian and Lorentzian profile,
//! commonly used in spectroscopy, astrophysics, and plasma physics.
//!
//! ## Mathematical Theory
//!
//! The Voigt profile V(x; σ, γ) is defined as the convolution:
//! ```text
//! V(x; σ, γ) = ∫_{-∞}^{∞} G(x'; σ) L(x - x'; γ) dx'
//! ```
//!
//! Where:
//! - G(x; σ) = (1/(σ√(2π))) exp(-x²/(2σ²)) is the Gaussian profile
//! - L(x; γ) = (γ/π) / (x² + γ²) is the Lorentzian profile
//!
//! ## Analytical Form
//!
//! The Voigt profile can be expressed using the Faddeeva function w(z):
//! ```text
//! V(x; σ, γ) = (Re[w(z)]) / (σ√(2π))
//! ```
//!
//! Where z = (x + iγ) / (σ√2)
//!
//! ## Properties
//! 1. **Normalization**: ∫_{-∞}^{∞} V(x; σ, γ) dx = 1
//! 2. **Symmetry**: V(-x; σ, γ) = V(x; σ, γ)
//! 3. **Limiting cases**:
//!    - γ → 0: V(x; σ, γ) → G(x; σ) (pure Gaussian)
//!    - σ → 0: V(x; σ, γ) → L(x; γ) (pure Lorentzian)

#![allow(dead_code)]

use crate::{faddeeva_complex, SpecialError, SpecialResult};
use scirs2_core::ndarray::{Array1, ArrayView1};
use scirs2_core::numeric::Complex64;
use scirs2_core::numeric::{Float, FromPrimitive};
use scirs2_core::validation::{check_finite, check_positive};
use std::fmt::{Debug, Display};

/// Voigt profile function
///
/// Computes the Voigt profile, which is the convolution of Gaussian and Lorentzian profiles.
/// This is commonly used in spectroscopy for modeling line shapes.
///
/// # Arguments
/// * `x` - Position coordinate
/// * `sigma` - Gaussian width parameter (σ > 0)
/// * `gamma` - Lorentzian width parameter (γ > 0)
///
/// # Returns
/// Value of the Voigt profile at x
///
/// # Mathematical Definition
/// V(x; σ, γ) = (Re[w(z)]) / (σ√(2π))
/// where z = (x + iγ) / (σ√2) and w(z) is the Faddeeva function
///
/// # Examples
/// ```
/// use scirs2_special::voigt_profile;
///
/// // Pure Gaussian limit (γ → 0)
/// let result = voigt_profile(0.0, 1.0, 1e-10).unwrap();
/// let gaussianmax = 1.0 / (2.0 * std::f64::consts::PI).sqrt();
/// assert!((result - gaussianmax).abs() < 1e-6);
///
/// // Symmetric property
/// let x = 1.5f64;
/// let sigma = 0.8f64;
/// let gamma = 0.3f64;
/// let v_pos = voigt_profile(x, sigma, gamma).unwrap();
/// let v_neg = voigt_profile(-x, sigma, gamma).unwrap();
/// assert!((v_pos - v_neg).abs() < 1e-10);
/// ```
#[allow(dead_code)]
pub fn voigt_profile<T>(x: T, sigma: T, gamma: T) -> SpecialResult<T>
where
    T: Float + FromPrimitive + Display + Copy + Debug,
{
    check_finite(x, "x value")?;
    check_positive(sigma, "sigma")?;
    check_positive(gamma, "gamma")?;

    let _sqrt2 = T::from_f64(std::f64::consts::SQRT_2).unwrap();
    let _sqrt2pi = T::from_f64((2.0 * std::f64::consts::PI).sqrt()).unwrap();

    // Convert to f64 for complex arithmetic (using wofz)
    let x_f64 = x
        .to_f64()
        .ok_or_else(|| SpecialError::DomainError("Cannot convert x to f64".to_string()))?;
    let sigma_f64 = sigma
        .to_f64()
        .ok_or_else(|| SpecialError::DomainError("Cannot convert sigma to f64".to_string()))?;
    let gamma_f64 = gamma
        .to_f64()
        .ok_or_else(|| SpecialError::DomainError("Cannot convert gamma to f64".to_string()))?;

    // Handle very small sigma values to avoid numerical instability
    if sigma_f64 < 1e-8 {
        // Pure Lorentzian limit
        let result = gamma_f64 / (std::f64::consts::PI * (x_f64 * x_f64 + gamma_f64 * gamma_f64));
        return T::from_f64(result).ok_or_else(|| {
            SpecialError::DomainError("Cannot convert result back to T".to_string())
        });
    }

    // Compute z = (x + iγ) / (σ√2)
    let denominator = sigma_f64 * std::f64::consts::SQRT_2;
    let z = Complex64::new(x_f64 / denominator, gamma_f64 / denominator);

    // Compute Faddeeva function w(z)
    let w_z = faddeeva_complex(z);

    // Check if the result is finite
    if !w_z.re.is_finite() {
        return Err(SpecialError::ConvergenceError(
            "Faddeeva function returned non-finite value".to_string(),
        ));
    }

    // Voigt profile: V(x) = Re[w(z)] / (σ√(2π))
    let normalization = sigma_f64 * (2.0 * std::f64::consts::PI).sqrt();
    let result = w_z.re / normalization;

    T::from_f64(result)
        .ok_or_else(|| SpecialError::DomainError("Cannot convert result back to T".to_string()))
}

/// Normalized Voigt profile
///
/// Computes the Voigt profile normalized to have unit area.
/// This is equivalent to the standard voigt_profile function.
///
/// # Arguments
/// * `x` - Position coordinate
/// * `sigma` - Gaussian width parameter (σ > 0)
/// * `gamma` - Lorentzian width parameter (γ > 0)
///
/// # Returns
/// Normalized Voigt profile value
#[allow(dead_code)]
pub fn voigt_profile_normalized<T>(x: T, sigma: T, gamma: T) -> SpecialResult<T>
where
    T: Float + FromPrimitive + Display + Copy + Debug,
{
    voigt_profile(x, sigma, gamma)
}

/// Voigt profile with different parameterization
///
/// Uses HWHM (Half-Width at Half-Maximum) parameters for both Gaussian and Lorentzian components.
///
/// # Arguments
/// * `x` - Position coordinate
/// * `fwhm_gaussian` - Gaussian FWHM (Full-Width at Half-Maximum)
/// * `fwhm_lorentzian` - Lorentzian FWHM
///
/// # Returns
/// Voigt profile value
///
/// # Note
/// The conversion from FWHM to standard parameters is:
/// - σ = fwhm_gaussian / (2√(2 ln 2))
/// - γ = fwhm_lorentzian / 2
#[allow(dead_code)]
pub fn voigt_profile_fwhm<T>(x: T, fwhm_gaussian: T, fwhmlorentzian: T) -> SpecialResult<T>
where
    T: Float + FromPrimitive + Display + Copy + Debug,
{
    check_positive(fwhm_gaussian, "fwhm_gaussian")?;
    check_positive(fwhmlorentzian, "fwhmlorentzian")?;

    // Convert FWHM to standard parameters
    let two = T::from_f64(2.0).unwrap();
    let ln2 = T::from_f64(std::f64::consts::LN_2).unwrap();
    let sqrt2ln2 = (two * ln2).sqrt();

    let sigma = fwhm_gaussian / (two * sqrt2ln2);
    let gamma = fwhmlorentzian / two;

    voigt_profile(x, sigma, gamma)
}

/// Voigt profile for arrays
///
/// Computes the Voigt profile for an array of x values.
///
/// # Arguments
/// * `x` - Array of position coordinates
/// * `sigma` - Gaussian width parameter
/// * `gamma` - Lorentzian width parameter
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_special::voigt_profile_array;
///
/// let x = array![-2.0f64, -1.0, 0.0, 1.0, 2.0];
/// let result = voigt_profile_array(&x.view(), 1.0f64, 0.5f64).unwrap();
///
/// // Check symmetry
/// assert!((result[0] - result[4]).abs() < 1e-10);
/// assert!((result[1] - result[3]).abs() < 1e-10);
///
/// // Maximum should be at center
/// assert!(result[2] >= result[0]);
/// assert!(result[2] >= result[1]);
/// ```
#[allow(dead_code)]
pub fn voigt_profile_array<T>(x: &ArrayView1<T>, sigma: T, gamma: T) -> SpecialResult<Array1<T>>
where
    T: Float + FromPrimitive + Display + Copy + Debug,
{
    let mut result = Array1::zeros(x.len());

    for (i, &val) in x.iter().enumerate() {
        result[i] = voigt_profile(val, sigma, gamma)?;
    }

    Ok(result)
}

/// Voigt profile FWHM for arrays
#[allow(dead_code)]
pub fn voigt_profile_fwhm_array<T>(
    x: &ArrayView1<T>,
    fwhm_gaussian: T,
    fwhm_lorentzian: T,
) -> SpecialResult<Array1<T>>
where
    T: Float + FromPrimitive + Display + Copy + Debug,
{
    let mut result = Array1::zeros(x.len());

    for (i, &val) in x.iter().enumerate() {
        result[i] = voigt_profile_fwhm(val, fwhm_gaussian, fwhm_lorentzian)?;
    }

    Ok(result)
}

/// Pseudo-Voigt approximation
///
/// Computes an approximation to the Voigt profile using a linear combination
/// of Gaussian and Lorentzian profiles. This is computationally faster but less accurate.
///
/// # Arguments
/// * `x` - Position coordinate
/// * `sigma` - Gaussian width parameter
/// * `gamma` - Lorentzian width parameter
/// * `eta` - Mixing parameter (0 ≤ η ≤ 1): η=0 pure Gaussian, η=1 pure Lorentzian
///
/// # Mathematical Definition
/// pV(x) = η·L(x; γ) + (1-η)·G(x; σ)
///
/// # Examples
/// ```
/// use scirs2_special::pseudo_voigt;
///
/// // Pure Gaussian (η = 0)
/// let result = pseudo_voigt(0.0, 1.0, 0.5, 0.0).unwrap();
/// let gaussianmax = 1.0 / (2.0 * std::f64::consts::PI).sqrt();
/// assert!((result - gaussianmax).abs() < 1e-10);
///
/// // Pure Lorentzian (η = 1)
/// let result = pseudo_voigt(0.0, 1.0, 0.5, 1.0).unwrap();
/// let lorentzianmax = 1.0 / (std::f64::consts::PI * 0.5);
/// assert!((result - lorentzianmax).abs() < 1e-10);
/// ```
#[allow(dead_code)]
pub fn pseudo_voigt<T>(x: T, sigma: T, gamma: T, eta: T) -> SpecialResult<T>
where
    T: Float + FromPrimitive + Display + Copy + Debug,
{
    check_finite(x, "x value")?;
    check_positive(sigma, "sigma")?;
    check_positive(gamma, "gamma")?;
    check_finite(eta, "eta value")?;

    let zero = T::zero();
    let one = T::one();

    if eta < zero || eta > one {
        return Err(SpecialError::DomainError(
            "eta must be between 0 and 1".to_string(),
        ));
    }

    let pi = T::from_f64(std::f64::consts::PI).unwrap();
    let two = T::from_f64(2.0).unwrap();

    // Gaussian component: G(x; σ) = (1/(σ√(2π))) exp(-x²/(2σ²))
    let gaussian_norm = one / (sigma * (two * pi).sqrt());
    let gaussian_exp = (-(x * x) / (two * sigma * sigma)).exp();
    let gaussian = gaussian_norm * gaussian_exp;

    // Lorentzian component: L(x; γ) = (γ/π) / (x² + γ²)
    let lorentzian = (gamma / pi) / (x * x + gamma * gamma);

    // Pseudo-Voigt: η·L + (1-η)·G
    Ok(eta * lorentzian + (one - eta) * gaussian)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::array;
    use std::f64::consts::PI;

    #[test]
    fn test_voigt_profile_basic() {
        let sigma = 1.0;
        let gamma = 0.5;

        // Test at center (should be maximum)
        let center = voigt_profile(0.0, sigma, gamma).unwrap();
        assert!(center > 0.0);

        // Test symmetry
        let x = 1.5;
        let v_pos = voigt_profile(x, sigma, gamma).unwrap();
        let v_neg = voigt_profile(-x, sigma, gamma).unwrap();
        assert_relative_eq!(v_pos, v_neg, epsilon = 1e-12);
    }

    #[test]
    fn test_voigt_limiting_cases() {
        let x = 0.0;
        let sigma = 1.0;
        let small_gamma = 1e-10;
        let _large_gamma = 1e10;

        // Test pure Gaussian limit (γ → 0)
        let gaussian_limit = voigt_profile(x, sigma, small_gamma).unwrap();
        let expected_gaussian = 1.0 / (sigma * (2.0 * PI).sqrt());
        assert_relative_eq!(gaussian_limit, expected_gaussian, epsilon = 1e-6);

        // For Lorentzian limit, we need σ → 0, not γ → ∞
        let small_sigma = 1e-10;
        let gamma = 1.0;
        let lorentzian_limit = voigt_profile(x, small_sigma, gamma).unwrap();
        let expected_lorentzian = gamma / (PI * (x * x + gamma * gamma));
        assert_relative_eq!(lorentzian_limit, expected_lorentzian, epsilon = 1e-6);
    }

    #[test]
    fn test_voigt_profile_fwhm() {
        let fwhm_g = 2.0;
        let fwhm_l = 1.0;

        // Test FWHM parameterization
        let result = voigt_profile_fwhm(0.0, fwhm_g, fwhm_l).unwrap();
        assert!(result > 0.0);

        // Test symmetry with FWHM parameterization
        let x = 1.0;
        let v_pos = voigt_profile_fwhm(x, fwhm_g, fwhm_l).unwrap();
        let v_neg = voigt_profile_fwhm(-x, fwhm_g, fwhm_l).unwrap();
        assert_relative_eq!(v_pos, v_neg, epsilon = 1e-12);
    }

    #[test]
    fn test_pseudo_voigt() {
        let sigma = 1.0;
        let gamma = 0.5;
        let x = 0.0;

        // Test pure Gaussian (η = 0)
        let pure_gaussian = pseudo_voigt(x, sigma, gamma, 0.0).unwrap();
        let expected_gaussian = 1.0 / (sigma * (2.0 * PI).sqrt());
        assert_relative_eq!(pure_gaussian, expected_gaussian, epsilon = 1e-12);

        // Test pure Lorentzian (η = 1)
        let pure_lorentzian = pseudo_voigt(x, sigma, gamma, 1.0).unwrap();
        let expected_lorentzian = gamma / (PI * (x * x + gamma * gamma));
        assert_relative_eq!(pure_lorentzian, expected_lorentzian, epsilon = 1e-12);

        // Test intermediate case
        let mixed = pseudo_voigt(x, sigma, gamma, 0.5).unwrap();
        assert!(mixed > 0.0);
        assert!(mixed < pure_gaussian.max(pure_lorentzian));
    }

    #[test]
    fn test_array_operations() {
        let x = array![-2.0, -1.0, 0.0, 1.0, 2.0];
        let sigma = 1.0;
        let gamma = 0.3;

        let result = voigt_profile_array(&x.view(), sigma, gamma).unwrap();
        assert_eq!(result.len(), 5);

        // Test symmetry in array
        assert_relative_eq!(result[0], result[4], epsilon = 1e-12);
        assert_relative_eq!(result[1], result[3], epsilon = 1e-12);

        // Maximum should be at center
        assert!(result[2] >= result[0]);
        assert!(result[2] >= result[1]);
        assert!(result[2] >= result[3]);
        assert!(result[2] >= result[4]);
    }

    #[test]
    fn test_error_conditions() {
        // Negative sigma
        assert!(voigt_profile(0.0, -1.0, 1.0).is_err());

        // Negative gamma
        assert!(voigt_profile(0.0, 1.0, -1.0).is_err());

        // Invalid eta for pseudo-Voigt
        assert!(pseudo_voigt(0.0, 1.0, 1.0, -0.1).is_err());
        assert!(pseudo_voigt(0.0, 1.0, 1.0, 1.1).is_err());
    }

    #[test]
    fn test_normalization_approximation() {
        // Test that the Voigt profile is approximately normalized
        // This is a numerical integration test
        let sigma = 1.0;
        let gamma = 0.5;
        let x_vals: Vec<f64> = (-100..=100).map(|i| i as f64 * 0.1).collect();
        let dx = 0.1;

        let mut integral = 0.0;
        for &x in &x_vals {
            integral += voigt_profile(x, sigma, gamma).unwrap() * dx;
        }

        // Should be approximately 1 (within numerical integration error)
        println!("Voigt integral: {}", integral);

        // Allow more error for numerical integration approximation
        assert!(
            (integral - 1.0).abs() < 0.1,
            "Integral {} should be close to 1.0",
            integral
        );
    }
}
