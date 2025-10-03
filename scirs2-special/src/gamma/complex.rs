//! Complex number support for gamma functions

use super::approximations::{improved_lanczos_gamma, improved_lanczos_gammaln};
use super::beta::beta;
use super::core::{gamma, gammaln};
use super::digamma::digamma;

use scirs2_core::numeric::Complex64;
use std::f64::consts::PI;

/// Complex gamma function using Lanczos approximation
///
/// Implements the complex gamma function Γ(z) for z ∈ ℂ.
/// Uses the reflection formula for Re(z) < 0.5 and Lanczos approximation otherwise.
///
/// # Arguments
///
/// * `z` - Complex input value
///
/// # Returns
///
/// * Complex gamma function value Γ(z)
///
/// # Examples
///
/// ```
/// use scirs2_special::gamma_complex;
/// use scirs2_core::numeric::Complex64;
///
/// let z = Complex64::new(1.0, 0.0);
/// let result = gamma_complex(z);
/// assert!((result.re - 1.0).abs() < 1e-10);
/// assert!(result.im.abs() < 1e-10);
/// ```
pub fn gamma_complex(z: Complex64) -> Complex64 {
    // Handle special cases
    if z.re == 0.0 && z.im == 0.0 {
        return Complex64::new(f64::INFINITY, f64::NAN);
    }

    // Check for negative integers
    if z.im == 0.0 && z.re < 0.0 && z.re.fract() == 0.0 {
        return Complex64::new(f64::INFINITY, f64::NAN);
    }

    // Use reflection formula for Re(z) < 0.5
    if z.re < 0.5 {
        // Γ(z) = π / (sin(πz) * Γ(1-z))
        let pi_z = Complex64::new(PI, 0.0) * z;
        let sin_pi_z = complex_sin(pi_z);

        if sin_pi_z.norm() < 1e-15 {
            return Complex64::new(f64::INFINITY, f64::NAN);
        }

        let pi = Complex64::new(PI, 0.0);
        let oneminus_z = Complex64::new(1.0, 0.0) - z;

        return pi / (sin_pi_z * gamma_complex(oneminus_z));
    }

    // Use Lanczos approximation for Re(z) >= 0.5
    lanczos_gamma_complex(z)
}

/// Complex log gamma function with careful branch cut handling
///
/// Implements the complex log gamma function log(Γ(z)) for z ∈ ℂ.
/// The branch cut is chosen to be continuous along the negative real axis.
///
/// # Arguments
///
/// * `z` - Complex input value
///
/// # Returns
///
/// * Complex log gamma function value log(Γ(z))
///
/// # Examples
///
/// ```
/// use scirs2_special::loggamma_complex;
/// use scirs2_core::numeric::Complex64;
///
/// let z = Complex64::new(2.0, 0.0);
/// let result = loggamma_complex(z);
/// assert!((result.re - 0.0).abs() < 1e-10); // log(Γ(2)) = log(1) = 0
/// assert!(result.im.abs() < 1e-10);
/// ```
pub fn loggamma_complex(z: Complex64) -> Complex64 {
    // Handle special cases
    if z.re == 0.0 && z.im == 0.0 {
        return Complex64::new(f64::INFINITY, 0.0);
    }

    // Check for negative integers
    if z.im == 0.0 && z.re < 0.0 && z.re.fract() == 0.0 {
        return Complex64::new(f64::INFINITY, 0.0);
    }

    // Use reflection formula for Re(z) < 0.5
    if z.re < 0.5 {
        // log(Γ(z)) = log(π) - log(sin(πz)) - log(Γ(1-z))
        let pi_z = Complex64::new(PI, 0.0) * z;
        let sin_pi_z = complex_sin(pi_z);

        if sin_pi_z.norm() < 1e-15 {
            return Complex64::new(f64::INFINITY, 0.0);
        }

        let log_pi = Complex64::new(PI.ln(), 0.0);
        let log_sin_pi_z = sin_pi_z.ln();
        let oneminus_z = Complex64::new(1.0, 0.0) - z;

        return log_pi - log_sin_pi_z - loggamma_complex(oneminus_z);
    }

    // Use Lanczos approximation for Re(z) >= 0.5
    lanczos_loggamma_complex(z)
}

/// Complex digamma (psi) function
///
/// Implements the complex digamma function ψ(z) = d/dz log(Γ(z)) for z ∈ ℂ.
///
/// # Arguments
///
/// * `z` - Complex input value
///
/// # Returns
///
/// * Complex digamma function value ψ(z)
///
/// # Examples
///
/// ```
/// use scirs2_special::digamma_complex;
/// use scirs2_core::numeric::Complex64;
///
/// let z = Complex64::new(1.0, 0.0);
/// let result = digamma_complex(z);
/// // ψ(1) = -γ (Euler-Mascheroni constant)
/// assert!((result.re + 0.5772156649015329).abs() < 1e-10);
/// assert!(result.im.abs() < 1e-10);
/// ```
pub fn digamma_complex(mut z: Complex64) -> Complex64 {
    // For real values, use the real digamma function for accuracy
    if z.im.abs() < 1e-15 && z.re > 0.0 {
        let real_result = digamma(z.re);
        return Complex64::new(real_result, 0.0);
    }

    // Handle special case
    if z.re == 0.0 && z.im == 0.0 {
        return Complex64::new(f64::NEG_INFINITY, 0.0);
    }

    // Check for negative integers
    if z.im == 0.0 && z.re < 0.0 && z.re.fract() == 0.0 {
        return Complex64::new(f64::INFINITY, 0.0);
    }

    let mut result = Complex64::new(0.0, 0.0);

    // Use recurrence relation to get Re(z) > 8
    while z.re < 8.0 {
        result -= Complex64::new(1.0, 0.0) / z;
        z += Complex64::new(1.0, 0.0);
    }

    // Use asymptotic expansion for large |z|
    if z.norm() > 8.0 {
        result += asymptotic_digamma_complex(z);
    } else {
        // Fall back to numerical differentiation
        let eps = 1e-8;
        let h = Complex64::new(eps, 0.0);
        let log_gamma_plus = loggamma_complex(z + h);
        let log_gammaminus = loggamma_complex(z - h);
        result += (log_gamma_plus - log_gammaminus) / (Complex64::new(2.0, 0.0) * h);
    }

    result
}

/// Complex beta function
///
/// Implements the complex beta function B(a,b) = Γ(a)Γ(b)/Γ(a+b) for a,b ∈ ℂ.
///
/// # Arguments
///
/// * `a` - First complex parameter
/// * `b` - Second complex parameter
///
/// # Returns
///
/// * Complex beta function value B(a,b)
///
/// # Examples
///
/// ```
/// use scirs2_special::beta_complex;
/// use scirs2_core::numeric::Complex64;
///
/// let a = Complex64::new(2.0, 0.0);
/// let b = Complex64::new(3.0, 0.0);
/// let result = beta_complex(a, b);
/// assert!((result.re - 1.0/12.0).abs() < 1e-10);
/// assert!(result.im.abs() < 1e-10);
/// ```
pub fn beta_complex(a: Complex64, b: Complex64) -> Complex64 {
    // For real values, use the real beta function for accuracy
    if a.im.abs() < 1e-15 && b.im.abs() < 1e-15 && a.re > 0.0 && b.re > 0.0 {
        let real_result = beta(a.re, b.re);
        return Complex64::new(real_result, 0.0);
    }

    // Use the logarithmic form for better numerical stability
    let log_beta = loggamma_complex(a) + loggamma_complex(b) - loggamma_complex(a + b);
    log_beta.exp()
}

/// Lanczos approximation for complex gamma function
fn lanczos_gamma_complex(z: Complex64) -> Complex64 {
    // For real values, use the real gamma function for accuracy
    if z.im.abs() < 1e-15 && z.re > 0.0 {
        let real_result = gamma(z.re);
        return Complex64::new(real_result, 0.0);
    }

    let g = 7.0;
    let sqrt_2pi = (2.0 * PI).sqrt();

    // Simplified Lanczos coefficients (g=7)
    let p = [
        0.999_999_999_999_809_9,
        676.5203681218851,
        -1259.1392167224028,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507343278686905,
        -0.13857109526572012,
        9.984_369_578_019_572e-6,
        1.5056327351493116e-7,
    ];

    let zminus_one = z - Complex64::new(1.0, 0.0);
    let mut acc = Complex64::new(p[0], 0.0);

    for (i, &p_val) in p.iter().enumerate().skip(1) {
        acc += Complex64::new(p_val, 0.0) / (zminus_one + Complex64::new(i as f64, 0.0));
    }

    let t = zminus_one + Complex64::new(g + 0.5, 0.0);
    let term1 = Complex64::new(sqrt_2pi, 0.0);
    let term2 = acc;
    let term3 = t.powc(zminus_one + Complex64::new(0.5, 0.0));
    let term4 = (-t).exp();

    term1 * term2 * term3 * term4
}

/// Lanczos approximation for complex log gamma function
fn lanczos_loggamma_complex(z: Complex64) -> Complex64 {
    // For real values, use the real loggamma function for accuracy
    if z.im.abs() < 1e-15 && z.re > 0.0 {
        let real_result = gammaln(z.re);
        return Complex64::new(real_result, 0.0);
    }

    let g = 7.0;
    let log_sqrt_2pi = (2.0 * PI).sqrt().ln();

    // Simplified Lanczos coefficients (g=7)
    let p = [
        0.999_999_999_999_809_9,
        676.5203681218851,
        -1259.1392167224028,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507343278686905,
        -0.13857109526572012,
        9.984_369_578_019_572e-6,
        1.5056327351493116e-7,
    ];

    let zminus_one = z - Complex64::new(1.0, 0.0);
    let mut acc = Complex64::new(p[0], 0.0);

    for (i, &p_val) in p.iter().enumerate().skip(1) {
        acc += Complex64::new(p_val, 0.0) / (zminus_one + Complex64::new(i as f64, 0.0));
    }

    let t = zminus_one + Complex64::new(g + 0.5, 0.0);
    let log_acc = acc.ln();
    let log_t = t.ln();

    Complex64::new(log_sqrt_2pi, 0.0) + log_acc + (zminus_one + Complex64::new(0.5, 0.0)) * log_t
        - t
}

/// Asymptotic expansion for complex digamma function
fn asymptotic_digamma_complex(z: Complex64) -> Complex64 {
    let z_inv = Complex64::new(1.0, 0.0) / z;
    let z_inv_2 = z_inv * z_inv;

    z.ln() - Complex64::new(0.5, 0.0) * z_inv - z_inv_2 / Complex64::new(12.0, 0.0)
        + z_inv_2 * z_inv_2 / Complex64::new(120.0, 0.0)
        - z_inv_2 * z_inv_2 * z_inv_2 / Complex64::new(252.0, 0.0)
}

/// Complex sine function
fn complex_sin(z: Complex64) -> Complex64 {
    z.sin()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_complex_gamma_real_values() {
        // Test real values match real gamma function
        let test_values = [1.0, 2.0, 3.0, 4.0, 5.0, 0.5, 1.5, 2.5];

        for &x in &test_values {
            let z = Complex64::new(x, 0.0);
            let complex_result = gamma_complex(z);
            let real_result = gamma(x);

            assert_relative_eq!(complex_result.re, real_result, epsilon = 1e-10);
            assert!(complex_result.im.abs() < 1e-12);
        }
    }

    #[test]
    fn test_complex_gamma_properties() {
        // Test recurrence relation: Γ(z+1) = z * Γ(z)
        let test_values = [
            Complex64::new(1.5, 0.5),
            Complex64::new(2.0, 1.0),
            Complex64::new(0.5, -0.5),
        ];

        for &z in &test_values {
            let gamma_z = gamma_complex(z);
            let gamma_z_plus_1 = gamma_complex(z + Complex64::new(1.0, 0.0));
            let expected = z * gamma_z;

            assert_relative_eq!(gamma_z_plus_1.re, expected.re, epsilon = 1e-10);
            assert_relative_eq!(gamma_z_plus_1.im, expected.im, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_complex_loggamma_real_values() {
        // Test real values match real loggamma function
        let test_values = [1.0, 2.0, 3.0, 4.0, 5.0, 0.5, 1.5, 2.5];

        for &x in &test_values {
            let z = Complex64::new(x, 0.0);
            let complex_result = loggamma_complex(z);
            let real_result = gammaln(x);

            assert_relative_eq!(complex_result.re, real_result, epsilon = 1e-10);
            assert!(complex_result.im.abs() < 1e-12);
        }
    }

    #[test]
    fn test_complex_digamma_real_values() {
        // Test real values match real digamma function
        let test_values = [1.0, 2.0, 3.0, 4.0, 5.0, 1.5, 2.5];

        for &x in &test_values {
            let z = Complex64::new(x, 0.0);
            let complex_result = digamma_complex(z);
            let real_result = digamma(x);

            assert_relative_eq!(complex_result.re, real_result, epsilon = 1e-8);
            assert!(complex_result.im.abs() < 1e-10);
        }
    }

    #[test]
    fn test_complex_beta_real_values() {
        // Test real values match real beta function
        let test_pairs = [(1.0, 1.0), (2.0, 3.0), (0.5, 0.5), (1.5, 2.5)];

        for &(a, b) in &test_pairs {
            let za = Complex64::new(a, 0.0);
            let zb = Complex64::new(b, 0.0);
            let complex_result = beta_complex(za, zb);
            let real_result = beta(a, b);

            assert_relative_eq!(complex_result.re, real_result, epsilon = 1e-10);
            assert!(complex_result.im.abs() < 1e-12);
        }
    }
}
