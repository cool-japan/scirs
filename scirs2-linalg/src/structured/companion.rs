//! Companion matrix eigenvalue / polynomial root finding
//!
//! The companion matrix of a monic polynomial
//!   p(x) = x^n + a_{n-1} x^{n-1} + ... + a_1 x + a_0
//! is the n x n matrix:
//!
//! ```text
//! [0  0  ... 0  -a_0  ]
//! [1  0  ... 0  -a_1  ]
//! [0  1  ... 0  -a_2  ]
//! [.  .  .   .   .    ]
//! [0  0  ... 1  -a_{n-1}]
//! ```
//!
//! Its eigenvalues are exactly the roots of p(x).

use crate::error::{LinalgError, LinalgResult};
use scirs2_core::ndarray::{Array1, Array2, ScalarOperand};
use scirs2_core::numeric::{Complex, Float, NumAssign, One, Zero};
use std::{fmt::Debug, iter::Sum};

/// Build the companion matrix from polynomial coefficients.
///
/// The coefficients should be given in **descending** power order:
/// `coeffs = [a_n, a_{n-1}, ..., a_1, a_0]` where a_n is the leading coefficient.
///
/// If a_n != 1 the polynomial is divided through to make it monic.
///
/// # Arguments
///
/// * `coeffs` - Polynomial coefficients in descending power order
///
/// # Returns
///
/// An n x n companion matrix (where n = degree of polynomial)
pub fn companion_matrix<F>(coeffs: &[F]) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Zero + Sum + One + Send + Sync + Debug,
{
    if coeffs.len() < 2 {
        return Err(LinalgError::InvalidInputError(
            "Polynomial must have at least degree 1 (coeffs must have at least 2 elements)"
                .to_string(),
        ));
    }
    let lead = coeffs[0];
    if lead.abs() < F::epsilon() {
        return Err(LinalgError::InvalidInputError(
            "Leading coefficient must be non-zero".to_string(),
        ));
    }
    let n = coeffs.len() - 1; // degree
    let mut c = Array2::<F>::zeros((n, n));

    // Sub-diagonal of ones
    for i in 1..n {
        c[[i, i - 1]] = F::one();
    }

    // Last column: -a_{n-1-k} / a_n  (negative monic coefficients)
    for i in 0..n {
        c[[i, n - 1]] = -(coeffs[n - i] / lead);
    }

    Ok(c)
}

/// Compute all roots of a polynomial using the companion matrix eigenvalues.
///
/// The polynomial is given in descending order:
/// `coeffs[0] * x^n + coeffs[1] * x^{n-1} + ... + coeffs[n-1] * x + coeffs[n]`
///
/// # Arguments
///
/// * `coeffs` - Polynomial coefficients in descending power order
///
/// # Returns
///
/// A vector of n complex roots (eigenvalues of the companion matrix)
pub fn poly_roots<F>(coeffs: &[F]) -> LinalgResult<Vec<Complex<F>>>
where
    F: Float
        + NumAssign
        + Zero
        + Sum
        + One
        + Send
        + Sync
        + Debug
        + scirs2_core::ndarray::ScalarOperand
        + 'static,
{
    if coeffs.len() < 2 {
        return Err(LinalgError::InvalidInputError(
            "Polynomial must have degree >= 1".to_string(),
        ));
    }

    // Strip leading zeros
    let start = coeffs
        .iter()
        .position(|&c| c.abs() > F::epsilon())
        .ok_or_else(|| LinalgError::InvalidInputError("All coefficients are zero".to_string()))?;
    let coeffs = &coeffs[start..];

    if coeffs.len() < 2 {
        return Err(LinalgError::InvalidInputError(
            "Polynomial (after stripping leading zeros) has degree 0; no roots".to_string(),
        ));
    }

    // Degree-1 case: a*x + b = 0 -> x = -b/a
    if coeffs.len() == 2 {
        let root = -(coeffs[1] / coeffs[0]);
        return Ok(vec![Complex::new(root, F::zero())]);
    }

    // Degree-2 case: quadratic formula (numerically stable)
    if coeffs.len() == 3 {
        return quadratic_roots(coeffs[0], coeffs[1], coeffs[2]);
    }

    // General: build companion matrix and compute eigenvalues
    let comp = companion_matrix(coeffs)?;
    let roots = companion_eigenvalues(&comp)?;
    Ok(roots)
}

/// Quadratic formula with numerical stability (Citardauq variant)
fn quadratic_roots<F>(a: F, b: F, c: F) -> LinalgResult<Vec<Complex<F>>>
where
    F: Float + NumAssign + Zero + Sum + One + Send + Sync + Debug,
{
    let disc = b * b - F::from(4.0).expect("convert") * a * c;
    if disc >= F::zero() {
        // Two real roots – use numerically stable form
        let sqrt_disc = disc.sqrt();
        let sign = if b >= F::zero() { F::one() } else { -F::one() };
        let q = -(b + sign * sqrt_disc) / F::from(2.0).expect("convert");
        let root1 = q / a;
        let root2 = c / q;
        Ok(vec![
            Complex::new(root1, F::zero()),
            Complex::new(root2, F::zero()),
        ])
    } else {
        // Two complex conjugate roots
        let sqrt_disc = (-disc).sqrt();
        let re = -b / (F::from(2.0).expect("convert") * a);
        let im = sqrt_disc / (F::from(2.0).expect("convert") * a);
        Ok(vec![Complex::new(re, im), Complex::new(re, -im)])
    }
}

/// Compute eigenvalues of the companion matrix using the Francis QR algorithm
/// (power-method / Hessenberg reduction approach).
///
/// This uses `scirs2_linalg::eigen::eig` under the hood.
fn companion_eigenvalues<F>(comp: &Array2<F>) -> LinalgResult<Vec<Complex<F>>>
where
    F: Float
        + NumAssign
        + Zero
        + Sum
        + One
        + Send
        + Sync
        + Debug
        + scirs2_core::ndarray::ScalarOperand
        + 'static,
{
    use crate::eigen::eig;
    let (eigenvalues, _) = eig(&comp.view(), None)?;
    Ok(eigenvalues.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_linear_root() {
        // 2x - 6 = 0 => x = 3
        let coeffs = [2.0_f64, -6.0];
        let roots = poly_roots(&coeffs).expect("poly_roots");
        assert_eq!(roots.len(), 1);
        assert_relative_eq!(roots[0].re, 3.0, epsilon = 1e-10);
        assert_relative_eq!(roots[0].im, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_quadratic_real_roots() {
        // x^2 - 5x + 6 = (x-2)(x-3)
        let coeffs = [1.0_f64, -5.0, 6.0];
        let mut roots = poly_roots(&coeffs).expect("poly_roots");
        roots.sort_by(|a, b| a.re.partial_cmp(&b.re).expect("cmp"));
        assert_eq!(roots.len(), 2);
        assert_relative_eq!(roots[0].re, 2.0, epsilon = 1e-8);
        assert_relative_eq!(roots[1].re, 3.0, epsilon = 1e-8);
    }

    #[test]
    fn test_quadratic_complex_roots() {
        // x^2 + 1 = 0 => roots = +/- i
        let coeffs = [1.0_f64, 0.0, 1.0];
        let roots = poly_roots(&coeffs).expect("poly_roots");
        assert_eq!(roots.len(), 2);
        for r in &roots {
            assert_relative_eq!(r.re, 0.0, epsilon = 1e-10);
            assert_relative_eq!(r.im.abs(), 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_cubic_roots() {
        // (x-1)(x-2)(x-3) = x^3 - 6x^2 + 11x - 6
        let coeffs = [1.0_f64, -6.0, 11.0, -6.0];
        let mut roots = poly_roots(&coeffs).expect("poly_roots");
        // Sort by real part
        roots.sort_by(|a, b| a.re.partial_cmp(&b.re).expect("cmp"));
        assert_eq!(roots.len(), 3);
        // All roots should be real (imaginary parts ~ 0)
        for r in &roots {
            assert_relative_eq!(r.im.abs(), 0.0, epsilon = 1e-6);
        }
        let real_roots: Vec<f64> = roots.iter().map(|r| r.re).collect();
        let mut sorted_real = real_roots.clone();
        sorted_real.sort_by(|a, b| a.partial_cmp(b).expect("cmp"));
        assert_relative_eq!(sorted_real[0], 1.0, epsilon = 1e-6);
        assert_relative_eq!(sorted_real[1], 2.0, epsilon = 1e-6);
        assert_relative_eq!(sorted_real[2], 3.0, epsilon = 1e-6);
    }

    #[test]
    fn test_companion_matrix() {
        // x^2 - 5x + 6  =>  companion:
        // [[0, -6], [1, 5]]
        let coeffs = [1.0_f64, -5.0, 6.0];
        let c = companion_matrix(&coeffs).expect("companion_matrix");
        assert_eq!(c.shape(), &[2, 2]);
        assert_relative_eq!(c[[0, 0]], 0.0, epsilon = 1e-14);
        assert_relative_eq!(c[[1, 0]], 1.0, epsilon = 1e-14);
        // last column: -a0/a_n = -(6/1)=-6, -a1/a_n = -(-5/1)=5
        assert_relative_eq!(c[[0, 1]], -6.0, epsilon = 1e-14);
        assert_relative_eq!(c[[1, 1]], 5.0, epsilon = 1e-14);
    }
}
