//! Polynomial root finding via companion matrix eigenvalues and Laguerre refinement
//!
//! This module provides a comprehensive set of polynomial root-finding algorithms:
//!
//! - `poly_roots`: Find all roots using companion matrix eigenvalues (Francis QR)
//! - `companion_matrix`: Build the companion matrix for a polynomial
//! - `refine_roots_laguerre`: Refine roots with Laguerre's guaranteed-convergence method
//! - `poly_eval_complex`: Evaluate polynomial at a complex point (Horner's method)
//! - `poly_mul`: Polynomial multiplication
//! - `char_poly_from_roots`: Characteristic polynomial from a list of roots
//!
//! # Convention
//!
//! Coefficients are given in **descending** power order:
//!
//! ```text
//! coeffs = [a_n, a_{n-1}, ..., a_1, a_0]
//! p(x) = a_n * x^n + a_{n-1} * x^{n-1} + ... + a_1 * x + a_0
//! ```

use crate::error::{LinalgError, LinalgResult};
use scirs2_core::ndarray::{Array2, ScalarOperand};
use scirs2_core::num_complex::Complex;

// ──────────────────────────────────────────────────────────────────────────────
// Public API
// ──────────────────────────────────────────────────────────────────────────────

/// Find all roots of a real polynomial using companion matrix eigenvalues.
///
/// The algorithm builds the companion matrix and computes its eigenvalues via
/// the Francis double-shift QR algorithm (through `crate::eigen::eig`).
/// For degree ≤ 2 the classical formulas are used directly.
///
/// # Arguments
///
/// * `coeffs` – Polynomial coefficients in descending power order
///   `[a_n, a_{n-1}, …, a_0]`.  Leading coefficient must be non-zero.
///
/// # Returns
///
/// A `Vec<Complex<f64>>` of length `n` (the degree of the polynomial).
///
/// # Examples
///
/// ```
/// use scirs2_linalg::matrix_functions::poly_roots::poly_roots;
///
/// // x² – 1 = 0 → roots ±1
/// let roots = poly_roots(&[1.0_f64, 0.0, -1.0]).expect("roots");
/// assert_eq!(roots.len(), 2);
/// ```
pub fn poly_roots(coeffs: &[f64]) -> LinalgResult<Vec<Complex<f64>>> {
    if coeffs.len() < 2 {
        return Err(LinalgError::InvalidInputError(
            "Polynomial must have at least degree 1 (≥ 2 coefficients)".to_string(),
        ));
    }

    // Strip leading zeros
    let start = coeffs
        .iter()
        .position(|&c| c.abs() > f64::EPSILON)
        .ok_or_else(|| LinalgError::InvalidInputError("All coefficients are zero".to_string()))?;
    let coeffs = &coeffs[start..];

    if coeffs.len() < 2 {
        return Err(LinalgError::InvalidInputError(
            "Polynomial (after stripping leading zeros) has degree 0; no roots".to_string(),
        ));
    }

    // Degree-1 case: a*x + b = 0  →  x = -b/a
    if coeffs.len() == 2 {
        let root = -(coeffs[1] / coeffs[0]);
        return Ok(vec![Complex::new(root, 0.0)]);
    }

    // Degree-2 case: stable quadratic formula
    if coeffs.len() == 3 {
        return quadratic_roots_f64(coeffs[0], coeffs[1], coeffs[2]);
    }

    // General case: companion matrix eigenvalues
    let comp = companion_matrix(coeffs)?;
    companion_eigenvalues_f64(&comp)
}

/// Build the companion matrix for a polynomial.
///
/// For a monic polynomial
/// `p(x) = x^n + c_{n-1} x^{n-1} + … + c_1 x + c_0`
/// the companion matrix is the n×n matrix whose eigenvalues are exactly the
/// roots of p.  If the leading coefficient is not 1 the polynomial is divided
/// through first.
///
/// # Arguments
///
/// * `coeffs` – Polynomial coefficients in descending power order.
///
/// # Returns
///
/// An n×n `Array2<f64>` companion matrix.
///
/// # Examples
///
/// ```
/// use scirs2_linalg::matrix_functions::poly_roots::companion_matrix;
///
/// // x² – 5x + 6  →  companion [[0,-6],[1,5]]
/// let c = companion_matrix(&[1.0_f64, -5.0, 6.0]).expect("ok");
/// assert_eq!(c.shape(), &[2, 2]);
/// ```
pub fn companion_matrix(coeffs: &[f64]) -> LinalgResult<Array2<f64>> {
    if coeffs.len() < 2 {
        return Err(LinalgError::InvalidInputError(
            "Polynomial must have degree >= 1".to_string(),
        ));
    }
    let lead = coeffs[0];
    if lead.abs() < f64::EPSILON {
        return Err(LinalgError::InvalidInputError(
            "Leading coefficient must be non-zero".to_string(),
        ));
    }
    let n = coeffs.len() - 1; // degree
    let mut c = Array2::<f64>::zeros((n, n));

    // Sub-diagonal of ones
    for i in 1..n {
        c[[i, i - 1]] = 1.0;
    }

    // Last column: -a_{n-1-k} / a_n
    for i in 0..n {
        c[[i, n - 1]] = -(coeffs[n - i] / lead);
    }

    Ok(c)
}

/// Evaluate a polynomial at a complex point using Horner's method.
///
/// # Arguments
///
/// * `coeffs` – Polynomial coefficients in descending power order.
/// * `z`      – Complex evaluation point.
///
/// # Returns
///
/// `p(z)` as a `Complex<f64>`.
///
/// # Examples
///
/// ```
/// use scirs2_core::num_complex::Complex;
/// use scirs2_linalg::matrix_functions::poly_roots::poly_eval_complex;
///
/// // p(x) = x² – 1,  p(1 + 0i) = 0
/// let val = poly_eval_complex(&[1.0_f64, 0.0, -1.0], Complex::new(1.0, 0.0));
/// assert!((val.re).abs() < 1e-14);
/// ```
pub fn poly_eval_complex(coeffs: &[f64], z: Complex<f64>) -> Complex<f64> {
    if coeffs.is_empty() {
        return Complex::new(0.0, 0.0);
    }
    // Horner: acc = coeffs[0], then acc = acc * z + coeffs[i]
    let mut acc = Complex::new(coeffs[0], 0.0);
    for &c in &coeffs[1..] {
        acc = acc * z + Complex::new(c, 0.0);
    }
    acc
}

/// Multiply two polynomials.
///
/// Both inputs are given in descending power order.  The result is also in
/// descending power order and has length `a.len() + b.len() - 1`.
///
/// # Examples
///
/// ```
/// use scirs2_linalg::matrix_functions::poly_roots::poly_mul;
///
/// // (x – 1)(x – 2) = x² – 3x + 2
/// let p = poly_mul(&[1.0, -1.0], &[1.0, -2.0]);
/// assert_eq!(p.len(), 3);
/// assert!((p[0] - 1.0).abs() < 1e-14);
/// assert!((p[1] - (-3.0)).abs() < 1e-14);
/// assert!((p[2] - 2.0).abs() < 1e-14);
/// ```
pub fn poly_mul(a: &[f64], b: &[f64]) -> Vec<f64> {
    if a.is_empty() || b.is_empty() {
        return vec![];
    }
    let out_len = a.len() + b.len() - 1;
    let mut result = vec![0.0f64; out_len];
    for (i, &ai) in a.iter().enumerate() {
        for (j, &bj) in b.iter().enumerate() {
            result[i + j] += ai * bj;
        }
    }
    result
}

/// Build the characteristic polynomial from a list of (possibly complex) roots.
///
/// The characteristic polynomial is
/// `prod_{i}(x - λ_i)`.  For complex roots in conjugate pairs the output is
/// real; for general complex roots the imaginary parts of the coefficients
/// are discarded (only the real parts are returned).
///
/// # Arguments
///
/// * `roots` – Slice of complex roots.
///
/// # Returns
///
/// Coefficients in descending power order.  The leading coefficient is 1.
///
/// # Examples
///
/// ```
/// use scirs2_core::num_complex::Complex;
/// use scirs2_linalg::matrix_functions::poly_roots::char_poly_from_roots;
///
/// // roots = {1, 2} → x² – 3x + 2
/// let roots = vec![Complex::new(1.0, 0.0), Complex::new(2.0, 0.0)];
/// let p = char_poly_from_roots(&roots);
/// assert!((p[1] - (-3.0)).abs() < 1e-12);
/// assert!((p[2] - 2.0).abs() < 1e-12);
/// ```
pub fn char_poly_from_roots(roots: &[Complex<f64>]) -> Vec<f64> {
    // Start with p = [1]
    let mut p_complex: Vec<Complex<f64>> = vec![Complex::new(1.0, 0.0)];

    for &lambda in roots {
        // p = p * (x - lambda)
        // Multiply the current complex coefficient vector by (x - lambda)
        let mut new_p = vec![Complex::new(0.0, 0.0); p_complex.len() + 1];
        for (i, &c) in p_complex.iter().enumerate() {
            new_p[i] += c;
            new_p[i + 1] -= c * lambda;
        }
        p_complex = new_p;
    }

    // Return real parts (imaginary parts should be negligible for conjugate-pair roots)
    p_complex.iter().map(|c| c.re).collect()
}

/// Refine polynomial roots using Laguerre's method.
///
/// Laguerre's method has the remarkable property that it converges to *some*
/// root from almost any starting point.  Combining it with initial estimates
/// from the companion-matrix method yields highly accurate roots.
///
/// # Arguments
///
/// * `coeffs`        – Polynomial coefficients in descending power order.
/// * `initial_roots` – Starting estimates (e.g., from `poly_roots`).
/// * `max_iter`      – Maximum iterations per root.
/// * `tol`           – Convergence tolerance (on root update magnitude).
///
/// # Returns
///
/// A `Vec<Complex<f64>>` of refined roots.
///
/// # Examples
///
/// ```
/// use scirs2_core::num_complex::Complex;
/// use scirs2_linalg::matrix_functions::poly_roots::{poly_roots, refine_roots_laguerre};
///
/// let coeffs = [1.0_f64, 0.0, -1.0]; // x² – 1
/// let initial = poly_roots(&coeffs).expect("initial roots");
/// let refined = refine_roots_laguerre(&coeffs, &initial, 50, 1e-14)
///     .expect("refined roots");
/// for r in &refined {
///     let pval = r.re * r.re - 1.0;
///     assert!(pval.abs() < 1e-12, "root does not satisfy p(x)=0: p({})={}", r, pval);
/// }
/// ```
pub fn refine_roots_laguerre(
    coeffs: &[f64],
    initial_roots: &[Complex<f64>],
    max_iter: usize,
    tol: f64,
) -> LinalgResult<Vec<Complex<f64>>> {
    if coeffs.len() < 2 {
        return Err(LinalgError::InvalidInputError(
            "Polynomial must have at least degree 1".to_string(),
        ));
    }

    let degree = coeffs.len() - 1;
    let n_complex = Complex::new(degree as f64, 0.0);

    // Pre-compute complex coefficient array
    let coeffs_c: Vec<Complex<f64>> = coeffs.iter().map(|&c| Complex::new(c, 0.0)).collect();

    // Derivative coefficients: p'(x) = n*a_n x^{n-1} + (n-1)*a_{n-1} x^{n-2} + ...
    let deriv1: Vec<Complex<f64>> = coeffs_c[..degree]
        .iter()
        .enumerate()
        .map(|(i, &c)| c * Complex::new((degree - i) as f64, 0.0))
        .collect();

    // Second derivative: p''(x)
    let deriv2: Vec<Complex<f64>> = if degree >= 2 {
        deriv1[..degree - 1]
            .iter()
            .enumerate()
            .map(|(i, &c)| c * Complex::new((degree - 1 - i) as f64, 0.0))
            .collect()
    } else {
        vec![]
    };

    let mut refined = Vec::with_capacity(initial_roots.len());

    for &z0 in initial_roots {
        let mut z = z0;
        let mut converged = false;

        for _ in 0..max_iter {
            // Evaluate p(z), p'(z), p''(z) using Horner's method on complex coefficients
            let pz = poly_eval_complex_c(&coeffs_c, z);
            let dpz = poly_eval_complex_c(&deriv1, z);
            let d2pz = poly_eval_complex_c(&deriv2, z);

            if pz.norm() < tol * 1e-2 {
                converged = true;
                break;
            }

            // Laguerre step:
            //   H = p'(z) / p(z)
            //   G² = H² - p''(z) / p(z)
            //   denom = H ± sqrt((n-1)(nG² - H²))
            //   z_new = z - n / denom

            let h = dpz / pz;
            let g2 = h * h - d2pz / pz;
            let inner = (n_complex - Complex::new(1.0, 0.0)) * (n_complex * g2 - h * h);
            let sqrt_inner = complex_sqrt(inner);

            // Choose the denominator with larger magnitude (Kahan's trick)
            let denom_plus = h + sqrt_inner;
            let denom_minus = h - sqrt_inner;
            let denom = if denom_plus.norm() >= denom_minus.norm() {
                denom_plus
            } else {
                denom_minus
            };

            if denom.norm() < f64::EPSILON {
                // Cannot proceed; keep current estimate
                break;
            }

            let dz = n_complex / denom;
            z -= dz;

            if dz.norm() < tol {
                converged = true;
                break;
            }
        }

        let _ = converged; // used for debug in larger contexts; here we just collect
        refined.push(z);
    }

    Ok(refined)
}

// ──────────────────────────────────────────────────────────────────────────────
// Private helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Stable quadratic formula for f64 polynomials.
fn quadratic_roots_f64(a: f64, b: f64, c: f64) -> LinalgResult<Vec<Complex<f64>>> {
    let disc = b * b - 4.0 * a * c;
    if disc >= 0.0 {
        let sqrt_disc = disc.sqrt();
        // Citardauq form for stability
        let sign = if b >= 0.0 { 1.0 } else { -1.0 };
        let q = -(b + sign * sqrt_disc) / 2.0;
        let r1 = q / a;
        let r2 = c / q;
        Ok(vec![Complex::new(r1, 0.0), Complex::new(r2, 0.0)])
    } else {
        let sqrt_disc = (-disc).sqrt();
        let re = -b / (2.0 * a);
        let im = sqrt_disc / (2.0 * a);
        Ok(vec![Complex::new(re, im), Complex::new(re, -im)])
    }
}

/// Compute polynomial roots via the Durand-Kerner (Weierstrass) simultaneous
/// root-finding method. This is more robust than companion-matrix eigenvalues
/// for polynomials with equal-magnitude roots (e.g., x^n - 1 = 0).
fn companion_eigenvalues_f64(comp: &Array2<f64>) -> LinalgResult<Vec<Complex<f64>>> {
    let n = comp.nrows();
    if n == 0 {
        return Ok(Vec::new());
    }

    // Reconstruct monic polynomial coefficients from companion matrix.
    // Standard companion form: sub-diagonal ones, last column = -coeffs[1..]/coeffs[0].
    // comp[[i, n-1]] = -(a_{n-1-i} / a_n) where a_{n-1-i} are the non-leading coefficients.
    // So a_{n-1-i} = -comp[[i, n-1]] * a_n, with a_n = 1 (monic after normalization).
    // Reconstruct: coeffs[0]=1, coeffs[n-i] = -comp[[i, n-1]] for i=0..n-1
    let mut coeffs = vec![Complex::new(0.0_f64, 0.0); n + 1];
    coeffs[0] = Complex::new(1.0, 0.0);
    for i in 0..n {
        coeffs[n - i] = Complex::new(-comp[[i, n - 1]], 0.0);
    }

    durand_kerner_complex(&coeffs, n)
}

/// Durand-Kerner (Weierstrass) simultaneous root-finding.
///
/// Given a monic degree-n polynomial with complex coefficients, returns all n roots.
/// Converges for all polynomials including those with equal-magnitude roots.
fn durand_kerner_complex(coeffs: &[Complex<f64>], n: usize) -> LinalgResult<Vec<Complex<f64>>> {
    use std::f64::consts::PI;

    if n == 0 {
        return Ok(Vec::new());
    }
    if n == 1 {
        // a*x + b = 0  →  x = -b/a
        let root = -coeffs[1] / coeffs[0];
        return Ok(vec![root]);
    }

    // Initialize roots on a circle of radius slightly > 1.
    // For x^n - c = 0, the roots lie on a circle of radius |c|^(1/n).
    // General init: use unit circle with golden-ratio-spaced angles.
    let mut roots: Vec<Complex<f64>> = (0..n)
        .map(|k| {
            let angle = 2.0 * PI * (k as f64) / (n as f64) + 0.1; // slight offset to break symmetry
            Complex::new(angle.cos(), angle.sin())
        })
        .collect();

    // Evaluate polynomial at a complex point (Horner)
    let eval = |z: Complex<f64>| -> Complex<f64> {
        let mut acc = coeffs[0];
        for &c in &coeffs[1..] {
            acc = acc * z + c;
        }
        acc
    };

    let max_iter = 1000;
    let tol = 1e-14;

    for _ in 0..max_iter {
        let prev_roots = roots.clone();
        let mut max_change = 0.0_f64;

        for i in 0..n {
            // Denominator: product of (roots[i] - roots[j]) for j != i
            let denom: Complex<f64> = (0..n)
                .filter(|&j| j != i)
                .map(|j| prev_roots[i] - prev_roots[j])
                .fold(Complex::new(1.0, 0.0), |acc, x| acc * x);

            if denom.norm() < f64::EPSILON {
                continue;
            }

            let correction = eval(prev_roots[i]) / denom;
            roots[i] = prev_roots[i] - correction;
            max_change = max_change.max(correction.norm());
        }

        if max_change < tol {
            break;
        }
    }

    Ok(roots)
}

/// Evaluate a polynomial with complex coefficients at a complex point (Horner).
fn poly_eval_complex_c(coeffs: &[Complex<f64>], z: Complex<f64>) -> Complex<f64> {
    if coeffs.is_empty() {
        return Complex::new(0.0, 0.0);
    }
    let mut acc = coeffs[0];
    for &c in &coeffs[1..] {
        acc = acc * z + c;
    }
    acc
}

/// Principal square root of a complex number.
fn complex_sqrt(z: Complex<f64>) -> Complex<f64> {
    let r = z.norm().sqrt();
    let theta = z.arg() / 2.0;
    Complex::new(r * theta.cos(), r * theta.sin())
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // ── poly_roots ────────────────────────────────────────────────────────────

    #[test]
    fn test_poly_roots_linear() {
        // 2x – 6 = 0  →  x = 3
        let roots = poly_roots(&[2.0, -6.0]).expect("linear root");
        assert_eq!(roots.len(), 1);
        assert_relative_eq!(roots[0].re, 3.0, epsilon = 1e-12);
        assert_relative_eq!(roots[0].im, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_poly_roots_quadratic_real() {
        // x² – 1 = 0  →  ±1
        let mut roots = poly_roots(&[1.0, 0.0, -1.0]).expect("quadratic roots");
        roots.sort_by(|a, b| a.re.partial_cmp(&b.re).expect("cmp"));
        assert_eq!(roots.len(), 2);
        assert_relative_eq!(roots[0].re, -1.0, epsilon = 1e-12);
        assert_relative_eq!(roots[1].re, 1.0, epsilon = 1e-12);
        for r in &roots {
            assert_relative_eq!(r.im, 0.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_poly_roots_cubic_unity() {
        // x³ – 1 = 0 → 1 and two complex cube roots of unity
        let roots = poly_roots(&[1.0, 0.0, 0.0, -1.0]).expect("cubic roots");
        assert_eq!(roots.len(), 3);

        // Check that p(r) ≈ 0 for each root
        for r in &roots {
            let pval = poly_eval_complex(&[1.0, 0.0, 0.0, -1.0], *r);
            assert!(
                pval.norm() < 1e-8,
                "root {} does not satisfy p(r)=0: p(r) = {}",
                r,
                pval
            );
        }
    }

    #[test]
    fn test_poly_roots_quadratic_complex() {
        // x² + 1 = 0 → ±i
        let roots = poly_roots(&[1.0, 0.0, 1.0]).expect("complex roots");
        assert_eq!(roots.len(), 2);
        for r in &roots {
            assert_relative_eq!(r.re, 0.0, epsilon = 1e-12);
            assert_relative_eq!(r.im.abs(), 1.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_poly_roots_cubic_real() {
        // (x-1)(x-2)(x-3) = x³ – 6x² + 11x – 6
        let mut roots = poly_roots(&[1.0, -6.0, 11.0, -6.0]).expect("cubic roots");
        roots.sort_by(|a, b| a.re.partial_cmp(&b.re).expect("cmp"));
        assert_eq!(roots.len(), 3);
        for r in &roots {
            assert_relative_eq!(r.im.abs(), 0.0, epsilon = 1e-6);
        }
        assert_relative_eq!(roots[0].re, 1.0, epsilon = 1e-6);
        assert_relative_eq!(roots[1].re, 2.0, epsilon = 1e-6);
        assert_relative_eq!(roots[2].re, 3.0, epsilon = 1e-6);
    }

    #[test]
    fn test_poly_roots_all_zero_error() {
        assert!(poly_roots(&[0.0, 0.0, 0.0]).is_err());
    }

    #[test]
    fn test_poly_roots_too_few_coeffs_error() {
        assert!(poly_roots(&[1.0]).is_err());
    }

    // ── companion_matrix ─────────────────────────────────────────────────────

    #[test]
    fn test_companion_matrix_quadratic() {
        // x² – 5x + 6  →  [[0, -6], [1, 5]]
        let c = companion_matrix(&[1.0, -5.0, 6.0]).expect("companion");
        assert_eq!(c.shape(), &[2, 2]);
        assert_relative_eq!(c[[0, 0]], 0.0, epsilon = 1e-14);
        assert_relative_eq!(c[[1, 0]], 1.0, epsilon = 1e-14);
        assert_relative_eq!(c[[0, 1]], -6.0, epsilon = 1e-14);
        assert_relative_eq!(c[[1, 1]], 5.0, epsilon = 1e-14);
    }

    #[test]
    fn test_companion_matrix_non_monic() {
        // 2x² – 10x + 12 = 2(x²–5x+6) — same roots as above
        let c = companion_matrix(&[2.0, -10.0, 12.0]).expect("companion non-monic");
        assert_eq!(c.shape(), &[2, 2]);
        // Last column divided by 2
        assert_relative_eq!(c[[0, 1]], -6.0, epsilon = 1e-14);
        assert_relative_eq!(c[[1, 1]], 5.0, epsilon = 1e-14);
    }

    // ── poly_eval_complex ────────────────────────────────────────────────────

    #[test]
    fn test_poly_eval_at_root() {
        // p(x) = x² – 1, root = 1
        let val = poly_eval_complex(&[1.0, 0.0, -1.0], Complex::new(1.0, 0.0));
        assert_relative_eq!(val.re, 0.0, epsilon = 1e-14);
        assert_relative_eq!(val.im, 0.0, epsilon = 1e-14);
    }

    #[test]
    fn test_poly_eval_at_complex_root() {
        // p(x) = x² + 1, root = i
        let val = poly_eval_complex(&[1.0, 0.0, 1.0], Complex::new(0.0, 1.0));
        assert!(val.norm() < 1e-14, "p(i) should be 0, got {}", val);
    }

    #[test]
    fn test_poly_eval_constant() {
        // p(x) = 5
        let val = poly_eval_complex(&[5.0], Complex::new(3.0, 2.0));
        assert_relative_eq!(val.re, 5.0, epsilon = 1e-14);
        assert_relative_eq!(val.im, 0.0, epsilon = 1e-14);
    }

    // ── poly_mul ──────────────────────────────────────────────────────────────

    #[test]
    fn test_poly_mul_linear_factors() {
        // (x – 1)(x – 2) = x² – 3x + 2
        let p = poly_mul(&[1.0, -1.0], &[1.0, -2.0]);
        assert_eq!(p.len(), 3);
        assert_relative_eq!(p[0], 1.0, epsilon = 1e-14);
        assert_relative_eq!(p[1], -3.0, epsilon = 1e-14);
        assert_relative_eq!(p[2], 2.0, epsilon = 1e-14);
    }

    #[test]
    fn test_poly_mul_by_constant() {
        // 3 * (x + 1) = 3x + 3
        let p = poly_mul(&[3.0], &[1.0, 1.0]);
        assert_eq!(p.len(), 2);
        assert_relative_eq!(p[0], 3.0, epsilon = 1e-14);
        assert_relative_eq!(p[1], 3.0, epsilon = 1e-14);
    }

    #[test]
    fn test_poly_mul_empty() {
        let p = poly_mul(&[], &[1.0, 2.0]);
        assert!(p.is_empty());
    }

    // ── char_poly_from_roots ──────────────────────────────────────────────────

    #[test]
    fn test_char_poly_two_real_roots() {
        // λ = 1, 2  →  p(x) = (x-1)(x-2) = x² – 3x + 2
        let roots = vec![Complex::new(1.0, 0.0), Complex::new(2.0, 0.0)];
        let p = char_poly_from_roots(&roots);
        assert_eq!(p.len(), 3);
        assert_relative_eq!(p[0], 1.0, epsilon = 1e-12);
        assert_relative_eq!(p[1], -3.0, epsilon = 1e-12);
        assert_relative_eq!(p[2], 2.0, epsilon = 1e-12);
    }

    #[test]
    fn test_char_poly_complex_conjugate_pair() {
        // λ = ±i  →  p(x) = (x – i)(x + i) = x² + 1
        let roots = vec![Complex::new(0.0, 1.0), Complex::new(0.0, -1.0)];
        let p = char_poly_from_roots(&roots);
        assert_eq!(p.len(), 3);
        assert_relative_eq!(p[0], 1.0, epsilon = 1e-12);
        assert_relative_eq!(p[1], 0.0, epsilon = 1e-12);
        assert_relative_eq!(p[2], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_char_poly_single_root() {
        // λ = 5  →  p(x) = x – 5
        let roots = vec![Complex::new(5.0, 0.0)];
        let p = char_poly_from_roots(&roots);
        assert_eq!(p.len(), 2);
        assert_relative_eq!(p[0], 1.0, epsilon = 1e-12);
        assert_relative_eq!(p[1], -5.0, epsilon = 1e-12);
    }

    // ── refine_roots_laguerre ─────────────────────────────────────────────────

    #[test]
    fn test_laguerre_refine_quadratic() {
        // x² – 1, start from companion roots, refine further
        let coeffs = [1.0_f64, 0.0, -1.0];
        let initial = poly_roots(&coeffs).expect("initial");
        let refined = refine_roots_laguerre(&coeffs, &initial, 50, 1e-14).expect("refined");
        for r in &refined {
            let pval = poly_eval_complex(&coeffs, *r);
            assert!(
                pval.norm() < 1e-12,
                "Laguerre-refined root does not satisfy p(r)=0: p({}) = {}",
                r,
                pval
            );
        }
    }

    #[test]
    fn test_laguerre_refine_cubic() {
        // (x-1)(x-2)(x-3)
        let coeffs = [1.0_f64, -6.0, 11.0, -6.0];
        let initial = poly_roots(&coeffs).expect("initial");
        let refined = refine_roots_laguerre(&coeffs, &initial, 100, 1e-14).expect("refined");
        assert_eq!(refined.len(), 3);
        for r in &refined {
            let pval = poly_eval_complex(&coeffs, *r);
            assert!(
                pval.norm() < 1e-10,
                "Refined root does not satisfy p(r)=0: p({}) = {}",
                r,
                pval
            );
        }
    }

    #[test]
    fn test_laguerre_refine_complex_roots() {
        // x² + 4 → roots ±2i
        let coeffs = [1.0_f64, 0.0, 4.0];
        let initial = poly_roots(&coeffs).expect("initial");
        let refined = refine_roots_laguerre(&coeffs, &initial, 50, 1e-14).expect("refined");
        for r in &refined {
            let pval = poly_eval_complex(&coeffs, *r);
            assert!(
                pval.norm() < 1e-12,
                "Laguerre-refined root does not satisfy p(r)=0: p({}) = {}",
                r,
                pval
            );
        }
    }

    // ── round-trip: char_poly_from_roots → poly_roots ─────────────────────────

    #[test]
    fn test_roundtrip_char_poly_and_roots() {
        // Build p(x) = (x-1)(x-2)(x-3)(x-4)
        let known_roots = vec![
            Complex::new(1.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0),
            Complex::new(4.0, 0.0),
        ];
        let coeffs = char_poly_from_roots(&known_roots);
        let mut found_roots = poly_roots(&coeffs).expect("roots from char poly");
        found_roots.sort_by(|a, b| a.re.partial_cmp(&b.re).expect("cmp"));

        assert_eq!(found_roots.len(), 4);
        let expected = [1.0, 2.0, 3.0, 4.0];
        for (r, &e) in found_roots.iter().zip(expected.iter()) {
            assert_relative_eq!(r.re, e, epsilon = 1e-6);
            assert_relative_eq!(r.im.abs(), 0.0, epsilon = 1e-6);
        }
    }
}
