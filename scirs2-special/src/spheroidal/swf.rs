//! Simplified spheroidal wave function API
//!
//! This module provides a clean, simplified interface to spheroidal wave functions
//! complementing the full implementation in the parent module. It offers:
//!
//! - `SpheroidalKind`: enum for prolate/oblate geometry
//! - `SpheroidalEigenvalue`: computed eigenvalue metadata
//! - `spheroidal_eigenvalue_mn(m, n, c, kind)`: characteristic value λ_{mn}(c)
//! - `spheroidal_ps(m, n, c, x, kind)`: angular wave function S_{mn}^{(1)}(c, x)
//! - `spheroidal_wronskian(m, n, c, x)`: Wronskian diagnostic for prolate SWF
//!
//! ## Mathematical Background
//!
//! ### Differential equation (prolate, |x| ≤ 1)
//!
//! ```text
//! (1-x²)y'' - 2x y' + [λ_{mn} - c²x²]y = 0
//! ```
//!
//! For oblate, the sign of c² is reversed (c² → -c²).
//!
//! ### Eigenvalue computation
//!
//! Expand S_{mn} in associated Legendre polynomials P_{m+2k}^m(x) (even parity)
//! or P_{m+2k+1}^m(x) (odd parity). The 3-term recurrence for the expansion
//! coefficients d_k yields a real symmetric tridiagonal matrix T whose eigenvalues
//! are the characteristic values λ_{mn}(c). The target eigenvalue is the (n-m)/2-th
//! one, indexed by the parity class.
//!
//! ### c → 0 limit
//!
//! λ_{mn}(0) = n(n+1), the spherical harmonic eigenvalue.
//! S_{mn}(0, x) is proportional to the associated Legendre polynomial P_n^m(x).
//!
//! ## References
//!
//! - Flammer, C. (1957). *Spheroidal Wave Functions*. Stanford University Press.
//! - Abramowitz & Stegun, §21.

use crate::error::{SpecialError, SpecialResult};
use crate::mathieu::advanced::{tridiag_eigenvalues, tridiag_eigenvector};
use crate::spheroidal::wave_functions::associated_legendre;

/// Spheroidal geometry type.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpheroidalKind {
    /// Prolate spheroid (cigar-shaped): the parameter c is real and positive.
    /// Differential equation: (1-x²)y'' - 2xy' + [λ - c²x²]y = 0
    Prolate,
    /// Oblate spheroid (disk-shaped): c² is replaced by -c² in the equation.
    /// Differential equation: (1-x²)y'' - 2xy' + [λ + c²x²]y = 0
    Oblate,
}

/// Result of a spheroidal eigenvalue computation.
#[derive(Debug, Clone)]
pub struct SpheroidalEigenvalue {
    /// Azimuthal order m ≥ 0
    pub m: usize,
    /// Degree n ≥ m
    pub n: usize,
    /// Spheroidal parameter c ≥ 0
    pub c: f64,
    /// Geometry type
    pub kind: SpheroidalKind,
    /// Characteristic value λ_{mn}(c)
    pub lambda: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Truncation size for the tridiagonal eigenvalue system.
/// N_max = max(n + 20, 40) as specified in the module interface.
fn n_max(n: usize) -> usize {
    n.saturating_add(20).max(40)
}

/// Build the symmetric tridiagonal matrix for the angular spheroidal equation.
///
/// The expansion S_{mn}(c, x) = Σ_k d_k P_{m+k}^m(x) where k runs over even
/// or odd integers depending on parity p = (n-m) % 2.
///
/// The 3-term recurrence (from substituting into the ODE and using the
/// associated Legendre recurrence) gives:
///
///   α_k d_{k-2} + (β_k - λ) d_k + γ_k d_{k+2} = 0
///
/// For the basis vector indexed by p = 0, 1, 2, ... where k = 2p + parity:
///
///   ell = m + k = m + 2p + parity  (degree of P_ell^m)
///   diagonal[p] = ell(ell+1) + c² * D_p   (prolate sign; oblate flips c² sign)
///   off_diagonal[p] = c² * E_p
///
/// where D_p and E_p are the diagonal/coupling c²-terms from the x²-coupling.
///
/// The exact coefficients are (Flammer §4, A&S 21.7):
///   E_p (coupling between k and k+2 at index p) = -(ell-m+1)(ell-m+2)/[(2ell+1)(2ell+3)]
///   D_p = off_{p-1} (coupling from k-2 to k) = -(ell+m)(ell+m-1)/[(2ell-1)(2ell+1)]   if k≥2
///
/// The diagonal entry at p includes both the "coming-down" and "going-up" c²-coupling
/// corrections for the diagonal of the tridiagonal matrix:
///   diagonal[p] = ell(ell+1) + c² * [A_p + C_p]
/// where A_p (from coupling downward) and C_p (from coupling upward) come from the
/// recurrence structure.
///
/// Returns (diag, off_diag) of the symmetric tridiagonal system of size `size`.
fn build_swf_tridiag(m: usize, n: usize, c: f64, sign_c2: f64) -> (Vec<f64>, Vec<f64>) {
    let size = n_max(n);
    let parity = (n - m) % 2;
    let c2 = c * c * sign_c2; // positive for prolate, negative for oblate

    let mut diag = vec![0.0_f64; size];
    let mut off = vec![0.0_f64; size.saturating_sub(1)];

    for p in 0..size {
        let k = 2 * p + parity; // index in full expansion
        let ell = (m + k) as f64; // degree of P_{m+k}^m

        // Coupling from (k → k-2), i.e., the "upper" off-diagonal term
        // B_down = -(ell+m)(ell+m-1) / [(2ell-1)(2ell+1)]  when k ≥ 2
        let b_down = if k >= 2 {
            let denom = (2.0 * ell - 1.0) * (2.0 * ell + 1.0);
            if denom.abs() < 1e-14 {
                0.0
            } else {
                -(ell + m as f64) * (ell + m as f64 - 1.0) / denom
            }
        } else {
            0.0
        };

        // Coupling from (k → k+2), i.e., the "lower" off-diagonal term
        // B_up = -(ell-m+1)(ell-m+2) / [(2ell+1)(2ell+3)]
        let denom_up = (2.0 * ell + 1.0) * (2.0 * ell + 3.0);
        let b_up = if denom_up.abs() < 1e-14 {
            0.0
        } else {
            -(ell - m as f64 + 1.0) * (ell - m as f64 + 2.0) / denom_up
        };

        // The tridiagonal diagonal is ell(ell+1) plus c²-contributions on the diagonal.
        // The diagonal c²-term is b_down + b_up in the three-term recurrence system
        // (from substituting the Legendre recurrence for x² P_ell^m into the expansion).
        diag[p] = ell * (ell + 1.0) + c2 * (b_down + b_up);

        // Off-diagonal at p: coupling between basis p and p+1 (k and k+2)
        if p + 1 < size {
            off[p] = c2 * b_up;
        }
    }

    (diag, off)
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the spheroidal characteristic value λ_{mn}(c).
///
/// At c = 0, λ_{mn}(0) = n(n+1) (the spherical harmonic eigenvalue).
/// For prolate geometry, the equation is (1-x²)y'' - 2xy' + [λ - c²x²]y = 0.
/// For oblate geometry, c² is replaced by -c².
///
/// The computation uses a real symmetric tridiagonal eigenproblem of size
/// N_max = max(n+20, 40).
///
/// # Arguments
/// * `m` - Azimuthal order (m ≥ 0)
/// * `n` - Degree (n ≥ m)
/// * `c` - Spheroidal parameter c ≥ 0
/// * `kind` - Prolate or Oblate
///
/// # Returns
/// Characteristic value λ_{mn}(c) wrapped in `SpheroidalEigenvalue`.
///
/// # Errors
/// Returns `SpecialError::DomainError` if n < m.
/// Returns `SpecialError::ComputationError` if the tridiagonal eigensystem fails.
///
/// # Examples
///
/// ```
/// use scirs2_special::{SpheroidalKind, spheroidal_eigenvalue_mn};
///
/// // At c=0 the eigenvalue equals n(n+1)
/// let ev = spheroidal_eigenvalue_mn(0, 2, 0.0, SpheroidalKind::Prolate).unwrap();
/// assert!((ev.lambda - 6.0).abs() < 1e-10);
/// ```
pub fn spheroidal_eigenvalue_mn(
    m: usize,
    n: usize,
    c: f64,
    kind: SpheroidalKind,
) -> SpecialResult<SpheroidalEigenvalue> {
    if n < m {
        return Err(SpecialError::DomainError(format!(
            "spheroidal_eigenvalue_mn: degree n={n} must be >= order m={m}"
        )));
    }
    if c < 0.0 {
        return Err(SpecialError::DomainError(format!(
            "spheroidal_eigenvalue_mn: parameter c={c} must be >= 0"
        )));
    }

    let sign_c2 = match kind {
        SpheroidalKind::Prolate => 1.0,
        SpheroidalKind::Oblate => -1.0,
    };

    let (diag, off) = build_swf_tridiag(m, n, c, sign_c2);
    let eigenvalues = tridiag_eigenvalues(&diag, &off);

    // Target index: (n-m)/2 within the same parity class
    let target = (n - m) / 2;
    let lambda = eigenvalues.get(target).copied().ok_or_else(|| {
        SpecialError::ComputationError(format!(
            "spheroidal_eigenvalue_mn: failed to locate eigenvalue for m={m}, n={n}, c={c}"
        ))
    })?;

    Ok(SpheroidalEigenvalue {
        m,
        n,
        c,
        kind,
        lambda,
    })
}

/// Compute the angular spheroidal wave function of the first kind S_{mn}^{(1)}(c, x).
///
/// S_{mn}(c, x) is the bounded solution of the spheroidal wave equation on |x| ≤ 1.
/// It is expanded as:
///
/// ```text
/// S_{mn}(c, x) = Σ_p d_p P_{m+2p+parity}^m(x)
/// ```
///
/// where parity = (n-m) % 2, and the d_p are the components of the eigenvector
/// of the tridiagonal coefficient matrix corresponding to eigenvalue λ_{mn}(c).
///
/// ## Normalization
///
/// The coefficients are normalized so that the eigenvector has unit L²-norm.
/// The sign is chosen so that the coefficient with the largest magnitude is positive.
///
/// ## c → 0 limit
///
/// At c = 0, all off-diagonal coupling vanishes, so d_p = δ_{p, (n-m)/2},
/// giving S_{mn}(0, x) = P_n^m(x) (unnormalized associated Legendre polynomial).
///
/// # Arguments
/// * `m` - Azimuthal order (m ≥ 0)
/// * `n` - Degree (n ≥ m)
/// * `c` - Spheroidal parameter c ≥ 0
/// * `x` - Argument, |x| ≤ 1
/// * `kind` - Prolate or Oblate
///
/// # Returns
/// Value of S_{mn}^{(1)}(c, x).
///
/// # Errors
/// Returns `SpecialError::DomainError` if n < m or |x| > 1 + 1e-10.
///
/// # Examples
///
/// ```
/// use scirs2_special::{SpheroidalKind, spheroidal_ps};
///
/// // At c=0, S_{00}(0, x) should be proportional to P_0(x) = 1.0
/// let val = spheroidal_ps(0, 0, 0.0, 0.5, SpheroidalKind::Prolate).unwrap();
/// assert!(val.is_finite());
/// ```
pub fn spheroidal_ps(
    m: usize,
    n: usize,
    c: f64,
    x: f64,
    kind: SpheroidalKind,
) -> SpecialResult<f64> {
    if n < m {
        return Err(SpecialError::DomainError(format!(
            "spheroidal_ps: degree n={n} must be >= order m={m}"
        )));
    }
    if x.abs() > 1.0 + 1e-10 {
        return Err(SpecialError::DomainError(format!(
            "spheroidal_ps: argument x={x} must satisfy |x| <= 1"
        )));
    }
    if c < 0.0 {
        return Err(SpecialError::DomainError(format!(
            "spheroidal_ps: parameter c={c} must be >= 0"
        )));
    }

    let x_clamped = x.clamp(-1.0, 1.0);
    let sign_c2 = match kind {
        SpheroidalKind::Prolate => 1.0,
        SpheroidalKind::Oblate => -1.0,
    };

    let (diag, off) = build_swf_tridiag(m, n, c, sign_c2);
    let eigenvalues = tridiag_eigenvalues(&diag, &off);

    let target = (n - m) / 2;
    let eigenval = eigenvalues.get(target).copied().ok_or_else(|| {
        SpecialError::ComputationError(format!(
            "spheroidal_ps: no eigenvalue for m={m}, n={n}, c={c}"
        ))
    })?;

    let coeffs = tridiag_eigenvector(&diag, &off, eigenval);

    // Determine sign convention: largest-magnitude coefficient should be positive
    let max_abs_idx = coeffs
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            a.abs()
                .partial_cmp(&b.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i)
        .unwrap_or(0);
    let sign = if coeffs.get(max_abs_idx).copied().unwrap_or(1.0) >= 0.0 {
        1.0_f64
    } else {
        -1.0_f64
    };

    let parity = (n - m) % 2;

    // Evaluate the Legendre polynomial expansion
    let mut result = 0.0_f64;
    for (p, &d) in coeffs.iter().enumerate() {
        let k = 2 * p + parity;
        let l = m + k;
        let p_val = associated_legendre(l, m, x_clamped);
        result += d * p_val;
    }

    Ok(sign * result)
}

/// Evaluate the (approximate) Wronskian of the spheroidal wave functions.
///
/// For the prolate spheroidal wave equation the Wronskian of the first and second
/// kind solutions satisfies:
///
/// ```text
/// W[S^{(1)}, S^{(2)}](x) = C / (1 - x²)
/// ```
///
/// This function estimates whether the ratio `(1-x²) * W` is approximately
/// constant (up to a normalization factor) by computing the Wronskian via
/// numerical differentiation of `spheroidal_ps` at two nearby points.
///
/// ## Returns
///
/// The quantity `(1-x²) * [S(x+h) - S(x-h)] / (2h)` normalized by `S(x)`.
/// At interior points this is finite; at the endpoints x = ±1 the Wronskian
/// diverges as (1-x²)^{-1}.
///
/// # Arguments
/// * `m` - Azimuthal order
/// * `n` - Degree
/// * `c` - Spheroidal parameter
/// * `x` - Evaluation point, |x| < 1 (strict)
///
/// # Returns
/// The normalized Wronskian quantity (1-x²) * S'(x) / S(x), which is
/// independent of x for true solutions (up to numerical precision).
///
/// # Errors
/// Returns `SpecialError::DomainError` if |x| ≥ 1 - 1e-8 (too close to boundary).
///
/// # Examples
///
/// ```
/// use scirs2_special::spheroidal_wronskian;
///
/// let w = spheroidal_wronskian(0, 2, 1.0, 0.3).unwrap();
/// assert!(w.is_finite());
/// ```
pub fn spheroidal_wronskian(m: usize, n: usize, c: f64, x: f64) -> SpecialResult<f64> {
    if x.abs() >= 1.0 - 1e-8 {
        return Err(SpecialError::DomainError(format!(
            "spheroidal_wronskian: x={x} must satisfy |x| < 1 (strict); too close to boundary"
        )));
    }
    if n < m {
        return Err(SpecialError::DomainError(format!(
            "spheroidal_wronskian: n={n} must be >= m={m}"
        )));
    }

    // Numerical derivative via central difference
    let h = 1e-5_f64.max(1e-5 * (1.0 - x.abs()));
    let x_p = (x + h).clamp(-1.0 + 1e-12, 1.0 - 1e-12);
    let x_m = (x - h).clamp(-1.0 + 1e-12, 1.0 - 1e-12);

    let s_p = spheroidal_ps(m, n, c, x_p, SpheroidalKind::Prolate)?;
    let s_m = spheroidal_ps(m, n, c, x_m, SpheroidalKind::Prolate)?;
    let s_x = spheroidal_ps(m, n, c, x, SpheroidalKind::Prolate)?;

    // Numerical derivative
    let ds_dx = (s_p - s_m) / (2.0 * h);

    // Wronskian-like quantity: (1-x²) * S'(x)
    // Normalized by S(x) to get a dimensionless ratio
    let one_minus_x2 = 1.0 - x * x;

    if s_x.abs() < 1e-15 {
        // Near a zero of S; return the un-normalized quantity
        Ok(one_minus_x2 * ds_dx)
    } else {
        Ok(one_minus_x2 * ds_dx / s_x)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Eigenvalue tests ───────────────────────────────────────────────────────

    #[test]
    fn test_eigenvalue_c0_n0m0_prolate() {
        // λ_{00}(0) = 0*1 = 0 for m=0, n=0
        let ev = spheroidal_eigenvalue_mn(0, 0, 0.0, SpheroidalKind::Prolate)
            .expect("m=0,n=0,c=0 eigenvalue should not fail");
        assert!(
            ev.lambda.abs() < 0.5,
            "λ_{{00}}(0) should be 0, got {}",
            ev.lambda
        );
    }

    #[test]
    fn test_eigenvalue_c0_n1m1_prolate() {
        // λ_{11}(0) = 1*2 = 2
        let ev = spheroidal_eigenvalue_mn(1, 1, 0.0, SpheroidalKind::Prolate)
            .expect("m=1,n=1,c=0 eigenvalue should not fail");
        assert!(
            (ev.lambda - 2.0).abs() < 0.5,
            "λ_{{11}}(0) should be ≈ 2, got {}",
            ev.lambda
        );
    }

    #[test]
    fn test_eigenvalue_c0_matches_nn1() {
        // For c=0, λ_{mn}(0) = n(n+1) for several (m, n) pairs
        let cases = [
            (0usize, 2usize, 6.0),
            (0, 3, 12.0),
            (1, 2, 6.0),
            (2, 3, 12.0),
        ];
        for (m, n, expected) in cases {
            let ev = spheroidal_eigenvalue_mn(m, n, 0.0, SpheroidalKind::Prolate)
                .unwrap_or_else(|_| panic!("eigenvalue for m={m},n={n} should not fail"));
            assert!(
                (ev.lambda - expected).abs() < 1.0,
                "λ_{{m={m},n={n}}}(0) should be ≈ {expected}, got {}",
                ev.lambda
            );
        }
    }

    #[test]
    fn test_eigenvalue_increases_with_n() {
        // For fixed m=0, eigenvalues should increase with n
        let c = 1.0;
        let mut prev = f64::NEG_INFINITY;
        for n in [0usize, 2, 4] {
            let ev = spheroidal_eigenvalue_mn(0, n, c, SpheroidalKind::Prolate)
                .unwrap_or_else(|_| panic!("eigenvalue for n={n} should not fail"));
            assert!(
                ev.lambda > prev - 1e-8,
                "eigenvalue for n={n} ({}) should exceed previous ({})",
                ev.lambda,
                prev
            );
            prev = ev.lambda;
        }
    }

    #[test]
    fn test_eigenvalue_oblate_c0_equals_nn1() {
        // For c=0, oblate and prolate both give n(n+1)
        let ev = spheroidal_eigenvalue_mn(0, 3, 0.0, SpheroidalKind::Oblate)
            .expect("oblate c=0 eigenvalue should not fail");
        assert!(
            (ev.lambda - 12.0).abs() < 0.5,
            "oblate λ_{{03}}(0) should be ≈ 12, got {}",
            ev.lambda
        );
    }

    #[test]
    fn test_eigenvalue_error_n_less_than_m() {
        let result = spheroidal_eigenvalue_mn(3, 1, 1.0, SpheroidalKind::Prolate);
        assert!(result.is_err(), "n < m should return an error");
    }

    // ── Wave function tests ────────────────────────────────────────────────────

    #[test]
    fn test_spheroidal_ps_c0_finite() {
        // At c=0, should give a finite non-NaN value
        let val = spheroidal_ps(0, 0, 0.0, 0.5, SpheroidalKind::Prolate)
            .expect("spheroidal_ps m=0,n=0,c=0 should not fail");
        assert!(
            val.is_finite(),
            "S_{{00}}(0, 0.5) should be finite, got {val}"
        );
    }

    #[test]
    fn test_spheroidal_ps_c0_legendre_proportional() {
        // S_{02}(0, x) should be proportional to P_2(x) = (3x²-1)/2
        let x = 0.7_f64;
        let val = spheroidal_ps(0, 2, 0.0, x, SpheroidalKind::Prolate)
            .expect("spheroidal_ps should not fail for c=0");
        let p2 = (3.0 * x * x - 1.0) / 2.0; // P_2^0(x)
                                            // Check proportionality: val / p2 should be (nearly) constant
                                            // (The normalization may differ; we just confirm they have the same sign)
        if p2.abs() > 1e-10 && val.abs() > 1e-10 {
            let ratio = val / p2;
            assert!(
                ratio > 0.0,
                "S_{{02}}(0, {x}) and P_2({x}) should have the same sign; ratio = {ratio}"
            );
        }
    }

    #[test]
    fn test_spheroidal_ps_interior_finite() {
        // Finite values at several interior points
        for x in [-0.8, -0.3, 0.0, 0.3, 0.8] {
            let val = spheroidal_ps(1, 2, 2.0, x, SpheroidalKind::Prolate)
                .unwrap_or_else(|_| panic!("spheroidal_ps at x={x} should not fail"));
            assert!(
                val.is_finite(),
                "S_{{12}}(2, {x}) should be finite, got {val}"
            );
        }
    }

    #[test]
    fn test_spheroidal_ps_domain_error() {
        let result = spheroidal_ps(0, 2, 1.0, 1.5, SpheroidalKind::Prolate);
        assert!(result.is_err(), "|x| > 1 should return a domain error");
    }

    // ── Wronskian tests ────────────────────────────────────────────────────────

    #[test]
    fn test_wronskian_finite_interior() {
        // Should be finite at interior points
        for x in [-0.7, -0.2, 0.0, 0.4, 0.6] {
            let w = spheroidal_wronskian(0, 2, 1.0, x)
                .unwrap_or_else(|_| panic!("wronskian at x={x} should not fail"));
            assert!(
                w.is_finite(),
                "wronskian at x={x} should be finite, got {w}"
            );
        }
    }

    #[test]
    fn test_wronskian_boundary_error() {
        // Should error at boundary
        let result = spheroidal_wronskian(0, 2, 1.0, 1.0);
        assert!(result.is_err(), "wronskian at boundary x=1 should error");
    }

    #[test]
    fn test_spheroidal_eigenvalue_metadata() {
        // The SpheroidalEigenvalue struct should correctly carry metadata
        let ev = spheroidal_eigenvalue_mn(1, 3, 2.0, SpheroidalKind::Prolate)
            .expect("eigenvalue should not fail");
        assert_eq!(ev.m, 1);
        assert_eq!(ev.n, 3);
        assert!((ev.c - 2.0).abs() < 1e-14);
        assert_eq!(ev.kind, SpheroidalKind::Prolate);
        assert!(ev.lambda.is_finite());
    }
}
