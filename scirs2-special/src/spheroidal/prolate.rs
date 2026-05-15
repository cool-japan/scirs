//! Prolate spheroidal wave functions
//!
//! This module provides implementations of prolate spheroidal functions, which arise in
//! the solution of the Helmholtz equation in prolate spheroidal coordinates. These
//! functions are particularly important in electromagnetic scattering by prolate
//! spheroids (cigar-shaped objects).
//!
//! ## Algorithm
//!
//! Wave 74 refactor (2026-05-08): all six routines below dispatch to the
//! Flammer / Bouwkamp d-coefficient pipeline implemented in
//! [`crate::spheroidal::cf_helpers`]. The legacy "continued-fraction
//! perturbation" path with hand-rolled Sturm-sequence Newton iteration has
//! been removed — it converged catastrophically wrong values for `|c| > 1`,
//! e.g. `pro_cv(0, 2, 2.0) = -0.018`, while the correct value is `+8.2257`
//! (SciPy `pro_cv(0, 2, 2.0)`).
//!
//! ## SciPy convention
//!
//! These routines match `scipy.special.{pro_cv, pro_cv_seq, pro_ang1,
//! pro_rad1, pro_rad2}` to ~1e-11 for the angular family and to ~1e-5 / m=0
//! for the radial family.
//!
//! ## Limitations
//!
//! The radial-2 (`pro_rad2`) implementation uses the simple spherical Bessel
//! `y_l` series (Flammer §4.4.4). This series diverges asymptotically for
//! `(m, n)` cases with odd `m` and odd parity (`(n − m) mod 2 = 1`), notably
//! `(m=1, n=2)`. Such cases require the alternative Wronskian-based
//! representation (Flammer §4.5), which is not yet implemented.
//! `pro_rad2` for those cases returns a value but with reduced accuracy or
//! incorrect sign.

use crate::error::{SpecialError, SpecialResult};
use crate::spheroidal::cf_helpers::{
    angular_function, flammer_eigenvalue, radial_function, SphericalBesselKind, SpheroidalParity,
    DEFAULT_D_LEN,
};

/// Computes the characteristic value `λ_{m,n}(c)` for prolate spheroidal wave functions.
///
/// `λ_{m,n}(c)` is the eigenvalue of the prolate spheroidal angular ODE
///
/// ```text
///   (1 − η²) y'' − 2η y' + [λ − c² η²] y = 0
/// ```
///
/// such that `y(η)` is regular at `η = ±1`. At `c = 0` the eigenvalue equals
/// `n(n + 1)`.
///
/// # Arguments
///
/// * `m` - Azimuthal order (≥ 0)
/// * `n` - Total degree (≥ m)
/// * `c` - Spheroidal parameter `c = ka` (real)
///
/// # Examples
///
/// ```
/// use scirs2_special::pro_cv;
///
/// // c = 0 reduces to n(n+1)
/// let lambda_00 = pro_cv(0, 0, 0.0).unwrap();
/// assert_eq!(lambda_00, 0.0);
///
/// let lambda_01 = pro_cv(0, 1, 0.0).unwrap();
/// assert_eq!(lambda_01, 2.0);
///
/// // Small-c perturbation
/// let lambda_small = pro_cv(0, 1, 0.1).unwrap();
/// assert!((lambda_small - 2.0).abs() < 0.01);
///
/// // Larger c (matches SciPy reference)
/// let lambda_mod = pro_cv(0, 2, 2.0).unwrap();
/// assert!((lambda_mod - 8.2257130011).abs() < 1e-6);
/// ```
#[allow(dead_code)]
pub fn pro_cv(m: i32, n: i32, c: f64) -> SpecialResult<f64> {
    if m < 0 || n < m {
        return Err(SpecialError::DomainError(
            "Parameters must satisfy m ≥ 0 and n ≥ m".to_string(),
        ));
    }
    if n > 1000 {
        return Err(SpecialError::DomainError(
            "Parameter n is too large (> 1000), may cause numerical instability".to_string(),
        ));
    }
    if c.abs() > 1000.0 {
        return Err(SpecialError::DomainError(
            "Parameter c is too large (|c| > 1000), may cause numerical instability".to_string(),
        ));
    }
    if c.is_nan() {
        return Ok(f64::NAN);
    }
    if c.is_infinite() {
        // Asymptotic limit: λ ≈ -c²/4 + (2n+1)|c|
        let n_f = n as f64;
        return Ok(-c.powi(2) / 4.0 + (2.0 * n_f + 1.0) * c.abs());
    }
    if c == 0.0 {
        return Ok(n as f64 * (n as f64 + 1.0));
    }
    flammer_eigenvalue(SpheroidalParity::Prolate, m, n, c.abs(), DEFAULT_D_LEN)
}

/// Computes a sequence of characteristic values for prolate spheroidal wave functions.
///
/// Returns `[λ_{m,m}(c), λ_{m,m+1}(c), …, λ_{m,n}(c)]`.
///
/// # Examples
///
/// ```
/// use scirs2_special::pro_cv_seq;
///
/// let values = pro_cv_seq(0, 3, 0.0).unwrap();
/// assert_eq!(values.len(), 4);
/// assert_eq!(values[0], 0.0);
/// assert_eq!(values[1], 2.0);
/// ```
#[allow(dead_code)]
pub fn pro_cv_seq(m: i32, n: i32, c: f64) -> SpecialResult<Vec<f64>> {
    if m < 0 || n < m {
        return Err(SpecialError::DomainError(
            "Parameters must satisfy m ≥ 0 and n ≥ m".to_string(),
        ));
    }
    if n - m > 199 {
        return Err(SpecialError::DomainError(
            "Difference between n and m is too large (max 199)".to_string(),
        ));
    }
    if c.is_nan() {
        return Ok(vec![f64::NAN; (n - m + 1) as usize]);
    }
    let mut result = Vec::with_capacity((n - m + 1) as usize);
    for degree in m..=n {
        result.push(pro_cv(m, degree, c)?);
    }
    Ok(result)
}

/// Computes the prolate spheroidal angular function of the first kind, `S_{m,n}(c, η)`.
///
/// The convention matches SciPy's `pro_ang1`: at `c = 0`, the function reduces to
/// the un-Condon–Shortley associated Legendre function, and the Meixner–Schäfke
/// normalisation enforces `S_{m,n}(c, 0) = P_n^m(0)` (even parity) or
/// `S'_{m,n}(c, 0) = (P_n^m)'(0)` (odd parity).
///
/// # Arguments
///
/// * `m` - Azimuthal order (≥ 0)
/// * `n` - Total degree (≥ m)
/// * `c` - Spheroidal parameter (real)
/// * `x` - Evaluation point `η ∈ [-1, 1]`
///
/// # Examples
///
/// ```
/// use scirs2_special::pro_ang1;
///
/// // c = 0 reduces to associated Legendre
/// let (p_mn, _) = pro_ang1(0, 1, 0.0, 0.5).unwrap();
/// assert!((p_mn - 0.5).abs() < 1e-12);
///
/// // c = 5 case (matches SciPy)
/// let (s, _) = pro_ang1(0, 2, 5.0, 0.5).unwrap();
/// assert!((s - 0.3843865522).abs() < 1e-6);
/// ```
#[allow(dead_code)]
pub fn pro_ang1(m: i32, n: i32, c: f64, x: f64) -> SpecialResult<(f64, f64)> {
    if m < 0 || n < m {
        return Err(SpecialError::DomainError(
            "Parameters must satisfy m ≥ 0 and n ≥ m".to_string(),
        ));
    }
    if !(-1.0..=1.0).contains(&x) {
        return Err(SpecialError::DomainError(
            "Angular coordinate x must be in range [-1, 1]".to_string(),
        ));
    }
    if c.is_nan() || x.is_nan() {
        return Ok((f64::NAN, f64::NAN));
    }
    angular_function(SpheroidalParity::Prolate, m, n, c.abs(), x)
}

/// Computes the prolate spheroidal radial function of the first kind, `R_{m,n}^{(1)}(c, ξ)`.
///
/// Uses the Flammer §4.4 expansion in spherical Bessel functions of the first
/// kind `j_l(c·ξ)`, normalised so that `R_{m,n}^{(1)}(c, ξ) → j_n(c·ξ)` as
/// `c → 0` for `m = 0`.
///
/// # Arguments
///
/// * `m` - Azimuthal order (≥ 0)
/// * `n` - Total degree (≥ m)
/// * `c` - Spheroidal parameter (real)
/// * `x` - Radial coordinate `ξ ≥ 1`
///
/// # Examples
///
/// ```
/// use scirs2_special::pro_rad1;
///
/// // c = 0 reduces to associated Legendre (function value at ξ = 1.5 = 1.5 for n=1, m=0)
/// // Note: the radial function at c = 0 is j_n(c·ξ) which is 0 at c = 0.
/// // For non-zero c:
/// let (r_val, r_prime) = pro_rad1(0, 1, 1.0, 1.5).unwrap();
/// assert!(r_val.is_finite());
/// assert!((r_val - 0.4138205450).abs() < 1e-6); // SciPy reference
/// assert!(r_prime.is_finite());
/// ```
#[allow(dead_code)]
pub fn pro_rad1(m: i32, n: i32, c: f64, x: f64) -> SpecialResult<(f64, f64)> {
    if m < 0 || n < m {
        return Err(SpecialError::DomainError(
            "Parameters must satisfy m ≥ 0 and n ≥ m".to_string(),
        ));
    }
    if x < 1.0 {
        return Err(SpecialError::DomainError(
            "Radial coordinate x must be ≥ 1.0".to_string(),
        ));
    }
    if c.is_nan() || x.is_nan() {
        return Ok((f64::NAN, f64::NAN));
    }
    if c == 0.0 {
        // R_{mn}^{(1)}(0, ξ) is proportional to associated Legendre P_n^m(ξ)
        // along the radial direction. SciPy returns P_n^m(ξ) for c = 0.
        let p_mn = crate::orthogonal::legendre_assoc(n as usize, m, x);
        let h = 1.0e-8_f64;
        let p_plus = crate::orthogonal::legendre_assoc(n as usize, m, x + h);
        let p_minus = crate::orthogonal::legendre_assoc(n as usize, m, x - h);
        let p_prime = (p_plus - p_minus) / (2.0 * h);
        return Ok((p_mn, p_prime));
    }
    radial_function(
        SpheroidalParity::Prolate,
        SphericalBesselKind::First,
        m,
        n,
        c.abs(),
        x,
    )
}

/// Computes the prolate spheroidal radial function of the second kind, `R_{m,n}^{(2)}(c, ξ)`.
///
/// Uses the Flammer §4.4 expansion in spherical Bessel functions of the second
/// kind `y_l(c·ξ)`. **Limitations**: this series is numerically unstable for some
/// odd-`m`/odd-parity combinations (notably `(m=1, n=2)`); see module-level
/// documentation.
///
/// # Arguments
///
/// * `m` - Azimuthal order (≥ 0)
/// * `n` - Total degree (≥ m)
/// * `c` - Spheroidal parameter (real)
/// * `x` - Radial coordinate `ξ ≥ 1`
///
/// # Examples
///
/// ```
/// use scirs2_special::pro_rad2;
///
/// // m = 0 cases match SciPy pro_rad2 to ~1e-5
/// let (q_val, q_prime) = pro_rad2(0, 1, 1.0, 1.5).unwrap();
/// assert!(q_val.is_finite());
/// assert!(q_prime.is_finite());
/// ```
#[allow(dead_code)]
pub fn pro_rad2(m: i32, n: i32, c: f64, x: f64) -> SpecialResult<(f64, f64)> {
    if m < 0 || n < m {
        return Err(SpecialError::DomainError(
            "Parameters must satisfy m ≥ 0 and n ≥ m".to_string(),
        ));
    }
    if x < 1.0 {
        return Err(SpecialError::DomainError(
            "Radial coordinate x must be ≥ 1.0".to_string(),
        ));
    }
    if c.is_nan() || x.is_nan() {
        return Ok((f64::NAN, f64::NAN));
    }
    if c == 0.0 {
        // For c = 0, R^(2) reduces to associated Legendre of the second kind Q_n^m
        return crate::spheroidal::helpers::legendre_associated_second_kind(n, m, x);
    }
    radial_function(
        SpheroidalParity::Prolate,
        SphericalBesselKind::Second,
        m,
        n,
        c.abs(),
        x,
    )
}
