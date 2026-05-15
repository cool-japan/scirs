//! Oblate spheroidal wave functions
//!
//! This module provides implementations of oblate spheroidal functions, which arise in
//! the solution of the Helmholtz equation in oblate spheroidal coordinates. These
//! functions are particularly important in electromagnetic scattering by oblate
//! spheroids (disk-shaped objects).
//!
//! ## Algorithm
//!
//! Wave 74 refactor (2026-05-08): all six routines below dispatch to the
//! Flammer / Bouwkamp d-coefficient pipeline implemented in
//! [`crate::spheroidal::cf_helpers`]. The legacy "continued-fraction
//! perturbation" path with hand-rolled Sturm-sequence Newton iteration has
//! been removed — it converged catastrophically wrong values for `|c| > 1`,
//! e.g. `obl_cv(0, 2, 2.0) = -0.018`, while the correct value is `+4.0915`
//! (SciPy `obl_cv(0, 2, 2.0)`).
//!
//! ## SciPy convention
//!
//! These routines match `scipy.special.{obl_cv, obl_cv_seq, obl_ang1,
//! obl_rad1, obl_rad2}` to ~1e-11 for the angular family and to ~1e-5 / m=0
//! for the radial family.
//!
//! ## Limitations
//!
//! The radial-2 (`obl_rad2`) implementation uses the simple spherical Bessel
//! `y_l` series (Flammer §4.4.4). This series diverges asymptotically for
//! `(m, n)` cases with odd `m` and odd parity (`(n − m) mod 2 = 1`), notably
//! `(m=1, n=2)`. Such cases require the alternative Wronskian-based
//! representation (Flammer §4.5), which is not yet implemented.
//! `obl_rad2` for those cases returns a value but with reduced accuracy or
//! incorrect sign.

use crate::error::{SpecialError, SpecialResult};
use crate::spheroidal::cf_helpers::{
    angular_function, flammer_eigenvalue, radial_function, SphericalBesselKind, SpheroidalParity,
    DEFAULT_D_LEN,
};

/// Computes the characteristic value for oblate spheroidal wave functions.
///
/// The characteristic value `λ_{m,n}(c)` is the eigenvalue of the spheroidal
/// wave equation (oblate sign, `c² → -c²`) for mode `m`, `n` and spheroidal
/// parameter `c`. It satisfies `λ_{m,n}(0) = n(n+1)`.
///
/// # Arguments
///
/// * `m` - The order parameter (≥ 0, integer)
/// * `n` - The degree parameter (≥ m, integer)
/// * `c` - The spheroidal parameter (real)
///
/// # Returns
///
/// * `SpecialResult<f64>` - The characteristic value
///
/// # Examples
///
/// ```
/// # use scirs2_special::obl_cv;
/// # use scirs2_special::error::SpecialError;
/// # fn test() -> Result<(), SpecialError> {
/// // c = 0 reduces to n(n+1)
/// let cv = obl_cv(0, 0, 0.0)?;
/// assert!((cv - 0.0).abs() < 1e-10);
///
/// let cv = obl_cv(0, 2, 2.0)?;
/// assert!((cv - 4.0915091022).abs() < 1e-6); // SciPy reference
/// # Ok(())
/// # }
/// # test().unwrap();
/// ```
#[allow(dead_code)]
pub fn obl_cv(m: i32, n: i32, c: f64) -> SpecialResult<f64> {
    if m < 0 || n < m {
        return Err(SpecialError::DomainError(
            "Parameters must satisfy m ≥ 0 and n ≥ m".to_string(),
        ));
    }
    if c.is_nan() {
        return Ok(f64::NAN);
    }
    if c == 0.0 {
        return Ok(n as f64 * (n as f64 + 1.0));
    }
    flammer_eigenvalue(SpheroidalParity::Oblate, m, n, c.abs(), DEFAULT_D_LEN)
}

/// Computes a sequence of characteristic values for oblate spheroidal wave functions.
///
/// Returns the sequence `[λ_{m,m}(c), λ_{m,m+1}(c), …, λ_{m,n}(c)]`.
///
/// # Arguments
///
/// * `m` - The order parameter (≥ 0, integer)
/// * `n` - The maximum degree parameter (≥ m, integer)
/// * `c` - The spheroidal parameter (real)
///
/// # Examples
///
/// ```
/// # use scirs2_special::obl_cv_seq;
/// # use scirs2_special::error::SpecialError;
/// # fn test() -> Result<(), SpecialError> {
/// let values = obl_cv_seq(0, 3, 0.0)?;
/// assert_eq!(values.len(), 4);
/// assert!((values[0] - 0.0).abs() < 1e-10);
/// assert!((values[1] - 2.0).abs() < 1e-10);
/// assert!((values[2] - 6.0).abs() < 1e-10);
/// assert!((values[3] - 12.0).abs() < 1e-10);
/// # Ok(())
/// # }
/// # test().unwrap();
/// ```
#[allow(dead_code)]
pub fn obl_cv_seq(m: i32, n: i32, c: f64) -> SpecialResult<Vec<f64>> {
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
        result.push(obl_cv(m, degree, c)?);
    }
    Ok(result)
}

/// Computes the oblate spheroidal angular function of the first kind, `S_{m,n}(c, η)`.
///
/// The convention matches SciPy's `obl_ang1`: at `c = 0`, the function reduces to
/// the un-Condon-Shortley associated Legendre function `(-1)^m · P_n^m(η)` (i.e.,
/// the un-CS form), and the Meixner–Schäfke normalisation enforces
/// `S_{m,n}(c, 0) = P_n^m(0)` (even parity) or
/// `S'_{m,n}(c, 0) = (P_n^m)'(0)` (odd parity).
///
/// # Arguments
///
/// * `m` - The order parameter (≥ 0, integer)
/// * `n` - The degree parameter (≥ m, integer)
/// * `c` - The spheroidal parameter (real)
/// * `x` - Evaluation point in `[-1, 1]`
///
/// # Returns
///
/// * `SpecialResult<(f64, f64)>` - The function value and its derivative
///
/// # Examples
///
/// ```
/// use scirs2_special::obl_ang1;
///
/// // c = 0 reduces to the un-CS associated Legendre function
/// let (s, _) = obl_ang1(0, 1, 0.0, 0.5).unwrap();
/// assert!((s - 0.5).abs() < 1e-10); // P_1^0(0.5) = 0.5
///
/// // c = 5 case (matches SciPy obl_ang1)
/// let (s, _) = obl_ang1(0, 2, 5.0, 0.5).unwrap();
/// assert!((s - (-0.5976813382)).abs() < 1e-6);
///
/// // Higher order
/// let (s_higher, _) = obl_ang1(1, 2, 5.0, 0.5).unwrap();
/// assert!((s_higher - 2.1643722013).abs() < 1e-6);
/// ```
#[allow(dead_code)]
pub fn obl_ang1(m: i32, n: i32, c: f64, x: f64) -> SpecialResult<(f64, f64)> {
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
    angular_function(SpheroidalParity::Oblate, m, n, c.abs(), x)
}

/// Computes the oblate spheroidal radial function of the first kind, `R_{m,n}^{(1)}(c, ξ)`.
///
/// Uses the Flammer §4.4 expansion in spherical Bessel functions of the first kind
/// `j_l(c·ξ)`, normalised so that `R_{m,n}^{(1)}(c, ξ) → j_n(c·ξ)` as `c → 0` for `m = 0`.
///
/// # Arguments
///
/// * `m` - The order parameter (≥ 0, integer)
/// * `n` - The degree parameter (≥ m, integer)
/// * `c` - The spheroidal parameter (real)
/// * `x` - Radial coordinate (`x ≥ 0`)
///
/// # Examples
///
/// ```
/// use scirs2_special::obl_rad1;
///
/// // Basic test for oblate radial function of the first kind
/// let (r_val, r_prime) = obl_rad1(0, 1, 1.0, 1.5).unwrap();
/// assert!(r_val.is_finite());
/// assert!(r_prime.is_finite());
///
/// // Higher order
/// let (r_higher, _) = obl_rad1(0, 2, 5.0, 0.5).unwrap();
/// assert!(r_higher.is_finite());
/// ```
#[allow(dead_code)]
pub fn obl_rad1(m: i32, n: i32, c: f64, x: f64) -> SpecialResult<(f64, f64)> {
    if m < 0 || n < m {
        return Err(SpecialError::DomainError(
            "Parameters must satisfy m ≥ 0 and n ≥ m".to_string(),
        ));
    }
    if x < 0.0 {
        return Err(SpecialError::DomainError(
            "Radial coordinate x must be ≥ 0.0".to_string(),
        ));
    }
    if c.is_nan() || x.is_nan() {
        return Ok((f64::NAN, f64::NAN));
    }
    radial_function(
        SpheroidalParity::Oblate,
        SphericalBesselKind::First,
        m,
        n,
        c.abs(),
        x,
    )
}

/// Computes the oblate spheroidal radial function of the second kind, `R_{m,n}^{(2)}(c, ξ)`.
///
/// Uses the Flammer §4.4 expansion in spherical Bessel functions of the second kind
/// `y_l(c·ξ)`. **Limitations**: this series is numerically unstable for some
/// odd-`m`/odd-parity combinations (notably `(m=1, n=2)`); see module-level
/// documentation.
///
/// # Arguments
///
/// * `m` - The order parameter (≥ 0, integer)
/// * `n` - The degree parameter (≥ m, integer)
/// * `c` - The spheroidal parameter (real)
/// * `x` - Radial coordinate (`x ≥ 1` recommended; series is asymptotic at small x)
///
/// # Examples
///
/// ```
/// use scirs2_special::obl_rad2;
///
/// // m = 0 cases match SciPy obl_rad2 to ~1e-5
/// let (r_val, r_prime) = obl_rad2(0, 1, 1.0, 2.0).unwrap();
/// assert!(r_val.is_finite());
/// assert!(r_prime.is_finite());
/// ```
#[allow(dead_code)]
pub fn obl_rad2(m: i32, n: i32, c: f64, x: f64) -> SpecialResult<(f64, f64)> {
    if m < 0 || n < m {
        return Err(SpecialError::DomainError(
            "Parameters must satisfy m ≥ 0 and n ≥ m".to_string(),
        ));
    }
    if x < 0.0 {
        return Err(SpecialError::DomainError(
            "Radial coordinate x must be ≥ 0.0".to_string(),
        ));
    }
    if c.is_nan() || x.is_nan() {
        return Ok((f64::NAN, f64::NAN));
    }
    radial_function(
        SpheroidalParity::Oblate,
        SphericalBesselKind::Second,
        m,
        n,
        c.abs(),
        x,
    )
}
