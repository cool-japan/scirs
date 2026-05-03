//! Symbolic derivative evaluation for spline types.
//!
//! This module provides analytic differentiation for the spline types in this crate.
//! The `Differentiable` trait returns a `PiecewisePolynomial<F>` for piecewise-cubic
//! types (PCHIP, Akima, CubicSpline) and a new `BSpline<T>` for B-splines.
//!
//! # B-spline derivative recurrence
//!
//! For a B-spline of degree `k` with knot vector `t` and coefficients `c`:
//!
//! ```text
//! c'_i = k * (c_{i+1} - c_i) / (t_{i+k+1} - t_{i+1}),  for 0 <= i < n-1
//! ```
//!
//! The derivative has degree `k-1`, knot vector `t[1..n+k]` (interior knots), and
//! `n-1` coefficients.
//!
//! # Piecewise-cubic derivative
//!
//! For each segment with local-coordinate polynomial
//! `a + b*(x - x_i) + c*(x - x_i)^2 + d*(x - x_i)^3`,
//! the `n`-th derivative reduces the degree by `n`.

use crate::bspline::BSpline;
use crate::error::{InterpolateError, InterpolateResult};
use crate::spline_calculus::PiecewisePolynomial;
use crate::traits::InterpolationFloat;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::Zero;
use std::fmt::{Debug, Display};
use std::ops::{AddAssign, DivAssign, MulAssign, RemAssign, SubAssign};

// ---------------------------------------------------------------------------
// Differentiable trait
// ---------------------------------------------------------------------------

/// Trait for types that can be analytically differentiated.
///
/// `type Derivative` is the type returned by `derivative`.  For B-splines this is
/// another `BSpline`; for piecewise-cubic splines this is a `PiecewisePolynomial`.
pub trait Differentiable {
    /// The type of the derivative.
    type Derivative;

    /// Return the `order`-th analytic derivative.
    ///
    /// Order 0 returns a representation of the function itself.
    /// For finite-degree splines, sufficiently high order returns the zero polynomial.
    fn derivative(&self, order: usize) -> InterpolateResult<Self::Derivative>;
}

// ---------------------------------------------------------------------------
// BSpline<T> implementation
// ---------------------------------------------------------------------------

impl<T> Differentiable for BSpline<T>
where
    T: InterpolationFloat
        + Zero
        + AddAssign
        + SubAssign
        + MulAssign
        + DivAssign
        + RemAssign
        + Debug
        + Display
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>,
{
    type Derivative = BSpline<T>;

    fn derivative(&self, order: usize) -> InterpolateResult<BSpline<T>> {
        if order == 0 {
            return Ok(self.clone());
        }
        let mut current = self.clone();
        for _ in 0..order {
            current = bspline_derivative_once(&current)?;
            // If degree reaches 0, further differentiation yields the zero spline
            if current.degree() == 0 {
                // Check whether we still have more orders to apply
                // The current degree-0 spline is the derivative; stop here.
                break;
            }
        }
        Ok(current)
    }
}

/// Differentiate a B-spline once using the B-spline recurrence.
///
/// For degree-0 splines the derivative is identically zero but is represented
/// as a degree-0 B-spline with all-zero coefficients.
fn bspline_derivative_once<T>(spline: &BSpline<T>) -> InterpolateResult<BSpline<T>>
where
    T: InterpolationFloat
        + Zero
        + AddAssign
        + SubAssign
        + MulAssign
        + DivAssign
        + RemAssign
        + Debug
        + Display
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>,
{
    let k = spline.degree();
    let t = spline.knot_vector();
    let c = spline.coefficients();
    let n = c.len(); // number of coefficients

    if k == 0 {
        // Degree-0: derivative is zero.  Return a degree-0 spline with zeroed coefficients.
        // Preserve knot structure: drop first and last knot (reduces by 1 each side).
        if t.len() < 3 {
            return Err(InterpolateError::invalid_input(
                "cannot differentiate a degree-0 B-spline with fewer than 3 knots".to_string(),
            ));
        }
        let new_t = t.slice(scirs2_core::ndarray::s![1..t.len() - 1]).to_owned();
        let new_c = Array1::<T>::zeros(new_t.len() - 1);
        return BSpline::new(&new_t.view(), &new_c.view(), 0, spline.extrapolate_mode());
    }

    if n < 2 {
        return Err(InterpolateError::invalid_input(
            "B-spline must have at least 2 coefficients to differentiate".to_string(),
        ));
    }

    // New degree = k - 1
    let new_k = k - 1;
    // New coefficients: n-1 values
    let new_n = n - 1;

    let k_f = T::from_usize(k).ok_or_else(|| {
        InterpolateError::ComputationError("failed to convert B-spline degree to float".to_string())
    })?;

    let mut new_c = Array1::<T>::zeros(new_n);
    for i in 0..new_n {
        // Denominator = t[i + k + 1] - t[i + 1]
        let denom = t[i + k + 1] - t[i + 1];
        if denom.abs() < T::epsilon() {
            // Repeated knot: contribution is zero
            new_c[i] = T::zero();
        } else {
            new_c[i] = k_f * (c[i + 1] - c[i]) / denom;
        }
    }

    // New knot vector: t[1..n+k] — drop first and last knot
    // Original has n + k + 1 knots; new has new_n + new_k + 1 = (n-1) + (k-1) + 1 = n + k - 1
    // That means we drop first and last from original -> t[1..n+k]
    let new_t = t.slice(scirs2_core::ndarray::s![1..t.len() - 1]).to_owned();

    if new_t.len() != new_n + new_k + 1 {
        return Err(InterpolateError::ComputationError(format!(
            "B-spline derivative knot count mismatch: expected {}, got {}",
            new_n + new_k + 1,
            new_t.len()
        )));
    }

    BSpline::new(
        &new_t.view(),
        &new_c.view(),
        new_k,
        spline.extrapolate_mode(),
    )
}

// ---------------------------------------------------------------------------
// PiecewiseCubicDifferentiable — helper for piecewise-cubic splines
// ---------------------------------------------------------------------------

/// Differentiate a piecewise-cubic represented as coefficient matrix `(n_seg, 4)`.
///
/// Each segment is  `a + b*(x - x_i) + c*(x - x_i)^2 + d*(x - x_i)^3`.
/// `order` must be in `[1, 4]`; order >= 4 yields the zero polynomial.
pub fn differentiate_piecewise_cubic<F: InterpolationFloat>(
    breakpoints: &Array1<F>,
    coeffs: &Array2<F>,
    order: usize,
) -> InterpolateResult<PiecewisePolynomial<F>> {
    if order == 0 {
        // Return unchanged representation
        return Ok(PiecewisePolynomial::new(
            breakpoints.clone(),
            coeffs.clone(),
        )?);
    }

    let n_seg = coeffs.nrows();
    if n_seg == 0 {
        return Err(InterpolateError::invalid_input(
            "cannot differentiate empty piecewise polynomial".to_string(),
        ));
    }
    if coeffs.ncols() != 4 {
        return Err(InterpolateError::invalid_input(
            "differentiate_piecewise_cubic expects coefficients of shape (n_seg, 4)".to_string(),
        ));
    }

    let remaining_cols = if order >= 4 { 0usize } else { 4 - order };

    if remaining_cols == 0 {
        // derivative of order >= 4 of a cubic = zero
        let zero_coeffs = Array2::<F>::zeros((n_seg, 1));
        return Ok(PiecewisePolynomial::new(breakpoints.clone(), zero_coeffs)?);
    }

    // Apply derivative `order` times.
    // After `d` differentiations: the coefficient of x^j becomes (j+d)! / j! * original[j+d]
    // Working iteratively:
    let mut work: Array2<F> = Array2::zeros((n_seg, 4));
    for i in 0..n_seg {
        for j in 0..4 {
            work[[i, j]] = coeffs[[i, j]];
        }
    }

    let mut current_cols = 4usize;
    for _ in 0..order {
        if current_cols <= 1 {
            let zero_coeffs = Array2::<F>::zeros((n_seg, 1));
            return Ok(PiecewisePolynomial::new(breakpoints.clone(), zero_coeffs)?);
        }
        let new_cols = current_cols - 1;
        let mut new_work: Array2<F> = Array2::zeros((n_seg, new_cols));
        for i in 0..n_seg {
            for j in 0..new_cols {
                let factor = F::from_usize(j + 1).ok_or_else(|| {
                    InterpolateError::ComputationError(
                        "Failed to convert derivative factor to float".to_string(),
                    )
                })?;
                new_work[[i, j]] = work[[i, j + 1]] * factor;
            }
        }
        work = new_work;
        current_cols = new_cols;
    }

    // Trim to remaining_cols
    let mut out: Array2<F> = Array2::zeros((n_seg, current_cols));
    for i in 0..n_seg {
        for j in 0..current_cols {
            out[[i, j]] = work[[i, j]];
        }
    }

    Ok(PiecewisePolynomial::new(breakpoints.clone(), out)?)
}

// ---------------------------------------------------------------------------
// CubicSpline<F> implementation
// ---------------------------------------------------------------------------

use crate::spline::CubicSpline;

impl<F: InterpolationFloat> Differentiable for CubicSpline<F> {
    type Derivative = PiecewisePolynomial<F>;

    fn derivative(&self, order: usize) -> InterpolateResult<PiecewisePolynomial<F>> {
        if order == 0 {
            return Ok(PiecewisePolynomial::new(
                self.x().clone(),
                self.coeffs().clone(),
            )?);
        }
        differentiate_piecewise_cubic(self.x(), self.coeffs(), order)
    }
}

// ---------------------------------------------------------------------------
// AkimaSpline<F> implementation (colocated here; accesses via pub(crate) accessors)
// ---------------------------------------------------------------------------

use crate::advanced::akima::AkimaSpline;

impl<F: InterpolationFloat> Differentiable for AkimaSpline<F> {
    type Derivative = PiecewisePolynomial<F>;

    fn derivative(&self, order: usize) -> InterpolateResult<PiecewisePolynomial<F>> {
        if order == 0 {
            return Ok(PiecewisePolynomial::new(
                self.breakpoints_owned(),
                self.segment_coeffs_owned(),
            )?);
        }
        differentiate_piecewise_cubic(
            &self.breakpoints_owned(),
            &self.segment_coeffs_owned(),
            order,
        )
    }
}

// ---------------------------------------------------------------------------
// PchipInterpolator implementation (generic F)
// ---------------------------------------------------------------------------

use crate::interp1d::pchip::PchipInterpolator;
use scirs2_core::numeric::FromPrimitive;

impl<F: InterpolationFloat + FromPrimitive> Differentiable for PchipInterpolator<F> {
    type Derivative = PiecewisePolynomial<F>;

    fn derivative(&self, order: usize) -> InterpolateResult<PiecewisePolynomial<F>> {
        // Convert PCHIP Hermite representation into piecewise-cubic power-basis coefficients
        let (breakpoints, cubic_coeffs) = self.to_power_basis_coeffs()?;
        if order == 0 {
            return Ok(PiecewisePolynomial::new(breakpoints, cubic_coeffs)?);
        }
        differentiate_piecewise_cubic(&breakpoints, &cubic_coeffs, order)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bspline::BSpline;
    use crate::bspline_modules::ExtrapolateMode;
    use crate::interp1d::pchip::PchipInterpolator;
    use crate::spline::CubicSpline;
    use scirs2_core::ndarray::array;

    #[test]
    fn bspline_derivative_degree_decrease_by_one() {
        // Cubic B-spline => degree-2 derivative.
        // Use UFCS to avoid ambiguity with BSpline's own evaluate methods.
        let t = array![0.0f64, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0];
        let c = array![0.0f64, 1.0, 2.0, 1.0, 0.0, 1.0];
        let spline = BSpline::new(&t.view(), &c.view(), 3, ExtrapolateMode::Extrapolate)
            .expect("Construction failed");
        let d = Differentiable::derivative(&spline, 1usize).expect("Derivative failed");
        assert_eq!(d.degree(), 2, "degree should decrease by 1");
        assert_eq!(d.coefficients().len(), c.len() - 1);
    }

    #[test]
    fn second_derivative_of_constant_spline_is_zero() {
        let x = array![0.0f64, 1.0, 2.0, 3.0];
        let y = array![5.0f64, 5.0, 5.0, 5.0];
        let spline = CubicSpline::new(&x.view(), &y.view()).expect("Construction failed");
        // UFCS avoids calling CubicSpline::derivative(xnew: F).
        let d2 = Differentiable::derivative(&spline, 2usize).expect("2nd derivative failed");
        for &v in d2.coeffs().iter() {
            assert!(v.abs() < 1e-10, "Expected zero, got {}", v);
        }
    }

    #[test]
    fn cubic_spline_derivative_matches_analytic_for_x_cubed() {
        // f(x) = x^3; f'(x) = 3x^2; sample at {0,1,2,3}
        let x = array![0.0f64, 1.0, 2.0, 3.0];
        let y = array![0.0f64, 1.0, 8.0, 27.0];
        let spline = CubicSpline::new(&x.view(), &y.view()).expect("Construction failed");
        let d1 = Differentiable::derivative(&spline, 1usize).expect("1st derivative failed");
        // Check finite-difference agreement at interior points.
        for xi in [0.5, 1.0, 1.5, 2.0, 2.5] {
            let h = 1e-6;
            let fd = (spline.evaluate(xi + h).expect("eval+h")
                - spline.evaluate(xi - h).expect("eval-h"))
                / (2.0 * h);
            let symbolic = d1.evaluate(xi).expect("symbolic eval");
            assert!(
                (symbolic - fd).abs() < 1e-4,
                "xi={xi}: symbolic={symbolic}, fd={fd}"
            );
        }
    }

    #[test]
    fn pchip_derivative_preserves_monotonicity() {
        let x = array![0.0f64, 1.0, 2.0, 3.0, 4.0];
        let y = array![0.0f64, 1.0, 2.0, 3.0, 4.0]; // strictly increasing linear
        let pchip =
            PchipInterpolator::new(&x.view(), &y.view(), false).expect("Construction failed");
        let d1 = Differentiable::derivative(&pchip, 1usize).expect("1st derivative failed");
        for xi in [0.5f64, 1.5, 2.5, 3.5] {
            let v = d1.evaluate(xi).expect("eval");
            assert!((v - 1.0).abs() < 1e-10, "xi={xi}: derivative={v}");
        }
    }
}
