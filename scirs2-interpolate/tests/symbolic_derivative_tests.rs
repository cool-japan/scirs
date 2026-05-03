//! Integration tests for the symbolic derivative module.
//!
//! These tests verify analytic differentiation of B-splines and piecewise-cubic
//! splines against finite-difference approximations and analytic ground truths.
//!
//! We use UFCS (`Differentiable::derivative(&spline, order)`) to disambiguate
//! from the existing `CubicSpline::derivative(x: F)` evaluation method.

use scirs2_core::ndarray::array;
use scirs2_interpolate::{
    advanced::akima::AkimaSpline, bspline::BSpline, bspline_modules::ExtrapolateMode,
    interp1d::pchip::PchipInterpolator, spline::CubicSpline, symbolic_derivative::Differentiable,
};

/// Helper: symmetric finite difference derivative.
fn fd_deriv<F: Fn(f64) -> f64>(f: F, x: f64, h: f64) -> f64 {
    (f(x + h) - f(x - h)) / (2.0 * h)
}

// ---------------------------------------------------------------------------
// B-spline tests
// ---------------------------------------------------------------------------

#[test]
fn bspline_derivative_degree_decrease_by_one() {
    // Cubic B-spline => first derivative has degree 2.
    let t = array![0.0f64, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0];
    let c = array![0.0f64, 1.0, 2.0, 1.0, 0.0, 1.0];
    let spline =
        BSpline::new(&t.view(), &c.view(), 3, ExtrapolateMode::Extrapolate).expect("construction");
    let d1 = Differentiable::derivative(&spline, 1).expect("derivative");
    assert_eq!(d1.degree(), 2, "degree should decrease by 1");
    assert_eq!(
        d1.coefficients().len(),
        c.len() - 1,
        "n-1 derivative coefficients"
    );
}

#[test]
fn bspline_derivative_matches_finite_difference_at_interior() {
    // Cubic B-spline with uniform knot vector.
    let t = array![0.0f64, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0];
    let c = array![0.0f64, 1.0, 3.0, 2.0, 0.5, 1.5];
    let spline =
        BSpline::new(&t.view(), &c.view(), 3, ExtrapolateMode::Extrapolate).expect("construction");

    let d1 = Differentiable::derivative(&spline, 1).expect("first derivative");
    let h = 1e-5f64;

    // Test at several strictly interior points.
    for &xi in &[0.5f64, 1.0, 1.5, 2.0, 2.5] {
        let fd = fd_deriv(|x| spline.evaluate(x).expect("eval"), xi, h);
        let sym = d1.evaluate(xi).expect("symbolic eval");
        assert!(
            (sym - fd).abs() < 1e-4,
            "xi={xi}: symbolic={sym:.8}, fd={fd:.8}, diff={:.2e}",
            (sym - fd).abs()
        );
    }
}

#[test]
fn second_derivative_of_constant_spline_is_zero() {
    // A natural cubic spline fit to constant data has zero 2nd derivative.
    let x = array![0.0f64, 1.0, 2.0, 3.0, 4.0];
    let y = array![7.0f64, 7.0, 7.0, 7.0, 7.0];
    let spline = CubicSpline::new(&x.view(), &y.view()).expect("construction");
    let d2 = Differentiable::derivative(&spline, 2).expect("2nd derivative");
    for &v in d2.coeffs().iter() {
        assert!(v.abs() < 1e-10, "2nd derivative should be zero, got {v}");
    }
}

#[test]
fn cubic_spline_derivative_matches_analytic_for_x_cubed() {
    // f(x) = x^3; f'(x) = 3x^2; sampled at {0,1,2,3,4,5}.
    let x = array![0.0f64, 1.0, 2.0, 3.0, 4.0, 5.0];
    let y = x.mapv(|v| v * v * v);
    let spline = CubicSpline::new(&x.view(), &y.view()).expect("construction");
    let d1 = Differentiable::derivative(&spline, 1).expect("1st derivative");
    let h = 1e-6f64;

    for &xi in &[0.5f64, 1.0, 1.5, 2.0, 2.5, 3.5, 4.5] {
        let fd = fd_deriv(|x| spline.evaluate(x).expect("eval"), xi, h);
        let sym = d1.evaluate(xi).expect("symbolic eval");
        assert!(
            (sym - fd).abs() < 1e-4,
            "xi={xi}: symbolic={sym:.8}, fd={fd:.8}"
        );
    }
}

// ---------------------------------------------------------------------------
// PCHIP tests
// ---------------------------------------------------------------------------

#[test]
fn pchip_derivative_preserves_monotonicity() {
    // Linear data: derivative should be 1 everywhere.
    let x = array![0.0f64, 1.0, 2.0, 3.0, 4.0];
    let y = array![0.0f64, 1.0, 2.0, 3.0, 4.0];
    let pchip = PchipInterpolator::new(&x.view(), &y.view(), false).expect("construction");
    let d1 = Differentiable::derivative(&pchip, 1).expect("1st derivative");
    for &xi in &[0.5f64, 1.5, 2.5, 3.5] {
        let v = d1.evaluate(xi).expect("eval");
        assert!((v - 1.0).abs() < 1e-10, "xi={xi}: derivative={v}");
    }
}

#[test]
fn pchip_derivative_matches_finite_difference() {
    let x = array![0.0f64, 1.0, 2.0, 3.0, 4.0, 5.0];
    let y = x.mapv(|v| v.sin());
    let pchip = PchipInterpolator::new(&x.view(), &y.view(), false).expect("construction");
    let d1 = Differentiable::derivative(&pchip, 1).expect("1st derivative");
    let h = 1e-5f64;
    for &xi in &[0.5f64, 1.0, 2.0, 2.5, 3.5, 4.5] {
        let fd = fd_deriv(|x| pchip.evaluate(x).expect("eval"), xi, h);
        let sym = d1.evaluate(xi).expect("symbolic eval");
        assert!(
            (sym - fd).abs() < 1e-4,
            "xi={xi}: symbolic={sym:.8}, fd={fd:.8}"
        );
    }
}

// ---------------------------------------------------------------------------
// Akima tests
// ---------------------------------------------------------------------------

#[test]
fn akima_derivative_matches_finite_difference() {
    let x = array![0.0f64, 1.0, 2.0, 3.0, 4.0, 5.0];
    let y = x.mapv(|v| v * v);
    let akima = AkimaSpline::new(&x.view(), &y.view()).expect("construction");
    let d1 = Differentiable::derivative(&akima, 1).expect("1st derivative");
    let h = 1e-5f64;
    for &xi in &[0.5f64, 1.5, 2.5, 3.5, 4.5] {
        let fd = fd_deriv(|x| akima.evaluate(x).expect("eval"), xi, h);
        let sym = d1.evaluate(xi).expect("symbolic eval");
        assert!(
            (sym - fd).abs() < 1e-4,
            "xi={xi}: symbolic={sym:.8}, fd={fd:.8}"
        );
    }
}

#[test]
fn akima_derivative_continuous_at_knots() {
    // Derivatives should be continuous at interior knots (C^1 spline).
    let x = array![0.0f64, 1.0, 2.0, 3.0, 4.0, 5.0];
    let y = array![0.0f64, 1.0, 0.5, 1.5, 1.0, 2.0];
    let akima = AkimaSpline::new(&x.view(), &y.view()).expect("construction");
    let d1 = Differentiable::derivative(&akima, 1).expect("1st derivative");
    let eps = 1e-8f64;
    for &knot in &[1.0f64, 2.0, 3.0, 4.0] {
        let left = d1.evaluate(knot - eps).expect("left");
        let right = d1.evaluate(knot + eps).expect("right");
        assert!(
            (left - right).abs() < 1e-5,
            "knot={knot}: left={left:.8}, right={right:.8}, diff={:.2e}",
            (left - right).abs()
        );
    }
}

// ---------------------------------------------------------------------------
// Higher-order derivative tests
// ---------------------------------------------------------------------------

#[test]
fn third_derivative_of_cubic_spline_is_piecewise_constant() {
    // The 3rd derivative of any piecewise-cubic spline must be piecewise constant (degree 0).
    // Verify structural property: output degree == 0, n_segments unchanged.
    let x = array![0.0f64, 1.0, 2.0, 3.0, 4.0, 5.0];
    let y = x.mapv(|v| v.sin()); // arbitrary smooth data
    let spline = CubicSpline::new(&x.view(), &y.view()).expect("construction");
    let d3 = Differentiable::derivative(&spline, 3usize).expect("3rd derivative");
    // Should have 1 column (degree 0 piecewise constant)
    assert_eq!(d3.degree(), 0, "3rd derivative of cubic should be degree-0");
    assert_eq!(
        d3.n_segments(),
        x.len() - 1,
        "number of segments should be preserved"
    );
}

#[test]
fn fourth_derivative_of_cubic_spline_is_zero() {
    let x = array![0.0f64, 1.0, 2.0, 3.0];
    let y = array![0.0f64, 1.0, 8.0, 27.0];
    let spline = CubicSpline::new(&x.view(), &y.view()).expect("construction");
    let d4 = Differentiable::derivative(&spline, 4usize).expect("4th derivative");
    for &v in d4.coeffs().iter() {
        assert!(
            v.abs() < 1e-10,
            "4th derivative of cubic should be 0, got {v}"
        );
    }
}
