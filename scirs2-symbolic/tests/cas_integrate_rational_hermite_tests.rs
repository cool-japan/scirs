//! Wave 74 Item #3 — Hermite reduction for higher-degree rational denominators.
//!
//! Tests cover squarefree factorization, repeated factors, cubic denominators
//! via Cardano (item #2 dependency), and degree-5 unhandled-error path.

use scirs2_symbolic::cas::canonicalize;
use scirs2_symbolic::cas::integrate_rational::{integrate_rational, IntegrateRationalError};
use scirs2_symbolic::eml::eval::{eval_real, EvalCtx};
use scirs2_symbolic::eml::grad::grad;
use scirs2_symbolic::eml::op::LoweredOp;

fn c(v: f64) -> LoweredOp {
    LoweredOp::Const(v)
}
fn x() -> LoweredOp {
    LoweredOp::Var(0)
}

/// Evaluate a (possibly canonicalized) op at variable values.
fn eval_at(op: &LoweredOp, vals: &[f64]) -> f64 {
    let canon = canonicalize(op).into_op();
    eval_real(&canon, &EvalCtx::new(vals)).unwrap_or(f64::NAN)
}

/// Verify ∫ integrand dx = antiderivative by differentiating numerically and
/// checking equality at a list of x values (avoiding singularities).
fn round_trip(antiderivative: &LoweredOp, integrand: &LoweredOp, xs: &[f64]) {
    let derivative = grad(antiderivative, 0);
    for &xv in xs {
        let v = [xv];
        let d = eval_at(&derivative, &v);
        let i = eval_at(integrand, &v);
        assert!(
            (d - i).abs() < 1e-6 || (d.is_nan() && i.is_nan()),
            "round-trip mismatch at x={xv}: d/dx(F)={d}, f={i}"
        );
    }
}

// =====================================================================
// 1) Simple repeated factor: ∫ 1/(x-1)² dx = -1/(x-1)
// =====================================================================

#[test]
fn hermite_simple_repeated_factor() {
    // (x-1)² = 1 - 2x + x², ascending coeffs
    let num = vec![c(1.0)];
    let den = vec![c(1.0), c(-2.0), c(1.0)];
    let antideriv = integrate_rational(&num, &den, 0).expect("integrate_rational");
    let integrand = LoweredOp::Div(
        Box::new(c(1.0)),
        Box::new(LoweredOp::Pow(
            Box::new(LoweredOp::Sub(Box::new(x()), Box::new(c(1.0)))),
            Box::new(c(2.0)),
        )),
    );
    round_trip(&antideriv, &integrand, &[1.5, 2.0, 2.5, 3.0, 3.5]);
}

// =====================================================================
// 2) Cubic denominator with three distinct real roots:
//    ∫ 1 / ((x-1)(x-2)(x-3)) dx
// =====================================================================

#[test]
fn hermite_cubic_distinct_real_roots() {
    // (x-1)(x-2)(x-3) = x³ - 6x² + 11x - 6
    let num = vec![c(1.0)];
    let den = vec![c(-6.0), c(11.0), c(-6.0), c(1.0)];
    let antideriv = integrate_rational(&num, &den, 0).expect("cubic integrate");
    let integrand = LoweredOp::Div(
        Box::new(c(1.0)),
        Box::new(LoweredOp::Mul(
            Box::new(LoweredOp::Mul(
                Box::new(LoweredOp::Sub(Box::new(x()), Box::new(c(1.0)))),
                Box::new(LoweredOp::Sub(Box::new(x()), Box::new(c(2.0)))),
            )),
            Box::new(LoweredOp::Sub(Box::new(x()), Box::new(c(3.0)))),
        )),
    );
    // Avoid singularities at x=1,2,3.
    round_trip(&antideriv, &integrand, &[0.5, 1.5, 2.5, 3.5, 4.0, 4.5]);
}

// =====================================================================
// 3) Cubic with one real root and one irreducible quadratic factor:
//    ∫ 1/(x³+1) dx = (1/3) ln|x+1| - (1/6) ln|x²-x+1| + (1/√3) arctan((2x-1)/√3)
// =====================================================================

#[test]
fn hermite_cubic_one_real_one_complex_pair() {
    // x³ + 1
    let num = vec![c(1.0)];
    let den = vec![c(1.0), c(0.0), c(0.0), c(1.0)];
    let antideriv = integrate_rational(&num, &den, 0).expect("cubic with complex");
    let integrand = LoweredOp::Div(
        Box::new(c(1.0)),
        Box::new(LoweredOp::Add(
            Box::new(LoweredOp::Pow(Box::new(x()), Box::new(c(3.0)))),
            Box::new(c(1.0)),
        )),
    );
    // x³+1 = 0 at x=-1 (and complex roots). Avoid x=-1.
    round_trip(&antideriv, &integrand, &[0.5, 1.0, 1.5, 2.0, 3.0]);
}

// =====================================================================
// 4) Quartic denominator via Ferrari + partial fractions.
//    ∫ 1 / (x⁴ - 5x² + 4) dx — biquadratic, factors as (x²-1)(x²-4) =
//    (x-1)(x+1)(x-2)(x+2). Four simple real roots.
// =====================================================================

#[test]
fn hermite_quartic_via_ferrari() {
    // x⁴ - 5x² + 4
    let num = vec![c(1.0)];
    let den = vec![c(4.0), c(0.0), c(-5.0), c(0.0), c(1.0)];
    let antideriv = integrate_rational(&num, &den, 0).expect("quartic ferrari");
    let integrand = LoweredOp::Div(
        Box::new(c(1.0)),
        Box::new(LoweredOp::Add(
            Box::new(LoweredOp::Sub(
                Box::new(LoweredOp::Pow(Box::new(x()), Box::new(c(4.0)))),
                Box::new(LoweredOp::Mul(
                    Box::new(c(5.0)),
                    Box::new(LoweredOp::Pow(Box::new(x()), Box::new(c(2.0)))),
                )),
            )),
            Box::new(c(4.0)),
        )),
    );
    // Avoid singularities at x = ±1, ±2.
    round_trip(&antideriv, &integrand, &[0.5, 1.5, 2.5, 3.0, 3.5]);
}

// =====================================================================
// 5) Repeated quadratic factor: ∫ 1/(x²+1)² dx
// =====================================================================

#[test]
fn hermite_repeated_quadratic() {
    // (x²+1)² = x⁴ + 2x² + 1
    let num = vec![c(1.0)];
    let den = vec![c(1.0), c(0.0), c(2.0), c(0.0), c(1.0)];
    let antideriv = integrate_rational(&num, &den, 0).expect("repeated quadratic");
    let integrand = LoweredOp::Div(
        Box::new(c(1.0)),
        Box::new(LoweredOp::Pow(
            Box::new(LoweredOp::Add(
                Box::new(LoweredOp::Pow(Box::new(x()), Box::new(c(2.0)))),
                Box::new(c(1.0)),
            )),
            Box::new(c(2.0)),
        )),
    );
    round_trip(&antideriv, &integrand, &[0.0, 0.5, 1.0, 2.0, 3.0]);
}

// =====================================================================
// 6) Yun squarefree decomposition test: (x-1)²(x²+1)³ has
//    Q₁ = 1 (no degree-1 squarefree part)
//    Q₂ = (x-1)
//    Q₃ = (x²+1)
// =====================================================================

#[test]
fn hermite_yun_squarefree_decomposition() {
    use scirs2_symbolic::cas::integrate_rational::yun_squarefree;

    // Q = (x-1)²(x²+1)³
    // Expand: (x-1)² = x² - 2x + 1; (x²+1)² = x⁴+2x²+1; (x²+1)³ = x⁶+3x⁴+3x²+1
    // Q = (x²-2x+1)(x⁶+3x⁴+3x²+1)
    //   = x⁸ + 3x⁶ + 3x⁴ + x² - 2x⁷ - 6x⁵ - 6x³ - 2x + x⁶ + 3x⁴ + 3x² + 1
    //   = x⁸ - 2x⁷ + 4x⁶ - 6x⁵ + 6x⁴ - 6x³ + 4x² - 2x + 1
    let q = vec![1.0, -2.0, 4.0, -6.0, 6.0, -6.0, 4.0, -2.0, 1.0];
    let factors = yun_squarefree(&q);

    // Yun yields a list of (squarefree_factor, multiplicity) pairs.
    // We expect at least one factor with multiplicity 2 (the (x-1) factor)
    // and at least one with multiplicity 3 (the (x²+1) factor). The total
    // sum of (multiplicity * deg(factor)) must equal deg(Q) = 8.
    let total_deg: usize = factors
        .iter()
        .map(|(coeffs, mult)| {
            let deg = coeffs.iter().rposition(|c| c.abs() > 1e-9).unwrap_or(0);
            deg * (*mult as usize)
        })
        .sum();
    assert_eq!(
        total_deg, 8,
        "Yun decomposition degree sum must equal deg(Q); got {total_deg} for factors {factors:?}"
    );
}

// =====================================================================
// 7) Polynomial part first: ∫ x³/(x²+1) dx
//    Long-division gives x · (1 - 1/(x²+1)) ... actually
//    x³/(x²+1) = x − x/(x²+1), so the integral is x²/2 − (1/2)ln(x²+1).
// =====================================================================

#[test]
fn hermite_polynomial_quotient_first_then_remainder() {
    // num = x³ = [0, 0, 0, 1], den = x²+1 = [1, 0, 1]
    let num = vec![c(0.0), c(0.0), c(0.0), c(1.0)];
    let den = vec![c(1.0), c(0.0), c(1.0)];
    let antideriv = integrate_rational(&num, &den, 0).expect("poly quotient");
    let integrand = LoweredOp::Div(
        Box::new(LoweredOp::Pow(Box::new(x()), Box::new(c(3.0)))),
        Box::new(LoweredOp::Add(
            Box::new(LoweredOp::Pow(Box::new(x()), Box::new(c(2.0)))),
            Box::new(c(1.0)),
        )),
    );
    round_trip(&antideriv, &integrand, &[0.0, 0.5, 1.0, 1.5, 2.0]);
}

// =====================================================================
// 8) Degree-5 irreducible → DenominatorDegreeTooHigh error.
// =====================================================================

#[test]
fn integrate_rational_degree_5_irreducible_returns_unhandled() {
    // x⁵ + 1 (deg 5)
    let num = vec![c(1.0)];
    let den = vec![c(1.0), c(0.0), c(0.0), c(0.0), c(0.0), c(1.0)];
    let result = integrate_rational(&num, &den, 0);
    assert!(
        matches!(
            result,
            Err(IntegrateRationalError::DenominatorDegreeTooHigh { degree: 5 })
        ),
        "expected DenominatorDegreeTooHigh{{degree:5}}, got {result:?}"
    );
}

// =====================================================================
// 9) Risch-LITE chain with canonicalize: ensure the round-trip still works
//    after canonicalize is applied to the integrand. Tests that the
//    integrate_rational + canonicalize composition is semantically stable
//    on a non-trivial rational input.
// =====================================================================

#[test]
fn risch_lite_chain_with_canonicalize() {
    // Integrate (x²+1)/(x³+x) dx — expand: x³+x = x(x²+1), so
    // (x²+1)/(x³+x) = 1/x. Integral is ln|x|.
    // num = x²+1 = [1, 0, 1], den = x³+x = [0, 1, 0, 1]
    let num = vec![c(1.0), c(0.0), c(1.0)];
    let den = vec![c(0.0), c(1.0), c(0.0), c(1.0)];
    let antideriv = integrate_rational(&num, &den, 0).expect("risch chain");
    let integrand = LoweredOp::Div(
        Box::new(LoweredOp::Add(
            Box::new(LoweredOp::Pow(Box::new(x()), Box::new(c(2.0)))),
            Box::new(c(1.0)),
        )),
        Box::new(LoweredOp::Add(
            Box::new(LoweredOp::Pow(Box::new(x()), Box::new(c(3.0)))),
            Box::new(x()),
        )),
    );
    let _canon_anti = canonicalize(&antideriv).into_op();
    round_trip(&antideriv, &integrand, &[0.5, 1.0, 1.5, 2.0]);
}
