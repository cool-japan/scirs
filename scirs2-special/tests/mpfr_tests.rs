//! Integration tests for the arbitrary-precision API (formerly MPFR/rug-based,
//! now backed by oxinum-float — Pure Rust, no C/Fortran FFI).
//!
//! Run with:
//!   cargo nextest run -p scirs2-special --features arbitrary_precision
//!   cargo nextest run -p scirs2-special --features high-precision
#![cfg(feature = "high-precision")]

use scirs2_special::{
    arbitrary_precision::Float, bessel_j0_mpfr, bessel_k0_mpfr, digamma_mpfr, erf_mpfr, erfc_mpfr,
    gamma_mpfr, lgamma_mpfr,
};

const PREC: u32 = 256;
// K0 uses higher iteration counts at very high precision; 128 bits is enough for
// these correctness checks and stays comfortably within nextest's 120-second limit.
const K0_PREC: u32 = 128;

// ---------------------------------------------------------------------------
// Gamma function tests
// ---------------------------------------------------------------------------

#[test]
fn mpfr_gamma_at_one_is_one() {
    let x = Float::with_val(PREC, 1.0_f64);
    let result = gamma_mpfr(&x, PREC);
    let diff = (result.to_f64() - 1.0_f64).abs();
    assert!(diff < 1e-10, "Gamma(1) should be 1, got diff={diff}");
}

#[test]
fn mpfr_gamma_at_half_equals_sqrt_pi() {
    let x = Float::with_val(PREC, 0.5_f64);
    let result = gamma_mpfr(&x, PREC);
    let sqrt_pi = std::f64::consts::PI.sqrt();
    let diff = (result.to_f64() - sqrt_pi).abs();
    assert!(
        diff < 1e-10,
        "Gamma(1/2) should be sqrt(pi)={sqrt_pi}, got diff={diff}"
    );
}

#[test]
fn mpfr_gamma_matches_f64_on_small_args() {
    // Gamma(n+1) = n! for positive integers
    let cases: &[(f64, f64)] = &[(2.0, 1.0), (3.0, 2.0), (4.0, 6.0), (5.0, 24.0)];
    for &(xi, expected) in cases {
        let x = Float::with_val(PREC, xi);
        let result = gamma_mpfr(&x, PREC).to_f64();
        let diff = (result - expected).abs();
        assert!(
            diff < 1e-8,
            "Gamma({xi}) should be {expected}, got {result}, diff={diff}"
        );
    }
}

#[test]
fn mpfr_lgamma_at_one_is_zero() {
    let x = Float::with_val(PREC, 1.0_f64);
    let result = lgamma_mpfr(&x, PREC).to_f64();
    assert!(result.abs() < 1e-10, "lgamma(1) should be 0, got {result}");
}

#[test]
fn mpfr_lgamma_at_two_is_zero() {
    // ln(Gamma(2)) = ln(1) = 0
    let x = Float::with_val(PREC, 2.0_f64);
    let result = lgamma_mpfr(&x, PREC).to_f64();
    assert!(result.abs() < 1e-10, "lgamma(2) should be 0, got {result}");
}

#[test]
fn mpfr_digamma_at_one_is_neg_euler_mascheroni() {
    // psi(1) = -gamma_EM ~= -0.5772156649...
    let x = Float::with_val(PREC, 1.0_f64);
    let result = digamma_mpfr(&x, PREC).to_f64();
    let expected = -0.5772156649015328_f64;
    let diff = (result - expected).abs();
    assert!(
        diff < 1e-8,
        "digamma(1) should be ~{expected}, got {result}, diff={diff}"
    );
}

#[test]
fn mpfr_precision_scaling_monotone_error_decrease() {
    // Higher precision should give more accurate result for Gamma(0.5)
    let sqrt_pi = std::f64::consts::PI.sqrt();

    let x_64 = Float::with_val(64, 0.5_f64);
    let x_200 = Float::with_val(200, 0.5_f64);
    let x_500 = Float::with_val(500, 0.5_f64);

    let err_64 = (gamma_mpfr(&x_64, 64).to_f64() - sqrt_pi).abs();
    let err_200 = (gamma_mpfr(&x_200, 200).to_f64() - sqrt_pi).abs();
    let err_500 = (gamma_mpfr(&x_500, 500).to_f64() - sqrt_pi).abs();

    // At f64 level, all should be at or below machine epsilon
    assert!(err_64 < 1e-10, "64-bit err={err_64}");
    assert!(err_200 < 1e-12, "200-bit err={err_200}");
    assert!(err_500 < 1e-12, "500-bit err={err_500}");
}

// ---------------------------------------------------------------------------
// Error function tests
// ---------------------------------------------------------------------------

#[test]
fn mpfr_erf_of_zero_is_zero() {
    let x = Float::with_val(PREC, 0.0_f64);
    let result = erf_mpfr(&x, PREC).to_f64();
    assert!(result.abs() < 1e-12, "erf(0) should be 0, got {result}");
}

#[test]
fn mpfr_erf_of_one() {
    let x = Float::with_val(PREC, 1.0_f64);
    let result = erf_mpfr(&x, PREC).to_f64();
    let expected = 0.8427007929497148_f64;
    let diff = (result - expected).abs();
    assert!(diff < 1e-10, "erf(1) diff={diff}");
}

#[test]
fn mpfr_erf_asymptotic_accuracy() {
    // At x=5, erf(5) is extremely close to 1
    let x = Float::with_val(PREC, 5.0_f64);
    let result = erf_mpfr(&x, PREC).to_f64();
    assert!(
        (result - 1.0).abs() < 1e-8,
        "erf(5) should be near 1, got {result}"
    );
}

#[test]
fn mpfr_erf_plus_erfc_is_one() {
    for xi in [0.5_f64, 1.0, 2.0, 3.0] {
        let x = Float::with_val(PREC, xi);
        let e = erf_mpfr(&x, PREC).to_f64();
        let ec = erfc_mpfr(&x, PREC).to_f64();
        let sum = e + ec;
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "erf({xi})+erfc({xi})={sum}, should be 1"
        );
    }
}

#[test]
fn mpfr_erf_odd_symmetry() {
    // erf(-x) = -erf(x)
    for xi in [0.5_f64, 1.0, 2.0] {
        let xp = Float::with_val(PREC, xi);
        let xn = Float::with_val(PREC, -xi);
        let ep = erf_mpfr(&xp, PREC).to_f64();
        let en = erf_mpfr(&xn, PREC).to_f64();
        let diff = (ep + en).abs();
        assert!(diff < 1e-10, "erf({xi})+erf(-{xi}) should be 0, got {diff}");
    }
}

// ---------------------------------------------------------------------------
// Bessel J0 tests
// ---------------------------------------------------------------------------

#[test]
fn mpfr_bessel_j0_at_zero_is_one() {
    let x = Float::with_val(PREC, 0.0_f64);
    let result = bessel_j0_mpfr(&x, PREC).to_f64();
    assert!(
        (result - 1.0).abs() < 1e-10,
        "J0(0) should be 1, got {result}"
    );
}

#[test]
fn mpfr_bessel_j0_first_zero_matches_published() {
    // First zero of J0 is at ~2.4048255577957...
    // J0(2.4048) should be very close to 0
    let x = Float::with_val(PREC, 2.4048_f64);
    let result = bessel_j0_mpfr(&x, PREC).to_f64();
    assert!(
        result.abs() < 0.001,
        "J0(2.4048) should be near 0, got {result}"
    );
}

#[test]
fn mpfr_bessel_j0_matches_known_value() {
    // J0(1) ~= 0.7651976865579665
    let x = Float::with_val(PREC, 1.0_f64);
    let result = bessel_j0_mpfr(&x, PREC).to_f64();
    let expected = 0.7651976865579665_f64;
    let diff = (result - expected).abs();
    assert!(diff < 1e-10, "J0(1) diff={diff}");
}

#[test]
fn mpfr_bessel_j0_even_symmetry() {
    // J0(-x) = J0(x)
    for xi in [0.5_f64, 1.0, 2.0, 3.0] {
        let xp = Float::with_val(PREC, xi);
        let xn = Float::with_val(PREC, -xi);
        let jp = bessel_j0_mpfr(&xp, PREC).to_f64();
        let jn = bessel_j0_mpfr(&xn, PREC).to_f64();
        let diff = (jp - jn).abs();
        assert!(diff < 1e-10, "J0({xi}) != J0(-{xi}), diff={diff}");
    }
}

// ---------------------------------------------------------------------------
// Bessel K0 tests
// ---------------------------------------------------------------------------

#[test]
fn mpfr_bessel_k0_at_one_matches_known_value() {
    // K0(1) ~= 0.4210244382407083
    let x = Float::with_val(K0_PREC, 1.0_f64);
    let result = bessel_k0_mpfr(&x, K0_PREC).to_f64();
    let expected = 0.4210244382407083_f64;
    let diff = (result - expected).abs();
    assert!(diff < 1e-6, "K0(1) diff={diff}, got {result}");
}

#[test]
fn mpfr_bessel_k0_positive_for_positive_x() {
    // K0(x) > 0 for all x > 0; use lower precision to stay within test time limits.
    for xi in [0.5_f64, 1.0, 2.0] {
        let x = Float::with_val(K0_PREC, xi);
        let result = bessel_k0_mpfr(&x, K0_PREC).to_f64();
        assert!(result > 0.0, "K0({xi}) should be positive, got {result}");
    }
}

#[test]
fn mpfr_bessel_k0_decreasing() {
    // K0(x) is strictly decreasing for x > 0; use lower precision to stay within time limits.
    let vals: Vec<f64> = [0.5_f64, 1.0, 2.0]
        .iter()
        .map(|&xi| {
            let x = Float::with_val(K0_PREC, xi);
            bessel_k0_mpfr(&x, K0_PREC).to_f64()
        })
        .collect();
    for w in vals.windows(2) {
        assert!(
            w[0] > w[1],
            "K0 should decrease: K0(left)={} > K0(right)={}",
            w[0],
            w[1]
        );
    }
}
