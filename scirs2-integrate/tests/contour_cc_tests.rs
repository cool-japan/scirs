//! Integration tests for Clenshaw-Curtis with contour deformation.
//!
//! Tests cover:
//!   1. Real-axis adaptive CC convergence for a smooth function
//!   2. Oscillatory Filon-CC integration (high ω)
//!   3. Pole-avoidance via IndentedReal contour (1/z around the origin)
//!   4. Gaussian integral via adaptive CC vs analytic value
//!   5. SemiCircle contour integral of z·dz (analytic = 0)
//!   6. Talbot contour construction and basic shape properties

use std::f64::consts::PI;

use scirs2_integrate::quadrature::contour_cc::{
    adaptive_cc, contour_integrate_cc, filon_cc_oscillatory, talbot_contour, ContourConfig,
    ContourType,
};

// ---------------------------------------------------------------------------
// 1. Real-axis adaptive CC — smooth function
// ---------------------------------------------------------------------------

/// ∫₋₁¹ exp(x) dx = e - 1/e ≈ 2.3504...
#[test]
fn cc_real_axis_exponential() {
    let cfg = ContourConfig {
        tol: 1e-10,
        n_initial: 16,
        max_levels: 8,
        contour_type: ContourType::Real,
    };
    let result = adaptive_cc(|x: f64| x.exp(), -1.0, 1.0, &cfg)
        .expect("adaptive_cc should succeed for exp(x)");

    // Exact value: e^1 - e^(-1)
    let exact = std::f64::consts::E - 1.0 / std::f64::consts::E;
    assert!(
        result.converged,
        "integration of exp(x) should converge; error estimate = {}",
        result.error
    );
    assert!(
        (result.value - exact).abs() < 1e-8,
        "exp(x) integral: got {}, expected {}",
        result.value,
        exact
    );
}

// ---------------------------------------------------------------------------
// 2. Filon-CC for highly oscillatory integral
// ---------------------------------------------------------------------------

/// ∫₋₁¹ cos(100x) dx = 2·sin(100)/100 ≈ -0.01736...
///
/// At n=128 the Filon-CC formula places nodes at the Chebyshev points and
/// applies the oscillatory cosine weight exactly at those nodes; this gives
/// much better accuracy than plain CC for large ω.
#[test]
fn cc_filon_oscillatory_cos100() {
    // Use n=128 for ω=100
    let val = filon_cc_oscillatory(|_x: f64| 1.0, 100.0, -1.0, 1.0, 128);

    // Analytic value: [sin(100x)/100]_{-1}^{1} = 2 sin(100)/100
    let exact = 2.0 * (100.0_f64).sin() / 100.0;

    assert!(
        (val - exact).abs() < 0.05,
        "Filon-CC cos(100x): got {val:.6}, exact {exact:.6}, diff {diff:.2e}",
        val = val,
        exact = exact,
        diff = (val - exact).abs()
    );
}

/// ∫₀^π sin(x) cos(10x) dx = -2/99  (Riemann-Lebesgue exact)
#[test]
fn cc_filon_oscillatory_sin_x_cos_10x() {
    let val = filon_cc_oscillatory(|x: f64| x.sin(), 10.0, 0.0, PI, 64);
    let exact = -2.0_f64 / 99.0;
    assert!(
        (val - exact).abs() < 1e-6,
        "Filon-CC sin(x)cos(10x): got {val:.9}, exact {exact:.9}",
        val = val,
        exact = exact
    );
}

// ---------------------------------------------------------------------------
// 3. Pole avoidance: IndentedReal contour for 1/z around the origin
// ---------------------------------------------------------------------------

/// The contour goes along the real line with a small upper semicircular
/// detour around z=0 to avoid the pole.
///
/// For f(z) = 1/z, the contribution of the upper semicircular arc of radius r
/// from 0 to π is:
///
///   ∫₀^π (1 / r·e^{iθ}) · (i r e^{iθ}) dθ = ∫₀^π i dθ = iπ
///
/// So Im ≈ π. The left/right real-axis segments cancel by antisymmetry (both
/// are of the form ∫ 1/x dx from ±r to ±10r, which cancel when the domains
/// are symmetric).
#[test]
fn cc_pole_avoidance_indented_real_1_over_z() {
    let r = 0.05_f64;
    let n_pts = 32;

    let result = contour_integrate_cc(
        |re, im| {
            // f(z) = 1/z = z̄ / |z|²  (complex reciprocal in (Re, Im) form)
            let denom = re * re + im * im;
            (re / denom, -im / denom)
        },
        &ContourType::IndentedReal {
            indent_radius: r,
            indent_at: vec![0.0],
        },
        n_pts,
    )
    .expect("IndentedReal contour integration should succeed");

    let (re_val, im_val) = result;

    // The Im part of the indentation around z=0 for 1/z is +π.
    // Real axis segments [-10r, -r] and [r, 10r] cancel by antisymmetry.
    assert!(
        (im_val - PI).abs() < 0.1,
        "IndentedReal 1/z: Im should be ≈ π, got {im_val:.6}",
        im_val = im_val
    );

    // Re part should be near 0 (symmetric cancellation on the real line)
    assert!(
        re_val.abs() < 0.5,
        "IndentedReal 1/z: Re should be ≈ 0, got {re_val:.6}",
        re_val = re_val
    );
}

// ---------------------------------------------------------------------------
// 4. Gaussian integral via adaptive CC
// ---------------------------------------------------------------------------

/// ∫₋₁¹ exp(-x²) dx = √π · erf(1) ≈ 1.4936482656248541
#[test]
fn cc_gaussian_integral_matches_analytic() {
    let cfg = ContourConfig {
        tol: 1e-12,
        n_initial: 32,
        max_levels: 8,
        contour_type: ContourType::Real,
    };

    let result = adaptive_cc(|x: f64| (-x * x).exp(), -1.0, 1.0, &cfg)
        .expect("adaptive_cc should succeed for Gaussian");

    // Hardcoded to 16 significant digits via Wolfram Alpha
    let exact = 1.4936482656248541_f64;
    assert!(
        result.converged,
        "Gaussian integral should converge; error estimate = {}",
        result.error
    );
    assert!(
        (result.value - exact).abs() < 1e-10,
        "Gaussian integral: got {:.15}, expected {:.15}",
        result.value,
        exact
    );
}

// ---------------------------------------------------------------------------
// 5. SemiCircle contour: ∫_C dz over full circle = 0
// ---------------------------------------------------------------------------

/// ∫_{-π}^{π} 1 · i r e^{iθ} dθ = i r ∫_{-π}^{π} e^{iθ} dθ = 0
#[test]
fn cc_semicircle_constant_integrand_is_zero() {
    let result = contour_integrate_cc(
        |_re, _im| (1.0, 0.0),
        &ContourType::SemiCircle { radius: 2.0 },
        64,
    )
    .expect("SemiCircle contour integration should succeed");

    let (re_val, im_val) = result;
    assert!(
        re_val.abs() < 1e-10,
        "Re part of ∫1 dz on circle should be 0, got {re_val:.2e}"
    );
    assert!(
        im_val.abs() < 1e-10,
        "Im part of ∫1 dz on circle should be 0, got {im_val:.2e}"
    );
}

/// ∫_C z dz = 0 for any closed contour (holomorphic integrand)
#[test]
fn cc_semicircle_z_dz_is_zero() {
    let result = contour_integrate_cc(
        |re, im| (re, im),
        &ContourType::SemiCircle { radius: 1.0 },
        64,
    )
    .expect("SemiCircle z·dz should succeed");

    let (re_val, im_val) = result;
    assert!(
        re_val.abs() < 1e-8,
        "Re of ∫z dz on circle should be 0, got {re_val:.2e}"
    );
    assert!(
        im_val.abs() < 1e-8,
        "Im of ∫z dz on circle should be 0, got {im_val:.2e}"
    );
}

// ---------------------------------------------------------------------------
// 6. Talbot contour — construction sanity
// ---------------------------------------------------------------------------

/// At θ=0 the Talbot contour reduces to (σ, 0): purely real, no imaginary shift.
#[test]
fn cc_talbot_contour_at_theta_zero() {
    let sigma = 2.5_f64;
    let nu = 0.6_f64;
    let (re, im) = talbot_contour(sigma, nu, 0.0);

    assert!(
        (re - sigma).abs() < 1e-12,
        "Talbot at θ=0: Re should be σ={sigma}, got {re}"
    );
    assert!(im.abs() < 1e-12, "Talbot at θ=0: Im should be 0, got {im}");
}

/// At θ=π/2 the contour is displaced upward: Im = ν²·π/2 > 0.
#[test]
fn cc_talbot_contour_upper_half_at_pi_over_2() {
    let sigma = 1.0_f64;
    let nu = 0.6_f64;
    let theta = PI / 2.0;
    let (_re, im) = talbot_contour(sigma, nu, theta);

    let expected_im = nu * nu * theta;
    assert!(
        im > 0.0,
        "Talbot at θ=π/2 should be in upper half-plane, Im={im}"
    );
    assert!(
        (im - expected_im).abs() < 1e-12,
        "Talbot Im at π/2: got {im}, expected {expected_im}"
    );
}

// ---------------------------------------------------------------------------
// 7. Adaptive CC error tolerance and convergence flag
// ---------------------------------------------------------------------------

/// A constant function integrates exactly (no subdivision needed).
#[test]
fn cc_constant_function_converges_immediately() {
    let cfg = ContourConfig {
        tol: 1e-12,
        n_initial: 8,
        max_levels: 4,
        contour_type: ContourType::Real,
    };
    // Use 1.5 to avoid triggering the clippy::approx_constant lint on 3.14
    let constant = 1.5_f64;
    let result = adaptive_cc(|_x: f64| constant, 0.0, 2.0, &cfg)
        .expect("adaptive_cc should succeed for constant");

    let exact = constant * 2.0;
    assert!(
        result.converged,
        "constant function should converge; error = {}",
        result.error
    );
    assert!(
        (result.value - exact).abs() < 1e-10,
        "constant integral: got {}, expected {}",
        result.value,
        exact
    );
}

/// Invalid bounds (a >= b) must return an error, not panic.
#[test]
fn cc_invalid_bounds_returns_error() {
    let cfg = ContourConfig::default();
    let result = adaptive_cc(|x: f64| x, 1.0, 0.0, &cfg);
    assert!(result.is_err(), "reversed bounds must return Err");
}
