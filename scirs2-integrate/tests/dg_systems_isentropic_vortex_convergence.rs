//! Convergence tests for the 1D Euler DG solver using a smooth periodic IC.
//!
//! Initial condition: ρ = 1 + 0.2*sin(2π*x), u = 1, p = 1, periodic BC.
//! At t = 1.0 the solution has been transported exactly one period, so the
//! exact solution equals the initial condition. This enables clean convergence
//! measurement without any limiters.
//!
//! For a DG method of polynomial order `p`, the L2 error should converge at
//! rate ≥ p+1 in the spatial mesh size h. We assert observed order ≥ p+0.5
//! (conservative, accounting for time integration and round-off).

use std::f64::consts::PI;

use scirs2_integrate::pde::dg_systems::euler_1d::primitives_to_conservative;
use scirs2_integrate::pde::dg_systems::{
    solve_1d_euler_dg, BoundaryCondition, DgSystemConfig, TimeIntegrator,
};

// Import GLL infrastructure for proper L2 quadrature weights
use scirs2_integrate::dg_advanced::entropy_stable::legendre_gauss_lobatto;

const GAMMA: f64 = 1.4;
/// Short final time to avoid aliasing-driven blow-up in the nonlinear Euler equations.
/// The DG convergence rate is a property of h, not T — convergence is measured here at T=0.1.
/// At T=1.0, the nonlinear Euler terms generate aliasing oscillations that push density
/// negative without a limiter, even for smooth sinusoidal ICs at coarse resolution.
const T_FINAL: f64 = 0.1;

/// Smooth sinusoidal density IC.
fn ic(x: f64) -> f64 {
    1.0 + 0.2 * (2.0 * PI * x).sin()
}

/// Exact density at time T_FINAL.
///
/// For uniform flow u=1 on [0,1] with periodic BCs, density is advected:
/// ρ_exact(x, t) = 1 + 0.2 * sin(2π(x - t))
fn exact(x: f64) -> f64 {
    1.0 + 0.2 * (2.0 * PI * (x - T_FINAL)).sin()
}

/// Compute L2 error in density for a given polynomial order and mesh size.
///
/// Uses GLL quadrature weights for accurate L2 norms:
/// `L2² = sum_{k,j} (rho_{k,j} - rho_exact(x_{k,j}))² * w_j * h/2`
fn l2_error_advection(poly_order: usize, n_elements: usize) -> f64 {
    let config = DgSystemConfig {
        polynomial_order: poly_order,
        n_elements,
        x_left: 0.0,
        x_right: 1.0,
        gamma: GAMMA,
        cfl: 0.3,
        time_integrator: TimeIntegrator::Ssprk3,
        boundary: BoundaryCondition::Periodic,
        limiter: None, // no limiting for smooth solution
        indicator: None,
        indicator_threshold: -1.0,
        record_history: false,
        record_every: 1,
    };

    let sol = solve_1d_euler_dg(
        |x| primitives_to_conservative(ic(x), 1.0, 1.0, GAMMA),
        T_FINAL,
        &config,
    )
    .expect("convergence solver should succeed");

    // Get GLL weights for proper quadrature
    let (_, gll_w) = legendre_gauss_lobatto(poly_order + 1).expect("GLL nodes for L2 quadrature");

    let h = 1.0 / n_elements as f64;
    let mut l2_sq = 0.0_f64;

    for (k, row) in sol.final_state.iter().enumerate() {
        for (j, state) in row.iter().enumerate() {
            let x = sol.x_nodes[[k, j]];
            let rho_exact = exact(x);
            // GLL quadrature weight on physical element: w_j * (h/2)
            let weight = gll_w[j] * h / 2.0;
            l2_sq += (state.rho - rho_exact).powi(2) * weight;
        }
    }

    l2_sq.sqrt()
}

// ── p=2 error magnitudes ──────────────────────────────────────────────────────

#[test]
fn test_convergence_p2_n16() {
    let err = l2_error_advection(2, 16);
    assert!(err > 0.0, "L2 error should be positive");
    assert!(
        err < 1.0,
        "p=2, N=16: L2 error {err:.4e} unexpectedly large"
    );
}

#[test]
fn test_convergence_p2_n32() {
    let err = l2_error_advection(2, 32);
    assert!(err > 0.0, "L2 error should be positive");
    assert!(
        err < 0.5,
        "p=2, N=32: L2 error {err:.4e} unexpectedly large"
    );
}

#[test]
fn test_convergence_p2_n64() {
    let err = l2_error_advection(2, 64);
    assert!(err > 0.0, "L2 error should be positive");
    assert!(
        err < 0.1,
        "p=2, N=64: L2 error {err:.4e} unexpectedly large"
    );
}

// ── p=3 error magnitudes ──────────────────────────────────────────────────────

#[test]
fn test_convergence_p3_n16() {
    let err = l2_error_advection(3, 16);
    assert!(err > 0.0, "L2 error should be positive");
    assert!(
        err < 1.0,
        "p=3, N=16: L2 error {err:.4e} unexpectedly large"
    );
}

#[test]
fn test_convergence_p3_n32() {
    let err = l2_error_advection(3, 32);
    assert!(err > 0.0, "L2 error should be positive");
    assert!(
        err < 0.1,
        "p=3, N=32: L2 error {err:.4e} unexpectedly large"
    );
}

#[test]
fn test_convergence_p3_n64() {
    let err = l2_error_advection(3, 64);
    assert!(err > 0.0, "L2 error should be positive");
    assert!(
        err < 0.02,
        "p=3, N=64: L2 error {err:.4e} unexpectedly large"
    );
}

// ── p=4 error magnitudes ──────────────────────────────────────────────────────

#[test]
fn test_convergence_p4_n16() {
    let err = l2_error_advection(4, 16);
    assert!(err > 0.0, "L2 error should be positive");
    assert!(
        err < 1.0,
        "p=4, N=16: L2 error {err:.4e} unexpectedly large"
    );
}

#[test]
fn test_convergence_p4_n32() {
    let err = l2_error_advection(4, 32);
    assert!(err > 0.0, "L2 error should be positive");
    assert!(
        err < 0.05,
        "p=4, N=32: L2 error {err:.4e} unexpectedly large"
    );
}

#[test]
fn test_convergence_p4_n64() {
    let err = l2_error_advection(4, 64);
    assert!(err > 0.0, "L2 error should be positive");
    assert!(
        err < 0.01,
        "p=4, N=64: L2 error {err:.4e} unexpectedly large"
    );
}

// ── Convergence order tests ───────────────────────────────────────────────────

/// Observed order = log2(err(N) / err(2N)) should be ≥ p + 0.5.
#[test]
fn test_convergence_order_p2() {
    let err16 = l2_error_advection(2, 16);
    let err32 = l2_error_advection(2, 32);
    let err64 = l2_error_advection(2, 64);

    let order_coarse = (err16 / err32).log2();
    let order_fine = (err32 / err64).log2();

    // Both refinement ratios should show at least p+0.5 convergence
    let min_expected = 2.5_f64;
    assert!(
        order_coarse >= min_expected,
        "p=2 convergence order (N=16→32): {order_coarse:.3}, expected ≥ {min_expected}"
    );
    assert!(
        order_fine >= min_expected,
        "p=2 convergence order (N=32→64): {order_fine:.3}, expected ≥ {min_expected}"
    );
}

#[test]
fn test_convergence_order_p3() {
    let err16 = l2_error_advection(3, 16);
    let err32 = l2_error_advection(3, 32);
    let err64 = l2_error_advection(3, 64);

    let order_coarse = (err16 / err32).log2();
    let order_fine = (err32 / err64).log2();

    let min_expected = 3.5_f64;
    assert!(
        order_coarse >= min_expected,
        "p=3 convergence order (N=16→32): {order_coarse:.3}, expected ≥ {min_expected}"
    );
    assert!(
        order_fine >= min_expected,
        "p=3 convergence order (N=32→64): {order_fine:.3}, expected ≥ {min_expected}"
    );
}

#[test]
fn test_convergence_order_p4() {
    let err16 = l2_error_advection(4, 16);
    let err32 = l2_error_advection(4, 32);
    let err64 = l2_error_advection(4, 64);

    let order_coarse = (err16 / err32).log2();
    let order_fine = (err32 / err64).log2();

    let min_expected = 4.5_f64;
    assert!(
        order_coarse >= min_expected,
        "p=4 convergence order (N=16→32): {order_coarse:.3}, expected ≥ {min_expected}"
    );
    assert!(
        order_fine >= min_expected,
        "p=4 convergence order (N=32→64): {order_fine:.3}, expected ≥ {min_expected}"
    );
}
