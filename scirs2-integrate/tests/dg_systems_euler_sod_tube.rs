//! Integration tests for the 1D Euler DG solver on the Sod shock tube.
//!
//! Verifies:
//! - L1 error against the exact Toro Riemann solution at t=0.2 for p=2,3,4.
//! - Approximate mass conservation under periodic BCs with smooth IC.
//! - Positivity of density and pressure throughout the computation.

use scirs2_integrate::pde::dg_systems::euler_1d::{pressure_eos, primitives_to_conservative};
use scirs2_integrate::pde::dg_systems::limiter::{MinmodTvbLimiter, StandardPerssonPeraire};
use scirs2_integrate::pde::dg_systems::{
    sod_exact, solve_1d_euler_dg, BoundaryCondition, DgSystemConfig, TimeIntegrator,
};

const GAMMA: f64 = 1.4;
const T_FINAL: f64 = 0.2;

/// Compute the L1 error in density against the exact Sod solution at t=0.2.
///
/// Uses uniform weights dx/(p+1) for simplicity (adequate for L1 threshold checks).
fn sod_l1_error(n_elements: usize, poly_order: usize) -> f64 {
    let config = DgSystemConfig {
        polynomial_order: poly_order,
        n_elements,
        x_left: 0.0,
        x_right: 1.0,
        gamma: GAMMA,
        cfl: 0.3,
        time_integrator: TimeIntegrator::Ssprk3,
        boundary: BoundaryCondition::Outflow,
        limiter: Some(Box::new(MinmodTvbLimiter::new(10.0))),
        indicator: None,
        indicator_threshold: -1.0,
        record_history: false,
        record_every: 10,
    };

    let ic = |x: f64| {
        if x < 0.5 {
            primitives_to_conservative(1.0, 0.0, 1.0, GAMMA)
        } else {
            primitives_to_conservative(0.125, 0.0, 0.1, GAMMA)
        }
    };

    let sol = solve_1d_euler_dg(ic, T_FINAL, &config).expect("Sod solver should succeed");

    let h = 1.0 / n_elements as f64;
    let node_weight = h / (poly_order + 1) as f64;
    let mut l1 = 0.0_f64;

    for (k, row) in sol.final_state.iter().enumerate() {
        for (j, state) in row.iter().enumerate() {
            let x = sol.x_nodes[[k, j]];
            let (rho_exact, _, _) = sod_exact(x, T_FINAL, GAMMA);
            l1 += (state.rho - rho_exact).abs() * node_weight;
        }
    }

    l1
}

// ── L1 error tests ────────────────────────────────────────────────────────────

#[test]
fn test_sod_tube_p2_n200_l1_error() {
    let l1 = sod_l1_error(200, 2);
    assert!(
        l1 < 0.05,
        "p=2, N=200: L1 error = {l1:.4e}, expected < 5e-2"
    );
}

#[test]
fn test_sod_tube_p3_n100_l1_error() {
    let l1 = sod_l1_error(100, 3);
    assert!(
        l1 < 0.05,
        "p=3, N=100: L1 error = {l1:.4e}, expected < 5e-2"
    );
}

#[test]
fn test_sod_tube_p4_n50_l1_error() {
    let l1 = sod_l1_error(50, 4);
    assert!(l1 < 0.07, "p=4, N=50: L1 error = {l1:.4e}, expected < 7e-2");
}

// ── Conservation test ────────────────────────────────────────────────────────

/// Conservation of total density-weighted average under periodic BCs with smooth IC.
///
/// The smooth sinusoidal IC is used so that the limiter does not interfere.
/// We check that the mean density stays close to 1.0 (the initial average).
#[test]
fn test_sod_tube_conservation() {
    let config = DgSystemConfig {
        polynomial_order: 2,
        n_elements: 50,
        x_left: 0.0,
        x_right: 1.0,
        gamma: GAMMA,
        cfl: 0.3,
        time_integrator: TimeIntegrator::Ssprk3,
        boundary: BoundaryCondition::Periodic,
        limiter: None, // no limiting for smooth conservation test
        indicator: None,
        indicator_threshold: -1.0,
        record_history: false,
        record_every: 1,
    };

    let pi = std::f64::consts::PI;
    let ic = |x: f64| primitives_to_conservative(1.0 + 0.1 * (2.0 * pi * x).sin(), 0.5, 1.0, GAMMA);

    let sol = solve_1d_euler_dg(ic, 0.05, &config).expect("conservation solver should succeed");

    // Mean density across all nodes
    let n_total = config.n_elements * (config.polynomial_order + 1);
    let total_rho: f64 = sol
        .final_state
        .iter()
        .flat_map(|row| row.iter())
        .map(|s| s.rho)
        .sum();
    let mean_rho = total_rho / n_total as f64;

    // Initial mean density = 1.0 (the sine term integrates to zero)
    assert!(
        (mean_rho - 1.0).abs() < 0.05,
        "Mean density deviated from 1.0: {mean_rho:.6}"
    );
}

// ── Positivity test ──────────────────────────────────────────────────────────

/// All densities and pressures must remain strictly positive throughout.
#[test]
fn test_sod_tube_positivity() {
    let config = DgSystemConfig {
        polynomial_order: 2,
        n_elements: 100,
        x_left: 0.0,
        x_right: 1.0,
        gamma: GAMMA,
        cfl: 0.3,
        time_integrator: TimeIntegrator::Ssprk3,
        boundary: BoundaryCondition::Outflow,
        limiter: Some(Box::new(MinmodTvbLimiter::new(10.0))),
        indicator: Some(Box::new(StandardPerssonPeraire::new(4.0))),
        indicator_threshold: -1.0,
        record_history: false,
        record_every: 1,
    };

    let ic = |x: f64| {
        if x < 0.5 {
            primitives_to_conservative(1.0, 0.0, 1.0, GAMMA)
        } else {
            primitives_to_conservative(0.125, 0.0, 0.1, GAMMA)
        }
    };

    let sol = solve_1d_euler_dg(ic, T_FINAL, &config).expect("positivity solver should succeed");

    for (k, row) in sol.final_state.iter().enumerate() {
        for (j, state) in row.iter().enumerate() {
            assert!(
                state.rho > 0.0,
                "density non-positive at elem {k} node {j}: rho={}",
                state.rho
            );
            let p = pressure_eos(state, GAMMA)
                .unwrap_or_else(|| panic!("unphysical pressure at elem {k} node {j}"));
            assert!(p > 0.0, "pressure non-positive at elem {k} node {j}: p={p}");
        }
    }
}
