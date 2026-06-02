//! Sod shock tube solved with the 1D compressible Euler DG method.
//!
//! Demonstrates p=2, N=200 elements with SSPRK3 time integration and
//! TVB-modified minmod slope limiter (M=10). Reports L1 error in density
//! against the exact Toro Riemann solution at t=0.2.
//!
//! # Running
//!
//! ```bash
//! cargo run --example sod_shock_tube_dg --features all
//! ```

use scirs2_integrate::pde::dg_systems::euler_1d::{
    conservative_to_primitives, primitives_to_conservative,
};
use scirs2_integrate::pde::dg_systems::limiter::MinmodTvbLimiter;
use scirs2_integrate::pde::dg_systems::{
    sod_exact, solve_1d_euler_dg, BoundaryCondition, DgSystemConfig, TimeIntegrator,
};

fn main() {
    let gamma = 1.4_f64;
    let t_final = 0.2_f64;
    let n_elements = 200_usize;
    let poly_order = 2_usize;

    println!("Sod Shock Tube — 1D Euler DG");
    println!("  polynomial order p = {poly_order}");
    println!("  elements          N = {n_elements}");
    println!("  t_final             = {t_final}");
    println!("  time integrator     = SSPRK3");
    println!("  limiter             = MinmodTVB(M=10)");
    println!();

    let config = DgSystemConfig {
        polynomial_order: poly_order,
        n_elements,
        x_left: 0.0,
        x_right: 1.0,
        gamma,
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
            primitives_to_conservative(1.0, 0.0, 1.0, gamma)
        } else {
            primitives_to_conservative(0.125, 0.0, 0.1, gamma)
        }
    };

    let sol = solve_1d_euler_dg(ic, t_final, &config).expect("Sod shock tube solver failed");

    // Compute L1 error in density
    let h = 1.0 / n_elements as f64;
    let node_weight = h / (poly_order + 1) as f64;
    let mut l1_rho = 0.0_f64;
    let mut l1_u = 0.0_f64;
    let mut l1_p = 0.0_f64;

    for (k, row) in sol.final_state.iter().enumerate() {
        for (j, state) in row.iter().enumerate() {
            let x = sol.x_nodes[[k, j]];
            let (rho_exact, u_exact, p_exact) = sod_exact(x, t_final, gamma);
            let (rho, u, p) =
                conservative_to_primitives(state, gamma).unwrap_or((state.rho, 0.0, 0.0));
            l1_rho += (rho - rho_exact).abs() * node_weight;
            l1_u += (u - u_exact).abs() * node_weight;
            l1_p += (p - p_exact).abs() * node_weight;
        }
    }

    println!("L1 errors vs. exact Toro Riemann solution:");
    println!("  density   ρ: {l1_rho:.4e}");
    println!("  velocity  u: {l1_u:.4e}");
    println!("  pressure  p: {l1_p:.4e}");
    println!();

    if l1_rho < 0.05 {
        println!("PASS: density L1 error {l1_rho:.4e} < 5e-2");
    } else {
        println!("NOTE: density L1 error {l1_rho:.4e} >= 5e-2");
    }

    // Print a short solution profile across a few representative points
    println!();
    println!(
        "{:>8} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "x", "rho_num", "rho_ex", "u_num", "u_ex", "p_num", "p_ex"
    );
    println!("{:-<72}", "");

    let n_print = 20_usize;
    let stride = (n_elements * (poly_order + 1)) / n_print;
    let mut printed = 0;
    'outer: for (k, row) in sol.final_state.iter().enumerate() {
        for (j, state) in row.iter().enumerate() {
            let flat_idx = k * (poly_order + 1) + j;
            if !flat_idx.is_multiple_of(stride) {
                continue;
            }
            let x = sol.x_nodes[[k, j]];
            let (rho_exact, u_exact, p_exact) = sod_exact(x, t_final, gamma);
            let (rho, u, p) =
                conservative_to_primitives(state, gamma).unwrap_or((state.rho, 0.0, 0.0));
            println!(
                "{:8.4} {:10.5} {:10.5} {:10.5} {:10.5} {:10.5} {:10.5}",
                x, rho, rho_exact, u, u_exact, p, p_exact
            );
            printed += 1;
            if printed >= n_print {
                break 'outer;
            }
        }
    }
}
