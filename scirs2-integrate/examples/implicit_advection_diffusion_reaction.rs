use scirs2_integrate::pde::implicit::{
    BackwardEuler1D, CrankNicolson1D, ImplicitMethod, ImplicitOptions,
};
use scirs2_integrate::pde::method_of_lines::{MOLOptions, MOLParabolicSolver1D};
use scirs2_integrate::pde::{BoundaryCondition, BoundaryConditionType, BoundaryLocation, Domain};
use std::f64::consts::PI;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Define domain: x ∈ [0, 1]
    let domain = Domain::new(vec![0.0..1.0], vec![101])?;

    // Define time range: t ∈ [0, 1.0]
    let time_range = [0.0, 1.0];

    // Create boundary conditions (Dirichlet on both ends)
    let boundary_conditions = vec![
        BoundaryCondition {
            bc_type: BoundaryConditionType::Dirichlet,
            location: BoundaryLocation::Lower,
            dimension: 0,
            value: 0.0,
            coefficients: None,
        },
        BoundaryCondition {
            bc_type: BoundaryConditionType::Dirichlet,
            location: BoundaryLocation::Upper,
            dimension: 0,
            value: 0.0,
            coefficients: None,
        },
    ];

    // Parameters for stiff system
    let epsilon = 0.01; // Small diffusion coefficient (stiffness parameter)
    let velocity = 1.0; // Advection velocity
    let reaction = 1.0; // Reaction rate

    // Define coefficients for the stiff advection-diffusion-reaction equation
    // ∂u/∂t = ε * ∂²u/∂x² - v * ∂u/∂x - k*u
    let diffusion_coeff = move |_x: f64, _t: f64, _u: f64| epsilon;
    let advection_coeff = move |_x: f64, _t: f64, _u: f64| velocity;
    let reaction_term = move |_x: f64, _t: f64, _u: f64| reaction;

    // Define the initial condition: u(x, 0) = sin(π*x)
    let initial_condition = |x: f64| (PI * x).sin();

    // Solve using explicit Method of Lines (for comparison)
    println!("Solving with Method of Lines (explicit)...");

    // Try with different time steps to show stability limitations
    for dt_exp in [0.01, 0.001, 0.0001] {
        let mol_options = MOLOptions {
            // Use smaller tolerances for more accurate comparison
            atol: 1e-6,
            rtol: 1e-5,
            max_steps: Some((time_range[1] as f64 / dt_exp).ceil() as usize * 2),
            verbose: true,
            ..Default::default()
        };

        let mol_solver = MOLParabolicSolver1D::new(
            domain.clone(),
            time_range,
            move |x, t, u| diffusion_coeff(x, t, u),
            initial_condition,
            boundary_conditions.clone(),
            Some(mol_options),
        )?
        .with_advection(move |x, t, u| advection_coeff(x, t, u))
        .with_reaction(move |x, t, u| reaction_term(x, t, u));

        println!("\nWith dt = {}", dt_exp);
        match mol_solver.solve() {
            Ok(result) => {
                println!("MOL solution successful.");
                println!(
                    "Number of time steps: {}",
                    (time_range[1] / dt_exp) as usize
                );
                println!(
                    "Solution min/max: {:.6}/{:.6}",
                    result.u[0].iter().fold(f64::INFINITY, |a, &b| a.min(b)),
                    result.u[0].iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
                );
            }
            Err(e) => {
                println!("MOL solution failed: {}", e);
            }
        }
    }

    // Define solver options with different time steps for implicit methods
    let dt_implicit = 0.01; // We can use much larger time steps with implicit methods

    let options_cn = ImplicitOptions {
        method: ImplicitMethod::CrankNicolson,
        dt: Some(dt_implicit),
        save_every: Some(10),
        verbose: true,
        ..Default::default()
    };

    let options_be = ImplicitOptions {
        method: ImplicitMethod::BackwardEuler,
        dt: Some(dt_implicit),
        save_every: Some(10),
        verbose: true,
        ..Default::default()
    };

    // Solve using Crank-Nicolson method
    println!(
        "\nSolving with Crank-Nicolson method (dt = {})...",
        dt_implicit
    );
    let cn_solver = CrankNicolson1D::new(
        domain.clone(),
        time_range,
        diffusion_coeff,
        initial_condition,
        boundary_conditions.clone(),
        Some(options_cn),
    )?
    .with_advection(advection_coeff)
    .with_reaction(reaction_term);

    let cn_result = cn_solver.solve()?;

    // Solve using Backward Euler method
    println!(
        "\nSolving with Backward Euler method (dt = {})...",
        dt_implicit
    );
    let be_solver = BackwardEuler1D::new(
        domain.clone(),
        time_range,
        diffusion_coeff,
        initial_condition,
        boundary_conditions.clone(),
        Some(options_be),
    )?
    .with_advection(advection_coeff)
    .with_reaction(reaction_term);

    let be_result = be_solver.solve()?;

    // Output summary
    let default_string = String::new();
    let cn_info = cn_result.info.as_ref().unwrap_or(&default_string);
    let be_info = be_result.info.as_ref().unwrap_or(&default_string);

    println!("\nResults summary:");
    println!("Crank-Nicolson: {}", cn_info);
    println!(
        "  Computation time: {:.4} seconds",
        cn_result.computation_time
    );
    println!(
        "  Solution min/max: {:.6}/{:.6}",
        cn_result
            .u
            .last()
            .unwrap()
            .iter()
            .fold(f64::INFINITY, |a, &b| a.min(b)),
        cn_result
            .u
            .last()
            .unwrap()
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b))
    );

    println!("Backward Euler: {}", be_info);
    println!(
        "  Computation time: {:.4} seconds",
        be_result.computation_time
    );
    println!(
        "  Solution min/max: {:.6}/{:.6}",
        be_result
            .u
            .last()
            .unwrap()
            .iter()
            .fold(f64::INFINITY, |a, &b| a.min(b)),
        be_result
            .u
            .last()
            .unwrap()
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b))
    );

    // Stability analysis
    println!("\nStability Analysis:");
    println!("Grid spacing (dx): {:.6}", 1.0 / 100.0);
    println!("Diffusion coefficient (ε): {:.6}", epsilon);
    println!("Advection velocity (v): {:.6}", velocity);
    println!("Reaction rate (k): {:.6}", reaction);

    // Calculate stability parameters
    let dx = 1.0 / 100.0;
    let diffusion_limit = dx * dx / (2.0 * epsilon); // dt < dx²/(2ε) for explicit methods
    let advection_limit = dx / velocity; // dt < dx/v (CFL condition)
    let reaction_limit = 1.0 / reaction; // dt < 1/k for reaction terms

    println!("Stability limits for explicit methods:");
    println!("  Diffusion limit: dt < {:.6}", diffusion_limit);
    println!("  Advection limit (CFL): dt < {:.6}", advection_limit);
    println!("  Reaction limit: dt < {:.6}", reaction_limit);
    println!(
        "  Combined limit: dt < {:.6}",
        diffusion_limit.min(advection_limit).min(reaction_limit)
    );

    println!("\nImplicit methods used dt = {}, which is:", dt_implicit);
    println!(
        "  {:.1}x larger than diffusion limit",
        dt_implicit / diffusion_limit
    );
    println!(
        "  {:.1}x larger than advection limit",
        dt_implicit / advection_limit
    );
    println!(
        "  {:.1}x larger than reaction limit",
        dt_implicit / reaction_limit
    );

    println!("\nThis demonstrates how implicit methods can handle stiff problems with larger time steps.");
    Ok(())
}
