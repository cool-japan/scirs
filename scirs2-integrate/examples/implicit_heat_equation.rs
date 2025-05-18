use ndarray::{s, Array1};
use scirs2_integrate::pde::implicit::{
    BackwardEuler1D, CrankNicolson1D, ImplicitMethod, ImplicitOptions, ImplicitResult,
};
use scirs2_integrate::pde::{BoundaryCondition, BoundaryConditionType, BoundaryLocation, Domain};
use std::f64::consts::PI;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Define domain: x ∈ [0, 1]
    let domain = Domain::new(vec![0.0..1.0], vec![101])?;

    // Define time range: t ∈ [0, 0.5]
    let time_range = [0.0, 0.5];

    // Create boundary conditions (Dirichlet: u(0,t) = 0, u(1,t) = 0)
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

    // Define the diffusion coefficient function: constant diffusion
    let diffusion_coeff = |_x: f64, _t: f64, _u: f64| 1.0;

    // Define the initial condition: u(x, 0) = sin(π*x)
    let initial_condition = |x: f64| (PI * x).sin();

    // Define solver options with different time steps
    let options_cn = ImplicitOptions {
        method: ImplicitMethod::CrankNicolson,
        dt: Some(0.01),
        save_every: Some(10),
        verbose: true,
        ..Default::default()
    };

    let options_be = ImplicitOptions {
        method: ImplicitMethod::BackwardEuler,
        dt: Some(0.01),
        save_every: Some(10),
        verbose: true,
        ..Default::default()
    };

    // Solve using Crank-Nicolson method
    println!("Solving with Crank-Nicolson method...");
    let cn_solver = CrankNicolson1D::new(
        domain.clone(),
        time_range,
        diffusion_coeff,
        initial_condition,
        boundary_conditions.clone(),
        Some(options_cn),
    )?;

    let cn_result = cn_solver.solve()?;

    // Solve using Backward Euler method
    println!("\nSolving with Backward Euler method...");
    let be_solver = BackwardEuler1D::new(
        domain.clone(),
        time_range,
        diffusion_coeff,
        initial_condition,
        boundary_conditions.clone(),
        Some(options_be),
    )?;

    let be_result = be_solver.solve()?;

    // Compare numerical solutions with exact solution
    compare_solutions(&cn_result, &be_result, time_range[1])?;

    println!("\nCompleted!");
    Ok(())
}

/// Analytical solution for the heat equation with given initial and boundary conditions
fn analytical_solution(x: f64, t: f64) -> f64 {
    (PI * x).sin() * (-PI * PI * t).exp()
}

/// Compare numerical solutions with the analytical solution
fn compare_solutions(
    cn_result: &ImplicitResult,
    be_result: &ImplicitResult,
    final_time: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    // Extract solution at final time
    let cn_final = &cn_result.u.last().unwrap();
    let be_final = &be_result.u.last().unwrap();

    // Get spatial grid
    let nx = cn_final.shape()[0];
    let x_grid = Array1::linspace(0.0, 1.0, nx);

    // Calculate analytical solution at final time
    let mut exact = Array1::zeros(nx);
    for i in 0..nx {
        exact[i] = analytical_solution(x_grid[i], final_time);
    }

    // Calculate errors
    let cn_error = (&cn_final.slice(s![.., 0]) - &exact)
        .mapv(|x| x.abs())
        .sum()
        / nx as f64;
    let be_error = (&be_final.slice(s![.., 0]) - &exact)
        .mapv(|x| x.abs())
        .sum()
        / nx as f64;

    println!("\nError analysis:");
    println!("Crank-Nicolson mean absolute error: {:.6e}", cn_error);
    println!("Backward Euler mean absolute error: {:.6e}", be_error);
    println!("Ratio (BE error / CN error): {:.2}", be_error / cn_error);

    // Output some solution values for verification
    println!("\nSolution values at selected points:");
    println!("  x  |  Exact  | Crank-Nicolson | Backward Euler");
    println!("-----|---------|----------------|---------------");

    for i in [0, nx / 4, nx / 2, 3 * nx / 4, nx - 1] {
        let x = x_grid[i];
        println!(
            "{:.2} | {:.6} | {:.6} | {:.6}",
            x,
            exact[i],
            cn_final[[i, 0]],
            be_final[[i, 0]]
        );
    }

    Ok(())
}
