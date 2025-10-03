use scirs2_core::ndarray::{s, Array2};
use scirs2_integrate::pde::implicit::{ImplicitOptions, ADI2D};
use scirs2_integrate::pde::{BoundaryCondition, BoundaryConditionType, BoundaryLocation, Domain};
use std::f64::consts::PI;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Define 2D domain: (x,y) ∈ [0,1]×[0,1]
    let domain = Domain::new(vec![0.0..1.0, 0.0..1.0], vec![51, 51])?;

    // Define time range: t ∈ [0, 0.5]
    let time_range = [0.0, 0.5];

    // Create boundary conditions (Dirichlet on all boundaries)
    let boundary_conditions = vec![
        // x-direction boundaries
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
        // y-direction boundaries
        BoundaryCondition {
            bc_type: BoundaryConditionType::Dirichlet,
            location: BoundaryLocation::Lower,
            dimension: 1,
            value: 0.0,
            coefficients: None,
        },
        BoundaryCondition {
            bc_type: BoundaryConditionType::Dirichlet,
            location: BoundaryLocation::Upper,
            dimension: 1,
            value: 0.0,
            coefficients: None,
        },
    ];

    // Define constant diffusion coefficients for both directions
    let diffusion_x = |_x: f64, _y: f64, _t: f64, _u: f64| 1.0;
    let diffusion_y = |_x: f64, _y: f64, _t: f64, _u: f64| 1.0;

    // Define initial condition: u(x, y, 0) = sin(πx) * sin(πy)
    let initial_condition = |x: f64, y: f64| (PI * x).sin() * (PI * y).sin();

    // Define solver options
    let options = ImplicitOptions {
        dt: Some(0.01),
        save_every: Some(10),
        verbose: true,
        ..Default::default()
    };

    // Create and solve using ADI method
    println!("Solving 2D heat equation using ADI method...");
    let adi_solver = ADI2D::new(
        domain.clone(),
        time_range,
        diffusion_x,
        diffusion_y,
        initial_condition,
        boundary_conditions.clone(),
        Some(options),
    )?;

    let result = adi_solver.solve()?;

    // Compare with analytical solution
    compare_with_analytical(&result, &domain)?;

    println!("\nADI method solution completed successfully.");
    println!("Computation time: {:.4} seconds", result.computation_time);
    println!("Number of time steps: {}", result.num_steps);
    println!("Number of linear solves: {}", result.num_linear_solves);

    if let Some(info) = result.info {
        println!("{info}");
    }

    Ok(())
}

/// Analytical solution for the 2D heat equation with given initial and boundary conditions
#[allow(dead_code)]
fn analytical_solution(x: f64, y: f64, t: f64) -> f64 {
    (PI * x).sin() * (PI * y).sin() * (-2.0 * PI * PI * t).exp()
}

/// Compare numerical and analytical solutions
#[allow(dead_code)]
fn compare_with_analytical(
    result: &scirs2_integrate::pde::implicit::ADIResult,
    domain: &Domain,
) -> Result<(), Box<dyn std::error::Error>> {
    // Get spatial grids
    let x_grid = domain.grid(0)?;
    let y_grid = domain.grid(1)?;
    let nx = x_grid.len();
    let ny = y_grid.len();

    // Get final time
    let final_time = result.t[result.t.len() - 1];

    // Get numerical solution at final time
    let numerical = &result.u[result.u.len() - 1].slice(s![.., .., 0]);

    // Compute analytical solution at final time
    let mut analytical = Array2::zeros((nx, ny));
    for i in 0..nx {
        for j in 0..ny {
            analytical[[i, j]] = analytical_solution(x_grid[i], y_grid[j], final_time);
        }
    }

    // Compute error
    let mut max_error: f64 = 0.0;
    let mut avg_error = 0.0;

    for i in 0..nx {
        for j in 0..ny {
            let error = (numerical[[i, j]] - analytical[[i, j]]).abs();
            max_error = max_error.max(error);
            avg_error += error;
        }
    }
    avg_error /= (nx * ny) as f64;

    println!("\nError analysis at t = {final_time:.4}:");
    println!("Maximum absolute error: {max_error:.6e}");
    println!("Average absolute error: {avg_error:.6e}");

    // Print solution at selected points
    println!("\nSolution values at selected points:");
    println!("  (x,y)  |  Analytical  |  Numerical  |  Error");
    println!("---------|--------------|-------------|--------");

    for &i in &[nx / 4, nx / 2, 3 * nx / 4] {
        for &j in &[ny / 4, ny / 2, 3 * ny / 4] {
            let x = x_grid[i];
            let y = y_grid[j];
            let ana_val = analytical_solution(x, y, final_time);
            let num_val = numerical[[i, j]];
            let error = (ana_val - num_val).abs();

            println!("({x:.2},{y:.2}) | {ana_val:.8} | {num_val:.8} | {error:.2e}");
        }
    }

    // Check solution at different time points
    println!("\nSolution at center point (0.5, 0.5) at different times:");
    println!("  Time   |  Analytical  |  Numerical  |  Error");
    println!("---------|--------------|-------------|--------");

    // Find center indices
    let i_center = nx / 2;
    let j_center = ny / 2;

    // Select a few time points to check
    let num_times = result.t.len();
    let time_indices = [
        0,
        num_times / 4,
        num_times / 2,
        3 * num_times / 4,
        num_times - 1,
    ];

    for &idx in &time_indices {
        let t = result.t[idx];
        let ana_val = analytical_solution(0.5, 0.5, t);
        let num_val = result.u[idx][[i_center, j_center, 0]];
        let error = (ana_val - num_val).abs();

        println!(" {t:.4} | {ana_val:.8} | {num_val:.8} | {error:.2e}");
    }

    Ok(())
}
