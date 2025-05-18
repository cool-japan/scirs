use ndarray::Array1;
use scirs2_integrate::{
    BoundaryCondition, BoundaryConditionType, BoundaryLocation, Domain, MOLOptions,
    MOLParabolicSolver3D, PDESolution,
};
use std::f64::consts::PI;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("3D Heat Equation Example");
    println!("Solving: ∂u/∂t = D(∂²u/∂x² + ∂²u/∂y² + ∂²u/∂z²)");
    println!("Domain: [0, 1] × [0, 1] × [0, 1]");
    println!("Initial condition: u(x, y, z, 0) = sin(πx) * sin(πy) * sin(πz)");
    println!("Boundary conditions: u = 0 at all boundaries");
    println!("Exact solution: u(x, y, z, t) = exp(-3π²Dt) * sin(πx) * sin(πy) * sin(πz)");

    // Diffusion coefficient
    let d = 0.1;

    // Set up domain
    let domain = Domain::new(
        vec![0.0..1.0, 0.0..1.0, 0.0..1.0], // 3D domain [0,1]×[0,1]×[0,1]
        vec![21, 21, 21],                   // Number of grid points in each dimension
    )?;

    // Time range
    let t_span = [0.0, 1.0];

    // Diffusion coefficients (constant for heat equation)
    let diffusion_x = move |_x: f64, _y: f64, _z: f64, _t: f64, _u: f64| d;
    let diffusion_y = move |_x: f64, _y: f64, _z: f64, _t: f64, _u: f64| d;
    let diffusion_z = move |_x: f64, _y: f64, _z: f64, _t: f64, _u: f64| d;

    // Initial condition: u(x,y,z,0) = sin(πx) * sin(πy) * sin(πz)
    let initial_condition =
        |x: f64, y: f64, z: f64| (PI * x).sin() * (PI * y).sin() * (PI * z).sin();

    // Boundary conditions (Dirichlet conditions with u = 0 at all boundaries)
    let boundary_conditions = vec![
        // X-direction boundaries
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
        // Y-direction boundaries
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
        // Z-direction boundaries
        BoundaryCondition {
            bc_type: BoundaryConditionType::Dirichlet,
            location: BoundaryLocation::Lower,
            dimension: 2,
            value: 0.0,
            coefficients: None,
        },
        BoundaryCondition {
            bc_type: BoundaryConditionType::Dirichlet,
            location: BoundaryLocation::Upper,
            dimension: 2,
            value: 0.0,
            coefficients: None,
        },
    ];

    // Solver options
    let options = MOLOptions {
        ode_method: scirs2_integrate::ODEMethod::Bdf,
        atol: 1e-6,
        rtol: 1e-3,
        max_steps: Some(5000),
        verbose: true,
    };

    // Create the PDE solver
    let solver = MOLParabolicSolver3D::new(
        domain,
        t_span,
        diffusion_x,
        diffusion_y,
        diffusion_z,
        initial_condition,
        boundary_conditions.clone(),
        Some(options),
    )?;

    // Solve the PDE
    println!("\nSolving 3D heat equation...");
    let start_time = Instant::now();
    let mol_result = solver.solve()?;
    let solve_time = start_time.elapsed().as_secs_f64();

    println!("Solution computed in {:.4} seconds", solve_time);
    println!("Time steps: {}", mol_result.t.len());
    println!(
        "ODE solver info: {}",
        mol_result.ode_info.as_ref().unwrap_or(&String::new())
    );

    // Convert to PDESolution
    let pde_solution: PDESolution<f64> = mol_result.into();

    // Get solution information
    let t_values = &pde_solution.grids[0];
    let z_grid = &pde_solution.grids[1];
    let y_grid = &pde_solution.grids[2];
    let x_grid = &pde_solution.grids[3];

    let nx = x_grid.len();
    let ny = y_grid.len();
    let nz = z_grid.len();
    let nt = t_values.len();

    println!("\nSolution grid dimensions:");
    println!("  - nx = {}, ny = {}, nz = {}, nt = {}", nx, ny, nz, nt);

    // Extract solution at the final time step
    let final_time_idx = nt - 1;
    let final_time = t_values[final_time_idx];
    let final_solution = &pde_solution.values[final_time_idx];

    // Calculate error compared to the exact solution
    println!("\nError analysis at t = {:.4}:", final_time);

    // Function for the exact solution
    let exact_solution = |x: f64, y: f64, z: f64, t: f64| -> f64 {
        (-3.0 * PI * PI * d * t).exp() * (PI * x).sin() * (PI * y).sin() * (PI * z).sin()
    };

    // Calculate error at the center of the domain and a few other points
    let test_points = [
        (0.5, 0.5, 0.5),    // Center
        (0.25, 0.25, 0.25), // Near (0,0,0)
        (0.75, 0.75, 0.75), // Near (1,1,1)
        (0.25, 0.5, 0.75),  // Mixed point
    ];

    println!("\nComparison at selected points (t = {:.4}):", final_time);
    println!(
        "{:<6} {:<6} {:<6} {:<15} {:<15} {:<10}",
        "x", "y", "z", "Numerical", "Exact", "Rel. Error"
    );

    let mut max_rel_error: f64 = 0.0;
    let mut l2_error = 0.0;
    let mut l2_norm = 0.0;

    for &(test_x, test_y, test_z) in &test_points {
        // Find the closest grid indices
        let i_x = find_closest_index(x_grid, test_x);
        let i_y = find_closest_index(y_grid, test_y);
        let i_z = find_closest_index(z_grid, test_z);

        let x = x_grid[i_x];
        let y = y_grid[i_y];
        let z = z_grid[i_z];

        let flat_idx = i_z * ny * nx + i_y * nx + i_x;
        let numerical = final_solution[[flat_idx, 0]];
        let exact = exact_solution(x, y, z, final_time);
        let rel_error = if exact.abs() > 1e-10 {
            (numerical - exact).abs() / exact.abs()
        } else {
            0.0
        };

        println!(
            "{:<6.2} {:<6.2} {:<6.2} {:<15.8e} {:<15.8e} {:<10.2e}",
            x, y, z, numerical, exact, rel_error
        );

        max_rel_error = max_rel_error.max(rel_error);
    }

    // Calculate overall error metrics
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let x = x_grid[i];
                let y = y_grid[j];
                let z = z_grid[k];

                let flat_idx = k * ny * nx + j * nx + i;
                let numerical = final_solution[[flat_idx, 0]];
                let exact = exact_solution(x, y, z, final_time);

                let error = (numerical - exact).powi(2);
                l2_error += error;
                l2_norm += exact.powi(2);
            }
        }
    }

    // Compute normalized L2 error
    l2_error = (l2_error / l2_norm).sqrt();

    println!("\nOverall error metrics:");
    println!("  - Maximum relative error: {:.6e}", max_rel_error);
    println!("  - Normalized L2 error: {:.6e}", l2_error);

    // Plot temperature evolution at the center of the domain
    let center_x = find_closest_index(x_grid, 0.5);
    let center_y = find_closest_index(y_grid, 0.5);
    let center_z = find_closest_index(z_grid, 0.5);

    println!("\nTemperature evolution at the center of the domain (x, y, z) = (0.5, 0.5, 0.5):");
    println!(
        "{:<10} {:<15} {:<15} {:<10}",
        "Time", "Numerical", "Exact", "Rel. Error"
    );

    for (t_idx, &t) in t_values
        .iter()
        .enumerate()
        .filter(|(i, _)| i % (nt / 10).max(1) == 0)
    {
        let flat_idx = center_z * ny * nx + center_y * nx + center_x;
        let u_numerical = pde_solution.values[t_idx][[flat_idx, 0]];
        let u_exact = exact_solution(0.5, 0.5, 0.5, t);
        let rel_error = if u_exact.abs() > 1e-10 {
            (u_numerical - u_exact).abs() / u_exact.abs()
        } else {
            0.0
        };

        println!(
            "{:<10.4} {:<15.8e} {:<15.8e} {:<10.2e}",
            t, u_numerical, u_exact, rel_error
        );
    }

    // Plot convergence for different grid resolutions
    println!("\nGrid resolution study:");
    let resolutions = [9, 17, 33];

    for &n in &resolutions {
        // Set up domain with different resolution
        let domain = Domain::new(vec![0.0..1.0, 0.0..1.0, 0.0..1.0], vec![n, n, n])?;

        let solver = MOLParabolicSolver3D::new(
            domain,
            t_span,
            diffusion_x,
            diffusion_y,
            diffusion_z,
            initial_condition,
            boundary_conditions.clone(),
            None,
        )?;

        let mol_result = solver.solve()?;
        let pde_solution: PDESolution<f64> = mol_result.into();

        // Calculate error
        let t_values = &pde_solution.grids[0];
        let z_grid = &pde_solution.grids[1];
        let y_grid = &pde_solution.grids[2];
        let x_grid = &pde_solution.grids[3];

        let nx = x_grid.len();
        let ny = y_grid.len();
        let nz = z_grid.len();
        let nt = t_values.len();

        // Extract solution at the final time step
        let final_time_idx = nt - 1;
        let final_time = t_values[final_time_idx];
        let final_solution = &pde_solution.values[final_time_idx];

        // Calculate error
        let mut l2_error = 0.0;
        let mut l2_norm = 0.0;

        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let x = x_grid[i];
                    let y = y_grid[j];
                    let z = z_grid[k];

                    let flat_idx = k * ny * nx + j * nx + i;
                    let numerical = final_solution[[flat_idx, 0]];
                    let exact = exact_solution(x, y, z, final_time);

                    let error = (numerical - exact).powi(2);
                    l2_error += error;
                    l2_norm += exact.powi(2);
                }
            }
        }

        // Compute normalized L2 error
        l2_error = (l2_error / l2_norm).sqrt();

        println!(
            "  - Resolution {}x{}x{}: L2 error = {:.6e}, dx = {:.4}",
            n,
            n,
            n,
            l2_error,
            1.0 / (n as f64 - 1.0)
        );
    }

    Ok(())
}

// Helper function to find the index of the closest value in an array
fn find_closest_index(arr: &Array1<f64>, value: f64) -> usize {
    let mut min_dist = f64::MAX;
    let mut min_idx = 0;

    for (i, &x) in arr.iter().enumerate() {
        let dist = (x - value).abs();
        if dist < min_dist {
            min_dist = dist;
            min_idx = i;
        }
    }

    min_idx
}
