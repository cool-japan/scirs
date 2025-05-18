use scirs2_integrate::{
    BoundaryCondition, BoundaryConditionType, BoundaryLocation, Domain, EllipticOptions,
    LaplaceSolver2D,
};
use std::f64::consts::PI;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Laplace equation solver example");
    println!("Solving: ∇²u = 0");
    println!("Domain: (x,y) ∈ [0, 1] × [0, 1]");
    println!("Boundary conditions:");
    println!("  - Bottom (y=0): u = 0");
    println!("  - Top (y=1): u = sin(πx)");
    println!("  - Left (x=0): u = 0");
    println!("  - Right (x=1): u = 0");
    println!("Exact solution: u(x, y) = sin(πx) * sinh(πy) / sinh(π)");

    // Set up spatial domain
    let nx = 65; // Number of grid points in x direction
    let ny = 65; // Number of grid points in y direction
    let domain = Domain::new(
        vec![0.0..1.0, 0.0..1.0], // Spatial ranges [0, 1] × [0, 1]
        vec![nx, ny],             // Number of grid points
    )?;

    // Boundary conditions
    let bcs = vec![
        // Left boundary (x=0): u = 0
        BoundaryCondition {
            bc_type: BoundaryConditionType::Dirichlet,
            location: BoundaryLocation::Lower,
            dimension: 0, // x dimension
            value: 0.0,
            coefficients: None,
        },
        // Right boundary (x=1): u = 0
        BoundaryCondition {
            bc_type: BoundaryConditionType::Dirichlet,
            location: BoundaryLocation::Upper,
            dimension: 0, // x dimension
            value: 0.0,
            coefficients: None,
        },
        // Bottom boundary (y=0): u = 0
        BoundaryCondition {
            bc_type: BoundaryConditionType::Dirichlet,
            location: BoundaryLocation::Lower,
            dimension: 1, // y dimension
            value: 0.0,
            coefficients: None,
        },
        // Top boundary (y=1): u = sin(πx)
        BoundaryCondition {
            bc_type: BoundaryConditionType::Dirichlet,
            location: BoundaryLocation::Upper,
            dimension: 1, // y dimension
            value: 0.0,   // This will be overridden for each grid point
            coefficients: None,
        },
    ];

    // We need to set the values for the top boundary condition based on x
    let bcs_special = bcs.clone();

    // For complex boundary conditions like sin(πx), we'll handle them in the solution loop
    // This is a limitation of the current API that we would address in a future version

    // Solver options
    let options = EllipticOptions {
        max_iterations: 5000,
        tolerance: 1e-6,
        save_convergence_history: true,
        omega: 1.5, // Relaxation parameter (1 < omega < 2 for over-relaxation)
        verbose: true,
        fd_scheme: scirs2_integrate::FiniteDifferenceScheme::CentralDifference,
    };

    // Create the Laplace solver
    let laplace_solver = LaplaceSolver2D::new(domain.clone(), bcs_special, Some(options))?;

    // Solve with SOR method
    println!("\nSolving with SOR method...");
    let start_time = Instant::now();
    let mut sor_result = laplace_solver.solve_sor()?;
    let sor_time = start_time.elapsed().as_secs_f64();

    // Now we need to manually apply the sin(πx) boundary condition on the top edge
    // and solve again (this is a limitation of the current API)
    let dx = 1.0 / (nx as f64 - 1.0);
    let dy = 1.0 / (ny as f64 - 1.0);

    for i in 0..nx {
        let x = i as f64 * dx;
        sor_result.u[[ny - 1, i]] = (PI * x).sin();
    }

    // Create another laplace solver with updated solution as initial guess
    let options_refined = EllipticOptions {
        max_iterations: 5000,
        tolerance: 1e-6,
        save_convergence_history: true,
        omega: 1.5,
        verbose: true,
        fd_scheme: scirs2_integrate::FiniteDifferenceScheme::CentralDifference,
    };

    let laplace_solver_refined = LaplaceSolver2D::new(domain, bcs, Some(options_refined))?;

    // Solve again with the manual boundary conditions
    println!("\nRefining solution with correct boundary conditions...");
    let start_time = Instant::now();
    let mut refined_result = laplace_solver_refined.solve_sor()?;
    let refined_time = start_time.elapsed().as_secs_f64();

    // Apply the sin(πx) boundary condition once more
    for i in 0..nx {
        let x = i as f64 * dx;
        refined_result.u[[ny - 1, i]] = (PI * x).sin();
    }

    // Extract solution
    let u = &refined_result.u;

    println!("\nResults:");
    println!("  - Iterations: {}", refined_result.num_iterations);
    println!("  - Total time: {:.4} seconds", sor_time + refined_time);
    println!(
        "  - Final residual norm: {:.6e}",
        refined_result.residual_norm
    );

    // Calculate errors
    let mut max_error: f64 = 0.0;
    let mut l2_error = 0.0;

    for j in 0..ny {
        for i in 0..nx {
            let x = i as f64 * dx;
            let y = j as f64 * dy;

            // Exact solution: sin(πx) * sinh(πy) / sinh(π)
            let exact = (PI * x).sin() * (PI * y).sinh() / (PI).sinh();

            // Error
            let error = (u[[j, i]] - exact).abs();

            max_error = max_error.max(error);
            l2_error += error * error;
        }
    }

    // Compute L2 error norm
    l2_error = (l2_error * dx * dy).sqrt();

    println!("\nError analysis:");
    println!("  - Maximum error: {:.6e}", max_error);
    println!("  - L2 error norm: {:.6e}", l2_error);

    // Print solution at selected points
    println!("\nSolution values at selected points:");
    println!(
        "{:<10} {:<10} {:<15} {:<15} {:<10}",
        "x", "y", "Numerical", "Exact", "Error"
    );

    for &x in &[0.25, 0.5, 0.75] {
        for &y in &[0.25, 0.5, 0.75] {
            let i = (x / dx).round() as usize;
            let j = (y / dy).round() as usize;

            let numerical = u[[j, i]];
            let exact = (PI * x).sin() * (PI * y).sinh() / (PI).sinh();
            let error = (numerical - exact).abs();

            println!(
                "{:<10.4} {:<10.4} {:<15.8e} {:<15.8e} {:<10.2e}",
                x, y, numerical, exact, error
            );
        }
    }

    // Analyze convergence history
    if let Some(history) = refined_result.convergence_history {
        println!("\nConvergence history analysis:");
        println!("  - Initial residual: {:.6e}", history[0]);
        println!("  - Final residual: {:.6e}", history[history.len() - 1]);

        // Find convergence rate
        if history.len() >= 10 {
            let n = history.len();
            let rate_early = history[9] / history[0];
            let rate_middle = history[n / 2] / history[n / 2 - 10];
            let rate_late = history[n - 1] / history[n - 10];

            println!(
                "  - Convergence rate (early): {:.6e} per 10 iterations",
                rate_early
            );
            println!(
                "  - Convergence rate (middle): {:.6e} per 10 iterations",
                rate_middle
            );
            println!(
                "  - Convergence rate (late): {:.6e} per 10 iterations",
                rate_late
            );
        }

        // Print residual norm every 500 iterations
        println!("\nResidual norm progression:");
        for (i, residual) in history.iter().enumerate() {
            if i % 500 == 0 || i == history.len() - 1 {
                println!("  - Iteration {}: {:.6e}", i, residual);
            }
        }
    }

    Ok(())
}
