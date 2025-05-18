use scirs2_integrate::{
    BoundaryCondition, BoundaryConditionType, BoundaryLocation, Domain, EllipticOptions,
    PoissonSolver2D,
};
use std::f64::consts::PI;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Poisson equation solver example");
    println!("Solving: ∇²u = -2π² sin(πx) sin(πy)");
    println!("Domain: (x,y) ∈ [0, 1] × [0, 1]");
    println!("Boundary conditions: u = 0 on all boundaries");
    println!("Exact solution: u(x, y) = sin(πx) sin(πy)");

    // Set up spatial domain
    let nx = 65; // Number of grid points in x direction
    let ny = 65; // Number of grid points in y direction
    let domain = Domain::new(
        vec![0.0..1.0, 0.0..1.0], // Spatial ranges [0, 1] × [0, 1]
        vec![nx, ny],             // Number of grid points
    )?;

    // Source term: -2π² sin(πx) sin(πy)
    let source_term = |x: f64, y: f64| -> f64 { -2.0 * PI * PI * (PI * x).sin() * (PI * y).sin() };

    // Boundary conditions: u = 0 on all boundaries (Dirichlet)
    let bcs = vec![
        // X-direction boundaries
        BoundaryCondition {
            bc_type: BoundaryConditionType::Dirichlet,
            location: BoundaryLocation::Lower,
            dimension: 0, // x dimension
            value: 0.0,
            coefficients: None,
        },
        BoundaryCondition {
            bc_type: BoundaryConditionType::Dirichlet,
            location: BoundaryLocation::Upper,
            dimension: 0, // x dimension
            value: 0.0,
            coefficients: None,
        },
        // Y-direction boundaries
        BoundaryCondition {
            bc_type: BoundaryConditionType::Dirichlet,
            location: BoundaryLocation::Lower,
            dimension: 1, // y dimension
            value: 0.0,
            coefficients: None,
        },
        BoundaryCondition {
            bc_type: BoundaryConditionType::Dirichlet,
            location: BoundaryLocation::Upper,
            dimension: 1, // y dimension
            value: 0.0,
            coefficients: None,
        },
    ];

    // Solver options
    let options = EllipticOptions {
        max_iterations: 10000,
        tolerance: 1e-6,
        save_convergence_history: true,
        omega: 1.5, // Relaxation parameter (1 < omega < 2 for over-relaxation)
        verbose: true,
        fd_scheme: scirs2_integrate::FiniteDifferenceScheme::CentralDifference,
    };

    // Create the Poisson solver
    let poisson_solver = PoissonSolver2D::new(domain, source_term, bcs, Some(options))?;

    // Compare iterative (SOR) and direct methods
    println!("\nSolving with SOR method...");
    let start_time = Instant::now();
    let sor_result = poisson_solver.solve_sor()?;
    let sor_time = start_time.elapsed().as_secs_f64();

    println!("\nSolving with direct method...");
    let start_time = Instant::now();
    let direct_result = poisson_solver.solve_direct()?;
    let direct_time = start_time.elapsed().as_secs_f64();

    // Extract solution
    let u_sor = &sor_result.u;
    let u_direct = &direct_result.u;

    println!("\nResults:");
    println!("SOR method:");
    println!("  - Iterations: {}", sor_result.num_iterations);
    println!("  - Time: {:.4} seconds", sor_time);
    println!("  - Residual norm: {:.6e}", sor_result.residual_norm);

    println!("Direct method:");
    println!("  - Time: {:.4} seconds", direct_time);
    println!("  - Residual norm: {:.6e}", direct_result.residual_norm);

    // Calculate errors
    let dx = 1.0 / (nx as f64 - 1.0);
    let dy = 1.0 / (ny as f64 - 1.0);

    let mut max_error_sor: f64 = 0.0;
    let mut max_error_direct: f64 = 0.0;
    let mut l2_error_sor = 0.0;
    let mut l2_error_direct = 0.0;

    for j in 0..ny {
        for i in 0..nx {
            let x = i as f64 * dx;
            let y = j as f64 * dy;

            // Exact solution: sin(πx) sin(πy)
            let exact = (PI * x).sin() * (PI * y).sin();

            // Errors
            let error_sor = (u_sor[[j, i]] - exact).abs();
            let error_direct = (u_direct[[j, i]] - exact).abs();

            max_error_sor = max_error_sor.max(error_sor);
            max_error_direct = max_error_direct.max(error_direct);

            l2_error_sor += error_sor * error_sor;
            l2_error_direct += error_direct * error_direct;
        }
    }

    // Compute L2 error norms
    l2_error_sor = (l2_error_sor * dx * dy).sqrt();
    l2_error_direct = (l2_error_direct * dx * dy).sqrt();

    println!("\nError analysis:");
    println!("SOR method:");
    println!("  - Maximum error: {:.6e}", max_error_sor);
    println!("  - L2 error norm: {:.6e}", l2_error_sor);

    println!("Direct method:");
    println!("  - Maximum error: {:.6e}", max_error_direct);
    println!("  - L2 error norm: {:.6e}", l2_error_direct);

    // Print solution at selected points
    println!("\nSolution values at selected points:");
    println!(
        "{:<10} {:<10} {:<15} {:<15} {:<15}",
        "x", "y", "Numerical (SOR)", "Numerical (Direct)", "Exact"
    );

    for &x in &[0.25, 0.5, 0.75] {
        for &y in &[0.25, 0.5, 0.75] {
            let i = (x / dx).round() as usize;
            let j = (y / dy).round() as usize;

            let numerical_sor = u_sor[[j, i]];
            let numerical_direct = u_direct[[j, i]];
            let exact = (PI * x).sin() * (PI * y).sin();

            println!(
                "{:<10.4} {:<10.4} {:<15.8e} {:<15.8e} {:<15.8e}",
                x, y, numerical_sor, numerical_direct, exact
            );
        }
    }

    // Analyze convergence history
    if let Some(history) = sor_result.convergence_history {
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

        // Print residual norm every 1000 iterations
        println!("\nResidual norm progression:");
        for (i, residual) in history.iter().enumerate() {
            if i % 1000 == 0 || i == history.len() - 1 {
                println!("  - Iteration {}: {:.6e}", i, residual);
            }
        }
    }

    Ok(())
}
