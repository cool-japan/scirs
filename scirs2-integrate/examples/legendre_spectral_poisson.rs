use scirs2_integrate::{
    legendre_points, BoundaryCondition, BoundaryConditionType, BoundaryLocation, Domain,
    LegendreSpectralSolver1D, SpectralBasis, SpectralOptions,
};
use std::f64::consts::PI;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Legendre Spectral Method example for the Poisson equation");
    println!("Solving: d²u/dx² = f(x) with Dirichlet boundary conditions");
    println!("Domain: x ∈ [-1, 1]");
    println!("Source term: f(x) = -π² sin(πx)");
    println!("Boundary conditions: u(-1) = u(1) = 0");
    println!("Exact solution: u(x) = sin(πx)");

    // Set up domain
    let domain = Domain::new(
        vec![-1.0..1.0], // Domain [-1, 1]
        vec![65],        // Number of grid points
    )?;

    // Define source term
    let source_term = |x: f64| -> f64 { -PI * PI * (PI * x).sin() };

    // Define boundary conditions
    let bcs = vec![
        // Left boundary (x = -1)
        BoundaryCondition {
            bc_type: BoundaryConditionType::Dirichlet,
            location: BoundaryLocation::Lower,
            dimension: 0,
            value: 0.0,
            coefficients: None,
        },
        // Right boundary (x = 1)
        BoundaryCondition {
            bc_type: BoundaryConditionType::Dirichlet,
            location: BoundaryLocation::Upper,
            dimension: 0,
            value: 0.0,
            coefficients: None,
        },
    ];

    // Solver options
    let options = SpectralOptions {
        basis: SpectralBasis::Legendre,
        num_modes: 65,
        max_iterations: 1,
        tolerance: 1e-10,
        save_convergence_history: false,
        use_real_transform: false, // Not used for Legendre
        use_dealiasing: false,     // Not used for Legendre
        verbose: true,
    };

    // Create the Legendre spectral solver
    let solver =
        LegendreSpectralSolver1D::new(domain, source_term, bcs.clone(), Some(options.clone()))?;

    // Solve the PDE
    println!("\nSolving with Legendre spectral method...");
    let start_time = Instant::now();
    let result = solver.solve()?;
    let solve_time = start_time.elapsed().as_secs_f64();

    // Extract results
    let u = &result.u;
    let grid = &result.grid;
    let coeffs = &result.coefficients;

    println!("Solution computed in {:.4} seconds", solve_time);
    println!("Residual norm: {:.6e}", result.residual_norm);

    // Calculate errors
    let mut max_error: f64 = 0.0;
    let mut l2_error = 0.0;

    for i in 0..grid.len() {
        let x = grid[i];

        // Exact solution: sin(πx)
        let exact = (PI * x).sin();

        // Error
        let error = (u[i] - exact).abs();
        max_error = max_error.max(error);
        l2_error += error * error;
    }

    // Compute L2 error norm
    l2_error = (l2_error / grid.len() as f64).sqrt();

    println!("\nError analysis:");
    println!("  - Maximum error: {:.6e}", max_error);
    println!("  - L2 error norm: {:.6e}", l2_error);

    // Print solution at selected points
    println!("\nSolution at selected points:");
    println!(
        "{:<10} {:<15} {:<15} {:<10}",
        "x", "Numerical", "Exact", "Error"
    );

    let points = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0];

    for &point in &points {
        // Find the closest grid point
        let mut idx = 0;
        let mut min_dist = f64::MAX;

        for (i, &x) in grid.iter().enumerate() {
            let dist = (x - point).abs();
            if dist < min_dist {
                min_dist = dist;
                idx = i;
            }
        }

        let x = grid[idx];
        let numerical = u[idx];
        let exact = (PI * x).sin();
        let error = (numerical - exact).abs();

        println!(
            "{:<10.4} {:<15.8e} {:<15.8e} {:<10.2e}",
            x, numerical, exact, error
        );
    }

    // Examine the distribution of Legendre points
    println!("\nLegendre-Gauss-Lobatto point distribution:");
    println!("Index  x          dx");

    // Get reference Legendre points in [-1, 1]
    let (lg_points, _) = legendre_points(65);

    // Print first few points and differences
    for i in 0..5 {
        let dx = if i > 0 {
            lg_points[i] - lg_points[i - 1]
        } else {
            0.0
        };
        println!("{:<6} {:<10.6} {:<10.6}", i, lg_points[i], dx);
    }

    // Print middle points
    let mid = lg_points.len() / 2;
    println!("...");
    for i in mid - 2..mid + 3 {
        let dx = lg_points[i] - lg_points[i - 1];
        println!("{:<6} {:<10.6} {:<10.6}", i, lg_points[i], dx);
    }

    // Print last few points
    println!("...");
    for i in lg_points.len() - 5..lg_points.len() {
        let dx = lg_points[i] - lg_points[i - 1];
        println!("{:<6} {:<10.6} {:<10.6}", i, lg_points[i], dx);
    }

    // Print largest Legendre coefficients
    println!("\nLargest Legendre coefficients:");
    println!("{:<10} {:<15}", "k", "|coeff[k]|");

    // First, find the largest coefficients
    let mut coeff_indices: Vec<usize> = (0..coeffs.len()).collect();
    coeff_indices.sort_by(|&i, &j| coeffs[j].abs().partial_cmp(&coeffs[i].abs()).unwrap());

    // Print the 5 largest coefficients
    for &i in coeff_indices.iter().take(5) {
        println!("{:<10} {:<15.8e}", i, coeffs[i].abs());
    }

    // Examine spectral convergence
    println!("\nSpectral convergence test:");
    println!("Solving the same problem with different resolutions...");

    for &n in &[17, 33, 65, 129] {
        let domain = Domain::new(vec![-1.0..1.0], vec![n])?;

        let options = SpectralOptions {
            basis: SpectralBasis::Legendre,
            num_modes: n,
            max_iterations: 1,
            tolerance: 1e-10,
            save_convergence_history: false,
            use_real_transform: false,
            use_dealiasing: false,
            verbose: false,
        };

        let solver =
            LegendreSpectralSolver1D::new(domain, source_term, bcs.clone(), Some(options))?;

        let result = solver.solve()?;

        // Calculate L2 error
        let u = &result.u;
        let grid = &result.grid;
        let mut l2_error = 0.0;

        for i in 0..grid.len() {
            let x = grid[i];
            let exact = (PI * x).sin();
            let error = (u[i] - exact).abs();
            l2_error += error * error;
        }

        l2_error = (l2_error / n as f64).sqrt();

        println!("  - N = {}: L2 error = {:.6e}", n, l2_error);
    }

    // Additional example with mixed boundary conditions
    println!("\nBonus example: Solving with mixed boundary conditions");
    println!("Left: Dirichlet u(-1) = 0");
    println!("Right: Neumann u'(1) = 0");

    // Define boundary conditions for mixed case
    let mixed_bcs = vec![
        // Left boundary (x = -1): Dirichlet
        BoundaryCondition {
            bc_type: BoundaryConditionType::Dirichlet,
            location: BoundaryLocation::Lower,
            dimension: 0,
            value: 0.0,
            coefficients: None,
        },
        // Right boundary (x = 1): Neumann
        BoundaryCondition {
            bc_type: BoundaryConditionType::Neumann,
            location: BoundaryLocation::Upper,
            dimension: 0,
            value: 0.0, // du/dx = 0 at x = 1
            coefficients: None,
        },
    ];

    // Create solver with mixed boundary conditions
    let mixed_solver = LegendreSpectralSolver1D::new(
        Domain::new(vec![-1.0..1.0], vec![65])?,
        source_term,
        mixed_bcs,
        Some(options),
    )?;

    // Solve with mixed boundary conditions
    let mixed_result = mixed_solver.solve()?;

    // Print results for mixed boundary conditions
    println!("\nSolution with mixed boundary conditions:");
    println!("{:<10} {:<15}", "x", "u(x)");

    for &point in &points {
        // Find the closest grid point
        let mut idx = 0;
        let mut min_dist = f64::MAX;

        for (i, &x) in mixed_result.grid.iter().enumerate() {
            let dist = (x - point).abs();
            if dist < min_dist {
                min_dist = dist;
                idx = i;
            }
        }

        let x = mixed_result.grid[idx];
        let value = mixed_result.u[idx];

        println!("{:<10.4} {:<15.8e}", x, value);
    }

    // Compare Legendre with Chebyshev
    println!("\nComparison between Legendre and Chebyshev:");
    println!("For this simple problem, both methods should give very similar results.");
    println!("The key differences are:");
    println!("1. Legendre points are more evenly distributed than Chebyshev points");
    println!("2. Legendre methods handle non-periodic boundary conditions more naturally");
    println!("3. Chebyshev methods are sometimes preferred due to FFT availability");
    println!("4. For smooth solutions, both methods exhibit spectral convergence");

    Ok(())
}
