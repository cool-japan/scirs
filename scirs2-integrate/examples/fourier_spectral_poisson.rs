// FIXME: This example has compilation errors - ambiguous type for max()
#![allow(dead_code)]

use scirs2_integrate::{
    BoundaryCondition, BoundaryConditionType, BoundaryLocation, Domain, FourierSpectralSolver1D,
    SpectralBasis, SpectralOptions,
};
use std::f64::consts::PI;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Fourier Spectral Method example for the Poisson equation");
    println!("Solving: d²u/dx² = f(x) with periodic boundary conditions");
    println!("Domain: x ∈ [0, 2π]");
    println!("Source term: f(x) = -sin(x) - 0.5*sin(3x)");
    println!("Exact solution: u(x) = sin(x) + sin(3x)/9");

    // Set up domain
    let domain = Domain::new(
        vec![0.0..2.0 * PI], // Domain [0, 2π]
        vec![128],           // Number of grid points
    )?;

    // Define source term
    let source_term = |x: f64| -> f64 { -x.sin() - 0.5 * (3.0 * x).sin() };

    // Define periodic boundary conditions
    let bcs = vec![BoundaryCondition {
        bc_type: BoundaryConditionType::Periodic,
        location: BoundaryLocation::Lower, // Only need to specify on one side for periodic
        dimension: 0,
        value: 0.0, // Ignored for periodic conditions
        coefficients: None,
    }];

    // Solver options
    let options = SpectralOptions {
        basis: SpectralBasis::Fourier,
        num_modes: 128,
        max_iterations: 1,
        tolerance: 1e-10,
        save_convergence_history: false,
        use_real_transform: true,
        use_dealiasing: false, // Not needed for this simple example
        verbose: true,
    };

    // Create the Fourier spectral solver
    let solver =
        FourierSpectralSolver1D::new(domain, source_term, bcs.clone(), Some(options.clone()))?;

    // Solve the PDE
    println!("\nSolving with Fourier spectral method...");
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
    let mut l2_error: f64 = 0.0;

    for i in 0..grid.len() {
        let x = grid[i];

        // Exact solution: sin(x) + sin(3x)/9
        let exact = x.sin() + (3.0 * x).sin() / 9.0;

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

    for &point in &[0.0, PI / 4.0, PI / 2.0, PI, 3.0 * PI / 2.0, 2.0 * PI - 0.01] {
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
        let exact = x.sin() + (3.0 * x).sin() / 9.0;
        let error = (numerical - exact).abs();

        println!(
            "{:<10.4} {:<15.8e} {:<15.8e} {:<10.2e}",
            x, numerical, exact, error
        );
    }

    // Print largest Fourier coefficients
    println!("\nLargest Fourier coefficients:");
    println!("{:<10} {:<15}", "k", "|u_hat[k]|");

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

    for &n in &[16, 32, 64, 128] {
        let domain = Domain::new(vec![0.0..2.0 * PI], vec![n])?;

        let options = SpectralOptions {
            basis: SpectralBasis::Fourier,
            num_modes: n,
            max_iterations: 1,
            tolerance: 1e-10,
            save_convergence_history: false,
            use_real_transform: true,
            use_dealiasing: false,
            verbose: false,
        };

        let solver = FourierSpectralSolver1D::new(domain, source_term, bcs.clone(), Some(options))?;

        let result = solver.solve()?;

        // Calculate L2 error
        let u = &result.u;
        let grid = &result.grid;
        let mut l2_error = 0.0;

        for i in 0..grid.len() {
            let x = grid[i];
            let exact = x.sin() + (3.0 * x).sin() / 9.0;
            let error = (u[i] - exact).abs();
            l2_error += error * error;
        }

        l2_error = (l2_error / n as f64).sqrt();

        println!("  - N = {}: L2 error = {:.6e}", n, l2_error);
    }

    Ok(())
}
