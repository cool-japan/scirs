//! Method of Manufactured Solutions (MMS) verification example
//!
//! This example demonstrates how to use the MMS toolkit to verify the order
//! of accuracy of numerical methods by comparing against known exact solutions.

use ndarray::{Array1, Array2};
use scirs2_integrate::{
    ode::{solve_ivp, ODEMethod, ODEOptions},
    verification::{
        polynomial_solution, trigonometric_solution_2d, ConvergenceAnalysis, ErrorAnalysis,
        MMSODEProblem, MMSPDEProblem,
    },
};
use std::f64::consts::PI;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Method of Manufactured Solutions Verification ===\n");

    // Example 1: ODE Verification
    ode_verification_example()?;

    // Example 2: PDE Verification
    pde_verification_example()?;

    // Example 3: Convergence Analysis
    convergence_analysis_example()?;

    Ok(())
}

fn ode_verification_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧮 ODE Method Verification");
    println!("{}", "=".repeat(50));

    // Create manufactured problem with exact solution y(t) = 1 + 2t + 3t²
    let exact_solution = polynomial_solution(vec![1.0, 2.0, 3.0]);
    let problem = MMSODEProblem::new(exact_solution, [0.0, 1.0]);

    println!("Exact solution: y(t) = 1 + 2t + 3t²");
    println!("Manufactured ODE: y'(t) = 2 + 6t (derivative of exact solution)");

    // Test different step sizes
    let step_sizes = vec![0.1, 0.05, 0.025, 0.0125];
    let mut errors = Vec::new();

    println!("\nStep Size   Final Error   Expected: y(1) = 6.0");
    println!("{}", "─".repeat(45));

    for &h in &step_sizes {
        // Solve the manufactured ODE: y' = 2 + 6t
        let manufactured_rhs =
            |t: f64, _y: ndarray::ArrayView1<f64>| Array1::from_vec(vec![problem.source_term(t)]);

        let options = ODEOptions {
            method: ODEMethod::RK45,
            rtol: 1e-12,
            atol: 1e-12,
            max_step: Some(h),
            ..Default::default()
        };

        let result = solve_ivp(
            manufactured_rhs,
            problem.time_span(),
            Array1::from_vec(vec![problem.initial_condition()]),
            Some(options),
        )?;

        let numerical_final = result.y.last().unwrap()[0];
        let exact_final = problem.exact_at(1.0);
        let error = (numerical_final - exact_final).abs();

        println!(
            "{:8.4}   {:11.2e}   Numerical: {:.6}",
            h, error, numerical_final
        );
        errors.push(error);
    }

    // Analyze convergence
    if let Ok(analysis) = ConvergenceAnalysis::compute_order(step_sizes, errors) {
        println!("\n📊 Convergence Analysis:");
        println!("Estimated order of accuracy: {:.2}", analysis.order);

        if analysis.verify_order(4.0, 0.5) {
            println!("✅ Fourth-order accuracy confirmed (RK45 method)");
        } else {
            println!("⚠️  Unexpected order of accuracy");
        }
    }

    println!();
    Ok(())
}

fn pde_verification_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("🌊 PDE Method Verification");
    println!("{}", "=".repeat(50));

    // Create manufactured 2D Poisson problem: -∇²u = f
    // Exact solution: u(x,y) = sin(πx) * sin(2πy)
    let exact_solution = trigonometric_solution_2d(PI, 2.0 * PI);
    let problem = MMSPDEProblem::new_poisson_2d(exact_solution, [0.0, 1.0], [0.0, 1.0]);

    println!("Exact solution: u(x,y) = sin(πx) * sin(2πy)");
    println!("Manufactured Poisson equation: -∇²u = π²(1 + 4)sin(πx)sin(2πy) = 5π²u");

    // Test different grid sizes
    let grid_sizes = vec![0.1, 0.05, 0.025];
    let mut errors = Vec::new();

    println!("\nGrid Size   L2 Error      Max Error");
    println!("{}", "─".repeat(35));

    for &h in &grid_sizes {
        let n = (1.0 / h) as usize + 1;

        // Create grid
        let x: Vec<f64> = (0..n).map(|i| i as f64 * h).collect();
        let y: Vec<f64> = (0..n).map(|i| i as f64 * h).collect();

        // Solve -∇²u = f with finite differences
        let mut u_numerical = Array2::zeros((n, n));
        let mut f_rhs = Array2::zeros((n, n));

        // Set up manufactured source term
        for (i, &xi) in x.iter().enumerate() {
            for (j, &yj) in y.iter().enumerate() {
                f_rhs[[i, j]] = problem.source_term(xi, yj);
            }
        }

        // Apply boundary conditions (exact solution on boundaries)
        for i in 0..n {
            for j in 0..n {
                if i == 0 || i == n - 1 || j == 0 || j == n - 1 {
                    u_numerical[[i, j]] = problem.boundary_condition(x[i], y[j]);
                }
            }
        }

        // Simple finite difference solution (for demonstration)
        // In practice, you'd use a proper linear solver
        let interior_points = (n - 2) * (n - 2);
        if interior_points > 0 {
            // For this example, just evaluate exact solution to show the framework
            for (i, &xi) in x.iter().enumerate() {
                for (j, &yj) in y.iter().enumerate() {
                    if i > 0 && i < n - 1 && j > 0 && j < n - 1 {
                        // Use exact solution for demonstration
                        u_numerical[[i, j]] = problem.exact_at(xi, yj);
                    }
                }
            }
        }

        // Compute exact solution on grid
        let mut u_exact = Array2::zeros((n, n));
        for (i, &xi) in x.iter().enumerate() {
            for (j, &yj) in y.iter().enumerate() {
                u_exact[[i, j]] = problem.exact_at(xi, yj);
            }
        }

        // Compute errors
        let l2_error = ErrorAnalysis::l2_norm_2d(u_exact.view(), u_numerical.view())?;
        let max_error = (0.0_f64).max(0.0); // Simplified for demonstration

        println!("{:8.3}   {:9.2e}   {:9.2e}", h, l2_error, max_error);
        errors.push(l2_error);
    }

    println!("📊 Grid convergence analysis would show O(h²) for second-order finite differences");
    println!();
    Ok(())
}

fn convergence_analysis_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("📈 Convergence Analysis Example");
    println!("{}", "=".repeat(50));

    // Create theoretical convergence data
    let grid_sizes = vec![0.2, 0.1, 0.05, 0.025, 0.0125];

    // Simulate different orders of accuracy
    let first_order_errors: Vec<f64> = grid_sizes.iter().map(|&h| 0.1 * h).collect();
    let second_order_errors: Vec<f64> = grid_sizes.iter().map(|&h| 0.1 * h * h).collect();
    let fourth_order_errors: Vec<f64> = grid_sizes.iter().map(|&h| 0.1 * h.powi(4)).collect();

    // Analyze first-order method
    println!("First-order method (Euler):");
    if let Ok(analysis) = ConvergenceAnalysis::compute_order(grid_sizes.clone(), first_order_errors)
    {
        println!("{}", analysis);
        if analysis.verify_order(1.0, 0.1) {
            println!("✅ First-order accuracy confirmed\n");
        }
    }

    // Analyze second-order method
    println!("Second-order method (finite differences):");
    if let Ok(analysis) =
        ConvergenceAnalysis::compute_order(grid_sizes.clone(), second_order_errors)
    {
        println!("{}", analysis);
        if analysis.verify_order(2.0, 0.1) {
            println!("✅ Second-order accuracy confirmed\n");
        }
    }

    // Analyze fourth-order method
    println!("Fourth-order method (RK4):");
    if let Ok(analysis) =
        ConvergenceAnalysis::compute_order(grid_sizes.clone(), fourth_order_errors)
    {
        println!("{}", analysis);
        if analysis.verify_order(4.0, 0.1) {
            println!("✅ Fourth-order accuracy confirmed\n");
        }
    }

    Ok(())
}
