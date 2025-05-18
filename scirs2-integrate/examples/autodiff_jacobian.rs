//! Automatic Differentiation for Jacobian Computation
//!
//! This example demonstrates the use of automatic differentiation
//! to compute exact Jacobian matrices for ODE systems, which can
//! significantly improve performance and accuracy for stiff problems.
//!
//! Note: This example requires the 'autodiff' feature to be enabled.
//! Build with: cargo run --example autodiff_jacobian --features autodiff

use ndarray::{array, ArrayView1};
use scirs2_integrate::error::IntegrateResult;
use scirs2_integrate::ode::utils::jacobian::{is_autodiff_available, JacobianStrategy};
use scirs2_integrate::ode::{solve_ivp, ODEMethod, ODEOptions, ODEResult};
use std::time::Instant;

fn main() -> IntegrateResult<()> {
    // Check if autodiff feature is available
    let autodiff_available = is_autodiff_available();

    println!("Autodiff feature available: {}", autodiff_available);

    if !autodiff_available {
        println!("\nTo enable autodiff, rebuild with: cargo run --example autodiff_jacobian --features autodiff");
        println!(
            "This example will continue, but will use finite differences for Jacobian computation."
        );
    }

    // Define a stiff ODE system: Van der Pol oscillator with μ=1000
    // y'₁ = y₂
    // y'₂ = 1000·(1 - y₁²)·y₂ - y₁
    let mu = 1000.0;
    let f = move |_t: f64, y: ArrayView1<f64>| array![y[1], mu * (1.0 - y[0] * y[0]) * y[1] - y[0]];

    // Initial conditions
    let y0 = array![2.0, 0.0];

    // Time span
    let t_span = [0.0, 3000.0];

    println!("\nSolving stiff Van der Pol oscillator (μ=1000)...");
    println!("This is a classic stiff ODE that requires accurate Jacobian computation.");

    // Compare different Jacobian strategies
    let strategies = vec![
        (JacobianStrategy::FiniteDifference, "Finite Difference"),
        (JacobianStrategy::AutoDiff, "Automatic Differentiation"),
        (
            JacobianStrategy::Adaptive,
            "Adaptive (Auto-selects best method)",
        ),
    ];

    let mut results = Vec::new();

    for (strategy, name) in strategies {
        // Skip AutoDiff strategy if not available
        if strategy == JacobianStrategy::AutoDiff && !autodiff_available {
            println!("\nSkipping {} strategy (feature not enabled)", name);
            continue;
        }

        println!("\nSolving with {} strategy...", name);

        // BDF method is well-suited for stiff problems
        let options = ODEOptions {
            method: ODEMethod::Bdf,
            rtol: 1e-6,
            atol: 1e-8,
            max_steps: 10000,
            jacobian_strategy: Some(strategy),
            ..Default::default()
        };

        // Solve with timing
        let start_time = Instant::now();
        let result = solve_ivp(f.clone(), t_span, y0.clone(), Some(options))?;
        let elapsed = start_time.elapsed();

        // Print statistics
        println!("  Completed in: {:.2?}", elapsed);
        println!("  Function evaluations: {}", result.n_eval);
        println!("  Jacobian evaluations: {}", result.n_jac);
        println!(
            "  Steps: {} (accepted: {}, rejected: {})",
            result.n_steps, result.n_accepted, result.n_rejected
        );

        // Store results for comparison
        results.push((name, elapsed, result));
    }

    // Compare final results
    println!("\nComparing final results:");
    println!("Strategy                  | Time        | Final y[0]    | Final y[1]");
    println!("--------------------------+-------------+---------------+---------------");

    for (name, elapsed, result) in &results {
        let _last_t = result.t.last().unwrap();
        let last_y = result.y.last().unwrap();

        println!(
            "{:<25} | {:<11.2?} | {:<13.6} | {:<13.6}",
            name, elapsed, last_y[0], last_y[1]
        );
    }

    // Compare efficiencies
    if results.len() > 1 {
        println!("\nEfficiency comparison (relative to finite difference):");

        let base_time = if let Some((_, time, _)) = results
            .iter()
            .find(|(name, _, _)| *name == "Finite Difference")
        {
            *time
        } else {
            results[0].1
        };

        for (name, elapsed, result) in &results {
            let speedup = base_time.as_secs_f64() / elapsed.as_secs_f64();

            println!(
                "{:<25} | {:.2}x speedup | {} function evals",
                name, speedup, result.n_eval
            );
        }
    }

    println!("\nConclusion:");
    if autodiff_available {
        println!("  Automatic differentiation can provide more accurate Jacobians");
        println!("  and improve performance for stiff ODE systems.");
    } else {
        println!("  Enable the 'autodiff' feature for potentially better performance");
        println!("  when solving stiff ODE systems.");
    }

    Ok(())
}

// Helper function for comparing solutions
#[allow(dead_code)]
fn solution_diff(res1: &ODEResult<f64>, res2: &ODEResult<f64>) -> f64 {
    // Find common time points to compare
    let mut max_diff: f64 = 0.0;

    // Sample a few points for comparison
    let sample_times = [
        res1.t[0],
        res1.t[res1.t.len() / 4],
        res1.t[res1.t.len() / 2],
        res1.t[3 * res1.t.len() / 4],
        *res1.t.last().unwrap(),
    ];

    for &t in &sample_times {
        // Find closest points in each solution
        let idx1 = res1
            .t
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| (t - **a).abs().partial_cmp(&(t - **b).abs()).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();

        let idx2 = res2
            .t
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| (t - **a).abs().partial_cmp(&(t - **b).abs()).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();

        // Compute difference
        let y1 = &res1.y[idx1];
        let y2 = &res2.y[idx2];

        let diff = (0..y1.len())
            .map(|i| (y1[i] - y2[i]).abs())
            .fold(0.0_f64, |a, b| a.max(b));

        max_diff = max_diff.max(diff);
    }

    max_diff
}
