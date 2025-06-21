//! Anderson acceleration example
//!
//! This example demonstrates how to use Anderson acceleration to speed up
//! the convergence of fixed-point iterations and nonlinear solvers.

use ndarray::{Array1, ArrayView1};
use scirs2_integrate::acceleration::{AcceleratorOptions, AitkenAccelerator, AndersonAccelerator};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Anderson Acceleration Examples ===\n");

    // Example 1: Simple scalar fixed-point iteration
    scalar_fixed_point_example()?;

    // Example 2: System of nonlinear equations
    nonlinear_system_example()?;

    // Example 3: Performance comparison
    performance_comparison_example()?;

    // Example 4: Aitken acceleration for sequences
    aitken_acceleration_example()?;

    Ok(())
}

fn scalar_fixed_point_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔄 Scalar Fixed-Point Iteration");
    println!("{}", "=".repeat(50));

    // Solve x = 0.8*cos(x) using fixed-point iteration
    // This has solution approximately x ≈ 0.6947

    println!("Problem: x = 0.8*cos(x)");
    println!("Exact solution: x ≈ 0.6947");

    // Standard fixed-point iteration
    let mut x_standard = 0.5f64;
    let mut x_anderson = Array1::from_vec(vec![0.5f64]);

    let mut accelerator = AndersonAccelerator::new(1, AcceleratorOptions::default());

    println!("\nIteration   Standard FP   Anderson Acc   Speedup Factor");
    println!("{}", "─".repeat(60));

    for iter in 1..=15 {
        // Standard iteration
        x_standard = 0.8f64 * x_standard.cos();

        // Anderson accelerated iteration
        let g_x = Array1::from_vec(vec![0.8f64 * x_anderson[0].cos()]);
        if let Some(x_new) = accelerator.accelerate(x_anderson.view(), g_x.view()) {
            x_anderson = x_new;
        }

        let exact = 0.6947f64;
        let error_standard = (x_standard - exact).abs();
        let error_anderson = (x_anderson[0] - exact).abs();

        let speedup = if error_anderson > 0.0 {
            error_standard / error_anderson
        } else {
            f64::INFINITY
        };

        println!(
            "{:9}   {:11.6}   {:12.6}   {:10.2}",
            iter, x_standard, x_anderson[0], speedup
        );

        if error_anderson < 1e-10 {
            println!("✅ Anderson acceleration converged in {} iterations", iter);
            break;
        }
    }

    println!();
    Ok(())
}

fn nonlinear_system_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 Nonlinear System Solving");
    println!("{}", "=".repeat(50));

    // Solve system:
    // x₁ = 0.3*x₁ + 0.2*x₂ + 1.0
    // x₂ = 0.1*x₁ + 0.4*x₂ + 0.5
    //
    // Exact solution: x₁ = 1.875, x₂ = 1.25

    println!("System:");
    println!("x₁ = 0.3*x₁ + 0.2*x₂ + 1.0");
    println!("x₂ = 0.1*x₁ + 0.4*x₂ + 0.5");
    println!("Exact solution: x₁ = 1.875, x₂ = 1.25");

    let options = AcceleratorOptions {
        memory_depth: 5,
        damping: 0.8f64,
        ..Default::default()
    };

    let mut accelerator = AndersonAccelerator::new(2, options);
    let mut x = Array1::from_vec(vec![0.0f64, 0.0f64]);

    let exact = Array1::from_vec(vec![1.875f64, 1.25f64]);

    println!("\nIteration   x₁        x₂        Error     Active");
    println!("{}", "─".repeat(50));

    for iter in 1..=20 {
        // Define the fixed-point map G(x)
        let g_x = Array1::from_vec(vec![
            0.3f64 * x[0] + 0.2f64 * x[1] + 1.0f64,
            0.1f64 * x[0] + 0.4f64 * x[1] + 0.5f64,
        ]);

        if let Some(x_new) = accelerator.accelerate(x.view(), g_x.view()) {
            x = x_new;
        }

        let error = ((x[0] - exact[0]).powi(2) + (x[1] - exact[1]).powi(2)).sqrt();
        let active = if accelerator.is_active() { "Yes" } else { "No" };

        println!(
            "{:9}   {:7.4}   {:7.4}   {:8.2e}   {}",
            iter, x[0], x[1], error, active
        );

        if error < 1e-10 {
            println!("✅ Converged in {} iterations", iter);
            break;
        }
    }

    println!();
    Ok(())
}

fn performance_comparison_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("⚡ Performance Comparison");
    println!("{}", "=".repeat(50));

    // Compare standard vs Anderson acceleration for a challenging problem
    let problems = vec![
        (
            "Linear system",
            Box::new(linear_fixed_point) as Box<dyn Fn(ArrayView1<f64>) -> Array1<f64>>,
        ),
        ("Nonlinear system", Box::new(nonlinear_fixed_point)),
        ("Stiff problem", Box::new(stiff_fixed_point)),
    ];

    println!("Problem           Standard  Anderson  Speedup");
    println!("{}", "─".repeat(45));

    for (name, problem_fn) in problems {
        let start = Instant::now();
        let iters_standard = solve_standard(&problem_fn, 1000, 1e-8);
        let _time_standard = start.elapsed();

        let start = Instant::now();
        let iters_anderson = solve_anderson(&problem_fn, 1000, 1e-8);
        let _time_anderson = start.elapsed();

        let speedup = if iters_anderson > 0 {
            iters_standard as f64 / iters_anderson as f64
        } else {
            f64::INFINITY
        };

        println!(
            "{:15}   {:7}   {:7}   {:6.2}x",
            name, iters_standard, iters_anderson, speedup
        );
    }

    println!();
    Ok(())
}

fn aitken_acceleration_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("📈 Aitken Acceleration for Sequences");
    println!("{}", "=".repeat(50));

    // Accelerate convergence of sequence: x_n = 1 - 1/n → 1
    println!("Sequence: x_n = 1 - 1/n → 1");

    let mut aitken = AitkenAccelerator::new();

    println!("\nn    x_n      Aitken   Error     Improvement");
    println!("{}", "─".repeat(45));

    for n in 1..=10 {
        let x_n = 1.0 - 1.0 / (n as f64);
        let x_aitken = aitken.accelerate(x_n).unwrap_or(x_n);

        let error_original = (x_n - 1.0).abs();
        let error_aitken = (x_aitken - 1.0).abs();

        let improvement = if error_aitken > 0.0 {
            error_original / error_aitken
        } else {
            f64::INFINITY
        };

        println!(
            "{:2}   {:6.4}   {:6.4}   {:8.2e}   {:8.2}x",
            n, x_n, x_aitken, error_aitken, improvement
        );
    }

    println!("\n✅ Aitken acceleration can provide dramatic improvements for slowly converging sequences");
    println!();
    Ok(())
}

// Helper functions for performance comparison

fn linear_fixed_point(x: ArrayView1<f64>) -> Array1<f64> {
    // x₁ = 0.7*x₁ + 0.1*x₂ + 1.0
    // x₂ = 0.2*x₁ + 0.6*x₂ + 0.5
    Array1::from_vec(vec![
        0.7 * x[0] + 0.1 * x[1] + 1.0,
        0.2 * x[0] + 0.6 * x[1] + 0.5,
    ])
}

fn nonlinear_fixed_point(x: ArrayView1<f64>) -> Array1<f64> {
    // More challenging nonlinear system
    Array1::from_vec(vec![
        0.5 * x[0] + 0.2 * (x[1] * x[1]).sin() + 0.8,
        0.3 * (x[0] * x[0]).cos() + 0.4 * x[1] + 0.6,
    ])
}

fn stiff_fixed_point(x: ArrayView1<f64>) -> Array1<f64> {
    // Stiff problem with large Jacobian eigenvalues
    Array1::from_vec(vec![
        0.95 * x[0] + 0.02 * x[1] + 0.1,
        0.01 * x[0] + 0.94 * x[1] + 0.05,
    ])
}

type IterationFunction = Box<dyn Fn(ArrayView1<f64>) -> Array1<f64>>;

fn solve_standard(problem: &IterationFunction, max_iters: usize, tolerance: f64) -> usize {
    let mut x = Array1::from_vec(vec![0.0, 0.0]);

    for iter in 1..=max_iters {
        let x_new = problem(x.view());
        let error = ((x_new[0] - x[0]).powi(2) + (x_new[1] - x[1]).powi(2)).sqrt();

        x = x_new;

        if error < tolerance {
            return iter;
        }
    }

    max_iters
}

fn solve_anderson(problem: &IterationFunction, max_iters: usize, tolerance: f64) -> usize {
    let mut accelerator = AndersonAccelerator::new(2, AcceleratorOptions::default());
    let mut x = Array1::from_vec(vec![0.0, 0.0]);
    let mut x_prev = x.clone();

    for iter in 1..=max_iters {
        let g_x = problem(x.view());

        if let Some(x_new) = accelerator.accelerate(x.view(), g_x.view()) {
            let error = ((x_new[0] - x_prev[0]).powi(2) + (x_new[1] - x_prev[1]).powi(2)).sqrt();

            x_prev = x.clone();
            x = x_new;

            if error < tolerance {
                return iter;
            }
        }
    }

    max_iters
}
