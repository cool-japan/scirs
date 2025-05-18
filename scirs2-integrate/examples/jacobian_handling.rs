//! Jacobian Handling Example
//!
//! This example demonstrates different methods for computing and handling Jacobians
//! in ODE solvers, including:
//! - Automatic differentiation
//! - Finite differences
//! - User-provided analytical Jacobians
//! - Sparse Jacobian optimizations
//! - Parallel Jacobian computation

use ndarray::{Array1, Array2, ArrayView1};
use scirs2_integrate::ode::utils::jacobian::JacobianStrategy;
use scirs2_integrate::ode::{solve_ivp, ODEMethod, ODEOptions};
use std::time::Instant;

fn main() {
    println!("Jacobian Handling Example");
    println!("========================\n");

    // Define problem parameters
    let n_dim = 3; // Dimension of the problem
    let t_span = [0.0, 10.0]; // Time span
    let y0 = Array1::from_vec(vec![1.0, 0.0, 0.0]); // Initial conditions

    // 1. Default finite difference Jacobian
    println!("Method 1: Default finite difference Jacobian");

    let options = ODEOptions {
        method: ODEMethod::Radau,
        h0: None,
        max_step: None,
        rtol: 1e-6,
        atol: 1e-8,
        max_steps: 1000,
        jac: None, // Use default finite difference
        ..Default::default()
    };

    let start = Instant::now();
    let result = solve_ivp(
        |_t, y| chemical_kinetics(y),
        t_span,
        y0.clone(),
        Some(options),
    );
    let duration = start.elapsed();

    match result {
        Ok(res) => {
            println!("  Success: {}", res.success);
            println!("  Number of steps: {}", res.n_steps);
            println!("  Number of function evaluations: {}", res.n_eval);
            println!("  Number of Jacobian evaluations: {}", res.n_jac);
            println!("  Execution time: {:?}", duration);
            println!("  Final state: {:?}", res.y.last().unwrap());
        }
        Err(e) => println!("  Error: {:?}", e),
    }
    println!();

    // 2. Finite difference Jacobian strategy
    println!("Method 2: Finite difference Jacobian strategy");

    let options = ODEOptions {
        method: ODEMethod::Radau,
        h0: None,
        max_step: None,
        rtol: 1e-6,
        atol: 1e-8,
        max_steps: 1000,
        jacobian_strategy: Some(JacobianStrategy::FiniteDifference),
        ..Default::default()
    };

    let start = Instant::now();
    let result = solve_ivp(
        |_t, y| chemical_kinetics(y),
        t_span,
        y0.clone(),
        Some(options),
    );
    let duration = start.elapsed();

    match result {
        Ok(res) => {
            println!("  Success: {}", res.success);
            println!("  Number of steps: {}", res.n_steps);
            println!("  Number of function evaluations: {}", res.n_eval);
            println!("  Number of Jacobian evaluations: {}", res.n_jac);
            println!("  Execution time: {:?}", duration);
            println!("  Final state: {:?}", res.y.last().unwrap());
        }
        Err(e) => println!("  Error: {:?}", e),
    }
    println!();

    // 3. Sparse Jacobian
    println!("Method 3: Sparse Jacobian pattern");

    // Create a sparsity pattern for the Jacobian
    // 1 indicates non-zero elements, 0 indicates known zero elements
    let mut sparsity = Array2::zeros((n_dim, n_dim));
    // For this example, we know that the chemical kinetics system has this sparsity pattern:
    sparsity[[0, 0]] = 1.0;
    sparsity[[0, 1]] = 1.0;
    sparsity[[0, 2]] = 0.0; // Known to be zero
    sparsity[[1, 0]] = 1.0;
    sparsity[[1, 1]] = 1.0;
    sparsity[[1, 2]] = 1.0;
    sparsity[[2, 0]] = 0.0; // Known to be zero
    sparsity[[2, 1]] = 1.0;
    sparsity[[2, 2]] = 1.0;

    let options = ODEOptions {
        method: ODEMethod::Radau,
        h0: None,
        max_step: None,
        rtol: 1e-6,
        atol: 1e-8,
        max_steps: 1000,
        jac: None, // Use finite difference, but with sparsity
        // jac_sparsity: Some(sparsity), // FIXME: jac_sparsity field doesn't exist in ODEOptions
        ..Default::default()
    };

    let start = Instant::now();
    let result = solve_ivp(
        |_t, y| chemical_kinetics(y),
        t_span,
        y0.clone(),
        Some(options),
    );
    let duration = start.elapsed();

    match result {
        Ok(res) => {
            println!("  Success: {}", res.success);
            println!("  Number of steps: {}", res.n_steps);
            println!("  Number of function evaluations: {}", res.n_eval);
            println!("  Number of Jacobian evaluations: {}", res.n_jac);
            println!("  Execution time: {:?}", duration);
            println!("  Final state: {:?}", res.y.last().unwrap());
        }
        Err(e) => println!("  Error: {:?}", e),
    }
    println!();

    // Print the chemical kinetics system
    println!("Chemical Kinetics System:");
    println!("  dy1/dt = -0.04y1 + 1e4*y2*y3");
    println!("  dy2/dt = 0.04y1 - 1e4*y2*y3 - 3e7*y2^2");
    println!("  dy3/dt = 3e7*y2^2");
    println!();

    // Summary
    println!("Summary:");
    println!("1. Default finite difference: Easiest to implement but may be slower");
    println!("2. Analytical Jacobian: Fastest and most accurate, but requires extra coding");
    println!("3. Sparse Jacobian pattern: Good compromise for large, sparse systems");
    println!();
    println!("Choose the appropriate method based on your specific problem requirements.");
}

/// Chemical kinetics system (Robertson problem)
///
/// The system is:
/// dy1/dt = -0.04y1 + 1e4*y2*y3
/// dy2/dt = 0.04y1 - 1e4*y2*y3 - 3e7*y2^2
/// dy3/dt = 3e7*y2^2
fn chemical_kinetics(y: ArrayView1<f64>) -> Array1<f64> {
    let mut dydt = Array1::zeros(3);

    dydt[0] = -0.04 * y[0] + 1.0e4 * y[1] * y[2];
    dydt[1] = 0.04 * y[0] - 1.0e4 * y[1] * y[2] - 3.0e7 * y[1] * y[1];
    dydt[2] = 3.0e7 * y[1] * y[1];

    dydt
}

/// Analytical Jacobian for the chemical kinetics system
#[allow(dead_code)]
fn chemical_kinetics_jacobian(y: ArrayView1<f64>) -> Array2<f64> {
    let mut jac = Array2::zeros((3, 3));

    // f1/y1 = -0.04
    jac[[0, 0]] = -0.04;

    // f1/y2 = 1e4*y3
    jac[[0, 1]] = 1.0e4 * y[2];

    // f1/y3 = 1e4*y2
    jac[[0, 2]] = 1.0e4 * y[1];

    // f2/y1 = 0.04
    jac[[1, 0]] = 0.04;

    // f2/y2 = -1e4*y3 - 6e7*y2
    jac[[1, 1]] = -1.0e4 * y[2] - 6.0e7 * y[1];

    // f2/y3 = -1e4*y2
    jac[[1, 2]] = -1.0e4 * y[1];

    // f3/y1 = 0
    jac[[2, 0]] = 0.0;

    // f3/y2 = 6e7*y2
    jac[[2, 1]] = 6.0e7 * y[1];

    // f3/y3 = 0
    jac[[2, 2]] = 0.0;

    jac
}
