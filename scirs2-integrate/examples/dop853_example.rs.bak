// FIXME: This example has compilation errors - field `final_step` doesn't exist
#![allow(dead_code)]

use ndarray::{array, ArrayView1};
use scirs2_integrate::ode::{solve_ivp, ODEMethod, ODEOptions};
use std::f64::consts::PI;

fn main() {
    println!("DOP853 High-Order Runge-Kutta Solver Example");
    println!("--------------------------------------------");

    // Example 1: Simple Harmonic Oscillator with High Accuracy
    println!("Example 1: Simple Harmonic Oscillator with High Accuracy");

    // Define harmonic oscillator: y'' + y = 0
    // As first-order system: y0' = y1, y1' = -y0
    let harmonic = |_t: f64, y: ArrayView1<f64>| array![y[1], -y[0]];

    // Solve with DOP853 method (high-order Runge-Kutta)
    let options = ODEOptions {
        method: ODEMethod::DOP853, // 8th order Dormand-Prince method
        rtol: 1e-10,               // Very strict relative tolerance
        atol: 1e-12,               // Very strict absolute tolerance
        ..Default::default()
    };

    // Initial condition [1, 0] (cosine solution)
    // Integrate over a long interval (0 to 10π) = 5 complete cycles
    let result = solve_ivp(harmonic, [0.0, 10.0 * PI], array![1.0, 0.0], Some(options)).unwrap();

    println!("\nSolving harmonic oscillator over t = [0, 10π] with high accuracy");
    println!("Exact solution is y(t) = cos(t), y'(t) = -sin(t)");
    println!("Initial condition: [1.0, 0.0]");

    // Check at specific points (multiples of π)
    let check_points = vec![0.0, PI, 2.0 * PI, 3.0 * PI, 4.0 * PI, 5.0 * PI];

    println!("\nChecking solution at key points:");
    println!("{:>10} {:>15} {:>15} {:>15}", "t", "y[0]", "exact", "error");
    println!("{:->10} {:->15} {:->15} {:->15}", "", "", "", "");

    for &t in &check_points {
        // Find closest time point
        let idx = result
            .t
            .iter()
            .enumerate()
            .min_by(|(_, &a), (_, &b)| (a - t).abs().partial_cmp(&(b - t).abs()).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        let y0 = result.y[idx][0];
        let exact0 = t.cos();
        let error = (y0 - exact0).abs();

        println!(
            "{:10.6} {:15.10} {:15.10} {:15.2e}",
            result.t[idx], y0, exact0, error
        );
    }

    println!("\nStatistics:");
    println!("  Number of steps: {}", result.n_steps);
    println!("  Number of function evaluations: {}", result.n_eval);
    println!("  Number of accepted steps: {}", result.n_accepted);
    println!("  Number of rejected steps: {}", result.n_rejected);
    // println!(
    //     "  Final step size: {:.6e}",
    //     result.final_step.unwrap_or(0.0)  // FIXME: final_step field doesn't exist in ODEResult
    // );
    println!("  Success: {}", result.success);

    // Check final state (should be close to [1, 0] after 5 full cycles)
    let final_y = result.y.last().unwrap();
    println!("\nFinal state: [{:.10}, {:.10}]", final_y[0], final_y[1]);
    println!(
        "Error from exact: [{:.2e}, {:.2e}]",
        (final_y[0] - 1.0).abs(),
        (final_y[1] - 0.0).abs()
    );

    // Example 2: Comparing Methods (Exponential Decay)
    println!("\n\nExample 2: Comparing Different ODE Methods");
    println!("Solving y' = -y, y(0) = 1 over t = [0, 5]");
    println!("Exact solution: y(t) = e^(-t)");

    // Define function: y' = -y
    let exp_decay = |_t: f64, y: ArrayView1<f64>| array![-y[0]];

    // Compare different methods
    let methods = vec![ODEMethod::RK23, ODEMethod::RK45, ODEMethod::DOP853];

    let method_names = vec!["RK23 (order 3)", "RK45 (order 5)", "DOP853 (order 8)"];

    println!(
        "\n{:>15} {:>12} {:>12} {:>15} {:>15}",
        "Method", "Steps", "Evaluations", "Final Value", "Error"
    );
    println!(
        "{:->15} {:->12} {:->12} {:->15} {:->15}",
        "", "", "", "", ""
    );

    for (i, &method) in methods.iter().enumerate() {
        let options = ODEOptions {
            method,
            rtol: 1e-8,
            atol: 1e-10,
            ..Default::default()
        };

        let result = solve_ivp(exp_decay, [0.0, 5.0], array![1.0], Some(options)).unwrap();

        let final_y = result.y.last().unwrap()[0];
        let exact = (-5.0f64).exp();
        let error = (final_y - exact).abs();

        println!(
            "{:>15} {:>12} {:>12} {:>15.10} {:>15.2e}",
            method_names[i], result.n_steps, result.n_eval, final_y, error
        );
    }

    // Example 3: Kepler Problem (Two-Body Orbital Motion)
    println!("\n\nExample 3: Kepler Problem (Two-Body Orbital Motion)");

    // Kepler's two-body problem in 2D
    // y[0], y[1]: position coordinates
    // y[2], y[3]: velocity components
    let kepler = |_t: f64, y: ArrayView1<f64>| {
        let r = (y[0] * y[0] + y[1] * y[1]).sqrt();
        let r3 = r * r * r;

        // Return derivatives [vx, vy, ax, ay]
        array![
            y[2],       // x' = vx
            y[3],       // y' = vy
            -y[0] / r3, // vx' = -x/r³
            -y[1] / r3, // vy' = -y/r³
        ]
    };

    // Initial conditions for circular orbit at unit distance
    // [x, y, vx, vy] = [1, 0, 0, 1]
    let y0 = array![1.0, 0.0, 0.0, 1.0];

    // Solve for one complete orbit (period = 2π for circular orbit)
    let options = ODEOptions {
        method: ODEMethod::DOP853,
        rtol: 1e-10,
        atol: 1e-12,
        ..Default::default()
    };

    let result = solve_ivp(kepler, [0.0, 2.0 * PI], y0, Some(options)).unwrap();

    println!("Solving for one complete orbit (period = 2π)");
    println!("Initial state: [x, y, vx, vy] = [1, 0, 0, 1]");
    println!("After one orbit, should return to initial state");

    let final_state = result.y.last().unwrap();
    println!(
        "\nFinal state: [{:.10}, {:.10}, {:.10}, {:.10}]",
        final_state[0], final_state[1], final_state[2], final_state[3]
    );

    // Calculate error relative to exact solution (should return to initial state)
    let errors = [
        (final_state[0] - 1.0).abs(),
        (final_state[1] - 0.0).abs(),
        (final_state[2] - 0.0).abs(),
        (final_state[3] - 1.0).abs(),
    ];

    println!(
        "Errors: [{:.2e}, {:.2e}, {:.2e}, {:.2e}]",
        errors[0], errors[1], errors[2], errors[3]
    );

    println!("\nStatistics:");
    println!("  Number of steps: {}", result.n_steps);
    println!("  Number of function evaluations: {}", result.n_eval);
    println!("  Number of accepted steps: {}", result.n_accepted);
    println!("  Number of rejected steps: {}", result.n_rejected);
}
