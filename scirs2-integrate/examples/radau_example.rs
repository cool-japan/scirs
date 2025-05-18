use ndarray::{array, ArrayView1};
use scirs2_integrate::ode::{solve_ivp, ODEMethod, ODEOptions};
use std::f64::consts::PI;

fn main() {
    println!("Radau Implicit Runge-Kutta Solver Examples");
    println!("-------------------------------------------");

    // Example 1: Simple problem (exponential decay)
    println!("\nExample 1: Simple Exponential Decay");
    println!("y' = -y, y(0) = 1 (exact solution: y(t) = e^(-t))");

    // Try Radau first, fall back to RK45 if it fails
    let result = match solve_ivp(
        |_, y| array![-y[0]],
        [0.0, 2.0],
        array![1.0],
        Some(ODEOptions {
            method: ODEMethod::Radau,
            rtol: 1e-6,
            atol: 1e-8,
            h0: Some(0.1),        // Specify initial step size
            min_step: Some(1e-6), // Use a larger minimum step size
            max_steps: 1000,      // Increase max steps
            ..Default::default()
        }),
    ) {
        Ok(res) => res,
        Err(e) => {
            println!("Radau method failed: {}. Trying RK45 instead.", e);
            solve_ivp(
                |_, y| array![-y[0]],
                [0.0, 2.0],
                array![1.0],
                Some(ODEOptions {
                    method: ODEMethod::RK45,
                    rtol: 1e-6,
                    atol: 1e-8,
                    ..Default::default()
                }),
            )
            .unwrap()
        }
    };

    // Print results at different time points
    println!("\nSolution at selected points:");
    println!("{:^10} {:^15} {:^15} {:^15}", "t", "y", "exact", "error");
    println!("{:-<10} {:-<15} {:-<15} {:-<15}", "", "", "", "");

    let exact_points = [0.0, 0.5, 1.0, 1.5, 2.0];
    for &t in &exact_points {
        // Find closest solution point
        let idx = result
            .t
            .iter()
            .position(|&rt| (rt - t as f64).abs() < 1e-10_f64)
            .unwrap_or_else(|| {
                result
                    .t
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| {
                        let a_diff = (*a - t as f64).abs();
                        let b_diff = (*b - t as f64).abs();
                        a_diff.partial_cmp(&b_diff).unwrap()
                    })
                    .map(|(i, _)| i)
                    .unwrap()
            });

        let y_val = result.y[idx][0];
        let exact = (-t as f64).exp();
        let error = (y_val - exact).abs();

        println!(
            "{:^10} {:^15} {:^15} {:^15.2e}",
            result.t[idx], y_val, exact, error
        );
    }

    // Print statistics
    println!("\nStatistics:");
    println!("  Number of steps: {}", result.n_steps);
    println!("  Number of function evaluations: {}", result.n_eval);
    println!("  Number of accepted steps: {}", result.n_accepted);
    println!("  Number of rejected steps: {}", result.n_rejected);
    // println!("  Final step size: {:.6e}", result.final_step.unwrap());
    println!("  Success: {}", result.success);
    if let Some(msg) = &result.message {
        println!("  Message: {}", msg);
    }

    // Example 2: Stiff problem (Van der Pol oscillator)
    println!("\n\nExample 2: Stiff Problem - Van der Pol Oscillator");
    println!("y'' - μ(1-y²)y' + y = 0 with μ = 1000 (very stiff)");
    println!("As a system: y₀' = y₁, y₁' = μ(1-y₀²)y₁ - y₀");

    // Large mu makes this problem very stiff
    let mu = 1000.0;

    let van_der_pol =
        |_t: f64, y: ArrayView1<f64>| array![y[1], mu * (1.0 - y[0].powi(2)) * y[1] - y[0]];

    // Comparing different solvers on this stiff problem
    println!("\nComparing different ODE solvers on this stiff problem:");

    // Solve with BDF (backward differentiation formula - good for stiff problems)
    let start_time = std::time::Instant::now();
    let result_bdf = solve_ivp(
        van_der_pol,
        [0.0, 3.0],
        array![2.0, 0.0],
        Some(ODEOptions {
            method: ODEMethod::Bdf,
            rtol: 1e-3,
            atol: 1e-6,
            h0: Some(0.01),
            min_step: Some(1e-6),
            max_steps: 1000,
            ..Default::default()
        }),
    );
    let bdf_time = start_time.elapsed();

    match result_bdf {
        Ok(result) => {
            println!("\nBDF method (Implicit, A-stable, variable order):");
            println!("  Computation time: {:.2?}", bdf_time);
            println!("  Steps taken: {}", result.n_steps);
            println!("  Function evaluations: {}", result.n_eval);
            println!("  Success: {}", result.success);

            // Print final solution
            let final_y = result.y.last().unwrap();
            println!("  Final state: [{}, {}]", final_y[0], final_y[1]);
        }
        Err(e) => {
            println!("BDF solver failed: {}", e);
        }
    }

    // Solve with DOP853 (explicit method, may struggle with stiff problems)
    let start_time = std::time::Instant::now();
    let result_dop853 = solve_ivp(
        van_der_pol,
        [0.0, 3.0],
        array![2.0, 0.0],
        Some(ODEOptions {
            method: ODEMethod::DOP853,
            rtol: 1e-3,
            atol: 1e-6,
            max_steps: 10000, // May need many steps for stiff problem
            ..Default::default()
        }),
    );
    let dop853_time = start_time.elapsed();

    match result_dop853 {
        Ok(result) => {
            println!("\nDOP853 method (Explicit, 8th order):");
            println!("  Computation time: {:.2?}", dop853_time);
            println!("  Steps taken: {}", result.n_steps);
            println!("  Function evaluations: {}", result.n_eval);
            println!("  Success: {}", result.success);

            // Print final solution
            let final_y = result.y.last().unwrap();
            println!("  Final state: [{}, {}]", final_y[0], final_y[1]);
        }
        Err(e) => {
            println!("DOP853 solver failed: {}", e);
        }
    }

    // Example 3: Harmonic oscillator with long-term integration
    println!("\n\nExample 3: Long-term Integration of Harmonic Oscillator");
    println!("y'' + y = 0 (harmonic oscillator)");
    println!("As a system: y₀' = y₁, y₁' = -y₀");
    println!("Initial condition: [1, 0], Exact solution: y₀(t) = cos(t), y₁(t) = -sin(t)");

    // Let's integrate for many periods to test stability
    let periods = 20;
    let t_end = 2.0 * PI * periods as f64;

    // Try Radau first, fall back to RK45 if it fails
    let result = match solve_ivp(
        |_, y| array![y[1], -y[0]],
        [0.0, t_end],
        array![1.0, 0.0],
        Some(ODEOptions {
            method: ODEMethod::Radau,
            rtol: 1e-6,
            atol: 1e-8,
            h0: Some(0.1),        // Specify initial step size
            min_step: Some(1e-6), // Use a larger minimum step size
            max_steps: 1000,      // Increase max steps
            ..Default::default()
        }),
    ) {
        Ok(res) => res,
        Err(e) => {
            println!("Radau method failed: {}. Trying RK45 instead.", e);
            solve_ivp(
                |_, y| array![y[1], -y[0]],
                [0.0, t_end],
                array![1.0, 0.0],
                Some(ODEOptions {
                    method: ODEMethod::RK45,
                    rtol: 1e-6,
                    atol: 1e-8,
                    ..Default::default()
                }),
            )
            .unwrap()
        }
    };

    // Check accuracy at the end (should return to initial condition)
    let final_y = result.y.last().unwrap();
    let exact_final = [1.0, 0.0]; // After complete periods, should return to initial state

    println!("\nAfter {} periods (t = {}):", periods, t_end);
    println!("  Final state: [{}, {}]", final_y[0], final_y[1]);
    println!("  Exact final: [{}, {}]", exact_final[0], exact_final[1]);
    println!(
        "  Errors: [{:.2e}, {:.2e}]",
        (final_y[0] - exact_final[0] as f64).abs(),
        (final_y[1] - exact_final[1] as f64).abs()
    );

    println!("\nStatistics:");
    println!("  Number of steps: {}", result.n_steps);
    println!("  Number of function evaluations: {}", result.n_eval);
    println!(
        "  Steps per period: {}",
        (result.n_steps as f64 / periods as f64)
    );
}
