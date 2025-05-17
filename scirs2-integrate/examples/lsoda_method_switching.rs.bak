use ndarray::{array, ArrayView1};
use scirs2_integrate::ode::{solve_ivp, ODEMethod, ODEOptions};
use std::fs::File;
use std::io::Write;

fn main() {
    println!("LSODA Method Switching Example");
    println!("------------------------------");
    println!("This example demonstrates how LSODA switches between Adams and BDF methods");
    println!("We'll solve problems that transition between non-stiff and stiff regions");

    // Example 1: A system that transitions from non-stiff to stiff behavior
    println!("\nExample 1: Trigonometric-Exponential System");
    println!("This system transitions from oscillatory (non-stiff) to rapidly decaying (stiff)");
    println!("  y'₁ = -δ(t)y₁, where δ(t) smoothly transitions from 0.1 to 1000");
    println!("  y'₂ = -y₂, a simple reference decay used for comparison");

    let mixed_system = |t: f64, y: ArrayView1<f64>| {
        // Stiffness parameter that smoothly transitions from non-stiff to stiff
        // Using a sigmoid function to transition from 0.1 to 1000 around t=5
        let delta = 0.1 + 999.9 / (1.0 + (-2.0 * (t - 5.0)).exp());

        // Component 1: Variable stiffness exponential decay
        // Component 2: Reference exponential decay with constant rate
        array![-delta * y[0], -y[1]]
    };

    let result = solve_ivp(
        mixed_system,
        [0.0, 10.0],
        array![1.0, 1.0],
        Some(ODEOptions {
            method: ODEMethod::LSODA,
            rtol: 1e-4,
            atol: 1e-6,
            max_steps: 2000,
            h0: Some(0.1),
            min_step: Some(1e-6),
            // No first_step_check option available
            ..Default::default()
        }),
    );

    match result {
        Ok(res) => {
            println!("Integration successful!");
            println!("Solver statistics:");
            println!("  Steps: {}", res.n_steps);
            println!("  Function evaluations: {}", res.n_eval);
            println!("  Accepted steps: {}", res.n_accepted);
            println!("  Rejected steps: {}", res.n_rejected);

            if let Some(msg) = &res.message {
                println!("  {}", msg);
            }

            // Save data to file for potential plotting
            let mut file = File::create("lsoda_method_switching_ex1.txt").unwrap();
            writeln!(&mut file, "# t, y1, y2, stiffness_param").unwrap();

            for i in 0..res.t.len() {
                let t = res.t[i];
                let y1 = res.y[i][0];
                let y2 = res.y[i][1];

                // Calculate the stiffness parameter at each time point
                let delta = 0.1 + 999.9 / (1.0 + (-2.0 * (t - 5.0)).exp());

                writeln!(&mut file, "{:.6} {:.10e} {:.10e} {:.6}", t, y1, y2, delta).unwrap();

                // Also print selected points to console
                if i % (res.t.len() / 10).max(1) == 0 || i == res.t.len() - 1 {
                    let stiff_region = if delta > 10.0 { "stiff" } else { "non-stiff" };
                    println!(
                        "  t={:.3}: y₁={:.6e}, y₂={:.6e}, δ={:.3} ({})",
                        t, y1, y2, delta, stiff_region
                    );
                }
            }

            println!("\nData saved to lsoda_method_switching_ex1.txt for plotting");
            println!("The stiffness parameter δ transitions from 0.1 to 1000 around t=5");
            println!("LSODA should switch from Adams to BDF method in the stiff region");
        }
        Err(e) => {
            println!("Integration failed: {}", e);
        }
    }

    // Example 2: Robertson chemical reaction system with time-varying reaction rate
    println!("\nExample 2: Modified Robertson Chemical Reaction");
    println!("This is a classic stiff problem from chemical kinetics");
    println!("We modify the middle reaction rate to vary with time, changing stiffness");
    println!("The system becomes most stiff when k₂ is largest (around t=50)");

    // Robertson chemical reaction system with time-dependent rate coefficient
    let robertson = |t: f64, y: ArrayView1<f64>| {
        // Traditional rate coefficients
        let k1 = 0.04;

        // Make k2 time-dependent - increases to peak at t=50, then decreases
        // This changes the stiffness of the system over time
        let base_k2 = 3.0e7;
        let k2 = base_k2 * (-(t - 50.0).powi(2) / 500.0).exp().max(0.01);

        let k3 = 1.0e4;

        array![
            -k1 * y[0] + k3 * y[1] * y[2],
            k1 * y[0] - k2 * y[1].powi(2) - k3 * y[1] * y[2],
            k2 * y[1].powi(2)
        ]
    };

    // Initial conditions for Robertson problem
    let y0 = array![1.0, 0.0, 0.0];

    let result = solve_ivp(
        robertson,
        [0.0, 100.0],
        y0,
        Some(ODEOptions {
            method: ODEMethod::LSODA,
            rtol: 1e-4,
            atol: 1e-7,
            max_steps: 5000,
            // No first_step_check option available
            ..Default::default()
        }),
    );

    match result {
        Ok(res) => {
            println!("Integration successful!");
            println!("Solver statistics:");
            println!("  Steps: {}", res.n_steps);
            println!("  Function evaluations: {}", res.n_eval);
            println!("  Accepted steps: {}", res.n_accepted);
            println!("  Rejected steps: {}", res.n_rejected);

            if let Some(msg) = &res.message {
                println!("  {}", msg);
            }

            // Save data to file for potential plotting
            let mut file = File::create("lsoda_method_switching_ex2.txt").unwrap();
            writeln!(&mut file, "# t, y1, y2, y3, k2_value").unwrap();

            for i in 0..res.t.len() {
                let t = res.t[i];
                let y1 = res.y[i][0];
                let y2 = res.y[i][1];
                let y3 = res.y[i][2];

                // Calculate k2 at this time point
                let base_k2 = 3.0e7;
                let k2 = base_k2 * (-(t - 50.0).powi(2) / 500.0).exp().max(0.01);

                writeln!(
                    &mut file,
                    "{:.6} {:.10e} {:.10e} {:.10e} {:.6e}",
                    t, y1, y2, y3, k2
                )
                .unwrap();

                // Print selected points to console
                if i % (res.t.len() / 10).max(1) == 0 || i == res.t.len() - 1 {
                    let stiff_level = if k2 > 1.0e7 {
                        "very stiff"
                    } else if k2 > 1.0e6 {
                        "moderately stiff"
                    } else {
                        "less stiff"
                    };

                    println!(
                        "  t={:.1}: y=[{:.3e}, {:.3e}, {:.3e}], k₂={:.3e} ({})",
                        t, y1, y2, y3, k2, stiff_level
                    );
                }
            }

            println!("\nData saved to lsoda_method_switching_ex2.txt for plotting");
            println!("The k₂ parameter peaks at t=50, making the system most stiff there");
            println!(
                "LSODA should use BDF method in the stiff regions and may switch back to Adams"
            );
        }
        Err(e) => {
            println!("Integration failed: {}", e);
        }
    }

    // Example 3: Comparison with fixed methods (Adams and BDF) for the same problem
    println!("\nExample 3: Method Comparison for the Same Problem");
    println!("We'll solve the first example with Adams-only, BDF-only, and LSODA");
    println!("This demonstrates the efficiency advantage of adaptive method switching");

    // Dictionary to store results from different methods
    let methods = [
        ("LSODA (adaptive switching)", ODEMethod::LSODA),
        ("RK45 (non-stiff method)", ODEMethod::RK45),
        ("BDF (stiff method)", ODEMethod::Bdf),
    ];

    for (method_name, method) in methods.iter() {
        println!("\nSolving with {}", method_name);

        let result = solve_ivp(
            mixed_system, // Reuse the first example system
            [0.0, 10.0],
            array![1.0, 1.0],
            Some(ODEOptions {
                method: *method,
                rtol: 1e-4,
                atol: 1e-6,
                max_steps: 5000,
                h0: Some(0.1),
                min_step: Some(1e-6),
                ..Default::default()
            }),
        );

        match result {
            Ok(res) => {
                println!("  Success!");
                println!("  Steps: {}", res.n_steps);
                println!("  Function evaluations: {}", res.n_eval);
                println!("  Accepted steps: {}", res.n_accepted);
                println!("  Rejected steps: {}", res.n_rejected);

                // Calculate final error compared to analytical solution for component 2
                let exact_y2 = (-res.t.last().unwrap()).exp();
                let error_y2 = (res.y.last().unwrap()[1] - exact_y2).abs();
                println!("  Final error for reference component: {:.6e}", error_y2);

                if let Some(msg) = &res.message {
                    println!("  {}", msg);
                }
            }
            Err(e) => {
                println!("  Failed: {}", e);
            }
        }
    }

    println!("\nLSODA Method Switching Analysis:");
    println!("--------------------------------");
    println!("1. LSODA starts with the Adams method (non-stiff)");
    println!("2. When stiffness is detected, it switches to BDF method");
    println!("3. It may switch back to Adams if the problem becomes non-stiff again");
    println!("4. LSODA requires at least 100 steps with a method before considering switching");
    println!("5. Switching criteria include:");
    println!("   - Step size relative to current time");
    println!("   - Step acceptance ratio");
    println!("   - Recent rejection rate");
    println!("6. The threshold for switching to BDF is lower than switching back to Adams");
    println!("7. This asymmetric hysteresis prevents oscillating between methods");
    println!("8. For problems that transition between stiff and non-stiff regions,");
    println!("   LSODA should be more efficient than either fixed method");
}
