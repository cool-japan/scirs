use ndarray::{array, ArrayView1};
use scirs2_integrate::ode::{solve_ivp, ODEMethod, ODEOptions, ODEResult};
use std::fs::File;
use std::io::Write;
use std::time::Instant;

fn main() {
    println!("LSODA Method Comparison Example");
    println!("-------------------------------");
    println!("This example compares LSODA with other ODE solvers on different problem types");
    println!("to demonstrate where each method excels");

    // Collect comparison data
    let mut results_table = String::new();
    results_table.push_str("Problem Type,Method,Steps,Function Evals,Solve Time (ms),Error\n");

    // Define different ODE methods to compare
    let methods = [
        ODEMethod::LSODA,  // Adaptive switching
        ODEMethod::RK45,   // Good for non-stiff problems
        ODEMethod::Bdf,    // Good for stiff problems
        ODEMethod::RK23,   // Low-order explicit method
        ODEMethod::DOP853, // High-order explicit method
        ODEMethod::Radau,  // Implicit method for stiff problems
    ];

    // ===== Problem 1: Non-stiff oscillatory system =====
    println!("\nProblem 1: Non-stiff Oscillatory System (Van der Pol with μ=1)");
    let mu = 1.0; // Non-stiff parameter

    let van_der_pol =
        move |_t: f64, y: ArrayView1<f64>| array![y[1], mu * (1.0 - y[0].powi(2)) * y[1] - y[0]];

    println!("Method       | Steps | Func Evals | Time (ms) | Error");
    println!("--------------------------------------------------------");

    // Run all methods and compare
    for method in &methods {
        let method_name = format!("{:?}", method);

        // Measure execution time
        let start = Instant::now();

        let result = solve_ivp(
            van_der_pol.clone(),
            [0.0, 20.0],
            array![2.0, 0.0],
            Some(ODEOptions {
                method: *method,
                rtol: 1e-6,
                atol: 1e-8,
                max_steps: 5000,
                h0: Some(0.1),
                min_step: Some(1e-6),
                ..Default::default()
            }),
        );

        let elapsed = start.elapsed().as_millis();

        match result {
            Ok(res) => {
                // Calculate "error" as deviation from the most accurate solution
                // (For now, just use the last value compared to a reference)
                let reference = 1.89; // Approximate last cycle amplitude
                let error = (res.y.last().unwrap()[0] - reference).abs();

                println!(
                    "{:<12} | {:5} | {:10} | {:8} | {:.2e}",
                    method_name, res.n_steps, res.n_eval, elapsed, error
                );

                // Add to results table
                results_table.push_str(&format!(
                    "Non-stiff Oscillatory,{},{},{},{},{:.6e}\n",
                    method_name, res.n_steps, res.n_eval, elapsed, error
                ));

                // Save solution trajectory for this method
                save_trajectory(&res, &format!("non_stiff_oscillatory_{}.csv", method_name));
            }
            Err(e) => {
                println!("{:<12} | Failed: {}", method_name, e);
                results_table.push_str(&format!(
                    "Non-stiff Oscillatory,{},Failed,Failed,Failed,Failed\n",
                    method_name
                ));
            }
        }
    }

    // ===== Problem 2: Stiff system =====
    println!("\nProblem 2: Stiff System (Van der Pol with μ=1000)");
    let mu = 1000.0; // Stiff parameter

    let stiff_van_der_pol =
        move |_t: f64, y: ArrayView1<f64>| array![y[1], mu * (1.0 - y[0].powi(2)) * y[1] - y[0]];

    println!("Method       | Steps | Func Evals | Time (ms) | Error");
    println!("--------------------------------------------------------");

    for method in &methods {
        let method_name = format!("{:?}", method);

        let start = Instant::now();

        let result = solve_ivp(
            stiff_van_der_pol.clone(),
            [0.0, 20.0],
            array![2.0, 0.0],
            Some(ODEOptions {
                method: *method,
                rtol: 1e-4, // Relaxed tolerance for stiff problem
                atol: 1e-6,
                max_steps: 10000,
                h0: Some(0.01), // Smaller initial step for stiff problem
                min_step: Some(1e-6),
                ..Default::default()
            }),
        );

        let elapsed = start.elapsed().as_millis();

        match result {
            Ok(res) => {
                // For stiff problems, amplitude approaches 2.0
                let reference = 2.0;
                let error = (res.y.last().unwrap()[0] - reference).abs();

                println!(
                    "{:<12} | {:5} | {:10} | {:8} | {:.2e}",
                    method_name, res.n_steps, res.n_eval, elapsed, error
                );

                results_table.push_str(&format!(
                    "Stiff System,{},{},{},{},{:.6e}\n",
                    method_name, res.n_steps, res.n_eval, elapsed, error
                ));

                save_trajectory(&res, &format!("stiff_system_{}.csv", method_name));
            }
            Err(e) => {
                println!("{:<12} | Failed: {}", method_name, e);
                results_table.push_str(&format!(
                    "Stiff System,{},Failed,Failed,Failed,Failed\n",
                    method_name
                ));
            }
        }
    }

    // ===== Problem 3: System with time-varying stiffness =====
    println!("\nProblem 3: System with Time-varying Stiffness");

    let varying_stiffness = |t: f64, y: ArrayView1<f64>| {
        // Stiffness varies from low to high in first half, then back to low
        let stiffness = if t < 10.0 {
            0.1 + 999.9 * (t / 10.0)
        } else {
            1000.0 - 999.9 * ((t - 10.0) / 10.0).min(1.0)
        };

        array![-stiffness * y[0], -y[1]]
    };

    println!("Method       | Steps | Func Evals | Time (ms) | Error");
    println!("--------------------------------------------------------");

    for method in &methods {
        let method_name = format!("{:?}", method);

        let start = Instant::now();

        let result = solve_ivp(
            varying_stiffness,
            [0.0, 20.0],
            array![1.0, 1.0],
            Some(ODEOptions {
                method: *method,
                rtol: 1e-4,
                atol: 1e-6,
                max_steps: 10000,
                h0: Some(0.1),
                min_step: Some(1e-6),
                ..Default::default()
            }),
        );

        let elapsed = start.elapsed().as_millis();

        match result {
            Ok(res) => {
                // For exponential decay, error is simply the final absolute value
                // (since it should decay to zero)
                let error = res.y.last().unwrap()[0].abs();

                println!(
                    "{:<12} | {:5} | {:10} | {:8} | {:.2e}",
                    method_name, res.n_steps, res.n_eval, elapsed, error
                );

                results_table.push_str(&format!(
                    "Time-varying Stiffness,{},{},{},{},{:.6e}\n",
                    method_name, res.n_steps, res.n_eval, elapsed, error
                ));

                save_trajectory(&res, &format!("varying_stiffness_{}.csv", method_name));
            }
            Err(e) => {
                println!("{:<12} | Failed: {}", method_name, e);
                results_table.push_str(&format!(
                    "Time-varying Stiffness,{},Failed,Failed,Failed,Failed\n",
                    method_name
                ));
            }
        }
    }

    // ===== Problem 4: Robertson chemical kinetics problem =====
    println!("\nProblem 4: Robertson Chemical Kinetics (Stiff)");

    let robertson = |_t: f64, y: ArrayView1<f64>| {
        let k1 = 0.04;
        let k2 = 3.0e7;
        let k3 = 1.0e4;

        array![
            -k1 * y[0] + k3 * y[1] * y[2],
            k1 * y[0] - k2 * y[1].powi(2) - k3 * y[1] * y[2],
            k2 * y[1].powi(2)
        ]
    };

    println!("Method       | Steps | Func Evals | Time (ms) | Error");
    println!("--------------------------------------------------------");

    for method in &methods {
        let method_name = format!("{:?}", method);

        let start = Instant::now();

        let result = solve_ivp(
            robertson,
            [0.0, 100.0],
            array![1.0, 0.0, 0.0],
            Some(ODEOptions {
                method: *method,
                rtol: 1e-4,
                atol: 1e-8, // Tighter for this problem
                max_steps: 10000,
                h0: Some(0.001), // Small initial step for stiff transient
                min_step: Some(1e-6),
                ..Default::default()
            }),
        );

        let elapsed = start.elapsed().as_millis();

        match result {
            Ok(res) => {
                // Check mass conservation: y₁ + y₂ + y₃ should be 1
                let y = res.y.last().unwrap();
                let mass_error = (y[0] + y[1] + y[2] - 1.0).abs();

                println!(
                    "{:<12} | {:5} | {:10} | {:8} | {:.2e}",
                    method_name, res.n_steps, res.n_eval, elapsed, mass_error
                );

                results_table.push_str(&format!(
                    "Robertson Chemical,{},{},{},{},{:.6e}\n",
                    method_name, res.n_steps, res.n_eval, elapsed, mass_error
                ));

                save_trajectory(&res, &format!("robertson_{}.csv", method_name));
            }
            Err(e) => {
                println!("{:<12} | Failed: {}", method_name, e);
                results_table.push_str(&format!(
                    "Robertson Chemical,{},Failed,Failed,Failed,Failed\n",
                    method_name
                ));
            }
        }
    }

    // ===== Problem 5: Non-stiff Lotka-Volterra system =====
    println!("\nProblem 5: Lotka-Volterra Predator-Prey System (Non-stiff)");

    let lotka_volterra = |_t: f64, y: ArrayView1<f64>| {
        let alpha = 1.5;
        let beta = 1.0;
        let gamma = 3.0;
        let delta = 1.0;

        let prey = y[0];
        let predator = y[1];

        array![
            alpha * prey - beta * prey * predator,       // prey growth
            -gamma * predator + delta * prey * predator  // predator growth
        ]
    };

    println!("Method       | Steps | Func Evals | Time (ms) | Error");
    println!("--------------------------------------------------------");

    for method in &methods {
        let method_name = format!("{:?}", method);

        let start = Instant::now();

        let result = solve_ivp(
            lotka_volterra,
            [0.0, 15.0],
            array![10.0, 5.0],
            Some(ODEOptions {
                method: *method,
                rtol: 1e-6,
                atol: 1e-8,
                max_steps: 3000,
                ..Default::default()
            }),
        );

        let elapsed = start.elapsed().as_millis();

        match result {
            Ok(res) => {
                // For this oscillatory system, we can't easily define an error
                // Just use the last value as a reference
                let reference = 10.0; // Approximate prey value after multiple cycles
                let error = (res.y.last().unwrap()[0] - reference).abs();

                println!(
                    "{:<12} | {:5} | {:10} | {:8} | {:.2e}",
                    method_name, res.n_steps, res.n_eval, elapsed, error
                );

                results_table.push_str(&format!(
                    "Lotka-Volterra,{},{},{},{},{:.6e}\n",
                    method_name, res.n_steps, res.n_eval, elapsed, error
                ));

                save_trajectory(&res, &format!("lotka_volterra_{}.csv", method_name));
            }
            Err(e) => {
                println!("{:<12} | Failed: {}", method_name, e);
                results_table.push_str(&format!(
                    "Lotka-Volterra,{},Failed,Failed,Failed,Failed\n",
                    method_name
                ));
            }
        }
    }

    // Save the comparative results
    let mut file = File::create("method_comparison_results.csv").unwrap();
    file.write_all(results_table.as_bytes()).unwrap();

    println!("\nComparison Summary - LSODA vs Other Methods");
    println!("------------------------------------------");
    println!("1. For non-stiff problems (like basic Lotka-Volterra):");
    println!("   - Explicit methods like DOP853 or RK45 are usually fastest");
    println!("   - LSODA should use Adams method and perform similarly to Adams-only");
    println!();
    println!("2. For stiff problems (like Robertson chemical kinetics):");
    println!("   - Implicit methods like BDF and Radau typically perform best");
    println!("   - LSODA should use BDF method and perform similarly to BDF-only");
    println!("   - Explicit methods often struggle or require tiny step sizes");
    println!();
    println!("3. For problems with varying stiffness:");
    println!("   - LSODA should outperform fixed methods by switching as needed");
    println!("   - This advantage grows as the problem complexity increases");
    println!();
    println!("4. LSODA's strengths:");
    println!("   - Adaptivity to problem dynamics without user intervention");
    println!("   - Efficiency across different problem regimes");
    println!("   - Robustness for problems where stiffness isn't known in advance");
    println!();
    println!("5. LSODA's limitations:");
    println!("   - Overhead of method switching logic");
    println!("   - Slightly lower efficiency than the best specialized method");
    println!("   - Challenges with problems that rapidly alternate stiffness");
    println!();
    println!("Results have been saved to method_comparison_results.csv");
    println!("Individual solution trajectories have been saved to CSV files");
}

// Helper function to save solution trajectory to a CSV file
fn save_trajectory(result: &ODEResult<f64>, filename: &str) {
    let mut file = File::create(filename).unwrap();
    writeln!(
        &mut file,
        "t,{}",
        (0..result.y[0].len())
            .map(|i| format!("y{}", i + 1))
            .collect::<Vec<_>>()
            .join(",")
    )
    .unwrap();

    for i in 0..result.t.len() {
        let t = result.t[i];
        let y_values = (0..result.y[i].len())
            .map(|j| format!("{:.10e}", result.y[i][j]))
            .collect::<Vec<_>>()
            .join(",");

        writeln!(&mut file, "{:.10e},{}", t, y_values).unwrap();
    }
}
