use ndarray::array;
use scirs2_integrate::ode::{solve_ivp, ODEMethod, ODEOptions};
use std::time::Instant;

/// A helper function to time and report the result of an integration method
fn time_integration<F, R>(name: &str, f: F) -> R
where
    F: FnOnce() -> R,
{
    let start = Instant::now();
    let result = f();
    let elapsed = start.elapsed();
    println!("{}: {:?}", name, elapsed);
    result
}

fn main() {
    println!("ODE solver examples");

    // Example 1: Exponential decay - y' = -y
    // Exact solution: y(t) = exp(-t)
    println!("\nExample 1: Exponential decay - y' = -y");
    let result = time_integration("RK4 fixed step method", || {
        solve_ivp(
            |_: f64, y| array![-y[0]],
            [0.0, 2.0],
            array![1.0],
            Some(ODEOptions {
                method: ODEMethod::RK4,
                h0: Some(0.1),
                ..Default::default()
            }),
        )
        .unwrap()
    });

    println!("Initial condition: y(0) = 1.0");
    println!("Final value at t = 2.0:");
    println!("  Calculated: {}", result.y.last().unwrap()[0]);
    println!("  Exact:      {}", (-2.0f64).exp());
    println!(
        "  Error:      {}",
        (result.y.last().unwrap()[0] - (-2.0f64).exp()).abs()
    );
    println!("  Steps:      {}", result.n_steps);
    println!("  Function evaluations: {}", result.n_eval);
    println!("  Success:    {}", result.success);

    // Now solve the same problem with adaptive step size
    println!("\nSolving the same problem with adaptive step size (RK45):");
    let result_adaptive = time_integration("RK45 adaptive step method", || {
        solve_ivp(
            |_: f64, y| array![-y[0]],
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
    });

    println!("Final value at t = 2.0:");
    println!("  Calculated: {}", result_adaptive.y.last().unwrap()[0]);
    println!("  Exact:      {}", (-2.0f64).exp());
    println!(
        "  Error:      {}",
        (result_adaptive.y.last().unwrap()[0] - (-2.0f64).exp()).abs()
    );
    println!("  Steps:      {}", result_adaptive.n_steps);
    println!("  Accepted steps: {}", result_adaptive.n_accepted);
    println!("  Rejected steps: {}", result_adaptive.n_rejected);
    println!("  Function evaluations: {}", result_adaptive.n_eval);
    // println!("  Final step size: {}", result_adaptive.final_step.unwrap()); // final_step field doesn't exist
    println!("  Success:    {}", result_adaptive.success);

    // Example 2: Harmonic oscillator - y'' + y = 0
    // As a system: y0' = y1, y1' = -y0
    // Exact solution: y0(t) = cos(t), y1(t) = -sin(t) with initial [1, 0]
    println!("\nExample 2: Harmonic oscillator - y'' + y = 0");
    let pi = std::f64::consts::PI;
    let result = time_integration("RK4 fixed step method", || {
        solve_ivp(
            |_: f64, y| array![y[1], -y[0]],
            [0.0, 2.0 * pi], // Integrate over [0, 2π]
            array![1.0, 0.0],
            Some(ODEOptions {
                method: ODEMethod::RK4,
                h0: Some(0.1),
                ..Default::default()
            }),
        )
        .unwrap()
    });

    println!("Initial condition: y(0) = [1.0, 0.0]");
    println!("Final value at t = 2π:");
    println!(
        "  Calculated: [{}, {}]",
        result.y.last().unwrap()[0],
        result.y.last().unwrap()[1]
    );
    println!("  Exact:      [1.0, 0.0]");
    println!(
        "  Error:      [{}, {}]",
        (result.y.last().unwrap()[0] - 1.0).abs(),
        (result.y.last().unwrap()[1] - 0.0).abs()
    );
    println!("  Steps:      {}", result.n_steps);
    println!("  Function evaluations: {}", result.n_eval);
    println!("  Success:    {}", result.success);

    // Now solve the same problem with adaptive step size
    println!("\nSolving the same problem with adaptive step size (RK45):");
    let result_adaptive = time_integration("RK45 adaptive step method", || {
        solve_ivp(
            |_: f64, y| array![y[1], -y[0]],
            [0.0, 2.0 * pi], // Integrate over [0, 2π]
            array![1.0, 0.0],
            Some(ODEOptions {
                method: ODEMethod::RK45,
                rtol: 1e-6,
                atol: 1e-8,
                ..Default::default()
            }),
        )
        .unwrap()
    });

    println!("Final value at t = 2π:");
    println!(
        "  Calculated: [{}, {}]",
        result_adaptive.y.last().unwrap()[0],
        result_adaptive.y.last().unwrap()[1]
    );
    println!("  Exact:      [1.0, 0.0]");
    println!(
        "  Error:      [{}, {}]",
        (result_adaptive.y.last().unwrap()[0] - 1.0).abs(),
        (result_adaptive.y.last().unwrap()[1] - 0.0).abs()
    );
    println!("  Steps:      {}", result_adaptive.n_steps);
    println!("  Accepted steps: {}", result_adaptive.n_accepted);
    println!("  Rejected steps: {}", result_adaptive.n_rejected);
    println!("  Function evaluations: {}", result_adaptive.n_eval);
    // println!("  Final step size: {}", result_adaptive.final_step.unwrap()); // final_step field doesn't exist
    println!("  Success:    {}", result_adaptive.success);

    // Example 3: Stiff ODE - Van der Pol oscillator
    // y'' - μ(1-y²)y' + y = 0
    // As a system: y0' = y1, y1' = μ(1-y0²)y1 - y0
    println!("\nExample 3: Stiff ODE - Van der Pol oscillator (μ=10)");

    // High value of μ makes this a stiff problem
    let mu = 10.0;

    let van_der_pol = |_t: f64, y: ndarray::ArrayView1<f64>| {
        array![y[1], mu * (1.0 - y[0].powi(2)) * y[1] - y[0]]
    };

    // Solve with RK23 (good for mildly stiff problems)
    println!("Solving with RK23 (efficient for moderately stiff problems):");
    let result_rk23 = time_integration("RK23 method", || {
        solve_ivp(
            van_der_pol,
            [0.0, 20.0],
            array![2.0, 0.0], // Start with displacement but no velocity
            Some(ODEOptions {
                method: ODEMethod::RK23,
                rtol: 1e-4,
                atol: 1e-6,
                ..Default::default()
            }),
        )
        .unwrap()
    });

    println!("  Steps:      {}", result_rk23.n_steps);
    println!("  Accepted steps: {}", result_rk23.n_accepted);
    println!("  Rejected steps: {}", result_rk23.n_rejected);
    println!("  Function evaluations: {}", result_rk23.n_eval);
    println!(
        "  Final values: [{}, {}]",
        result_rk23.y.last().unwrap()[0],
        result_rk23.y.last().unwrap()[1]
    );

    // For comparison, solve with RK45 (not as efficient for stiff problems)
    println!("\nFor comparison, solving with RK45:");
    let result_rk45 = time_integration("RK45 method", || {
        solve_ivp(
            van_der_pol,
            [0.0, 20.0],
            array![2.0, 0.0], // Start with displacement but no velocity
            Some(ODEOptions {
                method: ODEMethod::RK45,
                rtol: 1e-4,
                atol: 1e-6,
                ..Default::default()
            }),
        )
        .unwrap()
    });

    println!("  Steps:      {}", result_rk45.n_steps);
    println!("  Accepted steps: {}", result_rk45.n_accepted);
    println!("  Rejected steps: {}", result_rk45.n_rejected);
    println!("  Function evaluations: {}", result_rk45.n_eval);
    println!(
        "  Final values: [{}, {}]",
        result_rk45.y.last().unwrap()[0],
        result_rk45.y.last().unwrap()[1]
    );

    // Now solve with BDF (best for stiff problems)
    println!("\nNow solving with BDF (optimal for stiff equations):");
    let result_bdf = time_integration("BDF method", || {
        solve_ivp(
            van_der_pol,
            [0.0, 20.0],
            array![2.0, 0.0], // Start with displacement but no velocity
            Some(ODEOptions {
                method: ODEMethod::Bdf,
                max_order: Some(2), // BDF2 is a good balance of stability and accuracy
                rtol: 1e-4,
                atol: 1e-6,
                ..Default::default()
            }),
        )
        .unwrap()
    });

    println!("  Steps:      {}", result_bdf.n_steps);
    println!("  Accepted steps: {}", result_bdf.n_accepted);
    println!("  Rejected steps: {}", result_bdf.n_rejected);
    println!("  Function evaluations: {}", result_bdf.n_eval);
    println!(
        "  Final values: [{}, {}]",
        result_bdf.y.last().unwrap()[0],
        result_bdf.y.last().unwrap()[1]
    );

    // Print additional information about Newton iterations if available
    if let Some(msg) = &result_bdf.message {
        println!("  {}", msg);
    }

    println!("\nComparison of efficiency for stiff problem:");
    println!("  RK23: {} function evaluations", result_rk23.n_eval);
    println!("  RK45: {} function evaluations", result_rk45.n_eval);
    println!("  BDF:  {} function evaluations", result_bdf.n_eval);
    println!(
        "  RK45/RK23 ratio: {:.2}x",
        result_rk45.n_eval as f64 / result_rk23.n_eval as f64
    );
    println!(
        "  RK45/BDF ratio: {:.2}x",
        result_rk45.n_eval as f64 / result_bdf.n_eval as f64
    );

    // Example 4: Extremely stiff problem - Robertson chemical reaction system
    println!("\nExample 4: Extremely stiff ODE - Robertson chemical reactions");

    // The Robertson problem - a very stiff system of chemical reactions
    // y0' = -0.04*y0 + 1e4*y1*y2
    // y1' = 0.04*y0 - 1e4*y1*y2 - 3e7*y1^2
    // y2' = 3e7*y1^2
    let robertson = |_t: f64, y: ndarray::ArrayView1<f64>| {
        array![
            -0.04 * y[0] + 1.0e4 * y[1] * y[2],
            0.04 * y[0] - 1.0e4 * y[1] * y[2] - 3.0e7 * y[1] * y[1],
            3.0e7 * y[1] * y[1]
        ]
    };

    // This problem is extremely stiff and benefits greatly from implicit methods
    println!("Solving with BDF (necessary for very stiff problems):");
    let result_bdf_rob = time_integration("BDF method", || {
        solve_ivp(
            robertson,
            [0.0, 40.0],
            array![1.0, 0.0, 0.0], // Initial concentrations
            Some(ODEOptions {
                method: ODEMethod::Bdf,
                max_order: Some(3), // Higher order for this challenging problem
                rtol: 1e-4,
                atol: 1e-8,      // Stricter absolute tolerance for small values
                max_steps: 1000, // Allow more steps if needed
                ..Default::default()
            }),
        )
        .unwrap()
    });

    println!("  Steps:      {}", result_bdf_rob.n_steps);
    println!("  Accepted steps: {}", result_bdf_rob.n_accepted);
    println!("  Rejected steps: {}", result_bdf_rob.n_rejected);
    println!("  Function evaluations: {}", result_bdf_rob.n_eval);
    println!(
        "  Final values: [{:.4e}, {:.4e}, {:.4e}]",
        result_bdf_rob.y.last().unwrap()[0],
        result_bdf_rob.y.last().unwrap()[1],
        result_bdf_rob.y.last().unwrap()[2]
    );

    if let Some(msg) = &result_bdf_rob.message {
        println!("  {}", msg);
    }

    // For comparison, try to solve with an explicit method (unlikely to succeed or very inefficient)
    println!("\nAttempting to solve with RK45 (likely to be extremely inefficient):");
    let result_rk45_rob = time_integration("RK45 method", || {
        solve_ivp(
            robertson,
            [0.0, 40.0],
            array![1.0, 0.0, 0.0],
            Some(ODEOptions {
                method: ODEMethod::RK45,
                rtol: 1e-4,
                atol: 1e-8,
                max_steps: 5000, // Allow many steps since this will be inefficient
                ..Default::default()
            }),
        )
        .unwrap_or_else(|e| {
            println!("  Error: {}", e);
            // Return a dummy result
            scirs2_integrate::ODEResult {
                t: vec![0.0],
                y: vec![array![1.0, 0.0, 0.0]],
                n_steps: 0,
                n_eval: 0,
                n_accepted: 0,
                n_rejected: 0,
                n_jac: 0,
                n_lu: 0,
                success: false,
                message: Some("Failed to solve".to_string()),
                method: ODEMethod::RK45,
            }
        })
    });

    if result_rk45_rob.success {
        println!("  Steps:      {}", result_rk45_rob.n_steps);
        println!("  Accepted steps: {}", result_rk45_rob.n_accepted);
        println!("  Rejected steps: {}", result_rk45_rob.n_rejected);
        println!("  Function evaluations: {}", result_rk45_rob.n_eval);
        println!(
            "  Final values: [{:.4e}, {:.4e}, {:.4e}]",
            result_rk45_rob.y.last().unwrap()[0],
            result_rk45_rob.y.last().unwrap()[1],
            result_rk45_rob.y.last().unwrap()[2]
        );

        println!("\nComparison of efficiency for extremely stiff problem:");
        println!("  BDF:  {} function evaluations", result_bdf_rob.n_eval);
        println!("  RK45: {} function evaluations", result_rk45_rob.n_eval);
        println!(
            "  RK45/BDF ratio: {:.2}x",
            result_rk45_rob.n_eval as f64 / result_bdf_rob.n_eval as f64
        );
    } else {
        println!("  RK45 failed to solve the problem (as expected for very stiff ODEs)");
        println!("  This demonstrates why implicit methods like BDF are essential");
        println!("  for extremely stiff problems.");
    }
}
