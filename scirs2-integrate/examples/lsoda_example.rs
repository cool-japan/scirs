use ndarray::{array, ArrayView1};
use scirs2_integrate::ode::{solve_ivp, ODEMethod, ODEOptions};

fn main() {
    println!("LSODA Solver Example - Automatic Stiffness Detection");
    println!("--------------------------------------------------");
    println!("NOTE: The LSODA implementation is still experimental.");
    println!("      This example will run a series of tests with different parameters");
    println!("      to demonstrate when LSODA works well and when it struggles.");

    // First, demonstrate a simple case where LSODA works well
    println!("\nExponential Decay Example (Simple, Non-stiff)");
    println!("   y' = -y, y(0) = 1, exact solution: y(t) = exp(-t)");

    // Simple exponential decay
    let decay = |_t: f64, y: ArrayView1<f64>| array![-y[0]];

    // Use LSODA with large initial step size, which works well
    let result = solve_ivp(
        decay,
        [0.0, 5.0],
        array![1.0],
        Some(ODEOptions {
            method: ODEMethod::LSODA,
            rtol: 1e-4,
            atol: 1e-6,
            max_steps: 1000,
            h0: Some(0.1),        // Larger initial step for stability
            min_step: Some(1e-3), // Reasonable minimum step
            ..Default::default()
        }),
    )
    .unwrap_or_else(|e| {
        println!("LSODA method failed unexpectedly on simple problem: {}.", e);
        println!("Using DOP853 as fallback...");
        solve_ivp(decay, [0.0, 5.0], array![1.0], None).unwrap()
    });

    // Print statistics
    println!("Results for exponential decay:");
    println!("  Solver method: {:?}", result.method);
    println!(
        "  Steps: {}, Function evaluations: {}",
        result.n_steps, result.n_eval
    );
    println!(
        "  Final value: {:.6}, Exact: {:.6}",
        result.y.last().unwrap()[0],
        (-5.0f64).exp()
    );
    println!(
        "  Error: {:.2e}",
        (result.y.last().unwrap()[0] - (-5.0f64).exp()).abs()
    );

    if let Some(msg) = &result.message {
        println!("  {}", msg);
    }

    // Now demonstrate the Van der Pol oscillator with different stiffness parameters
    println!("\nVan der Pol Oscillator with Varying Mu Parameter");
    println!("y'' - μ(1-y²)y' + y = 0");
    println!("As system: y₀' = y₁, y₁' = μ(1-y₀²)y₁ - y₀");
    println!("This problem changes from non-stiff to stiff as μ increases");

    // Define an array of mu values to test (increasing stiffness)
    // Using a smaller set to avoid excessive output
    let mu_values = [1.0, 10.0, 100.0];

    for &mu in &mu_values {
        println!("\nSolving with μ = {}", mu);

        // Define the Van der Pol oscillator with the current mu value
        let van_der_pol = move |_t: f64, y: ArrayView1<f64>| {
            array![y[1], mu * (1.0 - y[0].powi(2)) * y[1] - y[0]]
        };

        // Use LSODA with parameters that work better for this problem
        let result = solve_ivp(
            van_der_pol,
            [0.0, 10.0], // Shorter time interval for demonstration
            array![2.0, 0.0],
            Some(ODEOptions {
                method: ODEMethod::LSODA,
                rtol: 1e-3,             // Slightly looser tolerance
                atol: 1e-5,             // Slightly looser tolerance
                max_steps: 3000,        // More steps for complex problems
                h0: Some(0.1),          // Larger initial step size for stability
                min_step: Some(0.0001), // Small enough to work but not too small
                ..Default::default()
            }),
        )
        .unwrap_or_else(|e| {
            println!("LSODA method failed with error: {}.", e);
            println!("Note: This is expected as LSODA implementation is still experimental.");
            println!("Using a more stable solver (DOP853) instead...");

            solve_ivp(
                van_der_pol,
                [0.0, 10.0],
                array![2.0, 0.0],
                Some(ODEOptions {
                    method: ODEMethod::DOP853,
                    rtol: 1e-4,
                    atol: 1e-6,
                    max_steps: 5000,
                    ..Default::default()
                }),
            )
            .unwrap()
        });

        // Print statistics about the solution
        println!("  Solver method: {:?}", result.method);
        println!("  Steps: {}", result.n_steps);
        println!("  Function evaluations: {}", result.n_eval);
        println!("  Accepted steps: {}", result.n_accepted);
        println!("  Rejected steps: {}", result.n_rejected);
        println!(
            "  Final state: [{:.4}, {:.4}]",
            result.y.last().unwrap()[0],
            result.y.last().unwrap()[1]
        );
        println!("  Success: {}", result.success);

        // LSODA now returns method switching statistics in the message
        if let Some(msg) = &result.message {
            println!("  {}", msg);
        }
    }

    println!("\nLSODA Method Analysis:");
    println!("- LSODA automatically switches between Adams method (for non-stiff regions)");
    println!("  and BDF method (for stiff regions) as needed during integration");
    println!("- For simple problems like exponential decay, LSODA works well");
    println!("- For complex problems, current implementation may struggle");
    println!("- Larger initial step sizes (0.1) tend to work better than small ones");
    println!("- Method switching should be visible in the statistics for stiff problems");
    println!("\nRecommendations for using LSODA:");
    println!("1. Start with larger initial step sizes (h0 ~ 0.1)");
    println!("2. Use larger minimum step sizes (min_step ~ 1e-3)");
    println!("3. For stiff problems, consider using BDF directly instead");
}
