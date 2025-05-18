use ndarray::{array, ArrayView1};
use scirs2_integrate::ode::{solve_ivp, ODEMethod, ODEOptions};

fn main() {
    println!("LSODA Solver Example - Simple Test Problem");
    println!("------------------------------------------");

    // Simple exponential decay: y' = -0.5 * y
    // This is a very simple non-stiff problem
    let simple_decay = |_t: f64, y: ArrayView1<f64>| array![-0.5 * y[0]];

    // Initial condition y(0) = 1.0
    let result = solve_ivp(
        simple_decay,
        [0.0, 10.0],
        array![1.0],
        Some(ODEOptions {
            method: ODEMethod::LSODA,
            rtol: 1e-4,
            atol: 1e-6,
            max_steps: 500,
            h0: Some(0.1),
            min_step: Some(1e-3),
            ..Default::default()
        }),
    )
    .unwrap_or_else(|e| {
        println!("LSODA method failed with error: {}.", e);
        println!("This is concerning as this problem is very simple.");

        // Fall back to DOP853
        solve_ivp(
            simple_decay,
            [0.0, 10.0],
            array![1.0],
            Some(ODEOptions {
                method: ODEMethod::DOP853,
                rtol: 1e-4,
                atol: 1e-6,
                max_steps: 100,
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
    println!("  Success: {}", result.success);

    // Print the solution at several time points
    println!("\nSolution (analytical solution is y(t) = exp(-0.5*t)):");
    println!("  t      y         exp(-0.5*t)  error");
    println!("  --------------------------------------");

    for i in 0..result.t.len().min(10) {
        let t = result.t[i];
        let y = result.y[i][0];
        let exact = (-0.5 * t).exp();
        let error = (y - exact).abs();
        println!("  {:.2}    {:.6}   {:.6}     {:.8}", t, y, exact, error);
    }

    // If there are more than 10 time points, print the last one
    if result.t.len() > 10 {
        let i = result.t.len() - 1;
        let t = result.t[i];
        let y = result.y[i][0];
        let exact = (-0.5 * t).exp();
        let error = (y - exact).abs();
        println!("  ...    ...       ...         ...");
        println!("  {:.2}    {:.6}   {:.6}     {:.8}", t, y, exact, error);
    }

    // Print message if it exists
    if let Some(msg) = &result.message {
        println!("\n  {}", msg);
    }
}
