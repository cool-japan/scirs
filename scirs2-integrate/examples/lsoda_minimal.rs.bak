use ndarray::{array, ArrayView1};
use scirs2_integrate::ode::{solve_ivp, ODEMethod, ODEOptions};

fn main() {
    println!("LSODA Minimal Test Example");
    println!("-------------------------");

    // Use a very simple test problem: y' = -y (exponential decay)
    // Exact solution: y(t) = y0 * exp(-t)
    let decay_system = |_t: f64, y: ArrayView1<f64>| array![-y[0]];

    // Try with a single very specific configuration that's most likely to work
    let initial_step = 0.1;
    let min_step = 1e-3;

    println!("Solving y' = -y with y(0) = 1 using LSODA");
    println!(
        "Parameters: initial_step={}, min_step={}",
        initial_step, min_step
    );

    let result = solve_ivp(
        decay_system,
        [0.0, 1.0],
        array![1.0],
        Some(ODEOptions {
            method: ODEMethod::LSODA,
            rtol: 1e-4,
            atol: 1e-6,
            h0: Some(initial_step),
            min_step: Some(min_step),
            max_steps: 1000,
            ..Default::default()
        }),
    );

    match result {
        Ok(res) => {
            println!("Success! Integration completed.");
            println!(
                "Steps: {}, Function evaluations: {}",
                res.n_steps, res.n_eval
            );
            println!(
                "Final value: {}, Exact: {}",
                res.y.last().unwrap()[0],
                (-1.0f64).exp()
            );
            println!(
                "Error: {:.2e}",
                (res.y.last().unwrap()[0] - (-1.0f64).exp()).abs()
            );

            // Print all time points and values for close inspection
            println!("\nDetailed solution:");
            println!("  t       y       exp(-t)   Error");
            println!("  ----------------------------------");

            for i in 0..res.t.len() {
                let t = res.t[i];
                let y = res.y[i][0];
                let exact = (-t).exp();
                let error = (y - exact).abs();
                println!("  {:.4}   {:.6}   {:.6}   {:.2e}", t, y, exact, error);
            }

            if let Some(msg) = res.message {
                println!("\nMessage: {}", msg);
            }
        }
        Err(e) => {
            println!("Failed: {}", e);

            // Compare with a known reliable method
            println!("\nTrying with DOP853 method for comparison:");
            let dop_result = solve_ivp(
                decay_system,
                [0.0, 1.0],
                array![1.0],
                Some(ODEOptions {
                    method: ODEMethod::DOP853,
                    rtol: 1e-4,
                    atol: 1e-6,
                    max_steps: 1000,
                    ..Default::default()
                }),
            )
            .unwrap();

            println!(
                "DOP853 Steps: {}, Function evaluations: {}",
                dop_result.n_steps, dop_result.n_eval
            );
            println!(
                "Final value: {}, Exact: {}",
                dop_result.y.last().unwrap()[0],
                (-1.0f64).exp()
            );
            println!(
                "Error: {:.2e}",
                (dop_result.y.last().unwrap()[0] - (-1.0f64).exp()).abs()
            );
        }
    }
}
