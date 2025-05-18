use ndarray::{array, ArrayView1};
use scirs2_integrate::ode::{solve_ivp, ODEMethod, ODEOptions};

fn main() {
    println!("LSODA Solver Debug Example");
    println!("-------------------------");
    println!("This example is for debugging the LSODA implementation");
    println!("We'll try different step sizes and parameters to understand the failures");

    // Use a very simple test problem: y' = -y (exponential decay)
    let decay_system = |_t: f64, y: ArrayView1<f64>| array![-y[0]];

    // Try with different initial step sizes and min step thresholds
    let step_sizes = [0.1, 0.01, 0.001];
    let min_steps = [1e-3, 1e-4, 1e-5, 1e-6];

    for &initial_step in &step_sizes {
        for &min_step in &min_steps {
            println!(
                "\nTrying with initial_step={}, min_step={}",
                initial_step, min_step
            );

            let result = solve_ivp(
                decay_system,
                [0.0, 1.0], // Short time interval for testing
                array![1.0],
                Some(ODEOptions {
                    method: ODEMethod::LSODA,
                    rtol: 1e-4,
                    atol: 1e-6,
                    max_steps: 100,
                    h0: Some(initial_step),
                    min_step: Some(min_step),
                    ..Default::default()
                }),
            );

            match result {
                Ok(res) => {
                    println!(
                        "  Success! Steps: {}, Function evaluations: {}",
                        res.n_steps, res.n_eval
                    );
                    println!(
                        "  Final value: {}, Exact: {}",
                        res.y.last().unwrap()[0],
                        (-1.0f64).exp()
                    );
                    println!(
                        "  Error: {:.2e}",
                        (res.y.last().unwrap()[0] - (-1.0f64).exp()).abs()
                    );

                    if let Some(msg) = res.message {
                        println!("  Message: {}", msg);
                    }
                }
                Err(e) => {
                    println!("  Failed: {}", e);
                }
            }
        }
    }

    // Try a stiff problem with various parameters
    println!("\n\nTesting with stiff Van der Pol oscillator");
    let mu = 1000.0; // Very stiff
    let van_der_pol =
        |_t: f64, y: ArrayView1<f64>| array![y[1], mu * (1.0 - y[0].powi(2)) * y[1] - y[0]];

    // Try with different parameters
    let configs = [(0.01, 1e-4, 500), (0.001, 1e-5, 1000), (0.0001, 1e-6, 2000)];

    for &(initial_step, min_step, max_steps) in &configs {
        println!(
            "\nTrying Van der Pol with: initial_step={}, min_step={}, max_steps={}",
            initial_step, min_step, max_steps
        );

        let result = solve_ivp(
            van_der_pol,
            [0.0, 2.0], // Short interval for testing
            array![2.0, 0.0],
            Some(ODEOptions {
                method: ODEMethod::LSODA,
                rtol: 1e-4,
                atol: 1e-6,
                max_steps,
                h0: Some(initial_step),
                min_step: Some(min_step),
                ..Default::default()
            }),
        );

        match result {
            Ok(res) => {
                println!(
                    "  Success! Steps: {}, Function evaluations: {}",
                    res.n_steps, res.n_eval
                );
                println!(
                    "  Final state: [{:.4}, {:.4}]",
                    res.y.last().unwrap()[0],
                    res.y.last().unwrap()[1]
                );

                if let Some(msg) = res.message {
                    println!("  Message: {}", msg);
                }
            }
            Err(e) => {
                println!("  Failed: {}", e);
            }
        }
    }
}
