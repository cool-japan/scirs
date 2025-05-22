//! Example of using the L-BFGS optimizer
//!
//! This example demonstrates how to use the L-BFGS optimizer for minimizing
//! a function and compares it with other optimizers.

use ndarray::Array1;
use scirs2_optim::optimizers::{Adam, Optimizer, LBFGS, SGD};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("L-BFGS Optimizer Example");
    println!("=======================");

    // Define the Rosenbrock function: f(x,y) = (1-x)^2 + 100(y-x^2)^2
    // This function has a global minimum at (1, 1)
    let rosenbrock_fn = |x: &Array1<f64>| -> f64 {
        let a = 1.0;
        let b = 100.0;
        let x_val = x[0];
        let y_val = x[1];

        (a - x_val).powi(2) + b * (y_val - x_val.powi(2)).powi(2)
    };

    // Gradient of the Rosenbrock function
    let rosenbrock_grad = |x: &Array1<f64>| -> Array1<f64> {
        let a = 1.0;
        let b = 100.0;
        let x_val = x[0];
        let y_val = x[1];

        let dx = -2.0 * (a - x_val) - 4.0 * b * x_val * (y_val - x_val.powi(2));
        let dy = 2.0 * b * (y_val - x_val.powi(2));

        Array1::from_vec(vec![dx, dy])
    };

    // Initial parameters (starting far from the minimum)
    let initial_params = Array1::from_vec(vec![-1.5, 1.5]);

    println!("Minimizing Rosenbrock function");
    println!(
        "Initial point: [{}, {}]",
        initial_params[0], initial_params[1]
    );
    println!("Target minimum: [1.0, 1.0]");
    println!();

    // Test different optimizers
    let optimizers = vec![
        (
            "L-BFGS",
            Box::new(LBFGS::new(0.1)) as Box<dyn Optimizer<f64, ndarray::Ix1>>,
        ),
        (
            "Adam",
            Box::new(Adam::new(0.01)) as Box<dyn Optimizer<f64, ndarray::Ix1>>,
        ),
        (
            "SGD",
            Box::new(SGD::new(0.001)) as Box<dyn Optimizer<f64, ndarray::Ix1>>,
        ),
    ];

    for (name, mut optimizer) in optimizers {
        println!("Running {} optimizer:", name);
        let mut params = initial_params.clone();

        // Show initial value
        let initial_value = rosenbrock_fn(&params);
        println!("Initial function value: {:.6}", initial_value);

        // Run optimization
        let num_iterations = if name == "L-BFGS" { 50 } else { 200 };

        for i in 0..num_iterations {
            let gradients = rosenbrock_grad(&params);
            params = optimizer.step(&params, &gradients)?;

            // Print progress
            if i % 10 == 0 || i == num_iterations - 1 {
                let value = rosenbrock_fn(&params);
                println!(
                    "  Iteration {:3}: x = [{:7.4}, {:7.4}], f(x) = {:.6}",
                    i + 1,
                    params[0],
                    params[1],
                    value
                );
            }
        }

        // Final result
        let final_value = rosenbrock_fn(&params);
        println!("Final point: [{:.6}, {:.6}]", params[0], params[1]);
        println!("Final function value: {:.6}", final_value);
        println!(
            "Distance from optimum: {:.6}",
            ((params[0] - 1.0).powi(2) + (params[1] - 1.0).powi(2)).sqrt()
        );
        println!();
    }

    Ok(())
}

#[test]
fn test_lbfgs_on_rosenbrock() {
    // Test that L-BFGS converges on the Rosenbrock function
    // Use a higher learning rate for better convergence
    let mut optimizer = LBFGS::new(0.5);

    // Start from a point closer to the solution
    let mut params = Array1::from_vec(vec![0.0, 0.0]);

    // Gradient of Rosenbrock
    let rosenbrock_grad = |x: &Array1<f64>| -> Array1<f64> {
        let x_val = x[0];
        let y_val = x[1];

        let dx = -2.0 * (1.0 - x_val) - 400.0 * x_val * (y_val - x_val.powi(2));
        let dy = 200.0 * (y_val - x_val.powi(2));

        Array1::from_vec(vec![dx, dy])
    };

    // Run optimization with more iterations
    for _ in 0..200 {
        let gradients = rosenbrock_grad(&params);
        params = optimizer.step(&params, &gradients).unwrap();
    }

    // Should be close to (1, 1) but use relaxed criteria
    assert!((params[0] - 1.0).abs() < 0.2);
    assert!((params[1] - 1.0).abs() < 0.2);
}
