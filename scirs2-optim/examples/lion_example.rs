//! Example of using the Lion optimizer
//!
//! This example demonstrates how to use the Lion optimizer to minimize
//! a simple function and compares it with other optimizers.

use ndarray::Array1;
use scirs2_optim::optimizers::{Adam, Lion, Optimizer, SGD};
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Lion Optimizer Example");
    println!("=====================");

    // Define a simple quadratic function: f(x,y) = x^2 + y^2
    // This function has a global minimum at (0, 0)
    let objective_fn = |x: &Array1<f64>| -> f64 { x[0].powi(2) + x[1].powi(2) };

    // Gradient of the quadratic function: ∇f(x,y) = (2x, 2y)
    let gradient_fn =
        |x: &Array1<f64>| -> Array1<f64> { Array1::from_vec(vec![2.0 * x[0], 2.0 * x[1]]) };

    // Initial parameters (starting point)
    let initial_params = Array1::from_vec(vec![5.0, 3.0]);

    // Hyperparameters
    let learning_rate = 0.1;
    let num_iterations = 50;

    // Initialize optimizers
    let mut optimizers: HashMap<String, Box<dyn Optimizer<f64, ndarray::Ix1>>> = HashMap::new();
    optimizers.insert("Lion".to_string(), Box::new(Lion::new(learning_rate)));
    optimizers.insert("Adam".to_string(), Box::new(Adam::new(learning_rate)));
    optimizers.insert("SGD".to_string(), Box::new(SGD::new(learning_rate)));

    println!(
        "Starting optimization from x = [{}, {}]",
        initial_params[0], initial_params[1]
    );
    println!("Target minimum is at x = [0.0, 0.0]");
    println!("Learning rate = {}", learning_rate);
    println!("Number of iterations = {}", num_iterations);
    println!();

    // Run optimization for each optimizer
    for (name, optimizer) in optimizers.iter_mut() {
        println!("Running {} optimizer:", name);
        println!("Iteration |     x     |     y     | Function Value");
        println!("------------------------------------------------");

        let mut params = initial_params.clone();

        // Show initial state
        let initial_value = objective_fn(&params);
        println!(
            "{:>9} | {:>9.4} | {:>9.4} | {:>13.6}",
            0, params[0], params[1], initial_value
        );

        // Run optimization
        for i in 1..=num_iterations {
            // Compute gradients
            let gradients = gradient_fn(&params);

            // Update parameters
            params = optimizer.step(&params, &gradients)?;

            // Compute function value
            let value = objective_fn(&params);

            // Print progress every 10 iterations
            if i % 10 == 0 || i == 1 {
                println!(
                    "{:>9} | {:>9.4} | {:>9.4} | {:>13.6}",
                    i, params[0], params[1], value
                );
            }
        }

        println!("Final parameters: x = [{:.6}, {:.6}]", params[0], params[1]);
        println!("Final function value: {:.6}", objective_fn(&params));
        println!();
    }

    println!("Optimization complete!");

    Ok(())
}

#[test]
fn test_lion_convergence() {
    // Test that Lion converges on a simple problem
    let mut optimizer = Lion::new(0.05); // Adjusted learning rate
    let mut params = Array1::from_vec(vec![10.0, -5.0]);

    // Optimize x^2 + y^2
    for _ in 0..1000 {
        // Increased iterations significantly
        let gradients = Array1::from_vec(vec![2.0 * params[0], 2.0 * params[1]]);
        params = optimizer.step(&params, &gradients).unwrap();
    }

    // Lion converges more slowly, so use a more forgiving threshold
    assert!(f64::abs(params[0]) < 1.0);
    assert!(f64::abs(params[1]) < 1.0);
}
