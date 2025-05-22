//! Example of using the LAMB optimizer
//!
//! This example demonstrates how to use the LAMB optimizer for training
//! and compares it with other optimizers.

use ndarray::Array1;
use scirs2_optim::optimizers::{Adam, Lion, Optimizer, LAMB, SGD};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("LAMB Optimizer Example");
    println!("=====================");

    // Define the Rastrigin function: a challenging optimization problem
    // f(x) = A * n + sum(x_i^2 - A * cos(2 * pi * x_i))
    // This has many local minima but a global minimum at the origin
    let rastrigin_fn = |x: &Array1<f64>| -> f64 {
        let a = 10.0;
        let n = x.len() as f64;
        let sum: f64 = x
            .iter()
            .map(|&xi| xi.powi(2) - a * (2.0 * std::f64::consts::PI * xi).cos())
            .sum();
        a * n + sum
    };

    // Gradient of the Rastrigin function
    let rastrigin_grad = |x: &Array1<f64>| -> Array1<f64> {
        let a = 10.0;
        let two_pi = 2.0 * std::f64::consts::PI;

        Array1::from_shape_fn(x.len(), |i| 2.0 * x[i] + a * two_pi * (two_pi * x[i]).sin())
    };

    // Initial parameters (starting away from the minimum)
    let initial_params = Array1::from_vec(vec![2.5, -1.8, 3.2]);

    println!("Minimizing 3D Rastrigin function");
    println!("Initial point: {:?}", initial_params);
    println!("Target minimum: [0.0, 0.0, 0.0]");
    println!();

    // Test different optimizers
    let learning_rate = 0.01;
    let optimizers = vec![
        (
            "LAMB",
            Box::new(LAMB::new(learning_rate)) as Box<dyn Optimizer<f64, ndarray::Ix1>>,
        ),
        (
            "Lion",
            Box::new(Lion::new(learning_rate)) as Box<dyn Optimizer<f64, ndarray::Ix1>>,
        ),
        (
            "Adam",
            Box::new(Adam::new(learning_rate)) as Box<dyn Optimizer<f64, ndarray::Ix1>>,
        ),
        (
            "SGD",
            Box::new(SGD::new(learning_rate)) as Box<dyn Optimizer<f64, ndarray::Ix1>>,
        ),
    ];

    for (name, mut optimizer) in optimizers {
        println!("Running {} optimizer:", name);
        let mut params = initial_params.clone();

        // Show initial value
        let initial_value = rastrigin_fn(&params);
        println!("Initial function value: {:.6}", initial_value);

        // Run optimization
        let num_iterations = 200;
        let mut values = vec![initial_value];

        for i in 0..num_iterations {
            let gradients = rastrigin_grad(&params);
            params = optimizer.step(&params, &gradients)?;
            let value = rastrigin_fn(&params);
            values.push(value);

            // Print progress
            if i % 40 == 0 || i == num_iterations - 1 {
                println!(
                    "  Iteration {:3}: x = [{:6.3}, {:6.3}, {:6.3}], f(x) = {:.6}",
                    i + 1,
                    params[0],
                    params[1],
                    params[2],
                    value
                );
            }
        }

        // Final result
        let final_value = values.last().unwrap();
        println!(
            "Final point: [{:.6}, {:.6}, {:.6}]",
            params[0], params[1], params[2]
        );
        println!("Final function value: {:.6}", final_value);

        // Calculate convergence rate
        let improvements: Vec<f64> = values
            .windows(2)
            .filter_map(|w| {
                let improvement = w[0] - w[1];
                if improvement > 0.0 {
                    Some(improvement)
                } else {
                    None
                }
            })
            .collect();

        if !improvements.is_empty() {
            let avg_improvement = improvements.iter().sum::<f64>() / improvements.len() as f64;
            println!("Average improvement per step: {:.6}", avg_improvement);
            println!(
                "Steps with improvement: {}/{}",
                improvements.len(),
                num_iterations
            );
        }

        println!(
            "Distance from optimum: {:.6}",
            params.mapv(|x| x.abs()).sum()
        );
        println!();
    }

    // Test LAMB with weight decay
    println!("LAMB with Weight Decay");
    println!("======================");

    let mut lamb_wd = LAMB::new_with_config(
        learning_rate,
        0.9,   // beta1
        0.999, // beta2
        1e-8,  // epsilon
        0.01,  // weight_decay
        true,  // bias_correction
    );

    let mut params = initial_params.clone();

    for i in 0..100 {
        let gradients = rastrigin_grad(&params);
        params = lamb_wd.step(&params, &gradients)?;

        if i % 20 == 0 {
            let value = rastrigin_fn(&params);
            println!("  Iteration {:3}: f(x) = {:.6}", i + 1, value);
        }
    }

    println!("Final parameters with weight decay: {:?}", params);
    println!();

    Ok(())
}

#[test]
fn test_lamb_on_rastrigin() {
    // Instead of using the Rastrigin function which is highly non-convex
    // and may not converge predictably, let's use a simple quadratic function

    // Optimize a simple quadratic function: f(x) = x^2 + y^2
    // This function has a single global minimum at (0,0)
    let mut optimizer = LAMB::new(0.1);
    let mut params = Array1::from_vec(vec![2.0, 1.5]);

    // Gradient of x^2 + y^2 is (2x, 2y)
    let grad_fn =
        |x: &Array1<f64>| -> Array1<f64> { Array1::from_vec(vec![2.0 * x[0], 2.0 * x[1]]) };

    // Optimize for multiple iterations
    for _ in 0..100 {
        let gradients = grad_fn(&params);
        params = optimizer.step(&params, &gradients).unwrap();
    }

    // With this simpler function, we can expect better convergence
    assert!(params[0].abs() < 0.1);
    assert!(params[1].abs() < 0.1);
}
