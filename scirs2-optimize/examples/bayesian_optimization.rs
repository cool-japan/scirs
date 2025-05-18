//! Example of using Bayesian optimization
//!
//! This example demonstrates how to use Bayesian optimization for global optimization
//! of expensive-to-evaluate functions.

use ndarray::ArrayView1;
use scirs2_optimize::global::{
    bayesian_optimization, AcquisitionFunctionType, BayesianOptimizationOptions, BayesianOptimizer,
    InitialPointGenerator, KernelType, Parameter, Space,
};
use std::time::Instant;

// Define a moderately expensive function to optimize
fn branin(x: &ArrayView1<f64>) -> f64 {
    // Simulate a computationally expensive function
    std::thread::sleep(std::time::Duration::from_millis(10));

    let x1 = x[0];
    let x2 = x[1];

    let t1 = x2 - 5.1 / (4.0 * std::f64::consts::PI).powi(2) * x1.powi(2)
        + 5.0 / std::f64::consts::PI * x1
        - 6.0;
    let t2 = 10.0 * (1.0 - 1.0 / (8.0 * std::f64::consts::PI)) * (x2 - 0.5);

    t1.powi(2) + t2.cos() + 10.0
}

// Function with multiple local minima
fn himmelblau(x: &ArrayView1<f64>) -> f64 {
    let x1 = x[0];
    let x2 = x[1];

    (x1.powi(2) + x2 - 11.0).powi(2) + (x1 + x2.powi(2) - 7.0).powi(2)
}

fn main() {
    println!("Bayesian Optimization Example");
    println!("-----------------------------");

    // Example 1: Using the simple API
    println!("\nExample 1: Simple API with Branin function");
    let bounds = vec![(-5.0, 10.0), (0.0, 15.0)];

    let options = BayesianOptimizationOptions {
        n_initial_points: 5,
        acq_func: AcquisitionFunctionType::ExpectedImprovement,
        seed: Some(42),
        ..Default::default()
    };

    let start = Instant::now();
    let result = bayesian_optimization(branin, bounds, 20, Some(options)).unwrap();
    let duration = start.elapsed();

    println!("Best parameters: [{:.4}, {:.4}]", result.x[0], result.x[1]);
    println!("Best value: {:.4}", result.fun);
    println!("Number of function evaluations: {}", result.nfev);
    println!("Time elapsed: {:?}", duration);

    // Example 2: Using the builder API
    println!("\nExample 2: Builder API with Himmelblau function");

    let space = Space::new()
        .add("x1", Parameter::Real(-5.0, 5.0))
        .add("x2", Parameter::Real(-5.0, 5.0));

    let options = BayesianOptimizationOptions {
        n_initial_points: 8,
        initial_point_generator: InitialPointGenerator::Random,
        acq_func: AcquisitionFunctionType::LowerConfidenceBound,
        kernel: KernelType::SquaredExponential,
        kappa: 2.5, // Higher exploration parameter
        seed: Some(123),
        n_restarts: 10,
        ..Default::default()
    };

    let mut optimizer = BayesianOptimizer::new(space, Some(options));

    let start = Instant::now();
    let result = optimizer.optimize(himmelblau, 30);
    let duration = start.elapsed();

    println!("Best parameters: [{:.4}, {:.4}]", result.x[0], result.x[1]);
    println!("Best value: {:.4}", result.fun);
    println!("Number of function evaluations: {}", result.nfev);
    println!("Time elapsed: {:?}", duration);

    // Example 3: Ask-Tell interface
    println!("\nExample 3: Ask-Tell interface");

    let space = Space::new()
        .add("x1", Parameter::Real(-5.0, 5.0))
        .add("x2", Parameter::Real(-5.0, 5.0));

    let options = BayesianOptimizationOptions {
        n_initial_points: 5,
        acq_func: AcquisitionFunctionType::ProbabilityOfImprovement,
        xi: 0.05, // Higher exploitation parameter
        seed: Some(456),
        ..Default::default()
    };

    let mut optimizer = BayesianOptimizer::new(space, Some(options));

    // Manual optimization loop
    for i in 0..15 {
        // Ask for the next point to evaluate
        let x = optimizer.ask();

        // Evaluate the objective function
        let y = himmelblau(&x.view());

        // Tell the optimizer about the result
        optimizer.tell(x.clone(), y);

        println!(
            "Iteration {}: x = [{:.4}, {:.4}], y = {:.4}",
            i + 1,
            x[0],
            x[1],
            y
        );
    }

    // Get the best result
    let result = optimizer.optimize(|_| 0.0, 0); // Just to get the final result
    println!(
        "\nBest parameters: [{:.4}, {:.4}]",
        result.x[0], result.x[1]
    );
    println!("Best value: {:.4}", result.fun);
}
