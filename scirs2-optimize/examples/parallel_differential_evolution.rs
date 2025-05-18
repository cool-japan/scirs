//! Example of using differential evolution with parallel computation
//!
//! This example demonstrates how to enable parallel evaluation for differential evolution,
//! which can significantly speed up optimization for expensive objective functions.

use ndarray::ArrayView1;
use scirs2_optimize::{
    global::{differential_evolution, DifferentialEvolutionOptions},
    parallel::ParallelOptions,
};
use std::time::Instant;

// A computationally expensive version of the Rosenbrock function
fn expensive_rosenbrock(x: &ArrayView1<f64>) -> f64 {
    // Simulate expensive computation
    std::thread::sleep(std::time::Duration::from_millis(10));

    let a = 1.0;
    let b = 100.0;
    let x0 = x[0];
    let x1 = x[1];
    (a - x0).powi(2) + b * (x1 - x0.powi(2)).powi(2)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Define bounds for the variables
    let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];

    // Test sequential execution
    println!("Running differential evolution sequentially...");
    let mut options = DifferentialEvolutionOptions::default();
    options.popsize = 10; // Keep population small for demo
    options.maxiter = 5;
    options.seed = Some(42);

    let start = Instant::now();
    let sequential_result = differential_evolution(
        expensive_rosenbrock,
        bounds.clone(),
        Some(options.clone()),
        None,
    )?;
    let sequential_time = start.elapsed();

    println!(
        "Sequential result: x = {:?}, f(x) = {}",
        sequential_result.x.as_slice().unwrap(),
        sequential_result.fun
    );
    println!("Sequential time: {:?}", sequential_time);
    println!("Function evaluations: {}", sequential_result.nfev);

    // Test parallel execution
    println!("\nRunning differential evolution in parallel...");
    options.parallel = Some(ParallelOptions {
        num_workers: None, // Use all available cores
        min_parallel_size: 4,
        chunk_size: 1,
        parallel_evaluations: true,
        parallel_gradient: true,
    });

    let start = Instant::now();
    let parallel_result =
        differential_evolution(expensive_rosenbrock, bounds, Some(options), None)?;
    let parallel_time = start.elapsed();

    println!(
        "Parallel result: x = {:?}, f(x) = {}",
        parallel_result.x.as_slice().unwrap(),
        parallel_result.fun
    );
    println!("Parallel time: {:?}", parallel_time);
    println!("Function evaluations: {}", parallel_result.nfev);

    // Calculate speedup
    let speedup = sequential_time.as_secs_f64() / parallel_time.as_secs_f64();
    println!("\nSpeedup: {:.2}x", speedup);

    Ok(())
}
