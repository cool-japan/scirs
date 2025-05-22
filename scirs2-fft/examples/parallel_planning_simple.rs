// Simple Parallel FFT Planning Example
//
// This is a simplified example showing how to use the parallel planning features
// to create and execute FFT plans in parallel.

use num_complex::Complex64;
use scirs2_fft::{ParallelExecutor, ParallelPlanner, ParallelPlanningConfig};
use std::time::Instant;

fn main() {
    println!("Simple Parallel FFT Planning Example");
    println!("===================================\n");

    // Create a parallel planner with custom configuration
    let config = ParallelPlanningConfig {
        // Lower the threshold to demonstrate parallel planning on smaller FFTs
        parallel_threshold: 512,
        // Use all available threads
        max_threads: None,
        // Enable parallel execution
        parallel_execution: true,
        // Use default settings for everything else
        ..Default::default()
    };

    // Create the planner
    let planner = ParallelPlanner::new(Some(config.clone()));

    // Define FFT sizes we want to plan for
    let fft_sizes = [
        vec![1024],          // 1D FFT of size 1024
        vec![512, 512],      // 2D FFT of size 512x512
        vec![128, 128, 128], // 3D FFT of size 128x128x128
        vec![2048],          // 1D FFT of size 2048
    ];

    // Convert to the format expected by plan_multiple
    let plan_specs: Vec<_> = fft_sizes
        .iter()
        .map(|shape| (shape.clone(), true, Default::default()))
        .collect();

    // Create multiple plans in parallel
    println!("Creating multiple FFT plans in parallel...");
    let start = Instant::now();
    let results = planner.plan_multiple(&plan_specs).unwrap();
    let elapsed = start.elapsed();

    println!("Created {} plans in {:?}", results.len(), elapsed);

    // Print plan details
    for (i, result) in results.iter().enumerate() {
        println!(
            "Plan {}: shape={:?}, created by thread {} in {:?}",
            i, result.shape, result.thread_id, result.creation_time
        );
    }

    // Use the first plan for execution
    println!("\nExecuting FFT using the first plan...");
    let plan = &results[0].plan;
    let executor = ParallelExecutor::new(plan.clone(), Some(config));

    // Create input data
    let size = plan.shape().iter().product::<usize>();
    let input = create_signal(size);
    let mut output = vec![Complex64::default(); size];

    // Execute the plan
    let start = Instant::now();
    executor.execute(&input, &mut output).unwrap();
    let elapsed = start.elapsed();

    println!("FFT of size {} executed in {:?}", size, elapsed);
    println!("First few output values: {:?}", &output[..4]);
}

// Create a simple test signal
fn create_signal(size: usize) -> Vec<Complex64> {
    (0..size)
        .map(|i| {
            let x = i as f64 / size as f64;
            // Create a sinusoidal signal with a few frequencies
            let val = (2.0 * std::f64::consts::PI * x * 5.0).sin()
                + 0.5 * (2.0 * std::f64::consts::PI * x * 10.0).sin();
            Complex64::new(val, 0.0)
        })
        .collect()
}
