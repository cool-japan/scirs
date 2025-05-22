// Parallel FFT Planning Example
//
// This example demonstrates the parallel planning capabilities of scirs2-fft.
// It shows how to:
// 1. Create a parallel planner with custom configuration
// 2. Create individual plans for different FFT sizes
// 3. Create multiple plans in parallel and measure the performance gain
// 4. Execute FFTs in parallel using the created plans
// 5. Perform batch FFT execution for multiple inputs

use num_complex::Complex64;
// Import from the main API instead of the internal module
use scirs2_fft::{ParallelExecutor, ParallelPlanner, ParallelPlanningConfig};
use std::time::{Duration, Instant};

fn main() {
    println!("Parallel FFT Planning Example");
    println!("=============================\n");

    // Configure parallel planning
    let config = ParallelPlanningConfig {
        parallel_threshold: 512, // Lower threshold for demo purposes
        ..Default::default()
    };

    // Create a parallel planner
    let planner = ParallelPlanner::new(Some(config.clone()));

    // Plan creation for different FFT sizes
    println!("Creating plans of different sizes...");

    // Single plan creation
    let small_size = 256;
    let start = Instant::now();
    let _small_plan = planner
        .plan_fft(&[small_size], true, Default::default())
        .unwrap();
    println!(
        "Small FFT plan ({}) created in {:?} (processed serially)",
        small_size,
        start.elapsed()
    );

    // Large plan (should use parallel planning)
    let large_size = 4096;
    let start = Instant::now();
    let large_plan = planner
        .plan_fft(&[large_size], true, Default::default())
        .unwrap();
    println!(
        "Large FFT plan ({}) created in {:?} (using parallel planning)",
        large_size,
        start.elapsed()
    );

    // Demonstrate multiple plan creation in parallel
    println!("\nCreating multiple plans in parallel...");
    let sizes = vec![
        (vec![512], true, Default::default()),
        (vec![1024], true, Default::default()),
        (vec![2048], true, Default::default()),
        (vec![4096], true, Default::default()),
    ];

    let start = Instant::now();
    let results = planner.plan_multiple(&sizes).unwrap();
    let total_time = start.elapsed();

    println!("Created {} plans in {:?}:", results.len(), total_time);
    for (i, result) in results.iter().enumerate() {
        println!(
            "  Plan {}: size={:?}, created in {:?} by thread {}",
            i, result.shape, result.creation_time, result.thread_id
        );
    }

    // Sequential creation time (estimate)
    let sequential_time: Duration = results
        .iter()
        .map(|r| r.creation_time)
        .fold(Duration::from_nanos(0), |acc, time| acc + time);

    println!("\nTotal parallel time: {:?}", total_time);
    println!("Estimated sequential time: {:?}", sequential_time);
    println!(
        "Speedup: {:.2}x",
        sequential_time.as_secs_f64() / total_time.as_secs_f64()
    );

    // Demonstrate parallel execution
    println!("\nParallel FFT Execution:");

    // Create test data
    let input = create_test_signal(large_size);
    let mut output = vec![Complex64::default(); large_size];

    // Create executor
    let executor = ParallelExecutor::new(large_plan, Some(config.clone()));

    // Execute the FFT
    let start = Instant::now();
    executor.execute(&input, &mut output).unwrap();
    println!("Single large FFT executed in {:?}", start.elapsed());

    // Batch execution
    println!("\nBatch FFT Execution:");
    let batch_size = 4;
    let mut inputs = Vec::with_capacity(batch_size);
    let mut outputs = Vec::with_capacity(batch_size);
    let mut output_refs = Vec::with_capacity(batch_size);

    // Create batch data
    for _i in 0..batch_size {
        inputs.push(create_test_signal(large_size));
        outputs.push(vec![Complex64::default(); large_size]);
    }

    // Get mutable references to outputs
    for output in &mut outputs {
        output_refs.push(&mut output[..]);
    }

    // Execute batch
    let start = Instant::now();
    let times = executor
        .execute_batch(
            &inputs.iter().map(|v| v.as_slice()).collect::<Vec<_>>(),
            &mut output_refs,
        )
        .unwrap();

    println!(
        "Batch of {} FFTs executed in {:?}",
        batch_size,
        start.elapsed()
    );
    println!("Individual FFT execution times:");
    for (i, time) in times.iter().enumerate() {
        println!("  FFT {}: {:?}", i, time);
    }

    let avg_time = times
        .iter()
        .fold(Duration::from_nanos(0), |acc, &time| acc + time)
        / times.len() as u32;
    println!("\nAverage execution time per FFT: {:?}", avg_time);
}

/// Create a test signal of the given size
fn create_test_signal(size: usize) -> Vec<Complex64> {
    let mut signal = Vec::with_capacity(size);
    for i in 0..size {
        let phase = 2.0 * std::f64::consts::PI * (i as f64) / (size as f64);
        signal.push(Complex64::new(phase.cos(), phase.sin()));
    }
    signal
}
