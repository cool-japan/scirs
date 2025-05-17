//! Example demonstrating worker pool management for parallel FFT operations
//!
//! This example shows how to control the number of worker threads for FFT operations.

use ndarray::Array2;
use num_complex::Complex64;
use scirs2_fft::{fft2, get_global_pool, get_workers, with_workers};
use std::time::Instant;

fn main() {
    println!("FFT Worker Pool Management Example");
    println!("==================================");
    println!();

    // Create a test 2D array
    let size = 512;
    let signal = Array2::from_shape_fn((size, size), |(i, j)| Complex64::new((i + j) as f64, 0.0));

    // Get current worker configuration
    let original_workers = get_workers();
    println!("Original number of workers: {}", original_workers);

    // Show worker pool info
    let pool = get_global_pool();
    let info = pool.get_info();
    println!("Worker pool info: {}", info);
    println!();

    // Test with different numbers of workers
    let worker_counts = vec![1, 2, 4, 8];

    for &num_workers in &worker_counts {
        println!("Testing with {} workers:", num_workers);

        let start = Instant::now();

        // Execute FFT with specific number of workers
        let _result = with_workers(num_workers, || {
            fft2(&signal.view(), None, None, None).unwrap()
        });

        let duration = start.elapsed();
        println!("  Time: {:?}", duration);
    }

    println!();

    // Test with parallelization disabled
    println!("Testing with parallelization disabled:");
    pool.set_enabled(false);

    let start = Instant::now();
    let _result = fft2(&signal.view(), None, None, None).unwrap();
    let duration = start.elapsed();

    println!("  Time: {:?}", duration);

    // Re-enable parallelization
    pool.set_enabled(true);

    println!();

    // Show environment variable support
    println!("Environment variable support:");
    println!("Set SCIRS2_FFT_WORKERS to control default worker count");
    println!();

    // Demonstrate worker context (temporary worker count)
    println!("Using temporary worker context:");

    let _result = with_workers(2, || {
        println!("  Inside context: {} workers", 2);
        // Perform FFT operation with 2 workers
        fft2(&signal.view(), None, None, None).unwrap()
    });

    println!("  After context: {} workers", get_workers());

    // Reset to original workers
    //set_workers(original_workers).ok();
}
