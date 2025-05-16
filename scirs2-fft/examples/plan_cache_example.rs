//! Example demonstrating FFT plan caching for improved performance
//!
//! This example shows how plan caching can significantly speed up repeated FFT operations
//! of the same size.

use rustfft::FftPlanner;
use scirs2_fft::{fft, get_global_cache, init_global_cache};
use std::time::{Duration, Instant};

fn main() {
    // Initialize the global cache with custom settings
    // This is optional - if not called, default settings will be used
    init_global_cache(64, Duration::from_secs(300)).ok();

    // Generate test signals of various sizes
    let sizes = vec![128, 256, 512, 1024, 2048];
    let num_iterations = 100;

    println!("FFT Plan Caching Demonstration");
    println!("==============================");
    println!();

    for &size in &sizes {
        println!("FFT size: {}", size);

        // Create a signal
        let signal: Vec<f64> = (0..size).map(|i| i as f64).collect();

        // First run (cold cache)
        let start = Instant::now();
        for _ in 0..num_iterations {
            let _ = fft(&signal, None).unwrap();
        }
        let cold_duration = start.elapsed();

        // Second run (warm cache)
        let start = Instant::now();
        for _ in 0..num_iterations {
            let _ = fft(&signal, None).unwrap();
        }
        let warm_duration = start.elapsed();

        // Calculate speedup
        let speedup = cold_duration.as_secs_f64() / warm_duration.as_secs_f64();

        println!(
            "  Cold cache: {:?} for {} iterations",
            cold_duration, num_iterations
        );
        println!(
            "  Warm cache: {:?} for {} iterations",
            warm_duration, num_iterations
        );
        println!("  Speedup: {:.2}x", speedup);
        println!();
    }

    // Print cache statistics
    let cache = get_global_cache();
    let stats = cache.get_stats();
    println!("Cache Statistics:");
    println!("{}", stats);
    println!();

    // Demonstrate pre-computing common sizes
    println!("Pre-computing common sizes...");
    let common_sizes = vec![128, 256, 512, 1024];

    // Clear cache and pre-compute
    cache.clear();

    // Pre-compute sizes
    let mut planner = FftPlanner::new();
    cache.precompute_common_sizes(&common_sizes, &mut planner);

    // Show updated stats
    let stats = cache.get_stats();
    println!("After pre-computing: {}", stats);

    // Test that pre-computed sizes work
    let signal_128: Vec<f64> = vec![1.0; 128];
    let _ = fft(&signal_128, None).unwrap();

    let stats = cache.get_stats();
    println!("After using pre-computed plan: {}", stats);
}
