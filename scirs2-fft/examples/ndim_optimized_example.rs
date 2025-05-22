//! Example demonstrating optimized N-dimensional FFT
//!
//! This example shows the performance benefits of the optimized
//! N-dimensional FFT implementation with better memory access patterns.

use ndarray::Array3;
use scirs2_fft::{fftn, fftn_memory_efficient, fftn_optimized};
use std::time::Instant;

fn main() {
    println!("Optimized N-dimensional FFT Example");
    println!("===================================");
    println!();

    // Test different array sizes
    let sizes = vec![(64, 64, 64), (128, 128, 32), (256, 64, 64)];

    for size in sizes {
        println!("Testing size: {:?}", size);

        // Create test array
        let array = Array3::from_shape_fn(size, |(i, j, k)| {
            ((i as f64).sin() + (j as f64).cos() + (k as f64).tan()) / 3.0
        });

        // Time standard FFT
        let start = Instant::now();
        let _result_std = fftn(&array.to_owned().into_dyn(), None, None, None, None, None).unwrap();
        let time_standard = start.elapsed();

        // Time optimized FFT
        let start = Instant::now();
        let _result_opt = fftn_optimized(&array.view(), None, None).unwrap();
        let time_optimized = start.elapsed();

        // Calculate speedup
        let speedup = time_standard.as_secs_f64() / time_optimized.as_secs_f64();

        println!("  Standard FFT: {:?}", time_standard);
        println!("  Optimized FFT: {:?}", time_optimized);
        println!("  Speedup: {:.2}x", speedup);
        println!();
    }

    // Demonstrate memory-efficient FFT for large arrays
    println!("Memory-efficient FFT for large arrays:");
    println!("-------------------------------------");

    // Create a large array
    let large_size = (512, 512, 16);
    let large_array = Array3::from_shape_fn(large_size, |(i, j, k)| (i + j + k) as f64);

    let memory_gb =
        (large_array.len() * std::mem::size_of::<f64>()) as f64 / (1024.0 * 1024.0 * 1024.0);
    println!("Array size: {:?}, Memory: {:.2} GB", large_size, memory_gb);

    // Test memory-efficient FFT with different memory limits
    let memory_limits = vec![0.5, 1.0, 2.0];

    for limit in memory_limits {
        println!("\nMemory limit: {:.1} GB", limit);

        let start = Instant::now();
        let result = fftn_memory_efficient(&large_array.view(), None, limit);
        let duration = start.elapsed();

        match result {
            Ok(_) => println!("  Success! Time: {:?}", duration),
            Err(e) => println!("  Error: {}", e),
        }
    }

    // Demonstrate axis ordering optimization
    println!("\n\nAxis ordering optimization:");
    println!("--------------------------");

    // Create array with different axis sizes
    let asymmetric = Array3::from_shape_fn((256, 32, 128), |(i, j, k)| (i * j * k) as f64);

    // Test with different axis orders
    let axis_orders = vec![vec![0, 1, 2], vec![1, 2, 0], vec![2, 0, 1]];

    for axes in axis_orders {
        println!("\nAxis order: {:?}", axes);

        let start = Instant::now();
        let _result = fftn_optimized(&asymmetric.view(), None, Some(axes)).unwrap();
        let duration = start.elapsed();

        println!("  Time: {:?}", duration);
    }

    // Demonstrate cache-friendly chunking
    println!("\n\nCache-friendly chunking:");
    println!("-----------------------");

    let chunk_test = Array3::from_shape_fn((128, 128, 128), |(i, j, k)| (i ^ j ^ k) as f64);

    // This will use optimized chunking internally
    let start = Instant::now();
    let _result = fftn_optimized(&chunk_test.view(), None, None).unwrap();
    let duration = start.elapsed();

    println!("FFT with optimized chunking: {:?}", duration);

    // Show how chunk size affects performance
    println!("\nChunk size analysis:");
    let total_elements = chunk_test.len();
    println!("Total elements: {}", total_elements);
    println!("Optimal chunk size: determined automatically based on cache size");

    // Verify correctness by comparing results
    println!("\n\nVerifying correctness:");
    println!("---------------------");

    let test_array = Array3::from_shape_fn((32, 32, 32), |(i, j, k)| {
        (i as f64 * 0.1) + (j as f64 * 0.01) + (k as f64 * 0.001)
    });

    let result_standard = fftn(
        &test_array.to_owned().into_dyn(),
        None,
        None,
        None,
        None,
        None,
    )
    .unwrap();
    let result_optimized = fftn_optimized(&test_array.view(), None, None).unwrap();

    // Compare results
    let max_diff = result_standard
        .iter()
        .zip(result_optimized.iter())
        .map(|(a, b)| ((a.re - b.re).abs() + (a.im - b.im).abs()))
        .fold(0.0, f64::max);

    println!(
        "Maximum difference between standard and optimized: {:.2e}",
        max_diff
    );
    println!("Results match: {}", max_diff < 1e-10);
}
