//! Performance showcase for scirs2-ndimage optimizations
//!
//! This example demonstrates the performance improvements achieved through
//! SIMD acceleration, parallel processing, and optimized algorithms.

use ndarray::Array2;
use scirs2_ndimage::filters::{filter_functions, generic_filter, BorderMode};
use std::time::Instant;

fn main() {
    println!("🚀 SciRS2-ndimage Performance Showcase");
    println!("=======================================\n");

    // Test different array sizes to demonstrate scaling behavior
    let sizes = vec![
        (50, 50, "Small (50x50)"),
        (100, 100, "Medium (100x100)"),
        (200, 200, "Large (200x200)"),
        (400, 400, "Very Large (400x400)"),
    ];

    for (rows, cols, label) in sizes {
        println!("Testing {} arrays:", label);

        // Create test image with interesting structure
        let input = Array2::from_shape_fn((rows, cols), |(i, j)| {
            let x = i as f64 / rows as f64;
            let y = j as f64 / cols as f64;
            (x * y).sin() * 100.0 + (x + y).cos() * 50.0
        });

        // Test various filter operations
        benchmark_filter(&input, "Mean Filter", filter_functions::mean);
        benchmark_filter(&input, "Range Filter", filter_functions::range);
        benchmark_filter(&input, "Variance Filter", filter_functions::variance);
        benchmark_filter(&input, "Median Filter", filter_functions::median);
        benchmark_filter(&input, "Maximum Filter", filter_functions::maximum);
        benchmark_filter(&input, "Minimum Filter", filter_functions::minimum);

        println!();
    }

    println!("🎯 Performance Notes:");
    println!("- Arrays larger than 10,000 elements automatically use parallel processing");
    println!("- SIMD acceleration is available when the 'simd' feature is enabled");
    println!("- Border mode handling optimizations reduce memory allocation overhead");
    println!("- Generic filter framework allows custom mathematical operations");

    #[cfg(feature = "simd")]
    println!("✅ SIMD acceleration: ENABLED");
    #[cfg(not(feature = "simd"))]
    println!("⚠️  SIMD acceleration: DISABLED (enable with --features simd)");

    #[cfg(feature = "parallel")]
    println!("✅ Parallel processing: ENABLED");
    #[cfg(not(feature = "parallel"))]
    println!("⚠️  Parallel processing: DISABLED (enable with --features parallel)");
}

fn benchmark_filter<F>(input: &Array2<f64>, name: &str, filter_fn: F)
where
    F: Fn(&[f64]) -> f64 + Send + Sync + Clone + 'static,
{
    let start = Instant::now();

    let result =
        generic_filter(input, filter_fn, &[3, 3], Some(BorderMode::Reflect), None).unwrap();

    let duration = start.elapsed();
    let elements = input.len();
    let throughput = elements as f64 / duration.as_secs_f64() / 1_000_000.0; // Million elements per second

    println!(
        "  {}: {:.2}ms ({:.2} MElements/s)",
        name,
        duration.as_millis(),
        throughput
    );

    // Verify the result is reasonable
    assert_eq!(result.shape(), input.shape());
    assert!(result.iter().all(|&x| x.is_finite()));
}
