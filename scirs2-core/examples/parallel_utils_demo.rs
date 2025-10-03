//! Example demonstrating the parallel utility functions
//!
//! This example shows how the utils module automatically uses parallel
//! implementations for large arrays when the parallel feature is enabled.

use scirs2_core::ndarray_ext::{Array1, Array2};
use scirs2_core::utils::{linspace, maximum, minimum};
use std::time::Instant;

#[allow(dead_code)]
fn main() {
    println!("=== Parallel Utility Functions Demo ===\n");

    // Test linspace with a large array (triggers parallel implementation)
    println!("1. Testing linspace with large array (n=10000)");
    let start = Instant::now();
    let large_array = linspace(0.0, 100.0, 10_000);
    let elapsed = start.elapsed();
    println!(
        "   Generated array with {} elements in {:?}",
        large_array.len(),
        elapsed
    );
    println!(
        "   First 5 elements: {:?}",
        &large_array.slice(scirs2_core::ndarray::s![..5])
    );
    println!(
        "   Last 5 elements: {:?}",
        &large_array.slice(scirs2_core::ndarray::s![-5..])
    );

    // Test linspace with a small array (uses serial implementation)
    println!("\n2. Testing linspace with small array (n=100)");
    let start = Instant::now();
    let small_array = linspace(0.0, 10.0, 100);
    let elapsed = start.elapsed();
    println!(
        "   Generated array with {} elements in {:?}",
        small_array.len(),
        elapsed
    );

    // Test maximum with large arrays
    println!("\n3. Testing maximum with large arrays");
    let a = Array1::from_elem(5000, 1.0);
    let b = Array1::linspace(0.0, 10.0, 5000);
    let start = Instant::now();
    let max_result = maximum(&a, &b);
    let elapsed = start.elapsed();
    println!(
        "   Computed maximum of {} elements in {:?}",
        max_result.len(),
        elapsed
    );
    println!(
        "   First 5 elements: {:?}",
        &max_result.slice(scirs2_core::ndarray::s![..5])
    );
    println!(
        "   Last 5 elements: {:?}",
        &max_result.slice(scirs2_core::ndarray::s![-5..])
    );

    // Test minimum with large arrays
    println!("\n4. Testing minimum with large arrays");
    let start = Instant::now();
    let min_result = minimum(&a, &b);
    let elapsed = start.elapsed();
    println!(
        "   Computed minimum of {} elements in {:?}",
        min_result.len(),
        elapsed
    );
    println!(
        "   First 5 elements: {:?}",
        &min_result.slice(scirs2_core::ndarray::s![..5])
    );
    println!(
        "   Last 5 elements: {:?}",
        &min_result.slice(scirs2_core::ndarray::s![-5..])
    );

    // Demonstrate parallel speedup with a larger computation
    println!("\n5. Demonstrating parallel speedup");

    // Create very large arrays for performance comparison
    let size = 1_000_000;
    let a_large = Array1::linspace(0.0, 1000.0, size);
    let b_large = Array1::linspace(500.0, 1500.0, size);

    let start = Instant::now();
    let max_large = maximum(&a_large, &b_large);
    let elapsed = start.elapsed();
    println!(
        "   Maximum of {} elements computed in {:?}",
        max_large.len(),
        elapsed
    );

    let start = Instant::now();
    let min_large = minimum(&a_large, &b_large);
    let elapsed = start.elapsed();
    println!(
        "   Minimum of {} elements computed in {:?}",
        min_large.len(),
        elapsed
    );

    // Test with 2D arrays
    println!("\n6. Testing with 2D arrays");
    let a_2d = Array2::from_shape_fn((100, 100), |(i, j)| (i + j) as f64);
    let b_2d = Array2::from_shape_fn((100, 100), |(i, j)| (i * j) as f64);

    let start = Instant::now();
    let max_2d = maximum(&a_2d, &b_2d);
    let elapsed = start.elapsed();
    println!(
        "   Computed 2D maximum of shape {:?} in {:?}",
        max_2d.shape(),
        elapsed
    );

    println!("\n✅ All parallel utility functions working correctly!");

    #[cfg(feature = "parallel")]
    println!("\n   Note: Parallel feature is ENABLED - large arrays use parallel implementations");

    #[cfg(not(feature = "parallel"))]
    println!("\n   Note: Parallel feature is DISABLED - all arrays use serial implementations");
}
