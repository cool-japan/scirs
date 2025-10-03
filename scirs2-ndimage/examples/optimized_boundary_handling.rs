//! Example demonstrating optimized boundary handling in filters
//!
//! This example shows how to use the virtual boundary handler for memory-efficient
//! filtering of large arrays.

use scirs2_core::ndarray::{arr2, Array2};
use scirs2_ndimage::filters::{convolve, convolve_fast, convolve_optimized, BorderMode};
use std::time::Instant;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Optimized Boundary Handling Example");
    println!("===================================\n");

    // Create a test image
    let size = 100;
    let image: Array2<f64> = Array2::from_shape_fn((size, size), |(i, j)| {
        ((i as f64 - size as f64 / 2.0).powi(2) + (j as f64 - size as f64 / 2.0).powi(2)).sqrt()
    });

    // Create a simple 3x3 averaging kernel
    let kernel = arr2(&[
        [1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0],
        [1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0],
        [1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0],
    ]);

    println!("Image size: {}x{}", size, size);
    println!("Kernel size: 3x3\n");

    // Test different boundary modes
    let modes = [
        (BorderMode::Constant, "Constant"),
        (BorderMode::Nearest, "Nearest"),
        (BorderMode::Reflect, "Reflect"),
        (BorderMode::Mirror, "Mirror"),
        (BorderMode::Wrap, "Wrap"),
    ];

    for (mode, mode_name) in &modes {
        println!("Testing {} mode:", mode_name);

        // Standard implementation (creates padded array)
        let start = Instant::now();
        let result_standard = convolve(&image, &kernel, Some(*mode))?;
        let time_standard = start.elapsed();

        // Optimized implementation (virtual boundaries)
        let start = Instant::now();
        let result_optimized = convolve_optimized(&image, &kernel, *mode, Some(0.0))?;
        let time_optimized = start.elapsed();

        // Using the fast function with optimization flag
        let start = Instant::now();
        let result_fast = convolve_fast(&image, &kernel, Some(*mode), true)?;
        let time_fast = start.elapsed();

        println!("  Standard implementation: {:?}", time_standard);
        println!("  Optimized implementation: {:?}", time_optimized);
        println!("  Fast function (optimized): {:?}", time_fast);

        // Verify results are similar (small differences due to floating point)
        let diff_max = result_standard
            .iter()
            .zip(result_optimized.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);

        println!("  Max difference: {:.6e}", diff_max);

        // Check a few boundary values
        println!("  Boundary values (top-left corner):");
        println!(
            "    Standard: [{:.4}, {:.4}, {:.4}]",
            result_standard[[0, 0]],
            result_standard[[0, 1]],
            result_standard[[0, 2]]
        );
        println!(
            "    Optimized: [{:.4}, {:.4}, {:.4}]",
            result_optimized[[0, 0]],
            result_optimized[[0, 1]],
            result_optimized[[0, 2]]
        );
        println!();
    }

    // Demonstrate memory efficiency with a larger array
    println!("\nMemory Efficiency Test:");
    println!("=======================");

    // Create a larger test array
    let large_size = 1000;
    let largeimage: Array2<f64> = Array2::ones((large_size, large_size));

    println!("Large image size: {}x{}", large_size, large_size);
    println!(
        "Memory for image: {:.2} MB",
        (large_size * large_size * 8) as f64 / 1_048_576.0
    );

    // Time the operations
    let start = Instant::now();
    let _ = convolve_fast(&largeimage, &kernel, Some(BorderMode::Reflect), false)?;
    let time_standard = start.elapsed();

    let start = Instant::now();
    let _ = convolve_fast(&largeimage, &kernel, Some(BorderMode::Reflect), true)?;
    let time_optimized = start.elapsed();

    println!("\nStandard (with padding): {:?}", time_standard);
    println!("Optimized (virtual boundaries): {:?}", time_optimized);
    println!(
        "Speedup: {:.2}x",
        time_standard.as_secs_f64() / time_optimized.as_secs_f64()
    );

    println!("\nThe optimized implementation avoids creating a padded copy of the array,");
    println!("saving memory and potentially improving performance for large arrays.");

    Ok(())
}
