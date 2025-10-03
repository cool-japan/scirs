//! Example demonstrating various filter operations in scirs2-ndimage
//!
//! This example shows how to use different filters:
//! - Uniform filter (box filter)
//! - Minimum and maximum filters
//! - Gaussian filter
//! - Median filter

use scirs2_core::ndarray::{array, Array2};
use scirs2_ndimage::filters::{
    gaussian_filter, maximum_filter, median_filter, minimum_filter, uniform_filter, BorderMode,
};

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== scirs2-ndimage Filter Operations Example ===\n");

    // Create a test image with some features
    let image = array![
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 2.0, 2.0, 2.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 2.0, 9.0, 2.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 2.0, 2.0, 2.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ];

    println!("Original image:");
    print_array(&image);
    println!();

    // 1. Apply uniform filter (box filter) with size 3x3
    println!("1. Uniform filter (3x3):");
    let uniform_filtered = uniform_filter(&image, &[3, 3], None, None)?;
    print_array(&uniform_filtered);
    println!();

    // 2. Apply minimum filter with size 3x3
    println!("2. Minimum filter (3x3):");
    let min_filtered = minimum_filter(&image, &[3, 3], None, None)?;
    print_array(&min_filtered);
    println!();

    // 3. Apply maximum filter with size 3x3
    println!("3. Maximum filter (3x3):");
    let max_filtered = maximum_filter(&image, &[3, 3], None, None)?;
    print_array(&max_filtered);
    println!();

    // 4. Apply Gaussian filter with sigma=1.0
    println!("4. Gaussian filter (sigma=1.0):");
    let gauss_filtered = gaussian_filter(&image, 1.0, None, None)?;
    print_array(&gauss_filtered);
    println!();

    // 5. Apply median filter with size 3x3
    println!("5. Median filter (3x3):");
    let median_filtered = median_filter(&image, &[3, 3], None)?;
    print_array(&median_filtered);
    println!();

    // 6. Demonstrate border modes with uniform filter
    println!("6. Border mode comparison (uniform filter 3x3):");
    let border_modes = [
        (BorderMode::Constant, "Constant"),
        (BorderMode::Reflect, "Reflect"),
        (BorderMode::Mirror, "Mirror"),
        (BorderMode::Wrap, "Wrap"),
        (BorderMode::Nearest, "Nearest"),
    ];

    for (mode, name) in &border_modes {
        println!("  Border mode: {}", name);
        let filtered = uniform_filter(&image, &[3, 3], Some(*mode), None)?;
        print_array(&filtered);
        println!();
    }

    println!("All filters applied successfully!");
    Ok(())
}

// Helper function to print an array in a grid format
#[allow(dead_code)]
fn print_array(array: &Array2<f64>) {
    let (rows, cols) = array.dim();

    // Find the width needed
    let max_chars = array.iter().fold(1, |max, &val| {
        let val_str = format!("{:.1}", val);
        if val_str.len() > max {
            val_str.len()
        } else {
            max
        }
    });

    for i in 0..rows {
        for j in 0..cols {
            print!("{:>width$.1} ", array[[i, j]], width = max_chars);
        }
        println!();
    }
}
