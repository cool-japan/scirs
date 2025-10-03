//! Example demonstrating edge detection filters in scirs2-ndimage
//!
//! This example shows how to use different edge detection filters:
//! - Sobel filter
//! - Prewitt filter
//! - Laplace filter
//! - Gradient magnitude

use scirs2_core::ndarray::{array, Array2};
use scirs2_ndimage::filters::{gradient_magnitude, laplace, prewitt, roberts, scharr, sobel};

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== scirs2-ndimage Edge Detection Example ===\n");

    // Create a test image with an edge
    let image = array![
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ];

    println!("Original image:");
    print_array(&image);
    println!();

    // 1. Apply Prewitt filter along x-axis
    println!("1. Prewitt filter (x-axis):");
    // Note: The convolution implementation is not complete yet, so this might return an error
    match prewitt(&image, 1, None) {
        Ok(filtered) => print_array(&filtered),
        Err(e) => println!("Error: {}", e),
    }
    println!();

    // 2. Apply Prewitt filter along y-axis
    println!("2. Prewitt filter (y-axis):");
    match prewitt(&image, 0, None) {
        Ok(filtered) => print_array(&filtered),
        Err(e) => println!("Error: {}", e),
    }
    println!();

    // 3. Apply Sobel filter along x-axis
    println!("3. Sobel filter (x-axis):");
    match sobel(&image, 1, None) {
        Ok(filtered) => print_array(&filtered),
        Err(e) => println!("Error: {}", e),
    }
    println!();

    // 4. Apply Laplace filter (4-connected)
    println!("4. Laplace filter (4-connected):");
    match laplace(&image, None, None) {
        Ok(filtered) => print_array(&filtered),
        Err(e) => println!("Error: {}", e),
    }
    println!();

    // 4b. Apply Laplace filter (8-connected)
    println!("4b. Laplace filter (8-connected):");
    match laplace(&image, None, Some(true)) {
        Ok(filtered) => print_array(&filtered),
        Err(e) => println!("Error: {}", e),
    }
    println!();

    // 5. Apply Roberts Cross filter
    println!("5. Roberts Cross filter:");
    match roberts(&image, None, None) {
        Ok(filtered) => print_array(&filtered),
        Err(e) => println!("Error: {}", e),
    }
    println!();

    // 5b. Apply Scharr filter
    println!("5b. Scharr filter (x-axis):");
    match scharr(&image, 1, None) {
        Ok(filtered) => print_array(&filtered),
        Err(e) => println!("Error: {}", e),
    }
    println!();

    // 6. Calculate gradient magnitude (using Sobel operator by default)
    println!("6. Gradient magnitude (Sobel):");
    match gradient_magnitude(&image, None, None) {
        Ok(filtered) => print_array(&filtered),
        Err(e) => println!("Error: {}", e),
    }
    println!();

    // 7. Calculate gradient magnitude using Prewitt
    println!("7. Gradient magnitude (Prewitt):");
    match gradient_magnitude(&image, None, Some("prewitt")) {
        Ok(filtered) => print_array(&filtered),
        Err(e) => println!("Error: {}", e),
    }
    println!();

    // 8. Calculate gradient magnitude using Roberts
    println!("8. Gradient magnitude (Roberts):");
    match gradient_magnitude(&image, None, Some("roberts")) {
        Ok(filtered) => print_array(&filtered),
        Err(e) => println!("Error: {}", e),
    }
    println!();

    // 9. Calculate gradient magnitude using Scharr
    println!("9. Gradient magnitude (Scharr):");
    match gradient_magnitude(&image, None, Some("scharr")) {
        Ok(filtered) => print_array(&filtered),
        Err(e) => println!("Error: {}", e),
    }
    println!();

    println!(
        "Note: Operations on arrays with more than 2 dimensions are not fully implemented yet."
    );
    Ok(())
}

// Helper function to print an array in a grid format
#[allow(dead_code)]
fn print_array(array: &Array2<f64>) {
    let (rows, cols) = array.dim();

    // Find the width needed
    let max_chars = array.iter().fold(1, |max, &val| {
        let val_str = format!("{:.2}", val);
        if val_str.len() > max {
            val_str.len()
        } else {
            max
        }
    });

    for i in 0..rows {
        for j in 0..cols {
            print!("{:>width$.2} ", array[[i, j]], width = max_chars);
        }
        println!();
    }
}
