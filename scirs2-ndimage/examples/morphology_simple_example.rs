//! Example demonstrating simple morphological operations
//!
//! This example shows how to use the simplified 2D morphological operations
//! that are more reliable and easier to use than the general n-dimensional versions.

use scirs2_core::ndarray::{s, Array2};
use scirs2_ndimage::morphology::simple_morph::{
    binary_closing_2d, binary_dilation_2d, binary_erosion_2d, binary_opening_2d, black_tophat_2d,
    grey_dilation_2d, grey_erosion_2d, morphological_gradient_2d, white_tophat_2d,
};

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Demonstrating simplified 2D morphological operations");

    // Example 1: Binary morphology
    println!("\n==== Binary Morphology (2D) ====");

    // Create a simple 7x7 binary image with a pattern
    let mut input = Array2::from_elem((7, 7), false);

    // Create a square pattern
    input.slice_mut(s![2..5, 2..5]).fill(true);

    // Add a small feature that should be removed by opening
    input[[1, 1]] = true;

    println!("Original binary image:");
    print_binary_2d(&input);

    // Apply erosion
    let eroded = binary_erosion_2d(&input, None, None, None, None)?;
    println!("\nAfter erosion:");
    print_binary_2d(&eroded);

    // Apply dilation to original
    let dilated = binary_dilation_2d(&input, None, None, None, None)?;
    println!("\nAfter dilation:");
    print_binary_2d(&dilated);

    // Apply opening (erosion followed by dilation)
    let opened = binary_opening_2d(&input, None, None, None, None)?;
    println!("\nAfter opening (removes small features):");
    print_binary_2d(&opened);

    // Create an image with a hole to demonstrate closing
    let mut hole_input = Array2::from_elem((7, 7), false);
    hole_input.slice_mut(s![2..5, 2..5]).fill(true);
    hole_input[[3, 3]] = false; // Create a hole

    println!("\nImage with a hole:");
    print_binary_2d(&hole_input);

    // Apply closing to fill the hole
    let closed = binary_closing_2d(&hole_input, None, None, None, None)?;
    println!("\nAfter closing (fills small holes):");
    print_binary_2d(&closed);

    // Example 2: Grayscale morphology
    println!("\n==== Grayscale Morphology (2D) ====");

    // Create a simple 7x7 grayscale image with a pattern
    let mut input = Array2::from_elem((7, 7), 0.0);

    // Create a plateau with a peak
    input.slice_mut(s![2..5, 2..5]).fill(1.0);
    input[[3, 3]] = 2.0; // Peak

    println!("Original grayscale image:");
    print_grayscale_2d(&input);

    // Apply grayscale erosion
    let eroded = grey_erosion_2d(&input, None, None, None, None)?;
    println!("\nAfter grayscale erosion (darkens image):");
    print_grayscale_2d(&eroded);

    // Apply grayscale dilation
    let dilated = grey_dilation_2d(&input, None, None, None, None)?;
    println!("\nAfter grayscale dilation (brightens image):");
    print_grayscale_2d(&dilated);

    // Apply morphological gradient (edge detection)
    let gradient = morphological_gradient_2d(&input, None, None, None, None)?;
    println!("\nMorphological gradient (edge detection):");
    print_grayscale_2d(&gradient);

    // Apply white tophat transform
    let tophat = white_tophat_2d(&input, None, None, None, None)?;
    println!("\nWhite tophat (extracts bright features):");
    print_grayscale_2d(&tophat);

    // Create an image with a dark spot for black tophat
    let mut dark_input = Array2::from_elem((7, 7), 1.0);
    dark_input[[3, 3]] = 0.0; // Dark spot

    println!("\nImage with a dark spot:");
    print_grayscale_2d(&dark_input);

    // Apply black tophat transform
    let black_hat = black_tophat_2d(&dark_input, None, None, None, None)?;
    println!("\nBlack tophat (extracts dark features):");
    print_grayscale_2d(&black_hat);

    println!("\nAll examples completed successfully!");
    Ok(())
}

// Helper function to print a 2D binary array
#[allow(dead_code)]
fn print_binary_2d(arr: &Array2<bool>) {
    for i in 0..arr.shape()[0] {
        for j in 0..arr.shape()[1] {
            if arr[[i, j]] {
                print!("█ ");
            } else {
                print!("· ");
            }
        }
        println!();
    }
}

// Helper function to print a 2D grayscale array
#[allow(dead_code)]
fn print_grayscale_2d(arr: &Array2<f64>) {
    for i in 0..arr.shape()[0] {
        for j in 0..arr.shape()[1] {
            let val = arr[[i, j]];
            if val <= 0.0 {
                print!("· ");
            } else if val < 0.5 {
                print!("░ ");
            } else if val < 1.0 {
                print!("▒ ");
            } else if val < 1.5 {
                print!("▓ ");
            } else {
                print!("█ ");
            }
        }
        println!();
    }
}
