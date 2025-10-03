//! Example demonstrating the improved morphological operations
//!
//! This example shows how to use the improved morphological operations that properly
//! handle dimensionality issues.

use scirs2_core::ndarray::{s, Array2, Array3};
use scirs2_ndimage::morphology::simple_morph::{
    binary_closing_2d, binary_dilation_2d, binary_erosion_2d, binary_opening_2d, black_tophat_2d,
    grey_dilation_2d, grey_erosion_2d, morphological_gradient_2d, white_tophat_2d,
};
use scirs2_ndimage::morphology::{generate_binary_structure, Connectivity};

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Demonstrating improved morphological operations");

    // Example 1: Binary morphology with 2D arrays
    println!("\n==== Binary Morphology (2D) ====");
    binary_2d_example()?;

    // Example 2: Grayscale morphology with 2D arrays
    println!("\n==== Grayscale Morphology (2D) ====");
    grayscale_2d_example()?;

    // Example 3: Binary morphology with 3D arrays
    println!("\n==== Binary Morphology (3D) ====");
    binary_3d_example()?;

    // Example 4: Grayscale morphology with 3D arrays
    println!("\n==== Grayscale Morphology (3D) ====");
    grayscale_3d_example()?;

    println!("\nAll examples completed successfully!");
    Ok(())
}

#[allow(dead_code)]
fn binary_2d_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create a simple 7x7 binary image with a pattern
    let mut input = Array2::from_elem((7, 7), false);

    // Create a square pattern
    input.slice_mut(s![2..5, 2..5]).fill(true);

    // Add a small feature that should be removed by opening
    input[[1, 1]] = true;

    println!("Original binary image:");
    print_binary_2d(&input);

    // Create a structuring element as a 2D array directly
    let structure_2d = generate_binary_structure(2, Connectivity::Face)?
        .into_dimensionality::<scirs2_core::ndarray::Ix2>()?; // Convert to 2D explicitly

    // Apply erosion using the 2D-specific function
    let eroded = binary_erosion_2d(&input, Some(&structure_2d), Some(1), None, None)?;
    println!("\nAfter erosion:");
    print_binary_2d(&eroded);

    // Apply dilation to original using the 2D-specific function
    let dilated = binary_dilation_2d(&input, Some(&structure_2d), Some(1), None, None)?;
    println!("\nAfter dilation:");
    print_binary_2d(&dilated);

    // Apply opening (erosion followed by dilation) using the 2D-specific function
    let opened = binary_opening_2d(&input, Some(&structure_2d), Some(1), None, None)?;
    println!("\nAfter opening (removes small features):");
    print_binary_2d(&opened);

    // Create an image with a hole to demonstrate closing
    let mut hole_input = Array2::from_elem((7, 7), false);
    hole_input.slice_mut(s![2..5, 2..5]).fill(true);
    hole_input[[3, 3]] = false; // Create a hole

    println!("\nImage with a hole:");
    print_binary_2d(&hole_input);

    // Apply closing to fill the hole using the 2D-specific function
    let closed = binary_closing_2d(&hole_input, Some(&structure_2d), Some(1), None, None)?;
    println!("\nAfter closing (fills small holes):");
    print_binary_2d(&closed);

    Ok(())
}

#[allow(dead_code)]
fn grayscale_2d_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create a simple 7x7 grayscale image with a pattern
    let mut input = Array2::from_elem((7, 7), 0.0);

    // Create a plateau with a peak
    input.slice_mut(s![2..5, 2..5]).fill(1.0);
    input[[3, 3]] = 2.0; // Peak

    println!("Original grayscale image:");
    print_grayscale_2d(&input);

    // Apply grayscale erosion using the 2D-specific function
    let eroded = grey_erosion_2d(&input, None, None, None, None)?;
    println!("\nAfter grayscale erosion (darkens image):");
    print_grayscale_2d(&eroded);

    // Apply grayscale dilation using the 2D-specific function
    let dilated = grey_dilation_2d(&input, None, None, None, None)?;
    println!("\nAfter grayscale dilation (brightens image):");
    print_grayscale_2d(&dilated);

    // Apply morphological gradient (edge detection) using the 2D-specific function
    let gradient = morphological_gradient_2d(&input, None, None, None, None)?;
    println!("\nMorphological gradient (edge detection):");
    print_grayscale_2d(&gradient);

    // Apply white tophat transform using the 2D-specific function
    let tophat = white_tophat_2d(&input, None, None, None, None)?;
    println!("\nWhite tophat (extracts bright features):");
    print_grayscale_2d(&tophat);

    // Create an image with a dark spot for black tophat
    let mut dark_input = Array2::from_elem((7, 7), 1.0);
    dark_input[[3, 3]] = 0.0; // Dark spot

    println!("\nImage with a dark spot:");
    print_grayscale_2d(&dark_input);

    // Apply black tophat transform using the 2D-specific function
    let black_hat = black_tophat_2d(&dark_input, None, None, None, None)?;
    println!("\nBlack tophat (extracts dark features):");
    print_grayscale_2d(&black_hat);

    Ok(())
}

#[allow(dead_code)]
fn binary_3d_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create a simple 5x5x5 binary 3D array
    let mut input = Array3::from_elem((5, 5, 5), false);

    // Create a cube in the center
    input.slice_mut(s![1..4, 1..4, 1..4]).fill(true);

    println!("Original 3D binary array (showing central slice):");
    print_binary_3d_slice(&input, 2);

    println!("\nNote: 3D morphological operations are not fully implemented yet.");
    println!("The current implementation only handles 1D and 2D arrays specifically.");
    println!("Support for arbitrary n-dimensional arrays will be added in a future update.");

    Ok(())
}

#[allow(dead_code)]
fn grayscale_3d_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create a simple 5x5x5 grayscale 3D array
    let mut input = Array3::from_elem((5, 5, 5), 0.0);

    // Create a cube in the center
    input.slice_mut(s![1..4, 1..4, 1..4]).fill(1.0);

    // Add a peak at the center
    input[[2, 2, 2]] = 2.0;

    println!("Original 3D grayscale array (showing central slice):");
    print_grayscale_3d_slice(&input, 2);

    println!("\nNote: 3D grayscale morphological operations are not fully implemented yet.");
    println!("The current implementation only handles 1D and 2D arrays specifically.");
    println!("Support for arbitrary n-dimensional arrays will be added in a future update.");

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

// Helper function to print a slice of a 3D binary array
#[allow(dead_code)]
fn print_binary_3d_slice(_arr: &Array3<bool>, sliceidx: usize) {
    let slice = _arr.index_axis(scirs2_core::ndarray::Axis(2), sliceidx);
    for i in 0..slice.shape()[0] {
        for j in 0..slice.shape()[1] {
            if slice[[i, j]] {
                print!("█ ");
            } else {
                print!("· ");
            }
        }
        println!();
    }
}

// Helper function to print a slice of a 3D grayscale array
#[allow(dead_code)]
fn print_grayscale_3d_slice(_arr: &Array3<f64>, sliceidx: usize) {
    let slice = _arr.index_axis(scirs2_core::ndarray::Axis(2), sliceidx);
    for i in 0..slice.shape()[0] {
        for j in 0..slice.shape()[1] {
            let val = slice[[i, j]];
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
