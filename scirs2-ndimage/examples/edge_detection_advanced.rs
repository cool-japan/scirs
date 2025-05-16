//! Advanced edge detection example
//!
//! This example demonstrates the advanced edge detection capabilities in scirs2-ndimage,
//! including the enhanced Canny edge detector and unified edge detection interface.

use ndarray::{array, Array2};
use scirs2_ndimage::features::{
    canny, edge_detector, edge_detector_simple, gradient_edges, EdgeDetectionAlgorithm,
    EdgeDetectionConfig, GradientMethod,
};
use scirs2_ndimage::filters::BorderMode;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== scirs2-ndimage Advanced Edge Detection Example ===\n");

    // Create a test image with a simple pattern
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

    // Clone to dynamic array for older API functions
    let image_dyn = image.clone().into_dyn();

    println!("Original image:");
    print_array(&image);

    // Part 1: Legacy API Usage
    println!("\n=== Legacy API Examples ===");

    // 1. Basic edge detection using gradient magnitude with Sobel operator
    println!("\n1. Edge detection using Sobel gradient magnitude (legacy API):");
    let edges_sobel = edge_detector_simple(&image_dyn, None, None)?;
    let edges_sobel_2d = edges_sobel.into_dimensionality::<ndarray::Ix2>().unwrap();
    print_array(&edges_sobel_2d);

    // 2. Edge detection using Scharr operator for better rotational invariance
    println!("\n2. Edge detection using Scharr gradient magnitude (legacy API):");
    let edges_scharr = edge_detector_simple(&image_dyn, Some(GradientMethod::Scharr), None)?;
    let edges_scharr_2d = edges_scharr.into_dimensionality::<ndarray::Ix2>().unwrap();
    print_array(&edges_scharr_2d);

    // 3. Canny edge detection with Sobel operator
    println!("\n3. Canny edge detection (Sobel):");
    let edges_canny_sobel = canny(&image, 1.0, 0.1, 0.2, None);
    print_binary_array(&edges_canny_sobel);

    // 4. Canny edge detection with Scharr operator for better performance
    println!("\n4. Canny edge detection (Scharr):");
    let edges_canny_scharr = canny(&image, 1.0, 0.1, 0.2, Some(GradientMethod::Scharr));
    print_binary_array(&edges_canny_scharr);

    // Part 2: New Unified API
    println!("\n=== New Unified API Examples ===");

    // 5. Using the unified edge_detector API with default settings (Canny)
    println!("\n5. Unified Edge Detector API (Default - Canny):");
    let unified_default = edge_detector(&image, EdgeDetectionConfig::default());
    print_array(&unified_default);

    // 6. Using Canny with custom parameters
    println!("\n6. Unified Edge Detector API (Custom Canny):");
    let canny_config = EdgeDetectionConfig {
        algorithm: EdgeDetectionAlgorithm::Canny,
        gradient_method: GradientMethod::Scharr,
        sigma: 1.5,
        low_threshold: 0.05,
        high_threshold: 0.15,
        border_mode: BorderMode::Reflect,
        return_magnitude: false,
    };
    let unified_canny = edge_detector(&image, canny_config);
    print_array(&unified_canny);

    // 7. Using the unified edge_detector API with Laplacian of Gaussian
    println!("\n7. Unified Edge Detector API (LoG with custom settings):");
    let log_config = EdgeDetectionConfig {
        algorithm: EdgeDetectionAlgorithm::LoG,
        sigma: 1.2,
        low_threshold: 0.08,
        return_magnitude: true,
        ..EdgeDetectionConfig::default()
    };
    let unified_log = edge_detector(&image, log_config);
    print_array(&unified_log);

    // 8. Using the unified edge_detector API with Gradient algorithm and Scharr method
    println!("\n8. Unified Edge Detector API (Gradient with Scharr method):");
    let gradient_config = EdgeDetectionConfig {
        algorithm: EdgeDetectionAlgorithm::Gradient,
        gradient_method: GradientMethod::Scharr,
        sigma: 1.0,
        low_threshold: 0.1,
        border_mode: BorderMode::Reflect,
        return_magnitude: true,
        ..EdgeDetectionConfig::default()
    };
    let unified_gradient = edge_detector(&image, gradient_config);
    print_array(&unified_gradient);

    // Part 3: Comparing gradient methods on diagonal edges
    println!("\n=== Gradient Method Comparison for Diagonal Edges ===");

    // Create a diagonal edge image
    let diagonal_image = array![
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ];

    println!("\nDiagonal edge image:");
    print_array(&diagonal_image);

    println!("\n9a. Sobel Gradient on Diagonal:");
    let sobel_diagonal = gradient_edges(&diagonal_image, Some(GradientMethod::Sobel), None, None);
    print_array(&sobel_diagonal);

    println!("\n9b. Scharr Gradient on Diagonal:");
    let scharr_diagonal = gradient_edges(&diagonal_image, Some(GradientMethod::Scharr), None, None);
    print_array(&scharr_diagonal);

    // Note the higher values in the diagonal with Scharr compared to Sobel
    println!("\nNote how Scharr produces stronger responses on diagonal edges compared to Sobel.");
    println!("This demonstrates the better rotational invariance of the Scharr operator.");

    Ok(())
}

// Helper function to print a float array in a grid format
fn print_array(array: &Array2<f32>) {
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

// Helper function to print a binary edge array (using threshold)
fn print_binary_array(array: &Array2<f32>) {
    let (rows, cols) = array.dim();

    for i in 0..rows {
        for j in 0..cols {
            if array[[i, j]] > 0.0 {
                print!("● "); // Unicode filled circle for edges
            } else {
                print!(". "); // Period for non-edges
            }
        }
        println!();
    }
}
