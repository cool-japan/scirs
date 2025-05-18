//! Example demonstrating Canny edge detection functionality
//!
//! This example shows how to:
//! 1. Load an image
//! 2. Apply Canny edge detection with different parameters
//! 3. Compare results with different sigma values and thresholds

use image::{DynamicImage, GrayImage};
use scirs2_vision::feature::{canny, canny_simple, PreprocessMode};
use std::error::Error;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn Error>> {
    println!("SciRS2 Vision - Canny Edge Detection Example");

    // In a real application, you would provide your own image file path
    let image_path = "examples/input/input.jpg"; // Change this to your image path
    println!("Attempting to load image from: {}", image_path);

    // Check if the image file exists
    let path = PathBuf::from(image_path);
    let img = if path.exists() {
        // Load existing image
        image::open(path)?
    } else {
        // Create a demo image with various features
        println!("Image file not found. Creating a demo image with features...");
        create_demo_image()
    };

    println!("Processing image: {}x{}", img.width(), img.height());

    // 1. Simple Canny with default parameters
    println!("\n1. Simple Canny edge detection (sigma=1.0)...");
    let edges_simple = canny_simple(&img, 1.0)?;

    // Save result (if possible)
    if let Err(e) = edges_simple.save("examples/output/canny_simple_output.png") {
        println!("Could not save output: {}", e);
    } else {
        println!("Saved result to examples/output/canny_simple_output.png");
    }

    // 2. Canny with custom parameters
    println!("\n2. Canny with custom parameters...");
    let edges_custom = canny(
        &img,
        2.0,        // sigma
        Some(0.05), // low threshold
        Some(0.15), // high threshold
        None,       // no mask
        false,      // use absolute thresholds
        PreprocessMode::Constant(0.0),
    )?;

    if let Err(e) = edges_custom.save("examples/output/canny_custom_output.png") {
        println!("Could not save output: {}", e);
    } else {
        println!("Saved result to examples/output/canny_custom_output.png");
    }

    // 3. Canny with quantile thresholds
    println!("\n3. Canny with quantile thresholds...");
    let edges_quantile = canny(
        &img,
        1.5,       // sigma
        Some(0.2), // 20th percentile
        Some(0.8), // 80th percentile
        None,      // no mask
        true,      // use quantile thresholds
        PreprocessMode::Constant(0.0),
    )?;

    if let Err(e) = edges_quantile.save("examples/output/canny_quantile_output.png") {
        println!("Could not save output: {}", e);
    } else {
        println!("Saved result to examples/output/canny_quantile_output.png");
    }

    // 4. Demonstrate effect of sigma parameter
    println!("\n4. Demonstrating effect of sigma parameter...");
    for sigma in &[0.5, 1.0, 2.0, 3.0] {
        let edges = canny_simple(&img, *sigma)?;
        let filename = format!("examples/output/canny_sigma_{}.png", sigma);

        if let Err(e) = edges.save(&filename) {
            println!("Could not save {}: {}", filename, e);
        } else {
            println!("Saved result with sigma={} to {}", sigma, filename);
        }
    }

    println!("\nCanny edge detection example complete!");
    println!("Check the output images to see the detected edges.");

    // Count edges in simple result
    let edge_count = count_edge_pixels(&edges_simple);
    println!("\nStatistics:");
    println!("  Edge pixels detected (simple): {}", edge_count);
    println!(
        "  Total pixels: {}",
        edges_simple.width() * edges_simple.height()
    );
    println!(
        "  Edge percentage: {:.2}%",
        edge_count as f32 / (edges_simple.width() * edges_simple.height()) as f32 * 100.0
    );

    Ok(())
}

/// Create a demo image with various features for edge detection
fn create_demo_image() -> DynamicImage {
    let size = 200;
    let mut img_buffer = GrayImage::new(size, size);

    // Create some features:
    // 1. A square in the center
    for y in 50..150 {
        for x in 50..150 {
            img_buffer.put_pixel(x, y, image::Luma([200]));
        }
    }

    // 2. A diagonal line
    for i in 0..size {
        if i < size {
            img_buffer.put_pixel(i, i, image::Luma([255]));
        }
    }

    // 3. A circle
    let center = size as f32 / 2.0;
    let radius = 30.0;
    for y in 0..size {
        for x in 0..size {
            let dx = x as f32 - center;
            let dy = y as f32 - center;
            let dist = (dx * dx + dy * dy).sqrt();

            if (dist - radius).abs() < 2.0 {
                img_buffer.put_pixel(x, y, image::Luma([128]));
            }
        }
    }

    // 4. Add some noise
    use scirs2_vision::preprocessing::gaussian_blur;
    let dynamic_img = DynamicImage::ImageLuma8(img_buffer);

    // Apply slight blur to make it more realistic
    match gaussian_blur(&dynamic_img, 0.5) {
        Ok(blurred) => blurred,
        Err(_) => dynamic_img,
    }
}

/// Count the number of edge pixels in an image
fn count_edge_pixels(img: &GrayImage) -> usize {
    let mut count = 0;
    for pixel in img.pixels() {
        if pixel[0] > 0 {
            count += 1;
        }
    }
    count
}
