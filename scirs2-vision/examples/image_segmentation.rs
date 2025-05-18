//! Example demonstrating image segmentation functionality
//!
//! This example shows how to:
//! 1. Apply binary thresholding
//! 2. Use Otsu's automatic thresholding
//! 3. Apply adaptive thresholding
//! 4. Perform connected component labeling

use image::DynamicImage;
use scirs2_vision::preprocessing::{gaussian_blur, normalize_brightness};
use scirs2_vision::segmentation::{
    adaptive_threshold, connected_components, otsu_threshold, threshold_binary, AdaptiveMethod,
};
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("SciRS2 Vision - Image Segmentation Example");

    // In a real application, you would provide your own image file path
    let image_path = "examples/input/input.jpg"; // Change this to your image path
    println!("Attempting to load image from: {}", image_path);

    // Check if the image file exists
    let path = PathBuf::from(image_path);
    if !path.exists() {
        println!("Image file not found. This example needs an input image.");
        println!("Please provide an image path as argument or place an 'input.jpg' in the current directory.");

        // For demo purposes, we'll create a simple 100x100 image with some shapes
        println!("Creating a demo image with shapes for demonstration...");
        let mut img_buffer = image::ImageBuffer::new(100, 100);

        // Fill with background (low intensity)
        for y in 0..100 {
            for x in 0..100 {
                img_buffer.put_pixel(x, y, image::Luma([50]));
            }
        }

        // Draw a rectangle
        for y in 20..40 {
            for x in 20..60 {
                img_buffer.put_pixel(x, y, image::Luma([200]));
            }
        }

        // Draw a circle
        for y in 0..100 {
            for x in 0..100 {
                let dx = (x as f32 - 70.0).powi(2);
                let dy = (y as f32 - 70.0).powi(2);
                let d = (dx + dy).sqrt();
                if d < 15.0 {
                    img_buffer.put_pixel(x, y, image::Luma([230]));
                }
            }
        }

        let img = DynamicImage::ImageLuma8(img_buffer);
        process_image(&img)?;
        return Ok(());
    }

    // Load image
    let img = image::open(path)?;
    println!(
        "Successfully loaded image: {}x{}",
        img.width(),
        img.height()
    );

    process_image(&img)?;

    Ok(())
}

fn process_image(img: &DynamicImage) -> Result<(), Box<dyn std::error::Error>> {
    // 1. Preprocess the image
    println!("Preprocessing image...");
    let normalized = normalize_brightness(img, 0.0, 1.0)?;
    let blurred = gaussian_blur(&normalized, 1.0)?;

    // 2. Apply binary thresholding
    println!("Applying binary thresholding...");
    let binary = threshold_binary(&blurred, 0.5)?;
    println!("Binary thresholding complete");

    // 3. Apply Otsu's thresholding
    println!("Applying Otsu's thresholding...");
    let (otsu_binary, otsu_threshold) = otsu_threshold(&blurred)?;
    println!("Otsu's threshold value: {:.3}", otsu_threshold);
    println!("Otsu's thresholding complete");

    // 4. Apply adaptive thresholding
    println!("Applying adaptive thresholding (mean method)...");
    let adaptive_mean = adaptive_threshold(&blurred, 11, 0.02, AdaptiveMethod::Mean)?;
    println!("Mean-based adaptive thresholding complete");

    println!("Applying adaptive thresholding (Gaussian method)...");
    let _adaptive_gaussian = adaptive_threshold(&blurred, 11, 0.02, AdaptiveMethod::Gaussian)?;
    println!("Gaussian-based adaptive thresholding complete");

    // 5. Apply connected components labeling
    println!("Performing connected component labeling...");
    let (labeled, num_components) = connected_components(&binary)?;
    println!("Found {} distinct components", num_components);
    println!("Connected component labeling complete");

    // Print some information about the results
    println!("Segmentation results:");
    println!(
        "- Original image dimensions: {}x{}",
        img.width(),
        img.height()
    );
    println!(
        "- Binary image dimensions: {}x{}",
        binary.width(),
        binary.height()
    );
    println!(
        "- Otsu binary image dimensions: {}x{}",
        otsu_binary.width(),
        otsu_binary.height()
    );
    println!(
        "- Adaptive (mean) image dimensions: {}x{}",
        adaptive_mean.width(),
        adaptive_mean.height()
    );
    println!(
        "- Labeled components image dimensions: {}x{}",
        labeled.width(),
        labeled.height()
    );

    println!("Segmentation processing complete!");

    Ok(())
}
