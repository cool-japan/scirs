//! Example demonstrating color transformation functionality
//!
//! This example shows how to:
//! 1. Convert between color spaces (RGB, HSV, LAB)
//! 2. Split and merge color channels
//! 3. Apply weighted grayscale conversion

use scirs2_vision::color::{
    hsv_to_rgb, lab_to_rgb, merge_channels, rgb_to_grayscale, rgb_to_hsv, rgb_to_lab,
    split_channels,
};
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("SciRS2 Vision - Color Transformations Example");

    // In a real application, you would provide your own image file path
    let image_path = "examples/input/input.jpg"; // Change this to your image path
    println!("Attempting to load image from: {}", image_path);

    // Check if the image file exists
    let path = PathBuf::from(image_path);
    if !path.exists() {
        println!("Image file not found. This example needs an input image.");
        println!("Please provide an image path as argument or place an 'input.jpg' in the current directory.");

        // For demo purposes, we'll create a simple 100x100 color gradient image
        println!("Creating a demo color gradient image for demonstration...");
        let mut img_buffer = image::ImageBuffer::new(100, 100);

        for y in 0..100 {
            for x in 0..100 {
                let r = ((x as f32 / 100.0) * 255.0) as u8;
                let g = ((y as f32 / 100.0) * 255.0) as u8;
                let b = ((x as f32 + y as f32) / 200.0 * 255.0) as u8;
                img_buffer.put_pixel(x, y, image::Rgb([r, g, b]));
            }
        }

        let img = image::DynamicImage::ImageRgb8(img_buffer);
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

fn process_image(img: &image::DynamicImage) -> Result<(), Box<dyn std::error::Error>> {
    // 1. RGB to Grayscale conversion
    println!("Converting to grayscale...");
    let grayscale = rgb_to_grayscale(img, None)?;
    println!("Standard grayscale conversion complete");

    // Custom weights for grayscale (emphasize green channel)
    let _custom_grayscale = rgb_to_grayscale(img, Some([0.1, 0.8, 0.1]))?;
    println!("Custom-weighted grayscale conversion complete");

    // 2. RGB to HSV conversion
    println!("Converting RGB to HSV...");
    let hsv = rgb_to_hsv(img)?;

    // Convert back to RGB
    println!("Converting HSV back to RGB...");
    let _rgb_from_hsv = hsv_to_rgb(&hsv)?;
    println!("HSV conversions complete");

    // 3. RGB to LAB conversion
    println!("Converting RGB to LAB...");
    let lab = rgb_to_lab(img)?;

    // Convert back to RGB
    println!("Converting LAB back to RGB...");
    let _rgb_from_lab = lab_to_rgb(&lab)?;
    println!("LAB conversions complete");

    // 4. Channel splitting and merging
    println!("Splitting color channels...");
    let (r_channel, g_channel, b_channel) = split_channels(img)?;

    println!("Merging color channels...");
    let merged = merge_channels(&r_channel, &g_channel, &b_channel)?;
    println!("Channel operations complete");

    // Print some information about the results
    println!("Color transformation results:");
    println!(
        "- Original image dimensions: {}x{}",
        img.width(),
        img.height()
    );
    println!(
        "- Grayscale image dimensions: {}x{}",
        grayscale.width(),
        grayscale.height()
    );
    println!("- HSV image dimensions: {}x{}", hsv.width(), hsv.height());
    println!("- LAB image dimensions: {}x{}", lab.width(), lab.height());
    println!(
        "- Merged channels image dimensions: {}x{}",
        merged.width(),
        merged.height()
    );

    println!("Color transformations complete!");

    Ok(())
}
