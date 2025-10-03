//! Example demonstrating morphological operations
//!
//! This example shows how to:
//! 1. Apply erosion and dilation
//! 2. Perform opening and closing
//! 3. Calculate morphological gradient
//! 4. Apply top-hat and black-hat transforms

use image::DynamicImage;
use scirs2_vision::preprocessing::{
    black_hat, closing, dilate, erode, morphological_gradient, opening, top_hat, StructuringElement,
};
use std::path::PathBuf;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("SciRS2 Vision - Morphological Operations Example");

    // In a real application, you would provide your own image file path
    let image_path = "examples/input/input.jpg"; // Change this to your image path
    println!("Attempting to load image from: {image_path}");

    // Check if the image file exists
    let path = PathBuf::from(image_path);
    if !path.exists() {
        println!("Image file not found. This example needs an input image.");
        println!("Please provide an image path as argument or place an 'input.jpg' in the current directory.");

        // For demo purposes, we'll create a simple 100x100 binary image with some shapes
        println!("Creating a demo binary image for demonstration...");
        let mut img_buffer = image::ImageBuffer::new(100, 100);

        // Fill with background (black)
        for y in 0..100 {
            for x in 0..100 {
                img_buffer.put_pixel(x, y, image::Luma([0]));
            }
        }

        // Draw a rectangle
        for y in 20..40 {
            for x in 20..60 {
                img_buffer.put_pixel(x, y, image::Luma([255]));
            }
        }

        // Draw a circle
        for y in 0..100 {
            for x in 0..100 {
                let dx = (x as f32 - 70.0).powi(2);
                let dy = (y as f32 - 70.0).powi(2);
                let d = (dx + dy).sqrt();
                if d < 15.0 {
                    img_buffer.put_pixel(x, y, image::Luma([255]));
                }
            }
        }

        // Add some noise
        for _ in 0..100 {
            let x = scirs2_core::random::random::<u32>() % 100;
            let y = scirs2_core::random::random::<u32>() % 100;
            img_buffer.put_pixel(x, y, image::Luma([255]));
        }

        let img = DynamicImage::ImageLuma8(img_buffer);
        process_image(&img)?;
        return Ok(());
    }

    // Load image and convert to grayscale
    let img = image::open(path)?.to_luma8();
    let img = DynamicImage::ImageLuma8(img);
    println!(
        "Successfully loaded image: {}x{}",
        img.width(),
        img.height()
    );

    process_image(&img)?;

    Ok(())
}

#[allow(dead_code)]
fn process_image(img: &DynamicImage) -> Result<(), Box<dyn std::error::Error>> {
    // Define structuring elements
    let rect_se = StructuringElement::Rectangle(3, 3);
    let ellipse_se = StructuringElement::Ellipse(5, 5);
    let cross_se = StructuringElement::Cross(3);

    // 1. Basic morphological operations
    println!("Applying erosion...");
    let eroded = erode(img, rect_se)?;
    println!("Erosion complete");

    println!("Applying dilation...");
    let dilated = dilate(img, rect_se)?;
    println!("Dilation complete");

    // 2. Compound morphological operations
    println!("Applying opening...");
    let opened = opening(img, ellipse_se)?;
    println!("Opening complete");

    println!("Applying closing...");
    let closed = closing(img, ellipse_se)?;
    println!("Closing complete");

    // 3. Morphological gradient
    println!("Calculating morphological gradient...");
    let gradient = morphological_gradient(img, cross_se)?;
    println!("Gradient calculation complete");

    // 4. Top-hat and black-hat transforms
    println!("Applying top-hat transform...");
    let _top_hat_result = top_hat(img, ellipse_se)?;
    println!("Top-hat transform complete");

    println!("Applying black-hat transform...");
    let _black_hat_result = black_hat(img, ellipse_se)?;
    println!("Black-hat transform complete");

    // Print some information about the results
    println!("Morphological operation results:");
    println!(
        "- Original image dimensions: {}x{}",
        img.width(),
        img.height()
    );
    println!(
        "- Eroded image dimensions: {}x{}",
        eroded.width(),
        eroded.height()
    );
    println!(
        "- Dilated image dimensions: {}x{}",
        dilated.width(),
        dilated.height()
    );
    println!(
        "- Opened image dimensions: {}x{}",
        opened.width(),
        opened.height()
    );
    println!(
        "- Closed image dimensions: {}x{}",
        closed.width(),
        closed.height()
    );
    println!(
        "- Gradient image dimensions: {}x{}",
        gradient.width(),
        gradient.height()
    );

    println!("Morphological operations complete!");

    Ok(())
}
