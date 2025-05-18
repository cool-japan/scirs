//! Example comparing different corner detection algorithms
//!
//! This example shows how to use various corner detection methods:
//! - Harris corner detection
//! - Shi-Tomasi corner detection (Good Features to Track)
//! - FAST corner detection

use image::DynamicImage;
use scirs2_vision::feature::{
    extract_feature_coordinates, fast_corners_simple, good_features_to_track, harris_corners,
    shi_tomasi_corners_simple,
};
use scirs2_vision::preprocessing::gaussian_blur;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("SciRS2 Vision - Corner Detection Comparison");

    // In a real application, you would provide your own image file path
    let image_path = "examples/input/input.jpg"; // Change this to your image path
    println!("Attempting to load image from: {}", image_path);

    // Check if the image file exists
    let path = PathBuf::from(image_path);
    let img = if !path.exists() {
        println!("Image file not found. This example needs an input image.");
        println!("Creating a demo checkerboard pattern...");

        // Create a demo checkerboard pattern
        let mut img_buffer = image::ImageBuffer::new(200, 200);

        // Background
        for y in 0..200 {
            for x in 0..200 {
                img_buffer.put_pixel(x, y, image::Luma([128u8]));
            }
        }

        // Checkerboard pattern
        let square_size = 25;
        for i in 0..8 {
            for j in 0..8 {
                if (i + j) % 2 == 0 {
                    let start_x = i * square_size;
                    let start_y = j * square_size;

                    for y in start_y..(start_y + square_size).min(200) {
                        for x in start_x..(start_x + square_size).min(200) {
                            img_buffer.put_pixel(x as u32, y as u32, image::Luma([255u8]));
                        }
                    }
                }
            }
        }

        // Add a rotated square in the middle
        let center = 100.0;
        let half_size = 30.0;
        let angle = std::f32::consts::PI / 6.0; // 30 degrees

        for y in 50..150 {
            for x in 50..150 {
                let dx = x as f32 - center;
                let dy = y as f32 - center;

                // Rotate coordinates
                let rx = dx * angle.cos() + dy * angle.sin();
                let ry = -dx * angle.sin() + dy * angle.cos();

                if rx.abs() <= half_size && ry.abs() <= half_size {
                    img_buffer.put_pixel(x, y, image::Luma([0u8]));
                }
            }
        }

        DynamicImage::ImageLuma8(img_buffer)
    } else {
        image::open(path)?
    };

    println!(
        "Successfully loaded image: {}x{}",
        img.width(),
        img.height()
    );

    // Preprocess the image
    println!("\nPreprocessing image...");
    let blurred = gaussian_blur(&img, 1.0)?;

    // 1. Harris corner detection
    println!("\n1. Applying Harris corner detection...");
    let harris_result = harris_corners(&blurred, 3, 0.04, 0.01)?;
    let harris_points = extract_feature_coordinates(&harris_result);
    println!("   Harris: {} corners detected", harris_points.len());

    // 2. Shi-Tomasi corner detection
    println!("\n2. Applying Shi-Tomasi corner detection...");
    let shi_tomasi_result = shi_tomasi_corners_simple(&blurred, 100)?;
    let shi_tomasi_points = extract_feature_coordinates(&shi_tomasi_result);
    println!(
        "   Shi-Tomasi: {} corners detected",
        shi_tomasi_points.len()
    );

    // 3. FAST corner detection
    println!("\n3. Applying FAST corner detection...");
    let fast_result = fast_corners_simple(&blurred, 20.0)?;
    let fast_points = extract_feature_coordinates(&fast_result);
    println!("   FAST: {} corners detected", fast_points.len());

    // 4. Good Features to Track (sub-pixel accuracy)
    println!("\n4. Extracting Good Features to Track...");
    let good_features = good_features_to_track(&blurred, 3, 0.01, 50, 10)?;
    println!("   Good Features: {} features found", good_features.len());

    // Show sub-pixel accuracy
    if !good_features.is_empty() {
        println!("\n   First 5 features with sub-pixel accuracy:");
        for (i, (x, y, score)) in good_features.iter().take(5).enumerate() {
            println!(
                "   Feature {}: ({:.2}, {:.2}) with score {:.4}",
                i + 1,
                x,
                y,
                score
            );
        }
    }

    // Compare results
    println!("\nCorner Detection Results Comparison:");
    println!("------------------------------------");
    println!("Harris:            {} corners", harris_points.len());
    println!("Shi-Tomasi:        {} corners", shi_tomasi_points.len());
    println!("FAST:              {} corners", fast_points.len());
    println!("Good Features:     {} features", good_features.len());

    // Save results
    println!("\nSaving corner detection results...");
    harris_result.save("examples/output/harris_corners.png")?;
    shi_tomasi_result.save("examples/output/shi_tomasi_corners.png")?;
    fast_result.save("examples/output/fast_corners.png")?;

    // Create visualization for good features
    let mut features_img = blurred.to_luma8();
    for (x, y, _) in &good_features {
        let px = x.round() as u32;
        let py = y.round() as u32;
        if px < features_img.width() && py < features_img.height() {
            features_img.put_pixel(px, py, image::Luma([255u8]));
        }
    }
    features_img.save("examples/output/good_features.png")?;

    println!("\nAll corner detection results saved successfully!");
    println!("\nKey Observations:");
    println!("- Harris: Classic corner detector, good for well-defined corners");
    println!("- Shi-Tomasi: Improved version of Harris, better feature quality");
    println!("- FAST: Very efficient, designed for real-time applications");
    println!("- Good Features: Provides sub-pixel accuracy, best for tracking");

    Ok(())
}
