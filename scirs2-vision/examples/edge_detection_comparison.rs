//! Example comparing different edge detection algorithms
//!
//! This example shows how to use various edge detection methods:
//! - Sobel edge detection
//! - Canny edge detection
//! - Prewitt edge detection
//! - Laplacian edge detection
//! - Laplacian of Gaussian (LoG)

use image::DynamicImage;
use scirs2_vision::feature::{
    canny_simple, laplacian_edges, laplacian_of_gaussian, laplacian_zero_crossing, prewitt_edges,
    sobel_edges,
};
use scirs2_vision::preprocessing::{gaussian_blur, normalize_brightness};
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("SciRS2 Vision - Edge Detection Comparison");

    // In a real application, you would provide your own image file path
    let image_path = "examples/input/input.jpg"; // Change this to your image path
    println!("Attempting to load image from: {}", image_path);

    // Check if the image file exists
    let path = PathBuf::from(image_path);
    let img = if !path.exists() {
        println!("Image file not found. This example needs an input image.");
        println!("Creating a demo image with geometric shapes...");

        // Create a demo image with various features
        let mut img_buffer = image::ImageBuffer::new(200, 200);

        // Background
        for y in 0..200 {
            for x in 0..200 {
                img_buffer.put_pixel(x, y, image::Luma([128u8]));
            }
        }

        // Rectangle
        for y in 20..80 {
            for x in 20..80 {
                img_buffer.put_pixel(x, y, image::Luma([255u8]));
            }
        }

        // Circle (approximate)
        let center = (150.0, 50.0);
        let radius = 30.0;
        for y in 0..200 {
            for x in 0..200 {
                let dist = ((x as f32 - center.0).powi(2) + (y as f32 - center.1).powi(2)).sqrt();
                if dist <= radius {
                    img_buffer.put_pixel(x, y, image::Luma([0u8]));
                }
            }
        }

        // Diagonal line
        for i in 0..100 {
            let x = i + 50;
            let y = i + 100;
            if x < 200 && y < 200 {
                for d in -2..=2 {
                    if x as i32 + d >= 0 && x as i32 + d < 200 {
                        img_buffer.put_pixel((x as i32 + d) as u32, y, image::Luma([255u8]));
                    }
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
    let normalized = normalize_brightness(&img, 0.0, 1.0)?;
    let blurred = gaussian_blur(&normalized, 0.5)?;

    // 1. Sobel edge detection
    println!("\n1. Applying Sobel edge detection...");
    let sobel_result = sobel_edges(&blurred, 0.1)?;
    println!("   Sobel edges detected");

    // 2. Canny edge detection
    println!("\n2. Applying Canny edge detection...");
    let canny_result = canny_simple(&blurred, 1.0)?;
    println!("   Canny edges detected");

    // 3. Prewitt edge detection
    println!("\n3. Applying Prewitt edge detection...");
    let prewitt_result = prewitt_edges(&blurred, 0.1)?;
    println!("   Prewitt edges detected");

    // 4. Laplacian edge detection (4-connected)
    println!("\n4. Applying Laplacian edge detection (4-connected)...");
    let laplacian_4_result = laplacian_edges(&blurred, 0.05, false)?;
    println!("   Laplacian (4-connected) edges detected");

    // 5. Laplacian edge detection (8-connected)
    println!("\n5. Applying Laplacian edge detection (8-connected)...");
    let laplacian_8_result = laplacian_edges(&blurred, 0.05, true)?;
    println!("   Laplacian (8-connected) edges detected");

    // 6. Laplacian of Gaussian (LoG)
    println!("\n6. Applying Laplacian of Gaussian...");
    let log_result = laplacian_of_gaussian(&img, 2.0, 0.05)?;
    println!("   LoG edges detected");

    // 7. Laplacian zero-crossing
    println!("\n7. Applying Laplacian zero-crossing detection...");
    let zero_crossing_result = laplacian_zero_crossing(&blurred, true)?;
    println!("   Zero-crossing edges detected");

    // Compare edge point counts
    println!("\nEdge Detection Results Comparison:");
    println!("----------------------------------");

    let count_edges =
        |img: &image::GrayImage| -> usize { img.pixels().filter(|p| p[0] > 0).count() };

    println!(
        "Sobel:                  {} edge pixels",
        count_edges(&sobel_result)
    );
    println!(
        "Canny:                  {} edge pixels",
        count_edges(&canny_result)
    );
    println!(
        "Prewitt:                {} edge pixels",
        count_edges(&prewitt_result)
    );
    println!(
        "Laplacian (4-conn):     {} edge pixels",
        count_edges(&laplacian_4_result)
    );
    println!(
        "Laplacian (8-conn):     {} edge pixels",
        count_edges(&laplacian_8_result)
    );
    println!(
        "Laplacian of Gaussian:  {} edge pixels",
        count_edges(&log_result)
    );
    println!(
        "Zero-crossing:          {} edge pixels",
        count_edges(&zero_crossing_result)
    );

    // Save results
    println!("\nSaving edge detection results...");
    sobel_result.save("examples/output/sobel_edges.png")?;
    canny_result.save("examples/output/canny_edges.png")?;
    prewitt_result.save("examples/output/prewitt_edges.png")?;
    laplacian_4_result.save("examples/output/laplacian_4_edges.png")?;
    laplacian_8_result.save("examples/output/laplacian_8_edges.png")?;
    log_result.save("examples/output/log_edges.png")?;
    zero_crossing_result.save("examples/output/zero_crossing_edges.png")?;

    println!("\nAll edge detection results saved successfully!");
    println!("\nKey Observations:");
    println!("- Sobel and Prewitt are first-order edge detectors (gradient-based)");
    println!("- Canny provides thin, well-connected edges with noise suppression");
    println!("- Laplacian is a second-order detector, sensitive to noise");
    println!("- LoG combines Gaussian smoothing with Laplacian for better noise handling");
    println!("- Zero-crossing detection finds edges at sign changes in the Laplacian");

    Ok(())
}
