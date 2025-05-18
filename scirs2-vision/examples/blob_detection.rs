//! Example demonstrating blob and region detection using various algorithms

use image::GenericImageView;
use scirs2_vision::feature::{
    blobs_to_image, dog_detect, draw_circles, hough_circles, log_blob_detect, log_blobs_to_image,
    mser_detect, mser_to_image, DogConfig, HoughCircleConfig, LogBlobConfig, MserConfig,
};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    // Load input image
    let image_path = "examples/input/input.jpg";
    let img = image::open(image_path)?;
    let (width, height) = img.dimensions();

    println!("Loaded image: {}x{}", width, height);

    // 1. Difference of Gaussians (DoG) blob detection
    println!("Applying DoG blob detection...");
    let dog_config = DogConfig {
        num_octaves: 4,
        scales_per_octave: 4,
        initial_sigma: 1.6,
        threshold: 0.01,
        non_max_suppression: true,
    };

    let dog_blobs = dog_detect(&img, dog_config)?;
    println!("Detected {} DoG blobs", dog_blobs.len());

    let dog_result = blobs_to_image(&dog_blobs, width, height)?;
    dog_result.save("examples/output/dog_blobs.png")?;

    // 2. Laplacian of Gaussian (LoG) blob detection
    println!("Applying LoG blob detection...");
    let log_config = LogBlobConfig {
        num_scales: 15,
        min_sigma: 2.0,
        max_sigma: 30.0,
        threshold: 0.01,
        non_max_suppression: true,
    };

    let log_blobs = log_blob_detect(&img, log_config)?;
    println!("Detected {} LoG blobs", log_blobs.len());

    let log_result = log_blobs_to_image(&log_blobs, width, height)?;
    log_result.save("examples/output/log_blobs.png")?;

    // 3. MSER (Maximally Stable Extremal Regions) detection
    println!("Applying MSER detection...");
    let mser_config = MserConfig {
        delta: 5,
        min_area: 60,
        max_area: 14400,
        max_variation: 0.25,
        min_diversity: 0.2,
    };

    let mser_regions = mser_detect(&img, mser_config)?;
    println!("Detected {} MSER regions", mser_regions.len());

    let mser_result = mser_to_image(&mser_regions, width, height)?;
    mser_result.save("examples/output/mser_regions.png")?;

    // 4. Hough Circle Transform
    println!("Applying Hough Circle Transform...");
    let hough_config = HoughCircleConfig {
        min_radius: 10,
        max_radius: 60,
        threshold: 0.3,
        min_distance: 20,
        edge_threshold: 0.1,
        max_circles: Some(10),
    };

    let circles = hough_circles(&img, hough_config)?;
    println!("Detected {} circles", circles.len());

    // Draw circles on a copy of the original image
    let mut circle_result = img.to_luma8();
    draw_circles(&mut circle_result, &circles, 255);
    circle_result.save("examples/output/hough_circles.png")?;

    // Print information about detected features
    println!("\nDoG Blobs (top 5):");
    for (i, blob) in dog_blobs.iter().take(5).enumerate() {
        println!(
            "  {} - Position: ({}, {}), Scale: {:.2}, Response: {:.4}",
            i + 1,
            blob.x,
            blob.y,
            blob.scale,
            blob.response
        );
    }

    println!("\nLoG Blobs (top 5):");
    for (i, blob) in log_blobs.iter().take(5).enumerate() {
        println!(
            "  {} - Position: ({}, {}), Sigma: {:.2}, Response: {:.4}",
            i + 1,
            blob.x,
            blob.y,
            blob.sigma,
            blob.response
        );
    }

    println!("\nMSER Regions (top 5):");
    for (i, region) in mser_regions.iter().take(5).enumerate() {
        println!(
            "  {} - Area: {}, Level: {}, Stability: {:.4}",
            i + 1,
            region.area,
            region.level,
            region.stability
        );
    }

    println!("\nHough Circles:");
    for (i, circle) in circles.iter().enumerate() {
        println!(
            "  {} - Center: ({}, {}), Radius: {}, Confidence: {:.4}",
            i + 1,
            circle.center_x,
            circle.center_y,
            circle.radius,
            circle.confidence
        );
    }

    println!("\nAll results saved to examples/output/");

    Ok(())
}
