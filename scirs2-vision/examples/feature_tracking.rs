//! Feature tracking example
//!
//! This example demonstrates how to track feature points across multiple frames
//! using the Lucas-Kanade tracker.

use image::{DynamicImage, ImageBuffer, Luma, Rgb, RgbImage};
use scirs2_core::random::prelude::*;
use scirs2_vision::error::Result;
use scirs2_vision::feature::{
    extract_feature_coordinates, harris_corners, LKTracker, TrackerParams,
};

#[allow(dead_code)]
fn main() -> Result<()> {
    println!("Feature Tracking Example");
    println!("=======================");

    // Create a sequence of synthetic frames with moving objects
    let frames = create_synthetic_sequence()?;

    // Initialize feature tracker
    let mut tracker = LKTracker::new(TrackerParams {
        max_features: 100,
        min_confidence: 0.5,
        max_distance: 30.0,
        use_backwards_check: true,
        backwards_threshold: 2.0,
        ..Default::default()
    });

    println!("Processing {} frames...", frames.len());

    for (i, frame) in frames.iter().enumerate() {
        println!("\nFrame {i}");

        if i == 0 {
            // Detect initial features using Harris corner detector
            let corners = harris_corners(frame, 7, 0.04, 0.01)?;
            let corner_coords = extract_feature_coordinates(&corners);

            // Convert to float coordinates
            let initial_features: Vec<(f32, f32)> = corner_coords
                .iter()
                .map(|&(x, y)| (x as f32, y as f32))
                .collect();

            println!("Detected {} initial features", initial_features.len());

            // Update tracker with initial features
            tracker.update(frame, Some(&initial_features))?;
        } else {
            // Track features from previous frame
            tracker.update(frame, None)?;
        }

        let features = tracker.get_features();
        println!("Tracking {} features", features.len());

        // Print feature statistics
        if !features.is_empty() {
            let avg_confidence =
                features.iter().map(|f| f.confidence).sum::<f32>() / features.len() as f32;
            let max_age = features.iter().map(|f| f.age).max().unwrap_or(0);
            let avg_velocity = {
                let vel_mag: f32 = features
                    .iter()
                    .map(|f| (f.velocity.0.powi(2) + f.velocity.1.powi(2)).sqrt())
                    .sum();
                vel_mag / features.len() as f32
            };

            println!("  Average confidence: {avg_confidence:.3}");
            println!("  Maximum age: {max_age} frames");
            println!("  Average velocity: {avg_velocity:.2} pixels/frame");
        }

        // Save visualization for the first few frames
        if i < 5 {
            let visualization = visualize_tracking(frame, tracker.get_features())?;
            let output_path = format!("output/tracking_frame_{i:02}.png");

            // Create output directory if it doesn't exist
            std::fs::create_dir_all("output").ok();

            match visualization.save(&output_path) {
                Ok(_) => println!("  Saved visualization: {output_path}"),
                Err(e) => println!("  Warning: Could not save {output_path}: {e}"),
            }
        }
    }

    // Print final tracking statistics
    println!("\nFinal Tracking Results");
    println!("=====================");

    let trajectories = tracker.get_trajectories();
    println!("Total trajectories: {}", trajectories.len());

    let long_tracks = trajectories
        .iter()
        .filter(|(_, positions)| positions.len() > 3)
        .count();
    println!("Long tracks (>3 frames): {long_tracks}");

    if !trajectories.is_empty() {
        let avg_length = trajectories.values().map(|v| v.len()).sum::<usize>() as f32
            / trajectories.len() as f32;
        println!("Average trajectory length: {avg_length:.1} frames");
    }

    println!("\nExample completed successfully!");
    println!("Check the 'output' directory for visualization images.");

    Ok(())
}

/// Create a synthetic sequence with moving objects
#[allow(dead_code)]
fn create_synthetic_sequence() -> Result<Vec<DynamicImage>> {
    let width = 640;
    let height = 480;
    let num_frames = 10;
    let mut frames = Vec::new();

    println!("Creating synthetic sequence...");

    for frame_idx in 0..num_frames {
        let mut img = ImageBuffer::new(width, height);

        // Fill with gradient background
        for y in 0..height {
            for x in 0..width {
                let intensity = ((x + y + frame_idx * 10) % 256) as u8;
                img.put_pixel(x, y, Luma([intensity / 4])); // Dim background
            }
        }

        let t = frame_idx as f32 * 0.1;

        // Add moving circles as trackable features
        let circles = vec![
            // Circle 1: Linear motion
            (100.0 + t * 30.0, 100.0 + t * 20.0, 15.0),
            // Circle 2: Circular motion
            (
                320.0 + 80.0 * (t * 2.0).cos(),
                240.0 + 80.0 * (t * 2.0).sin(),
                12.0,
            ),
            // Circle 3: Oscillating motion
            (450.0 + 50.0 * (t * 3.0).sin(), 150.0, 10.0),
            // Circle 4: Diagonal motion
            (50.0 + t * 40.0, 350.0 - t * 25.0, 8.0),
        ];

        for (cx, cy, radius) in circles {
            draw_circle(&mut img, cx as u32, cy as u32, radius as u32, 255);
        }

        // Add some static corner features
        let corners = vec![
            (50, 50),
            (590, 50),
            (50, 430),
            (590, 430),
            (200, 200),
            (440, 200),
            (320, 350),
        ];

        for (cx, cy) in corners {
            draw_corner(&mut img, cx, cy, 8, 200);
        }

        // Add some noise
        let mut rng = scirs2_core::random::rng();
        for _ in 0..100 {
            let x = rng.random_range(0..width);
            let y = rng.random_range(0..height);
            let intensity = rng.random_range(100u8..200u8);
            if x < width && y < height {
                img.put_pixel(x, y, Luma([intensity]));
            }
        }

        frames.push(DynamicImage::ImageLuma8(img));
    }

    println!("Created {} frames", frames.len());
    Ok(frames)
}

/// Draw a filled circle on the image
#[allow(dead_code)]
fn draw_circle(
    img: &mut ImageBuffer<Luma<u8>, Vec<u8>>,
    cx: u32,
    cy: u32,
    radius: u32,
    intensity: u8,
) {
    let (width, height) = img.dimensions();
    let r_sq = (radius * radius) as i32;

    for y in cy.saturating_sub(radius)..=cy.saturating_add(radius).min(height - 1) {
        for x in cx.saturating_sub(radius)..=cx.saturating_add(radius).min(width - 1) {
            let dx = x as i32 - cx as i32;
            let dy = y as i32 - cy as i32;

            if dx * dx + dy * dy <= r_sq {
                img.put_pixel(x, y, Luma([intensity]));
            }
        }
    }
}

/// Draw a corner pattern
#[allow(dead_code)]
fn draw_corner(
    img: &mut ImageBuffer<Luma<u8>, Vec<u8>>,
    cx: u32,
    cy: u32,
    size: u32,
    intensity: u8,
) {
    let (width, height) = img.dimensions();

    // Draw horizontal line
    for x in cx.saturating_sub(size)..=cx.saturating_add(size).min(width - 1) {
        if cy < height {
            img.put_pixel(x, cy, Luma([intensity]));
        }
    }

    // Draw vertical line
    for y in cy.saturating_sub(size)..=cy.saturating_add(size).min(height - 1) {
        if cx < width {
            img.put_pixel(cx, y, Luma([intensity]));
        }
    }
}

/// Visualize tracked features on the image
#[allow(dead_code)]
fn visualize_tracking(
    frame: &DynamicImage,
    features: &[scirs2_vision::feature::TrackedFeature],
) -> Result<RgbImage> {
    let gray = frame.to_luma8();
    let (width, height) = gray.dimensions();
    let mut rgb_img = RgbImage::new(width, height);

    // Convert grayscale to RGB
    for y in 0..height {
        for x in 0..width {
            let gray_val = gray.get_pixel(x, y)[0];
            rgb_img.put_pixel(x, y, Rgb([gray_val, gray_val, gray_val]));
        }
    }

    // Draw tracked features
    for feature in features {
        let x = feature.position.0 as u32;
        let y = feature.position.1 as u32;

        if x < width && y < height {
            // Color based on confidence: green for high confidence, red for low
            let confidence = feature.confidence.clamp(0.0, 1.0);
            let red = (255.0 * (1.0 - confidence)) as u8;
            let green = (255.0 * confidence) as u8;
            let blue = 0;

            // Draw a small cross
            for dx in -2..=2 {
                for dy in -2..=2 {
                    let px = (x as i32 + dx) as u32;
                    let py = (y as i32 + dy) as u32;

                    if px < width && py < height && (dx == 0 || dy == 0) {
                        rgb_img.put_pixel(px, py, Rgb([red, green, blue]));
                    }
                }
            }

            // Draw velocity vector if significant
            let vel_mag = (feature.velocity.0.powi(2) + feature.velocity.1.powi(2)).sqrt();
            if vel_mag > 1.0 {
                let scale = 5.0; // Scale factor for visualization
                let end_x = (x as f32 + feature.velocity.0 * scale) as u32;
                let end_y = (y as f32 + feature.velocity.1 * scale) as u32;

                if end_x < width && end_y < height {
                    // Simple line drawing (Bresenham would be better)
                    let steps = vel_mag.ceil() as u32;
                    for i in 0..=steps {
                        let t = i as f32 / steps as f32;
                        let line_x = (x as f32 + t * (end_x as f32 - x as f32)) as u32;
                        let line_y = (y as f32 + t * (end_y as f32 - y as f32)) as u32;

                        if line_x < width && line_y < height {
                            rgb_img.put_pixel(line_x, line_y, Rgb([0, 255, 255]));
                            // Cyan for velocity
                        }
                    }
                }
            }
        }
    }

    Ok(rgb_img)
}
