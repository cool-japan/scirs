//! Example demonstrating feature descriptors (ORB, BRIEF, HOG)

use image::{DynamicImage, GenericImageView, Pixel, Rgb, RgbImage};
use scirs2_vision::feature::{
    compute_brief_descriptors, compute_hog, detect_and_compute, detect_and_compute_orb,
    match_brief_descriptors, match_orb_descriptors, visualize_hog, BriefConfig,
    HogConfig, KeyPoint, OrbConfig,
};
use std::env;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get command line arguments
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <image_path>", args[0]);
        std::process::exit(1);
    }

    // Load image
    let img_path = Path::new(&args[1]);
    let img = image::open(img_path)?;
    println!("Loaded image: {:?}", img.dimensions());

    // Convert to RGB for visualization
    let rgb_img = img.to_rgb8();
    let (width, height) = rgb_img.dimensions();

    // 1. ORB Features
    println!("\n=== ORB Features ===");
    let orb_config = OrbConfig::default();
    let orb_descriptors = detect_and_compute_orb(&img, &orb_config)?;
    println!("Detected {} ORB features", orb_descriptors.len());

    // Create visualization for ORB
    let mut orb_img = rgb_img.clone();
    for descriptor in &orb_descriptors {
        let kp = &descriptor.keypoint;
        if kp.x >= 0.0 && kp.x < width as f32 && kp.y >= 0.0 && kp.y < height as f32 {
            // Draw circle
            draw_circle(
                &mut orb_img,
                kp.x as u32,
                kp.y as u32,
                (kp.scale * 3.0) as u32,
                Rgb([255, 0, 0]),
            );

            // Draw orientation
            let end_x = kp.x + kp.scale * 5.0 * kp.orientation.cos();
            let end_y = kp.y + kp.scale * 5.0 * kp.orientation.sin();
            draw_line(&mut orb_img, kp.x, kp.y, end_x, end_y, Rgb([0, 255, 0]));
        }
    }
    orb_img.save("orb_features.png")?;
    println!("Saved ORB visualization to orb_features.png");

    // 2. BRIEF Descriptors (using SIFT-like keypoints)
    println!("\n=== BRIEF Descriptors ===");
    let sift_descriptors = detect_and_compute(&img, 300, 10.0)?;
    let keypoints: Vec<KeyPoint> = sift_descriptors
        .iter()
        .map(|d| d.keypoint.clone())
        .collect();

    let brief_config = BriefConfig::default();
    let brief_descriptors = compute_brief_descriptors(&img, keypoints.clone(), &brief_config)?;
    println!("Computed {} BRIEF descriptors", brief_descriptors.len());

    // Create visualization for BRIEF
    let mut brief_img = rgb_img.clone();
    for descriptor in &brief_descriptors {
        let kp = &descriptor.keypoint;
        if kp.x >= 0.0 && kp.x < width as f32 && kp.y >= 0.0 && kp.y < height as f32 {
            draw_circle(
                &mut brief_img,
                kp.x as u32,
                kp.y as u32,
                3,
                Rgb([0, 255, 0]),
            );
        }
    }
    brief_img.save("brief_features.png")?;
    println!("Saved BRIEF visualization to brief_features.png");

    // 3. HOG Features
    println!("\n=== HOG Features ===");
    let hog_config = HogConfig::default();
    let hog_descriptor = compute_hog(&img, &hog_config)?;
    println!(
        "Computed HOG descriptor with {} features for {}x{} blocks",
        hog_descriptor.features.len(),
        hog_descriptor.blocks_x,
        hog_descriptor.blocks_y
    );

    // Visualize HOG
    let hog_vis = visualize_hog(&hog_descriptor, hog_config.cell_size, hog_config.num_bins);
    let hog_img = scirs2_vision::feature::array_to_image(&hog_vis)?;
    hog_img.save("hog_visualization.png")?;
    println!("Saved HOG visualization to hog_visualization.png");

    // 4. Feature Matching Example
    if let Ok(img2) = image::open(img_path) {
        println!("\n=== Feature Matching ===");

        // Simulate a transformed image by rotating slightly
        let img2 = img2; // In practice, load a second image

        // Match ORB descriptors
        let orb_descriptors2 = detect_and_compute_orb(&img2, &orb_config)?;
        let orb_matches = match_orb_descriptors(&orb_descriptors, &orb_descriptors2, 64);
        println!("Found {} ORB matches", orb_matches.len());

        // Match BRIEF descriptors
        let brief_descriptors2 =
            compute_brief_descriptors(&img2, keypoints.clone(), &brief_config)?;
        let brief_matches = match_brief_descriptors(&brief_descriptors, &brief_descriptors2, 64);
        println!("Found {} BRIEF matches", brief_matches.len());

        // Create match visualization for ORB
        let mut match_img = create_side_by_side(&img, &img2);
        for (idx1, idx2, _distance) in orb_matches.iter().take(50) {
            let kp1 = &orb_descriptors[*idx1].keypoint;
            let kp2 = &orb_descriptors2[*idx2].keypoint;

            let x2_offset = width;
            draw_line(
                &mut match_img,
                kp1.x,
                kp1.y,
                kp2.x + x2_offset as f32,
                kp2.y,
                Rgb([255, 255, 0]),
            );
        }
        match_img.save("orb_matches.png")?;
        println!("Saved match visualization to orb_matches.png");
    }

    println!("\nFeature descriptor example completed successfully!");
    Ok(())
}

/// Draw a circle on an image
fn draw_circle(img: &mut RgbImage, cx: u32, cy: u32, radius: u32, color: Rgb<u8>) {
    let (width, height) = img.dimensions();

    // Simple circle drawing using midpoint algorithm
    let mut x = radius as i32;
    let mut y = 0i32;
    let mut err = 0i32;

    while x >= y {
        // Draw 8 octants
        let points = [
            (cx as i32 + x, cy as i32 + y),
            (cx as i32 + y, cy as i32 + x),
            (cx as i32 - y, cy as i32 + x),
            (cx as i32 - x, cy as i32 + y),
            (cx as i32 - x, cy as i32 - y),
            (cx as i32 - y, cy as i32 - x),
            (cx as i32 + y, cy as i32 - x),
            (cx as i32 + x, cy as i32 - y),
        ];

        for (px, py) in &points {
            if *px >= 0 && *px < width as i32 && *py >= 0 && *py < height as i32 {
                img.put_pixel(*px as u32, *py as u32, color);
            }
        }

        if err <= 0 {
            y += 1;
            err += 2 * y + 1;
        }

        if err > 0 {
            x -= 1;
            err -= 2 * x + 1;
        }
    }
}

/// Draw a line on an image
fn draw_line(img: &mut RgbImage, x0: f32, y0: f32, x1: f32, y1: f32, color: Rgb<u8>) {
    let (width, height) = img.dimensions();

    // Bresenham's line algorithm
    let dx = (x1 - x0).abs();
    let dy = (y1 - y0).abs();
    let sx = if x0 < x1 { 1.0 } else { -1.0 };
    let sy = if y0 < y1 { 1.0 } else { -1.0 };

    let mut x = x0;
    let mut y = y0;
    let mut err = dx - dy;

    loop {
        if x >= 0.0 && x < width as f32 && y >= 0.0 && y < height as f32 {
            img.put_pixel(x as u32, y as u32, color);
        }

        if (x - x1).abs() < 0.5 && (y - y1).abs() < 0.5 {
            break;
        }

        let e2 = 2.0 * err;
        if e2 > -dy {
            err -= dy;
            x += sx;
        }
        if e2 < dx {
            err += dx;
            y += sy;
        }
    }
}

/// Create a side-by-side image for matching visualization
fn create_side_by_side(img1: &DynamicImage, img2: &DynamicImage) -> RgbImage {
    let (w1, h1) = img1.dimensions();
    let (w2, h2) = img2.dimensions();

    let width = w1 + w2;
    let height = h1.max(h2);

    let mut combined = RgbImage::new(width, height);

    // Copy first image
    for y in 0..h1 {
        for x in 0..w1 {
            let pixel = img1.get_pixel(x, y);
            combined.put_pixel(x, y, pixel.to_rgb());
        }
    }

    // Copy second image
    for y in 0..h2 {
        for x in 0..w2 {
            let pixel = img2.get_pixel(x, y);
            combined.put_pixel(x + w1, y, pixel.to_rgb());
        }
    }

    combined
}
