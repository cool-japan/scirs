//! Example demonstrating feature matching and homography estimation

use image::{GenericImageView, Pixel, Rgba, RgbaImage};
use scirs2_vision::feature::{
    detect_and_compute_orb, find_homography_from_matches, match_orb_descriptors,
    OrbConfig,
};
use scirs2_vision::transform::{warp_affine, AffineTransform};
use std::env;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get command line arguments
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: {} <image1_path> <image2_path>", args[0]);
        std::process::exit(1);
    }

    // Load images
    let img1_path = Path::new(&args[1]);
    let img2_path = Path::new(&args[2]);

    let img1 = image::open(img1_path)?;
    let img2 = image::open(img2_path)?;

    println!("Loaded images:");
    println!("  Image 1: {:?}", img1.dimensions());
    println!("  Image 2: {:?}", img2.dimensions());

    // Step 1: Detect ORB features
    println!("\nDetecting ORB features...");
    let orb_config = OrbConfig {
        num_features: 1000,
        scale_factor: 1.2,
        num_levels: 8,
        fast_threshold: 20,
        use_harris_detector: true,
        patch_size: 31,
    };

    let descriptors1 = detect_and_compute_orb(&img1, &orb_config)?;
    let descriptors2 = detect_and_compute_orb(&img2, &orb_config)?;

    println!("  Found {} features in image 1", descriptors1.len());
    println!("  Found {} features in image 2", descriptors2.len());

    // Step 2: Match features
    println!("\nMatching features...");
    let matches = match_orb_descriptors(&descriptors1, &descriptors2, 64);
    println!("  Found {} matches", matches.len());

    if matches.len() < 10 {
        println!("Not enough matches found for reliable homography estimation");
        std::process::exit(1);
    }

    // Step 3: Estimate homography
    println!("\nEstimating homography...");
    // Convert matches to the right format (from u32 to f32 distance)
    let matches_f32: Vec<(usize, usize, f32)> = matches
        .iter()
        .map(|(idx1, idx2, dist)| (*idx1, *idx2, *dist as f32))
        .collect();
    
    let (homography, mask) = find_homography_from_matches(
        &matches_f32,
        &descriptors1
            .iter()
            .map(|d| d.keypoint.clone())
            .collect::<Vec<_>>(),
        &descriptors2
            .iter()
            .map(|d| d.keypoint.clone())
            .collect::<Vec<_>>(),
        3.0,
        0.99,
    )?;

    let inliers = mask.iter().filter(|&&x| x).count();
    println!(
        "  Found {} inliers out of {} matches",
        inliers,
        matches.len()
    );

    // Step 4: Create visualizations
    // Draw matches
    let match_img = draw_matches(&img1, &img2, &descriptors1, &descriptors2, &matches, &mask);
    match_img.save("matches.png")?;
    println!("  Saved match visualization to matches.png");

    // Warp image 1 to align with image 2
    println!("\nWarping image...");

    // Use the homography matrix to create an affine transformation
    // Note: This is a simplification as homography is more general than affine
    let h = &homography.matrix;
    let affine = AffineTransform::new(
        h[[0, 0]],
        h[[0, 1]],
        h[[0, 2]],
        h[[1, 0]],
        h[[1, 1]],
        h[[1, 2]],
    );

    // BorderMode::Transparent for warp_affine
    let warped = warp_affine(
        &img1,
        &affine,
        Some(img2.width()),
        Some(img2.height()),
        scirs2_vision::transform::affine::BorderMode::Transparent,
    )?;

    warped.save("warped.png")?;
    println!("  Saved warped image to warped.png");

    // Create composite image showing alignment
    let composite = create_composite(&img2, &warped);
    composite.save("composite.png")?;
    println!("  Saved composite image to composite.png");

    println!("\nImage matching and alignment completed successfully!");
    Ok(())
}

/// Draw matches between two images
fn draw_matches(
    img1: &image::DynamicImage,
    img2: &image::DynamicImage,
    kps1: &[scirs2_vision::feature::OrbDescriptor],
    kps2: &[scirs2_vision::feature::OrbDescriptor],
    matches: &[(usize, usize, u32)],
    mask: &[bool],
) -> RgbaImage {
    let (w1, h1) = img1.dimensions();
    let (w2, h2) = img2.dimensions();

    let height = h1.max(h2);
    let width = w1 + w2;

    let mut result = RgbaImage::new(width, height);

    // Copy first image to left side
    for y in 0..h1 {
        for x in 0..w1 {
            let pixel = img1.get_pixel(x, y);
            result.put_pixel(x, y, pixel.to_rgba());
        }
    }

    // Copy second image to right side
    for y in 0..h2 {
        for x in 0..w2 {
            let pixel = img2.get_pixel(x, y);
            result.put_pixel(x + w1, y, pixel.to_rgba());
        }
    }

    // Draw matches
    for (i, &(idx1, idx2, _)) in matches.iter().enumerate() {
        let kp1 = &kps1[idx1].keypoint;
        let kp2 = &kps2[idx2].keypoint;

        let x1 = kp1.x as u32;
        let y1 = kp1.y as u32;
        let x2 = kp2.x as u32 + w1;
        let y2 = kp2.y as u32;

        // Use green for inliers, red for outliers
        let color = if i < mask.len() && mask[i] {
            Rgba([0, 255, 0, 255]) // Green for inliers
        } else {
            Rgba([255, 0, 0, 255]) // Red for outliers
        };

        // Draw circles for keypoints
        draw_circle(&mut result, x1, y1, 3, Rgba([255, 0, 0, 255]));
        draw_circle(&mut result, x2, y2, 3, Rgba([255, 0, 0, 255]));

        // Draw line connecting matches
        draw_line(&mut result, x1, y1, x2, y2, color);
    }

    result
}

/// Create a composite image showing alignment
fn create_composite(base: &image::DynamicImage, aligned: &image::DynamicImage) -> RgbaImage {
    let (width, height) = base.dimensions();
    let mut result = RgbaImage::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let pixel1 = base.get_pixel(x, y).to_rgba();
            let pixel2 = aligned.get_pixel(x, y).to_rgba();

            // Skip transparent pixels
            if pixel2[3] < 128 {
                result.put_pixel(x, y, pixel1);
                continue;
            }

            // Create a blend: red channel from image 1, green from image 2
            let composite = Rgba([pixel1[0], pixel2[1], 0, 255]);

            result.put_pixel(x, y, composite);
        }
    }

    result
}

/// Draw a circle on an image
fn draw_circle(img: &mut RgbaImage, cx: u32, cy: u32, radius: u32, color: Rgba<u8>) {
    let (width, height) = img.dimensions();

    for y in (cy.saturating_sub(radius))..=(cy + radius).min(height - 1) {
        for x in (cx.saturating_sub(radius))..=(cx + radius).min(width - 1) {
            let dx = x as i32 - cx as i32;
            let dy = y as i32 - cy as i32;

            if dx * dx + dy * dy <= radius as i32 * radius as i32 {
                img.put_pixel(x, y, color);
            }
        }
    }
}

/// Draw a line on an image
fn draw_line(img: &mut RgbaImage, x0: u32, y0: u32, x1: u32, y1: u32, color: Rgba<u8>) {
    let (width, height) = img.dimensions();

    // Bresenham's line algorithm
    let dx = (x1 as i32 - x0 as i32).abs();
    let dy = (y1 as i32 - y0 as i32).abs();

    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };

    let mut err = dx - dy;
    let mut x = x0 as i32;
    let mut y = y0 as i32;

    while x >= 0 && x < width as i32 && y >= 0 && y < height as i32 {
        img.put_pixel(x as u32, y as u32, color);

        if x == x1 as i32 && y == y1 as i32 {
            break;
        }

        let e2 = 2 * err;
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
