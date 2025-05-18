//! Example demonstrating affine transformations

use image::{GenericImageView, Pixel, RgbaImage};
use scirs2_vision::transform::{
    estimate_affine_transform, warp_affine, AffineTransform,
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

    // Create output directory if it doesn't exist
    std::fs::create_dir_all("output")?;

    // 1. Translation
    println!("\nApplying translation...");
    let translation = AffineTransform::translation(50.0, 30.0);
    let translated = warp_affine(&img, &translation, None, None, scirs2_vision::transform::affine::BorderMode::Transparent)?;
    translated.save("output/translated.png")?;
    println!("  Saved to output/translated.png");

    // 2. Scaling
    println!("\nApplying scaling...");
    let scaling = AffineTransform::scaling(1.5, 0.7);
    let scaled = warp_affine(&img, &scaling, None, None, scirs2_vision::transform::affine::BorderMode::Transparent)?;
    scaled.save("output/scaled.png")?;
    println!("  Saved to output/scaled.png");

    // 3. Rotation
    println!("\nApplying rotation...");
    let angle = std::f64::consts::PI / 6.0; // 30 degrees
    let rotation = AffineTransform::rotation(angle);

    // For rotation, we need to adjust the output size
    let (width, height) = img.dimensions();
    let rotated = warp_affine(
        &img,
        &rotation,
        Some((1.5 * width as f64) as u32),
        Some((1.5 * height as f64) as u32),
        scirs2_vision::transform::affine::BorderMode::Transparent,
    )?;
    rotated.save("output/rotated.png")?;
    println!("  Saved to output/rotated.png");

    // 4. Combined transformation (translate, rotate, scale)
    println!("\nApplying combined transformation...");

    // Create transformations (applied in reverse order)
    let scale = AffineTransform::scaling(1.2, 0.8);
    let rotate = AffineTransform::rotation(std::f64::consts::PI / 4.0); // 45 degrees
    let translate = AffineTransform::translation(100.0, 50.0);

    // Compose transformations: first scale, then rotate, then translate
    let combined = translate.compose(&rotate.compose(&scale));

    let transformed = warp_affine(
        &img,
        &combined,
        Some((2.0 * width as f64) as u32),
        Some((2.0 * height as f64) as u32),
        scirs2_vision::transform::affine::BorderMode::Transparent,
    )?;
    transformed.save("output/combined.png")?;
    println!("  Saved to output/combined.png");

    // 5. Demonstrate different border modes
    println!("\nDemonstrating border modes...");

    // Use small rotation to show border effects
    let small_rotation = AffineTransform::rotation(std::f64::consts::PI / 12.0); // 15 degrees

    // Transparent border (default)
    let transparent = warp_affine(&img, &small_rotation, None, None, scirs2_vision::transform::affine::BorderMode::Transparent)?;
    transparent.save("output/border_transparent.png")?;

    // Replicate border
    let replicate = warp_affine(&img, &small_rotation, None, None, scirs2_vision::transform::affine::BorderMode::Replicate)?;
    replicate.save("output/border_replicate.png")?;

    // Reflect border
    let reflect = warp_affine(&img, &small_rotation, None, None, scirs2_vision::transform::affine::BorderMode::Reflect)?;
    reflect.save("output/border_reflect.png")?;

    // Wrap border
    let wrap = warp_affine(&img, &small_rotation, None, None, scirs2_vision::transform::affine::BorderMode::Wrap)?;
    wrap.save("output/border_wrap.png")?;

    // Constant border (red)
    let constant = warp_affine(
        &img,
        &small_rotation,
        None,
        None,
        scirs2_vision::transform::affine::BorderMode::Constant(image::Rgba([255, 0, 0, 255])),
    )?;
    constant.save("output/border_constant.png")?;

    println!("  Saved all border mode examples to output/border_*.png");

    // 6. Estimating transformation from point correspondences
    println!("\nEstimating affine transform from points...");

    // Define source rectangle
    let src_points = vec![
        (100.0, 100.0), // top-left
        (400.0, 100.0), // top-right
        (400.0, 300.0), // bottom-right
        (100.0, 300.0), // bottom-left
    ];

    // Define destination quadrilateral (skewed)
    let dst_points = vec![
        (120.0, 80.0),  // top-left
        (450.0, 120.0), // top-right
        (420.0, 350.0), // bottom-right
        (80.0, 330.0),  // bottom-left
    ];

    // Estimate transformation
    let estimated_transform = estimate_affine_transform(&src_points, &dst_points)?;

    // Create a visualization
    let (width, height) = img.dimensions();
    let mut visualization = RgbaImage::new(width, height);

    // Draw the original image (partially transparent)
    for y in 0..height {
        for x in 0..width {
            let mut pixel = img.get_pixel(x, y).to_rgba();
            pixel[3] = 128; // 50% opacity
            visualization.put_pixel(x, y, pixel);
        }
    }

    // Draw source rectangle
    draw_quadrilateral(&mut visualization, &src_points, [0, 0, 255, 255]);

    // Draw destination quadrilateral
    draw_quadrilateral(&mut visualization, &dst_points, [255, 0, 0, 255]);

    // Transform the entire source rectangle using the estimated transform
    for x in 100..=400 {
        for y in 100..=300 {
            let (tx, ty) = estimated_transform.transform_point(x as f64, y as f64);

            // Draw a yellow pixel at the transformed position
            if tx >= 0.0 && tx < width as f64 && ty >= 0.0 && ty < height as f64 {
                visualization.put_pixel(tx as u32, ty as u32, image::Rgba([255, 255, 0, 255]));
            }
        }
    }

    visualization.save("output/estimated_transform.png")?;
    println!("  Saved visualization to output/estimated_transform.png");

    // Warp the image using the estimated transform
    let warped = warp_affine(
        &img,
        &estimated_transform,
        None,
        None,
        scirs2_vision::transform::affine::BorderMode::Transparent,
    )?;
    warped.save("output/warped_estimated.png")?;
    println!("  Saved warped image to output/warped_estimated.png");

    println!("\nAffine transformation example completed successfully!");
    Ok(())
}

/// Draw a quadrilateral on an image
fn draw_quadrilateral(img: &mut RgbaImage, points: &[(f64, f64)], color: [u8; 4]) {
    if points.len() != 4 {
        return;
    }

    let color = image::Rgba(color);

    // Draw lines connecting the points
    draw_line(img, points[0], points[1], color);
    draw_line(img, points[1], points[2], color);
    draw_line(img, points[2], points[3], color);
    draw_line(img, points[3], points[0], color);

    // Draw points
    for &(x, y) in points {
        draw_circle(img, x as u32, y as u32, 5, color);
    }
}

/// Draw a line on an image
fn draw_line(img: &mut RgbaImage, p1: (f64, f64), p2: (f64, f64), color: image::Rgba<u8>) {
    let (width, height) = img.dimensions();
    let (x0, y0) = (p1.0 as i32, p1.1 as i32);
    let (x1, y1) = (p2.0 as i32, p2.1 as i32);

    // Bresenham's line algorithm
    let dx = (x1 - x0).abs();
    let dy = (y1 - y0).abs();

    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };

    let mut err = dx - dy;
    let mut x = x0;
    let mut y = y0;

    while x >= 0 && x < width as i32 && y >= 0 && y < height as i32 {
        img.put_pixel(x as u32, y as u32, color);

        if x == x1 && y == y1 {
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

/// Draw a circle on an image
fn draw_circle(img: &mut RgbaImage, cx: u32, cy: u32, radius: u32, color: image::Rgba<u8>) {
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
