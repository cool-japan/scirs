//! Example demonstrating perspective transformations

use image::{GenericImageView, Pixel, RgbaImage};
use scirs2_vision::transform::{
    correct_perspective, warp_perspective, BorderMode, PerspectiveTransform,
};
use std::env;
use std::path::Path;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get command line arguments
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <image_path>", args[0]);
        eprintln!("Demonstrates perspective transformations on the provided image.");
        std::process::exit(1);
    }

    // Load image
    let img_path = Path::new(&args[1]);
    let img = image::open(img_path)?;
    println!("Loaded image: {:?}", img.dimensions());

    // Create output directory if it doesn't exist
    std::fs::create_dir_all("output")?;

    // Get image dimensions
    let (width, height) = img.dimensions();

    // Example 1: Apply a simple perspective tilt
    println!("Applying perspective tilt...");
    let transform = create_tilt_transform(width, height, 0.2);

    let tilted = warp_perspective(
        &img,
        &transform,
        None,
        None,
        BorderMode::Constant(image::Rgba([0, 0, 0, 255])),
    )?;

    tilted.save("output/perspective_tilt.png")?;
    println!("  Saved to output/perspective_tilt.png");

    // Example 2: Correct a perspective distortion
    println!("Simulating and correcting perspective distortion...");

    // Define a distorted quadrilateral (simulate document corners)
    let distorted_corners = [
        (width as f64 * 0.2, height as f64 * 0.1),  // Top-left
        (width as f64 * 0.8, height as f64 * 0.05), // Top-right
        (width as f64 * 0.9, height as f64 * 0.9),  // Bottom-right
        (width as f64 * 0.1, height as f64 * 0.85), // Bottom-left
    ];

    // First create the distorted image using from_quad
    let distort_transform =
        PerspectiveTransform::from_quad(0.0, 0.0, width as f64, height as f64, &distorted_corners)?;

    let distorted = warp_perspective(
        &img,
        &distort_transform,
        None,
        None,
        BorderMode::Transparent,
    )?;

    distorted.save("output/perspective_distorted.png")?;
    println!("  Saved distorted image to output/perspective_distorted.png");

    // Convert distorted image to Array2<f64> for perspective correction
    use scirs2_core::ndarray::Array2;
    let (dist_width, dist_height) = distorted.dimensions();
    let mut distorted_array = Array2::zeros((dist_height as usize, dist_width as usize));

    // Convert DynamicImage to grayscale Array2
    for y in 0..dist_height {
        for x in 0..dist_width {
            let pixel = distorted.get_pixel(x, y);
            let gray =
                (0.299 * pixel[0] as f64 + 0.587 * pixel[1] as f64 + 0.114 * pixel[2] as f64)
                    / 255.0;
            distorted_array[[y as usize, x as usize]] = gray;
        }
    }

    // Now correct the perspective
    let correction_transform = correct_perspective(&distorted_array, 0.1, 1000.0)?;
    let corrected = warp_perspective(
        &distorted,
        &correction_transform,
        None,
        None,
        BorderMode::Constant(image::Rgba([255, 255, 255, 255])),
    )?;

    corrected.save("output/perspective_corrected.png")?;
    println!("  Saved corrected image to output/perspective_corrected.png");

    // Example 3: Changing the viewpoint
    println!("Changing viewpoint...");

    // Create a perspective transform that changes the viewpoint
    let birds_eye_transform = create_birds_eye_transform(width, height);

    let birds_eye = warp_perspective(
        &img,
        &birds_eye_transform,
        None,
        None,
        BorderMode::Replicate,
    )?;

    birds_eye.save("output/perspective_birds_eye.png")?;
    println!("  Saved to output/perspective_birds_eye.png");

    // Example 4: Create a comparison grid showing different border modes
    println!("Creating border mode comparison...");
    create_border_mode_comparison(&img)?;
    println!("  Saved to output/perspective_border_modes.png");

    println!("Example completed successfully!");
    Ok(())
}

/// Create a transform that tilts the image (simulating a perspective viewing angle)
#[allow(dead_code)]
fn create_tilt_transform(_width: u32, height: u32, tiltfactor: f64) -> PerspectiveTransform {
    // Create a transform that makes the bottom of the image wider than the top
    let src_points = [
        (0.0, 0.0),                     // Top-left
        (_width as f64, 0.0),           // Top-right
        (_width as f64, height as f64), // Bottom-right
        (0.0, height as f64),           // Bottom-left
    ];

    let dst_points = [
        (_width as f64 * tiltfactor, 0.0),                   // Top-left
        (_width as f64 * (1.0 - tiltfactor), 0.0),           // Top-right
        (_width as f64 * (1.0 + tiltfactor), height as f64), // Bottom-right
        (_width as f64 * -tiltfactor, height as f64),        // Bottom-left
    ];

    PerspectiveTransform::from_points(&src_points, &dst_points)
        .expect("Failed to create tilt transform")
}

/// Create a transform that simulates a bird's-eye view
#[allow(dead_code)]
fn create_birds_eye_transform(width: u32, height: u32) -> PerspectiveTransform {
    // Create a transform that makes the image appear as if viewed from above
    let src_points = [
        (0.0, 0.0),                    // Top-left
        (width as f64, 0.0),           // Top-right
        (width as f64, height as f64), // Bottom-right
        (0.0, height as f64),          // Bottom-left
    ];

    let dst_points = [
        (width as f64 * 0.25, height as f64 * 0.25), // Top-left
        (width as f64 * 0.75, height as f64 * 0.25), // Top-right
        (width as f64 * 0.75, height as f64 * 0.75), // Bottom-right
        (width as f64 * 0.25, height as f64 * 0.75), // Bottom-left
    ];

    PerspectiveTransform::from_points(&src_points, &dst_points)
        .expect("Failed to create bird's-eye transform")
}

/// Create a comparison grid showing different border modes
#[allow(dead_code)]
fn create_border_mode_comparison(
    img: &image::DynamicImage,
) -> Result<(), Box<dyn std::error::Error>> {
    let (width, height) = img.dimensions();

    // Create a strong perspective transform to show border handling
    let transform = create_tilt_transform(width, height, 0.3);

    // Create warped images with different border modes
    let constant_black = warp_perspective(
        img,
        &transform,
        None,
        None,
        BorderMode::Constant(image::Rgba([0, 0, 0, 255])),
    )?;

    let constant_white = warp_perspective(
        img,
        &transform,
        None,
        None,
        BorderMode::Constant(image::Rgba([255, 255, 255, 255])),
    )?;

    let reflect = warp_perspective(img, &transform, None, None, BorderMode::Reflect)?;

    let replicate = warp_perspective(img, &transform, None, None, BorderMode::Replicate)?;

    let wrap = warp_perspective(img, &transform, None, None, BorderMode::Wrap)?;

    let transparent = warp_perspective(img, &transform, None, None, BorderMode::Transparent)?;

    // Create a grid of images
    let grid_width = width * 3;
    let grid_height = height * 2;
    let mut grid = RgbaImage::new(grid_width, grid_height);

    // Define the images and their positions in the grid
    let images = [
        (constant_black, "Constant (Black)", 0, 0),
        (constant_white, "Constant (White)", 1, 0),
        (reflect, "Reflect", 2, 0),
        (replicate, "Replicate", 0, 1),
        (wrap, "Wrap", 1, 1),
        (transparent, "Transparent", 2, 1),
    ];

    // Copy images to grid
    for (img, label, col, row) in &images {
        let x_offset = col * width;
        let y_offset = row * height;

        // Copy image
        for y in 0..height {
            for x in 0..width {
                let pixel = img.get_pixel(x, y).to_rgba();
                grid.put_pixel(x_offset + x, y_offset + y, pixel);
            }
        }

        // Draw label
        drawtext(&mut grid, label, x_offset + 10, y_offset + 20);
    }

    // Save grid
    image::DynamicImage::ImageRgba8(grid).save("output/perspective_border_modes.png")?;

    Ok(())
}

/// Draw simple text on an image
#[allow(dead_code)]
fn drawtext(img: &mut RgbaImage, text: &str, x: u32, y: u32) {
    let color = image::Rgba([255, 255, 255, 255]);
    let shadow_color = image::Rgba([0, 0, 0, 192]);

    // Draw shadow first for better visibility
    for (i, c) in text.chars().enumerate() {
        let cx = x + i as u32 * 8 + 1;
        let cy = y + 1;
        draw_char(img, c, cx, cy, shadow_color);
    }

    // Draw text
    for (i, c) in text.chars().enumerate() {
        let cx = x + i as u32 * 8;
        let cy = y;
        draw_char(img, c, cx, cy, color);
    }
}

/// Draw a single character (simple bitmap font)
#[allow(dead_code)]
fn draw_char(img: &mut RgbaImage, c: char, x: u32, y: u32, color: image::Rgba<u8>) {
    let (width, height) = img.dimensions();

    // Simple bitmap patterns for letters and numbers
    let pattern = match c {
        'A' => &[
            0b01100000, 0b10010000, 0b10010000, 0b11110000, 0b10010000, 0b10010000, 0b10010000,
        ],
        'B' => &[
            0b11100000, 0b10010000, 0b10010000, 0b11100000, 0b10010000, 0b10010000, 0b11100000,
        ],
        'C' => &[
            0b01110000, 0b10000000, 0b10000000, 0b10000000, 0b10000000, 0b10000000, 0b01110000,
        ],
        'D' => &[
            0b11100000, 0b10010000, 0b10010000, 0b10010000, 0b10010000, 0b10010000, 0b11100000,
        ],
        'E' => &[
            0b11110000, 0b10000000, 0b10000000, 0b11100000, 0b10000000, 0b10000000, 0b11110000,
        ],
        'F' => &[
            0b11110000, 0b10000000, 0b10000000, 0b11100000, 0b10000000, 0b10000000, 0b10000000,
        ],
        'G' => &[
            0b01110000, 0b10000000, 0b10000000, 0b10110000, 0b10010000, 0b10010000, 0b01110000,
        ],
        'H' => &[
            0b10010000, 0b10010000, 0b10010000, 0b11110000, 0b10010000, 0b10010000, 0b10010000,
        ],
        'I' => &[
            0b11100000, 0b01000000, 0b01000000, 0b01000000, 0b01000000, 0b01000000, 0b11100000,
        ],
        'L' => &[
            0b10000000, 0b10000000, 0b10000000, 0b10000000, 0b10000000, 0b10000000, 0b11110000,
        ],
        'N' => &[
            0b10001000, 0b11001000, 0b10101000, 0b10011000, 0b10001000, 0b10001000, 0b10001000,
        ],
        'O' => &[
            0b01100000, 0b10010000, 0b10010000, 0b10010000, 0b10010000, 0b10010000, 0b01100000,
        ],
        'P' => &[
            0b11100000, 0b10010000, 0b10010000, 0b11100000, 0b10000000, 0b10000000, 0b10000000,
        ],
        'R' => &[
            0b11100000, 0b10010000, 0b10010000, 0b11100000, 0b10100000, 0b10010000, 0b10010000,
        ],
        'S' => &[
            0b01110000, 0b10000000, 0b10000000, 0b01100000, 0b00010000, 0b00010000, 0b11100000,
        ],
        'T' => &[
            0b11111000, 0b00100000, 0b00100000, 0b00100000, 0b00100000, 0b00100000, 0b00100000,
        ],
        'U' => &[
            0b10010000, 0b10010000, 0b10010000, 0b10010000, 0b10010000, 0b10010000, 0b01100000,
        ],
        'W' => &[
            0b10001000, 0b10001000, 0b10001000, 0b10001000, 0b10101000, 0b11011000, 0b10001000,
        ],
        'a' => &[
            0b00000000, 0b00000000, 0b01100000, 0b00010000, 0b01110000, 0b10010000, 0b01110000,
        ],
        'b' => &[
            0b10000000, 0b10000000, 0b11100000, 0b10010000, 0b10010000, 0b10010000, 0b11100000,
        ],
        'c' => &[
            0b00000000, 0b00000000, 0b01110000, 0b10000000, 0b10000000, 0b10000000, 0b01110000,
        ],
        'd' => &[
            0b00010000, 0b00010000, 0b01110000, 0b10010000, 0b10010000, 0b10010000, 0b01110000,
        ],
        'e' => &[
            0b00000000, 0b00000000, 0b01100000, 0b10010000, 0b11110000, 0b10000000, 0b01110000,
        ],
        'g' => &[
            0b00000000, 0b00000000, 0b01110000, 0b10010000, 0b10010000, 0b01110000, 0b00010000,
        ],
        'i' => &[
            0b01000000, 0b00000000, 0b11000000, 0b01000000, 0b01000000, 0b01000000, 0b11100000,
        ],
        'k' => &[
            0b10000000, 0b10000000, 0b10010000, 0b10100000, 0b11000000, 0b10100000, 0b10010000,
        ],
        'l' => &[
            0b01100000, 0b00100000, 0b00100000, 0b00100000, 0b00100000, 0b00100000, 0b01110000,
        ],
        'n' => &[
            0b00000000, 0b00000000, 0b10100000, 0b11010000, 0b10010000, 0b10010000, 0b10010000,
        ],
        'o' => &[
            0b00000000, 0b00000000, 0b01100000, 0b10010000, 0b10010000, 0b10010000, 0b01100000,
        ],
        'p' => &[
            0b00000000, 0b00000000, 0b11100000, 0b10010000, 0b10010000, 0b11100000, 0b10000000,
        ],
        'r' => &[
            0b00000000, 0b00000000, 0b10110000, 0b11000000, 0b10000000, 0b10000000, 0b10000000,
        ],
        's' => &[
            0b00000000, 0b00000000, 0b01110000, 0b10000000, 0b01100000, 0b00010000, 0b11100000,
        ],
        't' => &[
            0b00100000, 0b00100000, 0b01110000, 0b00100000, 0b00100000, 0b00100000, 0b00110000,
        ],
        'u' => &[
            0b00000000, 0b00000000, 0b10010000, 0b10010000, 0b10010000, 0b10010000, 0b01110000,
        ],
        'w' => &[
            0b00000000, 0b00000000, 0b10001000, 0b10001000, 0b10101000, 0b11011000, 0b10001000,
        ],
        'y' => &[
            0b00000000, 0b00000000, 0b10010000, 0b10010000, 0b01110000, 0b00010000, 0b01100000,
        ],
        ' ' => &[
            0b00000000, 0b00000000, 0b00000000, 0b00000000, 0b00000000, 0b00000000, 0b00000000,
        ],
        '(' => &[
            0b00100000, 0b01000000, 0b10000000, 0b10000000, 0b10000000, 0b01000000, 0b00100000,
        ],
        ')' => &[
            0b10000000, 0b01000000, 0b00100000, 0b00100000, 0b00100000, 0b01000000, 0b10000000,
        ],
        '0' => &[
            0b01100000, 0b10010000, 0b10010000, 0b10010000, 0b10010000, 0b10010000, 0b01100000,
        ],
        '1' => &[
            0b00100000, 0b01100000, 0b00100000, 0b00100000, 0b00100000, 0b00100000, 0b01110000,
        ],
        '2' => &[
            0b01100000, 0b10010000, 0b00010000, 0b00100000, 0b01000000, 0b10000000, 0b11110000,
        ],
        '3' => &[
            0b01100000, 0b10010000, 0b00010000, 0b00100000, 0b00010000, 0b10010000, 0b01100000,
        ],
        _ => &[
            0b00000000, 0b00000000, 0b00000000, 0b00000000, 0b00000000, 0b00000000, 0b00000000,
        ],
    };

    // Draw the character
    for (dy, &row) in pattern.iter().enumerate() {
        for dx in 0..8 {
            let bit = (row >> (7 - dx)) & 1;
            if bit == 1 {
                let px = x + dx;
                let py = y + dy as u32;
                if px < width && py < height {
                    img.put_pixel(px, py, color);
                }
            }
        }
    }
}
