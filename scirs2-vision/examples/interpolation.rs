//! Example demonstrating various interpolation methods

use image::{GenericImageView, Pixel};
use scirs2_vision::transform::{
    resize, resize_bicubic, resize_edge_preserving, resize_lanczos, InterpolationMethod,
};
use std::env;
use std::path::Path;
use std::time::Instant;

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

    // Upscaling example (2x)
    println!("\nPerforming upscaling (2x)...");
    let (width, height) = img.dimensions();
    let new_width = width * 2;
    let new_height = height * 2;

    // Nearest neighbor interpolation
    let start = Instant::now();
    let upscaled_nearest = resize(&img, new_width, new_height, InterpolationMethod::Nearest)?;
    let duration = start.elapsed();
    upscaled_nearest.save("output/upscaled_nearest.png")?;
    println!("  Nearest neighbor: {}ms", duration.as_millis());

    // Bilinear interpolation
    let start = Instant::now();
    let upscaled_bilinear = resize(&img, new_width, new_height, InterpolationMethod::Bilinear)?;
    let duration = start.elapsed();
    upscaled_bilinear.save("output/upscaled_bilinear.png")?;
    println!("  Bilinear: {}ms", duration.as_millis());

    // Bicubic interpolation
    let start = Instant::now();
    let upscaled_bicubic = resize(&img, new_width, new_height, InterpolationMethod::Bicubic)?;
    let duration = start.elapsed();
    upscaled_bicubic.save("output/upscaled_bicubic.png")?;
    println!("  Bicubic: {}ms", duration.as_millis());

    // Lanczos interpolation
    let start = Instant::now();
    let upscaled_lanczos = resize(&img, new_width, new_height, InterpolationMethod::Lanczos3)?;
    let duration = start.elapsed();
    upscaled_lanczos.save("output/upscaled_lanczos.png")?;
    println!("  Lanczos: {}ms", duration.as_millis());

    // Optimized implementations
    let start = Instant::now();
    let upscaled_bicubic_opt = resize_bicubic(&img, new_width, new_height)?;
    let duration = start.elapsed();
    upscaled_bicubic_opt.save("output/upscaled_bicubic_opt.png")?;
    println!("  Bicubic (optimized): {}ms", duration.as_millis());

    let start = Instant::now();
    let upscaled_lanczos_opt = resize_lanczos(&img, new_width, new_height)?;
    let duration = start.elapsed();
    upscaled_lanczos_opt.save("output/upscaled_lanczos_opt.png")?;
    println!("  Lanczos (optimized): {}ms", duration.as_millis());

    // Edge-preserving interpolation
    let start = Instant::now();
    let upscaled_edge_preserving = resize(
        &img,
        new_width,
        new_height,
        InterpolationMethod::EdgePreserving,
    )?;
    let duration = start.elapsed();
    upscaled_edge_preserving.save("output/upscaled_edge_preserving.png")?;
    println!("  Edge-preserving: {}ms", duration.as_millis());

    // Direct edge-preserving interpolation
    let start = Instant::now();
    let upscaled_edge_preserving_direct = resize_edge_preserving(&img, new_width, new_height)?;
    let duration = start.elapsed();
    upscaled_edge_preserving_direct.save("output/upscaled_edge_preserving_direct.png")?;
    println!("  Edge-preserving (direct): {}ms", duration.as_millis());

    // Downscaling example (0.5x)
    println!("\nPerforming downscaling (0.5x)...");
    let new_width = width / 2;
    let new_height = height / 2;

    // Nearest neighbor interpolation
    let start = Instant::now();
    let downscaled_nearest = resize(&img, new_width, new_height, InterpolationMethod::Nearest)?;
    let duration = start.elapsed();
    downscaled_nearest.save("output/downscaled_nearest.png")?;
    println!("  Nearest neighbor: {}ms", duration.as_millis());

    // Bilinear interpolation
    let start = Instant::now();
    let downscaled_bilinear = resize(&img, new_width, new_height, InterpolationMethod::Bilinear)?;
    let duration = start.elapsed();
    downscaled_bilinear.save("output/downscaled_bilinear.png")?;
    println!("  Bilinear: {}ms", duration.as_millis());

    // Bicubic interpolation
    let start = Instant::now();
    let downscaled_bicubic = resize(&img, new_width, new_height, InterpolationMethod::Bicubic)?;
    let duration = start.elapsed();
    downscaled_bicubic.save("output/downscaled_bicubic.png")?;
    println!("  Bicubic: {}ms", duration.as_millis());

    // Lanczos interpolation
    let start = Instant::now();
    let downscaled_lanczos = resize(&img, new_width, new_height, InterpolationMethod::Lanczos3)?;
    let duration = start.elapsed();
    downscaled_lanczos.save("output/downscaled_lanczos.png")?;
    println!("  Lanczos: {}ms", duration.as_millis());

    // Optimized implementations
    let start = Instant::now();
    let downscaled_bicubic_opt = resize_bicubic(&img, new_width, new_height)?;
    let duration = start.elapsed();
    downscaled_bicubic_opt.save("output/downscaled_bicubic_opt.png")?;
    println!("  Bicubic (optimized): {}ms", duration.as_millis());

    let start = Instant::now();
    let downscaled_lanczos_opt = resize_lanczos(&img, new_width, new_height)?;
    let duration = start.elapsed();
    downscaled_lanczos_opt.save("output/downscaled_lanczos_opt.png")?;
    println!("  Lanczos (optimized): {}ms", duration.as_millis());

    // Edge-preserving interpolation
    let start = Instant::now();
    let downscaled_edge_preserving = resize(
        &img,
        new_width,
        new_height,
        InterpolationMethod::EdgePreserving,
    )?;
    let duration = start.elapsed();
    downscaled_edge_preserving.save("output/downscaled_edge_preserving.png")?;
    println!("  Edge-preserving: {}ms", duration.as_millis());

    // Direct edge-preserving interpolation
    let start = Instant::now();
    let downscaled_edge_preserving_direct = resize_edge_preserving(&img, new_width, new_height)?;
    let duration = start.elapsed();
    downscaled_edge_preserving_direct.save("output/downscaled_edge_preserving_direct.png")?;
    println!("  Edge-preserving (direct): {}ms", duration.as_millis());

    // Create comparison grid
    println!("\nCreating comparison grid...");

    // Downscale original image to a smaller size for faster processing
    let small_width = width.min(800);
    let small_height = (height as f64 * (small_width as f64 / width as f64)) as u32;
    let small_img = resize(
        &img,
        small_width,
        small_height,
        InterpolationMethod::Lanczos3,
    )?;

    // Upscale by 2x using each method
    let methods = [
        ("Original", small_img.clone()),
        (
            "Nearest",
            resize(
                &small_img,
                small_width * 2,
                small_height * 2,
                InterpolationMethod::Nearest,
            )?,
        ),
        (
            "Bilinear",
            resize(
                &small_img,
                small_width * 2,
                small_height * 2,
                InterpolationMethod::Bilinear,
            )?,
        ),
        (
            "Bicubic",
            resize(
                &small_img,
                small_width * 2,
                small_height * 2,
                InterpolationMethod::Bicubic,
            )?,
        ),
        (
            "Lanczos",
            resize(
                &small_img,
                small_width * 2,
                small_height * 2,
                InterpolationMethod::Lanczos3,
            )?,
        ),
        (
            "Edge-Preserving",
            resize(
                &small_img,
                small_width * 2,
                small_height * 2,
                InterpolationMethod::EdgePreserving,
            )?,
        ),
    ];

    // Calculate grid dimensions
    let grid_width = small_width * 3; // 3 images per row
    let grid_height = small_height * 2; // 2 rows (now 6 images total)
    let mut grid = image::RgbaImage::new(grid_width, grid_height);

    // Create the grid
    let mut row = 0;
    let mut col = 0;
    for (name, img) in &methods {
        let (img_width, img_height) = img.dimensions();
        let x_offset = col * small_width;
        let y_offset = row * small_height;

        // Copy image to grid
        for y in 0..img_height.min(small_height) {
            for x in 0..img_width.min(small_width) {
                if x_offset + x < grid_width && y_offset + y < grid_height {
                    let pixel = img.get_pixel(x, y);
                    grid.put_pixel(x_offset + x, y_offset + y, pixel.to_rgba());
                }
            }
        }

        // Draw method name
        draw_text(&mut grid, name, x_offset + 10, y_offset + 20);

        // Update position
        col += 1;
        if col >= 3 {
            col = 0;
            row += 1;
        }
    }

    // Save grid
    image::DynamicImage::ImageRgba8(grid).save("output/interpolation_comparison.png")?;
    println!("  Saved comparison grid to output/interpolation_comparison.png");

    // Create edge-preservation example
    println!("\nCreating edge-preservation example...");

    // Create a test image with sharp edges
    let edge_test_width = 512;
    let edge_test_height = 512;
    let mut edge_test = image::RgbaImage::new(edge_test_width, edge_test_height);

    // Draw alternating black and white blocks
    for y in 0..edge_test_height {
        for x in 0..edge_test_width {
            let block_x = x / 64;
            let block_y = y / 64;
            let color = if (block_x + block_y) % 2 == 0 {
                image::Rgba([0, 0, 0, 255])
            } else {
                image::Rgba([255, 255, 255, 255])
            };
            edge_test.put_pixel(x, y, color);
        }
    }

    // Add some noise
    for y in 0..edge_test_height {
        for x in 0..edge_test_width {
            if (x + y) % 8 == 0 {
                let pixel = edge_test.get_pixel_mut(x, y);
                for c in 0..3 {
                    // Add random noise between -20 and 20
                    let current = pixel[c] as i16;
                    let noise = (x as i16 * 7 + y as i16 * 11) % 41 - 20;
                    pixel[c] = (current + noise).clamp(0, 255) as u8;
                }
            }
        }
    }

    // Save the test image
    image::DynamicImage::ImageRgba8(edge_test.clone()).save("output/edge_test_original.png")?;

    // Downscale and then upscale using different methods
    let src_img = image::DynamicImage::ImageRgba8(edge_test);
    let downscaled = resize(
        &src_img,
        edge_test_width / 4,
        edge_test_height / 4,
        InterpolationMethod::Bicubic,
    )?;

    // Upscale with different methods
    let methods = [
        (
            "bicubic",
            resize(
                &downscaled,
                edge_test_width,
                edge_test_height,
                InterpolationMethod::Bicubic,
            )?,
        ),
        (
            "lanczos",
            resize(
                &downscaled,
                edge_test_width,
                edge_test_height,
                InterpolationMethod::Lanczos3,
            )?,
        ),
        (
            "edge_preserving",
            resize_edge_preserving(&downscaled, edge_test_width, edge_test_height)?,
        ),
    ];

    // Save each result
    for (name, img) in &methods {
        img.save(format!("output/edge_test_{}.png", name))?;
    }

    println!("  Saved edge-preservation examples to output/edge_test_*.png");

    // Create comparison grid for edge preservation
    println!("\nCreating edge-preservation comparison grid...");

    // Create grid layout
    let grid_width = edge_test_width;
    let grid_height = edge_test_height;
    let mut grid = image::RgbaImage::new(grid_width, grid_height);

    // Calculate cell dimensions
    let cell_width = edge_test_width / 2;
    let cell_height = edge_test_height / 2;

    // Place the images in a 2x2 grid
    for (idx, (name, img)) in methods.iter().enumerate() {
        let x_offset = ((idx % 2) as u32) * cell_width;
        let y_offset = ((idx / 2) as u32) * cell_height;

        // Resize the image to fit in the cell
        let cell_img = resize(img, cell_width, cell_height, InterpolationMethod::Lanczos3)?;

        // Copy to grid
        for y in 0..cell_height {
            for x in 0..cell_width {
                let pixel = cell_img.get_pixel(x, y);
                grid.put_pixel(x_offset + x, y_offset + y, pixel.to_rgba());
            }
        }

        // Draw method name
        draw_text(&mut grid, name, x_offset + 10, y_offset + 20);
    }

    // Save comparison grid
    image::DynamicImage::ImageRgba8(grid).save("output/edge_preservation_comparison.png")?;
    println!(
        "  Saved edge-preservation comparison grid to output/edge_preservation_comparison.png"
    );

    println!("\nInterpolation example completed successfully!");
    Ok(())
}

/// Draw simple text on an image
fn draw_text(img: &mut image::RgbaImage, text: &str, x: u32, y: u32) {
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
fn draw_char(img: &mut image::RgbaImage, c: char, x: u32, y: u32, color: image::Rgba<u8>) {
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
        'l' => &[
            0b01100000, 0b00100000, 0b00100000, 0b00100000, 0b00100000, 0b00100000, 0b01110000,
        ],
        'm' => &[
            0b00000000, 0b00000000, 0b11010000, 0b10101000, 0b10101000, 0b10101000, 0b10101000,
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
        't' => &[
            0b00100000, 0b00100000, 0b01110000, 0b00100000, 0b00100000, 0b00100000, 0b00110000,
        ],
        'u' => &[
            0b00000000, 0b00000000, 0b10010000, 0b10010000, 0b10010000, 0b10010000, 0b01110000,
        ],
        'z' => &[
            0b00000000, 0b00000000, 0b11110000, 0b00010000, 0b00100000, 0b01000000, 0b11110000,
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
