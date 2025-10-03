//! Example demonstrating noise reduction techniques
//!
//! This example demonstrates:
//! 1. Gaussian blur for general smoothing
//! 2. Bilateral filtering for edge-preserving noise reduction
//! 3. Median filtering for salt-and-pepper noise removal
//! 4. Comparison of different noise reduction methods

use image::{DynamicImage, GenericImage, GenericImageView, ImageBuffer, Rgba};
use scirs2_core::random::random;
use scirs2_vision::preprocessing::{bilateral_filter, gaussian_blur, median_filter};
use std::error::Error;
use std::path::PathBuf;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn Error>> {
    println!("SciRS2 Vision - Noise Reduction Example");

    // Load or create an image
    let input_image = get_input_image()?;
    let _width_height = input_image.dimensions();

    // Create a copy with random noise (salt and pepper)
    let mut noisy_image = input_image.to_rgba8();
    add_salt_and_pepper_noise(&mut noisy_image, 0.02);
    let noisy_image = DynamicImage::ImageRgba8(noisy_image);

    // Save the noisy image
    let output_dir = PathBuf::from("examples/output");
    if !output_dir.exists() {
        std::fs::create_dir_all(&output_dir)?;
    }
    noisy_image.save("examples/output/noisy_image.png")?;
    println!("Saved noisy image to examples/output/noisy_image.png");

    // 1. Apply Gaussian blur
    println!("Applying Gaussian blur...");
    let gaussian_result = gaussian_blur(&noisy_image, 1.5)?;
    gaussian_result.save("examples/output/gaussian_blur.png")?;
    println!("Saved Gaussian blur result to examples/output/gaussian_blur.png");

    // 2. Apply bilateral filter
    println!("Applying bilateral filter...");
    let bilateral_result = bilateral_filter(&noisy_image, 9, 75.0, 75.0)?;
    bilateral_result.save("examples/output/bilateral_filter.png")?;
    println!("Saved bilateral filter result to examples/output/bilateral_filter.png");

    // 3. Apply median filter
    println!("Applying median filter...");
    let median_result = median_filter(&noisy_image, 3)?;
    median_result.save("examples/output/median_filter.png")?;
    println!("Saved median filter result to examples/output/median_filter.png");

    // Create a comparison image
    println!("Creating comparison image...");
    let comparison = create_comparison_image(
        &input_image,
        &noisy_image,
        &gaussian_result,
        &bilateral_result,
        &median_result,
    )?;
    comparison.save("examples/output/noise_reduction_comparison.png")?;
    println!("Saved comparison image to examples/output/noise_reduction_comparison.png");

    println!("Noise reduction example completed successfully!");
    Ok(())
}

/// Get input image from file or create a test image
#[allow(dead_code)]
fn get_input_image() -> Result<DynamicImage, Box<dyn Error>> {
    // Try to load from standard locations
    let image_paths = [
        "examples/input/input.jpg",
        "input/input.jpg",
        "examples/data/lenna.png",
    ];

    for path in &image_paths {
        if let Ok(img) = image::open(path) {
            println!("Successfully loaded image from: {path}");
            return Ok(img);
        }
    }

    println!("No input image found. Creating a test image...");

    // Create a simple test image with gradients
    let width = 512;
    let height = 512;
    let mut img_buffer = ImageBuffer::new(width, height);

    for y in 0..height {
        for x in 0..width {
            // Create a gradient pattern
            let r = ((x as f32 / width as f32) * 255.0) as u8;
            let g = ((y as f32 / height as f32) * 255.0) as u8;
            let b = (((x + y) as f32 / (width + height) as f32) * 255.0) as u8;
            img_buffer.put_pixel(x, y, Rgba([r, g, b, 255]));
        }
    }

    // Add some geometric shapes for feature preservation testing
    draw_rectangle(&mut img_buffer, 100, 100, 300, 300, [200, 100, 100, 255]);
    draw_circle(&mut img_buffer, 250, 250, 100, [100, 200, 100, 255]);
    draw_rectangle(&mut img_buffer, 150, 150, 100, 100, [100, 100, 200, 255]);

    Ok(DynamicImage::ImageRgba8(img_buffer))
}

/// Add salt and pepper noise to an image
#[allow(dead_code)]
fn add_salt_and_pepper_noise(img: &mut ImageBuffer<Rgba<u8>, Vec<u8>>, noiseamount: f64) {
    let (width, height) = img.dimensions();
    let noise_pixels = (width * height) as f64 * noiseamount;

    for _ in 0..(noise_pixels as u32) {
        let x = (random::<f32>() * width as f32) as u32;
        let y = (random::<f32>() * height as f32) as u32;

        // Add either white (salt) or black (pepper) pixels
        let color = if random::<bool>() {
            Rgba([255, 255, 255, 255]) // Salt
        } else {
            Rgba([0, 0, 0, 255]) // Pepper
        };

        if x < width && y < height {
            img.put_pixel(x, y, color);
        }
    }
}

/// Draw a rectangle on an image
#[allow(dead_code)]
fn draw_rectangle(
    img: &mut ImageBuffer<Rgba<u8>, Vec<u8>>,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    color: [u8; 4],
) {
    for dy in 0..height {
        for dx in 0..width {
            let px = x + dx;
            let py = y + dy;
            if px < img.width() && py < img.height() {
                img.put_pixel(px, py, Rgba(color));
            }
        }
    }
}

/// Draw a circle on an image
#[allow(dead_code)]
fn draw_circle(
    img: &mut ImageBuffer<Rgba<u8>, Vec<u8>>,
    cx: u32,
    cy: u32,
    radius: u32,
    color: [u8; 4],
) {
    for y in 0..img.height() {
        for x in 0..img.width() {
            let dx = x as i32 - cx as i32;
            let dy = y as i32 - cy as i32;
            let distance_squared = dx * dx + dy * dy;

            if distance_squared <= (radius * radius) as i32 {
                img.put_pixel(x, y, Rgba(color));
            }
        }
    }
}

/// Create a side-by-side comparison of different noise reduction methods
#[allow(dead_code)]
fn create_comparison_image(
    original: &DynamicImage,
    noisy: &DynamicImage,
    gaussian: &DynamicImage,
    bilateral: &DynamicImage,
    median: &DynamicImage,
) -> Result<DynamicImage, Box<dyn Error>> {
    let (width, height) = original.dimensions();
    let new_width = width * 5; // 5 images side by side
    let new_height = height;

    let mut comparison = DynamicImage::new_rgba8(new_width, new_height);

    // Copy original image
    for y in 0..height {
        for x in 0..width {
            let pixel = original.get_pixel(x, y);
            comparison.put_pixel(x, y, pixel);
        }
    }

    // Copy noisy image
    for y in 0..height {
        for x in 0..width {
            let pixel = noisy.get_pixel(x, y);
            comparison.put_pixel(width + x, y, pixel);
        }
    }

    // Copy Gaussian blur result
    for y in 0..height {
        for x in 0..width {
            let pixel = gaussian.get_pixel(x, y);
            comparison.put_pixel(width * 2 + x, y, pixel);
        }
    }

    // Copy bilateral filter result
    for y in 0..height {
        for x in 0..width {
            let pixel = bilateral.get_pixel(x, y);
            comparison.put_pixel(width * 3 + x, y, pixel);
        }
    }

    // Copy median filter result
    for y in 0..height {
        for x in 0..width {
            let pixel = median.get_pixel(x, y);
            comparison.put_pixel(width * 4 + x, y, pixel);
        }
    }

    // Add text labels
    let labels = ["Original", "Noisy", "Gaussian", "Bilateral", "Median"];

    // The actual text rendering would require a font library
    // For simplicity, we'll just print the labels instead
    println!("Comparison image contains the following sections:");
    for (i, label) in labels.iter().enumerate() {
        println!("{}. {}", i + 1, label);
    }

    Ok(comparison)
}
