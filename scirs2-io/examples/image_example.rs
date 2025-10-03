//! Image file format example
//!
//! This example demonstrates basic image I/O operations:
//! - Creating and saving images
//! - Loading images from files
//! - Converting between image formats
//! - Working with image metadata

use scirs2_core::ndarray::Array3;
use scirs2_io::image::{
    convert_image, get_image_info, load_image, resize_image, save_image, ColorMode, ImageData,
    ImageFormat, ImageMetadata,
};
use std::error::Error;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn Error>> {
    println!("=== Image Module Example ===\n");

    // Create a simple RGB test image (100x100 pixels)
    println!("1. Creating a test image...");
    let mut test_array = Array3::zeros((100, 100, 3));

    // Create a red square in the top-left corner
    for y in 0..30 {
        for x in 0..30 {
            test_array[[y, x, 0]] = 255;
        }
    }

    // Create a green square in the top-right corner
    for y in 0..30 {
        for x in 70..100 {
            test_array[[y, x, 1]] = 255;
        }
    }

    // Create a blue square in the bottom-left corner
    for y in 70..100 {
        for x in 0..30 {
            test_array[[y, x, 2]] = 255;
        }
    }

    // Create a yellow square in the bottom-right corner (red + green)
    for y in 70..100 {
        for x in 70..100 {
            test_array[[y, x, 0]] = 255;
            test_array[[y, x, 1]] = 255;
        }
    }

    // Create metadata for the image
    let metadata = ImageMetadata {
        width: 100,
        height: 100,
        color_mode: ColorMode::RGB,
        format: ImageFormat::PNG,
        file_size: 0, // Will be set when saved
        exif: None,
    };

    let image_data = ImageData {
        data: test_array,
        metadata,
    };

    // Save the test image
    println!("2. Saving test image as PNG...");
    save_image(&image_data, "test_image.png", Some(ImageFormat::PNG))?;
    println!("✓ Saved as test_image.png");

    // Convert to different formats
    println!("\n3. Converting to different formats...");
    convert_image("test_image.png", "test_image.jpg", ImageFormat::JPEG)?;
    println!("✓ Converted to JPEG: test_image.jpg");

    convert_image("test_image.png", "test_image.bmp", ImageFormat::BMP)?;
    println!("✓ Converted to BMP: test_image.bmp");

    // Load and verify the saved image
    println!("\n4. Loading saved image...");
    let loaded_image = load_image("test_image.png")?;
    println!(
        "✓ Loaded image dimensions: {}x{}",
        loaded_image.metadata.width, loaded_image.metadata.height
    );
    println!("  Format: {:?}", loaded_image.metadata.format);
    println!("  Color mode: {:?}", loaded_image.metadata.color_mode);

    // Get image info without loading full data
    println!("\n5. Getting image info without loading full data...");
    let png_info = get_image_info("test_image.png")?;
    let jpg_info = get_image_info("test_image.jpg")?;
    let bmp_info = get_image_info("test_image.bmp")?;

    println!("File sizes:");
    println!("  PNG: {} bytes", png_info.file_size);
    println!("  JPG: {} bytes", jpg_info.file_size);
    println!("  BMP: {} bytes", bmp_info.file_size);

    // Resize image
    println!("\n6. Resizing image...");
    let resized_image = resize_image(&loaded_image, 50, 50)?;
    save_image(
        &resized_image,
        "test_image_resized.png",
        Some(ImageFormat::PNG),
    )?;
    println!("✓ Created 50x50 resized image: test_image_resized.png");

    // Create a grayscale gradient
    println!("\n7. Creating grayscale gradient...");
    let mut gray_array = Array3::zeros((100, 100, 3));
    for y in 0..100 {
        for x in 0..100 {
            let gray_value = ((x + y) as f32 / 200.0 * 255.0) as u8;
            gray_array[[y, x, 0]] = gray_value;
            gray_array[[y, x, 1]] = gray_value;
            gray_array[[y, x, 2]] = gray_value;
        }
    }

    let gray_metadata = ImageMetadata {
        width: 100,
        height: 100,
        color_mode: ColorMode::RGB, // Still RGB format but with grayscale values
        format: ImageFormat::PNG,
        file_size: 0,
        exif: None,
    };

    let gray_image = ImageData {
        data: gray_array,
        metadata: gray_metadata,
    };

    save_image(&gray_image, "test_gray.png", Some(ImageFormat::PNG))?;
    println!("✓ Created grayscale gradient: test_gray.png");

    // Clean up test files
    println!("\n8. Cleaning up test files...");
    let test_files = vec![
        "test_image.png",
        "test_image.jpg",
        "test_image.bmp",
        "test_image_resized.png",
        "test_gray.png",
    ];

    for file in test_files {
        if std::path::Path::new(file).exists() {
            std::fs::remove_file(file)?;
            println!("✓ Removed {}", file);
        }
    }

    println!("\n✓ All examples completed successfully!");
    Ok(())
}
