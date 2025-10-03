//! Image metadata example
//!
//! This example demonstrates working with image metadata including EXIF data

use scirs2_core::ndarray::Array3;
use scirs2_io::image::{
    get_image_info, load_image, read_exif_metadata, save_image, ColorMode, ImageData, ImageFormat,
    ImageMetadata,
};
use std::error::Error;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn Error>> {
    println!("=== Image Metadata Example ===\n");

    // Create a simple test image
    println!("1. Creating a test image...");
    let mut test_array = Array3::zeros((100, 100, 3));

    // Create a colorful gradient pattern
    for y in 0..100 {
        for x in 0..100 {
            test_array[[y, x, 0]] = ((x as f32 / 100.0) * 255.0) as u8; // Red increases with x
            test_array[[y, x, 1]] = ((y as f32 / 100.0) * 255.0) as u8; // Green increases with y
            test_array[[y, x, 2]] = (((x + y) as f32 / 200.0) * 255.0) as u8; // Blue diagonal
        }
    }

    // Create ImageData with metadata
    let metadata = ImageMetadata {
        width: 100,
        height: 100,
        color_mode: ColorMode::RGB,
        format: ImageFormat::PNG,
        file_size: 0, // Will be set when saved
        exif: None,   // No EXIF data for generated image
    };

    let image_data = ImageData {
        data: test_array,
        metadata,
    };

    // Save the test image
    println!("2. Saving test image...");
    save_image(&image_data, "test_metadata.png", Some(ImageFormat::PNG))?;
    println!("✓ Created test_metadata.png");

    // Get image info without loading the full data
    println!("\n3. Reading image info without loading full data...");
    let info = get_image_info("test_metadata.png")?;
    println!("Image Information:");
    println!("  - Dimensions: {}x{}", info.width, info.height);
    println!("  - Format: {:?}", info.format);
    println!("  - File size: {} bytes", info.file_size);
    println!("  - Color mode: {:?}", info.color_mode);

    // Load the image with metadata
    println!("\n4. Loading image with metadata...");
    let loaded_image = load_image("test_metadata.png")?;
    println!("✓ Loaded image successfully");
    println!("  - Data shape: {:?}", loaded_image.data.shape());
    println!("  - Metadata format: {:?}", loaded_image.metadata.format);

    // Try to read EXIF metadata (will return None for PNG)
    println!("\n5. Checking for EXIF metadata...");
    match read_exif_metadata("test_metadata.png")? {
        Some(exif) => {
            println!("EXIF metadata found:");
            if let Some(datetime) = exif.datetime {
                println!("  - Date taken: {}", datetime);
            }
            if let Some(gps) = exif.gps {
                if let (Some(lat), Some(lon)) = (gps.latitude, gps.longitude) {
                    println!("  - GPS: {:.6}, {:.6}", lat, lon);
                }
            }
            if let Some(make) = &exif.camera.make {
                println!("  - Camera make: {}", make);
            }
            if let Some(model) = &exif.camera.model {
                println!("  - Camera model: {}", model);
            }
        }
        None => {
            println!("No EXIF metadata found (expected for PNG files)");
        }
    }

    // Convert to JPEG format
    println!("\n6. Converting to JPEG format...");
    save_image(&loaded_image, "test_metadata.jpg", Some(ImageFormat::JPEG))?;
    println!("✓ Created test_metadata.jpg");

    // Check the new file's metadata
    let jpeg_info = get_image_info("test_metadata.jpg")?;
    println!("JPEG Information:");
    println!("  - Format: {:?}", jpeg_info.format);
    println!("  - File size: {} bytes", jpeg_info.file_size);

    // Demonstrate batch processing with metadata
    println!("\n7. Example of processing multiple images...");
    let formats = vec![
        (ImageFormat::PNG, "output_test.png"),
        (ImageFormat::JPEG, "output_test.jpg"),
        (ImageFormat::BMP, "output_test.bmp"),
    ];

    for (format, filename) in formats {
        save_image(&image_data, filename, Some(format))?;
        let info = get_image_info(filename)?;
        println!(
            "  - {}: {} bytes ({:?})",
            filename, info.file_size, info.format
        );
    }

    println!("\n✓ All examples completed successfully!");

    // Clean up test files
    println!("\nCleaning up test files...");
    let test_files = vec![
        "test_metadata.png",
        "test_metadata.jpg",
        "output_test.png",
        "output_test.jpg",
        "output_test.bmp",
    ];

    for file in test_files {
        if std::path::Path::new(file).exists() {
            std::fs::remove_file(file)?;
        }
    }
    println!("✓ Test files cleaned up");

    Ok(())
}
