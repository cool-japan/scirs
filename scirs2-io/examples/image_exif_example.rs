//! Image EXIF metadata example
//!
//! This example demonstrates the enhanced image capabilities with EXIF metadata:
//! - Reading EXIF data from JPEG and TIFF files
//! - Extracting camera settings and technical metadata
//! - Parsing GPS coordinates from images
//! - Working with datetime information
//! - Analyzing image metadata comprehensively

use ndarray::{Array2, Array3};
use scirs2_io::error::Result;
use scirs2_io::image::{
    read_enhanced_metadata, read_exif_metadata, read_image, write_image, ColorMode, ImageFormat,
};
use std::fs;

fn main() -> Result<()> {
    println!("=== Image EXIF Metadata Example ===");

    // Example 1: Create a sample image for testing
    create_sample_image()?;

    // Example 2: Demonstrate EXIF metadata extraction
    demonstrate_exif_extraction()?;

    // Example 3: Show enhanced metadata reading
    demonstrate_enhanced_metadata()?;

    // Example 4: Analyze different image formats
    analyze_image_formats()?;

    println!("Image EXIF example completed successfully!");
    println!("Note: EXIF data extraction requires images with embedded metadata.");
    Ok(())
}

fn create_sample_image() -> Result<()> {
    println!("\n1. Creating sample images for testing...");

    // Create a simple test image (RGB pattern)
    let height = 200;
    let width = 300;
    let mut image_data = Array3::<u8>::zeros((height, width, 3));

    // Create a gradient pattern
    for y in 0..height {
        for x in 0..width {
            let r = ((x as f64 / width as f64) * 255.0) as u8;
            let g = ((y as f64 / height as f64) * 255.0) as u8;
            let b = (((x + y) as f64 / (width + height) as f64) * 255.0) as u8;

            image_data[[y, x, 0]] = r;
            image_data[[y, x, 1]] = g;
            image_data[[y, x, 2]] = b;
        }
    }

    // Save as different formats
    write_image("test_image.png", &image_data, Some(ImageFormat::PNG), None)?;
    write_image("test_image.jpg", &image_data, Some(ImageFormat::JPEG), None)?;
    write_image("test_image.bmp", &image_data, Some(ImageFormat::BMP), None)?;

    println!("  Created test images:");
    println!("    - test_image.png");
    println!("    - test_image.jpg");
    println!("    - test_image.bmp");

    Ok(())
}

fn demonstrate_exif_extraction() -> Result<()> {
    println!("\n2. Demonstrating EXIF metadata extraction...");

    // Try to read EXIF from our test JPEG
    println!("  Reading EXIF from test_image.jpg...");
    match read_exif_metadata("test_image.jpg")? {
        Some(exif) => {
            println!("    EXIF data found:");

            // Display datetime information
            if let Some(datetime) = exif.datetime {
                println!(
                    "      Date/Time: {}",
                    datetime.format("%Y-%m-%d %H:%M:%S UTC")
                );
            } else {
                println!("      Date/Time: Not available");
            }

            // Display camera information
            println!("      Camera Information:");
            if let Some(make) = &exif.camera.make {
                println!("        Make: {}", make);
            }
            if let Some(model) = &exif.camera.model {
                println!("        Model: {}", model);
            }
            if let Some(lens) = &exif.camera.lens_model {
                println!("        Lens: {}", lens);
            }

            // Display technical settings
            println!("      Technical Settings:");
            if let Some(iso) = exif.camera.iso {
                println!("        ISO: {}", iso);
            }
            if let Some(aperture) = exif.camera.aperture {
                println!("        Aperture: f/{:.1}", aperture);
            }
            if let Some(shutter) = exif.camera.shutter_speed {
                println!("        Shutter Speed: 1/{:.0} sec", 1.0 / shutter);
            }
            if let Some(focal) = exif.camera.focal_length {
                println!("        Focal Length: {:.0}mm", focal);
            }

            // Display GPS information
            if let Some(gps) = &exif.gps {
                println!("      GPS Information:");
                println!(
                    "        Latitude: {:.6}° {}",
                    gps.latitude, gps.latitude_ref
                );
                println!(
                    "        Longitude: {:.6}° {}",
                    gps.longitude, gps.longitude_ref
                );
                if let Some(altitude) = gps.altitude {
                    println!("        Altitude: {:.1}m", altitude);
                }
            } else {
                println!("      GPS: Not available");
            }

            // Display other metadata
            if let Some(software) = &exif.software {
                println!("      Software: {}", software);
            }
            if let Some(artist) = &exif.artist {
                println!("      Artist: {}", artist);
            }
            if let Some(copyright) = &exif.copyright {
                println!("      Copyright: {}", copyright);
            }

            // Show raw EXIF tags
            if !exif.raw_tags.is_empty() {
                println!("      Raw EXIF Tags ({} total):", exif.raw_tags.len());
                let mut tags: Vec<_> = exif.raw_tags.iter().collect();
                tags.sort_by_key(|(k, _)| *k);
                for (tag, value) in tags.iter().take(10) {
                    println!("        {}: {}", tag, value);
                }
                if exif.raw_tags.len() > 10 {
                    println!("        ... and {} more tags", exif.raw_tags.len() - 10);
                }
            }
        }
        None => {
            println!("    No EXIF data found (expected for generated test image)");
        }
    }

    Ok(())
}

fn demonstrate_enhanced_metadata() -> Result<()> {
    println!("\n3. Demonstrating enhanced metadata reading...");

    let test_files = ["test_image.png", "test_image.jpg", "test_image.bmp"];

    for filename in &test_files {
        println!("  Analyzing {}...", filename);

        match read_enhanced_metadata(filename) {
            Ok(metadata) => {
                println!("    Basic Properties:");
                println!("      Dimensions: {}x{}", metadata.width, metadata.height);

                if let Some(color_mode) = metadata.color_mode {
                    println!(
                        "      Color Mode: {} ({} channels)",
                        color_mode.as_str(),
                        color_mode.channels()
                    );
                }

                if let Some(bits) = metadata.bits_per_channel {
                    println!("      Bits per Channel: {}", bits);
                }

                if let Some(format) = &metadata.format {
                    println!("      Format: {}", format);
                }

                if let Some(dpi) = metadata.dpi {
                    println!("      DPI: {}x{}", dpi.0, dpi.1);
                }

                // Check for EXIF data
                if let Some(exif) = &metadata.exif {
                    println!("    EXIF Metadata: Available");
                    if let Some(datetime) = exif.datetime {
                        println!(
                            "      Created: {}",
                            datetime.format("%Y-%m-%d %H:%M:%S UTC")
                        );
                    }
                    if exif.camera.make.is_some() || exif.camera.model.is_some() {
                        println!(
                            "      Camera: {:?} {:?}",
                            exif.camera.make, exif.camera.model
                        );
                    }
                } else {
                    println!("    EXIF Metadata: Not available");
                }

                // Show custom metadata
                if !metadata.custom.is_empty() {
                    println!("    Custom Metadata:");
                    for (key, value) in &metadata.custom {
                        println!("      {}: {}", key, value);
                    }
                }
            }
            Err(e) => {
                println!("    Error reading metadata: {}", e);
            }
        }
        println!();
    }

    Ok(())
}

fn analyze_image_formats() -> Result<()> {
    println!("4. Analyzing different image format capabilities...");

    let formats = [
        (ImageFormat::PNG, "PNG", "Lossless, supports transparency"),
        (
            ImageFormat::JPEG,
            "JPEG",
            "Lossy compression, supports EXIF",
        ),
        (ImageFormat::BMP, "BMP", "Uncompressed, simple format"),
        (
            ImageFormat::TIFF,
            "TIFF",
            "Flexible, supports EXIF and multiple pages",
        ),
    ];

    for (format, name, description) in &formats {
        println!("  {}:", name);
        println!("    Description: {}", description);
        println!("    Extension: .{}", format.extension());

        // Analyze EXIF support
        let exif_support = matches!(format, ImageFormat::JPEG | ImageFormat::TIFF);
        println!(
            "    EXIF Support: {}",
            if exif_support { "Yes" } else { "No" }
        );

        // Test with our sample file
        let filename = format!("test_image.{}", format.extension());
        if std::path::Path::new(&filename).exists() {
            match read_enhanced_metadata(&filename) {
                Ok(metadata) => {
                    println!(
                        "    File Size: {} bytes",
                        fs::metadata(&filename).map(|m| m.len()).unwrap_or(0)
                    );
                    println!("    Dimensions: {}x{}", metadata.width, metadata.height);

                    if let Some(color_mode) = metadata.color_mode {
                        println!("    Color Mode: {}", color_mode.as_str());
                    }
                }
                Err(e) => {
                    println!("    Error: {}", e);
                }
            }
        } else {
            println!("    Test file not found");
        }
        println!();
    }

    Ok(())
}

/// Utility function to display GPS coordinates in human-readable format
#[allow(dead_code)]
fn format_gps_coordinate(degrees: f64, is_latitude: bool) -> String {
    let abs_degrees = degrees.abs();
    let deg = abs_degrees.floor() as u32;
    let min_float = (abs_degrees - deg as f64) * 60.0;
    let min = min_float.floor() as u32;
    let sec = (min_float - min as f64) * 60.0;

    let direction = if is_latitude {
        if degrees >= 0.0 {
            "N"
        } else {
            "S"
        }
    } else {
        if degrees >= 0.0 {
            "E"
        } else {
            "W"
        }
    };

    format!("{}°{:02}'{:06.3}\"{}", deg, min, sec, direction)
}

/// Utility function to calculate image file size efficiency
#[allow(dead_code)]
fn calculate_compression_ratio(
    width: usize,
    height: usize,
    channels: usize,
    file_size: u64,
) -> f64 {
    let uncompressed_size = (width * height * channels) as u64;
    if file_size > 0 {
        uncompressed_size as f64 / file_size as f64
    } else {
        0.0
    }
}

/// Clean up test files
#[allow(dead_code)]
fn cleanup_test_files() {
    let files = ["test_image.png", "test_image.jpg", "test_image.bmp"];
    for file in &files {
        let _ = fs::remove_file(file);
    }
}
