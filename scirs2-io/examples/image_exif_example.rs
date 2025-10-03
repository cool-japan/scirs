//! Image EXIF metadata example
//!
//! This example demonstrates working with EXIF metadata:
//! - Reading EXIF data from JPEG files (when available)
//! - Understanding EXIF metadata structure
//! - Working with GPS coordinates and camera settings

use scirs2_core::ndarray::Array3;
use scirs2_io::error::Result;
use scirs2_io::image::{
    load_image, read_exif_metadata, save_image, ColorMode, ImageData, ImageFormat, ImageMetadata,
};
use std::fs;

#[allow(dead_code)]
fn main() -> Result<()> {
    println!("=== Image EXIF Metadata Example ===\n");

    // Example 1: Create sample images for testing
    create_sample_images()?;

    // Example 2: Demonstrate EXIF metadata reading
    demonstrate_exif_reading()?;

    // Example 3: Show metadata structure
    show_metadata_structure()?;

    // Clean up
    cleanup_files()?;

    Ok(())
}

#[allow(dead_code)]
fn create_sample_images() -> Result<()> {
    println!("1. Creating Sample Images");
    println!("-------------------------");

    // Create a test image
    let mut test_array = Array3::zeros((200, 300, 3));

    // Create a gradient pattern
    for y in 0..200 {
        for x in 0..300 {
            test_array[[y, x, 0]] = ((x as f32 / 300.0) * 255.0) as u8;
            test_array[[y, x, 1]] = ((y as f32 / 200.0) * 255.0) as u8;
            test_array[[y, x, 2]] = (((x + y) as f32 / 500.0) * 255.0) as u8;
        }
    }

    let metadata = ImageMetadata {
        width: 300,
        height: 200,
        color_mode: ColorMode::RGB,
        format: ImageFormat::JPEG,
        file_size: 0,
        exif: None,
    };

    let image_data = ImageData {
        data: test_array,
        metadata,
    };

    // Save as JPEG (which can contain EXIF)
    save_image(&image_data, "test_exif.jpg", Some(ImageFormat::JPEG))?;
    println!("✓ Created test_exif.jpg");

    // Save as PNG (typically no EXIF)
    save_image(&image_data, "test_no_exif.png", Some(ImageFormat::PNG))?;
    println!("✓ Created test_no_exif.png");

    println!();
    Ok(())
}

#[allow(dead_code)]
fn demonstrate_exif_reading() -> Result<()> {
    println!("2. Reading EXIF Metadata");
    println!("------------------------");

    // Try to read EXIF from JPEG
    println!("Checking test_exif.jpg for EXIF data...");
    match read_exif_metadata("test_exif.jpg")? {
        Some(exif) => {
            println!("EXIF metadata found:");

            // DateTime
            if let Some(datetime) = &exif.datetime {
                println!("  - DateTime: {}", datetime);
            }

            // Camera information
            if let Some(make) = &exif.camera.make {
                println!("  - Camera Make: {}", make);
            }
            if let Some(model) = &exif.camera.model {
                println!("  - Camera Model: {}", model);
            }

            // Camera settings
            if let Some(iso) = exif.camera.iso {
                println!("  - ISO: {}", iso);
            }
            if let Some(aperture) = exif.camera.aperture {
                println!("  - Aperture: f/{}", aperture);
            }
            if let Some(shutter) = exif.camera.shutter_speed {
                println!("  - Shutter Speed: {}", shutter);
            }

            // GPS data
            if let Some(gps) = &exif.gps {
                if let (Some(lat), Some(lon)) = (gps.latitude, gps.longitude) {
                    println!("  - GPS Coordinates: {:.6}, {:.6}", lat, lon);
                }
                if let Some(alt) = gps.altitude {
                    println!("  - Altitude: {:.1}m", alt);
                }
            }

            // Other metadata
            if let Some(orientation) = exif.orientation {
                println!("  - Orientation: {}", orientation);
            }
            if let Some(software) = &exif.software {
                println!("  - Software: {}", software);
            }
        }
        None => {
            println!("No EXIF metadata found (this is expected for newly created images)");
        }
    }

    // Try to read EXIF from PNG
    println!("\nChecking test_no_exif.png for EXIF data...");
    match read_exif_metadata("test_no_exif.png")? {
        Some(_) => {
            println!("EXIF metadata found (unexpected for PNG)");
        }
        None => {
            println!("No EXIF metadata found (expected for PNG files)");
        }
    }

    println!();
    Ok(())
}

#[allow(dead_code)]
fn show_metadata_structure() -> Result<()> {
    println!("3. Understanding Metadata Structure");
    println!("-----------------------------------");

    // Load an image and examine its metadata
    let image_data = load_image("test_exif.jpg")?;

    println!("Basic Image Metadata:");
    println!("  - Width: {} pixels", image_data.metadata.width);
    println!("  - Height: {} pixels", image_data.metadata.height);
    println!("  - Color Mode: {:?}", image_data.metadata.color_mode);
    println!("  - Format: {:?}", image_data.metadata.format);
    println!("  - File Size: {} bytes", image_data.metadata.file_size);

    println!("\nEXIF Metadata Structure:");
    println!("  ExifMetadata {{");
    println!("    datetime: Option<DateTime<Utc>>,");
    println!("    gps: Option<GpsCoordinates>,");
    println!("    camera: CameraSettings,");
    println!("    orientation: Option<u32>,");
    println!("    software: Option<String>,");
    println!("    copyright: Option<String>,");
    println!("    artist: Option<String>,");
    println!("    description: Option<String>,");
    println!("    rawtags: HashMap<String, String>");
    println!("  }}");

    println!("\nGPS Coordinates Structure:");
    println!("  GpsCoordinates {{");
    println!("    latitude: Option<f64>,   // Decimal degrees");
    println!("    longitude: Option<f64>,  // Decimal degrees");
    println!("    altitude: Option<f64>    // Meters");
    println!("  }}");

    println!("\nCamera Settings Structure:");
    println!("  CameraSettings {{");
    println!("    make: Option<String>,");
    println!("    model: Option<String>,");
    println!("    lensmodel: Option<String>,");
    println!("    iso: Option<u32>,");
    println!("    aperture: Option<f64>,      // f-number");
    println!("    shutterspeed: Option<f64>, // seconds");
    println!("    focallength: Option<f64>,  // mm");
    println!("    flash: Option<bool>,");
    println!("    whitebalance: Option<String>,");
    println!("    exposuremode: Option<String>,");
    println!("    meteringmode: Option<String>");
    println!("  }}");

    println!("\nNote: EXIF reading is currently limited in this implementation.");
    println!("Full EXIF support would require proper kamadak-exif integration.");

    println!();
    Ok(())
}

#[allow(dead_code)]
fn cleanup_files() -> Result<()> {
    println!("4. Cleaning Up");
    println!("--------------");

    let files = vec!["test_exif.jpg", "test_no_exif.png"];

    for file in files {
        if std::path::Path::new(file).exists() {
            fs::remove_file(file)
                .map_err(|e| scirs2_io::error::IoError::FileError(e.to_string()))?;
            println!("✓ Removed {}", file);
        }
    }

    println!("\n✓ All examples completed successfully!");
    Ok(())
}
