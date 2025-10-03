//! Enhanced image capabilities example
//!
//! This example demonstrates the advanced image processing features including:
//! - Multi-scale image pyramids for efficient processing at different resolutions
//! - Lossless compression using modern algorithms
//! - Advanced image processing operations (histogram equalization, sharpening, etc.)
//! - Efficient batch processing with enhanced compression
//! - Quality-preserving image operations

use scirs2_core::ndarray::Array3;
use scirs2_io::image::enhanced::{
    batch_convert_with_compression, create_image_pyramid, save_high_quality, save_lossless,
    CompressionOptions, CompressionQuality, EnhancedImageProcessor, InterpolationMethod,
    PyramidConfig,
};
use scirs2_io::image::{save_image, ColorMode, ImageData, ImageFormat, ImageMetadata};
use std::time::Instant;
use tempfile::tempdir;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🖼️  Enhanced Image Capabilities Example");
    println!("======================================");

    // Demonstrate multi-scale image pyramids
    demonstrate_image_pyramids()?;

    // Demonstrate lossless compression
    demonstrate_lossless_compression()?;

    // Demonstrate advanced processing operations
    demonstrate_advanced_processing()?;

    // Demonstrate enhanced batch processing
    demonstrate_batch_processing()?;

    // Demonstrate performance optimization
    demonstrate_performance_features()?;

    println!("\n✅ All enhanced image demonstrations completed successfully!");
    println!("💡 Key benefits of the enhanced image system:");
    println!("   - Multi-scale processing for efficient operations at different resolutions");
    println!("   - Lossless compression with advanced quality control");
    println!("   - Advanced image processing operations with quality preservation");
    println!("   - Efficient batch processing with memory optimization");
    println!("   - Performance-optimized algorithms for large images");

    Ok(())
}

#[allow(dead_code)]
fn demonstrate_image_pyramids() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📊 Demonstrating Multi-Scale Image Pyramids...");

    let temp_dir = tempdir()?;

    // Create a test image with gradient patterns
    println!("  🔹 Creating test image with detailed patterns:");
    let mut test_array = Array3::zeros((200, 200, 3));

    // Create a complex pattern to demonstrate pyramid levels
    for y in 0..200 {
        for x in 0..200 {
            // Checkerboard pattern
            let checker = ((x / 10) + (y / 10)) % 2;

            // Gradient overlay
            let grad_x = (x as f32 / 200.0 * 255.0) as u8;
            let grad_y = (y as f32 / 200.0 * 255.0) as u8;

            if checker == 0 {
                test_array[[y, x, 0]] = grad_x;
                test_array[[y, x, 1]] = grad_y;
                test_array[[y, x, 2]] = 255 - grad_x;
            } else {
                test_array[[y, x, 0]] = 255 - grad_x;
                test_array[[y, x, 1]] = 255 - grad_y;
                test_array[[y, x, 2]] = grad_x;
            }
        }
    }

    let metadata = ImageMetadata {
        width: 200,
        height: 200,
        color_mode: ColorMode::RGB,
        format: ImageFormat::PNG,
        file_size: 0,
        exif: None,
    };

    let image_data = ImageData {
        data: test_array,
        metadata,
    };

    // Save original image
    let original_path = temp_dir.path().join("original.png");
    save_image(&image_data, &original_path, Some(ImageFormat::PNG))?;
    println!("    Created 200x200 test image with complex patterns");

    // Create pyramid with default configuration
    println!("  🔹 Creating image pyramid with default configuration:");
    let pyramid_start = Instant::now();
    let pyramid = create_image_pyramid(&image_data)?;
    let pyramid_time = pyramid_start.elapsed();

    println!(
        "    Pyramid creation time: {:.2}ms",
        pyramid_time.as_secs_f64() * 1000.0
    );
    println!("    Number of pyramid levels: {}", pyramid.num_levels());

    // Save all pyramid levels
    for level in 0..pyramid.num_levels() {
        if let Some(level_image) = pyramid.get_level(level) {
            let level_path = temp_dir.path().join(format!("pyramid_level_{}.png", level));
            save_image(level_image, &level_path, Some(ImageFormat::PNG))?;
            println!(
                "      Level {}: {}x{} saved",
                level, level_image.metadata.width, level_image.metadata.height
            );
        }
    }

    // Demonstrate level selection for target sizes
    println!("  🔹 Demonstrating automatic level selection:");
    let target_sizes = vec![(200, 200), (100, 100), (50, 50), (25, 25)];

    for (target_width, target_height) in target_sizes {
        let best_level = pyramid.find_best_level(target_width, target_height);
        if let Some(level_image) = pyramid.get_level(best_level) {
            println!(
                "    Target {}x{} -> Level {} ({}x{})",
                target_width,
                target_height,
                best_level,
                level_image.metadata.width,
                level_image.metadata.height
            );
        }
    }

    // Create custom pyramid configuration
    println!("  🔹 Creating custom pyramid configuration:");
    let custom_config = PyramidConfig {
        levels: 6,
        scale_factor: 0.7,
        min_size: 16,
        interpolation: InterpolationMethod::Lanczos,
    };

    let processor = EnhancedImageProcessor::new();
    let custom_pyramid = processor.create_pyramid(&image_data, custom_config)?;
    println!("    Custom pyramid levels: {}", custom_pyramid.num_levels());

    for level in 0..custom_pyramid.num_levels() {
        if let Some(level_image) = custom_pyramid.get_level(level) {
            println!(
                "      Level {}: {}x{}",
                level, level_image.metadata.width, level_image.metadata.height
            );
        }
    }

    Ok(())
}

#[allow(dead_code)]
fn demonstrate_lossless_compression() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n💾 Demonstrating Lossless Compression...");

    let temp_dir = tempdir()?;

    // Create a photographic-style test image
    println!("  🔹 Creating photographic test image:");
    let mut photo_array = Array3::zeros((150, 150, 3));

    // Create smooth gradients and color variations typical of photos
    for y in 0..150 {
        for x in 0..150 {
            let center_x = 75.0;
            let center_y = 75.0;
            let distance = ((x as f32 - center_x).powi(2) + (y as f32 - center_y).powi(2)).sqrt();

            // Radial gradient with color variation
            let intensity = (255.0 * (1.0 - (distance / 106.0).min(1.0))) as u8;
            let hue_shift = (distance / 10.0) as u8;

            photo_array[[y, x, 0]] = intensity.saturating_sub(hue_shift);
            photo_array[[y, x, 1]] = intensity;
            photo_array[[y, x, 2]] = intensity.saturating_add(hue_shift / 2);
        }
    }

    let photo_metadata = ImageMetadata {
        width: 150,
        height: 150,
        color_mode: ColorMode::RGB,
        format: ImageFormat::PNG,
        file_size: 0,
        exif: None,
    };

    let photo_image = ImageData {
        data: photo_array,
        metadata: photo_metadata,
    };

    // Test different compression qualities
    println!("  🔹 Testing different compression qualities:");
    let processor = EnhancedImageProcessor::new();

    let quality_settings = vec![
        ("Lossless", CompressionQuality::Lossless),
        ("High", CompressionQuality::High),
        ("Medium", CompressionQuality::Medium),
        ("Low", CompressionQuality::Low),
        ("Custom 90", CompressionQuality::Custom(90)),
    ];

    let mut results = Vec::new();

    for (name, quality) in quality_settings {
        let compression = CompressionOptions {
            quality,
            progressive: true,
            optimize: true,
            compression_level: Some(9),
        };

        // Test with JPEG
        let jpeg_path = temp_dir
            .path()
            .join(format!("test_{}.jpg", name.to_lowercase()));
        let jpeg_start = Instant::now();
        processor.save_with_compression(
            &photo_image,
            &jpeg_path,
            ImageFormat::JPEG,
            Some(compression.clone()),
        )?;
        let jpeg_time = jpeg_start.elapsed();
        let jpeg_size = std::fs::metadata(&jpeg_path)?.len();

        // Test with PNG
        let png_path = temp_dir
            .path()
            .join(format!("test_{}.png", name.to_lowercase()));
        let png_start = Instant::now();
        processor.save_with_compression(
            &photo_image,
            &png_path,
            ImageFormat::PNG,
            Some(compression),
        )?;
        let png_time = png_start.elapsed();
        let png_size = std::fs::metadata(&png_path)?.len();

        results.push((
            name,
            quality.value(),
            jpeg_size,
            jpeg_time,
            png_size,
            png_time,
        ));

        println!(
            "    {}: Quality {} - JPEG: {:.1}KB ({:.1}ms), PNG: {:.1}KB ({:.1}ms)",
            name,
            quality.value(),
            jpeg_size as f64 / 1024.0,
            jpeg_time.as_secs_f64() * 1000.0,
            png_size as f64 / 1024.0,
            png_time.as_secs_f64() * 1000.0
        );
    }

    // Demonstrate convenience functions
    println!("  🔹 Testing convenience functions:");

    let lossless_path = temp_dir.path().join("convenience_lossless.png");
    let lossless_start = Instant::now();
    save_lossless(&photo_image, &lossless_path, ImageFormat::PNG)?;
    let lossless_time = lossless_start.elapsed();
    let lossless_size = std::fs::metadata(&lossless_path)?.len();

    let high_quality_path = temp_dir.path().join("convenience_high_quality.jpg");
    let hq_start = Instant::now();
    save_high_quality(&photo_image, &high_quality_path, ImageFormat::JPEG)?;
    let hq_time = hq_start.elapsed();
    let hq_size = std::fs::metadata(&high_quality_path)?.len();

    println!(
        "    Lossless PNG: {:.1}KB ({:.1}ms)",
        lossless_size as f64 / 1024.0,
        lossless_time.as_secs_f64() * 1000.0
    );
    println!(
        "    High Quality JPEG: {:.1}KB ({:.1}ms)",
        hq_size as f64 / 1024.0,
        hq_time.as_secs_f64() * 1000.0
    );

    // Analyze compression efficiency
    println!("  🔹 Compression efficiency analysis:");
    if let Some((_, _, best_jpeg_size, _, _, _)) = results
        .iter()
        .min_by_key(|(_, _, jpeg_size, _, _, _)| *jpeg_size)
    {
        if let Some((_, _, worst_jpeg_size, _, _, _)) = results
            .iter()
            .max_by_key(|(_, _, jpeg_size, _, _, _)| *jpeg_size)
        {
            let compression_ratio = *worst_jpeg_size as f64 / *best_jpeg_size as f64;
            println!("    Best JPEG compression ratio: {:.1}x", compression_ratio);
        }
    }

    Ok(())
}

#[allow(dead_code)]
fn demonstrate_advanced_processing() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🎨 Demonstrating Advanced Image Processing...");

    let temp_dir = tempdir()?;

    // Create a test image suitable for processing demos
    println!("  🔹 Creating test image for processing:");
    let mut source_array = Array3::zeros((120, 120, 3));

    // Create an image with various patterns and intensity levels
    for y in 0..120 {
        for x in 0..120 {
            // Create regions with different characteristics
            if x < 40 {
                // Dark region
                let intensity = 64 + (y as f32 / 120.0 * 96.0) as u8;
                source_array[[y, x, 0]] = intensity / 3;
                source_array[[y, x, 1]] = intensity / 2;
                source_array[[y, x, 2]] = intensity;
            } else if x < 80 {
                // Medium region
                let intensity = 128 + (y as f32 / 120.0 * 64.0) as u8;
                source_array[[y, x, 0]] = intensity;
                source_array[[y, x, 1]] = intensity;
                source_array[[y, x, 2]] = intensity / 2;
            } else {
                // Bright region
                let intensity = 192 + (y as f32 / 120.0 * 63.0) as u8;
                source_array[[y, x, 0]] = intensity;
                source_array[[y, x, 1]] = intensity / 2;
                source_array[[y, x, 2]] = intensity / 3;
            }
        }
    }

    let source_metadata = ImageMetadata {
        width: 120,
        height: 120,
        color_mode: ColorMode::RGB,
        format: ImageFormat::PNG,
        file_size: 0,
        exif: None,
    };

    let source_image = ImageData {
        data: source_array,
        metadata: source_metadata,
    };

    // Save original
    let original_path = temp_dir.path().join("processing_original.png");
    save_image(&source_image, &original_path, Some(ImageFormat::PNG))?;

    let processor = EnhancedImageProcessor::new();

    // Demonstrate grayscale conversion
    println!("  🔹 Converting to grayscale with luminance preservation:");
    let gray_start = Instant::now();
    let grayscale = processor.to_grayscale(&source_image)?;
    let gray_time = gray_start.elapsed();

    let gray_path = temp_dir.path().join("processing_grayscale.png");
    save_image(&grayscale, &gray_path, Some(ImageFormat::PNG))?;
    println!(
        "    Grayscale conversion time: {:.2}ms",
        gray_time.as_secs_f64() * 1000.0
    );

    // Demonstrate histogram equalization
    println!("  🔹 Applying histogram equalization for contrast enhancement:");
    let eq_start = Instant::now();
    let equalized = processor.histogram_equalization(&source_image)?;
    let eq_time = eq_start.elapsed();

    let eq_path = temp_dir.path().join("processing_equalized.png");
    save_image(&equalized, &eq_path, Some(ImageFormat::PNG))?;
    println!(
        "    Histogram equalization time: {:.2}ms",
        eq_time.as_secs_f64() * 1000.0
    );

    // Demonstrate Gaussian blur
    println!("  🔹 Applying Gaussian blur filters:");
    let blur_radii = vec![1.0, 2.5, 5.0];

    for radius in blur_radii {
        let blur_start = Instant::now();
        let blurred = processor.gaussian_blur(&source_image, radius)?;
        let blur_time = blur_start.elapsed();

        let blur_path = temp_dir
            .path()
            .join(format!("processing_blur_{:.1}.png", radius));
        save_image(&blurred, &blur_path, Some(ImageFormat::PNG))?;
        println!(
            "    Blur radius {:.1}: {:.2}ms",
            radius,
            blur_time.as_secs_f64() * 1000.0
        );
    }

    // Demonstrate sharpening
    println!("  🔹 Applying unsharp mask sharpening:");
    let sharpen_settings = vec![(0.5, 1.0), (1.0, 1.5), (1.5, 2.0)];

    for (amount, radius) in sharpen_settings {
        let sharpen_start = Instant::now();
        let sharpened = processor.sharpen(&source_image, amount, radius)?;
        let sharpen_time = sharpen_start.elapsed();

        let sharpen_path = temp_dir.path().join(format!(
            "processing_sharpen_{:.1}_{:.1}.png",
            amount, radius
        ));
        save_image(&sharpened, &sharpen_path, Some(ImageFormat::PNG))?;
        println!(
            "    Sharpen amount {:.1}, radius {:.1}: {:.2}ms",
            amount,
            radius,
            sharpen_time.as_secs_f64() * 1000.0
        );
    }

    // Demonstrate interpolation methods
    println!("  🔹 Testing different interpolation methods:");
    let target_size = (60, 60);
    let interpolation_methods = vec![
        ("Nearest", InterpolationMethod::Nearest),
        ("Linear", InterpolationMethod::Linear),
        ("Cubic", InterpolationMethod::Cubic),
        ("Lanczos", InterpolationMethod::Lanczos),
    ];

    for (name, method) in interpolation_methods {
        let interp_start = Instant::now();
        let resized = processor.resize_with_interpolation(
            &source_image,
            target_size.0,
            target_size.1,
            method,
        )?;
        let interp_time = interp_start.elapsed();

        let interp_path = temp_dir
            .path()
            .join(format!("processing_interp_{}.png", name.to_lowercase()));
        save_image(&resized, &interp_path, Some(ImageFormat::PNG))?;
        println!(
            "    {} interpolation: {:.2}ms",
            name,
            interp_time.as_secs_f64() * 1000.0
        );
    }

    println!("  ✅ All processing operations completed successfully!");

    Ok(())
}

#[allow(dead_code)]
fn demonstrate_batch_processing() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n⚡ Demonstrating Enhanced Batch Processing...");

    let temp_dir = tempdir()?;
    let input_dir = temp_dir.path().join("input");
    let output_dir = temp_dir.path().join("output");

    std::fs::create_dir_all(&input_dir)?;
    std::fs::create_dir_all(&output_dir)?;

    // Create several test images
    println!("  🔹 Creating test images for batch processing:");
    let test_images = vec![
        ("red_image", [255, 100, 100]),
        ("green_image", [100, 255, 100]),
        ("blue_image", [100, 100, 255]),
        ("yellow_image", [255, 255, 100]),
        ("purple_image", [255, 100, 255]),
    ];

    for (name, color) in &test_images {
        let mut image_array = Array3::zeros((80, 80, 3));

        // Fill with solid color and add some patterns
        for y in 0..80 {
            for x in 0..80 {
                let pattern = if (x / 10 + y / 10) % 2 == 0 { 0.8 } else { 1.0 };
                image_array[[y, x, 0]] = (color[0] as f32 * pattern) as u8;
                image_array[[y, x, 1]] = (color[1] as f32 * pattern) as u8;
                image_array[[y, x, 2]] = (color[2] as f32 * pattern) as u8;
            }
        }

        let metadata = ImageMetadata {
            width: 80,
            height: 80,
            color_mode: ColorMode::RGB,
            format: ImageFormat::PNG,
            file_size: 0,
            exif: None,
        };

        let image_data = ImageData {
            data: image_array,
            metadata,
        };

        let image_path = input_dir.join(format!("{}.png", name));
        save_image(&image_data, &image_path, Some(ImageFormat::PNG))?;
    }

    println!("    Created {} test images", test_images.len());

    // Demonstrate batch conversion with different compression settings
    println!("  🔹 Batch converting with lossless compression:");
    let lossless_compression = CompressionOptions {
        quality: CompressionQuality::Lossless,
        progressive: false,
        optimize: true,
        compression_level: Some(9),
    };

    let lossless_output = output_dir.join("lossless");
    let batch_start = Instant::now();
    batch_convert_with_compression(
        &input_dir,
        &lossless_output,
        ImageFormat::PNG,
        lossless_compression,
    )?;
    let batch_time = batch_start.elapsed();
    println!(
        "    Lossless batch conversion time: {:.2}ms",
        batch_time.as_secs_f64() * 1000.0
    );

    // Demonstrate batch conversion with high quality JPEG
    println!("  🔹 Batch converting with high quality JPEG:");
    let hq_compression = CompressionOptions {
        quality: CompressionQuality::High,
        progressive: true,
        optimize: true,
        compression_level: Some(8),
    };

    let hq_output = output_dir.join("high_quality");
    let hq_start = Instant::now();
    batch_convert_with_compression(&input_dir, &hq_output, ImageFormat::JPEG, hq_compression)?;
    let hq_time = hq_start.elapsed();
    println!(
        "    High quality batch conversion time: {:.2}ms",
        hq_time.as_secs_f64() * 1000.0
    );

    // Analyze file sizes
    println!("  🔹 Analyzing compression results:");
    for (name, _) in &test_images {
        let original_path = input_dir.join(format!("{}.png", name));
        let lossless_path = lossless_output.join(format!("{}.png", name));
        let hq_path = hq_output.join(format!("{}.jpg", name));

        let original_size = std::fs::metadata(&original_path)?.len();
        let lossless_size = std::fs::metadata(&lossless_path)?.len();
        let hq_size = std::fs::metadata(&hq_path)?.len();

        println!(
            "    {}: Original {:.1}KB, Lossless {:.1}KB, HQ JPEG {:.1}KB",
            name,
            original_size as f64 / 1024.0,
            lossless_size as f64 / 1024.0,
            hq_size as f64 / 1024.0
        );
    }

    Ok(())
}

#[allow(dead_code)]
fn demonstrate_performance_features() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🚀 Demonstrating Performance Features...");

    let temp_dir = tempdir()?;

    // Create large test image for performance testing
    println!("  🔹 Creating large test image for performance testing:");
    let large_size = 300;
    let mut large_array = Array3::zeros((large_size, large_size, 3));

    // Create complex pattern
    for y in 0..large_size {
        for x in 0..large_size {
            let r = ((x + y) as f32 / (large_size * 2) as f32 * 255.0) as u8;
            let g = ((x * y) as f32 / (large_size * large_size) as f32 * 255.0) as u8;
            let b = ((x.abs_diff(y)) as f32 / large_size as f32 * 255.0) as u8;

            large_array[[y, x, 0]] = r;
            large_array[[y, x, 1]] = g;
            large_array[[y, x, 2]] = b;
        }
    }

    let large_metadata = ImageMetadata {
        width: large_size as u32,
        height: large_size as u32,
        color_mode: ColorMode::RGB,
        format: ImageFormat::PNG,
        file_size: 0,
        exif: None,
    };

    let large_image = ImageData {
        data: large_array,
        metadata: large_metadata,
    };

    println!("    Created {}x{} test image", large_size, large_size);

    // Test pyramid creation performance
    println!("  🔹 Testing pyramid creation performance:");
    let pyramid_configs = vec![
        ("Standard", PyramidConfig::default()),
        (
            "More Levels",
            PyramidConfig {
                levels: 8,
                scale_factor: 0.6,
                min_size: 8,
                interpolation: InterpolationMethod::Linear,
            },
        ),
        (
            "High Quality",
            PyramidConfig {
                levels: 6,
                scale_factor: 0.5,
                min_size: 16,
                interpolation: InterpolationMethod::Lanczos,
            },
        ),
    ];

    let processor = EnhancedImageProcessor::new();

    for (name, config) in pyramid_configs {
        let pyramid_start = Instant::now();
        let pyramid = processor.create_pyramid(&large_image, config)?;
        let pyramid_time = pyramid_start.elapsed();

        println!(
            "    {} pyramid: {:.2}ms, {} levels",
            name,
            pyramid_time.as_secs_f64() * 1000.0,
            pyramid.num_levels()
        );
    }

    // Test processing operation performance
    println!("  🔹 Testing processing operation performance:");

    let operations = vec![
        (
            "Grayscale",
            Box::new(|p: &EnhancedImageProcessor, img: &ImageData| p.to_grayscale(img))
                as Box<
                    dyn Fn(
                        &EnhancedImageProcessor,
                        &ImageData,
                    ) -> Result<ImageData, scirs2_io::error::IoError>,
                >,
        ),
        (
            "Histogram Eq",
            Box::new(|p: &EnhancedImageProcessor, img: &ImageData| p.histogram_equalization(img)),
        ),
        (
            "Blur (r=2.0)",
            Box::new(|p: &EnhancedImageProcessor, img: &ImageData| p.gaussian_blur(img, 2.0)),
        ),
        (
            "Sharpen (1.0)",
            Box::new(|p: &EnhancedImageProcessor, img: &ImageData| p.sharpen(img, 1.0, 1.5)),
        ),
    ];

    for (name, operation) in operations {
        let op_start = Instant::now();
        let _result = operation(&processor, &large_image)?;
        let op_time = op_start.elapsed();

        println!("    {}: {:.2}ms", name, op_time.as_secs_f64() * 1000.0);
    }

    // Test compression performance
    println!("  🔹 Testing compression performance:");
    let compression_tests = vec![
        (
            "PNG Lossless",
            ImageFormat::PNG,
            CompressionQuality::Lossless,
        ),
        ("JPEG High", ImageFormat::JPEG, CompressionQuality::High),
        ("JPEG Medium", ImageFormat::JPEG, CompressionQuality::Medium),
    ];

    for (name, format, quality) in compression_tests {
        let compression = CompressionOptions {
            quality,
            progressive: true,
            optimize: true,
            compression_level: Some(9),
        };

        let comp_path = temp_dir.path().join(format!(
            "perf_test_{}.{}",
            name.to_lowercase().replace(" ", "_"),
            format.extension()
        ));

        let comp_start = Instant::now();
        processor.save_with_compression(&large_image, &comp_path, format, Some(compression))?;
        let comp_time = comp_start.elapsed();
        let file_size = std::fs::metadata(&comp_path)?.len();

        println!(
            "    {}: {:.2}ms, {:.1}KB",
            name,
            comp_time.as_secs_f64() * 1000.0,
            file_size as f64 / 1024.0
        );
    }

    // Test cache performance
    println!("  🔹 Testing cache system:");
    let mut cached_processor = EnhancedImageProcessor::new().with_cache_size(64);
    let (initial_count, max_size) = cached_processor.cache_stats();
    println!(
        "    Cache initialized: {} items, {}MB max",
        initial_count, max_size
    );

    // Simulate some operations that would benefit from caching
    for i in 0..3 {
        let cache_start = Instant::now();
        let _result = cached_processor.to_grayscale(&large_image)?;
        let cache_time = cache_start.elapsed();
        println!(
            "    Cache test {}: {:.2}ms",
            i + 1,
            cache_time.as_secs_f64() * 1000.0
        );
    }

    cached_processor.clear_cache();
    let (final_count, _) = cached_processor.cache_stats();
    println!("    Cache cleared: {} items remaining", final_count);

    println!("  ✅ Performance testing completed!");

    Ok(())
}
