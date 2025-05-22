use image::{open, DynamicImage, GenericImageView, ImageBuffer, Rgb, RgbImage, Rgba};
use scirs2_vision::transform::{
    generate_grid_points, warp_elastic, warp_non_rigid, warp_thin_plate_spline, BorderMode,
    ElasticDeformation,
};
use std::error::Error;
use std::time::Instant;

fn main() -> Result<(), Box<dyn Error>> {
    // Create output directory if it doesn't exist
    std::fs::create_dir_all("output")?;

    // Try to load test image from various possible locations
    let img = open("examples/data/lenna.png")
        .or_else(|_| open("examples/input/input.jpg"))
        .or_else(|_| open("input/input.jpg"))
        .expect("Failed to open image - please add a test image to examples/data/lenna.png or examples/input/input.jpg");
    println!("Loaded image: {}x{}", img.width(), img.height());

    // Example 1: Thin-Plate Spline transformation (smile effect)
    println!("\nExample 1: Thin-Plate Spline transformation (smile effect)");
    thin_plate_spline_example(&img)?;

    // Example 2: Elastic deformation
    println!("\nExample 2: Elastic deformation");
    elastic_deformation_example(&img)?;

    // Example 3: Grid deformation
    println!("\nExample 3: Grid deformation");
    grid_deformation_example()?;

    // Example 4: Compare border modes
    println!("\nExample 4: Compare border modes");
    compare_border_modes(&img)?;

    println!("\nAll examples completed. Check the 'output' directory for results.");
    Ok(())
}

/// Example 1: Thin-Plate Spline transformation (smile effect)
fn thin_plate_spline_example(img: &DynamicImage) -> Result<(), Box<dyn Error>> {
    let (width, height) = img.dimensions();

    // Create control points for smile effect
    let source_points = vec![
        // Corners
        (0.0, 0.0),
        (width as f64 - 1.0, 0.0),
        (0.0, height as f64 - 1.0),
        (width as f64 - 1.0, height as f64 - 1.0),
        // Face feature points (example for Lenna image)
        // Eyes
        (width as f64 * 0.35, height as f64 * 0.35), // Left eye
        (width as f64 * 0.65, height as f64 * 0.35), // Right eye
        // Mouth corners
        (width as f64 * 0.35, height as f64 * 0.65), // Left mouth corner
        (width as f64 * 0.65, height as f64 * 0.65), // Right mouth corner
        // Mouth center
        (width as f64 * 0.5, height as f64 * 0.68), // Mouth center
    ];

    // Target points for smile effect (move mouth corners up, center down)
    let mut target_points = source_points.clone();

    // Move mouth corners up for smile effect
    target_points[6].1 -= height as f64 * 0.05; // Left mouth corner up
    target_points[7].1 -= height as f64 * 0.05; // Right mouth corner up

    // Move mouth center slightly down
    target_points[8].1 += height as f64 * 0.03; // Mouth center down

    // Apply different regularization values
    let lambdas = [0.0, 0.1, 1.0];

    // Create a composite image to show all results
    let mut composite = ImageBuffer::<Rgba<u8>, Vec<u8>>::new(width * 2, height * 2);

    // Original image at top-left
    for y in 0..height {
        for x in 0..width {
            let pixel = img.get_pixel(x, y);
            composite.put_pixel(x, y, Rgba([pixel[0], pixel[1], pixel[2], 255]));
        }
    }

    // Process with different lambda values
    for (i, &lambda) in lambdas.iter().enumerate() {
        let start = Instant::now();

        // Apply thin-plate spline transformation
        let result = warp_thin_plate_spline(
            img,
            source_points.clone(),
            target_points.clone(),
            Some(lambda),
            BorderMode::Replicate,
        )?;

        let duration = start.elapsed();
        println!("  Lambda = {}: {}ms", lambda, duration.as_millis());

        // Save individual result
        result.save(format!("output/tps_smile_lambda_{}.png", lambda))?;

        // Add to composite
        let pos_x = if i == 0 {
            width
        } else {
            width * (i % 2) as u32
        };
        let pos_y = if i < 2 { 0 } else { height };

        for y in 0..height {
            for x in 0..width {
                let pixel = result.get_pixel(x, y);
                composite.put_pixel(pos_x + x, pos_y + y, pixel);
            }
        }
    }

    // Save composite image
    DynamicImage::ImageRgba8(composite).save("output/tps_smile_comparison.png")?;

    // Create an animation to show the transformation
    create_tps_animation(img, source_points, target_points, 10)?;

    Ok(())
}

/// Example 2: Elastic deformation
fn elastic_deformation_example(img: &DynamicImage) -> Result<(), Box<dyn Error>> {
    let (width, height) = img.dimensions();

    // Create a composite image for different parameters
    let mut composite = ImageBuffer::<Rgba<u8>, Vec<u8>>::new(width * 3, height * 2);

    // Original image in top-left
    for y in 0..height {
        for x in 0..width {
            let pixel = img.get_pixel(x, y);
            composite.put_pixel(x, y, Rgba([pixel[0], pixel[1], pixel[2], 255]));
        }
    }

    // Try different parameter combinations
    let alphas = [5.0, 15.0, 25.0];
    let sigmas = [3.0, 8.0, 15.0];

    let mut idx = 1;

    for &alpha in &alphas {
        for &sigma in &sigmas {
            if idx > 5 {
                break; // We only have space for 5 variations
            }

            let start = Instant::now();

            // Create elastic deformation with fixed seed for reproducibility
            let result = warp_elastic(
                img,
                alpha,
                sigma,
                Some(42), // Fixed seed
                BorderMode::Replicate,
            )?;

            let duration = start.elapsed();
            println!(
                "  Alpha = {}, Sigma = {}: {}ms",
                alpha,
                sigma,
                duration.as_millis()
            );

            // Save individual result
            result.save(format!(
                "output/elastic_alpha_{}_sigma_{}.png",
                alpha, sigma
            ))?;

            // Add to composite
            let pos_x = (idx % 3) as u32 * width;
            let pos_y = if idx < 3 { 0 } else { height };

            for y in 0..height {
                for x in 0..width {
                    let pixel = result.get_pixel(x, y);
                    composite.put_pixel(pos_x + x, pos_y + y, pixel);
                }
            }

            idx += 1;
        }
    }

    // Save composite image
    DynamicImage::ImageRgba8(composite).save("output/elastic_deformation_comparison.png")?;

    // Create animation with varying parameters
    create_elastic_animation(img, 10)?;

    Ok(())
}

/// Example 3: Grid deformation
fn grid_deformation_example() -> Result<(), Box<dyn Error>> {
    // Create a grid image
    let width = 400;
    let height = 400;
    let grid_spacing = 20;

    let mut grid_img = RgbImage::new(width, height);

    // Draw grid lines
    for y in 0..height {
        for x in 0..width {
            if x % grid_spacing == 0 || y % grid_spacing == 0 {
                grid_img.put_pixel(x, y, Rgb([0, 0, 0]));
            } else {
                grid_img.put_pixel(x, y, Rgb([255, 255, 255]));
            }
        }
    }

    let grid_dynamic = DynamicImage::ImageRgb8(grid_img);

    // Create source and target points for deformation
    // We'll create a bulge in the center
    let source_points = generate_grid_points(width, height, 7, 7);

    let mut target_points = source_points.clone();

    // Create a bulge effect
    let center_x = width as f64 / 2.0;
    let center_y = height as f64 / 2.0;
    let max_radius = (width.min(height) / 2) as f64;

    for i in 0..target_points.len() {
        let (x, y) = source_points[i];

        // Calculate distance from center
        let dx = x - center_x;
        let dy = y - center_y;
        let distance = (dx * dx + dy * dy).sqrt();

        // Apply bulge effect based on distance
        if distance < max_radius {
            // Strength decreases with distance
            let factor = 0.3 * (1.0 - distance / max_radius);

            // Push points outward
            target_points[i].0 = x + dx * factor;
            target_points[i].1 = y + dy * factor;
        }
    }

    // Apply TPS transformation
    let start = Instant::now();
    let result = warp_thin_plate_spline(
        &grid_dynamic,
        source_points,
        target_points,
        Some(0.1), // Small regularization
        BorderMode::Replicate,
    )?;

    let duration = start.elapsed();
    println!("  Grid deformation: {}ms", duration.as_millis());

    // Save results
    grid_dynamic.save("output/grid_original.png")?;
    result.save("output/grid_deformed.png")?;

    Ok(())
}

/// Example 4: Compare border modes
fn compare_border_modes(img: &DynamicImage) -> Result<(), Box<dyn Error>> {
    let (width, height) = img.dimensions();

    // Create a composite image for different border modes
    let mut composite = ImageBuffer::<Rgba<u8>, Vec<u8>>::new(width * 3, height * 2);

    // Create an elastic deformation with significant displacement
    let elastic = ElasticDeformation::new(width, height, 30.0, 15.0, Some(42))?;

    // Try different border modes
    let border_modes = [
        (BorderMode::Constant(Rgba([0, 0, 0, 255])), "constant"),
        (BorderMode::Reflect, "reflect"),
        (BorderMode::Replicate, "replicate"),
        (BorderMode::Wrap, "wrap"),
        (BorderMode::Transparent, "transparent"),
    ];

    for (i, (mode, name)) in border_modes.iter().enumerate() {
        let start = Instant::now();

        // Apply the same deformation with different border modes
        let result = warp_non_rigid(img, &elastic, *mode)?;

        let duration = start.elapsed();
        println!("  Border mode {}: {}ms", name, duration.as_millis());

        // Save individual result
        result.save(format!("output/border_mode_{}.png", name))?;

        // Add to composite
        let pos_x = (i % 3) as u32 * width;
        let pos_y = if i < 3 { 0 } else { height };

        for y in 0..height {
            for x in 0..width {
                let pixel = result.get_pixel(x, y);
                composite.put_pixel(pos_x + x, pos_y + y, pixel);
            }
        }
    }

    // Save composite image
    DynamicImage::ImageRgba8(composite).save("output/border_mode_comparison.png")?;

    Ok(())
}

/// Create an animation showing the thin-plate spline transformation
fn create_tps_animation(
    img: &DynamicImage,
    source_points: Vec<(f64, f64)>,
    target_points: Vec<(f64, f64)>,
    frames: usize,
) -> Result<(), Box<dyn Error>> {
    std::fs::create_dir_all("output/animation")?;

    for frame in 0..=frames {
        // Interpolate between source and target points
        let factor = frame as f64 / frames as f64;
        let mut intermediate_points = Vec::with_capacity(source_points.len());

        for i in 0..source_points.len() {
            let src = source_points[i];
            let dst = target_points[i];

            let x = src.0 + (dst.0 - src.0) * factor;
            let y = src.1 + (dst.1 - src.1) * factor;

            intermediate_points.push((x, y));
        }

        // Apply transformation
        let result = warp_thin_plate_spline(
            img,
            source_points.clone(),
            intermediate_points,
            Some(0.1),
            BorderMode::Replicate,
        )?;

        // Save frame
        result.save(format!("output/animation/tps_frame_{:02}.png", frame))?;
    }

    println!("  Animation frames saved to output/animation/");
    Ok(())
}

/// Create an animation showing elastic deformation
fn create_elastic_animation(img: &DynamicImage, frames: usize) -> Result<(), Box<dyn Error>> {
    let (width, height) = img.dimensions();

    std::fs::create_dir_all("output/animation")?;

    // Fix the seed for reproducibility
    let seed = 42;

    for frame in 0..=frames {
        // Vary alpha and sigma parameters
        let alpha = 5.0 + 15.0 * (frame as f64 / frames as f64);
        let sigma = 5.0 + 10.0 * ((frames - frame) as f64 / frames as f64);

        // Create elastic deformation
        let elastic = ElasticDeformation::new(width, height, alpha, sigma, Some(seed))?;

        // Apply deformation
        let result = warp_non_rigid(img, &elastic, BorderMode::Replicate)?;

        // Save frame
        result.save(format!("output/animation/elastic_frame_{:02}.png", frame))?;
    }

    println!("  Animation frames saved to output/animation/");
    Ok(())
}
