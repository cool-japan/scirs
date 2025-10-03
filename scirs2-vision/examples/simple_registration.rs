//! Simple image registration example
//!
//! This example demonstrates basic image registration techniques using
//! synthetic point matches and intensity-based methods.

use image::{DynamicImage, ImageBuffer, Luma};
use scirs2_core::ndarray::Array2;
use scirs2_core::random::prelude::*;
use scirs2_vision::error::Result;
use scirs2_vision::registration::{
    intensity::{register_images_intensity, IntensityRegistrationConfig, SimilarityMetric},
    ransac_estimate_transform, transform_point,
    warping::{warp_image, BoundaryMethod, InterpolationMethod},
    Point2D, PointMatch, RegistrationParams, TransformType,
};

#[allow(dead_code)]
fn main() -> Result<()> {
    println!("Simple Image Registration Example");
    println!("=================================");

    // Create synthetic test images
    let (reference, target, ground_truth_transform) = create_simple_test_images()?;

    println!("Created synthetic test images (200x200)");

    // 1. Point-based registration with synthetic matches
    println!("\n1. Point-based Registration (Synthetic Matches)");
    println!("-----------------------------------------------");

    let point_matches = create_synthetic_matches(20, &ground_truth_transform);
    let params = RegistrationParams::default();

    let point_result = ransac_estimate_transform(&point_matches, TransformType::Affine, &params)?;

    println!("Point-based registration completed:");
    println!("  Final cost: {:.6}", point_result.final_cost);
    println!(
        "  Inliers: {}/{}",
        point_result.inliers.len(),
        point_matches.len()
    );

    // 2. Intensity-based registration
    println!("\n2. Intensity-based Registration");
    println!("-------------------------------");

    let intensity_config = IntensityRegistrationConfig {
        metric: SimilarityMetric::NCC,
        use_pyramid: false, // Simplified for this example
        ..Default::default()
    };

    let ref_gray = reference.to_luma8();
    let target_gray = target.to_luma8();

    let intensity_result =
        register_images_intensity(&ref_gray, &target_gray, None, &intensity_config)?;

    println!("Intensity-based registration completed:");
    println!("  Final cost: {:.6}", intensity_result.final_cost);
    println!("  Iterations: {}", intensity_result.iterations);
    println!("  Converged: {}", intensity_result.converged);

    // 3. Transform comparison and error analysis
    println!("\n3. Transform Analysis");
    println!("--------------------");

    analyze_transform_accuracy(
        &ground_truth_transform,
        &point_result.transform,
        "Point-based",
    );
    analyze_transform_accuracy(
        &ground_truth_transform,
        &intensity_result.transform,
        "Intensity-based",
    );

    // 4. Apply transformations
    println!("\n4. Applying Transformations");
    println!("--------------------------");

    let registered_point = warp_image(
        &target_gray,
        &point_result.transform,
        (200, 200),
        InterpolationMethod::Bilinear,
        BoundaryMethod::Zero,
    )?;

    let registered_intensity = warp_image(
        &target_gray,
        &intensity_result.transform,
        (200, 200),
        InterpolationMethod::Bilinear,
        BoundaryMethod::Zero,
    )?;

    // 5. Save results
    std::fs::create_dir_all("output").ok();

    reference.save("output/simple_reference.png").ok();
    target.save("output/simple_target.png").ok();
    DynamicImage::ImageLuma8(registered_point.clone())
        .save("output/simple_point_result.png")
        .ok();
    DynamicImage::ImageLuma8(registered_intensity.clone())
        .save("output/simple_intensity_result.png")
        .ok();

    println!("  Saved reference and target images");
    println!("  Saved registration results");

    // 6. Quality assessment
    println!("\n5. Quality Assessment");
    println!("--------------------");

    let point_mse = compute_mse(&ref_gray, &registered_point);
    let intensity_mse = compute_mse(&ref_gray, &registered_intensity);

    println!("Mean Squared Error:");
    println!("  Point-based: {point_mse:.2}");
    println!("  Intensity-based: {intensity_mse:.2}");

    let point_ncc = compute_ncc(&ref_gray, &registered_point);
    let intensity_ncc = compute_ncc(&ref_gray, &registered_intensity);

    println!("Normalized Cross-Correlation:");
    println!("  Point-based: {point_ncc:.4}");
    println!("  Intensity-based: {intensity_ncc:.4}");

    println!("\nSimple registration example completed successfully!");
    println!("Check the 'output' directory for result images.");

    Ok(())
}

/// Create simple synthetic test images with known transformation
#[allow(dead_code)]
fn create_simple_test_images() -> Result<(DynamicImage, DynamicImage, Array2<f64>)> {
    let width = 200;
    let height = 200;

    // Create reference image with clear patterns
    let mut ref_img = ImageBuffer::new(width, height);

    // Simple checkerboard pattern
    for y in 0..height {
        for x in 0..width {
            let checker_x = (x / 20) % 2;
            let checker_y = (y / 20) % 2;
            let intensity = if checker_x == checker_y { 200 } else { 100 };
            ref_img.put_pixel(x, y, Luma([intensity]));
        }
    }

    // Add some distinctive features
    add_circle(&mut ref_img, 50, 50, 15, 255);
    add_rectangle(&mut ref_img, 120, 30, 40, 20, 80);
    add_circle(&mut ref_img, 150, 120, 12, 255);
    add_rectangle(&mut ref_img, 30, 140, 30, 30, 80);

    let reference = DynamicImage::ImageLuma8(ref_img);

    // Create known transformation (small translation + rotation)
    let mut transform = Array2::eye(3);
    let angle: f64 = 0.1; // ~6 degrees
    let cos_a = angle.cos();
    let sin_a = angle.sin();

    transform[[0, 0]] = cos_a;
    transform[[0, 1]] = -sin_a;
    transform[[0, 2]] = 10.0; // Translation x
    transform[[1, 0]] = sin_a;
    transform[[1, 1]] = cos_a;
    transform[[1, 2]] = 8.0; // Translation y

    // Apply transformation to create target image
    let ref_gray = reference.to_luma8();
    let target_gray = warp_image(
        &ref_gray,
        &transform,
        (width, height),
        InterpolationMethod::Bilinear,
        BoundaryMethod::Zero,
    )?;

    let target = DynamicImage::ImageLuma8(target_gray);

    Ok((reference, target, transform))
}

/// Add a filled circle to an image
#[allow(dead_code)]
fn add_circle(
    img: &mut ImageBuffer<Luma<u8>, Vec<u8>>,
    cx: u32,
    cy: u32,
    radius: u32,
    intensity: u8,
) {
    let (width, height) = img.dimensions();
    let r_sq = (radius * radius) as i32;

    for y in cy.saturating_sub(radius)..=cy.saturating_add(radius).min(height - 1) {
        for x in cx.saturating_sub(radius)..=cx.saturating_add(radius).min(width - 1) {
            let dx = x as i32 - cx as i32;
            let dy = y as i32 - cy as i32;

            if dx * dx + dy * dy <= r_sq {
                img.put_pixel(x, y, Luma([intensity]));
            }
        }
    }
}

/// Add a filled rectangle to an image
#[allow(dead_code)]
fn add_rectangle(
    img: &mut ImageBuffer<Luma<u8>, Vec<u8>>,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
    intensity: u8,
) {
    let (img_width, img_height) = img.dimensions();

    for py in y..=(y + h).min(img_height - 1) {
        for px in x..=(x + w).min(img_width - 1) {
            img.put_pixel(px, py, Luma([intensity]));
        }
    }
}

/// Create synthetic point matches based on known transformation
#[allow(dead_code)]
fn create_synthetic_matches(_num_matches: usize, truetransform: &Array2<f64>) -> Vec<PointMatch> {
    let mut _matches = Vec::new();
    let mut rng = scirs2_core::random::rng();

    // Create good _matches with known transformation
    for _ in 0..(_num_matches * 9 / 10) {
        // 90% good _matches
        let source = Point2D::new(rng.random_range(20.0..180.0), rng.random_range(20.0..180.0));

        let target = transform_point(source, truetransform);

        // Add small amount of noise
        let noisy_target = Point2D::new(
            target.x + rng.random_range(-1.0..1.0),
            target.y + rng.random_range(-1.0..1.0),
        );

        _matches.push(PointMatch {
            source,
            target: noisy_target,
            confidence: rng.random_range(0.8..1.0),
        });
    }

    // Add some outliers
    for _ in (_num_matches * 9 / 10).._num_matches {
        _matches.push(PointMatch {
            source: Point2D::new(rng.random_range(20.0..180.0), rng.random_range(20.0..180.0)),
            target: Point2D::new(rng.random_range(20.0..180.0), rng.random_range(20.0..180.0)),
            confidence: rng.random_range(0.3..0.7),
        });
    }

    _matches
}

/// Analyze transformation accuracy
#[allow(dead_code)]
fn analyze_transform_accuracy(
    ground_truth: &Array2<f64>,
    estimated: &Array2<f64>,
    method_name: &str,
) {
    // Compute differences in key parameters
    let gt_tx = ground_truth[[0, 2]];
    let gt_ty = ground_truth[[1, 2]];
    let est_tx = estimated[[0, 2]];
    let est_ty = estimated[[1, 2]];

    let translation_error = ((gt_tx - est_tx).powi(2) + (gt_ty - est_ty).powi(2)).sqrt();

    // Compute rotation difference (simplified)
    let gt_angle = ground_truth[[1, 0]].atan2(ground_truth[[0, 0]]);
    let est_angle = estimated[[1, 0]].atan2(estimated[[0, 0]]);
    let rotation_error = (gt_angle - est_angle).abs() * 180.0 / std::f64::consts::PI;

    println!("{method_name} transform accuracy:");
    println!("  Translation error: {translation_error:.2} pixels");
    println!("  Rotation error: {rotation_error:.2} degrees");
}

/// Compute Mean Squared Error between two images
#[allow(dead_code)]
fn compute_mse(
    img1: &ImageBuffer<Luma<u8>, Vec<u8>>,
    img2: &ImageBuffer<Luma<u8>, Vec<u8>>,
) -> f64 {
    let (width, height) = img1.dimensions();
    let mut mse = 0.0;
    let mut count = 0;

    for y in 0..height {
        for x in 0..width {
            let val1 = img1.get_pixel(x, y)[0] as f64;
            let val2 = img2.get_pixel(x, y)[0] as f64;

            // Skip zero pixels from warping
            if val2 > 0.0 {
                mse += (val1 - val2).powi(2);
                count += 1;
            }
        }
    }

    if count > 0 {
        mse / count as f64
    } else {
        0.0
    }
}

/// Compute Normalized Cross-Correlation between two images
#[allow(dead_code)]
fn compute_ncc(
    img1: &ImageBuffer<Luma<u8>, Vec<u8>>,
    img2: &ImageBuffer<Luma<u8>, Vec<u8>>,
) -> f64 {
    let (width, height) = img1.dimensions();

    let mut sum1 = 0.0;
    let mut sum2 = 0.0;
    let mut sum1_sq = 0.0;
    let mut sum2_sq = 0.0;
    let mut sum_cross = 0.0;
    let mut count = 0;

    for y in 0..height {
        for x in 0..width {
            let val1 = img1.get_pixel(x, y)[0] as f64;
            let val2 = img2.get_pixel(x, y)[0] as f64;

            // Skip zero pixels from warping
            if val2 > 0.0 {
                sum1 += val1;
                sum2 += val2;
                sum1_sq += val1 * val1;
                sum2_sq += val2 * val2;
                sum_cross += val1 * val2;
                count += 1;
            }
        }
    }

    if count == 0 {
        return 0.0;
    }

    let n = count as f64;
    let mean1 = sum1 / n;
    let mean2 = sum2 / n;

    let numerator = sum_cross - n * mean1 * mean2;
    let var1 = sum1_sq - n * mean1 * mean1;
    let var2 = sum2_sq - n * mean2 * mean2;

    let denominator = (var1 * var2).sqrt();

    if denominator > 1e-10 {
        numerator / denominator
    } else {
        0.0
    }
}
