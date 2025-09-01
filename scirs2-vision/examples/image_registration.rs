//! Image registration example
//!
//! This example demonstrates various image registration techniques including
//! feature-based and intensity-based methods.

use image::{DynamicImage, ImageBuffer, Luma, Rgb, RgbImage};
use ndarray::Array2;
use rand::prelude::*;
use scirs2_vision::error::Result;
use scirs2_vision::registration::{
    feature_based::{register_images, FeatureRegistrationConfig},
    intensity::{register_images_intensity, IntensityRegistrationConfig, SimilarityMetric},
    ransac_estimate_transform, transform_point,
    warping::{warp_image, BoundaryMethod, InterpolationMethod},
    Point2D, PointMatch, RegistrationParams, TransformType,
};

#[allow(dead_code)]
fn main() -> Result<()> {
    println!("Image Registration Example");
    println!("=========================");

    // Create synthetic test images with known transformations
    let (reference, target, ground_truth_transform) = create_test_images()?;

    println!("Created synthetic test images (300x300)");

    // 1. Feature-based registration
    println!("\n1. Feature-based Registration");
    println!("-----------------------------");

    let mut feature_config = FeatureRegistrationConfig {
        transform_type: TransformType::Affine,
        ..Default::default()
    };

    // Lower the threshold to detect more features
    feature_config.detector_params.threshold = 0.001;
    feature_config.detector_params.harris_k = 0.04;

    let feature_result = register_images(&reference, &target, &feature_config)?;

    println!("Feature-based registration completed:");
    println!("  Reference features: {}", feature_result.ref_features);
    println!("  Target features: {}", feature_result.target_features);
    println!("  Initial matches: {}", feature_result.initial_matches);
    println!("  Final inliers: {}", feature_result.final_matches);
    println!(
        "  Final cost: {:.6}",
        feature_result.registration.final_cost
    );
    println!("  Iterations: {}", feature_result.registration.iterations);

    // 2. Intensity-based registration
    println!("\n2. Intensity-based Registration");
    println!("-------------------------------");

    let intensity_config = IntensityRegistrationConfig {
        metric: SimilarityMetric::NCC,
        use_pyramid: true,
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

    // 3. Point-based registration with synthetic matches
    println!("\n3. Point-based Registration (Synthetic Matches)");
    println!("-----------------------------------------------");

    let point_matches = create_synthetic_matches(50);
    let params = RegistrationParams::default();

    let point_result = ransac_estimate_transform(&point_matches, TransformType::Affine, &params)?;

    println!("Point-based registration completed:");
    println!("  Final cost: {:.6}", point_result.final_cost);
    println!(
        "  Inliers: {}/{}",
        point_result.inliers.len(),
        point_matches.len()
    );

    // 4. Transform comparison and error analysis
    println!("\n4. Transform Analysis");
    println!("--------------------");

    analyze_transform_accuracy(
        &ground_truth_transform,
        &feature_result.registration.transform,
        "Feature-based",
    );
    analyze_transform_accuracy(
        &ground_truth_transform,
        &intensity_result.transform,
        "Intensity-based",
    );
    analyze_transform_accuracy(
        &ground_truth_transform,
        &point_result.transform,
        "Point-based",
    );

    // 5. Apply transformations and save results
    println!("\n5. Applying Transformations");
    println!("--------------------------");

    // Apply feature-based transform
    let registered_feature = warp_image(
        &target_gray,
        &feature_result.registration.transform,
        (300, 300),
        InterpolationMethod::Bilinear,
        BoundaryMethod::Zero,
    )?;

    // Apply intensity-based transform
    let registered_intensity = warp_image(
        &target_gray,
        &intensity_result.transform,
        (300, 300),
        InterpolationMethod::Bilinear,
        BoundaryMethod::Zero,
    )?;

    // Clone for saving since we need to use them later for quality assessment
    let registered_feature_clone = registered_feature.clone();
    let registered_intensity_clone = registered_intensity.clone();

    // Create visualizations
    let comparison = create_registration_comparison(
        &ref_gray,
        &target_gray,
        &registered_feature,
        &registered_intensity,
    )?;

    // Save outputs
    std::fs::create_dir_all("output").ok();

    reference.save("output/registration_reference.png").ok();
    target.save("output/registration_target.png").ok();

    if DynamicImage::ImageLuma8(registered_feature_clone)
        .save("output/registration_feature_result.png")
        .is_ok()
    {
        println!("  Saved feature-based result");
    }

    if DynamicImage::ImageLuma8(registered_intensity_clone)
        .save("output/registration_intensity_result.png")
        .is_ok()
    {
        println!("  Saved intensity-based result");
    }

    if DynamicImage::ImageRgb8(comparison)
        .save("output/registration_comparison.png")
        .is_ok()
    {
        println!("  Saved comparison visualization");
    }

    // 6. Registration quality assessment
    println!("\n6. Quality Assessment");
    println!("--------------------");

    let feature_mse = compute_mse(&ref_gray, &registered_feature);
    let intensity_mse = compute_mse(&ref_gray, &registered_intensity);

    println!("Mean Squared Error:");
    println!("  Feature-based: {feature_mse:.2}");
    println!("  Intensity-based: {intensity_mse:.2}");

    let feature_ncc = compute_ncc(&ref_gray, &registered_feature);
    let intensity_ncc = compute_ncc(&ref_gray, &registered_intensity);

    println!("Normalized Cross-Correlation:");
    println!("  Feature-based: {feature_ncc:.4}");
    println!("  Intensity-based: {intensity_ncc:.4}");

    println!("\nRegistration example completed successfully!");
    println!("Check the 'output' directory for result images.");

    Ok(())
}

/// Create synthetic test images with known transformation
#[allow(dead_code)]
fn create_test_images() -> Result<(DynamicImage, DynamicImage, Array2<f64>)> {
    let width = 300;
    let height = 300;

    // Create reference image with geometric patterns
    let mut ref_img = ImageBuffer::new(width, height);

    // Add gradient background
    for y in 0..height {
        for x in 0..width {
            let intensity = ((x + y) % 200 + 55) as u8;
            ref_img.put_pixel(x, y, Luma([intensity]));
        }
    }

    // Add geometric shapes with more corners
    add_circle(&mut ref_img, 75, 75, 30, 255);
    add_rectangle(&mut ref_img, 150, 50, 80, 50, 200);
    add_circle(&mut ref_img, 220, 180, 25, 180);
    add_rectangle(&mut ref_img, 50, 200, 60, 40, 220);

    // Add more corner patterns
    for i in 0..5 {
        for j in 0..5 {
            let x = 60 + i * 50;
            let y = 60 + j * 50;
            if x < width - 10 && y < height - 10 {
                add_corner_pattern(&mut ref_img, x, y, 8, 240);
            }
        }
    }

    // Add some noise
    let mut rng = rand::rng();
    for _ in 0..500 {
        let x = rng.random_range(0..width);
        let y = rng.random_range(0..height);
        let intensity = rng.random_range(100u8..255u8);
        ref_img.put_pixel(x, y, Luma([intensity]));
    }

    let reference = DynamicImage::ImageLuma8(ref_img);

    // Create known transformation (translation + rotation + small scale)
    let mut transform = Array2::eye(3);
    let angle: f64 = 0.2; // ~11 degrees
    let cos_a = angle.cos();
    let sin_a = angle.sin();
    let scale = 1.1;

    transform[[0, 0]] = scale * cos_a;
    transform[[0, 1]] = -scale * sin_a;
    transform[[0, 2]] = 20.0; // Translation x
    transform[[1, 0]] = scale * sin_a;
    transform[[1, 1]] = scale * cos_a;
    transform[[1, 2]] = 15.0; // Translation y

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

/// Add a corner pattern to create good feature points
#[allow(dead_code)]
fn add_corner_pattern(
    img: &mut ImageBuffer<Luma<u8>, Vec<u8>>,
    cx: u32,
    cy: u32,
    size: u32,
    intensity: u8,
) {
    let (width, height) = img.dimensions();

    // Draw horizontal line
    for x in cx.saturating_sub(size)..=cx.saturating_add(size).min(width - 1) {
        if cy < height {
            img.put_pixel(x, cy, Luma([intensity]));
        }
    }

    // Draw vertical line
    for y in cy.saturating_sub(size)..=cy.saturating_add(size).min(height - 1) {
        if cx < width {
            img.put_pixel(cx, y, Luma([intensity]));
        }
    }
}

/// Create synthetic point matches with some outliers
#[allow(dead_code)]
fn create_synthetic_matches(_nummatches: usize) -> Vec<PointMatch> {
    let mut _matches = Vec::new();
    let mut rng = rand::rng();

    // Create mostly good _matches with known transformation
    let true_transform = Array2::from_shape_vec(
        (3, 3),
        vec![
            1.1 * 0.8_f64.cos(),
            -1.1 * 0.8_f64.sin(),
            20.0,
            1.1 * 0.8_f64.sin(),
            1.1 * 0.8_f64.cos(),
            15.0,
            0.0,
            0.0,
            1.0,
        ],
    )
    .unwrap();

    for _ in 0..(_nummatches * 8 / 10) {
        // 80% good _matches
        let source = Point2D::new(rng.random_range(50.0..250.0), rng.random_range(50.0..250.0));

        let target = transform_point(source, &true_transform);

        // Add small amount of noise
        let noisy_target = Point2D::new(
            target.x + rng.random_range(-2.0..2.0),
            target.y + rng.random_range(-2.0..2.0),
        );

        _matches.push(PointMatch {
            source,
            target: noisy_target,
            confidence: rng.random_range(0.8..1.0),
        });
    }

    // Add outliers
    for _ in (_nummatches * 8 / 10).._nummatches {
        _matches.push(PointMatch {
            source: Point2D::new(rng.random_range(50.0..250.0), rng.random_range(50.0..250.0)),
            target: Point2D::new(rng.random_range(50.0..250.0), rng.random_range(50.0..250.0)),
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

    // Compute scale difference
    let gt_scale = (ground_truth[[0, 0]].powi(2) + ground_truth[[1, 0]].powi(2)).sqrt();
    let est_scale = (estimated[[0, 0]].powi(2) + estimated[[1, 0]].powi(2)).sqrt();
    let scale_error = ((gt_scale - est_scale) / gt_scale * 100.0).abs();

    println!("{method_name} transform accuracy:");
    println!("  Translation error: {translation_error:.2} pixels");
    println!("  Rotation error: {rotation_error:.2} degrees");
    println!("  Scale error: {scale_error:.2}%");
}

/// Create registration comparison visualization
#[allow(dead_code)]
fn create_registration_comparison(
    reference: &ImageBuffer<Luma<u8>, Vec<u8>>,
    target: &ImageBuffer<Luma<u8>, Vec<u8>>,
    registered_feature: &ImageBuffer<Luma<u8>, Vec<u8>>,
    registered_intensity: &ImageBuffer<Luma<u8>, Vec<u8>>,
) -> Result<RgbImage> {
    let (width, height) = reference.dimensions();
    let comp_width = width * 2;
    let comp_height = height * 2;

    let mut comparison = RgbImage::new(comp_width, comp_height);

    // Top-left: Reference
    for y in 0..height {
        for x in 0..width {
            let val = reference.get_pixel(x, y)[0];
            comparison.put_pixel(x, y, Rgb([val, val, val]));
        }
    }

    // Top-right: Target
    for y in 0..height {
        for x in 0..width {
            let val = target.get_pixel(x, y)[0];
            comparison.put_pixel(x + width, y, Rgb([val, val, val]));
        }
    }

    // Bottom-left: Feature-based result
    for y in 0..height {
        for x in 0..width {
            let val = registered_feature.get_pixel(x, y)[0];
            comparison.put_pixel(x, y + height, Rgb([val, val, val]));
        }
    }

    // Bottom-right: Intensity-based result
    for y in 0..height {
        for x in 0..width {
            let val = registered_intensity.get_pixel(x, y)[0];
            comparison.put_pixel(x + width, y + height, Rgb([val, val, val]));
        }
    }

    // Add labels (simplified - would need a text rendering library for proper labels)
    // For now, add colored borders to distinguish quadrants

    // Reference border (blue)
    for x in 0..width {
        comparison.put_pixel(x, 0, Rgb([0, 0, 255]));
        comparison.put_pixel(x, height - 1, Rgb([0, 0, 255]));
    }
    for y in 0..height {
        comparison.put_pixel(0, y, Rgb([0, 0, 255]));
        comparison.put_pixel(width - 1, y, Rgb([0, 0, 255]));
    }

    // Target border (red)
    for x in width..comp_width {
        comparison.put_pixel(x, 0, Rgb([255, 0, 0]));
        comparison.put_pixel(x, height - 1, Rgb([255, 0, 0]));
    }
    for y in 0..height {
        comparison.put_pixel(width, y, Rgb([255, 0, 0]));
        comparison.put_pixel(comp_width - 1, y, Rgb([255, 0, 0]));
    }

    Ok(comparison)
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
