//! Intensity-based image registration
//!
//! This module implements registration algorithms that directly use image intensity
//! values to find optimal alignment between images.

use crate::error::{Result, VisionError};
use crate::registration::warping::{warp_image, BoundaryMethod, InterpolationMethod};
use crate::registration::{
    identity_transform, RegistrationParams, RegistrationResult, TransformMatrix,
};
use image::GrayImage;
use ndarray::{Array1, Array2};

/// Similarity metric for intensity-based registration
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SimilarityMetric {
    /// Sum of Squared Differences
    SSD,
    /// Normalized Cross-Correlation
    NCC,
    /// Mutual Information
    MI,
    /// Normalized Mutual Information
    NMI,
    /// Cross-Correlation
    CC,
    /// Mean Squared Error
    MSE,
}

/// Optimization algorithm for registration
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OptimizationMethod {
    /// Gradient Descent
    GradientDescent,
    /// Powell's method
    Powell,
    /// Simplex method
    Simplex,
    /// Conjugate Gradient
    ConjugateGradient,
}

/// Intensity-based registration configuration
#[derive(Debug, Clone)]
pub struct IntensityRegistrationConfig {
    /// Similarity metric to optimize
    pub metric: SimilarityMetric,
    /// Optimization method
    pub optimizer: OptimizationMethod,
    /// Registration parameters
    pub params: RegistrationParams,
    /// Step size for gradient computation
    pub step_size: f64,
    /// Learning rate for optimization
    pub learning_rate: f64,
    /// Use multi-resolution pyramid
    pub use_pyramid: bool,
}

impl Default for IntensityRegistrationConfig {
    fn default() -> Self {
        Self {
            metric: SimilarityMetric::NCC,
            optimizer: OptimizationMethod::GradientDescent,
            params: RegistrationParams::default(),
            step_size: 0.1,
            learning_rate: 0.01,
            use_pyramid: true,
        }
    }
}

/// Register images using intensity-based methods
///
/// # Arguments
///
/// * `reference` - Reference image
/// * `moving` - Moving image to register
/// * `initial_transform` - Initial transformation estimate
/// * `config` - Registration configuration
///
/// # Returns
///
/// * Result containing registration result
pub fn register_images_intensity(
    reference: &GrayImage,
    moving: &GrayImage,
    initial_transform: Option<&TransformMatrix>,
    config: &IntensityRegistrationConfig,
) -> Result<RegistrationResult> {
    let initial_transform = initial_transform
        .cloned()
        .unwrap_or_else(identity_transform);

    if config.use_pyramid {
        multi_resolution_register(reference, moving, &initial_transform, config)
    } else {
        single_level_register(reference, moving, &initial_transform, config)
    }
}

/// Single-level intensity-based registration
fn single_level_register(
    reference: &GrayImage,
    moving: &GrayImage,
    initial_transform: &TransformMatrix,
    config: &IntensityRegistrationConfig,
) -> Result<RegistrationResult> {
    let mut current_transform = initial_transform.clone();
    let mut current_cost =
        compute_similarity(reference, moving, &current_transform, config.metric)?;
    let mut iteration = 0;
    let mut converged = false;

    while iteration < config.params.max_iterations && !converged {
        let (gradient, new_cost) = compute_gradient(reference, moving, &current_transform, config)?;

        // Check for convergence
        if (current_cost - new_cost).abs() < config.params.tolerance {
            converged = true;
            break;
        }

        // Update transform based on optimization method
        current_transform = match config.optimizer {
            OptimizationMethod::GradientDescent => update_transform_gradient_descent(
                &current_transform,
                &gradient,
                config.learning_rate,
            ),
            OptimizationMethod::Powell => {
                // Simplified Powell's method (would need full implementation)
                update_transform_gradient_descent(
                    &current_transform,
                    &gradient,
                    config.learning_rate,
                )
            }
            OptimizationMethod::Simplex => {
                // Simplified Simplex method (would need full implementation)
                update_transform_gradient_descent(
                    &current_transform,
                    &gradient,
                    config.learning_rate,
                )
            }
            OptimizationMethod::ConjugateGradient => {
                // Simplified Conjugate Gradient (would need full implementation)
                update_transform_gradient_descent(
                    &current_transform,
                    &gradient,
                    config.learning_rate,
                )
            }
        };

        current_cost = new_cost;
        iteration += 1;
    }

    Ok(RegistrationResult {
        transform: current_transform,
        final_cost: current_cost,
        iterations: iteration,
        converged,
        inliers: Vec::new(), // Not applicable for intensity-based methods
    })
}

/// Multi-resolution pyramid registration
fn multi_resolution_register(
    reference: &GrayImage,
    moving: &GrayImage,
    initial_transform: &TransformMatrix,
    config: &IntensityRegistrationConfig,
) -> Result<RegistrationResult> {
    // Build pyramids
    let ref_pyramid = build_image_pyramid(reference, config.params.pyramid_levels);
    let moving_pyramid = build_image_pyramid(moving, config.params.pyramid_levels);

    let mut current_transform = initial_transform.clone();
    let mut final_result = None;

    // Register from coarse to fine
    for level in (0..config.params.pyramid_levels).rev() {
        // Scale transform for current level
        let scale = 2.0_f64.powi(level as i32);
        let mut scaled_transform = current_transform.clone();
        scaled_transform[[0, 2]] /= scale;
        scaled_transform[[1, 2]] /= scale;

        // Register at current level
        let result = single_level_register(
            &ref_pyramid[level],
            &moving_pyramid[level],
            &scaled_transform,
            config,
        )?;

        current_transform = result.transform.clone();

        // Scale transform back up for next level
        if level > 0 {
            current_transform[[0, 2]] *= 2.0;
            current_transform[[1, 2]] *= 2.0;
        }

        final_result = Some(result);
    }

    final_result.ok_or_else(|| {
        VisionError::OperationError("Multi-resolution registration failed".to_string())
    })
}

/// Build image pyramid by downsampling
fn build_image_pyramid(image: &GrayImage, levels: usize) -> Vec<GrayImage> {
    let mut pyramid = vec![image.clone()];

    for _ in 1..levels {
        let prev = &pyramid[pyramid.len() - 1];
        let (width, height) = prev.dimensions();

        if width < 8 || height < 8 {
            break;
        }

        let downsampled = downsample_image(prev);
        pyramid.push(downsampled);
    }

    pyramid
}

/// Downsample image by factor of 2
fn downsample_image(image: &GrayImage) -> GrayImage {
    let (width, height) = image.dimensions();
    let new_width = width / 2;
    let new_height = height / 2;

    let mut downsampled = GrayImage::new(new_width, new_height);

    for y in 0..new_height {
        for x in 0..new_width {
            // Average 2x2 block
            let x2 = x * 2;
            let y2 = y * 2;

            let mut sum = image.get_pixel(x2, y2)[0] as u32;
            let mut count = 1;

            if x2 + 1 < width {
                sum += image.get_pixel(x2 + 1, y2)[0] as u32;
                count += 1;
            }
            if y2 + 1 < height {
                sum += image.get_pixel(x2, y2 + 1)[0] as u32;
                count += 1;
            }
            if x2 + 1 < width && y2 + 1 < height {
                sum += image.get_pixel(x2 + 1, y2 + 1)[0] as u32;
                count += 1;
            }

            downsampled.put_pixel(x, y, image::Luma([(sum / count) as u8]));
        }
    }

    downsampled
}

/// Compute similarity metric between images
fn compute_similarity(
    reference: &GrayImage,
    moving: &GrayImage,
    transform: &TransformMatrix,
    metric: SimilarityMetric,
) -> Result<f64> {
    let (width, height) = reference.dimensions();

    // Warp moving image
    let warped = warp_image(
        moving,
        transform,
        (width, height),
        InterpolationMethod::Bilinear,
        BoundaryMethod::Zero,
    )?;

    match metric {
        SimilarityMetric::SSD => compute_ssd(reference, &warped),
        SimilarityMetric::NCC => compute_ncc(reference, &warped),
        SimilarityMetric::MI => compute_mutual_information(reference, &warped),
        SimilarityMetric::NMI => compute_normalized_mutual_information(reference, &warped),
        SimilarityMetric::CC => compute_cross_correlation(reference, &warped),
        SimilarityMetric::MSE => compute_mse(reference, &warped),
    }
}

/// Compute gradient of similarity metric
fn compute_gradient(
    reference: &GrayImage,
    moving: &GrayImage,
    transform: &TransformMatrix,
    config: &IntensityRegistrationConfig,
) -> Result<(Array1<f64>, f64)> {
    let current_cost = compute_similarity(reference, moving, transform, config.metric)?;

    // Compute gradient using finite differences
    let mut gradient = Array1::zeros(6); // For affine transform parameters

    for i in 0..6 {
        let mut perturbed_transform = transform.clone();

        // Perturb parameter
        match i {
            0 => perturbed_transform[[0, 0]] += config.step_size,
            1 => perturbed_transform[[0, 1]] += config.step_size,
            2 => perturbed_transform[[0, 2]] += config.step_size,
            3 => perturbed_transform[[1, 0]] += config.step_size,
            4 => perturbed_transform[[1, 1]] += config.step_size,
            5 => perturbed_transform[[1, 2]] += config.step_size,
            _ => {}
        }

        let perturbed_cost =
            compute_similarity(reference, moving, &perturbed_transform, config.metric)?;
        gradient[i] = (perturbed_cost - current_cost) / config.step_size;
    }

    Ok((gradient, current_cost))
}

/// Update transformation using gradient descent
fn update_transform_gradient_descent(
    transform: &TransformMatrix,
    gradient: &Array1<f64>,
    learning_rate: f64,
) -> TransformMatrix {
    let mut updated = transform.clone();

    // Update parameters
    updated[[0, 0]] -= learning_rate * gradient[0];
    updated[[0, 1]] -= learning_rate * gradient[1];
    updated[[0, 2]] -= learning_rate * gradient[2];
    updated[[1, 0]] -= learning_rate * gradient[3];
    updated[[1, 1]] -= learning_rate * gradient[4];
    updated[[1, 2]] -= learning_rate * gradient[5];

    updated
}

/// Compute Sum of Squared Differences
fn compute_ssd(image1: &GrayImage, image2: &GrayImage) -> Result<f64> {
    let (width, height) = image1.dimensions();
    let mut ssd = 0.0;
    let mut count = 0;

    for y in 0..height {
        for x in 0..width {
            let val1 = image1.get_pixel(x, y)[0] as f64;
            let val2 = image2.get_pixel(x, y)[0] as f64;

            ssd += (val1 - val2).powi(2);
            count += 1;
        }
    }

    Ok(ssd / count as f64)
}

/// Compute Mean Squared Error
fn compute_mse(image1: &GrayImage, image2: &GrayImage) -> Result<f64> {
    compute_ssd(image1, image2) // MSE is the same as average SSD
}

/// Compute Normalized Cross-Correlation
fn compute_ncc(image1: &GrayImage, image2: &GrayImage) -> Result<f64> {
    let (width, height) = image1.dimensions();

    let mut sum1 = 0.0;
    let mut sum2 = 0.0;
    let mut sum1_sq = 0.0;
    let mut sum2_sq = 0.0;
    let mut sum_cross = 0.0;
    let mut count = 0;

    for y in 0..height {
        for x in 0..width {
            let val1 = image1.get_pixel(x, y)[0] as f64;
            let val2 = image2.get_pixel(x, y)[0] as f64;

            // Skip zero pixels (likely from warping)
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
        return Ok(0.0);
    }

    let n = count as f64;
    let mean1 = sum1 / n;
    let mean2 = sum2 / n;

    let numerator = sum_cross - n * mean1 * mean2;
    let var1 = sum1_sq - n * mean1 * mean1;
    let var2 = sum2_sq - n * mean2 * mean2;

    let denominator = (var1 * var2).sqrt();

    if denominator > 1e-10 {
        Ok(-numerator / denominator) // Negative because we want to minimize
    } else {
        Ok(0.0)
    }
}

/// Compute Cross-Correlation
fn compute_cross_correlation(image1: &GrayImage, image2: &GrayImage) -> Result<f64> {
    let (width, height) = image1.dimensions();
    let mut cc = 0.0;
    let mut count = 0;

    for y in 0..height {
        for x in 0..width {
            let val1 = image1.get_pixel(x, y)[0] as f64;
            let val2 = image2.get_pixel(x, y)[0] as f64;

            if val2 > 0.0 {
                cc += val1 * val2;
                count += 1;
            }
        }
    }

    if count > 0 {
        Ok(-cc / count as f64) // Negative because we want to maximize CC
    } else {
        Ok(0.0)
    }
}

/// Compute Mutual Information
fn compute_mutual_information(image1: &GrayImage, image2: &GrayImage) -> Result<f64> {
    let joint_hist = compute_joint_histogram(image1, image2, 256);
    let (hist1, hist2) = compute_marginal_histograms(&joint_hist);

    let mut mi = 0.0;
    let total = joint_hist.sum();

    for i in 0..256 {
        for j in 0..256 {
            let p_xy = joint_hist[[i, j]] / total;
            let p_x = hist1[i] / total;
            let p_y = hist2[j] / total;

            if p_xy > 1e-10 && p_x > 1e-10 && p_y > 1e-10 {
                mi += p_xy * (p_xy / (p_x * p_y)).ln();
            }
        }
    }

    Ok(-mi) // Negative because we want to maximize MI
}

/// Compute Normalized Mutual Information
fn compute_normalized_mutual_information(image1: &GrayImage, image2: &GrayImage) -> Result<f64> {
    let joint_hist = compute_joint_histogram(image1, image2, 256);
    let (hist1, hist2) = compute_marginal_histograms(&joint_hist);

    let total = joint_hist.sum();

    // Compute entropies
    let mut h1 = 0.0;
    let mut h2 = 0.0;
    let mut h12 = 0.0;

    for i in 0..256 {
        let p1 = hist1[i] / total;
        if p1 > 1e-10 {
            h1 -= p1 * p1.ln();
        }

        let p2 = hist2[i] / total;
        if p2 > 1e-10 {
            h2 -= p2 * p2.ln();
        }

        for j in 0..256 {
            let p_xy = joint_hist[[i, j]] / total;
            if p_xy > 1e-10 {
                h12 -= p_xy * p_xy.ln();
            }
        }
    }

    let nmi = (h1 + h2) / h12;
    Ok(-nmi) // Negative because we want to maximize NMI
}

/// Compute joint histogram of two images
fn compute_joint_histogram(image1: &GrayImage, image2: &GrayImage, bins: usize) -> Array2<f64> {
    let (width, height) = image1.dimensions();
    let mut hist = Array2::zeros((bins, bins));

    for y in 0..height {
        for x in 0..width {
            let val1 = image1.get_pixel(x, y)[0] as usize;
            let val2 = image2.get_pixel(x, y)[0] as usize;

            if val1 < bins && val2 < bins && val2 > 0 {
                hist[[val1, val2]] += 1.0;
            }
        }
    }

    hist
}

/// Compute marginal histograms from joint histogram
fn compute_marginal_histograms(joint_hist: &Array2<f64>) -> (Array1<f64>, Array1<f64>) {
    let (bins1, bins2) = joint_hist.dim();
    let mut hist1 = Array1::zeros(bins1);
    let mut hist2 = Array1::zeros(bins2);

    for i in 0..bins1 {
        for j in 0..bins2 {
            hist1[i] += joint_hist[[i, j]];
            hist2[j] += joint_hist[[i, j]];
        }
    }

    (hist1, hist2)
}

/// Rigid registration using intensity-based methods
pub fn rigid_register_intensity(
    reference: &GrayImage,
    moving: &GrayImage,
    config: &IntensityRegistrationConfig,
) -> Result<RegistrationResult> {
    // For rigid registration, we need to constrain the optimization
    // This is a simplified version - would need proper rigid constraint handling
    register_images_intensity(reference, moving, None, config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Luma};

    fn create_test_image(width: u32, height: u32, pattern: u8) -> GrayImage {
        ImageBuffer::from_fn(width, height, |x, y| {
            Luma([((x + y + pattern as u32) % 256) as u8])
        })
    }

    #[test]
    fn test_intensity_config() {
        let config = IntensityRegistrationConfig::default();
        assert_eq!(config.metric, SimilarityMetric::NCC);
        assert_eq!(config.optimizer, OptimizationMethod::GradientDescent);
    }

    #[test]
    fn test_ssd_computation() {
        let img1 = create_test_image(10, 10, 0);
        let img2 = create_test_image(10, 10, 1);

        let ssd = compute_ssd(&img1, &img2).unwrap();
        assert!(ssd > 0.0);
    }

    #[test]
    fn test_ncc_computation() {
        let img1 = create_test_image(10, 10, 0);
        let img2 = create_test_image(10, 10, 0); // Same image

        let ncc = compute_ncc(&img1, &img2).unwrap();
        assert!(ncc <= 0.0); // Should be close to -1 (perfect correlation)
    }

    #[test]
    fn test_pyramid_building() {
        let image = create_test_image(64, 64, 0);
        let pyramid = build_image_pyramid(&image, 3);

        assert_eq!(pyramid.len(), 3);
        assert_eq!(pyramid[0].dimensions(), (64, 64));
        assert_eq!(pyramid[1].dimensions(), (32, 32));
        assert_eq!(pyramid[2].dimensions(), (16, 16));
    }

    #[test]
    fn test_joint_histogram() {
        let img1 = create_test_image(10, 10, 0);
        let img2 = create_test_image(10, 10, 0);

        let hist = compute_joint_histogram(&img1, &img2, 256);
        assert!(hist.sum() > 0.0);
    }

    #[test]
    fn test_mutual_information() {
        let img1 = create_test_image(20, 20, 0);
        let img2 = create_test_image(20, 20, 1);

        let mi = compute_mutual_information(&img1, &img2).unwrap();
        assert!(mi.is_finite());
    }

    #[test]
    fn test_gradient_computation() {
        let img1 = create_test_image(20, 20, 0);
        let img2 = create_test_image(20, 20, 1);
        let transform = identity_transform();
        let config = IntensityRegistrationConfig::default();

        let result = compute_gradient(&img1, &img2, &transform, &config);
        assert!(result.is_ok());

        let (gradient, _cost) = result.unwrap();
        assert_eq!(gradient.len(), 6);
    }
}
