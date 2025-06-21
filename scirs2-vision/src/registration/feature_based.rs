//! Feature-based image registration
//!
//! This module implements registration algorithms that use detected features
//! and their matches to estimate transformations between images.

use crate::error::{Result, VisionError};
use crate::feature::{extract_feature_coordinates, harris_corners};
use crate::registration::{
    identity_transform, ransac_estimate_transform, Point2D, PointMatch, RegistrationParams,
    RegistrationResult, TransformType,
};
use image::{DynamicImage, GrayImage};

/// Feature-based registration configuration
#[derive(Debug, Clone)]
pub struct FeatureRegistrationConfig {
    /// Feature detection parameters
    pub detector_params: FeatureDetectorParams,
    /// Matching parameters
    pub matcher_params: MatcherParams,
    /// Registration parameters
    pub registration_params: RegistrationParams,
    /// Type of transformation to estimate
    pub transform_type: TransformType,
}

impl Default for FeatureRegistrationConfig {
    fn default() -> Self {
        Self {
            detector_params: FeatureDetectorParams::default(),
            matcher_params: MatcherParams::default(),
            registration_params: RegistrationParams::default(),
            transform_type: TransformType::Affine,
        }
    }
}

/// Feature detector parameters
#[derive(Debug, Clone)]
pub struct FeatureDetectorParams {
    /// Maximum number of features to detect
    pub max_features: usize,
    /// Feature detection threshold
    pub threshold: f32,
    /// Use ORB detector (otherwise use Harris)
    pub use_orb: bool,
    /// ORB parameters (placeholder)
    pub orb_params: (),
    /// Harris corner parameters
    pub harris_block_size: usize,
    /// Harris corner detection parameter k
    pub harris_k: f32,
}

impl Default for FeatureDetectorParams {
    fn default() -> Self {
        Self {
            max_features: 500,
            threshold: 0.01,
            use_orb: true,
            orb_params: (),
            harris_block_size: 7,
            harris_k: 0.04,
        }
    }
}

/// Matching parameters
#[derive(Debug, Clone)]
pub struct MatcherParams {
    /// Maximum distance for matches
    pub max_distance: f64,
    /// Use cross-check validation
    pub cross_check: bool,
    /// Use ratio test
    pub use_ratio_test: bool,
    /// Ratio test threshold
    pub ratio_threshold: f64,
}

impl Default for MatcherParams {
    fn default() -> Self {
        Self {
            max_distance: 0.8,
            cross_check: true,
            use_ratio_test: true,
            ratio_threshold: 0.75,
        }
    }
}

/// Feature-based registration result
#[derive(Debug, Clone)]
pub struct FeatureRegistrationResult {
    /// Registration result
    pub registration: RegistrationResult,
    /// Number of features detected in reference image
    pub ref_features: usize,
    /// Number of features detected in target image
    pub target_features: usize,
    /// Number of initial matches
    pub initial_matches: usize,
    /// Final inlier matches
    pub final_matches: usize,
}

/// Register two images using feature-based methods
///
/// # Arguments
///
/// * `reference` - Reference image
/// * `target` - Target image to register to reference
/// * `config` - Registration configuration
///
/// # Returns
///
/// * Result containing registration result and statistics
pub fn register_images(
    reference: &DynamicImage,
    target: &DynamicImage,
    config: &FeatureRegistrationConfig,
) -> Result<FeatureRegistrationResult> {
    // Detect features in both images
    let (ref_keypoints, ref_descriptors) = detect_and_describe(reference, &config.detector_params)?;
    let (target_keypoints, target_descriptors) =
        detect_and_describe(target, &config.detector_params)?;

    if ref_keypoints.is_empty() || target_keypoints.is_empty() {
        return Err(VisionError::OperationError(
            "Insufficient features detected".to_string(),
        ));
    }

    // Match features
    let matches = match_features(
        &ref_descriptors,
        &target_descriptors,
        &config.matcher_params,
    )?;

    if matches.len() < 4 {
        return Err(VisionError::OperationError(
            "Insufficient matches for registration".to_string(),
        ));
    }

    // Convert matches to point matches
    let point_matches: Vec<PointMatch> = matches
        .iter()
        .map(|m| PointMatch {
            source: Point2D::new(
                ref_keypoints[m.query_idx].x as f64,
                ref_keypoints[m.query_idx].y as f64,
            ),
            target: Point2D::new(
                target_keypoints[m.train_idx].x as f64,
                target_keypoints[m.train_idx].y as f64,
            ),
            confidence: m.confidence as f64,
        })
        .collect();

    // Estimate transformation using RANSAC
    let registration_result = ransac_estimate_transform(
        &point_matches,
        config.transform_type,
        &config.registration_params,
    )?;

    let final_matches = registration_result.inliers.len();

    Ok(FeatureRegistrationResult {
        registration: registration_result,
        ref_features: ref_keypoints.len(),
        target_features: target_keypoints.len(),
        initial_matches: matches.len(),
        final_matches,
    })
}

/// Detect features and compute descriptors
fn detect_and_describe(
    image: &DynamicImage,
    params: &FeatureDetectorParams,
) -> Result<(Vec<Keypoint>, Vec<Vec<u8>>)> {
    // Use Harris corner detector (ORB not available)
    let corners = harris_corners(
        image,
        params.harris_block_size,
        params.harris_k,
        params.threshold,
    )?;

    let coordinates = extract_feature_coordinates(&corners);

    // Convert to keypoints
    let keypoints: Vec<Keypoint> = coordinates
        .iter()
        .take(params.max_features)
        .map(|&(x, y)| Keypoint {
            x: x as f32,
            y: y as f32,
            scale: 1.0,
            angle: 0.0,
            response: 1.0,
        })
        .collect();

    // Generate simple descriptors for Harris corners
    let gray = image.to_luma8();
    let descriptors = generate_simple_descriptors(&gray, &keypoints)?;

    Ok((keypoints, descriptors))
}

/// Simple keypoint structure
#[derive(Debug, Clone)]
pub struct Keypoint {
    /// X coordinate of the keypoint
    pub x: f32,
    /// Y coordinate of the keypoint
    pub y: f32,
    /// Scale of the keypoint
    pub scale: f32,
    /// Orientation angle in radians
    pub angle: f32,
    /// Detection response strength
    pub response: f32,
}

/// Simple match structure
#[derive(Debug, Clone)]
pub struct FeatureMatch {
    /// Index of the query keypoint
    pub query_idx: usize,
    /// Index of the train keypoint
    pub train_idx: usize,
    /// Distance between feature descriptors
    pub distance: f32,
    /// Confidence score of the match
    pub confidence: f32,
}

/// Match features between two sets of descriptors
fn match_features(
    descriptors1: &[Vec<u8>],
    descriptors2: &[Vec<u8>],
    params: &MatcherParams,
) -> Result<Vec<FeatureMatch>> {
    if descriptors1.is_empty() || descriptors2.is_empty() {
        return Ok(Vec::new());
    }

    let mut matches = Vec::new();

    // Simple brute force matching
    for (i, desc1) in descriptors1.iter().enumerate() {
        let mut best_dist = f32::INFINITY;
        let mut second_best_dist = f32::INFINITY;
        let mut best_idx = None;

        for (j, desc2) in descriptors2.iter().enumerate() {
            // Compute Hamming distance
            let dist = hamming_distance(desc1, desc2);

            if dist < best_dist {
                second_best_dist = best_dist;
                best_dist = dist;
                best_idx = Some(j);
            } else if dist < second_best_dist {
                second_best_dist = dist;
            }
        }

        if let Some(idx) = best_idx {
            // Apply distance threshold
            if best_dist <= params.max_distance as f32 {
                // Apply ratio test if enabled
                let pass_ratio_test = if params.use_ratio_test {
                    second_best_dist > 0.0
                        && best_dist / second_best_dist < params.ratio_threshold as f32
                } else {
                    true
                };

                if pass_ratio_test {
                    matches.push(FeatureMatch {
                        query_idx: i,
                        train_idx: idx,
                        distance: best_dist,
                        confidence: (1.0 - best_dist / 256.0).max(0.0),
                    });
                }
            }
        }
    }

    // Apply cross-check if enabled
    if params.cross_check {
        matches = apply_cross_check(matches, descriptors1, descriptors2);
    }

    Ok(matches)
}

/// Compute Hamming distance between two descriptors
fn hamming_distance(desc1: &[u8], desc2: &[u8]) -> f32 {
    let min_len = desc1.len().min(desc2.len());
    let mut distance = 0;

    for i in 0..min_len {
        distance += (desc1[i] ^ desc2[i]).count_ones();
    }

    distance as f32
}

/// Apply cross-check validation
fn apply_cross_check(
    matches: Vec<FeatureMatch>,
    descriptors1: &[Vec<u8>],
    descriptors2: &[Vec<u8>],
) -> Vec<FeatureMatch> {
    let mut validated_matches = Vec::new();

    for m in matches {
        // Check if this match is also the best match in reverse direction
        let desc2 = &descriptors2[m.train_idx];
        let mut best_dist = f32::INFINITY;
        let mut best_idx = None;

        for (i, desc1) in descriptors1.iter().enumerate() {
            let dist = hamming_distance(desc2, desc1);
            if dist < best_dist {
                best_dist = dist;
                best_idx = Some(i);
            }
        }

        if best_idx == Some(m.query_idx) {
            validated_matches.push(m);
        }
    }

    validated_matches
}

/// Generate simple descriptors for Harris corners using patch-based approach
fn generate_simple_descriptors(image: &GrayImage, keypoints: &[Keypoint]) -> Result<Vec<Vec<u8>>> {
    let (width, height) = image.dimensions();
    let patch_size = 8;
    let half_patch = patch_size / 2;

    let mut descriptors = Vec::new();

    for kp in keypoints {
        let x = kp.x as i32;
        let y = kp.y as i32;

        // Skip keypoints too close to border
        if x < half_patch as i32
            || y < half_patch as i32
            || x >= (width - half_patch) as i32
            || y >= (height - half_patch) as i32
        {
            continue;
        }

        let mut descriptor = Vec::new();

        // Extract patch around keypoint
        for dy in -(half_patch as i32)..(half_patch as i32) {
            for dx in -(half_patch as i32)..(half_patch as i32) {
                let px = (x + dx) as u32;
                let py = (y + dy) as u32;

                if px < width && py < height {
                    descriptor.push(image.get_pixel(px, py)[0]);
                }
            }
        }

        descriptors.push(descriptor);
    }

    Ok(descriptors)
}

/// Multi-scale feature registration
pub fn multi_scale_register(
    reference: &DynamicImage,
    target: &DynamicImage,
    config: &FeatureRegistrationConfig,
    num_levels: usize,
) -> Result<FeatureRegistrationResult> {
    let mut current_transform = identity_transform();
    let mut best_result = None;
    let mut best_inliers = 0;

    // Build image pyramids
    let ref_pyramid = build_pyramid(reference, num_levels);
    let target_pyramid = build_pyramid(target, num_levels);

    // Register from coarse to fine
    for level in (0..num_levels).rev() {
        // Scale down the current transform
        let scale = 2.0_f64.powi(level as i32);
        let mut scaled_transform = current_transform.clone();

        // Scale translation components
        scaled_transform[[0, 2]] /= scale;
        scaled_transform[[1, 2]] /= scale;

        // Register at this level
        let level_config = config.clone();
        let result = register_images(&ref_pyramid[level], &target_pyramid[level], &level_config)?;

        // Update transform
        current_transform = result.registration.transform.clone();

        // Scale up transform for next level
        if level > 0 {
            current_transform[[0, 2]] *= 2.0;
            current_transform[[1, 2]] *= 2.0;
        }

        // Keep track of best result
        if result.final_matches > best_inliers {
            best_inliers = result.final_matches;
            best_result = Some(result);
        }
    }

    best_result
        .ok_or_else(|| VisionError::OperationError("Multi-scale registration failed".to_string()))
}

/// Build image pyramid
fn build_pyramid(image: &DynamicImage, levels: usize) -> Vec<DynamicImage> {
    let mut pyramid = vec![image.clone()];

    for _ in 1..levels {
        let prev = &pyramid[pyramid.len() - 1];
        let (width, height) = (prev.width(), prev.height());

        if width < 16 || height < 16 {
            break;
        }

        // Simple downsampling by factor of 2
        let new_width = width / 2;
        let new_height = height / 2;

        let downsampled =
            prev.resize_exact(new_width, new_height, image::imageops::FilterType::Lanczos3);

        pyramid.push(downsampled);
    }

    pyramid
}

/// Template matching based registration
pub fn template_register(
    reference: &DynamicImage,
    template: &DynamicImage,
    search_region: Option<(u32, u32, u32, u32)>, // (x, y, width, height)
) -> Result<RegistrationResult> {
    let ref_gray = reference.to_luma8();
    let template_gray = template.to_luma8();

    let (ref_width, ref_height) = ref_gray.dimensions();
    let (template_width, template_height) = template_gray.dimensions();

    // Define search region
    let (search_x, search_y, search_width, search_height) = search_region.unwrap_or((
        0,
        0,
        ref_width - template_width,
        ref_height - template_height,
    ));

    let mut best_match = (0, 0);
    let mut best_score = f64::NEG_INFINITY;

    // Search for best match using normalized cross-correlation
    for y in search_y..(search_y + search_height) {
        for x in search_x..(search_x + search_width) {
            if x + template_width <= ref_width && y + template_height <= ref_height {
                let score = compute_ncc(&ref_gray, &template_gray, x, y);

                if score > best_score {
                    best_score = score;
                    best_match = (x, y);
                }
            }
        }
    }

    // Create translation transform
    let mut transform = identity_transform();
    transform[[0, 2]] = best_match.0 as f64;
    transform[[1, 2]] = best_match.1 as f64;

    Ok(RegistrationResult {
        transform,
        final_cost: -best_score, // Negative because higher NCC is better
        iterations: 1,
        converged: true,
        inliers: vec![0], // Single match
    })
}

/// Compute normalized cross-correlation
fn compute_ncc(reference: &GrayImage, template: &GrayImage, offset_x: u32, offset_y: u32) -> f64 {
    let (template_width, template_height) = template.dimensions();

    let mut ref_sum = 0.0;
    let mut template_sum = 0.0;
    let mut ref_sq_sum = 0.0;
    let mut template_sq_sum = 0.0;
    let mut cross_sum = 0.0;
    let n = (template_width * template_height) as f64;

    // Compute statistics
    for y in 0..template_height {
        for x in 0..template_width {
            let ref_val = reference.get_pixel(offset_x + x, offset_y + y)[0] as f64;
            let template_val = template.get_pixel(x, y)[0] as f64;

            ref_sum += ref_val;
            template_sum += template_val;
            ref_sq_sum += ref_val * ref_val;
            template_sq_sum += template_val * template_val;
            cross_sum += ref_val * template_val;
        }
    }

    // Compute normalized cross-correlation
    let ref_mean = ref_sum / n;
    let template_mean = template_sum / n;

    let numerator = cross_sum - n * ref_mean * template_mean;
    let ref_var = ref_sq_sum - n * ref_mean * ref_mean;
    let template_var = template_sq_sum - n * template_mean * template_mean;

    let denominator = (ref_var * template_var).sqrt();

    if denominator > 1e-10 {
        numerator / denominator
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Luma};

    fn create_test_image(width: u32, height: u32, pattern: u8) -> DynamicImage {
        let img = ImageBuffer::from_fn(width, height, |x, y| {
            Luma([((x + y + pattern as u32) % 256) as u8])
        });
        DynamicImage::ImageLuma8(img)
    }

    #[test]
    fn test_feature_registration_config() {
        let config = FeatureRegistrationConfig::default();
        assert_eq!(config.transform_type, TransformType::Affine);
        assert!(config.detector_params.use_orb);
    }

    #[test]
    fn test_keypoint_creation() {
        let kp = Keypoint {
            x: 10.0,
            y: 20.0,
            scale: 1.5,
            angle: 0.5,
            response: 0.8,
        };

        assert_eq!(kp.x, 10.0);
        assert_eq!(kp.y, 20.0);
    }

    #[test]
    fn test_simple_descriptor_generation() {
        let image = create_test_image(100, 100, 0).to_luma8();
        let keypoints = vec![Keypoint {
            x: 50.0,
            y: 50.0,
            scale: 1.0,
            angle: 0.0,
            response: 1.0,
        }];

        let result = generate_simple_descriptors(&image, &keypoints);
        assert!(result.is_ok());

        let descriptors = result.unwrap();
        assert_eq!(descriptors.len(), 1);
        assert!(!descriptors[0].is_empty());
    }

    #[test]
    fn test_template_registration() {
        let reference = create_test_image(100, 100, 0);
        let template = create_test_image(20, 20, 0);

        let result = template_register(&reference, &template, None);
        assert!(result.is_ok());

        let reg_result = result.unwrap();
        assert!(reg_result.converged);
    }

    #[test]
    fn test_ncc_computation() {
        let ref_img = create_test_image(50, 50, 0).to_luma8();
        let template = create_test_image(10, 10, 0).to_luma8();

        let ncc = compute_ncc(&ref_img, &template, 0, 0);
        assert!(ncc >= -1.0 && ncc <= 1.0);
    }

    #[test]
    fn test_pyramid_building() {
        let image = create_test_image(64, 64, 0);
        let pyramid = build_pyramid(&image, 3);

        assert_eq!(pyramid.len(), 3);
        assert_eq!(pyramid[0].width(), 64);
        assert_eq!(pyramid[1].width(), 32);
        assert_eq!(pyramid[2].width(), 16);
    }
}
