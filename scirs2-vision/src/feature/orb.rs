//! ORB (Oriented FAST and Rotated BRIEF) feature detector and descriptor
//!
//! This module implements the ORB algorithm for feature detection and description.
//! ORB provides a fast binary descriptor that is rotation invariant.

use crate::error::Result;
use crate::feature::{image_to_array, KeyPoint};
use image::DynamicImage;
use ndarray::Array2;
// Using this in several places in the code
// PI is used in compute_orientation but this approach uses consts directly

/// Configuration for ORB detector
#[derive(Debug, Clone)]
pub struct OrbConfig {
    /// Number of features to detect
    pub num_features: usize,
    /// Scale factor between levels
    pub scale_factor: f32,
    /// Number of pyramid levels
    pub num_levels: usize,
    /// FAST detector threshold
    pub fast_threshold: u8,
    /// Harris score or FAST score
    pub use_harris_detector: bool,
    /// Patch size for BRIEF descriptor
    pub patch_size: usize,
}

impl Default for OrbConfig {
    fn default() -> Self {
        Self {
            num_features: 500,
            scale_factor: 1.2,
            num_levels: 8,
            fast_threshold: 20,
            use_harris_detector: true,
            patch_size: 31,
        }
    }
}

/// ORB descriptor (binary descriptor)
#[derive(Debug, Clone)]
pub struct OrbDescriptor {
    /// Associated keypoint
    pub keypoint: KeyPoint,
    /// Binary descriptor (256 bits stored as u32 array)
    pub descriptor: Vec<u32>,
}

/// Detect ORB features and compute descriptors
///
/// # Arguments
///
/// * `img` - Input image
/// * `config` - ORB configuration
///
/// # Returns
///
/// * Result containing vector of ORB descriptors
pub fn detect_and_compute_orb(
    img: &DynamicImage,
    config: &OrbConfig,
) -> Result<Vec<OrbDescriptor>> {
    let array = image_to_array(img)?;
    let (_height, _width) = array.dim();

    // Create image pyramid
    let pyramid = create_pyramid(&array, config.num_levels, config.scale_factor);

    // Detect keypoints at each level
    let mut all_keypoints = Vec::new();
    for (level, level_image) in pyramid.iter().enumerate() {
        let mut keypoints = detect_fast_keypoints(level_image, config.fast_threshold)?;

        // Apply non-maximum suppression
        if config.use_harris_detector {
            keypoints = refine_with_harris(level_image, keypoints)?;
        }

        // Scale keypoints to original image coordinates
        let scale = config.scale_factor.powi(level as i32);
        for kp in &mut keypoints {
            kp.x *= scale;
            kp.y *= scale;
            kp.scale = scale;
        }

        all_keypoints.extend(keypoints);
    }

    // Sort keypoints by response and select top features
    all_keypoints.sort_by(|a, b| {
        b.response
            .partial_cmp(&a.response)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    if all_keypoints.len() > config.num_features {
        all_keypoints.truncate(config.num_features);
    }

    // Compute orientation for each keypoint
    for kp in &mut all_keypoints {
        kp.orientation = compute_orientation(&array, kp)?;
    }

    // Compute BRIEF descriptors
    let mut descriptors = Vec::new();
    for kp in all_keypoints {
        if let Ok(descriptor) = compute_brief_descriptor(&array, &kp, config.patch_size) {
            descriptors.push(OrbDescriptor {
                keypoint: kp,
                descriptor,
            });
        }
    }

    Ok(descriptors)
}

/// Create image pyramid for multi-scale detection
fn create_pyramid(image: &Array2<f32>, num_levels: usize, scale_factor: f32) -> Vec<Array2<f32>> {
    let mut pyramid = vec![image.clone()];

    for _level in 1..num_levels {
        let prev_level = pyramid.last().unwrap();
        let (prev_h, prev_w) = prev_level.dim();

        let new_h = ((prev_h as f32) / scale_factor).round() as usize;
        let new_w = ((prev_w as f32) / scale_factor).round() as usize;

        // Simple downsampling - in practice, use proper image resize
        let mut new_level = Array2::zeros((new_h, new_w));
        for y in 0..new_h {
            for x in 0..new_w {
                let src_y = ((y as f32) * scale_factor).round() as usize;
                let src_x = ((x as f32) * scale_factor).round() as usize;

                if src_y < prev_h && src_x < prev_w {
                    new_level[[y, x]] = prev_level[[src_y, src_x]];
                }
            }
        }

        pyramid.push(new_level);
    }

    pyramid
}

/// FAST keypoint detection
fn detect_fast_keypoints(image: &Array2<f32>, threshold: u8) -> Result<Vec<KeyPoint>> {
    let (height, width) = image.dim();
    let mut keypoints = Vec::new();

    // FAST detector uses a circle of radius 3 pixels
    let fast_radius = 3;

    // Circular pattern for FAST detection (Bresenham circle)
    let circle_x = [0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1];
    let circle_y = [3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1, 0, 1, 2, 3];

    for y in fast_radius..(height as isize - fast_radius) {
        for x in fast_radius..(width as isize - fast_radius) {
            let center_val = image[[y as usize, x as usize]];

            // Count consecutive pixels that are brighter or darker
            let mut consecutive_brighter = 0;
            let mut consecutive_darker = 0;
            let mut max_consecutive = 0;

            for i in 0..32 {
                // Check twice around the circle for wrap-around
                let idx = i % 16;
                let px = x + circle_x[idx];
                let py = y + circle_y[idx];

                let pixel_val = image[[py as usize, px as usize]];
                let diff = pixel_val - center_val;

                if diff > threshold as f32 {
                    consecutive_brighter += 1;
                    consecutive_darker = 0;
                } else if diff < -(threshold as f32) {
                    consecutive_darker += 1;
                    consecutive_brighter = 0;
                } else {
                    consecutive_brighter = 0;
                    consecutive_darker = 0;
                }

                max_consecutive = max_consecutive
                    .max(consecutive_brighter)
                    .max(consecutive_darker);
            }

            // FAST requires at least 12 consecutive pixels
            if max_consecutive >= 12 {
                keypoints.push(KeyPoint {
                    x: x as f32,
                    y: y as f32,
                    scale: 1.0,
                    orientation: 0.0,
                    response: max_consecutive as f32,
                });
            }
        }
    }

    Ok(keypoints)
}

/// Refine keypoints using Harris corner measure
fn refine_with_harris(image: &Array2<f32>, keypoints: Vec<KeyPoint>) -> Result<Vec<KeyPoint>> {
    let (height, width) = image.dim();
    let mut refined = Vec::new();

    // Compute gradients
    let mut dx = Array2::zeros((height, width));
    let mut dy = Array2::zeros((height, width));

    for y in 1..(height - 1) {
        for x in 1..(width - 1) {
            dx[[y, x]] = image[[y, x + 1]] - image[[y, x - 1]];
            dy[[y, x]] = image[[y + 1, x]] - image[[y - 1, x]];
        }
    }

    // For each keypoint, compute Harris response
    for mut kp in keypoints {
        let x = kp.x as usize;
        let y = kp.y as usize;

        // Skip border points
        if x < 3 || x >= width - 3 || y < 3 || y >= height - 3 {
            continue;
        }

        // Compute structure tensor in 7x7 window
        let mut xx = 0.0;
        let mut yy = 0.0;
        let mut xy = 0.0;

        for win_y in -3..=3 {
            for win_x in -3..=3 {
                let py = (y as isize + win_y) as usize;
                let px = (x as isize + win_x) as usize;

                let ix = dx[[py, px]];
                let iy = dy[[py, px]];

                xx += ix * ix;
                yy += iy * iy;
                xy += ix * iy;
            }
        }

        // Harris corner response
        let det = xx * yy - xy * xy;
        let trace = xx + yy;
        let harris_response = det - 0.04 * trace * trace;

        kp.response = harris_response;
        refined.push(kp);
    }

    Ok(refined)
}

/// Compute orientation for a keypoint using intensity centroid
fn compute_orientation(image: &Array2<f32>, keypoint: &KeyPoint) -> Result<f32> {
    let x = keypoint.x as usize;
    let y = keypoint.y as usize;
    let (height, width) = image.dim();

    // Use a circular patch of radius 15
    let radius = 15;

    // Check bounds
    if x < radius || x >= width - radius || y < radius || y >= height - radius {
        return Ok(0.0);
    }

    let mut m01 = 0.0; // First moment in y
    let mut m10 = 0.0; // First moment in x

    for dy in -(radius as isize)..=(radius as isize) {
        for dx in -(radius as isize)..=(radius as isize) {
            // Check if point is within circle
            if dx * dx + dy * dy > (radius * radius) as isize {
                continue;
            }

            let px = (x as isize + dx) as usize;
            let py = (y as isize + dy) as usize;

            let intensity = image[[py, px]];
            m01 += (dy as f32) * intensity;
            m10 += (dx as f32) * intensity;
        }
    }

    // Compute orientation
    Ok(m01.atan2(m10))
}

/// Compute BRIEF descriptor with rotation
fn compute_brief_descriptor(
    image: &Array2<f32>,
    keypoint: &KeyPoint,
    patch_size: usize,
) -> Result<Vec<u32>> {
    let (height, width) = image.dim();
    let x = keypoint.x as usize;
    let y = keypoint.y as usize;

    // Check bounds
    let half_patch = patch_size / 2;
    if x < half_patch || x >= width - half_patch || y < half_patch || y >= height - half_patch {
        return Err(crate::error::VisionError::InvalidParameter(
            "Keypoint too close to image border".to_string(),
        ));
    }

    // Get rotation matrix
    let cos_angle = keypoint.orientation.cos();
    let sin_angle = keypoint.orientation.sin();

    // BRIEF descriptor with 256 bits (8 u32 values)
    let mut descriptor = vec![0u32; 8];

    // Use pre-defined sampling pattern for BRIEF
    // In practice, these should be loaded from a file or generated with a specific pattern
    let pattern = generate_brief_pattern();

    for (i, &(dx1, dy1, dx2, dy2)) in pattern.iter().enumerate() {
        // Rotate the sample points
        let rx1 = (cos_angle * dx1 as f32 - sin_angle * dy1 as f32).round() as isize;
        let ry1 = (sin_angle * dx1 as f32 + cos_angle * dy1 as f32).round() as isize;
        let rx2 = (cos_angle * dx2 as f32 - sin_angle * dy2 as f32).round() as isize;
        let ry2 = (sin_angle * dx2 as f32 + cos_angle * dy2 as f32).round() as isize;

        // Sample the image
        let px1 = (x as isize + rx1) as usize;
        let py1 = (y as isize + ry1) as usize;
        let px2 = (x as isize + rx2) as usize;
        let py2 = (y as isize + ry2) as usize;

        // Compare intensities
        if px1 < width
            && py1 < height
            && px2 < width
            && py2 < height
            && image[[py1, px1]] < image[[py2, px2]]
        {
            let word_idx = i / 32;
            let bit_idx = i % 32;
            descriptor[word_idx] |= 1 << bit_idx;
        }
    }

    Ok(descriptor)
}

/// Generate BRIEF sampling pattern
fn generate_brief_pattern() -> Vec<(isize, isize, isize, isize)> {
    // In practice, use a pre-computed pattern
    // This is a simplified random pattern
    let mut pattern = Vec::new();
    let max_offset = 12;

    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    let mut rng = StdRng::seed_from_u64(42); // Fixed seed for reproducibility

    for _ in 0..256 {
        let x1 = rng.random_range(-{ max_offset }..=max_offset) as isize;
        let y1 = rng.random_range(-{ max_offset }..=max_offset) as isize;
        let x2 = rng.random_range(-{ max_offset }..=max_offset) as isize;
        let y2 = rng.random_range(-{ max_offset }..=max_offset) as isize;
        pattern.push((x1, y1, x2, y2));
    }

    pattern
}

/// Match ORB descriptors using Hamming distance
///
/// # Arguments
///
/// * `descriptors1` - First set of descriptors
/// * `descriptors2` - Second set of descriptors
/// * `threshold` - Distance threshold for matching
///
/// # Returns
///
/// * Vector of matched descriptor indices
pub fn match_orb_descriptors(
    descriptors1: &[OrbDescriptor],
    descriptors2: &[OrbDescriptor],
    threshold: u32,
) -> Vec<(usize, usize, u32)> {
    let mut matches = Vec::new();

    for (i, desc1) in descriptors1.iter().enumerate() {
        let mut best_distance = u32::MAX;
        let mut best_index = 0;
        let mut second_best_distance = u32::MAX;

        for (j, desc2) in descriptors2.iter().enumerate() {
            let distance = hamming_distance(&desc1.descriptor, &desc2.descriptor);

            if distance < best_distance {
                second_best_distance = best_distance;
                best_distance = distance;
                best_index = j;
            } else if distance < second_best_distance {
                second_best_distance = distance;
            }
        }

        // Apply ratio test
        if best_distance < threshold && best_distance < (second_best_distance * 3 / 4) {
            matches.push((i, best_index, best_distance));
        }
    }

    matches
}

/// Calculate Hamming distance between binary descriptors
fn hamming_distance(desc1: &[u32], desc2: &[u32]) -> u32 {
    let mut distance = 0;

    for (&d1, &d2) in desc1.iter().zip(desc2.iter()) {
        distance += (d1 ^ d2).count_ones();
    }

    distance
}
