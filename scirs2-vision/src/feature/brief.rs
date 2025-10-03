//! BRIEF (Binary Robust Independent Elementary Features) descriptor
//!
//! This module implements the BRIEF binary descriptor for efficient feature matching.

use crate::error::Result;
use crate::feature::{image_to_array, KeyPoint};
use crate::preprocessing::gaussian_blur;
use image::DynamicImage;
use scirs2_core::ndarray::Array2;

/// Configuration for BRIEF descriptor
#[derive(Debug, Clone)]
pub struct BriefConfig {
    /// Descriptor length (128, 256, or 512 bits)
    pub descriptor_size: usize,
    /// Patch size for descriptor computation
    pub patch_size: usize,
    /// Apply Gaussian smoothing before descriptor computation
    pub use_smoothing: bool,
    /// Sigma for Gaussian smoothing
    pub smoothing_sigma: f32,
}

impl Default for BriefConfig {
    fn default() -> Self {
        Self {
            descriptor_size: 256,
            patch_size: 48,
            use_smoothing: true,
            smoothing_sigma: 2.0,
        }
    }
}

/// BRIEF descriptor
#[derive(Debug, Clone)]
pub struct BriefDescriptor {
    /// Associated keypoint
    pub keypoint: KeyPoint,
    /// Binary descriptor stored as array of u32
    pub descriptor: Vec<u32>,
}

/// Compute BRIEF descriptors for given keypoints
///
/// # Arguments
///
/// * `img` - Input image
/// * `keypoints` - Vector of keypoints
/// * `config` - BRIEF configuration
///
/// # Returns
///
/// * Result containing vector of BRIEF descriptors
#[allow(dead_code)]
pub fn compute_brief_descriptors(
    img: &DynamicImage,
    keypoints: Vec<KeyPoint>,
    config: &BriefConfig,
) -> Result<Vec<BriefDescriptor>> {
    // Convert to grayscale array
    let mut array = image_to_array(img)?;

    // Apply Gaussian smoothing if requested
    if config.use_smoothing {
        // Create a temporary DynamicImage for blurring
        let temp_img = crate::feature::array_to_image(&array)?;
        let temp_dynamic = image::DynamicImage::ImageLuma8(temp_img);
        let blurred = gaussian_blur(&temp_dynamic, config.smoothing_sigma)?;
        // Convert back to array
        array = image_to_array(&blurred)?;
    }

    // Generate sampling pattern
    let pattern = generate_test_pattern(config.descriptor_size, config.patch_size);

    // Compute descriptors for each keypoint
    let mut descriptors = Vec::new();

    for keypoint in keypoints {
        if let Ok(descriptor) =
            compute_descriptor_at_keypoint(&array, &keypoint, &pattern, config.patch_size)
        {
            descriptors.push(BriefDescriptor {
                keypoint,
                descriptor,
            });
        }
    }

    Ok(descriptors)
}

/// Compute BRIEF descriptor at a single keypoint
#[allow(dead_code)]
fn compute_descriptor_at_keypoint(
    image: &Array2<f32>,
    keypoint: &KeyPoint,
    pattern: &[(isize, isize, isize, isize)],
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

    // Compute descriptor bits
    let num_words = pattern.len().div_ceil(32);
    let mut descriptor = vec![0u32; num_words];

    for (i, &(dx1, dy1, dx2, dy2)) in pattern.iter().enumerate() {
        // Apply keypoint rotation if orientation is available
        let (rx1, ry1, rx2, ry2) = if keypoint.orientation != 0.0 {
            let cos_angle = keypoint.orientation.cos();
            let sin_angle = keypoint.orientation.sin();

            let rx1 = (cos_angle * dx1 as f32 - sin_angle * dy1 as f32).round() as isize;
            let ry1 = (sin_angle * dx1 as f32 + cos_angle * dy1 as f32).round() as isize;
            let rx2 = (cos_angle * dx2 as f32 - sin_angle * dy2 as f32).round() as isize;
            let ry2 = (sin_angle * dx2 as f32 + cos_angle * dy2 as f32).round() as isize;

            (rx1, ry1, rx2, ry2)
        } else {
            (dx1, dy1, dx2, dy2)
        };

        // Sample the image
        let px1 = (x as isize + rx1) as usize;
        let py1 = (y as isize + ry1) as usize;
        let px2 = (x as isize + rx2) as usize;
        let py2 = (y as isize + ry2) as usize;

        // Compare intensities and set descriptor bit
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

/// Generate test pattern for BRIEF descriptor
#[allow(dead_code)]
fn generate_test_pattern(
    descriptor_size: usize,
    patch_size: usize,
) -> Vec<(isize, isize, isize, isize)> {
    use scirs2_core::random::rngs::StdRng;
    use scirs2_core::random::{Rng, SeedableRng};

    let mut pattern = Vec::new();

    // Create RNG with fixed seed for reproducibility
    let mut rng = StdRng::seed_from_u64(12345);

    // Use Gaussian distribution for test locations (as in original BRIEF paper)
    let half_patch = (patch_size / 2) as f32;
    let sigma = half_patch / 5.0; // Standard deviation

    for _ in 0..descriptor_size {
        // Generate test locations using Gaussian-like distribution
        // Using Box-Muller transform to generate Gaussian-distributed values
        let u1: f32 = rng.random();
        let u2: f32 = rng.random();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;

        let z0 = r * theta.cos() * sigma;
        let z1 = r * theta.sin() * sigma;

        let u3: f32 = rng.random();
        let u4: f32 = rng.random();
        let r2 = (-2.0 * u3.ln()).sqrt();
        let theta2 = 2.0 * std::f32::consts::PI * u4;

        let z2 = r2 * theta2.cos() * sigma;
        let z3 = r2 * theta2.sin() * sigma;

        let x1 = (z0.round() as isize).clamp(-(half_patch as isize), half_patch as isize - 1);
        let y1 = (z1.round() as isize).clamp(-(half_patch as isize), half_patch as isize - 1);
        let x2 = (z2.round() as isize).clamp(-(half_patch as isize), half_patch as isize - 1);
        let y2 = (z3.round() as isize).clamp(-(half_patch as isize), half_patch as isize - 1);

        pattern.push((x1, y1, x2, y2));
    }

    pattern
}

/// Match BRIEF descriptors using Hamming distance
///
/// # Arguments
///
/// * `descriptors1` - First set of descriptors
/// * `descriptors2` - Second set of descriptors
/// * `max_distance` - Maximum Hamming distance for a valid match
///
/// # Returns
///
/// * Vector of matched descriptor indices with distances
#[allow(dead_code)]
pub fn match_brief_descriptors(
    descriptors1: &[BriefDescriptor],
    descriptors2: &[BriefDescriptor],
    max_distance: u32,
) -> Vec<(usize, usize, u32)> {
    let mut matches = Vec::new();

    for (i, desc1) in descriptors1.iter().enumerate() {
        let mut best_distance = u32::MAX;
        let mut best_index = 0;
        let mut second_best_distance = u32::MAX;

        for (j, desc2) in descriptors2.iter().enumerate() {
            let _distance = hamming_distance(&desc1.descriptor, &desc2.descriptor);

            if _distance < best_distance {
                second_best_distance = best_distance;
                best_distance = _distance;
                best_index = j;
            } else if _distance < second_best_distance {
                second_best_distance = _distance;
            }
        }

        // Apply _distance threshold and ratio test
        if best_distance < max_distance && best_distance < (second_best_distance * 7 / 10) {
            matches.push((i, best_index, best_distance));
        }
    }

    // Sort matches by _distance
    matches.sort_by_key(|&(_, _, dist)| dist);

    matches
}

/// Calculate Hamming distance between binary descriptors
#[allow(dead_code)]
pub fn hamming_distance(desc1: &[u32], desc2: &[u32]) -> u32 {
    desc1
        .iter()
        .zip(desc2.iter())
        .map(|(&d1, &d2)| (d1 ^ d2).count_ones())
        .sum()
}

/// Convert Hamming distance to normalized similarity score
#[allow(dead_code)]
pub fn hamming_to_similarity(_distance: u32, descriptorbits: usize) -> f32 {
    1.0 - (_distance as f32) / (descriptorbits as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_brief_pattern_generation() {
        let pattern = generate_test_pattern(256, 48);
        assert_eq!(pattern.len(), 256);

        // Check that all test locations are within patch bounds
        let half_patch = 24;
        for &(x1, y1, x2, y2) in &pattern {
            assert!(x1.abs() <= half_patch);
            assert!(y1.abs() <= half_patch);
            assert!(x2.abs() <= half_patch);
            assert!(y2.abs() <= half_patch);
        }
    }

    #[test]
    fn test_hamming_distance() {
        let desc1 = vec![0b11001100_11001100_11001100_11001100u32];
        let desc2 = vec![0b11001100_11001100_11001100_00000000u32];

        let distance = hamming_distance(&desc1, &desc2);
        assert_eq!(distance, 4); // 4 bits different (11001100 has 4 set bits)
    }

    #[test]
    fn test_similarity_score() {
        let similarity = hamming_to_similarity(0, 256);
        assert_eq!(similarity, 1.0); // Perfect match

        let similarity = hamming_to_similarity(256, 256);
        assert_eq!(similarity, 0.0); // No match

        let similarity = hamming_to_similarity(128, 256);
        assert_eq!(similarity, 0.5); // 50% match
    }
}
