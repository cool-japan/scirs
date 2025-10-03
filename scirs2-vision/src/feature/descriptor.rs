//! Feature descriptor module
//!
//! This module provides functionality for describing detected features
//! with descriptors like SIFT, BRIEF, and ORB.

use crate::error::Result;
use crate::feature::image_to_array;
use image::DynamicImage;
use scirs2_core::ndarray::Array2;
use std::f32::consts::PI;

/// Feature point with position, scale, and orientation
#[derive(Debug, Clone, Default)]
pub struct KeyPoint {
    /// X-coordinate
    pub x: f32,
    /// Y-coordinate
    pub y: f32,
    /// Scale
    pub scale: f32,
    /// Orientation in radians
    pub orientation: f32,
    /// Response strength
    pub response: f32,
}

/// Feature descriptor
#[derive(Debug, Clone)]
pub struct Descriptor {
    /// Associated keypoint
    pub keypoint: KeyPoint,
    /// Feature vector
    pub vector: Vec<f32>,
}

/// Detect features and compute simplified SIFT-like descriptors
///
/// # Arguments
///
/// * `img` - Input image
/// * `max_features` - Maximum number of features to detect
/// * `threshold` - Detection threshold
///
/// # Returns
///
/// * Result containing a vector of descriptors
#[allow(dead_code)]
pub fn detect_and_compute(
    img: &DynamicImage,
    max_features: usize,
    threshold: f32,
) -> Result<Vec<Descriptor>> {
    // Convert to grayscale
    let gray = img.to_luma8();
    let _width_height = gray.dimensions();

    // Convert to array for easier processing
    let array = image_to_array(img)?;
    let (height, width) = array.dim();

    // Step 1: Compute gradient magnitude and orientation
    let mut magnitude = Array2::zeros((height, width));
    let mut orientation = Array2::zeros((height, width));

    for y in 1..(height - 1) {
        for x in 1..(width - 1) {
            // Compute gradients using simple finite differences
            let dx = array[[y, x + 1]] - array[[y, x - 1]];
            let dy = array[[y + 1, x]] - array[[y - 1, x]];

            // Gradient magnitude and orientation
            magnitude[[y, x]] = (dx * dx + dy * dy).sqrt();
            orientation[[y, x]] = dy.atan2(dx);
        }
    }

    // Step 2: Find local maxima in the magnitude image
    let mut keypoints = Vec::new();

    for y in 2..(height - 2) {
        for x in 2..(width - 2) {
            let current_mag = magnitude[[y, x]];

            // Skip weak gradients
            if current_mag < threshold {
                continue;
            }

            // Check if local maximum in a 3x3 neighborhood
            let mut is_max = true;
            'neighborhood: for ny in (y - 1)..=(y + 1) {
                for nx in (x - 1)..=(x + 1) {
                    if (ny != y || nx != x) && magnitude[[ny, nx]] >= current_mag {
                        is_max = false;
                        break 'neighborhood;
                    }
                }
            }

            if is_max {
                keypoints.push(KeyPoint {
                    x: x as f32,
                    y: y as f32,
                    scale: 1.0,
                    orientation: orientation[[y, x]],
                    response: current_mag,
                });
            }
        }
    }

    // Sort keypoints by response and take the top max_features
    keypoints.sort_by(|a, b| {
        b.response
            .partial_cmp(&a.response)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    if keypoints.len() > max_features {
        keypoints.truncate(max_features);
    }

    // Step 3: Compute descriptors for each keypoint
    let mut descriptors = Vec::new();

    for kp in keypoints {
        // Skip keypoints too close to the border for descriptor computation
        if kp.x < 8.0 || kp.x >= (width as f32 - 8.0) || kp.y < 8.0 || kp.y >= (height as f32 - 8.0)
        {
            continue;
        }

        // Compute simplified SIFT-like descriptor
        let descriptor = compute_descriptor(&array, &magnitude, &orientation, &kp)?;

        descriptors.push(Descriptor {
            keypoint: kp,
            vector: descriptor,
        });
    }

    Ok(descriptors)
}

/// Compute a simplified SIFT-like descriptor for a keypoint
///
/// # Arguments
///
/// * `image` - Image array
/// * `magnitude` - Gradient magnitude array
/// * `orientation` - Gradient orientation array
/// * `keypoint` - Keypoint for which to compute the descriptor
///
/// # Returns
///
/// * Result containing a descriptor vector
#[allow(dead_code)]
fn compute_descriptor(
    image: &Array2<f32>,
    magnitude: &Array2<f32>,
    orientation: &Array2<f32>,
    keypoint: &KeyPoint,
) -> Result<Vec<f32>> {
    let (height, width) = image.dim();

    // Simplified descriptor: 4x4 spatial bins x 8 orientation bins = 128 dimensions
    let mut descriptor = vec![0.0; 128];

    // Calculate descriptor parameters
    let cos_angle = keypoint.orientation.cos();
    let sin_angle = keypoint.orientation.sin();

    // Precompute the weight sigma for the descriptor
    let sigma = 4.0; // Descriptor radius in pixels

    // Precompute histogram bin mappings
    let num_spatial_bins = 4;
    let num_orientation_bins = 8;
    let orientation_bin_width = 2.0 * PI / num_orientation_bins as f32;

    // For each position in the 16x16 patch around the keypoint
    for i in -8..8 {
        for j in -8..8 {
            // Rotate the patch coordinate
            let rotated_i = (cos_angle * i as f32 - sin_angle * j as f32).round() as isize;
            let rotated_j = (sin_angle * i as f32 + cos_angle * j as f32).round() as isize;

            // Calculate image coordinates
            let img_y = keypoint.y as isize + rotated_i;
            let img_x = keypoint.x as isize + rotated_j;

            // Skip if outside image bounds
            if img_y < 0 || img_y >= height as isize || img_x < 0 || img_x >= width as isize {
                continue;
            }

            // Get the gradient magnitude and orientation at this point
            let mag = magnitude[[img_y as usize, img_x as usize]];
            let ori = orientation[[img_y as usize, img_x as usize]];

            // Calculate spatial bin centers
            let bin_i = ((i as f32 + 8.0) * num_spatial_bins as f32 / 16.0).floor() as usize;
            let bin_j = ((j as f32 + 8.0) * num_spatial_bins as f32 / 16.0).floor() as usize;

            // Calculate the spatial bin indices, clamping to valid range
            let bin_i = bin_i.min(num_spatial_bins - 1);
            let bin_j = bin_j.min(num_spatial_bins - 1);

            // Calculate the orientation bin
            // Normalize orientation relative to keypoint orientation
            let rel_ori = (ori - keypoint.orientation + 2.0 * PI) % (2.0 * PI);
            let ori_bin = (rel_ori / orientation_bin_width).floor() as usize % num_orientation_bins;

            // Gaussian weight by distance from keypoint
            let weight =
                (-(i as f32 * i as f32 + j as f32 * j as f32) / (2.0 * sigma * sigma)).exp();

            // Add weighted magnitude to descriptor histogram
            let idx = (bin_i * num_spatial_bins + bin_j) * num_orientation_bins + ori_bin;
            descriptor[idx] += mag * weight;
        }
    }

    // Normalize the descriptor to unit length to achieve invariance to contrast changes
    let mut norm = 0.0;
    for val in &descriptor {
        norm += val * val;
    }
    norm = norm.sqrt();

    // Avoid division by zero
    if norm > 1e-6 {
        for val in &mut descriptor {
            *val /= norm;
        }
    }

    // Threshold large values to reduce the influence of large gradient magnitudes
    for val in &mut descriptor {
        *val = (*val).min(0.2);
    }

    // Normalize again after thresholding
    let mut norm = 0.0;
    for val in &descriptor {
        norm += val * val;
    }
    norm = norm.sqrt();

    // Avoid division by zero
    if norm > 1e-6 {
        for val in &mut descriptor {
            *val /= norm;
        }
    }

    Ok(descriptor)
}

/// Match descriptors using Euclidean distance
///
/// # Arguments
///
/// * `descriptors1` - First set of descriptors
/// * `descriptors2` - Second set of descriptors
/// * `threshold` - Distance threshold for matching
///
/// # Returns
///
/// * Vector of matched descriptor indices (idx1, idx2, distance)
#[allow(dead_code)]
pub fn match_descriptors(
    descriptors1: &[Descriptor],
    descriptors2: &[Descriptor],
    threshold: f32,
) -> Vec<(usize, usize, f32)> {
    let mut matches = Vec::new();

    for (i, desc1) in descriptors1.iter().enumerate() {
        let mut best_distance = f32::MAX;
        let mut best_index = 0;
        let mut second_best_distance = f32::MAX;

        // Find the closest and second closest matches
        for (j, desc2) in descriptors2.iter().enumerate() {
            let distance = euclidean_distance(&desc1.vector, &desc2.vector);

            if distance < best_distance {
                second_best_distance = best_distance;
                best_distance = distance;
                best_index = j;
            } else if distance < second_best_distance {
                second_best_distance = distance;
            }
        }

        // Apply Lowe's ratio test to filter ambiguous matches
        if best_distance < threshold && best_distance < 0.7 * second_best_distance {
            matches.push((i, best_index, best_distance));
        }
    }

    // Sort matches by distance
    matches.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

    matches
}

/// Calculate Euclidean distance between two descriptor vectors
///
/// # Arguments
///
/// * `vec1` - First descriptor vector
/// * `vec2` - Second descriptor vector
///
/// # Returns
///
/// * Euclidean distance
#[allow(dead_code)]
fn euclidean_distance(vec1: &[f32], vec2: &[f32]) -> f32 {
    let mut sum_sq = 0.0;

    for i in 0..vec1.len().min(vec2.len()) {
        let diff = vec1[i] - vec2[i];
        sum_sq += diff * diff;
    }

    sum_sq.sqrt()
}
