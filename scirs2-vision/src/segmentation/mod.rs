//! Image segmentation module
//!
//! This module provides functionality for segmenting images into regions
//! or partitioning images into meaningful parts.

pub mod mean_shift;
pub mod region_growing;
pub mod slic;
pub mod watershed;

pub use mean_shift::{mean_shift, MeanShiftParams};
pub use region_growing::{
    adaptive_region_growing, region_growing, region_labels_to_color, RegionGrowingParams, SeedPoint,
};
pub use slic::{draw_superpixel_boundaries, slic};
pub use watershed::{
    compute_gradient_magnitude, labels_to_color_image, watershed, watershed_markers,
};

use crate::error::{Result, VisionError};
use crate::feature::image_to_array;
use image::{DynamicImage, GrayImage, ImageBuffer, Luma};
// ndarray::Array2 is imported through image_to_array function

/// Adaptive thresholding method
#[derive(Debug, Clone, Copy)]
pub enum AdaptiveMethod {
    /// Mean of the neighborhood values
    Mean,
    /// Gaussian weighted mean of the neighborhood
    Gaussian,
}

/// Threshold an image to create a binary image
///
/// # Arguments
///
/// * `img` - Input image
/// * `threshold` - Threshold value (0.0 to 1.0)
///
/// # Returns
///
/// * Result containing a binary image
pub fn threshold_binary(img: &DynamicImage, threshold: f32) -> Result<GrayImage> {
    let array = image_to_array(img)?;
    let (height, width) = array.dim();

    let mut binary = ImageBuffer::new(width as u32, height as u32);

    for y in 0..height {
        for x in 0..width {
            let value = if array[[y, x]] >= threshold { 255 } else { 0 };
            binary.put_pixel(x as u32, y as u32, Luma([value]));
        }
    }

    Ok(binary)
}

/// Apply Otsu's automatic thresholding method
///
/// # Arguments
///
/// * `img` - Input image
///
/// # Returns
///
/// * Result containing a binary image and the computed threshold
pub fn otsu_threshold(img: &DynamicImage) -> Result<(GrayImage, f32)> {
    let gray = img.to_luma8();
    let (width, height) = gray.dimensions();
    let total_pixels = (width * height) as usize;

    // Calculate histogram
    let mut histogram = [0; 256];
    for pixel in gray.pixels() {
        histogram[pixel[0] as usize] += 1;
    }

    // Calculate running sum and weighted sum
    let mut sum = 0;
    for (i, &count) in histogram.iter().enumerate() {
        sum += i * count;
    }

    let mut sum_background = 0;
    let mut weight_background = 0;
    let mut max_variance = 0.0;
    let mut threshold = 0;

    for (i, &count) in histogram.iter().enumerate() {
        // Weight is the probability of the background
        weight_background += count;
        if weight_background == 0 {
            continue;
        }

        let weight_foreground = total_pixels - weight_background;
        if weight_foreground == 0 {
            break;
        }

        // Sum is the weighted mean of the background
        sum_background += i * histogram[i];

        // Calculate means
        let mean_background = sum_background as f32 / weight_background as f32;
        let mean_foreground = (sum - sum_background) as f32 / weight_foreground as f32;

        // Calculate between-class variance
        let variance = weight_background as f32
            * weight_foreground as f32
            * (mean_background - mean_foreground).powi(2);

        // Update threshold if variance is higher
        if variance > max_variance {
            max_variance = variance;
            threshold = i;
        }
    }

    // Create binary image using the computed threshold
    let threshold_f32 = threshold as f32 / 255.0;
    let binary = threshold_binary(img, threshold_f32)?;

    Ok((binary, threshold_f32))
}

/// Apply adaptive thresholding
///
/// # Arguments
///
/// * `img` - Input image
/// * `block_size` - Size of the neighborhood for calculating the threshold
/// * `c` - Constant subtracted from the mean or weighted sum
/// * `method` - Thresholding method
///
/// # Returns
///
/// * Result containing a binary image
pub fn adaptive_threshold(
    img: &DynamicImage,
    block_size: usize,
    c: f32,
    method: AdaptiveMethod,
) -> Result<GrayImage> {
    // Check if block size is valid
    if block_size % 2 == 0 || block_size < 3 {
        return Err(VisionError::InvalidParameter(
            "block_size must be odd and at least 3".to_string(),
        ));
    }

    let array = image_to_array(img)?;
    let (height, width) = array.dim();
    let radius = block_size / 2;

    let mut binary = ImageBuffer::new(width as u32, height as u32);

    for y in 0..height {
        for x in 0..width {
            // Define neighborhood bounds with padding at the edges
            let start_y = y.saturating_sub(radius);
            let end_y = (y + radius + 1).min(height);
            let start_x = x.saturating_sub(radius);
            let end_x = (x + radius + 1).min(width);

            // Calculate threshold based on method
            let threshold = match method {
                AdaptiveMethod::Mean => {
                    // Simple mean of neighborhood
                    let mut sum = 0.0;
                    let mut count = 0;

                    for ny in start_y..end_y {
                        for nx in start_x..end_x {
                            sum += array[[ny, nx]];
                            count += 1;
                        }
                    }

                    sum / count as f32 - c
                }
                AdaptiveMethod::Gaussian => {
                    // Gaussian weighted mean
                    let mut weighted_sum = 0.0;
                    let mut weight_sum = 0.0;

                    for ny in start_y..end_y {
                        for nx in start_x..end_x {
                            let dy = (ny as isize - y as isize).pow(2) as f32;
                            let dx = (nx as isize - x as isize).pow(2) as f32;
                            let dist = (dy + dx).sqrt();

                            // Gaussian weight
                            let sigma = radius as f32 / 2.0;
                            let weight = (-dist * dist / (2.0 * sigma * sigma)).exp();

                            weighted_sum += array[[ny, nx]] * weight;
                            weight_sum += weight;
                        }
                    }

                    weighted_sum / weight_sum - c
                }
            };

            // Apply threshold
            let value = if array[[y, x]] > threshold { 255 } else { 0 };
            binary.put_pixel(x as u32, y as u32, Luma([value]));
        }
    }

    Ok(binary)
}

/// Apply connected component labeling
///
/// # Arguments
///
/// * `binary` - Binary input image
///
/// # Returns
///
/// * Result containing a labeled image where each connected component has a unique label
///
/// Type alias for labeled image
pub type LabeledImage = ImageBuffer<Luma<u16>, Vec<u16>>;

/// Find connected components in a binary image using 8-connectivity
///
/// This function implements a two-pass algorithm to identify connected components
/// in a binary image. It assigns a unique label to each connected component and
/// returns both the labeled image and the number of labels found.
///
/// # Arguments
///
/// * `binary` - Input binary image where non-zero pixels are foreground
///
/// # Returns
///
/// * Result containing a tuple with:
///   - Labeled image where each pixel value is the label of its component
///   - Number of labels found (counting from 1)
pub fn connected_components(binary: &GrayImage) -> Result<(LabeledImage, u16)> {
    let (width, height) = binary.dimensions();
    let mut labels: ImageBuffer<Luma<u16>, Vec<u16>> = ImageBuffer::new(width, height);
    let mut label_equiv = vec![0u16; 65536]; // Union-find data structure
    let mut next_label = 1u16;

    // Initialize equivalence array
    // Fill with the index values
    for (i, val) in label_equiv.iter_mut().enumerate() {
        *val = i as u16;
    }

    // First pass: assign labels and record equivalences
    for y in 0..height {
        for x in 0..width {
            // Skip background
            if binary.get_pixel(x, y)[0] == 0 {
                labels.put_pixel(x, y, Luma([0]));
                continue;
            }

            // Check connected neighbors (4-connectivity)
            let mut neighbors = Vec::new();

            if x > 0 && binary.get_pixel(x - 1, y)[0] > 0 {
                neighbors.push(labels.get_pixel(x - 1, y)[0]);
            }

            if y > 0 && binary.get_pixel(x, y - 1)[0] > 0 {
                neighbors.push(labels.get_pixel(x, y - 1)[0]);
            }

            // If no labeled neighbors, create a new label
            if neighbors.is_empty() {
                labels.put_pixel(x, y, Luma([next_label]));
                next_label += 1;

                // Avoid overflow
                if next_label == 0 {
                    return Err(VisionError::OperationError(
                        "Too many components (label overflow)".to_string(),
                    ));
                }
            } else {
                // Find minimum label among neighbors
                let min_label = *neighbors.iter().min().unwrap();
                labels.put_pixel(x, y, Luma([min_label]));

                // Record equivalences
                for &neighbor_label in &neighbors {
                    if neighbor_label != min_label {
                        union(&mut label_equiv, min_label, neighbor_label);
                    }
                }
            }
        }
    }

    // Second pass: replace labels with their equivalence classes
    for y in 0..height {
        for x in 0..width {
            let label = labels.get_pixel(x, y)[0];
            if label > 0 {
                labels.put_pixel(x, y, Luma([find(&label_equiv, label)]));
            }
        }
    }

    // Count unique labels (excluding background)
    let mut unique_labels = std::collections::HashSet::new();
    for y in 0..height {
        for x in 0..width {
            let label = labels.get_pixel(x, y)[0];
            if label > 0 {
                unique_labels.insert(label);
            }
        }
    }

    Ok((labels, unique_labels.len() as u16))
}

// Union-find helper functions
fn find(labels: &[u16], x: u16) -> u16 {
    let mut y = x;
    while y != labels[y as usize] {
        y = labels[y as usize];
    }
    y
}

fn union(labels: &mut [u16], x: u16, y: u16) {
    let root_x = find(labels, x);
    let root_y = find(labels, y);
    if root_x <= root_y {
        labels[root_y as usize] = root_x;
    } else {
        labels[root_x as usize] = root_y;
    }
}
