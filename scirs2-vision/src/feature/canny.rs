//! Canny edge detector implementation
//!
//! Based on the scikit-image implementation of the Canny edge detection algorithm.
//! Reference: Canny, J., A Computational Approach To Edge Detection, IEEE Trans.
//! Pattern Analysis and Machine Intelligence, 8:679-714, 1986
//!
//! This module provides functionality for edge detection in images using the Canny algorithm,
//! which is known for its good detection, localization, and single response properties.

use crate::error::{Result, VisionError};
use image::{DynamicImage, GrayImage};
use ndarray::{Array2, Zip};
use std::collections::VecDeque;
use std::f32::consts::{FRAC_PI_4, PI};

/// Preprocessing mode for edge handling during Gaussian filtering
#[derive(Debug, Clone, Copy)]
pub enum PreprocessMode {
    /// Reflect boundary values
    Reflect,
    /// Use constant value for boundaries
    Constant(f32),
    /// Use nearest boundary value
    Nearest,
    /// Mirror boundary values
    Mirror,
    /// Wrap around boundaries
    Wrap,
}

/// Convert image to array with proper normalization
fn image_to_array_normalized(img: &DynamicImage) -> Result<Array2<f32>> {
    let gray = img.to_luma8();
    let (width, height) = gray.dimensions();

    let mut array = Array2::zeros((height as usize, width as usize));

    for y in 0..height {
        for x in 0..width {
            array[[y as usize, x as usize]] = gray.get_pixel(x, y)[0] as f32 / 255.0;
        }
    }

    Ok(array)
}

/// Convert array to image with proper scaling
fn array_to_binary_image(array: &Array2<bool>) -> Result<GrayImage> {
    let (height, width) = array.dim();
    let mut img = GrayImage::new(width as u32, height as u32);

    for y in 0..height {
        for x in 0..width {
            let value = if array[[y, x]] { 255 } else { 0 };
            img.put_pixel(x as u32, y as u32, image::Luma([value]));
        }
    }

    Ok(img)
}

/// Simple Gaussian kernel generation
fn gaussian_kernel(sigma: f32, size: usize) -> Vec<f32> {
    let mut kernel = vec![0.0; size];
    let center = (size as f32 - 1.0) / 2.0;
    let s = 2.0 * sigma * sigma;
    let mut sum = 0.0;

    for (i, val) in kernel.iter_mut().enumerate() {
        let x = i as f32 - center;
        *val = (-x * x / s).exp();
        sum += *val;
    }

    // Normalize
    for val in &mut kernel {
        *val /= sum;
    }

    kernel
}

/// Apply Gaussian filter to an array
fn gaussian_filter(image: &Array2<f32>, sigma: f32) -> Array2<f32> {
    if sigma <= 0.0 {
        return image.clone();
    }

    let kernel_size = ((6.0 * sigma + 1.0) as usize) | 1; // Make odd
    let kernel = gaussian_kernel(sigma, kernel_size);
    let radius = kernel_size / 2;

    let (height, width) = image.dim();
    let mut temp = Array2::zeros((height, width));
    let mut output = Array2::zeros((height, width));

    // Horizontal pass
    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0;
            let mut weight_sum = 0.0;

            for (i, &kernel_val) in kernel.iter().enumerate().take(kernel_size) {
                let offset = i as isize - radius as isize;
                let nx = (x as isize + offset).clamp(0, width as isize - 1) as usize;
                sum += image[[y, nx]] * kernel_val;
                weight_sum += kernel_val;
            }

            temp[[y, x]] = sum / weight_sum;
        }
    }

    // Vertical pass
    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0;
            let mut weight_sum = 0.0;

            for (i, &kernel_val) in kernel.iter().enumerate().take(kernel_size) {
                let offset = i as isize - radius as isize;
                let ny = (y as isize + offset).clamp(0, height as isize - 1) as usize;
                sum += temp[[ny, x]] * kernel_val;
                weight_sum += kernel_val;
            }

            output[[y, x]] = sum / weight_sum;
        }
    }

    output
}

/// Connected component labeling using flood fill
fn label(binary: &Array2<bool>) -> Result<(Array2<u32>, usize)> {
    let (height, width) = binary.dim();
    let mut labels = Array2::zeros((height, width));
    let mut current_label = 0u32;

    for y in 0..height {
        for x in 0..width {
            if binary[[y, x]] && labels[[y, x]] == 0 {
                current_label += 1;

                // Flood fill using BFS
                let mut queue = VecDeque::new();
                queue.push_back((y, x));
                labels[[y, x]] = current_label;

                while let Some((cy, cx)) = queue.pop_front() {
                    // Check 8-connected neighbors
                    for dy in -1..=1 {
                        for dx in -1..=1 {
                            if dy == 0 && dx == 0 {
                                continue;
                            }

                            let ny = cy as isize + dy;
                            let nx = cx as isize + dx;

                            if ny >= 0 && ny < height as isize && nx >= 0 && nx < width as isize {
                                let ny = ny as usize;
                                let nx = nx as usize;

                                if binary[[ny, nx]] && labels[[ny, nx]] == 0 {
                                    labels[[ny, nx]] = current_label;
                                    queue.push_back((ny, nx));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Ok((labels, current_label as usize))
}

/// Preprocess image with Gaussian smoothing
fn preprocess(
    image: &Array2<f32>,
    mask: Option<&Array2<bool>>,
    sigma: f32,
    _mode: PreprocessMode,
) -> Result<(Array2<f32>, Array2<bool>)> {
    let (height, width) = image.dim();

    // Create or process mask
    let (masked_image, eroded_mask) = if let Some(mask) = mask {
        // Apply mask to image
        let mut masked = Array2::zeros(image.raw_dim());
        Zip::from(&mut masked)
            .and(image)
            .and(mask)
            .for_each(|m, &img, &mask| {
                if mask {
                    *m = img;
                }
            });

        // Erode mask
        let mut eroded = mask.clone();
        // Set borders to false
        for i in 0..width {
            eroded[[0, i]] = false;
            eroded[[height - 1, i]] = false;
        }
        for i in 0..height {
            eroded[[i, 0]] = false;
            eroded[[i, width - 1]] = false;
        }

        (masked, eroded)
    } else {
        // No mask, use full image and create border mask
        let mut eroded_mask = Array2::from_elem((height, width), true);
        // Set borders to false
        for i in 0..width {
            eroded_mask[[0, i]] = false;
            eroded_mask[[height - 1, i]] = false;
        }
        for i in 0..height {
            eroded_mask[[i, 0]] = false;
            eroded_mask[[i, width - 1]] = false;
        }

        (image.clone(), eroded_mask)
    };

    // Apply Gaussian filtering
    let smoothed = gaussian_filter(&masked_image, sigma);

    Ok((smoothed, eroded_mask))
}

/// Compute Sobel gradients
fn compute_gradients(image: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
    let (height, width) = image.dim();
    let mut gx = Array2::zeros((height, width));
    let mut gy = Array2::zeros((height, width));

    // Apply Sobel operators
    for y in 1..(height - 1) {
        for x in 1..(width - 1) {
            // Horizontal gradient (Sobel-X)
            gx[[y, x]] = -1.0 * image[[y - 1, x - 1]]
                + 1.0 * image[[y - 1, x + 1]]
                + -2.0 * image[[y, x - 1]]
                + 2.0 * image[[y, x + 1]]
                + -1.0 * image[[y + 1, x - 1]]
                + 1.0 * image[[y + 1, x + 1]];

            // Vertical gradient (Sobel-Y)
            gy[[y, x]] = -1.0 * image[[y - 1, x - 1]]
                + -2.0 * image[[y - 1, x]]
                + -1.0 * image[[y - 1, x + 1]]
                + 1.0 * image[[y + 1, x - 1]]
                + 2.0 * image[[y + 1, x]]
                + 1.0 * image[[y + 1, x + 1]];
        }
    }

    (gx, gy)
}

/// Non-maximum suppression using bilinear interpolation
fn nonmaximum_suppression(
    gx: &Array2<f32>,
    gy: &Array2<f32>,
    magnitude: &Array2<f32>,
    mask: &Array2<bool>,
    low_threshold: f32,
) -> Array2<f32> {
    let (height, width) = magnitude.dim();
    let mut output = Array2::zeros((height, width));

    for y in 1..(height - 1) {
        for x in 1..(width - 1) {
            if !mask[[y, x]] || magnitude[[y, x]] < low_threshold {
                continue;
            }

            let mag = magnitude[[y, x]];
            let dx = gx[[y, x]];
            let dy = gy[[y, x]];

            // Calculate gradient direction
            let angle = dy.atan2(dx);

            // Determine neighbors to check based on gradient direction
            let (n1, n2) = if (-FRAC_PI_4..=FRAC_PI_4).contains(&angle)
                || (3.0 * FRAC_PI_4..=PI).contains(&angle)
                || (-PI..=-3.0 * FRAC_PI_4).contains(&angle)
            {
                // Horizontal edge
                (magnitude[[y, x - 1]], magnitude[[y, x + 1]])
            } else if (FRAC_PI_4..=3.0 * FRAC_PI_4).contains(&angle) {
                // Vertical edge
                (magnitude[[y - 1, x]], magnitude[[y + 1, x]])
            } else if angle > 0.0 {
                // Diagonal edge (/)
                (magnitude[[y - 1, x - 1]], magnitude[[y + 1, x + 1]])
            } else {
                // Anti-diagonal edge (\)
                (magnitude[[y - 1, x + 1]], magnitude[[y + 1, x - 1]])
            };

            // Keep only if local maximum
            if mag >= n1 && mag >= n2 {
                output[[y, x]] = mag;
            }
        }
    }

    output
}

/// Apply Canny edge detection algorithm
///
/// Detect edges in an image using the Canny algorithm.
///
/// # Arguments
///
/// * `image` - Input grayscale image
/// * `sigma` - Standard deviation for Gaussian kernel (default: 1.0)
/// * `low_threshold` - Lower threshold for edge linking (default: None, auto-computed)
/// * `high_threshold` - Upper threshold for edge linking (default: None, auto-computed)  
/// * `mask` - Optional mask to limit edge detection to certain areas
/// * `use_quantiles` - If true, treat thresholds as quantiles (default: false)
/// * `mode` - Edge handling mode for Gaussian filter (default: Constant(0.0))
///
/// # Returns
///
/// * Result containing binary edge map
///
/// # Examples
///
/// ```rust,ignore
/// use scirs2_vision::feature::canny;
/// use image::DynamicImage;
///
/// let img = image::open("input.jpg").unwrap();
/// let edges = canny(&img, 1.0, None, None, None, false, PreprocessMode::Constant(0.0))?;
/// ```
pub fn canny(
    image: &DynamicImage,
    sigma: f32,
    low_threshold: Option<f32>,
    high_threshold: Option<f32>,
    mask: Option<&Array2<bool>>,
    use_quantiles: bool,
    mode: PreprocessMode,
) -> Result<GrayImage> {
    // Convert image to normalized array
    let img_array = image_to_array_normalized(image)?;

    // Set default thresholds
    let mut low_thresh = low_threshold.unwrap_or(0.1);
    let mut high_thresh = high_threshold.unwrap_or(0.2);

    // Validate thresholds
    if use_quantiles && (!(0.0..=1.0).contains(&low_thresh) || !(0.0..=1.0).contains(&high_thresh))
    {
        return Err(VisionError::InvalidParameter(
            "Quantile thresholds must be between 0 and 1".to_string(),
        ));
    }

    if high_thresh < low_thresh {
        return Err(VisionError::InvalidParameter(
            "low_threshold should be lower than high_threshold".to_string(),
        ));
    }

    // Preprocess: smooth and prepare mask
    let (smoothed, eroded_mask) = preprocess(&img_array, mask, sigma, mode)?;

    // Compute gradients
    let (gx, gy) = compute_gradients(&smoothed);

    // Compute magnitude
    let mut magnitude = Array2::zeros(gx.raw_dim());
    Zip::from(&mut magnitude)
        .and(&gx)
        .and(&gy)
        .for_each(|m, &x, &y| {
            *m = (x * x + y * y).sqrt();
        });

    // Convert quantile thresholds to absolute if needed
    if use_quantiles {
        let mut mag_values: Vec<f32> = magnitude.iter().cloned().collect();
        mag_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let low_idx = (mag_values.len() as f32 * low_thresh) as usize;
        let high_idx = (mag_values.len() as f32 * high_thresh) as usize;

        low_thresh = mag_values[low_idx.min(mag_values.len() - 1)];
        high_thresh = mag_values[high_idx.min(mag_values.len() - 1)];
    }

    // Non-maximum suppression
    let suppressed = nonmaximum_suppression(&gx, &gy, &magnitude, &eroded_mask, low_thresh);

    // Double thresholding and edge tracking
    let low_mask = suppressed.mapv(|x| x > 0.0);
    let high_mask = suppressed.mapv(|x| x >= high_thresh);

    // Label connected components in low_mask
    let (labels, num_labels) = if let Ok(result) = label(&low_mask) {
        result
    } else {
        return array_to_binary_image(&Array2::from_elem(low_mask.raw_dim(), false));
    };

    // Find labels that contain high threshold pixels
    let mut good_labels = vec![false; num_labels + 1];
    Zip::from(&labels)
        .and(&high_mask)
        .for_each(|&label, &is_high| {
            if is_high && label > 0 {
                good_labels[label as usize] = true;
            }
        });

    // Create final output
    let output = labels.mapv(|label| {
        if label > 0 {
            good_labels[label as usize]
        } else {
            false
        }
    });

    array_to_binary_image(&output)
}

/// Simple convenience function for Canny edge detection with default parameters
///
/// # Arguments
///
/// * `image` - Input image
/// * `sigma` - Standard deviation for Gaussian smoothing
///
/// # Returns
///
/// * Result containing binary edge map
pub fn canny_simple(image: &DynamicImage, sigma: f32) -> Result<GrayImage> {
    canny(
        image,
        sigma,
        None,
        None,
        None,
        false,
        PreprocessMode::Constant(0.0),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Luma;

    #[test]
    fn test_canny_on_simple_edge() {
        // Create a simple test image with a vertical edge
        let mut img = GrayImage::new(10, 10);
        for y in 0..10 {
            for x in 0..10 {
                let value = if x < 5 { 0 } else { 255 };
                img.put_pixel(x, y, Luma([value]));
            }
        }

        let dynamic_img = DynamicImage::ImageLuma8(img);
        let result = canny_simple(&dynamic_img, 1.0);

        assert!(result.is_ok());
        let edges = result.unwrap();

        // Should detect edge around x=5
        let mut has_edge = false;
        for y in 1..9 {
            if edges.get_pixel(5, y)[0] > 0 {
                has_edge = true;
                break;
            }
        }
        assert!(has_edge, "Should detect vertical edge");
    }

    #[test]
    fn test_canny_with_custom_thresholds() {
        let img = GrayImage::new(10, 10);
        let dynamic_img = DynamicImage::ImageLuma8(img);

        let result = canny(
            &dynamic_img,
            1.0,
            Some(0.05),
            Some(0.15),
            None,
            false,
            PreprocessMode::Constant(0.0),
        );

        assert!(result.is_ok());
    }

    #[test]
    fn test_invalid_thresholds() {
        let img = GrayImage::new(10, 10);
        let dynamic_img = DynamicImage::ImageLuma8(img);

        // High threshold lower than low threshold
        let result = canny(
            &dynamic_img,
            1.0,
            Some(0.2),
            Some(0.1),
            None,
            false,
            PreprocessMode::Constant(0.0),
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_quantile_thresholds() {
        let img = GrayImage::new(10, 10);
        let dynamic_img = DynamicImage::ImageLuma8(img);

        let result = canny(
            &dynamic_img,
            1.0,
            Some(0.1),
            Some(0.9),
            None,
            true,
            PreprocessMode::Constant(0.0),
        );

        assert!(result.is_ok());
    }
}
